#!/usr/bin/env python3
"""
MP3 to Sheet Music Converter

Convert an MP3 audio file (monophonic melody) into sheet music notation.
Outputs MusicXML (always) and optionally PDF/PNG if MuseScore is installed.

Usage:
    python main.py input.mp3
    python main.py input.mp3 -o melody
    python main.py input.mp3 --bpm 120 --format musicxml pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.audio_processor import load_audio, preprocess
from src.onset_detector import detect_onsets, frames_to_time, get_note_segments
from src.pitch_detector import detect_pitch_pyin, get_stable_frequency, is_silence
from src.transcriber import (
    freq_to_note,
    estimate_bpm,
    quantize_duration,
    smooth_pitch_errors,
)
from src.sheet_generator import create_score, export_score


def transcribe_audio(
    mp3_path: str,
    manual_bpm: float | None = None,
    verbose: bool = True,
) -> tuple[list[dict], float]:
    """
    Full transcription pipeline: MP3 → note list.

    Args:
        mp3_path: Path to the input MP3 file.
        manual_bpm: Optional manual BPM override.
        verbose: Print progress information.

    Returns:
        (notes_data, detected_bpm) — notes ready for sheet generation.
    """
    # Step 1: Load audio
    if verbose:
        print(f"Loading: {mp3_path}")
    y, sr = load_audio(mp3_path)

    # Step 2: Preprocess
    if verbose:
        print("Preprocessing audio (trim silence, normalize)...")
    y = preprocess(y, sr)

    total_duration = len(y) / sr
    if verbose:
        print(f"  Duration: {total_duration:.1f}s, Sample rate: {sr}Hz")

    # Step 3: Detect onsets
    if verbose:
        print("Detecting note onsets...")
    onset_frames = detect_onsets(y, sr)

    if len(onset_frames) == 0:
        print("[Error] No notes detected in the audio.")
        sys.exit(1)

    onset_times = frames_to_time(onset_frames, sr=sr)
    if verbose:
        print(f"  Detected {len(onset_times)} onset(s)")

    # Step 4: Segment audio by note
    if verbose:
        print("Segmenting audio into notes...")
    segments = get_note_segments(y, sr, onset_times)

    # Step 5: Detect pitch for each segment
    if verbose:
        print("Detecting pitch for each note...")
    raw_notes = []

    for i, seg in enumerate(segments):
        # Check for silence / rest
        if is_silence(seg["samples"]):
            raw_notes.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["duration"],
                "midi": None,
                "full_name": "rest",
                "is_rest": True,
                "frequency": None,
            })
            if verbose:
                print(f"  Note {i + 1:3d}: rest ({seg['duration']:.2f}s)")
            continue

        # Detect pitch using PYIN
        f0_contour = detect_pitch_pyin(seg["samples"], seg["sample_rate"])
        freq = get_stable_frequency(f0_contour)

        if freq is None:
            raw_notes.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["duration"],
                "midi": None,
                "full_name": "rest",
                "is_rest": True,
                "frequency": None,
            })
            if verbose:
                print(f"  Note {i + 1:3d}: rest (unvoiced, {seg['duration']:.2f}s)")
            continue

        note_info = freq_to_note(freq)

        raw_notes.append({
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "duration": seg["duration"],
            "midi": note_info["midi"],
            "full_name": note_info["full_name"],
            "name": note_info["name"],
            "octave": note_info["octave"],
            "frequency": freq,
            "cents_offset": note_info["cents_offset"],
            "is_rest": False,
        })

        if verbose:
            cents_str = f"({note_info['cents_offset']:+.1f}¢)" if note_info['cents_offset'] else ""
            print(
                f"  Note {i + 1:3d}: {note_info['full_name']:4s} "
                f"freq={freq:7.1f}Hz, dur={seg['duration']:.2f}s {cents_str}"
            )

    # Step 6: Smooth pitch errors
    if verbose:
        print("Smoothing note errors...")
    smoothed = smooth_pitch_errors(raw_notes)

    # Step 7: Estimate BPM
    if manual_bpm:
        bpm = manual_bpm
        if verbose:
            print(f"Using manual BPM: {bpm}")
    else:
        durations = [n["duration"] for n in smoothed]
        bpm = estimate_bpm(durations)
        if verbose:
            print(f"Estimated BPM: {bpm:.0f}")

    # Step 8: Quantize durations
    for n in smoothed:
        n["duration_ql"] = quantize_duration(n["duration"], bpm)

    return smoothed, bpm


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP3 audio to sheet music notation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py song.mp3
  python main.py song.mp3 -o my_song
  python main.py song.mp3 --bpm 120 --format musicxml pdf
  python main.py song.mp3 --title "My Melody" --time-signature 3/4
        """,
    )

    parser.add_argument("input", help="Path to input .mp3 file")
    parser.add_argument("-o", "--output", help="Output file path prefix (without extension)")
    parser.add_argument("--bpm", type=float, help="Manually set BPM (skip auto-detection)")
    parser.add_argument(
        "--format",
        nargs="+",
        default=["musicxml"],
        choices=["musicxml", "pdf", "png", "midi"],
        help="Output formats (default: musicxml). PDF/PNG require MuseScore.",
    )
    parser.add_argument("--title", default=None, help="Title for the sheet music")
    parser.add_argument(
        "--time-signature", default="4/4", help='Time signature (default: "4/4")'
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"[Error] File not found: {args.input}")
        sys.exit(1)

    if not args.input.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
        print("[Warning] Unrecognized audio format. Attempting anyway...")

    # Determine output path
    if args.output:
        output_prefix = args.output
    else:
        input_path = Path(args.input)
        output_prefix = str(input_path.parent / input_path.stem)

    # Default title
    title = args.title or Path(args.input).stem

    verbose = not args.quiet

    # ------------------- Transcription -------------------
    notes_data, bpm = transcribe_audio(
        args.input,
        manual_bpm=args.bpm,
        verbose=verbose,
    )

    # ------------------- Sheet generation -------------------
    if verbose:
        print(f"\nGenerating sheet music...")

    score = create_score(
        notes_data,
        bpm=bpm,
        title=title,
        time_signature=args.time_signature,
    )

    # ------------------- Export -------------------
    if verbose:
        print(f"Exporting to: {args.format}")

    written_files = export_score(score, output_prefix, formats=args.format)

    if verbose:
        print(f"\n✓ Done! Output files:")
        for f in written_files:
            print(f"  {f}")

        if "pdf" in args.format or "png" in args.format:
            musicxml_path = output_prefix + ".musicxml"
            if musicxml_path in written_files:
                print("\n  To render as PDF/PNG, install MuseScore:")
                print("    https://musescore.org/")
                print("  Then open the .musicxml file in MuseScore to export.")


if __name__ == "__main__":
    main()
