#!/usr/bin/env python3
"""
MP3 to Sheet Music Converter

Convert an MP3 audio file into sheet music notation (MusicXML).
Supports both monophonic and polyphonic audio with optional melody extraction.

Usage:
    python main.py input.mp3
    python main.py input.mp3 -o melody
    python main.py input.mp3 --extract-melody --sensitivity high
    python main.py input.mp3 --bpm 120 --format musicxml pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.audio_processor import load_audio, preprocess
from src.onset_detector import detect_onsets, frames_to_time, get_note_segments
from src.pitch_detector import (
    detect_pitch_pyin,
    detect_pitch_yin,
    get_stable_frequency,
    is_silence,
)
from src.transcriber import (
    freq_to_note,
    estimate_bpm,
    quantize_duration,
    smooth_pitch_errors,
)
from src.sheet_generator import create_score, export_score


def transcribe_audio(
    mp3_path: str,
    manual_bpm: Optional[float] = None,
    sensitivity: str = "medium",
    extract_melody: bool = False,
    verbose: bool = True,
) -> tuple[list, float]:
    """
    Full transcription pipeline: MP3 → note list.

    Args:
        mp3_path: Path to the input audio file.
        manual_bpm: Optional manual BPM override.
        sensitivity: Onset detection sensitivity ('low'/'medium'/'high').
        extract_melody: Enable HPSS melody extraction for polyphonic audio.
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
    global_rms = float(np.sqrt(np.mean(y ** 2)))
    if verbose:
        print(f"  Duration: {total_duration:.1f}s, SR: {sr}Hz, RMS: {global_rms:.3f}")

    # Step 2.5: Optional melody extraction for polyphonic audio
    if extract_melody:
        if verbose:
            print("Extracting predominant melody (HPSS + spectral peaks)...")
        from src.melody_extractor import preprocess_for_melody

        y = preprocess_for_melody(y, sr, use_hpss=True)
        # Re-normalize after extraction
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        if verbose:
            print(f"  Melody extracted, new duration: {len(y)/sr:.1f}s")

    # Step 3: Detect onsets
    if verbose:
        print(f"Detecting note onsets (sensitivity: {sensitivity})...")
    onset_frames = detect_onsets(y, sr, sensitivity=sensitivity)

    if len(onset_frames) == 0:
        print("[Error] No notes detected in the audio.")
        print("  Try: --sensitivity high  or  --extract-melody")
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
        print("Detecting pitch for each note (PYIN + YIN fallback)...")
    raw_notes = []

    num_pyin_ok = 0
    num_yin_fallback = 0
    num_silence = 0
    num_unvoiced = 0

    for i, seg in enumerate(segments):
        # Check for silence
        if is_silence(seg["samples"], global_rms=global_rms):
            raw_notes.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "duration": seg["duration"],
                "midi": None,
                "full_name": "rest",
                "is_rest": True,
                "frequency": None,
            })
            num_silence += 1
            if verbose:
                print(f"  Note {i + 1:3d}: rest (silence, {seg['duration']:.2f}s)")
            continue

        # Step 5a: Primary — PYIN
        f0_contour = detect_pitch_pyin(seg["samples"], seg["sample_rate"])
        freq = get_stable_frequency(f0_contour)

        # Step 5b: Fallback — YIN (if PYIN fails)
        if freq is None:
            freq = detect_pitch_yin(seg["samples"], seg["sample_rate"])
            if freq is not None:
                num_yin_fallback += 1

        # Step 5c: Still no pitch → rest
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
            num_unvoiced += 1
            if verbose:
                print(f"  Note {i + 1:3d}: rest (unvoiced, {seg['duration']:.2f}s)")
            continue

        num_pyin_ok += 1
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

    if verbose:
        print(f"  Summary: {num_pyin_ok} PYIN, {num_yin_fallback} YIN-fallback, "
              f"{num_silence} silence, {num_unvoiced} unvoiced")

    # Step 6: Smooth pitch errors
    if verbose:
        print("Smoothing note errors...")
    smoothed = smooth_pitch_errors(raw_notes)

    actual_notes = [n for n in smoothed if not n.get("is_rest")]
    if verbose:
        print(f"  Final notes: {len(actual_notes)} pitched + "
              f"{len(smoothed) - len(actual_notes)} rests")

    if len(actual_notes) == 0:
        print("[Error] No pitched notes found. Try --extract-melody for polyphonic audio.")
        sys.exit(1)

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
  python main.py song.mp3 --extract-melody --sensitivity high
  python main.py song.mp3 --bpm 120 --format musicxml pdf
  python main.py song.mp3 --title "My Melody" --time-signature 3/4
        """,
    )

    parser.add_argument("input", help="Path to input audio file")
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
    parser.add_argument(
        "--sensitivity",
        default="medium",
        choices=["low", "medium", "high"],
        help="Onset detection sensitivity (default: medium). "
             "Use 'high' for complex/polyphonic audio, 'low' for clean solo recordings.",
    )
    parser.add_argument(
        "--extract-melody",
        action="store_true",
        help="Enable melody extraction for polyphonic audio "
             "(HPSS + spectral peak tracking). Recommended for BGM/multi-instrument tracks.",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"[Error] File not found: {args.input}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_prefix = args.output
    else:
        input_path = Path(args.input)
        output_prefix = str(input_path.parent / input_path.stem)

    title = args.title or Path(args.input).stem
    verbose = not args.quiet

    # ------------------- Transcription -------------------
    notes_data, bpm = transcribe_audio(
        args.input,
        manual_bpm=args.bpm,
        sensitivity=args.sensitivity,
        extract_melody=args.extract_melody,
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
        print(f"\nDone! Output files:")
        for f in written_files:
            print(f"  {f}")
        if len(written_files) == 1 and written_files[0].endswith(".musicxml"):
            print("\n  To view as sheet music, open the .musicxml in MuseScore:")
            print("    https://musescore.org/")


if __name__ == "__main__":
    main()
