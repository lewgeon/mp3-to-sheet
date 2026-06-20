"""
Sheet generator: build a music21 Score from transcribed notes and export to
various formats (MusicXML, PDF, PNG).
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, List

# Suppress music21 initialization warnings — especially from MuseScore
warnings.filterwarnings("ignore", module="music21")

from music21 import (
    stream,
    note,
    meter,
    key,
    tempo,
    metadata,
    environment,
    duration,
)


def _auto_key_signature(notes_list: list) -> key.Key:
    """
    Auto-detect the best key signature for a sequence of notes.

    Uses music21's built-in key analysis.

    Args:
        notes_list: List of (pitch_name, octave) tuples or note dicts.

    Returns:
        A music21 Key object.
    """
    if not notes_list:
        return key.Key("C")

    # Build a temporary stream for analysis
    s = stream.Stream()
    for n in notes_list:
        if isinstance(n, dict):
            pitch_str = n.get("full_name", "C4")
        else:
            pitch_str = f"{n[0]}{n[1]}"
        try:
            n_obj = note.Note(pitch_str)
            s.append(n_obj)
        except Exception:
            continue

    if len(s.notes) == 0:
        return key.Key("C")

    try:
        # Analyze the stream for key
        k = s.analyze("key")
        return k
    except Exception:
        return key.Key("C")


def _get_accidental_count_for_key(k: key.Key) -> int:
    """Count the number of accidentals in a key signature."""
    try:
        return k.sharps if k.sharps > 0 else abs(k.flats)
    except Exception:
        return 0


def create_score(
    notes_data: list[dict],
    bpm: float = 120,
    title: str = "Transcribed Melody",
    time_signature: str = "4/4",
) -> stream.Score:
    """
    Build a music21 Score from transcribed note data.

    Args:
        notes_data: List of note dicts. Each dict should have:
            - midi: MIDI note number, or None for rest
            - duration_ql: Duration in quarter-length units
            - full_name: Note name with octave (e.g. "C4")
        bpm: Tempo in beats per minute.
        title: Title of the piece.
        time_signature: Time signature string (e.g. "4/4").

    Returns:
        A music21 Score object.
    """
    score = stream.Score()
    part = stream.Part()

    # Metadata
    md = metadata.Metadata()
    md.title = title
    score.insert(0, md)

    # Time signature
    ts = meter.TimeSignature(time_signature)
    part.insert(0, ts)

    # Key signature — analyze from notes
    try:
        k = _auto_key_signature(notes_data)
        part.insert(0, k)
    except Exception:
        pass

    # Tempo
    mm = tempo.MetronomeMark(number=int(round(bpm)))
    part.insert(0, mm)

    # Add notes and rests
    for nd in notes_data:
        if nd.get("is_rest") or nd.get("midi") is None:
            # Rests
            r = note.Rest()
            r.duration = duration.Duration(nd["duration_ql"])
            part.append(r)
        else:
            # Pitched note
            n = note.Note()
            n.pitch.midi = nd["midi"]
            n.duration = duration.Duration(nd["duration_ql"])

            # Set volume based on confidence if available
            if "volume" in nd:
                n.volume.velocity = int(nd["volume"] * 127)

            part.append(n)

    # Make sure the part fills complete measures
    try:
        part.makeMeasures(inPlace=True)
    except Exception:
        pass

    score.insert(0, part)
    return score


def export_score(
    score_obj: stream.Score,
    output_path: str,
    formats: Optional[List[str]] = None,
) -> List[str]:
    """
    Export a music21 Score to file(s).

    Args:
        score_obj: The music21 Score to export.
        output_path: Output file path without extension.
        formats: List of formats to export. Supports "musicxml", "pdf", "png", "midi".
                 Default: ["musicxml"].

    Returns:
        List of successfully written file paths.
    """
    if formats is None:
        formats = ["musicxml"]

    written_files = []

    for fmt in formats:
        ext_map = {
            "musicxml": ".musicxml",
            "xml": ".musicxml",
            "pdf": ".pdf",
            "png": ".png",
            "midi": ".mid",
            "mid": ".mid",
        }

        ext = ext_map.get(fmt, f".{fmt}")
        file_path = output_path + ext

        try:
            if fmt in ("pdf", "png"):
                # For image rendering, music21 needs MuseScore or LilyPond
                score_obj.write(fmt, file_path)
            else:
                score_obj.write(fmt, file_path)
            written_files.append(file_path)
        except Exception as e:
            print(f"  [Warning] Could not export {fmt}: {e}")

            # Fallback: if PDF/PNG fails, suggest MusicXML
            if fmt in ("pdf", "png") and "musicxml" not in formats:
                try:
                    xml_path = output_path + ".musicxml"
                    score_obj.write("musicxml", xml_path)
                    written_files.append(xml_path)
                    print(f"  [Info] Exported MusicXML instead: {xml_path}")
                    print(f"  [Info] To render as image, open the .musicxml in MuseScore")
                except Exception:
                    pass

    return written_files
