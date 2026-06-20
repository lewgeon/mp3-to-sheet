"""
Transcriber: convert frequencies to musical notes and quantize durations.
"""

from typing import Optional

import math
import numpy as np

# MIDI note 69 = A4 = 440 Hz
A4_MIDI = 69
A4_FREQ = 440.0

# Note names (sharps)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard duration ratios (relative to a quarter note = 1.0)
DURATION_BINS = [
    (0.125, "32nd"),
    (0.25, "16th"),
    (0.375, "dotted 16th"),
    (0.5, "eighth"),
    (0.75, "dotted eighth"),
    (1.0, "quarter"),
    (1.5, "dotted quarter"),
    (2.0, "half"),
    (3.0, "dotted half"),
    (4.0, "whole"),
]


def freq_to_midi(freq: float) -> int:
    """
    Convert a frequency in Hz to the nearest MIDI note number.

    MIDI note 69 = A4 = 440 Hz.
    """
    if freq <= 0:
        return 0
    return int(round(A4_MIDI + 12 * math.log2(freq / A4_FREQ)))


def midi_to_note_name(midi_note: int) -> tuple[str, int]:
    """
    Convert a MIDI note number to pitch class name and octave.

    Args:
        midi_note: MIDI note number (0–127).

    Returns:
        (note_name, octave) — e.g. ("C#", 4) for C#4.
    """
    octave = (midi_note // 12) - 1
    pitch_class = midi_note % 12
    return NOTE_NAMES[pitch_class], octave


def freq_to_note(freq: float) -> Optional[dict]:
    """
    Convert a frequency to a full note description.

    Args:
        freq: Frequency in Hz.

    Returns:
        Dict with keys: midi, name, octave, full_name, cents_offset
        or None if freq is invalid.
    """
    if freq is None or freq <= 0 or math.isnan(freq):
        return None

    midi = freq_to_midi(freq)
    name, octave = midi_to_note_name(midi)

    # Exact frequency of the matched MIDI note
    exact_freq = A4_FREQ * (2 ** ((midi - A4_MIDI) / 12))

    # Cents deviation from exact pitch
    cents = 1200 * math.log2(freq / exact_freq) if exact_freq > 0 else 0

    return {
        "midi": midi,
        "name": name,
        "octave": octave,
        "full_name": f"{name}{octave}",
        "frequency": freq,
        "exact_frequency": exact_freq,
        "cents_offset": round(cents, 1),
    }


def estimate_bpm(note_durations: list[float]) -> float:
    """
    Estimate BPM from note durations.

    Uses median inter-onset interval and assumes it corresponds to
    the most common note value (quarter note).

    Args:
        note_durations: List of note durations in seconds.

    Returns:
        Estimated BPM.
    """
    if not note_durations:
        return 120.0

    # Filter out very short (< 50ms) and very long (> 5s) durations
    filtered = [d for d in note_durations if 0.05 < d < 5.0]
    if not filtered:
        return 120.0

    # Median duration — assume this is a quarter note
    median_dur = float(np.median(filtered))
    if median_dur <= 0:
        return 120.0

    # BPM = 60 / quarter_note_duration
    bpm = 60.0 / median_dur

    # Clamp to reasonable range (40–250 BPM)
    return max(40.0, min(250.0, bpm))


def quantize_duration(duration_sec: float, bpm: float) -> float:
    """
    Quantize a duration in seconds to the nearest standard music duration.

    Args:
        duration_sec: Note duration in seconds.
        bpm: Tempo in beats per minute.

    Returns:
        Duration in quarter-length units (music21 format).
    """
    # Convert seconds to quarter notes
    # At `bpm` BPM, one quarter note = 60/bpm seconds
    quarter_duration_sec = 60.0 / bpm
    quarters = duration_sec / quarter_duration_sec

    # Find nearest standard duration bin
    ratios = [b[0] for b in DURATION_BINS]
    nearest_ratio = min(ratios, key=lambda r: abs(r - quarters))

    return nearest_ratio


def quantize_duration_name(duration_sec: float, bpm: float) -> str:
    """
    Get the human-readable name of the quantized duration.

    Args:
        duration_sec: Note duration in seconds.
        bpm: Tempo in BPM.

    Returns:
        Duration name e.g. "quarter", "eighth".
    """
    quarter_duration_sec = 60.0 / bpm
    quarters = duration_sec / quarter_duration_sec

    ratios = [b[0] for b in DURATION_BINS]
    idx = min(range(len(ratios)), key=lambda i: abs(ratios[i] - quarters))

    return DURATION_BINS[idx][1]


def smooth_pitch_errors(
    notes: list[dict], min_note_duration: float = 0.08
) -> list[dict]:
    """
    Post-process detected notes:
    1. Merge very short notes into neighbors (likely detection errors).
    2. Merge consecutive notes with the same pitch (likely decay tails).
    3. Remove isolated single-frame detections.

    Args:
        notes: List of note dicts from detection.
        min_note_duration: Minimum note duration in seconds.

    Returns:
        Smoothed list of notes.
    """
    if not notes:
        return notes

    # Pass 1: merge short notes
    cleaned = []
    for note in notes:
        if note["duration"] < min_note_duration and cleaned:
            cleaned[-1]["duration"] += note["duration"]
            cleaned[-1]["end_time"] = note["end_time"]
            continue
        cleaned.append(note)

    if not cleaned:
        return cleaned

    # Pass 2: merge consecutive notes with same pitch
    # (common with decay tails or re-triggered onsets on the same note)
    merged = [cleaned[0]]
    for note in cleaned[1:]:
        prev = merged[-1]
        # Same MIDI pitch and the note isn't a rest
        if (
            prev.get("midi") is not None
            and note.get("midi") is not None
            and prev["midi"] == note["midi"]
            and not prev.get("is_rest")
            and not note.get("is_rest")
        ):
            prev["duration"] += note["duration"]
            prev["end_time"] = note["end_time"]
            # Recompute frequency as weighted average
            if prev.get("frequency") and note.get("frequency"):
                d1 = prev["duration"] - note["duration"]
                d2 = note["duration"]
                total = d1 + d2
                if total > 0:
                    prev["frequency"] = (prev["frequency"] * d1 + note["frequency"] * d2) / total
            continue
        merged.append(note)

    return merged
