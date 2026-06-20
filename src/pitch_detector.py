"""
Pitch detector: estimate fundamental frequency of audio segments using PYIN.
"""

from typing import Optional

import numpy as np
import librosa

# Valid pitch range: C2 (65.4 Hz) to C7 (2093 Hz) — covers most vocal/instrumental melodies
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")

# Frequencies below this threshold are considered silence
SILENCE_RMS_THRESHOLD = 0.01


def detect_pitch_pyin(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect pitch frequencies in an audio segment using the PYIN algorithm.

    PYIN (Probabilistic YIN) is a state-of-the-art monophonic pitch
    detection algorithm that handles vibrato and pitch variations well.

    Args:
        segment: Audio samples for one note segment.
        sr: Sample rate.

    Returns:
        Array of frequency values in Hz (NaN where pitch is unvoiced).
    """
    if len(segment) < sr * 0.02:  # Shorter than 20ms — too short
        return np.array([])

    f0, voiced_flag, _ = librosa.pyin(
        segment,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        fill_na=np.nan,
    )

    return f0


def detect_pitch_yin(segment: np.ndarray, sr: int) -> Optional[float]:
    """
    Fallback: detect a single pitch using YIN algorithm.
    Faster but less accurate for vibrato; good for short steady notes.

    Args:
        segment: Audio samples.
        sr: Sample rate.

    Returns:
        Estimated fundamental frequency in Hz, or None if unvoiced.
    """
    if len(segment) < sr * 0.02:
        return None

    try:
        f0 = librosa.yin(
            segment,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
        )
        # Take median of voiced frames (ignore NaN)
        valid = f0[~np.isnan(f0)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))
    except Exception:
        return None


def get_stable_frequency(pitch_values: np.ndarray) -> Optional[float]:
    """
    Extract a stable representative frequency from a pitch contour.

    Uses the median of voiced (non-NaN) frames to reject outliers.

    Args:
        pitch_values: Array of frequency values from PYIN (contains NaN for unvoiced).

    Returns:
        Median frequency in Hz, or None if mostly unvoiced.
    """
    voiced = pitch_values[~np.isnan(pitch_values)]

    if len(voiced) == 0:
        return None

    # Require at least 30% of frames to be voiced
    if len(voiced) / len(pitch_values) < 0.3:
        return None

    return float(np.median(voiced))


def is_silence(segment: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """
    Check if a segment is essentially silence (rest).

    Args:
        segment: Audio samples.
        threshold: RMS threshold below which the segment is considered silence.

    Returns:
        True if the segment is silence.
    """
    if len(segment) == 0:
        return True
    rms = np.sqrt(np.mean(segment**2))
    return rms < threshold
