"""
Pitch detector: estimate fundamental frequency of audio segments.

Primary: PYIN (Probabilistic YIN) — state-of-the-art monophonic pitch detection.
Fallback: YIN — simpler, more robust for short/noisy notes.
"""

from typing import Optional

import numpy as np
import librosa

# Valid pitch range: C2 (65.4 Hz) to C7 (2093 Hz) — covers most vocal/instrumental melodies
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")


def detect_pitch_pyin(segment: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect pitch frequencies using the PYIN algorithm.

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

    More robust than PYIN for very short notes or notes with
    weak fundamental frequencies.

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

    Uses median of voiced (non-NaN) frames. Requires at least 10%
    of frames to be voiced (relaxed from 30% to handle complex audio).

    Args:
        pitch_values: Array of frequency values from PYIN.

    Returns:
        Median frequency in Hz, or None if overwhelmingly unvoiced.
    """
    voiced = pitch_values[~np.isnan(pitch_values)]

    if len(voiced) == 0:
        return None

    # Relaxed: only require 10% voiced (was 30%)
    if len(voiced) / len(pitch_values) < 0.10:
        return None

    return float(np.median(voiced))


def is_silence(segment: np.ndarray, global_rms: float = None) -> bool:
    """
    Check if a segment is essentially silence.

    Uses an adaptive threshold: 5% of the global RMS, or a small
    absolute threshold if global RMS is not provided.

    Args:
        segment: Audio samples.
        global_rms: Overall RMS of the full audio (for adaptive threshold).

    Returns:
        True if the segment is silence.
    """
    if len(segment) == 0:
        return True

    rms = np.sqrt(np.mean(segment ** 2))

    if global_rms is not None and global_rms > 0:
        threshold = max(0.005, global_rms * 0.05)
    else:
        threshold = 0.01

    return rms < threshold
