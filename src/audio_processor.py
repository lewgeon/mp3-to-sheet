"""
Audio processor: load MP3 files, convert to mono, resample, and preprocess.
"""

import librosa
import numpy as np

# Target sample rate for pitch detection (PYIN works well at 22050 Hz)
TARGET_SR = 22050


def load_audio(file_path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file (MP3, WAV, etc.) and convert to mono.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sample rate in Hz.

    Returns:
        (audio_samples, sample_rate) — mono audio at target_sr.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y, sr


def preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply preprocessing: trim leading/trailing silence and normalize amplitude.

    Args:
        y: Audio samples.
        sr: Sample rate.

    Returns:
        Preprocessed audio.
    """
    # Trim leading and trailing silence (conservative: 40dB to preserve note gaps)
    y_trimmed, _ = librosa.effects.trim(y, top_db=40)

    # Normalize to peak amplitude of 1.0
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_normalized = y_trimmed / peak
    else:
        y_normalized = y_trimmed

    return y_normalized


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return audio duration in seconds."""
    return len(y) / sr
