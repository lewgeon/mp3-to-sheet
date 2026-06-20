"""
Onset detector: detect note start times and segment the audio into individual notes.
"""

import numpy as np
import librosa

# Minimum gap between onsets in seconds
MIN_ONSET_GAP = 0.08


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect note onset frames in the audio.

    Uses a multi-strategy approach:
    1. Spectral-based onset detection (good for real instruments/voice)
    2. Falls back to energy-based detection if too few onsets found

    Args:
        y: Audio samples.
        sr: Sample rate.

    Returns:
        Array of onset frame indices.
    """
    # --- Strategy 1: Spectral flux based ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Try multiple delta thresholds and pick the one that gives a reasonable count
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.15,
        wait=int(sr * MIN_ONSET_GAP),
    )

    if len(onset_frames) >= 2:
        return onset_frames

    # --- Strategy 2: Lower delta threshold ---
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True,
        pre_max=1,
        post_max=1,
        pre_avg=1,
        post_avg=1,
        delta=0.05,
        wait=int(sr * MIN_ONSET_GAP),
    )

    if len(onset_frames) >= 2:
        return onset_frames

    # --- Strategy 3: Energy-based detection ---
    onset_frames = energy_based_onsets(y, sr)

    return onset_frames


def energy_based_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect onsets using RMS energy — good for synthetic/simple audio
    where notes are separated by silence.

    Uses the standard librosa hop_length of 512 for compatibility
    with frames_to_time().

    Args:
        y: Audio samples.
        sr: Sample rate.

    Returns:
        Array of onset frame indices (hop_length=512).
    """
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    # Adaptive threshold: 30% of max RMS
    rms_max = np.max(rms)
    rms_min = np.min(rms)
    threshold = rms_min + 0.15 * (rms_max - rms_min)

    min_gap_frames = int(MIN_ONSET_GAP * sr / hop_length)

    onset_frames = []
    was_silent = True

    for i in range(len(rms)):
        if rms[i] > threshold and was_silent:
            if not onset_frames or (i - onset_frames[-1]) >= min_gap_frames:
                onset_frames.append(i)
            was_silent = False
        elif rms[i] <= threshold:
            was_silent = True

    return np.array(onset_frames)


def frames_to_time(onset_frames: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Convert onset frames to time in seconds.

    Handles both librosa's default hop_length (512) and energy-based (256).

    Args:
        onset_frames: Frame indices.
        sr: Sample rate.
        hop_length: Hop length used (default librosa: 512, energy-based: 256).

    Returns:
        Onset times in seconds.
    """
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def get_note_segments(
    y: np.ndarray, sr: int, onset_times: np.ndarray
) -> list[dict]:
    """
    Split audio into individual note segments based on onset times.

    Each note runs from its onset time to the next note's onset time
    (or end of audio for the last note).

    Args:
        y: Audio samples.
        sr: Sample rate.
        onset_times: Onset times in seconds.

    Returns:
        List of dicts with keys: start_time, end_time, duration, samples.
    """
    segments = []

    for i, start_t in enumerate(onset_times):
        # Determine end time: next onset or end of audio
        if i + 1 < len(onset_times):
            end_t = onset_times[i + 1]
        else:
            end_t = len(y) / sr

        # Extract samples for this segment
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        segment_samples = y[start_sample:end_sample]

        segments.append({
            "start_time": start_t,
            "end_time": end_t,
            "duration": end_t - start_t,
            "samples": segment_samples,
            "sample_rate": sr,
        })

    return segments


def detect_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """
    Detect beat positions and estimate tempo.

    Args:
        y: Audio samples.
        sr: Sample rate.

    Returns:
        (beat_times, tempo) — beat times in seconds and estimated BPM.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, float(tempo)
