"""
Onset detector: detect note start times and segment audio into individual notes.

Uses a parallel multi-strategy approach:
1. Superflux (multi-band) — best for polyphonic / complex music
2. Spectral flux — good for monophonic melodies
3. Energy-based RMS — reliable for clean note/silence separation
4. CQT energy — detects onsets in constant-Q spectrum (musically relevant bands)

All strategies run and the one with the most onsets is selected.
If all fail, falls back to uniform grid segmentation.
"""

import numpy as np
import librosa

# Minimum gap between onsets in seconds
MIN_ONSET_GAP = 0.06


def detect_onsets(y: np.ndarray, sr: int, sensitivity: str = "medium") -> np.ndarray:
    """
    Detect note onset frames using multiple parallel strategies.

    All strategies run and the result with the most onsets is selected.

    Args:
        y: Audio samples (mono).
        sr: Sample rate.
        sensitivity: 'low', 'medium', or 'high' — controls detection thresholds.

    Returns:
        Array of onset frame indices (hop_length=512).
    """
    # Sensitivity → delta multiplier
    # low = fewer onsets (bigger delta), high = more onsets (smaller delta)
    delta_map = {"low": 2.0, "medium": 1.0, "high": 0.4}
    delta_scale = delta_map.get(sensitivity, 1.0)

    wait_frames = int(sr * MIN_ONSET_GAP)

    all_results = []

    # --- Strategy 1: Superflux (multi-band spectral) ---
    # Best for polyphonic music — detects onsets across frequency bands
    try:
        onset_env_sf = librosa.onset.onset_strength_multi(
            y=y, sr=sr, hop_length=512,
            channels=24,  # mel bands
        )
        # Average across bands
        onset_env_sf_mean = np.mean(onset_env_sf, axis=0)

        for delta in [0.08 * delta_scale, 0.12 * delta_scale, 0.2 * delta_scale]:
            frames = librosa.onset.onset_detect(
                onset_envelope=onset_env_sf_mean,
                sr=sr,
                backtrack=True,
                delta=delta,
                wait=wait_frames,
            )
            if len(frames) >= 2:
                all_results.append(("superflux", frames))
                break
        else:
            all_results.append(("superflux", frames))
    except Exception:
        pass

    # --- Strategy 2: Spectral flux ---
    try:
        onset_env_spec = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

        for delta in [0.10 * delta_scale, 0.20 * delta_scale, 0.35 * delta_scale]:
            frames = librosa.onset.onset_detect(
                onset_envelope=onset_env_spec,
                sr=sr,
                backtrack=True,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=delta,
                wait=wait_frames,
            )
            if len(frames) >= 2:
                all_results.append(("spectral", frames))
                break
        else:
            all_results.append(("spectral", frames))
    except Exception:
        pass

    # --- Strategy 3: Energy-based RMS ---
    try:
        frames = energy_based_onsets(y, sr, delta_scale)
        all_results.append(("energy", frames))
    except Exception:
        pass

    # --- Strategy 4: CQT energy per band ---
    try:
        frames = cqt_energy_onsets(y, sr, delta_scale)
        all_results.append(("cqt", frames))
    except Exception:
        pass

    # --- Pick the best result (most onsets) ---
    if all_results:
        best = max(all_results, key=lambda r: len(r[1]))
        return best[1]

    # --- Ultimate fallback: uniform grid ---
    return uniform_grid_onsets(y, sr)


def energy_based_onsets(y: np.ndarray, sr: int, delta_scale: float = 0.6) -> np.ndarray:
    """
    Detect onsets using RMS energy changes.
    Good for audio with clear note/silence separation.

    Args:
        y: Audio samples.
        sr: Sample rate.
        delta_scale: Sensitivity multiplier (lower = more onsets).

    Returns:
        Array of onset frame indices.
    """
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    rms_max = np.max(rms)
    rms_min = np.min(rms)

    # Adaptive threshold
    threshold = rms_min + 0.10 * delta_scale * (rms_max - rms_min)

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


def cqt_energy_onsets(y: np.ndarray, sr: int, delta_scale: float = 0.6) -> np.ndarray:
    """
    Detect onsets by tracking energy changes in the constant-Q spectrum.
    More sensitive to musically-relevant frequency changes than RMS alone.

    Args:
        y: Audio samples.
        sr: Sample rate.
        delta_scale: Sensitivity multiplier.

    Returns:
        Array of onset frame indices.
    """
    hop_length = 512
    # Compute CQT spectrogram
    cqt = np.abs(librosa.cqt(
        y=y, sr=sr, hop_length=hop_length,
        n_bins=72, bins_per_octave=12,
        fmin=librosa.note_to_hz("C2"),
    ))

    # Sum energy across all bins per frame
    energy = np.sum(cqt, axis=0)

    if len(energy) < 3:
        return np.array([])

    # Compute frame-to-frame energy difference
    diff = np.diff(energy)
    diff = np.concatenate([[0], diff])
    diff[diff < 0] = 0  # Only rising energy matters

    threshold = np.mean(diff) + 0.5 * delta_scale * np.std(diff)
    min_gap_frames = int(MIN_ONSET_GAP * sr / hop_length)

    onset_frames = []
    for i in range(1, len(diff)):
        if diff[i] > threshold and diff[i - 1] <= threshold:
            if not onset_frames or (i - onset_frames[-1]) >= min_gap_frames:
                onset_frames.append(i)

    return np.array(onset_frames)


def uniform_grid_onsets(y: np.ndarray, sr: int, grid_sec: float = 0.25) -> np.ndarray:
    """
    Ultimate fallback: segment audio into uniform time grid.
    Each grid cell becomes a note segment.

    Args:
        y: Audio samples.
        sr: Sample rate.
        grid_sec: Grid spacing in seconds (default 0.25s = 16th note at 60 BPM).

    Returns:
        Array of onset frame indices.
    """
    hop_length = 512
    total_sec = len(y) / sr
    num_onsets = max(2, int(total_sec / grid_sec))

    times = np.linspace(0, total_sec, num_onsets + 1)[:-1]
    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop_length)

    return frames


def frames_to_time(onset_frames: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Convert onset frames to time in seconds."""
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def get_note_segments(
    y: np.ndarray, sr: int, onset_times: np.ndarray
) -> list[dict]:
    """
    Split audio into note segments based on onset times.

    Each note runs from its onset time to the next onset (or audio end).
    """
    segments = []

    for i, start_t in enumerate(onset_times):
        end_t = onset_times[i + 1] if i + 1 < len(onset_times) else len(y) / sr

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
    """Detect beat positions and estimate tempo."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, float(tempo)
