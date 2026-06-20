"""
Melody extractor: preprocess polyphonic audio to isolate the dominant melody.

Uses Harmonic-Percussive Source Separation (HPSS) to isolate the harmonic
content, then tracks spectral peaks to extract the predominant melody line.

Recommended for: multi-instrument BGM, songs with accompaniment, polyphonic audio.
Not needed for: solo instrument recordings, humming, monophonic input.
"""

import numpy as np
import librosa


def extract_harmonic(y: np.ndarray, sr: int, margin: float = 3.0) -> np.ndarray:
    """
    Separate harmonic (pitched) content from percussive (noise) content.

    Uses median-filtering based HPSS. The harmonic component contains
    sustained tones (instruments, voice) while the percussive component
    contains transients (drums, attacks).

    Args:
        y: Audio samples.
        sr: Sample rate.
        margin: Separation margin in dB (higher = more aggressive separation).

    Returns:
        Harmonic component (same shape as input).
    """
    # Compute STFT
    D = librosa.stft(y)

    # Separate harmonic and percussive
    H, P = librosa.decompose.hpss(D, margin=margin)

    # Reconstruct harmonic signal
    y_harmonic = librosa.istft(H, length=len(y))

    return y_harmonic


def extract_predominant_melody(
    y: np.ndarray, sr: int, hop_length: int = 512
) -> np.ndarray:
    """
    Extract the predominant melody line from audio by tracking
    spectral peaks in the frequency domain.

    This is a simplified approach: compute the STFT, find the
    strongest frequency in each frame, and build a "melody signal"
    by synthesizing from those frequencies.

    For better results with complex audio, consider using a dedicated
    melody extraction model (e.g., CREPE, Spleeter).

    Args:
        y: Audio samples (preferably harmonic component from HPSS).
        sr: Sample rate.
        hop_length: Hop length for STFT.

    Returns:
        Synthesized melody signal (mono, same length as input).
    """
    # Compute STFT
    D = librosa.stft(y, hop_length=hop_length)
    mag = np.abs(D)

    # Find dominant frequency per frame
    n_fft = 2048
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Limit to melody range: C3 (~131 Hz) to C6 (~1047 Hz)
    fmin_idx = np.searchsorted(freqs, librosa.note_to_hz("C3"))
    fmax_idx = np.searchsorted(freqs, librosa.note_to_hz("C6"))

    melody_freqs = []
    melody_mags = []

    for frame_idx in range(mag.shape[1]):
        frame_mag = mag[fmin_idx:fmax_idx, frame_idx]
        if len(frame_mag) == 0:
            melody_freqs.append(0.0)
            melody_mags.append(0.0)
            continue

        # Find the peak in the melody range
        peak_idx = np.argmax(frame_mag)
        peak_freq = freqs[fmin_idx + peak_idx]
        peak_mag = frame_mag[peak_idx]

        # Only keep if magnitude is significant
        if peak_mag > 0.05 * np.max(frame_mag):
            melody_freqs.append(peak_freq)
            melody_mags.append(peak_mag)
        else:
            melody_freqs.append(0.0)
            melody_mags.append(0.0)

    melody_freqs = np.array(melody_freqs)
    melody_mags = np.array(melody_mags)

    # Synthesize melody signal from tracked frequencies
    t_frames = librosa.frames_to_samples(
        np.arange(mag.shape[1]), hop_length=hop_length
    )
    t_frames = np.clip(t_frames, 0, len(y) - 1)

    # Build signal with overlap-add
    melody_signal = np.zeros(len(y))
    window = np.hanning(hop_length * 2)

    for i, (freq, mag_val) in enumerate(zip(melody_freqs, melody_mags)):
        if freq <= 0:
            continue
        start = t_frames[i]
        end = min(start + len(window), len(y))
        seg_len = end - start
        t = np.linspace(0, seg_len / sr, seg_len, endpoint=False)
        tone = mag_val * np.sin(2 * np.pi * freq * t)
        melody_signal[start:end] += tone * window[:seg_len]

    # Normalize
    peak = np.max(np.abs(melody_signal))
    if peak > 0:
        melody_signal /= peak

    return melody_signal


def preprocess_for_melody(
    y: np.ndarray, sr: int, use_hpss: bool = True,
) -> np.ndarray:
    """
    Full melody extraction pipeline.

    1. (Optional) HPSS to isolate harmonic content
    2. Track spectral peaks to extract predominant melody
    3. Synthesize clean melody signal

    Args:
        y: Original audio samples.
        sr: Sample rate.
        use_hpss: Whether to apply HPSS first (recommended for polyphonic audio).

    Returns:
        Synthesized melody signal ready for onset + pitch detection.
    """
    if use_hpss:
        y = extract_harmonic(y, sr)

    melody = extract_predominant_melody(y, sr)

    return melody
