# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MP3-to-Sheet converts audio files to sheet music (MusicXML). The pipeline: load audio → detect note onsets → segment by note → detect pitch per segment → convert frequency→MIDI→note name → quantize rhythm → build music21 Score → export.

Target: monophonic audio (solo instruments, humming). Polyphonic input has experimental support via `--extract-melody`.

## Commands

```bash
# Install deps (conda env pre-configured)
conda activate melody && pip install -r requirements.txt

# Run (all formats)
python main.py input.mp3
python main.py input.mp3 -o output_prefix --format musicxml pdf

# Polyphonic / complex audio
python main.py input.mp3 --extract-melody --sensitivity high

# Clean solo audio — fewer false onsets
python main.py input.mp3 --sensitivity low

# Quick test: synthetic C major scale
python -c "
import numpy as np, soundfile as sf
sr=22050; notes=[261.63,293.66,329.63,349.23,392.00,440.00,493.88,523.25]
audio=np.array([])
for f in notes:
    t=np.linspace(0,0.5,int(sr*0.5),endpoint=False)
    audio=np.concatenate([audio,0.5*np.sin(2*np.pi*f*t),np.zeros(int(sr*0.15))])
sf.write('test.wav',audio,sr)" && python main.py test.wav
```

## Architecture

### Data flow & the note dict

The central data structure is the **note dict**, threaded through every pipeline stage:

```python
{
    "start_time": float,      # seconds from audio start
    "end_time": float,
    "duration": float,        # seconds
    "midi": int | None,       # MIDI note number (None = rest)
    "full_name": str,         # "C4", "F#5", or "rest"
    "name": str,              # pitch class: "C", "F#"
    "octave": int,
    "frequency": float | None,
    "cents_offset": float,    # tuning deviation from equal temperament
    "is_rest": bool,
    "duration_ql": float,     # set by transcriber.quantize_duration — quarter-length units
}
```

### Pipeline stages (in main.py `transcribe_audio`)

1. **Load** (`audio_processor.py`): `librosa.load` → mono 22050Hz, trim silence (top_db=40), normalize peak to 1.0.
2. **Melody extract** (optional, `melody_extractor.py`): HPSS harmonic/percussive separation → spectral peak tracking → synthesize clean melody signal. Only invoked when `--extract-melody`.
3. **Onset detect** (`onset_detector.py`): Runs 4 strategies in parallel — superflux (multi-band), spectral flux, RMS energy, CQT energy — and selects the result with the most onsets. Falls back to uniform grid if all fail. Sensitivity knob (`--sensitivity low|medium|high`) scales delta thresholds.
4. **Segment** (`onset_detector.get_note_segments`): Splits audio at onset times; each segment spans [onset[i], onset[i+1]) or [onset[i], EOF).
5. **Pitch detect** (`pitch_detector.py`): PYIN first → if voiced ratio ≥ 10%, take median Hz. Falls back to YIN if PYIN fails. Segments below adaptive silence threshold (5% of global RMS) become rests.
6. **Transcribe** (`transcriber.py`): `freq_to_note` converts Hz → MIDI → note name; `estimate_bpm` from median inter-onset interval; `quantize_duration` snaps duration to nearest standard rhythm value.
7. **Smooth** (`transcriber.smooth_pitch_errors`): Two-pass cleaner — (a) merge notes shorter than 80ms into previous note, (b) merge consecutive notes with identical MIDI (decay tail artifacts).
8. **Generate** (`sheet_generator.py`): Builds `music21.stream.Score` with Part, TimeSignature, auto-detected KeySignature, MetronomeMark, Note/Rest objects. Exports via `music21.write()` to .musicxml (always), optionally .pdf/.png/.mid.

### Key module responsibilities

| Module | Owns |
|--------|------|
| `audio_processor.py` | `load_audio()`, `preprocess()` — target SR = 22050 |
| `onset_detector.py` | `detect_onsets(y,sr,sensitivity)`, `get_note_segments()` — 4 parallel strategies, hop_length=512 |
| `pitch_detector.py` | `detect_pitch_pyin()`, `detect_pitch_yin()`, `get_stable_frequency()`, `is_silence()` — FMIN=C2 FMAX=C7 |
| `melody_extractor.py` | `extract_harmonic()`, `extract_predominant_melody()`, `preprocess_for_melody()` — HPSS + spectral peaks |
| `transcriber.py` | `freq_to_midi()`, `freq_to_note()`, `estimate_bpm()`, `quantize_duration()`, `smooth_pitch_errors()` — A4=440Hz=69 |
| `sheet_generator.py` | `create_score(notes_data,bpm,title,time_signature)`, `export_score()` — music21 Score assembly + multi-format export |

### Important design notes

- **Onset hop_length is 512 everywhere**. `frames_to_time()` assumes this. Don't introduce a different hop_length without threading it through.
- **PYIN voiced threshold is deliberately low (10%)** after fixes for short/weak notes. Don't raise it without testing on real audio.
- **smooth_pitch_errors Pass 2** merges consecutive same-MIDI notes. This is critical — without it, decay tails or double-triggered onsets produce duplicate note entries.
- **music21 suppresses its own warnings** at import time in `sheet_generator.py` (MuseScore path noise). Don't remove that filter.
- **PDF/PNG export requires MuseScore or LilyPond** installed on the system. The code gracefully falls back to MusicXML-only with a warning.
- Python 3.9 target — no `X | Y` union syntax in runtime-evaluated annotations. Use `Optional[X]` or `from __future__ import annotations`.
