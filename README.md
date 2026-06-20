# MP3 to Sheet Music 🎵

> **🤖 AI-Generated Project** — This project was entirely created by AI (Claude Code, Anthropic). All code, documentation, and project structure were generated through natural language prompts.

Convert MP3 audio files into sheet music notation (五线谱). Automatically detects pitch, rhythm, and tempo from monophonic audio (solo instruments, vocals, humming) and generates standard MusicXML files that can be opened in any sheet music software.

## ✨ Features

- **Audio Input**: Supports MP3, WAV, M4A, FLAC, OGG formats
- **Pitch Detection**: Uses the PYIN (Probabilistic YIN) algorithm for accurate fundamental frequency estimation
- **Onset Detection**: Multi-strategy onset detection (spectral flux + energy-based) for robust note segmentation
- **Automatic BPM Estimation**: Derives tempo from note durations
- **Rhythm Quantization**: Maps detected durations to standard musical note values (whole, half, quarter, eighth, etc.)
- **Key Detection**: Automatically determines the best key signature for the transcribed melody
- **Multiple Output Formats**: MusicXML (always), plus PDF/PNG with MuseScore installed

## 📦 Installation

### Prerequisites

- Python 3.9+
- [MuseScore](https://musescore.org/) (optional — for PDF/PNG rendering)

### Setup

```bash
cd mp3-to-sheet
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading, onset detection, pitch estimation |
| `music21` | Music notation, MusicXML generation, score rendering |
| `numpy` | Numerical computation |
| `scipy` | Signal processing |
| `soundfile` | Audio I/O backend |

## 🚀 Usage

### Basic Usage

```bash
python main.py input.mp3
```

This produces `input.musicxml` in the same directory.

### Advanced Options

```bash
# Specify output path
python main.py song.mp3 -o my_song

# Export to multiple formats (PDF/PNG require MuseScore)
python main.py song.mp3 --format musicxml pdf png

# Manually set BPM (skip auto-detection)
python main.py song.mp3 --bpm 120

# Customize title and time signature
python main.py song.mp3 --title "My Melody" --time-signature 3/4

# Quiet mode — only show final result
python main.py song.mp3 --quiet
```

### Full CLI Reference

```
usage: main.py [-h] [-o OUTPUT] [--bpm BPM]
               [--format {musicxml,pdf,png,midi} [...]]
               [--title TITLE] [--time-signature TIME_SIGNATURE] [--quiet]
               input

positional arguments:
  input                 Path to input audio file

optional arguments:
  -o, --output          Output file path prefix (without extension)
  --bpm BPM             Manually set BPM (skip auto-detection)
  --format {...}        Output formats (default: musicxml)
  --title TITLE         Title for the sheet music
  --time-signature ...  Time signature (default: "4/4")
  --quiet               Suppress progress output
```

## 🎼 How It Works

```
MP3 File
  │
  ▼
audio_processor.py     Load audio → mono, 22050 Hz
  │
  ▼
onset_detector.py      Detect note boundaries (spectral + energy)
  │
  ▼
pitch_detector.py      Estimate pitch for each note (PYIN algorithm)
  │
  ▼
transcriber.py         Frequency → MIDI → Note name (C4, F#5, ...)
                       Duration → Quantized rhythm (quarter, eighth, ...)
                       Auto-detect BPM
  │
  ▼
sheet_generator.py     Build music21 Score with notes, rests,
                       key signature, time signature, tempo
  │
  ▼
Output                 .musicxml (always) + .pdf/.png (with MuseScore)
```

## 📁 Project Structure

```
mp3-to-sheet/
├── main.py                  # CLI entry point & pipeline orchestration
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── src/
    ├── __init__.py
    ├── audio_processor.py   # Audio loading, mono conversion, preprocessing
    ├── onset_detector.py    # Multi-strategy note onset detection
    ├── pitch_detector.py    # PYIN-based fundamental frequency estimation
    ├── transcriber.py       # Frequency→MIDI→Note name, BPM, rhythm quantization
    └── sheet_generator.py   # music21 Score builder & multi-format exporter
```

## 🎯 Best Results

This tool works best with:

- ✅ Solo instrument recordings (piano, flute, violin, etc.)
- ✅ Vocal melodies / humming
- ✅ Clean recordings with minimal background noise
- ✅ Distinct note attacks (clear separation between notes)

Limitations:

- ⚠️ Polyphonic audio (multiple simultaneous notes / chords) — limited support
- ⚠️ Heavy reverb or background noise may reduce detection accuracy
- ⚠️ Very fast passages (e.g., 32nd notes at high tempo) may need manual BPM

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
