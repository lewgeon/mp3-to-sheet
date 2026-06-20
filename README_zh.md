# MP3 转五线谱 🎵

> **🤖 AI 生成项目** — 本项目完全由 AI（Claude Code, Anthropic）创建。所有代码、文档和项目结构均通过自然语言提示生成。

将 MP3 音频文件转换为五线谱乐谱。自动检测单声道音频（独奏乐器、人声、哼唱）中的音高、节奏和速度，并生成标准 MusicXML 文件，可在任何打谱软件中打开。

## ✨ 功能特性

- **音频输入**：支持 MP3、WAV、M4A、FLAC、OGG 格式
- **音高检测**：使用 PYIN（概率 YIN）算法进行准确的基频估计
- **起始检测**：多策略起始检测（频谱通量 + 能量检测），实现稳健的音符分割
- **自动 BPM 估算**：根据音符时长推导出节奏速度
- **节奏量化**：将检测到的时长映射到标准音符时值（全音符、二分音符、四分音符、八分音符等）
- **调号检测**：自动确定最适合转录旋律的调号
- **多格式输出**：始终输出 MusicXML，安装 MuseScore 后可输出 PDF/PNG

## 📦 安装

### 前提条件

- Python 3.9+
- [MuseScore](https://musescore.org/)（可选 — 用于 PDF/PNG 渲染）

### 环境配置

```bash
cd mp3-to-sheet
pip install -r requirements.txt
```

### 依赖

| 包名 | 用途 |
|---------|---------|
| `librosa` | 音频加载、起始检测、音高估计 |
| `music21` | 音乐记谱、MusicXML 生成、乐谱渲染 |
| `numpy` | 数值计算 |
| `scipy` | 信号处理 |
| `soundfile` | 音频 I/O 后端 |

## 🚀 使用方法

### 基本用法

```bash
python main.py input.mp3
```

这会在同一目录下生成 `input.musicxml` 文件。

### 高级选项

```bash
# 指定输出路径
python main.py song.mp3 -o my_song

# 导出多种格式（PDF/PNG 需要安装 MuseScore）
python main.py song.mp3 --format musicxml pdf png

# 手动设置 BPM（跳过自动检测）
python main.py song.mp3 --bpm 120

# 自定义标题和拍号
python main.py song.mp3 --title "我的旋律" --time-signature 3/4

# 静默模式 — 仅显示最终结果
python main.py song.mp3 --quiet
```

### 完整命令行参考

```
用法: main.py [-h] [-o OUTPUT] [--bpm BPM]
              [--format {musicxml,pdf,png,midi} [...]]
              [--title TITLE] [--time-signature TIME_SIGNATURE] [--quiet]
              input

位置参数:
  input                 输入的音频文件路径

可选参数:
  -o, --output          输出文件路径前缀（不含扩展名）
  --bpm BPM             手动设置 BPM（跳过自动检测）
  --format {...}        输出格式（默认: musicxml）
  --title TITLE         乐谱标题
  --time-signature ...  拍号（默认: "4/4"）
  --quiet               关闭进度输出
```

## 🎼 工作原理

```
MP3 文件
  │
  ▼
audio_processor.py     加载音频 → 单声道，22050 Hz
  │
  ▼
onset_detector.py      检测音符边界（频谱 + 能量双策略）
  │
  ▼
pitch_detector.py      对每个音符估计音高（PYIN 算法）
  │
  ▼
transcriber.py         频率 → MIDI → 音名（C4, F#5, ...）
                       时长 → 量化节奏（四分音符、八分音符...）
                       自动检测 BPM
  │
  ▼
sheet_generator.py     构建 music21 乐谱，包含音符、休止符、
                       调号、拍号、速度标记
  │
  ▼
输出                   .musicxml（始终生成）+ .pdf/.png（需安装 MuseScore）
```

## 📁 项目结构

```
mp3-to-sheet/
├── main.py                  # CLI 入口 & 流程编排
├── requirements.txt         # Python 依赖
├── README.md                # 英文说明文档
├── README_zh.md             # 中文说明文档（本文件）
├── LICENSE                  # MIT 许可证
└── src/
    ├── __init__.py
    ├── audio_processor.py   # 音频加载、单声道转换、预处理
    ├── onset_detector.py    # 多策略音符起始检测
    ├── pitch_detector.py    # 基于 PYIN 的基频估计
    ├── transcriber.py       # 频率→MIDI→音名、BPM、节奏量化
    └── sheet_generator.py   # music21 乐谱构建 & 多格式导出
```

## 🎯 最佳使用场景

本工具最适合以下情况：

- ✅ 独奏乐器录音（钢琴、长笛、小提琴等）
- ✅ 人声旋律 / 哼唱
- ✅ 背景噪音较少的清晰录音
- ✅ 音符起止清晰的演奏

局限性：

- ⚠️ 多音音频（同时多个音符/和弦）— 支持有限
- ⚠️ 强混响或背景噪音可能降低检测准确率
- ⚠️ 高速度下的快速段落（如 32 分音符）可能需要手动设置 BPM

## 📄 许可证

本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。
