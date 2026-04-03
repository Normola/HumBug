# HumBug 🔊

Real-time Electric Network Frequency (ENF) analyser with automatic speaker diarization.

Stream any YouTube video, watch the mains hum frequency drift in real-time, and see each speaker's frequency trace on a separate tab — useful for verifying recording authenticity, detecting edits, or geolocation from a 50 Hz / 60 Hz signature.

![Overview screenshot placeholder](docs/screenshot.png)

---

## What is ENF analysis?

Every recording made with plugged-in equipment (cameras, microphones, laptops) picks up a faint 50 Hz or 60 Hz tone from the electrical grid — plus its harmonics at 100, 150, 200 Hz etc.  The grid doesn't run at *exactly* 50.000 Hz; it wobbles slightly as load fluctuates, creating a unique frequency fingerprint for each moment in time.

This fingerprint can be used to:
- **Timestamp** a recording by matching its ENF trace against reference grid data
- **Detect tampering** — cuts and splices break the continuity of the drift
- **Geolocate** a recording — 50 Hz means EU/UK/AU, 60 Hz means US/CA/MX, and a recording that switches between them reveals a video call between participants in different countries

HumBug tracks this drift live with ~0.003 Hz effective resolution using FFT with zero-padding and quadratic interpolation across multiple harmonics.

---

## Features

- **Dual-standard tracking** — 50 Hz and 60 Hz plots simultaneously; auto-detects which is dominant
- **Speaker diarization** — separate tab per speaker, colour-coded; background of overview plots shaded by active speaker
- **Real-time waveform** display
- **Optional audio playback** via `mpv` alongside analysis
- **Pre-trained ONNX models** for best accuracy (Silero VAD + WeSpeaker ResNet34)
- **Graceful fallback** to WebRTC VAD + MFCC if models haven't been downloaded

---

## Requirements

- Python 3.12+ (3.14 works; PyQt5 must be system-installed for 3.14 on ARM)
- `yt-dlp`, `ffmpeg`, `mpv`

Install on Fedora / RHEL:
```bash
sudo dnf install yt-dlp ffmpeg mpv python3-PyQt5 python3-pyqtgraph
```

On Debian / Ubuntu:
```bash
sudo apt install yt-dlp ffmpeg mpv python3-pyqt5 python3-pyqtgraph
```

---

## Setup

```bash
git clone https://github.com/yourname/HumBug
cd HumBug

# Create venv (--system-site-packages needed for PyQt5 on Python 3.14/ARM)
python3 -m venv --system-site-packages .venv
.venv/bin/pip install numpy scipy scikit-learn webrtcvad onnxruntime

# Download pre-trained ONNX models (~28 MB total)
.venv/bin/python3 download_models.py
```

---

## Usage

```bash
./run.sh
# or
.venv/bin/python3 humbug.py
```

1. Paste a YouTube URL into the URL field
2. Set the number of expected speakers (default: 2)
3. Tick **🔊 Play** if you want to hear the audio alongside the analysis
4. Press **▶ Start**

The **Overview** tab shows all data. Per-speaker tabs appear automatically after ~10 seconds once the diarizer has enough data to cluster.

---

## How speaker diarization works

| Stage | Model | Notes |
|---|---|---|
| **Voice Activity Detection** | [Silero VAD v5](https://github.com/snakers4/silero-vad) ONNX | 32 ms windows at 8 kHz; replaces WebRTC VAD |
| **Speaker embedding** | [WeSpeaker ResNet34-LM](https://github.com/wenet-e2e/wespeaker) ONNX | 80-dim log-mel fbank at 16 kHz → 256-dim L2-norm embedding |
| **Clustering** | k-means (scikit-learn) | Re-runs every 5 s; labels stabilised via Hungarian matching |

Without the ONNX models, the app falls back to WebRTC VAD + 13-coefficient MFCCs. Results are usable but speaker separation is less reliable for similar-sounding voices.

For significantly better diarization, install Python 3.12, add `pyannote.audio` (needs PyTorch + a free HuggingFace token), and swap in the `pyannote` pipeline.

---

## File structure

```
humbug.py            Main application
download_models.py   Downloads Silero VAD + WeSpeaker ONNX models
run.sh               Launcher
models/
  silero_vad.onnx         ~1.8 MB  Silero VAD v5
  wespeaker_resnet34.onnx ~26 MB   WeSpeaker ResNet34-LM (VoxCeleb)
.venv/               Python virtual environment
```

---

## Known limitations

- **ENF may be absent** in heavily compressed audio (e.g. 64 kbps AAC). Recordings made with camera microphones indoors tend to have the strongest signal.
- **Speaker diarization** can't distinguish speakers who sound very similar without a better embedding model.
- **First ~10 seconds** of speaker data show as "unknown" before the first clustering pass.
- **Analysis runs faster than real-time** — the ENF plots will race ahead of audio playback.

---

## Licence

MIT
