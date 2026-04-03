#!/usr/bin/env python3
"""
HumBug - Electric Network Frequency (ENF) Analyser with Speaker Diarization

Audio pipeline:  yt-dlp → ffmpeg (8 kHz mono PCM) → analysis thread
ENF analysis:    4-second rolling FFT window, quadratic interpolation across
                 multiple harmonics, ~0.003 Hz effective resolution
Speaker ID:      Silero VAD + WeSpeaker ResNet34 embeddings (ONNX) → k-means
                 with Hungarian stable labelling.  Falls back to webrtcvad +
                 MFCC if models haven't been downloaded yet (run
                 download_models.py first for best results).
"""

import sys
import subprocess
import threading
import numpy as np
from collections import deque, Counter
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QRadioButton, QButtonGroup, QSplitter,
    QFrame, QTabWidget, QSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg


MODELS_DIR = Path(__file__).parent / "models"

# ── Audio pipeline ────────────────────────────────────────────────────────────
SAMPLE_RATE  = 8000    # Hz — ENF pipeline stays at 8 kHz
WINDOW_SEC   = 4.0
HOP_SEC      = 0.5
FFT_PAD_MULT = 8       # → ~0.031 Hz bins before quad-interp

# ── ENF tracking ──────────────────────────────────────────────────────────────
ENF_RANGE    = 0.6
HARMONICS_50 = [1, 2, 3, 4, 5]
HARMONICS_60 = [1, 2, 3, 4]

# ── Display ───────────────────────────────────────────────────────────────────
HISTORY_SEC  = 300
PLOT_WINDOW  = 120

# ── Diarization ───────────────────────────────────────────────────────────────
VAD_SPEECH_THRESH = 0.45   # Silero probability threshold
RECLUSTER_SECS    = 5.0
MIN_CLUSTER_VECS  = 20

# ── WeSpeaker fbank parameters (16 kHz) ──────────────────────────────────────
_FB_SR      = 16000
_FB_FRAME_N = 400    # 25 ms
_FB_HOP_N   = 160    # 10 ms
_FB_FFT_N   = 512
_FB_N_MEL   = 80

# ── Colours ───────────────────────────────────────────────────────────────────
SPEAKER_COLORS   = ["#ef9a9a", "#90caf9", "#a5d6a7", "#ffe082", "#ce93d8", "#80deea"]
SPEAKER_BG_ALPHA = 35   # 0–255; background shading opacity on overview plots


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_fbank_matrix() -> np.ndarray:
    """80-bin mel filterbank for 16 kHz / 512-point FFT (Kaldi-style)."""
    fmin, fmax = 20.0, 7600.0
    to_mel   = lambda f: 1127 * np.log(1 + f / 700)
    from_mel = lambda m: 700 * (np.exp(m / 1127) - 1)
    mel_pts  = np.linspace(to_mel(fmin), to_mel(fmax), _FB_N_MEL + 2)
    hz_pts   = from_mel(mel_pts)
    bins     = np.floor((_FB_FFT_N + 1) * hz_pts / _FB_SR).astype(int)
    n_bins   = _FB_FFT_N // 2 + 1
    fb       = np.zeros((_FB_N_MEL, n_bins))
    for m in range(1, _FB_N_MEL + 1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        if fc > fl:
            for k in range(max(fl, 0), min(fc, n_bins)):
                fb[m-1, k] = (k - fl) / (fc - fl)
        if fr > fc:
            for k in range(max(fc, 0), min(fr, n_bins)):
                fb[m-1, k] = (fr - k) / (fr - fc)
    return fb


_FBANK_MAT  = _build_fbank_matrix()
_FBANK_HAMM = np.hamming(_FB_FRAME_N)


def compute_fbank(signal_16k: np.ndarray) -> np.ndarray | None:
    """
    80-dim log-mel fbank (Kaldi style) → (T, 80) float32, per-frame mean sub.
    Used as input to wespeaker ResNet34.
    """
    if len(signal_16k) < _FB_FRAME_N:
        return None
    sig      = np.concatenate([[signal_16k[0]], signal_16k[1:] - 0.97 * signal_16k[:-1]])
    n_frames = (len(sig) - _FB_FRAME_N) // _FB_HOP_N + 1
    idx      = (np.arange(_FB_FRAME_N)[None, :] +
                (np.arange(n_frames) * _FB_HOP_N)[:, None])
    frames   = sig[idx] * _FBANK_HAMM
    power    = (np.abs(np.fft.rfft(frames, n=_FB_FFT_N)) ** 2) / _FB_FFT_N
    mel      = np.dot(power, _FBANK_MAT.T)
    mel      = np.where(mel < 1e-10, 1e-10, mel)
    feats    = np.log(mel)
    feats   -= feats.mean(axis=0)
    return feats.astype(np.float32)


# Legacy MFCC fallback (used if wespeaker model not downloaded)
_MFCC_N    = 13; _MFCC_MEL = 26; _MFCC_FRAME = 200; _MFCC_HOP = 80; _MFCC_FFT = 256

def _build_mfcc_fb() -> np.ndarray:
    to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    fm     = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts    = np.linspace(to_mel(80), to_mel(3800), _MFCC_MEL + 2)
    bins   = np.floor((_MFCC_FFT + 1) * fm(pts) / SAMPLE_RATE).astype(int)
    nb     = _MFCC_FFT // 2 + 1
    fb     = np.zeros((_MFCC_MEL, nb))
    for m in range(1, _MFCC_MEL + 1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        if fc > fl:
            for k in range(max(fl, 0), min(fc, nb)): fb[m-1, k] = (k-fl)/(fc-fl)
        if fr > fc:
            for k in range(max(fc, 0), min(fr, nb)): fb[m-1, k] = (fr-k)/(fr-fc)
    return fb

_MFCC_FB   = _build_mfcc_fb()
_MFCC_HAMM = np.hamming(_MFCC_FRAME)

def compute_mfcc(signal: np.ndarray) -> np.ndarray | None:
    from scipy.fftpack import dct
    if len(signal) < _MFCC_FRAME: return None
    sig    = np.concatenate([[signal[0]], signal[1:] - 0.97 * signal[:-1]])
    nf     = (len(sig) - _MFCC_FRAME) // _MFCC_HOP + 1
    idx    = np.arange(_MFCC_FRAME)[None, :] + (np.arange(nf) * _MFCC_HOP)[:, None]
    frames = sig[idx] * _MFCC_HAMM
    power  = (np.abs(np.fft.rfft(frames, n=_MFCC_FFT)) ** 2) / _MFCC_FFT
    mel    = np.dot(power, _MFCC_FB.T); mel = np.where(mel < 1e-10, 1e-10, mel)
    return dct(np.log(mel), type=2, axis=1, norm="ortho")[:, :_MFCC_N].mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-trained ONNX model wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SileroVAD:
    """
    Silero VAD v5 ONNX — runs at 8 kHz with 256-sample windows (32 ms).
    State tensor shape: [2, 1, 128].
    Returns mean speech probability over a chunk.
    """
    CHUNK_N    = 256    # 32 ms at 8 kHz
    STATE_SIZE = 128    # v5 hidden dim

    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.log_severity_level   = 3
        self._sess = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._reset()

    def _reset(self):
        self._state = np.zeros((2, 1, self.STATE_SIZE), dtype=np.float32)

    def speech_prob(self, audio: np.ndarray) -> float:
        """audio: float32 [-1, 1] at 8 kHz.  Returns mean speech probability."""
        probs = []
        for start in range(0, len(audio) - self.CHUNK_N + 1, self.CHUNK_N):
            win = audio[start : start + self.CHUNK_N][np.newaxis, :].astype(np.float32)
            out, self._state = self._sess.run(
                None,
                {"input": win,
                 "state": self._state,
                 "sr":    np.array(SAMPLE_RATE, dtype=np.int64)},
            )
            probs.append(float(out[0][0]))
        return float(np.mean(probs)) if probs else 0.0


class WeSpeakerEmbedder:
    """
    WeSpeaker ResNet34-LM ONNX — VoxCeleb-trained speaker embeddings.
    Input tensor: 'feats'  [B, T, 80] fbank at 16 kHz.
    Output tensor: 'embs'  [B, 256] L2-normalised.
    Audio is internally upsampled from 8 kHz to 16 kHz before fbank extraction.
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])

    def embed(self, audio_8k: np.ndarray) -> np.ndarray | None:
        from scipy.signal import resample_poly
        audio_16k = resample_poly(audio_8k, up=2, down=1).astype(np.float32)
        feats = compute_fbank(audio_16k)        # (T, 80)
        if feats is None or len(feats) < 20:
            return None
        emb  = self._sess.run(None, {"feats": feats[np.newaxis]})[0][0]   # (256,)
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 1e-10 else emb


# ─────────────────────────────────────────────────────────────────────────────
# Speaker diarizer — uses pre-trained models with MFCC fallback
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerDiarizer:

    def __init__(self, n_speakers: int = 2):
        self.n_speakers = n_speakers
        self._vecs: list[tuple[float, np.ndarray]] = []
        self._labels: dict[float, int] = {}
        self._last_cluster = 0.0

        self._vad      = self._try_load_silero_vad()
        self._embedder = self._try_load_wespeaker()

        # Fallback VAD
        self._legacy_vad = None
        if self._vad is None:
            try:
                import webrtcvad
                self._legacy_vad = webrtcvad.Vad(2)
            except ImportError:
                pass

    def model_info(self) -> str:
        vad = "SileroVAD"   if self._vad      else "WebRTCVAD"
        emb = "WeSpeaker"   if self._embedder  else "MFCC (fallback)"
        return f"{vad} + {emb}"

    # ── Public API ────────────────────────────────────────────────────────────

    def process_chunk(self, timestamp: float, raw_int16: bytes) -> bool:
        signal = np.frombuffer(raw_int16, dtype=np.int16).astype(np.float32) / 32768.0

        is_speech = self._is_speech(signal, raw_int16)
        if not is_speech:
            return False

        vec = (self._embedder.embed(signal) if self._embedder is not None
               else compute_mfcc(signal))
        if vec is not None:
            self._vecs.append((timestamp, vec))
        return True

    def maybe_recluster(self, current_time: float) -> bool:
        if current_time - self._last_cluster < RECLUSTER_SECS:
            return False
        if len(self._vecs) < MIN_CLUSTER_VECS:
            return False
        self._do_cluster()
        self._last_cluster = current_time
        return True

    def get_speaker(self, window_start: float, window_end: float) -> int:
        in_win = [
            self._labels[t]
            for t, _ in self._vecs
            if window_start <= t <= window_end and t in self._labels
        ]
        return Counter(in_win).most_common(1)[0][0] if in_win else -1

    # ── Private ───────────────────────────────────────────────────────────────

    def _is_speech(self, signal: np.ndarray, raw_int16: bytes) -> bool:
        if self._vad is not None:
            return self._vad.speech_prob(signal) >= VAD_SPEECH_THRESH

        if self._legacy_vad is not None:
            frame_bytes = (SAMPLE_RATE * 30 // 1000) * 2   # 30 ms frames
            s = t = 0
            for off in range(0, len(raw_int16) - frame_bytes + 1, frame_bytes):
                try:
                    s += self._legacy_vad.is_speech(raw_int16[off:off+frame_bytes], SAMPLE_RATE)
                except Exception:
                    s += 1
                t += 1
            return (s / t >= 0.25) if t else True

        return True   # no VAD at all — treat everything as speech

    def _do_cluster(self):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        from scipy.optimize import linear_sum_assignment

        timestamps = [t for t, _ in self._vecs]
        X          = normalize(np.stack([v for _, v in self._vecs]), norm="l2")
        n          = min(self.n_speakers, len(X))

        km         = KMeans(n_clusters=n, n_init=5, random_state=42)
        new_labels = km.fit_predict(X)

        if self._labels:
            overlap = np.zeros((n, n), dtype=int)
            for i, t in enumerate(timestamps):
                if t in self._labels:
                    o, nw = self._labels[t], int(new_labels[i])
                    if 0 <= o < n and 0 <= nw < n:
                        overlap[o, nw] += 1
            r_idx, c_idx = linear_sum_assignment(-overlap)
            mapping = np.arange(n)
            for r, c in zip(r_idx, c_idx):
                mapping[c] = r
            new_labels = np.array([mapping[l] for l in new_labels])

        for t, lab in zip(timestamps, new_labels):
            self._labels[t] = int(lab)

    @staticmethod
    def _try_load_silero_vad():
        path = MODELS_DIR / "silero_vad.onnx"
        if not path.exists():
            return None
        try:
            return SileroVAD(str(path))
        except Exception:
            return None

    @staticmethod
    def _try_load_wespeaker():
        for name in ("wespeaker_resnet34.onnx", "wespeaker_ecapa512.onnx"):
            path = MODELS_DIR / name
            if path.exists():
                try:
                    return WeSpeakerEmbedder(str(path))
                except Exception:
                    pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

class ENFWorker(QObject):
    frequency_update = pyqtSignal(float, float, float, float, float, int)
    audio_chunk      = pyqtSignal(float, bytes)
    status_update    = pyqtSignal(str)
    model_info       = pyqtSignal(str)
    error_occurred   = pyqtSignal(str)
    finished         = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._stop_event  = threading.Event()
        self._process     = None
        self._mpv_process = None

    def start_analysis(self, url: str, play_audio: bool = False, n_speakers: int = 2):
        self._stop_event.clear()
        threading.Thread(
            target=self._run, args=(url, play_audio, n_speakers), daemon=True
        ).start()

    def stop(self):
        self._stop_event.set()
        for p in (self._process, self._mpv_process):
            if p:
                try: p.terminate()
                except Exception: pass

    def _run(self, url: str, play_audio: bool, n_speakers: int):
        try:
            self.status_update.emit("Resolving stream URL…")
            audio_url = self._get_audio_url(url)
            if not audio_url:
                self.error_occurred.emit("yt-dlp could not resolve an audio URL.")
                return

            if play_audio:
                self._mpv_process = subprocess.Popen(
                    ["mpv", "--no-video", "--really-quiet", audio_url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

            self.status_update.emit("Connecting to stream…")
            cmd = [
                "ffmpeg",
                "-reconnect", "1", "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-i", audio_url,
                "-vn", "-f", "s16le",
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-loglevel", "error", "pipe:1",
            ]
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            self.status_update.emit("Analysing…")
            self._analyse_stream(n_speakers)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            for p in (self._process, self._mpv_process):
                if p:
                    try: p.terminate(); p.wait(timeout=2)
                    except Exception: pass
            self.finished.emit()

    def _get_audio_url(self, url: str) -> str | None:
        for fmt in ["bestaudio", ""]:
            args = ["yt-dlp", "--get-url", "--no-playlist"]
            if fmt: args += ["-f", fmt]
            r = subprocess.run(args + [url], capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().split("\n")[0]
        return None

    def _analyse_stream(self, n_speakers: int):
        window_n  = int(WINDOW_SEC * SAMPLE_RATE)
        hop_n     = int(HOP_SEC    * SAMPLE_RATE)
        fft_n     = window_n * FFT_PAD_MULT
        hann      = np.hanning(window_n).astype(np.float64)
        freqs     = np.fft.rfftfreq(fft_n, d=1.0 / SAMPLE_RATE)
        buffer    = np.zeros(window_n, dtype=np.float64)
        elapsed   = 0.0

        diarizer = SpeakerDiarizer(n_speakers)
        self.model_info.emit(diarizer.model_info())

        while not self._stop_event.is_set():
            raw = self._read_exactly(self._process.stdout, hop_n * 2)
            if raw is None:
                break

            chunk   = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
            buffer  = np.roll(buffer, -hop_n)
            buffer[-hop_n:] = chunk
            elapsed += HOP_SEC

            self.audio_chunk.emit(elapsed, raw)
            diarizer.process_chunk(elapsed, raw)
            diarizer.maybe_recluster(elapsed)

            if elapsed < WINDOW_SEC:
                continue

            self.status_update.emit(f"Analysing — {elapsed:.0f} s")

            spectrum = np.abs(np.fft.rfft(buffer * hann, n=fft_n))
            f50, c50 = self._track_enf(spectrum, freqs, 50.0, HARMONICS_50)
            f60, c60 = self._track_enf(spectrum, freqs, 60.0, HARMONICS_60)
            spk      = diarizer.get_speaker(elapsed - WINDOW_SEC, elapsed)

            self.frequency_update.emit(elapsed, f50, c50, f60, c60, spk)

    def _read_exactly(self, stream, n_bytes: int) -> bytes | None:
        buf = b""
        while len(buf) < n_bytes:
            if self._stop_event.is_set(): return None
            chunk = stream.read(n_bytes - len(buf))
            if not chunk: return buf or None
            buf += chunk
        return buf

    @staticmethod
    def _track_enf(spectrum, freqs, nominal, harmonics):
        estimates, weights = [], []
        for h in harmonics:
            lo = nominal * h - ENF_RANGE * h; hi = nominal * h + ENF_RANGE * h
            mask = (freqs >= lo) & (freqs <= hi)
            if not mask.any(): continue
            ss = spectrum[mask]; sf = freqs[mask]
            pk = int(np.argmax(ss)); mag = float(ss[pk])
            if 0 < pk < len(ss) - 1:
                a, b, c = ss[pk-1], ss[pk], ss[pk+1]; d = a - 2*b + c
                pf = sf[pk] + (0.5*(a-c)/d if d else 0) * (sf[1]-sf[0] if len(sf)>1 else 0)
            else:
                pf = sf[pk]
            estimates.append(pf / h); weights.append(mag)
        if not estimates: return nominal, 0.0
        w = np.array(weights)
        return float(np.average(estimates, weights=w)), float(w.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Per-speaker tab data container
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerTab:
    def __init__(self, speaker_id: int, color: str):
        self.speaker_id = speaker_id
        self.color      = color
        maxlen          = int(HISTORY_SEC / HOP_SEC)
        self.times = deque(maxlen=maxlen)
        self.f50   = deque(maxlen=maxlen)
        self.f60   = deque(maxlen=maxlen)
        self.c50   = deque(maxlen=maxlen)
        self.c60   = deque(maxlen=maxlen)
        self.widget = self.curve50 = self.curve60 = self.pw50 = self.pw60 = None

    def append(self, t, f50, c50, f60, c60):
        self.times.append(t); self.f50.append(f50); self.c50.append(c50)
        self.f60.append(f60); self.c60.append(c60)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class HumBugWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HumBug — ENF Analyser")
        self.resize(1280, 820)

        maxlen = int(HISTORY_SEC / HOP_SEC)
        self._ov_times = deque(maxlen=maxlen)
        self._ov_f50   = deque(maxlen=maxlen)
        self._ov_f60   = deque(maxlen=maxlen)
        self._ov_c50   = deque(maxlen=maxlen)
        self._ov_c60   = deque(maxlen=maxlen)
        self._ov_spk   = deque(maxlen=maxlen)   # speaker id per ENF frame

        self._wave_buf = np.zeros(8 * SAMPLE_RATE, dtype=np.float32)
        self._wave_t   = np.linspace(0.0, 8.0, len(self._wave_buf))

        self._speaker_tabs: dict[int, SpeakerTab] = {}

        # Background shading regions on overview ENF plots
        self._bg_regions_50: list = []
        self._bg_regions_60: list = []

        self._worker = ENFWorker()
        self._worker.frequency_update.connect(self._on_freq)
        self._worker.audio_chunk.connect(self._on_audio_chunk)
        self._worker.status_update.connect(self._on_status)
        self._worker.model_info.connect(self._on_model_info)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._setup_ui()

        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self._refresh_plots)
        self._plot_timer.setInterval(250)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        pg.setConfigOptions(antialias=True, background="#1a1a1a", foreground="#dddddd")

        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(8, 8, 8, 4)
        vbox.setSpacing(6)

        # ── Controls row ──
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("URL:"))
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("https://www.youtube.com/watch?v=…")
        self._url_input.returnPressed.connect(self._start)
        ctrl.addWidget(self._url_input, stretch=1)

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedWidth(100)
        self._start_btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:white;font-weight:bold;"
            "padding:4px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#388e3c;}"
            "QPushButton:disabled{background:#424242;color:#757575;}"
        )
        self._start_btn.clicked.connect(self._start)
        ctrl.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedWidth(100)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton{background:#b71c1c;color:white;font-weight:bold;"
            "padding:4px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#c62828;}"
            "QPushButton:disabled{background:#424242;color:#757575;}"
        )
        self._stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self._stop_btn)

        self._play_chk = QCheckBox("🔊 Play")
        self._play_chk.setChecked(True)
        ctrl.addWidget(self._play_chk)

        ctrl.addWidget(QLabel("  Speakers:"))
        self._spk_spin = QSpinBox()
        self._spk_spin.setRange(1, 6)
        self._spk_spin.setValue(2)
        self._spk_spin.setFixedWidth(50)
        ctrl.addWidget(self._spk_spin)
        vbox.addLayout(ctrl)

        # ── Live readout row ──
        live = QHBoxLayout()
        live.addWidget(QLabel("Display:"))
        self._mode_group = QButtonGroup(self)
        for label in ("Both", "50 Hz", "60 Hz"):
            rb = QRadioButton(label)
            self._mode_group.addButton(rb)
            live.addWidget(rb)
        self._mode_group.buttons()[0].setChecked(True)
        self._mode_group.buttonClicked.connect(lambda _: self._refresh_plots())
        live.addSpacing(20)
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        live.addWidget(sep); live.addSpacing(10)
        self._lbl50   = QLabel("50 Hz: —")
        self._lbl50.setStyleSheet("color:#f48fb1;font-size:13px;font-weight:bold;min-width:175px;")
        self._lbl60   = QLabel("60 Hz: —")
        self._lbl60.setStyleSheet("color:#90caf9;font-size:13px;font-weight:bold;min-width:175px;")
        self._lbl_dom = QLabel("Dominant: —")
        self._lbl_dom.setStyleSheet("color:#a5d6a7;font-size:12px;font-weight:bold;")
        live.addWidget(self._lbl50); live.addWidget(self._lbl60)
        live.addSpacing(20); live.addWidget(self._lbl_dom); live.addStretch()
        vbox.addLayout(live)

        # ── Tabs ──
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_overview_tab(), "Overview")
        vbox.addWidget(self._tabs, stretch=1)

        self.statusBar().showMessage("Ready — enter a YouTube URL and press Start.")

    def _build_overview_tab(self) -> QWidget:
        w  = QWidget()
        v  = QVBoxLayout(w); v.setContentsMargins(0, 4, 0, 0)
        sp = QSplitter(Qt.Vertical)

        pw_wave = pg.PlotWidget(title="<b>Waveform</b>")
        pw_wave.setLabel("left", "Amplitude"); pw_wave.setLabel("bottom", "Window (s)")
        pw_wave.showGrid(x=True, y=False, alpha=0.15)
        pw_wave.setYRange(-1.0, 1.0); pw_wave.setMouseEnabled(x=False, y=False)
        pw_wave.addLine(y=0, pen=pg.mkPen("#333333", width=1))
        self._curve_wave = pw_wave.plot(pen=pg.mkPen("#80cbc4", width=1))
        sp.addWidget(pw_wave)

        pw50 = self._make_enf_plot("50 Hz  (EU/UK/AU)", 50.0)
        self._ov_curve50 = pw50.plot(pen=pg.mkPen("#f48fb1", width=2))
        self._ov_pw50    = pw50
        sp.addWidget(pw50)

        pw60 = self._make_enf_plot("60 Hz  (US/CA/MX)", 60.0)
        self._ov_curve60 = pw60.plot(pen=pg.mkPen("#90caf9", width=2))
        self._ov_pw60    = pw60
        sp.addWidget(pw60)

        sp.setSizes([120, 280, 280])
        v.addWidget(sp)
        return w

    def _build_speaker_widget(self, tab: SpeakerTab) -> QWidget:
        w  = QWidget()
        v  = QVBoxLayout(w); v.setContentsMargins(0, 4, 0, 0)
        sp = QSplitter(Qt.Vertical)

        pw50 = self._make_enf_plot(f"50 Hz — {self._spk_label(tab.speaker_id)}", 50.0)
        tab.curve50 = pw50.plot(pen=pg.mkPen(tab.color, width=2))
        tab.pw50    = pw50; sp.addWidget(pw50)

        pw60 = self._make_enf_plot(f"60 Hz — {self._spk_label(tab.speaker_id)}", 60.0)
        tab.curve60 = pw60.plot(pen=pg.mkPen(tab.color, width=2))
        tab.pw60    = pw60; sp.addWidget(pw60)

        v.addWidget(sp); tab.widget = w
        return w

    @staticmethod
    def _make_enf_plot(title: str, nominal: float) -> pg.PlotWidget:
        pw = pg.PlotWidget(title=f"<b>{title}</b>")
        pw.setLabel("left", "Frequency (Hz)")
        pw.setLabel("bottom", "Elapsed (s)")
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setYRange(nominal - 0.6, nominal + 0.6)
        pw.addLine(y=nominal,       pen=pg.mkPen("#555555", width=1, style=Qt.DashLine))
        pw.addLine(y=nominal - 0.2, pen=pg.mkPen("#2a2a40", width=1, style=Qt.DotLine))
        pw.addLine(y=nominal + 0.2, pen=pg.mkPen("#2a2a40", width=1, style=Qt.DotLine))
        return pw

    @staticmethod
    def _spk_label(spk_id: int) -> str:
        return f"Speaker {spk_id + 1}"

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _start(self):
        url = self._url_input.text().strip()
        if not url:
            self.statusBar().showMessage("Please enter a YouTube URL."); return

        for dq in (self._ov_times, self._ov_f50, self._ov_f60,
                   self._ov_c50, self._ov_c60, self._ov_spk):
            dq.clear()
        self._wave_buf[:] = 0.0
        self._curve_wave.setData(self._wave_t, self._wave_buf)
        self._ov_curve50.setData([], []); self._ov_curve60.setData([], [])
        self._clear_bg_regions()

        while self._tabs.count() > 1:
            self._tabs.removeTab(1)
        self._speaker_tabs.clear()

        self._lbl50.setText("50 Hz: —"); self._lbl60.setText("60 Hz: —")
        self._lbl_dom.setText("Dominant: —")
        self._start_btn.setEnabled(False); self._stop_btn.setEnabled(True)

        self._worker.start_analysis(
            url,
            play_audio=self._play_chk.isChecked(),
            n_speakers=self._spk_spin.value(),
        )
        self._plot_timer.start()

    def _stop(self):
        self._worker.stop()
        self._plot_timer.stop()
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped.")

    def _on_audio_chunk(self, _elapsed: float, raw: bytes):
        chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(chunk)
        self._wave_buf = np.roll(self._wave_buf, -n)
        self._wave_buf[-n:] = chunk
        self._curve_wave.setData(self._wave_t, self._wave_buf)

    def _on_freq(self, t, f50, c50, f60, c60, spk):
        self._ov_times.append(t); self._ov_f50.append(f50); self._ov_c50.append(c50)
        self._ov_f60.append(f60); self._ov_c60.append(c60); self._ov_spk.append(spk)

        if spk >= 0:
            if spk not in self._speaker_tabs:
                self._add_speaker_tab(spk)
            self._speaker_tabs[spk].append(t, f50, c50, f60, c60)

        self._lbl50.setText(f"50 Hz:  {f50:.4f} Hz  (snr {c50:.2e})")
        self._lbl60.setText(f"60 Hz:  {f60:.4f} Hz  (snr {c60:.2e})")

        if len(self._ov_c50) >= 5:
            a50 = np.mean(list(self._ov_c50)[-10:])
            a60 = np.mean(list(self._ov_c60)[-10:])
            r   = a50 / (a60 + 1e-9)
            dom = "50 Hz (EU/UK)" if r > 2.0 else "60 Hz (US/CA)" if r < 0.5 else "Uncertain"
            spk_str = f"  ·  {self._spk_label(spk)}" if spk >= 0 else ""
            self._lbl_dom.setText(f"Dominant: {dom}{spk_str}")

        self._highlight_speaker_tab(spk)

    def _on_model_info(self, info: str):
        self.statusBar().showMessage(f"Models: {info}")

    def _on_status(self, msg): self.statusBar().showMessage(msg)
    def _on_error(self, msg):  self.statusBar().showMessage(f"Error: {msg}"); self._stop()
    def _on_finished(self):
        self._plot_timer.stop()
        self._start_btn.setEnabled(True); self._stop_btn.setEnabled(False)

    # ── Speaker tab management ────────────────────────────────────────────────

    def _add_speaker_tab(self, spk_id: int):
        color  = SPEAKER_COLORS[spk_id % len(SPEAKER_COLORS)]
        tab    = SpeakerTab(spk_id, color)
        self._speaker_tabs[spk_id] = tab
        widget = self._build_speaker_widget(tab)
        self._tabs.addTab(widget, self._spk_label(spk_id))
        idx = self._tabs.count() - 1
        self._tabs.tabBar().setTabTextColor(idx, QColor(color))

    def _highlight_speaker_tab(self, active_spk: int):
        bar = self._tabs.tabBar()
        for spk_id, tab in self._speaker_tabs.items():
            idx = self._tabs.indexOf(tab.widget)
            if idx < 0: continue
            base = self._spk_label(spk_id)
            if spk_id == active_spk:
                bar.setTabText(idx, f"● {base}")
                bar.setTabTextColor(idx, QColor(tab.color))
            else:
                bar.setTabText(idx, f"  {base}")
                bar.setTabTextColor(idx, QColor("#666666"))

    # ── Background speaker regions on overview plots ──────────────────────────

    def _clear_bg_regions(self):
        for item in self._bg_regions_50:
            try: self._ov_pw50.removeItem(item)
            except Exception: pass
        for item in self._bg_regions_60:
            try: self._ov_pw60.removeItem(item)
            except Exception: pass
        self._bg_regions_50.clear()
        self._bg_regions_60.clear()

    def _rebuild_bg_regions(self):
        """Draw one LinearRegionItem per contiguous same-speaker run."""
        self._clear_bg_regions()

        times = list(self._ov_times)
        spks  = list(self._ov_spk)
        if not times or not any(s >= 0 for s in spks):
            return

        # Group consecutive same-speaker time points into segments
        segments: list[tuple[float, float, int]] = []
        seg_start = times[0]; seg_spk = spks[0]
        for i in range(1, len(times)):
            if spks[i] != seg_spk:
                segments.append((seg_start, times[i - 1] + HOP_SEC, seg_spk))
                seg_start = times[i]; seg_spk = spks[i]
        segments.append((seg_start, times[-1] + HOP_SEC, seg_spk))

        for t0, t1, spk in segments:
            if spk < 0:
                continue
            hex_color = SPEAKER_COLORS[spk % len(SPEAKER_COLORS)]
            c  = QColor(hex_color); c.setAlpha(SPEAKER_BG_ALPHA)
            br = pg.mkBrush(c)
            for pw, lst in ((self._ov_pw50, self._bg_regions_50),
                            (self._ov_pw60, self._bg_regions_60)):
                region = pg.LinearRegionItem(
                    [t0, t1], movable=False,
                    brush=br, pen=pg.mkPen(None),
                )
                region.setZValue(-10)   # behind the frequency curve
                pw.addItem(region)
                lst.append(region)

    # ── Plot refresh ──────────────────────────────────────────────────────────

    def _refresh_plots(self):
        if not self._ov_times:
            return
        mode  = self._mode_group.checkedButton().text()
        times = np.array(self._ov_times)
        t_max = float(times[-1])
        t_min = max(0.0, t_max - PLOT_WINDOW)

        if mode != "60 Hz":
            self._ov_curve50.setData(times, np.array(self._ov_f50))
            self._ov_pw50.setXRange(t_min, t_max, padding=0)
        if mode != "50 Hz":
            self._ov_curve60.setData(times, np.array(self._ov_f60))
            self._ov_pw60.setXRange(t_min, t_max, padding=0)

        # Rebuild background shading (cheap for typical segment counts)
        self._rebuild_bg_regions()

        # Speaker tabs
        for tab in self._speaker_tabs.values():
            if not tab.times: continue
            st = np.array(tab.times)
            sm = max(0.0, float(st[-1]) - PLOT_WINDOW); sx = float(st[-1])
            if mode != "60 Hz":
                tab.curve50.setData(st, np.array(tab.f50))
                tab.pw50.setXRange(sm, sx, padding=0)
            if mode != "50 Hz":
                tab.curve60.setData(st, np.array(tab.f60))
                tab.pw60.setXRange(sm, sx, padding=0)

    def closeEvent(self, event):
        self._worker.stop(); event.accept()


# ─────────────────────────────────────────────────────────────────────────────

def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(26,  26,  26))
    p.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.Base,            QColor(18,  18,  18))
    p.setColor(QPalette.AlternateBase,   QColor(35,  35,  35))
    p.setColor(QPalette.ToolTipBase,     QColor(50,  50,  50))
    p.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
    p.setColor(QPalette.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.Button,          QColor(45,  45,  45))
    p.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Highlight,       QColor(42,  130, 218))
    p.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    win = HumBugWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
