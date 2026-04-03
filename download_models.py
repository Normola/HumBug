#!/usr/bin/env python3
"""
Download pre-trained ONNX models for HumBug speaker diarization.

Models:
  silero_vad.onnx        ~1.8 MB   Silero VAD  (speech activity detection)
  wespeaker_resnet34.onnx ~26 MB   WeSpeaker ResNet34 speaker embeddings
                                    trained on VoxCeleb1+2
"""

import urllib.request
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

MODELS = {
    "silero_vad.onnx": (
        "https://raw.githubusercontent.com/snakers4/silero-vad/master"
        "/src/silero_vad/data/silero_vad.onnx"
    ),
    "wespeaker_resnet34.onnx": (
        "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main"
        "/wespeaker_en_voxceleb_resnet34_LM.onnx?download=true"
    ),
}


def _progress(label: str):
    downloaded = [0]

    def hook(count, block_size, total_size):
        downloaded[0] = count * block_size
        if total_size > 0:
            pct = min(100, downloaded[0] * 100 // total_size)
            mb  = downloaded[0] / 1_048_576
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  {label}: [{bar}] {pct:3d}%  {mb:.1f} MB", end="", flush=True)
        else:
            mb = downloaded[0] / 1_048_576
            print(f"\r  {label}: {mb:.1f} MB downloaded", end="", flush=True)

    return hook


def download_all():
    MODELS_DIR.mkdir(exist_ok=True)
    ok = True

    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  {filename}: already present ({dest.stat().st_size / 1_048_576:.1f} MB) — skipping")
            continue

        print(f"\nDownloading {filename} …")
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_progress(filename))
            tmp.rename(dest)
            print(f"\n  ✓ saved to {dest}")
        except Exception as exc:
            print(f"\n  ✗ failed: {exc}")
            if tmp.exists():
                tmp.unlink()
            ok = False

    return ok


if __name__ == "__main__":
    print("HumBug model downloader")
    print(f"Destination: {MODELS_DIR}\n")
    success = download_all()
    print("\nDone." if success else "\nSome downloads failed — check your internet connection.")
    sys.exit(0 if success else 1)
