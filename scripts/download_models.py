"""Download the official MediaPipe task bundles during local setup.

This script is intentionally separate from the application.  Once the files are
present, Gestures never accesses the network.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def download(name: str, url: str) -> None:
    destination = MODELS_DIR / name
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {name}...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size < 100_000:
            raise RuntimeError("the downloaded file is unexpectedly small")
        temporary.replace(destination)
        print(f"  saved to {destination}")
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for name, url in MODEL_URLS.items():
            if (MODELS_DIR / name).is_file():
                print(f"{name} already exists; leaving it unchanged.")
            else:
                download(name, url)
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1
    print("Models are ready. The application can now run without network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
