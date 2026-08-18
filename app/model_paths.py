"""Resolve the local MediaPipe task model files.

The application never downloads models at runtime.  Put the two .task files in
``models/`` before starting the application, or run ``scripts/download_models.py``
once during setup.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    """Paths to the two MediaPipe task bundles used by the application."""

    hand: Path
    face: Path

    def missing(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.hand, self.face) if not path.is_file())


def application_root() -> Path:
    """Return the source root or PyInstaller extraction root."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parents[1]


def get_model_paths() -> ModelPaths:
    """Return the expected local model locations."""

    models_dir = application_root() / "models"
    return ModelPaths(
        hand=models_dir / "hand_landmarker.task",
        face=models_dir / "face_landmarker.task",
    )


def model_install_message(paths: ModelPaths) -> str:
    """Build a user-facing message for missing model files."""

    missing = "\n".join(f"  - {path}" for path in paths.missing())
    return (
        "MediaPipe model files are missing. Download them once with "
        "scripts\\download_models.py, then restart Gestures.\n\n"
        f"Missing files:\n{missing}"
    )
