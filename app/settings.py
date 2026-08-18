"""Application settings and local persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SHORTCUT = "Alt + Tab"


class SettingsError(ValueError):
    """Raised when settings supplied by the UI are not usable."""


@dataclass(frozen=True)
class AppSettings:
    """All user-configurable application values.

    ``touch_threshold`` is measured as fingertip-to-nose distance divided by
    the current face width in normalized image coordinates.  It therefore
    adapts when the user moves closer to or farther from the webcam.
    """

    camera_index: int = 0
    detection_enabled: bool = True
    touch_threshold: float = 0.10
    touch_duration_ms: int = 0
    cooldown_ms: int = 500
    shortcut: str = DEFAULT_SHORTCUT
    preview_visible: bool = True
    debug_mode: bool = True
    start_with_windows: bool = False
    smoothing_frames: int = 5
    calibrated_face_width: float | None = None

    def validate(self) -> "AppSettings":
        """Validate values and return this immutable settings object."""

        if self.camera_index < 0:
            raise SettingsError("Camera index must be zero or greater.")
        if not 0.01 <= self.touch_threshold <= 0.5:
            raise SettingsError("Touch threshold must be between 0.01 and 0.50.")
        if not 0 <= self.touch_duration_ms <= 5000:
            raise SettingsError("Touch duration must be between 0 and 5000 ms.")
        if not 50 <= self.cooldown_ms <= 10000:
            raise SettingsError("Cooldown must be between 50 and 10000 ms.")
        if not self.shortcut.strip():
            raise SettingsError("Keyboard shortcut cannot be empty.")
        if not 1 <= self.smoothing_frames <= 15:
            raise SettingsError("Smoothing frames must be between 1 and 15.")
        if self.calibrated_face_width is not None and self.calibrated_face_width <= 0:
            raise SettingsError("Calibrated face width must be positive.")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        """Create settings from a JSON object with strict type conversion."""

        calibrated_width = data.get("calibrated_face_width")
        return cls(
            camera_index=int(data.get("camera_index", cls.camera_index)),
            detection_enabled=_as_bool(data.get("detection_enabled", cls.detection_enabled)),
            touch_threshold=float(data.get("touch_threshold", cls.touch_threshold)),
            touch_duration_ms=int(data.get("touch_duration_ms", cls.touch_duration_ms)),
            cooldown_ms=int(data.get("cooldown_ms", cls.cooldown_ms)),
            shortcut=str(data.get("shortcut", cls.shortcut)),
            preview_visible=_as_bool(data.get("preview_visible", cls.preview_visible)),
            debug_mode=_as_bool(data.get("debug_mode", cls.debug_mode)),
            start_with_windows=_as_bool(data.get("start_with_windows", cls.start_with_windows)),
            smoothing_frames=int(data.get("smoothing_frames", cls.smoothing_frames)),
            calibrated_face_width=(
                None if calibrated_width in (None, "") else float(calibrated_width)
            ),
        ).validate()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def settings_directory() -> Path:
    """Return a per-user directory; no settings are written into the repo."""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Gestures"
    return Path.home() / ".gestures"


class SettingsStore:
    """Read and write the small local settings JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings_directory() / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.is_file():
            return AppSettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise SettingsError("Settings file must contain a JSON object.")
            return AppSettings.from_dict(data)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # A damaged settings file should never prevent the app from opening.
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(settings.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
