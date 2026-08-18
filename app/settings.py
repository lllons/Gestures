"""Application settings and local persistence."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SHORTCUT = "Alt + Tab"
DEFAULT_PINCH_SHORTCUT = DEFAULT_SHORTCUT


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
    cooldown_ms: int = 0
    shortcut: str = DEFAULT_SHORTCUT
    pinch_shortcut: str = DEFAULT_PINCH_SHORTCUT
    air_mouse_enabled: bool = False
    preview_visible: bool = True
    debug_mode: bool = True
    start_with_windows: bool = False
    smoothing_frames: int = 5
    calibrated_face_width: float | None = None

    # Two-hand Blender navigation. These values are deliberately kept in the
    # same local settings file so the feature remains usable offline.
    navigation_enabled: bool = False
    navigation_mode: str = "Viewport"
    navigation_control_mode: str = "FULL 3D"
    navigation_activation_gesture: str = "Two open hands"
    navigation_deactivation_gesture: str = "Hands removed"
    navigation_activation_hold_ms: int = 700
    navigation_orbit_sensitivity: float = 4.0
    navigation_pan_sensitivity: float = 1.25
    navigation_zoom_sensitivity: float = 6.0
    navigation_roll_sensitivity: float = 2.5
    navigation_smoothing_frames: int = 5
    navigation_dead_zone: float = 0.004
    navigation_max_speed: float = 1.25
    navigation_min_confidence: float = 0.50
    navigation_invert_x: bool = False
    navigation_invert_y: bool = False
    navigation_invert_zoom: bool = False
    navigation_roll_enabled: bool = False
    blender_host: str = "127.0.0.1"
    blender_port: int = 8765
    blender_reply_port: int = 8766

    def validate(self) -> "AppSettings":
        """Validate values and return this immutable settings object."""

        if self.camera_index < 0:
            raise SettingsError("Camera index must be zero or greater.")
        if not all(
            math.isfinite(value)
            for value in (
                self.touch_threshold,
                self.navigation_orbit_sensitivity,
                self.navigation_pan_sensitivity,
                self.navigation_zoom_sensitivity,
                self.navigation_roll_sensitivity,
                self.navigation_dead_zone,
                self.navigation_max_speed,
            )
        ):
            raise SettingsError("Sensitivity values must be finite numbers.")
        if not 0.01 <= self.touch_threshold <= 0.5:
            raise SettingsError("Touch threshold must be between 0.01 and 0.50.")
        if not 0 <= self.touch_duration_ms <= 5000:
            raise SettingsError("Touch duration must be between 0 and 5000 ms.")
        if not 0 <= self.cooldown_ms <= 10000:
            raise SettingsError("Cooldown must be between 0 and 10000 ms.")
        if not self.shortcut.strip():
            raise SettingsError("Keyboard shortcut cannot be empty.")
        if not self.pinch_shortcut.strip():
            raise SettingsError("Pinch shortcut cannot be empty.")
        if not 1 <= self.smoothing_frames <= 15:
            raise SettingsError("Smoothing frames must be between 1 and 15.")
        if self.calibrated_face_width is not None and self.calibrated_face_width <= 0:
            raise SettingsError("Calibrated face width must be positive.")
        if self.navigation_mode not in {"Viewport", "Camera"}:
            raise SettingsError("Navigation mode must be Viewport or Camera.")
        if self.navigation_control_mode not in {"ORBIT", "PAN", "ZOOM", "FULL 3D"}:
            raise SettingsError("Navigation control mode is not supported.")
        if self.navigation_activation_gesture not in {"Two open hands", "Two closed hands"}:
            raise SettingsError("Navigation activation gesture is not supported.")
        if self.navigation_deactivation_gesture not in {
            "Hands removed",
            "Two closed hands",
            "Two open hands",
        }:
            raise SettingsError("Navigation deactivation gesture is not supported.")
        if not 250 <= self.navigation_activation_hold_ms <= 3000:
            raise SettingsError("Navigation activation hold must be between 250 and 3000 ms.")
        if not 0.1 <= self.navigation_orbit_sensitivity <= 20:
            raise SettingsError("Orbit sensitivity must be between 0.1 and 20.")
        if not 0.1 <= self.navigation_pan_sensitivity <= 10:
            raise SettingsError("Pan sensitivity must be between 0.1 and 10.")
        if not 0.1 <= self.navigation_zoom_sensitivity <= 20:
            raise SettingsError("Zoom sensitivity must be between 0.1 and 20.")
        if not 0.1 <= self.navigation_roll_sensitivity <= 20:
            raise SettingsError("Roll sensitivity must be between 0.1 and 20.")
        if not 1 <= self.navigation_smoothing_frames <= 20:
            raise SettingsError("Navigation smoothing must be between 1 and 20 frames.")
        if not 0 <= self.navigation_dead_zone <= 0.10:
            raise SettingsError("Navigation dead zone must be between 0 and 0.10.")
        if not 0.1 <= self.navigation_max_speed <= 5:
            raise SettingsError("Navigation max speed must be between 0.1 and 5.")
        if not math.isfinite(self.navigation_min_confidence) or not 0.1 <= self.navigation_min_confidence <= 1.0:
            raise SettingsError("Navigation confidence threshold must be between 0.1 and 1.0.")
        if not self.blender_host.strip():
            raise SettingsError("Blender host cannot be empty.")
        if self.blender_host.strip().casefold() not in {"127.0.0.1", "localhost"}:
            raise SettingsError("Blender connection must stay on localhost for offline safety.")
        if not 1 <= self.blender_port <= 65535:
            raise SettingsError("Blender port must be between 1 and 65535.")
        if not 1 <= self.blender_reply_port <= 65535:
            raise SettingsError("Blender reply port must be between 1 and 65535.")
        if self.blender_port == self.blender_reply_port:
            raise SettingsError("Blender command and reply ports must be different.")
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
            pinch_shortcut=str(data.get("pinch_shortcut", cls.pinch_shortcut)),
            air_mouse_enabled=_as_bool(data.get("air_mouse_enabled", cls.air_mouse_enabled)),
            preview_visible=_as_bool(data.get("preview_visible", cls.preview_visible)),
            debug_mode=_as_bool(data.get("debug_mode", cls.debug_mode)),
            start_with_windows=_as_bool(data.get("start_with_windows", cls.start_with_windows)),
            smoothing_frames=int(data.get("smoothing_frames", cls.smoothing_frames)),
            calibrated_face_width=(
                None if calibrated_width in (None, "") else float(calibrated_width)
            ),
            navigation_enabled=_as_bool(data.get("navigation_enabled", cls.navigation_enabled)),
            navigation_mode=str(data.get("navigation_mode", cls.navigation_mode)),
            navigation_control_mode=str(
                data.get("navigation_control_mode", cls.navigation_control_mode)
            ),
            navigation_activation_gesture=str(
                data.get("navigation_activation_gesture", cls.navigation_activation_gesture)
            ),
            navigation_deactivation_gesture=str(
                data.get("navigation_deactivation_gesture", cls.navigation_deactivation_gesture)
            ),
            navigation_activation_hold_ms=int(
                data.get("navigation_activation_hold_ms", cls.navigation_activation_hold_ms)
            ),
            navigation_orbit_sensitivity=float(
                data.get("navigation_orbit_sensitivity", cls.navigation_orbit_sensitivity)
            ),
            navigation_pan_sensitivity=float(
                data.get("navigation_pan_sensitivity", cls.navigation_pan_sensitivity)
            ),
            navigation_zoom_sensitivity=float(
                data.get("navigation_zoom_sensitivity", cls.navigation_zoom_sensitivity)
            ),
            navigation_roll_sensitivity=float(
                data.get("navigation_roll_sensitivity", cls.navigation_roll_sensitivity)
            ),
            navigation_smoothing_frames=int(
                data.get("navigation_smoothing_frames", cls.navigation_smoothing_frames)
            ),
            navigation_dead_zone=float(
                data.get("navigation_dead_zone", cls.navigation_dead_zone)
            ),
            navigation_max_speed=float(
                data.get("navigation_max_speed", cls.navigation_max_speed)
            ),
            navigation_min_confidence=float(
                data.get("navigation_min_confidence", cls.navigation_min_confidence)
            ),
            navigation_invert_x=_as_bool(
                data.get("navigation_invert_x", cls.navigation_invert_x)
            ),
            navigation_invert_y=_as_bool(
                data.get("navigation_invert_y", cls.navigation_invert_y)
            ),
            navigation_invert_zoom=_as_bool(
                data.get("navigation_invert_zoom", cls.navigation_invert_zoom)
            ),
            navigation_roll_enabled=_as_bool(
                data.get("navigation_roll_enabled", cls.navigation_roll_enabled)
            ),
            blender_host=str(data.get("blender_host", cls.blender_host)),
            blender_port=int(data.get("blender_port", cls.blender_port)),
            blender_reply_port=int(data.get("blender_reply_port", cls.blender_reply_port)),
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
