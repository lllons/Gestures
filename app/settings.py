"""Application settings and local persistence."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .input_profile import InputProfile, default_profile_data


DEFAULT_SHORTCUT = "Alt + Tab"
DEFAULT_PINCH_SHORTCUT = DEFAULT_SHORTCUT


class SettingsError(ValueError):
    """Raised when settings supplied by the UI are not usable."""


@dataclass(frozen=True)
class AppSettings:
    """All user-configurable application values.

    ``touch_threshold`` is measured as fingertip-to-nose distance divided by
    the current face width in normalized image coordinates. It therefore adapts
    when the user moves closer to or farther from the webcam.
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

    # Universal two-hand navigation settings. Profiles contain only generic
    # mouse/modifier mappings; there is no target-application integration.
    navigation_enabled: bool = False
    navigation_profile: str = "Generic 3D"
    navigation_profiles: dict[str, dict[str, Any]] = field(
        default_factory=default_profile_data
    )
    navigation_control_mode: str = "FULL 3D"
    navigation_activation_gesture: str = "Two open hands"
    navigation_deactivation_gesture: str = "Hands removed"
    navigation_pan_gesture: str = "Two closed hands"
    navigation_activation_hold_ms: int = 700
    navigation_orbit_sensitivity: float = 4.0
    navigation_pan_sensitivity: float = 1.25
    navigation_zoom_sensitivity: float = 6.0
    navigation_roll_sensitivity: float = 2.5
    navigation_smoothing_frames: int = 5
    navigation_dead_zone: float = 0.004
    navigation_max_speed: float = 1.25
    navigation_acceleration: float = 0.5
    navigation_mouse_scale: float = 280.0
    navigation_zoom_wheel_scale: float = 2.0
    navigation_min_confidence: float = 0.50
    navigation_invert_x: bool = False
    navigation_invert_y: bool = False
    navigation_invert_zoom: bool = False
    navigation_roll_enabled: bool = False

    # Low-latency navigation signal processing. Values are normalized hand
    # coordinates or normalized units/second, not screen pixels.
    navigation_smoothing: float = 0.45
    navigation_adaptive_smoothing: bool = True
    navigation_orbit_smoothing: float = 0.45
    navigation_pan_smoothing: float = 0.45
    navigation_zoom_smoothing: float = 0.55
    navigation_orbit_dead_zone: float = 0.004
    navigation_pan_dead_zone: float = 0.004
    navigation_zoom_dead_zone: float = 0.003
    navigation_orbit_acceleration: float = 0.55
    navigation_pan_acceleration: float = 0.45
    navigation_zoom_acceleration: float = 0.65
    navigation_orbit_max_speed: float = 1.35
    navigation_pan_max_speed: float = 1.0
    navigation_zoom_max_speed: float = 0.9
    navigation_motion_start_threshold: float = 0.16
    navigation_motion_stop_threshold: float = 0.08
    navigation_zoom_start_threshold: float = 0.12
    navigation_zoom_stop_threshold: float = 0.06
    navigation_hand_loss_grace_frames: int = 2
    navigation_outlier_threshold: float = 0.20
    navigation_one_euro_enabled: bool = True
    navigation_one_euro_min_cutoff: float = 1.0
    navigation_one_euro_beta: float = 0.08
    navigation_one_euro_derivative_cutoff: float = 1.0
    navigation_precision_modifier: str = "alt"
    navigation_fast_modifier: str = "ctrl"
    navigation_precision_scale: float = 0.35
    navigation_fast_scale: float = 1.75
    navigation_global_hotkey: str = "F8"
    navigation_emergency_hotkey: str = "F9"

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
                self.navigation_smoothing,
                self.navigation_orbit_smoothing,
                self.navigation_pan_smoothing,
                self.navigation_zoom_smoothing,
                self.navigation_orbit_dead_zone,
                self.navigation_pan_dead_zone,
                self.navigation_zoom_dead_zone,
                self.navigation_orbit_acceleration,
                self.navigation_pan_acceleration,
                self.navigation_zoom_acceleration,
                self.navigation_max_speed,
                self.navigation_orbit_max_speed,
                self.navigation_pan_max_speed,
                self.navigation_zoom_max_speed,
                self.navigation_motion_start_threshold,
                self.navigation_motion_stop_threshold,
                self.navigation_zoom_start_threshold,
                self.navigation_zoom_stop_threshold,
                self.navigation_outlier_threshold,
                self.navigation_one_euro_min_cutoff,
                self.navigation_one_euro_beta,
                self.navigation_one_euro_derivative_cutoff,
                self.navigation_mouse_scale,
                self.navigation_zoom_wheel_scale,
                self.navigation_precision_scale,
                self.navigation_fast_scale,
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
        if self.navigation_pan_gesture not in {
            "Disabled",
            "Two closed hands",
            "Two open hands",
        }:
            raise SettingsError("Navigation pan gesture is not supported.")
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
        if not 0 <= self.navigation_smoothing <= 1:
            raise SettingsError("Navigation smoothing must be between 0 and 1.")
        if not all(
            0 <= value <= 1
            for value in (
                self.navigation_orbit_smoothing,
                self.navigation_pan_smoothing,
                self.navigation_zoom_smoothing,
            )
        ):
            raise SettingsError("Channel smoothing must be between 0 and 1.")
        if not 0 <= self.navigation_orbit_dead_zone <= 0.10:
            raise SettingsError("Orbit dead zone must be between 0 and 0.10.")
        if not 0 <= self.navigation_pan_dead_zone <= 0.10:
            raise SettingsError("Pan dead zone must be between 0 and 0.10.")
        if not 0 <= self.navigation_zoom_dead_zone <= 0.10:
            raise SettingsError("Zoom dead zone must be between 0 and 0.10.")
        if not 0 <= self.navigation_acceleration <= 3:
            raise SettingsError("Navigation acceleration must be between 0 and 3.")
        if not all(
            0 <= value <= 3
            for value in (
                self.navigation_orbit_acceleration,
                self.navigation_pan_acceleration,
                self.navigation_zoom_acceleration,
            )
        ):
            raise SettingsError("Channel acceleration must be between 0 and 3.")
        if not 0.1 <= self.navigation_max_speed <= 5:
            raise SettingsError("Navigation max speed must be between 0.1 and 5.")
        if not all(
            0.1 <= value <= 5
            for value in (
                self.navigation_orbit_max_speed,
                self.navigation_pan_max_speed,
                self.navigation_zoom_max_speed,
            )
        ):
            raise SettingsError("Channel maximum speed must be between 0.1 and 5.")
        if not 0 <= self.navigation_motion_stop_threshold <= self.navigation_motion_start_threshold <= 5:
            raise SettingsError("Motion stop threshold must not exceed motion start threshold.")
        if not 0 <= self.navigation_zoom_stop_threshold <= self.navigation_zoom_start_threshold <= 5:
            raise SettingsError("Zoom stop threshold must not exceed zoom start threshold.")
        if not 0 <= self.navigation_hand_loss_grace_frames <= 5:
            raise SettingsError("Hand-loss grace must be between 0 and 5 frames.")
        if not 0.02 <= self.navigation_outlier_threshold <= 0.5:
            raise SettingsError("Outlier threshold must be between 0.02 and 0.50.")
        if not 0.05 <= self.navigation_one_euro_min_cutoff <= 20:
            raise SettingsError("One Euro minimum cutoff must be between 0.05 and 20.")
        if not 0 <= self.navigation_one_euro_beta <= 20:
            raise SettingsError("One Euro beta must be between 0 and 20.")
        if not 0.05 <= self.navigation_one_euro_derivative_cutoff <= 20:
            raise SettingsError("One Euro derivative cutoff must be between 0.05 and 20.")
        valid_modifiers = {"none", "shift", "ctrl", "alt", "cmd"}
        if self.navigation_precision_modifier not in valid_modifiers:
            raise SettingsError("Precision modifier is not supported.")
        if self.navigation_fast_modifier not in valid_modifiers:
            raise SettingsError("Fast modifier is not supported.")
        if (
            self.navigation_precision_modifier != "none"
            and self.navigation_precision_modifier == self.navigation_fast_modifier
        ):
            raise SettingsError("Precision and fast modifiers must be different.")
        if not 0.05 <= self.navigation_precision_scale <= 1:
            raise SettingsError("Precision scale must be between 0.05 and 1.")
        if not 1 <= self.navigation_fast_scale <= 3:
            raise SettingsError("Fast scale must be between 1 and 3.")
        if not 10 <= self.navigation_mouse_scale <= 2000:
            raise SettingsError("Mouse sensitivity must be between 10 and 2000.")
        if not 0.1 <= self.navigation_zoom_wheel_scale <= 20:
            raise SettingsError("Zoom wheel scale must be between 0.1 and 20.")
        if not math.isfinite(self.navigation_min_confidence) or not 0.1 <= self.navigation_min_confidence <= 1.0:
            raise SettingsError("Navigation confidence threshold must be between 0.1 and 1.0.")
        if not self.navigation_global_hotkey.strip() or not self.navigation_emergency_hotkey.strip():
            raise SettingsError("Navigation hotkeys cannot be empty.")
        if self.navigation_global_hotkey.casefold() == self.navigation_emergency_hotkey.casefold():
            raise SettingsError("Global and emergency navigation hotkeys must be different.")

        if not isinstance(self.navigation_profiles, dict) or not self.navigation_profiles:
            raise SettingsError("At least one input profile is required.")
        if self.navigation_profile not in self.navigation_profiles:
            raise SettingsError(f"Input profile {self.navigation_profile!r} was not found.")
        for profile_name, profile_data in self.navigation_profiles.items():
            try:
                InputProfile.from_dict(str(profile_name), profile_data)
            except (TypeError, ValueError) as exc:
                raise SettingsError(f"Invalid input profile {profile_name!r}: {exc}") from exc
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        """Create settings from a JSON object with backward-compatible defaults."""

        calibrated_width = data.get("calibrated_face_width")
        profiles = default_profile_data()
        stored_profiles = data.get("navigation_profiles")
        if isinstance(stored_profiles, dict):
            for profile_name, profile_data in stored_profiles.items():
                if isinstance(profile_data, dict):
                    profiles[str(profile_name)] = dict(profile_data)
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
            navigation_profile=str(data.get("navigation_profile", cls.navigation_profile)),
            navigation_profiles=profiles,
            navigation_control_mode=str(
                data.get("navigation_control_mode", cls.navigation_control_mode)
            ),
            navigation_activation_gesture=str(
                data.get("navigation_activation_gesture", cls.navigation_activation_gesture)
            ),
            navigation_deactivation_gesture=str(
                data.get("navigation_deactivation_gesture", cls.navigation_deactivation_gesture)
            ),
            navigation_pan_gesture=str(
                data.get("navigation_pan_gesture", cls.navigation_pan_gesture)
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
            navigation_acceleration=float(
                data.get("navigation_acceleration", cls.navigation_acceleration)
            ),
            navigation_mouse_scale=float(
                data.get("navigation_mouse_scale", cls.navigation_mouse_scale)
            ),
            navigation_zoom_wheel_scale=float(
                data.get("navigation_zoom_wheel_scale", cls.navigation_zoom_wheel_scale)
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
            navigation_smoothing=float(
                data.get("navigation_smoothing", cls.navigation_smoothing)
            ),
            navigation_adaptive_smoothing=_as_bool(
                data.get("navigation_adaptive_smoothing", cls.navigation_adaptive_smoothing)
            ),
            navigation_orbit_smoothing=float(
                data.get("navigation_orbit_smoothing", cls.navigation_orbit_smoothing)
            ),
            navigation_pan_smoothing=float(
                data.get("navigation_pan_smoothing", cls.navigation_pan_smoothing)
            ),
            navigation_zoom_smoothing=float(
                data.get("navigation_zoom_smoothing", cls.navigation_zoom_smoothing)
            ),
            navigation_orbit_dead_zone=float(
                data.get("navigation_orbit_dead_zone", cls.navigation_orbit_dead_zone)
            ),
            navigation_pan_dead_zone=float(
                data.get("navigation_pan_dead_zone", cls.navigation_pan_dead_zone)
            ),
            navigation_zoom_dead_zone=float(
                data.get("navigation_zoom_dead_zone", cls.navigation_zoom_dead_zone)
            ),
            navigation_orbit_acceleration=float(
                data.get("navigation_orbit_acceleration", cls.navigation_orbit_acceleration)
            ),
            navigation_pan_acceleration=float(
                data.get("navigation_pan_acceleration", cls.navigation_pan_acceleration)
            ),
            navigation_zoom_acceleration=float(
                data.get("navigation_zoom_acceleration", cls.navigation_zoom_acceleration)
            ),
            navigation_orbit_max_speed=float(
                data.get("navigation_orbit_max_speed", cls.navigation_orbit_max_speed)
            ),
            navigation_pan_max_speed=float(
                data.get("navigation_pan_max_speed", cls.navigation_pan_max_speed)
            ),
            navigation_zoom_max_speed=float(
                data.get("navigation_zoom_max_speed", cls.navigation_zoom_max_speed)
            ),
            navigation_motion_start_threshold=float(
                data.get("navigation_motion_start_threshold", cls.navigation_motion_start_threshold)
            ),
            navigation_motion_stop_threshold=float(
                data.get("navigation_motion_stop_threshold", cls.navigation_motion_stop_threshold)
            ),
            navigation_zoom_start_threshold=float(
                data.get("navigation_zoom_start_threshold", cls.navigation_zoom_start_threshold)
            ),
            navigation_zoom_stop_threshold=float(
                data.get("navigation_zoom_stop_threshold", cls.navigation_zoom_stop_threshold)
            ),
            navigation_hand_loss_grace_frames=int(
                data.get("navigation_hand_loss_grace_frames", cls.navigation_hand_loss_grace_frames)
            ),
            navigation_outlier_threshold=float(
                data.get("navigation_outlier_threshold", cls.navigation_outlier_threshold)
            ),
            navigation_one_euro_enabled=_as_bool(
                data.get("navigation_one_euro_enabled", cls.navigation_one_euro_enabled)
            ),
            navigation_one_euro_min_cutoff=float(
                data.get("navigation_one_euro_min_cutoff", cls.navigation_one_euro_min_cutoff)
            ),
            navigation_one_euro_beta=float(
                data.get("navigation_one_euro_beta", cls.navigation_one_euro_beta)
            ),
            navigation_one_euro_derivative_cutoff=float(
                data.get("navigation_one_euro_derivative_cutoff", cls.navigation_one_euro_derivative_cutoff)
            ),
            navigation_precision_modifier=str(
                data.get("navigation_precision_modifier", cls.navigation_precision_modifier)
            ).casefold(),
            navigation_fast_modifier=str(
                data.get("navigation_fast_modifier", cls.navigation_fast_modifier)
            ).casefold(),
            navigation_precision_scale=float(
                data.get("navigation_precision_scale", cls.navigation_precision_scale)
            ),
            navigation_fast_scale=float(
                data.get("navigation_fast_scale", cls.navigation_fast_scale)
            ),
            navigation_global_hotkey=str(
                data.get("navigation_global_hotkey", cls.navigation_global_hotkey)
            ),
            navigation_emergency_hotkey=str(
                data.get("navigation_emergency_hotkey", cls.navigation_emergency_hotkey)
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
