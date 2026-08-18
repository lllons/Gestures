"""Universal OS input output for two-hand 3D navigation.

This module intentionally knows nothing about Blender, Maya, CAD, or any other
receiver. It emits the same relative mouse moves, wheel events, buttons, and
modifiers a person would use in the currently focused application.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING

from .input_profile import InputProfile, get_profile
from .keyboard_controller import ShortcutError, parse_shortcut
from .settings import AppSettings

if TYPE_CHECKING:
    from .navigation import NavigationSnapshot


class InputState(str, Enum):
    IDLE = "IDLE"
    NAVIGATION_READY = "NAVIGATION_READY"
    ORBITING = "ORBITING"
    PANNING = "PANNING"
    ZOOMING = "ZOOMING"
    LOST_TRACKING = "LOST_TRACKING"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class InputStatus:
    state: InputState = InputState.IDLE
    global_enabled: bool = True
    feature_enabled: bool = False
    held_buttons: tuple[str, ...] = ()
    held_modifiers: tuple[str, ...] = ()
    message: str = "3D navigation is ready"
    error: str = ""
    precision_mode: bool = False
    speed_factor: float = 1.0
    output_x: int = 0
    output_y: int = 0
    output_zoom: int = 0


class InputBackend(Protocol):
    """Minimal backend contract, allowing deterministic controller tests."""

    def press_button(self, button: str) -> None: ...

    def release_button(self, button: str) -> None: ...

    def press_modifier(self, modifier: str) -> None: ...

    def release_modifier(self, modifier: str) -> None: ...

    def move_relative(self, dx: int, dy: int) -> None: ...

    def scroll(self, horizontal: int, vertical: int) -> None: ...


class PynputInputBackend:
    """Emit relative pointer and modifier events through the existing pynput dependency."""

    def __init__(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "pynput is not installed. Run python -m pip install -r requirements.txt."
            ) from exc
        self._keyboard = keyboard
        self._mouse = mouse
        self._mouse_controller = mouse.Controller()
        self._keyboard_controller = keyboard.Controller()

    def press_button(self, button: str) -> None:
        self._mouse_controller.press(self._mouse_button(button))

    def release_button(self, button: str) -> None:
        self._mouse_controller.release(self._mouse_button(button))

    def press_modifier(self, modifier: str) -> None:
        self._keyboard_controller.press(self._modifier(modifier))

    def release_modifier(self, modifier: str) -> None:
        self._keyboard_controller.release(self._modifier(modifier))

    def move_relative(self, dx: int, dy: int) -> None:
        # Controller.move is relative; it does not teleport the desktop cursor.
        self._mouse_controller.move(dx, dy)

    def scroll(self, horizontal: int, vertical: int) -> None:
        self._mouse_controller.scroll(horizontal, vertical)

    def _mouse_button(self, button: str) -> Any:
        return getattr(self._mouse.Button, button)

    def _modifier(self, modifier: str) -> Any:
        if modifier == "none":
            raise ValueError("The 'none' modifier cannot be emitted.")
        return getattr(self._keyboard.Key, modifier)


class NavigationInputController:
    """Convert active analog navigation snapshots into tracked OS input state.

    Every pressed button/modifier is recorded before the next frame is handled.
    ``stop`` and ``emergency_stop`` release only keys this controller pressed,
    so a lost hand can never leave Shift or a mouse button stuck down.
    """

    def __init__(
        self,
        settings: AppSettings,
        backend: InputBackend | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend or PynputInputBackend()
        self._lock = threading.RLock()
        self._global_enabled = True
        self._held_button: str | None = None
        self._held_modifiers: tuple[str, ...] = ()
        self._speed_modifiers: set[str] = set()
        self._motion_remainder_x = 0.0
        self._motion_remainder_y = 0.0
        self._wheel_remainder = 0.0
        self._state = InputState.IDLE
        self._message = "3D navigation is ready"
        self._error = ""
        self._last_output_x = 0
        self._last_output_y = 0
        self._last_output_zoom = 0
        self._listener: Any | None = None
        self._toggle_key: Any | None = None
        self._emergency_key: Any | None = None
        self._last_hotkey_error = ""

    @property
    def status(self) -> InputStatus:
        with self._lock:
            precision = self._is_modifier_down_locked(
                self._settings.navigation_precision_modifier
            )
            return InputStatus(
                state=self._state,
                global_enabled=self._global_enabled,
                feature_enabled=self._settings.navigation_enabled,
                held_buttons=(self._held_button,) if self._held_button else (),
                held_modifiers=self._held_modifiers,
                message=self._message,
                error=self._error,
                precision_mode=precision,
                speed_factor=self._speed_factor_locked(),
                output_x=self._last_output_x,
                output_y=self._last_output_y,
                output_zoom=self._last_output_zoom,
            )

    @property
    def last_hotkey_error(self) -> str:
        return self._last_hotkey_error

    def update_settings(self, settings: AppSettings) -> None:
        with self._lock:
            self._settings = settings
            if not settings.navigation_enabled:
                self._release_held_locked()
                self._state = InputState.DISABLED
                self._message = "3D navigation is disabled"
            elif self._state is InputState.DISABLED and self._global_enabled:
                self._state = InputState.NAVIGATION_READY
                self._message = "Press F8 or use the activation pose to navigate"
            self._configure_hotkey_keys_locked()

    def start_hotkeys(self) -> bool:
        """Start global safety and speed-modifier listeners without blocking inference."""

        with self._lock:
            if self._listener is not None:
                return True
            try:
                from pynput import keyboard

                self._configure_hotkey_keys_locked()
                if self._toggle_key is None or self._emergency_key is None:
                    return False
                listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                listener.daemon = True
                listener.start()
                self._listener = listener
                self._last_hotkey_error = ""
                return True
            except (ImportError, OSError, RuntimeError, ShortcutError, ValueError) as exc:
                self._last_hotkey_error = f"Global hotkeys unavailable: {exc}"
                return False

    def stop_hotkeys(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
            self._speed_modifiers.clear()
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass

    def toggle_global(self) -> bool:
        """Toggle the global safety gate; turning it off releases input first."""

        with self._lock:
            self._global_enabled = not self._global_enabled
            if not self._global_enabled:
                self._release_held_locked()
                self._state = InputState.DISABLED
                self._message = "Global navigation OFF (press F8 to re-enable)"
            else:
                self._state = InputState.NAVIGATION_READY
                self._message = "Global navigation ON; hold the activation pose"
            return self._global_enabled

    def emergency_stop(self) -> None:
        """Disable navigation and release every simulated control immediately."""

        with self._lock:
            self._global_enabled = False
            self._release_held_locked()
            self._state = InputState.DISABLED
            self._message = "Emergency stop: all simulated input released"

    def stop(self) -> None:
        """Release held controls while leaving the F8 global gate unchanged."""

        with self._lock:
            self._release_held_locked()
            self._state = (
                InputState.NAVIGATION_READY
                if self._settings.navigation_enabled and self._global_enabled
                else InputState.DISABLED
            )
            self._message = "Navigation stopped; all simulated input released"

    def close(self) -> None:
        self.stop()
        self.stop_hotkeys()

    def apply(self, snapshot: "NavigationSnapshot | None") -> InputStatus:
        """Apply one frame and return the complete simulated-input status."""

        with self._lock:
            self._last_output_x = 0
            self._last_output_y = 0
            self._last_output_zoom = 0
            try:
                if not self._settings.navigation_enabled or not self._global_enabled:
                    self._release_held_locked()
                    self._state = InputState.DISABLED
                    if not self._settings.navigation_enabled:
                        self._message = "3D navigation is disabled"
                    return self.status

                if (
                    snapshot is None
                    or not snapshot.active
                    or snapshot.hand_count != 2
                    or getattr(snapshot, "gesture", "") == "Deactivating"
                ):
                    was_holding = bool(self._held_button or self._held_modifiers)
                    self._release_held_locked()
                    snapshot_state = (
                        getattr(snapshot.state, "value", snapshot.state)
                        if snapshot is not None
                        else ""
                    )
                    self._state = (
                        InputState.LOST_TRACKING
                        if snapshot_state == "LOST"
                        else InputState.NAVIGATION_READY
                    )
                    self._message = (
                        "Tracking lost; all simulated input released"
                        if was_holding and snapshot_state == "LOST"
                        else (
                            "Deactivation pose held; simulated input released"
                            if snapshot is not None and getattr(snapshot, "gesture", "") == "Deactivating"
                            else (snapshot.message if snapshot is not None else "Waiting for two hands")
                        )
                    )
                    return self.status

                profile = get_profile(
                    self._settings.navigation_profile,
                    self._settings.navigation_profiles,
                )
                control_mode = snapshot.control_mode
                pan = (
                    (
                        control_mode == "PAN"
                        or (
                            control_mode == "FULL 3D"
                            and snapshot.pan_pose
                        )
                    )
                    and (abs(snapshot.pan_x) + abs(snapshot.pan_y) > 1e-12)
                )
                orbit = (
                    control_mode in {"ORBIT", "FULL 3D"}
                    and not pan
                    and (
                        abs(snapshot.orbit_x)
                        + abs(snapshot.orbit_y)
                        + abs(getattr(snapshot, "roll", 0.0))
                        > 1e-12
                    )
                )
                allow_zoom = control_mode in {"ZOOM", "FULL 3D"}
                if control_mode == "ZOOM":
                    orbit = False
                    pan = False

                desired_button: str | None = None
                desired_modifiers: tuple[str, ...] = ()
                if orbit:
                    desired_button = profile.orbit_button
                    desired_modifiers = profile.orbit_modifiers
                elif pan:
                    desired_button = profile.pan_button
                    desired_modifiers = profile.pan_modifiers
                self._transition_held_locked(desired_button, desired_modifiers)

                if orbit:
                    motion_x = snapshot.orbit_x + getattr(snapshot, "roll", 0.0)
                    motion_y = snapshot.orbit_y
                    self._emit_motion_locked(motion_x, motion_y)
                elif pan:
                    self._emit_motion_locked(snapshot.pan_x, snapshot.pan_y)

                if allow_zoom:
                    self._emit_zoom_locked(snapshot.zoom, profile)

                if pan:
                    self._state = InputState.PANNING
                elif orbit:
                    self._state = InputState.ORBITING
                elif allow_zoom and abs(snapshot.zoom) > 0:
                    self._state = InputState.ZOOMING
                else:
                    self._state = InputState.NAVIGATION_READY
                self._message = f"{profile.name}: {self._state.value.lower()}"
                self._error = ""
                return self.status
            except Exception as exc:  # pragma: no cover - backend/platform dependent
                self._release_held_locked()
                self._state = InputState.LOST_TRACKING
                self._error = f"OS input stopped: {exc}"
                self._message = self._error
                return self.status

    def _configure_hotkey_keys_locked(self) -> None:
        try:
            self._toggle_key = parse_shortcut(self._settings.navigation_global_hotkey).key
            self._emergency_key = parse_shortcut(self._settings.navigation_emergency_hotkey).key
        except (ShortcutError, ImportError, AttributeError) as exc:
            self._toggle_key = None
            self._emergency_key = None
            self._last_hotkey_error = f"Invalid navigation hotkey: {exc}"

    def _on_key_press(self, key: Any) -> None:
        with self._lock:
            modifier = _modifier_name_for_key(key)
            if modifier:
                self._speed_modifiers.add(modifier)
            if self._toggle_key is not None and _same_key(key, self._toggle_key):
                self.toggle_global()
            elif self._emergency_key is not None and _same_key(key, self._emergency_key):
                self.emergency_stop()

    def _on_key_release(self, key: Any) -> None:
        with self._lock:
            modifier = _modifier_name_for_key(key)
            if modifier:
                self._speed_modifiers.discard(modifier)

    def _is_modifier_down_locked(self, modifier: str) -> bool:
        return modifier != "none" and modifier in self._speed_modifiers

    def _speed_factor_locked(self) -> float:
        if self._is_modifier_down_locked(self._settings.navigation_precision_modifier):
            return self._settings.navigation_precision_scale
        if self._is_modifier_down_locked(self._settings.navigation_fast_modifier):
            return self._settings.navigation_fast_scale
        return 1.0

    def _transition_held_locked(
        self,
        button: str | None,
        modifiers: tuple[str, ...],
    ) -> None:
        modifiers = tuple(modifier for modifier in modifiers if modifier != "none")
        if self._held_button == button and self._held_modifiers == modifiers:
            return
        self._release_held_locked()
        self._held_modifiers = ()
        for modifier in modifiers:
            # Record before pressing so an exception cannot strand a modifier
            # that the platform accepted just before reporting an error.
            self._held_modifiers = (*self._held_modifiers, modifier)
            self._backend.press_modifier(modifier)
        if button is not None:
            self._held_button = button
            self._backend.press_button(button)

    def _release_held_locked(self) -> None:
        button = self._held_button
        modifiers = self._held_modifiers
        self._held_button = None
        self._held_modifiers = ()
        self._motion_remainder_x = 0.0
        self._motion_remainder_y = 0.0
        self._wheel_remainder = 0.0
        if button is not None:
            try:
                self._backend.release_button(button)
            except Exception:
                pass
        for modifier in reversed(modifiers):
            try:
                self._backend.release_modifier(modifier)
            except Exception:
                pass

    def _emit_motion_locked(self, x: float, y: float) -> None:
        speed_factor = self._speed_factor_locked()
        self._motion_remainder_x += x * speed_factor * self._settings.navigation_mouse_scale
        self._motion_remainder_y += y * speed_factor * self._settings.navigation_mouse_scale
        whole_x = int(self._motion_remainder_x)
        whole_y = int(self._motion_remainder_y)
        self._motion_remainder_x -= whole_x
        self._motion_remainder_y -= whole_y
        self._last_output_x = whole_x
        self._last_output_y = whole_y
        if whole_x or whole_y:
            self._backend.move_relative(whole_x, whole_y)

    def _emit_zoom_locked(self, zoom: float, profile: InputProfile) -> None:
        if not zoom:
            return
        self._wheel_remainder += (
            zoom
            * profile.zoom_in_direction
            * self._speed_factor_locked()
            * self._settings.navigation_zoom_wheel_scale
        )
        whole = int(self._wheel_remainder)
        self._wheel_remainder -= whole
        self._last_output_zoom = whole
        if whole:
            if profile.zoom_axis == "horizontal":
                self._backend.scroll(whole, 0)
            else:
                self._backend.scroll(0, whole)


def _modifier_name_for_key(key: Any) -> str | None:
    """Normalize left/right pynput modifier variants for speed controls."""

    try:
        from pynput import keyboard
    except ImportError:  # pragma: no cover - only used with the real listener
        return None
    candidates = {
        "alt": (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r),
        "shift": (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r),
        "ctrl": (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r),
        "cmd": (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r),
    }
    for name, values in candidates.items():
        if any(key == value for value in values if value is not None):
            return name
    return None


def _same_key(first: Any, second: Any) -> bool:
    if first == second:
        return True
    return str(first).casefold() == str(second).casefold()
