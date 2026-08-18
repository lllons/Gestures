"""Local mouse control for Air Mouse mode."""

from __future__ import annotations

import ctypes
import os
from typing import Any

from .hand_tracker import Landmark


_WHEEL_DELTA = 120
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x01000


class MouseController:
    """Move and click the OS pointer from normalized camera coordinates."""

    def __init__(self) -> None:
        try:
            from pynput import mouse
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "pynput is not installed. Run python -m pip install -r requirements.txt."
            ) from exc

        self._mouse = mouse
        self._controller = mouse.Controller()
        self._user32 = _windows_user32()
        self._origin_x, self._origin_y, self._width, self._height = _screen_geometry()
        self._scroll_remainder_x = 0.0
        self._scroll_remainder_y = 0.0

    def move_to(self, point: Landmark) -> None:
        """Map a normalized camera landmark to the virtual desktop."""

        position = map_normalized_to_screen(
            point,
            self._origin_x,
            self._origin_y,
            self._width,
            self._height,
        )
        if self._user32 is not None:
            # SetCursorPos avoids the active-window-dependent path that can make
            # pynput pointer updates feel slower outside the Gestures window.
            self._user32.SetCursorPos(position[0], position[1])
        else:
            self._controller.position = position

    def click(self) -> None:
        """Click once at the current pointer location."""

        if self._user32 is not None:
            self._user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            self._user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            self._controller.click(self._mouse.Button.left, 1)

    def scroll(self, horizontal_units: float, vertical_units: float) -> None:
        """Scroll by fractional wheel units, preserving sub-unit motion."""

        self._scroll_remainder_x += horizontal_units
        self._scroll_remainder_y += vertical_units
        whole_x = int(self._scroll_remainder_x)
        whole_y = int(self._scroll_remainder_y)
        self._scroll_remainder_x -= whole_x
        self._scroll_remainder_y -= whole_y
        if whole_x == 0 and whole_y == 0:
            return

        if self._user32 is not None:
            if whole_y:
                self._user32.mouse_event(
                    _MOUSEEVENTF_WHEEL,
                    0,
                    0,
                    whole_y * _WHEEL_DELTA,
                    0,
                )
            if whole_x:
                self._user32.mouse_event(
                    _MOUSEEVENTF_HWHEEL,
                    0,
                    0,
                    whole_x * _WHEEL_DELTA,
                    0,
                )
        else:
            self._controller.scroll(whole_x, whole_y)


def map_normalized_to_screen(
    point: Landmark,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert a normalized camera point to clamped virtual-screen pixels."""

    if width <= 0 or height <= 0:
        raise ValueError("Screen dimensions must be positive.")
    normalized_x = max(0.0, min(1.0, point.x))
    normalized_y = max(0.0, min(1.0, point.y))
    return (
        origin_x + round(normalized_x * max(0, width - 1)),
        origin_y + round(normalized_y * max(0, height - 1)),
    )


def _windows_user32() -> Any:
    if os.name != "nt":
        return None
    try:
        return ctypes.windll.user32
    except (AttributeError, OSError):
        return None


def _screen_geometry() -> tuple[int, int, int, int]:
    """Return the virtual desktop origin and size, with a safe fallback."""

    user32 = _windows_user32()
    if user32 is not None:
        try:
            origin_x = int(user32.GetSystemMetrics(76))
            origin_y = int(user32.GetSystemMetrics(77))
            width = int(user32.GetSystemMetrics(78))
            height = int(user32.GetSystemMetrics(79))
            if width > 0 and height > 0:
                return origin_x, origin_y, width, height
        except (AttributeError, OSError):
            pass

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return 0, 0, width, height
    except Exception:
        pass

    # The fallback is only for environments without a desktop display; Windows
    # uses the exact virtual desktop metrics above.
    return 0, 0, 1920, 1080
