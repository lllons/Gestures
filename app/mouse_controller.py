"""Local mouse control for Air Mouse mode."""

from __future__ import annotations

import ctypes
import os
from .hand_tracker import Landmark


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
        self._origin_x, self._origin_y, self._width, self._height = _screen_geometry()

    def move_to(self, point: Landmark) -> None:
        """Map a normalized camera landmark to the virtual desktop."""

        self._controller.position = map_normalized_to_screen(
            point,
            self._origin_x,
            self._origin_y,
            self._width,
            self._height,
        )

    def click(self) -> None:
        """Click once at the current pointer location."""

        self._controller.click(self._mouse.Button.left, 1)


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


def _screen_geometry() -> tuple[int, int, int, int]:
    """Return the virtual desktop origin and size, with a safe fallback."""

    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
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
