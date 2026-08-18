"""OpenCV webcam discovery and capture helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2


@dataclass(frozen=True)
class CameraInfo:
    index: int
    label: str


def _create_capture(index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows and falling back safely."""

    if os.name == "nt":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(index, cv2.CAP_ANY)


def enumerate_cameras(max_index: int = 10) -> list[CameraInfo]:
    """Probe camera indices without retaining any camera handles."""

    cameras: list[CameraInfo] = []
    for index in range(max_index):
        capture = _create_capture(index)
        try:
            if capture.isOpened():
                cameras.append(CameraInfo(index=index, label=f"Camera {index}"))
        finally:
            capture.release()
    return cameras


class CameraCapture:
    """A 640x480 webcam stream with reconnect support."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> bool:
        self.release()
        self._capture = _create_capture(self.index)
        if not self._capture.isOpened():
            self.release()
            return False
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture.set(cv2.CAP_PROP_FPS, 30)
        return True

    def read(self) -> tuple[bool, Any]:
        if not self.is_open:
            return False, None
        return self._capture.read()  # type: ignore[union-attr]

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
