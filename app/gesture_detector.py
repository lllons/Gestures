"""Nose-touch gesture state machine.

The detector consumes only normalized MediaPipe landmarks.  Distance is divided
by the detected face width, so a user can move toward or away from the camera
without changing the meaning of the configured threshold.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from .face_tracker import FaceDetection
from .hand_tracker import INDEX_FINGER_TIP_INDEX, HandDetection, Landmark
from .settings import AppSettings


class DetectionState(str, Enum):
    READY = "READY"
    APPROACHING = "APPROACHING"
    TOUCH_DETECTED = "TOUCH DETECTED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class GestureSnapshot:
    state: DetectionState
    hand_detected: bool
    face_detected: bool
    hand_count: int
    index_tip: Landmark | None
    nose: Landmark | None
    face_scale: float | None
    relative_distance: float | None
    triggered: bool
    touch_elapsed_ms: int
    cooldown_remaining_ms: int
    awaiting_release: bool
    message: str


class LandmarkSmoother:
    """Moving-average smoother for normalized landmark positions."""

    def __init__(self, window_size: int) -> None:
        self._window_size = max(1, window_size)
        self._points: deque[tuple[float, float, float]] = deque(maxlen=self._window_size)

    def set_window_size(self, window_size: int) -> None:
        self._window_size = max(1, window_size)
        self._points = deque(self._points, maxlen=self._window_size)

    def add(self, point: Landmark) -> Landmark:
        self._points.append((point.x, point.y, point.z))
        count = len(self._points)
        return Landmark(
            x=sum(item[0] for item in self._points) / count,
            y=sum(item[1] for item in self._points) / count,
            z=sum(item[2] for item in self._points) / count,
        )

    def clear(self) -> None:
        self._points.clear()


class GestureDetector:
    """Stateful one-shot detector for INDEX FINGER TIP -> NOSE."""

    RELEASE_MARGIN = 1.35
    APPROACH_MARGIN = 2.5
    RELEASE_CONFIRM_SECONDS = 0.12

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._smoother = LandmarkSmoother(settings.smoothing_frames)
        self._touch_started_at: float | None = None
        self._cooldown_until = 0.0
        self._awaiting_release = False
        self._release_started_at: float | None = None

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._smoother.set_window_size(settings.smoothing_frames)

    def reset(self) -> None:
        self._smoother.clear()
        self._touch_started_at = None
        self._cooldown_until = 0.0
        self._awaiting_release = False
        self._release_started_at = None

    def process(
        self,
        hands: list[HandDetection],
        face: FaceDetection | None,
        now: float | None = None,
        force_disabled: bool = False,
    ) -> GestureSnapshot:
        """Process one frame and report whether a new trigger occurred."""

        current_time = time.monotonic() if now is None else now
        selected_hand = self._select_hand(hands, face)
        raw_index_tip = _index_tip(selected_hand)
        if raw_index_tip is None:
            self._smoother.clear()
        index_tip = self._smoother.add(raw_index_tip) if raw_index_tip else None
        nose = face.nose if face else None
        relative_distance = _relative_distance(index_tip, nose, face.face_width if face else None)

        if force_disabled or not self._settings.detection_enabled:
            self.reset()
            return self._snapshot(
                state=DetectionState.READY,
                hands=hands,
                face=face,
                index_tip=index_tip,
                relative_distance=relative_distance,
                message="Detection disabled" if not force_disabled else "Calibration in progress",
            )

        if self._awaiting_release:
            released = relative_distance is None or relative_distance >= self._settings.touch_threshold * self.RELEASE_MARGIN
            if released:
                if self._release_started_at is None:
                    self._release_started_at = current_time
                elif current_time - self._release_started_at >= self.RELEASE_CONFIRM_SECONDS:
                    self._awaiting_release = False
                    self._release_started_at = None
            else:
                self._release_started_at = None

            cooldown_remaining = max(0.0, self._cooldown_until - current_time)
            if self._awaiting_release or cooldown_remaining > 0:
                return self._snapshot(
                    state=DetectionState.COOLDOWN,
                    hands=hands,
                    face=face,
                    index_tip=index_tip,
                    relative_distance=relative_distance,
                    cooldown_remaining_ms=round(cooldown_remaining * 1000),
                    awaiting_release=self._awaiting_release,
                    message=(
                        "Move your index finger away"
                        if self._awaiting_release and cooldown_remaining <= 0
                        else "Waiting for release / cooldown"
                    ),
                )

        if relative_distance is None:
            self._touch_started_at = None
            return self._snapshot(
                state=DetectionState.READY,
                hands=hands,
                face=face,
                index_tip=index_tip,
                relative_distance=None,
                message="Show one hand and your face",
            )

        threshold = self._settings.touch_threshold
        if relative_distance <= threshold:
            if self._touch_started_at is None:
                self._touch_started_at = current_time
            elapsed = current_time - self._touch_started_at
            required = self._settings.touch_duration_ms / 1000.0
            if elapsed >= required:
                self._touch_started_at = None
                self._cooldown_until = current_time + self._settings.cooldown_ms / 1000.0
                self._awaiting_release = True
                self._release_started_at = None
                return self._snapshot(
                    state=DetectionState.TOUCH_DETECTED,
                    hands=hands,
                    face=face,
                    index_tip=index_tip,
                    relative_distance=relative_distance,
                    triggered=True,
                    touch_elapsed_ms=round(elapsed * 1000),
                    cooldown_remaining_ms=self._settings.cooldown_ms,
                    awaiting_release=True,
                    message="Shortcut sent once",
                )
            return self._snapshot(
                state=DetectionState.APPROACHING,
                hands=hands,
                face=face,
                index_tip=index_tip,
                relative_distance=relative_distance,
                touch_elapsed_ms=round(elapsed * 1000),
                message="Hold fingertip at nose",
            )

        self._touch_started_at = None
        state = (
            DetectionState.APPROACHING
            if relative_distance <= threshold * self.APPROACH_MARGIN
            else DetectionState.READY
        )
        return self._snapshot(
            state=state,
            hands=hands,
            face=face,
            index_tip=index_tip,
            relative_distance=relative_distance,
            message="Index fingertip approaching nose" if state is DetectionState.APPROACHING else "Ready",
        )

    def _select_hand(
        self,
        hands: list[HandDetection],
        face: FaceDetection | None,
    ) -> HandDetection | None:
        valid_hands = [hand for hand in hands if _index_tip(hand) is not None]
        if not valid_hands:
            return None
        if face is None:
            return valid_hands[0]
        return min(
            valid_hands,
            key=lambda hand: _distance(_index_tip(hand), face.nose),  # type: ignore[arg-type]
        )

    def _snapshot(
        self,
        state: DetectionState,
        hands: list[HandDetection],
        face: FaceDetection | None,
        index_tip: Landmark | None,
        relative_distance: float | None,
        *,
        triggered: bool = False,
        touch_elapsed_ms: int = 0,
        cooldown_remaining_ms: int = 0,
        awaiting_release: bool | None = None,
        message: str,
    ) -> GestureSnapshot:
        return GestureSnapshot(
            state=state,
            hand_detected=bool(hands),
            face_detected=face is not None,
            hand_count=len(hands),
            index_tip=index_tip,
            nose=face.nose if face else None,
            face_scale=face.face_width if face else None,
            relative_distance=relative_distance,
            triggered=triggered,
            touch_elapsed_ms=touch_elapsed_ms,
            cooldown_remaining_ms=cooldown_remaining_ms,
            awaiting_release=(
                self._awaiting_release if awaiting_release is None else awaiting_release
            ),
            message=message,
        )


def _index_tip(hand: HandDetection | None) -> Landmark | None:
    if hand is None or len(hand.landmarks) <= INDEX_FINGER_TIP_INDEX:
        return None
    return hand.landmarks[INDEX_FINGER_TIP_INDEX]


def _distance(first: Landmark | None, second: Landmark | None) -> float:
    if first is None or second is None:
        return math.inf
    return math.sqrt((first.x - second.x) ** 2 + (first.y - second.y) ** 2)


def _relative_distance(
    index_tip: Landmark | None,
    nose: Landmark | None,
    face_width: float | None,
) -> float | None:
    if index_tip is None or nose is None or not face_width or face_width <= 0:
        return None
    return _distance(index_tip, nose) / face_width
