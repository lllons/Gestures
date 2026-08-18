"""Nose-touch and thumb-index pinch gesture state machines.

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
from .hand_tracker import (
    INDEX_FINGER_TIP_INDEX,
    MIDDLE_FINGER_MCP_INDEX,
    MIDDLE_FINGER_TIP_INDEX,
    THUMB_TIP_INDEX,
    WRIST_INDEX,
    HandDetection,
    Landmark,
)
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
    pinch_detected: bool = False
    pinch_distance: float | None = None
    pinch_triggered: bool = False
    pinch_cooldown_remaining_ms: int = 0
    pinch_awaiting_release: bool = False
    scroll_active: bool = False
    finger_separation: float | None = None
    scroll_delta_x: float = 0.0
    scroll_delta_y: float = 0.0


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
    """Stateful one-shot detector for nose touch and thumb-index pinch."""

    RELEASE_MARGIN = 1.35
    APPROACH_MARGIN = 2.5
    RELEASE_CONFIRM_SECONDS = 0.12
    PINCH_THRESHOLD = 0.35
    PINCH_RELEASE_THRESHOLD = 0.55
    SCROLL_FINGER_SEPARATION_THRESHOLD = 0.65
    SCROLL_DEADZONE = 0.001
    SCROLL_SENSITIVITY = 80.0

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._smoother = LandmarkSmoother(settings.smoothing_frames)
        self._touch_started_at: float | None = None
        self._cooldown_until = 0.0
        self._awaiting_release = False
        self._release_started_at: float | None = None
        self._pinch_cooldown_until = 0.0
        self._pinch_awaiting_release = False
        self._pinch_release_started_at: float | None = None
        self._frame_pinch_distance: float | None = None
        self._frame_pinch_detected = False
        self._frame_pinch_triggered = False
        self._frame_pinch_cooldown_remaining_ms = 0
        self._frame_pinch_awaiting_release = False
        self._previous_scroll_anchor: Landmark | None = None
        self._frame_scroll_active = False
        self._frame_finger_separation: float | None = None
        self._frame_scroll_delta_x = 0.0
        self._frame_scroll_delta_y = 0.0

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._smoother.set_window_size(settings.smoothing_frames)

    def reset(self) -> None:
        self._smoother.clear()
        self._touch_started_at = None
        self._cooldown_until = 0.0
        self._awaiting_release = False
        self._release_started_at = None
        self._pinch_cooldown_until = 0.0
        self._pinch_awaiting_release = False
        self._pinch_release_started_at = None
        self._previous_scroll_anchor = None

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
        pinch_distance = _closest_pinch_distance(hands)
        pinch_detected = (
            pinch_distance is not None and pinch_distance <= self.PINCH_THRESHOLD
        )
        finger_separation = normalized_finger_separation(selected_hand)
        scroll_active = (
            finger_separation is not None
            and finger_separation >= self.SCROLL_FINGER_SEPARATION_THRESHOLD
        )
        self._frame_finger_separation = finger_separation
        self._frame_scroll_active = scroll_active
        self._frame_scroll_delta_x = 0.0
        self._frame_scroll_delta_y = 0.0
        if scroll_active and selected_hand is not None:
            scroll_anchor = _finger_pair_center(selected_hand)
            if self._previous_scroll_anchor is not None:
                self._frame_scroll_delta_x = scroll_anchor.x - self._previous_scroll_anchor.x
                self._frame_scroll_delta_y = scroll_anchor.y - self._previous_scroll_anchor.y
                if abs(self._frame_scroll_delta_x) < self.SCROLL_DEADZONE:
                    self._frame_scroll_delta_x = 0.0
                if abs(self._frame_scroll_delta_y) < self.SCROLL_DEADZONE:
                    self._frame_scroll_delta_y = 0.0
            self._previous_scroll_anchor = scroll_anchor
        else:
            self._previous_scroll_anchor = None
        self._frame_pinch_distance = pinch_distance
        self._frame_pinch_detected = pinch_detected
        self._frame_pinch_triggered = False
        self._frame_pinch_cooldown_remaining_ms = 0
        self._frame_pinch_awaiting_release = self._pinch_awaiting_release

        if force_disabled or not self._settings.detection_enabled:
            self.reset()
            self._frame_pinch_detected = False
            self._frame_pinch_awaiting_release = False
            self._frame_scroll_active = False
            self._frame_finger_separation = finger_separation
            self._frame_scroll_delta_x = 0.0
            self._frame_scroll_delta_y = 0.0
            return self._snapshot(
                state=DetectionState.READY,
                hands=hands,
                face=face,
                index_tip=index_tip,
                relative_distance=relative_distance,
                message="Detection disabled" if not force_disabled else "Calibration in progress",
            )

        (
            self._frame_pinch_triggered,
            self._frame_pinch_cooldown_remaining_ms,
            self._frame_pinch_awaiting_release,
        ) = self._update_pinch(pinch_distance, current_time)

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
            pinch_detected=self._frame_pinch_detected,
            pinch_distance=self._frame_pinch_distance,
            pinch_triggered=self._frame_pinch_triggered,
            pinch_cooldown_remaining_ms=self._frame_pinch_cooldown_remaining_ms,
            pinch_awaiting_release=self._frame_pinch_awaiting_release,
            scroll_active=self._frame_scroll_active,
            finger_separation=self._frame_finger_separation,
            scroll_delta_x=self._frame_scroll_delta_x,
            scroll_delta_y=self._frame_scroll_delta_y,
        )

    def _update_pinch(
        self,
        pinch_distance: float | None,
        current_time: float,
    ) -> tuple[bool, int, bool]:
        """Return trigger/cooldown/release state for the thumb-index pinch."""

        pinch_active = (
            pinch_distance is not None and pinch_distance <= self.PINCH_THRESHOLD
        )
        if self._pinch_awaiting_release:
            released = (
                pinch_distance is None
                or pinch_distance >= self.PINCH_RELEASE_THRESHOLD
            )
            if released:
                if self._pinch_release_started_at is None:
                    self._pinch_release_started_at = current_time
                elif (
                    current_time - self._pinch_release_started_at
                    >= self.RELEASE_CONFIRM_SECONDS
                ):
                    self._pinch_awaiting_release = False
                    self._pinch_release_started_at = None
            else:
                self._pinch_release_started_at = None

        cooldown_remaining = max(0.0, self._pinch_cooldown_until - current_time)
        if self._pinch_awaiting_release or cooldown_remaining > 0:
            return (
                False,
                round(cooldown_remaining * 1000),
                self._pinch_awaiting_release,
            )

        if pinch_active:
            self._pinch_cooldown_until = (
                current_time + self._settings.cooldown_ms / 1000.0
            )
            self._pinch_awaiting_release = True
            self._pinch_release_started_at = None
            return True, self._settings.cooldown_ms, True

        return False, 0, False


def _index_tip(hand: HandDetection | None) -> Landmark | None:
    if hand is None or len(hand.landmarks) <= INDEX_FINGER_TIP_INDEX:
        return None
    return hand.landmarks[INDEX_FINGER_TIP_INDEX]


def normalized_pinch_distance(hand: HandDetection | None) -> float | None:
    """Return thumb-index distance normalized by the hand's palm length."""

    if hand is None or len(hand.landmarks) <= MIDDLE_FINGER_MCP_INDEX:
        return None
    thumb = hand.landmarks[THUMB_TIP_INDEX]
    index = hand.landmarks[INDEX_FINGER_TIP_INDEX]
    palm_length = _palm_length(hand)
    if palm_length <= 0:
        return None
    return _distance(thumb, index) / palm_length


def normalized_finger_separation(hand: HandDetection | None) -> float | None:
    """Return index/middle fingertip separation normalized by palm length."""

    if hand is None or len(hand.landmarks) <= MIDDLE_FINGER_TIP_INDEX:
        return None
    index = hand.landmarks[INDEX_FINGER_TIP_INDEX]
    middle = hand.landmarks[MIDDLE_FINGER_TIP_INDEX]
    palm_length = _palm_length(hand)
    if palm_length <= 0:
        return None
    return _distance(index, middle) / palm_length


def _palm_length(hand: HandDetection) -> float:
    if len(hand.landmarks) <= MIDDLE_FINGER_MCP_INDEX:
        return 0.0
    return _distance(
        hand.landmarks[WRIST_INDEX],
        hand.landmarks[MIDDLE_FINGER_MCP_INDEX],
    )


def _finger_pair_center(hand: HandDetection) -> Landmark:
    index = hand.landmarks[INDEX_FINGER_TIP_INDEX]
    middle = hand.landmarks[MIDDLE_FINGER_TIP_INDEX]
    return Landmark(
        x=(index.x + middle.x) / 2,
        y=(index.y + middle.y) / 2,
        z=(index.z + middle.z) / 2,
    )


def _closest_pinch_distance(hands: list[HandDetection]) -> float | None:
    distances = [
        distance
        for hand in hands
        if (distance := normalized_pinch_distance(hand)) is not None
    ]
    return min(distances) if distances else None


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
