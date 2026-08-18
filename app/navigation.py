"""Two-hand gesture interpretation for continuous Blender navigation.

This module deliberately has no GUI, socket, camera, or Blender dependencies.
It consumes the existing MediaPipe ``HandDetection`` values and produces a
small per-frame state packet that another output adapter can use.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any

from .hand_tracker import HandDetection, Landmark
from .settings import AppSettings


class NavigationState(str, Enum):
    DISABLED = "DISABLED"
    IDLE = "IDLE"
    ACTIVATING = "ACTIVATING"
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"
    LOST = "LOST"


@dataclass(frozen=True)
class NavigationSnapshot:
    """One frame of interpreted two-hand navigation data.

    Movement values are deltas for this frame, never accumulated positions.
    That makes a lost-hand packet naturally neutral and prevents tracking drift
    from slowly moving the Blender view.
    """

    state: NavigationState
    enabled: bool
    active: bool
    hand_count: int
    left_hand: Landmark | None
    right_hand: Landmark | None
    midpoint: Landmark | None
    distance: float | None
    distance_delta: float
    angle: float | None
    angle_delta: float
    midpoint_delta_x: float
    midpoint_delta_y: float
    left_velocity_x: float
    left_velocity_y: float
    right_velocity_x: float
    right_velocity_y: float
    confidence: float
    activation_progress: float
    gesture: str
    orbit_x: float
    orbit_y: float
    pan_x: float
    pan_y: float
    zoom: float
    roll: float
    calibration_active: bool = False
    calibration_completed: bool = False
    calibration_succeeded: bool = False
    target_mode: str = "Viewport"
    control_mode: str = "FULL 3D"
    message: str = ""
    neutral_midpoint: Landmark | None = None
    neutral_distance: float | None = None
    neutral_angle: float | None = None

    @property
    def hands_ready(self) -> bool:
        return (
            self.hand_count == 2
            and self.left_hand is not None
            and self.right_hand is not None
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible packet for the local Blender transport."""

        return {
            "type": "gestures_navigation",
            "version": 1,
            "state": self.state.value,
            "enabled": self.enabled,
            "mode": self.target_mode,
            "control_mode": self.control_mode,
            "active": self.active,
            "hands": self.hand_count,
            "left_hand": _point_payload(self.left_hand),
            "right_hand": _point_payload(self.right_hand),
            "midpoint": _point_payload(self.midpoint),
            "distance": self.distance,
            "distance_delta": self.distance_delta,
            "angle": self.angle,
            "angle_delta": self.angle_delta,
            "midpoint_delta": {
                "x": self.midpoint_delta_x,
                "y": self.midpoint_delta_y,
            },
            "hand_velocity": {
                "left": {"x": self.left_velocity_x, "y": self.left_velocity_y},
                "right": {"x": self.right_velocity_x, "y": self.right_velocity_y},
            },
            "gesture": self.gesture,
            "orbit_x": self.orbit_x,
            "orbit_y": self.orbit_y,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "zoom": self.zoom,
            "roll": self.roll,
            "confidence": self.confidence,
            "activation_progress": self.activation_progress,
            "calibration": {
                "complete": self.calibration_completed,
                "succeeded": self.calibration_succeeded,
                "neutral_midpoint": _point_payload(self.neutral_midpoint),
                "neutral_distance": self.neutral_distance,
                "neutral_angle": self.neutral_angle,
            },
            "message": self.message,
        }


@dataclass(frozen=True)
class _PairFeatures:
    left: Landmark
    right: Landmark
    midpoint: Landmark
    distance: float
    angle: float


class _PointSmoother:
    """Moving-average smoother for the palm center of one hand."""

    def __init__(self, window_size: int) -> None:
        self._points: deque[Landmark] = deque(maxlen=max(1, window_size))

    def set_window_size(self, window_size: int) -> None:
        self._points = deque(self._points, maxlen=max(1, window_size))

    def add(self, point: Landmark) -> Landmark:
        self._points.append(point)
        count = len(self._points)
        return Landmark(
            x=sum(item.x for item in self._points) / count,
            y=sum(item.y for item in self._points) / count,
            z=sum(item.z for item in self._points) / count,
        )

    def clear(self) -> None:
        self._points.clear()


class TwoHandNavigation:
    """Interpret two tracked hands as a safe, continuous navigation gesture.

    The feature is opt-in twice: the application setting must be enabled and
    the user must hold the configured two-hand pose for a short period. A
    missing hand or low-confidence pair clears the movement baseline and emits
    a neutral packet immediately.
    """

    CALIBRATION_SECONDS = 1.2
    CALIBRATION_TIMEOUT_SECONDS = 20.0
    MIN_CALIBRATION_SAMPLES = 8
    DEACTIVATION_HOLD_SECONDS = 0.35
    OPEN_FINGER_COUNT = 3
    OPEN_EXTENSION_RATIO = 1.08

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._left_smoother = _PointSmoother(settings.navigation_smoothing_frames)
        self._right_smoother = _PointSmoother(settings.navigation_smoothing_frames)
        self._active = False
        self._activation_started_at: float | None = None
        self._deactivation_started_at: float | None = None
        self._previous_features: _PairFeatures | None = None
        self._previous_time: float | None = None
        self._calibration_active = False
        self._calibration_started_at = 0.0
        self._calibration_samples: list[_PairFeatures] = []
        self._neutral_midpoint: Landmark | None = None
        self._neutral_distance: float | None = None
        self._neutral_angle: float | None = None

    @property
    def calibrated(self) -> bool:
        return self._neutral_midpoint is not None

    def update_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        self._left_smoother.set_window_size(settings.navigation_smoothing_frames)
        self._right_smoother.set_window_size(settings.navigation_smoothing_frames)
        if not settings.navigation_enabled:
            self.reset()

    def reset(self) -> None:
        """Stop navigation and clear temporal movement state."""

        self._active = False
        self._activation_started_at = None
        self._deactivation_started_at = None
        self._previous_features = None
        self._previous_time = None
        self._calibration_active = False
        self._calibration_samples.clear()
        self._left_smoother.clear()
        self._right_smoother.clear()
        # Keep a completed neutral calibration; reset is also used for a brief
        # tracking loss and should not discard the user's saved reference.

    def clear_calibration(self) -> None:
        """Forget the neutral reference as well as the current movement state."""

        self.reset()
        self._neutral_midpoint = None
        self._neutral_distance = None
        self._neutral_angle = None

    def begin_calibration(self, now: float | None = None) -> None:
        """Record both hands in a comfortable neutral position."""

        current_time = time.monotonic() if now is None else now
        self._active = False
        self._activation_started_at = None
        self._deactivation_started_at = None
        self._previous_features = None
        self._previous_time = None
        self._calibration_active = True
        self._calibration_started_at = current_time
        self._calibration_samples.clear()
        self._left_smoother.clear()
        self._right_smoother.clear()

    def process(
        self,
        hands: list[HandDetection],
        now: float | None = None,
    ) -> NavigationSnapshot:
        """Interpret one frame without blocking the camera worker."""

        current_time = time.monotonic() if now is None else now
        pair = _ordered_pair(hands, self._previous_features)
        confidence = _pair_confidence(pair)
        centers: tuple[Landmark, Landmark] | None = None

        if pair is None:
            self._clear_tracking_baseline()
        else:
            centers = (
                self._left_smoother.add(hand_center(pair[0])),
                self._right_smoother.add(hand_center(pair[1])),
            )

        if not self._settings.navigation_enabled:
            self.reset()
            return self._snapshot(
                state=NavigationState.DISABLED,
                hands=hands,
                pair=pair,
                centers=centers,
                confidence=confidence,
                message="3D navigation is disabled",
            )

        if self._calibration_active:
            return self._process_calibration(
                pair,
                centers,
                hands,
                confidence,
                current_time,
            )

        if pair is None or confidence < self._settings.navigation_min_confidence:
            was_active = self._active
            self._active = False
            self._activation_started_at = None
            self._deactivation_started_at = None
            self._clear_tracking_baseline()
            message = (
                "Tracking confidence low; navigation stopped"
                if pair is not None
                else ("Hands lost; navigation stopped" if was_active else "Show both hands")
            )
            return self._snapshot(
                state=NavigationState.LOST if was_active else NavigationState.IDLE,
                hands=hands,
                pair=None,
                confidence=confidence,
                message=message,
            )

        assert centers is not None
        features = _features(pair, centers)
        activation_match = _activation_pose(
            pair,
            self._settings.navigation_activation_gesture,
        )

        if not self._active:
            if activation_match:
                if self._activation_started_at is None:
                    self._activation_started_at = current_time
                elapsed = current_time - self._activation_started_at
                required = max(
                    0.001,
                    self._settings.navigation_activation_hold_ms / 1000.0,
                )
                progress = min(1.0, max(0.0, elapsed / required))
                if elapsed >= required:
                    self._active = True
                    self._activation_started_at = None
                    # Activation and calibration both create a fresh baseline;
                    # no old frame can create a jump on the first active packet.
                    self._previous_features = features
                    self._previous_time = current_time
                    return self._snapshot(
                        state=NavigationState.ACTIVE,
                        hands=hands,
                        pair=pair,
                        centers=centers,
                        confidence=confidence,
                        activation_progress=1.0,
                        gesture="Idle",
                        message="3D navigation active",
                    )
                return self._snapshot(
                    state=NavigationState.ACTIVATING,
                    hands=hands,
                    pair=pair,
                    centers=centers,
                    confidence=confidence,
                    activation_progress=progress,
                    message=(
                        f"Hold {self._settings.navigation_activation_gesture.lower()} "
                        f"({progress * 100:.0f}%)"
                    ),
                )
            self._activation_started_at = None
            return self._snapshot(
                state=NavigationState.IDLE,
                hands=hands,
                pair=pair,
                centers=centers,
                confidence=confidence,
                message=(
                    f"Hold {self._settings.navigation_activation_gesture.lower()} to activate"
                ),
            )

        if _deactivation_pose(pair, self._settings.navigation_deactivation_gesture):
            if self._deactivation_started_at is None:
                self._deactivation_started_at = current_time
            elapsed = current_time - self._deactivation_started_at
            if elapsed >= self.DEACTIVATION_HOLD_SECONDS:
                self.reset()
                return self._snapshot(
                    state=NavigationState.IDLE,
                    hands=hands,
                    pair=pair,
                    centers=centers,
                    confidence=confidence,
                    message="3D navigation deactivated",
                )
            # Do not apply motion while a deactivation gesture is being held.
            return self._snapshot(
                state=NavigationState.ACTIVE,
                hands=hands,
                pair=pair,
                centers=centers,
                confidence=confidence,
                gesture="Deactivating",
                message="Release or keep holding to deactivate",
            )
        self._deactivation_started_at = None

        return self._active_snapshot(features, hands, pair, confidence, current_time)

    def _clear_tracking_baseline(self) -> None:
        self._left_smoother.clear()
        self._right_smoother.clear()
        self._previous_features = None
        self._previous_time = None

    def _process_calibration(
        self,
        pair: tuple[HandDetection, HandDetection] | None,
        centers: tuple[Landmark, Landmark] | None,
        hands: list[HandDetection],
        confidence: float,
        current_time: float,
    ) -> NavigationSnapshot:
        elapsed = current_time - self._calibration_started_at
        if elapsed > self.CALIBRATION_TIMEOUT_SECONDS:
            self._calibration_active = False
            self._calibration_samples.clear()
            self._clear_tracking_baseline()
            return self._snapshot(
                state=NavigationState.IDLE,
                hands=hands,
                pair=pair,
                centers=centers,
                confidence=confidence,
                calibration_completed=True,
                calibration_succeeded=False,
                message="Calibration timed out; show both hands and try again",
            )

        if pair is not None and centers is not None and confidence >= self._settings.navigation_min_confidence:
            features = _features(pair, centers)
            self._calibration_samples.append(features)
            if (
                elapsed >= self.CALIBRATION_SECONDS
                and len(self._calibration_samples) >= self.MIN_CALIBRATION_SAMPLES
            ):
                self._neutral_midpoint = Landmark(
                    x=median(item.midpoint.x for item in self._calibration_samples),
                    y=median(item.midpoint.y for item in self._calibration_samples),
                    z=median(item.midpoint.z for item in self._calibration_samples),
                )
                self._neutral_distance = median(
                    item.distance for item in self._calibration_samples
                )
                self._neutral_angle = _circular_median(
                    [item.angle for item in self._calibration_samples]
                )
                self._calibration_active = False
                self._calibration_samples.clear()
                self._clear_tracking_baseline()
                return self._snapshot(
                    state=NavigationState.IDLE,
                    hands=hands,
                    pair=pair,
                    centers=centers,
                    confidence=confidence,
                    calibration_completed=True,
                    calibration_succeeded=True,
                    message="Calibration complete; neutral position saved",
                )
            message = (
                "Hold both hands in a comfortable neutral position "
                f"({len(self._calibration_samples)}/{self.MIN_CALIBRATION_SAMPLES})"
            )
        else:
            message = "Calibration paused; show both hands"
        return self._snapshot(
            state=NavigationState.CALIBRATING,
            hands=hands,
            pair=pair,
            centers=centers,
            confidence=confidence,
            calibration_active=True,
            message=message,
        )

    def _active_snapshot(
        self,
        features: _PairFeatures,
        hands: list[HandDetection],
        pair: tuple[HandDetection, HandDetection],
        confidence: float,
        current_time: float,
    ) -> NavigationSnapshot:
        previous = self._previous_features
        previous_time = self._previous_time
        self._previous_features = features
        self._previous_time = current_time
        if previous is None or previous_time is None:
            return self._snapshot(
                state=NavigationState.ACTIVE,
                hands=hands,
                pair=pair,
                centers=(features.left, features.right),
                confidence=confidence,
                message="3D navigation active",
            )

        dt = max(1.0 / 120.0, min(0.2, current_time - previous_time))
        max_delta = self._settings.navigation_max_speed * dt
        midpoint_dx = _limit(features.midpoint.x - previous.midpoint.x, max_delta)
        midpoint_dy = _limit(features.midpoint.y - previous.midpoint.y, max_delta)
        distance_delta = _limit(features.distance - previous.distance, max_delta)
        angle_delta = _limit(
            angle_delta_between(features.angle, previous.angle),
            max_delta * math.pi,
        )

        midpoint_dx = apply_dead_zone(midpoint_dx, self._settings.navigation_dead_zone)
        midpoint_dy = apply_dead_zone(midpoint_dy, self._settings.navigation_dead_zone)
        distance_delta = apply_dead_zone(
            distance_delta,
            max(self._settings.navigation_dead_zone / 2, 1e-6),
        )
        angle_delta = apply_dead_zone(
            angle_delta,
            max(self._settings.navigation_dead_zone, 1e-6),
        )

        direction_x = -1.0 if self._settings.navigation_invert_x else 1.0
        direction_y = -1.0 if self._settings.navigation_invert_y else 1.0
        direction_zoom = -1.0 if self._settings.navigation_invert_zoom else 1.0
        orbit_x = midpoint_dx * self._settings.navigation_orbit_sensitivity * direction_x
        orbit_y = -midpoint_dy * self._settings.navigation_orbit_sensitivity * direction_y
        pan_x = midpoint_dx * self._settings.navigation_pan_sensitivity * direction_x
        pan_y = -midpoint_dy * self._settings.navigation_pan_sensitivity * direction_y
        # Positive zoom means the hands moved farther apart (zoom in).
        zoom = distance_delta * self._settings.navigation_zoom_sensitivity * direction_zoom
        roll = (
            angle_delta * self._settings.navigation_roll_sensitivity
            if self._settings.navigation_roll_enabled
            else 0.0
        )

        mode = self._settings.navigation_control_mode
        if mode == "ORBIT":
            pan_x = pan_y = zoom = 0.0
        elif mode == "PAN":
            orbit_x = orbit_y = zoom = roll = 0.0
        elif mode == "ZOOM":
            orbit_x = orbit_y = pan_x = pan_y = roll = 0.0

        # Clamp the output after sensitivity is applied as a final safety valve.
        output_limit = self._settings.navigation_max_speed
        orbit_x = _limit(orbit_x, output_limit)
        orbit_y = _limit(orbit_y, output_limit)
        pan_x = _limit(pan_x, output_limit)
        pan_y = _limit(pan_y, output_limit)
        zoom = _limit(zoom, output_limit)
        roll = _limit(roll, output_limit)
        gesture = _gesture_name(orbit_x, orbit_y, pan_x, pan_y, zoom, roll)

        return self._snapshot(
            state=NavigationState.ACTIVE,
            hands=hands,
            pair=pair,
            centers=(features.left, features.right),
            confidence=confidence,
            distance_delta=distance_delta,
            angle_delta=angle_delta,
            midpoint_delta_x=midpoint_dx,
            midpoint_delta_y=midpoint_dy,
            left_velocity_x=_limit(
                (features.left.x - previous.left.x) / dt,
                self._settings.navigation_max_speed,
            ),
            left_velocity_y=_limit(
                (features.left.y - previous.left.y) / dt,
                self._settings.navigation_max_speed,
            ),
            right_velocity_x=_limit(
                (features.right.x - previous.right.x) / dt,
                self._settings.navigation_max_speed,
            ),
            right_velocity_y=_limit(
                (features.right.y - previous.right.y) / dt,
                self._settings.navigation_max_speed,
            ),
            gesture=gesture,
            orbit_x=orbit_x,
            orbit_y=orbit_y,
            pan_x=pan_x,
            pan_y=pan_y,
            zoom=zoom,
            roll=roll,
            message=gesture if gesture != "Idle" else "3D navigation active",
        )

    def _snapshot(
        self,
        *,
        state: NavigationState,
        hands: list[HandDetection],
        pair: tuple[HandDetection, HandDetection] | None,
        confidence: float,
        message: str,
        centers: tuple[Landmark, Landmark] | None = None,
        activation_progress: float = 0.0,
        distance_delta: float = 0.0,
        angle_delta: float = 0.0,
        midpoint_delta_x: float = 0.0,
        midpoint_delta_y: float = 0.0,
        left_velocity_x: float = 0.0,
        left_velocity_y: float = 0.0,
        right_velocity_x: float = 0.0,
        right_velocity_y: float = 0.0,
        gesture: str = "Idle",
        orbit_x: float = 0.0,
        orbit_y: float = 0.0,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        zoom: float = 0.0,
        roll: float = 0.0,
        calibration_active: bool = False,
        calibration_completed: bool = False,
        calibration_succeeded: bool = False,
    ) -> NavigationSnapshot:
        if centers is None and pair is not None:
            centers = (hand_center(pair[0]), hand_center(pair[1]))
        left_point = centers[0] if centers else None
        right_point = centers[1] if centers else None
        midpoint = midpoint_between_hands(left_point, right_point)
        distance = distance_between_hands(left_point, right_point) if midpoint else None
        angle = angle_between_hands(left_point, right_point) if midpoint else None
        return NavigationSnapshot(
            state=state,
            enabled=self._settings.navigation_enabled,
            active=self._active and state is NavigationState.ACTIVE,
            hand_count=len(hands),
            left_hand=left_point,
            right_hand=right_point,
            midpoint=midpoint,
            distance=distance,
            distance_delta=distance_delta,
            angle=angle,
            angle_delta=angle_delta,
            midpoint_delta_x=midpoint_delta_x,
            midpoint_delta_y=midpoint_delta_y,
            left_velocity_x=left_velocity_x,
            left_velocity_y=left_velocity_y,
            right_velocity_x=right_velocity_x,
            right_velocity_y=right_velocity_y,
            confidence=confidence,
            activation_progress=activation_progress,
            gesture=gesture,
            orbit_x=orbit_x,
            orbit_y=orbit_y,
            pan_x=pan_x,
            pan_y=pan_y,
            zoom=zoom,
            roll=roll,
            calibration_active=calibration_active,
            calibration_completed=calibration_completed,
            calibration_succeeded=calibration_succeeded,
            target_mode=self._settings.navigation_mode,
            control_mode=self._settings.navigation_control_mode,
            message=message,
            neutral_midpoint=self._neutral_midpoint,
            neutral_distance=self._neutral_distance,
            neutral_angle=self._neutral_angle,
        )


def hand_center(hand: HandDetection) -> Landmark:
    """Return a stable palm center from wrist and MCP landmarks."""

    indices = (0, 5, 9, 13, 17)
    available = [hand.landmarks[index] for index in indices if len(hand.landmarks) > index]
    if not available:
        return Landmark(0.0, 0.0, 0.0)
    count = len(available)
    return Landmark(
        x=sum(point.x for point in available) / count,
        y=sum(point.y for point in available) / count,
        z=sum(point.z for point in available) / count,
    )


def distance_between_hands(
    left: Landmark | None,
    right: Landmark | None,
) -> float:
    """Return the normalized 2D distance between two hand centers."""

    if left is None or right is None:
        return 0.0
    return math.hypot(right.x - left.x, right.y - left.y)


def midpoint_between_hands(
    left: Landmark | None,
    right: Landmark | None,
) -> Landmark | None:
    """Return the midpoint of two hand centers."""

    if left is None or right is None:
        return None
    return Landmark(
        x=(left.x + right.x) / 2.0,
        y=(left.y + right.y) / 2.0,
        z=(left.z + right.z) / 2.0,
    )


def angle_between_hands(
    left: Landmark | None,
    right: Landmark | None,
) -> float | None:
    """Return the line angle in radians using ``atan2(right-left)``."""

    if left is None or right is None:
        return None
    return math.atan2(right.y - left.y, right.x - left.x)


def movement_delta(current: Landmark, previous: Landmark) -> Landmark:
    """Return a frame-to-frame movement delta."""

    return Landmark(current.x - previous.x, current.y - previous.y, current.z - previous.z)


def angle_delta_between(current: float, previous: float) -> float:
    """Return the shortest signed angular difference in ``[-pi, pi)``."""

    return _wrap_angle(current - previous)


def apply_dead_zone(value: float, threshold: float) -> float:
    """Return zero for small noise without creating a slow drift."""

    return 0.0 if abs(value) < max(0.0, threshold) else value


def hand_is_open(hand: HandDetection) -> bool:
    """Use four radial finger-extension tests for the activation pose."""

    tip_indices = (8, 12, 16, 20)
    pip_indices = (6, 10, 14, 18)
    if len(hand.landmarks) <= max(tip_indices):
        return False
    wrist = hand.landmarks[0]
    extended = 0
    for tip_index, pip_index in zip(tip_indices, pip_indices):
        tip_distance = distance_between_hands(hand.landmarks[tip_index], wrist)
        pip_distance = distance_between_hands(hand.landmarks[pip_index], wrist)
        if tip_distance > pip_distance * TwoHandNavigation.OPEN_EXTENSION_RATIO:
            extended += 1
    return extended >= TwoHandNavigation.OPEN_FINGER_COUNT


def _activation_pose(
    pair: tuple[HandDetection, HandDetection],
    gesture: str,
) -> bool:
    if gesture == "Two closed hands":
        return all(not hand_is_open(hand) for hand in pair)
    return all(hand_is_open(hand) for hand in pair)


def _deactivation_pose(
    pair: tuple[HandDetection, HandDetection],
    gesture: str,
) -> bool:
    if gesture == "Two closed hands":
        return all(not hand_is_open(hand) for hand in pair)
    if gesture == "Two open hands":
        return all(hand_is_open(hand) for hand in pair)
    return False


def _ordered_pair(
    hands: list[HandDetection],
    previous: _PairFeatures | None = None,
) -> tuple[HandDetection, HandDetection] | None:
    """Return a stable left/right pair, preferring MediaPipe handedness."""

    valid = [hand for hand in hands if len(hand.landmarks) >= 18]
    if len(valid) < 2:
        return None
    if len(valid) > 2:
        valid = sorted(
            valid,
            key=lambda hand: hand.confidence if hand.confidence is not None else 1.0,
            reverse=True,
        )[:2]

    labelled_left = [
        hand for hand in valid if (hand.handedness or "").casefold() == "left"
    ]
    labelled_right = [
        hand for hand in valid if (hand.handedness or "").casefold() == "right"
    ]
    if len(labelled_left) == 1 and len(labelled_right) == 1:
        return labelled_left[0], labelled_right[0]

    by_x = sorted(valid, key=lambda hand: hand_center(hand).x)
    if previous is None:
        return by_x[0], by_x[1]

    # If handedness is unavailable or momentarily inconsistent, select the
    # assignment with the smallest movement from the previous frame.
    first, second = by_x[0], by_x[1]
    first_center = hand_center(first)
    second_center = hand_center(second)
    normal_cost = _point_distance(first_center, previous.left) + _point_distance(
        second_center, previous.right
    )
    swapped_cost = _point_distance(first_center, previous.right) + _point_distance(
        second_center, previous.left
    )
    return (first, second) if normal_cost <= swapped_cost else (second, first)


def _pair_confidence(
    pair: tuple[HandDetection, HandDetection] | None,
) -> float:
    if pair is None:
        return 0.0
    values = [hand.confidence for hand in pair if hand.confidence is not None]
    return max(0.0, min(1.0, min(values) if values else 1.0))


def _features(
    pair: tuple[HandDetection, HandDetection],
    centers: tuple[Landmark, Landmark] | None = None,
) -> _PairFeatures:
    left, right = centers or (hand_center(pair[0]), hand_center(pair[1]))
    midpoint = midpoint_between_hands(left, right)
    assert midpoint is not None
    return _PairFeatures(
        left=left,
        right=right,
        midpoint=midpoint,
        distance=distance_between_hands(left, right),
        angle=angle_between_hands(left, right) or 0.0,
    )


def _point_distance(first: Landmark, second: Landmark) -> float:
    return distance_between_hands(first, second)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _limit(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(-maximum, min(maximum, value))


def _gesture_name(
    orbit_x: float,
    orbit_y: float,
    pan_x: float,
    pan_y: float,
    zoom: float,
    roll: float,
) -> str:
    values = {
        "Orbit": max(abs(orbit_x), abs(orbit_y)),
        "Pan": max(abs(pan_x), abs(pan_y)),
        "Zoom": abs(zoom),
        "Roll": abs(roll),
    }
    active = [name for name, value in values.items() if value > 0.0]
    if not active:
        return "Idle"
    return " + ".join(active)


def _circular_median(values: list[float]) -> float:
    if not values:
        return 0.0
    reference = values[0]
    unwrapped = [reference + _wrap_angle(value - reference) for value in values]
    return median(unwrapped)


def _point_payload(point: Landmark | None) -> dict[str, float] | None:
    if point is None:
        return None
    return {"x": point.x, "y": point.y, "z": point.z}


# Compatibility aliases make the core formulas easy to discover in tests and
# in future integrations without exposing the class internals.
calculate_distance = distance_between_hands
calculate_midpoint = midpoint_between_hands
calculate_angle = angle_between_hands
calculate_distance_delta = lambda current, previous: current - previous
calculate_movement_delta = movement_delta
calculate_angle_delta = angle_delta_between
