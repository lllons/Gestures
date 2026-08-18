"""Two-hand gesture interpretation for universal 3D navigation.

The module deliberately has no GUI, socket, desktop-input, or application
integration dependencies. It consumes the existing MediaPipe ``HandDetection``
values and produces a per-frame analog motion snapshot for a generic OS input
adapter.

The signal path is intentionally explicit:

``raw hand centers -> confidence/outlier checks -> One Euro filters ->
frame-rate independent velocities -> dead zones/hysteresis -> response curves``

Translation is derived from an independently filtered midpoint, while zoom is
derived from an independently filtered hand distance. This keeps a symmetric
hand spread from accidentally orbiting the focused application and keeps a
still pair from slowly drifting the mouse.
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


DEFAULT_NAVIGATION_DEAD_ZONE = 0.004


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
    from slowly moving the focused 3D view.
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
    pan_pose: bool = False
    pose: str = "Orbit pose"
    calibration_active: bool = False
    calibration_completed: bool = False
    calibration_succeeded: bool = False
    profile_name: str = "Generic 3D"
    control_mode: str = "FULL 3D"
    message: str = ""
    neutral_midpoint: Landmark | None = None
    neutral_distance: float | None = None
    neutral_angle: float | None = None

    # Diagnostic values. ``left_hand``/``midpoint``/``distance`` are filtered;
    # these fields make it possible to compare the raw and stable signals in
    # the GUI without changing the control path.
    raw_left_hand: Landmark | None = None
    raw_right_hand: Landmark | None = None
    raw_midpoint: Landmark | None = None
    raw_distance: float | None = None
    midpoint_velocity_x: float = 0.0
    midpoint_velocity_y: float = 0.0
    distance_velocity: float = 0.0
    smoothing_amount: float = 0.0
    confidence_gain: float = 1.0
    dead_zone_active: bool = False
    outlier_rejected: bool = False
    hand_loss_frames: int = 0
    orbit_lock_active: bool = False
    orbit_lock_hands: tuple[str, ...] = ()
    left_ring_thumb_distance: float | None = None
    right_ring_thumb_distance: float | None = None
    left_ring_thumb_touch: bool = False
    right_ring_thumb_touch: bool = False

    @property
    def orbit_lock_hand(self) -> str:
        """Return the user-facing hand label for the active lock."""

        if not self.orbit_lock_active or not self.orbit_lock_hands:
            return "NONE"
        if len(self.orbit_lock_hands) > 1:
            return "BOTH"
        return self.orbit_lock_hands[0]

    @property
    def hands_ready(self) -> bool:
        return (
            self.hand_count == 2
            and self.left_hand is not None
            and self.right_hand is not None
        )

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible, application-independent debug packet."""

        return {
            "type": "gestures_navigation",
            "version": 3,
            "state": self.state.value,
            "enabled": self.enabled,
            "profile": self.profile_name,
            "control_mode": self.control_mode,
            "active": self.active,
            "hands": self.hand_count,
            "left_hand": _point_payload(self.left_hand),
            "right_hand": _point_payload(self.right_hand),
            "midpoint": _point_payload(self.midpoint),
            "raw_left_hand": _point_payload(self.raw_left_hand),
            "raw_right_hand": _point_payload(self.raw_right_hand),
            "raw_midpoint": _point_payload(self.raw_midpoint),
            "distance": self.distance,
            "raw_distance": self.raw_distance,
            "distance_delta": self.distance_delta,
            "distance_velocity": self.distance_velocity,
            "angle": self.angle,
            "angle_delta": self.angle_delta,
            "midpoint_delta": {
                "x": self.midpoint_delta_x,
                "y": self.midpoint_delta_y,
            },
            "midpoint_velocity": {
                "x": self.midpoint_velocity_x,
                "y": self.midpoint_velocity_y,
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
            "pan_pose": self.pan_pose,
            "pose": self.pose,
            "confidence": self.confidence,
            "confidence_gain": self.confidence_gain,
            "activation_progress": self.activation_progress,
            "smoothing": self.smoothing_amount,
            "dead_zone_active": self.dead_zone_active,
            "outlier_rejected": self.outlier_rejected,
            "hand_loss_frames": self.hand_loss_frames,
            "orbit_lock": {
                "active": self.orbit_lock_active,
                "hands": list(self.orbit_lock_hands),
                "activating_hand": self.orbit_lock_hand,
                "left_distance": self.left_ring_thumb_distance,
                "right_distance": self.right_ring_thumb_distance,
                "left_touch": self.left_ring_thumb_touch,
                "right_touch": self.right_ring_thumb_touch,
            },
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


@dataclass(frozen=True)
class _OrbitLockFrame:
    """The deterministic ring-thumb contact result for one camera frame."""

    active: bool
    hands: tuple[str, ...]
    sources: dict[str, HandDetection]
    left_distance: float | None
    right_distance: float | None
    left_touch: bool
    right_touch: bool


class OneEuroFilter:
    """Low-latency adaptive low-pass filter for one scalar signal.

    The One Euro filter raises its cutoff while the signal is moving quickly
    and lowers it while the signal is nearly still. It therefore removes slow
    landmark noise without adding the fixed latency of a large moving average.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.08,
        derivative_cutoff: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.min_cutoff = max(0.05, min_cutoff)
        self.beta = max(0.0, beta)
        self.derivative_cutoff = max(0.05, derivative_cutoff)
        self.enabled = enabled
        self._raw_previous: float | None = None
        self._filtered: float | None = None
        self._derivative: float = 0.0
        self._time: float | None = None

    def configure(
        self,
        *,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
        enabled: bool,
    ) -> None:
        self.min_cutoff = max(0.05, min_cutoff)
        self.beta = max(0.0, beta)
        self.derivative_cutoff = max(0.05, derivative_cutoff)
        self.enabled = enabled

    def reset(self) -> None:
        self._raw_previous = None
        self._filtered = None
        self._derivative = 0.0
        self._time = None

    def filter(self, value: float, timestamp: float) -> float:
        if not math.isfinite(value):
            return self._filtered if self._filtered is not None else 0.0
        if self._time is None or self._filtered is None or self._raw_previous is None:
            self._raw_previous = value
            self._filtered = value
            self._time = timestamp
            self._derivative = 0.0
            return value

        dt = max(1.0 / 240.0, min(0.25, timestamp - self._time))
        if timestamp <= self._time:
            timestamp = self._time + dt

        # ``enabled`` is an explicit bypass, not merely a request to disable
        # the adaptive beta. This keeps the advanced setting predictable while
        # preserving a fresh baseline when the filter is enabled again.
        if not self.enabled:
            self._raw_previous = value
            self._filtered = value
            self._derivative = 0.0
            self._time = timestamp
            return value

        raw_derivative = (value - self._raw_previous) / dt
        derivative_alpha = _low_pass_alpha(self.derivative_cutoff, dt)
        self._derivative += derivative_alpha * (raw_derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        alpha = _low_pass_alpha(cutoff, dt)
        self._filtered += alpha * (value - self._filtered)
        self._raw_previous = value
        self._time = timestamp
        return self._filtered


class _AngleFilter:
    """Filter angles after unwrapping them so +/- pi is not a false jump."""

    def __init__(self) -> None:
        self._filter = OneEuroFilter()
        self._unwrapped: float | None = None

    def configure(self, **kwargs: Any) -> None:
        self._filter.configure(**kwargs)

    def reset(self) -> None:
        self._filter.reset()
        self._unwrapped = None

    def filter(self, value: float, timestamp: float) -> float:
        if self._unwrapped is None:
            self._unwrapped = value
        else:
            self._unwrapped += _wrap_angle(value - self._unwrapped)
        return _wrap_angle(self._filter.filter(self._unwrapped, timestamp))


class _PointFilter:
    """Three independent One Euro filters for a normalized landmark center."""

    def __init__(self) -> None:
        self._x = OneEuroFilter()
        self._y = OneEuroFilter()
        self._z = OneEuroFilter()

    def configure(self, **kwargs: Any) -> None:
        self._x.configure(**kwargs)
        self._y.configure(**kwargs)
        self._z.configure(**kwargs)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()
        self._z.reset()

    def filter(self, point: Landmark, timestamp: float) -> Landmark:
        return Landmark(
            self._x.filter(point.x, timestamp),
            self._y.filter(point.y, timestamp),
            self._z.filter(point.z, timestamp),
        )


class _PointSmoother:
    """Compatibility helper retained for callers that used the old smoother."""

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
    """Interpret two tracked hands as a safe, smooth continuous gesture."""

    CALIBRATION_SECONDS = 1.2
    CALIBRATION_TIMEOUT_SECONDS = 20.0
    MIN_CALIBRATION_SAMPLES = 8
    DEACTIVATION_HOLD_SECONDS = 0.35
    OPEN_FINGER_COUNT = 3
    OPEN_EXTENSION_RATIO = 1.08

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._left_filter = _PointFilter()
        self._right_filter = _PointFilter()
        self._midpoint_filter = _PointFilter()
        self._distance_filter = OneEuroFilter()
        self._angle_filter = _AngleFilter()
        self._orbit_lock_filter = _PointFilter()
        self._configure_filters()

        self._active = False
        self._activation_started_at: float | None = None
        self._deactivation_started_at: float | None = None
        self._previous_features: _PairFeatures | None = None
        self._previous_time: float | None = None
        self._previous_raw_left: Landmark | None = None
        self._previous_raw_right: Landmark | None = None
        self._previous_raw_distance: float | None = None
        self._needs_rebaseline = False
        self._missed_frames = 0
        self._translation_engaged = False
        self._zoom_engaged = False
        self._roll_engaged = False
        self._orbit_lock_active = False
        self._orbit_lock_hands: tuple[str, ...] = ()
        self._orbit_lock_states = {"LEFT": False, "RIGHT": False}
        self._orbit_lock_source_centers: dict[str, Landmark] = {}
        self._orbit_lock_left_distance: float | None = None
        self._orbit_lock_right_distance: float | None = None
        self._orbit_lock_left_touch = False
        self._orbit_lock_right_touch = False
        self._orbit_lock_previous_point: Landmark | None = None
        self._orbit_lock_previous_time: float | None = None
        self._orbit_lock_just_released = False
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
        self._configure_filters()
        if not settings.navigation_enabled:
            self.reset()
        elif not settings.navigation_orbit_lock_enabled:
            # Disabling the optional lock must not deactivate ordinary 3D
            # navigation; it only clears the lock's private state.
            self._clear_orbit_lock()

    def reset(self) -> None:
        """Stop navigation and clear temporal movement state."""

        self._active = False
        self._activation_started_at = None
        self._deactivation_started_at = None
        self._calibration_active = False
        self._calibration_samples.clear()
        self._clear_orbit_lock()
        self._clear_tracking_baseline()
        # Keep a completed neutral calibration; reset is also used for camera
        # reconnects and should not discard the user's saved reference.

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
        self._calibration_active = True
        self._calibration_started_at = current_time
        self._calibration_samples.clear()
        self._clear_tracking_baseline()

    def process(
        self,
        hands: list[HandDetection],
        now: float | None = None,
    ) -> NavigationSnapshot:
        """Interpret one frame without blocking the camera worker."""

        current_time = time.monotonic() if now is None else now
        orbit_lock_frame = (
            self._update_orbit_lock(hands)
            if self._settings.navigation_enabled
            and self._settings.navigation_orbit_lock_enabled
            else self._clear_orbit_lock()
        )
        pair = _ordered_pair(hands, self._previous_features)
        confidence = _pair_confidence(pair) if pair is not None else _hands_confidence(hands)

        if pair is None:
            if orbit_lock_frame.active:
                return self._process_orbit_lock(
                    orbit_lock_frame,
                    hands,
                    confidence,
                    current_time,
                )
            return self._process_missing(hands, confidence, current_time)

        self._missed_frames = 0
        raw_left = hand_center(pair[0])
        raw_right = hand_center(pair[1])
        raw_left, raw_right, outlier_rejected = self._reject_outlier(raw_left, raw_right)
        raw_centers = (raw_left, raw_right)
        raw_midpoint = midpoint_between_hands(raw_left, raw_right)
        assert raw_midpoint is not None
        raw_distance = distance_between_hands(raw_left, raw_right)
        raw_distance_delta = (
            raw_distance - self._previous_raw_distance
            if self._previous_raw_distance is not None
            else None
        )
        raw_angle = angle_between_hands(raw_left, raw_right) or 0.0

        if self._needs_rebaseline:
            self._reset_filter_values()
            self._needs_rebaseline = False
            rebaselined = True
        else:
            rebaselined = False

        filtered_left = self._left_filter.filter(raw_left, current_time)
        filtered_right = self._right_filter.filter(raw_right, current_time)
        filtered_midpoint = self._midpoint_filter.filter(raw_midpoint, current_time)
        filtered_distance = self._distance_filter.filter(raw_distance, current_time)
        filtered_angle = self._angle_filter.filter(raw_angle, current_time)
        features = _PairFeatures(
            left=filtered_left,
            right=filtered_right,
            midpoint=filtered_midpoint,
            distance=filtered_distance,
            angle=filtered_angle,
        )
        self._previous_raw_left = raw_left
        self._previous_raw_right = raw_right
        self._previous_raw_distance = raw_distance

        if not self._settings.navigation_enabled:
            self.reset()
            return self._snapshot(
                state=NavigationState.DISABLED,
                hands=hands,
                pair=pair,
                centers=raw_centers,
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                confidence=confidence,
                outlier_rejected=outlier_rejected,
                message="3D navigation is disabled",
            )

        if self._calibration_active:
            return self._process_calibration(
                pair,
                features,
                raw_centers,
                raw_midpoint,
                raw_distance,
                hands,
                confidence,
                current_time,
                outlier_rejected,
            )

        if orbit_lock_frame.active:
            return self._process_orbit_lock(
                orbit_lock_frame,
                hands,
                confidence,
                current_time,
                pair=pair,
                features=features,
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                outlier_rejected=outlier_rejected,
            )

        if confidence < self._settings.navigation_min_confidence:
            was_active = self._active
            self._active = False
            self._activation_started_at = None
            self._deactivation_started_at = None
            self._clear_tracking_baseline()
            return self._snapshot(
                state=NavigationState.LOST if was_active else NavigationState.IDLE,
                hands=hands,
                pair=None,
                confidence=confidence,
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                outlier_rejected=outlier_rejected,
                message="Tracking confidence low; navigation stopped",
            )

        if rebaselined and self._orbit_lock_just_released:
            self._previous_features = features
            self._previous_time = current_time
            self._translation_engaged = False
            self._zoom_engaged = False
            self._roll_engaged = False
            return self._snapshot(
                state=NavigationState.ACTIVE if self._active else NavigationState.IDLE,
                hands=hands,
                pair=pair,
                centers=(features.left, features.right),
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                outlier_rejected=outlier_rejected,
                message="Orbit lock released; movement baseline reset",
            )

        if rebaselined and self._active:
            self._previous_features = features
            self._previous_time = current_time
            self._translation_engaged = False
            self._zoom_engaged = False
            self._roll_engaged = False
            return self._snapshot(
                state=NavigationState.ACTIVE,
                hands=hands,
                pair=pair,
                centers=(features.left, features.right),
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                confidence=confidence,
                outlier_rejected=outlier_rejected,
                pan_pose=_pan_pose(pair, self._settings.navigation_pan_gesture),
                pose=(
                    "Pan pose"
                    if _pan_pose(pair, self._settings.navigation_pan_gesture)
                    else "Orbit pose"
                ),
                message="Tracking reacquired; movement baseline reset",
            )

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
                    self._previous_features = features
                    self._previous_time = current_time
                    self._translation_engaged = False
                    self._zoom_engaged = False
                    self._roll_engaged = False
                    return self._snapshot(
                        state=NavigationState.ACTIVE,
                        hands=hands,
                        pair=pair,
                        centers=(features.left, features.right),
                        raw_centers=raw_centers,
                        raw_midpoint=raw_midpoint,
                        raw_distance=raw_distance,
                        confidence=confidence,
                        activation_progress=1.0,
                        outlier_rejected=outlier_rejected,
                        gesture="Idle",
                        pan_pose=_pan_pose(pair, self._settings.navigation_pan_gesture),
                        pose=(
                            "Pan pose"
                            if _pan_pose(pair, self._settings.navigation_pan_gesture)
                            else "Orbit pose"
                        ),
                        message="3D navigation active",
                    )
                return self._snapshot(
                    state=NavigationState.ACTIVATING,
                    hands=hands,
                    pair=pair,
                    centers=(features.left, features.right),
                    raw_centers=raw_centers,
                    raw_midpoint=raw_midpoint,
                    raw_distance=raw_distance,
                    confidence=confidence,
                    activation_progress=progress,
                    outlier_rejected=outlier_rejected,
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
                centers=(features.left, features.right),
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                confidence=confidence,
                outlier_rejected=outlier_rejected,
                message=(
                    f"Hold {self._settings.navigation_activation_gesture.lower()} to activate"
                ),
            )

        if _deactivation_pose(pair, self._settings.navigation_deactivation_gesture):
            if self._deactivation_started_at is None:
                self._deactivation_started_at = current_time
            elapsed = current_time - self._deactivation_started_at
            self._previous_features = features
            self._previous_time = current_time
            self._translation_engaged = False
            self._zoom_engaged = False
            self._roll_engaged = False
            if elapsed >= self.DEACTIVATION_HOLD_SECONDS:
                self.reset()
                return self._snapshot(
                    state=NavigationState.IDLE,
                    hands=hands,
                    pair=pair,
                    centers=(features.left, features.right),
                    raw_centers=raw_centers,
                    raw_midpoint=raw_midpoint,
                    raw_distance=raw_distance,
                    confidence=confidence,
                    outlier_rejected=outlier_rejected,
                    message="3D navigation deactivated",
                )
            return self._snapshot(
                state=NavigationState.ACTIVE,
                hands=hands,
                pair=pair,
                centers=(features.left, features.right),
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                confidence=confidence,
                outlier_rejected=outlier_rejected,
                gesture="Deactivating",
                message="Release or keep holding to deactivate",
            )
        self._deactivation_started_at = None

        return self._active_snapshot(
            features,
            hands,
            pair,
            confidence,
            current_time,
            raw_centers,
            raw_midpoint,
            raw_distance,
            raw_distance_delta,
            outlier_rejected,
        )

    def _configure_filters(self) -> None:
        general = max(0.0, min(1.0, self._settings.navigation_smoothing))
        # Keep the midpoint independently filtered instead of deriving it from
        # two already-noisy outputs. Orbit/pan controls share this stable signal.
        midpoint_smoothing = (
            self._settings.navigation_orbit_smoothing
            + self._settings.navigation_pan_smoothing
        ) / 2.0
        self._left_filter.configure(**_filter_parameters(self._settings, general))
        self._right_filter.configure(**_filter_parameters(self._settings, general))
        self._midpoint_filter.configure(
            **_filter_parameters(self._settings, midpoint_smoothing)
        )
        self._distance_filter.configure(
            **_filter_parameters(self._settings, self._settings.navigation_zoom_smoothing)
        )
        self._angle_filter.configure(
            **_filter_parameters(self._settings, self._settings.navigation_orbit_smoothing)
        )
        self._orbit_lock_filter.configure(
            **_filter_parameters(self._settings, self._settings.navigation_orbit_smoothing)
        )

    def _process_orbit_lock(
        self,
        lock: _OrbitLockFrame,
        hands: list[HandDetection],
        confidence: float,
        current_time: float,
        *,
        pair: tuple[HandDetection, HandDetection] | None = None,
        features: _PairFeatures | None = None,
        raw_centers: tuple[Landmark, Landmark] | None = None,
        raw_midpoint: Landmark | None = None,
        raw_distance: float | None = None,
        outlier_rejected: bool = False,
    ) -> NavigationSnapshot:
        """Generate orbit-only relative motion while contact is held.

        This path deliberately bypasses the normal two-hand activation pose and
        control-mode classifier. The contact rule is already the complete
        gesture decision, so one contacting hand can start orbiting immediately.
        """

        control_point = _orbit_lock_control_point(lock)
        if control_point is None:
            return self._process_missing(hands, confidence, current_time)

        filtered_point = self._orbit_lock_filter.filter(control_point, current_time)
        previous = self._orbit_lock_previous_point
        previous_time = self._orbit_lock_previous_time
        self._orbit_lock_previous_point = filtered_point
        self._orbit_lock_previous_time = current_time
        if features is not None:
            # Keep the general analyzer's baseline current while the lock owns
            # output, so releasing the lock cannot produce a catch-up jump.
            self._previous_features = features
            self._previous_time = current_time
            self._translation_engaged = False
            self._zoom_engaged = False
            self._roll_engaged = False

        orbit_x = 0.0
        orbit_y = 0.0
        velocity_x = 0.0
        velocity_y = 0.0
        confidence_gain = confidence_control_gain(
            confidence,
            self._settings.navigation_min_confidence,
        )
        if previous is not None and previous_time is not None:
            dt = _frame_delta(current_time, previous_time)
            velocity_x = (filtered_point.x - previous.x) / dt
            velocity_y = (filtered_point.y - previous.y) / dt
            velocity_x *= confidence_gain
            velocity_y *= confidence_gain
            dead_zone = _channel_dead_zone(
                self._settings.navigation_dead_zone,
                self._settings.navigation_orbit_dead_zone,
            ) * 60.0
            if math.hypot(velocity_x, velocity_y) < max(
                dead_zone,
                self._settings.navigation_motion_stop_threshold,
            ):
                velocity_x = velocity_y = 0.0
            direction_x = -1.0 if self._settings.navigation_invert_x else 1.0
            direction_y = -1.0 if self._settings.navigation_invert_y else 1.0
            orbit_x, orbit_y = _vector_response(
                velocity_x * direction_x,
                -velocity_y * direction_y,
                self._settings.navigation_orbit_sensitivity,
                self._settings.navigation_orbit_acceleration,
                self._settings.navigation_orbit_max_speed,
                dt,
                confidence_gain,
            )

        tracking_ok = confidence >= self._settings.navigation_min_confidence
        state = NavigationState.ACTIVE if tracking_ok else NavigationState.LOST
        if not tracking_ok:
            orbit_x = orbit_y = 0.0
        left_source = lock.sources.get("LEFT")
        right_source = lock.sources.get("RIGHT")
        left_point = hand_center(left_source) if left_source is not None else None
        right_point = hand_center(right_source) if right_source is not None else None
        message = (
            f"ORBIT LOCK: ACTIVE ({_orbit_lock_hand_label(lock.hands)})"
            if tracking_ok
            else "Orbit lock stopped; tracking confidence low"
        )
        return self._snapshot(
            state=state,
            hands=hands,
            pair=pair,
            left_hand_point=left_point,
            right_hand_point=right_point,
            raw_centers=raw_centers,
            raw_midpoint=raw_midpoint,
            raw_distance=raw_distance,
            filtered_distance=features.distance if features else None,
            filtered_angle=features.angle if features else None,
            confidence=confidence,
            confidence_gain=confidence_gain,
            midpoint_velocity_x=velocity_x,
            midpoint_velocity_y=velocity_y,
            left_velocity_x=velocity_x,
            left_velocity_y=velocity_y,
            right_velocity_x=velocity_x,
            right_velocity_y=velocity_y,
            orbit_x=orbit_x,
            orbit_y=orbit_y,
            gesture="Orbit Lock" if tracking_ok else "Idle",
            smoothing_amount=_effective_smoothing(self._settings),
            dead_zone_active=orbit_x == 0.0 and orbit_y == 0.0,
            outlier_rejected=outlier_rejected,
            message=message,
        )

    def _update_orbit_lock(self, hands: list[HandDetection]) -> _OrbitLockFrame:
        sources = _orbit_lock_sources(hands, self._orbit_lock_source_centers)
        previous_active = self._orbit_lock_active
        previous_hands = self._orbit_lock_hands
        activation_threshold = self._settings.navigation_orbit_lock_activation_threshold
        release_threshold = self._settings.navigation_orbit_lock_release_threshold
        for side in ("LEFT", "RIGHT"):
            source = sources.get(side)
            distance = ring_thumb_distance(source) if source is not None else None
            was_touching = self._orbit_lock_states.get(side, False)
            touching = (
                distance is not None
                and distance
                <= (release_threshold if was_touching else activation_threshold) + 1e-9
            )
            self._orbit_lock_states[side] = touching
            if side == "LEFT":
                self._orbit_lock_left_distance = distance
                self._orbit_lock_left_touch = touching
            else:
                self._orbit_lock_right_distance = distance
                self._orbit_lock_right_touch = touching
            if source is not None:
                self._orbit_lock_source_centers[side] = hand_center(source)

        active_hands = tuple(
            side for side in ("LEFT", "RIGHT") if self._orbit_lock_states.get(side, False)
        )
        self._orbit_lock_active = bool(active_hands)
        self._orbit_lock_hands = active_hands
        self._orbit_lock_just_released = previous_active and not self._orbit_lock_active
        if self._orbit_lock_active and (
            not previous_active or active_hands != previous_hands
        ):
            self._orbit_lock_filter.reset()
            self._orbit_lock_previous_point = None
            self._orbit_lock_previous_time = None
        elif self._orbit_lock_just_released:
            self._orbit_lock_filter.reset()
            self._orbit_lock_previous_point = None
            self._orbit_lock_previous_time = None
            self._needs_rebaseline = True
        return _OrbitLockFrame(
            active=self._orbit_lock_active,
            hands=active_hands,
            sources=dict(sources),
            left_distance=self._orbit_lock_left_distance,
            right_distance=self._orbit_lock_right_distance,
            left_touch=self._orbit_lock_left_touch,
            right_touch=self._orbit_lock_right_touch,
        )

    def _clear_orbit_lock(self) -> _OrbitLockFrame:
        self._orbit_lock_active = False
        self._orbit_lock_hands = ()
        self._orbit_lock_states = {"LEFT": False, "RIGHT": False}
        self._orbit_lock_source_centers.clear()
        self._orbit_lock_left_distance = None
        self._orbit_lock_right_distance = None
        self._orbit_lock_left_touch = False
        self._orbit_lock_right_touch = False
        self._orbit_lock_previous_point = None
        self._orbit_lock_previous_time = None
        self._orbit_lock_filter.reset()
        self._orbit_lock_just_released = False
        return _OrbitLockFrame(False, (), {}, None, None, False, False)

    def _reject_outlier(
        self,
        left: Landmark,
        right: Landmark,
    ) -> tuple[Landmark, Landmark, bool]:
        threshold = self._settings.navigation_outlier_threshold
        rejected = False
        if self._previous_raw_left is not None and _point_distance(left, self._previous_raw_left) > threshold:
            left = self._previous_raw_left
            rejected = True
        if self._previous_raw_right is not None and _point_distance(right, self._previous_raw_right) > threshold:
            right = self._previous_raw_right
            rejected = True
        return left, right, rejected

    def _process_missing(
        self,
        hands: list[HandDetection],
        confidence: float,
        current_time: float,
    ) -> NavigationSnapshot:
        del current_time
        self._missed_frames += 1
        self._needs_rebaseline = True
        self._previous_raw_left = None
        self._previous_raw_right = None
        self._previous_raw_distance = None
        was_active = self._active
        grace = self._settings.navigation_hand_loss_grace_frames
        if was_active and self._missed_frames <= grace:
            # The packet is neutral and the input controller releases buttons
            # because the hand count is not two. Keep the logical session alive
            # so a one-frame MediaPipe miss can resume without reactivation.
            return self._snapshot(
                state=NavigationState.LOST,
                hands=hands,
                pair=None,
                confidence=confidence,
                hand_loss_frames=self._missed_frames,
                message=(
                    f"Tracking gap ({self._missed_frames}/{grace}); simulated input released"
                ),
            )

        if was_active:
            self._active = False
        self._activation_started_at = None
        self._deactivation_started_at = None
        self._translation_engaged = False
        self._zoom_engaged = False
        self._roll_engaged = False
        loss_frames = self._missed_frames
        self._clear_tracking_baseline()
        return self._snapshot(
            state=NavigationState.LOST if was_active else NavigationState.IDLE,
            hands=hands,
            pair=None,
            confidence=confidence,
            hand_loss_frames=loss_frames,
            message=(
                "Hands lost; navigation stopped"
                if was_active
                else "Show both hands"
            ),
        )

    def _process_calibration(
        self,
        pair: tuple[HandDetection, HandDetection] | None,
        features: _PairFeatures | None,
        raw_centers: tuple[Landmark, Landmark] | None,
        raw_midpoint: Landmark | None,
        raw_distance: float | None,
        hands: list[HandDetection],
        confidence: float,
        current_time: float,
        outlier_rejected: bool,
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
                centers=(features.left, features.right) if features else None,
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                confidence=confidence,
                outlier_rejected=outlier_rejected,
                calibration_completed=True,
                calibration_succeeded=False,
                message="Calibration timed out; show both hands and try again",
            )

        if (
            pair is not None
            and features is not None
            and confidence >= self._settings.navigation_min_confidence
        ):
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
                    centers=(features.left, features.right),
                    raw_centers=raw_centers,
                    raw_midpoint=raw_midpoint,
                    raw_distance=raw_distance,
                    confidence=confidence,
                    outlier_rejected=outlier_rejected,
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
            centers=(features.left, features.right) if features else None,
            raw_centers=raw_centers,
            raw_midpoint=raw_midpoint,
            raw_distance=raw_distance,
            confidence=confidence,
            outlier_rejected=outlier_rejected,
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
        raw_centers: tuple[Landmark, Landmark],
        raw_midpoint: Landmark,
        raw_distance: float,
        raw_distance_delta: float | None,
        outlier_rejected: bool,
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
                raw_centers=raw_centers,
                raw_midpoint=raw_midpoint,
                raw_distance=raw_distance,
                confidence=confidence,
                outlier_rejected=outlier_rejected,
                message="3D navigation active",
            )

        dt = _frame_delta(current_time, previous_time)
        midpoint_dx = features.midpoint.x - previous.midpoint.x
        midpoint_dy = features.midpoint.y - previous.midpoint.y
        distance_delta = features.distance - previous.distance
        angle_delta = angle_delta_between(features.angle, previous.angle)
        midpoint_velocity_x = midpoint_dx / dt
        midpoint_velocity_y = midpoint_dy / dt
        distance_velocity = distance_delta / dt
        angle_velocity = angle_delta / dt

        confidence_gain = confidence_control_gain(
            confidence,
            self._settings.navigation_min_confidence,
        )
        midpoint_velocity_x *= confidence_gain
        midpoint_velocity_y *= confidence_gain
        distance_velocity *= confidence_gain
        angle_velocity *= confidence_gain

        orbit_dead_zone = _channel_dead_zone(
            self._settings.navigation_dead_zone,
            self._settings.navigation_orbit_dead_zone,
        ) * 60.0
        pan_dead_zone = _channel_dead_zone(
            self._settings.navigation_dead_zone,
            self._settings.navigation_pan_dead_zone,
        ) * 60.0
        zoom_dead_zone = _channel_dead_zone(
            self._settings.navigation_dead_zone,
            self._settings.navigation_zoom_dead_zone,
        ) * 60.0

        translation_speed = math.hypot(midpoint_velocity_x, midpoint_velocity_y)
        translation_signal = max(orbit_dead_zone, pan_dead_zone)
        if translation_speed < translation_signal:
            midpoint_velocity_x = midpoint_velocity_y = 0.0
            translation_speed = 0.0
        self._translation_engaged = hysteresis(
            self._translation_engaged,
            translation_speed,
            max(
                self._settings.navigation_motion_start_threshold,
                translation_signal,
            ),
            min(
                self._settings.navigation_motion_stop_threshold,
                max(
                    self._settings.navigation_motion_start_threshold,
                    translation_signal,
                ),
            ),
        )
        if not self._translation_engaged:
            midpoint_velocity_x = midpoint_velocity_y = 0.0

        # If the raw pair distance is stable, do not let filter catch-up after
        # an earlier spread gesture create a false zoom during translation.
        raw_distance_stable = (
            raw_distance_delta is not None
            and abs(raw_distance_delta) <= max(0.0005, zoom_dead_zone / 120.0)
        )
        if raw_distance_stable:
            distance_velocity = 0.0
        elif abs(distance_velocity) < zoom_dead_zone:
            distance_velocity = 0.0
        self._zoom_engaged = hysteresis(
            self._zoom_engaged,
            abs(distance_velocity),
            max(self._settings.navigation_zoom_start_threshold, zoom_dead_zone),
            min(
                self._settings.navigation_zoom_stop_threshold,
                max(self._settings.navigation_zoom_start_threshold, zoom_dead_zone),
            ),
        )
        if not self._zoom_engaged:
            distance_velocity = 0.0

        roll_dead_zone = max(0.01, orbit_dead_zone / 3.0)
        if abs(angle_velocity) < roll_dead_zone:
            angle_velocity = 0.0
        self._roll_engaged = hysteresis(
            self._roll_engaged,
            abs(angle_velocity),
            roll_dead_zone * 1.5,
            roll_dead_zone,
        )
        if not self._roll_engaged:
            angle_velocity = 0.0

        direction_x = -1.0 if self._settings.navigation_invert_x else 1.0
        direction_y = -1.0 if self._settings.navigation_invert_y else 1.0
        direction_zoom = -1.0 if self._settings.navigation_invert_zoom else 1.0
        orbit_x, orbit_y = _vector_response(
            midpoint_velocity_x * direction_x,
            -midpoint_velocity_y * direction_y,
            self._settings.navigation_orbit_sensitivity,
            self._settings.navigation_orbit_acceleration,
            self._settings.navigation_orbit_max_speed,
            dt,
            confidence_gain,
        )
        pan_x, pan_y = _vector_response(
            midpoint_velocity_x * direction_x,
            -midpoint_velocity_y * direction_y,
            self._settings.navigation_pan_sensitivity,
            self._settings.navigation_pan_acceleration,
            self._settings.navigation_pan_max_speed,
            dt,
            confidence_gain,
        )
        zoom = response_curve(
            distance_velocity * direction_zoom,
            self._settings.navigation_zoom_sensitivity,
            self._settings.navigation_zoom_acceleration,
            self._settings.navigation_zoom_max_speed,
            dt,
            confidence_gain,
        )
        roll = (
            response_curve(
                angle_velocity,
                self._settings.navigation_roll_sensitivity,
                self._settings.navigation_orbit_acceleration,
                max(0.1, self._settings.navigation_max_speed * math.pi),
                dt,
                confidence_gain,
            )
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

        pan_pose = _pan_pose(pair, self._settings.navigation_pan_gesture)
        gesture = _gesture_name(orbit_x, orbit_y, pan_x, pan_y, zoom, roll)
        if mode == "FULL 3D":
            labels: list[str] = []
            if max(abs(orbit_x), abs(orbit_y), abs(pan_x), abs(pan_y)) > 0.0:
                labels.append("Pan" if pan_pose else "Orbit")
            if abs(zoom) > 0.0:
                labels.append("Zoom")
            if abs(roll) > 0.0:
                labels.append("Roll")
            gesture = " + ".join(labels) if labels else "Idle"

        dead_zone_active = (
            not self._translation_engaged
            and not self._zoom_engaged
            and not self._roll_engaged
        )
        message = gesture if gesture != "Idle" else "3D navigation active"
        if outlier_rejected:
            message = "Outlier rejected; " + message

        return self._snapshot(
            state=NavigationState.ACTIVE,
            hands=hands,
            pair=pair,
            centers=(features.left, features.right),
            raw_centers=raw_centers,
            raw_midpoint=raw_midpoint,
            raw_distance=raw_distance,
            filtered_distance=features.distance,
            filtered_angle=features.angle,
            confidence=confidence,
            confidence_gain=confidence_gain,
            distance_delta=distance_delta,
            angle_delta=angle_delta,
            midpoint_delta_x=midpoint_dx,
            midpoint_delta_y=midpoint_dy,
            midpoint_velocity_x=midpoint_velocity_x,
            midpoint_velocity_y=midpoint_velocity_y,
            distance_velocity=distance_velocity,
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
            pan_pose=pan_pose,
            pose="Pan pose" if pan_pose else "Orbit pose",
            smoothing_amount=_effective_smoothing(self._settings),
            dead_zone_active=dead_zone_active,
            outlier_rejected=outlier_rejected,
            message=message,
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
        left_hand_point: Landmark | None = None,
        right_hand_point: Landmark | None = None,
        raw_centers: tuple[Landmark, Landmark] | None = None,
        raw_midpoint: Landmark | None = None,
        raw_distance: float | None = None,
        filtered_distance: float | None = None,
        filtered_angle: float | None = None,
        activation_progress: float = 0.0,
        distance_delta: float = 0.0,
        angle_delta: float = 0.0,
        midpoint_delta_x: float = 0.0,
        midpoint_delta_y: float = 0.0,
        midpoint_velocity_x: float = 0.0,
        midpoint_velocity_y: float = 0.0,
        distance_velocity: float = 0.0,
        left_velocity_x: float = 0.0,
        left_velocity_y: float = 0.0,
        right_velocity_x: float = 0.0,
        right_velocity_y: float = 0.0,
        confidence_gain: float = 1.0,
        gesture: str = "Idle",
        orbit_x: float = 0.0,
        orbit_y: float = 0.0,
        pan_x: float = 0.0,
        pan_y: float = 0.0,
        zoom: float = 0.0,
        roll: float = 0.0,
        pan_pose: bool = False,
        pose: str = "Orbit pose",
        smoothing_amount: float = 0.0,
        dead_zone_active: bool = False,
        outlier_rejected: bool = False,
        hand_loss_frames: int = 0,
        calibration_active: bool = False,
        calibration_completed: bool = False,
        calibration_succeeded: bool = False,
    ) -> NavigationSnapshot:
        if centers is None and pair is not None:
            centers = (hand_center(pair[0]), hand_center(pair[1]))
        if raw_centers is None and pair is not None:
            raw_centers = (hand_center(pair[0]), hand_center(pair[1]))
        left_point = left_hand_point if left_hand_point is not None else (centers[0] if centers else None)
        right_point = right_hand_point if right_hand_point is not None else (centers[1] if centers else None)
        midpoint = midpoint_between_hands(left_point, right_point)
        distance = (
            filtered_distance
            if filtered_distance is not None
            else (distance_between_hands(left_point, right_point) if midpoint else None)
        )
        angle = (
            filtered_angle
            if filtered_angle is not None
            else (angle_between_hands(left_point, right_point) if midpoint else None)
        )
        if raw_midpoint is None and raw_centers is not None:
            raw_midpoint = midpoint_between_hands(raw_centers[0], raw_centers[1])
        if raw_distance is None and raw_centers is not None:
            raw_distance = distance_between_hands(raw_centers[0], raw_centers[1])
        return NavigationSnapshot(
            state=state,
            enabled=self._settings.navigation_enabled,
            active=(self._active or self._orbit_lock_active) and state is NavigationState.ACTIVE,
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
            pan_pose=pan_pose,
            pose=pose,
            calibration_active=calibration_active,
            calibration_completed=calibration_completed,
            calibration_succeeded=calibration_succeeded,
            profile_name=self._settings.navigation_profile,
            control_mode=self._settings.navigation_control_mode,
            message=message,
            neutral_midpoint=self._neutral_midpoint,
            neutral_distance=self._neutral_distance,
            neutral_angle=self._neutral_angle,
            raw_left_hand=raw_centers[0] if raw_centers else None,
            raw_right_hand=raw_centers[1] if raw_centers else None,
            raw_midpoint=raw_midpoint,
            raw_distance=raw_distance,
            midpoint_velocity_x=midpoint_velocity_x,
            midpoint_velocity_y=midpoint_velocity_y,
            distance_velocity=distance_velocity,
            smoothing_amount=smoothing_amount,
            confidence_gain=confidence_gain,
            dead_zone_active=dead_zone_active,
            outlier_rejected=outlier_rejected,
            hand_loss_frames=hand_loss_frames,
            orbit_lock_active=self._orbit_lock_active,
            orbit_lock_hands=self._orbit_lock_hands,
            left_ring_thumb_distance=self._orbit_lock_left_distance,
            right_ring_thumb_distance=self._orbit_lock_right_distance,
            left_ring_thumb_touch=self._orbit_lock_left_touch,
            right_ring_thumb_touch=self._orbit_lock_right_touch,
        )

    def _clear_tracking_baseline(self) -> None:
        self._reset_filter_values()
        self._previous_features = None
        self._previous_time = None
        self._previous_raw_left = None
        self._previous_raw_right = None
        self._previous_raw_distance = None
        self._needs_rebaseline = False
        self._missed_frames = 0
        self._translation_engaged = False
        self._zoom_engaged = False
        self._roll_engaged = False

    def _reset_filter_values(self) -> None:
        self._left_filter.reset()
        self._right_filter.reset()
        self._midpoint_filter.reset()
        self._distance_filter.reset()
        self._angle_filter.reset()
        self._orbit_lock_filter.reset()


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


def ring_thumb_distance(hand: HandDetection | None) -> float | None:
    """Return the normalized 2D distance between thumb tip (4) and ring tip (16)."""

    if hand is None or len(hand.landmarks) <= 16:
        return None
    thumb = hand.landmarks[4]
    ring = hand.landmarks[16]
    return math.hypot(ring.x - thumb.x, ring.y - thumb.y)


def is_ring_thumb_touching(
    hand: HandDetection | None,
    threshold: float = 0.035,
) -> bool:
    """Return the deterministic activation result without retaining state."""

    distance = ring_thumb_distance(hand)
    return distance is not None and distance <= threshold + 1e-9


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


def confidence_control_gain(confidence: float, minimum: float) -> float:
    """Map confidence to a smooth 0..1 control gain."""

    if confidence <= minimum:
        return 0.0
    if minimum >= 1.0:
        return 1.0 if confidence >= 1.0 else 0.0
    normalized = max(0.0, min(1.0, (confidence - minimum) / (1.0 - minimum)))
    return normalized * normalized * (3.0 - 2.0 * normalized)


def hysteresis(
    currently_active: bool,
    magnitude: float,
    start_threshold: float,
    stop_threshold: float,
) -> bool:
    """Keep a gesture active until it falls below a lower stop threshold."""

    start = max(0.0, start_threshold)
    stop = max(0.0, min(stop_threshold, start))
    return magnitude >= (stop if currently_active else start)


def response_curve(
    velocity: float,
    sensitivity: float,
    acceleration: float,
    maximum_speed: float,
    delta_time: float,
    confidence: float = 1.0,
) -> float:
    """Convert normalized velocity to a bounded per-frame analog output.

    The response is linear near zero and gains smoothly as speed increases.
    ``delta_time`` is applied at the end, so equal physical velocities produce
    equal total movement at different camera frame rates.
    """

    if velocity == 0.0 or delta_time <= 0.0 or sensitivity <= 0.0 or confidence <= 0.0:
        return 0.0
    maximum = max(0.001, maximum_speed)
    limited = _limit(velocity, maximum)
    normalized = min(1.0, abs(limited) / maximum)
    gain = 1.0 + max(0.0, acceleration) * normalized * normalized
    # Acceleration may make the response reach its ceiling sooner, but it must
    # never defeat the configured maximum-speed safety cap.
    accelerated = _limit(limited * gain, maximum)
    return accelerated * sensitivity * delta_time * max(0.0, min(1.0, confidence))


def _vector_response(
    x: float,
    y: float,
    sensitivity: float,
    acceleration: float,
    maximum_speed: float,
    delta_time: float,
    confidence: float,
) -> tuple[float, float]:
    """Apply one speed curve to a vector while preserving its direction."""

    magnitude = math.hypot(x, y)
    if magnitude == 0.0:
        return 0.0, 0.0
    scaled_magnitude = response_curve(
        magnitude,
        sensitivity,
        acceleration,
        maximum_speed,
        delta_time,
        confidence,
    )
    scale = scaled_magnitude / magnitude
    return x * scale, y * scale


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


def _pan_pose(
    pair: tuple[HandDetection, HandDetection],
    gesture: str,
) -> bool:
    if gesture == "Two closed hands":
        return all(not hand_is_open(hand) for hand in pair)
    if gesture == "Two open hands":
        return all(hand_is_open(hand) for hand in pair)
    return False


def _deactivation_pose(
    pair: tuple[HandDetection, HandDetection],
    gesture: str,
) -> bool:
    if gesture == "Two closed hands":
        return all(not hand_is_open(hand) for hand in pair)
    if gesture == "Two open hands":
        return all(hand_is_open(hand) for hand in pair)
    return False


def _orbit_lock_sources(
    hands: list[HandDetection],
    previous_centers: dict[str, Landmark],
) -> dict[str, HandDetection]:
    """Assign one or two detected hands to stable LEFT/RIGHT slots."""

    valid = [hand for hand in hands if len(hand.landmarks) > 16]
    if len(valid) > 2:
        valid = sorted(
            valid,
            key=lambda hand: hand.confidence if hand.confidence is not None else 1.0,
            reverse=True,
        )[:2]
    if not valid:
        return {}

    labelled: dict[str, HandDetection] = {}
    unlabelled: list[HandDetection] = []
    for hand in valid:
        label = (hand.handedness or "").casefold()
        if label in {"left", "right"} and label.upper() not in labelled:
            labelled[label.upper()] = hand
        else:
            unlabelled.append(hand)

    if len(valid) == 1:
        if labelled:
            return labelled
        center = hand_center(valid[0])
        if previous_centers:
            side = min(
                previous_centers,
                key=lambda candidate: _point_distance(center, previous_centers[candidate]),
            )
        else:
            side = "LEFT" if center.x <= 0.5 else "RIGHT"
        return {side: valid[0]}

    if not labelled:
        ordered = sorted(valid, key=lambda hand: hand_center(hand).x)
        if "LEFT" in previous_centers and "RIGHT" in previous_centers:
            first_center = hand_center(ordered[0])
            second_center = hand_center(ordered[1])
            normal = _point_distance(first_center, previous_centers["LEFT"]) + _point_distance(
                second_center, previous_centers["RIGHT"]
            )
            swapped = _point_distance(first_center, previous_centers["RIGHT"]) + _point_distance(
                second_center, previous_centers["LEFT"]
            )
            return (
                {"LEFT": ordered[0], "RIGHT": ordered[1]}
                if normal <= swapped
                else {"LEFT": ordered[1], "RIGHT": ordered[0]}
            )
        return {"LEFT": ordered[0], "RIGHT": ordered[1]}

    result = dict(labelled)
    missing = [side for side in ("LEFT", "RIGHT") if side not in result]
    if missing and unlabelled:
        result[missing[0]] = unlabelled[0]
    return result


def _orbit_lock_control_point(lock: _OrbitLockFrame) -> Landmark | None:
    points = [hand_center(lock.sources[side]) for side in lock.hands if side in lock.sources]
    if not points:
        return None
    count = len(points)
    return Landmark(
        x=sum(point.x for point in points) / count,
        y=sum(point.y for point in points) / count,
        z=sum(point.z for point in points) / count,
    )


def _orbit_lock_hand_label(hands: tuple[str, ...]) -> str:
    if len(hands) > 1:
        return "BOTH"
    return hands[0] if hands else "NONE"


def _hands_confidence(hands: list[HandDetection]) -> float:
    values = [
        hand.confidence
        for hand in hands
        if hand.confidence is not None and math.isfinite(hand.confidence)
    ]
    return max(0.0, min(1.0, min(values) if values else 1.0)) if hands else 0.0


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


def _point_distance(first: Landmark, second: Landmark) -> float:
    return distance_between_hands(first, second)


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _limit(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return max(-maximum, min(maximum, value))


def _frame_delta(current: float, previous: float) -> float:
    return max(1.0 / 240.0, min(0.25, current - previous))


def _low_pass_alpha(cutoff: float, delta_time: float) -> float:
    safe_cutoff = max(0.05, cutoff)
    tau = 1.0 / (2.0 * math.pi * safe_cutoff)
    return 1.0 / (1.0 + tau / max(1.0 / 240.0, delta_time))


def _effective_smoothing(settings: AppSettings) -> float:
    return max(0.0, min(1.0, settings.navigation_smoothing))


def _filter_parameters(settings: AppSettings, smoothing: float) -> dict[str, Any]:
    smoothing = max(0.0, min(1.0, smoothing))
    # Higher smoothing lowers the stationary cutoff; the adaptive beta still
    # opens the filter during fast motion, keeping latency low.
    cutoff_scale = max(0.25, 1.65 - 1.25 * smoothing)
    return {
        "min_cutoff": settings.navigation_one_euro_min_cutoff * cutoff_scale,
        "beta": settings.navigation_one_euro_beta
        if settings.navigation_adaptive_smoothing
        else 0.0,
        "derivative_cutoff": settings.navigation_one_euro_derivative_cutoff,
        "enabled": settings.navigation_one_euro_enabled,
    }


def _channel_dead_zone(general: float, specific: float) -> float:
    # The original general setting remains a backwards-compatible override.
    if abs(general - DEFAULT_NAVIGATION_DEAD_ZONE) > 1e-9:
        return max(0.0, general)
    return max(0.0, specific)


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
