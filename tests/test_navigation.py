from __future__ import annotations

import math
import unittest

from app.hand_tracker import HandDetection, Landmark
from app.navigation import (
    NavigationState,
    TwoHandNavigation,
    angle_between_hands,
    angle_delta_between,
    apply_dead_zone,
    calculate_angle,
    calculate_distance,
    calculate_distance_delta,
    calculate_midpoint,
    distance_between_hands,
    movement_delta,
)
from app.settings import AppSettings


class NavigationMathTests(unittest.TestCase):
    def test_distance_midpoint_and_angle(self) -> None:
        left = Landmark(0.2, 0.3)
        right = Landmark(0.8, 0.7)

        self.assertAlmostEqual(distance_between_hands(left, right), math.sqrt(0.52))
        self.assertEqual(calculate_distance(left, right), distance_between_hands(left, right))
        midpoint = calculate_midpoint(left, right)
        assert midpoint is not None
        self.assertAlmostEqual(midpoint.x, 0.5)
        self.assertAlmostEqual(midpoint.y, 0.5)
        self.assertAlmostEqual(angle_between_hands(left, right), math.atan2(0.4, 0.6))
        self.assertAlmostEqual(calculate_angle(left, right), angle_between_hands(left, right))

    def test_frame_deltas_and_angle_wrap(self) -> None:
        current = Landmark(0.7, 0.2, 0.1)
        previous = Landmark(0.5, 0.3, 0.0)
        delta = movement_delta(current, previous)
        self.assertAlmostEqual(delta.x, 0.2)
        self.assertAlmostEqual(delta.y, -0.1)
        self.assertAlmostEqual(delta.z, 0.1)
        self.assertAlmostEqual(calculate_distance_delta(0.52, 0.50), 0.02)
        self.assertAlmostEqual(angle_delta_between(-math.pi + 0.05, math.pi - 0.05), 0.1)

    def test_dead_zone_removes_tracking_noise(self) -> None:
        self.assertEqual(apply_dead_zone(0.003, 0.004), 0.0)
        self.assertEqual(apply_dead_zone(-0.003, 0.004), 0.0)
        self.assertAlmostEqual(apply_dead_zone(0.02, 0.004), 0.02)


class TwoHandNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            navigation_enabled=True,
            navigation_smoothing_frames=1,
            navigation_dead_zone=0.0,
            navigation_max_speed=5.0,
            navigation_orbit_sensitivity=1.0,
            navigation_pan_sensitivity=1.0,
            navigation_zoom_sensitivity=1.0,
            navigation_roll_sensitivity=1.0,
        )

    def test_activation_requires_two_open_hands_and_a_hold(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        hands = self._pair(0.30, 0.70, open_hand=True)

        activating = navigation.process(hands, now=0.0)
        active = navigation.process(hands, now=0.71)

        self.assertEqual(activating.state, NavigationState.ACTIVATING)
        self.assertLess(activating.activation_progress, 1.0)
        self.assertEqual(active.state, NavigationState.ACTIVE)
        self.assertTrue(active.active)
        self.assertEqual(active.orbit_x, 0.0)
        self.assertEqual(active.zoom, 0.0)

    def test_smoothing_reduces_a_single_frame_jump(self) -> None:
        settings = AppSettings(
            **{
                **self.settings.to_dict(),
                "navigation_smoothing_frames": 2,
            }
        )
        navigation = TwoHandNavigation(settings)
        base = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(base, now=0.0)
        navigation.process(base, now=0.71)

        moved = navigation.process(self._pair(0.40, 0.80, open_hand=True), now=0.81)

        self.assertGreater(moved.midpoint_delta_x, 0.0)
        self.assertLess(moved.midpoint_delta_x, 0.10)

    def test_translation_produces_orbit_and_pan_without_zoom(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        hands = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(hands, now=0.0)
        navigation.process(hands, now=0.71)

        moved = navigation.process(self._pair(0.35, 0.75, open_hand=True), now=0.81)

        self.assertEqual(moved.state, NavigationState.ACTIVE)
        self.assertGreater(moved.midpoint_delta_x, 0.0)
        self.assertGreater(moved.orbit_x, 0.0)
        self.assertGreater(moved.pan_x, 0.0)
        self.assertEqual(moved.zoom, 0.0)
        self.assertAlmostEqual(moved.left_velocity_x, moved.right_velocity_x)

    def test_hands_moving_apart_zoom_in_and_together_zoom_out(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        base = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(base, now=0.0)
        navigation.process(base, now=0.71)

        farther = navigation.process(self._pair(0.25, 0.75, open_hand=True), now=0.81)
        closer = navigation.process(self._pair(0.29, 0.71, open_hand=True), now=0.91)

        self.assertGreater(farther.distance_delta, 0.0)
        self.assertGreater(farther.zoom, 0.0)
        self.assertLess(closer.distance_delta, 0.0)
        self.assertLess(closer.zoom, 0.0)

    def test_roll_is_optional_and_angle_delta_is_wrapped(self) -> None:
        settings = AppSettings(
            **{
                **self.settings.to_dict(),
                "navigation_roll_enabled": True,
            }
        )
        navigation = TwoHandNavigation(settings)
        # Start with a nearly horizontal pair, then rotate its line.
        navigation.process(self._pair(0.30, 0.70, open_hand=True, y=0.50), now=0.0)
        navigation.process(self._pair(0.30, 0.70, open_hand=True, y=0.50), now=0.71)
        rotated = navigation.process(
            [
                self._hand(0.40, 0.60, "Left", True, 0.95),
                self._hand(0.60, 0.40, "Right", True, 0.95),
            ],
            now=0.81,
        )

        self.assertNotEqual(rotated.angle_delta, 0.0)
        self.assertNotEqual(rotated.roll, 0.0)

    def test_held_deactivation_pose_stops_navigation(self) -> None:
        settings = AppSettings(
            **{
                **self.settings.to_dict(),
                "navigation_deactivation_gesture": "Two closed hands",
            }
        )
        navigation = TwoHandNavigation(settings)
        open_hands = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(open_hands, now=0.0)
        navigation.process(open_hands, now=0.71)

        stopping = navigation.process(self._pair(0.30, 0.70, open_hand=False), now=0.81)
        stopped = navigation.process(self._pair(0.30, 0.70, open_hand=False), now=1.20)

        self.assertTrue(stopping.active)
        self.assertEqual(stopping.gesture, "Deactivating")
        self.assertEqual(stopping.orbit_x, 0.0)
        self.assertEqual(stopped.state, NavigationState.IDLE)
        self.assertFalse(stopped.active)

    def test_calibration_records_neutral_reference_and_stops_motion(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        navigation.begin_calibration(now=0.0)
        result = None
        for index in range(9):
            result = navigation.process(
                self._pair(0.25, 0.75, open_hand=True),
                now=index * 0.2,
            )
            if result.calibration_completed:
                break

        assert result is not None
        self.assertTrue(result.calibration_completed)
        self.assertTrue(result.calibration_succeeded)
        self.assertTrue(navigation.calibrated)
        self.assertIsNotNone(result.neutral_midpoint)
        self.assertEqual(result.orbit_x, 0.0)
        self.assertEqual(result.zoom, 0.0)

    def test_one_hand_or_low_confidence_stops_and_clears_outputs(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        hands = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(hands, now=0.0)
        navigation.process(hands, now=0.71)
        moving = navigation.process(self._pair(0.35, 0.75, open_hand=True), now=0.81)
        self.assertNotEqual(moving.orbit_x, 0.0)

        lost = navigation.process([hands[0]], now=0.91)
        self.assertEqual(lost.state, NavigationState.LOST)
        self.assertFalse(lost.active)
        self.assertEqual(lost.orbit_x, 0.0)
        self.assertEqual(lost.pan_x, 0.0)
        self.assertEqual(lost.zoom, 0.0)

        low_confidence = navigation.process(
            self._pair(0.30, 0.70, open_hand=True, confidence=0.1),
            now=1.01,
        )
        self.assertEqual(low_confidence.confidence, 0.1)
        self.assertEqual(low_confidence.orbit_x, 0.0)

    @staticmethod
    def _pair(
        left_x: float,
        right_x: float,
        *,
        open_hand: bool,
        y: float = 0.5,
        confidence: float = 0.95,
    ) -> list[HandDetection]:
        return [
            TwoHandNavigationTests._hand(left_x, y, "Left", open_hand, confidence),
            TwoHandNavigationTests._hand(right_x, y, "Right", open_hand, confidence),
        ]

    @staticmethod
    def _hand(
        x: float,
        y: float,
        handedness: str,
        open_hand: bool,
        confidence: float,
    ) -> HandDetection:
        landmarks = [Landmark(x, y) for _ in range(21)]
        for index in (0, 5, 9, 13, 17):
            landmarks[index] = Landmark(x, y + 0.03)
        if open_hand:
            for tip_index, pip_index, offset in (
                (8, 6, -0.20),
                (12, 10, -0.22),
                (16, 14, -0.20),
                (20, 18, -0.18),
            ):
                landmarks[pip_index] = Landmark(x, y - 0.04)
                landmarks[tip_index] = Landmark(x, y + offset)
        else:
            for tip_index, pip_index in (
                (8, 6),
                (12, 10),
                (16, 14),
                (20, 18),
            ):
                landmarks[pip_index] = Landmark(x, y)
                landmarks[tip_index] = Landmark(x, y)
        return HandDetection(
            landmarks=tuple(landmarks),
            handedness=handedness,
            confidence=confidence,
        )


if __name__ == "__main__":
    unittest.main()
