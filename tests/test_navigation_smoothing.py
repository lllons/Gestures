from __future__ import annotations

import unittest

from app.hand_tracker import HandDetection, Landmark
from app.navigation import (
    OneEuroFilter,
    TwoHandNavigation,
    confidence_control_gain,
    hysteresis,
    response_curve,
)
from app.settings import AppSettings


class NavigationSignalTests(unittest.TestCase):
    def test_one_euro_filter_reduces_stationary_jitter(self) -> None:
        smoother = OneEuroFilter(min_cutoff=0.8, beta=0.05, derivative_cutoff=1.0)
        values = [
            smoother.filter(value, index / 60.0)
            for index, value in enumerate((0.500, 0.507, 0.496, 0.505, 0.498, 0.503, 0.499))
        ]

        self.assertLess(max(values) - min(values), 0.011)
        self.assertAlmostEqual(values[0], 0.5)

    def test_response_curve_scales_with_real_delta_time(self) -> None:
        at_10_fps = response_curve(0.5, 1.0, 0.0, 2.0, 0.1)
        at_20_fps = response_curve(0.5, 1.0, 0.0, 2.0, 0.05)

        self.assertAlmostEqual(at_10_fps, at_20_fps * 2.0)

    def test_response_curve_acceleration_respects_maximum_speed(self) -> None:
        output = response_curve(
            velocity=2.0,
            sensitivity=1.0,
            acceleration=3.0,
            maximum_speed=0.5,
            delta_time=0.1,
        )

        self.assertLessEqual(output, 0.5 * 0.1)

    def test_disabled_one_euro_filter_is_a_low_latency_bypass(self) -> None:
        smoother = OneEuroFilter(min_cutoff=0.8, beta=0.05, enabled=False)

        self.assertAlmostEqual(smoother.filter(0.5, 0.0), 0.5)
        self.assertAlmostEqual(smoother.filter(0.8, 1 / 60.0), 0.8)

    def test_confidence_gain_and_hysteresis_are_smooth(self) -> None:
        self.assertLess(
            confidence_control_gain(0.6, 0.4),
            confidence_control_gain(0.95, 0.4),
        )
        self.assertFalse(hysteresis(False, 0.029, 0.03, 0.015))
        self.assertTrue(hysteresis(False, 0.031, 0.03, 0.015))
        self.assertTrue(hysteresis(True, 0.020, 0.03, 0.015))
        self.assertFalse(hysteresis(True, 0.014, 0.03, 0.015))


class SmoothTwoHandNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            navigation_enabled=True,
            navigation_smoothing_frames=1,
            navigation_orbit_lock_enabled=False,
            navigation_smoothing=0.35,
            navigation_dead_zone=0.0,
            navigation_max_speed=5.0,
            navigation_orbit_max_speed=5.0,
            navigation_pan_max_speed=5.0,
            navigation_zoom_max_speed=5.0,
            navigation_orbit_sensitivity=1.0,
            navigation_pan_sensitivity=1.0,
            navigation_zoom_sensitivity=1.0,
            navigation_roll_sensitivity=1.0,
            navigation_motion_start_threshold=0.01,
            navigation_motion_stop_threshold=0.005,
            navigation_zoom_start_threshold=0.01,
            navigation_zoom_stop_threshold=0.005,
        )

    def test_still_hands_do_not_drift_after_activation(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        hands = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(hands, now=0.0)
        navigation.process(hands, now=0.71)

        outputs = [
            navigation.process(hands, now=0.71 + index / 60.0)
            for index in range(1, 601)
        ]

        self.assertTrue(all(output.orbit_x == 0.0 for output in outputs))
        self.assertTrue(all(output.pan_x == 0.0 for output in outputs))
        self.assertTrue(all(output.zoom == 0.0 for output in outputs))

    def test_midpoint_translation_and_symmetric_spread_are_independent(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        base = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(base, now=0.0)
        navigation.process(base, now=0.71)

        spread = navigation.process(self._pair(0.20, 0.80, open_hand=True), now=0.81)
        self.assertGreater(spread.zoom, 0.0)
        self.assertAlmostEqual(spread.midpoint_velocity_x, 0.0, places=6)
        self.assertEqual(spread.orbit_x, 0.0)

        translated = navigation.process(self._pair(0.25, 0.85, open_hand=True), now=0.91)
        self.assertGreater(translated.orbit_x, 0.0)
        self.assertEqual(translated.zoom, 0.0)

    def test_outlier_is_rejected_without_a_mouse_spike(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        base = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(base, now=0.0)
        navigation.process(base, now=0.71)

        spike = navigation.process(self._pair(0.90, 0.70, open_hand=True), now=0.81)

        self.assertTrue(spike.outlier_rejected)
        self.assertEqual(spike.orbit_x, 0.0)
        self.assertEqual(spike.pan_x, 0.0)
        self.assertEqual(spike.zoom, 0.0)

    def test_hand_loss_grace_releases_input_then_rebases_on_recovery(self) -> None:
        settings = AppSettings(
            **{
                **self.settings.to_dict(),
                "navigation_hand_loss_grace_frames": 2,
            }
        )
        navigation = TwoHandNavigation(settings)
        hands = self._pair(0.30, 0.70, open_hand=True)
        navigation.process(hands, now=0.0)
        navigation.process(hands, now=0.71)
        navigation.process(self._pair(0.35, 0.75, open_hand=True), now=0.81)

        gap = navigation.process([hands[0]], now=0.82)
        recovered = navigation.process(self._pair(0.35, 0.75, open_hand=True), now=0.83)
        moved = navigation.process(self._pair(0.40, 0.80, open_hand=True), now=0.93)

        self.assertEqual(gap.hand_loss_frames, 1)
        self.assertFalse(gap.active)
        self.assertTrue(recovered.active)
        self.assertEqual(recovered.orbit_x, 0.0)
        self.assertGreater(moved.orbit_x, 0.0)

    def test_confidence_reduces_control_output(self) -> None:
        high = TwoHandNavigation(self.settings)
        low_settings = AppSettings(
            **{
                **self.settings.to_dict(),
                "navigation_min_confidence": 0.3,
            }
        )
        low = TwoHandNavigation(low_settings)
        high_hands = self._pair(0.30, 0.70, open_hand=True, confidence=0.95)
        low_hands = self._pair(0.30, 0.70, open_hand=True, confidence=0.60)
        for navigation, hands in ((high, high_hands), (low, low_hands)):
            navigation.process(hands, now=0.0)
            navigation.process(hands, now=0.71)
        high_output = high.process(
            self._pair(0.35, 0.75, open_hand=True, confidence=0.95), now=0.81
        ).orbit_x
        low_output = low.process(
            self._pair(0.35, 0.75, open_hand=True, confidence=0.60), now=0.81
        ).orbit_x

        self.assertGreater(high_output, low_output)
        self.assertGreater(low_output, 0.0)

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
            SmoothTwoHandNavigationTests._hand(left_x, y, "Left", open_hand, confidence),
            SmoothTwoHandNavigationTests._hand(right_x, y, "Right", open_hand, confidence),
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
            for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
                landmarks[pip_index] = Landmark(x, y)
                landmarks[tip_index] = Landmark(x, y)
        return HandDetection(
            landmarks=tuple(landmarks),
            handedness=handedness,
            confidence=confidence,
        )


if __name__ == "__main__":
    unittest.main()
