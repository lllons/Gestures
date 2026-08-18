from __future__ import annotations

import unittest

from app.face_tracker import FaceDetection
from app.gesture_detector import DetectionState, GestureDetector
from app.hand_tracker import HandDetection, Landmark
from app.settings import AppSettings


class GestureDetectorActivationTests(unittest.TestCase):
    def test_defaults_use_a_larger_zone_and_instant_activation(self) -> None:
        settings = AppSettings()
        self.assertAlmostEqual(settings.touch_threshold, 0.10)
        self.assertEqual(settings.touch_duration_ms, 0)
        self.assertEqual(settings.cooldown_ms, 0)
        self.assertEqual(settings.pinch_shortcut, "Alt + Tab")

        detector = GestureDetector(settings)
        snapshot = detector.process(
            [self._hand_at(0.09)],
            self._face_at(0.0),
            now=10.0,
        )

        self.assertTrue(snapshot.triggered)
        self.assertEqual(snapshot.state, DetectionState.TOUCH_DETECTED)
        self.assertEqual(snapshot.touch_elapsed_ms, 0)

    def test_fingertip_outside_default_zone_does_not_trigger(self) -> None:
        detector = GestureDetector(AppSettings())
        snapshot = detector.process(
            [self._hand_at(0.11)],
            self._face_at(0.0),
            now=10.0,
        )

        self.assertFalse(snapshot.triggered)
        self.assertEqual(snapshot.state, DetectionState.APPROACHING)

    def test_thumb_index_pinch_triggers_once_until_release(self) -> None:
        detector = GestureDetector(AppSettings())

        first = detector.process([self._pinch_hand()], None, now=10.0)
        repeated = detector.process([self._pinch_hand()], None, now=10.01)
        detector.process([self._open_hand()], None, now=10.20)
        released = detector.process([self._open_hand()], None, now=10.33)
        second = detector.process([self._pinch_hand()], None, now=10.34)

        self.assertTrue(first.pinch_detected)
        self.assertTrue(first.pinch_triggered)
        self.assertEqual(first.pinch_cooldown_remaining_ms, 0)
        self.assertFalse(repeated.pinch_triggered)
        self.assertFalse(released.pinch_awaiting_release)
        self.assertTrue(second.pinch_triggered)

    @staticmethod
    def _pinch_hand() -> HandDetection:
        return GestureDetectorActivationTests._hand_with_thumb_index(0.2, 0.2)

    @staticmethod
    def _open_hand() -> HandDetection:
        return GestureDetectorActivationTests._hand_with_thumb_index(0.2, 0.9)

    @staticmethod
    def _hand_with_thumb_index(thumb_x: float, index_x: float) -> HandDetection:
        landmarks = [Landmark(0.0, 0.0) for _ in range(21)]
        landmarks[0] = Landmark(0.0, 0.0)
        landmarks[4] = Landmark(thumb_x, 0.0)
        landmarks[8] = Landmark(index_x, 0.0)
        landmarks[9] = Landmark(0.0, 1.0)
        return HandDetection(landmarks=tuple(landmarks))

    @staticmethod
    def _hand_at(x: float) -> HandDetection:
        landmarks = [Landmark(0.0, 0.0) for _ in range(9)]
        landmarks[8] = Landmark(x, 0.0)
        return HandDetection(landmarks=tuple(landmarks))

    @staticmethod
    def _face_at(x: float) -> FaceDetection:
        nose = Landmark(x, 0.0)
        return FaceDetection(landmarks=(nose,), nose=nose, face_width=1.0)


if __name__ == "__main__":
    unittest.main()
