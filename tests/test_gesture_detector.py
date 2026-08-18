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
