from __future__ import annotations

import unittest

from app.hand_tracker import Landmark
from app.mouse_controller import map_normalized_to_screen


class MouseMappingTests(unittest.TestCase):
    def test_maps_camera_corners_to_virtual_screen_corners(self) -> None:
        self.assertEqual(
            map_normalized_to_screen(Landmark(0.0, 0.0), -100, 20, 1000, 800),
            (-100, 20),
        )
        self.assertEqual(
            map_normalized_to_screen(Landmark(1.0, 1.0), -100, 20, 1000, 800),
            (899, 819),
        )

    def test_clamps_landmarks_outside_camera_bounds(self) -> None:
        self.assertEqual(
            map_normalized_to_screen(Landmark(-0.4, 1.4), 0, 0, 1920, 1080),
            (0, 1079),
        )

    def test_rejects_invalid_screen_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            map_normalized_to_screen(Landmark(0.5, 0.5), 0, 0, 0, 1080)


if __name__ == "__main__":
    unittest.main()
