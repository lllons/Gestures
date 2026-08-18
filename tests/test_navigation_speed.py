from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.input_controller import InputState, NavigationInputController
from app.navigation import NavigationState
from app.settings import AppSettings


class FakeBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, object | None]] = []

    def press_button(self, button: str) -> None:
        self.events.append(("press_button", button, None))

    def release_button(self, button: str) -> None:
        self.events.append(("release_button", button, None))

    def press_modifier(self, modifier: str) -> None:
        self.events.append(("press_modifier", modifier, None))

    def release_modifier(self, modifier: str) -> None:
        self.events.append(("release_modifier", modifier, None))

    def move_relative(self, dx: int, dy: int) -> None:
        self.events.append(("move_relative", dx, dy))

    def scroll(self, horizontal: int, vertical: int) -> None:
        self.events.append(("scroll", horizontal, vertical))


def snapshot(orbit_x: float) -> SimpleNamespace:
    return SimpleNamespace(
        active=True,
        hand_count=2,
        state=NavigationState.ACTIVE,
        message="active",
        control_mode="FULL 3D",
        pan_pose=False,
        orbit_x=orbit_x,
        orbit_y=0.0,
        pan_x=0.0,
        pan_y=0.0,
        zoom=0.0,
        roll=0.0,
    )


class NavigationSpeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            navigation_enabled=True,
            navigation_acceleration=0.0,
            navigation_mouse_scale=100.0,
            navigation_precision_scale=0.35,
            navigation_fast_scale=1.75,
        )
        self.backend = FakeBackend()
        self.controller = NavigationInputController(self.settings, self.backend)

    def tearDown(self) -> None:
        self.controller.close()

    def test_precision_modifier_reduces_relative_motion(self) -> None:
        self.controller.apply(snapshot(0.04))
        normal_motion = [event for event in self.backend.events if event[0] == "move_relative"][-1]
        self.backend.events.clear()
        self.controller._speed_modifiers.add("alt")

        status = self.controller.apply(snapshot(0.04))
        precision_motion = [event for event in self.backend.events if event[0] == "move_relative"][-1]

        self.assertTrue(status.precision_mode)
        self.assertAlmostEqual(status.speed_factor, 0.35)
        self.assertLess(abs(precision_motion[1]), abs(normal_motion[1]))

    def test_idle_frame_releases_orbit_button(self) -> None:
        self.controller.apply(snapshot(0.04))
        self.backend.events.clear()

        status = self.controller.apply(snapshot(0.0))

        self.assertEqual(status.state, InputState.NAVIGATION_READY)
        self.assertEqual(status.held_buttons, ())
        self.assertIn(("release_button", "middle", None), self.backend.events)


if __name__ == "__main__":
    unittest.main()
