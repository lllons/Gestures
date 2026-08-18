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


def snapshot(
    *,
    orbit_x: float = 0.0,
    orbit_y: float = 0.0,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    zoom: float = 0.0,
    control_mode: str = "FULL 3D",
    pan_pose: bool = False,
    active: bool = True,
    hand_count: int = 2,
    state: NavigationState = NavigationState.ACTIVE,
) -> SimpleNamespace:
    return SimpleNamespace(
        active=active,
        hand_count=hand_count,
        state=state,
        message="tracking lost" if not active else "active",
        control_mode=control_mode,
        pan_pose=pan_pose,
        orbit_x=orbit_x,
        orbit_y=orbit_y,
        pan_x=pan_x,
        pan_y=pan_y,
        zoom=zoom,
        roll=0.0,
    )


class NavigationInputControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            navigation_enabled=True,
            navigation_acceleration=0.0,
            navigation_mouse_scale=100.0,
            navigation_zoom_wheel_scale=10.0,
        )
        self.backend = FakeBackend()
        self.controller = NavigationInputController(self.settings, self.backend)

    def tearDown(self) -> None:
        self.controller.close()

    def test_orbit_holds_middle_button_and_moves_relative(self) -> None:
        status = self.controller.apply(snapshot(orbit_x=0.12, orbit_y=-0.03))

        self.assertEqual(status.state, InputState.ORBITING)
        self.assertEqual(status.held_buttons, ("middle",))
        self.assertEqual(status.held_modifiers, ())
        self.assertIn(("press_button", "middle", None), self.backend.events)
        self.assertIn(("move_relative", 12, -3), self.backend.events)

    def test_full_mode_closed_hands_switches_to_pan_profile(self) -> None:
        self.controller.apply(snapshot(orbit_x=0.1))
        self.backend.events.clear()

        status = self.controller.apply(snapshot(pan_x=0.08, pan_pose=True))

        self.assertEqual(status.state, InputState.PANNING)
        self.assertEqual(status.held_buttons, ("middle",))
        self.assertEqual(status.held_modifiers, ("shift",))
        self.assertEqual(
            self.backend.events[:4],
            [
                ("release_button", "middle", None),
                ("press_modifier", "shift", None),
                ("press_button", "middle", None),
                ("move_relative", 8, 0),
            ],
        )

    def test_zoom_accumulates_fractional_wheel_units(self) -> None:
        first = self.controller.apply(snapshot(zoom=0.05, control_mode="ZOOM"))
        second = self.controller.apply(snapshot(zoom=0.05, control_mode="ZOOM"))

        self.assertEqual(first.state, InputState.ZOOMING)
        self.assertEqual(second.state, InputState.ZOOMING)
        self.assertEqual(
            [event for event in self.backend.events if event[0] == "scroll"],
            [("scroll", 0, 1)],
        )
        self.assertEqual(second.held_buttons, ())

    def test_tracking_loss_releases_button_and_modifier(self) -> None:
        self.controller.apply(snapshot(pan_x=0.1, pan_pose=True))
        status = self.controller.apply(
            snapshot(active=False, hand_count=1, state=NavigationState.LOST)
        )

        self.assertEqual(status.state, InputState.LOST_TRACKING)
        self.assertEqual(status.held_buttons, ())
        self.assertEqual(status.held_modifiers, ())
        self.assertIn(("release_button", "middle", None), self.backend.events)
        self.assertIn(("release_modifier", "shift", None), self.backend.events)

    def test_emergency_stop_disables_and_releases_all_controls(self) -> None:
        self.controller.apply(snapshot(pan_x=0.1, pan_pose=True))
        self.controller.emergency_stop()
        status = self.controller.status

        self.assertFalse(status.global_enabled)
        self.assertEqual(status.state, InputState.DISABLED)
        self.assertEqual(status.held_buttons, ())
        self.assertEqual(status.held_modifiers, ())
        self.assertEqual(self.controller.apply(snapshot(orbit_x=0.5)).held_buttons, ())

        self.controller.toggle_global()
        self.assertTrue(self.controller.status.global_enabled)
        self.assertEqual(self.controller.status.state, InputState.NAVIGATION_READY)

    def test_feature_disable_releases_without_emitting_motion(self) -> None:
        self.controller.apply(snapshot(orbit_x=0.1))
        disabled = AppSettings(navigation_enabled=False)
        self.controller.update_settings(disabled)
        status = self.controller.apply(snapshot(orbit_x=0.1))

        self.assertEqual(status.state, InputState.DISABLED)
        self.assertEqual(status.held_buttons, ())
        self.assertEqual(
            [event for event in self.backend.events if event[0] == "move_relative"],
            [("move_relative", 10, 0)],
        )


if __name__ == "__main__":
    unittest.main()
