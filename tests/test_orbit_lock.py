from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.hand_tracker import HandDetection, Landmark
from app.input_controller import InputState, NavigationInputController
from app.navigation import (
    NavigationState,
    TwoHandNavigation,
    is_ring_thumb_touching,
    ring_thumb_distance,
)
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


def hand(
    x: float,
    y: float,
    side: str = "Left",
    *,
    ring_distance: float = 0.02,
    confidence: float = 0.95,
) -> HandDetection:
    landmarks = [Landmark(x, y) for _ in range(21)]
    for index in (0, 5, 9, 13, 17):
        landmarks[index] = Landmark(x, y + 0.03)
    landmarks[4] = Landmark(x + 0.01, y)
    landmarks[16] = Landmark(x + 0.01 + ring_distance, y)
    return HandDetection(
        landmarks=tuple(landmarks),
        handedness=side,
        confidence=confidence,
    )


def controller_snapshot(
    *,
    lock: bool,
    orbit_x: float = 0.0,
    zoom: float = 0.0,
    pan_x: float = 0.0,
    active: bool = True,
    hand_count: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        active=active,
        hand_count=hand_count,
        state=NavigationState.ACTIVE if active else NavigationState.LOST,
        message="active" if active else "lost",
        control_mode="ZOOM" if zoom else "FULL 3D",
        pan_pose=False,
        orbit_x=orbit_x,
        orbit_y=0.0,
        pan_x=pan_x,
        pan_y=0.0,
        zoom=zoom,
        roll=0.0,
        orbit_lock_active=lock,
        orbit_lock_hand="LEFT" if lock else "NONE",
    )


class OrbitLockDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            navigation_enabled=True,
            navigation_one_euro_enabled=False,
            navigation_dead_zone=0.0,
            navigation_orbit_dead_zone=0.0,
            navigation_orbit_sensitivity=1.0,
            navigation_orbit_max_speed=5.0,
            navigation_motion_start_threshold=0.0,
            navigation_motion_stop_threshold=0.0,
            navigation_orbit_lock_activation_threshold=0.035,
            navigation_orbit_lock_release_threshold=0.050,
        )

    def test_distance_rule_and_hysteresis_helpers(self) -> None:
        touching = hand(0.4, 0.5, ring_distance=0.034)
        separated = hand(0.4, 0.5, ring_distance=0.036)
        self.assertAlmostEqual(ring_thumb_distance(touching), 0.034)
        self.assertTrue(is_ring_thumb_touching(touching, 0.035))
        self.assertFalse(is_ring_thumb_touching(separated, 0.035))

    def test_orbit_lock_settings_round_trip(self) -> None:
        settings = AppSettings(
            navigation_enabled=True,
            navigation_orbit_lock_enabled=False,
            navigation_orbit_lock_activation_threshold=0.02,
            navigation_orbit_lock_release_threshold=0.08,
            navigation_orbit_lock_button="right",
        )
        restored = AppSettings.from_dict(settings.to_dict())
        self.assertFalse(restored.navigation_orbit_lock_enabled)
        self.assertAlmostEqual(restored.navigation_orbit_lock_activation_threshold, 0.02)
        self.assertAlmostEqual(restored.navigation_orbit_lock_release_threshold, 0.08)
        self.assertEqual(restored.navigation_orbit_lock_button, "right")

    def test_navigation_off_ignores_contact_completely(self) -> None:
        navigation = TwoHandNavigation(
            AppSettings(
                **{
                    **self.settings.to_dict(),
                    "navigation_enabled": False,
                }
            )
        )
        result = navigation.process([hand(0.4, 0.5)], now=0.0)
        self.assertFalse(result.orbit_lock_active)
        self.assertFalse(result.active)
        self.assertEqual(result.orbit_x, 0.0)

    def test_one_hand_contact_activates_immediately_and_moves_orbit(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        first = navigation.process([hand(0.3, 0.5)], now=0.0)
        moved = navigation.process([hand(0.36, 0.5)], now=0.1)

        self.assertEqual(first.state, NavigationState.ACTIVE)
        self.assertTrue(first.orbit_lock_active)
        self.assertEqual(first.orbit_lock_hand, "LEFT")
        self.assertEqual(first.gesture, "Orbit Lock")
        self.assertTrue(moved.orbit_lock_active)
        self.assertGreater(moved.orbit_x, 0.0)
        self.assertEqual(moved.zoom, 0.0)
        self.assertEqual(moved.pan_x, 0.0)

    def test_release_threshold_keeps_lock_then_releases(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        navigation.process([hand(0.3, 0.5, ring_distance=0.02)], now=0.0)
        held = navigation.process([hand(0.3, 0.5, ring_distance=0.045)], now=0.1)
        released = navigation.process([hand(0.3, 0.5, ring_distance=0.06)], now=0.2)

        self.assertTrue(held.orbit_lock_active)
        self.assertFalse(released.orbit_lock_active)
        self.assertEqual(released.orbit_x, 0.0)
        self.assertEqual(released.orbit_lock_hand, "NONE")

    def test_both_touching_hands_report_both_and_spread_is_not_zoom(self) -> None:
        navigation = TwoHandNavigation(self.settings)
        first = [hand(0.3, 0.5, "Left"), hand(0.7, 0.5, "Right")]
        second = [hand(0.2, 0.5, "Left"), hand(0.8, 0.5, "Right")]
        navigation.process(first, now=0.0)
        result = navigation.process(second, now=0.1)

        self.assertTrue(result.orbit_lock_active)
        self.assertEqual(result.orbit_lock_hand, "BOTH")
        self.assertEqual(result.zoom, 0.0)
        self.assertEqual(result.pan_x, 0.0)


class OrbitLockInputPriorityTests(unittest.TestCase):
    def test_lock_owns_button_and_suppresses_pan_and_zoom(self) -> None:
        settings = AppSettings(
            navigation_enabled=True,
            navigation_orbit_lock_button="right",
            navigation_mouse_scale=100.0,
        )
        backend = FakeBackend()
        controller = NavigationInputController(settings, backend)
        try:
            status = controller.apply(
                controller_snapshot(
                    lock=True,
                    orbit_x=0.04,
                    pan_x=0.8,
                    zoom=0.8,
                )
            )
            self.assertEqual(status.state, InputState.ORBITING)
            self.assertEqual(status.held_buttons, ("right",))
            self.assertEqual(
                [event for event in backend.events if event[0] == "scroll"], []
            )
            self.assertIn(("move_relative", 4, 0), backend.events)

            released = controller.apply(
                controller_snapshot(lock=False, active=False, hand_count=1)
            )
            self.assertEqual(released.held_buttons, ())
            self.assertIn(("release_button", "right", None), backend.events)
        finally:
            controller.close()

    def test_disabling_lock_releases_held_button_without_disabling_navigation(self) -> None:
        settings = AppSettings(navigation_enabled=True)
        backend = FakeBackend()
        controller = NavigationInputController(settings, backend)
        try:
            controller.apply(controller_snapshot(lock=True, orbit_x=0.1))
            controller.update_settings(
                AppSettings(
                    **{
                        **settings.to_dict(),
                        "navigation_orbit_lock_enabled": False,
                    }
                )
            )
            self.assertEqual(controller.status.held_buttons, ())
            self.assertEqual(controller.status.state, InputState.NAVIGATION_READY)
            self.assertIn(("release_button", "middle", None), backend.events)
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
