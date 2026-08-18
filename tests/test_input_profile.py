from __future__ import annotations

import unittest

from app.input_profile import InputProfile, default_profile_data, get_profile
from app.settings import AppSettings


class InputProfileTests(unittest.TestCase):
    def test_builtin_profiles_are_mappings_not_integrations(self) -> None:
        profiles = default_profile_data()
        self.assertIn("Generic 3D", profiles)
        self.assertIn("Blender", profiles)
        self.assertEqual(get_profile("Blender", profiles).orbit_button, "middle")
        self.assertEqual(get_profile("Blender", profiles).pan_modifiers, ("shift",))

    def test_custom_profile_round_trips_and_supports_multiple_modifiers(self) -> None:
        profile = InputProfile.from_dict(
            "Custom CAD",
            {
                "orbit_button": "right",
                "orbit_modifiers": ["alt"],
                "pan_button": "middle",
                "pan_modifiers": "alt + shift",
                "zoom_axis": "horizontal",
                "zoom_in_direction": "Inverted",
            },
        )

        restored = InputProfile.from_dict(profile.name, profile.to_dict())
        self.assertEqual(restored, profile)
        self.assertEqual(restored.pan_modifiers, ("alt", "shift"))
        self.assertEqual(restored.zoom_in_direction, -1)

    def test_settings_persist_profile_registry(self) -> None:
        settings = AppSettings(
            navigation_profile="Custom",
            navigation_profiles={
                **default_profile_data(),
                "Custom": InputProfile(
                    "Custom",
                    orbit_button="right",
                    pan_modifiers=("ctrl",),
                ).to_dict(),
            },
        )

        restored = AppSettings.from_dict(settings.to_dict())
        self.assertEqual(restored.navigation_profile, "Custom")
        self.assertEqual(get_profile("Custom", restored.navigation_profiles).orbit_button, "right")
        self.assertEqual(
            get_profile("Custom", restored.navigation_profiles).pan_modifiers,
            ("ctrl",),
        )

    def test_unknown_profile_falls_back_to_generic(self) -> None:
        profile = get_profile("does-not-exist", default_profile_data())
        self.assertEqual(profile.name, "Generic 3D")


if __name__ == "__main__":
    unittest.main()
