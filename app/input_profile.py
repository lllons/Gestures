"""Configurable input profiles for universal 3D navigation.

Profiles describe only the mouse and modifier gesture expected by a focused
application.  They do not identify or communicate with that application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


MOUSE_BUTTON_OPTIONS = ("left", "middle", "right")
MODIFIER_OPTIONS = ("none", "shift", "ctrl", "alt", "cmd")
MODIFIER_TEXT_OPTIONS = (
    "none",
    "shift",
    "ctrl",
    "alt",
    "cmd",
    "shift + alt",
    "alt + shift",
    "ctrl + shift",
)
ZOOM_DIRECTION_OPTIONS = ("Normal", "Inverted")
ZOOM_AXIS_OPTIONS = ("vertical", "horizontal")


@dataclass(frozen=True)
class InputProfile:
    """Mouse gesture mapping used by a focused 3D application."""

    name: str
    orbit_button: str = "middle"
    orbit_modifiers: tuple[str, ...] = ()
    pan_button: str = "middle"
    pan_modifiers: tuple[str, ...] = ("shift",)
    zoom_axis: str = "vertical"
    zoom_in_direction: int = 1
    description: str = ""

    def validate(self) -> "InputProfile":
        if not self.name.strip():
            raise ValueError("Input profile name cannot be empty.")
        if self.orbit_button not in MOUSE_BUTTON_OPTIONS:
            raise ValueError(f"Unsupported orbit mouse button: {self.orbit_button!r}.")
        if self.pan_button not in MOUSE_BUTTON_OPTIONS:
            raise ValueError(f"Unsupported pan mouse button: {self.pan_button!r}.")
        if not all(modifier in MODIFIER_OPTIONS for modifier in self.orbit_modifiers):
            raise ValueError("Unsupported orbit keyboard modifier.")
        if not all(modifier in MODIFIER_OPTIONS for modifier in self.pan_modifiers):
            raise ValueError("Unsupported pan keyboard modifier.")
        if self.zoom_axis not in ZOOM_AXIS_OPTIONS:
            raise ValueError(f"Unsupported zoom axis: {self.zoom_axis!r}.")
        if self.zoom_in_direction not in {-1, 1}:
            raise ValueError("Zoom direction must be 1 or -1.")
        return self

    @property
    def zoom_direction_label(self) -> str:
        return "Normal" if self.zoom_in_direction == 1 else "Inverted"

    @staticmethod
    def modifiers_text(modifiers: Iterable[str]) -> str:
        return " + ".join(modifier.title() for modifier in modifiers) or "None"

    @property
    def orbit_modifiers_text(self) -> str:
        return self.modifiers_text(self.orbit_modifiers)

    @property
    def pan_modifiers_text(self) -> str:
        return self.modifiers_text(self.pan_modifiers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "orbit_button": self.orbit_button,
            "orbit_modifiers": list(self.orbit_modifiers),
            "pan_button": self.pan_button,
            "pan_modifiers": list(self.pan_modifiers),
            "zoom_axis": self.zoom_axis,
            "zoom_in_direction": self.zoom_in_direction,
            "description": self.description,
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        data: Mapping[str, Any] | None,
    ) -> "InputProfile":
        values = dict(data or {})
        orbit_modifiers = _normalize_modifiers(
            values.get("orbit_modifiers", values.get("orbit_modifier", ()))
        )
        pan_modifiers = _normalize_modifiers(
            values.get("pan_modifiers", values.get("pan_modifier", ("shift",)))
        )
        raw_direction = values.get("zoom_in_direction", 1)
        if isinstance(raw_direction, str):
            raw_direction = -1 if raw_direction.casefold() in {"inverted", "-1"} else 1
        profile = cls(
            name=name,
            orbit_button=str(values.get("orbit_button", "middle")).casefold(),
            orbit_modifiers=orbit_modifiers,
            pan_button=str(values.get("pan_button", "middle")).casefold(),
            pan_modifiers=pan_modifiers,
            zoom_axis=str(values.get("zoom_axis", "vertical")).casefold(),
            zoom_in_direction=int(raw_direction),
            description=str(values.get("description", "")),
        )
        return profile.validate()


def _normalize_modifiers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        pieces = [value]
    elif isinstance(value, Iterable):
        pieces = list(value)
    else:
        pieces = [value]
    normalized: list[str] = []
    for piece in pieces:
        fragments = str(piece).replace(",", "+").split("+")
        for fragment in fragments:
            modifier = fragment.strip().casefold()
            if not modifier or modifier == "none":
                continue
            if modifier not in MODIFIER_OPTIONS:
                raise ValueError(f"Unsupported keyboard modifier: {modifier!r}.")
            if modifier not in normalized:
                normalized.append(modifier)
    return tuple(normalized)


def default_profile_data() -> dict[str, dict[str, Any]]:
    """Return built-in profiles as JSON-compatible dictionaries.

    The names are convenience starting points only.  Runtime behavior is
    selected entirely by these mappings, not by application detection.
    """

    generic = {
        "orbit_button": "middle",
        "orbit_modifiers": [],
        "pan_button": "middle",
        "pan_modifiers": ["shift"],
        "zoom_axis": "vertical",
        "zoom_in_direction": 1,
        "description": "Middle-drag orbit, Shift + middle-drag pan, wheel zoom.",
    }
    profiles = {
        "Generic 3D": generic,
        "Blender": {**generic, "description": "Blender-style middle-drag navigation."},
        "Maya": {
            **generic,
            "orbit_modifiers": ["alt"],
            "pan_modifiers": ["alt", "shift"],
            "description": "Alt + middle-drag orbit and Alt + Shift + middle-drag pan.",
        },
        "3ds Max": {**generic, "description": "3ds Max-style middle-drag navigation."},
        "Cinema 4D": {**generic, "description": "Cinema 4D-style middle-drag navigation."},
        "Fusion 360": {**generic, "description": "Fusion 360-style configurable starting point."},
        "CAD / SolidWorks": {
            **generic,
            "description": "CAD-style middle-drag navigation starting point.",
        },
        "Unity / Unreal": {**generic, "description": "Game-editor middle-drag navigation starting point."},
    }
    return profiles


def get_profile(
    name: str,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> InputProfile:
    """Resolve a profile, falling back to the generic mapping if necessary."""

    available = profiles or default_profile_data()
    raw = available.get(name)
    if raw is None:
        raw = available.get("Generic 3D", default_profile_data()["Generic 3D"])
        name = "Generic 3D"
    return InputProfile.from_dict(str(name), raw)


def profile_names(profiles: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[str, ...]:
    available = profiles or default_profile_data()
    return tuple(str(name) for name in available)
