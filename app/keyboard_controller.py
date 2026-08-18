"""Keyboard shortcut parsing and emission through pynput."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any


class ShortcutError(ValueError):
    """Raised when a shortcut cannot be parsed."""


@dataclass(frozen=True)
class ParsedShortcut:
    modifiers: tuple[Any, ...]
    key: Any
    display: str


_MODIFIER_NAMES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "win": "cmd",
    "windows": "cmd",
    "cmd": "cmd",
    "command": "cmd",
}

_SPECIAL_NAMES = {
    "space": "space",
    "tab": "tab",
    "enter": "enter",
    "return": "enter",
    "esc": "esc",
    "escape": "esc",
    "backspace": "backspace",
    "delete": "delete",
    "insert": "insert",
    "home": "home",
    "end": "end",
    "pageup": "page_up",
    "page up": "page_up",
    "pagedown": "page_down",
    "page down": "page_down",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


def parse_shortcut(shortcut: str) -> ParsedShortcut:
    """Parse strings such as ``Ctrl + Shift + S`` or ``Escape``."""

    try:
        from pynput import keyboard
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise ShortcutError(
            "pynput is not installed. Run python -m pip install -r requirements.txt."
        ) from exc

    tokens = [token.strip() for token in re.split(r"\s*\+\s*", shortcut) if token.strip()]
    if not tokens:
        raise ShortcutError("Enter a shortcut such as Alt + Tab or Space.")

    modifiers: list[Any] = []
    main_keys: list[Any] = []
    seen_modifiers: set[str] = set()
    for token in tokens:
        normalized = token.casefold()
        if normalized in _MODIFIER_NAMES:
            modifier_name = _MODIFIER_NAMES[normalized]
            if modifier_name in seen_modifiers:
                raise ShortcutError(f"Modifier {token!r} is listed more than once.")
            seen_modifiers.add(modifier_name)
            modifiers.append(getattr(keyboard.Key, modifier_name))
            continue
        main_keys.append(_key_for_token(keyboard, token))

    if len(main_keys) != 1:
        raise ShortcutError("A shortcut must contain exactly one non-modifier key.")
    return ParsedShortcut(
        modifiers=tuple(modifiers),
        key=main_keys[0],
        display=" + ".join(tokens),
    )


def _key_for_token(keyboard: Any, token: str) -> Any:
    normalized = token.casefold()
    special_name = _SPECIAL_NAMES.get(normalized)
    if special_name:
        return getattr(keyboard.Key, special_name)
    if re.fullmatch(r"f(?:[1-9]|1[0-9]|2[0-4])", normalized):
        return getattr(keyboard.Key, normalized)
    if len(token) == 1 and token.isprintable():
        return keyboard.KeyCode.from_char(token.lower())
    raise ShortcutError(
        f"Unknown key {token!r}. Use a letter, number, Space, Escape, Tab, "
        "arrow, function key, or a listed modifier."
    )


class KeyboardController:
    """Emit a parsed shortcut only when the gesture state machine requests it."""

    def __init__(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "pynput is not installed. Run python -m pip install -r requirements.txt."
            ) from exc
        self._keyboard = keyboard
        self._controller = keyboard.Controller()
        self._lock = threading.Lock()

    def press_shortcut(self, shortcut: str) -> None:
        parsed = parse_shortcut(shortcut)
        pressed_modifiers: list[Any] = []
        with self._lock:
            try:
                for modifier in parsed.modifiers:
                    self._controller.press(modifier)
                    pressed_modifiers.append(modifier)
                self._controller.press(parsed.key)
                self._controller.release(parsed.key)
            finally:
                for modifier in reversed(pressed_modifiers):
                    self._controller.release(modifier)
