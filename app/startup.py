"""Optional Windows startup integration."""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "GesturesNoseTouch"
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


class StartupError(RuntimeError):
    """Raised when the Windows startup entry cannot be changed."""


def set_start_with_windows(enabled: bool) -> None:
    """Create or remove a per-user startup entry in HKCU."""

    if os.name != "nt":
        if enabled:
            raise StartupError("Start with Windows is only available on Windows.")
        return

    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _RUN_KEY,
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, _startup_command())
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
    except OSError as exc:
        raise StartupError(f"Could not update the Windows startup entry: {exc}") from exc


def _startup_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'

    project_root = Path(__file__).resolve().parents[1]
    # repr() produces a quoted Python string with escaped backslashes, while
    # the outer quotes keep the Registry command a single command-line value.
    code = (
        f"import sys; sys.path.insert(0, {str(project_root)!r}); "
        "from app.main import main; main()"
    )
    return f'"{sys.executable}" -c "{code}"'
