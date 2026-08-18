"""Application entry point: ``python -m app.main``."""

from __future__ import annotations

if __package__:
    from .gui import launch
else:  # Support PyInstaller's script entry-point mode as well.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.gui import launch


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
