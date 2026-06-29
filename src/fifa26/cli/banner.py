"""ASCII header for the program: the WC26 PREDICTOR logo and its author.

The logo is the Big Money-nw rendering of "WC26 PREDICTOR" stored in `TITLE.txt` at the
project root. It is loaded at runtime so the artwork stays editable in one place; a
compact embedded fallback is used if the file is missing.
"""
from __future__ import annotations

from pathlib import Path

from fifa26.cli import ansi

AUTHOR = "Chigga21"
SUBTITLE = "World Cup 2026 Match Predictor with Machine Learning"

# Project root is three levels up from this file: cli/ -> fifa26/ -> src/ -> root.
_TITLE_PATH = Path(__file__).resolve().parents[3] / "TITLE.txt"

_FALLBACK = r"""
 _    _  ___ ___  __    ___  ___ ___ ___ ___ ___ _____ ___  ___
| |  | |/ __|_  )/ /   | _ \| _ \ __|   \_ _/ __|_   _/ _ \| _ \
| |/\| | (__ / // _ \  |  _/|   / _|| |) | | (__  | || (_) |   /
|__/\__|\___/___\___/  |_|  |_|_\___|___/___\___| |_| \___/|_|_\
""".strip("\n")


def _load_logo() -> str:
    try:
        text = _TITLE_PATH.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK
    # Drop trailing blank lines from the source art.
    lines = text.rstrip("\n").split("\n")
    return "\n".join(lines) if any(line.strip() for line in lines) else _FALLBACK


def render_header() -> None:
    """Print the coloured logo, subtitle and author block."""
    logo = _load_logo()
    print(ansi.title(logo))
    print()
    print(ansi.heading(SUBTITLE))
    print(ansi.title("   " + f"By {AUTHOR}"))
    print()
