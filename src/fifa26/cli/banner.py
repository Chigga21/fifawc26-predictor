"""Encabezado ASCII del programa con el logo WC26 PREDICTOR
y su autor. Se carga desde TITLE.txt!

@author Chigga21
"""
from __future__ import annotations

from pathlib import Path

from fifa26.cli import ansi

AUTHOR = "Chigga21"
SUBTITLE = "World Cup 2026 Match Predictor with Machine Learning"

# La raiz del proyecto esta tres niveles arriba de este archivo.
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
    lines = text.rstrip("\n").split("\n")
    return "\n".join(lines) if any(line.strip() for line in lines) else _FALLBACK


def render_header() -> None:
    """Imprime el logo en color, el autor centrado y el subtitulo"""
    logo = _load_logo()
    width = max((len(line) for line in logo.splitlines()), default=0)
    author = f"By {AUTHOR}".center(width)
    print(ansi.title(logo))
    print()
    print(ansi.title(author))
    print()
    print(ansi.heading(SUBTITLE))
    print()
