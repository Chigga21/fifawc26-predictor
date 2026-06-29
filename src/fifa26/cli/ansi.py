"""ANSI styling utilities for the terminal UI (no external dependencies).

Two layers live here:

  * Low-level SGR helpers (`bold`, `dim`, `color`) and cursor control built from
    standard ANSI escape sequences.
  * Semantic state styles (`title`, `heading`, `focused`, `active`, `confirm`, `hint`,
    `error`) that name *what a piece of text means* rather than its raw colour, so the
    palette stays consistent and is defined in exactly one place.

The UI is a continuous stream of output, so there is no screen-clearing helper here:
views are simply printed one after another.

Colour is disabled automatically when `NO_COLOR` is set or when stdout is not a TTY, so
piping the program to a file or another process yields clean plain text.
"""
from __future__ import annotations

import os
import sys

# ----------------------------------------------------------------- raw SGR codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_FG = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}


def color_enabled() -> bool:
    """Colour is on unless explicitly disabled or the stream is not a terminal."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FIFA26_NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def _wrap(text: str, *codes: str) -> str:
    if not codes or not color_enabled():
        return text
    return "".join(codes) + text + RESET


# ------------------------------------------------------------------- primitives
def bold(text: str) -> str:
    return _wrap(text, BOLD)


def dim(text: str) -> str:
    return _wrap(text, DIM)


def color(text: str, name: str, *, bold_: bool = False) -> str:
    codes = (_FG.get(name, ""),)
    if bold_:
        codes = (BOLD,) + codes
    return _wrap(text, *codes)


# --------------------------------------------------------------- semantic states
def title(text: str) -> str:
    """The big logo / product name: white and bold."""
    return color(text, "bright_white", bold_=True)


def heading(text: str) -> str:
    """A screen or section heading."""
    return color(text, "cyan", bold_=True)


def focused(text: str) -> str:
    """The option currently under the cursor in a menu (paired with a [*] marker)."""
    return color(text, "bright_yellow", bold_=True)


def active(text: str) -> str:
    """A choice the user has already locked in (paired with a [x] marker)."""
    return color(text, "bright_green", bold_=True)


def confirm(text: str) -> str:
    """A final result / predicted outcome worth emphasising."""
    return color(text, "bright_green", bold_=True)


def hint(text: str) -> str:
    """Low-emphasis help text, e.g. the available key bindings."""
    return dim(text)


def error(text: str) -> str:
    return color(text, "bright_red", bold_=True)


# ------------------------------------------------------------------ cursor control
def hide_cursor() -> None:
    if color_enabled():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor() -> None:
    if color_enabled():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def move_up(lines: int) -> None:
    """Move the cursor up `lines` rows (used by the in-place arrow menu redraw).

    Gated on a TTY, not on colour: cursor positioning is not styling, so `NO_COLOR`
    must not disable it.
    """
    if lines > 0 and sys.stdout.isatty():
        sys.stdout.write(f"\033[{lines}A")
        sys.stdout.flush()


def clear_line() -> None:
    """Erase the current line and return the cursor to its start."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
