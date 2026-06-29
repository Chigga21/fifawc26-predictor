"""Line-based input with a fixed `> ` prompt — the heart of the stream UI.

The whole interface is a continuous stream of output: nothing clears the screen and no
selection is ever auto-confirmed on a single keystroke. Every interaction is read as a
*line* and confirmed with Enter. The `> ` marker is always printed before the cursor so
the user can see, at any moment, that the program is waiting for input.

When stdin is not a TTY (piped input, tests) `input()` still works line by line, so the
same code path serves both interactive and automated use. End-of-input (Ctrl-D or an
exhausted pipe) ends the session cleanly via `KeyboardInterrupt`.
"""
from __future__ import annotations

import sys

from fifa26.cli import ansi

SYMBOL = "> "


def stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def read_line(symbol: str = SYMBOL) -> str:
    """Read one Enter-confirmed line, showing the fixed `> ` prompt before the cursor.

    The returned text is stripped. Raises `KeyboardInterrupt` on EOF so callers can let
    the session end cleanly instead of looping forever.
    """
    try:
        return input("  " + ansi.bold(symbol)).strip()
    except EOFError as exc:  # Ctrl-D or exhausted pipe.
        raise KeyboardInterrupt from exc
