"""Arrow-key selectable menu for the interactive terminal UI.

`arrow_select` shows a scrollable list where the user moves the highlight with the up/down
arrows, narrows the list by typing, and confirms with Enter. It redraws a fixed-height
option block *in place* (moving the cursor up and clearing lines) — the single screen
region the UI updates instead of streaming, active only while a menu is open. Everything
above the block (logo, previous steps) is left untouched, so the session still reads as a
continuous log once the menu closes.

It needs a real TTY and POSIX raw mode (stdlib `termios`/`tty`). When stdin/stdout are not
terminals (pipes, tests) it cannot run; callers check `supported()` and fall back to the
line-based `TeamSelector`. Raw mode is always restored and the cursor re-shown via the
`finally` block, even on Ctrl-C.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable

from fifa26.cli import ansi

try:  # POSIX only; absent on Windows.
    import select
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - non-POSIX platforms
    _HAS_TERMIOS = False

_WINDOW = 12  # most options visible at a time

# Symbolic keys decoded from raw input.
_UP = "up"
_DOWN = "down"
_ENTER = "enter"
_CANCEL = "cancel"
_BACKSPACE = "backspace"


def supported() -> bool:
    """True when an interactive arrow menu can run (POSIX raw mode on a TTY)."""
    return _HAS_TERMIOS and sys.stdin.isatty() and sys.stdout.isatty()


def arrow_select(
    title: str,
    options: Iterable[str],
    *,
    exclude: str | None = None,
    window: int = _WINDOW,
) -> str | None:
    """Pick one option with the arrow keys; return it, or ``None`` if cancelled.

    Up/Down move the highlight (the window scrolls past ``window`` items), printable
    characters narrow the list, Backspace deletes the last filter character, Enter confirms
    and Esc / Ctrl-C / ``q`` on an empty filter cancel.
    """
    pool = [o for o in options if o != exclude]
    print()
    print(ansi.heading(title))
    print(
        "  "
        + ansi.hint(
            "use up/down arrows to move, type to filter, Enter to select, Esc to cancel"
        )
    )

    menu = _Menu(pool, window)
    ansi.hide_cursor()
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        menu.render()
        while True:
            key = _read_key()
            if key == _ENTER:
                chosen = menu.current()
                if chosen is not None:
                    menu.finish()
                    print("  " + ansi.active(f"[x] {chosen}"))
                    return chosen
            elif key == _UP:
                menu.move(-1)
            elif key == _DOWN:
                menu.move(1)
            elif key == _CANCEL:
                menu.finish()
                print("  " + ansi.hint("cancelled"))
                return None
            elif key == _BACKSPACE:
                menu.backspace()
            elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                menu.type(key)
            else:
                continue
            menu.render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        ansi.show_cursor()


# --------------------------------------------------------------------- input decode
def _read_key() -> str:
    """Read one logical key from the raw stdin fd, decoding arrow escape sequences.

    Reads straight from the file descriptor with ``os.read`` (not buffered
    ``sys.stdin.read``) so that ``select`` and the read stay consistent: a buffered read
    could pull the whole ``\\x1b[B`` arrow sequence into Python's buffer where ``select``
    on the fd no longer sees it, making a real arrow key look like a bare Esc.
    """
    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)
    if not ch:  # EOF
        return _CANCEL
    if ch in (b"\r", b"\n"):
        return _ENTER
    if ch in (b"\x7f", b"\b"):
        return _BACKSPACE
    if ch == b"\x03":  # Ctrl-C
        raise KeyboardInterrupt
    if ch == b"\x1b":  # ESC: either a bare Esc or the start of an arrow sequence.
        if not _pending(fd):
            return _CANCEL
        seq = os.read(fd, 2)  # the rest of an arrow key arrives together, e.g. b"[A"
        if seq[:1] == b"[":
            code = seq[1:2]
            if code == b"A":
                return _UP
            if code == b"B":
                return _DOWN
        return ""  # left/right or unhandled sequence: ignore
    if ch == b"q":
        return _CANCEL
    try:
        return ch.decode()
    except UnicodeDecodeError:  # partial multibyte char: ignore (ASCII filtering suffices)
        return ""


def _pending(fd: int) -> bool:
    """True if more input is already available on the fd (distinguishes Esc from arrows)."""
    ready, _, _ = select.select([fd], [], [], 0.05)
    return bool(ready)


# ----------------------------------------------------------------------- menu state
class _Menu:
    """Holds the filter/cursor/scroll state and renders a fixed-height block in place."""

    def __init__(self, pool: list[str], window: int) -> None:
        self._pool = pool
        self._window = window
        self._filter = ""
        self._matches = list(pool)
        self._index = 0
        self._offset = 0
        self._lines = 0  # lines printed by the previous render (for in-place redraw)

    def current(self) -> str | None:
        return self._matches[self._index] if self._matches else None

    def move(self, delta: int) -> None:
        if not self._matches:
            return
        self._index = max(0, min(len(self._matches) - 1, self._index + delta))
        self._scroll()

    def type(self, ch: str) -> None:
        self._filter += ch
        self._refilter()

    def backspace(self) -> None:
        if self._filter:
            self._filter = self._filter[:-1]
            self._refilter()

    def render(self) -> None:
        """Redraw the option block over its previous position."""
        block = self._compose()
        if self._lines:
            ansi.move_up(self._lines)
        sys.stdout.write("".join("\r\033[2K" + line + "\n" for line in block))
        sys.stdout.flush()
        self._lines = len(block)

    def finish(self) -> None:
        """Erase the block so the chosen line can be printed cleanly in its place."""
        if self._lines:
            ansi.move_up(self._lines)
            sys.stdout.write("\r\033[J")  # clear from here to the end of the screen
            sys.stdout.flush()
        self._lines = 0

    # ------------------------------------------------------------------- internals
    def _refilter(self) -> None:
        from fifa26.cli.selector import _filter  # reuse the same ranking as the list UI

        self._matches = _filter(self._pool, self._filter)
        self._index = 0
        self._offset = 0

    def _scroll(self) -> None:
        if self._index < self._offset:
            self._offset = self._index
        elif self._index >= self._offset + self._window:
            self._offset = self._index - self._window + 1

    def _compose(self) -> list[str]:
        """Build exactly ``window + 1`` lines so the block height never changes."""
        rows: list[str] = []
        window = self._matches[self._offset : self._offset + self._window]
        for pos, team in enumerate(window):
            absolute = self._offset + pos
            if absolute == self._index:
                rows.append("  " + ansi.focused(f"[*] {team}"))
            else:
                rows.append("  " + f"[ ] {team}")
        rows += [""] * (self._window - len(rows))  # pad to a fixed height

        if not self._matches:
            footer = ansi.error(f"no matches for '{self._filter}'")
        else:
            shown = f"{self._index + 1}/{len(self._matches)}"
            filt = f"   filter: '{self._filter}'" if self._filter else ""
            footer = ansi.hint(f"{shown}{filt}")
        rows.append("  " + footer)
        return rows
