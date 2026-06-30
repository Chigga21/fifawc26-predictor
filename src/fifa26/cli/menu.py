"""Menu navegable con teclas de flecha para la UI interactiva de terminal.
Necesita necesariamente de un TTY para funcionar chido!!
@author Chigga21
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable

from fifa26.cli import ansi

try:  # Solo POSIX, si usas windows olvidate xd
    import select
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  
    _HAS_TERMIOS = False

_WINDOW = 12  # opciones visibles a la vez

_UP = "up"
_DOWN = "down"
_ENTER = "enter"
_CANCEL = "cancel"
_BACKSPACE = "backspace"


def supported() -> bool:
    """Verdadero cuando puede correr el menu de flechas, modo raw POSIX en un TTY."""
    return _HAS_TERMIOS and sys.stdin.isatty() and sys.stdout.isatty()


def arrow_select(
    title: str,
    options: Iterable[str],
    *,
    exclude: str | None = None,
    window: int = _WINDOW,
) -> str | None:
    """Elige una opcion con las flechas y la devuelve, o None si se cancela.

    Las flechas mueven el resaltado y la ventana se desplaza, los caracteres imprimibles
    filtran, Backspace borra el ultimo caracter del filtro, Enter confirma y Esc, Ctrl-C o
    la tecla q con el filtro vacio cancelan.
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


def _read_key() -> str:
    """Lee una tecla logica de stdin y decodifica las secuencias de flecha
    de la misma
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
    if ch == b"\x1b":  
        if not _pending(fd):
            return _CANCEL
        seq = os.read(fd, 2)  
        if seq[:1] == b"[":
            code = seq[1:2]
            if code == b"A":
                return _UP
            if code == b"B":
                return _DOWN
        return ""  # secuencia lateral o no manejada se ignora
    if ch == b"q":
        return _CANCEL
    try:
        return ch.decode()
    except UnicodeDecodeError:  
        return ""


def _pending(fd: int) -> bool:
    """Verdadero si ya hay mas entrada en el fd, distingue Esc de las flechas."""
    ready, _, _ = select.select([fd], [], [], 0.05)
    return bool(ready)


class _Menu:
    """Guarda el estado de filtro, cursor y scroll y redibuja un bloque fijo en sitio."""

    def __init__(self, pool: list[str], window: int) -> None:
        self._pool = pool
        self._window = window
        self._filter = ""
        self._matches = list(pool)
        self._index = 0
        self._offset = 0
        self._lines = 0  

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
        """Redibuja el bloque de opciones sobre su posicion anterior"""
        block = self._compose()
        if self._lines:
            ansi.move_up(self._lines)
        sys.stdout.write("".join("\r\033[2K" + line + "\n" for line in block))
        sys.stdout.flush()
        self._lines = len(block)

    def finish(self) -> None:
        """Borra el bloque para imprimir limpia la linea elegida en su lugar"""
        if self._lines:
            ansi.move_up(self._lines)
            sys.stdout.write("\r\033[J") 
            sys.stdout.flush()
        self._lines = 0

    def _refilter(self) -> None:
        from fifa26.cli.selector import _filter  

        self._matches = _filter(self._pool, self._filter)
        self._index = 0
        self._offset = 0

    def _scroll(self) -> None:
        if self._index < self._offset:
            self._offset = self._index
        elif self._index >= self._offset + self._window:
            self._offset = self._index - self._window + 1

    def _compose(self) -> list[str]:
        """Construye exactamente window+1 lineas para que la altura no cambie."""
        rows: list[str] = []
        window = self._matches[self._offset : self._offset + self._window]
        for pos, team in enumerate(window):
            absolute = self._offset + pos
            if absolute == self._index:
                rows.append("  " + ansi.focused(f"[*] {team}"))
            else:
                rows.append("  " + f"[ ] {team}")
        rows += [""] * (self._window - len(rows))  

        if not self._matches:
            footer = ansi.error(f"no matches for '{self._filter}'")
        else:
            shown = f"{self._index + 1}/{len(self._matches)}"
            filt = f"   filter: '{self._filter}'" if self._filter else ""
            footer = ansi.hint(f"{shown}{filt}")
        rows.append("  " + footer)
        return rows
