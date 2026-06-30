"""Entrada por linea con un prompt o caracter fijo.
"""
from __future__ import annotations

import sys

from fifa26.cli import ansi

SYMBOL = "> "


def stdin_is_tty() -> bool:
    return sys.stdin.isatty()


def read_line(symbol: str = SYMBOL) -> str:
    """Lee una linea confirmada con Enter mostrando el prompt
      fijo antes del cursor.
    """
    try:
        return input("  " + ansi.bold(symbol)).strip()
    except EOFError as exc:  # Ctrl-D o pipe agotado
        raise KeyboardInterrupt from exc
