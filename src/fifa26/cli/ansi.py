"""Utilidades de estilo ANSI para la UI de la terminal 
Controla además el color y estilo semántico de los textos 
mostrados en pantalla (negritas, bajo enfasis, etc...)

@author Chigga21
"""
from __future__ import annotations

import os
import sys

# ----------------------------------------------------------------- codigos SGR crudos
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
    """El color esta activo salvo que se desactive
      o el stream no sea una terminal"""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FIFA26_NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def _wrap(text: str, *codes: str) -> str:
    if not codes or not color_enabled():
        return text
    return "".join(codes) + text + RESET

def bold(text: str) -> str:
    return _wrap(text, BOLD)


def dim(text: str) -> str:
    return _wrap(text, DIM)


def color(text: str, name: str, *, bold_: bool = False) -> str:
    codes = (_FG.get(name, ""),)
    if bold_:
        codes = (BOLD,) + codes
    return _wrap(text, *codes)


def title(text: str) -> str:
    """El logo y nombre del producto, en blanco y negrita."""
    return color(text, "bright_white", bold_=True)


def heading(text: str) -> str:
    """Un encabezado de pantalla o seccion."""
    return color(text, "cyan", bold_=True)


def focused(text: str) -> str:
    """La opcion bajo el cursor en un menu, junto a un marcador entre corchetes"""
    return color(text, "bright_yellow", bold_=True)


def active(text: str) -> str:
    """Una eleccion que el usuario ya fijo, junto al marcador [X]"""
    return color(text, "bright_green", bold_=True)


def confirm(text: str) -> str:
    """Un resultado final o pronostico que conviene destacar"""
    return color(text, "bright_green", bold_=True)


def hint(text: str) -> str:
    """Texto de ayuda de bajo enfasis, como los atajos de teclado"""
    return dim(text)


def error(text: str) -> str:
    return color(text, "bright_red", bold_=True)

def hide_cursor() -> None:
    if color_enabled():
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


def show_cursor() -> None:
    if color_enabled():
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def move_up(lines: int) -> None:
    """Sube el cursor el numero de filas dado, para redibujar el estilo
    del menu
    """
    if lines > 0 and sys.stdout.isatty():
        sys.stdout.write(f"\033[{lines}A")
        sys.stdout.flush()


def clear_line() -> None:
    """Borra la linea actual y devuelve el cursor a su inicio."""
    if sys.stdout.isatty():
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
