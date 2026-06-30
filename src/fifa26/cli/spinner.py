"""Spinner sencillo que se muestra como indicador que 
un comando está ejecutándose (cargando)

@author Chigga21
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, TypeVar

from fifa26.cli import ansi

T = TypeVar("T")

_FRAMES = ("|", "/", "-", "\\")


def run_with_spinner(label: str, fn: Callable[[], T], interval: float = 0.1) -> T:
    if not sys.stdout.isatty():
        print(f"  * {label}")
        return fn()

    result: list[T] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(fn())
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    ansi.hide_cursor()
    thread.start()
    try:
        tick = 0
        while thread.is_alive():
            frame = _FRAMES[tick % len(_FRAMES)]
            line = f"  {ansi.focused('[' + frame + ']')} {label}"
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
            tick += 1
            time.sleep(interval)
        thread.join()
    finally:
        sys.stdout.write("\r\033[K")
        ansi.show_cursor()

    if error:
        raise error[0]
    print(f"  {ansi.active('[done]')} {label}")
    sys.stdout.flush()
    return result[0]
