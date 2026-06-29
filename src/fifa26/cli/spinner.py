"""Simple "loading dots" animation shown while a blocking step runs.

`run_with_spinner(label, fn)` executes `fn` in a worker thread and animates three dots
(`.`, `..`, `...`) after the label on the main thread until the work finishes, then
returns whatever `fn` returned (re-raising any exception).

We deliberately do *not* draw a progress bar: the underlying steps (data loading,
Dixon-Coles fit, model training) give no reliable progress signal, and a bar that fills
on a timer would be a misleading simulation that can freeze at an intermediate state.
A small set of moving dots keeps honest, constant visual feedback instead. The line is
rewritten in place with `\r`; once finished it collapses into a single `[done]` line so
the surrounding output stays a clean, continuous stream.

When stdout is not a TTY the animation is skipped: the label is printed once and `fn`
runs synchronously, so logs stay clean.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, TypeVar

from fifa26.cli import ansi

T = TypeVar("T")

_MAX_DOTS = 3


def run_with_spinner(label: str, fn: Callable[[], T], interval: float = 0.25) -> T:
    if not sys.stdout.isatty():
        print(f"  * {label}...")
        return fn()

    result: list[T] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    ansi.hide_cursor()
    thread.start()
    try:
        tick = 0
        while thread.is_alive():
            dots = "." * (tick % (_MAX_DOTS + 1))
            line = f"  {ansi.focused('[*]')} {label}{dots:<{_MAX_DOTS}}"
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
    return result[0]
