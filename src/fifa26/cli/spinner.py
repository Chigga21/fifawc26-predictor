"""Indicadores de carga construidos sobre la animacion desacoplada.

run_with_spinner muestra el spinner clasico que cierra en done para las tareas
que animan limpio, y run_with_dots usa puntos suspensivos para las tareas
pesadas donde un spinner se veria congelado, como Dixon-Coles o el MCMC. En
ambos la animacion corre en su propio hilo, independiente del computo.

@author Chigga21
"""
from __future__ import annotations

from typing import Callable, TypeVar

from fifa26.cli.indicator import run_with_indicator

T = TypeVar("T")


def run_with_spinner(label: str, fn: Callable[[], T]) -> T:
    """Corre fn con el spinner animado y cierra con el marcador done."""
    return run_with_indicator(label, fn, style="spinner")


def run_with_dots(label: str, fn: Callable[[], T], *, dim: bool = True) -> T:
    """Corre fn con puntos suspensivos animados, sin marcador congelable."""
    return run_with_indicator(label, fn, style="dots", dim=dim)
