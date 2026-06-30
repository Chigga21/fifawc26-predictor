"""Selector de equipos de la UI, menu de flechas con la UI
y flitro de texto

@author Chigga21
"""
from __future__ import annotations

from collections.abc import Iterable

from fifa26.cli import ansi, menu
from fifa26.cli.prompt import read_line

_WINDOW = 12  # coincidencias listadas a la vez


class TeamSelector:
    def __init__(self, teams: Iterable[str], window: int = _WINDOW) -> None:
        self._teams = list(teams)
        self._window = window

    def select(self, prompt: str, exclude: str | None = None) -> str | None:
        """Devuelve el equipo elegido, o None si el usuario cancela"""
        if menu.supported():
            return menu.arrow_select(
                prompt, self._teams, exclude=exclude, window=self._window
            )
        return self._select_line_based(prompt, exclude)

    def _select_line_based(self, prompt: str, exclude: str | None) -> str | None:
        """Escribir para filtrar si no hay un TTY"""
        pool = [t for t in self._teams if t != exclude]
        print()
        print(ansi.heading(prompt))
        print(
            "  "
            + ansi.hint(
                "type part of the name and press Enter to filter; "
                "then type the number to choose. (empty Enter cancels)"
            )
        )

        shown: list[str] = []
        while True:
            raw = read_line()
            if raw == "":
                return None  # cancelar
            if raw.isdigit():
                chosen = self._pick(shown, int(raw))
                if chosen is not None:
                    print("  " + ansi.active(f"[x] {chosen}"))
                    return chosen
                continue
            shown = self._show_matches(pool, raw)

    def _show_matches(self, pool: list[str], text: str) -> list[str]:
        matches = _filter(pool, text)
        if not matches:
            print("  " + ansi.error(f"no matches for '{text}'"))
            return []
        shown = matches[: self._window]
        for i, team in enumerate(shown, start=1):
            print(f"  [{i:>2}] {team}")
        if len(matches) > self._window:
            print("  " + ansi.hint(f"... {len(matches) - self._window} more; refine the filter"))
        print("  " + ansi.hint("type the number to choose, or filter again"))
        return shown

    def _pick(self, shown: list[str], number: int) -> str | None:
        index = number - 1
        if 0 <= index < len(shown):
            return shown[index]
        if not shown:
            print("  " + ansi.error("filter first to see options"))
        else:
            print("  " + ansi.error(f"number out of range (1-{len(shown)})"))
        return None

def _filter(teams: list[str], text: str) -> list[str]:
    if not text:
        return teams
    needle = text.lower()
    starts = [t for t in teams if t.lower().startswith(needle)]
    contains = [t for t in teams if needle in t.lower() and not t.lower().startswith(needle)]
    return starts + contains
