"""Team selector for the interactive UI: arrow menu on a TTY, type-to-filter otherwise.

When stdin/stdout are real terminals the selection is delegated to `cli.menu.arrow_select`
(up/down to move, type to filter, Enter to confirm). When they are not (pipes, tests) the
selector falls back to the original line-based path: with ~240 teams a flat numbered dump
is unusable, so the user *types to filter* and then picks a match by its number. That
fallback prints as a continuous stream — the matching list is written once per filter,
never redrawn over a cleared screen — and the fixed `> ` prompt is shown whenever input is
awaited. The same line-based path works with or without a TTY, so automated runs stay
deterministic.
"""
from __future__ import annotations

from collections.abc import Iterable

from fifa26.cli import ansi, menu
from fifa26.cli.prompt import read_line

_WINDOW = 12  # most matches listed at a time


class TeamSelector:
    def __init__(self, teams: Iterable[str], window: int = _WINDOW) -> None:
        self._teams = list(teams)
        self._window = window

    def select(self, prompt: str, exclude: str | None = None) -> str | None:
        """Return the chosen team, or `None` if the user cancels."""
        if menu.supported():
            return menu.arrow_select(
                prompt, self._teams, exclude=exclude, window=self._window
            )
        return self._select_line_based(prompt, exclude)

    def _select_line_based(self, prompt: str, exclude: str | None) -> str | None:
        """Fallback used without a TTY: type to filter, then enter the number."""
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
                return None  # cancel
            if raw.isdigit():
                chosen = self._pick(shown, int(raw))
                if chosen is not None:
                    print("  " + ansi.active(f"[x] {chosen}"))
                    return chosen
                continue
            shown = self._show_matches(pool, raw)

    # --------------------------------------------------------------------- helpers
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


# --------------------------------------------------------------------- filtering
def _filter(teams: list[str], text: str) -> list[str]:
    if not text:
        return teams
    needle = text.lower()
    starts = [t for t in teams if t.lower().startswith(needle)]
    contains = [t for t in teams if needle in t.lower() and not t.lower().startswith(needle)]
    return starts + contains
