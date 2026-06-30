"""Adaptador CSV del puerto de repositorio de partidos.
@author Chigga21
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fifa26.domain.interfaces import MatchRepository


class CsvMatchRepository(MatchRepository):
    """Carga los partidos desde el dataset international_results/results.csv.
    """

    REQUIRED_COLUMNS = (
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    )

    def __init__(self, results_csv: str | Path) -> None:
        self._path = Path(results_csv)

    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(f"No se encontró el dataset: {self._path}")
        df = pd.read_csv(self._path)
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en el dataset: {sorted(missing)}")
        return df
