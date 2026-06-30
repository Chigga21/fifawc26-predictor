"""Etapa de limpieza y filtrado de datos.
@author Chigga21
"""
from __future__ import annotations

import pandas as pd


class MatchCleaner:
    """Limpia y filtra el DataFrame crudo de partidos.
    """

    def __init__(self, min_year: int = 2018, min_matches_per_team: int = 8) -> None:
        self._min_year = min_year
        self._min_matches_per_team = min_matches_per_team

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home_score", "away_score"])
        df = df[df["date"].dt.year >= self._min_year].copy()

        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        df["neutral"] = self._as_bool(df["neutral"])
        df["year"] = df["date"].dt.year

        df = self._filter_rare_teams(df)
        return df.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _as_bool(series: pd.Series) -> pd.Series:
        if series.dtype == bool:
            return series
        truthy = {"TRUE", "1", "1.0", "YES", "Y", "T"}
        falsy = {"FALSE", "0", "0.0", "NO", "N", "F", "NAN", ""}
        normalized = series.astype(str).str.strip().str.upper()
        mapping = {v: True for v in truthy}
        mapping.update({v: False for v in falsy})
        return normalized.map(mapping).fillna(False).astype(bool)

    def _filter_rare_teams(self, df: pd.DataFrame) -> pd.DataFrame:
        appearances = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        valid = set(appearances[appearances >= self._min_matches_per_team].index)
        return df[df["home_team"].isin(valid) & df["away_team"].isin(valid)]
