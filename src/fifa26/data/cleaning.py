"""Data cleaning and filtering stage."""
from __future__ import annotations

import pandas as pd


class MatchCleaner:
    """Cleans and filters the raw match DataFrame.

    Single responsibility: turn the raw dataset into a tidy, analysis-ready
    frame. It does not know where the data came from nor what happens next.

    Steps:
      * parse dates and keep matches from `min_year` onwards,
      * drop rows with missing scores and cast scores to int,
      * normalise the `neutral` flag to bool,
      * (optionally) keep only teams with enough matches in the window so the
        strength estimates are not driven by one or two games.
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
        return series.astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)

    def _filter_rare_teams(self, df: pd.DataFrame) -> pd.DataFrame:
        appearances = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        valid = set(appearances[appearances >= self._min_matches_per_team].index)
        return df[df["home_team"].isin(valid) & df["away_team"].isin(valid)]
