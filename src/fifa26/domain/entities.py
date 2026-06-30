"""Entidades y objetos de valor del dominio.
@author Chigga21
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Outcome(Enum):
    """Los tres resultados posibles de un partido, 
    el marcador 1X2."""

    HOME_WIN = "1"
    DRAW = "X"
    AWAY_WIN = "2"

    @staticmethod
    def from_scores(home_score: int, away_score: int) -> "Outcome":
        if home_score > away_score:
            return Outcome.HOME_WIN
        if home_score < away_score:
            return Outcome.AWAY_WIN
        return Outcome.DRAW


@dataclass(frozen=True)
class Match:
    """Un partido internacional con marcador a tiempo completo sin penales"""

    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool

    @property
    def actual_outcome(self) -> Outcome:
        return Outcome.from_scores(self.home_score, self.away_score)


@dataclass(frozen=True)
class TeamStrength:
    """Ratings ofensivo y defensivo Dixon-Coles de un equipo.
    """
    team: str
    attack: float
    defense: float


@dataclass(frozen=True)
class ScoreMatrix:
    """Probabilidad conjunta de cada marcador exacto de un partido.
    La celda i, j es la probabilidad de que el local anote i
      y el visitante j. 
    """

    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray

    @property
    def max_goals(self) -> int:
        return self.matrix.shape[0] - 1


@dataclass(frozen=True)
class MatchPrediction:
    """Probabilidades 1X2 de un partido y el resultado mas probable que derivan"""

    home_team: str
    away_team: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float

    @property
    def predicted_outcome(self) -> Outcome:
        probs = {
            Outcome.HOME_WIN: self.prob_home_win,
            Outcome.DRAW: self.prob_draw,
            Outcome.AWAY_WIN: self.prob_away_win,
        }
        return max(probs, key=probs.get)
