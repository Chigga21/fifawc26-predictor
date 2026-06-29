"""Domain entities and value objects.

This module is the innermost layer of the Clean Architecture. It contains the
pure business concepts of the prediction problem and must NOT depend on any
external/IO library (pandas, pymc, xgboost, matplotlib, ...). Everything here is
immutable (`frozen=True`) so entities behave as value objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Outcome(Enum):
    """The three possible results of a football match (the 1X2 market)."""

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
    """A single international match (full-time score, no penalties)."""

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
    """Dixon-Coles offensive/defensive ratings for one team.

    Both ratings enter the log scoring rate as `attack_team - defense_opponent`:
    `attack`  > 0 -> scores more goals than an average team.
    `defense` > 0 -> concedes fewer goals than average (stronger defence).
    """

    team: str
    attack: float
    defense: float


@dataclass(frozen=True)
class ScoreMatrix:
    """Joint probability of every exact scoreline for a fixture.

    `matrix[i, j]` = P(home scores i, away scores j). It is the bridge between
    the expected goals (lambdas) produced by a model and the 1X2 outcome
    probabilities derived downstream.
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
    """1X2 probabilities for a fixture and the resulting most likely outcome."""

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
