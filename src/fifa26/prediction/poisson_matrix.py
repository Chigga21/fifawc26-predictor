"""Construye la matriz conjunta de marcadores a partir de los goles esperados.

@author Chigga21
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from fifa26.domain.entities import ScoreMatrix


class PoissonMatrixBuilder:
    """Convierte un par de goles esperados en una matriz de marcadores.
    """

    def __init__(self, max_goals: int = 10, rho: float = 0.0) -> None:
        self._max_goals = max_goals
        self._rho = rho

    def build(
        self,
        home_team: str,
        away_team: str,
        lambda_home: float,
        lambda_away: float,
    ) -> ScoreMatrix:
        goals = np.arange(self._max_goals + 1)
        home_pmf = poisson.pmf(goals, lambda_home)
        away_pmf = poisson.pmf(goals, lambda_away)
        matrix = np.outer(home_pmf, away_pmf)

        matrix = self._apply_tau(matrix, lambda_home, lambda_away)
        matrix = matrix / matrix.sum()
        return ScoreMatrix(
            home_team=home_team,
            away_team=away_team,
            lambda_home=float(lambda_home),
            lambda_away=float(lambda_away),
            matrix=matrix,
        )

    def _apply_tau(self, matrix: np.ndarray, lh: float, la: float) -> np.ndarray:
        rho = self._rho
        if rho == 0.0:
            return matrix
        m = matrix.copy()
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
        return np.clip(m, 0.0, None)
