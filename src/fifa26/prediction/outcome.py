"""Deriva las probabilidades 1X2 y los marcadores mas probables de una matriz.
@author Chigga21
"""
from __future__ import annotations

import numpy as np

from fifa26.domain.entities import MatchPrediction, ScoreMatrix


class OutcomeCalculator:
  
    def to_prediction(self, score_matrix: ScoreMatrix) -> MatchPrediction:
        m = score_matrix.matrix
        prob_home = float(np.tril(m, -1).sum())
        prob_draw = float(np.trace(m))
        prob_away = float(np.triu(m, 1).sum())
        return MatchPrediction(
            home_team=score_matrix.home_team,
            away_team=score_matrix.away_team,
            prob_home_win=prob_home,
            prob_draw=prob_draw,
            prob_away_win=prob_away,
        )

    def top_scorelines(
        self, score_matrix: ScoreMatrix, top_n: int = 10
    ) -> list[tuple[str, float]]:
        """Devuelve los top_n marcadores mas probables"""
        m = score_matrix.matrix
        flat = [
            (f"{i}-{j}", float(m[i, j]))
            for i in range(m.shape[0])
            for j in range(m.shape[1])
        ]
        flat.sort(key=lambda kv: kv[1], reverse=True)
        return flat[:top_n]
