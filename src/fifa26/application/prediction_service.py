"""Prediction service: turn a chosen fixture into a full match forecast.

Wraps the artifacts produced by `Trainer` (a trained `GoalModel`, the Poisson matrix
builder and the known team list) behind a single `predict(home, away, neutral)` call.
Both the interactive UI (result screen) and the batch pipeline (example fixture) depend
on this service instead of re-wiring the goal-model -> matrix -> outcome chain.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fifa26.application.training import TrainedArtifacts
from fifa26.domain.entities import MatchPrediction, ScoreMatrix
from fifa26.domain.interfaces import GoalModel
from fifa26.prediction.outcome import OutcomeCalculator
from fifa26.prediction.poisson_matrix import PoissonMatrixBuilder


@dataclass(frozen=True)
class MatchForecast:
    """The complete prediction for one fixture."""

    score_matrix: ScoreMatrix
    prediction: MatchPrediction
    top_scorelines: list[tuple[str, float]]


class PredictionService:
    def __init__(
        self,
        model: GoalModel,
        matrix_builder: PoissonMatrixBuilder,
        outcome_calculator: OutcomeCalculator,
        teams: list[str],
    ) -> None:
        self._model = model
        self._matrix_builder = matrix_builder
        self._outcome = outcome_calculator
        self._teams = list(teams)

    @classmethod
    def from_artifacts(
        cls, artifacts: TrainedArtifacts, outcome_calculator: OutcomeCalculator
    ) -> "PredictionService":
        return cls(
            model=artifacts.best_model,
            matrix_builder=artifacts.matrix_builder,
            outcome_calculator=outcome_calculator,
            teams=artifacts.teams,
        )

    @property
    def teams(self) -> list[str]:
        return self._teams

    @property
    def model_name(self) -> str:
        return self._model.name

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        tournament: str = "FIFA World Cup",
        top_n: int = 10,
    ) -> MatchForecast:
        fixture = pd.DataFrame(
            [
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "neutral": neutral,
                    "tournament": tournament,
                }
            ]
        )
        lambda_home, lambda_away = self._model.predict_expected_goals(fixture)
        score_matrix = self._matrix_builder.build(
            home_team, away_team, float(lambda_home[0]), float(lambda_away[0])
        )
        prediction = self._outcome.to_prediction(score_matrix)
        top = self._outcome.top_scorelines(score_matrix, top_n=top_n)
        return MatchForecast(
            score_matrix=score_matrix, prediction=prediction, top_scorelines=top
        )
