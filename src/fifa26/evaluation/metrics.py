"""1X2 evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass

from fifa26.domain.entities import MatchPrediction, Outcome


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of a model's 1X2 performance on a test set."""

    model_name: str
    accuracy: float
    n_matches: int

    def __str__(self) -> str:
        return f"{self.model_name:<14} accuracy 1X2 = {self.accuracy:.3f}  (n={self.n_matches})"


def accuracy_1x2(
    model_name: str,
    predictions: list[MatchPrediction],
    actual_outcomes: list[Outcome],
) -> EvaluationResult:
    """Share of matches where the most likely 1X2 outcome was correct."""
    if len(predictions) != len(actual_outcomes):
        raise ValueError("predicciones y resultados reales deben tener igual longitud")
    if not predictions:
        return EvaluationResult(model_name, 0.0, 0)

    hits = sum(
        pred.predicted_outcome == actual
        for pred, actual in zip(predictions, actual_outcomes)
    )
    return EvaluationResult(model_name, hits / len(predictions), len(predictions))
