"""Shared training stage: load -> clean -> split -> Dixon-Coles -> train -> pick best.

This is the single source of truth for *how a model is trained and evaluated*. Both
the batch pipeline (`PredictionPipeline`) and the interactive UI (`InteractiveApp`)
depend on the artifacts produced here instead of duplicating the sequence.

The steps are exposed granularly (`load_and_split`, `fit_features`, `train_model`) so a
caller can wrap each one in its own progress feedback: the batch pipeline prints a line,
the interactive UI runs it inside an ASCII spinner thread. `run()` is a convenience that
drives the whole sequence with an optional progress callback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from fifa26.data.cleaning import MatchCleaner
from fifa26.domain.entities import MatchPrediction, Outcome, TeamStrength
from fifa26.domain.interfaces import GoalModel, MatchRepository
from fifa26.evaluation.metrics import EvaluationResult, accuracy_1x2
from fifa26.features.dixon_coles import DixonColesEstimator
from fifa26.prediction.outcome import OutcomeCalculator
from fifa26.prediction.poisson_matrix import PoissonMatrixBuilder

# A progress hook receives a human-readable label for the step about to run.
ProgressHook = Callable[[str], None]


@dataclass
class TrainedArtifacts:
    """Everything downstream stages need once training is finished."""

    best_model: GoalModel
    best_accuracy: float
    models: list[GoalModel]
    evaluations: list[EvaluationResult]
    strengths: dict[str, TeamStrength]
    matrix_builder: PoissonMatrixBuilder
    teams: list[str]
    train: pd.DataFrame
    test: pd.DataFrame


class Trainer:
    """Runs (and remembers) the training/evaluation stages.

    Dependencies are injected; the trainer never builds its own collaborators, so
    each piece stays swappable and testable in isolation.
    """

    def __init__(
        self,
        repository: MatchRepository,
        cleaner: MatchCleaner,
        dixon_coles: DixonColesEstimator,
        models: list[GoalModel],
        outcome_calculator: OutcomeCalculator,
        test_year: int = 2024,
        max_goals: int = 10,
    ) -> None:
        self._repository = repository
        self._cleaner = cleaner
        self._dixon_coles = dixon_coles
        self._models = models
        self._outcome = outcome_calculator
        self._test_year = test_year
        self._max_goals = max_goals

        # Progressively populated state.
        self.train: pd.DataFrame | None = None
        self.test: pd.DataFrame | None = None
        self.strengths: dict[str, TeamStrength] = {}
        self.teams: list[str] = []
        self.matrix_builder: PoissonMatrixBuilder | None = None
        self.evaluations: list[EvaluationResult] = []
        self._actual: list[Outcome] = []

    # --------------------------------------------------------------- granular steps
    def load_and_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        matches = self._cleaner.clean(self._repository.load())
        self.train = matches[matches["year"] < self._test_year].copy()
        self.test = matches[matches["year"] == self._test_year].copy()
        return self.train, self.test

    def fit_features(self) -> dict[str, TeamStrength]:
        self._dixon_coles.fit(self.train)
        self.strengths = self._dixon_coles.strengths
        self.teams = sorted(self.strengths)
        self.matrix_builder = PoissonMatrixBuilder(
            self._max_goals, rho=self._dixon_coles.rho
        )
        self._actual = [
            Outcome.from_scores(h, a)
            for h, a in zip(self.test["home_score"], self.test["away_score"])
        ]
        return self.strengths

    def train_model(self, model: GoalModel) -> EvaluationResult:
        model.fit(self.train, self.strengths)
        preds = self._predict_fixtures(model, self.test)
        result = accuracy_1x2(model.name, preds, self._actual)
        self.evaluations.append(result)
        return result

    def artifacts(self) -> TrainedArtifacts:
        best_result, best_model = max(
            zip(self.evaluations, self._models), key=lambda pair: pair[0].accuracy
        )
        return TrainedArtifacts(
            best_model=best_model,
            best_accuracy=best_result.accuracy,
            models=list(self._models),
            evaluations=list(self.evaluations),
            strengths=self.strengths,
            matrix_builder=self.matrix_builder,
            teams=self.teams,
            train=self.train,
            test=self.test,
        )

    @property
    def models(self) -> list[GoalModel]:
        return list(self._models)

    @property
    def test_year(self) -> int:
        return self._test_year

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self._models]

    # --------------------------------------------------------------- convenience
    def run(self, on_step: ProgressHook | None = None) -> TrainedArtifacts:
        announce = on_step or (lambda _label: None)
        announce("Loading and cleaning data")
        self.load_and_split()
        announce("Estimating Dixon-Coles strengths")
        self.fit_features()
        for model in self._models:
            announce(f"Training model {model.name}")
            self.train_model(model)
        return self.artifacts()

    # ------------------------------------------------------------------- private
    def _predict_fixtures(
        self, model: GoalModel, fixtures: pd.DataFrame
    ) -> list[MatchPrediction]:
        lambda_home, lambda_away = model.predict_expected_goals(fixtures)
        predictions = []
        for (_, row), lh, la in zip(fixtures.iterrows(), lambda_home, lambda_away):
            sm = self.matrix_builder.build(row["home_team"], row["away_team"], lh, la)
            predictions.append(self._outcome.to_prediction(sm))
        return predictions
