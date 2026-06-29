"""Application orchestrator: the end-to-end batch prediction pipeline.

Used by `python main.py --eval`: it trains and evaluates every model, picks the best
one and renders the figures for an example fixture. The interactive UI shares the same
`Trainer` and `PredictionService` building blocks but drives them with its own feedback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fifa26.application.prediction_service import PredictionService
from fifa26.application.training import Trainer
from fifa26.domain.entities import MatchPrediction, TeamStrength
from fifa26.evaluation.metrics import EvaluationResult
from fifa26.prediction.outcome import OutcomeCalculator
from fifa26.visualization.plots import Visualizer, render_match_figures


@dataclass
class PipelineReport:
    evaluations: list[EvaluationResult] = field(default_factory=list)
    best_model: str = ""
    figures: list[Path] = field(default_factory=list)
    example_prediction: MatchPrediction | None = None


class PredictionPipeline:
    """Facade that runs the fixed sequence of stages (Template Method).

    Stages: train (delegated to `Trainer`) -> report the best model -> optionally predict a
    showcase fixture (the two strongest teams) and visualise. Dependencies are injected, so
    each piece is swappable and testable in isolation. Visualisation is conditional on
    `generate_graphs`.
    """

    def __init__(
        self,
        trainer: Trainer,
        outcome_calculator: OutcomeCalculator,
        visualizer: Visualizer,
        generate_graphs: bool = True,
    ) -> None:
        self._trainer = trainer
        self._outcome = outcome_calculator
        self._visualizer = visualizer
        self._generate_graphs = generate_graphs

    def run(self) -> PipelineReport:
        report = PipelineReport()

        print("* Loading and cleaning data...")
        train, test = self._trainer.load_and_split()
        print(f"  training: {len(train)} matches | test: {len(test)}")

        print("* Estimating Dixon-Coles strengths (intermediate features)...")
        self._trainer.fit_features()
        self._report_strengths(self._trainer.strengths)

        for model in self._trainer.models:
            print(f"* Training model {model.name}...")
            result = self._trainer.train_model(model)
            report.evaluations.append(result)
            print(f"  {result}")

        artifacts = self._trainer.artifacts()
        report.best_model = artifacts.best_model.name
        print(f"* Best model: {report.best_model} (accuracy {artifacts.best_accuracy:.3f})")

        if self._generate_graphs:
            print("* Generating visualisations...")
            service = PredictionService.from_artifacts(artifacts, self._outcome)
            report.figures, report.example_prediction = self._visualise(service)
            report.figures.append(self._visualizer.model_comparison(report.evaluations))
        else:
            print("* Skipping visualisations (graphs disabled).")
        return report

    # ----------------------------------------------------------------- private
    def _visualise(
        self, service: PredictionService
    ) -> tuple[list[Path], MatchPrediction]:
        home, away = self._showcase_matchup()
        forecast = service.predict(home, away, neutral=True)
        figures = render_match_figures(self._visualizer, forecast)
        return figures, forecast.prediction

    def _showcase_matchup(self) -> tuple[str, str]:
        """The two strongest teams by attack, used to illustrate the figures."""
        ranked = sorted(
            self._trainer.strengths.values(), key=lambda s: s.attack, reverse=True
        )
        return ranked[0].team, ranked[1].team

    @staticmethod
    def _report_strengths(strengths: dict[str, TeamStrength]) -> None:
        ranked = sorted(strengths.values(), key=lambda s: s.attack, reverse=True)
        top = ", ".join(f"{s.team} ({s.attack:+.2f})" for s in ranked[:5])
        print(f"  top attack: {top}")
