"""Composition Root for the WC26 / FIFA 26 match predictor.

This is the only place that knows about every concrete class. It wires the
dependencies together (Dependency Injection) and runs one of two front-ends:

    python main.py            interactive terminal UI (logo, team selection, 1X2)
    python main.py --eval     batch pipeline (train, evaluate 2024, save figures)

Changing a data source, adding a model or swapping the front-end is a change *here*,
not inside the business logic.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make `src/` importable without installation.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")

from fifa26.application.pipeline import PredictionPipeline  # noqa: E402
from fifa26.application.training import Trainer  # noqa: E402
from fifa26.cli.app import InteractiveApp  # noqa: E402
from fifa26.data.cleaning import MatchCleaner  # noqa: E402
from fifa26.data.csv_repository import CsvMatchRepository  # noqa: E402
from fifa26.features.dixon_coles import DixonColesEstimator  # noqa: E402
from fifa26.models.factory import ModelFactory  # noqa: E402
from fifa26.persistence.pickle_store import PickleArtifactStore  # noqa: E402
from fifa26.prediction.outcome import OutcomeCalculator  # noqa: E402
from fifa26.visualization.plots import Visualizer  # noqa: E402

DATA_DIR = ROOT / "international_results"
OUTPUT_DIR = ROOT / "outputs"
MODEL_FILE = ROOT / "artifacts" / "trained.pkl"


def build_trainer() -> tuple[Trainer, OutcomeCalculator]:
    repository = CsvMatchRepository(DATA_DIR / "results.csv")
    cleaner = MatchCleaner(min_year=2018, min_matches_per_team=8)
    dixon_coles = DixonColesEstimator(half_life_days=540)
    models = [
        ModelFactory.create("xgboost"),
        ModelFactory.create("bayesian", draws=1000, tune=1000, chains=4),
    ]
    outcome = OutcomeCalculator()
    trainer = Trainer(
        repository=repository,
        cleaner=cleaner,
        dixon_coles=dixon_coles,
        models=models,
        outcome_calculator=outcome,
        test_year=2024,
        max_goals=10,
    )
    return trainer, outcome


def run_interactive() -> int:
    trainer, outcome = build_trainer()
    store = PickleArtifactStore(MODEL_FILE)
    visualizer = Visualizer(OUTPUT_DIR)
    return InteractiveApp(trainer, outcome, store, visualizer).run()


def run_eval() -> None:
    trainer, outcome = build_trainer()
    pipeline = PredictionPipeline(
        trainer=trainer,
        outcome_calculator=outcome,
        visualizer=Visualizer(OUTPUT_DIR),
        generate_graphs=True,
    )
    print("=" * 64)
    print(" WC26 PREDICTOR - batch pipeline (Dixon-Coles + Bayes/XGBoost)")
    print("=" * 64)
    report = pipeline.run()

    print("\n" + "=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    for ev in report.evaluations:
        print(" ", ev)
    print(f"  Best model: {report.best_model}")
    if report.example_prediction is not None:
        p = report.example_prediction
        print(
            f"  Example {p.home_team} vs {p.away_team}: "
            f"1={p.prob_home_win:.1%}  X={p.prob_draw:.1%}  2={p.prob_away_win:.1%} "
            f"[pred: {p.predicted_outcome.name}]"
        )
    if report.figures:
        print("  Generated figures:")
        for fig in report.figures:
            print(f"    - {fig}")


def main() -> int:
    if "--eval" in sys.argv[1:]:
        run_eval()
        return 0
    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
