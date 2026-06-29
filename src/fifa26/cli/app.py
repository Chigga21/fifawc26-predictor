"""InteractiveApp: the linear terminal experience, printed as a continuous stream.

The whole session reads top to bottom like a log: nothing clears the screen and every
view is simply printed after the previous one. Whenever the program waits for the user
it shows the fixed `> ` prompt (see `cli.prompt`), and every choice is confirmed with an
explicit Enter — no action is triggered by a single keystroke.

Flow (each step is a method below):
    1. intro     - logo (WC26 PREDICTOR) + author + the list of models to train,
                   all printed at once; only then a single confirmation to start.
    2. train     - run each stage behind a loading-dots animation (no fake bars).
    3-5. predict - pick Team A, Team B and venue, confirm, show the 1X2 forecast.
    6. next      - predict another fixture (reusing the trained model) or quit.

The app depends only on the `application` services (`Trainer`, `PredictionService`) and
the sibling `cli` modules; it owns presentation, never business logic.
"""
from __future__ import annotations

from fifa26.application.artifact_store import ArtifactStore
from fifa26.application.prediction_service import (
    MatchForecast,
    PredictionService,
)
from fifa26.application.training import Trainer, TrainedArtifacts
from fifa26.cli import ansi
from fifa26.cli.banner import render_header
from fifa26.cli.prompt import read_line
from fifa26.cli.selector import TeamSelector
from fifa26.cli.spinner import run_with_spinner
from fifa26.domain.entities import Outcome
from fifa26.prediction.outcome import OutcomeCalculator
from fifa26.visualization.plots import Visualizer, open_figures, render_match_figures


class InteractiveApp:
    def __init__(
        self,
        trainer: Trainer,
        outcome_calculator: OutcomeCalculator,
        store: ArtifactStore,
        visualizer: Visualizer,
    ) -> None:
        self._trainer = trainer
        self._outcome = outcome_calculator
        self._store = store
        self._visualizer = visualizer
        self._service: PredictionService | None = None
        self._generate_graphs = False
        self._load_saved = False

    # ----------------------------------------------------------------------- run
    def run(self) -> int:
        try:
            if not self._main_menu():
                return 0
            self._prepare_service()
            if self._service is not None:
                self._predict_loop()
        except KeyboardInterrupt:
            print()
            print("  " + ansi.hint("Interrupted"))
        finally:
            ansi.show_cursor()
        return 0

    # ------------------------------------------------------------------- screens
    def _main_menu(self) -> bool:
        """Welcome banner plus the startup configuration, printed as one stream.

        Collects two settings before any work starts: whether to generate graphs, and
        whether to load a previously saved model or retrain from scratch. A single Enter
        confirmation then starts the session.
        """
        render_header()
        print(ansi.hint("Choose two national teams and estimate the 1X2 score for the matchup."))
        print()
        print(ansi.heading("[ 1 ] Main menu"))
        print()
        print(
            "  Generate graphs:   "
            + ansi.active("[ Y ] yes")
            + "    [ N ] no   "
            + ansi.hint("( Enter = no )")
        )
        print("  " + ansi.hint("when on, figures are saved to outputs/ and opened after each prediction"))
        self._generate_graphs = self._ask({"y": "yes", "n": "no", "enter": "no"}) == "yes"
        print("  " + ansi.hint(f"graphs: {'ON' if self._generate_graphs else 'OFF'}"))
        print()

        if self._store.exists():
            print("  A saved model was found.")
            print(
                "  "
                + ansi.hint("[ L ] load saved model    [ R ] retrain from scratch   ( Enter = load )")
            )
            self._load_saved = self._ask({"l": "load", "r": "retrain", "enter": "load"}) == "load"
        else:
            self._load_saved = False
            print("  No saved model yet. The following ML models will be trained:")
            for name in self._trainer.model_names:
                print("    " + ansi.active("[x] ") + name)
        print()
        print("  " + ansi.hint("[ Enter ] continue    [ Q ] quit"))
        return self._ask({"enter": "go", "q": "quit"}) == "go"

    def _prepare_service(self) -> None:
        """Build the prediction service either by loading from disk or by training."""
        artifacts: TrainedArtifacts | None = None
        if self._load_saved:
            artifacts = self._store.load()
            if artifacts is None:
                print("  " + ansi.hint("saved model could not be loaded; retraining"))
            else:
                print()
                print(
                    "  "
                    + ansi.confirm(
                        f"[done] Loaded saved model: {artifacts.best_model.name} "
                        f"(accuracy {artifacts.best_accuracy:.3f})"
                    )
                )
        if artifacts is None:
            artifacts = self._train()
            path = self._store.save(artifacts)
            print("  " + ansi.hint(f"model saved to {path}"))

        self._service = PredictionService.from_artifacts(artifacts, self._outcome)
        print()
        print("  " + ansi.hint("[ Enter ] go to team selection"))
        self._ask({"enter": "go"})

    def _train(self) -> TrainedArtifacts:
        print()
        train, test = run_with_spinner(
            "Loading and cleaning data", self._trainer.load_and_split
        )
        print(
            "  "
            + ansi.hint(f"Training: {len(train)} matches | test: {len(test)} matches")
        )
        run_with_spinner("Estimating offensive and defensive strengths with Dixon–Coles", self._trainer.fit_features)

        for model in self._trainer.models:
            result = run_with_spinner(
                f"Training model {model.name}",
                lambda m=model: self._trainer.train_model(m),
            )
            print("    " + ansi.hint(str(result)))

        artifacts = self._trainer.artifacts()
        print()
        print(
            "  "
            + ansi.confirm(
                f"[done] Best Model: {artifacts.best_model.name} "
                f"(accuracy {artifacts.best_accuracy:.3f})"
            )
        )
        return artifacts

    def _predict_loop(self) -> None:
        assert self._service is not None
        selector = TeamSelector(self._service.teams)
        while True:
            matchup = self._choose_matchup(selector)
            if matchup is None:
                if self._ask_quit():
                    return
                continue

            home, away, neutral = matchup
            if not self._confirm(home, away, neutral):
                continue

            forecast = self._service.predict(home, away, neutral=neutral)
            self._show_result(forecast)
            self._maybe_visualise(forecast)

            print()
            print("  " + ansi.hint("[ N ] new match    [ Q ] quit"))
            if self._ask({"n": "again", "enter": "again", "q": "quit"}) == "quit":
                return

    # ------------------------------------------------------------ selection step
    def _choose_matchup(self, selector: TeamSelector):
        home = selector.select("[ 2 ] Select Team A (local)")
        if home is None:
            return None
        away = selector.select(
            f"[ 3 ] Select Team B (away)   -   opponent of {home}", exclude=home
        )
        if away is None:
            return None
        neutral = self._choose_venue(home)
        return home, away, neutral

    def _choose_venue(self, home: str) -> bool:
        print()
        print(ansi.heading("[ 4 ] Match venue"))
        print()
        print("  " + ansi.active("[N] neutral venue  ") + ansi.hint("(World Cup context)"))
        print(f"  [H] {home} plays at home")
        print()
        print("  " + ansi.hint("[ N ] neutral    [ H ] Team A local    ( Enter = neutral )"))
        return self._ask({"n": "neutral", "enter": "neutral", "h": "home"}) == "neutral"

    def _confirm(self, home: str, away: str, neutral: bool) -> bool:
        print()
        print(ansi.heading("[ 5 ] Confirm match "))
        print()
        venue = "neutral venue" if neutral else f"{home} plays at home"
        print("    " + ansi.active(home) + ansi.bold("   vs   ") + ansi.active(away))
        print("    " + ansi.hint(venue))
        print()
        print("  " + ansi.hint("[ Enter ] confirm   [ E ] edit    [ Q ] quit"))
        action = self._ask({"enter": "ok", "e": "edit", "q": "quit"})
        if action == "quit":
            raise KeyboardInterrupt
        return action == "ok"

    # --------------------------------------------------------------- result step
    def _show_result(self, forecast: MatchForecast) -> None:
        sm = forecast.score_matrix
        pred = forecast.prediction
        print()
        print(ansi.heading(f"[ 6 ] Prediction  {pred.home_team}  vs  {pred.away_team}"))
        print()
        print(
            "  Expected goals:  "
            + ansi.bold(f"{pred.home_team} {sm.lambda_home:.2f}")
            + "   -   "
            + ansi.bold(f"{sm.lambda_away:.2f} {pred.away_team}")
        )
        print()
        print("  1X2 probabilities:")
        self._print_prob(f"[1] {pred.home_team} wins", pred.prob_home_win)
        self._print_prob("[X] Draw", pred.prob_draw)
        self._print_prob(f"[2] {pred.away_team} wins", pred.prob_away_win)

        print()
        label = self._outcome_label(pred.predicted_outcome, pred.home_team, pred.away_team)
        print("  Most likely result:  " + ansi.confirm(label))

        print()
        print("  " + ansi.hint("Most likely scorelines:"))
        for scoreline, prob in forecast.top_scorelines[:5]:
            print(f"    [ {scoreline} ]  {prob:5.1%}")

    def _print_prob(self, label: str, prob: float) -> None:
        bar = _bar(prob)
        print(f"    {label:<22} {ansi.dim(bar)}  {prob:5.1%}")

    # ------------------------------------------------------------- visualisation
    def _maybe_visualise(self, forecast: MatchForecast) -> None:
        """The single gate for interactive graphs: render+open only when enabled."""
        if not self._generate_graphs:
            return
        print()
        print("  " + ansi.hint("Generating graphs..."))
        paths = render_match_figures(self._visualizer, forecast)
        open_figures(paths)
        for path in paths:
            print("    " + ansi.hint(f"- {path}"))

    @staticmethod
    def _outcome_label(outcome: Outcome, home: str, away: str) -> str:
        if outcome is Outcome.HOME_WIN:
            return f"{home} wins"
        if outcome is Outcome.AWAY_WIN:
            return f"{away} wins"
        return "Draw"

    # ---------------------------------------------------------------- line prompts
    def _ask_quit(self) -> bool:
        print()
        print("  " + ansi.hint("[ Q ] quit    [ Enter ] main menu"))
        return self._ask({"q": "quit", "enter": "back"}) == "quit"

    def _ask(self, actions: dict[str, str]) -> str:
        """Read an Enter-confirmed line and resolve it to an action id from `actions`.

        Keys in `actions` are either 'enter' (empty line) or a single lowercase
        character (the first character of the typed line decides). Unrecognised input
        re-prompts instead of guessing, keeping the `> ` cursor visible until the user
        makes a valid, explicit choice.
        """
        while True:
            raw = read_line().lower()
            if raw == "":
                if "enter" in actions:
                    return actions["enter"]
                print("  " + ansi.hint("press Enter to continue"))
                continue
            char = raw[0]
            if char in actions:
                return actions[char]
            print("  " + ansi.hint("Not an option. Try again."))


def _bar(prob: float, width: int = 20) -> str:
    fill = int(round(prob * width))
    fill = max(0, min(width, fill))
    return "[" + "#" * fill + "-" * (width - fill) + "]"
