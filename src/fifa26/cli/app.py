"""Clase que implementa la experiencia de UI de terminal
impresa de manera continua.

@author Chigga21
"""
from __future__ import annotations

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
        visualizer: Visualizer,
    ) -> None:
        self._trainer = trainer
        self._outcome = outcome_calculator
        self._visualizer = visualizer
        self._service: PredictionService | None = None
        self._generate_graphs = False

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
        except Exception as exc:  # noqa BLE001 
            print()
            print("  " + ansi.error(f"Error: {exc}"))
            return 1
        finally:
            ansi.show_cursor()
        return 0

    def _main_menu(self) -> bool:
        """Banner de bienvenida y configuracion de arranque
        Pregunta al usuario si se generarán gráficos o no de los resultados.
        """
        render_header()
        print(
            "Generate graphs:   "
            + " [ Y ] yes"
            + "    [ N ] no   "
            + ansi.hint("( Enter = no )")
        )
        print(ansi.hint("when on, figures are saved to outputs/ and opened after each prediction"))
        self._generate_graphs = self._ask({"y": "yes", "n": "no", "enter": "no"}) == "yes"
        print("  " + ansi.hint(f"graphs: {'ON' if self._generate_graphs else 'OFF'}"))
        print()

        print("  The following ML models will be trained:")
        for name in self._trainer.model_names:
            print("    " + ansi.active("[x] ") + name)
        print()
        print("  " + ansi.hint("[ Enter ] continue    [ Q ] quit"))
        return self._ask({"enter": "go", "q": "quit"}) == "go"

    def _prepare_service(self) -> None:
        """Entrena los modelos y construye el servicio de prediccion con el resultado"""
        artifacts = self._train()
        self._service = PredictionService.from_artifacts(artifacts, self._outcome)

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
            if getattr(model, "verbose_training", False):
                # Imprime sus propios mensajes; sin spinner para no corromper la pantalla.
                print("  " + ansi.focused("[*]") + f" Training model {model.name}")
                model.on_progress = lambda msg: print("    " + ansi.hint(msg))
                result = self._trainer.train_model(model)
            else:
                result = run_with_spinner(
                    f"Training model {model.name}",
                    lambda m=model: self._trainer.train_model(m),
                )
            print("    " + ansi.hint(str(result)))

        # Reentrena el modelo ganador con todos los datos
        best = self._trainer.best_model()
        label = f"Refitting best model ({best.name}) on the full dataset"
        if getattr(best, "verbose_training", False):
            print("  " + ansi.focused("[*]") + " " + label)
            best.on_progress = lambda msg: print("    " + ansi.hint(msg))
            self._trainer.fit_production(best)
        else:
            run_with_spinner(label, lambda: self._trainer.fit_production(best))

        artifacts = self._trainer.artifacts()
        print()
        print(
            "  "
            + ansi.confirm(
                f"[done] Best Model: {artifacts.best_model.name} "
                f"(RPS {artifacts.best_rps:.4f} | accuracy {artifacts.best_accuracy:.3f})"
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
            decision = self._confirm(home, away, neutral)
            if decision == "quit":
                return
            if decision != "ok": 
                continue

            forecast = self._service.predict(home, away, neutral=neutral)
            self._show_result(forecast)
            self._maybe_visualise(forecast)

            print()
            print("  " + ansi.hint("[ N ] new match    [ Q ] quit"))
            if self._ask({"n": "again", "enter": "again", "q": "quit"}) == "quit":
                return

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

    def _confirm(self, home: str, away: str, neutral: bool) -> str:
        """Devuelve la accion elegida: 'ok', 'edit' o 'quit' (sin usar excepciones)."""
        print()
        print(ansi.heading("[ 5 ] Confirm match "))
        print()
        venue = "neutral venue" if neutral else f"{home} plays at home"
        print("    " + ansi.active(home) + ansi.bold("   vs   ") + ansi.active(away))
        print("    " + ansi.hint(venue))
        print()
        print("  " + ansi.hint("[ Enter ] confirm   [ E ] edit    [ Q ] quit"))
        return self._ask({"enter": "ok", "e": "edit", "q": "quit"})

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

    # ------------------------------------------------------------- visualizacion
    def _maybe_visualise(self, forecast: MatchForecast) -> None:
        """Dibuja las graficas si estan activas."""
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

    def _ask_quit(self) -> bool:
        print()
        print("  " + ansi.hint("[ Q ] quit    [ Enter ] main menu"))
        return self._ask({"q": "quit", "enter": "back"}) == "quit"

    def _ask(self, actions: dict[str, str]) -> str:
        """Lee una linea confirmada con Enter y la resuelve a una accion de "actions"
        """
        while True:
            raw = read_line().lower()
            if raw == "":
                if "enter" in actions:
                    return actions["enter"]
                continue
            char = raw[0]
            if char in actions:
                return actions[char]
            print("  " + ansi.hint("Not an option. Try again."))


def _bar(prob: float, width: int = 20) -> str:
    fill = int(round(prob * width))
    fill = max(0, min(width, fill))
    return "[" + "#" * fill + "-" * (width - fill) + "]"
