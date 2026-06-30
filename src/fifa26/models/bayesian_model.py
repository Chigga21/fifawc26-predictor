"""Modelo bayesiano de Poisson jerarquico via MCMC con PyMC, una estrategia GoalModel.
@author Chigga21
"""
from __future__ import annotations

import contextlib
import io
import logging
import time
import warnings
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from fifa26.domain.entities import TeamStrength
from fifa26.domain.interfaces import GoalModel


class BayesianPoissonModel(GoalModel):
    """Reestima ataque y defensas con incertidumbre completa
    """

    name = "Bayesian-MCMC"
    verbose_training = True

    def __init__(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        cores: int = 1,
    ) -> None:
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._target_accept = target_accept
        self._cores = cores
        self.mu: float = 0.0
        self.gamma: float = 0.0
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}
        self.rhat_max: float = float("nan")
        self.ess_bulk_min: float = float("nan")
        self.divergences: int = 0
        self.on_progress: Callable[[str], None] | None = None

    def _emit(self, message: str) -> None:
        if self.on_progress is not None:
            self.on_progress(message)

    def fit(self, matches: pd.DataFrame, strengths: dict[str, TeamStrength]) -> "BayesianPoissonModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = matches["home_team"].map(index).to_numpy()
        ai = matches["away_team"].map(index).to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        home_adv = (~matches["neutral"].to_numpy()).astype(float)

        # Estimaciones Dixon-Coles 
        att_prior = np.array([strengths.get(t, TeamStrength(t, 0, 0)).attack for t in teams])
        def_prior = np.array([strengths.get(t, TeamStrength(t, 0, 0)).defense for t in teams])

        with pm.Model():
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            gamma = pm.Normal("gamma", mu=0.3, sigma=0.5)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)

            # Desviaciones no centradas y de suma cero sobre los priors de Dixon-Coles
            attack_raw = pm.ZeroSumNormal("attack_raw", sigma=1.0, shape=n)
            defense_raw = pm.ZeroSumNormal("defense_raw", sigma=1.0, shape=n)
            attack = pm.Deterministic("attack", att_prior + attack_raw * sigma_att)
            defense = pm.Deterministic("defense", def_prior + defense_raw * sigma_def)

            log_lh = mu + gamma * home_adv + attack[hi] - defense[ai]
            log_la = mu + attack[ai] - defense[hi]
            pm.Poisson("home_goals", mu=pm.math.exp(log_lh), observed=hs)
            pm.Poisson("away_goals", mu=pm.math.exp(log_la), observed=as_)

            self._emit("Initializing NUTS...")
            self._emit(f"Sampling ({self._chains} chains in {self._cores} job(s))")
            start = time.perf_counter()
            # Silencia la salida cruda de PyMC y PyTensor para no romper la pantalla.
            with _quiet():
                idata = pm.sample(
                    draws=self._draws,
                    tune=self._tune,
                    chains=self._chains,
                    cores=self._cores,
                    target_accept=self._target_accept,
                    progressbar=False,
                    random_seed=42,
                )
            took = time.perf_counter() - start
            self._emit(
                f"Sampling {self._chains} chains for {self._tune:,} tune and "
                f"{self._draws:,} draw iterations, took {took:.0f} seconds"
            )

        post = idata.posterior
        self.mu = float(post["mu"].mean())
        self.gamma = float(post["gamma"].mean())
        att_mean = post["attack"].mean(dim=("chain", "draw")).to_numpy()
        def_mean = post["defense"].mean(dim=("chain", "draw")).to_numpy()
        self._attack = {t: float(att_mean[i]) for t, i in index.items()}
        self._defense = {t: float(def_mean[i]) for t, i in index.items()}

        self._report_diagnostics(idata)
        return self

    def _report_diagnostics(self, idata) -> None:
        """Calcula y reporta r-hat, ESS y divergencias del muestreo NUTS..
        """
        with _quiet():
            rhat = az.rhat(idata)
            ess = az.ess(idata)
        self.rhat_max = max(float(rhat[v].max()) for v in rhat.data_vars)
        self.ess_bulk_min = min(float(ess[v].min()) for v in ess.data_vars)
        if "diverging" in idata.sample_stats:
            self.divergences = int(idata.sample_stats["diverging"].sum())
        self._emit(
            f"Diagnostics: r-hat max={self.rhat_max:.3f}, "
            f"ESS min={self.ess_bulk_min:.0f}, divergences={self.divergences}"
        )
        if self.rhat_max > 1.01 or self.divergences > 0:
            warnings.warn(
                f"Convergencia bayesiana dudosa: r-hat max={self.rhat_max:.3f}, "
                f"divergencias={self.divergences}",
                RuntimeWarning,
                stacklevel=2,
            )

    def predict_expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        att_h = fixtures["home_team"].map(lambda t: self._attack.get(t, 0.0)).to_numpy()
        def_h = fixtures["home_team"].map(lambda t: self._defense.get(t, 0.0)).to_numpy()
        att_a = fixtures["away_team"].map(lambda t: self._attack.get(t, 0.0)).to_numpy()
        def_a = fixtures["away_team"].map(lambda t: self._defense.get(t, 0.0)).to_numpy()
        home_adv = (~fixtures["neutral"].to_numpy()).astype(float)

        lambda_home = np.exp(self.mu + self.gamma * home_adv + att_h - def_a)
        lambda_away = np.exp(self.mu + att_a - def_h)
        return lambda_home, lambda_away


@contextlib.contextmanager
def _quiet():
    """Redirige stdout y stderr y silencia los loggers de PyMC y PyTensor."""
    loggers = [logging.getLogger("pymc"), logging.getLogger("pytensor")]
    previous = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        for lg, level in zip(loggers, previous):
            lg.setLevel(level)
