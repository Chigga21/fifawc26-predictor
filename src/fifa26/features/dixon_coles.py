"""Dixon-Coles offensive/defensive strength estimator.

This is the feature-engineering stage. It fits the classic Dixon & Coles (1997)
bivariate-Poisson-with-correction model by weighted maximum likelihood and
exposes, for every team, an `attack` and `defense` rating. Those ratings are the
intermediate variables that feed the two downstream goal models.

Model
-----
    log(lambda_home) = mu + gamma + attack_home - defense_away   (home, non-neutral)
    log(lambda_away) = mu        + attack_away - defense_home

`gamma` is the home advantage (dropped on neutral ground), `mu` the baseline log
scoring rate. The low-score dependence is captured by Dixon-Coles' `tau`
correction with parameter `rho`. Recent matches weigh more through an
exponential time-decay with a configurable half-life.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from fifa26.domain.entities import TeamStrength


class DixonColesEstimator:
    def __init__(self, half_life_days: int = 540, max_iter: int = 200) -> None:
        self._half_life_days = half_life_days
        self._max_iter = max_iter
        # Learned parameters (available after `fit`).
        self.mu: float = 0.0
        self.gamma: float = 0.0
        self.rho: float = 0.0
        self.strengths: dict[str, TeamStrength] = {}
        self._teams: list[str] = []

    # ------------------------------------------------------------------ public
    def fit(self, matches: pd.DataFrame) -> "DixonColesEstimator":
        self._teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {t: i for i, t in enumerate(self._teams)}
        n = len(self._teams)

        hi = matches["home_team"].map(index).to_numpy()
        ai = matches["away_team"].map(index).to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        home_adv = (~matches["neutral"].to_numpy()).astype(float)
        weights = self._time_weights(matches["date"])

        # Pre-computed constant of the Poisson log-pmf (does not affect argmax,
        # kept so the reported log-likelihood is exact).
        log_fact = gammaln(hs + 1) + gammaln(as_ + 1)

        x0 = np.concatenate([[0.0, 0.3, 0.0], np.zeros(n), np.zeros(n)])
        bounds = [(-2, 2), (-1, 1), (-0.15, 0.15)] + [(-3, 3)] * (2 * n)

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(hi, ai, hs, as_, home_adv, weights, log_fact, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self._max_iter},
        )

        self.mu, self.gamma, self.rho, attack, defense = self._unpack(result.x, n)
        self.strengths = {
            t: TeamStrength(team=t, attack=float(attack[i]), defense=float(defense[i]))
            for t, i in index.items()
        }
        return self

    def expected_goals(
        self, home_team: str, away_team: str, neutral: bool
    ) -> tuple[float, float]:
        """Baseline Dixon-Coles expected goals for a fixture."""
        h, a = self.strengths[home_team], self.strengths[away_team]
        gamma = 0.0 if neutral else self.gamma
        lambda_home = np.exp(self.mu + gamma + h.attack - a.defense)
        lambda_away = np.exp(self.mu + a.attack - h.defense)
        return float(lambda_home), float(lambda_away)

    # ----------------------------------------------------------------- private
    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        latest = dates.max()
        age_days = (latest - dates).dt.days.to_numpy()
        xi = np.log(2) / self._half_life_days
        return np.exp(-xi * age_days)

    @staticmethod
    def _unpack(params: np.ndarray, n: int):
        mu, gamma, rho = params[0], params[1], params[2]
        attack = params[3 : 3 + n]
        defense = params[3 + n : 3 + 2 * n]
        # Identifiability: centre attack and defence (their global level is
        # absorbed by mu / gamma).
        attack = attack - attack.mean()
        defense = defense - defense.mean()
        return mu, gamma, rho, attack, defense

    @classmethod
    def _neg_log_likelihood(
        cls, params, hi, ai, hs, as_, home_adv, weights, log_fact, n
    ) -> float:
        mu, gamma, rho, attack, defense = cls._unpack(params, n)
        log_lh = mu + gamma * home_adv + attack[hi] - defense[ai]
        log_la = mu + attack[ai] - defense[hi]
        lam_h, lam_a = np.exp(log_lh), np.exp(log_la)

        # Poisson log-pmf for both scorelines.
        ll = hs * log_lh - lam_h + as_ * log_la - lam_a - log_fact

        # Dixon-Coles low-score correction tau.
        tau = cls._tau(hs, as_, lam_h, lam_a, rho)
        ll = ll + np.log(np.clip(tau, 1e-10, None))

        return -float(np.sum(weights * ll))

    @staticmethod
    def _tau(hs, as_, lam_h, lam_a, rho) -> np.ndarray:
        tau = np.ones_like(lam_h, dtype=float)
        m00 = (hs == 0) & (as_ == 0)
        m01 = (hs == 0) & (as_ == 1)
        m10 = (hs == 1) & (as_ == 0)
        m11 = (hs == 1) & (as_ == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho
        return tau
