"""Hierarchical Bayesian Poisson model via PyMC MCMC (a `GoalModel` strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from fifa26.domain.entities import TeamStrength
from fifa26.domain.interfaces import GoalModel


class BayesianPoissonModel(GoalModel):
    """Bayesian re-estimation of attack/defence with full uncertainty.

    A hierarchical Poisson model: per-team `attack` and `defense` are drawn from
    common Normal hyper-priors (partial pooling), so teams with few matches are
    shrunk toward the average instead of overfitting. The Dixon-Coles point
    estimates centre the priors, injecting that prior knowledge into the sampler.

        goals_home ~ Poisson(exp(mu + gamma*home_adv + att[home] - def[away]))
        goals_away ~ Poisson(exp(mu             + att[away] - def[home]))

    Sampling uses NUTS; we keep the posterior-mean attack/defence per team to
    produce the expected goals (lambdas) at prediction time.
    """

    name = "Bayesian-MCMC"

    def __init__(self, draws: int = 1000, tune: int = 1000, chains: int = 4, target_accept: float = 0.95) -> None:
        self._draws = draws
        self._tune = tune
        self._chains = chains
        self._target_accept = target_accept
        self.mu: float = 0.0
        self.gamma: float = 0.0
        self._attack: dict[str, float] = {}
        self._defense: dict[str, float] = {}

    def fit(self, matches: pd.DataFrame, strengths: dict[str, TeamStrength]) -> "BayesianPoissonModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = matches["home_team"].map(index).to_numpy()
        ai = matches["away_team"].map(index).to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        home_adv = (~matches["neutral"].to_numpy()).astype(float)

        # Dixon-Coles estimates as informative prior means.
        att_prior = np.array([strengths.get(t, TeamStrength(t, 0, 0)).attack for t in teams])
        def_prior = np.array([strengths.get(t, TeamStrength(t, 0, 0)).defense for t in teams])

        with pm.Model():
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            gamma = pm.Normal("gamma", mu=0.3, sigma=0.5)
            sigma_att = pm.HalfNormal("sigma_att", sigma=1.0)
            sigma_def = pm.HalfNormal("sigma_def", sigma=1.0)

            # Non-centred, sum-to-zero deviations around the Dixon-Coles priors.
            # ZeroSumNormal fixes the attack/defence level (identifiability)
            # cleanly, and the `* sigma` reparameterisation avoids the funnel
            # that a centred hierarchical prior would create.
            attack_raw = pm.ZeroSumNormal("attack_raw", sigma=1.0, shape=n)
            defense_raw = pm.ZeroSumNormal("defense_raw", sigma=1.0, shape=n)
            attack = pm.Deterministic("attack", att_prior + attack_raw * sigma_att)
            defense = pm.Deterministic("defense", def_prior + defense_raw * sigma_def)

            log_lh = mu + gamma * home_adv + attack[hi] - defense[ai]
            log_la = mu + attack[ai] - defense[hi]
            pm.Poisson("home_goals", mu=pm.math.exp(log_lh), observed=hs)
            pm.Poisson("away_goals", mu=pm.math.exp(log_la), observed=as_)

            idata = pm.sample(
                draws=self._draws,
                tune=self._tune,
                chains=self._chains,
                cores=1,
                target_accept=self._target_accept,
                progressbar=False,
                random_seed=42,
            )

        post = idata.posterior
        self.mu = float(post["mu"].mean())
        self.gamma = float(post["gamma"].mean())
        att_mean = post["attack"].mean(dim=("chain", "draw")).to_numpy()
        def_mean = post["defense"].mean(dim=("chain", "draw")).to_numpy()
        self._attack = {t: float(att_mean[i]) for t, i in index.items()}
        self._defense = {t: float(def_mean[i]) for t, i in index.items()}
        return self

    def predict_expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        att_h = fixtures["home_team"].map(lambda t: self._attack.get(t, 0.0)).to_numpy()
        def_h = fixtures["home_team"].map(lambda t: self._defense.get(t, 0.0)).to_numpy()
        att_a = fixtures["away_team"].map(lambda t: self._attack.get(t, 0.0)).to_numpy()
        def_a = fixtures["away_team"].map(lambda t: self._defense.get(t, 0.0)).to_numpy()
        home_adv = (~fixtures["neutral"].to_numpy()).astype(float)

        lambda_home = np.exp(self.mu + self.gamma * home_adv + att_h - def_a)
        lambda_away = np.exp(self.mu + att_a - def_h)
        return lambda_home, lambda_away
