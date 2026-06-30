"""Estimador de fuerzas ofensiva y defensiva Dixon-Coles.
Se encarga de las variables intermedias para entrenar
a los modelos
@author Chigga21
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from fifa26.domain.entities import TeamStrength


def _match_importance(tournaments: pd.Series) -> np.ndarray:
    """Peso de importancia por tipo de torneo al estilo del ranking FIFA.

    Los amistosos pesan menos y la fase final del Mundial pesa mas, de modo que
    los partidos competitivos influyen mas en las fuerzas estimadas. Las reglas
    son por coincidencia de texto y los valores son ajustables.
    """
    names = tournaments.astype(str)
    weight = np.full(len(names), 1.5, dtype=float)
    lowered = names.str.lower()

    is_qualification = lowered.str.contains("qualification")
    is_nations_league = lowered.str.contains("nations league")
    is_continental_final = names.str.contains(
        "UEFA Euro|Copa América|African Cup of Nations|AFC Asian Cup|Gold Cup",
        regex=True,
    ) & ~is_qualification
    is_world_cup = names.eq("FIFA World Cup")
    is_friendly = names.eq("Friendly")

    weight[is_nations_league.to_numpy()] = 2.5
    weight[is_continental_final.to_numpy()] = 3.0
    # La fase de clasificacion se evalua despues para que una clasificacion de
    # la Nations League cuente como clasificatorio y no como fase final.
    weight[is_qualification.to_numpy()] = 2.0
    weight[is_world_cup.to_numpy()] = 4.0
    weight[is_friendly.to_numpy()] = 1.0
    return weight


class DixonColesEstimator:
    def __init__(self, half_life_days: int = 540, max_iter: int = 2000) -> None:
        self._half_life_days = half_life_days
        self._max_iter = max_iter
        self.mu: float = 0.0
        self.gamma: float = 0.0
        self.rho: float = 0.0
        self.strengths: dict[str, TeamStrength] = {}
        self._teams: list[str] = []

    def fit(self, matches: pd.DataFrame) -> "DixonColesEstimator":
        self._teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {t: i for i, t in enumerate(self._teams)}
        n = len(self._teams)

        hi = matches["home_team"].map(index).to_numpy()
        ai = matches["away_team"].map(index).to_numpy()
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        home_adv = (~matches["neutral"].to_numpy()).astype(float)
        # El peso final combina el decaimiento temporal con la importancia del torneo.
        weights = self._time_weights(matches["date"]) * _match_importance(matches["tournament"])

        log_fact = gammaln(hs + 1) + gammaln(as_ + 1)

        x0 = np.concatenate([[0.0, 0.3, 0.0], np.zeros(n), np.zeros(n)])
        bounds = [(-2, 2), (-1, 1), (-0.15, 0.15)] + [(-3, 3)] * (2 * n)

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(hi, ai, hs, as_, home_adv, weights, log_fact, n),
            method="L-BFGS-B",
            jac=self._neg_log_likelihood_grad,
            bounds=bounds,
            options={"maxiter": self._max_iter, "maxfun": self._max_iter * 20},
        )

        if not result.success:
            warnings.warn(
                f"La optimizacion Dixon-Coles no convergio: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        self.mu, self.gamma, self.rho, attack, defense = self._unpack(result.x, n)
        self.strengths = {
            t: TeamStrength(team=t, attack=float(attack[i]), defense=float(defense[i]))
            for t, i in index.items()
        }
        return self

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

        # Log-pmf de Poisson para ambos marcadores.
        ll = hs * log_lh - lam_h + as_ * log_la - lam_a - log_fact

        # Correccion tau de Dixon-Coles
        tau = cls._tau(hs, as_, lam_h, lam_a, rho)
        ll = ll + np.log(np.clip(tau, 1e-10, None))

        return -float(np.sum(weights * ll))

    @classmethod
    def _neg_log_likelihood_grad(
        cls, params, hi, ai, hs, as_, home_adv, weights, log_fact, n
    ) -> np.ndarray:
        """Gradiente analitico de la log-verosimilitud negativa.
        """
        mu, gamma, rho, attack, defense = cls._unpack(params, n)
        log_lh = mu + gamma * home_adv + attack[hi] - defense[ai]
        log_la = mu + attack[ai] - defense[hi]
        lam_h, lam_a = np.exp(log_lh), np.exp(log_la)

        # Derivadas de log(tau) respecto a lam_h, lam_a y rho en las celdas bajas
        tau = np.clip(cls._tau(hs, as_, lam_h, lam_a, rho), 1e-10, None)
        d_lh = np.zeros_like(lam_h)
        d_la = np.zeros_like(lam_a)
        d_rho = np.zeros_like(lam_h)
        m00 = (hs == 0) & (as_ == 0)
        m01 = (hs == 0) & (as_ == 1)
        m10 = (hs == 1) & (as_ == 0)
        m11 = (hs == 1) & (as_ == 1)
        d_lh[m00] = -lam_a[m00] * rho
        d_la[m00] = -lam_h[m00] * rho
        d_rho[m00] = -lam_h[m00] * lam_a[m00]
        d_lh[m01] = rho
        d_rho[m01] = lam_h[m01]
        d_la[m10] = rho
        d_rho[m10] = lam_a[m10]
        d_rho[m11] = -1.0

        # Derivada de la log-verosimilitud respecto a log_lh y log_la
        r_h = (hs - lam_h) + (d_lh * lam_h) / tau
        r_a = (as_ - lam_a) + (d_la * lam_a) / tau
        r_h_w = weights * r_h
        r_a_w = weights * r_a

        g_mu = np.sum(r_h_w + r_a_w)
        g_gamma = np.sum(r_h_w * home_adv)
        g_rho = np.sum(weights * (d_rho / tau))
        g_attack = np.bincount(hi, r_h_w, n) + np.bincount(ai, r_a_w, n)
        g_defense = -np.bincount(ai, r_h_w, n) - np.bincount(hi, r_a_w, n)
        g_attack = g_attack - g_attack.mean()
        g_defense = g_defense - g_defense.mean()

        grad = np.concatenate([[g_mu, g_gamma, g_rho], g_attack, g_defense])
        return -grad

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
