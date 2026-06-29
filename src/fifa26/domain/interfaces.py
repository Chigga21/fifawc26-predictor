"""Abstract ports (interfaces) of the domain.

These ABCs define the contracts that outer layers must satisfy. The application
layer depends only on these abstractions, never on concrete implementations
(Dependency Inversion Principle).

Note: we deliberately allow `pandas` here. In a data pipeline the tabular frame
is the lingua franca between stages; forcing a conversion to/from lists of
`Match` entities at every boundary would add cost without buying isolation that
matters for this project.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from fifa26.domain.entities import TeamStrength


class MatchRepository(ABC):
    """Port that yields raw match data, regardless of the storage backend."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return matches as a DataFrame with the raw dataset columns."""
        raise NotImplementedError


class GoalModel(ABC):
    """Strategy port: anything that can predict expected goals for fixtures.

    Both the Bayesian (PyMC) and the XGBoost estimators implement this contract,
    so the pipeline can train and compare them interchangeably (Open/Closed).
    """

    name: str

    @abstractmethod
    def fit(
        self,
        matches: pd.DataFrame,
        strengths: dict[str, TeamStrength],
    ) -> "GoalModel":
        """Train using the matches and the Dixon-Coles strengths as features."""
        raise NotImplementedError

    @abstractmethod
    def predict_expected_goals(
        self, fixtures: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) arrays aligned with `fixtures`."""
        raise NotImplementedError
