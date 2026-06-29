"""Factory for goal models (centralises Strategy construction)."""
from __future__ import annotations

from fifa26.domain.interfaces import GoalModel
from fifa26.models.bayesian_model import BayesianPoissonModel
from fifa26.models.xgboost_model import XGBoostGoalModel


class ModelFactory:
    """Creates `GoalModel` strategies by name.

    Keeping construction in one place means callers (e.g. the pipeline) never
    import or instantiate concrete model classes directly: they ask the factory
    for a name and depend only on the `GoalModel` abstraction.
    """

    _REGISTRY = {
        "xgboost": XGBoostGoalModel,
        "bayesian": BayesianPoissonModel,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> GoalModel:
        key = name.lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Modelo desconocido '{name}'. Disponibles: {sorted(cls._REGISTRY)}"
            )
        return cls._REGISTRY[key](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._REGISTRY)
