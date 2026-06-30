"""Clase que crea modelos de goles para centralizar
la construccion de estrategias
@author Chigga21
"""
from __future__ import annotations

from fifa26.domain.interfaces import GoalModel
from fifa26.models.bayesian_model import BayesianPoissonModel
from fifa26.models.xgboost_model import XGBoostGoalModel


class ModelFactory:
    """Crea estrategias GoalModel por nombre
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
