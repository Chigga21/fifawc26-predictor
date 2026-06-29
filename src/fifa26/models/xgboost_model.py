"""XGBoost goal-regression model (a `GoalModel` strategy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from fifa26.domain.entities import TeamStrength
from fifa26.domain.interfaces import GoalModel

_FEATURES = ["attack", "opp_defense", "strength_diff", "is_home", "is_competitive"]


class XGBoostGoalModel(GoalModel):
    """Gradient-boosted Poisson regression of goals scored.

    Unlike the (log-linear) Dixon-Coles / Bayesian models, XGBoost can learn
    non-linear interactions between the engineered strength features. We frame
    the problem in a "long" layout (one row per team-side of each match) so a
    single regressor predicts the goals of whichever side we ask about, then
    reuse it for both the home and away lambdas.

    `objective="count:poisson"` guarantees non-negative predictions that are
    valid Poisson means, which is exactly what the downstream score matrix needs.
    """

    name = "XGBoost"

    def __init__(self, **xgb_kwargs) -> None:
        params = dict(
            objective="count:poisson",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        params.update(xgb_kwargs)
        self._model = XGBRegressor(**params)
        self._strengths: dict[str, TeamStrength] = {}

    def fit(self, matches: pd.DataFrame, strengths: dict[str, TeamStrength]) -> "XGBoostGoalModel":
        self._strengths = strengths
        home = self._side_frame(matches, scoring="home")
        away = self._side_frame(matches, scoring="away")
        long_df = pd.concat([home, away], ignore_index=True)
        self._model.fit(long_df[_FEATURES], long_df["goals"])
        return self

    def predict_expected_goals(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home_feats = self._fixture_features(fixtures, scoring="home")
        away_feats = self._fixture_features(fixtures, scoring="away")
        lambda_home = self._model.predict(home_feats[_FEATURES])
        lambda_away = self._model.predict(away_feats[_FEATURES])
        return np.asarray(lambda_home, float), np.asarray(lambda_away, float)

    # ----------------------------------------------------------------- private
    def _strength(self, team: str) -> TeamStrength:
        return self._strengths.get(team, TeamStrength(team, 0.0, 0.0))

    def _side_frame(self, matches: pd.DataFrame, scoring: str) -> pd.DataFrame:
        df = self._fixture_features(matches, scoring=scoring)
        df["goals"] = matches["home_score" if scoring == "home" else "away_score"].to_numpy()
        return df

    def _fixture_features(self, fixtures: pd.DataFrame, scoring: str) -> pd.DataFrame:
        if scoring == "home":
            team, opp = fixtures["home_team"], fixtures["away_team"]
            is_home = (~fixtures["neutral"]).astype(float)
        else:
            team, opp = fixtures["away_team"], fixtures["home_team"]
            is_home = np.zeros(len(fixtures))

        attack = team.map(lambda t: self._strength(t).attack).to_numpy()
        opp_defense = opp.map(lambda t: self._strength(t).defense).to_numpy()
        competitive = (fixtures["tournament"] != "Friendly").astype(float).to_numpy()
        return pd.DataFrame(
            {
                "attack": attack,
                "opp_defense": opp_defense,
                "strength_diff": attack - opp_defense,
                "is_home": np.asarray(is_home, float),
                "is_competitive": competitive,
            }
        )
