"""Targeting strategies and CATE estimators (PROJECT_SPEC.md §6.3).

Each learner exposes `fit(X, t, y)` and `score(X)`; higher score = target first. For uplift
learners the score is the estimated individual treatment effect tau(x).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from retentioniq import SEED

FEATURES = [f"f{i}" for i in range(12)]


def _lgbm_classifier(**overrides: Any) -> Any:
    import lightgbm as lgb

    params: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


def _lgbm_regressor(**overrides: Any) -> Any:
    import lightgbm as lgb

    params: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


@dataclass
class RandomTargeting:
    name: str = "random"
    seed: int = SEED

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "RandomTargeting":
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(x))


@dataclass
class ResponseModel:
    """Target the highest P(Y=1 | X), ignoring treatment ("propensity to convert")."""

    name: str = "response_model"
    model: Any = field(default_factory=_lgbm_classifier)

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "ResponseModel":
        self.model.fit(x[FEATURES], y)
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x[FEATURES])[:, 1])


@dataclass
class TwoModel:
    """T-learner: separate response models per arm; tau(x) = mu1(x) - mu0(x).

    With logistic base models this is the conventional "two-model" baseline; with LightGBM it
    is the T-learner of the estimator ladder.
    """

    name: str = "t_learner_lgbm"
    base: str = "lgbm"
    m1: Any = None
    m0: Any = None

    def _new(self) -> Any:
        if self.base == "logistic":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return _lgbm_classifier()

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "TwoModel":
        self.m1 = self._new().fit(x.loc[t == 1, FEATURES], y[t == 1])
        self.m0 = self._new().fit(x.loc[t == 0, FEATURES], y[t == 0])
        return self

    def mu(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (self.m1.predict_proba(x[FEATURES])[:, 1], self.m0.predict_proba(x[FEATURES])[:, 1])

    def score(self, x: pd.DataFrame) -> np.ndarray:
        mu1, mu0 = self.mu(x)
        return np.asarray(mu1 - mu0)


@dataclass
class SLearner:
    """Single model on (X, T); tau(x) = mu(x, 1) - mu(x, 0)."""

    name: str = "s_learner_lgbm"
    model: Any = field(default_factory=_lgbm_classifier)

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "SLearner":
        self.model.fit(x[FEATURES].assign(treatment=t), y)
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        x1 = x[FEATURES].assign(treatment=1)
        x0 = x[FEATURES].assign(treatment=0)
        return np.asarray(self.model.predict_proba(x1)[:, 1] - self.model.predict_proba(x0)[:, 1])


@dataclass
class XLearner:
    """X-learner (Kunzel et al. 2019) with the known randomization propensity e = P(T=1)."""

    name: str = "x_learner_lgbm"
    first_stage: TwoModel = field(default_factory=TwoModel)
    tau1: Any = None
    tau0: Any = None
    propensity: float = 0.5

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "XLearner":
        self.propensity = float(np.mean(t))
        self.first_stage.fit(x, t, y)
        mu1, mu0 = self.first_stage.mu(x)
        d1 = y[t == 1] - mu0[t == 1]  # imputed effect for treated
        d0 = mu1[t == 0] - y[t == 0]  # imputed effect for control
        self.tau1 = _lgbm_regressor().fit(x.loc[t == 1, FEATURES], d1)
        self.tau0 = _lgbm_regressor().fit(x.loc[t == 0, FEATURES], d0)
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        e = self.propensity
        # Weight toward the estimate built from the larger arm's residuals.
        return np.asarray(
            e * self.tau0.predict(x[FEATURES]) + (1 - e) * self.tau1.predict(x[FEATURES])
        )


@dataclass
class TransformedOutcome:
    """Regress the transformed outcome Z = Y (T - e) / (e (1 - e)); E[Z | X] = tau(X)."""

    name: str = "transformed_outcome_lgbm"
    model: Any = field(default_factory=lambda: _lgbm_regressor(min_child_samples=1000))

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "TransformedOutcome":
        e = float(np.mean(t))
        z = y * (t - e) / (e * (1 - e))
        self.model.fit(x[FEATURES], z)
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(x[FEATURES]))


@dataclass
class CausalForest:
    """Causal forest (Wager & Athey 2018) via EconML's CausalForestDML.

    Treatment is randomized, so the treatment model is the constant assignment rate; the
    outcome nuisance model is a small LightGBM. Fit on a stratified subsample of at most
    `max_rows` training rows (forest training cost; PROJECT_SPEC.md change log v0.5).
    """

    name: str = "causal_forest"
    max_rows: int = 1_000_000
    model: Any = None

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "CausalForest":
        from econml.dml import CausalForestDML
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split

        if len(x) > self.max_rows:
            strata = pd.Series(t).astype(str) + "_" + pd.Series(y).astype(str)
            idx, _ = train_test_split(
                np.arange(len(x)), train_size=self.max_rows, stratify=strata, random_state=SEED
            )
            x, t, y = x.iloc[idx], t[idx], y[idx]
        self.model = CausalForestDML(
            model_y=_lgbm_regressor(n_estimators=200),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            n_estimators=200,
            min_samples_leaf=500,
            max_samples=0.2,
            cv=2,
            random_state=SEED,
            n_jobs=-1,
        )
        self.model.fit(y, t, X=x[FEATURES].to_numpy())
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.effect(x[FEATURES].to_numpy())).ravel()


def strategies() -> list[Any]:
    """In ladder order: baselines first, then CATE estimators."""
    return [
        RandomTargeting(),
        ResponseModel(),
        TwoModel(name="two_model_logistic", base="logistic"),
        SLearner(),
        TwoModel(name="t_learner_lgbm"),
        XLearner(),
        TransformedOutcome(),
        CausalForest(),
    ]
