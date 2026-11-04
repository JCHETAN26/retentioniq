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
        # Bit-for-bit reproducible across runs (multithreaded histograms otherwise vary).
        deterministic=True,
        force_row_wise=True,
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
        # Bit-for-bit reproducible across runs (multithreaded histograms otherwise vary).
        deterministic=True,
        force_row_wise=True,
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

    def diagnostics(self, x_val: pd.DataFrame) -> dict[str, Any]:
        """Diagnostic metrics for S-learner: treatment feature importance and validation effects."""
        booster = getattr(self.model, "booster_", None)
        if booster is None:
            return {}
        feature_names = booster.feature_name()
        gain_importances = booster.feature_importance(importance_type="gain")
        split_importances = booster.feature_importance(importance_type="split")

        gain_dict = dict(zip(feature_names, [float(v) for v in gain_importances], strict=False))
        split_dict = dict(zip(feature_names, [int(v) for v in split_importances], strict=False))

        ranked = sorted(gain_dict.items(), key=lambda item: item[1], reverse=True)
        ranks = {name: rank + 1 for rank, (name, _) in enumerate(ranked)}

        scores = self.score(x_val)
        t_gain = float(gain_dict.get("treatment", 0.0))
        t_split = int(split_dict.get("treatment", 0))
        return {
            "treatment_split_count": t_split,
            "treatment_gain": t_gain,
            "treatment_split_gain": t_gain,
            "treatment_rank": int(ranks.get("treatment", len(feature_names))),
            "treatment_gain_rank": int(ranks.get("treatment", len(feature_names))),
            "n_features": len(feature_names),
            "zero_or_negligible": bool(t_split == 0 or t_gain < 1e-6),
            "val_effect_mean": float(np.mean(scores)),
            "tau_validation_mean": float(np.mean(scores)),
            "tau_validation_std": float(np.std(scores)),
            "tau_validation_share_zero": float(np.mean(scores == 0.0)),
            "tau_validation_share_negative": float(np.mean(scores < 0.0)),
        }


@dataclass
class XLearner:
    """X-learner (Künzel et al. 2019) with 2-fold cross-fitting and known propensity e = P(T=1)."""

    name: str = "x_learner_lgbm"
    first_stage: TwoModel = field(default_factory=TwoModel)
    tau1: Any = None
    tau0: Any = None
    propensity: float = 0.5

    def fit(self, x: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "XLearner":
        from sklearn.model_selection import StratifiedKFold

        self.propensity = float(np.mean(t))
        t_arr = np.asarray(t)
        y_arr = np.asarray(y)
        x_features = x[FEATURES].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        strata = t_arr.astype(str) + "_" + y_arr.astype(str)

        d1 = np.zeros(int(np.sum(t_arr == 1)), dtype=float)
        d0 = np.zeros(int(np.sum(t_arr == 0)), dtype=float)

        t1_indices = np.where(t_arr == 1)[0]
        t0_indices = np.where(t_arr == 0)[0]
        t1_pos = {idx: pos for pos, idx in enumerate(t1_indices)}
        t0_pos = {idx: pos for pos, idx in enumerate(t0_indices)}

        for train_idx, val_idx in skf.split(x_features, strata):
            m1_fold = self.first_stage._new().fit(
                x_features.iloc[train_idx][t_arr[train_idx] == 1],
                y_arr[train_idx][t_arr[train_idx] == 1],
            )
            m0_fold = self.first_stage._new().fit(
                x_features.iloc[train_idx][t_arr[train_idx] == 0],
                y_arr[train_idx][t_arr[train_idx] == 0],
            )

            val_t = t_arr[val_idx]
            val_y = y_arr[val_idx]
            val_x = x_features.iloc[val_idx]

            val_t1 = val_idx[val_t == 1]
            val_t0 = val_idx[val_t == 0]

            if len(val_t1) > 0:
                p_m0 = m0_fold.predict_proba(val_x[val_t == 1])[:, 1]
                for o_idx, p_val, y_val in zip(val_t1, p_m0, val_y[val_t == 1], strict=False):
                    d1[t1_pos[o_idx]] = y_val - p_val

            if len(val_t0) > 0:
                p_m1 = m1_fold.predict_proba(val_x[val_t == 0])[:, 1]
                for o_idx, p_val, y_val in zip(val_t0, p_m1, val_y[val_t == 0], strict=False):
                    d0[t0_pos[o_idx]] = p_val - y_val

        self.tau1 = _lgbm_regressor().fit(x_features.iloc[t1_indices], d1)
        self.tau0 = _lgbm_regressor().fit(x_features.iloc[t0_indices], d0)
        return self

    def score(self, x: pd.DataFrame) -> np.ndarray:
        e = self.propensity
        # Künzel weighting: with e = 0.85, weight goes to tau0, which uses mu1 fit on the
        # large treated arm (~85% of users), yielding a lower-variance CATE estimate.
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
        SLearner(name="s_learner_lgbm_fullcol", model=_lgbm_classifier(colsample_bytree=1.0)),
        TwoModel(name="t_learner_lgbm"),
        XLearner(),
        TransformedOutcome(),
        CausalForest(),
    ]


def get_strategy(name: str) -> Any:
    """Return a fresh instance of the named strategy."""
    for s in strategies():
        if s.name == name:
            return s
    raise ValueError(f"Unknown uplift strategy: {name}")
