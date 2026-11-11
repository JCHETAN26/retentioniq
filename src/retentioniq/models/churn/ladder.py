"""Model ladder M0-M5 (PROJECT_SPEC.md §5.2).

Every model exposes `fit(X_train, y_train, X_val, y_val)` and `predict_proba(X) -> P(inactive)`.
Hyperparameter grids are fixed here, before seeing test data; selection uses validation PR-AUC.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from retentioniq import SEED

# Heavy-tailed, non-negative count features get log1p before scaling for linear models.
_LOG_FEATURES_PREFIXES = (
    "views_",
    "sessions_",
    "transactions_",
    "carts_",
    "unique_",
    "items_",
    "events_",
    "active_",
    "purchase_sessions",
    "tenure_",
    "recency_",
    "days_since",
    "mean_",
    "std_",
)

COMPLEXITY_ORDER = [
    "M0_naive_recency",
    "M1_logistic_rfm",
    "M2_logistic_full",
    "M3_random_forest",
    "M4_lightgbm",
    "M5_pytorch_mlp",
]


class ChurnModel(Protocol):
    name: str
    params: dict[str, Any]

    def fit(
        self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> "ChurnModel": ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


def _select(
    candidates: list[Any], x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    best, best_score, best_params = None, -np.inf, {}
    for params, est in candidates:
        est.fit(x, y)
        score = average_precision_score(y_val, est.predict_proba(x_val)[:, 1])
        if score > best_score:
            best, best_score, best_params = est, score, params
    return best, best_params


@dataclass
class RecencyBaseline:
    """M0: inactive rate per training recency decile (ranks by recency, calibrated by lookup)."""

    name: str = "M0_naive_recency"
    params: dict[str, Any] = field(default_factory=dict)
    edges_: np.ndarray | None = None
    rates_: np.ndarray | None = None

    def fit(
        self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> "RecencyBaseline":
        r = x["recency_days"].to_numpy()
        self.edges_ = np.unique(np.quantile(r, np.linspace(0, 1, 11)[1:-1]))
        bins = np.searchsorted(self.edges_, r, side="right")
        overall = float(np.mean(y))
        self.rates_ = np.array(
            [
                np.mean(y[bins == b]) if (bins == b).any() else overall
                for b in range(len(self.edges_) + 1)
            ]
        )
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.edges_ is not None and self.rates_ is not None
        bins = np.searchsorted(self.edges_, x["recency_days"].to_numpy(), side="right")
        return np.asarray(self.rates_[bins], dtype=float)


def _linear_pipeline(columns: list[str], c: float) -> Pipeline:
    log_cols = [col for col in columns if col.startswith(_LOG_FEATURES_PREFIXES)]
    other = [col for col in columns if col not in log_cols]
    pre = ColumnTransformer(
        [
            (
                "log",
                Pipeline([("log1p", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]),
                log_cols,
            ),
            ("std", StandardScaler(), other),
        ]
    )
    clf = LogisticRegression(C=c, max_iter=2000, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


@dataclass
class Logistic:
    features: list[str]
    name: str = "logistic"
    params: dict[str, Any] = field(default_factory=dict)
    model_: Pipeline | None = None

    def fit(
        self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> "Logistic":
        cands = [({"C": c}, _linear_pipeline(self.features, c)) for c in (0.01, 0.1, 1.0, 10.0)]
        self.model_, self.params = _select(cands, x[self.features], y, x_val[self.features], y_val)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return np.asarray(self.model_.predict_proba(x[self.features])[:, 1])

    def linear_form(self) -> "LinearForm":
        """The fitted pipeline folded into NumPy arrays, for single-record serving."""
        assert self.model_ is not None
        cached = self.__dict__.get("_linear_form")
        if cached is None:
            cached = LinearForm.from_pipeline(self.model_)
            self.__dict__["_linear_form"] = cached
        return cached


@dataclass(frozen=True)
class LinearForm:
    """log1p on some columns -> standardise -> logistic, as plain arrays.

    Equivalent to `_linear_pipeline(...).predict_proba(x)[:, 1]`, without the per-call
    DataFrame, column-transformer and validation overhead (spec §5.5 serving budget).
    """

    features: tuple[str, ...]  # in the transformer's output order
    log_mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_pipeline(cls, pipe: Pipeline) -> "LinearForm":
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        feats: list[str] = []
        log_mask: list[bool] = []
        means: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for name, trans, cols in pre.transformers_:
            if name == "remainder" or len(cols) == 0:
                continue
            scaler = trans.named_steps["scale"] if name == "log" else trans
            feats += list(cols)
            log_mask += [name == "log"] * len(cols)
            means.append(np.asarray(scaler.mean_, dtype=np.float64))
            scales.append(np.asarray(scaler.scale_, dtype=np.float64))
        return cls(
            tuple(feats),
            np.asarray(log_mask, dtype=bool),
            np.concatenate(means),
            np.concatenate(scales),
            np.asarray(clf.coef_[0], dtype=np.float64),
            float(clf.intercept_[0]),
        )

    def predict_one(self, values: Any) -> float:
        x = np.fromiter((values[f] for f in self.features), np.float64, len(self.features))
        x[self.log_mask] = np.log1p(x[self.log_mask])
        z = float(((x - self.mean) / self.scale) @ self.coef) + self.intercept
        return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class RandomForest:
    features: list[str]
    name: str = "M3_random_forest"
    params: dict[str, Any] = field(default_factory=dict)
    model_: RandomForestClassifier | None = None

    def fit(
        self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> "RandomForest":
        cands = [
            (
                {"min_samples_leaf": leaf, "max_features": mf},
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=leaf,
                    max_features=mf,
                    n_jobs=-1,
                    random_state=SEED,
                ),
            )
            for leaf in (20, 100)
            for mf in ("sqrt", 0.5)
        ]
        self.model_, self.params = _select(cands, x[self.features], y, x_val[self.features], y_val)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        return np.asarray(self.model_.predict_proba(x[self.features])[:, 1])


@dataclass
class GradientBoosting:
    features: list[str]
    name: str = "M4_lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    model_: Any = None

    def fit(
        self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> "GradientBoosting":
        import lightgbm as lgb

        best_score = -np.inf
        for num_leaves in (15, 63):
            for lr in (0.03, 0.1):
                m = lgb.LGBMClassifier(
                    n_estimators=2000,
                    learning_rate=lr,
                    num_leaves=num_leaves,
                    min_child_samples=50,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=SEED,
                    verbose=-1,
                    deterministic=True,
                    force_row_wise=True,
                )
                m.fit(
                    x[self.features],
                    y,
                    eval_set=[(x_val[self.features], y_val)],
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
                p_val = np.asarray(m.predict_proba(x_val[self.features]))
                score = average_precision_score(y_val, p_val[:, 1])
                if score > best_score:
                    best_score, self.model_ = score, m
                    self.params = {
                        "num_leaves": num_leaves,
                        "learning_rate": lr,
                        "best_iteration": int(m.best_iteration_),
                    }
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(x[self.features])[:, 1])


@dataclass
class TorchMLP:
    """M5: a small MLP tuned over a 6-config grid. Kept only if it beats simpler models per §5.5."""

    features: list[str]
    name: str = "M5_pytorch_mlp"
    max_epochs: int = 100
    params: dict[str, Any] = field(default_factory=dict)
    model_: Any = None
    scaler_: Any = None

    def _prep(self, x: pd.DataFrame) -> np.ndarray:
        arr = x[self.features].to_numpy(dtype=np.float32)
        return np.sign(arr) * np.log1p(np.abs(arr))

    def fit(
        self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray
    ) -> "TorchMLP":
        import torch
        from torch import nn

        torch.set_num_threads(1)
        self.scaler_ = StandardScaler().fit(self._prep(x))
        xt = torch.tensor(self.scaler_.transform(self._prep(x)), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        xv = torch.tensor(self.scaler_.transform(self._prep(x_val)), dtype=torch.float32)

        configs: list[tuple[tuple[int, int], float]] = [
            ((h1, h2), drop)
            for (h1, h2) in ((64, 32), (128, 64), (256, 128))
            for drop in (0.1, 0.3)
        ]

        best_overall_score = -np.inf
        best_overall_state = None
        best_overall_params: dict[str, Any] = {}
        best_model: nn.Sequential | None = None

        for (h1, h2), drop in configs:
            torch.manual_seed(SEED)
            model = nn.Sequential(
                nn.Linear(xt.shape[1], h1),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(h2, 1),
            )
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss()
            gen = torch.Generator().manual_seed(SEED)
            best_score, best_state, patience, epochs = -np.inf, None, 0, 0

            for epoch in range(self.max_epochs):
                model.train()
                for idx in torch.randperm(len(xt), generator=gen).split(1024):
                    opt.zero_grad()
                    loss = loss_fn(model(xt[idx]).squeeze(1), yt[idx])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(model(xv).squeeze(1)).numpy()
                score = average_precision_score(y_val, pv)
                if score > best_score + 1e-4:
                    best_score, patience, epochs = score, 0, epoch + 1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 8:
                        break

            if best_score > best_overall_score and best_state is not None:
                best_overall_score = best_score
                best_overall_state = best_state
                best_overall_params = {
                    "hidden": [h1, h2],
                    "dropout": drop,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "epochs": epochs,
                }
                best_model = model

        assert best_overall_state is not None and best_model is not None
        best_model.load_state_dict(best_overall_state)
        self.model_ = best_model.eval()
        self.params = best_overall_params
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        with torch.no_grad():
            xt = torch.tensor(self.scaler_.transform(self._prep(x)), dtype=torch.float32)
            return np.asarray(torch.sigmoid(self.model_(xt).squeeze(1)).numpy(), dtype=float)
