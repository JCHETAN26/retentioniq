import numpy as np
import pandas as pd
import pytest

from retentioniq.models.churn.ladder import COMPLEXITY_ORDER
from retentioniq.models.churn.metrics import (
    classification_metrics,
    cost_optimal_threshold,
    paired_bootstrap_diff,
)


def test_returner_metrics_minority_class() -> None:
    # 80% positive (inactive), 20% negative (returners)
    y = np.array([1, 1, 1, 1, 0] * 20)
    p = np.array([0.9, 0.8, 0.85, 0.95, 0.1] * 20)
    m = classification_metrics(y, p)

    assert "pr_auc_returners" in m
    assert "returner_base_rate" in m
    assert "precision_returners@10" in m
    assert "lift_returners@10" in m

    assert m["returner_base_rate"] == pytest.approx(0.2)
    # The returners have low p, so high 1-p; precision@10 on returners should be high
    assert m["precision_returners@10"] > 0.5
    assert m["lift_returners@10"] > 2.0


def test_cost_optimal_threshold() -> None:
    # 100 users, half are churners
    y = np.array([1] * 50 + [0] * 50)
    # Higher predicted risk for churners
    p = np.linspace(0.9, 0.1, 100)

    # High value saved, low cost to contact: should contact many (lower threshold)
    low_cost_thr = cost_optimal_threshold(y, p, cost_contact=0.1, value_saved=100.0, save_rate=0.5)
    assert 0.1 <= low_cost_thr <= 0.7

    # Extremely high cost to contact, tiny value: should contact nobody (threshold = 1.0)
    high_cost_thr = cost_optimal_threshold(
        y, p, cost_contact=100.0, value_saved=1.0, save_rate=0.01
    )
    assert high_cost_thr == 1.0


def test_ladder_selection_rule_picks_simplest_not_significantly_worse() -> None:
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, n)

    # Suppose M4 is best by validation, but M2 is virtually identical on test
    base_signal = y * 0.4 + rng.normal(0, 0.1, n)
    preds = {
        "M0_naive_recency": np.full(n, 0.5),
        "M1_logistic_rfm": np.clip(base_signal * 0.5, 0, 1),
        "M2_logistic_full": np.clip(base_signal + rng.normal(0, 0.001, n), 0, 1),
        "M3_random_forest": np.clip(base_signal + rng.normal(0, 0.005, n), 0, 1),
        "M4_lightgbm": np.clip(base_signal, 0, 1),
        "M5_pytorch_mlp": np.clip(base_signal - 0.01, 0, 1),
    }

    best_val_model = "M4_lightgbm"
    best_idx = COMPLEXITY_ORDER.index(best_val_model)

    chosen = best_val_model
    for cand in COMPLEXITY_ORDER[:best_idx]:
        lo, hi = paired_bootstrap_diff(
            y,
            preds[cand],
            preds[best_val_model],
            lambda yt, pt: float(np.mean(yt * pt)),
            n_boot=100,
            seed=42,
        )
        if lo <= 0.0 <= hi:
            chosen = cand
            break

    # M2 is virtually identical to M4, so it should be picked over M4 because it is simpler
    assert chosen in ("M1_logistic_rfm", "M2_logistic_full", "M3_random_forest")
    assert COMPLEXITY_ORDER.index(chosen) < COMPLEXITY_ORDER.index("M4_lightgbm")


def test_torch_mlp_6_config_grid() -> None:
    pytest.importorskip("torch")
    from retentioniq.models.churn.ladder import TorchMLP

    rng = np.random.default_rng(0)
    features = ["recency_days", "sessions_obs"]
    x_tr = pd.DataFrame(rng.random((60, 2)), columns=features)
    y_tr = (x_tr["recency_days"] > 0.5).astype(int).to_numpy()
    x_val = pd.DataFrame(rng.random((30, 2)), columns=features)
    y_val = (x_val["recency_days"] > 0.5).astype(int).to_numpy()

    mlp = TorchMLP(features, max_epochs=3)
    mlp.fit(x_tr, y_tr, x_val, y_val)
    assert "hidden" in mlp.params
    assert "dropout" in mlp.params
    assert mlp.params["hidden"] in [[64, 32], [128, 64], [256, 128]]
    assert mlp.params["dropout"] in [0.1, 0.3]
    preds = mlp.predict_proba(x_val)
    assert len(preds) == len(x_val)
    assert (preds >= 0).all() and (preds <= 1).all()
