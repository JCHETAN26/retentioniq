import numpy as np
import pandas as pd
import pytest

from retentioniq.models.churn.ladder import COMPLEXITY_ORDER
from retentioniq.models.churn.metrics import (
    classification_metrics,
    cost_optimal_threshold,
)
from retentioniq.models.churn.selection import select_champion


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


def _ladder_val_preds(n: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 0.8, n)
    preds = {
        "M0_naive_recency": rng.random(n),  # no signal
        "M1_logistic_rfm": signal + rng.normal(0, 2.0, n),  # weak
        "M2_logistic_full": signal + rng.normal(0, 0.15, n),  # slightly worse, tied with the best
        "M3_random_forest": signal + rng.normal(0, 0.1, n),
        "M4_lightgbm": signal,  # best
        "M5_pytorch_mlp": signal + rng.normal(0, 0.5, n),
    }
    return y, preds


def test_selection_picks_simplest_model_tied_with_best_on_validation() -> None:
    y, preds = _ladder_val_preds(2000, 1)
    sel = select_champion(y, preds, COMPLEXITY_ORDER, n_boot=200, seed=0)
    assert sel.best_by_validation == "M4_lightgbm"
    assert sel.selected_champion == "M2_logistic_full"
    assert [c["model"] for c in sel.candidates_checked][:2] == [
        "M0_naive_recency",
        "M1_logistic_rfm",
    ]
    assert not sel.candidates_checked[0]["tied_with_best"]
    assert sel.candidates_checked[-1]["tied_with_best"]


def test_selection_keeps_best_when_every_simpler_model_is_worse() -> None:
    y, preds = _ladder_val_preds(2000, 2)
    del preds["M2_logistic_full"], preds["M3_random_forest"]
    sel = select_champion(y, preds, COMPLEXITY_ORDER, n_boot=200, seed=0)
    assert sel.selected_champion == sel.best_by_validation == "M4_lightgbm"
    assert all(not c["tied_with_best"] for c in sel.candidates_checked)
    assert "significantly worse on validation" in sel.reasoning


def test_selection_signature_accepts_no_test_data() -> None:
    """Structural leakage guard: the selector can only be given validation arrays."""
    import inspect

    params = set(inspect.signature(select_champion).parameters)
    assert params == {
        "y_val",
        "val_preds",
        "complexity_order",
        "groups",
        "n_boot",
        "seed",
        "latency_p95_ms",
        "latency_budget_ms",
    }


def test_selection_excludes_models_over_the_latency_budget() -> None:
    """A model tied on accuracy but too slow to serve cannot be champion (spec v0.6.3)."""
    y, preds = _ladder_val_preds(2000, 1)
    fast = dict.fromkeys(preds, 0.5)
    without = select_champion(y, preds, COMPLEXITY_ORDER, n_boot=200, seed=0, latency_p95_ms=fast)
    assert without.selected_champion == "M2_logistic_full"

    slow = {**fast, "M2_logistic_full": 40.0, "M3_random_forest": 50.0}
    sel = select_champion(y, preds, COMPLEXITY_ORDER, n_boot=200, seed=0, latency_p95_ms=slow)
    assert sel.selected_champion == "M4_lightgbm"
    assert {d["model"] for d in sel.ineligible} == {"M2_logistic_full", "M3_random_forest"}
    assert "Not eligible" in sel.reasoning
    # Ineligible models keep their validation score in the record, for reporting.
    assert "M3_random_forest" in sel.val_pr_auc


def test_selection_fails_loudly_when_nothing_is_servable() -> None:
    y, preds = _ladder_val_preds(500, 3)
    with pytest.raises(ValueError, match="no eligible"):
        select_champion(
            y, preds, COMPLEXITY_ORDER, n_boot=50, latency_p95_ms=dict.fromkeys(preds, 99.0)
        )


def test_score_one_fast_path_matches_predict_proba() -> None:
    """The booster fast path the API uses must give the same score as the sklearn path."""
    from retentioniq.models.churn.ladder import GradientBoosting, Logistic
    from retentioniq.models.churn.latency import measure_latency_ms, score_one

    rng = np.random.default_rng(0)
    feats = ["recency_days", "sessions_obs", "tenure_days"]
    x = pd.DataFrame(rng.random((400, 3)), columns=feats)
    y = (x["recency_days"] + rng.normal(0, 0.2, 400) > 0.5).astype(int).to_numpy()
    ident = lambda p: p  # noqa: E731
    for model in (GradientBoosting(features=feats), Logistic(features=feats, name="M2")):
        model.fit(x[:300], y[:300], x[300:], y[300:])
        ref = model.predict_proba(x[300:305])
        fast = [score_one(model, ident, feats, r) for r in x[300:305].to_dict("records")]
        np.testing.assert_allclose(fast, ref, rtol=1e-6)
        lat = measure_latency_ms(model, ident, feats, x, n=30, warmup=5)
        assert lat["n"] == 30 and lat["p95_ms"] >= lat["p50_ms"] > 0


def test_linear_fast_path_matches_pipeline() -> None:
    """Folded logistic coefficients must reproduce the sklearn pipeline exactly, for a mix of
    log1p and plain-scaled columns, and when the model uses a subset of the request's fields."""
    from retentioniq.models.churn.ladder import Logistic
    from retentioniq.models.churn.latency import score_one

    rng = np.random.default_rng(1)
    feats = ["cart_ratio", "recency_days", "is_weekend_shopper", "sessions_obs", "views_obs"]
    x = pd.DataFrame(rng.random((500, 5)) * [1, 60, 1, 20, 300], columns=feats)
    x["is_weekend_shopper"] = (x["is_weekend_shopper"] > 0.5).astype(float)
    y = (x["recency_days"] / 60 + rng.normal(0, 0.3, 500) > 0.5).astype(int).to_numpy()
    ident = lambda p: p  # noqa: E731
    subset = ["views_obs", "cart_ratio", "recency_days"]
    for model_feats in (feats, subset):
        model = Logistic(features=model_feats, name="M")
        model.fit(x[:400], y[:400], x[400:], y[400:])
        form = model.linear_form()
        assert form.log_mask.any() and not form.log_mask.all()
        ref = model.predict_proba(x[400:])
        fast = [score_one(model, ident, feats, r) for r in x[400:].to_dict("records")]
        np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-12)


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
