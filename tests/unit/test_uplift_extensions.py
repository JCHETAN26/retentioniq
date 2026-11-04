"""Unit tests for Phase 2 uplift modeling extensions."""

import numpy as np
import pandas as pd
import pytest

from retentioniq.models.uplift.learners import (
    FEATURES,
    SLearner,
    XLearner,
    get_strategy,
    strategies,
)
from retentioniq.models.uplift.metrics import (
    paired_bootstrap_qini_diff,
    top_k_counts,
    uplift_by_score_decile,
)


@pytest.fixture
def synthetic_criteo() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 400
    feat_data = {f: rng.normal(size=n).astype(np.float32) for f in FEATURES}
    df = pd.DataFrame(feat_data)
    # 85% treatment share
    t = (rng.uniform(size=n) < 0.85).astype(np.int8)
    # Synthetic uplift: base visit rate 0.05 + treatment effect 0.02 * (feat1 > 0)
    p_visit = 0.05 + 0.02 * t * (df["f0"].to_numpy() > 0)
    y = (rng.uniform(size=n) < p_visit).astype(np.int8)
    df["treatment"] = t
    df["visit"] = y
    df["conversion"] = (y & (rng.uniform(size=n) < 0.2)).astype(np.int8)
    return df, t, y


def test_strategies_includes_fullcol() -> None:
    strats = strategies()
    names = [s.name for s in strats]
    assert "s_learner_lgbm_fullcol" in names
    s_full = get_strategy("s_learner_lgbm_fullcol")
    assert s_full.name == "s_learner_lgbm_fullcol"


def test_x_learner_cross_fitting(
    synthetic_criteo: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    df, t, y = synthetic_criteo
    learner = XLearner()
    learner.fit(df, t, y)
    preds = learner.score(df)
    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_s_learner_diagnostics(
    synthetic_criteo: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    df, t, y = synthetic_criteo
    s_learner = SLearner()
    s_learner.fit(df, t, y)
    diag = s_learner.diagnostics(df)
    assert "treatment_split_count" in diag
    assert "treatment_gain" in diag
    assert "treatment_gain_rank" in diag
    assert "val_effect_mean" in diag
    assert "zero_or_negligible" in diag
    assert isinstance(diag["zero_or_negligible"], bool)


def test_paired_bootstrap_qini_diff(
    synthetic_criteo: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    _, t, y = synthetic_criteo
    rng = np.random.default_rng(123)
    score_a = rng.normal(size=len(y))
    score_b = score_a.copy()

    # Identical scores should yield delta 0 and CI covering 0
    res = paired_bootstrap_qini_diff(y, t, score_a, score_b, n_boot=50, seed=42)
    assert abs(res["delta_qini"]) < 1e-6
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
    assert not res["significant"]


def test_uplift_by_score_decile(
    synthetic_criteo: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    _, t, y = synthetic_criteo
    score = np.linspace(0, 1, len(y))
    deciles = uplift_by_score_decile(y, t, score, outcome_name="visit")
    assert len(deciles) == 10
    assert deciles[0]["decile"] == 1
    assert deciles[-1]["decile"] == 10
    # Decile 10 has highest predicted uplift
    assert deciles[-1]["predicted_uplift"] > deciles[0]["predicted_uplift"]
    for row in deciles:
        assert "rate_control" in row
        assert "rate_treatment" in row
        assert "observed_uplift" in row
        assert "ci_low" in row
        assert "ci_high" in row


def test_campaign_optimal_reach_logic() -> None:
    reaches = np.round(np.arange(0.01, 0.51, 0.01), 2)
    n = 1000
    rng = np.random.default_rng(42)
    y = (rng.uniform(size=n) < 0.1).astype(int)
    t = (rng.uniform(size=n) < 0.85).astype(int)
    # Perfect score: targets true positives
    score = y.astype(float) + rng.normal(0, 0.01, size=n)

    cost_per_t = 0.01
    val_per_c = 10.0
    strat_rows = []
    for r in reaches:
        yt, nt, yc, nc = top_k_counts(y, t, score, float(r))
        targeted = float(r) * n
        inc = (yt / nt - yc / nc) * targeted if nt and nc else 0.0
        cost = targeted * cost_per_t
        net = inc * val_per_c - cost
        strat_rows.append({"reach": float(r), "net_value": net})

    opt_idx = int(np.argmax([r["net_value"] for r in strat_rows]))
    opt_reach = strat_rows[opt_idx]["reach"]
    assert 0.01 <= opt_reach <= 0.50
