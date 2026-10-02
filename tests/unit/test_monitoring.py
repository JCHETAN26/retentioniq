from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from retentioniq.features.retailrocket import ALL_FEATURES
from retentioniq.monitoring.drift import level, monitor, psi


def test_psi_identical_is_zero_and_shift_is_large() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    assert psi(a, rng.normal(size=20_000)) < 0.01
    assert psi(a, rng.normal(1.0, 1, 20_000)) > 0.25


def test_levels() -> None:
    assert level(0.05, "feature_psi") == "ok"
    assert level(0.2, "feature_psi") == "warn"
    assert level(0.3, "feature_psi") == "alert"


def _frame(rng: np.random.Generator, shift: float = 0.0) -> pd.DataFrame:
    df = pd.DataFrame({f: rng.gamma(2, 2, 5000) for f in ALL_FEATURES})
    df["recency_days"] += shift
    df["inactive"] = (rng.random(5000) < 0.6).astype(int)
    return df


def test_monitor_flags_drifted_feature_and_staleness() -> None:
    rng = np.random.default_rng(1)
    ref, cur = _frame(rng), _frame(rng, shift=5.0)
    p = rng.random(5000)
    now = datetime(2026, 1, 10, tzinfo=UTC)
    rep = monitor(ref, cur, p, p, 0.6, data_as_of=now - timedelta(days=4), now=now)
    assert rep["drifted_features"] == ["recency_days"]
    assert rep["freshness"]["level"] == "alert"
    assert rep["status"] == "alert"
    assert "performance" in rep


def test_monitor_ok_when_stable() -> None:
    rng = np.random.default_rng(2)
    ref, cur = _frame(rng), _frame(rng)
    y = cur["inactive"].to_numpy()
    p = np.clip(0.6 + 0.3 * (y - 0.6) + rng.normal(0, 0.05, len(y)), 0, 1)
    rep = monitor(ref, cur, p, p, reference_pr_auc=0.9)
    assert rep["drifted_features"] == []
    assert rep["predictions"]["level"] == "ok"
