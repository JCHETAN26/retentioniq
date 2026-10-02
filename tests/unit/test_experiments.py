import numpy as np
import pandas as pd
import pytest

from retentioniq.analytics.experiments.inference import (
    bootstrap_ate_ci,
    compare_arms,
    holm_adjust,
    sample_ratio_test,
    standardized_mean_differences,
    wilson_interval,
)


def _experiment(n: int, p0: float, p1: float, share: float = 0.5, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < share).astype(int)
    y = (rng.random(n) < np.where(t == 1, p1, p0)).astype(int)
    return y, t


def test_compare_arms_recovers_effect_and_ci_covers_truth() -> None:
    y, t = _experiment(200_000, 0.04, 0.05)
    r = compare_arms(y, t, "visit")
    assert r.ate == pytest.approx(0.01, abs=0.002)
    for lo, hi in (r.ate_ci_newcombe, r.ate_ci_wald, bootstrap_ate_ci(y, t, seed=1)):
        assert lo < 0.01 < hi
    assert r.relative_lift_ci[0] < 0.25 < r.relative_lift_ci[1]
    assert r.p_value < 1e-6


def test_null_effect_is_not_significant() -> None:
    y, t = _experiment(50_000, 0.05, 0.05, seed=3)
    r = compare_arms(y, t, "visit")
    assert r.ate_ci_newcombe[0] < 0 < r.ate_ci_newcombe[1]
    assert r.p_value > 0.01


def test_wilson_interval_known_value() -> None:
    lo, hi = wilson_interval(10, 100)
    assert (round(lo, 4), round(hi, 4)) == (0.0552, 0.1744)


def test_holm_adjust() -> None:
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_balance_checks() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 10_000)
    x = pd.DataFrame({"a": rng.normal(size=10_000), "b": rng.normal(size=10_000) + 0.5 * t})
    smd = standardized_mean_differences(x, t)
    assert abs(smd["a"]) < 0.1 and smd["b"] > 0.4
    assert sample_ratio_test(t, 0.5)["p_value"] > 0.001
    assert sample_ratio_test(t, 0.85)["p_value"] < 1e-10
