import numpy as np
import pytest

from retentioniq.models.uplift.metrics import (
    evaluate_uplift,
    incremental_at_k,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)


def _data(n: int = 100_000, seed: int = 0):  # type: ignore[no-untyped-def]
    """Uplift is 0.10 for x > 0.8, 0 otherwise; baseline response 0.05."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    t = rng.integers(0, 2, n)
    tau = np.where(x > 0.8, 0.10, 0.0)
    y = (rng.random(n) < 0.05 + tau * t).astype(int)
    return x, t, y


def test_informative_score_beats_random() -> None:
    x, t, y = _data()
    rng = np.random.default_rng(1)
    good = qini_coefficient(y, t, x)
    rand = qini_coefficient(y, t, rng.random(len(x)))
    assert good > 0.1
    assert abs(rand) < 0.05
    assert good > 3 * abs(rand)
    assert qini_coefficient(y, t, y * t - y * (1 - t)) == pytest.approx(1.0)


def test_uplift_at_k_matches_known_effect() -> None:
    x, t, y = _data()
    assert uplift_at_k(y, t, x, 0.2) == pytest.approx(0.10, abs=0.015)
    assert uplift_at_k(y, t, -x, 0.2) == pytest.approx(0.0, abs=0.015)


def test_incremental_at_k_scales_and_has_ci() -> None:
    x, t, y = _data()
    r = incremental_at_k(y, t, x, 0.10, population=1_000_000, n_boot=500)
    # 100k targeted users, true uplift 0.10 -> ~10k incremental outcomes
    assert r["incremental"] == pytest.approx(10_000, rel=0.2)
    assert r["incremental_ci_low"] < r["incremental"] < r["incremental_ci_high"]


def test_qini_curve_endpoints() -> None:
    x, t, y = _data(1000)
    c = qini_curve(y, t, x)
    assert c.value[0] == 0
    n_t, n_c = t.sum(), (1 - t).sum()
    assert c.value[-1] == pytest.approx(y[t == 1].sum() - y[t == 0].sum() * n_t / n_c)


def test_evaluate_uplift_keys() -> None:
    x, t, y = _data(5000)
    out = evaluate_uplift(y, t, x)
    assert {"qini_coefficient", "auuc", "uplift@1", "uplift@30"} <= out.keys()
