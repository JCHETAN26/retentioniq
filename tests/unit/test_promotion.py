import numpy as np
import pandas as pd

from retentioniq.features.retailrocket import ALL_FEATURES
from retentioniq.training.promotion import evaluate_gates


def _frame(n: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-2 * signal))).astype(int)
    frame = pd.DataFrame(
        {
            "sessions_obs": rng.integers(2, 20, n),
            "tenure_days": rng.integers(0, 60, n),
            "ever_purchased": rng.integers(0, 2, n),
        }
    )  # fmt: skip
    return frame, y, signal


def _prob(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def test_better_model_passes_all_gates() -> None:
    frame, y, signal = _frame(4000)
    rng = np.random.default_rng(1)
    good = _prob(2 * signal)
    weak = _prob(0.5 * signal + rng.normal(size=len(y)))
    gates = evaluate_gates(y, good, weak, frame, True, ALL_FEATURES, n_boot=200)
    assert all(g.passed for g in gates), [(g.name, g.detail) for g in gates]


def test_worse_model_fails_non_inferiority() -> None:
    frame, y, signal = _frame(4000)
    rng = np.random.default_rng(1)
    good = _prob(2 * signal)
    weak = _prob(0.5 * signal + rng.normal(size=len(y)))
    gates = {g.name: g for g in evaluate_gates(y, weak, good, frame, True, ALL_FEATURES,
                                               n_boot=200)}  # fmt: skip
    assert not gates["pr_auc_non_inferior"].passed


def test_schema_and_data_gates() -> None:
    frame, y, signal = _frame(1000)
    p = _prob(signal)
    gates = {g.name: g for g in evaluate_gates(y, p, p, frame, False, ALL_FEATURES[:-1],
                                               n_boot=50)}  # fmt: skip
    assert not gates["data_validation"].passed
    assert not gates["schema_contract"].passed
    assert gates["pr_auc_non_inferior"].passed  # identical model is non-inferior
