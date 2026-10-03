import numpy as np
import pandas as pd

from retentioniq.analytics.visitors import FEATURES, segmentation


def _snap(n: int, split: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.gamma(2, 2, n) for f in FEATURES})
    df["activity_trend"] = rng.uniform(-1, 1, n)
    # Inactivity driven mostly by recency, so an R x F grid should carry real signal.
    p = 1 / (1 + np.exp(-(df["recency_days"] - 4)))
    df["inactive"] = (rng.random(n) < p).astype(int)
    df["visitor_id"] = np.arange(n) + (0 if split == "train" else 10_000)
    df["ever_purchased"] = rng.integers(0, 2, n)
    return df.assign(split=split)


def test_segmentation_is_fit_on_train_and_scored_on_test() -> None:
    snap = pd.concat([_snap(3000, "train", 0), _snap(1500, "test", 1)], ignore_index=True)
    out = segmentation(snap)
    assert 3 <= out["chosen_k"] <= 8
    assert out["comparison"]["rfm_grid"]["test_inactivity_auc"] > 0.75
    assert sum(r["visitors"] for r in out["cluster_profiles"]) == 1500
