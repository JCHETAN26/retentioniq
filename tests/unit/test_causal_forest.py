import numpy as np
import pandas as pd
import pytest

from retentioniq.models.uplift.learners import FEATURES, CausalForest
from retentioniq.models.uplift.metrics import qini_coefficient

pytest.importorskip("econml")


def test_causal_forest_ranks_responders_first() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    t = rng.integers(0, 2, n)
    tau = np.where(x["f0"] > 0.5, 0.15, 0.0)
    y = (rng.random(n) < 0.1 + tau * t).astype(int)
    model = CausalForest(max_rows=15_000).fit(x, t, y)
    score = model.score(x)
    assert score[x["f0"] > 0.5].mean() > score[x["f0"] <= 0.5].mean() + 0.05
    assert qini_coefficient(y, t, score) > 0.1
