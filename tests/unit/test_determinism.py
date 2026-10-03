import numpy as np
import pandas as pd

from retentioniq.models.uplift.learners import FEATURES, SLearner


def test_lightgbm_learners_are_bit_for_bit_reproducible() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.05 + 0.05 * t * (x["f0"] > 0)).astype(int)
    a = SLearner().fit(x, t, y).score(x)
    b = SLearner().fit(x, t, y).score(x)
    assert np.array_equal(a, b)
