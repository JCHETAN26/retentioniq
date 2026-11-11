"""Single-record scoring path and its latency budget (PROJECT_SPEC.md §5.5, §9.1, §9.2).

The API's `/v1/churn/predict` scores one record with `score_one`, and model selection and
promotion measure latency with that same function, so the number that gates a model is the
number of the code path that serves it.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

# 1 s / 200 req/s: the §9.2 per-task throughput SLO leaves 5 ms of scoring per request on one
# vCPU, before HTTP overhead. A model whose in-process p95 exceeds this cannot meet the SLO.
LATENCY_BUDGET_MS = 5.0


def score_one(
    model: Any, calibrator: Callable[[np.ndarray], np.ndarray], features: Sequence[str], values: Any
) -> float:
    """P(inactive) for one record (a mapping of feature -> value).

    LightGBM models go straight to the booster with a NumPy row, skipping the per-request
    DataFrame and the scikit-learn wrapper; every other model uses `predict_proba` on a
    one-row DataFrame.
    """
    booster = getattr(getattr(model, "model_", None), "booster_", None)
    if booster is not None:
        row = np.array([[values[f] for f in features]], dtype=np.float64)
        p = booster.predict(row, num_threads=1)
    else:
        p = model.predict_proba(pd.DataFrame([{f: values[f] for f in features}]))
    return float(np.clip(np.asarray(calibrator(np.asarray(p)), dtype=float), 0, 1)[0])


def measure_latency_ms(
    model: Any,
    calibrator: Callable[[np.ndarray], np.ndarray],
    features: Sequence[str],
    rows: pd.DataFrame,
    n: int = 200,
    warmup: int = 20,
) -> dict[str, float]:
    """In-process single-record latency of `score_one` over the first `n` rows (p50/p95, ms)."""
    records = rows[list(features)].head(n + warmup).to_dict("records")
    if len(records) <= warmup:
        raise ValueError("not enough rows to measure latency")
    for rec in records[:warmup]:
        score_one(model, calibrator, features, rec)
    times = []
    for rec in records[warmup:]:
        t0 = time.perf_counter()
        score_one(model, calibrator, features, rec)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "n": len(arr),
    }
