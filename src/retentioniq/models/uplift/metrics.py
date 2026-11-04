"""Uplift evaluation on randomized data (PROJECT_SPEC.md §6.4).

All curves rank rows by predicted uplift (descending) over the whole population ("overall"
strategy). With n_t(k), n_c(k), y_t(k), y_c(k) the treated/control counts and outcome sums
among the top k rows:

    uplift curve  u(k) = (y_t/n_t - y_c/n_c) * k            (incremental outcomes at depth k)
    Qini curve    q(k) = y_t - y_c * n_t / n_c

Areas are normalized against the random-targeting line and the perfect ranking, so a random
model scores ~0 and a perfect one 1.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Curve:
    depth: np.ndarray  # number of rows targeted, 0..n
    value: np.ndarray


def _cumulative(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    order = np.argsort(-score, kind="stable")
    y, t = np.asarray(y, dtype=float)[order], np.asarray(t, dtype=int)[order]
    n_t = np.concatenate([[0], np.cumsum(t)])
    n_c = np.concatenate([[0], np.cumsum(1 - t)])
    y_t = np.concatenate([[0], np.cumsum(y * t)])
    y_c = np.concatenate([[0], np.cumsum(y * (1 - t))])
    return n_t, n_c, y_t, y_c


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> Curve:
    n_t, n_c, y_t, y_c = _cumulative(y, t, score)
    return Curve(np.arange(len(n_t)), y_t - _safe_div(y_c * n_t, n_c))


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> Curve:
    n_t, n_c, y_t, y_c = _cumulative(y, t, score)
    depth = np.arange(len(n_t))
    return Curve(depth, (_safe_div(y_t, n_t) - _safe_div(y_c, n_c)) * depth)


def _perfect_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Treated responders first, control responders last (allows negative effects).
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=int)
    return y * t - y * (1 - t)


def _normalized_area(curve_fn, y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:  # type: ignore[no-untyped-def]
    model = curve_fn(y, t, score)
    perfect = curve_fn(y, t, _perfect_score(y, t))
    end = model.value[-1]
    random_line = np.linspace(0, end, len(model.value))
    model_area = np.trapezoid(model.value - random_line)
    perfect_area = np.trapezoid(perfect.value - random_line)
    return float(model_area / perfect_area) if perfect_area != 0 else 0.0


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Normalized area between the Qini curve and the random line (primary metric)."""
    return _normalized_area(qini_curve, y, t, score)


def auuc(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Normalized area between the uplift curve and the random line."""
    return _normalized_area(uplift_curve, y, t, score)


def top_k_counts(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, fraction: float
) -> tuple[int, int, int, int]:
    """(treated responders, treated n, control responders, control n) in the top fraction."""
    k = max(1, round(fraction * len(score)))
    n_t, n_c, y_t, y_c = _cumulative(y, t, score)
    return int(y_t[k]), int(n_t[k]), int(y_c[k]), int(n_c[k])


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, fraction: float) -> float:
    yt, nt, yc, nc = top_k_counts(y, t, score, fraction)
    if nt == 0 or nc == 0:
        return float("nan")
    return yt / nt - yc / nc


def incremental_at_k(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    fraction: float,
    population: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Estimated incremental outcomes from treating the top `fraction` of a population.

    The effect is estimated on the randomized evaluation rows; `population` (default: the
    number of evaluation rows) scales it to a campaign size. The CI resamples the four
    treated/control x outcome cells inside the top-k group (multinomial bootstrap,
    conditional on the ranking).
    """
    n = len(score)
    population = population or n
    yt, nt, yc, nc = top_k_counts(y, t, score, fraction)
    k = nt + nc
    targeted = fraction * population
    point = (yt / nt - yc / nc) * targeted if nt and nc else float("nan")

    rng = np.random.default_rng(seed)
    cells = rng.multinomial(k, np.array([yt, nt - yt, yc, nc - yc]) / k, size=n_boot)
    bt_nt = cells[:, 0] + cells[:, 1]
    bt_nc = cells[:, 2] + cells[:, 3]
    valid = (bt_nt > 0) & (bt_nc > 0)
    boot = (cells[valid, 0] / bt_nt[valid] - cells[valid, 2] / bt_nc[valid]) * targeted
    lo, hi = np.quantile(boot, [0.025, 0.975]) if valid.any() else (np.nan, np.nan)
    return {
        "fraction": fraction,
        "targeted": targeted,
        "uplift": (yt / nt - yc / nc) if nt and nc else float("nan"),
        "incremental": float(point),
        "incremental_ci_low": float(lo),
        "incremental_ci_high": float(hi),
    }


def evaluate_uplift(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.30),
) -> dict[str, float]:
    out = {"qini_coefficient": qini_coefficient(y, t, score), "auuc": auuc(y, t, score)}
    for f in fractions:
        out[f"uplift@{int(f * 100)}"] = uplift_at_k(y, t, score, f)
    return out


def curve_frame(curve: Curve, n_points: int = 101) -> pd.DataFrame:
    """Down-sample a curve to `n_points` evenly spaced depths for plotting/serving."""
    idx = np.unique(np.linspace(0, len(curve.depth) - 1, n_points).round().astype(int))
    n = curve.depth[-1]
    return pd.DataFrame({"fraction": curve.depth[idx] / n, "value": curve.value[idx]})


def paired_bootstrap_qini_diff(
    y: np.ndarray,
    t: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile CI and empirical p-value of qini_coefficient(a) - qini_coefficient(b),
    stratified by treatment arm.
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=float)
    t_arr = np.asarray(t, dtype=int)
    sa_arr = np.asarray(score_a, dtype=float)
    sb_arr = np.asarray(score_b, dtype=float)

    idx_1 = np.where(t_arr == 1)[0]
    idx_0 = np.where(t_arr == 0)[0]
    n1, n0 = len(idx_1), len(idx_0)

    qa_point = qini_coefficient(y_arr, t_arr, sa_arr)
    qb_point = qini_coefficient(y_arr, t_arr, sb_arr)
    delta_point = float(qa_point - qb_point)

    deltas = []
    for _ in range(n_boot):
        sample_1 = rng.choice(idx_1, size=n1, replace=True)
        sample_0 = rng.choice(idx_0, size=n0, replace=True)
        sample_idx = np.concatenate([sample_1, sample_0])
        qa = qini_coefficient(y_arr[sample_idx], t_arr[sample_idx], sa_arr[sample_idx])
        qb = qini_coefficient(y_arr[sample_idx], t_arr[sample_idx], sb_arr[sample_idx])
        deltas.append(qa - qb)

    deltas_arr = np.array(deltas)
    lo, hi = np.quantile(deltas_arr, [0.025, 0.975])
    p_val = 2.0 * min(float(np.mean(deltas_arr <= 0)), float(np.mean(deltas_arr >= 0)))
    p_val = min(1.0, max(0.0, p_val))
    sig = bool(lo > 0 or hi < 0)
    return {
        "delta_qini": delta_point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": p_val,
        "significant": sig,
    }


def uplift_by_score_decile(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    outcome_name: str = "visit",
) -> list[dict[str, Any]]:
    """Partition population into 10 score deciles (decile 10 = highest predicted uplift)
    and report observed treatment rate, control rate, uplift, Newcombe CI, and predicted uplift.
    """
    from retentioniq.analytics.experiments.inference import compare_arms

    # Rank ascending so decile 0 is lowest scores (decile 1) and decile 9 is highest (decile 10)
    ranks = pd.Series(score).rank(method="first", ascending=True)
    deciles = pd.qcut(ranks, 10, labels=False)
    y_arr = np.asarray(y)
    t_arr = np.asarray(t)
    score_arr = np.asarray(score, dtype=float)

    rows = []
    for d in range(10):
        mask = (deciles == d).to_numpy()
        cmp = compare_arms(y_arr[mask], t_arr[mask], outcome=outcome_name)
        rows.append(
            {
                "decile": d + 1,
                "n": int(mask.sum()),
                "rate_control": cmp.rate_control,
                "rate_treatment": cmp.rate_treatment,
                "observed_uplift": cmp.ate,
                "ci_low": cmp.ate_ci_newcombe[0],
                "ci_high": cmp.ate_ci_newcombe[1],
                "predicted_uplift": float(score_arr[mask].mean()) if mask.any() else 0.0,
            }
        )
    return rows
