"""Inactivity-prediction metrics (PROJECT_SPEC.md §5.4) and bootstrap CIs (§7.2)."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

K_FRACTIONS = (0.05, 0.10, 0.20)

Metric = Callable[[np.ndarray, np.ndarray], float]


def _top_k(score: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, round(fraction * len(score)))
    return np.argsort(-score, kind="stable")[:k]


def precision_at_k(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    return float(np.asarray(y)[_top_k(score, fraction)].mean())


def recall_at_k(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    y = np.asarray(y)
    return float(y[_top_k(score, fraction)].sum() / max(1, y.sum()))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-mass bins."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    n = len(p)
    return float(sum(len(b) / n * abs(np.mean(y[b]) - np.mean(p[b])) for b in bins if len(b)))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(p, kind="stable")
    return [
        {
            "mean_predicted": float(np.mean(p[b])),
            "observed_rate": float(np.mean(y[b])),
            "n": len(b),
        }
        for b in np.array_split(order, n_bins)
        if len(b)
    ]


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing F1 (chosen on validation, applied to test)."""
    order = np.argsort(-p, kind="stable")
    ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    tp = np.cumsum(ys)
    precision = tp / np.arange(1, len(ys) + 1)
    recall = tp / max(1, ys.sum())
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    return float(ps[int(np.argmax(f1))])


def f1_at(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    pred = np.asarray(p) >= threshold
    tp = float((pred & (np.asarray(y) == 1)).sum())
    denom = pred.sum() + np.asarray(y).sum()
    return 2 * tp / denom if denom else 0.0


def classification_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float | None = None
) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    base = float(y.mean())
    y_ret = (y == 0).astype(int)
    p_ret = 1.0 - p
    ret_base = float(y_ret.mean())
    pr_auc_ret = float(average_precision_score(y_ret, p_ret)) if 0 < ret_base < 1 else float("nan")
    pk_ret = precision_at_k(y_ret, p_ret, 0.10)
    lift_ret = pk_ret / ret_base if ret_base > 0 else float("nan")

    out = {
        "n": float(len(y)),
        "base_rate": base,
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < base < 1 else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "ece": expected_calibration_error(y, np.clip(p, 0, 1)),
        "returner_base_rate": ret_base,
        "pr_auc_returners": pr_auc_ret,
        "precision_returners@10": pk_ret,
        "lift_returners@10": lift_ret,
    }
    for f in K_FRACTIONS:
        pk = precision_at_k(y, p, f)
        out[f"precision@{int(f * 100)}"] = pk
        out[f"recall@{int(f * 100)}"] = recall_at_k(y, p, f)
        out[f"lift@{int(f * 100)}"] = pk / base if base else float("nan")
    if threshold is not None:
        out["f1"] = f1_at(y, p, threshold)
        out["threshold"] = threshold
    return out


def cost_optimal_threshold(
    y: np.ndarray,
    p: np.ndarray,
    cost_contact: float,
    value_saved: float,
    save_rate: float,
) -> float:
    """Find operating threshold maximizing expected profit using observed labels:
    sum(y_i * save_rate * value_saved) - n_contacted * cost_contact.
    """
    y_arr = np.asarray(y)
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(-p_arr, kind="stable")
    ys = y_arr[order]
    ps = p_arr[order]
    cum_saved = np.cumsum(ys) * save_rate * value_saved
    cum_cost = np.arange(1, len(ys) + 1) * cost_contact
    profits = cum_saved - cum_cost
    best_idx = int(np.argmax(profits))
    if profits[best_idx] <= 0:
        return 1.0
    return float(ps[best_idx])


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Metric,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI, resampling clusters (e.g. visitor_id) when `groups` is given."""
    return paired_bootstrap_diff(y, p, None, metric, groups, n_boot, seed)


def paired_bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray | None,
    metric: Metric,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """CI of metric(p_a) - metric(p_b) on the same resamples (or of metric(p_a) if p_b is None).

    Clustering by `visitor_id` (via `groups`) matters when a visitor appears more than once
    (such as across training snapshots). On a single snapshot (e.g. the test set), each
    visitor appears at most once, making this equivalent to an ordinary i.i.d. bootstrap.
    """
    rng = np.random.default_rng(seed)
    y, p_a = np.asarray(y), np.asarray(p_a)
    if groups is None:
        groups = np.arange(len(y))
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[members], np.arange(len(uniq)))
    sizes = np.diff(np.append(starts, len(members)))
    stats = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(uniq), len(uniq))
        idx = (
            np.concatenate([members[starts[c] : starts[c] + sizes[c]] for c in chosen])
            if (sizes > 1).any()
            else members[starts[chosen]]
        )
        if y[idx].min() == y[idx].max():
            continue
        val = metric(y[idx], p_a[idx])
        if p_b is not None:
            val -= metric(y[idx], np.asarray(p_b)[idx])
        stats.append(val)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi)
