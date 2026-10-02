"""Randomized-experiment inference for binary outcomes (PROJECT_SPEC.md §6.2)."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

Z95 = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class ArmComparison:
    outcome: str
    n_treatment: int
    n_control: int
    rate_treatment: float
    rate_control: float
    ate: float
    ate_ci_newcombe: tuple[float, float]
    ate_ci_wald: tuple[float, float]
    relative_lift: float
    relative_lift_ci: tuple[float, float]
    z: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def compare_arms(y: np.ndarray, t: np.ndarray, outcome: str) -> ArmComparison:
    """ATE = P(Y=1|T=1) - P(Y=1|T=0) with Newcombe (score) and Wald CIs, and a z-test."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=int)
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    x1, x0 = int(y[t == 1].sum()), int(y[t == 0].sum())
    p1, p0 = x1 / n1, x0 / n0
    diff = p1 - p0

    # Newcombe (1998) method 10: hybrid score interval for a difference of proportions.
    l1, u1 = wilson_interval(x1, n1)
    l0, u0 = wilson_interval(x0, n0)
    newcombe = (
        diff - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2),
        diff + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2),
    )
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    wald = (diff - Z95 * se_unpooled, diff + Z95 * se_unpooled)

    pooled = (x1 + x0) / (n1 + n0)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    # Relative lift CI via the delta method on log(p1 / p0).
    rel = p1 / p0 - 1 if p0 > 0 else float("nan")
    if x1 > 0 and x0 > 0:
        se_log = np.sqrt((1 - p1) / x1 + (1 - p0) / x0)
        log_rr = np.log(p1 / p0)
        rel_ci = (np.exp(log_rr - Z95 * se_log) - 1, np.exp(log_rr + Z95 * se_log) - 1)
    else:
        rel_ci = (float("nan"), float("nan"))

    return ArmComparison(
        outcome=outcome,
        n_treatment=n1,
        n_control=n0,
        rate_treatment=p1,
        rate_control=p0,
        ate=diff,
        ate_ci_newcombe=(float(newcombe[0]), float(newcombe[1])),
        ate_ci_wald=(float(wald[0]), float(wald[1])),
        relative_lift=float(rel),
        relative_lift_ci=(float(rel_ci[0]), float(rel_ci[1])),
        z=float(z),
        p_value=p_value,
    )


def bootstrap_ate_ci(
    y: np.ndarray, t: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the ATE, resampling within each arm (stratified)."""
    rng = np.random.default_rng(seed)
    y1, y0 = np.asarray(y)[np.asarray(t) == 1], np.asarray(y)[np.asarray(t) == 0]
    # Binary outcomes: resampled means are binomial draws, which is exact and memory-light.
    m1 = rng.binomial(len(y1), y1.mean(), n_boot) / len(y1)
    m0 = rng.binomial(len(y0), y0.mean(), n_boot) / len(y0)
    lo, hi = np.quantile(m1 - m0, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted


def standardized_mean_differences(x: pd.DataFrame, t: np.ndarray) -> pd.Series:
    """SMD per covariate: (mean_T - mean_C) / sqrt((var_T + var_C) / 2)."""
    t = np.asarray(t)
    xt, xc = x[t == 1], x[t == 0]
    pooled = np.sqrt((xt.var() + xc.var()) / 2).replace(0, np.nan)
    return ((xt.mean() - xc.mean()) / pooled).fillna(0.0)


def sample_ratio_test(t: np.ndarray, expected_treatment_share: float) -> dict[str, float]:
    """Chi-square test for sample-ratio mismatch against the designed allocation."""
    t = np.asarray(t)
    n, n1 = len(t), int((t == 1).sum())
    expected = np.array([n * expected_treatment_share, n * (1 - expected_treatment_share)])
    chi2, p = stats.chisquare([n1, n - n1], expected)
    return {
        "observed_share": n1 / n,
        "expected_share": expected_treatment_share,
        "chi2": float(chi2),
        "p_value": float(p),
    }
