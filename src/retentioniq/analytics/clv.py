"""Customer Lifetime Value (CLV) modeling using BG/NBD and a spend submodel (PROJECT_SPEC.md §4.3).

Evaluates probabilistic CLV against a held-out 90-day window on Online Retail II:
- Beta-Geometric/Negative Binomial (BG/NBD) model for expected transactions.
- Spend submodel for the expected value of a customer's future orders:
  Gamma-Gamma when its fitted parameters define a finite population mean (q > 1),
  otherwise an empirical-Bayes log-normal model (see `fit_spend_model`).
- Comparison against a historical-rate naive baseline (MAE and Spearman rank correlation).
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data
from scipy.special import hyp2f1
from scipy.stats import spearmanr

from retentioniq.analytics.customers import load_orders, rfm_scores

log = logging.getLogger(__name__)
HORIZON_DAYS = 90


def bgnbd_expected_purchases(
    params: dict[str, float],
    t: float,
    frequency: np.ndarray,
    recency: np.ndarray,
    age: np.ndarray,
) -> np.ndarray:
    """E[purchases in (T, T+t] | x, t_x, T] for BG/NBD (Fader, Hardie & Lee 2005, eq. 10).

    `lifetimes` evaluates this as exp(log(2F1(...))). When a + b + x - 1 < 0 (a valid
    region of the parameter space, reached here by one-time buyers) the hypergeometric
    term is legitimately negative, its log is NaN, and the library's fallback only
    catches inf. Evaluating the closed form directly avoids the log entirely.
    """
    r, alpha, a, b = (params[k] for k in ("r", "alpha", "a", "b"))
    x = np.asarray(frequency, dtype=float)
    tx = np.asarray(recency, dtype=float)
    big_t = np.asarray(age, dtype=float)

    hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + big_t + t))
    first = (a + b + x - 1) / (a - 1)
    second = 1 - ((alpha + big_t) / (alpha + big_t + t)) ** (r + x) * hyp
    # Inactive-probability term; (x > 0) guard because x = 0 customers cannot have dropped out.
    with np.errstate(divide="ignore", invalid="ignore"):
        dropout = np.where(
            x > 0,
            (a / (b + x - 1)) * ((alpha + big_t) / (alpha + tx)) ** (r + x),
            0.0,
        )
    expected: np.ndarray = first * second / (1 + dropout)
    if not np.all(np.isfinite(expected)):
        raise ValueError("BG/NBD produced non-finite expected purchases")
    return expected


def fit_spend_model(
    summary: pd.DataFrame, cal_orders: pd.DataFrame
) -> tuple[pd.Series, dict[str, Any]]:
    """Expected value of a future order per customer.

    Gamma-Gamma's population mean spend is v·p / (q - 1), which is finite only for q > 1.
    With q <= 1 the model says average spend is heavy-tailed beyond what it can represent:
    one-time buyers inherit a negative "population mean" and repeat buyers get weights > 1.
    In that case we fall back to an empirical-Bayes log-normal model fitted on *all*
    calibration orders (including each customer's first order, which Gamma-Gamma discards):

        log(order value) ~ N(mu_i, s2),  mu_i ~ N(mu0, tau2)
        E[order value_i] = exp(mu_post_i + s2/2 + var_post_i/2)

    The choice is made from the fitted parameters alone, never from holdout outcomes.
    """
    repeat = (summary["frequency_cal"] > 0) & (summary["monetary_value_cal"] > 0)
    info: dict[str, Any] = {"n_gamma_gamma_fit_customers": int(repeat.sum())}

    if repeat.sum() >= 10:
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(summary.loc[repeat, "frequency_cal"], summary.loc[repeat, "monetary_value_cal"])
        info["gamma_gamma_params"] = {k: round(float(v), 5) for k, v in ggf.params_.items()}
        q = float(ggf.params_["q"])
        info["gamma_gamma_valid"] = q > 1
        if q > 1:
            info["spend_model"] = "gamma_gamma"
            pop_mean = float(ggf.params_["v"] * ggf.params_["p"] / (q - 1))
            pred = ggf.conditional_expected_average_profit(
                summary["frequency_cal"], summary["monetary_value_cal"]
            )
            # One-time buyers have no repeat-order value; Gamma-Gamma gives them pop_mean.
            return pd.Series(pred, index=summary.index).fillna(pop_mean), info
        info["spend_model_reason"] = (
            f"Gamma-Gamma q={q:.3f} <= 1: population mean spend undefined; "
            "used empirical-Bayes log-normal instead"
        )
    else:
        info["gamma_gamma_valid"] = False
        info["spend_model_reason"] = "fewer than 10 repeat customers for Gamma-Gamma"

    info["spend_model"] = "empirical_bayes_lognormal"
    vals = cal_orders[cal_orders["order_value"] > 0]
    lv = pd.Series(np.log(vals["order_value"].to_numpy(dtype=float)), index=vals.index)
    grp = lv.groupby(vals["customer_id"])
    n = grp.size()
    ybar = grp.mean()
    dof = float((n - 1).sum())
    sq_dev = ((lv - grp.transform("mean")) ** 2).sum()
    s2 = float(sq_dev / dof) if dof > 0 else float(lv.var())
    mu0 = float(ybar.mean())
    tau2 = max(float(ybar.var()) - s2 * float(np.mean(1 / n)), 1e-6)
    prec = n / s2 + 1 / tau2
    mu_post = (n / s2 * ybar + mu0 / tau2) / prec
    expected = np.exp(mu_post + s2 / 2 + 1 / (2 * prec))
    info["empirical_bayes_params"] = {
        "mu0": round(mu0, 5),
        "tau2": round(tau2, 5),
        "s2": round(s2, 5),
    }
    pop = float(np.exp(mu0 + s2 / 2 + tau2 / 2))
    return expected.reindex(summary.index).fillna(pop), info


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Spearman rho, or None when either side is constant (rho is undefined)."""
    if a.nunique() < 2 or b.nunique() < 2:
        return None
    return round(float(spearmanr(a, b).statistic), 4)


def fit_evaluate_clv(
    orders: pd.DataFrame, horizon_days: int = HORIZON_DAYS
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit BG/NBD and the spend model on calibration orders; evaluate against holdout."""
    t_max = orders["order_date"].max()
    cal_end = t_max - pd.Timedelta(days=horizon_days)
    cal_orders = orders[orders["order_date"] <= cal_end]

    summary = calibration_and_holdout_data(
        orders,
        customer_id_col="customer_id",
        datetime_col="order_date",
        calibration_period_end=cal_end,
        observation_period_end=t_max,
        monetary_value_col="order_value",
    )

    # 1. BG/NBD model for transaction frequency
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency_cal"], summary["recency_cal"], summary["T_cal"])
    bg_params = {k: float(v) for k, v in bgf.params_.items()}
    summary["pred_purchases_90d"] = bgnbd_expected_purchases(
        bg_params,
        horizon_days,
        summary["frequency_cal"].to_numpy(),
        summary["recency_cal"].to_numpy(),
        summary["T_cal"].to_numpy(),
    )

    # 2. Spend model
    pred_avg_val, spend_info = fit_spend_model(summary, cal_orders)
    summary["pred_avg_order_value"] = pred_avg_val
    summary["pred_clv_90d"] = summary["pred_purchases_90d"] * summary["pred_avg_order_value"]

    # 3. Naive baseline: historical purchase rate x historical repeat-order value
    repeat = (summary["frequency_cal"] > 0) & (summary["monetary_value_cal"] > 0)
    pop_avg_spend = (
        float(summary.loc[repeat, "monetary_value_cal"].mean()) if repeat.any() else 50.0
    )
    naive_rate = summary["frequency_cal"] / summary["T_cal"].clip(lower=1.0)
    summary["naive_pred_purchases_90d"] = naive_rate * horizon_days
    hist_avg_val = summary["monetary_value_cal"].replace(0, np.nan).fillna(pop_avg_spend)
    summary["naive_pred_clv_90d"] = summary["naive_pred_purchases_90d"] * hist_avg_val

    # Actual holdout outcomes
    summary["actual_purchases_90d"] = summary["frequency_holdout"]
    summary["actual_spend_90d"] = summary["frequency_holdout"] * summary["monetary_value_holdout"]

    pred_cols = ["pred_purchases_90d", "pred_clv_90d", "naive_pred_clv_90d"]
    if not np.isfinite(summary[pred_cols].to_numpy()).all():
        raise ValueError("CLV predictions contain non-finite values")
    if (summary["pred_clv_90d"] < 0).any():
        raise ValueError("CLV predictions contain negative values")

    # 4. Evaluation metrics (inputs are finite, so no NaN can be silently skipped)
    def mae(pred: str, actual: str) -> float:
        return float((summary[pred] - summary[actual]).abs().mean())

    mae_bgnbd = mae("pred_purchases_90d", "actual_purchases_90d")
    mae_naive = mae("naive_pred_purchases_90d", "actual_purchases_90d")
    mae_reduction = ((mae_naive - mae_bgnbd) / mae_naive * 100) if mae_naive > 0 else 0.0

    metrics: dict[str, Any] = {
        "calibration_end": str(cal_end),
        "holdout_days": horizon_days,
        "n_customers": len(summary),
        "n_repeat_calibration_customers": int(repeat.sum()),
        "bgnbd_mae_purchases": round(mae_bgnbd, 4),
        "naive_mae_purchases": round(mae_naive, 4),
        "bgnbd_mae_reduction_pct": round(mae_reduction, 2),
        "clv_mae_spend": round(mae("pred_clv_90d", "actual_spend_90d"), 2),
        "naive_mae_spend": round(mae("naive_pred_clv_90d", "actual_spend_90d"), 2),
        "spearman_clv_vs_actual": _spearman(summary["pred_clv_90d"], summary["actual_spend_90d"]),
        "spearman_naive_vs_actual": _spearman(
            summary["naive_pred_clv_90d"], summary["actual_spend_90d"]
        ),
        "total_actual_holdout_revenue": round(float(summary["actual_spend_90d"].sum()), 2),
        "total_predicted_holdout_revenue": round(float(summary["pred_clv_90d"].sum()), 2),
        "total_naive_holdout_revenue": round(float(summary["naive_pred_clv_90d"].sum()), 2),
        "bgnbd_params": {k: round(v, 5) for k, v in bg_params.items()},
        **spend_info,
    }

    # 5. Segment breakdown table
    rfm = rfm_scores(cal_orders, cal_end).set_index("customer_id")
    summary_with_rfm = summary.join(rfm[["rfm_segment"]], how="left")
    summary_with_rfm["rfm_segment"] = summary_with_rfm["rfm_segment"].fillna("Other")

    seg_rows = []
    for seg, g in summary_with_rfm.groupby("rfm_segment"):
        s_mae = float((g["pred_purchases_90d"] - g["actual_purchases_90d"]).abs().mean())
        seg_rows.append(
            {
                "rfm_segment": str(seg),
                "n_customers": len(g),
                "avg_predicted_purchases_90d": round(float(g["pred_purchases_90d"].mean()), 2),
                "avg_actual_purchases_90d": round(float(g["actual_purchases_90d"].mean()), 2),
                "avg_predicted_spend_90d": round(float(g["pred_clv_90d"].mean()), 2),
                "avg_actual_spend_90d": round(float(g["actual_spend_90d"].mean()), 2),
                "total_predicted_spend_90d": round(float(g["pred_clv_90d"].sum()), 2),
                "total_actual_spend_90d": round(float(g["actual_spend_90d"].sum()), 2),
                "mae_purchases": round(s_mae, 4),
                # Empty (not 0.0) when undefined: e.g. a segment where nobody bought again.
                "spearman_rank_corr": _spearman(g["pred_clv_90d"], g["actual_spend_90d"])
                if len(g) > 2
                else None,
            }
        )
    by_segment = pd.DataFrame(seg_rows).sort_values("total_actual_spend_90d", ascending=False)
    return metrics, by_segment


def run(database_url: str, artifacts: Path) -> dict[str, Any]:
    """Train and evaluate probabilistic CLV models on warehouse marts."""
    artifacts.mkdir(parents=True, exist_ok=True)
    orders = load_orders(database_url)
    log.info("loaded %d qualifying orders for CLV modeling", len(orders))

    metrics, by_segment = fit_evaluate_clv(orders, HORIZON_DAYS)

    (artifacts / "clv_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    by_segment.to_csv(artifacts / "clv_by_segment.csv", index=False)
    log.info(
        "CLV evaluation complete: BG/NBD MAE=%.4f (naive=%.4f), spend model=%s, Spearman=%s",
        metrics["bgnbd_mae_purchases"],
        metrics["naive_mae_purchases"],
        metrics["spend_model"],
        metrics["spearman_clv_vs_actual"],
    )
    return metrics
