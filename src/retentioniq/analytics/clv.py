"""Customer Lifetime Value (CLV) modeling using BG/NBD and Gamma-Gamma (PROJECT_SPEC.md §4.3).

Evaluates probabilistic CLV against a held-out 90-day window on Online Retail II:
- Beta-Geometric/Negative Binomial (BG/NBD) model for expected transactions.
- Gamma-Gamma submodel for expected customer monetary value.
- Comparison against historical-rate naive baseline (MAE and Spearman rank correlation).
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data
from scipy.stats import spearmanr

from retentioniq.analytics.customers import load_orders, rfm_scores

log = logging.getLogger(__name__)
HORIZON_DAYS = 90


def fit_evaluate_clv(
    orders: pd.DataFrame, horizon_days: int = HORIZON_DAYS
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit BG/NBD and Gamma-Gamma on calibration orders; evaluate against holdout."""
    t_max = orders["order_date"].max()
    cal_end = t_max - pd.Timedelta(days=horizon_days)

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
    pred_tx = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, summary["frequency_cal"], summary["recency_cal"], summary["T_cal"]
    )
    summary["pred_purchases_90d"] = pred_tx

    # 2. Gamma-Gamma model for monetary value
    repeat_mask = (summary["frequency_cal"] > 0) & (summary["monetary_value_cal"] > 0)
    rep_summary = summary[repeat_mask]
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    if len(rep_summary) >= 10:
        ggf.fit(rep_summary["frequency_cal"], rep_summary["monetary_value_cal"])
        pred_avg_val = ggf.conditional_expected_average_profit(
            summary["frequency_cal"], summary["monetary_value_cal"]
        )
    else:
        # Fallback for small synthetic datasets
        mean_val = (
            float(rep_summary["monetary_value_cal"].mean()) if not rep_summary.empty else 50.0
        )
        pred_avg_val = pd.Series(mean_val, index=summary.index)

    # For zero-frequency calibration customers, fill with population average
    pop_avg_spend = (
        float(rep_summary["monetary_value_cal"].mean()) if not rep_summary.empty else 50.0
    )
    pred_avg_val = pred_avg_val.fillna(pop_avg_spend)
    summary["pred_avg_order_value"] = pred_avg_val
    summary["pred_clv_90d"] = summary["pred_purchases_90d"] * summary["pred_avg_order_value"]

    # 3. Naive baseline: historical purchase rate in calibration
    naive_rate = summary["frequency_cal"] / summary["T_cal"].clip(lower=1.0)
    summary["naive_pred_purchases_90d"] = naive_rate * horizon_days
    hist_avg_val = summary["monetary_value_cal"].replace(0, np.nan).fillna(pop_avg_spend)
    summary["naive_pred_clv_90d"] = summary["naive_pred_purchases_90d"] * hist_avg_val

    # Actual holdout outcomes
    summary["actual_purchases_90d"] = summary["frequency_holdout"]
    summary["actual_spend_90d"] = summary["frequency_holdout"] * summary["monetary_value_holdout"]

    # 4. Evaluation metrics
    mae_bgnbd = float(
        np.mean(np.abs(summary["pred_purchases_90d"] - summary["actual_purchases_90d"]))
    )
    mae_naive = float(
        np.mean(np.abs(summary["naive_pred_purchases_90d"] - summary["actual_purchases_90d"]))
    )
    mae_reduction = ((mae_naive - mae_bgnbd) / mae_naive * 100) if mae_naive > 0 else 0.0

    spearman_clv = float(spearmanr(summary["pred_clv_90d"], summary["actual_spend_90d"]).statistic)
    spearman_naive = float(
        spearmanr(summary["naive_pred_clv_90d"], summary["actual_spend_90d"]).statistic
    )

    metrics = {
        "calibration_end": str(cal_end),
        "holdout_days": horizon_days,
        "n_customers": len(summary),
        "n_repeat_calibration_customers": int(repeat_mask.sum()),
        "bgnbd_mae_purchases": round(mae_bgnbd, 4),
        "naive_mae_purchases": round(mae_naive, 4),
        "bgnbd_mae_reduction_pct": round(mae_reduction, 2),
        "spearman_clv_vs_actual": round(spearman_clv, 4),
        "spearman_naive_vs_actual": round(spearman_naive, 4),
        "total_actual_holdout_revenue": round(float(summary["actual_spend_90d"].sum()), 2),
        "total_predicted_holdout_revenue": round(float(summary["pred_clv_90d"].sum()), 2),
        "bgnbd_params": {k: round(float(v), 5) for k, v in bgf.params_.items()}
        if hasattr(bgf, "params_")
        else {},
        "gamma_gamma_params": {k: round(float(v), 5) for k, v in ggf.params_.items()}
        if hasattr(ggf, "params_")
        else {},
    }

    # 5. Segment breakdown table
    cal_orders = orders[orders["order_date"] <= cal_end]
    rfm = rfm_scores(cal_orders, cal_end).set_index("customer_id")
    summary_with_rfm = summary.join(rfm[["rfm_segment"]], how="left")
    summary_with_rfm["rfm_segment"] = summary_with_rfm["rfm_segment"].fillna("Other")

    seg_rows = []
    for seg, g in summary_with_rfm.groupby("rfm_segment"):
        s_mae = float(np.mean(np.abs(g["pred_purchases_90d"] - g["actual_purchases_90d"])))
        s_corr = (
            float(spearmanr(g["pred_clv_90d"], g["actual_spend_90d"]).statistic)
            if len(g) > 2
            else 0.0
        )
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
                "spearman_rank_corr": round(s_corr, 4),
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
        "CLV evaluation complete: BG/NBD MAE=%.4f (naive=%.4f), Spearman=%.4f",
        metrics["bgnbd_mae_purchases"],
        metrics["naive_mae_purchases"],
        metrics["spearman_clv_vs_actual"],
    )
    return metrics
