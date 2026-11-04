"""Unit tests for Customer Lifetime Value (CLV) probabilistic modeling."""

import numpy as np
import pandas as pd
import pytest

from retentioniq.analytics.clv import fit_evaluate_clv


@pytest.fixture
def synthetic_orders() -> pd.DataFrame:
    """Generate 600 orders across 60 customers spanning 365 days."""
    rng = np.random.default_rng(42)
    n_orders = 600
    n_cust = 60
    cust_ids = rng.choice(np.arange(1001, 1001 + n_cust), size=n_orders)
    # Order dates uniformly distributed across 2024
    start_ts = pd.Timestamp("2024-01-01").value
    end_ts = pd.Timestamp("2024-12-31").value
    random_ts = rng.integers(start_ts, end_ts, size=n_orders)
    order_dates = pd.to_datetime(random_ts)

    # Order values: log-normal distribution around $40
    order_values = np.round(rng.lognormal(mean=3.5, sigma=0.5, size=n_orders), 2)
    invoices = [f"INV-{i:05d}" for i in range(1, n_orders + 1)]

    df = pd.DataFrame(
        {
            "invoice": invoices,
            "customer_id": cust_ids,
            "order_date": order_dates,
            "order_value": order_values,
        }
    )
    return df.sort_values("order_date").reset_index(drop=True)


def test_fit_evaluate_clv_metrics(synthetic_orders: pd.DataFrame) -> None:
    metrics, by_segment = fit_evaluate_clv(synthetic_orders, horizon_days=90)

    # Validate metrics structure
    assert "bgnbd_mae_purchases" in metrics
    assert "naive_mae_purchases" in metrics
    assert "spearman_clv_vs_actual" in metrics
    assert "spearman_naive_vs_actual" in metrics
    assert "n_customers" in metrics
    assert "total_predicted_holdout_revenue" in metrics
    assert "total_actual_holdout_revenue" in metrics

    assert metrics["n_customers"] > 0
    assert metrics["bgnbd_mae_purchases"] >= 0.0
    assert metrics["naive_mae_purchases"] >= 0.0
    assert np.isfinite(metrics["spearman_clv_vs_actual"])

    # Validate segment table structure
    expected_cols = {
        "rfm_segment",
        "n_customers",
        "avg_predicted_purchases_90d",
        "avg_actual_purchases_90d",
        "avg_predicted_spend_90d",
        "avg_actual_spend_90d",
        "total_predicted_spend_90d",
        "total_actual_spend_90d",
        "mae_purchases",
        "spearman_rank_corr",
    }
    assert expected_cols.issubset(by_segment.columns)
    assert len(by_segment) > 0
    assert (by_segment["n_customers"] > 0).all()
    assert (by_segment["total_predicted_spend_90d"] >= 0).all()
