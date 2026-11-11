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


@pytest.mark.filterwarnings("ignore:invalid value encountered in log:RuntimeWarning")
def test_bgnbd_matches_lifetimes_and_handles_negative_hypergeometric() -> None:
    """Closed form equals lifetimes where it is finite, and stays finite where it is NaN."""
    from lifetimes import BetaGeoFitter

    from retentioniq.analytics.clv import bgnbd_expected_purchases

    # Parameters from the real Online Retail II fit: a + b - 1 < 0, so for x = 0 the
    # hypergeometric term is negative and lifetimes' log() turns it into NaN.
    params = {"r": 0.63094, "alpha": 62.21695, "a": 0.06982, "b": 0.66963}
    x = np.array([0.0, 0.0, 0.0, 1.0, 3.0, 10.0])
    tx = np.array([0.0, 0.0, 0.0, 40.0, 200.0, 300.0])
    age = np.array([5.0, 120.0, 400.0, 100.0, 300.0, 310.0])

    ours = bgnbd_expected_purchases(params, 90, x, tx, age)
    assert np.isfinite(ours).all()
    assert (ours >= 0).all()

    bgf = BetaGeoFitter()
    bgf.params_ = pd.Series(params)
    ref = np.asarray(bgf.conditional_expected_number_of_purchases_up_to_time(90, x, tx, age))
    finite = np.isfinite(ref)
    assert not finite.all(), "fixture should exercise the lifetimes NaN case"
    np.testing.assert_allclose(ours[finite], ref[finite], rtol=1e-9)


def test_spend_model_falls_back_when_gamma_gamma_is_invalid() -> None:
    """Heavy-tailed spend gives Gamma-Gamma q <= 1; the fallback must be positive and finite."""
    from lifetimes.utils import summary_data_from_transaction_data

    from retentioniq.analytics.clv import fit_spend_model

    rng = np.random.default_rng(7)
    rows = []
    for cust in range(400):
        scale = rng.pareto(0.8) * 20 + 5  # customer-level spend with an infinite-mean tail
        for k in range(int(rng.integers(1, 6))):
            rows.append((cust, pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * k), scale))
    orders = pd.DataFrame(rows, columns=["customer_id", "order_date", "order_value"])
    summary = summary_data_from_transaction_data(
        orders, "customer_id", "order_date", monetary_value_col="order_value"
    ).add_suffix("_cal")

    pred, info = fit_spend_model(summary, orders)
    assert info["spend_model"] == "empirical_bayes_lognormal"
    assert info["gamma_gamma_valid"] is False
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_fit_evaluate_clv_predictions_are_finite_and_non_negative(
    synthetic_orders: pd.DataFrame,
) -> None:
    metrics, by_segment = fit_evaluate_clv(synthetic_orders, horizon_days=90)
    assert metrics["spend_model"] in {"gamma_gamma", "empirical_bayes_lognormal"}
    assert np.isfinite(metrics["clv_mae_spend"])
    assert by_segment["avg_predicted_spend_90d"].ge(0).all()
