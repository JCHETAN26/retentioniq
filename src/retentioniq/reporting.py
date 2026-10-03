"""Publish evaluation outputs into Postgres `reporting.*` for BI tools (Power BI, Phase 10).

BI connects to one database: dbt marts (star schema) + these model-output tables. Tables
are replaced on every publish, so they always match the current artifacts.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from retentioniq.features.retailrocket import ALL_FEATURES

TABLES: dict[str, str] = {
    "churn_model_comparison.csv": "churn_model_comparison",
    "feature_ablation.csv": "churn_feature_ablation",
    "churn_segment_metrics.csv": "churn_segment_metrics",
    "shap_global.csv": "churn_shap_global",
    "customer_segments.csv": "customer_segments",
    "cohort_metrics.csv": "cohort_metrics",
    "survival_analysis.csv": "survival_km",
    "survival_cox_coefficients.csv": "survival_cox",
    "uplift_results.csv": "uplift_results",
    "uplift_curves.csv": "uplift_curves",
    "uplift_heterogeneity.csv": "uplift_heterogeneity",
    "campaign_simulation.csv": "campaign_simulation",
    "visitor_category_cohorts.csv": "visitor_category_cohorts",
}
RISK_TIERS = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))


def _tier(p: float) -> str:
    return next(name for threshold, name in RISK_TIERS if p >= threshold)


MAX_PARAMS = 60_000  # Postgres caps one statement at 65,535 bind parameters


def _write(df: pd.DataFrame, table: str, engine: Engine) -> None:
    df.to_sql(
        table,
        engine,
        schema="reporting",
        if_exists="replace",
        index=False,
        chunksize=max(1, MAX_PARAMS // max(1, len(df.columns))),
        method="multi",
    )


def publish(database_url: str, artifacts: Path) -> dict[str, int]:
    engine = create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS reporting"))
    written: dict[str, int] = {}
    for file, table in TABLES.items():
        path = artifacts / file
        if path.exists():
            df = pd.read_csv(path)
            _write(df, table, engine)
            written[table] = len(df)
    preds = artifacts / "churn_test_predictions.parquet"
    if preds.exists():
        df = pd.read_parquet(preds)
        keep = ["visitor_id", "cutoff", "p_inactive", "inactive", "top_category_id", *ALL_FEATURES]
        df = df[keep].assign(risk_tier=df["p_inactive"].map(_tier))
        _write(df, "churn_risk_scores", engine)
        written["churn_risk_scores"] = len(df)
    return written


# Power BI service (browser) cannot reach a private Postgres, so the star schema is exported
# to one workbook: each sheet becomes a table in a single semantic model. Line-level and
# visitor-day facts exceed Excel's row limit and are exported as daily aggregates.
POWERBI_QUERIES: dict[str, str] = {
    "dim_date": "SELECT * FROM marts.dim_date",
    "dim_customer": "SELECT * FROM marts.dim_customer",
    "dim_product": "SELECT stock_code, description, is_non_product FROM marts.dim_product",
    "fact_orders": "SELECT * FROM marts.fact_orders",
    "fact_sales_daily": """
        SELECT invoice_date AS date_day, country,
               count(*) AS lines,
               count(*) FILTER (WHERE is_cancellation) AS cancelled_lines,
               coalesce(sum(line_amount) FILTER (WHERE is_qualifying), 0) AS qualifying_value,
               coalesce(-sum(line_amount) FILTER (WHERE is_cancellation), 0) AS cancelled_value,
               coalesce(sum(line_amount) FILTER (WHERE NOT has_customer AND line_amount > 0), 0)
                   AS no_customer_value
        FROM marts.fact_transactions GROUP BY 1, 2""",
    "rfm": "SELECT * FROM marts.mart_customer_rfm",
    "cohort_retention": "SELECT * FROM marts.mart_customer_retention",
    "rr_funnel_daily": "SELECT * FROM marts.mart_engagement_funnel",
    "rr_activity_daily": """
        SELECT activity_date AS date_day, count(DISTINCT visitor_id) AS visitors,
               sum(n_events) AS events, sum(n_views) AS views, sum(n_carts) AS carts,
               sum(n_transactions) AS transactions, sum(n_sessions) AS sessions
        FROM marts.mart_customer_daily_activity GROUP BY 1""",
    "risk_scores": """
        SELECT visitor_id, cutoff, p_inactive, inactive, risk_tier, top_category_id,
               recency_days, sessions_obs, views_30d, transactions_obs, tenure_days,
               activity_trend
        FROM reporting.churn_risk_scores""",
    "model_comparison": "SELECT * FROM reporting.churn_model_comparison",
    "shap_global": "SELECT * FROM reporting.churn_shap_global",
    "customer_segments": "SELECT * FROM reporting.customer_segments",
    "segment_retention": "SELECT * FROM reporting.cohort_metrics WHERE group_type <> 'cohort'",
    "survival_km": "SELECT * FROM reporting.survival_km",
    "campaign_simulation": "SELECT * FROM reporting.campaign_simulation",
    "uplift_results": "SELECT * FROM reporting.uplift_results WHERE split = 'test'",
}


def export_powerbi(database_url: str, out_path: Path) -> dict[str, int]:
    engine = create_engine(database_url)
    sizes: dict[str, int] = {}
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet, query in POWERBI_QUERIES.items():
            df = pd.read_sql(text(query), engine)
            for col in df.select_dtypes(include=["datetimetz"]).columns:
                df[col] = df[col].dt.tz_localize(None)
            df.to_excel(writer, sheet_name=sheet, index=False)
            sizes[sheet] = len(df)
    return sizes
