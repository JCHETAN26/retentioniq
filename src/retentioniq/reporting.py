"""Publish evaluation outputs into Postgres `reporting.*` for BI tools (Power BI, Phase 10).

BI connects to one database: dbt marts (star schema) + these model-output tables. Tables
are replaced on every publish, so they always match the current artifacts.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

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
}
RISK_TIERS = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))


def _tier(p: float) -> str:
    return next(name for threshold, name in RISK_TIERS if p >= threshold)


def publish(database_url: str, artifacts: Path) -> dict[str, int]:
    engine = create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS reporting"))
    written: dict[str, int] = {}
    for file, table in TABLES.items():
        path = artifacts / file
        if path.exists():
            df = pd.read_csv(path)
            df.to_sql(
                table,
                engine,
                schema="reporting",
                if_exists="replace",
                index=False,
                chunksize=5000,
                method="multi",
            )
            written[table] = len(df)
    preds = artifacts / "churn_test_predictions.parquet"
    if preds.exists():
        df = pd.read_parquet(preds)
        keep = ["visitor_id", "cutoff", "p_inactive", "inactive", "top_category_id", *ALL_FEATURES]
        df = df[keep].assign(risk_tier=df["p_inactive"].map(_tier))
        df.to_sql(
            "churn_risk_scores",
            engine,
            schema="reporting",
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )
        written["churn_risk_scores"] = len(df)
    return written
