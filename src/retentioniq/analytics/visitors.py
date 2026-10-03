"""RetailRocket visitor segmentation and category cohorts (Phases 8-9, dataset A).

Segmentation: K-Means on behavioural features vs an RFM-style recency x frequency grid.
Both are fit on the training snapshots and judged on the test snapshot by how well segment
membership alone predicts 30-day inactivity (the same honest comparison as Online Retail).

Category cohorts: weekly first-visit cohorts by the top-level category of the first item
viewed, with week-k retention = share of the cohort with any session k weeks later.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from retentioniq import SEED

FEATURES = [
    "recency_days",
    "sessions_obs",
    "views_30d",
    "cart_rate",
    "conversion_rate",
    "activity_trend",
    "mean_days_between_sessions",
    "category_entropy",
    "unique_items_obs",
    "tenure_days",
]
LOG = {
    "recency_days",
    "sessions_obs",
    "views_30d",
    "mean_days_between_sessions",
    "unique_items_obs",
    "tenure_days",
}
WEEKS = 8


def _matrix(df: pd.DataFrame) -> np.ndarray:
    z = df[FEATURES].copy()
    for c in LOG:
        z[c] = np.log1p(z[c].clip(lower=0))
    return z.to_numpy(dtype=float)


def _segment_auc(
    train_seg: pd.Series, y_train: np.ndarray, test_seg: pd.Series, y_test: np.ndarray
) -> float:
    """AUC on test of the per-segment inactivity rate learned on train."""
    rates = pd.Series(y_train).groupby(train_seg.to_numpy()).mean()
    p = test_seg.map(rates).fillna(float(np.mean(y_train))).to_numpy()
    return float(roc_auc_score(y_test, p))


def segmentation(snap: pd.DataFrame) -> dict[str, Any]:
    train = snap[snap["split"] == "train"].reset_index(drop=True)
    test = snap[snap["split"] == "test"].reset_index(drop=True)
    y_tr, y_te = train["inactive"].to_numpy(), test["inactive"].to_numpy()

    # RFM-style grid: recency and frequency quintiles with edges learned on train.
    def grid(df: pd.DataFrame) -> pd.Series:
        r = np.searchsorted(r_edges, df["recency_days"], side="right")
        f = np.searchsorted(f_edges, df["sessions_obs"], side="right")
        return pd.Series([f"R{a}F{b}" for a, b in zip(r, f, strict=True)])

    r_edges = np.unique(np.quantile(train["recency_days"], [0.2, 0.4, 0.6, 0.8]))
    f_edges = np.unique(np.quantile(train["sessions_obs"], [0.2, 0.4, 0.6, 0.8]))

    scaler = StandardScaler().fit(_matrix(train))
    x_tr, x_te = scaler.transform(_matrix(train)), scaler.transform(_matrix(test))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x_tr), size=min(10_000, len(x_tr)), replace=False)
    search = []
    for k in range(3, 9):
        km = KMeans(k, n_init=5, random_state=SEED).fit(x_tr)
        search.append(
            {"k": k, "silhouette": float(silhouette_score(x_tr[sample], km.labels_[sample]))}
        )
    k_best = max(search, key=lambda r: r["silhouette"])["k"]
    km = KMeans(k_best, n_init=10, random_state=SEED).fit(x_tr)
    tr_cl, te_cl = pd.Series(km.labels_), pd.Series(km.predict(x_te))

    profile = (
        test.assign(cluster=te_cl.to_numpy())
        .groupby("cluster")
        .agg(
            visitors=("visitor_id", "size"),
            inactive_rate=("inactive", "mean"),
            median_recency_days=("recency_days", "median"),
            median_sessions=("sessions_obs", "median"),
            purchaser_share=("ever_purchased", "mean"),
            median_views_30d=("views_30d", "median"),
        )
        .reset_index()
    )
    return {
        "dataset": "retailrocket",
        "fit_on": "train snapshots",
        "evaluated_on": "test snapshot",
        "k_search": search,
        "chosen_k": int(k_best),
        "comparison": {
            "rfm_grid": {
                "n_groups": int(grid(train).nunique()),
                "test_inactivity_auc": _segment_auc(grid(train), y_tr, grid(test), y_te),
            },
            "kmeans_behavioural": {
                "n_groups": int(k_best),
                "test_inactivity_auc": _segment_auc(tr_cl, y_tr, te_cl, y_te),
            },
        },
        "cluster_profiles": profile.to_dict(orient="records"),
    }


CATEGORY_COHORTS_SQL = """
WITH RECURSIVE roots AS (
    SELECT category_id, category_id AS root_id, parent_category_id
    FROM staging.stg_retailrocket__category_tree
    UNION ALL
    SELECT r.category_id, t.category_id, t.parent_category_id
    FROM roots r
    JOIN staging.stg_retailrocket__category_tree t ON t.category_id = r.parent_category_id
),
root AS (
    SELECT category_id, root_id FROM roots WHERE parent_category_id IS NULL
),
first_event AS (
    SELECT DISTINCT ON (visitor_id) visitor_id, event_ts, category_id
    FROM marts.fact_events
    ORDER BY visitor_id, event_ts, event_id
),
cohort AS (
    SELECT f.visitor_id,
           date_trunc('week', f.event_ts)::date AS cohort_week,
           COALESCE(r.root_id::text, 'unknown') AS root_category
    FROM first_event f LEFT JOIN root r USING (category_id)
)
SELECT c.cohort_week, c.root_category,
       ((date_trunc('week', a.activity_date)::date - c.cohort_week) / 7) AS week_number,
       count(DISTINCT c.visitor_id) AS active_visitors
FROM cohort c
JOIN marts.mart_customer_daily_activity a ON a.visitor_id = c.visitor_id
GROUP BY 1, 2, 3
"""


def category_cohorts(database_url: str) -> pd.DataFrame:
    engine = create_engine(database_url)
    act = pd.read_sql(CATEGORY_COHORTS_SQL, engine, parse_dates=["cohort_week"])
    last_week = act["cohort_week"].max()
    # Only cohorts with all WEEKS observable contribute (no right-censored cells).
    act = act[act["cohort_week"] <= last_week - pd.Timedelta(weeks=WEEKS)]
    sizes = act[act["week_number"] == 0].groupby("root_category")["active_visitors"].sum()
    top = sizes.sort_values(ascending=False).head(6).index
    act["group"] = np.where(
        act["root_category"].isin(top), "category " + act["root_category"], "other categories"
    )
    act.loc[act["root_category"] == "unknown", "group"] = "unknown category"
    g = act[act["week_number"].between(0, WEEKS)].groupby(["group", "week_number"])
    out = g["active_visitors"].sum().reset_index()
    size = out[out["week_number"] == 0].set_index("group")["active_visitors"]
    out["cohort_size"] = out["group"].map(size)
    out["retention_rate"] = out["active_visitors"] / out["cohort_size"]
    return out.assign(group_type="rr_first_category")


def run(database_url: str, artifacts: Path, model_dir: Path) -> dict[str, Any]:
    snap = pd.read_parquet(model_dir / "rr_snapshots.parquet")
    seg = segmentation(snap)
    (artifacts / "visitor_segmentation.json").write_text(
        json.dumps(seg, indent=2, default=float) + "\n"
    )
    coh = category_cohorts(database_url)
    coh.to_csv(artifacts / "visitor_category_cohorts.csv", index=False)
    w4 = coh[coh["week_number"] == 4].set_index("group")["retention_rate"]
    return {
        "chosen_k": seg["chosen_k"],
        "auc_rfm_grid": round(seg["comparison"]["rfm_grid"]["test_inactivity_auc"], 4),
        "auc_kmeans": round(seg["comparison"]["kmeans_behavioural"]["test_inactivity_auc"], 4),
        "week4_retention_range": [round(float(w4.min()), 4), round(float(w4.max()), 4)],
    }
