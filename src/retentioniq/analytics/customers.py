"""Customer analytics on Online Retail II: RFM, behavioral segmentation, cohorts (§3 Q1-Q3).

Segmentations are compared on *future* behaviour: both are computed from orders before a
cutoff (the last `HORIZON_DAYS` are held out) and judged on how well they separate customers
who purchase again in the held-out window.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retentioniq import SEED

HORIZON_DAYS = 90
SEGMENT_ORDER = [
    "Champions",
    "Loyal",
    "Potential Loyalists",
    "New Customers",
    "Needs Attention",
    "At Risk",
    "Hibernating",
    "Lost",
]
CLUSTER_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "mean_interval_days",
    "interval_cv",
    "n_products",
    "orders_90d",
    "orders_prev_90d",
    "activity_slope",
]
LOG_FEATURES = {
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "mean_interval_days",
    "n_products",
    "orders_90d",
    "orders_prev_90d",
}


def load_orders(database_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    q = """
        SELECT o.invoice, o.customer_id, o.order_date, o.order_value, o.n_products
        FROM marts.fact_orders o
    """
    return pd.read_sql(q, create_engine(database_url), parse_dates=["order_date"])


def rfm_scores(orders: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """RFM per PROJECT_SPEC.md §4.3 (mirrors dbt `mart_customer_rfm`)."""
    g = orders.groupby("customer_id")
    df = pd.DataFrame(
        {
            "recency_days": (snapshot - g["order_date"].max()).dt.days,
            "frequency": g.size(),
            "monetary": g["order_value"].sum(),
        }
    ).reset_index()
    n = len(df)

    def ntile(order_cols: list[str], ascending: list[bool]) -> np.ndarray:
        ranks = np.empty(n, dtype=int)
        idx = df.sort_values(order_cols, ascending=ascending, kind="stable").index.to_numpy()
        # Postgres ntile(5): first (n % 5) buckets get one extra row.
        sizes = [n // 5 + (1 if i < n % 5 else 0) for i in range(5)]
        ranks[idx] = np.repeat(np.arange(1, 6), sizes)
        return ranks

    df["r_score"] = ntile(["recency_days", "customer_id"], [False, True])
    df["f_score"] = ntile(["frequency", "customer_id"], [True, True])
    df["m_score"] = ntile(["monetary", "customer_id"], [True, True])
    fm = (df["f_score"] + df["m_score"]) / 2
    r = df["r_score"]
    conditions = [
        (r >= 4) & (fm >= 4.5),
        (r >= 3) & (fm >= 3.5),
        (r >= 4) & (fm >= 2) & (fm < 3.5),
        (r == 5) & (fm < 2),
        (r <= 2) & (fm >= 3),
        (r == 2) & (fm < 3),
        (r == 1) & (fm < 3),
    ]
    labels = SEGMENT_ORDER[:4] + SEGMENT_ORDER[5:]
    df["rfm_segment"] = np.select(conditions, labels, default="Needs Attention")
    return df


def behavioral_features(orders: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    o = orders.sort_values(["customer_id", "order_date"])
    g = o.groupby("customer_id")
    gaps = o.groupby("customer_id")["order_date"].diff().dt.days
    f = pd.DataFrame(
        {
            "recency_days": (snapshot - g["order_date"].max()).dt.days,
            "frequency": g.size(),
            "monetary": g["order_value"].sum(),
            "tenure_days": (snapshot - g["order_date"].min()).dt.days,
            "n_products": g["n_products"].sum(),
            "mean_interval_days": gaps.groupby(o["customer_id"]).mean(),
            "interval_cv": gaps.groupby(o["customer_id"]).std()
            / gaps.groupby(o["customer_id"]).mean(),
        }
    )
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    d90 = snapshot - pd.Timedelta(days=90)
    d180 = snapshot - pd.Timedelta(days=180)
    f["orders_90d"] = o[o["order_date"] >= d90].groupby("customer_id").size()
    f["orders_prev_90d"] = (
        o[(o["order_date"] >= d180) & (o["order_date"] < d90)].groupby("customer_id").size()
    )
    f = f.fillna({"orders_90d": 0, "orders_prev_90d": 0})
    # Single-order customers have no interval: use tenure (time since that order) and CV 0.
    f["mean_interval_days"] = f["mean_interval_days"].fillna(f["tenure_days"])
    f["interval_cv"] = f["interval_cv"].fillna(0.0)
    f["activity_slope"] = (f["orders_90d"] - f["orders_prev_90d"]) / (
        f["orders_90d"] + f["orders_prev_90d"]
    ).replace(0, np.nan)
    f["activity_slope"] = f["activity_slope"].fillna(0.0)
    return f[CLUSTER_FEATURES].reset_index()


def _prepare(x: pd.DataFrame) -> np.ndarray:
    z = x.copy()
    for c in LOG_FEATURES:
        z[c] = np.log1p(z[c].clip(lower=0))
    return StandardScaler().fit_transform(z.to_numpy(dtype=float))


def _stability(x: np.ndarray, k: int, n_runs: int = 10) -> float:
    """Mean ARI between labels from bootstrap-fitted models and the full-data model."""
    rng = np.random.default_rng(SEED)
    ref = KMeans(k, n_init=10, random_state=SEED).fit(x)
    scores = []
    for i in range(n_runs):
        idx = rng.integers(0, len(x), len(x))
        m = KMeans(k, n_init=3, random_state=SEED + i).fit(x[idx])
        scores.append(adjusted_rand_score(ref.labels_, m.predict(x)))
    return float(np.mean(scores))


def segment_predictive_auc(labels: pd.Series, y: np.ndarray) -> float:
    """Cross-validated AUC of predicting future activity from segment membership alone."""
    enc = OneHotEncoder(handle_unknown="ignore").fit_transform(labels.to_frame().astype(str))
    p = cross_val_predict(LogisticRegression(max_iter=1000), enc, y, cv=5, method="predict_proba")
    return float(roc_auc_score(y, p[:, 1]))


def segment_profile(df: pd.DataFrame, by: str, future_col: str) -> pd.DataFrame:
    g = df.groupby(by)
    total_rev = df["monetary"].sum()
    return pd.DataFrame(
        {
            "customers": g.size(),
            "share_customers": g.size() / len(df),
            "revenue": g["monetary"].sum(),
            "share_revenue": g["monetary"].sum() / total_rev,
            "avg_order_value": g["monetary"].sum() / g["frequency"].sum(),
            "mean_frequency": g["frequency"].mean(),
            "median_recency_days": g["recency_days"].median(),
            "repeat_purchase_rate": g["frequency"].apply(lambda s: (s >= 2).mean()),
            "future_active_rate": g[future_col].mean(),
        }
    ).reset_index()


def run(database_url: str, artifacts: Path) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    orders = load_orders(database_url)
    end = orders["order_date"].max()
    cutoff = end - pd.Timedelta(days=HORIZON_DAYS - 1)
    past = orders[orders["order_date"] < cutoff]
    future_customers = set(orders.loc[orders["order_date"] >= cutoff, "customer_id"])
    snapshot = cutoff

    rfm = rfm_scores(past, snapshot)
    feats = behavioral_features(past, snapshot)
    df = rfm.merge(
        feats[["customer_id", *[c for c in CLUSTER_FEATURES if c not in rfm]]], on="customer_id"
    )
    df["future_active"] = df["customer_id"].isin(future_customers).astype(int)
    y = df["future_active"].to_numpy()
    x = _prepare(df[CLUSTER_FEATURES])

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x), size=min(10_000, len(x)), replace=False)
    k_search = []
    for k in range(3, 11):
        km = KMeans(k, n_init=10, random_state=SEED).fit(x)
        gmm = GaussianMixture(k, covariance_type="full", random_state=SEED).fit(x)
        k_search.append(
            {
                "k": k,
                "kmeans_silhouette": float(silhouette_score(x[sample], km.labels_[sample])),
                "kmeans_davies_bouldin": float(davies_bouldin_score(x, km.labels_)),
                "kmeans_inertia": float(km.inertia_),
                "kmeans_stability_ari": _stability(x, k),
                "gmm_bic": float(gmm.bic(x)),
                "kmeans_future_auc": segment_predictive_auc(pd.Series(km.labels_), y),
            }
        )
    ks = pd.DataFrame(k_search)
    # Choose k: most stable among the top-3 silhouette values (ties -> fewer clusters).
    top = ks.nlargest(3, "kmeans_silhouette")
    k_best = int(
        top.sort_values(["kmeans_stability_ari", "k"], ascending=[False, True]).iloc[0]["k"]
    )
    km = KMeans(k_best, n_init=10, random_state=SEED).fit(x)
    df["cluster"] = km.labels_
    gmm = GaussianMixture(k_best, covariance_type="full", random_state=SEED).fit(x)
    df["gmm_cluster"] = gmm.predict(x)
    # Density-based alternative: no k to choose, and points in sparse regions become noise (-1).
    hdb = HDBSCAN(min_cluster_size=max(25, len(x) // 100), min_samples=10).fit(x)
    df["hdbscan_cluster"] = hdb.labels_
    coords = PCA(2, random_state=SEED).fit_transform(x)
    df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]
    # UMAP is for visualization only, never evidence that clusters are real (Phase 8).
    try:
        import umap

        emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(x)
        df["umap1"], df["umap2"] = emb[:, 0], emb[:, 1]
    except ImportError:
        pass

    # Name clusters by their dominant RFM segment so they are interpretable.
    dominant = df.groupby("cluster")["rfm_segment"].agg(lambda s: s.value_counts().index[0])
    df["cluster_label"] = df["cluster"].map(lambda c: f"C{c}: mostly {dominant[c]}")

    comparison: dict[str, Any] = {}
    for name, col in (
        ("rfm_segments", "rfm_segment"),
        ("kmeans_clusters", "cluster"),
        ("gmm_clusters", "gmm_cluster"),
        ("hdbscan_clusters", "hdbscan_cluster"),
    ):
        comparison[name] = {
            "n_groups": int(df[col].nunique()),
            "future_activity_auc": segment_predictive_auc(df[col], y),
        }
    comparison["rfm_segments"]["silhouette"] = float(
        silhouette_score(x[sample], df["rfm_segment"].to_numpy()[sample])
    )
    comparison["kmeans_clusters"]["silhouette"] = float(
        silhouette_score(x[sample], df["cluster"].to_numpy()[sample])
    )
    clustered = df["hdbscan_cluster"].to_numpy() >= 0
    comparison["hdbscan_clusters"]["noise_share"] = float(1 - clustered.mean())
    in_sample = np.zeros(len(df), dtype=bool)
    in_sample[sample] = True
    keep = clustered & in_sample
    if len(set(df.loc[keep, "hdbscan_cluster"])) > 1:
        comparison["hdbscan_clusters"]["silhouette_excl_noise"] = float(
            silhouette_score(x[keep], df["hdbscan_cluster"].to_numpy()[keep])
        )
    comparison["agreement_ari_rfm_vs_kmeans"] = float(
        adjusted_rand_score(df["rfm_segment"], df["cluster"])
    )

    # Full-history RFM (for reporting, matches the dbt mart).
    full_rfm = rfm_scores(orders, end + pd.Timedelta(days=1))
    rfm_report = segment_profile(
        full_rfm.assign(future_active=0), "rfm_segment", "future_active"
    ).drop(columns="future_active_rate")
    out = {
        "dataset": "online-retail",
        "evaluation_cutoff": str(cutoff.date()),
        "horizon_days": HORIZON_DAYS,
        "customers_at_cutoff": len(df),
        "future_active_rate": float(y.mean()),
        "chosen_k": k_best,
        "k_search": ks.to_dict(orient="records"),
        "comparison": comparison,
        "rfm_segments_full_history": rfm_report.to_dict(orient="records"),
        "rfm_segments_at_cutoff": segment_profile(df, "rfm_segment", "future_active").to_dict(
            orient="records"
        ),
        "clusters_at_cutoff": segment_profile(df, "cluster_label", "future_active").to_dict(
            orient="records"
        ),
        "value_concentration": value_concentration(full_rfm["monetary"]),
    }
    (artifacts / "segmentation_metrics.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n"
    )
    df.to_csv(artifacts / "customer_segments.csv", index=False)
    cohort_metrics(database_url, df[["customer_id", "rfm_segment", "cluster_label"]]).to_csv(
        artifacts / "cohort_metrics.csv", index=False
    )
    return {"chosen_k": k_best, **comparison}


def value_concentration(monetary: pd.Series) -> dict[str, float]:
    v = np.sort(monetary.to_numpy())[::-1]
    total = v.sum()
    n = len(v)
    cum = np.cumsum(np.sort(v))
    gini = float((n + 1 - 2 * cum.sum() / cum[-1]) / n)
    return {
        "top_1pct_revenue_share": float(v[: max(1, n // 100)].sum() / total),
        "top_10pct_revenue_share": float(v[: max(1, n // 10)].sum() / total),
        "top_20pct_revenue_share": float(v[: max(1, n // 5)].sum() / total),
        "gini": gini,
    }


def cohort_metrics(database_url: str, segments: pd.DataFrame) -> pd.DataFrame:
    """Retention by first-purchase cohort (the matrix) and by segment groupings (curves)."""
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    matrix = pd.read_sql("SELECT * FROM marts.mart_customer_retention", engine)
    matrix = matrix.assign(group_type="cohort", group=matrix["cohort_month"].astype(str))

    activity = pd.read_sql(
        "SELECT customer_id, cohort_month, month_number FROM marts.mart_customer_cohorts", engine
    )
    last = activity["cohort_month"].max()
    # Only customers whose cohort has >= 12 observable months contribute to group curves.
    eligible = activity.groupby("customer_id")["cohort_month"].first()
    eligible = eligible[(pd.to_datetime(last) - pd.to_datetime(eligible)).dt.days >= 365].index
    act = activity[activity["customer_id"].isin(eligible) & (activity["month_number"] <= 12)]
    act = act.merge(segments, on="customer_id", how="inner")
    # Value tier from the FIRST order (known at acquisition); lifetime value would leak the
    # future spending that retention itself produces.
    first = pd.read_sql(
        "SELECT customer_id, order_value FROM marts.fact_orders WHERE customer_order_number = 1",
        engine,
    )
    first["value_tier"] = pd.qcut(
        first["order_value"],
        4,
        labels=["Q1 lowest first order", "Q2", "Q3", "Q4 highest first order"],
    ).astype(str)
    act = act.merge(first[["customer_id", "value_tier"]], on="customer_id", how="left")
    rows = []
    for group_type, col in (
        ("rfm_segment", "rfm_segment"),
        ("cluster", "cluster_label"),
        ("first_order_value", "value_tier"),
    ):
        sizes = act.groupby(col)["customer_id"].nunique()
        counts = (
            act.groupby([col, "month_number"])["customer_id"]
            .nunique()
            .reset_index(name="active_customers")
        )
        for rec in counts.to_dict("records"):
            size = int(sizes[rec[col]])
            rows.append(
                {
                    "group_type": group_type,
                    "group": rec[col],
                    "month_number": rec["month_number"],
                    "cohort_size": size,
                    "active_customers": int(rec["active_customers"]),
                    "retention_rate": rec["active_customers"] / size,
                }
            )
    curves = pd.DataFrame(rows)
    cols = [
        "group_type",
        "group",
        "month_number",
        "cohort_size",
        "active_customers",
        "retention_rate",
    ]
    return pd.concat([matrix[[*cols, "revenue"]], curves], ignore_index=True)
