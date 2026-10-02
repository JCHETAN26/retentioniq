"""Point-in-time inactivity snapshots for RetailRocket (PROJECT_SPEC.md §4.1, §5).

`build_snapshot(events, cutoff)` uses only events strictly before `cutoff` for features and
only events in `[cutoff, cutoff + horizon)` for the label. Feature columns are grouped into
the ablation families of §5.3 via `FEATURE_FAMILIES`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_SET_VERSION = "rr-features.v1"

FEATURE_FAMILIES: dict[str, list[str]] = {
    "rfm": ["recency_days", "sessions_obs", "transactions_obs"],
    "transactions": [
        "ever_purchased",
        "days_since_last_transaction",
        "purchase_sessions_obs",
        "items_purchased_obs",
    ],
    "engagement": [
        "views_7d",
        "views_30d",
        "sessions_7d",
        "sessions_30d",
        "active_days_obs",
        "events_per_session",
        "mean_session_minutes",
    ],
    "funnel": [
        "carts_obs",
        "cart_rate",
        "conversion_rate",
        "cart_session_rate",
        "cart_to_purchase",
    ],
    "temporal": [
        "tenure_days",
        "activity_trend",
        "session_trend",
        "mean_days_between_sessions",
        "std_days_between_sessions",
        "recency_vs_typical_gap",
    ],
    "product": [
        "unique_items_obs",
        "unique_categories_obs",
        "category_entropy",
        "repeat_item_rate",
        "top_category_share",
    ],
}
ABLATION_ORDER = ["rfm", "transactions", "engagement", "funnel", "temporal", "product"]
ALL_FEATURES = [f for fam in ABLATION_ORDER for f in FEATURE_FAMILIES[fam]]
DAY = pd.Timedelta(days=1)


def _days(td: pd.Series) -> pd.Series:
    days: pd.Series = td.dt.total_seconds() / 86_400
    return days


@dataclass(frozen=True)
class SnapshotConfig:
    observation_days: int = 30
    horizon_days: int = 30
    session_gap_minutes: int = 30
    min_sessions: int = 2


def sessionize(events: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    """Add `session_id` (visitor-scoped integer). Only looks backwards, so it is leak-free."""
    ev = events.sort_values(["visitor_id", "event_ts"], kind="stable").reset_index(drop=True)
    gap = ev.groupby("visitor_id")["event_ts"].diff()
    new = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    ev["session_id"] = new.cumsum()
    return ev


def derive_cutoffs(
    t_start: pd.Timestamp, t_end: pd.Timestamp, cfg: SnapshotConfig
) -> dict[str, list[pd.Timestamp]]:
    """Cutoffs per PROJECT_SPEC.md §4.1 (all labels observable before the next split)."""
    h = pd.Timedelta(days=cfg.horizon_days)
    test = t_end.floor("D") - h
    val = test - h
    earliest = t_start.ceil("D") + pd.Timedelta(days=cfg.observation_days)
    train = [val - h - pd.Timedelta(days=d) for d in (14, 7, 0)]
    train = [c for c in train if c >= earliest]
    if not train:
        raise ValueError("no feasible training cutoff for this configuration")
    return {"train": train, "validation": [val], "test": [test]}


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(0.0)


def build_snapshot(
    events: pd.DataFrame, cutoff: pd.Timestamp, cfg: SnapshotConfig | None = None
) -> pd.DataFrame:
    """Features + label for eligible visitors at `cutoff`.

    `events` needs columns: event_ts, visitor_id, event_type, item_id, category_id, session_id.
    """
    cfg = cfg or SnapshotConfig()
    obs_start = cutoff - pd.Timedelta(days=cfg.observation_days)
    horizon_end = cutoff + pd.Timedelta(days=cfg.horizon_days)
    past = events[events["event_ts"] < cutoff]
    obs = past[past["event_ts"] >= obs_start]

    sessions = (
        obs.assign(cart=obs["event_type"].eq("addtocart"), tx=obs["event_type"].eq("transaction"))
        .groupby(["visitor_id", "session_id"])
        .agg(
            start=("event_ts", "min"),
            end=("event_ts", "max"),
            n=("event_ts", "size"),
            carts=("cart", "sum"),
            purchases=("tx", "sum"),
        )
        .reset_index()
    )
    n_sessions = sessions.groupby("visitor_id").size()
    eligible = n_sessions[n_sessions >= cfg.min_sessions].index
    obs = obs[obs["visitor_id"].isin(eligible)]
    sessions = sessions[sessions["visitor_id"].isin(eligible)]

    f = pd.DataFrame(index=pd.Index(eligible, name="visitor_id"))
    is_view = obs["event_type"].eq("view")
    is_cart = obs["event_type"].eq("addtocart")
    is_tx = obs["event_type"].eq("transaction")
    g = obs.assign(view=is_view, cart=is_cart, tx=is_tx).groupby("visitor_id")
    views, carts, txs = g["view"].sum(), g["cart"].sum(), g["tx"].sum()
    last_event = g["event_ts"].max()

    # RFM
    f["recency_days"] = _days(cutoff - last_event)
    f["sessions_obs"] = n_sessions.reindex(f.index)
    f["transactions_obs"] = txs

    # Transactions (all history before cutoff for "ever"/"last")
    past_tx = past[past["event_type"].eq("transaction") & past["visitor_id"].isin(eligible)]
    last_tx = past_tx.groupby("visitor_id")["event_ts"].max()
    f["ever_purchased"] = f.index.isin(last_tx.index).astype(int)
    f["days_since_last_transaction"] = (
        _days(cutoff - last_tx).reindex(f.index).fillna(float(cfg.observation_days * 4))
    )
    f["purchase_sessions_obs"] = (sessions["purchases"] > 0).groupby(sessions["visitor_id"]).sum()
    f["items_purchased_obs"] = obs[is_tx].groupby("visitor_id")["item_id"].nunique()

    # Engagement
    for days in (7, 30):
        lo = cutoff - pd.Timedelta(days=days)
        w = obs[obs["event_ts"] >= lo]
        f[f"views_{days}d"] = w[w["event_type"].eq("view")].groupby("visitor_id").size()
        f[f"sessions_{days}d"] = w.groupby("visitor_id")["session_id"].nunique()
    f["active_days_obs"] = obs["event_ts"].dt.floor("D").groupby(obs["visitor_id"]).nunique()
    f["events_per_session"] = _safe_ratio(g.size(), f["sessions_obs"])
    minutes = _days(sessions["end"] - sessions["start"]) * 1440
    f["mean_session_minutes"] = minutes.groupby(sessions["visitor_id"]).mean()

    # Funnel
    f["carts_obs"] = carts
    f["cart_rate"] = _safe_ratio(carts, views)
    f["conversion_rate"] = _safe_ratio(txs, views)
    f["cart_session_rate"] = _safe_ratio(
        (sessions["carts"] > 0).groupby(sessions["visitor_id"]).sum(), f["sessions_obs"]
    )
    f["cart_to_purchase"] = _safe_ratio(txs, carts)

    # Temporal
    first_seen = past[past["visitor_id"].isin(eligible)].groupby("visitor_id")["event_ts"].min()
    f["tenure_days"] = _days(cutoff - first_seen)
    half = cutoff - pd.Timedelta(days=cfg.observation_days / 2)
    recent = obs["event_ts"] >= half
    ev_recent = obs[recent].groupby("visitor_id").size().reindex(f.index, fill_value=0)
    ev_early = obs[~recent].groupby("visitor_id").size().reindex(f.index, fill_value=0)
    f["activity_trend"] = (ev_recent - ev_early) / (ev_recent + ev_early)
    s_recent = (sessions["start"] >= half).astype(float).groupby(sessions["visitor_id"]).sum()
    s_early = (sessions["start"] < half).astype(float).groupby(sessions["visitor_id"]).sum()
    f["session_trend"] = (s_recent - s_early) / (s_recent + s_early)
    ordered = sessions.sort_values(["visitor_id", "start"])
    gaps = _days(ordered.groupby("visitor_id")["start"].diff()).groupby(ordered["visitor_id"])
    f["mean_days_between_sessions"] = gaps.mean()
    f["std_days_between_sessions"] = gaps.std()
    f["recency_vs_typical_gap"] = _safe_ratio(f["recency_days"], f["mean_days_between_sessions"])

    # Product
    f["unique_items_obs"] = g["item_id"].nunique()
    f["unique_categories_obs"] = g["category_id"].nunique()
    cat = obs.dropna(subset=["category_id"]).groupby(["visitor_id", "category_id"]).size()
    share = cat / cat.groupby(level=0).transform("sum")
    f["category_entropy"] = (-(share * np.log(share))).groupby(level=0).sum()
    f["top_category_share"] = share.groupby(level=0).max()
    f["repeat_item_rate"] = 1 - _safe_ratio(f["unique_items_obs"], g.size())

    top_cat = cat.groupby(level=0).idxmax().map(lambda ix: ix[1])
    f = f[ALL_FEATURES].astype(float).fillna(0.0)
    f["top_category_id"] = top_cat.reindex(f.index)  # for segment analysis only, not a feature

    # Label: no activity at all in [cutoff, cutoff + h)
    future = events[(events["event_ts"] >= cutoff) & (events["event_ts"] < horizon_end)]
    active_future = future["visitor_id"].unique()
    f["inactive"] = (~f.index.isin(active_future)).astype(int)
    future_tx = future[future["event_type"].eq("transaction")]["visitor_id"].unique()
    f["purchase_inactive"] = (~f.index.isin(future_tx)).astype(int)
    f["cutoff"] = cutoff
    return f.reset_index()


def build_snapshots(
    events: pd.DataFrame, cfg: SnapshotConfig | None = None
) -> tuple[pd.DataFrame, dict[str, list[pd.Timestamp]]]:
    """All snapshots with a `split` column, plus the cutoff manifest."""
    cfg = cfg or SnapshotConfig()
    ev = sessionize(events, cfg.session_gap_minutes)
    cutoffs = derive_cutoffs(ev["event_ts"].min(), ev["event_ts"].max(), cfg)
    frames = [
        build_snapshot(ev, c, cfg).assign(split=split) for split, cs in cutoffs.items() for c in cs
    ]
    return pd.concat(frames, ignore_index=True), cutoffs
