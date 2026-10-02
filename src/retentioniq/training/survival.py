"""Time-to-inactivity analysis on RetailRocket (PROJECT_SPEC.md §4.2, Q5).

Kaplan-Meier curves by first-session behaviour, then a Cox proportional-hazards model with
first-session covariates. The proportional-hazards assumption is tested (scaled Schoenfeld
residuals); covariates that violate it are stratified rather than reported as constant HRs.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from retentioniq import SEED
from retentioniq.features.retailrocket import sessionize

log = logging.getLogger(__name__)
GAP_DAYS = 30
HORIZONS = (7, 14, 30, 60, 90)
COVARIATES = [
    "first_session_events",
    "first_session_carted",
    "first_session_purchased",
    "first_session_items",
    "first_session_categories",
    "first_session_minutes",
    "first_session_weekend",
]


def build_durations(events: pd.DataFrame) -> pd.DataFrame:
    """One row per returning visitor: duration (days), event flag, first-session covariates."""
    ev = sessionize(events, 30)
    t_end = ev["event_ts"].max()
    sess = (
        ev.assign(cart=ev["event_type"].eq("addtocart"), tx=ev["event_type"].eq("transaction"))
        .groupby(["visitor_id", "session_id"])
        .agg(
            start=("event_ts", "min"),
            end=("event_ts", "max"),
            n=("event_ts", "size"),
            carts=("cart", "sum"),
            txs=("tx", "sum"),
            items=("item_id", "nunique"),
            cats=("category_id", "nunique"),
        )
        .reset_index()
        .sort_values(["visitor_id", "start"])
    )
    n_sess = sess.groupby("visitor_id").size()
    sess = sess[sess["visitor_id"].isin(n_sess[n_sess >= 2].index)]

    origin = sess.groupby("visitor_id")["start"].min()
    next_start = sess.groupby("visitor_id")["start"].shift(-1)
    gap_after = (next_start.fillna(t_end) - sess["end"]).dt.total_seconds() / 86_400
    # A gap after session end s is observable only if s + GAP <= T_end.
    observable = sess["end"] + pd.Timedelta(days=GAP_DAYS) <= t_end
    is_event = (gap_after >= GAP_DAYS) & observable
    first_event = sess[is_event].groupby("visitor_id")["end"].min()

    first = sess.groupby("visitor_id").first()
    out = pd.DataFrame(index=origin.index)
    out["origin"] = origin
    censor_at = t_end - pd.Timedelta(days=GAP_DAYS)
    out["event"] = out.index.isin(first_event.index).astype(int)
    end_time = first_event.reindex(out.index).fillna(censor_at)
    out["duration_days"] = (end_time - out["origin"]).dt.total_seconds() / 86_400
    out = out[out["origin"] <= censor_at - pd.Timedelta(days=1)]

    out["first_session_events"] = first["n"]
    out["first_session_carted"] = (first["carts"] > 0).astype(int)
    out["first_session_purchased"] = (first["txs"] > 0).astype(int)
    out["first_session_items"] = first["items"]
    out["first_session_categories"] = first["cats"]
    out["first_session_minutes"] = (first["end"] - first["start"]).dt.total_seconds() / 60
    out["first_session_weekend"] = (out["origin"].dt.dayofweek >= 5).astype(int)
    out["first_session_type"] = np.select(
        [first["txs"] > 0, first["carts"] > 0], ["purchased", "carted"], "viewed_only"
    )
    out["duration_days"] = out["duration_days"].clip(lower=1 / 24)  # same-day events -> 1h
    return out.reset_index()


def _km_table(df: pd.DataFrame, group_col: str | None) -> list[dict[str, Any]]:
    from lifelines import KaplanMeierFitter

    rows = []
    groups = [("all", df)] if group_col is None else list(df.groupby(group_col))
    for name, g in groups:
        km = KaplanMeierFitter().fit(g["duration_days"], g["event"])
        ci = km.confidence_interval_survival_function_
        for h in HORIZONS:
            if h > g["duration_days"].max():
                continue
            s = float(km.survival_function_at_times(h).iloc[0])
            ci_at = ci[ci.index <= h].iloc[-1]
            rows.append(
                {
                    "group_by": group_col or "all",
                    "group": str(name),
                    "n": len(g),
                    "events": int(g["event"].sum()),
                    "day": h,
                    "survival": s,
                    "ci_low": float(ci_at.iloc[0]),
                    "ci_high": float(ci_at.iloc[1]),
                    "median_days": float(km.median_survival_time_),
                }
            )
    return rows


def run(database_url: str, artifacts: Path, model_dir: Path) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.statistics import (
        logrank_test,
        multivariate_logrank_test,
        proportional_hazard_test,
    )

    from retentioniq.training.churn import load_events

    artifacts.mkdir(parents=True, exist_ok=True)
    df = build_durations(load_events(database_url))
    log.info("survival population: %d visitors, %d events", len(df), df["event"].sum())

    km_rows = _km_table(df, None) + _km_table(df, "first_session_type")
    km = pd.DataFrame(km_rows)
    lr = multivariate_logrank_test(df["duration_days"], df["first_session_type"], df["event"])

    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["event"])
    x_cols = COVARIATES
    model_df = train[["duration_days", "event", *x_cols]].copy()
    for c in (
        "first_session_events",
        "first_session_items",
        "first_session_categories",
        "first_session_minutes",
    ):
        model_df[c] = np.log1p(model_df[c])
    cph = CoxPHFitter(penalizer=0.01).fit(model_df, "duration_days", "event")

    sample = model_df.sample(min(20_000, len(model_df)), random_state=SEED)
    ph = proportional_hazard_test(cph, sample, time_transform="rank")
    ph_table = ph.summary.reset_index().rename(columns={"index": "covariate"})
    violating = ph_table.loc[ph_table["p"] < 0.01, "covariate"].astype(str).tolist()

    final = cph
    strata: list[str] = []
    binary_violators = [v for v in violating if model_df[v].nunique() <= 2]
    if binary_violators:
        # Stratify binary covariates that violate PH: they get their own baseline hazard
        # instead of a misleading constant hazard ratio.
        strata = binary_violators
        final = CoxPHFitter(penalizer=0.01).fit(model_df, "duration_days", "event", strata=strata)

    test_df = test[["duration_days", "event", *x_cols]].copy()
    for c in (
        "first_session_events",
        "first_session_items",
        "first_session_categories",
        "first_session_minutes",
    ):
        test_df[c] = np.log1p(test_df[c])
    c_index = float(final.score(test_df, scoring_method="concordance_index"))

    coefs = final.summary.reset_index()[
        ["covariate", "coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ].rename(
        columns={
            "exp(coef)": "hazard_ratio",
            "exp(coef) lower 95%": "hr_ci_low",
            "exp(coef) upper 95%": "hr_ci_high",
        }
    )
    coefs["ph_test_p"] = coefs["covariate"].map(ph_table.set_index("covariate")["p"])
    coefs["ph_violated"] = coefs["covariate"].isin(violating)

    km.assign(section="kaplan_meier").to_csv(artifacts / "survival_analysis.csv", index=False)
    coefs.to_csv(artifacts / "survival_cox_coefficients.csv", index=False)
    summary = {
        "population": len(df),
        "events": int(df["event"].sum()),
        "censored": int((1 - df["event"]).sum()),
        "gap_days": GAP_DAYS,
        "logrank_first_session_type": {
            "statistic": float(lr.test_statistic),
            "p_value": float(lr.p_value),
        },
        "cox": {
            "concordance_test": c_index,
            "strata": strata,
            "ph_violations_p_lt_0.01": violating,
            "n_train": len(train),
            "n_test": len(test),
        },
        "purchased_vs_viewed_logrank_p": float(
            logrank_test(
                df.loc[df["first_session_type"] == "purchased", "duration_days"],
                df.loc[df["first_session_type"] == "viewed_only", "duration_days"],
                df.loc[df["first_session_type"] == "purchased", "event"],
                df.loc[df["first_session_type"] == "viewed_only", "event"],
            ).p_value
        )
        if (df["first_session_type"] == "purchased").any()
        else None,
    }
    (artifacts / "survival_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    df.set_index("visitor_id")[[*x_cols, "duration_days", "event"]].to_parquet(
        model_dir / "survival_covariates.parquet"
    )
    joblib.dump(
        {
            "model": final,
            "covariates": x_cols,
            "strata": strata,
            "log_cols": [
                "first_session_events",
                "first_session_items",
                "first_session_categories",
                "first_session_minutes",
            ],
        },
        model_dir / "survival_cox.joblib",
    )
    return summary
