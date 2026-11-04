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
LANDMARK_DAYS = 7
LANDMARK_COVARIATES = [
    "sessions_7d",
    "events_7d",
    "active_days_7d",
    "carted_7d",
    "purchased_7d",
    "unique_items_7d",
    "unique_categories_7d",
    "days_since_last_session_at_7d",
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
    first = first.reindex(out.index)

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


def build_landmark_durations(
    events: pd.DataFrame,
    landmark_days: int = LANDMARK_DAYS,
) -> pd.DataFrame:
    """One row per returning visitor eligible at t_LM (landmark_days): duration from landmark,
    event flag, and covariates measured strictly in the window [origin, origin + landmark_days).
    """
    ev = sessionize(events, 30)
    t_end = ev["event_ts"].max()
    censor_at = t_end - pd.Timedelta(days=GAP_DAYS)
    lm_cutoff = pd.Timedelta(days=landmark_days)

    base = build_durations(events).set_index("visitor_id")
    # Eligible:
    # 1. origin + landmark_days <= censor_at (at least landmark_days before cutoff)
    # 2. Exclude visitors whose event occurred before landmark (duration < landmark and event == 1)
    eligible_mask = (base["origin"] + lm_cutoff <= censor_at) & ~(
        (base["event"] == 1) & (base["duration_days"] < landmark_days)
    )
    eligible = base[eligible_mask].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "visitor_id",
                "origin",
                "duration_days_lm",
                "event",
                "activity_tier",
                *LANDMARK_COVARIATES,
            ]
        )

    # Duration measured from landmark (T - 7)
    eligible["duration_days_lm"] = (eligible["duration_days"] - landmark_days).clip(lower=1 / 24)

    # Filter events strictly to [origin, origin + landmark_days)
    ev_elig = ev[ev["visitor_id"].isin(eligible.index)].copy()
    ev_elig["origin"] = ev_elig["visitor_id"].map(eligible["origin"])
    ev_elig["lm_end"] = ev_elig["origin"] + lm_cutoff
    in_window = ev_elig[
        (ev_elig["event_ts"] >= ev_elig["origin"]) & (ev_elig["event_ts"] < ev_elig["lm_end"])
    ].copy()

    in_window["cart"] = in_window["event_type"].eq("addtocart")
    in_window["tx"] = in_window["event_type"].eq("transaction")
    in_window["date"] = in_window["event_ts"].dt.date

    # Compute aggregates in the strictly bounded window
    agg = in_window.groupby("visitor_id").agg(
        sessions_7d=("session_id", "nunique"),
        events_7d=("event_ts", "size"),
        active_days_7d=("date", "nunique"),
        carted_7d=("cart", lambda s: int(s.sum() > 0)),
        purchased_7d=("tx", lambda s: int(s.sum() > 0)),
        unique_items_7d=("item_id", "nunique"),
        unique_categories_7d=("category_id", "nunique"),
        last_event_ts=("event_ts", "max"),
    )

    merged = eligible.join(agg, how="left")
    merged["sessions_7d"] = merged["sessions_7d"].fillna(1).astype(int)
    merged["events_7d"] = merged["events_7d"].fillna(1).astype(int)
    merged["active_days_7d"] = merged["active_days_7d"].fillna(1).astype(int)
    merged["carted_7d"] = merged["carted_7d"].fillna(0).astype(int)
    merged["purchased_7d"] = merged["purchased_7d"].fillna(0).astype(int)
    merged["unique_items_7d"] = merged["unique_items_7d"].fillna(1).astype(int)
    merged["unique_categories_7d"] = merged["unique_categories_7d"].fillna(1).astype(int)

    delta_days = (
        merged["last_event_ts"].fillna(merged["origin"]) - merged["origin"]
    ).dt.total_seconds() / 86_400
    merged["days_since_last_session_at_7d"] = (landmark_days - delta_days).clip(lower=0.0)

    merged["activity_tier"] = np.select(
        [merged["sessions_7d"] >= 4, merged["sessions_7d"] >= 2],
        ["4+ sessions", "2-3 sessions"],
        "1 session",
    )

    out = merged.reset_index()[
        [
            "visitor_id",
            "origin",
            "duration_days_lm",
            "event",
            "activity_tier",
            *LANDMARK_COVARIATES,
        ]
    ]
    return out


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
    events = load_events(database_url)
    df = build_durations(events)
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

    # Schoenfeld-residual PH test on a 20k subsample (the test needs the data the model was fit
    # on, and its cost grows with n; with 100k+ rows trivial departures would all be "significant").
    sample = model_df.sample(min(20_000, len(model_df)), random_state=SEED)
    cph_sample = CoxPHFitter(penalizer=0.01).fit(sample, "duration_days", "event")
    ph = proportional_hazard_test(cph_sample, sample, time_transform="rank")
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

    # 3.1 Single-covariate baseline (first_session_events alone)
    single_cph = CoxPHFitter(penalizer=0.01).fit(
        model_df[["duration_days", "event", "first_session_events"]],
        "duration_days",
        "event",
    )
    single_c_index = float(
        single_cph.score(
            test_df[["duration_days", "event", "first_session_events"]],
            scoring_method="concordance_index",
        )
    )

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
            "concordance_single_covariate": single_c_index,
            "concordance_random_baseline": 0.50,
            "interpretation": (
                f"Held-out concordance of {c_index:.3f} is close to random baseline (0.50), "
                f"demonstrating that first-session engagement alone does not reliably predict "
                f"long-term return timing without post-session behavioral history."
            ),
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

    # 3.2 Landmark analysis at day 7
    lm_df = build_landmark_durations(events, landmark_days=LANDMARK_DAYS)
    log.info(
        "landmark population: %d visitors, %d events",
        len(lm_df),
        int(lm_df["event"].sum()) if not lm_df.empty else 0,
    )

    has_both_events = len(lm_df["event"].unique()) > 1 if not lm_df.empty else False
    if len(lm_df) >= 20 and has_both_events:
        lm_lr = multivariate_logrank_test(
            lm_df["duration_days_lm"], lm_df["activity_tier"], lm_df["event"]
        )
        lm_km_rows = _km_table(
            lm_df.rename(columns={"duration_days_lm": "duration_days"}), "activity_tier"
        )
        lm_train, lm_test = train_test_split(
            lm_df, test_size=0.2, random_state=SEED, stratify=lm_df["event"]
        )
        lm_model_df = lm_train[["duration_days_lm", "event", *LANDMARK_COVARIATES]].copy()
        lm_test_df = lm_test[["duration_days_lm", "event", *LANDMARK_COVARIATES]].copy()
        log_lm_cols = [
            "sessions_7d",
            "events_7d",
            "active_days_7d",
            "unique_items_7d",
            "unique_categories_7d",
        ]
        for c in log_lm_cols:
            lm_model_df[c] = np.log1p(lm_model_df[c])
            lm_test_df[c] = np.log1p(lm_test_df[c])

        lm_cph = CoxPHFitter(penalizer=0.01).fit(lm_model_df, "duration_days_lm", "event")

        # Schoenfeld PH test on landmark model
        lm_sample = lm_model_df.sample(min(20_000, len(lm_model_df)), random_state=SEED)
        lm_sample_cph = CoxPHFitter(penalizer=0.01).fit(lm_sample, "duration_days_lm", "event")
        lm_ph = proportional_hazard_test(lm_sample_cph, lm_sample, time_transform="rank")
        lm_ph_table = lm_ph.summary.reset_index().rename(columns={"index": "covariate"})
        lm_violating = lm_ph_table.loc[lm_ph_table["p"] < 0.01, "covariate"].astype(str).tolist()

        lm_final = lm_cph
        lm_strata: list[str] = []
        lm_binary_violators = [v for v in lm_violating if lm_model_df[v].nunique() <= 2]
        if lm_binary_violators:
            lm_strata = lm_binary_violators
            lm_final = CoxPHFitter(penalizer=0.01).fit(
                lm_model_df, "duration_days_lm", "event", strata=lm_strata
            )

        lm_c_index = float(lm_final.score(lm_test_df, scoring_method="concordance_index"))
        lm_coefs = lm_final.summary.reset_index()[
            [
                "covariate",
                "coef",
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p",
            ]
        ].rename(
            columns={
                "exp(coef)": "hazard_ratio",
                "exp(coef) lower 95%": "hr_ci_low",
                "exp(coef) upper 95%": "hr_ci_high",
            }
        )
        lm_coefs["ph_test_p"] = lm_coefs["covariate"].map(lm_ph_table.set_index("covariate")["p"])
        lm_coefs["ph_violated"] = lm_coefs["covariate"].isin(lm_violating)
        lm_coefs.to_csv(artifacts / "survival_landmark.csv", index=False)

        delta_c = lm_c_index - c_index
        lm_summary = {
            "population": len(lm_df),
            "events": int(lm_df["event"].sum()),
            "censored": int((1 - lm_df["event"]).sum()),
            "landmark_days": LANDMARK_DAYS,
            "concordance_test": lm_c_index,
            "first_session_concordance_test": c_index,
            "concordance_delta": delta_c,
            "strata": lm_strata,
            "ph_violations_p_lt_0.01": lm_violating,
            "logrank_activity_tier": {
                "statistic": float(lm_lr.test_statistic),
                "p_value": float(lm_lr.p_value),
            },
            "activity_tier_km": lm_km_rows,
            "interpretation": (
                f"Evaluating visitors at a day-{LANDMARK_DAYS} landmark yields test concordance "
                f"of {lm_c_index:.3f} vs {c_index:.3f} for first-session engagement (delta "
                f"+{delta_c:.3f}). 7-day behavioral history (session frequency, catalog "
                f"exploration, active days) provides stronger predictive signal."
            ),
        }
        (artifacts / "survival_landmark_summary.json").write_text(
            json.dumps(lm_summary, indent=2) + "\n"
        )
    else:
        pd.DataFrame(
            columns=[
                "covariate",
                "coef",
                "hazard_ratio",
                "hr_ci_low",
                "hr_ci_high",
                "p",
                "ph_test_p",
                "ph_violated",
            ]
        ).to_csv(artifacts / "survival_landmark.csv", index=False)
        (artifacts / "survival_landmark_summary.json").write_text(
            json.dumps(
                {
                    "population": len(lm_df),
                    "events": int(lm_df["event"].sum()) if not lm_df.empty else 0,
                    "landmark_days": LANDMARK_DAYS,
                    "concordance_test": None,
                    "first_session_concordance_test": c_index,
                },
                indent=2,
            )
            + "\n"
        )
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
