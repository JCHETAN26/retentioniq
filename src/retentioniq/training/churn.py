"""Inactivity prediction pipeline: snapshots -> model ladder -> ablation -> sensitivity.

Writes the artifacts listed in PROJECT_SPEC.md §8 for the churn side and logs every model
to MLflow (when a tracking server is reachable).
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from retentioniq import SEED
from retentioniq.features.retailrocket import (
    ABLATION_ORDER,
    ALL_FEATURES,
    FEATURE_FAMILIES,
    FEATURE_SET_VERSION,
    SnapshotConfig,
    build_snapshots,
)
from retentioniq.models.churn.ladder import (
    COMPLEXITY_ORDER,
    ChurnModel,
    GradientBoosting,
    Logistic,
    RandomForest,
    RecencyBaseline,
    TorchMLP,
)
from retentioniq.models.churn.latency import measure_latency_ms
from retentioniq.models.churn.metrics import (
    best_f1_threshold,
    bootstrap_ci,
    classification_metrics,
    cost_optimal_threshold,
    paired_bootstrap_diff,
    reliability_table,
)
from retentioniq.models.churn.selection import select_champion
from retentioniq.tracking import log_run

log = logging.getLogger(__name__)
LABEL = "inactive"
N_BOOT = 1000
RFM = FEATURE_FAMILIES["rfm"]


def load_events(database_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    q = "SELECT event_ts, visitor_id, event_type, item_id, category_id FROM marts.fact_events"
    return pd.read_sql(q, engine, parse_dates=["event_ts"])


def _splits(snap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(
        snap[snap["split"] == s].reset_index(drop=True)  # type: ignore[return-value]
        for s in ("train", "validation", "test")
    )


def ladder(features: list[str] = ALL_FEATURES, include_torch: bool = True) -> list[ChurnModel]:
    models: list[ChurnModel] = [
        RecencyBaseline(),
        Logistic(RFM, name="M1_logistic_rfm"),
        Logistic(features, name="M2_logistic_full"),
        RandomForest(features),
        GradientBoosting(features),
    ]
    if include_torch:
        models.append(TorchMLP(features))
    return models


class Calibrated:
    """Isotonic calibration fit on validation, kept only if it lowers cross-fitted Brier."""

    def __init__(self, p_val: np.ndarray, y_val: np.ndarray) -> None:
        rng = np.random.default_rng(SEED)
        fold = rng.integers(0, 2, len(y_val))
        raw, cal = [], []
        for k in (0, 1):
            iso = IsotonicRegression(out_of_bounds="clip").fit(p_val[fold != k], y_val[fold != k])
            raw.append(brier_score_loss(y_val[fold == k], p_val[fold == k]))
            cal.append(brier_score_loss(y_val[fold == k], iso.predict(p_val[fold == k])))
        self.applied = float(np.mean(cal)) < float(np.mean(raw))
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
        self.val_brier_raw, self.val_brier_calibrated = float(np.mean(raw)), float(np.mean(cal))

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.iso.predict(p)) if self.applied else p


def _segments(test: pd.DataFrame) -> dict[str, np.ndarray]:
    tercile = test["sessions_obs"].quantile(1 / 3)
    segs = {
        "all": np.ones(len(test), dtype=bool),
        "new_visitors": (test["tenure_days"] < 14).to_numpy(),
        "low_activity": (test["sessions_obs"] <= tercile).to_numpy(),
        "purchasers": (test["ever_purchased"] == 1).to_numpy(),
        "non_purchasers": (test["ever_purchased"] == 0).to_numpy(),
    }
    for cat in test["top_category_id"].value_counts().head(5).index:
        segs[f"category_{int(cat)}"] = (test["top_category_id"] == cat).to_numpy()
    return segs


def run_ladder(
    snap: pd.DataFrame, out: Path, include_torch: bool = True
) -> tuple[pd.DataFrame, dict[str, Any], ChurnModel, Calibrated]:
    train, val, test = _splits(snap)
    y_tr, y_va, y_te = (d[LABEL].to_numpy() for d in (train, val, test))
    groups = test["visitor_id"].to_numpy()

    rows, calibration, preds = [], {}, {}
    val_preds: dict[str, np.ndarray] = {}
    fitted: dict[str, tuple[ChurnModel, Calibrated]] = {}
    for model in ladder(include_torch=include_torch):
        log.info("fitting %s", model.name)
        model.fit(train, y_tr, val, y_va)
        p_val = model.predict_proba(val)
        val_preds[model.name] = p_val
        cal = Calibrated(p_val, y_va)
        thr = best_f1_threshold(y_va, cal(p_val))
        p_test = cal(model.predict_proba(test))
        preds[model.name] = p_test
        fitted[model.name] = (model, cal)

        m = classification_metrics(y_te, p_test, threshold=thr)
        m["pr_auc_ci_low"], m["pr_auc_ci_high"] = bootstrap_ci(
            y_te, p_test, average_precision_score, groups, N_BOOT, SEED
        )

        def _ret_score(y_chunk: np.ndarray, p_chunk: np.ndarray) -> float:
            y_ret = (np.asarray(y_chunk) == 0).astype(int)
            if y_ret.min() == y_ret.max():
                return 0.0
            return float(average_precision_score(y_ret, 1.0 - np.asarray(p_chunk, dtype=float)))

        m["pr_auc_returners_ci_low"], m["pr_auc_returners_ci_high"] = bootstrap_ci(
            y_te, p_test, _ret_score, groups, N_BOOT, SEED
        )
        # Raw scores: the isotonic map is fit on validation, so calibrated validation scores
        # would be in-sample (and its ties change PR-AUC differently per model).
        m["val_pr_auc"] = float(average_precision_score(y_va, p_val))
        rows.append({"model": model.name, **m, "params": json.dumps(model.params)})
        calibration[model.name] = {
            "isotonic_applied": cal.applied,
            "val_brier_raw": cal.val_brier_raw,
            "val_brier_calibrated": cal.val_brier_calibrated,
            "test_brier": m["brier"],
            "test_ece": m["ece"],
            "reliability": reliability_table(y_te, p_test),
        }
        log_run(
            "retentioniq-churn",
            model.name,
            params=model.params,
            metrics=m,
            tags={"feature_set": FEATURE_SET_VERSION, "split": "test"},
        )

    comparison = pd.DataFrame(rows)
    # Paired differences vs. the M1 baseline and the M4 reference (PROJECT_SPEC.md §5.5, §7.2).
    for ref in ("M1_logistic_rfm", "M4_lightgbm"):
        lows, highs = [], []
        for name in comparison["model"]:
            lo, hi = paired_bootstrap_diff(
                y_te, preds[name], preds[ref], average_precision_score, groups, N_BOOT, SEED
            )
            lows.append(lo)
            highs.append(hi)
        comparison[f"delta_pr_auc_vs_{ref}_ci_low"] = lows
        comparison[f"delta_pr_auc_vs_{ref}_ci_high"] = highs

    # Test-set pairwise deltas, for reporting only; selection below never sees them.
    pairwise_rows = []
    models_present = [m for m in COMPLEXITY_ORDER if m in comparison["model"].to_numpy()]
    for i, model_a in enumerate(models_present):
        for model_b in models_present[:i]:
            score_a = float(comparison.loc[comparison["model"] == model_a, "pr_auc"].iloc[0])
            score_b = float(comparison.loc[comparison["model"] == model_b, "pr_auc"].iloc[0])
            delta = score_a - score_b
            lo, hi = paired_bootstrap_diff(
                y_te, preds[model_a], preds[model_b], average_precision_score, groups, N_BOOT, SEED
            )
            pairwise_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta_pr_auc": delta,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(not (lo <= 0 <= hi)),
                }
            )
    pd.DataFrame(pairwise_rows).to_csv(out / "churn_pairwise_deltas.csv", index=False)

    # Serving-cost eligibility (spec v0.6.3): single-record latency on the API's code path,
    # measured on validation rows.
    latency = {
        name: measure_latency_ms(mdl, cal_, ALL_FEATURES, val)
        for name, (mdl, cal_) in fitted.items()
    }
    comparison["serving_p50_ms"] = comparison["model"].map(lambda n: latency[n]["p50_ms"])
    comparison["serving_p95_ms"] = comparison["model"].map(lambda n: latency[n]["p95_ms"])

    # Complexity rule (spec §5.5, v0.6.2): decided on validation only.
    selection = select_champion(
        y_va,
        val_preds,
        COMPLEXITY_ORDER,
        groups=val["visitor_id"].to_numpy(),
        n_boot=N_BOOT,
        seed=SEED,
        latency_p95_ms={name: v["p95_ms"] for name, v in latency.items()},
    )
    champion = selection.selected_champion
    comparison["is_champion"] = comparison["model"] == champion
    pd.DataFrame(
        {"visitor_id": val["visitor_id"].to_numpy(), LABEL: y_va}
        | {name: p for name, p in val_preds.items()}
    ).to_parquet(out / "churn_validation_predictions.parquet")

    chosen = next(
        (c for c in selection.candidates_checked if c["model"] == champion),
        {"delta_pr_auc": 0.0, "ci_low": None, "ci_high": None, "tied_with_best": True},
    )
    selection_reasoning = {
        "selection_split": "validation",
        **selection.to_dict(),
        "delta_vs_best": {
            "delta_pr_auc": chosen["delta_pr_auc"],
            "ci_low": chosen["ci_low"],
            "ci_high": chosen["ci_high"],
            "significant": not chosen["tied_with_best"],
        },
        "is_lightgbm": champion == "M4_lightgbm",
    }
    (out / "churn_selection.json").write_text(json.dumps(selection_reasoning, indent=2) + "\n")

    p_va_champ = fitted[champion][1](fitted[champion][0].predict_proba(val))
    p_te_champ = preds[champion]

    # Cost-optimal operating threshold scenarios (§1.4)
    threshold_scenarios = []
    save_rate = 0.15
    for cost in (0.50, 1.00, 2.00):
        for val_saved in (20.0, 50.0, 100.0):
            thr = cost_optimal_threshold(y_va, p_va_champ, cost, val_saved, save_rate)
            contacted = p_te_champ >= thr
            share_contacted = float(np.mean(contacted))
            profit = float(
                np.sum(y_te[contacted] * save_rate * val_saved) - np.sum(contacted) * cost
            )
            threshold_scenarios.append(
                {
                    "cost_contact": cost,
                    "value_saved": val_saved,
                    "save_rate": save_rate,
                    "threshold": thr,
                    "share_contacted": share_contacted,
                    "expected_profit": profit,
                    "scenario_note": "SCENARIO - hypothetical cost/value, not observed data",
                }
            )
    pd.DataFrame(threshold_scenarios).to_csv(out / "churn_threshold_scenarios.csv", index=False)

    # Error analysis by segment (§1.5)
    thr_f1 = float(comparison.loc[comparison["model"] == champion, "threshold"].iloc[0])
    pred_inactive = p_te_champ >= thr_f1

    def _error_slice(dim: str, seg: str, mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        if n == 0:
            return {
                "dimension": dim,
                "segment": seg,
                "n": 0,
                "actual_inactive": 0,
                "actual_active": 0,
                "fp": 0,
                "fn": 0,
                "fpr": float("nan"),
                "fnr": float("nan"),
            }
        y_slice = y_te[mask]
        p_slice = pred_inactive[mask]
        n_inactive = int((y_slice == 1).sum())
        n_active = int((y_slice == 0).sum())
        fp = int(((y_slice == 0) & p_slice).sum())
        fn = int(((y_slice == 1) & (~p_slice)).sum())
        return {
            "dimension": dim,
            "segment": seg,
            "n": n,
            "actual_inactive": n_inactive,
            "actual_active": n_active,
            "fp": fp,
            "fn": fn,
            "fpr": fp / n_active if n_active > 0 else float("nan"),
            "fnr": fn / n_inactive if n_inactive > 0 else float("nan"),
        }

    err_rows = []
    tenure = test["tenure_days"].to_numpy()
    err_rows.append(_error_slice("tenure_days", "<7d", tenure < 7))
    err_rows.append(_error_slice("tenure_days", "7-14d", (tenure >= 7) & (tenure < 14)))
    err_rows.append(_error_slice("tenure_days", "14-30d", (tenure >= 14) & (tenure < 30)))
    err_rows.append(_error_slice("tenure_days", "30d+", tenure >= 30))

    s_obs = test["sessions_obs"].to_numpy()
    q1 = np.quantile(s_obs, 1 / 3)
    q2 = np.quantile(s_obs, 2 / 3)
    err_rows.append(_error_slice("sessions_obs", "low (T1)", s_obs <= q1))
    err_rows.append(_error_slice("sessions_obs", "medium (T2)", (s_obs > q1) & (s_obs <= q2)))
    err_rows.append(_error_slice("sessions_obs", "high (T3)", s_obs > q2))

    purchased = test["ever_purchased"].to_numpy()
    err_rows.append(_error_slice("ever_purchased", "non_purchasers (0)", purchased == 0))
    err_rows.append(_error_slice("ever_purchased", "purchasers (1)", purchased == 1))

    pd.DataFrame(err_rows).to_csv(out / "churn_error_analysis.csv", index=False)

    seg_rows = []
    for seg, mask in _segments(test).items():
        if mask.sum() < 50 or y_te[mask].min() == y_te[mask].max():
            continue
        for name in ("M1_logistic_rfm", champion):
            m = classification_metrics(y_te[mask], preds[name][mask])
            seg_rows.append(
                {
                    "segment": seg,
                    "model": name,
                    "n": int(mask.sum()),
                    "base_rate": m["base_rate"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "brier": m["brier"],
                }
            )
    pd.DataFrame(seg_rows).to_csv(out / "churn_segment_metrics.csv", index=False)

    model, cal = fitted[champion]
    lgbm, lgbm_cal = fitted["M4_lightgbm"]
    joblib.dump(
        {
            "model": lgbm,
            "calibrator": lgbm_cal,
            "features": ALL_FEATURES,
            "feature_set": FEATURE_SET_VERSION,
        },
        out / "churn_lgbm.joblib",
    )
    test.assign(p_inactive=preds[champion]).to_parquet(out / "churn_test_predictions.parquet")
    test[["visitor_id", "cutoff"]].assign(p_inactive=preds["M1_logistic_rfm"]).to_parquet(
        out / "churn_baseline_predictions.parquet"
    )
    return comparison, calibration, model, cal


def run_ablation(snap: pd.DataFrame) -> pd.DataFrame:
    train, val, test = _splits(snap)
    y_tr, y_va, y_te = (d[LABEL].to_numpy() for d in (train, val, test))
    groups = test["visitor_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    for model_cls, model_name in ((GradientBoosting, "M4_lightgbm"), (Logistic, "M2_logistic")):
        features: list[str] = []
        prev_pred: np.ndarray | None = None
        for family in ABLATION_ORDER:
            features = features + FEATURE_FAMILIES[family]
            m = model_cls(list(features)).fit(train, y_tr, val, y_va)
            pred = m.predict_proba(test)
            row: dict[str, Any] = {
                "model": model_name,
                "step": family,
                "features": "+".join(ABLATION_ORDER[: ABLATION_ORDER.index(family) + 1]),
                "n_features": len(features),
                "val_pr_auc": float(average_precision_score(y_va, m.predict_proba(val))),
                "test_pr_auc": float(average_precision_score(y_te, pred)),
            }
            if prev_pred is not None:
                row["delta_test_pr_auc"] = row["test_pr_auc"] - rows[-1]["test_pr_auc"]
                row["delta_ci_low"], row["delta_ci_high"] = paired_bootstrap_diff(
                    y_te, pred, prev_pred, average_precision_score, groups, N_BOOT, SEED
                )
            rows.append(row)
            prev_pred = pred
    return pd.DataFrame(rows)


SENSITIVITY: dict[str, SnapshotConfig] = {
    "primary": SnapshotConfig(),
    "horizon_21d": SnapshotConfig(horizon_days=21),
    "observation_21d": SnapshotConfig(observation_days=21),
    "session_gap_15m": SnapshotConfig(session_gap_minutes=15),
    "session_gap_60m": SnapshotConfig(session_gap_minutes=60),
    "min_sessions_1": SnapshotConfig(min_sessions=1),
    "min_sessions_3": SnapshotConfig(min_sessions=3),
}


def run_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, cfg in SENSITIVITY.items():
        snap, cutoffs = build_snapshots(events, cfg)
        train, val, test = _splits(snap)
        y_tr, y_va, y_te = (d[LABEL].to_numpy() for d in (train, val, test))
        for model in (Logistic(RFM, name="M1_logistic_rfm"), GradientBoosting(ALL_FEATURES)):
            model.fit(train, y_tr, val, y_va)
            m = classification_metrics(y_te, model.predict_proba(test))
            rows.append(
                {
                    "setting": name,
                    **asdict(cfg),
                    "model": model.name,
                    "n_test": len(test),
                    "base_rate": m["base_rate"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                    "test_cutoff": str(cutoffs["test"][0].date()),
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="setting", columns="model", values="pr_auc")
    df["ranking_stable"] = bool((pivot["M4_lightgbm"] > pivot["M1_logistic_rfm"]).all())
    return df


def run(
    database_url: str,
    artifacts: Path,
    model_dir: Path,
    include_torch: bool = True,
    sensitivity: bool = True,
) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(database_url)
    cfg = SnapshotConfig()
    snap, cutoffs = build_snapshots(events, cfg)
    manifest = {
        "feature_set": FEATURE_SET_VERSION,
        "config": asdict(cfg),
        "cutoffs": {k: [str(c.date()) for c in v] for k, v in cutoffs.items()},
        "rows": snap["split"].value_counts().to_dict(),
        "base_rate": snap.groupby("split")[LABEL].mean().round(4).to_dict(),
        "data_start": str(events["event_ts"].min()),
        "data_end": str(events["event_ts"].max()),
    }
    (artifacts / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    snap.to_parquet(model_dir / "rr_snapshots.parquet")

    comparison, calibration, champion, cal = run_ladder(snap, artifacts, include_torch)
    (artifacts / "churn_lgbm.joblib").replace(model_dir / "churn_lgbm.joblib")
    comparison.to_csv(artifacts / "churn_model_comparison.csv", index=False)
    (artifacts / "calibration_metrics.json").write_text(json.dumps(calibration, indent=2) + "\n")
    joblib.dump(
        {
            "model": champion,
            "calibrator": cal,
            "features": ALL_FEATURES,
            "feature_set": FEATURE_SET_VERSION,
            "config": asdict(cfg),
        },
        model_dir / "churn_champion.joblib",
    )

    run_ablation(snap).to_csv(artifacts / "feature_ablation.csv", index=False)
    if sensitivity:
        run_sensitivity(events).to_csv(artifacts / "churn_sensitivity.csv", index=False)
    return {"champion": champion.name, **manifest}


def score_frame(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Score feature rows with a saved champion bundle (used by the API)."""
    p = bundle["model"].predict_proba(features[bundle["features"]])
    return np.asarray(bundle["calibrator"](p))
