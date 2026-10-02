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
    ChurnModel,
    GradientBoosting,
    Logistic,
    RandomForest,
    RecencyBaseline,
    TorchMLP,
)
from retentioniq.models.churn.metrics import (
    best_f1_threshold,
    bootstrap_ci,
    classification_metrics,
    paired_bootstrap_diff,
    reliability_table,
)
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
    fitted: dict[str, tuple[ChurnModel, Calibrated]] = {}
    for model in ladder(include_torch=include_torch):
        log.info("fitting %s", model.name)
        model.fit(train, y_tr, val, y_va)
        p_val = model.predict_proba(val)
        cal = Calibrated(p_val, y_va)
        thr = best_f1_threshold(y_va, cal(p_val))
        p_test = cal(model.predict_proba(test))
        preds[model.name] = p_test
        fitted[model.name] = (model, cal)

        m = classification_metrics(y_te, p_test, threshold=thr)
        m["pr_auc_ci_low"], m["pr_auc_ci_high"] = bootstrap_ci(
            y_te, p_test, average_precision_score, groups, N_BOOT, SEED
        )
        m["val_pr_auc"] = float(average_precision_score(y_va, cal(p_val)))
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

    # Champion: highest validation PR-AUC. M5 must also beat M4 significantly (§5.5).
    ranked = comparison.sort_values("val_pr_auc", ascending=False)
    champion = ranked.iloc[0]["model"]
    if champion == "M5_pytorch_mlp":
        m5 = comparison.set_index("model").loc["M5_pytorch_mlp"]
        if not m5["delta_pr_auc_vs_M4_lightgbm_ci_low"] > 0:
            champion = "M4_lightgbm"
    comparison["is_champion"] = comparison["model"] == champion

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
    test.assign(p_inactive=preds[champion]).to_parquet(out / "churn_test_predictions.parquet")
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
