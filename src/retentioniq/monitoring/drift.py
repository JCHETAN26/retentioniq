"""Data and model monitoring (PROJECT_SPEC.md §9.3).

Compares a *current* scoring population against the *reference* (training) population:
- data:  schema, missingness change, per-feature PSI, freshness
- model: prediction-distribution PSI, calibration (ECE) and PR-AUC once labels arrive,
         segment-level PR-AUC
Each signal is classified ok / warn / alert using the thresholds in the spec. Evidently is
used for the HTML report when installed; the numbers in the JSON come from this module.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from retentioniq.features.retailrocket import ALL_FEATURES
from retentioniq.models.churn.metrics import expected_calibration_error

THRESHOLDS = {
    "feature_psi": (0.10, 0.25),
    "prediction_psi": (0.10, 0.25),
    "missingness_increase_pp": (5.0, 15.0),
    "freshness_days": (1.0, 3.0),
    "pr_auc_drop": (0.03, 0.05),
    "ece": (0.05, 0.10),
}


def level(value: float, signal: str) -> str:
    warn, alert = THRESHOLDS[signal]
    if value > alert:
        return "alert"
    return "warn" if value > warn else "ok"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index with reference-quantile bins (eps-smoothed)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    eps = 1e-6
    rp = r / r.sum() + eps
    cp = c / c.sum() + eps
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def monitor(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_reference: np.ndarray,
    p_current: np.ndarray,
    reference_pr_auc: float,
    data_as_of: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": (now or datetime.now(UTC)).isoformat(),
        "n_reference": len(reference),
        "n_current": len(current),
    }
    missing_cols = sorted(set(ALL_FEATURES) - set(current.columns))
    report["schema"] = {"missing_columns": missing_cols, "level": "alert" if missing_cols else "ok"}

    features = {}
    for f in ALL_FEATURES:
        if f in missing_cols:
            continue
        miss_pp = 100 * (current[f].isna().mean() - reference[f].isna().mean())
        value = psi(reference[f].to_numpy(), current[f].to_numpy())
        features[f] = {
            "psi": value,
            "psi_level": level(value, "feature_psi"),
            "missingness_increase_pp": float(miss_pp),
            "missingness_level": level(float(miss_pp), "missingness_increase_pp"),
        }
    report["features"] = features

    pred_psi = psi(p_reference, p_current)
    report["predictions"] = {
        "psi": pred_psi,
        "level": level(pred_psi, "prediction_psi"),
        "mean_reference": float(np.mean(p_reference)),
        "mean_current": float(np.mean(p_current)),
    }
    if data_as_of is not None:
        age = ((now or datetime.now(UTC)) - data_as_of).total_seconds() / 86_400
        report["freshness"] = {
            "data_as_of": data_as_of.isoformat(),
            "age_days": age,
            "level": level(age, "freshness_days"),
        }

    if "inactive" in current.columns and current["inactive"].notna().all():
        y = current["inactive"].to_numpy()
        pr = float(average_precision_score(y, p_current))
        ece = expected_calibration_error(y, np.clip(p_current, 0, 1))
        report["performance"] = {
            "pr_auc": pr,
            "reference_pr_auc": reference_pr_auc,
            "pr_auc_drop": reference_pr_auc - pr,
            "pr_auc_level": level(reference_pr_auc - pr, "pr_auc_drop"),
            "ece": ece,
            "ece_level": level(ece, "ece"),
        }
        tercile = current["sessions_obs"].quantile(1 / 3)
        segs = {
            "new_visitors": current["tenure_days"] < 14,
            "low_activity": current["sessions_obs"] <= tercile,
            "purchasers": current["ever_purchased"] == 1,
        }
        report["segments"] = {
            name: float(average_precision_score(y[m.to_numpy()], p_current[m.to_numpy()]))
            for name, m in segs.items()
            if m.sum() >= 50 and 0 < y[m.to_numpy()].mean() < 1
        }

    levels = [report["schema"]["level"], report["predictions"]["level"]]
    levels += [v["psi_level"] for v in features.values()]
    levels += [v["missingness_level"] for v in features.values()]
    for key in ("freshness", "performance"):
        if key in report:
            levels += [v for k, v in report[key].items() if k.endswith("level")]
    report["status"] = "alert" if "alert" in levels else ("warn" if "warn" in levels else "ok")
    report["drifted_features"] = sorted(f for f, v in features.items() if v["psi_level"] != "ok")
    return report


def run(artifacts: Path, model_dir: Path) -> dict[str, Any]:
    """Reference = training snapshots; current = the test snapshot (stand-in for live traffic,
    with labels available so performance monitoring can be exercised)."""
    import joblib

    from retentioniq.training.churn import score_frame

    snap = pd.read_parquet(model_dir / "rr_snapshots.parquet")
    bundle = joblib.load(model_dir / "churn_champion.joblib")
    ref = snap[snap["split"] == "train"]
    cur = snap[snap["split"] == "test"]
    p_ref, p_cur = score_frame(bundle, ref), score_frame(bundle, cur)
    val = snap[snap["split"] == "validation"]
    ref_pr = float(average_precision_score(val["inactive"], score_frame(bundle, val)))
    data_as_of = pd.Timestamp(cur["cutoff"].max()).to_pydatetime().replace(tzinfo=UTC)
    report = monitor(
        ref, cur, p_ref, p_cur, ref_pr, data_as_of=data_as_of, now=data_as_of
    )  # replayed: "now" is the test cutoff
    (artifacts / "monitoring_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _evidently_html(ref[ALL_FEATURES], cur[ALL_FEATURES], artifacts / "monitoring_drift.html")
    return {"status": report["status"], "drifted_features": report["drifted_features"]}


def _evidently_html(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> None:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return
    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    snapshot.save_html(str(path))
