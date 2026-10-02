"""SHAP explanations and error analysis for the inactivity model (§3 Q4, Phase 16).

Uses the LightGBM model (M4) because TreeSHAP is exact for it; if another model is the
champion, the explanation still describes the strongest tree model on the same features.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from retentioniq import SEED
from retentioniq.features.retailrocket import ALL_FEATURES, FEATURE_FAMILIES

FAMILY_OF = {f: fam for fam, feats in FEATURE_FAMILIES.items() for f in feats}


def shap_values(model: Any, x: pd.DataFrame) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def top_drivers(row_shap: np.ndarray, row_x: pd.Series, k: int = 5) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(row_shap))[:k]
    return [
        {
            "feature": ALL_FEATURES[i],
            "family": FAMILY_OF[ALL_FEATURES[i]],
            "value": float(row_x.iloc[i]),
            "shap": float(row_shap[i]),
            "direction": "raises risk" if row_shap[i] > 0 else "lowers risk",
        }
        for i in order
    ]


def run(model_dir: Path, artifacts: Path) -> dict[str, Any]:
    bundle = joblib.load(model_dir / "churn_lgbm.joblib")
    lgbm = bundle["model"].model_
    preds = pd.read_parquet(artifacts / "churn_test_predictions.parquet")
    x = preds[ALL_FEATURES]
    sample = x.sample(min(20_000, len(x)), random_state=SEED)
    sv = shap_values(lgbm, sample)

    glob = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
            "mean_shap": sv.mean(axis=0),
        }
    )
    glob["family"] = glob["feature"].map(FAMILY_OF)
    # Direction: correlation between feature value and its SHAP contribution.
    glob["value_shap_corr"] = [
        float(np.corrcoef(sample[f], sv[:, i])[0, 1]) if sample[f].std() > 0 else 0.0
        for i, f in enumerate(ALL_FEATURES)
    ]
    glob = glob.sort_values("mean_abs_shap", ascending=False)
    glob.to_csv(artifacts / "shap_global.csv", index=False)

    # Error analysis at the validation-chosen threshold of the champion.
    comparison = pd.read_csv(artifacts / "churn_model_comparison.csv")
    champ = comparison[comparison["is_champion"]].iloc[0]
    thr = float(champ["threshold"])
    pred_pos = preds["p_inactive"] >= thr
    actual = preds["inactive"] == 1
    preds["outcome"] = np.select(
        [pred_pos & actual, pred_pos & ~actual, ~pred_pos & actual], ["TP", "FP", "FN"], "TN"
    )
    profile = preds.groupby("outcome")[ALL_FEATURES].median().T
    profile["family"] = profile.index.map(FAMILY_OF)
    profile.to_csv(artifacts / "error_analysis_profiles.csv")

    examples: dict[str, list[dict[str, Any]]] = {}
    for kind in ("FP", "FN", "TP"):
        subset = preds[preds["outcome"] == kind]
        if subset.empty:
            continue
        # Most confident mistakes are the most informative to read.
        subset = subset.sort_values("p_inactive", ascending=(kind == "FN")).head(10)
        sv_sub = shap_values(lgbm, subset[ALL_FEATURES])
        records = subset.to_dict("records")
        examples[kind] = [
            {
                "visitor_id": int(r["visitor_id"]),
                "p_inactive": float(r["p_inactive"]),
                "inactive": int(r["inactive"]),
                "drivers": top_drivers(sv_sub[i], subset[ALL_FEATURES].iloc[i]),
            }
            for i, r in enumerate(records)
        ]
    counts = preds["outcome"].value_counts().to_dict()
    out = {
        "model_explained": "M4_lightgbm",
        "champion": champ["model"],
        "threshold": thr,
        "confusion": {k: int(v) for k, v in counts.items()},
        "examples": examples,
    }
    (artifacts / "error_analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    return {"top_features": glob["feature"].head(5).tolist(), "confusion": out["confusion"]}
