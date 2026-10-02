"""Champion/challenger promotion gates (PROJECT_SPEC.md §9.1).

A challenger is promoted only if every gate passes:
1. PR-AUC non-inferior: paired bootstrap CI of (challenger - champion) lies above -0.005.
2. Brier <= champion Brier x 1.05.
3. No evaluation segment loses more than 0.03 PR-AUC.
4. Data validation passed (ingestion reject gates and dbt tests).
5. The serving schema contract holds (feature list matches the API's).
With no champion yet, the reference is the M1 RFM-logistic baseline on the same test set.
"""

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from retentioniq import SEED
from retentioniq.features.retailrocket import ALL_FEATURES
from retentioniq.models.churn.metrics import paired_bootstrap_diff

NON_INFERIORITY_MARGIN = -0.005
BRIER_TOLERANCE = 1.05
MAX_SEGMENT_DROP = 0.03


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    promote: bool
    challenger: str
    reference: str
    gates: list[GateResult]
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _segment_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    tercile = df["sessions_obs"].quantile(1 / 3)
    return {
        "new_visitors": (df["tenure_days"] < 14).to_numpy(),
        "low_activity": (df["sessions_obs"] <= tercile).to_numpy(),
        "purchasers": (df["ever_purchased"] == 1).to_numpy(),
        "non_purchasers": (df["ever_purchased"] == 0).to_numpy(),
    }


def evaluate_gates(
    y: np.ndarray,
    p_challenger: np.ndarray,
    p_reference: np.ndarray,
    frame: pd.DataFrame,
    data_validation_ok: bool,
    serving_features: list[str],
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
) -> list[GateResult]:
    lo, hi = paired_bootstrap_diff(
        y, p_challenger, p_reference, average_precision_score, groups, n_boot, SEED
    )
    gates = [
        GateResult(
            "pr_auc_non_inferior",
            lo > NON_INFERIORITY_MARGIN,
            {"delta_ci": [lo, hi], "margin": NON_INFERIORITY_MARGIN},
        ),
    ]
    b_ch = float(brier_score_loss(y, p_challenger))
    b_ref = float(brier_score_loss(y, p_reference))
    gates.append(
        GateResult(
            "brier", b_ch <= b_ref * BRIER_TOLERANCE, {"challenger": b_ch, "reference": b_ref}
        )
    )
    drops = {}
    for seg, m in _segment_masks(frame).items():
        if m.sum() < 50 or y[m].min() == y[m].max():
            continue
        drops[seg] = float(
            average_precision_score(y[m], p_reference[m])
            - average_precision_score(y[m], p_challenger[m])
        )
    gates.append(
        GateResult(
            "segment_degradation",
            all(d <= MAX_SEGMENT_DROP for d in drops.values()),
            {"pr_auc_drop": drops, "max_allowed": MAX_SEGMENT_DROP},
        )
    )
    gates.append(GateResult("data_validation", data_validation_ok))
    gates.append(
        GateResult(
            "schema_contract",
            list(serving_features) == ALL_FEATURES,
            {"n_features": len(serving_features)},
        )
    )
    return gates


def data_validation_ok(artifacts: Path, dbt_target: Path) -> tuple[bool, dict[str, Any]]:
    """dbt tests must not have errors; the ingestion ledger enforces reject-rate gates."""
    results = dbt_target / "run_results.json"
    if not results.exists():
        return False, {"reason": "dbt run_results.json not found; run `make dbt`"}
    rr = json.loads(results.read_text())["results"]
    failed = [r["unique_id"] for r in rr if r["status"] in ("error", "fail")]
    return not failed, {"dbt_failed": failed, "dbt_nodes": len(rr)}


def decide(artifacts: Path, model_dir: Path, dbt_target: Path) -> Decision:
    preds = pd.read_parquet(artifacts / "churn_test_predictions.parquet")
    y = preds["inactive"].to_numpy()
    comparison = pd.read_csv(artifacts / "churn_model_comparison.csv")
    challenger = str(comparison.loc[comparison["is_champion"], "model"].iloc[0])

    p_ref: np.ndarray | None = None
    reference = ""
    champion_preds = model_dir / "production" / "test_predictions.parquet"
    if champion_preds.exists():
        ref = pd.read_parquet(champion_preds)[["visitor_id", "cutoff", "p_inactive"]]
        merged = preds.merge(ref, on=["visitor_id", "cutoff"], suffixes=("", "_ref"))
        if len(merged) == len(preds):  # same test snapshot -> compare to production
            p_ref, reference = merged["p_inactive_ref"].to_numpy(), "production"
    if p_ref is None:  # first deployment, or a new snapshot: compare to the M1 baseline
        base = pd.read_parquet(artifacts / "churn_baseline_predictions.parquet")
        p_ref, reference = base["p_inactive"].to_numpy(), "M1_logistic_rfm"

    ok, detail = data_validation_ok(artifacts, dbt_target)
    import joblib

    bundle = joblib.load(model_dir / "churn_champion.joblib")
    gates = evaluate_gates(
        y,
        preds["p_inactive"].to_numpy(),
        p_ref,
        preds,
        ok,
        bundle["features"],
        preds["visitor_id"].to_numpy(),
    )
    gates[3].detail = detail
    return Decision(all(g.passed for g in gates), challenger, reference, gates)


def promote(artifacts: Path, model_dir: Path, dbt_target: Path) -> Decision:
    decision = decide(artifacts, model_dir, dbt_target)
    (artifacts / "promotion_decision.json").write_text(
        json.dumps(decision.to_dict(), indent=2) + "\n"
    )
    if decision.promote:
        prod = model_dir / "production"
        prod.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir / "churn_champion.joblib", prod / "churn_champion.joblib")
        shutil.copy2(
            artifacts / "churn_test_predictions.parquet", prod / "test_predictions.parquet"
        )
        _register_mlflow(model_dir / "churn_champion.joblib", decision)
    return decision


def _register_mlflow(bundle_path: Path, decision: Decision) -> None:
    """Log the bundle as an MLflow pyfunc model and point the `champion` alias at it."""
    from retentioniq.tracking import mlflow_available

    if not mlflow_available():
        return
    import mlflow
    from mlflow import MlflowClient

    from retentioniq import SPEC_VERSION
    from retentioniq.config import settings
    from retentioniq.tracking import git_sha

    class ChurnPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
        def load_context(self, context: Any) -> None:
            import joblib

            self.bundle = joblib.load(context.artifacts["bundle"])

        def predict(self, context: Any, model_input: pd.DataFrame) -> np.ndarray:
            from retentioniq.training.churn import score_frame

            return score_frame(self.bundle, model_input)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("retentioniq-churn")
    with mlflow.start_run(run_name=f"promote-{decision.challenger}"):
        mlflow.set_tags(
            {
                "spec_version": SPEC_VERSION,
                "git_sha": git_sha(),
                "promotion_reference": decision.reference,
            }
        )
        info = mlflow.pyfunc.log_model(
            name="model", python_model=ChurnPyfunc(), artifacts={"bundle": str(bundle_path)}
        )
    mv = mlflow.register_model(info.model_uri, "retentioniq-churn")
    MlflowClient().set_registered_model_alias("retentioniq-churn", "champion", mv.version)
