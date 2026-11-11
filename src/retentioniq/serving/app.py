"""RetentionIQ serving API (Phase 22).

Endpoints
- GET  /health                      liveness + which models are loaded
- GET  /v1/models                   model metadata and headline test metrics
- POST /v1/churn/predict            one visitor's features -> P(inactive), tier, drivers
- POST /v1/churn/batch              up to 10k feature rows
- GET  /v1/churn/visitors/{id}      demo lookup on the test snapshot (risk, drivers, survival)
- POST /v1/uplift/score             Criteo features -> estimated uplift and targeting decision
- GET  /v1/artifacts/{name}         evaluation artifacts (JSON/CSV) for the web app
- GET  /metrics                     Prometheus metrics
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, create_model

from retentioniq import SPEC_VERSION, __version__
from retentioniq.config import REPO_ROOT, settings
from retentioniq.features.retailrocket import ALL_FEATURES
from retentioniq.models.churn.latency import score_one
from retentioniq.models.uplift.learners import FEATURES as UPLIFT_FEATURES

MODEL_DIR = Path(str(REPO_ROOT / "data" / "models"))
MAX_BATCH = 10_000
RISK_TIERS = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))

REQUESTS = Counter("riq_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram(
    "riq_request_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Histogram(
    "riq_churn_prediction", "Predicted P(inactive)", buckets=tuple(np.linspace(0.1, 1, 10))
)
UPLIFT_SCORES = Histogram(
    "riq_uplift_score",
    "Predicted uplift (visit)",
    buckets=(-0.01, 0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)


class State:
    churn: dict[str, Any] | None = None
    lgbm: dict[str, Any] | None = None
    uplift: dict[str, Any] | None = None
    survival: dict[str, Any] | None = None
    explainer: Any = None
    snapshot: pd.DataFrame | None = None
    survival_cov: pd.DataFrame | None = None


state = State()


def _load(name: str) -> dict[str, Any] | None:
    path = MODEL_DIR / name
    return joblib.load(path) if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    state.churn = _load("churn_champion.joblib")
    state.lgbm = _load("churn_lgbm.joblib")
    state.uplift = _load("uplift_champion.joblib")
    state.survival = _load("survival_cox.joblib")
    if state.lgbm is not None:
        # LightGBM's built-in TreeSHAP (pred_contrib) gives exact SHAP values in log-odds,
        # identical to shap.TreeExplainer, without shipping the shap package in the image.
        state.explainer = state.lgbm["model"].model_.booster_
    cov = MODEL_DIR / "survival_covariates.parquet"
    if cov.exists():
        state.survival_cov = pd.read_parquet(cov)
    preds = settings.artifacts_dir / "churn_test_predictions.parquet"
    if preds.exists():
        state.snapshot = pd.read_parquet(preds).set_index("visitor_id")
    yield


app = FastAPI(title="RetentionIQ API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def observe(request: Request, call_next):  # type: ignore[no-untyped-def]
    route = request.scope.get("route")
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        endpoint = getattr(request.scope.get("route"), "path", route or request.url.path)
        LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(endpoint, str(status)).inc()


ChurnFeatures = create_model(  # type: ignore[call-overload]
    "ChurnFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{f: (float, Field(..., ge=-1.0 if "trend" in f else 0.0)) for f in ALL_FEATURES},
)
UpliftFeatures = create_model(  # type: ignore[call-overload]
    "UpliftFeatures",
    __config__=ConfigDict(extra="forbid"),
    **{f: (float, ...) for f in UPLIFT_FEATURES},
)


class Driver(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str


class ChurnPrediction(BaseModel):
    p_inactive: float
    risk_tier: str
    model: str
    drivers: list[Driver] = []


class BatchRequest(BaseModel):
    rows: list[ChurnFeatures]


class UpliftPrediction(BaseModel):
    uplift_visit: float
    uplift_conversion: float
    percentile: float
    treat_at_reach: dict[str, bool]
    strategy: str


def _require(bundle: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if bundle is None:
        raise HTTPException(503, f"{name} model not loaded; run `make evaluate` first")
    return bundle


def _tier(p: float) -> str:
    return next(name for threshold, name in RISK_TIERS if p >= threshold)


def _score_churn(frame: pd.DataFrame) -> np.ndarray:
    bundle = _require(state.churn, "churn")
    p = bundle["model"].predict_proba(frame[bundle["features"]])
    return np.clip(np.asarray(bundle["calibrator"](p), dtype=float), 0, 1)


def _score_row(values: dict[str, float]) -> float:
    """Single-record path; the same function gates models on latency (models/churn/latency)."""
    bundle = _require(state.churn, "churn")
    return score_one(bundle["model"], bundle["calibrator"], bundle["features"], values)


def _drivers(frame: pd.DataFrame, k: int = 5) -> list[list[Driver]]:
    if state.explainer is None:
        return [[] for _ in range(len(frame))]
    contrib = state.explainer.predict(
        frame[ALL_FEATURES].to_numpy(dtype=np.float64), pred_contrib=True
    )
    sv = np.asarray(contrib)[:, :-1]  # last column is the expected value (bias)
    out = []
    for i in range(len(frame)):
        order = np.argsort(-np.abs(sv[i]))[:k]
        out.append(
            [
                Driver(
                    feature=ALL_FEATURES[j],
                    value=float(frame.iloc[i][ALL_FEATURES[j]]),
                    shap=float(sv[i, j]),
                    direction="raises risk" if sv[i, j] > 0 else "lowers risk",
                )
                for j in order
            ]
        )
    return out


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "spec_version": SPEC_VERSION,
        "models": {
            "churn": state.churn is not None,
            "uplift": state.uplift is not None,
            "survival": state.survival is not None,
        },
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    out: dict[str, Any] = {}
    cmp_path = settings.artifacts_dir / "churn_model_comparison.csv"
    if state.churn is not None:
        info: dict[str, Any] = {
            "name": state.churn["model"].name,
            "feature_set": state.churn["feature_set"],
            "features": len(state.churn["features"]),
        }
        if cmp_path.exists():
            cmp = pd.read_csv(cmp_path).set_index("model")
            row = cmp.loc[state.churn["model"].name]
            info["test_metrics"] = {
                k: float(row[k]) for k in ("pr_auc", "roc_auc", "brier", "ece", "precision@10")
            }
        out["churn"] = info
    if state.uplift is not None:
        out["uplift"] = {"name": state.uplift["name"], "outcome": state.uplift["outcome"]}
    if state.survival is not None:
        out["survival"] = {"name": "cox_ph", "covariates": state.survival["covariates"]}
    return out


@app.post("/v1/churn/predict", response_model=ChurnPrediction)
def churn_predict(features: ChurnFeatures, explain: bool = True) -> ChurnPrediction:
    values = features.model_dump()
    p = _score_row(values)
    PREDICTIONS.observe(p)
    assert state.churn is not None
    frame = pd.DataFrame([values]) if explain else None
    return ChurnPrediction(
        p_inactive=p,
        risk_tier=_tier(p),
        model=state.churn["model"].name,
        drivers=_drivers(frame)[0] if frame is not None else [],
    )


@app.post("/v1/churn/batch")
def churn_batch(req: BatchRequest) -> dict[str, Any]:
    if len(req.rows) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} rows per request")
    frame = pd.DataFrame([r.model_dump() for r in req.rows])
    p = _score_churn(frame)
    for v in p:
        PREDICTIONS.observe(float(v))
    return {
        "n": len(p),
        "p_inactive": p.round(6).tolist(),
        "risk_tier": [_tier(float(v)) for v in p],
    }


@app.get("/v1/churn/visitors")
def list_visitors(tier: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if state.snapshot is None:
        raise HTTPException(503, "no snapshot loaded")
    s = state.snapshot.sort_values("p_inactive", ascending=False)
    if tier:
        s = s[[_tier(float(p)) == tier for p in s["p_inactive"]]]
    return [
        {
            "visitor_id": int(str(i)),
            "p_inactive": float(r["p_inactive"]),
            "risk_tier": _tier(float(r["p_inactive"])),
            "sessions_obs": float(r["sessions_obs"]),
            "recency_days": float(r["recency_days"]),
        }
        for i, r in s.head(min(limit, 200)).iterrows()
    ]


@app.get("/v1/churn/visitors/{visitor_id}")
def visitor(visitor_id: int) -> dict[str, Any]:
    if state.snapshot is None or visitor_id not in state.snapshot.index:
        raise HTTPException(404, "visitor not in the evaluation snapshot")
    row = state.snapshot.loc[[visitor_id]]
    p = float(_score_churn(row)[0])
    out: dict[str, Any] = {
        "visitor_id": visitor_id,
        "cutoff": str(row["cutoff"].iloc[0]),
        "p_inactive": p,
        "risk_tier": _tier(p),
        "actual_inactive": int(row["inactive"].iloc[0]),
        "features": {f: float(row[f].iloc[0]) for f in ALL_FEATURES},
        "drivers": [d.model_dump() for d in _drivers(row)[0]],
    }
    out["survival"] = _survival_for(row)
    return out


def _survival_for(row: pd.DataFrame) -> dict[str, Any] | None:
    """P(no 30-day inactivity gap starts within the next h days | active so far), from the Cox
    model with the visitor's first-session covariates (PROJECT_SPEC.md §4.2)."""
    if state.survival is None or state.survival_cov is None:
        return None
    vid = int(row.index[0])
    if vid not in state.survival_cov.index:
        return None
    cph = state.survival["model"]
    cov = state.survival_cov.loc[[vid], state.survival["covariates"]].copy()
    for c in state.survival["log_cols"]:
        cov[c] = np.log1p(cov[c])
    age = float(row["tenure_days"].iloc[0])
    horizons = [30, 60, 90]
    sf = cph.predict_survival_function(cov, times=horizons, conditional_after=[age])
    max_follow_up = float(state.survival_cov["duration_days"].max())
    return {
        "age_days": age,
        "survival": {f"{h}d": float(sf.iloc[k, 0]) for k, h in enumerate(horizons)},
        "beyond_observed_follow_up": [f"{h}d" for h in horizons if age + h > max_follow_up],
    }


@app.post("/v1/uplift/score", response_model=UpliftPrediction)
def uplift_score(features: UpliftFeatures) -> UpliftPrediction:
    bundle = _require(state.uplift, "uplift")
    frame = pd.DataFrame([features.model_dump()])
    tau = float(bundle["strategy"].score(frame)[0])
    tau_c = float(bundle["conversion_strategy"].score(frame)[0])
    UPLIFT_SCORES.observe(tau)
    q = np.asarray(bundle["tau_quantiles"])
    pct = float(np.searchsorted(q, tau) / (len(q) - 1))
    reach = {f"{int(r * 100)}%": pct >= 1 - r for r in (0.01, 0.05, 0.10, 0.20, 0.30)}
    return UpliftPrediction(
        uplift_visit=tau,
        uplift_conversion=tau_c,
        percentile=pct,
        treat_at_reach=reach,
        strategy=bundle["name"],
    )


ARTIFACTS = {
    "segmentation_metrics.json",
    "cohort_metrics.csv",
    "churn_model_comparison.csv",
    "churn_pairwise_deltas.csv",
    "churn_selection.json",
    "churn_threshold_scenarios.csv",
    "churn_error_analysis.csv",
    "feature_ablation.csv",
    "calibration_metrics.json",
    "survival_analysis.csv",
    "survival_summary.json",
    "survival_cox_coefficients.csv",
    "survival_landmark.csv",
    "survival_landmark_summary.json",
    "experiment_results.json",
    "uplift_results.csv",
    "uplift_curves.csv",
    "uplift_heterogeneity.csv",
    "uplift_tau_distribution.csv",
    "uplift_s_learner_diagnostics.json",
    "uplift_pairwise_qini.csv",
    "uplift_calibration_deciles.csv",
    "campaign_simulation.csv",
    "campaign_optimal_reach.csv",
    "uplift_learning_curve.csv",
    "serving_benchmark.json",
    "churn_sensitivity.csv",
    "churn_segment_metrics.csv",
    "shap_global.csv",
    "visitor_segmentation.json",
    "visitor_category_cohorts.csv",
    "eda_summary.json",
    "error_analysis.json",
    "split_manifest.json",
    "manifest.json",
    "customer_segments.csv",
    "clv_metrics.json",
    "clv_by_segment.csv",
}


@app.get("/v1/artifacts/{name}")
def artifact(name: str, limit: int | None = None) -> Any:
    if name not in ARTIFACTS:
        raise HTTPException(404, "unknown artifact")
    path = settings.artifacts_dir / name
    if not path.exists():
        raise HTTPException(404, "artifact not generated yet; run `make evaluate`")
    if name.endswith(".json"):
        return json.loads(path.read_text())
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return json.loads(df.to_json(orient="records"))


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
