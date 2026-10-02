from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from retentioniq.features.retailrocket import ALL_FEATURES
from retentioniq.models.uplift.learners import FEATURES as UPLIFT_FEATURES
from retentioniq.serving import app as api


@dataclass
class FakeChurn:
    name: str = "fake"

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.clip(x["recency_days"].to_numpy() / 30, 0, 1)


@dataclass
class FakeUplift:
    def score(self, x: pd.DataFrame) -> np.ndarray:
        return x["f0"].to_numpy() / 100


@pytest.fixture
def client() -> TestClient:
    api.state.churn = {"model": FakeChurn(), "calibrator": lambda p: p, "features": ALL_FEATURES,
                       "feature_set": "test"}  # fmt: skip
    api.state.uplift = {"strategy": FakeUplift(), "conversion_strategy": FakeUplift(),
                        "name": "fake_uplift", "outcome": "visit",
                        "tau_quantiles": list(np.linspace(-0.01, 0.05, 101))}  # fmt: skip
    api.state.explainer = None
    return TestClient(api.app)


def _features(recency: float) -> dict[str, float]:
    return {f: (recency if f == "recency_days" else 1.0) for f in ALL_FEATURES}


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["models"]["churn"]


def test_predict_returns_probability_and_tier(client: TestClient) -> None:
    r = client.post("/v1/churn/predict", json=_features(27))
    assert r.status_code == 200
    assert r.json()["p_inactive"] == pytest.approx(0.9) and r.json()["risk_tier"] == "high"


def test_predict_rejects_bad_input(client: TestClient) -> None:
    bad = _features(5) | {"recency_days": -3}
    assert client.post("/v1/churn/predict", json=bad).status_code == 422
    extra = _features(5) | {"unknown": 1}
    assert client.post("/v1/churn/predict", json=extra).status_code == 422
    missing = _features(5)
    missing.pop("views_7d")
    assert client.post("/v1/churn/predict", json=missing).status_code == 422


def test_batch(client: TestClient) -> None:
    r = client.post("/v1/churn/batch", json={"rows": [_features(3), _features(30)]})
    assert r.json()["risk_tier"] == ["low", "high"]


def test_uplift_score(client: TestClient) -> None:
    body = {f: 0.0 for f in UPLIFT_FEATURES} | {"f0": 4.0}
    r = client.post("/v1/uplift/score", json=body).json()
    assert r["uplift_visit"] == pytest.approx(0.04)
    # 0.04 sits at the 83rd percentile of the fake tau distribution
    assert r["percentile"] == pytest.approx(0.84)
    assert r["treat_at_reach"]["30%"] and not r["treat_at_reach"]["10%"]


def test_metrics_endpoint(client: TestClient) -> None:
    client.post("/v1/churn/predict", json=_features(10))
    text = client.get("/metrics").text
    assert "riq_request_seconds" in text and "riq_churn_prediction" in text


def test_missing_model_is_503(client: TestClient) -> None:
    api.state.churn = None
    assert client.post("/v1/churn/predict", json=_features(1)).status_code == 503
