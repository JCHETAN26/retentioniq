"""Load-test the API and write artifacts/serving_benchmark.json (PROJECT_SPEC.md §9.2).

Usage: uv run python evaluation/benchmark_serving.py --url http://localhost:8000
Closed-loop load: `concurrency` workers each send requests back-to-back for `duration` s.
"""

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SLO = {"p50_ms": 25, "p95_ms": 100, "p99_ms": 250, "error_rate": 0.001, "throughput_rps": 200}


def payloads() -> list[dict[str, float]]:
    from retentioniq.features.retailrocket import ALL_FEATURES

    preds = pd.read_parquet(ROOT / "artifacts" / "churn_test_predictions.parquet")
    return preds[ALL_FEATURES].sample(500, random_state=0).to_dict("records")


async def worker(
    client: httpx.AsyncClient,
    path: str,
    bodies: list[Any],
    stop: float,
    lat: list[float],
    errors: list[int],
) -> None:
    i = 0
    while time.perf_counter() < stop:
        body = bodies[i % len(bodies)]
        i += 1
        t0 = time.perf_counter()
        try:
            r = await client.post(path, json=body)
            if r.status_code >= 500:
                errors.append(r.status_code)
        except httpx.HTTPError:
            errors.append(-1)
        lat.append(time.perf_counter() - t0)


async def scenario(
    url: str, path: str, bodies: list[Any], concurrency: int, duration: float
) -> dict[str, Any]:
    lat: list[float] = []
    errors: list[int] = []
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=10, limits=limits) as client:
        for b in bodies[:20]:  # warm-up
            await client.post(path, json=b)
        stop = time.perf_counter() + duration
        start = time.perf_counter()
        await asyncio.gather(
            *(worker(client, path, bodies, stop, lat, errors) for _ in range(concurrency))
        )
        elapsed = time.perf_counter() - start
    ms = np.array(lat) * 1000
    return {
        "endpoint": path,
        "concurrency": concurrency,
        "duration_s": round(elapsed, 2),
        "requests": len(lat),
        "throughput_rps": round(len(lat) / elapsed, 1),
        "p50_ms": round(float(np.percentile(ms, 50)), 2),
        "p95_ms": round(float(np.percentile(ms, 95)), 2),
        "p99_ms": round(float(np.percentile(ms, 99)), 2),
        "error_rate": len(errors) / max(1, len(lat)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--duration", type=float, default=20)
    args = ap.parse_args()
    bodies = payloads()
    runs = []
    for c in (1, 8, 32):
        runs.append(
            asyncio.run(
                scenario(args.url, "/v1/churn/predict?explain=false", bodies, c, args.duration)
            )
        )
    runs.append(asyncio.run(scenario(args.url, "/v1/churn/predict", bodies, 8, args.duration)))
    batch = [{"rows": bodies[i : i + 100]} for i in range(0, 500, 100)]
    runs.append(asyncio.run(scenario(args.url, "/v1/churn/batch", batch, 4, args.duration)))
    single = runs[1]
    out = {
        "tool": "evaluation/benchmark_serving.py (httpx async, closed loop)",
        "target": args.url,
        "hardware": {
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "runs": runs,
        "slo": SLO,
        "slo_check_single_c8": {
            "p50": single["p50_ms"] <= SLO["p50_ms"],
            "p95": single["p95_ms"] <= SLO["p95_ms"],
            "p99": single["p99_ms"] <= SLO["p99_ms"],
            "errors": single["error_rate"] <= SLO["error_rate"],
            "throughput": max(r["throughput_rps"] for r in runs[:3]) >= SLO["throughput_rps"],
        },
    }
    (ROOT / "artifacts" / "serving_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["slo_check_single_c8"]), json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
