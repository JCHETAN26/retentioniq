"""Thin MLflow wrapper: runs are logged when a tracking server is reachable, skipped otherwise.

Every run is tagged with the spec version and git SHA so results trace back to the contract
and code they were produced under (PROJECT_SPEC.md §7.1).
"""

import logging
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

from retentioniq import SPEC_VERSION
from retentioniq.config import REPO_ROOT, settings

log = logging.getLogger(__name__)


@cache
def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@cache
def mlflow_available() -> bool:
    try:
        import httpx
        import mlflow  # noqa: F401

        httpx.get(f"{settings.mlflow_tracking_uri}/health", timeout=2).raise_for_status()
        return True
    except Exception:
        log.warning("MLflow not reachable at %s; skipping tracking", settings.mlflow_tracking_uri)
        return False


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: list[Path] | None = None,
) -> str | None:
    if not mlflow_available():
        return None
    import math

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"spec_version": SPEC_VERSION, "git_sha": git_sha(), **(tags or {})})
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        if metrics:
            clean = {
                k.replace("@", "_at_"): float(v)
                for k, v in metrics.items()
                if isinstance(v, int | float) and math.isfinite(v)
            }
            mlflow.log_metrics(clean)
        for a in artifacts or []:
            mlflow.log_artifact(str(a))
        return str(run.info.run_id)
