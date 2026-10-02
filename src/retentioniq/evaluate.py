"""`make evaluate`: regenerate every reported number from the warehouse (PROJECT_SPEC.md §8).

Each stage writes its artifacts to `artifacts/`; a manifest records what ran, under which
spec version and git SHA, and how long it took. No number in the README, dashboards or web
app is typed by hand: they all read these files.
"""

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from retentioniq import SPEC_VERSION
from retentioniq.config import REPO_ROOT, settings
from retentioniq.tracking import git_sha

log = logging.getLogger(__name__)
MODEL_DIR = REPO_ROOT / "data" / "models"
STAGES = ("customers", "churn", "explain", "survival", "uplift")


def _stages(include_torch: bool, sensitivity: bool) -> dict[str, Callable[[], Any]]:
    from retentioniq.analytics import customers
    from retentioniq.training import churn, explain, survival, uplift

    db, out = settings.database_url, settings.artifacts_dir
    return {
        "customers": lambda: customers.run(db, out),
        "churn": lambda: churn.run(db, out, MODEL_DIR, include_torch, sensitivity),
        "explain": lambda: explain.run(MODEL_DIR, out),
        "survival": lambda: survival.run(db, out, MODEL_DIR),
        "uplift": lambda: uplift.run(db, out, MODEL_DIR),
    }


def run(
    only: list[str] | None = None, include_torch: bool = True, sensitivity: bool = True
) -> dict[str, Any]:
    out = settings.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
    )
    stages = _stages(include_torch, sensitivity)
    for name in only or list(STAGES):
        log.info("stage %s", name)
        start = time.perf_counter()
        summary = stages[name]()
        manifest["stages"][name] = {
            "finished_at": datetime.now(UTC).isoformat(),
            "seconds": round(time.perf_counter() - start, 1),
            "spec_version": SPEC_VERSION,
            "git_sha": git_sha(),
            "summary": summary,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def artifact_paths() -> list[Path]:
    return sorted(p for p in settings.artifacts_dir.iterdir() if p.is_file())
