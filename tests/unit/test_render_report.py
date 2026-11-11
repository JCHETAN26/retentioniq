"""The report renderer must only publish numbers that exist in artifacts/."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "evaluation" / "render_report.py"


def _load(tmp_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_report", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = tmp_path
    mod.ART = tmp_path / "artifacts"
    mod.FIG = tmp_path / "docs" / "figures"
    mod.ART.mkdir()
    mod.FIG.mkdir(parents=True)
    return mod


def test_refuses_to_overwrite_reports_without_artifacts(tmp_path: Path) -> None:
    mod = _load(tmp_path)
    results = tmp_path / "docs" / "RESULTS.md"
    results.write_text("previous report\n")
    with pytest.raises(SystemExit):
        mod.main()
    assert results.read_text() == "previous report\n"


def test_findings_are_computed_from_artifacts(tmp_path: Path) -> None:
    mod = _load(tmp_path)
    art = mod.ART
    pd.DataFrame(
        {
            "model": ["M0_naive_recency", "M4_lightgbm"],
            "pr_auc": [0.9300, 0.9521],
            "pr_auc_ci_low": [0.9268, 0.9493],
            "pr_auc_ci_high": [0.9334, 0.9550],
            "lift_returners@10": [1.5, 2.2],
            "is_champion": [False, True],
        }
    ).to_csv(art / "churn_model_comparison.csv", index=False)
    (art / "split_manifest.json").write_text(json.dumps({"base_rate": {"test": 0.8735}}))
    (art / "survival_summary.json").write_text(json.dumps({"cox": {"concordance_test": 0.507}}))
    (art / "survival_landmark_summary.json").write_text(
        json.dumps(
            {
                "concordance_test": 0.49,
                "concordance_delta": -0.017,
                "landmark_days": 7,
                "population": 1234,
                "logrank_activity_tier": {"p_value": 0.01},
            }
        )
    )
    (art / "experiment_results.json").write_text(
        json.dumps(
            {
                "n": 1000,
                "selected_strategy": "s_learner_lgbm",
                "outcomes": {"visit": {"ate": 0.01, "relative_lift": 0.27}},
            }
        )
    )
    pd.DataFrame(
        {
            "split": ["test", "test"],
            "outcome": ["visit", "visit"],
            "strategy": ["s_learner_lgbm", "response_model"],
            "qini_coefficient": [0.0921, 0.0866],
        }
    ).to_csv(art / "uplift_results.csv", index=False)

    mod.generate_findings()
    text = (tmp_path / "docs" / "FINDINGS.md").read_text()

    assert "M4_lightgbm" in text and "0.9521" in text and "0.9300" in text
    assert "2.20x" in text
    assert "0.507" in text
    # Directional language follows the numbers: a worse landmark model is not "higher".
    assert "not higher" in text and "-0.017" in text
    assert "`s_learner_lgbm`, beats the response" in text
    # None of the old typed-in fallbacks survive.
    for stale in ("0.91+", "0.70+", "X-Learner", "3x", "~0.53"):
        assert stale not in text
    # Figures are only linked when the file exists.
    assert "figures/" not in text
