"""Experiment analysis and uplift targeting on Criteo (PROJECT_SPEC.md §6).

Steps: randomization checks + ATE on the full population -> 60/20/20 stratified split ->
fit each targeting strategy per outcome -> select on validation Qini (visit) -> evaluate on
test -> campaign simulation at fixed reach levels.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg
from sklearn.model_selection import train_test_split

from retentioniq import SEED
from retentioniq.analytics.experiments.inference import (
    bootstrap_ate_ci,
    compare_arms,
    holm_adjust,
    sample_ratio_test,
    standardized_mean_differences,
)
from retentioniq.ingestion.load import libpq_url
from retentioniq.models.uplift.learners import FEATURES, ResponseModel, strategies
from retentioniq.models.uplift.metrics import (
    curve_frame,
    evaluate_uplift,
    incremental_at_k,
    qini_curve,
    uplift_curve,
)
from retentioniq.tracking import log_run

log = logging.getLogger(__name__)

OUTCOMES = ("visit", "conversion")
PRIMARY = "visit"
REACH = (0.01, 0.05, 0.10, 0.20, 0.30)
# Criteo uplift v2.1 documentation: ~85% of users were assigned to treatment.
DESIGNED_TREATMENT_SHARE = 0.85
# Hypothetical economics for the cost-sensitivity scenario. NOT observed data.
SCENARIO = {"cost_per_treatment": 0.01, "value_per_conversion": 10.0}


def load_population(database_url: str) -> pd.DataFrame:
    """Stream the leakage-safe feature view out of Postgres via COPY (fast for ~14M rows)."""
    cols = ["row_id", *FEATURES, "treatment", "visit", "conversion"]
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        with psycopg.connect(libpq_url(database_url)) as conn, conn.cursor() as cur:
            q = f"COPY (SELECT {', '.join(cols)} FROM marts.mart_uplift_features) TO STDOUT CSV"
            with cur.copy(q) as copy, Path(tmp.name).open("wb") as f:
                for block in copy:
                    f.write(block)
        dtypes = {c: "float32" for c in FEATURES} | {
            "row_id": "int64",
            "treatment": "int8",
            "visit": "int8",
            "conversion": "int8",
        }
        return pd.read_csv(tmp.name, names=cols, dtype=dtypes, engine="pyarrow")


def experiment_analysis(df: pd.DataFrame) -> dict[str, Any]:
    t = df["treatment"].to_numpy()
    smd = standardized_mean_differences(df[FEATURES], t)
    results: dict[str, Any] = {
        "n": len(df),
        "sample_ratio": sample_ratio_test(t, DESIGNED_TREATMENT_SHARE),
        "balance": {
            "smd": smd.round(5).to_dict(),
            "max_abs_smd": float(smd.abs().max()),
            "passes": bool((smd.abs() < 0.1).all()),
        },
        "outcomes": {},
    }
    for outcome in OUTCOMES:
        y = df[outcome].to_numpy()
        cmp = compare_arms(y, t, outcome).to_dict()
        cmp["ate_ci_bootstrap"] = bootstrap_ate_ci(y, t, n_boot=1000, seed=SEED)
        results["outcomes"][outcome] = cmp
    adjusted = holm_adjust({o: results["outcomes"][o]["p_value"] for o in OUTCOMES})
    for o in OUTCOMES:
        results["outcomes"][o]["p_value_holm"] = adjusted[o]
        results["outcomes"][o]["significant_at_0.05"] = adjusted[o] < 0.05
    results["primary_outcome"] = PRIMARY
    return results


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strata = df["treatment"].astype(str) + "_" + df["visit"].astype(str)
    train, rest = train_test_split(df, test_size=0.4, stratify=strata, random_state=SEED)
    val, test = train_test_split(
        rest, test_size=0.5, stratify=strata.loc[rest.index], random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def heterogeneity_by_decile(test: pd.DataFrame, response_score: np.ndarray) -> pd.DataFrame:
    """ATE within deciles of baseline response propensity (§6.2 heterogeneity screen)."""
    deciles = pd.qcut(pd.Series(response_score).rank(method="first"), 10, labels=False)
    rows = []
    for d in range(10):
        m = (deciles == d).to_numpy()
        for outcome in OUTCOMES:
            c = compare_arms(
                test.loc[m, outcome].to_numpy(), test.loc[m, "treatment"].to_numpy(), outcome
            )
            rows.append(
                {
                    "decile": d + 1,
                    "outcome": outcome,
                    "n": int(m.sum()),
                    "rate_control": c.rate_control,
                    "rate_treatment": c.rate_treatment,
                    "ate": c.ate,
                    "ci_low": c.ate_ci_newcombe[0],
                    "ci_high": c.ate_ci_newcombe[1],
                }
            )
    return pd.DataFrame(rows)


def run(database_url: str, artifacts: Path, model_dir: Path) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = load_population(database_url)
    log.info("loaded %d rows", len(df))

    exp = experiment_analysis(df)
    train, val, test = split(df)
    exp["splits"] = {"train": len(train), "validation": len(val), "test": len(test)}
    del df

    results: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    scores_test: dict[tuple[str, str], np.ndarray] = {}
    fitted: dict[tuple[str, str], Any] = {}
    for outcome in OUTCOMES:
        y_tr, t_tr = train[outcome].to_numpy(), train["treatment"].to_numpy()
        for strategy in strategies():
            log.info("fitting %s on %s", strategy.name, outcome)
            strategy.fit(train, t_tr, y_tr)
            fitted[(strategy.name, outcome)] = strategy
            for split_name, frame in (("validation", val), ("test", test)):
                s = strategy.score(frame)
                y, t = frame[outcome].to_numpy(), frame["treatment"].to_numpy()
                m = evaluate_uplift(y, t, s, REACH)
                results.append(
                    {"strategy": strategy.name, "outcome": outcome, "split": split_name, **m}
                )
                if split_name == "test":
                    scores_test[(strategy.name, outcome)] = s
                    for kind, fn in (("qini", qini_curve), ("uplift", uplift_curve)):
                        cf = curve_frame(fn(y, t, s))
                        curves.append(
                            cf.assign(strategy=strategy.name, outcome=outcome, curve=kind)
                        )
            val_m = results[-2]
            log_run(
                "retentioniq-uplift",
                f"{strategy.name}-{outcome}",
                metrics={f"val_{k}": v for k, v in val_m.items() if isinstance(v, float)},
                tags={"outcome": outcome},
            )

    res = pd.DataFrame(results)
    val_primary = res[(res["split"] == "validation") & (res["outcome"] == PRIMARY)]
    best = str(val_primary.sort_values("qini_coefficient", ascending=False).iloc[0]["strategy"])
    res["selected"] = res["strategy"] == best
    res.to_csv(artifacts / "uplift_results.csv", index=False)
    pd.concat(curves).to_csv(artifacts / "uplift_curves.csv", index=False)

    # Campaign simulation on the held-out test population.
    sim_rows = []
    for reach in REACH:
        for name in dict.fromkeys(["random", "response_model", best]):
            row: dict[str, Any] = {"reach": reach, "strategy": name}
            for outcome in OUTCOMES:
                inc = incremental_at_k(
                    test[outcome].to_numpy(),
                    test["treatment"].to_numpy(),
                    scores_test[(name, outcome)],
                    reach,
                    n_boot=1000,
                    seed=SEED,
                )
                row[f"incremental_{outcome}s"] = inc["incremental"]
                row[f"incremental_{outcome}s_ci_low"] = inc["incremental_ci_low"]
                row[f"incremental_{outcome}s_ci_high"] = inc["incremental_ci_high"]
                row["targeted"] = inc["targeted"]
            cost = row["targeted"] * SCENARIO["cost_per_treatment"]
            row["scenario_net_value"] = (
                row["incremental_conversions"] * SCENARIO["value_per_conversion"] - cost
            )
            sim_rows.append(row)
    sim = pd.DataFrame(sim_rows)
    sim["scenario_note"] = "SCENARIO - hypothetical cost/value, not observed data"
    sim["population"] = len(test)
    sim.to_csv(artifacts / "campaign_simulation.csv", index=False)

    resp = fitted[("response_model", "visit")]
    assert isinstance(resp, ResponseModel)
    heterogeneity_by_decile(test, resp.score(test)).to_csv(
        artifacts / "uplift_heterogeneity.csv", index=False
    )
    tau = scores_test[(best, PRIMARY)]
    hist, edges = np.histogram(tau, bins=50)
    pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}).to_csv(
        artifacts / "uplift_tau_distribution.csv", index=False
    )

    exp["selected_strategy"] = best
    (artifacts / "experiment_results.json").write_text(json.dumps(exp, indent=2, default=float))
    joblib.dump(
        {
            "strategy": fitted[(best, PRIMARY)],
            "conversion_strategy": fitted[(best, "conversion")],
            "features": FEATURES,
            "name": best,
            "outcome": PRIMARY,
            "tau_quantiles": np.quantile(tau, np.linspace(0, 1, 101)).tolist(),
        },
        model_dir / "uplift_champion.joblib",
    )
    return {"selected_strategy": best, **exp["splits"]}
