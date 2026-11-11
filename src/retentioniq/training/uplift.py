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
from retentioniq.models.uplift.learners import (
    FEATURES,
    ResponseModel,
    get_strategy,
    strategies,
)
from retentioniq.models.uplift.metrics import (
    curve_frame,
    evaluate_uplift,
    incremental_at_k,
    paired_bootstrap_qini_diff,
    qini_coefficient,
    qini_curve,
    top_k_counts,
    uplift_by_score_decile,
    uplift_curve,
)
from retentioniq.tracking import log_run

log = logging.getLogger(__name__)

OUTCOMES = ("visit", "conversion")
PRIMARY = "visit"
REACH = (0.01, 0.05, 0.10, 0.20, 0.30)
# Learners are fit on a stratified subsample of the training split (spec v0.4, compute budget);
# validation and test always use their full rows.
TRAIN_ROWS = 3_000_000
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
        df = pd.read_csv(tmp.name, names=cols, dtype=dtypes, engine="pyarrow")
    return canonical_order(df)


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by `row_id` (the source-file row, unique per dbt test) before any seeded split.

    A SELECT without ORDER BY returns rows in Postgres's physical order, which differs between
    loads. The seeded splits and subsamples are positional, so without this they pick different
    rows on every run and the uplift results are not reproducible (spec v0.6.4).
    """
    if not df["row_id"].is_unique:
        raise ValueError("row_id is not unique; cannot fix a deterministic row order")
    return df.sort_values("row_id", kind="stable").reset_index(drop=True)


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


def run(
    database_url: str, artifacts: Path, model_dir: Path, learning_curve: bool = True
) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    df = load_population(database_url)
    log.info("loaded %d rows", len(df))

    exp = experiment_analysis(df)
    train, val, test = split(df)
    if len(train) > TRAIN_ROWS:
        strata = train["treatment"].astype(str) + "_" + train["visit"].astype(str)
        train, _ = train_test_split(
            train, train_size=TRAIN_ROWS, stratify=strata, random_state=SEED
        )
        train = train.reset_index(drop=True)
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

    # 2.2 S-learner diagnostics check
    s_lgbm = fitted.get(("s_learner_lgbm", PRIMARY))
    if s_lgbm is not None and hasattr(s_lgbm, "diagnostics"):
        diag = s_lgbm.diagnostics(val)
        (artifacts / "uplift_s_learner_diagnostics.json").write_text(
            json.dumps(diag, indent=2, default=float)
        )

    # 2.3 Paired bootstrap Qini difference test
    # Take a 500k row stratified subsample (fixed seed) for fast bootstrap
    boot_test_rows = 500_000
    if len(test) > boot_test_rows:
        strata_test = test["treatment"].astype(str) + "_" + test["visit"].astype(str)
        test_boot, _ = train_test_split(
            test,
            train_size=boot_test_rows,
            stratify=strata_test,
            random_state=SEED,
        )
        boot_idx = test_boot.index.to_numpy()
    else:
        test_boot = test
        boot_idx = np.arange(len(test))

    pairwise_rows = []
    for outcome in OUTCOMES:
        y_boot = test_boot[outcome].to_numpy()
        t_boot = test_boot["treatment"].to_numpy()
        val_out = res[(res["split"] == "validation") & (res["outcome"] == outcome)]
        top3 = (
            val_out.sort_values("qini_coefficient", ascending=False)["strategy"]
            .unique()[:3]
            .tolist()
        )
        strats_to_compare = list(top3)
        if "response_model" not in strats_to_compare:
            strats_to_compare.append("response_model")
        for i in range(len(strats_to_compare)):
            for j in range(i + 1, len(strats_to_compare)):
                sa_name, sb_name = strats_to_compare[i], strats_to_compare[j]
                sa = scores_test[(sa_name, outcome)][boot_idx]
                sb = scores_test[(sb_name, outcome)][boot_idx]
                diff = paired_bootstrap_qini_diff(y_boot, t_boot, sa, sb, n_boot=1000, seed=SEED)
                pairwise_rows.append(
                    {
                        "outcome": outcome,
                        "strategy_a": sa_name,
                        "strategy_b": sb_name,
                        **diff,
                    }
                )
    pd.DataFrame(pairwise_rows).to_csv(artifacts / "uplift_pairwise_qini.csv", index=False)

    # 2.4 Uplift calibration deciles table
    calib_rows = []
    for strat_name in dict.fromkeys([best, "response_model"]):
        for outcome in OUTCOMES:
            s_test = scores_test[(strat_name, outcome)]
            dec_rows = uplift_by_score_decile(
                test[outcome].to_numpy(),
                test["treatment"].to_numpy(),
                s_test,
                outcome_name=outcome,
            )
            for r in dec_rows:
                calib_rows.append({"strategy": strat_name, "outcome": outcome, **r})
    pd.DataFrame(calib_rows).to_csv(artifacts / "uplift_calibration_deciles.csv", index=False)

    # Coarse campaign simulation on the held-out test population.
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

    # 2.5 Fine reach grid campaign simulation and optimal reach
    fine_reaches = np.round(np.arange(0.01, 0.51, 0.01), 2)
    fine_sim_rows: list[dict[str, Any]] = []
    strategies_fine = list(dict.fromkeys(["random", "response_model", best]))
    for name in strategies_fine:
        s_conv = scores_test[(name, "conversion")]
        y_conv = test["conversion"].to_numpy()
        t_conv = test["treatment"].to_numpy()
        n_pop = len(test)
        strat_rows: list[dict[str, Any]] = []
        for reach in fine_reaches:
            yt, nt, yc, nc = top_k_counts(y_conv, t_conv, s_conv, float(reach))
            targeted = float(reach) * n_pop
            inc_conv = float((yt / nt - yc / nc) * targeted if nt and nc else 0.0)
            cost = float(targeted * SCENARIO["cost_per_treatment"])
            net_val = float(inc_conv * SCENARIO["value_per_conversion"] - cost)
            strat_rows.append(
                {
                    "strategy": name,
                    "reach": float(reach),
                    "targeted": targeted,
                    "incremental_conversions": inc_conv,
                    "cost": cost,
                    "net_value": net_val,
                }
            )
        opt_idx = int(np.argmax([r["net_value"] for r in strat_rows]))
        opt_reach = strat_rows[opt_idx]["reach"]
        max_net = strat_rows[opt_idx]["net_value"]
        for r in strat_rows:
            r["optimal_reach"] = opt_reach
            r["max_net_value"] = max_net
            r["is_optimal"] = bool(r["reach"] == opt_reach)
            fine_sim_rows.append(r)
    sim_fine = pd.DataFrame(fine_sim_rows)
    sim_fine["scenario_note"] = "SCENARIO - hypothetical cost/value, not observed data"
    sim_fine.to_csv(artifacts / "campaign_optimal_reach.csv", index=False)

    # 2.6 Training-size learning curve
    if learning_curve:
        curve_rows = []
        target_sizes = [500_000, 1_000_000, 2_000_000, 3_000_000]
        sizes = [s for s in target_sizes if s <= len(train)]
        if not sizes:
            sizes = [max(10, len(train) // 4), max(20, len(train) // 2), len(train)]
        for s in sizes:
            if s < len(train):
                strata_sub = train["treatment"].astype(str) + "_" + train["visit"].astype(str)
                sub_train, _ = train_test_split(
                    train, train_size=s, stratify=strata_sub, random_state=SEED
                )
            else:
                sub_train = train
            inst = get_strategy(best)
            inst.fit(sub_train, sub_train["treatment"].to_numpy(), sub_train[PRIMARY].to_numpy())
            s_val = inst.score(val)
            s_test = inst.score(test)
            q_val = qini_coefficient(val[PRIMARY].to_numpy(), val["treatment"].to_numpy(), s_val)
            q_test = qini_coefficient(
                test[PRIMARY].to_numpy(), test["treatment"].to_numpy(), s_test
            )
            curve_rows.append(
                {
                    "train_size": s,
                    "val_qini": float(q_val),
                    "test_qini": float(q_test),
                }
            )
        pd.DataFrame(curve_rows).to_csv(artifacts / "uplift_learning_curve.csv", index=False)

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
