"""Render docs/RESULTS.md from artifacts/ — the only place results are written in prose.

Every number below is read from an artifact produced by `make evaluate`; nothing is typed by
hand (PROJECT_SPEC.md §8). Re-run after every evaluation: `make report`.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _json(name: str) -> dict[str, Any] | None:
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def _csv(name: str) -> pd.DataFrame | None:
    p = ART / name
    return pd.read_csv(p) if p.exists() else None


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def customers() -> list[str]:
    seg = _json("segmentation_metrics.json")
    if not seg:
        return []
    cmp = seg["comparison"]
    rows = [
        {
            "segmentation": k,
            "groups": v["n_groups"],
            "future-activity AUC": v["future_activity_auc"],
            "silhouette": v.get("silhouette"),
        }
        for k, v in cmp.items()
        if isinstance(v, dict)
    ]
    rfm = pd.DataFrame(seg["rfm_segments_full_history"])[
        [
            "rfm_segment",
            "customers",
            "share_customers",
            "share_revenue",
            "avg_order_value",
            "mean_frequency",
            "repeat_purchase_rate",
        ]
    ].sort_values("share_revenue", ascending=False)
    vc = seg["value_concentration"]
    out = [
        "## Customer intelligence (Online Retail II)",
        "",
        f"Revenue concentration: the top 10% of customers generate "
        f"{_pct(vc['top_10pct_revenue_share'])} of qualifying revenue "
        f"(top 1%: {_pct(vc['top_1pct_revenue_share'])}; Gini {vc['gini']:.2f}).",
        "",
        "### RFM segments (full history)",
        _table(rfm, ".3f"),
        "",
        f"### RFM vs learned segmentation (cutoff {seg['evaluation_cutoff']}, "
        f"{seg['horizon_days']}-day future window, chosen k = {seg['chosen_k']})",
        "",
        "Both segmentations use only pre-cutoff orders; *future-activity AUC* measures how well "
        "segment membership alone separates customers who buy again in the held-out window.",
        "",
        _table(pd.DataFrame(rows), ".3f"),
        "",
        f"Agreement between RFM segments and K-Means clusters (ARI): "
        f"{cmp['agreement_ari_rfm_vs_kmeans']:.3f}.",
    ]
    coh = _csv("cohort_metrics.csv")
    if coh is not None:
        m = coh[coh["group_type"] == "cohort"].pivot_table(
            index="group", columns="month_number", values="retention_rate"
        )
        avail = [c for c in (1, 3, 6, 12) if c in m.columns]
        summary = m[avail].mean().to_frame("mean retention").T
        summary.columns = [f"M{c}" for c in avail]
        out += [
            "",
            "### Cohort retention (mean across first-purchase cohorts)",
            summary.to_markdown(index=False, floatfmt=".3f"),
        ]
    return out


def visitors() -> list[str]:
    seg = _json("visitor_segmentation.json")
    coh = _csv("visitor_category_cohorts.csv")
    if not seg:
        return []
    cmp = pd.DataFrame(
        [
            {
                "segmentation": k,
                "groups": v["n_groups"],
                "test inactivity AUC": v["test_inactivity_auc"],
            }
            for k, v in seg["comparison"].items()
        ]
    )
    out = [
        "## Visitor segmentation (RetailRocket)",
        "",
        f"Fit on training snapshots, judged on the test snapshot (chosen k = {seg['chosen_k']}).",
        "",
        _table(cmp, ".3f"),
    ]
    if coh is not None:
        pv = (
            coh[coh["week_number"].isin([1, 2, 4, 8])]
            .pivot_table(index="group", columns="week_number", values="retention_rate")
            .reset_index()
        )
        pv.columns = ["first category", *[f"W{c}" for c in pv.columns[1:]]]
        out += ["", "### Weekly retention by category of first item viewed", _table(pv, ".3f")]
    return out


def churn() -> list[str]:
    cmp = _csv("churn_model_comparison.csv")
    if cmp is None:
        return []
    man = _json("split_manifest.json") or {}
    cols = [
        "model",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "roc_auc",
        "precision@10",
        "recall@10",
        "brier",
        "ece",
        "is_champion",
    ]
    out = [
        "## Inactivity prediction (RetailRocket)",
        "",
        f"Cutoffs: {man.get('cutoffs')}. Rows: {man.get('rows')}. "
        f"Base rate (inactive): {man.get('base_rate')}.",
        "",
        "### Model ladder — test set (PR-AUC is primary; 95% bootstrap CIs)",
        _table(cmp[cols]),
    ]
    abl = _csv("feature_ablation.csv")
    if abl is not None:
        out += [
            "",
            "### Feature-family ablation (cumulative)",
            _table(
                abl[
                    [
                        "model",
                        "features",
                        "n_features",
                        "test_pr_auc",
                        "delta_test_pr_auc",
                        "delta_ci_low",
                        "delta_ci_high",
                    ]
                ]
            ),
        ]
    sens = _csv("churn_sensitivity.csv")
    if sens is not None:
        pv = sens.pivot_table(index="setting", columns="model", values="pr_auc").reset_index()
        stable = bool(sens["ranking_stable"].iloc[0])
        out += [
            "",
            "### Sensitivity of the label definition",
            f"Model ranking (M4 > M1) stable across all settings: **{stable}**.",
            "",
            _table(pv),
        ]
    shap = _csv("shap_global.csv")
    if shap is not None:
        out += [
            "",
            "### What drives inactivity risk (mean |SHAP|, M4)",
            _table(shap.head(10)[["feature", "family", "mean_abs_shap", "value_shap_corr"]]),
        ]
    seg = _csv("churn_segment_metrics.csv")
    if seg is not None:
        out += ["", "### Segment performance", _table(seg)]
    return out


def survival() -> list[str]:
    s = _json("survival_summary.json")
    km = _csv("survival_analysis.csv")
    cox = _csv("survival_cox_coefficients.csv")
    if not s or km is None:
        return []
    out = [
        "## Time to inactivity (survival)",
        "",
        f"Population {s['population']:,} returning visitors; {s['events']:,} events, "
        f"{s['censored']:,} censored. Log-rank test across first-session types: "
        f"p = {s['logrank_first_session_type']['p_value']:.2g}. "
        f"Cox concordance on held-out visitors: {s['cox']['concordance_test']:.3f}; "
        f"PH violations (p < 0.01): {s['cox']['ph_violations_p_lt_0.01'] or 'none'}; "
        f"stratified on: {s['cox']['strata'] or 'none'}.",
        "",
        _table(km.drop(columns=["section"]), ".3f"),
    ]
    if cox is not None:
        out += ["", "### Cox hazard ratios", _table(cox, ".3f")]
    return out


def uplift() -> list[str]:
    exp = _json("experiment_results.json")
    res = _csv("uplift_results.csv")
    sim = _csv("campaign_simulation.csv")
    if not exp or res is None:
        return []
    rows = []
    for o, r in exp["outcomes"].items():
        rows.append(
            {
                "outcome": o,
                "control": r["rate_control"],
                "treatment": r["rate_treatment"],
                "ATE": r["ate"],
                "ATE 95% CI": f"[{r['ate_ci_newcombe'][0]:.5f}, {r['ate_ci_newcombe'][1]:.5f}]",
                "relative lift": r["relative_lift"],
                "p (Holm)": r["p_value_holm"],
            }
        )
    test = res[res["split"] == "test"]
    out = [
        "## Experiment & uplift (Criteo)",
        "",
        f"N = {exp['n']:,}. Treatment share {exp['sample_ratio']['observed_share']:.4f} vs "
        f"designed {exp['sample_ratio']['expected_share']}; covariate balance max |SMD| = "
        f"{exp['balance']['max_abs_smd']:.4f} (passes: {exp['balance']['passes']}).",
        "",
        _table(pd.DataFrame(rows), ".5f"),
        "",
        f"### Targeting strategies — test set (selected on validation Qini: "
        f"**{exp['selected_strategy']}**)",
        _table(
            test[
                [
                    "strategy",
                    "outcome",
                    "qini_coefficient",
                    "auuc",
                    "uplift@1",
                    "uplift@10",
                    "uplift@30",
                ]
            ]
        ),
    ]
    if sim is not None:
        out += [
            "",
            "### Campaign simulation — incremental outcomes on the test population",
            _table(
                sim[
                    [
                        "reach",
                        "strategy",
                        "targeted",
                        "incremental_visits",
                        "incremental_conversions",
                        "incremental_conversions_ci_low",
                        "incremental_conversions_ci_high",
                    ]
                ],
                ".1f",
            ),
            "",
            f"_{sim['scenario_note'].iloc[0]}_ (scenario net value column in CSV).",
        ]
    return out


def serving() -> list[str]:
    b = _json("serving_benchmark.json")
    if not b:
        return []
    return [
        "## Serving benchmark",
        "",
        f"Tool: {b['tool']}; hardware: {b['hardware']}.",
        "",
        _table(pd.DataFrame(b["runs"]), ".2f"),
        "",
        f"SLO check (single-record, concurrency 8): {b['slo_check_single_c8']}",
    ]


def main() -> None:
    manifest = _json("manifest.json") or {}
    stages = manifest.get("stages", {}).values()
    shas = {v.get("git_sha", "")[:8] for v in stages}
    specs = {v.get("spec_version") for v in stages}
    lines = [
        "# RetentionIQ — Results",
        "",
        "_Generated by `evaluation/render_report.py` from `artifacts/`. Do not edit by hand._",
        "_Exploratory analysis and the hypotheses tested here: [EDA.md](EDA.md)._",
        f"_Spec version(s): {sorted(specs)}; git: {sorted(shas)}._",
        "",
    ]
    for section in (customers, visitors, churn, survival, uplift, serving):
        part = section()
        if part:
            lines += [*part, ""]
    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
