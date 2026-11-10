"""Render docs/RESULTS.md and docs/FINDINGS.md from artifacts/ (PROJECT_SPEC.md §8).

Every number is read from an artifact produced by `make evaluate`; nothing is typed by
hand. Figures are dynamically generated in docs/figures/. Re-run: `make report`.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ROOT / "docs" / "figures"


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
    all_candidate_cols = [
        "model",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "pr_auc_returners",
        "returner_base_rate",
        "precision_returners@10",
        "lift_returners@10",
        "roc_auc",
        "precision@10",
        "recall@10",
        "brier",
        "ece",
        "is_champion",
    ]
    cols = [c for c in all_candidate_cols if c in cmp.columns]
    out = [
        "## Inactivity prediction (RetailRocket)",
        "",
        f"Cutoffs: {man.get('cutoffs')}. Rows: {man.get('rows')}. "
        f"Base rate (inactive): {man.get('base_rate')}.",
        "",
        "### Model ladder — test set (PR-AUC is primary; 95% bootstrap CIs)",
        _table(cmp[cols]),
    ]

    sel = _json("churn_selection.json")
    if sel is not None:
        # Keys as written by training/churn.py (`selection_reasoning`).
        d = sel["delta_vs_best"]
        out += [
            "",
            "### Model selection rationale",
            f"- **Best validation model**: `{sel['best_by_validation']}`",
            f"- **Selected champion**: `{sel['selected_champion']}`",
            f"- **Delta vs best**: {d['delta_pr_auc']:.4f} "
            f"(95% CI: [{d['ci_low']:.4f}, {d['ci_high']:.4f}])",
            f"- **Rationale**: {sel['reasoning']}",
        ]

    deltas = _csv("churn_pairwise_deltas.csv")
    if deltas is not None:
        out += [
            "",
            "### Pairwise model deltas (PR-AUC)",
            _table(deltas, ".4f"),
        ]

    scenarios = _csv("churn_threshold_scenarios.csv")
    if scenarios is not None:
        out += [
            "",
            "### Cost-optimal operating threshold scenarios",
            "_SCENARIO - hypothetical cost/value, not observed data_",
            "",
            _table(scenarios, ".4f"),
        ]

    errs = _csv("churn_error_analysis.csv")
    if errs is not None:
        out += [
            "",
            "### Error analysis by segment (champion model)",
            _table(errs, ".4f"),
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

    figs = []
    if (FIG / "churn_ladder_pr_auc.png").exists():
        figs.append("![Model Ladder PR-AUC](figures/churn_ladder_pr_auc.png)")
    if (FIG / "churn_ablation_deltas.png").exists():
        figs.append("![Ablation Deltas](figures/churn_ablation_deltas.png)")
    if (FIG / "churn_shap_importance.png").exists():
        figs.append("![Top-10 Feature Drivers](figures/churn_shap_importance.png)")
    if figs:
        out += ["", "### Inactivity figures", *figs]
    return out


def survival() -> list[str]:
    s = _json("survival_summary.json")
    km = _csv("survival_analysis.csv")
    cox = _csv("survival_cox_coefficients.csv")
    if not s or km is None:
        return []

    c_info = s.get("cox", {})
    c_test = c_info.get("concordance_test")
    c_single = c_info.get("concordance_single_covariate")
    c_rand = c_info.get("concordance_random_baseline", 0.50)

    out = [
        "## Time to inactivity (survival)",
        "",
        f"Population {s['population']:,} returning visitors; {s['events']:,} events, "
        f"{s['censored']:,} censored. Log-rank test across first-session types: "
        f"p = {s['logrank_first_session_type']['p_value']:.2g}.",
        "",
        "### First-session Cox model concordance & baselines",
        f"- **Held-out test concordance**: {c_test:.3f}" if c_test is not None else "",
        f"- **Single-covariate baseline (`first_session_events`)**: {c_single:.3f}"
        if c_single is not None
        else "",
        f"- **Random-score baseline**: {c_rand:.2f}",
        f"- **PH violations (p < 0.01)**: {c_info.get('ph_violations_p_lt_0.01') or 'none'}",
        f"- **Stratified covariates**: {c_info.get('strata') or 'none'}",
    ]
    if c_info.get("interpretation"):
        out += [f"- **Interpretation**: {c_info['interpretation']}"]

    out += ["", _table(km.drop(columns=["section"]), ".3f")]
    if cox is not None:
        out += ["", "### Cox hazard ratios", _table(cox, ".3f")]

    lm_s = _json("survival_landmark_summary.json")
    lm_cox = _csv("survival_landmark.csv")
    if lm_s is not None and lm_s.get("concordance_test") is not None:
        c_lm = lm_s.get("concordance_test", 0.0)
        c_first = lm_s.get("first_session_concordance_test", 0.0)
        delta_c = lm_s.get("concordance_delta", c_lm - c_first)
        out += [
            "",
            "### Day-7 Landmark survival analysis",
            f"- **Landmark horizon ($t_{{LM}}$)**: {lm_s.get('landmark_days', 7)} days",
            f"- **Eligible population**: {lm_s.get('population', 0):,} visitors "
            f"({lm_s.get('events', 0):,} events, {lm_s.get('censored', 0):,} censored)",
            f"- **Held-out concordance**: {c_lm:.3f} "
            f"(vs first-session {c_first:.3f}; delta **+{delta_c:.3f}**)",
            f"- **Activity tier log-rank test**: "
            f"p = {lm_s.get('logrank_activity_tier', {}).get('p_value', 1.0):.2e}",
            f"- **Stratified covariates**: {lm_s.get('strata') or 'none'}",
        ]
        if lm_s.get("interpretation"):
            out += [f"- **Interpretation**: {lm_s['interpretation']}"]
        if lm_cox is not None and not lm_cox.empty:
            out += ["", "#### Landmark Cox hazard ratios", _table(lm_cox, ".3f")]
        if lm_s.get("activity_tier_km"):
            out += [
                "",
                "#### Landmark Kaplan-Meier survival by 7-day activity tier",
                _table(pd.DataFrame(lm_s["activity_tier_km"]), ".3f"),
            ]

    if (FIG / "survival_landmark_km.png").exists():
        out += [
            "",
            "### Landmark survival curves",
            "![Day-7 Landmark KM Curves](figures/survival_landmark_km.png)",
        ]
    return [line for line in out if line != ""]


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

    diag = _json("uplift_s_learner_diagnostics.json")
    if diag is not None:
        warn = (
            "**WARNING: Treatment regularized towards zero** (negligible gain or split count)."
            if diag.get("zero_or_negligible")
            else "Treatment feature split count and gain active."
        )
        t_split = diag.get("treatment_split_count", 0)
        t_gain = diag.get("treatment_split_gain", diag.get("treatment_gain", 0.0))
        t_rank = diag.get("treatment_rank", diag.get("treatment_gain_rank", "N/A"))
        tot_feat = diag.get("n_features", diag.get("total_features", "N/A"))
        out += [
            "",
            "### S-Learner diagnostics (treatment indicator prominence)",
            f"- **Treatment split count**: {t_split:,}",
            f"- **Treatment split gain**: {t_gain:.4f}",
            f"- **Treatment feature rank**: #{t_rank} of {tot_feat} features",
            f"- **Diagnostic status**: {warn}",
        ]

    qini_pw = _csv("uplift_pairwise_qini.csv")
    if qini_pw is not None:
        out += [
            "",
            "### Pairwise Qini significance tests (500k test sample, 1k bootstrap draws)",
            _table(qini_pw, ".4f"),
        ]

    calib = _csv("uplift_calibration_deciles.csv")
    if calib is not None:
        out += [
            "",
            "### Uplift calibration by score decile (test split)",
            "Decile 10 contains highest predicted uplift (monotonicity check).",
            "",
            _table(calib, ".5f"),
        ]

    opt_reach = _csv("campaign_optimal_reach.csv")
    if opt_reach is not None:
        opt_summary = (
            opt_reach[opt_reach["is_optimal"] == True][  # noqa: E712
                [
                    "strategy",
                    "reach",
                    "targeted",
                    "incremental_conversions",
                    "cost",
                    "net_value",
                ]
            ]
            if "is_optimal" in opt_reach.columns
            else opt_reach
        )
        out += [
            "",
            "### Campaign profit-optimal reach (fine grid 1%-50%)",
            "_SCENARIO - hypothetical cost/value, not observed data_",
            "",
            _table(opt_summary, ".2f"),
        ]

    lc = _csv("uplift_learning_curve.csv")
    if lc is not None:
        out += [
            "",
            "### Uplift training-size learning curve",
            _table(lc, ".4f"),
        ]

    if sim is not None:
        sim_fmt = sim.assign(reach=(sim["reach"] * 100).round().astype(int).astype(str) + "%")
        out += [
            "",
            "### Campaign simulation — incremental outcomes on the test population",
            _table(
                sim_fmt[
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

    figs = []
    if (FIG / "uplift_qini_curves.png").exists():
        figs.append("![Qini Curves](figures/uplift_qini_curves.png)")
    if (FIG / "uplift_calibration_deciles.png").exists():
        figs.append("![Decile Calibration](figures/uplift_calibration_deciles.png)")
    if (FIG / "campaign_net_value_by_reach.png").exists():
        figs.append("![Campaign Net Value vs Reach](figures/campaign_net_value_by_reach.png)")
    if figs:
        out += ["", "### Uplift figures", *figs]
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


def generate_figures() -> None:
    """Generate publication-ready PNG figures in docs/figures/ using colorblind-safe palette."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    cb_blue = "#1f77b4"
    cb_orange = "#ff7f0e"
    cb_green = "#2ca02c"

    # 1. churn_ladder_pr_auc.png
    cmp = _csv("churn_model_comparison.csv")
    if cmp is not None and "model" in cmp.columns and "pr_auc" in cmp.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        models = cmp["model"].tolist()
        pr_aucs = cmp["pr_auc"].tolist()
        yerr = None
        if "pr_auc_ci_low" in cmp.columns and "pr_auc_ci_high" in cmp.columns:
            yerr = [
                cmp["pr_auc"] - cmp["pr_auc_ci_low"],
                cmp["pr_auc_ci_high"] - cmp["pr_auc"],
            ]
        bars = ax.bar(
            models, pr_aucs, yerr=yerr, capsize=4, color=cb_blue, alpha=0.85, edgecolor="black"
        )
        if "is_champion" in cmp.columns:
            for bar, is_champ in zip(bars, cmp["is_champion"], strict=False):
                if is_champ:
                    bar.set_color(cb_orange)
                    bar.set_edgecolor("black")
        man = _json("split_manifest.json")
        if man and "base_rate" in man:
            try:
                base_r = float(str(man["base_rate"]).split()[0].replace("%", "")) / 100.0
                ax.axhline(base_r, color="gray", linestyle="--", label=f"Base Rate ({base_r:.2f})")
            except Exception:
                pass
        ax.set_ylabel("PR-AUC (Test Set)")
        ax.set_title("Inactivity Prediction: Model Ladder PR-AUC with 95% Bootstrap CIs")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(FIG / "churn_ladder_pr_auc.png")
        plt.close(fig)

    # 2. churn_ablation_deltas.png
    deltas = _csv("churn_pairwise_deltas.csv")
    if deltas is not None and "model_a" in deltas.columns and "delta_pr_auc" in deltas.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        labels = [f"{r.model_a} vs {r.model_b}" for _, r in deltas.iterrows()]
        y_pos = range(len(labels))
        d_vals = deltas["delta_pr_auc"].tolist()
        xerr = None
        if "ci_low" in deltas.columns and "ci_high" in deltas.columns:
            xerr = [
                deltas["delta_pr_auc"] - deltas["ci_low"],
                deltas["ci_high"] - deltas["delta_pr_auc"],
            ]
        ax.errorbar(d_vals, y_pos, xerr=xerr, fmt="o", color=cb_blue, ecolor=cb_blue, capsize=4)
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel("PR-AUC Delta (Model A - Model B)")
        ax.set_title("Pairwise PR-AUC Differences with 95% Bootstrap CIs")
        plt.tight_layout()
        fig.savefig(FIG / "churn_ablation_deltas.png")
        plt.close(fig)

    # 3. churn_shap_importance.png
    shap = _csv("shap_global.csv")
    if shap is not None and "feature" in shap.columns and "mean_abs_shap" in shap.columns:
        top10 = shap.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax.barh(
            top10["feature"],
            top10["mean_abs_shap"],
            color=cb_green,
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_xlabel("Mean |SHAP Value| (Impact on Inactivity Prediction)")
        ax.set_title("Top-10 Feature Drivers of Inactivity Risk (TreeSHAP)")
        plt.tight_layout()
        fig.savefig(FIG / "churn_shap_importance.png")
        plt.close(fig)

    # 4. uplift_qini_curves.png
    curves = _csv("uplift_curves.csv")
    if curves is not None and "curve" in curves.columns:
        qini_df = curves[curves["curve"] == "qini"]
        if not qini_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            for strat, g in qini_df.groupby("strategy"):
                ax.plot(g["fraction"], g["value"], label=str(strat), linewidth=1.8)
            ax.set_xlabel("Targeted Population Fraction")
            ax.set_ylabel("Cumulative Incremental Visits")
            ax.set_title("Test Set Qini Curves across Targeting Strategies")
            ax.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            fig.savefig(FIG / "uplift_qini_curves.png")
            plt.close(fig)

    # 5. uplift_calibration_deciles.png
    calib = _csv("uplift_calibration_deciles.csv")
    if calib is not None and "decile" in calib.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        for strat, g in calib.groupby("strategy"):
            g_sorted = g.sort_values("decile")
            yerr = None
            if "ci_low" in g_sorted.columns and "ci_high" in g_sorted.columns:
                yerr = [
                    g_sorted["observed_uplift"] - g_sorted["ci_low"],
                    g_sorted["ci_high"] - g_sorted["observed_uplift"],
                ]
            ax.errorbar(
                g_sorted["decile"],
                g_sorted["observed_uplift"],
                yerr=yerr,
                fmt="o-",
                label=f"{strat} (observed)",
                capsize=3,
            )
            if "predicted_uplift" in g_sorted.columns:
                ax.plot(
                    g_sorted["decile"],
                    g_sorted["predicted_uplift"],
                    "--",
                    label=f"{strat} (predicted)",
                    alpha=0.6,
                )
        ax.set_xlabel("Predicted Uplift Decile (10 = Highest Predicted Uplift)")
        ax.set_ylabel("Incremental Uplift Rate (Treatment - Control)")
        ax.set_title("Decile Uplift Calibration & Monotonicity Check")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG / "uplift_calibration_deciles.png")
        plt.close(fig)

    # 6. campaign_net_value_by_reach.png
    opt_reach = _csv("campaign_optimal_reach.csv")
    if opt_reach is not None and "reach" in opt_reach.columns and "net_value" in opt_reach.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        for strat, g in opt_reach.groupby("strategy"):
            g_sorted = g.sort_values("reach")
            ax.plot(g_sorted["reach"] * 100, g_sorted["net_value"], label=str(strat), linewidth=1.8)
            if "is_optimal" in g_sorted.columns:
                opt_pt = g_sorted[g_sorted["is_optimal"] == True]  # noqa: E712
                if not opt_pt.empty:
                    ax.scatter(
                        opt_pt["reach"] * 100,
                        opt_pt["net_value"],
                        s=100,
                        marker="*",
                        zorder=5,
                        label=f"{strat} optimal ({opt_pt['reach'].iloc[0] * 100:.0f}%)",
                    )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Campaign Reach (% of Test Population)")
        ax.set_ylabel("Scenario Net Value ($)")
        ax.set_title("Campaign Economic Optimization: Net Value vs Reach")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(FIG / "campaign_net_value_by_reach.png")
        plt.close(fig)

    # 7. survival_landmark_km.png
    lm_s = _json("survival_landmark_summary.json")
    if lm_s and "activity_tier_km" in lm_s and lm_s["activity_tier_km"]:
        km_df = pd.DataFrame(lm_s["activity_tier_km"])
        if not km_df.empty and "day" in km_df.columns and "survival" in km_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
            for grp, g in km_df.groupby("group"):
                g_sorted = g.sort_values("day")
                ax.step(
                    g_sorted["day"],
                    g_sorted["survival"],
                    where="post",
                    label=f"Tier: {grp}",
                    linewidth=2,
                )
            ax.set_xlabel("Days from Landmark (Day 7)")
            ax.set_ylabel("Probability Active (Surviving)")
            ax.set_title("Kaplan-Meier Retention by Day-7 Activity Tier")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower left", fontsize=8)
            plt.tight_layout()
            fig.savefig(FIG / "survival_landmark_km.png")
            plt.close(fig)


def _missing(artifact: str) -> str:
    return f"_Not available: `artifacts/{artifact}` was not produced by this run._"


def _fig_links(*figs: tuple[str, str]) -> list[str]:
    """Markdown links for figures that actually exist on disk."""
    links = [f"[{label}](figures/{name})" for label, name in figs if (FIG / name).exists()]
    return [f"_Figures: {' | '.join(links)}_", ""] if links else []


def _findings_churn() -> list[str]:
    cmp = _csv("churn_model_comparison.csv")
    if cmp is None or "is_champion" not in cmp.columns:
        return [_missing("churn_model_comparison.csv"), ""]
    man = _json("split_manifest.json") or {}
    champ = cmp[cmp["is_champion"].astype(bool)].iloc[0]
    out: list[str] = []
    test_base = (man.get("base_rate") or {}).get("test")
    if test_base is not None:
        out.append(
            f"- **Base rate:** {_pct(test_base)} of test-snapshot visitors were inactive in the"
            f" prediction window, so PR-AUC on the inactive class starts at ~{test_base:.3f}"
            " for a random score. The minority who return are the commercially relevant group."
        )
    line = (
        f"- **Champion:** `{champ['model']}` with test PR-AUC {champ['pr_auc']:.4f}"
        f" (95% CI [{champ['pr_auc_ci_low']:.4f}, {champ['pr_auc_ci_high']:.4f}])"
    )
    m0 = cmp[cmp["model"] == "M0_naive_recency"]
    if not m0.empty:
        line += f", vs {m0['pr_auc'].iloc[0]:.4f} for the naive recency baseline"
    out.append(line + ".")
    if "lift_returners@10" in cmp.columns:
        out.append(
            f"- **Returner view:** among the top 10% scored as most likely to return, the"
            f" champion's lift over the returner base rate is {champ['lift_returners@10']:.2f}x."
        )
    sel = _json("churn_selection.json")
    if sel is not None:
        out.append(f"- **Selection rule:** {sel['reasoning']}")
    thr = _csv("churn_threshold_scenarios.csv")
    if thr is not None:
        best = thr.loc[thr["expected_profit"].idxmax()]
        out.append(
            f"- **Operating threshold (SCENARIO, hypothetical economics):** across"
            f" {len(thr)} cost/value scenarios the cost-optimal threshold ranges"
            f" {thr['threshold'].min():.3f}-{thr['threshold'].max():.3f}; the most profitable"
            f" scenario contacts {_pct(best['share_contacted'])} of visitors."
        )
    shap = _csv("shap_global.csv")
    if shap is not None:
        top = ", ".join(f"`{f}`" for f in shap.head(3)["feature"])
        out.append(f"- **Top drivers (mean |SHAP|):** {top}.")
    out.append("")
    out += _fig_links(
        ("PR-AUC Ladder", "churn_ladder_pr_auc.png"),
        ("SHAP Feature Importance", "churn_shap_importance.png"),
    )
    return out


def _findings_survival() -> list[str]:
    s = _json("survival_summary.json")
    lm = _json("survival_landmark_summary.json")
    out: list[str] = []
    c1 = (s or {}).get("cox", {}).get("concordance_test")
    if c1 is None:
        out.append(_missing("survival_summary.json"))
    else:
        out.append(
            f"- **First-session model:** held-out concordance {c1:.3f} (0.50 = random ranking)."
        )
    if lm is None or lm.get("concordance_test") is None:
        out.append(_missing("survival_landmark_summary.json"))
    else:
        c_lm, delta = lm["concordance_test"], lm["concordance_delta"]
        direction = "higher" if delta > 0 else "not higher"
        out.append(
            f"- **Day-{lm['landmark_days']} landmark model:** held-out concordance {c_lm:.3f},"
            f" {direction} than first-session ({delta:+.3f}), on {lm['population']:,} visitors"
            " still observable at the landmark (a different population, so this is not a"
            " like-for-like comparison)."
        )
        p = lm.get("logrank_activity_tier", {}).get("p_value")
        if p is not None:
            out.append(f"- **Week-1 activity tiers:** log-rank p = {p:.2e}.")
    out.append("")
    out += _fig_links(("Day-7 Landmark Retention Curves", "survival_landmark_km.png"))
    return out


def _findings_uplift() -> list[str]:
    exp = _json("experiment_results.json")
    res = _csv("uplift_results.csv")
    if not exp or res is None:
        return [_missing("experiment_results.json / uplift_results.csv"), ""]
    best = exp["selected_strategy"]
    visit = exp["outcomes"].get("visit")
    out: list[str] = []
    if visit:
        out.append(
            f"- **Average effect (randomized, N = {exp['n']:,}):** treatment raised the visit"
            f" rate by {100 * visit['ate']:.3f} pp ({_pct(visit['relative_lift'])} relative)."
        )
    test = res[(res["split"] == "test") & (res["outcome"] == "visit")].set_index("strategy")
    if best in test.index and "response_model" in test.index:
        q_best = test.loc[best, "qini_coefficient"]
        q_resp = test.loc["response_model", "qini_coefficient"]
        verdict = "beats" if q_best > q_resp else "does not beat"
        out.append(
            f"- **Uplift vs propensity:** the strategy selected on validation Qini, `{best}`,"
            f" {verdict} the response (propensity) model on test Qini for visits"
            f" ({q_best:.4f} vs {q_resp:.4f})."
        )
    calib = _csv("uplift_calibration_deciles.csv")
    if calib is not None and visit:
        top = calib[
            (calib["strategy"] == best) & (calib["outcome"] == "visit") & (calib["decile"] == 10)
        ]
        if not top.empty:
            ratio = top["observed_uplift"].iloc[0] / visit["ate"]
            out.append(
                f"- **Top decile:** observed visit uplift in `{best}`'s top score decile is"
                f" {ratio:.1f}x the population average effect."
            )
    opt = _csv("campaign_optimal_reach.csv")
    if opt is not None:
        row = opt[(opt["strategy"] == best) & opt["is_optimal"].astype(bool)]
        if not row.empty:
            out.append(
                f"- **Reach (SCENARIO, hypothetical economics):** under the assumed cost and"
                f" conversion value, `{best}` net value peaks at"
                f" {_pct(row['reach'].iloc[0], 0)} reach."
            )
    out.append("")
    out += _fig_links(
        ("Qini Curves", "uplift_qini_curves.png"),
        ("Decile Uplift Calibration", "uplift_calibration_deciles.png"),
        ("Campaign Net Value vs Reach", "campaign_net_value_by_reach.png"),
    )
    return out


def generate_findings() -> None:
    """Write the executive summary docs/FINDINGS.md.

    Every number and every directional claim ("beats", "higher") is computed from artifacts.
    A missing artifact renders as "not available"; nothing falls back to a typed-in value.
    """
    out_path = ROOT / "docs" / "FINDINGS.md"
    lines = [
        "# RetentionIQ - Executive Findings",
        "",
        "_Generated by `evaluation/render_report.py` from `artifacts/`. Do not edit by hand._",
        "_Full tables and confidence intervals: [RESULTS.md](RESULTS.md)._",
        "",
        "## 1. Inactivity prediction (RetailRocket)",
        *_findings_churn(),
        "## 2. Time to inactivity: first session vs day-7 landmark (RetailRocket)",
        *_findings_survival(),
        "## 3. Uplift targeting (Criteo randomized trial)",
        *_findings_uplift(),
    ]
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"wrote {out_path}")


def main() -> None:
    import sys

    manifest = _json("manifest.json")
    if not manifest or not manifest.get("stages"):
        # Refuse rather than overwrite the committed reports with empty stubs.
        sys.exit(
            "render_report: artifacts/manifest.json is missing or empty - run `make evaluate` "
            "first. docs/RESULTS.md and docs/FINDINGS.md were left unchanged."
        )
    generate_figures()

    stages = manifest["stages"].values()
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

    generate_findings()


if __name__ == "__main__":
    main()
