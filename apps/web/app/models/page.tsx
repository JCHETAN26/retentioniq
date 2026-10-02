import { Bars, Lines } from "@/components/charts";
import { Card, DataTable, Missing, PageHeader } from "@/components/ui";
import { artifact, pct, type Row } from "@/lib/api";

export const dynamic = "force-dynamic";

type Calib = Record<string, { isotonic_applied: boolean; test_brier: number; test_ece: number;
  reliability: { mean_predicted: number; observed_rate: number; n: number }[] }>;

const f3 = (v: unknown) => Number(v).toFixed(3);

export default async function ModelLab() {
  const [cmp, abl, calib, sens, shap, manifest] = await Promise.all([
    artifact<Row[]>("churn_model_comparison.csv"),
    artifact<Row[]>("feature_ablation.csv"),
    artifact<Calib>("calibration_metrics.json"),
    artifact<Row[]>("churn_sensitivity.csv"),
    artifact<Row[]>("shap_global.csv"),
    artifact<{ cutoffs: Record<string, string[]>; rows: Record<string, number>; base_rate: Record<string, number> }>("split_manifest.json"),
  ]);
  if (!cmp) return <><PageHeader title="Model Lab" subtitle="" /><Missing what="Model comparison" /></>;
  const champion = String(cmp.find((r) => r.is_champion === true || r.is_champion === "True")?.model ?? "");
  const calibChamp = calib?.[champion];
  const reliability = calibChamp?.reliability.map((r) => ({ x: r.mean_predicted, observed: r.observed_rate, ideal: r.mean_predicted })) ?? [];

  return (
    <>
      <PageHeader title="Model Lab"
        subtitle={<>Predicting 30-day inactivity on RetailRocket with temporal validation. Primary metric: PR-AUC (test base rate {manifest ? pct(manifest.base_rate.test) : "—"}).
          Every model up the ladder must beat the one below it; whiskers are 95% bootstrap CIs.</>} />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Test PR-AUC by model" note={`Champion (selected on validation): ${champion}`}>
          <Bars data={cmp} label="model" value="pr_auc" ciLow="pr_auc_ci_low" ciHigh="pr_auc_ci_high" highlight={champion} />
        </Card>
        <Card title="Calibration of the champion" note={calibChamp ? `Brier ${f3(calibChamp.test_brier)} · ECE ${f3(calibChamp.test_ece)} · isotonic ${calibChamp.isotonic_applied ? "applied" : "not needed"}` : ""}>
          {reliability.length ? (
            <Lines data={reliability} x="x" series={[{ key: "observed", label: "Observed rate" }]} diagonal
              xFormat={(v) => pct(v, 0)} yFormat={(v) => pct(v, 0)} xLabel="Mean predicted P(inactive)" />
          ) : <Missing what="Calibration" />}
        </Card>
      </div>

      <Card className="mt-4" title="All metrics (test set)">
        <DataTable
          columns={[{ key: "model", label: "Model", align: "left" }, { key: "pr_auc", label: "PR-AUC" }, { key: "roc_auc", label: "ROC-AUC" },
            { key: "p10", label: "Precision@10%" }, { key: "r10", label: "Recall@10%" }, { key: "f1", label: "F1" },
            { key: "brier", label: "Brier" }, { key: "ece", label: "ECE" }, { key: "d", label: "Δ PR-AUC vs M1 (CI)" }]}
          rows={cmp.map((r) => ({
            model: <span className={r.model === champion ? "font-semibold" : ""}>{String(r.model)}</span>,
            pr_auc: f3(r.pr_auc), roc_auc: f3(r.roc_auc), p10: f3(r["precision@10"]), r10: f3(r["recall@10"]),
            f1: f3(r.f1), brier: f3(r.brier), ece: f3(r.ece),
            d: `[${f3(r["delta_pr_auc_vs_M1_logistic_rfm_ci_low"])}, ${f3(r["delta_pr_auc_vs_M1_logistic_rfm_ci_high"])}]`,
          }))}
        />
      </Card>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Where the improvement comes from" note="Cumulative feature-family ablation (LightGBM). Δ is vs the previous step, with paired-bootstrap CI.">
          {abl ? (
            <DataTable
              columns={[{ key: "step", label: "+ family", align: "left" }, { key: "n", label: "Features" },
                { key: "pr", label: "Test PR-AUC" }, { key: "d", label: "Δ" }, { key: "ci", label: "95% CI" }]}
              rows={abl.filter((r) => r.model === "M4_lightgbm").map((r) => ({
                step: String(r.step), n: String(r.n_features), pr: f3(r.test_pr_auc),
                d: r.delta_test_pr_auc == null ? "—" : (Number(r.delta_test_pr_auc) >= 0 ? "+" : "") + f3(r.delta_test_pr_auc),
                ci: r.delta_ci_low == null ? "—" : `[${f3(r.delta_ci_low)}, ${f3(r.delta_ci_high)}]`,
              }))}
            />
          ) : <Missing what="Ablation" />}
        </Card>
        <Card title="Global drivers of inactivity" note="Mean |SHAP| on the test snapshot (LightGBM)">
          {shap ? <Bars data={shap.slice(0, 10).map((r) => ({ ...r, feature: String(r.feature).replaceAll("_", " ") }))}
            label="feature" value="mean_abs_shap" format={(v) => v.toFixed(3)} /> : <Missing what="SHAP" />}
        </Card>
      </div>

      {sens && (
        <Card className="mt-4" title="Sensitivity to the label definition" note="Test PR-AUC under alternative windows, session gaps and eligibility rules">
          <DataTable
            columns={[{ key: "setting", label: "Setting", align: "left" }, { key: "base", label: "Base rate" },
              { key: "m1", label: "M1 logistic (RFM)" }, { key: "m4", label: "M4 LightGBM" }]}
            rows={[...new Set(sens.map((r) => String(r.setting)))].map((s) => {
              const m1 = sens.find((r) => r.setting === s && r.model === "M1_logistic_rfm");
              const m4 = sens.find((r) => r.setting === s && r.model === "M4_lightgbm");
              return { setting: s, base: pct(Number(m4?.base_rate)), m1: f3(m1?.pr_auc), m4: f3(m4?.pr_auc) };
            })}
          />
        </Card>
      )}
    </>
  );
}
