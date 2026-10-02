import { Columns } from "@/components/charts";
import { Card, DataTable, Missing, PageHeader, Stat } from "@/components/ui";
import { artifact, num, pct, type Row } from "@/lib/api";

export const dynamic = "force-dynamic";

type Arm = { rate_control: number; rate_treatment: number; ate: number; ate_ci_newcombe: [number, number];
  ate_ci_bootstrap: [number, number]; relative_lift: number; relative_lift_ci: [number, number]; p_value_holm: number;
  n_treatment: number; n_control: number };
type Exp = { n: number; sample_ratio: { observed_share: number; expected_share: number; p_value: number };
  balance: { max_abs_smd: number; passes: boolean; smd: Record<string, number> }; outcomes: Record<string, Arm>; primary_outcome: string };

export default async function ExperimentLab() {
  const [exp, het] = await Promise.all([artifact<Exp>("experiment_results.json"), artifact<Row[]>("uplift_heterogeneity.csv")]);
  if (!exp) return <><PageHeader title="Experiment Lab" subtitle="" /><Missing what="Experiment results" /></>;
  const visit = exp.outcomes.visit;
  const conv = exp.outcomes.conversion;
  const ci = (c: [number, number], d = 4) => `[${(100 * c[0]).toFixed(d)}, ${(100 * c[1]).toFixed(d)}] pp`;
  return (
    <>
      <PageHeader title="Experiment Lab"
        subtitle="Criteo randomized incrementality test, analysed as an experiment before any modelling: randomization checks, then average treatment effects with confidence intervals." />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Users" value={num(exp.n)} hint={`${pct(exp.sample_ratio.observed_share, 2)} treated`} />
        <Stat label="Covariate balance" value={exp.balance.passes ? "Pass" : "Fail"} hint={`max |SMD| ${exp.balance.max_abs_smd.toFixed(4)} (< 0.1)`} />
        <Stat label="Visit lift (ATE)" value={`+${(100 * visit.ate).toFixed(2)} pp`} hint={`relative +${pct(visit.relative_lift)}`} />
        <Stat label="Conversion lift (ATE)" value={`+${(100 * conv.ate).toFixed(3)} pp`} hint={`relative +${pct(conv.relative_lift)}`} />
      </div>
      <Card className="mt-4" title="Average treatment effect" note={`Primary outcome: ${exp.primary_outcome}. Newcombe score CI, bootstrap cross-check, Holm-adjusted p-values.`}>
        <DataTable
          columns={[{ key: "o", label: "Outcome", align: "left" }, { key: "c", label: "Control rate" }, { key: "t", label: "Treatment rate" },
            { key: "ate", label: "ATE" }, { key: "ci", label: "95% CI (Newcombe)" }, { key: "bci", label: "95% CI (bootstrap)" },
            { key: "rel", label: "Relative lift (CI)" }, { key: "p", label: "p (Holm)" }]}
          rows={Object.entries(exp.outcomes).map(([o, r]) => ({
            o, c: pct(r.rate_control, 3), t: pct(r.rate_treatment, 3), ate: `${(100 * r.ate).toFixed(4)} pp`,
            ci: ci(r.ate_ci_newcombe), bci: ci(r.ate_ci_bootstrap),
            rel: `${pct(r.relative_lift)} [${pct(r.relative_lift_ci[0])}, ${pct(r.relative_lift_ci[1])}]`,
            p: r.p_value_holm < 1e-4 ? "< 0.0001" : r.p_value_holm.toFixed(4),
          }))}
        />
        <p className="mt-3 text-xs text-ink-3">
          Sample-ratio check vs the designed {pct(exp.sample_ratio.expected_share, 0)} allocation: p = {exp.sample_ratio.p_value.toExponential(2)}.
          The deviation is a property of the published dataset; with balanced covariates it does not by itself bias the ATE.
        </p>
      </Card>
      {het && (
        <Card className="mt-4" title="Is the effect heterogeneous?" note="Visit ATE within deciles of baseline propensity to visit (test split), with 95% CIs. Uneven effects are what uplift targeting exploits.">
          <Columns data={het.filter((r) => r.outcome === "visit").map((r) => ({ decile: Number(r.decile), ate: Number(r.ate),
            lo: Number(r.ci_low), hi: Number(r.ci_high) }))} x="decile" y="ate" ciLow="lo" ciHigh="hi"
            xFormat={(v) => `D${v}`} yFormat={(v) => `${(100 * v).toFixed(2)} pp`} />
        </Card>
      )}
    </>
  );
}
