import { Bars, Heatmap, Lines } from "@/components/charts";
import { Card, DataTable, Missing, PageHeader, Stat } from "@/components/ui";
import { artifact, num, pct, type Row } from "@/lib/api";

export const dynamic = "force-dynamic";

type Seg = {
  evaluation_cutoff: string; horizon_days: number; chosen_k: number; customers_at_cutoff: number;
  future_active_rate: number;
  comparison: Record<string, { n_groups: number; future_activity_auc: number; silhouette?: number } | number>;
  rfm_segments_full_history: Row[]; clusters_at_cutoff: Row[];
  value_concentration: Record<string, number>;
};

const CURVE_SEGMENTS = ["Champions", "Potential Loyalists", "At Risk"];

export default async function CustomerIntelligence() {
  const [seg, cohorts] = await Promise.all([
    artifact<Seg>("segmentation_metrics.json"),
    artifact<Row[]>("cohort_metrics.csv"),
  ]);
  if (!seg) return <><PageHeader title="Customer Intelligence" subtitle="" /><Missing what="Segmentation" /></>;

  const segments = [...seg.rfm_segments_full_history].sort((a, b) => Number(b.share_revenue) - Number(a.share_revenue));
  const totalCustomers = segments.reduce((s, r) => s + Number(r.customers), 0);
  const matrix = (cohorts ?? []).filter((r) => r.group_type === "cohort");
  const cohortRows = [...new Set(matrix.map((r) => String(r.group)))].sort();
  const lookup = new Map(matrix.map((r) => [`${r.group}|${r.month_number}`, Number(r.retention_rate)]));
  const meanAt = (m: number) => {
    const v = cohortRows.map((c) => lookup.get(`${c}|${m}`)).filter((x): x is number => x !== undefined);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
  };
  const curves = Array.from({ length: 13 }, (_, m) => {
    const row: Record<string, number> = { month: m };
    for (const s of CURVE_SEGMENTS) {
      const hit = (cohorts ?? []).find((r) => r.group_type === "rfm_segment" && r.group === s && Number(r.month_number) === m);
      if (hit) row[s] = Number(hit.retention_rate);
    }
    return row;
  });
  const comparison = Object.entries(seg.comparison)
    .filter(([, v]) => typeof v === "object")
    .map(([k, v]) => ({ name: { rfm_segments: "RFM segments", kmeans_clusters: "K-Means clusters", gmm_clusters: "GMM clusters" }[k] ?? k,
      auc: (v as { future_activity_auc: number }).future_activity_auc }));

  return (
    <>
      <PageHeader
        title="Customer Intelligence"
        subtitle="UCI Online Retail II: who the customers are, where the value sits, and how cohorts retain. Qualifying purchases only (PROJECT_SPEC §4.3)."
      />
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Customers" value={num(totalCustomers)} hint="with ≥1 qualifying order" />
        <Stat label="Top 10% revenue share" value={pct(seg.value_concentration.top_10pct_revenue_share)}
          hint={`Gini ${seg.value_concentration.gini.toFixed(2)}`} />
        <Stat label="Month-1 retention" value={pct(meanAt(1))} hint="mean across cohorts" />
        <Stat label="Month-3 retention" value={pct(meanAt(3))} hint="mean across cohorts" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="RFM segments" note="Share of customers vs share of revenue, full history">
          <DataTable
            columns={[
              { key: "segment", label: "Segment", align: "left" }, { key: "customers", label: "Customers" },
              { key: "sc", label: "% customers" }, { key: "sr", label: "% revenue" },
              { key: "aov", label: "AOV" }, { key: "rep", label: "Repeat rate" },
            ]}
            rows={segments.map((r) => ({
              segment: String(r.rfm_segment), customers: num(Number(r.customers)), sc: pct(Number(r.share_customers)),
              sr: pct(Number(r.share_revenue)), aov: `£${num(Number(r.avg_order_value), 0)}`, rep: pct(Number(r.repeat_purchase_rate)),
            }))}
          />
        </Card>
        <Card title="Does learned segmentation beat RFM?"
          note={`Cross-validated AUC of predicting a purchase in the ${seg.horizon_days} days after ${seg.evaluation_cutoff} from segment membership alone (k = ${seg.chosen_k}).`}>
          <Bars data={comparison} label="name" value="auc" format={(v) => v.toFixed(3)} highlight="K-Means clusters" />
          <p className="mt-3 text-xs text-ink-3">
            {num(seg.customers_at_cutoff)} customers at cutoff; {pct(seg.future_active_rate)} purchased again in the window.
          </p>
        </Card>
      </div>

      <Card className="mt-4" title="Cohort retention" note="Share of each first-purchase cohort with a qualifying purchase in month M (only fully observed months)">
        {matrix.length ? (
          <Heatmap rows={cohortRows} cols={Array.from({ length: 13 }, (_, i) => i)}
            value={(r, c) => lookup.get(`${r}|${c}`)} rowLabel="Cohort" colLabel={(c) => `M${c}`} />
        ) : <Missing what="Cohort data" />}
      </Card>

      <Card className="mt-4" title="Retention curves by RFM segment" note="Customers with ≥12 observable months; segment assigned at the evaluation cutoff">
        <Lines data={curves} x="month" series={CURVE_SEGMENTS.map((s) => ({ key: s, label: s }))}
          xFormat={(v) => `M${v}`} yFormat={(v) => pct(v, 0)} />
      </Card>
    </>
  );
}
