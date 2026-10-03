import { Heatmap, Intervals, Lines } from "@/components/charts";
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
const VALUE_TIERS = ["Q4 highest first order", "Q1 lowest first order"];

type VisitorSeg = { chosen_k: number;
  comparison: Record<string, { n_groups: number; test_inactivity_auc: number }> };

function curve(rows: Row[], groupType: string, groups: string[], maxMonth: number) {
  return Array.from({ length: maxMonth + 1 }, (_, m) => {
    const row: Record<string, number> = { month: m };
    for (const g of groups) {
      const hit = rows.find((r) => r.group_type === groupType && r.group === g && Number(r.month_number ?? r.week_number) === m);
      if (hit) row[g] = Number(hit.retention_rate);
    }
    return row;
  });
}

export default async function CustomerIntelligence() {
  const [seg, cohorts, vseg, vcoh] = await Promise.all([
    artifact<Seg>("segmentation_metrics.json"),
    artifact<Row[]>("cohort_metrics.csv"),
    artifact<VisitorSeg>("visitor_segmentation.json"),
    artifact<Row[]>("visitor_category_cohorts.csv"),
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
    .map(([k, v]) => ({ name: { rfm_segments: "RFM segments", kmeans_clusters: "K-Means clusters", gmm_clusters: "GMM clusters", hdbscan_clusters: "HDBSCAN clusters" }[k] ?? k,
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
          <Intervals rows={comparison.map((c) => ({ label: c.name, value: c.auc }))} highlight="RFM segments" />
          <p className="mt-3 text-xs text-ink-3">
            {num(seg.customers_at_cutoff)} customers at cutoff; {pct(seg.future_active_rate)} purchased again in the window.
          </p>
        </Card>
      </div>

      <Card className="mt-4" title="Cohort retention" note="Share of each first-purchase cohort with a qualifying purchase in month M (only fully observed months)">
        {matrix.length ? (
          <Heatmap rows={cohortRows} cols={Array.from({ length: 13 }, (_, i) => i)}
            cells={Object.fromEntries(lookup)} rowLabel="Cohort" />
        ) : <Missing what="Cohort data" />}
      </Card>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Retention curves by RFM segment" note="Customers with ≥12 observable months; segment assigned at the evaluation cutoff">
          <Lines data={curves} x="month" series={CURVE_SEGMENTS.map((s) => ({ key: s, label: s }))}
            xFmt="month" yFmt="pct0" />
        </Card>
        <Card title="Retention by first-order value" note="Top vs bottom quartile of first order value (known at acquisition, so no look-ahead)">
          <Lines data={curve(cohorts ?? [], "first_order_value", VALUE_TIERS, 12)} x="month"
            series={VALUE_TIERS.map((s) => ({ key: s, label: s }))} xFmt="month" yFmt="pct0" />
        </Card>
      </div>

      {vseg && (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          <Card title="RetailRocket visitors: behavioural segments vs recency × frequency"
            note={`AUC of predicting 30-day inactivity on the test snapshot from segment membership alone (k = ${vseg.chosen_k}).`}>
            <Intervals rows={Object.entries(vseg.comparison).map(([k, v]) => ({
              label: k === "rfm_grid" ? `R×F grid (${v.n_groups})` : `K-Means (${v.n_groups})`,
              value: v.test_inactivity_auc }))} />
          </Card>
          <Card title="Weekly retention by category of first item viewed" note="RetailRocket first-visit weekly cohorts with 8 observable weeks">
            {vcoh ? (
              <DataTable
                columns={[{ key: "g", label: "First category", align: "left" }, { key: "n", label: "Visitors" },
                  { key: "w1", label: "W1" }, { key: "w4", label: "W4" }, { key: "w8", label: "W8" }]}
                rows={[...new Set(vcoh.map((r) => String(r.group)))].map((g) => {
                  const at = (w: number) => vcoh.find((r) => r.group === g && Number(r.week_number) === w);
                  return { g, n: num(Number(at(0)?.cohort_size)), w1: pct(Number(at(1)?.retention_rate ?? 0), 2),
                    w4: pct(Number(at(4)?.retention_rate ?? 0), 2), w8: pct(Number(at(8)?.retention_rate ?? 0), 2) };
                })}
              />
            ) : <Missing what="Category cohorts" />}
          </Card>
        </div>
      )}
    </>
  );
}
