import { Columns, Lines } from "@/components/charts";
import { Card, DataTable, Missing, PageHeader } from "@/components/ui";
import { artifact, STRATEGY_LABEL, type Row } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function UpliftLab() {
  const [res, curves, tau, exp] = await Promise.all([
    artifact<Row[]>("uplift_results.csv"), artifact<Row[]>("uplift_curves.csv"),
    artifact<Row[]>("uplift_tau_distribution.csv"), artifact<{ selected_strategy: string }>("experiment_results.json"),
  ]);
  if (!res || !curves) return <><PageHeader title="Uplift Lab" subtitle="" /><Missing what="Uplift results" /></>;
  const best = exp?.selected_strategy ?? "";
  const shown = ["random", "response_model", best];
  const qini = (outcome: string) => {
    const pts = new Map<number, Record<string, number>>();
    for (const r of curves.filter((c) => c.curve === "qini" && c.outcome === outcome && shown.includes(String(c.strategy)))) {
      const f = Math.round(Number(r.fraction) * 100) / 100;
      const row = pts.get(f) ?? { fraction: f };
      row[String(r.strategy)] = Number(r.value);
      pts.set(f, row);
    }
    return [...pts.values()].sort((a, b) => a.fraction - b.fraction);
  };
  const test = res.filter((r) => r.split === "test");
  const series = shown.map((s) => ({ key: s, label: STRATEGY_LABEL[s] ?? s }));
  return (
    <>
      <PageHeader title="Uplift Lab"
        subtitle={<>Who responds <em>because of</em> the treatment. Strategies are trained on 60% of the experiment, selected on validation Qini (visit), and evaluated once on the held-out 20%. Selected: <strong>{STRATEGY_LABEL[best] ?? best}</strong>.</>} />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Qini curve — visits" note="Incremental visits captured as targeting depth grows (test split). Higher and earlier is better.">
          <Lines data={qini("visit")} x="fraction" series={series} xFormat={(v) => `${Math.round(v * 100)}%`} yFormat={(v) => v.toFixed(0)} xLabel="Share of population targeted" />
        </Card>
        <Card title="Qini curve — conversions" note="Secondary outcome; conversions are rare, so curves are noisier.">
          <Lines data={qini("conversion")} x="fraction" series={series} xFormat={(v) => `${Math.round(v * 100)}%`} yFormat={(v) => v.toFixed(0)} xLabel="Share of population targeted" />
        </Card>
      </div>
      <Card className="mt-4" title="Strategy comparison (test split)" note="Qini coefficient and AUUC are normalized: 0 = random, 1 = perfect ranking.">
        <DataTable
          columns={[{ key: "s", label: "Strategy", align: "left" }, { key: "o", label: "Outcome", align: "left" }, { key: "q", label: "Qini" },
            { key: "a", label: "AUUC" }, { key: "u1", label: "Uplift@1%" }, { key: "u10", label: "Uplift@10%" }, { key: "u30", label: "Uplift@30%" }]}
          rows={test.map((r) => ({
            s: <span className={r.strategy === best ? "font-semibold" : ""}>{STRATEGY_LABEL[String(r.strategy)] ?? String(r.strategy)}</span>,
            o: String(r.outcome), q: Number(r.qini_coefficient).toFixed(3), a: Number(r.auuc).toFixed(3),
            u1: `${(100 * Number(r["uplift@1"])).toFixed(2)} pp`, u10: `${(100 * Number(r["uplift@10"])).toFixed(2)} pp`,
            u30: `${(100 * Number(r["uplift@30"])).toFixed(2)} pp`,
          }))}
        />
      </Card>
      {tau && (
        <Card className="mt-4" title="Distribution of estimated treatment effects" note={`τ̂(x) for visits from ${STRATEGY_LABEL[best] ?? best} on the test split`}>
          <Columns data={tau.map((r) => ({ x: (Number(r.bin_left) + Number(r.bin_right)) / 2, count: Number(r.count) }))}
            x="x" y="count" xFormat={(v) => `${(100 * v).toFixed(1)}pp`} yFormat={(v) => v.toLocaleString()} />
        </Card>
      )}
    </>
  );
}
