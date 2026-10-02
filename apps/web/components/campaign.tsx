"use client";

import { useState } from "react";
import { Bars, Lines, SERIES } from "@/components/charts";
import { Card, Stat } from "@/components/ui";
import { num, STRATEGY_LABEL, type Row } from "@/lib/api";

export function CampaignSimulator({ rows }: { rows: Row[] }) {
  const reaches = [...new Set(rows.map((r) => Number(r.reach)))].sort((a, b) => a - b);
  const strategies = [...new Set(rows.map((r) => String(r.strategy)))];
  const [reach, setReach] = useState(reaches.includes(0.1) ? 0.1 : reaches[0]);
  const [outcome, setOutcome] = useState<"conversions" | "visits">("conversions");
  const key = `incremental_${outcome}`;
  const at = rows.filter((r) => Number(r.reach) === reach);
  const get = (s: string) => at.find((r) => r.strategy === s);
  const uplift = strategies.find((s) => s !== "random" && s !== "response_model") ?? strategies[strategies.length - 1];
  const gainVsProp = Number(get(uplift)?.[key]) - Number(get("response_model")?.[key]);
  const curve = reaches.map((r) => {
    const row: Record<string, number> = { reach: r };
    for (const s of strategies) row[s] = Number(rows.find((x) => Number(x.reach) === r && x.strategy === s)?.[key]);
    return row;
  });
  const population = Number(rows[0]?.population);
  // One color per strategy across every chart on the page (same slot order as the Uplift Lab).
  const colorOf = (s: string) => SERIES[strategies.indexOf(s)] ?? "var(--text-muted)";

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center gap-4 rounded-xl border border-border bg-surface-1 p-4">
        <label className="text-sm font-medium" htmlFor="reach">Campaign reach</label>
        <div id="reach" className="flex gap-1" role="radiogroup">
          {reaches.map((r) => (
            <button key={r} role="radio" aria-checked={r === reach} onClick={() => setReach(r)}
              className={`rounded-lg px-3 py-1.5 text-sm tabular-nums ${r === reach ? "bg-accent font-medium text-white" : "bg-surface-0 text-ink-2"}`}>
              {Math.round(r * 100)}%
            </button>
          ))}
        </div>
        <div className="ml-auto flex gap-1">
          {(["conversions", "visits"] as const).map((o) => (
            <button key={o} onClick={() => setOutcome(o)}
              className={`rounded-lg px-3 py-1.5 text-sm ${o === outcome ? "bg-ink font-medium text-surface-1" : "bg-surface-0 text-ink-2"}`}>{o}</button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {strategies.map((s) => (
          <Stat key={s} label={<span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: colorOf(s) }} />{STRATEGY_LABEL[s] ?? s}</span>} value={num(Number(get(s)?.[key]))}
            hint={`95% CI ${num(Number(get(s)?.[`${key}_ci_low`]))} – ${num(Number(get(s)?.[`${key}_ci_high`]))}`} />
        ))}
      </div>
      <p className="mt-3 text-sm text-ink-2">
        Treating {num(Number(at[0]?.targeted))} of {num(population)} users ({Math.round(reach * 100)}%): uplift targeting yields{" "}
        <strong className="text-ink">{gainVsProp >= 0 ? "+" : ""}{num(gainVsProp)}</strong> incremental {outcome} vs targeting the highest propensity.
      </p>
      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title={`Incremental ${outcome} at ${Math.round(reach * 100)}% reach`} note="Point estimate with 95% multinomial-bootstrap CI">
          <Bars data={at.map((r) => ({ ...r, name: STRATEGY_LABEL[String(r.strategy)] ?? String(r.strategy) }))} label="name" value={key}
            ciLow={`${key}_ci_low`} ciHigh={`${key}_ci_high`} fmt="int"
            colorFor={(r) => colorOf(String(r.strategy))} />
        </Card>
        <Card title="Across campaign sizes" note="Where uplift targeting's advantage is largest">
          <Lines data={curve} x="reach" series={strategies.map((s) => ({ key: s, label: STRATEGY_LABEL[s] ?? s }))}
            xFmt="reach" yFmt="int" xLabel="Campaign reach" xTicks={reaches}
            colors={strategies.map(colorOf)} />
        </Card>
      </div>
      <p className="mt-4 text-xs text-ink-3">{String(rows[0]?.scenario_note ?? "")}: the CSV also carries a net-value column under a hypothetical cost per treatment and value per conversion.</p>
    </>
  );
}
