"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const AXIS = { stroke: "var(--grid)", tick: { fill: "var(--text-secondary)", fontSize: 12 } };
const fmtPct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;

type TipItem = { name?: string; value?: number | string; color?: string; payload?: Record<string, unknown> };

function TipBox({ title, items, format }: { title: ReactNode; items: TipItem[]; format: (v: number) => string }) {
  return (
    <div className="rounded-lg border border-border bg-surface-1 px-3 py-2 text-xs shadow-md">
      <div className="mb-1 font-medium text-ink">{title}</div>
      {items.map((it, i) => (
        <div key={i} className="flex items-center gap-2 text-ink-2">
          <span className="h-2 w-2 rounded-full" style={{ background: it.color }} />
          <span>{it.name}</span>
          <span className="ml-auto pl-3 font-medium text-ink tabular-nums">
            {typeof it.value === "number" ? format(it.value) : it.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Multi-series line chart (<= 3 series, palette slots 1-3) with crosshair tooltip. */
export function Lines({
  data, x, series, xFormat = (v) => String(v), yFormat = (v) => v.toFixed(3), height = 280,
  xLabel, yLabel, diagonal = false,
}: {
  data: Record<string, number>[]; x: string; series: { key: string; label: string }[];
  xFormat?: (v: number) => string; yFormat?: (v: number) => string; height?: number;
  xLabel?: string; yLabel?: string; diagonal?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: xLabel ? 20 : 4, left: 8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey={x} type="number" domain={["dataMin", "dataMax"]} tickFormatter={xFormat} {...AXIS}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, fill: "var(--text-muted)", fontSize: 12 } : undefined} />
        <YAxis tickFormatter={yFormat} width={56} {...AXIS}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 } : undefined} />
        {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-muted)" strokeDasharray="4 4" />}
        <Tooltip
          cursor={{ stroke: "var(--text-muted)", strokeWidth: 1 }}
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <TipBox title={xFormat(Number(label))} items={payload as unknown as TipItem[]} format={yFormat} />
            ) : null
          }
        />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />}
        {series.map((s, i) => (
          <Line key={s.key} dataKey={s.key} name={s.label} stroke={SERIES[i]} strokeWidth={2}
            dot={false} activeDot={{ r: 4, stroke: "var(--surface-1)", strokeWidth: 2 }} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal bars for one measure, optional 95% CI whiskers. */
export function Bars({
  data, label, value, ciLow, ciHigh, format = (v) => v.toFixed(3), height, highlight,
  colorFor,
}: {
  data: Record<string, unknown>[]; label: string; value: string; ciLow?: string; ciHigh?: string;
  format?: (v: number) => string; height?: number; highlight?: string; colorFor?: (row: Record<string, unknown>) => string;
}) {
  const rows: Record<string, unknown>[] = data.map((d) => ({
    ...d,
    __err: ciLow && ciHigh ? [Number(d[value]) - Number(d[ciLow]), Number(d[ciHigh]) - Number(d[value])] : undefined,
  }));
  return (
    <ResponsiveContainer width="100%" height={height ?? 40 + rows.length * 36}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }} barCategoryGap={8}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" tickFormatter={format} {...AXIS} />
        <YAxis type="category" dataKey={label} width={170} {...AXIS} />
        <Tooltip
          cursor={{ fill: "var(--grid)", opacity: 0.5 }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as Record<string, unknown>;
            const items: TipItem[] = [{ name: "value", value: Number(p[value]), color: "var(--series-1)" }];
            if (ciLow && ciHigh) items.push({ name: "95% CI", value: `${format(Number(p[ciLow]))} – ${format(Number(p[ciHigh]))}` });
            return <TipBox title={String(p[label])} items={items} format={format} />;
          }}
        />
        <ReferenceLine x={0} stroke="var(--text-muted)" />
        <Bar dataKey={value} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {rows.map((r, i) => (
            <Cell key={i} fill={colorFor ? colorFor(r) : r[label] === highlight ? "var(--series-1)" : "var(--seq-100)"} />
          ))}
          {ciLow && <ErrorBar dataKey="__err" direction="x" width={6} stroke="var(--text-secondary)" strokeWidth={1.5} />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Vertical column chart for distributions / grouped values (single series). */
export function Columns({
  data, x, y, xFormat = (v) => String(v), yFormat = (v) => String(v), height = 240, ciLow, ciHigh,
}: {
  data: Record<string, number>[]; x: string; y: string; xFormat?: (v: number) => string;
  yFormat?: (v: number) => string; height?: number; ciLow?: string; ciHigh?: string;
}) {
  const rows = data.map((d) => ({ ...d, __err: ciLow && ciHigh ? [d[y] - d[ciLow], d[ciHigh] - d[y]] : undefined }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: 8 }} barCategoryGap={2}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey={x} tickFormatter={xFormat} {...AXIS} />
        <YAxis tickFormatter={yFormat} width={56} {...AXIS} />
        <ReferenceLine y={0} stroke="var(--text-muted)" />
        <Tooltip
          cursor={{ fill: "var(--grid)", opacity: 0.5 }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as Record<string, number>;
            const items: TipItem[] = [{ name: y, value: p[y], color: "var(--series-1)" }];
            if (ciLow && ciHigh) items.push({ name: "95% CI", value: `${yFormat(p[ciLow])} – ${yFormat(p[ciHigh])}` });
            return <TipBox title={xFormat(p[x])} items={items} format={yFormat} />;
          }}
        />
        <Bar dataKey={y} fill="var(--series-1)" radius={[4, 4, 0, 0]} isAnimationActive={false}>
          {ciLow && <ErrorBar dataKey="__err" width={4} stroke="var(--text-secondary)" strokeWidth={1.5} />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Cohort retention heatmap: sequential single-hue ramp, hover shows the exact cell. */
export function Heatmap({ rows, cols, value, rowLabel, colLabel, format = fmtPct }: {
  rows: string[]; cols: number[]; value: (r: string, c: number) => number | undefined;
  rowLabel: string; colLabel: (c: number) => string; format?: (v: number) => string;
}) {
  const vals = rows.flatMap((r) => cols.map((c) => value(r, c))).filter((v): v is number => v !== undefined && v < 1);
  const max = Math.max(...vals, 0.01);
  return (
    <div className="overflow-x-auto">
      <table className="text-xs tabular-nums">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left font-medium text-ink-2">{rowLabel}</th>
            {cols.map((c) => <th key={c} className="px-1 py-1 font-medium text-ink-2">{colLabel(c)}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <td className="whitespace-nowrap px-2 py-0.5 text-ink-2">{r}</td>
              {cols.map((c) => {
                const v = value(r, c);
                if (v === undefined) return <td key={c} />;
                const t = c === 0 ? 1 : Math.min(1, v / max);
                return (
                  <td key={c} className="p-[1px]">
                    <div
                      title={`${r} · ${colLabel(c)}: ${format(v)}`}
                      className="flex h-6 min-w-10 items-center justify-center rounded-[3px]"
                      style={{
                        background: `color-mix(in oklab, var(--seq-700) ${Math.round(t * 100)}%, var(--seq-100))`,
                        color: t > 0.55 ? "var(--surface-1)" : "var(--text-primary)",
                      }}
                    >
                      {format(v)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** SHAP drivers: diverging bars (raises risk = warm, lowers = cool) with value labels. */
export function Drivers({ drivers }: { drivers: { feature: string; value: number; shap: number }[] }) {
  const rows = drivers.map((d) => ({ ...d, label: d.feature.replaceAll("_", " ") }));
  return (
    <Bars
      data={rows}
      label="label"
      value="shap"
      format={(v) => (v > 0 ? "+" : "") + v.toFixed(2)}
      colorFor={(r) => (Number(r.shap) > 0 ? "var(--div-pos)" : "var(--div-neg)")}
    />
  );
}
