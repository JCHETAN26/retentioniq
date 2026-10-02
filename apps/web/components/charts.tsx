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

// Server pages can only pass serializable props to client charts, so formats are named.
const FORMATS = {
  pct0: (v: number) => `${(v * 100).toFixed(0)}%`,
  pct1: (v: number) => `${(v * 100).toFixed(1)}%`,
  f3: (v: number) => v.toFixed(3),
  int: (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 }),
  pp2: (v: number) => `${(100 * v).toFixed(2)} pp`,
  pp1s: (v: number) => `${(100 * v).toFixed(1)} pp`,
  pp1: (v: number) => `${(100 * v).toFixed(1)}pp`,
  signed2: (v: number) => (v > 0 ? "+" : "") + v.toFixed(2),
  month: (v: number) => `M${v}`,
  reach: (v: number) => `${Math.round(v * 100)}%`,
  decile: (v: number) => `D${v}`,
  raw: (v: number) => String(v),
} satisfies Record<string, (v: number) => string>;
export type Fmt = keyof typeof FORMATS;
const AXIS = { stroke: "var(--grid)", tick: { fill: "var(--text-secondary)", fontSize: 12 } };

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
  data, x, series, xFmt = "raw", yFmt = "f3", height = 280, xLabel, yLabel, xTicks, colors,
  dashedLast = false,
}: {
  data: Record<string, number>[]; x: string; series: { key: string; label: string }[];
  xFmt?: Fmt; yFmt?: Fmt; height?: number; xLabel?: string; yLabel?: string; xTicks?: number[];
  colors?: string[]; dashedLast?: boolean;
}) {
  const xFormat = FORMATS[xFmt];
  const yFormat = FORMATS[yFmt];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 16, bottom: xLabel ? 20 : 4, left: 8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey={x} type="number" domain={["dataMin", "dataMax"]} ticks={xTicks} tickFormatter={xFormat} {...AXIS}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, fill: "var(--text-muted)", fontSize: 12 } : undefined} />
        <YAxis tickFormatter={yFormat} width={56} {...AXIS}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 } : undefined} />
        <Tooltip
          cursor={{ stroke: "var(--text-muted)", strokeWidth: 1 }}
          content={({ active, payload, label }) =>
            active && payload?.length ? (
              <TipBox title={xFormat(Number(label))} items={payload as unknown as TipItem[]} format={yFormat} />
            ) : null
          }
        />
        {series.length > 1 && (
          <Legend verticalAlign="top" align="right" height={28} iconType="plainline"
            wrapperStyle={{ fontSize: 12 }}
            formatter={(v) => <span style={{ color: "var(--text-secondary)" }}>{v}</span>} />
        )}
        {series.map((s, i) => (
          <Line key={s.key} dataKey={s.key} name={s.label}
            stroke={dashedLast && i === series.length - 1 ? "var(--text-muted)" : (colors ?? SERIES)[i]}
            strokeDasharray={dashedLast && i === series.length - 1 ? "4 4" : undefined} strokeWidth={2}
            dot={false} activeDot={{ r: 4, stroke: "var(--surface-1)", strokeWidth: 2 }} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal bars for one measure, optional 95% CI whiskers. */
export function Bars({
  data, label, value, ciLow, ciHigh, fmt = "f3", height, highlight, colorFor,
}: {
  data: Record<string, unknown>[]; label: string; value: string; ciLow?: string; ciHigh?: string;
  fmt?: Fmt; height?: number; highlight?: string; colorFor?: (row: Record<string, unknown>) => string;
}) {
  const format = FORMATS[fmt];
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
  data, x, y, xFmt = "raw", yFmt = "raw", height = 240, ciLow, ciHigh,
}: {
  data: Record<string, number>[]; x: string; y: string; xFmt?: Fmt; yFmt?: Fmt; height?: number;
  ciLow?: string; ciHigh?: string;
}) {
  const xFormat = FORMATS[xFmt];
  const yFormat = FORMATS[yFmt];
  const rows = data.map((d) => ({ ...d, __err: ciLow && ciHigh ? [d[y] - d[ciLow], d[ciHigh] - d[y]] : undefined }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: 8 }} barCategoryGap={2}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey={x} tickFormatter={xFormat} minTickGap={28} {...AXIS} />
        <YAxis tickFormatter={yFormat} width={72} {...AXIS} />
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
export function Heatmap({ rows, cols, cells, rowLabel, colPrefix = "M", fmt = "pct0" }: {
  rows: string[]; cols: number[]; cells: Record<string, number>; rowLabel: string; colPrefix?: string;
  fmt?: Fmt;
}) {
  const format = FORMATS[fmt];
  const value = (r: string, c: number): number | undefined => cells[`${r}|${c}`];
  const colLabel = (c: number) => `${colPrefix}${c}`;
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
      fmt="signed2"
      colorFor={(r) => (Number(r.shap) > 0 ? "var(--div-pos)" : "var(--div-neg)")}
    />
  );
}

/** Point estimates with optional 95% CIs on a zoomed axis (bars would have to start at 0). */
export function Intervals({ rows, fmt = "f3", highlight }: {
  rows: { label: string; value: number; lo?: number; hi?: number }[]; fmt?: Fmt; highlight?: string;
}) {
  const format = FORMATS[fmt];
  const lo = Math.min(...rows.map((r) => r.lo ?? r.value));
  const hi = Math.max(...rows.map((r) => r.hi ?? r.value));
  const pad = (hi - lo) * 0.15 || 0.01;
  const [d0, d1] = [lo - pad, hi + pad];
  const pos = (v: number) => `${((v - d0) / (d1 - d0)) * 100}%`;
  const ticks = Array.from({ length: 5 }, (_, i) => d0 + ((d1 - d0) * i) / 4);
  return (
    <div className="text-xs">
      {rows.map((r) => {
        const on = r.label === highlight;
        const tip = `${r.label}: ${format(r.value)}${r.lo !== undefined ? ` (95% CI ${format(r.lo)}–${format(r.hi!)})` : ""}`;
        return (
          <div key={r.label} className="group flex items-center gap-3 py-1.5" title={tip}>
            <div className={`w-40 shrink-0 truncate text-right ${on ? "font-semibold text-ink" : "text-ink-2"}`}>{r.label}</div>
            <div className="relative h-5 flex-1 rounded group-hover:bg-surface-0">
              {ticks.map((t, i) => <div key={i} className="absolute inset-y-0 w-px bg-[var(--grid)]" style={{ left: pos(t) }} />)}
              {r.lo !== undefined && r.hi !== undefined && (
                <div className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-[var(--text-secondary)]"
                  style={{ left: pos(r.lo), width: `calc(${pos(r.hi)} - ${pos(r.lo)})` }} />
              )}
              <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[var(--surface-1)]"
                style={{ left: pos(r.value), background: on ? "var(--series-1)" : "var(--text-muted)" }} />
            </div>
            <div className="w-14 shrink-0 text-right tabular-nums text-ink">{format(r.value)}</div>
          </div>
        );
      })}
      <div className="flex gap-3">
        <div className="w-40 shrink-0" />
        <div className="relative h-4 flex-1 text-ink-3">
          {ticks.map((t, i) => (
            <span key={i} className="absolute -translate-x-1/2 tabular-nums" style={{ left: pos(t) }}>{format(t)}</span>
          ))}
        </div>
        <div className="w-14 shrink-0" />
      </div>
    </div>
  );
}
