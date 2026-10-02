import type { ReactNode } from "react";

export function PageHeader({ title, subtitle }: { title: string; subtitle: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      <p className="mt-1 max-w-3xl text-sm text-ink-2">{subtitle}</p>
    </header>
  );
}

export function Card({ title, note, children, className = "" }: {
  title?: string; note?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-border bg-surface-1 p-5 ${className}`}>
      {title && <h2 className="text-sm font-semibold">{title}</h2>}
      {note && <p className="mt-0.5 text-xs text-ink-3">{note}</p>}
      <div className={title ? "mt-4" : ""}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-surface-1 p-4">
      <div className="text-xs font-medium text-ink-2">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

export function Missing({ what }: { what: string }) {
  return (
    <div className="rounded-xl border border-dashed border-border bg-surface-1 p-6 text-sm text-ink-2">
      {what} is not available yet. Start the API and run <code className="font-mono">make evaluate</code>.
    </div>
  );
}

export function DataTable({ rows, columns }: {
  rows: Record<string, ReactNode>[]; columns: { key: string; label: string; align?: "left" | "right" }[];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <thead>
          <tr className="border-b border-border text-xs text-ink-2">
            {columns.map((c) => (
              <th key={c.key} className={`px-2 py-2 font-medium ${c.align === "left" ? "text-left" : "text-right"}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-border last:border-0">
              {columns.map((c) => (
                <td key={c.key} className={`px-2 py-2 ${c.align === "left" ? "text-left" : "text-right"}`}>
                  {r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TIER_STYLE: Record<string, { color: string; icon: string }> = {
  high: { color: "var(--critical)", icon: "▲" },
  medium: { color: "var(--serious)", icon: "■" },
  low: { color: "var(--good)", icon: "●" },
};

export function RiskBadge({ tier }: { tier: string }) {
  const s = TIER_STYLE[tier] ?? TIER_STYLE.low;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border px-2 py-0.5 text-xs font-medium">
      <span aria-hidden style={{ color: s.color }}>{s.icon}</span>
      {tier} risk
    </span>
  );
}
