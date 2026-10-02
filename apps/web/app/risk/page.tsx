import Link from "next/link";
import { Drivers } from "@/components/charts";
import { Card, DataTable, Missing, PageHeader, RiskBadge, Stat } from "@/components/ui";
import { api, num, pct } from "@/lib/api";

export const dynamic = "force-dynamic";

type Visitor = {
  visitor_id: number; cutoff: string; p_inactive: number; risk_tier: string; actual_inactive: number;
  features: Record<string, number>; drivers: { feature: string; value: number; shap: number; direction: string }[];
  survival: { age_days: number; survival: Record<string, number>; beyond_observed_follow_up: string[] } | null;
};
type ListRow = { visitor_id: number; p_inactive: number; risk_tier: string; sessions_obs: number; recency_days: number };

const KEY_FEATURES = ["recency_days", "sessions_obs", "sessions_7d", "views_30d", "transactions_obs", "cart_rate", "activity_trend", "tenure_days"];

export default async function CustomerRisk(props: PageProps<"/risk">) {
  const sp = await props.searchParams;
  const id = typeof sp.visitor === "string" ? sp.visitor : undefined;
  const tier = typeof sp.tier === "string" ? sp.tier : "high";
  let list: ListRow[] = [];
  let visitor: Visitor | null = null;
  let error: string | null = null;
  try {
    list = await api<ListRow[]>(`/v1/churn/visitors?tier=${tier}&limit=12`);
    if (id) visitor = await api<Visitor>(`/v1/churn/visitors/${encodeURIComponent(id)}`);
  } catch (e) {
    error = (e as Error).message;
  }

  return (
    <>
      <PageHeader title="Customer Risk"
        subtitle="RetailRocket visitors on the held-out test snapshot: predicted probability of no activity in the next 30 days, the drivers behind it, and time-to-inactivity." />
      <form className="mb-4 flex flex-wrap gap-2" action="/risk">
        <input name="visitor" defaultValue={id} placeholder="Visitor ID"
          className="w-48 rounded-lg border border-border bg-surface-1 px-3 py-2 text-sm" />
        <button className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white">Look up</button>
      </form>
      {error && !visitor && <Missing what={`Risk data (${error})`} />}

      {visitor && (
        <div className="mb-4 grid gap-4 xl:grid-cols-3">
          <Card title={`Visitor ${visitor.visitor_id}`} note={`Scored at cutoff ${visitor.cutoff.slice(0, 10)}`}>
            <div className="text-4xl font-semibold tabular-nums">{pct(visitor.p_inactive, 0)}</div>
            <div className="mt-2 flex items-center gap-2"><RiskBadge tier={visitor.risk_tier} />
              <span className="text-xs text-ink-3">actual: {visitor.actual_inactive ? "became inactive" : "stayed active"}</span></div>
            {visitor.survival && (
              <div className="mt-5">
                <div className="text-xs font-medium text-ink-2">Still active (Cox, conditional on {num(visitor.survival.age_days)} days so far)</div>
                <div className="mt-2 grid grid-cols-3 gap-2">
                  {Object.entries(visitor.survival.survival).map(([h, s]) => (
                    <Stat key={h} label={h} value={pct(s, 0)}
                      hint={visitor.survival!.beyond_observed_follow_up.includes(h) ? "extrapolated" : undefined} />
                  ))}
                </div>
              </div>
            )}
          </Card>
          <Card className="xl:col-span-2" title="Why this score" note="Top SHAP contributions (LightGBM). Warm = raises inactivity risk; cool = lowers it.">
            {visitor.drivers.length ? <Drivers drivers={visitor.drivers} /> : <p className="text-sm text-ink-3">No explainer loaded.</p>}
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {KEY_FEATURES.map((f) => (
                <div key={f} className="rounded-lg bg-surface-0 px-3 py-2">
                  <div className="text-[11px] text-ink-3">{f.replaceAll("_", " ")}</div>
                  <div className="text-sm font-medium tabular-nums">{num(visitor.features[f], 2)}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      <Card title="Highest-risk visitors" note={
        <span>Filter: {["high", "medium", "low"].map((t) => (
          <Link key={t} href={`/risk?tier=${t}`} className={`ml-2 ${t === tier ? "font-semibold text-ink" : "underline"}`}>{t}</Link>))}</span>}>
        {list.length ? (
          <DataTable
            columns={[{ key: "id", label: "Visitor", align: "left" }, { key: "p", label: "P(inactive)" },
              { key: "tier", label: "Tier" }, { key: "s", label: "Sessions (30d)" }, { key: "r", label: "Days since last visit" }]}
            rows={list.map((r) => ({
              id: <Link className="text-accent underline" href={`/risk?visitor=${r.visitor_id}&tier=${tier}`}>{r.visitor_id}</Link>,
              p: pct(r.p_inactive), tier: <RiskBadge tier={r.risk_tier} />, s: num(r.sessions_obs), r: num(r.recency_days, 1),
            }))}
          />
        ) : <p className="text-sm text-ink-3">No visitors in this tier.</p>}
      </Card>
    </>
  );
}
