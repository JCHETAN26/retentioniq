"""Exploratory data analysis (Phase 6) -> docs/EDA.md + artifacts/eda_summary.json.

EDA comes before modelling and produces *hypotheses*, each tied to the phase that tests it.
The hypothesis wording is fixed here; every number is computed from the warehouse marts.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine

# Validated reference palette (single series + recessive structure), light mode.
INK, INK2, GRID, SERIES = "#0b0b0b", "#52514e", "#ebe9e4", "#2a78d6"


def _q(series: pd.Series, qs: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.99)) -> dict[str, float]:
    return {f"p{int(q * 100)}": float(series.quantile(q)) for q in qs}


def _style(ax: Any, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def online_retail(engine: Any) -> dict[str, Any]:
    orders = pd.read_sql(
        "SELECT customer_id, order_ts, order_date, order_value, n_units, customer_order_number, "
        "days_since_previous_order FROM marts.fact_orders",
        engine,
        parse_dates=["order_ts", "order_date"],
    )
    lines = pd.read_sql(
        "SELECT is_cancellation, has_customer, is_non_product, is_qualifying, line_amount "
        "FROM marts.fact_transactions",
        engine,
    )
    per_cust = orders.groupby("customer_id").agg(
        orders=("order_value", "size"),
        first=("order_date", "min"),
        last=("order_date", "max"),
    )
    gross = lines.loc[lines["line_amount"] > 0, "line_amount"].sum()
    monthly = orders.set_index("order_ts")["order_value"].resample("MS").sum()
    return {
        "customers": len(per_cust),
        "orders": len(orders),
        "repeat_customer_share": float((per_cust["orders"] >= 2).mean()),
        "orders_per_customer": _q(per_cust["orders"]),
        "tenure_days": _q((per_cust["last"] - per_cust["first"]).dt.days),
        "days_between_orders": _q(orders["days_since_previous_order"].dropna()),
        "order_value": _q(orders["order_value"]),
        "units_per_order": _q(orders["n_units"]),
        "cancelled_line_share": float(lines["is_cancellation"].mean()),
        "cancelled_value_share_of_gross": float(
            -lines.loc[lines["is_cancellation"], "line_amount"].sum() / gross
        ),
        "lines_without_customer_share": float((~lines["has_customer"]).mean()),
        "value_without_customer_share": float(
            lines.loc[~lines["has_customer"] & (lines["line_amount"] > 0), "line_amount"].sum()
            / gross
        ),
        "revenue_by_weekday_share": (
            orders.groupby(orders["order_ts"].dt.day_name())["order_value"].sum()
            / orders["order_value"].sum()
        )
        .round(4)
        .to_dict(),
        "peak_month": str(monthly.idxmax())[:10],
        "trough_month": str(monthly.iloc[1:-1].idxmin())[:10],
        "_monthly": monthly,
    }


def retailrocket(engine: Any) -> dict[str, Any]:
    sessions = pd.read_sql(
        "SELECT visitor_id, session_start_ts, n_views, n_carts, n_transactions "
        "FROM intermediate.int_retailrocket__sessions",
        engine,
        parse_dates=["session_start_ts"],
    )
    funnel = pd.read_sql("SELECT * FROM marts.mart_engagement_funnel", engine)
    per_visitor = sessions.groupby("visitor_id").agg(
        sessions=("n_views", "size"), purchases=("n_transactions", "sum")
    )
    s = sessions.sort_values(["visitor_id", "session_start_ts"])
    gaps_days = s.groupby("visitor_id")["session_start_ts"].diff().dt.total_seconds() / 86_400
    daily = sessions.groupby(sessions["session_start_ts"].dt.floor("D")).size().rename("sessions")
    tot = funnel[
        ["sessions", "sessions_with_view", "sessions_with_cart", "sessions_with_transaction"]
    ].sum()
    return {
        "visitors": len(per_visitor),
        "sessions": len(sessions),
        "single_session_visitor_share": float((per_visitor["sessions"] == 1).mean()),
        "sessions_per_visitor": _q(per_visitor["sessions"]),
        "purchasing_visitor_share": float((per_visitor["purchases"] > 0).mean()),
        "session_view_to_cart": float(tot["sessions_with_cart"] / tot["sessions_with_view"]),
        "session_cart_to_purchase": float(
            tot["sessions_with_transaction"] / tot["sessions_with_cart"]
        ),
        "session_conversion": float(tot["sessions_with_transaction"] / tot["sessions"]),
        "inter_session_gap_days": _q(gaps_days.dropna()),
        "gap_over_30d_share_of_return_gaps": float((gaps_days.dropna() > 30).mean()),
        "days_covered": int(daily.index.nunique()),
        "_daily": daily,
        "_sessions_per_visitor": per_visitor["sessions"],
    }


def criteo(engine: Any) -> dict[str, Any]:
    arms = pd.read_sql("SELECT * FROM marts.mart_experiment_results ORDER BY treatment", engine)
    n = int(arms["n"].sum())
    return {
        "users": n,
        "treatment_share": float(arms.loc[arms["treatment"] == 1, "n"].sum() / n),
        "visit_rate": float(arms["visits"].sum() / n),
        "conversion_rate": float(arms["conversions"].sum() / n),
        "conversion_given_visit": float(arms["conversions"].sum() / arms["visits"].sum()),
        "exposure_rate_treated": float(
            arms.loc[arms["treatment"] == 1, "exposures"].sum()
            / arms.loc[arms["treatment"] == 1, "n"].sum()
        ),
    }


def _figures(rr: dict[str, Any], orr: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    d = rr["_daily"]
    ax.plot(d.index, d.to_numpy(), color=SERIES, linewidth=1.6)
    _style(ax, "RetailRocket — sessions per day")
    fig.tight_layout()
    fig.savefig(out / "eda_rr_daily_sessions.png")
    names.append("eda_rr_daily_sessions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    counts = rr["_sessions_per_visitor"].clip(upper=10).value_counts().sort_index()
    labels = [str(int(i)) if i < 10 else "10+" for i in counts.index]
    # Points, not bars: a log axis has no zero, so bar lengths would carry no meaning.
    ax.plot(labels, counts.to_numpy(), color=SERIES, linewidth=1.6, marker="o", markersize=6)
    ax.set_yscale("log")
    _style(ax, "RetailRocket — visitors by number of sessions (log scale)")
    fig.tight_layout()
    fig.savefig(out / "eda_rr_sessions_per_visitor.png")
    names.append("eda_rr_sessions_per_visitor.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    m = orr["_monthly"]
    ax.bar(m.index.strftime("%Y-%m"), m.to_numpy() / 1000, color=SERIES, width=0.8)
    ax.tick_params(axis="x", labelrotation=90)
    _style(ax, "Online Retail II — qualifying revenue per month (£ thousands)")
    fig.tight_layout()
    fig.savefig(out / "eda_or_monthly_revenue.png")
    names.append("eda_or_monthly_revenue.png")
    plt.close(fig)
    return names


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def render(orr: dict[str, Any], rr: dict[str, Any], cr: dict[str, Any], figs: list[str]) -> str:
    gap = rr["inter_session_gap_days"]
    return "\n".join(
        [
            "# RetentionIQ — Exploratory data analysis",
            "",
            "_Generated by `retentioniq evaluate --only eda` from the warehouse marts. Numbers are "
            "computed, not typed._",
            "",
            "## Online Retail II (transactions)",
            "",
            f"- {orr['customers']:,} customers placed {orr['orders']:,} qualifying orders; "
            f"{_pct(orr['repeat_customer_share'])} ordered more than once.",
            f"- Orders per customer — median {orr['orders_per_customer']['p50']:.0f}, p90 "
            f"{orr['orders_per_customer']['p90']:.0f}, p99 {orr['orders_per_customer']['p99']:.0f}: "
            "a long tail of heavy buyers.",
            f"- Order value — median £{orr['order_value']['p50']:,.0f}, p90 "
            f"£{orr['order_value']['p90']:,.0f}, p99 £{orr['order_value']['p99']:,.0f}.",
            f"- Median gap between a customer's orders: {orr['days_between_orders']['p50']:.0f} days "
            f"(p75 {orr['days_between_orders']['p75']:.0f}).",
            f"- Cancellations are {_pct(orr['cancelled_line_share'])} of lines and "
            f"{_pct(orr['cancelled_value_share_of_gross'])} of gross line value; lines without a "
            f"customer ID are {_pct(orr['lines_without_customer_share'])} of lines "
            f"({_pct(orr['value_without_customer_share'])} of gross value) and are excluded from "
            "customer analytics (PROJECT_SPEC §4.3).",
            f"- Strong seasonality: revenue peaks in {orr['peak_month']} and is lowest in "
            f"{orr['trough_month']} (partial edge months excluded).",
            "",
            f"![monthly revenue](figures/{figs[2]})",
            "",
            "## RetailRocket (behaviour)",
            "",
            f"- {rr['visitors']:,} visitors, {rr['sessions']:,} sessions over {rr['days_covered']} "
            f"days; **{_pct(rr['single_session_visitor_share'])} of visitors never return** after "
            "their first session.",
            f"- Only {_pct(rr['purchasing_visitor_share'], 2)} of visitors ever purchase. Session "
            f"funnel: view→cart {_pct(rr['session_view_to_cart'])}, cart→purchase "
            f"{_pct(rr['session_cart_to_purchase'])}, overall {_pct(rr['session_conversion'], 2)}.",
            f"- Among returning visitors, the gap between sessions has median "
            f"{gap['p50']:.1f} days (p75 {gap['p75']:.1f}, p90 {gap['p90']:.1f}); "
            f"{_pct(rr['gap_over_30d_share_of_return_gaps'])} of return gaps exceed 30 days.",
            "",
            f"![sessions per visitor](figures/{figs[1]})",
            "",
            f"![daily sessions](figures/{figs[0]})",
            "",
            "## Criteo (experiment)",
            "",
            f"- {cr['users']:,} users, {_pct(cr['treatment_share'], 2)} treated. Visit rate "
            f"{_pct(cr['visit_rate'], 2)}, conversion rate {_pct(cr['conversion_rate'], 3)} "
            f"({_pct(cr['conversion_given_visit'], 1)} of visits convert). Only "
            f"{_pct(cr['exposure_rate_treated'], 1)} of treated users were actually exposed — "
            "`exposure` is post-treatment and excluded from models (§4.4).",
            "",
            "## Hypotheses carried into modelling",
            "",
            "| # | Hypothesis (from the evidence above) | Where it is tested |",
            "|---|---|---|",
            f"| H1 | With {_pct(rr['single_session_visitor_share'])} single-session visitors, "
            '"inactivity" for all visitors is trivial; the modelling population must be '
            "returning visitors. | §4.1 eligibility; `min_sessions` sensitivity |",
            "| H2 | Recency *relative to a visitor's own rhythm* predicts inactivity better than raw "
            "recency, because session gaps vary widely between visitors. | Temporal family in the "
            "ablation; SHAP |",
            f"| H3 | Carts and purchases are rare (view→cart {_pct(rr['session_view_to_cart'])}), so "
            "funnel features add little beyond engagement volume. | Funnel step of the ablation |",
            f"| H4 | Value is concentrated in a small set of frequent buyers (median "
            f"{orr['orders_per_customer']['p50']:.0f} order, p99 "
            f"{orr['orders_per_customer']['p99']:.0f}); simple RFM already captures most of the "
            "structure that predicts future buying. | RFM vs learned segmentation |",
            "| H5 | Retention drops sharply after the first month and then flattens. | Cohort "
            "matrices and curves |",
            f"| H6 | With a {_pct(cr['visit_rate'], 1)} visit rate and a "
            f"{_pct(cr['conversion_rate'], 2)} conversion rate, treatment effects on conversions "
            "will be too sparse to rank reliably below the most engaged users. | Heterogeneity "
            "deciles; uplift vs propensity targeting |",
            "",
        ]
    )


def run(database_url: str, artifacts: Path, docs_dir: Path) -> dict[str, Any]:
    engine = create_engine(database_url)
    orr, rr, cr = online_retail(engine), retailrocket(engine), criteo(engine)
    figs = _figures(rr, orr, docs_dir / "figures")
    (docs_dir / "EDA.md").write_text(render(orr, rr, cr, figs))
    summary = {
        "online_retail": {k: v for k, v in orr.items() if not k.startswith("_")},
        "retailrocket": {k: v for k, v in rr.items() if not k.startswith("_")},
        "criteo": cr,
    }
    (artifacts / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return {
        "single_session_share": round(rr["single_session_visitor_share"], 4),
        "repeat_customer_share": round(orr["repeat_customer_share"], 4),
        "figures": len(figs),
    }
