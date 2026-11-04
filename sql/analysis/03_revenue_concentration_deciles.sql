-- 03_revenue_concentration_deciles.sql
--
-- Business Question:
-- How heavily concentrated is qualifying revenue among top customers? By partitioning customers
-- into 10 equal deciles (decile 10 = top spenders), what share of total revenue is contributed by
-- each tier, and what does the cumulative distribution (Lorenz curve) look like?
--
-- Target Warehouse Marts:
--   - marts.dim_customer: customer-level grain with lifetime_value and lifetime_orders
--
-- Edge Cases Handled:
--   - NTILE(10) with tied revenue: secondary ordering by customer_id guarantees deterministic decile assignment.
--   - Decile numbering convention: decile 10 = top 10% highest lifetime spenders (the whale tier).
--   - Cumulative sum over deciles using window functions: computes cumulative share of total revenue.
--
-- Index & Performance Awareness:
--   - Operates on pre-aggregated dim_customer (one row per customer, ~4-5k rows).
--   - Window function evaluation completes in a single sequential in-memory sort pass.

with ranked_customers as (
    select
        customer_id,
        lifetime_value,
        lifetime_orders,
        ntile(10) over (order by lifetime_value asc, customer_id asc) as revenue_decile
    from marts.dim_customer
    where lifetime_value > 0
),

decile_aggregates as (
    select
        revenue_decile,
        count(*) as customer_count,
        min(lifetime_value) as min_customer_ltv,
        max(lifetime_value) as max_customer_ltv,
        avg(lifetime_value) as avg_customer_ltv,
        avg(lifetime_orders) as avg_customer_orders,
        sum(lifetime_value) as decile_revenue
    from ranked_customers
    group by revenue_decile
),

totals as (
    select sum(decile_revenue) as total_company_revenue
    from decile_aggregates
)

select
    d.revenue_decile,
    d.customer_count,
    round(d.min_customer_ltv::numeric, 2) as min_customer_ltv,
    round(d.max_customer_ltv::numeric, 2) as max_customer_ltv,
    round(d.avg_customer_ltv::numeric, 2) as avg_customer_ltv,
    round(d.avg_customer_orders::numeric, 1) as avg_customer_orders,
    round(d.decile_revenue::numeric, 2) as decile_revenue,
    round((d.decile_revenue / t.total_company_revenue * 100)::numeric, 2) as decile_revenue_share_pct,
    round(
        (sum(d.decile_revenue) over (order by d.revenue_decile asc)
        / t.total_company_revenue * 100)::numeric, 2
    ) as cumulative_revenue_share_pct
from decile_aggregates d
cross join totals t
order by d.revenue_decile desc;
