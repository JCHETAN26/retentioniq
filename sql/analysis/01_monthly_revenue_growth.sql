-- 01_monthly_revenue_growth.sql
--
-- Business Question:
-- What is the month-over-month (MoM) trajectory of total qualifying revenue, active customer count,
-- and average order value (AOV)? Furthermore, how is monthly revenue split between newly acquired
-- customers (making their first purchase) versus returning customers?
--
-- Target Warehouse Marts:
--   - marts.fact_orders: order-level grain with order_value, order_date, and customer_id
--   - marts.dim_customer: customer-level grain with cohort_month (first order month)
--
-- Edge Cases Handled:
--   - First calendar month has no preceding month: MoM percentage changes evaluate gracefully to NULL via LAG().
--   - Multi-order customers within a month: customer-level new vs returning status is determined by their
--     cohort month, preventing classification ambiguity within multi-order months.
--   - Divisor-by-zero protection via NULLIF when computing growth rates and AOV.
--
-- Index & Performance Awareness:
--   - Aggregation scans fact_orders grouped by DATE_TRUNC('month', order_date).
--   - Joining dim_customer on indexed customer_id enables fast hash or merge join.

with monthly_customer_orders as (
    select
        date_trunc('month', o.order_date)::date as order_month,
        o.customer_id,
        c.cohort_month,
        case
            when date_trunc('month', o.order_date)::date = c.cohort_month then 'new'
            else 'returning'
        end as customer_type,
        count(distinct o.order_id) as n_orders,
        sum(o.order_value) as customer_monthly_revenue
    from marts.fact_orders o
    inner join marts.dim_customer c
        on o.customer_id = c.customer_id
    group by 1, 2, 3
),

monthly_aggregates as (
    select
        order_month,
        count(distinct customer_id) as active_customers,
        count(distinct case when customer_type = 'new' then customer_id end) as new_customers,
        count(distinct case when customer_type = 'returning' then customer_id end) as returning_customers,
        sum(n_orders) as total_orders,
        sum(customer_monthly_revenue) as total_revenue,
        sum(case when customer_type = 'new' then customer_monthly_revenue else 0 end) as new_customer_revenue,
        sum(case when customer_type = 'returning' then customer_monthly_revenue else 0 end) as returning_customer_revenue
    from monthly_customer_orders
    group by order_month
)

select
    order_month,
    active_customers,
    new_customers,
    returning_customers,
    total_orders,
    round(total_revenue::numeric, 2) as total_revenue,
    round(new_customer_revenue::numeric, 2) as new_customer_revenue,
    round(returning_customer_revenue::numeric, 2) as returning_customer_revenue,
    round(
        (new_customer_revenue / nullif(total_revenue, 0) * 100)::numeric, 1
    ) as pct_new_customer_revenue,
    round(
        (total_revenue / nullif(total_orders, 0))::numeric, 2
    ) as avg_order_value,
    round(
        ((total_revenue - lag(total_revenue) over (order by order_month))
        / nullif(lag(total_revenue) over (order by order_month), 0) * 100)::numeric, 1
    ) as mom_revenue_growth_pct,
    round(
        ((active_customers - lag(active_customers) over (order by order_month))::numeric
        / nullif(lag(active_customers) over (order by order_month), 0) * 100), 1
    ) as mom_customer_growth_pct
from monthly_aggregates
order by order_month;
