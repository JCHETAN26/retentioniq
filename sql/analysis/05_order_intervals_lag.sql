-- 05_order_intervals_lag.sql
--
-- Business Question:
-- What is the typical repurchase cadence (days between consecutive orders), and how does the inter-order
-- latency distribution vary between low-frequency (2-3 orders), mid-frequency (4-6 orders), and
-- high-frequency (7+ orders) customers?
--
-- Target Warehouse Marts:
--   - marts.fact_orders: order-level grain with customer_id, order_date, customer_order_number
--   - marts.dim_customer: customer-level total lifetime_orders for frequency tiering
--
-- Edge Cases Handled:
--   - A customer's initial order has no predecessor: LAG(order_date) is NULL and is filtered out.
--   - Multiple orders placed on the exact same date yield interval = 0 days.
--   - Outlier repurchases: percentiles (p25, p50, p75, p90) provide robust non-parametric summary
--     via PostgreSQL's PERCENTILE_CONT(...) WITHIN GROUP.
--
-- Index & Performance Awareness:
--   - Window function partitions by customer_id and orders by order_date.
--   - Indexed on (customer_id, order_date) for efficient stream sorting.

with order_intervals as (
    select
        o.customer_id,
        o.order_date,
        c.lifetime_orders,
        case
            when c.lifetime_orders between 2 and 3 then '2-3 orders (Occasional)'
            when c.lifetime_orders between 4 and 6 then '4-6 orders (Regular)'
            when c.lifetime_orders >= 7 then '7+ orders (Frequent)'
            else '1 order (Single-buyer)'
        end as frequency_tier,
        o.order_date - lag(o.order_date) over (
            partition by o.customer_id order by o.order_date, o.invoice
        ) as days_since_prev_order
    from marts.fact_orders o
    inner join marts.dim_customer c
        on o.customer_id = c.customer_id
),

valid_intervals as (
    select
        customer_id,
        frequency_tier,
        days_since_prev_order
    from order_intervals
    where days_since_prev_order is not null
      and frequency_tier != '1 order (Single-buyer)'
)

select
    frequency_tier,
    count(*) as repeat_order_events,
    count(distinct customer_id) as customer_count,
    round(avg(days_since_prev_order)::numeric, 1) as mean_days_between_orders,
    round(
        percentile_cont(0.25) within group (order by days_since_prev_order)::numeric, 1
    ) as p25_days,
    round(
        percentile_cont(0.50) within group (order by days_since_prev_order)::numeric, 1
    ) as p50_median_days,
    round(
        percentile_cont(0.75) within group (order by days_since_prev_order)::numeric, 1
    ) as p75_days,
    round(
        percentile_cont(0.90) within group (order by days_since_prev_order)::numeric, 1
    ) as p90_days
from valid_intervals
group by frequency_tier
order by frequency_tier;
