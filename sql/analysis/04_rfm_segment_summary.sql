-- 04_rfm_segment_summary.sql
--
-- Business Question:
-- How are customers distributed across behavioral RFM segments (Champions, Loyal, At Risk, Hibernating, etc.)?
-- What share of revenue does each segment generate, what is their repeat purchase rate, and what are
-- their average recency and order frequency characteristics?
--
-- Target Warehouse Marts:
--   - marts.mart_customer_rfm: customer-level table with rfm_segment, recency_days, frequency, monetary
--
-- Edge Cases Handled:
--   - Single-order customers (frequency = 1) vs repeat purchasers (frequency >= 2):
--     repeat purchase rate is calculated as COUNT(*) FILTER (WHERE frequency >= 2) / COUNT(*).
--   - Overall revenue share: window function SUM(monetary) OVER () computes percentage of entire portfolio.
--   - Ordering segments logically by economic value contribution (total revenue share descending).
--
-- Index & Performance Awareness:
--   - Grouping by rfm_segment on mart_customer_rfm.
--   - Avoids joins by reading directly from the pre-scored RFM mart.

with segment_stats as (
    select
        rfm_segment,
        count(*) as customer_count,
        sum(monetary) as total_segment_revenue,
        avg(monetary) as avg_monetary_value,
        avg(frequency) as avg_frequency,
        avg(recency_days) as avg_recency_days,
        count(*) filter (where frequency >= 2) as repeat_customers
    from marts.mart_customer_rfm
    group by rfm_segment
),

total_summary as (
    select
        sum(customer_count) as total_customers,
        sum(total_segment_revenue) as total_revenue
    from segment_stats
)

select
    s.rfm_segment,
    s.customer_count,
    round((s.customer_count::numeric / t.total_customers * 100), 1) as customer_share_pct,
    round(s.total_segment_revenue::numeric, 2) as total_revenue,
    round((s.total_segment_revenue / t.total_revenue * 100)::numeric, 1) as revenue_share_pct,
    round(s.avg_monetary_value::numeric, 2) as avg_monetary_value,
    round(s.avg_frequency::numeric, 2) as avg_frequency,
    round(s.avg_recency_days::numeric, 1) as avg_recency_days,
    round((s.repeat_customers::numeric / s.customer_count * 100), 1) as repeat_purchase_rate_pct
from segment_stats s
cross join total_summary t
order by s.total_segment_revenue desc;
