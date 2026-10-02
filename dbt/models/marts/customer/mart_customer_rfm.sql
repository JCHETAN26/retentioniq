-- RFM per PROJECT_SPEC.md §4.3: quintile scores (5 = best) and rule-based segments.
with base as (
    select
        customer_id,
        (select max(order_date) from {{ ref('int_online_retail__orders') }}) + 1 as snapshot_date,
        max(order_date) as last_order_date,
        count(*) as frequency,
        sum(order_value) as monetary
    from {{ ref('int_online_retail__orders') }}
    group by customer_id
),

scored as (
    select
        *,
        snapshot_date - last_order_date as recency_days,
        monetary / frequency as avg_order_value,
        ntile(5) over (order by snapshot_date - last_order_date desc, customer_id) as r_score,
        ntile(5) over (order by frequency, customer_id) as f_score,
        ntile(5) over (order by monetary, customer_id) as m_score
    from base
),

segmented as (
    select
        *,
        (f_score + m_score) / 2.0 as fm_score,
        r_score::text || f_score::text || m_score::text as rfm_code
    from scored
)

select
    *,
    case
        when r_score >= 4 and fm_score >= 4.5 then 'Champions'
        when r_score >= 3 and fm_score >= 3.5 then 'Loyal'
        when r_score >= 4 and fm_score >= 2 and fm_score < 3.5 then 'Potential Loyalists'
        when r_score = 5 and fm_score < 2 then 'New Customers'
        when r_score <= 2 and fm_score >= 3 then 'At Risk'
        when r_score = 2 and fm_score < 3 then 'Hibernating'
        when r_score = 1 and fm_score < 3 then 'Lost'
        else 'Needs Attention'
    end as rfm_segment
from segmented
