-- Cohort retention matrix in long form: M0..Mn per first-purchase month.
-- Only month numbers fully observable for the cohort are emitted (no right-censored cells).
with sizes as (
    select cohort_month, count(*) as cohort_size
    from {{ ref('dim_customer') }}
    group by cohort_month
),

last_month as (
    select date_trunc('month', max(order_date))::date as m
    from {{ ref('int_online_retail__orders') }}
),

grid as (
    select s.cohort_month, s.cohort_size, g.month_number
    from sizes s
    cross join last_month l
    cross join lateral generate_series(
        0,
        (extract(year from age(l.m, s.cohort_month)) * 12
            + extract(month from age(l.m, s.cohort_month)))::int
    ) as g(month_number)
),

active as (
    select cohort_month, month_number,
           count(distinct customer_id) as active_customers,
           sum(revenue) as revenue
    from {{ ref('mart_customer_cohorts') }}
    group by 1, 2
)

select
    g.cohort_month,
    g.month_number,
    g.cohort_size,
    coalesce(a.active_customers, 0) as active_customers,
    coalesce(a.active_customers, 0)::numeric / g.cohort_size as retention_rate,
    coalesce(a.revenue, 0) as revenue
from grid g
left join active a using (cohort_month, month_number)
