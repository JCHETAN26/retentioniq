-- 02_cohort_retention_matrix.sql
--
-- Business Question:
-- How does customer retention decay over a 12-month lifecycle across monthly acquisition cohorts?
-- Produce a classic triangular cohort retention matrix showing the percentage of the original
-- acquisition cohort that placed at least one repeat order in month M1 through M12.
--
-- Target Warehouse Marts:
--   - marts.dim_customer: defines the cohort_month for every qualifying customer
--   - marts.fact_orders: order activity across subsequent calendar months
--
-- Edge Cases Handled:
--   - Month 0 is always 100% by definition of cohort inception; repeat retention starts at Month 1.
--   - Right-censoring: recent cohorts have not yet reached month M12. Unobserved future months
--     return NULL or are excluded rather than 0% retention.
--   - Calendar month difference: computed precisely via integer month arithmetic:
--     (EXTRACT(year FROM order_month) - EXTRACT(year FROM cohort_month)) * 12
--     + (EXTRACT(month FROM order_month) - EXTRACT(month FROM cohort_month)).
--
-- Index & Performance Awareness:
--   - Filter order activity to within 12 months of cohort origin to limit scanning.
--   - Pivot uses standard PostgreSQL conditional aggregation (FILTER clauses).

with customer_cohorts as (
    select
        customer_id,
        cohort_month
    from marts.dim_customer
),

customer_monthly_activity as (
    select distinct
        o.customer_id,
        c.cohort_month,
        (
            (extract(year from o.order_date) - extract(year from c.cohort_month)) * 12
            + (extract(month from o.order_date) - extract(month from c.cohort_month))
        )::int as month_number
    from marts.fact_orders o
    inner join customer_cohorts c
        on o.customer_id = c.customer_id
    where o.order_date >= c.cohort_month
),

cohort_sizes as (
    select
        cohort_month,
        count(distinct customer_id) as cohort_size
    from customer_cohorts
    group by cohort_month
),

retention_counts as (
    select
        cohort_month,
        month_number,
        count(distinct customer_id) as active_customers
    from customer_monthly_activity
    where month_number between 0 and 12
    group by cohort_month, month_number
)

select
    s.cohort_month,
    s.cohort_size,
    round(100.0 * max(case when r.month_number = 1 then r.active_customers end) / s.cohort_size, 1) as m01_retention_pct,
    round(100.0 * max(case when r.month_number = 2 then r.active_customers end) / s.cohort_size, 1) as m02_retention_pct,
    round(100.0 * max(case when r.month_number = 3 then r.active_customers end) / s.cohort_size, 1) as m03_retention_pct,
    round(100.0 * max(case when r.month_number = 4 then r.active_customers end) / s.cohort_size, 1) as m04_retention_pct,
    round(100.0 * max(case when r.month_number = 5 then r.active_customers end) / s.cohort_size, 1) as m05_retention_pct,
    round(100.0 * max(case when r.month_number = 6 then r.active_customers end) / s.cohort_size, 1) as m06_retention_pct,
    round(100.0 * max(case when r.month_number = 7 then r.active_customers end) / s.cohort_size, 1) as m07_retention_pct,
    round(100.0 * max(case when r.month_number = 8 then r.active_customers end) / s.cohort_size, 1) as m08_retention_pct,
    round(100.0 * max(case when r.month_number = 9 then r.active_customers end) / s.cohort_size, 1) as m09_retention_pct,
    round(100.0 * max(case when r.month_number = 10 then r.active_customers end) / s.cohort_size, 1) as m10_retention_pct,
    round(100.0 * max(case when r.month_number = 11 then r.active_customers end) / s.cohort_size, 1) as m11_retention_pct,
    round(100.0 * max(case when r.month_number = 12 then r.active_customers end) / s.cohort_size, 1) as m12_retention_pct
from cohort_sizes s
left join retention_counts r
    on s.cohort_month = r.cohort_month
group by s.cohort_month, s.cohort_size
order by s.cohort_month;
