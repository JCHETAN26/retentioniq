-- Customer x activity month, with month number relative to the first-purchase cohort.
with activity as (
    select
        customer_id,
        date_trunc('month', order_date)::date as activity_month,
        count(*) as orders,
        sum(order_value) as revenue
    from {{ ref('int_online_retail__orders') }}
    group by 1, 2
)

select
    a.customer_id,
    c.cohort_month,
    a.activity_month,
    (extract(year from age(a.activity_month, c.cohort_month)) * 12
        + extract(month from age(a.activity_month, c.cohort_month)))::int as month_number,
    a.orders,
    a.revenue
from activity a
join {{ ref('dim_customer') }} c using (customer_id)
