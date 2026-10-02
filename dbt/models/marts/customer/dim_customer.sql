-- Online Retail II customers with at least one qualifying order.
with orders as (select * from {{ ref('int_online_retail__orders') }}),

country as (
    select distinct on (customer_id) customer_id, country
    from orders
    group by customer_id, country
    order by customer_id, count(*) desc, country
)

select
    o.customer_id,
    c.country,
    min(o.order_date) as first_order_date,
    max(o.order_date) as last_order_date,
    date_trunc('month', min(o.order_date))::date as cohort_month,
    count(*) as lifetime_orders,
    sum(o.order_value) as lifetime_value,
    sum(o.order_value) / count(*) as avg_order_value
from orders o
join country c using (customer_id)
group by o.customer_id, c.country
