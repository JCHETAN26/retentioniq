select
    o.*,
    row_number() over (partition by o.customer_id order by o.order_ts, o.invoice) as customer_order_number,
    o.order_date - lag(o.order_date) over (
        partition by o.customer_id order by o.order_ts, o.invoice
    ) as days_since_previous_order
from {{ ref('int_online_retail__orders') }} o
