-- One row per qualifying invoice.
select
    invoice,
    customer_id,
    min(invoice_ts) as order_ts,
    min(invoice_date) as order_date,
    max(country) as country,
    count(*) as n_lines,
    sum(quantity) as n_units,
    count(distinct stock_code) as n_products,
    sum(line_amount) as order_value
from {{ ref('int_online_retail__qualifying_lines') }}
group by invoice, customer_id
