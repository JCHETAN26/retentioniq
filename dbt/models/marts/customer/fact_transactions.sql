-- All invoice lines (including cancellations and lines without a customer) with qualification
-- flags, so revenue reconciliation and cancellation analysis are possible from one fact table.
select
    t.line_id,
    t.invoice,
    t.invoice_ts,
    t.invoice_date,
    t.customer_id,
    t.stock_code,
    t.country,
    t.quantity,
    t.price,
    t.line_amount,
    t.is_cancellation,
    t.is_non_product,
    t.has_customer,
    (t.has_customer and not t.is_cancellation and not t.is_non_product
        and t.quantity > 0 and t.price > 0) as is_qualifying
from {{ ref('stg_online_retail__transactions') }} t
