-- One row per invoice line, with the flags needed to apply PROJECT_SPEC.md §4.3.
-- The two source sheets overlap (Dec 2010); rows from the older sheet that fall inside the
-- newer sheet's date range are dropped so each line is counted once.
with source as (
    select * from {{ source('raw', 'online_retail_transactions') }}
),

newer_sheet_start as (
    select min(invoice_date) as start_ts
    from source
    where source_sheet = 'Year 2010-2011'
),

deduped as (
    select s.*
    from source s
    cross join newer_sheet_start n
    where not (s.source_sheet = 'Year 2009-2010' and s.invoice_date >= n.start_ts)
)

select
    md5(_source_file || ':' || _source_row::text) as line_id,
    invoice,
    stock_code,
    nullif(trim(description), '') as description,
    quantity,
    invoice_date as invoice_ts,
    date_trunc('day', invoice_date)::date as invoice_date,
    price,
    customer_id,
    country,
    quantity * price as line_amount,
    invoice like 'C%' as is_cancellation,
    (
        upper(stock_code) in (
            {%- for code in var('online_retail_non_product_codes') %}
            '{{ code }}'{{ "," if not loop.last }}
            {%- endfor %}
        )
        or upper(stock_code) like 'ADJUST%'
        or upper(stock_code) like 'TEST%'
        or upper(stock_code) like 'GIFT%'
    ) as is_non_product,
    customer_id is not null as has_customer
from deduped
