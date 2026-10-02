select distinct on (stock_code)
    stock_code,
    description,
    is_non_product
from {{ ref('stg_online_retail__transactions') }}
order by stock_code, (description is null), description
