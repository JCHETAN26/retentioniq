-- Qualifying purchase lines, PROJECT_SPEC.md §4.3.
select *
from {{ ref('stg_online_retail__transactions') }}
where has_customer
  and not is_cancellation
  and not is_non_product
  and quantity > 0
  and price > 0
