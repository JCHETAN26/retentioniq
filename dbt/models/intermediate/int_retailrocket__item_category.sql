-- Latest known category per item (the `categoryid` property; changes over time are rare).
select distinct on (item_id)
    item_id,
    value::bigint as category_id
from {{ ref('stg_retailrocket__item_properties') }}
where property = 'categoryid' and value ~ '^\d+$'
order by item_id, valid_from desc
