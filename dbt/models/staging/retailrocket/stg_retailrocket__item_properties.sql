select
    itemid as item_id,
    property,
    value,
    (to_timestamp(timestamp / 1000.0) at time zone 'UTC') as valid_from
from {{ source('raw', 'retailrocket_item_properties') }}
