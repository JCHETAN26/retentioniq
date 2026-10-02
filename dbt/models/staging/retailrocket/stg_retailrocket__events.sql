-- Events with UTC timestamps; exact duplicate rows in the source are collapsed.
with source as (
    select distinct
        (to_timestamp(timestamp / 1000.0) at time zone 'UTC') as event_ts,
        visitorid as visitor_id,
        event as event_type,
        itemid as item_id,
        transactionid as transaction_id
    from {{ source('raw', 'retailrocket_events') }}
)

select
    row_number() over (
        order by event_ts, visitor_id, item_id, event_type, transaction_id
    ) as event_id,
    *
from source
