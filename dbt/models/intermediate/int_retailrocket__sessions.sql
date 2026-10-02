select
    session_id,
    visitor_id,
    min(event_ts) as session_start_ts,
    max(event_ts) as session_end_ts,
    count(*) as n_events,
    count(*) filter (where event_type = 'view') as n_views,
    count(*) filter (where event_type = 'addtocart') as n_carts,
    count(*) filter (where event_type = 'transaction') as n_transactions,
    count(distinct item_id) as n_items,
    count(distinct category_id) as n_categories
from {{ ref('int_retailrocket__sessionized_events') }}
group by session_id, visitor_id
