-- RetailRocket visitor x day activity (dataset A). The basis for point-in-time features.
select
    visitor_id,
    event_date as activity_date,
    count(*) as n_events,
    count(*) filter (where event_type = 'view') as n_views,
    count(*) filter (where event_type = 'addtocart') as n_carts,
    count(*) filter (where event_type = 'transaction') as n_transactions,
    count(distinct session_id) as n_sessions,
    count(distinct item_id) as n_items,
    count(distinct category_id) as n_categories
from {{ ref('fact_events') }}
group by visitor_id, event_date
