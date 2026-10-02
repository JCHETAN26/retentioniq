-- Daily session-level funnel: view -> cart -> transaction.
select
    session_start_ts::date as activity_date,
    count(*) as sessions,
    count(*) filter (where n_views > 0) as sessions_with_view,
    count(*) filter (where n_carts > 0) as sessions_with_cart,
    count(*) filter (where n_transactions > 0) as sessions_with_transaction,
    count(distinct visitor_id) as visitors
from {{ ref('int_retailrocket__sessions') }}
group by 1
