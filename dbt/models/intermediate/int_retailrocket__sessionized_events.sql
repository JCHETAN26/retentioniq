-- Assigns sessions: a new session starts after a gap of more than `session_gap_minutes`.
with ordered as (
    select
        e.*,
        lag(e.event_ts) over (partition by e.visitor_id order by e.event_ts, e.event_id) as prev_ts
    from {{ ref('stg_retailrocket__events') }} e
),

flagged as (
    select
        *,
        case
            when prev_ts is null
              or event_ts - prev_ts > interval '{{ var("session_gap_minutes") }} minutes'
            then 1 else 0
        end as is_session_start
    from ordered
)

select
    f.event_id,
    f.event_ts,
    f.event_ts::date as event_date,
    f.visitor_id,
    f.event_type,
    f.item_id,
    c.category_id,
    f.transaction_id,
    f.visitor_id::text || '-' || sum(f.is_session_start) over (
        partition by f.visitor_id order by f.event_ts, f.event_id rows unbounded preceding
    )::text as session_id
from flagged f
left join {{ ref('int_retailrocket__item_category') }} c using (item_id)
