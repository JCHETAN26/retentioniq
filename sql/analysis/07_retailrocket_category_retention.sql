-- 07_retailrocket_category_retention.sql
--
-- Business Question:
-- For visitors whose initial discovery session centered on each of the top-10 most popular product categories,
-- what is their 30-day and 60-day return rate? Does initial exploration in specific catalog categories
-- yield systematically stickier visitor retention?
--
-- Target Warehouse Marts / Intermediate Models:
--   - intermediate.int_retailrocket__sessionized_events: event timestamps, session_id, visitor_id, category_id
--   - staging.stg_retailrocket__events / category metadata
--
-- Edge Cases Handled:
--   - Right-censoring: visitors whose first session occurred within 30 or 60 days of the dataset end
--     (max event timestamp) are excluded from the respective retention calculation to prevent artificial deflation.
--   - First category attribution: determined by the earliest event timestamp for each returning visitor.
--   - NULL categories: filtered out (`category_id IS NOT NULL`).
--
-- Index & Performance Awareness:
--   - Aggregates by visitor_id in a CTE before joining category summaries.

with dataset_horizon as (
    select max(event_ts) as max_ts
    from marts.fact_events
),

visitor_first_events as (
    select distinct on (visitor_id)
        visitor_id,
        event_ts as first_event_ts,
        category_id
    from marts.fact_events
    where category_id is not null
    order by visitor_id, event_ts asc
),

visitor_subsequent_activity as (
    select
        f.visitor_id,
        f.category_id,
        f.first_event_ts,
        -- Has the visitor returned within 30 days?
        case
            when exists (
                select 1
                from marts.fact_events e
                where e.visitor_id = f.visitor_id
                  and e.event_ts > f.first_event_ts
                  and e.event_ts <= f.first_event_ts + interval '30 days'
            ) then 1
            else 0
        end as returned_30d,
        -- Has the visitor returned within 60 days?
        case
            when exists (
                select 1
                from marts.fact_events e
                where e.visitor_id = f.visitor_id
                  and e.event_ts > f.first_event_ts
                  and e.event_ts <= f.first_event_ts + interval '60 days'
            ) then 1
            else 0
        end as returned_60d,
        -- Observability flags for censoring
        case when f.first_event_ts + interval '30 days' <= h.max_ts then 1 else 0 end as observable_30d,
        case when f.first_event_ts + interval '60 days' <= h.max_ts then 1 else 0 end as observable_60d
    from visitor_first_events f
    cross join dataset_horizon h
),

top_categories as (
    select
        category_id,
        count(*) as total_first_visitors
    from visitor_first_events
    group by category_id
    order by total_first_visitors desc
    limit 10
)

select
    c.category_id,
    c.total_first_visitors,
    count(*) filter (where v.observable_30d = 1) as observable_visitors_30d,
    sum(v.returned_30d) filter (where v.observable_30d = 1) as returned_visitors_30d,
    round(
        (sum(v.returned_30d) filter (where v.observable_30d = 1)::numeric
        / nullif(count(*) filter (where v.observable_30d = 1), 0) * 100), 2
    ) as return_rate_30d_pct,
    count(*) filter (where v.observable_60d = 1) as observable_visitors_60d,
    sum(v.returned_60d) filter (where v.observable_60d = 1) as returned_visitors_60d,
    round(
        (sum(v.returned_60d) filter (where v.observable_60d = 1)::numeric
        / nullif(count(*) filter (where v.observable_60d = 1), 0) * 100), 2
    ) as return_rate_60d_pct
from top_categories c
inner join visitor_subsequent_activity v
    on c.category_id = v.category_id
group by c.category_id, c.total_first_visitors
order by c.total_first_visitors desc;
