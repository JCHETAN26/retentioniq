-- 06_retailrocket_funnel_weekly.sql
--
-- Business Question:
-- How does session-level traffic flow through the e-commerce conversion funnel (view -> cart -> transaction)
-- on a weekly basis? What are the stage-by-stage drop-off rates and the overall visit-to-purchase
-- conversion rate across RetailRocket sessions?
--
-- Target Warehouse Marts:
--   - marts.mart_engagement_funnel: daily activity counts of sessions with views, carts, and transactions
--
-- Edge Cases Handled:
--   - Sessions with cart activity but 0 views: handled by counting independent stage activations.
--   - Week truncation: DATE_TRUNC('week', activity_date) standardizes reporting boundaries.
--   - Drop-off rates: computed as 100% minus the stage progression rate, guarded by NULLIF.
--
-- Index & Performance Awareness:
--   - Operates on daily mart_engagement_funnel (~140 rows), executing in milliseconds.

with weekly_funnel as (
    select
        date_trunc('week', activity_date)::date as activity_week,
        sum(visitors) as total_visitors,
        sum(sessions) as total_sessions,
        sum(sessions_with_view) as sessions_with_view,
        sum(sessions_with_cart) as sessions_with_cart,
        sum(sessions_with_transaction) as sessions_with_transaction
    from marts.mart_engagement_funnel
    group by 1
)

select
    activity_week,
    total_visitors,
    total_sessions,
    sessions_with_view,
    sessions_with_cart,
    sessions_with_transaction,
    -- Stage conversion percentages
    round(
        (sessions_with_cart::numeric / nullif(sessions_with_view, 0) * 100), 2
    ) as view_to_cart_pct,
    round(
        (sessions_with_transaction::numeric / nullif(sessions_with_cart, 0) * 100), 2
    ) as cart_to_purchase_pct,
    round(
        (sessions_with_transaction::numeric / nullif(total_sessions, 0) * 100), 2
    ) as overall_session_conversion_pct,
    -- Drop-off percentages
    round(
        (100.0 - (sessions_with_cart::numeric / nullif(sessions_with_view, 0) * 100)), 2
    ) as view_dropoff_pct,
    round(
        (100.0 - (sessions_with_transaction::numeric / nullif(sessions_with_cart, 0) * 100)), 2
    ) as cart_dropoff_pct
from weekly_funnel
order by activity_week;
