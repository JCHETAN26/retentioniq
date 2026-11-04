-- 08_criteo_ate_by_feature_decile.sql
--
-- Business Question:
-- How does the Average Treatment Effect (ATE) of ad exposure on conversion vary across deciles of
-- customer feature `f0` in the Criteo randomized trial? This SQL query detects treatment effect
-- heterogeneity directly within the warehouse, identifying which visitor segments exhibit high incremental
-- lift versus segments where the ad intervention produces zero or negative return.
--
-- Target Warehouse Marts:
--   - marts.mart_uplift_features: contains row_id, f0..f11, treatment, visit, conversion
--
-- Edge Cases Handled:
--   - Imbalanced assignment ratio (~85% treatment / ~15% control): rates are calculated within arms
--     independently before computing the difference (ATE = Treatment Rate - Control Rate).
--   - Relative lift computation: guarded with NULLIF(control_rate, 0).
--   - NTILE(10) ties: ordered deterministically with row_id.
--
-- Index & Performance Awareness:
--   - Mart view filters out post-treatment leakage (`exposure` is excluded).
--   - Analytical window function NTILE(10) operates over the full experiment population.

with deciled_population as (
    select
        row_id,
        f0,
        treatment,
        visit,
        conversion,
        ntile(10) over (order by f0 asc, row_id asc) as f0_decile
    from marts.mart_uplift_features
),

decile_metrics as (
    select
        f0_decile,
        min(f0) as f0_min,
        max(f0) as f0_max,
        count(*) as total_users,
        count(*) filter (where treatment = 1) as treated_users,
        count(*) filter (where treatment = 0) as control_users,
        -- Conversions
        sum(conversion) filter (where treatment = 1) as treated_conversions,
        sum(conversion) filter (where treatment = 0) as control_conversions,
        -- Visits
        sum(visit) filter (where treatment = 1) as treated_visits,
        sum(visit) filter (where treatment = 0) as control_visits
    from deciled_population
    group by f0_decile
)

select
    f0_decile,
    round(f0_min::numeric, 4) as f0_min,
    round(f0_max::numeric, 4) as f0_max,
    total_users,
    treated_users,
    control_users,
    -- Conversion rates per arm
    round((treated_conversions::numeric / nullif(treated_users, 0) * 100), 4) as treated_conversion_rate_pct,
    round((control_conversions::numeric / nullif(control_users, 0) * 100), 4) as control_conversion_rate_pct,
    -- Conversion ATE (Incremental Lift in percentage points)
    round(
        ((treated_conversions::numeric / nullif(treated_users, 0)) -
         (control_conversions::numeric / nullif(control_users, 0))) * 100, 4
    ) as conversion_ate_pp,
    -- Relative Lift in conversions
    round(
        (((treated_conversions::numeric / nullif(treated_users, 0)) -
          (control_conversions::numeric / nullif(control_users, 0)))
         / nullif(control_conversions::numeric / nullif(control_users, 0), 0) * 100), 2
    ) as conversion_relative_lift_pct,
    -- Visit ATE (Incremental visit rate in percentage points)
    round(
        ((treated_visits::numeric / nullif(treated_users, 0)) -
         (control_visits::numeric / nullif(control_users, 0))) * 100, 4
    ) as visit_ate_pp
from decile_metrics
order by f0_decile asc;
