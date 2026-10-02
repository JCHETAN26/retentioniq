-- Per-arm outcome counts and rates. Inference (CIs, tests) lives in Python
-- (retentioniq.analytics.experiments) so it is computed exactly once.
select
    treatment,
    count(*) as n,
    sum(visit) as visits,
    sum(conversion) as conversions,
    sum(exposure) as exposures,
    avg(visit::numeric) as visit_rate,
    avg(conversion::numeric) as conversion_rate
from {{ ref('fact_experiment_population') }}
group by treatment
