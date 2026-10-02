-- The only columns uplift models may use: pre-treatment features, treatment, outcomes.
-- `exposure` is deliberately excluded (post-treatment; PROJECT_SPEC.md §4.4).
{{ config(materialized='view') }}
select
    row_id,
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
    treatment,
    visit,
    conversion
from {{ ref('fact_experiment_population') }}
