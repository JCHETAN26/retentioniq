-- `exposure` is post-treatment and is kept only for descriptive analysis; it must never be a
-- model feature (PROJECT_SPEC.md §4.4). `row_id` is the 1-based row in the source file.
select
    _source_row as row_id,
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
    treatment,
    visit,
    conversion,
    exposure
from {{ source('raw', 'criteo_uplift') }}
