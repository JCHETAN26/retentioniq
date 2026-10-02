{{ config(materialized="view") }}
-- A view over staging: the 14M-row table would only duplicate raw.criteo_uplift on disk.
select * from {{ ref('stg_criteo__uplift') }}
