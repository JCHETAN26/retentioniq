{{ config(materialized="view") }}
select * from {{ ref('int_retailrocket__sessionized_events') }}
