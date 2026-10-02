select
    categoryid as category_id,
    parentid as parent_category_id
from {{ source('raw', 'retailrocket_category_tree') }}
