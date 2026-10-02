{# Use the configured schema verbatim (staging / intermediate / marts) instead of public_<schema>. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name if custom_schema_name is not none else target.schema }}
{%- endmacro %}
