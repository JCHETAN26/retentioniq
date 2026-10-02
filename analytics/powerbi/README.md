# Power BI report

This is the analyst-facing report. It connects to the **warehouse**, never to raw CSVs:
- `marts.*`: dbt star schema
- `reporting.*`: model outputs written by `retentioniq publish`

Power BI Desktop runs only on Windows, so the `.pbix` is built there. Everything it needs is
in the repository: the views, the measures in [`measures.dax`](measures.dax), and the page spec below.

## 1. Make the data reachable

The database runs in Postgres (Docker Compose, either locally or in the codespace).

1. Generate the marts and the outputs:
   ```bash
   make up ingest dbt evaluate
   uv run retentioniq publish
   ```
2. **If it runs in a codespace**, forward the port on the Windows machine:
   ```powershell
   gh codespace ports forward 5432:5432 -c <codespace-name>
   ```
   Postgres is then available at `localhost:5432`. Leave the command running while you refresh.

## 2. Connect

In Power BI Desktop, go to **Get Data → PostgreSQL database** and enter:
- Server: `localhost:5432`
- Database: `retentioniq`
- Mode: Import
- Credentials: the `.env` values (default `retentioniq` / `retentioniq`)

Select these tables:

| Schema | Tables |
|---|---|
| marts | `dim_date`, `dim_customer`, `dim_product`, `fact_orders`, `fact_transactions`, `mart_customer_rfm`, `mart_customer_retention`, `mart_engagement_funnel`, `mart_customer_daily_activity` |
| reporting | `churn_risk_scores`, `churn_shap_global`, `churn_model_comparison`, `customer_segments`, `campaign_simulation`, `survival_km` |

`fact_events` is a 2.7M-row view. Leave it out and use the daily-activity mart instead.

## 3. Model (star schema)

Mark `marts_dim_date` as the **date table**, using `date_day` as the date column.

| From (many) | To (one) | Notes |
|---|---|---|
| `fact_orders[order_date]` | `dim_date[date_day]` | active |
| `fact_orders[customer_id]` | `dim_customer[customer_id]` | |
| `fact_transactions[invoice_date]` | `dim_date[date_day]` | |
| `fact_transactions[stock_code]` | `dim_product[stock_code]` | |
| `fact_transactions[customer_id]` | `dim_customer[customer_id]` | some rows have no customer, by design |
| `mart_customer_rfm[customer_id]` | `dim_customer[customer_id]` | 1:1, both directions |
| `mart_engagement_funnel[activity_date]` | `dim_date[date_day]` | RetailRocket, 2015 |
| `mart_customer_daily_activity[activity_date]` | `dim_date[date_day]` | RetailRocket, 2015 |

The model has no relationship between RetailRocket tables and Online Retail tables. The
datasets are independent (PROJECT_SPEC §2), and the report must not imply a join between them.

Hide technical columns such as `_run_id` and the `line_id` hash. Set the formats: currency £
for value measures, % for rates.

## 4. Pages

| Page | Visuals (measure / field) |
|---|---|
| **Executive Overview** | KPI cards: Revenue, Customers, AOV, Repeat Purchase Rate, M1 Retention, Revenue YoY %. Line: Revenue by `dim_date[year_month]`. Column: New Customers by month. Slicers: year, country |
| **Customer Segments** | Bar: Segment Revenue Share vs Segment Customer Share by `rfm_segment`. Table: segment, Customers, AOV, Orders per Customer, Repeat Purchase Rate. Scatter: `customer_segments[pc1]` × `pc2`, colored by `cluster_label` (≤ 6 clusters) |
| **Cohort Retention** | Matrix: rows `cohort_month`, columns `month_number`, values Retention Rate, with a single-hue color scale (conditional formatting, light → dark blue). Line: Retention Rate by `month_number` for the selected cohorts |
| **Customer Value** | Card: Top 10% Customer Revenue Share. Histogram: `dim_customer[lifetime_value]` (bins). Pareto: cumulative revenue % by customer rank. Table: top customers |
| **Engagement Funnel** | Funnel: Sessions → Sessions with Cart → Sessions with Purchase. Line: View to Cart Rate and Cart to Purchase Rate by date (two lines, one axis, both %) |
| **Risk Analytics** | Cards: Scored Visitors, High-risk Share, Precision in High Tier, Calibration Gap. Bar: `churn_shap_global` mean_abs_shap by feature (top 10). Column: visitors by `risk_tier`. Table: highest-risk visitors with recency, sessions and p_inactive. Small table: `churn_model_comparison` (model, pr_auc, roc_auc, brier) |

Design rules, applied the same way as in the web app:
- one value axis per chart (never dual axes)
- a single hue for magnitude
- risk tiers always carry a text label, not color alone
- every page footer states the data source and the cutoff date

## 5. Refresh

To update the data, re-run `make evaluate && uv run retentioniq publish`, then press
**Refresh** in Power BI. Never type numbers into the report by hand; every value comes from the warehouse.
