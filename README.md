# RetentionIQ

A retention decision platform built on three public datasets. Starting from raw downloads, it
lets a team:

> understand customers → discover segments → measure retention → predict inactivity →
> understand when and why customers disengage → analyse experiments → find who responds
> *because of* a treatment → compare targeting strategies.

Every sophisticated component sits on top of a simpler baseline and is measured against it:
- RFM comes before clustering.
- Logistic regression comes before LightGBM and PyTorch.
- Propensity targeting comes before uplift targeting.

**Start with [`PROJECT_SPEC.md`](PROJECT_SPEC.md).** It fixes every definition, metric, split
and promotion gate before any modeling, and keeps a change log of every amendment.
**Results are in [`docs/RESULTS.md`](docs/RESULTS.md),** which is generated from
`artifacts/`. This README contains no hand-typed performance numbers.

## The three case studies (never joined)

| Case study | Dataset | Question |
|---|---|---|
| Customer intelligence | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | RFM, revenue concentration, cohorts, segments |
| Customer intelligence | [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | Behavior → engagement → 30-day inactivity, time to inactivity |
| Treatment intelligence | [Criteo Uplift v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | Randomized experiment → ATE → uplift → campaign targeting |

## Architecture

```
download (checksummed, read-only)  →  ingest (Pandera contracts, quarantine, idempotent ledger)
   →  Postgres raw  →  dbt staging → intermediate → marts (with tests)
   →  retentioniq evaluate:
        eda         descriptive stats + hypotheses         → docs/EDA.md, eda_summary.json
        customers   RFM · K-Means/GMM/HDBSCAN · cohorts    → segmentation_metrics.json, cohort_metrics.csv
        churn       M0–M5 ladder · ablation · sensitivity  → churn_model_comparison.csv, feature_ablation.csv, …
        explain     SHAP · FP/FN analysis                  → shap_global.csv, error_analysis.json
        visitors    visitor segments · category cohorts    → visitor_segmentation.json, visitor_category_cohorts.csv
        survival    Kaplan–Meier · Cox PH + PH tests       → survival_analysis.csv, survival_summary.json
        uplift      ATE · S/T/X-learners · causal forest · campaign sim    → experiment_results.json, uplift_results.csv, campaign_simulation.csv
   →  retentioniq promote (gated champion/challenger → MLflow registry)
   →  FastAPI (/v1/churn, /v1/uplift, /metrics)  →  Next.js app   ·   Power BI on marts + reporting
   →  monitoring: Prometheus/Grafana (system) · drift report (data/model) · CloudWatch (AWS)
```

| Path | What |
|---|---|
| `src/retentioniq/` | Python package: `ingestion`, `quality`, `features`, `models`, `training`, `analytics`, `serving`, `monitoring` |
| `dbt/` | Warehouse models and tests |
| `apps/api/` | API container (code in `src/retentioniq/serving`) |
| `apps/web/` | Next.js app: Customer Intelligence, Risk, Model Lab, Experiment Lab, Uplift Lab, Campaign Simulator |
| `analytics/powerbi/` | Power BI build guide and DAX measures |
| `pipelines/airflow/` | Weekly training DAG |
| `monitoring/` | Prometheus alerts, Grafana dashboard |
| `infrastructure/` | Postgres init, MLflow image, AWS Terraform (ECS Fargate) |
| `evaluation/` | Serving load test, results-report generator |

## Reproduce

Requires Docker, [uv](https://docs.astral.sh/uv/) and about 15 GB of disk. The repo also opens
directly in GitHub Codespaces (`.devcontainer/`).

```bash
cp .env.example .env
make install            # Python deps + pre-commit
make reproduce          # Postgres+MLflow → download → ingest → dbt build → evaluate → report
```

The same pipeline step by step:

```bash
make up                 # Postgres :5432, MLflow :5000
make data               # download the three datasets (idempotent, checksummed)
make ingest             # validate + load into Postgres (re-runs are no-ops)
make dbt                # build + test the warehouse
make evaluate           # all artifacts in artifacts/ (~1 h on 4 cores)
make report             # docs/RESULTS.md
uv run retentioniq promote      # promotion gates (PROJECT_SPEC §9.1)
uv run retentioniq monitor      # drift / calibration report
uv run retentioniq publish      # model outputs → Postgres reporting.* for Power BI
```

Serve and explore:

```bash
uv run uvicorn retentioniq.serving.app:app --port 8000      # or: docker compose --profile serving up -d api
cd apps/web && npm install && RIQ_API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
make benchmark                                               # serving_benchmark.json
docker compose --profile monitoring up -d                    # Prometheus :9090, Grafana :3001
```

## Quality gates

- `make check`: ruff, mypy (strict) and unit tests.
- `make test-integration`: ingestion idempotency against real Postgres.
- `make dbt`: warehouse tests.
- CI runs all of these on every push (`.github/workflows/ci.yml`).
