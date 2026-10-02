# RetentionIQ

A platform for customer intelligence, retention, and treatment optimization. It covers
segmentation, cohort retention, inactivity prediction, survival analysis, experiment
analysis, and uplift-based targeting, built on three public datasets that are kept independent.

**Start with [`PROJECT_SPEC.md`](PROJECT_SPEC.md).** It fixes every definition, metric,
split, and promotion gate before any modeling begins.

| Case study | Dataset | Purpose |
|---|---|---|
| Customer Intelligence | RetailRocket | behavior → engagement → inactivity / survival |
| Customer Intelligence | UCI Online Retail II | RFM → revenue → cohorts → value |
| Treatment Intelligence | Criteo Uplift | experiment → treatment effect → uplift → targeting |

## Layout

```
src/retentioniq/     Python package: ingestion, quality, features, training, models, analytics
apps/api             FastAPI serving (Phase 22)
apps/web             Next.js app (Phase 24)
dbt/                 staging → intermediate → marts (Phase 5)
analytics/powerbi    Power BI report (Phase 10)
infrastructure/      Docker images, Postgres init, AWS IaC
monitoring/          Prometheus / Grafana / Evidently config
data/raw/            Immutable source files (git-ignored)
data/samples/        Small committed samples for tests
artifacts/           Output of `make evaluate`, the only source of reported numbers
```

## Quickstart

```bash
cp .env.example .env
make install      # uv sync + pre-commit hooks
make check        # ruff + mypy + pytest
make up           # Postgres + MLflow (http://localhost:5000)
```

Each phase installs its heavier dependencies as an extra, for example
`uv sync --extra ml --extra survival`.
