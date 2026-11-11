# RetentionIQ: Project Status & Handover Report

| | |
|---|---|
| **Report date** | 2026-11-10 |
| **Repository** | `github.com/JCHETAN26/retentioniq` (public), branch `main` |
| **HEAD** | `696613b` (2026-11-04), "Portfolio upgrade v0.6" |
| **Spec version** | `PROJECT_SPEC.md` v0.6 (`SPEC_VERSION = "0.6"` in `src/retentioniq/__init__.py`) |
| **Package version** | `0.1.0` (`pyproject.toml`; not bumped alongside the spec) |
| **History** | 19 commits, 2026-10-02 → 2026-11-04; one merged PR (#1, ingestion) |
| **CI** | Green on every one of the last 5 pushes (all 5 jobs) |

---

> **Update: v0.6 results produced (2026-11-10, evening).** The full pipeline ran in a 4-core
> GitHub Codespace on the real data. `make check`, `make reproduce`, `make promote` and
> `make benchmark` all exited 0. `docs/RESULTS.md`, `docs/FINDINGS.md` and `docs/figures/` are
> now generated from those artifacts (spec v0.6, git `20f499a`). Sections 1, 7 and 8 below are
> kept as written before the run, as a record. **Read §13 first: it supersedes them.**

## 1. TL;DR for the incoming engineer

RetentionIQ is a retention-decision platform. It runs three **separate** case studies on three public datasets: customer intelligence, inactivity/survival modeling, and causal uplift targeting. Around them is a production-style stack: Postgres + dbt warehouse, MLflow, a FastAPI serving layer, a Next.js app, Power BI, Airflow, Prometheus/Grafana, and AWS Terraform.

**Engineering state is good.** Lint, format, strict mypy and all 64 unit tests pass locally. All 5 CI jobs pass.

**The reported results are not in a consistent state. Fix this first.** The v0.6 commit rebuilt the reports with an empty `artifacts/` directory, so:

1. `docs/RESULTS.md` is now an empty stub. The last real report (spec v0.5, git `bcce5ce`) can be read with `git show 1a8b06b:docs/RESULTS.md`.
2. `docs/FINDINGS.md` and the README show **hard-coded fallback numbers** from `evaluation/render_report.py`, not pipeline outputs. Some of them contradict the last real run (details in §7).
3. The six result figures the README embeds do not exist, so they show as broken images on GitHub.
4. None of the features added in v0.6 (X-learner cross-fitting, day-7 landmark survival, CLV, parsimony rule, MLP grid) has published results in the repo.

**Bottom line:** the code for v0.6 is done and tested. The *evidence* for v0.6 has not been produced yet. You need one full `make reproduce` run on the real data, then a re-render of the reports.

---

## 2. What the project does

The three datasets are **never joined**. There is no shared entity, and PROJECT_SPEC §2 forbids claims that combine evidence across them.

| Spec ID | Dataset | Unit | Questions answered |
|---|---|---|---|
| A | RetailRocket clickstream (~2.76M events, ~1.4M visitors, May–Sep 2015) | `visitor_id` | Who goes inactive? When (survival)? Why (SHAP)? |
| B | UCI Online Retail II (Dec 2009 – Dec 2011) | `customer_id` | RFM segments, revenue concentration, cohorts, CLV |
| C | Criteo Uplift v2.1 (~14M rows, randomized, 85% treated) | row | ATE, CATE/uplift, targeting strategy, campaign economics |

**Guiding principle (spec §1):** every sophisticated component is benchmarked against a simpler baseline. If it does not win, that is reported.

**Headline claims (README):**
1. Targeting on uplift beats targeting on propensity, because it avoids spending on "always-takers".
2. A day-7 landmark beats first-session features for survival. First-session concordance is near random.
3. A threshold based on unit economics beats a 0.50 probability cutoff.

---

## 3. Repository map

```
src/retentioniq/          Python package (~5.8k LOC, 44 modules, mypy --strict)
  ingestion/              download (checksums), readers, load (idempotent, advisory-locked Postgres)
  quality/contracts.py    Pandera raw-data contracts + quarantine routing
  features/retailrocket.py  point-in-time snapshots, 30 features in 6 ablation families
  models/churn/           M0–M5 ladder + metrics (clustered bootstrap CIs)
  models/uplift/          S/T/X learners, transformed outcome, Qini/AUUC metrics
  training/               churn, survival (Cox, KM, day-7 landmark), uplift, explain (SHAP), promotion gates
  analytics/              customers (RFM, clustering), clv (BG/NBD + Gamma-Gamma), visitors, eda, experiments/inference
  serving/app.py          FastAPI API
  monitoring/drift.py     PSI, missingness, ECE
  cli.py                  Typer CLI (`retentioniq ...`)
  evaluate.py             stage runner for `make evaluate`
  reporting.py, tracking.py  Postgres publish for BI; MLflow helpers
dbt/                      staging → intermediate → marts (Postgres)
sql/analysis/             8 showcase analytical queries (+ README catalog)
notebooks/                01–03 showcase notebooks (run on data/sample)
evaluation/               render_report.py (RESULTS/FINDINGS/figures), benchmark_serving.py
apps/api/Dockerfile       slim serving image
apps/web/                 Next.js 16 / React 19 / Tailwind 4 / Recharts
analytics/powerbi/        Power BI guide + DAX measures (service and desktop routes)
pipelines/airflow/        weekly training DAG
monitoring/               Prometheus config + alert rules, Grafana dashboard
infrastructure/           postgres init.sql, MLflow image, AWS Terraform (ECS Fargate/ALB/ECR/S3/CloudWatch)
tests/unit (17 files), tests/integration (Postgres ingest)
PROJECT_SPEC.md           the experimental contract (definitions, splits, metrics, gates, change log)
KNOWLEDGE_BANK.md         long-form narrative + benchmarks (duplicated at docs/KNOWLEDGE_BANK.md)
```

**Empty placeholder packages** (only an empty `__init__.py`): `models/survival`, `models/segmentation`, `analytics/rfm`, `analytics/cohorts`, `transformations`. The real logic for these lives in `training/` and `analytics/`.

---

## 4. Architecture & data flow

```
download (checksums) → Pandera validation → raw.* / quarantine.*   [retentioniq download / ingest]
        → dbt: staging → intermediate (sessionization, orders) → marts   [make dbt]
        → retentioniq evaluate (stages: eda, customers, clv, churn, explain, visitors, survival, uplift)
              → artifacts/*.json|csv, data/models/*, MLflow runs
        → retentioniq promote (gates, MLflow registry)
        → evaluation/render_report.py → docs/RESULTS.md, docs/FINDINGS.md, docs/figures/*.png
        → serving: FastAPI reads data/models + artifacts → Next.js web app
        → retentioniq publish → Postgres reporting.* → Power BI (or export-powerbi → .xlsx)
        → Prometheus scrapes /metrics → Grafana; retentioniq monitor → drift report
```

**dbt marts:**
- Customer (B): `fact_orders`, `fact_transactions`, `dim_customer`, `dim_product`, `dim_date`, `mart_customer_rfm`, `mart_customer_cohorts`, `mart_customer_retention`.
- Engagement (A): `fact_events`, `mart_engagement_funnel`, `mart_customer_daily_activity`.
- Experiment (C): `fact_experiment_population`, `mart_experiment_results`, `mart_uplift_features`.
- Custom generic tests: `between`, `expression_is_true`.

**Postgres schemas:** `raw`, `quarantine`, `meta`, `staging`, `intermediate`, `marts`, `reporting`.

---

## 5. Components in detail

### 5.1 CLI (`retentioniq`)
`version`, `download`, `ingest`, `evaluate [--only <stage>]`, `promote`, `monitor`, `publish`, `export-powerbi`.

### 5.2 Serving API (`src/retentioniq/serving/app.py`)
| Route | Purpose |
|---|---|
| `GET /health`, `GET /v1/models` | liveness, loaded model metadata |
| `POST /v1/churn/predict?explain=` | single-record score + top TreeSHAP drivers |
| `POST /v1/churn/batch` | batch scoring |
| `GET /v1/churn/visitors[/{id}]` | scored visitors by tier; per-visitor detail incl. survival |
| `POST /v1/uplift/score` | CATE score |
| `GET /v1/artifacts/{name}` | serves evaluation artifacts to the web app |
| `GET /metrics` | Prometheus |

The fast path calls the LightGBM booster directly and computes SHAP natively inside the booster. The API needs `data/models/` and `artifacts/` to exist, and both are gitignored, so it cannot start from a fresh clone without `make evaluate`.

### 5.3 Web app (`apps/web`)
Pages: home, `risk`, `models`, `experiment`, `uplift`, `campaign`. Every number comes from the API (`RIQ_API_URL` / `NEXT_PUBLIC_RIQ_API_URL`, default `http://localhost:8000`). CI runs tsc, lint and build on it.

### 5.4 Orchestration, monitoring, infra
- **Airflow** (`retentioniq_training`, `@weekly`): download+ingest → dbt build → {evaluate churn → explain → promote, evaluate customers+survival}. Every task calls the same CLI.
- **docker-compose:** `postgres`, `mlflow` (default). Optional profiles: `serving` (api), `orchestration` (airflow), `monitoring` (prometheus :9090, grafana :3001).
- **AWS Terraform:** ECS Fargate behind an ALB, ECR, versioned S3 bucket for raw data, CloudWatch alarms to SNS. It uses **HTTP only** and an open `allowed_cidrs`; the README calls this demo-grade. Cost is about $1.50/day, so run `terraform destroy` when not demoing. As far as I can tell from the repo, it is not currently deployed (no state is committed, by design).
- **Power BI:** a service route (export to `artifacts/retentioniq_powerbi.xlsx` with 17 sheets) or a desktop route (direct Postgres). DAX is in `analytics/powerbi/measures*.dax`.

### 5.5 Promotion gates & SLOs (spec §9)
- **Gates:** PR-AUC non-inferiority (paired bootstrap CI > −0.005), Brier ≤ 1.05× champion, no segment loses > 0.03 PR-AUC, data validation passed, schema contract test passed.
- **Serving SLOs:** p50 ≤ 25 ms, p95 ≤ 100 ms, p99 ≤ 250 ms, ≥ 200 req/s per 1 vCPU task, 5xx < 0.1%.
- **Monitoring thresholds:** PSI warn > 0.1 / alert > 0.25, ECE warn > 0.05 / alert > 0.10, plus freshness, missingness and latency thresholds.

---

## 6. Verification performed for this report (2026-11-10, macOS, Python 3.11)

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | ✅ pass |
| Format | `ruff format --check .` | ✅ pass (82 files) |
| Types | `mypy` (strict) | ✅ no issues, 44 files (**requires** all extras installed, see note) |
| Unit tests | `pytest -m "not integration and not data"` | ✅ 64 passed, 2 deselected, 3 warnings |
| SQL showcase | `python scripts/sql_check.py` | ✅ 8/8 pass |
| CI (latest run, `696613b`) | GitHub Actions | ✅ lint-type-test, integration, dbt-parse, web, demo-and-sql |

**Not verified here:** integration tests (need `make up`), `dbt build` against real data, `make evaluate` on real data, the serving benchmark, the AWS deploy, and running the notebooks locally (CI's `demo-and-sql` job does run them).

**Environment note:** a plain `uv run` syncs only the base dependencies and *removes* the extras. In that state mypy reports 16 errors and 10 test modules fail to import (`sklearn` is missing). Always run `make install` (`uv sync --group dev --all-extras`) first, or use `uv run --no-sync`. During this review I re-ran `uv sync --group dev --all-extras` to put the environment back.

**Warnings worth handling eventually:** Starlette deprecates `httpx` in its TestClient. SHAP's LightGBM output format has changed (the code should tolerate both formats; the test passes). There is also an sklearn small-class warning in the X-learner cross-fit test fixture.

---

## 7. Results: what is actually established

### 7.1 Last real full run: spec v0.5, git `bcce5ce`
Source: `git show 1a8b06b:docs/RESULTS.md` and `KNOWLEDGE_BANK.md` §4.

**Inactivity ladder** (RetailRocket test snapshot 2015-08-19, N = 34,241, base rate 0.8735, 1,000-sample clustered bootstrap):

| Model | PR-AUC [95% CI] | ROC-AUC | Brier |
|---|---|---|---|
| M0 naive recency | 0.9300 [0.9268, 0.9334] | 0.702 | 0.104 |
| M1 logistic RFM | 0.9384 | 0.724 | 0.099 |
| M2 logistic full | 0.9496 | 0.769 | 0.095 |
| M3 random forest | 0.9498 | 0.771 | 0.094 |
| **M4 LightGBM (champion)** | **0.9521 [0.9493, 0.9550]** | 0.775 | 0.093 |
| M5 MLP | 0.9499 | 0.773 | 0.094 |

- **Ablation:** temporal features give the largest gain (+0.0080). Funnel features add nothing (CI spans 0).
- **Caveat:** the M2–M5 CIs overlap heavily. The v0.6 parsimony rule ("simplest model within the delta CI of the best") may pick a different champion once it is re-run.

**Survival (v0.5):** first-session Cox concordance on held-out visitors was **0.507**. Log-rank across first-session types gave p = 0.00057. Proportional-hazards violations were found on 3 covariates; the model was stratified on `first_session_weekend`.

**Criteo experiment:** N = 13,979,592, max |SMD| = 0.049 (balanced).
- Visit ATE +1.034 pp [+1.005, +1.063], or +27% relative.
- Conversion ATE +0.115 pp, or +59% relative.

**Uplift:** the strategy chosen on validation Qini was **`s_learner_lgbm`**, with test Qini 0.0921 vs 0.0866 for the response model. The causal forest scored 0.0924 on test but was not chosen, because selection happens on validation. The X-learner reached 0.0851, below the response model. At 1% reach the S-learner gives about 2.9× the incremental visits of the response model, but **not** more incremental conversions (1,050 vs 1,169, with overlapping CIs).

**Customer intelligence (B):** 5,852 customers, 36,594 orders. The top 10% of customers bring in 63.8% of revenue (Gini 0.74). Simple RFM rules (future-activity AUC 0.769) beat K-Means (0.761), GMM (0.758) and HDBSCAN (0.727).

**Serving benchmark** (local, 4 CPU, concurrency 8): 670 req/s, p50 9.0 ms, p95 28 ms, p99 50 ms, 0% errors. All SLOs were met.

### 7.2 Where current docs disagree with the evidence

| Claim (README / FINDINGS.md) | Evidence | Source of claim |
|---|---|---|
| Champion PR-AUC "0.91+" | 0.9521 in v0.5 | hard-coded in `render_report.py:694` |
| First-session concordance "~0.53" | 0.507 in v0.5 | hard-coded fallback, `render_report.py:702` |
| Day-7 landmark concordance "0.70+" | **no run exists** | hard-coded fallback, `render_report.py:703` |
| Uplift champion "X-Learner" | v0.5 selected `s_learner_lgbm`; X-learner came in below the response model | hard-coded fallback, `render_report.py:706` |
| "3x higher incremental lift" in top deciles | not in any artifact | hard-coded text, `render_report.py:801` |
| README: endpoints `/v1/churn/score` | actual route is `/v1/churn/predict` | README §5 |
| README: "<10 ms p99" | measured p99 2.7 ms at concurrency 1, 50 ms at concurrency 8 | README |
| README: "Zero hand-typed performance numbers" | contradicted by the fallbacks above | — |
| KNOWLEDGE_BANK: "51 unit tests", "Spec v0.5 frozen", "Production / Benchmark-Verified" | 64 tests; spec is v0.6; v0.6 not benchmarked | KNOWLEDGE_BANK §1 |

The X-learner in v0.6 is now cross-fitted, so it *may* do better after a re-run, but nobody has measured that yet. **Until `make reproduce` is run, treat every v0.6-specific claim as unverified.**

### 7.3 Sample data is synthetic
`data/sample/*.parquet` (~3.9 MB) are **generated synthetically** by `scripts/make_sample.py` with a seeded RNG. They are not subsamples of the real datasets, although the script and README call them "representative". The notebooks therefore show the *method* running end to end, not real findings. This should be stated clearly in the README and in each notebook. Also, `make demo` calls `make sample` first, so it regenerates (overwrites) these committed files on every run.

---

## 8. Known issues & gaps (prioritized)

### P0: correctness and credibility of what is published
1. **Regenerate the results.** Run `make reproduce` on the real data (§9). The current `docs/RESULTS.md` is empty and six README figures are missing (`churn_ladder_pr_auc`, `churn_shap_importance`, `survival_landmark_km`, `uplift_qini_curves`, `uplift_calibration_deciles`, `campaign_net_value_by_reach`).
2. **Remove the hard-coded fallbacks** in `evaluation/render_report.py` (~L694–806). If an artifact is missing, render "not available" or fail loudly. Do not print plausible-looking numbers.
3. **Make the README and FINDINGS agree with the artifacts** after the re-run. That covers the champion strategy, the concordance values and the "3x" claim, and correcting `/v1/churn/score` to `/v1/churn/predict`.
4. **Guard against this happening again.** Have `make report` refuse to overwrite `RESULTS.md` when `artifacts/` is empty, or add a CI check that the README's figure links resolve.

### P1: engineering hygiene
5. **Undeclared dependency:** `scripts/sql_check.py` imports `sqlparse`, which only arrives transitively through mlflow/dbt. Add it to the dev group.
6. **The fresh-clone experience depends on gitignored outputs.** The API image does `COPY data/models` and `COPY artifacts`, so `docker compose --profile serving up` fails unless `make evaluate` has run first. Consider publishing a small model bundle, or pulling one from the MLflow registry / S3 at startup.
7. **Duplicate docs:** `KNOWLEDGE_BANK.md` exists at the root and in `docs/` (identical). Keep one. Its status section is out of date (§7.2).
8. **Clean up the empty placeholder packages** listed in §3.
9. **Version drift:** the package is at `0.1.0` while the spec is at v0.6. There is no CHANGELOG apart from the spec's change log.
10. **Stale remote branch** `origin/phase-1-3-ingestion` (already merged in PR #1). Safe to delete.
11. **`data/samples/`** (plural, empty) sits alongside `data/sample/`. Looks like a leftover.

### P2: production hardening (if this ever takes real traffic)
12. The AWS ALB is HTTP-only with an open CIDR. It needs ACM/HTTPS and a narrower `allowed_cidrs`.
13. The API has no authentication and no rate limiting.
14. The defaults in `.env.example` and docker-compose use `retentioniq/retentioniq` credentials. Grafana allows anonymous viewing.
15. Drift monitoring is batch (`retentioniq monitor`). Nothing feeds outcome labels back for the live PR-AUC/ECE alerts in spec §9.3.
16. Silence or fix the three test warnings (§6).

---

## 9. How to run it

**Prerequisites:** Python 3.11 (`.python-version`), `uv`, Docker, Node 22 (web), and Kaggle credentials for RetailRocket (`KAGGLE_USERNAME` / `KAGGLE_KEY`). Copy `.env.example` to `.env`. A devcontainer is provided; it keeps the uv cache on `/tmp` because a fresh-clone install once ran out of disk.

```bash
make install            # uv sync --group dev --all-extras + pre-commit
make check              # lint + mypy + unit tests

# Quick demo (no DB, synthetic sample data)
make demo               # regenerates data/sample, executes notebooks 01–03
make sql-check

# Full pipeline (real data; downloads are large, Criteo ~14M rows)
make up                 # postgres + mlflow
make data ingest dbt    # download → validate/load → warehouse
make evaluate           # all stages → artifacts/ + data/models/
make report             # → docs/RESULTS.md, FINDINGS.md, figures
make promote            # gated MLflow registration
#   or simply: make reproduce   (= up data ingest dbt evaluate report)

make serve              # API on :8000
make benchmark          # → artifacts/serving_benchmark.json
cd apps/web && npm ci && npm run dev   # web on :3000
docker compose --profile monitoring up -d   # prometheus :9090, grafana :3001
make publish            # model outputs → Postgres reporting.* for Power BI
```

**Compute notes** (spec change log):
- A LightGBM fit on the full 8.4M Criteo training rows took about 15 minutes on 4 cores. Uplift learners are therefore trained on a 3M-row stratified subsample (v0.4), and the causal forest on 1M rows (v0.5).
- Plan for a multi-hour `make evaluate`.
- `--learning-curve` optionally measures how results scale across 500k, 1M, 2M and 3M rows.

**Determinism:** a global `SEED = 20261002` and deterministic LightGBM settings. Each evaluate stage records `spec_version`, `git_sha` and timing in the artifacts run manifest.

---

## 10. Spec change log (summary)

| Ver. | Date | Change |
|---|---|---|
| 0.1 | 10-02 | Initial contract |
| 0.2 | 10-02 | Inactivity observation window 60 → 30 days (60 days did not fit RetailRocket's ~138-day span) |
| 0.3 | 10-02 | Survival population = returning visitors; first-session-only covariates (avoids immortal-time bias) |
| 0.4 | 10-02 | Uplift learners on a 3M-row subsample (compute budget) |
| 0.5 | 10-03 | Causal forest added (1M-row subsample) |
| 0.6 | 11-04 | Parsimony rule across the ladder, MLP grid, cross-fitted X-learner + S-learner diagnostics, day-7 landmark survival, CLV. **No results published under v0.6 yet.** |

Every change before v0.6 was recorded *before* the affected models were fitted. Keep doing that: the spec requires that any change to a definition, split, metric or gate is logged with its reason and date.

---

## 11. Recommended first week

1. `make install && make check` to confirm the environment (§6).
2. Run `make reproduce` on a machine with ≥ 4 cores and enough disk, then `make promote` and `make benchmark`.
3. Fix the `render_report.py` fallbacks (P0 #2), re-render, and commit the regenerated `RESULTS.md`, `FINDINGS.md` and figures.
4. Update the README and KNOWLEDGE_BANK claims to match the new artifacts (P0 #3). Label the sample data as synthetic (§7.3).
5. Work through the P1 hygiene items.
6. Decide whether the AWS/production path matters. If it does, do the P2 items before exposing anything publicly.

## 12. Key documents
- `PROJECT_SPEC.md`: the source of truth for definitions, splits, metrics and gates.
- `KNOWLEDGE_BANK.md`: deep dive on design, trade-offs and edge cases (status section outdated).
- `docs/EDA.md`: the exploratory analysis and the hypotheses tested.
- `docs/RESULTS.md` / `docs/FINDINGS.md`: generated reports (currently stale, see §7).
- `sql/README.md`, `analytics/powerbi/README.md`, `infrastructure/aws/README.md`, `apps/web/README.md`.

---

## 13. v0.6 run results and what changed (supersedes §1, §7, §8 where they conflict)

**How it was run.**
- Machine: GitHub Codespace `standardLinux32gb` (4 cores, 16 GB RAM, 32 GB workspace disk plus about 118 GB on `/tmp`).
- Environment: the devcontainer's `make install` fails on a fresh Codespace with "No space left on device". The venv, raw data and Docker's data root had to move to `/tmp` (environment-only change, nothing in the repo). `data/models` must stay on the same filesystem as `artifacts/`, because `training/churn.py:455` moves files with `Path.replace`.
- Timing: evaluate took about 1h50m in total; uplift was about 75 minutes of it.

**Fixed in the repo during the run (branch `v0.6-report-cleanup`).**
- Criteo download URL: `go.criteo.net` returns 404. It now points to Criteo's own ungated Hugging Face copy. The row count matches v0.5 exactly (13,979,592).
- `render_report.py`:
  - The uplift Qini and decile figures mixed the visit and conversion outcomes into one line per strategy. They now plot visit only.
  - `RESULTS.md` had a trailing-blank-line issue.
  - The uplift findings now report the paired-bootstrap significance, the conversion outcome and the response-model profit comparison.

**Results (v0.6).**

| Area | Result | Claim status |
|---|---|---|
| Churn ladder | M4 LightGBM PR-AUC 0.9521 [0.9493, 0.9550], same as v0.5 to 4 dp; parsimony rule keeps M4; M5 0.9514 | Holds |
| Promotion | All 5 gates pass vs M1 reference; registered `retentioniq-churn` v1 | Holds |
| Survival | First-session concordance 0.507; **day-7 landmark 0.506** | **"Day-7 breakthrough" refuted** |
| Uplift (visit), v0.6.4 | S-learner 0.0927 vs response 0.0858; bootstrap **significant** (p = 0.020, CI [0.0018, 0.0241]); causal forest also significant (p < 0.001). The v0.6 run's split gave p = 0.096 | Holds on visits, split-sensitive |
| Uplift (conversion), v0.6.4 | Response model best (0.1851 vs S-learner 0.1763; no pairwise difference significant) | **Refuted for conversions** |
| Campaign scenario, v0.6.4 | Response model peaks at 22% reach / $21.4k; S-learner 19% / $21.3k | Level under the assumed economics |
| CLV (re-run under v0.6.1 after the fix below) | BG/NBD purchase MAE 0.694 vs naive 0.701; spend-ranking Spearman 0.579 vs 0.525 | Better at ranking; not a revenue forecast |
| Serving | 668 req/s; p99 38 ms at concurrency 8; all SLOs met | Holds |

**New open issues found by the run (prioritized).**
1. **CLV bug (P0). FIXED in v0.6.1 (`67d6365`).** Two causes. (a) `lifetimes` takes `log()` of a hypergeometric term that is legitimately negative when a + b + x − 1 < 0, giving NaN purchase predictions for 304 one-time buyers; `np.mean` on a Series skipped them, so the MAE looked fine. Now evaluated in closed form (matches lifetimes to 3e-13 where it is finite). (b) Gamma-Gamma fitted q = 0.34; with q ≤ 1 the population mean is undefined and 1,643 customers got negative spend. Spend now uses Gamma-Gamma only when q > 1, else an empirical-Bayes log-normal model. Predictions are asserted finite and non-negative; 3 unit tests added. Re-run from `fact_orders` rebuilt in pandas from the raw xlsx, which reproduced every pre-fix metric exactly.
2. **`clv_by_segment.csv` "New Customers" row (P1). FIXED** by the same change: the segment means were silently skipping the NaN predictions.
3. **Test-set leakage in model selection (P1). FIXED in spec v0.6.2, re-run under v0.6.3.** The parsimony rule's tie test now uses paired bootstrap CIs on validation (clustered by visitor), on uncalibrated validation scores (`models/churn/selection.py`, whose signature accepts no test data). The re-run showed that the leak had decided the outcome: on validation, M3 random forest ties M4, and it was selected and promoted, but it served 28 req/s (p95 384 ms) against the 200 req/s / 100 ms SLO. Spec v0.6.3 (declared post-hoc in the change log) adds serving-cost eligibility (p95 ≤ 5 ms on the API's own scoring path) and a 6th promotion gate. Final: M4 champion, all 6 gates pass, 664 req/s with p95 29 ms. Follow-up done: logistic models now score from folded coefficients (`LinearForm`), about 0.006 ms per record instead of 2–6 ms, so M1/M2 no longer sit near the budget. The champion is unaffected; the published latency column is from the v0.6.3 run until the next re-run.
4. **Pairwise Qini bootstrap is single-threaded (P2).** `training/uplift.py:240` makes 12 comparisons × 1,000 draws × 500k rows, about 30 minutes on 1 core. Parallelizing it with `joblib` would cut that roughly 4×.
5. **Fragile file moves (P2).** `training/churn.py:455` uses `Path.replace`; switch to `shutil.move` so model dirs can live on another volume.
6. **Devcontainer disk (P2).** `.devcontainer/post-create.sh` should put `UV_PROJECT_ENVIRONMENT` and Docker's data root on `/tmp`, so a fresh Codespace installs without manual steps.
7. **Odd threshold scenario (P2).** The most profitable churn threshold scenario contacts 99.8% of visitors. That is plausible given an 87% inactive base rate, but worth a sanity check.
8. **Uplift results depended on database row order (P1). FIXED in spec v0.6.4 (`2034f37`).** `load_population` read the mart without ORDER BY, and the seeded splits are positional, so two runs of identical code trained and tested on different rows (S-learner vs propensity on visits: p = 0.096, then p = 0.020). The population is now sorted by `row_id` before splitting; a real-data check confirmed that a shuffled load gives identical splits, and the re-run reproduced the second run exactly. Churn was unaffected (it sorts its inputs).
8. **Noisy rationale text (P2).** When the best model is also the simplest qualifying one, `churn_selection.json` reports a trivial CI ([0, 0]). The text should say so.
