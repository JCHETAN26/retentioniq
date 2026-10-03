.PHONY: report promote monitor publish serve benchmark reproduce dbt install lint format typecheck test test-integration check up down data ingest evaluate

install:        ## All deps (core, ML, survival, deep, dbt, serving, monitoring) + dev tools
	uv sync --group dev --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest -m "not integration and not data"

test-integration:  ## needs `make up`
	uv run pytest tests/integration -m integration

check: lint typecheck test

up:             ## Postgres + MLflow
	docker compose up -d --build

down:
	docker compose down

data:           ## Phase 1: download raw datasets (RetailRocket needs KAGGLE_USERNAME/KAGGLE_KEY)
	uv run retentioniq download

ingest:         ## Phase 3: validate + load raw data into Postgres (idempotent)
	uv run retentioniq ingest

dbt:            ## Phase 5: build + test the warehouse (staging -> intermediate -> marts)
	cd dbt && uv run --extra dbt dbt deps --profiles-dir . && uv run --extra dbt dbt build --profiles-dir .

evaluate:       ## Regenerate every artifact in PROJECT_SPEC.md §8 (needs `make up data ingest dbt`)
	uv run retentioniq evaluate

report:         ## Render docs/RESULTS.md from artifacts (no hand-typed numbers)
	uv run python evaluation/render_report.py

promote:        ## Promotion gates (PROJECT_SPEC §9.1); registers the champion in MLflow
	uv run retentioniq promote

monitor:        ## Drift / calibration / performance report (PROJECT_SPEC §9.3)
	uv run retentioniq monitor

publish:        ## Model outputs -> Postgres reporting.* for Power BI
	uv run retentioniq publish

serve:          ## Run the API locally on :8000
	uv run uvicorn retentioniq.serving.app:app --host 0.0.0.0 --port 8000 --workers 2

benchmark:      ## Load-test a running API -> artifacts/serving_benchmark.json
	uv run python evaluation/benchmark_serving.py --url http://localhost:8000

reproduce: up data ingest dbt evaluate report  ## Everything, from raw downloads to the results report
