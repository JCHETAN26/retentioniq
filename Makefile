.PHONY: reproduce dbt install lint format typecheck test test-integration check up down data ingest evaluate

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

reproduce: up data ingest dbt evaluate  ## Everything, from raw downloads to artifacts
