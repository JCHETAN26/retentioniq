.PHONY: install lint format typecheck test test-integration check up down data ingest evaluate

install:        ## Core + dev deps
	uv sync --group dev
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
	uv run pytest -m integration

check: lint typecheck test

up:             ## Postgres + MLflow
	docker compose up -d --build

down:
	docker compose down

data:           ## Phase 1: download raw datasets (RetailRocket needs KAGGLE_USERNAME/KAGGLE_KEY)
	uv run retentioniq download

ingest:         ## Phase 3: validate + load raw data into Postgres (idempotent)
	uv run retentioniq ingest

evaluate:       ## Regenerates every artifact in PROJECT_SPEC.md §8 (built up phase by phase)
	@echo "Not implemented yet — see PROJECT_SPEC.md §8" && exit 1
