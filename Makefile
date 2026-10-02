.PHONY: install lint format typecheck test check up down evaluate

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

check: lint typecheck test

up:             ## Postgres + MLflow
	docker compose up -d --build

down:
	docker compose down

evaluate:       ## Regenerates every artifact in PROJECT_SPEC.md §8 (built up phase by phase)
	@echo "Not implemented yet — see PROJECT_SPEC.md §8" && exit 1
