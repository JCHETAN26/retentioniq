# RetentionIQ API image. Build from the repo root:
#   docker build -f apps/api/Dockerfile -t retentioniq-api .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serving --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra serving

# Models and the artifacts the API serves are produced by `make evaluate`.
COPY data/models ./data/models
COPY artifacts ./artifacts

ENV RIQ_ARTIFACTS_DIR=/app/artifacts PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "retentioniq.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
