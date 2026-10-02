# Airflow with the RetentionIQ project installed, so tasks can call `uv run retentioniq`.
FROM apache/airflow:2.10.5-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
USER airflow
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/home/airflow/riq-venv
WORKDIR /opt/retentioniq
