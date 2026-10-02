"""RetentionIQ training pipeline (Phase 21).

validate -> transform -> features+train+evaluate -> explain -> promote (gated) -> report

Every task shells out to the same CLI used locally (`retentioniq ...`, `dbt ...`), so the
DAG adds scheduling, retries and lineage without a second implementation of the pipeline.
`promote` exits non-zero when a gate fails, which fails the run and leaves the current
production model in place.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO = "/opt/retentioniq"
RIQ = f"cd {REPO} && uv run retentioniq"

default_args = {
    "owner": "retentioniq",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retentioniq_training",
    description="Validate, transform, train, evaluate and (gated) promote the churn model",
    start_date=datetime(2026, 10, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["retentioniq", "ml"],
) as dag:
    validate = BashOperator(
        task_id="validate_and_ingest",
        # Idempotent: unchanged files are skipped via the checksum ledger; reject-rate gates
        # abort the load (and this task) if a source degrades.
        bash_command=f"{RIQ} download && {RIQ} ingest",
    )
    transform = BashOperator(
        task_id="transform_dbt",
        bash_command=f"cd {REPO}/dbt && uv run --extra dbt dbt build --profiles-dir .",
    )
    train = BashOperator(
        task_id="features_train_evaluate",
        bash_command=f"{RIQ} evaluate --only churn",
        execution_timeout=timedelta(hours=3),
    )
    explain = BashOperator(task_id="explain", bash_command=f"{RIQ} evaluate --only explain")
    promote = BashOperator(task_id="compare_champion_and_register", bash_command=f"{RIQ} promote")
    analytics = BashOperator(
        task_id="refresh_analytics",
        bash_command=f"{RIQ} evaluate --only customers --only survival",
    )

    validate >> transform >> [train, analytics]
    train >> explain >> promote
