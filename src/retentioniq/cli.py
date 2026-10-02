"""Command-line entry point: `uv run retentioniq --help`."""

from typing import Annotated

import typer

from retentioniq import SPEC_VERSION, __version__
from retentioniq.config import settings
from retentioniq.ingestion.sources import DATASETS

app = typer.Typer(no_args_is_help=True, help="RetentionIQ pipelines and evaluation.")

DatasetArg = Annotated[
    list[str] | None,
    typer.Argument(help=f"Datasets to process (default: all). Choices: {', '.join(DATASETS)}"),
]
ForceOpt = Annotated[bool, typer.Option("--force", help="Redo even if already up to date.")]


def _resolve(names: list[str] | None) -> list[str]:
    names = names or list(DATASETS)
    unknown = set(names) - DATASETS.keys()
    if unknown:
        raise typer.BadParameter(f"unknown dataset(s): {sorted(unknown)}")
    return names


@app.command()
def version() -> None:
    """Print package and spec versions."""
    typer.echo(f"retentioniq {__version__} (spec v{SPEC_VERSION})")


@app.command()
def download(datasets: DatasetArg = None, force: ForceOpt = False) -> None:
    """Download raw datasets into data/raw/<dataset>/ with a checksum manifest."""
    from retentioniq.ingestion.download import download as _download

    for name in _resolve(datasets):
        path = _download(DATASETS[name], settings.raw_dir, force=force)
        typer.echo(f"{name}: ready at {path}")


@app.command()
def ingest(datasets: DatasetArg = None, force: ForceOpt = False) -> None:
    """Validate raw files and load them into Postgres (raw / quarantine / meta schemas)."""
    from retentioniq.ingestion.load import ingest_dataset

    for name in _resolve(datasets):
        for r in ingest_dataset(settings.database_url, DATASETS[name], settings.raw_dir, force):
            typer.echo(
                f"{r.dataset}/{r.source_file}: {r.status} "
                f"rows={r.row_count} loaded={r.loaded_count} rejected={r.rejected_count}"
            )


@app.command()
def evaluate(
    only: Annotated[
        list[str] | None, typer.Option(help="Run only these stages (repeatable).")
    ] = None,
    torch: Annotated[bool, typer.Option(help="Include the PyTorch model (M5).")] = True,
    sensitivity: Annotated[bool, typer.Option(help="Run churn sensitivity analysis.")] = True,
) -> None:
    """Regenerate all artifacts in PROJECT_SPEC.md §8 from the warehouse marts."""
    import logging

    from retentioniq.evaluate import STAGES, run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    unknown = set(only or []) - set(STAGES)
    if unknown:
        raise typer.BadParameter(f"unknown stage(s): {sorted(unknown)}; choose from {STAGES}")
    manifest = run(only, include_torch=torch, sensitivity=sensitivity)
    for name, info in manifest["stages"].items():
        typer.echo(f"{name}: {info['seconds']}s  {info['summary']}")


@app.command()
def promote() -> None:
    """Apply the promotion gates (PROJECT_SPEC.md §9.1) and promote the challenger if all pass."""
    from retentioniq.config import REPO_ROOT
    from retentioniq.evaluate import MODEL_DIR
    from retentioniq.training.promotion import promote as _promote

    decision = _promote(settings.artifacts_dir, MODEL_DIR, REPO_ROOT / "dbt" / "target")
    for g in decision.gates:
        typer.echo(f"{'PASS' if g.passed else 'FAIL'}  {g.name}  {g.detail}")
    typer.echo(
        f"{'PROMOTED' if decision.promote else 'NOT PROMOTED'}: {decision.challenger} "
        f"(reference: {decision.reference})"
    )
    if not decision.promote:
        raise typer.Exit(code=1)


@app.command()
def monitor() -> None:
    """Data/model drift and performance report (PROJECT_SPEC.md §9.3)."""
    from retentioniq.evaluate import MODEL_DIR
    from retentioniq.monitoring.drift import run

    result = run(settings.artifacts_dir, MODEL_DIR)
    typer.echo(f"status={result['status']} drifted={result['drifted_features']}")


@app.command()
def publish() -> None:
    """Load evaluation outputs into Postgres `reporting.*` for Power BI."""
    from retentioniq.reporting import publish as _publish

    for table, n in _publish(settings.database_url, settings.artifacts_dir).items():
        typer.echo(f"reporting.{table}: {n} rows")


if __name__ == "__main__":
    app()
