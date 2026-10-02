"""Command-line entry point: `uv run retentioniq --help`."""

import typer

from retentioniq import SPEC_VERSION, __version__

app = typer.Typer(no_args_is_help=True, help="RetentionIQ pipelines and evaluation.")


@app.command()
def version() -> None:
    """Print package and spec versions."""
    typer.echo(f"retentioniq {__version__} (spec v{SPEC_VERSION})")


if __name__ == "__main__":
    app()
