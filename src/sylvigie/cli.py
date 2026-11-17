"""Command-line interface.

One command per pipeline stage of the roadmap. Stages that are not built yet
exit with a non-zero code and say which roadmap step will implement them, so
the shape of the pipeline is visible from the start and each step fills in
its own command.
"""

import typer

from sylvigie import __version__

app = typer.Typer(
    help="Sylvigie: clear-cut and dieback detection from Sentinel time series.",
    no_args_is_help=True,
)


def _not_implemented(step: int) -> None:
    typer.echo(f"Not implemented yet (roadmap step {step}).", err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed Sylvigie version."""
    typer.echo(__version__)


@app.command()
def grid() -> None:
    """Build the reference grid, forest mask and SUFOSAT labels for a zone."""
    _not_implemented(step=2)


@app.command()
def ingest() -> None:
    """Download Sentinel-2 data and write the masked index cube (Zarr)."""
    _not_implemented(step=3)


@app.command()
def detect() -> None:
    """Detect clear-cuts and write alert polygons (GeoPackage)."""
    _not_implemented(step=4)


@app.command()
def evaluate() -> None:
    """Score detections against SUFOSAT (area and object metrics, delay)."""
    _not_implemented(step=4)
