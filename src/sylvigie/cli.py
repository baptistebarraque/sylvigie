"""Command-line interface.

One command per pipeline stage of the roadmap. Stages that are not built yet
exit with a non-zero code and say which roadmap step will implement them, so
the shape of the pipeline is visible from the start and each step fills in
its own command.
"""

import logging
from pathlib import Path

import typer

from sylvigie import __version__

app = typer.Typer(
    help="Sylvigie: clear-cut and dieback detection from Sentinel time series.",
    no_args_is_help=True,
)

ConfigOption = typer.Option(
    ..., "--config", "-c", exists=True, dir_okay=False, help="Zone config (YAML)."
)
DataDirOption = typer.Option(Path("data"), "--data-dir", help="Root data folder.")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _not_implemented(step: str) -> None:
    typer.echo(f"Not implemented yet (roadmap step {step}).", err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed Sylvigie version."""
    typer.echo(__version__)


@app.command("suggest-zone")
def suggest_zone(config: Path = ConfigOption, data_dir: Path = DataDirOption) -> None:
    """Rank candidate pilot windows from SUFOSAT clear-cut density."""
    from sylvigie.config import load_config
    from sylvigie.zone import suggest_zone as run

    cfg = load_config(config)
    result = run(cfg, data_dir)
    sel = cfg.selection
    typer.echo(
        f"\n{result.n_valid_windows} windows meet the thresholds "
        f">= {sel.min_cuts_per_year} cuts/year, >= {sel.min_pnr_fraction:.0%} in park."
    )
    typer.echo(
        f"Park profile: cut rate {100 * result.profile.cut_rate:.2f} %/yr, "
        f"conifer share {result.profile.conifer_share:.0%}.\n"
    )
    if result.candidates.empty:
        typer.echo("No candidate: relax the thresholds in the config.", err=True)
        raise typer.Exit(code=1)
    table = result.candidates.assign(
        bbox=[
            "[" + ", ".join(f"{v:.0f}" for v in geom.bounds) + "]"
            for geom in result.candidates.geometry
        ]
    ).drop(columns="geometry")
    typer.echo(table.round(3).to_string(index=False))
    typer.echo("")
    for name, path in result.outputs.items():
        typer.echo(f"{name:>14}: {path}")


@app.command()
def grid(config: Path = ConfigOption, data_dir: Path = DataDirOption) -> None:
    """Build the reference grid, forest mask and SUFOSAT labels of each window."""
    from sylvigie.config import load_config
    from sylvigie.reference import build_reference

    cfg = load_config(config)
    for window in build_reference(cfg, data_dir):
        typer.echo(f"\n=== {window.name} window (inside the margin)")
        typer.echo(window.summary.round(1).to_string(index=False))
        shares = ", ".join(f"{y}: {s:.1%}" for y, s in window.coverage.items())
        typer.echo(f"Clear-cut pixels inside the forest mask: {shares}")
        for name, path in window.outputs.items():
            typer.echo(f"{name:>14}: {path}")


@app.command()
def ingest() -> None:
    """Download Sentinel-2 data and write the masked index cube (Zarr)."""
    _not_implemented("3")


@app.command()
def detect() -> None:
    """Detect clear-cuts and write alert polygons (GeoPackage)."""
    _not_implemented("4")


@app.command()
def evaluate() -> None:
    """Score detections against SUFOSAT (area and object metrics, delay)."""
    _not_implemented("4")
