"""Per-pixel index time series of typical forest situations, for visual checks.

Pixels are picked automatically from the step 2b layers: for each situation,
the pixel deepest inside a patch of it (largest distance to its edge), so
the series is not blurred by neighbouring stands or by the cut's border.
"""

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import ndimage

from sylvigie.forest import BEECH, DOUGLAS_FIR, FIR_SPRUCE, OAK, OTHER_BROADLEAF
from sylvigie.labels import NO_CUT, days_to_year
from sylvigie.preprocess.indices import compute_index
from sylvigie.viz.maps import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

SERIES_COLOR = "#2a78d6"  # categorical slot 1
WINTER_FILL = "#ecebe7"
STABLE_BUFFER_PX = 10  # a stable pixel is at least 100 m from any SUFOSAT cut


@dataclass(frozen=True)
class Situation:
    key: str
    label: str
    forest_types: tuple[int, ...]
    cut_year: int | None  # None: stable, never cut according to SUFOSAT


def situations(cut_year: int) -> list[Situation]:
    return [
        Situation(
            "stable_conifer", "Stable conifer\n(Douglas fir)", (DOUGLAS_FIR,), None
        ),
        Situation(
            "stable_broadleaf",
            "Stable broadleaf\n(oak, beech...)",
            (OAK, BEECH, OTHER_BROADLEAF),
            None,
        ),
        Situation(
            "douglas_cut",
            f"Douglas fir\nclear-cut {cut_year}",
            (DOUGLAS_FIR,),
            cut_year,
        ),
        Situation(
            "spruce_cut", f"Fir / spruce\nclear-cut {cut_year}", (FIR_SPRUCE,), cut_year
        ),
    ]


@dataclass(frozen=True)
class SamplePixel:
    situation: Situation
    row: int
    col: int
    depth_px: float  # distance to the patch edge
    cut_date: dt.date | None


def pick_pixels(
    forest_type: np.ndarray,
    sufosat_date: np.ndarray,
    cases: list[Situation],
    within: np.ndarray | None = None,
) -> list[SamplePixel]:
    """The deepest pixel of each situation; situations absent here are skipped."""
    cut_year = days_to_year(sufosat_date)
    near_any_cut = ndimage.binary_dilation(
        sufosat_date != NO_CUT, iterations=STABLE_BUFFER_PX
    )
    picked = []
    for case in cases:
        region = np.isin(forest_type, case.forest_types)
        if case.cut_year is None:
            region &= ~near_any_cut
        else:
            region &= cut_year == case.cut_year
        if within is not None:
            region &= within
        depth = ndimage.distance_transform_edt(region)
        if depth.max() == 0:
            continue
        row, col = np.unravel_index(np.argmax(depth), depth.shape)
        days = int(sufosat_date[row, col])
        cut_date = None if days == NO_CUT else dt.date(1970, 1, 1) + dt.timedelta(days)
        picked.append(
            SamplePixel(case, int(row), int(col), float(depth.max()), cut_date)
        )
    return picked


def pixel_series(
    cube: xr.Dataset,
    pixel: SamplePixel,
    indices: list[str],
    scl_valid: list[int],
    dilation_px: int,
) -> pd.DataFrame:
    """Index values of one pixel over time; cloudy dates are dropped.

    A small neighbourhood is loaded so the cloud-mask dilation sees the
    clouds next to the pixel.
    """
    pad = dilation_px + 1
    rows = slice(max(pixel.row - pad, 0), pixel.row + pad + 1)
    cols = slice(max(pixel.col - pad, 0), pixel.col + pad + 1)
    patch = cube.isel(y=rows, x=cols).load()
    r, c = pixel.row - rows.start, pixel.col - cols.start
    table = pd.DataFrame(
        {
            name: compute_index(patch, name, scl_valid, dilation_px)
            .isel(y=r, x=c)
            .to_series()
            for name in indices
        }
    )
    return table.dropna(how="all")


PLOT_INDICES = ["NBR", "NDVI", "CRSWIR"]


def plot_window_series(
    cfg, window: str, data_dir: Path, subset_km: float | None = None
) -> tuple[Path, Path, list[SamplePixel]]:
    """Pick the sample pixels of a window's cube and plot their series.

    Returns the figure path, a CSV of the plotted values and the pixels.
    """
    from sylvigie.ingest.sentinel2 import window_paths
    from sylvigie.preprocess.indices import cube_grid, open_cube
    from sylvigie.raster import read_on_grid

    paths = window_paths(cfg, window, data_dir, subset_km)
    if not paths.cube.exists():
        raise FileNotFoundError(f"{paths.cube} missing: run `sylvigie ingest` first")
    cube = open_cube(paths.cube)
    grid = cube_grid(cube)
    layers = paths.forest_mask.parent
    forest_type = read_on_grid(layers / "forest_type.tif", grid)
    sufosat_date = read_on_grid(layers / "sufosat_date.tif", grid)
    calibration_year = cfg.period.detection_years[0]
    pixels = pick_pixels(forest_type, sufosat_date, situations(calibration_year))
    s2 = cfg.s2
    series = [
        (p, pixel_series(cube, p, PLOT_INDICES, s2.scl_valid, s2.mask_dilation_px))
        for p in pixels
    ]
    out = paths.cube.parent
    figure = plot_pixel_series(
        series,
        PLOT_INDICES,
        out / "timeseries.png",
        title=f"{cfg.name.capitalize()} {paths.cube.parent.name}: Sentinel-2 index "
        "time series of typical pixels",
    )
    table = pd.concat(
        {p.situation.key: t for p, t in series}, names=["situation", "time"]
    ).reset_index()
    csv = out / "timeseries.csv"
    table.to_csv(csv, index=False)
    return figure, csv, pixels


def plot_pixel_series(
    series: list[tuple[SamplePixel, pd.DataFrame]],
    indices: list[str],
    path: Path,
    title: str,
) -> Path:
    """Small multiples: one row per pixel, one column per index (own y-axis)."""
    n_rows, n_cols = len(series), len(indices)
    fig = Figure(figsize=(4.6 * n_cols, 2.3 * n_rows + 0.8), facecolor=SURFACE)
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey="col", squeeze=False)
    start = min(table.index.min() for _, table in series)
    end = max(table.index.max() for _, table in series)

    for i, (pixel, table) in enumerate(series):
        for j, name in enumerate(indices):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for year in range(start.year - 1, end.year + 1):
                # Leaf-off season: broadleaf indices drop for reasons other than cuts.
                ax.axvspan(
                    pd.Timestamp(year, 11, 1),
                    pd.Timestamp(year + 1, 3, 31),
                    color=WINTER_FILL,
                    lw=0,
                    zorder=0,
                )
            values = table[name].dropna()
            ax.plot(values.index, values.values, color=SERIES_COLOR, lw=0.8, alpha=0.5)
            ax.plot(values.index, values.values, "o", color=SERIES_COLOR, ms=2.5)
            if pixel.cut_date is not None:
                ax.axvline(
                    pd.Timestamp(pixel.cut_date), color=TEXT_PRIMARY, lw=1.2, ls="--"
                )
            ax.set_xlim(start, end)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
            ax.grid(axis="y", color="#e2e1dc", lw=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if i == 0:
                ax.set_title(name, color=TEXT_PRIMARY, fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel(pixel.situation.label, color=TEXT_PRIMARY, fontsize=9)

    fig.legend(
        handles=[
            Line2D(
                [], [], color=SERIES_COLOR, marker="o", ms=3, lw=0.8, label="Clear date"
            ),
            Line2D(
                [], [], color=TEXT_PRIMARY, ls="--", lw=1.2, label="SUFOSAT cut date"
            ),
            Line2D([], [], color=WINTER_FILL, lw=8, label="November-March"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    return path
