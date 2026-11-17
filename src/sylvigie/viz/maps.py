"""Quick static maps."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from sylvigie.aoi import Grid

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
# Single-hue sequential ramp (light = low, dark = high).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

_km = FuncFormatter(lambda v, _: f"{v / 1000:.0f}")


def plot_zone_candidates(
    density: np.ndarray,
    density_grid: Grid,
    park: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    path: Path,
    title: str,
    years: list[int],
) -> Path:
    """Clear-cut density, park boundary and numbered candidate windows."""
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)
    cmap.set_bad(alpha=0)  # cells without clear-cuts show the surface

    xmin, ymin, xmax, ymax = density_grid.bounds
    # Object-oriented API: no pyplot state, no GUI backend needed.
    fig = Figure(figsize=(8, 10), facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    image = ax.imshow(
        np.ma.masked_less_equal(density, 0),
        cmap=cmap,
        extent=(xmin, xmax, ymin, ymax),
        interpolation="nearest",
    )
    park.to_crs(density_grid.crs).boundary.plot(ax=ax, color=TEXT_SECONDARY, lw=1.2)

    for row in candidates.itertuples():
        x0, y0, x1, y1 = row.geometry.bounds
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=TEXT_PRIMARY, lw=2)
        )
        ax.text(
            x0 + 800,
            y1 - 800,
            str(row.candidate),
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=TEXT_PRIMARY,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": SURFACE,
                "ec": "none",
                "alpha": 0.85,
            },
        )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    colorbar.set_label(
        f"Clear-cut area, mean {years[0]}-{years[-1]} (ha per km² per year)",
        color=TEXT_SECONDARY,
    )
    colorbar.outline.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], color=TEXT_SECONDARY, lw=1.2, label="Park boundary"),
            Line2D([], [], color=TEXT_PRIMARY, lw=2, label="Candidate window"),
        ],
        loc="lower left",
        frameon=False,
    )
    ax.xaxis.set_major_formatter(_km)
    ax.yaxis.set_major_formatter(_km)
    ax.set_xlabel(f"Easting, km ({density_grid.crs.to_string()})", color=TEXT_SECONDARY)
    ax.set_ylabel("Northing, km", color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return path
