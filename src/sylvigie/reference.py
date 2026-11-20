"""Reference layers of each pilot window, on its own 10 m grid.

For a window and its margin: forest mask, forest type, SUFOSAT clear-cut date
and probability, and the evaluation area (the window without its margin).
Every later step reads these GeoTIFFs and loads satellite data on the same
grid, so pixel (i, j) is the same 10 x 10 m of ground in every layer.
"""

import datetime as dt
import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.warp import transform_bounds
from shapely.geometry import box

from sylvigie import __version__
from sylvigie.aoi import Grid
from sylvigie.config import Bbox, ZoneConfig
from sylvigie.forest import FOREST_GROUPS, FOREST_TYPE_NAMES, NON_FOREST, forest_layers
from sylvigie.ingest.ancillary import (
    BDFORET_V2_LAYER,
    MASQUE_FORET_LAYER,
    cached_gpkg,
    download_sufosat,
    fetch_bdforet_v2,
    fetch_masque_foret,
)
from sylvigie.labels import NO_CUT, days_to_year, yyddd_to_days
from sylvigie.raster import read_on_grid, write_geotiff

log = logging.getLogger(__name__)

SUFOSAT_SOURCE = "SUFOSAT v3.0.1, Mermoz et al. 2024, Zenodo record 17253008"
# Display colours for the review map (categorical slots, fixed order).
GROUP_COLORS = {
    "broadleaf": "#2a78d6",
    "conifer": "#1baf7a",
    "mixed": "#6250d6",
    "no tree cover": "#8f8d86",
    "unknown": "#c3c2b7",
}
OUTSIDE_MASK_COLOR = "#e34948"


@dataclass(frozen=True)
class ReferenceLayers:
    grid: Grid
    eval_area: np.ndarray  # bool: inside the window (not the margin)
    forest_mask: np.ndarray  # bool
    forest_type: np.ndarray  # uint8, sylvigie.forest codes
    sufosat_date: np.ndarray  # int32, days since 1970-01-01, 0 = no cut
    sufosat_prob: np.ndarray  # uint8, %, 0 = no cut


def window_grid(
    bounds: Bbox, margin_m: float, crs: str, resolution: float
) -> tuple[Grid, np.ndarray]:
    """Grid of the window plus its margin, and the window's pixels (bool)."""
    xmin, ymin, xmax, ymax = bounds
    grid = Grid.from_bounds(
        (xmin - margin_m, ymin - margin_m, xmax + margin_m, ymax + margin_m),
        crs,
        resolution,
    )
    (rows, cols), _ = grid.crop(bounds)
    eval_area = np.zeros(grid.shape, dtype=bool)
    eval_area[rows, cols] = True
    return grid, eval_area


def label_layers(
    dates_path: Path, prob_path: Path, grid: Grid
) -> tuple[np.ndarray, np.ndarray]:
    """SUFOSAT date (days since 1970) and probability (%) on `grid`.

    Nearest neighbour only: a date or a probability must never be averaged.
    """
    days = yyddd_to_days(read_on_grid(dates_path, grid))
    prob = read_on_grid(prob_path, grid)
    prob = np.where(days != NO_CUT, prob, 0).clip(0, 100).astype(np.uint8)
    return days, prob


def build_layers(
    grid: Grid,
    eval_area: np.ndarray,
    bdforet: gpd.GeoDataFrame,
    masque: gpd.GeoDataFrame,
    dates_path: Path,
    prob_path: Path,
) -> ReferenceLayers:
    # The forest layers are built from the forest maps alone: no label input.
    forest_mask, forest_type = forest_layers(bdforet, masque, grid)
    sufosat_date, sufosat_prob = label_layers(dates_path, prob_path, grid)
    return ReferenceLayers(
        grid, eval_area, forest_mask, forest_type, sufosat_date, sufosat_prob
    )


def summarize(
    layers: ReferenceLayers, years: list[int]
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Area and SUFOSAT clear-cut area by forest type, inside the window.

    Also returns, per year, the share of clear-cut pixels inside the forest
    mask: cuts outside it can never be detected.
    """
    pixel_ha = layers.grid.resolution**2 / 10_000
    inside = layers.eval_area
    cut_year = days_to_year(layers.sufosat_date)
    group_of = {code: g for g, codes in FOREST_GROUPS.items() for code in codes}

    def row(name: str, group: str, where: np.ndarray) -> dict:
        record = {
            "forest_type": name,
            "group": group,
            "area_ha": where.sum() * pixel_ha,
        }
        for year in years:
            record[f"cut_ha_{year}"] = (where & (cut_year == year)).sum() * pixel_ha
        return record

    rows = [
        row(name, group_of[code], inside & (layers.forest_type == code))
        for code, name in FOREST_TYPE_NAMES.items()
        if code != NON_FOREST
    ]
    rows.append(row("outside forest mask", "-", inside & ~layers.forest_mask))
    table = pd.DataFrame(rows)

    coverage = {}
    for year in years:
        cut = inside & (cut_year == year)
        coverage[year] = float((cut & layers.forest_mask).sum() / max(cut.sum(), 1))
    return table, coverage


@dataclass(frozen=True)
class WindowReference:
    name: str
    summary: pd.DataFrame
    coverage: dict[int, float]
    outputs: dict[str, Path]


def _retrieved(path: Path) -> str:
    return dt.date.fromtimestamp(path.stat().st_mtime).isoformat()


def build_reference(cfg: ZoneConfig, data_dir: Path) -> list[WindowReference]:
    """Build and write the reference layers of every window in the config."""
    if cfg.windows is None:
        raise ValueError("the config has no windows: run suggest-zone and fill them in")
    raw = data_dir / "raw"
    dates_path = download_sufosat(cfg.sufosat.dates, raw)
    prob_path = download_sufosat(cfg.sufosat.prob, raw)
    results = []
    for name, bounds in cfg.windows:
        log.info("Window %s %s", name, bounds)
        results.append(
            _build_window(cfg, name, bounds, data_dir, dates_path, prob_path)
        )
    return results


def _build_window(
    cfg: ZoneConfig,
    name: str,
    bounds: Bbox,
    data_dir: Path,
    dates_path: Path,
    prob_path: Path,
) -> WindowReference:
    grid, eval_area = window_grid(bounds, cfg.grid.margin_m, cfg.crs, cfg.resolution)
    log.info(
        "Grid %d x %d pixels (margin %d m)", grid.width, grid.height, cfg.grid.margin_m
    )
    l93 = transform_bounds(grid.crs, "EPSG:2154", *grid.bounds, densify_pts=21)
    ign = data_dir / "raw" / "ign"
    bdforet_path = ign / f"bdforet_v2_{cfg.name}_{name}.gpkg"
    masque_path = ign / f"masque_foret_2021_2023_{cfg.name}_{name}.gpkg"
    bdforet = cached_gpkg(bdforet_path, functools.partial(fetch_bdforet_v2, l93))
    masque = cached_gpkg(masque_path, functools.partial(fetch_masque_foret, l93))
    vintages = sorted(int(v) for v in masque["millesime"].dropna().unique())
    log.info("Masque Forêt vintages in window: %s", vintages)

    layers = build_layers(grid, eval_area, bdforet, masque, dates_path, prob_path)
    summary, coverage = summarize(layers, cfg.period.years)

    out = data_dir / "interim" / cfg.name / name
    common = {
        "sylvigie_version": __version__,
        "zone": cfg.name,
        "window": name,
        "window_bounds": json.dumps(list(bounds)),
        "created": dt.date.today().isoformat(),
    }
    forest_source = (
        f"union of BD Forêt V2 (WFS {BDFORET_V2_LAYER}, retrieved "
        f"{_retrieved(bdforet_path)}) and IGN Masque Forêt (WFS {MASQUE_FORET_LAYER}, "
        f"vintages {vintages}, retrieved {_retrieved(masque_path)})"
    )
    sufosat_files = f"{cfg.sufosat.dates.filename} md5 {cfg.sufosat.dates.md5}"
    outputs = {
        "eval_area": write_geotiff(
            eval_area.astype(np.uint8),
            grid,
            out / "eval_area.tif",
            tags=common | {"meaning": "1 = window (evaluated), 0 = margin"},
        ),
        "forest_mask": write_geotiff(
            layers.forest_mask.astype(np.uint8),
            grid,
            out / "forest_mask.tif",
            tags=common | {"source": forest_source, "meaning": "1 = forest"},
        ),
        "forest_type": write_geotiff(
            layers.forest_type,
            grid,
            out / "forest_type.tif",
            tags=common
            | {
                "source": forest_source + "; species from BD Forêt V2 `essence`",
                "codes": json.dumps(FOREST_TYPE_NAMES, ensure_ascii=False),
                "groups": json.dumps(FOREST_GROUPS),
            },
        ),
        "sufosat_date": write_geotiff(
            layers.sufosat_date,
            grid,
            out / "sufosat_date.tif",
            nodata=NO_CUT,
            tags=common
            | {
                "source": f"{SUFOSAT_SOURCE}; {sufosat_files}",
                "meaning": "clear-cut date, days since 1970-01-01; 0 = no clear-cut",
            },
        ),
        "sufosat_prob": write_geotiff(
            layers.sufosat_prob,
            grid,
            out / "sufosat_prob.tif",
            nodata=0,
            tags=common
            | {
                "source": f"{SUFOSAT_SOURCE}; {cfg.sufosat.prob.filename}",
                "meaning": "clear-cut probability, %; 0 = no clear-cut",
            },
        ),
        "summary": out / "summary.csv",
        "map": out / "reference_map.html",
    }
    summary.round(2).to_csv(outputs["summary"], index=False)
    _write_map(cfg, name, layers, coverage, outputs["map"], bounds)
    return WindowReference(name, summary, coverage, outputs)


def _write_map(
    cfg: ZoneConfig,
    name: str,
    layers: ReferenceLayers,
    coverage: dict[int, float],
    path: Path,
    bounds: Bbox,
) -> None:
    from sylvigie.viz.webmap import write_reference_webmap, year_color

    groups = list(FOREST_GROUPS)
    group_codes = np.zeros(256, dtype=np.uint8)
    for i, group in enumerate(groups, start=1):
        group_codes[list(FOREST_GROUPS[group])] = i
    forest_display = group_codes[layers.forest_type]

    years = cfg.period.years
    cut_year = days_to_year(layers.sufosat_date)
    cuts_display = np.zeros(layers.grid.shape, dtype=np.uint8)
    for i, year in enumerate(years, start=1):
        cuts_display[cut_year == year] = i
    outside = np.where((cuts_display > 0) & ~layers.forest_mask, 1, 0).astype(np.uint8)

    rasters = [
        {
            "label": "Forest type",
            "codes": forest_display,
            "colors": {i: GROUP_COLORS[g] for i, g in enumerate(groups, start=1)},
            "names": dict(enumerate(groups, start=1)),
            "opacity": 0.45,
            "visible": True,
        },
        {
            "label": "SUFOSAT clear-cuts",
            "codes": cuts_display,
            "colors": {i: year_color(y) for i, y in enumerate(years, start=1)},
            "names": {i: str(y) for i, y in enumerate(years, start=1)},
            "opacity": 0.85,
            "visible": True,
        },
        {
            "label": "Clear-cuts outside forest mask",
            "codes": outside,
            "colors": {1: OUTSIDE_MASK_COLOR},
            "names": {1: f"{years[0]}-{years[-1]}"},
            "opacity": 1.0,
            "visible": True,
        },
    ]
    for raster in rasters:
        raster["grid"] = layers.grid
    crs = layers.grid.crs
    write_reference_webmap(
        title=f"{cfg.name.capitalize()} {name} window: reference layers",
        rasters=rasters,
        extent=gpd.GeoDataFrame(geometry=[box(*layers.grid.bounds)], crs=crs),
        eval_area=gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs),
        coverage=coverage,
        path=path,
    )
