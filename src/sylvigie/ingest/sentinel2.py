"""Sentinel-2 L2A time series of a pilot window, step by step and resumable.

1. plan: search the catalogue, drop duplicate items, group them by solar day;
   the day list is saved as a manifest (CSV) with one status per day.
2. screen: load only the scene classification (SCL, 20 m) of each day and
   measure the share of the window's forest pixels that are clear; days
   under `min_clear_fraction` are dropped before downloading any band.

Every step updates the manifest as it goes, so an interrupted run picks up
where it stopped.
"""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac
from rasterio.warp import transform_bounds

from sylvigie.aoi import Grid
from sylvigie.config import ZoneConfig
from sylvigie.ingest.stac import search_s2
from sylvigie.preprocess.s2 import (
    clear_fraction,
    day_table,
    deduplicate,
    item_table,
    valid_mask,
)
from sylvigie.raster import read_on_grid
from sylvigie.reference import window_grid

log = logging.getLogger(__name__)

# Planetary Computer does not declare data types or nodata for these assets:
# without this, odc-stac loads floats and treats 0 (no data) as a value.
STAC_CFG = {
    "sentinel-2-l2a": {
        "assets": {
            "*": {"data_type": "uint16", "nodata": 0},
            "SCL": {"data_type": "uint8", "nodata": 0},
        }
    }
}
LOAD_ATTEMPTS = 3
LOAD_FIRST_WAIT_S = 10.0
SCREEN_WORKERS = 4

# Manifest statuses
PLANNED, KEPT, DROPPED, FAILED = "planned", "kept", "dropped", "failed"

_sleep = time.sleep  # replaced in tests


@dataclass(frozen=True)
class WindowPaths:
    raw: Path  # manifest and item list
    manifest: Path
    items: Path
    forest_mask: Path  # from `sylvigie grid` (step 2b)


def window_paths(
    cfg: ZoneConfig, window: str, data_dir: Path, subset_km: float | None
) -> WindowPaths:
    name = window if subset_km is None else f"{window}_sub{subset_km:g}km"
    raw = data_dir / "raw" / "s2" / cfg.name / name
    return WindowPaths(
        raw=raw,
        manifest=raw / "manifest.csv",
        items=raw / "items.json",
        forest_mask=data_dir / "interim" / cfg.name / window / "forest_mask.tif",
    )


def ingest_grid(cfg: ZoneConfig, window: str, subset_km: float | None) -> Grid:
    """The window's reference grid, or a centred square of `subset_km` for tests.

    The subset keeps edges on multiples of 20 m, so its pixels coincide with
    the window grid's and with the Sentinel-2 10 m and 20 m pixels.
    """
    if cfg.windows is None:
        raise ValueError("the config has no windows: run suggest-zone first")
    bounds = getattr(cfg.windows, window)
    grid, _ = window_grid(bounds, cfg.grid.margin_m, cfg.crs, cfg.resolution)
    if subset_km is None:
        return grid
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    half = subset_km * 1000 / 2
    return Grid.from_bounds(
        (cx - half, cy - half, cx + half, cy + half),
        cfg.crs,
        cfg.resolution,
        anchor=2 * cfg.resolution,
    )


def save_manifest(days: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    days.to_csv(tmp, index=False)
    tmp.replace(path)  # never leave a half-written manifest


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        parse_dates=["datetime"],
        dtype={
            column: str
            for column in [
                "items",
                "tiles",
                "orbits",
                "platforms",
                "baselines",
                "offsets",
            ]
        },
        converters={"day": lambda v: pd.Timestamp(v).date()},
    )


def plan(cfg: ZoneConfig, grid: Grid, paths: WindowPaths) -> pd.DataFrame:
    """Day list of the window, from the saved manifest or a new search."""
    if paths.manifest.exists():
        log.info("Resuming from %s", paths.manifest)
        return load_manifest(paths.manifest)
    lonlat = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    s2 = cfg.s2
    items = search_s2(
        lonlat,
        cfg.period.load_from,
        cfg.period.load_until,
        s2.max_tile_cloud,
        s2.collection,
    )
    table = deduplicate(item_table(items))
    log.info("%d items found, %d after deduplication", len(items), len(table))
    kept_ids = set(table["id"])
    paths.raw.mkdir(parents=True, exist_ok=True)
    pystac.ItemCollection([i for i in items if i.id in kept_ids]).save_object(
        str(paths.items)
    )
    days = day_table(table).assign(clear_fraction=np.nan, status=PLANNED)
    save_manifest(days, paths.manifest)
    return days


def load_items(paths: WindowPaths) -> dict[str, pystac.Item]:
    collection = pystac.ItemCollection.from_file(str(paths.items))
    return {item.id: item for item in collection}


def with_retries(load: Callable[[], object], what: str) -> object:
    """Run `load`, retrying with a doubling wait: remote reads fail now and then."""
    wait = LOAD_FIRST_WAIT_S
    for attempt in range(1, LOAD_ATTEMPTS + 1):
        try:
            return load()
        except Exception as error:  # rasterio/odc raise many error types
            if attempt == LOAD_ATTEMPTS:
                raise
            log.warning(
                "%s: attempt %d failed (%r), retry in %.0f s",
                what,
                attempt,
                error,
                wait,
            )
            _sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")


def load_scl(items: list[pystac.Item], grid: Grid, collection: str) -> np.ndarray:
    """Scene classification of one solar day on `grid` (nearest neighbour)."""
    ds = odc.stac.load(
        items,
        bands=["SCL"],
        geobox=grid.to_geobox(),
        groupby="solar_day",
        resampling="nearest",
        patch_url=planetary_computer.sign,
        stac_cfg={collection: STAC_CFG["sentinel-2-l2a"]},
    )
    return ds["SCL"].isel(time=0).values


def screen(
    cfg: ZoneConfig,
    grid: Grid,
    days: pd.DataFrame,
    paths: WindowPaths,
    items: dict[str, pystac.Item],
    load: Callable[[list[pystac.Item], Grid, str], np.ndarray] = load_scl,
) -> pd.DataFrame:
    """Measure each planned day's clear fraction over forest; keep or drop it.

    Works at 20 m, the native SCL resolution, so screening costs about a
    quarter of a 10 m band. Failed days are retried on the next run.
    """
    if not paths.forest_mask.exists():
        raise FileNotFoundError(
            f"{paths.forest_mask} missing: run `sylvigie grid` first"
        )
    grid20 = grid.coarsen(2)
    forest20 = read_on_grid(paths.forest_mask, grid20).astype(bool)
    # Same ground distance as the 10 m dilation.
    dilation20 = max(1, cfg.s2.mask_dilation_px // 2)
    todo = days.index[days["status"].isin([PLANNED, FAILED])]
    log.info("Screening %d days on %d forest pixels at 20 m", len(todo), forest20.sum())

    def screen_day(i: int) -> tuple[int, float]:
        day_items = [items[item_id] for item_id in days.at[i, "items"].split(";")]
        scl = with_retries(
            lambda: load(day_items, grid20, cfg.s2.collection),
            f"SCL {days.at[i, 'day']}",
        )
        valid = valid_mask(scl, cfg.s2.scl_valid, dilation20)
        return i, clear_fraction(valid, forest20)

    days = days.copy()
    done = 0
    with ThreadPoolExecutor(max_workers=SCREEN_WORKERS) as pool:
        futures = [pool.submit(screen_day, i) for i in todo]
        for future in as_completed(futures):
            try:
                i, fraction = future.result()
            except Exception as error:
                log.error("Screening failed: %r", error)
                continue
            days.at[i, "clear_fraction"] = fraction
            days.at[i, "status"] = (
                KEPT if fraction >= cfg.s2.min_clear_fraction else DROPPED
            )
            done += 1
            if done % 20 == 0:
                save_manifest(days, paths.manifest)
                log.info("Screened %d / %d days", done, len(todo))
    failed = days.index.isin(todo) & days["clear_fraction"].isna()
    days.loc[failed, "status"] = FAILED
    save_manifest(days, paths.manifest)
    return days


@dataclass(frozen=True)
class IngestReport:
    window: str
    grid: Grid
    days: pd.DataFrame
    paths: WindowPaths


def build_s2_cube(
    cfg: ZoneConfig, window: str, data_dir: Path, subset_km: float | None = None
) -> IngestReport:
    """Plan and screen the window's Sentinel-2 days (download comes in 3b)."""
    grid = ingest_grid(cfg, window, subset_km)
    paths = window_paths(cfg, window, data_dir, subset_km)
    log.info(
        "Window %s: grid %d x %d px, %s", window, grid.width, grid.height, grid.bounds
    )
    days = plan(cfg, grid, paths)
    days = screen(cfg, grid, days, paths, load_items(paths))
    return IngestReport(window, grid, days, paths)
