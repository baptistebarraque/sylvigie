"""Sentinel-2 L2A time series of a pilot window, step by step and resumable.

1. plan: search the catalogue, drop duplicate items, group them by solar day;
   the day list is saved as a manifest (CSV) with one status per day.
2. screen: load only the scene classification (SCL, 20 m) of each day and
   measure the share of the window's forest pixels that are clear; days
   under `min_clear_fraction` are dropped before downloading any band.
3. download: load the bands and SCL of each kept day onto the window grid,
   harmonise each item (processing-baseline offset), fuse the day's items
   and write the day to its own small Zarr store ("staging").
4. build: gather the day stores into one cube whose chunks hold the whole
   time series of 256 x 256 pixel blocks, the layout per-pixel time-series
   methods read fastest.

Every step records its progress (manifest statuses, one store per finished
day), so an interrupted run picks up where it stopped. The cloud mask is not
applied here: SCL is stored and the mask is applied when the cube is read.
"""

import datetime as dt
import logging
import shutil
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
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.warp import transform_bounds
from zarr.codecs import ZstdCodec

from sylvigie import __version__
from sylvigie.aoi import Grid
from sylvigie.config import ZoneConfig
from sylvigie.ingest.stac import search_s2
from sylvigie.preprocess.s2 import (
    NODATA,
    REFLECTANCE_SCALE,
    boa_offset,
    clear_fraction,
    day_table,
    deduplicate,
    harmonise,
    item_table,
    valid_mask,
)
from sylvigie.raster import read_on_grid
from sylvigie.reference import window_grid

log = logging.getLogger(__name__)

# Planetary Computer does not declare data types or nodata for these assets:
# without this, odc-stac loads floats and treats 0 (no data) as a value.
STAC_CFG = {
    "assets": {
        "*": {"data_type": "uint16", "nodata": 0},
        "SCL": {"data_type": "uint8", "nodata": 0},
    }
}
SCL_NODATA = 0
LOAD_ATTEMPTS = 3
LOAD_FIRST_WAIT_S = 10.0
SCREEN_WORKERS = 4
DOWNLOAD_WORKERS = 3
BLOCK = 256  # rows per inner chunk of a day store; cube chunks are BLOCK x BLOCK
_COMPRESSORS = [ZstdCodec(level=3)]

# Manifest statuses
PLANNED, KEPT, DROPPED, FAILED, WRITTEN = (
    "planned",
    "kept",
    "dropped",
    "failed",
    "written",
)

_sleep = time.sleep  # replaced in tests


@dataclass(frozen=True)
class WindowPaths:
    raw: Path  # manifest, item list and day stores
    manifest: Path
    items: Path
    days: Path  # one Zarr store per downloaded day (staging)
    cube: Path
    forest_mask: Path  # from `sylvigie grid` (step 2b)

    def day_store(self, day: dt.date) -> Path:
        return self.days / f"{day.isoformat()}.zarr"


def window_paths(
    cfg: ZoneConfig, window: str, data_dir: Path, subset_km: float | None
) -> WindowPaths:
    name = window if subset_km is None else f"{window}_sub{subset_km:g}km"
    raw = data_dir / "raw" / "s2" / cfg.name / name
    return WindowPaths(
        raw=raw,
        manifest=raw / "manifest.csv",
        items=raw / "items.json",
        days=raw / "days",
        cube=data_dir / "interim" / cfg.name / name / "s2_cube.zarr",
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


# --- 1. plan ---------------------------------------------------------------


def save_manifest(days: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    days.to_csv(tmp, index=False)
    tmp.replace(path)  # never leave a half-written manifest


def load_manifest(path: Path) -> pd.DataFrame:
    text_columns = ["items", "tiles", "orbits", "platforms", "baselines", "offsets"]
    return pd.read_csv(
        path,
        parse_dates=["datetime"],
        dtype=dict.fromkeys(text_columns, str),
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


def _run_days(
    days: pd.DataFrame,
    todo: pd.Index,
    work: Callable[[int], object],
    on_result: Callable[[int, object], None],
    workers: int,
    manifest: Path,
    what: str,
) -> pd.DataFrame:
    """Run `work` on each day of `todo` in threads; save the manifest as it goes."""
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, i): i for i in todo}
        for future in as_completed(futures):
            i = futures[future]
            try:
                on_result(i, future.result())
            except Exception as error:
                log.error("%s %s failed: %r", what, days.at[i, "day"], error)
                continue
            done += 1
            if done % 20 == 0:
                save_manifest(days, manifest)
                log.info("%s: %d / %d days", what, done, len(todo))
    save_manifest(days, manifest)
    return days


# --- 2. screen -------------------------------------------------------------


def load_scl(items: list[pystac.Item], grid: Grid, collection: str) -> np.ndarray:
    """Scene classification of one solar day on `grid` (nearest neighbour)."""
    ds = odc.stac.load(
        items,
        bands=["SCL"],
        geobox=grid.to_geobox(),
        groupby="solar_day",
        resampling="nearest",
        patch_url=planetary_computer.sign,
        stac_cfg={collection: STAC_CFG},
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
    days = days.copy()
    todo = days.index[days["status"].isin([PLANNED, FAILED])]
    days.loc[todo, "status"] = FAILED  # until proven otherwise
    log.info("Screening %d days on %d forest pixels at 20 m", len(todo), forest20.sum())

    def work(i: int) -> float:
        day_items = [items[item_id] for item_id in days.at[i, "items"].split(";")]
        scl = with_retries(
            lambda: load(day_items, grid20, cfg.s2.collection),
            f"SCL {days.at[i, 'day']}",
        )
        return clear_fraction(valid_mask(scl, cfg.s2.scl_valid, dilation20), forest20)

    def on_result(i: int, fraction: float) -> None:
        days.at[i, "clear_fraction"] = fraction
        days.at[i, "status"] = (
            KEPT if fraction >= cfg.s2.min_clear_fraction else DROPPED
        )

    return _run_days(
        days, todo, work, on_result, SCREEN_WORKERS, paths.manifest, "Screen"
    )


# --- 3. download -----------------------------------------------------------


def load_item(item: pystac.Item, grid: Grid, bands: list[str], collection: str) -> dict:
    """Digital numbers of one item on `grid`: {band: uint16, "SCL": uint8}.

    20 m bands are upsampled bilinearly; SCL, a class map, by nearest neighbour.
    """
    ds = odc.stac.load(
        [item],
        bands=[*bands, "SCL"],
        geobox=grid.to_geobox(),
        groupby="id",
        resampling={**dict.fromkeys(bands, "bilinear"), "SCL": "nearest"},
        patch_url=planetary_computer.sign,
        stac_cfg={collection: STAC_CFG},
    )
    return {name: ds[name].isel(time=0).values for name in [*bands, "SCL"]}


def fuse(layers: list[np.ndarray], nodata: int) -> np.ndarray:
    """First valid value across layers (items of one day, e.g. adjacent tiles)."""
    out = layers[0].copy()
    for layer in layers[1:]:
        missing = out == nodata
        out[missing] = layer[missing]
    return out


def day_dataset(
    data: dict[str, np.ndarray], grid: Grid, when: pd.Timestamp
) -> xr.Dataset:
    """One day as a (time=1, y, x) dataset with pixel-centre coordinates and CRS."""
    x0, y0 = grid.bounds[0], grid.bounds[3]
    res = grid.resolution
    coords = {
        "time": [when],
        "y": y0 - (np.arange(grid.height) + 0.5) * res,
        "x": x0 + (np.arange(grid.width) + 0.5) * res,
    }
    ds = xr.Dataset(
        {name: (("time", "y", "x"), array[None]) for name, array in data.items()},
        coords=coords,
    )
    return ds.rio.write_crs(grid.crs)


def _encoding(ds: xr.Dataset, chunks: dict[str, tuple[int, ...]]) -> dict:
    encoding = {}
    for name in ds.data_vars:
        fill = SCL_NODATA if name == "SCL" else NODATA
        encoding[name] = {"_FillValue": fill, "compressors": _COMPRESSORS} | chunks
    return encoding


def write_day(ds: xr.Dataset, path: Path) -> Path:
    """Write a day store atomically: a half-written day never looks finished.

    One shard (file) per band holds the whole image, cut inside into strips
    of BLOCK rows that the cube builder reads one at a time.
    """
    height, width = ds.sizes["y"], ds.sizes["x"]
    strip = min(BLOCK, height)
    # A shard must hold whole inner chunks; the last shard may overhang the array.
    shard_height = -(-height // strip) * strip
    chunks = {"chunks": (1, strip, width), "shards": (1, shard_height, width)}
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(
        tmp, mode="w", zarr_format=3, consolidated=False, encoding=_encoding(ds, chunks)
    )
    tmp.replace(path)
    return path


def download_day(
    day: pd.Series,
    items: dict[str, pystac.Item],
    grid: Grid,
    bands: list[str],
    collection: str,
    load: Callable[[pystac.Item, Grid, list[str], str], dict] = load_item,
) -> xr.Dataset:
    """Load, harmonise and fuse the items of one solar day."""
    day_items = sorted(
        (items[item_id] for item_id in day["items"].split(";")),
        key=lambda item: item.properties["eo:cloud_cover"],  # clearest tile first
    )
    layers: dict[str, list[np.ndarray]] = {name: [] for name in [*bands, "SCL"]}
    for item in day_items:
        raw = with_retries(
            lambda item=item: load(item, grid, bands, collection), item.id
        )
        offset = boa_offset(item.properties["s2:processing_baseline"])
        for band in bands:
            layers[band].append(harmonise(raw[band], offset))
        layers["SCL"].append(raw["SCL"].astype(np.uint8))
    data = {band: fuse(layers[band], NODATA) for band in bands}
    data["SCL"] = fuse(layers["SCL"], SCL_NODATA)
    when = pd.Timestamp(day["datetime"])
    if when.tzinfo is not None:
        when = when.tz_convert(None)  # naive UTC, as xarray stores times
    return day_dataset(data, grid, when)


def download(
    cfg: ZoneConfig,
    grid: Grid,
    days: pd.DataFrame,
    paths: WindowPaths,
    items: dict[str, pystac.Item],
    load: Callable[[pystac.Item, Grid, list[str], str], dict] = load_item,
) -> pd.DataFrame:
    """Download every kept day that has no store yet."""
    days = days.copy()
    stored = days["day"].map(lambda d: paths.day_store(d).exists())
    days.loc[(days["status"] == KEPT) & stored, "status"] = WRITTEN
    days.loc[(days["status"] == WRITTEN) & ~stored, "status"] = KEPT
    todo = days.index[days["status"] == KEPT]
    log.info("Downloading %d days (%d already stored)", len(todo), int(stored.sum()))

    def work(i: int) -> Path:
        ds = download_day(
            days.loc[i], items, grid, cfg.s2.bands, cfg.s2.collection, load
        )
        return write_day(ds, paths.day_store(days.at[i, "day"]))

    def on_result(i: int, _: Path) -> None:
        days.at[i, "status"] = WRITTEN

    return _run_days(
        days, todo, work, on_result, DOWNLOAD_WORKERS, paths.manifest, "Download"
    )


# --- 4. build the time-series cube -------------------------------------------


def build_cube(days: pd.DataFrame, paths: WindowPaths, attrs: dict[str, str]) -> Path:
    """Gather the day stores into one cube chunked (time=all, BLOCK, BLOCK).

    Done one band and one strip of BLOCK rows at a time, so memory stays at
    about n_days x BLOCK x width values whatever the window size.
    """
    written = days[days["status"] == WRITTEN].sort_values("datetime")
    if written.empty:
        raise ValueError("no downloaded day to build a cube from")
    stores = [
        xr.open_zarr(paths.day_store(day), consolidated=False, mask_and_scale=False)
        for day in written["day"]
    ]
    # Only variables that already have a time axis are stacked; the CRS
    # (`spatial_ref`) is the same in every store and is taken from the first.
    combined = xr.concat(
        stores, dim="time", data_vars="minimal", coords="minimal", compat="override"
    )
    if "spatial_ref" in combined.data_vars:
        combined = combined.set_coords("spatial_ref")
    per_day = {
        "solar_day": written["day"].astype(str).to_numpy(dtype=str),
        "orbit": written["orbits"].to_numpy(dtype=str),
        "platform": written["platforms"].to_numpy(dtype=str),
        "baseline": written["baselines"].to_numpy(dtype=str),
        "clear_fraction": written["clear_fraction"].to_numpy(dtype=np.float32),
    }
    combined = combined.assign_coords({k: ("time", v) for k, v in per_day.items()})
    combined.attrs = attrs
    for name in combined.data_vars:  # the fill value goes in the encoding instead
        combined[name].attrs.pop("_FillValue", None)
    n_time, height, width = (combined.sizes[d] for d in ("time", "y", "x"))
    chunks = {"chunks": (n_time, min(BLOCK, height), min(BLOCK, width))}

    tmp = paths.cube.with_name(paths.cube.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # Chunk the data variables only: coordinates stay in memory, so this call
    # writes them now while the bands are filled strip by strip below.
    template = combined.copy()
    for name in combined.data_vars:
        template[name] = combined[name].chunk({"time": -1, "y": BLOCK, "x": BLOCK})
    template.to_zarr(
        tmp,
        mode="w",
        zarr_format=3,
        consolidated=False,
        compute=False,
        encoding=_encoding(combined, chunks),
    )
    for name in combined.data_vars:
        for row in range(0, height, BLOCK):
            rows = slice(row, min(row + BLOCK, height))
            strip = (
                combined[[name]]
                .isel(y=rows)
                .drop_vars([c for c in combined.coords if "y" not in combined[c].dims])
            )
            strip.load().to_zarr(tmp, region={"y": rows}, consolidated=False)
        log.info("Cube: %s written", name)
    shutil.rmtree(paths.cube, ignore_errors=True)
    tmp.replace(paths.cube)
    return paths.cube


def cube_is_current(days: pd.DataFrame, paths: WindowPaths) -> bool:
    """True when the cube holds exactly the downloaded days."""
    if not paths.cube.exists():
        return False
    cube = xr.open_zarr(paths.cube, consolidated=False)
    written = set(days.loc[days["status"] == WRITTEN, "day"].astype(str))
    return set(cube["solar_day"].values.astype(str)) == written


@dataclass(frozen=True)
class IngestReport:
    window: str
    grid: Grid
    days: pd.DataFrame
    paths: WindowPaths


def build_s2_cube(
    cfg: ZoneConfig,
    window: str,
    data_dir: Path,
    subset_km: float | None = None,
    keep_staging: bool = False,
) -> IngestReport:
    """Plan, screen, download and gather the window's Sentinel-2 time series."""
    grid = ingest_grid(cfg, window, subset_km)
    paths = window_paths(cfg, window, data_dir, subset_km)
    log.info(
        "Window %s: grid %d x %d px, %s", window, grid.width, grid.height, grid.bounds
    )
    days = plan(cfg, grid, paths)
    items = load_items(paths)
    days = screen(cfg, grid, days, paths, items)
    if cube_is_current(days, paths) and not (days["status"] == KEPT).any():
        log.info("Cube already up to date: %s", paths.cube)
        return IngestReport(window, grid, days, paths)
    days = download(cfg, grid, days, paths, items)
    pending = int(days["status"].isin([KEPT, FAILED]).sum())
    attrs = {
        "title": f"Sentinel-2 L2A time series, {cfg.name} {window}",
        "source": f"Microsoft Planetary Computer STAC, collection {cfg.s2.collection}",
        "reflectance": f"band value / {REFLECTANCE_SCALE}; nodata {NODATA}",
        "processing": "duplicates dropped (latest baseline); +1000 offset removed "
        "for baseline >= 04.00; 20 m bands bilinear to 10 m; SCL nearest; "
        "cloud mask not applied (use SCL)",
        "sylvigie_version": __version__,
        "created": dt.date.today().isoformat(),
    }
    build_cube(days, paths, attrs)
    if pending:
        log.warning("%d days still pending: re-run to complete the cube", pending)
    elif not keep_staging:
        shutil.rmtree(paths.days, ignore_errors=True)
        log.info("Day stores removed (use --keep-staging to keep them)")
    return IngestReport(window, grid, days, paths)
