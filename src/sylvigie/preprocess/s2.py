"""Sentinel-2 L2A rules: deduplication, solar days, harmonisation, cloud mask.

Pure functions on tables and arrays, so they are tested without network.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pystac
from scipy import ndimage

# Reflectance is stored as int16 = reflectance x 10 000, like the L2A product.
NODATA = -32768
REFLECTANCE_SCALE = 10_000
# From processing baseline 04.00 (January 2022), L2A digital numbers carry a
# +1000 offset (BOA_ADD_OFFSET) so that dark pixels can go below zero.
# Planetary Computer serves the raw numbers: remove it item by item.
BOA_OFFSET = 1000
OFFSET_SINCE_BASELINE = (4, 0)


def parse_baseline(baseline: str) -> tuple[int, int]:
    """'05.10' -> (5, 10); compares correctly, unlike the string ('5.9' > '05.10')."""
    major, minor = baseline.split(".")
    return int(major), int(minor)


def boa_offset(baseline: str) -> int:
    return BOA_OFFSET if parse_baseline(baseline) >= OFFSET_SINCE_BASELINE else 0


def item_table(items: Iterable[pystac.Item]) -> pd.DataFrame:
    """One row per STAC item with the properties the ingestion needs."""
    rows = []
    for item in items:
        props = item.properties
        west, _, east, _ = item.bbox
        rows.append(
            {
                "id": item.id,
                "datetime": pd.Timestamp(item.datetime),
                "tile": props["s2:mgrs_tile"],
                "orbit": props["sat:relative_orbit"],
                "platform": props["platform"],
                "baseline": props["s2:processing_baseline"],
                "tile_cloud": props["eo:cloud_cover"],
                "lon": (west + east) / 2,
            }
        )
    return pd.DataFrame(rows)


def deduplicate(table: pd.DataFrame) -> pd.DataFrame:
    """Keep one item per (tile, acquisition time): the latest processing baseline.

    Planetary Computer lists ESA reprocessings next to the original products.
    """
    version = table["baseline"].map(parse_baseline)
    order = table.assign(_version=version).sort_values(["tile", "datetime", "_version"])
    kept = order.groupby(["tile", "datetime"], sort=False).tail(1)
    return kept.drop(columns="_version").sort_values("datetime", ignore_index=True)


def solar_day(table: pd.DataFrame) -> pd.Series:
    """Local solar date of each item: UTC time shifted by longitude / 15 h.

    Same rule as odc-stac's `groupby="solar_day"`: images of neighbouring
    tiles taken during one overpass fall on the same day.
    """
    shift = pd.to_timedelta(table["lon"] / 15, unit="h")
    return (table["datetime"] + shift).dt.date


def day_table(items: pd.DataFrame) -> pd.DataFrame:
    """One row per solar day, listing the items to fuse for that day."""
    items = items.assign(day=solar_day(items), offset=items["baseline"].map(boa_offset))

    def join(values: pd.Series) -> str:
        return ";".join(str(v) for v in dict.fromkeys(values))

    return (
        items.groupby("day", sort=True)
        .agg(
            datetime=("datetime", "min"),
            items=("id", join),
            tiles=("tile", join),
            orbits=("orbit", join),
            platforms=("platform", join),
            baselines=("baseline", join),
            offsets=("offset", join),
            tile_cloud=("tile_cloud", "min"),
        )
        .reset_index()
    )


def harmonise(digital_numbers: np.ndarray, offset: int) -> np.ndarray:
    """L2A digital numbers (0 or NaN = no data) to int16 reflectance x 10 000."""
    dn = np.asarray(digital_numbers, dtype=np.float64)
    valid = np.isfinite(dn) & (dn > 0)
    out = np.full(dn.shape, NODATA, dtype=np.int16)
    # Values can go slightly negative after removing the offset (dark pixels,
    # noise): kept as is, as ESA recommends.
    out[valid] = np.clip(
        np.rint(dn[valid]) - offset, NODATA + 1, np.iinfo(np.int16).max
    )
    return out


def valid_mask(scl: np.ndarray, keep: Iterable[int], dilation_px: int) -> np.ndarray:
    """Clear pixels from the scene classification, cloud edges removed.

    Invalid pixels (clouds, shadows, snow, no data...) are grown by
    `dilation_px`: the SCL misses the thin, bright edges of clouds.
    """
    invalid = ~np.isin(scl, list(keep))
    if dilation_px > 0:
        structure = np.ones((3, 3), dtype=bool)
        invalid = ndimage.binary_dilation(invalid, structure, iterations=dilation_px)
    return ~invalid


def clear_fraction(valid: np.ndarray, forest: np.ndarray) -> float:
    """Share of forest pixels that are clear (0 when there is no forest)."""
    n_forest = int(forest.sum())
    return float((valid & forest).sum() / n_forest) if n_forest else 0.0
