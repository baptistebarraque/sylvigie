"""Spectral indices from the Sentinel-2 cube, cloud-masked when read.

Each index answers a forestry question:
- NDVI (B08, B04): how green and dense the canopy is.
- NBR (B08, B12), main index of the v0 detector: drops sharply when trees are
  removed, as near-infrared (leaves) falls and shortwave infrared (bare soil,
  dry slash) rises.
- NDMI (B08, B11): canopy water content.
- NDRE (B8A, B05): red-edge chlorophyll, reacts early to stress.
- CRSWIR (B8A, B11, B12), from fordead: B11 against the line joining B8A and
  B12 at B11's wavelength; rises as foliage dries, the early sign of bark
  beetle attacks on spruce.
"""

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor used by cube_grid)
import xarray as xr
from scipy import ndimage

from sylvigie.aoi import Grid
from sylvigie.preprocess.s2 import REFLECTANCE_SCALE

# Centre wavelengths (nm) used by fordead's CRSWIR (fordead.preprocess,
# DEFAULT_INDICES, checked 2026-11-21).
WAVELENGTH_NM = {"B8A": 864.0, "B11": 1610.4, "B12": 2185.7}
NORMALIZED_DIFFERENCES = {
    "NDVI": ("B08", "B04"),
    "NBR": ("B08", "B12"),
    "NDMI": ("B08", "B11"),
    "NDRE": ("B8A", "B05"),
}
INDICES = [*NORMALIZED_DIFFERENCES, "CRSWIR"]


def open_cube(path) -> xr.Dataset:
    """The cube with nodata decoded to NaN (bands become float32)."""
    return xr.open_zarr(path, consolidated=False)


def cube_grid(cube: xr.Dataset) -> Grid:
    """The pixel grid of a cube, from its pixel-centre coordinates and CRS."""
    res = float(cube.x[1] - cube.x[0])
    xmin = float(cube.x[0]) - res / 2
    ymax = float(cube.y[0]) + res / 2
    bounds = (xmin, ymax - cube.sizes["y"] * res, xmin + cube.sizes["x"] * res, ymax)
    return Grid.from_bounds(bounds, cube.rio.crs, res)


def _dilate(invalid: np.ndarray, iterations: int) -> np.ndarray:
    # Grow along y and x only, never across dates.
    structure = np.ones((1, 3, 3), dtype=bool)
    return ndimage.binary_dilation(invalid, structure, iterations=iterations)


def clear_mask(scl: xr.DataArray, keep: list[int], dilation_px: int) -> xr.DataArray:
    """True where the scene classification says clear, cloud edges removed.

    Works lazily on dask-backed cubes: each block is dilated with a halo of
    `dilation_px` borrowed from its neighbours, so block edges are exact.
    """
    invalid = ~scl.fillna(0).astype(np.uint8).isin(keep)
    if dilation_px == 0:
        return ~invalid
    data = invalid.data
    if hasattr(data, "map_overlap"):
        grown = data.map_overlap(
            _dilate,
            depth={0: 0, 1: dilation_px, 2: dilation_px},
            boundary=False,
            iterations=dilation_px,
            dtype=bool,
        )
    else:
        grown = _dilate(np.asarray(data), dilation_px)
    return ~invalid.copy(data=grown)


def reflectance(cube: xr.Dataset, band: str) -> xr.DataArray:
    """Reflectance 0-1. Negative values (dark winter conifers in the red band,
    sensor noise) are set to 0, otherwise NDVI could exceed 1."""
    return (cube[band].astype(np.float32) / REFLECTANCE_SCALE).clip(min=0)


def normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    return (a - b) / (a + b)


def crswir(b8a: xr.DataArray, b11: xr.DataArray, b12: xr.DataArray) -> xr.DataArray:
    """B11 over the B8A-B12 line interpolated at B11's wavelength (fordead)."""
    slope = (WAVELENGTH_NM["B11"] - WAVELENGTH_NM["B8A"]) / (
        WAVELENGTH_NM["B12"] - WAVELENGTH_NM["B8A"]
    )
    return b11 / (b8a + (b12 - b8a) * slope)


def compute_index(
    cube: xr.Dataset, name: str, scl_valid: list[int], dilation_px: int
) -> xr.DataArray:
    """One index over the whole cube, NaN where cloudy or without data."""
    if name in NORMALIZED_DIFFERENCES:
        a, b = NORMALIZED_DIFFERENCES[name]
        values = normalized_difference(reflectance(cube, a), reflectance(cube, b))
    elif name == "CRSWIR":
        values = crswir(*(reflectance(cube, band) for band in ("B8A", "B11", "B12")))
    else:
        raise ValueError(f"unknown index {name}; available: {INDICES}")
    clear = clear_mask(cube["SCL"], scl_valid, dilation_px)
    # Division by zero (all bands 0) gives inf: treat as missing.
    values = values.where(clear & np.isfinite(values))
    return values.astype(np.float32).rename(name)
