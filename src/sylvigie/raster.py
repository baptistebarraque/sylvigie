"""Thin rasterio wrappers: read onto a grid, rasterise vectors, write GeoTIFF."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features, shutil
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject

from sylvigie.aoi import Grid


def read_on_grid(
    path: Path, grid: Grid, resampling: Resampling = Resampling.nearest
) -> np.ndarray:
    """Read band 1 of a raster reprojected onto `grid`.

    The source CRS is read from the file. Only the source blocks overlapping
    the grid are read. Pixels outside the source get its nodata value.
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1)


def reproject_array(
    array: np.ndarray,
    grid: Grid,
    dst_crs: str,
    resampling: Resampling = Resampling.nearest,
    fill: int | float = 0,
) -> tuple[np.ndarray, Grid]:
    """Reproject an in-memory array to `dst_crs` at about the same resolution.

    Every source value is data. Destination pixels outside the source get
    `fill`. No nodata value is declared: GDAL would otherwise silently shift
    source pixels equal to it (e.g. 0 becomes 1) to keep them distinct.
    """
    transform, width, height = calculate_default_transform(
        grid.crs, dst_crs, grid.width, grid.height, *grid.bounds
    )
    dst_grid = Grid(CRS.from_user_input(dst_crs), transform, width, height)
    out = np.full(dst_grid.shape, fill, dtype=array.dtype)
    reproject(
        array,
        out,
        src_transform=grid.transform,
        src_crs=grid.crs,
        dst_transform=transform,
        dst_crs=dst_grid.crs,
        resampling=resampling,
        src_nodata=None,
        dst_nodata=None,
        init_dest_nodata=False,  # keep `fill` where the source does not reach
    )
    return out, dst_grid


def rasterize_on_grid(
    gdf: gpd.GeoDataFrame,
    grid: Grid,
    values: str | None = None,
    fill: int = 0,
    dtype: str = "uint8",
) -> np.ndarray:
    """Burn polygons onto `grid`, taking a pixel when its centre is inside.

    Burns the `values` column, or 1 when `values` is None.
    """
    gdf = gdf.to_crs(grid.crs)
    if values is None:
        shapes = ((geom, 1) for geom in gdf.geometry)
    else:
        shapes = zip(gdf.geometry, gdf[values], strict=True)
    return features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=fill,
        dtype=dtype,
    )


def write_geotiff(
    array: np.ndarray,
    grid: Grid,
    path: Path,
    nodata: float | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a single-band Cloud-Optimized GeoTIFF (deflate compression).

    `tags` are stored as GDAL metadata, e.g. sources and their versions.
    """
    if array.shape != grid.shape:
        raise ValueError(f"array shape {array.shape} != grid shape {grid.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": array.dtype.name,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "nodata": nodata,
    }
    # The COG driver can only copy an existing dataset, so build it in memory first.
    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(array, 1)
            if tags:
                tmp.update_tags(**tags)
        with mem.open() as tmp:
            shutil.copy(tmp, path, driver="COG", compress="DEFLATE")
    return path
