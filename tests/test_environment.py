"""Smoke tests for the geospatial stack.

The PyPI wheels of rasterio and pyogrio bundle their own GDAL build. These
tests catch a broken install, or a bundled GDAL missing a driver the pipeline
relies on, before it surfaces deep inside a long processing run.
"""

import importlib

import pytest

CORE_MODULES = [
    "dask.array",
    "geopandas",
    "matplotlib",
    "odc.stac",
    "planetary_computer",
    "pydantic",
    "pystac_client",
    "rasterio",
    "rioxarray",
    "shapely",
    "xarray",
    "yaml",
    "zarr",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
    importlib.import_module(module)


def test_rasterio_can_write_geotiff():
    import rasterio

    # GTiff/COG carry the reference grid, forest mask and label rasters.
    with rasterio.Env() as env:
        assert {"GTiff", "COG"} <= set(env.drivers())


def test_pyogrio_can_write_geopackage():
    import pyogrio

    # GPKG is the output format for alert polygons (roadmap step 4).
    assert "w" in pyogrio.list_drivers().get("GPKG", "")
