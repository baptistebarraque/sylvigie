import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sylvigie.preprocess.indices import (
    clear_mask,
    compute_index,
    crswir,
    cube_grid,
    reflectance,
)

BANDS = ["B04", "B05", "B08", "B8A", "B11", "B12"]


def _cube(values, scl=4, shape=(2, 6, 6)):
    """Decoded cube (float32, NaN = no data) with constant band values."""
    t, h, w = shape
    coords = {
        "time": pd.date_range("2023-06-01", periods=t, freq="5D"),
        "y": 5230005.0 - 10 * np.arange(h),
        "x": 580005.0 + 10 * np.arange(w),
    }
    data = {
        band: (("time", "y", "x"), np.full(shape, values.get(band, 1000), np.float32))
        for band in BANDS
    }
    data["SCL"] = (("time", "y", "x"), np.full(shape, scl, np.float32))
    return xr.Dataset(data, coords=coords).rio.write_crs("EPSG:32631")


def test_ndvi_and_nbr_values():
    cube = _cube({"B04": 300, "B08": 3000, "B12": 1000})

    ndvi = compute_index(cube, "NDVI", [4, 5, 6], 0)
    nbr = compute_index(cube, "NBR", [4, 5, 6], 0)

    assert float(ndvi[0, 0, 0]) == pytest.approx((0.3 - 0.03) / (0.3 + 0.03))
    assert float(nbr[0, 0, 0]) == pytest.approx(0.5)
    assert ndvi.dtype == np.float32


def test_crswir_matches_fordead_formula():
    value = crswir(xr.DataArray(0.3), xr.DataArray(0.2), xr.DataArray(0.1))

    # B11 / (B8A + (B12 - B8A) * (1610.4 - 864) / (2185.7 - 864))
    assert float(value) == pytest.approx(0.2 / (0.3 - 0.2 * 746.4 / 1321.7))


def test_negative_reflectance_is_clipped():
    cube = _cube({"B04": -150, "B08": 2000})

    assert float(reflectance(cube, "B04").min()) == 0
    assert float(compute_index(cube, "NDVI", [4], 0).max()) == pytest.approx(1.0)


def test_cloudy_and_missing_pixels_are_nan():
    cube = _cube({"B08": 3000})
    cube["SCL"][0, 2, 2] = 9  # cloud
    cube["B08"][1, 0, 0] = np.nan  # no data

    ndvi = compute_index(cube, "NDVI", [4, 5, 6], dilation_px=1)

    assert np.isnan(ndvi[0, 1:4, 1:4]).all()  # cloud and its 8 neighbours
    assert int(np.isnan(ndvi[0]).sum()) == 9
    assert np.isnan(ndvi[1, 0, 0])
    assert int(np.isnan(ndvi[1]).sum()) == 1  # dilation never crosses dates


def test_clear_mask_is_the_same_lazily_and_eagerly():
    rng = np.random.default_rng(0)
    scl = xr.DataArray(
        rng.choice([4, 9], size=(3, 20, 20), p=[0.9, 0.1]), dims=("time", "y", "x")
    )

    eager = clear_mask(scl, [4], 2)
    lazy = clear_mask(scl.chunk({"time": -1, "y": 7, "x": 7}), [4], 2).compute()

    np.testing.assert_array_equal(eager.values, lazy.values)  # block edges exact


def test_cube_grid_recovers_the_pixel_grid():
    grid = cube_grid(_cube({}))

    assert grid.bounds == (580000.0, 5229950.0, 580060.0, 5230010.0)
    assert grid.resolution == 10
    assert grid.crs.to_epsg() == 32631
