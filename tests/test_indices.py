import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sylvigie.preprocess.indices import (
    at_10m,
    clear_mask,
    compute_index,
    crop,
    crswir,
    cube_grid,
    reflectance,
)

FINE = ["B04", "B08"]
COARSE = ["B05", "B8A", "B11", "B12", "SCL"]


def _cube(values, scl=4, t=2, h=12, w=12):
    """Decoded cube (float32, NaN = no data): 10 m and 20 m bands, constant values."""
    coords = {
        "time": pd.date_range("2023-06-01", periods=t, freq="5D"),
        "y": 5230005.0 - 10 * np.arange(h),
        "x": 580005.0 + 10 * np.arange(w),
        "y20": 5230000.0 - 20 * np.arange(h // 2) - 10 + 20,
        "x20": 580010.0 + 20 * np.arange(w // 2),
    }
    data = {
        band: (
            ("time", "y", "x"),
            np.full((t, h, w), values.get(band, 1000), np.float32),
        )
        for band in FINE
    }
    for band in COARSE:
        value = scl if band == "SCL" else values.get(band, 1000)
        data[band] = (
            ("time", "y20", "x20"),
            np.full((t, h // 2, w // 2), value, np.float32),
        )
    return xr.Dataset(data, coords=coords).rio.write_crs("EPSG:32631")


def test_ndvi_and_nbr_values():
    cube = _cube({"B04": 300, "B08": 3000, "B12": 1000})

    ndvi = compute_index(cube, "NDVI", [4, 5, 6], 0)
    nbr = compute_index(cube, "NBR", [4, 5, 6], 0)

    assert ndvi.sizes == {"time": 2, "y": 12, "x": 12}
    assert float(ndvi[0, 0, 0]) == pytest.approx((0.3 - 0.03) / (0.3 + 0.03))
    assert float(nbr[0, 5, 5]) == pytest.approx(0.5)  # B12 upsampled from 20 m
    assert ndvi.dtype == np.float32


def test_crswir_matches_fordead_formula():
    value = crswir(xr.DataArray(0.3), xr.DataArray(0.2), xr.DataArray(0.1))

    # B11 / (B8A + (B12 - B8A) * (1610.4 - 864) / (2185.7 - 864))
    assert float(value) == pytest.approx(0.2 / (0.3 - 0.2 * 746.4 / 1321.7))


def test_negative_reflectance_is_clipped():
    cube = _cube({"B04": -150, "B08": 2000})

    assert float(reflectance(cube, "B04").min()) == 0
    assert float(compute_index(cube, "NDVI", [4], 0).max()) == pytest.approx(1.0)


def test_twenty_metre_bands_are_upsampled_on_read():
    cube = _cube({"B12": 2000})

    b12 = at_10m(cube, "B12")

    assert b12.sizes == {"time": 2, "y": 12, "x": 12}
    np.testing.assert_allclose(b12.values, 2000)
    assert (b12.x == cube.x).all()


def test_cloud_mask_is_dilated_at_20m_then_covers_two_by_two():
    cube = _cube({"B08": 3000})
    cube["SCL"][0, 3, 3] = 9  # one 20 m cloud pixel
    cube["B08"][1, 0, 0] = np.nan  # no data at 10 m

    # 2 px at 10 m = 1 px at 20 m: the cloud and its 8 neighbours at 20 m.
    ndvi = compute_index(cube, "NDVI", [4, 5, 6], dilation_px=2)

    assert np.isnan(ndvi[0, 4:10, 4:10]).all()
    assert int(np.isnan(ndvi[0]).sum()) == 36
    assert int(np.isnan(ndvi[1]).sum()) == 1  # dilation never crosses dates


def test_clear_mask_is_the_same_lazily_and_eagerly():
    rng = np.random.default_rng(0)
    scl = xr.DataArray(
        rng.choice([4, 9], size=(3, 20, 20), p=[0.9, 0.1]), dims=("time", "y", "x")
    )

    eager = clear_mask(scl, [4], 2)
    lazy = clear_mask(scl.chunk({"time": -1, "y": 7, "x": 7}), [4], 2).compute()

    np.testing.assert_array_equal(eager.values, lazy.values)  # block edges exact


def test_crop_cuts_both_resolutions_on_whole_20m_pixels():
    cube = _cube({})

    cut, row0, col0 = crop(cube, slice(3, 8), slice(-2, 5))

    assert (row0, col0) == (2, 0)
    assert cut.sizes["y"] == 6 and cut.sizes["x"] == 6
    assert cut.sizes["y20"] == 3 and cut.sizes["x20"] == 3


def test_cube_grid_recovers_the_pixel_grid():
    grid = cube_grid(_cube({}))

    assert grid.bounds == (580000.0, 5229890.0, 580120.0, 5230010.0)
    assert grid.resolution == 10
    assert grid.crs.to_epsg() == 32631
