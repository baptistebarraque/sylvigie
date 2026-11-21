import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

from sylvigie.forest import BEECH, DOUGLAS_FIR, FIR_SPRUCE
from sylvigie.labels import yyddd_to_days
from sylvigie.viz.timeseries import (
    pick_pixels,
    pixel_series,
    plot_pixel_series,
    situations,
)


def _layers():
    forest = np.full((60, 60), BEECH, dtype=np.uint8)
    forest[:, 30:] = DOUGLAS_FIR
    forest[40:, :20] = FIR_SPRUCE
    dates = np.zeros((60, 60), dtype=np.int32)
    dates[5:15, 40:50] = yyddd_to_days(np.array(23150))  # Douglas cut, 2023
    dates[45:55, 5:15] = yyddd_to_days(np.array(23200))  # spruce cut, 2023
    return forest, dates


def test_pick_pixels_finds_the_centre_of_each_situation():
    forest, dates = _layers()

    picked = {p.situation.key: p for p in pick_pixels(forest, dates, situations(2023))}

    assert set(picked) == {
        "stable_conifer",
        "stable_broadleaf",
        "douglas_cut",
        "spruce_cut",
    }
    cut = picked["douglas_cut"]
    assert (cut.row, cut.col) in {(9, 44), (9, 45), (10, 44), (10, 45)}
    assert cut.cut_date == dt.date(2023, 5, 30)
    stable = picked["stable_conifer"]
    assert forest[stable.row, stable.col] == DOUGLAS_FIR
    assert stable.cut_date is None
    # Stable pixels stay at least 100 m away from any cut.
    rows, cols = np.nonzero(dates)
    assert np.min(np.hypot(rows - stable.row, cols - stable.col)) > 10


def test_pick_pixels_skips_absent_situations():
    forest, dates = _layers()
    dates[:] = 0

    keys = {p.situation.key for p in pick_pixels(forest, dates, situations(2023))}

    assert keys == {"stable_conifer", "stable_broadleaf"}


def _cube():
    t = 4
    shape = (t, 9, 9)
    data = {
        b: (("time", "y", "x"), np.full(shape, 2000, np.float32))
        for b in ["B04", "B05", "B08", "B8A", "B11", "B12"]
    }
    data["B04"] = (("time", "y", "x"), np.full(shape, 500, np.float32))
    scl = np.full(shape, 4, np.float32)
    scl[1] = 9  # second date fully cloudy
    data["SCL"] = (("time", "y", "x"), scl)
    coords = {
        "time": pd.date_range("2023-06-01", periods=t, freq="10D"),
        "y": np.arange(9.0),
        "x": np.arange(9.0),
    }
    return xr.Dataset(data, coords=coords)


def test_pixel_series_drops_cloudy_dates(tmp_path):
    forest, dates = _layers()
    pixel = pick_pixels(forest, dates, situations(2023))[0]
    pixel = type(pixel)(pixel.situation, 4, 4, pixel.depth_px, None)

    table = pixel_series(_cube(), pixel, ["NDVI", "NBR"], [4, 5, 6], 1)

    assert len(table) == 3
    assert table["NDVI"].iloc[0] == np.float32((0.2 - 0.05) / (0.2 + 0.05))

    path = plot_pixel_series(
        [(pixel, table)], ["NDVI", "NBR"], tmp_path / "ts.png", "t"
    )
    assert path.exists() and path.stat().st_size > 10_000
