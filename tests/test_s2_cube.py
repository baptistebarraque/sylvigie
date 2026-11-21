"""Download and cube building, with a fake loader instead of the network."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pystac
import pytest
import xarray as xr

from sylvigie.config import load_config
from sylvigie.ingest import sentinel2
from sylvigie.ingest.sentinel2 import (
    KEPT,
    WRITTEN,
    WindowPaths,
    build_cube,
    cube_is_current,
    download,
    download_day,
    fuse,
    ingest_grid,
)
from sylvigie.preprocess.s2 import NODATA

CONFIG = Path(__file__).parents[1] / "configs" / "morvan.yaml"
BANDS = ["B04", "B05", "B08", "B8A", "B11", "B12"]


@pytest.fixture
def cfg():
    return load_config(CONFIG)


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    # Exercise the strip-by-strip logic on tiny arrays.
    monkeypatch.setattr(sentinel2, "BLOCK", 8)
    monkeypatch.setattr(sentinel2, "_sleep", lambda s: None)


def _item(item_id, baseline, cloud=10.0):
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=[3.9, 47.0, 4.3, 47.3],
        datetime=dt.datetime(2023, 6, 1, 10, 50, tzinfo=dt.UTC),
        properties={"s2:processing_baseline": baseline, "eo:cloud_cover": cloud},
    )


def _fake_load(values):
    """Loader returning constant digital numbers per item id (0 = no data)."""

    def load(item, grid, bands, collection):
        dn = values[item.id]
        out = {band: np.full(grid.shape, dn, dtype=np.uint16) for band in bands}
        out["SCL"] = np.full(grid.shape, 4 if dn else 0, dtype=np.uint8)
        return out

    return load


def _day(day, items, when="2023-06-01T10:50Z"):
    return pd.Series({"day": day, "items": items, "datetime": pd.Timestamp(when)})


def test_fuse_takes_first_valid_value():
    first = np.array([NODATA, 5, NODATA], dtype=np.int16)
    second = np.array([7, 8, NODATA], dtype=np.int16)

    np.testing.assert_array_equal(fuse([first, second], NODATA), [7, 5, NODATA])


def test_download_day_removes_offset_per_item(cfg):
    grid = ingest_grid(cfg, "dev", 0.2)
    items = {"old": _item("old", "03.00"), "new": _item("new", "05.10")}

    ds = download_day(
        _day(dt.date(2022, 1, 10), "old"),
        items,
        grid,
        BANDS,
        "c",
        _fake_load({"old": 1500}),
    )
    ds2 = download_day(
        _day(dt.date(2023, 6, 1), "new"),
        items,
        grid,
        BANDS,
        "c",
        _fake_load({"new": 1500}),
    )

    assert int(ds["B04"].max()) == 1500  # baseline 03.00: no offset
    assert int(ds2["B04"].max()) == 500  # baseline 05.10: -1000
    assert ds2["B04"].dtype == np.int16
    assert ds2.rio.crs.to_epsg() == 32631
    assert ds2.sizes == {"time": 1, "y": 20, "x": 20}
    # Pixel-centre coordinates of the grid.
    assert float(ds2.x[0]) == grid.bounds[0] + 5
    assert float(ds2.y[0]) == grid.bounds[3] - 5


def test_download_day_fuses_tiles_clearest_first(cfg):
    grid = ingest_grid(cfg, "dev", 0.2)
    items = {
        "cloudy": _item("cloudy", "05.10", 50),
        "clear": _item("clear", "05.10", 5),
    }
    values = {"cloudy": 3000, "clear": 2000}

    ds = download_day(
        _day(dt.date(2023, 6, 1), "cloudy;clear"),
        items,
        grid,
        BANDS,
        "c",
        _fake_load(values),
    )

    assert int(ds["B04"].max()) == 1000  # from the clearest item


@pytest.fixture
def staged(tmp_path, cfg):
    grid = ingest_grid(cfg, "dev", 0.2)  # 20 x 20 px: 3 strips of 8 rows
    paths = WindowPaths(
        raw=tmp_path,
        manifest=tmp_path / "manifest.csv",
        items=tmp_path / "items.json",
        days=tmp_path / "days",
        cube=tmp_path / "cube" / "s2_cube.zarr",
        forest_mask=tmp_path / "forest_mask.tif",
    )
    days = pd.DataFrame(
        {
            "day": [dt.date(2023, 6, d) for d in (1, 6, 11)],
            "datetime": pd.to_datetime(
                ["2023-06-01T10:50Z", "2023-06-06T10:50Z", "2023-06-11T10:50Z"]
            ),
            "items": ["a", "b", "c"],
            "orbits": "51",
            "platforms": "sentinel-2a",
            "baselines": "05.10",
            "clear_fraction": [0.9, 0.5, 0.7],
            "status": KEPT,
        }
    )
    items = {i: _item(i, "05.10") for i in "abc"}
    return cfg, grid, paths, days, items


def test_download_writes_one_store_per_day_and_resumes(staged):
    cfg, grid, paths, days, items = staged
    calls = []

    def load(item, grid, bands, collection):
        calls.append(item.id)
        return _fake_load(dict.fromkeys("abc", 1500))(item, grid, bands, collection)

    days = download(cfg, grid, days, paths, items, load)

    assert set(days["status"]) == {WRITTEN}
    assert all(paths.day_store(d).exists() for d in days["day"])
    assert not list(paths.days.glob("*.tmp"))
    calls.clear()
    download(cfg, grid, days, paths, items, load)
    assert calls == []  # nothing downloaded twice


def test_download_retries_a_failed_day_on_next_run(staged):
    cfg, grid, paths, days, items = staged

    def flaky(item, grid, bands, collection):
        if item.id == "b":
            raise OSError("HTTP 503")
        return _fake_load(dict.fromkeys("abc", 1500))(item, grid, bands, collection)

    days = download(cfg, grid, days, paths, items, flaky)

    assert days["status"].tolist() == [WRITTEN, KEPT, WRITTEN]
    days = download(
        cfg, grid, days, paths, items, _fake_load(dict.fromkeys("abc", 1500))
    )
    assert set(days["status"]) == {WRITTEN}


def test_build_cube_holds_the_time_series_in_blocks(staged):
    cfg, grid, paths, days, items = staged
    values = {"a": 1100, "b": 1200, "c": 1300}
    days = download(cfg, grid, days, paths, items, _fake_load(values))

    build_cube(days, paths, {"title": "test"})

    cube = xr.open_zarr(paths.cube, consolidated=False)
    assert cube.sizes == {"time": 3, "y": 20, "x": 20}
    assert cube["B04"].encoding["chunks"] == (3, 8, 8)
    assert cube["B04"].isel(y=19, x=19).values.tolist() == [100, 200, 300]
    assert cube["solar_day"].values.tolist() == [
        "2023-06-01",
        "2023-06-06",
        "2023-06-11",
    ]
    assert cube["clear_fraction"].values.tolist() == pytest.approx([0.9, 0.5, 0.7])
    assert cube.rio.crs.to_epsg() == 32631
    assert cube.attrs["title"] == "test"
    assert cube_is_current(days, paths)


def test_build_cube_keeps_nodata_as_nan(staged):
    cfg, grid, paths, days, items = staged
    days = download(
        cfg, grid, days, paths, items, _fake_load({"a": 0, "b": 1200, "c": 0})
    )

    build_cube(days, paths, {})

    decoded = xr.open_zarr(paths.cube, consolidated=False)
    assert decoded["B04"].isel(time=0).isnull().all()
    raw = xr.open_zarr(paths.cube, consolidated=False, mask_and_scale=False)
    assert int(raw["B04"].isel(time=0, y=0, x=0)) == NODATA
    assert int(raw["SCL"].isel(time=0, y=0, x=0)) == 0


def test_cube_is_not_current_when_a_day_is_missing(staged):
    cfg, grid, paths, days, items = staged
    days = download(
        cfg, grid, days, paths, items, _fake_load(dict.fromkeys("abc", 1500))
    )
    build_cube(days.iloc[:2], paths, {})

    assert not cube_is_current(days, paths)
