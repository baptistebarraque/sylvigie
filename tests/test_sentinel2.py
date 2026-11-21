import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sylvigie.config import load_config
from sylvigie.ingest import sentinel2
from sylvigie.ingest.sentinel2 import (
    DROPPED,
    FAILED,
    KEPT,
    PLANNED,
    WindowPaths,
    ingest_grid,
    load_manifest,
    save_manifest,
    screen,
)
from sylvigie.raster import write_geotiff

CONFIG = Path(__file__).parents[1] / "configs" / "morvan.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sentinel2, "_sleep", lambda s: None)


def test_ingest_grid_subset_is_centred_and_aligned(cfg):
    full = ingest_grid(cfg, "dev", None)
    sub = ingest_grid(cfg, "dev", 5)

    assert full.shape == (2600, 2600)
    assert sub.shape == (500, 500)
    assert sub.bounds == (579000, 5215000, 584000, 5220000)
    # Same pixel lattice as the window grid: offsets are whole pixels.
    assert (sub.bounds[0] - full.bounds[0]) % 10 == 0
    assert all(v % 20 == 0 for v in sub.bounds)


def _days(n):
    return pd.DataFrame(
        {
            "day": [dt.date(2023, 6, d + 1) for d in range(n)],
            "datetime": pd.to_datetime(
                [f"2023-06-{d + 1:02d}T10:50Z" for d in range(n)]
            ),
            "items": [f"item{d}" for d in range(n)],
            "tiles": "31TEN",
            "orbits": "51",
            "platforms": "sentinel-2a",
            "baselines": "05.10",
            "offsets": "1000",
            "tile_cloud": 10.0,
            "clear_fraction": np.nan,
            "status": PLANNED,
        }
    )


def test_manifest_roundtrip(tmp_path):
    days = _days(3)
    path = tmp_path / "manifest.csv"

    save_manifest(days, path)
    back = load_manifest(path)

    assert back["day"].tolist() == days["day"].tolist()
    assert back["offsets"].tolist() == ["1000"] * 3
    assert back["status"].tolist() == [PLANNED] * 3


@pytest.fixture
def window(tmp_path, cfg):
    grid = ingest_grid(cfg, "dev", 0.2)  # 20 x 20 px
    forest = np.zeros(grid.shape, dtype=np.uint8)
    forest[:, :10] = 1  # forest on the left half
    paths = WindowPaths(
        raw=tmp_path,
        manifest=tmp_path / "manifest.csv",
        items=tmp_path / "items.json",
        forest_mask=write_geotiff(forest, grid, tmp_path / "forest_mask.tif"),
    )
    return grid, paths


def _fake_scl(clear_columns):
    """SCL at 20 m: vegetation (4) on the first `clear_columns`, cloud (9) after."""

    def load(items, grid, collection):
        scl = np.full(grid.shape, 9, dtype=np.uint8)
        scl[:, :clear_columns] = 4
        return scl

    return load


def test_screen_keeps_days_clear_over_forest(cfg, window):
    grid, paths = window
    items = {f"item{d}": object() for d in range(2)}

    days = screen(cfg, grid, _days(2), paths, items, load=_fake_scl(5))

    # Forest is the left 5 columns at 20 m; 1 column lost to the dilation.
    assert days["clear_fraction"].tolist() == pytest.approx([0.8, 0.8])
    assert set(days["status"]) == {KEPT}
    assert load_manifest(paths.manifest)["status"].tolist() == [KEPT, KEPT]


def test_screen_drops_cloudy_days(cfg, window):
    grid, paths = window

    days = screen(cfg, grid, _days(1), paths, {"item0": object()}, load=_fake_scl(0))

    assert days.loc[0, "status"] == DROPPED


def test_screen_marks_failures_and_retries_them_later(cfg, window):
    grid, paths = window
    calls = []

    def broken(items, grid, collection):
        calls.append(1)
        raise OSError("HTTP 503")

    days = screen(cfg, grid, _days(1), paths, {"item0": object()}, load=broken)

    assert days.loc[0, "status"] == FAILED
    assert len(calls) == sentinel2.LOAD_ATTEMPTS
    again = screen(cfg, grid, days, paths, {"item0": object()}, load=_fake_scl(5))
    assert again.loc[0, "status"] == KEPT


def test_screen_skips_days_already_screened(cfg, window):
    grid, paths = window
    days = _days(2)
    days.loc[0, ["status", "clear_fraction"]] = [DROPPED, 0.1]
    seen = []

    def load(items, grid, collection):
        seen.append(items)
        return _fake_scl(5)(items, grid, collection)

    screen(cfg, grid, days, paths, {"item0": "a", "item1": "b"}, load=load)

    assert seen == [["b"]]


def test_screen_needs_the_forest_mask(cfg, window):
    grid, paths = window
    paths.forest_mask.unlink()

    with pytest.raises(FileNotFoundError, match="sylvigie grid"):
        screen(cfg, grid, _days(1), paths, {}, load=_fake_scl(5))
