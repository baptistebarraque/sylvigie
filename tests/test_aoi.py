import pytest

from sylvigie.aoi import Grid


def test_from_bounds_snaps_outwards_to_anchor():
    grid = Grid.from_bounds((1234, 5678, 4321, 8765), "EPSG:32631", 10, anchor=1000)

    assert grid.bounds == (1000, 5000, 5000, 9000)
    assert grid.shape == (400, 400)
    assert grid.crs.to_epsg() == 32631


def test_from_bounds_defaults_to_pixel_anchor():
    grid = Grid.from_bounds((3, 3, 27, 27), "EPSG:32631", 10)

    assert grid.bounds == (0, 0, 30, 30)


def test_anchor_must_be_multiple_of_resolution():
    with pytest.raises(ValueError):
        Grid.from_bounds((0, 0, 100, 100), "EPSG:32631", 10, anchor=15)


def test_crop():
    grid = Grid.from_bounds((0, 0, 1000, 1000), "EPSG:32631", 10)

    (rows, cols), sub = grid.crop((200, 600, 500, 900))

    assert (rows, cols) == (slice(10, 40), slice(20, 50))
    assert sub.bounds == (200, 600, 500, 900)
    assert sub.shape == (30, 30)


def test_crop_clips_to_grid():
    grid = Grid.from_bounds((0, 0, 1000, 1000), "EPSG:32631", 10)

    _, sub = grid.crop((-500, -500, 100, 100))

    assert sub.bounds == (0, 0, 100, 100)


def test_to_geobox_keeps_the_grid():
    grid = Grid.from_bounds((1000, 5000, 2000, 6000), "EPSG:32631", 10)

    geobox = grid.to_geobox()

    assert geobox.shape == grid.shape
    assert geobox.affine == grid.transform
    assert geobox.crs.epsg == 32631


def test_coarsen():
    grid = Grid.from_bounds((0, 0, 1000, 2000), "EPSG:32631", 10)

    coarse = grid.coarsen(10)

    assert coarse.shape == (20, 10)
    assert coarse.resolution == 100
    assert coarse.bounds == grid.bounds


def test_coarsen_rejects_ragged_factor():
    grid = Grid.from_bounds((0, 0, 1000, 1000), "EPSG:32631", 10)

    with pytest.raises(ValueError):
        grid.coarsen(3)
