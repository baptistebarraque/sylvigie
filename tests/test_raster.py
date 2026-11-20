import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from sylvigie.aoi import Grid
from sylvigie.raster import (
    rasterize_on_grid,
    read_on_grid,
    reproject_array,
    write_geotiff,
)

# A Lambert-93 patch in the Morvan and the UTM 31N grid covering it.
L93_GRID = Grid.from_bounds((780000, 6680000, 781000, 6681000), "EPSG:2154", 10)
UTM_GRID = Grid.from_bounds((620000, 5230000, 621000, 5231000), "EPSG:32631", 10)


def test_write_geotiff_roundtrip(tmp_path):
    values = np.arange(100 * 100, dtype=np.int16).reshape(100, 100)

    path = write_geotiff(values, L93_GRID, tmp_path / "out.tif", nodata=0)

    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 2154
        assert src.transform == L93_GRID.transform
        assert src.nodata == 0
        assert src.profile["compress"].lower() == "deflate"
        np.testing.assert_array_equal(src.read(1), values)


def test_write_geotiff_keeps_tags(tmp_path):
    values = np.zeros(L93_GRID.shape, dtype=np.uint8)

    path = write_geotiff(values, L93_GRID, tmp_path / "t.tif", tags={"source": "IGN"})

    with rasterio.open(path) as src:
        assert src.tags()["source"] == "IGN"


def test_write_geotiff_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        write_geotiff(np.zeros((3, 3)), L93_GRID, tmp_path / "out.tif")


def test_read_on_grid_reprojects_categories_without_new_values(tmp_path):
    # Dates must survive reprojection unchanged: nearest neighbour only.
    rng = np.random.default_rng(0)
    values = rng.choice([0, 23150, 24200], size=L93_GRID.shape).astype(np.int16)
    path = write_geotiff(values, L93_GRID, tmp_path / "dates.tif", nodata=0)
    grid = Grid.from_bounds(
        transform_bounds("EPSG:2154", "EPSG:32631", *L93_GRID.bounds),
        "EPSG:32631",
        10,
    )

    out = read_on_grid(path, grid)

    assert out.shape == grid.shape
    assert set(np.unique(out)) <= {0, 23150, 24200}
    assert (out == 23150).any() and (out == 24200).any()


def test_read_on_grid_fills_outside_with_nodata(tmp_path):
    path = write_geotiff(
        np.full(L93_GRID.shape, 7, dtype=np.int16), L93_GRID, tmp_path / "a.tif", 0
    )
    far_away = Grid.from_bounds((0, 0, 100, 100), "EPSG:32631", 10)

    assert (read_on_grid(path, far_away) == 0).all()


def test_reproject_array_keeps_every_value(recwarn):
    # Regression: declaring 0 as nodata made GDAL turn source zeros into ones.
    codes = np.zeros(UTM_GRID.shape, dtype=np.uint8)
    codes[:, 50:] = 1

    out, grid = reproject_array(codes, UTM_GRID, "EPSG:3857", fill=255)

    assert grid.crs.to_epsg() == 3857
    values, counts = np.unique(out, return_counts=True)
    shares = dict(zip(values.tolist(), counts / out.size, strict=True))
    assert set(shares) <= {0, 1, 255}
    assert shares[0] == pytest.approx(shares[1], abs=0.05)  # halves stay halves
    assert not [w for w in recwarn if "changed to 1" in str(w.message)]


def test_rasterize_on_grid_burns_values_after_reprojection():
    left_half = box(620000, 5230000, 620500, 5231000)
    gdf = gpd.GeoDataFrame({"code": [3]}, geometry=[left_half], crs="EPSG:32631")

    out = rasterize_on_grid(gdf.to_crs("EPSG:2154"), UTM_GRID, values="code")

    assert out.shape == UTM_GRID.shape
    # Reprojected edges are not exactly on pixel edges: allow one pixel column.
    assert (out[:, :49] == 3).all()
    assert (out[:, 51:] == 0).all()
