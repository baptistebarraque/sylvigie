import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from sylvigie.aoi import Grid
from sylvigie.forest import BEECH, DOUGLAS_FIR, UNKNOWN
from sylvigie.labels import NO_CUT, days_to_year
from sylvigie.raster import write_geotiff
from sylvigie.reference import ReferenceLayers, build_layers, summarize, window_grid

UTM = "EPSG:32631"
WINDOW = (620000, 5230000, 621000, 5231000)  # 1 km, 100 x 100 px


def test_window_grid_adds_margin_and_marks_the_window():
    grid, eval_area = window_grid(WINDOW, 200, UTM, 10)

    assert grid.bounds == (619800, 5229800, 621200, 5231200)
    assert grid.shape == (140, 140)
    assert eval_area.sum() == 100 * 100
    assert not eval_area[:20].any() and not eval_area[:, -20:].any()


def _sufosat(tmp_path, yyddd, prob, name):
    """SUFOSAT-like rasters in Lambert-93 covering the test window."""
    l93 = Grid.from_bounds((780000, 6680000, 784000, 6684000), "EPSG:2154", 10)
    dates = np.full(l93.shape, yyddd, dtype=np.int16)
    probs = np.full(l93.shape, prob, dtype=np.int16)
    return (
        write_geotiff(dates, l93, tmp_path / f"{name}_dates.tif", nodata=0),
        write_geotiff(probs, l93, tmp_path / f"{name}_prob.tif", nodata=0),
    )


@pytest.fixture
def forest_inputs():
    # Window centred on the Lambert-93 test rasters.
    bdforet = gpd.GeoDataFrame(
        {"essence": ["Douglas"], "tfv_g11": ["Forêt fermée conifères"]},
        geometry=[box(781500, 6681500, 782000, 6682500)],
        crs="EPSG:2154",
    )
    masque = gpd.GeoDataFrame(
        geometry=[box(782000, 6681500, 782500, 6682500)], crs="EPSG:2154"
    )
    return bdforet, masque


def _window_on_test_rasters():
    from rasterio.warp import transform_bounds

    xmin, ymin, xmax, ymax = transform_bounds(
        "EPSG:2154", UTM, 781200, 6681200, 782800, 6682800
    )
    return Grid.from_bounds((xmin, ymin, xmax, ymax), UTM, 10, anchor=20)


def test_build_layers_converts_labels(tmp_path, forest_inputs):
    grid = _window_on_test_rasters()
    dates, prob = _sufosat(tmp_path, 23150, 92, "a")

    layers = build_layers(grid, np.ones(grid.shape, bool), *forest_inputs, dates, prob)

    assert (days_to_year(layers.sufosat_date) == 2023).all()
    assert layers.sufosat_date.dtype == np.int32
    assert (layers.sufosat_prob == 92).all()
    assert DOUGLAS_FIR in layers.forest_type and UNKNOWN in layers.forest_type


def test_forest_layers_do_not_depend_on_labels(tmp_path, forest_inputs):
    # Guard against leakage: changing the labels must not change the mask.
    grid = _window_on_test_rasters()
    everywhere = np.ones(grid.shape, bool)
    cut_everywhere = build_layers(
        grid, everywhere, *forest_inputs, *_sufosat(tmp_path, 23150, 99, "a")
    )
    never_cut = build_layers(
        grid, everywhere, *forest_inputs, *_sufosat(tmp_path, 0, 0, "b")
    )

    np.testing.assert_array_equal(cut_everywhere.forest_mask, never_cut.forest_mask)
    np.testing.assert_array_equal(cut_everywhere.forest_type, never_cut.forest_type)
    assert (never_cut.sufosat_date == NO_CUT).all()
    assert (never_cut.sufosat_prob == 0).all()


def test_summarize_counts_inside_the_window_only():
    grid, eval_area = window_grid((0, 0, 100, 100), 20, UTM, 10)  # 14 x 14 px
    forest_type = np.zeros(grid.shape, np.uint8)
    forest_type[:, :7] = BEECH
    forest_mask = forest_type > 0
    days = np.zeros(grid.shape, np.int32)
    day_2023 = int(np.datetime64("2023-06-01", "D").astype(int))
    days[5, 3] = day_2023  # in the window, in beech forest
    days[5, 10] = day_2023  # in the window, outside the mask
    days[0, 0] = day_2023  # in the margin: ignored
    layers = ReferenceLayers(
        grid, eval_area, forest_mask, forest_type, days, np.zeros_like(forest_type)
    )

    table, coverage = summarize(layers, [2022, 2023])

    beech = table.set_index("forest_type").loc["beech"]
    outside = table.set_index("forest_type").loc["outside forest mask"]
    assert beech["area_ha"] == pytest.approx(
        10 * 5 * 0.01
    )  # 10 rows x 5 cols in window
    assert beech["cut_ha_2023"] == pytest.approx(0.01)
    assert outside["cut_ha_2023"] == pytest.approx(0.01)
    assert coverage == {2022: 0.0, 2023: 0.5}
