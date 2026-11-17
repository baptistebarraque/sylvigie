import numpy as np
import pytest

from sylvigie.aoi import Grid
from sylvigie.zone import (
    BROADLEAF,
    CONIFER,
    MIXED,
    NON_FOREST,
    Profile,
    area_profile,
    block_sum,
    compute_cell_stats,
    count_per_cell,
    cut_polygons,
    find_cut_objects,
    pick_distinct,
    rank_windows,
    sufosat_years,
    window_bounds,
    window_overlap,
    window_sums,
)


def test_sufosat_years_decodes_yyddd():
    dates = np.array([[0, 23001], [24366, 18150]], dtype=np.int16)

    years = sufosat_years(dates)

    np.testing.assert_array_equal(years, [[0, 2023], [2024, 2018]])


def test_find_cut_objects_keeps_large_patches_with_centroids():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:2] = True  # 4 px
    mask[5:8, 5:8] = True  # 9 px, centroid (6, 6)

    objects = find_cut_objects(mask, min_pixels=5)

    np.testing.assert_array_equal(objects.sizes, [9])
    np.testing.assert_allclose(objects.rows, [6.0])
    np.testing.assert_allclose(objects.cols, [6.0])


def test_find_cut_objects_joins_diagonal_neighbours():
    mask = np.eye(4, dtype=bool)

    objects = find_cut_objects(mask, min_pixels=1)

    np.testing.assert_array_equal(objects.sizes, [4])


def test_find_cut_objects_handles_empty_mask():
    objects = find_cut_objects(np.zeros((4, 4), dtype=bool), min_pixels=1)

    assert objects.sizes.size == 0


def test_block_sum():
    values = np.arange(16).reshape(4, 4)

    np.testing.assert_array_equal(block_sum(values, 2), [[10, 18], [42, 50]])


def test_block_sum_rejects_ragged_shape():
    with pytest.raises(ValueError):
        block_sum(np.ones((5, 4)), 2)


def test_count_per_cell():
    counts = count_per_cell(np.array([0.5, 1.9, 3.2]), np.array([0, 1, 3]), 2, (2, 2))

    np.testing.assert_array_equal(counts, [[2, 0], [0, 1]])


def test_window_sums_matches_brute_force():
    cells = np.random.default_rng(0).integers(0, 10, size=(7, 9))
    size = 3

    sums = window_sums(cells, size)

    expected = np.array(
        [
            [cells[i : i + size, j : j + size].sum() for j in range(9 - size + 1)]
            for i in range(7 - size + 1)
        ]
    )
    np.testing.assert_array_equal(sums, expected)


def test_window_overlap():
    assert window_overlap((0, 0), (0, 0), 4) == 1
    assert window_overlap((0, 0), (0, 2), 4) == 0.5
    assert window_overlap((0, 0), (4, 0), 4) == 0


def test_area_profile():
    years_px = np.array([[2023, 0], [2024, 0]])
    forest = np.array([[CONIFER, BROADLEAF], [MIXED, NON_FOREST]])
    mask = np.ones((2, 2), dtype=bool)

    profile = area_profile(years_px, forest, mask, [2023, 2024])

    assert profile.cut_rate == pytest.approx(1 / 3)  # 1 cut px per year, 3 forest px
    assert profile.conifer_share == pytest.approx(1.5 / 3)  # mixed counts half


def _synthetic_park():
    """8 x 8 cells of 4 x 4 px: cuts are dense in the left half only."""
    cell, n = 4, 8
    years_px = np.zeros((cell * n, cell * n), dtype=np.int16)
    for i in range(0, cell * n, cell):
        for j in range(0, cell * n // 2, cell):
            years_px[i, j] = 2023  # one 1-px cut per cell and per year
            years_px[i + 2, j + 2] = 2024
    forest = np.full(years_px.shape, BROADLEAF, dtype=np.uint8)
    forest[:, ::2] = CONIFER
    park = np.ones(years_px.shape, dtype=bool)
    return years_px, forest, park, cell


def test_rank_windows_applies_thresholds():
    years_px, forest, park, cell = _synthetic_park()
    stats = compute_cell_stats(years_px, forest, park, [2023, 2024], cell, 1)
    profile = area_profile(years_px, forest, park, [2023, 2024])

    ranked = rank_windows(stats, [2023, 2024], 4, profile, 16, 0.5, 0.01)

    # A 4-cell window holds 16 cuts per year only inside the left half.
    assert set(ranked["col"]) == {0}
    assert (ranked["cuts_2023"] == 16).all()
    assert (ranked["cuts_2024"] == 16).all()


def test_rank_windows_prefers_windows_like_the_park():
    years_px, forest, park, cell = _synthetic_park()
    stats = compute_cell_stats(years_px, forest, park, [2023, 2024], cell, 1)
    # Park-wide rate is half the left-half rate: the best windows straddle both halves.
    profile = area_profile(years_px, forest, park, [2023, 2024])

    ranked = rank_windows(stats, [2023, 2024], 4, profile, 0, 0.5, 0.01)

    assert ranked.loc[0, "col"] == 2
    assert ranked["score"].is_monotonic_increasing


def test_rank_windows_rejects_windows_outside_park():
    years_px, forest, park, cell = _synthetic_park()
    park[:, 8:] = False  # park covers the first 2 cell columns only
    stats = compute_cell_stats(years_px, forest, park, [2023, 2024], cell, 1)
    profile = Profile(cut_rate=0.01, conifer_share=0.5)

    ranked = rank_windows(stats, [2023, 2024], 4, profile, 0, 0.5, 0.01)

    assert set(ranked["col"]) == {0}  # only col 0 has >= 50 % of its area in park


def test_pick_distinct_skips_overlapping_windows():
    years_px, forest, park, cell = _synthetic_park()
    stats = compute_cell_stats(years_px, forest, park, [2023, 2024], cell, 1)
    profile = area_profile(years_px, forest, park, [2023, 2024])
    ranked = rank_windows(stats, [2023, 2024], 4, profile, 0, 0.5, 0.01)

    picked = pick_distinct(ranked, 4, max_overlap=0.5, n=3)

    positions = list(zip(picked["row"], picked["col"], strict=True))
    assert len(positions) == 3
    for k, a in enumerate(positions):
        for b in positions[k + 1 :]:
            assert window_overlap(a, b, 4) <= 0.5


def test_cut_polygons_keeps_large_patches_inside_bounds():
    grid = Grid.from_bounds((0, 0, 1000, 1000), "EPSG:32631", 10)
    years_px = np.zeros(grid.shape, dtype=np.int16)
    years_px[10:20, 10:20] = 2023  # 1 ha
    years_px[50:52, 50:52] = 2023  # 0.04 ha, too small
    years_px[30:40, 30:40] = 2024  # 1 ha
    years_px[90:100, 90:100] = 2024  # 1 ha, outside the bounds

    cuts = cut_polygons(years_px, grid, [2023, 2024], 0.5, (0, 0, 800, 1000))

    assert sorted(cuts["year"]) == [2023, 2024]
    assert cuts["area_ha"].tolist() == pytest.approx([1.0, 1.0])
    assert cuts.crs.to_epsg() == 32631
    assert cuts.geometry.iloc[0].bounds == (100.0, 800.0, 200.0, 900.0)


def test_window_bounds():
    grid = Grid.from_bounds((0, 0, 1000, 1000), "EPSG:32631", 10)

    assert window_bounds(grid, 1, 2, 3, 10) == (200.0, 600.0, 500.0, 900.0)
