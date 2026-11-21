import dask.array as da
import numpy as np
import pytest

from sylvigie.preprocess.resample import bilinear_x2, nearest_x2, upsample_x2


def test_bilinear_keeps_a_constant_field():
    a = np.full((2, 3, 4), 7.0)

    up = bilinear_x2(a)

    assert up.shape == (2, 6, 8)
    np.testing.assert_allclose(up, 7.0)


def test_bilinear_is_exact_on_a_ramp_away_from_edges():
    # A field linear in x at 20 m pixel centres (10, 30, 50, ...).
    x20 = 10 + 20 * np.arange(6)
    a = np.tile(x20.astype(float), (4, 1))

    up = bilinear_x2(a)

    x10 = 5 + 10 * np.arange(12)
    np.testing.assert_allclose(up[0, 1:-1], x10[1:-1])  # edges repeat the edge pixel


def test_nearest_copies_each_pixel_into_two_by_two():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    np.testing.assert_array_equal(
        nearest_x2(a), [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    )


def test_missing_value_spreads_to_its_neighbours_only():
    a = np.ones((1, 6, 6))
    a[0, 3, 3] = np.nan

    up = bilinear_x2(a)

    assert np.isnan(up).sum() == 16  # its own 2 x 2 and the half-pixels around


@pytest.mark.parametrize("method", ["bilinear", "nearest"])
def test_lazy_equals_eager_across_block_edges(method):
    rng = np.random.default_rng(0)
    a = rng.random((3, 20, 22)).astype(np.float32)

    eager = upsample_x2(a, method)
    lazy = upsample_x2(da.from_array(a, chunks=(3, 7, 6)), method)

    assert lazy.chunks[1] == (14, 14, 12)
    np.testing.assert_allclose(lazy.compute(), eager, rtol=1e-6)
