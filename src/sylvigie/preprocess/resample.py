"""Exact x2 upsampling from the 20 m to the 10 m grid.

Our grids have edges on multiples of 20 m, so each 20 m pixel covers exactly
2 x 2 pixels of 10 m. Bilinear interpolation between pixel centres then has
fixed weights: a 10 m pixel sits 5 m from the centre of its 20 m parent and
15 m from the neighbour's, so it takes 3/4 of its parent and 1/4 of the
neighbour, along y and along x. At the array edge the edge pixel is repeated.

Works on NumPy arrays and lazily on dask arrays (blocks borrow a 1-pixel
halo from their neighbours, so block edges are exact).
"""

import numpy as np


def _bilinear_axis(a: np.ndarray, axis: int) -> np.ndarray:
    axis = axis % a.ndim
    n = a.shape[axis]
    before = np.concatenate(
        [np.take(a, [0], axis), np.take(a, range(n - 1), axis)], axis
    )
    after = np.concatenate(
        [np.take(a, range(1, n), axis), np.take(a, [n - 1], axis)], axis
    )
    first = 0.75 * a + 0.25 * before  # upper or left 10 m pixel
    second = 0.75 * a + 0.25 * after  # lower or right 10 m pixel
    shape = list(a.shape)
    shape[axis] = 2 * n
    return np.stack([first, second], axis=axis + 1).reshape(shape)


def bilinear_x2(a: np.ndarray) -> np.ndarray:
    """Bilinear x2 upsampling over the last two axes (y, x)."""
    return _bilinear_axis(_bilinear_axis(a, -2), -1)


def nearest_x2(a: np.ndarray) -> np.ndarray:
    """Each pixel copied into 2 x 2 pixels; for class maps such as SCL."""
    return np.repeat(np.repeat(a, 2, axis=-2), 2, axis=-1)


def _bilinear_block(block: np.ndarray) -> np.ndarray:
    # The block carries a 1-pixel halo on y and x; drop its 2 upsampled pixels.
    return bilinear_x2(block)[..., 2:-2, 2:-2]


def upsample_x2(data, method: str = "bilinear"):
    """x2 upsampling of a (..., y, x) array, NumPy or dask."""
    if method not in ("bilinear", "nearest"):
        raise ValueError(f"unknown method {method}")
    if not hasattr(data, "map_overlap"):
        values = np.asarray(data)
        return bilinear_x2(values) if method == "bilinear" else nearest_x2(values)
    doubled = tuple(
        c if axis < data.ndim - 2 else tuple(2 * s for s in c)
        for axis, c in enumerate(data.chunks)
    )
    if method == "nearest":
        return data.map_blocks(nearest_x2, chunks=doubled, dtype=data.dtype)
    depth = {axis: (1 if axis >= data.ndim - 2 else 0) for axis in range(data.ndim)}
    return data.map_overlap(
        _bilinear_block,
        depth=depth,
        boundary="nearest",
        trim=False,
        chunks=doubled,
        dtype=np.result_type(data.dtype, np.float32),
    )
