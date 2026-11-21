import datetime as dt

import numpy as np
import pandas as pd
import pytest

from sylvigie.preprocess.s2 import (
    NODATA,
    boa_offset,
    clear_fraction,
    day_table,
    deduplicate,
    harmonise,
    parse_baseline,
    solar_day,
    valid_mask,
)


def test_parse_baseline_compares_numerically():
    assert parse_baseline("05.10") == (5, 10)
    assert parse_baseline("05.09") < parse_baseline("05.10")


@pytest.mark.parametrize(
    ("baseline", "offset"), [("03.00", 0), ("04.00", 1000), ("05.11", 1000)]
)
def test_boa_offset_starts_at_baseline_04(baseline, offset):
    assert boa_offset(baseline) == offset


def _items(rows):
    columns = [
        "id",
        "datetime",
        "tile",
        "orbit",
        "platform",
        "baseline",
        "tile_cloud",
        "lon",
    ]
    table = pd.DataFrame(rows, columns=columns)
    table["datetime"] = pd.to_datetime(table["datetime"], utc=True)
    return table


def test_deduplicate_keeps_latest_baseline_per_tile_and_time():
    table = _items(
        [
            ("old", "2023-06-14T10:50Z", "31TEN", 51, "sentinel-2a", "05.09", 3.0, 4.0),
            ("new", "2023-06-14T10:50Z", "31TEN", 51, "sentinel-2a", "05.10", 3.0, 4.0),
            (
                "other_tile",
                "2023-06-14T10:50Z",
                "31TFN",
                51,
                "sentinel-2a",
                "05.09",
                9.0,
                5.0,
            ),
        ]
    )

    kept = deduplicate(table)

    assert sorted(kept["id"]) == ["new", "other_tile"]


def test_solar_day_shifts_by_longitude():
    table = _items(
        [
            ("a", "2023-06-14T23:30Z", "31TEN", 51, "s2a", "05.10", 1.0, 30.0),  # +2 h
            ("b", "2023-06-14T23:30Z", "31TEN", 51, "s2a", "05.10", 1.0, 4.0),
        ]
    )

    assert solar_day(table).tolist() == [dt.date(2023, 6, 15), dt.date(2023, 6, 14)]


def test_day_table_groups_items_of_one_day():
    table = _items(
        [
            ("a", "2022-01-05T10:50Z", "31TEN", 8, "sentinel-2a", "03.00", 40.0, 4.0),
            ("b", "2022-01-05T10:50Z", "31TFN", 8, "sentinel-2a", "04.00", 20.0, 5.0),
            ("c", "2022-01-10T10:50Z", "31TEN", 51, "sentinel-2b", "04.00", 60.0, 4.0),
        ]
    )

    days = day_table(table)

    assert days["day"].tolist() == [dt.date(2022, 1, 5), dt.date(2022, 1, 10)]
    assert days.loc[0, "items"] == "a;b"
    assert days.loc[0, "offsets"] == "0;1000"  # mixed baselines on one day
    assert days.loc[0, "tile_cloud"] == 20.0


def test_harmonise_removes_offset_and_keeps_nodata():
    dn = np.array([[0, 1000, 1500], [np.nan, 900, 65535]], dtype=np.float32)

    out = harmonise(dn, offset=1000)

    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, [[NODATA, 0, 500], [NODATA, -100, 32767]])


def test_harmonise_without_offset_rounds():
    assert harmonise(np.array([1146.56]), offset=0).tolist() == [1147]


def test_valid_mask_dilates_invalid_pixels():
    scl = np.full((7, 7), 4, dtype=np.uint8)
    scl[3, 3] = 9  # one cloud pixel

    valid = valid_mask(scl, keep=[4, 5, 6], dilation_px=1)

    assert (~valid).sum() == 9  # the cloud and its 8 neighbours
    assert valid_mask(scl, keep=[4, 5, 6], dilation_px=0).sum() == 48


def test_clear_fraction_counts_forest_only():
    valid = np.array([[True, False], [True, True]])
    forest = np.array([[True, True], [False, False]])

    assert clear_fraction(valid, forest) == 0.5
    assert clear_fraction(valid, np.zeros_like(forest)) == 0.0
