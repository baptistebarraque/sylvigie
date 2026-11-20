import numpy as np
import pytest

from sylvigie.labels import NO_CUT, days_to_year, yyddd_to_days


def test_yyddd_to_days_matches_numpy_dates():
    yyddd = np.array([[0, 23001], [24060, 24366]], dtype=np.int16)

    days = yyddd_to_days(yyddd)

    assert days.dtype == np.int32
    assert days[0, 0] == NO_CUT
    dates = days.astype("datetime64[D]")
    assert str(dates[0, 1]) == "2023-01-01"
    assert str(dates[1, 0]) == "2024-02-29"  # 2024 is a leap year
    assert str(dates[1, 1]) == "2024-12-31"


def test_delay_is_a_subtraction():
    days = yyddd_to_days(np.array([23360, 24005]))

    assert days[1] - days[0] == 10


@pytest.mark.parametrize("bad", [23000, 23366, 24367])
def test_yyddd_to_days_rejects_impossible_days(bad):
    with pytest.raises(ValueError):
        yyddd_to_days(np.array([bad]))


def test_days_to_year():
    days = yyddd_to_days(np.array([0, 22365, 23001]))

    np.testing.assert_array_equal(days_to_year(days), [0, 2022, 2023])
