"""Clear-cut labels: SUFOSAT dates as days since 1970-01-01.

Days since the Unix epoch are NumPy's `datetime64[D]` convention: a delay
between two dates is a plain subtraction, and `.astype("datetime64[D]")`
gives calendar dates back. 0 (1970-01-01) means "no clear-cut".
"""

import numpy as np

NO_CUT = 0


def yyddd_to_days(yyddd: np.ndarray) -> np.ndarray:
    """SUFOSAT YYDDD (year 20YY, day of year DDD; 0 = no cut) to days since 1970."""
    yyddd = np.asarray(yyddd)
    cut = yyddd > 0
    year = 2000 + yyddd // 1000
    doy = yyddd % 1000
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    bad = cut & ((doy < 1) | (doy > 365 + leap))
    if bad.any():
        raise ValueError(f"invalid YYDDD values: {np.unique(yyddd[bad])[:10]}")
    jan1 = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    days = (jan1 + (doy - 1)).astype(np.int64)
    return np.where(cut, days, NO_CUT).astype(np.int32)


def days_to_year(days: np.ndarray) -> np.ndarray:
    """Calendar year of each date; 0 where there is no cut."""
    years = days.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    return np.where(days != NO_CUT, years, 0)
