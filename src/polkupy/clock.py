"""Time-of-day alignment shared across polkupy."""

from __future__ import annotations

import pandas as pd


def add_time_of_day(df: pd.DataFrame, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """Align every ride to a shared 24h clock, ignoring the calendar date.

    Rides recorded on different days become directly comparable: a ride
    that started at 08:00 and one that started at 08:05 can be replayed
    side by side. Adds ``ride_start_s`` (seconds since local midnight the
    ride started) and ``virtual_s`` (``ride_start_s`` plus elapsed time
    into the ride).

    Args:
        df (pandas.DataFrame): Points with ``ride_id``, ``time`` columns,
            e.g. :attr:`Rides.data <polkupy.core.rides.Rides.data>`.
        tz (str): IANA timezone name used to resolve local time of day.

    Returns:
        pandas.DataFrame: Copy of ``df`` with ``ride_start_s`` and
        ``virtual_s`` columns added.
    """
    df = df.copy()
    df["time"] = df["time"].dt.tz_convert(tz)
    ride_start = df.groupby("ride_id")["time"].transform("min")
    midnight = ride_start.dt.normalize()
    df["ride_start_s"] = (ride_start - midnight).dt.total_seconds()
    df["virtual_s"] = df["ride_start_s"] + (df["time"] - ride_start).dt.total_seconds()
    return df
