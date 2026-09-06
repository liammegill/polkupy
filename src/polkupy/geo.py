"""Generic geographic calculations shared across polkupy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

EARTH_RADIUS_M = 6371000  # mean Earth radius, in metres


def haversine(
    lon1: npt.ArrayLike, lat1: npt.ArrayLike, lon2: npt.ArrayLike, lat2: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Great-circle distance between two points on Earth using the
    haversine formula.

    Args:
        lon1 (numpy.typing.ArrayLike): Longitude of the first point(s), in
            degrees.
        lat1 (numpy.typing.ArrayLike): Latitude of the first point(s), in
            degrees.
        lon2 (numpy.typing.ArrayLike): Longitude of the second point(s), in
            degrees.
        lat2 (numpy.typing.ArrayLike): Latitude of the second point(s), in
            degrees.

    Returns:
        numpy.typing.NDArray[numpy.float64]: Distance between the two
            points, in metres. Matches the shape of whichever argument(s)
            were array-like.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * EARTH_RADIUS_M


def calc_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-point speed from consecutive GPS points.

    Args:
        df (pandas.DataFrame): Points with ``time``, ``lat``, ``lon``
            columns, e.g. :attr:`Ride.data <polkupy.core.ride.Ride.data>`.
            If a ``dist_m`` column is already present, it is reused.

    Returns:
        pandas.DataFrame: Copy of ``df`` with ``dist_m`` (great-circle
            distance from the previous point, in metres) and ``speed_kmh``
            (that distance divided by elapsed time, in km/h) columns added.
            The first row has no previous point, so both are ``NaN``.
    """
    df = df.copy()
    if "dist_m" not in df.columns:
        df["dist_m"] = haversine(
            df["lon"].shift(1), df["lat"].shift(1), df["lon"], df["lat"]
        )
    dt_s = (df["time"] - df["time"].shift(1)).dt.total_seconds()
    df["speed_kmh"] = df["dist_m"] / dt_s * 3.6
    return df
