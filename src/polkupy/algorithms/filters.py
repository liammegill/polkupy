"""Filters over a collection of rides."""

from __future__ import annotations

import pandas as pd


def filter_rides_in_bbox(
    df: pd.DataFrame, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> pd.DataFrame:
    """Keep only rides that pass through a geographic bounding box. A ride is
    kept in full (not point-cropped) as long as at least one of its points
    falls inside the box.

    Args:
        df (pandas.DataFrame): Points with ``ride_id``, ``lat``, ``lon``
            columns, e.g. :attr:`Rides.data <polkupy.core.rides.Rides.data>`.
        lat_min (float): minimum latitude of the box, in degrees.
        lat_max (float): maximum latitude of the box, in degrees.
        lon_min (float): minimum longitude of the box, in degrees.
        lon_max (float): maximum longitude of the box, in degrees.

    Returns:
        pandas.DataFrame: Rows belonging to rides that intersect the box.
    """
    in_box = df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    ride_ids = df.loc[in_box, "ride_id"].unique()
    return df[df["ride_id"].isin(ride_ids)].reset_index(drop=True)
