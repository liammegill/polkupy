"""
Functions for processing data.
"""

import pandas as pd
import numpy as np


def apply_rolling_filter(df: pd.DataFrame):

    # apply rolling median filter to latitude, longitude and
    # other columns (as required)
    # the window size needs to be dependent on the sampling rate!
    df['lat'] = df['lat'].rolling(window=5, min_periods=1).median()
    df['lon'] = df['lon'].rolling(window=5, min_periods=1).median()
    df['speed'] = df['speed'].rolling(window=5, min_periods=1).median()

    return df


def filter_rides_in_bbox(df: pd.DataFrame, lat_min: float, lat_max: float,
                          lon_min: float, lon_max: float) -> pd.DataFrame:
    """Keep only rides that pass through a geographic bounding box.

    A ride is kept in full (not point-cropped) as long as at least one of
    its points falls inside the box, so routes that only partially cross
    the boundary aren't cut off mid-line when plotted.

    Args:
        df (pd.DataFrame): Points with 'ride_id', 'lat', 'lon' columns,
            e.g. as returned by `load_rides`.
        lat_min, lat_max, lon_min, lon_max (float): Bounding box in
            degrees.

    Returns:
        pd.DataFrame: Rows belonging to rides that intersect the box.
    """
    in_box = df['lat'].between(lat_min, lat_max) & df['lon'].between(lon_min, lon_max)
    ride_ids = df.loc[in_box, 'ride_id'].unique()
    return df[df['ride_id'].isin(ride_ids)].reset_index(drop=True)


def add_time_of_day(df: pd.DataFrame, tz: str = 'Europe/Berlin') -> pd.DataFrame:
    """Align every ride to a shared 24h clock, ignoring the calendar date.

    Rides recorded on different days become directly comparable: a ride
    that started at 08:00 and one that started at 08:05 can be replayed
    side by side. Adds 'ride_start_s' (seconds since local midnight the
    ride started) and 'virtual_s' (ride_start_s plus elapsed time into the
    ride), both anchored to the ride's own start rather than wall-clock
    date.

    Args:
        df (pd.DataFrame): Points with 'ride_id', 'time' columns, e.g. as
            returned by `load_rides`.
        tz (str): IANA timezone name used to resolve local time of day.

    Returns:
        pd.DataFrame: Copy of `df` with 'ride_start_s' and 'virtual_s'
            columns added.
    """
    df = df.copy()
    df['time'] = df['time'].dt.tz_convert(tz)
    ride_start = df.groupby('ride_id')['time'].transform('min')
    midnight = ride_start.dt.normalize()
    df['ride_start_s'] = (ride_start - midnight).dt.total_seconds()
    df['virtual_s'] = df['ride_start_s'] + (df['time'] - ride_start).dt.total_seconds()
    return df

