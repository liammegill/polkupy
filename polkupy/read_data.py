"""
Read data from file.
"""

from pathlib import Path

import gpxpy
import pandas as pd
import numpy as np


def load_rides(data_dir: str = "data", bikes: list[str] = None,
                extensions: list[str] = None) -> pd.DataFrame:
    """Load every GPX ride from a directory of per-bike subfolders.

    Args:
        data_dir (str): Directory containing one subfolder per bike, each
            holding that bike's GPX files (e.g. "data/katie/*.gpx").
        bikes (list[str], optional): Subset of bike subfolder names to
            load. Defaults to all subfolders found in `data_dir`.
        extensions (list[str], optional): Extensions of the GPX files to
            read, passed through to `load_gpx`.

    Returns:
        pd.DataFrame: All rides concatenated, with added 'ride_id' (unique
            per GPX file) and 'bike' columns.
    """
    data_path = Path(data_dir)
    bike_dirs = (
        [data_path / bike for bike in bikes] if bikes
        else sorted(p for p in data_path.iterdir() if p.is_dir())
    )

    rides = []
    for bike_dir in bike_dirs:
        for gpx_path in sorted(bike_dir.glob("*.gpx")):
            ride = load_gpx(str(gpx_path), extensions=extensions)
            if ride.empty:
                continue
            ride["ride_id"] = f"{bike_dir.name}/{gpx_path.stem}"
            ride["bike"] = bike_dir.name
            rides.append(ride)

    return pd.concat(rides, ignore_index=True)


def load_gpx(filepath: str, extensions: list[str] = None) -> pd.DataFrame:
    """Load a GPX file and extract data into a pandas DataFrame.

    Args:
        filepath (str): Path to the GPX file.
        extensions (list[str], optional): Extensions of the GPX file to read.

    Returns:
        pd.DataFrame: A DataFrame containing the GPX data with columns for
            time, latitude, longitude, elevation (uses None if not provided), 
            and any specified extensions.
    """
    # TODO: add checks on latitude and longitude values

    if extensions is None:
        extensions = []

    # open and parse the GPX file
    with open(filepath, 'r', encoding='utf-8') as file:
        gpx = gpxpy.parse(file)

    # extract data from GPX tracks
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                point_data = {
                    "time": point.time.isoformat(),
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "ele": point.elevation if point.elevation else None,
                }
                # initialise all extension fields to None
                point_data.update({ext: None for ext in extensions})

                # parse extensions
                if point.extensions:
                    for ext_element in point.extensions:
                        for child in ext_element.iter():
                            tag_name = child.tag.lower()
                            for ext in extensions:
                                if ext.lower() in tag_name:
                                    try:
                                        point_data[ext] = float(child.text)
                                    except (ValueError, TypeError):
                                        point_data[ext] = None

                # append the point data to the list
                data.append(point_data)

    # create pandas DataFrame and add speed
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])  # ensure datetime format
    return df


def haversine(lon1, lat1, lon2, lat2):
    """Calculates the distance between two points on Earth using the Haversine
    formula.
    From https://github.com/liammegill/contrail-limiting-factors/

    Args:
        lon1 (float): Longitude of the first point in degrees. 
        lat1 (float): Latitude of the first point in degrees. 
        lon2 (float): Longitude of the second point in degrees.
        lat2 (float): Latitude of the second point in degrees.

    Returns:
        float: The distance between the two points in meters.

    Notes:
    - The input longitudes and latitudes are assumed to be in degrees.
    - The Earth's radius is taken as 6371 kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Difference in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371e3 # Radius of Earth in metres
    distance = c * r

    return distance


def calc_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speed from latitude, longitude and time in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'time'
            columns.

    Returns:
        pd.DataFrame: DataFrame with an additional 'speed' column in km/h.
    """

    # calculate distance using haversine formula
    df['dist'] = haversine(
        df['lon'].shift(1), df['lat'].shift(1),
        df['lon'], df['lat']
    )

    # calculate speed
    dt = (df['time'] - df['time'].shift(1)).dt.total_seconds()
    df['speed'] = df['dist'] / dt * 3.6  # convert m/s to km/h

    return df
