"""
Read data from file.
"""

import gpxpy
import pandas as pd
import numpy as np


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
