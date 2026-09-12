"""Open and parse .gpx files."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import gpxpy
import gpxpy.gpx
import pandas as pd

# first-order sanity check
LATITUDE_RANGE = (-90.0, 90.0)
LONGITUDE_RANGE = (-180.0, 180.0)
ELEVATION_RANGE = (-500.0, 9000.0)


def load_gpx(filepath: str, extensions: list[str] | None = None) -> pd.DataFrame:
    """Load a GPX file and extract data into a :class:`pandas.DataFrame`.

    ``latitude`` and ``longitude`` are mandatory GPX attributes, so
    :mod:`gpxpy` raises :class:`~gpxpy.gpx.GPXException` while parsing if
    either is missing. ``time`` is optional in the GPX schema, so :mod:`gpxpy`
    does *not* validate it. However, it is required by
    :class:`~polkupy.core.ride.Ride`, so it is explicity checked and raises
    :class:`ValueError` if absent. A requested extension that is not found in
    the file raises a :class:`UserWarning`, since the data is still usable
    without the extension.

    Args:
        filepath (str): Path to the GPX file.
        extensions (list[str], optional): Extensions of the GPX file to read,
            e.g. heart rate data.

    Returns:
        pandas.DataFrame: A DataFrame containing the GPX data with columns for
        time, latitude, longitude, and optionally elevation and other specified
        extensions.

    Raises:
        ValueError: If the file has no track points, a point has no
            timestamp, or a point's latitude, longitude, or elevation is
            outside a physically plausible range.
        gpxpy.gpx.GPXException: If the file is missing a mandatory GPX
            field (latitude or longitude).
    """
    extensions = extensions or []
    gpx = _parse_gpx(filepath)

    data = [
        _point_to_dict(point, extensions, filepath)
        for track in gpx.tracks
        for segment in track.segments
        for point in segment.points
    ]
    if not data:
        raise ValueError(f"{filepath}: no track points found")

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])

    for ext in extensions:
        if df[ext].isna().all():
            warnings.warn(
                f"{filepath}: extension {ext!r} was requested but not found "
                "in any track point",
                stacklevel=2,
            )

    return df


def load_gpx_dir(
    data_dir: str = "data",
    bikes: list[str] | None = None,
    extensions: list[str] | None = None,
) -> pd.DataFrame:
    """Load every GPX ride from a directory of per-bike subfolders.

    A file that fails to load is skipped with a :class:`UserWarning`.

    Args:
        data_dir (str): Directory containing one subfolder per bike, each
            holding that bike's GPX files (e.g. ``"data/ktm/*.gpx"``).
        bikes (list[str], optional): Subset of bike subfolder names to
            load. Defaults to every subfolder found in ``data_dir``.
        extensions (list[str], optional): Extra GPX extension fields to
            read, passed through to :func:`load_gpx`.

    Returns:
        pandas.DataFrame: All rides concatenated, with added ``ride_id``
        (unique per GPX file) and ``bike_id`` columns.

    Raises:
        ValueError: If no ride was successfully loaded from ``data_dir``.
    """
    data_path = Path(data_dir)
    bike_dirs = (
        [data_path / bike for bike in bikes]
        if bikes
        else sorted(p for p in data_path.iterdir() if p.is_dir())
    )

    rides = []
    for bike_dir in bike_dirs:
        for gpx_path in sorted(bike_dir.glob("*.gpx")):
            try:
                ride = load_gpx(str(gpx_path), extensions=extensions)
            except (ValueError, gpxpy.gpx.GPXException) as exc:
                warnings.warn(f"Skipping ride: {exc}", stacklevel=2)
                continue
            ride["ride_id"] = f"{bike_dir.name}/{gpx_path.stem}"
            ride["bike_id"] = bike_dir.name
            rides.append(ride)

    if not rides:
        raise ValueError(f"{data_dir}: no rides found")

    return pd.concat(rides, ignore_index=True)


def list_gpx_extensions(filepath: str) -> set[str]:
    """List every extension tag name found across the GPX file's track points.

    See :func:`load_gpx` for how these names are matched (by substring,
    case-insensitively) when using the ``extensions`` argument.

    Args:
        filepath (str): Path to the GPX file.

    Returns:
        set[str]: Every distinct extension tag's local name, e.g.
        ``{"TrackPointExtension", "hr", "cad"}``.

    Raises:
        gpxpy.gpx.GPXException: If the file is missing a mandatory GPX
            field (latitude or longitude).
    """
    gpx = _parse_gpx(filepath)
    return {
        _local_name(child.tag)
        for track in gpx.tracks
        for segment in track.segments
        for point in segment.points
        for ext_element in point.extensions
        for child in ext_element.iter()
    }


def _parse_gpx(filepath: str) -> gpxpy.gpx.GPX:
    """Parse a GPX file, prefixing a raised GPXException with ``filepath``."""
    with Path(filepath).open(encoding="utf-8") as file:
        try:
            return gpxpy.parse(file)
        except gpxpy.gpx.GPXException as exc:
            raise gpxpy.gpx.GPXException(f"{filepath}: {exc}") from exc


def _local_name(tag: str) -> str:
    """Strip a Clark-notation XML namespace (``{uri}name``) down to ``name``."""
    return tag.rsplit("}", 1)[-1]


def _point_to_dict(
    point: gpxpy.gpx.GPXTrackPoint, extensions: list[str], filepath: str
) -> dict[str, Any]:
    """Extract one track point's core fields plus any requested ``extensions``.

    Args:
        point (gpxpy.gpx.GPXTrackPoint): GPX track point
        extensions (list[str]): list of additional data to be loaded (e.g.
            heart rate)
        filepath (str): path to gpx file (for better warnings)

    Returns:
        dict: Dictionary with data for one track point.
    """
    # check timestamp is present (gpxpy doesn't check this)
    time = point.time
    if time is None:
        raise ValueError(
            f"{filepath}: track point at ({point.latitude}, {point.longitude}) "
            "is missing a timestamp"
        )

    # check to ensure that data is plausible
    lat, lon, ele = point.latitude, point.longitude, point.elevation
    _check_range(lat, LATITUDE_RANGE, name="latitude", time=time, filepath=filepath)
    _check_range(lon, LONGITUDE_RANGE, name="longitude", time=time, filepath=filepath)
    _check_range(ele, ELEVATION_RANGE, name="elevation", time=time, filepath=filepath)

    # create dictionary of data
    point_data: dict[str, Any] = {
        "time": time.isoformat(),
        "lat": lat,
        "lon": lon,
        "ele": ele,
        **dict.fromkeys(extensions),
    }
    if point.extensions and extensions:
        point_data.update(_extension_values(point.extensions, extensions))
    return point_data


def _check_range(
    value: float | None,
    bounds: tuple[float, float],
    *,
    name: str,
    time: datetime,
    filepath: str,
) -> None:
    """Checks whether the data is within first-order plausibility.

    Raises a ValueError otherwise.

    Args:
        value (float, optional): value to check, e.g. a latitude. Skipped
            if ``None`` because some fields (e.g. elevation) are not required
            and could thus be missing.
        bounds (tuple[float, float]): ``(low, high)`` (inclusive) range that
            ``value`` must fall within.
        name (str): name for ``value`` used in the error message (e.g.
            ``"latitude"``).
        time (datetime): timestamp of the point being checked, used in the
            error message to identify which point failed.
        filepath (str): path of the GPX file being loaded, used in the
            error message to identify which file failed.

    Raises:
        ValueError: If ``value`` is not ``None`` and falls outside ``bounds``.
    """
    low, high = bounds
    if value is not None and not low <= value <= high:
        raise ValueError(
            f"{filepath}: point at {time.isoformat()} has {name}={value}, "
            f"outside the valid range [{low}, {high}]"
        )


def _extension_values(
    ext_elements: list[Any], extensions: list[str]
) -> dict[str, float | None]:
    """Match each raw extension element's tag against the requested extensions.

    ``extensions`` are matched by substring, and each element's text is
    parsed as a float.

    Args:
        ext_elements (list): Raw ``<extensions>`` child elements from
            :attr:`gpxpy.gpx.GPXTrackPoint.extensions`, e.g. a
            ``TrackPointExtension`` wrapper containing ``hr``, ``cad`` etc.
            tags.
        extensions (list[str]): Extension names requested by the caller
            (see :func:`load_gpx`). Matching is by substring against the
            lowercased XML tag name, so requesting e.g. ``"hr"`` matches a
            tag like ``gpxtpx:hr``.

    Returns:
        dict[str, float | None]: Requested extension name -> parsed float
        value (or ``None`` if its text wasn't numeric).
    """
    values: dict[str, float | None] = {}
    children = (child for ext_element in ext_elements for child in ext_element.iter())
    for child in children:
        tag_name = child.tag.lower()
        for ext in extensions:
            if ext.lower() in tag_name:
                values[ext] = _as_float(child.text)
    return values


def _as_float(value: str | None) -> float | None:
    """Try to convert input to a float value."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
