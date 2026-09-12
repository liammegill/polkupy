"""Open and parse .fit files."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import fitdecode
import pandas as pd

# maybe we should have a generic module to load these from?
from .gpx import ELEVATION_RANGE, LATITUDE_RANGE, LONGITUDE_RANGE, _check_range

_LAT_FIELD = "position_lat"
_LON_FIELD = "position_long"
_ELEVATION_FIELDS = ("enhanced_altitude", "altitude")
_DISTANCE_FIELD = "distance"  # km from fitdecode
_SPEED_FIELDS = ("enhanced_speed", "speed")  # km/h from fitdecode


def load_fit(filepath: str, extensions: list[str] | None = None) -> pd.DataFrame:
    """Load a FIT file and extract data into a :class:`pandas.DataFrame`.

    This function uses :mod:`fitdecode` to decode the input. ``timestamp``
    becomes `time`, ``position_lat``/``position_long`` become ``lat``/``lon`
    (converted from semicircles to degrees by
    :class:`fitdecode.StandardUnitsDataProcessor`), and
    ``enhanced_altitude``/``altitude`` are coalesced into ``ele``. A
    requested extension that is not found in the file raises a
    :class:`UserWarning`, since the data is still usable without it.

    If the file records its own ``distance`` and/or ``speed``/``enhanced_speed``
    fields, they are read into `dist_km`/`speed_kmh` too, the same columns
    :meth:`Ride.with_distance <polkupy.core.ride.Ride.with_distance>` and
    :meth:`Ride.with_speed <polkupy.core.ride.Ride.with_speed>` would
    otherwise compute from GPS. Those methods keep this sensor-reported
    data as-is unless called with ``overwrite=True``.

    Although FIT is a common protocol, it isn't used the same way by all
    devices. For example, the ``record`` messages aren't guaranteed to carry
    every field at every timestamp: some devices split a single sample across
    two ``record`` messages that share a timestamp, e.g. one with GPS position
    and another with cadence/power. This function thus merges same-timestamp
    messages before dropping points that still lack a GPS fix afterwards
    (since the point of this package is to analyse location data). This could
    in theory be expanded later, but it's not a priority.

    Args:
        filepath (str): Path to the FIT file.
        extensions (list[str], optional): Extra field names to read, e.g.
            ``"heart_rate"`` or ``"power"``. Matched by substring
            (case-insensitive) against the record fields' names, so e.g.
            requesting ``"speed"`` also matches ``enhanced_speed``. If several
            fields match, the first non-null value per point wins.

    Returns:
        pandas.DataFrame: A DataFrame containing the FIT data with columns
        for time, latitude, longitude, and optionally elevation and other
        specified extensions.

    Raises:
        ValueError: If the file has no ``record`` message, none of them has
            a GPS fix, or a point's latitude, longitude, or elevation is
            outside a physically plausible range.
        fitdecode.FitError: If the file is corrupt or malformed.
    """
    extensions = extensions or []
    rows = _read_records(filepath)

    if not rows:
        raise ValueError(f"{filepath}: no record messages found")

    # merge same-timestamp messages
    raw = pd.DataFrame(rows)
    merged = raw.groupby("timestamp", as_index=False).first()

    # ensure at least one record has a GPS fix
    if _LAT_FIELD not in merged.columns or _LON_FIELD not in merged.columns:
        raise ValueError(f"{filepath}: no record message has a GPS fix")
    merged = merged.dropna(subset=[_LAT_FIELD, _LON_FIELD]).reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"{filepath}: no record message has a GPS fix")

    # create dataframe, coalesce elevation
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(merged["timestamp"]),
            "lat": merged[_LAT_FIELD],
            "lon": merged[_LON_FIELD],
            "ele": _coalesce(merged, _ELEVATION_FIELDS),
        }
    )

    # add distance and speed (if present)
    if _DISTANCE_FIELD in merged.columns:
        df["dist_km"] = merged[_DISTANCE_FIELD].diff()
    if any(field in merged.columns for field in _SPEED_FIELDS):
        df["speed_kmh"] = _coalesce(merged, _SPEED_FIELDS)

    for raw_time, raw_lat, raw_lon, raw_ele in zip(
        df["time"], df["lat"], df["lon"], df["ele"], strict=True
    ):
        point_time = raw_time.to_pydatetime()
        ele = None if pd.isna(raw_ele) else float(raw_ele)
        _check_range(
            float(raw_lat),
            LATITUDE_RANGE,
            name="latitude",
            time=point_time,
            filepath=filepath,
        )
        _check_range(
            float(raw_lon),
            LONGITUDE_RANGE,
            name="longitude",
            time=point_time,
            filepath=filepath,
        )
        _check_range(
            ele, ELEVATION_RANGE, name="elevation", time=point_time, filepath=filepath
        )

    for ext in extensions:
        matches = [c for c in merged.columns if ext.lower() in c.lower()]
        df[ext] = _coalesce(merged, matches) if matches else None
        if df[ext].isna().all():
            warnings.warn(
                f"{filepath}: extension {ext!r} was requested but not found "
                "in any record message",
                stacklevel=2,
            )

    return df


def list_fit_fields(filepath: str) -> set[str]:
    """List every field name found across the FIT file's ``record`` messages.

    A device may redefine which fields a ``record`` message carries partway
    through a file (this is also how the same-timestamp splitting described
    in :func:`load_fit` happens), so this reads the whole file rather than
    just its first message.

    Args:
        filepath (str): Path to the FIT file.

    Returns:
        set[str]: Every distinct field name seen, e.g.
        ``{"timestamp", "position_lat", "position_long", "heart_rate", ...}``.
        ``timestamp``, ``position_lat``, ``position_long`` and, if present,
        ``altitude``, ``distance`` and ``speed`` (or the "enhanced" versions
        thereof) are already read automatically by :func:`load_fit` and don't
        need to be provided to that function.

    Raises:
        fitdecode.FitError: If the file is corrupt or malformed.
    """
    return {name for row in _read_records(filepath) for name in row}


def _read_records(filepath: str) -> list[dict[str, object]]:
    """Read every FIT ``record`` message's fields as ``{field_name: value}``."""
    with fitdecode.FitReader(
        filepath, processor=fitdecode.StandardUnitsDataProcessor()
    ) as fit:
        return [
            {field.name: field.value for field in frame.fields}
            for frame in fit
            if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record"
        ]


def load_fit_dir(
    data_dir: str = "data",
    bikes: list[str] | None = None,
    extensions: list[str] | None = None,
) -> pd.DataFrame:
    """Load every FIT ride from a directory of per-bike subfolders.

    A file that fails to load is skipped with a :class:`UserWarning`.

    Args:
        data_dir (str): Directory containing one subfolder per bike, each
            holding that bike's FIT files (e.g. ``"data/ktm/*.fit"``).
        bikes (list[str], optional): Subset of bike subfolder names to
            load. Defaults to every subfolder found in ``data_dir``.
        extensions (list[str], optional): Extra FIT fields to read, passed
            through to :func:`load_fit`.

    Returns:
        pandas.DataFrame: All rides concatenated, with added ``ride_id``
        (unique per FIT file) and ``bike_id`` columns.

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
        for fit_path in sorted(bike_dir.glob("*.fit")):
            try:
                ride = load_fit(str(fit_path), extensions=extensions)
            except (ValueError, fitdecode.FitError) as exc:
                warnings.warn(f"Skipping ride: {exc}", stacklevel=2)
                continue
            ride["ride_id"] = f"{bike_dir.name}/{fit_path.stem}"
            ride["bike_id"] = bike_dir.name
            rides.append(ride)

    if not rides:
        raise ValueError(f"{data_dir}: no rides found")

    return pd.concat(rides, ignore_index=True)


def _coalesce(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """First non-null value across ``columns``, row-wise.

    Args:
        df (pandas.DataFrame): DataFrame to pull ``columns`` from.
        columns (Sequence[str]): Column names to coalesce, in priority
            order. Names not present in ``df`` are ignored.

    Returns:
        pandas.Series: The coalesced values, or all-``NaN`` if none of
        ``columns`` are present in ``df``.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    result = df[present[0]]
    for c in present[1:]:
        result = result.combine_first(df[c])
    return result
