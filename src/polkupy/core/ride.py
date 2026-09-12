"""Provides the :class:`Ride` class.

This is the base of the whole polkupy package. The :class:`Ride` is a single
ride as a chainable wrapper around a :class:`pandas.DataFrame` of points, which
can currently be read in from a GPX file (more options will be added in the
future).
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import polkupy functions
from ..clock import add_time_of_day
from ..geo import calc_distance, calc_speed, haversine
from ..io.fit import load_fit
from ..io.gpx import load_gpx

if TYPE_CHECKING:
    from .trips import Trips


class Ride:
    """A single ride.

    The class includes the points of one track plus its metadata. Rides
    are the building block of all processing methods, built on top of a
    :class:`pandas.DataFrame`. The minimum required columns are:

    - ``time``: timezone-aware timestamp
    - ``lat``: latitude in degrees
    - ``lon``: longitude in degrees

    Optional but highly recommended are:

    - ``bike_id``: an identifier for the bicycle used
    - ``ride_id``: an identifier for the ride itself

    Any other data or metadata is optional and simply carried along on
    :attr:`data` untouched.

    Methods that transform the data return a new :class:`Ride` rather than
    mutating in place, so they can be chained, e.g.
    ``ride.localized().with_speed().smoothed()``. The underlying DataFrame
    is always available as :attr:`data`.
    """

    REQUIRED_COLUMNS = frozenset({"time", "lat", "lon"})

    def __init__(
        self, data: pd.DataFrame, ride_id: str | None = None, bike_id: str | None = None
    ):
        """Wrap `data` as a :class:`Ride`.

        Args:
            data (pandas.DataFrame): Points for one ride. Must contain the
                columns in :attr:`REQUIRED_COLUMNS`. If it also has
                ``ride_id``/``bike_id`` columns, their first value is used
                as a fallback whenever `ride_id`/`bike_id` are not passed
                explicitly.
            ride_id (str, optional): Identifier for the ride. Falls back
                to `data`'s ``ride_id`` column, then ``None``.
            bike_id (str, optional): Identifier for the bicycle used.
                Falls back to `data`'s ``bike_id`` column, then ``None``.

        Raises:
            ValueError: If `data` is missing any of :attr:`REQUIRED_COLUMNS`.
        """
        # ensure required columns exist
        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(
                f"Ride data is missing required column(s): {', '.join(sorted(missing))}"
            )

        # load inputs into class
        data = data.reset_index(drop=True)
        self.ride_id = ride_id or (
            data["ride_id"].iloc[0] if "ride_id" in data.columns and len(data) else None
        )
        self.bike_id = bike_id or (
            data["bike_id"].iloc[0] if "bike_id" in data.columns and len(data) else None
        )
        if "ride_id" not in data.columns and self.ride_id is not None:
            data = data.assign(ride_id=self.ride_id)
        self.data = data

    # -- special methods ------------------------------------------------------

    def __add__(self, other: Literal[0] | Ride | Trips) -> Trips:
        """Concatenate this ride with another ride or collection.

        Accepts ``0`` as well as :class:`Ride`/:class:`Trips` so that
        :func:`sum` works.

        Args:
            other (Literal[0] | Ride | Trips): Another ride or collection to
                concatenate onto this one, or ``0``.

        Returns:
            Trips: If ``other`` is ``0``, a :class:`Trips` wrapping just this
            ride. Otherwise, a new :class:`Trips` collection holding every
            point from both ``self`` and ``other``.
        """
        # imported here to prevent recursion
        from .trips import Trips  # noqa: PLC0415

        # convert Ride -> Trips if there is nothing to add to
        if other == 0:
            return Trips(self.data)

        return Trips.from_rides([self, other])

    def __radd__(self, other: Literal[0] | Ride | Trips) -> Trips:
        """Reflected concatenation.

        Args:
            other (Literal[0] | Ride | Trips): Another ride or collection to
                concatenate onto this one, or ``0``.

        Returns:
            Trips: See :meth:`__add__`.
        """
        return self + other

    def __len__(self) -> int:
        """The number of points in :attr:`data`.

        Returns:
            int: ``len(self.data)``.
        """
        return len(self.data)

    # -- jupyter viewing ------------------------------------------------------

    def _repr_png_(self) -> bytes:
        """Render the route as a small map, as PNG bytes.

        The start and end points are marked with a green "play" triangle and a
        red "stop" square respectively.
        """
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_aspect(1 / np.cos(np.radians(self.data["lat"].mean())))
        ax.axis("off")
        ax.plot(self.data["lon"], self.data["lat"], lw=0.9, color="#9CA986", zorder=1)
        start, end = self.data.iloc[0], self.data.iloc[-1]
        ax.plot(
            start["lon"],
            start["lat"],
            marker=">",
            markersize=7,
            linestyle="none",
            color="#2E7D32",
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.plot(
            end["lon"],
            end["lat"],
            marker="s",
            markersize=6,
            linestyle="none",
            color="#C62828",
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        return buf.getvalue()

    def _repr_html_(self) -> str:
        """Render ride metadata plus a small map.

        This means that a :class:`Ride` displays as a summary card just by
        being the last line of a Jupyter cell.
        """
        items: list[str] = []
        if self.ride_id is not None:
            items.append(f"<b>ride ID</b>: {self.ride_id}")
        if self.bike_id is not None:
            items.append(f"<b>bike ID</b>: {self.bike_id}")
        items += [
            f"<b>start</b>: {self.start_time}",
            f"<b>end</b>: {self.end_time}",
            f"<b>duration</b>: {self.duration}",
            f"<b>sampling rate</b>: {self.sampling_rate.total_seconds():.1f} s",
        ]
        bullets = "".join(f"<li>{item}</li>" for item in items)
        img_b64 = base64.b64encode(self._repr_png_()).decode("ascii")
        img_style = "margin-top: 0.5em;"
        return (
            "<div><h3>Ride</h3>"
            f"<ul>{bullets}</ul>"
            f'<img src="data:image/png;base64,{img_b64}" style="{img_style}"/></div>'
        )

    # -- data entry -----------------------------------------------------------

    @classmethod
    def from_gpx(
        cls,
        filepath: str,
        bike_id: str | None = None,
        ride_id: str | None = None,
        extensions: list[str] | None = None,
    ) -> Ride:
        """Load a single GPX file as a :class:`Ride`.

        See :func:`~polkupy.io.gpx.load_gpx` for how the file itself is
        parsed and validated.

        Args:
            filepath (str): Path to the GPX file.
            bike_id (str, optional): Identifier for the bicycle used.
            ride_id (str, optional): Identifier for the ride. Defaults to
                ``f"{bike_id}/{stem}"`` if ``bike_id`` is given, otherwise
                just the filename stem.
            extensions (list[str], optional): Extra GPX extension fields to
                read (e.g. heart rate), passed through to
                :func:`~polkupy.io.gpx.load_gpx`.

        Returns:
            Ride: The ride loaded from `filepath`.
        """
        data = load_gpx(filepath, extensions=extensions)

        # add bike and ride IDs if provided
        if bike_id is not None:
            data["bike_id"] = bike_id
        if ride_id is None:
            ride_id = (
                f"{bike_id}/{Path(filepath).stem}" if bike_id else Path(filepath).stem
            )
        return cls(data, ride_id=ride_id, bike_id=bike_id)

    @classmethod
    def from_fit(
        cls,
        filepath: str,
        bike_id: str | None = None,
        ride_id: str | None = None,
        extensions: list[str] | None = None,
    ) -> Ride:
        """Load a single FIT file as a :class:`Ride`.

        See :func:`~polkupy.io.fit.load_fit` for how the file itself is
        parsed and validated.

        Args:
            filepath (str): Path to the FIT file.
            bike_id (str, optional): Identifier for the bicycle used.
            ride_id (str, optional): Identifier for the ride. Defaults to
                ``f"{bike_id}/{stem}"`` if ``bike_id`` is given, otherwise
                just the filename stem.
            extensions (list[str], optional): Extra FIT fields to read (e.g.
                heart rate), passed through to :func:`~polkupy.io.fit.load_fit`.

        Returns:
            Ride: The ride loaded from `filepath`.
        """
        data = load_fit(filepath, extensions=extensions)

        # add bike and ride IDs if provided
        if bike_id is not None:
            data["bike_id"] = bike_id
        if ride_id is None:
            ride_id = (
                f"{bike_id}/{Path(filepath).stem}" if bike_id else Path(filepath).stem
            )
        return cls(data, ride_id=ride_id, bike_id=bike_id)

    # -- metadata -------------------------------------------------------------

    @property
    def start_time(self) -> pd.Timestamp:
        """The timestamp of the first point, as a :class:`pandas.Timestamp`.

        Raises:
            IndexError: If :attr:`data` has no points.
        """
        return self.data["time"].iloc[0]

    @property
    def end_time(self) -> pd.Timestamp:
        """The timestamp of the last point, as a :class:`pandas.Timestamp`.

        Raises:
            IndexError: If :attr:`data` has no points.
        """
        return self.data["time"].iloc[-1]

    @property
    def duration(self) -> pd.Timedelta:
        """Elapsed wall-clock time from :attr:`start_time` to :attr:`end_time`.

        The output is a :class:`pandas.Timedelta`. This is total elapsed time,
        not moving time. It includes any time spent stopped (e.g. at traffic
        lights).
        """
        return self.end_time - self.start_time

    @property
    def sampling_rate(self) -> pd.Timedelta:
        """Typical interval between consecutive points.

        Calculated as the median of per-point time differences, as a
        :class:`pandas.Timedelta`.
        """
        return pd.Timedelta(self.data["time"].diff().median())

    @property
    def distance_km(self) -> float:
        """Total distance covered in km.

        Calculated as the cumulative great-circle distance between consecutive
        points.
        """
        lat, lon = self.data["lat"].to_numpy(), self.data["lon"].to_numpy()
        if len(lat) < 2:  # noqa: PLR2004
            return 0.0
        dist_m = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
        return float(np.nansum(dist_m)) / 1e3

    # -- geography ------------------------------------------------------------

    def starts_near(
        self, lat: float, lon: float, radius_km: float = 0.5, n_points: int = 5
    ) -> bool:
        """Whether the ride starts near a given point.

        See :meth:`_near` for how "near" is checked.

        Args:
            lat (float): target latitude, in degrees.
            lon (float): target longitude, in degrees.
            radius_km (float): how close counts as "near" (in km) as the
                great-circle distance to the target point. Defaults to
                ``0.5``.
            n_points (int): how many of the ride's first points to check.
                Defaults to ``5``.

        Returns:
            bool: Whether any of the first ``n_points`` points fall within
            ``radius_km`` of ``(lat, lon)``.
        """
        return self._near(self.data.iloc[:n_points], lat, lon, radius_km)

    def ends_near(
        self, lat: float, lon: float, radius_km: float = 0.5, n_points: int = 5
    ) -> bool:
        """Whether the ride ends near a given point.

        See :meth:`_near` for how "near" is checked.

        Args:
            lat (float): target latitude, in degrees.
            lon (float): target longitude, in degrees.
            radius_km (float): how close counts as "near" (in km) as the
                great-circle distance to the target point. Defaults to
                ``0.5``.
            n_points (int): how many of the ride's first points to check.
                Defaults to ``5``.

        Returns:
            bool: Whether any of the last ``n_points`` points fall within
            ``radius_km`` of ``(lat, lon)``.
        """
        return self._near(self.data.iloc[-n_points:], lat, lon, radius_km)

    @staticmethod
    def _near(points: pd.DataFrame, lat: float, lon: float, radius_km: float) -> bool:
        """Whether ``points`` is near ``(lat, lon)``.

        This uses a true circular radius calculated via :func:`~polkupy.geo.haversine`.
        "near" is satisfied as soon as *any single* row is within ``radius_km``
        great-circle distance of ``(lat, lon)``.

        Args:
            points (pandas.DataFrame): points to check, with ``lat`` & ``lon``
                columns.
            lat (float): target latitude, in degrees.
            lon (float): target longitude, in degrees.
            radius_km (float): how close counts as "near" (in km).

        Returns:
            bool: Whether ``points`` is near ``(lat, lon)``.
        """
        distance_m = haversine(points["lon"], points["lat"], lon, lat)
        return bool((distance_m / 1000 <= radius_km).any())

    # -- transforms (return a new Ride) ---------------------------------------

    def _with_data(self, data: pd.DataFrame) -> Ride:
        """Build a new :class:`Ride` from ``data``.

        This function preserves this ride's identity (:attr:`ride_id`/:attr:`bike_id`)
        regardless of whether ``data`` happens to carry them as columns.

        Args:
            data (pandas.DataFrame): Replacement data, e.g. from one of the
                ``polkupy.geo``/``polkupy.clock`` transform functions.

        Returns:
            Ride: A new :class:`Ride` wrapping ``data``.
        """
        return Ride(data, ride_id=self.ride_id, bike_id=self.bike_id)

    def localised(self, tz: str = "Europe/Berlin") -> Ride:
        """Align the ride to a 24h clock, ignoring the calendar date.

        See :func:`~polkupy.clock.add_time_of_day`.

        Args:
            tz (str): IANA timezone name used to resolve local time of day.

        Returns:
            Ride: A :class:`Ride` with ``time`` converted to ``tz``, and
            ``ride_start_s`` and ``virtual_s`` columns added.
        """
        return self._with_data(add_time_of_day(self.data, tz=tz))

    def with_distance(self, *, overwrite: bool = False) -> Ride:
        """Add ``dist_km`` (km from the previous point) column.

        See :func:`~polkupy.geo.calc_distance`.

        Args:
            overwrite (bool): If ``True``, recompute ``dist_km`` from GPS
                even if already present. Defaults to ``False``, so sensor data
                is kept as-is.

        Returns:
            Ride: A :class:`Ride` with ``dist_km`` column added. The first
            point has no previous point, so it is ``NaN``.
        """
        return self._with_data(calc_distance(self.data, overwrite=overwrite))

    def with_speed(self, *, overwrite: bool = False) -> Ride:
        """Add the ``dist_km`` (if not already present) and ``speed_kmh`` columns.

        See :func:`~polkupy.geo.calc_speed`.

        Args:
            overwrite (bool): If ``True``, recompute ``dist_km``/``speed_kmh``
                from GPS even if already present. Defaults to ``False``, so
                sensor data is kept as-is.

        Returns:
            Ride: A :class:`Ride` with ``dist_km`` and ``speed_kmh`` columns
            added. The first point has no previous point, so both are
            ``NaN``.
        """
        return self._with_data(calc_speed(self.data, overwrite=overwrite))
