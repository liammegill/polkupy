"""
Provides the :class:`Ride` class: a single ride as a chainable wrapper
around a :class:`pandas.DataFrame` of points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TYPE_CHECKING

import pandas as pd

# import polkupy functions
from ..geo import haversine
from ..io.gpx import load_gpx

if TYPE_CHECKING:
    from .rides import Rides


class Ride:
    """A single ride: the points of one GPX track, plus its metadata. Rides
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

    def __add__(self, other: Literal[0] | Ride | "Rides") -> "Rides":
        """Concatenate this ride with another ride or collection. Accepts ``0``
        as well as :class:`Ride`/:class:`Rides` so that :func:`sum` works.

        Args:
            other (Literal[0] | Ride | Rides): Another ride or collection to
                concatenate onto this one, or ``0``.

        Returns:
            Rides: If `other` is ``0``, a :class:`Rides` wrapping just this
            ride. Otherwise, a new :class:`Rides` collection holding every
            point from both `self` and `other`.
        """
        # imported here to prevent recursion
        from .rides import Rides  # pylint: disable=import-outside-toplevel

        # convert Ride -> Rides if there is nothing to add to
        if other == 0:
            return Rides(self.data)

        return Rides.from_rides([self, other])

    def __radd__(self, other: Literal[0] | Ride | "Rides") -> "Rides":
        """Reflected concatenation.

        Args:
            other (Literal[0] | Ride | Rides): Another ride or collection to
                concatenate onto this one, or ``0``.

        Returns:
            Rides: See :meth:`__add__`.
        """
        return self + other

    def __len__(self) -> int:
        """The number of points in :attr:`data`.

        Returns:
            int: ``len(self.data)``.
        """
        return len(self.data)

    # -- data entry -----------------------------------------------------------

    @classmethod
    def from_gpx(
        cls,
        filepath: str,
        bike_id: str | None = None,
        ride_id: str | None = None,
        extensions: list[str] | None = None,
    ) -> "Ride":
        """Load a single GPX file as a :class:`Ride`.

        See :func:`~polkupy.io.gpx.load_gpx` for how the file itself is
        parsed and validated.

        Args:
            filepath (str): Path to the GPX file.
            bike_id (str, optional): Identifier for the bicycle used.
            ride_id (str, optional): Identifier for the ride. Defaults to
                ``f"{bike_id}/{stem}"`` if `bike_id` is given, otherwise
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
    def from_fit(cls) -> "Ride":
        """Load a single FIT file as a :class:`Ride`.

        Raises:
            NotImplementedError: Always. FIT import isn't built yet.
        """
        raise NotImplementedError

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
        """Elapsed wall-clock time from :attr:`start_time` to
        :attr:`end_time`, as a :class:`pandas.Timedelta`.

        This is total elapsed time, not moving time. It includes any
        time spent stopped (e.g. at traffic lights).
        """
        return self.end_time - self.start_time

    # -- geography ------------------------------------------------------------

    def starts_near(
        self, lat: float, lon: float, radius_km: float = 0.5, n_points: int = 5
    ) -> bool:
        """Whether the ride starts near a given point. See :meth:`_near` for
        how "near" is checked.

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
        """Whether the ride ends near a given point. See :meth:`_near` for how
        "near" is checked.

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
        """Whether ``points`` is near ``(lat, lon)``. This uses a true circular
        radius calculated via :func:`~polkupy.geo.haversine`. "near" is
        satisfied as soon as *any single* row is within ``radius_km``
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
