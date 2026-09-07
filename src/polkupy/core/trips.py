"""
Provides the :class:`Trips` class: a collection of rides, backed by one
concatenated :class:`pandas.DataFrame` grouped by ``ride_id``.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Literal

import pandas as pd

# import polkupy functions
from ..algorithms.filters import filter_rides_in_bbox
from ..clock import add_time_of_day
from ..io.gpx import load_gpx_dir
from .ride import Ride


class Trips:
    """A collection of rides, backed by one :class:`pandas.DataFrame`
    grouped by ``ride_id``.

    Iterating yields :class:`Ride` objects; methods that filter or
    transform the collection return a new :class:`Trips` so they can be
    chained.
    """

    REQUIRED_COLUMNS = Ride.REQUIRED_COLUMNS | {"ride_id"}

    def __init__(self, data: pd.DataFrame):
        """Wrap ``data`` as a :class:`Trips` collection.

        Args:
            data (pandas.DataFrame): Points for one or more rides. Must
                contain the columns in :attr:`REQUIRED_COLUMNS` (the same as
                for :class:`Ride`), plus ``ride_id`` identifying which ride
                each row belongs to.

        Raises:
            ValueError: If ``data`` is missing any of :attr:`REQUIRED_COLUMNS`.
        """
        missing = self.REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(
                f"Trips data is missing required column(s): {', '.join(sorted(missing))}"
            )
        self.data = data

    # -- data entry -----------------------------------------------------------

    @classmethod
    def from_gpx(
        cls,
        data_dir: str = "data",
        bikes: list[str] | None = None,
        extensions: list[str] | None = None,
    ) -> "Trips":
        """Load every GPX ride from a directory of per-bike subfolders as a
        :class:`Trips` collection. This will need to be updated to become more
        flexible in the future.

        Args:
            data_dir (str): Directory containing one subfolder per bike.
            bikes (list[str], optional): Subset of bike subfolder names to
                load. Defaults to every subfolder found in ``data_dir``.
            extensions (list[str], optional): Extra GPX extension fields to
                read, passed through to
                :func:`~polkupy.io.gpx.load_gpx_dir`.

        Returns:
            Trips: The rides loaded from ``data_dir``.
        """
        return cls(load_gpx_dir(data_dir, bikes=bikes, extensions=extensions))

    @classmethod
    def from_rides(cls, rides: "Iterable[Ride | Trips]") -> "Trips":
        """Concatenate several :class:`Ride`/:class:`Trips` into one
        :class:`Trips` collection.

        Args:
            rides (Iterable[Ride | Trips]): :class:`Ride`s and/or
                :class:`Trips` collections to concatenate, e.g. from
                ``ride1 + ride2``.

        Returns:
            Trips: A new collection holding every ride.
        """
        return cls(pd.concat([r.data for r in rides], ignore_index=True))

    # -- special methods ------------------------------------------------------

    def __add__(self, other: Literal[0] | Ride | "Trips") -> "Trips":
        """Concatenate this collection with another ride or collection.
        Accepts ``0`` as well as :class:`Ride`/:class:`Trips` so that
        :func:`sum` works, e.g. ``sum(list_of_rides)``.

        Args:
            other (Literal[0] | Ride | Trips): Another ride or collection to
                concatenate onto this one, or ``0``.

        Returns:
            Trips: ``self`` if ``other`` is ``0``. Otherwise, a new
            :class:`Trips` collection holding every point from both
            ``self`` and ``other``.
        """
        if other == 0:
            return self
        return Trips.from_rides([self, other])

    def __radd__(self, other: Literal[0] | Ride | "Trips") -> "Trips":
        """Reflected concatenation.

        Args:
            other (Literal[0] | Ride | Trips): Another ride or collection to
                concatenate onto this one, or ``0``.

        Returns:
            Trips: See :meth:`__add__`.
        """
        return self + other

    def __len__(self) -> int:
        """The number of distinct rides in this collection.

        Returns:
            int: The number of unique ``ride_id`` values in :attr:`data`.
        """
        return self.data["ride_id"].nunique()

    def __iter__(self) -> Iterator[Ride]:
        """Iterate over the rides in this collection.

        Yields:
            Ride: One :class:`Ride` per distinct ``ride_id`` in :attr:`data`.
        """
        for ride_id, group in self.data.groupby("ride_id", sort=False):
            yield Ride(group, ride_id=str(ride_id))

    def __getitem__(self, ride_id: str) -> Ride:
        """Look up a single ride by id.

        Args:
            ride_id (str): The ``ride_id`` to look up.

        Returns:
            Ride: The matching ride.

        Raises:
            KeyError: If no row in :attr:`data` has this `ride_id`.
        """
        subset = self.data[self.data["ride_id"] == ride_id]
        if subset.empty:
            raise KeyError(ride_id)
        return Ride(subset, ride_id=ride_id)

    def __repr__(self) -> str:
        return f"Trips({len(self)} rides, {len(self.data)} points)"

    # -- jupyter viewing ------------------------------------------------------

    _REPR_MAX_ROWS = 15

    def _repr_html_(self) -> str:
        """Render collection metadata plus a per-ride table, so a
        :class:`Trips` displays as a summary card just by being the last
        line of a Jupyter cell."""
        return (
            "<div><h3>Trips</h3>"
            f"<ul>{self._repr_bullets()}</ul>"
            f"{self._repr_ride_table()}</div>"
        )

    def _repr_bullets(self) -> str:
        """The ``<li>`` summary bullets (ride/bike counts, start/end)."""
        has_bike_id = "bike_id" in self.data.columns
        items = [f"<b>rides</b>: {len(self)}"]
        if has_bike_id:
            items.append(f"<b>bikes</b>: {self.data['bike_id'].nunique()}")
        items += [
            f"<b>start</b>: {self.data['time'].min()}",
            f"<b>end</b>: {self.data['time'].max()}",
        ]
        return "".join(f"<li>{item}</li>" for item in items)

    def _repr_ride_table(self) -> str:
        """The per-ride ``<table>`` (ride ID, bike ID, point count), capped
        at :attr:`_REPR_MAX_ROWS` rides with a "... and N more" footer."""
        has_bike_id = "bike_id" in self.data.columns
        counts = self.data.groupby("ride_id").size()
        bike_ids = (
            self.data.groupby("ride_id")["bike_id"].first() if has_bike_id else None
        )

        ride_ids = list(counts.index)
        rows = [
            self._repr_ride_row(ride_id, counts.loc[ride_id], bike_ids)
            for ride_id in ride_ids[: self._REPR_MAX_ROWS]
        ]

        footer = ""
        if len(ride_ids) > self._REPR_MAX_ROWS:
            remaining = len(ride_ids) - self._REPR_MAX_ROWS
            footer = f'<p style="margin-top: 0.5em;"><em>... and {remaining} more rides</em></p>'

        cell_style = 'style="text-align: left; padding: 2px 1em 2px 0;"'
        return (
            '<table style="margin-top: 0.5em; border-collapse: collapse;">'
            f"<thead><tr><th {cell_style}>ride ID</th><th {cell_style}>bike ID</th>"
            f'<th style="text-align: left; padding: 2px 0;">count</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>{footer}"
        )

    @staticmethod
    def _repr_ride_row(ride_id: str, count: int, bike_ids: "pd.Series | None") -> str:
        """One ``<tr>`` of the per-ride table."""
        bike_id = bike_ids.loc[ride_id] if bike_ids is not None else None
        bike_id = "" if bike_id is None or pd.isna(bike_id) else bike_id
        return f"<tr><td>{ride_id}</td><td>{bike_id}</td><td>{count}</td></tr>"

    # -- filtering (return a new Trips) ---------------------------------------

    def filter(self, predicate: Callable[[Ride], bool]) -> "Trips":
        """Keep only rides for which `predicate(ride)` is True.

        Args:
            predicate (Callable[[Ride], bool]): Called once per ride (via
                :meth:`__iter__`).

        Returns:
            Trips: A new collection holding only the matching rides.
        """
        keep_ids = [ride.ride_id for ride in self if predicate(ride)]
        return Trips(self.data[self.data["ride_id"].isin(keep_ids)])

    def in_bbox(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> "Trips":
        """Keep rides that pass through a geographic bounding box. See
        :func:`~polkupy.algorithms.filters.filter_rides_in_bbox`.

        Args:
            lat_min (float): minimum latitude of the box, in degrees.
            lat_max (float): maximum latitude of the box, in degrees.
            lon_min (float): minimum longitude of the box, in degrees.
            lon_max (float): maximum longitude of the box, in degrees.

        Returns:
            Trips: Rides that have at least one point inside the box, kept
            in full.
        """
        return Trips(
            filter_rides_in_bbox(self.data, lat_min, lat_max, lon_min, lon_max)
        )

    def between(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        radius_km: float = 0.5,
    ) -> "Trips":
        """Keep rides that start near ``start`` (lat, lon) and end near ``end``.
        Useful e.g. for locating groups of similar rides, e.g. a commutes. See
        :meth:`Ride.starts_near`/:meth:`Ride.ends_near` for how "near" is
        checked.

        Args:
            start (tuple[float, float]): target ``(lat, lon)`` the ride
                must start near.
            end (tuple[float, float]): target ``(lat, lon)`` the ride
                must end near.
            radius_km (float): how close counts as "near", in km.

        Returns:
            Trips: Rides that both start near ``start`` and end near ``end``.
        """
        return self.filter(
            lambda ride: ride.starts_near(*start, radius_km)
            and ride.ends_near(*end, radius_km)
        )

    # -- transforms (return a new Trips) --------------------------------------

    def localised(self, tz: str = "Europe/Berlin") -> "Trips":
        """Align every ride to a shared 24h clock, ignoring the calendar
        date.

        See :func:`~polkupy.clock.add_time_of_day`.

        Args:
            tz (str): IANA timezone name used to resolve local time of day.

        Returns:
            Trips: A new collection with ``time`` converted to ``tz``, and
            ``ride_start_s``/``virtual_s`` columns added.
        """
        return Trips(add_time_of_day(self.data, tz=tz))
