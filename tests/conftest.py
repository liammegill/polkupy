"""Fixtures shared across the whole test suite."""

# fixtures depending on other fixtures (ride_factory -> ride/trips_factory ->
# trips) is standard pytest composition, not a real shadowing bug.
# pylint: disable=redefined-outer-name

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from polkupy.core.ride import Ride
from polkupy.core.trips import Trips

GPX_HEADER = (
    '<?xml version="1.0"?>\n'
    '<gpx version="1.1" creator="test" '
    'xmlns="http://www.topografix.com/GPX/1/1">\n'
)


@pytest.fixture
def write_gpx():
    """Factory for writing a minimal one-track, one-segment GPX file.

    The file is written at an arbitrary path (parent directories are
    created as needed).
    """

    def _write(path: Path, trkpts: str) -> str:
        xml = f"{GPX_HEADER}<trk><trkseg>{trkpts}</trkseg></trk></gpx>"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml, encoding="utf-8")
        return str(path)

    return _write


def _points(
    n: int = 10,
    lat_start: float = 52.0,
    lon: float = 13.0,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    """``n`` points, 1 degree of latitude and 1000s apart at constant longitude.

    This is the same known-value shape used in tests/test_geo.py, so each
    consecutive gap is exactly ``EARTH_RADIUS_M * radians(1)`` metres.
    """
    start_ts = pd.Timestamp(start)
    time = start_ts + pd.to_timedelta([i * 1000 for i in range(n)], unit="s")
    return pd.DataFrame(
        {
            "time": time,
            "lat": [lat_start + i for i in range(n)],
            "lon": [lon] * n,
        }
    )


@pytest.fixture
def ride_factory():
    """Factory for building a :class:`Ride` with custom identity/geography."""

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def _make(  # noqa: PLR0913, PLR0917 - one arg per Ride field
        ride_id: str = "ride1",
        bike_id: str | None = None,
        n: int = 10,
        lat_start: float = 52.0,
        lon: float = 13.0,
        start: str = "2026-01-01T00:00:00Z",
    ) -> Ride:
        return Ride(_points(n, lat_start, lon, start), ride_id=ride_id, bike_id=bike_id)

    return _make


@pytest.fixture
def ride(ride_factory) -> Ride:
    """A single, minimal-but-realistic :class:`Ride`.

    10 points, 1 degree of latitude and 1000s apart, starting at
    2026-01-01T00:00:00Z.
    """
    return ride_factory()


@pytest.fixture
def trips_factory(ride_factory):
    """Factory for building a :class:`Trips` from several distinct rides.

    Ride ``i`` covers latitudes ``[10*i, 10*i + 9]`` and starts at a
    distinct hour so per-ride localisation is independently verifiable.
    """

    def _make(n_rides: int = 3) -> Trips:
        rides = [
            ride_factory(
                ride_id=f"ride{i}",
                lat_start=10.0 * i,
                start=f"2026-01-0{i + 1}T{8 + i:02d}:00:00Z",
            )
            for i in range(n_rides)
        ]
        return Trips.from_rides(rides)

    return _make


@pytest.fixture
def trips(trips_factory) -> Trips:
    """A :class:`Trips` collection of three distinct rides."""
    return trips_factory()
