"""Tests for polkupy.core.trips.Trips, one class per method."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from typing import cast

import pandas as pd
import pytest

from polkupy.core.ride import Ride
from polkupy.core.trips import Trips


class TestTripsInit:
    """Trips(data) construction."""

    def test_missing_required_column_raises_value_error(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "lat": [52.0],
                "ride_id": ["a"],
            }
        )
        with pytest.raises(ValueError, match="lon"):
            Trips(df)

    def test_requires_ride_id_column(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "lat": [52.0],
                "lon": [13.0],
            }
        )
        with pytest.raises(ValueError, match="ride_id"):
            Trips(df)


class TestTripsAddAndRadd:
    """__add__/__radd__: concatenation, including sum()."""

    def test_trips_plus_zero_returns_self(self, trips):
        assert (trips + 0) is trips

    def test_zero_plus_trips_returns_self(self, trips):
        assert (0 + trips) is trips

    def test_trips_plus_ride(self, trips, ride_factory):
        extra = ride_factory(ride_id="extra")
        result = trips + extra
        assert len(result) == len(trips) + 1

    def test_trips_plus_trips(self, trips_factory, ride_factory):
        t1 = trips_factory(n_rides=2)  # ride0, ride1
        t2 = Trips.from_rides(
            [
                ride_factory(ride_id="other0", lat_start=100.0),
                ride_factory(ride_id="other1", lat_start=110.0),
            ]
        )
        result = t1 + t2
        assert len(result) == 4
        assert len(result.data) == len(t1.data) + len(t2.data)

    def test_sum_of_a_ride_and_a_trips(self, ride_factory, trips):
        extra = ride_factory(ride_id="extra")
        result = cast(Trips, sum([extra, trips]))
        assert len(result) == len(trips) + 1


class TestTripsLen:
    """__len__: number of distinct rides in the collection."""

    def test_counts_distinct_rides(self, trips):
        assert len(trips) == 3


class TestTripsIter:
    """__iter__: yields one Ride per distinct ride_id."""

    def test_yields_one_ride_per_ride_id(self, trips):
        rides = list(trips)
        assert {r.ride_id for r in rides} == {"ride0", "ride1", "ride2"}
        assert all(isinstance(r, Ride) for r in rides)

    def test_each_yielded_ride_has_only_its_own_points(self, trips):
        rides = {r.ride_id: r for r in trips}
        assert len(rides["ride0"]) == 10
        assert set(rides["ride0"].data["lat"]) == set(range(0, 10))


class TestTripsGetitem:
    """__getitem__: look up a single ride by id."""

    def test_returns_matching_ride(self, trips):
        ride = trips["ride1"]
        assert isinstance(ride, Ride)
        assert ride.ride_id == "ride1"
        assert len(ride) == 10

    def test_missing_ride_id_raises_key_error(self, trips):
        with pytest.raises(KeyError):
            trips["nonexistent"]  # pylint: disable=pointless-statement


class TestTripsRepr:
    """__repr__: formats ride and point counts."""

    def test_formats_ride_and_point_counts(self, trips):
        assert repr(trips) == "Trips(3 rides, 30 points)"


class TestTripsReprHtml:
    """_repr_html_(): renders collection metadata plus a per-ride table."""

    def test_includes_ride_count_and_start_end(self, trips):
        html = trips._repr_html_()  # pylint: disable=protected-access
        assert "<b>rides</b>: 3" in html
        assert "<b>start</b>:" in html
        assert "<b>end</b>:" in html

    def test_omits_bikes_bullet_when_bike_id_absent(self, trips):
        # conftest's ride_factory never writes bike_id onto `.data`.
        html = trips._repr_html_()  # pylint: disable=protected-access
        assert "<b>bikes</b>" not in html

    def test_includes_bikes_bullet_and_table_when_bike_id_present(self, ride_factory):
        r1, r2 = ride_factory(ride_id="a", n=3), ride_factory(ride_id="b", n=5)
        r1.data["bike_id"] = "ktm"
        r2.data["bike_id"] = "canyon"
        trips = Trips.from_rides([r1, r2])

        html = trips._repr_html_()  # pylint: disable=protected-access

        assert "<b>bikes</b>: 2" in html
        assert "<td>a</td><td>ktm</td><td>3</td>" in html
        assert "<td>b</td><td>canyon</td><td>5</td>" in html

    def test_truncates_long_ride_lists(self, ride_factory):
        rides = [ride_factory(ride_id=f"r{i}", n=1) for i in range(20)]
        trips = Trips.from_rides(rides)

        html = trips._repr_html_()  # pylint: disable=protected-access

        assert "... and 5 more rides" in html


class TestTripsFromGpx:
    """from_gpx(): loads every GPX ride from a directory."""

    def test_delegates_to_load_gpx_dir(self, monkeypatch):
        captured = {}

        def fake_load_gpx_dir(data_dir, bikes=None, extensions=None):
            captured["args"] = (data_dir, bikes, extensions)
            return pd.DataFrame(
                {
                    "time": pd.to_datetime(
                        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z"]
                    ),
                    "lat": [52.0, 52.001],
                    "lon": [13.0, 13.0],
                    "ride_id": ["r1", "r1"],
                }
            )

        monkeypatch.setattr("polkupy.core.trips.load_gpx_dir", fake_load_gpx_dir)

        result = Trips.from_gpx("some/dir", bikes=["ktm"], extensions=["hr"])

        assert captured["args"] == ("some/dir", ["ktm"], ["hr"])
        assert isinstance(result, Trips)
        assert len(result) == 1
        assert len(result.data) == 2


class TestTripsFromRides:
    """from_rides(): concatenates several Ride/Trips into one Trips."""

    def test_concatenates_rides(self, ride_factory):
        r1, r2 = ride_factory(ride_id="a"), ride_factory(ride_id="b")
        result = Trips.from_rides([r1, r2])
        assert len(result) == 2
        assert len(result.data) == len(r1) + len(r2)

    def test_accepts_a_mix_of_rides_and_trips(self, ride_factory, trips):
        extra = ride_factory(ride_id="extra")
        result = Trips.from_rides([extra, trips])
        assert len(result) == len(trips) + 1


class TestTripsFilter:
    """filter(): keeps only rides matching a predicate."""

    def test_keeps_only_matching_rides(self, trips):
        result = trips.filter(lambda ride: ride.ride_id != "ride1")
        assert {r.ride_id for r in result} == {"ride0", "ride2"}
        assert len(result) == 2


class TestTripsInBbox:
    """in_bbox(): keeps rides that pass through a bounding box.

    ride0 covers lat [0, 9], ride1 lat [10, 19], ride2 lat [20, 29]
    (all at lon 13); see conftest.trips_factory.
    """

    def test_keeps_rides_that_intersect_in_full(self, trips):
        result = trips.in_bbox(lat_min=9.0, lat_max=11.0, lon_min=0.0, lon_max=20.0)
        assert {r.ride_id for r in result} == {"ride0", "ride1"}
        assert len(result["ride0"]) == 10  # kept whole, not point-cropped
        assert len(result["ride1"]) == 10

    def test_excludes_rides_entirely_outside(self, trips):
        result = trips.in_bbox(lat_min=9.0, lat_max=11.0, lon_min=0.0, lon_max=20.0)
        assert "ride2" not in {r.ride_id for r in result}


class TestTripsBetween:
    """between(): keeps rides that start/end near given points."""

    def test_keeps_rides_matching_start_and_end(self, trips):
        # ride0 starts at (0, 13) and ends at (9, 13).
        result = trips.between(start=(0.0, 13.0), end=(9.0, 13.0))
        assert {r.ride_id for r in result} == {"ride0"}

    def test_no_match_returns_empty(self, trips):
        result = trips.between(start=(999.0, 999.0), end=(999.0, 999.0))
        assert len(result) == 0


class TestTripsLocalised:
    """localised(): aligns every ride to a shared 24h clock.

    ride0/1/2 start at 08:00/09:00/10:00 UTC on consecutive days; see
    conftest.trips_factory. Europe/Berlin is UTC+1 in January (no DST).
    """

    def test_converts_timezone_for_every_ride(self, trips):
        result = trips.localised(tz="Europe/Berlin")
        assert result.data["time"].iloc[0].hour == 9  # ride0: 08:00 UTC

    def test_computes_ride_start_s_independently_per_ride(self, trips):
        result = trips.localised(tz="Europe/Berlin")
        first_per_ride = result.data.groupby("ride_id")["ride_start_s"].first()
        assert first_per_ride["ride0"] == 9 * 3600.0
        assert first_per_ride["ride1"] == 10 * 3600.0
        assert first_per_ride["ride2"] == 11 * 3600.0

    def test_virtual_s_adds_elapsed_time(self, trips):
        result = trips.localised(tz="Europe/Berlin")
        last_per_ride = result.data.groupby("ride_id")["virtual_s"].last()
        # each ride has 9 gaps of 1000s = 9000s elapsed by its last point.
        assert last_per_ride["ride0"] == 9 * 3600.0 + 9000.0
