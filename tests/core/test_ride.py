"""Tests for polkupy.core.ride.Ride, one class per method/property."""

# pylint: disable=missing-function-docstring,too-few-public-methods

import math
from typing import cast

import pandas as pd
import pytest

from polkupy.core.ride import Ride
from polkupy.core.trips import Trips
from polkupy.geo import EARTH_RADIUS_M


class TestRideInit:
    """Ride(data, ride_id=None, bike_id=None) construction."""

    def test_missing_required_column_raises_value_error(self):
        df = pd.DataFrame(
            {"time": [pd.Timestamp("2026-01-01T00:00:00Z")], "lat": [52.0]}
        )
        with pytest.raises(ValueError, match="lon"):
            Ride(df)

    def test_ride_id_falls_back_to_column(self, ride_factory):
        ride = ride_factory(ride_id="from-arg")
        ride2 = Ride(ride.data)  # no explicit ride_id; must fall back to the column
        assert ride2.ride_id == "from-arg"

    def test_ride_id_none_when_absent(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "lat": [52.0],
                "lon": [13.0],
            }
        )
        assert Ride(df).ride_id is None

    def test_explicit_ride_id_is_added_as_a_column(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "lat": [52.0],
                "lon": [13.0],
            }
        )
        ride = Ride(df, ride_id="abc")
        assert (ride.data["ride_id"] == "abc").all()

    def test_bike_id_is_not_added_as_a_column(self):
        """Unlike ride_id, an explicit bike_id is not written back onto `.data`.

        This asymmetry is harmless for transforms (they pass ride_id/bike_id
        through explicitly rather than re-deriving from columns -- see
        TestRideLocalised), but Trips.__iter__/__getitem__ still recover
        bike_id from the column alone, so it would be lost there for a Ride
        constructed this way.
        """
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "lat": [52.0],
                "lon": [13.0],
            }
        )
        ride = Ride(df, bike_id="ktm")
        assert ride.bike_id == "ktm"
        assert "bike_id" not in ride.data.columns

    def test_index_is_reset(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z"]
                ),
                "lat": [52.0, 52.1],
                "lon": [13.0, 13.1],
            },
            index=[5, 6],
        )
        assert list(Ride(df).data.index) == [0, 1]


class TestRideAddAndRadd:
    """__add__/__radd__: concatenation into a Trips, including sum()."""

    def test_ride_plus_zero_wraps_in_trips(self, ride):
        result = ride + 0
        assert isinstance(result, Trips)
        assert len(result) == 1
        assert len(result.data) == len(ride)

    def test_zero_plus_ride_wraps_in_trips(self, ride):
        result = 0 + ride
        assert isinstance(result, Trips)
        assert len(result) == 1

    def test_ride_plus_ride(self, ride_factory):
        r1, r2 = ride_factory(ride_id="a"), ride_factory(ride_id="b")
        result = r1 + r2
        assert isinstance(result, Trips)
        assert len(result) == 2
        assert len(result.data) == len(r1) + len(r2)

    def test_ride_plus_trips(self, ride_factory, trips):
        extra = ride_factory(ride_id="extra")
        result = extra + trips
        assert isinstance(result, Trips)
        assert len(result) == len(trips) + 1

    def test_sum_of_several_rides(self, ride_factory):
        rides = [ride_factory(ride_id=f"r{i}") for i in range(4)]
        result = sum(rides)
        assert isinstance(result, Trips)
        assert len(result) == 4


class TestRideLen:
    """__len__: number of points in the ride."""

    def test_matches_number_of_points(self, ride_factory):
        assert len(ride_factory(n=7)) == 7


class TestRideReprPng:
    """_repr_png_(): renders the route as a PNG map."""

    def test_returns_a_valid_png(self, ride):
        png_bytes = ride._repr_png_()  # pylint: disable=protected-access
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


class TestRideReprHtml:
    """_repr_html_(): renders metadata plus the map as a Jupyter summary."""

    def test_includes_ride_id_and_map(self, ride):
        html = ride._repr_html_()  # pylint: disable=protected-access
        assert f"<b>ride ID</b>: {ride.ride_id}" in html
        assert "data:image/png;base64," in html

    def test_omits_bike_id_bullet_when_absent(self, ride):
        html = ride._repr_html_()  # pylint: disable=protected-access
        assert "<b>bike ID</b>" not in html

    def test_includes_bike_id_bullet_when_present(self, ride_factory):
        ride = ride_factory(bike_id="ktm")
        html = ride._repr_html_()  # pylint: disable=protected-access
        assert "<b>bike ID</b>: ktm" in html

    def test_formats_sampling_rate_in_seconds(self, ride):
        html = ride._repr_html_()  # pylint: disable=protected-access
        assert "<b>sampling rate</b>: 1000.0 s" in html


class TestRideFromGpx:
    """from_gpx(): loads a single GPX file as a Ride."""

    def test_reads_points(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>'
            '<trkpt lat="52.1" lon="13.1"><time>2026-01-01T00:00:10Z</time></trkpt>',
        )
        ride = Ride.from_gpx(path)
        assert len(ride) == 2
        assert ride.data.loc[1, "lat"] == 52.1

    def test_ride_id_defaults_to_filename_stem(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "my_ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        assert Ride.from_gpx(path).ride_id == "my_ride"

    def test_ride_id_defaults_to_bike_and_stem_when_bike_given(
        self, tmp_path, write_gpx
    ):
        path = write_gpx(
            tmp_path / "my_ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        ride = Ride.from_gpx(path, bike_id="ktm")
        assert ride.ride_id == "ktm/my_ride"

    def test_explicit_ride_id_overrides_default(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "my_ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        ride = Ride.from_gpx(path, ride_id="custom")
        assert ride.ride_id == "custom"

    def test_bike_id_is_set_as_attribute_and_column(self, tmp_path, write_gpx):
        """Unlike the bare constructor, from_gpx writes bike_id onto the DataFrame too.

        This means it survives later transforms.
        """
        path = write_gpx(
            tmp_path / "my_ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        ride = Ride.from_gpx(path, bike_id="ktm")
        assert ride.bike_id == "ktm"
        assert (ride.data["bike_id"] == "ktm").all()


class TestRideFromFit:
    """from_fit(): not implemented yet."""

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Ride.from_fit()


class TestRideStartTime:
    """start_time: timestamp of the first point."""

    def test_matches_first_point(self, ride):
        assert ride.start_time == pd.Timestamp("2026-01-01T00:00:00Z")


class TestRideEndTime:
    """end_time: timestamp of the last point."""

    def test_matches_last_point(self, ride_factory):
        ride = ride_factory(n=10)
        assert ride.end_time == pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(
            seconds=9000
        )


class TestRideDuration:
    """duration: elapsed wall-clock time, end_time minus start_time."""

    def test_is_end_minus_start(self, ride_factory):
        ride = ride_factory(n=10)
        assert ride.duration == pd.Timedelta(seconds=9000)


class TestRideSamplingRate:
    """sampling_rate: median interval between consecutive points."""

    def test_is_median_of_consecutive_gaps(self, ride):
        assert ride.sampling_rate == pd.Timedelta(seconds=1000)


class TestRideDistanceKm:
    """distance_km: cumulative great-circle distance between points."""

    def test_matches_cumulative_haversine(self, ride_factory):
        ride = ride_factory(n=10)  # 9 gaps of 1 degree of latitude each
        expected_km = 9 * EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert ride.distance_km == pytest.approx(expected_km)

    def test_is_zero_for_a_single_point(self, ride_factory):
        assert ride_factory(n=1).distance_km == 0.0


class TestRideStartsNear:
    """starts_near(): whether the ride starts near a given point."""

    def test_true_for_exact_match(self, ride):
        assert ride.starts_near(52.0, 13.0) is True

    def test_false_when_far_away(self, ride):
        assert ride.starts_near(61.0, 13.0) is False

    def test_only_checks_first_n_points(self, ride):
        # index 4 (lat 56) is within the default first 5 points...
        assert ride.starts_near(56.0, 13.0, n_points=5) is True
        # ...but not within just the first 1.
        assert ride.starts_near(56.0, 13.0, n_points=1) is False

    def test_respects_radius_km(self, ride):
        two_degrees_km = 2 * EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert ride.starts_near(50.0, 13.0, radius_km=two_degrees_km + 1) is True
        assert ride.starts_near(50.0, 13.0, radius_km=two_degrees_km - 1) is False


class TestRideEndsNear:
    """ends_near(): whether the ride ends near a given point."""

    def test_true_for_exact_match(self, ride):
        assert ride.ends_near(61.0, 13.0) is True

    def test_false_when_far_away(self, ride):
        assert ride.ends_near(52.0, 13.0) is False

    def test_only_checks_last_n_points(self, ride):
        # index 5 (lat 57) is within the default last 5 points...
        assert ride.ends_near(57.0, 13.0, n_points=5) is True
        # ...but not within just the last 1.
        assert ride.ends_near(57.0, 13.0, n_points=1) is False

    def test_respects_radius_km(self, ride):
        two_degrees_km = 2 * EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert ride.ends_near(63.0, 13.0, radius_km=two_degrees_km + 1) is True
        assert ride.ends_near(63.0, 13.0, radius_km=two_degrees_km - 1) is False


class TestRideLocalised:
    """localised(): aligns the ride to a shared 24h clock."""

    def test_converts_timezone(self, ride):
        result = ride.localised(tz="Europe/Berlin")
        # 2026-01-01 is winter (CET, UTC+1, no DST): 00:00 UTC -> 01:00 local.
        assert result.data.loc[0, "time"].hour == 1

    def test_adds_ride_start_s_and_virtual_s(self, ride):
        result = ride.localised(tz="Europe/Berlin")
        assert result.data.loc[0, "ride_start_s"] == 3600.0
        assert result.data.loc[9, "virtual_s"] == 3600.0 + 9000.0

    def test_preserves_ride_id(self, ride):
        assert ride.localised().ride_id == ride.ride_id

    def test_preserves_bike_id_even_when_not_a_column(self, ride_factory):
        """bike_id is passed through explicitly rather than re-derived from `.data`.

        So it survives even when (unlike ride_id) it was never written back
        onto the DataFrame -- see TestRideInit.
        """
        ride = ride_factory(bike_id="ktm")
        assert ride.localised().bike_id == "ktm"


class TestRideWithDistance:
    """with_distance(): adds a dist_m column."""

    def test_first_point_has_no_distance(self, ride):
        result = ride.with_distance()
        assert math.isnan(cast(float, result.data.loc[0, "dist_m"]))

    def test_matches_known_distance(self, ride):
        result = ride.with_distance()
        expected = EARTH_RADIUS_M * math.radians(1.0)
        assert result.data.loc[1, "dist_m"] == pytest.approx(expected)

    def test_preserves_ride_id(self, ride):
        assert ride.with_distance().ride_id == ride.ride_id

    def test_preserves_bike_id_even_when_not_a_column(self, ride_factory):
        ride = ride_factory(bike_id="ktm")
        assert ride.with_distance().bike_id == "ktm"


class TestRideWithSpeed:
    """with_speed(): adds dist_m and speed_kmh columns."""

    def test_first_point_has_no_speed(self, ride):
        result = ride.with_speed()
        assert math.isnan(cast(float, result.data.loc[0, "speed_kmh"]))

    def test_matches_known_speed(self, ride):
        result = ride.with_speed()
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        expected_speed_kmh = expected_dist_km / (1000 / 3600)
        assert result.data.loc[1, "speed_kmh"] == pytest.approx(expected_speed_kmh)

    def test_preserves_ride_id(self, ride):
        assert ride.with_speed().ride_id == ride.ride_id

    def test_preserves_bike_id_even_when_not_a_column(self, ride_factory):
        ride = ride_factory(bike_id="ktm")
        assert ride.with_speed().bike_id == "ktm"
