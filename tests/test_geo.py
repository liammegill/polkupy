"""Tests for polkupy.geo: haversine distance and per-point speed."""

# pylint: disable=C0116

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from polkupy.geo import EARTH_RADIUS_M, calc_distance, calc_speed, haversine


class TestHaversine:
    """haversine() computes great-circle distance between two points."""

    def test_same_point_is_zero_distance(self):
        assert haversine(13.0, 52.0, 13.0, 52.0) == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_of_latitude_matches_meridian_arc_length(self):
        # along a meridian (same longitude), haversine reduces to a plain
        # great-circle arc: distance = R * delta_latitude (in radians).
        expected = EARTH_RADIUS_M * math.radians(1.0)
        assert haversine(13.0, 52.0, 13.0, 53.0) == pytest.approx(expected)

    def test_antipodal_points_span_half_the_circumference(self):
        expected = EARTH_RADIUS_M * math.pi
        assert haversine(0.0, 0.0, 180.0, 0.0) == pytest.approx(expected)

    def test_is_symmetric(self):
        forward = haversine(13.0, 52.0, 8.5, 47.4)
        backward = haversine(8.5, 47.4, 13.0, 52.0)
        assert forward == pytest.approx(backward)

    def test_never_negative(self):
        assert haversine(-179.0, -89.0, 179.0, 89.0) >= 0.0

    def test_vectorised_over_arrays(self):
        lon1 = np.array([13.0, 0.0])
        lat1 = np.array([52.0, 0.0])
        lon2 = np.array([13.0, 180.0])
        lat2 = np.array([53.0, 0.0])

        result = haversine(lon1, lat1, lon2, lat2)

        assert result.shape == (2,)
        assert result[0] == pytest.approx(EARTH_RADIUS_M * math.radians(1.0))
        assert result[1] == pytest.approx(EARTH_RADIUS_M * math.pi)

    def test_broadcasts_scalar_against_array(self):
        # mirrors real usage in Ride._near: several points vs. one target.
        lon1 = [13.0, 13.0]
        lat1 = [52.0, 52.0]

        result = haversine(lon1, lat1, 13.0, 53.0)

        expected = EARTH_RADIUS_M * math.radians(1.0)
        assert result[0] == pytest.approx(expected)
        assert result[1] == pytest.approx(expected)

    def test_accepts_pandas_series_and_propagates_nan(self):
        lon1 = pd.Series([13.0, np.nan])
        lat1 = pd.Series([52.0, np.nan])
        lon2 = pd.Series([13.0, 13.0])
        lat2 = pd.Series([53.0, 53.0])

        result = haversine(lon1, lat1, lon2, lat2)

        assert result[0] == pytest.approx(EARTH_RADIUS_M * math.radians(1.0))
        assert math.isnan(result[1])


class TestCalcDistance:
    """calc_distance() adds a per-point dist_km column."""

    @pytest.fixture
    def points_df(self) -> pd.DataFrame:
        # three points, 1 degree of latitude apart -- each consecutive gap
        # is EARTH_RADIUS_M * radians(1) metres.
        return pd.DataFrame({"lat": [52.0, 53.0, 54.0], "lon": [13.0, 13.0, 13.0]})

    def test_first_row_has_no_previous_point(self, points_df):
        result = calc_distance(points_df)
        assert math.isnan(cast(float, result.loc[0, "dist_km"]))

    def test_computes_distance_matching_haversine(self, points_df):
        result = calc_distance(points_df)
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert result.loc[1, "dist_km"] == pytest.approx(expected_dist_km)

    def test_does_not_mutate_input(self, points_df):
        original = points_df.copy()
        calc_distance(points_df)
        pd.testing.assert_frame_equal(points_df, original)

    def test_keeps_existing_dist_km_by_default(self, points_df):
        # e.g. a device's own distance sensor, read in by load_fit().
        points_df["dist_km"] = [np.nan, 1.234, 5.678]
        result = calc_distance(points_df)
        assert result.loc[1, "dist_km"] == 1.234
        assert result.loc[2, "dist_km"] == 5.678

    def test_overwrite_recomputes_existing_dist_km(self, points_df):
        points_df["dist_km"] = [np.nan, 1.234, 5.678]
        result = calc_distance(points_df, overwrite=True)
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert result.loc[1, "dist_km"] == pytest.approx(expected_dist_km)


class TestCalcSpeed:
    """calc_speed() adds per-point dist_km/speed_kmh columns."""

    @pytest.fixture
    def points_df(self) -> pd.DataFrame:
        # three points, 1 degree of latitude and 1000s apart -- each
        # consecutive gap is EARTH_RADIUS_M * radians(1) metres.
        return pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:16:40Z",
                        "2026-01-01T00:33:20Z",
                    ]
                ),
                "lat": [52.0, 53.0, 54.0],
                "lon": [13.0, 13.0, 13.0],
            }
        )

    def test_first_row_has_no_previous_point(self, points_df):
        result = calc_speed(points_df)
        assert math.isnan(cast(float, result.loc[0, "dist_km"]))
        assert math.isnan(cast(float, result.loc[0, "speed_kmh"]))

    def test_computes_distance_matching_haversine(self, points_df):
        result = calc_speed(points_df)
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert result.loc[1, "dist_km"] == pytest.approx(expected_dist_km)
        assert result.loc[2, "dist_km"] == pytest.approx(expected_dist_km)

    def test_computes_speed_in_kmh(self, points_df):
        result = calc_speed(points_df)
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        expected_speed_kmh = expected_dist_km / 1000 * 3600  # 1000s elapsed
        assert result.loc[1, "speed_kmh"] == pytest.approx(expected_speed_kmh)

    def test_does_not_mutate_input(self, points_df):
        original = points_df.copy()
        calc_speed(points_df)
        pd.testing.assert_frame_equal(points_df, original)

    def test_reuses_existing_dist_km_column_instead_of_recomputing(self, points_df):
        points_df["dist_km"] = [np.nan, 1.234, 5.678]

        result = calc_speed(points_df)

        assert result.loc[1, "dist_km"] == 1.234
        assert result.loc[2, "dist_km"] == 5.678
        # speed is derived from the *provided* dist_km, not the true
        # haversine distance (~111 km), which would give ~400 km/h instead.
        assert result.loc[1, "speed_kmh"] == pytest.approx(1.234 / 1000 * 3600)

    def test_zero_elapsed_time_gives_infinite_speed(self, points_df):
        points_df.loc[1, "time"] = points_df.loc[0, "time"]  # duplicate timestamp
        result = calc_speed(points_df)
        assert math.isinf(cast(float, result.loc[1, "speed_kmh"]))

    def test_single_row_is_all_nan_without_error(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                "lat": [52.0],
                "lon": [13.0],
            }
        )

        result = calc_speed(df)

        assert math.isnan(cast(float, result.loc[0, "dist_km"]))
        assert math.isnan(cast(float, result.loc[0, "speed_kmh"]))

    def test_preserves_other_columns(self, points_df):
        points_df["ride_id"] = "abc"
        result = calc_speed(points_df)
        assert (result["ride_id"] == "abc").all()

    def test_keeps_existing_speed_kmh_by_default(self, points_df):
        # e.g. a device's own speed sensor, read in by load_fit().
        points_df["speed_kmh"] = [np.nan, 99.0, 99.0]
        result = calc_speed(points_df)
        assert result.loc[1, "speed_kmh"] == 99.0
        assert result.loc[2, "speed_kmh"] == 99.0

    def test_overwrite_recomputes_existing_speed_kmh(self, points_df):
        points_df["speed_kmh"] = [np.nan, 99.0, 99.0]
        result = calc_speed(points_df, overwrite=True)
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        expected_speed_kmh = expected_dist_km / 1000 * 3600
        assert result.loc[1, "speed_kmh"] == pytest.approx(expected_speed_kmh)

    def test_overwrite_also_recomputes_existing_dist_km(self, points_df):
        points_df["dist_km"] = [np.nan, 1.234, 5.678]
        result = calc_speed(points_df, overwrite=True)
        expected_dist_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        assert result.loc[1, "dist_km"] == pytest.approx(expected_dist_km)
