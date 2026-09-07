"""Tests for polkupy.algorithms.filters.filter_rides_in_bbox."""

# pylint: disable=C0116

import pandas as pd
import pytest

from polkupy.algorithms.filters import filter_rides_in_bbox


class TestFilterRidesInBbox:
    """Tests function filter_rides_in_bbox."""

    @pytest.fixture
    def multi_ride_df(self) -> pd.DataFrame:
        """Three rides, 5 points each: "a" at lat [0, 4], "b" at lat [10, 14]
        (both lon 13), and "c" at lat [50, 54], lon 100 (far away)."""
        rows = []
        for ride_id, lat_start, lon in [
            ("a", 0.0, 13.0),
            ("b", 10.0, 13.0),
            ("c", 50.0, 100.0),
        ]:
            for i in range(5):
                rows.append({"ride_id": ride_id, "lat": lat_start + i, "lon": lon})
        return pd.DataFrame(rows)

    def test_keeps_ride_with_any_point_inside_box(self, multi_ride_df):
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=3.0, lat_max=4.0, lon_min=10.0, lon_max=20.0
        )
        assert set(result["ride_id"]) == {"a"}

    def test_keeps_matching_ride_in_full_not_cropped(self, multi_ride_df):
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=3.0, lat_max=4.0, lon_min=10.0, lon_max=20.0
        )
        assert len(result) == 5

    def test_excludes_ride_entirely_outside(self, multi_ride_df):
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=0.0, lat_max=14.0, lon_min=0.0, lon_max=20.0
        )
        assert "c" not in set(result["ride_id"])

    def test_multiple_rides_can_match(self, multi_ride_df):
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=4.0, lat_max=10.0, lon_min=0.0, lon_max=20.0
        )
        assert set(result["ride_id"]) == {"a", "b"}

    def test_boundary_values_are_inclusive(self, multi_ride_df):
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=4.0, lat_max=4.0, lon_min=13.0, lon_max=13.0
        )
        assert set(result["ride_id"]) == {"a"}

    def test_no_matching_points_returns_empty(self, multi_ride_df):
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=1000.0, lat_max=1001.0, lon_min=0.0, lon_max=0.0
        )
        assert len(result) == 0
        assert list(result.columns) == list(multi_ride_df.columns)

    def test_resets_index(self, multi_ride_df):
        # "b" occupies rows 5-9 in the input; the result must not carry
        # that original positional index forward.
        result = filter_rides_in_bbox(
            multi_ride_df, lat_min=10.0, lat_max=14.0, lon_min=0.0, lon_max=20.0
        )
        assert list(result.index) == list(range(len(result)))

    def test_does_not_mutate_input(self, multi_ride_df):
        original = multi_ride_df.copy()
        filter_rides_in_bbox(
            multi_ride_df, lat_min=0.0, lat_max=4.0, lon_min=0.0, lon_max=20.0
        )
        pd.testing.assert_frame_equal(multi_ride_df, original)
