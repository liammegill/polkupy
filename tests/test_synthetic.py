"""Tests for polkupy.synthetic: generate_track."""

# pylint: disable=C0116

import math

import numpy as np
import pandas as pd
import pytest

from polkupy.geo import EARTH_RADIUS_M
from polkupy.synthetic import generate_track


class TestGenerateTrack:
    """generate_track() interpolates between waypoints at a constant speed."""

    def test_requires_at_least_two_waypoints(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            generate_track([(52.0, 13.0)])

    def test_requires_positive_speed(self):
        with pytest.raises(ValueError, match="speed_kmh must be positive"):
            generate_track([(52.0, 13.0), (53.0, 13.0)], speed_kmh=0)

    def test_requires_positive_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate_s must be positive"):
            generate_track([(52.0, 13.0), (53.0, 13.0)], sample_rate_s=0)

    def test_returns_required_ride_columns(self):
        df = generate_track([(52.0, 13.0), (53.0, 13.0)])
        assert list(df.columns) == ["time", "lat", "lon"]

    def test_omits_ele_column_by_default(self):
        df = generate_track([(52.0, 13.0), (53.0, 13.0)])
        assert "ele" not in df.columns

    def test_first_point_matches_first_waypoint(self):
        df = generate_track([(52.0, 13.0), (53.0, 14.0)])
        assert df.loc[0, "lat"] == pytest.approx(52.0)
        assert df.loc[0, "lon"] == pytest.approx(13.0)

    def test_last_point_matches_last_waypoint(self):
        df = generate_track([(52.0, 13.0), (53.0, 14.0)])
        assert df.iloc[-1]["lat"] == pytest.approx(53.0)
        assert df.iloc[-1]["lon"] == pytest.approx(14.0)

    def test_passes_through_intermediate_waypoint(self):
        # a straight line north, so the midpoint waypoint's lat must appear
        # exactly among the generated points (lon is constant throughout).
        df = generate_track([(52.0, 13.0), (52.5, 13.0), (53.0, 13.0)])
        assert np.isclose(df["lat"].to_numpy(), 52.5, atol=1e-9).any()

    def test_total_distance_matches_waypoints(self):
        # one degree of latitude, matching the known-value shape in tests/test_geo.py.
        df = generate_track([(52.0, 13.0), (53.0, 13.0)], speed_kmh=36.0)
        expected_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        travelled_km = (
            (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 3600 * 36.0
        )
        assert travelled_km == pytest.approx(expected_km)

    def test_respects_constant_speed(self):
        expected_km = EARTH_RADIUS_M * math.radians(1.0) / 1000
        df = generate_track([(52.0, 13.0), (53.0, 13.0)], speed_kmh=20.0)
        duration_h = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 3600
        assert duration_h == pytest.approx(expected_km / 20.0)

    def test_points_are_roughly_sample_rate_apart(self):
        df = generate_track(
            [(52.0, 13.0), (53.0, 13.0)], speed_kmh=20.0, sample_rate_s=10.0
        )
        gaps_s = df["time"].diff().dropna().dt.total_seconds()
        # every gap is exactly sample_rate_s, except the final one (the
        # track's total duration isn't a multiple of sample_rate_s).
        assert gaps_s.iloc[:-1].to_numpy() == pytest.approx(10.0)
        assert gaps_s.iloc[-1] <= 10.0

    def test_start_string_becomes_utc_when_naive(self):
        df = generate_track([(52.0, 13.0), (53.0, 13.0)], start="2026-06-01T12:00:00")
        assert str(df.loc[0, "time"].tzinfo) == "UTC"

    def test_start_offset_is_preserved(self):
        df = generate_track(
            [(52.0, 13.0), (53.0, 13.0)], start="2026-06-01T12:00:00+02:00"
        )
        assert df.loc[0, "time"] == pd.Timestamp("2026-06-01T10:00:00Z")

    def test_scalar_elevation_is_constant(self):
        df = generate_track([(52.0, 13.0), (53.0, 13.0)], elevation_m=400.0)
        assert df["ele"].to_numpy() == pytest.approx(400.0)

    def test_per_waypoint_elevation_is_interpolated(self):
        df = generate_track(
            [(52.0, 13.0), (52.5, 13.0), (53.0, 13.0)],
            elevation_m=[400.0, 500.0, 400.0],
        )
        assert df.loc[0, "ele"] == pytest.approx(400.0)
        assert df.iloc[-1]["ele"] == pytest.approx(400.0)
        midpoint = df.iloc[len(df) // 2]
        assert midpoint["ele"] == pytest.approx(500.0, abs=5.0)

    def test_elevation_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="expected 1 or 3"):
            generate_track(
                [(52.0, 13.0), (52.5, 13.0), (53.0, 13.0)], elevation_m=[400.0, 500.0]
            )

    def test_zero_gps_noise_is_reproducible_without_seed(self):
        # gps_noise_m defaults to 0, so no randomness is invoked at all.
        first = generate_track([(52.0, 13.0), (53.0, 13.0)])
        second = generate_track([(52.0, 13.0), (53.0, 13.0)])
        pd.testing.assert_frame_equal(first, second)

    def test_seed_makes_gps_noise_reproducible(self):
        first = generate_track([(52.0, 13.0), (53.0, 13.0)], gps_noise_m=5.0, seed=42)
        second = generate_track([(52.0, 13.0), (53.0, 13.0)], gps_noise_m=5.0, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_gps_noise_perturbs_position(self):
        clean = generate_track([(52.0, 13.0), (53.0, 13.0)])
        noisy = generate_track([(52.0, 13.0), (53.0, 13.0)], gps_noise_m=50.0, seed=1)
        assert not clean["lat"].equals(noisy["lat"])

    def test_elevation_noise_perturbs_elevation(self):
        noisy = generate_track(
            [(52.0, 13.0), (53.0, 13.0)],
            elevation_m=400.0,
            elevation_noise_m=10.0,
            seed=1,
        )
        assert noisy["ele"].to_numpy() != pytest.approx(400.0)

    def test_elevation_noise_without_elevation_is_ignored(self):
        # no ele column requested at all, so there's nothing to add noise to.
        df = generate_track(
            [(52.0, 13.0), (53.0, 13.0)], elevation_noise_m=10.0, seed=1
        )
        assert "ele" not in df.columns

    def test_omits_speed_kmh_column_by_default(self):
        df = generate_track([(52.0, 13.0), (53.0, 13.0)])
        assert "speed_kmh" not in df.columns

    def test_speed_noise_adds_speed_kmh_column(self):
        df = generate_track(
            [(52.0, 13.0), (53.0, 13.0)], speed_kmh=20.0, speed_noise_kmh=2.0, seed=1
        )
        assert "speed_kmh" in df.columns
        assert len(df["speed_kmh"]) == len(df)

    def test_speed_noise_perturbs_speed(self):
        noisy = generate_track(
            [(52.0, 13.0), (53.0, 13.0)],
            speed_kmh=20.0,
            speed_noise_kmh=2.0,
            seed=1,
        )
        assert noisy["speed_kmh"].to_numpy() != pytest.approx(20.0)

    def test_speed_noise_is_reproducible_with_seed(self):
        first = generate_track(
            [(52.0, 13.0), (53.0, 13.0)], speed_noise_kmh=2.0, seed=42
        )
        second = generate_track(
            [(52.0, 13.0), (53.0, 13.0)], speed_noise_kmh=2.0, seed=42
        )
        pd.testing.assert_frame_equal(first, second)
