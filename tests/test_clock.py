"""Tests for polkupy.clock.add_time_of_day."""

# pylint: disable=C0116

import pandas as pd
import pytest

from polkupy.clock import add_time_of_day


class TestAddTimeOfDay:
    """add_time_of_day() aligns rides to a shared 24h clock."""

    @pytest.fixture
    def multi_ride_df(self) -> pd.DataFrame:
        """Two rides, 3 points each, 1000s apart.

        Each ride starts on a different UTC hour/day, so per-ride
        independence is verifiable.
        """
        times = []
        ride_ids = []
        for ride_id, start in [
            ("a", "2026-01-01T08:00:00Z"),
            ("b", "2026-01-02T09:00:00Z"),
        ]:
            start_ts = pd.Timestamp(start)
            for i in range(3):
                times.append(start_ts + pd.Timedelta(seconds=i * 1000))
                ride_ids.append(ride_id)
        return pd.DataFrame({"ride_id": ride_ids, "time": pd.to_datetime(times)})

    def test_converts_timezone(self, multi_ride_df):
        result = add_time_of_day(multi_ride_df, tz="Europe/Berlin")
        assert result["time"].iloc[0].hour == 9  # 08:00 UTC -> 09:00 CET (winter)

    def test_ride_start_s_is_seconds_since_local_midnight(self, multi_ride_df):
        result = add_time_of_day(multi_ride_df, tz="Europe/Berlin")
        first_per_ride = result.groupby("ride_id")["ride_start_s"].first()
        assert first_per_ride["a"] == 9 * 3600.0
        assert first_per_ride["b"] == 10 * 3600.0

    def test_ride_start_s_is_constant_within_a_ride(self, multi_ride_df):
        result = add_time_of_day(multi_ride_df, tz="Europe/Berlin")
        assert result.loc[result["ride_id"] == "a", "ride_start_s"].nunique() == 1

    def test_virtual_s_adds_elapsed_time(self, multi_ride_df):
        result = add_time_of_day(multi_ride_df, tz="Europe/Berlin")
        last_per_ride = result.groupby("ride_id")["virtual_s"].last()
        assert last_per_ride["a"] == 9 * 3600.0 + 2000.0  # 2 gaps of 1000s

    def test_each_ride_is_independent(self, multi_ride_df):
        result = add_time_of_day(multi_ride_df, tz="Europe/Berlin")
        first_per_ride = result.groupby("ride_id")["ride_start_s"].first()
        assert first_per_ride["a"] != first_per_ride["b"]

    def test_default_timezone_is_europe_berlin(self, multi_ride_df):
        default_result = add_time_of_day(multi_ride_df)
        explicit_result = add_time_of_day(multi_ride_df, tz="Europe/Berlin")
        pd.testing.assert_frame_equal(default_result, explicit_result)

    def test_does_not_mutate_input(self, multi_ride_df):
        original = multi_ride_df.copy()
        add_time_of_day(multi_ride_df)
        pd.testing.assert_frame_equal(multi_ride_df, original)

    def test_naive_time_column_raises(self):
        df = pd.DataFrame(
            {"ride_id": ["a"], "time": pd.to_datetime(["2026-01-01T00:00:00"])}
        )
        with pytest.raises(TypeError):
            add_time_of_day(df)
