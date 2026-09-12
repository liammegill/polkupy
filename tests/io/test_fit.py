"""Tests for polkupy.io.fit: load_fit, load_fit_dir, and list_fit_fields.

fitdecode.FitReader is monkeypatched to yield synthetic "record" frames
instead of parsing real FIT bytes -- constructing a real FIT file by hand
requires implementing its binary framing (definition messages, per-field
base types, CRC16), which buys nothing here since load_fit's own logic
(merging split records, dropping GPS-less points, range checks, extension
matching) is independent of how fitdecode gets its field values. Field
values are given already "processed" (e.g. lat/lon in degrees), matching
what fitdecode.StandardUnitsDataProcessor -- used by load_fit -- yields.
"""

# pylint: disable=C0116

import warnings
from datetime import UTC, datetime

import fitdecode
import pandas as pd
import pytest

from polkupy.io.fit import list_fit_fields, load_fit, load_fit_dir

VALID_RECORD = {
    "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
    "position_lat": 52.0,
    "position_long": 13.0,
}


class _FakeField:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _FakeFrame:
    def __init__(self, fields):
        self.frame_type = fitdecode.FIT_FRAME_DATA
        self.name = "record"
        self.fields = [_FakeField(name, value) for name, value in fields.items()]


class _FakeFitReader:
    def __init__(self, records):
        self._records = records

    def __enter__(self):
        return [_FakeFrame(r) for r in self._records]

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def fake_fit(tmp_path, monkeypatch):
    """Factory: write an (empty, content-irrelevant) .fit file for `records`.

    `path.glob("*.fit")` in load_fit_dir needs a real file to find, but
    fitdecode.FitReader is patched to ignore its contents and yield
    `records` (a list of {field_name: value} dicts, one per synthetic
    "record" message) instead.
    """
    registry = {}
    monkeypatch.setattr(
        "polkupy.io.fit.fitdecode.FitReader",
        lambda filepath, **kwargs: _FakeFitReader(registry[str(filepath)]),
    )

    def _write(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        registry[str(path)] = records
        return str(path)

    return _write


class TestLoadFit:
    """load_fit() parses (fake) FIT record messages into a DataFrame."""

    def test_reads_core_columns(self, tmp_path, fake_fit):
        path = fake_fit(
            tmp_path / "ride.fit",
            [{**VALID_RECORD, "enhanced_altitude": 34.5}],
        )
        df = load_fit(path)
        assert list(df.columns) == ["time", "lat", "lon", "ele"]
        assert df.loc[0, "lat"] == 52.0
        assert df.loc[0, "lon"] == 13.0
        assert df.loc[0, "ele"] == 34.5

    def test_reads_sensor_distance_as_dist_m(self, tmp_path, fake_fit):
        path = fake_fit(
            tmp_path / "ride.fit",
            [
                {**VALID_RECORD, "distance": 0.0},
                {
                    **VALID_RECORD,
                    "timestamp": datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                    "distance": 0.01,  # 10 m further along
                },
            ],
        )
        df = load_fit(path)
        assert pd.isna(df.loc[0, "dist_m"])
        assert df.loc[1, "dist_m"] == pytest.approx(10.0)

    def test_reads_sensor_speed_preferring_enhanced(self, tmp_path, fake_fit):
        path = fake_fit(
            tmp_path / "ride.fit",
            [{**VALID_RECORD, "speed": 10.0, "enhanced_speed": 28.4}],
        )
        df = load_fit(path)
        assert df.loc[0, "speed_kmh"] == 28.4

    def test_omits_dist_m_and_speed_kmh_when_not_in_file(self, tmp_path, fake_fit):
        # with_distance()/with_speed() (see polkupy.geo.calc_distance and
        # calc_speed) treat a *present* dist_m/speed_kmh column as
        # sensor-provided and keep it as-is by default, so these columns
        # must only show up when the file actually has the data.
        path = fake_fit(tmp_path / "ride.fit", [VALID_RECORD])
        df = load_fit(path)
        assert "dist_m" not in df.columns
        assert "speed_kmh" not in df.columns

    def test_merges_records_split_across_same_timestamp(self, tmp_path, fake_fit):
        # observed on a Bosch head unit: GPS and other sensors land in
        # separate "record" messages that share one timestamp.
        path = fake_fit(
            tmp_path / "ride.fit",
            [
                {"timestamp": VALID_RECORD["timestamp"], "cadence": 80},
                {
                    "timestamp": VALID_RECORD["timestamp"],
                    "position_lat": 52.0,
                    "position_long": 13.0,
                },
            ],
        )
        df = load_fit(path, extensions=["cadence"])
        assert len(df) == 1
        assert df.loc[0, "lat"] == 52.0
        assert df.loc[0, "cadence"] == 80

    def test_drops_points_without_gps_fix(self, tmp_path, fake_fit):
        path = fake_fit(
            tmp_path / "ride.fit",
            [
                VALID_RECORD,
                {
                    "timestamp": datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                    "cadence": 80,
                },  # no GPS fix at this timestamp
            ],
        )
        df = load_fit(path)
        assert len(df) == 1

    def test_no_record_messages_raises_value_error(self, tmp_path, fake_fit):
        path = fake_fit(tmp_path / "ride.fit", [])
        with pytest.raises(ValueError, match="no record messages found"):
            load_fit(path)

    def test_no_gps_fix_raises_value_error(self, tmp_path, fake_fit):
        path = fake_fit(
            tmp_path / "ride.fit", [{"timestamp": VALID_RECORD["timestamp"]}]
        )
        with pytest.raises(ValueError, match="no record message has a GPS fix"):
            load_fit(path)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            pytest.param("position_lat", 120.0, "latitude=120.0", id="lat-too-high"),
            pytest.param("position_lat", -95.0, "latitude=-95.0", id="lat-too-low"),
            pytest.param("position_long", 200.0, "longitude=200.0", id="lon-too-high"),
            pytest.param("position_long", -200.0, "longitude=-200.0", id="lon-too-low"),
            pytest.param(
                "enhanced_altitude", 50000.0, "elevation=50000.0", id="ele-too-high"
            ),
            pytest.param(
                "enhanced_altitude", -9000.0, "elevation=-9000.0", id="ele-too-low"
            ),
        ],
    )
    def test_out_of_range_value_raises_value_error(
        self, tmp_path, fake_fit, field, value, match
    ):
        path = fake_fit(tmp_path / "ride.fit", [{**VALID_RECORD, field: value}])
        with pytest.raises(ValueError, match=match):
            load_fit(path)

    def test_boundary_values_are_valid(self, tmp_path, fake_fit):
        path = fake_fit(
            tmp_path / "ride.fit",
            [
                {
                    "timestamp": VALID_RECORD["timestamp"],
                    "position_lat": 90.0,
                    "position_long": -180.0,
                }
            ],
        )
        df = load_fit(path)
        assert df.loc[0, "lat"] == 90.0
        assert df.loc[0, "lon"] == -180.0

    def test_missing_elevation_is_nan(self, tmp_path, fake_fit):
        path = fake_fit(tmp_path / "ride.fit", [VALID_RECORD])
        df = load_fit(path)
        assert pd.isna(df.loc[0, "ele"])

    def test_reads_requested_extension_matching_enhanced_field(
        self, tmp_path, fake_fit
    ):
        # "speed" should match "enhanced_speed" too (see load_fit's docstring).
        path = fake_fit(
            tmp_path / "ride.fit", [{**VALID_RECORD, "enhanced_speed": 28.4}]
        )
        df = load_fit(path, extensions=["speed"])
        assert df.loc[0, "speed"] == 28.4

    def test_requested_extension_not_found_warns(self, tmp_path, fake_fit):
        path = fake_fit(tmp_path / "ride.fit", [VALID_RECORD])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_fit(path, extensions=["heart_rate"])

        assert len(caught) == 1
        assert "heart_rate" in str(caught[0].message)
        assert pd.isna(df.loc[0, "heart_rate"])


class TestLoadFitDir:
    """load_fit_dir() loads every FIT ride from per-bike subfolders."""

    def test_reads_every_bike_by_default(self, tmp_path, fake_fit):
        fake_fit(tmp_path / "ktm" / "ride1.fit", [VALID_RECORD])
        fake_fit(tmp_path / "canyon" / "ride2.fit", [VALID_RECORD])

        df = load_fit_dir(str(tmp_path))

        assert set(df["bike_id"]) == {"ktm", "canyon"}
        assert set(df["ride_id"]) == {"ktm/ride1", "canyon/ride2"}
        assert len(df) == 2

    def test_filters_to_requested_bikes(self, tmp_path, fake_fit):
        fake_fit(tmp_path / "ktm" / "ride1.fit", [VALID_RECORD])
        fake_fit(tmp_path / "canyon" / "ride2.fit", [VALID_RECORD])
        fake_fit(tmp_path / "brompton" / "ride3.fit", [VALID_RECORD])

        df = load_fit_dir(str(tmp_path), bikes=["ktm", "brompton"])

        assert set(df["bike_id"]) == {"ktm", "brompton"}

    def test_loads_multiple_files_per_bike(self, tmp_path, fake_fit):
        fake_fit(tmp_path / "ktm" / "a.fit", [VALID_RECORD])
        fake_fit(tmp_path / "ktm" / "b.fit", [VALID_RECORD])

        df = load_fit_dir(str(tmp_path))

        assert set(df["ride_id"]) == {"ktm/a", "ktm/b"}
        assert len(df) == 2

    def test_skips_invalid_file_with_warning(self, tmp_path, fake_fit):
        fake_fit(tmp_path / "ktm" / "good.fit", [VALID_RECORD])
        fake_fit(tmp_path / "ktm" / "bad.fit", [])  # no records -> ValueError

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_fit_dir(str(tmp_path))

        assert len(caught) == 1
        assert "Skipping ride" in str(caught[0].message)
        assert set(df["ride_id"]) == {"ktm/good"}

    def test_raises_when_no_rides_found(self, tmp_path):
        (tmp_path / "ktm").mkdir()  # bike folder exists but has no FIT files
        with pytest.raises(ValueError, match="no rides found"):
            load_fit_dir(str(tmp_path))

    def test_raises_when_data_dir_has_no_bike_folders(self, tmp_path):
        with pytest.raises(ValueError, match="no rides found"):
            load_fit_dir(str(tmp_path))

    def test_passes_extensions_through(self, tmp_path, fake_fit):
        fake_fit(tmp_path / "ktm" / "ride1.fit", [{**VALID_RECORD, "cadence": 80}])

        df = load_fit_dir(str(tmp_path), extensions=["cadence"])

        assert df.loc[0, "cadence"] == 80


class TestListFitFields:
    """list_fit_fields() lists every record field name found in a FIT file."""

    def test_lists_every_field_seen(self, tmp_path, fake_fit):
        path = fake_fit(tmp_path / "ride.fit", [{**VALID_RECORD, "cadence": 80}])
        assert list_fit_fields(path) == {
            "timestamp",
            "position_lat",
            "position_long",
            "cadence",
        }

    def test_unions_fields_across_split_records(self, tmp_path, fake_fit):
        # a field that only ever appears on a later, redefined "record"
        # message must still show up (see load_fit's docstring).
        path = fake_fit(
            tmp_path / "ride.fit",
            [VALID_RECORD, {"timestamp": VALID_RECORD["timestamp"], "power": 200}],
        )
        assert "power" in list_fit_fields(path)

    def test_empty_file_gives_empty_set(self, tmp_path, fake_fit):
        path = fake_fit(tmp_path / "ride.fit", [])
        assert list_fit_fields(path) == set()
