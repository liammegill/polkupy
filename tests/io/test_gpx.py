"""Tests for polkupy.io.gpx: load_gpx and load_gpx_dir.

gpxpy itself validates lat/lon (mandatory GPX attributes) at parse time and
raises gpxpy.gpx.GPXException; those tests just confirm load_gpx doesn't
swallow that. time is optional per the GPX schema, so gpxpy does *not*
validate it, and elevation/extensions are optional by design -- those are
the cases load_gpx has to check for itself.
"""

# pylint: disable=C0116

import warnings

import gpxpy.gpx
import pytest

from polkupy.io.gpx import load_gpx, load_gpx_dir

VALID_TRKPT = '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>'


class TestLoadGpx:
    """load_gpx() parses a single GPX file into a DataFrame."""

    def test_reads_core_columns(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><ele>34.5</ele>'
            "<time>2026-01-01T00:00:00Z</time></trkpt>",
        )
        df = load_gpx(path)
        assert list(df.columns) == ["time", "lat", "lon", "ele"]
        assert df.loc[0, "lat"] == 52.0
        assert df.loc[0, "lon"] == 13.0
        assert df.loc[0, "ele"] == 34.5

    def test_missing_time_raises_value_error(self, tmp_path, write_gpx):
        path = write_gpx(tmp_path / "ride.gpx", '<trkpt lat="52.0" lon="13.0"></trkpt>')
        with pytest.raises(ValueError, match="missing a timestamp"):
            load_gpx(path)

    def test_missing_lat_raises_gpx_exception(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        with pytest.raises(gpxpy.gpx.GPXException, match=path):
            load_gpx(path)

    def test_missing_lon_raises_gpx_exception(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        with pytest.raises(gpxpy.gpx.GPXException, match=path):
            load_gpx(path)

    @pytest.mark.parametrize(
        ("trkpt", "match"),
        [
            pytest.param(
                '<trkpt lat="120.0" lon="13.0">'
                "<time>2026-01-01T00:00:00Z</time></trkpt>",
                "latitude=120.0",
                id="lat-too-high",
            ),
            pytest.param(
                '<trkpt lat="-95.0" lon="13.0">'
                "<time>2026-01-01T00:00:00Z</time></trkpt>",
                "latitude=-95.0",
                id="lat-too-low",
            ),
            pytest.param(
                '<trkpt lat="52.0" lon="200.0">'
                "<time>2026-01-01T00:00:00Z</time></trkpt>",
                "longitude=200.0",
                id="lon-too-high",
            ),
            pytest.param(
                '<trkpt lat="52.0" lon="-200.0">'
                "<time>2026-01-01T00:00:00Z</time></trkpt>",
                "longitude=-200.0",
                id="lon-too-low",
            ),
            pytest.param(
                '<trkpt lat="52.0" lon="13.0"><ele>50000</ele>'
                "<time>2026-01-01T00:00:00Z</time></trkpt>",
                "elevation=50000.0",
                id="ele-too-high",
            ),
            pytest.param(
                '<trkpt lat="52.0" lon="13.0"><ele>-9000</ele>'
                "<time>2026-01-01T00:00:00Z</time></trkpt>",
                "elevation=-9000.0",
                id="ele-too-low",
            ),
        ],
    )
    def test_out_of_range_value_raises_value_error(
        self, tmp_path, write_gpx, trkpt, match
    ):
        path = write_gpx(tmp_path / "ride.gpx", trkpt)
        with pytest.raises(ValueError, match=match):
            load_gpx(path)

    def test_boundary_values_are_valid(self, tmp_path, write_gpx):
        """The range checks are inclusive.

        Exactly +-90 lat / +-180 lon are real, valid coordinates (the
        poles and the antimeridian).
        """
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="90.0" lon="-180.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        df = load_gpx(path)
        assert df.loc[0, "lat"] == 90.0
        assert df.loc[0, "lon"] == -180.0

    def test_no_track_points_raises_value_error(self, tmp_path, write_gpx):
        path = write_gpx(tmp_path / "empty.gpx", "")  # valid file, zero trkpts
        with pytest.raises(ValueError, match="no track points found"):
            load_gpx(path)

    def test_elevation_zero_is_not_coerced_to_none(self, tmp_path, write_gpx):
        """point.elevation is falsy at 0m.

        A naive `if elevation else None` would wrongly drop real
        sea-level readings.
        """
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><ele>0</ele>'
            "<time>2026-01-01T00:00:00Z</time></trkpt>",
        )
        df = load_gpx(path)
        assert df.loc[0, "ele"] == 0.0

    def test_missing_elevation_is_none(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        df = load_gpx(path)
        assert df.loc[0, "ele"] is None

    def test_reads_requested_extension(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time>'
            "<extensions>"
            "<gpxtpx:TrackPointExtension xmlns:gpxtpx="
            '"http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
            "<gpxtpx:hr>145</gpxtpx:hr>"
            "</gpxtpx:TrackPointExtension>"
            "</extensions></trkpt>",
        )
        df = load_gpx(path, extensions=["hr"])
        assert df.loc[0, "hr"] == 145.0

    def test_requested_extension_not_found_warns(self, tmp_path, write_gpx):
        path = write_gpx(
            tmp_path / "ride.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time></trkpt>',
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_gpx(path, extensions=["heart_rate"])

        assert len(caught) == 1
        assert "heart_rate" in str(caught[0].message)
        assert df.loc[0, "heart_rate"] is None


class TestLoadGpxDir:
    """load_gpx_dir() loads every GPX ride from per-bike subfolders."""

    def test_reads_every_bike_by_default(self, tmp_path, write_gpx):
        write_gpx(tmp_path / "ktm" / "ride1.gpx", VALID_TRKPT)
        write_gpx(tmp_path / "canyon" / "ride2.gpx", VALID_TRKPT)

        df = load_gpx_dir(str(tmp_path))

        assert set(df["bike_id"]) == {"ktm", "canyon"}
        assert set(df["ride_id"]) == {"ktm/ride1", "canyon/ride2"}
        assert len(df) == 2

    def test_filters_to_requested_bikes(self, tmp_path, write_gpx):
        write_gpx(tmp_path / "ktm" / "ride1.gpx", VALID_TRKPT)
        write_gpx(tmp_path / "canyon" / "ride2.gpx", VALID_TRKPT)
        write_gpx(tmp_path / "brompton" / "ride3.gpx", VALID_TRKPT)

        df = load_gpx_dir(str(tmp_path), bikes=["ktm", "brompton"])

        assert set(df["bike_id"]) == {"ktm", "brompton"}

    def test_loads_multiple_files_per_bike(self, tmp_path, write_gpx):
        write_gpx(tmp_path / "ktm" / "a.gpx", VALID_TRKPT)
        write_gpx(tmp_path / "ktm" / "b.gpx", VALID_TRKPT)

        df = load_gpx_dir(str(tmp_path))

        assert set(df["ride_id"]) == {"ktm/a", "ktm/b"}
        assert len(df) == 2

    def test_skips_invalid_file_with_warning(self, tmp_path, write_gpx):
        write_gpx(tmp_path / "ktm" / "good.gpx", VALID_TRKPT)
        write_gpx(
            tmp_path / "ktm" / "bad.gpx", '<trkpt lat="52.0" lon="13.0"></trkpt>'
        )  # missing timestamp -> ValueError

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_gpx_dir(str(tmp_path))

        assert len(caught) == 1
        assert "Skipping ride" in str(caught[0].message)
        assert set(df["ride_id"]) == {"ktm/good"}

    def test_raises_when_no_rides_found(self, tmp_path):
        (tmp_path / "ktm").mkdir()  # bike folder exists but has no GPX files
        with pytest.raises(ValueError, match="no rides found"):
            load_gpx_dir(str(tmp_path))

    def test_raises_when_data_dir_has_no_bike_folders(self, tmp_path):
        with pytest.raises(ValueError, match="no rides found"):
            load_gpx_dir(str(tmp_path))

    def test_passes_extensions_through(self, tmp_path, write_gpx):
        write_gpx(
            tmp_path / "ktm" / "ride1.gpx",
            '<trkpt lat="52.0" lon="13.0"><time>2026-01-01T00:00:00Z</time>'
            "<extensions>"
            "<gpxtpx:TrackPointExtension xmlns:gpxtpx="
            '"http://www.garmin.com/xmlschemas/TrackPointExtension/v1">'
            "<gpxtpx:hr>145</gpxtpx:hr>"
            "</gpxtpx:TrackPointExtension>"
            "</extensions></trkpt>",
        )

        df = load_gpx_dir(str(tmp_path), extensions=["hr"])

        assert df.loc[0, "hr"] == 145.0
