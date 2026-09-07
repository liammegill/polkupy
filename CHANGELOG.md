# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-07

Initial release.

### Added

- **Core classes** (`polkupy.Ride`, `polkupy.Trips`), chainable wrappers around
  `pandas.DataFrame`s of GPS points:
  - `Ride`: a single track, with metadata (`start_time`, `end_time`,
    `duration`, `sampling_rate`, `distance_km`), geography checks
    (`starts_near`/`ends_near`), transforms (`localised`, `with_distance`,
    `with_speed`), and concatenation via `+`/`sum()`.
  - `Trips`: a collection of rides, with filtering (`filter`, `in_bbox`,
    `between`), transforms (`localised`), iteration/indexing by `ride_id`,
    and concatenation via `+`/`sum()`.
  - Rich Jupyter display for both: `Ride` renders as a small route map with
    start/end markers plus a metadata summary; `Trips` renders as a
    collection summary with a per-ride table.
- **GPX loading** (`polkupy.io.gpx`): `load_gpx`/`load_gpx_dir`, with
  validation of timestamps and physically plausible latitude/longitude/
  elevation ranges, and support for reading extra extension fields (e.g.
  heart rate).
- **Geography utilities** (`polkupy.geo`): haversine great-circle distance,
  and per-point distance/speed calculation from consecutive GPS points.
- **Clock utilities** (`polkupy.clock`): aligning rides recorded on
  different calendar days to a shared 24h clock for direct comparison.
- **Bounding-box filtering** (`polkupy.algorithms.filters`) for ride
  collections.
- Sphinx documentation (autodoc + napoleon + furo theme).
- `pixi`-based development environment (`default`/`dev`/`nb`/`docs`
  features) with `test`, `typecheck`, `lint`, `check`, and `docs-build`
  tasks.
- A comprehensive `pytest` suite.
- CI workflows: `quick-test` (single Python version, every push/PR),
  `pip-install-test` (Python 3.11-3.14 matrix, weekly + on push), and
  `publish` (build, then publish to TestPyPI/PyPI via Trusted Publishing).
