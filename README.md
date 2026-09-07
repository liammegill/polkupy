# polkupy

[![Latest tag](https://img.shields.io/github/v/tag/liammegill/polkupy)](https://github.com/liammegill/polkupy/tags)
[![Commits since last release](https://img.shields.io/github/commits-since/liammegill/polkupy/latest.svg)](https://github.com/liammegill/polkupy/commits/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![License](https://img.shields.io/github/license/liammegill/polkupy)](https://github.com/liammegill/polkupy/blob/main/LICENSE)

Python package to analyse and visualise personal cycling activities. Loads
GPX tracks into chainable `pandas`-backed objects, with built-in filtering,
geography helpers, and rich Jupyter display.

## Installation

Install with `pixi` using:

```bash
pixi install --all
pixi shell -e dev
```

## Quick start

```python
from polkupy import Ride, Trips

ride = Ride.from_gpx("path/to/ride.gpx")
ride  # -> route map + metadata in Jupyter

trips = Trips.from_gpx("data")  # one subfolder per bike
commutes = trips.between(start=(48.14, 11.56), end=(48.35, 11.79))
```

## Structure

`Ride`/`Trips` wrap the underlying pandas DataFrame(s) in a chainable API.

```text
src/polkupy/
├── __init__.py       # public API: Ride, Trips
├── core/
│   ├── ride.py       # Ride: one track
│   └── trips.py      # Trips: a collection of rides
├── io/
│   └── gpx.py
├── algorithms/
│   └── filters.py
├── clock.py
└── geo.py
```

## Development

```bash
pixi run -e dev test         # pytest
pixi run -e dev check        # mypy + pylint
pixi run -e dev docs-build   # Sphinx HTML docs -> docs/_build/html
```

See [CHANGELOG.md](CHANGELOG.md) for release history.
