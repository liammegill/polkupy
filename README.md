# polkupy

Python package to analyse and visualise personal cycling activities.

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

## Installation

Install with `uv` using:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```
