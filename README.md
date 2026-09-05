# polkupy

Python package to analyse and visualise personal cycling activities.

## Structure

`Ride`/`Rides` wrap the underlying pandas DataFrame(s) in a chainable API.

```text
src/polkupy/
├── __init__.py       # public API: Ride, Rides
├── core/
│   ├── ride.py       # Ride: one track
│   └── rides.py      # Rides: a collection of rides
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
