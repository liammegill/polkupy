# polkupy
Python package to visualise personal cycling activities.

Load GPX rides from Strava-style exports and generate an animated "day in
the life" GIF of your riding within a geographic boundary: every ride is
replayed on a shared 24h clock (ignoring which calendar date it actually
happened on), appearing as a fading comet trail once the clock reaches its
start time and building into a permanent network of past routes.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from polkupy import load_rides, animate_day

df = load_rides("data")  # expects data/<bike>/*.gpx
animate_day(df, bbox=(48.00, 48.25, 11.20, 11.70), out_path="output/munich_day.gif")
```

Or from the command line:

```bash
python -m polkupy --bbox 48.00 48.25 11.20 11.70 --out output/munich_day.gif
```

Run `python -m polkupy --help` for all options (start time, seamless
looping, trail length, colors, timezone, etc.).
