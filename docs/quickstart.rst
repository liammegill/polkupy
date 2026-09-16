Quickstart
==========

polkupy wraps your ride data in two chainable classes: :class:`~polkupy.Ride`
for a single track, and :class:`~polkupy.Trips` for a collection of them.
Both are thin wrappers around a :class:`pandas.DataFrame`, so ``Ride.data`` and
``Trips.data`` are always there if you need to drop down to plain
`pandas <https://pandas.pydata.org>`__ (for example to run a function not (yet)
implemented in polkupy).

A single ride
--------------

Load a single GPX or FIT file with :meth:`~polkupy.Ride.from_gpx` or
:meth:`~polkupy.Ride.from_fit`:

.. code-block:: python

   from polkupy import Ride

   ride = Ride.from_gpx("data/ktm/activity_12345.gpx", bike_id="ktm")
   ride  # in Jupyter: renders as a route map + metadata card

:class:`~polkupy.Ride` exposes a few metadata properties out of the box:

.. code-block:: python

   ride.start_time     # pandas.Timestamp of the first point
   ride.end_time       # pandas.Timestamp of the last point
   ride.duration       # pandas.Timedelta, elapsed (not moving) time
   ride.sampling_rate  # median interval between points
   ride.distance_km    # total great-circle distance covered

Methods that add data return a *new* :class:`~polkupy.Ride`, so they chain:

.. code-block:: python

   ride = ride.with_distance().with_speed().localised(tz="Europe/Berlin")
   ride.data.head()

:meth:`~polkupy.Ride.with_distance`/:meth:`~polkupy.Ride.with_speed` add
``dist_km``/``speed_kmh`` columns computed from GPS, unless the file already
carries sensor-reported values (FIT files often do). To force recomputation
from GPS data, pass ``overwrite=True``. :meth:`~polkupy.Ride.localised` aligns
the ride to a 24h clock, ignoring the calendar date, so rides on different days
become directly comparable.

You can also check whether a ride starts or ends near a given point:

.. code-block:: python

   ride.starts_near(48.14, 11.56, radius_km=0.5)
   ride.ends_near(48.35, 11.79, radius_km=0.5)


A collection of rides
-----------------------

To load every ride from a directory of per-bike subfolders (e.g.
``data/ktm/*.gpx``, ``data/gravel/*.gpx``, ...), use
:meth:`~polkupy.Trips.from_gpx` or :meth:`~polkupy.Trips.from_fit`:

.. code-block:: python

   from polkupy import Trips

   trips = Trips.from_gpx("data")          # every subfolder in data/
   trips = Trips.from_gpx("data", bikes=["ktm", "gravel"])  # a subset

:class:`~polkupy.Trips` is iterable, yielding one :class:`~polkupy.Ride` per
ride, and supports lookup by ``ride_id``:

.. code-block:: python

   len(trips)             # number of distinct rides
   for ride in trips:
       print(ride.ride_id, ride.distance_km)

   trips["ktm/activity_12345"]  # a single Ride

Filtering returns a new :class:`~polkupy.Trips`, so it also chains.
:meth:`~polkupy.Trips.between` is handy for finding recurring rides, e.g. a
commute:

.. code-block:: python

   commutes = trips.between(start=(48.14, 11.56), end=(48.35, 11.79))
   nearby = trips.in_bbox(lat_min=48.0, lat_max=48.5, lon_min=11.3, lon_max=11.9)
   long_rides = trips.filter(lambda ride: ride.distance_km > 50)

:meth:`~polkupy.Trips.localised` works the same way as on :class:`~polkupy.Ride`,
but across the whole collection at once, so every ride lands on a shared 24h
clock:

.. code-block:: python

   trips = trips.localised(tz="Europe/Berlin")

Combining rides
-----------------

:class:`~polkupy.Ride` and :class:`~polkupy.Trips` objects can be concatenated
with ``+`` or :func:`sum`, which is useful for building a collection up
manually or merging collections loaded from different sources (e.g. GPX and
FIT):

.. code-block:: python

   two_rides = ride_a + ride_b                # -> Trips
   all_rides = sum([ride_a, ride_b, ride_c])  # -> Trips
   combined = Trips.from_gpx("data") + Trips.from_fit("data")
