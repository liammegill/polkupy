Examples
========

Synthetic commute: Munich Hbf to Pasing
---------------------------------------

The example below uses :func:`~polkupy.synthetic.generate_track` rather
than a real recorded ride, so it contains no personal information and can
be freely shared in these docs. The route is a synthetic version of a real
commute - roughly following Landsberger Straße from Munich Hauptbahnhof to
Pasing Bahnhof - built from just three waypoints, a constant speed, and a
bit of GPS/speed sensor noise:

.. jupyter-execute::

   from polkupy import Ride
   from polkupy.synthetic import generate_track

   df = generate_track(
       [(48.1393, 11.5579), (48.1427, 11.5033), (48.1488, 11.4612)],
       speed_kmh=15.0,
       elevation_m=550,
       gps_noise_m=5.0,
       speed_noise_kmh=1.0,
       seed=42,
   )
   ride = Ride(df, ride_id="commute")
   ride

That last line renders ``ride`` as a small route map plus a metadata card,
just like it would for a ride loaded with :meth:`~polkupy.Ride.from_gpx`
and :meth:`~polkupy.Ride.from_fit` (see also :doc:`quickstart`).
``elevation_m`` and ``speed_noise_kmh`` were both given above, so ``df``
already carries ``ele`` and ``speed_kmh`` columns alongside the required
``time``, ``lat`` and ``lon``:

.. jupyter-execute::

   df.columns.tolist()

:class:`~polkupy.Ride`'s metadata properties work exactly as they
would on a real ride, with no code path aware that this one is synthetic:

.. jupyter-execute::

   ride.distance_km, ride.duration, ride.sampling_rate

``speed_kmh`` here simulates a device's own speed sensor, independent of
GPS. Because it's already present, :meth:`~polkupy.Ride.with_speed`
keeps it as-is rather than recomputing from position - pass
``overwrite=True`` to compare the two:

.. jupyter-execute::

   gps_speed = ride.with_speed(overwrite=True)
   ride.data["speed_kmh"].mean(), gps_speed.data["speed_kmh"].mean()

Finally, plotting the reported speed over the course of the ride:

.. jupyter-execute::

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   ax.plot(ride.data["time"], ride.data["speed_kmh"])
   ax.set_xlabel("time")
   ax.set_ylabel("speed (km/h)")
