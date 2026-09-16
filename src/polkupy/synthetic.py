"""Generate synthetic ride data, e.g. for test fixtures or documentation examples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .geo import haversine

DEFAULT_START = "2026-01-01T08:00:00Z"
_METRES_PER_DEGREE_LAT = 111_320.0


def generate_track(  # noqa: PLR0913
    waypoints: Sequence[tuple[float, float]],
    *,
    start: str | pd.Timestamp = DEFAULT_START,
    speed_kmh: float = 20.0,
    sample_rate_s: float = 5.0,
    elevation_m: float | Sequence[float] | None = None,
    gps_noise_m: float = 0.0,
    elevation_noise_m: float = 0.0,
    speed_noise_kmh: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic ride by interpolating between waypoints at constant speed.

    This is a straight-line ("as the crow flies") interpolation between
    consecutive ``waypoints``, not a road-following route - useful for test
    fixtures and documentation examples where a plausible-looking, GPS-shaped
    track is needed without real ride data (which could contain personal
    information).

    Args:
        waypoints (Sequence[tuple[float, float]]): ``(lat, lon)`` points the
            track passes through, in order. Must contain at least 2.
        start (str | pandas.Timestamp): Timestamp of the first point. If
            no timezone is provided, then treated as UTC.
        speed_kmh (float): Constant speed used to convert the waypoints'
            total distance into elapsed time. Must be positive.
        sample_rate_s (float): Interval between generated points, in
            seconds. Must be positive.
        elevation_m (float | Sequence[float], optional): Either a single value
            broadcast to all waypoints, or a sequence of elevations, in metres.
            Defaults to ``None``, which omits the ``ele`` column.
        gps_noise_m (float, optional): Standard deviation of gaussian noise
            added to each point's position, in metres, simulating GPS jitter.
            Defaults to ``0.0`` (no noise).
        elevation_noise_m (float, optional): Standard deviation of gaussian
            noise added to each point's elevation, in metres. Only applied if
            `elevation_m` is given. Defaults to ``0.0`` (no noise).
        speed_noise_kmh (float, optional): Standard deviation of gaussian
            noise added around ``speed_kmh`` to produce a reported
            ``speed_kmh`` column, simulating a device's own speed sensor.
            Defaults to ``0.0``, which omits the ``speed_kmh`` column.
        seed (int, optional): Seed for the random noise generator, for
            reproducible output. Defaults to ``None`` (nondeterministic).

    Returns:
        pandas.DataFrame: Points with ``time``, ``lat``, ``lon`` columns
        (plus ``ele``/``speed_kmh`` if `elevation_m`/`speed_noise_kmh` are
        given). Identical in shape to what :class:`~polkupy.core.ride.Ride`
        expects.

    Raises:
        ValueError: If fewer than 2 ``waypoints`` are given, or ``speed_kmh``
            or ``sample_rate_s`` is not positive.
    """
    # check inputs
    if len(waypoints) < 2:  # noqa: PLR2004
        raise ValueError("waypoints must contain at least 2 points")
    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be positive")
    if sample_rate_s <= 0:
        raise ValueError("sample_rate_s must be positive")

    lats = np.array([lat for lat, _ in waypoints], dtype=float)
    lons = np.array([lon for _, lon in waypoints], dtype=float)
    cum_dist_km = _cumulative_distance_km(lats, lons)

    # interpolate between lat/lon waypoints
    total_duration_s = float(cum_dist_km[-1]) / speed_kmh * 3600.0
    elapsed_s = np.arange(0.0, total_duration_s, sample_rate_s)
    if elapsed_s.size == 0 or elapsed_s[-1] != total_duration_s:
        elapsed_s = np.append(elapsed_s, total_duration_s)
    target_dist_km = elapsed_s / 3600.0 * speed_kmh
    lat = np.interp(target_dist_km, cum_dist_km, lats)
    lon = np.interp(target_dist_km, cum_dist_km, lons)

    # add normal noise to lat/lon values to simulate GPS jitter
    rng = np.random.default_rng(seed)
    if gps_noise_m:
        lat = lat + rng.normal(0.0, gps_noise_m, lat.shape) / _METRES_PER_DEGREE_LAT
        metres_per_degree_lon = _METRES_PER_DEGREE_LAT * np.cos(np.radians(lat))
        lon = lon + rng.normal(0.0, gps_noise_m, lon.shape) / metres_per_degree_lon

    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    time = start_ts + pd.to_timedelta(elapsed_s, unit="s")

    # create dictionary and add elevation
    data: dict[str, object] = {"time": time, "lat": lat, "lon": lon}
    if elevation_m is not None:
        elevations = _broadcast_to_waypoints(elevation_m, n_waypoints=len(waypoints))
        ele = np.interp(target_dist_km, cum_dist_km, elevations)
        if elevation_noise_m:
            ele = ele + rng.normal(0.0, elevation_noise_m, ele.shape)
        data["ele"] = ele

    # add a reported speed_kmh column
    if speed_noise_kmh:
        data["speed_kmh"] = speed_kmh + rng.normal(
            0.0, speed_noise_kmh, elapsed_s.shape
        )

    return pd.DataFrame(data)


def _broadcast_to_waypoints(
    values: float | Sequence[float], *, n_waypoints: int
) -> npt.NDArray[np.float64]:
    """Broadcast a scalar to every waypoint, or validate a per-waypoint sequence.

    Args:
        values (float | Sequence[float]): A single value to use for every
            waypoint, or one value per waypoint.
        n_waypoints (int): Expected length if ``values`` is a sequence.

    Returns:
        numpy.typing.NDArray[numpy.float64]: One value per waypoint.

    Raises:
        ValueError: If ``values`` is a sequence whose length doesn't match
            ``n_waypoints``.
    """
    if isinstance(values, int | float):
        return np.full(n_waypoints, float(values))
    if len(values) != n_waypoints:
        raise ValueError(
            f"elevation_m has {len(values)} value(s), expected 1 or {n_waypoints} "
            "(one per waypoint)"
        )
    return np.array(values, dtype=float)


def _cumulative_distance_km(
    lats: npt.NDArray[np.float64], lons: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Cumulative great-circle distance along a polyline, in km.

    Args:
        lats (numpy.typing.NDArray[numpy.float64]): Latitudes of the
            polyline's vertices, in degrees.
        lons (numpy.typing.NDArray[numpy.float64]): Longitudes of the
            polyline's vertices, in degrees.

    Returns:
        numpy.typing.NDArray[numpy.float64]: Distance from the first vertex
        to each vertex, in km. Starts at ``0.0``.
    """
    segment_km = haversine(lons[:-1], lats[:-1], lons[1:], lats[1:]) / 1000.0
    return np.concatenate([[0.0], np.cumsum(segment_km)])
