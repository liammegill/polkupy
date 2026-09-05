"""Generic geographic calculations shared across polkupy."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_M = 6371000  # mean Earth radius, in metres


def haversine(
    lon1: npt.ArrayLike, lat1: npt.ArrayLike, lon2: npt.ArrayLike, lat2: npt.ArrayLike
) -> npt.ArrayLike:
    """Great-circle distance between two points on Earth using the
    haversine formula.

    Args:
        lon1 (numpy.typing.ArrayLike): Longitude of the first point(s), in
            degrees.
        lat1 (numpy.typing.ArrayLike): Latitude of the first point(s), in
            degrees.
        lon2 (numpy.typing.ArrayLike): Longitude of the second point(s), in
            degrees.
        lat2 (numpy.typing.ArrayLike): Latitude of the second point(s), in
            degrees.

    Returns:
        numpy.typing.ArrayLike: Distance between the two points, in metres.
            Matches the shape of whichever argument(s) were array-like.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * EARTH_RADIUS_M
