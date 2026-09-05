"""
Animated visualisations of cycling activity.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from polkupy.process import filter_rides_in_bbox, add_time_of_day


def _parse_hour(value) -> float:
    """Parse an hour given as a number or an 'HH:MM' string into decimal hours."""
    if isinstance(value, str):
        hh, mm = value.split(":")
        return int(hh) + int(mm) / 60
    return float(value)


def animate_day(df: pd.DataFrame, bbox: tuple[float, float, float, float],
                 out_path: str = "output/day_animation.gif", tz: str = "Europe/Berlin",
                 step_minutes: float = 5, fps: int = 12,
                 start_time=None, loop: bool = False, trail_minutes: float = 20,
                 trail_color: str = "#9CA986", active_color: str = "#D98E3B",
                 background_color: str = "#FAF9F4", text_color: str = "#3A3A32",
                 figsize: tuple[float, float] = (6, 6), dpi: int = 130) -> str:
    """Animate a day of riding within a geographic boundary.

    Every ride that crosses the bounding box is replayed on a shared 24h
    clock, ignoring which calendar date it actually happened on: a ride
    only appears once the clock reaches its start time of day, is drawn
    progressively over its own recorded duration, and then remains on the
    map as a faded trail for the rest of the animation.

    Args:
        df (pd.DataFrame): Ride points with 'ride_id', 'time', 'lat',
            'lon' columns, e.g. as returned by `load_rides`.
        bbox (tuple[float, float, float, float]): (lat_min, lat_max,
            lon_min, lon_max) boundary to crop the map to.
        out_path (str): Path to write the output GIF to.
        tz (str): IANA timezone name used to resolve local time of day.
        step_minutes (float): Virtual clock time advanced per frame.
        fps (int): Playback speed of the output GIF.
        start_time (float | str, optional): Hour the animated clock should
            begin at, as a decimal hour (6.0) or an "HH:MM" string
            ("06:00"). When given, the animation always spans a full 24h
            loop, wrapping any ride that starts earlier in the day around
            to the end (e.g. with start_time="06:00", a ride starting at
            05:00 appears near the very end of the loop). Defaults to the
            natural span actually covered by the cropped rides, unshifted.
        loop (bool): If True, the full route network is drawn from the
            first frame and stays constant throughout - only the
            currently-active routes are highlighted and animate on top of
            it - so the GIF loops seamlessly with no build-up/tear-down.
            If False (default), the network builds up progressively as
            rides start and finish.
        trail_minutes (float): How many minutes of recent riding to show as
            a bright comet trail behind each active ride's head dot. The
            trail fades from full active_color right behind the dot to
            fully transparent at trail_minutes old, drawn on top of the
            faded background network.
        trail_color, active_color (str): Colors for completed/background
            rides and rides currently being drawn.
        background_color, text_color (str): Map background and clock/
            counter text colors.
        figsize, dpi: Passed to `plt.subplots`.

    Returns:
        str: `out_path`, once the GIF has been written.
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    cropped = filter_rides_in_bbox(df, lat_min, lat_max, lon_min, lon_max)
    if cropped.empty:
        raise ValueError("No rides intersect the given bounding box.")

    cropped = add_time_of_day(cropped, tz=tz)
    cropped = cropped.sort_values(["ride_id", "virtual_s"])

    rides = {
        ride_id: group[["virtual_s", "lon", "lat"]].to_numpy()
        for ride_id, group in cropped.groupby("ride_id")
    }

    step_s = step_minutes * 60

    if start_time is not None:
        # anchor the loop at start_time and wrap any ride that starts
        # earlier in the day around to the end of the loop
        t_min = _parse_hour(start_time) * 3600
        for arr in rides.values():
            if arr[0, 0] < t_min:
                arr[:, 0] += 86400
        t_max = t_min + 86400
        frame_times = np.arange(t_min, t_max, step_s)
    else:
        t_min = min(arr[0, 0] for arr in rides.values())
        t_max = max(arr[-1, 0] for arr in rides.values())
        frame_times = np.arange(t_min, t_max + step_s, step_s)

    ride_start = {rid: arr[0, 0] for rid, arr in rides.items()}
    ride_end = {rid: arr[-1, 0] for rid, arr in rides.items()}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
    ax.axis("off")
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.02, right=0.98)

    lines = {
        rid: ax.plot([], [], lw=0.9, color=trail_color, alpha=0.35,
                     solid_capstyle="round", zorder=1)[0]
        for rid in rides
    }
    trails = {
        rid: ax.add_collection(LineCollection([], linewidths=1.12, capstyle="round", zorder=2))
        for rid in rides
    }
    heads = {
        rid: ax.plot([], [], marker="o", ms=3.5, color=active_color, lw=0, zorder=3)[0]
        for rid in rides
    }

    trail_s = trail_minutes * 60
    active_rgb = to_rgba(active_color)[:3]

    clock_text = fig.text(0.5, 0.94, "", ha="center", va="center",
                           fontsize=22, style="italic", family="serif", color=text_color)
    count_text = fig.text(0.5, 0.885, "", ha="center", va="center",
                           fontsize=10, color=text_color, alpha=0.7)

    def update(frame_idx):
        t = frame_times[frame_idx]
        n_active = 0
        for rid, arr in rides.items():
            visible = arr[arr[:, 0] <= t]
            active = ride_start[rid] <= t < ride_end[rid]

            if loop:
                # whole network always visible; only the head dot and trail animate
                lines[rid].set_data(arr[:, 1], arr[:, 2])
            elif visible.shape[0] == 0:
                lines[rid].set_data([], [])
                heads[rid].set_data([], [])
                trails[rid].set_segments([])
                continue
            else:
                lines[rid].set_data(visible[:, 1], visible[:, 2])

            if active:
                n_active += 1
                heads[rid].set_data([visible[-1, 1]], [visible[-1, 2]])
            else:
                heads[rid].set_data([], [])

            # the trail fades by point age relative to *now*, independent of
            # whether the ride is still active, so it keeps cooling down for
            # trail_minutes after the ride finishes instead of popping off
            trail_pts = visible[visible[:, 0] >= t - trail_s]
            if len(trail_pts) >= 2:
                segments = np.stack([trail_pts[:-1, 1:], trail_pts[1:, 1:]], axis=1)
                mid_t = (trail_pts[:-1, 0] + trail_pts[1:, 0]) / 2
                age_frac = np.clip((t - mid_t) / trail_s, 0, 1)
                alphas = np.exp(-5.0 * age_frac)  # fast initial fade, long faint taper
                colors = np.column_stack([np.tile(active_rgb, (len(alphas), 1)), alphas])
                trails[rid].set_segments(segments)
                trails[rid].set_color(colors)
            else:
                trails[rid].set_segments([])

        hh = int(t // 3600) % 24
        mm = int((t % 3600) // 60)
        clock_text.set_text(f"{hh}:{mm:02d}")
        count_text.set_text(f"{n_active} route{'s' if n_active != 1 else ''} in progress")
        return [*lines.values(), *trails.values(), *heads.values(), clock_text, count_text]

    ani = FuncAnimation(fig, update, frames=len(frame_times), blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return out_path
