"""
Command-line entry point for generating a day animation.

Usage:
    python -m polkupy --bbox 48.00 48.25 11.20 11.70 --out output/munich_day.gif
"""

import argparse

from polkupy import load_rides, animate_day


def parse_args():
    parser = argparse.ArgumentParser(
        prog="polkupy",
        description="Animate cycling GPX rides within a geographic boundary, "
                     "replayed on a shared 24h clock regardless of the date "
                     "each ride was actually recorded on.",
    )
    parser.add_argument("--bbox", type=float, nargs=4, required=True,
                         metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                         help="Geographic boundary to crop the map to.")
    parser.add_argument("--data-dir", default="data",
                         help="Directory containing one subfolder per bike (default: data)")
    parser.add_argument("--bikes", nargs="+", default=None,
                         help="Subset of bike subfolders to load (default: all)")
    parser.add_argument("--out", dest="out_path", default="output/day_animation.gif",
                         help="Output GIF path (default: output/day_animation.gif)")
    parser.add_argument("--tz", default="Europe/Berlin",
                         help="IANA timezone for resolving local time of day (default: Europe/Berlin)")
    parser.add_argument("--step-minutes", type=float, default=5,
                         help="Virtual clock minutes advanced per frame (default: 5)")
    parser.add_argument("--fps", type=int, default=12,
                         help="Playback speed of the output GIF (default: 12)")
    parser.add_argument("--start-time", default=None,
                         help="Hour the animated clock should begin at, as a decimal "
                              "(6) or HH:MM (06:00). The animation then spans a full "
                              "24h loop, wrapping earlier rides around to the end "
                              "(default: natural span of the data, unshifted)")
    parser.add_argument("--loop", action="store_true",
                         help="Draw the whole route network from the first frame so "
                              "the GIF loops seamlessly, instead of building it up "
                              "over the course of the animation")
    parser.add_argument("--trail-minutes", type=float, default=20,
                         help="Minutes of recent riding shown as a bright, fading comet "
                              "trail behind each active ride's head dot (default: 20)")
    parser.add_argument("--trail-color", default="#9CA986",
                         help="Color for completed rides (default: #9CA986)")
    parser.add_argument("--active-color", default="#D98E3B",
                         help="Color for rides currently being drawn (default: #D98E3B)")
    parser.add_argument("--background-color", default="#FAF9F4",
                         help="Map background color (default: #FAF9F4)")
    parser.add_argument("--text-color", default="#3A3A32",
                         help="Clock/counter text color (default: #3A3A32)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6, 6),
                         metavar=("WIDTH", "HEIGHT"),
                         help="Figure size in inches (default: 6 6)")
    parser.add_argument("--dpi", type=int, default=130,
                         help="Figure resolution (default: 130)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading rides from {args.data_dir}...")
    df = load_rides(args.data_dir, bikes=args.bikes)
    print(f"Loaded {df['ride_id'].nunique()} rides ({len(df)} points).")

    out_path = animate_day(
        df,
        bbox=tuple(args.bbox),
        out_path=args.out_path,
        tz=args.tz,
        step_minutes=args.step_minutes,
        fps=args.fps,
        start_time=args.start_time,
        loop=args.loop,
        trail_minutes=args.trail_minutes,
        trail_color=args.trail_color,
        active_color=args.active_color,
        background_color=args.background_color,
        text_color=args.text_color,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
