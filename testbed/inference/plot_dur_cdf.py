#!/usr/bin/env python3
"""
Generate a CDF plot for durations extracted from a log file.

By default, the script looks for tokens like: dur_ns=123456
and builds an empirical CDF of those values.

Example:
  python3 plot_dur_cdf.py /path/to/client_0.log --xunit ms \
          --out /path/to/client_0_dur_cdf.png --csv /path/to/client_0_dur_ecdf.csv
"""

import argparse
import math
import os
import re
import sys
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CDF of durations from a log file")
    parser.add_argument("log", help="Path to log file to parse")
    parser.add_argument(
        "--regex",
        default=r"dur_ns=(\d+)",
        help="Regex with one capturing group for the duration value (default: dur_ns)",
    )
    parser.add_argument(
        "--xunit",
        choices=["ns", "us", "ms", "s"],
        default="ms",
        help="Unit for x-axis (converted from nanoseconds). Default: ms",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: <log_basename>_cdf.png in the same directory",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write ECDF data as CSV (dur,ecdf)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. Default: 'CDF of dur_ns - <log_basename>'",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable grid on the plot",
    )
    return parser.parse_args()


def load_durations_from_log(log_path: str, pattern: str) -> List[int]:
    compiled = re.compile(pattern)
    values: List[int] = []
    with open(log_path, "r") as f:
        for line in f:
            match = compiled.search(line)
            if match:
                values.append(int(match.group(1)))
    return values


def compute_ecdf(values: List[int]) -> Tuple[List[int], List[float]]:
    if not values:
        return [], []
    values_sorted = sorted(values)
    n = len(values_sorted)
    y = [(i + 1) / n for i in range(n)]
    return values_sorted, y


def compute_percentile(values_sorted: List[int], p: float) -> float:
    if not values_sorted:
        return float("nan")
    if p <= 0:
        return float(values_sorted[0])
    if p >= 100:
        return float(values_sorted[-1])
    n = len(values_sorted)
    k = (p / 100.0) * (n - 1)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(values_sorted[int(k)])
    return values_sorted[lower] + (values_sorted[upper] - values_sorted[lower]) * (k - lower)


def get_scale(unit: str) -> Tuple[float, str]:
    # Convert from nanoseconds to the unit
    if unit == "ns":
        return 1.0, "ns"
    if unit == "us":
        return 1e3, "µs"
    if unit == "ms":
        return 1e6, "ms"
    if unit == "s":
        return 1e9, "s"
    # Fallback
    return 1e6, "ms"


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.log):
        print(f"Log not found: {args.log}")
        return 2

    durations_ns = load_durations_from_log(args.log, args.regex)
    if not durations_ns:
        print(f"No values matched regex in {args.log}")
        return 1

    xs_ns, ys = compute_ecdf(durations_ns)

    # Stats in ns
    n = len(xs_ns)
    p50 = compute_percentile(xs_ns, 50)
    p90 = compute_percentile(xs_ns, 90)
    p99 = compute_percentile(xs_ns, 99)
    print(
        f"Count={n}\nmin={xs_ns[0]} ns, p50={p50:.0f} ns, p90={p90:.0f} ns, p99={p99:.0f} ns, max={xs_ns[-1]} ns"
    )

    # Convert unit for plotting
    scale, unit_label = get_scale(args.xunit)
    xs_scaled = [v / scale for v in xs_ns]

    # Write CSV if requested
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w") as out:
            out.write("dur,ecdf\n")
            for x, y in zip(xs_scaled, ys):
                out.write(f"{x},{y}\n")
        print(f"Wrote ECDF CSV -> {args.csv}")

    # Prepare output path
    if args.out:
        out_png = args.out
    else:
        base_dir = os.path.dirname(os.path.abspath(args.log))
        base_name = os.path.splitext(os.path.basename(args.log))[0]
        out_png = os.path.join(base_dir, f"{base_name}_dur_cdf.png")

    title = args.title or f"CDF of dur_ns - {os.path.basename(args.log)}"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required to plot. Install with: pip install matplotlib")
        print("Error:", e)
        return 3

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs_scaled, ys, drawstyle="steps-post")
    plt.xlabel(f"duration ({unit_label})")
    plt.ylabel("CDF")
    plt.title(title)
    if not args.no_grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot -> {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



