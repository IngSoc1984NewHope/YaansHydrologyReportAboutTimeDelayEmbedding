# -*- coding: utf-8 -*-
"""Sample sin(t) on [T_LOWER, T_UPPER] and write the rows to Exp1Sin.csv.

Change N_POINTS, T_LOWER, and T_UPPER below to resize or move the sample.
"""

import csv
import math
from pathlib import Path

# Number of samples, including both endpoints of [T_LOWER, T_UPPER].
N_POINTS = 101
T_LOWER = 0.0
T_UPPER = 10.0
OUTPUT_PATH = Path(__file__).with_name("Exp1Sin.csv")


def sample_times(n_points, t_lower, t_upper):
    """Return n_points values spaced evenly on the closed interval [t_lower, t_upper]."""
    if type(n_points) is not int or n_points < 1:
        raise ValueError("N_POINTS must be a positive integer.")
    if n_points == 1:
        return [t_lower]
    # Include both endpoints, with equal gaps between neighbouring samples.
    step = (t_upper - t_lower) / (n_points - 1)
    return [t_lower + i * step for i in range(n_points)]


def main():
    times = sample_times(N_POINTS, T_LOWER, T_UPPER)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "sin(t)"])
        for t in times:
            writer.writerow([f"{t:.10g}", f"{math.sin(t):.10g}"])
    print(f"Wrote {len(times)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
