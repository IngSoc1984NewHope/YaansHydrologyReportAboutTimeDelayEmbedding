# -*- coding: utf-8 -*-
"""Sample sin(t) for t in (0, 10) and write at most 100 rows to Exp1Sin.csv."""

import csv
import math
from pathlib import Path

# Keep the sample size at or below the required limit of 100 points.
N_POINTS = 100
T_LOWER = 0.0
T_UPPER = 10.0
OUTPUT_PATH = Path(__file__).with_name("Exp1Sin.csv")


def interior_times(n_points, t_lower, t_upper):
    """Return n_points values spaced evenly inside the open interval (t_lower, t_upper)."""
    if not 1 <= n_points <= 100:
        raise ValueError("The sample must contain between 1 and 100 data points.")
    # Split (t_lower, t_upper) into n_points + 1 equal gaps and drop both endpoints.
    step = (t_upper - t_lower) / (n_points + 1)
    return [t_lower + (i + 1) * step for i in range(n_points)]


def main():
    times = interior_times(N_POINTS, T_LOWER, T_UPPER)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "sin(t)"])
        for t in times:
            writer.writerow([f"{t:.10g}", f"{math.sin(t):.10g}"])
    print(f"Wrote {len(times)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
