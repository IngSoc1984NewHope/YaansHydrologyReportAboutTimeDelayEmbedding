# -*- coding: utf-8 -*-
"""Choose a delay and a connection threshold, then draw the Vietoris-Rips complex.

This script runs on its own. EMBEDDING_DELAY and the Rips threshold are
selected from the series. The other experiment controls are set below.
"""

import csv
from pathlib import Path

import gudhi
import matplotlib

# Write the figure to a file. An interactive window is not required.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.distance import cdist

# =============================================================================
# Controls
# These are the values most likely to change between experiments.
# =============================================================================

# Source table. Row order is the time order. A non-numeric first row is a header.
#DATASET_PATH = Path(__file__).with_name("Exp1Sin.csv")
DATASET_PATH = Path(__file__).with_name("MaeChaem_Q_H.csv")

# Scalar series to embed. None selects the file when it has exactly one column.
#SERIES_COLUMN = "sin(t)"
SERIES_COLUMN = "Q"

# Timestamp column. None uses the other column when the file has two columns.
# A file with only the series is treated as evenly spaced with step 1.
#TIME_COLUMN = "t"
TIME_COLUMN = "Date"

# Binary delay embedding uses dimension 2: each point is
# (x[t], x[t + delay], ..., x[t + (dimension - 1) * delay]).
EMBEDDING_DIMENSION = 2

# EMBEDDING_DELAY is chosen from the series by this rule.
# "average_displacement" is the reconstruction-expansion rule of
# Rosenstein, Collins and De Luca, Physica D 73 (1994) 82-98.
# Add a branch in select_embedding_delay before choosing another name.
DELAY_METHOD = "average_displacement"

# The chosen lag is the first one whose average-displacement slope is below
# this fraction of the initial slope. The 1994 paper uses 0.4.
SLOPE_FRACTION = 0.4

# Largest lag, in samples, included in the search.
# None searches every lag that still leaves one embedding window.
MAX_CANDIDATE_DELAY = None

# Pairwise distance that decides which edges are admitted.
# "euclidean" is the ordinary Euclidean distance.
# A later distance (for example a geodesic approximation) can be added as
# another branch in pairwise_distance_matrix, then selected by this name.
DISTANCE = "euclidean"

# Rule for the Rips threshold used in the figure and the reported complex.
# "connected_loop" is the smallest radius at which
# beta_0 <= 0.1 * (time steps - empty steps - EMBEDDING_DELAY)
# and the complex still has a one-dimensional loop.
# Add a branch in select_threshold before choosing another name.
THRESHOLD_METHOD = "connected_loop"

# Fraction of the effective sample allowed to remain as separate components.
COMPONENT_FRACTION = 0.1

# Upper bound of the one Vietoris-Rips filtration used to read persistence.
# This is not the selected threshold. Increase it when no feasible threshold
# is found inside the computed filtration.
FILTRATION_MAX_EDGE_LENGTH = 9

# A selected loop must still be alive for at least this much additional radius.
# 0 keeps every loop that is present at the threshold. A positive value rejects
# a loop that dies too soon after the threshold.
MIN_LOOP_LIFETIME = 0.0

# Highest simplex dimension to materialise.
# 0 = vertices, 1 = edges, 2 = triangles.
MAX_SIMPLEX_DIMENSION = 2

# Homology is reported for H_0, ..., H_MAX_HOMOLOGY_DIMENSION.
# Computing H_k uses simplices of dimension k and k + 1, so this value
# should be at most MAX_SIMPLEX_DIMENSION. When the two are equal, classes
# in the top degree have no higher simplex that can fill them.
MAX_HOMOLOGY_DIMENSION = 1

# Prime used as the coefficient field Z/pZ. Betti numbers over Z/2Z are the usual choice.
HOMOLOGY_COEFFICIENT_FIELD = 2

# Figure dimension: 2, 3, or "auto".
# "auto" draws a 2D figure for a 2D point cloud and a 3D figure otherwise.
FIGURE_DIMENSION = "auto"

# "auto" saves "<dataset stem>_auto_complex.png" next to the dataset.
# Set a file name or path to choose the location.
FIGURE_PATH = "auto"

# Projection used when the point cloud has more axes than the figure.
# "pca" keeps the principal directions of largest variance.
PROJECTION = "pca"

# The four triangular faces of a tetrahedron, by vertex position inside the simplex.
TETRAHEDRON_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
VERTEX_COLOR = "#222222"
EDGE_COLOR = "#C0392B"
TRIANGLE_COLOR = "#4C78A8"
TETRAHEDRON_COLOR = "#E6A817"


def validate_controls():
    """Reject control values that cannot define a Vietoris-Rips homology run."""
    if SERIES_COLUMN is not None and not isinstance(SERIES_COLUMN, str):
        raise ValueError("SERIES_COLUMN must be a column name or None.")
    if TIME_COLUMN is not None and (not isinstance(TIME_COLUMN, str) or not TIME_COLUMN):
        raise ValueError("TIME_COLUMN must be a column name or None.")
    if type(EMBEDDING_DIMENSION) is not int or EMBEDDING_DIMENSION < 1:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer.")
    if not isinstance(DELAY_METHOD, str) or not DELAY_METHOD:
        raise ValueError("DELAY_METHOD must be a non-empty method name.")
    if not np.isfinite(SLOPE_FRACTION):
        raise ValueError("SLOPE_FRACTION must be a finite number.")
    if MAX_CANDIDATE_DELAY is not None and (
        type(MAX_CANDIDATE_DELAY) is not int or MAX_CANDIDATE_DELAY < 1
    ):
        raise ValueError("MAX_CANDIDATE_DELAY must be a positive integer or None.")
    if not isinstance(DISTANCE, str) or not DISTANCE:
        raise ValueError("DISTANCE must be a non-empty distance name.")
    if not isinstance(THRESHOLD_METHOD, str) or not THRESHOLD_METHOD:
        raise ValueError("THRESHOLD_METHOD must be a non-empty method name.")
    if not np.isfinite(FILTRATION_MAX_EDGE_LENGTH) or FILTRATION_MAX_EDGE_LENGTH < 0:
        raise ValueError("FILTRATION_MAX_EDGE_LENGTH must be a finite number greater than or equal to 0.")
    if not np.isfinite(MIN_LOOP_LIFETIME) or MIN_LOOP_LIFETIME < 0:
        raise ValueError("MIN_LOOP_LIFETIME must be a finite number greater than or equal to 0.")
    if type(MAX_SIMPLEX_DIMENSION) is not int or MAX_SIMPLEX_DIMENSION < 0:
        raise ValueError("MAX_SIMPLEX_DIMENSION must be a non-negative integer.")
    if type(MAX_HOMOLOGY_DIMENSION) is not int or MAX_HOMOLOGY_DIMENSION < 0:
        raise ValueError("MAX_HOMOLOGY_DIMENSION must be a non-negative integer.")
    if MAX_HOMOLOGY_DIMENSION > MAX_SIMPLEX_DIMENSION:
        raise ValueError(
            "MAX_HOMOLOGY_DIMENSION cannot exceed MAX_SIMPLEX_DIMENSION. "
            "H_k is computed from simplices of dimension k and k + 1."
        )
    if type(HOMOLOGY_COEFFICIENT_FIELD) is not int or HOMOLOGY_COEFFICIENT_FIELD < 2:
        raise ValueError("HOMOLOGY_COEFFICIENT_FIELD must be a prime number greater than or equal to 2.")
    if FIGURE_DIMENSION not in ("auto", 2, 3):
        raise ValueError("FIGURE_DIMENSION must be 'auto', 2, or 3.")
    if not isinstance(FIGURE_PATH, (str, Path)) or str(FIGURE_PATH).strip() == "":
        raise ValueError("FIGURE_PATH must be 'auto' or a file path.")
    if PROJECTION != "pca":
        raise ValueError(
            "Unknown projection. Implemented projections: 'pca'. "
            "Add another branch in display_coordinates before selecting it."
        )


# Blank cells and these tokens are missing observations, not numbers.
MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a", "#n/a"}


def parse_cell(cell, path, row_number, column_number):
    """Parse one CSV cell. An empty or missing token becomes NaN."""
    text = cell.strip()
    if text.lower() in MISSING_TOKENS:
        return np.nan
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value {cell!r} in {path.name} at data row {row_number}, "
            f"column {column_number}."
        ) from exc


def load_point_cloud(path):
    """Load a CSV table. Missing cells are NaN. Returns shape (n_rows, n_columns) and names."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.reader(handle)
            if row and any(cell.strip() for cell in row)
        ]
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    def row_is_numeric(row):
        try:
            for column_number, cell in enumerate(row, start=1):
                parse_cell(cell, path, 1, column_number)
        except ValueError:
            return False
        return True

    if row_is_numeric(rows[0]):
        columns = [f"x{index}" for index in range(len(rows[0]))]
        data_rows = rows
    else:
        columns = [cell.strip() or f"x{index}" for index, cell in enumerate(rows[0])]
        data_rows = rows[1:]

    if not data_rows:
        raise ValueError(f"Dataset has a header but no data rows: {path}")

    points = []
    for row_number, row in enumerate(data_rows, start=1):
        # A line such as "1.2," or "1.2" has an empty final field.
        if len(row) < len(columns):
            row = row + [""] * (len(columns) - len(row))
        elif len(row) > len(columns):
            raise ValueError(f"Row {row_number} of {path.name} has more than {len(columns)} columns.")
        points.append(
            [
                parse_cell(cell, path, row_number, column_number)
                for column_number, cell in enumerate(row, start=1)
            ]
        )

    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[0] == 0:
        raise ValueError(f"Dataset does not contain a point cloud: {path}")
    return cloud, columns


def resolve_time_column(columns, series_name):
    """Return the timestamp column, or None when the series itself supplies the clock."""
    if TIME_COLUMN is not None:
        if TIME_COLUMN not in columns:
            names = ", ".join(columns)
            raise ValueError(f"Column {TIME_COLUMN!r} is not in the dataset. Columns: {names}.")
        if TIME_COLUMN == series_name:
            raise ValueError("TIME_COLUMN and SERIES_COLUMN must name different columns.")
        return TIME_COLUMN
    others = [name for name in columns if name != series_name]
    if len(others) == 1:
        return others[0]
    if len(columns) == 1:
        return None
    names = ", ".join(columns)
    raise ValueError(f"Set TIME_COLUMN to one of: {names}.")


def load_timed_series(path, column_name):
    """Return timestamps, series values, and the series name.

    Empty series values stay in place as NaN. Their timestamps still define the
    sampling interval, so a later delay can count real time steps across a gap.
    """
    cloud, columns = load_point_cloud(path)
    if column_name is None:
        if len(columns) != 1:
            names = ", ".join(columns)
            raise ValueError(f"Set SERIES_COLUMN to one of: {names}.")
        series_name = columns[0]
        values = cloud[:, 0]
    else:
        if column_name not in columns:
            names = ", ".join(columns)
            raise ValueError(f"Column {column_name!r} is not in {path.name}. Columns: {names}.")
        series_name = column_name
        values = cloud[:, columns.index(column_name)]
    if np.isinf(values).any():
        raise ValueError(f"Column {series_name!r} in {path.name} contains an infinite value.")

    time_name = resolve_time_column(columns, series_name)
    if time_name is None:
        times = np.arange(len(values), dtype=float)
    else:
        times = cloud[:, columns.index(time_name)]
        if np.isinf(times).any():
            raise ValueError(f"Column {time_name!r} in {path.name} contains an infinite value.")
    if not np.isfinite(values).any():
        raise ValueError(f"Column {series_name!r} in {path.name} has no numeric values.")
    return times, values, series_name, time_name


def align_to_time_grid(times, values, series_name, time_name):
    """Place observations on a regular time grid. Missing values remain NaN.

    The step is the median positive gap between successive timestamps, including
    timestamps whose series value is empty.
    """
    if time_name is None:
        print(f"Time column: none. Time step: 1")
        return 0.0, 1.0, np.asarray(values, dtype=float)

    finite_time = np.isfinite(times)
    if not finite_time.all():
        print(f"Omitted {int((~finite_time).sum())} row(s) with an empty {time_name} value.")
    times = np.asarray(times[finite_time], dtype=float)
    values = np.asarray(values[finite_time], dtype=float)
    if len(times) == 0:
        raise ValueError(f"Column {time_name!r} has no numeric timestamps.")
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]
    if len(times) == 1:
        step = 1.0
    else:
        gaps = np.diff(times)
        positive = gaps[gaps > 0]
        if len(positive) == 0:
            raise ValueError(f"Column {time_name!r} does not advance in time.")
        step = float(np.median(positive))
    origin = float(times[0])
    slots = np.rint((times - origin) / step).astype(int)
    deviation = np.abs(times - (origin + slots * step))
    if np.any(deviation > 0.05 * step):
        raise ValueError(
            f"Timestamps in {time_name!r} are not equally spaced, "
            "so a delay cannot be counted in equal time steps."
        )
    grid = np.full(int(slots.max()) + 1, np.nan)
    for slot, value in zip(slots, values):
        current = grid[slot]
        if np.isfinite(current) and np.isfinite(value) and not np.isclose(current, value):
            raise ValueError(f"Column {time_name!r} repeats a timestamp with two different values.")
        if np.isfinite(value):
            grid[slot] = value
    missing = int(np.isnan(grid).sum())
    print(f"Time column: {time_name}")
    print(f"Time step: {step:.8g}")
    if missing:
        print(
            f"{missing} time step(s) have an empty {series_name} value. "
            "Those timestamps remain in the grid and are not closed up."
        )
    return origin, step, grid


def delay_embed(grid, dimension, delay):
    """Build delay vectors separated by `delay` time steps.

    A window is kept only when every timestamp it needs has a numeric value.
    A gap is not replaced by the next available observation.
    """
    window = (dimension - 1) * delay
    if len(grid) <= window:
        raise ValueError(
            "The time span is shorter than one embedding window. "
            "Reduce EMBEDDING_DIMENSION or EMBEDDING_DELAY."
        )
    length = len(grid) - window
    columns = [grid[lag * delay : lag * delay + length] for lag in range(dimension)]
    embedded = np.column_stack(columns)
    complete = np.isfinite(embedded).all(axis=1)
    if not complete.any():
        raise ValueError(
            "No delay window has a value at every required timestamp. "
            "Reduce EMBEDDING_DELAY or fill the missing observations."
        )
    skipped = int((~complete).sum())
    if skipped:
        print(
            f"Skipped {skipped} delay window(s) that land on an empty timestamp."
        )
    return embedded[complete]


def embedded_column_names(series_name, dimension, delay):
    """Label each coordinate by its lag in samples."""
    names = []
    for lag_index in range(dimension):
        lag = lag_index * delay
        if lag == 0:
            names.append(f"{series_name}[t]")
        else:
            names.append(f"{series_name}[t+{lag}]")
    return names


def average_displacement(series, dimension, delay):
    """Return S_m(delay), the mean distance from the identity line.

    For an m-dimensional delay vector the 1994 paper uses
    S_m(tau) = mean_i sqrt(sum_{j=1}^{m-1} (x_{i+j*tau} - x_i)^2).
    When m is 2 this is the mean absolute difference |x_{i+tau} - x_i|.
    """
    if delay < 1:
        return 0.0
    window = (dimension - 1) * delay
    length = len(series) - window
    if length < 1:
        return np.nan
    base = series[:length]
    squared = np.zeros(length, dtype=float)
    valid = np.isfinite(base)
    for lag_index in range(1, dimension):
        shifted = series[lag_index * delay : lag_index * delay + length]
        valid &= np.isfinite(shifted)
        squared += np.where(valid, shifted - base, 0.0) ** 2
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.sqrt(squared[valid])))


def select_average_displacement(series, dimension, slope_fraction, max_delay):
    """Choose tau where the slope of S_m first falls below slope_fraction of S'(0+).

    S'(0+) is the first difference, S_m(1) - S_m(0). The slope at lag tau is
    S_m(tau) - S_m(tau-1). The selected lag estimates an upper bound on a
    useful delay, not the lag that maximises expansion.
    """
    if dimension < 2:
        raise ValueError("Average-displacement selection needs EMBEDDING_DIMENSION >= 2.")
    feasible = (len(series) - 1) // (dimension - 1)
    limit = feasible if max_delay is None else min(feasible, max_delay)
    if limit < 2:
        raise ValueError(
            "The series is too short to compare an initial slope with a later delay. "
            "Use a longer series or a smaller EMBEDDING_DIMENSION."
        )

    displacements = np.zeros(limit + 1, dtype=float)
    usable = limit
    for delay in range(1, limit + 1):
        value = average_displacement(series, dimension, delay)
        if not np.isfinite(value):
            usable = delay - 1
            break
        displacements[delay] = value
    if usable < 2:
        raise ValueError(
            "Not enough complete delay windows to estimate a slope. "
            "Empty timestamps remove every window at a short delay."
        )
    limit = usable
    displacements = displacements[: limit + 1]
    slopes = np.diff(displacements)
    initial_slope = float(slopes[0])
    threshold = slope_fraction * initial_slope
    chosen = None
    for delay in range(2, limit + 1):
        if slopes[delay - 1] < threshold:
            chosen = delay
            break
    return {
        "delay": chosen if chosen is not None else limit,
        "criterion_met": chosen is not None,
        "displacements": displacements,
        "slopes": slopes,
        "initial_slope": initial_slope,
        "threshold": threshold,
    }


def select_embedding_delay(series, dimension):
    """Dispatch to the delay rule named by DELAY_METHOD."""
    if DELAY_METHOD == "average_displacement":
        return select_average_displacement(
            series,
            dimension,
            SLOPE_FRACTION,
            MAX_CANDIDATE_DELAY,
        )
    raise ValueError(
        f"Unknown delay method {DELAY_METHOD!r}. "
        "Implemented methods: 'average_displacement'."
    )


def print_delay_selection(selection):
    """Print the chosen lag and the slope test that produced it."""
    delay = selection["delay"]
    print("Delay selection")
    print(f"Method: {DELAY_METHOD}")
    print("Reference: Rosenstein, Collins and De Luca, Physica D 73 (1994) 82-98")
    print(f"Slope fraction: {SLOPE_FRACTION:g}")
    print(f"Initial slope: {selection['initial_slope']:.8g}")
    print(f"Slope threshold: {selection['threshold']:.8g}")
    print(f"Chosen EMBEDDING_DELAY: {delay}")
    print(f"Average displacement S({delay}): {selection['displacements'][delay]:.8g}")
    print(f"Slope at {delay}: {selection['slopes'][delay - 1]:.8g}")
    if selection["criterion_met"]:
        print("The slope first fell below the threshold at this delay.")
    else:
        print(
            "The slope stayed at or above the threshold over the whole search. "
            f"The largest tested delay, {delay}, is reported instead."
        )


def interval_array(intervals):
    """Return persistence intervals as an array of shape (n, 2)."""
    values = np.asarray(intervals, dtype=float)
    if values.size == 0:
        return np.empty((0, 2))
    return values.reshape(-1, 2)


def maximum_component_count(n_total, n_empty, delay):
    """Return the largest beta_0 allowed by the sample-size bound.

    The bound is COMPONENT_FRACTION * (time steps - empty steps - delay).
    """
    effective = n_total - n_empty - delay
    if effective <= 0:
        raise ValueError(
            "The effective sample size must be positive. "
            "It is the number of time steps, minus empty steps, minus EMBEDDING_DELAY."
        )
    return COMPONENT_FRACTION * effective


def components_alive(h0_intervals, radius):
    """Count H_0 classes whose death is strictly after radius."""
    if len(h0_intervals) == 0:
        return 0
    return int(np.sum(h0_intervals[:, 1] > radius))


def select_connected_loop(
    h0_intervals, h1_intervals, filtration_max, min_lifetime, maximum_components
):
    """Return the smallest radius whose beta_0 is within the sample-size bound.

    The bound replaces a requirement of one component. From that radius onward,
    a loop that is already alive, or born later, supplies the selected threshold.
    """
    h0_intervals = interval_array(h0_intervals)
    h1_intervals = interval_array(h1_intervals)
    allowed = int(np.floor(maximum_components + 1e-9))
    n_bars = len(h0_intervals)
    infinite_h0 = np.isinf(h0_intervals[:, 1]) if n_bars else np.empty(0, dtype=bool)
    components_at_cap = int(np.count_nonzero(infinite_h0)) if n_bars else 0
    if allowed < 1 or components_at_cap > allowed:
        return {
            "feasible": False,
            "reason": "not_connected",
            "components_at_cap": components_at_cap,
            "allowed_components": allowed,
            "maximum_components": maximum_components,
        }

    finite_deaths = np.sort(h0_intervals[~infinite_h0, 1]) if n_bars else np.empty(0)
    merges_needed = n_bars - allowed
    if merges_needed <= 0:
        connectivity_radius = 0.0
    else:
        connectivity_radius = float(finite_deaths[merges_needed - 1])
    if len(h1_intervals) == 0:
        return {
            "feasible": False,
            "reason": "no_loop",
            "connectivity_radius": connectivity_radius,
            "allowed_components": allowed,
            "maximum_components": maximum_components,
            "beta_0": components_alive(h0_intervals, connectivity_radius),
        }

    earliest = np.maximum(h1_intervals[:, 0], connectivity_radius)
    if min_lifetime > 0:
        valid = earliest <= h1_intervals[:, 1] - min_lifetime
    else:
        # Intervals are half-open, so a class born at b is alive on [b, d).
        valid = earliest < h1_intervals[:, 1]
    valid &= earliest <= filtration_max
    if not np.any(valid):
        return {
            "feasible": False,
            "reason": "no_surviving_loop",
            "connectivity_radius": connectivity_radius,
            "allowed_components": allowed,
            "maximum_components": maximum_components,
            "beta_0": components_alive(h0_intervals, connectivity_radius),
        }

    radius = float(np.min(earliest[valid]))
    alive = (h1_intervals[:, 0] <= radius) & (radius < h1_intervals[:, 1])
    return {
        "feasible": True,
        "connectivity_radius": connectivity_radius,
        "threshold": radius,
        "beta_0": components_alive(h0_intervals, radius),
        "beta_1": int(np.count_nonzero(alive)),
        "allowed_components": allowed,
        "maximum_components": maximum_components,
    }


def select_threshold(h0_intervals, h1_intervals, maximum_components):
    """Dispatch to the threshold rule named by THRESHOLD_METHOD."""
    if THRESHOLD_METHOD == "connected_loop":
        return select_connected_loop(
            h0_intervals,
            h1_intervals,
            FILTRATION_MAX_EDGE_LENGTH,
            MIN_LOOP_LIFETIME,
            maximum_components,
        )
    raise ValueError(
        f"Unknown threshold method {THRESHOLD_METHOD!r}. "
        "Implemented methods: 'connected_loop'."
    )


def print_threshold_selection(selection):
    """Print the connectivity radius and the selected connection threshold."""
    print("Threshold selection")
    print(f"Method: {THRESHOLD_METHOD}")
    print(f"Filtration upper bound: {FILTRATION_MAX_EDGE_LENGTH:g}")
    print(f"Minimum loop lifetime: {MIN_LOOP_LIFETIME:g}")
    print(f"Allowed beta_0: <= {selection['maximum_components']:.8g}")
    if not selection["feasible"]:
        reason = selection["reason"]
        if reason == "not_connected":
            print(
                "beta_0 never falls to the allowed bound inside the filtration. "
                f"beta_0 at the upper bound is {selection['components_at_cap']}."
            )
        elif reason == "no_loop":
            print(
                "beta_0 reaches the allowed bound at "
                f"{selection['connectivity_radius']:.8g}, but the filtration contains no H_1 interval."
            )
        else:
            print(
                "No H_1 class is still alive for the requested lifetime once beta_0 "
                "is within the allowed bound. That radius is "
                f"{selection['connectivity_radius']:.8g}, "
                f"where beta_0 is {selection['beta_0']}."
            )
        print("Increase FILTRATION_MAX_EDGE_LENGTH or reduce MIN_LOOP_LIFETIME.")
        return
    print(f"Radius where beta_0 reaches the bound: {selection['connectivity_radius']:.8g}")
    print(f"Selected threshold: {selection['threshold']:.8g}")
    print(f"beta_0 at the selected threshold: {selection['beta_0']}")
    print(f"beta_1 at the selected threshold: {selection['beta_1']}")


def print_embedding(dataset_path, series_name, embedded, columns, time_step):
    """Describe the delay embedding whose rows are the vertices."""
    print("Time-delay embedding")
    print(f"Dataset: {dataset_path}")
    print(f"Series: {series_name}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(
        f"Delay: {EMBEDDING_DELAY} time step(s) "
        f"= {EMBEDDING_DELAY * time_step:.8g} time units"
    )
    print(f"Coordinates: {', '.join(columns)}")
    print(f"Embedded points: {len(embedded)}")
    print("Vertex i is a complete window on the original time grid.")


def pairwise_distance_matrix(points, distance_name):
    """Return the square pairwise distance matrix selected by distance_name."""
    if distance_name == "euclidean":
        # Ordinary Euclidean distance between rows.
        return cdist(points, points, metric="euclidean")
    raise ValueError(
        f"Unknown distance {distance_name!r}. Implemented distances: 'euclidean'."
    )


def build_rips_complex(distances, max_edge_length, max_simplex_dimension):
    """Expand the Vietoris-Rips complex up to the requested simplex dimension."""
    rips = gudhi.RipsComplex(
        distance_matrix=np.asarray(distances, dtype=float),
        max_edge_length=float(max_edge_length),
    )
    return rips.create_simplex_tree(max_dimension=max_simplex_dimension)


def format_simplex(simplex):
    """Format a simplex as a bracketed list of vertex indices."""
    vertices = ", ".join(str(vertex) for vertex in sorted(simplex))
    return f"[{vertices}]"


def print_complex(simplex_tree, dataset_path, columns):
    """Print every simplex together with the scale at which it appears."""
    print("Vietoris-Rips complex")
    print(f"Dataset: {dataset_path}")
    print(f"Columns: {', '.join(columns)}")
    print(f"Distance: {DISTANCE}")
    print(f"Selected threshold: {SELECTED_EDGE_LENGTH:.8g}")
    print(f"Max simplex dimension: {MAX_SIMPLEX_DIMENSION}")

    grouped = {}
    for simplex, filtration in simplex_tree.get_filtration():
        dimension = len(simplex) - 1
        grouped.setdefault(dimension, []).append((simplex, filtration))

    for dimension in range(MAX_SIMPLEX_DIMENSION + 1):
        simplices = grouped.get(dimension, [])
        print(f"Dimension {dimension}: {len(simplices)} simplices")
        # A dense dimension is already visible in the figure, so do not list every simplex.
        if dimension != 0 and len(simplices) > 40:
            continue
        # Every embedded window is a vertex born at scale 0, so list that set once.
        if dimension == 0 and simplices:
            vertices = sorted(simplex[0] for simplex, _ in simplices)
            all_born_at_zero = all(filtration == 0 for _, filtration in simplices)
            if all_born_at_zero and vertices == list(range(len(vertices))):
                print(f"  [0], [1], ..., [{vertices[-1]}]  filtration=0")
                continue
        for simplex, filtration in simplices:
            print(f"  {format_simplex(simplex)}  filtration={filtration:.8g}")


def print_homology(simplex_tree, max_homology_dimension, max_simplex_dimension):
    """Compute and print persistence intervals for H_0 through the requested degree.

    Intervals are half-open [birth, death). An infinite death means the class
    is still alive at SELECTED_EDGE_LENGTH, so it contributes to the homology of
    the printed complex.
    """
    # The top-dimensional homology is omitted unless requested explicitly,
    # because no simplex of dimension k + 1 is available to fill an H_k class.
    include_top_dimension = max_homology_dimension >= max_simplex_dimension
    simplex_tree.persistence(
        homology_coeff_field=HOMOLOGY_COEFFICIENT_FIELD,
        min_persistence=0,
        persistence_dim_max=include_top_dimension,
    )

    print("Homology")
    print(f"Coefficient field: Z/{HOMOLOGY_COEFFICIENT_FIELD}Z")
    if include_top_dimension:
        print(
            f"H_{max_homology_dimension} uses the highest simplices in this complex. "
            "Nothing of higher dimension is present to fill those classes."
        )

    for dimension in range(max_homology_dimension + 1):
        intervals = np.asarray(
            simplex_tree.persistence_intervals_in_dimension(dimension),
            dtype=float,
        )
        if intervals.size == 0:
            intervals = np.empty((0, 2))
        order = np.lexsort((intervals[:, 1], intervals[:, 0]))
        intervals = intervals[order]
        betti_number = int(np.isinf(intervals[:, 1]).sum()) if len(intervals) else 0
        print(
            f"H_{dimension}: {len(intervals)} persistence intervals; "
            f"Betti number at the threshold = {betti_number}"
        )
        if len(intervals) == 0:
            print("  (none)")
            continue
        start = 0
        while start < len(intervals):
            birth, death = intervals[start]
            multiplicity = 1
            while start + multiplicity < len(intervals) and np.array_equal(
                intervals[start + multiplicity], intervals[start]
            ):
                multiplicity += 1
            death_text = "inf" if np.isinf(death) else f"{death:.8g}"
            suffix = f"  multiplicity={multiplicity}" if multiplicity > 1 else ""
            print(f"  [{birth:.8g}, {death_text}){suffix}")
            start += multiplicity


def resolve_figure_dimension(ambient_dimension):
    """Choose 2 or 3 display axes from the figure control and the point cloud."""
    if FIGURE_DIMENSION == "auto":
        return 2 if ambient_dimension == 2 else 3
    return FIGURE_DIMENSION


def resolve_figure_path(dataset_path):
    """Return the file that receives the complex figure."""
    if str(FIGURE_PATH) == "auto":
        return dataset_path.with_name(f"{dataset_path.stem}_auto_complex.png")
    path = Path(FIGURE_PATH)
    if not path.is_absolute():
        path = dataset_path.parent / path
    return path


def pca_project(points, n_components):
    """Project rows of points onto their leading principal axes."""
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    available = min(n_components, axes.shape[0])
    projected = centered @ axes[:available].T
    if available == n_components:
        return projected
    padded = np.zeros((len(points), n_components), dtype=float)
    padded[:, :available] = projected
    return padded


def display_coordinates(points, columns, figure_dimension):
    """Return coordinates with exactly figure_dimension columns, plus axis labels.

    A lower-dimensional cloud is padded with constant 0. A higher-dimensional
    cloud is reduced with the selected projection.
    """
    ambient_dimension = points.shape[1]
    if ambient_dimension == figure_dimension:
        return np.asarray(points, dtype=float), list(columns), "original"
    if ambient_dimension < figure_dimension:
        displayed = np.zeros((len(points), figure_dimension), dtype=float)
        displayed[:, :ambient_dimension] = points
        labels = list(columns) + ["constant 0"] * (figure_dimension - ambient_dimension)
        return displayed, labels, "padded"
    if PROJECTION == "pca":
        labels = [f"PC{index}" for index in range(1, figure_dimension + 1)]
        return pca_project(points, figure_dimension), labels, "pca"
    raise ValueError(
        f"Unknown projection {PROJECTION!r}. Implemented projections: 'pca'."
    )


def skeletons_for_display(simplex_tree):
    """Split the complex into edges, free triangles, and tetrahedron faces.

    A triangular face of a tetrahedron is drawn once, in the tetrahedron style.
    Simplices of dimension greater than 3 remain visible through these faces.
    """
    edges = []
    triangles = []
    tetrahedra = []
    highest_dimension = 0
    for simplex, _filtration in simplex_tree.get_simplices():
        vertices = tuple(sorted(int(vertex) for vertex in simplex))
        dimension = len(vertices) - 1
        highest_dimension = max(highest_dimension, dimension)
        if dimension == 1:
            edges.append(vertices)
        elif dimension == 2:
            triangles.append(vertices)
        elif dimension == 3:
            tetrahedra.append(vertices)

    tetrahedron_faces = set()
    for tetrahedron in tetrahedra:
        for face in TETRAHEDRON_FACES:
            tetrahedron_faces.add(tuple(sorted(tetrahedron[index] for index in face)))

    free_triangles = [triangle for triangle in triangles if triangle not in tetrahedron_faces]
    for face in tetrahedron_faces:
        if face not in triangles:
            triangles.append(face)
    bound_triangles = [triangle for triangle in triangles if triangle in tetrahedron_faces]
    return edges, free_triangles, bound_triangles, highest_dimension


def polygons_from_indices(coords, simplices):
    """Return each simplex as an array of vertex coordinates."""
    return [coords[list(simplex)] for simplex in simplices]


def add_edges(ax, coords, edges, zorder=2):
    """Draw 1-simplices on either a 2D or a 3D axes."""
    if not edges:
        return
    segments = polygons_from_indices(coords, edges)
    if coords.shape[1] == 2:
        ax.add_collection(
            LineCollection(segments, colors=EDGE_COLOR, linewidths=1.8, zorder=zorder)
        )
    else:
        ax.add_collection3d(Line3DCollection(segments, colors=EDGE_COLOR, linewidths=1.4))


def add_faces(ax, coords, faces, color, alpha):
    """Draw filled triangular faces on either a 2D or a 3D axes."""
    if not faces:
        return
    polygons = polygons_from_indices(coords, faces)
    if coords.shape[1] == 2:
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors=color,
                edgecolors="none",
                alpha=alpha,
                zorder=1,
            )
        )
    else:
        ax.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolors=color,
                edgecolors="none",
                alpha=alpha,
            )
        )


def add_legend(figure, has_edges, has_triangles, has_tetrahedron_faces):
    """Add one legend entry for each simplex dimension present in the figure."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=VERTEX_COLOR,
            markersize=4,
            label="0-simplices",
        )
    ]
    if has_edges:
        handles.append(Line2D([0], [0], color=EDGE_COLOR, linewidth=1.2, label="1-simplices"))
    if has_triangles:
        handles.append(
            Patch(facecolor=TRIANGLE_COLOR, edgecolor="none", alpha=0.4, label="2-simplices")
        )
    if has_tetrahedron_faces:
        handles.append(
            Patch(
                facecolor=TETRAHEDRON_COLOR,
                edgecolor="none",
                alpha=0.2,
                label="3-simplex faces",
            )
        )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )


def set_equal_scale(ax, coords):
    """Use the same scale on every displayed axis, including flat point clouds."""
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = maximum - minimum
    positive = span[span > 0]
    fallback = float(positive.max() * 0.25) if len(positive) else 1.0
    span = np.where(span > 0, span, fallback)
    center = (minimum + maximum) / 2
    half = span * 0.56
    ax.set_xlim(center[0] - half[0], center[0] + half[0])
    ax.set_ylim(center[1] - half[1], center[1] + half[1])
    if coords.shape[1] == 3:
        ax.set_zlim(center[2] - half[2], center[2] + half[2])
        ax.set_box_aspect(tuple(span))
    else:
        ax.set_aspect("equal", adjustable="box")


def render_complex(simplex_tree, points, columns, figure_dimension, figure_path, title=None):
    """Draw the complex and save it. The axes are always 2D or 3D."""
    coords, labels, coordinate_mode = display_coordinates(points, columns, figure_dimension)
    edges, free_triangles, tetrahedron_faces, highest_dimension = skeletons_for_display(
        simplex_tree
    )

    if coords.shape[1] == 2:
        span = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1e-9)
        height = float(np.clip(9.0 * span[1] / span[0], 2.6, 8.0))
        figure, axes = plt.subplots(figsize=(9.0, height), layout="constrained")
    else:
        figure = plt.figure(figsize=(8.0, 7.2), layout="constrained")
        axes = figure.add_subplot(111, projection="3d")
        axes.view_init(elev=24, azim=-60)

    add_faces(axes, coords, tetrahedron_faces, TETRAHEDRON_COLOR, 0.18)
    add_faces(axes, coords, free_triangles, TRIANGLE_COLOR, 0.4)
    if coords.shape[1] == 2:
        axes.scatter(
            coords[:, 0],
            coords[:, 1],
            c=VERTEX_COLOR,
            s=10,
            zorder=3,
            linewidths=0,
        )
        # Draw edges above the markers so a short edge remains visible.
        add_edges(axes, coords, edges, zorder=4)
    else:
        add_edges(axes, coords, edges)
        axes.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=VERTEX_COLOR,
            s=12,
            depthshade=False,
            linewidths=0,
        )

    set_equal_scale(axes, coords)
    axes.set_xlabel(labels[0], labelpad=8)
    axes.set_ylabel(labels[1], labelpad=8 if coords.shape[1] == 2 else 26)
    if coords.shape[1] == 3:
        axes.set_zlabel(labels[2], labelpad=12)
        axes.tick_params(axis="y", pad=2)
    axes.set_title(
        title or f"Vietoris-Rips complex, {DISTANCE}, threshold {SELECTED_EDGE_LENGTH:.8g}"
    )
    axes.grid(True, alpha=0.3)
    add_legend(
        figure,
        has_edges=bool(edges),
        has_triangles=bool(free_triangles),
        has_tetrahedron_faces=bool(tetrahedron_faces),
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("Figure")
    print(f"Path: {figure_path}")
    print(f"Dimension: {figure_dimension}")
    if coordinate_mode == "pca":
        print(f"Coordinates: {PROJECTION} projection onto {figure_dimension} axes")
    elif coordinate_mode == "padded":
        print("Coordinates: original axes, with constant 0 appended for the 3D view")
    else:
        print("Coordinates: displayed without projection")
    if highest_dimension > 3:
        print(
            f"The complex contains simplices up to dimension {highest_dimension}. "
            "The figure shows their vertices, edges, and triangular faces."
        )


def save_complex_figure(simplex_tree, points, columns, dataset_path, title=None):
    """Resolve the figure controls and draw the complex."""
    figure_dimension = resolve_figure_dimension(points.shape[1])
    figure_path = resolve_figure_path(dataset_path)
    render_complex(
        simplex_tree,
        points,
        columns,
        figure_dimension,
        figure_path,
        title=title,
    )


def main():
    global EMBEDDING_DELAY, SELECTED_EDGE_LENGTH
    validate_controls()
    times, values, series_name, time_name = load_timed_series(DATASET_PATH, SERIES_COLUMN)
    _origin, time_step, grid = align_to_time_grid(times, values, series_name, time_name)
    delay_selection = select_embedding_delay(grid, EMBEDDING_DIMENSION)
    EMBEDDING_DELAY = delay_selection["delay"]
    print_delay_selection(delay_selection)
    print()
    embedded = delay_embed(grid, EMBEDDING_DIMENSION, EMBEDDING_DELAY)
    columns = embedded_column_names(series_name, EMBEDDING_DIMENSION, EMBEDDING_DELAY)
    distances = pairwise_distance_matrix(embedded, DISTANCE)
    # One filtration up to the cap. Its persistence intervals determine the threshold.
    filtration = build_rips_complex(
        distances,
        FILTRATION_MAX_EDGE_LENGTH,
        MAX_SIMPLEX_DIMENSION,
    )
    filtration.persistence(
        homology_coeff_field=HOMOLOGY_COEFFICIENT_FIELD,
        min_persistence=0,
        persistence_dim_max=MAX_HOMOLOGY_DIMENSION >= MAX_SIMPLEX_DIMENSION,
    )
    n_total = len(grid)
    n_empty = int(np.isnan(grid).sum())
    component_limit = maximum_component_count(n_total, n_empty, EMBEDDING_DELAY)
    threshold_selection = select_threshold(
        filtration.persistence_intervals_in_dimension(0),
        filtration.persistence_intervals_in_dimension(1),
        component_limit,
    )
    print_threshold_selection(threshold_selection)
    print()
    if not threshold_selection["feasible"]:
        raise RuntimeError(
            "No Rips threshold inside FILTRATION_MAX_EDGE_LENGTH satisfies the selected rule."
        )
    SELECTED_EDGE_LENGTH = threshold_selection["threshold"]
    simplex_tree = build_rips_complex(
        distances,
        SELECTED_EDGE_LENGTH,
        MAX_SIMPLEX_DIMENSION,
    )
    print_embedding(DATASET_PATH, series_name, embedded, columns, time_step)
    print()
    print_complex(simplex_tree, DATASET_PATH, columns)
    print()
    save_complex_figure(
        simplex_tree,
        embedded,
        columns,
        DATASET_PATH,
        title=(
            f"Delay embedding, EMBEDDING_DELAY={EMBEDDING_DELAY}, "
            f"{DISTANCE}, threshold {SELECTED_EDGE_LENGTH:.6g}"
        ),
    )
    print()
    print_homology(simplex_tree, MAX_HOMOLOGY_DIMENSION, MAX_SIMPLEX_DIMENSION)


if __name__ == "__main__":
    main()
