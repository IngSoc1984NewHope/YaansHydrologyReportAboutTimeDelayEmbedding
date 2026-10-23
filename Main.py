# -*- coding: utf-8 -*-
"""Build a Vietoris-Rips complex of a point cloud and compute its homology.

Each row of the chosen CSV is one point, and each column is a coordinate.
The complex is the filtered Vietoris-Rips complex of all simplices whose
longest edge is at most MAX_EDGE_LENGTH.
"""

import csv
from pathlib import Path

import gudhi
import numpy as np
from scipy.spatial.distance import cdist

# =============================================================================
# Controls
# These are the values most likely to change between experiments.
# =============================================================================

# Point cloud. A non-numeric first row is read as a column header.
DATASET_PATH = Path(__file__).with_name("Exp1Sin.csv")

# Pairwise distance that decides which edges are admitted.
# "euclidean" is the ordinary Euclidean distance.
# A later distance (for example a geodesic approximation) can be added as
# another branch in pairwise_distance_matrix, then selected by this name.
DISTANCE = "euclidean"

# Longest edge allowed in the Vietoris-Rips complex.
MAX_EDGE_LENGTH = 0.1

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


def validate_controls():
    """Reject control values that cannot define a Vietoris-Rips homology run."""
    if not isinstance(DISTANCE, str) or not DISTANCE:
        raise ValueError("DISTANCE must be a non-empty distance name.")
    if not np.isfinite(MAX_EDGE_LENGTH) or MAX_EDGE_LENGTH < 0:
        raise ValueError("MAX_EDGE_LENGTH must be a finite number greater than or equal to 0.")
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


def load_point_cloud(path):
    """Load a CSV point cloud. Returns coordinates of shape (n_points, dimension) and column names."""
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

    def parse_row(row):
        return [float(cell) for cell in row]

    try:
        parse_row(rows[0])
        columns = [f"x{index}" for index in range(len(rows[0]))]
        data_rows = rows
    except ValueError:
        columns = [cell.strip() or f"x{index}" for index, cell in enumerate(rows[0])]
        data_rows = rows[1:]

    if not data_rows:
        raise ValueError(f"Dataset has a header but no data rows: {path}")

    points = []
    for offset, row in enumerate(data_rows, start=1):
        if len(row) != len(columns):
            raise ValueError(f"Row {offset} of {path.name} does not have {len(columns)} columns.")
        try:
            points.append(parse_row(row))
        except ValueError as exc:
            raise ValueError(f"Non-numeric value in {path.name} at data row {offset}.") from exc

    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[0] == 0:
        raise ValueError(f"Dataset does not contain a point cloud: {path}")
    if not np.isfinite(cloud).all():
        raise ValueError(f"Dataset contains NaN or infinite coordinates: {path}")
    return cloud, columns


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
    print("Each vertex index is the corresponding data row, starting at 0.")
    print(f"Distance: {DISTANCE}")
    print(f"Max edge length: {MAX_EDGE_LENGTH}")
    print(f"Max simplex dimension: {MAX_SIMPLEX_DIMENSION}")

    grouped = {}
    for simplex, filtration in simplex_tree.get_filtration():
        dimension = len(simplex) - 1
        grouped.setdefault(dimension, []).append((simplex, filtration))

    for dimension in range(MAX_SIMPLEX_DIMENSION + 1):
        simplices = grouped.get(dimension, [])
        print(f"Dimension {dimension}: {len(simplices)} simplices")
        # Every data row is a vertex born at scale 0, so list that set once.
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
    is still alive at MAX_EDGE_LENGTH, so it contributes to the homology of
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


def main():
    validate_controls()
    points, columns = load_point_cloud(DATASET_PATH)
    distances = pairwise_distance_matrix(points, DISTANCE)
    simplex_tree = build_rips_complex(
        distances,
        MAX_EDGE_LENGTH,
        MAX_SIMPLEX_DIMENSION,
    )
    print_complex(simplex_tree, DATASET_PATH, columns)
    print()
    print_homology(simplex_tree, MAX_HOMOLOGY_DIMENSION, MAX_SIMPLEX_DIMENSION)


if __name__ == "__main__":
    main()
