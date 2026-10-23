# -*- coding: utf-8 -*-
"""Delay-embed a scalar series and analyse its Vietoris-Rips complex.

A binary embedding represents each time by the current sample and one lagged
sample. The lag and the embedding dimension are controls, as are the distance,
the Rips threshold, and the simplex and homology dimensions.
"""

from pathlib import Path

import numpy as np

import Main

# =============================================================================
# Controls
# These are the values most likely to change between experiments.
# =============================================================================

# Source table. Row order is the time order. A non-numeric first row is a header.
DATASET_PATH = Path(__file__).with_name("Exp1Sin.csv")

# Scalar series to embed. None selects the file when it has exactly one column.
SERIES_COLUMN = "sin(t)"

# Binary delay embedding uses dimension 2: each point is
# (x[t], x[t + delay], ..., x[t + (dimension - 1) * delay]).
EMBEDDING_DIMENSION = 2

# Lag between consecutive coordinates, measured in samples.
# 1 pairs each sample with the immediately following sample.
EMBEDDING_DELAY = 1

# Pairwise distance used to build the complex.
# "euclidean" is ordinary Euclidean distance.
# Add a branch in Main.pairwise_distance_matrix before selecting another name.
DISTANCE = "euclidean"

# Longest edge allowed in the Vietoris-Rips complex.
MAX_EDGE_LENGTH = 0.1

# Highest simplex dimension to materialise.
# 0 = vertices, 1 = edges, 2 = triangles.
MAX_SIMPLEX_DIMENSION = 2

# Homology is reported for H_0, ..., H_MAX_HOMOLOGY_DIMENSION.
# Computing H_k needs simplices through dimension k + 1, so this value
# should be at most MAX_SIMPLEX_DIMENSION.
MAX_HOMOLOGY_DIMENSION = 1

# Prime field for homology. Betti numbers over Z/2Z are the usual choice.
HOMOLOGY_COEFFICIENT_FIELD = 2

# Figure dimension: 2, 3, or "auto".
# "auto" uses a 2D figure for a 2D embedding and a 3D figure otherwise.
FIGURE_DIMENSION = "auto"

# "auto" saves "<dataset stem>_delay_complex.png" next to the dataset.
FIGURE_PATH = "auto"

# Projection used when an embedding has more axes than the figure.
PROJECTION = "pca"


def validate_embedding_controls():
    """Reject embedding controls that cannot build a delay window."""
    if SERIES_COLUMN is not None and not isinstance(SERIES_COLUMN, str):
        raise ValueError("SERIES_COLUMN must be a column name or None.")
    if type(EMBEDDING_DIMENSION) is not int or EMBEDDING_DIMENSION < 1:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer.")
    if type(EMBEDDING_DELAY) is not int or EMBEDDING_DELAY < 1:
        raise ValueError("EMBEDDING_DELAY must be a positive integer.")


def publish_pipeline_controls():
    """Copy the shared complex controls into Main, which draws the figure."""
    Main.DISTANCE = DISTANCE
    Main.MAX_EDGE_LENGTH = MAX_EDGE_LENGTH
    Main.MAX_SIMPLEX_DIMENSION = MAX_SIMPLEX_DIMENSION
    Main.MAX_HOMOLOGY_DIMENSION = MAX_HOMOLOGY_DIMENSION
    Main.HOMOLOGY_COEFFICIENT_FIELD = HOMOLOGY_COEFFICIENT_FIELD
    Main.FIGURE_DIMENSION = FIGURE_DIMENSION
    Main.FIGURE_PATH = FIGURE_PATH
    Main.PROJECTION = PROJECTION
    Main.validate_controls()


def load_series(path, column_name):
    """Return the chosen scalar column and its name. Row order is preserved."""
    cloud, columns = Main.load_point_cloud(path)
    if column_name is None:
        if len(columns) != 1:
            names = ", ".join(columns)
            raise ValueError(f"Set SERIES_COLUMN to one of: {names}.")
        return cloud[:, 0], columns[0]
    if column_name not in columns:
        names = ", ".join(columns)
        raise ValueError(f"Column {column_name!r} is not in {path.name}. Columns: {names}.")
    return cloud[:, columns.index(column_name)], column_name


def delay_embed(series, dimension, delay):
    """Build delay vectors (x[t], x[t + delay], ..., x[t + (dimension - 1) * delay])."""
    window = (dimension - 1) * delay
    if len(series) <= window:
        raise ValueError(
            "The series is shorter than one embedding window. "
            "Reduce EMBEDDING_DIMENSION or EMBEDDING_DELAY."
        )
    length = len(series) - window
    return np.column_stack(
        [series[lag * delay : lag * delay + length] for lag in range(dimension)]
    )


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


def resolve_figure_path(dataset_path):
    """Return the image path for the embedded complex."""
    if str(FIGURE_PATH) == "auto":
        return dataset_path.with_name(f"{dataset_path.stem}_delay_complex.png")
    path = Path(FIGURE_PATH)
    if not path.is_absolute():
        path = dataset_path.parent / path
    return path


def print_embedding(dataset_path, series_name, embedded, columns):
    """Describe the delay embedding whose rows are the vertices."""
    print("Time-delay embedding")
    print(f"Dataset: {dataset_path}")
    print(f"Series: {series_name}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"Delay: {EMBEDDING_DELAY} sample(s)")
    print(f"Coordinates: {', '.join(columns)}")
    print(f"Embedded points: {len(embedded)}")
    print("Vertex i is the window starting at series index i.")


def print_complex_summary(simplex_tree):
    """Print how many simplices were materialised in each dimension."""
    counts = {}
    for simplex, _filtration in simplex_tree.get_simplices():
        dimension = len(simplex) - 1
        counts[dimension] = counts.get(dimension, 0) + 1
    print("Vietoris-Rips complex")
    print(f"Distance: {DISTANCE}")
    print(f"Max edge length: {MAX_EDGE_LENGTH}")
    print(f"Max simplex dimension: {MAX_SIMPLEX_DIMENSION}")
    for dimension in range(MAX_SIMPLEX_DIMENSION + 1):
        print(f"Dimension {dimension}: {counts.get(dimension, 0)} simplices")


def main():
    validate_embedding_controls()
    publish_pipeline_controls()
    series, series_name = load_series(DATASET_PATH, SERIES_COLUMN)
    embedded = delay_embed(series, EMBEDDING_DIMENSION, EMBEDDING_DELAY)
    columns = embedded_column_names(series_name, EMBEDDING_DIMENSION, EMBEDDING_DELAY)

    distances = Main.pairwise_distance_matrix(embedded, DISTANCE)
    simplex_tree = Main.build_rips_complex(
        distances,
        MAX_EDGE_LENGTH,
        MAX_SIMPLEX_DIMENSION,
    )
    print_embedding(DATASET_PATH, series_name, embedded, columns)
    print()
    print_complex_summary(simplex_tree)
    print()
    figure_dimension = Main.resolve_figure_dimension(embedded.shape[1])
    figure_path = resolve_figure_path(DATASET_PATH)
    Main.render_complex(
        simplex_tree,
        embedded,
        columns,
        figure_dimension,
        figure_path,
        title=(
            f"Delay embedding, {DISTANCE}, threshold {MAX_EDGE_LENGTH:g}"
        ),
    )
    print()
    Main.print_homology(simplex_tree, MAX_HOMOLOGY_DIMENSION, MAX_SIMPLEX_DIMENSION)


if __name__ == "__main__":
    main()
