# -*- coding: utf-8 -*-
"""Delay-embed a scalar series, then build and draw its Vietoris-Rips complex.

A binary embedding represents each time by the current sample and one lagged
sample. The lag, embedding dimension, distance, Rips threshold, and the
simplex and homology dimensions are controls at the top of this file.
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
DATASET_PATH = Path(__file__).with_name("Exp1Sin.csv")

# Scalar series to embed. None selects the file when it has exactly one column.
SERIES_COLUMN = "sin(t)"

# Binary delay embedding uses dimension 2: each point is
# (x[t], x[t + delay], ..., x[t + (dimension - 1) * delay]).
EMBEDDING_DIMENSION = 2

# Lag between consecutive coordinates, measured in samples.
# 1 pairs each sample with the immediately following sample.
EMBEDDING_DELAY = 1

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

# Figure dimension: 2, 3, or "auto".
# "auto" draws a 2D figure for a 2D point cloud and a 3D figure otherwise.
FIGURE_DIMENSION = "auto"

# "auto" saves "<dataset stem>_complex.png" next to the dataset.
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
    if type(EMBEDDING_DIMENSION) is not int or EMBEDDING_DIMENSION < 1:
        raise ValueError("EMBEDDING_DIMENSION must be a positive integer.")
    if type(EMBEDDING_DELAY) is not int or EMBEDDING_DELAY < 1:
        raise ValueError("EMBEDDING_DELAY must be a positive integer.")
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
    if FIGURE_DIMENSION not in ("auto", 2, 3):
        raise ValueError("FIGURE_DIMENSION must be 'auto', 2, or 3.")
    if not isinstance(FIGURE_PATH, (str, Path)) or str(FIGURE_PATH).strip() == "":
        raise ValueError("FIGURE_PATH must be 'auto' or a file path.")
    if PROJECTION != "pca":
        raise ValueError(
            "Unknown projection. Implemented projections: 'pca'. "
            "Add another branch in display_coordinates before selecting it."
        )


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


def load_series(path, column_name):
    """Return the chosen scalar column and its name. Row order is preserved."""
    cloud, columns = load_point_cloud(path)
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
    print(f"Max edge length: {MAX_EDGE_LENGTH}")
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


def resolve_figure_dimension(ambient_dimension):
    """Choose 2 or 3 display axes from the figure control and the point cloud."""
    if FIGURE_DIMENSION == "auto":
        return 2 if ambient_dimension == 2 else 3
    return FIGURE_DIMENSION


def resolve_figure_path(dataset_path):
    """Return the file that receives the complex figure."""
    if str(FIGURE_PATH) == "auto":
        return dataset_path.with_name(f"{dataset_path.stem}_complex.png")
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
    axes.set_title(title or f"Vietoris-Rips complex, {DISTANCE}, threshold {MAX_EDGE_LENGTH:g}")
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
    validate_controls()
    series, series_name = load_series(DATASET_PATH, SERIES_COLUMN)
    embedded = delay_embed(series, EMBEDDING_DIMENSION, EMBEDDING_DELAY)
    columns = embedded_column_names(series_name, EMBEDDING_DIMENSION, EMBEDDING_DELAY)
    distances = pairwise_distance_matrix(embedded, DISTANCE)
    simplex_tree = build_rips_complex(
        distances,
        MAX_EDGE_LENGTH,
        MAX_SIMPLEX_DIMENSION,
    )
    print_embedding(DATASET_PATH, series_name, embedded, columns)
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
            f"{DISTANCE}, threshold {MAX_EDGE_LENGTH:g}"
        ),
    )
    print()
    print_homology(simplex_tree, MAX_HOMOLOGY_DIMENSION, MAX_SIMPLEX_DIMENSION)


if __name__ == "__main__":
    main()
