# -*- coding: utf-8 -*-
"""Estimate one delay and one Rips threshold, then write a MATLAB window viewer.

The delay and the connection threshold are computed on the whole series by the
same rules as AutoParamter.py. Each MATLAB frame is the complex of one sliding
window of raw observations. The window length is WINDOW_DAYS plus the extra
time needed for the delay coordinates.
"""

from pathlib import Path

import numpy as np
from scipy.io import savemat

import AutoParamter as pipeline

# =============================================================================
# Controls
# These are the values most likely to change between experiments.
# =============================================================================

# Source table. Row order is the time order. A non-numeric first row is a header.
DATASET_PATH = Path(__file__).with_name("MaeChaem_Q_H.csv")

# Scalar series to embed. None selects the file when it has exactly one column.
SERIES_COLUMN = "Q"

# Timestamp column. None uses the other column when the file has two columns.
# A file with only the series is treated as evenly spaced with step 1.
TIME_COLUMN = "Date"

# Binary delay embedding uses dimension 2: each point is
# (x[t], x[t + delay], ..., x[t + (dimension - 1) * delay]).
EMBEDDING_DIMENSION = 2

# EMBEDDING_DELAY is chosen from the whole series by this rule.
# "average_displacement" is the reconstruction-expansion rule of
# Rosenstein, Collins and De Luca, Physica D 73 (1994) 82-98.
DELAY_METHOD = "average_displacement"

# The chosen lag is the first one whose average-displacement slope is below
# this fraction of the initial slope. The 1994 paper uses 0.4.
SLOPE_FRACTION = 0.4

# Largest lag, in samples, included in the search.
# None searches every lag that still leaves one embedding window.
MAX_CANDIDATE_DELAY = None

# Pairwise distance that decides which edges are admitted.
DISTANCE = "euclidean"

# Rule for the Rips threshold used for every sliding window.
# "connected_loop" is the smallest radius at which
# beta_0 <= COMPONENT_FRACTION * (time steps - empty steps - EMBEDDING_DELAY)
# and the complex still has a one-dimensional loop.
THRESHOLD_METHOD = "connected_loop"

# Fraction of the effective sample allowed to remain as separate components.
COMPONENT_FRACTION = 0.1

# Upper bound of the filtration used to read persistence on the whole series.
# This is not the selected connection threshold.
FILTRATION_MAX_EDGE_LENGTH = 9

# A selected loop must still be alive for at least this much additional radius.
MIN_LOOP_LIFETIME = 0.0

# Highest simplex dimension to materialise.
MAX_SIMPLEX_DIMENSION = 2

# Homology is reported for H_0, ..., H_MAX_HOMOLOGY_DIMENSION.
MAX_HOMOLOGY_DIMENSION = 1

# Prime used as the coefficient field Z/pZ.
HOMOLOGY_COEFFICIENT_FIELD = 2

# Number of raw time steps represented by the embedded points of one frame.
# The raw window also includes the delay coordinates, so its length is
# WINDOW_DAYS + (EMBEDDING_DIMENSION - 1) * delay.
WINDOW_DAYS = 30

# How far the next window start moves, in time steps.
WINDOW_STEP_DAYS = 30

# "auto" writes "<dataset stem>_windows.m" and a matching .mat file.
MATLAB_PATH = "auto"


def validate_controls():
    """Reject controls that cannot define the global estimate or the windows."""
    if type(WINDOW_DAYS) is not int or WINDOW_DAYS < 1:
        raise ValueError("WINDOW_DAYS must be a positive integer.")
    if type(WINDOW_STEP_DAYS) is not int or WINDOW_STEP_DAYS < 1:
        raise ValueError("WINDOW_STEP_DAYS must be a positive integer.")
    if not isinstance(MATLAB_PATH, (str, Path)) or str(MATLAB_PATH).strip() == "":
        raise ValueError("MATLAB_PATH must be 'auto' or a file path.")


def apply_pipeline_controls():
    """Copy the shared controls into the pipeline that estimates delay and threshold."""
    pipeline.DATASET_PATH = DATASET_PATH
    pipeline.SERIES_COLUMN = SERIES_COLUMN
    pipeline.TIME_COLUMN = TIME_COLUMN
    pipeline.EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
    pipeline.DELAY_METHOD = DELAY_METHOD
    pipeline.SLOPE_FRACTION = SLOPE_FRACTION
    pipeline.MAX_CANDIDATE_DELAY = MAX_CANDIDATE_DELAY
    pipeline.DISTANCE = DISTANCE
    pipeline.THRESHOLD_METHOD = THRESHOLD_METHOD
    pipeline.COMPONENT_FRACTION = COMPONENT_FRACTION
    pipeline.FILTRATION_MAX_EDGE_LENGTH = FILTRATION_MAX_EDGE_LENGTH
    pipeline.MIN_LOOP_LIFETIME = MIN_LOOP_LIFETIME
    pipeline.MAX_SIMPLEX_DIMENSION = MAX_SIMPLEX_DIMENSION
    pipeline.MAX_HOMOLOGY_DIMENSION = MAX_HOMOLOGY_DIMENSION
    pipeline.HOMOLOGY_COEFFICIENT_FIELD = HOMOLOGY_COEFFICIENT_FIELD
    pipeline.FIGURE_DIMENSION = "auto"
    pipeline.FIGURE_PATH = "auto"
    pipeline.PROJECTION = "pca"
    pipeline.validate_controls()


def complete_embedding(grid, dimension, delay):
    """Return delay vectors whose every timestamp has a numeric value."""
    window = (dimension - 1) * delay
    if len(grid) <= window:
        return np.zeros((0, dimension))
    length = len(grid) - window
    columns = [grid[lag * delay : lag * delay + length] for lag in range(dimension)]
    embedded = np.column_stack(columns)
    complete = np.isfinite(embedded).all(axis=1)
    return embedded[complete]


def simplex_arrays(simplex_tree):
    """Return edge and triangle vertex indices, or empty arrays when none exist."""
    edges = []
    triangles = []
    for simplex, _filtration in simplex_tree.get_simplices():
        vertices = tuple(sorted(int(vertex) for vertex in simplex))
        if len(vertices) == 2:
            edges.append(vertices)
        elif len(vertices) == 3:
            triangles.append(vertices)
    edge_array = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    triangle_array = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    return edge_array, triangle_array


def betti_numbers(simplex_tree):
    """Return beta_0, ..., beta_MAX_HOMOLOGY_DIMENSION for the supplied complex."""
    counts = [0] * (MAX_HOMOLOGY_DIMENSION + 1)
    if simplex_tree.num_simplices() == 0:
        return counts
    simplex_tree.persistence(
        homology_coeff_field=HOMOLOGY_COEFFICIENT_FIELD,
        min_persistence=0,
        persistence_dim_max=MAX_HOMOLOGY_DIMENSION >= MAX_SIMPLEX_DIMENSION,
    )
    for dimension in range(MAX_HOMOLOGY_DIMENSION + 1):
        intervals = pipeline.interval_array(
            simplex_tree.persistence_intervals_in_dimension(dimension)
        )
        counts[dimension] = int(np.isinf(intervals[:, 1]).sum()) if len(intervals) else 0
    return counts


def frame_complex(points, threshold):
    """Build the complex of one window at the universal connection threshold."""
    if len(points) == 0:
        return (
            np.zeros((0, 2)),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3), dtype=np.int32),
            [0] * (MAX_HOMOLOGY_DIMENSION + 1),
        )
    distances = pipeline.pairwise_distance_matrix(points, DISTANCE)
    simplex_tree = pipeline.build_rips_complex(
        distances,
        threshold,
        MAX_SIMPLEX_DIMENSION,
    )
    edges, triangles = simplex_arrays(simplex_tree)
    return points, edges, triangles, betti_numbers(simplex_tree)


def sliding_windows(grid, delay):
    """Yield (start index, embedded points) for each raw window of 30 + delay span."""
    raw_length = WINDOW_DAYS + (EMBEDDING_DIMENSION - 1) * delay
    if len(grid) < raw_length:
        raise ValueError(
            "The series is shorter than one sliding window of "
            f"{raw_length} time steps."
        )
    start = 0
    while start + raw_length <= len(grid):
        segment = grid[start : start + raw_length]
        yield start, complete_embedding(segment, EMBEDDING_DIMENSION, delay)
        start += WINDOW_STEP_DAYS


def resolve_matlab_paths(dataset_path):
    """Return the MATLAB script path and the MAT-file it loads."""
    if str(MATLAB_PATH) == "auto":
        script_path = dataset_path.with_name(f"{dataset_path.stem}_windows.m")
    else:
        script_path = Path(MATLAB_PATH)
        if not script_path.is_absolute():
            script_path = dataset_path.parent / script_path
        if script_path.suffix.lower() != ".m":
            script_path = script_path.with_suffix(".m")
    return script_path, script_path.with_suffix(".mat")


def matlab_function_name(script_path):
    """Return a MATLAB function name matching the script file."""
    letters = []
    for character in script_path.stem:
        letters.append(character if character.isalnum() else "_")
    name = "".join(letters)
    if not name or name[0].isdigit():
        name = "windows_" + name
    return name


def write_matlab_script(script_path, mat_name, delay, threshold, time_step):
    """Write a MATLAB function that draws one precomputed frame per slider value."""
    function_name = matlab_function_name(script_path)
    raw_length = WINDOW_DAYS + (EMBEDDING_DIMENSION - 1) * delay
    script = f"""function {function_name}
% Sliding-window Vietoris-Rips complexes.
% Generated by Analysis.py.
% Universal embedding delay: {delay} time steps ({delay * time_step:.8g} time units).
% Universal connection threshold: {threshold:.8g}.
% Each frame uses {raw_length} raw time steps ({WINDOW_DAYS} plus the delay span)
% and advances by {WINDOW_STEP_DAYS} time steps.

here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, '{mat_name}'));
frameCount = numel(S.frameStart);
figureHandle = uifigure('Name', 'Sliding-window complex', 'Position', [100 100 720 560]);
axesHandle = uiaxes(figureHandle, 'Position', [40 90 640 430]);
labelHandle = uilabel(figureHandle, 'Position', [40 520 640 22], ...
    'Text', frameCaption(S, 1));
if frameCount < 2
    showFrame(1);
    return
end
sliderHandle = uislider(figureHandle, 'Position', [40 48 640 3], ...
    'Limits', [1 frameCount], 'Value', 1, ...
    'MajorTicks', unique([1 round(frameCount / 2) frameCount]));
sliderHandle.ValueChangedFcn = @(~, event) showFrame(round(event.Value));
sliderHandle.ValueChangingFcn = @(~, event) showFrame(round(event.Value));
showFrame(1);

    function showFrame(frameIndex)
        frameIndex = min(max(frameIndex, 1), frameCount);
        labelHandle.Text = frameCaption(S, frameIndex);
        cla(axesHandle);
        hold(axesHandle, 'on');
        points = S.points{{frameIndex}};
        triangles = S.triangles{{frameIndex}};
        edges = S.edges{{frameIndex}};
        if ~isempty(triangles)
            patch(axesHandle, 'Faces', triangles, 'Vertices', points, ...
                'FaceColor', [0.30 0.47 0.66], 'FaceAlpha', 0.35, 'EdgeColor', 'none');
        end
        if ~isempty(edges)
            edgeX = [points(edges(:, 1), 1).'; points(edges(:, 2), 1).'; nan(1, size(edges, 1))];
            edgeY = [points(edges(:, 1), 2).'; points(edges(:, 2), 2).'; nan(1, size(edges, 1))];
            plot(axesHandle, edgeX(:), edgeY(:), 'Color', [0.75 0.22 0.17], 'LineWidth', 1.2);
        end
        if ~isempty(points)
            scatter(axesHandle, points(:, 1), points(:, 2), 18, [0.13 0.13 0.13], 'filled');
        end
        hold(axesHandle, 'off');
        axis(axesHandle, 'equal');
        grid(axesHandle, 'on');
        xlabel(axesHandle, S.coordinateNames{{1}});
        if numel(S.coordinateNames) > 1
            ylabel(axesHandle, S.coordinateNames{{2}});
        end
        title(axesHandle, sprintf('delay = %d, threshold = %.6g', S.embeddingDelay, S.threshold));
    end
end

function caption = frameCaption(S, frameIndex)
caption = sprintf('Frame %d/%d   start %.6g   beta_0 = %g   beta_1 = %g', ...
    frameIndex, numel(S.frameStart), S.frameStart(frameIndex), ...
    S.beta0(frameIndex), S.beta1(frameIndex));
end
"""
    # The source above escapes MATLAB cell indexing for this Python string.
    script = script.replace("{{", "{").replace("}}", "}")
    script_path.write_text(script, encoding="utf-8")


def write_matlab_data(mat_path, frames, coordinate_names, delay, threshold, time_step, origin):
    """Save one cell array of geometry for each sliding window."""
    frame_count = len(frames)
    points = np.empty(frame_count, dtype=object)
    edges = np.empty(frame_count, dtype=object)
    triangles = np.empty(frame_count, dtype=object)
    frame_start = np.zeros(frame_count, dtype=float)
    beta0 = np.zeros(frame_count, dtype=float)
    beta1 = np.zeros(frame_count, dtype=float)
    for index, frame in enumerate(frames):
        points[index] = frame["points"]
        # MATLAB vertex numbering starts at 1.
        edges[index] = frame["edges"] + (1 if len(frame["edges"]) else 0)
        triangles[index] = frame["triangles"] + (1 if len(frame["triangles"]) else 0)
        frame_start[index] = origin + frame["start"] * time_step
        beta0[index] = frame["betti"][0]
        beta1[index] = frame["betti"][1] if len(frame["betti"]) > 1 else 0
    savemat(
        mat_path,
        {
            "embeddingDelay": np.array([[delay]], dtype=float),
            "threshold": np.array([[threshold]], dtype=float),
            "timeStep": np.array([[time_step]], dtype=float),
            "windowDays": np.array([[WINDOW_DAYS]], dtype=float),
            "windowStepDays": np.array([[WINDOW_STEP_DAYS]], dtype=float),
            "frameStart": frame_start.reshape(1, -1),
            "beta0": beta0.reshape(1, -1),
            "beta1": beta1.reshape(1, -1),
            "points": points.reshape(-1, 1),
            "edges": edges.reshape(-1, 1),
            "triangles": triangles.reshape(-1, 1),
            "coordinateNames": np.array(coordinate_names, dtype=object).reshape(-1, 1),
        },
        do_compression=True,
    )


def main():
    validate_controls()
    apply_pipeline_controls()
    times, values, series_name, time_name = pipeline.load_timed_series(DATASET_PATH, SERIES_COLUMN)
    origin, time_step, grid = pipeline.align_to_time_grid(times, values, series_name, time_name)
    delay_selection = pipeline.select_embedding_delay(grid, EMBEDDING_DIMENSION)
    delay = delay_selection["delay"]
    pipeline.EMBEDDING_DELAY = delay
    pipeline.print_delay_selection(delay_selection)
    print()

    embedded = pipeline.delay_embed(grid, EMBEDDING_DIMENSION, delay)
    distances = pipeline.pairwise_distance_matrix(embedded, DISTANCE)
    filtration = pipeline.build_rips_complex(
        distances,
        FILTRATION_MAX_EDGE_LENGTH,
        MAX_SIMPLEX_DIMENSION,
    )
    filtration.persistence(
        homology_coeff_field=HOMOLOGY_COEFFICIENT_FIELD,
        min_persistence=0,
        persistence_dim_max=MAX_HOMOLOGY_DIMENSION >= MAX_SIMPLEX_DIMENSION,
    )
    component_limit = pipeline.maximum_component_count(
        len(grid),
        int(np.isnan(grid).sum()),
        delay,
    )
    threshold_selection = pipeline.select_threshold(
        filtration.persistence_intervals_in_dimension(0),
        filtration.persistence_intervals_in_dimension(1),
        component_limit,
    )
    pipeline.print_threshold_selection(threshold_selection)
    print()
    if not threshold_selection["feasible"]:
        raise RuntimeError(
            "No Rips threshold inside FILTRATION_MAX_EDGE_LENGTH satisfies the selected rule."
        )
    threshold = threshold_selection["threshold"]

    frames = []
    for start, points in sliding_windows(grid, delay):
        _points, edges, triangles, betti = frame_complex(points, threshold)
        frames.append(
            {
                "start": start,
                "points": np.asarray(_points, dtype=float),
                "edges": np.asarray(edges, dtype=np.int32),
                "triangles": np.asarray(triangles, dtype=np.int32),
                "betti": betti,
            }
        )
    script_path, mat_path = resolve_matlab_paths(DATASET_PATH)
    coordinate_names = pipeline.embedded_column_names(series_name, EMBEDDING_DIMENSION, delay)
    write_matlab_data(
        mat_path,
        frames,
        coordinate_names,
        delay,
        threshold,
        time_step,
        origin,
    )
    write_matlab_script(script_path, mat_path.name, delay, threshold, time_step)
    raw_length = WINDOW_DAYS + (EMBEDDING_DIMENSION - 1) * delay
    print("Sliding windows")
    print(f"Raw window: {raw_length} time steps")
    print(f"Window step: {WINDOW_STEP_DAYS} time steps")
    print(f"Frames: {len(frames)}")
    print(f"MATLAB script: {script_path}")
    print(f"MATLAB data: {mat_path}")


if __name__ == "__main__":
    main()
