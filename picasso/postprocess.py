"""
picasso.postprocess
~~~~~~~~~~~~~~~~~~~

Data analysis of localization lists.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import itertools
import os
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Literal, NamedTuple
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from threading import Thread

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.spatial import KDTree
from scipy.stats import rankdata

from . import (
    io,
    lib,
    clusterer,
    render,
    imageprocess,
    masking,
    spatial_index,
)

__all__ = [
    # picks
    "picked_locs",
    "pick_similar",
    "remove_locs_in_picks",
    "pick_properties",
    # binding events
    "link",
    "compute_dark_times",
    "dark_times",
    "select_binding_event_cores",
    "combine_locs_in_picks",
    "groupprops",
    # drift and alignment
    "undrift",
    "undrift_from_fiducials",
    "undrift_from_picked",
    "apply_drift",
    "plot_drift",
    "align",
    "align_rcc",
    "align_from_picked",
    # resolution
    "nena",
    "plot_nena",
    "frc",
    "frc_rois",
    "plot_frc",
    # spatial analysis
    "nn_analysis",
    "pair_correlation",
    "distance_histogram",
    "compute_local_density",
    "cbc",
    "cbc_all_pairs",
    "resi",
    # spatial statistics
    "ripley_h",
    "plot_ripley_h",
    "rdf",
    "plot_rdf",
    "nnd_distributions",
    "plot_nnd_distributions",
]

#: Convergence tolerance (nm) of the center-of-mass iterations of
#: ``pick_similar``.
PICK_CONVERGENCE = 0.1
#: Pixel size (nm) of the images rendered for undrifting by RCC
#: (``undrift``, ``segment``).
UNDRIFT_PIXEL_SIZE = 100.0
#: Largest shift (nm) between two segments considered by ``undrift``.
UNDRIFT_MAX_SHIFT = 10000.0
#: Fewest segments ``undrift`` accepts: the drift between them is
#: interpolated by a cubic spline.
MIN_UNDRIFT_SEGMENTS = 4
#: Pixel size (nm) of the images rendered by ``align_rcc``.
ALIGN_PIXEL_SIZE = 100.0
#: Convergence tolerance (nm) of the iterations of ``align_rcc``.
ALIGN_CONVERGENCE = 0.1


def _is_pyramid(index_blocks) -> bool:
    """Whether a pick index is a ``spatial_index.RenderIndexPyramid``
    (built once per channel, any pick size) rather than the tuple of
    ``get_index_blocks`` (built for one block size)."""
    return isinstance(index_blocks, spatial_index.RenderIndexPyramid)


def get_index_blocks(
    locs: pd.DataFrame,
    info: list[dict],
    size: float,
) -> tuple:
    """Split localizations into blocks of the given size. Used for fast
    localization indexing (e.g., for picking).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations list.
    size : float
        Size of the blocks in nm. For circular picks, this is the pick
        radius.

    Returns
    -------
    locs : pd.DataFrame
        Localizations in the specified blocks.
    size : float
        Size of the blocks in nm.
    x_index : lib.IntArray1D
        x indices of the localizations in the blocks.
    y_index : lib.IntArray1D
        y indices of the localizations in the blocks.
    block_starts : lib.IntArray2D
        Block start indices.
    block_ends : lib.IntArray2D
        Block end indices.
    K : int
        Number of blocks in y direction.
    L : int
        Number of blocks in x direction.
    """
    locs = lib.ensure_sanity(locs, info)
    # Sort locs by indices
    x_index = np.uint32(locs["x"].to_numpy() / size)
    y_index = np.uint32(locs["y"].to_numpy() / size)
    sort_indices = np.lexsort([x_index, y_index])
    locs = locs.iloc[sort_indices]
    x_index = x_index[sort_indices]
    y_index = y_index[sort_indices]
    # Allocate block info arrays
    n_blocks_y, n_blocks_x = _index_blocks_shape(info, size)
    block_starts = np.zeros((n_blocks_y, n_blocks_x), dtype=np.uint32)
    block_ends = np.zeros((n_blocks_y, n_blocks_x), dtype=np.uint32)
    K, L = block_starts.shape
    # Fill in block starts and ends
    thread = Thread(
        target=_fill_index_blocks,
        args=(block_starts, block_ends, x_index, y_index),
    )
    thread.start()
    thread.join()
    return locs, size, x_index, y_index, block_starts, block_ends, K, L


def _index_blocks_shape(info: list[dict], size: float) -> tuple[int, int]:
    """Return the shape of the index grid, given the movie and grid
    sizes.

    Parameters
    ----------
    info : list of dicts
        Metadata of the localizations list.
    size : float
        Size of the blocks (nm).

    Returns
    -------
    n : tuple
        Number of blocks in y and x.
    """
    width, height = lib.field_of_view(info)
    n_blocks_x = int(np.ceil(width / size))
    n_blocks_y = int(np.ceil(height / size))
    n = (n_blocks_y, n_blocks_x)
    return n


@numba.jit(nopython=True, nogil=True)
def _fill_index_blocks(
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    x_index: lib.IntArray1D,
    y_index: lib.IntArray1D,
) -> None:
    """Fill the block starts and ends arrays with the indices of
    localizations in the blocks."""
    Y, X = block_starts.shape
    N = len(x_index)
    k = 0
    for i in range(Y):
        for j in range(X):
            k = _fill_index_block(
                block_starts, block_ends, N, x_index, y_index, i, j, k
            )


@numba.jit(nopython=True, nogil=True)
def _fill_index_block(
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    N: int,
    x_index: lib.IntArray1D,
    y_index: lib.IntArray1D,
    i: int,
    j: int,
    k: int,
) -> int:
    """Fill the block starts and ends arrays for a single block."""
    block_starts[i, j] = k
    while k < N and y_index[k] == i and x_index[k] == j:
        k += 1
    block_ends[i, j] = k
    return k


def _picked_circular_locs(
    locs: pd.DataFrame,
    info: list[dict],
    picks: list[tuple],
    radius: float,
    index_blocks: tuple | None,
    add_group: bool,
    progress: lib.Progress,
) -> list[pd.DataFrame]:
    """Helper function for picking localizations using circular picks.
    See ``picked_locs`` for more details."""
    picked_locs = []
    if _is_pyramid(index_blocks):
        # the load-time pyramid: no pick-size specific indexing needed
        pyramid = index_blocks
        xs = locs["x"].to_numpy()
        ys = locs["y"].to_numpy()
    else:
        pyramid = None
        if index_blocks is None:
            index_blocks = get_index_blocks(locs, info, radius)
        locs_xy = index_blocks[0][["x", "y"]].to_numpy().T
    for i, pick in enumerate(picks):
        x, y = pick
        if pyramid is not None:
            idx = spatial_index.query_circle(pyramid, xs, ys, x, y, radius)
            group_locs = locs.iloc[idx].copy()
        else:
            x_, y_ = int(x / radius), int(y / radius)
            block_locs_idx = _get_block_locs_at(
                x_,
                y_,
                index_blocks[4],
                index_blocks[5],
                index_blocks[6],
                index_blocks[7],
            )
            block_locs = index_blocks[0].iloc[block_locs_idx]
            group_locs_idx = lib.is_loc_in_circle(
                x, y, locs_xy[:, block_locs_idx], radius
            )
            group_locs = block_locs.iloc[group_locs_idx].copy()

        if add_group:
            group_locs = lib.append_group(group_locs, i)
        group_locs.sort_values(
            by="frame",
            kind="quicksort",
            inplace=True,
        )
        picked_locs.append(group_locs)

        progress.set_value(i + 1)
    return picked_locs


def _picked_rectangular_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    pick_size: float,
    add_group: bool,
    progress: lib.Progress,
) -> list[pd.DataFrame]:
    """Helper function for picking localizations using rectangular
    picks. See ``picked_locs`` for more details."""
    picked_locs = []
    for i, pick in enumerate(picks):
        (xs, ys), (xe, ye) = pick
        X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
        x_min = min(X)
        x_max = max(X)
        y_min = min(Y)
        y_max = max(Y)
        mask = (
            (locs["x"] > x_min)
            & (locs["x"] < x_max)
            & (locs["y"] > y_min)
            & (locs["y"] < y_max)
        )
        group_locs = lib.locs_in_rectangle(locs[mask], X, Y).copy()
        # store rotated coordinates in x_rot and y_rot
        angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
        x_shifted = group_locs["x"] - xs
        y_shifted = group_locs["y"] - ys
        x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
        y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)
        group_locs["x_pick_rot"] = x_pick_rot
        group_locs["y_pick_rot"] = y_pick_rot
        if add_group:
            group_locs = lib.append_group(group_locs, i)
        group_locs.sort_values(by="frame", kind="quicksort", inplace=True)
        picked_locs.append(group_locs)

        progress.set_value(i + 1)
    return picked_locs


def _picked_polygonal_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    add_group: bool,
    progress: lib.Progress,
):
    """Helper function for picking localizations using polygonal picks. See
    ``picked_locs`` for more details."""
    picked_locs = []
    for i, pick in enumerate(picks):
        X, Y = lib.get_pick_polygon_corners(pick)
        if X is None:
            progress.set_value(i + 1)
            continue
        mask = (
            (locs["x"] > min(X))
            & (locs["x"] < max(X))
            & (locs["y"] > min(Y))
            & (locs["y"] < max(Y))
        )
        group_locs = lib.locs_in_polygon(locs[mask], X, Y).copy()
        if add_group:
            group_locs = lib.append_group(group_locs, i)
        group_locs.sort_values(by="frame", kind="quicksort", inplace=True)
        picked_locs.append(group_locs)

        progress.set_value(i + 1)
    return picked_locs


def _picked_square_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    pick_size: float,
    add_group: bool,
    progress: lib.Progress,
) -> list[pd.DataFrame]:
    """Helper function for picking localizations using square picks. See
    ``picked_locs`` for more details."""
    picked_locs = []
    for i, pick in enumerate(picks):
        x, y = pick
        half_a = pick_size / 2
        x_min = x - half_a
        x_max = x + half_a
        y_min = y - half_a
        y_max = y + half_a
        mask = (
            (locs["x"] > x_min)
            & (locs["x"] < x_max)
            & (locs["y"] > y_min)
            & (locs["y"] < y_max)
        )
        group_locs = locs[mask].copy()
        if add_group:
            group_locs = lib.append_group(group_locs, i)
        group_locs.sort_values(by="frame", kind="quicksort", inplace=True)
        picked_locs.append(group_locs)

        progress.set_value(i + 1)
    return picked_locs


def _picked_box_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    add_group: bool,
    progress: lib.Progress,
) -> list[pd.DataFrame]:
    """Helper function for picking localizations using box picks. See
    ``picked_locs`` for more details."""
    picked_locs = []
    for i, pick in enumerate(picks):
        (x0, y0), (x1, y1) = pick
        x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
        y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
        mask = (
            (locs["x"] > x_min)
            & (locs["x"] < x_max)
            & (locs["y"] > y_min)
            & (locs["y"] < y_max)
        )
        group_locs = locs[mask].copy()
        if add_group:
            group_locs = lib.append_group(group_locs, i)
        group_locs.sort_values(by="frame", kind="quicksort", inplace=True)
        picked_locs.append(group_locs)

        progress.set_value(i + 1)
    return picked_locs


def _picked_brush_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    add_group: bool,
    progress: lib.Progress,
) -> list[pd.DataFrame]:
    """Helper function for picking localizations using brush picks. See
    ``picked_locs`` for more details."""
    picked_locs = []
    for i, pick in enumerate(picks):
        x_min, x_max, y_min, y_max = lib.pick_bounds(pick, "Brush", None)
        mask = (
            (locs["x"] > x_min)
            & (locs["x"] < x_max)
            & (locs["y"] > y_min)
            & (locs["y"] < y_max)
        )
        group_locs = lib.locs_in_brush(locs[mask], pick).copy()
        if add_group:
            group_locs = lib.append_group(group_locs, i)
        group_locs.sort_values(by="frame", kind="quicksort", inplace=True)
        picked_locs.append(group_locs)

        progress.set_value(i + 1)
    return picked_locs


def picked_locs(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float = None,
    add_group: bool = True,
    index_blocks: tuple = None,
    progress: lib.Progress | Literal["console"] | None = None,
) -> list[pd.DataFrame]:
    """Find picked localizations, i.e., localizations within the given
    regions of interest.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations list.
    picks : list
        List of picks.
    pick_shape : {'Circle', 'Rectangle', 'Polygon', 'Square', 'Box', 'Brush'}
        Shape of the pick.
    pick_size : float, optional
        Size of the pick in nm. Diameter for the circles, width
        for the rectangles, side length for squares, None for the
        polygons, boxes and brush picks (they carry their own extent).
        Default is None.
    add_group : boolean, optional
        True if group id should be added to locs. Each pick will be
        assigned a different id. Default is True.
    index_blocks : tuple or spatial_index.RenderIndexPyramid, optional
        Used only for circular picks. Precomputed index blocks for
        localizations, see ``get_index_blocks`` (built with block size
        ``pick_size / 2``, the pick radius), or the pick-size
        independent pyramid of
        ``spatial_index.build_render_index``. If None, index blocks
        will be calculated internally. Default is None.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Picking localizations" over
        the picks. Default is None.

    Returns
    -------
    picked_locs : list of pd.DataFrames
        List of pd.DataFrames, each containing locs from one pick.
    """
    _valid_shapes = lib.PICK_SHAPES
    assert (
        pick_shape in _valid_shapes
    ), f"Invalid pick shape: {pick_shape}. Choose one of {_valid_shapes}."
    if len(picks) == 0:
        return []

    with lib.tracked_progress(
        progress, "Picking localizations", unit="pick"
    ) as progress:
        lib.start_phase(
            progress, "Picking localizations", len(picks), unit="pick"
        )
        progress.set_value(0)
        if pick_shape == "Circle":
            picked_locs = _picked_circular_locs(
                locs=locs,
                info=info,
                picks=picks,
                radius=pick_size / 2,
                index_blocks=index_blocks,
                add_group=add_group,
                progress=progress,
            )
        elif pick_shape == "Rectangle":
            picked_locs = _picked_rectangular_locs(
                locs=locs,
                picks=picks,
                pick_size=pick_size,
                add_group=add_group,
                progress=progress,
            )
        elif pick_shape == "Polygon":
            picked_locs = _picked_polygonal_locs(
                locs=locs,
                picks=picks,
                add_group=add_group,
                progress=progress,
            )
        elif pick_shape == "Square":
            picked_locs = _picked_square_locs(
                locs=locs,
                picks=picks,
                pick_size=pick_size,
                add_group=add_group,
                progress=progress,
            )
        elif pick_shape == "Box":
            picked_locs = _picked_box_locs(
                locs=locs,
                picks=picks,
                add_group=add_group,
                progress=progress,
            )
        elif pick_shape == "Brush":
            picked_locs = _picked_brush_locs(
                locs=locs,
                picks=picks,
                add_group=add_group,
                progress=progress,
            )
        progress.set_value(len(picks))
    return picked_locs


def pick_similar(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple],
    pick_shape: Literal["Circle", "Rectangle", "Square", "Box"] = "Circle",
    pick_size: float = None,
    std_range: float = 2.0,
    index_blocks: tuple = None,
    grid_spacing: float = None,
) -> list[tuple]:
    """Find picks similar to the given ones, based on the number of
    localizations and their RMSD.

    Mean number of localizations and RMSD in picks are calculated and
    the allowed range is defined by the given standard deviation range
    (``std_range``). Only picks with number of localizations and RMSD
    within these ranges are returned. The input picks are always part of
    the output.

    A grid of overlapping candidate picks covering the field of view is
    scanned and each candidate is shifted towards the center of mass of
    the localizations within it. Rectangular picks are additionally
    rotated onto the principal axis of the enclosed localizations, so
    elongated structures are found at any orientation; they all take the
    median length of the input picks. Box picks likewise all take the
    median width and height of the input picks. Candidates overlapping
    an already accepted pick are discarded.

    Not implemented for polygonal picks, which have no size or
    canonical form to replicate.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    picks : list
        List of picks. ``(x, y)`` coordinates for circular and square
        picks, ``((x_start, y_start), (x_end, y_end))`` center-axis
        points for rectangular picks, and the two opposite corners
        ``((x0, y0), (x1, y1))`` for box picks.
    pick_shape : {'Circle', 'Rectangle', 'Square', 'Box'}, optional
        Shape of the picks. Default is 'Circle'.
    pick_size : float, optional
        Size of the pick in nm. Diameter for circles, side length for
        squares, width for rectangles. Ignored for boxes,
        which carry their own extent. Default is None.
    std_range : float, optional
        Standard deviation range for picking similar localizations.
        Default is 2.0.
    index_blocks : tuple, optional
        Precomputed index blocks for localizations, see
        ``get_index_blocks``. Rebuilt internally if None or if the block
        size does not match the one required by ``pick_shape``. Default
        is None.
    grid_spacing : float, optional
        Distance between neighboring candidate positions in nm. Only
        used for rectangular picks, where the default
        (a quarter of the median pick length) trades speed for the
        chance of missing closely spaced structures. Default is None.

    Returns
    -------
    new_picks : list of tuples
        List of similar picks, in the same format as ``picks``.
    """
    _validate_pick_similar_args(pick_shape, pick_size, grid_spacing)
    if len(picks) == 0:
        return []

    # the index grid size must guarantee that the 3x3 block neighborhood
    # around a pick's center contains all localizations in that pick
    block_size = _pick_similar_block_size(pick_shape, picks, pick_size)
    index_blocks = _resolve_pick_similar_index_blocks(
        locs, info, index_blocks, pick_shape, block_size
    )

    if pick_shape == "Circle":
        return _pick_similar_circular(
            locs, info, picks, pick_size, std_range, index_blocks
        )
    elif pick_shape == "Square":
        return _pick_similar_square(
            info, picks, pick_size, std_range, index_blocks
        )
    elif pick_shape == "Box":
        box_w, box_h = _median_box_size(picks)
        return _pick_similar_box(
            info, picks, box_w, box_h, std_range, index_blocks
        )
    else:
        return _pick_similar_rectangular(
            info, picks, pick_size, std_range, index_blocks, grid_spacing
        )


def _validate_pick_similar_args(
    pick_shape: str, pick_size: float | None, grid_spacing: float | None
) -> None:
    """Validate ``pick_similar``'s shape/size/spacing arguments."""
    _valid_shapes = ("Circle", "Rectangle", "Square", "Box")
    assert (
        pick_shape in _valid_shapes
    ), f"Invalid pick shape: {pick_shape}. Choose one of {_valid_shapes}."
    if pick_shape != "Box":
        assert isinstance(
            pick_size, (int, float)
        ), "pick_size must be a number."
    if grid_spacing is not None and pick_shape != "Rectangle":
        raise ValueError(
            "grid_spacing is only supported for rectangular picks."
        )


def _pick_similar_block_size(
    pick_shape: str, picks: list[tuple], pick_size: float | None
) -> float:
    """Index-block half-size covering a pick's 3x3 block neighborhood."""
    if pick_shape == "Rectangle":
        length = _median_pick_length(picks)
        return np.sqrt(length**2 + pick_size**2) / 2
    if pick_shape == "Box":
        box_w, box_h = _median_box_size(picks)
        # a box reaches at most half its longer side in x and y
        return max(box_w, box_h) / 2
    # circles and squares reach at most pick_size / 2 in x and y
    return pick_size / 2


def _resolve_pick_similar_index_blocks(
    locs: pd.DataFrame,
    info: list[dict],
    index_blocks: tuple | None,
    pick_shape: str,
    block_size: float,
) -> tuple:
    """Reuse ``index_blocks`` when compatible, else rebuild for ``block_size``.

    A pyramid has no fixed block size; keep it only for circular picks,
    which use it to characterize the current picks rather than to walk
    the grid search (see ``_pick_similar_circular``).
    """
    if _is_pyramid(index_blocks):
        if pick_shape != "Circle":
            index_blocks = None
    elif index_blocks is not None and not np.isclose(
        index_blocks[1], block_size
    ):
        index_blocks = None
    if index_blocks is None:
        index_blocks = get_index_blocks(locs, info, block_size)
    return index_blocks


def _median_pick_length(picks: list[tuple]) -> float:
    """Return the median length of the center axes of rectangular
    picks. The median (rather than the mean) keeps a single carelessly
    drawn pick from skewing every candidate."""
    lengths = [
        np.hypot(end[0] - start[0], end[1] - start[1]) for start, end in picks
    ]
    return float(np.median(lengths))


def _pick_similar_circular(
    locs: pd.DataFrame,
    info: list[dict],
    picks: list[tuple],
    d: float,
    std_range: float,
    index_blocks: tuple,
) -> list[tuple]:
    """Helper function for finding picks similar to circular picks.
    See ``pick_similar`` for more details.

    Calls ``_pick_similar``, which is implemented in numba for speed.
    """
    r = d / 2
    d2 = d**2
    # extract n_locs and rmsd from current picks
    pyramid = index_blocks if _is_pyramid(index_blocks) else None
    if pyramid is not None or index_blocks is None:
        # the grid search needs blocks of the pick size
        index_blocks = get_index_blocks(locs, info, r)
    locs_xy = index_blocks[0][["x", "y"]].to_numpy().T
    if pyramid is not None:
        xs = locs["x"].to_numpy()
        ys = locs["y"].to_numpy()
    n_locs = []
    rmsd = []
    for i, pick in enumerate(picks):
        x, y = pick
        if pyramid is not None:
            idx = spatial_index.query_circle(pyramid, xs, ys, x, y, r)
            pick_locs_xy = np.stack((xs[idx], ys[idx]))
        else:
            block_locs_xy = get_block_locs_at(
                int(x / r),
                int(y / r),
                locs_xy,
                index_blocks[4],
                index_blocks[5],
                index_blocks[6],
                index_blocks[7],
            )
            pick_locs_xy = lib.locs_in_circle(x, y, block_locs_xy, r)
        n_locs.append(pick_locs_xy.shape[1])
        rmsd.append(lib.rmsd_at_com(pick_locs_xy))

    # calculate min and max n_locs and rmsd for picking similar
    mean_n_locs = np.mean(n_locs)
    mean_rmsd = np.mean(rmsd)
    std_n_locs = np.std(n_locs)
    std_rmsd = np.std(rmsd)
    min_n_locs = max(2, mean_n_locs - std_range * std_n_locs)
    max_n_locs = mean_n_locs + std_range * std_n_locs
    min_rmsd = mean_rmsd - std_range * std_rmsd
    max_rmsd = mean_rmsd + std_range * std_rmsd

    # x, y coordinates of found regions:
    x_similar = np.array([_[0] for _ in picks], dtype=np.float64)
    y_similar = np.array([_[1] for _ in picks], dtype=np.float64)

    # preparations for grid search
    fov_width, fov_height = lib.field_of_view(info)
    x_range = np.arange(d / 2, fov_width, np.sqrt(3) * d / 2)
    y_range_base = np.arange(d / 2, fov_height - d / 2, d)
    y_range_shift = y_range_base + d / 2
    locs_temp, size, _, _, block_starts, block_ends, K, L = index_blocks
    locs_xy = np.stack((locs_temp.x, locs_temp.y))
    x_r = np.uint64(x_range / size)
    y_r1 = np.uint64(y_range_shift / size)
    y_r2 = np.uint64(y_range_base / size)
    # pick similar
    x_similar, y_similar = _pick_similar(
        x_range,
        y_range_shift,
        y_range_base,
        min_n_locs,
        max_n_locs,
        min_rmsd,
        max_rmsd,
        x_r,
        y_r1,
        y_r2,
        locs_xy,
        block_starts,
        block_ends,
        K,
        L,
        x_similar,
        y_similar,
        r,
        d2,
    )
    new_picks = list(zip(x_similar, y_similar))
    return new_picks


@numba.jit(nopython=True, nogil=True, cache=True)
def _pick_similar(  # noqa: C901
    x: lib.FloatArray1D,
    y_shift: lib.FloatArray1D,
    y_base: lib.FloatArray1D,
    min_n_locs: int,
    max_n_locs: int,
    min_rmsd: float,
    max_rmsd: float,
    x_r: lib.IntArray1D,
    y_r1: lib.IntArray1D,
    y_r2: lib.IntArray1D,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    x_similar: lib.FloatArray1D,
    y_similar: lib.FloatArray1D,
    r: float,
    d2: float,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Find similar picks based on the number of localizations and
    RMSD. Only implemented for circular picks.

    Takes the grid of overlapping picks of the given size (defined by
    ``x``, ``y_shift`` and ``y_base``) and shifts each pick towards the
    center of mass of the localizations within the pick. If the picked
    localizations have the required number of localizations and the
    RMSD, it is added to the output list (``x_similar`` and
    ``y_similar``).

    This function is implemented in numba for speed, called from
    ``pick_similar``. See that function for more user-friendly
    interface.

    Parameters
    ----------
    x : lib.FloatArray1D
        x coordinates of the picks.
    y_shift : lib.FloatArray1D
        y coordinates of the picks, shifted for odd columns.
    y_base : lib.FloatArray1D
        y coordinates of the picks, not shifted.
    min_n_locs, max_n_locs : int
        Minimum and maximum number of localizations in the pick.
    min_rmsd, max_rmsd : float
        Minimum and maximum RMSD for the pick.
    x_r, y_r1, y_r2 : lib.IntArray1D
        x and y ranges for the picks.
    locs_xy : lib.FloatArray2D
        Localizations in the blocks.
    block_starts : lib.IntArray2D
        Block start indices.
    block_ends : lib.IntArray2D
        Block end indices.
    K, L : int
        Number of blocks in y and x direction.
    x_similar, y_similar : lib.FloatArray1D
        Arrays to store the x and y coordinates of the similar picks.
    r : float
        Radius for the picks.
    d2 : float
        Squared distance threshold for the picks.

    Returns
    -------
    x_similar, y_similar : lib.FloatArray1D
        Arrays with the x and y coordinates of the similar picks.
    """
    for i, x_grid in enumerate(x):
        x_range = x_r[i]
        # y_grid is shifted for odd columns
        if i % 2:
            y = y_shift
            y_r = y_r1
        else:
            y = y_base
            y_r = y_r2
        for j, y_grid in enumerate(y):
            y_range = y_r[j]
            n_block_locs = _n_block_locs_at(
                x_range,
                y_range,
                K,
                L,
                block_starts,
                block_ends,
            )
            if n_block_locs >= min_n_locs:
                block_locs_xy = get_block_locs_at(
                    x_range,
                    y_range,
                    locs_xy,
                    block_starts,
                    block_ends,
                    K,
                    L,
                )
                picked_locs_xy = lib.locs_in_circle(
                    x_grid, y_grid, block_locs_xy, r
                )
                if picked_locs_xy.shape[1] > 1:
                    # Move to COM peak
                    x_test_old = x_grid
                    y_test_old = y_grid
                    x_test = np.mean(picked_locs_xy[0])
                    y_test = np.mean(picked_locs_xy[1])
                    count = 0
                    while (
                        np.abs(x_test - x_test_old) > PICK_CONVERGENCE
                        or np.abs(y_test - y_test_old) > PICK_CONVERGENCE
                    ):
                        count += 1
                        # skip the locs if the loop is too long
                        if count > 500:
                            break
                        x_test_old = x_test
                        y_test_old = y_test
                        picked_locs_xy = lib.locs_in_circle(
                            x_test, y_test, block_locs_xy, r
                        )
                        if picked_locs_xy.shape[1] > 1:
                            x_test = np.mean(picked_locs_xy[0])
                            y_test = np.mean(picked_locs_xy[1])
                        else:
                            break
                    if np.all(
                        (x_similar - x_test) ** 2 + (y_similar - y_test) ** 2
                        > d2
                    ):
                        if min_n_locs <= picked_locs_xy.shape[1] <= max_n_locs:
                            if (
                                min_rmsd
                                <= lib.rmsd_at_com(picked_locs_xy)
                                <= max_rmsd
                            ):
                                x_similar = np.append(x_similar, x_test)
                                y_similar = np.append(y_similar, y_test)
    return x_similar, y_similar


def _pick_similar_grid(
    info: list[dict],
    spacing: float,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Return the candidate positions for picking similar.

    The positions form a triangular lattice with nearest neighbor
    distance ``spacing``, i.e., the densest lattice for a given number
    of candidates. This is the same lattice as the one used for
    circular picks, flattened to 1D arrays.

    Parameters
    ----------
    info : list of dicts
        Metadata of the localizations.
    spacing : float
        Distance between neighboring candidates in nm.

    Returns
    -------
    grid_x, grid_y : lib.FloatArray1D
        Coordinates of the candidate positions (nm).
    """
    width, height = lib.field_of_view(info)
    x_range = np.arange(spacing / 2, width, np.sqrt(3) * spacing / 2)
    y_range = np.arange(spacing / 2, height - spacing / 2, spacing)
    n_cols = len(x_range)
    n_rows = len(y_range)
    grid_x = np.repeat(x_range, n_rows)
    grid_y = np.tile(y_range, n_cols)
    # every other column is shifted by half the spacing
    is_odd_column = np.repeat(np.arange(n_cols) % 2, n_rows)
    grid_y = grid_y + is_odd_column * spacing / 2
    return grid_x, grid_y


def _similarity_window(
    values: list[float],
    std_range: float,
    minimum: float = -np.inf,
) -> tuple[float, float]:
    """Return the ``mean +/- std_range * std`` acceptance window for one
    similarity measure, clipped at ``minimum`` from below."""
    mean = np.mean(values)
    std = np.std(values)
    return (
        max(minimum, mean - std_range * std),
        mean + std_range * std,
    )


@numba.jit(nopython=True, nogil=True, cache=True)
def _circle_moments_at(
    xc: float,
    yc: float,
    r: float,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    x_index: int,
    y_index: int,
) -> tuple[int, float, float, float, float, float]:
    """Return the number of localizations, their center of mass and
    their central second moments within a circle of radius ``r``
    centered at ``(xc, yc)``.

    The localizations are read directly from the 3x3 block neighborhood
    around ``(x_index, y_index)``, without allocating intermediate
    arrays - the picking kernels call this for every candidate position
    and iteration, so the allocations would otherwise dominate the
    runtime. Coordinates are shifted by ``(xc, yc)`` before accumulation
    to avoid loss of precision in the second moments.

    Note that the block bounds are inclusive of the first row and
    column (``0 <= k < K``), like ``_get_block_locs_at`` and
    unlike ``_n_block_locs_at``.

    Parameters
    ----------
    xc, yc : float
        Center of the circle.
    r : float
        Radius of the circle.
    locs_xy : lib.FloatArray2D
        Localization coordinates, shape ``(2, N)``, sorted by index
        blocks.
    block_starts, block_ends : lib.IntArray2D
        Block start and end indices.
    K, L : int
        Number of blocks in y and x direction.
    x_index, y_index : int
        Block indices of the neighborhood to scan.

    Returns
    -------
    n : int
        Number of localizations within the circle.
    mx, my : float
        Center of mass of these localizations.
    sxx, sxy, syy : float
        Their central second moments.
    """
    r2 = r**2
    n = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for ll in range(x_index - 1, x_index + 2):
                if 0 <= ll < L:
                    start = np.int64(block_starts[k, ll])
                    end = np.int64(block_ends[k, ll])
                    for i in range(start, end):
                        dx = locs_xy[0, i] - xc
                        dy = locs_xy[1, i] - yc
                        if dx**2 + dy**2 < r2:
                            n += 1
                            sx += dx
                            sy += dy
                            sxx += dx * dx
                            sxy += dx * dy
                            syy += dy * dy
    return _central_moments(n, xc, yc, sx, sy, sxx, sxy, syy)


@numba.jit(nopython=True, nogil=True, cache=True)
def _square_moments_at(
    xc: float,
    yc: float,
    a: float,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    x_index: int,
    y_index: int,
) -> tuple[int, float, float, float, float, float]:
    """Return the number of localizations, their center of mass and
    their central second moments within an axis-aligned square of side
    length ``a`` centered at ``(xc, yc)``. See ``_circle_moments_at``.
    """
    half_a = 0.5 * a
    n = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for ll in range(x_index - 1, x_index + 2):
                if 0 <= ll < L:
                    start = np.int64(block_starts[k, ll])
                    end = np.int64(block_ends[k, ll])
                    for i in range(start, end):
                        dx = locs_xy[0, i] - xc
                        if -half_a < dx < half_a:
                            dy = locs_xy[1, i] - yc
                            if -half_a < dy < half_a:
                                n += 1
                                sx += dx
                                sy += dy
                                sxx += dx * dx
                                sxy += dx * dy
                                syy += dy * dy
    return _central_moments(n, xc, yc, sx, sy, sxx, sxy, syy)


@numba.jit(nopython=True, nogil=True, cache=True)
def _box_moments_at(
    xc: float,
    yc: float,
    w: float,
    h: float,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    x_index: int,
    y_index: int,
) -> tuple[int, float, float, float, float, float]:
    """Return the number of localizations, their center of mass and
    their central second moments within an axis-aligned box of width
    ``w`` and height ``h`` centered at ``(xc, yc)``. Generalization of
    ``_square_moments_at`` to independent side lengths.
    """
    half_w = 0.5 * w
    half_h = 0.5 * h
    n = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for ll in range(x_index - 1, x_index + 2):
                if 0 <= ll < L:
                    start = np.int64(block_starts[k, ll])
                    end = np.int64(block_ends[k, ll])
                    for i in range(start, end):
                        dx = locs_xy[0, i] - xc
                        if -half_w < dx < half_w:
                            dy = locs_xy[1, i] - yc
                            if -half_h < dy < half_h:
                                n += 1
                                sx += dx
                                sy += dy
                                sxx += dx * dx
                                sxy += dx * dy
                                syy += dy * dy
    return _central_moments(n, xc, yc, sx, sy, sxx, sxy, syy)


@numba.jit(nopython=True, nogil=True, cache=True)
def _rectangle_moments_at(
    xc: float,
    yc: float,
    theta: float,
    length: float,
    width: float,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    x_index: int,
    y_index: int,
) -> tuple[int, float, float, float, float, float]:
    """Return the number of localizations, their center of mass and
    their central second moments within an oriented rectangle centered
    at ``(xc, yc)``. See ``_circle_moments_at``."""
    ct = np.cos(theta)
    st = np.sin(theta)
    half_l = 0.5 * length
    half_w = 0.5 * width
    n = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for ll in range(x_index - 1, x_index + 2):
                if 0 <= ll < L:
                    start = np.int64(block_starts[k, ll])
                    end = np.int64(block_ends[k, ll])
                    for i in range(start, end):
                        dx = locs_xy[0, i] - xc
                        dy = locs_xy[1, i] - yc
                        u = dx * ct + dy * st
                        if -half_l < u < half_l:
                            v = -dx * st + dy * ct
                            if -half_w < v < half_w:
                                n += 1
                                sx += dx
                                sy += dy
                                sxx += dx * dx
                                sxy += dx * dy
                                syy += dy * dy
    return _central_moments(n, xc, yc, sx, sy, sxx, sxy, syy)


@numba.jit(nopython=True, nogil=True, cache=True)
def _central_moments(
    n: int,
    xc: float,
    yc: float,
    sx: float,
    sy: float,
    sxx: float,
    sxy: float,
    syy: float,
) -> tuple[int, float, float, float, float, float]:
    """Convert the raw moments accumulated relative to ``(xc, yc)`` into
    the center of mass and the central second moments."""
    if n == 0:
        return 0, xc, yc, 0.0, 0.0, 0.0
    inv_n = 1.0 / n
    mx = sx * inv_n
    my = sy * inv_n
    return (
        n,
        xc + mx,
        yc + my,
        sxx * inv_n - mx * mx,
        sxy * inv_n - mx * my,
        syy * inv_n - my * my,
    )


def _pick_similar_square(
    info: list[dict],
    picks: list[tuple],
    a: float,
    std_range: float,
    index_blocks: tuple,
) -> list[tuple]:
    """Helper function for finding picks similar to square picks. See
    ``pick_similar`` for more details.

    Square picks reach at most ``a / 2`` in x and y, exactly like
    circular picks of diameter ``a``, so the index blocks, the candidate
    lattice and the mean shift are the same as for circles; only the
    membership test and the overlap criterion differ.
    """
    locs_temp, block_size, _, _, block_starts, block_ends, K, L = index_blocks
    locs_xy = np.stack((locs_temp["x"].to_numpy(), locs_temp["y"].to_numpy()))

    # extract n_locs and rmsd from the current picks
    n_locs = []
    rmsd = []
    for x, y in picks:
        n, _, _, sxx, _, syy = _square_moments_at(
            x,
            y,
            a,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            int(x / block_size),
            int(y / block_size),
        )
        if n < 2:
            warnings.warn(
                f"Pick at ({x:.2f}, {y:.2f}) contains fewer than 2 "
                "localizations and is ignored when calculating the "
                "similarity criteria."
            )
            continue
        n_locs.append(n)
        rmsd.append(np.sqrt(sxx + syy))
    if not n_locs:
        raise ValueError(
            "None of the picks contains enough localizations to define "
            "the similarity criteria."
        )

    min_n_locs, max_n_locs = _similarity_window(n_locs, std_range, minimum=2)
    min_rmsd, max_rmsd = _similarity_window(rmsd, std_range)

    # the current picks are kept and block any overlapping candidates
    x_similar = np.array([_[0] for _ in picks], dtype=np.float64)
    y_similar = np.array([_[1] for _ in picks], dtype=np.float64)

    grid_x, grid_y = _pick_similar_grid(info, a)
    x_similar, y_similar = _pick_similar_square_kernel(
        grid_x,
        grid_y,
        locs_xy,
        block_starts,
        block_ends,
        K,
        L,
        block_size,
        a,
        min_n_locs,
        max_n_locs,
        min_rmsd,
        max_rmsd,
        x_similar,
        y_similar,
    )
    new_picks = list(zip(x_similar, y_similar))
    return new_picks


@numba.jit(nopython=True, nogil=True, cache=True)
def _pick_similar_square_kernel(
    grid_x: lib.FloatArray1D,
    grid_y: lib.FloatArray1D,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    block_size: float,
    a: float,
    min_n_locs: float,
    max_n_locs: float,
    min_rmsd: float,
    max_rmsd: float,
    x_similar: lib.FloatArray1D,
    y_similar: lib.FloatArray1D,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Scan the candidate lattice for squares similar to the given ones.

    Each candidate is shifted towards the center of mass of the
    localizations within it until it converges. The 3x3 block
    neighborhood is re-read at the current position in every iteration,
    so localizations are not lost when a candidate moves across a block
    boundary.

    Implemented in numba for speed, called from
    ``_pick_similar_square``. See ``pick_similar`` for a more
    user-friendly interface.

    Parameters
    ----------
    grid_x, grid_y : lib.FloatArray1D
        Candidate positions.
    locs_xy : lib.FloatArray2D
        Localization coordinates, shape ``(2, N)``, sorted by index
        blocks.
    block_starts, block_ends : lib.IntArray2D
        Block start and end indices.
    K, L : int
        Number of blocks in y and x direction.
    block_size : float
        Size of the index blocks in nm.
    a : float
        Side length of the square picks.
    min_n_locs, max_n_locs : float
        Minimum and maximum number of localizations in the pick.
    min_rmsd, max_rmsd : float
        Minimum and maximum RMSD for the pick.
    x_similar, y_similar : lib.FloatArray1D
        Centers of the accepted picks, seeded with the input picks.

    Returns
    -------
    x_similar, y_similar : lib.FloatArray1D
        Centers of the accepted picks.
    """
    for i in range(len(grid_x)):
        x_test = grid_x[i]
        y_test = grid_y[i]
        n = 0
        # move to the center of mass
        for _ in range(500):
            n, mx, my, _, _, _ = _square_moments_at(
                x_test,
                y_test,
                a,
                locs_xy,
                block_starts,
                block_ends,
                K,
                L,
                int(x_test / block_size),
                int(y_test / block_size),
            )
            if n < 2:
                break
            dx = mx - x_test
            dy = my - y_test
            x_test = mx
            y_test = my
            if (
                np.abs(dx) <= PICK_CONVERGENCE
                and np.abs(dy) <= PICK_CONVERGENCE
            ):
                break
        if n < 2:
            continue
        # measure at the converged position
        n, _, _, sxx, _, syy = _square_moments_at(
            x_test,
            y_test,
            a,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            int(x_test / block_size),
            int(y_test / block_size),
        )
        if not (min_n_locs <= n <= max_n_locs):
            continue
        if not (min_rmsd <= np.sqrt(sxx + syy) <= max_rmsd):
            continue
        # two axis-aligned squares of side a overlap if and only if
        # their centers are closer than a in both x and y
        if np.all(
            (np.abs(x_similar - x_test) > a) | (np.abs(y_similar - y_test) > a)
        ):
            x_similar = np.append(x_similar, x_test)
            y_similar = np.append(y_similar, y_test)
    return x_similar, y_similar


def _median_box_size(picks: list[tuple]) -> tuple[float, float]:
    """Return the median width and height of box picks. The median
    (rather than the mean) keeps a single carelessly drawn pick from
    skewing every candidate."""
    widths = [abs(x1 - x0) for (x0, _), (x1, _) in picks]
    heights = [abs(y1 - y0) for (_, y0), (_, y1) in picks]
    return float(np.median(widths)), float(np.median(heights))


def _pick_similar_box(
    info: list[dict],
    picks: list[tuple],
    w: float,
    h: float,
    std_range: float,
    index_blocks: tuple,
) -> list[tuple]:
    """Helper function for finding picks similar to box picks. See
    ``pick_similar`` for more details.

    Boxes carry their own extent, so unlike squares they have no single
    side length to replicate. All candidates take the median width and
    height of the input picks, and the input picks are measured in that
    canonical size (at their drawn centers) rather than exactly as
    drawn - otherwise one oversized pick would shift the acceptance
    windows away from every candidate. Their drawn geometry is still
    what is returned.
    """
    locs_temp, block_size, _, _, block_starts, block_ends, K, L = index_blocks
    locs_xy = np.stack((locs_temp["x"].to_numpy(), locs_temp["y"].to_numpy()))

    # extract n_locs and rmsd from the current picks
    n_locs = []
    rmsd = []
    for (x0, y0), (x1, y1) in picks:
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        n, _, _, sxx, _, syy = _box_moments_at(
            xc,
            yc,
            w,
            h,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            int(xc / block_size),
            int(yc / block_size),
        )
        if n < 2:
            warnings.warn(
                f"Pick at ({xc:.2f}, {yc:.2f}) contains fewer than 2 "
                "localizations and is ignored when calculating the "
                "similarity criteria."
            )
            continue
        n_locs.append(n)
        rmsd.append(np.sqrt(sxx + syy))
    if not n_locs:
        raise ValueError(
            "None of the picks contains enough localizations to define "
            "the similarity criteria."
        )

    min_n_locs, max_n_locs = _similarity_window(n_locs, std_range, minimum=2)
    min_rmsd, max_rmsd = _similarity_window(rmsd, std_range)

    # the current picks are kept as drawn and block any overlapping
    # candidates, so their own sizes take part in the overlap test
    x_similar = np.array(
        [0.5 * (p[0][0] + p[1][0]) for p in picks], dtype=np.float64
    )
    y_similar = np.array(
        [0.5 * (p[0][1] + p[1][1]) for p in picks], dtype=np.float64
    )
    w_similar = np.array(
        [abs(p[1][0] - p[0][0]) for p in picks], dtype=np.float64
    )
    h_similar = np.array(
        [abs(p[1][1] - p[0][1]) for p in picks], dtype=np.float64
    )
    n_drawn = len(picks)

    # the lattice must be fine enough along the shorter side
    grid_x, grid_y = _pick_similar_grid(info, min(w, h))
    x_similar, y_similar, w_similar, h_similar = _pick_similar_box_kernel(
        grid_x,
        grid_y,
        locs_xy,
        block_starts,
        block_ends,
        K,
        L,
        block_size,
        w,
        h,
        min_n_locs,
        max_n_locs,
        min_rmsd,
        max_rmsd,
        x_similar,
        y_similar,
        w_similar,
        h_similar,
    )
    # return the drawn picks unchanged and the new ones as corner pairs
    new_picks = list(picks[:n_drawn])
    for xc, yc, w_, h_ in zip(
        x_similar[n_drawn:],
        y_similar[n_drawn:],
        w_similar[n_drawn:],
        h_similar[n_drawn:],
    ):
        new_picks.append(
            (
                (xc - 0.5 * w_, yc - 0.5 * h_),
                (xc + 0.5 * w_, yc + 0.5 * h_),
            )
        )
    return new_picks


@numba.jit(nopython=True, nogil=True, cache=True)
def _pick_similar_box_kernel(
    grid_x: lib.FloatArray1D,
    grid_y: lib.FloatArray1D,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    block_size: float,
    w: float,
    h: float,
    min_n_locs: float,
    max_n_locs: float,
    min_rmsd: float,
    max_rmsd: float,
    x_similar: lib.FloatArray1D,
    y_similar: lib.FloatArray1D,
    w_similar: lib.FloatArray1D,
    h_similar: lib.FloatArray1D,
) -> tuple[
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
]:
    """Scan the candidate lattice for boxes similar to the given ones.

    Each candidate is shifted towards the center of mass of the
    localizations within it until it converges, exactly as in
    ``_pick_similar_square_kernel``; only the membership test and the
    overlap criterion account for independent side lengths.

    Parameters
    ----------
    grid_x, grid_y : lib.FloatArray1D
        Candidate positions.
    locs_xy : lib.FloatArray2D
        Localization coordinates, shape ``(2, N)``, sorted into blocks.
    block_starts, block_ends : lib.IntArray2D
        First and last index of the localizations in each block.
    K, L : int
        Number of blocks in y and x direction.
    block_size : float
        Size of the index blocks in nm.
    w, h : float
        Width and height of the candidate boxes.
    min_n_locs, max_n_locs : float
        Minimum and maximum number of localizations in the pick.
    min_rmsd, max_rmsd : float
        Minimum and maximum RMSD for the pick.
    x_similar, y_similar : lib.FloatArray1D
        Centers of the accepted picks, seeded with the input picks.
    w_similar, h_similar : lib.FloatArray1D
        Side lengths of the accepted picks, seeded with the input picks
        as drawn.

    Returns
    -------
    x_similar, y_similar : lib.FloatArray1D
        Centers of the accepted picks.
    w_similar, h_similar : lib.FloatArray1D
        Side lengths of the accepted picks.
    """
    for i in range(len(grid_x)):
        x_test = grid_x[i]
        y_test = grid_y[i]
        n = 0
        # move to the center of mass
        for _ in range(500):
            n, mx, my, _, _, _ = _box_moments_at(
                x_test,
                y_test,
                w,
                h,
                locs_xy,
                block_starts,
                block_ends,
                K,
                L,
                int(x_test / block_size),
                int(y_test / block_size),
            )
            if n < 2:
                break
            dx = mx - x_test
            dy = my - y_test
            x_test = mx
            y_test = my
            if (
                np.abs(dx) <= PICK_CONVERGENCE
                and np.abs(dy) <= PICK_CONVERGENCE
            ):
                break
        if n < 2:
            continue
        # measure at the converged position
        n, _, _, sxx, _, syy = _box_moments_at(
            x_test,
            y_test,
            w,
            h,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            int(x_test / block_size),
            int(y_test / block_size),
        )
        if not (min_n_locs <= n <= max_n_locs):
            continue
        if not (min_rmsd <= np.sqrt(sxx + syy) <= max_rmsd):
            continue
        # two axis-aligned boxes overlap if and only if their centers
        # are closer than the sum of their half-sides in both x and y
        if np.all(
            (np.abs(x_similar - x_test) > 0.5 * (w_similar + w))
            | (np.abs(y_similar - y_test) > 0.5 * (h_similar + h))
        ):
            x_similar = np.append(x_similar, x_test)
            y_similar = np.append(y_similar, y_test)
            w_similar = np.append(w_similar, w)
            h_similar = np.append(h_similar, h)
    return x_similar, y_similar, w_similar, h_similar


def _pick_similar_rectangular(
    info: list[dict],
    picks: list[tuple],
    width: float,
    std_range: float,
    index_blocks: tuple,
    grid_spacing: float | None,
) -> list[tuple]:
    """Helper function for finding picks similar to rectangular picks.
    See ``pick_similar`` for more details.

    Rectangular picks carry an orientation and a length on top of a
    center, so unlike circles and squares they cannot be described by
    their center alone. All candidates take the median length of the
    input picks and are rotated onto the principal axis of the
    localizations they contain, and the similarity criteria use the RMSD
    along and across that axis rather than the isotropic RMSD.

    The input picks are measured in canonical form (median length,
    converged onto the localizations) rather than exactly as drawn -
    otherwise a carelessly drawn pick would shift the acceptance windows
    away from every candidate. Their drawn geometry is still what is
    returned.
    """
    locs_temp, block_size, _, _, block_starts, block_ends, K, L = index_blocks
    locs_xy = np.stack((locs_temp["x"].to_numpy(), locs_temp["y"].to_numpy()))
    length = _median_pick_length(picks)
    r_circ = 0.5 * np.sqrt(length**2 + width**2)

    # extract n_locs and the anisotropic RMSDs from the current picks
    n_locs = []
    rmsd_along = []
    rmsd_across = []
    xc_similar = []
    yc_similar = []
    theta_similar = []
    length_similar = []
    for (x_start, y_start), (x_end, y_end) in picks:
        xc = 0.5 * (x_start + x_end)
        yc = 0.5 * (y_start + y_end)
        theta = lib.wrap_angle_pi(np.arctan2(y_end - y_start, x_end - x_start))
        drawn_length = np.hypot(x_end - x_start, y_end - y_start)
        xc_similar.append(xc)
        yc_similar.append(yc)
        theta_similar.append(theta)
        length_similar.append(drawn_length)
        # initialize from the pick as drawn, using all localizations
        # because a long pick may reach outside its block neighborhood
        drawn_locs_xy = lib.locs_in_oriented_rectangle(
            xc, yc, theta, drawn_length, width, locs_xy
        )
        if drawn_locs_xy.shape[1] < 3:
            warnings.warn(
                f"Pick at ({xc:.2f}, {yc:.2f}) contains fewer than 3 "
                "localizations and is ignored when calculating the "
                "similarity criteria."
            )
            continue
        x_init = np.mean(drawn_locs_xy[0])
        y_init = np.mean(drawn_locs_xy[1])
        theta_init, _, _ = lib.principal_axis(
            np.mean((drawn_locs_xy[0] - x_init) ** 2),
            np.mean((drawn_locs_xy[0] - x_init) * (drawn_locs_xy[1] - y_init)),
            np.mean((drawn_locs_xy[1] - y_init) ** 2),
        )
        if np.abs(lib.wrap_angle_pi(theta_init - theta)) > np.deg2rad(20):
            warnings.warn(
                f"Pick at ({xc:.2f}, {yc:.2f}) is drawn more than 20 deg "
                "away from the principal axis of the localizations it "
                "contains. Check that it covers the intended structure."
            )
        # measure in canonical form, exactly like the candidates
        n, _, _, _, along, across = _refine_rectangle(
            x_init,
            y_init,
            theta_init,
            length,
            width,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            block_size,
        )
        if n < 3:
            warnings.warn(
                f"Pick at ({xc:.2f}, {yc:.2f}) contains fewer than 3 "
                "localizations and is ignored when calculating the "
                "similarity criteria."
            )
            continue
        n_locs.append(n)
        rmsd_along.append(along)
        rmsd_across.append(across)
    if not n_locs:
        raise ValueError(
            "None of the picks contains enough localizations to define "
            "the similarity criteria."
        )

    min_n_locs, max_n_locs = _similarity_window(n_locs, std_range, minimum=3)
    min_along, max_along = _similarity_window(rmsd_along, std_range)
    min_across, max_across = _similarity_window(rmsd_across, std_range)

    # the current picks are kept and block any overlapping candidates
    n_picks = len(picks)
    xc_similar = np.array(xc_similar, dtype=np.float64)
    yc_similar = np.array(yc_similar, dtype=np.float64)
    theta_similar = np.array(theta_similar, dtype=np.float64)
    length_similar = np.array(length_similar, dtype=np.float64)
    r_similar = 0.5 * np.sqrt(length_similar**2 + width**2)

    if grid_spacing is None:
        grid_spacing = length / 4
    grid_x, grid_y = _pick_similar_grid(info, grid_spacing)
    (
        xc_similar,
        yc_similar,
        theta_similar,
        length_similar,
        _,
    ) = _pick_similar_rectangle_kernel(
        grid_x,
        grid_y,
        locs_xy,
        block_starts,
        block_ends,
        K,
        L,
        block_size,
        length,
        width,
        r_circ,
        min_n_locs,
        max_n_locs,
        min_along,
        max_along,
        min_across,
        max_across,
        xc_similar,
        yc_similar,
        theta_similar,
        length_similar,
        r_similar,
    )

    # the input picks are returned exactly as drawn
    new_picks = list(picks)
    for i in range(n_picks, len(xc_similar)):
        half_x = 0.5 * length_similar[i] * np.cos(theta_similar[i])
        half_y = 0.5 * length_similar[i] * np.sin(theta_similar[i])
        new_picks.append(
            (
                (xc_similar[i] - half_x, yc_similar[i] - half_y),
                (xc_similar[i] + half_x, yc_similar[i] + half_y),
            )
        )
    return new_picks


@numba.jit(nopython=True, nogil=True, cache=True)
def _refine_rectangle(
    xc: float,
    yc: float,
    theta: float,
    length: float,
    width: float,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    block_size: float,
    max_iter: int = 100,
) -> tuple[int, float, float, float, float, float]:
    """Fit a rectangle of the given size onto the localizations around
    ``(xc, yc)``.

    Alternates between moving the rectangle onto the center of mass of
    the localizations it contains and rotating it onto their principal
    axis, until both converge. Each iteration re-reads the 3x3 block
    neighborhood at the current center, so localizations are not lost
    when the rectangle moves across a block boundary.

    Parameters
    ----------
    xc, yc : float
        Initial center of the rectangle.
    theta : float
        Initial angle of the center axis (radians).
    length, width : float
        Size of the rectangle in nm.
    locs_xy : lib.FloatArray2D
        Localization coordinates, shape ``(2, N)``, sorted by index
        blocks.
    block_starts, block_ends : lib.IntArray2D
        Block start and end indices.
    K, L : int
        Number of blocks in y and x direction.
    block_size : float
        Size of the index blocks in nm.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    n : int
        Number of localizations in the converged rectangle, 0 if the
        rectangle ran out of localizations.
    xc, yc : float
        Center of the converged rectangle.
    theta : float
        Angle of the converged rectangle, wrapped to ``[-pi/2, pi/2)``.
    rmsd_along, rmsd_across : float
        RMSD of the enclosed localizations along and across the center
        axis.
    """
    for _ in range(max_iter):
        n, mx, my, sxx, sxy, syy = _rectangle_moments_at(
            xc,
            yc,
            theta,
            length,
            width,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            int(xc / block_size),
            int(yc / block_size),
        )
        if n < 3:
            return 0, xc, yc, theta, 0.0, 0.0
        theta_new, along, across = lib.principal_axis(sxx, sxy, syy)
        # the principal axis is undefined for an isotropic point cloud
        if along**2 - across**2 <= 1e-12 * (along**2 + across**2):
            theta_new = theta
        d_theta = lib.wrap_angle_pi(theta_new - theta)
        dx = mx - xc
        dy = my - yc
        xc = mx
        yc = my
        theta = theta_new
        if (
            np.abs(dx) <= PICK_CONVERGENCE
            and np.abs(dy) <= PICK_CONVERGENCE
            and np.abs(d_theta) <= 1e-4
        ):
            break
    # measure at the converged pose
    n, _, _, sxx, sxy, syy = _rectangle_moments_at(
        xc,
        yc,
        theta,
        length,
        width,
        locs_xy,
        block_starts,
        block_ends,
        K,
        L,
        int(xc / block_size),
        int(yc / block_size),
    )
    if n < 3:
        return 0, xc, yc, theta, 0.0, 0.0
    _, along, across = lib.principal_axis(sxx, sxy, syy)
    return n, xc, yc, theta, along, across


@numba.jit(nopython=True, nogil=True, cache=True)
def _pick_similar_rectangle_kernel(
    grid_x: lib.FloatArray1D,
    grid_y: lib.FloatArray1D,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
    block_size: float,
    length: float,
    width: float,
    r_circ: float,
    min_n_locs: float,
    max_n_locs: float,
    min_along: float,
    max_along: float,
    min_across: float,
    max_across: float,
    xc_similar: lib.FloatArray1D,
    yc_similar: lib.FloatArray1D,
    theta_similar: lib.FloatArray1D,
    length_similar: lib.FloatArray1D,
    r_similar: lib.FloatArray1D,
) -> tuple[
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
]:
    """Scan the candidate lattice for rectangles similar to the given
    ones.

    Each candidate is initialized from a circle of radius
    ``length / 2``, which gives an orientation estimate that does not
    depend on a guessed angle, and is then fitted with
    ``_refine_rectangle``. This is why the lattice only has to bring a
    candidate close to a structure, not onto it, and can be much
    sparser than the pick width.

    Implemented in numba for speed, called from
    ``_pick_similar_rectangular``. See ``pick_similar`` for a more
    user-friendly interface.

    Parameters
    ----------
    grid_x, grid_y : lib.FloatArray1D
        Candidate positions.
    locs_xy : lib.FloatArray2D
        Localization coordinates, shape ``(2, N)``, sorted by index
        blocks.
    block_starts, block_ends : lib.IntArray2D
        Block start and end indices.
    K, L : int
        Number of blocks in y and x direction.
    block_size : float
        Size of the index blocks in nm.
    length, width : float
        Size of the candidate rectangles in nm.
    r_circ : float
        Circumscribed circle radius of the candidate rectangles.
    min_n_locs, max_n_locs : float
        Minimum and maximum number of localizations in the pick.
    min_along, max_along : float
        Minimum and maximum RMSD along the center axis.
    min_across, max_across : float
        Minimum and maximum RMSD across the center axis.
    xc_similar, yc_similar : lib.FloatArray1D
        Centers of the accepted picks, seeded with the input picks.
    theta_similar : lib.FloatArray1D
        Angles of the accepted picks, seeded with the input picks.
    length_similar : lib.FloatArray1D
        Lengths of the accepted picks, seeded with the input picks.
    r_similar : lib.FloatArray1D
        Circumscribed radii of the accepted picks, seeded with the input
        picks.

    Returns
    -------
    xc_similar, yc_similar : lib.FloatArray1D
        Centers of the accepted picks.
    theta_similar : lib.FloatArray1D
        Angles of the accepted picks.
    length_similar : lib.FloatArray1D
        Lengths of the accepted picks.
    r_similar : lib.FloatArray1D
        Circumscribed radii of the accepted picks.
    """
    bootstrap_r = 0.5 * length
    for i in range(len(grid_x)):
        # isotropic initialization - no orientation is assumed
        n, mx, my, sxx, sxy, syy = _circle_moments_at(
            grid_x[i],
            grid_y[i],
            bootstrap_r,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            int(grid_x[i] / block_size),
            int(grid_y[i] / block_size),
        )
        if n < 3 or n < min_n_locs:
            continue
        theta, _, _ = lib.principal_axis(sxx, sxy, syy)
        n, xc, yc, theta, along, across = _refine_rectangle(
            mx,
            my,
            theta,
            length,
            width,
            locs_xy,
            block_starts,
            block_ends,
            K,
            L,
            block_size,
        )
        # cheap tests first, the overlap test scans all accepted picks
        if n < 3:
            continue
        if not (min_n_locs <= n <= max_n_locs):
            continue
        if not (min_along <= along <= max_along):
            continue
        if not (min_across <= across <= max_across):
            continue
        overlaps = False
        for j in range(len(xc_similar)):
            if lib.rectangles_overlap(
                xc,
                yc,
                theta,
                length,
                width,
                r_circ,
                xc_similar[j],
                yc_similar[j],
                theta_similar[j],
                length_similar[j],
                width,
                r_similar[j],
            ):
                overlaps = True
                break
        if overlaps:
            continue
        xc_similar = np.append(xc_similar, xc)
        yc_similar = np.append(yc_similar, yc)
        theta_similar = np.append(theta_similar, theta)
        length_similar = np.append(length_similar, length)
        r_similar = np.append(r_similar, r_circ)
    return xc_similar, yc_similar, theta_similar, length_similar, r_similar


def remove_locs_in_picks(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None = None,
    index_blocks: tuple = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Remove localizations in picks.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Localization metadata.
    picks : list of tuples
        List of picks, each pick is a list of coordinates of the pick
        corners. See ``io.load_picks``.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"}
        Shape of picks.
    pick_size : float or None
        Size of picks in nm. For circles - diameters. For
        rectangles - width. For squares - side length. For polygons,
        boxes and brush picks - ignored. Ignored if picks are loaded
        from a YAML file.
    index_blocks : tuple, optional
        Used only for circular picks. Precomputed index blocks for
        localizations, see  ``get_index_blocks``. If None, they will be
        calculated internally. Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with localizations in picks removed.
    info : list of dicts
        Metadata of the remaining localizations: ``info`` with an entry
        recording the removal appended, see ``lib.append_metadata``.
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}."
    if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE:
        assert isinstance(
            pick_size, (int, float)
        ), "pick_size must be a number."
    if pick_shape != "Circle":
        index_blocks = None  # ignore index blocks, only used for circle picks
    all_picked_locs = picked_locs(
        locs,
        info,
        picks=picks,
        pick_shape=pick_shape,
        pick_size=pick_size,
        add_group=False,
        index_blocks=index_blocks,
    )
    # store indices of picked locs
    idx = np.concatenate([_.index for _ in all_picked_locs])
    locs = locs.drop(index=idx)
    entry = _picks_entry(
        "Remove picked localizations", picks, pick_shape, pick_size
    )
    return locs, lib.append_metadata(info, entry)


def _picks_entry(
    step: str,
    picks: list[tuple],
    pick_shape: str,
    pick_size: float | None,
) -> dict:
    """Metadata entry recording a step on picked localizations: the
    number, shape and size of the picks (the diameter for circles)."""
    entry = {
        **lib.step_info(step),
        "Number of picks": len(picks),
        "Pick shape": pick_shape,
    }
    if pick_shape == "Circle":
        entry["Pick diameter (nm)"] = float(pick_size)
    elif pick_size is not None:
        entry["Pick size (nm)"] = float(pick_size)
    return entry


@numba.jit(nopython=True, nogil=True)
def _n_block_locs_at(
    x_range: int,
    y_range: int,
    K: int,
    L: int,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
) -> int:
    """Return the number of localizations in the blocks around the
    given coordinates."""
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for ll in range(x_range - 1, x_range + 2):
                if 0 < ll < L:
                    if step == 0:
                        n_block_locs = np.uint32(
                            block_ends[k][ll] - block_starts[k][ll]
                        )
                        step = 1
                    else:
                        n_block_locs += np.uint32(
                            block_ends[k][ll] - block_starts[k][ll]
                        )
    return n_block_locs


@numba.jit(nopython=True, nogil=True, cache=True)
def _get_block_locs_at(
    x_index: int,
    y_index: int,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
) -> lib.IntArray1D:
    """Return the indices of localizations in the blocks around the
    given coordinates."""
    step = 0
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for ll in range(x_index - 1, x_index + 2):
                if 0 <= ll < L:
                    if block_ends[k, ll] - block_starts[k, ll] > 0:
                        # numba does not work if you attach concatenate
                        # to an empty list so the first step is
                        # different
                        if step == 0:
                            indices = np.arange(
                                float(block_starts[k, ll]),
                                float(block_ends[k, ll]),
                                dtype=np.uint32,
                            )
                            step = 1
                        else:
                            indices = np.concatenate(
                                (
                                    indices,
                                    np.arange(
                                        float(block_starts[k, ll]),
                                        float(block_ends[k, ll]),
                                        dtype=np.uint32,
                                    ),
                                )
                            )
    return indices


@numba.jit(nopython=True, nogil=True, cache=True)
def get_block_locs_at(
    x_index: int,
    y_index: int,
    locs_xy: lib.FloatArray2D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    K: int,
    L: int,
) -> lib.FloatArray2D:
    """Return the localizations in the blocks around the given coordinates.

    Parameters
    ----------
    x_index, y_index : int
        Block indices of the block whose 3x3 neighborhood is collected.
    locs_xy : lib.FloatArray2D
        ``(2, n_locs)`` block-sorted x and y coordinates.
    block_starts, block_ends : lib.IntArray2D
        ``(K, L)`` first and one-past-last index of each block in
        ``locs_xy``.
    K, L : int
        Number of blocks along y and x.

    Returns
    -------
    locs_xy : lib.FloatArray2D
        ``(2, n_neighbors)`` coordinates of the localizations in the
        neighborhood.
    """
    indices = _get_block_locs_at(
        x_index,
        y_index,
        block_starts,
        block_ends,
        K,
        L,
    )
    return locs_xy[:, indices]


@numba.jit(nopython=True, nogil=True)
def _distance_histogram(  # noqa: C901
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    bin_size: float,
    r_max: float,
    x_index: lib.IntArray1D,
    y_index: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    start: int,
    chunk: int,
) -> lib.IntArray1D:
    """Calculate the distance histogram for a chunk of localizations."""
    dh_len = np.uint32(r_max / bin_size)
    dh = np.zeros(dh_len, dtype=np.uint32)
    r_max_2 = r_max**2
    K, L = block_starts.shape
    end = min(start + chunk, len(x))
    for i in range(start, end):
        xi = x[i]
        yi = y[i]
        ki = y_index[i]
        li = x_index[i]
        for k in range(ki, ki + 2):
            if k < K:
                for ll in range(li, li + 2):
                    if ll < L:
                        for j in range(block_starts[k, ll], block_ends[k, ll]):
                            if j > i:
                                dx2 = (xi - x[j]) ** 2
                                if dx2 < r_max_2:
                                    dy2 = (yi - y[j]) ** 2
                                    if dy2 < r_max_2:
                                        d = np.sqrt(dx2 + dy2)
                                        if d < r_max:
                                            bin = np.uint32(d / bin_size)
                                            if bin < dh_len:
                                                dh[bin] += 1
    return dh


def distance_histogram(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    bin_size: float,
    r_max: float,
) -> lib.IntArray1D:
    """Calculate the distance histogram for the given localizations,
    i.e., the pairwise distances between localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list[dict]
        Metadata of the localizations.
    bin_size : float
        Size of the bins for the histogram (nm).
    r_max : float
        Maximum distance probed in the histogram (nm).

    Returns
    -------
    dh : lib.IntArray1D
        Distance histogram.
    """
    locs, size, x_index, y_index, b_starts, b_ends, K, L = get_index_blocks(
        locs, info, r_max
    )
    N = len(locs)
    n_threads = lib.n_workers()
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (
            locs["x"].to_numpy(),
            locs["y"].to_numpy(),
            bin_size,
            r_max,
            x_index,
            y_index,
            b_starts,
            b_ends,
            start,
            chunk,
        )
        for start in starts
    ]
    with _ThreadPoolExecutor() as executor:
        futures = [executor.submit(_distance_histogram, *_) for _ in args]
    results = [future.result() for future in futures]
    dh = np.sum(results, axis=0)
    return dh


def nena(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[dict, float]:
    """Calculate NeNA - experimental estimate of localization
    precision. Please refer to the original paper for details:
    Endesfelder, et al. Histochemistry and Cell Biology, 2014.
    DOI: 10.1007/s00418-014-1192-3.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts or None, optional
        Metadata of ``locs``. Not used; accepted so that the call
        matches the other functions on localizations. Default is None.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating NeNA" over the
        localizations. Default is None.

    Returns
    -------
    result : dict
        Data on the results, including the distances probed (nm), best
        fit and fitted parameters.
    s : float
        Estimated localization precision in nm.
    """
    with lib.tracked_progress(
        progress, "Calculating NeNA", unit="loc"
    ) as progress:
        bin_centers, dnfl = _next_frame_neighbor_distance_histogram(
            locs, progress
        )

    def func(d, delta_a, s, ac, dc, sc):
        a = ac + delta_a  # make sure a >= ac
        p_single = a * (d / (2 * s**2)) * np.exp(-(d**2) / (4 * s**2))
        p_short = (
            ac
            / (sc * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((d - dc) / sc) ** 2)
        )
        return p_single + p_short

    area = np.trapezoid(dnfl, bin_centers)
    # nanmedian: a single non-converged fit carries NaN in lpx/lpy, and a NaN
    # in p0 makes curve_fit reject the initial guess as out of bounds.
    median_lp = np.mean([np.nanmedian(locs["lpx"]), np.nanmedian(locs["lpy"])])
    peak = bin_centers[np.argmax(dnfl)]
    starts = [
        [0.8 * area, median_lp, 0.1 * area, 2 * median_lp, median_lp],
        [
            0.8 * area,
            peak / np.sqrt(2),
            0.1 * area,
            0.7 * bin_centers[-1],
            0.3 * bin_centers[-1],
        ],
        [0.8 * area, median_lp, 0.1 * area, 0.5 * bin_centers[-1], median_lp],
    ]
    bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
    popt = None
    best = np.inf
    errors = []
    for p0 in starts:
        if not np.all(np.isfinite(p0)):
            continue
        try:
            candidate, _ = curve_fit(
                func, bin_centers, dnfl, p0=p0, bounds=bounds
            )
        except (RuntimeError, ValueError) as error:
            errors.append(error)
            continue
        residual = np.sum((func(bin_centers, *candidate) - dnfl) ** 2)
        if residual < best:
            best, popt = residual, candidate
    if popt is None:
        raise RuntimeError(
            "NeNA could not be fitted to the next-frame neighbor distance "
            f"histogram. Errors: {errors}"
        )
    s = popt[1]  # NeNA
    result = {
        "d": bin_centers,  # distances probed
        "data": dnfl,
        "best_fit": func(bin_centers, *popt),
        "best_values": {
            "delta_a": popt[0],
            "s": popt[1],
            "ac": popt[2],
            "dc": popt[3],
            "sc": popt[4],
        },
    }
    return result, s


def plot_nena(
    nena_result: dict,
    fig: plt.Figure = None,
    fill: bool = True,
    outline: bool = False,
) -> plt.Figure:
    """Plot the results of NeNA.

    Parameters
    ----------
    nena_result : dict
        Data on the results from function ``nena``, including the
        distances probed (nm), best fit and fitted parameters.
    fig : plt.Figure
        Figure to plot on. If None, a new figure and axes are
        created.
    fill, outline : bool, optional
        Whether the histogram bars are filled and/or outlined, see
        ``lib.histogram_style``. Default is filled without an outline.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot.
    """
    if fig is None:
        fig = plt.Figure(constrained_layout=True)
    else:
        fig.clear()
    d = deepcopy(nena_result["d"])
    ax = fig.add_subplot(111)
    ax.set_title(
        "Next frame neighbor distance histogram, "
        f"\u03c3 = {nena_result['best_values']['s']:.2f} nm"
    )
    # the data is a histogram over bin centers: draw it as filled bars
    d = np.asarray(d, dtype=float)
    half_bin = (d[1] - d[0]) / 2 if len(d) > 1 else 0.5
    edges = np.append(d - half_bin, d[-1] + half_bin)
    # colors of the active style, resolved now (lines look "CN" colors
    # up only when drawn)
    data_color, fit_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
        :2
    ]
    ax.stairs(
        nena_result["data"],
        edges,
        label="Data",
        **lib.histogram_style(data_color, fill, outline),
    )
    ax.plot(d, nena_result["best_fit"], color=fit_color, label="Fit")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distance (nm)")
    ax.set_ylabel("Counts")
    ax.legend(loc="best")
    return fig


def _next_frame_neighbor_distance_histogram(
    locs: pd.DataFrame,
    progress: lib.Progress,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the next frame neighbor distance histogram (NFNDH).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    progress : lib.Progress
        Normalized progress tracker; reports "Calculating NeNA" over the
        localizations.

    Returns
    -------
    bin_centers : lib.FloatArray1D
        Centers of the bins for the histogram (nm, 0.2 nm bins up to
        200 nm).
    dnfl : lib.FloatArray1D
        Distance histogram of next frame neighbors.
    """
    # sort a copy: callers (e.g. the Render GUI) hold position-aligned
    # arrays such as group colors that an in-place sort would scramble
    locs = locs.sort_values(kind="quicksort", by="frame")
    frame = locs["frame"].to_numpy()
    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()
    if "group" in locs.columns:
        group = locs["group"].to_numpy()
    else:
        group = np.zeros(len(locs), dtype=np.int32)
    bin_size = 0.2  # nm
    d_max = 200.0  # nm
    return _nfndh(frame, x, y, group, d_max, bin_size, progress)


def _nfndh(
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    group: lib.IntArray1D,
    d_max: float,
    bin_size: float,
    progress: lib.Progress,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the next frame neighbor distance histogram (NFNDH) of
    localizations sorted by frame.

    All localizations are processed, in up to 100 chunks; ``progress``
    reports "Calculating NeNA" over the localizations after each chunk.
    """
    N = len(frame)
    bins = np.arange(0, d_max, bin_size)
    dnfl = np.zeros(len(bins))
    lib.start_phase(progress, "Calculating NeNA", N, unit="loc")
    progress.set_value(0)
    bounds = np.linspace(0, N, min(N, 100) + 1).astype(np.int64)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        for i in range(start, stop):
            _fill_dnfl(N, frame, x, y, group, i, d_max, dnfl, bin_size)
        progress.set_value(int(stop))
    progress.set_value(N)
    bin_centers = bins + bin_size / 2
    return bin_centers, dnfl


@numba.jit(nopython=True)
def _fill_dnfl(
    N: int,
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    group: lib.IntArray1D,
    i: int,
    d_max: float,
    dnfl: lib.FloatArray1D,
    bin_size: float,
) -> None:
    """Fill the next frame neighbor distance histogram (NFNDH) for a
    single localization."""
    frame_i = frame[i]
    x_i = x[i]
    y_i = y[i]
    group_i = group[i]
    # locs are sorted by frame: [min_index, max_index) is the next frame
    next_frame = frame_i + 1
    min_index = i + 1
    while min_index < N and frame[min_index] < next_frame:
        min_index += 1
    max_index = min_index
    while max_index < N and frame[max_index] <= next_frame:
        max_index += 1
    d_max_2 = d_max**2
    for j in range(min_index, max_index):
        if group[j] == group_i:
            dx2 = (x_i - x[j]) ** 2
            if dx2 <= d_max_2:
                dy2 = (y_i - y[j]) ** 2
                if dy2 <= d_max_2:
                    d = np.sqrt(dx2 + dy2)
                    # the last bin ends below d_max; numba does not
                    # check the index, so a distance of d_max wrote
                    # outside the histogram
                    bin = int(d / bin_size)
                    if d < d_max and bin < dnfl.shape[0]:
                        dnfl[bin] += 1


def plot_frc(
    frc_result: dict,
    fig: plt.Figure = None,
) -> plt.Figure:
    """Plot the results of the Fourier Ring Correlation (FRC) resolution
    estimation.

    Parameters
    ----------
    frc_result : dict
        Dictionary result of ``frc``.
    fig : plt.Figure
        Figure to plot on. If None, a new figure and axes are
        created.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot.
    """
    if fig is None:
        fig = plt.Figure(constrained_layout=True)
    else:
        fig.clear()
    q = frc_result["frequencies"]
    frc_curve = frc_result["frc_curve"]
    frc_curve_smooth = frc_result["frc_curve_smooth"]
    res = frc_result["resolution"]
    ax = fig.add_subplot(111)
    ax.plot(q, frc_curve, color="gray", alpha=0.5, label="FRC curve")
    ax.plot(q, frc_curve_smooth, label="Smoothed")
    ax.axhline(
        1 / 7,
        # black by default; follows the theme of a plot style
        color=plt.rcParams["text.color"],
        linewidth=1.0,
        linestyle="--",
        label="1/7 threshold",
    )
    ax.set_xlabel("Spatial frequency (nm\u207b\u00b9)")
    ax.set_ylabel("FRC")
    if res is None:
        ax.set_title("FIRE resolution: n/a (no 1/7 crossing)")
    else:
        ax.set_title(f"FIRE resolution: {res:.2f} nm")
    ax.legend()
    return fig


def frc(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    random_seed: int = 42,
    lp: float | None = None,
) -> dict:
    """Calculate the Fourier Ring Correlation (FRC) resolution.

    See Nieuwenhuizen et al., Nat. Methods 10, 557–562 (2013).

    Parameters
    ----------
    locs : pd.DataFrame
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    viewport : tuple of floats
        Viewport ((y_min, x_min), (y_max, x_max)) in nm for rendering
        the images. Note that the origin of the image is in the top-left
        corner.
    random_seed : int, optional
        Random seed for splitting the data into halves. Default is 42.
    lp : float or None, optional
        Localization precision (nm) that sets the bin size of the
        rendered images (``lp / 2``). If None (default), NeNA of
        ``locs`` is used.

    Returns
    -------
    frc_result : dict
        Dictionary with keys "frc_curve", "frc_curve_smooth",
        "frequencies" (for spatial frequencies probed (nm^-1)),
        "resolution" (estimated resolution in nm) and "images"
        (2 grayscale images rendered and masked).
    """
    if lp is None:
        lp = nena(locs, info)[1]
    # correct for the viewport to be square
    viewport_width = viewport[1][1] - viewport[0][1]
    viewport_height = viewport[1][0] - viewport[0][0]
    if viewport_width < viewport_height:
        y_center = 0.5 * (viewport[0][0] + viewport[1][0])
        y_min = y_center - viewport_width / 2
        y_max = y_center + viewport_width / 2
        viewport = ((y_min, viewport[0][1]), (y_max, viewport[1][1]))
    elif viewport_height < viewport_width:
        x_center = 0.5 * (viewport[0][1] + viewport[1][1])
        x_min = x_center - viewport_height / 2
        x_max = x_center + viewport_height / 2
        viewport = ((viewport[0][0], x_min), (viewport[1][0], x_max))

    # select the locs within the viewport
    (y_min, x_min), (y_max, x_max) = viewport
    in_view = (
        (locs["x"] > x_min)
        & (locs["y"] > y_min)
        & (locs["x"] < x_max)
        & (locs["y"] < y_max)
    )
    locs = locs.loc[in_view]

    np.random.seed(random_seed)
    # split locs randomly into two halves
    r_idx = np.random.permutation(len(locs))
    locs1 = locs.iloc[r_idx[: len(r_idx) // 2]]
    locs2 = locs.iloc[r_idx[len(r_idx) // 2 :]]
    # run FRC
    frc_curve, frc_curve_smooth, frequencies, resolution, images = _frc(
        locs1, locs2, info, lp, viewport
    )
    # smooth the FRC curve and summarize findings
    frc_result = {
        "frc_curve": frc_curve,
        "frc_curve_smooth": frc_curve_smooth,
        "frequencies": frequencies,
        "resolution": resolution,
        "images": images,
    }
    return frc_result


def frc_rois(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    n_rois: int = 30,
    roi_size: float = 5000.0,
    min_locs: int = 1000,
    random_seed: int = 42,
    progress: lib.Progress | Literal["console"] | None = None,
) -> dict:
    """Calculate the FRC resolution in several random ROIs.

    Places up to ``n_rois`` non-overlapping square ROIs in ``viewport``
    (see ``lib.select_frc_rois``) and runs ``frc`` in each of them. The
    image bin size is set by the NeNA of all ``locs``, computed once,
    so that the ROIs are comparable. The spread of the resolutions
    across ROIs estimates the uncertainty of the FRC resolution.

    Parameters
    ----------
    locs : pd.DataFrame
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    viewport : tuple of floats
        Region ((y_min, x_min), (y_max, x_max)) in nm in which the ROIs
        are placed.
    n_rois : int, optional
        Maximum number of ROIs. Default is 30.
    roi_size : float, optional
        Side length of the square ROIs in nm. Default is 5000.
    min_locs : int, optional
        Minimum number of localizations per ROI. Default is 1000.
    random_seed : int, optional
        Random seed for the ROI placement and the splitting into
        halves. Default is 42.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating NeNA" over the
        localizations, then "Calculating FRC" over the ROIs. Default is
        None.

    Returns
    -------
    result : dict
        Dictionary with keys "rois" (viewports of the ROIs in nm),
        "n_locs" (localizations per ROI), "frc_results" (list of
        ``frc`` results, one per ROI), "resolutions" (in nm, NaN where
        the FRC curve does not cross the 1/7 threshold) and "lp" (NeNA
        in nm).
    """
    rois = lib.select_frc_rois(
        locs,
        info,
        viewport,
        n_rois=n_rois,
        roi_size=roi_size,
        min_locs=min_locs,
        random_seed=random_seed,
    )
    with lib.tracked_progress(
        progress, "Calculating NeNA", unit="loc"
    ) as progress:
        lp = nena(locs, info, progress=progress)[1]
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        n_locs = []
        frc_results = []
        lib.start_phase(progress, "Calculating FRC", len(rois), unit="ROI")
        progress.set_value(0)
        for k, ((y0, x0), (y1, x1)) in enumerate(rois):
            in_roi = (x > x0) & (y > y0) & (x < x1) & (y < y1)
            n_locs.append(int(in_roi.sum()))
            frc_result = frc(
                locs.loc[in_roi],
                info,
                viewport=((y0, x0), (y1, x1)),
                random_seed=random_seed,
                lp=lp,
            )
            # drop the rendered images: tens of ROIs would hold GBs
            del frc_result["images"]
            frc_results.append(frc_result)
            progress.set_value(k + 1)
        progress.set_value(len(rois))
    resolutions = np.array(
        [
            np.nan if _["resolution"] is None else _["resolution"]
            for _ in frc_results
        ]
    )
    return {
        "rois": rois,
        "n_locs": np.array(n_locs, dtype=int),
        "frc_results": frc_results,
        "resolutions": resolutions,
        "lp": lp,
    }


def _frc(
    locs1: pd.DataFrame,
    locs2: pd.DataFrame,
    info: list[dict],
    lp: float,
    viewport: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    float | None,
    tuple[lib.FloatArray2D, lib.FloatArray2D],
]:
    """Calculate the Fourier Ring Correlation (FRC) resolution once.

    Generates images from the two sets of localizations and calculates
    the FRC curve and resolution (at 1/7 threshold).

    See Nieuwenhuizen et al., Nat. Methods 10, 557–562 (2013).

    Note: do not use this function directly, use ``frc`` instead.

    Parameters
    ----------
    locs1, locs2 : pd.DataFrame
        Localization lists already split to render images.
    info : list of dicts
        Metadata of the localizations.
    lp : float
        Average localization precision (NeNA, nm), used for bin size of
        the rendered images (binsize = lp / 2).
    viewport : tuple of floats
        Viewport ((y_min, x_min), (y_max, x_max)) in nm for rendering
        the images. Note that the origin of the image is in the top-left
        corner.

    Returns
    -------
    frc_curve : lib.FloatArray1D
        FRC curve.
    frc_curve_smooth : lib.FloatArray1D
        Smoothed FRC curve (LOESS).
    frequencies : lib.FloatArray1D
        Spatial frequencies corresponding to the FRC curve (nm^-1).
    resolution : float or None
        Estimated resolution in nm, given the 1/7 threshold. None if
        resolution could not be determined.
    images : tuple of 2 lib.FloatArray2D
        2 grayscale images used for calculating FRC. Already masked.
    """
    # render images
    disp_px_size = lp / 2  # nm
    im1 = render.render(
        locs1,
        info,
        disp_px_size=disp_px_size,
        viewport=viewport,
        blur_method=None,
    )[1]
    im2 = render.render(
        locs2,
        info,
        disp_px_size=disp_px_size,
        viewport=viewport,
        blur_method=None,
    )[1]

    # ensure the images are odd-sized and mask them (tukey)
    if im1.shape[0] % 2 == 0:
        im1 = im1[:-1, :-1]
        im2 = im2[:-1, :-1]
    mask = masking.threshold_tukey(im1)
    im1 *= mask
    im2 *= mask

    # fourier
    f1 = np.fft.fftshift(np.fft.fft2(im1))
    f2 = np.fft.fftshift(np.fft.fft2(im2))

    # frc curve
    frc_num = np.real(imageprocess.radial_sum(f1 * np.conj(f2)))
    frc_denom = np.sqrt(
        np.abs(
            imageprocess.radial_sum(np.abs(f1) ** 2)
            * imageprocess.radial_sum(np.abs(f2) ** 2)
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        frc_curve = frc_num / frc_denom
    frc_curve[np.isnan(frc_curve)] = 0

    # smooth the frc curve
    sspan = max(int(np.ceil(int(im1.shape[0] / 2) / 20)), 5)
    frc_curve_smooth = masking.loess_smooth(frc_curve, sspan)

    # find the frequencies (q) and resolution
    frequencies = np.arange(len(frc_curve)) / im1.shape[0] / disp_px_size
    threshold = 1 / 7  # resolution at 1/7 threshold
    resolution = None
    for i in range(1, len(frc_curve_smooth)):
        if (
            frc_curve_smooth[i - 1] >= threshold
            and frc_curve_smooth[i] < threshold
        ):
            # linear interpolation
            f1 = frequencies[i - 1]
            f2 = frequencies[i]
            r1 = frc_curve_smooth[i - 1]
            r2 = frc_curve_smooth[i]
            f_res = f1 + (threshold - r1) * (f2 - f1) / (r2 - r1)
            resolution = 1 / f_res  # in nm
            break
    return frc_curve, frc_curve_smooth, frequencies, resolution, (im1, im2)


def pair_correlation(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    bin_size: float,
    r_max: float,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the pair correlation function for the given
    localizations, as the histogram of their distances divided by the
    area of each ring (1/nm^2, no edge correction or normalization by
    the density). See ``rdf`` for the normalized, edge-corrected radial
    distribution function.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    bin_size : float
        Size of the bins for the histogram (nm).
    r_max : float
        Maximum distance for the histogram (nm).

    Returns
    -------
    bins_lower : lib.FloatArray1D
        Lower bounds of the bins for the histogram (nm).
    pc : lib.FloatArray1D
        Pair correlation function (1/nm^2).
    """
    dh = distance_histogram(locs, info, bin_size=bin_size, r_max=r_max)
    # Start with r-> otherwise area will be 0
    bins_lower = np.arange(bin_size, r_max + bin_size, bin_size)

    if bins_lower.shape[0] > dh.shape[0]:
        bins_lower = bins_lower[:-1]
    area = np.pi * bin_size * (2 * bins_lower + bin_size)
    pc = dh / area
    return bins_lower, pc


@numba.jit(nopython=True, nogil=True)
def _local_density(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    radius: float,
    x_index: lib.IntArray1D,
    y_index: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    start: int,
    chunk: int,
) -> lib.IntArray1D:
    """Calculate densities in blocks around each localization."""
    N = len(x)
    r2 = radius**2
    end = min(start + chunk, N)
    density = np.zeros(N, dtype=np.uint32)
    for i in range(start, end):
        yi = y[i]
        xi = x[i]
        ki = y_index[i]
        li = x_index[i]
        di = 0
        for k in range(ki - 1, ki + 2):
            for ll in range(li - 1, li + 2):
                j_min = block_starts[k, ll]
                j_max = block_ends[k, ll]
                for j in range(j_min, j_max):
                    dx2 = (xi - x[j]) ** 2
                    if dx2 < r2:
                        dy2 = (yi - y[j]) ** 2
                        if dy2 < r2:
                            d2 = dx2 + dy2
                            if d2 < r2:
                                di += 1
        density[i] = di
    return density


def compute_local_density(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    radius: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Compute the local density of localizations in blocks around
    each localization.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    radius : float
        Radius for local density computation (nm).

    Returns
    -------
    locs : pd.DataFrame
        Localizations with added 'density' field/column.
    info : list of dicts
        Metadata of ``locs``: ``info`` with an entry recording the step
        appended, see ``lib.append_metadata``.
    """
    locs, size, x_index, y_index, block_starts, block_ends, K, L = (
        get_index_blocks(locs, info, radius)
    )
    N = len(locs)
    n_threads = lib.n_workers()
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (
            locs["x"].to_numpy(),
            locs["y"].to_numpy(),
            radius,
            x_index,
            y_index,
            block_starts,
            block_ends,
            start,
            chunk,
        )
        for start in starts
    ]
    with _ThreadPoolExecutor() as executor:
        futures = [executor.submit(_local_density, *_) for _ in args]
    density = np.sum([future.result() for future in futures], axis=0)
    locs["density"] = density
    entry = lib.step_info("Density", **{"Radius (nm)": radius})
    return locs, lib.append_metadata(info, entry)


def evaluate_picks(
    picked_locs: list[pd.DataFrame],
    info: list[dict],
    *,
    max_dark_time: int = 3,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    pd.DataFrame,
]:
    """Calculate pick statistics: number of localizations and binding
    events, rmsd, bright and dark times.

    Returned arrays may contain NaNs (``np.empty`` is used for
    initialization and not all picks may be evaluated successfully).

    Parameters
    ----------
    picked_locs : list of pd.DataFrame
        List of dataframes, each containing the localizations in a
        picked region.
    info : list of dicts
        Metadata of the localizations.
    max_dark_time : int
        Maximum dark time (in frames) between detected localizations to
        consider them as part of the same binding event. Default is 3
        frames.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Evaluating picks" over the
        picks. Default is None.

    Returns
    -------
    N : lib.FloatArray1D
        Array of number of localizations in each pick.
    n_events : lib.FloatArray1D
        Array of number of binding events in each pick.
    rmsd : lib.FloatArray1D
        Array of RMSD of localizations in each pick in nm.
    rmsd_z : lib.FloatArray1D
        Array of RMSD of localizations in z in each pick in nm.
    length : lib.FloatArray1D
        Array of estimated mean bright times in each pick in frames.
    dark : lib.FloatArray1D
        Array of estimated mean dark times in each pick in frames.
    new_locs : pd.DataFrame
        Dataframe containing the localizations in all picked regions
        with added 'length' and 'dark' fields/columns.
    """
    n_picks = len(picked_locs)
    N = np.empty(n_picks)  # number of locs per pick
    n_events = np.empty(n_picks)  # number of events per pick
    rmsd = np.empty(n_picks)  # rmsd in each pick
    length = np.empty(n_picks)  # estimated mean bright time
    dark = np.empty(n_picks)  # estimated mean dark time
    has_z = "z" in picked_locs[0].columns
    rmsd_z = np.empty(n_picks)
    new_locs = []  # linked locs in each pick
    # e.g., the statistics of one-localization picks; scoped, so that
    # the warning filters of the caller are kept
    with (
        lib.tracked_progress(
            progress, "Evaluating picks", unit="pick"
        ) as progress,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lib.start_phase(progress, "Evaluating picks", n_picks, unit="pick")
        progress.set_value(0)
        for i in range(n_picks):
            pick_locs = picked_locs[i]
            if not len(pick_locs):
                progress.set_value(i + 1)
                continue

            N[i] = len(pick_locs)
            com_x = pick_locs["x"].mean()
            com_y = pick_locs["y"].mean()
            rmsd[i] = np.sqrt(
                np.mean(
                    (pick_locs["x"] - com_x) ** 2
                    + (pick_locs["y"] - com_y) ** 2
                )
            )
            if has_z:
                rmsd_z[i] = np.sqrt(
                    np.mean((pick_locs["z"] - pick_locs["z"].mean()) ** 2)
                )
            if "len" not in pick_locs.columns:
                pick_locs = _link(
                    pick_locs, info, r_max=999999, max_dark_time=max_dark_time
                )
            pick_locs = _compute_dark_times(pick_locs)
            n_events[i] = len(pick_locs)  # linked locs are binding events
            length[i] = lib.estimate_kinetic_rate(pick_locs["len"].to_numpy())
            dark[i] = lib.estimate_kinetic_rate(pick_locs["dark"].to_numpy())
            new_locs.append(pick_locs)
            progress.set_value(i + 1)
        progress.set_value(n_picks)
    new_locs = pd.concat(new_locs, ignore_index=True)
    return N, n_events, rmsd, rmsd_z, length, dark, new_locs


def _pick_kinetics_single(
    pick_locs: pd.DataFrame,
    info: list[dict],
    max_dark_time: int,
) -> tuple[pd.DataFrame, float, float] | None:
    """Compute kinetics for a single picked region. Returns None if the
    region has no usable data or kinetic rate estimation fails."""
    if not len(pick_locs):
        return None
    if "len" not in pick_locs.columns:
        pick_locs = _link(
            pick_locs,
            info,
            r_max=999999,  # link all locs in the pick
            max_dark_time=max_dark_time,
        )
    if not len(pick_locs):
        return None
    pick_locs = _compute_dark_times(pick_locs)
    if not len(pick_locs):
        return None
    try:
        l_ = lib.estimate_kinetic_rate(pick_locs["len"].to_numpy())
        d_ = lib.estimate_kinetic_rate(pick_locs["dark"].to_numpy())
    except RuntimeError:
        return None
    return pick_locs, l_, d_


def pick_kinetics(
    picked_locs: list[pd.DataFrame],
    info: list[dict],
    *,
    max_dark_time: int = 3,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.IntArray1D,
    pd.DataFrame,
    lib.IntArray1D,
]:
    """Calculate kinetics per picked region. Assumes picked
    localizations, see ``picked_locs``.

    Parameters
    ----------
    picked_locs : list of pd.DataFrame
        List of dataframes, each containing the localizations in a picked
        region.
    info : list of dicts
        Metadata of the localizations.
    max_dark_time : int
        Maximum dark time (in frames) between detected localizations
        to consider them as part of the same binding event. Default is 3
        frames.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating kinetics" over
        the picks. Default is None.

    Returns
    -------
    length : lib.FloatArray1D
        Array of lengths of binding events in each picked region in
        units of frames.
    dark : lib.FloatArray1D
        Array of dark times between binding events in each picked region
        in units of frames.
    no_locs : lib.IntArray1D
        Array of number of localizations in each picked region (before
        linking into binding events).
    out_locs : pd.DataFrame
        Dataframe containing the localizations in all picked regions with
        added 'length', 'dark' and 'n' fields/columns. Pick regions
        where binding kinetics could not be estimated (e.g., because
        of too little data or unsuccessful fitting) are removed.
    kept_indices : lib.IntArray1D
        Indices (into ``picked_locs``) of the picks that were retained,
        i.e., those for which kinetics could be estimated. Aligns
        row-for-row with ``length``, ``dark`` and ``no_locs``.
    """
    n_picks = len(picked_locs)
    out_locs = []
    dark = []  # estimated mean dark time
    length = []  # estimated mean bright time
    no_locs = []  # number of locs
    kept_indices = []  # picks for which kinetics could be estimated
    with lib.tracked_progress(
        progress, "Calculating kinetics", unit="pick"
    ) as progress:
        lib.start_phase(progress, "Calculating kinetics", n_picks, unit="pick")
        progress.set_value(0)
        for i in range(n_picks):
            result = _pick_kinetics_single(picked_locs[i], info, max_dark_time)
            if result is not None:
                pick_locs, l_, d_ = result
                length.append(l_)
                dark.append(d_)
                no_locs.append(len(picked_locs[i]))
                out_locs.append(pick_locs)
                kept_indices.append(i)
            progress.set_value(i + 1)
        progress.set_value(n_picks)
    length = np.array(length)
    dark = np.array(dark)
    no_locs = np.array(no_locs)
    kept_indices = np.array(kept_indices, dtype=int)
    if out_locs:
        out_locs = pd.concat(out_locs, ignore_index=True)
    elif n_picks:
        # no pick survived: keep the columns of the input
        out_locs = picked_locs[0].iloc[:0].copy()
    else:
        out_locs = pd.DataFrame()
    return length, dark, no_locs, out_locs, kept_indices


def pick_properties(
    picked_locs: list[pd.DataFrame],
    info: list[dict],
    *,
    max_dark_time: int = 3,
    influx_rate: float = 0.03,
    pick_areas: lib.FloatArray1D | None = None,
    progress: lib.Progress | Literal["console"] | None = None,
) -> pd.DataFrame:
    """Calculate pick properties.

    Properties include number of localizations, mean and std of all
    localizations dtypes (x, y, photons, etc), qPAINT number of binding
    sites and the kinetics CDFs.

    Parameters
    ----------
    picked_locs : list of pd.DataFrame
        List of dataframes with localizations, one per picked region.
    info : list of dicts
        Metadata of the localizations.
    max_dark_time : int
        Maximum dark time (in frames) passed to ``pick_kinetics``.
    influx_rate : float
        Influx rate used to estimate the number of binding sites
        (``n_units = 1 / (influx_rate * dark)``).
    pick_areas : FloatArray1D or None
        Optional per-pick area in um^2 to attach to the output.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating kinetics" over
        the picks, then "Calculating group properties" over the picks
        whose kinetics could be estimated. Default is None.

    Returns
    -------
    pick_props : pd.DataFrame
        Each row gives the properties per pick.
    """
    with (
        lib.tracked_progress(
            progress, "Calculating kinetics", unit="pick"
        ) as progress,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter(
            "ignore", category=(OptimizeWarning, RuntimeWarning)
        )
        length, dark, no_locs, out_locs, kept_indices = pick_kinetics(
            picked_locs=picked_locs,
            info=info,
            max_dark_time=max_dark_time,
            progress=progress,
        )
        pick_props = groupprops(out_locs, progress=progress)
        if pick_areas is not None:
            # only the picks that survived kinetics estimation are present
            # in pick_props, so subset the areas accordingly to avoid a
            # length mismatch (e.g. picks empty in this channel are dropped)
            pick_props["pick_area_um2"] = np.asarray(pick_areas)[kept_indices]

    pick_props["n_units"] = 1 / (influx_rate * dark)
    pick_props["locs"] = no_locs
    pick_props["length_cdf"] = length
    pick_props["dark_cdf"] = dark
    pick_props["qpaint_idx_cdf"] = dark**-1

    return pick_props


def compute_dark_times(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    group: lib.IntArray1D | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Compute dark time for each binding event.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations that were linked, i.e., binding events.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    group : lib.IntArray1D, optional
        Grouping array for binding events. If None, all binding events
        are considered to be in the same group.

    Returns
    -------
    locs : pd.DataFrame
        Binding events with added 'dark' field/column, which contains
        the dark time preceding each binding event in frames (see
        ``dark_times`` for the convention). Binding events that are not
        preceded by another binding event in the same group are
        removed.
    info : list of dicts
        Metadata of ``locs``: ``info`` with an entry recording the step
        (``"Step": "Dark"``) appended, see ``lib.append_metadata``.
        ``io.load_locs`` reads the Picasso version of this entry to
        tell the dark-time convention.
    """
    locs = _compute_dark_times(locs, group)
    return locs, lib.append_metadata(info, lib.step_info("Dark"))


def _compute_dark_times(
    locs: pd.DataFrame, group: lib.IntArray1D | None = None
) -> pd.DataFrame:
    """The binding events with dark times of ``compute_dark_times``,
    without the metadata (for loops over picks)."""
    if "len" not in locs.columns:
        raise AttributeError(
            "Length not found. Please link localizations first."
        )
    dark = dark_times(locs, group)
    locs = locs.assign(dark=np.int32(dark))
    return locs[locs.dark != -1]


def dark_times(
    locs: pd.DataFrame,
    group: lib.IntArray1D | None = None,
) -> lib.IntArray1D:
    """Calculate dark times for each binding event.

    The dark time of a binding event is the number of frames without
    signal between the end of the closest preceding binding event in
    the same group and its own first frame, i.e.,
    ``first_frame - previous_last_frame - 1``. For example, two binding
    events of one frame each, in frames 100 and 5000, are separated by
    a dark time of 4899 frames. Since ``link`` merges localizations
    separated by up to ``max_dark_time`` frames without signal into one
    binding event, dark times at the same site are longer than
    ``max_dark_time``. Together with the bright time (``len``, see
    ``link``), each counts the frames spent in its state, so that bright
    plus dark time equals the time between the starts of consecutive
    binding events.

    Both are unbiased estimates of the underlying continuous times when
    a frame is detected as bright once the binding event covers half of
    it. With a lower detection threshold, bright times are overestimated
    and dark times underestimated, by up to one frame each; with a
    higher one, the other way round. Their sum is unbiased either way.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations that were linked, i.e., binding events.
    group : lib.IntArray1D, optional
        Grouping array for binding events. If None, all binding events
        are considered to be in the same group.

    Returns
    -------
    dark : lib.IntArray1D
        Array of dark times for each binding event in frames. If a
        binding event is not preceded by another binding event in the
        same group, the dark time is set to -1.

    Notes
    -----
    Before Picasso 0.11.3, the dark time was ``first_frame -
    previous_last_frame``, i.e., one frame longer than now.
    """
    frame = locs["frame"].to_numpy()
    lens = locs["len"].to_numpy()
    last_frame = frame + lens - 1
    if group is None:
        if "group" in locs.columns:
            group = locs["group"].to_numpy()
        else:
            group = np.zeros(len(locs))
    dark = _dark_times(frame, group, last_frame)
    return dark


@numba.jit(nopython=True)
def _dark_times(
    frame: lib.IntArray1D,
    group: lib.IntArray1D,
    last_frame: lib.IntArray1D,
) -> lib.IntArray1D:
    """Calculate dark times for each binding event, see
    ``dark_times``."""
    N = len(frame)
    dark = -np.ones(N, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            # j must end before i starts; checked before subtracting
            # since the frames are unsigned
            if (group[i] == group[j]) and (frame[i] > last_frame[j]):
                dark_ij = frame[i] - last_frame[j] - 1
                if (dark[i] == -1) or (dark_ij < dark[i]):
                    dark[i] = dark_ij
    return dark


def link(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    r_max: float = 10.0,
    max_dark_time: int = 3,
    combine_mode: Literal["average", "refit"] = "average",
    remove_ambiguous_lengths: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Link localizations, i.e., group them into binding events based
    on their spatiotemporal proximity.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    r_max : float, optional
        Maximum distance (nm) for linking localizations. Default is 10.
    max_dark_time : int, optional
        Maximum number of frames without a localization between two
        localizations that are still linked into one binding event.
        Default is 3.
    combine_mode : {'average', 'refit'}, optional
        Mode for combining linked localizations. 'average' calculates
        the average position and properties of the linked localizations,
        while 'refit' would refit the linked localizations to a model.
        'refit' is not implemented yet. Default is 'average'.
    remove_ambiguous_lengths : bool, optional
        If True, removes linked localizations with ambiguous lengths,
        i.e., localizations that are linked to multiple binding events
        with different lengths. Default is True.

    Returns
    -------
    linked_locs : pd.DataFrame
        Linked localizations, i.e., binding events with their
        properties. The column ``len`` holds the bright time of each
        binding event in frames, counted from its first to its last
        frame (``last_frame - first_frame + 1``), including frames
        bridged by ``max_dark_time``. See ``dark_times`` for the
        matching dark time convention.
    info : list of dicts
        Metadata of the linked localizations: ``info`` with an entry
        recording the linking appended, see ``lib.append_metadata``.
    """
    linked_locs = _link(
        locs,
        info,
        r_max,
        max_dark_time,
        combine_mode=combine_mode,
        remove_ambiguous_lengths=remove_ambiguous_lengths,
    )
    entry = lib.step_info(
        "Link",
        **{
            "Max. distance (nm)": r_max,
            "Max. transient dark frames": max_dark_time,
            "Removed ambiguous lengths": remove_ambiguous_lengths,
        },
    )
    return linked_locs, lib.append_metadata(info, entry)


def _link(
    locs: pd.DataFrame,
    info: list[dict],
    r_max: float,
    max_dark_time: int,
    combine_mode: Literal["average", "refit"] = "average",
    remove_ambiguous_lengths: bool = True,
) -> pd.DataFrame:
    """The linked localizations of ``link``, without the metadata (for
    loops over picks, which need no metadata of their own)."""
    if len(locs) == 0:  # special case of an empty localization list
        linked_locs = locs.copy()
        if "frame" in locs.columns:
            linked_locs["len"] = np.array([], dtype=np.int32)
            linked_locs["n"] = np.array([], dtype=np.int32)
        if "photons" in locs.columns:
            linked_locs["photon_rate"] = np.array([], dtype=np.float32)
    else:
        locs = locs.sort_values(kind="quicksort", by="frame")
        if "group" in locs.columns:
            group = locs["group"].to_numpy()
        else:
            group = np.zeros(len(locs), dtype=np.int32)
        frame = locs["frame"].to_numpy()
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        link_group = _get_link_groups(frame, x, y, r_max, max_dark_time, group)
        if combine_mode == "average":
            linked_locs = _link_loc_groups(
                locs,
                info,
                link_group,
                remove_ambiguous_lengths=remove_ambiguous_lengths,
            )
        elif combine_mode == "refit":
            raise NotImplementedError(
                "Refit mode is not implemented yet. Please use 'average' mode."
            )
    return linked_locs


def select_binding_event_cores(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    r_max: float = 10.0,
    max_dark_time: int = 0,
    min_n_locs: int = 3,
) -> tuple[pd.DataFrame, list[dict]]:
    """Select the localizations in the temporal centers of binding
    events.

    Localizations are grouped into binding events by their
    spatiotemporal proximity (as in ``link``), and only the
    localizations that are not at the borders of a binding event are
    kept, i.e., those in the first and the last frame of each event are
    discarded. Such localizations only capture a part of the emission
    event (the imager binds or unbinds during the camera exposure) and
    thus have a biased photon count, see Steen et al., Nat Methods 21,
    1755-1762 (2024), Extended Data Fig. 1f.

    Each retained binding event is assigned a unique value in the
    'group' column. If ``locs`` was already grouped, the previous
    grouping is preserved in the 'group_input' column (and the binding
    events do not span several input groups).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts or None, optional
        Metadata of ``locs``, carried over into the returned metadata.
        Default is None.
    r_max : float, optional
        Maximum distance (nm) between localizations to be considered as
        originating from the same binding event. Default is 10.
    max_dark_time : int, optional
        Maximum number of frames between localizations to be considered
        as originating from the same binding event. Default is 0.
    min_n_locs : int, optional
        Minimum number of localizations in a binding event for it to be
        considered. Events with fewer localizations are discarded
        entirely. Since the first and the last localization of each
        event are discarded, values below 3 leave no localizations.
        Default is 3.

    Returns
    -------
    out_locs : pd.DataFrame
        Localizations in the cores of binding events, with the 'group'
        column assigned uniquely to each binding event. If ``locs`` was
        already grouped, the previous grouping is kept in
        'group_input'.
    info : list of dicts
        Metadata of ``out_locs``: ``info`` with an entry recording the
        selection appended, see ``lib.append_metadata``.
    """
    entry = lib.step_info(
        "Binding event cores",
        **{
            "Max. distance (nm)": r_max,
            "Max. transient dark frames": max_dark_time,
            "Min. localizations per event": min_n_locs,
        },
    )
    info = lib.append_metadata(info, entry)
    out_locs = locs.sort_values(kind="quicksort", by="frame")
    if len(out_locs) == 0:
        empty = np.array([], dtype=np.int32)
        return lib.append_group(out_locs.copy(), empty), info

    if "group" in out_locs.columns:
        group = out_locs["group"].to_numpy()
    else:
        group = np.zeros(len(out_locs), dtype=np.int32)
    frame = out_locs["frame"].to_numpy()
    x = out_locs["x"].to_numpy()
    y = out_locs["y"].to_numpy()
    link_group = _get_link_groups(frame, x, y, r_max, max_dark_time, group)

    # find the first and the last frame of each binding event; a link
    # group holds at most one localization per frame
    n_locs = len(link_group)
    n_groups = link_group.max() + 1
    n_locs_per_event = _link_group_count(link_group, n_locs, n_groups)
    first_frame, last_frame = _link_group_min_max(
        frame, link_group, n_locs, n_groups
    )

    keep = (
        (n_locs_per_event[link_group] >= min_n_locs)
        & (frame != first_frame[link_group])
        & (frame != last_frame[link_group])
    )
    out_locs = out_locs[keep].copy()
    # relabel the surviving events with consecutive group ids
    event_id = np.unique(link_group[keep], return_inverse=True)[1]
    return lib.append_group(out_locs, event_id.astype(np.int32)), info


def combine_locs_in_picks(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None = None,
    index_blocks: tuple | None = None,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Combine localizations in picked regions, i.e., link all
    localizations in each pick into one.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    picks : list of tuples
        List of pick positions. See ``io.load_picks``.
    pick_shape : {'Circle', 'Rectangle', 'Polygon', 'Square', 'Box', 'Brush'}
        Shape of the picks.
    pick_size : float or None, optional
        Size of the picks. For circular picks, the size is the diameter;
        for rectangular picks, the size is the width; for square picks,
        the size is the side length. None for polygonal, box and brush
        picks (size not defined).
    index_blocks : tuple or None, optional
        Precomputed spatial index over ``locs`` as returned by
        ``get_index_blocks`` (built with block size equal to the pick
        radius). When provided, used to skip re-indexing inside circular
        ``picked_locs``. Ignored for non-circular pick shapes. Default
        is None (index is computed on demand).
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Picking localizations" over
        the picks, then "Combining picks" over the picks. Default is
        None.

    Returns
    -------
    out_locs : pd.DataFrame
        Localizations after combining localizations in the picked
        regions.
    info : list of dicts
        Metadata of ``out_locs``: ``info`` with an entry recording the
        combining appended, see ``lib.append_metadata``.
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}."
    if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE:
        assert (
            pick_size is not None
        ), "Pick size must be provided for this pick shape."
    with lib.tracked_progress(
        progress, "Picking localizations", unit="pick"
    ) as progress:
        pl = picked_locs(
            locs,
            info,
            picks=picks,
            pick_shape=pick_shape,
            pick_size=pick_size,
            index_blocks=index_blocks,
            progress=progress,
        )
        # use very large values for linking localizations
        r_max = 2 * max(lib.field_of_view(info))
        max_dark = lib.get_from_metadata(info, "Frames", default=10_000)

        # link every localization in each pick
        lib.start_phase(progress, "Combining picks", len(pl), unit="pick")
        progress.set_value(0)
        out_locs = []
        for i, pick_locs in enumerate(pl):
            pick_locs_out = _link(
                pick_locs,
                info,
                r_max=r_max,
                max_dark_time=max_dark,
                remove_ambiguous_lengths=False,
            )
            if len(pick_locs_out):
                out_locs.append(pick_locs_out)
            progress.set_value(i + 1)
        progress.set_value(len(pl))
    out_locs = pd.concat(out_locs, ignore_index=True)
    entry = _picks_entry("Combine picks", picks, pick_shape, pick_size)
    return out_locs, lib.append_metadata(info, entry)


@numba.jit(nopython=True)
def _get_link_groups(
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    d_max: float,
    max_dark_time: int,
    group: lib.IntArray1D,
) -> lib.IntArray1D:
    """Find the groups for linking localizations into binding events.
    Assumes that ``locs`` are sorted by frame.

    Parameters
    ----------
    frame : lib.IntArray1D
        Frame numbers of localizations.
    x, y : lib.FloatArray1D
        Coordinates of localizations.
    d_max : float
        Maximum distance for linking localizations.
    max_dark_time : int
        Maximum number of frames between localizations to be considered
        as originating from the same binding event.
    group : lib.IntArray1D
        Grouping array for binding events. If None, all binding events
        are considered to be in the same group.

    Returns
    -------
    link_group : lib.IntArray1D
        Array of link groups for each localization. Each group is
        represented by a unique integer. Localizations that are not
        linked to any other localization are assigned -1.
    """
    N = len(x)
    link_group = -np.ones(N, dtype=np.int32)
    current_link_group = -1
    for i in range(N):
        if link_group[i] == -1:  # loc has no group yet
            current_link_group += 1
            link_group[i] = current_link_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(
                current_index,
                link_group,
                N,
                frame,
                x,
                y,
                d_max,
                max_dark_time,
                group,
            )
            while next_loc_index_in_group != -1:
                link_group[next_loc_index_in_group] = current_link_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(
                    current_index,
                    link_group,
                    N,
                    frame,
                    x,
                    y,
                    d_max,
                    max_dark_time,
                    group,
                )
    return link_group


@numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(  # noqa: C901
    current_index: int,
    link_group: lib.IntArray1D,
    N: int,
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    d_max: float,
    max_dark_time: float,
    group: lib.IntArray1D,
) -> int:
    """Find the next localization index in the link group for a given
    current localization index. The next localization is the one that
    is in the same group, has a frame greater than the current frame
    plus one, and is within the maximum distance defined by d_max.
    If no such localization is found, returns -1."""
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    current_group = group[current_index]
    min_frame = current_frame + 1
    for min_index in range(current_index + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = current_frame + max_dark_time + 1
    for max_index in range(min_index, N):
        if frame[max_index] > max_frame:
            break
    else:
        max_index = N
    d_max_2 = d_max**2
    for j in range(min_index, max_index):
        if group[j] == current_group:
            if link_group[j] == -1:
                dx2 = (current_x - x[j]) ** 2
                if dx2 <= d_max_2:
                    dy2 = (current_y - y[j]) ** 2
                    if dy2 <= d_max_2:
                        if dx2 + dy2 <= d_max_2:
                            return j
    return -1


@numba.jit(nopython=True)
def _link_group_count(
    link_group: lib.IntArray1D,
    n_locs: int,
    n_groups: int,
) -> lib.IntArray1D:
    """Count the number of localizations in each link group."""
    result = np.zeros(n_groups, dtype=np.uint32)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += 1
    return result


@numba.jit(nopython=True)
def _link_group_sum(
    column: lib.IntArray1D | lib.FloatArray1D,
    link_group: lib.IntArray1D,
    n_locs: int,
    n_groups: int,
) -> lib.IntArray1D | lib.FloatArray1D:
    """Sum the values of a column for each link group."""
    result = np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += column[i]
    return result


@numba.jit(nopython=True)
def _link_group_mean(
    column: lib.IntArray1D | lib.FloatArray1D,
    link_group: lib.IntArray1D,
    n_locs: int,
    n_groups: int,
    n_locs_per_group: lib.IntArray1D,
) -> lib.FloatArray1D:
    """Calculate the mean of a column for each link group."""
    group_sum = _link_group_sum(column, link_group, n_locs, n_groups)
    result = np.empty(
        n_groups, dtype=np.float32
    )  # this ensures float32 after the division
    result[:] = group_sum / n_locs_per_group
    return result


@numba.jit(nopython=True)
def _link_group_weighted_mean(
    column: lib.IntArray1D | lib.FloatArray1D,
    weights: lib.FloatArray1D,
    link_group: lib.IntArray1D,
    n_locs: int,
    n_groups: int,
    n_locs_per_group: lib.IntArray1D,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the mean of a column for each link group and the sum
    of the weights."""
    sum_weights = _link_group_sum(weights, link_group, n_locs, n_groups)
    return (
        _link_group_mean(
            column * weights,
            link_group,
            n_locs,
            n_groups,
            sum_weights,
        ),
        sum_weights,
    )


@numba.jit(nopython=True)
def _link_group_min_max(
    column: lib.IntArray1D | lib.FloatArray1D,
    link_group: lib.IntArray1D,
    n_locs: int,
    n_groups: int,
) -> tuple[
    lib.IntArray1D | lib.FloatArray1D, lib.IntArray1D | lib.FloatArray1D
]:
    """Calculate the minimum and maximum of a column for each link
    group."""
    min_ = np.empty(n_groups, dtype=column.dtype)
    max_ = np.empty(n_groups, dtype=column.dtype)
    min_[:] = column.max()
    max_[:] = column.min()
    for i in range(n_locs):
        i_ = link_group[i]
        value = column[i]
        if value < min_[i_]:
            min_[i_] = value
        if value > max_[i_]:
            max_[i_] = value
    return min_, max_


@numba.jit(nopython=True)
def _link_group_last(
    column: lib.IntArray1D | lib.FloatArray1D,
    link_group: lib.IntArray1D,
    n_locs: int,
    n_groups: int,
) -> lib.IntArray1D | lib.FloatArray1D:
    """Return the last value of a column for each link group."""
    result = np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] = column[i]
    return result


def _link_loc_groups(  # noqa: C901
    locs: pd.DataFrame,
    info: list[dict],
    link_group: lib.IntArray1D,
    remove_ambiguous_lengths: bool = True,
) -> pd.DataFrame:
    """Combine localizations into binding events based on the
    spatiotemporal proximity defined by the ``link_group``. Takes the
    average position to calculate the coordinates of the binding events.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localization list.
    link_group : lib.IntArray1D
        Array that defines the link groups for the localizations.
    remove_ambiguous_lengths : bool, optional
        If True, removes linked localizations with ambiguous lengths,
        i.e., localizations that are linked to multiple binding events
        with different lengths. Default is True.

    Returns
    -------
    linked_locs : pd.DataFrame
        Linked localizations, i.e., binding events with their
        properties.
    """
    n_locs = len(link_group)
    n_groups = link_group.max() + 1
    n_ = _link_group_count(link_group, n_locs, n_groups)
    columns = OrderedDict()
    if "frame" in locs.columns:
        first_frame_, last_frame_ = _link_group_min_max(
            locs["frame"].to_numpy(), link_group, n_locs, n_groups
        )
        columns["frame"] = first_frame_
    if "x" in locs.columns:
        weights_x = 1 / locs["lpx"].to_numpy() ** 2
        columns["x"], sum_weights_x_ = _link_group_weighted_mean(
            locs["x"].to_numpy(), weights_x, link_group, n_locs, n_groups, n_
        )
    if "y" in locs.columns:
        weights_y = 1 / locs["lpy"].to_numpy() ** 2
        columns["y"], sum_weights_y_ = _link_group_weighted_mean(
            locs["y"].to_numpy(), weights_y, link_group, n_locs, n_groups, n_
        )
    if "photons" in locs.columns:
        columns["photons"] = _link_group_sum(
            locs["photons"].to_numpy(),
            link_group,
            n_locs,
            n_groups,
        )
    if "sx" in locs.columns:
        columns["sx"] = _link_group_mean(
            locs["sx"].to_numpy(),
            link_group,
            n_locs,
            n_groups,
            n_,
        )
    if "sy" in locs.columns:
        columns["sy"] = _link_group_mean(
            locs["sy"].to_numpy(),
            link_group,
            n_locs,
            n_groups,
            n_,
        )
    if "bg" in locs.columns:
        columns["bg"] = _link_group_sum(
            locs["bg"].to_numpy(),
            link_group,
            n_locs,
            n_groups,
        )
    if "x" in locs.columns:
        columns["lpx"] = np.sqrt(1 / sum_weights_x_)
    if "y" in locs.columns:
        columns["lpy"] = np.sqrt(1 / sum_weights_y_)
    if "ellipticity" in locs.columns:
        columns["ellipticity"] = _link_group_mean(
            locs["ellipticity"].to_numpy(), link_group, n_locs, n_groups, n_
        )
    if "net_gradient" in locs.columns:
        columns["net_gradient"] = _link_group_mean(
            locs["net_gradient"].to_numpy(), link_group, n_locs, n_groups, n_
        )
    for col in (
        "log_likelihood",
        "likelihood",
        "chi_square",
        "reduced_chi_square",
    ):
        # "likelihood" is the old name of the column, kept for files saved
        # with earlier versions of Picasso. "chi_square" is its least-squares
        # counterpart and "reduced_chi_square" the normalized form of both;
        # averaged the same way (a mean over the linked locs).
        if col in locs.columns:
            columns[col] = _link_group_mean(
                locs[col].to_numpy(), link_group, n_locs, n_groups, n_
            )
    if "iterations" in locs.columns:
        columns["iterations"] = _link_group_mean(
            locs["iterations"].to_numpy(), link_group, n_locs, n_groups, n_
        )
    if "z" in locs.columns:
        if "lpz" in locs.columns:
            weights_z = 1 / locs["lpz"].to_numpy() ** 2
            columns["z"], sum_weights_z_ = _link_group_weighted_mean(
                locs["z"].to_numpy(),
                weights_z,
                link_group,
                n_locs,
                n_groups,
                n_,
            )
            columns["lpz"] = np.sqrt(1 / sum_weights_z_)
        else:
            columns["z"] = _link_group_mean(
                locs["z"].to_numpy(),
                link_group,
                n_locs,
                n_groups,
                n_,
            )
    if "d_zcalib" in locs.columns:
        columns["d_zcalib"] = _link_group_mean(
            locs["d_zcalib"].to_numpy(), link_group, n_locs, n_groups, n_
        )
    if "group" in locs.columns:
        columns["group"] = _link_group_last(
            locs["group"].to_numpy(),
            link_group,
            n_locs,
            n_groups,
        )
    if "frame" in locs.columns:
        columns["len"] = last_frame_ - first_frame_ + 1
    columns["n"] = n_
    if "photons" in locs.columns:
        columns["photon_rate"] = np.float32(columns["photons"] / n_)
    linked_locs = pd.DataFrame(columns)
    if remove_ambiguous_lengths:
        valid = np.logical_and(
            first_frame_ > 0,
            last_frame_ < info[0]["Frames"],
        )
        linked_locs = linked_locs[valid]
    return linked_locs


def n_segments(info: list[dict], segmentation: int) -> int:
    """Calculate the number of segments for the given segmentation
    for undrifting.

    Parameters
    ----------
    info : list of dicts
        Metadata of the localizations list.
    segmentation : int
        Number of segments to divide the data into.

    Returns
    -------
    n_segments : int
        Number of segments based on the total number of frames and the
        segmentation value.
    """
    n_frames = lib.get_from_metadata(info, "Frames")
    n_segments = int(np.round(n_frames / segmentation))
    return n_segments


def segment(
    locs: pd.DataFrame,
    info: list[dict],
    segmentation: int,
    kwargs: dict = {},
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[lib.IntArray1D, lib.FloatArray3D]:
    """Split localizations into temporal segments (number of segments
    is defined by the segmentation parameter) and render each segment
    into a 2D image.

    Parameters
    ----------
    locs : pd.DataFrame
        Localization list.
    info : list of dicts
        Metadata of the localization list.
    segmentation : int
        Number of segments to divide the data into.
    kwargs : dict, optional
        Additional keyword arguments for the rendering function, e.g.,
        ``disp_px_size`` (nm, default ``UNDRIFT_PIXEL_SIZE``) and
        ``min_blur_width`` (nm). Default is an empty dictionary.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Generating segments" over the
        segments. Default is None.

    Returns
    -------
    bounds : lib.IntArray1D
        Array of bounds for each segment, where each bound is the
        starting frame of the segment.
    segments : lib.FloatArray3D
        3D array of segments, where each segment is a 2D image of the
        localizations in that segment.
    """
    X, Y = lib.field_of_view(info)
    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
    kwargs = dict(kwargs)
    disp_px_size = kwargs.pop("disp_px_size", UNDRIFT_PIXEL_SIZE)
    oversampling = 1 / disp_px_size  # rendered pixels per nm
    n_pixel_y = int(np.ceil(oversampling * Y))
    n_pixel_x = int(np.ceil(oversampling * X))
    n_seg = n_segments(info, segmentation)
    bounds = np.linspace(0, n_frames - 1, n_seg + 1, dtype=np.uint32)
    segments = np.zeros((n_seg, n_pixel_y, n_pixel_x))
    with lib.tracked_progress(
        progress, "Generating segments", unit="segment"
    ) as progress:
        lib.start_phase(progress, "Generating segments", n_seg, unit="segment")
        progress.set_value(0)
        for i in range(n_seg):
            segment_locs = locs[
                (locs["frame"] >= bounds[i]) & (locs["frame"] < bounds[i + 1])
            ]
            _, segments[i] = render.render(
                segment_locs, info, disp_px_size=disp_px_size, **kwargs
            )
            progress.set_value(i + 1)
        progress.set_value(n_seg)
    return bounds, segments


def undrift(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    segmentation: int,
    display: bool = True,
    progress: lib.Progress | Literal["console"] | None = None,
    disp_px_size: float = UNDRIFT_PIXEL_SIZE,
    max_shift: float = UNDRIFT_MAX_SHIFT,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Undrift by RCC. See Wang, Schnitzbauer, et al. Optics Express,
    2014.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to undrift.
    info : list of dicts
        Metadata of the localization list.
    segmentation : int
        Number of segments to divide the data into for undrifting.
    display : bool, optional
        If True, displays the estimated drift. Default is True.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Generating segments" over the
        segments, then "Correlating image pairs" over the segment pairs.
        Default is None.
    disp_px_size : float, optional
        Pixel size (nm) of the rendered segments, which are blurred by
        one pixel. Default is 100.
    max_shift : float, optional
        Largest shift (nm) between two segments considered. Default is
        10000.

    Returns
    -------
    locs : pd.DataFrame
        Undrifted localization list with the drift applied to the 'x'
        and 'y' coordinates.
    info : list of dicts
        Metadata of the undrifted localizations: ``info`` with an entry
        recording the undrifting appended, see ``lib.append_metadata``.
    drift : pd.DataFrame
        Estimated drift (nm) as a DataFrame with columns 'x' and 'y'.

    Raises
    ------
    ValueError
        If the segmentation is not positive or divides the movie into
        fewer than 4 segments, the fewest the drift interpolation needs.
    """
    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
    if not segmentation > 0:
        raise ValueError(
            f"The segmentation must be a positive number of frames, got "
            f"{segmentation:g}."
        )
    n_segs = n_segments(info, segmentation)
    if n_segs < MIN_UNDRIFT_SEGMENTS:
        message = (
            f"A segmentation of {segmentation:g} frames divides the "
            f"{n_frames} frames into {n_segs} segment(s), but undrifting "
            f"by RCC needs at least {MIN_UNDRIFT_SEGMENTS}."
        )
        if n_frames >= MIN_UNDRIFT_SEGMENTS:
            message += (
                " Use a segmentation of at most "
                f"{n_frames // MIN_UNDRIFT_SEGMENTS} frames."
            )
        raise ValueError(message)
    locs = locs.copy()
    with lib.tracked_progress(
        progress, "Generating segments", unit="segment"
    ) as progress:
        bounds, segments = segment(
            locs,
            info,
            segmentation,
            {
                "disp_px_size": disp_px_size,
                "blur_method": "gaussian",
                "min_blur_width": disp_px_size,  # one rendered pixel
            },
            progress=progress,
        )
        # rcc works in rendered pixels; the shifts are converted to nm
        shift_y, shift_x = imageprocess.rcc(
            segments, max_shift / disp_px_size, progress=progress
        )
    shift_x = np.asarray(shift_x, dtype=np.float64) * disp_px_size
    shift_y = np.asarray(shift_y, dtype=np.float64) * disp_px_size
    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = interpolate.InterpolatedUnivariateSpline(t, shift_x, k=3)
    drift_y_pol = interpolate.InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = np.arange(info[0]["Frames"])
    drift_ = (drift_x_pol(t_inter), drift_y_pol(t_inter))
    drift = pd.DataFrame({"x": drift_[0], "y": drift_[1]})
    if display:
        plot_drift(drift)
        plt.show()
    locs = _apply_drift(locs, drift)
    entry = lib.step_info(
        "Undrift",
        **{
            "Method": "RCC",
            "Segmentation": segmentation,
            "Display pixel size (nm)": disp_px_size,
            "Max. shift (nm)": max_shift,
        },
    )
    return locs, lib.append_metadata(info, entry), drift


def undrift_from_fiducials(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple] | None = None,
    pick_size: float | None = None,
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ] = "Circle",
    undrift_z: bool = True,
    index_blocks: tuple | None = None,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Undrift localizations based on picked regions (fiducial markers).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be undrifted.
    info : list of dicts
        Localizations' metadata.
    picks : list of tuples or None, optional
        Coordinates of picked regions, in the format required by
        ``pick_shape``. If None (default), circular fiducials are
        automatically detected using
        ``picasso.imageprocess.find_fiducials``.
    pick_size : float or None, optional
        Size of the picks in nm: the diameter for circles, the width
        for rectangles, the side length for squares.
        Required when ``picks`` is a list of coordinates, unless
        ``pick_shape`` carries its own extent ("Polygon", "Box").
        Ignored when ``picks`` is None (determined by
        ``find_fiducials``).
    pick_shape : str, optional
        Shape of the given picks, one of ``lib.PICK_SHAPES``. Forced to
        "Circle" when ``picks`` is None. Default is "Circle".
    undrift_z : bool, optional
        If True, also undrift the z coordinate if it exists in the
        localizations. Default is True.
    index_blocks : tuple or None, optional
        Precomputed spatial index over ``locs`` as returned by
        ``get_index_blocks`` (built with block size equal to the pick
        radius, ``pick_size / 2``). When provided, used to skip
        re-indexing inside circular ``picked_locs``. Ignored for every
        other pick shape, and when ``picks`` is None (auto-detected
        fiducials use a size that may not match the precomputed index).
        Default is None.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Finding fiducials" (busy
        indicator, only when ``picks`` is None), then "Picking
        localizations" over the picks, then "Calculating drift" (busy
        indicator). Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Undrifted localizations.
    info : list of dicts
        Metadata of the undrifted localizations: ``info`` with an entry
        recording the undrifting appended, see ``lib.append_metadata``.
    drift : pd.DataFrame
        Drift in x and y (and optionally z) directions, in nm.

    Raises
    ------
    ValueError
        If ``pick_size`` is not provided when ``picks`` is a list of
        coordinates of a shape that needs one.
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}."
    locs = locs.copy()

    with lib.tracked_progress(
        progress, "Picking localizations", unit="pick"
    ) as progress:
        if picks is None:
            # auto-detect fiducials; these are always circular
            lib.start_phase(progress, "Finding fiducials", 0)
            progress.set_value(0)
            picks, box = imageprocess.find_fiducials(locs, info)
            progress.set_value(0)
            pick_shape = "Circle"
            pick_size = box
            # passed-in index_blocks was built for a different size;
            # drop (the pyramid serves any size)
            if not _is_pyramid(index_blocks):
                index_blocks = None
        else:
            # user-provided list of pick coordinates
            needs_size = pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE
            if pick_size is None and needs_size:
                raise ValueError(
                    "pick_size (diameter in nm for circular picks) "
                    "must be provided when picks are given as a list of "
                    "coordinates."
                )

        if pick_shape != "Circle":
            # the index is built for circular lookups only
            index_blocks = None

        if len(picks) == 0:
            raise ValueError("No picks found for drift correction.")

        # get picked localizations
        pl = picked_locs(
            locs,
            info,
            picks=picks,
            pick_shape=pick_shape,
            pick_size=pick_size,
            add_group=False,
            index_blocks=index_blocks,
            progress=progress,
        )

        # calculate drift
        lib.start_phase(progress, "Calculating drift", 0)
        progress.set_value(0)
        drift = undrift_from_picked(pl, info)
        progress.set_value(0)
    if not undrift_z:
        drift = drift.drop(columns="z", errors="ignore")
    locs = _apply_drift(locs, drift)

    entry = _picks_entry("Undrift from picked", picks, pick_shape, pick_size)
    return locs, lib.append_metadata(info, entry), drift


def undrift_from_picked(
    picked_locs: list[pd.DataFrame], info: list[dict]
) -> pd.DataFrame:
    """Find drift from picked localizations. Note that unlike other
    undrifting functions, this function does not return undrifted
    localizations but only drift.

    Parameters
    ----------
    picked_locs : list of pd.DataFrames
        List of picked localizations, where each element is a data frame
        of localizations for a single pick.
    info : list of dicts
        Metadata of the localization list, where each element
        corresponds to the metadata of the localizations in
        ``picked_locs``.

    Returns
    -------
    drift : pd.DataFrame
        Estimated drift as a DataFrame with columns 'x', 'y', and
        optionally 'z' if the z coordinate exists in the picked
        localizations.
    """
    n_frames = info[0]["Frames"]
    # the picks' localizations as flat arrays, with the index of the pick
    lengths = np.array([len(_) for _ in picked_locs], dtype=np.int64)
    pick = np.repeat(np.arange(len(picked_locs)), lengths)
    if len(pick):
        frame = np.concatenate(
            [_["frame"].to_numpy() for _ in picked_locs]
        ).astype(np.int64)
    else:
        frame = np.empty(0, dtype=np.int64)
    # one entry per pick and frame: the pick's last localization in it
    key = pick * n_frames + frame
    _, first_in_reversed = np.unique(key[::-1], return_index=True)
    entries = len(key) - 1 - first_in_reversed

    def coordinate_drift(coordinate: str) -> lib.FloatArray1D:
        if len(pick):
            values = np.concatenate(
                [_[coordinate].to_numpy() for _ in picked_locs]
            )
        else:
            values = np.empty(0)
        return _undrift_from_picked_coordinate(
            values, pick, frame, lengths, entries, n_frames
        )

    # A data frame to store the applied drift
    drift = pd.DataFrame(
        {"x": coordinate_drift("x"), "y": coordinate_drift("y")}
    )
    # If z coordinate exists, also apply drift there
    if all(["z" in _.columns for _ in picked_locs]):
        drift["z"] = coordinate_drift("z")
    return drift


def _undrift_from_picked_coordinate(
    values: lib.FloatArray1D,
    pick: lib.IntArray1D,
    frame: lib.IntArray1D,
    lengths: lib.IntArray1D,
    entries: lib.IntArray1D,
    n_frames: int,
) -> lib.FloatArray1D:
    """Calculate drift in one coordinate from picked localizations.

    Each pick's drift is its localizations' coordinate minus the pick's
    center of mass. The drift of a frame is the average over the picks
    with a localization in it (one per pick; the last one if a pick has
    several), each pick weighted by the inverse of its mean square
    deviation from the unweighted average drift. Frames without
    localizations are interpolated linearly.

    The statistics are accumulated over the localizations, never over a
    (picks x frames) table, whose memory would grow with the number of
    picks times the number of frames.

    Parameters
    ----------
    values : lib.FloatArray1D
        Coordinate of all picked localizations, pick by pick.
    pick : lib.IntArray1D
        Index of the pick of each localization in ``values``.
    frame : lib.IntArray1D
        Frame of each localization in ``values``.
    lengths : lib.IntArray1D
        Number of localizations in each pick.
    entries : lib.IntArray1D
        Indices into ``values`` of the localization that represents
        each pick in each of its frames.
    n_frames : int
        Number of frames.

    Returns
    -------
    drift_mean : lib.FloatArray1D
        Average drift across picks for all frames.
    """
    n_picks = len(lengths)
    with np.errstate(invalid="ignore", divide="ignore"):
        # center of mass of each pick, over all of its localizations
        com = np.bincount(pick, weights=values, minlength=n_picks) / lengths
        pick_e = pick[entries]
        frame_e = frame[entries]
        drift = values[entries] - com[pick_e]
        # mean drift over picks
        n_per_frame = np.bincount(frame_e, minlength=n_frames)
        drift_mean = (
            np.bincount(frame_e, weights=drift, minlength=n_frames)
            / n_per_frame
        )
        # mean square deviation of each pick's drift to the mean drift
        sd = (drift - drift_mean[frame_e]) ** 2
        msd = np.bincount(pick_e, weights=sd, minlength=n_picks) / (
            np.bincount(pick_e, minlength=n_picks)
        )
        # new mean drift over picks, where each pick is weighted
        # according to its msd
        weights = 1 / msd[pick_e]
        drift_mean = np.bincount(
            frame_e, weights=weights * drift, minlength=n_frames
        ) / np.bincount(frame_e, weights=weights, minlength=n_frames)
    drift_mean[n_per_frame == 0] = np.nan

    # Linear interpolation for frames without localizations
    nans = np.isnan(drift_mean)
    drift_mean[nans] = np.interp(
        np.flatnonzero(nans), np.flatnonzero(~nans), drift_mean[~nans]
    )
    return drift_mean


def _apply_drift(locs: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    """Apply drift to localizations. This is a helper function that assumes
    the drift is already in the correct format and that the number of
    frames matches."""
    frames = locs["frame"]
    locs["x"] -= drift["x"].iloc[frames].to_numpy()
    locs["y"] -= drift["y"].iloc[frames].to_numpy()
    if "z" in drift.columns and "z" in locs.columns:
        locs["z"] -= drift["z"].iloc[frames].to_numpy()
    return locs


def apply_drift(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    drift: pd.DataFrame | lib.FloatArray2D,
) -> tuple[pd.DataFrame, list[dict]]:
    """Convenience function to apply drift to localizations. Runs checks
    to ensure correct formats. The coordinates of ``locs`` are shifted
    in place, i.e., the returned localizations are ``locs`` itself.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to apply drift to.
    info : list of dicts
        Metadata of the localization list.
    drift : pd.DataFrame or lib.FloatArray2D
        Drift to apply. If a DataFrame, it should have columns 'x' and
        'y', and optionally 'z'. If a numpy array, it should have shape
        (n_frames, 2) for x and y drift, or (n_frames, 3) for x, y, and
        z drift.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with drift applied to the 'x' and 'y' coordinates
        (and 'z' if it exists in the drift).
    info : list of dicts
        Metadata of ``locs``: ``info`` with an entry recording the step
        appended, see ``lib.append_metadata``.
    """
    assert isinstance(
        drift, (pd.DataFrame, np.ndarray)
    ), "Drift must be a DataFrame or numpy array"
    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
    if isinstance(drift, pd.DataFrame):
        required_columns = {"x", "y"}
        if not required_columns.issubset(drift.columns):
            raise ValueError(
                f"Drift DataFrame must contain columns {required_columns}"
            )
    elif isinstance(drift, np.ndarray):
        if not (drift.shape[1] in [2, 3] and drift.shape[0] == n_frames):
            raise ValueError(
                "Drift array must have shape (n_frames, 2) for x and y drift, "
                "or (n_frames, 3) for x, y, and z drift."
            )
        drift = pd.DataFrame(
            drift,
            columns=["x", "y"] + (["z"] if drift.shape[1] == 3 else []),
        )
    entry = lib.step_info("Apply drift")
    return _apply_drift(locs, drift), lib.append_metadata(info, entry)


def plot_drift(
    drift: pd.DataFrame,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Convenience function to plot 2D or 3D drift from a DataFrame.

    Parameters
    ----------
    drift : pd.DataFrame
        DataFrame containing the drift (nm) to plot. Should have columns
        'x' and 'y', and optionally 'z'.
    fig : plt.Figure or None, optional
        Matplotlib figure to plot on. If None (default), a new figure is
        created.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    """
    assert isinstance(drift, pd.DataFrame), "Drift must be a DataFrame."
    assert (
        "x" in drift.columns and "y" in drift.columns
    ), "Drift must have 'x' and 'y' columns."
    if fig is None:
        fig = plt.Figure(figsize=(10, 6), constrained_layout=True)
    else:
        fig.clear()

    if "z" in drift.columns:
        ax1 = fig.add_subplot(131)
        ax1.plot(drift["x"], label="x")
        ax1.plot(drift["y"], label="y")
        ax1.legend(loc="best")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Drift (nm)")
        ax2 = fig.add_subplot(132)
        ax2.plot(
            drift.x,
            drift.y,
            color=list(plt.rcParams["axes.prop_cycle"])[2]["color"],
        )
        ax2.set_aspect("equal")
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        ax2.invert_yaxis()
        ax3 = fig.add_subplot(133)
        # fourth color, so that z is not mistaken for x
        ax3.plot(
            drift.z,
            label="z",
            color=list(plt.rcParams["axes.prop_cycle"])[3]["color"],
        )
        ax3.legend(loc="best")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Drift (nm)")
    else:
        ax1 = fig.add_subplot(121)
        ax1.plot(drift["x"], label="x")
        ax1.plot(drift["y"], label="y")
        ax1.legend(loc="best")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Drift (nm)")
        ax2 = fig.add_subplot(122)
        ax2.plot(
            drift.x,
            drift.y,
            color=list(plt.rcParams["axes.prop_cycle"])[2]["color"],
        )
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        ax2.invert_yaxis()
        ax2.set_aspect("equal")
    return fig


def align(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    method: Literal["rcc", "picked"] = "rcc",
    picks: list[tuple] | None = None,
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ] = "Circle",
    pick_size: float | None = None,
    index_blocks: list[tuple | None] | None = None,
    display: bool = False,
    return_shifts: bool = False,
    keep_non_overlapping: bool = True,
) -> (
    tuple[list[pd.DataFrame], list[list[dict]]]
    | tuple[list[pd.DataFrame], list[list[dict]], tuple]
):
    """Align localizations from multiple channels (one per element in
    ``locs``).

    The shifts between the channels are found either by RCC of the
    rendered images (``method="rcc"``, see ``align_rcc``) or from the
    localizations in picked regions (``method="picked"``, see
    ``align_from_picked``). The input DataFrames are not modified.

    Parameters
    ----------
    locs : list of pd.DataFrames
        Localizations of each channel.
    infos : list of list of dicts
        Metadata of each channel in ``locs``.
    method : {"rcc", "picked"}, optional
        How the shifts are found. Default is "rcc".
    picks : list or None, optional
        Picked regions, in the format required by ``pick_shape``, see
        ``io.load_picks``. Required for ``method="picked"`` and not
        accepted otherwise. Default is None.
    pick_shape : str, optional
        Shape of the picks, one of ``lib.PICK_SHAPES``. Only used for
        ``method="picked"``. Default is "Circle".
    pick_size : float or None, optional
        Size of the picks: the diameter of circular picks, the side
        length of square picks and the width of rectangular picks. None
        for polygon, box and brush picks. Only used for
        ``method="picked"``. Default is None.
    index_blocks : list of tuple or None, optional
        Per-channel precomputed spatial indices for circular picks, see
        ``align_from_picked``. Only used for ``method="picked"``.
        Default is None.
    display : bool, optional
        If True, plots the mean shift per RCC iteration. Only used for
        ``method="rcc"``. Default is False.
    return_shifts : bool, optional
        If True, also returns the shifts that were subtracted from each
        channel. Default is False.
    keep_non_overlapping : bool, optional
        If False, localizations outside the region where all aligned
        channels overlap are removed, see
        ``remove_non_overlapping_locs``. Default is True (all
        localizations are kept).

    Returns
    -------
    locs : list of pd.DataFrames
        Aligned localizations of each channel.
    infos : list of lists of dicts
        Metadata of each aligned channel: its ``infos`` element with an
        entry recording the alignment (the method and the shift
        subtracted from the channel) appended, see
        ``lib.append_metadata``.
    shifts : tuple of np.ndarrays
        ``(shift_x, shift_y)``, or ``(shift_x, shift_y, shift_z)`` if
        z is aligned too (picked regions only). Each
        array holds the shift subtracted from each channel, in nm.
        Returned only if ``return_shifts`` is True.

    Raises
    ------
    ValueError
        If ``method`` is unknown, if ``picks`` are missing for
        ``method="picked"`` or given for ``method="rcc"``, or if
        ``keep_non_overlapping`` is False and the aligned channels do
        not overlap.
    """
    if method == "rcc":
        if picks is not None:
            raise ValueError(
                "picks are only used with method='picked', not 'rcc'."
            )
        locs, infos, shifts = align_rcc(
            locs,
            infos,
            display=display,
            return_shifts=True,
            keep_non_overlapping=keep_non_overlapping,
        )
    elif method == "picked":
        if picks is None:
            raise ValueError("method='picked' requires picks.")
        locs, infos, shifts = align_from_picked(
            locs,
            infos,
            picks=picks,
            pick_shape=pick_shape,
            pick_size=pick_size,
            index_blocks=index_blocks,
            return_shifts=True,
            keep_non_overlapping=keep_non_overlapping,
        )
    else:
        raise ValueError(
            f"Unknown alignment method {method!r}, use 'rcc' or 'picked'."
        )
    if return_shifts:
        return locs, infos, shifts
    return locs, infos


def _aligned_infos(
    infos: list[list[dict]], shifts: tuple, entry: dict
) -> list[list[dict]]:
    """Metadata of aligned channels: each element of ``infos`` with a
    copy of ``entry`` appended that also holds the shift subtracted
    from the channel (nm)."""
    new_infos = []
    for i, info in enumerate(infos):
        channel_entry = dict(entry)
        for axis, shift in zip("xyz", shifts):
            channel_entry[f"Shift {axis} (nm)"] = float(shift[i])
        new_infos.append(lib.append_metadata(info, channel_entry))
    return new_infos


def _rcc_shifts(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
) -> tuple[np.ndarray, np.ndarray]:
    """One RCC pass over the rendered images of all channels, see
    ``align_rcc``.

    Parameters
    ----------
    locs : list of pd.DataFrames
        Localizations of each channel.
    infos : list of list of dicts
        Metadata of each channel in ``locs``.

    Returns
    -------
    shift_x, shift_y : np.ndarrays
        Shift of each channel in nm, to be subtracted from its 'x' and
        'y' coordinates.
    """
    images = []
    disp_px_size = ALIGN_PIXEL_SIZE
    # one canvas for all channels, so that the images have the same
    # shape even if the channels' canvases differ (e.g., after moving
    # a channel in Render, see lib.fit_canvas)
    fovs = [lib.field_of_view(info_) for info_ in infos]
    width = max(fov[0] for fov in fovs)
    height = max(fov[1] for fov in fovs)
    viewport = ((0, 0), (height, width))
    for locs_, info_ in zip(locs, infos):
        _, image = render.render(
            locs_,
            info_,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method="smooth",
        )
        images.append(image)
    shift_y, shift_x = imageprocess.rcc(images)
    # `rcc` returns shifts in rendered-image pixels; convert them to nm,
    # the units of the localizations' 'x' and 'y' coordinates
    shift_x = np.asarray(shift_x, dtype=float) * disp_px_size
    shift_y = np.asarray(shift_y, dtype=float) * disp_px_size
    return shift_x, shift_y


def align_rcc(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    display: bool = False,
    return_shifts: bool = False,
    keep_non_overlapping: bool = True,
) -> (
    tuple[list[pd.DataFrame], list[list[dict]]]
    | tuple[list[pd.DataFrame], list[list[dict]], tuple]
):
    """Align localizations from multiple channels (one per each element
    in `locs`) by calculating the shifts between the rendered images
    using RCC. The RCC is repeated until the shifts converge (at most
    five times). The input DataFrames are not modified.

    Parameters
    ----------
    locs : list of pd.DataFrames
        List of localization datasets which are to be aligned.
    infos : list of list of dicts
        List of metadata dictionaries corresponding to each
        localization DataFrame in `locs`.
    display : bool, optional
        If True, plots the mean shift per iteration. Default is False.
    return_shifts : bool, optional
        If True, also returns the calculated shifts for each channel.
        Default is False.
    keep_non_overlapping : bool, optional
        If False, localizations outside the region where all aligned
        channels overlap are removed, see
        ``remove_non_overlapping_locs``. Default is True (all
        localizations are kept).

    Returns
    -------
    locs : list of pd.DataFrames
        Aligned localizations with the shifts applied to the 'x' and
        'y' coordinates.
    infos : list of lists of dicts
        Metadata of each aligned channel: its ``infos`` element with an
        entry recording the alignment (and the shift subtracted from the
        channel) appended, see ``lib.append_metadata``.
    shifts : tuple of np.ndarrays
        ``(shift_x, shift_y)``. Each array holds the total shift
        subtracted from each channel, in nm. Returned only if
        ``return_shifts`` is True.

    Raises
    ------
    ValueError
        If ``keep_non_overlapping`` is False and the aligned channels
        do not overlap.
    """
    locs = [locs_.copy() for locs_ in locs]
    max_iterations = 5
    convergence = ALIGN_CONVERGENCE  # nm
    total_x = np.zeros(len(locs))
    total_y = np.zeros(len(locs))
    mean_x = []
    mean_y = []
    for _ in range(max_iterations):
        shift_x, shift_y = _rcc_shifts(locs, infos)
        for locs_, dx, dy in zip(locs, shift_x, shift_y):
            locs_["x"] -= dx
            locs_["y"] -= dy
        total_x += shift_x
        total_y += shift_y
        mean_x.append(np.mean(shift_x))
        mean_y.append(np.mean(shift_y))
        # skip when converged
        if np.all(np.abs(shift_x) + np.abs(shift_y) <= convergence):
            break

    if display:
        fig1 = plt.figure(figsize=(8, 8), constrained_layout=True)
        plt.suptitle("Shift")
        plt.subplot(1, 1, 1)
        plt.plot(mean_x, "o-", label="x shift")
        plt.plot(mean_y, "o-", label="y shift")
        plt.xlabel("Iteration")
        plt.ylabel("Mean shift per iteration (nm)")
        plt.legend(loc="best")
        fig1.show()

    if not keep_non_overlapping:
        locs = remove_non_overlapping_locs(locs)
    shifts = (total_x, total_y)
    entry = lib.step_info(
        "Align",
        **{
            "Method": "RCC",
            "Remove non-overlapping": not keep_non_overlapping,
        },
    )
    infos = _aligned_infos(infos, shifts, entry)
    if return_shifts:
        return locs, infos, shifts
    return locs, infos


def align_from_picked(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None = None,
    return_shifts: bool = False,
    index_blocks: list[tuple | None] | None = None,
    keep_non_overlapping: bool = True,
) -> (
    tuple[list[pd.DataFrame], list[list[dict]]]
    | tuple[list[pd.DataFrame], list[list[dict]], tuple]
):
    """Align localizations from multiple channels using picked
    localizations. The input DataFrames are not modified.

    Parameters
    ----------
    locs : list of pd.DataFrames
        List of localization datasets.
    infos : list of list of dicts
        List of metadata dictionaries corresponding to each localization
        dataset in ``locs``.
    picks : list of (2,) tuples
        Coordinates of picked regions as (x, y) tuples. See
        ``io.load_picks``.
    pick_shape : str
        Shape of the picks, one of ``lib.PICK_SHAPES``.
    pick_size : float or None, optional
        Size of the picks. For circular picks, the size is the diameter.
        For square picks, the size is the side length. For rectangular
        picks, the size is the width. None for polygon, box and brush
        picks. Default is None.
    return_shifts : bool, optional
        If True, also returns the calculated shifts for each channel.
        Default is False.
    index_blocks : list of tuple or None, optional
        Per-channel precomputed spatial indices (one entry per dataset
        in ``locs``, aligned by position) as returned by
        ``get_index_blocks``. Each entry may be ``None`` to recompute
        for that channel. Used to skip re-indexing inside circular
        ``picked_locs``. Ignored for non-circular pick shapes. Default
        is None (all channels recompute on demand).
    keep_non_overlapping : bool, optional
        If False, localizations outside the region where all aligned
        channels overlap are removed, see
        ``remove_non_overlapping_locs``. Default is True (all
        localizations are kept).

    Returns
    -------
    aligned_locs : list of pd.DataFrames
        List of aligned localization datasets, where the localizations
        have been shifted according to the average shift calculated from
        the picked localizations.
    infos : list of lists of dicts
        Metadata of each aligned channel: its ``infos`` element with an
        entry recording the alignment (the picks and the shift
        subtracted from the channel) appended, see
        ``lib.append_metadata``.
    shifts : tuple of np.ndarrays
        ``(shift_x, shift_y)``, or ``(shift_x, shift_y, shift_z)`` if
        all channels have z. Each array holds the shift
        subtracted from each dataset in ``locs``, calculated as the
        average shift from the picked localizations, in nm. Returned
        only if ``return_shifts`` is True.

    Raises
    ------
    ValueError
        If ``keep_non_overlapping`` is False and the aligned channels
        do not overlap.
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}"
    if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE:
        assert (
            pick_size is not None
        ), "pick_size must be provided when picks is a list of coordinates"
    ib_list = index_blocks if index_blocks is not None else [None] * len(locs)
    pl = [
        picked_locs(
            locs_,
            info,
            picks=picks,
            pick_shape=pick_shape,
            pick_size=pick_size,
            index_blocks=ib,
        )
        for locs_, info, ib in zip(locs, infos, ib_list)
    ]
    dy = _shifts_from_picked_coordinate(pl, coordinate="y")
    dx = _shifts_from_picked_coordinate(pl, coordinate="x")
    if all(["z" in _[0].columns for _ in pl]):
        dz = _shifts_from_picked_coordinate(pl, coordinate="z")
    else:
        dz = None
    # minimize_shifts returns (y, x[, z]); the shifts are returned
    # x first, like those of align and align_rcc
    shift_y, shift_x, *shift_z = lib.minimize_shifts(dx, dy, shifts_z=dz)
    shifts = tuple(
        np.asarray(_, dtype=float) for _ in (shift_x, shift_y, *shift_z)
    )

    # align each channel
    aligned_locs = []
    for i, locs_ in enumerate(locs):
        locs_ = locs_.copy()
        locs_["x"] -= shifts[0][i]
        locs_["y"] -= shifts[1][i]
        if len(shifts) == 3:
            locs_["z"] -= shifts[2][i]
        aligned_locs.append(locs_)
    if not keep_non_overlapping:
        aligned_locs = remove_non_overlapping_locs(aligned_locs)
    entry = _picks_entry("Align", picks, pick_shape, pick_size)
    entry["Method"] = "Picked"
    entry["Remove non-overlapping"] = not keep_non_overlapping
    infos = _aligned_infos(infos, shifts, entry)
    if return_shifts:
        return aligned_locs, infos, shifts
    return aligned_locs, infos


def overlap_region(
    locs: list[pd.DataFrame],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find the region in which all channels overlap, i.e., the
    intersection of the rectangles spanned by the x and y coordinates of
    the localizations of each channel.

    Parameters
    ----------
    locs : list of pd.DataFrames
        Localizations of each channel, e.g., after alignment (see
        ``align_rcc`` and ``align_from_picked``).

    Returns
    -------
    region : tuple or None
        ``((x_min, x_max), (y_min, y_max))`` of the overlap in nm, or
        None if the channels do not overlap (or any channel has no
        localizations).
    """
    if any(len(locs_) == 0 for locs_ in locs):
        return None
    x_min = max(locs_["x"].min() for locs_ in locs)
    x_max = min(locs_["x"].max() for locs_ in locs)
    y_min = max(locs_["y"].min() for locs_ in locs)
    y_max = min(locs_["y"].max() for locs_ in locs)
    if x_min > x_max or y_min > y_max:
        return None
    return (x_min, x_max), (y_min, y_max)


def remove_non_overlapping_locs(
    locs: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Remove the localizations that lie outside the region in which
    all channels overlap, see ``overlap_region``.

    After aligning channels (``align_rcc`` and ``align_from_picked``),
    large shifts can move parts of a channel to where the other channels
    have no localizations, for example, when the channels were imaged
    over shifted fields of view. Only x and y are considered.

    Parameters
    ----------
    locs : list of pd.DataFrames
        Localizations of each channel.

    Returns
    -------
    locs : list of pd.DataFrames
        Copies of the localizations of each channel inside the overlap
        region, with a reset index.

    Raises
    ------
    ValueError
        If the channels do not overlap.
    """
    region = overlap_region(locs)
    if region is None:
        raise ValueError(
            "The channels do not overlap, so no localizations would be"
            " kept."
        )
    (x_min, x_max), (y_min, y_max) = region
    return [
        locs_[
            locs_["x"].between(x_min, x_max) & locs_["y"].between(y_min, y_max)
        ].reset_index(drop=True)
        for locs_ in locs
    ]


def _shifts_from_picked_coordinate(
    locs: list[list[pd.DataFrame]],
    infos: None = None,
    *,
    coordinate: Literal["x", "y", "z"] = "x",
):
    """Calculate shifts between channels along a given coordinate.

    Parameters
    ----------
    locs : list of lists of pd.DataFrames
        Each element stores picked localizations from a channel, pick
        by pick, see `picked_locs`.
    infos : None
        Ignored, kept for compatibility.
    coordinate : {'x', 'y', 'z'}
        Specifies which coordinate should be used.

    Returns
    -------
    shifts : lib.FloatArray2D
        Array of shape (n_channels, n_channels) with shifts between
        all channels.
    """
    n_channels = len(locs)
    # Calculating center of mass for each channel and pick
    coms = []
    for channel_locs in locs:
        coms.append([])
        for group_locs in channel_locs:
            group_com = getattr(group_locs, coordinate).mean()
            coms[-1].append(group_com)
    # Calculating image shifts
    shifts = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            shifts[i, j] = np.nanmean(
                [cj - ci for ci, cj in zip(coms[i], coms[j])]
            )
    return shifts


def groupprops(
    locs: pd.DataFrame,
    progress: lib.Progress | Literal["console"] | None = None,
) -> pd.DataFrame:
    """Calculate group statistics for localizations, such as mean and
    standard deviation.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with a 'group' field that defines the groups.
        Binding events with ``dark == -1`` (no preceding event, see
        ``compute_dark_times``) are excluded if a 'dark' column exists.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating group properties"
        over the groups. Default is None.

    Returns
    -------
    groups : pd.DataFrame
        Group statistics for each group in the localization list.
    """
    if "dark" in locs.columns:
        locs = locs[locs["dark"] != -1]
    group_ids = np.unique(locs["group"])
    n = len(group_ids)
    names = ["group", "n_events"] + list(
        itertools.chain(*[(_ + "_mean", _ + "_std") for _ in locs.columns])
    )
    groups = pd.DataFrame(np.empty((n, len(names))), columns=names)

    with lib.tracked_progress(
        progress, "Calculating group properties", unit="group"
    ) as progress:
        lib.start_phase(
            progress, "Calculating group properties", n, unit="group"
        )
        progress.set_value(0)
        for i in range(n):
            group_id = group_ids[i]
            group_locs = locs[locs["group"] == group_id]
            groups.loc[i, "group"] = group_id
            groups.loc[i, "n_events"] = len(group_locs)
            for name in locs.columns:
                groups.loc[i, name + "_mean"] = group_locs[name].mean()
                groups.loc[i, name + "_std"] = group_locs[name].std()
            progress.set_value(i + 1)
        progress.set_value(n)

    # set dtypes
    groups = groups.astype(
        {
            "group": np.int32,
            "n_events": np.int32,
            **{name + "_mean": np.float32 for name in locs.columns},
            **{name + "_std": np.float32 for name in locs.columns},
        }
    )
    # add qpaint idx
    if "dark_mean" in groups.columns:
        groups["qpaint_idx"] = 1 / groups["dark_mean"]
    return groups


def calculate_fret(
    acc_locs: pd.DataFrame,
    don_locs: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Calculate the FRET efficiency in picked regions, for one trace.

    Parameters
    ----------
    acc_locs : pd.DataFrame
        Acceptor-channel localizations of the pick.
    don_locs : pd.DataFrame
        Donor-channel localizations of the pick.

    Returns
    -------
    fret_dict : dict
        With ``"fret_events"`` (the efficiencies in (0, 1)),
        ``"fret_timepoints"`` (their frames), ``"acc_trace"`` and
        ``"don_trace"`` (background-subtracted photons per frame),
        ``"frames"`` (the frame axis) and ``"maxframes"``.
    f_locs : pd.DataFrame or list
        The donor localizations of the FRET frames, with an added ``fret``
        column. An empty list when no frame shows FRET.
    """
    fret_dict = {}
    if len(acc_locs) == 0:
        max_frames = don_locs["frame"].max()
    elif len(don_locs) == 0:
        max_frames = acc_locs["frame"].max()
    else:
        max_frames = max([acc_locs["frame"].max(), don_locs["frame"].max()])

    # Initialize a vector filled with zeros for the duration of the movie
    xvec = np.arange(max_frames + 1)
    yvec = xvec[:] * 0
    acc_trace = yvec.copy()
    don_trace = yvec.copy()
    # Fill vector with the photon numbers of events that happend
    acc_trace[acc_locs["frame"]] = acc_locs["photons"] - acc_locs["bg"]
    don_trace[don_locs["frame"]] = don_locs["photons"] - don_locs["bg"]

    # Calculate the FRET efficiency
    fret_trace = acc_trace / (acc_trace + don_trace)
    # Only select FRET values between 0 and 1
    selector = np.logical_and(fret_trace > 0, fret_trace < 1)

    # Select the final fret events based on the 0 to 1 range
    fret_events = fret_trace[selector]
    fret_timepoints = np.arange(len(fret_trace))[selector]

    f_locs = []
    if len(fret_timepoints) > 0:
        # Calculate FRET locs: Select the locs when FRET happens
        sel_locs = []
        for element in fret_timepoints:
            sel_locs.append(don_locs[don_locs["frame"] == element])

        f_locs = pd.concat(sel_locs, ignore_index=True)
        f_locs["fret"] = np.array(fret_events)

    fret_dict["fret_events"] = np.array(fret_events)
    fret_dict["fret_timepoints"] = fret_timepoints
    fret_dict["acc_trace"] = acc_trace
    fret_dict["don_trace"] = don_trace
    fret_dict["frames"] = xvec
    fret_dict["maxframes"] = max_frames

    return fret_dict, f_locs


def nn_analysis(
    X1: lib.FloatArray2D,
    X2: lib.FloatArray2D,
    nn_count: int,
) -> lib.FloatArray2D:
    """Find the nearest neighbors between two sets of localizations.

    Parameters
    ----------
    X1, X2 : lib.FloatArray2D
        Arrays of shape (N, D) and (M, D) representing the coordinates
        of the two sets of localizations, where N and M are the number
        of localizations in each set, and D is the number of spatial
        dimensions (2 or 3).
    nn_count : int
        Number of nearest neighbors to find for each localization in
        the second set.

    Returns
    -------
    nnd : lib.FloatArray2D
        Array of nearest neighbors distances, where each row corresponds
        to a localization in the first set and contains the distances to
        its nearest neighbors in the second set.
    """
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of dimensions.")
    tree = KDTree(X2)
    if np.array_equal(X1, X2):
        distances, indices = tree.query(X1, k=nn_count + 1)
        nn = distances[:, 1:]
    else:
        distances, indices = tree.query(X1, k=nn_count)
        nn = distances
    nn = nn.reshape(-1, nn_count)  # k=1 gives a flat array
    return nn


def _cbc_column(name: str) -> str:
    """Name of the column holding CBC values against a partner called
    ``name``: ``"cbc_"`` followed by the name with every run of
    characters other than letters, digits and underscores replaced by
    one underscore (e.g., ``"Cav2.1"`` gives ``"cbc_Cav2_1"``), so that
    it can be used in filter expressions. Raises ValueError if ``name``
    contains no letter, digit or underscore."""
    clean = re.sub(r"\W+", "_", str(name)).strip("_")
    if not clean:
        raise ValueError(f"Invalid CBC partner name: {name!r}.")
    return f"cbc_{clean}"


def cbc(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    partner_locs: pd.DataFrame | list[pd.DataFrame],
    partner_names: str | list[str] | None = None,
    r_max: float = 200.0,
    steps: int = 100,
    use_z: bool | None = None,
    undefined: float = 0.0,
    remove_undefined: bool = False,
    chunk_size: int = 20_000,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Coordinate-based colocalization (CBC) of localizations with one
    or more partner localization lists, following Malkusch, et al.
    (2012), see References.

    Every localization of ``locs`` gets a CBC value against each
    partner, from -1 (anticorrelated) through 0 (uncorrelated) to 1
    (colocalized). For a localization, the numbers of its own channel's
    and of the partner's localizations within radii r up to ``r_max``
    are normalized by the area (2D) or volume (3D) searched, the two
    distributions are compared with the Spearman rank correlation and
    the result is weighted by ``exp(-E / r_max)``, where E is the
    distance to the nearest partner localization.

    The CBC is not symmetric: the values of ``locs`` against a partner
    differ from the values of the partner against ``locs``. To get both
    directions, call the function with the roles swapped, or use
    ``cbc_all_pairs`` to score every channel against every other one.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations that are scored.
    info : list of dicts
        Metadata of ``locs``.
    partner_locs : pd.DataFrame or list of pd.DataFrames
        Partner localizations that ``locs`` are scored against, in the
        same coordinate system (nm) as ``locs``.
        The neighbor counts of ``locs`` within their own channel are
        shared between the partners, so passing several partners at
        once is faster than one call per partner.
    partner_names : str, list of str or None, optional
        Short names of the partners, one per partner. The CBC values
        against a partner are saved in the column ``cbc_<name>``, where
        characters other than letters, digits and underscores are
        replaced by underscores (e.g., ``"Cav2.1"`` gives
        ``"cbc_Cav2_1"``). If None, the column is called ``"cbc"`` for a
        single partner and ``"cbc_0"``, ``"cbc_1"``, ... for a list of
        partners. Existing columns of the same name are overwritten.
        Default is None.
    r_max : float, optional
        Maximum radius in nm. Default is 200.
    steps : int, optional
        Number of radii between 0 (excluded) and ``r_max``, at least
        2. Default is 100.
    use_z : bool or None, optional
        Whether to compute the CBC in 3D (x, y, z). If None, 3D is used
        when ``locs`` and all partners contain z. Default is None.
    undefined : float, optional
        CBC given to localizations without any partner localization, or
        without any localization of their own channel other than
        themselves, within ``r_max``, where the CBC is not defined.
        Default is 0. ``np.nan`` marks them as missing, but
        ``io.save_locs`` removes localizations with NaN values.
    remove_undefined : bool, optional
        If True, localizations where the CBC is not defined against at
        least one partner are removed instead of getting ``undefined``.
        Default is False.
    chunk_size : int, optional
        Number of localizations processed at once, limits memory usage.
        Default is 20,000.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating CBC" over the
        localizations of ``locs``. Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Copy of the input localizations with the new CBC column(s)
        (float32), without the localizations where the CBC is not
        defined if ``remove_undefined``.
    info : list of dicts
        Metadata of the returned localizations: ``info`` with an entry
        recording the CBC parameters appended, see
        ``lib.append_metadata``.

    Raises
    ------
    ValueError
        If the parameters are invalid, the number of partner names does
        not match the number of partners, two partner names give the
        same column, or 3D is requested for localizations without z.

    Warns
    -----
    UserWarning
        If ``locs`` or any partner is not linked (no ``len`` column).
        Unlinked localizations count every binding event several times
        and inflate the neighbor counts; link them first with
        ``postprocess.link``.

    See Also
    --------
    cbc_all_pairs : CBC of every channel against every other channel.

    References
    ----------
    Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based
    colocalization analysis of single-molecule localization microscopy
    data. Histochem Cell Biol 137, 1-10 (2012).
    https://doi.org/10.1007/s00418-011-0880-5
    """
    single = isinstance(partner_locs, pd.DataFrame)
    partners = [partner_locs] if single else list(partner_locs)
    if not partners:
        raise ValueError("At least one partner is required.")
    if partner_names is None:
        columns = (
            ["cbc"] if single else [f"cbc_{i}" for i in range(len(partners))]
        )
    else:
        if isinstance(partner_names, str):
            partner_names = [partner_names]
        partner_names = list(partner_names)
        if len(partner_names) != len(partners):
            raise ValueError(
                f"{len(partner_names)} partner names given for "
                f"{len(partners)} partners."
            )
        columns = [_cbc_column(name) for name in partner_names]
    use_z = _validate_cbc_args([locs] + partners, columns, r_max, steps, use_z)
    _warn_if_not_linked([locs] + partners)
    with lib.tracked_progress(
        progress, "Calculating CBC", unit="loc"
    ) as progress:
        lib.start_phase(progress, "Calculating CBC", len(locs), unit="loc")
        progress.set_value(0)
        new_locs, new_info = _cbc_channel(
            locs,
            info,
            partners,
            partner_names,
            columns,
            r_max,
            steps,
            use_z,
            undefined,
            remove_undefined,
            chunk_size,
            progress,
        )
        progress.set_value(len(locs))
    return new_locs, new_info


def cbc_all_pairs(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    names: list[str] | None = None,
    r_max: float = 200.0,
    steps: int = 100,
    use_z: bool | None = None,
    undefined: float = 0.0,
    remove_undefined: bool = False,
    chunk_size: int = 20_000,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[list[pd.DataFrame], list[list[dict]]]:
    """Coordinate-based colocalization (CBC) of every channel against
    every other channel, following Malkusch, et al. (2012), see
    References.

    The CBC is not symmetric, so both directions of each pair are
    computed: channel i gets one column ``cbc_<name of j>`` for every
    other channel j. See ``cbc`` for the method.

    Parameters
    ----------
    locs : list of pd.DataFrames
        Localizations of at least two channels, in the same coordinate
        system (nm).
    infos : list of list of dicts
        Metadata of each channel.
    names : list of str or None, optional
        Short names of the channels, used for the column names
        ``cbc_<name>`` (see ``cbc``). If None, the channels are called ``"0"``,
        ``"1"``, ... Default is None.
    r_max : float, optional
        Maximum radius in nm. Default is 200.
    steps : int, optional
        Number of radii between 0 (excluded) and ``r_max``, at least
        2. Default is 100.
    use_z : bool or None, optional
        Whether to compute the CBC in 3D (x, y, z). If None, 3D is used
        when all channels contain z. Default is None.
    undefined : float, optional
        CBC given to localizations without any partner localization, or
        without any localization of their own channel other than
        themselves, within ``r_max``. Default is 0. See ``cbc``.
    remove_undefined : bool, optional
        If True, localizations where the CBC is not defined against at
        least one other channel are removed instead of getting
        ``undefined``. Default is False.
    chunk_size : int, optional
        Number of localizations processed at once, limits memory usage.
        Default is 20,000.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating CBC" over the
        localizations of all channels. Default is None.

    Returns
    -------
    locs : list of pd.DataFrames
        Copy of each channel's localizations with one CBC column per
        other channel (float32), without the localizations where the
        CBC is not defined if ``remove_undefined``.
    infos : list of list of dicts
        Metadata of each channel with a new entry describing the CBC
        parameters.

    Raises
    ------
    ValueError
        If fewer than two channels are given, the numbers of channels,
        metadata and names differ, two names give the same column, or
        the parameters are invalid.

    Warns
    -----
    UserWarning
        If any channel is not linked (no ``len`` column), see ``cbc``.

    See Also
    --------
    cbc : CBC of one channel against one or more partners.

    References
    ----------
    Malkusch, S., Endesfelder, U., Mondry, J. et al. Coordinate-based
    colocalization analysis of single-molecule localization microscopy
    data. Histochem Cell Biol 137, 1-10 (2012).
    https://doi.org/10.1007/s00418-011-0880-5
    """
    n_channels = len(locs)
    if n_channels < 2:
        raise ValueError("CBC of all pairs requires at least two channels.")
    if len(infos) != n_channels:
        raise ValueError(
            f"{len(infos)} metadata given for {n_channels} channels."
        )
    if names is None:
        names = [str(i) for i in range(n_channels)]
    names = list(names)
    if len(names) != n_channels:
        raise ValueError(
            f"{len(names)} names given for {n_channels} channels."
        )
    columns = [_cbc_column(name) for name in names]
    use_z = _validate_cbc_args(locs, columns, r_max, steps, use_z)
    _warn_if_not_linked(locs)

    total = sum(len(locs_) for locs_ in locs)
    out_locs, out_infos = [], []
    offset = 0
    with lib.tracked_progress(
        progress, "Calculating CBC", unit="loc"
    ) as progress:
        lib.start_phase(progress, "Calculating CBC", total, unit="loc")
        progress.set_value(0)
        for i in range(n_channels):
            others = [j for j in range(n_channels) if j != i]
            new_locs, new_info = _cbc_channel(
                locs[i],
                infos[i],
                [locs[j] for j in others],
                [names[j] for j in others],
                [columns[j] for j in others],
                r_max,
                steps,
                use_z,
                undefined,
                remove_undefined,
                chunk_size,
                progress,
                offset,
            )
            new_info[-1]["CBC name"] = names[i]
            out_locs.append(new_locs)
            out_infos.append(new_info)
            offset += len(locs[i])
        progress.set_value(total)
    return out_locs, out_infos


def _validate_cbc_args(
    all_locs: list[pd.DataFrame],
    columns: list[str],
    r_max: float,
    steps: int,
    use_z: bool | None,
) -> bool:
    """Check the CBC parameters shared by ``cbc`` and
    ``cbc_all_pairs`` and return whether to use z."""
    if len(set(columns)) != len(columns):
        raise ValueError(
            f"The CBC column names {columns} are not unique; please use "
            "distinct names."
        )
    if r_max <= 0:
        raise ValueError("r_max must be positive.")
    if steps < 2:
        raise ValueError("steps must be at least 2.")
    has_z = all("z" in locs_.columns for locs_ in all_locs)
    if use_z is None:
        return has_z
    if use_z and not has_z:
        raise ValueError("3D CBC requires z in all localizations.")
    return bool(use_z)


def _warn_if_not_linked(all_locs: list[pd.DataFrame]) -> None:
    """Warn if any of the localizations is not linked (no ``len``
    column); the warning points at the caller of the public function."""
    if not all("len" in locs_.columns for locs_ in all_locs):
        warnings.warn(
            "The localizations are not linked (no 'len' column), so each "
            "binding event is counted several times in the CBC. Consider "
            "linking them first with picasso.postprocess.link.",
            UserWarning,
            stacklevel=3,
        )


def _cbc_channel(
    locs: pd.DataFrame,
    info: list[dict],
    partners: list[pd.DataFrame],
    partner_names: list[str] | None,
    columns: list[str],
    r_max: float,
    steps: int,
    use_z: bool,
    undefined: float,
    remove_undefined: bool,
    chunk_size: int,
    progress: lib.Progress,
    offset: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    """CBC of one channel against its partners, with validated
    parameters. See ``cbc``. ``progress`` is set to ``offset`` plus the
    number of localizations processed after each chunk."""
    coords = _cbc_coords(locs, use_z)
    partner_coords = [_cbc_coords(p, use_z) for p in partners]
    values, defined = _cbc(
        coords,
        partner_coords,
        r_max,
        steps,
        undefined,
        chunk_size,
        progress,
        offset,
    )
    locs = locs.copy()
    for column, cbc_values in zip(columns, values):
        locs[column] = cbc_values.astype(np.float32)
    n_removed = 0
    if remove_undefined:
        keep = np.logical_and.reduce(defined)
        n_removed = int(len(locs) - keep.sum())
        locs = locs[keep].reset_index(drop=True)
    new_info = {
        **lib.step_info("CBC"),
        "CBC columns": list(columns),
    }
    if partner_names is not None:
        new_info["CBC partner names"] = list(partner_names)
    new_info.update(
        {
            "Max. radius (nm)": float(r_max),
            "Radius steps": int(steps),
            "Dimensions": 3 if use_z else 2,
        }
    )
    if remove_undefined:
        new_info["Localizations without neighbors removed"] = n_removed
    else:
        new_info["Undefined CBC value"] = float(undefined)
    return locs, lib.append_metadata(info, new_info)


def _cbc_coords(locs: pd.DataFrame, use_z: bool) -> lib.FloatArray2D:
    """Coordinates of localizations in nm, (N, 2) or (N, 3) if
    ``use_z``."""
    xy = locs[["x", "y"]].to_numpy(np.float64)
    if use_z:
        return np.column_stack([xy, locs["z"].to_numpy(np.float64)])
    return xy


def _cbc(
    coords: lib.FloatArray2D,
    partner_coords: list[lib.FloatArray2D],
    r_max: float,
    steps: int,
    undefined: float,
    chunk_size: int,
    progress: lib.Progress,
    offset: int = 0,
) -> tuple[list[lib.FloatArray1D], list[lib.BoolArray1D]]:
    """CBC values of ``coords`` against each array in
    ``partner_coords`` (all in nm), and whether they are defined (the
    localization has neighbors of both channels within ``r_max``), see
    ``cbc`` for details. The localizations are processed in chunks; the
    neighbor counts within the own channel are computed once per chunk
    for all partners. After each chunk, ``progress`` is set to ``offset``
    plus the number of localizations processed."""
    n = len(coords)
    radii = np.linspace(r_max / steps, r_max, steps)  # 0 excluded
    # D(r) normalizes N(r) by the area (2D) or volume (3D) searched:
    # D(r) = [N(r) / N(r_max)] * [r_max / r]^n_dims, with r = (i + 1) *
    # r_max / steps. The Spearman correlation only uses the order of D(r)
    # within each localization, so the constant factors are left out and
    # D(r) is taken as N(r) / (i + 1)^n_dims: one correctly rounded
    # division of integers, which keeps radii with exactly equal D(r)
    # tied (several roundings break such ties at random)
    shells = np.arange(1, steps + 1, dtype=np.float64) ** coords.shape[1]

    values = [np.full(n, undefined, dtype=np.float64) for _ in partner_coords]
    defined = [np.zeros(n, dtype=bool) for _ in partner_coords]
    if n == 0:
        return values, defined
    tree = KDTree(coords)
    partner_trees = [KDTree(c) if len(c) else None for c in partner_coords]

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk_tree = KDTree(coords[start:stop])
        # A-A counts, excluding the localization itself
        n_a_r = _cbc_cumulative_counts(chunk_tree, tree, radii) - 1
        for partner_tree, cbc_values, cbc_defined in zip(
            partner_trees, values, defined
        ):
            if partner_tree is None:
                continue  # no partner localizations, CBC undefined
            n_b_r = _cbc_cumulative_counts(chunk_tree, partner_tree, radii)
            # localizations without neighbors within r_max are undefined
            valid = (n_a_r[:, -1] > 0) & (n_b_r[:, -1] > 0)
            if not valid.any():
                continue
            s = _rowwise_spearman(n_a_r[valid] / shells, n_b_r[valid] / shells)
            # weighting by the distance to the nearest partner
            idx = start + np.flatnonzero(valid)
            e_ab, _ = partner_tree.query(coords[idx], k=1)
            cbc_values[idx] = s * np.exp(-e_ab / r_max)
            cbc_defined[idx] = True
        progress.set_value(offset + stop)
    return values, defined


def _cbc_cumulative_counts(
    chunk_tree: KDTree,
    other_tree: KDTree,
    radii: lib.FloatArray1D,
) -> lib.IntArray2D:
    """Number of points of ``other_tree`` within each radius (<= r) of
    each point of ``chunk_tree``, as an (n_chunk, len(radii)) array.
    The neighbors within the largest radius are queried once and binned
    into all radii."""
    n_chunk = chunk_tree.n
    steps = len(radii)
    pairs = chunk_tree.sparse_distance_matrix(
        other_tree, radii[-1], output_type="ndarray"
    )
    # first radius at which each neighbor is counted (d <= r)
    bins = np.searchsorted(radii, pairs["v"], side="left")
    counts = np.bincount(
        pairs["i"] * steps + bins, minlength=n_chunk * steps
    ).reshape(n_chunk, steps)
    return np.cumsum(counts, axis=1, dtype=np.int32)


def _rowwise_spearman(
    x: lib.FloatArray2D, y: lib.FloatArray2D
) -> lib.FloatArray1D:
    """Spearman rank correlation between the corresponding rows of
    ``x`` and ``y``. Rows with constant values (undefined correlation)
    give 0."""
    rx = rankdata(x, axis=1)
    ry = rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    num = np.sum(rx * ry, axis=1)
    den = np.sqrt(np.sum(rx**2, axis=1) * np.sum(ry**2, axis=1))
    s = np.zeros(len(x))
    valid = den > 0
    s[valid] = num[valid] / den[valid]
    return s


# Spatial statistics: how the molecules of one species are distributed
# (clustered, random or dispersed) within regions of interest

#: Largest weight of Ripley's isotropic edge correction, as in
#: spatstat's ``edge.Ripley``. Caps the influence of a pair whose circle
#: lies almost entirely outside the region, e.g., close to a corner.
_MAX_EDGE_WEIGHT = 100.0

#: Number of angles sampled on a circle to measure the fraction of it
#: inside a brush pick (exact for the other pick shapes).
_BRUSH_ANGLES = 72

#: Gauss-Legendre nodes and weights on [-1, 1] of the integration over
#: heights in the 3D edge correction, see ``_sphere_fraction_inside``
_SPHERE_NODES, _SPHERE_WEIGHTS = np.polynomial.legendre.leggauss(16)

# kinds of regions of interest, see ``_Region``
_REGION_CIRCLE = 0
_REGION_POLYGON = 1
_REGION_BRUSH = 2

_point_segment_distance_sq = lib._point_segment_distance_sq

#: Columns that molecules (one localization per molecule) have and
#: localizations do not: written by ``clusterer.find_cluster_centers``
#: and ``g5m.g5m``
_MOLECULE_COLUMNS = ("n_events", "std_frame")


def _are_molecules(locs: pd.DataFrame) -> bool:
    """Whether ``locs`` are molecules rather than localizations, i.e.,
    contain any of ``_MOLECULE_COLUMNS``."""
    return any(column in locs.columns for column in _MOLECULE_COLUMNS)


class _Region(NamedTuple):
    """Geometry of a region of interest, in nm.

    ``kind`` is ``_REGION_CIRCLE`` (``circle`` holds the center x, y
    and the radius), ``_REGION_POLYGON`` (each row of ``segments`` is an
    edge ``x0, y0, x1, y1``, the fifth column is unused) or
    ``_REGION_BRUSH`` (each row of ``segments`` is a segment of a stroke
    path with the stroke's radius in the fifth column). ``bounds`` is
    ``(x_min, x_max, y_min, y_max)``. For a rectangular polygon,
    ``rectangle`` holds a corner and the two edges leaving it, ``x0,
    y0, e1x, e1y, e2x, e2y``, otherwise it is None."""

    kind: int
    circle: lib.FloatArray1D
    segments: lib.FloatArray2D
    bounds: tuple[float, float, float, float]
    area: float
    rectangle: lib.FloatArray1D | None = None


def _polygon_region(X: lib.FloatArray1D, Y: lib.FloatArray1D) -> _Region:
    """Region enclosed by the polygon with corners ``(X, Y)`` in nm,
    given open or closed (first corner repeated at the end)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) > 1 and X[0] == X[-1] and Y[0] == Y[-1]:
        X, Y = X[:-1], Y[:-1]
    segments = np.column_stack(
        [X, Y, np.roll(X, -1), np.roll(Y, -1), np.zeros(len(X))]
    )
    rectangle = None
    if len(X) == 4:
        e1 = np.array([X[1] - X[0], Y[1] - Y[0]])
        e2 = np.array([X[3] - X[0], Y[3] - Y[0]])
        scale = np.dot(e1, e1) + np.dot(e2, e2)
        opposite = np.array([X[2] - X[0], Y[2] - Y[0]]) - e1 - e2
        if (
            scale > 0
            and abs(np.dot(e1, e2)) <= 1e-9 * scale
            and np.dot(opposite, opposite) <= 1e-18 * scale
        ):
            rectangle = np.array([X[0], Y[0], *e1, *e2])
    return _Region(
        _REGION_POLYGON,
        np.zeros(3),
        segments,
        (X.min(), X.max(), Y.min(), Y.max()),
        lib.polygon_area(X, Y),
        rectangle,
    )


def _pick_region(
    pick: tuple,
    pick_shape: str,
    pick_size: float | None,
) -> _Region | None:
    """Geometry of a pick (nm), or None for a pick without area (open
    polygon, empty brush pick, zero-length rectangle)."""
    if pick_shape == "Circle":
        x, y = pick
        radius = pick_size / 2
        region = _Region(
            _REGION_CIRCLE,
            np.array([x, y, radius], dtype=np.float64),
            np.zeros((0, 5)),
            (x - radius, x + radius, y - radius, y + radius),
            np.pi * radius**2,
        )
    elif pick_shape == "Brush":
        if not len(pick):
            return None
        segments = []
        for stroke in pick:
            width, X, Y = lib.brush_stroke_arrays(stroke)
            radius = width / 2
            if len(X) == 1:  # a dot
                segments.append([X[0], Y[0], X[0], Y[0], radius])
            for k in range(len(X) - 1):
                segments.append([X[k], Y[k], X[k + 1], Y[k + 1], radius])
        x_min, x_max, y_min, y_max = lib._brush_bounds(pick)
        region = _Region(
            _REGION_BRUSH,
            np.zeros(3),
            np.array(segments, dtype=np.float64),
            (x_min, x_max, y_min, y_max),
            lib.pick_areas([pick], "Brush", None)[0],
        )
    else:
        if pick_shape == "Square":
            x, y = pick
            half = pick_size / 2
            X = [x - half, x + half, x + half, x - half]
            Y = [y - half, y - half, y + half, y + half]
        elif pick_shape == "Rectangle":
            (xs, ys), (xe, ye) = pick
            X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
        elif pick_shape == "Box":
            X, Y = lib.get_pick_box_corners(pick)
        elif pick_shape == "Polygon":
            X, Y = lib.get_pick_polygon_corners(pick)
            if X is None:  # not closed
                return None
        else:
            raise ValueError(f"Unknown pick shape: {pick_shape}")
        region = _polygon_region(
            np.asarray(X, dtype=np.float64),
            np.asarray(Y, dtype=np.float64),
        )
    if not region.area > 0:
        return None
    return region


@numba.jit(nopython=True, nogil=True, cache=True)
def _in_region(
    x: float,
    y: float,
    kind: int,
    circle: lib.FloatArray1D,
    segments: lib.FloatArray2D,
) -> bool:
    """Whether the point ``(x, y)`` lies inside the region, see
    ``_Region``."""
    if kind == _REGION_CIRCLE:
        dx = x - circle[0]
        dy = y - circle[1]
        return dx * dx + dy * dy < circle[2] * circle[2]
    if kind == _REGION_POLYGON:  # ray casting
        inside = False
        for k in range(segments.shape[0]):
            x0 = segments[k, 0]
            y0 = segments[k, 1]
            x1 = segments[k, 2]
            y1 = segments[k, 3]
            if (y0 > y) != (y1 > y):
                if x < x0 + (x1 - x0) * (y - y0) / (y1 - y0):
                    inside = not inside
        return inside
    for k in range(segments.shape[0]):
        d2 = _point_segment_distance_sq(
            x,
            y,
            segments[k, 0],
            segments[k, 1],
            segments[k, 2],
            segments[k, 3],
        )
        if d2 <= segments[k, 4] ** 2:
            return True
    return False


@numba.jit(nopython=True, nogil=True, cache=True)
def _in_region_many(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    kind: int,
    circle: lib.FloatArray1D,
    segments: lib.FloatArray2D,
) -> lib.BoolArray1D:
    """Which of the points ``(x, y)`` lie inside the region."""
    inside = np.zeros(len(x), dtype=np.bool_)
    for i in range(len(x)):
        inside[i] = _in_region(x[i], y[i], kind, circle, segments)
    return inside


@numba.jit(nopython=True, nogil=True, cache=True)
def _boundary_distances(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    kind: int,
    circle: lib.FloatArray1D,
    segments: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Distance from each point inside the region to its boundary, or
    a lower bound of it for brush picks. Circles up to this radius
    around the point lie entirely inside the region."""
    distances = np.zeros(len(x))
    for i in range(len(x)):
        if kind == _REGION_CIRCLE:
            distances[i] = circle[2] - np.sqrt(
                (x[i] - circle[0]) ** 2 + (y[i] - circle[1]) ** 2
            )
        elif kind == _REGION_POLYGON:
            best = np.inf
            for k in range(segments.shape[0]):
                d2 = _point_segment_distance_sq(
                    x[i],
                    y[i],
                    segments[k, 0],
                    segments[k, 1],
                    segments[k, 2],
                    segments[k, 3],
                )
                if d2 < best:
                    best = d2
            distances[i] = np.sqrt(best)
        else:
            # depth inside the deepest stroke segment; the union of
            # the strokes reaches at least as far
            best = 0.0
            for k in range(segments.shape[0]):
                d2 = _point_segment_distance_sq(
                    x[i],
                    y[i],
                    segments[k, 0],
                    segments[k, 1],
                    segments[k, 2],
                    segments[k, 3],
                )
                depth = segments[k, 4] - np.sqrt(d2)
                if depth > best:
                    best = depth
            distances[i] = best
    return distances


@numba.jit(nopython=True, nogil=True, cache=True)
def _circle_fraction_inside(
    px: float,
    py: float,
    d: float,
    kind: int,
    circle: lib.FloatArray1D,
    segments: lib.FloatArray2D,
) -> float:
    """Fraction of the circumference of the circle of radius ``d``
    around ``(px, py)`` that lies inside the region. Exact for circles
    and polygons (from the intersections of the circle with the edges),
    sampled at ``_BRUSH_ANGLES`` angles for brush picks."""
    if kind == _REGION_CIRCLE:
        R = circle[2]
        s = np.sqrt((px - circle[0]) ** 2 + (py - circle[1]) ** 2)
        if d <= R - s:
            return 1.0
        if s == 0.0:
            return 0.0
        c = (s * s + d * d - R * R) / (2 * s * d)
        if c >= 1.0:
            return 0.0
        if c <= -1.0:
            return 1.0
        return np.arccos(c) / np.pi
    if kind == _REGION_POLYGON:
        n_edges = segments.shape[0]
        angles = np.empty(2 * n_edges)
        m = 0
        for k in range(n_edges):
            fx = segments[k, 0] - px
            fy = segments[k, 1] - py
            ex = segments[k, 2] - segments[k, 0]
            ey = segments[k, 3] - segments[k, 1]
            a = ex * ex + ey * ey
            if a == 0.0:
                continue
            b = 2 * (fx * ex + fy * ey)
            c = fx * fx + fy * fy - d * d
            disc = b * b - 4 * a * c
            if disc < 0.0:
                continue
            sq = np.sqrt(disc)
            t = (-b - sq) / (2 * a)
            if 0.0 <= t <= 1.0:
                angles[m] = np.arctan2(fy + t * ey, fx + t * ex)
                m += 1
            t = (-b + sq) / (2 * a)
            if sq > 0.0 and 0.0 <= t <= 1.0:
                angles[m] = np.arctan2(fy + t * ey, fx + t * ex)
                m += 1
        if m == 0:  # entirely inside or outside
            if _in_region(px + d, py, kind, circle, segments):
                return 1.0
            return 0.0
        angles = np.sort(angles[:m])
        inside = 0.0
        for k in range(m):
            a0 = angles[k]
            a1 = angles[k + 1] if k + 1 < m else angles[0] + 2 * np.pi
            if a1 <= a0:
                continue
            mid = 0.5 * (a0 + a1)
            if _in_region(
                px + d * np.cos(mid),
                py + d * np.sin(mid),
                kind,
                circle,
                segments,
            ):
                inside += a1 - a0
        return inside / (2 * np.pi)
    n_inside = 0
    for k in range(_BRUSH_ANGLES):
        angle = (k + 0.5) * 2 * np.pi / _BRUSH_ANGLES
        if _in_region(
            px + d * np.cos(angle),
            py + d * np.sin(angle),
            kind,
            circle,
            segments,
        ):
            n_inside += 1
    return n_inside / _BRUSH_ANGLES


@numba.jit(nopython=True, nogil=True, cache=True)
def _sphere_fraction_inside(
    px: float,
    py: float,
    pz: float,
    d: float,
    z_min: float,
    z_max: float,
    boundary_xy: float,
    kind: int,
    circle: lib.FloatArray1D,
    segments: lib.FloatArray2D,
) -> float:
    """Fraction of the surface of the sphere of radius ``d`` around
    ``(px, py, pz)`` that lies inside the region extended from
    ``z_min`` to ``z_max``. The surface of a sphere is spread evenly
    over its height (Archimedes), so the fraction is the mean over the
    heights t inside the z range of the fraction of the circle of
    radius ``sqrt(d**2 - t**2)`` inside the region in xy, integrated
    by Gauss-Legendre quadrature. Circles up to ``boundary_xy`` lie
    entirely inside the region in xy."""
    t_low = max(-d, z_min - pz)
    t_high = min(d, z_max - pz)
    if t_high <= t_low:
        return 0.0
    if d <= boundary_xy:  # only the z planes cut the sphere
        return (t_high - t_low) / (2 * d)
    half = 0.5 * (t_high - t_low)
    middle = 0.5 * (t_high + t_low)
    total = 0.0
    for k in range(len(_SPHERE_NODES)):
        t = middle + half * _SPHERE_NODES[k]
        radius = np.sqrt(max(d * d - t * t, 0.0))
        if radius <= boundary_xy:
            fraction = 1.0
        else:
            fraction = _circle_fraction_inside(
                px, py, radius, kind, circle, segments
            )
        total += _SPHERE_WEIGHTS[k] * fraction
    return total * half / (2 * d)


@numba.jit(nopython=True, nogil=True, cache=True)
def _pair_counts(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    boundary: lib.FloatArray1D,
    boundary_xy: lib.FloatArray1D,
    cell_x: lib.IntArray1D,
    cell_y: lib.IntArray1D,
    tx: lib.FloatArray1D,
    ty: lib.FloatArray1D,
    tz: lib.FloatArray1D,
    cell_starts: lib.IntArray1D,
    n_cells_x: int,
    n_cells_y: int,
    r_max: float,
    steps: int,
    edge_correction: bool,
    use_z: bool,
    z_min: float,
    z_max: float,
    kind: int,
    circle: lib.FloatArray1D,
    segments: lib.FloatArray2D,
    exclude_self: bool,
    start: int,
    end: int,
) -> lib.FloatArray1D:
    """Edge-corrected counts of the pairs of a center i in ``[start,
    end)`` (``x``, ``y``, ``z``) and a target j (``tx``, ``ty``,
    ``tz``) by distance (3D if ``use_z``, the region then extends from
    ``z_min`` to ``z_max``): element k sums the weights of the pairs at
    distances in ``(k, k + 1] * r_max / steps``, the weight being
    centered on i. The targets are sorted by grid cell in xy (side at
    least ``r_max``), ``cell_starts`` holds the first target of each
    cell and ``cell_x``, ``cell_y`` the cell of each center.
    ``boundary`` is the distance of each center to the region's
    boundary (in 3D including the z planes), ``boundary_xy`` that in
    xy only. If ``exclude_self``, the centers are the targets and the
    pairs (i, i) are skipped."""
    counts = np.zeros(steps)
    dr = r_max / steps
    r_max2 = r_max * r_max
    # brush picks: the sampled fraction is computed once per radius
    # step (at its center) and localization, not for every pair
    sampled = kind == _REGION_BRUSH
    fractions = np.empty(steps)
    for i in range(start, end):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        if sampled:
            fractions[:] = -1.0
        for cy in range(max(cell_y[i] - 1, 0), min(cell_y[i] + 2, n_cells_y)):
            for cx in range(
                max(cell_x[i] - 1, 0), min(cell_x[i] + 2, n_cells_x)
            ):
                cell = cy * n_cells_x + cx
                for j in range(cell_starts[cell], cell_starts[cell + 1]):
                    if exclude_self and j == i:
                        continue
                    d2 = (tx[j] - xi) ** 2 + (ty[j] - yi) ** 2
                    if use_z:
                        d2 += (tz[j] - zi) ** 2
                    if d2 > r_max2:
                        continue
                    d = np.sqrt(d2)
                    k = int(np.ceil(d / dr)) - 1
                    if k < 0:
                        k = 0
                    elif k >= steps:
                        k = steps - 1
                    weight = 1.0
                    if edge_correction and d > boundary[i]:
                        if sampled and fractions[k] >= 0.0:
                            fraction = fractions[k]
                        else:
                            radius = (k + 0.5) * dr if sampled else d
                            if use_z:
                                fraction = _sphere_fraction_inside(
                                    xi,
                                    yi,
                                    zi,
                                    radius,
                                    z_min,
                                    z_max,
                                    boundary_xy[i],
                                    kind,
                                    circle,
                                    segments,
                                )
                            else:
                                fraction = _circle_fraction_inside(
                                    xi, yi, radius, kind, circle, segments
                                )
                            if sampled:
                                fractions[k] = fraction
                        if fraction * _MAX_EDGE_WEIGHT <= 1.0:
                            weight = _MAX_EDGE_WEIGHT
                        else:
                            weight = 1.0 / fraction
                    counts[k] += weight
    return counts


class _GridPoints(NamedTuple):
    """Points of a region in nm sorted by the cell of a grid in xy,
    see ``_grid_points``."""

    x: lib.FloatArray1D
    y: lib.FloatArray1D
    z: lib.FloatArray1D
    cell_x: lib.IntArray1D
    cell_y: lib.IntArray1D
    cell_starts: lib.IntArray1D
    n_cells_x: int
    n_cells_y: int
    boundary: lib.FloatArray1D
    boundary_xy: lib.FloatArray1D


def _grid_points(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D | None,
    region: _Region,
    z_range: tuple[float, float] | None,
    r_max: float,
) -> _GridPoints:
    """Sort the points ``(x, y, z)`` (nm, inside ``region``; z None in
    2D) by the cell of a grid over the region with cells of at least
    ``r_max``, so that the neighbors within ``r_max`` of a point are in
    the adjacent cells, and find their distances to the boundary of
    the region (in 3D extended over ``z_range``)."""
    use_z = z is not None
    if not use_z:
        z = np.zeros(len(x))
    x_min, x_max, y_min, y_max = region.bounds
    # at most ~1000 cells per side
    cell_size = max(r_max, (x_max - x_min) / 1024, (y_max - y_min) / 1024)
    n_cells_x = int((x_max - x_min) // cell_size) + 1
    n_cells_y = int((y_max - y_min) // cell_size) + 1
    cell_x = np.clip(
        ((x - x_min) // cell_size).astype(np.int64), 0, n_cells_x - 1
    )
    cell_y = np.clip(
        ((y - y_min) // cell_size).astype(np.int64), 0, n_cells_y - 1
    )
    cell = cell_y * n_cells_x + cell_x
    order = np.argsort(cell, kind="stable")
    x = np.ascontiguousarray(x[order], dtype=np.float64)
    y = np.ascontiguousarray(y[order], dtype=np.float64)
    z = np.ascontiguousarray(z[order], dtype=np.float64)
    cell_starts = np.searchsorted(
        cell[order], np.arange(n_cells_x * n_cells_y + 1)
    )
    boundary_xy = _boundary_distances(
        x, y, region.kind, region.circle, region.segments
    )
    if use_z:
        z_min, z_max = z_range
        boundary = np.minimum(boundary_xy, np.minimum(z - z_min, z_max - z))
    else:
        boundary = boundary_xy
    return _GridPoints(
        x,
        y,
        z,
        cell_x[order],
        cell_y[order],
        cell_starts,
        n_cells_x,
        n_cells_y,
        boundary,
        boundary_xy,
    )


def _weighted_pair_counts(
    centers: _GridPoints,
    targets: _GridPoints,
    region: _Region,
    z_range: tuple[float, float] | None,
    r_max: float,
    steps: int,
    edge_correction: bool,
    exclude_self: bool,
) -> lib.FloatArray1D:
    """Edge-corrected counts of the pairs of ``centers`` and
    ``targets`` (the same object if ``exclude_self``) in the ``steps``
    shells up to ``r_max``, see ``_pair_counts``; in 3D if
    ``z_range`` is given."""
    use_z = z_range is not None
    z_min, z_max = z_range if use_z else (0.0, 0.0)
    args = (
        centers.x,
        centers.y,
        centers.z,
        centers.boundary,
        centers.boundary_xy,
        centers.cell_x,
        centers.cell_y,
        targets.x,
        targets.y,
        targets.z,
        targets.cell_starts,
        targets.n_cells_x,
        targets.n_cells_y,
        float(r_max),
        int(steps),
        bool(edge_correction),
        use_z,
        float(z_min),
        float(z_max),
        region.kind,
        region.circle,
        region.segments,
        exclude_self,
    )
    n = len(centers.x)
    n_threads = lib.n_workers() if n > 1_000 else 1
    if n_threads > 1:
        bounds = np.linspace(0, n, n_threads + 1).astype(int)
        with _ThreadPoolExecutor(n_threads) as executor:
            futures = [
                executor.submit(_pair_counts, *args, start, end)
                for start, end in zip(bounds[:-1], bounds[1:])
            ]
            return np.sum([f.result() for f in futures], axis=0)
    return _pair_counts(*args, 0, n)


def _pair_statistic_region(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D | None,
    region: _Region,
    z_range: tuple[float, float] | None,
    r_max: float,
    steps: int,
    edge_correction: bool,
    statistic: Literal["h", "g"],
    partner: tuple | None = None,
) -> lib.FloatArray1D:
    """Ripley's H function (``statistic="h"``) at the radii ``(1, ...,
    steps) * r_max / steps`` or the radial distribution function
    (``"g"``) in the ``steps`` shells between them of the points
    ``(x, y)`` (nm, all inside ``region``); in 3D if ``z`` is given,
    the region then extends over ``z_range`` (nm), which contains all
    ``z``. With ``partner`` points ``(x, y, z)``, the cross function
    between both sets of points, combining both directions (Lotwick
    and Silverman, 1982); otherwise at least two points are needed."""
    use_z = z is not None
    z_range = tuple(z_range) if use_z else None
    first = _grid_points(x, y, z, region, z_range, r_max)
    kwargs = dict(
        region=region,
        z_range=z_range,
        r_max=r_max,
        steps=steps,
        edge_correction=edge_correction,
    )
    size = region.area * (z_range[1] - z_range[0]) if use_z else region.area
    n1 = len(x)
    if partner is None:
        counts = _weighted_pair_counts(
            first, first, exclude_self=True, **kwargs
        )
        # counts per pair of localizations, relative to the density
        counts *= size / (n1 * (n1 - 1))
    else:
        second = _grid_points(*partner, region, z_range, r_max)
        n2 = len(partner[0])
        counts_12 = _weighted_pair_counts(
            first, second, exclude_self=False, **kwargs
        )
        counts_21 = _weighted_pair_counts(
            second, first, exclude_self=False, **kwargs
        )
        # the edge correction centers on the first point of a pair, so
        # both directions differ slightly; weighted by the number of
        # points of the other set (Lotwick and Silverman, 1982)
        counts = (
            size * (n2 * counts_12 + n1 * counts_21) / ((n1 + n2) * n1 * n2)
        )
    r = np.arange(steps + 1) * (r_max / steps)  # shell edges
    if statistic == "g":
        if use_z:
            shells = 4 / 3 * np.pi * np.diff(r**3)
        else:
            shells = np.pi * np.diff(r**2)
        return counts / shells
    k_function = np.cumsum(counts)
    if use_z:
        return np.cbrt(3 * k_function / (4 * np.pi)) - r[1:]
    return np.sqrt(k_function / np.pi) - r[1:]


def _locs_in_region(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    region: _Region,
) -> lib.IntArray1D:
    """Indices of the points ``(x, y)`` (nm) that lie inside
    ``region``."""
    x_min, x_max, y_min, y_max = region.bounds
    near = np.flatnonzero(
        (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    )
    inside = _in_region_many(
        x[near], y[near], region.kind, region.circle, region.segments
    )
    return near[inside]


def _uniform_in_region(
    n: int,
    region: _Region,
    rng: np.random.Generator,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """``n`` points drawn uniformly at random inside ``region``
    (complete spatial randomness), by rejection from the bounding
    box."""
    x_min, x_max, y_min, y_max = region.bounds
    box_area = (x_max - x_min) * (y_max - y_min)
    acceptance = min(max(region.area / box_area, 1e-3), 1.0)
    xs, ys = [], []
    n_found = 0
    while n_found < n:
        m = int((n - n_found) / acceptance * 1.1) + 16
        x = rng.uniform(x_min, x_max, m)
        y = rng.uniform(y_min, y_max, m)
        inside = _in_region_many(
            x, y, region.kind, region.circle, region.segments
        )
        xs.append(x[inside])
        ys.append(y[inside])
        n_found += int(inside.sum())
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def _random_shift(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D | None,
    region: _Region,
    z_range: tuple[float, float] | None,
    rng: np.random.Generator,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D, lib.FloatArray1D | None]:
    """Shift the points ``(x, y, z)`` (nm) inside ``region`` by a random
    offset, wrapping around a rectangular region (toroidal shift), or
    rotate them by a random angle around the center of a circular one;
    in 3D, also shift z by a random offset, wrapping around
    ``z_range``. Keeps the arrangement of the points among themselves,
    so that two channels are made independent of each other."""
    if region.rectangle is not None:
        x0, y0, e1x, e1y, e2x, e2y = region.rectangle
        dx, dy = x - x0, y - y0
        a = (dx * e1x + dy * e1y) / (e1x**2 + e1y**2)
        b = (dx * e2x + dy * e2y) / (e2x**2 + e2y**2)
        a = (a + rng.uniform()) % 1.0
        b = (b + rng.uniform()) % 1.0
        x = x0 + a * e1x + b * e2x
        y = y0 + a * e1y + b * e2y
    elif region.kind == _REGION_CIRCLE:
        cx, cy, _ = region.circle
        angle = rng.uniform(0, 2 * np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        dx, dy = x - cx, y - cy
        x = cx + cos * dx - sin * dy
        y = cy + sin * dx + cos * dy
    else:
        raise ValueError(_SHIFT_REGIONS_ERROR)
    if z is not None:
        z_min, z_max = z_range
        depth = z_max - z_min
        z = z_min + (z - z_min + rng.uniform(0, depth)) % depth
    return x, y, z


#: Error for random shifts in regions that cannot be wrapped around
_SHIFT_REGIONS_ERROR = (
    "Random shifts of the partner need rectangular regions (box, square "
    "or rectangle picks, rectangular polygons, the current view or the "
    "field of view) or circular picks; use cross_null='labels' for other "
    "polygon picks and brush picks."
)


def _statistic_regions(
    info: list[dict],
    picks: list[tuple] | None,
    pick_shape: str | None,
    pick_size: float | None,
    shift: bool,
) -> list[_Region | None]:
    """The regions of interest in nm: the picks (None for those
    without area) or, if ``picks`` is None, the field of view. Raises
    ValueError for invalid picks and, if ``shift`` (random shifts of a
    channel are requested), for regions other than rectangles and
    circles."""
    if picks is None:
        width, height = lib.field_of_view(info)
        regions = [
            _polygon_region(
                np.array([0, width, width, 0], dtype=np.float64),
                np.array([0, 0, height, height], dtype=np.float64),
            )
        ]
    else:
        if pick_shape not in lib.PICK_SHAPES:
            raise ValueError(
                f"Invalid pick shape: {pick_shape}. Choose one of "
                f"{lib.PICK_SHAPES}."
            )
        if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE and not pick_size:
            raise ValueError(f"{pick_shape} picks require a pick size.")
        regions = [_pick_region(pick, pick_shape, pick_size) for pick in picks]
    if shift:
        for region in regions:
            if (
                region is not None
                and region.rectangle is None
                and region.kind != _REGION_CIRCLE
            ):
                raise ValueError(_SHIFT_REGIONS_ERROR)
    return regions


def _warn_if_not_molecules(
    locs: pd.DataFrame, subject: str, stacklevel: int
) -> None:
    """Warn if ``locs`` are not molecules; ``subject`` starts the
    message (e.g., "The localizations are")."""
    if not _are_molecules(locs):
        warnings.warn(
            f"{subject} not molecules (no 'n_events' or 'std_frame' "
            "column), so each molecule is counted as many times as it was "
            "localized, which appears as clustering at short distances. "
            "Consider finding the molecules first, e.g., with "
            "picasso.g5m.g5m or picasso.clusterer.find_cluster_centers.",
            UserWarning,
            stacklevel=stacklevel,
        )


def _pair_statistic(
    locs: pd.DataFrame,
    info: list[dict],
    picks: list[tuple] | None,
    pick_shape: str | None,
    pick_size: float | None,
    statistic: Literal["h", "g"],
    *,
    r_max: float,
    steps: int,
    edge_correction: bool,
    n_simulations: int,
    use_z: bool,
    z_range: tuple[float, float] | None,
    simulated_z: Literal["uniform", "observed"],
    partner_locs: pd.DataFrame | None,
    cross_null: Literal["shift", "labels"],
    random_seed: int | None,
    progress: lib.Progress,
) -> dict:
    """Ripley's H function (``statistic="h"``) or the radial
    distribution function (``"g"``) of localizations in regions of
    interest, or the cross function with ``partner_locs``, with the
    envelope of the simulations; shared by ``ripley_h`` and ``rdf``,
    see there for the parameters. Returns the radii ("r"), the
    function of each region ("values") and their mean ("mean") along
    with the keys common to both results. ``progress`` (a normalized
    tracker) reports "Calculating Ripley's H" (or "Calculating RDF")
    over the regions, then "Simulating envelope" over the
    simulations."""
    cross = partner_locs is not None
    if r_max <= 0:
        raise ValueError("r_max must be positive.")
    if steps < 2:
        raise ValueError("steps must be at least 2.")
    if n_simulations < 0:
        raise ValueError("n_simulations must not be negative.")
    if cross and cross_null not in ("shift", "labels"):
        raise ValueError(
            f"Invalid cross_null: {cross_null!r}; choose 'shift' or "
            "'labels'."
        )
    if use_z:
        if "z" not in locs.columns or (
            cross and "z" not in partner_locs.columns
        ):
            raise ValueError("3D requires z in the localizations.")
        if z_range is not None and not z_range[0] < z_range[1]:
            raise ValueError("z_range must be (z_min, z_max), z_min < z_max.")
        if not cross and simulated_z not in ("uniform", "observed"):
            raise ValueError(
                f"Invalid simulated_z: {simulated_z!r}; choose 'uniform' "
                "or 'observed'."
            )
    regions = _statistic_regions(
        info, picks, pick_shape, pick_size, cross and cross_null == "shift"
    )
    for locs_, which in ((locs, "localizations"), (partner_locs, "partner")):
        if locs_ is not None:
            _warn_if_not_molecules(locs_, f"The {which} are", stacklevel=4)

    # coordinates in nm, z None in 2D, of each channel
    channels = [locs] + ([partner_locs] if cross else [])
    coordinates = [
        (
            locs_["x"].to_numpy(dtype=np.float64),
            locs_["y"].to_numpy(dtype=np.float64),
            locs_["z"].to_numpy(dtype=np.float64) if use_z else None,
        )
        for locs_ in channels
    ]
    dr = r_max / steps
    if statistic == "h":
        r = np.arange(1, steps + 1) * dr
    else:  # centers of the shells
        r = (np.arange(steps) + 0.5) * dr
    n_regions = len(regions)
    values = np.full((n_regions, steps), np.nan)
    n_locs = np.zeros((len(channels), n_regions), dtype=int)
    areas = np.zeros(n_regions)
    z_ranges = np.full((n_regions, 2), np.nan)
    # coordinates of each channel's localizations in each region
    region_points = [None] * n_regions
    label = "Calculating Ripley's H" if statistic == "h" else "Calculating RDF"
    lib.start_phase(progress, label, n_regions, unit="region")
    progress.set_value(0)
    for k, region in enumerate(regions):
        if region is not None:
            areas[k] = region.area
            points = []
            for x_all, y_all, z_all in coordinates:
                inside = _locs_in_region(x_all, y_all, region)
                if use_z and z_range is not None:
                    z = z_all[inside]
                    inside = inside[(z >= z_range[0]) & (z <= z_range[1])]
                points.append(
                    (
                        x_all[inside],
                        y_all[inside],
                        z_all[inside] if use_z else None,
                    )
                )
            n_locs[:, k] = [len(x) for x, _, _ in points]
            if use_z and z_range is not None:
                z_ranges[k] = z_range
            elif use_z and n_locs[:, k].sum():
                z = np.concatenate([z for _, _, z in points])
                z_ranges[k] = z.min(), z.max()
            region_points[k] = points
            # in 3D, the region needs a depth
            has_size = not use_z or z_ranges[k, 1] > z_ranges[k, 0]
            enough = n_locs[:, k].min() >= 1 if cross else n_locs[0, k] >= 2
            if enough and has_size:
                values[k] = _pair_statistic_region(
                    *points[0],
                    region,
                    tuple(z_ranges[k]),
                    r_max,
                    steps,
                    edge_correction,
                    statistic,
                    points[1] if cross else None,
                )
        progress.set_value(k + 1)
    progress.set_value(n_regions)
    valid = np.flatnonzero(~np.isnan(values[:, 0]))
    if not len(valid):
        raise ValueError(
            (
                "No region contains localizations of both channels"
                if cross
                else "No region contains at least two localizations"
            )
            + (" at different z." if use_z else ".")
        )
    mean = values[valid].mean(axis=0)
    if use_z:
        depths = z_ranges[valid, 1] - z_ranges[valid, 0]
        n_thin = int((3 * r_max > depths).sum())
        if n_thin:
            warnings.warn(
                f"r_max ({r_max:g} nm) exceeds a third of the z range "
                f"of {n_thin} of {len(valid)} regions (smallest z "
                f"range: {depths.min():.0f} nm); {statistic.upper()} at "
                "large radii relies on large edge correction weights and "
                "is noisy.",
                UserWarning,
                stacklevel=3,
            )

    envelope_low = envelope_high = None
    if n_simulations:
        rng = np.random.default_rng(random_seed)
        simulated = np.zeros((n_simulations, steps))
        lib.start_phase(
            progress,
            "Simulating envelope",
            n_simulations,
            unit="simulation",
        )
        progress.set_value(0)
        for s in range(n_simulations):
            for k in valid:
                region = regions[k]
                z_range_k = tuple(z_ranges[k])
                points = _simulated_points(
                    region_points[k],
                    region,
                    z_range_k,
                    cross,
                    cross_null,
                    simulated_z,
                    rng,
                )
                simulated[s] += _pair_statistic_region(
                    *points[0],
                    region,
                    z_range_k,
                    r_max,
                    steps,
                    edge_correction,
                    statistic,
                    points[1] if cross else None,
                )
            simulated[s] /= len(valid)
            progress.set_value(s + 1)
        progress.set_value(n_simulations)
        envelope_low = simulated.min(axis=0)
        envelope_high = simulated.max(axis=0)

    return {
        "r": r,
        "values": values,
        "mean": mean,
        "envelope_low": envelope_low,
        "envelope_high": envelope_high,
        "n_locs": n_locs[0],
        "n_locs_partner": n_locs[1] if cross else None,
        "areas": areas,
        "z_ranges": z_ranges if use_z else None,
        "volumes": (
            areas * (z_ranges[:, 1] - z_ranges[:, 0]) if use_z else None
        ),
        "simulated_z": simulated_z if use_z and not cross else None,
        "cross_null": cross_null if cross else None,
    }


def _simulated_points(
    points: list[tuple],
    region: _Region,
    z_range: tuple[float, float],
    cross: bool,
    cross_null: Literal["shift", "labels"],
    simulated_z: Literal["uniform", "observed"],
    rng: np.random.Generator,
) -> list[tuple]:
    """The points ``(x, y, z)`` (z None in 2D) of each channel in a
    region for one simulation: as many uniformly random points as
    localizations (complete spatial randomness) for one channel; for
    two channels, the partner (second channel) shifted at random
    (``cross_null="shift"``) or the channels of all points shuffled
    (``"labels"``)."""
    x, y, z = points[0]
    if not cross:
        x_new, y_new = _uniform_in_region(len(x), region, rng)
        z_new = None
        if z is not None and simulated_z == "uniform":
            z_new = rng.uniform(*z_range, len(x))
        elif z is not None:
            z_new = rng.permutation(z)
        return [(x_new, y_new, z_new)]
    if cross_null == "shift":
        return [points[0], _random_shift(*points[1], region, z_range, rng)]
    # random labels: the same points, channels shuffled
    n1 = len(x)
    order = rng.permutation(n1 + len(points[1][0]))
    pooled = [
        (None if values[0] is None else np.concatenate(values)[order])
        for values in zip(points[0], points[1])
    ]
    return [
        tuple(None if v is None else v[:n1] for v in pooled),
        tuple(None if v is None else v[n1:] for v in pooled),
    ]


def ripley_h(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple] | None = None,
    pick_shape: (
        Literal["Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"]
        | None
    ) = None,
    pick_size: float | None = None,
    r_max: float = 500.0,
    steps: int = 100,
    edge_correction: bool = True,
    n_simulations: int = 99,
    use_z: bool = False,
    z_range: tuple[float, float] | None = None,
    simulated_z: Literal["uniform", "observed"] = "uniform",
    partner_locs: pd.DataFrame | None = None,
    cross_null: Literal["shift", "labels"] = "shift",
    random_seed: int | None = 0,
    progress: lib.Progress | Literal["console"] | None = None,
) -> dict:
    """Ripley's H function of localizations in regions of interest,
    which shows whether molecules are clustered, randomly distributed
    or dispersed, and on which length scales.

    Ripley's K function counts the neighbors within a distance r of
    each localization, normalized by the density (Ripley, 1977, see
    References)::

        K(r) = A / (n (n - 1)) * sum_i sum_(j != i) w_ij * 1(d_ij <= r)

    where A is the area of the region, n the number of localizations
    in it, d_ij the distance between localizations i and j and w_ij the
    edge correction weight. The H function, H(r) = sqrt(K(r) / pi) - r,
    is 0 for complete spatial randomness (CSR), positive for clustering
    and negative for dispersion at the scale r.

    By default, the function is computed in 2D (x and y). With
    ``use_z``, it is computed in 3D: each region extends over a z range
    (by default that of its localizations), A is replaced by the
    volume and H(r) = (3 K(r) / (4 pi))^(1/3) - r.

    The function is computed in each region separately, and the mean
    over the regions with at least two localizations (in 3D, at
    different z) is reported. Each region is simulated
    ``n_simulations`` times with as many uniformly random points as it
    holds localizations; the lowest and highest mean H of the
    simulations at each radius give the CSR envelope.

    With ``partner_locs``, the cross H function between two channels
    is computed instead (Lotwick and Silverman, 1982): K counts the
    partner localizations around each localization, normalized by both
    densities (A / (n1 n2) instead of A / (n (n - 1))), and both
    directions are combined into one symmetric estimate. The cross H
    function is positive if the channels are found near each other
    (colocalization), 0 if they are independent and negative if they
    avoid each other (segregation). The channels must be aligned.

    Parameters
    ----------
    locs : pd.DataFrame
        Molecules, i.e., one localization per molecule, such as the
        molecules found by ``g5m.g5m`` or the cluster centers of
        ``clusterer.find_cluster_centers``, since H measures how the
        molecules are distributed.
    info : list of dicts
        Metadata of the localizations, used for the field of view if
        ``picks`` is None.
    picks : list of tuples or None, optional
        Regions of interest in nm, in the format of
        ``pick_shape`` (see ``picked_locs``). Regions should be well
        inside the area covered by the sample (e.g., the cell): the
        density is the number of localizations divided by the area of
        the region, so empty parts of a region appear as clustering. If
        None, the whole field of view (``Width`` x ``Height`` in the
        metadata) is used. Default is None.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", \
"Brush"} or None, optional
        Shape of the picks. Required if ``picks`` is given. Default is
        None.
    pick_size : float or None, optional
        Size of the picks in nm: diameter for circles, width for
        rectangles, side length for squares, None for polygons, boxes
        and brush picks. Default is None.
    r_max : float, optional
        Largest radius in nm. It should be well below the size of the
        regions; the edge correction is less reliable for radii
        approaching the region size. Default is 500.
    steps : int, optional
        Number of radii between 0 (excluded) and ``r_max``, at least 2.
        Default is 100.
    edge_correction : bool, optional
        If True, Ripley's isotropic edge correction is applied: a pair
        at distance d is weighted by the inverse of the fraction of the
        circle (in 3D, the sphere surface) of radius d around the first
        localization that lies inside the region (at most 100), since
        neighbors outside the region are not seen. Without it, H is
        biased downwards near the region edges. Default is True.
    n_simulations : int, optional
        Number of CSR simulations for the envelope. With the lowest
        and highest simulated values as the envelope, a mean H outside
        of it is significant at a pointwise level of
        ``2 / (n_simulations + 1)``, i.e., 0.02 for 99 simulations. 0
        skips the simulations. Default is 99.
    use_z : bool, optional
        If True, H is computed in 3D (x, y and z in nm), which requires
        a ``z`` column. 3D samples are usually thin (about 1 um), so
        most spheres reach the top or bottom of the region and H relies
        on large edge correction weights; keep ``r_max`` below about a
        third of the z range (a warning is given otherwise). The poorer
        axial precision also stretches clusters in z. Default is False.
    z_range : tuple of floats or None, optional
        ``(z_min, z_max)`` in nm over which every region extends in 3D;
        localizations outside of it are ignored. If None, each region
        extends over the z range of its localizations. Only used in 3D.
        Default is None.
    simulated_z : {"uniform", "observed"}, optional
        z of the simulated points in 3D. "uniform" spreads them evenly
        over the z range, which is complete spatial randomness in 3D;
        an uneven z distribution of the localizations (e.g., from the
        detection falling off away from focus) then appears as
        clustering. "observed" shuffles the z values of the
        localizations in the region instead, which tests for clustering
        beyond the z distribution but also hides real clustering in z
        (e.g., a layer of molecules). Only used in 3D without
        ``partner_locs``. Default is "uniform".
    partner_locs : pd.DataFrame or None, optional
        Molecules of a second channel, in the same coordinates as
        ``locs`` (aligned). If given, the cross function between both
        channels is computed in the same regions. Default is None.
    cross_null : {"shift", "labels"}, optional
        How the envelope is simulated with ``partner_locs``. "shift"
        shifts the partner by a random offset, wrapping around the
        region (toroidal shift; in circles, a random rotation; in 3D,
        also a random shift in z), which keeps the arrangement of each
        channel and tests whether the channels are independent. It
        requires rectangular regions (box, square or rectangle picks,
        rectangular polygons, the current view or the field of view)
        or circular picks. "labels" shuffles which channel each
        localization belongs to, which works in any region and tests
        whether both channels follow the same distribution; a channel
        whose molecules are clustered more strongly than the other's
        then also differs from it. Default is "shift".
    random_seed : int or None, optional
        Seed of the CSR simulations. None gives a different envelope
        on each call. Default is 0.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating Ripley's H" over
        the regions, then (if ``n_simulations > 0``) "Simulating
        envelope" over the simulations. Default is None.

    Returns
    -------
    result : dict
        Dictionary with the keys:

        - "r": radii in nm, shape (steps,).
        - "h": H function of each region in nm, shape (n_regions,
          steps); NaN for regions with fewer than two localizations or
          without area (e.g., open polygons).
        - "h_mean": mean H function over the regions with at least two
          localizations, in nm.
        - "envelope_low", "envelope_high": CSR envelope of the mean H
          function in nm, or None without simulations.
        - "n_locs": number of localizations in each region.
        - "n_locs_partner": number of partner localizations in each
          region, None without ``partner_locs``.
        - "areas": area of each region in nm^2.
        - "z_ranges": z range of each region in nm, shape (n_regions,
          2), NaN for regions without localizations; None in 2D.
        - "volumes": volume of each region in nm^3; None in 2D.
        - "r_h_max": radius of maximum H of each region in nm (NaN
          where H is NaN).
        - "n_simulations": number of CSR simulations.
        - "edge_correction": whether the edge correction was applied.
        - "use_z": whether H was computed in 3D.
        - "simulated_z": z of the simulated points in 3D, None in 2D
          and with ``partner_locs``.
        - "cross_null": how the envelope was simulated with
          ``partner_locs``, None without.

    Raises
    ------
    ValueError
        If the parameters are invalid, z is missing in 3D, no region
        contains at least two localizations (in 3D, at different z; with
        ``partner_locs``, localizations of both channels) or random
        shifts are requested in regions other than rectangles and
        circles.

    Warns
    -----
    UserWarning
        If ``locs`` or ``partner_locs`` are not molecules (no
        ``n_events`` or ``std_frame`` column). Each molecule is then
        counted as many times as it was localized, which appears as
        clustering at short distances. In 3D, also if ``r_max`` exceeds
        a third of the z range of a region.

    See Also
    --------
    plot_ripley_h : Plot the result.

    References
    ----------
    Ripley, B. D. Modelling spatial patterns. J R Stat Soc B 39,
    172-192 (1977). https://doi.org/10.1111/j.2517-6161.1977.tb01615.x

    Lotwick, H. W., Silverman, B. W. Methods for analysing spatial
    processes of several types of points. J R Stat Soc B 44, 406-413
    (1982). https://doi.org/10.1111/j.2517-6161.1982.tb01221.x
    """
    with lib.tracked_progress(
        progress, "Calculating Ripley's H", unit="region"
    ) as progress:
        result = _pair_statistic(
            locs,
            info,
            picks,
            pick_shape,
            pick_size,
            "h",
            r_max=r_max,
            steps=steps,
            edge_correction=edge_correction,
            n_simulations=n_simulations,
            use_z=use_z,
            z_range=z_range,
            simulated_z=simulated_z,
            partner_locs=partner_locs,
            cross_null=cross_null,
            random_seed=random_seed,
            progress=progress,
        )
    h = result["values"]
    valid = ~np.isnan(h[:, 0])
    r_h_max = np.full(len(h), np.nan)
    r_h_max[valid] = result["r"][np.argmax(h[valid], axis=1)]
    return {
        "r": result["r"],
        "h": h,
        "h_mean": result["mean"],
        "envelope_low": result["envelope_low"],
        "envelope_high": result["envelope_high"],
        "n_locs": result["n_locs"],
        "n_locs_partner": result["n_locs_partner"],
        "areas": result["areas"],
        "z_ranges": result["z_ranges"],
        "volumes": result["volumes"],
        "r_h_max": r_h_max,
        "n_simulations": n_simulations,
        "edge_correction": edge_correction,
        "use_z": use_z,
        "simulated_z": result["simulated_z"],
        "cross_null": result["cross_null"],
    }


def rdf(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    picks: list[tuple] | None = None,
    pick_shape: (
        Literal["Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"]
        | None
    ) = None,
    pick_size: float | None = None,
    r_max: float = 500.0,
    steps: int = 50,
    edge_correction: bool = True,
    n_simulations: int = 99,
    use_z: bool = False,
    z_range: tuple[float, float] | None = None,
    simulated_z: Literal["uniform", "observed"] = "uniform",
    partner_locs: pd.DataFrame | None = None,
    cross_null: Literal["shift", "labels"] = "shift",
    random_seed: int | None = 0,
    progress: lib.Progress | Literal["console"] | None = None,
) -> dict:
    """Radial distribution function (RDF), also called pair
    correlation function, g(r), of localizations in regions of
    interest: the density of neighbors at the distance r from a
    localization, relative to the mean density.

    g(r) is 1 for complete spatial randomness (CSR), above 1 at the
    distances where molecules are more likely to be found than at
    random (e.g., within clusters or at a regular spacing) and below 1
    where they are depleted. Unlike Ripley's H function (``ripley_h``),
    which accumulates all neighbors up to r, g(r) counts the neighbors
    in a thin ring (in 3D, a shell) at r::

        g(r) = A / (n (n - 1)) * sum_i sum_(j != i) w_ij
               * 1(r - dr / 2 < d_ij <= r + dr / 2) / (ring area)

    with the area A of the region (in 3D, the volume), n localizations
    in it, the distance d_ij between localizations i and j, the edge
    correction weight w_ij and the ring width dr = ``r_max / steps``.
    The ring area is 2 pi r dr (in 3D, the shell volume 4 pi r^2 dr).

    The regions, the edge correction, 3D, the CSR envelope and the
    cross function between two channels (``partner_locs``) are the
    same as for ``ripley_h``. The cross g(r) is the density of partner
    localizations at the distance r from a localization, relative to
    the mean density of the partner: above 1 at the distances where
    the channels are found together, 1 if they are independent. Since
    each ring holds fewer pairs than the disk of Ripley's K function,
    g(r) is noisier; widen the rings (fewer ``steps``) for regions
    with few localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Molecules, i.e., one localization per molecule, such as the
        molecules found by ``g5m.g5m`` or the cluster centers of
        ``clusterer.find_cluster_centers``. Repeated localizations of
        a molecule appear as a peak at short distances.
    info : list of dicts
        Metadata of the localizations, used for the field of view if
        ``picks`` is None.
    picks : list of tuples or None, optional
        Regions of interest in nm, in the format of
        ``pick_shape``. If None, the whole field of view is used. See
        ``ripley_h``. Default is None.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", \
"Brush"} or None, optional
        Shape of the picks. Required if ``picks`` is given. Default is
        None.
    pick_size : float or None, optional
        Size of the picks in nm: diameter for circles, width for
        rectangles, side length for squares, None for polygons, boxes
        and brush picks. Default is None.
    r_max : float, optional
        Largest radius in nm. It should be well below the size of the
        regions. Default is 500.
    steps : int, optional
        Number of rings between 0 and ``r_max``, at least 2. Default is
        50, i.e., rings of 10 nm.
    edge_correction : bool, optional
        If True, Ripley's isotropic edge correction is applied, see
        ``ripley_h``. Without it, g is biased downwards, increasingly
        with the radius. Default is True.
    n_simulations : int, optional
        Number of CSR simulations for the envelope, see ``ripley_h``.
        0 skips the simulations. Default is 99.
    use_z : bool, optional
        If True, g is computed in 3D (x, y and z in nm), which requires
        a ``z`` column, see ``ripley_h``. Default is False.
    z_range : tuple of floats or None, optional
        ``(z_min, z_max)`` in nm over which every region extends in 3D.
        If None, each region extends over the z range of its
        localizations. Only used in 3D. Default is None.
    simulated_z : {"uniform", "observed"}, optional
        z of the simulated points in 3D, see ``ripley_h``. Only used in
        3D without ``partner_locs``. Default is "uniform".
    partner_locs : pd.DataFrame or None, optional
        Molecules of a second channel, in the same coordinates as
        ``locs`` (aligned). If given, the cross function between both
        channels is computed, see ``ripley_h``. Default is None.
    cross_null : {"shift", "labels"}, optional
        How the envelope is simulated with ``partner_locs``: random
        shifts of the partner (independent channels; rectangular and
        circular regions only) or random labels (both channels follow
        the same distribution), see ``ripley_h``. Default is "shift".
    random_seed : int or None, optional
        Seed of the CSR simulations. None gives a different envelope
        on each call. Default is 0.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating RDF" over the
        regions, then (if ``n_simulations > 0``) "Simulating envelope"
        over the simulations. Default is None.

    Returns
    -------
    result : dict
        Dictionary with the keys:

        - "r": centers of the rings in nm, shape (steps,).
        - "g": radial distribution function of each region, shape
          (n_regions, steps); NaN for regions with fewer than two
          localizations (in 3D, at different z) or without area.
        - "g_mean": mean over the regions with at least two
          localizations.
        - "envelope_low", "envelope_high": CSR envelope of the mean,
          or None without simulations.
        - "n_locs", "n_locs_partner", "areas", "z_ranges", "volumes",
          "n_simulations", "edge_correction", "use_z", "simulated_z",
          "cross_null": as in ``ripley_h``.

    Raises
    ------
    ValueError
        If the parameters are invalid, z is missing in 3D, no region
        contains enough localizations or random shifts are requested in
        regions other than rectangles and circles, see ``ripley_h``.

    Warns
    -----
    UserWarning
        If ``locs`` or ``partner_locs`` are not molecules (no
        ``n_events`` or ``std_frame`` column), and in 3D if ``r_max``
        exceeds a third of the z range of a region.

    See Also
    --------
    ripley_h : Ripley's H function, the cumulative counterpart.
    plot_rdf : Plot the result.

    References
    ----------
    Ripley, B. D. Modelling spatial patterns. J R Stat Soc B 39,
    172-192 (1977). https://doi.org/10.1111/j.2517-6161.1977.tb01615.x

    Lotwick, H. W., Silverman, B. W. Methods for analysing spatial
    processes of several types of points. J R Stat Soc B 44, 406-413
    (1982). https://doi.org/10.1111/j.2517-6161.1982.tb01221.x
    """
    with lib.tracked_progress(
        progress, "Calculating RDF", unit="region"
    ) as progress:
        result = _pair_statistic(
            locs,
            info,
            picks,
            pick_shape,
            pick_size,
            "g",
            r_max=r_max,
            steps=steps,
            edge_correction=edge_correction,
            n_simulations=n_simulations,
            use_z=use_z,
            z_range=z_range,
            simulated_z=simulated_z,
            partner_locs=partner_locs,
            cross_null=cross_null,
            random_seed=random_seed,
            progress=progress,
        )
    return {
        "r": result["r"],
        "g": result["values"],
        "g_mean": result["mean"],
        "envelope_low": result["envelope_low"],
        "envelope_high": result["envelope_high"],
        "n_locs": result["n_locs"],
        "n_locs_partner": result["n_locs_partner"],
        "areas": result["areas"],
        "z_ranges": result["z_ranges"],
        "volumes": result["volumes"],
        "n_simulations": n_simulations,
        "edge_correction": edge_correction,
        "use_z": use_z,
        "simulated_z": result["simulated_z"],
        "cross_null": result["cross_null"],
    }


def plot_rdf(
    rdf_result: dict,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot the radial distribution function: the mean over the
    regions, the CSR envelope and, for several regions, each region's
    function.

    Parameters
    ----------
    rdf_result : dict
        Result of ``rdf``.
    fig : plt.Figure or None, optional
        Figure to plot on, cleared first. If None, a new figure is
        created. Default is None.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot.
    """
    r = rdf_result["r"]
    return _plot_pair_statistic(
        rdf_result,
        rdf_result["g"],
        rdf_result["g_mean"],
        fig,
        title=(
            "Cross radial distribution function"
            if rdf_result["cross_null"]
            else "Radial distribution function"
        ),
        ylabel="g(r)",
        label="g(r)",
        random_value=1.0,
        x_max=r[-1] + 0.5 * (r[1] - r[0]),
    )


def plot_ripley_h(
    ripley_result: dict,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot Ripley's H function: the mean over the regions, the CSR
    envelope and, for several regions, each region's function.

    Parameters
    ----------
    ripley_result : dict
        Result of ``ripley_h``.
    fig : plt.Figure or None, optional
        Figure to plot on, cleared first. If None, a new figure is
        created. Default is None.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot.
    """
    return _plot_pair_statistic(
        ripley_result,
        ripley_result["h"],
        ripley_result["h_mean"],
        fig,
        title=(
            "Ripley's cross H function"
            if ripley_result["cross_null"]
            else "Ripley's H function"
        ),
        ylabel="H(r) (nm)",
        label="H(r)",
        random_value=0.0,
        x_max=ripley_result["r"][-1],
    )


#: Legend of the envelope by how it was simulated (``cross_null``)
_ENVELOPE_LABELS = {
    None: "CSR envelope ({n} simulations)",
    "shift": "Independence envelope ({n} random shifts)",
    "labels": "Random labeling envelope ({n} simulations)",
}


def _plot_pair_statistic(
    result: dict,
    values: lib.FloatArray2D,
    mean: lib.FloatArray1D,
    fig: plt.Figure | None,
    *,
    title: str,
    ylabel: str,
    label: str,
    random_value: float,
    x_max: float,
) -> plt.Figure:
    """Plot the mean over the regions of ``ripley_h`` or ``rdf``
    (``mean``), the CSR envelope, each region's function (``values``)
    if there are several, and a dashed line at the value of a random
    distribution."""
    if fig is None:
        fig = plt.Figure(constrained_layout=True)
    else:
        fig.clear()
    ax = fig.add_subplot(111)
    r = result["r"]
    valid = ~np.isnan(values[:, 0])
    n_valid = int(valid.sum())
    # colors of the active style, resolved now (lines look "CN" colors
    # up only when drawn)
    data_color, envelope_color = plt.rcParams["axes.prop_cycle"].by_key()[
        "color"
    ][:2]
    if n_valid > 1:
        for k, region_values in enumerate(values[valid]):
            ax.plot(
                r,
                region_values,
                color=data_color,
                alpha=0.25,
                linewidth=0.8,
                label=f"Regions ({n_valid})" if k == 0 else None,
            )
    if result["envelope_low"] is not None:
        ax.fill_between(
            r,
            result["envelope_low"],
            result["envelope_high"],
            color=envelope_color,
            alpha=0.3,
            linewidth=0,
            label=_ENVELOPE_LABELS[result["cross_null"]].format(
                n=result["n_simulations"]
            ),
        )
    ax.plot(
        r,
        mean,
        color=data_color,
        linewidth=2,
        label="Mean" if n_valid > 1 else label,
    )
    ax.axhline(
        random_value,
        color=plt.rcParams["text.color"],
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_title(title + (" (3D)" if result["use_z"] else ""))
    ax.set_xlim(0, x_max)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return fig


def _ripley_h_tables(
    ripley_result: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The result of ``ripley_h`` as two tables: the H functions by
    radius (mean, CSR envelope and each region) and the summary of each
    region (number of localizations, area, density, maximum of H and
    its radius). Used for saving the result as .csv files."""
    curves = _pair_statistic_curves(
        ripley_result, ripley_result["h"], ripley_result["h_mean"], "h", "_nm"
    )
    regions = _pair_statistic_regions(ripley_result)
    regions["h_max_nm"] = np.max(ripley_result["h"], axis=1)
    regions["r_at_h_max_nm"] = ripley_result["r_h_max"]
    return pd.DataFrame(curves), pd.DataFrame(regions)


def _rdf_tables(rdf_result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The result of ``rdf`` as two tables: the radial distribution
    functions by radius (mean, CSR envelope and each region) and the
    summary of each region (number of localizations, area, density).
    Used for saving the result as .csv files."""
    curves = _pair_statistic_curves(
        rdf_result, rdf_result["g"], rdf_result["g_mean"], "g", ""
    )
    return pd.DataFrame(curves), pd.DataFrame(
        _pair_statistic_regions(rdf_result)
    )


def _pair_statistic_curves(
    result: dict,
    values: lib.FloatArray2D,
    mean: lib.FloatArray1D,
    name: str,
    unit: str,
) -> dict:
    """Columns of the functions by radius: the radius, the mean, the
    CSR envelope and each region's function, called ``<name>_...``
    with the suffix ``unit``."""
    curves = {"r_nm": result["r"], f"{name}_mean{unit}": mean}
    if result["envelope_low"] is not None:
        curves[f"envelope_low{unit}"] = result["envelope_low"]
        curves[f"envelope_high{unit}"] = result["envelope_high"]
    for k, region_values in enumerate(values):
        curves[f"{name}_region_{k}{unit}"] = region_values
    return curves


def _pair_statistic_regions(result: dict) -> dict:
    """Columns of the summary of each region: the number of
    localizations, area and density (per area in 2D, per volume with
    the z range and volume in 3D)."""
    n_locs = result["n_locs"]
    n_locs_partner = result["n_locs_partner"]
    regions = {
        "region": np.arange(len(n_locs)),
        "n_locs": n_locs,
    }
    if n_locs_partner is not None:
        regions["n_locs_partner"] = n_locs_partner
    regions["area_um2"] = result["areas"] / 1e6
    if result["use_z"]:
        regions["z_min_nm"] = result["z_ranges"][:, 0]
        regions["z_max_nm"] = result["z_ranges"][:, 1]
        regions["volume_um3"] = result["volumes"] / 1e9
        size, unit = regions["volume_um3"], "um3"
    else:
        size, unit = regions["area_um2"], "um2"
    with np.errstate(invalid="ignore", divide="ignore"):
        regions[f"density_per_{unit}"] = np.where(
            size > 0, n_locs / size, np.nan
        )
        if n_locs_partner is not None:
            regions[f"partner_density_per_{unit}"] = np.where(
                size > 0, n_locs_partner / size, np.nan
            )
    return regions


def nnd_distributions(
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict],
    *,
    picks: list[tuple] | None = None,
    pick_shape: (
        Literal["Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"]
        | None
    ) = None,
    pick_size: float | None = None,
    names: list[str] | None = None,
    r_max: float = 200.0,
    steps: int = 40,
    n_simulations: int = 99,
    use_z: bool = False,
    z_range: tuple[float, float] | None = None,
    simulated_z: Literal["uniform", "observed"] = "uniform",
    cross_null: Literal["shift", "labels"] = "shift",
    random_seed: int | None = 0,
    progress: lib.Progress | Literal["console"] | None = None,
) -> dict:
    """Distributions of the first nearest neighbor distances (NNDs)
    within and between channels in regions of interest, with the
    distributions expected without interaction from simulations.

    For every ordered pair of channels (A, B), the distance from each
    molecule of A to the nearest molecule of B in the same region is
    histogrammed (within a channel, to the nearest other molecule). NNDs
    are not symmetric: A to B differs from B to A, e.g., if B labels
    only some of A's structures. Molecules of a region without a
    molecule of B (within a channel, without another molecule) have no
    NND. The NNDs of all regions are pooled.

    Each region is simulated ``n_simulations`` times. Within a channel,
    its molecules are replaced by as many uniformly random points
    (complete spatial randomness, CSR). Between channels, B is shifted
    at random, wrapping around the region, which keeps the arrangement
    of each channel (``cross_null="shift"``, independent channels), or
    the channel of each molecule of A and B is shuffled (``"labels"``,
    both channels follow the same distribution); see ``ripley_h``. The
    mean of the simulated histograms is the distribution expected
    without interaction, and their lowest and highest values give the
    envelope. The simulations are in the same regions as the data, so
    NNDs that are longer near the edges of a region (where neighbors
    outside are not seen) are expected; no edge correction is needed.

    Parameters
    ----------
    locs : pd.DataFrame or list of pd.DataFrames
        Molecules of one or more channels in the same coordinates
        (aligned), one localization per molecule, such as the molecules
        found by ``g5m.g5m`` or the cluster centers of
        ``clusterer.find_cluster_centers``.
    info : list of dicts
        Metadata of the localizations (e.g., of the first channel), used
        for the field of view if ``picks`` is None.
    picks : list of tuples or None, optional
        Regions of interest in nm, in the format of
        ``pick_shape``. If None, the whole field of view is used. See
        ``ripley_h``. Default is None.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", \
"Brush"} or None, optional
        Shape of the picks. Required if ``picks`` is given. Default is
        None.
    pick_size : float or None, optional
        Size of the picks in nm: diameter for circles, width for
        rectangles, side length for squares, None for polygons, boxes
        and brush picks. Default is None.
    names : list of str or None, optional
        Names of the channels for the plot and the saved tables. If
        None, the channels are called "0", "1", ... Default is None.
    r_max : float, optional
        Largest NND of the histograms in nm; longer NNDs are counted in
        the totals, not in the histograms. Default is 200.
    steps : int, optional
        Number of histogram bins up to ``r_max``. Default is 40, i.e.,
        bins of 5 nm.
    n_simulations : int, optional
        Number of simulations. 0 skips them. Default is 99.
    use_z : bool, optional
        If True, the NNDs are computed in 3D (x, y and z in nm), which
        requires a ``z`` column in all channels. Default is False.
    z_range : tuple of floats or None, optional
        ``(z_min, z_max)`` in nm over which every region extends in 3D;
        localizations outside of it are ignored. If None, each region
        extends over the z range of the molecules of all channels in
        it. Only used in 3D. Default is None.
    simulated_z : {"uniform", "observed"}, optional
        z of the simulated points within a channel in 3D, see
        ``ripley_h``. Only used in 3D. Default is "uniform".
    cross_null : {"shift", "labels"}, optional
        How two channels are simulated: random shifts of the second
        channel (independent channels; rectangular and circular regions
        only) or random labels (both channels follow the same
        distribution), see ``ripley_h``. Only used with several
        channels. Default is "shift".
    random_seed : int or None, optional
        Seed of the simulations. None gives different simulations on
        each call. Default is 0.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Calculating NNDs" over the
        regions, then (if ``n_simulations > 0``) "Simulating envelope"
        over the simulations. Default is None.

    Returns
    -------
    result : dict
        Dictionary with the keys:

        - "names": names of the channels.
        - "edges": edges of the histogram bins in nm, shape (steps + 1,).
        - "histograms": fraction of the molecules of channel A (first
          axis) whose NND to channel B (second axis) falls in each bin,
          shape (n_channels, n_channels, steps); NaN for pairs without
          any NND.
        - "simulated_mean", "envelope_low", "envelope_high": mean,
          lowest and highest simulated fractions, same shape, or None
          without simulations.
        - "n_molecules": number of NNDs of each pair, i.e., of molecules
          of A with a molecule of B in their region.
        - "medians": median NND of each pair in nm.
        - "simulated_medians": mean over the simulations of the median
          NND of each pair in nm, or None without simulations.
        - "n_regions", "n_simulations", "use_z", "simulated_z" (None in
          2D), "cross_null" (None for one channel).

    Raises
    ------
    ValueError
        If the parameters are invalid, z is missing in 3D, no pair of
        channels has an NND or random shifts are requested in regions
        other than rectangles and circles.

    Warns
    -----
    UserWarning
        If a channel is not molecules (no ``n_events`` or ``std_frame``
        column).

    See Also
    --------
    plot_nnd_distributions : Plot the result.
    nn_analysis : NNDs of each localization.
    """
    if isinstance(locs, pd.DataFrame):
        locs = [locs]
    n_channels = len(locs)
    if not n_channels:
        raise ValueError("At least one channel is required.")
    if names is None:
        names = [str(i) for i in range(n_channels)]
    names = [str(name) for name in names]
    if len(names) != n_channels:
        raise ValueError(
            f"{len(names)} names given for {n_channels} channels."
        )
    if r_max <= 0:
        raise ValueError("r_max must be positive.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if n_simulations < 0:
        raise ValueError("n_simulations must not be negative.")
    cross = n_channels > 1
    if cross and cross_null not in ("shift", "labels"):
        raise ValueError(
            f"Invalid cross_null: {cross_null!r}; choose 'shift' or "
            "'labels'."
        )
    if use_z:
        if not all("z" in locs_.columns for locs_ in locs):
            raise ValueError("3D requires z in all channels.")
        if z_range is not None and not z_range[0] < z_range[1]:
            raise ValueError("z_range must be (z_min, z_max), z_min < z_max.")
        if simulated_z not in ("uniform", "observed"):
            raise ValueError(
                f"Invalid simulated_z: {simulated_z!r}; choose 'uniform' "
                "or 'observed'."
            )
    regions = _statistic_regions(
        info, picks, pick_shape, pick_size, cross and cross_null == "shift"
    )
    for locs_, name in zip(locs, names):
        _warn_if_not_molecules(locs_, f"Channel {name} is", stacklevel=3)

    coordinates = [
        (
            locs_["x"].to_numpy(dtype=np.float64),
            locs_["y"].to_numpy(dtype=np.float64),
            locs_["z"].to_numpy(dtype=np.float64) if use_z else None,
        )
        for locs_ in locs
    ]
    edges = np.arange(steps + 1) * (r_max / steps)
    pairs = [(a, b) for a in range(n_channels) for b in range(n_channels)]
    counts = np.zeros((n_channels, n_channels, steps))
    n_molecules = np.zeros((n_channels, n_channels), dtype=int)
    nnds = {pair: [] for pair in pairs}
    region_points = []  # (region, z range, points of each channel)
    n_regions = len(regions)
    with lib.tracked_progress(
        progress, "Calculating NNDs", unit="region"
    ) as progress:
        lib.start_phase(progress, "Calculating NNDs", n_regions, unit="region")
        progress.set_value(0)
        for k, region in enumerate(regions):
            data = _nnd_region_points(region, coordinates, use_z, z_range)
            if data is not None:
                points = data[1]
                for a, b in pairs:
                    region_nnds = _first_nnds(points[a], points[b], a == b)
                    nnds[a, b].append(region_nnds)
                    counts[a, b] += np.histogram(region_nnds, edges)[0]
                    n_molecules[a, b] += len(region_nnds)
                region_points.append((region, *data))
            progress.set_value(k + 1)
        progress.set_value(n_regions)
        if not n_molecules.any():
            raise ValueError(
                "No region contains two molecules of a channel or "
                "molecules of two channels."
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            histograms = counts / n_molecules[:, :, None]
        medians = np.full((n_channels, n_channels), np.nan)
        for a, b in pairs:
            if n_molecules[a, b]:
                medians[a, b] = np.median(np.concatenate(nnds[a, b]))

        simulated_mean = envelope_low = envelope_high = None
        simulated_medians = None
        if n_simulations:
            rng = np.random.default_rng(random_seed)
            simulated = np.zeros(
                (n_simulations, n_channels, n_channels, steps)
            )
            medians_sim = np.full(
                (n_simulations, n_channels, n_channels), np.nan
            )
            lib.start_phase(
                progress,
                "Simulating envelope",
                n_simulations,
                unit="simulation",
            )
            progress.set_value(0)
            for s in range(n_simulations):
                sim_counts = np.zeros((n_channels, n_channels, steps))
                sim_nnds = {pair: [] for pair in pairs}
                for region, region_z, points in region_points:
                    for a, b in pairs:
                        if a == b:
                            simulated_points = _simulated_points(
                                [points[a]],
                                region,
                                region_z,
                                False,
                                cross_null,
                                simulated_z,
                                rng,
                            )
                            source = target = simulated_points[0]
                        else:
                            source, target = _simulated_points(
                                [points[a], points[b]],
                                region,
                                region_z,
                                True,
                                cross_null,
                                simulated_z,
                                rng,
                            )
                        region_nnds = _first_nnds(source, target, a == b)
                        sim_nnds[a, b].append(region_nnds)
                        sim_counts[a, b] += np.histogram(region_nnds, edges)[0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    simulated[s] = sim_counts / n_molecules[:, :, None]
                for a, b in pairs:
                    if n_molecules[a, b]:
                        medians_sim[s, a, b] = np.median(
                            np.concatenate(sim_nnds[a, b])
                        )
                progress.set_value(s + 1)
            progress.set_value(n_simulations)
            simulated_mean = simulated.mean(axis=0)
            envelope_low = simulated.min(axis=0)
            envelope_high = simulated.max(axis=0)
            simulated_medians = medians_sim.mean(axis=0)

    return {
        "names": names,
        "edges": edges,
        "histograms": histograms,
        "simulated_mean": simulated_mean,
        "envelope_low": envelope_low,
        "envelope_high": envelope_high,
        "n_molecules": n_molecules,
        "medians": medians,
        "simulated_medians": simulated_medians,
        "n_regions": n_regions,
        "n_simulations": n_simulations,
        "use_z": use_z,
        "simulated_z": simulated_z if use_z else None,
        "cross_null": cross_null if cross else None,
    }


def _nnd_region_points(
    region: _Region | None,
    coordinates: list[tuple],
    use_z: bool,
    z_range: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, list[tuple]] | None:
    """The z range of a region (None in 2D) and the points ``(x, y,
    z)`` of each channel (``coordinates``, nm) inside it, or None for a
    region without area or, in 3D, without depth."""
    if region is None:
        return None
    points = []
    for x_all, y_all, z_all in coordinates:
        inside = _locs_in_region(x_all, y_all, region)
        if use_z and z_range is not None:
            z = z_all[inside]
            inside = inside[(z >= z_range[0]) & (z <= z_range[1])]
        points.append(
            (x_all[inside], y_all[inside], z_all[inside] if use_z else None)
        )
    region_z = None
    if use_z:
        if z_range is not None:
            region_z = tuple(z_range)
        elif sum(len(x) for x, _, _ in points):
            z = np.concatenate([z for _, _, z in points])
            region_z = (z.min(), z.max())
        if region_z is None or region_z[1] <= region_z[0]:
            return None
    return region_z, points


def _first_nnds(source: tuple, target: tuple, same: bool) -> lib.FloatArray1D:
    """Distances (nm) from each of the ``source`` points ``(x, y, z)``
    (z None in 2D) to the nearest ``target`` point; if ``same``, the
    source and target are the same points and the nearest other point
    is used. Empty if there is no such point."""
    n_targets = len(target[0])
    if not len(source[0]) or n_targets < (2 if same else 1):
        return np.zeros(0)
    columns = [c for c in source if c is not None]
    tree = KDTree(np.column_stack([c for c in target if c is not None]))
    distances, _ = tree.query(np.column_stack(columns), k=2 if same else 1)
    return distances[:, 1] if same else distances


def plot_nnd_distributions(
    nnd_result: dict,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot the first nearest neighbor distance (NND) distributions of
    every ordered pair of channels in a grid (rows: from, columns: to),
    with the mean and the range of the simulations.

    Parameters
    ----------
    nnd_result : dict
        Result of ``nnd_distributions``.
    fig : plt.Figure or None, optional
        Figure to plot on, cleared first. If None, a new figure is
        created. Default is None.

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot.
    """
    if fig is None:
        fig = plt.Figure(constrained_layout=True)
    else:
        fig.clear()
    names = nnd_result["names"]
    n_channels = len(names)
    edges = nnd_result["edges"]
    simulated = nnd_result["simulated_mean"] is not None
    # colors of the active style, resolved now (lines look "CN" colors
    # up only when drawn)
    data_color, simulation_color = plt.rcParams["axes.prop_cycle"].by_key()[
        "color"
    ][:2]
    axes = fig.subplots(n_channels, n_channels, sharex=True, squeeze=False)
    for a in range(n_channels):
        for b in range(n_channels):
            ax = axes[a, b]
            ax.set_title(f"{names[a]} → {names[b]}" if n_channels > 1 else "")
            histogram = nnd_result["histograms"][a, b]
            if np.isnan(histogram).all():
                ax.text(
                    0.5,
                    0.5,
                    "No NNDs",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue
            if simulated:
                ax.fill_between(
                    edges,
                    np.append(nnd_result["envelope_low"][a, b], np.nan),
                    np.append(nnd_result["envelope_high"][a, b], np.nan),
                    step="post",
                    color=simulation_color,
                    alpha=0.3,
                    linewidth=0,
                    label="Simulations (range)",
                )
                ax.stairs(
                    nnd_result["simulated_mean"][a, b],
                    edges,
                    color=simulation_color,
                    linewidth=1.2,
                    linestyle="--",
                    label="Simulations (mean)",
                )
            # the data on top of the simulations
            ax.stairs(
                histogram, edges, color=data_color, linewidth=1.5, label="Data"
            )
            ax.set_xlim(0, edges[-1])
            ax.set_ylim(bottom=0)
            if a == n_channels - 1:
                ax.set_xlabel("1st NND (nm)")
            if b == 0:
                ax.set_ylabel("Fraction of molecules")
    title = "Nearest neighbor distances" + (
        " (3D)" if nnd_result["use_z"] else ""
    )
    if simulated:
        axes[0, 0].legend(loc="best")
        null = {
            None: "random positions",
            "shift": "random positions within channels,\nrandom shifts "
            "between channels",
            "labels": "random positions within channels,\nrandom labels "
            "between channels",
        }[nnd_result["cross_null"]]
        title += f"\n{nnd_result['n_simulations']} simulations: {null}"
    fig.suptitle(title)
    return fig


def _nnd_tables(nnd_result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The result of ``nnd_distributions`` as two tables: the
    histograms by bin (data, simulated mean and range of each ordered
    pair of channels, as fractions of molecules) and the summary of
    each pair (number of molecules and median NND, measured and
    simulated). Used for saving the result as .csv files."""
    names = nnd_result["names"]
    edges = nnd_result["edges"]
    curves = {"bin_start_nm": edges[:-1], "bin_end_nm": edges[1:]}
    summary = []
    for a, name_a in enumerate(names):
        for b, name_b in enumerate(names):
            pair = f"{name_a}_to_{name_b}"
            curves[pair] = nnd_result["histograms"][a, b]
            if nnd_result["simulated_mean"] is not None:
                curves[f"{pair}_simulated_mean"] = nnd_result[
                    "simulated_mean"
                ][a, b]
                curves[f"{pair}_envelope_low"] = nnd_result["envelope_low"][
                    a, b
                ]
                curves[f"{pair}_envelope_high"] = nnd_result["envelope_high"][
                    a, b
                ]
            row = {
                "from": name_a,
                "to": name_b,
                "n_molecules": nnd_result["n_molecules"][a, b],
                "median_nnd_nm": nnd_result["medians"][a, b],
            }
            if nnd_result["simulated_medians"] is not None:
                row["simulated_median_nnd_nm"] = nnd_result[
                    "simulated_medians"
                ][a, b]
            summary.append(row)
    return pd.DataFrame(curves), pd.DataFrame(summary)


def resi(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    radius_xy: float | list[float],
    radius_z: float | list[float] | None = None,
    min_locs: int | list[int] = 10,
    apply_fa: bool = True,
    save_clustered_locs: bool = False,
    save_cluster_centers: bool = False,
    resi_path: str | None = None,
    output_paths: list[str] | None = None,
    suffix_locs: str = "_clustered",
    suffix_centers: str = "_cluster_centers",
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Perform RESI (REsolution by Sequential Imaging) analysis on
    multiple channels.

    Clusters localizations from each channel using the SMLM clusterer,
    extracts cluster centers, and combines them into a single DataFrame
    with channel IDs.

    Parameters
    ----------
    locs : list of pd.DataFrames
        List of localization datasets, one DataFrame per channel.
    infos : list of list of dicts
        List of metadata dictionaries for each channel.
    radius_xy : float or list of float
        Clustering radius in xy (nm). If a single float is provided, it
        is applied to all channels. If a list, must have length equal
        to the number of channels.
    radius_z : float, list of float, or None, optional
        Clustering radius in z (nm). Only used for 3D data.
        If a float, applied to all channels. If a list, must have length
        equal to the number of channels. Default is None.
    min_locs : int or list of int, optional
        Minimum number of localizations in a cluster. If an int, applied
        to all channels. If a list, must have length equal to the number
        of channels. Default is 10.
    apply_fa : bool, optional
        If True, apply basic frame analysis to clustered localizations.
        Default is True.
    save_clustered_locs : bool, optional
        If True, save clustered localizations for each channel to a
        file. Requires output_paths to be provided. Default is False.
    save_cluster_centers : bool, optional
        If True, save cluster centers for each channel to a file.
        Requires output_paths to be provided. Default is False.
    resi_path : str or None, optional
        Path to save the combined RESI cluster centers with metadata. If
        None, the combined cluster centers will not be saved. Default is
        None.
    output_paths : list of str or None, optional
        List of paths to save cluster centers for each channel. If None
        and save_* parameters are True, clustered data will not be
        saved. Default is None.
    suffix_locs : str, optional
        Suffix appended to output_paths for saved clustered
        localizations. Default is "_clustered".
    suffix_centers : str, optional
        Suffix appended to output_paths for saved cluster centers from
        individual channels. Default is "_cluster_centers".
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports, for each channel in turn, the
        phases of ``clusterer.cluster`` ("Clustering" over the
        localizations) and ``clusterer.find_cluster_centers``
        ("Finding cluster centers" over the clusters). Default is None.

    Returns
    -------
    resi_centers : pd.DataFrame
        Combined cluster centers from all channels. Contains all columns
        from the original localizations plus a 'resi_channel_id' column
        indicating which channel each cluster belongs to. The 'group'
        column is renamed to 'cluster_id'.
    resi_info : list of dicts
        Metadata of the RESI cluster centers: the metadata of the first
        channel with an entry recording the clustering parameters of
        each channel appended, see ``lib.append_metadata``.

    Raises
    ------
    ValueError
        If fewer than 2 channels are provided, or if list parameters
        have incorrect lengths.

    Notes
    -----
    RESI (REsolution by Sequential Imaging) relies on sequential imaging
    to ensure sufficient sparsity of binding sites. Therefore, at least
    2 channels are required.

    If output_paths are provided, the combined RESI cluster centers will
    be saved with a new metadata entry containing clustering parameters
    for each channel.
    """
    n_channels = len(locs)
    if n_channels < 2:
        raise ValueError(
            f"RESI requires at least 2 channels, but got {n_channels}. "
            "Consider using SMLM Clusterer for single-channel clustering."
        )

    # Ensure all parameters are lists for consistent handling
    if isinstance(radius_xy, (int, float)):
        radius_xy = [radius_xy] * n_channels
    elif len(radius_xy) != n_channels:
        raise ValueError(
            f"radius_xy list length ({len(radius_xy)}) must match "
            f"number of channels ({n_channels})"
        )

    if radius_z is not None:
        if isinstance(radius_z, (int, float)):
            radius_z = [radius_z] * n_channels
        elif len(radius_z) != n_channels:
            raise ValueError(
                f"radius_z list length ({len(radius_z)}) must match "
                f"number of channels ({n_channels})"
            )
    else:
        radius_z = [None] * n_channels

    if isinstance(min_locs, int):
        min_locs = [min_locs] * n_channels
    elif len(min_locs) != n_channels:
        raise ValueError(
            f"min_locs list length ({len(min_locs)}) must match "
            f"number of channels ({n_channels})"
        )
    with lib.tracked_progress(progress, "Clustering", unit="loc") as progress:
        return _resi(
            locs=locs,
            infos=infos,
            radius_xy=radius_xy,
            radius_z=radius_z,
            min_locs=min_locs,
            apply_fa=apply_fa,
            save_clustered_locs=save_clustered_locs,
            save_cluster_centers=save_cluster_centers,
            resi_path=resi_path,
            output_paths=output_paths,
            suffix_locs=suffix_locs,
            suffix_centers=suffix_centers,
            progress=progress,
        )


def _resi(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    radius_xy: list[float],
    radius_z: list[float] | None = None,
    min_locs: list[int] = 10,
    apply_fa: bool = True,
    save_clustered_locs: bool = False,
    save_cluster_centers: bool = False,
    resi_path: str | None = None,
    output_paths: list[str] | None = None,
    suffix_locs: str = "_clustered",
    suffix_centers: str = "_cluster_centers",
    progress: lib.Progress | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Internal function to perform RESI analysis, assumes all
    parameters are in the correct format and that there are at least 2
    channels. See `resi` for details; ``progress`` is a normalized
    tracker."""
    ndim = 3 if all(["z" in locs_.columns for locs_ in locs]) else 2

    # Process each channel
    resi_channels = []
    for i in range(len(locs)):
        locs_ = locs[i]
        info_ = infos[i]
        r_xy = radius_xy[i]
        r_z = radius_z[i]
        min_locs_ = min_locs[i]

        # Cluster localizations for this channel
        clustered_locs, clustered_info = clusterer.cluster(
            locs_,
            info_,
            radius_xy=r_xy,
            min_locs=min_locs_,
            frame_analysis=apply_fa,
            radius_z=r_z if ndim == 3 else None,
            progress=progress,
        )

        # Save clustered localizations if requested
        if save_clustered_locs and output_paths is not None:
            save_path = (
                os.path.splitext(output_paths[i])[0] + f"{suffix_locs}.hdf5"
            )
            io.save_locs(save_path, clustered_locs, clustered_info)

        # Extract cluster centers from clustered localizations
        centers, centers_info = clusterer.find_cluster_centers(
            clustered_locs, clustered_info, progress=progress
        )

        # Save cluster centers if requested
        if save_cluster_centers and output_paths is not None:
            save_path = (
                os.path.splitext(output_paths[i])[0] + f"{suffix_centers}.hdf5"
            )
            io.save_locs(save_path, centers, centers_info)

        # Add RESI channel ID to identify which channel this cluster
        # belongs to
        centers["resi_channel_id"] = i * np.ones(
            len(centers),
            dtype=np.int8,
        )
        resi_channels.append(centers)

    # Combine cluster centers from all channels; they carry the columns of
    # their channel's localizations, which need not all be the same
    all_resi = lib.concat_locs(resi_channels)

    # Rename 'group' to 'cluster_id' for clarity
    all_resi["cluster_id"] = all_resi["group"]
    all_resi.drop(columns=["group"], inplace=True)
    all_resi.sort_values(kind="quicksort", by="frame", inplace=True)

    entry = {
        **lib.step_info("RESI"),
        "Clustering radius xy (nm) for each channel": [
            float(r) for r in radius_xy
        ],
        "Min. number of locs in a cluster for each channel": list(min_locs),
        "Basic frame analysis": apply_fa,
    }
    if ndim == 3:
        entry["Clustering radius z (nm) for each channel"] = [
            float(r) for r in radius_z
        ]
    new_info = lib.append_metadata(infos[0], entry)
    # Save combined RESI results if output paths are provided
    if resi_path is not None:
        io.save_locs(resi_path, all_resi, new_info)

    return all_resi, new_info
