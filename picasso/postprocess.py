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
import warnings
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from typing import Literal
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from threading import Thread

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.spatial import distance, KDTree
from tqdm import tqdm, trange

from . import (
    io,
    lib,
    clusterer,
    render,
    imageprocess,
    masking,
    spatial_index,
    __version__,
)


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
        Size of the blocks in camera pixels. For circular picks, this
        is pick radius.

    Returns
    -------
    locs : pd.DataFrame
        Localizations in the specified blocks.
    size : float
        Size of the blocks in camera pixels.
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
        Size of the blocks.

    Returns
    -------
    n : tuple
        Number of blocks in y and x.
    """
    width = lib.get_from_metadata(info, "Width", raise_error=True)
    height = lib.get_from_metadata(info, "Height", raise_error=True)
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
    pick_size: float,
    index_blocks: tuple | None,
    add_group: bool,
    callback: Callable[[int], None] | Literal["console"] | None,
    progress: tqdm | None,
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
            index_blocks = get_index_blocks(locs, info, pick_size)
        locs_xy = index_blocks[0][["x", "y"]].to_numpy().T
    for i, pick in enumerate(picks):
        x, y = pick
        if pyramid is not None:
            idx = spatial_index.query_circle(pyramid, xs, ys, x, y, pick_size)
            group_locs = locs.iloc[idx].copy()
        else:
            x_, y_ = int(x / pick_size), int(y / pick_size)
            block_locs_idx = _get_block_locs_at_numba(
                x_,
                y_,
                index_blocks[4],
                index_blocks[5],
                index_blocks[6],
                index_blocks[7],
            )
            block_locs = index_blocks[0].iloc[block_locs_idx]
            group_locs_idx = lib.is_loc_at_numba(
                x, y, locs_xy[:, block_locs_idx], pick_size
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

        if callback == "console":
            progress.update(1)
        elif callback is not None:
            callback(i + 1)
    return picked_locs


def _picked_rectangular_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    pick_size: float,
    add_group: bool,
    callback: Callable[[int], None] | Literal["console"] | None,
    progress: tqdm | None,
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

        if callback == "console":
            progress.update(1)
        elif callback is not None:
            callback(i + 1)
    return picked_locs


def _picked_polygonal_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    add_group: bool,
    callback: Callable[[int], None] | Literal["console"] | None,
    progress: tqdm | None,
):
    """Helper function for picking localizations using polygonal picks. See
    ``picked_locs`` for more details."""
    picked_locs = []
    for i, pick in enumerate(picks):
        X, Y = lib.get_pick_polygon_corners(pick)
        if X is None:
            if callback == "console":
                progress.update(1)
            elif callback is not None:
                callback(i + 1)
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

        if callback == "console":
            progress.update(1)
        elif callback is not None:
            callback(i + 1)
    return picked_locs


def _picked_square_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    pick_size: float,
    add_group: bool,
    callback: Callable[[int], None] | Literal["console"] | None,
    progress: tqdm | None,
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

        if callback == "console":
            progress.update(1)
        elif callback is not None:
            callback(i + 1)
    return picked_locs


def _picked_box_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    add_group: bool,
    callback: Callable[[int], None] | Literal["console"] | None,
    progress: tqdm | None,
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

        if callback == "console":
            progress.update(1)
        elif callback is not None:
            callback(i + 1)
    return picked_locs


def _picked_brush_locs(
    locs: pd.DataFrame,
    picks: list[tuple],
    add_group: bool,
    callback: Callable[[int], None] | Literal["console"] | None,
    progress: tqdm | None,
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

        if callback == "console":
            progress.update(1)
        elif callback is not None:
            callback(i + 1)
    return picked_locs


def picked_locs(
    locs: pd.DataFrame,
    info: list[dict],
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float = None,
    add_group: bool = True,
    index_blocks: tuple = None,
    callback: Callable[[int], None] | Literal["console"] | None = None,
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
        Size of the pick in camera pixels. Radius for the circles, width
        for the rectangles, side length for squares, None for the
        polygons, boxes and brush picks (they carry their own extent).
        Default is None.
    add_group : boolean, optional
        True if group id should be added to locs. Each pick will be
        assigned a different id. Default is True.
    index_blocks : tuple or spatial_index.RenderIndexPyramid, optional
        Used only for circular picks. Precomputed index blocks for
        localizations, see ``get_index_blocks`` (built with block size
        ``pick_size``), or the pick-size independent pyramid of
        ``spatial_index.build_render_index``. If None, index blocks
        will be calculated internally. Default is None.
    callback : Callable[[int], None] | Literal["console"] | None, optional
        Function to display progress. If "console", tqdm is used to
        display the progress. If None, no progress is displayed. Default
        is None.

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

    picked_locs = []
    if callback == "console":
        progress = tqdm(
            range(len(picks)),
            desc="Picking locs",
            unit="pick",
        )
    else:
        progress = None

    if pick_shape == "Circle":
        picked_locs = _picked_circular_locs(
            locs=locs,
            info=info,
            picks=picks,
            pick_size=pick_size,
            index_blocks=index_blocks,
            add_group=add_group,
            callback=callback,
            progress=progress,
        )
    elif pick_shape == "Rectangle":
        picked_locs = _picked_rectangular_locs(
            locs=locs,
            picks=picks,
            pick_size=pick_size,
            add_group=add_group,
            callback=callback,
            progress=progress,
        )
    elif pick_shape == "Polygon":
        picked_locs = _picked_polygonal_locs(
            locs=locs,
            picks=picks,
            add_group=add_group,
            callback=callback,
            progress=progress,
        )
    elif pick_shape == "Square":
        picked_locs = _picked_square_locs(
            locs=locs,
            picks=picks,
            pick_size=pick_size,
            add_group=add_group,
            callback=callback,
            progress=progress,
        )
    elif pick_shape == "Box":
        picked_locs = _picked_box_locs(
            locs=locs,
            picks=picks,
            add_group=add_group,
            callback=callback,
            progress=progress,
        )
    elif pick_shape == "Brush":
        picked_locs = _picked_brush_locs(
            locs=locs,
            picks=picks,
            add_group=add_group,
            callback=callback,
            progress=progress,
        )
    return picked_locs


def pick_similar(
    locs: pd.DataFrame,
    info: list[dict],
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
        Size of the pick in camera pixels. Diameter for circles, side
        length for squares, width for rectangles. Ignored for boxes,
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
        Distance between neighboring candidate positions in camera
        pixels. Only used for rectangular picks, where the default
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
            block_locs_xy = get_block_locs_at_numba(
                int(x / r),
                int(y / r),
                locs_xy,
                index_blocks[4],
                index_blocks[5],
                index_blocks[6],
                index_blocks[7],
            )
            pick_locs_xy = lib.locs_at_numba(x, y, block_locs_xy, r)
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
    x_range = np.arange(d / 2, info[0]["Width"], np.sqrt(3) * d / 2)
    y_range_base = np.arange(d / 2, info[0]["Height"] - d / 2, d)
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
                block_locs_xy = get_block_locs_at_numba(
                    x_range,
                    y_range,
                    locs_xy,
                    block_starts,
                    block_ends,
                    K,
                    L,
                )
                picked_locs_xy = lib.locs_at_numba(
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
                        np.abs(x_test - x_test_old) > 1e-3
                        or np.abs(y_test - y_test_old) > 1e-3
                    ):
                        count += 1
                        # skip the locs if the loop is too long
                        if count > 500:
                            break
                        x_test_old = x_test
                        y_test_old = y_test
                        picked_locs_xy = lib.locs_at_numba(
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
        Distance between neighboring candidates in camera pixels.

    Returns
    -------
    grid_x, grid_y : lib.FloatArray1D
        Coordinates of the candidate positions.
    """
    width = lib.get_from_metadata(info, "Width", raise_error=True)
    height = lib.get_from_metadata(info, "Height", raise_error=True)
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
    column (``0 <= k < K``), like ``_get_block_locs_at_numba`` and
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
        Size of the index blocks in camera pixels.
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
            if np.abs(dx) <= 1e-3 and np.abs(dy) <= 1e-3:
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
        Size of the index blocks in camera pixels.
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
            if np.abs(dx) <= 1e-3 and np.abs(dy) <= 1e-3:
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
        drawn_locs_xy = lib.locs_in_rectangle_numba(
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
        Size of the rectangle in camera pixels.
    locs_xy : lib.FloatArray2D
        Localization coordinates, shape ``(2, N)``, sorted by index
        blocks.
    block_starts, block_ends : lib.IntArray2D
        Block start and end indices.
    K, L : int
        Number of blocks in y and x direction.
    block_size : float
        Size of the index blocks in camera pixels.
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
            np.abs(dx) <= 1e-3
            and np.abs(dy) <= 1e-3
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
        Size of the index blocks in camera pixels.
    length, width : float
        Size of the candidate rectangles in camera pixels.
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
) -> pd.DataFrame:
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
        Size of picks in camera pixels. For circles - diameters. For
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
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}."
    if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE:
        assert isinstance(
            pick_size, (int, float)
        ), "pick_size must be a number."
    if pick_shape == "Circle":
        pick_size /= 2  # convert diameter to radius
    else:
        index_blocks = None  # ignore index blocks, only used for circle picks
    all_picked_locs = picked_locs(
        locs=locs,
        info=info,
        picks=picks,
        pick_shape=pick_shape,
        pick_size=pick_size,
        add_group=False,
        index_blocks=index_blocks,
    )
    # store indices of picked locs
    idx = np.concatenate([_.index for _ in all_picked_locs])
    locs.drop(index=idx, inplace=True)
    return locs


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
def _get_block_locs_at_numba(
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
def get_block_locs_at_numba(
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
    indices = _get_block_locs_at_numba(
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
        Size of the bins for the histogram.
    r_max : float
        Maximum distance probed in the histogram.

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
    info: any = None,
    callback: Callable[[int], None] | None = None,
) -> tuple[dict, float]:
    """Calculate NeNA - experimental estimate of localization
    precision. Please refer to the original paper for details:
    Endesfelder, et al. Histochemistry and Cell Biology, 2014.
    DOI: 10.1007/s00418-014-1192-3.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : any
        Metadata of the localizations. Not used.
    callback : function or None
        Function to display progress (takes in an integer). If None, no
        progress is displayed.

    Returns
    -------
    result : dict
        Data on the results, including the distances probed, best fit
        and fitted parameters.
    s : float
        Estimated localization precision in camera pixels.
    """
    bin_centers, dnfl = _next_frame_neighbor_distance_histogram(locs, callback)

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
        "pixelsize": lib.get_from_metadata(info, "Pixelsize", default="N/A"),
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
        distances probed, best fit and fitted parameters. If "pixelsize"
        is included, the distances will be plotted in nm, otherwise in
        camera pixels.
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
    pixelsize = (
        nena_result["pixelsize"] if nena_result["pixelsize"] != "N/A" else 1
    )
    unit = "nm" if nena_result["pixelsize"] != "N/A" else "pixels"
    d *= nena_result["pixelsize"]
    ax.set_title(
        "Next frame neighbor distance histogram, "
        f"\u03c3 = {nena_result['best_values']['s'] * pixelsize:.2f} {unit}"
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
    ax.set_xlabel(f"Distance ({unit})")
    ax.set_ylabel("Counts")
    ax.legend(loc="best")
    return fig


def _next_frame_neighbor_distance_histogram(
    locs: pd.DataFrame,
    callback: Callable[[int], None] | None = None,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the next frame neighbor distance histogram (NFNDH).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    callback : function or None
        Function to display progress. If None, no progress is displayed.

    Returns
    -------
    bin_centers : lib.FloatArray1D
        Centers of the bins for the histogram.
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
    bin_size = 0.001
    d_max = 1.0
    return _nfndh(frame, x, y, group, d_max, bin_size, callback)


def _nfndh(
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    group: lib.IntArray1D,
    d_max: float,
    bin_size: float,
    callback: Callable[[int], None] | None = None,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the next frame neighbor distance histogram (NFNDH)."""
    N = len(frame)
    bins = np.arange(0, d_max, bin_size)
    dnfl = np.zeros(len(bins))
    one_percent = int(N / 100)
    starts = one_percent * np.arange(100)
    for k, start in enumerate(starts):
        for i in range(start, start + one_percent):
            _fill_dnfl(N, frame, x, y, group, i, d_max, dnfl, bin_size)
        if callback is not None:
            callback(k + 1)
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
    min_frame = frame_i + 1
    for min_index in range(i + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = frame_i + 1
    for max_index in range(min_index, N):
        if frame[max_index] > max_frame:
            break
    d_max_2 = d_max**2
    for j in range(min_index, max_index):
        if group[j] == group_i:
            dx2 = (x_i - x[j]) ** 2
            if dx2 <= d_max_2:
                dy2 = (y_i - y[j]) ** 2
                if dy2 <= d_max_2:
                    d = np.sqrt(dx2 + dy2)
                    if d <= d_max:
                        bin = int(d / bin_size)
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
    viewport: tuple[tuple[float, float], tuple[float, float]],
    *,
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
        Viewport ((y_min, x_min), (y_max, x_max)) for rendering the
        images. Note that the origin of the image is in the top-left
        corner.
    random_seed : int, optional
        Random seed for splitting the data into halves. Default is 42.
    lp : float or None, optional
        Localization precision (camera pixels) that sets the bin size of
        the rendered images. If None (default), NeNA of ``locs`` is
        used.

    Returns
    -------
    frc_result : dict
        Dictionary with keys "frc_curve", "frc_curve_smooth",
        "frequencies" (for spatial frequencies probed (nm^-1)),
        "resolution" (estimated resolution in nm) and "images"
        (2 grayscale images rendered and masked).
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
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
        locs1, locs2, pixelsize, lp, viewport
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
    viewport: tuple[tuple[float, float], tuple[float, float]],
    *,
    n_rois: int = 30,
    roi_size: float = 5000.0,
    min_locs: int = 1000,
    random_seed: int = 42,
    callback: Callable[[int], None] | None = None,
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
        Region ((y_min, x_min), (y_max, x_max)) in camera pixels in
        which the ROIs are placed.
    n_rois : int, optional
        Maximum number of ROIs. Default is 30.
    roi_size : float, optional
        Side length of the square ROIs in nm. Default is 5000.
    min_locs : int, optional
        Minimum number of localizations per ROI. Default is 1000.
    random_seed : int, optional
        Random seed for the ROI placement and the splitting into
        halves. Default is 42.
    callback : function or None, optional
        Function to display progress, called with the number of ROIs
        processed. If None, no progress is displayed.

    Returns
    -------
    result : dict
        Dictionary with keys "rois" (viewports of the ROIs in camera
        pixels), "n_locs" (localizations per ROI), "frc_results"
        (list of ``frc`` results, one per ROI), "resolutions" (in nm,
        NaN where the FRC curve does not cross the 1/7 threshold) and
        "lp" (NeNA in camera pixels).
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
    lp = nena(locs, info)[1]
    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()
    n_locs = []
    frc_results = []
    for k, ((y0, x0), (y1, x1)) in enumerate(rois):
        if callable(callback):
            callback(k)
        in_roi = (x > x0) & (y > y0) & (x < x1) & (y < y1)
        n_locs.append(int(in_roi.sum()))
        frc_result = frc(
            locs.loc[in_roi],
            info,
            ((y0, x0), (y1, x1)),
            random_seed=random_seed,
            lp=lp,
        )
        # drop the rendered images: tens of ROIs would hold GBs
        del frc_result["images"]
        frc_results.append(frc_result)
    if callable(callback):
        callback(len(rois))
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
    pixelsize: float,
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
    pixelsize : float
        Camera pixel size in nm.
    lp : float
        Average localization precision (NeNA), used for bin size of the
        rendered images (binsize = lp / 2).
    viewport : tuple of floats
        Viewport ((y_min, x_min), (y_max, x_max)) for rendering the
        images. Note that the origin of the image is in the top-left
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
    binsize = lp / 2
    disp_px_size = pixelsize * binsize
    info = {"Pixelsize": pixelsize}
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
    frequencies = (
        np.arange(len(frc_curve)) / im1.shape[0] / (pixelsize * binsize)
    )
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
    bin_size: float,
    r_max: float,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Calculate the pair correlation function for the given
    localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    bin_size : float
        Size of the bins for the histogram.
    r_max : float
        Maximum distance for the histogram.

    Returns
    -------
    bins_lower : lib.FloatArray1D
        Lower bounds of the bins for the histogram.
    pc : lib.FloatArray1D
        Pair correlation function.
    """
    dh = distance_histogram(locs, info, bin_size, r_max)
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
    radius: float,
) -> pd.DataFrame:
    """Compute the local density of localizations in blocks around
    each localization.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    radius : float
        Radius for local density computation.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with added 'density' field/column.
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
    return locs


def evaluate_picks(
    picked_locs: list[pd.DataFrame],
    info: list[dict],
    *,
    max_dark_time: int = 3,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
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
    progress_callback : function, "console" or None
        Function to display progress (takes in an integer). If "console",
        progress is printed to the console. If None, no progress is
        displayed.

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
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(
            range(len(picked_locs)), desc="Evaluating picks", unit="pick"
        )
    else:
        iter_range = range(len(picked_locs))

    pixelsize = lib.get_from_metadata(info, "Pixelsize", default=1.0)
    n_picks = len(picked_locs)
    N = np.empty(n_picks)  # number of locs per pick
    n_events = np.empty(n_picks)  # number of events per pick
    rmsd = np.empty(n_picks)  # rmsd in each pick
    length = np.empty(n_picks)  # estimated mean bright time
    dark = np.empty(n_picks)  # estimated mean dark time
    has_z = "z" in picked_locs[0].columns
    rmsd_z = np.empty(n_picks)
    new_locs = []  # linked locs in each pick
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for i in iter_range:
        if callable(progress_callback):
            progress_callback(i)
        pick_locs = picked_locs[i]
        if not len(pick_locs):
            continue

        N[i] = len(pick_locs)
        com_x = pick_locs["x"].mean()
        com_y = pick_locs["y"].mean()
        rmsd[i] = (
            np.sqrt(
                np.mean(
                    (pick_locs["x"] - com_x) ** 2
                    + (pick_locs["y"] - com_y) ** 2
                )
            )
            * pixelsize
        )
        if has_z:
            rmsd_z[i] = np.sqrt(
                np.mean((pick_locs["z"] - pick_locs["z"].mean()) ** 2)
            )
        if "len" not in pick_locs.columns:
            pick_locs = link(
                pick_locs, info, r_max=999999, max_dark_time=max_dark_time
            )
        pick_locs = compute_dark_times(pick_locs)
        n_events[i] = len(pick_locs)  # linked locs are binding events
        length[i] = lib.estimate_kinetic_rate(pick_locs["len"].to_numpy())
        dark[i] = lib.estimate_kinetic_rate(pick_locs["dark"].to_numpy())
        new_locs.append(pick_locs)
    warnings.simplefilter("default", category=RuntimeWarning)
    if callable(progress_callback):
        progress_callback(n_picks)
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
        pick_locs = link(
            pick_locs,
            info,
            r_max=999999,  # link all locs in the pick
            max_dark_time=max_dark_time,
        )
    if not len(pick_locs):
        return None
    pick_locs = compute_dark_times(pick_locs)
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
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
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
    progress_callback : function, "console" or None
        Function to display progress (takes in an integer). If "console",
        progress is printed to the console. If None, no progress is
        displayed.

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
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(
            range(len(picked_locs)), desc="Calculating kinetics", unit="pick"
        )
    else:
        iter_range = range(len(picked_locs))

    out_locs = []
    dark = []  # estimated mean dark time
    length = []  # estimated mean bright time
    no_locs = []  # number of locs
    kept_indices = []  # picks for which kinetics could be estimated
    for i in iter_range:
        if callable(progress_callback):
            progress_callback(i)
        result = _pick_kinetics_single(picked_locs[i], info, max_dark_time)
        if result is None:
            continue
        pick_locs, l_, d_ = result
        length.append(l_)
        dark.append(d_)
        no_locs.append(len(picked_locs[i]))
        out_locs.append(pick_locs)
        kept_indices.append(i)
    if callable(progress_callback):
        progress_callback(i + 1)
    length = np.array(length)
    dark = np.array(dark)
    no_locs = np.array(no_locs)
    kept_indices = np.array(kept_indices, dtype=int)
    out_locs = pd.concat(out_locs, ignore_index=True)
    return length, dark, no_locs, out_locs, kept_indices


def pick_properties(
    picked_locs: list[pd.DataFrame],
    info: list[dict],
    *,
    max_dark_time: int = 3,
    influx_rate: float = 0.03,
    pick_areas: lib.FloatArray1D | None = None,
    kinetics_progress: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    groupprops_progress: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> pd.DataFrame:
    """Calculate pick properties and save them to ``path``.

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
    kinetics_progress, groupprops_progress : callable, "console" or None
        Progress callbacks forwarded to ``pick_kinetics`` and
        ``groupprops``, respectively.

    Returns
    -------
    pick_props : pd.DataFrame
        Each row gives the properties per pick.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=(OptimizeWarning, RuntimeWarning)
        )
        length, dark, no_locs, out_locs, kept_indices = pick_kinetics(
            picked_locs=picked_locs,
            info=info,
            max_dark_time=max_dark_time,
            progress_callback=kinetics_progress,
        )
        pick_props = groupprops(out_locs, callback=groupprops_progress)
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
    group: lib.IntArray1D | None = None,
) -> pd.DataFrame:
    """Compute dark time for each binding event.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations that were linked, i.e., binding events.
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
    """
    if "len" not in locs.columns:
        raise AttributeError(
            "Length not found. Please link localizations first."
        )
    dark = dark_times(locs, group)
    locs["dark"] = np.int32(dark)
    locs = locs[locs.dark != -1]
    return locs


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
    r_max: float = 0.05,
    max_dark_time: int = 3,
    combine_mode: Literal["average", "refit"] = "average",
    remove_ambiguous_lengths: bool = True,
) -> pd.DataFrame:
    """Link localizations, i.e., group them into binding events based
    on their spatiotemporal proximity.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Metadata of the localizations.
    r_max : float, optional
        Maximum distance for linking localizations. Default is 0.05.
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
    """
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
    r_max: float = 0.1,
    max_dark_time: int = 0,
    min_n_locs: int = 3,
) -> pd.DataFrame:
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
    r_max : float, optional
        Maximum distance (camera pixels) between localizations to be
        considered as originating from the same binding event. Default
        is 0.1.
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
    """
    out_locs = locs.sort_values(kind="quicksort", by="frame")
    if len(out_locs) == 0:
        return lib.append_group(out_locs.copy(), np.array([], dtype=np.int32))

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
    return lib.append_group(out_locs, event_id.astype(np.int32))


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
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> pd.DataFrame:
    """Combine localizations in picked regions.

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
    progress_callback : callable, 'console' or None, optional
        Function to display progress (takes in an integer, maximum is
        the number of picks). If 'console', progress is displayed in the
        console. If None, no progress is displayed. Default is None.

    Returns
    -------
    out_locs : pd.DataFrame
        Localizations after combining localizations in the picked
        regions.
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}."
    if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE:
        assert (
            pick_size is not None
        ), "Pick size must be provided for this pick shape."
    if pick_shape == "Circle":
        pick_size /= 2  # convert diameter to radius
    pl = picked_locs(
        locs=locs,
        info=info,
        picks=picks,
        pick_shape=pick_shape,
        pick_size=pick_size,
        index_blocks=index_blocks,
    )
    # use very large values for linking localizations
    r_max = 2 * max(
        lib.get_from_metadata(info, "Height"),
        lib.get_from_metadata(info, "Width"),
    )
    max_dark = lib.get_from_metadata(info, "Frames", default=10_000)

    # link every localization in each pick
    if progress_callback == "console":
        iter_range = tqdm(range(len(pl)), desc="Combining picks", unit="pick")
    else:
        iter_range = range(len(pl))
    out_locs = []
    for i in iter_range:
        if callable(progress_callback):
            progress_callback(i)
        pick_locs = pl[i]
        pick_locs_out = link(
            pick_locs,
            info,
            r_max=r_max,
            max_dark_time=max_dark,
            remove_ambiguous_lengths=False,
        )
        if len(pick_locs_out):
            out_locs.append(pick_locs_out)
    if callable(progress_callback):
        progress_callback(len(pl))
    out_locs = pd.concat(out_locs, ignore_index=True)
    return out_locs


# Combine localizations: calculate the properties of the group
def cluster_combine(locs: pd.DataFrame) -> pd.DataFrame:
    """Combine localizations into clusters and calculate their
    properties such as center of mass, standard deviation, and number
    of localizations in each cluster.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with 'group' and 'cluster' columns.

    Returns
    -------
    combined_locs : pd.DataFrame
        Combined localizations with calculated properties for each
        cluster.
    """
    combined_locs = []
    if "z" in locs.columns:
        for group in tqdm(np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = np.unique(temp["cluster"].to_numpy())
            n_cluster = len(cluster)
            mean_frame = np.zeros(n_cluster)
            std_frame = np.zeros(n_cluster)
            com_x = np.zeros(n_cluster)
            com_y = np.zeros(n_cluster)
            com_z = np.zeros(n_cluster)
            std_x = np.zeros(n_cluster)
            std_y = np.zeros(n_cluster)
            std_z = np.zeros(n_cluster)
            group_id = np.zeros(n_cluster)
            n = np.zeros(n_cluster, dtype=np.int32)
            for i, clusterval in enumerate(cluster):
                cluster_locs = temp[temp["cluster"] == clusterval]
                mean_frame[i] = cluster_locs["frame"].mean()
                com_x[i] = np.average(
                    cluster_locs["x"].to_numpy(),
                    weights=cluster_locs["photons"].to_numpy(),
                )
                com_y[i] = np.average(
                    cluster_locs["y"].to_numpy(),
                    weights=cluster_locs["photons"].to_numpy(),
                )
                com_z[i] = np.average(
                    cluster_locs["z"].to_numpy(),
                    weights=cluster_locs["photons"].to_numpy(),
                )
                std_frame[i] = cluster_locs["frame"].std()
                std_x[i] = cluster_locs["x"].std() / np.sqrt(len(cluster_locs))
                std_y[i] = cluster_locs["y"].std() / np.sqrt(len(cluster_locs))
                std_z[i] = cluster_locs["z"].std() / np.sqrt(len(cluster_locs))
                n[i] = len(cluster_locs)
                group_id[i] = group
            clusters = pd.DataFrame(
                {
                    "group": group_id,
                    "cluster": cluster,
                    "mean_frame": mean_frame.astype(np.float32),
                    "x": com_x.astype(np.float32),
                    "y": com_y.astype(np.float32),
                    "z": com_z.astype(np.float32),
                    "std_frame": std_frame.astype(np.float32),
                    "lpx": std_x.astype(np.float32),
                    "lpy": std_y.astype(np.float32),
                    "lpz": std_z.astype(np.float32),
                    "n": n.astype(np.int32),
                }
            )
            combined_locs.append(clusters)
    else:
        for group in tqdm(np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = np.unique(temp["cluster"].to_numpy())
            n_cluster = len(cluster)
            mean_frame = np.zeros(n_cluster)
            std_frame = np.zeros(n_cluster)
            com_x = np.zeros(n_cluster)
            com_y = np.zeros(n_cluster)
            std_x = np.zeros(n_cluster)
            std_y = np.zeros(n_cluster)
            group_id = np.zeros(n_cluster)
            n = np.zeros(n_cluster, dtype=np.int32)
            for i, clusterval in enumerate(cluster):
                cluster_locs = temp[temp["cluster"] == clusterval]
                mean_frame[i] = cluster_locs["frame"].mean()
                com_x[i] = np.average(
                    cluster_locs["x"].to_numpy(),
                    weights=cluster_locs["photons"].to_numpy(),
                )
                com_y[i] = np.average(
                    cluster_locs["y"].to_numpy(),
                    weights=cluster_locs["photons"].to_numpy(),
                )
                std_frame[i] = cluster_locs["frame"].std()
                std_x[i] = cluster_locs["x"].std() / np.sqrt(len(cluster_locs))
                std_y[i] = cluster_locs["y"].std() / np.sqrt(len(cluster_locs))
                n[i] = len(cluster_locs)
                group_id[i] = group
            clusters = pd.DataFrame(
                {
                    "group": group_id,
                    "cluster": cluster,
                    "mean_frame": mean_frame.astype(np.float32),
                    "x": com_x.astype(np.float32),
                    "y": com_y.astype(np.float32),
                    "std_frame": std_frame.astype(np.float32),
                    "lpx": std_x.astype(np.float32),
                    "lpy": std_y.astype(np.float32),
                    "n": n.astype(np.int32),
                }
            )
            combined_locs.append(clusters)

    combined_locs = pd.concat(combined_locs, ignore_index=True)
    return combined_locs


def cluster_combine_dist(
    locs: pd.DataFrame, pixelsize: float | None = None
) -> pd.DataFrame:
    """Similar to ``cluster_combine``, but also calculates the distance
    to the nearest neighbor in the same group and the distance to the
    nearest neighbor in the same cluster in the same group.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with 'group' and 'cluster' fields.
    pixelsize : float or None, optional
        Pixel size in nm for z-scaling. If None, defaults to 130 nm.

    Returns
    -------
    combined_locs : pd.DataFrame
        Combined localizations with calculated properties for each
        cluster, including distances to nearest neighbors.
    """
    if "z" in locs.columns:
        pixelsize = 130 if pixelsize is None else pixelsize
        combined_locs = []
        for group in tqdm(np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = temp["mean_frame"].to_numpy()
            std_frame = temp["std_frame"].to_numpy()
            com_x = temp["x"].to_numpy()
            com_y = temp["y"].to_numpy()
            com_z = temp["z"].to_numpy()
            std_x = temp["lpx"].to_numpy()
            std_y = temp["lpy"].to_numpy()
            std_z = temp["lpz"].to_numpy()
            group_id = temp["group"].to_numpy()
            n = temp["n"].to_numpy()
            min_dist = np.zeros(n_cluster)
            min_dist_xy = np.zeros(n_cluster)
            for i, clusterval in enumerate(cluster):
                # find nearest neighbor in xyz
                group_locs = temp[temp["cluster"] != clusterval]
                cluster_locs = temp[temp["cluster"] == clusterval]
                ref_point = np.stack(
                    (
                        cluster_locs["x"].to_numpy(),
                        cluster_locs["y"].to_numpy(),
                        cluster_locs["z"].to_numpy() / pixelsize,
                    ),
                    axis=1,
                )
                all_points = np.stack(
                    (
                        group_locs["x"].to_numpy(),
                        group_locs["y"].to_numpy(),
                        group_locs["z"].to_numpy() / pixelsize,
                    ),
                    axis=1,
                )
                distances = distance.cdist(ref_point, all_points)
                min_dist[i] = np.amin(distances)
                # find nearest neighbor in xy
                ref_point_xy = np.array(cluster_locs[["x", "y"]])
                all_points_xy = np.array(group_locs[["x", "y"]])
                distances_xy = distance.cdist(ref_point_xy, all_points_xy)
                min_dist_xy[i] = np.amin(distances_xy)

            clusters = pd.DataFrame(
                {
                    "group": group_id,
                    "cluster": cluster,
                    "mean_frame": mean_frame.astype(np.float32),
                    "x": com_x.astype(np.float32),
                    "y": com_y.astype(np.float32),
                    "z": com_z.astype(np.float32),
                    "std_frame": std_frame.astype(np.float32),
                    "lpx": std_x.astype(np.float32),
                    "lpy": std_y.astype(np.float32),
                    "lpz": std_z.astype(np.float32),
                    "n": n.astype(np.int32),
                    "min_dist": min_dist.astype(np.float32),
                    "mind_dist_xy": min_dist_xy.astype(np.float32),
                }
            )
            combined_locs.append(clusters)

    else:  # 2D case
        combined_locs = []
        for group in tqdm(np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = temp["mean_frame"].to_numpy()
            std_frame = temp["std_frame"].to_numpy()
            com_x = temp["x"].to_numpy()
            com_y = temp["y"].to_numpy()
            std_x = temp["lpx"].to_numpy()
            std_y = temp["lpy"].to_numpy()
            group_id = temp["group"].to_numpy()
            n = temp["n"].to_numpy()
            min_dist = np.zeros(n_cluster)

            for i, clusterval in enumerate(cluster):
                # find nearest neighbor in xyz
                group_locs = temp[temp["cluster"] != clusterval]
                cluster_locs = temp[temp["cluster"] == clusterval]
                ref_point_xy = np.array(cluster_locs[["x", "y"]])
                all_points_xy = np.array(group_locs[["x", "y"]])
                distances_xy = distance.cdist(ref_point_xy, all_points_xy)
                min_dist[i] = np.amin(distances_xy)

            clusters = pd.DataFrame(
                {
                    "group": group_id,
                    "cluster": cluster,
                    "mean_frame": mean_frame.astype(np.float32),
                    "x": com_x.astype(np.float32),
                    "y": com_y.astype(np.float32),
                    "std_frame": std_frame.astype(np.float32),
                    "lpx": std_x.astype(np.float32),
                    "lpy": std_y.astype(np.float32),
                    "n": n.astype(np.int32),
                    "min_dist": min_dist.astype(np.float32),
                }
            )
            combined_locs.append(clusters)

    combined_locs = pd.concat(combined_locs, ignore_index=True)
    return combined_locs


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
    callback: Callable[[int], None] = None,
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
        Additional keyword arguments for the rendering function.
        Default is an empty dictionary.
    callback : Callable[[int], None], optional
        Callback function to report progress. It should accept an
        integer argument representing the current segment index.
        Default is None, which means no callback is used.

    Returns
    -------
    bounds : lib.IntArray1D
        Array of bounds for each segment, where each bound is the
        starting frame of the segment.
    segments : lib.FloatArray3D
        3D array of segments, where each segment is a 2D image of the
        localizations in that segment.
    """
    Y = lib.get_from_metadata(info, "Height", raise_error=True)
    X = lib.get_from_metadata(info, "Width", raise_error=True)
    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    disp_px_size = kwargs.get("disp_px_size", pixelsize)
    oversampling = pixelsize / disp_px_size
    n_pixel_y = int(np.ceil(oversampling * Y))
    n_pixel_x = int(np.ceil(oversampling * X))
    n_seg = n_segments(info, segmentation)
    bounds = np.linspace(0, n_frames - 1, n_seg + 1, dtype=np.uint32)
    segments = np.zeros((n_seg, n_pixel_y, n_pixel_x))
    if callback is None:
        it = trange(n_seg, desc="Generating segments", unit="segments")
    else:
        callback(0)
        it = range(n_seg)
    for i in it:
        segment_locs = locs[
            (locs["frame"] >= bounds[i]) & (locs["frame"] < bounds[i + 1])
        ]
        _, segments[i] = render.render(
            segment_locs, info, disp_px_size=disp_px_size, **kwargs
        )
        if callback is not None:
            callback(i + 1)
    return bounds, segments


def undrift(
    locs: pd.DataFrame,
    info: list[dict],
    segmentation: int,
    display: bool = True,
    segmentation_callback: Callable[[int], None] = None,
    rcc_callback: Callable[[int], None] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    segmentation_callback : Callable[[int], None], optional
        Callback function to report progress during segmentation. It
        should accept an integer argument representing the current
        segment index. Default is None, which means no callback is used.
    rcc_callback : Callable[[int], None], optional
        Callback function to report progress during RCC calculation.
        It should accept an integer argument representing the current
        segment index. Default is None, which means no callback is used.

    Returns
    -------
    drift : pd.DataFrame
        Estimated drift as a DataFrame with columns 'x' and 'y'.
    locs : pd.DataFrame
        Undrifted localization list with the drift applied to the 'x'
        and 'y' coordinates.
    """
    locs = locs.copy()
    bounds, segments = segment(
        locs,
        info,
        segmentation,
        {"blur_method": "gaussian", "min_blur_width": 1},
        segmentation_callback,
    )
    shift_y, shift_x = imageprocess.rcc(segments, 32, rcc_callback)
    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = interpolate.InterpolatedUnivariateSpline(t, shift_x, k=3)
    drift_y_pol = interpolate.InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = np.arange(info[0]["Frames"])
    drift_ = (drift_x_pol(t_inter), drift_y_pol(t_inter))
    drift = pd.DataFrame({"x": drift_[0], "y": drift_[1]})
    if display:
        pixelsize = lib.get_from_metadata(info, "Pixelsize", 1.0)
        plot_drift(drift, pixelsize)
        plt.show()
    locs = apply_drift(locs, info, drift=drift)
    return drift, locs


def undrift_from_fiducials(
    locs: pd.DataFrame,
    info: list[dict],
    picks: list[tuple] | None = None,
    pick_size: float | None = None,
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ] = "Circle",
    undrift_z: bool = True,
    index_blocks: tuple | None = None,
    progress: lib.ProgressDialog | Literal["console"] | None = None,
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
        Size of the picks in camera pixels: the pick **radius** for
        circles, the width for rectangles, the side length for squares.
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
        ``get_index_blocks`` (built with block size equal to
        ``pick_size``). When provided, used to skip re-indexing inside
        circular ``picked_locs``. Ignored for every other pick shape,
        and when ``picks`` is None (auto-detected fiducials use a radius
        that may not match the precomputed index). Default is None.
    progress : picasso.lib.ProgressDialog or "console" or None, optional
        Tracks progress of picking the localizations, one step per pick.
        If "console", a tqdm progress bar is shown in the console. If a
        ProgressDialog, a GUI progress bar is updated. If None (default),
        no progress is displayed. Same convention as ``picasso.aim.aim``.

    Returns
    -------
    locs : pd.DataFrame
        Undrifted localizations.
    new_info : list of dicts
        Updated metadata.
    drift : pd.DataFrame
        Drift in x and y (and optionally z) directions.

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
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)

    if picks is None:
        # auto-detect fiducials; these are always circular
        picks, box = imageprocess.find_fiducials(locs, info)
        pick_shape = "Circle"
        pick_radius = box / 2
        # passed-in index_blocks was built for a different radius; drop
        # (the pyramid serves any radius)
        if not _is_pyramid(index_blocks):
            index_blocks = None
    else:
        # user-provided list of pick coordinates
        needs_size = pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE
        if pick_size is None and needs_size:
            raise ValueError(
                "pick_size (radius in camera pixels for circular picks) "
                "must be provided when picks are given as a list of "
                "coordinates."
            )
        pick_radius = pick_size

    if pick_shape != "Circle":
        # the index is built for circular lookups only
        index_blocks = None

    if len(picks) == 0:
        raise ValueError("No picks found for drift correction.")

    # get picked localizations
    progress = lib.normalize_progress(
        progress, description="Picking localizations", unit="pick"
    )
    progress.setMaximum(len(picks))
    progress.zero_progress("Picking localizations")
    pl = picked_locs(
        locs,
        info,
        picks,
        pick_shape,
        pick_size=pick_radius,
        add_group=False,
        index_blocks=index_blocks,
        callback=progress.set_value,
    )

    # calculate drift
    progress.setMaximum(0)
    progress.zero_progress("Calculating drift")
    drift = undrift_from_picked(pl, info)
    progress.close()
    if not undrift_z:
        drift = drift.drop(columns="z", errors="ignore")
    locs = apply_drift(locs, info, drift=drift)

    pick_info = {
        "Generated by": (f"Picasso v{__version__} Undrift from picked"),
        "Number of picks": len(picks),
        "Pick shape": pick_shape,
    }
    if pick_shape == "Circle":
        pick_info["Pick radius (nm)"] = pick_radius * pixelsize
    elif pick_radius is not None:
        pick_info["Pick size (nm)"] = pick_radius * pixelsize
    new_info = info + [pick_info]

    return locs, new_info, drift


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
):
    """Convenience function to apply drift to localizations. Runs checks
    to ensure correct formats.

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
    return _apply_drift(locs, drift)


def plot_drift(
    drift: pd.DataFrame,
    pixelsize: int | float,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Convenience function to plot 2D or 3D drift from a DataFrame.

    Parameters
    ----------
    drift : pd.DataFrame
        DataFrame containing the drift to plot. Should have columns 'x'
        and 'y', and optionally 'z'.
    pixelsize : int or float
        Pixel size in nm to convert drift from pixels to nm for
        plotting.
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
        ax1.plot(drift["x"] * pixelsize, label="x")
        ax1.plot(drift["y"] * pixelsize, label="y")
        ax1.legend(loc="best")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Drift (nm)")
        ax2 = fig.add_subplot(132)
        ax2.plot(
            drift.x * pixelsize,
            drift.y * pixelsize,
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
        ax1.plot(drift["x"] * pixelsize, label="x")
        ax1.plot(drift["y"] * pixelsize, label="y")
        ax1.legend(loc="best")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Drift (nm)")
        ax2 = fig.add_subplot(122)
        ax2.plot(
            drift.x * pixelsize,
            drift.y * pixelsize,
            color=list(plt.rcParams["axes.prop_cycle"])[2]["color"],
        )
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        ax2.invert_yaxis()
        ax2.set_aspect("equal")
    return fig


def align(
    locs: list[pd.DataFrame],
    infos: list[dict],
    display: bool = False,
    *,
    apply_shifts: bool = True,
    return_shifts: bool = False,
) -> pd.DataFrame:
    """Align localizations from multiple channels (one per each element
    in `locs`) by calculating the shifts between the rendered images
    using RCC.

    TODO: v1.0: This should be the main function that uses align_rcc or
        align_from_picked.

    Parameters
    ----------
    locs : list of pd.DataFrames
        List of localization datasets, where each element is a
        DataFrame of localizations for a single image.
    infos : list of dicts
        List of metadata dictionaries corresponding to each
        localization array in `locs`.
    display : bool, optional
        Not used.
    apply_shifts : bool, optional
        If True, applies the calculated shifts to the 'x' and 'y'
        coordinates of the localizations. If False, returns the original
        localizations without applying the shifts. Default is True.
    return_shifts : bool, optional
        If True, also returns the calculated shifts for each channel.

    Returns
    -------
    locs : list of pd.DataFrames
        Aligned localizations with the shifts applied to the 'x' and
        'y' coordinates.
    shifts : tuple
        ``(shift_x, shift_y)`` per channel, in camera pixels. Returned only
        if ``return_shifts`` is True.
    """
    images = []
    disp_px_size = 100  # nm
    for locs_, info_ in zip(locs, infos):
        _, image = render.render(
            locs_, info_, disp_px_size=disp_px_size, blur_method="smooth"
        )
        images.append(image)
    shift_y, shift_x = imageprocess.rcc(
        images, callback=lib.MockProgress().set_value
    )
    # `rcc` returns shifts in rendered-image pixels (disp_px_size = 100 nm
    # per pixel). Convert them to camera pixels so that both the applied
    # and the returned shifts are in the same units as the localizations'
    # 'x' and 'y' coordinates.
    shift_x = np.asarray(shift_x, dtype=float)
    shift_y = np.asarray(shift_y, dtype=float)
    for i, info_ in enumerate(infos):
        pixelsize = lib.get_from_metadata(info_, "Pixelsize")
        oversampling = pixelsize / disp_px_size
        shift_x[i] /= oversampling
        shift_y[i] /= oversampling
    if apply_shifts:
        for locs_, dx, dy in zip(locs, shift_x, shift_y):
            locs_["y"] -= dy
            locs_["x"] -= dx
    if return_shifts:
        shifts = (shift_x, shift_y)
        return locs, shifts
    else:
        return locs


def align_rcc(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    display: bool = False,
    return_shifts: bool = False,
) -> pd.DataFrame:
    """Align localizations from multiple channels (one per each element
    in `locs`) by calculating the shifts between the rendered images
    using RCC. This is a wrapper around `align` for backward compatibility.

    Parameters
    ----------
    locs : list of pd.DataFrames
        List of localization datasets which are to be aligned.
    infos : list of list of dicts
        List of metadata dictionaries corresponding to each
        localization DataFrame in `locs`.
    display : bool, optional
        If True, displays the estimated shifts. Default is False.
    return_shifts : bool, optional
        If True, also returns the calculated shifts for each channel.
        Default is False.

    Returns
    -------
    locs : list of pd.DataFrames
        Aligned localizations with the shifts applied to the 'x' and
        'y' coordinates.
    shifts : tuple
        ``(shift_x, shift_y)`` per channel, in camera pixels. Returned only
        if ``return_shifts`` is True.
    """
    locs = deepcopy(locs)
    max_iterations = 5
    iteration = 0
    convergence = 0.001  # (camera pixels), around 0.1 nm
    shift_x = []
    shift_y = []
    shift_z = []
    for iteration in range(max_iterations):
        completed = True

        # find shift between channels
        shift = align(
            locs, infos, display=False, apply_shifts=False, return_shifts=True
        )[1]
        temp_shift_x = []
        temp_shift_y = []
        temp_shift_z = []
        for i, locs_ in enumerate(locs):
            if (
                np.absolute(shift[0][i]) + np.absolute(shift[1][i])
                > convergence
            ):
                completed = False

            # shift each channel
            locs_["x"] -= shift[0][i]
            locs_["y"] -= shift[1][i]

            temp_shift_x.append(shift[0][i])
            temp_shift_y.append(shift[1][i])

            if len(shift) == 3:
                locs_["z"] -= shift[2][i]
                temp_shift_z.append(shift[2][i])
        shift_x.append(np.mean(temp_shift_x))
        shift_y.append(np.mean(temp_shift_y))
        if len(shift) == 3:
            shift_z.append(np.mean(temp_shift_z))
        iteration += 1

        # Skip when converged:
        if completed:
            break

    # Plot shift
    if display:
        fig1 = plt.figure(figsize=(8, 8), constrained_layout=True)
        plt.suptitle("Shift")
        plt.subplot(1, 1, 1)
        plt.plot(shift_x, "o-", label="x shift")
        plt.plot(shift_y, "o-", label="y shift")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Shift per Iteration (Px)")
        plt.legend(loc="best")
        fig1.show()

    if return_shifts:
        shifts = list(zip(shift_x, shift_y))
        if len(shift) == 3:
            shifts = list(zip(shift_x, shift_y, shift_z))
        return locs, shifts
    else:
        return locs


def align_from_picked(
    all_locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None = None,
    return_shifts: bool = False,
    index_blocks: list[tuple | None] | None = None,
):
    """Align picked localizations from multiple channels using picked
    localizations.

    Parameters
    ----------
    all_locs : list of pd.DataFrames
        List of localization datasets.
    infos : list of list of dicts
        List of metadata dictionaries corresponding to each localization
        dataset in `all_locs`.
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
        in ``all_locs``, aligned by position) as returned by
        ``get_index_blocks``. Each entry may be ``None`` to recompute
        for that channel. Used to skip re-indexing inside circular
        ``picked_locs``. Ignored for non-circular pick shapes. Default
        is None (all channels recompute on demand).

    Returns
    -------
    aligned_locs : list of pd.DataFrames
        List of aligned localization datasets, where the localizations
        have been shifted according to the average shift calculated from
        the picked localizations.
    shifts : list of tuples
        List of (dx, dy) shifts applied to each localization dataset in
        `all_locs`, calculated as the average shift from the picked
        localizations. Returned only if `return_shifts` is True.
    """
    assert (
        pick_shape in lib.PICK_SHAPES
    ), f"pick_shape must be one of {lib.PICK_SHAPES}"
    if pick_shape not in lib.PICK_SHAPES_WITHOUT_SIZE:
        assert (
            pick_size is not None
        ), "pick_size must be provided when picks is a list of coordinates"
    if pick_shape == "Circle":
        pick_size = pick_size / 2  # convert diameter to radius
    ib_list = (
        index_blocks if index_blocks is not None else [None] * len(all_locs)
    )
    pl = [
        picked_locs(locs, i, picks, pick_shape, pick_size, index_blocks=ib)
        for locs, i, ib in zip(all_locs, infos, ib_list)
    ]
    dy = _shifts_from_picked_coordinate(pl, coordinate="y")
    dx = _shifts_from_picked_coordinate(pl, coordinate="x")
    if all(["z" in _[0].columns for _ in pl]):
        dz = _shifts_from_picked_coordinate(pl, coordinate="z")
    else:
        dz = None
    shift = lib.minimize_shifts(dx, dy, shifts_z=dz)

    # align each channel
    aligned_locs = []
    for i, locs_ in enumerate(all_locs):
        locs_.y -= shift[0][i]
        locs_.x -= shift[1][i]
        if len(shift) == 3:
            locs_.z -= shift[2][i]
        aligned_locs.append(locs_.copy())

    if return_shifts:
        return aligned_locs, shift
    else:
        return aligned_locs


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
    callback: Callable[[int], None] | Literal["console"] | None = None,
) -> pd.DataFrame:
    """Calculate group statistics for localizations, such as mean and
    standard deviation.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with a 'group' field that defines the groups.
    callback : callable, "console" or None, optional
        Callback function to report progress. It should accept an
        integer argument representing the current group index. If
        "console", uses tqdm to display progress in the console.
        Default is None, which means no progress is reported.

    Returns
    -------
    groups : pd.DataFrame
        Group statistics for each group in the localization list.
    """
    try:
        locs = locs[locs["dark"] != -1]
    except AttributeError:
        pass
    group_ids = np.unique(locs["group"])
    n = len(group_ids)
    names = ["group", "n_events"] + list(
        itertools.chain(*[(_ + "_mean", _ + "_std") for _ in locs.columns])
    )
    groups = pd.DataFrame(np.empty((n, len(names))), columns=names)

    # progress reporting
    use_tqdm = callback == "console"
    if use_tqdm:
        iter_range = tqdm(
            total=len(group_ids),
            desc="Calculating group statistics",
            unit="Groups",
        )
    else:
        iter_range = range(len(group_ids))

    for i in iter_range:
        if callable(callback):
            callback(i)
        group_id = group_ids[i]
        group_locs = locs[locs["group"] == group_id]
        groups.loc[i, "group"] = group_id
        groups.loc[i, "n_events"] = len(group_locs)
        for name in locs.columns:
            groups.loc[i, name + "_mean"] = group_locs[name].mean()
            groups.loc[i, name + "_std"] = group_locs[name].std()
    if callable(callback):  # close the progress dialog
        callback(len(group_ids))

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
    nn.reshape(-1, nn_count)  # ensure the shape is (N, nn_count)
    return nn


def resi(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
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
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
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
        Clustering radius in xy (camera pixels). If a single float is
        provided, it is applied to all channels. If a list, must have
        length equal to the number of channels.
    radius_z : float, list of float, or None, optional
        Clustering radius in z (camera pixels). Only used for 3D data.
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
    progress_callback : {callable, "console", None}, optional
        Callback function to report progress where the input integer is
        the index of the channel currently processed. If "console", uses
        a simple console print. If None, no progress is reported.
        Default is None.

    Returns
    -------
    resi_centers : pd.DataFrame
        Combined cluster centers from all channels. Contains all columns
        from the original localizations plus a 'resi_channel_id' column
        indicating which channel each cluster belongs to. The 'group'
        column is renamed to 'cluster_id'.
    resi_info : list of dicts
        Metadata for the RESI cluster centers, containing clustering
        parameters for each channel.

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
        progress_callback=progress_callback,
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
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Internal function to perform RESI analysis, assumes all
    parameters are in the correct format and that there are at least 2
    chennels. See `resi` for details."""
    ndim = 3 if all(["z" in locs_.columns for locs_ in locs]) else 2
    pixelsize = lib.get_from_metadata(infos[0], "Pixelsize", raise_error=True)

    # Process each channel
    resi_channels = []
    if progress_callback == "console":
        iter_range = tqdm(
            total=len(locs), desc="Processing channels", unit="Channels"
        )
    else:
        iter_range = range(len(locs))
    for i in iter_range:
        if callable(progress_callback):
            progress_callback(i)
        locs_ = locs[i]
        info_ = infos[i]
        r_xy = radius_xy[i]
        r_z = radius_z[i]
        min_locs_ = min_locs[i]

        # Cluster localizations for this channel
        clustered_locs, new_info = clusterer.cluster(
            locs_,
            radius_xy=r_xy,
            min_locs=min_locs_,
            frame_analysis=apply_fa,
            radius_z=r_z if ndim == 3 else None,
            pixelsize=pixelsize,
        )
        if ndim == 3:
            new_info["Clustering radius z (nm)"] = r_z * pixelsize

        # Save clustered localizations if requested
        if save_clustered_locs and output_paths is not None:
            save_path = (
                os.path.splitext(output_paths[i])[0] + f"{suffix_locs}.hdf5"
            )
            io.save_locs(save_path, clustered_locs, info_ + [new_info])

        # Extract cluster centers from clustered localizations
        centers = clusterer.find_cluster_centers(clustered_locs, pixelsize)

        # Save cluster centers if requested
        if save_cluster_centers and output_paths is not None:
            save_path = (
                os.path.splitext(output_paths[i])[0] + f"{suffix_centers}.hdf5"
            )
            io.save_locs(save_path, centers, info_ + [new_info])

        # Add RESI channel ID to identify which channel this cluster belongs to
        centers["resi_channel_id"] = i * np.ones(
            len(centers),
            dtype=np.int8,
        )
        resi_channels.append(centers)
    if callable(progress_callback):  # close the progress dialog
        progress_callback(len(locs))

    # Combine cluster centers from all channels; they carry the columns of
    # their channel's localizations, which need not all be the same
    all_resi = lib.concat_locs(resi_channels)

    # Rename 'group' to 'cluster_id' for clarity
    all_resi["cluster_id"] = all_resi["group"]
    all_resi.drop(columns=["group"], inplace=True)
    all_resi.sort_values(kind="quicksort", by="frame", inplace=True)

    new_info = {
        "Generated by": "RESI analysis",
        "Clustering radius xy (nm) for each channel": [
            float(r * pixelsize) for r in radius_xy
        ],
        "Min. number of locs in a cluster for each channel": list(min_locs),
        "Basic frame analysis": apply_fa,
    }
    if ndim == 3:
        new_info["Clustering radius z (nm) for each channel"] = [
            float(r * pixelsize) for r in radius_z
        ]
    new_info = infos[0] + [new_info]
    # Save combined RESI results if output paths are provided
    if resi_path is not None:
        io.save_locs(resi_path, all_resi, new_info)

    return all_resi, new_info
