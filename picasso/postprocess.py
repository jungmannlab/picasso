"""
    gui/postprocess
    ~~~~~~~~~~~~~~~~~~~~

    Data analysis of localization lists

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2015-2018
    :copyright: Copyright (c) 2015-2018 Jungmann Lab, MPI Biochemistry
"""

from collections.abc import Callable
from typing import Literal

import numpy as _np
import numba as _numba

from scipy import interpolate as _interpolate
from scipy.optimize import curve_fit as _curve_fit
from scipy.special import iv as _iv
from scipy.spatial import distance

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import multiprocessing as _multiprocessing
import matplotlib.pyplot as _plt
import itertools as _itertools
from collections import OrderedDict as _OrderedDict
from . import lib as _lib
from . import render as _render
from . import imageprocess as _imageprocess
from threading import Thread as _Thread
from tqdm import tqdm as _tqdm
from tqdm import trange as _trange
from numpy.lib.recfunctions import stack_arrays
from sklearn.neighbors import NearestNeighbors as NN


def get_index_blocks(
    locs: _np.recarray, 
    info: list[dict], 
    size: float,
) -> tuple:
    """Splits localizations into blocks of the given size. Used for fast
    localization indexing (e.g., for picking).
    
    Parameters
    ----------
    locs : np.recarray
        Localizations.
    info : list of dicts
        Metadata of the localizations list.
    size : float
        Size of the blocks.
    
    Returns
    -------
    locs : np.recarray
        Localizations in the specified blocks.
    size : float
        Size of the blocks.
    x_index : np.ndarray
        x indices of the localizations in the blocks.
    y_index : np.ndarray
        y indices of the localizations in the blocks.
    block_starts : np.ndarray
        Block start indices.
    block_ends : np.ndarray
        Block end indices.
    K : int
        Number of blocks in y direction.
    L : int
        Number of blocks in x direction.
    """

    locs = _lib.ensure_sanity(locs, info)
    # Sort locs by indices
    x_index = _np.uint32(locs.x / size)
    y_index = _np.uint32(locs.y / size)
    sort_indices = _np.lexsort([x_index, y_index])
    locs = locs[sort_indices]
    x_index = x_index[sort_indices]
    y_index = y_index[sort_indices]
    # Allocate block info arrays
    n_blocks_y, n_blocks_x = index_blocks_shape(info, size)
    block_starts = _np.zeros((n_blocks_y, n_blocks_x), dtype=_np.uint32)
    block_ends = _np.zeros((n_blocks_y, n_blocks_x), dtype=_np.uint32)
    K, L = block_starts.shape
    # Fill in block starts and ends
    thread = _Thread(
        target=_fill_index_blocks,
        args=(block_starts, block_ends, x_index, y_index),
    )
    thread.start()
    thread.join()
    return locs, size, x_index, y_index, block_starts, block_ends, K, L


def index_blocks_shape(info: list[dict], size: float) -> tuple[int, int]:
    """Returns the shape of the index grid, given the movie and grid
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
        Number of blocks in y direction and x direction.
    """

    n_blocks_x = int(_np.ceil(info[0]["Width"] / size))
    n_blocks_y = int(_np.ceil(info[0]["Height"] / size))
    n = (n_blocks_y, n_blocks_x)
    return n


@_numba.jit(nopython=True, nogil=True)
def n_block_locs_at(
    x: float, 
    y: float, 
    size: float, 
    K: int, 
    L: int, 
    block_starts: _np.ndarray, 
    block_ends: _np.ndarray,
) -> int:
    """Returns the number of localizations in the blocks around the
    given coordinates.
    
    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    size : float
        Size of the blocks.
    K : int
        Number of blocks in y direction.
    L : int
        Number of blocks in x direction.
    block_starts : np.ndarray
        Block start indices.
    block_ends : np.ndarray
        Block end indices.

    Returns
    -------
    n_block_locs : int
        Number of localizations in the blocks around the given
        coordinates.
    """
    
    x_index = _np.uint32(x / size)
    y_index = _np.uint32(y / size)
    step = 0
    for k in range(y_index - 1, y_index + 2):
        if 0 < k < K:
            for l in range(x_index - 1, x_index + 2):
                if 0 < l < L:
                    if step == 0:
                        n_block_locs = _np.uint32(block_ends[k][l] - block_starts[k][l])
                        step = 1
                    else:
                        n_block_locs += _np.uint32(block_ends[k][l] - block_starts[k][l])
    return n_block_locs


def get_block_locs_at(x: float, y: float, index_blocks: tuple) -> _np.ndarray:
    """Returns the localizations in the blocks around the given
    coordinates.

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    index_blocks : tuple
        Index blocks information.

    Returns
    -------
    _np.ndarray
        Localizations in the blocks around the given coordinates.
    """

    locs, size, _, _, block_starts, block_ends, K, L = index_blocks
    x_index = _np.uint32(x / size)
    y_index = _np.uint32(y / size) 
    indices = []
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for l in range(x_index - 1, x_index + 2):
                if 0 <= l < L:
                    indices.append(
                        list(range(block_starts[k, l], block_ends[k, l]))
                    )
    indices = list(_itertools.chain(*indices))
    return locs[indices]


@_numba.jit(nopython=True, nogil=True)
def _fill_index_blocks(
    block_starts: _np.ndarray,
    block_ends: _np.ndarray,
    x_index: _np.ndarray,
    y_index: _np.ndarray,
) -> None:
    """Fills the block starts and ends arrays with the indices of
    localizations in the blocks."""

    Y, X = block_starts.shape
    N = len(x_index)
    k = 0
    for i in range(Y):
        for j in range(X):
            k = _fill_index_block(
                block_starts, block_ends, N, x_index, y_index, i, j, k
            )


@_numba.jit(nopython=True, nogil=True)
def _fill_index_block(
    block_starts: _np.ndarray, 
    block_ends: _np.ndarray, 
    N: int, 
    x_index: _np.ndarray, 
    y_index: _np.ndarray, 
    i: int, 
    j: int, 
    k: int,
) -> int:
    """Fills the block starts and ends arrays for a single block."""

    block_starts[i, j] = k
    while k < N and y_index[k] == i and x_index[k] == j:
        k += 1
    block_ends[i, j] = k
    return k


def picked_locs(
    locs: _np.recarray, 
    info: list[dict], 
    picks: list[tuple], 
    pick_shape: str, 
    pick_size: float = None, 
    add_group: bool = True, 
    callback: Callable[[int], None] | Literal["console"] | None = None,
) -> list[_np.recarray]:
    """Finds picked localizations.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    picks : list
        List of picks.
    pick_shape : {'Circle', 'Rectangle', 'Polygon'}
        Shape of the pick.
    pick_size : float (default=None)
        Size of the pick. Radius for the circles, width for the 
        rectangles, None for the polygons.
    add_group : boolean (default=True)
        True if group id should be added to locs. Each pick will be
        assigned a different id.
    callback : function (default=None)
        Function to display progress. If "console", tqdm is used to
        display the progress. If None, no progress is displayed.

    Returns
    -------
    picked_locs : list of np.recarrays 
        List of np.recarrays, each containing locs from one pick.
    """

    if len(picks):
        picked_locs = []
        if callback == "console":
            progress = _tqdm(range(len(picks)), desc="Picking locs", unit="pick")

        if pick_shape == "Circle":
            index_blocks = get_index_blocks(locs, info, pick_size)
            for i, pick in enumerate(picks):
                x, y = pick
                block_locs = get_block_locs_at(
                    x, y, index_blocks
                )
                group_locs = _lib.locs_at(x, y, block_locs, pick_size)
                if add_group:
                    group = i * _np.ones(len(group_locs), dtype=_np.int32)
                    group_locs = _lib.append_to_rec(
                        group_locs, group, "group"
                    )
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        elif pick_shape == "Rectangle":
            for i, pick in enumerate(picks):
                (xs, ys), (xe, ye) = pick
                X, Y = _lib.get_pick_rectangle_corners(
                    xs, ys, xe, ye, pick_size
                )
                x_min = min(X)
                x_max = max(X)
                y_min = min(Y)
                y_max = max(Y)
                group_locs = locs[locs.x > x_min]
                group_locs = group_locs[group_locs.x < x_max]
                group_locs = group_locs[group_locs.y > y_min]
                group_locs = group_locs[group_locs.y < y_max]
                group_locs = _lib.locs_in_rectangle(group_locs, X, Y)
                # store rotated coordinates in x_rot and y_rot
                angle = 0.5 *_np.pi - _np.arctan2((ye - ys), (xe - xs))
                x_shifted = group_locs.x - xs
                y_shifted = group_locs.y - ys
                x_pick_rot = x_shifted * _np.cos(
                    angle
                ) - y_shifted * _np.sin(angle)
                y_pick_rot = x_shifted * _np.sin(
                    angle
                ) + y_shifted * _np.cos(angle)
                group_locs = _lib.append_to_rec(
                    group_locs, x_pick_rot, "x_pick_rot"
                )
                group_locs = _lib.append_to_rec(
                    group_locs, y_pick_rot, "y_pick_rot"
                )
                if add_group:
                    group = i * _np.ones(len(group_locs), dtype=_np.int32)
                    group_locs = _lib.append_to_rec(
                        group_locs, group, "group"
                    )
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        elif pick_shape == "Polygon":
            for i, pick in enumerate(picks):
                X, Y = _lib.get_pick_polygon_corners(pick)
                if X is None:
                    if callback == "console":
                        progress.update(1)
                    elif callback is not None:
                        callback(i + 1)
                    continue
                group_locs = locs[locs.x > min(X)]
                group_locs = group_locs[group_locs.x < max(X)]
                group_locs = group_locs[group_locs.y > min(Y)]
                group_locs = group_locs[group_locs.y < max(Y)]
                group_locs = _lib.locs_in_polygon(group_locs, X, Y)
                if add_group:
                    group = i * _np.ones(len(group_locs), dtype=_np.int32)
                    group_locs = _lib.append_to_rec(
                        group_locs, group, "group"
                    )
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        else:
            raise ValueError(
                "Invalid pick shape. Please choose from 'Circle', 'Rectangle', "
                "'Polygon'."
            )
        
        return picked_locs


@_numba.jit(nopython=True, nogil=True, cache=True)
def pick_similar(
    x: _np.ndarray, 
    y_shift: _np.ndarray, 
    y_base: _np.ndarray,
    min_n_locs: int, 
    max_n_locs: int, 
    min_rmsd: float, 
    max_rmsd: float, 
    x_r: _np.ndarray, 
    y_r1: _np.ndarray, 
    y_r2: _np.ndarray,
    locs_xy: _np.ndarray, 
    block_starts: _np.ndarray, 
    block_ends: _np.ndarray, 
    K: int, 
    L: int,
    x_similar: _np.ndarray, 
    y_similar: _np.ndarray, 
    r: float, 
    d2: float,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Finds similar picks based on the number of localizations and 
    RMSD.
    
    Parameters
    ----------
    x : np.ndarray
        x coordinates of the picks.
    y_shift : np.ndarray
        y coordinates of the picks, shifted for odd columns.
    y_base : np.ndarray
        y coordinates of the picks, not shifted.
    min_n_locs, max_n_locs : int
        Minimum and maximum number of localizations in the pick.
    min_rmsd, max_rmsd : float
        Minimum and maximum RMSD for the pick.
    x_r, y_r1, y_r2 : np.ndarray
        x and y ranges for the picks.
    locs_xy : np.ndarray
        Localizations in the blocks.
    block_starts : np.ndarray
        Block start indices.
    block_ends : np.ndarray
        Block end indices.
    K, L : int
        Number of blocks in y and x direction.
    x_similar, y_similar : np.ndarray
        Arrays to store the x and y coordinates of the similar picks.
    r : float
        Radius for the picks.
    d2 : float
        Squared distance threshold for the picks.
    
    Returns
    -------
    x_similar, y_similar : np.ndarray
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
                x_range, y_range, K, L, block_starts, block_ends
            )
            if n_block_locs >= min_n_locs:
                block_locs_xy = _get_block_locs_at(
                    x_range, y_range, 
                    locs_xy, block_starts, block_ends, K, L,
                )
                picked_locs_xy = _locs_at(
                    x_grid, y_grid, block_locs_xy, r
                )
                if picked_locs_xy.shape[1] > 1:
                    # Move to COM peak
                    x_test_old = x_grid
                    y_test_old = y_grid
                    x_test = _np.mean(picked_locs_xy[0])
                    y_test = _np.mean(picked_locs_xy[1])
                    count = 0
                    while (
                        _np.abs(x_test - x_test_old) > 1e-3
                        or _np.abs(y_test - y_test_old) > 1e-3
                    ):
                        count += 1
                        # skip the locs if the loop is too long
                        if count > 500:
                            break
                        x_test_old = x_test
                        y_test_old = y_test
                        picked_locs_xy = _locs_at(
                            x_test, y_test, block_locs_xy, r
                        )
                        if picked_locs_xy.shape[1] > 1:
                            x_test = _np.mean(picked_locs_xy[0])
                            y_test = _np.mean(picked_locs_xy[1])
                        else:
                            break
                    if _np.all(
                        (x_similar - x_test) ** 2
                        + (y_similar - y_test) ** 2
                        > d2
                    ):
                        if min_n_locs <= picked_locs_xy.shape[1] <= max_n_locs:
                            if (
                                min_rmsd
                                <= _rmsd_at_com(picked_locs_xy)
                                <= max_rmsd
                            ):
                                x_similar = _np.append(
                                    x_similar, x_test
                                )
                                y_similar = _np.append(
                                    y_similar, y_test
                                )
    return x_similar, y_similar


@_numba.jit(nopython=True, nogil=True)
def _n_block_locs_at(
    x_range: int, 
    y_range: int, 
    K: int, 
    L: int, 
    block_starts: _np.ndarray, 
    block_ends: _np.ndarray,
) -> int:
    """Returns the number of localizations in the blocks around the
    given coordinates."""
    
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for l in range(x_range - 1, x_range + 2):
                if 0 < l < L:
                    if step == 0:
                        n_block_locs = _np.uint32(block_ends[k][l] - block_starts[k][l])
                        step = 1
                    else:
                        n_block_locs += _np.uint32(block_ends[k][l] - block_starts[k][l])
    return n_block_locs


@_numba.jit(nopython=True, nogil=True, cache=True)
def _get_block_locs_at(
    x_range: int, 
    y_range: int, 
    locs_xy: _np.ndarray, 
    block_starts: _np.ndarray, 
    block_ends: _np.ndarray, 
    K: int, 
    L: int,
) -> _np.ndarray:
    """Returns the localizations in the blocks around the given 
    coordinates."""

    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for l in range(x_range - 1, x_range + 2):
                if 0 < l < L:
                    if block_ends[k, l] - block_starts[k, l] > 0:
                        # numba does not work if you attach arange to an empty list so the first step is different
                        # this is because of dtype issues
                        if step == 0:
                            indices = _np.arange(float(block_starts[k, l]), float(block_ends[k, l]), 
                                dtype=_np.uint32)
                            step = 1
                        else:
                            indices = _np.concatenate((indices, 
                                _np.arange(float(block_starts[k, l]), float(block_ends[k, l]), dtype=_np.uint32)
                            ))
    return locs_xy[:, indices]


@_numba.jit(nopython=True, nogil=True, cache=True)
def _locs_at(x: float, y: float, locs_xy: _np.ndarray, r: float) -> _np.ndarray:
    """Returns the localizations at the given coordinates within radius
    r."""

    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r ** 2
    is_picked = dx ** 2 + dy ** 2 < r2
    return locs_xy[:, is_picked]


@_numba.jit(nopython=True, nogil=True)
def _rmsd_at_com(locs_xy: _np.ndarray) -> float:
    """Calculates the RMSD of the localizations at the center of mass
    (COM) of the localizations."""

    com_x = _np.mean(locs_xy[0])
    com_y = _np.mean(locs_xy[1])
    return _np.sqrt(_np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2))


@_numba.jit(nopython=True, nogil=True)
def _distance_histogram(
    locs: _np.recarray,
    bin_size: float,
    r_max: float,
    x_index: _np.ndarray,
    y_index: _np.ndarray,
    block_starts: _np.ndarray,
    block_ends: _np.ndarray,
    start: int,
    chunk: int,
) -> _np.ndarray:
    """Calculates the distance histogram for a chunk of 
    localizations."""

    x = locs.x
    y = locs.y
    dh_len = _np.uint32(r_max / bin_size)
    dh = _np.zeros(dh_len, dtype=_np.uint32)
    r_max_2 = r_max**2
    K, L = block_starts.shape
    end = min(start + chunk, len(locs))
    for i in range(start, end):
        xi = x[i]
        yi = y[i]
        ki = y_index[i]
        li = x_index[i]
        for k in range(ki, ki + 2):
            if k < K:
                for l in range(li, li + 2):
                    if l < L:
                        for j in range(block_starts[k, l], block_ends[k, l]):
                            if j > i:
                                dx2 = (xi - x[j]) ** 2
                                if dx2 < r_max_2:
                                    dy2 = (yi - y[j]) ** 2
                                    if dy2 < r_max_2:
                                        d = _np.sqrt(dx2 + dy2)
                                        if d < r_max:
                                            bin = _np.uint32(d / bin_size)
                                            if bin < dh_len:
                                                dh[bin] += 1
    return dh


def distance_histogram(
    locs: _np.recarray, 
    info: list[dict], 
    bin_size: float, 
    r_max: float,
) -> _np.ndarray:
    """Calculates the distance histogram for the given localizations, 
    i.e., the pairwise distances between localizations.

    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list[dict]
        Metadata of the localizations list.
    bin_size : float
        Size of the bins for the histogram.
    r_max : float
        Maximum distance for the histogram.
    
    Returns
    -------
    dh : np.ndarray
        Distance histogram.
    """

    locs, size, x_index, y_index, b_starts, b_ends, K, L = get_index_blocks(
        locs, info, r_max
    )
    N = len(locs)
    n_threads = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (
            locs,
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
    dh = _np.sum(results, axis=0)
    return dh


def nena(
    locs: _np.recarray, 
    info: None, 
    callback: Callable[[int], None] | None = None,
) -> tuple[dict, float]:
    """Calculates NeNA - experimental estimate of localization 
    precision. Please refer to the original paper for details:
    Endesfelder, et al. Histochemistry and Cell Biology, 2014.
    DOI: 10.1007/s00418-014-1192-3.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : None
        Metadata of the localizations list. Not used.
    callback : function or None
        Function to display progress. If None, no progress is displayed.
        
    Returns
    -------
    result : dict
        Data on the results, including the distances probed, best fit
        and fitted parameters.
    s : float
        Estimated localization precision.
    """

    bin_centers, dnfl = next_frame_neighbor_distance_histogram(locs, callback)

    def func(d, delta_a, s, ac, dc, sc):
        a = ac + delta_a # make sure a >= ac
        p_single = a * (d / (2 * s**2)) * _np.exp(-d**2 / (4* s**2))
        p_short = (
            ac / (sc * _np.sqrt(2 * _np.pi)) *
            _np.exp(-0.5 * ((d - dc) / sc)**2)
        )
        return p_single + p_short
    
    area = _np.trapz(dnfl, bin_centers)
    median_lp = _np.mean([_np.median(locs.lpx), _np.median(locs.lpy)])
    p0 = [0.8*area, median_lp, 0.1*area, 2*median_lp, median_lp]
    bounds = ([0, 0, 0, 0, 0], [_np.inf, _np.inf, _np.inf, _np.inf, _np.inf])
    popt, _ = _curve_fit(func, bin_centers, dnfl, p0=p0, bounds=bounds)
    s = popt[1] # NeNA
    result = {
        "d": bin_centers, # distances probed
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


def next_frame_neighbor_distance_histogram(
    locs: _np.recarray, 
    callback: Callable[[int], None] | None = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Calculates the next frame neighbor distance histogram (NFNDH).

    Parameters
    ----------
    locs : np.recarray
        Localization list.
    callback : function or None
        Function to display progress. If None, no progress is displayed.
    
    Returns
    -------
    bin_centers : np.ndarray
        Centers of the bins for the histogram.
    dnfl : np.ndarray
        Distance histogram of next frame neighbors.
    """

    locs.sort(kind="mergesort", order="frame")
    frame = locs.frame
    x = locs.x
    y = locs.y
    if hasattr(locs, "group"):
        group = locs.group
    else:
        group = _np.zeros(len(locs), dtype=_np.int32)
    bin_size = 0.001
    d_max = 1.0
    return _nfndh(frame, x, y, group, d_max, bin_size, callback)


def _nfndh(
    frame: _np.ndarray, 
    x: _np.ndarray, 
    y: _np.ndarray, 
    group: _np.ndarray, 
    d_max: float, 
    bin_size: float, 
    callback: Callable[[int], None] | None = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Calculates the next frame neighbor distance histogram (NFNDH)."""

    N = len(frame)
    bins = _np.arange(0, d_max, bin_size)
    dnfl = _np.zeros(len(bins))
    one_percent = int(N / 100)
    starts = one_percent * _np.arange(100)
    for k, start in enumerate(starts):
        for i in range(start, start + one_percent):
            _fill_dnfl(N, frame, x, y, group, i, d_max, dnfl, bin_size)
        if callback is not None:
            callback(k + 1)
    bin_centers = bins + bin_size / 2
    return bin_centers, dnfl


@_numba.jit(nopython=True)
def _fill_dnfl(
    N: int, 
    frame: _np.ndarray, 
    x: _np.ndarray, 
    y: _np.ndarray, 
    group: _np.ndarray, 
    i: int, 
    d_max: float, 
    dnfl: _np.ndarray, 
    bin_size: float,
) -> None:
    """Fills the next frame neighbor distance histogram (NFNDH) for a
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
                    d = _np.sqrt(dx2 + dy2)
                    if d <= d_max:
                        bin = int(d / bin_size)
                        dnfl[bin] += 1


def pair_correlation(
    locs: _np.recarray, 
    info: list[dict], 
    bin_size: float, 
    r_max: float,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Calculates the pair correlation function for the given
    localizations.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    bin_size : float
        Size of the bins for the histogram.
    r_max : float
        Maximum distance for the histogram.
    
    Returns
    -------
    bins_lower : np.ndarray
        Lower bounds of the bins for the histogram.
    dh : np.ndarray
        Pair correlation function.
    """

    dh = distance_histogram(locs, info, bin_size, r_max)
    # Start with r-> otherwise area will be 0
    bins_lower = _np.arange(bin_size, r_max + bin_size, bin_size)

    if bins_lower.shape[0] > dh.shape[0]:
        bins_lower = bins_lower[:-1]
    area = _np.pi * bin_size * (2 * bins_lower + bin_size)
    return bins_lower, dh / area


@_numba.jit(nopython=True, nogil=True)
def _local_density(
    locs: _np.recarray, 
    radius: float, 
    x_index: _np.ndarray, 
    y_index: _np.ndarray, 
    block_starts: _np.ndarray, 
    block_ends: _np.ndarray, 
    start: int, 
    chunk: int,
) -> _np.ndarray:
    """Calculates densities in blocks around each localization."""

    x = locs.x
    y = locs.y
    N = len(x)
    r2 = radius**2
    end = min(start + chunk, N)
    density = _np.zeros(N, dtype=_np.uint32)
    for i in range(start, end):
        yi = y[i]
        xi = x[i]
        ki = y_index[i]
        li = x_index[i]
        di = 0
        for k in range(ki - 1, ki + 2):
            for l in range(li - 1, li + 2):
                j_min = block_starts[k, l]
                j_max = block_ends[k, l]
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
    locs: _np.recarray, 
    info: list[dict], 
    radius: float,
) -> _np.recarray:
    """Computes the local density of localizations in blocks around
    each localization.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    radius : float
        Radius for local density computation.
    
    Returns
    -------
    locs : np.recarray
        Localization list with added 'density' field/column.
    """

    locs, x_index, y_index, block_starts, block_ends, K, L = get_index_blocks(
        locs, info, radius
    )
    N = len(locs)
    n_threads = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (
            locs,
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
    density = _np.sum([future.result() for future in futures], axis=0)
    locs = _lib.remove_from_rec(locs, "density")
    return _lib.append_to_rec(locs, density, "density")


def compute_dark_times(
    locs: _np.recarray, 
    group: _np.ndarray | None = None,
) -> _np.recarray:
    """Computes dark time for each binding event.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list that were linked, i.e., binding events.
    group : np.ndarray, optional
        Grouping array for binding events. If None, all binding events
        are considered to be in the same group.
    
    Returns
    -------
    locs : np.recarray
        binding events with added 'dark' field/column, which contains
        the dark time for each binding event. If a binding event is not
        followed by another binding event in the same group, the dark
        time is set to -1.
    """

    if "len" not in locs.dtype.names:
        raise AttributeError("Length not found. Please link localizations first.")
    dark = dark_times(locs, group)
    locs = _lib.append_to_rec(locs, _np.int32(dark), "dark")
    locs = locs[locs.dark != -1]
    return locs


def dark_times(
    locs: _np.recarray, 
    group: _np.ndarray | None = None,
) -> _np.ndarray:
    """Calculates dark times for each binding event.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list that were linked, i.e., binding events.
    group : np.ndarray, optional
        Grouping array for binding events. If None, all binding events
        are considered to be in the same group.
    
    Returns
    -------
    dark : np.ndarray
        Array of dark times for each binding event. If a binding event
        is not followed by another binding event in the same group, the
        dark time is set to -1.
    """
    
    last_frame = locs.frame + locs.len - 1
    if group is None:
        if hasattr(locs, "group"):
            group = locs.group
        else:
            group = _np.zeros(len(locs))
    dark = _dark_times(locs, group, last_frame)
    return dark


@_numba.jit(nopython=True)
def _dark_times(
    locs: _np.recarray, 
    group: _np.ndarray, 
    last_frame: _np.ndarray,
) -> _np.ndarray:
    """Calculates dark times for each binding event."""

    N = len(locs)
    max_frame = locs.frame.max()
    dark = max_frame * _np.ones(len(locs), dtype=_np.int32)
    for i in range(N):
        for j in range(N):
            if (group[i] == group[j]) and (i != j):
                dark_ij = locs.frame[i] - last_frame[j]
                if (dark_ij > 0) and (dark_ij < dark[i]):
                    dark[i] = dark_ij
    for i in range(N):
        if dark[i] == max_frame:
            dark[i] = -1
    return dark


def link(
    locs: _np.recarray,
    info: list[dict],
    r_max: float = 0.05,
    max_dark_time: int = 1,
    combine_mode: Literal["average", "refit"] = "average",
    remove_ambiguous_lengths: bool = True,
) -> _np.recarray:
    """Links localizations, i.e., groups them into binding events based
    on their spatiotemporal proximity.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    r_max : float, optional
        Maximum distance for linking localizations. Default is 0.05.
    max_dark_time : int, optional
        Maximum dark time for linking localizations. Default is 1.
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
    linked_locs : np.recarray
        Linked localizations, i.e., binding events with their 
        properties.
    """

    if len(locs) == 0:  # special case of an empty localization list
        linked_locs = locs.copy()
        if hasattr(locs, "frame"):
            linked_locs = _lib.append_to_rec(
                linked_locs, _np.array([], dtype=_np.int32), "len"
            )
            linked_locs = _lib.append_to_rec(
                linked_locs, _np.array([], dtype=_np.int32), "n"
            )
        if hasattr(locs, "photons"):
            linked_locs = _lib.append_to_rec(
                linked_locs, _np.array([], dtype=_np.float32), "photon_rate"
            )
    else:
        locs.sort(kind="mergesort", order="frame")
        if hasattr(locs, "group"):
            group = locs.group
        else:
            group = _np.zeros(len(locs), dtype=_np.int32)
        link_group = get_link_groups(locs, r_max, max_dark_time, group)
        if combine_mode == "average":
            linked_locs = link_loc_groups(
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


# def weighted_variance(locs):
#     n = len(locs)
#     w = locs.photons
#     x = locs.x
#     y = locs.y
#     xWbarx = _np.average(locs.x, weights=w)
#     xWbary = _np.average(locs.y, weights=w)
#     wbarx = _np.mean(locs.lpx)
#     wbary = _np.mean(locs.lpy)
#     variance_x = (
#         n
#         / ((n - 1) * sum(w) ** 2)
#         * (
#             sum((w * x - wbarx * xWbarx) ** 2)
#             - 2 * xWbarx * sum((w - wbarx) * (w * x - wbarx * xWbarx))
#             + xWbarx**2 * sum((w - wbarx) ** 2)
#         )
#     )
#     variance_y = (
#         n
#         / ((n - 1) * sum(w) ** 2)
#         * (
#             sum((w * y - wbary * xWbary) ** 2)
#             - 2 * xWbary * sum((w - wbary) * (w * y - wbary * xWbary))
#             + xWbary**2 * sum((w - wbary) ** 2)
#         )
#     )
#     return variance_x, variance_y


# Combine localizations: calculate the properties of the group
def cluster_combine(locs: _np.recarray) -> _np.recarray:
    """Combines localizations into clusters and calculates their
    properties such as center of mass, standard deviation, and number
    of localizations in each cluster.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list with 'group' and 'cluster' fields.
    Returns
    -------
    combined_locs : np.recarray
        Combined localizations with calculated properties for each
        cluster.
    """

    print("Combining localizations...")
    combined_locs = []
    if hasattr(locs[0], "z"):
        print("z-mode")
        for group in _tqdm(_np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = _np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = _np.zeros(n_cluster)
            std_frame = _np.zeros(n_cluster)
            com_x = _np.zeros(n_cluster)
            com_y = _np.zeros(n_cluster)
            com_z = _np.zeros(n_cluster)
            std_x = _np.zeros(n_cluster)
            std_y = _np.zeros(n_cluster)
            std_z = _np.zeros(n_cluster)
            group_id = _np.zeros(n_cluster)
            n = _np.zeros(n_cluster, dtype=_np.int32)
            for i, clusterval in enumerate(cluster):
                cluster_locs = temp[temp["cluster"] == clusterval]
                mean_frame[i] = _np.mean(cluster_locs.frame)
                com_x[i] = _np.average(cluster_locs.x, weights=cluster_locs.photons)
                com_y[i] = _np.average(cluster_locs.y, weights=cluster_locs.photons)
                com_z[i] = _np.average(cluster_locs.z, weights=cluster_locs.photons)
                std_frame[i] = _np.std(cluster_locs.frame)
                std_x[i] = _np.std(cluster_locs.x) / _np.sqrt(len(cluster_locs))
                std_y[i] = _np.std(cluster_locs.y) / _np.sqrt(len(cluster_locs))
                std_z[i] = _np.std(cluster_locs.z) / _np.sqrt(len(cluster_locs))
                n[i] = len(cluster_locs)
                group_id[i] = group
            clusters = _np.rec.array(
                (
                    group_id,
                    cluster,
                    mean_frame,
                    com_x,
                    com_y,
                    com_z,
                    std_frame,
                    std_x,
                    std_y,
                    std_z,
                    n,
                ),
                dtype=[
                    ("group", group.dtype),
                    ("cluster", cluster.dtype),
                    ("mean_frame", "f4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("std_frame", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("lpz", "f4"),
                    ("n", "i4"),
                ],
            )
            combined_locs.append(clusters)
    else:
        for group in _tqdm(_np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = _np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = _np.zeros(n_cluster)
            std_frame = _np.zeros(n_cluster)
            com_x = _np.zeros(n_cluster)
            com_y = _np.zeros(n_cluster)
            std_x = _np.zeros(n_cluster)
            std_y = _np.zeros(n_cluster)
            group_id = _np.zeros(n_cluster)
            n = _np.zeros(n_cluster, dtype=_np.int32)
            for i, clusterval in enumerate(cluster):
                cluster_locs = temp[temp["cluster"] == clusterval]
                mean_frame[i] = _np.mean(cluster_locs.frame)
                com_x[i] = _np.average(cluster_locs.x, weights=cluster_locs.photons)
                com_y[i] = _np.average(cluster_locs.y, weights=cluster_locs.photons)
                std_frame[i] = _np.std(cluster_locs.frame)
                std_x[i] = _np.std(cluster_locs.x) / _np.sqrt(len(cluster_locs))
                std_y[i] = _np.std(cluster_locs.y) / _np.sqrt(len(cluster_locs))
                n[i] = len(cluster_locs)
                group_id[i] = group
            clusters = _np.rec.array(
                (
                    group_id,
                    cluster,
                    mean_frame,
                    com_x,
                    com_y,
                    std_frame,
                    std_x,
                    std_y,
                    n,
                ),
                dtype=[
                    ("group", group.dtype),
                    ("cluster", cluster.dtype),
                    ("mean_frame", "f4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("std_frame", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("n", "i4"),
                ],
            )
            combined_locs.append(clusters)

    combined_locs = stack_arrays(combined_locs, asrecarray=True, usemask=False)

    return combined_locs


def cluster_combine_dist(locs: _np.recarray) -> _np.recarray:
    """Similar to `cluster_combine`, but also calculates the distance to 
    the nearest neighbor in the same group and the distance to the
    nearest neighbor in the same cluster in the same group.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list with 'group' and 'cluster' fields.

    Returns
    -------
    combined_locs : np.recarray
        Combined localizations with calculated properties for each
        cluster, including distances to nearest neighbors.
    """
    
    print("Calculating distances...")

    if hasattr(locs, "z"):
        print("XYZ")

        combined_locs = []
        for group in _tqdm(_np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = _np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = temp["mean_frame"]
            std_frame = temp["std_frame"]
            com_x = temp["x"]
            com_y = temp["y"]
            com_z = temp["z"]
            std_x = temp["lpx"]
            std_y = temp["lpy"]
            std_z = temp["lpz"]
            group_id = temp["group"]
            n = temp["n"]
            min_dist = _np.zeros(n_cluster)
            min_dist_xy = _np.zeros(n_cluster)
            for i, clusterval in enumerate(cluster):
                # find nearest neighbor in xyz
                group_locs = temp[temp["cluster"] != clusterval]
                cluster_locs = temp[temp["cluster"] == clusterval]
                ref_point = _np.array(
                    [
                        cluster_locs.x,
                        cluster_locs.y,
                        cluster_locs.z,
                    ]
                )
                all_points = _np.array(
                    [group_locs.x, group_locs.y, group_locs.z]
                )
                distances = distance.cdist(
                    ref_point.transpose(), all_points.transpose()
                )
                min_dist[i] = _np.amin(distances)
                # find nearest neighbor in xy
                ref_point_xy = _np.array([cluster_locs.x, cluster_locs.y])
                all_points_xy = _np.array([group_locs.x, group_locs.y])
                distances_xy = distance.cdist(
                    ref_point_xy.transpose(), all_points_xy.transpose()
                )
                min_dist_xy[i] = _np.amin(distances_xy)

            clusters = _np.rec.array(
                (
                    group_id,
                    cluster,
                    mean_frame,
                    com_x,
                    com_y,
                    com_z,
                    std_frame,
                    std_x,
                    std_y,
                    std_z,
                    n,
                    min_dist,
                    min_dist_xy,
                ),
                dtype=[
                    ("group", group.dtype),
                    ("cluster", cluster.dtype),
                    ("mean_frame", "f4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("std_frame", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("lpz", "f4"),
                    ("n", "i4"),
                    ("min_dist", "f4"),
                    ("mind_dist_xy", "f4"),
                ],
            )
            combined_locs.append(clusters)

    else:  # 2D case
        print("XY")
        combined_locs = []
        for group in _tqdm(_np.unique(locs["group"])):
            temp = locs[locs["group"] == group]
            cluster = _np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = temp["mean_frame"]
            std_frame = temp["std_frame"]
            com_x = temp["x"]
            com_y = temp["y"]
            std_x = temp["lpx"]
            std_y = temp["lpy"]
            group_id = temp["group"]
            n = temp["n"]
            min_dist = _np.zeros(n_cluster)

            for i, clusterval in enumerate(cluster):
                # find nearest neighbor in xyz
                group_locs = temp[temp["cluster"] != clusterval]
                cluster_locs = temp[temp["cluster"] == clusterval]
                ref_point_xy = _np.array([cluster_locs.x, cluster_locs.y])
                all_points_xy = _np.array([group_locs.x, group_locs.y])
                distances_xy = distance.cdist(
                    ref_point_xy.transpose(), all_points_xy.transpose()
                )
                min_dist[i] = _np.amin(distances_xy)

            clusters = _np.rec.array(
                (
                    group_id,
                    cluster,
                    mean_frame,
                    com_x,
                    com_y,
                    std_frame,
                    std_x,
                    std_y,
                    n,
                    min_dist,
                ),
                dtype=[
                    ("group", group.dtype),
                    ("cluster", cluster.dtype),
                    ("mean_frame", "f4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("std_frame", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("n", "i4"),
                    ("min_dist", "f4"),
                ],
            )
            combined_locs.append(clusters)

    combined_locs = stack_arrays(combined_locs, asrecarray=True, usemask=False)
    return combined_locs


@_numba.jit(nopython=True)
def get_link_groups(
    locs: _np.recarray,
    d_max: float,
    max_dark_time: float,
    group: _np.ndarray,
) -> _np.ndarray:
    """Finds the groups for linking localizations into binding events.
    Assumes that locs are sorted by frame.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list that were linked, i.e., binding events.
    d_max : float
        Maximum distance for linking localizations.
    max_dark_time : float
        Maximum dark time for linking localizations.
    group : np.ndarray
        Grouping array for binding events. If None, all binding events
        are considered to be in the same group.
    
    Returns
    -------
    link_group : np.ndarray
        Array of link groups for each localization. Each group is
        represented by a unique integer. Localizations that are not
        linked to any other localization are assigned -1.
    """

    frame = locs.frame
    x = locs.x
    y = locs.y
    N = len(x)
    link_group = -_np.ones(N, dtype=_np.int32)
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


@_numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(
    current_index: int, 
    link_group: _np.ndarray,
    N: int,
    frame: _np.ndarray,
    x: _np.ndarray,
    y: _np.ndarray,
    d_max: float,
    max_dark_time: float,
    group: _np.ndarray,
) -> int:
    """Finds the next localization index in the link group for a given
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


@_numba.jit(nopython=True)
def _link_group_count(
    link_group: _np.ndarray, 
    n_locs: int, 
    n_groups: int,
) -> _np.ndarray:
    """Counts the number of localizations in each link group."""
    
    result = _np.zeros(n_groups, dtype=_np.uint32)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += 1
    return result


@_numba.jit(nopython=True)
def _link_group_sum(
    column: _np.ndarray, 
    link_group: _np.ndarray, 
    n_locs: int, 
    n_groups: int,
) -> _np.ndarray:
    """Sums the values of a column for each link group."""
    
    result = _np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += column[i]
    return result


@_numba.jit(nopython=True)
def _link_group_mean(
    column: _np.ndarray, 
    link_group: _np.ndarray, 
    n_locs: int, 
    n_groups: int, 
    n_locs_per_group: _np.ndarray
) -> _np.ndarray:
    """Calculates the mean of a column for each link group."""

    group_sum = _link_group_sum(column, link_group, n_locs, n_groups)
    result = _np.empty(
        n_groups, dtype=_np.float32
    )  # this ensures float32 after the division
    result[:] = group_sum / n_locs_per_group
    return result


@_numba.jit(nopython=True)
def _link_group_weighted_mean(
    column: _np.ndarray, 
    weights: _np.ndarray, 
    link_group: _np.ndarray, 
    n_locs: int, 
    n_groups: int, 
    n_locs_per_group: _np.ndarray
) -> tuple[_np.ndarray, _np.ndarray]:
    """Calculates the mean of a column for each link group and the sum
    of the weights."""
    
    sum_weights = _link_group_sum(weights, link_group, n_locs, n_groups)
    return (
        _link_group_mean(column * weights, link_group, n_locs, n_groups, sum_weights),
        sum_weights,
    )


@_numba.jit(nopython=True)
def _link_group_min_max(
    column: _np.ndarray, 
    link_group: _np.ndarray, 
    n_locs: int, 
    n_groups: int,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Calculates the minimum and maximum of a column for each link
    group."""

    min_ = _np.empty(n_groups, dtype=column.dtype)
    max_ = _np.empty(n_groups, dtype=column.dtype)
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


@_numba.jit(nopython=True)
def _link_group_last(
    column: _np.ndarray, 
    link_group: _np.ndarray, 
    n_locs: int, 
    n_groups: int,
) -> _np.ndarray:
    """Returns the last value of a column for each link group."""

    result = _np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] = column[i]
    return result


def link_loc_groups(
    locs: _np.recarray, 
    info: list[dict], 
    link_group: _np.ndarray, 
    remove_ambiguous_lengths: bool = True,
) -> _np.recarray:
    """Combines localizations into binding events based on the
    spatiotemporal proximity defined by the link_group. Takes the
    average position to calculate the coordinates of the binding events.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localization list.
    link_group : np.ndarray
        Array that defines the link groups for the localizations.
    remove_ambiguous_lengths : bool, optional
        If True, removes linked localizations with ambiguous lengths,
        i.e., localizations that are linked to multiple binding events
        with different lengths. Default is True.
        
    Returns
    -------
    linked_locs : np.recarray
        Linked localizations, i.e., binding events with their 
        properties.
    """

    n_locs = len(link_group)
    n_groups = link_group.max() + 1
    n_ = _link_group_count(link_group, n_locs, n_groups)
    columns = _OrderedDict()
    if hasattr(locs, "frame"):
        first_frame_, last_frame_ = _link_group_min_max(
            locs.frame, link_group, n_locs, n_groups
        )
        columns["frame"] = first_frame_
    if hasattr(locs, "x"):
        weights_x = 1 / locs.lpx**2
        columns["x"], sum_weights_x_ = _link_group_weighted_mean(
            locs.x, weights_x, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "y"):
        weights_y = 1 / locs.lpy**2
        columns["y"], sum_weights_y_ = _link_group_weighted_mean(
            locs.y, weights_y, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "photons"):
        columns["photons"] = _link_group_sum(locs.photons, link_group, n_locs, n_groups)
    if hasattr(locs, "sx"):
        columns["sx"] = _link_group_mean(locs.sx, link_group, n_locs, n_groups, n_)
    if hasattr(locs, "sy"):
        columns["sy"] = _link_group_mean(locs.sy, link_group, n_locs, n_groups, n_)
    if hasattr(locs, "bg"):
        columns["bg"] = _link_group_sum(locs.bg, link_group, n_locs, n_groups)
    if hasattr(locs, "x"):
        columns["lpx"] = _np.sqrt(1 / sum_weights_x_)
    if hasattr(locs, "y"):
        columns["lpy"] = _np.sqrt(1 / sum_weights_y_)
    if hasattr(locs, "ellipticity"):
        columns["ellipticity"] = _link_group_mean(
            locs.ellipticity, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "net_gradient"):
        columns["net_gradient"] = _link_group_mean(
            locs.net_gradient, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "likelihood"):
        columns["likelihood"] = _link_group_mean(
            locs.likelihood, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "iterations"):
        columns["iterations"] = _link_group_mean(
            locs.iterations, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "z"):
        columns["z"] = _link_group_mean(locs.z, link_group, n_locs, n_groups, n_)
    if hasattr(locs, "d_zcalib"):
        columns["d_zcalib"] = _link_group_mean(
            locs.d_zcalib, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "group"):
        columns["group"] = _link_group_last(locs.group, link_group, n_locs, n_groups)
    if hasattr(locs, "frame"):
        columns["len"] = last_frame_ - first_frame_ + 1
    columns["n"] = n_
    if hasattr(locs, "photons"):
        columns["photon_rate"] = _np.float32(columns["photons"] / n_)
    linked_locs = _np.rec.array(list(columns.values()), names=list(columns.keys()))
    if remove_ambiguous_lengths:
        valid = _np.logical_and(first_frame_ > 0, last_frame_ < info[0]["Frames"])
        linked_locs = linked_locs[valid]
    return linked_locs


def localization_precision(
    photons: _np.ndarray, 
    s: _np.ndarray, 
    bg: _np.ndarray, 
    em: bool
) -> _np.ndarray:
    """Calculates the theoretical localization precision according to
    Mortensen et al., Nat Meth, 2010 for a 2D unweighted Gaussian fit.

    Parameters
    ----------
    photons : np.ndarray
        Number of photons collected for the localization.
    s : np.ndarray
        Size of the single-emitter image for each localization.
    bg : np.ndarray
        Background signal for each localization (per pixel).
    em : bool
        Whether EMCCD was used for the localization.

    Returns
    -------
    np.ndarray
        Cramer-Rao lower bound for localization precision for each 
        localization.
    """

    s2 = s**2
    sa2 = s2 + 1 / 12
    v = sa2 * (16 / 9 + (8 * _np.pi * sa2 * bg) / photons) / photons
    if em:
        v *= 2
    with _np.errstate(invalid="ignore"):
        return _np.sqrt(v)
    

def n_segments(info: list[dict], segmentation: int) -> int:
    """Calculates the number of segments for the given segmentation
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
        segmentation value."""
    
    n_frames = info[0]["Frames"]
    n_segments = int(_np.round(n_frames / segmentation))
    return n_segments


def segment(
    locs: _np.ndarray, 
    info: list[dict], 
    segmentation: int, 
    kwargs: dict = {}, 
    callback: Callable[[int], None] = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Splits localizations into temporal segments (number of segments
    is defined by the segmentation parameter) and renders each segment
    into a 2D image.
    
    Parameters
    ----------
    locs : np.ndarray
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
    bounds : np.ndarray
        Array of bounds for each segment, where each bound is the
        starting frame of the segment.
    segments : np.ndarray
        3D array of segments, where each segment is a 2D image of the
        localizations in that segment.
    """

    Y = info[0]["Height"]
    X = info[0]["Width"]
    n_frames = info[0]["Frames"]
    n_seg = n_segments(info, segmentation)
    bounds = _np.linspace(0, n_frames - 1, n_seg + 1, dtype=_np.uint32)
    segments = _np.zeros((n_seg, Y, X))
    if callback is None:
        it = _trange(n_seg, desc="Generating segments", unit="segments")
    else:
        callback(0)
        it = range(n_seg)
    for i in it:
        segment_locs = locs[
            (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
        ]
        _, segments[i] = _render.render(segment_locs, info, **kwargs)
        if callback is not None:
            callback(i + 1)
    return bounds, segments

def undrift(
    locs: _np.recarray,
    info: list[dict],
    segmentation: int,
    display: bool = True,
    segmentation_callback: Callable[[int], None] = None,
    rcc_callback: Callable[[int], None] = None,
) -> tuple[_np.recarray, _np.recarray]:
    """Undrift by RCC. See Wang, Schnitzbauer, et al. Optics Express,
    2014.
    
    Parameters
    ----------
    locs : np.recarray
        Localization list to undrift.
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
    drift : np.recarray
        Estimated drift as a record array with fields 'x' and 'y'.
    locs : np.recarray
        Undrifted localization list with the drift applied to the 'x'
        and 'y' coordinates.
    """

    bounds, segments = segment(
        locs,
        info,
        segmentation,
        {"blur_method": "gaussian", "min_blur_width": 1},
        segmentation_callback,
    )
    shift_y, shift_x = _imageprocess.rcc(segments, 32, rcc_callback)
    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = _interpolate.InterpolatedUnivariateSpline(t, shift_x, k=3)
    drift_y_pol = _interpolate.InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = _np.arange(info[0]["Frames"])
    drift = (drift_x_pol(t_inter), drift_y_pol(t_inter))
    drift = _np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])
    if display:
        fig1 = _plt.figure(figsize=(17, 6))
        _plt.suptitle("Estimated drift")
        _plt.subplot(1, 2, 1)
        _plt.plot(drift.x, label="x interpolated")
        _plt.plot(drift.y, label="y interpolated")
        t = (bounds[1:] + bounds[:-1]) / 2
        _plt.plot(
            t,
            shift_x,
            "o",
            color=list(_plt.rcParams["axes.prop_cycle"])[0]["color"],
            label="x",
        )
        _plt.plot(
            t,
            shift_y,
            "o",
            color=list(_plt.rcParams["axes.prop_cycle"])[1]["color"],
            label="y",
        )
        _plt.legend(loc="best")
        _plt.xlabel("Frame")
        _plt.ylabel("Drift (pixel)")
        _plt.subplot(1, 2, 2)
        _plt.plot(
            drift.x,
            drift.y,
            color=list(_plt.rcParams["axes.prop_cycle"])[2]["color"],
        )
        _plt.plot(
            shift_x,
            shift_y,
            "o",
            color=list(_plt.rcParams["axes.prop_cycle"])[2]["color"],
        )
        _plt.axis("equal")
        _plt.xlabel("x")
        _plt.ylabel("y")
        fig1.show()
    locs.x -= drift.x[locs.frame]
    locs.y -= drift.y[locs.frame]
    return drift, locs


def undrift_from_picked(
    picked_locs: list[_np.recarray], 
    info: list[dict]
) -> _np.recarray:
    """Finds drift from picked localizations. Note that unlike other 
    undrifting functions, this function does not return undrifted 
    localizations but only drift.
    
    Parameters
    ----------
    picked_locs : list of np.recarrays
        List of picked localizations, where each element is a
        recarray of localizations for a single pick.
    info : list of dicts
        Metadata of the localization list, where each element
        corresponds to the metadata of the localizations in
        `picked_locs`.

    Returns
    -------
    drift : np.recarray
        Estimated drift as a record array with fields 'x', 'y', and
        optionally 'z' if the z coordinate exists in the picked
        localizations.
    """

    drift_x = _undrift_from_picked_coordinate(picked_locs, info, "x") 
    drift_y = _undrift_from_picked_coordinate(picked_locs, info, "y") 

    # A rec array to store the applied drift
    drift = (drift_x, drift_y)
    drift = _np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])

    # If z coordinate exists, also apply drift there
    if all([hasattr(_, "z") for _ in picked_locs]):
        drift_z = _undrift_from_picked_coordinate(picked_locs, info, "z")
        drift = _lib.append_to_rec(drift, drift_z, "z")
    return drift


def _undrift_from_picked_coordinate(
    picked_locs: list[_np.recarray], 
    info: list[dict], 
    coordinate: Literal["x", "y", "z"],
):
    """Calculates drift in a given coordinate from picked localizations.
    Uses the center of mass of each pick to find the drift
    in the specified coordinate across all frames. The drift is
    calculated as the average of the localizations' coordinates
    minus the mean of the coordinates for each pick. 

    Parameters
    ----------
    picked_locs : list of np.recarrays
        List of np.recarrays with locs for each pick.
    info : list of dicts
        Localizations' metadeta.
    coordinate : {"x", "y", "z"}
        Spatial coordinate where drift is to be found.

    Returns
    -------
    drift_mean : np.array
        Average drift across picks for all frames
    """

    n_picks = len(picked_locs)
    n_frames = info[0]["Frames"]

    # Drift per pick per frame
    drift = _np.empty((n_picks, n_frames))
    drift.fill(_np.nan)

    # Remove center of mass offset
    for i, locs in enumerate(picked_locs):
        coordinates = getattr(locs, coordinate)
        drift[i, locs.frame] = coordinates - _np.mean(coordinates)

    # Mean drift over picks
    drift_mean = _np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = _np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    nan_mask = _np.isnan(drift)
    drift = _np.ma.MaskedArray(drift, mask=nan_mask)
    drift_mean = _np.ma.average(drift, axis=0, weights=1/msd)
    drift_mean = drift_mean.filled(_np.nan)

    # Linear interpolation for frames without localizations
    def nan_helper(y):
        return _np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = _np.interp(
        nonzero(nans), nonzero(~nans), drift_mean[~nans]
    )
    return drift_mean


def align(
    locs: list[_np.recarray], 
    infos: list[dict], 
    display: bool = False,
) -> _np.recarray:
    """Aligns localizations from multiple channels (one per each element
    in `locs`) by calculating the shifts between the rendered images 
    using RCC.
    
    Parameters
    ----------
    locs : list of np.recarrays
        List of localization arrays, where each element is a
        recarray of localizations for a single image.
    infos : list of dicts
        List of metadata dictionaries corresponding to each
        localization array in `locs`.
    display : bool, optional
        Not used.
    
    Returns
    -------
    locs : list of np.recarrays
        Aligned localizations with the shifts applied to the 'x' and
        'y' coordinates.
    """
    
    images = []
    for i, (locs_, info_) in enumerate(zip(locs, infos)):
        _, image = _render.render(locs_, info_, blur_method="smooth")
        images.append(image)
    shift_y, shift_x = _imageprocess.rcc(images)
    print("Image x shifts: {}".format(shift_x))
    print("Image y shifts: {}".format(shift_y))
    for i, (locs_, dx, dy) in enumerate(zip(locs, shift_x, shift_y)):
        locs_.y -= dy
        locs_.x -= dx
    return locs


def groupprops(
    locs: _np.recarray, 
    callback: Callable[[int], None] | None = None,
) -> _np.recarray:
    """Calculates group statistics for localizations, such as mean and
    standard deviation.

    Parameters
    ----------
    locs : np.recarray
        Localization list with a 'group' field that defines the groups.
    callback : Callable[[int], None], optional
        Callback function to report progress. It should accept an
        integer argument representing the current group index.
        Default is None, which means no callback is used.
    """

    try:
        locs = locs[locs.dark != -1]
    except AttributeError:
        pass
    group_ids = _np.unique(locs.group)
    n = len(group_ids)
    n_cols = len(locs.dtype)
    names = ["group", "n_events"] + list(
        _itertools.chain(*[(_ + "_mean", _ + "_std") for _ in locs.dtype.names])
    )
    formats = ["i4", "i4"] + 2 * n_cols * ["f4"]
    groups = _np.recarray(n, formats=formats, names=names)
    if callback is not None:
        callback(0)
        it = enumerate(group_ids)
    else:
        it = enumerate(_tqdm(
            group_ids, desc="Calculating group statistics", unit="Groups"
        ))
    for i, group_id in it:
        group_locs = locs[locs.group == group_id]
        groups["group"][i] = group_id
        groups["n_events"][i] = len(group_locs)
        for name in locs.dtype.names:
            groups[name + "_mean"][i] = _np.mean(group_locs[name])
            groups[name + "_std"][i] = _np.std(group_locs[name])
        if callback is not None:
            callback(i + 1)
    return groups


# def calculate_fret(acc_locs, don_locs):
#     """
#     Calculate the FRET efficiency in picked regions, this is for one trace
#     """
#     fret_dict = {}
#     if len(acc_locs) == 0:
#         max_frames = _np.max(don_locs["frame"])
#     elif len(don_locs) == 0:
#         max_frames = _np.max(acc_locs["frame"])
#     else:
#         max_frames = _np.max([_np.max(acc_locs["frame"]), _np.max(don_locs["frame"])])

#     # Initialize a vector filled with zeros for the duration of the movie
#     xvec = _np.arange(max_frames + 1)
#     yvec = xvec[:] * 0
#     acc_trace = yvec.copy()
#     don_trace = yvec.copy()
#     # Fill vector with the photon numbers of events that happend
#     acc_trace[acc_locs["frame"]] = acc_locs["photons"] - acc_locs["bg"]
#     don_trace[don_locs["frame"]] = don_locs["photons"] - don_locs["bg"]

#     # Calculate the FRET efficiency
#     fret_trace = acc_trace / (acc_trace + don_trace)
#     # Only select FRET values between 0 and 1
#     selector = _np.logical_and(fret_trace > 0, fret_trace < 1)

#     # Select the final fret events based on the 0 to 1 range
#     fret_events = fret_trace[selector]
#     fret_timepoints = _np.arange(len(fret_trace))[selector]

#     f_locs = []
#     if len(fret_timepoints) > 0:
#         # Calculate FRET locs: Select the locs when FRET happens
#         sel_locs = []
#         for element in fret_timepoints:
#             sel_locs.append(don_locs[don_locs["frame"] == element])

#         f_locs = stack_arrays(sel_locs, asrecarray=True, usemask=False)
#         f_locs = _lib.append_to_rec(f_locs, _np.array(fret_events), "fret")

#     fret_dict["fret_events"] = _np.array(fret_events)
#     fret_dict["fret_timepoints"] = fret_timepoints
#     fret_dict["acc_trace"] = acc_trace
#     fret_dict["don_trace"] = don_trace
#     fret_dict["frames"] = xvec
#     fret_dict["maxframes"] = max_frames

#     return fret_dict, f_locs

def nn_analysis(
    x1: _np.ndarray, x2: _np.ndarray, 
    y1: _np.ndarray, y2: _np.ndarray, 
    z1: _np.ndarray, z2: _np.ndarray,
    nn_count: int, 
    same_channel: bool = True, 
) -> _np.ndarray:
    """Finds the nearest neighbors between two sets of localizations.
    
    Parameters
    ----------
    x1, y1, z1 : np.ndarray
        Coordinates of the first set of localizations.
    x2, y2, z2 : np.ndarray
        Coordinates of the second set of localizations.
    nn_count : int
        Number of nearest neighbors to find for each localization in
        the second set.
    same_channel : bool, optional
        If True, the first set of localizations is considered to be
        from the same channel as the second set, and the nearest
        neighbor with zero distance is ignored. If False, all nearest
        neighbors are considered, including the zero distance one.
        Default is True. 
    
    Returns
    -------
    nn : np.ndarray
        Array of nearest neighbors, where each row corresponds to a
        localization in the second set and contains the indices of its
        nearest neighbors in the first set. The number of columns is
        `nn_count` if `same_channel` is True, or `nn_count + 1` if
        `same_channel` is False. Each column contains the index of a
        nearest neighbor in the first set.
    """

    # coordinates are in nm
    if z1 is not None: # 3D
        input1 = _np.stack((x1, y1, z1)).T
        input2 = _np.stack((x2, y2, z2)).T
    else: # 2D
        input1 = _np.stack((x1, y1)).T
        input2 = _np.stack((x2, y2)).T
    if same_channel:
        model = NN(n_neighbors=nn_count+1)
    else:
        model = NN(n_neighbors=nn_count)
    model.fit(input1)
    nn, _ = model.kneighbors(input2)
    if same_channel:
        nn = nn[:, 1:] # ignore the zero distance
    return nn

def mask_locs(
    locs: _np.recarray, 
    mask: _np.ndarray, 
    width: float, 
    height: float,
) -> tuple[_np.recarray, _np.recarray]:
    """Masks localizations given a binary mask.
    
    Parameters
    ----------
    locs : np.recarray
        Localizations to be masked.
    mask : np.ndarray
        Binary mask where True indicates the area to keep.
    width : float
        Maximum x coordinate of the localizations.
    height : float
        Maximum y coordinate of the localizations.
    
    Returns
    -------
    locs_in : np.recarray
        Localizations inside the mask.
    locs_out : np.recarray
        Localizations outside the mask.
    """

    x_ind = (_np.floor(locs["x"] / width * mask.shape[0])).astype(int)
    y_ind = (_np.floor(locs["y"] / height * mask.shape[1])).astype(int)

    index = mask[y_ind, x_ind].astype(bool)
    locs_in = locs[index]
    locs_in.sort(kind="mergesort", order="frame")
    locs_out = locs[~index]
    locs_out.sort(kind="mergesort", order="frame")

    return locs_in, locs_out