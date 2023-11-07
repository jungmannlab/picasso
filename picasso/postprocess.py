"""
    gui/postprocess
    ~~~~~~~~~~~~~~~~~~~~

    Data analysis of localization lists

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2015-2018
    :copyright: Copyright (c) 2015-2018 Jungmann Lab, MPI Biochemistry
"""

import numpy as _np
import numba as _numba

from scipy import interpolate as _interpolate
from scipy.special import iv as _iv
from scipy.spatial import distance

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import multiprocessing as _multiprocessing
import matplotlib.pyplot as _plt
import itertools as _itertools
import lmfit as _lmfit
from collections import OrderedDict as _OrderedDict
from . import lib as _lib
from . import render as _render
from . import imageprocess as _imageprocess
from threading import Thread as _Thread
from tqdm import tqdm as _tqdm
from tqdm import trange as _trange
from numpy.lib.recfunctions import stack_arrays
from sklearn.neighbors import NearestNeighbors as NN


def get_index_blocks(locs, info, size, callback=None):
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


def index_blocks_shape(info, size):
    """Returns the shape of the index grid, given the movie and grid sizes"""
    n_blocks_x = int(_np.ceil(info[0]["Width"] / size))
    n_blocks_y = int(_np.ceil(info[0]["Height"] / size))
    return n_blocks_y, n_blocks_x


@_numba.jit(nopython=True, nogil=True)
def n_block_locs_at(x, y, size, K, L, block_starts, block_ends):
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


def get_block_locs_at(x, y, index_blocks):
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
    block_starts, block_ends, x_index, y_index
):
    Y, X = block_starts.shape
    N = len(x_index)
    k = 0
    for i in range(Y):
        for j in range(X):
            k = _fill_index_block(
                block_starts, block_ends, N, x_index, y_index, i, j, k
            )

@_numba.jit(nopython=True, nogil=True)
def _fill_index_block(block_starts, block_ends, N, x_index, y_index, i, j, k):
    block_starts[i, j] = k
    while k < N and y_index[k] == i and x_index[k] == j:
        k += 1
    block_ends[i, j] = k
    return k

@_numba.jit(nopython=True, nogil=True, cache=True)
def pick_similar(
        x, y_shift, y_base,
        min_n_locs, max_n_locs, min_rmsd, max_rmsd, 
        x_r, y_r1, y_r2,
        locs_xy, block_starts, block_ends, K, L,
        x_similar, y_similar, r, d2,
    ):
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
def _n_block_locs_at(x_range, y_range, K, L, block_starts, block_ends, cache=True):
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
        x_range, y_range, locs_xy, 
        block_starts, block_ends, K, L,
    ):
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
def _locs_at(x, y, locs_xy, r):
    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r ** 2
    is_picked = dx ** 2 + dy ** 2 < r2
    return locs_xy[:, is_picked]

@_numba.jit(nopython=True, nogil=True)
def _rmsd_at_com(locs_xy):
    com_x = _np.mean(locs_xy[0])
    com_y = _np.mean(locs_xy[1])
    return _np.sqrt(_np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2))

@_numba.jit(nopython=True, nogil=True)
def _distance_histogram(
    locs,
    bin_size,
    r_max,
    x_index,
    y_index,
    block_starts,
    block_ends,
    start,
    chunk,
):
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


def distance_histogram(locs, info, bin_size, r_max):
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
    return _np.sum(results, axis=0)


def nena(locs, info, callback=None):
    bin_centers, dnfl_ = next_frame_neighbor_distance_histogram(locs, callback)

    def func(d, a, s, ac, dc, sc):
        f = a * (d / s**2) * _np.exp(-0.5 * d**2 / s**2)
        fc = (
            ac
            * (d / sc**2)
            * _np.exp(-0.5 * (d**2 + dc**2) / sc**2)
            * _iv(0, d * dc / sc)
        )
        return f + fc

    pdf_model = _lmfit.Model(func)
    params = _lmfit.Parameters()
    area = _np.trapz(dnfl_, bin_centers)
    median_lp = _np.mean([_np.median(locs.lpx), _np.median(locs.lpy)])
    params.add("a", value=area / 2, min=0)
    params.add("s", value=median_lp, min=0)
    params.add("ac", value=area / 2, min=0)
    params.add("dc", value=2 * median_lp, min=0)
    params.add("sc", value=median_lp, min=0)
    result = pdf_model.fit(dnfl_, params, d=bin_centers)
    return result, result.best_values["s"]


def next_frame_neighbor_distance_histogram(locs, callback=None):
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


def _nfndh(frame, x, y, group, d_max, bin_size, callback=None):
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
def _fill_dnfl(N, frame, x, y, group, i, d_max, dnfl, bin_size):
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


def pair_correlation(locs, info, bin_size, r_max):
    dh = distance_histogram(locs, info, bin_size, r_max)
    # Start with r-> otherwise area will be 0
    bins_lower = _np.arange(bin_size, r_max + bin_size, bin_size)

    if bins_lower.shape[0] > dh.shape[0]:
        bins_lower = bins_lower[:-1]
    area = _np.pi * bin_size * (2 * bins_lower + bin_size)
    return bins_lower, dh / area

@_numba.jit(nopython=True, nogil=True)
def _local_density(
    locs, radius, x_index, y_index, block_starts, block_ends, start, chunk
):
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

def compute_local_density(locs, info, radius):
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


def compute_dark_times(locs, group=None):

    if "len" not in locs.dtype.names:
        raise AttributeError("Length not found. Please link localizations first.")
    dark = dark_times(locs, group)
    locs = _lib.append_to_rec(locs, _np.int32(dark), "dark")
    locs = locs[locs.dark != -1]
    return locs


def dark_times(locs, group=None):
    last_frame = locs.frame + locs.len - 1
    if group is None:
        if hasattr(locs, "group"):
            group = locs.group
        else:
            group = _np.zeros(len(locs))
    dark = _dark_times(locs, group, last_frame)
    return dark


@_numba.jit(nopython=True)
def _dark_times(locs, group, last_frame):
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
    locs,
    info,
    r_max=0.05,
    max_dark_time=1,
    combine_mode="average",
    remove_ambiguous_lengths=True,
):
    if len(locs) == 0:
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
            pass  # TODO
    return linked_locs


def weighted_variance(locs):
    n = len(locs)
    w = locs.photons
    x = locs.x
    y = locs.y
    xWbarx = _np.average(locs.x, weights=w)
    xWbary = _np.average(locs.y, weights=w)
    wbarx = _np.mean(locs.lpx)
    wbary = _np.mean(locs.lpy)
    variance_x = (
        n
        / ((n - 1) * sum(w) ** 2)
        * (
            sum((w * x - wbarx * xWbarx) ** 2)
            - 2 * xWbarx * sum((w - wbarx) * (w * x - wbarx * xWbarx))
            + xWbarx**2 * sum((w - wbarx) ** 2)
        )
    )
    variance_y = (
        n
        / ((n - 1) * sum(w) ** 2)
        * (
            sum((w * y - wbary * xWbary) ** 2)
            - 2 * xWbary * sum((w - wbary) * (w * y - wbary * xWbary))
            + xWbary**2 * sum((w - wbary) ** 2)
        )
    )
    return variance_x, variance_y


# Combine localizations: calculate the properties of the group
def cluster_combine(locs):
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


def cluster_combine_dist(locs):
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
def get_link_groups(locs, d_max, max_dark_time, group):
    """Assumes that locs are sorted by frame"""
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
    current_index, link_group, N, frame, x, y, d_max, max_dark_time, group
):
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
def _link_group_count(link_group, n_locs, n_groups):
    result = _np.zeros(n_groups, dtype=_np.uint32)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += 1
    return result


@_numba.jit(nopython=True)
def _link_group_sum(column, link_group, n_locs, n_groups):
    result = _np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += column[i]
    return result


@_numba.jit(nopython=True)
def _link_group_mean(column, link_group, n_locs, n_groups, n_locs_per_group):
    group_sum = _link_group_sum(column, link_group, n_locs, n_groups)
    result = _np.empty(
        n_groups, dtype=_np.float32
    )  # this ensures float32 after the division
    result[:] = group_sum / n_locs_per_group
    return result


@_numba.jit(nopython=True)
def _link_group_weighted_mean(
    column, weights, link_group, n_locs, n_groups, n_locs_per_group
):
    sum_weights = _link_group_sum(weights, link_group, n_locs, n_groups)
    return (
        _link_group_mean(column * weights, link_group, n_locs, n_groups, sum_weights),
        sum_weights,
    )


@_numba.jit(nopython=True)
def _link_group_min_max(column, link_group, n_locs, n_groups):
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
def _link_group_last(column, link_group, n_locs, n_groups):
    result = _np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] = column[i]
    return result


def link_loc_groups(locs, info, link_group, remove_ambiguous_lengths=True):
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


def localization_precision(photons, s, bg, em):
    """
    Calculates the theoretical localization precision according to
    Mortensen et al., Nat Meth, 2010 for a 2D unweighted Gaussian fit.
    """
    s2 = s**2
    sa2 = s2 + 1 / 12
    v = sa2 * (16 / 9 + (8 * _np.pi * sa2 * bg) / photons) / photons
    if em:
        v *= 2
    with _np.errstate(invalid="ignore"):
        return _np.sqrt(v)

def n_segments(info, segmentation):
    n_frames = info[0]["Frames"]
    return int(_np.round(n_frames / segmentation))

def segment(locs, info, segmentation, kwargs={}, callback=None):
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
    locs,
    info,
    segmentation,
    display=True,
    segmentation_callback=None,
    rcc_callback=None,
):
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


def align(locs, infos, display=False):
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


def groupprops(locs, callback=None):
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
    x1, x2, 
    y1, y2, 
    z1, z2,
    nn_count, 
    same_channel, 
):
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