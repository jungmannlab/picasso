"""
    gui/postprocess
    ~~~~~~~~~~~~~~~~~~~~

    Data analysis of localization lists

    :author: Joerg Schnitzbauer, 2015
"""

import numpy as _np
import numba as _numba
from sklearn.cluster import DBSCAN as _DBSCAN
import os.path as _ospath
import sys as _sys
from tqdm import tqdm as _tqdm
from scipy import interpolate as _interpolate
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import multiprocessing as _multiprocessing
import matplotlib.pyplot as _plt
from scipy.optimize import minimize as _minimize


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)
_parent_directory = _ospath.dirname(_this_directory)
_sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import lib as _lib
from picasso import render as _render
from picasso import imageprocess as _imageprocess
from picasso.localize import LOCS_DTYPE as _LOCS_DTYPE


def _get_index_blocks(locs, info, size):
    # Sort locs by indices
    x_index = _np.uint32(locs.x / size)
    y_index = _np.uint32(locs.y / size)
    sort_indices = _np.lexsort([x_index, y_index])
    locs = locs[sort_indices]
    x_index = x_index[sort_indices]
    y_index = y_index[sort_indices]
    # Allocate block info arrays
    n_blocks_x = int(_np.ceil(info[0]['Width'] / size))
    n_blocks_y = int(_np.ceil(info[0]['Height'] / size))
    block_starts = _np.zeros((n_blocks_y, n_blocks_x), dtype=_np.uint32)
    block_ends = _np.zeros((n_blocks_y, n_blocks_x), dtype=_np.uint32)
    # Fill in block starts and ends
    _fill_index_blocks(block_starts, block_ends, x_index, y_index)
    return locs, x_index, y_index, block_starts, block_ends


@_numba.jit(nopython=True)
def _fill_index_blocks(block_starts, block_ends, x_index, y_index):
    Y, X = block_starts.shape
    N = len(x_index)
    k = 0
    for i in range(Y):
        for j in range(X):
            block_starts[i, j] = k
            while k < N and y_index[k] == i and x_index[k] == j:
                k += 1
            block_ends[i, j] = k


@_numba.jit(nopython=True, nogil=True)
def _distance_histogram(locs, bin_size, r_max, x_index, y_index, block_starts, block_ends, start, chunk):
    x = locs.x
    y = locs.y
    dh_len = _np.uint32(r_max / bin_size)
    dh = _np.zeros(dh_len, dtype=_np.uint32)
    r_max_2 = r_max**2
    K, L = block_starts.shape
    end = min(start+chunk, len(locs))
    for i in range(start, end):
        xi = x[i]
        yi = y[i]
        ki = y_index[i]
        li = x_index[i]
        for k in range(ki, ki+2):
            if k < K:
                for l in range(li, li+2):
                    if l < L:
                        for j in range(block_starts[k, l], block_ends[k, l]):
                            if j > i:
                                dx2 = (xi - x[j])**2
                                if dx2 < r_max_2:
                                    dy2 = (yi - y[j])**2
                                    if dy2 < r_max_2:
                                        d = _np.sqrt(dx2 + dy2)
                                        if d < r_max:
                                            bin = _np.uint32(d / bin_size)
                                            dh[bin] += 1
    return dh


def distance_histogram(locs, info, bin_size, r_max):
    locs, x_index, y_index, block_starts, block_ends = _get_index_blocks(locs, info, r_max)
    N = len(locs)
    n_threads = _multiprocessing.cpu_count()
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [(locs, bin_size, r_max, x_index, y_index, block_starts, block_ends, start, chunk) for start in starts]
    with _ThreadPoolExecutor() as executor:
        futures = [executor.submit(_distance_histogram, *_) for _ in args]
    results = [future.result() for future in futures]
    return _np.sum(results, axis=0)


def pair_correlation(locs, info, bin_size, r_max):
    dh = distance_histogram(locs, info, bin_size, r_max)
    bins_lower = _np.arange(0, r_max, bin_size)
    area = _np.pi * bin_size * (2 * bins_lower + bin_size)
    return bins_lower, dh / area


def dbscan(locs, radius, min_density):
    print('Identifying clusters...')
    locs = _lib.ensure_finite(locs)
    locs = locs[_np.isfinite(locs.x) & _np.isfinite(locs.y)]
    X = _np.vstack((locs.x, locs.y)).T
    db = _DBSCAN(eps=radius, min_samples=min_density).fit(X)
    group = _np.int32(db.labels_)       # int32 for Origin compatiblity
    locs = _lib.append_to_rec(locs, group, 'group')
    locs = locs[locs.group != -1]
    print('Generating cluster information...')
    groups = _np.unique(locs.group)
    n_groups = len(groups)
    mean_frame = _np.zeros(n_groups)
    std_frame = _np.zeros(n_groups)
    com_x = _np.zeros(n_groups)
    com_y = _np.zeros(n_groups)
    std_x = _np.zeros(n_groups)
    std_y = _np.zeros(n_groups)
    n = _np.zeros(n_groups, dtype=_np.int32)
    for i, group in enumerate(groups):
        group_locs = locs[locs.group == i]
        mean_frame[i] = _np.mean(group_locs.frame)
        com_x[i] = _np.mean(group_locs.x)
        com_y[i] = _np.mean(group_locs.y)
        std_frame[i] = _np.std(group_locs.frame)
        std_x[i] = _np.std(group_locs.x)
        std_y[i] = _np.std(group_locs.y)
        n[i] = len(group_locs)
    clusters = _np.rec.array((groups, mean_frame, com_x, com_y, std_frame, std_x, std_y, n),
                             dtype=[('groups', groups.dtype), ('mean_frame', 'f4'), ('com_x', 'f4'), ('com_y', 'f4'),
                             ('std_frame', 'f4'), ('std_x', 'f4'), ('std_y', 'f4'), ('n', 'i4')])
    return clusters, locs


@_numba.jit(nopython=True, nogil=True)
def _local_density(locs, radius, x_index, y_index, block_starts, block_ends, start, chunk):
    x = locs.x
    y = locs.y
    N = len(x)
    r2 = radius**2
    end = min(start+chunk, N)
    density = _np.zeros(N, dtype=_np.uint32)
    for i in range(start, end):
        yi = y[i]
        xi = x[i]
        ki = y_index[i]
        li = x_index[i]
        di = 0
        for k in range(ki-1, ki+2):
            for l in range(li-1, li+2):
                for j in range(block_starts[k, l], block_ends[k, l]):
                    dx2 = (xi - x[j])**2
                    if dx2 < r2:
                        dy2 = (yi - y[j])**2
                        if dy2 < r2:
                            d2 = dx2 + dy2
                            if d2 < r2:
                                di += 1
        density[i] = di
    return density


def compute_local_density(locs, info, radius):
    locs, x_index, y_index, block_starts, block_ends = _get_index_blocks(locs, info, radius)
    N = len(locs)
    n_threads = _multiprocessing.cpu_count()
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [(locs, radius, x_index, y_index, block_starts, block_ends, start, chunk) for start in starts]
    with _ThreadPoolExecutor() as executor:
        futures = [executor.submit(_local_density, *_) for _ in args]
    density = _np.sum([future.result() for future in futures], axis=0)
    locs = _lib.remove_from_rec(locs, 'density')
    return _lib.append_to_rec(locs, density, 'density')


def compute_dark_times(locs):
    last_frame = locs.frame + locs.len - 1
    dark = _compute_dark_times(locs, last_frame)
    locs = _lib.append_to_rec(locs, _np.int32(dark), 'dark')        # int32 for Origin compatiblity
    return locs


@_numba.jit(nopython=True, cache=True)
def _compute_dark_times(locs, last_frame):
    N = len(locs)
    max_frame = locs.frame.max()
    dark = max_frame * _np.ones(len(locs), dtype=_np.int32)
    for i in range(N):
        for j in range(N):
            if (locs.group[i] == locs.group[j]) and (i != j):
                dark_ij = locs.frame[i] - last_frame[j]
                if (dark_ij > 0) and (dark_ij < dark[i]):
                    dark[i] = dark_ij
    for i in range(N):
        if dark[i] == max_frame:
            dark[i] = -1
    return dark


def link(locs, info, min_prob=0.05, max_dark_time=1, combine_mode='average'):
    locs = _lib.ensure_finite(locs)
    locs = locs[locs.lpx > 0.0]
    locs = locs[locs.lpy > 0.0]
    locs.sort(kind='mergesort', order='frame')
    if hasattr(locs, 'group'):
        group = locs.group
    else:
        group = _np.zeros(len(locs), dtype=_np.int32)
    link_group = get_link_groups(locs, min_prob, max_dark_time, group)
    if combine_mode == 'average':
        linked_locs = link_loc_groups(locs, link_group)
    elif combine_mode == 'refit':
        pass    # TODO
    return linked_locs


@_numba.jit(nopython=True)
def get_link_groups(locs, min_prob, max_dark_time, group):
    ''' Assumes that locs are sorted by frame '''
    frame = locs.frame
    x = locs.x
    y = locs.y
    lpx = locs.lpx
    lpy = locs.lpy
    N = len(x)
    link_group = -_np.ones(N, dtype=_np.int32)
    current_link_group = -1
    for i in range(N):
        if link_group[i] == -1:  # loc has no group yet
            current_link_group += 1
            link_group[i] = current_link_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, link_group, N, frame, x, y, lpx, lpy, min_prob, max_dark_time, group)
            while next_loc_index_in_group != -1:
                link_group[next_loc_index_in_group] = current_link_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, link_group, N, frame, x, y, lpx, lpy, min_prob, max_dark_time, group)
    return link_group


@_numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(current_index, link_group, N, frame, x, y, lpx, lpy, min_prob, max_dark_time, group):
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    current_group = group[current_index]
    min_frame = current_frame + 1
    for min_index in range(current_index + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = current_frame + max_dark_time + 1
    for max_index in range(min_index + 1, N):
        if frame[max_index] > max_frame:
            break
    current_lpx = lpx[current_index]
    current_lpy = lpy[current_index]
    for j in range(min_index, max_index):
        if group[j] == current_group:
            if link_group[j] == -1:
                dx2 = (current_x - x[j])**2
                dy2 = (current_y - y[j])**2
                prob_of_same = _np.exp(-(dx2/(2*current_lpx) + dy2/(2*current_lpy)) - (dx2/(2*lpx[j]) + dy2/(2*lpy[j])))
                if prob_of_same > min_prob:
                    return j
    return -1


def link_loc_groups(locs, link_group):
    is_grouped = hasattr(locs, 'group')
    if is_grouped:
        group = locs.group
    else:
        group = _np.zeros(len(locs), dtype=_np.int32)
    linked_locs_data = _link_loc_groups(locs, link_group, group)
    dtype = _LOCS_DTYPE[:-1] + [('iterations', 'f4'), ('len', 'i4'), ('n', 'u4'), ('photon_rate', 'f4')]
    if is_grouped:
        dtype.append(('group', 'i4'))
        return _np.rec.array(linked_locs_data, dtype=dtype)
    return _np.rec.array(linked_locs_data[:-1], dtype=dtype)


@_numba.jit(nopython=True)
def _link_loc_groups(locs, link_group, group):
    N_linked = link_group.max() + 1
    frame_ = locs.frame.max() * _np.ones(N_linked, dtype=_np.uint32)
    x_ = _np.zeros(N_linked, dtype=_np.float32)
    y_ = _np.zeros(N_linked, dtype=_np.float32)
    photons_ = _np.zeros(N_linked, dtype=_np.float32)
    sx_ = _np.zeros(N_linked, dtype=_np.float32)
    sy_ = _np.zeros(N_linked, dtype=_np.float32)
    bg_ = _np.zeros(N_linked, dtype=_np.float32)
    lpx_ = _np.zeros(N_linked, dtype=_np.float32)
    lpy_ = _np.zeros(N_linked, dtype=_np.float32)
    likelihood_ = _np.zeros(N_linked, dtype=_np.float32)
    iterations_ = _np.zeros(N_linked, dtype=_np.float32)
    len_ = _np.zeros(N_linked, dtype=_np.int32)
    n_ = _np.zeros(N_linked, dtype=_np.uint32)
    last_frame_ = _np.zeros(N_linked, dtype=_np.uint32)
    group_ = _np.zeros(N_linked, dtype=_np.int32)
    weights_x = 1/locs.lpx**2
    weights_y = 1/locs.lpy**2
    sum_weights_x_ = _np.zeros(N_linked, dtype=_np.float32)
    sum_weights_y_ = _np.zeros(N_linked, dtype=_np.float32)
    N = len(link_group)
    for i in range(N):
        i_ = link_group[i]
        n_[i_] += 1
        x_[i_] += weights_x[i] * locs.x[i]
        sum_weights_x_[i_] += weights_x[i]
        y_[i_] += weights_y[i] * locs.y[i]
        sum_weights_y_[i_] += weights_y[i]
        photons_[i_] += locs.photons[i]
        sx_[i_] += locs.sx[i]
        sy_[i_] += locs.sy[i]
        bg_[i_] += locs.bg[i]
        likelihood_[i_] += locs.likelihood[i]
        iterations_[i_] += locs.iterations[i]
        if locs.frame[i] < frame_[i_]:
            frame_[i_] = locs.frame[i]
        if locs.frame[i] > last_frame_[i_]:
            last_frame_[i_] = locs.frame[i]
        group_[i_] = group[i]
    x_ = x_ / sum_weights_x_
    y_ = y_ / sum_weights_y_
    sx_ = sx_ / n_
    sy_ = sy_ / n_
    bg_ = bg_ / n_
    lpx_ = _np.sqrt(1/sum_weights_x_)
    lpy_ = _np.sqrt(1/sum_weights_y_)
    likelihood_ = likelihood_ / n_
    iterations_ = iterations_ / n_
    len_ = last_frame_ - frame_ + 1
    photon_rate_ = photons_ / n_
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, likelihood_, iterations_, len_, n_, photon_rate_, group_


def undrift(locs, info, segmentation, mode='render', movie=None, display=True):
    locs = _lib.ensure_finite(locs)
    if mode in ['render', 'std']:
        drift = get_drift_rcc(locs, info, segmentation, mode, movie, display)
    elif mode == 'framepair':
        drift = get_drift_framepair(locs, info, display)
    locs.x -= drift[1][locs.frame]
    locs.y -= drift[0][locs.frame]
    drift = _np.rec.array(drift, dtype=[('x', 'f'), ('y', 'f')])
    return drift, locs


@_numba.jit(nopython=True, cache=True)
def get_frame_shift(locs, i, j, min_prob, k):
    N = len(locs)
    frame = locs.frame
    x = locs.x
    y = locs.y
    lpx = locs.lpx
    lpy = locs.lpy
    shift_x = 0.0
    shift_y = 0.0
    n = 0
    sum_weights_x = 0.0
    sum_weights_y = 0.0
    while frame[k] == i:
        xk = x[k]
        yk = y[k]
        lpxk = lpx[k]
        lpyk = lpy[k]
        for l in range(k+1, N):
            if frame[l] == j:
                dx = x[l] - xk
                dx2 = dx**2
                if dx2 < 1:
                    dy = y[l] - yk
                    dy2 = dy**2
                    if dy2 < 1:
                        lpxl = lpx[l]
                        lpyl = lpy[l]
                        prob_of_same = _np.exp(-(dx2/(2*lpxk) + dy2/(2*lpyk)) - (dx2/(2*lpxl) + dy2/(2*lpyl)))
                        if prob_of_same > min_prob:
                            weight_x = 1/(lpxk**2 + lpxl**2)
                            weight_y = 1/(lpyk**2 + lpyl**2)
                            sum_weights_x += weight_x
                            sum_weights_y += weight_y
                            shift_x += weight_x * dx
                            shift_y += weight_y * dy
                            n += 1
            elif frame[l] > j:
                break
        k += 1
    if n > 0:
        shift_x /= sum_weights_x
        shift_y /= sum_weights_y
    return n, shift_y, shift_x, k


def get_drift_framepair(locs, info, display=True):
    locs.sort(kind='mergesort', order='frame')
    n_frames = info[0]['Frames']
    shift_x = _np.zeros(n_frames)
    shift_y = _np.zeros(n_frames)
    n = _np.zeros(n_frames)
    with _tqdm(total=n_frames-1, desc='Computing frame shifts', unit='frames') as progress_bar:
        k = 0
        for f in range(1, n_frames):
            progress_bar.update()
            n[f], shift_y[f], shift_x[f], k = get_frame_shift(locs, f-1, f, 0.001, k)
    # _plt.hist(n)
    # _plt.show()
    # Sliding window average
    window_size = 10
    window = _np.ones(window_size) / window_size
    shift_x = _np.convolve(shift_x, window, 'same')
    shift_y = _np.convolve(shift_y, window, 'same')
    drift = (_np.cumsum(shift_y), _np.cumsum(shift_x))
    if display:
        _plt.figure(figsize=(17, 6))
        _plt.suptitle('Estimated drift')
        _plt.subplot(1, 2, 1)
        _plt.plot(drift[1], label='x')
        _plt.plot(drift[0], label='y')
        _plt.legend(loc='best')
        _plt.xlabel('Frame')
        _plt.ylabel('Drift (pixel)')
        _plt.subplot(1, 2, 2)
        _plt.plot(drift[1], drift[0], color=list(_plt.rcParams['axes.prop_cycle'])[2]['color'])
        _plt.axis('equal')
        _plt.xlabel('x')
        _plt.ylabel('y')
        _plt.show()
    return drift


def get_drift_rcc(locs, info, segmentation, mode='render', movie=None, display=True):
    roi = 32           # Maximum shift is 32 pixels
    Y = info[0]['Height']
    X = info[0]['Width']
    n_frames = info[0]['Frames']
    n_segments = int(_np.round(n_frames/segmentation))
    n_pairs = int(n_segments * (n_segments - 1) / 2)
    bounds = _np.linspace(0, n_frames-1, n_segments+1, dtype=_np.uint32)
    segments = _np.zeros((n_segments, Y, X))

    if mode == 'render':
        with _tqdm(total=n_segments, desc='Generating segments', unit='segments') as progress_bar:
            for i in range(n_segments):
                progress_bar.update()
                segment_locs = locs[(locs.frame >= bounds[i]) & (locs.frame < bounds[i+1])]
                _, segments[i] = _render.render(segment_locs, info, oversampling=1, blur_method='gaussian', min_blur_width=1)
    elif mode == 'std':
        with _tqdm(total=n_segments, desc='Generating segments', unit='segments') as progress_bar:
            for i in range(n_segments):
                progress_bar.update()
                segments[i] = _np.std(movie[bounds[i]:bounds[i+1]], axis=0)

    rij = _np.zeros((n_pairs, 2))
    A = _np.zeros((n_pairs, n_segments - 1))
    flag = 0

    with _tqdm(total=n_pairs, desc='Correlating segment pairs', unit='pairs') as progress_bar:
        for i in range(n_segments - 1):
            for j in range(i+1, n_segments):
                progress_bar.update()
                dyij, dxij = _imageprocess.get_image_shift(segments[i], segments[j], 5, roi)
                rij[flag, 0] = dyij
                rij[flag, 1] = dxij
                A[flag, i:j] = 1
                flag += 1

    Dj = _np.dot(_np.linalg.pinv(A), rij)
    drift_y = _np.insert(_np.cumsum(Dj[:, 0]), 0, 0)
    drift_x = _np.insert(_np.cumsum(Dj[:, 1]), 0, 0)

    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = _interpolate.InterpolatedUnivariateSpline(t, drift_x, k=3)
    drift_y_pol = _interpolate.InterpolatedUnivariateSpline(t, drift_y, k=3)
    t_inter = _np.arange(n_frames)
    drift = (drift_y_pol(t_inter), drift_x_pol(t_inter))
    if display:
        _plt.figure(figsize=(17, 6))
        _plt.suptitle('Estimated drift')
        _plt.subplot(1, 2, 1)
        _plt.plot(drift[1], label='x interpolated')
        _plt.plot(drift[0], label='y interpolated')
        t = (bounds[1:] + bounds[:-1]) / 2
        _plt.plot(t, drift_x, 'o', color=list(_plt.rcParams['axes.prop_cycle'])[0]['color'], label='x')
        _plt.plot(t, drift_y, 'o', color=list(_plt.rcParams['axes.prop_cycle'])[1]['color'], label='y')
        _plt.legend(loc='best')
        _plt.xlabel('Frame')
        _plt.ylabel('Drift (pixel)')
        _plt.subplot(1, 2, 2)
        _plt.plot(drift[1], drift[0], color=list(_plt.rcParams['axes.prop_cycle'])[2]['color'])
        _plt.plot(drift_x, drift_y, 'o', color=list(_plt.rcParams['axes.prop_cycle'])[2]['color'])
        _plt.axis('equal')
        _plt.xlabel('x')
        _plt.ylabel('y')
        _plt.show()
    return drift


def align(target_locs, target_info, locs, info, affine=False, display=False):
    target_locs = _lib.ensure_finite(target_locs)
    locs = _lib.ensure_finite(locs)
    N_target, target_image = _render.render(target_locs, target_info, oversampling=1,
                                            blur_method='gaussian', min_blur_width=1)
    N, image = _render.render(locs, info, oversampling=1, blur_method='gaussian', min_blur_width=1)
    target_pad = [int(_/4) for _ in target_image.shape]
    target_image_pad = _np.pad(target_image, target_pad, 'constant')
    image_pad = [int(_/4) for _ in image.shape]
    image = _np.pad(image, image_pad, 'constant')
    dy, dx = _imageprocess.get_image_shift(target_image_pad, image, 7, None, display=display)
    print('Image shift: dx={}, dy={}.'.format(dx, dy))
    locs.y -= dy
    locs.x -= dx
    if affine:
        print('Attempting affine transformation - this may take a while...')
        locsT = _np.rec.array((locs.x, locs.y, locs.lpx, locs.lpy), dtype=[('x', 'f4'), ('y', 'f4'), ('lpx', 'f4'), ('lpy', 'f4')])

        def apply_transforms(locs, T):
            Ox, Oy, W, H, theta, A, B, X, Y = T
            # Origin shift and scale
            x = W * (locs.x - Ox)
            y = H * (locs.y - Oy)
            # Rotate
            x_ = _np.cos(theta) * x + _np.sin(theta) * y
            y_ = -_np.sin(theta) * x + _np.cos(theta) * y
            x = x_.copy()
            y = y_.copy()
            # Shearing
            x_ = x + A * y
            y_ = B * x + y
            x = x_.copy()
            y = y_.copy()
            # Translate and origin backshift
            x += X + Ox
            y += Y + Oy
            return x, y

        def affine_xcorr_negmax(T, ref_image, locs):
            locsT.x, locsT.y = apply_transforms(locs, T)
            N_T, imageT = _render.render(locsT, info, oversampling=1, blur_method='gaussian', min_blur_width=1)
            xcorr = _imageprocess.xcorr(ref_image, imageT)
            return -xcorr.max()

        Ox = _np.mean(locs.x)
        Oy = _np.mean(locs.y)
        W = H = 1
        theta = A = B = X = Y = 0
        T0 = _np.array([Ox, Oy, W, H, theta, A, B, X, Y])
        init_steps = [0.05, 0.05, 0.05, 0.05, 0.02, 0.05, 0.05, 0.05, 0.05]
        result = _minimize(affine_xcorr_negmax, T0, args=(target_image, locs),
                           method='COBYLA', options={'rhobeg': init_steps})
        Ox, Oy, W, H, theta, A, B, X, Y = result.x
        print('''Origin shift (x,y): {}, {}
Scale (x,y): {}, {}
Rotation (deg): {}
Shear (x,y): {}, {}
Translation (x,y): {}, {}'''.format(Ox, Oy, W, H, 360*theta/(2*_np.pi), A, B, X, Y))
        locs.x, locs.y = apply_transforms(locs, result.x)
    return locs
