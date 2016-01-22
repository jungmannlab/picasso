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
import lmfit as _lmfit
from scipy import interpolate as _interpolate
import matplotlib.pyplot as _plt
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import multiprocessing as _multiprocessing


_plt.style.use('ggplot')


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)
_parent_directory = _ospath.dirname(_this_directory)
_sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import io as _io
from picasso import lib as _lib


def dbscan(locs, radius, min_density):
    locs = locs[_np.isfinite(locs.x) & _np.isfinite(locs.y)]
    X = _np.vstack((locs.x, locs.y)).T
    db = _DBSCAN(eps=radius, min_samples=min_density).fit(X)
    group = _np.int32(db.labels_)       # int32 for Origin compatiblity
    return _lib.append_to_rec(locs, group, 'group')


def compute_local_density(locs, radius):
    N = len(locs)
    n_threads = 2 * _multiprocessing.cpu_count()
    chunksize = int(N / n_threads)
    starts = range(0, N, chunksize)
    density = _np.zeros(N, dtype=_np.uint32)
    with _ThreadPoolExecutor(max_workers=n_threads) as executor:
        for start in starts:
            executor.submit(_compute_local_density_partially, locs, radius, start, chunksize, density)
    locs = _lib.remove_from_rec(locs, 'density')
    return _lib.append_to_rec(locs, density, 'density')


@_numba.jit(nopython=True, nogil=True)
def _compute_local_density_partially(locs, radius, start, chunksize, density):
    r2 = radius**2
    N = len(locs)
    end = min(N, start + chunksize)
    for i in range(start, end):
        xi = locs.x[i]
        yi = locs.y[i]
        for j in range(N):
            dx2 = (xi - locs.x[j])**2
            if dx2 < r2:
                dy2 = (yi - locs.y[j])**2
                if dy2 < r2:
                    d = _np.sqrt(dx2 + dy2)
                    if d < radius:
                        density[i] += 1
    return density


@_numba.jit(nopython=True)
def _compute_local_density(locs, radius):
    N = len(locs)
    r2 = radius**2
    density = _np.zeros(N, dtype=_np.uint32)
    for i in range(N):
        if i % 1000 == 0:
            print(i, N)
        xi = locs.x[i]
        yi = locs.y[i]
        for j in range(N):
            if i != j:
                dx2 = (xi - locs.x[j])**2
                if dx2 < r2:
                    dy2 = (yi - locs.y[j])**2
                    if dy2 < r2:
                        d = _np.sqrt(dx2 + dy2)
                        if d < radius:
                            density[i] += 1
    return density


def compute_dark_times(locs):
    last_frame = locs.frame + locs.len - 1
    dark = _compute_dark_times(locs, last_frame)
    return _lib.append_to_rec(locs, _np.int32(dark), 'dark')        # int32 for Origin compatiblity


@_numba.jit(nopython=True)
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


def link(locs, radius, max_dark_time, combine_mode='average'):
    locs = locs[_np.all(_np.array([_np.isfinite(locs[_]) for _ in locs.dtype.names]), axis=0)]
    locs.sort(kind='mergesort', order='frame')
    group = get_link_groups(locs, radius, max_dark_time)
    if combine_mode == 'average':
        linked_locs = link_loc_groups(locs, group)
        # TODO: set len to -1 if loc lasts until last frame or starts at first frame
    elif combine_mode == 'refit':
        pass    # TODO
    return linked_locs


@_numba.jit(nopython=True)
def get_link_groups(locs, radius, max_dark_time):
    ''' Assumes that locs are sorted by frame '''
    frame = locs.frame
    x = locs.x
    y = locs.y
    N = len(x)
    group = -_np.ones(N, dtype=_np.int32)
    current_group = -1
    for i in range(N):
        if group[i] == -1:  # loc has no group yet
            current_group += 1
            group[i] = current_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, radius, max_dark_time)
            while next_loc_index_in_group != -1:
                group[next_loc_index_in_group] = current_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, radius, max_dark_time)
    return group


@_numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, radius, max_dark_time):
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    min_frame = current_frame + 1
    for min_index in range(current_index + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = current_frame + max_dark_time + 1
    for max_index in range(min_index + 1, N):
        if frame[max_index] > max_frame:
            break
    for j in range(min_index, max_index):
        if group[j] == -1:
            if (abs(current_x - x[j]) < radius) and (abs(current_y - y[j]) < radius):
                distance_to_current = _np.sqrt((current_x - x[j])**2 + (current_y - y[j])**2)
                if distance_to_current < radius:
                    return j
    return -1


def link_loc_groups(locs, group):
    linked_locs_data = _link_loc_groups(locs, group)
    dtype = locs.dtype.descr + [('len', 'u4'), ('n', 'u4')]
    return _np.rec.array(linked_locs_data, dtype=dtype)


@_numba.jit(nopython=True)
def _link_loc_groups(locs, group):
    N_linked = group.max() + 1
    frame_ = locs.frame.max() * _np.ones(N_linked, dtype=_np.uint32)
    x_ = _np.zeros(N_linked, dtype=_np.float32)
    y_ = _np.zeros(N_linked, dtype=_np.float32)
    photons_ = _np.zeros(N_linked, dtype=_np.float32)
    sx_ = _np.zeros(N_linked, dtype=_np.float32)
    sy_ = _np.zeros(N_linked, dtype=_np.float32)
    bg_ = _np.zeros(N_linked, dtype=_np.float32)
    lpx_ = _np.zeros(N_linked, dtype=_np.float32)
    lpy_ = _np.zeros(N_linked, dtype=_np.float32)
    len_ = _np.zeros(N_linked, dtype=_np.uint32)
    n_ = _np.zeros(N_linked, dtype=_np.uint32)
    last_frame_ = _np.zeros(N_linked, dtype=_np.uint32)
    weights_x = 1/locs.lpx**2
    weights_y = 1/locs.lpy**2
    sum_weights_x_ = _np.zeros(N_linked, dtype=_np.float32)
    sum_weights_y_ = _np.zeros(N_linked, dtype=_np.float32)
    N = len(group)
    for i in range(N):
        i_ = group[i]
        n_[i_] += 1
        x_[i_] += weights_x[i] * locs.x[i]
        sum_weights_x_[i_] += weights_x[i]
        y_[i_] += weights_y[i] * locs.y[i]
        sum_weights_y_[i_] += weights_y[i]
        photons_[i_] += locs.photons[i]
        sx_[i_] += locs.sx[i]
        sy_[i_] += locs.sy[i]
        bg_[i_] += locs.bg[i]
        if locs.frame[i] < frame_[i_]:
            frame_[i_] = locs.frame[i]
        if locs.frame[i] > last_frame_[i_]:
            last_frame_[i_] = locs.frame[i]
    x_ = x_ / sum_weights_x_
    y_ = y_ / sum_weights_y_
    sx_ = sx_ / n_
    sy_ = sy_ / n_
    bg_ = bg_ / n_
    lpx_ = _np.sqrt(1/sum_weights_x_)
    lpy_ = _np.sqrt(1/sum_weights_y_)
    len_ = last_frame_ - frame_ + 1
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, len_, n_


@_numba.jit(nopython=True)
def __link_loc_groups(locs, group):
    N_linked = group.max() + 1
    frame_ = locs.frame.max() * _np.ones(N_linked, dtype=_np.uint32)
    x_ = _np.zeros(N_linked, dtype=_np.float32)
    y_ = _np.zeros(N_linked, dtype=_np.float32)
    photons_ = _np.zeros(N_linked, dtype=_np.float32)
    sx_ = _np.zeros(N_linked, dtype=_np.float32)
    sy_ = _np.zeros(N_linked, dtype=_np.float32)
    bg_ = _np.zeros(N_linked, dtype=_np.float32)
    lpx_ = _np.zeros(N_linked, dtype=_np.float32)
    lpy_ = _np.zeros(N_linked, dtype=_np.float32)
    len_ = _np.zeros(N_linked, dtype=_np.uint32)
    n_ = _np.zeros(N_linked, dtype=_np.uint32)
    last_frame_ = _np.zeros(N_linked, dtype=_np.uint32)
    N = len(group)
    for i in range(N):
        i_ = group[i]
        n_[i_] += 1
        x_[i_] += locs.x[i]
        y_[i_] += locs.y[i]
        photons_[i_] += locs.photons[i]
        sx_[i_] += locs.sx[i]
        sy_[i_] += locs.sy[i]
        bg_[i_] += locs.bg[i]
        lpx_[i_] += locs.lpx[i]**2
        lpy_[i_] += locs.lpy[i]**2
        if locs.frame[i] < frame_[i_]:
            frame_[i_] = locs.frame[i]
        if locs.frame[i] > last_frame_[i_]:
            last_frame_[i_] = locs.frame[i]
    x_ = x_ / n_
    y_ = y_ / n_
    sx_ = sx_ / n_
    sy_ = sy_ / n_
    bg_ = bg_ / n_
    lpx_ = _np.sqrt(lpx_) / n_
    lpy_ = _np.sqrt(lpy_) / n_
    len_ = last_frame_ - frame_ + 1
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, len_, n_


def undrift(locs, info, segmentation, display=True):
    fit_roi = 5
    movie_file = info[0]['Raw File']
    movie, _ = _io.load_raw(movie_file)
    frames, Y, X = movie.shape
    n_segments = int(_np.round(frames/segmentation))
    n_pairs = int(n_segments * (n_segments - 1) / 2)
    bounds = _np.linspace(0, frames-1, n_segments+1, dtype=_np.uint32)
    segments = _np.zeros((n_segments, movie.shape[1], movie.shape[1]))
    with _tqdm(total=n_segments, desc='Generating segments', unit='segments') as progress_bar:
        for i in range(n_segments):
            progress_bar.update()
            segments[i] = _np.std(movie[bounds[i]:bounds[i+1]], axis=0)
    fit_X = int(fit_roi/2)
    y, x = _np.mgrid[-fit_X:fit_X+1, -fit_X:fit_X+1]
    Y_ = Y / 4
    X_ = X / 4
    rij = _np.zeros((n_pairs, 2))
    A = _np.zeros((n_pairs, n_segments - 1))
    flag = 0

    def _gaussian2d(a, xc, yc, s, b):
        A = a * _np.exp(-0.5 * ((x - xc)**2 + (y - yc)**2) / s**2) + b
        return A.flatten()
    gaussian2d = _lmfit.Model(_gaussian2d, name='2D Gaussian', independent_vars=[])

    def fit_gaussian(I):
        I_ = I[Y_:-Y_, X_:-X_]
        y_max, x_max = _np.unravel_index(I_.argmax(), I_.shape)
        y_max += Y_
        x_max += X_
        I_ = I[y_max-fit_X:y_max+fit_X+1, x_max-fit_X:x_max+fit_X+1]
        params = _lmfit.Parameters()
        params.add('a', value=I_.max(), vary=True, min=0)
        params.add('xc', value=0, vary=True)
        params.add('yc', value=0, vary=True)
        params.add('s', value=1, vary=True, min=0)
        params.add('b', value=I_.min(), vary=True, min=0)
        results = gaussian2d.fit(I_.flatten(), params)
        xc = results.best_values['xc']
        yc = results.best_values['yc']
        xc += x_max
        yc += y_max
        return yc, xc

    with _tqdm(total=n_pairs, desc='Correlating segment pairs', unit='pairs') as progress_bar:
        for i in range(n_segments - 1):
            autocorr = _lib.xcorr_fft(segments[i], segments[i])
            cyii, cxii = fit_gaussian(autocorr)
            for j in range(i+1, n_segments):
                progress_bar.update()
                xcorr = _lib.xcorr_fft(segments[i], segments[j])
                cyij, cxij = fit_gaussian(xcorr)
                rij[flag, 0] = cyii - cyij
                rij[flag, 1] = cxii - cxij
                A[flag, i:j] = 1
                flag += 1

    Dj = _np.dot(_np.linalg.pinv(A), rij)
    drift_y = _np.insert(_np.cumsum(Dj[:, 0]), 0, 0)
    drift_x = _np.insert(_np.cumsum(Dj[:, 1]), 0, 0)

    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = _interpolate.InterpolatedUnivariateSpline(t, drift_x, k=3)
    drift_y_pol = _interpolate.InterpolatedUnivariateSpline(t, drift_y, k=3)
    t_inter = _np.arange(frames)
    drift_x_inter = drift_x_pol(t_inter)
    drift_y_inter = drift_y_pol(t_inter)

    if display:
        _plt.figure(figsize=(17, 6))
        _plt.suptitle('Estimated drift')
        _plt.subplot(1, 2, 1)
        ax = _plt.plot(t_inter, drift_x_inter, label='x interpolated')
        color_x = ax[0].get_color()
        ax = _plt.plot(t_inter, drift_y_inter, label='y interpolated')
        color_y = ax[0].get_color()
        _plt.plot(t, drift_x, 'o', color=color_x, label='x measured')
        _plt.plot(t, drift_y, 'o', color=color_y, label='y measured')
        _plt.legend(loc='best')
        _plt.xlabel('Frame')
        _plt.ylabel('Drift (pixel)')
        _plt.subplot(1, 2, 2)
        ax = _plt.plot(drift_x_inter, drift_y_inter, color=list(_plt.rcParams['axes.prop_cycle'])[2]['color'])
        _plt.plot(drift_x, drift_y, 'o', color=list(_plt.rcParams['axes.prop_cycle'])[2]['color'])
        _plt.axis('equal')
        _plt.xlabel('x')
        _plt.ylabel('y')
        _plt.show()

    locs.x -= drift_x_inter[locs.frame]
    locs.y -= drift_y_inter[locs.frame]
    return locs
