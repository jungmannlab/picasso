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
import time as _time
import matplotlib.pyplot as _plt
from scipy.optimize import minimize as _minimize


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)
_parent_directory = _ospath.dirname(_this_directory)
_sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import lib as _lib
from picasso import render as _render
from picasso import imageprocess as _imageprocess


@_numba.jit(nopython=True)
def distance_histogram(locs, bin_size, r_max):
    x = locs.x
    y = locs.y
    dh_len = _np.uint32(r_max / bin_size)
    dh = _np.zeros(dh_len, dtype=_np.uint32)
    r_max_2 = r_max**2
    N = len(x)
    for i in range(N):
        xi = x[i]
        yi = y[i]
        for j in range(i+1, N):
            dx2 = (xi - x[j])**2
            if dx2 < r_max_2:
                dy2 = (yi - y[j])**2
                if dy2 < r_max_2:
                    d = _np.sqrt(dx2 + dy2)
                    if d < r_max:
                        bin = _np.uint32(d / bin_size)
                        dh[bin] += 1
    return dh


def pair_correlation(locs, bin_size, r_max):
    bins_lower = _np.arange(0, r_max, bin_size)
    dh = distance_histogram(locs, bin_size, r_max)
    area = _np.pi * bin_size * (2 * bins_lower + bin_size)
    return bins_lower, dh / area


def dbscan(locs, radius, min_density):
    locs = locs[_np.isfinite(locs.x) & _np.isfinite(locs.y)]
    X = _np.vstack((locs.x, locs.y)).T
    db = _DBSCAN(eps=radius, min_samples=min_density).fit(X)
    group = _np.int32(db.labels_)       # int32 for Origin compatiblity
    locs = _lib.append_to_rec(locs, group, 'group')
    return locs[locs.group != -1]


def compute_local_density(locs, radius):
    N = len(locs)
    n_threads = int(0.75 * _multiprocessing.cpu_count())
    chunksize = int(N / n_threads)
    starts = range(0, N, chunksize)
    density = _np.zeros(N, dtype=_np.uint32)
    counters = [_np.zeros(1, dtype=_np.uint64) for _ in range(len(starts))]
    with _ThreadPoolExecutor(max_workers=n_threads) as executor:
        for start, counter in zip(starts, counters):
            executor.submit(_compute_local_density_partially, locs, radius, start, chunksize, density, counter)
        done = 0
        t0 = _time.time()
        while done < N:
            dt = _time.time() - t0
            if done > 0:
                secsleft = (N - done) * dt / done + 1
                minleft = int(secsleft / 60)
                if minleft > 0:
                    msg = '{} mins'.format(minleft)
                else:
                    msg = '{} secs'.format(int(secsleft))
                print('Evaluated {:,}/{:,} locs. Time left: '.format(done, N) + msg, end='\r')
            _time.sleep(0.1)
            done = int(_np.sum(counters))
        print()
    locs = _lib.remove_from_rec(locs, 'density')
    return _lib.append_to_rec(locs, density, 'density')


@_numba.jit(nopython=True, nogil=True)
def _compute_local_density_partially(locs, radius, start, chunksize, density, counter):
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
        counter[0] = i - start + 1
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
    locs = _lib.append_to_rec(locs, _np.int32(dark), 'dark')        # int32 for Origin compatiblity
    return locs[locs.dark != -1]


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


def link(locs, min_prob=0.05, max_dark_time=1, combine_mode='average'):
    locs = locs[_np.all(_np.array([_np.isfinite(locs[_]) for _ in locs.dtype.names]), axis=0)]
    locs.sort(kind='mergesort', order='frame')
    group = get_link_groups(locs, min_prob, max_dark_time)
    if combine_mode == 'average':
        linked_locs = link_loc_groups(locs, group)
        # TODO: set len to -1 if loc lasts until last frame or starts at first frame
    elif combine_mode == 'refit':
        pass    # TODO
    return linked_locs[linked_locs.len != -1]


@_numba.jit(nopython=True)
def get_link_groups(locs, min_prob, max_dark_time):
    ''' Assumes that locs are sorted by frame '''
    frame = locs.frame
    x = locs.x
    y = locs.y
    lpx = locs.lpx
    lpy = locs.lpy
    N = len(x)
    group = -_np.ones(N, dtype=_np.int32)
    current_group = -1
    for i in range(N):
        if group[i] == -1:  # loc has no group yet
            current_group += 1
            group[i] = current_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, lpx, lpy, min_prob, max_dark_time)
            while next_loc_index_in_group != -1:
                group[next_loc_index_in_group] = current_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, lpx, lpy, min_prob, max_dark_time)
    return group


@_numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(current_index, group, N, frame, x, y, lpx, lpy, min_prob, max_dark_time):
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
    current_lpx = lpx[current_index]
    current_lpy = lpy[current_index]
    for j in range(min_index, max_index):
        if group[j] == -1:
            dx2 = (current_x - x[j])**2
            dy2 = (current_y - y[j])**2
            prob_of_same = _np.exp(-(dx2/(2*current_lpx) + dy2/(2*current_lpy)) - (dx2/(2*lpx[j]) + dy2/(2*lpy[j])))
            if prob_of_same > min_prob:
                return j
    return -1


def link_loc_groups(locs, group):
    linked_locs_data = _link_loc_groups(locs, group)
    dtype = locs.dtype.descr + [('len', 'u4'), ('n', 'u4'), ('photon_rate', 'f4')]
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
    likelihood_ = _np.zeros(N_linked, dtype=_np.float32)
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
        likelihood_[i_] += locs.likelihood[i]
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
    likelihood_ = likelihood_ / n_
    len_ = last_frame_ - frame_ + 1
    photon_rate_ = photons_ / n_
    return frame_, x_, y_, photons_, sx_, sy_, bg_, lpx_, lpy_, likelihood_, len_, n_, photon_rate_


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


@_numba.jit(nopython=True, nogil=True)
def get_frame_shift(locs, i, j, min_prob):
    N = len(locs)
    frame = locs.frame
    x = locs.x
    y = locs.y
    lpx = locs.lpx
    lpy = locs.lpy
    shift_x = 0
    shift_y = 0
    n = 0
    for k in range(N):
        if frame[k] == i:
            xk = x[k]
            yk = y[k]
            lpxk = lpx[k]
            lpyk = lpy[k]
            for l in range(N):
                if frame[l] == j:
                    dx = x[l] - xk
                    dx2 = dx**2
                    if dx2 < 1:
                        dy = y[l] - yk
                        dy2 = dy**2
                        if dy2 < 1:
                            prob_of_same = _np.exp(-(dx2/(2*lpxk) + dy2/(2*lpyk)) - (dx2/(2*lpx[l]) + dy2/(2*lpy[l])))
                            if prob_of_same > min_prob:
                                shift_x += dx
                                shift_y += dy
                                n += 1
    if n > 0:
        shift_x /= n
        shift_y /= n
    return shift_y, shift_x


def get_drift_framepair(locs, info, display=True):
    # roi = 32           # Maximum shift is 32 pixels
    n_frames = info[0]['Frames']
    shift_x = _np.zeros(n_frames)
    shift_y = _np.zeros(n_frames)
    with _tqdm(total=n_frames-1, desc='Computing frame shifts', unit='frames') as progress_bar:
        for f in range(1, n_frames):
            progress_bar.update()
            shift_y[f], shift_x[f] = get_frame_shift(locs, f-1, f, 0.1)
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


def align(target_locs, target_info, locs, info, display=False):
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
    affine = True
    if affine:
        print('Attempting affine transformation...')

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
        locsT = _np.rec.array((locs.x, locs.y), dtype=[('x', 'f4'), ('y', 'f4')])

        def affine_xcorr_negmax(T, ref_image, locs):
            locsT.x, locsT.y = apply_transforms(locs, T)
            N_T, imageT = _render.render(locsT, info, oversampling=1, blur_method='gaussian', min_blur_width=1)
            xcorr = _imageprocess.xcorr(ref_image, imageT)
            return -xcorr.max()
        Ox = _np.mean(locs.x)
        Oy = _np.mean(locs.y)
        W = 1
        H = 1
        theta = 0
        A = 0
        B = 0
        X = 0
        Y = 0
        T0 = _np.array([Ox, Oy, W, H, theta, A, B, X, Y])
        init_steps = [0.1, 0.1, 0.05, 0.05, 0.02, 0.05, 0.05]
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
