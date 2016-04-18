"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in a frame sequence

    :author: Joerg Schnitzbauer, 2015
"""

import sys as _sys
import numpy as _np
import numba as _numba
import multiprocessing as _multiprocessing
import ctypes as _ctypes
import os.path as _ospath
import yaml as _yaml
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import threading as _threading
from itertools import chain as _chain


_C_FLOAT_POINTER = _ctypes.POINTER(_ctypes.c_float)
LOCS_DTYPE = [('frame', 'u4'), ('x', 'f4'), ('y', 'f4'),
              ('photons', 'f4'), ('sx', 'f4'), ('sy', 'f4'),
              ('bg', 'f4'), ('lpx', 'f4'), ('lpy', 'f4'), ('likelihood', 'f4'), ('iterations', 'i4')]


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)
_parent_directory = _ospath.dirname(_this_directory)
_sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import gaussmle as _gaussmle


with open(_ospath.join(_this_directory, 'config.yaml'), 'r') as config_file:
    CONFIG = _yaml.load(config_file)


@_numba.jit(nopython=True, nogil=True)
def local_maxima_map(frame, box):
    """ Finds pixels with maximum value within a region of interest """
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    box_half = int(box / 2)
    for i in range(box, Y - box):
        for j in range(box, X - box):
            local_frame = frame[i - box_half:i + box_half + 1, j - box_half:j + box_half + 1]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    return maxima_map


@_numba.jit(nopython=True, nogil=True)
def local_gradient_magnitude(frame, box):
    """ Returns the sum of the absolute gradient within a box around each pixel """
    gm = gradient_magnitude(frame)
    Y, X = gm.shape
    lgm = _np.zeros_like(gm)
    box_half = int(box / 2)
    for i in range(box_half, Y - box + box_half + 1):
        for j in range(box_half, X - box + box_half + 1):
            local_gradient = gm[i - box_half:i + box_half + 1, j - box_half:j + box_half + 1]
            lgm[i, j] = _np.sum(local_gradient)
    return lgm


@_numba.jit(nopython=True, nogil=True)
def gradient_magnitude(frame):
    Y, X = frame.shape
    gm = _np.zeros((Y, X), dtype=_np.float32)
    for i in range(1, Y-1):
        for j in range(1, X-1):
            dy = frame[i+1, j] - frame[i-1, j]
            dx = frame[i, j+1] - frame[i, j-1]
            gm[i, j] = 0.5 * _np.sqrt(dy**2 + dx**2)
    return gm


def identify_in_frame(frame, minimum_lgm, box, roi=None):
    if roi is not None:
        frame = frame[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
    s_map = local_gradient_magnitude(frame, box)
    lm_map = local_maxima_map(frame, box)
    s_map_thesholded = s_map > minimum_lgm
    combined_map = (lm_map * s_map_thesholded) > 0.5
    y, x = _np.where(combined_map)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x


def identify_by_frame_number(movie, minimum_lgm, box, frame_number, roi=None):
    frame = movie[frame_number]
    y, x = identify_in_frame(frame, minimum_lgm, box, roi)
    frame = frame_number * _np.ones(len(x))
    return _np.rec.array((frame, x, y), dtype=[('frame', 'i'), ('x', 'i'), ('y', 'i')])


def _identify_worker(movie, current, minimum_lgm, box, roi, lock):
    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(identify_by_frame_number(movie, minimum_lgm, box, index, roi))


def identifications_from_futures(futures):
    identifications_list_of_lists = [_.result() for _ in futures]
    identifications_list = _chain(*identifications_list_of_lists)
    return _np.hstack(identifications_list).view(_np.recarray)


def identify_async(movie, minimum_lgm, box, roi=None):
    n_workers = int(0.75 * _multiprocessing.cpu_count())
    current = [0]
    executor = _ThreadPoolExecutor(n_workers)
    lock = _threading.Lock()
    f = [executor.submit(_identify_worker, movie, current, minimum_lgm, box, roi, lock) for _ in range(n_workers)]
    executor.shutdown(wait=False)
    return current, f


def identify(movie, minimum_lgm, box, threaded=True):
    if threaded:
        N = len(movie)
        current, futures = identify_async(movie, minimum_lgm, box)
        while current[0] < N:
            pass
        return  # TODO
    else:
        identifications = [identify_by_frame_number(movie, minimum_lgm, box, i) for i in range(len(movie))]
    return _np.hstack(identifications).view(_np.recarray)


@_numba.jit(nopython=True)
def _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box):
    n_spots = len(ids_x)
    r = int(box/2)
    spots = _np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc-r:yc+r+1, xc-r:xc+r+1]
    return spots


@_numba.jit(nopython=True)
def _cut_spots_frame(frame, frame_number, ids_frame, ids_x, ids_y, r, start, N, spots):
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc-r:yc+r+1, xc-r:xc+r+1]
    return j


def _cut_spots(movie, identifications, box):
    ids_frame, ids_x, ids_y = (identifications[_] for _ in identifications.dtype.names)
    if isinstance(movie, _np.ndarray):
        return _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box)
    else:
        ''' Assumes that identifications are in order of frames! '''
        r = int(box/2)
        N = len(ids_frame)
        spots = _np.zeros((N, box, box), dtype=movie.dtype)
        start = 0
        for frame_number, frame in enumerate(movie):
            start = _cut_spots_frame(frame, frame_number, ids_frame, ids_x, ids_y, r, start, N, spots)
        return spots


def _to_photons(spots, camera_info):
    spots = _np.float32(spots)
    if camera_info['sensor'] == 'EMCCD':
        return (spots - 100) * camera_info['sensitivity'] / (camera_info['gain'] * camera_info['qe'])
    elif camera_info['sensor'] == 'sCMOS':
        return (spots - 100) * camera_info['sensitivity'] / camera_info['qe']
    elif camera_info['sensor'] == 'Simulation':
        return spots - 100
    else:
        raise TypeError('Unknown camera type')


def _get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def fit(movie, camera_info, identifications, box, eps=0.001):
    spots = _get_spots(movie, identifications, box, camera_info)
    theta, CRLBs, likelihoods, iterations = _gaussmle.gaussmle_sigmaxy(spots, eps)
    return locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box)


def fit_async(movie, camera_info, identifications, box, eps=0.001):
    spots = _get_spots(movie, identifications, box, camera_info)
    return _gaussmle.gaussmle_sigmaxy_async(spots, eps)


def locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box):
    box_offset = int(box/2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    lpy = _np.sqrt(CRLBs[:, 0])
    lpx = _np.sqrt(CRLBs[:, 1])
    return _np.rec.array((identifications.frame, x, y,
                          theta[:, 2], theta[:, 5], theta[:, 4],
                          theta[:, 3], lpx, lpy, likelihoods, iterations),
                         dtype=LOCS_DTYPE)


def localize(movie, info, parameters):
    identifications = identify(movie, parameters)
    return fit(movie, info, identifications, parameters['Box Size'])
