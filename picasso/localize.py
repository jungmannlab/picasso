"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in a frame sequence

    :author: Joerg Schnitzbauer, 2015
"""

import numpy as _np
import numba as _numba
import multiprocessing as _multiprocessing
import ctypes as _ctypes
import threading as _threading
import os.path as _ospath
from collections import namedtuple as _namedtuple
import yaml as _yaml
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import wait as _wait


_C_FLOAT_POINTER = _ctypes.POINTER(_ctypes.c_float)
LOCS_DTYPE = [('frame', 'u4'), ('x', 'f4'), ('y', 'f4'),
              ('photons', 'f4'), ('sx', 'f4'), ('sy', 'f4'),
              ('bg', 'f4'), ('lpx', 'f4'), ('lpy', 'f4'), ('likelihood', 'f4')]


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)

with open(_ospath.join(_this_directory, 'config.yaml'), 'r') as config_file:
    CONFIG = _yaml.load(config_file)

_woehrlok_file = _ospath.join(_this_directory, 'WoehrLok.dll')
WoehrLok = _ctypes.CDLL(_woehrlok_file)
# Visual C compiler mangles up the function name (thanks Microsoft):
WoehrLok.fnWoehrLokMLEFitAll = getattr(WoehrLok, '?fnWoehrLokMLEFitAll@@YAXHPEBMMHHPEAM11KHPEAK@Z')


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
def local_gradient_magnitude(gm, box):
    """ Returns the sum of the absolute gradient within a box around each pixel """
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


def identify_in_frame(frame, parameters, roi=None):
    frame = _np.float32(frame)  # For some reason frame sometimes comes with different type, so we don't want to confuse numba
    if roi is not None:
        frame = frame[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
    gm = gradient_magnitude(frame)
    box = parameters['Box Size']
    s_map = local_gradient_magnitude(gm, box)
    lm_map = local_maxima_map(frame, box)
    s_map_thesholded = s_map > parameters['Minimum LGM']
    combined_map = (lm_map * s_map_thesholded) > 0.5
    y, x = _np.where(combined_map)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x


def identify_by_frame_number(movie, parameters, frame_number, roi=None):
    frame = movie[frame_number]
    y, x = identify_in_frame(frame, parameters, roi)
    frame = frame_number * _np.ones(len(x))
    return _np.rec.array((frame, x, y), dtype=[('frame', 'i'), ('x', 'i'), ('y', 'i')])


def identify_async(movie, parameters, roi=None):
    n_frames = len(movie)
    n_threads = int(0.75 * _multiprocessing.cpu_count())
    executor = _ThreadPoolExecutor(n_threads)
    futures = [executor.submit(identify_by_frame_number, movie, parameters, _, roi) for _ in range(n_frames)]
    executor.shutdown(wait=False)
    return futures


def identify(movie, parameters, threaded=True):
    if threaded:
        futures = identify_async(movie, parameters)
        done, not_done = _wait(futures)
        identifications = [future.result() for future in done]
    else:
        identifications = [identify_by_frame_number(movie, parameters, i) for i in range(movie)]
    return _np.hstack(identifications).view(_np.recarray)


@_numba.jit(nopython=True)
def _get_spots(movie, ids_frame, ids_x, ids_y, box):
    n_spots = len(ids_x)
    r = int(box/2)
    spots = _np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        for yi, y in enumerate(range(yc-r, yc+r+1)):
            for xi, x in enumerate(range(xc-r, xc+r+1)):
                spots[id, yi, xi] = movie[frame, y, x]
    return spots


def _to_photons(spots, camera_info):
    spots = _np.float32(spots)
    if camera_info['sensor'] == 'EMCCD':
        return (spots - 100) * camera_info['sensitivity'] / (camera_info['gain'] * camera_info['qe'])
    elif camera_info['sensor'] == 'sCMOS':
        return (spots - 100) * 0.45 / camera_info['qe']        # sensitivity is for 200 MHz, 16 bit, Miblab Zyla
    else:
        raise TypeError('Unknown camera type')


def _generate_fit_info(movie, camera_info, identifications, box):
    spots = _get_spots(movie, identifications.frame, identifications.x, identifications.y, box)
    n_spots, box, box = spots.shape
    fit_type = _ctypes.c_int(4)
    spots = _to_photons(spots, camera_info)
    spots_pointer = spots.ctypes.data_as(_C_FLOAT_POINTER)
    psf_sigma = _ctypes.c_float(1.0)
    box = _ctypes.c_int(box)
    n_iterations = _ctypes.c_int(30)
    params = _np.zeros((6, n_spots), dtype=_np.float32)
    params_pointer = params.ctypes.data_as(_C_FLOAT_POINTER)
    CRLBs = _np.zeros((6, n_spots), dtype=_np.float32)
    CRLBs_pointer = CRLBs.ctypes.data_as(_C_FLOAT_POINTER)
    likelihoods = _np.zeros(n_spots, dtype=_np.float32)
    likelihoods_pointer = likelihoods.ctypes.data_as(_C_FLOAT_POINTER)
    n_spots = _ctypes.c_ulong(n_spots)
    n_cpus = _multiprocessing.cpu_count()
    n_threads = _ctypes.c_int(int(0.75 * n_cpus))
    current = _np.array(0, dtype=_np.uint32)
    current_pointer = current.ctypes.data_as(_ctypes.POINTER(_ctypes.c_ulong))
    gaussmle_args = (fit_type, spots_pointer, psf_sigma, box, n_iterations, params_pointer, CRLBs_pointer,
                     likelihoods_pointer, n_spots, n_threads, current_pointer)
    FitInfo = _namedtuple('FitInfo', 'n_spots spots gaussmle_args current params CRLBs likelihoods')
    fit_info = FitInfo(n_spots.value, spots, gaussmle_args, current, params, CRLBs, likelihoods)
    return fit_info


def fit(movie, camera_info, identifications, box):
    fit_info = _generate_fit_info(movie, camera_info, identifications, box)
    WoehrLok.fnWoehrLokMLEFitAll(*fit_info.gaussmle_args)
    return locs_from_fit_info(fit_info, identifications, box), fit_info


def fit_async(movie, camera_info, identifications, box):
    fit_info = _generate_fit_info(movie, camera_info, identifications, box)
    thread = _threading.Thread(target=WoehrLok.fnWoehrLokMLEFitAll, args=fit_info.gaussmle_args)
    thread.start()
    return thread, fit_info


def locs_from_fit_info(fit_info, identifications, box):
    box_offset = int(box/2)
    x = fit_info.params[0] + identifications.x - box_offset
    y = fit_info.params[1] + identifications.y - box_offset
    lpx = _np.sqrt(fit_info.CRLBs[0])
    lpy = _np.sqrt(fit_info.CRLBs[1])
    return _np.rec.array((identifications.frame, x, y,
                          fit_info.params[2], fit_info.params[4], fit_info.params[5],
                          fit_info.params[3], lpx, lpy, fit_info.likelihoods),
                         dtype=LOCS_DTYPE)


def localize(movie, info, parameters):
    identifications = identify(movie, parameters)
    return fit(movie, info, identifications, parameters['Box Size'])
