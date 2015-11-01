"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in an frame sequence

    :author: Joerg Schnitzbauer, 2015
"""

import numpy as np
import numba
import multiprocessing
import functools
import ctypes
import threading


C_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)


WoehrLok = ctypes.CDLL('WoehrLok')
# Visual C compiler mangles up the function name (thanks Microsoft!)
WoehrLok.fnWoehrLokMLEFitAll = getattr(WoehrLok, '?fnWoehrLokMLEFitAll@@YAXHPEBMMHHPEAM11KHPEAK@Z')


@numba.jit(nopython=True)
def local_maxima_map(frame, roi):
    """ Finds pixels with maximum value within a region of interest """
    X, Y = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    roi_half = int(roi / 2)
    for i in range(roi, X - roi):
        for j in range(roi, Y - roi):
            local_frame = frame[i - roi_half:i + roi_half + 1, j - roi_half:j + roi_half + 1]
            flat_max = np.argmax(local_frame)
            j_local_max = int(flat_max / roi)
            i_local_max = int(flat_max % roi)
            if (i_local_max == roi_half) and (j_local_max == roi_half):
                maxima_map[i, j] = 1
    return maxima_map


@numba.jit(nopython=True)
def local_gradient_magnitude(frame, roi, abs_gradient):
    """ Returns the sum of the absolute gradient within a ROI around each pixel """
    X, Y = frame.shape
    lgm = np.zeros_like(abs_gradient)
    roi_half = int(roi / 2)
    for i in range(roi, X - roi):
        for j in range(roi, Y - roi):
            local_gradient = abs_gradient[i - roi_half:i + roi_half + 1, j - roi_half:j + roi_half + 1]
            lgm[i, j] = np.sum(local_gradient)
    return lgm


def identify_frame(frame, parameters):
    gradient_x, gradient_y = np.gradient(np.float32(frame))
    abs_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    roi = parameters['ROI']
    s_map = local_gradient_magnitude(frame, roi, abs_gradient)
    lm_map = local_maxima_map(frame, roi)
    s_map_thesholded = s_map > parameters['Minimum LGM']
    combined_map = (lm_map * s_map_thesholded) > 0.5
    return np.vstack(np.where(combined_map)).T


# The target function for the processing pool, ready for functools.partial
def _identify_frame_async(parameters, counter, lock, frame):
    with lock:
        counter.value += 1
    return identify_frame(frame, parameters)


def identify_async(movie, parameters):
    n_cpus = multiprocessing.cpu_count()
    if len(movie) < n_cpus:
        n_processes = len(movie)
    else:
        n_processes = 2 * n_cpus
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    # _counter.value = 0
    targetfunc = functools.partial(_identify_frame_async, parameters, counter, lock)
    pool = multiprocessing.Pool(processes=n_processes)
    result = pool.map_async(targetfunc, movie)
    return result, counter, pool


def identify(movie, parameters):
    return [identify_frame(frame, parameters) for frame in movie]


def get_spots(movie, identifications, roi):
    n_spots_frame = [len(_) for _ in identifications]
    n_spots = sum(n_spots_frame)
    spots = np.zeros((n_spots, roi, roi), dtype=np.float32)
    r = int(roi/2)
    i = 0
    for frame_number, identifications_frame in enumerate(identifications):
        for x, y in identifications_frame:
            spots[i] = movie[frame_number, x-r:x+r+1, y-r:y+r+1]
    return n_spots, n_spots_frame, spots


def prepare_fit(spots, info):
    args = {}
    n_spots, roi, _ = spots.shape
    fit_type = ctypes.c_int(4)
    spots = np.float32(spots)
    spots = (spots - info['Baseline']) * info['Quantum Efficiency'] * info['Pre-Amp Gain'] / info['Real EM Gain']
    spots_pointer = spots.ctypes.data_as(C_FLOAT_POINTER)
    psf_sigma = ctypes.c_float(1.0)
    roi = ctypes.c_uint(roi)
    n_iterations = ctypes.c_uint(20)
    params = np.zeros((6, n_spots), dtype=np.float32)
    params_pointer = params.ctypes.data_as(C_FLOAT_POINTER)
    CRLBs = np.zeros((6, n_spots), dtype=np.float32)
    CRLBs_pointer = CRLBs.ctypes.data_as(C_FLOAT_POINTER)
    likelihoods = np.zeros(n_spots, dtype=np.float32)
    likelihoods_pointer = likelihoods.ctypes.data_as(C_FLOAT_POINTER)
    n_spots = ctypes.c_long(n_spots)
    n_cpus = multiprocessing.cpu_count()
    n_threads = ctypes.c_uint(2*n_cpus)
    current = ctypes.c_long(0)
    args = (fit_type, spots_pointer, psf_sigma, roi, n_iterations, params_pointer, CRLBs_pointer,
            likelihoods_pointer, n_spots, n_threads, ctypes.byref(current))
    results = (current, params, CRLBs, likelihoods)
    return args, results


def fit(spots, info):
    args, results = prepare_fit(spots, info)
    WoehrLok.fnWoehrLokMLEFitAll(*args)
    return results[1:]


def fit_async(spots, info):
    args, results = prepare_fit(spots, info)
    thread = threading.Thread(target=WoehrLok.fnWoehrLokMLEFitAll, args=args)
    thread.start()
    return (thread,) + results
