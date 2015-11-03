"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in an frame sequence

    :author: Joerg Schnitzbauer, 2015
"""

import numpy as np
import numba
import multiprocessing
import ctypes
import threading
import os.path
from collections import namedtuple
from numpy.lib import recfunctions


C_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)


this_file = os.path.abspath(__file__)
this_directory = os.path.dirname(this_file)
woehrlok_file = os.path.join(this_directory, 'WoehrLok.dll')
WoehrLok = ctypes.CDLL(woehrlok_file)
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


def identify_frame(frame, parameters, frame_number=None):
    gradient_x, gradient_y = np.gradient(np.float32(frame))
    abs_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    roi = parameters['ROI']
    s_map = local_gradient_magnitude(frame, roi, abs_gradient)
    lm_map = local_maxima_map(frame, roi)
    s_map_thesholded = s_map > parameters['Minimum LGM']
    combined_map = (lm_map * s_map_thesholded) > 0.5
    if frame_number is not None:
        x, y = np.where(combined_map)
        frame = frame_number * np.ones(len(x))
        return np.rec.array((frame, x, y), dtype=[('frame', 'i'), ('x', 'i'), ('y', 'i')])
    else:
        return np.rec.array(np.where(combined_map), dtype=[('x', 'i'), ('y', 'i')])


def _identify_frame_async(frame, parameters, frame_number, counter, lock):
    with lock:
        counter.value += 1
    return identify_frame(frame, parameters, frame_number)


def identify_async(movie, parameters):
    n_frames = len(movie)
    n_cpus = multiprocessing.cpu_count()
    if n_frames < n_cpus:
        n_processes = n_frames
    else:
        n_processes = 2 * n_cpus
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    pool = multiprocessing.Pool(processes=n_processes)
    args = [(movie[_], parameters, _, counter, lock) for _ in range(n_frames)]
    result = pool.starmap_async(_identify_frame_async, args)
    return result, counter, pool


def identify(movie, parameters, threaded=True):
    if threaded:
        n_frames = len(movie)
        n_cpus = multiprocessing.cpu_count()
        if n_frames < n_cpus:
            n_processes = n_frames
        else:
            n_processes = 2 * n_cpus
        pool = multiprocessing.Pool(processes=n_processes)
        args = [(movie[_], parameters) for _ in range(n_frames)]
        result = pool.starmap(identify_frame, args)
        identifications = result.get()
    else:
        identifications = [identify_frame(parameters, frame) for frame in movie]
    return np.hstack(identifications).view(np.recarray)


@numba.jit(nopython=True)
def get_spots(movie, ids_frame, ids_x, ids_y, roi):
    n_spots = len(ids_x)
    r = int(roi/2)
    spots = np.zeros((n_spots, roi, roi), dtype=movie.dtype)
    for frame, xc, yc in zip(ids_frame, ids_x, ids_y):
        for xi, x in enumerate(range(xc-r, xc+r+1)):
            for yi, y in enumerate(range(yc-r, yc+r+1)):
                spots[frame, xi, yi] = movie[frame, x, y]
    return spots


FitInfo = namedtuple('FitInfo', 'n_spots spots gaussmle_args current params CRLBs likelihoods')


def prepare_fit(movie, info, identifications, roi):
    spots = get_spots(movie, identifications.frame, identifications.x, identifications.y, roi)
    n_spots, roi, roi = spots.shape
    fit_type = ctypes.c_int(4)
    spots = np.float32(spots)
    spots = (spots - info['baseline']) * info['quantum efficiency'] * info['preamp gain'] / info['em realgain']
    spots_pointer = spots.ctypes.data_as(C_FLOAT_POINTER)
    psf_sigma = ctypes.c_float(1.0)
    roi = ctypes.c_int(roi)
    n_iterations = ctypes.c_int(20)
    params = np.zeros((6, n_spots), dtype=np.float32)
    params_pointer = params.ctypes.data_as(C_FLOAT_POINTER)
    CRLBs = np.zeros((6, n_spots), dtype=np.float32)
    CRLBs_pointer = CRLBs.ctypes.data_as(C_FLOAT_POINTER)
    likelihoods = np.zeros(n_spots, dtype=np.float32)
    likelihoods_pointer = likelihoods.ctypes.data_as(C_FLOAT_POINTER)
    n_spots = ctypes.c_ulong(n_spots)
    n_cpus = multiprocessing.cpu_count()
    n_threads = ctypes.c_int(2*n_cpus)
    current = np.array(0, dtype=np.uint32)
    current_pointer = current.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong))
    gaussmle_args = (fit_type, spots_pointer, psf_sigma, roi, n_iterations, params_pointer, CRLBs_pointer,
                     likelihoods_pointer, n_spots, n_threads, current_pointer)
    fit_info = FitInfo(n_spots.value, spots, gaussmle_args, current, params, CRLBs, likelihoods)
    return fit_info


def fit(movie, info, identifications, roi):
    fit_info = prepare_fit(movie, info, identifications, roi)
    WoehrLok.fnWoehrLokMLEFitAll(*fit_info.gaussmle_args)
    locs = np.rec.array(fit_info.params.T.flatten(), dtype=[('x', 'f4'), ('y', 'f4'), ('photons', 'f4'),
                                                            ('bg', 'f4'), ('sx', 'f4'), ('sy', 'f4')])
    identifications_flat = np.vstack(identifications)
    locs.x += identifications_flat[:, 1] - roi/2 + 1
    locs.y += identifications_flat[:, 0] - roi/2 + 1
    frames = np.zeros(fit_info.n_spots, dtype=np.uint32)
    end_frame_indices = list(np.cumsum(fit_info.n_spots_frame))
    start_frame_indices = [0] + end_frame_indices[:-1]
    for frame_number, (start_frame_index, end_frame_index) in enumerate(zip(start_frame_indices, end_frame_indices)):
        frames[start_frame_index:end_frame_index] = frame_number
    locs = recfunctions.append_fields(locs, ['frame', 'likelihood'], [frames, fit_info.likelihoods],
                                      usemask=False, asrecarray=True)
    return locs


def fit_async(movie, info, identifications, roi):
    fit_info = prepare_fit(movie, info, identifications, roi)
    thread = threading.Thread(target=WoehrLok.fnWoehrLokMLEFitAll, args=fit_info.gaussmle_args)
    thread.start()
    return thread, fit_info
