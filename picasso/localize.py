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
def local_maxima_map(frame, roi):
    """ Finds pixels with maximum value within a region of interest """
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    roi_half = int(roi / 2)
    for i in range(roi, Y - roi):
        for j in range(roi, X - roi):
            local_frame = frame[i - roi_half:i + roi_half + 1, j - roi_half:j + roi_half + 1]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / roi)
            j_local_max = int(flat_max % roi)
            if (i_local_max == roi_half) and (j_local_max == roi_half):
                maxima_map[i, j] = 1
    return maxima_map


@_numba.jit(nopython=True, nogil=True)
def local_gradient_magnitude(frame, roi, abs_gradient):
    """ Returns the sum of the absolute gradient within a ROI around each pixel """
    Y, X = frame.shape
    lgm = _np.zeros_like(abs_gradient)
    roi_half = int(roi / 2)
    for i in range(roi_half, Y - roi + roi_half + 1):
        for j in range(roi_half, X - roi + roi_half + 1):
            local_gradient = abs_gradient[i - roi_half:i + roi_half + 1, j - roi_half:j + roi_half + 1]
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


def identify_in_frame(frame, parameters):
    gm = gradient_magnitude(frame)
    roi = parameters['ROI']
    s_map = local_gradient_magnitude(frame, roi, gm)
    lm_map = local_maxima_map(frame, roi)
    s_map_thesholded = s_map > parameters['Minimum LGM']
    combined_map = (lm_map * s_map_thesholded) > 0.5
    y, x = _np.where(combined_map)
    return y, x


def identify_by_frame_number(movie, parameters, frame_number):
    frame = movie[frame_number]
    y, x = identify_in_frame(frame, parameters)
    frame = frame_number * _np.ones(len(x))
    return _np.rec.array((frame, x, y), dtype=[('frame', 'i'), ('x', 'i'), ('y', 'i')])


def identify_async(movie, parameters):
    n_frames = len(movie)
    n_threads = int(0.75 * _multiprocessing.cpu_count())
    executor = _ThreadPoolExecutor(n_threads)
    futures = [executor.submit(identify_by_frame_number, movie, parameters, _) for _ in range(n_frames)]
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
def _get_spots(movie, ids_frame, ids_x, ids_y, roi):
    n_spots = len(ids_x)
    r = int(roi/2)
    spots = _np.zeros((n_spots, roi, roi), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        for yi, y in enumerate(range(yc-r, yc+r+1)):
            for xi, x in enumerate(range(xc-r, xc+r+1)):
                spots[id, yi, xi] = movie[frame, y, x]
    return spots


def _to_photons(spots, info):
    spots = _np.float32(spots)
    if info[0]['Camera'] == 'Andor Zyla':
        return spots - 100
    if info[0]['Camera']['Manufacturer'] == 'Andor':
        type = info[0]['Camera']['Type']
        model = info[0]['Camera']['Model']
        serial_number = info[0]['Camera']['Serial Number']
        camera_config = CONFIG['Cameras']['Andor'][type][model][serial_number]
        baseline = camera_config['Baseline']
        em = info[0]['Electron Multiplying']
        if em:
            gain = info[0]['EM Real Gain']
        else:
            gain = 1
        preamp_gain = info[0]['Pre-Amp Gain']
        read_mode = info[0]['Readout Mode']
        sensitivity = camera_config['Sensitivity'][em][read_mode][preamp_gain-1]
        excitation = info[0]['Excitation Wavelength']
        try:
            qe = camera_config['Quantum Efficiency'][excitation]
        except KeyError:
            _ = list(camera_config['Quantum Efficiency'].keys())
            raise Exception('Valid excitation wavelengths are: {}\nAdjust your yaml file!'.format(_))
        return (spots - baseline) * sensitivity / (gain * qe)
    else:
        raise Exception("No configuration found for camera '{}''".format(info[0]['Camera']))


def _generate_fit_info(movie, info, identifications, roi):
    spots = _get_spots(movie, identifications.frame, identifications.x, identifications.y, roi)
    n_spots, roi, roi = spots.shape
    fit_type = _ctypes.c_int(4)
    spots = _to_photons(spots, info)
    spots_pointer = spots.ctypes.data_as(_C_FLOAT_POINTER)
    psf_sigma = _ctypes.c_float(1.0)
    roi = _ctypes.c_int(roi)
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
    gaussmle_args = (fit_type, spots_pointer, psf_sigma, roi, n_iterations, params_pointer, CRLBs_pointer,
                     likelihoods_pointer, n_spots, n_threads, current_pointer)
    FitInfo = _namedtuple('FitInfo', 'n_spots spots gaussmle_args current params CRLBs likelihoods')
    fit_info = FitInfo(n_spots.value, spots, gaussmle_args, current, params, CRLBs, likelihoods)
    return fit_info


def fit(movie, info, identifications, roi):
    fit_info = _generate_fit_info(movie, info, identifications, roi)
    WoehrLok.fnWoehrLokMLEFitAll(*fit_info.gaussmle_args)
    return locs_from_fit_info(fit_info, identifications, roi), fit_info


def fit_async(movie, info, identifications, roi):
    fit_info = _generate_fit_info(movie, info, identifications, roi)
    thread = _threading.Thread(target=WoehrLok.fnWoehrLokMLEFitAll, args=fit_info.gaussmle_args)
    thread.start()
    return thread, fit_info


def locs_from_fit_info(fit_info, identifications, roi):
    roi_offset = int(roi/2)
    x = fit_info.params[0] + identifications.x - roi_offset
    y = fit_info.params[1] + identifications.y - roi_offset
    lpx = _np.sqrt(fit_info.CRLBs[0])
    lpy = _np.sqrt(fit_info.CRLBs[1])
    return _np.rec.array((identifications.frame, x, y,
                          fit_info.params[2], fit_info.params[4], fit_info.params[5],
                          fit_info.params[3], lpx, lpy, fit_info.likelihoods),
                         dtype=LOCS_DTYPE)


def localize(movie, info, parameters):
    identifications = identify(movie, parameters)
    return fit(movie, info, identifications, parameters['ROI'])
