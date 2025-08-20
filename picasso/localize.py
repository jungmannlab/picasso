"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in a frame sequence

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations
from typing import Literal
from typing import Callable

import numpy as _np
import dask.array as _da
import numba as _numba
import multiprocessing as _multiprocessing
import ctypes as _ctypes
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import threading as _threading
from itertools import chain as _chain
import matplotlib.pyplot as _plt
from . import gaussmle as _gaussmle
from . import io as _io
from . import postprocess as _postprocess
from . import __main__ as main
import os
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

MAX_LOCS = int(1e6)

_C_FLOAT_POINTER = _ctypes.POINTER(_ctypes.c_float)
LOCS_DTYPE = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
]

MEAN_COLS = [
    "frame",
    "x",
    "y",
    "photons",
    "sx",
    "sy",
    "bg",
    "lpx",
    "lpy",
    "ellipticity",
    "net_gradient",
    "z",
    "d_zcalib",
]
SET_COLS = ["Frames", "Height", "Width", "Box Size", "Min. Net Gradient", "Pixelsize"]

_plt.style.use("ggplot")


@_numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(
    frame: _np.ndarray, 
    box: int
) -> tuple[_np.ndarray, _np.ndarray]:
    """Finds pixels with maximum value within a region of interest.
    
    Parameters
    ----------
    frame : np.ndarray
        An image frame, 2D array of shape (Y, X).
    box : int
        Size of the box to search for local maxima. Should be an odd 
        integer.
        
    Returns
    -------
    y : np.ndarray
        y-coordinates of the local maxima.
    x : np.ndarray
        x-coordinates of the local maxima.
    """
    
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = _np.where(maxima_map)
    return y, x


@_numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(
    frame: _np.ndarray, 
    y: int, 
    x: int, 
    i: int,
) -> tuple[float, float]:
    """Calculates the gradient at a specific pixel in the frame.
    
    Parameters
    ----------
    frame : np.ndarray
        An image frame, 2D array of shape (Y, X).
    y, x : int
        Coordinates of the pixel where the gradient is calculated.
    i : int
        Index of the pixel in the list of maxima. Not used in this 
        function.
    
    Returns
    -------
    gy : float
        Gradient in the y-direction at the pixel (y, x).
    gx : float
        Gradient in the x-direction at the pixel (y, x).
    """

    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@_numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(
    frame: _np.ndarray, 
    y: _np.ndarray, 
    x: _np.ndarray, 
    box: int, 
    uy: _np.ndarray, 
    ux: _np.ndarray,
) -> _np.ndarray:
    """Calculates the net gradient at the identified maxima in the 
    frame.

    Parameters
    ----------
    frame : np.ndarray
        An image frame, 2D array of shape (Y, X).
    y, x : np.ndarray
        Coordinates of the identified maxima in the frame.
    box : int
        Size of the box used for calculating the gradient.
    uy, ux : np.ndarray
        Arrays of shape (box, box) containing the y and x components
        of the gradient, respectively.

    Returns
    -------
    ng : np.ndarray
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """

    box_half = int(box / 2)
    ng = _np.zeros(len(x), dtype=_np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
    return ng


@_numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(
    image: _np.ndarray, 
    minimum_ng: float, 
    box: int,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Identifies local maxima in the image and calculates the net
    gradient at those maxima.
    
    Parameters
    ----------
    image : np.ndarray
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer. 
        
    Returns
    -------
    y : np.ndarray
        y-coordinates of the identified maxima.
    x : np.ndarray
        x-coordinates of the identified maxima.
    ng : np.ndarray
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """

    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = _np.zeros((box, box), dtype=_np.float32)
    uy = _np.zeros((box, box), dtype=_np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = _np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(
    frame: _np.ndarray, 
    minimum_ng: float, 
    box: int, 
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Identifies local maxima in a single frame with an optionally
    specified subregion (ROI) and calculates the net gradient at those 
    maxima.
    
    Parameters
    ----------
    frame : np.ndarray
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates 
        (y_start, x_start) and the second tuple contains the end 
        coordinates (y_end, x_end). If None, the entire frame is used. 
    
    Returns
    -------
    y : np.ndarray
        y-coordinates of the identified maxima.
    x : np.ndarray
        x-coordinates of the identified maxima.
    net_gradient : np.ndarray
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """

    if roi is not None:
        frame = frame[roi[0][0] : roi[1][0], roi[0][1] : roi[1][1]]
    image = _np.float32(frame)  # otherwise numba goes crazy
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x, net_gradient


# def identify_frame(
#     frame: _np.ndarray, 
#     minimum_ng: float, 
#     box: int, 
#     frame_number: int, 
#     roi: tuple[tuple[int, int], tuple[int, int]] | None = None, 
#     resultqueue: _multiprocessing.Queue | None = None
# ) -> _np.recarray:
#     """Identifies local maxima in a single frame and calculates the net
#     gradient at those maxima. Optionally, the results can be put into a
#     multiprocessing queue.

#     Parameters
#     ----------
#     frame : np.ndarray
#         An image frame, 2D array of shape (Y, X).
#     minimum_ng : float
#         Minimum net gradient value to consider a maximum as valid.
#     box : int
#         Size of the box used for calculating the gradient. Should be
#         an odd integer.
#     frame_number : int
#         The index of the frame in the movie sequence.
#     roi : tuple, optional
#         Region of interest (ROI) defined as a tuple of two tuples,
#         where the first tuple contains the start coordinates 
#         (y_start, x_start) and the second tuple contains the end 
#         coordinates (y_end, x_end). If None, the entire frame is used. 
#     resultqueue : multiprocessing.Queue, optional
#         If provided, the results will be put into this queue as a
#         structured numpy array. 

#     Returns
#     -------
#     result : np.recarray
#         A structured numpy array containing the frame number, x and y
#         coordinates of the identified maxima, and their net gradient.   
#     """

#     y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
#     frame = frame_number * _np.ones(len(x))
#     result = _np.rec.array(
#         (frame, x, y, net_gradient),
#         dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
#     )
#     if resultqueue is not None:
#         resultqueue.put(result)
#     return result


def identify_by_frame_number(
    movie: _np.ndarray, 
    minimum_ng: float, 
    box: int, 
    frame_number: int, 
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None, 
    lock: _threading.Lock | None = None,
) -> _np.recarray:
    """Identifies local maxima in a specific frame of a movie and
    calculates the net gradient at those maxima. Optionally, a lock can
    be used to ensure thread safety when accessing the movie data.
    
    Parameters
    ----------
    movie : np.ndarray
        A 3D array representing the movie of shape (N, Y, X), where N is
        the number of frames, Y is the height, and X is the width.
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.
    frame_number : int
        The index of the frame in the movie sequence to be processed.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates 
        (y_start, x_start) and the second tuple contains the end 
        coordinates (y_end, x_end). If None, the entire frame is used.
    lock : threading.Lock, optional
        If provided, this lock will be used to ensure thread safety when
        accessing the movie data. This is useful in a multithreaded
        environment.

    Returns
    -------
    result : np.recarray
        A structured numpy array containing the frame number, x and y
        coordinates of the identified maxima, and their net gradient.
    """

    if lock is not None:
        with lock:
            frame = movie[frame_number]
    else:
        frame = movie[frame_number]
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * _np.ones(len(x))
    return _np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
    )


def _identify_worker(
    movie: _np.ndarray, 
    current: list[int], 
    minimum_ng: float,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None,
    lock: _threading.Lock | None,
) -> list[_np.recarray]:
    """Worker function for identifying local maxima in a movie. This   
    function is designed to be run in a separate thread and processes
    each frame independently."""

    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(
            identify_by_frame_number(movie, minimum_ng, box, index, roi, lock)
        )


def identifications_from_futures(
    futures: list[_multiprocessing.pool.Future],
) -> _np.recarray:
    """Collects the results from a list of futures and combines them
    into a single structured numpy array.

    Parameters
    ----------
    futures : list of _multiprocessing.pool.Future's
        A list of futures representing the asynchronous tasks.

    Returns
    -------
    ids : _np.recarray
        A structured numpy array containing the combined results from 
        all futures. Contains fields `frame`, `x`, `y`, and 
        `net_gradient`.
    """

    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = list(_chain(*ids_list_of_lists))
    ids = _np.hstack(ids_list).view(_np.recarray)
    ids.sort(kind="mergesort", order="frame")
    return ids


def identify_async(
    movie: _np.ndarray, 
    minimum_ng: float, 
    box: int, 
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[list[int], list[_multiprocessing.pool.Future]]:
    """Asynchronously (i.e., using multithreading) identifies local 
    maxima in a movie using multiple threads. This function divides the 
    work among a specified number of threads.
    
    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
    box : int
        The size of the box to extract around each spot.
    roi : tuple[tuple[int, int], tuple[int, int]] | None
        The region of interest (ROI) for the analysis.

    Returns
    -------
    current : list[int]
        A list of frame indices representing the current processing 
        state.
    f : list[_multiprocessing.pool.Future]
            A list of futures representing the asynchronous tasks.
    """

    # Use the user settings to define the number of workers that are being used
    settings = _io.load_user_settings()

    # avoid the problem when cpu_utilization is not set
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
    except KeyError:
        cpu_utilization = 0.8

    if isinstance(cpu_utilization, float):
        if cpu_utilization >= 1:
            cpu_utilization = 0.8
    else:
        print("CPU utilization was not set. Setting to 0.8")
        cpu_utilization = 0.8
    settings["Localize"]["cpu_utilization"] = cpu_utilization
    _io.save_user_settings(settings)

    n_workers = min(
        60, max(1, int(cpu_utilization * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores

    lock = _threading.Lock()
    current = [0]
    executor = _ThreadPoolExecutor(n_workers)
    f = [
        executor.submit(_identify_worker, movie, current, minimum_ng, box, roi, lock)
        for _ in range(n_workers)
    ]
    executor.shutdown(wait=False)
    return current, f


def identify(
    movie: _np.ndarray, 
    minimum_ng: float, 
    box: int, 
    threaded: bool = True,
) -> _np.recarray:
    """Identifies local maxima in a movie and calculates the net
    gradient at those maxima. This function can run in a threaded or
    non-threaded mode.

    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
    box : int
        The size of the box to extract around each spot.
    threaded : bool
        Whether to use threading for the identification process.

    Returns
    -------
    ids : _np.recarray
        A structured numpy array containing the identified spots. 
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    """

    if threaded:
        current, futures = identify_async(movie, minimum_ng, box)
        identifications = [_.result() for _ in futures]
        identifications = [_np.hstack(_) for _ in identifications]
    else:
        identifications = [
            identify_by_frame_number(movie, minimum_ng, box, i)
            for i in range(len(movie))
        ]
    ids = _np.hstack(identifications).view(_np.recarray)
    return ids


@_numba.jit(nopython=True, cache=False)
def _cut_spots_numba(
    movie: _np.ndarray, 
    ids_frame: _np.ndarray, 
    ids_x: _np.ndarray, 
    ids_y: _np.ndarray, 
    box: int,
) -> _np.ndarray:
    """Extracts the spots out of a movie using Numba for performance."""
    
    n_spots = len(ids_x)
    r = int(box / 2)
    spots = _np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


@_numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame: _np.ndarray, 
    frame_number: int, 
    ids_frame: _np.ndarray, 
    ids_x: _np.ndarray, 
    ids_y: _np.ndarray, 
    r: int, 
    start: int, 
    N: int, 
    spots: _np.ndarray,
) -> int:
    """Extracts spots from a movie frame."""

    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        if ids_frame[j] < frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return j


@_numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(movie, l_mov, ids_frame, ids_x, ids_y, box, spots):
    """Extracts the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    l_mov : _np.ndarray
        Length of the movie, a 1D array with a single element.
    ids_frame, ids_x, ids_y : _np.ndarray
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : _np.ndarray
        3D array to store the cut spots, with shape (k, box, box), 
        where k is the number of spots identified.
    
    Returns
    -------
    spots : _np.ndarray
        3D array with extracted spots of shape (k, box, box), where k is 
        the number of spots identified.
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number in range(l_mov[0]):
        frame = movie[frame_number, :, :]
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots_framebyframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Extracts the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    ids_frame, ids_x, ids_y : _np.ndarray
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : _np.ndarray
        3D array to store the cut spots, with shape (k, box, box), 
        where k is the number of spots identified.
    
    Returns
    -------
    spots : _np.ndarray
        3D array with extracted spots of shape (k, box, box), where k is 
        the number of spots identified.
    """

    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number, frame in enumerate(movie):
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots(movie: _np.ndarray, ids: _np.ndarray, box: int) -> _np.ndarray:
    """Cuts out spots from a movie based on the identified positions."""

    N = len(ids.frame)
    if isinstance(movie, _np.ndarray):
        return _cut_spots_numba(movie, ids.frame, ids.x, ids.y, box)
    elif isinstance(movie, _io.ND2Movie) and movie.use_dask:
    # elif movie.use_dask:
        """ Assumes that identifications are in order of frames! """
        spots = _np.zeros((N, box, box), dtype=movie.dtype)
        spots = _da.apply_gufunc(
            _cut_spots_daskmov,
            '(p,n,m),(b),(k),(k),(k),(),(k,l,l)->(k,l,l)',
            movie.data, _np.array([len(movie)]), ids.frame, ids.x, ids.y, box, spots,
            output_dtypes=[movie.dtype], allow_rechunk=True).compute()
        return spots
    else:
        """Assumes that identifications are in order of frames!"""
        
        r = int(box / 2)
        N = len(ids.frame)
        spots = _np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie, ids.frame, ids.x, ids.y, box, spots)
        return spots


def _to_photons(spots: _np.ndarray, camera_info: dict) -> _np.ndarray:
    """Converts the cut spots to photon counts based on camera 
    information."""

    spots = _np.float32(spots)
    baseline = camera_info["Baseline"]
    sensitivity = camera_info["Sensitivity"]
    gain = camera_info["Gain"]
    # since v0.6.0: remove quantum efficiency to better reflect precision
    # qe = camera_info["Qe"]
    return (spots - baseline) * sensitivity / (gain)


def get_spots(
    movie: _np.ndarray, 
    identifications: _np.recarray, 
    box: int, 
    camera_info: dict,
) -> _np.ndarray:
    """Extracts spots from a movie based on the identified positions and
    converts camera signal to photon counts.
    
    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    identifications : _np.recarray
        A structured numpy array containing the identified spots.
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    
    Returns
    -------
    spots : _np.ndarray
        A 3D numpy array containing the extracted spots, with shape
        (k, box, box), where k is the number of spots identified.
    """

    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def fit(
    movie: _np.ndarray,
    camera_info: dict,
    identifications: _np.recarray,
    box: int,
    eps: float = 0.001,
    max_it: int = 100,
    method: Literal["sigma", "sigmaxy"] = "sigma",
) -> _np.recarray:
    """Fits Gaussians using Maximum Likelihood Estimation (MLE) to the 
    identified spots in a movie to localize fluorescent molecules. See
    Smith, et al. Nature Methods, 2010. DOI: 10.1038/nmeth.1449.
    
    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    identifications : _np.recarray
        A structured numpy array containing the identified spots.
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    eps : float, optional
        The convergence criterion for the fitting algorithm. Default is
        0.001.
    max_it : int, optional
        The maximum number of iterations for the fitting algorithm. 
        Default is 100.
    method : Literal["sigma", "sigmaxy"], optional
        The method used for fitting (impose same sigma in x and y or 
        not, respectively). Default is "sigma".

    Returns
    -------
    locs : _np.recarray
        A structured numpy array containing the localized spots. The
        fields include `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`, 
        `lpx`, `lpy`, `net_gradient`, `likelihood`, and `iterations`.
    """
    
    spots = get_spots(movie, identifications, box, camera_info)
    theta, CRLBs, likelihoods, iterations = _gaussmle.gaussmle(
        spots, eps, max_it, method=method
    )
    return locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box)


def fit_async(
    movie: _np.ndarray,
    camera_info: dict,
    identifications: _np.recarray,
    box: int,
    eps: float = 0.001,
    max_it: int = 100,
    method: Literal["sigma", "sigmaxy"] = "sigmaxy",
) -> tuple[int, _np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray]:
    """Asynchronously fits Gaussians using Maximum Likelihood Estimation
    (MLE) to the identified spots in a movie to localize fluorescent
    molecules. This function is designed to run in a separate thread or
    process. See Smith, et al. Nature Methods, 2010. 
    DOI: 10.1038/nmeth.1449.

    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    identifications : _np.recarray
        A structured numpy array containing the identified spots.
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    eps : float, optional       
        The convergence criterion for the fitting algorithm. Default is
        0.001.
    max_it : int, optional  
        The maximum number of iterations for the fitting algorithm.
        Default is 100.
    method : Literal["sigma", "sigmaxy"], optional
        The method used for fitting (impose same sigma in x and y or 
        not, respectively). Default is "sigmaxy".

    Returns
    -------
    current : int
        Index of the currently processed spot.
    thetas : _np.ndarray
        The fitted Gaussian parameters for each spot (x, y positions, 
        photon counts, background, single-emitter image size in x and 
        y).
    CRLBs : _np.ndarray
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : _np.ndarray
        The log-likelihoods of the fitted models.
    iterations : _np.ndarray
        The number of iterations taken to converge for each spot.
    """

    spots = get_spots(movie, identifications, box, camera_info)
    return _gaussmle.gaussmle_async(spots, eps, max_it, method=method)


def locs_from_fits(
    identifications: _np.recarray, 
    theta: _np.ndarray, 
    CRLBs: _np.ndarray, 
    likelihoods: _np.ndarray, 
    iterations: _np.ndarray, 
    box: int,
) -> _np.recarray:
    """Converts the resulting localizations from the list of Futures 
    into a structured array.

    Parameters
    ----------
    identifications : _np.recarray
        A structured numpy array containing the identified spots.
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    theta : _np.ndarray
        The fitted Gaussian parameters for each spot (x, y positions, 
        photon counts, background, single-emitter image size in x and 
        y).
    CRLBs : _np.ndarray
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : _np.ndarray
        The log-likelihoods of the fitted models.
    iterations : _np.ndarray
        The number of iterations taken to converge for each spot.       
    box : int
        Size of the box used for fitting. Should be an odd integer. 
    
    Returns
    -------
    locs : _np.recarray
        A structured numpy array containing the localized spots. The
        fields include `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`, 
        `lpx`, `lpy`, `net_gradient`, `likelihood`, and `iterations`.
    """

    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    lpy = _np.sqrt(CRLBs[:, 0])
    lpx = _np.sqrt(CRLBs[:, 1])
    locs = _np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 2],
            theta[:, 5],
            theta[:, 4],
            theta[:, 3],
            lpx,
            lpy,
            identifications.net_gradient,
            likelihoods,
            iterations,
        ),
        dtype=LOCS_DTYPE,
    )
    locs.sort(kind="mergesort", order="frame")
    return locs


def localize(
    movie: _np.ndarray, 
    camera_info: dict, 
    parameters: dict,
) -> _np.recarray:
    """Localizes (i.e., identifies and fits) spots in a movie using
    the specified parameters. This function combines the identification
    and fitting steps into a single process.
    
    Parameters
    ----------
    movie : _np.ndarray
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    parameters : dict
        A dictionary containing localization parameters, including:
        - `Min. Net Gradient`: Minimum net gradient for spot
          identification.
        - `Box Size`: Size of the box to cut out around each spot.

    Returns
    -------
    locs : _np.recarray
        A structured numpy array containing the localized spots.
        The fields include `frame`, `x`, `y`, `photons`, `sx`, `sy`,
        `bg`, `lpx`, `lpy`, `net_gradient`, `likelihood`, and 
        `iterations`.
    """

    identifications = identify(
        movie,
        parameters["Min. Net Gradient"],
        parameters["Box Size"],
    )
    locs = fit(movie, camera_info, identifications, parameters["Box Size"])
    return locs


def check_nena(
    locs: _np.recarray, 
    info: None, 
    callback: Callable[[int], None] = None,
) -> float:
    """Calculates the NeNA (experimental localization precision) from
    localizations.
    
    Parameters
    ----------
    locs : _np.recarray
        A structured numpy array containing the localized spots.
    info : None
        Not used.
    callback : Callable[[int], None], optional
        A callback function that can be used to report progress. It
        should accept an integer argument representing the current
        step or frame number. Default is None.
    
    Returns
    -------
    nena_px : float
        The NeNA value in pixels, representing the experimental
        localization precision.
    """

    # Nena
    print('Calculating NeNA.. ', end ='')
    locs = locs[0:MAX_LOCS]
    try:
        result, nena_px = _postprocess.nena(locs, None, callback=callback)
    except Exception as e:
        print(e)
        nena_px = float("nan")

    print(f"{nena_px:.2f} px.")

    return nena_px


def check_kinetics(locs: _np.recarray, info: list[dict]) -> float:
    """Calculates the mean length of binding events from localizations.

    Parameters
    ----------
    locs : _np.recarray
        A structured numpy array containing the localized spots.
    info : list[dict]
        A list of dictionaries containing metadata about the movie.
    
    Returns
    -------
    len_mean : float
        The mean length of binding events in frames.    
    """

    print("Linking.. ", end ='')
    locs = locs[0:MAX_LOCS]
    locs = _postprocess.link(locs, info=info)
    len_mean = locs.len.mean()
    print(f"Mean length {len_mean:.2f} frames.")

    return len_mean


def check_drift(
    locs: _np.recarray, 
    info: list[dict], 
    callback: Callable[[int], None] = None,
) -> tuple[float, float]:
    """Estimates the drift of localizations in x and y directions.
    
    Parameters
    ----------
    locs : _np.recarray
        A structured numpy array containing the localized spots.
    info : list[dict]
        A list of dictionaries containing metadata about the movie.
    callback : Callable[[int], None], optional
        A callback function that can be used to report progress. It 
        should accept an integer argument representing the current
        step or frame number. Default is None.

    Returns
    -------
    drift_x : float
        The estimated drift in the x direction.
    drift_y : float
        The estimated drift in the y direction.
    """

    steps = int(len(locs) // (MAX_LOCS))
    steps = max(1, steps)

    locs = locs[::steps]

    n_frames = info[0]["Frames"]

    segmentation = max(1, int(n_frames // 10))

    print(f"Estimating drift with segmentation {segmentation}")
    drift, locs = _postprocess.undrift(
        locs,
        info,
        segmentation,
        display=False,
        rcc_callback=callback,
    )

    drift_x = float(drift["x"].mean())
    drift_y = float(drift["y"].mean())

    print(f"Drift is X: {drift_x:.2f}, Y: {drift_y:.2f}.")

    return (drift_x, drift_y)


def get_file_summary(
    file: str, 
    file_hdf: str, 
    drift: tuple[float, float] | None = None, 
    len_mean: float | None = None, 
    nena: float | None = None,
) -> dict:
    """Generates a summary of the localization file, including metadata
    and statistics about the localizations.

    Parameters
    ----------
    file : str
        The path to the localization file (HDF5 format).
    file_hdf : str
        The path to the HDF5 file containing localizations.
    drift : tuple[float, float] | None, optional
        A tuple containing the drift in x and y directions. If None,
        the drift will be calculated from the localizations.
    len_mean : float | None, optional
        The mean length of binding events in frames. If None, it will
        be calculated from the localizations.
    nena : float | None, optional
        The NeNA value in pixels. If None, it will be calculated from
        the localizations.

    Returns
    -------
    summary : dict
        A dictionary containing the summary of the localization file,
        including metadata and statistics about the localizations.
    """

    if file_hdf is None:
        base, ext = os.path.splitext(file)
        file_hdf = base + "_locs.hdf5"

    locs, info = _io.load_locs(file_hdf)

    summary = {}

    for col in MEAN_COLS:
        try:
            summary[col + "_mean"] = locs[col].mean()
            summary[col + "_std"] = locs[col].std()
        except ValueError:
            summary[col + "_mean"] = float("nan")
            summary[col + "_std"] = float("nan")

    for col in SET_COLS:
        col_ = col.lower()
        for inf in info:
            if col in inf:
                summary[col_] = inf[col]

    for col in SET_COLS:
        col_ = col.lower()
        if col_ not in summary:
            summary[col_] = float("nan")

    if nena is None:
        summary["nena_px"] = check_nena(locs, info)
    else:
        summary["nena_px"] = nena

    if len_mean is None:
        len_mean = check_kinetics(locs, info)
    else:
        len_mean = len_mean

    if drift is None:
        drift_x, drift_y = check_drift(locs, info)
    else:
        drift_x, drift_y = drift

    summary["len_mean"] = len_mean
    summary["n_locs"] = len(locs)
    summary["locs_frame"] = len(locs) / summary["frames"]

    summary["drift_x"] = drift_x
    summary["drift_y"] = drift_y

    summary["nena_nm"] = summary["nena_px"] * summary["pixelsize"]

    summary["filename"] = os.path.normpath(file)
    summary["filename_hdf"] = file_hdf
    summary["file_created"] = datetime.fromtimestamp(os.path.getmtime(file))
    summary["entry_created"] = datetime.now()

    return summary


def _db_filename() -> str:
    """Returns the path to the SQLite database file used for storing
    localization summaries. The database is stored in the user's home
    directory under the `.picasso` folder."""

    home = os.path.expanduser("~")
    picasso_dir = os.path.join(home, ".picasso")
    os.makedirs(picasso_dir, exist_ok=True)
    return os.path.abspath(os.path.join(picasso_dir, "app_0410.db"))


def save_file_summary(summary: dict) -> None:
    """Saves the summary of a localization file to a SQLite database."""

    engine = create_engine("sqlite:///" + _db_filename(), echo=False)
    s = pd.Series(summary, index=summary.keys()).to_frame().T
    s.to_sql("files", con=engine, if_exists="append", index=False)


def add_file_to_db(
    file: str, 
    file_hdf: str, 
    drift: tuple[float, float] | None = None, 
    len_mean: float | None = None, 
    nena: float | None = None
) -> None:
    """Adds a localization file summary to the SQLite database."""
    
    summary = get_file_summary(file, file_hdf, drift, len_mean, nena)
    save_file_summary(summary)