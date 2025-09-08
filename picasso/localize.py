"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in a frame
    sequence.

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Literal
from typing import Callable
from datetime import datetime

import numba
import numpy as np
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from . import io, gaussmle, postprocess

plt.style.use("ggplot")


MAX_LOCS = int(1e6)
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
SET_COLS = [
    "Frames", "Height", "Width", "Box Size", "Min. Net Gradient", "Pixelsize",
]


@numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(
    frame: np.ndarray,
    box: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find pixels with maximum value within a region of interest.

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
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half:i + box_half + 1,
                j - box_half:j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(
    frame: np.ndarray,
    y: int,
    x: int,
    i: int,
) -> tuple[float, float]:
    """Calculate the gradient at a specific pixel in the frame.

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


@numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(
    frame: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    box: int,
    uy: np.ndarray,
    ux: np.ndarray,
) -> np.ndarray:
    """Calculate the net gradient at the identified maxima in the
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
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(
                range(xi - box_half, xi + box_half + 1)
            ):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(
    image: np.ndarray,
    minimum_ng: float,
    box: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify local maxima in the image and calculate the net gradient
    at those maxima.

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
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(
    frame: np.ndarray,
    minimum_ng: float,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify local maxima in a single frame with an optionally
    specified subregion (ROI) and calculate the net gradient at those
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
        frame = frame[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
    image = np.float32(frame)  # otherwise numba goes crazy
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x, net_gradient


def identify_by_frame_number(
    movie: np.ndarray,
    minimum_ng: float,
    box: int,
    frame_number: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    lock: threading.Lock | None = None,
) -> np.recarray:
    """Identify local maxima in a specific frame of a movie and
    calculate the net gradient at those maxima. Optionally, a lock can
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
    frame = frame_number * np.ones(len(x))
    return np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
    )


def _identify_worker(
    movie: np.ndarray,
    current: list[int],
    minimum_ng: float,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None,
    lock: threading.Lock | None,
) -> list[np.recarray]:
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
    futures: list[multiprocessing.pool.Future],
) -> np.recarray:
    """Collect the results from a list of futures and combines them
    into a single structured numpy array.

    Parameters
    ----------
    futures : list of multiprocessing.pool.Future's
        A list of futures representing the asynchronous tasks.

    Returns
    -------
    ids : np.recarray
        A structured numpy array containing the combined results from
        all futures. Contains fields `frame`, `x`, `y`, and
        `net_gradient`.
    """
    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = list(chain(*ids_list_of_lists))
    ids = np.hstack(ids_list).view(np.recarray)
    ids.sort(kind="mergesort", order="frame")
    return ids


def identify_async(
    movie: np.ndarray,
    minimum_ng: float,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[list[int], list[multiprocessing.pool.Future]]:
    """Asynchronously (i.e., using multithreading) identify local
    maxima in a movie using multiple threads. This function divides the
    work among a specified number of threads.

    Parameters
    ----------
    movie : np.ndarray
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
    f : list[multiprocessing.pool.Future]
            A list of futures representing the asynchronous tasks.
    """
    # Use the user settings to define the number of workers that are being used
    settings = io.load_user_settings()

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
    io.save_user_settings(settings)

    n_workers = min(
        60, max(1, int(cpu_utilization * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores

    lock = threading.Lock()
    current = [0]
    executor = ThreadPoolExecutor(n_workers)
    f = [
        executor.submit(
            _identify_worker, movie, current, minimum_ng, box, roi, lock,
        )
        for _ in range(n_workers)
    ]
    executor.shutdown(wait=False)
    return current, f


def identify(
    movie: np.ndarray,
    minimum_ng: float,
    box: int,
    threaded: bool = True,
) -> np.recarray:
    """Identify local maxima in a movie and calculate the net
    gradient at those maxima. This function can run in a threaded or
    non-threaded mode.

    Parameters
    ----------
    movie : np.ndarray
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
    box : int
        The size of the box to extract around each spot.
    threaded : bool
        Whether to use threading for the identification process.

    Returns
    -------
    ids : np.recarray
        A structured numpy array containing the identified spots.
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    """
    if threaded:
        current, futures = identify_async(movie, minimum_ng, box)
        identifications = [_.result() for _ in futures]
        identifications = [np.hstack(_) for _ in identifications]
    else:
        identifications = [
            identify_by_frame_number(movie, minimum_ng, box, i)
            for i in range(len(movie))
        ]
    ids = np.hstack(identifications).view(np.recarray)
    return ids


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba(
    movie: np.ndarray,
    ids_frame: np.ndarray,
    ids_x: np.ndarray,
    ids_y: np.ndarray,
    box: int,
) -> np.ndarray:
    """Extract the spots out of a movie using Numba for performance."""
    n_spots = len(ids_x)
    r = int(box / 2)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r:yc + r + 1, xc - r:xc + r + 1]
    return spots


@numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame: np.ndarray,
    frame_number: int,
    ids_frame: np.ndarray,
    ids_x: np.ndarray,
    ids_y: np.ndarray,
    r: int,
    start: int,
    N: int,
    spots: np.ndarray,
) -> int:
    """Extract spots from a movie frame."""
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        if ids_frame[j] < frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r:yc + r + 1, xc - r:xc + r + 1]
    return j


@numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(movie, l_mov, ids_frame, ids_x, ids_y, box, spots):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : np.ndarray
        The input movie data as a 3D numpy array.
    l_mov : np.ndarray
        Length of the movie, a 1D array with a single element.
    ids_frame, ids_x, ids_y : np.ndarray
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : np.ndarray
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.

    Returns
    -------
    spots : np.ndarray
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
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : np.ndarray
        The input movie data as a 3D numpy array.
    ids_frame, ids_x, ids_y : np.ndarray
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : np.ndarray
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.

    Returns
    -------
    spots : np.ndarray
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


def _cut_spots(movie: np.ndarray, ids: np.ndarray, box: int) -> np.ndarray:
    """Cut out spots from a movie based on the identified positions."""
    N = len(ids.frame)
    if isinstance(movie, np.ndarray):
        return _cut_spots_numba(movie, ids.frame, ids.x, ids.y, box)
    elif isinstance(movie, io.ND2Movie) and movie.use_dask:
        """ Assumes that identifications are in order of frames! """
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = da.apply_gufunc(
            _cut_spots_daskmov,
            '(p,n,m),(b),(k),(k),(k),(),(k,l,l)->(k,l,l)',
            movie.data,
            np.array([len(movie)]),
            ids.frame,
            ids.x,
            ids.y,
            box,
            spots,
            output_dtypes=[movie.dtype],
            allow_rechunk=True,
        ).compute()
        return spots
    else:
        """Assumes that identifications are in order of frames!"""
        N = len(ids.frame)
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie, ids.frame, ids.x, ids.y, box, spots)
        return spots


def _to_photons(spots: np.ndarray, camera_info: dict) -> np.ndarray:
    """Convert the cut spots to photon counts based on camera
    information."""
    spots = np.float32(spots)
    baseline = camera_info["Baseline"]
    sensitivity = camera_info["Sensitivity"]
    gain = camera_info["Gain"]
    # since v0.6.0: remove quantum efficiency to better reflect precision
    # qe = camera_info["Qe"]
    return (spots - baseline) * sensitivity / (gain)


def get_spots(
    movie: np.ndarray,
    identifications: np.recarray,
    box: int,
    camera_info: dict,
) -> np.ndarray:
    """Extract the spots from a movie based on the identified positions
    and convert camera signal to photon counts.

    Parameters
    ----------
    movie : np.ndarray
        The input movie data as a 3D numpy array.
    identifications : np.recarray
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
    spots : np.ndarray
        A 3D numpy array containing the extracted spots, with shape
        (k, box, box), where k is the number of spots identified.
    """
    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def fit(
    movie: np.ndarray,
    camera_info: dict,
    identifications: np.recarray,
    box: int,
    eps: float = 0.001,
    max_it: int = 100,
    method: Literal["sigma", "sigmaxy"] = "sigma",
) -> np.recarray:
    """Fit Gaussians using Maximum Likelihood Estimation (MLE) to the
    identified spots in a movie to localize fluorescent molecules. See
    Smith, et al. Nature Methods, 2010. DOI: 10.1038/nmeth.1449.

    Parameters
    ----------
    movie : np.ndarray
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    identifications : np.recarray
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
    locs : np.recarray
        A structured numpy array containing the localized spots. The
        fields include `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`,
        `lpx`, `lpy`, `net_gradient`, `likelihood`, and `iterations`.
    """
    spots = get_spots(movie, identifications, box, camera_info)
    theta, CRLBs, likelihoods, iterations = gaussmle.gaussmle(
        spots, eps, max_it, method=method
    )
    return locs_from_fits(
        identifications, theta, CRLBs, likelihoods, iterations, box,
    )


def fit_async(
    movie: np.ndarray,
    camera_info: dict,
    identifications: np.recarray,
    box: int,
    eps: float = 0.001,
    max_it: int = 100,
    method: Literal["sigma", "sigmaxy"] = "sigmaxy",
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Asynchronously fit Gaussians using Maximum Likelihood Estimation
    (MLE) to the identified spots in a movie to localize fluorescent
    molecules. This function is designed to run in a separate thread or
    process. See Smith, et al. Nature Methods, 2010.
    DOI: 10.1038/nmeth.1449.

    Parameters
    ----------
    movie : np.ndarray
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    identifications : np.recarray
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
    thetas : np.ndarray
        The fitted Gaussian parameters for each spot (x, y positions,
        photon counts, background, single-emitter image size in x and
        y).
    CRLBs : np.ndarray
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : np.ndarray
        The log-likelihoods of the fitted models.
    iterations : np.ndarray
        The number of iterations taken to converge for each spot.
    """
    spots = get_spots(movie, identifications, box, camera_info)
    return gaussmle.gaussmle_async(spots, eps, max_it, method=method)


def locs_from_fits(
    identifications: np.recarray,
    theta: np.ndarray,
    CRLBs: np.ndarray,
    likelihoods: np.ndarray,
    iterations: np.ndarray,
    box: int,
) -> np.recarray:
    """Convert the resulting localizations from the list of Futures
    into a structured array.

    Parameters
    ----------
    identifications : np.recarray
        A structured numpy array containing the identified spots.
        Contains fields `frame`, `x`, `y`, and `net_gradient`.
    theta : np.ndarray
        The fitted Gaussian parameters for each spot (x, y positions,
        photon counts, background, single-emitter image size in x and
        y).
    CRLBs : np.ndarray
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : np.ndarray
        The log-likelihoods of the fitted models.
    iterations : np.ndarray
        The number of iterations taken to converge for each spot.
    box : int
        Size of the box used for fitting. Should be an odd integer.

    Returns
    -------
    locs : np.recarray
        A structured numpy array containing the localized spots. The
        fields include `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`,
        `lpx`, `lpy`, `net_gradient`, `likelihood`, and `iterations`.
    """
    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])
    locs = np.rec.array(
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
    movie: np.ndarray,
    camera_info: dict,
    parameters: dict,
) -> np.recarray:
    """Localize (i.e., identify and fit) spots in a movie using
    the specified parameters.

    Parameters
    ----------
    movie : np.ndarray
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
    locs : np.recarray
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
    locs: np.recarray,
    info: None,
    callback: Callable[[int], None] = None,
) -> float:
    """Calculate the NeNA (experimental localization precision) from
    localizations.

    Parameters
    ----------
    locs : np.recarray
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
    print('Calculating NeNA.. ', end='')
    locs = locs[0:MAX_LOCS]
    try:
        result, nena_px = postprocess.nena(locs, None, callback=callback)
    except Exception as e:
        print(e)
        nena_px = float("nan")
    print(f"{nena_px:.2f} px.")
    return nena_px


def check_kinetics(locs: np.recarray, info: list[dict]) -> float:
    """Calculate the mean length of binding events from localizations.

    Parameters
    ----------
    locs : np.recarray
        A structured numpy array containing the localized spots.
    info : list of dicts
        A list of dictionaries containing metadata about the movie.

    Returns
    -------
    len_mean : float
        The mean length of binding events in frames.
    """
    print("Linking.. ", end='')
    locs = locs[0:MAX_LOCS]
    locs = postprocess.link(locs, info=info)
    len_mean = locs.len.mean()
    print(f"Mean length {len_mean:.2f} frames.")
    return len_mean


def check_drift(
    locs: np.recarray,
    info: list[dict],
    callback: Callable[[int], None] = None,
) -> tuple[float, float]:
    """Estimate the drift of localizations in x and y directions.

    Parameters
    ----------
    locs : np.recarray
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
    drift, locs = postprocess.undrift(
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
    """Generate a summary of the localization file, including metadata
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

    locs, info = io.load_locs(file_hdf)

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
    """Return the path to the SQLite database file used for storing
    localization summaries. The database is stored in the user's home
    directory under the ``.picasso`` folder."""
    home = os.path.expanduser("~")
    picasso_dir = os.path.join(home, ".picasso")
    os.makedirs(picasso_dir, exist_ok=True)
    return os.path.abspath(os.path.join(picasso_dir, "app_0410.db"))


def save_file_summary(summary: dict) -> None:
    """Save the summary of a localization file to a SQLite database."""
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
    """Add a localization file summary to the SQLite database."""
    summary = get_file_summary(file, file_hdf, drift, len_mean, nena)
    save_file_summary(summary)
