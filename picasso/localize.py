"""
picasso.localize
~~~~~~~~~~~~~~~~

Identify and localize fluorescent single molecules in a frame
sequence.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import multiprocessing
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
from itertools import chain
from typing import Literal
from typing import Callable
from datetime import datetime

import numba
import numpy as np
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sqlalchemy import create_engine

from .ext import bitplane

from . import (
    io,
    lib,
    gausslq,
    gaussmle,
    avgroi,
    postprocess,
    zfit,
    __version__,
)

plt.style.use("ggplot")


MAX_LOCS = int(1e6)
# The columns under base are always available and the keys such as "3D
# only" will be displayed in the save columns dialog in the GUI for
# clarity
LOCALIZATION_COLUMNS = {
    "Base": [
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
    ],
    "3D only": ["z", "d_zcalib", "lpz"],
    "Picked spots only": ["n_id"],
    "MLE only": ["log_likelihood", "iterations"],
}
# For database:
MEAN_COLS = LOCALIZATION_COLUMNS["Base"] + LOCALIZATION_COLUMNS["3D only"]
SET_COLS = [
    "Frames",
    "Height",
    "Width",
    "Box Size",
    "Min. Net Gradient",
    "Pixelsize",
]


@numba.jit(nopython=True, nogil=True, cache=False)
def _local_maxima(
    frame: lib.IntArray2D, box: int
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Find pixels with maximum value within a region of interest.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    box : int
        Size of the box to search for local maxima. Should be an odd
        integer.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the local maxima.
    x : lib.IntArray1D
        x-coordinates of the local maxima.
    """
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def _gradient_at(
    frame: lib.IntArray2D,
    y: int,
    x: int,
    i: int,
) -> tuple[float, float]:
    """Calculate the gradient at a specific pixel in the frame.

    Parameters
    ----------
    frame : lib.IntArray2D
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
def _net_gradient(
    frame: lib.IntArray2D,
    y: lib.IntArray1D,
    x: lib.IntArray1D,
    box: int,
    uy: lib.FloatArray2D,
    ux: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Calculate the net gradient at the identified maxima in the
    frame.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    y, x : lib.IntArray1D
        Coordinates of the identified maxima in the frame.
    box : int
        Size of the box used for calculating the gradient.
    uy, ux : lib.FloatArray2D
        Arrays of shape (box, box) containing the y and x components
        of the gradient, respectively.

    Returns
    -------
    ng : lib.FloatArray1D
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
                    gy, gx = _gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(
    image: lib.IntArray2D,
    minimum_ng: float,
    box: int,
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Identify local maxima in the image and calculate the net gradient
    at those maxima.

    Parameters
    ----------
    image : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the identified maxima.
    x : lib.IntArray1D
        x-coordinates of the identified maxima.
    ng : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    y, x = _local_maxima(image, box)
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
    ng = _net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(
    frame: lib.IntArray2D,
    minimum_ng: float,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Identify local maxima in a single frame with an optionally
    specified subregion (ROI) and calculate the net gradient at those
    maxima.

    Parameters
    ----------
    frame : lib.IntArray2D
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
    y : lib.IntArray1D
        y-coordinates of the identified maxima.
    x : lib.IntArray1D
        x-coordinates of the identified maxima.
    net_gradient : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    if roi is not None:
        frame = frame[roi[0][0] : roi[1][0], roi[0][1] : roi[1][1]]
    image = np.float32(frame)  # otherwise numba goes crazy
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x, net_gradient


def identify_by_frame_number(
    movie: lib.IntArray3D,
    minimum_ng: float,
    box: int,
    frame_number: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    lock: threading.Lock | None = None,
) -> pd.DataFrame:
    """Identify local maxima in a specific frame of a movie and
    calculate the net gradient at those maxima. Optionally, a lock can
    be used to ensure thread safety when accessing the movie data.

    Parameters
    ----------
    movie : lib.IntArray3D
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
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. If only min or max
        is to be specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    lock : threading.Lock, optional
        If provided, this lock will be used to ensure thread safety when
        accessing the movie data. This is useful in a multithreaded
        environment. Default is None.

    Returns
    -------
    identifications : pd.DataFrame
        DataFrame containing the frame number, x and y coordinates of
        the identified maxima, and their net gradient.
    """
    if lock is not None:
        with lock:
            frame = movie[frame_number]
    else:
        frame = movie[frame_number]
    # check frame bounds
    min_max = (0, len(movie))
    if frame_bounds is not None:
        if frame_bounds[0] is not None:
            min_max = (max(frame_bounds[0], min_max[0]), min_max[1])
        if frame_bounds[1] is not None:
            min_max = (min_max[0], min(frame_bounds[1], min_max[1]))
        if not (min_max[0] <= frame_number <= min_max[1]):
            return pd.DataFrame(
                {
                    "frame": pd.Series(dtype=int),
                    "x": pd.Series(dtype=int),
                    "y": pd.Series(dtype=int),
                    "net_gradient": pd.Series(dtype=np.float32),
                }
            )
    # identify
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * np.ones(len(x))
    identifications = pd.DataFrame(
        {
            "frame": frame.astype(int),
            "x": x.astype(int),
            "y": y.astype(int),
            "net_gradient": net_gradient.astype(np.float32),
        }
    )
    return identifications


def _identify_worker(
    movie: lib.IntArray3D,
    current: list[int],
    minimum_ng: float,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | None,
    frame_bounds: tuple[int, int] | None,
    lock: threading.Lock | None,
) -> list[pd.DataFrame]:
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
            identify_by_frame_number(
                movie,
                minimum_ng,
                box,
                index,
                roi=roi,
                frame_bounds=frame_bounds,
                lock=lock,
            )
        )


def identifications_from_futures(
    futures: list[multiprocessing.pool.Future],
) -> pd.DataFrame:
    """Collect the results from a list of futures and combines them
    into a single ``DataFrame``.

    Parameters
    ----------
    futures : list of multiprocessing.pool.Future's
        A list of futures representing the asynchronous tasks.

    Returns
    -------
    ids : pd.DataFrame
        Data frame containing the combined results from
        all futures. Contains fields ``frame``, ``x``, ``y``, and
        ``net_gradient``.
    """
    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = list(chain(*ids_list_of_lists))
    ids = pd.concat(ids_list, ignore_index=True)
    ids.sort_values(by="frame", kind="quicksort", inplace=True)
    return ids


def identify_async(
    movie: lib.IntArray3D,
    minimum_ng: float,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
) -> tuple[list[int], list[multiprocessing.pool.Future]]:
    """Asynchronously (i.e., using multithreading) identify local
    maxima in a movie using multiple threads. This function divides the
    work among a specified number of threads.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
    box : int
        The size of the box to extract around each spot.
    roi : tuple[tuple[int, int], tuple[int, int]] | None
        The region of interest (ROI) for the analysis.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. If only min or max
        is to be specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.

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
            _identify_worker,
            movie,
            current,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            lock,
        )
        for _ in range(n_workers)
    ]
    executor.shutdown(wait=False)
    return current, f


def _identify_threaded(
    movie,
    minimum_ng,
    box,
    roi,
    frame_bounds,
    progress_callback,
    abort_callback,
):
    """Run identify_async and drive its progress loop.

    Returns the identifications, or None if aborted.
    """
    N = len(movie)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(total=N, desc="Identifying spots", unit="frame")
        if use_tqdm
        else None
    )
    current, futures = identify_async(
        movie, minimum_ng, box, roi=roi, frame_bounds=frame_bounds
    )
    last = 0
    while current[0] < N:
        if abort_callback is not None and abort_callback():
            for f in futures:
                f.cancel()
            if use_tqdm:
                iter_range.close()
            return None
        if use_tqdm:
            iter_range.update(current[0] - last)
            last = current[0]
        elif callable(progress_callback):
            progress_callback(current[0])
        time.sleep(0.2)
    if use_tqdm:
        iter_range.update(N - last)
        iter_range.close()
    return identifications_from_futures(futures)


def _identify_serial(
    movie,
    minimum_ng,
    box,
    roi,
    frame_bounds,
    progress_callback,
):
    """Identify spots frame-by-frame in the current thread."""
    N = len(movie)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(range(N), desc="Identifying spots", unit="frame")
        if use_tqdm
        else range(N)
    )
    identifications = []
    for i in iter_range:
        identifications.append(
            identify_by_frame_number(
                movie,
                minimum_ng,
                box,
                i,
                roi=roi,
                frame_bounds=frame_bounds,
            )
        )
        if callable(progress_callback):
            progress_callback(i)
    ids = pd.concat(identifications, ignore_index=True)
    ids.sort_values(by="frame", kind="quicksort", inplace=True)
    return ids


def identify(
    movie: lib.IntArray3D,
    minimum_ng: float,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    threaded: bool = True,
    progress_callback: (
        Callable[[list[int]], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Identify local maxima in a movie and calculate the net
    gradient at those maxima. This function can run in a threaded or
    non-threaded mode.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    minimum_ng : float
        The minimum net gradient for a spot to be considered.
    box : int
        The size of the box to extract around each spot.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. If only min or max
        is to be specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    threaded : bool, optional
        Whether to use threading for the identification process. Default
        is True.
    progress_callback : callable, "console" or None, optional
        A callback function to report the progress of the identification
        process. If "console", progress will be printed to the console.
        If None, no progress will be reported. Default is None.
    abort_callback : callable, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted. Default is
        None.
    return_info : bool, optional
        Whether to return additional information about the fitting
        process. Default is True. If True, a tuple of (locs, info) is
        returned. In v0.12.0 return_info will be removed and the
        function will always return info.

    Returns
    -------
    ids : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    info : dict, optional
        Additional information about the identification process, such as
        the time taken for identification. Only returned if `return_info`
        is True.
    """
    if not return_info:
        # TODO: remove in v0.12.0
        lib.deprecation_warning(
            "In version 0.12, return_info argument will be removed such "
            "that picasso.localize.localize() will always return both "
            "the localizations and the metadata dictionary."
        )
    if threaded:
        ids = _identify_threaded(
            movie,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            progress_callback,
            abort_callback,
        )
        if ids is None:
            return
    else:
        ids = _identify_serial(
            movie,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            progress_callback,
        )
    if return_info:
        info = {
            "Generated by": f"Picasso: v{__version__} Identify",
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
            "ROI": roi,
            "Frame Bounds": frame_bounds,
        }
        return ids, info
    else:
        return ids


def picks_to_identifications(
    picks: list[tuple],
    *,
    n_frames: int | None = None,
    drift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert circular picks (from Picasso: Render) to identifications.
    Only circular picks are allowed.

    Parameters
    ----------
    picks : list of tuples
        List of circular picks positions (centers). See
        ``io.load_picks``.
    n_frames : int, optional
        Number of frames in the acquisition movie. If None is given,
        it will be extracted from the drift file (if provided).
        Otherwise, an error is raised.
    drift : pd.DataFrame or None, optional
        A data frame of length n_frames and with columns 'x' and 'y'.
        Used to adjust the positions of identifications throughout
        acquisition. Only x and y drift is used; if 'z' is present, it
        is ignored.

    Returns
    -------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`. Note that `net_gradient`
        is a dummy value.

    Raises
    ------
    ValueError
        If `n_frames` and `drift` are not provided.
    """
    assert isinstance(picks, (list, tuple)), "picks must be a list or a tuple."
    assert all([len(_) == 2 for _ in picks]), (
        "Circular picks are required. Each element in 'picks' must "
        "contain two numbers (x and y coordinates)."
    )
    if isinstance(drift, pd.DataFrame):
        assert all(
            col in drift.columns for col in ["x", "y"]
        ), "Drift data frame must contain 'x' and 'y' columns."
    if n_frames is None:
        if drift is None:
            raise ValueError(
                "n_frames must be given if no drift file is provided"
            )
        else:
            n_frames = len(drift)
    else:
        assert isinstance(n_frames, int), "n_frames must be an integer."
        if drift is not None:
            assert n_frames == len(drift), (
                f"{n_frames} frames were provided but the drift suggests"
                f" {len(drift)} frames."
            )
    return _picks_to_identifications(picks, n_frames, drift)


def _picks_to_identifications(
    picks: list[tuple],
    n_frames: int,
    drift: pd.DataFrame | None,
) -> pd.DataFrame:
    """Convert circular picks to identifications, can be drift-corrected.
    Assumes correct inputs. See ``picks_to_identifications`` for more
    details."""
    data = []
    n_id = 0
    for pick_x, pick_y in picks:
        # drifted:
        xloc = np.ones((n_frames,), dtype=float) * pick_x
        yloc = np.ones((n_frames,), dtype=float) * pick_y
        if drift is not None:
            xloc += drift["x"].to_numpy()
            yloc += drift["y"].to_numpy()

        frames = np.arange(n_frames)
        gradient = np.ones(n_frames) + 100
        n_id_all = np.ones(n_frames) + n_id
        temp = np.array([frames, xloc, yloc, gradient, n_id_all])
        data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
        n_id += 1

    data = [item for sublist in data for item in sublist]
    identifications = pd.DataFrame(
        {
            "frame": [item[0] for item in data],
            "x": [item[1] for item in data],
            "y": [item[2] for item in data],
            "net_gradient": [item[3] for item in data],
            "n_id": [item[4] for item in data],
        }
    )
    identifications.sort_values(
        by="frame",
        inplace=True,
        kind="quicksort",
    )
    return identifications


def locs_to_identifications(
    locs: pd.DataFrame,
    movie_info: list[dict],
    n_frames: int,
) -> pd.DataFrame:
    """Convert localizations to identifications.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    movie_info : list of dicts
        Movie file metadata.
    n_frames : int
        Number of frames around localizations that are to be used for
        extracting identifications.

    Returns
    -------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`. Note that `net_gradient`
        is a dummy value.
    """
    assert isinstance(
        locs, pd.DataFrame
    ), "Localizations must be a pandas data frame"
    assert (
        isinstance(n_frames, int) and n_frames >= 0
    ), "n_frames must be a non-negative integer"
    max_frames = lib.get_from_metadata(movie_info, "Frames", raise_error=True)
    data = []
    n_id = 0
    for _, element in locs.iterrows():
        currframe = element["frame"]
        if currframe > n_frames and currframe < (max_frames - n_frames):
            xloc = np.ones((2 * n_frames + 1,), dtype=float) * element["x"]
            yloc = np.ones((2 * n_frames + 1,), dtype=float) * element["y"]
            frames = np.arange(
                currframe - n_frames,
                currframe + n_frames + 1,
            )
            gradient = np.ones(2 * n_frames + 1) + 100
            n_id_all = np.ones(2 * n_frames + 1) + n_id
            temp = np.array([frames, xloc, yloc, gradient, n_id_all])
            data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
        n_id += 1
    data = [item for sublist in data for item in sublist]
    identifications = pd.DataFrame(
        {
            "frame": [item[0] for item in data],
            "x": [item[1] for item in data],
            "y": [item[2] for item in data],
            "net_gradient": [item[3] for item in data],
            "n_id": [item[4] for item in data],
        }
    )
    return identifications


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba(
    movie: lib.IntArray3D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
) -> lib.IntArray3D:
    """Extract the spots out of a movie using Numba for performance."""
    n_spots = len(ids_x)
    r = int(box / 2)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


@numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame: lib.IntArray2D,
    frame_number: int,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    r: int,
    start: int,
    N: int,
    spots: lib.IntArray3D,
) -> int:
    """Extract spots from a movie frame."""
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        if ids_frame[j] < frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return j


@numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(
    movie: lib.IntArray3D,
    l_mov: lib.IntArray1D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    l_mov : lib.IntArray1D
        Length of the movie, a 1D array with a single element.
    ids_frame, ids_x, ids_y : lib.IntArray1D
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : lib.IntArray3D
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.

    Returns
    -------
    spots : lib.IntArray3D
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


def _cut_spots_framebyframe(
    movie: lib.IntArray3D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    ids_frame, ids_x, ids_y : lib.IntArray1D
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : lib.IntArray3D
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.

    Returns
    -------
    spots : lib.IntArray3D
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


def _cut_spots(
    movie: lib.IntArray3D, ids: pd.DataFrame, box: int
) -> lib.IntArray3D:
    """Cut out spots from a movie based on the identified positions."""
    N = len(ids)
    if isinstance(movie, np.ndarray):
        return _cut_spots_numba(
            movie,
            ids["frame"].to_numpy(),
            ids["x"].to_numpy(),
            ids["y"].to_numpy(),
            box,
        )
    elif isinstance(movie, io.ND2Movie) and movie.use_dask:
        """Assumes that identifications are in order of frames!"""
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = da.apply_gufunc(
            _cut_spots_daskmov,
            "(p,n,m),(b),(k),(k),(k),(),(k,l,l)->(k,l,l)",
            movie.data,
            np.array([len(movie)]),
            ids["frame"].to_numpy(),
            ids["x"].to_numpy(),
            ids["y"].to_numpy(),
            box,
            spots,
            output_dtypes=[movie.dtype],
            allow_rechunk=True,
        ).compute()
        return spots
    else:
        """Assumes that identifications are in order of frames!"""
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie,
            ids["frame"].to_numpy(),
            ids["x"].to_numpy(),
            ids["y"].to_numpy(),
            box,
            spots,
        )
        return spots


def _to_photons(
    spots: lib.FloatArray3D, camera_info: dict
) -> lib.FloatArray3D:
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
    movie: lib.IntArray3D,
    identifications: pd.DataFrame,
    box: int,
    camera_info: dict,
) -> lib.FloatArray3D:
    """Extract the spots from a movie based on the identified positions
    and convert camera signal to photon counts.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.

    Returns
    -------
    spots : lib.FloatArray3D
        A 3D numpy array containing the extracted spots, with shape
        (k, box, box), where k is the number of spots identified.
    """
    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    CRLBs: lib.FloatArray2D,
    likelihoods: lib.FloatArray1D,
    iterations: lib.FloatArray1D,
    box: int,
) -> pd.DataFrame:
    """Convert the resulting localizations from the list of Futures
    into a data frame.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    theta : lib.FloatArray2D
        The fitted Gaussian parameters for each spot (x, y positions,
        photon counts, background, single-emitter image size in x and
        y).
    CRLBs : lib.FloatArray2D
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : lib.FloatArray1D
        The log-likelihoods of the fitted models.
    iterations : lib.FloatArray1D
        The number of iterations taken to converge for each spot.
    box : int
        Size of the box used for fitting. Should be an odd integer.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. The fields include
        `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`, `lpx`, `lpy`,
        `net_gradient`, `likelihood`, and `iterations`.
    """
    # box_offset = int(box / 2)
    y = theta[:, 0] + identifications["y"]  # - box_offset
    x = theta[:, 1] + identifications["x"]  # - box_offset
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])
    locs = pd.DataFrame(
        {
            "frame": identifications["frame"].astype(np.uint32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": theta[:, 2].astype(np.float32),
            "sx": theta[:, 5].astype(np.float32),
            "sy": theta[:, 4].astype(np.float32),
            "bg": theta[:, 3].astype(np.float32),
            "lpx": lpx.astype(np.float32),
            "lpy": lpy.astype(np.float32),
            "net_gradient": (
                identifications["net_gradient"].astype(np.float32)
            ),
            "likelihood": likelihoods.astype(np.float32),
            "iterations": iterations.astype(np.int32),
        }
    )
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def fit2D(
    movie: lib.IntArray3D,
    movie_info: list[dict],
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    fitting_method: Literal[
        "gausslq", "gausslq-gpu", "gaussmle", "avg"
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Fit 2D localizations to a movie, given positions of the detected
    spots (identifications).

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    movie_info : list of dicts
        Movie metadata.
    camera_info : dict
        A dictionary containing camera information: "Baseline",
        "Sensitivity", "Gain" and "Pixelsize".
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    fitting_method : {"gausslq", "gausslq-gpu", "gaussmle" or "avg"}, \
            optional
        Which 2D fitting algorithm to use. "gausslq" for least-squares
        fitting of a 2D Gaussian. "gausslq-gpu" for its GPU
        implemntation (if available). "gaussmle" for MLE 2D Gaussian
        fitting. "avg" for taking the average of each spot.
    eps : float, optional
        The convergence criterion for MLE fitting. Ignored for other
        methods. Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for MLE fitting. Ignored for
        other methods. Default is 100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for MLE fitting (impose same sigma in x and y or
        not, respectively). Default is "sigmaxy".
    multiprocess: bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
    progress_callback : callable, "console" or None, optional
        If a callable provided, it must accept one integer input (number
        of localized spots). If "console", tqdm is used to display
        progress. If None, progress is not tracked.
    abort_callback : callable or None, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted. Default is
        None.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. Returns None if
        fitting was aborted.
    new_info : dict
        New metadata.
    """
    accepted_movie_types = (io.AbstractPicassoMovie, np.memmap)
    if bitplane.IMSWRITER:
        accepted_movie_types += (
            bitplane.MovieMapper,
            bitplane.MovieMapperStack,
        )
    assert isinstance(
        movie, accepted_movie_types
    ), "movie must be a movie loaded by picasso.io.load_movie"
    assert isinstance(movie_info, list), "movie_info must be a list"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert isinstance(
        identifications, pd.DataFrame
    ), "identifications must be a DataFrame"
    assert isinstance(box, int) and box > 0, "box must be a positive integer"
    assert fitting_method in ["gausslq", "gausslq-gpu", "gaussmle", "avg"], (
        "fitting_method must be one of 'gausslq', 'gausslq-gpu',"
        " 'gaussmle', or 'avg'"
    )
    assert (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number"
    assert (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer"
    assert mle_method in [
        "sigma",
        "sigmaxy",
    ], "mle_method must be 'sigma' or 'sigmaxy'"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    if "Pixelsize" not in camera_info:
        warnings.warn(
            "Camera info in picasso.localize.fit2D does not contain "
            "'Pixelsize', i.e., effective camera pixel size in nm. "
            "Assuming 130."
        )
        camera_info["Pixelsize"] = 130

    spots = get_spots(movie, identifications, box, camera_info)
    em = camera_info["Gain"] > 1
    if fitting_method == "gausslq":
        locs = _fit2d_gausslq(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            multiprocess=multiprocess,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
        )
    elif fitting_method == "gausslq-gpu":
        if callable(progress_callback):
            progress_callback(1)
        locs = _fit2d_gausslq_gpu(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
        )
    elif fitting_method == "gaussmle":
        locs = _fit2d_gaussmle(
            spots=spots,
            identifications=identifications,
            box=box,
            eps=eps,
            max_it=max_it,
            mle_method=mle_method,
            multiprocess=multiprocess,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
        )
    elif fitting_method == "avg":
        locs = _fit2d_avg(
            spots,
            identifications,
            box,
            em,
            multiprocess,
            progress_callback,
            abort_callback,
        )
    # updated metadata
    localize_info = {
        "Generated by": f"Picasso: v{__version__} Fit 2D",
        "Fit method": fitting_method,
    }
    if fitting_method == "gaussmle":
        localize_info["Convergence criterion"] = eps
        localize_info["Max iterations"] = max_it
    new_info = localize_info | camera_info
    return locs, new_info


def _fit2d_gausslq(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> pd.DataFrame | None:
    """Fit 2D Gaussians using least-squares fitting (CPU). See ``fit_2D``
    for more details."""
    N = len(identifications)
    if multiprocess:
        fs = gausslq.fit_spots_parallel(spots, asynch=True)
        theta = _process_fitting_futures(
            fs, N, progress_callback, abort_callback
        )
        if theta is None:
            return
    else:
        theta = gausslq.fit_spots(spots, progress_callback)
    locs = gausslq.locs_from_fits(
        identifications,
        theta,
        box,
        em,
    )
    return locs


def _fit2d_gausslq_gpu(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
) -> pd.DataFrame:
    """Fit 2D Gaussians using least-squares fitting and GPU. See
    ``fit_2D`` for more details."""
    theta = gausslq.fit_spots_gpufit(spots)
    locs = gausslq.locs_from_fits_gpufit(identifications, theta, box, em)
    return locs


def _fit2d_gaussmle(
    spots,
    identifications: pd.DataFrame,
    box: int,
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> pd.DataFrame | None:
    """Fit 2D Gaussians using MLE fitting. See ``fit_2D`` for more
    details."""
    N = len(identifications)
    # MLE API is a bit different (at least for now) so we cannot use
    # _process_fitting_futures here
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(total=N, desc="Fitting", unit="spot")
    if multiprocess:
        curr, thetas, CRLBs, llhoods, iterations = gaussmle.gaussmle_async(
            spots, eps, max_it, method=mle_method
        )
        last = 0
        while curr[0] < N:
            # abort check
            if callable(abort_callback) and abort_callback():
                if use_tqdm:
                    iter_range.close()
                return

            # progress update
            if use_tqdm:
                iter_range.update(curr[0] - last)
                last = curr[0]
            elif callable(progress_callback):
                progress_callback(curr[0])
            time.sleep(0.2)
        if use_tqdm:
            iter_range.update(N - last)
            iter_range.close()
    else:
        thetas, CRLBs, llhoods, iterations = gaussmle.gaussmle(
            spots, eps, max_it, mle_method, progress_callback
        )
    locs = gaussmle.locs_from_fits(
        identifications,
        thetas,
        CRLBs,
        llhoods,
        iterations,
        box,
    )
    return locs


def _fit2d_avg(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> pd.DataFrame | None:
    """Take localizations at the average value of the spots, see
    ``fit_2D`` for more details."""
    N = len(identifications)
    if multiprocess:
        fs = avgroi.fit_spots_parallel(spots, asynch=True)
        theta = _process_fitting_futures(
            fs, N, progress_callback, abort_callback
        )
        if theta is None:
            return
    else:
        theta = avgroi.fit_spots(spots, progress_callback)
    locs = avgroi.locs_from_fits(
        identifications,
        theta,
        box,
        em,
    )
    return locs


def _process_fitting_futures(
    fs: list[Future],
    N: int,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> lib.FloatArray2D | None:
    """Convenience function for processing progress of fitting using
    multiprocessing. See ``_fit2d_gausslq``, ``_fit2d_avg``."""
    n_tasks = len(fs)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(total=N, desc="Fitting", unit="spot")

    while lib.n_futures_done(fs) < n_tasks:
        # check for abort
        if callable(abort_callback) and abort_callback():
            for f in fs:
                f.cancel()
            if use_tqdm:
                iter_range.close()
            return

        # update progress
        n_finished = round(N * lib.n_futures_done(fs) / n_tasks)
        if use_tqdm:
            iter_range.update(n_finished - iter_range.n)
        elif callable(progress_callback):
            progress_callback(n_finished)
        time.sleep(0.2)
    if use_tqdm:
        iter_range.update(N - iter_range.n)
        iter_range.close()
    theta = avgroi.fits_from_futures(fs)
    return theta


def localize(
    movie: lib.IntArray3D,
    camera_info: dict,
    parameters: dict,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    movie_info: list[dict] | None = None,
    fitting_method: Literal[
        "gausslq", "gausslq-gpu", "gaussmle", "avg"
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    threaded: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in 2D in a movie using
    the specified parameters.

    Since v0.10.0: support for frame bounds and ROI for identification +
    all fitting methods.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    parameters : dict
        A dictionary containing localization parameters, including:
        - `Min. Net Gradient`: Minimum net gradient for spot
          identification.
        - `Box Size`: Size of the box to cut out around each spot.
    threaded : bool, optional
        Whether to use multithreading/multiprocessing. Default is True.
    movie_info : list[dict], optional
        Movie metadata. If None, an empty list is used. Default is None.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. Default is None.
    fitting_method : {"gausslq", "gausslq-gpu", "gaussmle" or "avg"}, \
            optional
        Which 2D fitting algorithm to use. Default is "gausslq".
    eps : float, optional
        The convergence criterion for MLE fitting. Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for MLE fitting. Default is
        100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for MLE fitting. Default is "sigmaxy".
    identification_progress_callback : callable or "console" or None
        A callback for progress updates during identification. If
        "console", progress will be printed to the console. If None,
        progress is not reported. Default is None.
    fit_progress_callback : callable or "console" or None
        A callback for progress updates during fitting. If "console",
        progress will be printed to the console. If None, progress is
        not reported. Default is None.
    return_info : bool, optional
        Whether to return additional information about the fitting
        process. Default is True. If True, a tuple of (locs, info) is
        returned. In v0.12.0 return_info will be removed and the
        function will always return info.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list[dict], optional
        A list of dictionaries containing metadata about the movie and
        the fitting process. Only returned if `return_info` is True.
    """
    if not return_info:
        # TODO: remove in v0.12.0
        lib.deprecation_warning(
            "In version 0.12, return_info argument will be removed such "
            "that picasso.localize.localize() will always return both "
            "the localizations and the metadata dictionary."
        )

    # Use empty list as default for movie_info
    if movie_info is None:
        movie_info = []

    # Identify spots
    identifications, identify_info = identify(
        movie,
        parameters["Min. Net Gradient"],
        parameters["Box Size"],
        roi=roi,
        frame_bounds=frame_bounds,
        threaded=threaded,
        progress_callback=identification_progress_callback,
    )

    # Fit spots
    locs, fit_info = fit2D(
        movie=movie,
        movie_info=movie_info,
        camera_info=camera_info,
        identifications=identifications,
        box=parameters["Box Size"],
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        multiprocess=threaded,
        progress_callback=fit_progress_callback,
    )
    info = movie_info + [identify_info] + [fit_info]
    if return_info:
        return locs, info
    return locs


def localize_3D(
    movie: lib.IntArray3D,
    *,
    movie_info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    calibration_3d: dict,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-gpu",
        "gaussmle",
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    multiprocess: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in 3D in a movie using
    the specified parameters. First runs 2D localizations, followed
    by z position fitting assuming astigmatism, see Huang, et al.
    Science, 2008.

    Parameters
    ----------
    movie : lib.IntArray3D
        The input movie data as a 3D numpy array.
    movie_info : list of dicts
        Movie metadata.
    camera_info : dict
        A dictionary containing camera information: "Baseline",
        "Sensitivity", "Gain" and "Pixelsize".
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    minimum_ng : float
        Minimum net gradient for spot identification.
    calibration_3d : path or dict
        Either a path to a YAML file containing the calibration data or
        an already loaded calibration dictionary containing the
        following keys:

        - "X Coefficients": list of 7 floats, polynomial coefficients
            for the x-axis calibration curve;
        - "Y Coefficients": list of 7 floats, polynomial coefficients
            for the y-axis calibration curve;
        - "Magnification factor": float, magnification factor of the
            microscope, i.e., the ratio between the actual z position of
            the calibration sample and the estimated z position from the
            localization data.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. If only min or max
        is to be specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    fitting_method : {"gausslq", "gausslq-gpu", "gaussmle" or "avg"}, \
            optional
        Which 2D fitting algorithm to use. "gausslq" for least-squares
        fitting of a 2D Gaussian. "gausslq-gpu" for its GPU
        implemntation (if available). "gaussmle" for MLE 2D Gaussian
        fitting. "avg" for taking the average of each spot.
    eps : float, optional
        The convergence criterion for MLE fitting. Ignored for other
        methods. Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for MLE fitting. Ignored for
        other methods. Default is 100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for MLE fitting (impose same sigma in x and y or
        not, respectively). Default is "sigmaxy".
    multiprocess: bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
    progress_callbacks : callable, "console" or None, optional
        If a callable provided, it must accept one integer input (number
        of movie frames, or spots for identifying and fitting callbacks,
        respectively). If "console", tqdm is used to display
        progress. If None, progress is not tracked.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots in 3D.
    info : list[dict]
        A list of dictionaries containing metadata about the movie and
        the fitting processes.
    """
    assert isinstance(
        movie, (np.ndarray, io.ND2Movie)
    ), "movie must be a numpy array or ND2Movie"
    assert isinstance(movie_info, list), "movie_info must be a list"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert (
        isinstance(box, int) and box > 0 and box % 2 == 1
    ), "box must be a positive odd integer"
    assert isinstance(minimum_ng, (int, float)), "minimum_ng must be a number"
    assert isinstance(
        calibration_3d, (dict, str)
    ), "calibration_3d must be a dict or a path to a YAML file"
    assert fitting_method in [
        "gausslq",
        "gausslq-gpu",
        "gaussmle",
    ], "fitting_method must be one of 'gausslq', 'gausslq-gpu', or 'gaussmle'"
    assert (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number"
    assert (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer"
    assert mle_method in [
        "sigma",
        "sigmaxy",
    ], "mle_method must be 'sigma' or 'sigmaxy'"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    return _localize_3D(
        movie=movie,
        movie_info=movie_info,
        camera_info=camera_info,
        box=box,
        minimum_ng=minimum_ng,
        calibration_3d=calibration_3d,
        roi=roi,
        frame_bounds=frame_bounds,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        multiprocess=multiprocess,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        fit_z_progress_callback=fit_z_progress_callback,
    )


def _localize_3D(
    movie: lib.IntArray3D,
    *,
    movie_info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    calibration_3d: dict,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-gpu",
        "gaussmle",
    ] = "gausslq",
    eps: float = 0.001,
    max_it: int = 100,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    multiprocess: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Internal function for `localize_3D`, assumes validated inputs."""
    locs, info = localize(
        movie=movie,
        camera_info=camera_info,
        parameters={
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
        },
        roi=roi,
        frame_bounds=frame_bounds,
        movie_info=movie_info,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        threaded=multiprocess,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        return_info=True,  # TODO: remove in v0.12.0
    )
    fitting_method_3d = (
        "gausslq"
        if fitting_method in ["gausslq", "gausslq-gpu"]
        else "gaussmle"
    )
    locs, info = zfit.zfit(
        locs=locs,
        info=info,
        calibration=calibration_3d,
        fitting_method=fitting_method_3d,
        filter=0,
        multiprocess=multiprocess,
        progress_callback=fit_z_progress_callback,
    )
    return locs, info


def check_nena(
    locs: pd.DataFrame,
    info: None,
    callback: Callable[[int], None] = None,
) -> float:
    """Calculate the NeNA (experimental localization precision) from
    localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
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
    print("Calculating NeNA.. ", end="")
    locs = locs[0:MAX_LOCS]
    try:
        result, nena_px = postprocess.nena(locs, info, callback=callback)
    except Exception as e:
        print(e)
        nena_px = float("nan")
    print(f"{nena_px:.2f} px.")
    return nena_px


def check_kinetics(locs: pd.DataFrame, info: list[dict]) -> float:
    """Calculate the mean length of binding events from localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list of dicts
        A list of dictionaries containing metadata about the movie.

    Returns
    -------
    len_mean : float
        The mean length of binding events in frames.
    """
    print("Linking.. ", end="")
    locs = locs.iloc[0:MAX_LOCS]
    locs = postprocess.link(locs, info=info)
    len_mean = locs.len.mean()
    print(f"Mean length {len_mean:.2f} frames.")
    return len_mean


def check_drift(
    locs: pd.DataFrame,
    info: list[dict],
    callback: Callable[[int], None] = None,
) -> tuple[float, float]:
    """Estimate the drift of localizations in x and y directions.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
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
        except KeyError:
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

    nena_px = check_nena(locs, info) if nena is None else nena
    len_mean = check_kinetics(locs, info) if len_mean is None else len_mean
    drift_x, drift_y = check_drift(locs, info) if drift is None else drift

    summary["len_mean"] = len_mean
    summary["n_locs"] = len(locs)
    summary["locs_frame"] = len(locs) / summary["frames"]
    summary["drift_x"] = drift_x
    summary["drift_y"] = drift_y
    summary["nena_px"] = nena_px
    summary["nena_nm"] = nena_px * summary["pixelsize"]
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


def _save_file_summary(summary: dict) -> None:
    """Save the summary of a localization file to a SQLite database."""
    engine = create_engine("sqlite:///" + _db_filename(), echo=False)
    s = pd.Series(summary, index=summary.keys()).to_frame().T
    s.to_sql("files", con=engine, if_exists="append", index=False)


def add_file_to_db(
    file: str,
    file_hdf: str,
    drift: tuple[float, float] | None = None,
    len_mean: float | None = None,
    nena: float | None = None,
) -> None:
    """Add a localization file summary to the SQLite database."""
    summary = get_file_summary(file, file_hdf, drift, len_mean, nena)
    _save_file_summary(summary)
