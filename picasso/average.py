"""
picasso.average
~~~~~~~~~~~~~~~

Average super-resolution images of particles by alignment and rotation.

:author: Joerg Schnitzbauer, 2015
:copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Any

import functools
import multiprocessing
from multiprocessing import sharedctypes

import ctypes
import numba
import numpy as np
import pandas as pd
import scipy.sparse

from . import lib, render


@numba.jit(nopython=True, nogil=True)
def render_hist(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    t_min: float,
    t_max: float,
) -> tuple[int, lib.FloatArray2D]:
    """Calculate 2D histogram of xy coordinates.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        1D arrays of xy coordinates.
    oversampling : float
        Number of histogram pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds of the histogram.

    Returns
    -------
    n : int
        Number of localizations in the histogram.
    image : lib.FloatArray2D
        2D histogram of xy coordinates.
    """
    n_pixel = int(np.ceil(oversampling * (t_max - t_min)))
    in_view = (x > t_min) & (y > t_min) & (x < t_max) & (y < t_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - t_min)
    y = oversampling * (y - t_min)
    image = np.zeros((n_pixel, n_pixel), dtype=np.float32)
    render._fill(image, x, y)
    return len(x), image


def compute_xcorr(
    CF_image_avg: np.ndarray, image: lib.FloatArray2D
) -> lib.FloatArray2D:
    """Compute cross-correlation between two images.

    Parameters
    ----------
    CF_image_avg : np.ndarray
        Conjugate Fourier transform of the average image.
    image : lib.FloatArray2D
        Image to correlate with the average image.

    Returns
    -------
    xcorr : lib.FloatArray2D
        Cross-correlation of the two images.
    """
    F_image = np.fft.fft2(image)
    xcorr = np.fft.fftshift(np.real(np.fft.ifft2((F_image * CF_image_avg))))
    return xcorr


def align_group_core(
    index: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    angles: lib.FloatArray1D,
    oversampling: float,
    t_min: float,
    t_max: float,
    CF_image_avg: np.ndarray,
    image_half: float,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Align (shift and rotate) a single group of localizations.

    Parameters
    ----------
    index : lib.IntArray1D
        Indices of localizations belonging to this group.
    x, y : lib.FloatArray1D
        Arrays of x and y coordinates for all localizations.
    angles : lib.FloatArray1D
        Array of rotation angles to test.
    oversampling : float
        Number of display pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds for the histogram.
    CF_image_avg : np.ndarray
        Conjugate Fourier transform of the average image.
    image_half : float
        Half the size of the rendered image (in pixels).

    Returns
    -------
    x_aligned : lib.FloatArray1D
        Aligned x coordinates for this group.
    y_aligned : lib.FloatArray1D
        Aligned y coordinates for this group.
    """
    x_rot = x[index].copy()
    y_rot = y[index].copy()
    x_original = x_rot.copy()
    y_original = y_rot.copy()
    xcorr_max = 0.0
    rot = 0.0
    dy = 0.0
    dx = 0.0

    for angle in angles:
        # rotate locs
        x_rot = np.cos(angle) * x_original - np.sin(angle) * y_original
        y_rot = np.sin(angle) * x_original + np.cos(angle) * y_original
        # render group image
        N, image = render_hist(x_rot, y_rot, oversampling, t_min, t_max)
        # calculate cross-correlation
        xcorr = compute_xcorr(CF_image_avg, image)
        # find the brightest pixel
        y_max, x_max = np.unravel_index(xcorr.argmax(), xcorr.shape)
        # store the transformation if the correlation is larger than before
        if xcorr[y_max, x_max] > xcorr_max:
            xcorr_max = xcorr[y_max, x_max]
            rot = angle
            dy = np.ceil(y_max - image_half) / oversampling
            dx = np.ceil(x_max - image_half) / oversampling

    # rotate and shift
    x_aligned = np.cos(rot) * x_original - np.sin(rot) * y_original - dx
    y_aligned = np.sin(rot) * x_original + np.cos(rot) * y_original - dy

    return x_aligned, y_aligned


def _init_pool_worker(
    x_: ctypes.Array[ctypes.c_float],
    y_: ctypes.Array[ctypes.c_float],
    group_index_: scipy.sparse.lil_matrix,
) -> None:
    """Initialize pool process variables.

    Parameters
    ----------
    x_ : ctypes.Array
        Shared x coordinates array.
    y_ : ctypes.Array
        Shared y coordinates array.
    group_index_ : scipy.sparse.lil_matrix
        Sparse matrix indexing groups.
    """
    global x, y, group_index
    x = np.ctypeslib.as_array(x_)
    y = np.ctypeslib.as_array(y_)
    group_index = group_index_


def _align_group_worker(
    angles: lib.FloatArray1D,
    oversampling: float,
    t_min: float,
    t_max: float,
    CF_image_avg: np.ndarray,
    image_half: float,
    counter: multiprocessing.managers.ValueProxy,
    lock: multiprocessing.managers.AcquirerProxy,
    group: int,
) -> None:
    """Worker function for aligning a single group in a process pool.

    Parameters
    ----------
    angles : lib.FloatArray1D
        Array of rotation angles.
    oversampling : float
        Number of display pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds for the histogram.
    CF_image_avg : np.ndarray
        Conjugate Fourier transform of the average image.
    image_half : float
        Half the size of the rendered image.
    counter : multiprocessing.managers.ValueProxy
        Shared counter for processed groups.
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing counter access.
    group : int
        Index of the group to align.
    """
    with lock:
        counter.value += 1

    index = group_index[group].nonzero()[1]
    x_aligned, y_aligned = align_group_core(
        index,
        x,
        y,
        angles,
        oversampling,
        t_min,
        t_max,
        CF_image_avg,
        image_half,
    )

    # Update global arrays
    x[index] = x_aligned
    y[index] = y_aligned


def average(
    locs: pd.DataFrame,
    info: list[dict],
    group_index: scipy.sparse.lil_matrix,
    oversampling: float,
    iterations: int = 3,
    progress_callback: callable | None = None,
) -> pd.DataFrame:
    """Average super-resolution images of particles by alignment and rotation.

    Iteratively aligns and rotates particle images by searching for the
    rotation angle and shift that maximizes cross-correlation with the
    average image.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with group indices (``group`` column).
    info : list[dict]
        Metadata for localizations.
    group_index : scipy.sparse.lil_matrix
        Sparse matrix where rows are groups and columns are localization
        indices. Element (i, j) is True if localization j belongs to group i.
    oversampling : float
        Number of display pixels per camera pixel.
    iterations : int, optional
        Number of averaging iterations (default=3).
    progress_callback : callable, optional
        Callback function called with progress info after each iteration.
        Signature: callback(iteration, total_iterations, locs).

    Returns
    -------
    locs_averaged : pd.DataFrame
        Averaged localizations with coordinates centered around origin.
    """
    locs = locs.copy()
    n_groups = group_index.shape[0]
    r = 2 * np.sqrt((locs["x"] ** 2 + locs["y"] ** 2).mean())
    t_min = -r
    t_max = r

    # Calculate angle step for rotation search
    a_step = np.arcsin(1 / (oversampling * r))
    angles = np.arange(0, 2 * np.pi, a_step)

    # Setup multiprocessing
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
    manager = multiprocessing.Manager()
    counter = manager.Value("d", 0)
    lock = manager.Lock()
    groups_per_worker = max(1, int(n_groups / n_workers))

    # Initialize shared arrays
    x = sharedctypes.RawArray("f", locs["x"].to_numpy())
    y = sharedctypes.RawArray("f", locs["y"].to_numpy())

    pool = multiprocessing.Pool(
        n_workers,
        _init_pool_worker,
        (x, y, group_index),
    )

    try:
        for it in range(iterations):
            counter.value = 0

            # Render average image
            N_avg, image_avg = render_hist(
                np.ctypeslib.as_array(x),
                np.ctypeslib.as_array(y),
                oversampling,
                t_min,
                t_max,
            )
            n_pixel, _ = image_avg.shape
            image_half = n_pixel / 2
            CF_image_avg = np.conj(np.fft.fft2(image_avg))

            # Align all groups
            fc = functools.partial(
                _align_group_worker,
                angles,
                oversampling,
                t_min,
                t_max,
                CF_image_avg,
                image_half,
                counter,
                lock,
            )
            result = pool.map_async(fc, range(n_groups), groups_per_worker)

            # Wait for completion and report progress
            while not result.ready():
                if progress_callback:
                    locs_current = locs.copy()
                    locs_current["x"] = np.ctypeslib.as_array(x)
                    locs_current["y"] = np.ctypeslib.as_array(y)
                    progress_callback(it + 1, iterations, locs_current)

            # Update localizations from shared arrays
            locs["x"] = np.ctypeslib.as_array(x)
            locs["y"] = np.ctypeslib.as_array(y)
            locs["x"] -= np.mean(locs["x"])
            locs["y"] -= np.mean(locs["y"])

            if progress_callback:
                progress_callback(it + 1, iterations, locs)

    finally:
        pool.close()
        pool.join()

    return locs
