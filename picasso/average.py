"""
picasso.average
~~~~~~~~~~~~~~~

Average super-resolution images of particles by alignment and rotation.

:author: Joerg Schnitzbauer, 2015
:copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import functools
import multiprocessing
from multiprocessing import sharedctypes
from typing import Callable, Literal

import ctypes
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

from . import lib, render, __version__


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
        N, image = render.render_hist_numba(
            x_rot, y_rot, oversampling, t_min, t_max
        )
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


def build_group_index(
    locs: pd.DataFrame,
) -> scipy.sparse.lil_matrix:
    """Build a sparse boolean group index from localization group labels.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with a ``group`` column.

    Returns
    -------
    group_index : scipy.sparse.lil_matrix
        Boolean sparse matrix of shape ``(n_groups, n_locs)`` where
        element ``(i, j)`` is ``True`` if localization ``j`` belongs to
        group ``i``.
    """
    groups = np.unique(locs["group"])
    n_groups = len(groups)
    n_locs = len(locs)
    group_index = scipy.sparse.lil_matrix((n_groups, n_locs), dtype=bool)
    for i, group in enumerate(groups):
        index = np.where(locs["group"] == group)[0]
        group_index[i, index] = True
    return group_index


def com_align(
    locs: pd.DataFrame,
    group_index: scipy.sparse.lil_matrix,
) -> pd.DataFrame:
    """Center each group of localizations around the origin (COM alignment).

    For each group, subtracts the mean x and y coordinates from all
    localizations belonging to that group.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x``, ``y`` columns.
    group_index : scipy.sparse.lil_matrix
        Sparse group index as returned by :func:`build_group_index`.

    Returns
    -------
    locs : pd.DataFrame
        Copy of ``locs`` with per-group COM alignment applied.
    """
    locs = locs.copy()
    n_groups = group_index.shape[0]
    for i in range(n_groups):
        index = group_index[i, :].nonzero()[1]
        locs.loc[index, "x"] -= np.mean(locs.loc[index, "x"])
        locs.loc[index, "y"] -= np.mean(locs.loc[index, "y"])
    return locs


def prepare_locs_for_save(
    locs: pd.DataFrame,
    info: list[dict],
    params: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Shift localizations and update metadata for saving.

    Parameters
    ----------
    locs : pd.DataFrame
        Averaged localizations.
    info : list of dicts
        Original metadata.
    params : dict
        Dictionary with parameters used for averaging.

    Returns
    -------
    locs : pd.DataFrame
        Localizations shifted to positive coordinates.
    new_info : list of dicts
        Updated metadata with new width and height.
    """
    cx = lib.get_from_metadata(info, "Width") / 2
    cy = lib.get_from_metadata(info, "Height") / 2
    locs["x"] += cx
    locs["y"] += cy
    avg_info = {
        "Generated by": f"Picasso {__version__} Average",
        "Display pixel size (nm)": params["disp_px_size"],
        "Iterations": params["it"],
    }
    new_info = info + [avg_info]
    return locs, new_info


def average(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    display_pixel_size: float = 5.0,
    iterations: int = 3,
    return_shifted_locs: bool = False,
    progress_callback: callable | Literal["console"] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]] | None:
    """Average super-resolution images of particles by alignment and rotation.

    Builds the group index and applies per-group center-of-mass alignment
    internally before iteratively aligning and rotating particle images by
    searching for the rotation angle and shift that maximizes
    cross-correlation with the average image.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with a ``group`` column.
    info : list[dict]
        Metadata for localizations.
    display_pixel_size : float, optional
        Display pixel size in nm used in averaging (default=5.0).
    iterations : int, optional
        Number of averaging iterations (default=3).
    return_shifted_locs : bool, optional
        If True, return localizations shifted to positive coordinates
        and updated metadata. If False, only localizations are returned
        without shifts.
    progress_callback : callable or "console" or None, optional
        Controls progress reporting. Pass a callable with signature
        ``callback(iteration, total_iterations, locs)`` to receive
        per-iteration updates. Pass ``"console"`` to display tqdm
        progress bars in the terminal. Pass ``None`` (default) for no
        progress reporting.
    abort_callback : callable or None, optional
        Callable with no arguments returning a bool. If it returns
        True, averaging is aborted, the worker pool is terminated, and
        ``None`` is returned. Default is None (no abort).

    Returns
    -------
    locs_averaged : pd.DataFrame or None
        Averaged localizations with coordinates centered at (0, 0).
        Returns ``None`` if the process was aborted via
        ``abort_callback``. If ``return_shifted_locs`` is True, the
        localizations are shifted towards the center of the FOV.
    info : list of dicts, optional
        If ``return_shifted_locs`` is True, updated metadata with the
        average info is returned.
    """
    assert (
        "group" in locs.columns
    ), "Localizations DataFrame must have a 'group' column."
    group_index = build_group_index(locs)
    locs = com_align(locs, group_index)
    n_groups = group_index.shape[0]
    r = 2 * np.sqrt((locs["x"] ** 2 + locs["y"] ** 2).mean())
    t_min = -r
    t_max = r

    # Calculate angle step for rotation search
    camera_pixelsize = lib.get_from_metadata(
        info, "Pixelsize", raise_error=True
    )
    oversampling = camera_pixelsize / display_pixel_size
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

    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_pbar = tqdm(total=iterations, desc="Averaging", unit="iter")
        group_pbar = tqdm(total=n_groups, desc="Groups", unit="group")
    else:
        iter_pbar = None
        group_pbar = None

    aborted = False
    try:
        for it in range(iterations):
            if callable(abort_callback) and abort_callback():
                aborted = True
                break
            counter.value = 0
            if use_tqdm:
                group_pbar.reset()
                group_pbar.set_description(f"Iteration {it + 1}/{iterations}")

            # Render average image
            N_avg, image_avg = render.render_hist_numba(
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
            if use_tqdm:
                last_count = 0
                while not result.ready():
                    if callable(abort_callback) and abort_callback():
                        aborted = True
                        break
                    current = int(counter.value)
                    group_pbar.update(current - last_count)
                    last_count = current
                if aborted:
                    break
                group_pbar.update(n_groups - last_count)
            else:
                while not result.ready():
                    if callable(abort_callback) and abort_callback():
                        aborted = True
                        break
                    if callable(progress_callback):
                        locs_current = locs.copy()
                        locs_current["x"] = np.ctypeslib.as_array(x)
                        locs_current["y"] = np.ctypeslib.as_array(y)
                        progress_callback(
                            it + 1,
                            iterations,
                            locs_current,
                            int(counter.value),
                            n_groups,
                        )
                if aborted:
                    break

            # Update localizations from shared arrays
            locs["x"] = np.ctypeslib.as_array(x)
            locs["y"] = np.ctypeslib.as_array(y)
            locs["x"] -= np.mean(locs["x"])
            locs["y"] -= np.mean(locs["y"])

            if use_tqdm:
                iter_pbar.update(1)
            if callable(progress_callback):
                progress_callback(it + 1, iterations, locs, n_groups, n_groups)

    finally:
        if use_tqdm:
            group_pbar.close()
            iter_pbar.close()
        if aborted:
            pool.terminate()
        else:
            pool.close()
        pool.join()

    if aborted:
        return None

    if return_shifted_locs:
        params = {"disp_px_size": display_pixel_size, "it": iterations}
        locs, info = prepare_locs_for_save(locs, info, params)
        return locs, info
    else:
        return locs
