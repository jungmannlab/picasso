"""
picasso.average
~~~~~~~~~~~~~~~

Average super-resolution images of particles by alignment and rotation.

:authors: Joerg Schnitzbauer
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
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

from . import lib, render

__all__ = ["average", "AveragePool"]


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
        Rendered pixels per nm, i.e., 1 / display pixel size.
    t_min, t_max : float
        Minimum and maximum bounds for the histogram (nm).
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
        N, image = render.render_hist(x_rot, y_rot, oversampling, t_min, t_max)
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
        Rendered pixels per nm, i.e., 1 / display pixel size.
    t_min, t_max : float
        Minimum and maximum bounds for the histogram (nm).
    CF_image_avg : np.ndarray
        Conjugate Fourier transform of the average image.
    image_half : float
        Half the size of the rendered image.
    counter : multiprocessing.managers.ValueProxy
        Shared counter of the groups aligned so far, incremented once the
        group's coordinates are written.
    lock : multiprocessing.managers.AcquirerProxy
        Lock for synchronizing counter access.
    group : int
        Index of the group to align.
    """
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

    with lock:
        counter.value += 1


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
    params: dict = {},
) -> tuple[pd.DataFrame, list[dict]]:
    """Shift localizations and update metadata for saving.

    Parameters
    ----------
    locs : pd.DataFrame
        Averaged localizations.
    info : list of dicts
        Original metadata.
    params : dict, optional
        Dictionary with parameters used for averaging. Accepts
        `disp_px_size` and `it` (iterations).

    Returns
    -------
    locs : pd.DataFrame
        Copy of the localizations, shifted to positive coordinates.
    new_info : list of dicts
        ``info`` with an entry recording the averaging appended, see
        ``lib.append_metadata``.
    """
    width, height = lib.field_of_view(info)
    cx = width / 2
    cy = height / 2
    # shift a copy: the caller's localizations (e.g., the average shown
    # in Picasso: Average) stay centered at (0, 0)
    locs = locs.copy()
    locs["x"] += cx
    locs["y"] += cy
    return locs, lib.append_metadata(info, _average_entry(params))


def _average_entry(params: dict) -> dict:
    """Metadata entry recording the averaging with ``params`` (see
    ``prepare_locs_for_save``)."""
    entry = lib.step_info("Average")
    if "disp_px_size" in params:
        entry["Display pixel size (nm)"] = params["disp_px_size"]
    if "it" in params:
        entry["Iterations"] = params["it"]
    return entry


class AveragePool:
    """Reusable pool of worker processes for particle averaging.

    Starting a process pool is expensive: every worker process imports
    ``picasso`` and receives a copy of the group index. Keeping one pool
    alive across several calls to :func:`average` (e.g. in ``Picasso:
    Average``, where the same file is usually averaged repeatedly with
    different settings) avoids paying that cost at every run.

    The pool is bound to the localizations it was created with. The
    group index and the shared coordinate arrays are set up once; the
    coordinates are overwritten at the beginning of each averaging run,
    so the pool stays valid after a previous run has shifted the
    localizations. It must be recreated if a different file (or a
    different number of localizations or groups) is to be averaged.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with a ``group`` column.
    n_workers : int, optional
        Number of worker processes. Default is ``lib.n_workers()``.

    Attributes
    ----------
    group_index : scipy.sparse.lil_matrix
        Sparse matrix indexing groups.
    n_groups : int
        Number of groups.
    n_locs : int
        Number of localizations the pool was created with.
    n_workers : int
        Number of worker processes.
    groups_per_worker : int
        Chunk size used when mapping groups onto the workers.
    x, y : ctypes.Array
        Shared coordinate arrays that the workers read from and write
        into.
    counter : multiprocessing.managers.ValueProxy
        Shared counter for the number of groups aligned in the current
        iteration.
    lock : multiprocessing.managers.AcquirerProxy
        Lock synchronizing access to ``counter``.
    pool : multiprocessing.Pool
        The worker pool itself.
    closed : bool
        True if the pool has been shut down and can no longer be used.
    """

    def __init__(
        self, locs: pd.DataFrame, n_workers: int | None = None
    ) -> None:
        assert (
            "group" in locs.columns
        ), "Localizations DataFrame must have a 'group' column."
        self.group_index = build_group_index(locs)
        self.n_groups = self.group_index.shape[0]
        self.n_locs = len(locs)
        self.n_workers = lib.n_workers() if n_workers is None else n_workers
        self.groups_per_worker = max(1, int(self.n_groups / self.n_workers))
        self.x = sharedctypes.RawArray("f", locs["x"].to_numpy())
        self.y = sharedctypes.RawArray("f", locs["y"].to_numpy())
        self.manager = multiprocessing.Manager()
        self.counter = self.manager.Value("d", 0)
        self.lock = self.manager.Lock()
        self.pool = multiprocessing.Pool(
            self.n_workers,
            _init_pool_worker,
            (self.x, self.y, self.group_index),
        )
        self.closed = False

    def set_coordinates(self, locs: pd.DataFrame) -> None:
        """Copy localization coordinates into the shared arrays.

        Parameters
        ----------
        locs : pd.DataFrame
            Localizations with the same number of rows as the ones the
            pool was created with.

        Raises
        ------
        ValueError
            If ``locs`` does not match the localizations the pool was
            created with.
        """
        if len(locs) != self.n_locs:
            raise ValueError(
                "The given localizations do not match the ones this pool"
                f" was created with ({len(locs)} instead of {self.n_locs}"
                " localizations). Please create a new AveragePool."
            )
        np.ctypeslib.as_array(self.x)[:] = locs["x"].to_numpy()
        np.ctypeslib.as_array(self.y)[:] = locs["y"].to_numpy()

    def close(self) -> None:
        """Let the workers finish and shut the pool down."""
        self._shutdown(terminate=False)

    def terminate(self) -> None:
        """Kill the workers immediately and shut the pool down."""
        self._shutdown(terminate=True)

    def _shutdown(self, terminate: bool) -> None:
        if self.closed:
            return
        self.closed = True
        if terminate:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()
        self.manager.shutdown()

    def __enter__(self) -> "AveragePool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()


def _finalize_average(
    locs,
    info,
    aborted,
    return_shifted_locs,
    display_pixel_size,
    iterations,
):
    if aborted:
        return None
    params = {"disp_px_size": display_pixel_size, "it": iterations}
    if return_shifted_locs:
        return prepare_locs_for_save(locs, info, params)
    return locs, lib.append_metadata(info, _average_entry(params))


#: Seconds between two polls of the workers while a round of alignment
#: runs.
_POLL_INTERVAL = 0.05


def _wait_for_alignment(
    result: multiprocessing.pool.AsyncResult,
    abort_callback: Callable[[], bool] | None,
    counter: multiprocessing.managers.ValueProxy,
    n_groups: int,
    progress: lib.Progress,
) -> bool:
    """Wait for one round of group alignment to finish.

    Polls every ``_POLL_INTERVAL`` seconds, advancing ``progress`` (a
    normalized tracker whose phase the caller started) to the number of
    groups aligned, and checking ``abort_callback``.

    Returns
    -------
    aborted : bool
        True if ``abort_callback`` asked to stop before the round
        finished.
    """
    while not result.ready():
        if callable(abort_callback) and abort_callback():
            return True
        progress.set_value(int(counter.value))
        result.wait(_POLL_INTERVAL)
    result.get()  # re-raise a worker exception
    progress.set_value(n_groups)
    return False


def average(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    display_pixel_size: float = 5.0,
    iterations: int = 3,
    return_shifted_locs: bool = False,
    progress: lib.Progress | Literal["console"] | None = None,
    iteration_callback: Callable[[int, pd.DataFrame], None] | None = None,
    abort_callback: Callable[[], bool] | None = None,
    pool: AveragePool | None = None,
) -> tuple[pd.DataFrame, list[dict]] | None:
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
        If True, the returned localizations are shifted from (0, 0) to
        the center of the field of view, so that they can be saved.
        Default is False.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports one phase per iteration,
        "Averaging (iteration k/N)" over the groups aligned. Default is
        None.
    iteration_callback : callable or None, optional
        Data hook, not progress reporting: called after each finished
        iteration as ``iteration_callback(iterations_done, locs)``, with
        the number of iterations finished so far and the localizations
        as they stand after it (centered at (0, 0)), e.g. for a live
        preview of the average. ``locs`` is the working DataFrame of the
        averaging, which the next iteration overwrites; copy it to keep
        it, and do not modify it. Called from the thread that called
        this function. Default is None.
    abort_callback : callable or None, optional
        Callable with no arguments returning a bool. If it returns
        True, averaging is aborted, the worker pool is terminated, and
        ``None`` is returned. Default is None (no abort).
    pool : AveragePool or None, optional
        Pool of worker processes to average with, see
        :class:`AveragePool`. Pass one to reuse the same workers across
        several calls, which avoids restarting them every time. It must
        have been created with the same localizations. If None
        (default), a pool is started and shut down internally. Note that
        an externally provided pool is terminated (and thus rendered
        unusable) if averaging is aborted.

    Returns
    -------
    locs_averaged : pd.DataFrame
        Averaged localizations with coordinates centered at (0, 0), or
        at the center of the field of view if ``return_shifted_locs`` is
        True.
    info : list of dicts
        Metadata of the averaged localizations: ``info`` with an entry
        recording the averaging appended, see ``lib.append_metadata``.

    None is returned instead of the tuple if the averaging was aborted
    via ``abort_callback``.
    """
    assert (
        "group" in locs.columns
    ), "Localizations DataFrame must have a 'group' column."
    own_pool = pool is None
    if own_pool:
        pool = AveragePool(locs)
    elif pool.closed:
        raise ValueError(
            "The given AveragePool has been shut down. Please create a"
            " new one."
        )
    group_index = pool.group_index
    locs = com_align(locs, group_index)
    n_groups = pool.n_groups
    r = 2 * np.sqrt((locs["x"] ** 2 + locs["y"] ** 2).mean())
    t_min = -r
    t_max = r

    # Calculate angle step for rotation search: one rendered pixel at
    # the radius r (a quarter turn for particles smaller than a pixel)
    oversampling = 1 / display_pixel_size  # rendered pixels per nm
    a_step = np.arcsin(min(1.0, 1 / (oversampling * r)))
    angles = np.arange(0, 2 * np.pi, a_step)

    # Setup multiprocessing; the shared arrays are (re-)filled with the
    # current coordinates, so that a reused pool starts from them
    counter = pool.counter
    lock = pool.lock
    x = pool.x
    y = pool.y
    pool.set_coordinates(locs)

    # not a ``with lib.tracked_progress`` block, which would re-indent the
    # whole loop; the same close-only-if-created semantics
    progress_arg = progress
    progress = lib.normalize_progress(progress, "Averaging", unit="group")
    aborted = False
    result = None
    try:
        for it in range(iterations):
            if callable(abort_callback) and abort_callback():
                aborted = True
                break
            counter.value = 0
            lib.start_phase(
                progress,
                f"Averaging (iteration {it + 1}/{iterations})",
                n_groups,
                unit="group",
            )
            progress.set_value(0)

            # Render average image
            N_avg, image_avg = render.render_hist(
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
            result = pool.pool.map_async(
                fc, range(n_groups), pool.groups_per_worker
            )

            # Wait for completion and report progress
            aborted = _wait_for_alignment(
                result, abort_callback, counter, n_groups, progress
            )
            if aborted:
                break

            # Update localizations from shared arrays
            locs["x"] = np.ctypeslib.as_array(x)
            locs["y"] = np.ctypeslib.as_array(y)
            locs["x"] -= np.mean(locs["x"])
            locs["y"] -= np.mean(locs["y"])

            if iteration_callback is not None:
                iteration_callback(it + 1, locs)

    finally:
        if progress is not progress_arg:
            progress.close()
        # an aborted (or canceled, or failed) run leaves work queued in
        # the pool, so it is terminated even if it was provided by the
        # caller
        if aborted or (result is not None and not result.ready()):
            pool.terminate()
        elif own_pool:
            pool.close()

    return _finalize_average(
        locs,
        info,
        aborted,
        return_shifted_locs,
        display_pixel_size,
        iterations,
    )
