"""
picasso.avgroi
~~~~~~~~~~~~~~

Fits spots, i.e., finds the average of the pixels in a region of
interest (ROI).

:authors: Maximilian Thomas Strauss
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from concurrent import futures
from typing import Callable, Literal

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import lib
from .fitting import precision


@numba.jit(nopython=True, nogil=True)
def _sum(spot: lib.FloatArray2D, size: int) -> float:
    """Calculate the sum of all pixels in a spot."""
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            _sum_ += spot[i, j]

    return _sum_


def fit_spot(spot: lib.FloatArray2D) -> list[float]:
    """Fit a single spot, i.e. average its pixels.

    Parameters
    ----------
    spot : lib.FloatArray2D
        ``(box, box)`` photon counts of one spot.

    Returns
    -------
    theta : list of float
        ``[x, y, photons, bg, sx, sy]``, where ``x``/``y`` are 0 (the ROI is
        not localized), ``photons`` and ``bg`` are both the ROI sum, and the
        widths are 1.
    """
    size = spot.shape[0]
    avg_roi = _sum(spot, size)
    # result is [x, y, photons, bg, sx, sy]
    result = [0, 0, avg_roi, avg_roi, 1, 1]
    return result


def fit_spots(
    spots: lib.FloatArray3D,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> lib.FloatArray2D:
    """Fit several spots, one after another.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts.
    progress_callback : callable, "console" or None, optional
        ``"console"`` shows a tqdm bar; a callable is invoked with the index
        of the spot just fitted. Default None.

    Returns
    -------
    theta : lib.FloatArray2D
        ``(n_spots, 6)`` fit parameters, each row as :func:`fit_spot`.
    """
    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(range(len(spots)), desc="Fitting...", unit="spot")
    else:
        iter_range = range(len(spots))
    for i in iter_range:
        spot = spots[i]
        theta[i] = fit_spot(spot)
        if callable(progress_callback):
            progress_callback(i)
    return theta


def fit_spots_parallel(
    spots: lib.FloatArray3D,
    asynch: bool = False,
) -> lib.FloatArray2D | list[futures.Future]:
    """Fit spots in a pool of worker processes.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts.
    asynch : bool, optional
        If True, return the pending futures immediately instead of waiting.
        Default False.

    Returns
    -------
    theta : lib.FloatArray2D or list of futures.Future
        ``(n_spots, 6)`` fit parameters, or the list of futures if
        ``asynch``; pass those to :func:`fits_from_futures`.
    """
    n_workers = lib.n_workers()
    n_spots = len(spots)
    n_tasks = 100 * n_workers
    spots_per_task = [
        (
            int(n_spots / n_tasks + 1)
            if _ < n_spots % n_tasks
            else int(n_spots / n_tasks)
        )
        for _ in range(n_tasks)
    ]
    start_indices = np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = futures.ProcessPoolExecutor(n_workers)
    for i, n_spots_task in zip(start_indices, spots_per_task):
        fs.append(executor.submit(fit_spots, spots[i : i + n_spots_task]))
    if asynch:
        return fs
    with tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return fits_from_futures(fs)


def fits_from_futures(futures: list[futures.Future]) -> lib.FloatArray2D:
    """Collect fit results from futures.

    Parameters
    ----------
    futures : list of futures.Future
        The futures returned by :func:`fit_spots_parallel` with
        ``asynch=True``.

    Returns
    -------
    theta : lib.FloatArray2D
        ``(n_spots, 6)`` fit parameters of every task, stacked in order.
    """
    theta = [_.result() for _ in futures]
    return np.vstack(theta)


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: float,
    readout_variance: lib.FloatArray1D | float = 0.0,
) -> pd.DataFrame:
    """Convert fit results into a data frame of localizations.

    Parameters
    ----------
    identifications : pd.DataFrame
        The identifications the spots were cut from.
    theta : lib.FloatArray2D
        ``(n_spots, 6)`` fit parameters from :func:`fit_spots`.
    box : int
        Box side length (camera pixels).
    em : float
        Whether EMCCD was used, which doubles the variance.
    readout_variance : lib.FloatArray1D or float, optional
        Mean sCMOS readout variance over each spot's box, in photoelectrons
        squared; it adds to the background term of the closed-form precision.
        See ``picasso.fitting.precision``. Default 0.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, sorted by frame.
    """
    x = theta[:, 0] + identifications["x"]
    y = theta[:, 1] + identifications["y"]
    lpx = precision.localization_precision(
        theta[:, 2],
        theta[:, 4],
        theta[:, 5],
        theta[:, 3],
        em=em,
        readout_variance=readout_variance,
    )
    lpy = precision.localization_precision(
        theta[:, 2],
        theta[:, 5],
        theta[:, 4],
        theta[:, 3],
        em=em,
        readout_variance=readout_variance,
    )
    a = np.maximum(theta[:, 4], theta[:, 5])
    b = np.minimum(theta[:, 4], theta[:, 5])
    ellipticity = (a - b) / a
    if "n_id" in identifications.columns:
        locs = pd.DataFrame(
            {
                "frame": identifications["frame"].to_numpy().astype(np.uint32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "photons": theta[:, 2].astype(np.float32),
                "sx": theta[:, 4].astype(np.float32),
                "sy": theta[:, 5].astype(np.float32),
                "bg": theta[:, 3].astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
                "ellipticity": ellipticity.astype(np.float32),
                **lib.net_gradient_column(identifications),
                "n_id": identifications["n_id"].to_numpy().astype(np.uint32),
            }
        )
        locs.sort_values(by="n_id", kind="quicksort", inplace=True)
    else:
        locs = pd.DataFrame(
            {
                "frame": identifications["frame"].to_numpy().astype(np.uint32),
                "x": x.astype(np.float32),
                "y": y.astype(np.float32),
                "photons": theta[:, 2].astype(np.float32),
                "sx": theta[:, 4].astype(np.float32),
                "sy": theta[:, 5].astype(np.float32),
                "bg": theta[:, 3].astype(np.float32),
                "lpx": lpx.astype(np.float32),
                "lpy": lpy.astype(np.float32),
                "ellipticity": ellipticity.astype(np.float32),
                **lib.net_gradient_column(identifications),
            }
        )
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs
