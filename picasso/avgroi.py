"""
picasso.avgroi
~~~~~~~~~~~~~~

Fits spots, i.e., finds the average of the pixels in a region of
interest (ROI).

:authors: Maximilian Thomas Strauss
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from concurrent import futures
from typing import Literal

import numba
import numpy as np
import pandas as pd

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
    progress: lib.Progress | Literal["console"] | None = None,
) -> lib.FloatArray2D:
    """Fit several spots, one after another.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports one phase, "Fitting spots", over
        the spots. Default is None.

    Returns
    -------
    theta : lib.FloatArray2D
        ``(n_spots, 6)`` fit parameters, each row as :func:`fit_spot`.
    """
    n_spots = len(spots)
    theta = np.empty((n_spots, 6), dtype=np.float32)
    theta.fill(np.nan)
    with lib.tracked_progress(progress, "Fitting spots", "spot") as progress:
        lib.start_phase(progress, "Fitting spots", n_spots, unit="spot")
        for i in range(n_spots):
            theta[i] = fit_spot(spots[i])
            progress.set_value(i + 1)
        progress.set_value(n_spots)
    return theta


def fit_spots_parallel(
    spots: lib.FloatArray3D,
    asynch: bool = False,
    progress: lib.Progress | Literal["console"] | None = None,
) -> lib.FloatArray2D | list[futures.Future]:
    """Fit spots in a pool of worker processes.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts.
    asynch : bool, optional
        If True, return the pending futures immediately instead of waiting.
        Default False.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports one phase, "Fitting spots", over
        the spots. Unused if ``asynch``. Default is None.

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
    try:
        with lib.tracked_progress(
            progress, "Fitting spots", "spot"
        ) as progress:
            lib.start_phase(progress, "Fitting spots", n_spots, unit="spot")
            done = 0
            sizes = dict(zip(fs, spots_per_task))
            for f in futures.as_completed(fs):
                done += sizes[f]
                progress.set_value(done)
            progress.set_value(n_spots)
    except BaseException:
        for f in fs:
            f.cancel()
        raise
    finally:
        executor.shutdown(wait=False)
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
    *,
    pixelsize: float,
) -> pd.DataFrame:
    """Convert fit results into a data frame of localizations (lengths
    in nm).

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
    pixelsize : float
        Camera pixel size in nm; the positions, widths and precisions are
        converted to nm with it.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, sorted by frame.
    """
    pixelsize = float(pixelsize)
    x = (theta[:, 0] + identifications["x"]) * pixelsize
    y = (theta[:, 1] + identifications["y"]) * pixelsize
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
                "sx": (theta[:, 4] * pixelsize).astype(np.float32),
                "sy": (theta[:, 5] * pixelsize).astype(np.float32),
                "bg": theta[:, 3].astype(np.float32),
                "lpx": (lpx * pixelsize).astype(np.float32),
                "lpy": (lpy * pixelsize).astype(np.float32),
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
                "sx": (theta[:, 4] * pixelsize).astype(np.float32),
                "sy": (theta[:, 5] * pixelsize).astype(np.float32),
                "bg": theta[:, 3].astype(np.float32),
                "lpx": (lpx * pixelsize).astype(np.float32),
                "lpy": (lpy * pixelsize).astype(np.float32),
                "ellipticity": ellipticity.astype(np.float32),
                **lib.net_gradient_column(identifications),
            }
        )
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs
