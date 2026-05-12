"""
picasso.avgroi
~~~~~~~~~~~~~~

Fits spots, i.e., finds the average of the pixels in a region of
interest (ROI).

:authors: Maximilian Thomas Strauss
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

import multiprocessing
from concurrent import futures
from typing import Callable, Literal

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import gausslq, lib


@numba.jit(nopython=True, nogil=True)
def _sum(spot: lib.FloatArray2D, size: int) -> float:
    """Calculate the sum of all pixels in a spot."""
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            _sum_ += spot[i, j]

    return _sum_


def fit_spot(spot: lib.FloatArray2D) -> list[float]:
    """Fit a single spot and return fit parameters."""
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
    """Fit spots and return fit parameters."""
    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(len(spots), desc="Fitting...", unit="spot")
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
    """Fit spots in parallel (if ``asynch`` is True)."""
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
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
    """Collect fit results from futures."""
    theta = [_.result() for _ in futures]
    return np.vstack(theta)


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: float,
) -> pd.DataFrame:
    """Convert fit results to localization DataFrame."""
    x = theta[:, 0] + identifications["x"]
    y = theta[:, 1] + identifications["y"]
    lpx = gausslq.localization_precision(
        theta[:, 2], theta[:, 4], theta[:, 5], theta[:, 3], em=em
    )
    lpy = gausslq.localization_precision(
        theta[:, 2], theta[:, 5], theta[:, 4], theta[:, 3], em=em
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
                "net_gradient": (
                    identifications["net_gradient"]
                    .to_numpy()
                    .astype(np.float32)
                ),
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
                "net_gradient": (
                    identifications["net_gradient"]
                    .to_numpy()
                    .astype(np.float32)
                ),
            }
        )
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs
