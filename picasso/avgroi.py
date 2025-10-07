"""
    picasso.avgroi
    ~~~~~~~~~~~~~~

    Fits spots, i.e., finds the average of the pixels in a region of
    interest (ROI).

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

import multiprocessing
from concurrent import futures

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import postprocess


@numba.jit(nopython=True, nogil=True)
def _sum(spot: np.ndarray, size: int) -> float:
    """Calculate the sum of all pixels in a spot."""
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            _sum_ += spot[i, j]

    return _sum_


def fit_spot(spot: np.ndarray) -> np.ndarray:
    """Fit a single spot and return fit parameters."""
    size = spot.shape[0]
    avg_roi = _sum(spot, size)
    # result is [x, y, photons, bg, sx, sy]
    result = [0, 0, avg_roi, avg_roi, 1, 1]
    return result


def fit_spots(spots: np.ndarray) -> np.ndarray:
    """Fit spots and return fit parameters."""
    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    for i, spot in enumerate(spots):
        theta[i] = fit_spot(spot)
    return theta


def fit_spots_parallel(
    spots: np.ndarray,
    asynch: bool = False,
) -> np.ndarray | list[futures.Future]:
    """Fit spots in parallel (if ``asynch`` is True)."""
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
    n_spots = len(spots)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_spots / n_tasks + 1) if _ < n_spots % n_tasks else int(
            n_spots / n_tasks
        )
        for _ in range(n_tasks)
    ]
    start_indices = np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = futures.ProcessPoolExecutor(n_workers)
    for i, n_spots_task in zip(start_indices, spots_per_task):
        fs.append(executor.submit(fit_spots, spots[i:i + n_spots_task]))
    if asynch:
        return fs
    with tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return fits_from_futures(fs)


def fits_from_futures(futures: list[futures.Future]) -> np.ndarray:
    """Collect fit results from futures."""
    theta = [_.result() for _ in futures]
    return np.vstack(theta)


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: np.ndarray,
    box: int,
    em: float,
) -> pd.DataFrame:
    """Convert fit results to localization DataFrame."""
    box_offset = int(box/2)
    x = theta[:, 0] + identifications["x"] - box_offset
    y = theta[:, 1] + identifications["y"] - box_offset
    lpx = postprocess.localization_precision(
        theta[:, 2], theta[:, 4], theta[:, 3], em=em
    )
    lpy = postprocess.localization_precision(
        theta[:, 2], theta[:, 5], theta[:, 3], em=em
    )
    a = np.maximum(theta[:, 4], theta[:, 5])
    b = np.minimum(theta[:, 4], theta[:, 5])
    ellipticity = (a - b) / a
    if hasattr(identifications, "n_id"):
        locs = pd.DataFrame(
            {
                "frame": identifications["frame"].values.astype(np.uint32),
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
                    identifications["net_gradient"].values.astype(np.float32)
                ),
                "n_id": identifications["n_id"].values.astype(np.uint32),
            }
        )
        locs.sort_values(by="n_id", kind="mergesort", inplace=True)
    else:
        locs = pd.DataFrame(
            {
                "frame": identifications["frame"].values.astype(np.uint32),
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
                    identifications["net_gradient"].values.astype(np.float32)
                ),
            }
        )
        locs.sort_values(by="frame", kind="mergesort", inplace=True)
    return locs
