"""
    picasso.avgroi
    ~~~~~~~~~~~~~~~~

    Return average intensity of Spot

    :author: Maximilian Thomas Strauss, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from scipy import optimize as _optimize
import numpy as _np
from tqdm import tqdm as _tqdm
import numba as _numba
import multiprocessing as _multiprocessing
from concurrent import futures as _futures
from . import postprocess as _postprocess


@_numba.jit(nopython=True, nogil=True)
def _sum(spot, size):
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            _sum_ += spot[i, j]

    return _sum_


def fit_spot(spot):
    size = spot.shape[0]
    avg_roi = _sum(spot, size)
    # result is [x, y, photons, bg, sx, sy]
    result = [0, 0, avg_roi, avg_roi, 1, 1]
    return result


def fit_spots(spots):
    theta = _np.empty((len(spots), 6), dtype=_np.float32)
    theta.fill(_np.nan)
    for i, spot in enumerate(spots):
        theta[i] = fit_spot(spot)
    return theta


def fit_spots_parallel(spots, asynch=False):
    n_workers = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    n_spots = len(spots)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_spots / n_tasks + 1) if _ < n_spots % n_tasks else int(n_spots / n_tasks)
        for _ in range(n_tasks)
    ]
    start_indices = _np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = _futures.ProcessPoolExecutor(n_workers)
    for i, n_spots_task in zip(start_indices, spots_per_task):
        fs.append(executor.submit(fit_spots, spots[i : i + n_spots_task]))
    if asynch:
        return fs
    with _tqdm(total=n_tasks, unit="task") as progress_bar:
        for f in _futures.as_completed(fs):
            progress_bar.update()
    return fits_from_futures(fs)


def fits_from_futures(futures):
    theta = [_.result() for _ in futures]
    return _np.vstack(theta)


def locs_from_fits(identifications, theta, box, em):
    # box_offset = int(box/2)
    x = theta[:, 0] + identifications.x  # - box_offset
    y = theta[:, 1] + identifications.y  # - box_offset
    lpx = _postprocess.localization_precision(
        theta[:, 2], theta[:, 4], theta[:, 3], em=em
    )
    lpy = _postprocess.localization_precision(
        theta[:, 2], theta[:, 5], theta[:, 3], em=em
    )
    a = _np.maximum(theta[:, 4], theta[:, 5])
    b = _np.minimum(theta[:, 4], theta[:, 5])
    ellipticity = (a - b) / a
    if hasattr(identifications, "n_id"):
        locs = _np.rec.array(
            (
                identifications.frame,
                x,
                y,
                theta[:, 2],
                theta[:, 4],
                theta[:, 5],
                theta[:, 3],
                lpx,
                lpy,
                ellipticity,
                identifications.net_gradient,
                identifications.n_id,
            ),
            dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("sx", "f4"),
                ("sy", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
                ("ellipticity", "f4"),
                ("net_gradient", "f4"),
                ("n_id", "u4"),
            ],
        )
        locs.sort(kind="mergesort", order="n_id")
    else:
        locs = _np.rec.array(
            (
                identifications.frame,
                x,
                y,
                theta[:, 2],
                theta[:, 4],
                theta[:, 5],
                theta[:, 3],
                lpx,
                lpy,
                ellipticity,
                identifications.net_gradient,
            ),
            dtype=[
                ("frame", "u4"),
                ("x", "f4"),
                ("y", "f4"),
                ("photons", "f4"),
                ("sx", "f4"),
                ("sy", "f4"),
                ("bg", "f4"),
                ("lpx", "f4"),
                ("lpy", "f4"),
                ("ellipticity", "f4"),
                ("net_gradient", "f4"),
            ],
        )
        locs.sort(kind="mergesort", order="frame")
    return locs
