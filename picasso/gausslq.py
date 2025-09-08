"""
    picasso.gausslq
    ~~~~~~~~~~~~~~~

    Fit spots (single-molecule images) with 2D Gaussian least squares.

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import multiprocessing
from concurrent import futures

import numba
import numpy as np
from scipy import optimize
from tqdm import tqdm

from . import postprocess

try:
    from pygpufit import gpufit as gf
except ImportError:
    pass


@numba.jit(nopython=True, nogil=True)
def _gaussian(mu: float, sigma: float, grid: np.ndarray) -> np.ndarray:
    """Compute a Gaussian PDF on a grid."""
    norm = 0.3989422804014327 / sigma
    return norm * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)


"""
def integrated_gaussian(mu, sigma, grid):
    norm = 0.70710678118654757 / sigma   # sq_norm = sqrt(0.5/sigma**2)
    integrated_gaussian =  0.5 *
    (erf((grid - mu + 0.5) * norm) - erf((grid - mu - 0.5) * norm))
    return integrated_gaussian
"""


@numba.jit(nopython=True, nogil=True)
def _sum_and_center_of_mass(
    spot: np.ndarray,
    size: int,
) -> tuple[float, float, float]:
    """Calculate the sum and center of mass of a 2D spot."""
    x = 0.0
    y = 0.0
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            x += spot[i, j] * i
            y += spot[i, j] * j
            _sum_ += spot[i, j]
    x /= _sum_
    y /= _sum_
    return _sum_, y, x


@numba.jit(nopython=True, nogil=True)
def _initial_sigmas(
    spot: np.ndarray,
    y: float,
    x: float,
    sum: float,
    size: int,
) -> tuple[float, float]:
    """Initialize the sizes of the single-emitter images (sigmas of the
    Gaussian fit) in x and y independently."""
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    for i in range(size):
        for j in range(size):
            sum_deviation_y += spot[i, j] * (i - y) ** 2
            sum_deviation_x += spot[i, j] * (j - x) ** 2
    sy = np.sqrt(sum_deviation_y / sum)
    sx = np.sqrt(sum_deviation_x / sum)
    return sy, sx


@numba.jit(nopython=True, nogil=True)
def _initial_parameters(
    spot: np.ndarray,
    size: int,
    size_half: int,
) -> np.ndarray:
    """Initialize the parameters for the Gaussian fit - x, y, photons,
    background, sigma_x, sigma_y."""
    theta = np.zeros(6, dtype=np.float32)
    theta[3] = np.min(spot)
    spot_without_bg = spot - theta[3]
    sum, theta[1], theta[0] = _sum_and_center_of_mass(spot_without_bg, size)
    theta[2] = np.maximum(1.0, sum)
    theta[5], theta[4] = _initial_sigmas(
        spot - theta[3], theta[1], theta[0], sum, size
    )
    theta[0:2] -= size_half
    return theta


def initial_parameters_gpufit(spots: np.ndarray, size: int) -> np.ndarray:
    """Initialize the parameters for the GPU fit - photons, x, y, sx,
    sy, bg."""
    center = (size / 2.0) - 0.5
    initial_width = np.amax([size / 5.0, 1.0])

    spot_max = np.amax(spots, axis=(1, 2))
    spot_min = np.amin(spots, axis=(1, 2))

    initial_parameters = np.empty((len(spots), 6), dtype=np.float32)

    initial_parameters[:, 0] = spot_max - spot_min
    initial_parameters[:, 1] = center
    initial_parameters[:, 2] = center
    initial_parameters[:, 3] = initial_width
    initial_parameters[:, 4] = initial_width
    initial_parameters[:, 5] = spot_min

    return initial_parameters


@numba.jit(nopython=True, nogil=True)
def _outer(
    a: np.ndarray,
    b: np.ndarray,
    size: int,
    model: np.ndarray,
    n: float,
    bg: float
) -> None:
    """Compute the outer product of two vectors a and b, scaled by n and
    added a background value bg, and store the result in model."""
    for i in range(size):
        for j in range(size):
            model[i, j] = n * a[i] * b[j] + bg


@numba.jit(nopython=True, nogil=True)
def _compute_model(
    theta: np.ndarray,
    grid: np.ndarray,
    size: int,
    model_x: np.ndarray,
    model_y: np.ndarray,
    model: np.ndarray,
) -> np.ndarray:
    """Compute the model of a Gaussian spot (2D) based on the parameters
    in theta, which contains the x and y positions, the number of
    photons, background, and the sigmas in x and y."""
    model_x[:] = _gaussian(
        theta[0], theta[4], grid
    )  # sx and sy are wrong with integrated gaussian
    model_y[:] = _gaussian(
        theta[1], theta[5], grid
    )
    _outer(model_y, model_x, size, model, theta[2], theta[3])
    return model


@numba.jit(nopython=True, nogil=True)
def _compute_residuals(
    theta: np.ndarray,
    spot: np.ndarray,
    grid: np.ndarray,
    size: int,
    model_x: np.ndarray,
    model_y: np.ndarray,
    model: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Compute the residuals (i.e., the difference in pixel values)
    between the observed spot and the model computed from the parameters
    in theta."""
    _compute_model(theta, grid, size, model_x, model_y, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


def fit_spot(spot: np.ndarray) -> np.ndarray:
    """Fit a single spot using least squares optimization. The spot is a
    2D array representing the pixel values of the spot image. The
    function returns the optimized parameters as a 1D array with the
    following order: [x, y, photons, bg, sx, sy].

    The parameters are initialized based on the spot's pixel values, and
    the optimization is performed using the least squares method. The
    optimization minimizes the residuals between the observed spot and
    the model computed from the parameters.

    Parameters
    ----------
    spot : np.ndarray
        A 2D array representing the pixel values of the spot image.
        The shape of the array should be (size, size), where size is the
        length of one side of the square spot image.

    Returns
    -------
    result_ : np.ndarray
        A 1D array containing the optimized parameters in the following order:
        [x, y, photons, bg, sx, sy].
    """
    size = spot.shape[0]
    size_half = int(size / 2)
    grid = np.arange(-size_half, size_half + 1, dtype=np.float32)
    model_x = np.empty(size, dtype=np.float32)
    model_y = np.empty(size, dtype=np.float32)
    model = np.empty((size, size), dtype=np.float32)
    residuals = np.empty((size, size), dtype=np.float32)
    # theta is [x, y, photons, bg, sx, sy]
    theta0 = _initial_parameters(spot, size, size_half)
    args = (spot, grid, size, model_x, model_y, model, residuals)
    result = optimize.leastsq(
        _compute_residuals, theta0, args=args, ftol=1e-2, xtol=1e-2
    )  # leastsq is much faster than least_squares
    """
    model = compute_model(result[0], grid, size, model_x, model_y, model)
    plt.figure()
    plt.subplot(121)
    plt.imshow(spot, interpolation='none')
    plt.subplot(122)
    plt.imshow(model, interpolation='none')
    plt.colorbar()
    plt.show()
    """
    result_ = result[0]
    return result_


def fit_spots(spots: np.ndarray) -> np.ndarray:
    """Fit multiple spots using least squares optimization. Each spot is
    a 2D array representing the pixel values of the spot image. The
    function returns a 2D array with the optimized parameters for each
    spot, where each row corresponds to a spot and the columns are the
    parameters in the following order: [x, y, photons, bg, sx, sy].

    Parameters
    ----------
    spots : np.ndarray
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.

    Returns
    -------
    theta : np.ndarray
        A 2D array with the optimized parameters for each spot. The
        columns correspond to [x, y, photons, bg, sx, sy].
    """
    theta = np.empty((len(spots), 6), dtype=np.float32)
    theta.fill(np.nan)
    for i, spot in enumerate(spots):
        theta[i] = fit_spot(spot)
    return theta


def fit_spots_parallel(
    spots: np.ndarray,
    asynch: bool = False,
) -> np.ndarray | list[futures.Future]:
    """Allows for running ``fit_spots`` asynchronously
    (multiprocessing).

    Parameters
    ----------
    spots : np.ndarray
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.
    asynch : bool, optional
        If True, the function returns a list of futures that can be
        processed asynchronously. If False, the function waits for all
        futures to complete and returns the results as a 2D array.

    Returns
    -------
    np.ndarray | list[_futures.Future]
        If `asynch` is False, returns a 2D array with the optimized
        parameters for each spot, where each row corresponds to a spot
        and the columns are the parameters in the following order:
        [x, y, photons, bg, sx, sy]. If `asynch` is True, returns a list
        of futures that can be processed asynchronously.
    """
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
    n_spots = len(spots)
    n_tasks = 100 * n_workers
    spots_per_task = [
        int(n_spots / n_tasks + 1)
        if _ < n_spots % n_tasks
        else int(n_spots / n_tasks)
        for _ in range(n_tasks)
    ]
    start_indices = np.cumsum([0] + spots_per_task[:-1])
    fs = []
    executor = futures.ProcessPoolExecutor(n_workers)
    for i, n_spots_task in zip(start_indices, spots_per_task):
        fs.append(executor.submit(fit_spots, spots[i:i + n_spots_task]))
    if asynch:
        return fs
    with tqdm(desc="LQ fitting", total=n_tasks, unit="task") as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return fits_from_futures(fs)


def fit_spots_gpufit(spots: np.ndarray) -> np.ndarray:
    """Fit multiple spots using GPU-based Gaussian fitting. Each spot is
    a 2D array representing the pixel values of the spot image. The
    function returns a 2D array with the optimized parameters for each
    spot, where each row corresponds to a spot and the columns are the
    parameters in the following order: [photons, x, y, sx, sy, bg].

    Cite: Przybylski, et al. Scientific Reports, 2017.
    DOI: 10.1038/s41598-017-15313-9

    Parameters
    ----------
    spots : np.ndarray
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.

    Returns
    -------
    parameters : np.ndarray
        A 2D array with the optimized parameters for each spot. The
        columns correspond to [photons, x, y, sx, sy, bg].
    """
    size = spots.shape[1]
    initial_parameters = initial_parameters_gpufit(spots, size)
    spots.shape = (len(spots), (size * size))
    model_id = gf.ModelID.GAUSS_2D_ELLIPTIC

    parameters, states, chi_squares, number_iterations, exec_time = gf.fit(
        spots,
        None,
        model_id,
        initial_parameters,
        tolerance=1e-2,
        max_number_iterations=20,
    )

    parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]

    return parameters


def fits_from_futures(futures: list[futures.Future]) -> np.ndarray:
    """Collect results from futures and stack them into a 2D array."""
    theta = [_.result() for _ in futures]
    return np.vstack(theta)


def locs_from_fits(
    identifications: np.recarray,
    theta: np.ndarray,
    box: int,
    em: bool,
) -> np.recarray:
    """Convert the fit results into a structured array of localizations.

    Parameters
    ----------
    identifications : np.recarray
        A structured array containing the identifications of the spots,
        including frame numbers, x and y coordinates, and net gradient.
    theta : np.ndarray
        A 2D array with the optimized parameters for each spot, where
        each row corresponds to a spot and the columns are the
        parameters in the following order: [x, y, photons, bg, sx, sy].
    box : int
        The size of the box used for localization, which is used to
        calculate the offsets for the x and y coordinates.
    em : bool
        Whether EMCCD was used for the localization.

    Returns
    -------
    locs : np.recarray
        A structured array containing the localized spots.
    """
    # box_offset = int(box/2)
    x = theta[:, 0] + identifications.x  # - box_offset
    y = theta[:, 1] + identifications.y  # - box_offset
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
        locs = np.rec.array(
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
        locs = np.rec.array(
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


def locs_from_fits_gpufit(
    identifications: np.recarray,
    theta: np.ndarray,
    box: int,
    em: bool,
) -> np.recarray:
    """Convert the fit results from GPU-based fitting into a structured
    array of localizations.

    Parameters
    ----------
    identifications : np.recarray
        A structured array containing the identifications of the spots,
        including frame numbers, x and y coordinates, and net gradient.
    theta : np.ndarray
        A 2D array with the optimized parameters for each spot, where
        each row corresponds to a spot and the columns are the
        parameters in the following order: [photons, x, y, sx, sy, bg].
    box : int
        The size of the box used for localization, which is used to
        calculate the offsets for the x and y coordinates.
    em : bool
        Whether EMCCD was used for the localization.

    Returns
    -------
    locs : np.recarray
        A structured array containing the localized spots.
    """
    box_offset = int(box / 2)
    x = theta[:, 1] + identifications.x - box_offset
    y = theta[:, 2] + identifications.y - box_offset
    lpx = postprocess.localization_precision(
        theta[:, 0], theta[:, 3], theta[:, 5], em=em
    )
    lpy = postprocess.localization_precision(
        theta[:, 0], theta[:, 4], theta[:, 5], em=em
    )
    a = np.maximum(theta[:, 3], theta[:, 4])
    b = np.minimum(theta[:, 3], theta[:, 4])
    ellipticity = (a - b) / a
    locs = np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 0],
            theta[:, 3],
            theta[:, 4],
            theta[:, 5],
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
