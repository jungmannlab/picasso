"""
picasso.gausslq
~~~~~~~~~~~~~~~

Fit spots (single-molecule images) with 2D Gaussian least squares.

.. deprecated:: 0.11
    **This whole module will be removed in Picasso 1.0.** Every public
    name in it now warns. All fitting lives in :mod:`picasso.fitting`:

    ============================  ============================================
    this module                   replacement
    ============================  ============================================
    ``fit_spot`` / ``fit_spots``  ``fitting.gaussfit.fit_spots``
    ``fit_spots_parallel``        ``fitting.gaussfit.fit_spots_async``
    ``fit_spots_gauss_gpu``       ``fitting.gaussfit_cuda.fit_spots``
    ``locs_from_fits``            ``localize.locs_from_fits_gauss``
    ``localization_precision``    ``fitting.precision.localization_precision``
    ``sigma_uncertainty``         ``fitting.precision.sigma_uncertainty_lsq``
    ============================  ============================================

The optimizer here is SciPy's ``leastsq`` (MINPACK) and is *not* derived from
Gpufit. Its GPU counterpart is: ``fit_spots_gauss_gpu`` below is a thin shim
onto :mod:`picasso.fitting.gaussfit_cuda`, whose Levenberg-Marquardt driver and
models are a port of Gpufit (Przybylski et al., Scientific Reports 7,
15722, 2017; license in ``LICENSES/Gpufit-LICENSE.txt``). Both sample the
Gaussian at the pixel center, but they parameterize the amplitude differently
(here ``photons`` scales a normalized PDF; there it is the peak height, which
``picasso.localize`` converts afterwards), so the two are not interchangeable
at the array level even though the fitted positions and widths agree.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from concurrent import futures
from typing import Callable, Literal

import numba
import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from picasso import lib
from picasso.fitting import precision

# Convergence schedule. ``TOLERANCE`` is MINPACK's relative reduction in both
# the sum of squares (``ftol``) and the parameter vector (``xtol``).
TOLERANCE = 1e-2
MAX_ITERATIONS = 200

# The whole module is deprecated. Every public name is a thin wrapper that
# warns and delegates to a private implementation (or to the new home of the
# code); Picasso's own callers use those, because a library warning about its
# own internals is noise rather than a signal.
_DEPRECATION_MESSAGE = (
    "picasso.gausslq is deprecated and will be removed in Picasso 1.0. All "
    "fitting now lives in picasso.fitting: use "
    "picasso.fitting.gaussfit.fit_spots (or fit_spots_async) for the fit, "
    "picasso.localize.locs_from_fits_gauss to build the localizations, "
    "and picasso.fitting.precision.localization_precision / "
    "sigma_uncertainty_lsq for the analytic precisions."
)


def _max_function_evaluations(max_iterations: int, n_parameters: int) -> int:
    """``leastsq``'s ``maxfev`` for ``max_iterations`` LM iterations."""
    return int(max_iterations) * (int(n_parameters) + 1)


@numba.jit(nopython=True, nogil=True)
def _gaussian(
    mu: float, sigma: float, grid: lib.FloatArray1D
) -> lib.FloatArray1D:
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
    spot: lib.FloatArray2D,
    size: int,
) -> tuple[float, float, float]:
    """Calculate the sum and center of mass of a 2D spot."""
    y = 0.0
    x = 0.0
    _sum_ = 0.0
    for i in range(size):
        for j in range(size):
            y += spot[i, j] * i
            x += spot[i, j] * j
            _sum_ += spot[i, j]
    if _sum_ <= 0.0:
        # Degenerate (flat) spot: fall back to geometric center so the
        # caller can still produce sane initial parameters.
        return 0.01, (size - 1) / 2.0, (size - 1) / 2.0
    y /= _sum_
    x /= _sum_
    return _sum_, y, x


@numba.jit(nopython=True, nogil=True)
def _initial_sigmas(
    spot: lib.FloatArray2D,
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
    spot: lib.FloatArray2D,
    size: int,
    size_half: int,
) -> lib.FloatArray1D:
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


@numba.jit(nopython=True, nogil=True)
def _initial_parameters_sigma(
    spot: lib.FloatArray2D,
    size: int,
    size_half: int,
) -> lib.FloatArray1D:
    """Initialize the parameters for the spherical (isotropic) Gaussian
    fit - x, y, photons, background, sigma. A single width is used; it
    is the average of the per-axis width estimates."""
    theta = np.zeros(5, dtype=np.float32)
    theta[3] = np.min(spot)
    spot_without_bg = spot - theta[3]
    sum, theta[1], theta[0] = _sum_and_center_of_mass(spot_without_bg, size)
    theta[2] = np.maximum(1.0, sum)
    sy, sx = _initial_sigmas(spot - theta[3], theta[1], theta[0], sum, size)
    theta[4] = (sx + sy) / 2.0
    theta[0:2] -= size_half
    return theta


@numba.jit(nopython=True, nogil=True)
def _initial_parameters_rotated(
    spot: lib.FloatArray2D,
    size: int,
    size_half: int,
) -> lib.FloatArray1D:
    """Initialize the parameters for the rotated elliptical Gaussian fit
    - x, y, photons, background, sigma_x, sigma_y, angle (radians)."""
    theta = np.zeros(7, dtype=np.float32)
    theta[3] = np.min(spot)
    spot_without_bg = spot - theta[3]
    sum, theta[1], theta[0] = _sum_and_center_of_mass(spot_without_bg, size)
    theta[2] = np.maximum(1.0, sum)
    sy, sx = _initial_sigmas(spot - theta[3], theta[1], theta[0], sum, size)
    # Break the sx/sy symmetry so the angle has a well-defined gradient at
    # the start (a circular seed makes the model independent of the angle),
    # mirroring the GPU initialization.
    theta[4] = sx * 1.1
    theta[5] = sy * 0.9
    theta[6] = 0.0
    theta[0:2] -= size_half
    return theta


@numba.jit(nopython=True, nogil=True)
def _outer(
    a: lib.FloatArray1D,
    b: lib.FloatArray1D,
    size: int,
    model: lib.FloatArray2D,
    n: float,
    bg: float,
) -> None:
    """Compute the outer product of two vectors a and b, scaled by n and
    added a background value bg, and store the result in model."""
    for i in range(size):
        for j in range(size):
            model[i, j] = n * a[i] * b[j] + bg


@numba.jit(nopython=True, nogil=True)
def _compute_model(
    theta: lib.FloatArray1D,
    grid: lib.FloatArray1D,
    size: int,
    model_x: lib.FloatArray1D,
    model_y: lib.FloatArray1D,
    model: lib.FloatArray2D,
) -> lib.FloatArray2D:
    """Compute the model of a Gaussian spot (2D) based on the parameters
    in theta, which contains the x and y positions, the number of
    photons, background, and the sigmas in x and y."""
    model_x[:] = _gaussian(
        theta[0], theta[4], grid
    )  # sx and sy are wrong with integrated gaussian
    model_y[:] = _gaussian(theta[1], theta[5], grid)
    _outer(model_y, model_x, size, model, theta[2], theta[3])
    return model


@numba.jit(nopython=True, nogil=True)
def _compute_model_sigma(
    theta: lib.FloatArray1D,
    grid: lib.FloatArray1D,
    size: int,
    model_x: lib.FloatArray1D,
    model_y: lib.FloatArray1D,
    model: lib.FloatArray2D,
) -> lib.FloatArray2D:
    """Compute the model of a spherical (isotropic) Gaussian spot (2D)
    based on the parameters in theta, which contains the x and y
    positions, the number of photons, background, and a single sigma
    used for both axes."""
    model_x[:] = _gaussian(theta[0], theta[4], grid)
    model_y[:] = _gaussian(theta[1], theta[4], grid)
    _outer(model_y, model_x, size, model, theta[2], theta[3])
    return model


@numba.jit(nopython=True, nogil=True)
def _compute_model_rotated(
    theta: lib.FloatArray1D,
    grid: lib.FloatArray1D,
    size: int,
    model: lib.FloatArray2D,
) -> lib.FloatArray2D:
    """Compute the model of a rotated elliptical Gaussian spot (2D) based
    on the parameters in theta - x, y, photons, background, sigma_x,
    sigma_y and the rotation angle (radians). Unlike the non-rotated
    models, the rotated Gaussian is not separable, so the full 2D
    exponential is evaluated per pixel (point-sampled at the pixel
    centers, matching the GPU's GAUSS_2D_ROTATED model)."""
    ct = np.cos(theta[6])
    st = np.sin(theta[6])
    sx = theta[4]
    sy = theta[5]
    norm = theta[2] / (2.0 * np.pi * sx * sy)
    for i in range(size):
        dy = grid[i] - theta[1]
        for j in range(size):
            dx = grid[j] - theta[0]
            u = dx * ct - dy * st
            w = dx * st + dy * ct
            model[i, j] = (
                norm * np.exp(-0.5 * (u**2 / sx**2 + w**2 / sy**2)) + theta[3]
            )
    return model


@numba.jit(nopython=True, nogil=True)
def _compute_residuals(
    theta: lib.FloatArray1D,
    spot: lib.FloatArray2D,
    grid: lib.FloatArray1D,
    size: int,
    model_x: lib.FloatArray1D,
    model_y: lib.FloatArray1D,
    model: lib.FloatArray2D,
    residuals: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Compute the residuals (i.e., the difference in pixel values)
    between the observed spot and the model computed from the parameters
    in theta."""
    _compute_model(theta, grid, size, model_x, model_y, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


@numba.jit(nopython=True, nogil=True)
def _compute_residuals_sigma(
    theta: lib.FloatArray1D,
    spot: lib.FloatArray2D,
    grid: lib.FloatArray1D,
    size: int,
    model_x: lib.FloatArray1D,
    model_y: lib.FloatArray1D,
    model: lib.FloatArray2D,
    residuals: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Compute the residuals between the observed spot and the spherical
    (isotropic) Gaussian model computed from the parameters in theta."""
    _compute_model_sigma(theta, grid, size, model_x, model_y, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


@numba.jit(nopython=True, nogil=True)
def _compute_residuals_rotated(
    theta: lib.FloatArray1D,
    spot: lib.FloatArray2D,
    grid: lib.FloatArray1D,
    size: int,
    model: lib.FloatArray2D,
    residuals: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Compute the residuals between the observed spot and the rotated
    elliptical Gaussian model computed from the parameters in theta."""
    _compute_model_rotated(theta, grid, size, model)
    residuals[:, :] = spot - model
    return residuals.flatten()


def _chi_square_of(result: tuple) -> float:
    """Chi-square of a ``leastsq(..., full_output=1)`` result.

    ``result[2]["fvec"]`` is the residual vector at the returned parameters,
    so this is the sum of squared residuals at the fit optimum - the
    least-squares goodness of fit, in photons squared (spots are
    gain-converted before fitting)."""
    return float(np.sum(np.asarray(result[2]["fvec"], dtype=np.float64) ** 2))


def _with_chi_square(result: tuple, return_chi_square: bool):
    """The fitted parameters, with the chi-square appended if asked for."""
    if not return_chi_square:
        return result[0]
    return np.append(result[0], _chi_square_of(result))


def fit_spot(
    spot: lib.FloatArray2D,
    spherical: bool = False,
    rotated: bool = False,
    return_chi_square: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> lib.FloatArray1D:
    """Fit a single spot using least squares optimization.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Use
        :func:`picasso.fitting.gaussfit.fit_spots`, which fits the same
        sampled Gaussian with the Levenberg-Marquardt driver shared with the
        GPU backend, on either device.

    See :func:`_fit_spot` for the full description.

    Parameters
    ----------
    spot : lib.FloatArray2D
        ``(size, size)`` pixel values of the spot image.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single width.
        Default is False.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian. Default is False.
    return_chi_square : bool, optional
        If True, append the chi-square at the fit optimum to the returned
        parameters. Default is False.
    tolerance : float or None, optional
        Convergence criterion. None (the default) uses :data:`TOLERANCE`.
    max_iterations : int or None, optional
        Maximum number of iterations. None (the default) uses
        :data:`MAX_ITERATIONS`.

    Returns
    -------
    result_ : lib.FloatArray1D
        Optimized parameters [x, y, photons, bg, sx, sy] (plus the angle in
        radians if ``rotated``, and the chi-square if
        ``return_chi_square``).
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return _fit_spot(
        spot,
        spherical=spherical,
        rotated=rotated,
        return_chi_square=return_chi_square,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _fit_spot(
    spot: lib.FloatArray2D,
    spherical: bool = False,
    rotated: bool = False,
    return_chi_square: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> lib.FloatArray1D:
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
    spot : lib.FloatArray2D
        A 2D array representing the pixel values of the spot image.
        The shape of the array should be (size, size), where size is the
        length of one side of the square spot image.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single
        width. The returned parameters still use the elliptical layout
        with sx == sy so the rest of the pipeline is unchanged. Default
        is False.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian; the returned array
        has a seventh element, the rotation angle in radians. Cannot be
        combined with ``spherical``. Default is False.
    return_chi_square : bool, optional
        If True, the returned array carries one extra trailing element:
        the chi-square (residual sum of squares) at the fit optimum, the
        least-squares goodness-of-fit measure. It is appended to the
        parameters rather than returned separately so that the
        multiprocessing plumbing (``fit_spots_parallel`` /
        ``fits_from_futures``) keeps stacking plain 2D arrays.
        ``locs_from_fits`` takes it as its own ``chi_square`` argument, so
        callers split it off (see ``localize._fit2d_gausslq``). Default is
        False.
    tolerance : float or None, optional
        Convergence criterion, passed to ``leastsq`` as both ``ftol`` and
        ``xtol``. None (the default) uses :data:`TOLERANCE`.
    max_iterations : int or None, optional
        Maximum number of Levenberg-Marquardt iterations. None (the default)
        uses :data:`MAX_ITERATIONS`. See :func:`_max_function_evaluations` for
        the conversion to ``leastsq``'s ``maxfev``.

    Returns
    -------
    result_ : lib.FloatArray1D
        A 1D array containing the optimized parameters in the following
        order: [x, y, photons, bg, sx, sy], or, if ``rotated``,
        [x, y, photons, bg, sx, sy, angle (radians)]. If
        ``return_chi_square``, the chi-square is appended as the last
        element.
    """
    size = spot.shape[0]
    size_half = int(size / 2)
    grid = np.arange(-size_half, size_half + 1, dtype=np.float32)
    model_x = np.empty(size, dtype=np.float32)
    model_y = np.empty(size, dtype=np.float32)
    model = np.empty((size, size), dtype=np.float32)
    residuals = np.empty((size, size), dtype=np.float32)
    tol = TOLERANCE if tolerance is None else float(tolerance)
    max_it = MAX_ITERATIONS if max_iterations is None else int(max_iterations)
    # full_output exposes leastsq's infodict, whose "fvec" is the residual
    # vector at the returned parameters - the chi-square is then free, with
    # no extra model evaluation.
    full_output = 1 if return_chi_square else 0
    if rotated:
        # theta is [x, y, photons, bg, sx, sy, angle]; the rotated model
        # is not separable, so it does not use the per-axis buffers.
        theta0 = _initial_parameters_rotated(spot, size, size_half)
        result = optimize.leastsq(
            _compute_residuals_rotated,
            theta0,
            args=(spot, grid, size, model, residuals),
            ftol=tol,
            xtol=tol,
            maxfev=_max_function_evaluations(max_it, len(theta0)),
            full_output=full_output,
        )
        return _with_chi_square(result, return_chi_square)
    args = (spot, grid, size, model_x, model_y, model, residuals)
    if spherical:
        # theta is [x, y, photons, bg, sigma]
        theta0 = _initial_parameters_sigma(spot, size, size_half)
        result = optimize.leastsq(
            _compute_residuals_sigma,
            theta0,
            args=args,
            ftol=tol,
            xtol=tol,
            maxfev=_max_function_evaluations(max_it, len(theta0)),
            full_output=full_output,
        )
        fitted = result[0]
        # Expand the single width into sx == sy so downstream code
        # (locs_from_fits) sees the standard 6-parameter layout.
        result_ = np.empty(6, dtype=fitted.dtype)
        result_[0:5] = fitted
        result_[5] = fitted[4]
        if return_chi_square:
            return np.append(result_, _chi_square_of(result))
        return result_
    # theta is [x, y, photons, bg, sx, sy]
    theta0 = _initial_parameters(spot, size, size_half)
    result = optimize.leastsq(
        _compute_residuals,
        theta0,
        args=args,
        ftol=tol,
        xtol=tol,
        maxfev=_max_function_evaluations(max_it, len(theta0)),
        full_output=full_output,
    )  # leastsq is much faster than least_squares
    return _with_chi_square(result, return_chi_square)


def fit_spots(
    spots: lib.FloatArray3D,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    spherical: bool = False,
    rotated: bool = False,
    return_chi_square: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> lib.FloatArray2D:
    """Fit multiple spots using least squares optimization.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Use
        :func:`picasso.fitting.gaussfit.fit_spots`, which fits the same
        sampled Gaussian with the Levenberg-Marquardt driver shared with the
        GPU backend, on either device.

    See :func:`_fit_spots` for the full description.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, size, size)`` spot images.
    progress_callback : callable, "console" or None, optional
        A callable receives the number of localized spots; "console" shows
        a tqdm bar; None disables progress tracking.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single width.
        Default is False.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian. Default is False.
    return_chi_square : bool, optional
        If True, append the per-spot chi-square as one extra trailing
        column. Default is False.
    tolerance : float or None, optional
        Convergence criterion. None (the default) uses :data:`TOLERANCE`.
    max_iterations : int or None, optional
        Maximum number of iterations per spot. None (the default) uses
        :data:`MAX_ITERATIONS`.

    Returns
    -------
    theta : lib.FloatArray2D
        Optimized parameters per spot, columns [x, y, photons, bg, sx, sy]
        (plus the angle in radians if ``rotated``, and the chi-square if
        ``return_chi_square``).
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return _fit_spots(
        spots,
        progress_callback,
        spherical=spherical,
        rotated=rotated,
        return_chi_square=return_chi_square,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _fit_spots(
    spots: lib.FloatArray3D,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    spherical: bool = False,
    rotated: bool = False,
    return_chi_square: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> lib.FloatArray2D:
    """Fit multiple spots using least squares optimization. Each spot is
    a 2D array representing the pixel values of the spot image. The
    function returns a 2D array with the optimized parameters for each
    spot, where each row corresponds to a spot and the columns are the
    parameters in the following order: [x, y, photons, bg, sx, sy]
    (or, if ``rotated``, [x, y, photons, bg, sx, sy, angle (radians)]).

    Parameters
    ----------
    spots : lib.FloatArray3D
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.
    progress_callback : callable or None
        If a callable provided, it must accept one integer input (number
        of localized spots). If "console", tqdm is used to display
        progress. If None, progress is not tracked.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single
        width; the resulting sx and sy columns are identical. Default is
        False.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian; the returned array
        has a seventh column, the rotation angle in radians. Cannot be
        combined with ``spherical``. Default is False.
    return_chi_square : bool, optional
        If True, append the per-spot chi-square (residual sum of squares
        at the fit optimum) as one extra trailing column. See
        ``fit_spot``. Default is False.
    tolerance : float or None, optional
        Convergence criterion; None (the default) uses :data:`TOLERANCE`.
        See ``fit_spot``.
    max_iterations : int or None, optional
        Maximum number of iterations per spot; None (the default) uses
        :data:`MAX_ITERATIONS`. See ``fit_spot``.

    Returns
    -------
    theta : lib.FloatArray2D
        A 2D array with the optimized parameters for each spot. The
        columns correspond to [x, y, photons, bg, sx, sy] (or, if
        ``rotated``, [x, y, photons, bg, sx, sy, angle (radians)]), plus
        a trailing chi-square column if ``return_chi_square``.
    """
    n_params = 7 if rotated else 6
    if return_chi_square:
        n_params += 1
    theta = np.empty((len(spots), n_params), dtype=np.float32)
    theta.fill(np.nan)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(range(len(spots)), desc="Fitting...", unit="spot")
    else:
        iter_range = range(len(spots))
    for i in iter_range:
        spot = spots[i]
        theta[i] = _fit_spot(
            spot,
            spherical=spherical,
            rotated=rotated,
            return_chi_square=return_chi_square,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        if callable(progress_callback):
            progress_callback(i + 1)
    return theta


def fit_spots_parallel(
    spots: lib.FloatArray3D,
    asynch: bool = False,
    spherical: bool = False,
    rotated: bool = False,
    return_chi_square: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> lib.FloatArray2D | list[futures.Future]:
    """Allows for running ``fit_spots`` asynchronously (multiprocessing).

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Use
        :func:`picasso.fitting.gaussfit.fit_spots`, which fits the same
        sampled Gaussian with the Levenberg-Marquardt driver shared with the
        GPU backend, on either device.

    :func:`picasso.fitting.gaussfit.fit_spots_async` is the direct
    replacement, and uses threads rather than up to 60 worker
    processes.

    See :func:`_fit_spots_parallel` for the full description.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, size, size)`` spot images.
    asynch : bool, optional
        If True, return the futures immediately instead of waiting for the
        results. Default is False.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single width.
        Default is False.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian. Default is False.
    return_chi_square : bool, optional
        If True, append the per-spot chi-square as one extra trailing
        column. Default is False.
    tolerance : float or None, optional
        Convergence criterion. None (the default) uses :data:`TOLERANCE`.
    max_iterations : int or None, optional
        Maximum number of iterations per spot. None (the default) uses
        :data:`MAX_ITERATIONS`.

    Returns
    -------
    lib.FloatArray2D or list of futures.Future
        The stacked parameters (as in :func:`fit_spots`) if ``asynch`` is
        False, else the list of futures.
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return _fit_spots_parallel(
        spots,
        asynch=asynch,
        spherical=spherical,
        rotated=rotated,
        return_chi_square=return_chi_square,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _fit_spots_parallel(
    spots: lib.FloatArray3D,
    asynch: bool = False,
    spherical: bool = False,
    rotated: bool = False,
    return_chi_square: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> lib.FloatArray2D | list[futures.Future]:
    """Allows for running ``fit_spots`` asynchronously
    (multiprocessing).

    Parameters
    ----------
    spots : lib.FloatArray3D
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.
    asynch : bool, optional
        If True, the function returns a list of futures that can be
        processed asynchronously. If False, the function waits for all
        futures to complete and returns the results as a 2D array.
    spherical : bool, optional
        If True, fit a spherical (isotropic) Gaussian with a single
        width; the resulting sx and sy columns are identical. Default is
        False.
    rotated : bool, optional
        If True, fit a rotated elliptical Gaussian; the returned array
        has a seventh column, the rotation angle in radians. Cannot be
        combined with ``spherical``. Default is False.
    return_chi_square : bool, optional
        If True, append the per-spot chi-square (residual sum of squares
        at the fit optimum) as one extra trailing column. See
        ``fit_spot``. Default is False.
    tolerance : float or None, optional
        Convergence criterion; None (the default) uses :data:`TOLERANCE`.
        See ``fit_spot``.
    max_iterations : int or None, optional
        Maximum number of iterations per spot; None (the default) uses
        :data:`MAX_ITERATIONS`. See ``fit_spot``.

    Returns
    -------
    lib.FloatArray2D | list[futures.Future]
        If `asynch` is False, returns a 2D array with the optimized
        parameters for each spot, where each row corresponds to a spot
        and the columns are the parameters in the following order:
        [x, y, photons, bg, sx, sy], plus a trailing chi-square column if
        ``return_chi_square``. If `asynch` is True, returns a list of
        futures that can be processed asynchronously.
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
        fs.append(
            executor.submit(
                _fit_spots,
                spots[i : i + n_spots_task],
                spherical=spherical,
                rotated=rotated,
                return_chi_square=return_chi_square,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        )
    if asynch:
        return fs
    with tqdm(desc="LQ fitting", total=n_tasks, unit="task") as progress_bar:
        for f in futures.as_completed(fs):
            progress_bar.update()
    return _fits_from_futures(fs)


def fit_spots_gauss_gpu(spots: lib.FloatArray3D) -> lib.FloatArray2D:
    """Fit multiple spots with a (non-rotated) elliptical 2D Gaussian
    using least-squares fitting on the GPU.

    .. deprecated:: 0.11
        Removed in Picasso 1.0. Use
        ``picasso.localize.fit_spots_gauss_gpu``, which additionally
        supports the rotated elliptical Gaussian model and the MLE
        estimator, or
        :func:`picasso.fitting.gaussfit_cuda.fit_spots` directly.

    Parameters
    ----------
    spots : lib.FloatArray3D
        A 3D array of shape (n_spots, size, size), where n_spots is the
        number of spots and size is the length of one side of the square
        spot image. Each slice along the first axis represents a single
        spot image.

    Returns
    -------
    parameters : lib.FloatArray2D
        A 2D array with the optimized parameters for each spot. The
        columns correspond to [photons, x, y, sx, sy, bg].
    """
    from picasso import localize

    lib.deprecation_warning(
        "picasso.gausslq.fit_spots_gauss_gpu is deprecated and will "
        "be removed in Picasso 1.0. Use "
        "picasso.localize.fit_spots_gauss_gpu, or "
        "picasso.fitting.gaussfit_cuda.fit_spots directly."
    )
    return localize.fit_spots_gauss_gpu(spots)


def fits_from_futures(futures: list[futures.Future]) -> lib.FloatArray2D:
    """Collect results from futures and stack them into a 2D array.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Plumbing for
        ``fit_spots_parallel``;
        :func:`picasso.fitting.gaussfit.fit_spots_async` needs no
        equivalent, since its threads write into shared arrays.

    Parameters
    ----------
    futures : list of futures.Future
        Futures returned by :func:`fit_spots_parallel` with ``asynch=True``.

    Returns
    -------
    lib.FloatArray2D
        The per-task parameter arrays, stacked row-wise.
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return _fits_from_futures(futures)


def _fits_from_futures(futures: list[futures.Future]) -> lib.FloatArray2D:
    """Collect results from futures and stack them into a 2D array."""
    theta = [_.result() for _ in futures]
    return np.vstack(theta)


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    spherical: bool = False,
    chi_square: lib.FloatArray1D | None = None,
) -> pd.DataFrame:
    """Convert the fit results into a data frame of localizations.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Use
        ``picasso.localize.locs_from_fits_gauss``, which builds the same
        table from the parameter layout ``picasso.fitting.gaussfit`` returns.

    See :func:`_locs_from_fits` for the full description.

    Parameters
    ----------
    identifications : pd.DataFrame
        Identifications of the spots (frame, x, y and net gradient).
    theta : lib.FloatArray2D
        Optimized parameters per spot, columns [x, y, photons, bg, sx, sy]
        (plus the rotation angle in radians, if present).
    box : int
        Size of the box used for localization.
    em : bool
        Whether EMCCD was used for the localization.
    spherical : bool, optional
        If True, the fit was a spherical Gaussian and the ``ellipticity``
        column is omitted. Default is False.
    chi_square : lib.FloatArray1D, optional
        Per-spot chi-square; if provided, the ``chi_square`` column is
        added.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return _locs_from_fits(
        identifications, theta, box, em, spherical, chi_square
    )


def _locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    spherical: bool = False,
    chi_square: lib.FloatArray1D | None = None,
) -> pd.DataFrame:
    """Convert the fit results into a data frame of localizations.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identifications of the spots,
        including frame numbers, x and y coordinates, and net gradient.
    theta : lib.FloatArray2D
        A 2D array with the optimized parameters for each spot, where
        each row corresponds to a spot and the columns are the
        parameters in the following order: [x, y, photons, bg, sx, sy].
        If a seventh column is present it is interpreted as the rotation
        angle (in radians) of a rotated elliptical Gaussian, and the
        resulting data frame contains the column ``angle`` (in degrees).
    box : int
        The size of the box used for localization, which is used to
        calculate the offsets for the x and y coordinates.
    em : bool
        Whether EMCCD was used for the localization.
    spherical : bool, optional
        If True, the fit was a spherical (isotropic) Gaussian, so
        ``sx == sy`` and the ellipticity is always 0. The
        ``ellipticity`` column is then omitted as it carries no
        information. Default is False.
    chi_square : lib.FloatArray1D, optional
        The per-spot chi-square (residual sum of squares at the fit
        optimum, see ``fit_spot``). If provided, the ``chi_square``
        column is added. It is the least-squares goodness of fit, in
        photons squared, so it scales with the spot brightness and the
        box size and is only comparable between fits of the same box
        size.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    """
    # box_offset = int(box / 2)
    rotated = theta.shape[1] == 7
    x = theta[:, 0] + identifications["x"]  # - box_offset
    y = theta[:, 1] + identifications["y"]  # - box_offset
    lpx = precision.localization_precision(
        theta[:, 2], theta[:, 4], theta[:, 5], theta[:, 3], em=em
    )
    lpy = precision.localization_precision(
        theta[:, 2], theta[:, 5], theta[:, 4], theta[:, 3], em=em
    )
    columns = {
        "frame": identifications["frame"].astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "photons": theta[:, 2].astype(np.float32),
        "sx": theta[:, 4].astype(np.float32),
        "sy": theta[:, 5].astype(np.float32),
        "bg": theta[:, 3].astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
    }
    if not spherical:
        # For a spherical (isotropic) Gaussian sx == sy, so the
        # ellipticity is always 0 and carries no information.
        a = np.maximum(theta[:, 4], theta[:, 5])
        b = np.minimum(theta[:, 4], theta[:, 5])
        ellipticity = (a - b) / a
        columns["ellipticity"] = ellipticity.astype(np.float32)
    columns.update(lib.net_gradient_column(identifications))
    if rotated:
        # Match the fitting subpackage's convention (see
        # localize.locs_from_fits_gauss):
        # negate, convert to degrees and normalize to [-90, 90) since the
        # ellipse repeats every half turn.
        angle = -np.rad2deg(theta[:, 6])
        angle = np.mod(angle + 90.0, 180.0) - 90.0
        columns["angle"] = angle.astype(np.float32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)

    if "n_id" in identifications.columns:
        columns["n_id"] = identifications["n_id"].astype(np.uint32)
        locs = pd.DataFrame(columns)
        locs.sort_values(by="n_id", kind="quicksort", inplace=True)
    else:
        locs = pd.DataFrame(columns)
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def locs_from_fits_gauss_gpu(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
) -> pd.DataFrame:
    """Convert the fit results from GPU-based fitting into a data frame
    of localizations.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Use
        ``picasso.localize.locs_from_fits_gauss``, which this already
        forwards to.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identifications of the spots,
        including frame numbers, x and y coordinates, and net gradient.
    theta : lib.FloatArray2D
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
    locs : pd.DataFrame
        Data frame containing the localized spots.
    """
    from picasso import localize

    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return localize.locs_from_fits_gauss(identifications, theta, box, em)


def localization_precision(
    photons: lib.FloatArray1D,
    s: lib.FloatArray1D,
    s_orth: lib.FloatArray1D,
    bg: lib.FloatArray1D,
    em: bool,
) -> lib.FloatArray1D:
    """Theoretical localization precision of a 2D unweighted Gaussian fit.

    Mortensen et al., Nature Methods, 2010.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Moved verbatim to
        :func:`picasso.fitting.precision.localization_precision`, which this
        now forwards to.

    Parameters
    ----------
    photons : lib.FloatArray1D
        Number of photons collected for the localization.
    s : lib.FloatArray1D
        Size of the single-emitter image for each localization.
    s_orth : lib.FloatArray1D
        Size of the single-emitter image in the orthogonal direction
        for each localization.
    bg : lib.FloatArray1D
        Background signal for each localization (per pixel).
    em : bool
        Whether EMCCD was used for the localization.

    Returns
    -------
    lib.FloatArray1D
        Cramer-Rao lower bound for localization precision for each
        localization.
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return precision.localization_precision(photons, s, s_orth, bg, em)


def sigma_uncertainty(
    sigma: lib.SeriesOrFloatArray1D,
    sigma_orth: lib.SeriesOrFloatArray1D,
    photons: lib.SeriesOrFloatArray1D,
    bg: lib.SeriesOrFloatArray1D,
) -> lib.FloatArray1D:
    """Standard error of a least-squares fitted sigma.

    .. deprecated:: 0.11
        This whole module is removed in Picasso 1.0. Moved verbatim to
        :func:`picasso.fitting.precision.sigma_uncertainty_lsq` - renamed
        because ``picasso.gaussmle`` defined a different formula under this
        name.

    Parameters
    ----------
    sigma : lib.SeriesOrFloatArray1D
        Fitted sigma values in camera pixels.
    sigma_orth : lib.SeriesOrFloatArray1D
        Fitted sigma values in the orthogonal direction in camera
        pixels.
    photons : lib.SeriesOrFloatArray1D
        Number of photons.
    bg : lib.SeriesOrFloatArray1D
        Background photons per pixel.

    Returns
    -------
    se_sigma : lib.FloatArray1D
        Standard error of fitted sigma values in camera pixels.
    """
    lib.deprecation_warning(_DEPRECATION_MESSAGE)
    return precision.sigma_uncertainty_lsq(sigma, sigma_orth, photons, bg)
