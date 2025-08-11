"""
    picasso/gaussmle
    ~~~~~~~~~~~~~~~~

    Maximum likelihood fits for single particle localization. Based on
    Smith, et al. Nature Methods, 2010. 

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations
from typing import Literal

import numpy as _np
import numba as _numba
import math as _math
import multiprocessing as _multiprocessing
import threading as _threading
from concurrent import futures as _futures


GAMMA = _np.array([1.0, 1.0, 0.5, 1.0, 1.0, 1.0])


@_numba.jit(nopython=True, nogil=True)
def _sum_and_center_of_mass(
    spot: _np.ndarray, 
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


@_numba.jit(nopython=True, nogil=True)
def mean_filter(spot: _np.ndarray, size: int) -> _np.ndarray:
    """Apply a mean filter to the spot. This function computes the mean 
    of each pixel in a 3x3 neighborhood.
    
    Parameters
    ----------
    spot : _np.ndarray
        The input image.
    size : int
        The size of the patch (assumed to be square).

    Returns
    -------
    filtered_spot : _np.ndarray
        The filtered image patch.
    """

    filtered_spot = _np.zeros_like(spot)
    for k in range(size):
        for l in range(size):
            min_m = _np.maximum(0, k - 1)
            max_m = _np.minimum(size, k + 2)
            min_n = _np.maximum(0, l - 1)
            max_n = _np.minimum(size, l + 2)
            N = (max_m - min_m) * (max_n - min_n)
            Nsum = 0.0
            for m in range(min_m, max_m):
                for n in range(min_n, max_n):
                    Nsum += spot[m, n]
            filtered_spot[k, l] = Nsum / N
    return filtered_spot


@_numba.jit(nopython=True, nogil=True)
def _initial_sigmas(
    spot: _np.ndarray, 
    y: None, 
    x: None, 
    size: int,
) -> tuple[float, float]:
    """Initialize the sizes of the single-emitter images (sigmas of the
    Gaussian fit) in x and y independently."""

    size_half = int(size / 2)
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    sum_y = 0.0
    sum_x = 0.0
    for i in range(size):
        d2 = (i - size_half) ** 2
        sum_deviation_y += spot[i, size_half] * d2
        sum_deviation_x += spot[size_half, i] * d2
        sum_y += spot[i, size_half]
        sum_x += spot[size_half, i]
    sy = _np.sqrt(sum_deviation_y / sum_y)
    sx = _np.sqrt(sum_deviation_x / sum_x)
    if ~_np.isfinite(sy):
        sy = 0.01
    if ~_np.isfinite(sx):
        sx = 0.01
    if sx == 0:
        sx = 0.01
    if sy == 0:
        sy = 0.01
    return sy, sx


@_numba.jit(nopython=True, nogil=True)
def _initial_parameters(
    spot: _np.ndarray, 
    size: int,
) -> tuple[float, float, float, float, float, float]:
    """Initialize the parameters for the Gaussian fit - x, y, photons,
    background, sigma_x and sigma_y."""

    sum, y, x = _sum_and_center_of_mass(spot, size)
    bg = _np.min(mean_filter(spot, size))
    photons = sum - size * size * bg
    photons_sane = _np.maximum(1.0, photons)
    sy, sx = _initial_sigmas(spot - bg, y, x, size)
    return x, y, photons_sane, bg, sx, sy


@_numba.jit(nopython=True, nogil=True)
def _initial_theta_sigma(spot: _np.ndarray, size: int) -> _np.ndarray:
    """Initialize the parameters for the Gaussian fit with a single 
    sigma for both x and y dimensions - x, y, photons, background, 
    sigma."""

    theta = _np.zeros(5, dtype=_np.float32)
    theta[0], theta[1], theta[2], theta[3], sx, sy = _initial_parameters(spot, size)
    theta[4] = (sx + sy) / 2
    return theta


@_numba.jit(nopython=True, nogil=True)
def _initial_theta_sigmaxy(spot: _np.ndarray, size: int) -> _np.ndarray:
    """Initialize the parameters for the Gaussian fit with separate
    sigmas for x and y dimensions - x, y, photons, background, sigma_x
    and sigma_y."""

    theta = _np.zeros(6, dtype=_np.float32)
    theta[0], theta[1], theta[2], theta[3], theta[4], theta[5] = _initial_parameters(
        spot, size
    )
    return theta


@_numba.vectorize(nopython=True)
def _erf(x: float) -> float:
    """Currently not needed, but might be useful for a CUDA 
    implementation."""
    
    ax = _np.abs(x)
    if ax < 0.5:
        t = x * x
        top = (
            (
                (
                    (0.771058495001320e-04 * t - 0.133733772997339e-02) * t
                    + 0.323076579225834e-01
                )
                * t
                + 0.479137145607681e-01
            )
            * t
            + 0.128379167095513e00
        ) + 1.0
        bot = (
            (0.301048631703895e-02 * t + 0.538971687740286e-01) * t
            + 0.375795757275549e00
        ) * t + 1.0
        return x * (top / bot)
    if ax < 4.0:
        top = (
            (
                (
                    (
                        (
                            (-1.36864857382717e-07 * ax + 5.64195517478974e-01) * ax
                            + 7.21175825088309e00
                        )
                        * ax
                        + 4.31622272220567e01
                    )
                    * ax
                    + 1.52989285046940e02
                )
                * ax
                + 3.39320816734344e02
            )
            * ax
            + 4.51918953711873e02
        ) * ax + 3.00459261020162e02
        bot = (
            (
                (
                    (
                        ((1.0 * ax + 1.27827273196294e01) * ax + 7.70001529352295e01)
                        * ax
                        + 2.77585444743988e02
                    )
                    * ax
                    + 6.38980264465631e02
                )
                * ax
                + 9.31354094850610e02
            )
            * ax
            + 7.90950925327898e02
        ) * ax + 3.00459260956983e02
        erf = 0.5 + (0.5 - _np.exp(-x * x) * top / bot)
        if x < 0.0:
            erf = -erf
        return erf
    if ax < 5.8:
        x2 = x * x
        t = 1.0 / x2
        top = (
            ((2.10144126479064e00 * t + 2.62370141675169e01) * t + 2.13688200555087e01)
            * t
            + 4.65807828718470e00
        ) * t + 2.82094791773523e-01
        bot = (
            ((9.41537750555460e01 * t + 1.87114811799590e02) * t + 9.90191814623914e01)
            * t
            + 1.80124575948747e01
        ) * t + 1.0
        erf = (0.564189583547756e0 - top / (x2 * bot)) / ax
        erf = 0.5 + (0.5 - _np.exp(-x2) * erf)
        if x < 0.0:
            erf = -erf
        return erf
    return _np.sign(x)


@_numba.jit(nopython=True, nogil=True, cache=False)
def _gaussian_integral(x: float, mu: float, sigma: float) -> float: #TODO: check this is correct?
    """Calculate the integral of a Gaussian function from negative 
    infinity to x."""

    sq_norm = 0.70710678118654757 / sigma  # sq_norm = sqrt(0.5/sigma**2)
    d = x - mu
    return 0.5 * (_math.erf((d + 0.5) * sq_norm) - _math.erf((d - 0.5) * sq_norm))


@_numba.jit(nopython=True, nogil=True, cache=False)
def _derivative_gaussian_integral(
    x: float, 
    mu: float, 
    sigma: float, 
    photons: float, 
    PSFc: float,
) -> tuple[float, float]:
    """Calculate the first and second derivatives of the integral of a
    Gaussian function with respect to x."""

    d = x - mu
    a = _np.exp(-0.5 * ((d + 0.5) / sigma) ** 2)
    b = _np.exp(-0.5 * ((d - 0.5) / sigma) ** 2)
    dudt = -photons * PSFc * (a - b) / (_np.sqrt(2.0 * _np.pi) * sigma)
    d2udt2 = (
        -photons
        * ((d + 0.5) * a - (d - 0.5) * b)
        * PSFc
        / (_np.sqrt(2.0 * _np.pi) * sigma**3)
    )
    return dudt, d2udt2


@_numba.jit(nopython=True, nogil=True, cache=False)
def _derivative_gaussian_integral_1d_sigma(
    x: float, 
    mu: float, 
    sigma: float, 
    photons: float, 
    PSFc: float,
) -> tuple[float, float]:
    """Calculate the first and second derivatives of the integral of a
    Gaussian function with respect to sigma in 1D."""

    ax = _np.exp(-0.5 * ((x + 0.5 - mu) / sigma) ** 2)
    bx = _np.exp(-0.5 * ((x - 0.5 - mu) / sigma) ** 2)
    dudt = (
        -photons
        * (ax * (x + 0.5 - mu) - bx * (x - 0.5 - mu))
        * PSFc
        / (_np.sqrt(2.0 * _np.pi) * sigma**2)
    )
    d2udt2 = -2.0 * dudt / sigma - photons * (
        ax * (x + 0.5 - mu) ** 3 - bx * (x - 0.5 - mu) ** 3
    ) * PSFc / (_np.sqrt(2.0 * _np.pi) * sigma**5)
    return dudt, d2udt2


@_numba.jit(nopython=True, nogil=True)
def _derivative_gaussian_integral_2d_sigma(
    x: float, 
    y: float, 
    mu: float, 
    nu: float, 
    sigma: float, 
    photons: float, 
    PSFx: float, 
    PSFy: float
) -> tuple[float, float]:
    """Calculate the first and second derivatives of the integral of a
    Gaussian function with respect to sigma in 2D."""

    dSx, ddSx = _derivative_gaussian_integral_1d_sigma(x, mu, sigma, photons, PSFy)
    dSy, ddSy = _derivative_gaussian_integral_1d_sigma(y, nu, sigma, photons, PSFx)
    dudt = dSx + dSy
    d2udt2 = ddSx + ddSy
    return dudt, d2udt2


def _worker(
    func,
    spots,
    thetas,
    CRLBs,
    likelihoods,
    iterations,
    eps,
    max_it,
    current,
    lock,
):
    """Worker function for asynchronous Gaussian fitting."""
    N = len(spots)
    while True:
        with lock:
            index = current[0]
            if index == N:
                return
            current[0] += 1
        func(spots, index, thetas, CRLBs, likelihoods, iterations, eps, max_it)


def gaussmle(
    spots: _np.ndarray, 
    eps: float, 
    max_it: int, 
    method: Literal["sigma", "sigmaxy"] = "sigma",
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray]:
    """Fits Gaussians using Maximum Likelihood Estimation (MLE) to the
    extracted spots.
    
    Parameters
    ----------
    spots : _np.ndarray
        The input image patches containing the spots of shape 
        (N, size, size), where N is the number of spots and size is the
        size of the square patch.
    eps : float
        The convergence criterion for the fitting algorithm.
    max_it : int
        The maximum number of iterations for the fitting algorithm.
    method : Literal["sigma", "sigmaxy"]
        The method to use for fitting the Gaussian.

    Returns
    -------
    thetas : _np.ndarray
        The fitted parameters for each spot, shape (N, 6) or (N, 5) 
        depending on the method. The columns are x, y, photons, 
        background and sigma (or sigmax, sigmay).
    CRLBs : _np.ndarray
        The Cramer-Rao Lower Bounds for the fitted parameters, shape
        (N, 6) or (N, 5).
    likelihoods : _np.ndarray
        The log-likelihoods for each fitted spot, shape (N,).
    iterations : _np.ndarray
        The number of iterations taken to converge for each spot,
        shape (N,).
    """

    N = len(spots)
    thetas = _np.zeros((N, 6), dtype=_np.float32)
    CRLBs = _np.inf * _np.ones((N, 6), dtype=_np.float32)
    likelihoods = _np.zeros(N, dtype=_np.float32)
    iterations = _np.zeros(N, dtype=_np.int32)
    if method == "sigma":
        func = _mlefit_sigma
    elif method == "sigmaxy":
        func = _mlefit_sigmaxy
    else:
        raise ValueError("Method not available.")
    for i in range(N):
        func(spots, i, thetas, CRLBs, likelihoods, iterations, eps, max_it)
    return thetas, CRLBs, likelihoods, iterations


def gaussmle_async(
    spots: _np.ndarray, 
    eps: float, 
    max_it: int, 
    method: Literal["sigma", "sigmaxy"] = "sigma",
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray, _np.ndarray]:
    """Runs ``gaussmle`` asynchronously (multiprocessing) to fit
    Gaussians using Maximum Likelihood Estimation (MLE) to the
    extracted spots. See ``gaussmle`` for parameter details.
    
    Returns
    -------
    current : _np.ndarray
        A single-element array containing the current index of the
        spot being processed.
    thetas, CRLBs, likelihoods, iterations : _np.ndarrays
        The same as in ``gaussmle``.
    """

    N = len(spots)
    thetas = _np.zeros((N, 6), dtype=_np.float32)
    CRLBs = _np.inf * _np.ones((N, 6), dtype=_np.float32)
    likelihoods = _np.zeros(N, dtype=_np.float32)
    iterations = _np.zeros(N, dtype=_np.int32)
    n_workers = min(
        60, max(1, int(0.75 * _multiprocessing.cpu_count()))
    ) # Python crashes when using >64 cores
    lock = _threading.Lock()
    current = [0]
    if method == "sigma":
        func = _mlefit_sigma
    elif method == "sigmaxy":
        func = _mlefit_sigmaxy
    else:
        raise ValueError("Method not available.")
    executor = _futures.ThreadPoolExecutor(n_workers)
    for i in range(n_workers):
        executor.submit(
            _worker,
            func,
            spots,
            thetas,
            CRLBs,
            likelihoods,
            iterations,
            eps,
            max_it,
            current,
            lock,
        )
    executor.shutdown(wait=False)
    # A synchronous single-threaded version for debugging:
    # for i in range(N):
    #     print('Spot', i)
    #     func(spots, i, thetas, CRLBs, likelihoods, iterations, eps, max_it)
    return current, thetas, CRLBs, likelihoods, iterations


@_numba.jit(nopython=True, nogil=True)
def _mlefit_sigma(
    spots: _np.ndarray, 
    index: int, 
    thetas: _np.ndarray, 
    CRLBs: _np.ndarray, 
    likelihoods: _np.ndarray, 
    iterations: _np.ndarray, 
    eps: float, 
    max_it: int,
) -> None:
    """Fits a Gaussian to a single spot using Maximum Likelihood
    Estimation (MLE) with a single sigma for both x and y dimensions."""

    n_params = 5

    spot = spots[index]
    size, _ = spot.shape

    # theta is [x, y, N, bg, S]
    theta = _initial_theta_sigma(spot, size)
    max_step = _np.zeros(n_params, dtype=_np.float32)
    max_step[0:2] = theta[4]
    max_step[2:4] = 0.1 * theta[2:4]
    max_step[4] = 0.2 * theta[4]

    # Memory allocation
    # (we do that outside of the loops to avoid huge delays in threaded code):
    dudt = _np.zeros(n_params, dtype=_np.float32)
    d2udt2 = _np.zeros(n_params, dtype=_np.float32)
    numerator = _np.zeros(n_params, dtype=_np.float32)
    denominator = _np.zeros(n_params, dtype=_np.float32)

    old_x = theta[0]
    old_y = theta[1]

    kk = 0
    while (
        kk < max_it
    ):  # we do this instead of a for loop for the special case of max_it=0
        kk += 1

        numerator[:] = 0.0
        denominator[:] = 0.0

        for ii in range(size):
            for jj in range(size):
                PSFx = _gaussian_integral(ii, theta[0], theta[4])
                PSFy = _gaussian_integral(jj, theta[1], theta[4])

                # Derivatives
                dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                    ii, theta[0], theta[4], theta[2], PSFy
                )
                dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                    jj, theta[1], theta[4], theta[2], PSFx
                )
                dudt[2] = PSFx * PSFy
                d2udt2[2] = 0.0
                dudt[3] = 1.0
                d2udt2[3] = 0.0
                dudt[4], d2udt2[4] = _derivative_gaussian_integral_2d_sigma(
                    ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy
                )

                model = theta[2] * dudt[2] + theta[3]
                cf = df = 0.0
                data = spot[ii, jj]
                if model > 10e-3:
                    cf = data / model - 1
                    df = data / model**2
                cf = _np.minimum(cf, 10e4)
                df = _np.minimum(df, 10e4)

                for ll in range(n_params):
                    numerator[ll] += cf * dudt[ll]
                    denominator[ll] += cf * d2udt2[ll] - df * dudt[ll] ** 2

        # The update
        for ll in range(n_params):
            if denominator[ll] == 0.0:
                update = _np.sign(numerator[ll] * max_step[ll])
            else:
                update = _np.minimum(
                    _np.maximum(numerator[ll] / denominator[ll], -max_step[ll]),
                    max_step[ll],
                )
            if kk < 5:
                update *= GAMMA[ll]
            theta[ll] -= update

        # Other constraints
        theta[2] = _np.maximum(theta[2], 1.0)
        theta[3] = _np.maximum(theta[3], 0.01)
        theta[4] = _np.maximum(theta[4], 0.01)
        theta[4] = _np.minimum(theta[4], size)

        # Check for convergence
        if (_np.abs(old_x - theta[0]) < eps) and (_np.abs(old_y - theta[1]) < eps):
            break
        else:
            old_x = theta[0]
            old_y = theta[1]

    thetas[index, 0:5] = theta
    thetas[index, 5] = theta[4]
    iterations[index] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    M = _np.zeros((n_params, n_params), dtype=_np.float32)
    for ii in range(size):
        for jj in range(size):
            PSFx = _gaussian_integral(ii, theta[0], theta[4])
            PSFy = _gaussian_integral(jj, theta[1], theta[4])
            model = theta[3] + theta[2] * PSFx * PSFy

            # Calculating derivatives
            dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                ii, theta[0], theta[4], theta[2], PSFy
            )
            dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                jj, theta[1], theta[4], theta[2], PSFx
            )
            dudt[4], d2udt2[4] = _derivative_gaussian_integral_2d_sigma(
                ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy
            )
            dudt[2] = PSFx * PSFy
            dudt[3] = 1.0

            # Building the Fisher Information Matrix
            model = theta[3] + theta[2] * dudt[2]
            for kk in range(n_params):
                for ll in range(kk, n_params):
                    M[kk, ll] += dudt[ll] * dudt[kk] / model
                    M[ll, kk] = M[kk, ll]

            # LogLikelihood
            if model > 0:
                data = spot[ii, jj]
                if data > 0:
                    Div += data * _np.log(model) - model - data * _np.log(data) + data
                else:
                    Div += -model

    likelihoods[index] = Div

    # Matrix inverse (CRLB=F^-1)
    Minv = _np.linalg.pinv(M)
    CRLB = _np.zeros(n_params, dtype=_np.float32)
    for kk in range(n_params):
        CRLB[kk] = Minv[kk, kk]
    CRLBs[index, 0:5] = CRLB
    CRLBs[index, 5] = CRLB[4]


@_numba.jit(nopython=True, nogil=True)
def _mlefit_sigmaxy(
    spots: _np.ndarray, 
    index: int, 
    thetas: _np.ndarray, 
    CRLBs: _np.ndarray, 
    likelihoods: _np.ndarray, 
    iterations: _np.ndarray, 
    eps: float,
    max_it: int,
) -> None:
    """Fits a Gaussian to a single spot using Maximum Likelihood
    Estimation (MLE) with separate sigmas for x and y dimensions."""

    n_params = 6

    spot = spots[index]
    size, _ = spot.shape

    # Initial values
    # theta is [x, y, N, bg, Sx, Sy]
    theta = _initial_theta_sigmaxy(spot, size)
    max_step = _np.zeros(n_params, dtype=_np.float32)
    max_step[0:2] = theta[4]
    max_step[2:4] = 0.1 * theta[2:4]
    max_step[4:6] = 0.2 * theta[4:6]

    # Memory allocation
    # (we do that outside of the loops to avoid huge delays in threaded code):
    dudt = _np.zeros(n_params, dtype=_np.float32)
    d2udt2 = _np.zeros(n_params, dtype=_np.float32)
    numerator = _np.zeros(n_params, dtype=_np.float32)
    denominator = _np.zeros(n_params, dtype=_np.float32)

    old_x = theta[0]
    old_y = theta[1]
    old_sx = theta[4]
    old_sy = theta[5]

    kk = 0
    while (
        kk < max_it
    ):  # we do this instead of a for loop for the special case of max_it=0
        kk += 1

        numerator[:] = 0.0
        denominator[:] = 0.0

        for ii in range(size):
            for jj in range(size):
                PSFx = _gaussian_integral(ii, theta[0], theta[4])
                PSFy = _gaussian_integral(jj, theta[1], theta[5])
                # Derivatives
                dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                    ii, theta[0], theta[4], theta[2], PSFy
                )
                dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                    jj, theta[1], theta[5], theta[2], PSFx
                )
                dudt[2] = PSFx * PSFy
                d2udt2[2] = 0.0
                dudt[3] = 1.0
                d2udt2[3] = 0.0
                dudt[4], d2udt2[4] = _derivative_gaussian_integral_1d_sigma(
                    ii, theta[0], theta[4], theta[2], PSFy
                )
                dudt[5], d2udt2[5] = _derivative_gaussian_integral_1d_sigma(
                    jj, theta[1], theta[5], theta[2], PSFx
                )

                model = theta[2] * dudt[2] + theta[3]
                cf = df = 0.0
                data = spot[ii, jj]
                if model > 10e-3:
                    cf = data / model - 1
                    df = data / model**2
                cf = _np.minimum(cf, 10e4)
                df = _np.minimum(df, 10e4)

                for ll in range(n_params):
                    numerator[ll] += cf * dudt[ll]
                    denominator[ll] += cf * d2udt2[ll] - df * dudt[ll] ** 2

        # The update
        for ll in range(n_params):
            if denominator[ll] == 0.0:
                # This is case is not handled in Lidke's code
                # but it seems to be a problem here
                # (maybe due to many iterations)
                theta[ll] -= GAMMA[ll] * _np.sign(numerator[ll]) * max_step[ll]
            else:
                theta[ll] -= GAMMA[ll] * _np.minimum(
                    _np.maximum(numerator[ll] / denominator[ll], -max_step[ll]),
                    max_step[ll],
                )

        # Other constraints
        theta[2] = _np.maximum(theta[2], 1.0)
        theta[3] = _np.maximum(theta[3], 0.01)
        theta[4] = _np.maximum(theta[4], 0.01)
        theta[5] = _np.maximum(theta[5], 0.01)

        # Check for convergence
        if _np.abs(old_x - theta[0]) < eps:
            if _np.abs(old_y - theta[1]) < eps:
                if _np.abs(old_sx - theta[4]) < eps:
                    if _np.abs(old_sy - theta[5]) < eps:
                        break
        old_x = theta[0]
        old_y = theta[1]
        old_sx = theta[4]
        old_sy = theta[5]
    thetas[index] = theta
    iterations[index] = kk

    # Calculating the CRLB and LogLikelihood
    Div = 0.0
    M = _np.zeros((n_params, n_params), dtype=_np.float32)
    for ii in range(size):
        for jj in range(size):
            PSFx = _gaussian_integral(ii, theta[0], theta[4])
            PSFy = _gaussian_integral(jj, theta[1], theta[5])
            model = theta[3] + theta[2] * PSFx * PSFy

            # Calculating derivatives
            dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                ii, theta[0], theta[4], theta[2], PSFy
            )
            dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                jj, theta[1], theta[5], theta[2], PSFx
            )
            dudt[4], d2udt2[4] = _derivative_gaussian_integral_1d_sigma(
                ii, theta[0], theta[4], theta[2], PSFy
            )
            dudt[5], d2udt2[5] = _derivative_gaussian_integral_1d_sigma(
                jj, theta[1], theta[5], theta[2], PSFx
            )
            dudt[2] = PSFx * PSFy
            dudt[3] = 1.0

            # Building the Fisher Information Matrix
            model = theta[3] + theta[2] * dudt[2]
            for kk in range(n_params):
                for ll in range(kk, n_params):
                    M[kk, ll] += dudt[ll] * dudt[kk] / model
                    M[ll, kk] = M[kk, ll]

            # LogLikelihood
            if model > 0:
                data = spot[ii, jj]
                if data > 0:
                    Div += data * _np.log(model) - model - data * _np.log(data) + data
                else:
                    Div += -model

    likelihoods[index] = Div

    # Matrix inverse (CRLB=F^-1)
    Minv = _np.linalg.pinv(M)
    CRLB = _np.zeros(n_params, dtype=_np.float32)
    for kk in range(n_params):
        CRLB[kk] = Minv[kk, kk]
    CRLBs[index] = CRLB


def locs_from_fits(
    identifications: _np.recarray, 
    theta: _np.ndarray, 
    CRLBs: _np.ndarray, 
    likelihoods: _np.ndarray, 
    iterations: _np.ndarray, 
    box: int,
) -> _np.recarray:
    """Converts the results of Gaussian fits into a structured array
    suitable for further analysis or visualization.
    
    Parameters
    ----------
    identifications : _np.recarray
        The structured array containing the identifications of the 
        spots, which should include 'frame', 'x', 'y' and 
        'net_gradient'.
    theta : _np.ndarray
        The fitted parameters for each spot, shape (N, 6) or (N, 5) 
        depending on the method used.
    CRLBs : _np.ndarray
        The Cramer-Rao Lower Bounds for the fitted parameters, shape
        (N, 6) or (N, 5).
    likelihoods : _np.ndarray
        The log-likelihoods for each fitted spot, shape (N,).  
    iterations : _np.ndarray
        The number of iterations taken to converge for each spot,
        shape (N,).
    box : int
        The size of the box used for fitting, which is used to
        calculate the offsets for the x and y coordinates.
    
    Returns
    -------
    locs : _np.recarray
        A structured array containing the fitted parameters and
        additional information for each spot, including frame, x, y,
        photons, sigma_x, sigma_y, background, localization precision
        (lpx, lpy), ellipticity, net gradient and identification ID
        (if available).
    """

    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    with _np.errstate(invalid="ignore"):
        lpy = _np.sqrt(CRLBs[:, 0])
        lpx = _np.sqrt(CRLBs[:, 1])
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
