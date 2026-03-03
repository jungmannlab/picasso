"""
picasso.gaussmle
~~~~~~~~~~~~~~~~

Maximum likelihood fits for single particle localization. Based on
Smith, et al. Nature Methods, 2010.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
:copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
import multiprocessing
import threading
from concurrent import futures
from typing import Literal

import numba
import numpy as np
import pandas as pd


@numba.jit(nopython=True, nogil=True)
def _sum_and_center_of_mass(
    spot: np.ndarray,
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
    y /= _sum_
    x /= _sum_
    return _sum_, y, x


@numba.jit(nopython=True, nogil=True)
def mean_filter(spot: np.ndarray, size: int) -> np.ndarray:
    """Apply a mean filter to the spot. This function computes the mean
    of each pixel in a 3x3 neighborhood.

    Parameters
    ----------
    spot : np.ndarray
        The input image.
    size : int
        The size of the patch (assumed to be square).

    Returns
    -------
    filtered_spot : np.ndarray
        The filtered image patch.
    """
    filtered_spot = np.zeros_like(spot)
    for k in range(size):
        for ll in range(size):
            min_m = np.maximum(0, k - 1)
            max_m = np.minimum(size, k + 2)
            min_n = np.maximum(0, ll - 1)
            max_n = np.minimum(size, ll + 2)
            N = (max_m - min_m) * (max_n - min_n)
            Nsum = 0.0
            for m in range(min_m, max_m):
                for n in range(min_n, max_n):
                    Nsum += spot[m, n]
            filtered_spot[k, ll] = Nsum / N
    return filtered_spot


@numba.jit(nopython=True, nogil=True)
def _initial_sigmas(
    spot: np.ndarray,
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
    sy = np.sqrt(sum_deviation_y / sum_y)
    sx = np.sqrt(sum_deviation_x / sum_x)
    if ~np.isfinite(sy):
        sy = 0.01
    if ~np.isfinite(sx):
        sx = 0.01
    if sx == 0:
        sx = 0.01
    if sy == 0:
        sy = 0.01
    return sy, sx


@numba.jit(nopython=True, nogil=True)
def _initial_parameters(
    spot: np.ndarray,
    size: int,
) -> tuple[float, float, float, float, float, float]:
    """Initialize the parameters for the Gaussian fit - x, y, photons,
    background, sigma_x and sigma_y."""
    sum, y, x = _sum_and_center_of_mass(spot, size)
    bg = np.min(mean_filter(spot, size))
    photons = sum - size * size * bg
    photons_sane = np.maximum(1.0, photons)
    sy, sx = _initial_sigmas(spot - bg, y, x, size)
    return x, y, photons_sane, bg, sx, sy


@numba.jit(nopython=True, nogil=True)
def _initial_theta_sigma(spot: np.ndarray, size: int) -> np.ndarray:
    """Initialize the parameters for the Gaussian fit with a single
    sigma for both x and y dimensions - x, y, photons, background,
    sigma."""
    theta = np.zeros(5, dtype=np.float32)
    theta[0], theta[1], theta[2], theta[3], sx, sy = _initial_parameters(
        spot, size
    )
    theta[4] = (sx + sy) / 2
    return theta


@numba.jit(nopython=True, nogil=True)
def _initial_theta_sigmaxy(spot: np.ndarray, size: int) -> np.ndarray:
    """Initialize the parameters for the Gaussian fit with separate
    sigmas for x and y dimensions - x, y, photons, background, sigma_x
    and sigma_y."""
    theta = np.zeros(6, dtype=np.float32)
    theta[0], theta[1], theta[2], theta[3], theta[4], theta[5] = (
        _initial_parameters(spot, size)
    )
    return theta


@numba.vectorize(nopython=True)
def _erf(x: float) -> float:
    """Currently not needed, but might be useful for a CUDA
    implementation."""
    ax = np.abs(x)
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
                            (-1.36864857382717e-07 * ax + 5.64195517478974e-01)
                            * ax
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
                        (
                            (1.0 * ax + 1.27827273196294e01) * ax
                            + 7.70001529352295e01
                        )
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
        erf = 0.5 + (0.5 - np.exp(-x * x) * top / bot)
        if x < 0.0:
            erf = -erf
        return erf
    if ax < 5.8:
        x2 = x * x
        t = 1.0 / x2
        top = (
            (
                (2.10144126479064e00 * t + 2.62370141675169e01) * t
                + 2.13688200555087e01
            )
            * t
            + 4.65807828718470e00
        ) * t + 2.82094791773523e-01
        bot = (
            (
                (9.41537750555460e01 * t + 1.87114811799590e02) * t
                + 9.90191814623914e01
            )
            * t
            + 1.80124575948747e01
        ) * t + 1.0
        erf = (0.564189583547756e0 - top / (x2 * bot)) / ax
        erf = 0.5 + (0.5 - np.exp(-x2) * erf)
        if x < 0.0:
            erf = -erf
        return erf
    return np.sign(x)


@numba.jit(nopython=True, nogil=True, cache=False)
def _gaussian_integral(x: float, mu: float, sigma: float) -> float:
    """Calculate the integral of a Gaussian function in a pixel in one
    dimension (deltaE, equations 4a,b in Smith, et al (supplement)).

    Note that the paper gives a wrong formula, the denominator within
    ERF should be sqrt(2) * sigma, not sigma**2. This has been corrected
    here."""
    sq_norm = 0.70710678118654757 / sigma  # sq_norm = sqrt(0.5/sigma**2)
    d = x - mu
    return 0.5 * (
        math.erf((d + 0.5) * sq_norm) - math.erf((d - 0.5) * sq_norm)
    )


@numba.jit(nopython=True, nogil=True, cache=False)
def _derivative_gaussian_integral(
    x: float,  # x_k
    mu: float,  # theta_x
    sigma: float,
    photons: float,  # theta_I_0
    PSFy: float,  # delta_E_y
) -> tuple[float, float]:
    """Calculate the first and second derivatives of the integral of
    mu_k w.r.t theta_x, see equations 11a and 14a."""
    d = x - mu
    a = np.exp(-0.5 * ((d + 0.5) / sigma) ** 2)
    b = np.exp(-0.5 * ((d - 0.5) / sigma) ** 2)
    dudt = photons * PSFy * (b - a) / (np.sqrt(2.0 * np.pi) * sigma)
    d2udt2 = (
        photons
        * ((d - 0.5) * b - (d + 0.5) * a)
        * PSFy
        / (np.sqrt(2.0 * np.pi) * sigma**3)
    )
    return dudt, d2udt2


@numba.jit(nopython=True, nogil=True, cache=False)
def _G(n, m, x, mu, sigma_x):
    """Helper function for finding derivatives of the model w.r.t sigma
    in the anisotropic case, see equation 20a."""
    a_minus = x - mu - 0.5
    a_plus = x - mu + 0.5
    exp_minus = np.exp(-(a_minus**2) / (2 * sigma_x**2))
    exp_plus = np.exp(-(a_plus**2) / (2 * sigma_x**2))
    return (a_minus**m * exp_minus - a_plus**m * exp_plus) / (
        sigma_x**n * np.sqrt(2 * np.pi)
    )


@numba.jit(nopython=True, nogil=True, cache=False)
def _derivative_gaussian_integral_sigma(
    x: float,  # x_k
    mu: float,  # theta_x
    sigma_x: float,
    photons: float,  # theta_I_0
    PSFy: float,  # delta_E_y
) -> tuple[float, float]:
    """Used for calculating the first and second derivatives of the
    integral of mu_k w.r.t sigma in the anisotropic case, sigma_x !=
    sigma_y. Based on equations 21a and 21b."""
    dudt = photons * PSFy * _G(2, 1, x, mu, sigma_x)
    d2udt2 = (
        photons
        * PSFy
        * (_G(5, 3, x, mu, sigma_x) - 2 * _G(3, 1, x, mu, sigma_x))
    )
    return dudt, d2udt2


@numba.jit(nopython=True, nogil=True)
def _derivative_gaussian_integral_iso_sigma(
    x: float,  # x_k
    y: float,  # y_k
    mu: float,  # theta_x
    nu: float,  # theta_y
    sigma: float,
    photons: float,  # theta_I_0
    PSFx: float,  # delta_E_x
    PSFy: float,  # delta_E_y
) -> tuple[float, float]:
    """Calculate the first and second derivatives of the integral of
    mu_k w.r.t sigma for the case of isotropic sigma. While Smith et al
    do not provide the formula, it can be easily derived, similarly to
    equations 10, 11 14 and 21."""
    a_plus = (x - mu + 0.5) / (np.sqrt(2.0) * sigma)
    a_minus = (x - mu - 0.5) / (np.sqrt(2.0) * sigma)
    b_plus = (y - nu + 0.5) / (np.sqrt(2.0) * sigma)
    b_minus = (y - nu - 0.5) / (np.sqrt(2.0) * sigma)

    Fx = a_minus * np.exp(-(a_minus**2)) - a_plus * np.exp(-(a_plus**2))
    Fy = b_minus * np.exp(-(b_minus**2)) - b_plus * np.exp(-(b_plus**2))
    dPSFxdt = Fx / (np.sqrt(np.pi) * sigma)
    dPSFydt = Fy / (np.sqrt(np.pi) * sigma)

    dFxdt = (
        a_plus * np.exp(-(a_plus**2)) * (1 - 2 * a_plus**2)
        - a_minus * np.exp(-(a_minus**2)) * (1 - 2 * a_minus**2)
    ) / sigma
    dFydy = (
        b_plus * np.exp(-(b_plus**2)) * (1 - 2 * b_plus**2)
        - b_minus * np.exp(-(b_minus**2)) * (1 - 2 * b_minus**2)
    ) / sigma
    d2PSFxdt2 = (1 / np.sqrt(np.pi)) * (
        (-Fx / sigma**2) + sigma ** (-1) * dFxdt
    )
    d2PSFydt2 = (1 / np.sqrt(np.pi)) * (
        (-Fy / sigma**2) + sigma ** (-1) * dFydy
    )

    dudt = photons * (PSFy * dPSFxdt + PSFx * dPSFydt)
    d2udt2 = (
        photons * PSFy * d2PSFxdt2 + 2 * dPSFxdt * dPSFydt + PSFx * d2PSFydt2
    )
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
    spots: np.ndarray,
    eps: float,
    max_it: int,
    method: Literal["sigma", "sigmaxy"] = "sigmaxy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits Gaussians using Maximum Likelihood Estimation (MLE) to the
    extracted spots.

    Parameters
    ----------
    spots : np.ndarray
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
    thetas : np.ndarray
        The fitted parameters for each spot, shape (N, 6) or (N, 5)
        depending on the method. The columns are x, y, photons,
        background and sigma (or sigmax, sigmay).
    CRLBs : np.ndarray
        The Cramer-Rao Lower Bounds for the fitted parameters, shape
        (N, 6) or (N, 5).
    likelihoods : np.ndarray
        The log-likelihoods for each fitted spot, shape (N,).
    iterations : np.ndarray
        The number of iterations taken to converge for each spot,
        shape (N,).
    """
    N = len(spots)
    thetas = np.zeros((N, 6), dtype=np.float32)
    CRLBs = np.inf * np.ones((N, 6), dtype=np.float32)
    likelihoods = np.zeros(N, dtype=np.float32)
    iterations = np.zeros(N, dtype=np.int32)
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
    spots: np.ndarray,
    eps: float,
    max_it: int,
    method: Literal["sigma", "sigmaxy"] = "sigmaxy",
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs ``gaussmle`` asynchronously (multiprocessing) to fit
    Gaussians using Maximum Likelihood Estimation (MLE) to the
    extracted spots. See ``gaussmle`` for parameter details.

    Returns
    -------
    current : np.ndarray
        A single-element array containing the current index of the
        spot being processed.
    thetas, CRLBs, likelihoods, iterations : np.ndarrays
        The same as in ``gaussmle``.
    """
    N = len(spots)
    thetas = np.zeros((N, 6), dtype=np.float32)
    CRLBs = np.inf * np.ones((N, 6), dtype=np.float32)
    likelihoods = np.zeros(N, dtype=np.float32)
    iterations = np.zeros(N, dtype=np.int32)
    n_workers = min(
        60, max(1, int(0.75 * multiprocessing.cpu_count()))
    )  # Python crashes when using >64 cores
    lock = threading.Lock()
    current = [0]
    if method == "sigma":
        func = _mlefit_sigma
    elif method == "sigmaxy":
        func = _mlefit_sigmaxy
    else:
        raise ValueError("Method not available.")
    executor = futures.ThreadPoolExecutor(n_workers)
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
    return current, thetas, CRLBs, likelihoods, iterations


@numba.jit(nopython=True, nogil=True)
def _mlefit_sigma(
    spots: np.ndarray,
    index: int,
    thetas: np.ndarray,
    CRLBs: np.ndarray,
    likelihoods: np.ndarray,
    iterations: np.ndarray,
    eps: float,
    max_it: int,
) -> None:
    """Fits a Gaussian to a single spot using Maximum Likelihood
    Estimation (MLE) with a single sigma for both x and y dimensions.

    Based on the work of Smith, et al. Nature Methods, 2010. The
    equations mentioned below refer to the supplementary information of
    that paper."""
    n_params = 5

    spot = spots[index]
    size, _ = spot.shape

    # theta is [x, y, N, bg, S]
    theta = _initial_theta_sigma(spot, size)
    # Set maximum iteration for each parameter
    max_step = np.zeros(n_params, dtype=np.float32)
    max_step[0:2] = theta[4]
    max_step[2:4] = 0.1 * theta[2:4]
    max_step[4] = 0.2 * theta[4]

    # Memory allocation
    # (we do that outside of the loops to avoid huge delays in threaded code):
    dudt = np.zeros(n_params, dtype=np.float32)
    d2udt2 = np.zeros(n_params, dtype=np.float32)
    numerator = np.zeros(n_params, dtype=np.float32)
    denominator = np.zeros(n_params, dtype=np.float32)

    old_x = theta[0]
    old_y = theta[1]

    kk = 0
    while (
        kk < max_it
    ):  # We do this instead of a for loop for the special case of max_it=0
        kk += 1

        numerator[:] = 0.0
        denominator[:] = 0.0

        # At each iteration (theta update) we sum across all pixels in the spot,
        # see equation 13
        for ii in range(size):
            for jj in range(size):
                # this is delta E_x
                PSFx = _gaussian_integral(ii, theta[0], theta[4])
                # this is delta E_y
                PSFy = _gaussian_integral(jj, theta[1], theta[4])

                # Partial derivatives (PDs) (first and second order) of mu_k
                # with respect to each of the model parameters (x, y, N,
                # bg, sigma), used in equation 13
                dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                    ii,
                    theta[0],
                    theta[4],
                    theta[2],
                    PSFy,
                )  # PDs with respect to theta_x
                dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                    jj, theta[1], theta[4], theta[2], PSFx
                )  # PDs with respect to theta_y
                dudt[2] = PSFx * PSFy  # PDs w.r.t theta_I_0 (photons)
                d2udt2[2] = 0.0
                dudt[3] = 1.0  # PDs w.r.t theta_bg (background)
                d2udt2[3] = 0.0
                dudt[4], d2udt2[4] = _derivative_gaussian_integral_iso_sigma(
                    ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy
                )  # PDs w.r.t theta_sigma; note that the paper does not
                # give an explicit formula for this but it can be derived
                # fairly easily following the logic of equations 10, 11
                # and 14

                # equation 2, model := mu_k
                model = theta[2] * PSFx * PSFy + theta[3]
                cf = df = 0.0
                data = spot[jj, ii]  # data := x_k
                if model > 10e-3:
                    cf = data / model - 1  # cf := (x_k / mu_k - 1) (eq 13)
                    df = data / model**2  # df := x_k / mu_k^2 (eq 13)
                cf = np.minimum(cf, 10e4)
                df = np.minimum(df, 10e4)

                for ll in range(n_params):
                    numerator[ll] += cf * dudt[ll]
                    denominator[ll] += cf * d2udt2[ll] - df * dudt[ll] ** 2

        # The theta update
        for ll in range(n_params):
            if denominator[ll] == 0.0:
                update = np.sign(numerator[ll] * max_step[ll])
            else:
                update = np.minimum(
                    np.maximum(numerator[ll] / denominator[ll], -max_step[ll]),
                    max_step[ll],
                )
            theta[ll] -= update

        # Other constraints
        theta[2] = np.maximum(theta[2], 1.0)
        theta[3] = np.maximum(theta[3], 0.01)
        theta[4] = np.maximum(theta[4], 0.01)
        theta[4] = np.minimum(theta[4], size)

        # Check for convergence
        if (np.abs(old_x - theta[0]) < eps) and (
            np.abs(old_y - theta[1]) < eps
        ):
            break
        else:
            old_x = theta[0]
            old_y = theta[1]

    # Fitting is finished here, we save the results in the output arrays
    thetas[index, 0:5] = theta
    thetas[index, 5] = theta[4]
    iterations[index] = kk

    # Calculating the CRLB and log-likelihood
    log_likelihood = 0.0
    M = np.zeros((n_params, n_params), dtype=np.float32)  # Fisher matrix
    # Sum over all pixels
    for ii in range(size):
        for jj in range(size):
            PSFx = _gaussian_integral(ii, theta[0], theta[4])
            PSFy = _gaussian_integral(jj, theta[1], theta[4])

            # Calculating derivatives (only first order is needed for
            # CRLB)
            dudt[0], _ = _derivative_gaussian_integral(
                ii, theta[0], theta[4], theta[2], PSFy
            )
            dudt[1], _ = _derivative_gaussian_integral(
                jj, theta[1], theta[4], theta[2], PSFx
            )
            dudt[4], _ = _derivative_gaussian_integral_iso_sigma(
                ii, jj, theta[0], theta[1], theta[4], theta[2], PSFx, PSFy
            )
            dudt[2] = PSFx * PSFy
            dudt[3] = 1.0

            # Building the Fisher Information Matrix
            model = theta[2] * PSFx * PSFy + theta[3]  # model := mu_k
            for kk in range(n_params):
                for ll in range(kk, n_params):
                    M[kk, ll] += dudt[ll] * dudt[kk] / model
                    M[ll, kk] = M[kk, ll]

            # log-likelihood, see equation 7 + Stirling approximation
            if model > 0:
                data = spot[jj, ii]
                if data > 0:
                    log_likelihood += (
                        data * np.log(model)
                        - model
                        - data * np.log(data)
                        + data
                    )
                else:
                    log_likelihood += -model

    likelihoods[index] = log_likelihood

    # Matrix inverse (CRLB = M^-1)
    Minv = np.linalg.pinv(M)
    CRLB = np.zeros(n_params, dtype=np.float32)
    for kk in range(n_params):
        CRLB[kk] = Minv[kk, kk]
    CRLBs[index, 0:5] = CRLB
    CRLBs[index, 5] = CRLB[4]


@numba.jit(nopython=True, nogil=True)
def _mlefit_sigmaxy(
    spots: np.ndarray,
    index: int,
    thetas: np.ndarray,
    CRLBs: np.ndarray,
    likelihoods: np.ndarray,
    iterations: np.ndarray,
    eps: float,
    max_it: int,
) -> None:
    """Fit a Gaussian to a single spot using Maximum Likelihood
    Estimation (MLE) with separate sigmas for x and y dimensions.

    Based on the work of Smith, et al. Nature Methods, 2010. The
    equations mentioned below refer to the supplementary information of
    that paper."""
    n_params = 6

    spot = spots[index]
    size, _ = spot.shape

    # theta is [x, y, N, bg, Sx (sigma_x), Sy (sigma_y)]
    theta = _initial_theta_sigmaxy(spot, size)
    # Set maximum iteration for each parameter
    max_step = np.zeros(n_params, dtype=np.float32)
    max_step[0:2] = theta[4]
    max_step[2:4] = 0.1 * theta[2:4]
    max_step[4:6] = 0.2 * theta[4:6]

    # Memory allocation
    # (we do that outside of the loops to avoid huge delays in threaded code):
    dudt = np.zeros(n_params, dtype=np.float32)
    d2udt2 = np.zeros(n_params, dtype=np.float32)
    numerator = np.zeros(n_params, dtype=np.float32)
    denominator = np.zeros(n_params, dtype=np.float32)

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

        # At each iteration (theta update) we sum across all pixels in the spot,
        # see equation 13
        for ii in range(size):
            for jj in range(size):
                # delta_Ex and delta_Ey
                PSFx = _gaussian_integral(ii, theta[0], theta[4])
                PSFy = _gaussian_integral(jj, theta[1], theta[5])

                # Partial derivatives (PDs) (first and second order) of mu_k
                # with respect to each of the model parameters (x, y, N,
                # bg, sigma), used in equation 13
                dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                    ii, theta[0], theta[4], theta[2], PSFy
                )  # PDs with respect to theta_x
                dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                    jj, theta[1], theta[5], theta[2], PSFx
                )  # PDs with respect to theta_y
                dudt[2] = PSFx * PSFy  # PDs w.r.t theta_I_0 (photons)
                d2udt2[2] = 0.0
                dudt[3] = 1.0  # PDs w.r.t theta_bg (background)
                d2udt2[3] = 0.0
                dudt[4], d2udt2[4] = _derivative_gaussian_integral_sigma(
                    ii, theta[0], theta[4], theta[2], PSFy
                )
                dudt[5], d2udt2[5] = _derivative_gaussian_integral_sigma(
                    jj, theta[1], theta[5], theta[2], PSFx
                )  # PDs w.r.t sigma_y; note that the paper does not
                # give an explicit formula for this but it can be derived
                # fairly easily following the logic of equations 10, 11
                # and 14

                # equation 2, model := mu_k
                model = theta[2] * PSFx * PSFy + theta[3]
                cf = df = 0.0
                data = spot[jj, ii]  # data := x_k
                if model > 10e-3:
                    cf = data / model - 1  # cf := (x_k / mu_k - 1) (eq 13)
                    df = data / model**2  # df := x_k / mu_k^2 (eq 13)
                cf = np.minimum(cf, 10e4)
                df = np.minimum(df, 10e4)

                for ll in range(n_params):
                    numerator[ll] += cf * dudt[ll]
                    denominator[ll] += cf * d2udt2[ll] - df * dudt[ll] ** 2

        # The theta update
        for ll in range(n_params):
            if denominator[ll] == 0.0:
                # This is case is not handled in Lidke's code
                # but it seems to be a problem here
                # (maybe due to many iterations)
                theta[ll] -= np.sign(numerator[ll]) * max_step[ll]
            else:
                theta[ll] -= np.minimum(
                    np.maximum(numerator[ll] / denominator[ll], -max_step[ll]),
                    max_step[ll],
                )

        # Other constraints
        theta[2] = np.maximum(theta[2], 1.0)
        theta[3] = np.maximum(theta[3], 0.01)
        theta[4] = np.maximum(theta[4], 0.01)
        theta[5] = np.maximum(theta[5], 0.01)

        # Check for convergence
        if np.abs(old_x - theta[0]) < eps:
            if np.abs(old_y - theta[1]) < eps:
                if np.abs(old_sx - theta[4]) < eps:
                    if np.abs(old_sy - theta[5]) < eps:
                        break
        old_x = theta[0]
        old_y = theta[1]
        old_sx = theta[4]
        old_sy = theta[5]

    # Fitting is finished here, we save the results in the output arrays
    thetas[index] = theta
    iterations[index] = kk

    # Calculating the CRLB and log-likelihood
    log_likelihood = 0.0
    M = np.zeros((n_params, n_params), dtype=np.float32)
    for ii in range(size):
        for jj in range(size):
            PSFx = _gaussian_integral(ii, theta[0], theta[4])
            PSFy = _gaussian_integral(jj, theta[1], theta[5])

            # Calculating derivatives (only first order is needed for
            # CRLB)
            dudt[0], d2udt2[0] = _derivative_gaussian_integral(
                ii, theta[0], theta[4], theta[2], PSFy
            )
            dudt[1], d2udt2[1] = _derivative_gaussian_integral(
                jj, theta[1], theta[5], theta[2], PSFx
            )
            dudt[4], d2udt2[4] = _derivative_gaussian_integral_sigma(
                ii, theta[0], theta[4], theta[2], PSFy
            )
            dudt[5], d2udt2[5] = _derivative_gaussian_integral_sigma(
                jj, theta[1], theta[5], theta[2], PSFx
            )
            dudt[2] = PSFx * PSFy
            dudt[3] = 1.0

            # Building the Fisher Information Matrix
            model = theta[2] * PSFx * PSFy + theta[3]  # model := mu_k
            for kk in range(n_params):
                for ll in range(kk, n_params):
                    M[kk, ll] += dudt[ll] * dudt[kk] / model
                    M[ll, kk] = M[kk, ll]

            # log-likelihood, see equation 7 + Stirling approximation
            if model > 0:
                data = spot[jj, ii]
                if data > 0:
                    log_likelihood += (
                        data * np.log(model)
                        - model
                        - data * np.log(data)
                        + data
                    )
                else:
                    log_likelihood += -model

    likelihoods[index] = log_likelihood

    # Matrix inverse (CRLB=M^-1)
    Minv = np.linalg.pinv(M)
    CRLB = np.zeros(n_params, dtype=np.float32)
    for kk in range(n_params):
        CRLB[kk] = Minv[kk, kk]
    CRLBs[index] = CRLB


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: np.ndarray,
    CRLBs: np.ndarray,
    log_likelihoods: np.ndarray,
    iterations: np.ndarray,
    box: int,
) -> pd.DataFrame:
    """Convert the results of Gaussian fits into a data frame array
    suitable for further analysis or visualization.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identifications of the
        spots, which should include 'frame', 'x', 'y' and
        'net_gradient'.
    theta : np.ndarray
        The fitted parameters for each spot, shape (N, 6) or (N, 5)
        depending on the method used.
    CRLBs : np.ndarray
        The Cramer-Rao Lower Bounds for the fitted parameters, shape
        (N, 6) or (N, 5).
    likelihoods : np.ndarray
        The log-likelihoods for each fitted spot, shape (N,).
    iterations : np.ndarray
        The number of iterations taken to converge for each spot,
        shape (N,).
    box : int
        The size of the box used for fitting, which is used to
        calculate the offsets for the x and y coordinates.

    Returns
    -------
    locs : pd.DataFrame
        DataFrame containing the fitted parameters and additional
        information for each spot, including frame, x, y, photons,
        sigma_x, sigma_y, background, localization precision
        (lpx, lpy), ellipticity, net gradient and identification ID
        (if available).
    """
    box_offset = int(box / 2)
    x = theta[:, 0] + identifications["x"] - box_offset
    y = theta[:, 1] + identifications["y"] - box_offset
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(CRLBs[:, 0])
        lpy = np.sqrt(CRLBs[:, 1])
        a = np.maximum(theta[:, 4], theta[:, 5])
        b = np.minimum(theta[:, 4], theta[:, 5])
        ellipticity = (a - b) / a
        photons_unc = np.sqrt(CRLBs[:, 2])
        bg_unc = np.sqrt(CRLBs[:, 3])
        sx_unc = np.sqrt(CRLBs[:, 4])
        sy_unc = np.sqrt(CRLBs[:, 5])
    locs = pd.DataFrame(
        {
            "frame": identifications["frame"].to_numpy(dtype=np.uint32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": theta[:, 2].astype(np.float32),
            "sx": theta[:, 4].astype(np.float32),
            "sy": theta[:, 5].astype(np.float32),
            "bg": theta[:, 3].astype(np.float32),
            "lpx": lpx.astype(np.float32),
            "lpy": lpy.astype(np.float32),
            "ellipticity": ellipticity.astype(np.float32),
            "net_gradient": identifications["net_gradient"].astype(np.float32),
            "log_likelihood": log_likelihoods.astype(np.float32),
            "iterations": iterations.astype(np.uint32),
            "photons_unc": photons_unc.astype(np.float32),
            "bg_unc": bg_unc.astype(np.float32),
            "sx_unc": sx_unc.astype(np.float32),
            "sy_unc": sy_unc.astype(np.float32),
        }
    )
    if "n_id" in identifications.columns:
        locs["n_id"] = identifications.n_id.astype(np.uint32)
        locs.sort_values(by=["n_id"], kind="quicksort", inplace=True)
    else:
        locs.sort_values(by=["frame"], kind="quicksort", inplace=True)
    return locs


def sigma_uncertainty(
    sigma: pd.Series | np.ndarray,
    sigma_orth: pd.Series | np.ndarray,
    photons: pd.Series | np.ndarray,
    bg: pd.Series | np.ndarray,
) -> np.ndarray:
    """Calculate standard error of fitted sigma based on the MLE 2D
    Gaussian/Poisson noise model (picasso.gaussmle).

    Based on the approximation by Rieger and Stallinga, ChemPhysChem,
    2014.

    Parameters
    ----------
    sigma : pd.Series | np.ndarray
        Fitted sigma values in camera pixels.
    sigma_orth : pd.Series | np.ndarray
        Fitted sigma values in the orthogonal direction in camera
        pixels.
    photons : pd.Series | np.ndarray
        Number of photons.
    bg : pd.Series | np.ndarray
        Background photons per pixel.

    Returns
    -------
    se_sigma : np.ndarray
        Standard error of fitted sigma values in camera pixels.
    """
    sa2 = sigma**2 + 1 / 12
    tau = (2 * np.pi * sa2 * bg) / (photons)
    delta_sigma_sq = (sigma**2 / (4 * photons)) * (
        1 + 8 * tau + np.sqrt((8 * tau) / (1 + 2 * tau))
    )
    return np.sqrt(delta_sigma_sq)
