"""
picasso.fitting.gaussfit
~~~~~~~~~~~~~~~~~~~~~~~~

CPU 2D Gaussian PSF fitting: the CPU twin of
:mod:`picasso.fitting.gaussfit_cuda`.

:func:`fit_spots` takes and returns exactly what ``gaussfit_cuda.fit_spots``
does, so the two are interchangeable backends and ``picasso.localize`` can
dispatch between them on a single flag - the arrangement
:mod:`picasso.fitting.splinefit` and ``splinefit_cuda`` already use for the
spline models. The models, the Levenberg-Marquardt driver, the damping rule and
the estimators are the same port of Gpufit.

Three models, all evaluated at the pixel center:

===================  ==========  ==============================================
model                parameters  layout
===================  ==========  ==============================================
:data:`SPHERICAL`             5  ``[peak, x, y, s, bg]``
:data:`ELLIPTIC`              6  ``[peak, x, y, sx, sy, bg]``
:data:`ROTATED`               7  ``[peak, x, y, sx, sy, bg, angle]``
===================  ==========  ==============================================

The amplitude is the Gaussian's *peak height*, not its integral;
``picasso.localize`` converts it to photons afterwards.


References
----------
The fitting algorithm and all three models are a port of Gpufit
(``models/gauss_2d*.cuh``, ``Cpufit/lm_fit_cpp.cpp``):

Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
License (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
from concurrent import futures
from typing import Callable, Literal

import numba
import numpy as np
from tqdm import tqdm

from picasso.fitting.splinefit import (  # noqa: F401  (re-exported)
    AsyncFit,
    FIT_STATE_CONVERGED,
    FIT_STATE_MAX_ITERATION,
    FIT_STATE_NEG_CURVATURE_MLE,
    FIT_STATE_SINGULAR_HESSIAN,
    _FASTMATH,
    _LAMBDA_DOWN,
    _LAMBDA_INITIAL,
    _LAMBDA_UP,
    allocate_outputs,
    _lm_solve_step,
    n_workers,
    resolve_variance,
)

# Model identifiers. Defined here and imported by ``gaussfit_cuda`` so the two
# devices cannot disagree about what model 1 is, exactly as ``lmfit_cuda``
# imports its constants from ``splinefit``.
SPHERICAL = 0
ELLIPTIC = 1
ROTATED = 2

_N_PARAMS = {SPHERICAL: 5, ELLIPTIC: 6, ROTATED: 7}

# Convergence schedule, from the arguments the Gpufit path passed for these
# models. It happens to coincide with the spline single-start schedule, but the
# two are independent settings and are kept separate deliberately.
TOLERANCE = 1e-2
MAX_ITERATIONS = 20

# Convergence schedule of the CPU least-squares path, inherited from the
# retired ``picasso.gausslq``: ``TOLERANCE_LSQ_CPU`` was MINPACK's relative
# reduction in both the sum of squares (``ftol``) and the parameter vector
# (``xtol``). Kept separate from the schedule above so that least-squares
# results do not move now that the fit runs here. Used by
# ``picasso.localize.gauss_schedule``.
TOLERANCE_LSQ_CPU = 1e-2
MAX_ITERATIONS_LSQ_CPU = 200


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _estimator_terms(
    mle: bool, value: float, data: float, var: float
) -> tuple:
    """Per-pixel ``(chi_square, weight, factor, ok)``.

    ``weight`` multiplies the Hessian outer product and ``factor`` the
    gradient, so a caller accumulates ``grad_k += d_k * factor`` and
    ``hess_kl += weight * d_k * d_l``.

    ``var`` is the pixel's sCMOS readout variance in photoelectrons squared,
    and zero when no camera calibration is in use. See
    ``picasso.fitting.splinefit._estimator_terms`` for the shift it produces
    and why least squares is untouched by it.

    ``ok`` is False when a maximum-likelihood fit's *shifted* model value is
    not finite **or not positive**: the fit is abandoned rather than floored.
    See the module docstring, and ``picasso.fitting.splinefit.MU_FLOOR`` for
    why the spline models do the opposite."""
    if mle:
        shifted_value = value + var
        shifted_data = data + var
        if not (np.isfinite(shifted_value) and shifted_value > 0.0):
            return np.inf, 0.0, 0.0, False
        if shifted_data > 0.0:
            return (
                2.0
                * (
                    (shifted_value - shifted_data)
                    - shifted_data * np.log(shifted_value / shifted_data)
                ),
                shifted_data / (shifted_value * shifted_value),
                -(1.0 - shifted_data / shifted_value),
                True,
            )
        # An empty (or clipped) pixel: Gpufit's data == 0 term.
        return 2.0 * shifted_value, 0.0, -1.0, True
    # Least squares: the shift cancels, so it is never formed.
    deviation = value - data
    return deviation * deviation, 1.0, -deviation, True


# ----------------------------------------------------------------------
# Per-model accumulators
#
# Each fills ``hess`` and ``grad`` at ``theta`` and returns
# ``(chi_square, ok)``. ``spots`` is ``(n_spots, box, box)`` indexed
# ``[spot, row = y, column = x]``. These transcribe Gpufit's
# ``models/gauss_2d{,_elliptic,_rotated}.cuh`` and are the line-for-line CPU
# twins of the device accumulators in ``gaussfit_cuda``.
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_spherical(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Isotropic Gaussian, ``[amplitude, x, y, s, bg]`` (``GAUSS_2D``)."""
    box = spots.shape[1]
    amp = theta[0]
    cx = theta[1]
    cy = theta[2]
    sigma = theta[3]
    offset = theta[4]
    if not (
        np.isfinite(amp)
        and np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(sigma)
        and np.isfinite(offset)
        # A zero width divides by zero in every derivative.
        and abs(sigma) > 0.0
    ):
        return np.inf, False
    inv_s2 = 1.0 / (sigma * sigma)
    inv_s3 = 1.0 / (sigma * sigma * sigma)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = 0.0
    h00 = h01 = h02 = h03 = h04 = 0.0
    h11 = h12 = h13 = h14 = 0.0
    h22 = h23 = h24 = 0.0
    h33 = h34 = 0.0
    h44 = 0.0
    for j in range(box):
        dy = j - cy
        for i in range(box):
            dx = i - cx
            ex = np.exp(-0.5 * (dx * dx + dy * dy) * inv_s2)
            value = amp * ex + offset
            data = spots[index, j, i]
            d0 = ex
            d1 = amp * ex * dx * inv_s2
            d2 = amp * ex * dy * inv_s2
            d3 = amp * ex * (dx * dx + dy * dy) * inv_s3
            # d4 == 1 (offset).
            var = 0.0
            if use_variance:
                var = variance[index, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += d3 * factor
            g4 += factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            w3 = weight * d3
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0 * d3
            h04 += w0
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1 * d3
            h14 += w1
            h22 += w2 * d2
            h23 += w2 * d3
            h24 += w2
            h33 += w3 * d3
            h34 += w3
            h44 += weight
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[4, 4] = h44
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_elliptic(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Elliptic Gaussian, ``[amplitude, x, y, sx, sy, bg]``."""
    box = spots.shape[1]
    amp = theta[0]
    cx = theta[1]
    cy = theta[2]
    sx = theta[3]
    sy = theta[4]
    offset = theta[5]
    if not (
        np.isfinite(amp)
        and np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(sx)
        and np.isfinite(sy)
        and np.isfinite(offset)
        and abs(sx) > 0.0
        and abs(sy) > 0.0
    ):
        return np.inf, False
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    inv_sx3 = 1.0 / (sx * sx * sx)
    inv_sy3 = 1.0 / (sy * sy * sy)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = g5 = 0.0
    h00 = h01 = h02 = h03 = h04 = h05 = 0.0
    h11 = h12 = h13 = h14 = h15 = 0.0
    h22 = h23 = h24 = h25 = 0.0
    h33 = h34 = h35 = 0.0
    h44 = h45 = 0.0
    h55 = 0.0
    for j in range(box):
        dy = j - cy
        for i in range(box):
            dx = i - cx
            ex = np.exp(-0.5 * (dx * dx * inv_sx2 + dy * dy * inv_sy2))
            value = amp * ex + offset
            data = spots[index, j, i]
            d0 = ex
            d1 = amp * ex * dx * inv_sx2
            d2 = amp * ex * dy * inv_sy2
            d3 = amp * ex * dx * dx * inv_sx3
            d4 = amp * ex * dy * dy * inv_sy3
            # d5 == 1 (offset).
            var = 0.0
            if use_variance:
                var = variance[index, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += d3 * factor
            g4 += d4 * factor
            g5 += factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            w3 = weight * d3
            w4 = weight * d4
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0 * d3
            h04 += w0 * d4
            h05 += w0
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1 * d3
            h14 += w1 * d4
            h15 += w1
            h22 += w2 * d2
            h23 += w2 * d3
            h24 += w2 * d4
            h25 += w2
            h33 += w3 * d3
            h34 += w3 * d4
            h35 += w3
            h44 += w4 * d4
            h45 += w4
            h55 += weight
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    grad[5] = g5
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[0, 5] = hess[5, 0] = h05
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[1, 5] = hess[5, 1] = h15
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[2, 5] = hess[5, 2] = h25
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[3, 5] = hess[5, 3] = h35
    hess[4, 4] = h44
    hess[4, 5] = hess[5, 4] = h45
    hess[5, 5] = h55
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_rotated(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Rotated elliptic Gaussian, ``[amplitude, x, y, sx, sy, bg, angle]``.

    The angle derivative vanishes identically when ``sx == sy``, which makes
    the first Hessian singular;
    ``picasso.fitting.seeds.initial_parameters_gauss`` breaks that symmetry in
    its seed on purpose, and this model relies on it."""
    box = spots.shape[1]
    amp = theta[0]
    cx = theta[1]
    cy = theta[2]
    sx = theta[3]
    sy = theta[4]
    offset = theta[5]
    angle = theta[6]
    if not (
        np.isfinite(amp)
        and np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(sx)
        and np.isfinite(sy)
        and np.isfinite(offset)
        and np.isfinite(angle)
        and abs(sx) > 0.0
        and abs(sy) > 0.0
    ):
        return np.inf, False
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    inv_sx3 = 1.0 / (sx * sx * sx)
    inv_sy3 = 1.0 / (sy * sy * sy)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = g5 = g6 = 0.0
    h00 = h01 = h02 = h03 = h04 = h05 = h06 = 0.0
    h11 = h12 = h13 = h14 = h15 = h16 = 0.0
    h22 = h23 = h24 = h25 = h26 = 0.0
    h33 = h34 = h35 = h36 = 0.0
    h44 = h45 = h46 = 0.0
    h55 = h56 = 0.0
    h66 = 0.0
    for j in range(box):
        dy = j - cy
        for i in range(box):
            dx = i - cx
            arga = dx * cos_a - dy * sin_a
            argb = dx * sin_a + dy * cos_a
            ex = np.exp(-0.5 * (arga * arga * inv_sx2 + argb * argb * inv_sy2))
            value = amp * ex + offset
            data = spots[index, j, i]
            d0 = ex
            d1 = (
                amp * cos_a * arga * inv_sx2 + amp * sin_a * argb * inv_sy2
            ) * ex
            d2 = (
                -amp * sin_a * arga * inv_sx2 + amp * cos_a * argb * inv_sy2
            ) * ex
            d3 = amp * arga * arga * inv_sx3 * ex
            d4 = amp * argb * argb * inv_sy3 * ex
            # d5 == 1 (offset).
            d6 = amp * arga * argb * (inv_sx2 - inv_sy2) * ex
            var = 0.0
            if use_variance:
                var = variance[index, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += d3 * factor
            g4 += d4 * factor
            g5 += factor
            g6 += d6 * factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            w3 = weight * d3
            w4 = weight * d4
            w6 = weight * d6
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0 * d3
            h04 += w0 * d4
            h05 += w0
            h06 += w0 * d6
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1 * d3
            h14 += w1 * d4
            h15 += w1
            h16 += w1 * d6
            h22 += w2 * d2
            h23 += w2 * d3
            h24 += w2 * d4
            h25 += w2
            h26 += w2 * d6
            h33 += w3 * d3
            h34 += w3 * d4
            h35 += w3
            h36 += w3 * d6
            h44 += w4 * d4
            h45 += w4
            h46 += w4 * d6
            h55 += weight
            h56 += w6
            h66 += w6 * d6
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    grad[5] = g5
    grad[6] = g6
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[0, 5] = hess[5, 0] = h05
    hess[0, 6] = hess[6, 0] = h06
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[1, 5] = hess[5, 1] = h15
    hess[1, 6] = hess[6, 1] = h16
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[2, 5] = hess[5, 2] = h25
    hess[2, 6] = hess[6, 2] = h26
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[3, 5] = hess[5, 3] = h35
    hess[3, 6] = hess[6, 3] = h36
    hess[4, 4] = h44
    hess[4, 5] = hess[5, 4] = h45
    hess[4, 6] = hess[6, 4] = h46
    hess[5, 5] = h55
    hess[5, 6] = hess[6, 5] = h56
    hess[6, 6] = h66
    return chi_square, True


# ----------------------------------------------------------------------
# Levenberg-Marquardt driver
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate(
    model: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Dispatch to the accumulator of ``model``."""
    if model == SPHERICAL:
        return _accumulate_spherical(
            spots, index, variance, use_variance, theta, mle, hess, grad
        )
    if model == ELLIPTIC:
        return _accumulate_elliptic(
            spots, index, variance, use_variance, theta, mle, hess, grad
        )
    return _accumulate_rotated(
        spots, index, variance, use_variance, theta, mle, hess, grad
    )


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_seed(
    model: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
    hess_ok: np.ndarray,
    grad_ok: np.ndarray,
    n_params: int,
) -> tuple:
    """Evaluate chi-square and curvature at the seed parameters.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.
    spots : np.ndarray
        ``(n_spots, box, box)`` photon counts, indexed ``[spot, y, x]``.
    index : int
        Index of the spot to fit.
    variance : np.ndarray
        Per-pixel sCMOS readout variance, laid out like ``spots``; ignored
        unless ``use_variance``.
    use_variance : bool
        Whether ``variance`` is applied.
    theta : np.ndarray
        ``(n_params,)`` seed parameters.
    mle : bool
        Use the Poisson maximum-likelihood estimator instead of least squares.
    hess, grad : np.ndarray
        Scratch Hessian ``(n_params, n_params)`` and gradient
        ``(n_params,)``, filled at ``theta``.
    hess_ok, grad_ok : np.ndarray
        Curvature of the last accepted iteration; set to ``hess``/``grad``
        when the seed is valid.
    n_params : int
        Number of model parameters.

    Returns
    -------
    chi_square : float
        Chi-square at the seed.
    state : int
        :data:`FIT_STATE_CONVERGED`, unless the seed model is non-finite or
        non-positive, in which case it is
        :data:`FIT_STATE_NEG_CURVATURE_MLE` and the fit never enters the
        iteration loop.
    """
    chi_square, ok = _accumulate(
        model, spots, index, variance, use_variance, theta, mle, hess, grad
    )
    if not ok:
        return chi_square, FIT_STATE_NEG_CURVATURE_MLE
    for p in range(n_params):
        grad_ok[p] = grad[p]
        for q in range(n_params):
            hess_ok[p, q] = hess[p, q]
    return chi_square, FIT_STATE_CONVERGED


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_reject_step(
    theta: np.ndarray,
    theta_previous: np.ndarray,
    previous_chi_square: float,
    lam: float,
    state: int,
    iteration: int,
    max_iterations: int,
    n_params: int,
) -> tuple:
    """Undo a trial step whose model went non-positive or non-finite.

    That is a property of the *step*, not of the fit: undo it, damp harder
    and try again, exactly as for a step that merely worsened chi-square.
    Aborting here would return the seed unchanged, which for a wide box
    shows up as sigma pinned to the seed width and integer coordinates.

    Parameters
    ----------
    theta : np.ndarray
        ``(n_params,)`` trial parameters; restored in place to
        ``theta_previous``.
    theta_previous : np.ndarray
        ``(n_params,)`` parameters before the trial step.
    previous_chi_square : float
        Chi-square of the last accepted iteration.
    lam : float
        Current Levenberg-Marquardt damping.
    state : int
        Current fit state.
    iteration : int
        Index of the current iteration.
    max_iterations : int
        Maximum number of iterations.
    n_params : int
        Number of model parameters.

    Returns
    -------
    chi_square : float
        ``previous_chi_square``, the chi-square of the restored parameters.
    lam : float
        Increased damping.
    state : int
        Updated fit state; :data:`FIT_STATE_NEG_CURVATURE_MLE` if this was
        the last iteration.
    """
    for p in range(n_params):
        theta[p] = theta_previous[p]
    lam *= _LAMBDA_UP
    if iteration == max_iterations - 1:
        state = FIT_STATE_NEG_CURVATURE_MLE
    return previous_chi_square, lam, state


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_accept_or_reject(
    theta: np.ndarray,
    theta_previous: np.ndarray,
    grad: np.ndarray,
    grad_ok: np.ndarray,
    hess: np.ndarray,
    hess_ok: np.ndarray,
    n_params: int,
    chi_square: float,
    previous_chi_square: float,
    lam: float,
    tolerance: float,
    iteration: int,
    max_iterations: int,
    state: int,
) -> tuple:
    """Update curvature, damping and convergence after a valid trial step.

    Parameters
    ----------
    theta : np.ndarray
        ``(n_params,)`` trial parameters; restored in place to
        ``theta_previous`` if the step is rejected.
    theta_previous : np.ndarray
        ``(n_params,)`` parameters before the trial step.
    grad : np.ndarray
        Gradient evaluated at the trial parameters.
    grad_ok : np.ndarray
        Gradient of the last accepted iteration; refreshed from ``grad`` on
        an improving step.
    hess : np.ndarray
        Hessian evaluated at the trial parameters.
    hess_ok : np.ndarray
        Hessian of the last accepted iteration; refreshed from ``hess`` on
        an improving step.
    n_params : int
        Number of model parameters.
    chi_square : float
        Chi-square at the trial parameters.
    previous_chi_square : float
        Chi-square of the last accepted iteration.
    lam : float
        Current Levenberg-Marquardt damping.
    tolerance : float
        Convergence tolerance on the change in chi-square.
    iteration : int
        Index of the current iteration.
    max_iterations : int
        Maximum number of iterations.
    state : int
        Current fit state.

    Returns
    -------
    chi_square : float
        Chi-square of the parameters kept.
    previous_chi_square : float
        Updated chi-square of the last accepted iteration.
    lam : float
        Updated damping.
    state : int
        Updated fit state; :data:`FIT_STATE_MAX_ITERATION` if the last
        iteration did not converge.
    converged : bool
        Whether the change in chi-square fell below the tolerance.
    """
    if chi_square < previous_chi_square or previous_chi_square == 0.0:
        # Only an improving iteration refreshes the curvature the next step
        # is damped from (Gpufit skips the gradient/Hessian kernels
        # entirely on a failed iteration).
        for p in range(n_params):
            grad_ok[p] = grad[p]
            for q in range(n_params):
                hess_ok[p, q] = hess[p, q]
    converged = abs(chi_square - previous_chi_square) < max(
        tolerance, tolerance * abs(chi_square)
    )
    if not converged and iteration == max_iterations - 1:
        state = FIT_STATE_MAX_ITERATION
    if chi_square < previous_chi_square:
        lam *= _LAMBDA_DOWN
        previous_chi_square = chi_square
    else:
        lam *= _LAMBDA_UP
        chi_square = previous_chi_square
        for p in range(n_params):
            theta[p] = theta_previous[p]
    return chi_square, previous_chi_square, lam, state, converged


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_iterate(
    model: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    theta: np.ndarray,
    theta_previous: np.ndarray,
    grad: np.ndarray,
    grad_ok: np.ndarray,
    delta: np.ndarray,
    scaling: np.ndarray,
    hess: np.ndarray,
    hess_ok: np.ndarray,
    hess_damped: np.ndarray,
    indxc: np.ndarray,
    indxr: np.ndarray,
    ipiv: np.ndarray,
    n_params: int,
    chi_square: float,
) -> tuple:
    """Run the Levenberg-Marquardt loop from the seeded curvature.

    ``theta`` is updated in place with the best parameters found.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.
    spots : np.ndarray
        ``(n_spots, box, box)`` photon counts, indexed ``[spot, y, x]``.
    index : int
        Index of the spot to fit.
    variance : np.ndarray
        Per-pixel sCMOS readout variance, laid out like ``spots``; ignored
        unless ``use_variance``.
    use_variance : bool
        Whether ``variance`` is applied.
    mle : bool
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance : float
        Convergence tolerance on the change in chi-square.
    max_iterations : int
        Maximum number of iterations.
    theta : np.ndarray
        ``(n_params,)`` seed parameters, updated in place.
    theta_previous, grad, grad_ok, delta, scaling : np.ndarray
        ``(n_params,)`` scratch buffers.
    hess, hess_ok, hess_damped : np.ndarray
        ``(n_params, n_params)`` scratch buffers; ``hess_ok``/``grad_ok``
        must hold the seed curvature from :func:`_lm_seed`.
    indxc, indxr, ipiv : np.ndarray
        ``(n_params,)`` integer scratch buffers of the linear solver.
    n_params : int
        Number of model parameters.
    chi_square : float
        Chi-square at the seed.

    Returns
    -------
    chi_square : float
        Chi-square at the best parameters found.
    state : int
        Final fit state.
    n_iterations : int
        Number of iterations run.
    """
    state = FIT_STATE_CONVERGED
    lam = _LAMBDA_INITIAL
    previous_chi_square = chi_square
    n_iterations = 0

    for iteration in range(max_iterations):
        if not _lm_solve_step(
            hess_ok,
            grad_ok,
            scaling,
            lam,
            hess_damped,
            delta,
            n_params,
            indxc,
            indxr,
            ipiv,
        ):
            # The step is garbage, so it is not applied; the parameters of the
            # last accepted iteration stand.
            state = FIT_STATE_SINGULAR_HESSIAN
            break
        for p in range(n_params):
            theta_previous[p] = theta[p]
            theta[p] += delta[p]
        new_chi_square, ok = _accumulate(
            model, spots, index, variance, use_variance, theta, mle, hess, grad
        )
        n_iterations = iteration + 1
        if not ok:
            chi_square, lam, state = _lm_reject_step(
                theta,
                theta_previous,
                previous_chi_square,
                lam,
                state,
                iteration,
                max_iterations,
                n_params,
            )
            continue
        chi_square, previous_chi_square, lam, state, converged = (
            _lm_accept_or_reject(
                theta,
                theta_previous,
                grad,
                grad_ok,
                hess,
                hess_ok,
                n_params,
                new_chi_square,
                previous_chi_square,
                lam,
                tolerance,
                iteration,
                max_iterations,
                state,
            )
        )
        if converged:
            break

    return chi_square, state, n_iterations


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_finalize(
    theta: np.ndarray,
    chi_square: float,
    state: int,
    n_iterations: int,
    mle: bool,
    n_params: int,
    thetas: np.ndarray,
    chi_squares: np.ndarray,
    states: np.ndarray,
    iterations: np.ndarray,
    index: int,
) -> None:
    """Write the fit outcome for one spot into the preallocated outputs.

    A diverged fit (non-finite chi-square or parameters) is reported as NaN
    parameters with an infinite chi-square;
    ``localize.locs_from_fits_gauss`` turns those into NaN precisions.
    """
    finite = np.isfinite(chi_square)
    if finite:
        for p in range(n_params):
            if not np.isfinite(theta[p]):
                finite = False
                break
    if finite:
        for p in range(n_params):
            thetas[index, p] = theta[p]
        chi_squares[index] = chi_square
        states[index] = state
        iterations[index] = n_iterations
    else:
        for p in range(n_params):
            thetas[index, p] = np.nan
        chi_squares[index] = np.inf
        if mle:
            states[index] = FIT_STATE_NEG_CURVATURE_MLE
        else:
            states[index] = FIT_STATE_SINGULAR_HESSIAN
        iterations[index] = n_iterations


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _fit_gauss_spot(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    model: int,
    init: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    thetas: np.ndarray,
    chi_squares: np.ndarray,
    states: np.ndarray,
    iterations: np.ndarray,
) -> None:
    """Fit one spot with the Levenberg-Marquardt driver ported from Gpufit.

    The same driver as :func:`picasso.fitting.splinefit._fit_spline_spot`,
    minus the axial multi-start - a Gaussian has no axial coordinate, so there
    is exactly one start per spot and nothing to rank.

    Results are written into the preallocated ``thetas``, ``chi_squares``,
    ``states`` and ``iterations`` at row ``index``.
    """
    n_params = init.shape[1]
    # All scratch is allocated once per spot, outside the iteration loop -
    # allocating inside it dominates the runtime of a threaded numba kernel.
    theta = np.empty(n_params)
    theta_previous = np.empty(n_params)
    grad = np.zeros(n_params)
    grad_ok = np.zeros(n_params)
    delta = np.empty(n_params)
    scaling = np.empty(n_params)
    hess = np.zeros((n_params, n_params))
    hess_ok = np.zeros((n_params, n_params))
    hess_damped = np.empty((n_params, n_params))
    indxc = np.empty(n_params, dtype=np.int32)
    indxr = np.empty(n_params, dtype=np.int32)
    ipiv = np.empty(n_params, dtype=np.int32)

    for p in range(n_params):
        theta[p] = init[index, p]
        scaling[p] = 0.0

    chi_square, state = _lm_seed(
        model,
        spots,
        index,
        variance,
        use_variance,
        theta,
        mle,
        hess,
        grad,
        hess_ok,
        grad_ok,
        n_params,
    )
    n_iterations = 0
    if state == FIT_STATE_CONVERGED:
        chi_square, state, n_iterations = _lm_iterate(
            model,
            spots,
            index,
            variance,
            use_variance,
            mle,
            tolerance,
            max_iterations,
            theta,
            theta_previous,
            grad,
            grad_ok,
            delta,
            scaling,
            hess,
            hess_ok,
            hess_damped,
            indxc,
            indxr,
            ipiv,
            n_params,
            chi_square,
        )

    _lm_finalize(
        theta,
        chi_square,
        state,
        n_iterations,
        mle,
        n_params,
        thetas,
        chi_squares,
        states,
        iterations,
        index,
    )


# ----------------------------------------------------------------------
# Host layer
# ----------------------------------------------------------------------


def n_parameters(model: int) -> int:
    """Parameter count of a model.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.

    Returns
    -------
    n_params : int
        5, 6 or 7 respectively.
    """
    return _N_PARAMS[model]


def _prepare(
    model: int,
    spots: np.ndarray,
    initial_parameters: np.ndarray,
    tolerance: float | None,
    max_iterations: int | None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Validate and normalize the inputs both entry points share."""
    if model not in _N_PARAMS:
        raise ValueError(f"Unknown Gaussian model {model}.")
    spots = np.asarray(spots)
    if spots.ndim != 3 or spots.shape[1] != spots.shape[2]:
        raise ValueError(
            f"spots must have shape (n_spots, box, box), got {spots.shape}."
        )
    n_params = _N_PARAMS[model]
    initial_parameters = np.asarray(initial_parameters)
    if initial_parameters.shape != (len(spots), n_params):
        raise ValueError(
            "initial_parameters must have shape "
            f"{(len(spots), n_params)}, got {initial_parameters.shape}."
        )
    if tolerance is None:
        tolerance = TOLERANCE
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=3)
    outputs = allocate_outputs(len(spots), n_params)
    return (
        spots,
        variance,
        use_variance,
        initial_parameters,
        float(tolerance),
        int(max_iterations),
        outputs,
    )


def _kernel_args(
    variance: np.ndarray,
    use_variance: bool,
    model: int,
    initial_parameters: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    outputs: tuple,
) -> tuple:
    """Freeze everything :func:`_fit_gauss_spot` needs after ``index``.

    ``variance``/``use_variance`` lead, so they land immediately after
    ``index`` in the kernel signature and every ``_fit_gauss_spot(spots, index,
    *args)`` call site stays as it was."""
    thetas, chi_squares, states, iterations = outputs
    return (
        variance,
        bool(use_variance),
        int(model),
        initial_parameters,
        bool(mle),
        float(tolerance),
        int(max_iterations),
        thetas,
        chi_squares,
        states,
        iterations,
    )


def fit_spots(
    model: int,
    spots: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Fit spots with a 2D Gaussian model on the CPU, serially.

    Signature-compatible with
    :func:`picasso.fitting.gaussfit_cuda.fit_spots`, so the two are
    interchangeable backends.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.
    spots : np.ndarray
        ``(n_spots, box, box)`` photon counts, indexed ``[spot, y, x]``.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds, from
        ``picasso.fitting.seeds.initial_parameters_gauss``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance : float, optional
        Convergence tolerance. ``None`` (the default) uses
        :data:`TOLERANCE`.
    max_iterations : int, optional
        Maximum number of iterations. ``None`` (the default) uses
        :data:`MAX_ITERATIONS`.
    progress_callback : callable, "console" or None, optional
        ``"console"`` shows a tqdm bar; a callable is invoked with the
        cumulative number of spots fitted.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like ``spots``. ``None`` (the default) fits the plain Poisson
        model.

    Returns
    -------
    thetas : np.ndarray
        ``(n_spots, n_params)`` fitted parameters.
    chi_squares : np.ndarray
        ``(n_spots,)`` chi-square at the optimum (twice the negative Poisson
        log-likelihood for ``mle``).
    states : np.ndarray
        ``(n_spots,)`` per-spot fit state, using Gpufit's codes (see
        ``picasso.fitting.splinefit.FIT_STATE_CONVERGED``).
    iterations : np.ndarray
        ``(n_spots,)`` iterations used.
    """
    (
        spots,
        variance,
        use_variance,
        initial_parameters,
        tolerance,
        max_iterations,
        outputs,
    ) = _prepare(
        model, spots, initial_parameters, tolerance, max_iterations, variance
    )
    args = _kernel_args(
        variance,
        use_variance,
        model,
        initial_parameters,
        mle,
        tolerance,
        max_iterations,
        outputs,
    )
    use_tqdm = progress_callback == "console"
    index_range = range(len(spots))
    if use_tqdm:
        index_range = tqdm(index_range, desc="Fitting", unit="spot")
    for index in index_range:
        _fit_gauss_spot(spots, index, *args)
        if callable(progress_callback):
            progress_callback(index + 1)
    return outputs


def _worker(
    spots: np.ndarray, args: tuple, current: list, lock, stop: np.ndarray
) -> None:
    """Worker for asynchronous fitting: claim the next spot and fit it."""
    n_spots = len(spots)
    while True:
        with lock:
            if stop[0] or current[0] == n_spots:
                return
            index = current[0]
            current[0] += 1
        _fit_gauss_spot(spots, index, *args)


def fit_spots_async(
    model: int,
    spots: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    n_threads: int | None = None,
    variance: np.ndarray | None = None,
) -> AsyncFit:
    """Fit spots with a 2D Gaussian model on several CPU threads.

    Returns immediately, so the caller can poll for progress, abort, or check
    for worker errors while the fit runs.

    The kernels are ``nogil``, so this is *threads*, not processes - unlike
    the ``gausslq.fit_spots_parallel`` it replaces, which forked up to 60
    worker processes and pickled the spots into each of them.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.
    spots : np.ndarray
        ``(n_spots, box, box)`` photon counts, indexed ``[spot, y, x]``.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds, from
        ``picasso.fitting.seeds.initial_parameters_gauss``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance : float, optional
        Convergence tolerance. ``None`` (the default) uses
        :data:`TOLERANCE`.
    max_iterations : int, optional
        Maximum number of iterations. ``None`` (the default) uses
        :data:`MAX_ITERATIONS`.
    n_threads : int, optional
        Number of worker threads. ``None`` (the default) uses
        ``picasso.fitting.splinefit.n_workers``, and the count is clipped to
        at most one thread per spot.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance, as in :func:`fit_spots`.

    Returns
    -------
    async_fit : picasso.fitting.splinefit.AsyncFit
        Handle on the running fit. Its result arrays are filled in place; call
        ``finished()`` to poll, ``results()`` for
        ``(thetas, chi_squares, states, iterations)`` once done,
        ``raise_errors()`` to surface worker exceptions and ``stop()`` to
        abort.
    """
    (
        spots,
        variance,
        use_variance,
        initial_parameters,
        tolerance,
        max_iterations,
        outputs,
    ) = _prepare(
        model, spots, initial_parameters, tolerance, max_iterations, variance
    )
    args = _kernel_args(
        variance,
        use_variance,
        model,
        initial_parameters,
        mle,
        tolerance,
        max_iterations,
        outputs,
    )
    n_spots = len(spots)
    if n_threads is None:
        n_threads = n_workers()
    n_threads = max(1, min(int(n_threads), max(n_spots, 1)))
    lock = threading.Lock()
    current = [0]
    stop = np.zeros(1, dtype=np.uint8)
    executor = futures.ThreadPoolExecutor(n_threads)
    fs = [
        executor.submit(_worker, spots, args, current, lock, stop)
        for _ in range(n_threads)
    ]
    executor.shutdown(wait=False)
    return AsyncFit(current, *outputs, fs, stop)


# ----------------------------------------------------------------------
# Multichannel (joint) spherical Gaussian
#
# Several registered channels are fitted at once with one shared position and
# width, the globLoc arrangement ``picasso.fitting.splinefit`` already uses for
# the spline PSF (Li et al., Nat. Commun. 13, 3133, 2022). Only the isotropic
# Gaussian has a multichannel model.
#
# ``spots`` is channel-major ``(n_spots, n_channels, box, box)``, indexed
# ``[spot, channel, row = y, column = x]``, as the spline kernels take it.
#
# Each channel sees the *shared* lateral shift through its own registration:
# ``jac`` is that channel's local Jacobian ``[a00, a01, a10, a11]`` and ``res``
# its sub-pixel ROI offset, the fractional part the integer box origin cannot
# express. Together they place channel ``c``'s model at ``T_c(x + theta)`` to
# first order; see ``picasso.localize.channel_roi_geometry``. The single-
# channel fit is the identity-Jacobian, zero-residual case.
# ----------------------------------------------------------------------

#: One amplitude and one background shared by every channel;
#: ``[amplitude, x_shift, y_shift, sigma, offset]``.
MULTI_KIND_SHARED = 0
#: Photon-decoupled: position and width shared, every channel its own photon
#: count and background; ``[x_shift, y_shift, sigma, N_0.., bg_0..]``.
MULTI_KIND_DECOUPLED = 1

_N_PARAMS_MULTI_SHARED = 5


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_spherical_multichannel(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    jac: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Shared-amplitude multichannel isotropic Gaussian.

    Parameters are ``[amplitude, x_shift, y_shift, sigma, offset]`` - the same
    five as the single-channel model, because the channels add pixels rather
    than parameters. Every channel contributes to one combined 5x5 system.

    The amplitude is shared *literally*: this model has no per-channel
    brightness scale (unlike the spline model, whose per-channel coefficient
    table carries one), so it assumes every channel collects the same number of
    photons. Use :data:`MULTI_KIND_DECOUPLED` when they do not.
    """
    n_channels = spots.shape[1]
    box = spots.shape[2]
    amp = theta[0]
    x_shift = theta[1]
    y_shift = theta[2]
    sigma = theta[3]
    offset = theta[4]
    if not (
        np.isfinite(amp)
        and np.isfinite(x_shift)
        and np.isfinite(y_shift)
        and np.isfinite(sigma)
        and np.isfinite(offset)
        # A zero width divides by zero in every derivative.
        and abs(sigma) > 0.0
    ):
        return np.inf, False
    inv_s2 = 1.0 / (sigma * sigma)
    inv_s3 = 1.0 / (sigma * sigma * sigma)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = 0.0
    h00 = h01 = h02 = h03 = h04 = 0.0
    h11 = h12 = h13 = h14 = 0.0
    h22 = h23 = h24 = 0.0
    h33 = h34 = 0.0
    h44 = 0.0
    # The channel Jacobian linearizes the *displacement* of the
    # emitter from the box center, not the box coordinate itself:
    # channel c sees center + J @ (shift - center) + residual. Off by
    # the center, a mirrored or rotated channel lands outside its own
    # box and its photon count runs away.
    center = 0.5 * box - 0.5
    dx_shift = x_shift - center
    dy_shift = y_shift - center
    for ch in range(n_channels):
        # Constant over the box, so hoisted.
        a00 = jac[index, ch, 0]
        a01 = jac[index, ch, 1]
        a10 = jac[index, ch, 2]
        a11 = jac[index, ch, 3]
        sx = center + a00 * dx_shift + a01 * dy_shift + res[index, ch, 0]
        sy = center + a10 * dx_shift + a11 * dy_shift + res[index, ch, 1]
        for j in range(box):
            pos_y = j - sy
            for i in range(box):
                pos_x = i - sx
                r2 = pos_x * pos_x + pos_y * pos_y
                ex = np.exp(-0.5 * r2 * inv_s2)
                value = amp * ex + offset
                data = spots[index, ch, j, i]
                # Local gradient of the unit-height Gaussian.
                gx = -ex * pos_x * inv_s2
                gy = -ex * pos_y * inv_s2
                d0 = ex
                # The lateral pair picks up the transpose of the channel
                # Jacobian (shift = J @ theta, so d(shift_x)/d(theta_x) = a00
                # and d(shift_y)/d(theta_x) = a10); the leading minus is the
                # chain rule of position = pixel - shift.
                d1 = -amp * (a00 * gx + a10 * gy)
                d2 = -amp * (a01 * gx + a11 * gy)
                # sigma is orthogonal to the lateral registration, so unlike
                # x and y it needs no Jacobian.
                d3 = amp * ex * r2 * inv_s3
                # d4 == 1 (offset).
                var = 0.0
                if use_variance:
                    var = variance[index, ch, j, i]
                contrib, weight, factor, ok = _estimator_terms(
                    mle, value, data, var
                )
                if not ok:
                    return np.inf, False
                chi_square += contrib
                g0 += d0 * factor
                g1 += d1 * factor
                g2 += d2 * factor
                g3 += d3 * factor
                g4 += factor
                w0 = weight * d0
                w1 = weight * d1
                w2 = weight * d2
                w3 = weight * d3
                h00 += w0 * d0
                h01 += w0 * d1
                h02 += w0 * d2
                h03 += w0 * d3
                h04 += w0
                h11 += w1 * d1
                h12 += w1 * d2
                h13 += w1 * d3
                h14 += w1
                h22 += w2 * d2
                h23 += w2 * d3
                h24 += w2
                h33 += w3 * d3
                h34 += w3
                h44 += weight
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[4, 4] = h44
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _reset_decoupled_scratch(
    theta: np.ndarray, n_params: int, grad: np.ndarray, hess: np.ndarray
) -> bool:
    """Zero ``grad``/``hess`` for the decoupled accumulator.

    Parameters
    ----------
    theta : np.ndarray
        ``(n_params,)`` current parameters, checked for finiteness.
    n_params : int
        Number of model parameters.
    grad : np.ndarray
        ``(n_params,)`` gradient, zeroed in place.
    hess : np.ndarray
        ``(n_params, n_params)`` Hessian, zeroed in place.

    Returns
    -------
    bool
        False if any parameter is non-finite, in which case ``grad``/
        ``hess`` are left as found.
    """
    for p in range(n_params):
        if not np.isfinite(theta[p]):
            return False
        grad[p] = 0.0
        for q in range(n_params):
            hess[p, q] = 0.0
    return True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _mirror_upper_triangle(hess: np.ndarray, n_params: int) -> None:
    """Mirror the upper triangle of ``hess`` into the lower triangle."""
    for p in range(n_params):
        for q in range(p):
            hess[p, q] = hess[q, p]


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_spherical_decoupled(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    jac: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Photon-decoupled multichannel isotropic Gaussian.

    Parameters are ``[x_shift, y_shift, sigma, N_0..N_{C-1}, bg_0..bg_{C-1}]``:
    position **and width** are shared across channels while every channel fits
    its own photon count and background. Makes no assumption about the relative
    brightness of the channels, so it is the safe choice for an unevenly split
    beam.

    A pixel of channel ``ch`` only ever sees five of the ``3 + 2C`` parameters,
    so the gradient and Hessian are accumulated block-sparse. See
    :func:`_accumulate_spherical_multichannel`."""
    n_channels = spots.shape[1]
    box = spots.shape[2]
    n_params = 3 + 2 * n_channels
    x_shift = theta[0]
    y_shift = theta[1]
    sigma = theta[2]
    if not _reset_decoupled_scratch(theta, n_params, grad, hess):
        return np.inf, False
    if not abs(sigma) > 0.0:
        return np.inf, False
    inv_s2 = 1.0 / (sigma * sigma)
    inv_s3 = 1.0 / (sigma * sigma * sigma)
    chi_square = 0.0
    # The channel Jacobian linearizes the *displacement* of the
    # emitter from the box center, not the box coordinate itself:
    # channel c sees center + J @ (shift - center) + residual. Off by
    # the center, a mirrored or rotated channel lands outside its own
    # box and its photon count runs away.
    center = 0.5 * box - 0.5
    dx_shift = x_shift - center
    dy_shift = y_shift - center
    for ch in range(n_channels):
        amp = theta[3 + ch]
        offset = theta[3 + n_channels + ch]
        # Global parameter indices of this channel's photon count and
        # background; the three shared parameters are 0, 1, 2.
        ia = 3 + ch
        ib = 3 + n_channels + ch
        a00 = jac[index, ch, 0]
        a01 = jac[index, ch, 1]
        a10 = jac[index, ch, 2]
        a11 = jac[index, ch, 3]
        sx = center + a00 * dx_shift + a01 * dy_shift + res[index, ch, 0]
        sy = center + a10 * dx_shift + a11 * dy_shift + res[index, ch, 1]
        for j in range(box):
            pos_y = j - sy
            for i in range(box):
                pos_x = i - sx
                r2 = pos_x * pos_x + pos_y * pos_y
                ex = np.exp(-0.5 * r2 * inv_s2)
                value = amp * ex + offset
                data = spots[index, ch, j, i]
                gx = -ex * pos_x * inv_s2
                gy = -ex * pos_y * inv_s2
                d0 = -amp * (a00 * gx + a10 * gy)
                d1 = -amp * (a01 * gx + a11 * gy)
                d2 = amp * ex * r2 * inv_s3
                # d(value)/d(N_ch) = ex, d(value)/d(bg_ch) = 1; both zero for
                # every other channel.
                var = 0.0
                if use_variance:
                    var = variance[index, ch, j, i]
                contrib, weight, factor, ok = _estimator_terms(
                    mle, value, data, var
                )
                if not ok:
                    return np.inf, False
                chi_square += contrib
                grad[0] += d0 * factor
                grad[1] += d1 * factor
                grad[2] += d2 * factor
                grad[ia] += ex * factor
                grad[ib] += factor
                w0 = weight * d0
                w1 = weight * d1
                w2 = weight * d2
                wa = weight * ex
                hess[0, 0] += w0 * d0
                hess[0, 1] += w0 * d1
                hess[0, 2] += w0 * d2
                hess[0, ia] += w0 * ex
                hess[0, ib] += w0
                hess[1, 1] += w1 * d1
                hess[1, 2] += w1 * d2
                hess[1, ia] += w1 * ex
                hess[1, ib] += w1
                hess[2, 2] += w2 * d2
                hess[2, ia] += w2 * ex
                hess[2, ib] += w2
                hess[ia, ia] += wa * ex
                hess[ia, ib] += wa
                hess[ib, ib] += weight
    # Only the upper triangle was filled (0 < 1 < 2 < ia < ib always holds);
    # mirror it once at the end rather than per pixel.
    _mirror_upper_triangle(hess, n_params)
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_multichannel(
    kind: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    jac: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Dispatch to the accumulator of ``kind``."""
    if kind == MULTI_KIND_SHARED:
        return _accumulate_spherical_multichannel(
            spots,
            index,
            variance,
            use_variance,
            jac,
            res,
            theta,
            mle,
            hess,
            grad,
        )
    return _accumulate_spherical_decoupled(
        spots, index, variance, use_variance, jac, res, theta, mle, hess, grad
    )


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_seed_multichannel(
    kind: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    jac: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
    hess_ok: np.ndarray,
    grad_ok: np.ndarray,
    n_params: int,
) -> tuple:
    """The multichannel twin of :func:`_lm_seed`, over
    :func:`_accumulate_multichannel`."""
    chi_square, ok = _accumulate_multichannel(
        kind,
        spots,
        index,
        variance,
        use_variance,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
    )
    if not ok:
        return chi_square, FIT_STATE_NEG_CURVATURE_MLE
    for p in range(n_params):
        grad_ok[p] = grad[p]
        for q in range(n_params):
            hess_ok[p, q] = hess[p, q]
    return chi_square, FIT_STATE_CONVERGED


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _lm_iterate_multichannel(
    kind: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    jac: np.ndarray,
    res: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    theta: np.ndarray,
    theta_previous: np.ndarray,
    grad: np.ndarray,
    grad_ok: np.ndarray,
    delta: np.ndarray,
    scaling: np.ndarray,
    hess: np.ndarray,
    hess_ok: np.ndarray,
    hess_damped: np.ndarray,
    indxc: np.ndarray,
    indxr: np.ndarray,
    ipiv: np.ndarray,
    n_params: int,
    chi_square: float,
) -> tuple:
    """The multichannel twin of :func:`_lm_iterate`, over
    :func:`_accumulate_multichannel`."""
    state = FIT_STATE_CONVERGED
    lam = _LAMBDA_INITIAL
    previous_chi_square = chi_square
    n_iterations = 0

    for iteration in range(max_iterations):
        if not _lm_solve_step(
            hess_ok,
            grad_ok,
            scaling,
            lam,
            hess_damped,
            delta,
            n_params,
            indxc,
            indxr,
            ipiv,
        ):
            state = FIT_STATE_SINGULAR_HESSIAN
            break
        for p in range(n_params):
            theta_previous[p] = theta[p]
            theta[p] += delta[p]
        new_chi_square, ok = _accumulate_multichannel(
            kind,
            spots,
            index,
            variance,
            use_variance,
            jac,
            res,
            theta,
            mle,
            hess,
            grad,
        )
        n_iterations = iteration + 1
        if not ok:
            chi_square, lam, state = _lm_reject_step(
                theta,
                theta_previous,
                previous_chi_square,
                lam,
                state,
                iteration,
                max_iterations,
                n_params,
            )
            continue
        chi_square, previous_chi_square, lam, state, converged = (
            _lm_accept_or_reject(
                theta,
                theta_previous,
                grad,
                grad_ok,
                hess,
                hess_ok,
                n_params,
                new_chi_square,
                previous_chi_square,
                lam,
                tolerance,
                iteration,
                max_iterations,
                state,
            )
        )
        if converged:
            break

    return chi_square, state, n_iterations


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _fit_gauss_spot_multichannel(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    kind: int,
    jac: np.ndarray,
    res: np.ndarray,
    init: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    thetas: np.ndarray,
    chi_squares: np.ndarray,
    states: np.ndarray,
    iterations: np.ndarray,
) -> None:
    """Fit one multichannel spot. The driver of :func:`_fit_gauss_spot`, over
    the multichannel accumulators."""
    n_params = init.shape[1]
    theta = np.empty(n_params)
    theta_previous = np.empty(n_params)
    grad = np.zeros(n_params)
    grad_ok = np.zeros(n_params)
    delta = np.empty(n_params)
    scaling = np.empty(n_params)
    hess = np.zeros((n_params, n_params))
    hess_ok = np.zeros((n_params, n_params))
    hess_damped = np.empty((n_params, n_params))
    indxc = np.empty(n_params, dtype=np.int32)
    indxr = np.empty(n_params, dtype=np.int32)
    ipiv = np.empty(n_params, dtype=np.int32)

    for p in range(n_params):
        theta[p] = init[index, p]
        scaling[p] = 0.0

    chi_square, state = _lm_seed_multichannel(
        kind,
        spots,
        index,
        variance,
        use_variance,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
        hess_ok,
        grad_ok,
        n_params,
    )
    n_iterations = 0
    if state == FIT_STATE_CONVERGED:
        chi_square, state, n_iterations = _lm_iterate_multichannel(
            kind,
            spots,
            index,
            variance,
            use_variance,
            jac,
            res,
            mle,
            tolerance,
            max_iterations,
            theta,
            theta_previous,
            grad,
            grad_ok,
            delta,
            scaling,
            hess,
            hess_ok,
            hess_damped,
            indxc,
            indxr,
            ipiv,
            n_params,
            chi_square,
        )

    _lm_finalize(
        theta,
        chi_square,
        state,
        n_iterations,
        mle,
        n_params,
        thetas,
        chi_squares,
        states,
        iterations,
        index,
    )


def n_parameters_multichannel(kind: int, n_channels: int) -> int:
    """Parameter count of a multichannel model.

    Parameters
    ----------
    kind : int
        :data:`MULTI_KIND_SHARED` or :data:`MULTI_KIND_DECOUPLED`.
    n_channels : int
        Number of channels fitted jointly. Only the decoupled model's count
        depends on it; the shared model has five parameters whatever it is.

    Returns
    -------
    n_params : int
        5 for the shared model, ``3 + 2 * n_channels`` for the decoupled one.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the two multichannel models.
    """
    if kind == MULTI_KIND_SHARED:
        return _N_PARAMS_MULTI_SHARED
    if kind == MULTI_KIND_DECOUPLED:
        return 3 + 2 * int(n_channels)
    raise ValueError(f"Unknown multichannel Gaussian model {kind}.")


def _check_inputs_multichannel(
    kind: int,
    spots: np.ndarray,
    jacobians: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
) -> int:
    """Validate the multichannel arrays and return the parameter count."""
    if spots.ndim != 4 or spots.shape[2] != spots.shape[3]:
        raise ValueError(
            "spots must be channel-major (n_spots, n_channels, box, box), got "
            f"{spots.shape}."
        )
    n_spots, n_channels = spots.shape[0], spots.shape[1]
    n_params = n_parameters_multichannel(kind, n_channels)
    if jacobians.shape != (n_spots, n_channels, 4):
        raise ValueError(
            f"jacobians must have shape {(n_spots, n_channels, 4)}, got "
            f"{jacobians.shape}."
        )
    if residuals.shape != (n_spots, n_channels, 2):
        raise ValueError(
            f"residuals must have shape {(n_spots, n_channels, 2)}, got "
            f"{residuals.shape}."
        )
    if initial_parameters.shape != (n_spots, n_params):
        raise ValueError(
            "initial_parameters must have shape "
            f"{(n_spots, n_params)}, got {initial_parameters.shape}."
        )
    return n_params


def _prepare_multichannel(
    kind: int,
    spots: np.ndarray,
    jacobians: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    tolerance: float | None,
    max_iterations: int | None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Validate and normalize the inputs both multichannel entry points
    share."""
    spots = np.asarray(spots)
    jacobians = np.asarray(jacobians)
    residuals = np.asarray(residuals)
    initial_parameters = np.asarray(initial_parameters)
    n_params = _check_inputs_multichannel(
        kind, spots, jacobians, residuals, initial_parameters
    )
    if tolerance is None:
        tolerance = TOLERANCE
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    jacobians = np.ascontiguousarray(jacobians, dtype=np.float64)
    residuals = np.ascontiguousarray(residuals, dtype=np.float64)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=4)
    outputs = allocate_outputs(len(spots), n_params)
    return (
        spots,
        variance,
        use_variance,
        jacobians,
        residuals,
        initial_parameters,
        float(tolerance),
        int(max_iterations),
        outputs,
    )


def _kernel_args_multichannel(
    variance: np.ndarray,
    use_variance: bool,
    kind: int,
    jacobians: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    outputs: tuple,
) -> tuple:
    """Freeze everything :func:`_fit_gauss_spot_multichannel` needs after
    ``index``."""
    thetas, chi_squares, states, iterations = outputs
    return (
        variance,
        bool(use_variance),
        int(kind),
        jacobians,
        residuals,
        initial_parameters,
        bool(mle),
        float(tolerance),
        int(max_iterations),
        thetas,
        chi_squares,
        states,
        iterations,
    )


def fit_spots_multichannel(
    kind: int,
    spots: np.ndarray,
    jacobians: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Jointly fit several registered channels with an isotropic Gaussian.

    Signature-compatible with
    :func:`picasso.fitting.gaussfit_cuda.fit_spots_multichannel`, so the two
    are interchangeable backends.

    Parameters
    ----------
    kind : int
        :data:`MULTI_KIND_SHARED` or :data:`MULTI_KIND_DECOUPLED`.
    spots : np.ndarray
        Channel-major ``(n_spots, n_channels, box, box)`` photon counts.
    jacobians : np.ndarray
        ``(n_spots, n_channels, 4)`` per-spot, per-channel local Jacobian
        ``[a00, a01, a10, a11]`` of the channel transform.
    residuals : np.ndarray
        ``(n_spots, n_channels, 2)`` sub-pixel ROI offsets in ``[x, y]``, the
        fractional part the integer box origin cannot express. Channel 0 is
        zero.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds, from
        ``picasso.fitting.seeds.initial_parameters_gauss_multichannel``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance, max_iterations : optional
        ``None`` uses :data:`TOLERANCE` / :data:`MAX_ITERATIONS`.
    progress_callback : callable, "console" or None, optional
        Reported per spot.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance laid out exactly like ``spots``.

    Returns
    -------
    thetas, chi_squares, states, iterations
    """
    (
        spots,
        variance,
        use_variance,
        jacobians,
        residuals,
        initial_parameters,
        tolerance,
        max_iterations,
        outputs,
    ) = _prepare_multichannel(
        kind,
        spots,
        jacobians,
        residuals,
        initial_parameters,
        tolerance,
        max_iterations,
        variance,
    )
    args = _kernel_args_multichannel(
        variance,
        use_variance,
        kind,
        jacobians,
        residuals,
        initial_parameters,
        mle,
        tolerance,
        max_iterations,
        outputs,
    )
    use_tqdm = progress_callback == "console"
    index_range = range(len(spots))
    if use_tqdm:
        index_range = tqdm(index_range, desc="Fitting", unit="spot")
    for index in index_range:
        _fit_gauss_spot_multichannel(spots, index, *args)
        if callable(progress_callback):
            progress_callback(index + 1)
    return outputs


def _worker_multichannel(
    spots: np.ndarray, args: tuple, current: list, lock, stop: np.ndarray
) -> None:
    """Worker for asynchronous multichannel fitting."""
    n_spots = len(spots)
    while True:
        with lock:
            if stop[0] or current[0] == n_spots:
                return
            index = current[0]
            current[0] += 1
        _fit_gauss_spot_multichannel(spots, index, *args)


def fit_spots_multichannel_async(
    kind: int,
    spots: np.ndarray,
    jacobians: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    n_threads: int | None = None,
    variance: np.ndarray | None = None,
) -> AsyncFit:
    """Jointly fit registered channels on several CPU threads.

    The multichannel counterpart of :func:`fit_spots_async`: it returns
    immediately and the caller polls the handle for progress, aborts it, or
    checks it for worker errors. The kernels are ``nogil``, so this is
    *threads*, not processes.

    See :func:`fit_spots_multichannel` for the full description of the
    arguments it shares with the serial fit.

    Parameters
    ----------
    kind : int
        :data:`MULTI_KIND_SHARED` or :data:`MULTI_KIND_DECOUPLED`.
    spots : np.ndarray
        Channel-major ``(n_spots, n_channels, box, box)`` photon counts.
    jacobians : np.ndarray
        ``(n_spots, n_channels, 4)`` per-spot, per-channel local Jacobian
        ``[a00, a01, a10, a11]`` of the channel transform.
    residuals : np.ndarray
        ``(n_spots, n_channels, 2)`` sub-pixel ROI offsets in ``[x, y]``.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance, max_iterations : optional
        ``None`` uses :data:`TOLERANCE` / :data:`MAX_ITERATIONS`.
    n_threads : int, optional
        Worker threads to run. None (the default) uses :func:`n_workers`.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance laid out exactly like ``spots``.

    Returns
    -------
    job : picasso.fitting.splinefit.AsyncFit
        Handle on the running fit. Its result arrays are filled in place, so
        they are only meaningful once ``job.finished()`` is True.
    """
    (
        spots,
        variance,
        use_variance,
        jacobians,
        residuals,
        initial_parameters,
        tolerance,
        max_iterations,
        outputs,
    ) = _prepare_multichannel(
        kind,
        spots,
        jacobians,
        residuals,
        initial_parameters,
        tolerance,
        max_iterations,
        variance,
    )
    args = _kernel_args_multichannel(
        variance,
        use_variance,
        kind,
        jacobians,
        residuals,
        initial_parameters,
        mle,
        tolerance,
        max_iterations,
        outputs,
    )
    n_spots = len(spots)
    if n_threads is None:
        n_threads = n_workers()
    n_threads = max(1, min(int(n_threads), max(n_spots, 1)))
    lock = threading.Lock()
    current = [0]
    stop = np.zeros(1, dtype=np.uint8)
    executor = futures.ThreadPoolExecutor(n_threads)
    fs = [
        executor.submit(_worker_multichannel, spots, args, current, lock, stop)
        for _ in range(n_threads)
    ]
    executor.shutdown(wait=False)
    return AsyncFit(current, *outputs, fs, stop)
