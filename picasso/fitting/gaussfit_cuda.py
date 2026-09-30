"""
picasso.fitting.gaussfit_cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU Gaussian PSF fitting: the GPU twin of :mod:`picasso.fitting.gaussfit`, over
the same Levenberg-Marquardt driver as :mod:`picasso.fitting.splinefit_cuda`.

Every model identifier, parameter count and schedule constant is imported from
the CPU module, so the two devices cannot disagree about what a model is; only
the device code lives here. The models are transcribed from Gpufit's
``models/gauss_2d.cuh``, ``gauss_2d_elliptic.cuh`` and
``gauss_2d_rotated.cuh``.

===================  ==========  ==============================================
model                parameters  layout
===================  ==========  ==============================================
:data:`SPHERICAL`             5  ``[peak, x, y, s, bg]``
:data:`ELLIPTIC`              6  ``[peak, x, y, sx, sy, bg]``
:data:`ROTATED`               7  ``[peak, x, y, sx, sy, bg, angle]``
===================  ==========  ==============================================

The amplitude is the Gaussian's *peak height*; ``picasso.localize`` converts it
to photons afterwards. There is no axial multi-start here - these models have
no axial coordinate - so each spot is fitted once from its initial parameters.

References
----------
The fitting algorithm and all three Gaussian models are a port of Gpufit
(``models/gauss_2d*.cuh``, ``cuda_kernels.cu``):

Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
License (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
from numba import cuda
from tqdm import tqdm

from picasso.fitting import lmfit_cuda
from picasso.fitting.splinefit import allocate_outputs, resolve_variance

# Everything that defines *what model this is* and *where a fit stops* comes
# from the CPU twin, so the two devices cannot drift apart - the same
# arrangement as ``lmfit_cuda`` importing its constants from ``splinefit``.
from picasso.fitting.gaussfit import (  # noqa: F401  (re-exported)
    ELLIPTIC,
    MAX_ITERATIONS,
    MULTI_KIND_DECOUPLED,
    MULTI_KIND_SHARED,
    ROTATED,
    SPHERICAL,
    TOLERANCE,
    _N_PARAMS,
    _N_PARAMS_MULTI_SHARED,
    _check_inputs_multichannel,
    n_parameters,
    n_parameters_multichannel,
)
from picasso.fitting.lmfit_cuda import (
    CUDA_THREADS,
    _INF,
    _estimator_terms_strict,
    make_fit_kernel,
    make_lm_driver,
)


def _make_accumulate_spherical(ftype):
    """Isotropic Gaussian, ``[photons, x, y, s, bg]`` (Gpufit's
    ``GAUSS_2D``)."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
        spots,
        index,
        variance,
        use_variance,
        coeff,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
    ):
        box = spots.shape[2]
        amp = theta[0]
        cx = theta[1]
        cy = theta[2]
        sigma = theta[3]
        offset = theta[4]
        if not (
            math.isfinite(amp)
            and math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(sigma)
            and math.isfinite(offset)
            # A zero width divides by zero in every derivative.
            and abs(sigma) > 0.0
        ):
            return _INF, False
        inv_s2 = ftype(1.0 / (sigma * sigma))
        inv_s3 = ftype(1.0 / (sigma * sigma * sigma))
        chi_square = 0.0
        g0 = g1 = g2 = g3 = g4 = 0.0
        h00 = h01 = h02 = h03 = h04 = 0.0
        h11 = h12 = h13 = h14 = 0.0
        h22 = h23 = h24 = 0.0
        h33 = h34 = 0.0
        h44 = 0.0
        for j in range(box):
            dy = ftype(j - cy)
            for i in range(box):
                dx = ftype(i - cx)
                ex = math.exp(-half * (dx * dx + dy * dy) * inv_s2)
                value = amp * ex + offset
                data = spots[index, 0, j, i]
                d0 = ex
                d1 = amp * ex * dx * inv_s2
                d2 = amp * ex * dy * inv_s2
                d3 = amp * ex * (dx * dx + dy * dy) * inv_s3
                # d4 == 1 (offset).
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms_strict(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


def _make_accumulate_elliptic(ftype):
    """Elliptic Gaussian, ``[photons, x, y, sx, sy, bg]``."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
        spots,
        index,
        variance,
        use_variance,
        coeff,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
    ):
        box = spots.shape[2]
        amp = theta[0]
        cx = theta[1]
        cy = theta[2]
        sx = theta[3]
        sy = theta[4]
        offset = theta[5]
        if not (
            math.isfinite(amp)
            and math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(sx)
            and math.isfinite(sy)
            and math.isfinite(offset)
            and abs(sx) > 0.0
            and abs(sy) > 0.0
        ):
            return _INF, False
        inv_sx2 = ftype(1.0 / (sx * sx))
        inv_sy2 = ftype(1.0 / (sy * sy))
        inv_sx3 = ftype(1.0 / (sx * sx * sx))
        inv_sy3 = ftype(1.0 / (sy * sy * sy))
        chi_square = 0.0
        g0 = g1 = g2 = g3 = g4 = g5 = 0.0
        h00 = h01 = h02 = h03 = h04 = h05 = 0.0
        h11 = h12 = h13 = h14 = h15 = 0.0
        h22 = h23 = h24 = h25 = 0.0
        h33 = h34 = h35 = 0.0
        h44 = h45 = 0.0
        h55 = 0.0
        for j in range(box):
            dy = ftype(j - cy)
            for i in range(box):
                dx = ftype(i - cx)
                ex = math.exp(-half * (dx * dx * inv_sx2 + dy * dy * inv_sy2))
                value = amp * ex + offset
                data = spots[index, 0, j, i]
                d0 = ex
                d1 = amp * ex * dx * inv_sx2
                d2 = amp * ex * dy * inv_sy2
                d3 = amp * ex * dx * dx * inv_sx3
                d4 = amp * ex * dy * dy * inv_sy3
                # d5 == 1 (offset).
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms_strict(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


def _make_accumulate_rotated(ftype):
    """Rotated elliptic Gaussian, ``[photons, x, y, sx, sy, bg, angle]``.

    The angle derivative vanishes identically when ``sx == sy``, which makes
    the first Hessian singular;
    ``picasso.fitting.seeds.initial_parameters_gauss`` breaks that symmetry in
    its seed on purpose, and this model relies on it."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
        spots,
        index,
        variance,
        use_variance,
        coeff,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
    ):
        box = spots.shape[2]
        amp = theta[0]
        cx = theta[1]
        cy = theta[2]
        sx = theta[3]
        sy = theta[4]
        offset = theta[5]
        angle = theta[6]
        if not (
            math.isfinite(amp)
            and math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(sx)
            and math.isfinite(sy)
            and math.isfinite(offset)
            and math.isfinite(angle)
            and abs(sx) > 0.0
            and abs(sy) > 0.0
        ):
            return _INF, False
        cos_a = ftype(math.cos(angle))
        sin_a = ftype(math.sin(angle))
        inv_sx2 = ftype(1.0 / (sx * sx))
        inv_sy2 = ftype(1.0 / (sy * sy))
        inv_sx3 = ftype(1.0 / (sx * sx * sx))
        inv_sy3 = ftype(1.0 / (sy * sy * sy))
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
            dy = ftype(j - cy)
            for i in range(box):
                dx = ftype(i - cx)
                arga = dx * cos_a - dy * sin_a
                argb = dx * sin_a + dy * cos_a
                ex = math.exp(
                    -half * (arga * arga * inv_sx2 + argb * argb * inv_sy2)
                )
                value = amp * ex + offset
                data = spots[index, 0, j, i]
                d0 = ex
                d1 = (
                    amp * cos_a * arga * inv_sx2 + amp * sin_a * argb * inv_sy2
                ) * ex
                d2 = (
                    -amp * sin_a * arga * inv_sx2
                    + amp * cos_a * argb * inv_sy2
                ) * ex
                d3 = amp * arga * arga * inv_sx3 * ex
                d4 = amp * argb * argb * inv_sy3 * ex
                # d5 == 1 (offset).
                d6 = amp * arga * argb * (inv_sx2 - inv_sy2) * ex
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms_strict(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


_ACCUMULATORS = {
    SPHERICAL: _make_accumulate_spherical,
    ELLIPTIC: _make_accumulate_elliptic,
    ROTATED: _make_accumulate_rotated,
}

_KERNEL_CACHE: dict = {}


def _get_kernel(model: int, single_precision: bool):
    """Memoized fit kernel for one model and precision."""
    key = (int(model), bool(single_precision))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        if model not in _ACCUMULATORS:
            raise ValueError(f"Unknown Gaussian model {model}.")
        ftype = np.float32 if single_precision else np.float64
        accumulate = _ACCUMULATORS[model](ftype)
        # No axial coordinate, so no multi-start: ``z_col`` is unused.
        driver = make_lm_driver(
            accumulate, _N_PARAMS[model], 0, seedable=False
        )
        kernel = make_fit_kernel(driver)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _check_inputs(
    model: int, spots: np.ndarray, initial_parameters: np.ndarray
) -> int:
    """Validate ``spots``/``initial_parameters``.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.
    spots : np.ndarray
        Spots, expected with shape ``(n_spots, box, box)``.
    initial_parameters : np.ndarray
        Seeds, expected with shape ``(n_spots, n_params)``.

    Returns
    -------
    n_params : int
        Parameter count of ``model``.
    """
    if spots.ndim != 3 or spots.shape[1] != spots.shape[2]:
        raise ValueError(
            f"spots must have shape (n_spots, box, box), got {spots.shape}."
        )
    n_params = _N_PARAMS[model]
    if initial_parameters.shape != (len(spots), n_params):
        raise ValueError(
            "initial_parameters must have shape "
            f"{(len(spots), n_params)}, got {initial_parameters.shape}."
        )
    return n_params


def _resolve_schedule(
    tolerance: float | None, max_iterations: int | None
) -> tuple:
    """Fall back to the module defaults for an unset tolerance/iterations."""
    if tolerance is None:
        tolerance = TOLERANCE
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS
    return tolerance, max_iterations


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
    abort_callback: Callable[[], bool] | None = None,
    single_precision: bool = True,
    variance: np.ndarray | None = None,
) -> tuple:
    """Fit spots with a 2D Gaussian model on the GPU.

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
        cumulative number of spots fitted. As in
        :func:`picasso.fitting.splinefit_cuda.fit_spots`.
    abort_callback : callable or None, optional
        Polled between chunks; returning True stops the fit. As in
        :func:`picasso.fitting.splinefit_cuda.fit_spots`.
    single_precision : bool, optional
        Evaluate the model in single precision (the default). As in
        :func:`picasso.fitting.splinefit_cuda.fit_spots`.
    variance : np.ndarray, optional
        ``(n_spots, box, box)`` per-pixel sCMOS readout variance in
        photoelectrons squared, laid out exactly like ``spots``. ``None``
        (the default) fits the plain Poisson model.

    Returns
    -------
    thetas : np.ndarray
        ``(n_spots, n_params)`` fitted parameters.
    chi_squares : np.ndarray
        ``(n_spots,)`` chi-square at the optimum.
    states : np.ndarray
        ``(n_spots,)`` per-spot fit state, using Gpufit's codes (see
        ``splinefit.FIT_STATE_CONVERGED``).
    iterations : np.ndarray
        ``(n_spots,)`` iterations used.
    """
    lmfit_cuda.require_cuda()
    spots = np.asarray(spots)
    n_params = _check_inputs(model, spots, initial_parameters)
    tolerance, max_iterations = _resolve_schedule(tolerance, max_iterations)

    n_spots, box = spots.shape[0], spots.shape[1]
    thetas, chi_squares, states, iterations = allocate_outputs(
        n_spots, n_params
    )
    if n_spots == 0:
        return thetas, chi_squares, states, iterations

    # The shared driver and the accumulators index spots channel-major, as the
    # spline models do; a Gaussian fit is the single-channel case. This is a
    # reshape of a C-contiguous array, so it costs nothing.
    spots = np.ascontiguousarray(spots, dtype=np.float32).reshape(
        n_spots, 1, box, box
    )
    # The variance patch is cut from the same ROIs and rides along in exactly
    # the same layout, so the same free reshape applies.
    variance, use_variance = resolve_variance(
        variance, (n_spots, box, box), ndim=4
    )
    if use_variance:
        variance = variance.reshape(n_spots, 1, box, box)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )

    kernel = _get_kernel(model, single_precision)
    # The driver takes the spline models' coefficient, Jacobian and residual
    # arrays; a Gaussian has no channel geometry, so these are unused
    # placeholders of the right rank.
    d_unused_coeff = cuda.to_device(np.zeros((1, 1, 1, 4, 4)))
    d_unused_jac = cuda.to_device(np.zeros((1, 4)))
    d_seeds = cuda.to_device(np.zeros(1))
    # Without a noise model the variance is a four-byte dummy, uploaded once
    # rather than per chunk.
    d_dummy_variance = None if use_variance else cuda.to_device(variance)

    bytes_per_row = 4 * box * box + 8 * (2 * n_params + 2) + 16
    if use_variance:
        bytes_per_row += 4 * box * box
    chunk = min(n_spots, lmfit_cuda.chunk_rows(bytes_per_row, max_iterations))

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    pbar = (
        tqdm(total=n_spots, desc="Fitting", unit="spot") if use_tqdm else None
    )
    try:
        for start in range(0, n_spots, chunk):
            if abort_callback is not None and abort_callback():
                break
            stop = min(start + chunk, n_spots)
            n = stop - start
            d_spots = cuda.to_device(spots[start:stop])
            d_variance = (
                cuda.to_device(variance[start:stop])
                if use_variance
                else d_dummy_variance
            )
            d_init = cuda.to_device(initial_parameters[start:stop])
            d_res = cuda.to_device(np.zeros((n, 1, 2)))
            d_thetas = cuda.device_array((n, n_params), dtype=np.float64)
            d_chi = cuda.device_array(n, dtype=np.float64)
            d_states = cuda.device_array(n, dtype=np.int32)
            d_iterations = cuda.device_array(n, dtype=np.int32)
            blocks = (n + CUDA_THREADS - 1) // CUDA_THREADS
            kernel[blocks, CUDA_THREADS](
                d_spots,
                d_variance,
                use_variance,
                d_unused_coeff,
                d_unused_jac,
                d_res,
                d_init,
                d_seeds,
                False,
                bool(mle),
                float(tolerance),
                int(max_iterations),
                d_thetas,
                d_chi,
                d_states,
                d_iterations,
            )
            thetas[start:stop] = d_thetas.copy_to_host()
            chi_squares[start:stop] = d_chi.copy_to_host()
            states[start:stop] = d_states.copy_to_host()
            iterations[start:stop] = d_iterations.copy_to_host()
            if use_tqdm:
                pbar.update(n)
            elif do_callback:
                progress_callback(stop)
    finally:
        if use_tqdm:
            pbar.close()
    return thetas, chi_squares, states, iterations


# ----------------------------------------------------------------------
# Multichannel (joint) spherical Gaussian
#
# The GPU twin of ``gaussfit._accumulate_spherical_multichannel`` and
# ``_accumulate_spherical_decoupled``. Unlike the single-channel models above,
# these use the driver's ``jac`` and ``res`` arguments for real: each channel
# sees the shared lateral shift through its own local Jacobian and sits on its
# own sub-pixel ROI offset. ``coeff`` is the unused placeholder here, the exact
# inverse of the single-channel case.
# ----------------------------------------------------------------------


def _make_accumulate_spherical_multichannel(ftype):
    """Shared-amplitude multichannel isotropic Gaussian,
    ``[amplitude, x_shift, y_shift, sigma, offset]``.

    One kernel serves any channel count: the parameter count is fixed, so the
    channels are looped at run time."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
        spots,
        index,
        variance,
        use_variance,
        coeff,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
    ):
        n_channels = spots.shape[1]
        box = spots.shape[2]
        amp = theta[0]
        x_shift = theta[1]
        y_shift = theta[2]
        sigma = theta[3]
        offset = theta[4]
        if not (
            math.isfinite(amp)
            and math.isfinite(x_shift)
            and math.isfinite(y_shift)
            and math.isfinite(sigma)
            and math.isfinite(offset)
            # A zero width divides by zero in every derivative.
            and abs(sigma) > 0.0
        ):
            return _INF, False
        inv_s2 = ftype(1.0 / (sigma * sigma))
        inv_s3 = ftype(1.0 / (sigma * sigma * sigma))
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
            a00 = jac[index, ch, 0]
            a01 = jac[index, ch, 1]
            a10 = jac[index, ch, 2]
            a11 = jac[index, ch, 3]
            sx = center + a00 * dx_shift + a01 * dy_shift + res[index, ch, 0]
            sy = center + a10 * dx_shift + a11 * dy_shift + res[index, ch, 1]
            for j in range(box):
                pos_y = ftype(j - sy)
                for i in range(box):
                    pos_x = ftype(i - sx)
                    r2 = pos_x * pos_x + pos_y * pos_y
                    ex = math.exp(-half * r2 * inv_s2)
                    value = amp * ex + offset
                    data = spots[index, ch, j, i]
                    gx = -ex * pos_x * inv_s2
                    gy = -ex * pos_y * inv_s2
                    d0 = ex
                    d1 = -amp * (a00 * gx + a10 * gy)
                    d2 = -amp * (a01 * gx + a11 * gy)
                    d3 = amp * ex * r2 * inv_s3
                    # d4 == 1 (offset).
                    var = 0.0
                    if use_variance:
                        var = variance[index, ch, j, i]
                    contrib, weight, factor, ok = _estimator_terms_strict(
                        mle, value, data, var
                    )
                    if not ok:
                        return _INF, False
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

    return accumulate


@cuda.jit(device=True)
def _reset_decoupled_scratch(theta, n_params, grad, hess):
    """Zero ``grad``/``hess`` for the decoupled accumulator.

    Returns False if any parameter is non-finite, in which case ``grad``/
    ``hess`` are left as found."""
    for p in range(n_params):
        if not math.isfinite(theta[p]):
            return False
        grad[p] = 0.0
        for q in range(n_params):
            hess[p, q] = 0.0
    return True


@cuda.jit(device=True)
def _mirror_upper_triangle(hess, n_params):
    """Mirror the upper triangle of ``hess`` into the lower triangle."""
    for p in range(n_params):
        for q in range(p):
            hess[p, q] = hess[q, p]


def _make_accumulate_spherical_decoupled(ftype, n_channels: int):
    """Photon-decoupled multichannel isotropic Gaussian at a fixed channel
    count, ``[x_shift, y_shift, sigma, N_0..N_{C-1}, bg_0..bg_{C-1}]``.

    A pixel of channel ``ch`` touches only five of the ``3 + 2C`` parameters,
    so - as in ``splinefit_cuda._make_accumulate_link_xyz`` - the fifteen
    affected quantities live in registers and are written out once per channel
    rather than round-tripping through local memory per pixel. The summation
    order matches ``gaussfit._accumulate_spherical_decoupled``, so the two
    devices agree to rounding."""
    half = ftype(0.5)
    n_ch = n_channels

    @cuda.jit(device=True)
    def accumulate(
        spots,
        index,
        variance,
        use_variance,
        coeff,
        jac,
        res,
        theta,
        mle,
        hess,
        grad,
    ):
        box = spots.shape[2]
        n_params = 3 + 2 * n_ch
        x_shift = theta[0]
        y_shift = theta[1]
        sigma = theta[2]
        if not _reset_decoupled_scratch(theta, n_params, grad, hess):
            return _INF, False
        if not abs(sigma) > 0.0:
            return _INF, False
        inv_s2 = ftype(1.0 / (sigma * sigma))
        inv_s3 = ftype(1.0 / (sigma * sigma * sigma))
        chi_square = 0.0
        # Shared x/y/sigma block, accumulated across every channel.
        g0 = g1 = g2 = 0.0
        h00 = h01 = h02 = 0.0
        h11 = h12 = 0.0
        h22 = 0.0
        # The channel Jacobian linearizes the *displacement* of the
        # emitter from the box center, not the box coordinate itself:
        # channel c sees center + J @ (shift - center) + residual. Off by
        # the center, a mirrored or rotated channel lands outside its own
        # box and its photon count runs away.
        center = 0.5 * box - 0.5
        dx_shift = x_shift - center
        dy_shift = y_shift - center
        for ch in range(n_ch):
            amp = theta[3 + ch]
            offset = theta[3 + n_ch + ch]
            ia = 3 + ch
            ib = 3 + n_ch + ch
            a00 = jac[index, ch, 0]
            a01 = jac[index, ch, 1]
            a10 = jac[index, ch, 2]
            a11 = jac[index, ch, 3]
            sx = center + a00 * dx_shift + a01 * dy_shift + res[index, ch, 0]
            sy = center + a10 * dx_shift + a11 * dy_shift + res[index, ch, 1]
            # This channel's own photon (a) and background (b) entries.
            ga = gb = 0.0
            h0a = h1a = h2a = 0.0
            h0b = h1b = h2b = 0.0
            haa = hab = hbb = 0.0
            for j in range(box):
                pos_y = ftype(j - sy)
                for i in range(box):
                    pos_x = ftype(i - sx)
                    r2 = pos_x * pos_x + pos_y * pos_y
                    ex = math.exp(-half * r2 * inv_s2)
                    value = amp * ex + offset
                    data = spots[index, ch, j, i]
                    gx = -ex * pos_x * inv_s2
                    gy = -ex * pos_y * inv_s2
                    d0 = -amp * (a00 * gx + a10 * gy)
                    d1 = -amp * (a01 * gx + a11 * gy)
                    d2 = amp * ex * r2 * inv_s3
                    # d(value)/d(N_ch) = ex, d(value)/d(bg_ch) = 1; both zero
                    # for every other channel.
                    var = 0.0
                    if use_variance:
                        var = variance[index, ch, j, i]
                    contrib, weight, factor, ok = _estimator_terms_strict(
                        mle, value, data, var
                    )
                    if not ok:
                        return _INF, False
                    chi_square += contrib
                    g0 += d0 * factor
                    g1 += d1 * factor
                    g2 += d2 * factor
                    ga += ex * factor
                    gb += factor
                    w0 = weight * d0
                    w1 = weight * d1
                    w2 = weight * d2
                    wa = weight * ex
                    h00 += w0 * d0
                    h01 += w0 * d1
                    h02 += w0 * d2
                    h0a += w0 * ex
                    h0b += w0
                    h11 += w1 * d1
                    h12 += w1 * d2
                    h1a += w1 * ex
                    h1b += w1
                    h22 += w2 * d2
                    h2a += w2 * ex
                    h2b += w2
                    haa += wa * ex
                    hab += wa
                    hbb += weight
            grad[ia] = ga
            grad[ib] = gb
            hess[0, ia] = h0a
            hess[1, ia] = h1a
            hess[2, ia] = h2a
            hess[0, ib] = h0b
            hess[1, ib] = h1b
            hess[2, ib] = h2b
            hess[ia, ia] = haa
            hess[ia, ib] = hab
            hess[ib, ib] = hbb
        grad[0] = g0
        grad[1] = g1
        grad[2] = g2
        hess[0, 0] = h00
        hess[0, 1] = h01
        hess[0, 2] = h02
        hess[1, 1] = h11
        hess[1, 2] = h12
        hess[2, 2] = h22
        # Only the upper triangle was filled (0 < 1 < 2 < ia < ib always
        # holds); mirror it once at the end. The cross-channel photon and
        # background blocks are structurally zero and were cleared above.
        _mirror_upper_triangle(hess, n_params)
        return chi_square, True

    return accumulate


_KERNEL_CACHE_MULTI: dict = {}


def _build_kernel_multichannel(
    kind: int, n_channels: int, single_precision: bool
):
    """Compile the multichannel fit kernel for one model, channel count and
    precision."""
    ftype = np.float32 if single_precision else np.float64
    if kind == MULTI_KIND_SHARED:
        accumulate = _make_accumulate_spherical_multichannel(ftype)
        n_params = _N_PARAMS_MULTI_SHARED
    elif kind == MULTI_KIND_DECOUPLED:
        accumulate = _make_accumulate_spherical_decoupled(ftype, n_channels)
        n_params = 3 + 2 * int(n_channels)
    else:
        raise ValueError(f"Unknown multichannel Gaussian model {kind}.")
    # No axial coordinate, so no multi-start: ``z_col`` is unused.
    driver = make_lm_driver(accumulate, n_params, 0, seedable=False)
    return make_fit_kernel(driver)


def _get_kernel_multichannel(
    kind: int, n_channels: int, single_precision: bool
):
    """Memoized :func:`_build_kernel_multichannel`.

    Only the photon-decoupled model needs one kernel per channel count - its
    parameter count is ``3 + 2C``, and a device-local array needs a
    compile-time shape. The shared-amplitude model has a fixed parameter count
    and loops over channels at run time, so a single kernel serves every
    channel count."""
    key = (
        int(kind),
        int(n_channels) if kind == MULTI_KIND_DECOUPLED else 0,
        bool(single_precision),
    )
    kernel = _KERNEL_CACHE_MULTI.get(key)
    if kernel is None:
        kernel = _build_kernel_multichannel(kind, n_channels, single_precision)
        _KERNEL_CACHE_MULTI[key] = kernel
    return kernel


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
    abort_callback: Callable[[], bool] | None = None,
    single_precision: bool = True,
    variance: np.ndarray | None = None,
) -> tuple | None:
    """Jointly fit several registered channels on the GPU.

    Signature-compatible with
    :func:`picasso.fitting.gaussfit.fit_spots_multichannel`, so the two are
    interchangeable backends; both run the same algorithm, so the choice only
    affects speed.

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
        Channel 0 is zero.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds, from
        ``picasso.fitting.seeds.initial_parameters_gauss_multichannel``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance, max_iterations : optional
        ``None`` uses :data:`TOLERANCE` / :data:`MAX_ITERATIONS`.
    progress_callback : callable, "console" or None, optional
        Reported once per chunk, not per spot: one launch fits a whole chunk.
    abort_callback : callable, optional
        Polled between chunks; returning True stops the fit.
    single_precision : bool, optional
        Compile the kernel in float32 (default) or float64. Double precision is
        what makes a bit-for-bit comparison against the CPU backend meaningful.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like ``spots``. ``None`` (the default) fits the plain Poisson
        model.

    Returns
    -------
    thetas, chi_squares, states, iterations
        As :func:`picasso.fitting.gaussfit.fit_spots_multichannel`, using
        Gpufit's state codes (see
        ``picasso.fitting.splinefit.FIT_STATE_CONVERGED``).
    None
        If ``abort_callback`` asked to stop before every chunk ran.
    """
    lmfit_cuda.require_cuda()
    spots = np.asarray(spots)
    jacobians = np.asarray(jacobians)
    residuals = np.asarray(residuals)
    initial_parameters = np.asarray(initial_parameters)
    n_params = _check_inputs_multichannel(
        kind, spots, jacobians, residuals, initial_parameters
    )
    tolerance, max_iterations = _resolve_schedule(tolerance, max_iterations)

    n_spots, n_channels, box = spots.shape[0], spots.shape[1], spots.shape[2]
    thetas, chi_squares, states, iterations = allocate_outputs(
        n_spots, n_params
    )
    if n_spots == 0:
        return thetas, chi_squares, states, iterations

    spots = np.ascontiguousarray(spots, dtype=np.float32)
    jacobians = np.ascontiguousarray(jacobians, dtype=np.float64)
    residuals = np.ascontiguousarray(residuals, dtype=np.float64)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=4)

    kernel = _get_kernel_multichannel(kind, n_channels, single_precision)
    # These models use jac/res for real; it is the spline coefficient table
    # that has no counterpart here, so that is the unused placeholder.
    d_unused_coeff = cuda.to_device(np.zeros((1, 1, 1, 4, 4)))
    d_seeds = cuda.to_device(np.zeros(1))
    d_dummy_variance = None if use_variance else cuda.to_device(variance)

    bytes_per_row = (
        4 * n_channels * box * box
        + 8 * (2 * n_params + 2)
        + 8 * n_channels * 6
        + 16
    )
    if use_variance:
        bytes_per_row += 4 * n_channels * box * box
    chunk = min(n_spots, lmfit_cuda.chunk_rows(bytes_per_row, max_iterations))

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    pbar = (
        tqdm(total=n_spots, desc="Fitting", unit="spot") if use_tqdm else None
    )
    aborted = False
    try:
        for start in range(0, n_spots, chunk):
            if abort_callback is not None and abort_callback():
                aborted = True
                break
            stop = min(start + chunk, n_spots)
            n = stop - start
            d_spots = cuda.to_device(spots[start:stop])
            d_variance = (
                cuda.to_device(variance[start:stop])
                if use_variance
                else d_dummy_variance
            )
            d_init = cuda.to_device(initial_parameters[start:stop])
            d_jac = cuda.to_device(jacobians[start:stop])
            d_res = cuda.to_device(residuals[start:stop])
            d_thetas = cuda.device_array((n, n_params), dtype=np.float64)
            d_chi = cuda.device_array(n, dtype=np.float64)
            d_states = cuda.device_array(n, dtype=np.int32)
            d_iterations = cuda.device_array(n, dtype=np.int32)
            blocks = (n + CUDA_THREADS - 1) // CUDA_THREADS
            kernel[blocks, CUDA_THREADS](
                d_spots,
                d_variance,
                use_variance,
                d_unused_coeff,
                d_jac,
                d_res,
                d_init,
                d_seeds,
                False,
                bool(mle),
                float(tolerance),
                int(max_iterations),
                d_thetas,
                d_chi,
                d_states,
                d_iterations,
            )
            thetas[start:stop] = d_thetas.copy_to_host()
            chi_squares[start:stop] = d_chi.copy_to_host()
            states[start:stop] = d_states.copy_to_host()
            iterations[start:stop] = d_iterations.copy_to_host()
            if use_tqdm:
                pbar.update(n)
            elif do_callback:
                progress_callback(stop)
    finally:
        if use_tqdm:
            pbar.close()
    if aborted:
        return None
    return thetas, chi_squares, states, iterations
