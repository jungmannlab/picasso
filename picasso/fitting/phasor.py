"""
picasso.fitting.phasor
~~~~~~~~~~~~~~~~~~~~~~

Phasor localization (pSMLM) of Martens et al. (2018): a non-iterative,
model-free estimate of an emitter's position from the first Fourier
coefficients of its box, with the astigmatism of the spot as a by-product.

For a box of ``P`` pixels (``P`` odd) with the pixel coordinates
``k = 0, ..., P - 1`` counted from the center pixel ``P // 2``, and
``omega = 2 pi / P``, the box is summed over its rows (giving the x profile)
and over its columns (giving the y profile), and the first Fourier
coefficient of each profile is taken::

    F_x = sum_k profile_x[k] exp(-i omega k)

The angle of ``F_x`` gives the position, ``x = -angle(F_x) / omega`` pixels
from the center pixel (Martens et al., Eq. S3), and its magnitude falls as
the spot widens in x. A uniform background adds nothing to ``F_x`` (the
first harmonic of a constant is zero), which is why the estimate needs no
background term. The phasor magnitudes this module reports, ``|F_x| / F_0``
and ``|F_y| / F_0`` with ``F_0`` the sum of the box, are the normalized ones
of the paper's Figs. S6-S8; their ratio does not depend on the background or
on the brightness of the spot, and an astigmatic calibration of the ratio
against z gives z (see :func:`picasso.zfit.calibrate_z_phasor`).

The photons and the background come from aperture photometry, as in the
authors' ThunderSTORM implementation (paper, Sec. S11): the pixels within
``P // 2`` of the center pixel are the signal; those beyond it, up to
``P // 2 + 2`` pixels away, the background ring. The background per pixel is
the 56th percentile of the ring (the value at the 0-based index
``ceil(0.56 n_ring)`` of the sorted ring), the photons are the signal sum
less ``n_signal`` times the background, and at least 0. The radii follow
the authors' code and the paper's Fig. S11, which differ from the wording
of Sec. S11.

The published method gives no localization precision. Picasso adds one
(not part of the paper): the variance of each pixel - its photons, doubled
by an EMCCD's excess noise, plus an sCMOS pixel's readout variance - is
propagated through the closed-form estimator to first order. With
``theta_x = angle(F_x)`` and the per-column variance sums ``V_x[k]``::

    var(x) = sum_k sin^2(omega k + theta_x) V_x[k] / (|F_x|^2 omega^2)

and similarly for y and for the log of the magnitude ratio, whose
derivative with respect to a pixel is
``cos(omega k_x + theta_x) / |F_x| - cos(omega k_y + theta_y) / |F_y|``.
This is the precision of this estimator, not a Cramer-Rao bound, and it
needs no PSF model; see :func:`picasso.fitting.precision.phasor_precision`
for what it covers.

The kernel computes all of this in one pass per spot, on the CPU (Numba),
optionally on all cores.

This module is documented for reference only and is not part of the public
API; :func:`picasso.localize.fit` runs it (``fitting_method="phasor"``).

References
----------
Martens, K. J. A., Bader, A. N., Baas, S., Rieger, B. & Hohlbein, J.
"Phasor based single-molecule localization microscopy in 3D (pSMLM-3D): An
algorithm for MHz localization rates using standard CPUs." J. Chem. Phys.
148, 123311 (2018). https://doi.org/10.1063/1.5005899

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numba
import numpy as np

from picasso import lib

__all__ = [
    "BACKGROUND_PERCENTILE",
    "RING_WIDTH",
    "COLUMNS",
    "aperture_masks",
    "fit_spots",
]

#: Percentile of the background ring taken as the background per pixel
#: (paper, Sec. S11).
BACKGROUND_PERCENTILE = 0.56

#: Width in pixels of the background ring around the signal disc, as in
#: the authors' implementation.
RING_WIDTH = 2

#: The columns of :func:`fit_spots`' result, in order: the offsets of the
#: emitter from the center pixel (pixels), the photons and the background
#: per pixel (photons), the normalized phasor magnitudes, the propagated
#: variances of the offsets (pixels squared) and of the log of the
#: magnitude ratio ``|F_x| / |F_y|``.
COLUMNS = (
    "x",
    "y",
    "photons",
    "bg",
    "magnitude_x",
    "magnitude_y",
    "var_x",
    "var_y",
    "var_log_ratio",
)
_N_COLUMNS = len(COLUMNS)

# spots per kernel call, i.e., per progress update and abort check
_CHUNK = 1 << 16


def aperture_masks(box: int) -> tuple[lib.BoolArray2D, lib.BoolArray2D]:
    """The pixels of the aperture photometry in a box.

    Parameters
    ----------
    box : int
        Box side length in pixels, odd and at least 3.

    Returns
    -------
    signal, background : lib.BoolArray2D
        ``(box, box)`` masks of the signal pixels (at most ``box // 2``
        pixels from the center pixel) and of the background ring (more
        than ``box // 2`` and at most ``box // 2 + RING_WIDTH`` pixels
        away).
    """
    box = _check_box(box)
    r = box // 2
    k = np.arange(box) - r
    d = np.hypot(k[:, None], k[None, :])
    signal = d <= r
    background = (d > r) & (d <= r + RING_WIDTH)
    return signal, background


def _check_box(box: int) -> int:
    box = int(box)
    if box < 3 or box % 2 == 0:
        raise ValueError(
            f"Phasor localization needs an odd box of at least 3 pixels, "
            f"got {box}."
        )
    return box


def _tables(box: int) -> tuple:
    """cos/sin of ``omega k`` and the flat pixel indices of the aperture,
    for the kernel."""
    r = box // 2
    omega = 2 * np.pi / box
    k = np.arange(box) - r
    cos_t = np.cos(omega * k)
    sin_t = np.sin(omega * k)
    signal, background = aperture_masks(box)
    signal_idx = np.flatnonzero(signal).astype(np.int64)
    ring_idx = np.flatnonzero(background).astype(np.int64)
    bg_rank = int(math.ceil(BACKGROUND_PERCENTILE * len(ring_idx)))
    return cos_t, sin_t, signal_idx, ring_idx, min(bg_rank, len(ring_idx) - 1)


# error_model="numpy": an empty box gives inf/NaN instead of raising
@numba.njit(nogil=True, cache=True, error_model="numpy")
def _fit_spot(
    spot,
    var,
    use_var,
    em_factor,
    cos_t,
    sin_t,
    omega,
    signal_idx,
    ring_idx,
    bg_rank,
    col_sum,
    row_sum,
    col_var,
    row_var,
    ring,
    out,
):
    """Phasor estimates of one ``(P, P)`` spot (``[y, x]``) into ``out``
    (``COLUMNS``). ``col_sum`` ... ``ring`` are scratch buffers."""
    size = spot.shape[0]
    for j in range(size):
        col_sum[j] = 0.0
        row_sum[j] = 0.0
        col_var[j] = 0.0
        row_var[j] = 0.0
    total = 0.0
    for i in range(size):
        for j in range(size):
            value = spot[i, j]
            v = em_factor * value if value > 0 else 0.0
            if use_var:
                v += var[i, j]
            col_sum[j] += value
            row_sum[i] += value
            col_var[j] += v
            row_var[i] += v
            total += value

    # first Fourier coefficients of the x and y profiles
    re_x = 0.0
    im_x = 0.0
    re_y = 0.0
    im_y = 0.0
    for j in range(size):
        re_x += col_sum[j] * cos_t[j]
        im_x -= col_sum[j] * sin_t[j]
        re_y += row_sum[j] * cos_t[j]
        im_y -= row_sum[j] * sin_t[j]
    abs_x = math.hypot(re_x, im_x)
    abs_y = math.hypot(re_y, im_y)
    out[0] = -math.atan2(im_x, re_x) / omega
    out[1] = -math.atan2(im_y, re_y) / omega

    # aperture photometry
    flat = spot.ravel()
    signal = 0.0
    for m in range(signal_idx.shape[0]):
        signal += flat[signal_idx[m]]
    for m in range(ring_idx.shape[0]):
        ring[m] = flat[ring_idx[m]]
    ring.sort()
    bg = ring[bg_rank]
    out[2] = max(0.0, signal - signal_idx.shape[0] * bg)
    out[3] = bg

    out[4] = abs_x / total
    out[5] = abs_y / total

    # propagated variances; sin/cos(omega k + theta) from the tables,
    # with cos/sin(theta) = re/|F|, im/|F|
    cx = re_x / abs_x
    sx = im_x / abs_x
    cy = re_y / abs_y
    sy = im_y / abs_y
    var_x = 0.0
    var_y = 0.0
    for j in range(size):
        s_x = sin_t[j] * cx + cos_t[j] * sx
        s_y = sin_t[j] * cy + cos_t[j] * sy
        var_x += s_x * s_x * col_var[j]
        var_y += s_y * s_y * row_var[j]
    out[6] = var_x / (abs_x * abs_x * omega * omega)
    out[7] = var_y / (abs_y * abs_y * omega * omega)
    var_log_ratio = 0.0
    for i in range(size):
        g_y = (cos_t[i] * cy - sin_t[i] * sy) / abs_y
        for j in range(size):
            g = (cos_t[j] * cx - sin_t[j] * sx) / abs_x - g_y
            value = spot[i, j]
            v = em_factor * value if value > 0 else 0.0
            if use_var:
                v += var[i, j]
            var_log_ratio += g * g * v
    out[8] = var_log_ratio


@numba.njit(nogil=True, cache=True, error_model="numpy")
def _fit_spots_serial(
    spots,
    var,
    use_var,
    em_factor,
    cos_t,
    sin_t,
    omega,
    signal_idx,
    ring_idx,
    bg_rank,
    out,
):
    size = spots.shape[1]
    col_sum = np.empty(size)
    row_sum = np.empty(size)
    col_var = np.empty(size)
    row_var = np.empty(size)
    ring = np.empty(ring_idx.shape[0])
    for n in range(spots.shape[0]):
        _fit_spot(
            spots[n],
            var[n] if use_var else var[0],
            use_var,
            em_factor,
            cos_t,
            sin_t,
            omega,
            signal_idx,
            ring_idx,
            bg_rank,
            col_sum,
            row_sum,
            col_var,
            row_var,
            ring,
            out[n],
        )


@numba.njit(nogil=True, cache=True, parallel=True, error_model="numpy")
def _fit_spots_parallel(
    spots,
    var,
    use_var,
    em_factor,
    cos_t,
    sin_t,
    omega,
    signal_idx,
    ring_idx,
    bg_rank,
    out,
    n_threads,
):
    size = spots.shape[1]
    n_spots = spots.shape[0]
    per_thread = (n_spots + n_threads - 1) // n_threads
    for t in numba.prange(n_threads):
        col_sum = np.empty(size)
        row_sum = np.empty(size)
        col_var = np.empty(size)
        row_var = np.empty(size)
        ring = np.empty(ring_idx.shape[0])
        for n in range(t * per_thread, min(n_spots, (t + 1) * per_thread)):
            _fit_spot(
                spots[n],
                var[n] if use_var else var[0],
                use_var,
                em_factor,
                cos_t,
                sin_t,
                omega,
                signal_idx,
                ring_idx,
                bg_rank,
                col_sum,
                row_sum,
                col_var,
                row_var,
                ring,
                out[n],
            )


def fit_spots(
    spots: lib.FloatArray3D,
    *,
    em: bool = False,
    variance: lib.FloatArray3D | None = None,
    parallel: bool = True,
    progress: lib.Progress | Literal["console"] | None = None,
    abort_callback: Callable[[], bool] | None = None,
) -> lib.FloatArray2D | None:
    """Phasor localization of spots.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts, indexed ``[spot, y, x]``;
        ``box`` odd and at least 3.
    em : bool, optional
        Whether the camera multiplies the signal (EMCCD), which doubles
        the photon variance in the propagated precision. Default is False.
    variance : lib.FloatArray3D or None, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid
        out like ``spots``; it adds to the photon variance in the
        propagated precision. Default is None (no readout noise).
    parallel : bool, optional
        Whether to use all CPU cores. Default is True.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports one phase, "Fitting spots", over
        the spots. Default is None.
    abort_callback : callable or None, optional
        Polled between chunks of spots; returning True stops the fit and
        the function returns None. Default is None.

    Returns
    -------
    theta : lib.FloatArray2D or None
        ``(n_spots, len(COLUMNS))`` float64 estimates, columns as in
        ``COLUMNS``: offsets from the center pixel ``box // 2`` in
        pixels, photons, background per pixel, normalized phasor
        magnitudes, propagated variances. A spot whose magnitudes or
        variances cannot be computed (e.g. an empty box) has non-finite
        entries. None if ``abort_callback`` asked to stop.
    """
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    if spots.ndim != 3 or spots.shape[1] != spots.shape[2]:
        raise ValueError(
            "spots must be an (n_spots, box, box) array, got shape "
            f"{spots.shape}."
        )
    box = _check_box(spots.shape[1])
    use_var = variance is not None
    if use_var:
        var = np.ascontiguousarray(variance, dtype=np.float32)
        if var.shape != spots.shape:
            raise ValueError(
                f"variance has shape {var.shape}, spots {spots.shape}; "
                "they must match."
            )
    else:
        var = np.zeros((1, box, box), dtype=np.float32)
    cos_t, sin_t, signal_idx, ring_idx, bg_rank = _tables(box)
    omega = 2 * np.pi / box
    em_factor = 2.0 if em else 1.0
    if parallel:
        # one set of scratch buffers per thread (allocating them per spot
        # would cost more than the estimate)
        n_threads = numba.get_num_threads()

        def kernel(*args):
            _fit_spots_parallel(*args, n_threads)

    else:
        kernel = _fit_spots_serial
    n_spots = len(spots)
    theta = np.full((n_spots, _N_COLUMNS), np.nan)
    with lib.tracked_progress(progress, "Fitting spots", "spot") as progress:
        lib.start_phase(progress, "Fitting spots", n_spots, unit="spot")
        for start in range(0, n_spots, _CHUNK):
            if callable(abort_callback) and abort_callback():
                return None
            stop = min(n_spots, start + _CHUNK)
            kernel(
                spots[start:stop],
                var[start:stop] if use_var else var,
                use_var,
                em_factor,
                cos_t,
                sin_t,
                omega,
                signal_idx,
                ring_idx,
                bg_rank,
                theta[start:stop],
            )
            progress.set_value(stop)
        progress.set_value(n_spots)
    return theta
