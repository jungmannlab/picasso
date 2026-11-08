"""
picasso.fitting.residualfit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-emitter fitting of whole frames: find the peaks in what the current
fit leaves unexplained, fit them, and repeat - the 3D-DAOSTORM analysis of
Babcock, Sigal & Zhuang (2012), ported from its maintained reference
implementation, storm-analysis (v2.4.1, maximum likelihood), with three of
its PSF models (``PSF_MODELS``):

- ``"fixed"`` (storm-analysis ``2dfixed``): every emitter has the same,
  fixed PSF sigma;
- ``"spherical"`` (``2d``): one sigma per emitter, the same in x and y,
  fitted;
- ``"elliptical"`` (``3d``): a sigma in x and one in y per emitter, both
  fitted, from which an astigmatic calibration gives z after the fit.

A fitted sigma is kept within 0.5 to 5 times the nominal one, and the
region of a peak is 1.5 times as large as with a fixed sigma, as in
storm-analysis.

The fourth model, ``"spline"``, is the cubic spline PSF of Babcock &
Zhuang (2017) in the same analysis: an experimental PSF, a Picasso spline
calibration (see :func:`spline_psf`), sampled at the pixels, with photons,
x, y, background and, for a 3D calibration, z fitted per emitter. It is
implemented from the paper, not ported: storm-analysis's spline fitter
(``spliner``) is licensed for non-commercial use only. The paper's
algorithm is the 3D-DAOSTORM one above, differing only where it finds and
fits the peaks: the residual is filtered with the spline PSF at one or
more z values (``peak_search_z``), and a new peak starts at the z of the
plane where it is brightest. Picasso's choices where the paper is silent:
the region of a peak, and the matched filter, are set apart from the
calibration box (``roi`` of :func:`spline_psf`), at most the box less 2
pixels, so that the region stays within the spline wherever the peak
moves inside its pixel; z is kept within the calibrated range; the PSF
sums to one in focus, so its amplitude is in photons, as in Picasso's
single-emitter spline fit; and
the calibration's Gaussian-equivalent sigma (``effective_sigma``) takes
the place of the nominal sigma in pruning and in the neighborhoods.

Every frame is analyzed on its own, in cycles:

1. The background is a broad Gaussian low pass (``background_sigma``) of
   the frame minus the current fit.
2. Peaks are found in the residual - frame minus background minus fit -
   smoothed with the PSF and divided by its expected noise, i.e., in units
   of the signal-to-noise ratio: local maxima above ``threshold`` within
   ``peak_radius``. A pixel is returned at most once per frame.
3. The new peaks join the fit, with the nominal sigma. Each peak is
   fitted on its own, with every other peak held fixed in the model, by
   Levenberg-Marquardt steps on the Poisson deviance of its region (a disc
   of ``roi_size`` pixels). Peaks near a new peak or near a removed one
   are fitted again, until all have converged.
4. Of two peaks closer than 1 nominal PSF sigma, the dimmer one is
   removed, and so is every peak whose significance is not above
   ``threshold``.

The cycles stop at the first cycle that finds no new peak, or after
``n_cycles``; peaks that did not converge are dropped. (storm-analysis
cycles on until its threshold schedule has run down, at least five
cycles, but a cycle without new peaks changes nothing, so the results are
the same.) The model of a pixel is the sum of the photons of every peak
whose region covers it plus the mean of their backgrounds, so the
background is counted once, however many peaks overlap.

Picasso's port differs from storm-analysis in four ways. The PSF is the
Gaussian integrated over each pixel, with the photon count as its
amplitude, instead of a Gaussian sampled at the pixel centers with the
peak height as its amplitude; a fitted width is the sigma itself, where
storm-analysis fits ``1 / (2 sigma^2)``; the filters are separable
convolutions with wrap-around edges on the frame padded by mirroring,
where storm-analysis convolves by FFT; and every emitter gets a
Cramer-Rao bound from the Fisher matrix of itself and its neighbors,
which storm-analysis does not compute.

This module is documented for reference only and is not part of the
public API; :func:`picasso.localize.localize_multi_emitter` runs it.

References
----------
Babcock, H., Sigal, Y. M. & Zhuang, X. "A high-density 3D localization
algorithm for stochastic optical reconstruction microscopy." Optical
Nanoscopy 1, 6 (2012). https://doi.org/10.1186/2192-2853-1-6

storm-analysis (Zhuang Lab, Harvard University), v2.4.1:
``sa_library/fitting.py``, ``sa_library/multi_fit.c``,
``sa_library/dao_fit.c``, ``sa_library/ia_utilities.c`` and
``daostorm_3d/find_peaks.py``. https://github.com/ZhuangLab/storm-analysis
License (MIT): ``LICENSES/storm-analysis-LICENSE.txt``.

Huang, F., Schwartz, S. L., Byars, J. M. & Lidke, K. A. "Simultaneous
multiple-emitter fitting for single molecule super-resolution imaging."
Biomed. Opt. Express 2, 1377-1393 (2011).
https://doi.org/10.1364/BOE.2.001377 (the Fisher matrix correction for
close emitters, Eq. 11)

Babcock, H. P. & Zhuang, X. "Analyzing single molecule localization
microscopy data using cubic splines." Sci. Rep. 7, 552 (2017).
https://doi.org/10.1038/s41598-017-00622-w (the spline model)

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np

from picasso.fitting.gaussfit import (
    _estimator_terms,
    _lm_accept_or_reject,
    _lm_finalize,
    _lm_reject_step,
)
from picasso.fitting.splinefit import (
    _FASTMATH,
    _LAMBDA_INITIAL,
    _eval_spline_3d,
    _lm_solve_step,
    FIT_STATE_CONVERGED,
    FIT_STATE_NEG_CURVATURE_MLE,
    FIT_STATE_SINGULAR_HESSIAN,
)

__all__ = [
    "PSF_MODELS",
    "THRESHOLD",
    "N_CYCLES",
    "PEAK_RADIUS",
    "BACKGROUND_SIGMA",
    "SplinePSF",
    "spline_psf",
    "NO_SPLINE",
    "roi_size",
    "padding",
    "FrameResult",
    "analyze_frame",
    "find_peaks",
    "fit_free_sigma",
]

#: PSF models and their codes; a Gaussian model's code is the number of
#: sigmas it fits per emitter.
PSF_MODELS = {"fixed": 0, "spherical": 1, "elliptical": 2, "spline": 3}
_SPLINE = PSF_MODELS["spline"]
#: Default peak threshold, in units of the signal-to-noise ratio, and the
#: minimum significance a peak keeps. storm-analysis uses 6.
THRESHOLD = 15.0
#: Default number of find-fit-subtract cycles per frame.
N_CYCLES = 20
#: Default radius (camera pixels) within which a peak must be the maximum.
PEAK_RADIUS = 5.0
#: Default sigma (camera pixels) of the Gaussian that estimates the
#: background.
BACKGROUND_SIGMA = 8.0

# Peak status (ia_utilities_c.py, multi_fit.h).
_RUNNING = 0
_CONVERGED = 1
_ERROR = 2

# Levenberg-Marquardt damping, per peak (multi_fit.h).
_LAMBDA_START = 1.0
_LAMBDA_DOWN = 0.75
_LAMBDA_MAX = 1.0e20
_LAMBDA_MIN = 1.0e-3
_LAMBDA_UP = 4.0
# Relative change of a peak's error below which it has converged
# (MultiFitter.default_tol).
_TOLERANCE = 1.0e-6
# Most fitting sweeps over the running peaks per call (MultiFitter.doFit).
_MAX_FIT_ITERATIONS = 2000
# Peaks closer than this many PSF sigma to a new or a removed peak are
# fitted again (PeakFinderFitter.unconverged_dist).
_NEIGHBORHOOD = 5.0
# Floor of the variance the residual is divided by (PeakFinderGaussian).
_VARIANCE_FLOOR = 0.1
# Filter kernels reach this many standard deviations.
_KERNEL_TRUNCATE = 4.0
# The region of a peak is a disc of int(8 sigma) pixels (PeakFinderGaussian),
# 1.5 times as large where the sigma is fitted.
_ROI_SIZE_FACTOR = 8.0
_ROI_FREE_WIDTH_FACTOR = 1.5
# A fitted sigma stays within these multiples of the nominal one
# (find_peaks.py: sigma_range).
_SIGMA_RANGE = (0.5, 5.0)
# Emitters closer than this many PSF sigma are fitted jointly in the
# Cramer-Rao bound and counted in ``n_emitters``.
_NEIGHBOR_DISTANCE = 4.0
# Most neighbors (nearest first) in an emitter's joint Fisher matrix.
_MAX_NEIGHBORS = 8
# Pseudo-inverse of that matrix: eigenvalues at or below this times the
# largest are dropped (numpy.linalg.pinv's default), and the Jacobi
# rotations stop after this many sweeps, or once the squared off-diagonal
# norm is at most this times the squared Frobenius norm (as
# precision._sym_pinv_device).
_PINV_RCOND = 1.0e-15
_JACOBI_SWEEPS = 30
_JACOBI_TOL = 1.0e-30

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
# Relative energy of a matched-filter kernel its separable terms may leave
# out (see spline_psf).
_KERNEL_TOLERANCE = 1.0e-6


def _roi_size(sigma, n_widths):
    """``roi_size`` for a model fitting ``n_widths`` sigmas per peak."""
    size = int(_ROI_SIZE_FACTOR * sigma)
    if n_widths > 0:
        size = int(_ROI_FREE_WIDTH_FACTOR * size)
    return size


_roi_size_jit = numba.njit(nogil=True, cache=True)(_roi_size)


class SplinePSF(NamedTuple):
    """A cubic spline PSF for :func:`analyze_frame`, made by
    :func:`spline_psf`; :data:`NO_SPLINE` for the Gaussian models."""

    #: Tricubic coefficients ``(1, niz, niy, nix, 4, 4, 4)``, the
    #: single-channel layout of :mod:`picasso.fitting.splinefit`, indexed
    #: ``[channel, z cell, y cell, x cell, z power, y power, x power]``,
    #: of a PSF whose knots lie on the camera pixel grid, centered at
    #: ``nix / 2``, in units of photons (it sums to one in focus).
    coeff: np.ndarray
    #: Whether z is fitted (a 3D spline) or held at 0 (a 2D one).
    fit_z: bool
    #: Side length (camera pixels) of the disc a peak is fitted in.
    roi: int
    #: Separable terms of the matched filters: the kernel along x and the
    #: one along y of each term, and the search plane it belongs to.
    search_kx: np.ndarray
    search_ky: np.ndarray
    search_plane: np.ndarray
    #: The same for the squared kernels, which filter the variance.
    variance_kx: np.ndarray
    variance_ky: np.ndarray
    variance_plane: np.ndarray
    #: z (spline slices) of the search planes.
    search_z: np.ndarray


def _empty_spline() -> SplinePSF:
    kernels = np.zeros((0, 1))
    planes = np.zeros(0, dtype=np.int64)
    return SplinePSF(
        np.zeros((1, 1, 1, 1, 4, 4, 4)),
        False,
        0,
        kernels,
        kernels,
        planes,
        kernels,
        kernels,
        planes,
        np.zeros(1),
    )


#: The ``spline`` argument of :func:`analyze_frame` for the Gaussian
#: models.
NO_SPLINE = _empty_spline()


def _separable_terms(kernel: np.ndarray) -> tuple:
    """Kernels along x and along y whose outer products sum to ``kernel``
    (rows y, columns x) up to ``_KERNEL_TOLERANCE`` of its energy, by
    singular value decomposition."""
    u, s, vt = np.linalg.svd(kernel)
    energy = np.cumsum(s * s)
    n = int(np.searchsorted(energy, (1.0 - _KERNEL_TOLERANCE) * energy[-1]))
    n = min(n + 1, len(s))
    return vt[:n].copy(), (u[:, :n] * s[:n]).T.copy()


def spline_psf(
    coefficients: np.ndarray,
    fit_z: bool,
    search_z: tuple | list | np.ndarray = (0.0,),
    roi: int | None = None,
) -> SplinePSF:
    """A cubic spline PSF for :func:`analyze_frame`, with its matched
    filters.

    The matched filter of a search plane is the PSF at its z, sampled at
    the pixels of the region of a peak centered on a pixel and normalized
    to sum to one, as a sum of separable terms; the variance is filtered
    with its square.

    Parameters
    ----------
    coefficients : np.ndarray
        Tricubic coefficients ``(1, niz, niy, nix, 4, 4, 4)`` with knots
        on the camera pixel grid, see :attr:`SplinePSF.coeff`; a 2D
        spline has ``niz = 1`` and only z power 0.
    fit_z : bool
        Whether z is fitted (3D) or held at 0 (2D).
    search_z : sequence of float, optional
        z (spline slices) of the matched-filter planes. Default is (0,).
    roi : int or None, optional
        Side length (camera pixels) of the disc a peak is fitted in, from
        3 to ``nix - 1``, the calibration box less 2, so that the region
        stays within the spline wherever the peak moves inside its
        pixel. None (the default) takes ``nix - 1``.

    Returns
    -------
    spline : SplinePSF

    Raises
    ------
    ValueError
        If ``roi`` is out of range, or the PSF at a search plane does not
        sum to a positive value.
    """
    coeff = np.ascontiguousarray(coefficients, dtype=np.float64)
    nix = coeff.shape[3]
    if roi is None:
        roi = nix - 1
    if not 3 <= roi <= nix - 1:
        raise ValueError(
            f"The region of a peak must be 3 to {nix - 1} pixels, the "
            f"calibration box ({nix + 1}) less 2; got {roi}."
        )
    center = 0.5 * nix
    # odd kernels centered on a pixel, within the region
    radius = int(roi) // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    search_z = np.asarray(search_z, dtype=np.float64).ravel()
    if not fit_z:
        search_z = np.zeros(1)
    terms = ([], [], [], [], [], [])
    for plane, z in enumerate(search_z):
        kernel = np.array(
            [
                [
                    _eval_spline_3d(coeff, 0, center + dx, center + dy, z)[0]
                    for dx in offsets
                ]
                for dy in offsets
            ]
        )
        total = kernel.sum()
        if not total > 0.0:
            raise ValueError(
                f"The spline PSF at z = {z} slices does not sum to a "
                "positive value."
            )
        kernel /= total
        for k, values in enumerate((kernel, kernel * kernel)):
            kx, ky = _separable_terms(values)
            terms[3 * k].append(kx)
            terms[3 * k + 1].append(ky)
            terms[3 * k + 2].append(np.full(len(kx), plane, dtype=np.int64))
    return SplinePSF(
        coeff,
        bool(fit_z),
        int(roi),
        *(np.ascontiguousarray(np.concatenate(t)) for t in terms),
        np.ascontiguousarray(search_z),
    )


def roi_size(
    sigma: float,
    psf_model: str = "fixed",
    spline: SplinePSF | None = None,
) -> int:
    """Side length (camera pixels) of the disc a peak is fitted in.

    Parameters
    ----------
    sigma : float
        Nominal PSF standard deviation in camera pixels.
    psf_model : str, optional
        One of ``PSF_MODELS``. Default is "fixed".
    spline : SplinePSF or None, optional
        The PSF of the "spline" model. Default is None.

    Returns
    -------
    roi_size : int
        ``int(8 * sigma)`` for a fixed sigma, 1.5 times that (rounded
        down) where the sigma is fitted, as in storm-analysis; the
        spline's ``roi`` (see :func:`spline_psf`).
    """
    if psf_model == "spline":
        return spline.roi
    return _roi_size(sigma, PSF_MODELS[psf_model])


def padding(
    sigma: float,
    psf_model: str = "fixed",
    spline: SplinePSF | None = None,
) -> int:
    """Margin (camera pixels) by which a frame is padded, by mirroring.

    Parameters
    ----------
    sigma : float
        Nominal PSF standard deviation in camera pixels.
    psf_model : str, optional
        One of ``PSF_MODELS``. Default is "fixed".
    spline : SplinePSF or None, optional
        The PSF of the "spline" model. Default is None.

    Returns
    -------
    padding : int
        ``int(roi_size / 2 + 2)``: the region of a peak on the edge of the
        frame stays within the padded frame.
    """
    return int(roi_size(sigma, psf_model, spline) / 2 + 2)


# ----------------------------------------------------------------------
# Filters (PeakFinderGaussian)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _gaussian_kernel(sigma: float, size: int, square: bool) -> np.ndarray:
    """Normalized 1D Gaussian kernel, or its square (for the variance of a
    smoothed image), reaching ``_KERNEL_TRUNCATE`` standard deviations but
    no further than half the image."""
    radius = int(math.ceil(_KERNEL_TRUNCATE * sigma))
    radius = max(0, min(radius, (size - 1) // 2))
    kernel = np.empty(2 * radius + 1)
    total = 0.0
    for i in range(-radius, radius + 1):
        value = math.exp(-0.5 * i * i / (sigma * sigma))
        kernel[i + radius] = value
        total += value
    for i in range(2 * radius + 1):
        kernel[i] /= total
        if square:
            kernel[i] *= kernel[i]
    return kernel


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _convolve(image, kernel, out, tmp):
    """Separable convolution with wrap-around edges: the circular
    convolution storm-analysis runs by FFT on its padded frames."""
    _convolve_xy(image, kernel, kernel, out, tmp)


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _convolve_xy(image, kernel, kernel_y, out, tmp):
    """:func:`_convolve` with ``kernel`` along x and ``kernel_y`` along y,
    both of the same odd length."""
    height, width = image.shape
    radius = len(kernel) // 2
    # The kernel taps are the outer loops and the pixels of a row the
    # inner ones, which run over contiguous memory; every pixel still sums
    # its taps in kernel order. Along x, each row is first extended by the
    # columns it wraps around to.
    row = np.empty(width + 2 * radius)
    for y in range(height):
        for x in range(width + 2 * radius):
            xx = x - radius
            if xx < 0:
                xx += width
            elif xx >= width:
                xx -= width
            row[x] = image[y, xx]
        for x in range(width):
            tmp[y, x] = 0.0
        for k in range(2 * radius + 1):
            weight = kernel[k]
            for x in range(width):
                tmp[y, x] += weight * row[x + k]
    for y in range(height):
        for x in range(width):
            out[y, x] = 0.0
        for k in range(2 * radius + 1):
            yy = y + k - radius
            if yy < 0:
                yy += height
            elif yy >= height:
                yy -= height
            weight = kernel_y[k]
            for x in range(width):
                out[y, x] += weight * tmp[yy, x]


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _filter_plane(image, kx, ky, planes, plane, out, conv, tmp):
    """``image`` filtered with the separable terms of one search plane
    (see :class:`SplinePSF`), summed into ``out``."""
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            out[y, x] = 0.0
    for t in range(len(planes)):
        if planes[t] != plane:
            continue
        _convolve_xy(image, kx[t], ky[t], conv, tmp)
        for y in range(height):
            for x in range(width):
                out[y, x] += conv[y, x]


# ----------------------------------------------------------------------
# Local maxima (ia_utilities.c: findLocalMaxima, isLocalMaxima)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _is_local_maximum(image, value, radius, cx, cy):
    """Whether ``value`` at ``(cx, cy)`` is the maximum within ``radius``.

    Of two equal pixels, the one with the larger x and y wins, so that
    exactly one of them is a maximum."""
    height, width = image.shape
    rr = int(radius * radius)
    sy = max(cy - int(radius), 0)
    ey = min(cy + int(radius), height - 1)
    reach = int(math.ceil(radius))
    sx = max(cx - reach, 0)
    ex = min(cx + reach, width - 1)
    for y in range(sy, ey + 1):
        dy = (y - cy) * (y - cy)
        for x in range(sx, ex + 1):
            dx = (x - cx) * (x - cx)
            if dx + dy <= rr:
                if y <= cy and x <= cx:
                    if image[y, x] > value:
                        return False
                else:
                    if image[y, x] >= value:
                        return False
    return True


@numba.njit(nogil=True, cache=True)
def _find_maxima(image, threshold, radius, margin, taken, out_x, out_y):
    """Local maxima above ``threshold``, at least ``margin`` from the
    edges, that were not returned before (``taken``). Returns the number
    found, written into ``out_x``/``out_y``."""
    height, width = image.shape
    n = 0
    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            value = image[y, x]
            if value > threshold and taken[y, x] < 1:
                if _is_local_maximum(image, value, radius, x, y):
                    taken[y, x] += 1
                    if n < len(out_x):
                        out_x[n] = x
                        out_y[n] = y
                        n += 1
    return n


# ----------------------------------------------------------------------
# Neighbor search on a grid of cells
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _build_grid(x, y, n, cell):
    """Bucket ``n`` points into square cells of side ``cell``; returns
    (cell of each point, start of each cell in ``order``, ``order``,
    cells along x, cells along y, x origin, y origin)."""
    x0 = 0.0
    y0 = 0.0
    nx = 1
    ny = 1
    if n > 0:
        x0 = x[0]
        y0 = y[0]
        x1 = x[0]
        y1 = y[0]
        for i in range(n):
            x0 = min(x0, x[i])
            y0 = min(y0, y[i])
            x1 = max(x1, x[i])
            y1 = max(y1, y[i])
        nx = int((x1 - x0) / cell) + 1
        ny = int((y1 - y0) / cell) + 1
    cells = np.empty(n, dtype=np.int64)
    counts = np.zeros(nx * ny + 1, dtype=np.int64)
    for i in range(n):
        cx = int((x[i] - x0) / cell)
        cy = int((y[i] - y0) / cell)
        cells[i] = cy * nx + cx
        counts[cells[i] + 1] += 1
    for c in range(nx * ny):
        counts[c + 1] += counts[c]
    order = np.empty(n, dtype=np.int64)
    fill = counts[:-1].copy()
    for i in range(n):
        order[fill[cells[i]]] = i
        fill[cells[i]] += 1
    return cells, counts, order, nx, ny, x0, y0


# fastmath given, as _prune's, not that of _local_crlb (see
# _pinv_symmetric)
@numba.njit(nogil=True, cache=True, fastmath=False)
def _neighbors(qx, qy, radius, x, y, counts, order, nx, ny, x0, y0, cell, out):
    """Indices of the points within ``radius`` (<= ``cell``) of
    ``(qx, qy)``, the query point included; returns their number."""
    cx = int(math.floor((qx - x0) / cell))
    cy = int(math.floor((qy - y0) / cell))
    rr = radius * radius
    n = 0
    for gy in range(max(cy - 1, 0), min(cy + 2, ny)):
        for gx in range(max(cx - 1, 0), min(cx + 2, nx)):
            c = gy * nx + gx
            for k in range(counts[c], counts[c + 1]):
                i = order[k]
                dx = x[i] - qx
                dy = y[i] - qy
                if dx * dx + dy * dy <= rr:
                    if n < len(out):
                        out[n] = i
                        n += 1
    return n


# ----------------------------------------------------------------------
# The pixel-integrated Gaussian of a peak over its region
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _profile(center, origin, size, sigma, e, de, ds):
    """Gaussian integrated over the pixels ``origin .. origin + size - 1``
    (pixel ``c`` spans ``[c - 0.5, c + 0.5]``), and its derivatives with
    respect to ``center`` and to ``sigma``."""
    inv = _INV_SQRT2 / sigma
    norm = _INV_SQRT2PI / sigma
    norm_s = norm / sigma
    inv2 = 0.5 / (sigma * sigma)
    for i in range(size):
        a_plus = origin + i - center + 0.5
        a_minus = origin + i - center - 0.5
        g_plus = math.exp(-a_plus * a_plus * inv2)
        g_minus = math.exp(-a_minus * a_minus * inv2)
        e[i] = 0.5 * (math.erf(a_plus * inv) - math.erf(a_minus * inv))
        de[i] = -norm * (g_plus - g_minus)
        ds[i] = -norm_s * (a_plus * g_plus - a_minus * g_minus)


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _peak_shape(
    x, y, xi, yi, sx, sy, roi_dx, roi_dy, ex, dex, dsx, ey, dey, dsy, psf
):
    """Fill ``psf`` with the unit-integral PSF of a peak at ``(x, y)`` with
    sigmas ``sx`` and ``sy`` over the disc of its region, which starts at
    ``(xi, yi)``."""
    size = len(ex)
    _profile(x, xi, size, sx, ex, dex, dsx)
    _profile(y, yi, size, sy, ey, dey, dsy)
    for j in range(len(roi_dx)):
        psf[j] = ex[roi_dx[j]] * ey[roi_dy[j]]


# ----------------------------------------------------------------------
# The cubic spline PSF of a peak over its region (Babcock & Zhuang 2017,
# Eqs. 7-11)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _spline_peak_shape(
    coeff, x, y, z, xi, yi, roi_dx, roi_dy, psf, gx, gy, gz
):
    """Fill ``psf`` with the spline PSF of a peak at ``(x, y)`` and ``z``
    (spline slices), sampled at the pixels of the disc of its region,
    which starts at ``(xi, yi)``, and ``gx``, ``gy``, ``gz`` with its
    derivatives with respect to ``x``, ``y`` and ``z``. The spline is
    evaluated as in the single-emitter spline fit
    (:func:`picasso.fitting.splinefit._eval_spline_3d`)."""
    center = 0.5 * coeff.shape[3]
    # spline coordinates of the region's first pixel
    ux = xi - x + center
    uy = yi - y + center
    for j in range(len(roi_dx)):
        value, dvx, dvy, dvz = _eval_spline_3d(
            coeff, 0, ux + roi_dx[j], uy + roi_dy[j], z
        )
        psf[j] = value
        # the spline coordinate falls as the peak moves up
        gx[j] = -dvx
        gy[j] = -dvy
        gz[j] = dvz


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _spline_jacobian_hessian(
    photons,
    xi,
    yi,
    roi_dx,
    roi_dy,
    psf,
    gx,
    gy,
    gz,
    x_data,
    f_data,
    bg_data,
    bg_counts,
    jac,
    hess,
):
    """:func:`_jacobian_hessian` of the spline PSF, with respect to
    ``[photons, x, y, bg]`` followed by z if ``jac`` has room for it."""
    n_par = len(jac)
    for p in range(n_par):
        jac[p] = 0.0
        for q in range(n_par):
            hess[p, q] = 0.0
    jt = np.empty(n_par)
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        fi = f_data[row, col] + bg_data[row, col] / bg_counts[row, col]
        xv = x_data[row, col]
        jt[0] = psf[j]
        jt[1] = photons * gx[j]
        jt[2] = photons * gy[j]
        jt[3] = 1.0
        if n_par == 5:
            jt[4] = photons * gz[j]
        t1 = 2.0 * (1.0 - xv / fi)
        t2 = 2.0 * xv / (fi * fi)
        for p in range(n_par):
            jac[p] += t1 * jt[p]
            for q in range(p, n_par):
                hess[p, q] += t2 * jt[p] * jt[q]
    for p in range(n_par):
        for q in range(p):
            hess[p, q] = hess[q, p]


# ----------------------------------------------------------------------
# The fit image (multi_fit.c: mFitAddPeak, mFitSubtractPeak, mFitCalcErr)
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _add_peak(
    photons,
    bg,
    xi,
    yi,
    psf,
    roi_dx,
    roi_dy,
    variance,
    use_var,
    f_data,
    bg_data,
    bg_counts,
    sign,
):
    """Add (``sign`` 1) or subtract (``sign`` -1) a peak to or from the
    fit image."""
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        var = variance[row, col] if use_var else 0.0
        f_data[row, col] += sign * photons * psf[j]
        bg_data[row, col] += sign * (bg + var)
        bg_counts[row, col] += sign


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _peak_error(xi, yi, roi_dx, roi_dy, x_data, f_data, bg_data, bg_counts):
    """Poisson deviance of the current fit over a peak's region; returns
    (ok, error), not ok where the model is not positive."""
    error = 0.0
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        fi = f_data[row, col] + bg_data[row, col] / bg_counts[row, col]
        if fi <= 0.0:
            return False, np.inf
        xv = x_data[row, col]
        error += 2.0 * ((fi - xv) - xv * math.log(fi / xv))
    return True, error


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _jacobian_hessian(
    photons,
    xi,
    yi,
    roi_dx,
    roi_dy,
    ex,
    dex,
    dsx,
    ey,
    dey,
    dsy,
    x_data,
    f_data,
    bg_data,
    bg_counts,
    jac,
    hess,
):
    """Gradient ``jac`` and Hessian ``hess`` of the deviance of a peak's
    region with respect to ``[photons, x, y, bg]``, followed by its sigma
    (one free sigma) or its sigmas in x and y (two), as many as ``jac``
    has room for (dao_fit.c: daoCalcJH2DFixed, daoCalcJH2D, daoCalcJH3D).
    The background enters with derivative 1, as in storm-analysis,
    although a pixel averages the backgrounds of the peaks covering
    it."""
    n_par = len(jac)
    for p in range(n_par):
        jac[p] = 0.0
        for q in range(n_par):
            hess[p, q] = 0.0
    jt = np.empty(n_par)
    for j in range(len(roi_dx)):
        row = yi + roi_dy[j]
        col = xi + roi_dx[j]
        fi = f_data[row, col] + bg_data[row, col] / bg_counts[row, col]
        xv = x_data[row, col]
        a = roi_dx[j]
        b = roi_dy[j]
        jt[0] = ex[a] * ey[b]
        jt[1] = photons * dex[a] * ey[b]
        jt[2] = photons * ex[a] * dey[b]
        jt[3] = 1.0
        if n_par == 5:
            jt[4] = photons * (dsx[a] * ey[b] + ex[a] * dsy[b])
        elif n_par == 6:
            jt[4] = photons * dsx[a] * ey[b]
            jt[5] = photons * ex[a] * dsy[b]
        t1 = 2.0 * (1.0 - xv / fi)
        t2 = 2.0 * xv / (fi * fi)
        for p in range(n_par):
            jac[p] += t1 * jt[p]
            for q in range(p, n_par):
                hess[p, q] += t2 * jt[p] * jt[q]
    for p in range(n_par):
        for q in range(p):
            hess[p, q] = hess[q, p]


# fastmath given, as that of its caller _iterate (see _pinv_symmetric)
@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _solve_cholesky(a, b, n):
    """Solve ``a x = b`` for symmetric positive definite ``a`` in place
    (the solution is left in ``b``); False if ``a`` is not positive
    definite or the solution is not finite (LAPACK dposv)."""
    for j in range(n):
        s = a[j, j]
        for k in range(j):
            s -= a[j, k] * a[j, k]
        if not s > 0.0:
            return False
        a[j, j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = a[i, j]
            for k in range(j):
                s -= a[i, k] * a[j, k]
            a[i, j] = s / a[j, j]
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= a[i, k] * b[k]
        b[i] = s / a[i, i]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= a[k, i] * b[k]
        b[i] = s / a[i, i]
    for i in range(n):
        if not np.isfinite(b[i]):
            return False
    return True


# fastmath given: otherwise numba builds a function with that of the
# function that compiles it first (none for a direct call) and keeps that
# one build per signature, so its rounding would depend on what ran first
@numba.njit(nogil=True, cache=True, fastmath=False)
def _pinv_symmetric(a):
    """Pseudo-inverse of the symmetric ``a``, as ``numpy.linalg.pinv``
    gives it, without LAPACK: called by the threads of a frame pool at
    once, OpenBLAS (built for fewer threads) can crash. A positive
    definite ``a`` whose condition number is below ``1 / _PINV_RCOND``,
    which ``tr(a) tr(a^-1)`` bounds, so that pinv drops nothing, is
    inverted by Cholesky (LAPACK dpotri); any other is diagonalized by
    cyclic Jacobi rotations, dropping the eigenvalues at or below
    ``_PINV_RCOND`` times the largest in magnitude. NaN where the
    rotations do not converge."""
    n = a.shape[0]
    low = a.copy()
    definite = True
    for j in range(n):
        s = low[j, j]
        for k in range(j):
            s -= low[j, k] * low[j, k]
        if not s > 0.0:
            definite = False
            break
        low[j, j] = math.sqrt(s)
        for i in range(j + 1, n):
            s = low[i, j]
            for k in range(j):
                s -= low[i, k] * low[j, k]
            low[i, j] = s / low[j, j]
    if definite:
        # L^-1, lower triangular, then a^-1 = L^-T L^-1
        inv_low = np.zeros((n, n))
        for c in range(n):
            for i in range(c, n):
                s = 1.0 if i == c else 0.0
                for k in range(c, i):
                    s -= low[i, k] * inv_low[k, c]
                inv_low[i, c] = s / low[i, i]
        out = np.empty((n, n))
        trace = 0.0
        trace_inv = 0.0
        for p in range(n):
            for q in range(p, n):
                s = 0.0
                for k in range(q, n):
                    s += inv_low[k, p] * inv_low[k, q]
                out[p, q] = s
                out[q, p] = s
            trace += a[p, p]
            trace_inv += out[p, p]
        if trace * trace_inv < 1.0 / _PINV_RCOND:
            return out
    d = a.copy()
    v = np.eye(n)
    converged = False
    for _ in range(_JACOBI_SWEEPS):
        off = 0.0
        fro = 0.0
        for p in range(n):
            fro += d[p, p] * d[p, p]
            for q in range(p + 1, n):
                off += d[p, q] * d[p, q]
        fro += 2.0 * off
        if off <= _JACOBI_TOL * fro:
            converged = True
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                dpq = d[p, q]
                if dpq == 0.0:
                    continue
                # the smaller root of t^2 + 2 theta t - 1 = 0: a rotation
                # by less than 45 degrees
                theta = (d[q, q] - d[p, p]) / (2.0 * dpq)
                sign = 1.0 if theta >= 0.0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    dkp = d[k, p]
                    dkq = d[k, q]
                    d[k, p] = c * dkp - s * dkq
                    d[k, q] = s * dkp + c * dkq
                for k in range(n):
                    dpk = d[p, k]
                    dqk = d[q, k]
                    d[p, k] = c * dpk - s * dqk
                    d[q, k] = s * dpk + c * dqk
                for k in range(n):
                    vkp = v[k, p]
                    vkq = v[k, q]
                    v[k, p] = c * vkp - s * vkq
                    v[k, q] = s * vkp + c * vkq
    out = np.full((n, n), np.nan)
    if not converged:
        return out
    biggest = 0.0
    for k in range(n):
        biggest = max(biggest, abs(d[k, k]))
    cutoff = _PINV_RCOND * biggest
    for p in range(n):
        for q in range(p, n):
            s = 0.0
            for k in range(n):
                if abs(d[k, k]) > cutoff:
                    s += v[p, k] * v[q, k] / d[k, k]
            out[p, q] = s
            out[q, p] = s
    return out


# ----------------------------------------------------------------------
# One frame
# ----------------------------------------------------------------------


class FrameResult(NamedTuple):
    """The emitters found in one frame by :func:`analyze_frame`, in the
    pixel coordinates of the frame (pixel centers on integers)."""

    x: np.ndarray
    y: np.ndarray
    photons: np.ndarray
    bg: np.ndarray
    #: PSF sigma in x and in y (camera pixels), fitted or fixed; NaN for
    #: the spline.
    sx: np.ndarray
    sy: np.ndarray
    #: z (spline slices) of a 3D spline; NaN otherwise.
    z: np.ndarray
    significance: np.ndarray
    #: Poisson deviance of the fit over the emitter's region.
    error: np.ndarray
    iterations: np.ndarray
    #: Number of pixels in the region the deviance is summed over.
    n_pixels: np.ndarray
    #: Number of fitted parameters per emitter.
    n_parameters: int
    #: Cramer-Rao variances of photons, x, y, bg, sx, sy and z, from the
    #: joint Fisher matrix of the emitter and its neighbors; NaN where not
    #: fitted, the same in sx and sy for one fitted sigma.
    crlb: np.ndarray
    #: Emitters in that matrix: the emitter itself and its neighbors
    #: within 4 nominal PSF sigma, at most the 8 nearest.
    n_emitters: np.ndarray


@numba.njit(nogil=True, cache=True)
def _grow(array, capacity):
    """1D ``array`` copied into one ``capacity`` long."""
    out = np.zeros(capacity, dtype=array.dtype)
    out[: len(array)] = array
    return out


@numba.njit(nogil=True, cache=True)
def _grow2(array, capacity):
    """2D ``array`` copied into one with ``capacity`` rows."""
    out = np.zeros((capacity, array.shape[1]), dtype=array.dtype)
    out[: len(array)] = array
    return out


@numba.njit(nogil=True, cache=True)
def _compact(
    keep,
    n,
    p_n,
    p_x,
    p_y,
    p_bg,
    p_sx,
    p_sy,
    p_z,
    p_xi,
    p_yi,
    p_status,
    p_lambda,
    p_error,
    p_iter,
    p_added,
    p_psf,
):
    """Remove the peaks that ``keep`` excludes, preserving the order;
    returns the new count."""
    m = 0
    for i in range(n):
        if keep[i]:
            if i != m:
                p_n[m] = p_n[i]
                p_x[m] = p_x[i]
                p_y[m] = p_y[i]
                p_bg[m] = p_bg[i]
                p_sx[m] = p_sx[i]
                p_sy[m] = p_sy[i]
                p_z[m] = p_z[i]
                p_xi[m] = p_xi[i]
                p_yi[m] = p_yi[i]
                p_status[m] = p_status[i]
                p_lambda[m] = p_lambda[i]
                p_error[m] = p_error[i]
                p_iter[m] = p_iter[i]
                p_added[m] = p_added[i]
                for j in range(p_psf.shape[1]):
                    p_psf[m, j] = p_psf[i, j]
            m += 1
    return m


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _iterate(
    n,
    p_n,
    p_x,
    p_y,
    p_bg,
    p_sx,
    p_sy,
    p_z,
    p_xi,
    p_yi,
    p_status,
    p_lambda,
    p_error,
    p_iter,
    p_added,
    p_psf,
    n_widths,
    sigma_min,
    sigma_max,
    roi,
    xoff,
    roi_dx,
    roi_dy,
    x_data,
    variance,
    use_var,
    f_data,
    bg_data,
    bg_counts,
    tolerance,
    ex,
    dex,
    dsx,
    ey,
    dey,
    dsy,
    psf,
    spline,
    coeff,
    n_z,
    gx,
    gy,
    gz,
):
    """One Levenberg-Marquardt step of every running peak, the others held
    fixed, then recenter the regions (multi_fit.c: mFitIterateLM,
    mFitRecenterPeaks). ``n_widths`` sigmas per peak are fitted (0, 1 or
    2, see ``PSF_MODELS``), clamped to ``[sigma_min, sigma_max]``
    (dao_fit.c: daoCheck2D, daoCheck3D). With a ``spline`` PSF
    (``coeff``), z is fitted if ``n_z`` is 1, within the spline."""
    height, width = x_data.shape
    n_par = 4 + n_widths + n_z
    z_max = float(coeff.shape[1])
    jac = np.empty(n_par)
    hess = np.empty((n_par, n_par))
    w_jac = np.empty(n_par)
    w_hess = np.empty((n_par, n_par))
    for i in range(n):
        if p_status[i] != _RUNNING:
            continue
        lam = p_lambda[i]
        ok, start_error = _peak_error(
            p_xi[i],
            p_yi[i],
            roi_dx,
            roi_dy,
            x_data,
            f_data,
            bg_data,
            bg_counts,
        )
        if spline:
            _spline_peak_shape(
                coeff,
                p_x[i],
                p_y[i],
                p_z[i],
                p_xi[i],
                p_yi[i],
                roi_dx,
                roi_dy,
                psf,
                gx,
                gy,
                gz,
            )
            _spline_jacobian_hessian(
                p_n[i],
                p_xi[i],
                p_yi[i],
                roi_dx,
                roi_dy,
                psf,
                gx,
                gy,
                gz,
                x_data,
                f_data,
                bg_data,
                bg_counts,
                jac,
                hess,
            )
        else:
            _profile(p_x[i], p_xi[i], roi, p_sx[i], ex, dex, dsx)
            _profile(p_y[i], p_yi[i], roi, p_sy[i], ey, dey, dsy)
            _jacobian_hessian(
                p_n[i],
                p_xi[i],
                p_yi[i],
                roi_dx,
                roi_dy,
                ex,
                dex,
                dsx,
                ey,
                dey,
                dsy,
                x_data,
                f_data,
                bg_data,
                bg_counts,
                jac,
                hess,
            )
        # the peak leaves the fit image while it is improved
        _add_peak(
            p_n[i],
            p_bg[i],
            p_xi[i],
            p_yi[i],
            p_psf[i],
            roi_dx,
            roi_dy,
            variance,
            use_var,
            f_data,
            bg_data,
            bg_counts,
            -1,
        )
        added = False
        status = _RUNNING
        w_n = p_n[i]
        w_x = p_x[i]
        w_y = p_y[i]
        w_bg = p_bg[i]
        w_sx = p_sx[i]
        w_sy = p_sy[i]
        w_z = p_z[i]
        w_xi = p_xi[i]
        w_yi = p_yi[i]
        error = start_error
        iterations = p_iter[i]
        while True:
            if lam > _LAMBDA_MAX:
                status = _ERROR
                break
            iterations += 1
            status = _RUNNING
            for p in range(n_par):
                w_jac[p] = jac[p]
                for q in range(n_par):
                    w_hess[p, q] = hess[p, q]
                w_hess[p, p] *= 1.0 + lam
            if not _solve_cholesky(w_hess, w_jac, n_par):
                status = _ERROR
                lam *= _LAMBDA_UP
                continue
            w_n = p_n[i] - w_jac[0]
            w_x = p_x[i] - w_jac[1]
            w_y = p_y[i] - w_jac[2]
            w_bg = p_bg[i] - w_jac[3]
            if n_widths == 1:
                w_sx = p_sx[i] - w_jac[4]
                # a range clamp rather than an error (daoCheck2D)
                w_sx = min(max(w_sx, sigma_min), sigma_max)
                w_sy = w_sx
            elif n_widths == 2:
                w_sx = min(max(p_sx[i] - w_jac[4], sigma_min), sigma_max)
                w_sy = min(max(p_sy[i] - w_jac[5], sigma_min), sigma_max)
            if n_z == 1:
                # within the calibrated range
                w_z = min(max(p_z[i] - w_jac[4], 0.0), z_max)
            w_xi = p_xi[i]
            w_yi = p_yi[i]
            # mFitUpdate: move the region with the peak
            dx = w_x - xoff - w_xi
            if dx < -0.5 or dx > 1.5:
                w_xi = int(math.floor(w_x - xoff))
            dy = w_y - xoff - w_yi
            if dy < -0.5 or dy > 1.5:
                w_yi = int(math.floor(w_y - xoff))
            # mFitCheck: within the image and a non-negative amplitude
            if (
                w_xi < 0
                or w_xi >= width - roi
                or w_yi < 0
                or w_yi >= height - roi
                or w_n < 0.0
                or not np.isfinite(w_x + w_y + w_n + w_bg + w_sx + w_sy)
                or not np.isfinite(w_z)
            ):
                lam *= _LAMBDA_UP
                status = _ERROR
                continue
            if spline:
                _spline_peak_shape(
                    coeff,
                    w_x,
                    w_y,
                    w_z,
                    w_xi,
                    w_yi,
                    roi_dx,
                    roi_dy,
                    psf,
                    gx,
                    gy,
                    gz,
                )
            else:
                _peak_shape(
                    w_x,
                    w_y,
                    w_xi,
                    w_yi,
                    w_sx,
                    w_sy,
                    roi_dx,
                    roi_dy,
                    ex,
                    dex,
                    dsx,
                    ey,
                    dey,
                    dsy,
                    psf,
                )
            _add_peak(
                w_n,
                w_bg,
                w_xi,
                w_yi,
                psf,
                roi_dx,
                roi_dy,
                variance,
                use_var,
                f_data,
                bg_data,
                bg_counts,
                1,
            )
            ok, error = _peak_error(
                w_xi,
                w_yi,
                roi_dx,
                roi_dy,
                x_data,
                f_data,
                bg_data,
                bg_counts,
            )
            if not ok:
                _add_peak(
                    w_n,
                    w_bg,
                    w_xi,
                    w_yi,
                    psf,
                    roi_dx,
                    roi_dy,
                    variance,
                    use_var,
                    f_data,
                    bg_data,
                    bg_counts,
                    -1,
                )
                lam *= _LAMBDA_UP
                status = _ERROR
                continue
            if error > start_error:
                if (error - start_error) / start_error < tolerance:
                    status = _CONVERGED
                    added = True
                    break
                _add_peak(
                    w_n,
                    w_bg,
                    w_xi,
                    w_yi,
                    psf,
                    roi_dx,
                    roi_dy,
                    variance,
                    use_var,
                    f_data,
                    bg_data,
                    bg_counts,
                    -1,
                )
                lam *= _LAMBDA_UP
            else:
                if (start_error - error) / start_error < tolerance:
                    status = _CONVERGED
                    added = True
                    break
                if lam > _LAMBDA_MIN:
                    lam *= _LAMBDA_DOWN
                added = True
                break
        p_iter[i] = iterations
        p_lambda[i] = lam
        p_status[i] = status
        if added:
            p_n[i] = w_n
            p_x[i] = w_x
            p_y[i] = w_y
            p_bg[i] = w_bg
            p_sx[i] = w_sx
            p_sy[i] = w_sy
            p_z[i] = w_z
            p_xi[i] = w_xi
            p_yi[i] = w_yi
            p_error[i] = error
            for j in range(len(roi_dx)):
                p_psf[i, j] = psf[j]
            p_added[i] = True
        else:
            p_added[i] = False
            p_status[i] = _ERROR
    # mFitRecenterPeaks
    for i in range(n):
        if p_status[i] != _RUNNING:
            continue
        dx = p_x[i] - xoff - p_xi[i]
        dy = p_y[i] - xoff - p_yi[i]
        if dx < -0.25 or dx > 1.25 or dy < -0.25 or dy > 1.25:
            _add_peak(
                p_n[i],
                p_bg[i],
                p_xi[i],
                p_yi[i],
                p_psf[i],
                roi_dx,
                roi_dy,
                variance,
                use_var,
                f_data,
                bg_data,
                bg_counts,
                -1,
            )
            p_xi[i] = int(math.floor(p_x[i] - xoff))
            p_yi[i] = int(math.floor(p_y[i] - xoff))
            if (
                p_xi[i] < 0
                or p_xi[i] >= width - roi
                or p_yi[i] < 0
                or p_yi[i] >= height - roi
                or p_n[i] < 0.0
            ):
                p_status[i] = _ERROR
                p_added[i] = False
            else:
                if spline:
                    _spline_peak_shape(
                        coeff,
                        p_x[i],
                        p_y[i],
                        p_z[i],
                        p_xi[i],
                        p_yi[i],
                        roi_dx,
                        roi_dy,
                        psf,
                        gx,
                        gy,
                        gz,
                    )
                else:
                    _peak_shape(
                        p_x[i],
                        p_y[i],
                        p_xi[i],
                        p_yi[i],
                        p_sx[i],
                        p_sy[i],
                        roi_dx,
                        roi_dy,
                        ex,
                        dex,
                        dsx,
                        ey,
                        dey,
                        dsy,
                        psf,
                    )
                for j in range(len(roi_dx)):
                    p_psf[i, j] = psf[j]
                _add_peak(
                    p_n[i],
                    p_bg[i],
                    p_xi[i],
                    p_yi[i],
                    p_psf[i],
                    roi_dx,
                    roi_dy,
                    variance,
                    use_var,
                    f_data,
                    bg_data,
                    bg_counts,
                    1,
                )


@numba.njit(nogil=True, cache=True)
def _n_running(n, p_status):
    count = 0
    for i in range(n):
        if p_status[i] == _RUNNING:
            count += 1
    return count


@numba.njit(nogil=True, cache=True)
def _significance(
    n, p_n, p_xi, p_yi, p_psf, roi_dx, roi_dy, x_data, variance, use_var, out
):
    """Signal over the square root of the background, both weighted by
    the PSF (multi_fit.c: mFitPeakFgSum, mFitPeakBgSum)."""
    for i in range(n):
        psf_sum = 0.0
        fg_sum = 0.0
        bg_sum = 0.0
        for j in range(len(roi_dx)):
            row = p_yi[i] + roi_dy[j]
            col = p_xi[i] + roi_dx[j]
            psf = p_psf[i, j]
            psf_sum += psf
            fg_sum += psf * psf
            var = variance[row, col] if use_var else 0.0
            bg = (x_data[row, col] - var) - p_n[i] * psf
            bg_sum += bg * psf * psf
        fg_sum = p_n[i] * fg_sum
        if psf_sum > 0.0:
            fg_sum /= psf_sum
        else:
            fg_sum = 1.0
        if bg_sum > 0.0 and psf_sum > 0.0:
            bg_sum /= psf_sum * psf_sum
        else:
            bg_sum = 1.0
        out[i] = fg_sum / math.sqrt(bg_sum)


@numba.njit(nogil=True, cache=True)
def _do_fit(
    n,
    p_n,
    p_x,
    p_y,
    p_bg,
    p_sx,
    p_sy,
    p_z,
    p_xi,
    p_yi,
    p_status,
    p_lambda,
    p_error,
    p_iter,
    p_added,
    p_psf,
    n_widths,
    sigma_min,
    sigma_max,
    roi,
    xoff,
    roi_dx,
    roi_dy,
    x_data,
    variance,
    use_var,
    f_data,
    bg_data,
    bg_counts,
    tolerance,
    ex,
    dex,
    dsx,
    ey,
    dey,
    dsy,
    psf,
    spline,
    coeff,
    n_z,
    gx,
    gy,
    gz,
):
    """Iterate until no peak is running (MultiFitter.doFit)."""
    i = -1
    while i < 0 or (_n_running(n, p_status) > 0 and i < _MAX_FIT_ITERATIONS):
        _iterate(
            n,
            p_n,
            p_x,
            p_y,
            p_bg,
            p_sx,
            p_sy,
            p_z,
            p_xi,
            p_yi,
            p_status,
            p_lambda,
            p_error,
            p_iter,
            p_added,
            p_psf,
            n_widths,
            sigma_min,
            sigma_max,
            roi,
            xoff,
            roi_dx,
            roi_dy,
            x_data,
            variance,
            use_var,
            f_data,
            bg_data,
            bg_counts,
            tolerance,
            ex,
            dex,
            dsx,
            ey,
            dey,
            dsy,
            psf,
            spline,
            coeff,
            n_z,
            gx,
            gy,
            gz,
        )
        i += 1


@numba.njit(nogil=True, cache=True)
def _mark_running_near(new_x, new_y, n_new, n, p_x, p_y, p_status, radius):
    """Converged peaks within ``radius`` of a new peak run again
    (ia_utilities.c: runningIfHasNeighbors)."""
    if n_new == 0 or n == 0:
        return
    cells, counts, order, nx, ny, x0, y0 = _build_grid(
        new_x, new_y, n_new, radius
    )
    out = np.empty(n_new, dtype=np.int64)
    for i in range(n):
        if p_status[i] != _CONVERGED:
            continue
        if _neighbors(
            p_x[i],
            p_y[i],
            radius,
            new_x,
            new_y,
            counts,
            order,
            nx,
            ny,
            x0,
            y0,
            radius,
            out,
        ):
            p_status[i] = _RUNNING


@numba.njit(nogil=True, cache=True)
def _prune(
    n, p_n, p_x, p_y, p_status, significance, sigma, threshold, neighborhood
):
    """Remove the dimmer of two peaks closer than ``sigma`` and the peaks
    not more significant than ``threshold``; their neighbors run again
    (ia_utilities.c: markDimmerPeaks, markLowSignificancePeaks). Returns
    the number of peaks marked for removal."""
    if n == 0:
        return 0
    cells, counts, order, nx, ny, x0, y0 = _build_grid(
        p_x, p_y, n, neighborhood
    )
    out = np.empty(n, dtype=np.int64)
    removed = 0
    for i in range(n):
        if p_status[i] == _ERROR:
            continue
        m = _neighbors(
            p_x[i],
            p_y[i],
            sigma,
            p_x,
            p_y,
            counts,
            order,
            nx,
            ny,
            x0,
            y0,
            neighborhood,
            out,
        )
        dimmer = False
        for k in range(m):
            if p_n[out[k]] > p_n[i]:
                dimmer = True
                break
        if dimmer:
            removed += 1
            p_status[i] = _ERROR
            m = _neighbors(
                p_x[i],
                p_y[i],
                neighborhood,
                p_x,
                p_y,
                counts,
                order,
                nx,
                ny,
                x0,
                y0,
                neighborhood,
                out,
            )
            for k in range(m):
                if p_status[out[k]] == _CONVERGED:
                    p_status[out[k]] = _RUNNING
    for i in range(n):
        if p_status[i] == _ERROR:
            continue
        if significance[i] > threshold:
            continue
        p_status[i] = _ERROR
        removed += 1
        m = _neighbors(
            p_x[i],
            p_y[i],
            neighborhood,
            p_x,
            p_y,
            counts,
            order,
            nx,
            ny,
            x0,
            y0,
            neighborhood,
            out,
        )
        for k in range(m):
            if p_status[out[k]] == _CONVERGED:
                p_status[out[k]] = _RUNNING
    return removed


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _local_crlb(
    i,
    n,
    p_n,
    p_x,
    p_y,
    p_sx,
    p_sy,
    p_z,
    p_xi,
    p_yi,
    n_widths,
    spline,
    coeff,
    n_z,
    roi,
    roi_dx,
    roi_dy,
    f_data,
    bg_data,
    bg_counts,
    cell_counts,
    cell_order,
    nx,
    ny,
    x0,
    y0,
    neighbor_distance,
    max_neighbors,
    out_var,
    candidates,
):
    """Cramer-Rao variances of ``[photons, x, y, bg, sx, sy, z]`` of peak
    ``i`` (NaN for a sigma or z not fitted) from the Fisher matrix of it
    and its nearest neighbors (at most ``max_neighbors`` within
    ``neighbor_distance``) over their regions, with ``n_widths`` fitted
    sigmas, or with a ``spline`` PSF ``n_z`` fitted z, per peak among the
    parameters; where that matrix is singular, Huang et al.'s correction
    of the x-x and y-y terms of close emitters (Eq. 11). Returns the
    number of emitters in the matrix. ``candidates`` is scratch of at
    least ``n`` entries."""
    m = _neighbors(
        p_x[i],
        p_y[i],
        neighbor_distance,
        p_x,
        p_y,
        cell_counts,
        cell_order,
        nx,
        ny,
        x0,
        y0,
        neighbor_distance,
        candidates,
    )
    # i first, then the nearest neighbors
    dist = np.empty(m)
    for k in range(m):
        j = candidates[k]
        dist[k] = (p_x[j] - p_x[i]) ** 2 + (p_y[j] - p_y[i]) ** 2
        if j == i:
            dist[k] = -1.0
    order = np.argsort(dist)
    n_local = min(m, max_neighbors + 1)
    members = np.empty(n_local, dtype=np.int64)
    for k in range(n_local):
        members[k] = candidates[order[k]]
    stride = 4 + n_widths + n_z
    n_par = stride * n_local
    fisher = np.zeros((n_par, n_par))
    # the union of the regions: a pixel enters once
    seen = np.zeros((roi * 3, roi * 3), dtype=np.bool_)
    ox = p_xi[i] - roi
    oy = p_yi[i] - roi
    ex = np.empty(roi)
    dex = np.empty(roi)
    dsx = np.empty(roi)
    ey = np.empty(roi)
    dey = np.empty(roi)
    dsy = np.empty(roi)
    grad = np.zeros(n_par)
    profiles = np.empty((n_local, 6, roi))
    disc = np.zeros((roi, roi), dtype=np.bool_)
    # the pixel of the disc at each place of the region
    disc_index = np.full((roi, roi), -1, dtype=np.int64)
    for r in range(len(roi_dx)):
        disc[roi_dy[r], roi_dx[r]] = True
        disc_index[roi_dy[r], roi_dx[r]] = r
    n_roi = len(roi_dx)
    # the spline PSF of each emitter and its x, y and z derivatives
    images = np.empty((n_local if spline else 0, 4, n_roi))
    psf = np.empty(n_roi)
    gx = np.empty(n_roi)
    gy = np.empty(n_roi)
    gz = np.empty(n_roi)
    for k in range(n_local):
        j = members[k]
        if spline:
            _spline_peak_shape(
                coeff,
                p_x[j],
                p_y[j],
                p_z[j],
                p_xi[j],
                p_yi[j],
                roi_dx,
                roi_dy,
                psf,
                gx,
                gy,
                gz,
            )
            for r in range(n_roi):
                images[k, 0, r] = psf[r]
                images[k, 1, r] = gx[r]
                images[k, 2, r] = gy[r]
                images[k, 3, r] = gz[r]
            continue
        _profile(p_x[j], p_xi[j], roi, p_sx[j], ex, dex, dsx)
        _profile(p_y[j], p_yi[j], roi, p_sy[j], ey, dey, dsy)
        for c in range(roi):
            profiles[k, 0, c] = ex[c]
            profiles[k, 1, c] = dex[c]
            profiles[k, 2, c] = ey[c]
            profiles[k, 3, c] = dey[c]
            profiles[k, 4, c] = dsx[c]
            profiles[k, 5, c] = dsy[c]
    for k in range(n_local):
        j = members[k]
        for r in range(len(roi_dx)):
            row = p_yi[j] + roi_dy[r]
            col = p_xi[j] + roi_dx[r]
            sx = col - ox
            sy = row - oy
            if sx < 0 or sy < 0 or sx >= 3 * roi or sy >= 3 * roi:
                continue
            if seen[sy, sx]:
                continue
            seen[sy, sx] = True
            count = bg_counts[row, col]
            if count <= 0:
                continue
            mu = f_data[row, col] + bg_data[row, col] / count
            if not mu > 0.0:
                continue
            for p in range(n_par):
                grad[p] = 0.0
            for kk in range(n_local):
                jj = members[kk]
                a = col - p_xi[jj]
                b = row - p_yi[jj]
                if a < 0 or b < 0 or a >= roi or b >= roi or not disc[b, a]:
                    continue
                if spline:
                    r = disc_index[b, a]
                    base = stride * kk
                    grad[base] = images[kk, 0, r]
                    grad[base + 1] = p_n[jj] * images[kk, 1, r]
                    grad[base + 2] = p_n[jj] * images[kk, 2, r]
                    grad[base + 3] = 1.0 / count
                    if n_z == 1:
                        grad[base + 4] = p_n[jj] * images[kk, 3, r]
                    continue
                exv = profiles[kk, 0, a]
                dexv = profiles[kk, 1, a]
                eyv = profiles[kk, 2, b]
                deyv = profiles[kk, 3, b]
                base = stride * kk
                grad[base] = exv * eyv
                grad[base + 1] = p_n[jj] * dexv * eyv
                grad[base + 2] = p_n[jj] * exv * deyv
                grad[base + 3] = 1.0 / count
                if n_widths == 1:
                    grad[base + 4] = p_n[jj] * (
                        profiles[kk, 4, a] * eyv + exv * profiles[kk, 5, b]
                    )
                elif n_widths == 2:
                    grad[base + 4] = p_n[jj] * profiles[kk, 4, a] * eyv
                    grad[base + 5] = p_n[jj] * exv * profiles[kk, 5, b]
            for p in range(n_par):
                if grad[p] == 0.0:
                    continue
                gp = grad[p] / mu
                for q in range(p, n_par):
                    fisher[p, q] += gp * grad[q]
    for p in range(n_par):
        for q in range(p):
            fisher[p, q] = fisher[q, p]
    cov = _pinv_symmetric(fisher)
    regular = True
    for p in range(4):
        if not (np.isfinite(cov[p, p]) and cov[p, p] > 0.0):
            regular = False
    if not regular and n_local > 1:
        # A singular matrix: Huang et al. (2011), Eq. 11. Sigma from the
        # matrix without the x-x and y-y terms between emitters, then those
        # terms scaled by A / (A + 1), A = (theta_p - theta_q)^2 /
        # (sigma_p sigma_q). Applied only here: with free photons and
        # backgrounds it can make a regular matrix indefinite, and the
        # pruning keeps emitters 1 PSF sigma apart, where the plain bound
        # holds (their Fig. 4a).
        zeroed = fisher.copy()
        for a in range(n_local):
            for b in range(n_local):
                if a != b:
                    zeroed[stride * a + 1, stride * b + 1] = 0.0
                    zeroed[stride * a + 2, stride * b + 2] = 0.0
        cov0 = _pinv_symmetric(zeroed)
        for a in range(n_local):
            for b in range(n_local):
                if a == b:
                    continue
                for axis in (1, 2):
                    p = stride * a + axis
                    q = stride * b + axis
                    s = math.sqrt(abs(cov0[p, p] * cov0[q, q]))
                    if axis == 1:
                        diff = p_x[members[a]] - p_x[members[b]]
                    else:
                        diff = p_y[members[a]] - p_y[members[b]]
                    if s > 0.0:
                        big_a = diff * diff / s
                        fisher[p, q] *= big_a / (big_a + 1.0)
        cov = _pinv_symmetric(fisher)
    for p in range(4):
        out_var[p] = cov[p, p]
    out_var[4] = cov[4, 4] if n_widths > 0 else np.nan
    out_var[5] = cov[3 + n_widths, 3 + n_widths] if n_widths > 0 else np.nan
    out_var[6] = cov[4, 4] if n_z > 0 else np.nan
    return n_local


@numba.njit(nogil=True, cache=True)
def analyze_frame(
    frame,
    variance,
    use_variance,
    peak_mask,
    seed_x,
    seed_y,
    sigma,
    psf_model,
    threshold,
    n_cycles,
    peak_radius,
    background_sigma,
    spline,
):
    """Find and fit the emitters of one frame.

    Parameters
    ----------
    frame : np.ndarray
        ``(height, width)`` frame in photons (float64), padded by
        :func:`padding` pixels on every side by mirroring. Values below one
        photon are taken as one, as the Poisson likelihood needs positive
        data.
    variance : np.ndarray
        Per-pixel sCMOS readout variance in photons squared, padded like
        ``frame``; ignored unless ``use_variance``.
    use_variance : bool
        Whether ``variance`` is applied (Huang et al., Nat. Methods 10,
        653-658, 2013): it is added to the data and the model, and to the
        noise that the peak threshold is relative to.
    peak_mask : np.ndarray
        ``(height, width)`` bool, where new peaks may be found.
    seed_x, seed_y : np.ndarray
        Integer pixel positions (padded frame) that start the first cycle
        instead of the peak finder; empty for none.
    sigma : float
        Nominal PSF standard deviation in camera pixels: fixed, or the
        start of a fitted one. The peaks are found and pruned with it. For
        the spline, its Gaussian-equivalent sigma, which only prunes.
    psf_model : int
        Code of the PSF model (``PSF_MODELS``); a Gaussian model's code is
        the number of sigmas it fits per emitter: 0 (fixed), 1 (the same
        in x and y) or 2 (one in x, one in y).
    threshold : float
        Peak threshold and minimum significance, in units of the
        signal-to-noise ratio.
    n_cycles : int
        Most find-fit-subtract cycles.
    peak_radius : float
        Radius (camera pixels) within which a new peak must be the
        maximum.
    background_sigma : float
        Width (camera pixels) of the background low pass.
    spline : SplinePSF
        The PSF of the "spline" model (see :func:`spline_psf`), and
        :data:`NO_SPLINE` for the others.

    Returns
    -------
    result : FrameResult
        The emitters, in the coordinates of the padded frame.
    """
    height, width = frame.shape
    is_spline = psf_model == _SPLINE
    n_widths = 0 if is_spline else psf_model
    n_z = 1 if is_spline and spline.fit_z else 0
    coeff = spline.coeff
    if is_spline:
        roi = spline.roi
    else:
        roi = _roi_size_jit(sigma, n_widths)
    margin = int(roi / 2 + 2)
    xoff = 0.5 * roi - 1.0
    sigma_min = _SIGMA_RANGE[0] * sigma
    sigma_max = _SIGMA_RANGE[1] * sigma
    # disc of the region (multi_fit.c: mFitInitializeROIIndexing)
    roi_dx_list = []
    roi_dy_list = []
    center = 0.5 * roi
    for i in range(roi):
        dyy = i - center + 0.5
        for j in range(roi):
            dxx = j - center + 0.5
            if dxx * dxx + dyy * dyy <= center * center:
                roi_dx_list.append(j)
                roi_dy_list.append(i)
    n_roi = len(roi_dx_list)
    roi_dx = np.empty(n_roi, dtype=np.int64)
    roi_dy = np.empty(n_roi, dtype=np.int64)
    for j in range(n_roi):
        roi_dx[j] = roi_dx_list[j]
        roi_dy[j] = roi_dy_list[j]

    image = np.empty((height, width))
    x_data = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            value = max(frame[y, x], 1.0)
            image[y, x] = value
            x_data[y, x] = value + (variance[y, x] if use_variance else 0.0)
    f_data = np.zeros((height, width))
    bg_data = np.zeros((height, width))
    bg_counts = np.zeros((height, width), dtype=np.int64)
    background = np.empty((height, width))
    work = np.empty((height, width))
    work2 = np.empty((height, width))
    fg = np.empty((height, width))
    snr = np.empty((height, width))
    tmp = np.empty((height, width))
    taken = np.zeros((height, width), dtype=np.int64)
    size = min(height, width)
    bg_kernel = _gaussian_kernel(background_sigma, size, False)
    # the foreground is smoothed with the PSF: its point-sampled
    # equivalent is a Gaussian of sqrt(sigma^2 + 1/12)
    fg_sigma = math.sqrt(sigma * sigma + 1.0 / 12.0)
    fg_kernel = _gaussian_kernel(fg_sigma, size, False)
    fg_kernel2 = _gaussian_kernel(fg_sigma, size, True)

    capacity = 256
    p_n = np.zeros(capacity)
    p_x = np.zeros(capacity)
    p_y = np.zeros(capacity)
    p_bg = np.zeros(capacity)
    p_sx = np.zeros(capacity)
    p_sy = np.zeros(capacity)
    p_z = np.zeros(capacity)
    p_xi = np.zeros(capacity, dtype=np.int64)
    p_yi = np.zeros(capacity, dtype=np.int64)
    p_status = np.zeros(capacity, dtype=np.int64)
    p_lambda = np.zeros(capacity)
    p_error = np.zeros(capacity)
    p_iter = np.zeros(capacity, dtype=np.int64)
    p_added = np.zeros(capacity, dtype=np.bool_)
    p_psf = np.zeros((capacity, n_roi))
    n = 0
    ex = np.empty(roi)
    dex = np.empty(roi)
    dsx = np.empty(roi)
    ey = np.empty(roi)
    dey = np.empty(roi)
    dsy = np.empty(roi)
    psf = np.empty(n_roi)
    gx = np.empty(n_roi)
    gy = np.empty(n_roi)
    gz = np.empty(n_roi)
    # the spline's matched filters: one more filtered image, and the
    # search plane each pixel is brightest in
    filter_shape = (height, width) if is_spline else (1, 1)
    conv = np.empty(filter_shape)
    acc = np.empty(filter_shape)
    best_plane = np.zeros(filter_shape, dtype=np.int64)
    significance = np.zeros(capacity)
    max_new = height * width // 4 + 16
    new_x = np.empty(max_new, dtype=np.int64)
    new_y = np.empty(max_new, dtype=np.int64)
    neighborhood = _NEIGHBORHOOD * sigma

    for cycle in range(n_cycles):
        # 1. background of what the fit does not explain
        for y in range(height):
            for x in range(width):
                work[y, x] = image[y, x] - f_data[y, x]
        _convolve(work, bg_kernel, background, tmp)
        # 2. new peaks
        if cycle == 0 and len(seed_x) > 0:
            n_new = 0
            for k in range(len(seed_x)):
                if n_new < max_new:
                    new_x[n_new] = seed_x[k]
                    new_y[n_new] = seed_y[k]
                    taken[seed_y[k], seed_x[k]] += 1
                    n_new += 1
        else:
            for y in range(height):
                for x in range(width):
                    var = background[y, x] + f_data[y, x]
                    if use_variance:
                        var += variance[y, x]
                    work[y, x] = var
                    work2[y, x] = image[y, x] - background[y, x] - f_data[y, x]
            if is_spline:
                # the residual filtered with the PSF at every search
                # plane; a pixel keeps its best ratio and that plane
                for plane in range(len(spline.search_z)):
                    _filter_plane(
                        work2,
                        spline.search_kx,
                        spline.search_ky,
                        spline.search_plane,
                        plane,
                        fg,
                        conv,
                        tmp,
                    )
                    _filter_plane(
                        work,
                        spline.variance_kx,
                        spline.variance_ky,
                        spline.variance_plane,
                        plane,
                        acc,
                        conv,
                        tmp,
                    )
                    for y in range(height):
                        for x in range(width):
                            var = acc[y, x]
                            if var <= _VARIANCE_FLOOR:
                                var = _VARIANCE_FLOOR
                            value = fg[y, x] / math.sqrt(var)
                            if plane == 0 or value > snr[y, x]:
                                snr[y, x] = value
                                best_plane[y, x] = plane
                for y in range(height):
                    for x in range(width):
                        if not peak_mask[y, x]:
                            snr[y, x] = 0.0
            else:
                _convolve(work, fg_kernel2, snr, tmp)
                _convolve(work2, fg_kernel, fg, tmp)
                for y in range(height):
                    for x in range(width):
                        var = snr[y, x]
                        if var <= _VARIANCE_FLOOR:
                            var = _VARIANCE_FLOOR
                        value = fg[y, x] / math.sqrt(var)
                        snr[y, x] = value if peak_mask[y, x] else 0.0
            n_new = _find_maxima(
                snr, threshold, peak_radius, margin + 1, taken, new_x, new_y
            )
        # A cycle without new peaks leaves the fit as it is, so every
        # later cycle would find the same nothing. storm-analysis cycles
        # on until its threshold schedule has run down (at least five
        # cycles), to the same result.
        if n_new == 0:
            break
        # 3. fit (PeakFitter.fitPeaks)
        if n > 0:
            fnx = np.empty(n_new)
            fny = np.empty(n_new)
            for k in range(n_new):
                fnx[k] = new_x[k]
                fny[k] = new_y[k]
            _mark_running_near(
                fnx, fny, n_new, n, p_x, p_y, p_status, neighborhood
            )
        if n + n_new > capacity:
            capacity = 2 * (n + n_new)
            p_n = _grow(p_n, capacity)
            p_x = _grow(p_x, capacity)
            p_y = _grow(p_y, capacity)
            p_bg = _grow(p_bg, capacity)
            p_sx = _grow(p_sx, capacity)
            p_sy = _grow(p_sy, capacity)
            p_z = _grow(p_z, capacity)
            p_xi = _grow(p_xi, capacity)
            p_yi = _grow(p_yi, capacity)
            p_status = _grow(p_status, capacity)
            p_lambda = _grow(p_lambda, capacity)
            p_error = _grow(p_error, capacity)
            p_iter = _grow(p_iter, capacity)
            p_added = _grow(p_added, capacity)
            p_psf = _grow2(p_psf, capacity)
            significance = np.zeros(capacity)
        start = n
        # dao_fit.c: daoNewPeaks ("finder" peaks)
        for k in range(n_new):
            i = n
            p_x[i] = float(new_x[k])
            p_y[i] = float(new_y[k])
            p_xi[i] = int(math.floor(p_x[i] - xoff))
            p_yi[i] = int(math.floor(p_y[i] - xoff))
            p_status[i] = _RUNNING
            p_lambda[i] = _LAMBDA_START
            p_error[i] = 0.0
            p_iter[i] = 0
            p_added[i] = False
            p_n[i] = 1.0
            p_bg[i] = 1.0
            p_sx[i] = sigma
            p_sy[i] = sigma
            if n_z == 1:
                # the search plane the peak is brightest in; seeds start
                # at the first
                plane = 0
                if not (cycle == 0 and len(seed_x) > 0):
                    plane = best_plane[new_y[k], new_x[k]]
                p_z[i] = spline.search_z[plane]
            else:
                p_z[i] = 0.0
            n += 1
            if (
                p_xi[i] < 0
                or p_xi[i] >= width - roi
                or p_yi[i] < 0
                or p_yi[i] >= height - roi
            ):
                p_status[i] = _ERROR
                continue
            if is_spline:
                _spline_peak_shape(
                    coeff,
                    p_x[i],
                    p_y[i],
                    p_z[i],
                    p_xi[i],
                    p_yi[i],
                    roi_dx,
                    roi_dy,
                    psf,
                    gx,
                    gy,
                    gz,
                )
            else:
                _peak_shape(
                    p_x[i],
                    p_y[i],
                    p_xi[i],
                    p_yi[i],
                    sigma,
                    sigma,
                    roi_dx,
                    roi_dy,
                    ex,
                    dex,
                    dsx,
                    ey,
                    dey,
                    dsy,
                    psf,
                )
            bg_sum = 0.0
            sp = 0.0
            sx = 0.0
            for j in range(n_roi):
                row = p_yi[i] + roi_dy[j]
                col = p_xi[i] + roi_dx[j]
                bg_sum += background[row, col]
                t1 = psf[j]
                sp += t1 * t1 * t1
                sx += (
                    t1
                    * t1
                    * (
                        x_data[row, col]
                        - f_data[row, col]
                        - background[row, col]
                    )
                )
                p_psf[i, j] = psf[j]
            p_bg[i] = bg_sum / n_roi
            p_n[i] = sx / sp
            if p_n[i] <= 0.0:
                p_status[i] = _ERROR
                continue
            _add_peak(
                p_n[i],
                p_bg[i],
                p_xi[i],
                p_yi[i],
                p_psf[i],
                roi_dx,
                roi_dy,
                variance,
                use_variance,
                f_data,
                bg_data,
                bg_counts,
                1,
            )
            p_added[i] = True
        for i in range(start, n):
            if p_status[i] == _RUNNING:
                ok, p_error[i] = _peak_error(
                    p_xi[i],
                    p_yi[i],
                    roi_dx,
                    roi_dy,
                    x_data,
                    f_data,
                    bg_data,
                    bg_counts,
                )
        _do_fit(
            n,
            p_n,
            p_x,
            p_y,
            p_bg,
            p_sx,
            p_sy,
            p_z,
            p_xi,
            p_yi,
            p_status,
            p_lambda,
            p_error,
            p_iter,
            p_added,
            p_psf,
            n_widths,
            sigma_min,
            sigma_max,
            roi,
            xoff,
            roi_dx,
            roi_dy,
            x_data,
            variance,
            use_variance,
            f_data,
            bg_data,
            bg_counts,
            _TOLERANCE,
            ex,
            dex,
            dsx,
            ey,
            dey,
            dsy,
            psf,
            is_spline,
            coeff,
            n_z,
            gx,
            gy,
            gz,
        )
        keep = p_status[:n] != _ERROR
        n = _compact(
            keep,
            n,
            p_n,
            p_x,
            p_y,
            p_bg,
            p_sx,
            p_sy,
            p_z,
            p_xi,
            p_yi,
            p_status,
            p_lambda,
            p_error,
            p_iter,
            p_added,
            p_psf,
        )
        _significance(
            n,
            p_n,
            p_xi,
            p_yi,
            p_psf,
            roi_dx,
            roi_dy,
            x_data,
            variance,
            use_variance,
            significance,
        )
        removed = _prune(
            n,
            p_n,
            p_x,
            p_y,
            p_status,
            significance,
            sigma,
            threshold,
            neighborhood,
        )
        if removed > 0:
            for i in range(n):
                if p_status[i] == _ERROR and p_added[i]:
                    _add_peak(
                        p_n[i],
                        p_bg[i],
                        p_xi[i],
                        p_yi[i],
                        p_psf[i],
                        roi_dx,
                        roi_dy,
                        variance,
                        use_variance,
                        f_data,
                        bg_data,
                        bg_counts,
                        -1,
                    )
                    p_added[i] = False
            keep = p_status[:n] != _ERROR
            n = _compact(
                keep,
                n,
                p_n,
                p_x,
                p_y,
                p_bg,
                p_sx,
                p_sy,
                p_z,
                p_xi,
                p_yi,
                p_status,
                p_lambda,
                p_error,
                p_iter,
                p_added,
                p_psf,
            )
        if _n_running(n, p_status) > 0:
            _do_fit(
                n,
                p_n,
                p_x,
                p_y,
                p_bg,
                p_sx,
                p_sy,
                p_z,
                p_xi,
                p_yi,
                p_status,
                p_lambda,
                p_error,
                p_iter,
                p_added,
                p_psf,
                n_widths,
                sigma_min,
                sigma_max,
                roi,
                xoff,
                roi_dx,
                roi_dy,
                x_data,
                variance,
                use_variance,
                f_data,
                bg_data,
                bg_counts,
                _TOLERANCE,
                ex,
                dex,
                dsx,
                ey,
                dey,
                dsy,
                psf,
                is_spline,
                coeff,
                n_z,
                gx,
                gy,
                gz,
            )
            keep = p_status[:n] != _ERROR
            n = _compact(
                keep,
                n,
                p_n,
                p_x,
                p_y,
                p_bg,
                p_sx,
                p_sy,
                p_z,
                p_xi,
                p_yi,
                p_status,
                p_lambda,
                p_error,
                p_iter,
                p_added,
                p_psf,
            )

    # peaks that did not converge are dropped (removeRunningPeaks)
    for i in range(n):
        if p_status[i] == _RUNNING and p_added[i]:
            _add_peak(
                p_n[i],
                p_bg[i],
                p_xi[i],
                p_yi[i],
                p_psf[i],
                roi_dx,
                roi_dy,
                variance,
                use_variance,
                f_data,
                bg_data,
                bg_counts,
                -1,
            )
            p_added[i] = False
    keep = p_status[:n] == _CONVERGED
    n = _compact(
        keep,
        n,
        p_n,
        p_x,
        p_y,
        p_bg,
        p_sx,
        p_sy,
        p_z,
        p_xi,
        p_yi,
        p_status,
        p_lambda,
        p_error,
        p_iter,
        p_added,
        p_psf,
    )
    _significance(
        n,
        p_n,
        p_xi,
        p_yi,
        p_psf,
        roi_dx,
        roi_dy,
        x_data,
        variance,
        use_variance,
        significance,
    )
    # the final deviance of each region, with every peak in place
    for i in range(n):
        ok, p_error[i] = _peak_error(
            p_xi[i],
            p_yi[i],
            roi_dx,
            roi_dy,
            x_data,
            f_data,
            bg_data,
            bg_counts,
        )
    crlb = np.full((n, 7), np.nan)
    n_emitters = np.ones(n, dtype=np.int64)
    if n > 0:
        neighbor_distance = _NEIGHBOR_DISTANCE * sigma
        cells, counts, order, nx, ny, x0, y0 = _build_grid(
            p_x, p_y, n, neighbor_distance
        )
        out_var = np.empty(7)
        # Once per frame, not once per emitter: above 16 KB (2048 peaks),
        # Windows serializes the allocation across threads, which made
        # the bounds of a dense frame several times slower on 48 threads.
        candidates = np.empty(n, dtype=np.int64)
        for i in range(n):
            n_emitters[i] = _local_crlb(
                i,
                n,
                p_n,
                p_x,
                p_y,
                p_sx,
                p_sy,
                p_z,
                p_xi,
                p_yi,
                n_widths,
                is_spline,
                coeff,
                n_z,
                roi,
                roi_dx,
                roi_dy,
                f_data,
                bg_data,
                bg_counts,
                counts,
                order,
                nx,
                ny,
                x0,
                y0,
                neighbor_distance,
                _MAX_NEIGHBORS,
                out_var,
                candidates,
            )
            for p in range(7):
                crlb[i, p] = out_var[p]
    n_pixels = np.full(n, n_roi, dtype=np.int64)
    sx_out = p_sx[:n].copy()
    sy_out = p_sy[:n].copy()
    z_out = p_z[:n].copy()
    if is_spline:
        sx_out[:] = np.nan
        sy_out[:] = np.nan
    if n_z == 0:
        z_out[:] = np.nan
    return FrameResult(
        p_x[:n].copy(),
        p_y[:n].copy(),
        p_n[:n].copy(),
        p_bg[:n].copy(),
        sx_out,
        sy_out,
        z_out,
        significance[:n].copy(),
        p_error[:n].copy(),
        p_iter[:n].copy(),
        n_pixels,
        4 + n_widths + n_z,
        crlb,
        n_emitters,
    )


# ----------------------------------------------------------------------
# Peaks of the first cycle, and single emitters with a free width: the
# PSF sigma of a movie is the median sigma of its isolated emitters.
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def find_peaks(
    frame, peak_mask, sigma, threshold, peak_radius, background_sigma
):
    """The peaks the first cycle of :func:`analyze_frame` finds.

    Parameters
    ----------
    frame : np.ndarray
        Padded frame in photons, as for :func:`analyze_frame`.
    peak_mask : np.ndarray
        Where peaks may be found.
    sigma : float
        PSF standard deviation (camera pixels) the residual is smoothed
        with.
    threshold : float
        Peak threshold in units of the signal-to-noise ratio.
    peak_radius : float
        Radius (camera pixels) within which a peak is the maximum.
    background_sigma : float
        Width (camera pixels) of the background low pass.

    Returns
    -------
    x, y : np.ndarray
        Integer pixel positions of the peaks in the padded frame.
    """
    height, width = frame.shape
    roi = int(_ROI_SIZE_FACTOR * sigma)
    margin = int(roi / 2 + 2)
    image = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            image[y, x] = max(frame[y, x], 1.0)
    background = np.empty((height, width))
    tmp = np.empty((height, width))
    size = min(height, width)
    _convolve(
        image,
        _gaussian_kernel(background_sigma, size, False),
        background,
        tmp,
    )
    fg_sigma = math.sqrt(sigma * sigma + 1.0 / 12.0)
    work = np.empty((height, width))
    for y in range(height):
        for x in range(width):
            work[y, x] = image[y, x] - background[y, x]
    fg = np.empty((height, width))
    _convolve(work, _gaussian_kernel(fg_sigma, size, False), fg, tmp)
    var = np.empty((height, width))
    _convolve(background, _gaussian_kernel(fg_sigma, size, True), var, tmp)
    for y in range(height):
        for x in range(width):
            v = max(var[y, x], _VARIANCE_FLOOR)
            work[y, x] = fg[y, x] / math.sqrt(v) if peak_mask[y, x] else 0.0
    taken = np.zeros((height, width), dtype=np.int64)
    out_x = np.empty(height * width // 4 + 16, dtype=np.int64)
    out_y = np.empty(height * width // 4 + 16, dtype=np.int64)
    n = _find_maxima(
        work, threshold, peak_radius, margin + 1, taken, out_x, out_y
    )
    return out_x[:n].copy(), out_y[:n].copy()


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _profile_sigma(center, sigma, size, e, de, ds):
    """Gaussian integrated over the pixels ``0 .. size - 1``, and its
    derivatives with respect to ``center`` and to ``sigma``."""
    inv = _INV_SQRT2 / sigma
    norm = _INV_SQRT2PI / sigma
    norm_s = _INV_SQRT2PI / (sigma * sigma)
    inv2 = 0.5 / (sigma * sigma)
    for c in range(size):
        a_plus = c - center + 0.5
        a_minus = c - center - 0.5
        g_plus = math.exp(-a_plus * a_plus * inv2)
        g_minus = math.exp(-a_minus * a_minus * inv2)
        e[c] = 0.5 * (math.erf(a_plus * inv) - math.erf(a_minus * inv))
        de[c] = -norm * (g_plus - g_minus)
        ds[c] = -norm_s * (a_plus * g_plus - a_minus * g_minus)


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_free_sigma(
    spots,
    index,
    variance,
    use_variance,
    theta,
    ex,
    dex,
    dsx,
    ey,
    dey,
    dsy,
    d,
    hess,
    grad,
):
    """Poisson deviance, gradient and Hessian of the pixel-integrated
    Gaussian ``[photons, x, y, sigma, bg]``."""
    box = spots.shape[1]
    for p in range(5):
        if not np.isfinite(theta[p]):
            return np.inf, False
        grad[p] = 0.0
        for q in range(5):
            hess[p, q] = 0.0
    photons = theta[0]
    sigma = theta[3]
    if not sigma > 0.05:
        return np.inf, False
    _profile_sigma(theta[1], sigma, box, ex, dex, dsx)
    _profile_sigma(theta[2], sigma, box, ey, dey, dsy)
    chi_square = 0.0
    for j in range(box):
        for i in range(box):
            g = ex[i] * ey[j]
            value = photons * g + theta[4]
            d[0] = g
            d[1] = photons * dex[i] * ey[j]
            d[2] = photons * ex[i] * dey[j]
            d[3] = photons * (dsx[i] * ey[j] + ex[i] * dsy[j])
            d[4] = 1.0
            var = variance[index, j, i] if use_variance else 0.0
            contrib, weight, factor, ok = _estimator_terms(
                True, value, spots[index, j, i], var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            for p in range(5):
                grad[p] += d[p] * factor
                wp = weight * d[p]
                for q in range(p, 5):
                    hess[p, q] += wp * d[q]
    for p in range(5):
        for q in range(p):
            hess[p, q] = hess[q, p]
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _fit_free_sigma_spots(
    spots,
    variance,
    use_variance,
    init,
    tolerance,
    max_iterations,
    thetas,
    chi_squares,
    states,
    iterations,
):
    """Levenberg-Marquardt fits of :func:`_accumulate_free_sigma`, the
    driver of ``picasso.fitting.gaussfit``."""
    n_params = 5
    box = spots.shape[1]
    theta = np.empty(n_params)
    theta_previous = np.empty(n_params)
    grad = np.zeros(n_params)
    grad_ok = np.zeros(n_params)
    delta = np.empty(n_params)
    scaling = np.zeros(n_params)
    hess = np.zeros((n_params, n_params))
    hess_ok = np.zeros((n_params, n_params))
    hess_damped = np.empty((n_params, n_params))
    indxc = np.empty(n_params, dtype=np.int32)
    indxr = np.empty(n_params, dtype=np.int32)
    ipiv = np.empty(n_params, dtype=np.int32)
    ex = np.empty(box)
    dex = np.empty(box)
    dsx = np.empty(box)
    ey = np.empty(box)
    dey = np.empty(box)
    dsy = np.empty(box)
    d = np.empty(n_params)
    for index in range(len(spots)):
        for p in range(n_params):
            theta[p] = init[index, p]
            scaling[p] = 0.0
        chi_square, ok = _accumulate_free_sigma(
            spots,
            index,
            variance,
            use_variance,
            theta,
            ex,
            dex,
            dsx,
            ey,
            dey,
            dsy,
            d,
            hess,
            grad,
        )
        n_iterations = 0
        state = FIT_STATE_CONVERGED
        if not ok:
            state = FIT_STATE_NEG_CURVATURE_MLE
        else:
            for p in range(n_params):
                grad_ok[p] = grad[p]
                for q in range(n_params):
                    hess_ok[p, q] = hess[p, q]
            lam = _LAMBDA_INITIAL
            previous_chi_square = chi_square
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
                new_chi_square, ok = _accumulate_free_sigma(
                    spots,
                    index,
                    variance,
                    use_variance,
                    theta,
                    ex,
                    dex,
                    dsx,
                    ey,
                    dey,
                    dsy,
                    d,
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
        _lm_finalize(
            theta,
            chi_square,
            state,
            n_iterations,
            True,
            n_params,
            thetas,
            chi_squares,
            states,
            iterations,
            index,
        )


def fit_free_sigma(
    spots: np.ndarray,
    variance: np.ndarray | None = None,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit single emitters with a pixel-integrated Gaussian of free width.

    The maximum-likelihood fit ``[photons, x, y, sigma, bg]`` on each
    spot, seeded at the background-subtracted center of mass with a
    width of one pixel.

    Parameters
    ----------
    spots : np.ndarray
        ``(n_spots, box, box)`` spots in photons, at least zero.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance in photons squared, laid out like
        ``spots``. Default is None.
    tolerance : float, optional
        Relative convergence tolerance of the deviance. Default is 1e-5.
    max_iterations : int, optional
        Most iterations per spot. Default is 100.

    Returns
    -------
    thetas : np.ndarray
        ``(n_spots, 5)`` parameters, NaN where the fit failed; x and y are
        box pixel coordinates.
    deviance : np.ndarray
        ``(n_spots,)`` Poisson deviance at the optimum.
    """
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    n, box, _ = spots.shape
    background = spots.reshape(n, -1).min(axis=1).astype(np.float64)
    weights = np.clip(spots - background[:, None, None], 0, None).astype(
        np.float64
    )
    total = weights.sum(axis=(1, 2))
    grid = np.arange(box, dtype=np.float64)
    center = 0.5 * (box - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (weights.sum(axis=1) * grid).sum(axis=1) / total
        cy = (weights.sum(axis=2) * grid).sum(axis=1) / total
    cx = np.where(np.isfinite(cx), cx, center)
    cy = np.where(np.isfinite(cy), cy, center)
    init = np.column_stack(
        [np.maximum(total, 1.0), cx, cy, np.ones(n), background]
    )
    if variance is None:
        variance = np.zeros((1, 1, 1), dtype=np.float32)
        use_variance = False
    else:
        variance = np.ascontiguousarray(variance, dtype=np.float32)
        use_variance = True
    thetas = np.full((n, 5), np.nan)
    chi_squares = np.full(n, np.inf)
    states = np.zeros(n, dtype=np.int32)
    iterations = np.zeros(n, dtype=np.int32)
    if n:
        _fit_free_sigma_spots(
            spots,
            variance,
            use_variance,
            init,
            float(tolerance),
            int(max_iterations),
            thetas,
            chi_squares,
            states,
            iterations,
        )
    return thetas, chi_squares
