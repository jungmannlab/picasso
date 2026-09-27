"""
picasso.fitting.seeds
~~~~~~~~~~~~~~~~~~~~~

Initial parameters for the Levenberg-Marquardt PSF fits.

Every fitter in this package takes the initial parameter vector from its
caller - the Gpufit contract, which the ported modules keep. This module is
the other half of it: the closed-form seeds Picasso hands them, one function
per model family, each returning the ``(n_spots, n_parameters)`` float32 array
the corresponding ``fit_spots`` expects.

It lives beside the models rather than in ``picasso.localize`` because the
column order of every array it builds is defined by the fit kernels
(``_accumulate_rotated``, ``_accumulate_link_xyz`` and the rest).

=========================================  ===================================
function                                   seeds
=========================================  ===================================
:func:`initial_parameters_gauss`           the spherical, elliptical and
                                           rotated 2D Gaussians
:func:`initial_parameters_gauss_multichannel`  the multichannel spherical
                                           Gaussian, both photon-linked and
                                           photon-decoupled
:func:`initial_parameters_spline`          the 2D, 3D and multichannel
                                           cubic-spline PSFs
=========================================  ===================================

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np

from picasso import lib
from picasso.fitting import precision

__all__ = [
    "initial_parameters_gauss",
    "initial_parameters_gauss_multichannel",
    "initial_parameters_spline",
]


# ----------------------------------------------------------------------
# 2D Gaussian PSFs
# ----------------------------------------------------------------------


def _initial_widths_gauss(
    spots: lib.FloatArray3D,
    size: int,
    background: lib.FloatArray1D,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Seed the Gaussian widths from the second moment of the spot's central
    row and column."""
    half = int(size / 2)
    # float64 throughout: the moment sums photons weighted by a squared
    # distance, which overflows a float32 accumulator on a bright spot.
    d2 = (np.arange(size, dtype=np.float64) - half) ** 2
    # spots is (n, y, x): the central column varies along y, the row along x.
    profile_y = spots[:, :, half].astype(np.float64) - background[:, None]
    profile_x = spots[:, half, :].astype(np.float64) - background[:, None]
    np.clip(profile_y, 0.0, None, out=profile_y)
    np.clip(profile_x, 0.0, None, out=profile_x)
    # A non-finite profile (an inf from a gain map, say) falls through to the
    # fallback below rather than warning here.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        width_y = np.sqrt(profile_y @ d2 / profile_y.sum(axis=1))
        width_x = np.sqrt(profile_x @ d2 / profile_x.sum(axis=1))
    # An empty or single-pixel profile yields 0/0 or a zero width.
    fallback = max(size / 5.0, 1.0)
    width_y[~np.isfinite(width_y)] = fallback
    width_x[~np.isfinite(width_x)] = fallback
    np.minimum(width_y, fallback, out=width_y)
    np.minimum(width_x, fallback, out=width_x)
    np.clip(width_y, 0.5, size / 3.0, out=width_y)
    np.clip(width_x, 0.5, size / 3.0, out=width_x)
    return width_x, width_y


def _initial_shape_gauss_rotated(
    spots: lib.FloatArray3D,
    size: int,
    background: lib.FloatArray1D,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D, lib.FloatArray1D]:
    """Seed the widths and the rotation angle of a rotated elliptical
    Gaussian from the spot's 2D second-moment tensor.

    The central row and column (see ``_initial_widths_gauss``) say nothing
    about the orientation, so seeding the angle with zero leaves the fit to
    find it on its own - which it fails to do for the steeper angles, where
    the least-squares problem has the optimizer walk the angle away by
    several turns. The moment tensor gives both principal widths and the
    orientation directly.

    Returns ``(width_u, width_v, angle)``, where ``width_u`` is the width
    along the axis the model's ``angle`` refers to.
    """
    half = int(size / 2)
    # float64: the moments weight photons by squared distances, which
    # overflows a float32 accumulator on a bright spot.
    g = np.arange(size, dtype=np.float64) - half
    image = spots.astype(np.float64) - background[:, None, None]
    np.clip(image, 0.0, None, out=image)
    # spots is (n, y, x)
    x = g[None, None, :]
    y = g[None, :, None]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        total = image.sum(axis=(1, 2))[:, None, None]
        weights = image / total
        dx = x - (weights * x).sum(axis=(1, 2))[:, None, None]
        dy = y - (weights * y).sum(axis=(1, 2))[:, None, None]
        m_xx = (weights * dx**2).sum(axis=(1, 2))
        m_yy = (weights * dy**2).sum(axis=(1, 2))
        m_xy = (weights * dx * dy).sum(axis=(1, 2))
        # eigenvalues of [[m_xx, m_xy], [m_xy, m_yy]]
        mean = 0.5 * (m_xx + m_yy)
        spread = np.sqrt((0.5 * (m_xx - m_yy)) ** 2 + m_xy**2)
        width_u = np.sqrt(mean + spread)
        width_v = np.sqrt(mean - spread)
        # The model's u axis is ``(cos angle, -sin angle)`` in image
        # coordinates, so the angle is minus the usual orientation of the
        # major axis.
        angle = -0.5 * np.arctan2(2 * m_xy, m_xx - m_yy)

    fallback = max(size / 5.0, 1.0)
    invalid = ~(np.isfinite(width_u) & np.isfinite(width_v))
    width_u[invalid] = fallback
    width_v[invalid] = fallback
    angle[invalid | ~np.isfinite(angle)] = 0.0
    # On a wide box most pixels are background, which inflates the moments;
    # cap them exactly as ``_initial_widths_gauss`` does, since a seed
    # several times wider than the PSF makes the first (undamped) MLE step
    # overshoot the background below zero.
    np.minimum(width_u, fallback, out=width_u)
    np.minimum(width_v, fallback, out=width_v)
    np.clip(width_u, 0.5, size / 3.0, out=width_u)
    np.clip(width_v, 0.5, size / 3.0, out=width_v)
    # With equal widths the rotated Gaussian does not depend on the angle,
    # so its derivative is exactly zero and the first LM Hessian is
    # singular - the fit then aborts, returning the initial parameters.
    # Break the symmetry to keep the angle parameter well-defined.
    degenerate = width_u < 1.01 * width_v
    width_u[degenerate] *= 1.1
    width_v[degenerate] *= 0.9
    return width_u, width_v, angle


def initial_parameters_gauss(
    spots: lib.FloatArray3D,
    size: int,
    rotated: bool = False,
    spherical: bool = False,
) -> lib.FloatArray2D:
    """Initialize the parameters for a 2D Gaussian fit.

    The layout is amplitude, x, y, sx, sy, bg (plus the rotation angle if
    ``rotated``). If ``spherical``, a single width is used and the layout is
    amplitude, x, y, s, bg (the isotropic ``GAUSS_2D`` model).

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, size, size)`` spots, indexed ``[spot, y, x]``.
    size : int
        Box side length in camera pixels.
    rotated : bool, optional
        Seed the rotated elliptical Gaussian, with the widths and angle taken
        from the spot's second-moment tensor. Default False.
    spherical : bool, optional
        Seed the isotropic Gaussian with a single width. Takes precedence
        over ``rotated``. Default False.

    Returns
    -------
    initial_parameters : lib.FloatArray2D
        ``(n_spots, n_params)`` float32 seeds, with ``n_params`` 5
        (spherical), 6 (elliptical) or 7 (rotated). The position is seeded at
        the box center.
    """
    center = (size / 2.0) - 0.5

    spot_max = np.amax(spots, axis=(1, 2))
    spot_min = np.amin(spots, axis=(1, 2))

    width_x, width_y = _initial_widths_gauss(spots, size, spot_min)

    if spherical:
        # GAUSS_2D: photons, x, y, s (single width), bg.
        initial_parameters = np.empty((len(spots), 5), dtype=np.float32)
        initial_parameters[:, 0] = spot_max - spot_min
        initial_parameters[:, 1] = center
        initial_parameters[:, 2] = center
        initial_parameters[:, 3] = 0.5 * (width_x + width_y)
        initial_parameters[:, 4] = spot_min
        return initial_parameters

    n_parameters = 7 if rotated else 6
    initial_parameters = np.empty((len(spots), n_parameters), dtype=np.float32)

    initial_parameters[:, 0] = spot_max - spot_min
    initial_parameters[:, 1] = center
    initial_parameters[:, 2] = center
    initial_parameters[:, 3] = width_x
    initial_parameters[:, 4] = width_y
    initial_parameters[:, 5] = spot_min
    if rotated:
        # the central row and column carry no orientation, so the widths
        # and the angle are seeded from the full second-moment tensor
        width_u, width_v, angle = _initial_shape_gauss_rotated(
            spots, size, spot_min
        )
        initial_parameters[:, 3] = width_u
        initial_parameters[:, 4] = width_v
        initial_parameters[:, 6] = angle

    return initial_parameters


def initial_parameters_gauss_multichannel(
    spots: np.ndarray,
    size: int,
    link_photons: bool = True,
) -> lib.FloatArray2D:
    """Seeds for a multichannel spherical Gaussian fit.

    ``spots`` is channel-major ``(n_spots, n_channels, box, box)``. The shared
    position and width are seeded from the **reference channel**, which is
    exactly right: its Jacobian is the identity and its ROI residual is zero,
    so the position that describes its spot *is* the shared position the fit
    solves for.

    With ``link_photons`` the layout is the single-channel one
    (``[amplitude, x, y, sigma, background]``) and
    :func:`initial_parameters_gauss` is reused unchanged. Otherwise each
    channel's own photon count and background are seeded from that channel's
    own spot, giving ``[x, y, sigma, N_0.., bg_0..]``.

    Parameters
    ----------
    spots : np.ndarray
        Channel-major ``(n_spots, n_channels, size, size)`` spots; channel 0
        is the reference channel.
    size : int
        Box side length in camera pixels.
    link_photons : bool, optional
        Seed the photon-linked model (one amplitude and background shared by
        all channels) instead of the photon-decoupled one. Default True.

    Returns
    -------
    initial_parameters : lib.FloatArray2D
        ``(n_spots, 5)`` float32 seeds if ``link_photons``, else
        ``(n_spots, 3 + 2 * n_channels)``.
    """
    n_channels = spots.shape[1]
    reference = np.ascontiguousarray(spots[:, 0])
    shared = initial_parameters_gauss(reference, size, spherical=True)
    if link_photons:
        return shared
    n_spots = len(spots)
    initial = np.empty((n_spots, 3 + 2 * n_channels), dtype=np.float32)
    initial[:, 0] = shared[:, 1]
    initial[:, 1] = shared[:, 2]
    initial[:, 2] = shared[:, 3]
    for c in range(n_channels):
        channel = spots[:, c]
        channel_min = np.amin(channel, axis=(1, 2))
        initial[:, 3 + c] = np.amax(channel, axis=(1, 2)) - channel_min
        initial[:, 3 + n_channels + c] = channel_min
    return initial


# ----------------------------------------------------------------------
# Cubic-spline PSFs
# ----------------------------------------------------------------------


def initial_parameters_spline(
    spots: lib.FloatArray3D, calibration: dict
) -> lib.FloatArray2D:
    """Initialize spline fit parameters per spot.

    Parameter order matches the spline models:
    ``[amplitude, x_shift, y_shift, offset]`` (2D) or
    ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D and 3D
    multichannel). The spline model evaluates the spline at
    ``position = pixel_index - parameter`` (see ``spline_3d.cuh``), so:

    - x_shift/y_shift are the emitter's lateral offset from the (centered)
      template, i.e. 0 for a spot centered in its ROI.

    For the multichannel model ``spots`` is channel-stacked
    ``(n, box, box, n_channels)``; amplitude/offset are estimated across all
    channels. The photon-decoupled (link-xyz) model is seeded as
    ``[x_shift, y_shift, z_shift, N_0.., bg_0..]``, with each channel's
    photons and background estimated from that channel alone.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n, box, box)`` spots, or ``(n, box, box, n_channels)`` for the
        multichannel models.
    calibration : dict
        The spline PSF calibration; its ``"model"`` selects the layout and
        its ``"z_init"`` (or ``"z_center"``) the initial axial position.

    Returns
    -------
    initial_parameters : lib.FloatArray2D
        ``(n, n_params)`` float32 seeds in the column order above.
    """
    model = calibration["model"]
    if model == precision._LINK_XYZ_MODEL:
        # Photon-decoupled (link-XYZ) model: parameters
        # [x_shift, y_shift, z_shift, N_0..N_{c-1}, bg_0..bg_{c-1}], with the
        # photon amplitude and background estimated PER CHANNEL (that is the
        # whole point). spots is (n, box, box, n_channels).
        n_channels = precision._spline_n_channels(calibration)
        per_ch = np.asarray(spots)  # (n, box, box, n_channels)
        ch_max = np.amax(per_ch, axis=(1, 2))  # (n, n_channels)
        ch_min = np.amin(per_ch, axis=(1, 2))  # (n, n_channels)
        z_init = float(
            calibration.get("z_init", calibration.get("z_center", 0.0))
        )
        initial = np.zeros((len(spots), 3 + 2 * n_channels), dtype=np.float32)
        # x_shift (0), y_shift (1) start at 0 (spot centered); z_shift (2).
        initial[:, 2] = -z_init
        initial[:, 3 : 3 + n_channels] = ch_max - ch_min  # per-channel photons
        initial[:, 3 + n_channels :] = ch_min  # per-channel background
        return initial
    if model == "spline-2d":
        n_parameters = 4
    else:
        n_parameters = 5
    # spots is (n, box, box) or, for multichannel, (n, box, box, n_channels)
    reduce_axes = tuple(range(1, spots.ndim))
    spot_max = np.amax(spots, axis=reduce_axes)
    spot_min = np.amin(spots, axis=reduce_axes)
    initial = np.zeros((len(spots), n_parameters), dtype=np.float32)
    initial[:, 0] = spot_max - spot_min  # amplitude
    # x_shift (col 1) and y_shift (col 2) start at 0 (spot centered in the
    # ROI).
    if model == "spline-2d":
        initial[:, 3] = spot_min  # offset
    else:
        z_init = float(
            calibration.get("z_init", calibration.get("z_center", 0.0))
        )
        initial[:, 3] = -z_init  # z_shift (in-focus start; see docstring)
        initial[:, 4] = spot_min  # offset
    return initial
