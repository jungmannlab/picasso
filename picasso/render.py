"""
picasso.render
~~~~~~~~~~~~~~

Render single molecule localizations to a super-resolution image.

Provides functions for painting onto rendered images (QImage), such as
scale bar and picks.

:authors: Joerg Schnitzbauer 2015, Rafal Kowalewski 2023
:copyright: Copyright (c) 2015 Jungmann Lab, MPI of Biochemistry
"""

from email.mime import image
from typing import Literal

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.transform import Rotation
from PyQt6 import QtGui, QtCore, QtSvg

from . import lib


_DRAW_MAX_SIGMA = 3  # max. sigma from mean to render (mu +/- 3 sigma)
N_GROUP_COLORS = 8
POLYGON_POINTER_SIZE = 16  # must be even


def render(
    locs: pd.DataFrame,
    info: dict | None = None,
    oversampling: float = 1.0,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | None = None,
    disp_px_size: float | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations given FOV and blur method.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    info : dict, optional
        Contains localizations metadata. Needed only if no viewport
        specified.
    oversampling : float, optional
        Number of super-resolution pixels per camera pixel. Default is
        1. Deprecated, use disp_px_size instead. Will be removed in
        v0.11.0. Ignored if disp_px_size is specified.
    viewport : tuple, optional
        Field of view to be rendered (in camera pixels). The input is
        ``((y_min, x_min), (y_max, x_max))``. If None, all localizations
        are rendered.
    blur_method : {"gaussian", "gaussian_iso", "smooth", "convolve"} or None, \
            optional
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, no
        blurring is applied. 'gaussian' uses localization precisions
        of each localization to blur it (different in each dimension).
        'gaussian_iso' is similar but averages x and y localization
        precisions, so that blur is isotropic. 'smooth' applies a one
        pixel blur. 'convolve' applies the same blur to all
        localizations which is the median localization precision.
    min_blur_width : float, optional
        Minimum size of blur (camera pixels).
    ang : tuple, optional
        Rotation angles of locs around x, y and z axes in radians. If
        None, locs are not rotated.
    disp_px_size : float, optional
        Display pixel size in nm. Will replace oversampling in v0.11.0.

    Raises
    ------
    Exception
        If blur_method not one of 'gaussian', 'gaussian_iso', 'smooth',
        'convolve' or None.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    if disp_px_size is None:
        lib.deprecation_warning(
            "Deprecation warning: the 'oversampling' parameter is "
            "deprecated and will be removed in v0.11.0. Use "
            "'disp_px_size' instead."
        )
        disp_px_size = pixelsize / oversampling
    oversampling = pixelsize / disp_px_size

    if viewport is None:
        try:
            # all locs
            viewport = [(0, 0), (info[0]["Height"], info[0]["Width"])]
        except TypeError:
            raise ValueError("Need info if no viewport is provided.")
    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        # no blur
        return _render_hist(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            ang=ang,
        )
    elif blur_method == "gaussian":
        # individual localization precision
        return _render_gaussian(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "gaussian_iso":
        # individual localization precision (same for x and y)
        return _render_gaussian_iso(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "smooth":
        # one pixel blur
        return _render_smooth(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            ang=ang,
        )
    elif blur_method == "convolve":
        # global localization precision
        return _render_convolve(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    else:
        raise Exception("blur_method not understood.")


@numba.njit
def _render_setup(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
) -> tuple[
    lib.FloatArray2D,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered and sets up an empty image
    array.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        x and y coordinates of the localizations to be rendered (1D
        arrays).
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).

    Returns
    -------
    image : lib.FloatArray2D
        Empty image array.
    n_pixel_y : int
        Number of pixels in y.
    n_pixel_x : int
        Number of pixels in x.
    x : lib.FloatArray1D
        x coordinates to be rendered.
    y : lib.FloatArray1D
        y coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _render_setup_anisotropic(  # used in Average
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling_x: float,
    oversampling_y: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
) -> tuple[
    lib.FloatArray2D,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered and sets up an empty image
    array. Allows for different pixel sizes in x and y (oversampling).

    Parameters
    ----------
    x, y : lib.FloatArray1D
        x and y coordinates of the localizations to be rendered (1D
        arrays).
    oversampling_x, oversampling_y : float
        Number of super-resolution pixels per camera pixel in x and y.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).

    Returns
    -------
    image : lib.FloatArray2D
        Empty image array.
    n_pixel_y : int
        Number of pixels in y.
    n_pixel_x : int
        Number of pixels in x.
    x : lib.FloatArray1D
        x coordinates to be rendered.
    y : lib.FloatArray1D
        y coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling_y * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling_x * (x_max - x_min)))
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling_x * (x - x_min)
    y = oversampling_y * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _render_setup3d(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[
    lib.FloatArray3D,
    int,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered in 3D and sets up an empty image
    array.

    Parameters
    ----------
    x, y, z : lib.FloatArray1D
        x, y and z coordinates of the localizations to be rendered (1D
        arrays).
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size, used for converting z coordinates.

    Returns
    -------
    image : lib.FloatArray3D
        Empty image array.
    n_pixel_y, n_pixel_x, n_pixel_z : int
        Number of pixels in y, x, and z.
    x, y, z : lib.FloatArray1D
        x, y, z coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    n_pixel_z = int(np.ceil(oversampling * (z_max - z_min)))
    z /= pixelsize
    in_view = (
        (x > x_min)
        & (y > y_min)
        & (z > z_min)
        & (x < x_max)
        & (y < y_max)
        & (z < z_max)
    )
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    z = oversampling * (z - z_min)
    image = np.zeros((n_pixel_y, n_pixel_x, n_pixel_z), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view


@numba.njit
def _render_setup3d_anisotropic(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling_x: float,
    oversampling_y: float,
    oversampling_z: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[
    lib.FloatArray3D,
    int,
    int,
    int,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.FloatArray1D,
    lib.BoolArray1D,
]:
    """Find coordinates to be rendered in 3D and sets up an empty image
    array. Allows for different pixel sizes in x, y and z
    (oversampling).

    Parameters
    ----------
    x, y, z : lib.FloatArray1D
        x, y and z coordinates of the localizations to be rendered (1D
        arrays).
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size, used for converting z coordinates.

    Returns
    -------
    image : lib.FloatArray3D
        Empty image array.
    n_pixel_y, n_pixel_x, n_pixel_z : int
        Number of pixels in y, x, and z.
    x, y, z : lib.FloatArray1D
        x, y, z coordinates to be rendered.
    in_view : lib.BoolArray1D
        Indeces of the localizations to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling_y * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling_x * (x_max - x_min)))
    n_pixel_z = int(np.ceil(oversampling_z * (z_max - z_min)))
    z /= pixelsize
    in_view = (
        (x > x_min)
        & (y > y_min)
        & (z > z_min)
        & (x < x_max)
        & (y < y_max)
        & (z < z_max)
    )
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling_x * (x - x_min)
    y = oversampling_y * (y - y_min)
    z = oversampling_z * (z - z_min)
    image = np.zeros((n_pixel_y, n_pixel_x, n_pixel_z), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view


@numba.njit
def _fill(
    image: lib.FloatArray2D, x: lib.FloatArray1D, y: lib.FloatArray1D
) -> None:
    """Fill image with x and y coordinates. Image is not blurred.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        x and y coordinates to be rendered.
    """
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@numba.njit
def _fill3d(
    image: lib.FloatArray3D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
) -> None:
    """Fill image with x, y and z coordinates. Image is not blurred.

    Parameters
    ----------
    image : lib.FloatArray3D
        Empty image array.
    x, y, z : lib.FloatArray1D
        x, y and z coordinates to be rendered.
    """
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    z = z.astype(np.int32)
    z += np.min(z)  # because z takes also negative values
    for i, j, k in zip(x, y, z):
        image[j, i, k] += 1


@numba.njit
def _fill_gaussian(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Fill image with blurred x and y coordinates. Each localization
    is rendered as a 2D Gaussian centered at (x, y) with standard
    deviations (sx, sy).

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        x and y coordinates to be rendered.
    sx, sy : lib.FloatArray1D
        Localization precision in x and y for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    """
    # render each localization separately
    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):

        # get min and max indeces to draw the given localization
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        # draw a localization as a 2D guassian PDF
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * np.pi * sx_ * sy_)


@numba.njit
def _fill_gaussian_rot(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    sz: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
    ang: tuple[float, float, float],
) -> None:
    """Fill image with rotated gaussian-blurred localizations.

    Localization precisions (sx, sy and sz) are treated as standard
    deviations of the gaussians to be rendered.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y, z : lib.FloatArray1D
        3D coordinates to be rendered.
    sx, sy, sz : lib.FloatArray1D
        Localization precision in x, y and z for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    ang : tuple
        Rotation angles of locs around x, y and z axes (radians).
    """
    (angx, angy, angz) = ang

    rot_mat_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angx), np.sin(angx)],
            [0.0, -np.sin(angx), np.cos(angx)],
        ],
        dtype=np.float32,
    )
    rot_mat_y = np.array(
        [
            [np.cos(angy), 0.0, np.sin(angy)],
            [0.0, 1.0, 0.0],
            [-np.sin(angy), 0.0, np.cos(angy)],
        ],
        dtype=np.float32,
    )
    rot_mat_z = np.array(
        [
            [np.cos(angz), -np.sin(angz), 0.0],
            [np.sin(angz), np.cos(angz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_matrix = rot_mat_x @ rot_mat_y @ rot_mat_z
    rot_matrixT = np.transpose(rot_matrix)

    for x_, y_, z_, sx_, sy_, sz_ in zip(x, y, z, sx, sy, sz):

        # rotated 3D covariance
        cov = np.array(
            [
                [sx_**2, 0.0, 0.0],
                [0.0, sy_**2, 0.0],
                [0.0, 0.0, sz_**2],
            ],
            dtype=np.float32,
        )
        cov_rot = rot_matrix @ cov @ rot_matrixT

        # we only need the top-left 2x2 part for rendering
        s00 = cov_rot[0, 0]  # var_x
        s01 = cov_rot[0, 1]  # cov_xy
        s10 = cov_rot[1, 0]  # cov_yx (= s01)
        s11 = cov_rot[1, 1]  # var_y

        # inverse of 2x2 matrix
        det2d = s00 * s11 - s01 * s10
        if det2d < 1e-10:
            continue
        inv00 = s11 / det2d
        inv01 = -s01 / det2d
        inv10 = -s10 / det2d
        inv11 = s00 / det2d

        norm = 1.0 / (2.0 * np.pi * np.sqrt(det2d))

        # use the larger effective sigma for draw bounds
        eff_sx = np.sqrt(s00)
        eff_sy = np.sqrt(s11)
        max_x = _DRAW_MAX_SIGMA * 2.5 * eff_sx
        max_y = _DRAW_MAX_SIGMA * 2.5 * eff_sy

        j_min = int(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = int(x_ + max_x + 1)
        if j_max > n_pixel_x:
            j_max = n_pixel_x
        i_min = int(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = int(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y

        # 2D Gaussian (no z-loop!)
        for i in range(i_min, i_max):
            b = np.float32(i + 0.5 - y_)
            for j in range(j_min, j_max):
                a = np.float32(j + 0.5 - x_)
                exponent = (
                    a * a * inv00 + a * b * (inv01 + inv10) + b * b * inv11
                )
                image[i, j] += norm * np.exp(-0.5 * exponent)


@numba.njit
def inverse_3x3(a: lib.Array3x3) -> lib.Array3x3:
    """Calculate inverse of a 3x3 matrix. This function is faster than
    ``np.linalg.inv``.

    Parameters
    ----------
    a : lib.Array3x3
        3x3 matrix.

    Returns
    -------
    c : lib.Array3x3
        Inverse of ``a``.
    """
    c = np.zeros((3, 3), dtype=np.float32)
    det = determinant_3x3(a)

    c[0, 0] = (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) / det
    c[0, 1] = (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) / det
    c[0, 2] = (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) / det

    c[1, 0] = (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) / det
    c[1, 1] = (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) / det
    c[1, 2] = (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) / det

    c[2, 0] = (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]) / det
    c[2, 1] = (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) / det
    c[2, 2] = (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) / det

    return c


@numba.njit
def determinant_3x3(a: lib.Array3x3) -> np.float32:
    """Calculate determinant of a 3x3 matrix. This function is faster
    than ``np.linalg.det``.

    Parameters
    ----------
    a : lib.Array3x3
        3x3 matrix.

    Returns
    -------
    det : np.float32
        Determinant of ``a``.
    """
    det = np.float32(
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[2, 0] * a[1, 2])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[2, 0] * a[1, 1])
    )
    return det


@numba.jit(nopython=True, nogil=True)
def render_hist_numba(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    t_min: float,
    t_max: float,
) -> tuple[int, lib.FloatArray2D]:
    """Calculate 2D histogram of xy coordinates. Similar to
    ``render_hist`` but modified to work with numba.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        1D arrays of xy coordinates.
    oversampling : float
        Number of histogram pixels per camera pixel.
    t_min, t_max : float
        Minimum and maximum bounds of the histogram.

    Returns
    -------
    n : int
        Number of localizations in the histogram.
    image : lib.FloatArray2D
        2D histogram of xy coordinates.
    """
    n_pixel = int(np.ceil(oversampling * (t_max - t_min)))
    in_view = (x > t_min) & (y > t_min) & (x < t_max) & (y < t_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - t_min)
    y = oversampling * (y - t_min)
    image = np.zeros((n_pixel, n_pixel), dtype=np.float32)
    _fill(image, x, y)
    return len(x), image


def render_hist(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Alias for _render_hist which will be a private function in
    v0.11.0. Kept for backward compatibility but will be removed in
    v0.11.0. Use _render_hist instead if necessary."""
    lib.deprecation_warning(
        "Deprecation warning: the 'render_hist' function is deprecated "
        "and will be removed in v0.11.0. Use _render_hist instead if "
        "necessary."
    )
    return _render_hist(
        locs, oversampling, y_min, x_min, y_max, x_max, ang=ang
    )


def _render_hist(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with no blur by assigning them to pixels.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels)
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes in radians. If
        None, locs are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs["x"].to_numpy(),
        locs["y"].to_numpy(),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    if ang:
        x, y, _, _ = locs_rotation(
            locs,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
    _fill(image, x, y)
    n = len(x)
    return n, image


@numba.jit(nopython=True, nogil=True)
def render_hist3d(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[int, lib.FloatArray3D]:
    """Render localizations in 3D with no blur by assigning them to
    pixels.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size in nm, used for converting z coordinates.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray3D
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = _render_setup3d(
        x,
        y,
        z,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        z_min,
        z_max,
        pixelsize,
    )
    _fill3d(image, x, y, z)
    n = len(x)
    z *= pixelsize  # convert back to nm
    return n, image


@numba.jit(nopython=True, nogil=True)
def render_hist3d_anisotropic(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling_x: float,
    oversampling_y: float,
    oversampling_z: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[int, lib.FloatArray3D]:
    """Render localizations in 3D with no blur by assigning them to
    pixels. Allows for different pixel sizes in x, y and z
    (oversampling).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling_x, oversampling_y, oversampling_z : float (default=1)
        Number of super-resolution pixels per camera pixel in x, y, and
        z directions.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size in nm, used for converting z coordinates.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray3D
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = (
        _render_setup3d_anisotropic(
            x,
            y,
            z,
            oversampling_x,
            oversampling_y,
            oversampling_z,
            y_min,
            x_min,
            y_max,
            x_max,
            z_min,
            z_max,
            pixelsize,
        )
    )
    _fill3d(image, x, y, z)
    n = len(x)
    z *= pixelsize  # convert back to nm
    return n, image


def render_gaussian(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Alias for _render_gaussian which will be a private function in
    v0.11.0. Kept for backward compatibility but will be removed in v0.11.0. Use
    _render_gaussian instead if necessary."""
    lib.deprecation_warning(
        "Deprecation warning: the 'render_gaussian' function is deprecated "
        "and will be removed in v0.11.0. Use _render_gaussian instead if "
        "necessary."
    )
    return _render_gaussian(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        min_blur_width,
        ang=ang,
    )


def _render_gaussian(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with with individual localization precision
    which differs in x and y.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, y_max : float
        Minimum and maximum y coordinates to be rendered (camera pixels).
    x_min, x_max : float
        Minimum and maximum x coordinates to be rendered (camera pixels).
    min_blur_width : float
        Minimum localization precision (camera pixels).
    ang : tuple, optional
        Rotation angles of localizations around x, y and z axes in
        radians. If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs["x"].to_numpy(),
        locs["y"].to_numpy(),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if not ang:  # not rotated
        blur_width = oversampling * np.maximum(
            locs["lpx"].to_numpy(), min_blur_width
        )
        blur_height = oversampling * np.maximum(
            locs["lpy"].to_numpy(), min_blur_width
        )
        sy = blur_height[in_view]
        sx = blur_width[in_view]

        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = locs_rotation(
            locs,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(
            locs["lpx"].to_numpy(), min_blur_width
        )
        blur_height = oversampling * np.maximum(
            locs["lpy"].to_numpy(), min_blur_width
        )
        # if lpz not found, make it twice the mean of lpx and lpy
        if "lpz" in locs:
            lpz = locs["lpz"].to_numpy()
        else:
            lpz = 2 * locs[["lpx", "lpy"]].to_numpy().mean(axis=1)
        blur_depth = oversampling * np.maximum(lpz, min_blur_width)

        sy = blur_height[in_view]
        sx = blur_width[in_view]
        sz = blur_depth[in_view]

        _fill_gaussian_rot(
            image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang
        )

    n = len(x)
    return n, image


def render_gaussian_iso(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Alias for _render_gaussian_iso which will be a private function in
    v0.11.0. Kept for backward compatibility but will be removed in v0.11.0. Use
    _render_gaussian_iso instead if necessary."""
    lib.deprecation_warning(
        "Deprecation warning: the 'render_gaussian_iso' function is "
        "deprecated and will be removed in v0.11.0. Use "
        "_render_gaussian_iso instead if necessary."
    )
    return _render_gaussian_iso(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        min_blur_width,
        ang=ang,
    )


def _render_gaussian_iso(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Same as ``_render_gaussian``, but uses the same localization
    precision in x and y."""
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs["x"].to_numpy(),
        locs["y"].to_numpy(),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if not ang:  # not rotated
        blur_width = oversampling * np.maximum(
            locs["lpx"].to_numpy(), min_blur_width
        )
        blur_height = oversampling * np.maximum(
            locs["lpy"].to_numpy(), min_blur_width
        )
        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy

        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = locs_rotation(
            locs,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(
            locs["lpx"].to_numpy(), min_blur_width
        )
        blur_height = oversampling * np.maximum(
            locs["lpy"].to_numpy(), min_blur_width
        )
        # for now, let lpz be twice the mean of lpx and lpy
        if "lpz" in locs:
            lpz = locs["lpz"].to_numpy()
        else:
            lpz = 2 * locs[["lpx", "lpy"]].to_numpy().mean(axis=1)
        blur_depth = oversampling * np.maximum(lpz, min_blur_width)

        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
        sz = blur_depth[in_view]

        _fill_gaussian_rot(
            image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang
        )

    return len(x), image


def render_convolve(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Alias for _render_convolve which will be a private function in v0.11.0.
    Kept for backward compatibility but will be removed in v0.11.0. Use
    _render_convolve instead if necessary."""
    lib.deprecation_warning(
        "Deprecation warning: the 'render_convolve' function is "
        "deprecated and will be removed in v0.11.0. Use "
        "_render_convolve instead if necessary."
    )
    return _render_convolve(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        min_blur_width,
        ang=ang,
    )


def _render_convolve(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with with global localization precision,
    i.e. each localization is blurred by the median localization
    precision in x and y.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    min_blur_width : float
        Minimum localization precision (camera pixels).
    ang : tuple, optional
        Rotation angles of localizations around x, y and z axes in
        radians. If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs["x"].to_numpy(),
        locs["y"].to_numpy(),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    if ang:  # rotate
        x, y, in_view, _ = locs_rotation(
            locs,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        blur_width = oversampling * max(
            np.median(locs["lpx"].to_numpy()[in_view]), min_blur_width
        )
        blur_height = oversampling * max(
            np.median(locs["lpy"].to_numpy()[in_view]), min_blur_width
        )
        return n, _fftconvolve(image, blur_width, blur_height)


def render_smooth(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Alias for _render_smooth which will be a private function in v0.11.0. Kept for
    backward compatibility but will be removed in v0.11.0. Use _render_smooth
    instead if necessary."""
    lib.deprecation_warning(
        "Deprecation warning: the 'render_smooth' function is deprecated and "
        "will be removed in v0.11.0. Use _render_smooth instead if necessary."
    )
    return _render_smooth(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        ang=ang,
    )


def _render_smooth(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with with blur of one display pixel (set by
    oversampling).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    ang : tuple, optional
        Rotation angles of localizations around x, y and z axes in
        radians. If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs["x"].to_numpy(),
        locs["y"].to_numpy(),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang:
        x, y, _, _ = locs_rotation(
            locs,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(
    image: lib.FloatArray2D,
    blur_width: float,
    blur_height: float,
) -> lib.FloatArray2D:
    """Blur (convolves) 2D image using fast fourier transform.

    Parameters
    ----------
    image : lib.FloatArray2D
        Image with rendered but not blurred localizations.
    blur_width, blur_height : float
        Blur width and height in pixels.

    Returns
    -------
    image : lib.FloatArray2D
        Blurred image.
    """
    kernel_width = 10 * int(np.round(blur_width)) + 1
    kernel_height = 10 * int(np.round(blur_height)) + 1
    kernel_y = signal.windows.gaussian(kernel_height, blur_height)
    kernel_x = signal.windows.gaussian(kernel_width, blur_width)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    image = signal.fftconvolve(image, kernel, mode="same")
    return image.astype(np.float32)


def rotation_matrix(angx: float, angy: float, angz: float) -> Rotation:
    """Find rotation matrix given rotation angles around axes.

    Parameters
    ----------
    angx, angy, angz : float
        Rotation angles around x, y and z axes in radians.

    Returns
    -------
    scipy.spatial.transform.Rotation
        Scipy class that can be applied to rotate an Nx3 np.ndarray
    """
    rot_mat_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angx), np.sin(angx)],
            [0.0, -np.sin(angx), np.cos(angx)],
        ]
    )  # rotation matrix around x axis
    rot_mat_y = np.array(
        [
            [np.cos(angy), 0.0, np.sin(angy)],
            [0.0, 1.0, 0.0],
            [-np.sin(angy), 0.0, np.cos(angy)],
        ]
    )  # rotation matrix around y axis
    rot_mat_z = np.array(
        [
            [np.cos(angz), -np.sin(angz), 0.0],
            [np.sin(angz), np.cos(angz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )  # rotation matrix around z axis
    rot_mat = rot_mat_x @ rot_mat_y @ rot_mat_z
    return Rotation.from_matrix(rot_mat)


def locs_rotation(
    locs: pd.DataFrame,
    oversampling: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ang: tuple[float, float, float],
) -> tuple[
    lib.FloatArray1D, lib.FloatArray1D, lib.BoolArray1D, lib.FloatArray1D
]:
    """Rotate localizations within a FOV.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rotated.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    ang : tuple
        Rotation angles of localizations around x, y and z axes in
        radians.

    Returns
    -------
    x : lib.FloatArray1D
        New (rotated) x coordinates
    y : lib.FloatArray1D
        New y coordinates
    in_view : lib.BoolArray1D
        Indeces of locs that are rendered
    z : lib.FloatArray1D
        New z coordinates
    """
    # z is translated to pixels
    locs_coord = locs[["x", "y", "z"]].to_numpy()

    # x and y are in range (x_min/y_min, x_max/y_max) so they need to be
    # shifted (scipy rotation is around origin)
    locs_coord[:, 0] -= x_min + (x_max - x_min) / 2
    locs_coord[:, 1] -= y_min + (y_max - y_min) / 2

    # rotate locs
    R = rotation_matrix(ang[0], ang[1], ang[2])
    locs_coord = R.apply(locs_coord)

    # unshift locs
    locs_coord[:, 0] += x_min + (x_max - x_min) / 2
    locs_coord[:, 1] += y_min + (y_max - y_min) / 2

    # output
    x = locs_coord[:, 0]
    y = locs_coord[:, 1]
    z = locs_coord[:, 2]
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    z *= oversampling
    return x, y, in_view, z


def export_qimage_to_pdf(
    image: QtGui.QImage, path: str, dpi: int = 96
) -> None:
    writer = QtGui.QPdfWriter(path)

    # Fixed physical page size (1 image pixel = 1/96 inch, regardless of dpi)
    width_mm = image.width() * 25.4 / 96
    height_mm = image.height() * 25.4 / 96

    page_size = QtGui.QPageSize(
        QtCore.QSizeF(width_mm, height_mm),
        QtGui.QPageSize.Unit.Millimeter,
    )
    writer.setPageSize(page_size)
    writer.setResolution(dpi)

    # Painter coordinates: 1 unit = 1/dpi inch, so full page =
    # (width_mm / 25.4) * dpi = image.width() * dpi / 96
    draw_width = image.width() * dpi / 96
    draw_height = image.height() * dpi / 96

    painter = QtGui.QPainter(writer)
    painter.drawImage(QtCore.QRectF(0, 0, draw_width, draw_height), image)
    painter.end()


def export_qimage_to_svg(image: QtGui.QImage, path: str):
    generator = QtSvg.QSvgGenerator()
    generator.setFileName(path)
    generator.setSize(image.size())
    generator.setViewBox(QtCore.QRect(0, 0, image.width(), image.height()))

    painter = QtGui.QPainter(generator)
    painter.drawImage(0, 0, image)
    painter.end()


def get_colors_from_colormap(
    n_channels: int,
    cmap: str = "gist_rainbow",
) -> list[tuple[int, int, int]]:
    """Create a list with rgb channels for each of the channels used in
    rendering property using the gist_rainbow colormap, see:
    https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Parameters
    ----------
    n_channels : int
        Number of locs channels.
    cmap : str, optional
        Colormap name. Default is 'gist_rainbow'.

    Returns
    -------
    colors : list of tuples
        Contains tuples with RGB channels ranging between 0 and 255.
    """
    # array of shape (256, 3) with RGB channels with 256 colors
    base = plt.get_cmap(cmap)(np.arange(256))[:, :3]
    # indeces to draw from base
    idx = np.linspace(0, 255, n_channels).astype(int)
    # extract the colors of interest
    colors = base[idx]
    return colors


def get_group_color(locs: pd.DataFrame) -> lib.IntArray1D:
    """Find group color for each localization in single channel data
    with group info.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.

    Returns
    -------
    colors : lib.IntArray1D
        Array with integer group color index for each localization.
    """
    colors = locs["group"].to_numpy().astype(int) % N_GROUP_COLORS
    return colors


def viewport_height(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> float:
    """Calculate viewport height in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    height : float
        Viewport height in camera pixels.
    """
    return viewport[1][0] - viewport[0][0]


def viewport_width(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> float:
    """Calculate viewport width in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    width : float
        Viewport width in camera pixels.
    """
    return viewport[1][1] - viewport[0][1]


def viewport_size(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Calculate viewport size in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    height, width : float
        Viewport height and width in camera pixels.
    """
    height = viewport_height(viewport)
    width = viewport_width(viewport)
    return height, width


def adjust_viewport_to_aspect_ratio(
    image: QtGui.QImage,
    viewport: list[tuple[float, float], tuple[float, float]],
) -> list[tuple[float, float], tuple[float, float]]:
    """Adjust viewport to match the aspect ratio of the image.

    Parameters
    ----------
    image : QtGui.QImage
        Image of rendered localizations.
    viewport : list of tuples
        Viewport coordinates in camera pixels, ((y_min, y_max), (x_min,
        x_max)).

    Returns
    -------
    viewport : list of tuples
        Adjusted viewport coordinates in camera pixels, ((y_min, y_max),
        (x_min, x_max)).
    """
    viewport_height, viewport_width = viewport_size(viewport)
    view_height = image.height()
    view_width = image.width()
    viewport_aspect = viewport_width / viewport_height
    view_aspect = view_width / view_height
    if view_aspect >= viewport_aspect:
        y_min = viewport[0][0]
        y_max = viewport[1][0]
        x_range = viewport_height * view_aspect
        x_margin = (x_range - viewport_width) / 2
        x_min = viewport[0][1] - x_margin
        x_max = viewport[1][1] + x_margin
    else:
        x_min = viewport[0][1]
        x_max = viewport[1][1]
        y_range = viewport_width / view_aspect
        y_margin = (y_range - viewport_height) / 2
        y_min = viewport[0][0] - y_margin
        y_max = viewport[1][0] + y_margin
    return ((y_min, x_min), (y_max, x_max))


def adjust_viewport_decorator(func):
    """Decorator that adjusts viewport to match image aspect ratio before
    calling the decorated function.

    Parameters
    ----------
    func : callable
        Function that takes `image` and `viewport` as arguments.

    Returns
    -------
    wrapper : callable
        Wrapped function with automatic viewport adjustment.
    """

    def wrapper(image, viewport, *args, **kwargs):
        adjusted_viewport = adjust_viewport_to_aspect_ratio(image, viewport)
        if adjusted_viewport != viewport:
            print("Adjusted viewport to match image aspect ratio.")
        return func(image, adjusted_viewport, *args, **kwargs)

    return wrapper


def map_to_view(
    x: float,
    y: float,
    image_size: QtCore.QSize,
    viewport: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[int, int]:
    """Convert (x, y) from camera pixels to display pixels."""
    image_width = image_size.width()
    image_height = image_size.height()
    cx = image_width * (x - viewport[0][1]) / viewport_width(viewport)
    cy = image_height * (y - viewport[0][0]) / viewport_height(viewport)
    return int(cx), int(cy)


def get_rectangle_pick_polygon(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float,
    return_most_right: bool = False,
) -> QtGui.QPolygonF | tuple[float, float]:
    """Find QtGui.QPolygonF object used for drawing a rectangular
    pick.

    Returns
    -------
    p : QtGui.QPolygonF
        The polygon.
    """
    X, Y = lib.get_pick_rectangle_corners(
        start_x, start_y, end_x, end_y, width
    )
    p = QtGui.QPolygonF()
    for x, y in zip(X, Y):
        p.append(QtCore.QPointF(x, y))
    if return_most_right:
        ix_most_right = np.argmax(X)
        x_most_right = X[ix_most_right]
        y_most_right = Y[ix_most_right]
        return p, (x_most_right, y_most_right)
    return p


def _draw_picks_circle(
    image: QtGui.QImage,
    viewport: list[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # pick coords in camera pixels
    pick_size: float,  # diameter in camera pixels
    point_picks: bool = False,
    annotate_picks: bool = False,
    color: QtGui.QColor = QtGui.QColor("yellow"),
) -> QtGui.QImage:
    """Draw circular picks onto the image of rendered localizations.
    See ``draw_picks`` for more details."""
    if point_picks:  # draw circular picks as points
        painter = QtGui.QPainter(image)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(color)
        for i, pick in enumerate(picks):
            # convert from camera units to display units
            cx, cy = map_to_view(*pick, image.size(), viewport)
            painter.drawEllipse(QtCore.QPoint(cx, cy), 3, 3)
            if annotate_picks:
                painter.drawText(cx + 20, cy + 20, str(i))

    else:  # draw circles
        d = int(pick_size * image.width() / viewport_width(viewport))
        painter = QtGui.QPainter(image)
        painter.setPen(color)
        for i, pick in enumerate(picks):
            # check that the pick is within the view
            if (
                pick[0] < viewport[0][1]
                or pick[0] > viewport[1][1]
                or pick[1] < viewport[0][0]
                or pick[1] > viewport[1][0]
            ):
                continue

            # convert from camera units to display units
            cx, cy = map_to_view(*pick, image.size(), viewport)
            painter.drawEllipse(int(cx - d / 2), int(cy - d / 2), d, d)
            if annotate_picks:
                painter.drawText(int(cx + d / 2), int(cy + d / 2), str(i))
    painter.end()
    return image


def _draw_picks_rectangle(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    pick_size: float,  # width in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor = QtGui.QColor("yellow"),
) -> QtGui.QImage:
    """Draw rectangular picks onto the image of rendered
    localizations. See ``draw_picks`` for more details."""
    w = pick_size * image.width() / viewport_width(viewport)
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        # convert from camera units to display units
        start_x, start_y = map_to_view(*pick[0], image.size(), viewport)
        end_x, end_y = map_to_view(*pick[1], image.size(), viewport)
        # draw a straight line across the pick
        painter.drawLine(start_x, start_y, end_x, end_y)
        # draw a rectangle
        polygon, most_right = get_rectangle_pick_polygon(
            start_x, start_y, end_x, end_y, w, return_most_right=True
        )
        painter.drawPolygon(polygon)
        if annotate_picks:
            painter.drawText(*most_right, str(i))
    painter.end()
    return image


def _draw_picks_polygon(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor = QtGui.QColor("yellow"),
) -> QtGui.QImage:
    """Draw polygon picks onto the image of rendered localizations. See
    ``draw_picks`` for more details."""
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        oldpoint = []
        for point in pick:
            cx, cy = map_to_view(*point, image.size(), viewport)
            painter.drawEllipse(
                QtCore.QPoint(cx, cy),
                int(POLYGON_POINTER_SIZE / 2),
                int(POLYGON_POINTER_SIZE / 2),
            )
            if oldpoint != []:  # draw the line
                ox, oy = map_to_view(*oldpoint, image.size(), viewport)
                painter.drawLine(cx, cy, ox, oy)
            oldpoint = point

        # annotate picks
        if len(pick) and annotate_picks:
            painter.drawText(
                cx + int(POLYGON_POINTER_SIZE / 2) + 10,
                cy + int(POLYGON_POINTER_SIZE / 2) + 10,
                str(i),
            )
    painter.end()
    return image


def _draw_picks_square(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    pick_size: float,  # side length in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor = QtGui.QColor("yellow"),
) -> QtGui.QImage:
    """Draw square picks onto the image of rendered localizations."""
    w = int(pick_size * image.width() / viewport_width(viewport))
    painter = QtGui.QPainter(image)
    painter.setPen(color)
    for i, pick in enumerate(picks):
        # check that the pick is within the view
        if (
            pick[0] < viewport[0][1]
            or pick[0] > viewport[1][1]
            or pick[1] < viewport[0][0]
            or pick[1] > viewport[1][0]
        ):
            continue

        # convert from camera units to display units
        cx, cy = map_to_view(*pick, image.size(), viewport)
        painter.drawRect(int(cx - w / 2), int(cy - w / 2), w, w)

        # annotate picks
        if annotate_picks:
            painter.drawText(
                int(cx + w / 2) + 10, int(cy + w / 2) + 10, str(i)
            )
    painter.end()
    return image


@adjust_viewport_decorator
def draw_picks(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    pick_shape: Literal["Circle", "Rectangle", "Polygon", "Square"],
    picks: list[tuple],  # pick coords in camera pixels
    pick_size: float | None,  # diameter in camera pixels
    point_picks: bool = False,
    annotate_picks: bool = False,
    color: QtGui.QColor = QtGui.QColor("yellow"),
) -> QtGui.QImage:
    """Draw all selected picks onto the image (QImage) of rendered
    localizations.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    pick_shape: {"Circle", "Rectangle", "Polygon", "Square"}
        Shape of the picks to be drawn.
    picks: list of tuples
        List of picks, where each pick is a tuple specifying the pick
        coordinates. Note: this must match the format of the given pick
        shape.
    pick_size : float or None
        Size of the picks in camera pixels. For "Circle", this is the
        diameter; for "Rectangle", this is the width; for "Square", this
        is the side length. This parameter is ignored for "Polygon"
        picks.
    point_picks : bool, optional
        If True and pick_shape is "Circle", draw picks as points instead
        of circles. Default is False.
    annotate_picks : bool, optional
        If True, annotate each pick with its index in the picks list.
        Default is False.
    color : QtGui.QColor, optional
        Color of the picks. Default is yellow.

    Returns
    -------
    image : QImage
        Image with the drawn picks.
    """
    image = image.copy()
    if pick_shape == "Circle":
        return _draw_picks_circle(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            point_picks=point_picks,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Rectangle":
        return _draw_picks_rectangle(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Polygon":
        return _draw_picks_polygon(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
        )
    elif pick_shape == "Square":
        return _draw_picks_square(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            annotate_picks=annotate_picks,
            color=color,
        )


@adjust_viewport_decorator
def draw_points(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    points: list[tuple],  # points in camera pixels,
    pixelsize: int | float,  # camera pixel size in nm
    color: QtGui.QColor = QtGui.QColor("yellow"),
) -> QtGui.QImage:
    """Draw points, lines and distances between them onto image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    points : list of tuples
        List of points, where each point is a tuple specifying the point
        coordinates in camera pixels.
    pixelsize : int or float
        Camera pixel size in nm.
    color : QtGui.QColor, optional
        Color of the points, lines and text. Default is yellow.

    Returns
    -------
    image : QImage
        Image with the drawn points.
    """
    d = 20  # width of the drawn crosses (display pixels)
    painter = QtGui.QPainter(image)
    painter.setPen(color)

    cx = []
    cy = []
    ox = []  # together with oldpoint used for drawing
    oy = []  # lines between points
    oldpoint = []
    for point in points:
        # convert to display units
        if oldpoint != []:
            ox, oy = map_to_view(*oldpoint, image.size(), viewport=viewport)
        cx, cy = map_to_view(*point, image.size(), viewport=viewport)

        # draw a cross
        painter.drawPoint(cx, cy)
        painter.drawLine(cx, cy, int(cx + d / 2), cy)
        painter.drawLine(cx, cy, cx, int(cy + d / 2))
        painter.drawLine(cx, cy, int(cx - d / 2), cy)
        painter.drawLine(cx, cy, cx, int(cy - d / 2))

        # draw a line between points and show distance
        if oldpoint != []:
            painter.drawLine(cx, cy, ox, oy)
            font = painter.font()
            font.setPixelSize(20)
            painter.setFont(font)

            # get distance with 2 decimal places
            distance = (
                float(
                    int(
                        np.sqrt(
                            (
                                (oldpoint[0] - point[0]) ** 2
                                + (oldpoint[1] - point[1]) ** 2
                            )
                        )
                        * pixelsize
                        * 100
                    )
                )
                / 100
            )
            painter.drawText(
                int((cx + ox) / 2 + d),
                int((cy + oy) / 2 + d),
                str(distance) + " nm",
            )
        oldpoint = point
    painter.end()
    return image


@adjust_viewport_decorator
def draw_scalebar(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    scalebar_length_nm: int | float,  # scalebar length in nm
    pixelsize: int | float,  # camera pixel size in nm
    display_length: bool = True,  # whether to display scalebar length in nm
    color: QtGui.QColor = QtGui.QColor("white"),
) -> QtGui.QImage:
    """Draw a scalebar into rendered localizations (QImage).

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    scalebar_length_nm : int or float
        Scale bar length in nm.
    pixelsize : int or float
        Camera pixel size in nm.

    Returns
    -------
    image : QImage
        Image with the drawn scalebar.
    """
    length_camerapxl = scalebar_length_nm / pixelsize
    length_displaypxl = int(
        round(image.width() * length_camerapxl / viewport_width(viewport))
    )
    height = 10  # display pixels
    painter = QtGui.QPainter(image)
    painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    painter.setBrush(QtGui.QBrush(color))

    # draw a rectangle
    x = image.width() - length_displaypxl - 35
    y = image.height() - height - 20
    painter.drawRect(x, y, length_displaypxl + 0, height + 0)

    # display scalebar's length
    if display_length:
        font = painter.font()
        font.setPixelSize(20)
        painter.setFont(font)
        painter.setPen(color)
        text_spacer = 40
        text_width = length_displaypxl + 2 * text_spacer
        text_height = text_spacer
        painter.drawText(
            x - text_spacer,
            y - 25,
            text_width,
            text_height,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
            str(scalebar_length_nm) + " nm",
        )
    return image


def draw_legend(
    image: QtGui.QImage,
    channel_names: list[str],
    channel_colors: list[tuple[int, int, int]],
) -> QtGui.QImage:
    """Draw a legend for multichannel data in the top left corner over
    rendered localizations (QImage).

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    channel_names : list of str
        List of channel names to be displayed in the legend.
    channel_colors : list of tuples
        List of RGB tuples corresponding to the colors of the channels.
        Must range between 0 and 255.

    Returns
    -------
    image : QImage
        Image with the drawn legend.
    """
    assert len(channel_names) == len(channel_colors), (
        "Length of channel_names must match number of channels in " "dataset."
    )
    n_channels = len(channel_names)
    painter = QtGui.QPainter(image)
    # initial positions
    x = 12
    y = 26
    dy = 24  # space between names
    padding = 4  # padding around text
    font = painter.font()
    font.setPixelSize(16)
    painter.setFont(font)
    fm = QtGui.QFontMetrics(font)
    for i in range(n_channels):
        text = channel_names[i]
        # draw black background
        text_rect = fm.boundingRect(text)
        bg_rect = QtCore.QRect(
            x - padding,
            y - fm.ascent() - padding,
            text_rect.width() + 2 * padding,
            fm.height() + 2 * padding,
        )
        painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.black))
        painter.drawRect(bg_rect)
        # draw colored text
        color_rgb = channel_colors[i]
        color = QtGui.QColor(color_rgb[0], color_rgb[1], color_rgb[2])
        painter.setPen(QtGui.QPen(color))
        painter.drawText(QtCore.QPoint(x, y), text)
        y += dy
    return image


@adjust_viewport_decorator
def draw_minimap(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    max_viewport_size: tuple[float, float],  # in camera pixels,
    color_main: QtGui.QColor = QtGui.QColor("yellow"),
    color_frame: QtGui.QColor = QtGui.QColor("white"),
) -> QtGui.QImage:
    """Draw a minimap showing the position of current viewport.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    max_viewport_size : tuple
        Maximum viewport size in camera pixels, (max_height, max_width).
    color_main, color_frame : QColor, optional
        Colors of the viewport and the minimap frame. Default is yellow
        and white, respectively.

    Returns
    -------
    image : QImage
        Image with the drawn minimap.
    """
    movie_height, movie_width = max_viewport_size
    length_minimap = 100
    height_minimap = int(movie_height / movie_width * 100)
    # draw in the upper right corner, overview rectangle
    x = image.width() - length_minimap - 20
    y = 20
    painter = QtGui.QPainter(image)
    painter.setPen(color_frame)
    painter.drawRect(x, y, length_minimap + 0, height_minimap + 0)
    painter.setPen(color_main)
    length = int(viewport_width(viewport) / movie_width * length_minimap)
    length = max(5, length)
    height = int(viewport_height(viewport) / movie_height * height_minimap)
    height = max(5, height)
    x_vp = int(viewport[0][1] / movie_width * length_minimap)
    y_vp = int(viewport[0][0] / movie_height * length_minimap)
    painter.drawRect(x + x_vp, y + y_vp, length + 0, height + 0)
    return image


def render_scene(
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | None = None,
    autoscale: bool = False,
    single_channel_colormap: str = "magma",
    colors: list | None = None,
    return_qimage: bool = False,
) -> lib.IntArray3D | QtGui.QImage:
    """Render localizations into a colored image (either QImage or a 
    numpy array).
    
    Parameters
    ----------
    locs: pd.DataFrame or list of pd.DataFrame
        Localizations to be rendered. Can be either one localization
        file or a list thereof.
    info: list of dict or list of list of dict
        List of info dictionaries corresponding to the localization
        file(s).
    disp_px_size : float
        Display pixel size in nm.
    viewport : tuple, optional
        Field of view to be rendered (in camera pixels). The input is
        ``((y_min, x_min), (y_max, x_max))``. If None, all localizations
        are rendered.
    blur_method : {"gaussian", "gaussian_iso", "smooth", "convolve"} or None, \
            optional
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, no
        blurring is applied. 'gaussian' uses localization precisions
        of each localization to blur it (different in each dimension).
        'gaussian_iso' is similar but averages x and y localization
        precisions, so that blur is isotropic. 'smooth' applies a one
        pixel blur. 'convolve' applies the same blur to all
        localizations which is the median localization precision.
    min_blur_width : float, optional
        Minimum size of blur (camera pixels).
    ang : tuple, optional
        Rotation angles of locs around x, y and z axes in radians. If
        None, locs are not rotated.
    autoscale : bool, optional
        True if optimally adjust contrast. Default is False.
    single_channel_colormap : str, optional
        Colormap to use for single channel data. Default is 'magma'.
    colors : list of tuples, optional
        List of RGB tuples corresponding to the colors of the channels.
        Only needs to be specified for multi-channel data. Must range
        between 0 and 255. Default is None.  #TODO: see if this is true
    return_qimage: bool, optional
        If True, return a QImage. If False, return a numpy array.
        Default is False.

    Returns
    -------
    image : IntArray3D or QImage
        RGB image of rendered localizations. Either a numpy array of
        shape (height, width, 3) with integer values between 0 and 255
        or a QImage (if return_qimage is True).
    """
    if isinstance(locs, pd.DataFrame):
        image = _render_single_channel(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            ang=ang,
            autoscale=autoscale,
            single_channel_colormap=single_channel_colormap,
        )
    elif (
        isinstance(locs, list)
        and len(locs) == 1
        and "group" not in locs[0].columns
    ):
        image = _render_single_channel(
            locs=locs[0],
            info=info[0],
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            ang=ang,
            autoscale=autoscale,
            single_channel_colormap=single_channel_colormap,
        )
    else:
        assert len(colors) == len(locs), (
            f"Mismatch between {len(colors)} colors and {len(locs)} "
            "localization files."
        )
        image = _render_multi_channel(  # TODO: render multi channel could call render_Singel_channel if one chanenl present without group column
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            ang=ang,
            autoscale=autoscale,
            colors=colors,
        )
    if return_qimage:
        bgra = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        bgra[:, :, 0] = image[:, :, 2]  # R -> B
        bgra[:, :, 1] = image[:, :, 1]  # G -> G
        bgra[:, :, 2] = image[:, :, 0]  # B -> R
        bgra[:, :, 3] = 255  # A -> 255 (opaque)
        Y, X = image.shape[:2]
        qimage = QtGui.QImage(
            bgra.data, X, Y, QtGui.QImage.Format.Format_RGB32
        )
        return qimage
    else:
        return image
