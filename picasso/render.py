"""
picasso.render
~~~~~~~~~~~~~~

Render single molecule localizations to a super-resolution image.

Provides functions for painting onto rendered images (QImage), such as
scale bar and picks.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from typing import Literal, Callable

import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy import signal, ndimage
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from PyQt6 import QtGui, QtCore, QtSvg

from . import io, lib, __version__


_DRAW_MAX_SIGMA = 3  # max. sigma from mean to render (mu +/- 3 sigma)
N_GROUP_COLORS = 8
POLYGON_POINTER_SIZE = 16  # must be even


def render(
    locs: pd.DataFrame,
    info: dict,
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations given FOV and blur method.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    info : dict
        Contains localizations metadata.
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
    oversampling = pixelsize / disp_px_size

    if viewport is None:
        height = lib.get_from_metadata(info, "Height", raise_error=True)
        width = lib.get_from_metadata(info, "Width", raise_error=True)
        viewport = [(0, 0), (height, width)]

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


@numba.njit(cache=True)
def _draw_gaussian_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Render a single separable 2D Gaussian into ``image``."""
    max_y_off = _DRAW_MAX_SIGMA * sy_
    i_min = np.int32(y_ - max_y_off)
    if i_min < 0:
        i_min = 0
    i_max = np.int32(y_ + max_y_off + 1)
    if i_max > n_pixel_y:
        i_max = n_pixel_y
    max_x_off = _DRAW_MAX_SIGMA * sx_
    j_min = np.int32(x_ - max_x_off)
    if j_min < 0:
        j_min = 0
    j_max = np.int32(x_ + max_x_off) + 1
    if j_max > n_pixel_x:
        j_max = n_pixel_x
    nx = j_max - j_min
    ny = i_max - i_min
    if nx <= 0 or ny <= 0:
        return
    inv_2sx2 = 1.0 / (2.0 * sx_ * sx_)
    inv_2sy2 = 1.0 / (2.0 * sy_ * sy_)
    norm = 1.0 / (2.0 * np.pi * sx_ * sy_)
    # Separable kernel: factor exp(-(dx^2/(2sx^2) + dy^2/(2sy^2)))
    # into 1D gx * 1D gy. O(K) exp calls per loc instead of O(K^2).
    gx = np.empty(nx, dtype=np.float32)
    gy = np.empty(ny, dtype=np.float32)
    for jj in range(nx):
        dx = (j_min + jj) + 0.5 - x_
        gx[jj] = np.exp(-dx * dx * inv_2sx2)
    for ii in range(ny):
        dy = (i_min + ii) + 0.5 - y_
        gy[ii] = norm * np.exp(-dy * dy * inv_2sy2)
    for ii in range(ny):
        gy_i = gy[ii]
        row = image[i_min + ii]
        for jj in range(nx):
            row[j_min + jj] += gy_i * gx[jj]


@numba.njit(cache=True)
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
    n_locs = len(x)
    if n_locs == 0:
        return

    for i in range(n_locs):
        _draw_gaussian_loc(
            image, x[i], y[i], sx[i], sy[i], n_pixel_x, n_pixel_y
        )


@numba.njit(cache=True)
def _draw_gaussian_rot_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    sz_: float,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
    rot_matrixT: lib.Array3x3,
) -> None:
    """Render a single rotated 2D Gaussian (projected from 3D) into
    ``image``."""
    cov = np.zeros((3, 3), dtype=np.float32)
    cov[0, 0] = sx_ * sx_
    cov[1, 1] = sy_ * sy_
    cov[2, 2] = sz_ * sz_
    cov_rot = rot_matrix @ cov @ rot_matrixT
    s00 = cov_rot[0, 0]
    s01 = cov_rot[0, 1]
    s10 = cov_rot[1, 0]
    s11 = cov_rot[1, 1]
    det2d = s00 * s11 - s01 * s10
    if det2d < 1e-10:
        return
    inv00 = s11 / det2d
    inv01 = -s01 / det2d
    inv10 = -s10 / det2d
    inv11 = s00 / det2d
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det2d))
    max_x_off = _DRAW_MAX_SIGMA * np.sqrt(s00)
    max_y_off = _DRAW_MAX_SIGMA * np.sqrt(s11)
    j_min = int(x_ - max_x_off)
    if j_min < 0:
        j_min = 0
    j_max = int(x_ + max_x_off + 1)
    if j_max > n_pixel_x:
        j_max = n_pixel_x
    i_min = int(y_ - max_y_off)
    if i_min < 0:
        i_min = 0
    i_max = int(y_ + max_y_off + 1)
    if i_max > n_pixel_y:
        i_max = n_pixel_y
    for i in range(i_min, i_max):
        b = np.float32(i + 0.5 - y_)
        for j in range(j_min, j_max):
            a = np.float32(j + 0.5 - x_)
            exponent = a * a * inv00 + a * b * (inv01 + inv10) + b * b * inv11
            image[i, j] += norm * np.exp(-0.5 * exponent)


@numba.njit(cache=True)
def _fill_gaussian_rot(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
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
    n_locs = len(x)
    if n_locs == 0:
        return
    angx, angy, angz = ang
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
    rot_matrix = (rot_mat_x @ rot_mat_y @ rot_mat_z).astype(np.float32)
    rot_matrixT = np.ascontiguousarray(rot_matrix.T)

    for i in range(n_locs):
        _draw_gaussian_rot_loc(
            image,
            x[i],
            y[i],
            sx[i],
            sy[i],
            sz[i],
            n_pixel_x,
            n_pixel_y,
            rot_matrix,
            rot_matrixT,
        )


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
    ``_render_hist`` but modified to work with numba.

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
    if ang is not None:
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

    if ang is None:  # not rotated
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

        _fill_gaussian_rot(image, x, y, sx, sy, sz, n_pixel_x, n_pixel_y, ang)

    n = len(x)
    return n, image


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

    if ang is None:  # not rotated
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

        _fill_gaussian_rot(image, x, y, sx, sy, sz, n_pixel_x, n_pixel_y, ang)

    return len(x), image


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
    if ang is not None:  # rotate
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

    if ang is not None:  # rotate
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
    """Blur (convolves) 2D image using fast fourier transform or with
    Gaussian filter applied (faster for small kernels).

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
    # Spatial separable convolution is faster than FFT for the small
    # kernels typical of SMLM precisions (~1-3 px). Switch to FFT only
    # when the kernel is large relative to the image.
    n_y, n_x = image.shape
    spatial = (
        kernel_height < 0.05 * n_y
        and kernel_width < 0.05 * n_x
        and max(kernel_height, kernel_width) <= 101
    )
    if spatial:
        out = np.empty_like(image, dtype=np.float32)
        ndimage.gaussian_filter(
            image,
            sigma=(blur_height, blur_width),
            output=out,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )
        return out
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
        Scipy class that can be applied to rotate an Nx3 array.
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


def solid_to_lut(rgb: tuple[float, float, float]) -> lib.FloatArray2D:
    """Build a (256, 3) float32 LUT that linearly ramps from black to
    the given RGB color.

    The returned LUT is the input format expected by
    :func:`_render_multi_channel` (and therefore :func:`render_scene`)
    when colors are passed as per-channel lookup tables. A solid-color
    channel rendered through this LUT is mathematically identical to
    the legacy ``intensity * rgb`` blend.

    Parameters
    ----------
    rgb : sequence of 3 floats
        Target RGB color, each component in range [0, 1].

    Returns
    -------
    lut : lib.FloatArray2D
        LUT with generated colormap of shape (256, 3).

    Examples
    --------
    >>> lut = solid_to_lut((1.0, 0.0, 0.0))   # black -> red
    >>> render_scene(locs=..., info=..., colors=[lut, ...], ...)
    """
    rgb_arr = np.asarray(rgb, dtype=np.float32).reshape(3)
    return np.linspace(
        np.zeros(3, dtype=np.float32), rgb_arr, 256, dtype=np.float32
    )


def stops_to_lut(
    stops: list[tuple[float, float, float, float]],
) -> lib.FloatArray2D:
    """Build a (256, 3) float32 LUT by linearly interpolating between
    color stops.

    Parameters
    ----------
    stops : sequence of (position, r, g, b) tuples
        Each ``position`` must be in [0, 1], strictly increasing, with
        the first stop at 0.0 and the last at 1.0. ``r``, ``g``, ``b``
        are also in [0, 1].

    Returns
    -------
    lut : lib.FloatArray2D
        LUT with generated colormap of shape (256, 3).

    Examples
    --------
    A 3-stop "fire" gradient (black -> red -> yellow):

    >>> lut = stops_to_lut([
    ...     (0.0, 0, 0, 0),
    ...     (0.5, 1, 0, 0),
    ...     (1.0, 1, 1, 0),
    ... ])
    >>> render_scene(locs=..., info=..., colors=[lut], ...)
    """
    arr = np.asarray(stops, dtype=np.float32)
    positions = arr[:, 0]
    rgb = arr[:, 1:4]
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(x, positions, rgb[:, c])
    return lut


def get_colors_from_colormap(
    n_channels: int,
    cmap: str = "gist_rainbow",
) -> list[tuple[float, float, float]]:
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
        Contains tuples with RGB channels ranging between 0 and 1.
    """
    # array of shape (256, 3) with RGB channels with 256 colors
    base = plt.get_cmap(cmap)(np.arange(256))[:, :3]
    # indeces to draw from base
    idx = np.linspace(0, 255, n_channels).astype(int)
    # extract the colors of interest
    colors = base[idx]
    return colors  # value ranging between 0 and 1


def get_group_color(
    locs: pd.DataFrame,
    shuffle: bool = False,
) -> lib.IntArray1D:
    """Find group color for each localization in single channel data
    with group info.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations. Must contain a ``group`` column.
    shuffle : bool, optional
        If True, build a lookup of ``np.arange(max(group) + 1)``,
        randomly permute it, and take it mod ``N_GROUP_COLORS`` before
        indexing by ``group``. This scatters adjacent group ids across
        color slots. Default is False (plain ``group % N_GROUP_COLORS``).

    Returns
    -------
    colors : lib.IntArray1D
        Array with integer group color index for each localization.
    """
    groups = locs["group"].to_numpy().astype(int)
    if shuffle:
        lookup = np.arange(groups.max() + 1)
        np.random.shuffle(lookup)
        lookup %= N_GROUP_COLORS
        return lookup[groups]
    return groups % N_GROUP_COLORS


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


def viewport_center(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Calculate viewport center in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    center : tuple
        Viewport center coordinates in camera pixels (y, x).
    """
    center = (
        ((viewport[1][0] + viewport[0][0]) / 2),
        ((viewport[1][1] + viewport[0][1]) / 2),
    )
    return center


def shift_viewport(
    viewport: tuple[tuple[float, float], tuple[float, float]],
    dx: float,
    dy: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shift the viewport by the given shift vector (toward the bottom
    right corner).

    Parameters
    ----------
    viewport : tuple
        Current viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    dx, dy : float
        Shifts in camera pixels.

    Returns
    -------
    new_viewport : tuple
        New viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    """
    (ymin, xmin), (ymax, xmax) = viewport
    new_viewport = ((ymin + dy, xmin + dx), (ymax + dy, xmax + dx))
    return new_viewport


def zoom_viewport(
    viewport: tuple[tuple[float, float], tuple[float, float]],
    factor: float,
    cursor_position: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Zoom the viewport by the given factor.

    Parameters
    ----------
    viewport : tuple
        Current viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    factor : float
        Zoom factor. Values > 1 will zoom in, values < 1 will zoom out.
    cursor_position : tuple, optional
        Cursor's position on the screen. If None, zooming is centered
        around viewport's center. Default is None.

    Returns
    -------
    new_viewport : tuple
        New viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    """
    viewport_height, viewport_width = viewport_size(viewport)
    new_viewport_height = viewport_height * factor
    new_viewport_width = viewport_width * factor

    if cursor_position is not None:  # wheelEvent
        old_viewport_center = viewport_center(viewport)
        rel_pos_x = (
            cursor_position[0] - old_viewport_center[1]
        ) / viewport_width
        rel_pos_y = (
            cursor_position[1] - old_viewport_center[0]
        ) / viewport_height
        new_viewport_center_x = (
            cursor_position[0] - rel_pos_x * new_viewport_width
        )
        new_viewport_center_y = (
            cursor_position[1] - rel_pos_y * new_viewport_height
        )
    else:
        new_viewport_center_y, new_viewport_center_x = viewport_center(
            viewport
        )

    new_viewport = [
        (
            new_viewport_center_y - new_viewport_height / 2,
            new_viewport_center_x - new_viewport_width / 2,
        ),
        (
            new_viewport_center_y + new_viewport_height / 2,
            new_viewport_center_x + new_viewport_width / 2,
        ),
    ]
    return new_viewport


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

    Note that this assumes image and viewport to be the first two
    arguments of the decorated function

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
    mark_width: int = 20,  # width of the drawn crosses in display pixels
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
    mark_width : int, optional
        Width of the drawn crosses in display pixels. Default is 20.

    Returns
    -------
    image : QImage
        Image with the drawn points.
    """
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
        painter.drawLine(cx, cy, int(cx + mark_width / 2), cy)
        painter.drawLine(cx, cy, cx, int(cy + mark_width / 2))
        painter.drawLine(cx, cy, int(cx - mark_width / 2), cy)
        painter.drawLine(cx, cy, cx, int(cy - mark_width / 2))

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
                int((cx + ox) / 2 + mark_width),
                int((cy + oy) / 2 + mark_width),
                str(distance) + " nm",
            )
        oldpoint = point
    painter.end()
    return image


@adjust_viewport_decorator
def draw_scalebar(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    scalebar_length_nm: int | float,
    pixelsize: int | float,
    display_length: bool = True,
    color: QtGui.QColor = QtGui.QColor("white"),
    display_height: int = 10,
    margin: tuple[int, int] = (35, 20),
    text_spacer: int = 40,
    text_fontsize: int = 20,
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
    color : QColor, optional
        Color of the scalebar and text. Default is white.
    display_length : bool, optional
        Whether to display scalebar length in nm. Default is True.
    margin : tuple of int, optional
        Margins from the right and bottom edges in display pixels.
        Default is (35, 20).
    text_spacer : int, optional
        Spacing between the scalebar and the displayed length text in
        display pixels. Only used if display_length is True. Default is
        40.
    text_fontsize : int, optional
        Font size of the displayed length text in display pixels. Only
        used if display_length is True. Default is 20.

    Returns
    -------
    image : QImage
        Image with the drawn scalebar.
    """
    painter = QtGui.QPainter(image)
    painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    painter.setBrush(QtGui.QBrush(color))

    length_camerapxl = scalebar_length_nm / pixelsize
    length_displaypxl = int(
        round(image.width() * length_camerapxl / viewport_width(viewport))
    )

    # draw a rectangle
    x = image.width() - length_displaypxl - margin[0]
    y = image.height() - display_height - margin[1]
    painter.drawRect(x, y, length_displaypxl, display_height)

    # display scalebar's length
    if display_length:
        font = painter.font()
        font.setPixelSize(text_fontsize)
        painter.setFont(font)
        painter.setPen(color)
        text_width = length_displaypxl + 2 * text_spacer
        text_height = text_spacer
        painter.drawText(
            x - text_spacer,
            y - 25,
            text_width,
            text_height,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
            f"{str(scalebar_length_nm)} nm",
        )
    return image


def draw_legend(
    image: QtGui.QImage,
    channel_names: list[str],
    channel_colors: list[tuple[int, int, int]],
    init_pos: tuple[int, int] = (12, 26),
    dy: int = 24,
    padding: int = 4,
    text_fontsize: int = 16,
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
    init_pos : tuple of int, optional
        Initial position (x, y) of the first channel name in display
        pixels. Default is (12, 26).
    dy : int, optional
        Space between channel names in display pixels. Default is 24.
    padding : int, optional
        Padding around the text in display pixels. Default is 4.
    text_fontsize : int, optional
        Font size of the channel names in display pixels. Default is 16.

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
    x, y = init_pos
    font = painter.font()
    font.setPixelSize(text_fontsize)
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
    length_minimap: int = 100,
    margin: tuple[int, int] = (20, 20),
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
        Maximum viewport size in camera pixels, i.e., the acquired
        movie size (height, width).
    color_main, color_frame : QColor, optional
        Colors of the viewport and the minimap frame. Default is yellow
        and white, respectively.
    length_minimap : int, optional
        Length of the minimap in pixels. Default is 100.
    margin : tuple of int, optional
        Margins from the right and top edges in display pixels.
        Default is (20, 20).

    Returns
    -------
    image : QImage
        Image with the drawn minimap.
    """
    movie_height, movie_width = max_viewport_size
    height_minimap = int(movie_height / movie_width * length_minimap)
    # draw in the upper right corner, overview rectangle
    x = image.width() - length_minimap - margin[0]
    y = margin[1]
    painter = QtGui.QPainter(image)
    painter.setPen(color_frame)
    painter.drawRect(x, y, length_minimap, height_minimap)
    painter.setPen(color_main)
    length = int(viewport_width(viewport) / movie_width * length_minimap)
    length = max(5, length)
    height = int(viewport_height(viewport) / movie_height * height_minimap)
    height = max(5, height)
    x_vp = int(viewport[0][1] / movie_width * length_minimap)
    y_vp = int(viewport[0][0] / movie_height * height_minimap)
    painter.drawRect(x + x_vp, y + y_vp, length, height)
    return image


def draw_rotation(
    image: QtGui.QImage,
    ang: tuple[float, float, float],
    axis_length: int = 30,
    axis_center: tuple[int, int] = (50, -50),  # bottom left
) -> QtGui.QImage:
    """Draw rotation axes icon on the image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    ang : tuple of float
        Rotation angles around x, y, and z axes in radians.
    axis_length : int, optional
        Length of the rotation axes in display pixels. Default is 30.
    axis_center : tuple of int, optional
        Position of the rotation axes icon in display pixels, with
        origin in the top left corner. Negative values indicated
        counting from the bottom right corner. Default is (50, -50).

    Returns
    -------
    image : QImage
        Image with the drawn rotation axes icon.
    """
    painter = QtGui.QPainter(image)
    x = (
        axis_center[0]
        if axis_center[0] >= 0
        else image.width() + axis_center[0]
    )
    y = (
        axis_center[1]
        if axis_center[1] >= 0
        else image.height() + axis_center[1]
    )
    center = QtCore.QPoint(x, y)

    # set the ends of the x line
    xx = axis_length
    xy = 0
    xz = 0

    # set the ends of the y line
    yx = 0
    yy = axis_length
    yz = 0

    # set the ends of the z line
    zx = 0
    zy = 0
    zz = axis_length

    # rotate these points
    coordinates = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    R = rotation_matrix(*ang)
    coordinates = R.apply(coordinates).astype(int)
    (xx, xy, xz) = coordinates[0]
    (yx, yy, yz) = coordinates[1]
    (zx, zy, zz) = coordinates[2]

    # translate the x and y coordinates of the end points towards
    # bottom right edge of the window
    xx += x
    xy += y
    yx += x
    yy += y
    zx += x
    zy += y

    # set the points at the ends of the lines
    point_x = QtCore.QPoint(xx, xy)
    point_y = QtCore.QPoint(yx, yy)
    point_z = QtCore.QPoint(zx, zy)
    line_x = QtCore.QLine(center, point_x)
    line_y = QtCore.QLine(center, point_y)
    line_z = QtCore.QLine(center, point_z)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(1, 0, 0, 1)))
    painter.drawLine(line_x)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 1, 1)))
    painter.drawLine(line_y)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 0, 1)))
    painter.drawLine(line_z)
    return image


def draw_rotation_angles(
    image: QtGui.QImage,
    ang: tuple[float, float, float],
    color: QtGui.QColor = QtGui.QColor("white"),
) -> QtGui.QImage:
    """Draw rotation angles (numbers in degrees) on the image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    ang : tuple of float
        Rotation angles around x, y, and z axes in radians.
    color : QColor, optional
        Color of the text. Default is white.

    Returns
    -------
    image : QImage
        Image with the drawn rotation angles.
    """
    angx, angy, angz = [int(np.round(_ * 180 / np.pi, 0)) for _ in ang]
    text = f"{angx} {angy} {angz}"
    x = image.width() - len(text) * 8 - 10
    y = image.height() - 20
    painter = QtGui.QPainter(image)
    font = painter.font()
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(QtCore.QPoint(x, y), text)
    return image


def render_scene(
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    *,
    disp_px_size: float = 100.0,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | None = None,
    contrast: tuple[float, float] | None = None,
    invert_colors: bool = False,
    single_channel_colormap: str | lib.FloatArray2D = "magma",
    colors: list | None = None,
    relative_intensities: list[float] | None = None,
    raw_image_cache: lib.FloatArray2D | lib.FloatArray3D | None = None,
    return_contrast_limits: bool = False,
    return_raw_image: bool = False,
) -> (
    tuple[QtGui.QImage, int]
    | tuple[QtGui.QImage, int, tuple[float, float]]
    | tuple[QtGui.QImage, int, lib.FloatArray2D | lib.FloatArray3D]
    | tuple[
        QtGui.QImage,
        int,
        tuple[float, float],
        lib.FloatArray2D | lib.FloatArray3D,
    ]
):
    """Render localizations into a colored image (either QImage or a 
    numpy array).

    For single channel images without group info, the colormap is
    specified by `single_channel_colormap`. For single channel images
    with group info, the colormap is determined by `get_group_color` and
    `lib.get_colors`. For multi-channel images, the colors are specified
    by `colors`.

    If `raw_image_cache` is provided (the raw grayscale image of
    localizations, i.e., obtained with ``render.render``; 2D array for
    single-channel data, 3D array for multi-channel data), some of the
    arguments are not used: `locs`, `info`, `disp_px_size`, `viewport`,
    `blur_method`, `min_blur_width`, `ang`.

    Optionally, the user can request the raw grayscale image of
    localizations and/or the contrast limits used for scaling to be
    returned together with the rendered QImage and number of
    localizations rendered.

    Parameters
    ----------
    locs: pd.DataFrame or list of pd.DataFrame
        Localizations to be rendered. Can be either one localization
        file or a list thereof. If a single DataFrame is provided,
        localizations will be rendered in a single channel, i.e., using
        a color map specified by `single_channel_colormap`. If a list of
        DataFrames is provided, localizations will be rendered in
        multiple channels, and the color of each channel can be
        specified by `colors`.
    info: list of dict or list of list of dict
        List of info dictionaries corresponding to the localization
        file(s).
    disp_px_size : float, optional
        Display pixel size in nm. Default is 100.0.
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
    contrast : tuple of float, optional
        Contrast limits for scaling. If None, contrast is automatically
        determined.
    invert_colors : bool, optional
        If True, invert colors of the rendered image. Default is False.
    single_channel_colormap : str | lib.FloatArray2D, optional
        Colormap to use for single channel data. If a str, the
        corresponding pyplot colormap is selected. If a 2D array, a
        256x4  array is expected with values between 0 and 1. Default is
        'magma'.
    colors : list of tuples, optional
        List of RGB tuples corresponding to the colors of the channels.
        Only needs to be specified for multi-channel data. Must range
        between 0 and 1. Default is None.
    relative_intensities : list of float, optional
        List of relative intensities for each channel. Only needs to be
        specified for multi-channel data. Default is None, in which
        case all channels are rendered with the same intensity.
    raw_image_cache: lib.FloatArray2D or lib.FloatArray3D, optional
        If provided, this raw grayscale image of localizations, i.e.,
        obtained with ``render.render`` (2D array for single-channel
        data, 3D array for multi-channel data) is used instead of
        recomputing it. Some of the arguments are not used if this is
        provided: `locs`, `info`, `disp_px_size`, `viewport`,
        `blur_method`, `min_blur_width`, `ang`.
    return_contrast_limits : bool, optional
        If True, return the contrast limits used for scaling. Default is
        False.
    return_raw_image : bool, optional
        If True, return the raw grayscale image of localizations (2D
        array for single-channel data, 3D array for multi-channel data).
        Default is False.

    Returns
    -------
    qimage : QtGui.QImage
        RGB image of rendered localizations as a QImage object.
    n_locs : int
        Total number of localizations rendered.
    contrast_limits : tuple of float, optional
        The contrast limits used for scaling. Only returned if
        return_contrast_limits is True.
    raw_image : FloatArray2D or FloatArray3D, optional
        Raw grayscale image of localizations (2D array for single-channel
        data, 3D array for multi-channel data). Only returned if
        return_raw_image is True.
    """
    if isinstance(locs, pd.DataFrame):
        n_locs, rgb, contrast_limits, raw_image = _render_single_channel(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            ang=ang,
            contrast=contrast,
            invert_colors=invert_colors,
            single_channel_colormap=single_channel_colormap,
            raw_image_cache=raw_image_cache,
        )
    elif len(locs) == 0:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        n_locs = 0
        contrast_limits = contrast if contrast is not None else (0.0, 1.0)
        raw_image = np.zeros((1, 1), dtype=np.float32)
    else:
        if colors is not None:
            assert len(colors) == len(locs) == len(info), (
                f"Mismatch between {len(colors)} colors, {len(locs)} "
                f"localization files, and {len(info)} info dictionaries."
            )
        else:
            assert len(locs) == len(info), (
                f"Mismatch between {len(locs)} localization files and "
                f"{len(info)} info dictionaries."
            )
        n_locs, rgb, contrast_limits, raw_image = _render_multi_channel(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            colors=colors,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            ang=ang,
            contrast=contrast,
            relative_intensities=relative_intensities,
            invert_colors=invert_colors,
            raw_image_cache=raw_image_cache,
        )
    qimage = rgb_to_qimage(rgb)
    if return_raw_image and return_contrast_limits:
        return qimage, n_locs, contrast_limits, raw_image
    elif return_raw_image:
        return qimage, n_locs, raw_image
    elif return_contrast_limits:
        return qimage, n_locs, contrast_limits
    else:
        return qimage, n_locs


def _render_multi_channel(
    locs: list[pd.DataFrame],
    info: list[list[dict]],
    *,
    disp_px_size: float,
    colors: list[tuple[int, int, int]] | list[np.ndarray],
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | None = None,
    contrast: tuple[float, float] | None = None,
    relative_intensities: list[float] | None = None,
    invert_colors: bool = False,
    raw_image_cache: lib.FloatArray3D | None = None,
) -> tuple[int, lib.IntArray3D, tuple[float, float], lib.FloatArray3D]:
    """Render multi-channel localizations into an RGB 8bit image
    (numpy array). See ``render_scene`` for more details.

    ``colors`` may be either a list of ``(r, g, b)`` triplets (legacy
    behaviour: each channel rendered as ``intensity × rgb``, additive
    blend) or a list of ``(256, 3)`` LUTs (each channel indexed into
    its LUT before additive blending — supports per-channel
    matplotlib colormaps and user-defined colormaps from the GUI).
    """
    if raw_image_cache is not None:
        assert raw_image_cache.ndim == 3, "raw_image_cache must be a 3D array."
        raw_image = raw_image_cache
        n_locs = 0
    else:
        renderings = [  # monochromatic images of localizations
            render(
                locs=locs[i],
                info=info[i],
                disp_px_size=disp_px_size,
                viewport=viewport,
                blur_method=blur_method,
                min_blur_width=min_blur_width,
                ang=ang,
            )
            for i in range(len(locs))
        ]
        n_locs = sum([rendering[0] for rendering in renderings])
        raw_image = np.array([rendering[1] for rendering in renderings])

    # scale contrast and intensities
    vmin, vmax = contrast if contrast is not None else (None, None)
    autoscale = True if contrast is None else False
    images, contrast_limits = scale_contrast(
        raw_image, vmin, vmax, autoscale=autoscale, return_contrast_limits=True
    )
    images = scale_intensities(
        images, relative_intensities=relative_intensities
    )

    # color the images
    if colors is None:  # fallback if the user did not specify colors
        colors = lib.get_colors(len(images))
    colors_arr = np.asarray(colors, dtype=np.float32)
    images_f32 = np.ascontiguousarray(images, dtype=np.float32)
    if colors_arr.ndim == 2:
        # legacy path: each channel is a single (r, g, b)
        rgb = np.tensordot(images_f32, colors_arr, axes=([0], [0]))
    else:
        # LUT path: each channel is a (256, 3) lookup table
        idx = np.clip((images_f32 * 255.0).astype(np.int32), 0, 255)
        rgb = np.zeros(
            (images_f32.shape[1], images_f32.shape[2], 3), dtype=np.float32
        )
        for c in range(images_f32.shape[0]):
            rgb += colors_arr[c][idx[c]]
    # clip to max value of 1 (preserves relative brightness)
    np.minimum(rgb, 1.0, out=rgb)
    rgb = to_8bit(rgb)
    if invert_colors:
        rgb = 255 - rgb
    return n_locs, rgb, contrast_limits, raw_image


def _render_single_channel(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | None = None,
    contrast: tuple[float, float] | None = None,
    invert_colors: bool = False,
    single_channel_colormap: str = "magma",
    raw_image_cache: lib.FloatArray2D | None = None,
) -> tuple[int, lib.IntArray3D, tuple[float, float], lib.FloatArray2D]:
    """Render single-channel localizations into an RGB 8bit image (numpy
    array). See ``render_scene`` for more details."""
    if raw_image_cache is not None:
        assert raw_image_cache.ndim == 2, "raw_image_cache must be a 2D array."
        raw_image = raw_image_cache
        n_locs = 0
    else:
        n_locs, raw_image = render(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            ang=ang,
        )
    vmin, vmax = contrast if contrast is not None else (None, None)
    autoscale = True if contrast is None else False
    image, contrast_limits = scale_contrast(
        raw_image, vmin, vmax, autoscale=autoscale, return_contrast_limits=True
    )
    image = to_8bit(image)
    rgb = apply_colormap(image, single_channel_colormap)
    if invert_colors:
        rgb = 255 - rgb
    return n_locs, rgb, contrast_limits, raw_image


def rgb_to_qimage(
    image: lib.IntArray3D, return_bgra: bool = False
) -> QtGui.QImage | tuple[QtGui.QImage, lib.IntArray3D]:
    """Convert a numpy array of shape (height, width, 3) with integer
    values between 0 and 255 to a QImage.

    Parameters
    ----------
    image : IntArray3D
        RGB image as a numpy array of shape (height, width, 3) with
        integer values between 0 and 255.
    return_bgra : bool, optional
        If True, return the BGRA numpy array instead of a QImage.
        Default is False.

    Returns
    -------
    qimage : QImage
        The converted QImage.
    bgra : IntArray3D
        The BGRA numpy array. Only returned if return_bgra is True.
    """
    bgra = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    bgra[:, :, 0] = image[:, :, 2]  # R -> B
    bgra[:, :, 1] = image[:, :, 1]  # G -> G
    bgra[:, :, 2] = image[:, :, 0]  # B -> R
    bgra[:, :, 3] = 255  # A -> 255 (opaque)
    Y, X = image.shape[:2]
    qimage = QtGui.QImage(bgra.data, X, Y, QtGui.QImage.Format.Format_RGB32)
    qimage = qimage.copy()  # make a deep copy to own the data DO NOT DELETE
    if return_bgra:
        return qimage, bgra
    return qimage


def scale_contrast(
    image: lib.FloatArray2D | lib.FloatArray3D,
    vmin: float | None = None,
    vmax: float | None = None,
    autoscale: bool = False,
    return_contrast_limits: bool = False,
) -> (
    lib.FloatArray2D
    | lib.FloatArray3D
    | tuple[lib.FloatArray2D | lib.FloatArray3D, tuple[float, float]]
):
    """Scale contrast of the image (2D array) or images (3D array)
    according to the given contrast limits or automatically.

    Parameters
    ----------
    image : FloatArray2D or FloatArray3D
        Image (2D array) or images (3D array) to be contrast scaled.
    vmin : float or None, optional
        Minimum contrast limit. If None, the minimum pixel value of the
        image(s) is used. Default is None.
    vmax : float or None, optional
        Maximum contrast limit. If None, the maximum pixel value of the
        image(s) is used. Default is None.
    autoscale : bool, optional
        If True, automatically adjust contrast limits to optimally use
        the full range of pixel values. Default is False.
    return_contrast_limits : bool, optional
        If True, return the contrast limits used for scaling. Default is
        False.

    Returns
    -------
    scaled_images : FloatArray2D or FloatArray3D
        Contrast scaled image(s).
    contrast_limits : tuple of float, optional
        The contrast limits used for scaling. Only returned if
        return_contrast_limits is True.
    """
    if autoscale:
        if image.ndim == 2:
            max_ = image.max()
        else:
            # lowest max value from all channels, given it's not
            # an empty image
            max_ = min([_.max() for _ in image if _.max() > 0])
        vmax = 0.5 * max_
        vmin = 0.0
    vmin = vmin if vmin is not None else image.min()
    vmax = vmax if vmax is not None else image.max()
    if vmin == vmax:
        vmax = vmin + 1e-6
    scaled_image = (image - vmin) / (vmax - vmin)
    scaled_image[~np.isfinite(scaled_image)] = 0.0
    scaled_image = np.clip(scaled_image, 0.0, 1.0)
    if return_contrast_limits:
        return scaled_image, (vmin, vmax)
    return scaled_image


def scale_intensities(
    images: lib.FloatArray3D,
    relative_intensities: list[float] | None = None,
) -> lib.FloatArray3D:
    """Scale intensities across images.

    Parameters
    ----------
    image : FloatArray3D
        Image(s) to be intensity scaled.
    relative_intensities : list of float, optional
        List of relative intensities for each channel. If None, all
        channels are rendered with the same intensity. Default is None.

    Returns
    -------
    scaled_images : FloatArray3D
        Intensity scaled images.
    """
    if relative_intensities is not None:
        assert len(relative_intensities) == images.shape[0], (
            "Length of relative_intensities must match number of channels "
            "in images."
        )
        for i in range(images.shape[0]):
            images[i] *= relative_intensities[i]
    return images


def to_8bit(
    image: lib.FloatArray2D | lib.FloatArray3D,
) -> lib.IntArray2D | lib.IntArray3D:
    """Convert a float image with values between 0 and 1 to an 8-bit image
    with values between 0 and 255."""
    # normalize to max value of 1 and convert to 8-bit
    image /= image.max() if image.max() > 0 else 1.0
    return np.round(image * 255).astype(np.uint8)


def apply_colormap(
    image: lib.IntArray2D, colormap: str | lib.FloatArray2D
) -> lib.IntArray3D:
    """Apply a colormap to a single-channel image (2D array) and return an
    RGB image (3D array).

    Parameters
    ----------
    image : IntArray2D
        Single-channel image as a 2D numpy array with integer values
        between 0 and 255 (8bit).
    colormap : str or FloatArray2D
        If a str, the corresponding pyplot colormap is selected. If a 2D
        array, a 256x4 or 256x3 array is expected with values between 0
        and 1. Note: the alpha channel (if present) is ignored and the
        colormap is applied as if all values were fully opaque.
    """
    if isinstance(colormap, str):
        cmap = np.uint8(np.round(255 * plt.get_cmap(colormap)(np.arange(256))))
    else:
        cmap = np.uint8(np.round(255 * colormap))
    image = cmap[image][:, :, :3]  # drop alpha channel if present
    return image


def split_locs_by_property(
    locs: pd.DataFrame,
    *,
    property_name: str,
    n_colors: int = 32,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[pd.DataFrame]:
    """Split localizations into groups based on a specified property and
    return a list of DataFrames, one for each group.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    property_name : str
        Name of the property to split the localizations by.
    n_colors : int, optional
        Number of color groups to create. Default is 32.
    min_value : float, optional
        Minimum value of the property for scaling. If None, the minimum
        value in the data is used.
    max_value : float, optional
        Maximum value of the property for scaling. If None, the maximum
        value in the data is used.

    Returns
    -------
    locs_groups : list of pd.DataFrame
        Each element corresponds to a group of localizations with
        similar property values.
    """
    assert (
        property_name in locs.columns
    ), f"Property '{property_name}' not found in localizations."
    values = locs[property_name]
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()

    step = (max_value - min_value) / n_colors
    color = np.floor((values - min_value) / step).astype(int)
    color = np.clip(color, 0, n_colors - 1)

    locs_groups = []
    for i in range(n_colors):
        locs_groups.append(locs[color == i])
    return locs_groups


def split_locs_by_group(
    locs: pd.DataFrame,
    n_colors: int = N_GROUP_COLORS,
    group_color: lib.IntArray1D | None = None,
) -> list[pd.DataFrame]:
    """Split localizations into groups based on the 'group' column and
    return a list of DataFrames, one for each group.

    If no 'group' column is present, all localizations are returned as
    single-element list.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    n_colors : int, optional
        Number of color groups to create if 'group' column is not present.
        Default is 8.
    group_color : IntArray1D or None, optional
        If provided, specifies the group color ids (up to `n_colors`)
        for each localization.
    """
    if group_color is not None:
        assert len(group_color) == len(
            locs
        ), "Length of group_color must match number of localizations."
        locs_groups = [locs[group_color == _] for _ in range(n_colors)]
    elif "group" in locs.columns:
        groups = locs["group"].unique()
        locs_groups = [locs[locs["group"] == group] for group in groups]
    else:
        locs_groups = [locs]
    return locs_groups


def optimal_scalebar_length(pixelsize: int | float, width: int | float) -> int:
    """Calculate optimal scale bar length in nm based on the image
    width.

    Parameters
    ----------
    pixelsize : int or float
        Camera pixel size in nm.
    width : int or float
        Image width in camera pixels.

    Returns
    -------
    scalebar : int
        Suggested scale bar length in nm.
    """
    width_nm = width * pixelsize
    optimal_scalebar = width_nm / 8
    # approximate to the nearest thousands, hundreds, tens or ones
    if optimal_scalebar > 10_000:
        scalebar = 10_000
    elif optimal_scalebar > 1_000:
        scalebar = int(1_000 * round(optimal_scalebar / 1_000))
    elif optimal_scalebar > 100:
        scalebar = int(100 * round(optimal_scalebar / 100))
    elif optimal_scalebar > 10:
        scalebar = int(10 * round(optimal_scalebar / 10))
    else:
        scalebar = int(round(optimal_scalebar))
    return scalebar


def _animation_sequence(
    positions: list[list[float, float, float, tuple]],
    durations: list[float],
    fps: int,
) -> tuple[list, list]:
    """Calculate the sequence of angles and viewports for the animation.
    See ``build_animation`` for more details."""
    n_frames = [0]
    for i in range(len(positions) - 1):
        n_frames.append(int(fps * durations[i]))

    # find rotation angles and viewport for each frame
    angles = []
    viewports = []
    for i in range(len(positions) - 1):
        # angles
        x1, y1, z1 = positions[i][:3]
        x2, y2, z2 = positions[i + 1][:3]
        current_angles = np.linspace(
            [x1, y1, z1], [x2, y2, z2], n_frames[i + 1]
        )
        angles.extend(current_angles)

        # viewports
        vp1 = positions[i][3]
        vp2 = positions[i + 1][3]
        ymin = np.linspace(vp1[0][0], vp2[0][0], n_frames[i + 1])
        xmin = np.linspace(vp1[0][1], vp2[0][1], n_frames[i + 1])
        ymax = np.linspace(vp1[1][0], vp2[1][0], n_frames[i + 1])
        xmax = np.linspace(vp1[1][1], vp2[1][1], n_frames[i + 1])
        current_viewports = [
            ((ymin[j], xmin[j]), (ymax[j], xmax[j])) for j in range(len(ymin))
        ]
        viewports.extend(current_viewports)
    return angles, viewports


def build_animation(
    path: str,
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    *,
    positions: list[tuple[float, float, float, tuple]],
    durations: list[float],
    disp_px_size: int | float,  # nm
    image_size: tuple[int, int],
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    contrast: tuple[float, float] | None = None,
    invert_colors: bool = False,
    single_channel_colormap: str | lib.FloatArray2D = "magma",
    colors: list | None = None,
    relative_intensities: list[float] | None = None,
    fps: int = 30,
    adjust_pixel_size: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> None:
    """Build an animation of rendered localizations given the
    checkpoints (angle, viewport, etc) and the time between them.

    Parameters
    ----------
    path : str
        Path to the animation file to be created. Must end with .mp4.
    locs : pd.DataFrame or list of pd.DataFrame
        Localizations to be rendered. Can be either one localization
        file or a list thereof.
    info : list of dict or list of list of dict
        List of info dictionaries corresponding to the localization
        file(s).
    disp_px_size : int or float
        Display pixel size in nm. If 'adjust_pixel_size' is True,
        disp_px_size defines the pixel size in the last frame of the
        animation and will be adjusted if the viewport is zoomed in or
        out such that the number of display pixels remains the same.
        If 'adjust_pixel_size' is False, disp_px_size remains the same
        across the animation
    image_size : tuple of int
        Size of the rendered image in pixels, given as (width, height).
    positions : list
        Each element determines the checkpoint of the animation, which
        is a tuple of 4 elements: (angle_x, angle_y, angle_z, viewport).
        Angles are in radians. Viewport is given as ((y_min, x_min),
        (y_max, x_max)) in camera pixels.
    durations : list
        List of durations in seconds between the checkpoints. Must have
        the same length as positions - 1.
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
    contrast : tuple of float, optional
        Contrast limits for scaling. If None, contrast is automatically
        determined. If given, only the last checkpoint is used to
        determine the contrast limits and the limits will be adjusted
        if the viewport is zoomed in or out.
    invert_colors : bool, optional
        If True, invert colors of the rendered image. Default is False.
    single_channel_colormap : str | lib.FloatArray2D, optional
        Colormap to use for single channel data. If a str, the
        corresponding pyplot colormap is selected. If a 2D array, a
        256x4  array is expected with values between 0 and 1. Default is
        'magma'.
    colors : list of tuples, optional
        List of RGB tuples corresponding to the colors of the channels.
        Only needs to be specified for multi-channel data. Must range
        between 0 and 1. Default is None.
    relative_intensities : list of float, optional
        List of relative intensities for each channel. Only needs to be
        specified for multi-channel data. Default is None, in which
        case all channels are rendered with the same intensity.
    fps : int, optional
        Frames per second of the animation. Default is 30.
    adjust_pixel_size : bool, optional
        If True, adjust disp_px_size on the go such that the number of
        display pixels remains the same if the viewport is zoomed in or
        out. If False, disp_px_size remains the same across the
        animation.
    progress_callback : callable, "console", or None, optional
        If a callable, it is called with the current frame number as an
        argument after each frame is rendered. If "console", a progress
        bar is printed to the console. If None, no progress is reported.
        Default is None.
    """
    assert isinstance(path, str) and path.endswith(
        ".mp4"
    ), "path must be a string ending with '.mp4'."
    assert isinstance(
        locs, (pd.DataFrame, list)
    ), "locs must be a pd.DataFrame or a list of pd.DataFrames."
    if isinstance(locs, list):
        assert all(
            isinstance(locs_, pd.DataFrame) for locs_ in locs
        ), "All elements of locs must be pd.DataFrames."
        assert len(locs) >= 1, "locs must contain at least one DataFrame."
    assert (
        isinstance(info, list) and len(info) >= 1
    ), "info must be a non-empty list."
    assert (
        isinstance(positions, list) and len(positions) >= 2
    ), "positions must be a list with at least 2 elements."
    assert all(len(p) == 4 for p in positions), (
        "Each position must be a tuple/list of 4 elements: "
        "(angle_x, angle_y, angle_z, viewport)."
    )
    assert (
        isinstance(durations, list) and len(durations) == len(positions) - 1
    ), "durations must be a list of length len(positions) - 1."
    assert all(d > 0 for d in durations), "All durations must be positive."
    assert (
        isinstance(disp_px_size, (int, float)) and disp_px_size > 0
    ), "disp_px_size must be a positive number."
    assert (
        isinstance(image_size, (tuple, list))
        and len(image_size) == 2
        and all(isinstance(s, int) and s > 0 for s in image_size)
    ), "image_size must be a tuple of two positive integers (width, height)."
    assert blur_method in (
        "gaussian",
        "gaussian_iso",
        "smooth",
        "convolve",
        None,
    ), (
        "blur_method must be one of 'gaussian', 'gaussian_iso', 'smooth', "
        "'convolve', or None."
    )
    assert (
        isinstance(min_blur_width, (int, float)) and min_blur_width >= 0
    ), "min_blur_width must be a non-negative number."
    if contrast is not None:
        assert (
            isinstance(contrast, (tuple, list))
            and len(contrast) == 2
            and contrast[0] < contrast[1]
        ), "contrast must be a tuple (vmin, vmax) with vmin < vmax."
    assert isinstance(invert_colors, bool), "invert_colors must be a bool."
    if not isinstance(single_channel_colormap, str):
        assert (
            hasattr(single_channel_colormap, "shape")
            and single_channel_colormap.ndim == 2
            and single_channel_colormap.shape[0] == 256
            and single_channel_colormap.shape[1] in (3, 4)
        ), (
            "single_channel_colormap must be a str or a 256x3 / 256x4 "
            "float array with values between 0 and 1."
        )
    if colors is not None:
        n_channels = len(locs) if isinstance(locs, list) else 1
        assert (
            len(colors) == n_channels
        ), "colors must have one entry per channel."
        assert all(
            len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in colors
        ), "Each color must be an RGB tuple with values between 0 and 1."
    if relative_intensities is not None:
        n_channels = len(locs) if isinstance(locs, list) else 1
        assert (
            len(relative_intensities) == n_channels
        ), "relative_intensities must have one entry per channel."
        assert all(
            v >= 0 for v in relative_intensities
        ), "All relative_intensities must be non-negative."
    assert isinstance(fps, int) and fps > 0, "fps must be a positive integer."
    assert isinstance(
        adjust_pixel_size, bool
    ), "adjust_pixel_size must be a bool."
    assert (
        progress_callback is None
        or progress_callback == "console"
        or callable(progress_callback)
    ), "progress_callback must be None, 'console', or a callable."

    _build_animation(
        path=path,
        locs=locs,
        info=info,
        positions=positions,
        durations=durations,
        disp_px_size=disp_px_size,
        image_size=image_size,
        blur_method=blur_method,
        min_blur_width=min_blur_width,
        contrast=contrast,
        invert_colors=invert_colors,
        single_channel_colormap=single_channel_colormap,
        colors=colors,
        relative_intensities=relative_intensities,
        fps=fps,
        adjust_pixel_size=adjust_pixel_size,
        progress_callback=progress_callback,
    )


def _build_animation(
    path: str,
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    positions: list[tuple[float, float, float, tuple]],
    durations: list[float],
    disp_px_size: int | float,
    image_size: tuple[int, int],
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ),
    min_blur_width: float,
    contrast: tuple[float, float] | None,
    invert_colors: bool,
    single_channel_colormap: str | lib.FloatArray2D,
    colors: list | None,
    relative_intensities: list[float] | None,
    fps: int,
    adjust_pixel_size: bool,
    progress_callback: Callable[[int], None] | Literal["console"] | None,
) -> None:
    """Internal function to build an animation of rendered localizations
    given the checkpoints. See ``build_animation`` for more details."""
    angles, viewports = _animation_sequence(positions, durations, fps)

    # width and height for building the animation; must be divisible by 16
    # as ffmpeg codecs require this for proper encoding
    width, height = image_size
    width = ((width + 15) // 16) * 16
    height = ((height + 15) // 16) * 16

    # render all frames and save in RAM
    video_writer = imageio.get_writer(path, fps=fps)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(
            range(len(angles)), desc="Building animation", unit="frame"
        )
    else:
        iter_range = range(len(angles))

    for i in iter_range:
        if callable(progress_callback):
            progress_callback(i)

        disp_px_size_ = (
            _adjust_disp_px_size(disp_px_size, viewports[-1], viewports[i])
            if adjust_pixel_size
            else disp_px_size
        )
        contrast_ = _adjust_contrast(contrast, viewports[-1], viewports[i])
        qimage = render_scene(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size_,
            viewport=viewports[i],
            ang=angles[i],
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            contrast=contrast_,
            invert_colors=invert_colors,
            single_channel_colormap=single_channel_colormap,
            colors=colors,
            relative_intensities=relative_intensities,
        )[0]
        qimage = qimage.scaled(
            width,
            height,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
        )

        # convert to a np.array and append
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        frame = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]  # invert RGB to BGR
        video_writer.append_data(frame)

    if callable(progress_callback):
        progress_callback(len(angles))
    video_writer.close()

    # save a yaml with animation settings, note that yaml does not support
    # numpy types and arrays
    angles_yaml = [
        (
            float(np.degrees(p[0])),
            float(np.degrees(p[1])),
            float(np.degrees(p[2])),
        )
        for p in positions
    ]
    viewports_yaml = [
        (
            (float(p[3][0][0]), float(p[3][0][1])),
            (float(p[3][1][0]), float(p[3][1][1])),
        )
        for p in positions
    ]
    anim_settings = {
        "Generated by": f"Picasso v{__version__} Render 3D Animation",
        "FPS": fps,
        "Angles at checkpoints (x, y, z) (deg)": angles_yaml,
        "Viewports at checkpoints (camera pixels)": viewports_yaml,
        "Durations (s)": durations,
    }
    info_path = os.path.splitext(path)[0] + ".yaml"
    io.save_info(info_path, [anim_settings])


def _adjust_disp_px_size(
    disp_px_size_ref: float,
    viewport_ref: tuple[tuple[float, float], tuple[float, float]],
    new_viewport: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Adjust display pixel size based on the change in viewport to keep
    the number of display pixels the same."""
    ref_width = viewport_width(viewport_ref)
    new_width = viewport_width(new_viewport)
    # below could be ref_height / new_height, should be the same since
    # we assume the shape of the viewport stays the same
    zoom_factor = ref_width / new_width
    return disp_px_size_ref / zoom_factor


def _adjust_contrast(
    contrast_ref: tuple[float, float] | None,
    viewport_ref: tuple[tuple[float, float], tuple[float, float]],
    new_viewport: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """Adjust contrast limits based on the change in viewport to keep the
    same contrast across zoom levels."""
    if contrast_ref is None:
        return None
    ref_width = viewport_width(viewport_ref)
    new_width = viewport_width(new_viewport)
    zoom_factor = ref_width / new_width
    vmin_ref, vmax_ref = contrast_ref
    vmin_new = vmin_ref / zoom_factor**2
    vmax_new = vmax_ref / zoom_factor**2
    return vmin_new, vmax_new
