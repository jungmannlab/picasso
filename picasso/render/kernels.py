"""
picasso.render.kernels
~~~~~~~~~~~~~~~~~~~~~~

Numba-compiled CPU kernels for the render package: splat setup and
drawing, histogram filling and the fused image-composition kernels.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numba
import numpy as np

from .. import lib


_DRAW_MAX_SIGMA = 3  # max. sigma from mean to render (mu +/- 3 sigma)


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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
    # divide on a copy -- rendering must never mutate the caller's arrays
    # (see TestRenderPurity in test_render.py)
    z = z.copy()
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


@numba.njit(nogil=True)
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
    oversampling_x, oversampling_y, oversampling_z : float
        Number of super-resolution pixels per camera pixel in x, y and
        z.
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
    # divide on a copy -- rendering must never mutate the caller's arrays
    # (see TestRenderPurity in test_render.py)
    z = z.copy()
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


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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
    for i, j, k in zip(x, y, z):
        image[j, i, k] += 1


@numba.njit(cache=True, nogil=True)
def _gaussian_bbox(
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    n_pixel_x: int,
    n_pixel_y: int,
) -> tuple[int, int, int, int]:
    """Clamp the +/- ``_DRAW_MAX_SIGMA`` box of a Gaussian to the image.

    Parameters
    ----------
    x_, y_ : float
        Center of the Gaussian (display pixels).
    sx_, sy_ : float
        Standard deviations of the Gaussian in x and y (display
        pixels).
    n_pixel_x, n_pixel_y : int
        Image size in x and y (display pixels).

    Returns
    -------
    i_min, i_max, j_min, j_max : int
        Row and column bounds of the box (``i_max``/``j_max``
        exclusive).
    """
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
    return i_min, i_max, j_min, j_max


@numba.njit(cache=True, nogil=True)
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
    if not (sx_ > 0.0 and sy_ > 0.0):
        # Degenerate localization (e.g. lpx/lpy of exactly 0 from a
        # singular CRLB fit); also catches NaN. Skip instead of
        # dividing by zero.
        return
    i_min, i_max, j_min, j_max = _gaussian_bbox(
        x_, y_, sx_, sy_, n_pixel_x, n_pixel_y
    )
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


@numba.njit(cache=True, nogil=True)
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


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_theta_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    angle_: float,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Render a single in-plane rotated 2D Gaussian into ``image``.

    The elliptical Gaussian with standard deviations (``sx_``, ``sy_``)
    is rotated in the image plane by ``angle_`` (radians) via its 2x2
    covariance matrix. Unlike ``_draw_gaussian_loc`` the rotated kernel
    is not separable (it has a cross term), so pixels are evaluated with
    the full bivariate quadratic form, as in ``_draw_gaussian_rot_loc``.
    """
    c = np.cos(angle_)
    s = np.sin(angle_)
    vx = sx_ * sx_
    vy = sy_ * sy_
    cxx = vx * c * c + vy * s * s
    cyy = vx * s * s + vy * c * c
    cxy = (vx - vy) * s * c
    det2d = cxx * cyy - cxy * cxy
    if det2d < 1e-10:
        return
    inv_xx = cyy / det2d
    inv_yy = cxx / det2d
    inv_xy = -cxy / det2d
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det2d))
    max_x_off = _DRAW_MAX_SIGMA * np.sqrt(cxx)
    max_y_off = _DRAW_MAX_SIGMA * np.sqrt(cyy)
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
            exponent = a * a * inv_xx + 2.0 * a * b * inv_xy + b * b * inv_yy
            image[i, j] += norm * np.exp(-0.5 * exponent)


@numba.njit(cache=True, nogil=True)
def _fill_gaussian_theta(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    angle: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Fill image with in-plane rotated gaussian-blurred localizations.

    Each localization is rendered as a 2D Gaussian centered at (x, y)
    with standard deviations (sx, sy) rotated in the image plane by its
    own ``angle`` (radians).

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        x and y coordinates to be rendered.
    sx, sy : lib.FloatArray1D
        Localization precision in x and y for each localization.
    angle : lib.FloatArray1D
        In-plane rotation angle (radians) for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    """
    n_locs = len(x)
    if n_locs == 0:
        return

    for i in range(n_locs):
        _draw_gaussian_theta_loc(
            image, x[i], y[i], sx[i], sy[i], angle[i], n_pixel_x, n_pixel_y
        )


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_cov3d_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    cov: lib.Array3x3,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
    rot_matrixT: lib.Array3x3,
) -> None:
    """Render a single 3D Gaussian with local covariance ``cov`` into
    ``image``: ``cov`` is rotated by the global ``rot_matrix`` and the
    top-left 2x2 block is projected, inverted and drawn as a bivariate
    Gaussian. Shared by ``_draw_gaussian_rot_loc`` (diagonal ``cov``) and
    ``_draw_gaussian_rot_theta_loc`` (in-plane rotated ``cov``)."""
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


@numba.njit(cache=True, nogil=True)
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
    _draw_gaussian_cov3d_loc(
        image, x_, y_, cov, n_pixel_x, n_pixel_y, rot_matrix, rot_matrixT
    )


@numba.njit(cache=True, nogil=True)
def _draw_gaussian_rot_theta_loc(
    image: lib.FloatArray2D,
    x_: float,
    y_: float,
    sx_: float,
    sy_: float,
    sz_: float,
    angle_: float,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
    rot_matrixT: lib.Array3x3,
) -> None:
    """Render a single rotated 2D Gaussian (projected from 3D) into
    ``image``. The in-plane precision ellipse (``sx_``, ``sy_``) is
    first rotated by ``angle_`` (radians) about the z-axis, then the
    full 3D covariance is rotated by the global ``rot_matrix``."""
    c = np.cos(angle_)
    s = np.sin(angle_)
    vx = sx_ * sx_
    vy = sy_ * sy_
    cov = np.zeros((3, 3), dtype=np.float32)
    cov[0, 0] = vx * c * c + vy * s * s
    cov[1, 1] = vx * s * s + vy * c * c
    cov[0, 1] = (vx - vy) * c * s
    cov[1, 0] = cov[0, 1]
    cov[2, 2] = sz_ * sz_
    _draw_gaussian_cov3d_loc(
        image, x_, y_, cov, n_pixel_x, n_pixel_y, rot_matrix, rot_matrixT
    )


@numba.njit(cache=True, nogil=True)
def _fill_gaussian_rot(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    sz: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
) -> None:
    """Fill image with rotated gaussian-blurred localizations.

    Localization precisions (sx, sy and sz) are treated as standard
    deviations of the gaussians to be rendered.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        Rotated x and y coordinates to be rendered (display pixels).
    sx, sy, sz : lib.FloatArray1D
        Localization precision in x, y and z for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    rot_matrix : lib.Array3x3
        Rotation matrix (float32) applied to the localizations.
    """
    n_locs = len(x)
    if n_locs == 0:
        return
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


@numba.njit(cache=True, nogil=True)
def _fill_gaussian_rot_theta(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    sx: lib.FloatArray1D,
    sy: lib.FloatArray1D,
    sz: lib.FloatArray1D,
    angle: lib.FloatArray1D,
    n_pixel_x: int,
    n_pixel_y: int,
    rot_matrix: lib.Array3x3,
) -> None:
    """Fill image with rotated gaussian-blurred localizations, each with
    its own in-plane rotation.

    Same as ``_fill_gaussian_rot`` but the precision ellipse (sx, sy) of
    every localization is first rotated in the image plane by its own
    ``angle`` (radians) about the z-axis, before the global rotation.

    Parameters
    ----------
    image : lib.FloatArray2D
        Empty image array.
    x, y : lib.FloatArray1D
        Rotated x and y coordinates to be rendered (display pixels).
    sx, sy, sz : lib.FloatArray1D
        Localization precision in x, y and z for each localization.
    angle : lib.FloatArray1D
        In-plane rotation angle (radians) for each localization,
        applied about the z-axis before the global ``rot_matrix``.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    rot_matrix : lib.Array3x3
        Rotation matrix (float32) applied to the localizations.
    """
    n_locs = len(x)
    if n_locs == 0:
        return
    rot_matrixT = np.ascontiguousarray(rot_matrix.T)

    for i in range(n_locs):
        _draw_gaussian_rot_theta_loc(
            image,
            x[i],
            y[i],
            sx[i],
            sy[i],
            sz[i],
            angle[i],
            n_pixel_x,
            n_pixel_y,
            rot_matrix,
            rot_matrixT,
        )


@numba.njit(nogil=True)
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


@numba.njit(nogil=True)
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


@numba.njit(cache=True, nogil=True)
def _lut_pixel_rgb(
    raw: lib.FloatArray3D,
    luts: lib.FloatArray3D,
    i: int,
    j: int,
    n_channels: int,
    vmin32: np.float32,
    rng32: np.float32,
    rel: lib.FloatArray1D,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Additive RGB (and total coverage) of pixel ``(i, j)`` over all
    channels: contrast scale, clip, LUT gather, blend."""
    one = np.float32(1.0)
    zero = np.float32(0.0)
    r = zero
    g = zero
    b = zero
    coverage = zero
    for c in range(n_channels):
        v = (raw[c, i, j] - vmin32) / rng32
        if not np.isfinite(v):
            v = zero
        if v < zero:
            v = zero
        elif v > one:
            v = one
        v = v * rel[c]
        idx = np.int32(v * np.float32(255.0))
        if idx < 0:
            idx = 0
        elif idx > 255:
            idx = 255
        r += luts[c, idx, 0]
        g += luts[c, idx, 1]
        b += luts[c, idx, 2]
        coverage += v
    if r > one:
        r = one
    if g > one:
        g = one
    if b > one:
        b = one
    return r, g, b, coverage


@numba.njit(cache=True, nogil=True)
def _composite_bg(
    r: np.float32,
    g: np.float32,
    b: np.float32,
    coverage: np.float32,
    bg: lib.FloatArray1D,
) -> tuple[np.float32, np.float32, np.float32]:
    """Blend the background color into the uncovered fraction of a
    pixel, then clip back to [0, 1]."""
    one = np.float32(1.0)
    zero = np.float32(0.0)
    if coverage < zero:
        coverage = zero
    elif coverage > one:
        coverage = one
    remainder = one - coverage
    r += bg[0] * remainder
    g += bg[1] * remainder
    b += bg[2] * remainder
    if r > one:
        r = one
    if g > one:
        g = one
    if b > one:
        b = one
    return r, g, b


@numba.njit(cache=True, nogil=True)
def _compose_multi_lut(
    raw: lib.FloatArray3D,
    luts: lib.FloatArray3D,
    vmin: float,
    vmax: float,
    rel: lib.FloatArray1D,
    bg: lib.FloatArray1D,
    has_bg: bool,
) -> tuple[lib.FloatArray3D, np.float32]:
    """Fused replacement for the multi-channel numpy post-processing
    chain (contrast scale -> intensity scale -> LUT gather -> additive
    blend -> background compositing), reading the raw stack once.

    Reproduces the chain's float32 arithmetic step for step (subtract
    then divide, non-finite to zero, clip, truncating int cast for the
    LUT index) so results match the legacy path bit-for-bit up to
    quantization. Returns the float RGB image plus its global maximum,
    which ``_quantize_rgb`` needs for ``to_8bit``'s renormalization.
    """
    n_channels, n_y, n_x = raw.shape
    vmin32 = np.float32(vmin)
    rng32 = np.float32(vmax - vmin)
    zero = np.float32(0.0)
    rgb = np.empty((n_y, n_x, 3), dtype=np.float32)
    max_value = zero
    for i in range(n_y):
        for j in range(n_x):
            r, g, b, coverage = _lut_pixel_rgb(
                raw, luts, i, j, n_channels, vmin32, rng32, rel
            )
            if has_bg:
                r, g, b = _composite_bg(r, g, b, coverage, bg)
            rgb[i, j, 0] = r
            rgb[i, j, 1] = g
            rgb[i, j, 2] = b
            if r > max_value:
                max_value = r
            if g > max_value:
                max_value = g
            if b > max_value:
                max_value = b
    return rgb, max_value


@numba.njit(cache=True, nogil=True)
def _quantize_rgb(
    rgb: lib.FloatArray3D, max_value: np.float32
) -> lib.IntArray3D:
    """``to_8bit`` as a kernel: divide by the global maximum (when
    positive) and round to uint8, matching numpy's half-to-even."""
    n_y, n_x, _ = rgb.shape
    denom = max_value if max_value > np.float32(0.0) else np.float32(1.0)
    out = np.empty((n_y, n_x, 3), dtype=np.uint8)
    for i in range(n_y):
        for j in range(n_x):
            for k in range(3):
                t = (rgb[i, j, k] / denom) * np.float32(255.0)
                out[i, j, k] = np.uint8(np.rint(t))
    return out


@numba.njit(cache=True, nogil=True)
def _compose_single(
    raw: lib.FloatArray2D,
    cmap: lib.IntArray2D,
    vmin: float,
    vmax: float,
) -> lib.IntArray3D:
    """Fused replacement for the single-channel chain
    (``scale_contrast`` -> ``to_8bit`` -> ``apply_colormap``): contrast
    scale with clipping, renormalize by the global maximum, round to the
    256-entry colormap index and gather RGB."""
    n_y, n_x = raw.shape
    vmin32 = np.float32(vmin)
    rng32 = np.float32(vmax - vmin)
    one = np.float32(1.0)
    zero = np.float32(0.0)
    scaled = np.empty((n_y, n_x), dtype=np.float32)
    max_value = zero
    for i in range(n_y):
        for j in range(n_x):
            v = (raw[i, j] - vmin32) / rng32
            if not np.isfinite(v):
                v = zero
            if v < zero:
                v = zero
            elif v > one:
                v = one
            scaled[i, j] = v
            if v > max_value:
                max_value = v
    denom = max_value if max_value > zero else one
    out = np.empty((n_y, n_x, 3), dtype=np.uint8)
    for i in range(n_y):
        for j in range(n_x):
            t = (scaled[i, j] / denom) * np.float32(255.0)
            idx = np.int64(np.rint(t))
            out[i, j, 0] = cmap[idx, 0]
            out[i, j, 1] = cmap[idx, 1]
            out[i, j, 2] = cmap[idx, 2]
    return out


@numba.njit(cache=True, nogil=True)
def _quadtree_spread_leaf(
    image: lib.FloatArray2D,
    count: int,
    side: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x_min: float,
    y_min: float,
    oversampling: float,
    px: float,
    n_px: int,
    n_py: int,
) -> None:
    """Spread a leaf's count over the display pixels its cell overlaps,
    in proportion to the pixel/cell overlap area."""
    rho = count / (side * side)
    j0 = max(int(np.floor((x0 - x_min) * oversampling)), 0)
    j1 = min(int(np.floor((x1 - x_min) * oversampling)), n_px - 1)
    i0 = max(int(np.floor((y0 - y_min) * oversampling)), 0)
    i1 = min(int(np.floor((y1 - y_min) * oversampling)), n_py - 1)
    for i in range(i0, i1 + 1):
        py0 = y_min + i * px
        oy = min(py0 + px, y1) - max(py0, y0)
        if oy <= 0:
            continue
        for j in range(j0, j1 + 1):
            qx0 = x_min + j * px
            ox = min(qx0 + px, x1) - max(qx0, x0)
            if ox > 0:
                image[i, j] += rho * ox * oy


@numba.njit(cache=True, nogil=True)
def _quadtree_bin_rows(
    image: lib.FloatArray2D,
    perm: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    s: int,
    e: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    oversampling: float,
    n_px: int,
    n_py: int,
) -> None:
    """Bin rows ``perm[s:e]`` one by one, as ``_fill`` does."""
    for k in range(s, e):
        p = perm[k]
        xx = x[p]
        yy = y[p]
        if xx > x_min and xx < x_max and yy > y_min and yy < y_max:
            j = int(oversampling * (xx - x_min))
            i = int(oversampling * (yy - y_min))
            if i < n_py and j < n_px:
                image[i, j] += 1.0


@numba.njit(cache=True, nogil=True)
def _quadtree_push_children(
    sorted_keys: lib.IntArray1D,
    st_d: lib.IntArray1D,
    st_ix: lib.IntArray1D,
    st_iy: lib.IntArray1D,
    st_pre: np.ndarray,
    st_s: lib.IntArray1D,
    st_e: lib.IntArray1D,
    top: int,
    d: int,
    ix: int,
    iy: int,
    pre: np.uint64,
    s: int,
    e: int,
    total_bits: int,
) -> int:
    """Push a node's four children onto the traversal stack.

    The children's permutation ranges are found by binary search on the
    sorted keys.

    Parameters
    ----------
    sorted_keys : lib.IntArray1D
        Sorted Morton keys of the rows, see
        ``picasso.spatial_index.quadtree_layout``.
    st_d, st_ix, st_iy : lib.IntArray1D
        Stack buffers of the node depth and x/y cell indices.
    st_pre : np.ndarray
        Stack buffer of the node key prefixes (uint64).
    st_s, st_e : lib.IntArray1D
        Stack buffers of the start (inclusive) and end (exclusive) of
        the nodes' permutation ranges.
    top : int
        Current stack top (number of entries on the stack).
    d : int
        Depth of the node.
    ix, iy : int
        x and y cell indices of the node at depth ``d``.
    pre : np.uint64
        Key prefix of the node.
    s, e : int
        Start (inclusive) and end (exclusive) of the node's permutation
        range.
    total_bits : int
        Number of bits per coordinate of the Morton keys (tree depth).

    Returns
    -------
    int
        The updated stack top.
    """
    shift = np.uint64(2 * (total_bits - d - 1))
    base_pre = pre << np.uint64(2)
    lo = s
    for c in range(4):
        if c < 3:
            hi_key = (base_pre + np.uint64(c + 1)) << shift
            hi = s + np.searchsorted(sorted_keys[s:e], hi_key)
        else:
            hi = e
        st_d[top] = d + 1
        st_ix[top] = 2 * ix + (c & 1)
        st_iy[top] = 2 * iy + (c >> 1)
        st_pre[top] = base_pre + np.uint64(c)
        st_s[top] = lo
        st_e[top] = hi
        top += 1
        lo = hi
    return top


@numba.njit(cache=True, nogil=True)
def _quadtree_fill(
    image: lib.FloatArray2D,
    sorted_keys: lib.IntArray1D,
    perm: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    root_px: float,
    total_bits: int,
    capacity: int,
) -> None:
    """Paint the quad-tree adaptive histogram (Baddeley, Cannell &
    Soeller 2010) of the viewport into ``image``.

    ``sorted_keys``, ``perm``, ``root_px`` and ``total_bits`` describe
    the tree implicit in a channel's spatial index, see
    ``picasso.spatial_index.quadtree_layout`` for the contract. Depth-
    first descent from the root over the nodes overlapping the
    viewport: a node holding more than ``capacity`` rows and wider
    than a display pixel is split into its four children (their ranges
    of the permutation found by binary search on the sorted keys); a
    leaf wider than a pixel spreads its count over the display pixels
    it covers in proportion to the overlap (the image is in
    localizations per display pixel, like the histogram); a node no
    wider than a pixel (or at the finest cell) bins its rows one by
    one, exactly as ``_fill`` does, so a capacity of 0 gives the
    histogram.
    """
    n = perm.shape[0]
    n_py, n_px = image.shape
    px = 1.0 / oversampling
    cap = 4 * (total_bits + 2)
    st_d = np.empty(cap, dtype=np.int64)
    st_ix = np.empty(cap, dtype=np.int64)
    st_iy = np.empty(cap, dtype=np.int64)
    st_pre = np.empty(cap, dtype=np.uint64)
    st_s = np.empty(cap, dtype=np.int64)
    st_e = np.empty(cap, dtype=np.int64)
    st_d[0] = 0
    st_ix[0] = 0
    st_iy[0] = 0
    st_pre[0] = np.uint64(0)
    st_s[0] = 0
    st_e[0] = n
    top = 1
    while top > 0:
        top -= 1
        d = st_d[top]
        ix = st_ix[top]
        iy = st_iy[top]
        pre = st_pre[top]
        s = st_s[top]
        e = st_e[top]
        count = e - s
        if count == 0:
            continue
        side = root_px / (1 << d)
        x0 = ix * side
        y0 = iy * side
        x1 = x0 + side
        y1 = y0 + side
        if x1 <= x_min or x0 >= x_max or y1 <= y_min or y0 >= y_max:
            continue
        if count > capacity and side > px and d < total_bits:
            top = _quadtree_push_children(
                sorted_keys,
                st_d,
                st_ix,
                st_iy,
                st_pre,
                st_s,
                st_e,
                top,
                d,
                ix,
                iy,
                pre,
                s,
                e,
                total_bits,
            )
        elif side > px:
            # a leaf wider than a pixel: its density over its area
            _quadtree_spread_leaf(
                image,
                count,
                side,
                x0,
                y0,
                x1,
                y1,
                x_min,
                y_min,
                oversampling,
                px,
                n_px,
                n_py,
            )
        else:
            # within a pixel (or the finest cell): bin the rows exactly
            _quadtree_bin_rows(
                image,
                perm,
                x,
                y,
                s,
                e,
                x_min,
                x_max,
                y_min,
                y_max,
                oversampling,
                n_px,
                n_py,
            )


@numba.njit(cache=True, nogil=True)
def _triangle_row_span(
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    xc: float,
    yc: float,
    yc_row: float,
) -> tuple[float, float]:
    """Intersections of the row ``y == yc_row`` with the triangle's
    three edges, as the row's (left, right) x-span."""
    x_left = 1e300
    x_right = -1e300
    for k in range(3):
        if k == 0:
            x0, y0, x1, y1 = xa, ya, xb, yb
        elif k == 1:
            x0, y0, x1, y1 = xb, yb, xc, yc
        else:
            x0, y0, x1, y1 = xc, yc, xa, ya
        if (y0 <= yc_row < y1) or (y1 <= yc_row < y0):
            xi = x0 + (yc_row - y0) * (x1 - x0) / (y1 - y0)
            if xi < x_left:
                x_left = xi
            if xi > x_right:
                x_right = xi
    return x_left, x_right


@numba.njit(cache=True, nogil=True)
def _fill_triangles(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    simplices: lib.IntArray2D,
    mass: float,
) -> None:
    """Paint triangles into ``image`` (display pixel coordinates), each
    carrying ``mass`` localizations spread evenly over its area, so the
    intensity is ``mass / area`` per pixel: the triangulation render of
    Baddeley, Cannell & Soeller (2010).

    A triangle wider than a pixel is scan-converted row by row (the
    pixels whose centers lie inside it); a triangle no wider than a
    pixel puts its whole mass into the pixel holding its centroid, so
    at an overview the image tends to the histogram and no mass is
    lost.
    """
    n_py, n_px = image.shape
    for t in range(simplices.shape[0]):
        a = simplices[t, 0]
        b = simplices[t, 1]
        c = simplices[t, 2]
        xa, ya = x[a], y[a]
        xb, yb = x[b], y[b]
        xc, yc = x[c], y[c]
        area = 0.5 * abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))
        if area <= 0.0:
            continue
        x_lo = min(xa, xb, xc)
        x_hi = max(xa, xb, xc)
        y_lo = min(ya, yb, yc)
        y_hi = max(ya, yb, yc)
        if x_hi - x_lo <= 1.0 and y_hi - y_lo <= 1.0:
            # within a pixel: its mass into the centroid's pixel
            j = int((xa + xb + xc) / 3.0)
            i = int((ya + yb + yc) / 3.0)
            if 0 <= i < n_py and 0 <= j < n_px:
                image[i, j] += mass
            continue
        density = mass / area
        i0 = max(int(np.floor(y_lo - 0.5)), 0)
        i1 = min(int(np.ceil(y_hi - 0.5)), n_py - 1)
        for i in range(i0, i1 + 1):
            yc_row = i + 0.5  # the row's pixel centers
            x_left, x_right = _triangle_row_span(
                xa, ya, xb, yb, xc, yc, yc_row
            )
            if x_right < x_left:
                continue
            j0 = max(int(np.ceil(x_left - 0.5)), 0)
            j1 = min(int(np.floor(x_right - 0.5)), n_px - 1)
            for j in range(j0, j1 + 1):
                image[i, j] += density
