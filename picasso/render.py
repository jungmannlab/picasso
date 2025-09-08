"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :authors: Joerg Schnitzbauer 2015, Rafal Kowalewski 2023
    :copyright: Copyright (c) 2015 Jungmann Lab, MPI of Biochemistry
"""

from typing import Literal
import numpy as np
import numba
from scipy import signal
from scipy.spatial.transform import Rotation


_DRAW_MAX_SIGMA = 3  # max. sigma from mean to render (mu +/- 3 sigma)


def render(
    locs: np.recarray,
    info: dict | None = None,
    oversampling: float = 1,
    viewport: list | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0,
    ang: tuple | None = None,
) -> tuple[int, np.ndarray]:
    """Render localizations given FOV and blur method.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered.
    info : dict, optional
        Contains metadata for locs. Needed only if no viewport
        specified.
    oversampling : float, optional
        Number of super-resolution pixels per camera pixel.
    viewport : list or tuple, optional
        Field of view to be rendered. If None, all locs are rendered
    blur_method : {"gaussian", "gaussian_iso", "smooth", "convolve"} or None, \
            optional
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, no
        blurring is applied.
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
    image : np.ndarray
        Rendered image.
    """
    if viewport is None:
        try:
            # all locs
            viewport = [(0, 0), (info[0]["Height"], info[0]["Width"])]
        except TypeError:
            raise ValueError("Need info if no viewport is provided.")
    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        # no blur
        return render_hist(
            locs,
            oversampling,
            y_min, x_min, y_max, x_max,
            ang=ang,
        )
    elif blur_method == "gaussian":
        # individual localization precision
        return render_gaussian(
            locs,
            oversampling,
            y_min, x_min, y_max, x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "gaussian_iso":
        # individual localization precision (same for x and y)
        return render_gaussian_iso(
            locs,
            oversampling,
            y_min, x_min, y_max, x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "smooth":
        # one pixel blur
        return render_smooth(
            locs,
            oversampling,
            y_min, x_min, y_max, x_max,
            ang=ang,
        )
    elif blur_method == "convolve":
        # global localization precision
        return render_convolve(
            locs,
            oversampling,
            y_min, x_min, y_max, x_max,
            min_blur_width,
            ang=ang,
        )
    else:
        raise Exception("blur_method not understood.")


@numba.njit
def _render_setup(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float, y_max: float, x_max: float,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Find coordinates to be rendered and sets up an empty image
    array.

    Parameters
    ----------
    locs : np.recarray
        Localizations.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).

    Returns
    -------
    image : np.ndarray
        Empty image array.
    n_pixel_y : int
        Number of pixels in y.
    n_pixel_x : int
        Number of pixels in x.
    x : np.ndarray
        x coordinates to be rendered.
    y : np.ndarray
        y coordinates to be rendered.
    in_view : np.ndarray
        Indeces of locs to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    x = locs.x
    y = locs.y
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _render_setup3d(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float,
    y_max: float, x_max: float,
    z_min: float, z_max: float,
    pixelsize: float,
) -> tuple[
    np.ndarray,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Find coordinates to be rendered in 3D and sets up an empty image
    array. Used by Picasso: Average3.

    Parameters
    ----------
    locs : np.recarray
        Localizations.
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
    image : np.ndarray
        Empty image array.
    n_pixel_y, n_pixel_x, n_pixel_z : int
        Number of pixels in y, x, and z.
    x, y, z : np.ndarray
        x, y, z coordinates to be rendered.
    in_view : np.ndarray
        Indeces of locs to be rendered.
    """
    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    n_pixel_z = int(np.ceil(oversampling * (z_max - z_min)))
    x = locs.x
    y = locs.y
    z = locs.z / pixelsize
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

# @numba.njit
# def _render_setupz(
#     locs, oversampling, x_min, z_min, x_max, z_max
# ):
#     n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
#     n_pixel_z = int(np.ceil(oversampling * (z_max - z_min)))
#     x = locs.x
#     z = locs.z
#     in_view = (x > x_min) & (z > z_min) & (x < x_max) & (z < z_max)
#     x = x[in_view]
#     z = z[in_view]
#     x = oversampling * (x - x_min)
#     z = oversampling * (z - z_min)
#     image = np.zeros((n_pixel_x, n_pixel_z), dtype=np.float32)
#     return image, n_pixel_z, n_pixel_x, x, z, in_view


@numba.njit
def _fill(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Fill image with x and y coordinates. Image is not blurred.

    Parameters
    ----------
    image : np.ndarray
        Empty image array.
    x, y : np.ndarray
        x and y coordinates to be rendered.
    """
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@numba.njit
def _fill3d(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> None:
    """Fill image with x, y and z coordinates. Image is not blurred.
    Used by ``Picasso: Average3``.

    Parameters
    ----------
    image : np.ndarray
        Empty image array.
    x, y, z : np.ndarray
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
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    n_pixel_x: int,
    n_pixel_y: int,
) -> None:
    """Fill image with blurred x and y coordinates. Each localization
    is rendered as a 2D Gaussian centered at (x, y) with standard
    deviations (sx, sy).

    Parameters
    ----------
    image : np.ndarray
        Empty image array.
    x, y : np.ndarray
        x and y coordinates to be rendered.
    sx, sy : np.ndarray
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
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    n_pixel_x: int,
    n_pixel_y: int,
    ang: tuple[float, float, float],
) -> None:
    """Fill image with rotated gaussian-blurred localizations.

    Localization precisions (sx, sy and sz) are treated as standard
    deviations of the guassians to be rendered.

    See https://cs229.stanford.edu/section/gaussians.pdf

    Parameters
    ----------
    image : np.ndarray
        Empty image array.
    x, y, z : np.ndarray
        3D coordinates to be rendered.
    sx, sy, sz : np.ndarray
        Localization precision in x, y and z for each localization.
    n_pixel_x, n_pixel_y : int
        Number of pixels in x and y.
    ang : tuple
        Rotation angles of locs around x, y and z axes (radians).
    """
    (angx, angy, angz) = ang  # rotation angles

    rot_mat_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angx), np.sin(angx)],
            [0.0, -np.sin(angx), np.cos(angx)],
        ], dtype=np.float32
    )  # rotation matrix around x axis
    rot_mat_y = np.array(
        [
            [np.cos(angy), 0.0, np.sin(angy)],
            [0.0, 1.0, 0.0],
            [-np.sin(angy), 0.0, np.cos(angy)],
        ], dtype=np.float32
    )  # rotation matrix around y axis
    rot_mat_z = np.array(
        [
            [np.cos(angz), -np.sin(angz), 0.0],
            [np.sin(angz), np.cos(angz), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32
    )  # rotation matrix around z axis
    rot_matrix = rot_mat_x @ rot_mat_y @ rot_mat_z  # rotation matrix
    rot_matrixT = np.transpose(rot_matrix)  # ...and its transpose

    # draw each localization separately
    for x_, y_, z_, sx_, sy_, sz_ in zip(x, y, z, sx, sy, sz):

        # get min and max indeces to draw the given localization
        max_y = (_DRAW_MAX_SIGMA * 2.5) * sy_
        i_min = int(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = int(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = (_DRAW_MAX_SIGMA * 2.5) * sx_
        j_min = int(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = int(x_ + max_x + 1)
        if j_max > n_pixel_x:
            j_max = n_pixel_x
        max_z = (_DRAW_MAX_SIGMA * 2.5) * sz_
        k_min = int(z_ - max_z)
        k_max = int(z_ + max_z + 1)

        # rotate localization precisions in 3D
        cov_matrix = np.array(
            [
                [sx_**2, 0, 0],
                [0, sy_**2, 0],
                [0, 0, sz_**2],
            ], dtype=np.float32
        )  # covariance matrix (CM)
        cov_rot = rot_matrix @ cov_matrix @ rot_matrixT  # rotated CM
        cri = inverse_3x3(cov_rot)  # inverse of rotated CM
        dcr = determinant_3x3(cov_rot)  # determinant of rotated CM

        # draw a localization in 2D - sum z coordinates;
        # PDF of a rotated gaussian in 3D is calculated and
        # image is summed over z axis
        for i in range(i_min, i_max):
            b = np.float32(i + 0.5 - y_)
            for j in range(j_min, j_max):
                a = np.float32(j + 0.5 - x_)
                for k in range(k_min, k_max):
                    c = np.float32(k + 0.5 - z_)
                    exponent = (
                        a * a * cri[0, 0]
                        + a * b * cri[0, 1]
                        + a * c * cri[0, 2]
                        + a * b * cri[1, 0]
                        + b * b * cri[1, 1]
                        + b * c * cri[1, 2]
                        + a * c * cri[2, 0]
                        + b * c * cri[2, 1]
                        + c * c * cri[2, 2]
                    )  # Mahalanobis distance
                    image[i, j] += (
                        np.exp(-0.5 * exponent)
                        / (((2 * np.pi) ** 3 * dcr) ** 0.5)
                    )


@numba.njit
def inverse_3x3(a: np.ndarray) -> np.ndarray:
    """Calculate inverse of a 3x3 matrix. This function is faster than
    ``np.linalg.inv``.

    Parameters
    ----------
    a : np.ndarray
        3x3 matrix.

    Returns
    -------
    c : np.ndarray
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
def determinant_3x3(a: np.ndarray) -> np.float32:
    """Calculate determinant of a 3x3 matrix. This function is faster
    than ``np.linalg.det``.

    Parameters
    ----------
    a : np.ndarray
        3x3 matrix.

    Returns
    -------
    det : float
        Determinant of ``a``.
    """
    det = np.float32(
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[2, 0] * a[1, 2])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[2, 0] * a[1, 1])
    )
    return det


def render_hist(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float, y_max: float, x_max: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, np.ndarray]:
    """Render localizations with no blur by assigning them to pixels.

    Parameters
    ----------
    locs : np.recarray
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
    image : np.ndarray
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min, x_min, y_max, x_max,
    )
    if ang:
        x, y, _, _ = locs_rotation(
            locs,
            oversampling,
            x_min, x_max, y_min, y_max,
            ang,
        )
    _fill(image, x, y)
    n = len(x)
    return n, image


# @numba.jit(nopython=True, nogil=True)
# def render_histz(locs, oversampling, x_min, z_min, x_max, z_max):
#     image, n_pixel_z, n_pixel_x, x, z, in_view = _render_setupz(
#         locs, oversampling, x_min, z_min, x_max, z_max
#     )
#     _fill(image, z, x)
#     return len(x), image


@numba.jit(nopython=True, nogil=True)
def render_hist3d(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float,
    y_max: float, x_max: float,
    z_min: float, z_max: float,
    pixelsize: float,
) -> tuple[int, np.ndarray]:
    """Render localizations in 3D with no blur by assigning them to
    pixels. Used by ``Picasso: Average3``.

    Parameters
    ----------
    locs : np.recarray
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
    image : np.ndarray
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = _render_setup3d(
        locs,
        oversampling,
        y_min, x_min, y_max, x_max, z_min, z_max,
        pixelsize,
    )
    _fill3d(image, x, y, z)
    n = len(x)
    return n, image


def render_gaussian(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float, y_max: float, x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, np.ndarray]:
    """Render localizations with with individual localization precision
    which differs in x and y.

    Parameters
    ----------
    locs : np.recarray
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
        Rotation angles of locs around x, y and z axes in radians. If
        None, locs are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : np.ndarray
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min, x_min, y_max, x_max,
    )

    if not ang:  # not rotated
        blur_width = oversampling * np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(locs.lpy, min_blur_width)
        sy = blur_height[in_view]
        sx = blur_width[in_view]

        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = locs_rotation(
            locs,
            oversampling,
            x_min, x_max, y_min, y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(locs.lpy, min_blur_width)
        # for now, let lpz be twice the mean of lpx and lpy (TODO):
        if hasattr(locs, "lpz"):
            lpz = locs.lpz  # NOTE: lpz must be in the same units as lpx
        else:
            lpz = 2 * np.mean(np.stack((locs.lpx, locs.lpy)), axis=0)
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
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float, y_max: float, x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, np.ndarray]:
    """Same as ``render_gaussian``, but uses the same localization
    precision in x and y."""
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min, x_min, y_max, x_max,
    )

    if not ang:  # not rotated
        blur_width = oversampling * np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(locs.lpy, min_blur_width)
        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy

        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = locs_rotation(
            locs,
            oversampling,
            x_min, x_max, y_min, y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(locs.lpy, min_blur_width)
        # for now, let lpz be twice the mean of lpx and lpy (TODO):
        if hasattr(locs, "lpz"):
            lpz = locs.lpz  # NOTE: lpz must be in the same units as lpx
        else:
            lpz = 2 * np.mean(np.stack((locs.lpx, locs.lpy)), axis=0)
        blur_depth = oversampling * np.maximum(lpz, min_blur_width)

        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
        sz = blur_depth[in_view]

        _fill_gaussian_rot(
            image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang
        )

    return len(x), image


def render_convolve(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float, y_max: float, x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, np.ndarray]:
    """Render localizations with with global localization precision,
    i.e. each localization is blurred by the median localization
    precision in x and y.

    Parameters
    ----------
    locs : np.recarray
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
        Rotation angles of locs around x, y and z axes in radians. If
        None, locs are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : np.ndarray
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min, x_min, y_max, x_max,
    )
    if ang:  # rotate
        x, y, in_view, _ = locs_rotation(
            locs,
            oversampling,
            x_min, x_max, y_min, y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        blur_width = oversampling * max(
            np.median(locs.lpx[in_view]), min_blur_width
        )
        blur_height = oversampling * max(
            np.median(locs.lpy[in_view]), min_blur_width
        )
        return n, _fftconvolve(image, blur_width, blur_height)


def render_smooth(
    locs: np.recarray,
    oversampling: float,
    y_min: float, x_min: float, y_max: float, x_max: float,
    ang: tuple[float, float, float] | None = None,
) -> tuple[int, np.ndarray]:
    """Render localizations with with blur of one display pixel (set by
    oversampling).

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    ang : tuple, optional
        Rotation angles of locs around x, y and z axes in radians. If
        None, locs are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : np.array
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min, x_min, y_max, x_max,
    )

    if ang:
        x, y, _, _ = locs_rotation(
            locs,
            oversampling,
            x_min, x_max, y_min, y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(
    image: np.ndarray,
    blur_width: float,
    blur_height: float,
) -> np.ndarray:
    """Blur (convolves) 2D image using fast fourier transform.

    Parameters
    ----------
    image : np.ndarray
        Image with rendered but not blurred locs.
    blur_width, blur_height : float
        Blur width and height in pixels.

    Returns
    -------
    image : np.ndarray
        Blurred image.
    """
    kernel_width = 10 * int(np.round(blur_width)) + 1
    kernel_height = 10 * int(np.round(blur_height)) + 1
    kernel_y = signal.gaussian(kernel_height, blur_height)
    kernel_x = signal.gaussian(kernel_width, blur_width)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    image = signal.fftconvolve(image, kernel, mode="same")
    return image


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
    locs: np.recarray,
    oversampling: float,
    x_min: float, x_max: float, y_min: float, y_max: float,
    ang: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate localizations within a FOV.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rotated.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinate to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinate to be rendered (camera pixels).
    ang : tuple
        Rotation angles of locs around x, y and z axes in radians.

    Returns
    -------
    x : np.ndarray
        New (rotated) x coordinates
    y : np.ndarray
        New y coordinates
    in_view : np.ndarray
        Indeces of locs that are rendered
    z : np.ndarray
        New z coordinates
    """
    # z is translated to pixels
    locs_coord = np.stack((locs.x, locs.y, locs.z)).T

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
