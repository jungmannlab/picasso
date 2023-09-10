"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, MPI of Biochemistry
"""
import time
import os
import sys

import numpy as _np
import numba as _numba
import scipy.signal as _signal
from scipy.spatial.transform import Rotation as _Rotation


_DRAW_MAX_SIGMA = 3


def render(
    locs,
    info=None,
    oversampling=1,
    viewport=None,
    blur_method=None,
    min_blur_width=0,
    ang=None,
):  
    """
    Renders locs.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    info : dict (default=None)
        Contains metadata for locs. Needed only if no viewport 
        specified
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    viewport : list or tuple (default=None)
        Field of view to be rendered. If None, all locs are rendered
    blur_method : str (default=None)
        Defines localizations' blur. The string has to be one of 
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, 
        no blurring is applied.
    min_blur_width : float (default=0)
        Minimum size of blur (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None, 
        locs are not rotated.

    Raises
    ------
    Exception
        If blur_method not one of 'gaussian', 'gaussian_iso', 'smooth', 
        'convolve' or None

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
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


@_numba.njit
def _render_setup(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max,
):
    """
    Finds coordinates to be rendered and sets up an empty image array.

    Parameters
    ----------
    locs : np.recarray
        Localizations
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    np.array
        Empty image array
    int
        Number of pixels in y
    int 
        Number of pixels in x
    np.array 
        x coordinates to be rendered
    np.array 
        y coordinates to be rendered
    np.array
        Indeces of locs to be rendered
    """

    n_pixel_y = int(_np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
    x = locs.x
    y = locs.y
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = _np.zeros((n_pixel_y, n_pixel_x), dtype=_np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@_numba.njit
def _render_setup3d(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, z_min, z_max, 
    pixelsize,
):
    """
    Finds coordinates to be rendered in 3D and sets up an empty image 
    array. Used by Picasso: Average3.

    Parameters
    ----------
    locs : np.recarray
        Localizations
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    z_min : float
        Minimum z coordinate to be rendered (nm)
    z_max : float
        Maximum z coordinate to be rendered (nm)
    pixelsize : float
        Camera pixel size, used for converting z coordinates

    Returns
    -------
    np.array
        Empty image array
    int
        Number of pixels in y
    int 
        Number of pixels in x
    int
        Number of pixels in z
    np.array 
        x coordinates to be rendered
    np.array 
        y coordinates to be rendered
    np.array
        z coordinates to be rendered
    np.array
        Indeces of locs to be rendered
    """

    n_pixel_y = int(_np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
    n_pixel_z = int(_np.ceil(oversampling * (z_max - z_min)))
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
    image = _np.zeros((n_pixel_y, n_pixel_x, n_pixel_z), dtype=_np.float32)
    return image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view

# @_numba.njit
# def _render_setupz(
#     locs, oversampling, x_min, z_min, x_max, z_max
# ):
#     n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
#     n_pixel_z = int(_np.ceil(oversampling * (z_max - z_min)))
#     x = locs.x
#     z = locs.z
#     in_view = (x > x_min) & (z > z_min) & (x < x_max) & (z < z_max)
#     x = x[in_view]
#     z = z[in_view]
#     x = oversampling * (x - x_min)
#     z = oversampling * (z - z_min)
#     image = _np.zeros((n_pixel_x, n_pixel_z), dtype=_np.float32)
#     return image, n_pixel_z, n_pixel_x, x, z, in_view


@_numba.njit
def _fill(image, x, y):
    """
    Fills image with x and y coordinates. 
    Image is not blurred.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    """

    x = x.astype(_np.int32)
    y = y.astype(_np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@_numba.njit
def _fill3d(image, x, y, z):
    """
    Fills image with x, y and z coordinates.
    Image is not blurred.
    Used by Picasso: Average3.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    z : np.array
        z coordinates to be rendered
    """

    x = x.astype(_np.int32)
    y = y.astype(_np.int32)
    z = z.astype(_np.int32)
    z += _np.min(z) # because z takes also negative values
    for i, j, k in zip(x, y, z):
        image[j, i, k] += 1


@_numba.njit
def _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y):
    """
    Fills image with blurred x and y coordinates.
    Localization precisions (sx and sy) are treated as standard
    deviations of the guassians to be rendered.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    sx : np.array
        Localization precision in x for each loc
    sy : np.array
        Localization precision in y for each loc
    n_pixel_x : int
        Number of pixels in x
    n_pixel_y : int
        Number of pixels in y
    """

    # render each localization separately
    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):

        # get min and max indeces to draw the given localization
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = _np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = _np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = _np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = _np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        # draw a localization as a 2D guassian PDF
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += _np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * _np.pi * sx_ * sy_)


@_numba.njit
def _fill_gaussian_rot(
    image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang
):
    """
    Fills image with rotated gaussian-blurred localizations.

    Localization precisions (sx, sy and sz) are treated as standard
    deviations of the guassians to be rendered.

    See https://cs229.stanford.edu/section/gaussians.pdf

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    z: np.array
        z coordinates to be rendered
    sx : np.array
        Localization precision in x for each loc
    sy : np.array
        Localization precision in y for each loc
    sz : np.array
        Localization precision in z for each loc
    n_pixel_x : int
        Number of pixels in x
    n_pixel_y : int
        Number of pixels in y
    ang : tuple
        Rotation angles of locs around x, y and z axes.
    """

    (angx, angy, angz) = ang # rotation angles

    rot_mat_x = _np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, _np.cos(angx), _np.sin(angx)],
            [0.0, -_np.sin(angx), _np.cos(angx)],
        ], dtype=_np.float32
    ) # rotation matrix around x axis
    rot_mat_y = _np.array(
        [
            [_np.cos(angy), 0.0, _np.sin(angy)],
            [0.0, 1.0, 0.0],
            [-_np.sin(angy), 0.0, _np.cos(angy)],
        ], dtype=_np.float32
    ) # rotation matrix around y axis
    rot_mat_z = _np.array(
        [
            [_np.cos(angz), -_np.sin(angz), 0.0],
            [_np.sin(angz), _np.cos(angz), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=_np.float32
    ) # rotation matrix around z axis
    # rot_matrix = _np.matmul(
    #     _np.matmul(
    #         rot_mat_x, rot_mat_y, dtype=_np.float32
    #     ), rot_mat_z, dtype=_np.float32
    # )
    rot_matrix = rot_mat_x @ rot_mat_y @ rot_mat_z # rotation matrix
    rot_matrixT = _np.transpose(rot_matrix) # ...and its transpose

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
        cov_matrix = _np.array(
            [
                [sx_**2, 0, 0], 
                [0, sy_**2, 0], 
                [0, 0, sz_**2],
            ], dtype=_np.float32
        ) # covariance matrix (CM)
        cov_rot = rot_matrix @ cov_matrix @ rot_matrixT # rotated CM
        cri = inverse_3x3(cov_rot) # inverse of rotated CM
        dcr = determinant_3x3(cov_rot) # determinant of rotated CM

        # draw a localization in 2D - sum z coordinates;
        # PDF of a rotated gaussian in 3D is calculated and 
        # image is summed over z axis
        for i in range(i_min, i_max):
            b = _np.float32(i + 0.5 - y_)
            for j in range(j_min, j_max):
                a = _np.float32(j + 0.5 - x_)
                for k in range(k_min, k_max):   
                    c = _np.float32(k + 0.5 - z_)
                    exponent = (
                        a*a * cri[0,0] + a*b * cri[0,1] + a*c * cri[0,2] 
                        + a*b * cri[1,0] + b*b * cri[1,1] + b*c * cri[1,2]
                        + a*c * cri[2,0] + b*c * cri[2,1] + c*c * cri[2,2]
                    ) # Mahalanobis distance
                    image[i,j] += (
                        _np.exp(-0.5 * exponent)
                        / (((2*_np.pi) ** 3 * dcr) ** 0.5)
                    )


@_numba.njit
def inverse_3x3(a):
    """
    Calculates inverse of a 3x3 matrix.

    This function is faster than np.linalg.inv.

    Parameters
    ----------
    a : np.array
        3x3 matrix

    Returns
    -------
    np.array
        Inverse of a
    """

    c = _np.zeros((3,3), dtype=_np.float32)
    det = determinant_3x3(a)

    c[0,0] = (a[1,1] * a[2,2] - a[1,2] * a[2,1]) / det
    c[0,1] = (a[0,2] * a[2,1] - a[0,1] * a[2,2]) / det
    c[0,2] = (a[0,1] * a[1,2] - a[0,2] * a[1,1]) / det

    c[1,0] = (a[1,2] * a[2,0] - a[1,0] * a[2,2]) / det
    c[1,1] = (a[0,0] * a[2,2] - a[0,2] * a[2,0]) / det
    c[1,2] = (a[0,2] * a[1,0] - a[0,0] * a[1,2]) / det

    c[2,0] = (a[1,0] * a[2,1] - a[1,1] * a[2,0]) / det
    c[2,1] = (a[0,1] * a[2,0] - a[0,0] * a[2,1]) / det
    c[2,2] = (a[0,0] * a[1,1] - a[0,1] * a[1,0]) / det

    return c    


@_numba.njit
def determinant_3x3(a):
    """
    Calculates determinant of a 3x3 matrix.

    This function is faster than np.linalg.det.

    Parameters
    ----------
    a : np.array
        3x3 matrix

    Returns
    -------
    float
        Determinant of a
    """

    return _np.float32(
        a[0,0] * (a[1,1] * a[2,2] - a[1,2] * a[2,1]) 
        - a[0,1] * (a[1,0] * a[2,2] - a[2,0] * a[1,2]) 
        + a[0,2] * (a[1,0] * a[2,1] - a[2,0] * a[1,1])
    )


def render_hist(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, 
    ang=None, 
):
    """
    Renders locs with no blur.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None, 
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
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
    return len(x), image


# @_numba.jit(nopython=True, nogil=True)
# def render_histz(locs, oversampling, x_min, z_min, x_max, z_max):
#     image, n_pixel_z, n_pixel_x, x, z, in_view = _render_setupz(
#         locs, oversampling, x_min, z_min, x_max, z_max
#     )
#     _fill(image, z, x)
#     return len(x), image


@_numba.jit(nopython=True, nogil=True)
def render_hist3d(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, z_min, z_max, 
    pixelsize,
):
    """
    Renders locs in 3D with no blur.
    Used by Picasso: Average3.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    z_min : float
        Minimum z coordinate to be rendered (nm)
    z_max : float
        Maximum z coordinate to be rendered (nm)
    pixelsize : float
        Camera pixel size, used for converting z coordinates

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered 3D image
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
    return len(x), image


def render_gaussian(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, 
    min_blur_width, 
    ang=None, 
):
    """
    Renders locs with with individual localization precision which 
    differs in x and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None, 
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, 
        oversampling, 
        y_min, x_min, y_max, x_max,
    )

    if not ang: # not rotated
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        sy = blur_height[in_view]
        sx = blur_width[in_view]
    
        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else: # rotated
        x, y, in_view, z = locs_rotation(
            locs, 
            oversampling, 
            x_min, x_max, y_min, y_max, 
            ang, 
        )
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        # for now, let lpz be twice the mean of lpx and lpy (TODO):
        if hasattr(locs, "lpz"):
            lpz = locs.lpz #NOTE: lpz must be in the same units as lpx
        else:
            lpz = 2 * _np.mean(_np.stack((locs.lpx, locs.lpy)), axis=0)
        blur_depth = oversampling * _np.maximum(lpz, min_blur_width)

        sy = blur_height[in_view]
        sx = blur_width[in_view]
        sz = blur_depth[in_view]

        _fill_gaussian_rot(
            image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang
        )

    return len(x), image


def render_gaussian_iso(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, 
    min_blur_width, 
    ang=None, 
):
    """
    Renders locs with with individual localization precision which 
    is the same in x and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None, 
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, 
        oversampling, 
        y_min, x_min, y_max, x_max,
    )

    if not ang: # not rotated
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
    
        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else: # rotated
        x, y, in_view, z = locs_rotation(
            locs, 
            oversampling, 
            x_min, x_max, y_min, y_max, 
            ang, 
        )
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        # for now, let lpz be twice the mean of lpx and lpy (TODO):
        if hasattr(locs, "lpz"):
            lpz = locs.lpz #NOTE: lpz must be in the same units as lpx
        else:
            lpz = 2 * _np.mean(_np.stack((locs.lpx, locs.lpy)), axis=0)
        blur_depth = oversampling * _np.maximum(lpz, min_blur_width)

        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
        sz = blur_depth[in_view]

        _fill_gaussian_rot(
            image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang
        )

    return len(x), image


def render_convolve(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, 
    min_blur_width, 
    ang=None, 
):
    """
    Renders locs with with global localization precision, i.e. each
    localization is blurred by the median localization precision in x
    and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None, 
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, 
        oversampling, 
        y_min, x_min, y_max, x_max,
    )
    if ang: # rotate
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
            _np.median(locs.lpx[in_view]), min_blur_width
        )
        blur_height = oversampling * max(
            _np.median(locs.lpy[in_view]), min_blur_width
        )
        return n, _fftconvolve(image, blur_width, blur_height)


def render_smooth(
    locs, 
    oversampling, 
    y_min, x_min, y_max, x_max, 
    ang=None, 
):
    """
    Renders locs with with blur of one display pixel (set by 
    oversampling)

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None, 
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
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


def _fftconvolve(image, blur_width, blur_height): 
    """
    Blurs (convolves) 2D image using fast fourier transform.

    Parameters
    ----------
    image : np.array
        Image with renderd but not blurred locs
    blur_width : float
        Blur width
    blur_height
        Blur height

    Returns
    -------
    np.array
        Blurred image
    """

    kernel_width = 10 * int(_np.round(blur_width)) + 1
    kernel_height = 10 * int(_np.round(blur_height)) + 1
    kernel_y = _signal.gaussian(kernel_height, blur_height)
    kernel_x = _signal.gaussian(kernel_width, blur_width)
    kernel = _np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    return _signal.fftconvolve(image, kernel, mode="same")


def rotation_matrix(angx, angy, angz):
    """
    Finds rotation matrix given rotation angles around axes.

    Parameters
    ----------
    angx : float
        Rotation angle around x axis
    angy : float
        Rotation angle around y axis
    angz : float
        Rotation angle around z axis

    Returns
    -------
    scipy.spatial.transform.Rotation
        Scipy class that can be applied to rotate an Nx3 np.array
    """

    rot_mat_x = _np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, _np.cos(angx), _np.sin(angx)],
            [0.0,-_np.sin(angx), _np.cos(angx)],
        ]
    ) # rotation matrix around x axis
    rot_mat_y = _np.array(
        [
            [_np.cos(angy), 0.0, _np.sin(angy)],
            [0.0, 1.0, 0.0],
            [-_np.sin(angy), 0.0, _np.cos(angy)],
        ]
    ) # rotation matrix around y axis
    rot_mat_z = _np.array(
        [
            [_np.cos(angz), -_np.sin(angz), 0.0],
            [_np.sin(angz), _np.cos(angz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ) # rotation matrix around z axis
    rot_mat = rot_mat_x @ rot_mat_y @ rot_mat_z
    return _Rotation.from_matrix(rot_mat)


def locs_rotation(
    locs, 
    oversampling,
    x_min, x_max, y_min, y_max, 
    ang, 
):
    """
    Rotates localizations within a FOV.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rotated
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    ang : tuple
        Rotation angles of locs around x, y and z axes.

    Returns
    -------
    np.array
        New (rotated) x coordinates
    np.array
        New y coordinates
    np.array
        Indeces of locs that are rendered
    np.array
        New z coordinates
    """

    # z is translated to pixels
    locs_coord = _np.stack((locs.x ,locs.y, locs.z)).T

    # x and y are in range (x_min/y_min, x_max/y_max) so they need to be
    # shifted (scipy rotation is around origin)
    locs_coord[:, 0] -= x_min + (x_max - x_min) / 2
    locs_coord[:, 1] -= y_min + (y_max - y_min) / 2

    # rotate locs
    R = rotation_matrix(ang[0], ang[1], ang[2])
    locs_coord = R.apply(locs_coord)

    # unshift locs
    locs_coord[:,0] += x_min + (x_max - x_min) / 2
    locs_coord[:,1] += y_min + (y_max - y_min) / 2

    # output
    x = locs_coord[:,0]
    y = locs_coord[:,1]
    z = locs_coord[:,2]
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]        
    y = y[in_view]
    z = z[in_view] 
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    z *= oversampling
    return x, y, in_view, z
