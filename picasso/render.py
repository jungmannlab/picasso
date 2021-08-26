"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, MPI of Biochemistry
"""
import numpy as _np
import numba as _numba
import scipy.signal as _signal
from scipy.spatial.transform import Rotation
from tqdm import trange as _trange
from icecream import ic
import time
import pyximport
pyximport.install(setup_args={"include_dirs":_np.get_include()},
                  reload_support=True)
import sys
sys.path.append(sys.path[0] + "\\picasso")
from cfill_gaussian import cfill_gaussian_rot


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

    if viewport is None:
        try:
            viewport = [(0, 0), (info[0]["Height"], info[0]["Width"])]
        except TypeError:
            raise ValueError("Need info if no viewport is provided.")
    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        return render_hist(locs, oversampling, y_min, x_min, y_max, x_max, ang=ang)
    elif blur_method == "gaussian":
        return render_gaussian(
            locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width, ang=ang)
    elif blur_method == "gaussian_iso":
        return render_gaussian_iso(
            locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width, ang=ang)
    elif blur_method == "smooth":
        return render_smooth(locs, oversampling, y_min, x_min, y_max, x_max, ang=ang)
    elif blur_method == "convolve":
        return render_convolve(
            locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width, ang=ang)
    else:
        raise Exception("blur_method not understood.")


@_numba.jit(nopython=True, nogil=True)
def _render_setup(locs, oversampling, y_min, x_min, y_max, x_max):
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


@_numba.jit(nopython=True, nogil=True)
def _render_setup3d(
    locs, oversampling, y_min, x_min, y_max, x_max, z_min, z_max, pixelsize
):
    n_pixel_y = int(_np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
    n_pixel_z = int(_np.ceil(oversampling * (z_max - z_min) / pixelsize))
    x = locs.x
    y = locs.y
    z = locs.z
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
    z = oversampling * (z - z_min) / pixelsize
    image = _np.zeros((n_pixel_y, n_pixel_x, n_pixel_z), dtype=_np.float32)
    return image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view


@_numba.jit(nopython=True, nogil=True)
def _render_setupz(locs, oversampling, x_min, z_min, x_max, z_max, pixelsize):
    n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
    n_pixel_z = int(_np.ceil(oversampling * (z_max - z_min) / pixelsize))
    x = locs.x
    z = locs.z
    in_view = (x > x_min) & (z > z_min) & (x < x_max) & (z < z_max)
    x = x[in_view]
    z = z[in_view]
    x = oversampling * (x - x_min)
    z = oversampling * (z - z_min) / pixelsize
    image = _np.zeros((n_pixel_x, n_pixel_z), dtype=_np.float32)
    return image, n_pixel_z, n_pixel_x, x, z, in_view


@_numba.jit(nopython=True, nogil=True)
def _fill(image, x, y):
    x = x.astype(_np.int16)
    y = y.astype(_np.int16)
    for i, j in zip(x, y):
        image[j, i] += 1


@_numba.jit(nopython=True, nogil=True)
def _fill3d(image, x, y, z):
    x = x.astype(_np.int32)
    y = y.astype(_np.int32)
    z = z.astype(_np.int32)
    z += _np.min(z) # because z takes also negative values
    for i, j, k in zip(x, y, z):
        image[j, i, k] += 1
    return image

@_numba.jit(nopython=True, nogil=True)
def _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y):

    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):
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

        # mean_vec = _np.array([x_, y_])
        # cov_mat = _np.array([[sx_**2, 0], [0, sy_**2]])
        # cov_mat_inv = _np.linalg.inv(cov_mat)

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += _np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_ ** 2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_ ** 2)
                    )
                ) / (2 * _np.pi * sx_ * sy_)
                # coord = _np.array([j + 0.5, i + 0.5])
                # coord_rel = coord - mean_vec
                # coord_rel = _np.expand_dims(coord_rel, 1)
                # exponent = cov_mat_inv @ coord_rel
                # exponent1 = coord_rel.T @ exponent
                # temp = _np.exp(-0.5 * exponent1) / (2 * _np.pi * (_np.linalg.det(cov_mat))**0.5)
                # temp = temp.astype(_np.float32)
                # image[i,j] += temp[0, 0]

@_numba.jit(nopython=True, nogil=True)
def _fill_gaussian_rot(image, x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang):
    count = 0
    for x_, y_, z_, sx_, sy_, sz_ in zip(x, y, z, sx, sy, sz):
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
        j_max = _np.int32(x_ + max_x + 1)
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        max_z = _DRAW_MAX_SIGMA * sz_
        k_min = _np.int32(z_ - max_z)
        k_max = _np.int32(z_ + max_z + 1)

        angx = ang[0]
        angy = ang[1]
        mean_vec = _np.array([x_, y_, z_])
        cov_matrix = _np.array([[sx_**2, 0, 0], [0, sy_**2, 0], [0, 0, sz_**2]]) #covariance matrix 
        rot_mat_x = _np.array([[1.0,0.0,0.0],[0.0,_np.cos(angx),_np.sin(angx)],[0.0,-_np.sin(angx), _np.cos(angx)]])
        rot_mat_y = _np.array([[_np.cos(angy),0.0,_np.sin(angy)],[0.0,1.0,0.0],[-_np.sin(angy),0.0,_np.cos(angy)]])
        rot_matrix = rot_mat_x @ rot_mat_y
        cov_rot1 = cov_matrix @ rot_matrix
        cov_rot = rot_matrix.T @ cov_rot1
        cri = _np.linalg.inv(cov_rot) #stands for covariance rotated inverse
        count += 1

        if count % 50 == 0:
            print(count)

        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                a = j + 0.5 - x_
                b = i + 0.5 - y_
                for k in range(k_min, k_max):                   
                    c = k + 0.5 - z_
                    exponent = a**2 * cri[0,0] + a*b * cri[0,1] + a*c * cri[0,2] + \
                               a*b * cri[1,0] + b**2 * cri[1,1] + b*c * cri[1,2] + \
                               a*c * cri[2,0] + b*c * cri[2,1] + c**2 * cri[2,2]
                    image[i,j] += _np.exp(-0.5 * exponent) / (_np.sqrt((2*_np.pi)**3 * _np.linalg.det(cov_rot)))

def render_hist(locs, oversampling, y_min, x_min, y_max, x_max, ang=None):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, oversampling, y_min, x_min, y_max, x_max)
    if ang is not None:
        x, y, _ = locs_rotation(locs, x_min, x_max, y_min, y_max, oversampling, ang)
    _fill(image, x, y)
    return len(x), image


@_numba.jit(nopython=True, nogil=True)
def render_histz(locs, oversampling, x_min, z_min, x_max, z_max, pixelsize):
    image, n_pixel_z, n_pixel_x, x, z, in_view = _render_setupz(
        locs, oversampling, x_min, z_min, x_max, z_max, pixelsize
    )
    _fill(image, z, x)
    return len(x), image


@_numba.jit(nopython=True, nogil=True)
def render_hist3d(
    locs, oversampling, y_min, x_min, y_max, x_max, z_min, z_max, pixelsize
):
    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = _render_setup3d(
        locs, oversampling, y_min, x_min, y_max, x_max, z_min, z_max, pixelsize
    )
    _fill3d(image, x, y, z)
    return len(x), image


def render_gaussian(
    locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width, ang=None
):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, oversampling, y_min, x_min, y_max, x_max
    )

    if ang is None:
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        sy = blur_height[in_view]
        sx = blur_width[in_view]
    
        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:
        x, y, in_view, z = locs_rotation(locs, x_min, x_max, y_min, y_max, oversampling, ang, get_z=True)
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        blur_depth = oversampling * _np.maximum(locs.d_zcalib, min_blur_width)

        sy = blur_height[in_view]
        sx = blur_width[in_view]
        sz = blur_depth[in_view]

        image = cfill_gaussian_rot(x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang)

    return len(x), image


def render_gaussian_iso(
    locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width, ang=None
):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, oversampling, y_min, x_min, y_max, x_max
    )

    if ang is None:
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
    
        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:
        x, y, in_view, z = locs_rotation(locs, x_min, x_max, y_min, y_max, oversampling, ang, get_z=True)
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        blur_depth = oversampling * _np.maximum(locs.d_zcalib, min_blur_width)

        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
        sz = blur_depth[in_view]

        image = cfill_gaussian_rot(x, y, z, sx, sy, sz, n_pixel_x, n_pixel_y, ang)

    return len(x), image


def render_convolve(
    locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width, ang=None
):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, oversampling, y_min, x_min, y_max, x_max)

    if ang is not None:
        x, y, _ = locs_rotation(locs, x_min, x_max, y_min, y_max, oversampling, ang) 

    _fill(image, x, y)

    n = len(x)
    if n == 0:
        return 0, image
    else:
        blur_width = oversampling * max(
            _np.median(locs.lpx[in_view]), min_blur_width
        )
        blur_height = oversampling * max(
            _np.median(locs.lpy[in_view]), min_blur_width
        )
        return n, _fftconvolve(image, blur_width, blur_height)


def render_smooth(locs, oversampling, y_min, x_min, y_max, x_max, ang=None):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs, oversampling, y_min, x_min, y_max, x_max
    )

    if ang is not None:
        x, y, _= locs_rotation(locs, x_min, x_max, y_min, y_max, oversampling, ang)

    _fill(image, x, y)
    n = len(x)
    if n == 0:
        return 0, image
    else:
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(image, blur_width, blur_height): 
    kernel_width = 10 * int(_np.round(blur_width)) + 1
    kernel_height = 10 * int(_np.round(blur_height)) + 1
    kernel_y = _signal.gaussian(kernel_height, blur_height)
    kernel_x = _signal.gaussian(kernel_width, blur_width)
    kernel = _np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    return _signal.fftconvolve(image, kernel, mode="same")


def segment(locs, info, segmentation, kwargs={}, callback=None):
    Y = info[0]["Height"]
    X = info[0]["Width"]
    n_frames = info[0]["Frames"]
    n_seg = n_segments(info, segmentation)
    bounds = _np.linspace(0, n_frames - 1, n_seg + 1, dtype=_np.uint32)
    segments = _np.zeros((n_seg, Y, X))
    if callback is not None:
        callback(0)
    for i in _trange(n_seg, desc="Generating segments", unit="segments"):
        segment_locs = locs[
            (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
        ]
        _, segments[i] = render(segment_locs, info, **kwargs)
        if callback is not None:
            callback(i + 1)
    return bounds, segments


def n_segments(info, segmentation):
    n_frames = info[0]["Frames"]
    return int(_np.round(n_frames / segmentation))

def rotation_matrix(angx, angy, raw=False):
    #gives the rotation matrix which then can be applied to a (N,3) localization array
    rot_mat_x = _np.array([[1,0,0],[0,_np.cos(angx),_np.sin(angx)],[0,-_np.sin(angx), _np.cos(angx)]])
    rot_mat_y = _np.array([[_np.cos(angy),0,_np.sin(angy)],[0,1,0],[-_np.sin(angy),0,_np.cos(angy)]])
    rotation = Rotation.from_matrix(rot_mat_x @ rot_mat_y)
    if raw:
        return rot_mat_x @ rot_mat_y
    else:
        return rotation

def locs_rotation(locs, x_min, x_max, y_min, y_max, oversampling, ang, get_z=False):

    pixelsize = 130 #todo: make it work for other cameras
    locs_coord = _np.stack((locs.x ,locs.y, locs.z/pixelsize)).T

    locs_coord[:,0] -= x_min + (x_max-x_min)/2
    locs_coord[:,1] -= y_min + (y_max-y_min)/2

    R = rotation_matrix(ang[0], ang[1])
    locs_coord = R.apply(locs_coord)

    locs_coord[:,0] += x_min + (x_max-x_min)/2
    locs_coord[:,1] += y_min + (y_max-y_min)/2

    x = locs_coord[:,0]
    y = locs_coord[:,1]
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]        
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)

    if get_z:
        z = locs_coord[:,2]
        z = z[in_view] 
        z  = oversampling * (z - _np.min(z))
        return x, y, in_view, z
    else:
        return x, y, in_view