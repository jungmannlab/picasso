"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, Max Planck Institute of Biochemistry
"""
import numpy as _np
import numba as _numba
import scipy.signal as _signal
from tqdm import trange as _trange


_DRAW_MAX_SIGMA = 3


def render(locs, info=None, oversampling=1, viewport=None, blur_method=None, min_blur_width=0):
    if viewport is None:
        try:
            viewport = [(0, 0), (info[0]['Height'], info[0]['Width'])]
        except TypeError:
            raise ValueError('Need info if no viewport is provided.')
    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        return render_hist(locs, oversampling, y_min, x_min, y_max, x_max)
    elif blur_method == 'gaussian':
        return render_gaussian(locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width)
    elif blur_method == 'smooth':
        return render_smooth(locs, oversampling, y_min, x_min, y_max, x_max)
    elif blur_method == 'convolve':
        return render_convolve(locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width)
    else:
        raise Exception('blur_method not understood.')


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
def _fill(image, x, y):
    # x = _np.int32(x)
    # y = _np.int32(y)
    x = x.astype(_np.int32)
    y = y.astype(_np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@_numba.jit(nopython=True, nogil=True)
def _gaussian_kernel(size, sigma):
    x = _np.arange(size)
    x0 = 0.5 * (size - 1)
    return _np.exp(-(x-x0)**2/(2*sigma**2))


@_numba.jit(nopython=True, nogil=True)
def _outer(a, b):
    n = len(a)
    out = _np.zeros((n, n), dtype=_np.float32)
    for i in range(n):
        for j in range(n):
            out[i, j] = a[i]*b[j]
    return out


@_numba.jit(nopython=True, nogil=True)
def render_hist(locs, oversampling, y_min, x_min, y_max, x_max):
    image, n_pixel_y, n_pixel_x, x, y, _ = _render_setup(locs, oversampling, y_min, x_min, y_max, x_max)
    _fill(image, x, y)
    return len(x), image


@_numba.jit(nopython=True, nogil=True)
def render_gaussian(locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(locs, oversampling, y_min, x_min, y_max, x_max)
    blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
    blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
    sy = blur_height[in_view]
    sx = blur_width[in_view]
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
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += _np.exp(-((j - x_ + 0.5)**2/(2 * sx_**2) + (i - y_ + 0.5)**2/(2 * sy_**2))) / (2 * _np.pi * sx_ * sy_)
    return len(x), image


def render_convolve(locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width):
    n, image, kernel = _render_convolve(locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width)
    image = _signal.fftconvolve(image, kernel, mode='same')
    return n, image


@_numba.jit(nopython=True, nogil=True)
def _render_convolve(locs, oversampling, y_min, x_min, y_max, x_max, min_blur_width):
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(locs, oversampling, y_min, x_min, y_max, x_max)
    _fill(image, x, y)
    blur_width = oversampling * max(_np.median(locs.lpx[in_view]), min_blur_width)
    blur_height = oversampling * max(_np.median(locs.lpy[in_view]), min_blur_width)
    kernel_width = 10 * int(_np.round(blur_width)) + 1
    kernel_height = 10 * int(_np.round(blur_height)) + 1
    kernel_y = _gaussian_kernel(kernel_height, blur_height)
    kernel_x = _gaussian_kernel(kernel_width, blur_width)
    kernel = _outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    return len(x), image, kernel


def render_smooth(locs, oversampling, y_min, x_min, y_max, x_max):
    n, image, kernel = _render_smooth(locs, oversampling, y_min, x_min, y_max, x_max)
    image = _signal.fftconvolve(image, kernel, mode='same')
    return n, image


@_numba.jit(nopython=True, nogil=True)
def _render_smooth(locs, oversampling, y_min, x_min, y_max, x_max):
    n, image = render_hist(locs, oversampling, y_min, x_min, y_max, x_max)
    kernel = _gaussian_kernel(11, 1)
    kernel = _outer(kernel, kernel)
    kernel /= kernel.sum()
    return n, image, kernel


def segment(locs, info, segmentation, kwargs={}, callback=None):
    Y = info[0]['Height']
    X = info[0]['Width']
    n_frames = info[0]['Frames']
    n_seg = n_segments(info, segmentation)
    bounds = _np.linspace(0, n_frames-1, n_seg+1, dtype=_np.uint32)
    segments = _np.zeros((n_seg, Y, X))
    if callback is not None:
        callback(0)
    for i in _trange(n_seg, desc='Generating segments', unit='segments'):
        segment_locs = locs[(locs.frame >= bounds[i]) & (locs.frame < bounds[i+1])]
        _, segments[i] = render(segment_locs, info, **kwargs)
        if callback is not None:
            callback(i+1)
    return bounds, segments


def n_segments(info, segmentation):
    n_frames = info[0]['Frames']
    return int(_np.round(n_frames/segmentation))
