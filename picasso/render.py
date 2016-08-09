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


_DRAW_MAX_SIGMA = 3


def render(locs, info=None, oversampling=1, viewport=None, blur_method=None, min_blur_width=0):
    if viewport is None:
        try:
            viewport = [(0, 0), (info[0]['Height'], info[0]['Width'])]
        except TypeError:
            raise ValueError('Need info if no viewport is provided.')
    (y_min, x_min), (y_max, x_max) = viewport
    n_pixel_y = int(_np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(_np.ceil(oversampling * (x_max - x_min)))
    x = locs.x
    y = locs.y
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    if _np.sum(in_view) == 0:
        image = _np.zeros((n_pixel_y, n_pixel_x), dtype=_np.float32)
        return 0, image
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    if blur_method is None:
        x = _np.int32(x)
        y = _np.int32(y)
        image = _np.zeros((n_pixel_y, n_pixel_x), dtype=_np.float32)
        return len(x), _fill(image, x, y)
    elif blur_method in ['smooth', 'convolve']:
        x = _np.int32(x)
        y = _np.int32(y)
        image = _np.zeros((n_pixel_y, n_pixel_x), dtype=_np.float32)
        image = _fill(image, x, y)
        if blur_method == 'smooth':
            blur_width = blur_height = 1
        else:
            blur_width = oversampling * max(_np.median(locs.lpx[in_view]), min_blur_width)
            blur_height = oversampling * max(_np.median(locs.lpy[in_view]), min_blur_width)
        kernel_width = 10 * int(_np.round(blur_width)) + 1
        kernel_height = 10 * int(_np.round(blur_height)) + 1
        kernel_y = _signal.gaussian(kernel_height, blur_height)
        kernel_x = _signal.gaussian(kernel_width, blur_width)
        kernel = _np.outer(kernel_y, kernel_x)
        kernel /= kernel.sum()
        image = _signal.fftconvolve(image, kernel, mode='same')
        return len(x), image
    elif blur_method == 'gaussian':
        blur_width = oversampling * _np.maximum(locs.lpx, min_blur_width)
        blur_height = oversampling * _np.maximum(locs.lpy, min_blur_width)
        sy = blur_width[in_view]
        sx = blur_width[in_view]
        return len(x), _fill_gaussians(n_pixel_x, n_pixel_y, x, y, sx, sy)
    else:
        raise Exception('blur_method not understood.')


@_numba.jit(nopython=True)
def _fill(image, x, y):
    for i, j in zip(x, y):
        image[j, i] += 1
    return image


@_numba.jit(nopython=True)
def _fill_gaussians(X, Y, x, y, sx, sy):
    image = _np.zeros((Y, X), dtype=_np.float32)
    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = _np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = _np.int32(y_ + max_y + 1)
        if i_max > Y:
            i_max = Y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = _np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = _np.int32(x_ + max_x) + 1
        if j_max > X:
            j_max = X
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += _np.exp(-((j - x_ + 0.5)**2/(2 * sx_**2) + (i - y_ + 0.5)**2/(2 * sy_**2))) / (2 * _np.pi * sx_ * sy_)
    return image
