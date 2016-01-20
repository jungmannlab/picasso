"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :author: Joerg Schnitzbauer, 2015
"""
import numpy as _np
import numba as _numba
import scipy.signal as _signal


def render(locs, info, oversampling=1, viewport=None, blur_method=None):
    if blur_method is None:
        image = bin_locs(locs, info, oversampling, viewport)
    elif blur_method == 'convolve':
        image = bin_locs(locs, info, oversampling, viewport)
        lpy = oversampling * _np.median(locs.lpy[_np.isfinite(locs.lpy)])
        lpx = oversampling * _np.median(locs.lpx[_np.isfinite(locs.lpx)])
        kernel_height = 10 * int(_np.round(lpy)) + 1
        kernel_width = 10 * int(_np.round(lpx)) + 1
        kernel_y = _signal.gaussian(kernel_height, lpy)
        kernel_x = _signal.gaussian(kernel_width, lpx)
        kernel = _np.outer(kernel_y, kernel_x)
        image = _signal.fftconvolve(image, kernel, mode='same')
        image = len(locs) * image / image.sum()
        image[_np.logical_not(_np.isfinite(image))] = 0
    return image


def bin_locs(locs, info, oversampling=1, viewport=None):
    minfo = info[0]
    if viewport is None:
        viewport = [(0, 0), (minfo['Width'], minfo['Height'])]
    (y_min, x_min), (y_max, x_max) = viewport
    in_view = (locs.x > x_min) & (locs.y > y_min) & (locs.x < x_max) & (locs.y < y_max)
    x = locs.x[in_view] - x_min
    y = locs.y[in_view] - y_min
    x = _np.int32(oversampling * x)
    y = _np.int32(oversampling * y)
    n_pixel_x = int(_np.round(oversampling * (y_max - y_min)))
    n_pixel_y = int(_np.round(oversampling * (x_max - x_min)))
    return _bin_locs(x, y, n_pixel_x, n_pixel_y)


@_numba.jit(nopython=True)
def _bin_locs(x, y, n_pixel_x, n_pixel_y):
    image = _np.zeros((n_pixel_x, n_pixel_y), dtype=_np.float32)
    for i, j in zip(x, y):
        image[j, i] += 1
    return image
