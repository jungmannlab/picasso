"""
    picasso.render
    ~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :author: Joerg Schnitzbauer, 2015
"""
import numpy as _np
import numba as _numba
import scipy.signal as _signal


def render(locs, info, oversampling='auto', blur_method='convolve'):
    if oversampling == 'auto':
        auto_oversample = True
        lpy = _np.median(locs.lpy[_np.isfinite(locs.lpy)])
        lpx = _np.median(locs.lpx[_np.isfinite(locs.lpx)])
        lp = (lpx + lpy) / 2
        oversampling = 2 / lp
    else:
        auto_oversample = False
    if blur_method == 'convolve':
        image = bin_locs(locs, info, oversampling)
        lpy = oversampling * _np.median(locs.lpy[_np.isfinite(locs.lpy)])
        lpx = oversampling * _np.median(locs.lpx[_np.isfinite(locs.lpx)])
        kernel_height = 10 * int(_np.round(lpy)) + 1
        kernel_width = 10 * int(_np.round(lpx)) + 1
        kernel_y = _signal.gaussian(kernel_height, lpy)
        kernel_x = _signal.gaussian(kernel_width, lpx)
        kernel = _np.outer(kernel_y, kernel_x)
        image = _signal.fftconvolve(image, kernel, mode='same')
        image = len(locs) * image / image.sum()
    if auto_oversample:
        return image, oversampling
    return image


def bin_locs(locs, info, oversampling):
    minfo = info[0]
    x = _np.int32(oversampling * locs.x)
    y = _np.int32(oversampling * locs.y)
    n_pixel_x = int(_np.ceil(oversampling * minfo['Height']))
    n_pixel_y = int(_np.ceil(oversampling * minfo['Width']))
    return _bin_locs(x, y, n_pixel_x, n_pixel_y)


@_numba.jit(nopython=True)
def _bin_locs(x, y, n_pixel_x, n_pixel_y):
    image = _np.zeros((n_pixel_x, n_pixel_y), dtype=_np.float32)
    for i, j in zip(x, y):
        if (0 <= i < n_pixel_x) and (0 <= j <= n_pixel_y):
            image[j, i] += 1
    return image
