"""
    picasso/lib
    ~~~~~~~~~~~~~~~~~~~~

    Handy functions and classes

    :author: Joerg Schnitzbauer, 2016
"""


import numpy as _np
from numpy.lib.recfunctions import append_fields as _append_fields


def calculate_optimal_bins(data, max_n_bins=None):
    iqr = _np.subtract(*_np.percentile(data, [75, 25]))
    bin_size = 2 * iqr * len(data)**(-1/3)
    if data.dtype.kind in ('u', 'i') and bin_size < 1:
        bin_size = 1
    bin_min = max(data.min() - bin_size / 2, 0)
    n_bins = int(_np.ceil((data.max() - bin_min) / bin_size))
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    return _np.linspace(bin_min, data.max(), n_bins)


def xcorr_fft(A, B):
    return _np.fft.fftshift(_np.real(_np.fft.ifft2(_np.fft.fft2(A)*_np.conj(_np.fft.fft2(B)))))


def append_to_rec(rec_array, data, name):
    return _append_fields(rec_array, name, data, dtypes=data.dtype, usemask=False, asrecarray=True)
