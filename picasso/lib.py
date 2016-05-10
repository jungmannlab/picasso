"""
    picasso/lib
    ~~~~~~~~~~~~~~~~~~~~

    Handy functions and classes

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, Max Planck Institute of Biochemistry
"""


import numpy as _np
from numpy.lib.recfunctions import append_fields as _append_fields
from numpy.lib.recfunctions import drop_fields as _drop_fields


def calculate_optimal_bins(data, max_n_bins=None):
    iqr = _np.subtract(*_np.percentile(data, [75, 25]))
    bin_size = 2 * iqr * len(data)**(-1/3)
    if data.dtype.kind in ('u', 'i') and bin_size < 1:
        bin_size = 1
    bin_min = data.min() - bin_size / 2
    n_bins = int(_np.ceil((data.max() - bin_min) / bin_size))
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    return _np.linspace(bin_min, data.max(), n_bins)


def append_to_rec(rec_array, data, name):
    return _append_fields(rec_array, name, data, dtypes=data.dtype, usemask=False, asrecarray=True)
    return rec_array


def ensure_finite(rec_array):
    return rec_array[_np.all(_np.array([_np.isfinite(rec_array[_]) for _ in rec_array.dtype.names]), axis=0)]


def remove_from_rec(rec_array, name):
    return _drop_fields(rec_array, name, usemask=False, asrecarray=True)
