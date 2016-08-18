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
import collections as _collections
import glob as _glob
import os.path as _ospath
import numba as _numba
from picasso import io as _io


class AutoDict(_collections.defaultdict):
    '''
    A defaultdict whose auto-generated values are defaultdicts itself.
    This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


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


def ensure_sanity(locs, info):
    # no inf or nan:
    locs = locs[_np.all(_np.array([_np.isfinite(locs[_]) for _ in locs.dtype.names]), axis=0)]
    # other sanity checks:
    locs = locs[locs.x > 0]
    locs = locs[locs.y > 0]
    locs = locs[locs.x < info[0]['Width']]
    locs = locs[locs.y < info[0]['Height']]
    locs = locs[locs.lpx > 0]
    locs = locs[locs.lpy > 0]
    return locs


def locs_at(x, y, locs, r):
    dx = locs.x - x
    dy = locs.y - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return locs[is_picked]


def remove_from_rec(rec_array, name):
    return _drop_fields(rec_array, name, usemask=False, asrecarray=True)


def locs_glob_map(func, pattern, args=[], kwargs={}, extension=''):
    '''
    Maps a function to localization files, specified by a unix style path pattern.
    The function must take two arguments: locs and info. It may take additional args and kwargs which
    are supplied to this map function.
    A new locs file will be saved if an extension is provided. In that case the mapped function must return
    new locs and a new info dict.
    '''
    paths = _glob.glob(pattern)
    for path in paths:
        locs, info = _io.load_locs(path)
        result = func(locs, info, path, *args, **kwargs)
        if extension:
            base, ext = _ospath.splitext(path)
            out_path = base + '_' + extension + '.hdf5'
            locs, info = result
            _io.save_locs(out_path, locs, info)
