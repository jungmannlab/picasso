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
from picasso import io as _io
from PyQt4 import QtGui, QtCore
from lmfit import Model as _Model


class ProgressDialog(QtGui.QProgressDialog):

    def __init__(self, description, minimum, maximum, parent):
        super().__init__(description, None, minimum, maximum, parent, QtCore.Qt.CustomizeWindowHint)
        self.setMinimumDuration(500)
        self.setModal(True)
        self.app = QtCore.QCoreApplication.instance()

    def set_value(self, value):
        self.setValue(value)
        self.app.processEvents()


class StatusDialog(QtGui.QDialog):

    def __init__(self, description, parent):
        super(StatusDialog, self).__init__(parent, QtCore.Qt.CustomizeWindowHint)
        vbox = QtGui.QVBoxLayout(self)
        label = QtGui.QLabel(description)
        vbox.addWidget(label)
        self.show()
        QtCore.QCoreApplication.instance().processEvents()


class AutoDict(_collections.defaultdict):
    '''
    A defaultdict whose auto-generated values are defaultdicts itself.
    This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


def cumulative_exponential(x, a, t, c):
    return a * (1 - _np.exp(-(x/t))) + c


CumulativeExponentialModel = _Model(cumulative_exponential)


def calculate_optimal_bins(data, max_n_bins=None):
    iqr = _np.subtract(*_np.percentile(data, [75, 25]))
    bin_size = 2 * iqr * len(data)**(-1/3)
    if data.dtype.kind in ('u', 'i') and bin_size < 1:
        bin_size = 1
    bin_min = data.min() - bin_size / 2
    n_bins = _np.ceil((data.max() - bin_min) / bin_size)
    try:
        n_bins = int(n_bins)
    except ValueError:
        return None
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    return _np.linspace(bin_min, data.max(), n_bins)


def append_to_rec(rec_array, data, name):
    if hasattr(rec_array, name):
        rec_array = remove_from_rec(rec_array, name)
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


def is_loc_at(x, y, locs, r):
    dx = locs.x - x
    dy = locs.y - y
    r2 = r**2
    return dx**2 + dy**2 < r2


def locs_at(x, y, locs, r):
    is_picked = is_loc_at(x, y, locs, r)
    return locs[is_picked]


def minimize_shifts(shifts_x, shifts_y, shifts_z=None):
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = _np.zeros((n_pairs, n_dims))
    A = _np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i+1, n_channels):
            rij[flag, 0] = shifts_y[i, j]
            rij[flag, 1] = shifts_x[i, j]
            if n_dims == 3:
                rij[flag, 2] = shifts_z[i, j]
            A[flag, i:j] = 1
            flag += 1
    Dj = _np.dot(_np.linalg.pinv(A), rij)
    shift_y = _np.insert(_np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = _np.insert(_np.cumsum(Dj[:, 1]), 0, 0)
    if n_dims == 2:
        return shift_y, shift_x
    else:
        shift_z = _np.insert(_np.cumsum(Dj[:, 2]), 0, 0)
        return shift_y, shift_x, shift_z


def n_futures_done(futures):
    return sum([_.done() for _ in futures])


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
