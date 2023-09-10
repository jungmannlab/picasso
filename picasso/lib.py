"""
    picasso/lib
    ~~~~~~~~~~~~~~~~~~~~

    Handy functions and classes

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""


import numba as _numba
import numpy as _np
from lmfit import Model as _Model
from numpy.lib.recfunctions import append_fields as _append_fields
from numpy.lib.recfunctions import drop_fields as _drop_fields
import collections as _collections
import glob as _glob
import os.path as _ospath
from picasso import io as _io
from PyQt5 import QtCore, QtWidgets

# A global variable where we store all open progress and status dialogs.
# In case of an exception, we close them all,
# so that the GUI remains responsive.
_dialogs = []


class ProgressDialog(QtWidgets.QProgressDialog):
    def __init__(self, description, minimum, maximum, parent):
        super().__init__(
            description,
            None,
            minimum,
            maximum,
            parent,
            QtCore.Qt.CustomizeWindowHint,
        )
        self.initalized = None

    def init(self):
        _dialogs.append(self)
        self.setMinimumDuration(500)
        self.setModal(True)
        self.app = QtCore.QCoreApplication.instance()
        self.initalized = True

    def set_value(self, value):
        if not self.initalized:
            self.init()
        self.setValue(value)
        self.app.processEvents()

    def closeEvent(self, event):
        _dialogs.remove(self)


class StatusDialog(QtWidgets.QDialog):
    def __init__(self, description, parent):
        super(StatusDialog, self).__init__(parent, QtCore.Qt.CustomizeWindowHint)
        _dialogs.append(self)
        vbox = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(description)
        vbox.addWidget(label)
        self.show()
        QtCore.QCoreApplication.instance().processEvents()

    def closeEvent(self, event):
        _dialogs.remove(self)


class AutoDict(_collections.defaultdict):
    """
    A defaultdict whose auto-generated values are defaultdicts itself.
    This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


def cancel_dialogs():
    dialogs = [_ for _ in _dialogs]
    for dialog in dialogs:
        if isinstance(dialog, ProgressDialog):
            dialog.cancel()
        else:
            dialog.close()
    QtCore.QCoreApplication.instance().processEvents()  # just in case...


def cumulative_exponential(x, a, t, c):
    return a * (1 - _np.exp(-(x / t))) + c


CumulativeExponentialModel = _Model(cumulative_exponential)


def calculate_optimal_bins(data, max_n_bins=None):
    iqr = _np.subtract(*_np.percentile(data, [75, 25]))
    bin_size = 2 * iqr * len(data) ** (-1 / 3)
    if data.dtype.kind in ("u", "i") and bin_size < 1:
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
    return _append_fields(
        rec_array,
        name,
        data,
        dtypes=data.dtype,
        usemask=False,
        asrecarray=True,
    )
    return rec_array

def ensure_sanity(locs, info):
    # no inf or nan:
    locs = locs[
        _np.all(
            _np.array([_np.isfinite(locs[_]) for _ in locs.dtype.names]),
            axis=0,
        )
    ]
    # other sanity checks:
    locs = locs[locs.x > 0]
    locs = locs[locs.y > 0]
    locs = locs[locs.x < info[0]["Width"]]
    locs = locs[locs.y < info[0]["Height"]]
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


@_numba.jit(nopython=True)
def check_if_in_rectangle(x, y, X, Y):
    """
    Checks if locs with coordinates (x, y) are in rectangle with corners (X, Y)
    by counting the number of rectangle sides which are hit by a ray
    originating from each loc to the right. If the number of hit rectangle
    sides is odd, then the loc is in the rectangle
    """
    n_locs = len(x)
    ray_hits_rectangle_side = _np.zeros((n_locs, 4))
    for i in range(4):
        # get two y coordinates of corner points forming one rectangle side
        y_corner_1 = Y[i]
        # take the first if we're at the last side:
        y_corner_2 = Y[0] if i == 3 else Y[i + 1]
        y_corners_min = min(y_corner_1, y_corner_2)
        y_corners_max = max(y_corner_1, y_corner_2)
        for j in range(n_locs):
            y_loc = y[j]
            # only if loc is on level of rectangle side, its ray can hit:
            if y_corners_min <= y_loc <= y_corners_max:
                x_corner_1 = X[i]
                # take the first if we're at the last side:
                x_corner_2 = X[0] if i == 3 else X[i + 1]
                # calculate intersection point of ray and side:
                m_inv = (x_corner_2 - x_corner_1) / (y_corner_2 - y_corner_1)
                x_intersect = m_inv * (y_loc - y_corner_1) + x_corner_1
                x_loc = x[j]
                if x_intersect >= x_loc:
                    # ray hits rectangle side on the right side
                    ray_hits_rectangle_side[j, i] = 1
    n_sides_hit = _np.sum(ray_hits_rectangle_side, axis=1)
    is_in_rectangle = n_sides_hit % 2 == 1
    return is_in_rectangle


def locs_in_rectangle(locs, X, Y):
    is_in_rectangle = check_if_in_rectangle(locs.x, locs.y, _np.array(X), _np.array(Y))
    return locs[is_in_rectangle]


def minimize_shifts(shifts_x, shifts_y, shifts_z=None):
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = _np.zeros((n_pairs, n_dims))
    A = _np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
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


def locs_glob_map(func, pattern, args=[], kwargs={}, extension=""):
    """
    Maps a function to localization files, specified by a unix style path
    pattern.
    The function must take two arguments: locs and info. It may take additional
    args and kwargs which are supplied to this map function.
    A new locs file will be saved if an extension is provided. In that case the
    mapped function must return new locs and a new info dict.
    """
    paths = _glob.glob(pattern)
    for path in paths:
        locs, info = _io.load_locs(path)
        result = func(locs, info, path, *args, **kwargs)
        if extension:
            base, ext = _ospath.splitext(path)
            out_path = base + "_" + extension + ".hdf5"
            locs, info = result
            _io.save_locs(out_path, locs, info)
