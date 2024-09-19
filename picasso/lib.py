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
from numpy.lib.recfunctions import stack_arrays as _stack_arrays
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
    """ProgressDialog displays a progress dialog with a progress bar."""

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

    def zero_progress(self, description=None):
        """Sets progress dialog to zero and changes title if given."""

        if description:
            self.setLabelText(description)
        self.set_value(0)


class StatusDialog(QtWidgets.QDialog):
    """StatusDialog displays the description string in a dialog."""

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
    """Closes all open dialogs."""

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
    """Calculates the optimal bins for display.
    
    Parameters
    ----------
    data : numpy.1darray
        Data to be binned.
    max_n_bins : int (default=None)
        Maximum number of bins.
    
    Returns
    -------
    bins : numpy.1darray
        Bins for display.
    """
    
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
    bins = _np.linspace(bin_min, data.max(), n_bins)
    return bins


def append_to_rec(rec_array, data, name):
    """Appends a new column to the existing np.recarray.
    
    Parameters
    ----------
    rec_array : np.rec.array
        Recarray to which the new column is appended.
    data : np.1darray
        Data to be appended.
    name : str
        Name of the new column.
    
    Returns
    -------
    rec_array : np.rec.array
        Recarray with the new column.
    """
    
    if hasattr(rec_array, name):
        rec_array = remove_from_rec(rec_array, name)
    rec_array = _append_fields(
        rec_array,
        name,
        data,
        dtypes=data.dtype,
        usemask=False,
        asrecarray=True,
    )
    return rec_array


def merge_locs(locs_list, increment_frames=True):
    """Merges localization lists into one file. Can increment frames
    to avoid overlapping frames.
    
    Parameters
    ----------
    locs_list : list of np.rec.arrays
        List of localization lists to be merged.
    increment_frames : bool (default=True)
        If True, increments frames of each localization list by the
        maximum frame number of the previous localization list. Useful
        when the localization lists are from different movies but 
        represent the same stack.
    
    Returns
    locs : np.rec.array
        Merged localizations.
    """

    if increment_frames:
        last_frame = 0
        for i, locs in enumerate(locs_list):
            locs["frame"] += last_frame
            last_frame = locs["frame"][-1].max()
            locs_list[i] = locs    
    locs = _stack_arrays(locs_list, usemask=False, asrecarray=True)
    return locs


def ensure_sanity(locs, info):
    """Ensures that localizations are within the image dimensions
    and have positive localization precisions.
    
    Parameters
    ----------
    locs : np.rec.array
        Localizations list.
    info : list of dicts
        Localization metadata.
    
    Returns
    -------
    locs : np.rec.array
        Localizations that pass the sanity checks.
    """
    
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
    """Checks if localizations are at position (x, y) within radius r.
    
    Parameters
    ----------
    x : float
        x-coordinate of the position.
    y : float
        y-coordinate of the position.
    locs : np.rec.array
        Localizations list.
    r : float
        Radius.
    
    Returns
    -------
    is_picked : np.ndarray
        Boolean array indicating if localization is at position.
    """
    
    dx = locs.x - x
    dy = locs.y - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked


def locs_at(x, y, locs, r):
    """Returns localizations at position (x, y) within radius r.

    Parameters
    ----------
    x : float
        x-coordinate of the position.
    y : float
        y-coordinate of the position.
    locs : np.rec.array
        Localizations list.
    r : float   
        Radius.

    Returns 
    -------
    picked_locs : np.rec.array
        Localizations at position.
    """

    is_picked = is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    return picked_locs


@_numba.jit(nopython=True)
def check_if_in_polygon(x, y, X, Y):
    """Checks if points (x, y) are in polygon defined by corners (X, Y).
    Uses the ray casting algorithm, see check_if_in_rectangle for 
    details.
    
    Parameters
    ----------
    x : numpy.1darray
        x-coordinates of points.
    y : numpy.1darray
        y-coordinates of points.
    X : numpy.1darray
        x-coordinates of polygon corners.
    Y : numpy.1darray
        y-coordinates of polygon corners.
    
    Returns
    ------- 
    is_in_polygon : numpy.ndarray
        Boolean array indicating if point is in polygon.
    """

    n_locs = len(x)
    n_polygon = len(X)
    is_in_polygon = _np.zeros(n_locs, dtype=_np.bool_)

    for i in range(n_locs):
        count = 0
        for j in range(n_polygon):
            j_next = (j + 1) % n_polygon
            if (
                ((Y[j] > y[i]) != (Y[j_next] > y[i])) and
                (x[i] < X[j] + (X[j_next] - X[j]) * (y[i] - Y[j]) / (Y[j_next] - Y[j]))
            ):
                count += 1
        if count % 2 == 1:
            is_in_polygon[i] = True

    return is_in_polygon


def locs_in_polygon(locs, X, Y):
    """Returns localizations in polygon defined by corners (X, Y).
    
    Parameters
    ----------
    locs : numpy.recarray
        Localizations.
    X : list
        x-coordinates of polygon corners.
    Y : list
        y-coordinates of polygon corners.
    
    Returns
    -------
    picked_locs : numpy.recarray
        Localizations in polygon.
    """

    is_in_polygon = check_if_in_polygon(
        locs.x, locs.y, _np.array(X), _np.array(Y)
    )
    return locs[is_in_polygon]


@_numba.jit(nopython=True)
def check_if_in_rectangle(x, y, X, Y):
    """
    Checks if locs with coordinates (x, y) are in rectangle with corners (X, Y)
    by counting the number of rectangle sides which are hit by a ray
    originating from each loc to the right. If the number of hit rectangle
    sides is odd, then the loc is in the rectangle

    Parameters
    ----------
    x : numpy.1darray
        x-coordinates of points.
    y : numpy.1darray
        y-coordinates of points.
    X : numpy.1darray
        x-coordinates of polygon corners.
    Y : numpy.1darray
        y-coordinates of polygon corners.
    
    Returns
    ------- 
    is_in_polygon : numpy.ndarray
        Boolean array indicating if point is in polygon.
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
    """Returns localizations in rectangle defined by corners (X, Y).
    
    Parameters
    ----------
    locs : numpy.recarray
        Localizations list.
    X : list
        x-coordinates of rectangle corners.
    Y : list
        y-coordinates of rectangle corners.
    
    Returns
    -------
    picked_locs : numpy.recarray
        Localizations in rectangle.
    """
    
    is_in_rectangle = check_if_in_rectangle(
        locs.x, locs.y, _np.array(X), _np.array(Y)
    )
    picked_locs = locs[is_in_rectangle]
    return picked_locs


def minimize_shifts(shifts_x, shifts_y, shifts_z=None):
    """Minimizes shifts in x, y, and z directions. Used for drift correction.
    
    Parameters
    ---------- 
    shifts_x : numpy.2darray
        Shifts in x direction.
    shifts_y : numpy.2darray
        Shifts in y direction.
    shifts_z : numpy.2darray (default=None)
        Shifts in z direction.
        
    Returns
    -------
    shift_y : numpy.1darray
        Minimized shifts in y direction.
    shift_x : numpy.1darray
        Minimized shifts in x direction.
    shift_z : numpy.1darray (optional)
        Minimized shifts in z direction if shifts_z is not None.
    """
    
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
    """Returns the number of finished futures, used in multiprocessing."""

    return sum([_.done() for _ in futures])


def remove_from_rec(rec_array, name):
    """Removes a column from the existing np.recarray.

    Parameters
    ----------
    rec_array : np.rec.array
        Recarray from which the column is removed.
    name : str
        Name of the column to be removed.
    
    Returns
    -------
    rec_array : np.rec.array
        Recarray without the column.
    """

    rec_array = _drop_fields(rec_array, name, usemask=False, asrecarray=True)
    return rec_array


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


def get_pick_polygon_corners(pick):
    """Returns X and Y coordinates of a pick polygon.
        
    Returns None, None if the pick is not a closed polygon."""

    if len(pick) < 3 or pick[0] != pick[-1]:
        return None, None
    else:
        X = [_[0] for _ in pick]
        Y = [_[1] for _ in pick]
        return X, Y


def get_pick_rectangle_corners(start_x, start_y, end_x, end_y, width):
        """Finds the positions of corners of a rectangular pick.
        Rectangular pick is defined by:
            [(start_x, start_y), (end_x, end_y)]
        and its width. (all values in camera pixels)

        Returns
        -------
        corners : tuple
            Contains corners' x and y coordinates in two lists
        """

        if end_x == start_x:
            alpha = _np.pi / 2
        else:
            alpha = _np.arctan((end_y - start_y) / (end_x - start_x))
        dx = width * _np.sin(alpha) / 2
        dy = width * _np.cos(alpha) / 2
        x1 = float(start_x - dx)
        x2 = float(start_x + dx)
        x4 = float(end_x - dx)
        x3 = float(end_x + dx)
        y1 = float(start_y + dy)
        y2 = float(start_y - dy)
        y4 = float(end_y + dy)
        y3 = float(end_y - dy)
        corners = ([x1, x2, x3, x4], [y1, y2, y3, y4])
        return corners


# def pick_areas_circle(picks, r):
#     """Returns pick areas for each pick in picks.
    
#     Parameters
#     ----------
#     picks : list
#         List of picks, each pick is a list of x and y coordinates.
#     r : float
#         Pick radius.
    
#     Returns
#     -------
#     areas : np.1darray
#         Pick areas, same units as r.
#     """

#     areas = _np.ones(len(picks)) * _np.pi * r**2
#     return areas


def polygon_area(X, Y):
    """Finds the area of a polygon defined by corners X and Y.
    
    Parameters
    ----------
    X : numpy.1darray
        x-coordinates of the polygon corners.
    Y : numpy.1darray
        y-coordinates of the polygon corners.
    
    Returns
    -------
    area : float
        Area of the polygon.
    """

    n_corners = len(X)
    area = 0
    for i in range(n_corners):
        j = (i + 1) % n_corners # next corner
        area += X[i] * Y[j] - X[j] * Y[i]
    area = abs(area) / 2
    return area


def pick_areas_polygon(picks):
    """Returns pick areas for each pick in picks.
    
    Parameters
    ----------
    picks : list
        List of picks, each pick is a list of coordinates of the 
        polygon corners.
    
    Returns
    -------
    areas : np.1darray
        Pick areas.
    """

    areas = _np.zeros(len(picks))
    for i, pick in enumerate(picks):
        if len(pick) < 3 or pick[0] != pick[-1]: # not a closed polygon
            areas[i] = 0
            continue
        X, Y = get_pick_polygon_corners(pick)
        areas[i] = polygon_area(X, Y)
    areas = areas[areas > 0] # remove open polygons
    return areas


def pick_areas_rectangle(picks, w):
    """Returns pick areas for each pick in picks.
    
    Parameters
    ----------
    picks : list
        List of picks, each pick is a list of coordinates of the 
        rectangle corners.
    w : float
        Pick width.
    
    Returns
    -------
    areas : np.1darray
        Pick areas, same units as w.
    """

    areas = _np.zeros(len(picks))
    for i, pick in enumerate(picks):
        (xs, ys), (xe, ye) = pick
        areas[i] = w * _np.sqrt((xe - xs) ** 2 + (ye - ys) ** 2)
    return areas