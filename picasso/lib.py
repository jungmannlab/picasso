"""
    picasso.lib
    ~~~~~~~~~~~

    Handy functions and classes.

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import collections
import colorsys
import os
import time
from typing import Any
from collections.abc import Callable
from asyncio import Future

import numba
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields
from numpy.lib.recfunctions import drop_fields
from numpy.lib.recfunctions import stack_arrays
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT
)
from PyQt5 import QtCore, QtWidgets, QtGui
from playsound3 import playsound

from picasso import io

# A global variable where we store all open progress and status dialogs.
# In case of an exception, we close them all,
# so that the GUI remains responsive.
_dialogs = []

# Min. time to use sound notification when ProcessDialog or
# StatusDialog is finished
SOUND_NOTIFICATION_DURATION = 60  # seconds


class ProgressDialog(QtWidgets.QProgressDialog):
    """ProgressDialog displays a progress dialog with a progress bar."""

    def __init__(self, description, minimum, maximum, parent):
        # append time estimate to description
        super().__init__(
            description,
            None,
            minimum,
            maximum,
            parent,
            QtCore.Qt.CustomizeWindowHint,
        )
        self.description_base = description  # without time estimate
        self.initalized = None

    def init(self):
        _dialogs.append(self)
        self.setMinimumDuration(500)
        self.setModal(True)
        self.t0 = time.time()
        self.app = QtCore.QCoreApplication.instance()
        self.initalized = True
        self.finished = False
        # sound notification
        self.sound_notification_path = get_sound_notification_path()

    def set_value(self, value):
        if not self.initalized:
            self.init()
        self.setValue(value)
        # estimate time left
        elapsed = time.time() - self.t0
        remaining = int((self.maximum() - value) * elapsed / (value + 1e-6))
        # convert to hh-mm-ss
        hours, remainder = divmod(remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        # format time estimate
        if hours > 0:
            hours = min(10, hours)  # limit hours to 10 for display
            time_estimate = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
        else:
            time_estimate = f"{minutes:02d}m:{seconds:02d}s"
        # set label text with time estimate
        description = (
            f"{self.description_base}"
            f"\nEstimated time remaining: {time_estimate}"
        )
        self.setLabelText(description)
        if value >= self.maximum() and self.finished is False:
            self.finished = True
            self.play_sound_notification()
        self.app.processEvents()

    def closeEvent(self, event):
        _dialogs.remove(self)
        if self.finished is False:
            self.finished = True
            self.play_sound_notification()

    def zero_progress(self, description=None):
        """Set progress dialog to zero and changes title if given."""
        if description:
            self.setLabelText(description)
            self.description_base = description
        self.set_value(0)

    def play_sound_notification(self):
        """Play a sound notification if a sound file is specified and
        at least a minute has passed since the dialog was opened."""
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                playsound(self.sound_notification_path, block=False)


class StatusDialog(QtWidgets.QDialog):
    """StatusDialog displays the description string in a dialog."""

    def __init__(self, description, parent):
        super(StatusDialog, self).__init__(
            parent, QtCore.Qt.CustomizeWindowHint,
        )
        _dialogs.append(self)
        vbox = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(description)
        vbox.addWidget(label)
        self.sound_notification_path = get_sound_notification_path()
        self.t0 = time.time()
        self.show()
        QtCore.QCoreApplication.instance().processEvents()

    def closeEvent(self, event):
        _dialogs.remove(self)
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                playsound(self.sound_notification_path, block=False)


class MockProgress():
    """Class to mock a progress bar or dialog, allowing for calling
    the same methods but not displaying anything."""

    def __init__(self, *args):
        pass

    def init(self):
        pass

    def set_value(self, value):
        pass

    def update(self, value):
        pass

    def closeEvent(self, event):
        pass

    def zero_progress(self, description=None):
        pass

    def close(self):
        pass

    def setLabelText(self, text):
        pass

    def play_sound_notification(self):
        pass


class ScrollableGroupBox(QtWidgets.QGroupBox):
    """QGroupBox with QScrollArea as the top widget that enables
    scrolling."""

    def __init__(self, title, parent=None, layout="grid"):
        super().__init__(title, parent=parent)

        # Create a layout for the content of the group box
        if layout == "grid":
            self.content_layout = QtWidgets.QGridLayout(self)
        elif layout == "form":
            self.content_layout = QtWidgets.QFormLayout(self)
        self.content_layout.setAlignment(QtCore.Qt.AlignTop)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)

        # Create a scroll area and set its content to the content layout
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QtWidgets.QWidget(self))
        self.scroll_area.widget().setLayout(self.content_layout)

        # Set the layout of the group box to the scroll area
        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().addWidget(self.scroll_area, 0, 0, 1, 2)

    def add_widget(self, widget, row, column, height=1, width=1):
        """Add a widget to the grid layout inside the scroll area."""
        self.content_layout.addWidget(widget, row, column, height, width)

    def remove_widget(self, widget):
        """Remove a widget from the grid layout inside the scroll
        area."""
        self.content_layout.removeWidget(widget)

    def remove_all_widgets(self, keep_labels=False):
        """Remove all widgets. If ``keep_labels`` is True, the QLabels
        are kept."""
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if keep_labels and isinstance(widget, QtWidgets.QLabel):
                continue
            widget.setParent(None)
            del widget


class GenericPlotWindow(QtWidgets.QTabWidget):
    """Interface for displaying matplotlib plots in a separate
    window."""

    def __init__(self, window_title, app_name):
        super().__init__()
        self.setWindowTitle(window_title)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", f"{app_name}.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        vbox.addWidget(self.toolbar)


class AutoDict(collections.defaultdict):
    """A defaultdict whose auto-generated values are defaultdicts
    itself. This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


def cancel_dialogs():
    """Closes all open dialogs (``ProgressDialog`` and ``StatusDialog``)
    in the GUI."""
    dialogs = [_ for _ in _dialogs]
    for dialog in dialogs:
        if isinstance(dialog, ProgressDialog):
            dialog.cancel()
        else:
            dialog.close()
    QtCore.QCoreApplication.instance().processEvents()  # just in case...


def get_sound_notification_path() -> str | None:
    """Return the path to the sound notification file from the user
    settings file. If the file is not found or not specified, return
    None.

    Returns
    -------
    path : str or None
        Path to the sound notification file or None if not found or not
        specified.
    """
    settings = io.load_user_settings()
    if "Sound_notification" not in settings:  # add default settings (no sound)
        settings["Sound_notification"]["filename"] = None
        io.save_user_settings(settings)
    filename = settings["Sound_notification"]["filename"]
    sounds_dir = _sound_notification_dir()
    if (
        filename is not None and
        os.path.isfile(os.path.join(sounds_dir, filename))
    ):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".mp3", ".wav"]:
            path = None
        else:
            path = os.path.join(sounds_dir, filename)
    else:
        path = None
    return path


def get_available_sound_notifications() -> list[str | None]:
    """Get a list of file names of the available sound notifications in
    the folder ``resources/notification_sounds``.

    Returns
    -------
    filenames : list of strs
        List of file names of the available sound notifications.
    """
    sounds_dir = _sound_notification_dir()
    filenames = [
        _ for _ in os.listdir(sounds_dir)
        if os.path.isfile(os.path.join(sounds_dir, _)) and
        os.path.splitext(_)[1].lower() in [".mp3", ".wav"]
    ]
    filenames = ["None"] + filenames
    return filenames


def set_sound_notification(action: QtWidgets.QAction) -> None:
    """Save the selected sound notification in the user settings
    file.

    Parameters
    ----------
    action : QtWidgets.QAction
        The action representing the selected sound notification.
    """
    settings = io.load_user_settings()
    selected_sound = action.objectName()  # file name with extension
    settings["Sound_notification"]["filename"] = selected_sound
    io.save_user_settings(settings)
    # play selected sound as a preview
    play_path = get_sound_notification_path()
    playsound(play_path, block=False) if play_path is not None else None


def _sound_notification_dir() -> str:
    """Return the path to the sound notification folder."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "resources",
        "notification_sounds",
    )


def get_colors(n_channels):
    """Create a list with rgb channels for each channel.

    Colors go from red to green, blue, pink and red again.

    Parameters
    ----------
    n_channels : int
        Number of locs channels

    Returns
    -------
    list
        Contains tuples with rgb channels
    """
    hues = np.arange(0, 1, 1 / n_channels)
    colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
    return colors


def is_hexadecimal(text):
    """Check if text represents a hexadecimal code for rgb, for
    example ``#ff02d4``.

    Parameters
    ----------
    text : str
        String to be checked.

    Returns
    -------
    bool
        True if text represents rgb, False otherwise.
    """
    allowed_characters = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f',
        'A', 'B', 'C', 'D', 'E', 'F',
    ]
    sum_char = 0
    if isinstance(text, str):
        if text[0] == '#':
            if len(text) == 7:
                for char in text[1:]:
                    if char in allowed_characters:
                        sum_char += 1
                if sum_char == 6:
                    return True
    return False


def cumulative_exponential(
    x: np.ndarray,
    a: float,
    t: float,
    c: float,
) -> np.ndarray:
    """Used for binding kinetics estimation."""
    return a * (1 - np.exp(-(x / t))) + c


def calculate_optimal_bins(
    data: np.ndarray,
    max_n_bins: int | None = None,
) -> np.ndarray:
    """Calculate the optimal bins for display, for example, in
    Picasso: Filter.

    Parameters
    ----------
    data : np.ndarray
        Data to be binned.
    max_n_bins : int | None, optional
        Maximum number of bins.

    Returns
    -------
    bins : np.ndarray
        Bins for display.
    """
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return np.array([data[0] - 1.0, data[0] + 1.0])
    bin_size = 2 * iqr * len(data) ** (-1 / 3)
    if data.dtype.kind in ("u", "i") and bin_size < 1:
        bin_size = 1
    bin_min = data.min() - bin_size / 2
    try:
        n_bins = (data.max() - bin_min) / bin_size
        n_bins = int(n_bins)
    except Exception:
        n_bins = 10
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    bins = np.linspace(bin_min, data.max(), n_bins)
    return bins


def append_to_rec(
    rec_array: np.recarray,
    data: np.ndarray,
    name: str,
) -> np.recarray:
    """Append a new column to the existing np.recarray.

    Parameters
    ----------
    rec_array : np.recarray
        Recarray to which the new column is appended.
    data : np.ndarray
        1D data to be appended.
    name : str
        Name of the new column.

    Returns
    -------
    rec_array : np.recarray
        Recarray with the new column.
    """
    if hasattr(rec_array, name):
        rec_array = remove_from_rec(rec_array, name)
    rec_array = append_fields(
        rec_array,
        name,
        data,
        dtypes=data.dtype,
        usemask=False,
        asrecarray=True,
    )
    return rec_array


def merge_locs(
    locs_list: list[np.recarray],
    increment_frames: bool = True,
) -> np.recarray:
    """Merge localization lists into one file. Can increment frames
    to avoid overlapping frames.

    Parameters
    ----------
    locs_list : list of np.recarrays
        List of localization lists to be merged.
    increment_frames : bool, optional
        If True, increments frames of each localization list by the
        maximum frame number of the previous localization list. Useful
        when the localization lists are from different movies but
        represent the same stack. Default is True.

    Returns
    -------
    locs : np.recarray
        Merged localizations.
    """
    if increment_frames:
        last_frame = 0
        for i, locs in enumerate(locs_list):
            locs["frame"] += last_frame
            last_frame = locs["frame"][-1].max()
            locs_list[i] = locs
    locs = stack_arrays(locs_list, usemask=False, asrecarray=True)
    return locs


def ensure_sanity(locs: np.recarray, info: list[dict]) -> np.recarray:
    """Ensure that localizations are within the image dimensions
    and have positive localization precisions and other parameters.

    Parameters
    ----------
    locs : np.recarray
        Localizations list.
    info : list of dicts
        Localization metadata.

    Returns
    -------
    locs : np.recarray
        Localizations that pass the sanity checks.
    """
    # no inf or nan:
    locs = locs[
        np.all(
            np.array([np.isfinite(locs[_]) for _ in locs.dtype.names]),
            axis=0,
        )
    ]
    # other sanity checks:
    locs = locs[locs.x < info[0]["Width"]]
    locs = locs[locs.y < info[0]["Height"]]
    for attr in ["x", "y", "lpx", "lpy", "photons", "ellipticity", "sx", "sy"]:
        if hasattr(locs, attr):
            locs = locs[locs[attr] >= 0]
    return locs


def is_loc_at(x: float, y: float, locs: np.recarray, r: float) -> np.ndarray:
    """Check which localizations are within radius ``r`` from position
    ``(x, y)``.

    Parameters
    ----------
    x, y : float
        x and y-coordinate of the position.
    locs : np.recarray
        Localizations list.
    r : float
        Radius.

    Returns
    -------
    is_picked : np.ndarray
        Boolean array - True if a localization is within radius r
        of position (x, y).
    """
    dx = locs.x - x
    dy = locs.y - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked


def locs_at(x: float, y: float, locs: np.recarray, r: float) -> np.recarray:
    """Return localizations within radius ``r`` from the position
    ``(x, y)``.

    Parameters
    ----------
    x, y : float
        x and y-coordinate of the position.
    locs : np.recarray
        Localizations list.
    r : float
        Radius.

    Returns
    -------
    picked_locs : np.recarray
        Localizations in the specified area.
    """
    is_picked = is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    return picked_locs


@numba.jit(nopython=True)
def check_if_in_polygon(
    x: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Check if points ``(x, y)`` are within the polygon defined by
    corners ``(X, Y)``. Uses the ray casting algorithm, see
    ``check_if_in_rectangle`` for details.

    Parameters
    ----------
    x, y : np.ndarray
        x and y coordinates of points.
    X, Y : np.ndarray
        x and y coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : np.ndarray
        Boolean array indicating which points are in the polygon.
    """
    n_locs = len(x)
    n_polygon = len(X)
    is_in_polygon = np.zeros(n_locs, dtype=np.bool_)

    for i in range(n_locs):
        count = 0
        for j in range(n_polygon):
            j_next = (j + 1) % n_polygon
            if (
                ((Y[j] > y[i]) != (Y[j_next] > y[i])) and
                (
                    (x[i] < X[j] + (X[j_next] - X[j])
                     * (y[i] - Y[j]) / (Y[j_next] - Y[j]))
                )
            ):
                count += 1
        if count % 2 == 1:
            is_in_polygon[i] = True

    return is_in_polygon


def locs_in_polygon(
    locs: np.recarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.recarray:
    """Return localizations within the polygon defined by corners
    ``(X, Y)``.

    Parameters
    ----------
    locs : np.recarray
        Localizations.
    X, Y : list
        x and y-coordinates of polygon corners.

    Returns
    -------
    picked_locs : np.recarray
        Localizations in polygon.
    """
    is_in_polygon = check_if_in_polygon(
        locs.x, locs.y, np.array(X), np.array(Y)
    )
    return locs[is_in_polygon]


@numba.jit(nopython=True)
def check_if_in_rectangle(
    x: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """Check if locs with coordinates (x, y) are in rectangle with
    corners (X, Y) by counting the number of rectangle sides which are
    hit by a ray originating from each loc to the right. If the number
    of hit rectangle sides is odd, then the loc is in the rectangle.

    Parameters
    ----------
    x, y : np.ndarray
        x and y coordinates of points.
    X, Y : np.ndarray
        x and y coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : np.ndarray
        Boolean array indicating if point is in polygon.
    """
    n_locs = len(x)
    ray_hits_rectangle_side = np.zeros((n_locs, 4))
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
    n_sides_hit = np.sum(ray_hits_rectangle_side, axis=1)
    is_in_rectangle = n_sides_hit % 2 == 1
    return is_in_rectangle


def locs_in_rectangle(
    locs: np.recarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.recarray:
    """Return localizations within the rectangle defined by corners
    ``(X, Y)``.

    Parameters
    ----------
    locs : np.recarray
        Localizations list.
    X, Y : list
        x and y coordinates of rectangle corners.

    Returns
    -------
    picked_locs : np.recarray
        Localizations in rectangle.
    """
    is_in_rectangle = check_if_in_rectangle(
        locs.x, locs.y, np.array(X), np.array(Y)
    )
    picked_locs = locs[is_in_rectangle]
    return picked_locs


def minimize_shifts(
    shifts_x: np.ndarray,
    shifts_y: np.ndarray,
    shifts_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Minimize shifts in x, y, and z directions. Used for drift
    correction.

    Parameters
    ----------
    shifts_x, shifts_y : np.ndarray
        Shifts in x and y directions, shape (n_channels, n_channels).
    shifts_z : np.ndarray, optional
        Shifts in z direction, shape (n_channels, n_channels). If None,
        only x and y shifts are minimized.

    Returns
    -------
    shift_y, shift_x : np.ndarray
        Minimized shifts in y and x direction.
    shift_z : np.ndarray, optional
        Minimized shifts in z direction if ``shifts_z`` is specified.
    """
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = np.zeros((n_pairs, n_dims))
    A = np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            rij[flag, 0] = shifts_y[i, j]
            rij[flag, 1] = shifts_x[i, j]
            if n_dims == 3:
                rij[flag, 2] = shifts_z[i, j]
            A[flag, i:j] = 1
            flag += 1
    Dj = np.dot(np.linalg.pinv(A), rij)
    shift_y = np.insert(np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = np.insert(np.cumsum(Dj[:, 1]), 0, 0)
    if n_dims == 2:
        return shift_y, shift_x
    else:
        shift_z = np.insert(np.cumsum(Dj[:, 2]), 0, 0)
        return shift_y, shift_x, shift_z


def n_futures_done(futures: list[Future]) -> int:
    """Return the number of finished futures, used in
    multiprocessing."""
    return sum([_.done() for _ in futures])


def remove_from_rec(rec_array: np.recarray, name: str) -> np.recarray:
    """Remove a column from the existing recarray.

    Parameters
    ----------
    rec_array : np.recarray
        Recarray from which the column is removed.
    name : str
        Name of the column to be removed.

    Returns
    -------
    rec_array : np.recarray
        Recarray without the column.
    """
    rec_array = drop_fields(rec_array, name, usemask=False, asrecarray=True)
    return rec_array


def locs_glob_map(
    func: Callable[
        [np.recarray, dict, str, Any], tuple[np.recarray, list[dict]]
    ],
    pattern: str,
    args: list = [],
    kwargs: dict = {},
    extension: str = "",
) -> None:
    """Map a function to localization files, specified by the unix style
    path pattern.

    The function must take two arguments: ``locs`` and ``info``. It may
    take additional args and kwargs which are supplied to this map
    function. A new locs file will be saved if an extension is provided.
    In that case, the mapped function must return new locs and a new
    info dict.

    Parameters
    ----------
    func : Callable
        Function to be mapped to each locs file. It must take
        locs, info, path, and any additional args and kwargs.
    pattern : str
        Unix style path pattern to match locs files.
    args : list, optional
        Additional positional arguments to be passed to the function.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the function.
    extension : str, optional
        If provided, the mapped function must return new locs and info
        dict, and a new locs file will be saved with this extension.
        If not provided, the function is expected to modify locs and
        info in place.
    """
    paths = glob.glob(pattern)
    for path in paths:
        locs, info = io.load_locs(path)
        result = func(locs, info, path, *args, **kwargs)
        if extension:
            base, ext = os.path.splitext(path)
            out_path = base + "_" + extension + ".hdf5"
            locs, info = result
            io.save_locs(out_path, locs, info)


def get_pick_polygon_corners(
    pick: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Return X and Y coordinates of a pick polygon.
    Return (None, None) if the pick is not a closed polygon.

    Parameters
    ----------
    pick : list of tuples
        List of tuples, each tuple contains x and y coordinates of a
        polygon corner.

    Returns
    -------
    X, Y : list of floats
        Lists of x and y coordinates of the polygon corners.
        Return (None, None) if the pick is not a closed polygon.
    """
    if len(pick) < 3 or pick[0] != pick[-1]:
        return None, None
    else:
        X = [_[0] for _ in pick]
        Y = [_[1] for _ in pick]
        return X, Y


def get_pick_rectangle_corners(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float,
) -> tuple[list[float], list[float]]:
    """Find the positions of corners of a rectangular pick.
    A rectangular pick is defined by:
        [(start_x, start_y), (end_x, end_y)]
    and its width. (all values in camera pixels).

    Parameters
    ----------
    start_x, start_y : float
        Starting point of the pick.
    end_x, end_y : float
        Ending point of the pick.
    width : float
        Width of the pick in camera pixels.

    Returns
    -------
    corners : tuple
        Contains corners' x and y coordinates in two lists.
    """
    if end_x == start_x:
        alpha = np.pi / 2
    else:
        alpha = np.arctan((end_y - start_y) / (end_x - start_x))
    dx = width * np.sin(alpha) / 2
    dy = width * np.cos(alpha) / 2
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


def polygon_area(X: np.ndarray, Y: np.ndarray) -> float:
    """Find the area of a polygon defined by corners X and Y.

    Parameters
    ----------
    X, Y : np.ndarray
        x-coordinates and y-coordinates of the polygon corners.

    Returns
    -------
    area : float
        Area of the polygon.
    """
    n_corners = len(X)
    area = 0
    for i in range(n_corners):
        j = (i + 1) % n_corners  # next corner
        area += X[i] * Y[j] - X[j] * Y[i]
    area = abs(area) / 2
    return area


def pick_areas_polygon(picks: list[list[tuple[float, float]]]) -> np.ndarray:
    """Return pick areas for each polygonal pick in picks.

    Parameters
    ----------
    picks : list of lists of tuples
        List of picks, each pick is a list of (x, y) coordinates of the
        polygon corners.

    Returns
    -------
    areas : np.ndarray
        Pick areas.
    """
    areas = []
    for i, pick in enumerate(picks):
        if len(pick) < 3 or pick[0] != pick[-1]:  # not a closed polygon
            continue
        X, Y = get_pick_polygon_corners(pick)
        areas.append(polygon_area(X, Y))
    areas = np.array(areas)
    areas = areas[areas > 0]  # remove open polygons
    return areas


def pick_areas_rectangle(
    picks: list[list[tuple[float, float]]],
    w: float,
) -> np.ndarray:
    """Return pick areas for each pick in picks.

    Parameters
    ----------
    picks : list
        List of picks, each pick is a list of coordinates of the
        rectangle corners.
    w : float
        Pick width.

    Returns
    -------
    areas : np.ndarray
        Pick areas, same units as ``w``.
    """
    areas = np.zeros(len(picks))
    for i, pick in enumerate(picks):
        (xs, ys), (xe, ye) = pick
        areas[i] = w * np.sqrt((xe - xs) ** 2 + (ye - ys) ** 2)
    return areas
