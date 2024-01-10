"""
    gui/render
    ~~~~~~~~~~~~~~~~~~~~
    Graphical user interface for rendering localization images
    :author: Joerg Schnitzbauer & Maximilian Strauss 
        & Rafal Kowalewski, 2017-2022
    :copyright: Copyright (c) 2017 Jungmann Lab, MPI of Biochemistry
"""
import os
import sys
import traceback
import copy
import time
import os.path
import importlib, pkgutil
from math import ceil
from functools import partial

import lmfit
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
    as NavigationToolbar

from scipy.ndimage.filters import gaussian_filter
from numpy.lib.recfunctions import stack_arrays
from PyQt5 import QtCore, QtGui, QtWidgets

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from collections import Counter

import colorsys

from .. import imageprocess, io, lib, postprocess, render, clusterer, __version__
from .rotation import RotationWindow

# PyImarisWrite works on windows only
from ..ext.bitplane import IMSWRITER
if IMSWRITER:
    from .. ext.bitplane import numpy_to_imaris
    from PyImarisWriter.ImarisWriterCtypes import *
    from PyImarisWriter import PyImarisWriter as PW

if sys.platform == "darwin": # plots do not work on mac os
    matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({"axes.titlesize": "large"})


DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5
ZOOM = 9 / 7
N_GROUP_COLORS = 8
N_Z_COLORS = 32


def get_colors(n_channels):
    """ 
    Creates a list with rgb channels for each locs channel.
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


def get_render_properties_colors(n_channels, cmap='gist_rainbow'):
    """
    Creates a list with rgb channels for each of the channels used in
    rendering property using the gist_rainbow colormap, see:
    https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Parameters
    ----------
    n_channels : int
        Number of locs channels.
    cmap : str (default='gist_rainbow')
        Colormap name.

    Returns
    -------
    colors : list of tuples
        Contains tuples with rgb channels.
    """

    # array of shape (256, 3) with rbh channels with 256 colors
    base = plt.get_cmap(cmap)(np.arange(256))[:, :3]
    # indeces to draw from base
    idx = np.linspace(0, 255, n_channels).astype(int)
    # extract the colors of interest
    colors = base[idx]
    return colors


def is_hexadecimal(text):
    """ 
    Checks if text represents a hexadecimal code for rgb, e.g. #ff02d4.
    
    Parameters
    ----------
    text : str
        String to be checked

    Returns
    -------
    boolean
        True if text represents rgb, False otherwise
    """

    allowed_characters = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f',
        'A', 'B', 'C', 'D', 'E', 'F',
    ]
    sum_char = 0
    if type(text) == str:
        if text[0] == '#':
            if len(text) == 7:
                for char in text[1:]:
                    if char in allowed_characters:
                        sum_char += 1
                if sum_char == 6:
                    return True
    return False


def fit_cum_exp(data):
    """ 
    Returns an lmfit Model class fitted to a 3-parameter cumulative
    exponential.
    """

    data.sort()
    n = len(data)
    y = np.arange(1, n + 1)
    data_min = data.min()
    data_max = data.max()
    params = lmfit.Parameters()
    params.add("a", value=n, vary=True, min=0)
    params.add("t", value=np.mean(data), vary=True, min=data_min, max=data_max)
    params.add("c", value=data_min, vary=True, min=0)
    result = lib.CumulativeExponentialModel.fit(y, params, x=data)
    return result


def kinetic_rate_from_fit(data):
    """ Finds the mean dark time from the lmfit fitted Model. """

    if len(data) > 2:
        if data.ptp() == 0:
            rate = np.nanmean(data)
        else:
            result = fit_cum_exp(data)
            rate = result.best_values["t"]
    else:
        rate = np.nanmean(data)
    return rate


estimate_kinetic_rate = kinetic_rate_from_fit


def check_pick(f):
    """ Decorator verifying if there is at least one pick. """

    def wrapper(*args):
        if len(args[0]._picks) == 0:
            QtWidgets.QMessageBox.information(
                args[0],
                "Pick Error",
                ("No localizations picked." " Please pick first."),
            )
        else:
            return f(args[0])

    return wrapper


def check_picks(f):
    """ Decorator verifying if there are at least two picks. """

    def wrapper(*args):
        if len(args[0]._picks) < 2:
            QtWidgets.QMessageBox.information(
                args[0],
                "Pick Error",
                "Please pick at least twice.",
            )
        else:
            return f(args[0])

    return wrapper


class FloatEdit(QtWidgets.QLineEdit):
    """
    A class used for manipulating the influx rate in the info dialog.
    """

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.editingFinished.connect(self.onEditingFinished)

    def onEditingFinished(self):
        value = self.value()
        self.valueChanged.emit(value)

    def setValue(self, value):
        text = "{:.10e}".format(value)
        self.setText(text)

    def value(self):
        text = self.text()
        value = float(text)
        return value


class GenericPlotWindow(QtWidgets.QTabWidget):
    """
    A class used to display trace in a pick.

    ...

    Attributes
    ----------
    canvas : FigureCanvas
        PyQt5 backend used for displaying plots
    figure : plt.Figure
    toolbar : NavigationToolbar2QT
        PyQt5 backend used for displaying plot manipulation functions,
        e.g., save, zoom.
    """

    def __init__(self, window_title):
        super().__init__()
        self.setWindowTitle(window_title)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        vbox.addWidget(self.toolbar)


class PickHistWindow(QtWidgets.QTabWidget):
    """
    A class to display binding kinetics plots.

    ...

    Attributes
    ----------
    canvas : FigureCanvas
        PyQt5 backend used for displaying plots
    figure : plt.Figure
    toolbar : NavigationToolbar2QT
        PyQt5 backend used for displaying plot manipulation functions,
        e.g., save, zoom.

    Methods
    -------
    plot(pooled_locs, fit_result_len, fit_result_dark)
        Plots two histograms for experimental data and exponential fits
    """

    def __init__(self, info_dialog):
        super().__init__()
        self.setWindowTitle("Pick Histograms")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        vbox.addWidget((NavigationToolbar(self.canvas, self)))

    def plot(self, pooled_locs, fit_result_len, fit_result_dark):
        """ 
        Plots two histograms for experimental data and exponential fits.

        Parameters
        ----------
        pooled_locs : np.recarray
            All picked localizations
        fit_result_len : lmfit.Model
            Fitted model of a 3-parameter cumulative exponential for
            lenghts of each localization
        fit_result_dark : lmfit.Model
            Fitted model of a 3-parameter cumulative exponential 
        """

        self.figure.clear()

        # Length
        axes = self.figure.add_subplot(121)

        a = fit_result_len.best_values["a"]
        t = fit_result_len.best_values["t"]
        c = fit_result_len.best_values["c"]

        axes.set_title(
            "Length (cumulative) \n"
            r"$Fit: {:.2f}\cdot(1-exp(x/{:.2f}))+{:.2f}$".format(a, t, c)
        )
        data = pooled_locs.len
        data.sort()
        y = np.arange(1, len(data) + 1)
        axes.semilogx(data, y, label="data")
        axes.semilogx(data, fit_result_len.best_fit, label="fit")
        axes.legend(loc="best")
        axes.set_xlabel("Duration (frames)")
        axes.set_ylabel("Frequency")

        # Dark
        axes = self.figure.add_subplot(122)

        a = fit_result_dark.best_values["a"]
        t = fit_result_dark.best_values["t"]
        c = fit_result_dark.best_values["c"]

        axes.set_title(
            "Dark time (cumulative) \n"
            r"$Fit: {:.2f}\cdot(1-exp(x/{:.2f}))+{:.2f}$".format(a, t, c)
        )
        data = pooled_locs.dark
        data.sort()
        y = np.arange(1, len(data) + 1)
        axes.semilogx(data, y, label="data")
        axes.semilogx(data, fit_result_dark.best_fit, label="fit")
        axes.legend(loc="best")
        axes.set_xlabel("Duration (frames)")
        axes.set_ylabel("Frequency")
        self.canvas.draw()


class ApplyDialog(QtWidgets.QDialog): 
    """
    A class for the Apply Dialog.
    Apply expressions to manipulate localizations' display.

    ...

    Attributes
    ----------
    channel : QComboBox
        Points to the index of the channel to be manipulated
    cmd : QLineEdit
        Enter the expression here
    label : QLabel
        Displays which locs properties can be manipulated

    Methods
    -------
    getCmd(parent=None)
        Used for obtaining the expression
    update_vars(index)
        Update the variables that can be manipulated and show them in
        self.label

    Examples
    --------
    The examples below are to be input in self.cmd (Expression):

    x += 10
        Move x coordinate 10 units to the right (pixels)
    y -= 3
        Move y coordinate 3 units upwards (pixels)
    flip x z
        Exchange x- and z-axes
    spiral 2 3
        Plot each localization over time in a spiral with radius 2
        pixels and 3 turns
    uspiral
        Undo the last spiral action
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        # vars = self.view.locs[0].dtype.names
        self.setWindowTitle("Apply expression")
        vbox = QtWidgets.QVBoxLayout(self)
        layout = QtWidgets.QGridLayout()
        vbox.addLayout(layout)
        layout.addWidget(QtWidgets.QLabel("Channel:"), 0, 0)
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(self.window.view.locs_paths)
        layout.addWidget(self.channel, 0, 1)
        self.channel.currentIndexChanged.connect(self.update_vars)
        layout.addWidget(QtWidgets.QLabel("Pre-defined variables:"), 1, 0)
        self.label = QtWidgets.QLabel()
        layout.addWidget(self.label, 1, 1)
        self.update_vars(0)
        layout.addWidget(QtWidgets.QLabel("Expression:"), 2, 0)
        self.cmd = QtWidgets.QLineEdit()
        layout.addWidget(self.cmd, 2, 1)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getCmd(parent=None):
        """ 
        Obtain the expression as a string and the channel to be 
        manipulated. 
        """

        dialog = ApplyDialog(parent)
        result = dialog.exec_()
        cmd = dialog.cmd.text()
        channel = dialog.channel.currentIndex()
        return (cmd, channel, result == QtWidgets.QDialog.Accepted)

    def update_vars(self, index):
        """
        Update the variables that can be manipulated and show them in
        self.label
        """

        vars = self.window.view.locs[index].dtype.names
        self.label.setText(str(vars))


class DatasetDialog(QtWidgets.QDialog):
    """
    A class to handle the Dataset Dialog:
    Show legend, show white background.
    Tick and untick, change title of, set color, set relative intensity,
    and close each channel.

    ...

    Attributes
    ----------
    auto_display : QCheckBox
        Tick to automatically adjust the rendered localizations. Untick
        to not change the rendering of localizations
    auto_colors : QCheckBox
        Tick to automatically color each channel. Untick to manually 
        change colors.
    checks : list
        List with QPushButtons for ticking/unticking each channel
    closebuttons : list
        List of QPushButtons to close each channel
    colordisp_all : list
        List of QLabels showing the color selected for each channel
    colorselection : list
        List of QComboBoxes specifying the color displayed for each
        channel
    default_colors : list
        List of strings specifying the default 14 colors
    intensitysettings : list
        List of QDoubleSpinBoxes specifying relative intensity of each
        channel
    legend : QCheckBox
        Used to show/hide legend
    rgbf : list
        List of lists of 3 elements specifying the corresponding colors
        as RGB channels
    title : list
        List of QPushButtons to change the title of each channel
    warning : boolean
        Used to memorize if the warning about multiple channels is to
        be displayed
    wbackground : QCheckBox
        Used to (de)activate white background for multichannel or
        to invert colors for single channel
    window : Window(QMainWindow)
        Main window instance

    Methods
    -------
    add_entry(path)
        Adds the new channel for the given path
    change_title(button_name)
        Opens QInputDialog to enter the new title for a given channel
    close_file(i)
        Closes a given channel and delets all corresponding attributes
    load_colors()
        Loads a list of colors from a .yaml file
    save_colors()
        Saves the list of colors as a .yaml file
    set_color(n)
        Sets colorsdisp_all and colorselection in the given channel
    update_colors()
        Changes colors in self.colordisp_all and updates the scene in
        the main window
    update_viewport()
        Updates the scene in the main window
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Datasets")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.warning = True
        self.checks = []
        self.title = []
        self.closebuttons = []
        self.colorselection = []
        self.colordisp_all = []
        self.intensitysettings = []
        self.setLayout(self.layout)
        self.legend = QtWidgets.QCheckBox("Show legend")
        self.wbackground = QtWidgets.QCheckBox(
            "Invert colors / white background"
        )
        self.auto_display = QtWidgets.QCheckBox("Automatic display update")
        self.auto_display.setChecked(True)
        self.auto_colors = QtWidgets.QCheckBox("Automatic coloring")
        self.layout.addWidget(self.legend, 0, 0, 1, 2)
        self.layout.addWidget(self.auto_display, 1, 0, 1, 2)
        self.layout.addWidget(self.wbackground, 2, 0, 1, 2)
        self.layout.addWidget(self.auto_colors, 3, 0, 1, 2)
        self.layout.addWidget(QtWidgets.QLabel("Files"), 4, 0)
        self.layout.addWidget(QtWidgets.QLabel("Change title"), 4, 1)
        self.layout.addWidget(QtWidgets.QLabel("Color"), 4, 2)
        self.layout.addWidget(QtWidgets.QLabel(""), 4, 3)
        self.layout.addWidget(QtWidgets.QLabel("Rel. Intensity"), 4, 4)
        self.layout.addWidget(QtWidgets.QLabel("Close"), 4, 5)
        self.legend.stateChanged.connect(self.update_viewport)
        self.wbackground.stateChanged.connect(self.update_viewport)
        self.auto_display.stateChanged.connect(self.update_viewport)
        self.auto_colors.stateChanged.connect(self.update_colors)

        # save and load color list
        save_button = QtWidgets.QPushButton("Save colors")
        self.layout.addWidget(
            save_button, 0, self.layout.columnCount() - 2, 1, 2
        )
        save_button.setFocusPolicy(QtCore.Qt.NoFocus)
        save_button.clicked.connect(self.save_colors)
        load_button = QtWidgets.QPushButton("Load colors")
        self.layout.addWidget(
            load_button, 1, self.layout.columnCount() - 2, 1, 2
        )
        load_button.setFocusPolicy(QtCore.Qt.NoFocus)
        load_button.clicked.connect(self.load_colors)

        self.default_colors = [ 
            "red",
            "cyan",
            "green",
            "yellow",
            "blue",
            "magenta",
            "orange",
            "amethyst",
            "forestgreen",
            "carmine",
            "purple",
            "sage",
            "jade",
            "azure",
        ]
        self.rgbf = [
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0.5, 0],
            [0.5, 0.5, 1],
            [0, 0.5, 0],
            [0.5, 0, 0],
            [0.5, 0, 1],
            [0.5, 0.5, 0],
            [0, 0.5, 0.5],
            [0, 0.5, 1],
        ]

    def add_entry(self, path):
        """ Adds the new channel for the given path. """

        # display only the characters after the last '/' 
        # for a long path
        if len(path) > 40:
            path = os.path.basename(path)
            path, ext = os.path.splitext(path)

        # Create 3 buttons for checking, naming and closing the channel
        c = QtWidgets.QCheckBox(path)
        currentline = self.layout.rowCount()
        t = QtWidgets.QPushButton("#")
        t.setObjectName(str(currentline))
        p = QtWidgets.QPushButton("x")
        p.setObjectName(str(currentline))

        # Append and setup the buttons
        self.checks.append(c)
        self.checks[-1].setChecked(True)
        self.checks[-1].stateChanged.connect(self.update_viewport)

        self.title.append(t)
        self.title[-1].setAutoDefault(False)
        self.title[-1].clicked.connect(
            partial(self.change_title, t.objectName())
        )

        self.closebuttons.append(p)
        self.closebuttons[-1].setAutoDefault(False)
        self.closebuttons[-1].clicked.connect(
            partial(self.close_file, p.objectName(), True)
        )

        # create the self.colorselection widget
        colordrop = QtWidgets.QComboBox(self)
        colordrop.setEditable(True)
        colordrop.lineEdit().setMaxLength(12)
        for color in self.default_colors:
            colordrop.addItem(color)
        index = np.min([len(self.checks)-1, len(self.rgbf)-1])
        colordrop.setCurrentText(self.default_colors[index])
        colordrop.activated.connect(self.update_colors)
        self.colorselection.append(colordrop)  
        self.colorselection[-1].currentIndexChanged.connect(
            partial(self.set_color, t.objectName())
        )      

        # create the label widget to show current color
        colordisp = QtWidgets.QLabel("      ")
        palette = colordisp.palette()
        if self.auto_colors.isChecked():
            colors = get_colors(len(self.checks) + 1)
            r, g, b = colors[-1]
            palette.setColor(
                QtGui.QPalette.Window,
                QtGui.QColor.fromRgbF(r, g, b, 1)
            )
        else:
            palette.setColor(
                QtGui.QPalette.Window, 
                QtGui.QColor.fromRgbF(
                    self.rgbf[index][0], 
                    self.rgbf[index][1], 
                    self.rgbf[index][2], 
                    1,
                )
            )
        colordisp.setAutoFillBackground(True)
        colordisp.setPalette(palette)
        self.colordisp_all.append(colordisp)

        # create the relative intensity widget
        intensity = QtWidgets.QDoubleSpinBox(self)
        intensity.setKeyboardTracking(False)
        intensity.setDecimals(2)
        intensity.setValue(1.00)
        self.intensitysettings.append(intensity)
        self.intensitysettings[-1].valueChanged.connect(self.update_viewport)

        # add all the widgets to the Dataset Dialog
        self.layout.addWidget(c, currentline, 0)
        self.layout.addWidget(t, currentline, 1)
        self.layout.addWidget(colordrop, currentline, 2)
        self.layout.addWidget(colordisp, currentline, 3)
        self.layout.addWidget(intensity, currentline, 4)
        self.layout.addWidget(p, currentline, 5)

        # check if the number of channels surpassed the number of 
        # default colors
        if len(self.checks) == len(self.default_colors):
            if self.warning:
                text = (
                    "The number of channels passed the number of default "
                    " colors. In case you would like to use your own color, "
                    " please insert the color's hexadecimal expression,"
                    " starting with '#', e.g. '#ffcdff' for pink or choose"
                    " the automatic coloring in the Files dialog."
                )
                QtWidgets.QMessageBox.information(self, "Warning", text)
                self.warning = False

    def update_colors(self):
        """
        Changes colors in self.colordisp_all and updates the scene in
        the main window
        """

        n_channels = len(self.checks)
        for i in range(n_channels):
            self.set_color(i)
        self.update_viewport()

    def change_title(self, button_name):
        """ 
        Opens QInputDialog to enter the new title for a given channel. 
        """

        for i in range(len(self.title)):
            if button_name == self.title[i].objectName():
                new_title, ok = QtWidgets.QInputDialog.getText(
                    self, 
                    "Set the new title", 
                    'Type "reset" to get the original title.'
                )
                if ok:
                    if new_title == "Reset" or new_title == "reset":
                        path = self.window.view.locs_paths[i]
                        if len(path) > 40:
                                path = os.path.basename(path)
                                new_title, ext = os.path.splitext(path)
                        self.checks[i].setText(new_title)
                    else:
                        self.checks[i].setText(new_title)
                    self.update_viewport()
                    # change size of the dialog
                    self.adjustSize()
                    # change name in the fast render dialog
                    self.window.fast_render_dialog.channel.setItemText(
                        i+1, new_title
                    )
                break

    def close_file(self, i, render=True):
        """
        Closes a given channel and delets all corresponding attributes.
        """

        if type(i) == str:
            for j in range(len(self.closebuttons)):
                if i == self.closebuttons[j].objectName():
                    i = j

        # restart the main window if the last channel is closed
        if len(self.closebuttons) == 1:
            self.window.remove_locs()
        else:
            # remove widgets from the Dataset Dialog
            self.layout.removeWidget(self.checks[i])
            self.layout.removeWidget(self.title[i])
            self.layout.removeWidget(self.colorselection[i])
            self.layout.removeWidget(self.colordisp_all[i])
            self.layout.removeWidget(self.intensitysettings[i])
            self.layout.removeWidget(self.closebuttons[i])

            # delete the widgets from the lists
            del self.checks[i]
            del self.title[i]
            del self.colorselection[i]
            del self.colordisp_all[i]
            del self.intensitysettings[i]
            del self.closebuttons[i]

            # delete all the View attributes
            del self.window.view.locs[i]
            del self.window.view.locs_paths[i]
            del self.window.view.infos[i]
            del self.window.view.index_blocks[i]

            # delete zcoord from slicer dialog
            try:
                self.window.slicer_dialog.zcoord[i]
            except:
                pass

            # delete attributes from the fast render dialog
            del self.window.view.all_locs[i]
            self.window.fast_render_dialog.on_file_closed(i)

            # adjust group color if needed
            if len(self.window.view.locs) == 1:
                if hasattr(self.window.view.locs[0], "group"):
                    self.window.view.group_color = (
                        self.window.view.get_group_color(
                            self.window.view.locs[0]
                        )
                    )

            # delete drift data if provided
            try:
                del self._drift[i]
                del self._driftfiles[i]
                del self.currentdrift[i]
            except:
                pass

            # update the window and adjust the size of the 
            # Dataset Dialog
            if render:
                self.update_viewport()
                self.adjustSize()

    def update_viewport(self):
        """ Updates the scene in the main window. """

        if self.auto_display.isChecked():
            if self.window.view.viewport:
                self.window.view.update_scene()

    def set_color(self, n):
        """ 
        Sets colorsdisp_all and colorselection in the given channel. 
        """

        if type(n) == str:
            for j in range(len(self.title)):
                if n == self.title[j].objectName():
                    n = j

        palette = self.colordisp_all[n].palette()
        color = self.colorselection[n].currentText()
        if self.auto_colors.isChecked():
            n_channels = len(self.checks)
            r, g, b = get_colors(n_channels)[n]
            palette.setColor(
                QtGui.QPalette.Window, 
                QtGui.QColor.fromRgbF(r, g, b, 1)
            )
        elif is_hexadecimal(color):
            color = color.lstrip("#")
            r, g, b = tuple(
                int(color[i: i + 2], 16) / 255 for i in (0, 2, 4)
            )
            palette.setColor(
                QtGui.QPalette.Window, QtGui.QColor.fromRgbF(r, g, b, 1))
        elif color in self.default_colors:
            i = self.default_colors.index(color)
            palette.setColor(
                QtGui.QPalette.Window, 
                QtGui.QColor.fromRgbF(
                    self.rgbf[i][0], 
                    self.rgbf[i][1], 
                    self.rgbf[i][2], 1
                )
            )
        self.colordisp_all[n].setPalette(palette)

    def save_colors(self):
        """ Saves the list of colors as a .yaml file. """

        colornames = [_.currentText() for _ in self.colorselection]

        out_path = self.window.view.locs_paths[0].replace(
            ".hdf5", "_colors.txt"
        )
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save colors to", out_path, filter="*.txt"
        )
        if path:
            with open(path, "w") as file:
                for color in colornames:
                    file.write(color + "\n")

    def load_colors(self):
        """ Loads a list of colors from a .yaml file. """

        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Load colors from .txt", 
            directory=self.window.pwd, 
            filter="*.txt",
        )
        if path:
            with open(path, "r") as file:
                colors = file.readlines()
                colornames = [color.rstrip() for color in colors]

            # check that the number of channels is smaller than
            # or equal to the number of color names in the .txt
            if len(self.checks) > len(colornames):
                raise ValueError("Txt file contains too few names")

            # check that all the names are valid
            for i, color in enumerate(colornames):
                if (
                    not color in self.default_colors
                    and not is_hexadecimal(color)
                ):
                    raise ValueError(
                        f"'{color}' at position {i+1} is invalid."
                    )

            # add the names to the 'Color' column (self.colorseletion)
            for i, color_ in enumerate(self.colorselection):
                color_.setCurrentText(colornames[i])

            self.update_colors()

class PlotDialog(QtWidgets.QDialog):
    """ 
    A class to plot a 3D scatter of picked localizations. 
    Allows the user to keep the given picks of remove them.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Structure")
        layout_grid = QtWidgets.QGridLayout(self)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.label = QtWidgets.QLabel()

        layout_grid.addWidget(self.label, 0, 0, 1, 3)
        layout_grid.addWidget(self.canvas, 1, 0, 1, 3)

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Yes
            | QtWidgets.QDialogButtonBox.No
            | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        layout_grid.addWidget(self.buttons)
        self.buttons.button(QtWidgets.QDialogButtonBox.Yes).clicked.connect(
            self.on_accept
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.No).clicked.connect(
            self.on_reject
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(
            self.on_cancel
        )

    def on_accept(self):
        self.setResult(1)
        self.result = 1
        self.close()

    def on_reject(self):
        self.setResult(0)
        self.result = 0
        self.close()

    def on_cancel(self):
        self.setResult(2)
        self.result = 2
        self.close()

    @staticmethod
    def getParams(all_picked_locs, current, length, mode, color_sys):
        """
        Plots the 3D scatter and returns the clicked button.
        mode == 0 means that the locs in picks are combined.
        mode == 1 means that locs from a given channel are plotted.
        """

        dialog = PlotDialog(None)
        fig = dialog.figure
        ax = fig.add_subplot(111, projection="3d")
        dialog.label.setText(
            "3D Scatterplot of pick {} of {}.".format(current + 1, length)
        )

        if mode == 1:
            locs = all_picked_locs[current]
            locs = stack_arrays(locs, asrecarray=True, usemask=False)

            colors = locs["z"][:]
            colors[
                colors > np.mean(locs["z"]) + 3 * np.std(locs["z"])
            ] = np.mean(locs["z"]) + 3 * np.std(locs["z"])
            colors[
                colors < np.mean(locs["z"]) - 3 * np.std(locs["z"])
            ] = np.mean(locs["z"]) - 3 * np.std(locs["z"])
            ax.scatter(
                locs["x"], locs["y"], locs["z"], c=colors, cmap="jet", s=2
            )
            ax.set_xlabel("X [Px]")
            ax.set_ylabel("Y [Px]")
            ax.set_zlabel("Z [nm]")
            ax.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax.set_ylim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax.set_zlim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )
            plt.gca().patch.set_facecolor("black")
            ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
        else:
            colors = color_sys
            for l in range(len(all_picked_locs)):
                locs = all_picked_locs[l][current]
                locs = stack_arrays(locs, asrecarray=True, usemask=False)
                ax.scatter(locs["x"], locs["y"], locs["z"], c=colors[l], s=2)

            ax.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax.set_ylim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax.set_zlim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )

            ax.set_xlabel("X [Px]")
            ax.set_ylabel("Y [Px]")
            ax.set_zlabel("Z [nm]")

            plt.gca().patch.set_facecolor("black")
            ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))

        result = dialog.exec_()
        return dialog.result


class PlotDialogIso(QtWidgets.QDialog):
    """ 
    A class to plot 4 scatter plots: XY, XZ and YZ projections and a
    3D plot.
    Allows the user to keep the given picks of remove them.
    Everything but the getParams method is identical to PlotDialog.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Structure")
        layout_grid = QtWidgets.QGridLayout(self)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.label = QtWidgets.QLabel()

        layout_grid.addWidget(self.label, 0, 0, 1, 3)
        layout_grid.addWidget(self.canvas, 1, 0, 1, 3)

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Yes
            | QtWidgets.QDialogButtonBox.No
            | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        layout_grid.addWidget(self.buttons)
        self.buttons.button(QtWidgets.QDialogButtonBox.Yes).clicked.connect(
            self.on_accept
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.No).clicked.connect(
            self.on_reject
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(
            self.on_cancel
        )

    def on_accept(self):
        self.setResult(1)
        self.result = 1
        self.close()

    def on_reject(self):
        self.setResult(0)
        self.result = 0
        self.close()

    def on_cancel(self):
        self.setResult(2)
        self.result = 2
        self.close()

    @staticmethod
    def getParams(all_picked_locs, current, length, mode, color_sys):
        """
        Plots the 3D scatter and 3 projections and returns the clicked 
        button.
        mode == 0 means that the locs in picks are combined.
        mode == 1 means that locs from a given channel are plotted.
        """

        dialog = PlotDialogIso(None)
        fig = dialog.figure
        ax = fig.add_subplot(221, projection="3d")
        dialog.label.setText(
            "3D Scatterplot of pick {} of {}.".format(current + 1, length)
        )
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        if mode == 1:
            locs = all_picked_locs[current]
            locs = stack_arrays(locs, asrecarray=True, usemask=False)

            colors = locs["z"][:]
            colors[
                colors > np.mean(locs["z"]) + 3 * np.std(locs["z"])
            ] = np.mean(locs["z"]) + 3 * np.std(locs["z"])
            colors[
                colors < np.mean(locs["z"]) - 3 * np.std(locs["z"])
            ] = np.mean(locs["z"]) - 3 * np.std(locs["z"])

            ax.scatter(
                locs["x"], locs["y"], locs["z"], c=colors, cmap="jet", s=2
            )
            ax.set_xlabel("X [Px]")
            ax.set_ylabel("Y [Px]")
            ax.set_zlabel("Z [nm]")
            ax.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax.set_ylim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax.set_zlim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )
            ax.set_title("3D")
            # plt.gca().patch.set_facecolor('black')
            ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))

            # AXES 2
            ax2.scatter(locs["x"], locs["y"], c=colors, cmap="jet", s=2)
            ax2.set_xlabel("X [Px]")
            ax2.set_ylabel("Y [Px]")
            ax2.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax2.set_ylim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax2.set_title("XY")
            ax2.set_facecolor("black")

            # AXES 3
            ax3.scatter(locs["x"], locs["z"], c=colors, cmap="jet", s=2)
            ax3.set_xlabel("X [Px]")
            ax3.set_ylabel("Z [nm]")
            ax3.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax3.set_ylim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )
            ax3.set_title("XZ")
            ax3.set_facecolor("black")

            # AXES 4
            ax4.scatter(locs["y"], locs["z"], c=colors, cmap="jet", s=2)
            ax4.set_xlabel("Y [Px]")
            ax4.set_ylabel("Z [nm]")
            ax4.set_xlim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax4.set_ylim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )
            ax4.set_title("YZ")
            ax4.set_facecolor("black")

        else:
            colors = color_sys
            for l in range(len(all_picked_locs)):
                locs = all_picked_locs[l][current]
                locs = stack_arrays(locs, asrecarray=True, usemask=False)
                ax.scatter(locs["x"], locs["y"], locs["z"], c=colors[l], s=2)
                ax2.scatter(locs["x"], locs["y"], c=colors[l], s=2)
                ax3.scatter(locs["x"], locs["z"], c=colors[l], s=2)
                ax4.scatter(locs["y"], locs["z"], c=colors[l], s=2)

            ax.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax.set_ylim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax.set_zlim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )

            ax.set_xlabel("X [Px]")
            ax.set_ylabel("Y [Px]")
            ax.set_zlabel("Z [nm]")

            ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))

            # AXES 2
            ax2.set_xlabel("X [Px]")
            ax2.set_ylabel("Y [Px]")
            ax2.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax2.set_ylim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax2.set_title("XY")
            ax2.set_facecolor("black")

            # AXES 3
            ax3.set_xlabel("X [Px]")
            ax3.set_ylabel("Z [nm]")
            ax3.set_xlim(
                np.mean(locs["x"]) - 3 * np.std(locs["x"]),
                np.mean(locs["x"]) + 3 * np.std(locs["x"]),
            )
            ax3.set_ylim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )
            ax3.set_title("XZ")
            ax3.set_facecolor("black")

            # AXES 4
            ax4.set_xlabel("Y [Px]")
            ax4.set_ylabel("Z [nm]")
            ax4.set_xlim(
                np.mean(locs["y"]) - 3 * np.std(locs["y"]),
                np.mean(locs["y"]) + 3 * np.std(locs["y"]),
            )
            ax4.set_ylim(
                np.mean(locs["z"]) - 3 * np.std(locs["z"]),
                np.mean(locs["z"]) + 3 * np.std(locs["z"]),
            )
            ax4.set_title("YZ")
            ax4.set_facecolor("black")

        result = dialog.exec_()

        return dialog.result


class ClsDlg3D(QtWidgets.QDialog):
    """
    A class to cluster picked locs with k-means clustering in 3D.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Structure")
        self.showMaximized()
        self.layout_grid = QtWidgets.QGridLayout(self)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.label = QtWidgets.QLabel()

        self.layout_grid.addWidget(self.label, 0, 0, 1, 5)
        self.layout_grid.addWidget(self.canvas, 1, 0, 8, 5)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Yes
            | QtWidgets.QDialogButtonBox.No
            | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        self.layout_grid.addWidget(self.buttons, 10, 0, 1, 3)
        self.layout_grid.addWidget(
            QtWidgets.QLabel("No clusters:"), 10, 3, 1, 1
        )

        self.n_clusters_spin = QtWidgets.QSpinBox()

        self.layout_grid.addWidget(self.n_clusters_spin, 10, 4, 1, 1)

        self.buttons.button(QtWidgets.QDialogButtonBox.Yes).clicked.connect(
            self.on_accept
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.No).clicked.connect(
            self.on_reject
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(
            self.on_cancel
        )

        self.start_clusters = 0
        self.n_clusters_spin.valueChanged.connect(self.on_cluster)
        self.n_lines = 12
        self.layout_grid.addWidget(QtWidgets.QLabel("Select"), 11, 4, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("X-Center"), 11, 0, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("Y-Center"), 11, 1, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("Z-Center"), 11, 2, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("Counts"), 11, 3, 1, 1)
        self.checks = []

    def add_clusters(self, element, x_mean, y_mean, z_mean):
        c = QtWidgets.QCheckBox(str(element[0] + 1))

        self.layout_grid.addWidget(c, self.n_lines, 4, 1, 1)
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(x_mean)), self.n_lines, 0, 1, 1
        )
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(y_mean)), self.n_lines, 1, 1, 1
        )
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(z_mean)), self.n_lines, 2, 1, 1
        )
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(element[1])), self.n_lines, 3, 1, 1
        )
        self.n_lines += 1
        self.checks.append(c)
        self.checks[-1].setChecked(True)

    def on_accept(self):
        self.setResult(1)
        self.result = 1
        self.close()

    def on_reject(self):
        self.setResult(0)
        self.result = 0
        self.close()

    def on_cancel(self):
        self.setResult(2)
        self.result = 2
        self.close()

    def on_cluster(self):
        if (
            self.n_clusters_spin.value() != self.start_clusters
        ):  # only execute once the cluster number is changed
            self.setResult(3)
            self.result = 3
            self.close()

    @staticmethod
    def getParams(
        all_picked_locs, current, length, n_clusters, color_sys, pixelsize
    ):

        dialog = ClsDlg3D(None)

        dialog.start_clusters = n_clusters
        dialog.n_clusters_spin.setValue(n_clusters)

        fig = dialog.figure
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        dialog.label.setText(
            "3D Scatterplot of Pick "
            + str(current + 1)
            + "  of: "
            + str(length)
            + "."
        )

        locs = all_picked_locs[current]
        locs = stack_arrays(locs, asrecarray=True, usemask=False)

        est = KMeans(n_clusters=n_clusters, n_init='auto')

        scaled_locs = lib.append_to_rec(
            locs, locs["x"] * pixelsize, "x_scaled"
        )
        scaled_locs = lib.append_to_rec(
            scaled_locs, locs["y"] * pixelsize, "y_scaled"
        )

        X = np.asarray(scaled_locs["x_scaled"])
        Y = np.asarray(scaled_locs["y_scaled"])
        Z = np.asarray(scaled_locs["z"])

        est.fit(np.stack((X, Y, Z), axis=1))

        labels = est.labels_

        counts = list(Counter(labels).items())
        # l_locs = lib.append_to_rec(l_locs,labels,'cluster')

        ax1.scatter(
            locs["x"], locs["y"], locs["z"], c=labels.astype(float), s=2
        )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        counts = list(Counter(labels).items())
        cent = est.cluster_centers_

        ax2.scatter(cent[:, 0], cent[:, 1], cent[:, 2], s=2)
        for element in counts:
            x_mean = cent[element[0], 0]
            y_mean = cent[element[0], 1]
            z_mean = cent[element[0], 2]
            dialog.add_clusters(element, x_mean, y_mean, z_mean)
            ax2.text(x_mean, y_mean, z_mean, element[1], fontsize=12)

        ax1.set_xlabel("X [Px]")
        ax1.set_ylabel("Y [Px]")
        ax1.set_zlabel("Z [Px]")

        ax2.set_xlabel("X [nm]")
        ax2.set_ylabel("Y [nm]")
        ax2.set_zlabel("Z [nm]")

        ax1.w_xaxis.set_pane_color((0, 0, 0, 1.0))
        ax1.w_yaxis.set_pane_color((0, 0, 0, 1.0))
        ax1.w_zaxis.set_pane_color((0, 0, 0, 1.0))
        plt.gca().patch.set_facecolor("black")

        result = dialog.exec_()

        checks = [not _.isChecked() for _ in dialog.checks]
        checks = np.asarray(np.where(checks)) + 1
        checks = checks[0]

        labels += 1
        labels = [0 if x in checks else x for x in labels]
        labels = np.asarray(labels)

        l_locs = lib.append_to_rec(scaled_locs, labels, "cluster")
        l_locs_new_group = l_locs.copy()
        power = np.round(n_clusters / 10) + 1
        l_locs_new_group["group"] = (
            l_locs_new_group["group"] * 10 ** power
            + l_locs_new_group["cluster"]
        )

        # Combine clustered locs
        clustered_locs = []
        for element in np.unique(labels):
            if element != 0:
                clustered_locs.append(
                    l_locs_new_group[l_locs["cluster"] == element]
                )

        return (
            dialog.result,
            dialog.n_clusters_spin.value(),
            l_locs,
            clustered_locs,
        )


class ClsDlg2D(QtWidgets.QDialog):
    """
    A class to cluster picked locs with k-means clustering in 2D.
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Structure")
        self.layout_grid = QtWidgets.QGridLayout(self)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.label = QtWidgets.QLabel()

        self.layout_grid.addWidget(self.label, 0, 0, 1, 5)
        self.layout_grid.addWidget(self.canvas, 1, 0, 1, 5)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Yes
            | QtWidgets.QDialogButtonBox.No
            | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        self.layout_grid.addWidget(self.buttons, 2, 0, 1, 3)
        self.layout_grid.addWidget(QtWidgets.QLabel("No clusters:"), 2, 3, 1, 1)

        self.n_clusters_spin = QtWidgets.QSpinBox()

        self.layout_grid.addWidget(self.n_clusters_spin, 2, 4, 1, 1)

        self.buttons.button(QtWidgets.QDialogButtonBox.Yes).clicked.connect(
            self.on_accept
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.No).clicked.connect(
            self.on_reject
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(
            self.on_cancel
        )

        self.start_clusters = 0
        self.n_clusters_spin.valueChanged.connect(self.on_cluster)
        self.n_lines = 4
        self.layout_grid.addWidget(QtWidgets.QLabel("Select"), 3, 3, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("X-Center"), 3, 0, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("Y-Center"), 3, 1, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel("Counts"), 3, 2, 1, 1)
        self.checks = []

    def add_clusters(self, element, x_mean, y_mean):
        c = QtWidgets.QCheckBox(str(element[0] + 1))

        self.layout_grid.addWidget(c, self.n_lines, 3, 1, 1)
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(x_mean)), self.n_lines, 0, 1, 1
        )
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(y_mean)), self.n_lines, 1, 1, 1
        )
        self.layout_grid.addWidget(
            QtWidgets.QLabel(str(element[1])), self.n_lines, 2, 1, 1
        )
        self.n_lines += 1
        self.checks.append(c)
        self.checks[-1].setChecked(True)

    def on_accept(self):
        self.setResult(1)
        self.result = 1
        self.close()

    def on_reject(self):
        self.setResult(0)
        self.result = 0
        self.close()

    def on_cancel(self):
        self.setResult(2)
        self.result = 2
        self.close()

    def on_cluster(self):
        if (
            self.n_clusters_spin.value() != self.start_clusters
        ):  # only execute once the cluster number is changed
            self.setResult(3)
            self.result = 3
            self.close()

    @staticmethod
    def getParams(all_picked_locs, current, length, n_clusters, color_sys):

        dialog = ClsDlg2D(None)

        dialog.start_clusters = n_clusters
        dialog.n_clusters_spin.setValue(n_clusters)

        fig = dialog.figure
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        dialog.label.setText(
            "2D Scatterplot of Pick "
            + str(current + 1)
            + "  of: "
            + str(length)
            + "."
        )

        locs = all_picked_locs[current]
        locs = stack_arrays(locs, asrecarray=True, usemask=False)

        est = KMeans(n_clusters=n_clusters, n_init='auto')

        scaled_locs = lib.append_to_rec(locs, locs["x"], "x_scaled")
        scaled_locs = lib.append_to_rec(scaled_locs, locs["y"], "y_scaled")

        X = np.asarray(scaled_locs["x_scaled"])
        Y = np.asarray(scaled_locs["y_scaled"])

        est.fit(np.stack((X, Y), axis=1))

        labels = est.labels_

        counts = list(Counter(labels).items())
        # l_locs = lib.append_to_rec(l_locs,labels,'cluster')

        ax1.scatter(locs["x"], locs["y"], c=labels.astype(float), s=2)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        counts = list(Counter(labels).items())
        cent = est.cluster_centers_

        ax2.scatter(cent[:, 0], cent[:, 1], s=2)
        for element in counts:
            x_mean = cent[element[0], 0]
            y_mean = cent[element[0], 1]
            dialog.add_clusters(element, x_mean, y_mean)
            ax2.text(x_mean, y_mean, element[1], fontsize=12)

        ax1.set_xlabel("X [Px]")
        ax1.set_ylabel("Y [Px]")

        ax2.set_xlabel("X [nm]")
        ax2.set_ylabel("Y [nm]")

        result = dialog.exec_()

        checks = [not _.isChecked() for _ in dialog.checks]
        checks = np.asarray(np.where(checks)) + 1
        checks = checks[0]

        labels += 1
        labels = [0 if x in checks else x for x in labels]
        labels = np.asarray(labels)

        l_locs = lib.append_to_rec(scaled_locs, labels, "cluster")
        l_locs_new_group = l_locs.copy()
        power = np.round(n_clusters / 10) + 1
        l_locs_new_group["group"] = (
            l_locs_new_group["group"] * 10 ** power
            + l_locs_new_group["cluster"]
        )

        # Combine clustered locs
        clustered_locs = []
        for element in np.unique(labels):
            if element != 0:
                clustered_locs.append(
                    l_locs_new_group[l_locs["cluster"] == element]
                )

        return (
            dialog.result,
            dialog.n_clusters_spin.value(),
            l_locs,
            clustered_locs,
        )


class LinkDialog(QtWidgets.QDialog):
    """
    A class to obtain inputs for linking localizations.

    ...

    Attributes
    ----------
    max_dark_time : QDoubleSpinBox
        contains the maximum gap between localizations (frames) to be
        considered as belonging to the same group of linked locs
    max_distance : QDoubleSpinBox
        contains the maximum distance (pixels) between locs to be
        considered as belonging to the same group of linked locs

    Methods
    -------
    getParams(parent=None)
        Creates the dialog and returns the requested values for linking
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter parameters")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Max. distance (pixels):"), 0, 0)
        self.max_distance = QtWidgets.QDoubleSpinBox()
        self.max_distance.setRange(0, 1e6)
        self.max_distance.setValue(1)
        grid.addWidget(self.max_distance, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Max. transient dark frames:"), 1, 0)
        self.max_dark_time = QtWidgets.QDoubleSpinBox()
        self.max_dark_time.setRange(0, 1e9)
        self.max_dark_time.setValue(1)
        grid.addWidget(self.max_dark_time, 1, 1)
        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent=None):
        """
        Creates the dialog and returns the requested values for 
        linking.
        """

        dialog = LinkDialog(parent)
        result = dialog.exec_()
        return (
            dialog.max_distance.value(),
            dialog.max_dark_time.value(),
            result == QtWidgets.QDialog.Accepted,
        )


class DbscanDialog(QtWidgets.QDialog):
    """
    A class to obtain inputs for DBSCAN.
    See scikit-learn DBSCAN for more info.
    
    ...
    
    Attributes
    ----------
    density : QSpinBox
        contains min_samples for DBSCAN (see scikit-learn)
    radius : QDoubleSpinBox
        contains epsilon (pixels) for DBSCAN (see scikit-learn)
    
    Methods
    -------
    getParams(parent=None)
        Creates the dialog and returns the requested values for DBSCAN
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter parameters")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Radius (pixels):"), 0, 0)
        self.radius = QtWidgets.QDoubleSpinBox()
        self.radius.setRange(0.001, 1e6)
        self.radius.setValue(0.1)
        self.radius.setDecimals(4)
        self.radius.setSingleStep(0.001)
        grid.addWidget(self.radius, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min. samples:"), 1, 0)
        self.density = QtWidgets.QSpinBox()
        self.density.setRange(1, int(1e6))
        self.density.setValue(4)
        grid.addWidget(self.density, 1, 1)
        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # save cluster centers
        self.save_centers = QtWidgets.QCheckBox("Save cluster centers")
        self.save_centers.setChecked(False)
        grid.addWidget(self.save_centers, 2, 0, 1, 2)

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent=None):
        """ 
        Creates the dialog and returns the requested values for DBSCAN.
        """

        dialog = DbscanDialog(parent)
        result = dialog.exec_()
        return (
            dialog.radius.value(),
            dialog.density.value(),
            dialog.save_centers.isChecked(),
            result == QtWidgets.QDialog.Accepted,
        )


class HdbscanDialog(QtWidgets.QDialog):
    """
    A class to obtain inputs for HDBSCAN.
    See https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan
    for more info.

    ...

    Attributes
    ----------
    cluster_eps : QDoubleSpinBox
        contains cluster_selection_epsilon (pixels), see the website
    min_cluster : QSpinBox
        contains the minimum number of locs in a cluster
    min_samples : QSpinBox
        contains the number of locs in a neighbourhood for a loc to be 
        considered a core point, see the website

    Methods
    -------
    getParams(parent=None)
        Creates the dialog and returns the requested values for HDBSCAN
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter parameters")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Min. cluster size:"), 0, 0)
        self.min_cluster = QtWidgets.QSpinBox()
        self.min_cluster.setRange(1, int(1e6))
        self.min_cluster.setValue(10)
        grid.addWidget(self.min_cluster, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min. samples:"), 1, 0)
        self.min_samples = QtWidgets.QSpinBox()
        self.min_samples.setRange(1, int(1e6))
        self.min_samples.setValue(10)
        grid.addWidget(self.min_samples, 1, 1)
        grid.addWidget(QtWidgets.QLabel(
            "Intercluster max.\ndistance (pixels):"), 2, 0
        )
        self.cluster_eps = QtWidgets.QDoubleSpinBox()
        self.cluster_eps.setRange(0, 1e6)
        self.cluster_eps.setValue(0.0)
        self.cluster_eps.setDecimals(3)
        self.cluster_eps.setSingleStep(0.001)
        grid.addWidget(self.cluster_eps, 2, 1)
        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # save cluster centers
        self.save_centers = QtWidgets.QCheckBox("Save cluster centers")
        self.save_centers.setChecked(False)
        grid.addWidget(self.save_centers, 3, 0, 1, 2)

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent=None):
        """
        Creates the dialog and returns the requested values for 
        HDBSCAN.
        """

        dialog = HdbscanDialog(parent)
        result = dialog.exec_()
        return (
            dialog.min_cluster.value(),
            dialog.min_samples.value(),
            dialog.cluster_eps.value(),
            dialog.save_centers.isChecked(),
            result == QtWidgets.QDialog.Accepted,
        )

class SMLMDialog3D(QtWidgets.QDialog):
    """
    A class to obtain inputs for SMLM clusterer (3D).

    ...

    Attributes
    ----------
    radius_xy : QDoubleSpinBox
        contains clustering radius in x and y directions
    radius_z : QDoubleSpinBox
        contains clustering radius in z direction 
        (typically =2*radius_xy)
    min_locs : QSpinBox
        contains minimum number of locs in cluster

    Methods
    -------
    getParams(parent=None)
        Creates the dialog and returns the requested values for 
        clustering
    """
    
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter parameters (3D)")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        # radius xy
        grid.addWidget(QtWidgets.QLabel("Cluster radius xy (pixels):"), 0, 0)
        self.radius_xy = QtWidgets.QDoubleSpinBox()
        self.radius_xy.setRange(0.0001, 1e3)
        self.radius_xy.setDecimals(4)
        self.radius_xy.setValue(0.1)
        grid.addWidget(self.radius_xy, 0, 1)
        # radius z
        grid.addWidget(QtWidgets.QLabel("Cluster radius z (pixels):"), 1, 0)
        self.radius_z = QtWidgets.QDoubleSpinBox()
        self.radius_z.setRange(0, 1e3)
        self.radius_z.setDecimals(4)
        self.radius_z.setValue(0.25)
        grid.addWidget(self.radius_z, 1, 1)
        # min no. locs
        grid.addWidget(QtWidgets.QLabel("Min. no. locs:"), 2, 0)
        self.min_locs = QtWidgets.QSpinBox()
        self.min_locs.setRange(1, int(1e6))
        self.min_locs.setValue(10)
        grid.addWidget(self.min_locs, 2, 1)
        # save cluster centers
        self.save_centers = QtWidgets.QCheckBox("Save cluster centers")
        self.save_centers.setChecked(False)
        grid.addWidget(self.save_centers, 3, 0, 1, 2)
        # perform basic frame analysis
        self.frame_analysis = QtWidgets.QCheckBox(
            "Perform basic frame analysis"
        )
        self.frame_analysis.setChecked(True)
        grid.addWidget(self.frame_analysis, 4, 0, 1, 2)

        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent=None):
        """
        Creates the dialog and returns the requested values for 
        SMLM clusterer (3D).
        """

        dialog = SMLMDialog3D(parent)
        result = dialog.exec_()
        return (
            dialog.radius_xy.value(),
            dialog.radius_z.value(),
            dialog.min_locs.value(),
            dialog.save_centers.isChecked(),
            dialog.frame_analysis.isChecked(),
            result == QtWidgets.QDialog.Accepted,
        )    


class SMLMDialog2D(QtWidgets.QDialog):
    """
    A class to obtain inputs for SMLM clusterer (2D).

    ...

    Attributes
    ----------
    radius : QDoubleSpinBox
        contains clustering radius in x and y directions
    min_locs : QSpinBox
        contains minimum number of locs in cluster

    Methods
    -------
    getParams(parent=None)
        Creates the dialog and returns the requested values for 
        clustering
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter parameters (2D)")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        # clustering radius
        grid.addWidget(QtWidgets.QLabel("Cluster radius (pixels):"), 0, 0)
        self.radius = QtWidgets.QDoubleSpinBox()
        self.radius.setRange(0.0001, 1e3)
        self.radius.setDecimals(4)
        self.radius.setValue(0.1)
        grid.addWidget(self.radius, 0, 1)
        # min no. locs
        grid.addWidget(QtWidgets.QLabel("Min. no. locs:"), 1, 0)
        self.min_locs = QtWidgets.QSpinBox()
        self.min_locs.setRange(1, int(1e6))
        self.min_locs.setValue(10)
        grid.addWidget(self.min_locs, 1, 1)
        # save cluster centers
        self.save_centers = QtWidgets.QCheckBox("Save cluster centers")
        self.save_centers.setChecked(False)
        grid.addWidget(self.save_centers, 2, 0, 1, 2)
        # perform basic frame analysis
        self.frame_analysis = QtWidgets.QCheckBox(
            "Perform basic frame analysis"
        )
        self.frame_analysis.setChecked(True)
        grid.addWidget(self.frame_analysis, 3, 0, 1, 2)

        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent=None):
        """
        Creates the dialog and returns the requested values for 
        SMLM clusterer (2D).
        """

        dialog = SMLMDialog2D(parent)
        result = dialog.exec_()
        return (
            dialog.radius.value(),
            dialog.min_locs.value(),
            dialog.save_centers.isChecked(),
            dialog.frame_analysis.isChecked(),
            result == QtWidgets.QDialog.Accepted,
        )  


class TestClustererDialog(QtWidgets.QDialog):
    """
    A class to test clustering paramaters on a region of interest.

    The user needs to pick a single region of interest using the Pick
    tool. Use Alt + {W, A, S, D, -, =} to change field of view. 

    ...

    Attributes
    ----------
    channel : int
        Channel index for localizations that are tested
    clusterer_name : QComboBox
        contains all clusterer types available in Picasso: Render
    display_all_locs : QCheckBox
        if ticked, unclustered locs are displayed in separete channel
    pick : list
        coordinates of the last pick (region of interest) that was 
        displayed
    pick_size : float
        width (if rectangular) or diameter (if circular) of the pick
    test_dbscan_params : QWidget
        contains widgets with parameters for DBSCAN
    test_hdbscan_params : QWidget
        contains widgets with parameters for HDBSCAN
    test_smlm_params : QWidget
        contains widhtes with parameters for SMLM clusterer
    view : QLabel
        widget for displaying rendered clustered localizations
    window : QMainWindow
        instance of the main Picasso: Render window
    
    Methods
    -------
    assign_groups(locs, labels)
        Filters out non-clustered locs and adds group column to locs
    cluster(locs, params):
        Clusters locs using the chosen clusterer and its params
    get_cluster_params()
        Extracts clustering parameters for a given clusterer into a 
        dictionary
    get_full_fov()
        Updates viewport in self.view
    pick_changed()
        Checks if region of interest has changed since the last 
        rendering
    test_clusterer()
        Prepares clustering parameters, performs clustering and 
        renders localizations    
    """

    def __init__(self, window):
        super().__init__()
        self.setWindowTitle("Test Clusterer")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path) 
        self.setWindowIcon(icon)  
        self.pick = None
        self.pick_size = None
        self.window = window
        self.view = TestClustererView(self)
        layout = QtWidgets.QGridLayout(self)
        self.setLayout(layout)

        # explanation
        layout.addWidget(
            QtWidgets.QLabel(
                "Pick a region of interest and test different clustering\n"
                "algorithms and parameters.\n\n"
                "Use shortcuts Alt + {W, A, S, D, -, =} to change FOV.\n"
            ), 0, 0
        )

        # parameters
        parameters_box = QtWidgets.QGroupBox("Parameters")
        layout.addWidget(parameters_box, 1, 0)
        parameters_grid = QtWidgets.QGridLayout(parameters_box)

        # parameters - choose clusterer
        self.clusterer_name = QtWidgets.QComboBox()
        for name in ['DBSCAN', 'HDBSCAN', 'SMLM']:
            self.clusterer_name.addItem(name)
        parameters_grid.addWidget(self.clusterer_name, 0, 0)

        # parameters - clusterer parameters
        parameters_stack = QtWidgets.QStackedWidget()
        parameters_grid.addWidget(parameters_stack, 1, 0, 1, 2)
        self.clusterer_name.currentIndexChanged.connect(
            parameters_stack.setCurrentIndex
        )
        self.test_dbscan_params = TestDBSCANParams(self)
        parameters_stack.addWidget(self.test_dbscan_params)
        self.test_hdbscan_params = TestHDBSCANParams(self)
        parameters_stack.addWidget(self.test_hdbscan_params)
        self.test_smlm_params = TestSMLMParams(self)
        parameters_stack.addWidget(self.test_smlm_params)

        # parameters - display modes
        self.one_pixel_blur = QtWidgets.QCheckBox("One pixel blur")
        self.one_pixel_blur.setChecked(False)
        self.one_pixel_blur.stateChanged.connect(self.view.update_scene)
        parameters_grid.addWidget(self.one_pixel_blur, 2, 0, 1, 2)

        self.display_all_locs = QtWidgets.QCheckBox(
            "Display non-clustered localizations"
        )
        self.display_all_locs.setChecked(False)
        self.display_all_locs.stateChanged.connect(self.view.update_scene)
        parameters_grid.addWidget(self.display_all_locs, 3, 0, 1, 2)

        self.display_centers = QtWidgets.QCheckBox("Display cluster centers")
        self.display_centers.setChecked(False)
        self.display_centers.stateChanged.connect(self.view.update_scene)
        parameters_grid.addWidget(self.display_centers, 4, 0, 1, 2)

        # parameters - xy, xz, yz projections
        xy_proj = QtWidgets.QPushButton("XY projection")
        xy_proj.clicked.connect(self.on_xy_proj)
        parameters_grid.addWidget(xy_proj, 5, 0, 1, 2)

        xz_proj = QtWidgets.QPushButton("XZ projection")
        xz_proj.clicked.connect(self.on_xz_proj)
        parameters_grid.addWidget(xz_proj, 6, 0)

        yz_proj = QtWidgets.QPushButton("YZ projection")
        yz_proj.clicked.connect(self.on_yz_proj)
        parameters_grid.addWidget(yz_proj, 6, 1)

        # parameters - test
        test_button = QtWidgets.QPushButton("Test")
        test_button.clicked.connect(self.test_clusterer)
        test_button.setDefault(True)
        parameters_grid.addWidget(test_button, 7, 0)

        # display settings - return to full FOV
        full_fov = QtWidgets.QPushButton("Full FOV")
        full_fov.clicked.connect(self.get_full_fov)
        parameters_grid.addWidget(full_fov, 7, 1)

        # view
        view_box = QtWidgets.QGroupBox("View")
        layout.addWidget(view_box, 0, 1, 3, 1)
        view_grid = QtWidgets.QGridLayout(view_box)
        view_grid.addWidget(self.view)

        # shortcuts for navigating in View
        # arrows
        left_action = QtWidgets.QAction(self)
        left_action.setShortcut("Alt+A")
        left_action.triggered.connect(self.view.to_left)
        self.addAction(left_action)

        right_action = QtWidgets.QAction(self)
        right_action.setShortcut("Alt+D")
        right_action.triggered.connect(self.view.to_right)
        self.addAction(right_action)

        up_action = QtWidgets.QAction(self)
        up_action.setShortcut("Alt+W")
        up_action.triggered.connect(self.view.to_up)
        self.addAction(up_action)

        down_action = QtWidgets.QAction(self)
        down_action.setShortcut("Alt+S")
        down_action.triggered.connect(self.view.to_down)
        self.addAction(down_action)

        # zooming
        zoomin_action = QtWidgets.QAction(self)
        zoomin_action.setShortcut("Alt+=")
        zoomin_action.triggered.connect(self.view.zoom_in)
        self.addAction(zoomin_action)

        zoomout_action = QtWidgets.QAction(self)
        zoomout_action.setShortcut("Alt+-")
        zoomout_action.triggered.connect(self.view.zoom_out)
        self.addAction(zoomout_action)

    def on_xy_proj(self):
        self.view.ang = None
        self.view.update_scene()

    def on_xz_proj(self):
        if self.view.locs is not None and hasattr(self.view.locs, "z"):
            self.view.ang = [1.5708, 0, 0] # 90 deg rotation
        else:
            self.view.ang = None
        self.view.update_scene()

    def on_yz_proj(self):
        if self.view.locs is not None and hasattr(self.view.locs, "z"):
            self.view.ang = [0, 1.5708, 0] # 90 deg rotation
        else:
            self.view.ang = None
        self.view.update_scene()

    def cluster(self, locs, params):
        """ 
        Clusters locs using the chosen clusterer. 

        Parameters
        ----------
        locs : np.recarray
            Contains all picked localizations from a given channel
        params : dict
            Contains clustering paramters for a given clusterer

        Returns
        -------
        np.recarray
            Contains localizations that were clustered. Cluster label
            is saved in "group" dtype.
        """

        # for converting z coordinates
        pixelsize = self.window.display_settings_dlg.pixelsize.value()

        clusterer_name = self.clusterer_name.currentText()
        if clusterer_name == "DBSCAN":
            locs = clusterer.dbscan(
                locs, 
                params["radius"],
                params["min_samples"],
                pixelsize,
            )
        elif clusterer_name == "HDBSCAN":
            locs = clusterer.hdbscan(
                locs,
                params["min_cluster_size"],
                params["min_samples"],
                pixelsize,
                params["intercluster_radius"],
            )
        elif clusterer_name == "SMLM":
            if params["frame_analysis"]:
                frame = locs.frame
            else:
                frame = None

            if hasattr(locs, "z"):
                params_c = [
                    params["radius_xy"],
                    params["radius_z"],
                    params["min_cluster_size"],
                    None,
                    params["frame_analysis"],
                    None,
                ]
            else:
                params_c = [
                    params["radius_xy"],
                    params["min_cluster_size"],
                    None,
                    params["frame_analysis"],
                    None,
                ]

            locs = clusterer.cluster(locs, params_c, pixelsize)
        
        if len(locs):
            self.view.group_color = self.window.view.get_group_color(locs)

        # scale z axis if applicable
        if hasattr(locs, "z"):
            locs.z /= pixelsize
        
        return locs

    def get_cluster_params(self):
        """
        Extracts clustering parameters for a given clusterer into a 
        dictionary.
        """

        params = {}
        clusterer_name = self.clusterer_name.currentText()
        if clusterer_name == "DBSCAN":
            params["radius"] = self.test_dbscan_params.radius.value()
            params["min_samples"] = self.test_dbscan_params.min_samples.value()
        elif clusterer_name == "HDBSCAN":
            params["min_cluster_size"] = (
                self.test_hdbscan_params.min_cluster_size.value()
            )
            params["min_samples"] = (
                self.test_hdbscan_params.min_samples.value()
            )
            params["intercluster_radius"] = (
                self.test_hdbscan_params.cluster_eps.value()
            )
        elif clusterer_name == "SMLM":
            params["radius_xy"] = self.test_smlm_params.radius_xy.value()
            params["radius_z"] = self.test_smlm_params.radius_z.value()
            params["min_cluster_size"] = self.test_smlm_params.min_locs.value()
            params["frame_analysis"] = self.test_smlm_params.fa.isChecked()
        return params

    def get_full_fov(self):
        """ Updates viewport in self.view. """

        if self.view.locs is not None:
            self.view.viewport = self.view.get_full_fov()
            self.view.update_scene()

    def test_clusterer(self):
        """
        Prepares clustering parameters, performs clustering and 
        renders localizations.
        """

        # make sure one pick is present
        if len(self.window.view._picks) != 1:
            # display qt warning
            message = "Choose only one pick region"
            QtWidgets.QMessageBox.information(self, "No pick", message)
            return
        # get clustering parameters
        params = self.get_cluster_params()
        # extract picked locs
        self.channel = self.window.view.get_channel("Test clusterer")
        locs = self.window.view.picked_locs(self.channel)[0]
        # cluster picked locs
        self.view.locs = self.cluster(locs, params)
        # calculate cluster centers
        self.view.centers = clusterer.find_cluster_centers(
            self.view.locs,
            self.window.display_settings_dlg.pixelsize.value(),
        )
        # update viewport if pick has changed
        if self.pick_changed():
            self.view.viewport = self.view.get_full_fov()
        if self.view.locs is None:
            message = (
                "No HDBSCAN detected.\nPlease install the python package"
                " HDBSCAN (pip install hdbscan)."
            )
            QtWidgets.QMessageBox.information(self, "No HDBSCAN", message)
            return
        # render clustered locs
        self.view.update_scene()

    def pick_changed(self):
        """
        Checks if region of interest has changed since the last 
        rendering.
        """

        pick = self.window.view._picks[0]
        if self.window.tools_settings_dialog.pick_shape == "Circle":
            pick_size = self.window.tools_settings_dialog.pick_diameter.value()
        else:
            pick_size = self.window.tools_settings_dialog.pick_width.value()
        if pick != self.pick or pick_size != self.pick_size:
            self.pick = pick
            self.pick_size = pick_size
            return True
        else:
            return False


class TestDBSCANParams(QtWidgets.QWidget):
    """
    Class containing user-selected clustering parameters for DBSCAN.
    """

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("Radius (pixels):"), 0, 0)
        self.radius = QtWidgets.QDoubleSpinBox()
        self.radius.setKeyboardTracking(False)
        self.radius.setRange(0.001, 1e6)
        self.radius.setValue(0.1)
        self.radius.setDecimals(3)
        self.radius.setSingleStep(0.001)
        grid.addWidget(self.radius, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Min. samples:"), 1, 0)
        self.min_samples = QtWidgets.QSpinBox()
        self.min_samples.setKeyboardTracking(False)
        self.min_samples.setValue(4)
        self.min_samples.setRange(1, int(1e6))
        self.min_samples.setSingleStep(1)
        grid.addWidget(self.min_samples, 1, 1)
        grid.setRowStretch(2, 1)
 

class TestHDBSCANParams(QtWidgets.QWidget):
    """
    Class containing user-selected clustering parameters for HDBSCAN.
    """

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("Min. cluster size:"), 0, 0)
        self.min_cluster_size = QtWidgets.QSpinBox()
        self.min_cluster_size.setKeyboardTracking(False)
        self.min_cluster_size.setValue(10)
        self.min_cluster_size.setRange(1, int(1e6))
        self.min_cluster_size.setSingleStep(1)
        grid.addWidget(self.min_cluster_size, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Min. samples"), 1, 0)     
        self.min_samples = QtWidgets.QSpinBox()
        self.min_samples.setKeyboardTracking(False)
        self.min_samples.setValue(10)
        self.min_samples.setRange(1, int(1e6))
        self.min_samples.setSingleStep(1)
        grid.addWidget(self.min_samples, 1, 1)

        grid.addWidget(
            QtWidgets.QLabel("Intercluster max.\ndistance (pixels):"), 2, 0
        )
        self.cluster_eps = QtWidgets.QDoubleSpinBox()
        self.cluster_eps.setKeyboardTracking(False)
        self.cluster_eps.setRange(0, 1e6)
        self.cluster_eps.setValue(0.0)
        self.cluster_eps.setDecimals(3)
        self.cluster_eps.setSingleStep(0.001)
        grid.addWidget(self.cluster_eps, 2, 1)
        grid.setRowStretch(3, 1)

class TestSMLMParams(QtWidgets.QWidget):
    """
    Class containing user-selected clustering parameters for SMLM 
    clusterer.
    """

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        grid = QtWidgets.QGridLayout(self)
        grid.addWidget(QtWidgets.QLabel("Radius xy [pixels]:"), 0, 0)
        self.radius_xy = QtWidgets.QDoubleSpinBox()
        self.radius_xy.setKeyboardTracking(False)
        self.radius_xy.setValue(0.1)
        self.radius_xy.setRange(0.0001, 1e3)
        self.radius_xy.setDecimals(4)
        grid.addWidget(self.radius_xy, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Radius z (3D only):"), 1, 0)
        self.radius_z = QtWidgets.QDoubleSpinBox()
        self.radius_z.setKeyboardTracking(False)
        self.radius_z.setValue(0.25)
        self.radius_z.setRange(0, 1e3)
        self.radius_z.setDecimals(4)
        grid.addWidget(self.radius_z, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Min. no. locs"), 2, 0)     
        self.min_locs = QtWidgets.QSpinBox()
        self.min_locs.setKeyboardTracking(False)
        self.min_locs.setValue(10)
        self.min_locs.setRange(1, int(1e6))
        self.min_locs.setSingleStep(1)
        grid.addWidget(self.min_locs, 2, 1)

        self.fa = QtWidgets.QCheckBox("Frame analysis")
        self.fa.setChecked(True)
        grid.addWidget(self.fa, 3, 0, 1, 2)
        grid.setRowStretch(4, 1)

class TestClustererView(QtWidgets.QLabel):
    """
    Class used for rendering and displaying clustered localizations.

    ...

    Attributes
    ----------
    dialog : QDialog
        Instance of the Test Clusterer dialog
    locs : np.recarray
        Clustered localizations
    _size : int
        Specifies size of this widget (display pixels)
    view : QLabel
        Instance of View class. Used for calling functions
    viewport : list
        Contains two elements specifying min and max values of x and y
        to be displayed.

    Methods
    -------
    get_full_fov()
        Finds field of view that contains all localizations
    get_optimal_oversampling()
        Finds optimal oversampling for the current viewport
    scale_contrast(images)
        Finds optimal contrast for images
    shift_viewport(dx, dy)
        Moves viewport by a specified amount
    split_locs()
        Splits self.locs into a list. It has either two, three or 
        N_GROUP_COLORS elements (each for one group color).
    to_down()
        Shifts viewport downwards
    to_left()
        Shifts viewport to the left
    to_right()
        Shifts viewport to the right
    to_up()
        Shifts viewport upwards
    update_scene()
        Renders localizations
    viewport_height()
        Returns viewport's height in pixels
    viewport_width()
        Returns viewport's width in pixels
    zoom(factor)
        Changes size of viewport given factor
    zoom_in()
        Decreases size of viewport
    zoom_out()
        Increases size of viewport
    """

    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog
        self.view = dialog.window.view
        self.viewport = None
        self.locs = None
        self.ang = None
        self._size = 500
        self.setMinimumSize(self._size, self._size)
        self.setMaximumSize(self._size, self._size)

    def to_down(self):
        """ Shifts viewport downwards. """

        if self.viewport is not None:
            h = self.viewport_height()
            dy = 0.3 * h
            self.shift_viewport(0, dy)

    def to_left(self):
        """ Shifts viewport to the left. """

        if self.viewport is not None:
            w = self.viewport_width()
            dx = -0.3 * w
            self.shift_viewport(dx, 0)

    def to_right(self):
        """ Shifts viewport to the right. """

        if self.viewport is not None:
            w = self.viewport_width()
            dx = 0.3 * w
            self.shift_viewport(dx, 0)

    def to_up(self):
        """ Shifts viewport upwards. """

        if self.viewport is not None:
            h = self.viewport_height()
            dy = -0.3 * h
            self.shift_viewport(0, dy)

    def zoom_in(self):
        """ Decreases size of viewport. """
        
        if self.viewport is not None:
            self.zoom(1 / ZOOM)

    def zoom_out(self):
        """ Increases size of viewport. """

        if self.viewport is not None:
            self.zoom(ZOOM)

    def zoom(self, factor):
        """ 
        Changes size of viewport. 

        Paramaters
        ----------
        factor : float
            Specifies the factor by which viewport is changed
        """

        height = self.viewport_height()
        width = self.viewport_width()
        new_height = height * factor
        new_width = width * factor
        center_y, center_x = self.view.viewport_center(self.viewport)
        self.viewport = [
            (center_y - new_height / 2, center_x - new_width / 2),
            (center_y + new_height / 2, center_x + new_width / 2),
        ]
        self.update_scene()

    def viewport_width(self):
        """ Returns viewport's width in pixels. """

        return self.viewport[1][1] - self.viewport[0][1]

    def viewport_height(self):
        """ Returns viewport's height in pixels. """

        return self.viewport[1][0] - self.viewport[0][0]

    def shift_viewport(self, dx, dy):
        """ Moves viewport by a specified amount. """

        (y_min, x_min), (y_max, x_max) = self.viewport
        self.viewport = [(y_min + dy, x_min + dx), (y_max + dy, x_max + dx)]
        self.update_scene()   

    def update_scene(self):
        """ Renders localizations. """

        if not len(self.locs):
            self.setText("No clusters found with the current settings.")
            return

        if self.viewport is None:
            self.viewport = self.get_full_fov()

        # split locs according to their group colors
        locs = self.split_locs()

        # render kwargs
        if self.dialog.one_pixel_blur.isChecked():
            blur_method = 'smooth'
        else:
            blur_method = 'convolve'
        kwargs = {
            'oversampling': self.get_optimal_oversampling(),
            'viewport': self.viewport,
            'blur_method': blur_method,
            'min_blur_width': 0,
            'ang': self.ang,
        }

        # render images for all channels
        images = [render.render(_, **kwargs)[1] for _ in locs]

        # scale image 
        images = self.scale_contrast(images)

        # create image to display
        Y, X = images.shape[1:]
        bgra = np.zeros((Y, X, 4), dtype=np.float32)
        colors = get_colors(images.shape[0])
        for color, image in zip(colors, images): # color each channel
            bgra[:, :, 0] += color[2] * image
            bgra[:, :, 1] += color[1] * image
            bgra[:, :, 2] += color[0] * image
        bgra = np.minimum(bgra, 1)
        bgra = self.view.to_8bit(bgra)
        bgra[:, :, 3].fill(255) # black background
        qimage = QtGui.QImage(
            bgra.data, X, Y, QtGui.QImage.Format_RGB32
        ).scaled(
            self._size, 
            self._size, 
            QtCore.Qt.KeepAspectRatioByExpanding
        )
        self.setPixmap(QtGui.QPixmap.fromImage(qimage))

    def split_locs(self):
        """
        Splits self.locs into a list that specifies either separate
        channels (all localizations, clusters and cluster centers) or
        it separates clustered localizations by color (based on group,
        i.e., the cluster id).
        """

        if (
            self.dialog.display_all_locs.isChecked()
            and not self.dialog.display_centers.isChecked()
        ): # two channels, all locs and clustered locs
            channel = self.dialog.channel
            all_locs = self.dialog.window.view.picked_locs(channel)[0]
            if hasattr(all_locs, "z"):
                all_locs.z /= (
                    self.dialog.window.display_settings_dlg.pixelsize.value()
                )
            locs = [
                all_locs,
                self.locs,
            ]
        elif (
            not self.dialog.display_all_locs.isChecked()
            and self.dialog.display_centers.isChecked()
        ): # two channels, clustered locs and cluster centers
            locs = [
                self.locs,
                self.centers,
            ]
        elif (
            self.dialog.display_all_locs.isChecked()
            and self.dialog.display_centers.isChecked()
        ): # three channels, all locs, clustered locs and cluster centers
            channel = self.dialog.channel
            all_locs = self.dialog.window.view.picked_locs(channel)[0]
            if hasattr(all_locs, "z"):
                all_locs.z /= (
                    self.dialog.window.display_settings_dlg.pixelsize.value()
                )
            locs = [
                all_locs,
                self.locs,
                self.centers,
            ]  
        else:
            # multiple channels, each for one group color
            locs = [
                self.locs[self.group_color == _] for _ in range(N_GROUP_COLORS)
            ]
        return locs

    def get_optimal_oversampling(self):
        """
        Finds optimal oversampling for the current viewport.

        Returns
        -------
        float
            The optimal oversampling, i.e. number of display pixels per
            camera pixels 
        """

        height = self.viewport_height()
        width = self.viewport_width()
        return (self._size / min(height, width)) / 1.05

    def scale_contrast(self, images):
        """
        Finds optimal contrast for images.

        Parameters
        ----------
        images : list of np.arrays
            Arrays with rendered locs (grayscale)

        Returns
        -------
        list of np.arrays
            Scaled images
        """

        upper = min(
            [
                _.max() 
                for _ in images  # if no locs were clustered
                if _.max() != 0  # the maximum value in image is 0.0
            ]
        ) / 4
        # upper = INITIAL_REL_MAXIMUM * max_

        images = images / upper
        images[~np.isfinite(images)] = 0
        images = np.minimum(images, 1.0)
        images = np.maximum(images, 0.0)
        return images

    def get_full_fov(self):
        """
        Finds field of view that contains all localizations.

        Returns
        -------
        list
            Specifies viewport
        """

        if not len(self.locs):
            return
        x_min = np.min(self.locs.x) - 1
        x_max = np.max(self.locs.x) + 1
        y_min = np.min(self.locs.y) - 1
        y_max = np.max(self.locs.y) + 1
        return ([y_min, x_min], [y_max, x_max])


class DriftPlotWindow(QtWidgets.QTabWidget):
    """
    A class to plot drift (2D or 3D).

    ...

    Attributes
    ----------
    figure : plt.Figure
    canvas : FigureCanvas
        PyQt5 backend used for displaying plots

    Methods
    -------
    plot_2D(drift)
        Creates 2 plots with drift
    plot_3d(drift)
        Creates 3 plots with drift
    """

    def __init__(self, info_dialog):
        super().__init__()
        self.setWindowTitle("Drift Plot")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        vbox.addWidget((NavigationToolbar(self.canvas, self)))


    def plot_3d(self, drift):
        """
        Creates 3 plots: frames vs x/y, x vs y in time, frames vs z.

        Parameters
        ----------
        drift : np.recarray
            Contains 3 dtypes: x, y and z. Stores drift in each 
            coordinate (pixels)
        """

        self.figure.clear()

        ax1 = self.figure.add_subplot(131)
        ax1.plot(drift.x, label="x")
        ax1.plot(drift.y, label="y")
        ax1.legend(loc="best")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Drift (pixel)")
        ax2 = self.figure.add_subplot(132)
        ax2.plot(
          drift.x,
          drift.y,
          color=list(plt.rcParams["axes.prop_cycle"])[2][
              "color"
          ],
        )

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax3 = self.figure.add_subplot(133)
        ax3.plot(drift.z, label="z")
        ax3.legend(loc="best")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Drift (nm)")
        self.canvas.draw()

    def plot_2d(self, drift):
        """
        Creates 2 plots: frames vs x/y, x vs y in time.

        Parameters
        ----------
        drift : np.recarray
            Contains 2 dtypes: x and y. Stores drift in each 
            coordinate (pixels)
        """

        self.figure.clear()

        ax1 = self.figure.add_subplot(121)
        ax1.plot(drift.x, label="x")
        ax1.plot(drift.y, label="y")
        ax1.legend(loc="best")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Drift (pixel)")
        ax2 = self.figure.add_subplot(122)
        ax2.plot(
          drift.x,
          drift.y,
          color=list(plt.rcParams["axes.prop_cycle"])[2][
              "color"
          ],
        )

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        self.canvas.draw()


class ChangeFOV(QtWidgets.QDialog):
    """
    A class for manually changing field of view.

    ...

    Attributes
    ----------
    h_box : QDoubleSpinBox
        contains the height of the viewport (pixels)
    w_box : QDoubleSpinBox
        contains the width of the viewport (pixels)
    x_box : QDoubleSpinBox
        contains the minimum x coordinate (pixels) to be displayed
    y_box : QDoubleSpinBox
        contains the minimum y coordinate (pixels) to be displayed
    
    Methods
    -------
    load_fov()
        Used for loading a FOV from a .txt file
    save_fov()
        Used for saving the current FOV as a .txt file
    update_scene()
        Updates the scene in the main window and Display section of the
        Info Dialog
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Change FOV")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.x_box = QtWidgets.QDoubleSpinBox()
        self.x_box.setKeyboardTracking(False)
        self.x_box.setRange(-100, 1e6)
        self.layout.addWidget(self.x_box, 0, 1)
        self.layout.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
        self.y_box = QtWidgets.QDoubleSpinBox()
        self.y_box.setKeyboardTracking(False)
        self.y_box.setRange(-100, 1e6)
        self.layout.addWidget(self.y_box, 1, 1)
        self.layout.addWidget(QtWidgets.QLabel("Width:"), 2, 0)
        self.w_box = QtWidgets.QDoubleSpinBox()
        self.w_box.setKeyboardTracking(False)
        self.w_box.setRange(0, 1e3)
        self.layout.addWidget(self.w_box, 2, 1)
        self.layout.addWidget(QtWidgets.QLabel("Height:"), 3, 0)
        self.h_box = QtWidgets.QDoubleSpinBox()
        self.h_box.setKeyboardTracking(False)
        self.h_box.setRange(0, 1e3)
        self.layout.addWidget(self.h_box, 3, 1)
        self.apply = QtWidgets.QPushButton("Apply")
        self.layout.addWidget(self.apply, 4, 0)
        self.apply.clicked.connect(self.update_scene)
        self.savefov = QtWidgets.QPushButton("Save FOV")
        self.layout.addWidget(self.savefov, 5, 0)
        self.savefov.clicked.connect(self.save_fov)
        self.loadfov = QtWidgets.QPushButton("Load FOV")
        self.layout.addWidget(self.loadfov, 6, 0)
        self.loadfov.clicked.connect(self.load_fov)

    def save_fov(self):
        """ Used for saving the current FOV as a .txt file. """
        
        path = self.window.view.locs_paths[0]
        base, ext = os.path.splitext(path)
        out_path = base + "_fov.txt"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save FOV to", out_path, filter="*.txt"
        )
        fov = np.array([
            self.x_box.value(),
            self.y_box.value(),
            self.w_box.value(),
            self.h_box.value(),
        ])
        np.savetxt(path, fov)

    def load_fov(self):
        """ Used for loading a FOV from a .txt file. """

        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load FOV from", filter="*.txt"
        )
        [x, y, w, h] = np.loadtxt(path)
        self.x_box.setValue(x)
        self.y_box.setValue(y)
        self.w_box.setValue(w)
        self.h_box.setValue(h)
        self.update_scene()

    def update_scene(self):
        """ 
        Updates the scene in the main window and Display section of the
        Info Dialog.
        """
        
        x_min = self.x_box.value()
        y_min = self.y_box.value()
        x_max = self.x_box.value() + self.w_box.value()
        y_max = self.y_box.value() + self.h_box.value()
        viewport = [(y_min, x_min), (y_max, x_max)]
        self.window.view.update_scene(viewport=viewport)
        self.window.info_dialog.xy_label.setText(
            "{:.2f} / {:.2f} ".format(x_min, y_min)
        )
        self.window.info_dialog.wh_label.setText(
            "{:.2f} / {:.2f} pixel".format(
            self.w_box.value(), self.h_box.value()
            )
        )


class InfoDialog(QtWidgets.QDialog):
    """
    A class to show information about the current display, fit
    precision, number of locs and picks, including QPAINT.

    ...

    Attributes
    ----------    
    change_display : QPushButton
        opens self.change_fov
    change_fov : ChangeFOV(QDialog)
        dialog for changing field of view
    height_label : QLabel
        contains the height of the window (pixels)
    dark_mean : QLabel
        shows the mean dark time (frames) in all picks
    dark_std : QLabel
        shows the std dark time (frames) in all picks
    fit_precision : QLabel
        shows median fit precision of the first channel (pixels)
    influx_rate : FloadEdit(QLineEdit)
        contains the calculated or input influx rate (1/frames)
    locs_label : QLabel
        shows the number of locs in the current FOV
    lp: float
        NeNA localization precision (pixels). None, if not calculated yet
    max_dark_time : QSpinBox
        contains the maximum gap between localizations (frames) to be
        considered as belonging to the same group of linked locs
    movie_grid : QGridLayout
        contains all the info about the fit precision
    nena_button : QPushButton
        calculates nearest neighbor based analysis fit precision
    n_localization_mean : QLabel
        shows the mean number of locs in all picks
    n_localization_std : QLabel
        shows the std number of locs in all picks
    n_picks : QLabel
        shows the number of picks
    n_units_mean : QLabel
        shows the calculated mean number of binding sites in all picks
    n_units_std : QLabel
        shows the calculated std number of binding sites in all picks
    picks_grid : QGridLayout
        contains all the info about the picks
    rmsd_mean : QLabel
        shows the mean root mean square displacement in all picks in
        x and y axes
    rmsd_std : QLabel
        shows the std root mean square displacement in all picks in
        x and y axes
    rmsd_z_mean : QLabel
        shows the mean root mean square displacement in all picks in 
        z axis
    rmsd_z_std : QLabel
        shows the std root mean square displacement in all picks in
        z axis
    units_per_pick : QSpinBox
        contains the number of binding sites per pick
    wh_label : QLabel
        displays the width and height of FOV (pixels)
    window : Window(QMainWindow)
        main window instance
    width_label : QLabel
        contains the width of the window (pixels)
    xy_label : QLabel
        shows the minimum y and u coordinates in FOV (pixels) 

    Methods
    -------
    calibrate_influx()
        Calculates influx rate (1/frames)
    calculate_nena_lp()
        Calculates and plots NeNA precision in a given channel
    calculate_n_units()
        Calculates number of units in each pick
    udpate_n_units()
        Displays the mean and std number of units in the Dialog
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Info")
        self.setModal(False)
        self.lp = None
        self.nena_calculated = False
        self.change_fov = ChangeFOV(self.window)
        vbox = QtWidgets.QVBoxLayout(self)
        # Display
        display_groupbox = QtWidgets.QGroupBox("Display")
        vbox.addWidget(display_groupbox)
        display_grid = QtWidgets.QGridLayout(display_groupbox)
        display_grid.addWidget(QtWidgets.QLabel("Image width:"), 0, 0)
        self.width_label = QtWidgets.QLabel()
        display_grid.addWidget(self.width_label, 0, 1)
        display_grid.addWidget(QtWidgets.QLabel("Image height:"), 1, 0)
        self.height_label = QtWidgets.QLabel()
        display_grid.addWidget(self.height_label, 1, 1)

        display_grid.addWidget(QtWidgets.QLabel("View X / Y:"), 2, 0)
        self.xy_label = QtWidgets.QLabel()
        display_grid.addWidget(self.xy_label, 2, 1)

        display_grid.addWidget(QtWidgets.QLabel("View width / height:"), 3, 0)
        self.wh_label = QtWidgets.QLabel()
        display_grid.addWidget(self.wh_label, 3, 1)

        self.change_display = QtWidgets.QPushButton("Change field of view")
        display_grid.addWidget(self.change_display, 4, 0)
        self.change_display.clicked.connect(self.change_fov.show)

        # Movie
        movie_groupbox = QtWidgets.QGroupBox("Movie")
        vbox.addWidget(movie_groupbox)
        self.movie_grid = QtWidgets.QGridLayout(movie_groupbox)
        self.movie_grid.addWidget(
            QtWidgets.QLabel("Median fit precision:"), 0, 0
        )
        self.fit_precision = QtWidgets.QLabel("-")
        self.movie_grid.addWidget(self.fit_precision, 0, 1)
        self.movie_grid.addWidget(QtWidgets.QLabel("NeNA precision:"), 1, 0)
        self.nena_button = QtWidgets.QPushButton("Calculate")
        self.nena_button.clicked.connect(self.calculate_nena_lp)
        self.nena_button.setDefault(False)
        self.nena_button.setAutoDefault(False)
        self.movie_grid.addWidget(self.nena_button, 1, 1)
        # FOV
        fov_groupbox = QtWidgets.QGroupBox("Field of view")
        vbox.addWidget(fov_groupbox)
        fov_grid = QtWidgets.QGridLayout(fov_groupbox)
        fov_grid.addWidget(QtWidgets.QLabel("# Localizations:"), 0, 0)
        self.locs_label = QtWidgets.QLabel()
        fov_grid.addWidget(self.locs_label, 0, 1)

        # Picks
        picks_groupbox = QtWidgets.QGroupBox("Picks")
        vbox.addWidget(picks_groupbox)
        self.picks_grid = QtWidgets.QGridLayout(picks_groupbox)
        self.picks_grid.addWidget(QtWidgets.QLabel("# Picks:"), 0, 0)
        self.n_picks = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.n_picks, 0, 1)
        compute_pick_info_button = QtWidgets.QPushButton(
            "Calculate info below"
        )
        compute_pick_info_button.clicked.connect(
            self.window.view.update_pick_info_long
        )
        self.picks_grid.addWidget(compute_pick_info_button, 1, 0, 1, 3)
        self.picks_grid.addWidget(QtWidgets.QLabel("<b>Mean</b"), 2, 1)
        self.picks_grid.addWidget(QtWidgets.QLabel("<b>Std</b>"), 2, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(QtWidgets.QLabel("# Localizations:"), row, 0)
        self.n_localizations_mean = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.n_localizations_mean, row, 1)
        self.n_localizations_std = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.n_localizations_std, row, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(QtWidgets.QLabel("RMSD to COM:"), row, 0)
        self.rmsd_mean = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.rmsd_mean, row, 1)
        self.rmsd_std = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.rmsd_std, row, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(QtWidgets.QLabel("RMSD in z:"), row, 0)
        self.rmsd_z_mean = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.rmsd_z_mean, row, 1)
        self.rmsd_z_std = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.rmsd_z_std, row, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(
            QtWidgets.QLabel("Ignore dark times <="), row, 0
        )
        self.max_dark_time = QtWidgets.QSpinBox()
        self.max_dark_time.setRange(0, int(1e9))
        self.max_dark_time.setValue(1)
        self.picks_grid.addWidget(self.max_dark_time, row, 1, 1, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(QtWidgets.QLabel("Length:"), row, 0)
        self.length_mean = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.length_mean, row, 1)
        self.length_std = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.length_std, row, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(QtWidgets.QLabel("Dark time:"), row, 0)
        self.dark_mean = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.dark_mean, row, 1)
        self.dark_std = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.dark_std, row, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(
            QtWidgets.QLabel("# Units per pick:"), row, 0
        )
        self.units_per_pick = QtWidgets.QSpinBox()
        self.units_per_pick.setRange(1, int(1e6))
        self.units_per_pick.setValue(1)
        self.picks_grid.addWidget(self.units_per_pick, row, 1, 1, 2)
        calculate_influx_button = QtWidgets.QPushButton("Calibrate influx")
        calculate_influx_button.clicked.connect(self.calibrate_influx)
        self.picks_grid.addWidget(
            calculate_influx_button, self.picks_grid.rowCount(), 0, 1, 3
        )
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(
            QtWidgets.QLabel("Influx rate (1/frames):"), row, 0
        )
        self.influx_rate = FloatEdit()
        self.influx_rate.setValue(0.03)
        self.influx_rate.valueChanged.connect(self.update_n_units)
        self.picks_grid.addWidget(self.influx_rate, row, 1, 1, 2)
        row = self.picks_grid.rowCount()
        self.picks_grid.addWidget(QtWidgets.QLabel("# Units:"), row, 0)
        self.n_units_mean = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.n_units_mean, row, 1)
        self.n_units_std = QtWidgets.QLabel()
        self.picks_grid.addWidget(self.n_units_std, row, 2)
        self.pick_hist_window = PickHistWindow(self)
        pick_hists = QtWidgets.QPushButton("Histograms")
        pick_hists.clicked.connect(self.pick_hist_window.show)
        self.picks_grid.addWidget(
            pick_hists, self.picks_grid.rowCount(), 0, 1, 3
        )

    def calculate_nena_lp(self):
        """ 
        Calculates and plots NeNA precision in a given channel. """

        channel = self.window.view.get_channel("Calculate NeNA precision")
        if channel is not None:
            locs = self.window.view.locs[channel]
            info = self.window.view.infos[channel]

            # calculate nena
            progress = lib.ProgressDialog(
                "Calculating NeNA precision", 0, 100, self
            )
            result_lp = postprocess.nena(locs, info, progress.set_value)

            # modify the movie grid
            if not self.nena_calculated: # if nena calculated first time
                self.nena_button.setParent(None)
                self.movie_grid.removeWidget(self.nena_button)
            else:
                self.movie_grid.removeWidget(self.nena_label)
                self.movie_grid.removeWidget(self.show_plot_button)

            self.nena_label = QtWidgets.QLabel()
            self.movie_grid.addWidget(self.nena_label, 1, 1)
            self.nena_result, self.lp = result_lp
            self.lp *= self.window.display_settings_dlg.pixelsize.value()
            self.nena_label.setText("{:.3} nm".format(self.lp))

            # Nena plot
            self.nena_window = NenaPlotWindow(self)
            self.nena_window.plot(self.nena_result)

            self.show_plot_button = QtWidgets.QPushButton("Show plot")
            self.show_plot_button.clicked.connect(self.nena_window.show)
            self.movie_grid.addWidget(self.show_plot_button, 0, 2)
            
            if not self.nena_calculated:
                # add recalculate nena
                recalculate_nena = QtWidgets.QPushButton("Recalculate NeNA")
                recalculate_nena.clicked.connect(self.calculate_nena_lp)
                recalculate_nena.setDefault(False)
                recalculate_nena.setAutoDefault(False)
                self.movie_grid.addWidget(recalculate_nena, 1, 2)

            self.nena_calculated = True

    def calibrate_influx(self):
        """ Calculates influx rate (1/frames). """

        influx = (
            1 / self.pick_info["pooled dark"] / self.units_per_pick.value()
        )
        self.influx_rate.setValue(influx)
        self.update_n_units()

    def calculate_n_units(self, dark):
        """ Calculates number of units in each pick. """

        influx = self.influx_rate.value()
        return 1 / (influx * dark)

    def update_n_units(self):
        """ 
        Displays the mean and std number of units in the 
        Dialog.
        """
        
        n_units = self.calculate_n_units(self.pick_info["dark"])
        self.n_units_mean.setText("{:,.2f}".format(np.mean(n_units)))
        self.n_units_std.setText("{:,.2f}".format(np.std(n_units)))


class NenaPlotWindow(QtWidgets.QTabWidget):
    """ A class to plot NeNA precision. """

    def __init__(self, info_dialog):
        super().__init__()
        self.setWindowTitle("Nena Plot")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        vbox.addWidget((NavigationToolbar(self.canvas, self)))


    def plot(self, nena_result):
        self.figure.clear()
        d = nena_result.userkws["d"]
        ax = self.figure.add_subplot(111)
        ax.set_title("Next frame neighbor distance histogram")
        ax.plot(d, nena_result.data, label="Data")
        ax.plot(d, nena_result.best_fit, label="Fit")
        ax.set_xlabel("Distance (Px)")
        ax.set_ylabel("Counts")
        ax.legend(loc="best")

        self.canvas.draw()


class MaskSettingsDialog(QtWidgets.QDialog):
    """
    A class to mask localizations based on their density.

    ...

    Attributes
    ----------
    ax1 : plt.axes.Axes
        axis where all locs are shown with a given oversampling
    ax2 : plt.axes.Axes
        axis where blurred locs are shown
    ax3 : plt.axes.Axes
        axis where binary mask is shown
    ax4 : plt.axes.Axes
        axis where masked locs are shown (initially shows only zeros)
    cached_blur : int
        0 if image is to be blurred, 1 otherwise
    cached_oversampling : int
        0 if image is to be redrawn, 1 otherwise
    cached_thresh : int
        0 if mask is to be calculated, 1 otherwise
    canvas : FigureCanvas
        canvas used for plotting
    channel : int
        channel of localizations that are plotted in the canvas
    cmap : str
        colormap used in displaying images, same as in the main window
    disp_px_size : QSpinBox
        contains the display pixel size [nm]
    figure : plt.figure.Figure
        figure containg subplots
    index_locs : list
        localizations that were masked; may contain a single or all
        channels
    index_locs_out : list
        localizations that were not masked; may contain a single or 
        all channels
    infos : list
        contains .yaml metadata files for all locs channels loaded when
        starting the dialog
    H : np.array
        histogram displaying all localizations loaded; displayed in ax1
    H_blur : np.array
        histogram displaying blurred localizations; displayed in ax2
    H_new : np.array
        histogram displaying masked localizations; displayed in ax4
    locs : list
        contains all localizations loaded when starting the dialog
    mask : np.array
        histogram displaying binary mask; displayed in ax3
    mask_blur : QDoubleSpinBox
        contains the blur value
    mask_loaded : bool
        True, if mask was loaded from an external file
    mask_thresh : QDoubleSpinBox
        contains the threshold value for masking
    paths : list
        contains paths to all localizations loaded when starting the
        dialog
    save_all : QCheckBox
        if checked, all channels loaded are masked; otherwise only 
        one channel
    save_button : QPushButton
        used for saving masked localizations
    save_mask_button : QPushButton
        used for saving the current mask as a .npy
    _size_hint : tuple
        determines the minimum size of the dialog
    window : QMainWindow
        instance of the main window
    x_max : float
        width of the loaded localizations
    y_max : float
        height of the loaded localizations

    Methods
    -------
    blur_image()
        Blurs localizations using a Gaussian filter
    generate_image()
        Histograms loaded localizations from a given channel
    init_dialog()
        Initializes dialog when called from the main window
    load_mask()
        Loads binary mask from .npy format
    mask_image()
        Calculates binary mask based on threshold
    mask_locs()
        Masks localizations from a single or all channels
    _mask_locs(locs)
        Masks locs given a mask
    save_mask()
        Saves binary mask into .npy format
    save_locs()
        Saves masked localizations
    save_locs_multi()
        Saves masked localizations for all loaded channels
    update_plots()
        Plots in all 4 axes
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Generate Mask")
        self.setModal(False)
        self.channel = 0
        self._size_hint = (670, 840)
        self.setMinimumSize(*self._size_hint)

        vbox = QtWidgets.QVBoxLayout(self)
        mask_groupbox = QtWidgets.QGroupBox("Mask Settings")
        vbox.addWidget(mask_groupbox)
        mask_grid = QtWidgets.QGridLayout(mask_groupbox)

        mask_grid.addWidget(QtWidgets.QLabel("Display pixel size [nm]"), 0, 0)
        self.disp_px_size = QtWidgets.QSpinBox()
        self.disp_px_size.setRange(10, 99999)
        self.disp_px_size.setValue(300)
        self.disp_px_size.setSingleStep(10)
        self.disp_px_size.setKeyboardTracking(False)
        self.disp_px_size.valueChanged.connect(self.update_plots)
        mask_grid.addWidget(self.disp_px_size, 0, 1)

        mask_grid.addWidget(QtWidgets.QLabel("Blur"), 1, 0)
        self.mask_blur = QtWidgets.QDoubleSpinBox()
        self.mask_blur.setRange(0, 9999)
        self.mask_blur.setValue(1)
        self.mask_blur.setSingleStep(0.1)
        self.mask_blur.setDecimals(5)
        self.mask_blur.setKeyboardTracking(False)
        self.mask_blur.valueChanged.connect(self.update_plots)
        mask_grid.addWidget(self.mask_blur, 1, 1)

        mask_grid.addWidget(QtWidgets.QLabel("Threshold"), 2, 0)
        self.mask_thresh = QtWidgets.QDoubleSpinBox()
        self.mask_thresh.setRange(0, 1)
        self.mask_thresh.setValue(0.5)
        self.mask_thresh.setSingleStep(0.01)
        self.mask_thresh.setDecimals(5)
        self.mask_thresh.setKeyboardTracking(False)
        self.mask_thresh.valueChanged.connect(self.update_plots)
        mask_grid.addWidget(self.mask_thresh, 2, 1)

        gridspec_dict = {
            'bottom': 0.05, 
            'top': 0.95, 
            'left': 0.05, 
            'right': 0.95,
        }
        (
            self.figure, 
            ((self.ax1, self.ax2), (self.ax3, self.ax4)),
        ) = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw=gridspec_dict)
        self.canvas = FigureCanvas(self.figure)
        mask_grid.addWidget(self.canvas, 3, 0, 1, 2)

        self.save_all = QtWidgets.QCheckBox("Mask all channels")
        self.save_all.setChecked(False)
        mask_grid.addWidget(self.save_all, 4, 0)

        load_mask_button = QtWidgets.QPushButton("Load Mask")
        load_mask_button.setFocusPolicy(QtCore.Qt.NoFocus)
        load_mask_button.clicked.connect(self.load_mask)
        mask_grid.addWidget(load_mask_button, 5, 0)

        self.save_mask_button = QtWidgets.QPushButton("Save Mask")
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_mask_button.clicked.connect(self.save_mask)
        mask_grid.addWidget(self.save_mask_button, 5, 1)

        mask_button = QtWidgets.QPushButton("Mask")
        mask_button.setFocusPolicy(QtCore.Qt.NoFocus)
        mask_button.clicked.connect(self.mask_locs)
        mask_grid.addWidget(mask_button, 6, 0)

        self.save_button = QtWidgets.QPushButton("Save localizations")
        self.save_button.setEnabled(False)
        self.save_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_button.clicked.connect(self.save_locs)
        mask_grid.addWidget(self.save_button, 6, 1)

        self.cached_oversampling = 0
        self.cached_blur = 0
        self.cached_thresh = 0
        self.mask_loaded = False

    def init_dialog(self):
        """
        Initializes dialog when called from the main window.

        Loades localizations and metadata, updates plots.
        """

        self.mask_loaded = False
        self.locs = self.window.view.locs
        self.paths = self.window.view.locs_paths
        self.infos = self.window.view.infos
        # which channel to plot
        self.channel = self.window.view.get_channel("Mask image")
        self.cmap = self.window.display_settings_dlg.colormap.currentText()
        self.show()
        locs = self.locs[self.channel]
        info = self.infos[self.channel][0]
        self.x_max = info["Width"]
        self.y_max = info["Height"]
        self.update_plots()

    def generate_image(self):
        """ Histograms loaded localizations from a given channel. """

        locs = self.locs[self.channel]
        oversampling = (
            self.window.display_settings_dlg.pixelsize.value() 
            / self.disp_px_size.value()
        )
        viewport = ((0, 0), (self.y_max, self.x_max))
        _, H = render.render(
            locs, 
            oversampling=oversampling, 
            viewport=viewport, 
            blur_method=None,
        )
        self.H = H / H.max()

    def blur_image(self):
        """ Blurs localizations using a Gaussian filter. """

        H_blur = gaussian_filter(self.H, sigma=self.mask_blur.value())
        H_blur = H_blur / np.max(H_blur)
        self.H_blur = H_blur # image to be displayed in self.ax2

    def save_mask(self):
        """ Saves binary mask into .npy format. """

        # get name for saving mask
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save mask to", filter="*.npy"
        )
        if path:
            np.save(path, self.mask)

    def load_mask(self):
        """ Loads binary mask from .npy format. """

        # choose which file to load
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load mask", filter="*.npy"
        )
        if path:
            self.mask_loaded = True # will block changing of the mask
            self.mask = np.load(path)
            # update plots without drawing a new mask
            self.update_plots(new_mask=False)

    def mask_image(self):
        """ Calculates binary mask based on threshold. """

        if not self.mask_loaded:
            mask = np.zeros(self.H_blur.shape, dtype=np.int8)
            mask[self.H_blur > self.mask_thresh.value()] = 1
            self.mask = mask
            self.save_mask_button.setEnabled(True)

    def update_plots(self, new_mask=True):
        """ 
        Plots in all 4 axes.

        Parameters
        ----------
        new_mask : boolean (default=True)
            True if new mask is to be calculated
        """

        if self.mask_blur.value() == 0.00000:
            self.mask_blur.setValue(0.00001)

        if new_mask:
            if self.cached_oversampling:
                self.cached_oversampling = 0

            if self.cached_blur: 
                self.cached_blur = 0

            if self.cached_thresh:
                self.cached_thresh = 0

            if not self.cached_oversampling:
                self.generate_image()
                self.blur_image()
                self.mask_image()
                self.cached_oversampling = 1
                self.cached_blur = 1
                self.cached_thresh = 1

            if not self.cached_blur:
                self.blur_image()
                self.mask_image()
                self.cached_blur = 1
                self.cached_thresh = 1

            if not self.cached_thresh:
                self.mask_image()
                self.cached_thresh = 1

        self.ax1.imshow(self.H, cmap=self.cmap)
        self.ax1.set_title("Original")
        self.ax2.imshow(self.H_blur, cmap=self.cmap)
        self.ax2.set_title("Blurred")
        self.ax3.imshow(self.mask, cmap='Greys_r')
        self.ax3.set_title("Mask")
        self.ax4.imshow(np.zeros_like(self.H), cmap=self.cmap)
        self.ax4.set_title("Masked image")

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.grid(False)
            ax.axis('off')

        self.canvas.draw()

    def mask_locs(self):
        """ Masks localizations from a single or all channels. """

        self.index_locs = [] # locs in the mask
        self.index_locs_out = [] # locs outside the mask
        if self.save_all.isChecked(): # all channels
            for locs in self.locs:
                self._mask_locs(locs)
        else: # only the current channel
            locs = self.locs[self.channel]
            self._mask_locs(locs)

    def _mask_locs(self, locs):
        """ 
        Masks locs given a mask. 

        Parameters
        ----------
        locs : np.recarray
            Localizations to be masked
        """

        x_ind = (
            np.floor(locs["x"] / self.x_max * self.mask.shape[0])
        ).astype(int)
        y_ind = (
            np.floor(locs["y"] / self.y_max * self.mask.shape[1])
        ).astype(int)

        index = self.mask[y_ind, x_ind].astype(bool)
        locs_in = locs[index]
        locs_in.sort(kind="mergesort", order="frame")
        locs_out = locs[~index]
        locs_out.sort(kind="mergesort", order="frame")

        self.index_locs.append(locs_in) # locs in the mask
        self.index_locs_out.append(locs_out) # locs outside the mask

        if (
            (
                self.save_all.isChecked() 
                and len(self.index_locs) == self.channel + 1
            )
            or not self.save_all.isChecked()
        ): # update masked locs plot if the current channel is masked
            _, self.H_new = render.render(
                self.index_locs[-1],
                oversampling=(
                    self.window.display_settings_dlg.pixelsize.value()
                    / self.disp_px_size.value()
                ),
                viewport=((0, 0), (self.y_max, self.x_max)),
                blur_method=None,
            )

            self.ax4.imshow(self.H_new, cmap=self.cmap)
            self.ax4.grid(False)
            self.ax4.axis('off')
            self.save_button.setEnabled(True)
            self.canvas.draw()

    def save_locs(self):
        """ Saves masked localizations. """

        if self.save_all.isChecked(): # save all channels
            self.save_locs_multi()
        else:
            out_path = self.paths[self.channel].replace(
                ".hdf5", "_mask_in.hdf5"
            )
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Save localizations within mask", 
                out_path, 
                filter="*.hdf5",
            )
            if path:
                info = self.infos[self.channel] + [
                    {
                        "Generated by": "Picasso Render : Mask in ",
                        "Display pixel size [nm]": self.disp_px_size.value(),
                        "Blur": self.mask_blur.value(),
                        "Threshold": self.mask_thresh.value(),
                    }
                ]
                io.save_locs(path, self.index_locs[0], info)

            out_path = self.paths[self.channel].replace(
                ".hdf5", "_mask_out.hdf5"
            )
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save localizations outside of mask",
                out_path,
                filter="*.hdf5",
            )
            if path:
                info = self.infos[self.channel] + [
                    {
                        "Generated by": "Picasso Render : Mask out",
                        "Display pixel size [nm]": self.disp_px_size.value(),
                        "Blur": self.mask_blur.value(),
                        "Threshold": self.mask_thresh.value(),
                    }
                ]
                io.save_locs(path, self.index_locs_out[0], info)

    def save_locs_multi(self):
        """ Saves masked localizations for all loaded channels. """

        suffix_in, ok1 = QtWidgets.QInputDialog.getText(
            self, 
            "", 
            "Enter suffix for localizations inside the mask",
            QtWidgets.QLineEdit.Normal,
            "_mask_in",
        )
        if ok1:
            suffix_out, ok2 = QtWidgets.QInputDialog.getText(
                self,
                "",
                "Enter suffix for localizations outside the mask",
                QtWidgets.QLineEdit.Normal,
                "_mask_out",
            )
            if ok2:
                for channel in range(len(self.index_locs)):
                    out_path = self.paths[channel].replace(
                        ".hdf5", f"{suffix_in}.hdf5"
                    )
                    info = self.infos[channel] + [
                        {
                            "Generated by": "Picasso Render : Mask in",
                        "Display pixel size [nm]": self.disp_px_size.value(),
                        "Blur": self.mask_blur.value(),
                        "Threshold": self.mask_thresh.value(),
                        }
                    ]
                    io.save_locs(out_path, self.index_locs[channel], info)

                    out_path = self.paths[channel].replace(
                        ".hdf5", f"{suffix_out}.hdf5"
                    )
                    info = self.infos[channel] + [
                        {
                            "Generated by": "Picasso Render : Mask out",
                        "Display pixel size [nm]": self.disp_px_size.value(),
                        "Blur": self.mask_blur.value(),
                        "Threshold": self.mask_thresh.value(),
                        }
                    ]
                    io.save_locs(out_path, self.index_locs_out[channel], info)

class PickToolCircleSettings(QtWidgets.QWidget):
    """ A class contating information about circular pick. """

    def __init__(self, window, tools_settings_dialog):
        super().__init__()
        self.grid = QtWidgets.QGridLayout(self)
        self.window = window
        self.grid.addWidget(QtWidgets.QLabel("Diameter (cam. pixel):"), 0, 0)
        self.pick_diameter = QtWidgets.QDoubleSpinBox()
        self.pick_diameter.setRange(0.001, 999999)
        self.pick_diameter.setValue(1)
        self.pick_diameter.setSingleStep(0.1)
        self.pick_diameter.setDecimals(3)
        self.pick_diameter.setKeyboardTracking(False)
        self.pick_diameter.valueChanged.connect(
            tools_settings_dialog.on_pick_dimension_changed
        )
        self.grid.addWidget(self.pick_diameter, 0, 1)
        self.grid.addWidget(
            QtWidgets.QLabel("Pick similar +/- range (std)"), 1, 0
        )
        self.pick_similar_range = QtWidgets.QDoubleSpinBox()
        self.pick_similar_range.setRange(0, 100000)
        self.pick_similar_range.setValue(2)
        self.pick_similar_range.setSingleStep(0.1)
        self.pick_similar_range.setDecimals(2)
        self.grid.addWidget(self.pick_similar_range, 1, 1)


class PickToolRectangleSettings(QtWidgets.QWidget):
    """ A class containing information about rectangular pick. """

    def __init__(self, window, tools_settings_dialog):
        super().__init__()
        self.window = window
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.addWidget(QtWidgets.QLabel("Width (cam. pixel):"), 0, 0)
        self.pick_width = QtWidgets.QDoubleSpinBox()
        self.pick_width.setRange(0.001, 999999)
        self.pick_width.setValue(1)
        self.pick_width.setSingleStep(0.1)
        self.pick_width.setDecimals(3)
        self.pick_width.setKeyboardTracking(False)
        self.pick_width.valueChanged.connect(
            tools_settings_dialog.on_pick_dimension_changed
        )
        self.grid.addWidget(self.pick_width, 0, 1)
        self.grid.setRowStretch(1, 1)


class RESIDialog(QtWidgets.QDialog):
    """ RESI dialog.

    Allows for clustering multiple channels with user-defined
    clustering parameters using the SMLM clusterer; saves cluster 
    centers in a single .hdf5 file that contains an extra column with 
    resi channel ids and a metadata .yaml file.

    ...

    Attributes
    ----------
    apply_fa : QCheckBox
        If checked, apply basic frame analysis (just like in the case
        of SMLM clustering)
    locs : list of np.recarrays
        List of localization lists that are loaded in the main window
        when opening the RESI dialog.
    min_locs : list of QSpinBoxes
        List of widgets holding minimum number of localizations used in
        SMLM clusterer.
    n_dim : int
        Dimensionality of loaded localizations (2 or 3).
    n_channels : int
        Number of channels used in RESI analysis.
    paths : list of strings
        Paths to localization lists used for RESI analysis.
    radius_xy : list of QDoubleSpinBoxes
        List of widgets holding radius in x and y used in SMLM 
        clusterer.
    radius_z : list of QDoubleSpinBoxes
        List of widgets holding radius in z used in SMLM clusterer.
        Only applied when 3D data is loaded. Otherwise is not used.
    save_cluster_centers : QCheckBox
        If checked, saves cluster centers for each loaded localization
        list while performing RESI analysis.
    save_clustered_locs : QCheckBox
        If checked, saves clustered localizations for each loaded 
        localization list while performing RESI analysis.
    window : QMainWindow
        Instance of the main Picasso Render window.

    Methods
    -------
    on_same_params_clicked()
        Sets all clustering parameters to have the same value as in
        the first row.
    perform_resi()
        Performs RESI and saves RESI cluster centers. Saves clustered 
        localizations and cluster centers if requested. 
    """

    def __init__(self, window):
        super().__init__()
        self.setWindowTitle("RESI")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)

        self.window = window
        self.locs = window.view.locs
        self.n_channels = len(self.locs)
        self.paths = window.view.locs_paths
        self.ndim = 2
        if all([hasattr(_, "z") for _ in self.locs]):
            self.ndim = 3

        self.radius_xy = []
        self.radius_z = []
        self.min_locs = []

        ### layout ###
        vbox = QtWidgets.QVBoxLayout(self)

        ## clustering parameters - apply the same to all channels
        params_box = QtWidgets.QGroupBox("")
        vbox.addWidget(params_box)
        params_grid = QtWidgets.QGridLayout(params_box)

        same_params = QtWidgets.QPushButton(
            "Apply the same clustering parameters to all channels"
        )
        same_params.setAutoDefault(False)
        same_params.clicked.connect(self.on_same_params_clicked)
        params_grid.addWidget(same_params, 0, 0, 1, 4)

        ## clustering parameters - labels
        params_grid.addWidget(QtWidgets.QLabel("RESI channel"), 2, 0)
        if self.ndim == 2:
            params_grid.addWidget(
                QtWidgets.QLabel("Radius\n[cam. pixel]"), 2, 1
            )
            params_grid.addWidget(
                QtWidgets.QLabel("Min # localizations"), 2, 2, 1, 2
            )
        else:
            params_grid.addWidget(
                QtWidgets.QLabel("Radius xy\n[cam. pixel]"), 2, 1
            )
            params_grid.addWidget(
                QtWidgets.QLabel("Radius z\n[cam. pixel]"), 2, 2
            )
            params_grid.addWidget(
                QtWidgets.QLabel("Min # localizations"), 2, 3
            )

        ## clustering parameters - values
        for i in range(self.n_channels):
            channel_name = self.window.dataset_dialog.checks[i].text()
            count = params_grid.rowCount()

            r_xy = QtWidgets.QDoubleSpinBox()
            r_xy.setRange(0.0001, 1e3)
            r_xy.setDecimals(4)
            r_xy.setValue(0.1)
            r_xy.setSingleStep(0.01)
            self.radius_xy.append(r_xy)

            r_z = QtWidgets.QDoubleSpinBox()
            r_z.setRange(0.0001, 1e3)
            r_z.setDecimals(4)
            r_z.setValue(0.25)
            r_z.setSingleStep(0.01)
            self.radius_z.append(r_z)

            min_locs = QtWidgets.QSpinBox()
            min_locs.setRange(1, int(1e6))
            min_locs.setValue(10)
            min_locs.setSingleStep(1)
            self.min_locs.append(min_locs)

            params_grid.addWidget(QtWidgets.QLabel(channel_name), count, 0)
            params_grid.addWidget(r_xy, count, 1)
            if self.ndim == 3:
                params_grid.addWidget(r_z, count, 2)
                params_grid.addWidget(min_locs, count, 3)
            else:
                params_grid.addWidget(min_locs, count, 2, 1, 2)

        ## perform clustering
        # what to save
        self.save_clustered_locs = QtWidgets.QCheckBox(
            "Save clustered localizations\nof individual channels"
        )
        self.save_clustered_locs.setChecked(False)
        params_grid.addWidget(
            self.save_clustered_locs, params_grid.rowCount(), 0, 1, 2
        )
        # individual cluster centers
        self.save_cluster_centers = QtWidgets.QCheckBox(
            "Save cluster centers\nof individual channels"
        )
        self.save_cluster_centers.setChecked(False)
        params_grid.addWidget(
            self.save_cluster_centers, params_grid.rowCount()-1, 2, 1, 2
        )
        # apply basic frame analysis
        self.apply_fa = QtWidgets.QCheckBox(
            "Apply frame analysis\nto clustered localizations"
        )
        self.apply_fa.setChecked(True)
        params_grid.addWidget(self.apply_fa, params_grid.rowCount(), 0, 1, 2)

        ## perform resi button
        resi_button = QtWidgets.QPushButton("Perform RESI analysis")
        resi_button.clicked.connect(self.perform_resi)
        params_grid.addWidget(resi_button, params_grid.rowCount()-1, 2, 1, 2)

    def on_same_params_clicked(self):
        """ Sets all clustering parameters to have the same value as in
        the first row.
        """
        
        for r_xy, r_z, m in zip(self.radius_xy, self.radius_z, self.min_locs):
            r_xy.setValue(self.radius_xy[0].value())
            r_z.setValue(self.radius_z[0].value())
            m.setValue(self.min_locs[0].value())

    def perform_resi(self):
        """ Performs RESI analysis on loaded localizations, using 
        user-defined clustering parameters.
        """

        ### Sanity check if more than one channel is present
        if self.n_channels < 2:
            message = (
                "RESI relies on sequential imaging to assure sufficient"
                " sparsity of the binding sites. Thus, it requires at least"
                " two localization lists to be loaded.\n"
                "If you wish to extract cluster centers, please use\n"
                "Postprocess > Clustering > SMLM Clusterer"
            )
            QtWidgets.QMessageBox.information(self, "Warning", message)
            return
        
        ### Prepare data
        # extract clustering parameters
        r_xy = [_.value() for _ in self.radius_xy]
        r_z = [_.value() for _ in self.radius_z]
        min_locs = [_.value() for _ in self.min_locs]

        # get camera pixel size
        pixelsize = self.window.display_settings_dlg.pixelsize.value()

        # saving: path and info for the resi file, suffices for saving
        # clustered localizations and cluster centers if requested
        suffix_locs = None # suffix added to clustered locs
        suffix_centers = None # suffix added to cluster centers
        apply_fa = self.apply_fa.isChecked() # apply basic frame analysis?

        resi_path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self.window,
            "Save RESI cluster centers",
            self.paths[0].replace(".hdf5", "_resi.hdf5"),
            filter="*.hdf5",
        )
        info = self.window.view.infos[0]
        new_info = {
            "Paths to RESI channels": self.paths,
            "Clustering radius xy [cam. pixels] for each channel": r_xy,
            "Min. number of locs in a cluster for each channel": min_locs,
            "Basic frame analysis": apply_fa,
        }
        if self.ndim == 3:
            new_info[
                "Clustering radius z [cam. pixels] for each channel"
             ] = r_z
        resi_info = info + [new_info]

        if resi_path:
            ok1 = False
            if self.save_clustered_locs.isChecked():
                suffix_locs, ok1 = QtWidgets.QInputDialog.getText(
                    self,
                    "",
                    "Enter suffix for saving clustered localizations",
                    QtWidgets.QLineEdit.Normal,
                    "_clustered",
                )
            ok2 = False
            if self.save_cluster_centers.isChecked():
                suffix_centers, ok2 = QtWidgets.QInputDialog.getText(
                    self,
                    "",
                    "Enter suffix for saving cluster centers",
                    QtWidgets.QLineEdit.Normal,
                    "_cluster_centers",
                )

            ### Perform RESI            
            progress = lib.ProgressDialog(
                "Performing RESI analysis...", 0, self.n_channels, self.window
            )
            progress.set_value(0)
            progress.show()

            resi_channels = [] # holds each channel's cluster centers
            for i, locs in enumerate(self.locs):
                # cluster each channel using SMLM clusterer
                if self.ndim == 3:
                    params = [r_xy[i], r_z[i], min_locs[i], 0, apply_fa, 0]
                else:
                    params = [r_xy[i], min_locs[i], 0, apply_fa, 0]

                clustered_locs = clusterer.cluster(locs, params, pixelsize)

                # save clustered localizations if requested
                if ok1:
                    new_info = {
                        "Clustering radius xy [cam. pixels]": r_xy[i],
                        "Min. number of locs": min_locs[i],
                        "Basic frame analysis": apply_fa,
                    }
                    if self.ndim == 3:
                        new_info["Clustering radius z [cam. pixels]"] = r_z[i]
                    io.save_locs(
                        self.paths[i].replace(
                            ".hdf5", f"{suffix_locs}.hdf5"
                        ),
                        clustered_locs,
                        self.window.view.infos[i] + [new_info],
                    )

                # extract cluster centers for each channel
                centers = clusterer.find_cluster_centers(
                    clustered_locs, pixelsize
                )
                # save cluster centers if requested
                if ok2:
                    new_info = {
                        "Clustering radius xy [cam. pixels]": r_xy[i],
                        "Min. number of locs": min_locs[i],
                        "Basic frame analysis": apply_fa,
                    }
                    if self.ndim == 3:
                        new_info["Clustering radius z [cam. pixels]"] = r_z[i]
                    io.save_locs(
                        self.paths[i].replace(
                            ".hdf5", f"{suffix_centers}.hdf5"
                        ),
                        centers,
                        self.window.view.infos[i] + [new_info],
                    )
                # append resi channel id 
                centers = lib.append_to_rec(
                    centers, 
                    i*np.ones(len(centers), dtype=np.int8),
                    "resi_channel_id",
                )
                resi_channels.append(centers)
                progress.set_value(i)
            progress.close()

            # combine resi cluster centers from all channels
            all_resi = stack_arrays(
                resi_channels, 
                asrecarray=True, 
                usemask=False, 
                autoconvert=True,
            )
            # change the group name in all_resi
            all_resi = all_resi.astype([
                ("cluster_id", d[1]) if d[0]=="group" else d
                for d in all_resi.dtype.descr
            ])
            all_resi = lib.remove_from_rec(all_resi, "group")
            # sort like all Picasso localization lists
            all_resi.sort(kind="mergesort", order="frame")

            # save resi cluster centers
            io.save_locs(resi_path, all_resi, resi_info)


class ToolsSettingsDialog(QtWidgets.QDialog):
    """
    A dialog class to customize picks - vary shape and size, annotate,
    change std for picking similar.

    ...

    Attributes
    ----------
    pick_annotation : QCheckBox
        tick to display picks' indeces 
    pick_diameter : QDoubleSpinBox
        contains the diameter of circular picks (pixels)
    pick_shape : QComboBox
        contains the str with the shape of picks (circle or rectangle)
    pick_width : QDoubleSpinBox
        contains the width of rectangular picks (pixels)
    point_picks : QCheckBox
        tick to display circular picks as 3-pixels-wide points

    Methods
    -------
    on_pick_dimension_changed(*args)
        Resets index_blocks in View and updates the scene
    update_scene_with_cache(*args)
        Quick (cached) update of the current view when picks change
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Tools Settings")
        self.setModal(False)
        self.vbox = QtWidgets.QVBoxLayout(self)

        self.pick_groupbox = QtWidgets.QGroupBox("Pick")
        self.vbox.addWidget(self.pick_groupbox)
        pick_grid = QtWidgets.QGridLayout(self.pick_groupbox)

        pick_grid.addWidget(QtWidgets.QLabel("Shape:"), 1, 0)
        self.pick_shape = QtWidgets.QComboBox()
        self.pick_shape.addItems(["Circle", "Rectangle"])
        pick_grid.addWidget(self.pick_shape, 1, 1)
        pick_stack = QtWidgets.QStackedWidget()
        pick_grid.addWidget(pick_stack, 2, 0, 1, 2)
        self.pick_shape.currentIndexChanged.connect(pick_stack.setCurrentIndex)

        # Circle
        self.pick_circle_settings = PickToolCircleSettings(window, self)
        pick_stack.addWidget(self.pick_circle_settings)
        self.pick_similar_range = self.pick_circle_settings.pick_similar_range
        self.pick_diameter = self.pick_circle_settings.pick_diameter

        # Rectangle
        self.pick_rectangle_settings = PickToolRectangleSettings(window, self)
        pick_stack.addWidget(self.pick_rectangle_settings)
        self.pick_width = self.pick_rectangle_settings.pick_width

        self.pick_annotation = QtWidgets.QCheckBox("Annotate picks")
        self.pick_annotation.stateChanged.connect(self.update_scene_with_cache)
        pick_grid.addWidget(self.pick_annotation, 3, 0)

        self.point_picks = QtWidgets.QCheckBox(
            "Display circular picks as points"
        )
        self.point_picks.stateChanged.connect(self.update_scene_with_cache)
        pick_grid.addWidget(self.point_picks, 4, 0)

    def on_pick_dimension_changed(self, *args):
        """ Resets index_blokcs in View and updates the scene. """

        self.window.view.index_blocks = [
            None for _ in self.window.view.index_blocks
        ]
        self.update_scene_with_cache()

    def update_scene_with_cache(self, *args):
        """
        Quick (cached) update of the current view when picks change.
        """

        self.window.view.update_scene(use_cache=True)


class DisplaySettingsDialog(QtWidgets.QDialog):
    """
    A class to change display settings, e.g.: zoom, display pixel size, 
    contrast and blur.

    ...

    Attributes
    ----------
    blur_buttongroup : QButtonGroup
        contains available localization blur methods
    colormap : QComboBox
        contains strings with available colormaps (single channel only)
    color_step : QSpinBox
        defines how many colors are to be rendered
    disp_px_size : QDoubleSpinBox
        contains the size of super-resolution pixels in nm
    dynamic_disp_px : QCheckBox
        tick to automatically adjust to current window size when zooming.
    maximum : QDoubleSpinBox
        defines at which number of localizations per super-resolution
        pixel the maximum color of the colormap should be applied
    maximum_render : QDoubleSpinBox
        contains the maximum value of the parameter to be rendered
    min_blur_width : QDoubleSpinBox
        contains the minimum blur for each localization (pixels)
    minimap : QCheckBox
        tick to display minimap showing current FOV
    minimum : QDoubleSpinBox
        defines at which number of localizations per super-resolution
        pixel the minimum color of the colormap should be applied
    minimum_render : QDoubleSpinBox
        contains the minimum value of the parameter to be rendered
    parameter : QComboBox
        defines what property should be rendered, e.g.: z, photons
    pixelsize : QDoubleSpinBox
        contains the camera pixel size (nm)
    render_check : QCheckBox
        tick to activate parameter rendering
    scalebar : QSpinBox
        contains the scale bar's length (nm)
    scalebar_groupbox : QGroupBox
        group with options for customizing scale bar, tick to display
    scalebar_text : QCheckBox
        tick to display scale bar's length (nm)
    show_legend : QPushButton
        click to display parameter rendering's legend
    _silent_disp_px_update : boolean
        True if update display pixel size in background
    zoom : QDoubleSpinBox
        contains zoom's magnitude

    Methods
    -------
    on_cmap_changed()
        Loads custom colormap if requested
    on_disp_px_changed(value)
        Sets new display pixel size, updates contrast and updates scene 
        in the main window
    on_zoom_changed(value)
        Zooms the image in the main window
    render_scene(*args, **kwargs)
        Updates scene in the main window
    set_dynamic_disp_px(state)
        Updates scene if dynamic display pixel size is checked
    set_disp_px_silently(disp_px_size)
        Changes the value of display pixel size in background
    set_zoom_silently(zoom)
        Changes the value of zoom in the background
    silent_maximum_update(value)
        Changes the value of self.maximum in the background
    silent_minimum_update(value)
        Changes the value of self.minimum in the background 
    update_scene(*args, **kwargs)
        Updates scene with cache
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Display Settings")
        self.resize(200, 0)
        self.setModal(False)
        vbox = QtWidgets.QVBoxLayout(self)

        # General
        general_groupbox = QtWidgets.QGroupBox("General")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)
        general_grid.addWidget(QtWidgets.QLabel("Zoom:"), 0, 0)
        self.zoom = QtWidgets.QDoubleSpinBox()
        self.zoom.setKeyboardTracking(False)
        self.zoom.setRange(10 ** (-self.zoom.decimals()), 1e6)
        self.zoom.valueChanged.connect(self.on_zoom_changed)
        general_grid.addWidget(self.zoom, 0, 1)
        general_grid.addWidget(
            QtWidgets.QLabel("Display pixel size [nm]:"), 1, 0
        )
        self._disp_px_size = 130 / DEFAULT_OVERSAMPLING 
        self.disp_px_size = QtWidgets.QDoubleSpinBox()
        self.disp_px_size.setRange(0.00001, 100000)
        self.disp_px_size.setSingleStep(1)
        self.disp_px_size.setDecimals(5)
        self.disp_px_size.setValue(self._disp_px_size)
        self.disp_px_size.setKeyboardTracking(False)
        self.disp_px_size.valueChanged.connect(self.on_disp_px_changed)
        general_grid.addWidget(self.disp_px_size, 1, 1)
        self.dynamic_disp_px = QtWidgets.QCheckBox("dynamic")
        self.dynamic_disp_px.setChecked(True)
        self.dynamic_disp_px.toggled.connect(
            self.set_dynamic_disp_px
        )
        general_grid.addWidget(self.dynamic_disp_px, 2, 1)
        self.minimap = QtWidgets.QCheckBox("show minimap")
        general_grid.addWidget(self.minimap, 3, 1)
        self.minimap.stateChanged.connect(self.update_scene)

        # Contrast
        contrast_groupbox = QtWidgets.QGroupBox("Contrast")
        vbox.addWidget(contrast_groupbox)
        contrast_grid = QtWidgets.QGridLayout(contrast_groupbox)
        minimum_label = QtWidgets.QLabel("Min. Density:")
        contrast_grid.addWidget(minimum_label, 0, 0)
        self.minimum = QtWidgets.QDoubleSpinBox()
        self.minimum.setRange(0, 999999)
        self.minimum.setSingleStep(5)
        self.minimum.setValue(0)
        self.minimum.setDecimals(6)
        self.minimum.setKeyboardTracking(False)
        self.minimum.valueChanged.connect(self.update_scene)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtWidgets.QLabel("Max. Density:")
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = QtWidgets.QDoubleSpinBox()
        self.maximum.setRange(0, 999999)
        self.maximum.setSingleStep(5)
        self.maximum.setValue(100)
        self.maximum.setDecimals(6)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.update_scene)
        contrast_grid.addWidget(self.maximum, 1, 1)
        contrast_grid.addWidget(QtWidgets.QLabel("Colormap:"), 2, 0)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(plt.colormaps())
        self.colormap.addItem("Custom")
        contrast_grid.addWidget(self.colormap, 2, 1)
        self.colormap.currentIndexChanged.connect(
            self.on_cmap_changed
        )

        # Blur
        blur_groupbox = QtWidgets.QGroupBox("Blur")
        blur_grid = QtWidgets.QGridLayout(blur_groupbox)
        self.blur_buttongroup = QtWidgets.QButtonGroup()
        points_button = QtWidgets.QRadioButton("None")
        self.blur_buttongroup.addButton(points_button)
        smooth_button = QtWidgets.QRadioButton("One-Pixel-Blur")
        self.blur_buttongroup.addButton(smooth_button)
        convolve_button = QtWidgets.QRadioButton("Global Localization Precision")
        self.blur_buttongroup.addButton(convolve_button)
        gaussian_button = QtWidgets.QRadioButton(
            "Individual Localization Precision"
        )
        self.blur_buttongroup.addButton(gaussian_button)
        gaussian_iso_button = QtWidgets.QRadioButton(
            "Individual Localization Precision, iso"
        )
        self.blur_buttongroup.addButton(gaussian_iso_button)

        blur_grid.addWidget(points_button, 0, 0, 1, 2)
        blur_grid.addWidget(smooth_button, 1, 0, 1, 2)
        blur_grid.addWidget(convolve_button, 2, 0, 1, 2)
        blur_grid.addWidget(gaussian_button, 3, 0, 1, 2)
        blur_grid.addWidget(gaussian_iso_button, 4, 0, 1, 2)
        convolve_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.render_scene)
        blur_grid.addWidget(
            QtWidgets.QLabel("Min. Blur (cam. pixel):"), 5, 0, 1, 1
        )
        self.min_blur_width = QtWidgets.QDoubleSpinBox()
        self.min_blur_width.setRange(0, 999999)
        self.min_blur_width.setSingleStep(0.01)
        self.min_blur_width.setValue(0)
        self.min_blur_width.setDecimals(3)
        self.min_blur_width.setKeyboardTracking(False)
        self.min_blur_width.valueChanged.connect(self.render_scene)
        blur_grid.addWidget(self.min_blur_width, 5, 1, 1, 1)

        vbox.addWidget(blur_groupbox)
        self.blur_methods = {
            points_button: None,
            smooth_button: "smooth",
            convolve_button: "convolve",
            gaussian_button: "gaussian",
            gaussian_iso_button: "gaussian_iso",
        }

        # Camera_parameters
        camera_groupbox = QtWidgets.QGroupBox("Camera")
        self.camera_grid = QtWidgets.QGridLayout(camera_groupbox)
        self.camera_grid.addWidget(QtWidgets.QLabel("Pixel Size (nm):"), 0, 0)
        self.pixelsize = QtWidgets.QDoubleSpinBox()
        self.pixelsize.setRange(1, 100000)
        self.pixelsize.setValue(130)
        self.pixelsize.setKeyboardTracking(False)
        self.pixelsize.valueChanged.connect(self.update_scene)
        self.camera_grid.addWidget(self.pixelsize, 0, 1)
        vbox.addWidget(camera_groupbox)

        # Scalebar
        self.scalebar_groupbox = QtWidgets.QGroupBox("Scale Bar")
        self.scalebar_groupbox.setCheckable(True)
        self.scalebar_groupbox.setChecked(False)
        self.scalebar_groupbox.toggled.connect(self.update_scene)
        vbox.addWidget(self.scalebar_groupbox)
        scalebar_grid = QtWidgets.QGridLayout(self.scalebar_groupbox)
        scalebar_grid.addWidget(
            QtWidgets.QLabel("Scale Bar Length (nm):"), 0, 0
        )
        self.scalebar = QtWidgets.QSpinBox()
        self.scalebar.setRange(1, 100000)
        self.scalebar.setValue(500)
        self.scalebar.setKeyboardTracking(False)
        self.scalebar.valueChanged.connect(self.update_scene)
        scalebar_grid.addWidget(self.scalebar, 0, 1)
        self.scalebar_text = QtWidgets.QCheckBox("Print scale bar length")
        self.scalebar_text.stateChanged.connect(self.update_scene)
        scalebar_grid.addWidget(self.scalebar_text, 1, 0)
        self._silent_disp_px_update = False

        # Render
        self.render_groupbox = QtWidgets.QGroupBox("Render properties")

        vbox.addWidget(self.render_groupbox)
        render_grid = QtWidgets.QGridLayout(self.render_groupbox)
        render_grid.addWidget(QtWidgets.QLabel("Parameter:"), 0, 0)
        self.parameter = QtWidgets.QComboBox()
        render_grid.addWidget(self.parameter, 0, 1)
        self.parameter.activated.connect(self.window.view.set_property)

        minimum_label_render = QtWidgets.QLabel("Min.:")
        render_grid.addWidget(minimum_label_render, 1, 0)
        self.minimum_render = QtWidgets.QDoubleSpinBox()
        self.minimum_render.setRange(-999999, 999999)
        self.minimum_render.setSingleStep(5)
        self.minimum_render.setValue(0)
        self.minimum_render.setDecimals(2)
        self.minimum_render.setKeyboardTracking(False)
        self.minimum_render.setEnabled(False)
        self.minimum_render.valueChanged.connect(
            self.window.view.activate_render_property
        )
        render_grid.addWidget(self.minimum_render, 1, 1)
        maximum_label_render = QtWidgets.QLabel("Max.:")
        render_grid.addWidget(maximum_label_render, 2, 0)
        self.maximum_render = QtWidgets.QDoubleSpinBox()
        self.maximum_render.setRange(-999999, 999999)
        self.maximum_render.setSingleStep(5)
        self.maximum_render.setValue(100)
        self.maximum_render.setDecimals(2)
        self.maximum_render.setKeyboardTracking(False)
        self.maximum_render.setEnabled(False)
        self.maximum_render.valueChanged.connect(
            self.window.view.activate_render_property
        )
        render_grid.addWidget(self.maximum_render, 2, 1)
        color_step_label = QtWidgets.QLabel("Colors:")
        render_grid.addWidget(color_step_label, 3, 0)
        self.color_step = QtWidgets.QSpinBox()
        self.color_step.setRange(1, 256)
        self.color_step.setSingleStep(16)
        self.color_step.setValue(32)
        self.color_step.setKeyboardTracking(False)
        self.color_step.setEnabled(False)
        self.color_step.valueChanged.connect(
            self.window.view.activate_render_property
        )
        render_grid.addWidget(self.color_step, 3, 1)

        self.render_check = QtWidgets.QCheckBox("Render")
        self.render_check.stateChanged.connect(
            self.window.view.activate_render_property
        )
        self.render_check.setEnabled(False)
        render_grid.addWidget(self.render_check, 4, 0)

        self.show_legend = QtWidgets.QPushButton("Show legend")
        render_grid.addWidget(self.show_legend, 4, 1)
        self.show_legend.setEnabled(False)
        self.show_legend.setAutoDefault(False)
        self.show_legend.clicked.connect(self.window.view.show_legend)

    def on_cmap_changed(self):
        """ Loads custom colormap if requested. """

        if self.colormap.currentText() == "Custom":
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load custom colormap", filter="*.npy"
            )
            if path:
                cmap = np.load(path)
                if cmap.shape != (256, 4):
                    raise ValueError(
                        "Colormap must be of shape (256, 4)\n"
                        f"The loaded colormap has shape {cmap.shape}"
                    )
                    self.colormap.setCurrentText("magma")
                elif not np.all((cmap >= 0) & (cmap <= 1)):
                    raise ValueError(
                        "All elements of the colormap must be between\n"
                        "0 and 1"
                    )
                    self.colormap.setCurrentText("magma")
                else:
                    self.window.view.custom_cmap = cmap
            else:
                self.colormap.setCurrentText("magma")
        self.update_scene()

    def on_disp_px_changed(self, value):
        """
        Sets new display pixel size, updates contrast and updates scene
        in the main window.
        """

        contrast_factor = (value / self._disp_px_size) ** 2
        self._disp_px_size = value
        self.silent_minimum_update(contrast_factor * self.minimum.value())
        self.silent_maximum_update(contrast_factor * self.maximum.value())
        if not self._silent_disp_px_update:
            self.dynamic_disp_px.setChecked(False)
            self.window.view.update_scene()

    def on_zoom_changed(self, value):
        """ Zooms the image in the main window. """

        self.window.view.set_zoom(value)

    def set_disp_px_silently(self, disp_px_size):
        """ Changes the value of self.disp_px_size in the background. """

        self._silent_disp_px_update = True
        self.disp_px_size.setValue(disp_px_size)
        self._silent_disp_px_update = False

    def set_zoom_silently(self, zoom):
        """ Changes the value of zoom in the background. """

        self.zoom.blockSignals(True)
        self.zoom.setValue(zoom)
        self.zoom.blockSignals(False)

    def silent_minimum_update(self, value):
        """ Changes the value of self.minimum in the background. """

        self.minimum.blockSignals(True)
        self.minimum.setValue(value)
        self.minimum.blockSignals(False)

    def silent_maximum_update(self, value):
        """ Changes the value of self.maximum in the background. """

        self.maximum.blockSignals(True)
        self.maximum.setValue(value)
        self.maximum.blockSignals(False)

    def render_scene(self, *args, **kwargs):
        """ Updates scene in the main window. """

        self.window.view.update_scene()

    def set_dynamic_disp_px(self, state):
        """ Updates scene if dynamic display pixel size is checked. """

        if state:
            self.window.view.update_scene()

    def update_scene(self, *args, **kwargs):
        """ Updates scene with cache. """

        self.window.view.update_scene(use_cache=True)


class FastRenderDialog(QtWidgets.QDialog):
    """
    A class to randomly sample a given percentage of locs to increase
    the speed of rendering.

    ...

    Attributes
    ----------
    channel : QComboBox
        contains the channel where fast rendering is to be applied
    fraction : QSpinBox
        contains the percentage of locs to be sampled
    fractions : list
        contains the percetanges for all channels of locs to be sampled
    sample_button : QPushButton
        click to sample locs according to the percetanges specified by
        self.fractions
    window : QMainWindow
        instance of the main window
    
    Methods
    -------
    on_channel_changed()
        Retrieves value in self.fraction to the last chosen one
    on_file_added()
        Adds new item in self.channel
    on_file_closed(idx)
        Removes item in self.channel
    on_fraction_changed()
        Updates self.fractions 
    sample_locs()
        Draws a fraction of locs specified by self.fractions
    """

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setWindowTitle("Fast Render")
        self.setWindowIcon(self.window.icon)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.fractions = [100]

        # info explaining what is this dialog
        self.layout.addWidget(QtWidgets.QLabel(
            (
                "Change percentage of locs displayed in each\n"
                "channel to increase the speed of rendering.\n\n"
                "NOTE: sampling locs may lead to unexpected behaviour\n"
                "when using some of Picasso : Render functions.\n"
                "Please set the percentage below to 100 to avoid\n"
                "such situations."
            )
        ), 0, 0, 1, 2)

        # choose channel
        self.layout.addWidget(QtWidgets.QLabel("Channel: "), 1, 0)
        self.channel = QtWidgets.QComboBox(self)
        self.channel.setEditable(False)
        self.channel.addItem("All channels")
        self.channel.activated.connect(self.on_channel_changed)
        self.layout.addWidget(self.channel, 1, 1)

        # choose percentage
        self.layout.addWidget(
            QtWidgets.QLabel(
                "Percentage of localizations\nto be displayed"
            ), 2, 0
        )
        self.fraction = QtWidgets.QSpinBox(self)
        self.fraction.setSingleStep(1)
        self.fraction.setMinimum(1)
        self.fraction.setMaximum(100)
        self.fraction.setValue(100)
        self.fraction.valueChanged.connect(self.on_fraction_changed)
        self.layout.addWidget(self.fraction, 2, 1)

        # randomly draw localizations in each channel
        self.sample_button = QtWidgets.QPushButton(
            "Randomly sample\nlocalizations"
        )
        self.sample_button.clicked.connect(self.sample_locs)
        self.layout.addWidget(self.sample_button, 3, 1)

    def on_channel_changed(self):
        """ 
        Retrieves value in self.fraction to the last chosen one. 
        """

        idx = self.channel.currentIndex()
        self.fraction.blockSignals(True)
        self.fraction.setValue(self.fractions[idx])
        self.fraction.blockSignals(False)

    def on_file_added(self):
        """ Adds new item in self.channel. """

        self.channel.addItem(self.window.dataset_dialog.checks[-1].text())
        self.fractions.append(100)

    def on_file_closed(self, idx):
        """ Removes item from self.channel. """
        
        self.channel.removeItem(idx+1)
        del self.fractions[idx+1]

    def on_fraction_changed(self):
        """ Updates self.fractions. """

        idx = self.channel.currentIndex()
        self.fractions[idx] = self.fraction.value()

    def sample_locs(self):
        """ Draws a fraction of locs specified by self.fractions. """
        
        idx = self.channel.currentIndex()
        if idx == 0: # all channels share the same fraction
            for i in range(len(self.window.view.locs_paths)):
                n_locs = len(self.window.view.all_locs[i])
                rand_idx = np.random.choice(
                    n_locs, 
                    size=int(n_locs * self.fractions[0] / 100),
                    replace=False,
                ) # random indeces to extract locs
                self.window.view.locs[i] = (
                    self.window.view.all_locs[i][rand_idx]
                ) # assign new localizations to be displayed
        else: # each channel individually
            for i in range(len(self.window.view.locs_paths)):
                n_locs = len(self.window.view.all_locs[i])
                rand_idx = np.random.choice(
                    n_locs,
                    size=int(n_locs * self.fractions[i+1] / 100),
                    replace=False,
                ) # random indeces to extract locs
                self.window.view.locs[i] = (
                    self.window.view.all_locs[i][rand_idx]
                ) # assign new localizations to be displayed
        # update view.group_color if needed:
        if (len(self.fractions) == 2 and
            hasattr(self.window.view.locs[0], "group")
        ):
            self.window.view.group_color = (
                self.window.view.get_group_color(
                    self.window.view.locs[0]
                )
            )
        self.index_blocks = [None] * len(self.window.view.locs)
        self.window.view.update_scene()


class SlicerDialog(QtWidgets.QDialog):
    """
    A class to customize slicing 3D data in z axis.

    ...

    Attributes
    ----------
    bins : np.array
        contatins bins used in plotting the histogram
    canvas : FigureCanvas
        contains the histogram of number of locs in slices
    colors : list
        contains rgb channels for each localization channel
    export_button : QPushButton
        click to export slices into .tif files
    full_check : QCheckBox
        tick to save the whole FOV, untick to save only the current
        viewport
    patches : list
        contains plt.artists used in creating histograms
    pick_slice : QDoubleSpinBox
        contains slice thickness (nm)
    separate_check : QCheckBox
        tick to save channels separately when exporting slice
    sl : QSlider
        points to the slice to be displayed
    slicer_cache : dict
        contains QPixmaps that have been drawn for each slice
    slicermax : float
        maximum value of self.sl
    slicermin : float
        minimum value of self.sl
    slicerposition : float
        current position of self.sl
    slicer_radio_button : QCheckBox
        tick to slice locs
    window : QMainWindow
        instance of the main window
    zcoord : list
        z coordinates of each channel of localization (nm);
        added when loading each channel (see View.add)

    Methods
    -------
    calculate_histogram()
        Calculates and histograms z coordintes of each channel
    export_stack()
        Saves all slices as .tif files
    initialize()
        Called when the dialog is open, calculates the histograms and 
        shows the dialog
    on_pick_slice_changed()
        Modifies histograms when slice thickness changes
    on_slice_position_changed(position)
        Changes some properties and updates scene in the main window
    toggle_slicer()
        Updates scene in the main window when slicing is called
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("3D Slicer")
        self.setModal(False)
        self.setMinimumSize(550, 690) # to display the histogram
        vbox = QtWidgets.QVBoxLayout(self)
        slicer_groupbox = QtWidgets.QGroupBox("Slicer Settings")

        vbox.addWidget(slicer_groupbox)
        slicer_grid = QtWidgets.QGridLayout(slicer_groupbox)
        slicer_grid.addWidget(
            QtWidgets.QLabel("Slice Thickness [nm]:"), 0, 0
        )
        self.pick_slice = QtWidgets.QDoubleSpinBox()
        self.pick_slice.setRange(0.01, 99999)
        self.pick_slice.setValue(50)
        self.pick_slice.setSingleStep(1)
        self.pick_slice.setDecimals(2)
        self.pick_slice.setKeyboardTracking(False)
        self.pick_slice.valueChanged.connect(self.on_pick_slice_changed)
        slicer_grid.addWidget(self.pick_slice, 0, 1)

        self.sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(50)
        self.sl.setValue(25)
        self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sl.setTickInterval(1)
        self.sl.valueChanged.connect(self.on_slice_position_changed)
        slicer_grid.addWidget(self.sl, 1, 0, 1, 2)

        self.figure, self.ax = plt.subplots(1, figsize=(3, 3))
        self.canvas = FigureCanvas(self.figure)
        slicer_grid.addWidget(self.canvas, 2, 0, 1, 2)

        self.slicer_radio_button = QtWidgets.QCheckBox("Slice Dataset")
        self.slicer_radio_button.stateChanged.connect(self.toggle_slicer)
        slicer_grid.addWidget(self.slicer_radio_button, 3, 0)

        self.separate_check = QtWidgets.QCheckBox("Export channels separate")
        slicer_grid.addWidget(self.separate_check, 4, 0)
        self.full_check = QtWidgets.QCheckBox("Export full image")
        slicer_grid.addWidget(self.full_check, 5, 0)
        self.export_button = QtWidgets.QPushButton("Export Slices")
        self.export_button.setAutoDefault(False)
        self.export_button.clicked.connect(self.export_stack)
        slicer_grid.addWidget(self.export_button, 6, 0)

        self.zcoord = []

    def initialize(self):
        """ 
        Called when the dialog is open, calculates the histograms and 
        shows the dialog.
        """

        self.calculate_histogram()
        self.show()

    def calculate_histogram(self):
        """ Calculates and histograms z coordintes of each channel. """

        # slice thickness
        slice = self.pick_slice.value()
        # ax = self.figure.add_subplot(111)

        # # clear the plot
        # plt.cla()
        self.ax.clear()
        n_channels = len(self.zcoord)

        # get colors for each channel (from dataset dialog)
        colors = [
            _.palette().color(QtGui.QPalette.Window)
            for _ in self.window.dataset_dialog.colordisp_all
        ]
        self.colors = [
            [_.red() / 255, _.green() / 255, _.blue() / 255] for _ in colors
        ]

        # get bins, starting with minimum z and ending with max z
        self.bins = np.arange(
            np.amin(np.hstack(self.zcoord)),
            np.amax(np.hstack(self.zcoord)),
            slice,
        )

        # plot histograms
        self.patches = []
        for i in range(len(self.zcoord)):
            _, _, patches = self.ax.hist(
                self.zcoord[i],
                self.bins,
                density=True,
                facecolor=self.colors[i],
                alpha=0.5,
            )
            self.patches.append(patches)

        self.ax.set_xlabel("z-coordinate [nm]")
        self.ax.set_ylabel("Rel. frequency")
        self.ax.set_title(r"$\mathrm{Histogram\ of\ Z:}$")
        self.canvas.draw()
        self.sl.setMaximum(int(len(self.bins)) - 2)
        self.sl.setValue(int(len(self.bins) / 2))

        # reset cache
        self.slicer_cache = {}

    def on_pick_slice_changed(self):
        """ Modifies histograms when slice thickness changes. """

        # reset cache
        self.slicer_cache = {}
        if len(self.bins) < 3:  # in case there should be only 1 bin
            self.calculate_histogram()
        else:
            self.calculate_histogram()
            self.sl.setValue(int(len(self.bins) / 2))
            # self.on_slice_position_changed(self.sl.value())

    def toggle_slicer(self):
        """ Updates scene in the main window slicing is called. """

        self.window.view.update_scene()

    def on_slice_position_changed(self, position):
        """
        Changes some properties and updates scene in the main window.
        """

        for i in range(len(self.zcoord)):
            for patch in self.patches[i]:
                patch.set_facecolor(self.colors[i])
            self.patches[i][position].set_facecolor("black")

        self.slicerposition = position
        self.canvas.draw()
        self.slicermin = self.bins[position]
        self.slicermax = self.bins[position + 1]
        self.window.view.update_scene_slicer()

    def export_stack(self):
        """ Saves all slices as .tif files. """

        # get filename for saving
        try:
            base, ext = os.path.splitext(self.window.view.locs_paths[0])
        except AttributeError:
            return
        out_path = base + ".tif"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save z slices", out_path, filter="*.tif"
        )
        if path:
            base, ext = os.path.splitext(path)
            if self.separate_check.isChecked(): # each channel individually
                # Uncheck all
                for checks in self.window.dataset_dialog.checks:
                    checks.setChecked(False)

                for j in range(len(self.window.view.locs)):
                    # load a single channel
                    self.window.dataset_dialog.checks[j].setChecked(True)
                    progress = lib.ProgressDialog(
                        "Exporting slices..", 0, self.sl.maximum(), self
                    )
                    progress.set_value(0)
                    progress.show()

                    # save each channel one by one
                    for i in range(self.sl.maximum() + 1):
                        self.sl.setValue(i)
                        out_path = (
                            base
                            + "_Z"
                            + "{num:03d}".format(num=i)
                            + "_CH"
                            + "{num:03d}".format(num=j+1)
                            + ".tif"
                        )
                        if self.full_check.isChecked(): # full FOV
                            movie_height, movie_width = (
                                self.window.view.movie_size()
                            )
                            viewport = [(0, 0), (movie_height, movie_width)]
                            qimage = self.window.view.render_scene(
                                cache=False, viewport=viewport
                            )
                            gray = qimage.convertToFormat(
                                QtGui.QImage.Format_RGB16
                            )
                        else: # current FOV
                            gray = self.window.view.qimage.convertToFormat(
                                QtGui.QImage.Format_RGB16
                            )
                        gray.save(out_path)
                        progress.set_value(i)
                    progress.close()
                    self.window.dataset_dialog.checks[j].setChecked(False)
                for checks in self.window.dataset_dialog.checks:
                    checks.setChecked(True)
            else: # all channels at once
                progress = lib.ProgressDialog(
                    "Exporting slices..", 0, self.sl.maximum(), self
                )
                progress.set_value(0)
                progress.show()

                for i in range(self.sl.maximum() + 1):
                    self.sl.setValue(i)
                    out_path = (
                        base
                        + "_Z"
                        + "{num:03d}".format(num=i)
                        + "_CH001"
                        + ".tif"
                    )
                    if self.full_check.isChecked(): # full FOV
                        movie_height, movie_width = (
                            self.window.view.movie_size()
                        )
                        viewport = [(0, 0), (movie_height, movie_width)]
                        qimage = self.window.view.render_scene(
                            cache=False, viewport=viewport
                        )
                        qimage.save(out_path)
                    else: # current FOV
                        self.window.view.qimage.save(out_path)
                    progress.set_value(i)
                progress.close()


class View(QtWidgets.QLabel):
    """
    A class to display super-resolution datasets.

    ...

    Attributes
    ----------
    all_locs : list
        contains a np.recarray with localizations for each channel; 
        important for fast rendering
    currentdrift : list
        contains the most up-to-date drift for each channel
    custom_cmap : np.array
        custom colormap loaded from .npy, see DisplaySettingsDialog
    _drift : list
        contains np.recarrays with drift info for each channel, None if 
        no drift found/calculated
    _driftfiles : list
        contains paths to drift .txt files for each channel
    group_color : np.array
        important for single channel data with group info (picked or
        clustered locs); contains an integer index for each loc 
        defining its color
    image : np.array
        Unprocessed image of rendered localizations
    index_blocks : list
        contains tuples with info about indexed locs for each channel, 
        None if not calculated yet
    infos : list
        contains a dictionary with metadata for each channel
    locs : list
        contains a np.recarray with localizations for each channel, 
        reduced in case of fast rendering
    locs_paths : list
        contains a str defining the path for each channel
    median_lp : float
        median lateral localization precision of the first locs file
        (pixels)
    _mode : str
        defines current mode (zoom, pick or measure); important for 
        mouseEvents
    n_locs : int
        number of localizations loaded; if multichannel, the sum is
        given
    origin : QPoint
        position of the origin of the zoom-in rectangle
    _pan : boolean
        indicates if image is currently panned
    pan_start_x : float
        x coordinate of panning's starting position
    pan_start_y : float
        y coordinate of panning's starting position
    _picks : list
        contains the coordatines of current picks
    _pixmap : QPixMap
        Pixmap currently displayed
    _points : list
        contains the coordinates of points to measure distances
        between them
    qimage : QImage
        current image of rendered locs, picks and other drawings
    qimage_no_picks : QImage
        current image of rendered locs without picks and measuring
        points
    rectangle_pick_current_x : float
        x coordinate of the leading edge of the drawn rectangular pick
    rectangle_pick_current_y : float
        y coordinate of the leading edge of the drawn rectangular pick
    _rectangle_pick_ongoing : boolean
        indicates if a rectangular pick is currently drawn
    rectangle_pick_start : tuple
        (rectangle_pick_start_x, rectangle_pick_start_y), see below
    rectangle_pick_start_x : float
        x coordinate of the starting edge of the drawn rectangular pick
    rectangle_pick_start_y : float
        y coordinate of the starting edge of the drawn rectangular pick
    rubberband : QRubberBand
        draws a rectangle used in zooming in
    _size_hint : tuple
        used for size adjustment
    unfold_status : str
        specifies if unfold/refold groups
    window : QMainWindow
        instance of the main window
    x_color : np.array
        indexes each loc according to its parameter value; 
        see self.activate_render_property
    x_locs : list
        contains np.recarrays with locs to be rendered by property; one
        per color
    x_render_cache : list
        contains dicts with caches for storing info about locs rendered
        by a property
    x_render_state : boolean
        indicates if rendering by property is used

    Methods
    -------
    activate_property_menu()
        Allows changing render parameters
    activate_render_property()
        Assigns locs by color to render a chosen property
    add(path)
        Loads a .hdf5 and .yaml files
    add_drift(channel, drift)
        Assigns attributes and saves .txt drift file
    add_multiple(paths)
        Loads several .hdf5 and .yaml files
    add_pick(position)
        Adds a pick at a given position
    add_point(position)
        Adds a point at a given position for measuring distances
    add_picks(positions)
        Adds several picks
    adjust_viewport_to_view(viewport)
        Adds space to viewport to match self.window's aspect ratio
    align()
        Align channels by RCC or from picked locs
    analyze_cluster()
        Clusters picked locs using k-means clustering
    apply_drift()
        Applies drift to locs from a .txt file
    combine()
        Combines all locs in each pick into one localization
    clear_picks()
        Deletes all current picks
    CPU_or_GPU_box()
        Creates a message box with buttons to choose between 
        CPU and GPU SMLM clustering.
    dbscan()
        Gets channel, parameters and path for DBSCAN
    _dbscan(channel, path, params)
        Performs DBSCAN in a given channel with user-defined parameters
    deactivate_property_menu()
        Blocks changing render parameters
    display_pixels_per_viewport_pixels()
        Returns optimal oversampling
    dragEnterEvent(event)
        Defines what happens when a file is dragged onto the main 
        window
    draw_minimap(image)
        Draws a minimap showing the position of the current viewport
    draw_legend(image)
        Draws legend for multichannel data
    draw_picks(image)
        Draws all picks onto rendered localizations
    draw_points(image)
        Draws points and lines and distances between them
    draw_rectangle_pick_ongoing(image)
        Draws an ongoing rectangular pick onto rendered localizations
    draw_scalebar(image)
        Draws a scalebar
    draw_scene(viewport)
        Renders locs in the given viewport and draws picks, legend, etc
    draw_scene_slicer(viewport)
        Renders sliced locs in the given viewport and draws picks etc
    dropEvent(event)
        Defines what happens when a file is dropped onto the window
    export_trace()
        Saves trace as a .csv
    filter_picks()
        Filters picks by number of locs
    fit_in_view()
        Updates scene with all locs shown
    get_channel()
        Opens an input dialog to ask for a channel
    get_channel3d()
        Similar to get_channel, used in selecting 3D picks
    get_channel_all_seq()
        Similar to get_channel, adds extra index for applying to all
        channels
    get_group_color(locs)
        Finds group color index for each localization
    get_index_blocks(channel)
        Calls self.index_locs if not calculated earlier
    get_pick_rectangle_corners(start_x, start_y, end_x, end_y, width)
        Finds the positions of a rectangular pick's corners
    get_pick_rectangle_polygon(start_x, start_y, end_x, end_y, width)
        Finds a PyQt5 object used for drawing a rectangular pick
    get_render_kwargs()
        Returns a dictionary to be used for the kwargs of render.render
    hdscan()
        Gets channel, parameters and path for HDBSCAN
    _hdbscan(channel, path, params)
        Performs HDBSCAN in a given channel with user-defined 
        parameters
    index_locs(channel)
        Indexes locs from channel in a grid
    load_picks(path)
        Loads picks from .yaml file defined by path
    link
    map_to_movie(position)
        Converts coordinates from display units to camera units
    map_to_view(x,y)
        Converts coordinates from camera units to display units
    max_movie_height()
        Returns maximum height of all loaded images
    max_movie_width()
        Returns maximum width of all loaded images
    mouseMoveEvent(event)
        Defines actions taken when moving mouse
    mousePressEvent(event)
        Defines actions taken when pressing mouse button
    mouseReleaseEvent(event)
        Defines actions taken when releasing mouse button
    move_to_pick()
        Change viewport to show a pick identified by its id
    movie_size()
        Returns tuple with movie height and width
    nearest_neighbor()
        Gets channels for nearest neighbor analysis
    _nearest_neighbor(channel1, channel2)
        Calculates and saves distances of the nearest neighbors between
        localizations in channels 1 and 2
    on_pick_shape_changed(pick_shape_index)
        Assigns attributes and updates scene if new pick shape chosen
    pan_relative(dy, dx)
        Moves viewport by a given relative distance
    pick_message_box(params)
        Returns a message box for selecting picks
    pick_similar()
        Searches picks similar to the current picks
    picked_locs(channel)
        Returns picked localizations in the specified channel
    read_colors()
        Finds currently selected colors for multicolor rendering
    refold_groups()
        Refolds grouped locs across x axis
    relative_position(viewport_center, cursor_position)
        Finds the position of the cursor relative to the viewport's
        center
    remove_points()
        Removes all distance measurement points
    remove_picks(position)
        Deletes picks at a given position
    remove_picked_locs()
        Gets channel for removing picked localizations
    _remove_picked_locs(channel)
        Deletes localizations in picks in channel
    render_multi_channel(kwargs)
        Renders and paints multichannel locs
    render_scene()
        Returns QImage with rendered localizations
    render_single_channel(kwargs)
        Renders single channel localizations
    resizeEvent()
        Defines what happens when window is resized
    rmsd_at_com(locs)
        Calculates root mean square displacement at center of mass
    save_channel()
        Opens an input dialog asking which channel of locs to save
    save_channel_pickprops()
        Opens an input dialog asking which channel to use in saving 
        pick properties
    save_pick_properties(path, channel)
        Saves picks' properties in a given channel to path
    save_picked_locs(path, channel)
        Saves picked locs from channel to path
    save_picked_locs_multi(path)
        Saves picked locs combined from all channels to path
    save_picks(path)
        Saves picked regions in .yaml format
    scale_contrast(image)
        Scales image based on contrast value from Display Settings
        Dialog
    select_traces()
        Lets user to select picks based on their traces
    set_mode()
        Sets self._mode for QMouseEvents
    set_property()
        Activates rendering by property
    set_zoom(zoom)
        Zooms in/out to the given value
    shifts_from_picked_coordinate(locs, coordinate)
        Calculates shifts between channels along a given coordinate
    shift_from_picked()
        For each pick, calculate the center of mass and rcc based on 
        shifts
    shift_from_rcc()
        Estimates image shifts based on whole images' rcc
    show_drift()
        Plots current drift
    show_legend()
        Displays legend for rendering by property
    show_pick()
        Lets user select picks based on their 2D scatter
    show_pick_3d
        Lets user select picks based on their 3D scatter
    show_pick_3d_iso
        Lets user select picks based on their 3D scatter and 
        projections
    show_trace()
        Displays x and y coordinates of locs in picks in time
    sizeHint()
        Returns recommended window size
    smlm_clusterer()
        Gets channel, parameters and path for SMLM clustering
    _smlm_clusterer(channel, path, params)
        Performs SMLM clustering in a given channel with user-defined 
        parameters
    subtract_picks(path)
        Clears current picks that cover the picks loaded from path
    to_8bit(image)
        Converted normalised image to 8 bit
    to_down()
        Called on pressing down arrow; moves FOV
    to_left()
        Called on pressing left arrow; moves FOV
    to_right()
        Called on pressing right arrow; moves FOV
    to_up()
        Called on pressing up arrow; moves FOV
    undrift()
        Undrifts with RCC
    undrift_from_picked
        Gets channel for undrifting from picked locs
    _undrift_from_picked
        Undrifts based on picked locs in a given channel
    _undrift_from_picked_coordinate
        Calculates drift in a given coordinate
    undrift_from_picked2d
        Gets channel for undrifting from picked locs in 2D
    _undrift_from_picked2d
        Undrifts in x and y based on picked locs in a given channel
    undo_drift
        Gets channel for undoing drift
    _undo_drift
        Deletes the latest drift in a given channel
    unfold_groups()
        Shifts grouped locs across x axis
    unfold_groups_square()
        Shifts grouped locs onto a rectangular grid of chosen length
    update_cursor()
        Changes cursor according to self._mode
    update_pick_info_long()
        Called when evaluating picks statistics in Info Dialog
    update_pick_info_short()
        Updates number of picks in Info Dialog
    update_scene()
        Updates the view of rendered locs as well as cursor
    update_scene_slicer()
        Updates the view of rendered locs if they are sliced
    viewport_center()
        Finds viewport's center (pixels)
    viewport_height()
        Finds viewport's height (pixels)
    viewport_size()
        Finds viewport's height and width (pixels)
    viewport_width()
        Finds viewport's width (pixels)
    wheelEvent(QWheelEvent)
        Defines what happens when mouse wheel is used
    zoom(factor)
        Changes zoom relatively to factor
    zoom_in()
        Zooms in by a constant factor
    zoom_out()
        Zooms out by a constant factor
    """

    def __init__(self, window):
        super().__init__()
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setAcceptDrops(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.rubberband = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self
        )
        self.rubberband.setStyleSheet("selection-background-color: white")
        self.window = window
        self._pixmap = None
        self.all_locs = [] # for fast render
        self.locs = []
        self.infos = []
        self.locs_paths = []
        self.group_color = []
        self._mode = "Zoom"
        self._pan = False
        self._rectangle_pick_ongoing = False
        self._size_hint = (768, 768)
        self.n_locs = 0
        self._picks = []
        self._points = []
        self.index_blocks = []
        self._drift = []
        self._driftfiles = []
        self.currentdrift = []
        self.x_render_cache = []
        self.x_render_state = False

    def get_group_color(self, locs):
        """ 
        Finds group color for each localization in single channel data
        with group info.

        Parameters
        ----------
        locs : np.recarray
            Array with all localizations

        Returns
        -------
        np.array
            Array with int group color index for each loc
        """

        return locs.group.astype(int) % N_GROUP_COLORS

    def add(self, path, render=True):
        """
        Loads a .hdf5 localizations and the associated .yaml metadata 
        files. 

        Parameters
        ----------
        path : str
            String specifying the path to the .hdf5 file
        render : boolean, optional
            Specifies if the loaded files should be rendered 
            (default True)
        """

        # read .hdf5 and .yaml files
        try:
            locs, info = io.load_locs(path, qt_parent=self)
        except io.NoMetadataFileError:
            return
        locs = lib.ensure_sanity(locs, info)

        # update pixelsize
        for element in info:
            if "Picasso Localize" in element.values():
                if "Pixelsize" in element:
                    self.window.display_settings_dlg.pixelsize.setValue(
                        element["Pixelsize"]
                    )

        # append loaded data
        self.locs.append(locs)
        self.all_locs.append(copy.copy(locs)) # for fast rendering
        self.infos.append(info)
        self.locs_paths.append(path)
        self.index_blocks.append(None)

        # try to load a drift .txt file:
        drift = None
        if "Last driftfile" in info[-1]:
            driftpath = info[-1]["Last driftfile"]
            if driftpath is not None:
                try:
                    with open(driftpath, "r") as f:
                        drifttxt = np.loadtxt(f)
                    drift_x = drifttxt[:, 0]
                    drift_y = drifttxt[:, 1]

                    if drifttxt.shape[1] == 3:
                        drift_z = drifttxt[:, 2]
                        drift = (drift_x, drift_y, drift_z)
                        drift = np.rec.array(
                            drift, dtype=[("x", "f"), ("y", "f"), ("z", "f")]
                        )
                    else:
                        drift = (drift_x, drift_y)
                        drift = np.rec.array(
                            drift, dtype=[("x", "f"), ("y", "f")]
                        )
                except Exception as e:
                    print(e)
                    # drift already initialized before
                    pass

        # append drift info
        self._drift.append(drift)
        self._driftfiles.append(None)
        self.currentdrift.append(None)

        # if this is the first loc file, find the median localization
        # precision and set group colors, if needed
        if len(self.locs) == 1:
            self.median_lp = np.mean(
                [np.median(locs.lpx), np.median(locs.lpy)]
            )
            if hasattr(locs, "group"):
                if len(self.group_color) == 0 and locs.group.size:
                    self.group_color = self.get_group_color(self.locs[0])

        # render the loaded file
        if render:
            self.fit_in_view(autoscale=True)
            self.update_scene()

        # add options to rendering by parameter
        self.window.display_settings_dlg.parameter.addItems(locs.dtype.names)

        if hasattr(locs, "z"):
            # append z coordinates for slicing
            self.window.slicer_dialog.zcoord.append(locs.z)
            # unlock 3D settings
            for action in self.window.actions_3d:
                action.setVisible(True)

        # allow using View, Tools and Postprocess menus
        for menu in self.window.menus:
            menu.setDisabled(False)

        # change current working directory
        os.chdir(os.path.dirname(path))

        # add the locs to the dataset dialog
        self.window.dataset_dialog.add_entry(path)

        self.window.setWindowTitle(
            "Picasso v{}: Render. File: {}".format(
                __version__, os.path.basename(path)
            )
        )

        # fast rendering add channel
        self.window.fast_render_dialog.on_file_added()

    def add_multiple(self, paths):
        """ Loads several .hdf5 and .yaml files. 
        
        Parameters
        ----------
        paths: list
            Contains the paths to the files to be loaded
        """

        if len(paths):
            fit_in_view = len(self.locs) == 0
            paths = sorted(paths)
            pd = lib.ProgressDialog(
                "Loading channels", 0, len(paths), self
            )
            pd.set_value(0)
            pd.setModal(False)
            for i, path in enumerate(paths):
                try:
                    self.add(path, render=False)
                except:
                    pass
                pd.set_value(i+1)
            if len(self.locs):  # if loading was successful
                if fit_in_view:
                    self.fit_in_view(autoscale=True)
                else:
                    self.update_scene()

    def add_pick(self, position, update_scene=True):
        """ Adds a pick at a given position. """

        self._picks.append(position)
        self.update_pick_info_short()
        if update_scene:
            self.update_scene(picks_only=True)

    def add_picks(self, positions):
        """ Adds several picks. """

        for position in positions:
            self.add_pick(position, update_scene=False)
        self.update_scene(picks_only=True)

    def add_point(self, position, update_scene=True):
        """ 
        Adds a point at a given position for measuring distances.
        """

        self._points.append(position)
        if update_scene:
            self.update_scene()

    def adjust_viewport_to_view(self, viewport):
        """
        Adds space to a desired viewport, such that it matches the 
        window aspect ratio. Returns a viewport.
        """

        viewport_height = viewport[1][0] - viewport[0][0]
        viewport_width = viewport[1][1] - viewport[0][1]
        view_height = self.height()
        view_width = self.width()
        viewport_aspect = viewport_width / viewport_height
        view_aspect = view_width / view_height
        if view_aspect >= viewport_aspect:
            y_min = viewport[0][0]
            y_max = viewport[1][0]
            x_range = viewport_height * view_aspect
            x_margin = (x_range - viewport_width) / 2
            x_min = viewport[0][1] - x_margin
            x_max = viewport[1][1] + x_margin
        else:
            x_min = viewport[0][1]
            x_max = viewport[1][1]
            y_range = viewport_width / view_aspect
            y_margin = (y_range - viewport_height) / 2
            y_min = viewport[0][0] - y_margin
            y_max = viewport[1][0] + y_margin
        return [(y_min, x_min), (y_max, x_max)]

    def align(self):
        """ Align channels by RCC or from picked localizations. """

        if len(self._picks) > 0: # shift from picked
            # find shift between channels
            shift = self.shift_from_picked()
            print("Shift {}".format(shift))
            sp = lib.ProgressDialog(
                "Shifting channels", 0, len(self.locs), self
            )
            sp.set_value(0)

            # align each channel
            for i, locs_ in enumerate(self.locs):
                locs_.y -= shift[0][i]
                locs_.x -= shift[1][i]
                if len(shift) == 3:
                    locs_.z -= shift[2][i]
                self.all_locs[i] = copy.copy(locs_)
                # Cleanup
                self.index_blocks[i] = None
                sp.set_value(i + 1)
            self.update_scene()

        else: # align using whole images
            max_iterations = 5
            iteration = 0
            convergence = 0.001  # (pixels), around 0.1 nm
            shift_x = []
            shift_y = []
            shift_z = []
            display = False

            progress = lib.ProgressDialog(
                "Aligning images..", 0, max_iterations, self
            )
            progress.show()
            progress.set_value(0)

            for iteration in range(max_iterations):
                completed = True
                progress.set_value(iteration)

                # find shift between channels
                shift = self.shift_from_rcc()
                sp = lib.ProgressDialog(
                    "Shifting channels", 0, len(self.locs), self
                )
                sp.set_value(0)
                temp_shift_x = []
                temp_shift_y = []
                temp_shift_z = []
                for i, locs_ in enumerate(self.locs):
                    if (
                        np.absolute(shift[0][i]) + np.absolute(shift[1][i])
                        > convergence
                    ):
                        completed = False

                    # shift each channel
                    locs_.y -= shift[0][i]
                    locs_.x -= shift[1][i]

                    temp_shift_x.append(shift[1][i])
                    temp_shift_y.append(shift[0][i])

                    if len(shift) == 3:
                        locs_.z -= shift[2][i]
                        temp_shift_z.append(shift[2][i])
                    sp.set_value(i + 1)
                self.all_locs = copy.copy(self.locs)
                shift_x.append(np.mean(temp_shift_x))
                shift_y.append(np.mean(temp_shift_y))
                if len(shift) == 3:
                    shift_z.append(np.mean(temp_shift_z))
                iteration += 1
                self.update_scene()

                # Skip when converged:
                if completed:
                    break

            progress.close()

            # Plot shift
            if display:
                fig1 = plt.figure(figsize=(8, 8))
                plt.suptitle("Shift")
                plt.subplot(1, 1, 1)
                plt.plot(shift_x, "o-", label="x shift")
                plt.plot(shift_y, "o-", label="y shift")
                plt.xlabel("Iteration")
                plt.ylabel("Mean Shift per Iteration (Px)")
                plt.legend(loc="best")
                fig1.show()

    @check_pick
    def combine(self):
        """ 
        Combines locs in picks. 

        Works by linking all locs in each pick region, leading to only
        one loc per pick.

        See View.link for more info.
        """

        channel = self.get_channel()
        picked_locs = self.picked_locs(channel, add_group=False)
        out_locs = []

        # use very large values for linking localizations
        r_max = 2 * max(
            self.infos[channel][0]["Height"], self.infos[channel][0]["Width"]
        )
        max_dark = self.infos[channel][0]["Frames"]
        progress = lib.ProgressDialog(
            "Combining localizations in picks", 0, len(picked_locs), self
        )

        # link every localization in each pick
        for i, pick_locs in enumerate(picked_locs):
            pick_locs_out = postprocess.link(
                pick_locs,
                self.infos[channel],
                r_max=r_max,
                max_dark_time=max_dark,
                remove_ambiguous_lengths=False,
            )
            if not pick_locs_out:
                print("no locs in pick - skipped")
            else:
                out_locs.append(pick_locs_out)
            progress.set_value(i + 1)
        self.all_locs[channel] = stack_arrays(
            out_locs, asrecarray=True, usemask=False
        )
        self.locs[channel] = copy.copy(self.all_locs[channel])

        if hasattr(self.all_locs[channel], "group"):
            groups = np.unique(self.all_locs[channel].group)
            # In case a group is missing
            groups = np.arange(np.max(groups) + 1)
            np.random.shuffle(groups)
            groups %= N_GROUP_COLORS
            self.group_color = groups[self.all_locs[channel].group]

        self.update_scene()

    def link(self):
        """
        Link localizations 
        """

        channel = self.get_channel()
        if hasattr(self.all_locs[channel], "len"):
            QtWidgets.QMessageBox.information(
                self, "Link", "Localizations are already linked. Aborting..."
            )
            return
        else:
            r_max, max_dark, ok = LinkDialog.getParams()
            if ok:
                status = lib.StatusDialog("Linking localizations...", self)
                self.all_locs[channel] = postprocess.link(
                    self.all_locs[channel],
                    self.infos[channel],
                    r_max=r_max,
                    max_dark_time=max_dark,
                )
                status.close()
                if hasattr(self.all_locs[channel], "group"):
                    groups = np.unique(self.all_locs[channel].group)
                    groups = np.arange(np.max(groups) + 1)
                    np.random.shuffle(groups)
                    groups %= N_GROUP_COLORS
                    self.group_color = groups[self.all_locs[channel].group]
                self.locs[channel] = copy.copy(self.all_locs[channel])
                self.update_scene()

    def dbscan(self):
        """
        Gets channel, parameters and path for DBSCAN.
        """

        channel = self.get_channel_all_seq("Cluster")

        # get DBSCAN parameters
        params = DbscanDialog.getParams()
        ok = params[-1] # true if parameters were given

        if ok:
            if channel == len(self.locs_paths): # apply to all channels
                # get saving name suffix
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_clustered",
                )
                if ok:
                    for channel in range(len(self.locs_paths)):
                        path = self.locs_paths[channel].replace(
                            ".hdf5", f"{suffix}.hdf5"
                        )
                        self._dbscan(channel, path, params)
            else:
                # get the path to save
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save clustered locs",
                    self.locs_paths[channel].replace(".hdf5", "_clustered.hdf5"),
                    filter="*.hdf5",
                )
                if path:
                    self._dbscan(channel, path, params)

    def _dbscan(self, channel, path, params):
        """
        Performs DBSCAN in a given channel with user-defined parameters
        and saves the result.

        Parameters
        ----------
        channel : int
            Index of the channel were clustering is performed
        path : str
            Path to save clustered localizations
        params : list
            DBSCAN parameters
        """

        radius, min_density, save_centers, _ = params
        status = lib.StatusDialog(
            "Applying DBSCAN. This may take a while.", self
        )
        # keep group info if already present
        if hasattr(self.all_locs[channel], "group"):
            locs = lib.append_to_rec(
                self.all_locs[channel],
                self.all_locs[channel].group,
                "group_input",
            )
        else: 
            locs = self.all_locs[channel]

        pixelsize = self.window.display_settings_dlg.pixelsize.value()

        # perform DBSCAN in a channel
        locs = clusterer.dbscan(
            locs, 
            radius, 
            min_density,
            pixelsize=pixelsize,
        )
        dbscan_info = {
            "Generated by": "Picasso DBSCAN",
            "Number of clusters": len(np.unique(locs.group)),
            "Radius [cam. px]": radius,
            "Minimum local density": min_density,
        }

        io.save_locs(path, locs, self.infos[channel] + [dbscan_info])
        status.close()
        if save_centers:
            status = lib.StatusDialog("Calculating cluster centers", self)
            path = path.replace(".hdf5", "_cluster_centers.hdf5")
            centers = clusterer.find_cluster_centers(locs, pixelsize=pixelsize)
            io.save_locs(path, centers, self.infos[channel] + [dbscan_info])
            status.close()

    def hdbscan(self):
        """
        Gets channel, parameters and path for HDBSCAN.
        """

        channel = self.get_channel_all_seq("Cluster")

        # get HDBSCAN parameters
        params = HdbscanDialog.getParams()
        ok = params[-1] # true if parameters were given

        if ok:
            if channel == len(self.locs_paths): # apply to all channels
                # get saving name suffix
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_clustered",
                )
                if ok:
                    for channel in range(len(self.locs_paths)):
                        path = self.locs_paths[channel].replace(
                            ".hdf5", f"{suffix}.hdf5"
                        )
                        self._hdbscan(channel, path, params)
            else:
                # get the path to save
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save clustered locs",
                    self.locs_paths[channel].replace(
                        ".hdf5", 
                        "_clustered.hdf5"
                    ),
                    filter="*.hdf5",
                )
                if path:
                    self._hdbscan(channel, path, params)

    def _hdbscan(self, channel, path, params):
        """
        Performs HDBSCAN in a given channel with user-defined 
        parameters and saves the result.

        Parameters
        ----------
        channel : int
            Index of the channel were clustering is performed
        path : str
            Path to save clustered localizations
        params : list
            HDBSCAN parameters
        """

        min_cluster, min_samples, cluster_eps, save_centers, _ = params
        status = lib.StatusDialog(
            "Applying HDBSCAN. This may take a while.", self
        )
        # keep group info if already present
        if hasattr(self.all_locs[channel], "group"):
            locs = lib.append_to_rec(
                self.all_locs[channel],
                self.all_locs[channel].group,
                "group_input",
            )
        else: 
            locs = self.all_locs[channel]

        pixelsize = self.window.display_settings_dlg.pixelsize.value()

        # perform HDBSCAN for each channel
        locs = clusterer.hdbscan(
            locs,
            min_cluster, 
            min_samples, 
            pixelsize=pixelsize,
            cluster_eps=cluster_eps,
        )
        hdbscan_info = {
            "Generated by": "Picasso HDBSCAN",
            "Number of clusters": len(np.unique(locs.group)),
            "Min. cluster": min_cluster,
            "Min. samples": min_samples,
            "Intercluster distance": cluster_eps,
        }

        io.save_locs(path, locs, self.infos[channel] + [hdbscan_info])
        status.close()
        if save_centers:
            status = lib.StatusDialog("Calculating cluster centers", self)
            path = path.replace(".hdf5", "_cluster_centers.hdf5")
            centers = clusterer.find_cluster_centers(locs, pixelsize=pixelsize)
            io.save_locs(path, centers, self.infos[channel] + [hdbscan_info])
            status.close()

    def smlm_clusterer(self):
        """
        Gets channel, parameters and path for SMLM clustering
        """

        channel = self.get_channel_all_seq("Cluster")

        # get clustering parameters
        if any([hasattr(_, "z") for _ in self.all_locs]):
            params = SMLMDialog3D.getParams()
        else:
            params = SMLMDialog2D.getParams()

        ok = params[-1] # true if parameters were given

        if ok:

            if channel == len(self.locs_paths): # apply to all
                # get saving name suffix
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_clustered",
                )
                if ok:
                    for channel in range(len(self.locs_paths)):
                        path = self.locs_paths[channel].replace(
                            ".hdf5", f"{suffix}.hdf5"
                        ) # add the suffix to the current path
                        self._smlm_clusterer(channel, path, params)
            else:
                # get the path to save
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save clustered locs",
                    self.locs_paths[channel].replace(
                        ".hdf5", "_clustered.hdf5"
                    ),
                    filter="*.hdf5",
                )
                if path:
                    self._smlm_clusterer(channel, path, params)

    def _smlm_clusterer(self, channel, path, params):
        """
        Performs SMLM clustering in a given channel with user-defined
        parameters and saves the result.

        Parameters
        ----------
        channel : int
            Index of the channel were clustering is performed
        path : str
            Path to save clustered localizations
        params : list
            SMLM clustering parameters        
        """

        # for converting z coordinates
        pixelsize = self.window.display_settings_dlg.pixelsize.value()

        if len(self._picks): # cluster only picked localizations
            clustered_locs = [] # list with picked locs after clustering
            picked_locs = self.picked_locs(channel, add_group=False)
            group_offset = 1
            pd = lib.ProgressDialog(
                "Clustering in picks", 0, len(picked_locs), self
            )
            pd.set_value(0)
            for i in range(len(picked_locs)):
                locs = picked_locs[i]

                # save pick index as group_input
                locs = lib.append_to_rec(
                    locs,
                    i * np.ones(len(locs), dtype=np.int32), 
                    "group_input",
                )

                if len(locs) > 0:
                    temp_locs = clusterer.cluster(
                        locs, params, pixelsize=pixelsize
                    )

                    if len(temp_locs) > 0:
                        # make sure each picks produces unique cluster ids
                        temp_locs.group += group_offset
                        clustered_locs.append(temp_locs)
                        group_offset += np.max(temp_locs.group) + 1
                pd.set_value(i + 1)
            clustered_locs = stack_arrays(
                clustered_locs, asrecarray=True, usemask=False
            ) # np.recarray with all clustered locs to be saved

        else: # cluster all locs
            status = lib.StatusDialog("Clustering localizations", self)

            # keep group info if already present
            if hasattr(self.all_locs[channel], "group"):
                locs = lib.append_to_rec(
                    self.all_locs[channel], 
                    self.all_locs[channel].group, 
                    "group_input",
                )
            else:
                locs = self.all_locs[channel]

            clustered_locs = clusterer.cluster(
                locs, params, pixelsize=pixelsize
            )
            status.close()

        # saving
        if hasattr(self.all_locs[channel], "z"):
            new_info = {
                "Generated by": "Picasso Render SMLM clusterer 3D",
                "Number of clusters": len(np.unique(clustered_locs.group)),
                "Clustering radius xy [cam. px]": params[0],
                "Clustering radius z [cam. px]": params[1],
                "Min. cluster size": params[2],
                "Performed basic frame analysis": params[-2],
            }            
        else:
            new_info = {
                "Generated by": "Picasso Render SMLM clusterer 2D",
                "Number of clusters": len(np.unique(clustered_locs.group)),
                "Clustering radius [cam. px]": params[0],
                "Min. cluster size": params[1],
                "Performed basic frame analysis": params[-2],
            }
        info = self.infos[channel] + [new_info]

        # save locs
        io.save_locs(path, clustered_locs, info)
        # save cluster centers
        if params[-3]:
            status = lib.StatusDialog("Calculating cluster centers", self)
            path = path.replace(".hdf5", "_cluster_centers.hdf5")
            centers = clusterer.find_cluster_centers(clustered_locs, pixelsize)
            io.save_locs(path, centers, info)
            status.close()

    def shifts_from_picked_coordinate(self, locs, coordinate):
        """
        Calculates shifts between channels along a given coordinate.

        Parameters
        ----------
        locs : np.recarray
            Picked locs from all channels
        coordinate : str
            Specifies which coordinate should be used (x, y, z)

        Returns
        -------
        np.array
            Array of shape (n_channels, n_channels) with shifts between
            all channels
        """

        n_channels = len(locs)
        # Calculating center of mass for each channel and pick
        coms = []
        for channel_locs in locs:
            coms.append([])
            for group_locs in channel_locs:
                group_com = np.mean(getattr(group_locs, coordinate))
                coms[-1].append(group_com)
        # Calculating image shifts
        d = np.zeros((n_channels, n_channels))
        for i in range(n_channels - 1):
            for j in range(i + 1, n_channels):
                d[i, j] = np.nanmean(
                    [cj - ci for ci, cj in zip(coms[i], coms[j])]
                )
        return d

    def shift_from_picked(self):
        """
        Used by align. For each pick, calculate the center of mass and 
        rcc based on shifts.

        Returns
        -------
        tuple
            With shifts; shape (2,) or (3,) (if z coordinate present)
        """
        n_channels = len(self.locs)
        locs = [self.picked_locs(_) for _ in range(n_channels)]
        dy = self.shifts_from_picked_coordinate(locs, "y")
        dx = self.shifts_from_picked_coordinate(locs, "x")
        if all([hasattr(_[0], "z") for _ in locs]):
            dz = self.shifts_from_picked_coordinate(locs, "z")
        else:
            dz = None
        return lib.minimize_shifts(dx, dy, shifts_z=dz)

    def shift_from_rcc(self):
        """
        Used by align. Estimates image shifts based on whole images' 
        rcc.

        Returns
        -------
        tuple
            With shifts; shape (2,) or (3,) (if z coordinate present)
        """
        n_channels = len(self.locs)
        rp = lib.ProgressDialog("Rendering images", 0, n_channels, self)
        rp.set_value(0)
        images = []
        # render each channel and save it in images
        for i, (locs_, info_) in enumerate(zip(self.locs, self.infos)):
            _, image = render.render(locs_, info_, blur_method="smooth")
            images.append(image)
            rp.set_value(i + 1)
        n_pairs = int(n_channels * (n_channels - 1) / 2)
        rc = lib.ProgressDialog("Correlating image pairs", 0, n_pairs, self)
        return imageprocess.rcc(images, callback=rc.set_value)

    @check_pick
    def clear_picks(self):
        """ Deletes all current picks. """

        self._picks = []
        self.window.info_dialog.n_picks.setText(str(len(self._picks)))
        self.update_scene(picks_only=True)

    def dragEnterEvent(self, event):
        """ 
        Defines what happens when a file is dragged onto the window.
        """

        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def get_pick_rectangle_corners(
        self, start_x, start_y, end_x, end_y, width
    ):
        """
        Finds the positions of corners of a rectangular pick.
        Rectangular pick is defined by:
            [(start_x, start_y), (end_x, end_y)]
        and its width. (all values in pixels)

        Returns
        -------
        tuple
            Contains corners' x and y coordinates in two lists
        """

        if end_x == start_x:
            alpha = np.pi / 2
        else:
            alpha = np.arctan((end_y - start_y) / (end_x - start_x))
        dx = width * np.sin(alpha) / 2
        dy = width * np.cos(alpha) / 2
        x1 = int(start_x - dx)
        x2 = int(start_x + dx)
        x4 = int(end_x - dx)
        x3 = int(end_x + dx)
        y1 = int(start_y + dy)
        y2 = int(start_y - dy)
        y4 = int(end_y + dy)
        y3 = int(end_y - dy)
        return [x1, x2, x3, x4], [y1, y2, y3, y4]

    def get_pick_rectangle_polygon(
        self, start_x, start_y, end_x, end_y, width, return_most_right=False
    ):
        """
        Finds QtGui.QPolygonF object used for drawing a rectangular
        pick.

        Returns
        -------
        QtGui.QPolygonF
        """

        X, Y = self.get_pick_rectangle_corners(
            start_x, start_y, end_x, end_y, width
        )
        p = QtGui.QPolygonF()
        for x, y in zip(X, Y):
            p.append(QtCore.QPointF(x, y))
        if return_most_right:
            ix_most_right = np.argmax(X)
            x_most_right = X[ix_most_right]
            y_most_right = Y[ix_most_right]
            return p, (x_most_right, y_most_right)
        return p

    def draw_picks(self, image):
        """ 
        Draws all current picks onto rendered locs.
        
        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn picks
        """

        image = image.copy()
        t_dialog = self.window.tools_settings_dialog

        # draw circular picks
        if self._pick_shape == "Circle":

            # draw circular picks as points
            if t_dialog.point_picks.isChecked():
                painter = QtGui.QPainter(image)
                painter.setBrush(QtGui.QBrush(QtGui.QColor("yellow")))
                painter.setPen(QtGui.QColor("yellow"))

                # yellow is barely visible on white background
                if self.window.dataset_dialog.wbackground.isChecked():
                    painter.setBrush(QtGui.QBrush(QtGui.QColor("red")))
                    painter.setPen(QtGui.QColor("red"))

                for i, pick in enumerate(self._picks):

                    # convert from camera units to display units
                    cx, cy = self.map_to_view(*pick)
                    painter.drawEllipse(QtCore.QPoint(cx, cy), 3, 3)

                    # annotate picks
                    if t_dialog.pick_annotation.isChecked():
                        painter.drawText(cx + 20, cy + 20, str(i))
                painter.end()

            # draw circles
            else:
                d = t_dialog.pick_diameter.value()
                d *= self.width() / self.viewport_width()
                d = int(d)

                painter = QtGui.QPainter(image)
                painter.setPen(QtGui.QColor("yellow"))

                # yellow is barely visible on white background
                if self.window.dataset_dialog.wbackground.isChecked():
                    painter.setPen(QtGui.QColor("red"))

                for i, pick in enumerate(self._picks):

                    # convert from camera units to display units
                    cx, cy = self.map_to_view(*pick)
                    painter.drawEllipse(int(cx - d / 2), int(cy - d / 2), d, d)

                    # annotate picks
                    if t_dialog.pick_annotation.isChecked():
                        painter.drawText(
                            int(cx + d / 2), int(cy + d / 2), str(i)
                        )
                painter.end()

        # draw rectangular picks
        elif self._pick_shape == "Rectangle":
            w = t_dialog.pick_width.value()
            w *= self.width() / self.viewport_width()

            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QColor("yellow"))

            # yellow is barely visible on white background
            if self.window.dataset_dialog.wbackground.isChecked():
                painter.setPen(QtGui.QColor("red"))

            for i, pick in enumerate(self._picks):

                # convert from camera units to display units
                start_x, start_y = self.map_to_view(*pick[0])
                end_x, end_y = self.map_to_view(*pick[1])

                # draw a straight line across the pick
                painter.drawLine(start_x, start_y, end_x, end_y)

                # draw a rectangle
                polygon, most_right = self.get_pick_rectangle_polygon(
                    start_x, start_y, end_x, end_y, w, return_most_right=True
                )
                painter.drawPolygon(polygon)

                # annotate picks
                if t_dialog.pick_annotation.isChecked():
                    painter.drawText(*most_right, str(i))
            painter.end()
        return image

    def draw_rectangle_pick_ongoing(self, image):
        """ 
        Draws an ongoing rectangular pick onto image.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn pick
        """
        
        image = image.copy()
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("green"))

        # draw a line across the pick
        painter.drawLine(
            self.rectangle_pick_start_x,
            self.rectangle_pick_start_y,
            self.rectangle_pick_current_x,
            self.rectangle_pick_current_y,
        )

        w = self.window.tools_settings_dialog.pick_width.value()

        # convert from camera units to display units
        w *= self.width() / self.viewport_width()

        polygon = self.get_pick_rectangle_polygon(
            self.rectangle_pick_start_x,
            self.rectangle_pick_start_y,
            self.rectangle_pick_current_x,
            self.rectangle_pick_current_y,
            w,
        )

        # draw a rectangle
        painter.drawPolygon(polygon)
        painter.end()
        return image

    def draw_points(self, image):
        """
        Draws points and lines and distances between them onto image.
        
        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn points
        """

        image = image.copy()
        d = 20 # width of the drawn crosses (window pixels)
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("yellow"))

        # yellow is barely visible on white background
        if self.window.dataset_dialog.wbackground.isChecked():
            painter.setPen(QtGui.QColor("red"))

        cx = []
        cy = []
        ox = [] # together with oldpoint used for drawing
        oy = [] # lines between points
        oldpoint = []
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
        for point in self._points:
            if oldpoint != []:
                ox, oy = self.map_to_view(*oldpoint) # convert to display units
            cx, cy = self.map_to_view(*point) # convert to display units

            # draw a cross
            painter.drawPoint(cx, cy)
            painter.drawLine(cx, cy, int(cx + d / 2), cy)
            painter.drawLine(cx, cy, cx, int(cy + d / 2))
            painter.drawLine(cx, cy, int(cx - d / 2), cy)
            painter.drawLine(cx, cy, cx, int(cy - d / 2))

            # draw a line between points and show distance
            if oldpoint != []:
                painter.drawLine(cx, cy, ox, oy)
                font = painter.font()
                font.setPixelSize(20)
                painter.setFont(font)

                # get distance with 2 decimal places
                distance = (
                    float(
                        int(
                            np.sqrt(
                                (
                                    (oldpoint[0] - point[0]) ** 2
                                    + (oldpoint[1] - point[1]) ** 2
                                )
                            )
                            * pixelsize
                            * 100
                        )
                    )
                    / 100
                )
                painter.drawText(
                    int((cx + ox) / 2 + d), 
                    int((cy + oy) / 2 + d), 
                    str(distance) + " nm",
                )
            oldpoint = point
        painter.end()
        return image

    def draw_scalebar(self, image):
        """
        Draws a scalebar.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn scalebar        
        """

        if self.window.display_settings_dlg.scalebar_groupbox.isChecked():
            pixelsize = self.window.display_settings_dlg.pixelsize.value()

            # length (nm)
            scalebar = self.window.display_settings_dlg.scalebar.value()
            length_camerapxl = scalebar / pixelsize
            length_displaypxl = int(
                round(self.width() * length_camerapxl / self.viewport_width())
            )
            height = 10 # display pixels
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))

            # white scalebar not visible on white background
            if self.window.dataset_dialog.wbackground.isChecked():
                painter.setBrush(QtGui.QBrush(QtGui.QColor("black")))

            # draw a rectangle
            x = self.width() - length_displaypxl - 35
            y = self.height() - height - 20
            painter.drawRect(x, y, length_displaypxl + 0, height + 0)

            # display scalebar's length
            if self.window.display_settings_dlg.scalebar_text.isChecked():
                font = painter.font()
                font.setPixelSize(20)
                painter.setFont(font)
                painter.setPen(QtGui.QColor("white"))

                # white scalebar not visible on white background
                if self.window.dataset_dialog.wbackground.isChecked():
                    painter.setPen(QtGui.QColor("black"))
                text_spacer = 40
                text_width = length_displaypxl + 2 * text_spacer
                text_height = text_spacer
                painter.drawText(
                    x - text_spacer,
                    y - 25,
                    text_width,
                    text_height,
                    QtCore.Qt.AlignHCenter,
                    str(scalebar) + " nm",
                )
        return image

    def draw_legend(self, image):
        """
        Draws a legend for multichannel data. 
        Displayed in the top left corner, shows the color and the name 
        of each channel.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn legend
        """

        if self.window.dataset_dialog.legend.isChecked():
            n_channels = len(self.locs_paths)
            painter = QtGui.QPainter(image)
            # initial positions
            x = 12
            y = 26
            dy = 24 # space between names
            for i in range(n_channels):
                if self.window.dataset_dialog.checks[i].isChecked():
                    painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                    colordisp = self.window.dataset_dialog.colordisp_all[i]
                    color = colordisp.palette().color(QtGui.QPalette.Window)
                    painter.setPen(QtGui.QPen(color))
                    font = painter.font()
                    font.setPixelSize(16)
                    painter.setFont(font)
                    text = self.window.dataset_dialog.checks[i].text()
                    painter.drawText(QtCore.QPoint(x, y), text)
                    y += dy
        return image

    def draw_minimap(self, image):
        """
        Draw a minimap showing the position of current viewport.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn minimap
        """

        if self.window.display_settings_dlg.minimap.isChecked():
            movie_height, movie_width = self.movie_size()
            length_minimap = 100
            height_minimap = int(movie_height / movie_width * 100)
            # draw in the upper right corner, overview rectangle
            x = self.width() - length_minimap - 20
            y = 20
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QColor("white"))
            if self.window.dataset_dialog.wbackground.isChecked():
                painter.setPen(QtGui.QColor("black"))
            painter.drawRect(x, y, length_minimap + 0, height_minimap + 0)
            painter.setPen(QtGui.QColor("yellow"))
            if self.window.dataset_dialog.wbackground.isChecked():
                painter.setPen(QtGui.QColor("red"))
            length = int(self.viewport_width() / movie_width * length_minimap)
            height = int(self.viewport_height() / movie_height * height_minimap)
            x_vp = int(self.viewport[0][1] / movie_width * length_minimap)
            y_vp = int(self.viewport[0][0] / movie_height * length_minimap)
            painter.drawRect(x + x_vp, y + y_vp, length + 0, height + 0)
        return image

    def draw_scene(
        self,
        viewport,
        autoscale=False,
        use_cache=False,
        picks_only=False,
    ):
        """
        Renders localizations in the given viewport and draws picks,
        legend, etc.

        Parameters
        ----------
        viewport : tuple
            Viewport defining the rendered FOV
        autoscale : boolean (default=False)
            True if contrast should be optimally adjusted
        use_cache : boolean (default=False)
            True if saved QImage of rendered locs is to be used
        picks_only : boolean (default=False)
            True if only picks and points are to be rendered
        """

        if not picks_only:
            # make sure viewport has the same shape as the main window
            self.viewport = self.adjust_viewport_to_view(viewport)
            # render locs
            qimage = self.render_scene(
                autoscale=autoscale, use_cache=use_cache
            )
            # scale image's size to the window
            qimage = qimage.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.KeepAspectRatioByExpanding,
            )
            # draw scalebar, minimap and legend
            self.qimage_no_picks = self.draw_scalebar(qimage)
            self.qimage_no_picks = self.draw_minimap(self.qimage_no_picks)
            self.qimage_no_picks = self.draw_legend(self.qimage_no_picks)
            # adjust zoom in Display Setting sDialog
            dppvp = self.display_pixels_per_viewport_pixels()
            self.window.display_settings_dlg.set_zoom_silently(dppvp)
        # draw picks and points
        self.qimage = self.draw_picks(self.qimage_no_picks)
        self.qimage = self.draw_points(self.qimage)
        if self._rectangle_pick_ongoing:
            self.qimage = self.draw_rectangle_pick_ongoing(self.qimage)

        # convert to pixmap
        self.pixmap = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.pixmap)
        self.window.update_info()

    def draw_scene_slicer(
        self,
        viewport,
        autoscale=False,
        use_cache=False,
        picks_only=False,
    ):
        """
        Renders sliced localizations in the given viewport and draws 
        picks, legend, etc.

        Parameters
        ----------
        viewport : tuple
            Viewport defining the current FOV
        autoscale : boolean (default=False)
            True if contrast should be optimally adjusted
        use_cache : boolean (default=False)
            True if saved QImage of rendered locs is to be used
        picks_only : boolean (default=False)
            True if only picks and points are to be rendered
        """

        # try to get a saved pixmap
        slicerposition = self.window.slicer_dialog.slicerposition
        pixmap = self.window.slicer_dialog.slicer_cache.get(slicerposition)

        if pixmap is None: # if no pixmap found
            self.draw_scene(
                viewport,
                autoscale=autoscale,
                use_cache=use_cache,
                picks_only=picks_only,
            )
            self.window.slicer_dialog.slicer_cache[slicerposition] = (
                self.pixmap
            )
        else:
            self.setPixmap(pixmap)

    def dropEvent(self, event):
        """ 
        Defines what happens when a file is dropped onto the window.
        If the file has ending .hdf5, attempts to load locs.
        """

        urls = event.mimeData().urls()
        paths = [_.toLocalFile() for _ in urls]
        extensions = [os.path.splitext(_)[1].lower() for _ in paths]
        if extensions == [".txt"]: # just one txt dropped
            self.load_fov_drop(paths[0])
        else:
            paths = [
                path 
                for path, ext in zip(paths, extensions) 
                if ext == ".hdf5"
            ]
            self.add_multiple(paths)


    def fit_in_view(self, autoscale=False):
        """ Updates scene with all locs shown. """

        movie_height, movie_width = self.movie_size()
        viewport = [(0, 0), (movie_height, movie_width)]
        self.update_scene(viewport=viewport, autoscale=autoscale)

    def move_to_pick(self):
        """ Adjust viewport to show a pick identified by its id. """

        # raise error when no picks found
        if len(self._picks) == 0:
            raise ValueError("No picks detected")

        # get pick id
        pick_no, ok = QtWidgets.QInputDialog.getInt(
                    self, "", "Input pick number: ", 0, 0
                )
        if ok:
            # raise error when pick id too high
            if pick_no >= len(self._picks):
                raise ValueError("Pick number provided too high")
            else: # calculate new viewport
                if self._pick_shape == "Circle":
                    r = (
                        self.window.tools_settings_dialog.pick_diameter.value()
                        / 2
                    )
                    x, y = self._picks[pick_no]
                    x_min = x - 1.4 * r
                    x_max = x + 1.4 * r
                    y_min = y - 1.4 * r
                    y_max = y + 1.4 * r
                else:
                    (xs, ys), (xe, ye) = self._picks[pick_no]
                    xc = np.mean([xs, xe])
                    yc = np.mean([ys, ye])
                    w = self.window.tools_settings_dialog.pick_width.value()
                    X, Y = self.get_pick_rectangle_corners(xs, ys, xe, ye, w)
                    x_min = min(X) - (0.2 * (xc - min(X)))
                    x_max = max(X) + (0.2 * (max(X) - xc))
                    y_min = min(Y) - (0.2 * (yc - min(Y)))
                    y_max = max(Y) + (0.2 * (max(Y) - yc))
                viewport = [(y_min, x_min), (y_max, x_max)] 
                self.update_scene(viewport=viewport)

    def get_channel(self, title="Choose a channel"):
        """ 
        Opens an input dialog to ask for a channel. 
        Returns a channel index or None if no locs loaded.

        Returns
        -------
        None if no locs loaded or channel picked, int otherwise
            Index of the chosen channel
        """

        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            index, ok = QtWidgets.QInputDialog.getItem(
                self, title, "Channel:", pathlist, editable=False
            )
            if ok:
                return pathlist.index(index)
            else:
                return None

    def save_channel(self, title="Choose a channel"):
        """
        Opens an input dialog to ask which channel to save.
        There is an option to save all channels separetely or merge
        them together.

        Returns
        None if no locs found or channel picked, int otherwise
            Index of the chosen channel
        """

        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif n_channels > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Apply to all sequentially")
            pathlist.append("Combine all channels")
            index, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Save localizations",
                "Channel:",
                pathlist,
                editable=False,
            )
            if ok:
                return pathlist.index(index)
            else:
                return None

    def get_channel_all_seq(self, title="Choose a channel"):
        """ 
        Opens an input dialog to ask for a channel. 
        Returns a channel index or None if no locs loaded.
        If apply to all at once is chosen, the index is equal to the
        number of channels loaded.

        Returns
        -------
        None if no locs loaded or channel picked, int otherwise
            Index of the chosen channel
        """

        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Apply to all sequentially")
            index, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Save localizations",
                "Channel:",
                pathlist,
                editable=False,
            )
            if ok:
                return pathlist.index(index)
            else:
                return None        

    def get_channel3d(self, title="Choose a channel"):
        """
        Similar to View.get_channel, used in selecting 3D picks.
        Adds an option to show all channels simultaneously.
        """

        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Show all channels")
            index, ok = QtWidgets.QInputDialog.getItem(
                self, "Select channel", "Channel:", pathlist, editable=False
            )
            if ok:
                return pathlist.index(index)
            else:
                return None

    def get_render_kwargs(self, viewport=None):
        """
        Returns a dictionary to be used for the keyword arguments of 
        render.render.

        Parameters
        ----------
        viewport : list (default=None)
            Specifies the FOV to be rendered. If None, the current 
            viewport is taken.

        Returns
        -------
        dict
            Contains blur method, oversampling, viewport and min blur
            width
        """

        # blur method
        if self._pan: # no blur when panning
            blur_method = None
        else: # selected method
            blur_method = self.window.display_settings_dlg.blur_methods[
                self.window.display_settings_dlg.blur_buttongroup.checkedButton()
            ]

        # oversampling
        optimal_oversampling = (
            self.display_pixels_per_viewport_pixels()
        )
        if self.window.display_settings_dlg.dynamic_disp_px.isChecked():
            oversampling = optimal_oversampling
            self.window.display_settings_dlg.set_disp_px_silently(
                self.window.display_settings_dlg.pixelsize.value()
                / optimal_oversampling 
            )
        else:
            oversampling = float(
                self.window.display_settings_dlg.pixelsize.value()
                / self.window.display_settings_dlg.disp_px_size.value()
            )
            if oversampling > optimal_oversampling:
                QtWidgets.QMessageBox.information(
                    self,
                    "Display pixel size too low",
                    (
                        "Oversampling will be adjusted to"
                        " match the display pixel density."
                    ),
                )
                oversampling = optimal_oversampling
                self.window.display_settings_dlg.set_disp_px_silently(
                    self.window.display_settings_dlg.pixelsize.value()
                    / optimal_oversampling
                )

        # viewport
        if viewport is None:
            viewport = self.viewport

        return {
            "oversampling": oversampling,
            "viewport": viewport,
            "blur_method": blur_method,
            "min_blur_width": float(
                self.window.display_settings_dlg.min_blur_width.value()
            ),
        }

    def load_fov_drop(self, path):
        """
        Checks if path is a fov .txt file (4 coordinates) and loads FOV.

        Parameters
        ----------
        path : str
            Path specifiying .txt file
        """

        try:
            file = np.loadtxt(path)
        except: # not a np array
            return

        if file.shape == (4,):
            (x, y, w, h) = file
            if w > 0 and h > 0:
                viewport = [(y, x), (y + h, x + w)]
                self.update_scene(viewport=viewport)
                self.window.info_dialog.xy_label.setText(
                    "{:.2f} / {:.2f} ".format(x, y)
                )
                self.window.info_dialog.wh_label.setText(
                    "{:.2f} / {:.2f} pixel".format(w, h)
                )

    def load_picks(self, path):
        """ 
        Loads picks from .yaml file. 
        
        Parameters
        ----------
        path : str
            Path specifiying .yaml file

        Raises
        ------
        ValueError
            If .yaml file is not recognized
        """
        
        # load the file
        with open(path, "r") as f:
            regions = yaml.full_load(f)

        # Backwards compatibility for old picked region files
        if "Shape" in regions:
            loaded_shape = regions["Shape"]
        elif "Centers" in regions and "Diameter" in regions:
            loaded_shape = "Circle"
        else:
            raise ValueError("Unrecognized picks file")

        # change pick shape in Tools Settings Dialog
        shape_index = self.window.tools_settings_dialog.pick_shape.findText(
            loaded_shape
        )
        self.window.tools_settings_dialog.pick_shape.setCurrentIndex(
            shape_index
        )

        # assign loaded picks and pick size
        if loaded_shape == "Circle":
            self._picks = regions["Centers"]
            self.window.tools_settings_dialog.pick_diameter.setValue(
                regions["Diameter"]
            )
        elif loaded_shape == "Rectangle":
            self._picks = regions["Center-Axis-Points"]
            self.window.tools_settings_dialog.pick_width.setValue(
                regions["Width"]
            )
        else:
            raise ValueError("Unrecognized pick shape")

        # update Info Dialog
        self.update_pick_info_short()
        self.update_scene(picks_only=True)

    def subtract_picks(self, path):
        """
        Clears current picks that cover the picks loaded from path.

        Parameters
        ----------
        path : str
            Path specifiying .yaml file with picks

        Raises
        ------
        ValueError
            If .yaml file is not recognized
        NotImplementedError
            Rectangular picks have not been implemented yet
        """

        if self._pick_shape == "Rectangle":
            raise NotImplementedError(
                "Subtracting picks not implemented for rectangle picks"
            )
        oldpicks = self._picks.copy()

        # load .yaml
        with open(path, "r") as f:
            regions = yaml.full_load(f)
            self._picks = regions["Centers"]
            diameter = regions["Diameter"]

            # calculate which picks are to stay
            distances = (
                np.sum(
                    (euclidean_distances(oldpicks, self._picks) < diameter / 2)
                    * 1,
                    axis=1,
                )
                >= 1
            )
            filtered_list = [i for (i, v) in zip(oldpicks, distances) if not v]

            self._picks = filtered_list

            self.update_pick_info_short()
            self.window.tools_settings_dialog.pick_diameter.setValue(
                regions["Diameter"]
            )
            self.update_scene(picks_only=True)

    def map_to_movie(self, position):
        """ Converts coordinates from display units to camera units. """

        x_rel = position.x() / self.width()
        x_movie = x_rel * self.viewport_width() + self.viewport[0][1]
        y_rel = position.y() / self.height()
        y_movie = y_rel * self.viewport_height() + self.viewport[0][0]
        return x_movie, y_movie

    def map_to_view(self, x, y):
        """ Converts coordinates from camera units to display units. """

        cx = self.width() * (x - self.viewport[0][1]) / self.viewport_width()
        cy = self.height() * (y - self.viewport[0][0]) / self.viewport_height()
        return int(cx), int(cy)

    def max_movie_height(self):
        """ Returns maximum height of all loaded images. """

        return max(info[0]["Height"] for info in self.infos)

    def max_movie_width(self):
        """ Returns maximum width of all loaded images. """

        return max([info[0]["Width"] for info in self.infos])

    def mouseMoveEvent(self, event):
        """
        Defines actions taken when moving mouse.

        Drawing zoom-in rectangle, panning or drawing a rectangular
        pick.

        Parameters
        ----------
        event : QMouseEvent
        """

        if self._mode == "Zoom":
            # if zooming in
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(
                    QtCore.QRect(self.origin, event.pos())
                )
            # if panning
            if self._pan:
                rel_x_move = (event.x() - self.pan_start_x) / self.width()
                rel_y_move = (event.y() - self.pan_start_y) / self.height()
                self.pan_relative(rel_y_move, rel_x_move)
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
        # if drawing a rectangular pick
        elif self._mode == "Pick":
            if self._pick_shape == "Rectangle":
                if self._rectangle_pick_ongoing:
                    self.rectangle_pick_current_x = event.x()
                    self.rectangle_pick_current_y = event.y()
                    self.update_scene(picks_only=True)

    def mousePressEvent(self, event):
        """
        Defines actions taken when pressing mouse button.

        Start drawing a zoom-in rectangle, start padding, start 
        drawing a pick rectangle.

        Parameters
        ----------
        event : QMouseEvent
        """

        if self._mode == "Zoom":
            # start drawing a zoom-in rectangle
            if event.button() == QtCore.Qt.LeftButton:
                if len(self.locs) > 0: # locs are loaded already
                    if not self.rubberband.isVisible():
                        self.origin = QtCore.QPoint(event.pos())
                        self.rubberband.setGeometry(
                            QtCore.QRect(self.origin, QtCore.QSize())
                        )
                        self.rubberband.show()
            # start panning
            elif event.button() == QtCore.Qt.RightButton:
                self._pan = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
            else:
                event.ignore()
        # start drawing rectangular pick
        elif self._mode == "Pick":
            if event.button() == QtCore.Qt.LeftButton:
                if self._pick_shape == "Rectangle":
                    self._rectangle_pick_ongoing = True
                    self.rectangle_pick_start_x = event.x()
                    self.rectangle_pick_start_y = event.y()
                    self.rectangle_pick_start = self.map_to_movie(event.pos())

    def mouseReleaseEvent(self, event):
        """
        Defines actions taken when releasing mouse button.

        Zoom in, stop panning, add and remove picks, add and remove
        measure points.

        Parameters
        ----------
        event : QMouseEvent
        """

        if self._mode == "Zoom":
            if (
                event.button() == QtCore.Qt.LeftButton
                and self.rubberband.isVisible()
            ): # zoom in if the zoom-in rectangle is visible
                end = QtCore.QPoint(event.pos())
                if end.x() > self.origin.x() and end.y() > self.origin.y():
                    x_min_rel = self.origin.x() / self.width()
                    x_max_rel = end.x() / self.width()
                    y_min_rel = self.origin.y() / self.height()
                    y_max_rel = end.y() / self.height()
                    viewport_height, viewport_width = self.viewport_size()
                    x_min = self.viewport[0][1] + x_min_rel * viewport_width
                    x_max = self.viewport[0][1] + x_max_rel * viewport_width
                    y_min = self.viewport[0][0] + y_min_rel * viewport_height
                    y_max = self.viewport[0][0] + y_max_rel * viewport_height
                    viewport = [(y_min, x_min), (y_max, x_max)]
                    self.update_scene(viewport)
                self.rubberband.hide()
            # stop panning
            elif event.button() == QtCore.Qt.RightButton:
                self._pan = False
                self.setCursor(QtCore.Qt.ArrowCursor)
                event.accept()
                self.update_scene()
            else:
                event.ignore()
        elif self._mode == "Pick":
            if self._pick_shape == "Circle":
                # add pick
                if event.button() == QtCore.Qt.LeftButton:
                    x, y = self.map_to_movie(event.pos())
                    self.add_pick((x, y))
                    event.accept()
                # remove pick
                elif event.button() == QtCore.Qt.RightButton:
                    x, y = self.map_to_movie(event.pos())
                    self.remove_picks((x, y))
                    event.accept()
                else:
                    event.ignore()
            elif self._pick_shape == "Rectangle":
                if event.button() == QtCore.Qt.LeftButton:
                    # finish drawing rectangular pick and add it
                    rectangle_pick_end = self.map_to_movie(event.pos())
                    self._rectangle_pick_ongoing = False
                    self.add_pick(
                        (self.rectangle_pick_start, rectangle_pick_end)
                    )
                    event.accept()
                elif event.button() == QtCore.Qt.RightButton:
                    # remove pick
                    x, y = self.map_to_movie(event.pos())
                    self.remove_picks((x, y))
                    event.accept()
                else:
                    event.ignore()
        elif self._mode == "Measure":
            if event.button() == QtCore.Qt.LeftButton:
                # add measure point
                x, y = self.map_to_movie(event.pos())
                self.add_point((x, y))
                event.accept()
            elif event.button() == QtCore.Qt.RightButton:
                # remove measure points
                x, y = self.map_to_movie(event.pos())
                self.remove_points()
                event.accept()
            else:
                event.ignore()

    def movie_size(self):
        """ Returns tuple with movie height and width. """

        movie_height = self.max_movie_height()
        movie_width = self.max_movie_width()
        return (movie_height, movie_width)

    def nearest_neighbor(self):
        """ Gets channels for nearest neighbor analysis. """

        # choose both channels
        channel1 = self.get_channel("Nearest Neighbor Analysis")
        channel2 = self.get_channel("Nearest Neighbor Analysis")
        self._nearest_neighbor(channel1, channel2)

    def _nearest_neighbor(self, channel1, channel2):
        """
        Calculates and saves distances of the nearest neighbors between
        localizations in channels 1 and 2

        Saves calculated distances in .csv format.

        Parameters
        ----------
        channel1 : int
            Channel to calculate nearest neighbors distances
        channel2 : int
            Second channel to calculate nearest neighbor distances
        """

        # ask how many nearest neighbors
        nn_count, ok = QtWidgets.QInputDialog.getInt(
            self, "Input Dialog", "Number of nearest neighbors: ", 0, 1, 100
        )
        if ok:
            pixelsize = self.window.display_settings_dlg.pixelsize.value()
            # extract x, y and z from both channels 
            x1 = self.locs[channel1].x * pixelsize
            x2 = self.locs[channel2].x * pixelsize
            y1 = self.locs[channel1].y * pixelsize
            y2 = self.locs[channel2].y * pixelsize
            if (
                hasattr(self.locs[channel1], "z")
                and hasattr(self.locs[channel2], "z")
            ):
                z1 = self.locs[channel1].z
                z2 = self.locs[channel2].z
            else: 
                z1 = None
                z2 = None

            # used for avoiding zero distances (to self)
            same_channel = channel1 == channel2

            # get saved file name
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, 
                "Save nearest neighbor distances", 
                self.locs_paths[channel1].replace(".hdf5", "_nn.csv"),
                filter="*.csv",
            )
            nn = postprocess.nn_analysis(
                x1, x2, 
                y1, y2, 
                z1, z2,
                nn_count, 
                same_channel, 
            )
            # save as .csv
            np.savetxt(path, nn, delimiter=',')

    def display_pixels_per_viewport_pixels(self):
        """ Returns optimal oversampling. """

        os_horizontal = self.width() / self.viewport_width()
        os_vertical = self.height() / self.viewport_height()
        # The values are almost the same and we choose max
        return max(os_horizontal, os_vertical)

    def pan_relative(self, dy, dx):
        """ 
        Moves viewport by a given relative distance.

        Parameters
        ----------
        dy : float
            Relative displacement of the viewport in y axis
        dx : float
            Relative displacement of the viewport in x axis
        """

        viewport_height, viewport_width = self.viewport_size()
        x_move = dx * viewport_width
        y_move = dy * viewport_height
        x_min = self.viewport[0][1] - x_move
        x_max = self.viewport[1][1] - x_move
        y_min = self.viewport[0][0] - y_move
        y_max = self.viewport[1][0] - y_move
        viewport = [(y_min, x_min), (y_max, x_max)]
        self.update_scene(viewport)

    @check_pick
    def show_trace(self):
        """ Displays x and y coordinates of locs in picks in time. """

        self.current_trace_x = 0 # used for exporing
        self.current_trace_y = 0

        channel = self.get_channel("Show trace")
        if channel is not None:
            locs = self.picked_locs(channel)
            locs = stack_arrays(locs, asrecarray=True, usemask=False)

            n_frames = self.infos[channel][0]["Frames"]
            xvec = np.arange(n_frames)
            yvec = xvec[:] * 0
            yvec[locs["frame"]] = 1
            self.current_trace_x = xvec
            self.current_trace_y = yvec
            self.channel = channel

            self.canvas = GenericPlotWindow("Trace")

            self.canvas.figure.clear()

            # Three subplots sharing x axes
            ax1, ax2, ax3 = self.canvas.figure.subplots(3, sharex=True)

            # frame vs x
            ax1.scatter(locs["frame"], locs["x"], s=2)
            ax1.set_title("X-pos vs frame")
            ax1.set_xlim(0, n_frames)
            ax1.set_ylabel("X-pos [Px]")

            # frame vs y
            ax2.scatter(locs["frame"], locs["y"], s=2)
            ax2.set_title("Y-pos vs frame")
            ax2.set_ylabel("Y-pos [Px]")

            # locs in time
            ax3.plot(xvec, yvec, linewidth=1)
            ax3.fill_between(xvec, 0, yvec, facecolor="red")
            ax3.set_title("Localizations")
            ax3.set_xlabel("Frames")
            ax3.set_ylabel("ON")
            ax3.set_yticks([0, 1])
            ax3.set_ylim([-0.1, 1.1])

            self.export_trace_button = QtWidgets.QPushButton("Export (*.csv)")
            self.canvas.toolbar.addWidget(self.export_trace_button)
            self.export_trace_button.clicked.connect(self.export_trace)

            self.canvas.canvas.draw()
            self.canvas.show()

    def export_trace(self):
        """ Saves trace as a .csv. """

        trace = np.array([self.current_trace_x, self.current_trace_y])
        base, ext = os.path.splitext(self.locs_paths[self.channel])
        out_path = base + ".trace.txt"

        # get the name for saving
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save trace as txt", out_path, filter="*.trace.txt"
        )

        if path:
            np.savetxt(path, trace, fmt="%i", delimiter=",")

    def pick_message_box(self, params):
        """ 
        Returns a message box for selecting picks. 

        Displays number of picks selected, removed, the ratio and time
        elapsed. Contains 4 buttons for manipulating picks.
        
        Parameters
        ----------
        params : dict
            Stores info about picks selected

        Returns
        -------
        QMessageBox
            With buttons for selecting picks
        """

        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setWindowTitle("Select picks")
        msgBox.setWindowIcon(self.icon)

        if params["i"] == 0:
            keep_ratio = 0
        else:
            keep_ratio = params["n_kept"] / (params["i"])

        dt = time.time() - params["t0"]

        msgBox.setText(
            (
                "Keep pick No: {} of {} ?\n"
                "Picks removed: {} Picks kept: {} Keep Ratio: {:.2f} % \n"
                "Time elapsed: {:.2f} Minutes, "
                " Picks per Minute: {:.2f}"
            ).format(
                params["i"] + 1,
                params["n_total"],
                params["n_removed"],
                params["n_kept"],
                keep_ratio * 100,
                dt / 60,
                params["i"] / dt * 60,
            )
        )


        msgBox.addButton(
            QtWidgets.QPushButton("Accept"), QtWidgets.QMessageBox.YesRole
        ) # keep the pick
        msgBox.addButton(
            QtWidgets.QPushButton("Reject"), QtWidgets.QMessageBox.NoRole
        ) # remove the pick
        msgBox.addButton(
            QtWidgets.QPushButton("Back"), QtWidgets.QMessageBox.ResetRole
        ) # go one pick back
        msgBox.addButton(
            QtWidgets.QPushButton("Cancel"), QtWidgets.QMessageBox.RejectRole
        ) # leave selecting picks

        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        msgBox.move(qr.topLeft())

        return msgBox

    @check_pick
    def select_traces(self):
        """ 
        Lets the user to select picks based on their traces.
        Opens self.pick_message_box to display information.
        """

        removelist = [] # picks to be removed
        channel = self.get_channel("Select traces")

        if channel is not None:
            if self._picks: # if there are picks present
                params = {} # stores info about selecting picks
                params["t0"] = time.time()
                all_picked_locs = self.picked_locs(channel)
                i = 0 # index of the currently shown pick
                n_frames = self.infos[channel][0]["Frames"]
                while i < len(self._picks):
                    fig, (ax1, ax2, ax3) = plt.subplots(
                        3, 1, figsize=(5, 5), constrained_layout=True
                    )
                    fig.canvas.manager.set_window_title("Trace")
                    pick = self._picks[i]
                    locs = all_picked_locs[i]
                    locs = stack_arrays(locs, asrecarray=True, usemask=False)

                    xvec = np.arange(n_frames)
                    yvec = np.ones_like(xvec, dtype=float) * -1
                    yvec[locs["frame"]] = locs["x"]
                    ax1.set_title(
                        "Scatterplot of Pick "
                        + str(i + 1)
                        + "  of: "
                        + str(len(self._picks))
                        + "."
                    )
                    ax1.set_title(
                        "Scatterplot of Pick "
                        + str(i + 1)
                        + "  of: "
                        + str(len(self._picks))
                        + "."
                    )
                    ax1.scatter(xvec, yvec, s=2)
                    ax1.set_ylabel("X-pos [Px]")
                    ax1.set_title("X-pos vs frame")
                    if locs.size:
                        ax1.set_ylim(yvec[yvec>0].min(), yvec.max())
                    plt.setp(ax1.get_xticklabels(), visible=False)

                    yvec = np.ones_like(xvec, dtype=float) * -1
                    yvec[locs["frame"]] = locs["y"]
                    ax2.scatter(xvec, yvec, s=2)
                    ax2.set_title("Y-pos vs frame")
                    ax2.set_ylabel("Y-pos [Px]")
                    if locs.size:
                        ax2.set_ylim(yvec[yvec>0].min(), yvec.max())
                    plt.setp(ax2.get_xticklabels(), visible=False)

                    yvec = xvec[:] * 0
                    yvec[locs["frame"]] = 1
                    ax3.plot(xvec, yvec)
                    ax3.set_title("Localizations")
                    ax3.set_xlabel("Frames")
                    ax3.set_ylabel("ON")
                    ax3.set_yticks([0, 1])

                    fig.canvas.draw()
                    width, height = fig.canvas.get_width_height()

                    # View will display traces instead of rendered locs
                    im = QtGui.QImage(
                        fig.canvas.buffer_rgba(),
                        width,
                        height,
                        QtGui.QImage.Format_ARGB32,
                    )

                    self.setPixmap((QtGui.QPixmap(im)))
                    self.setAlignment(QtCore.Qt.AlignCenter)

                    # update info
                    params["n_removed"] = len(removelist)
                    params["n_kept"] = i - params["n_removed"]
                    params["n_total"] = len(self._picks)
                    params["i"] = i

                    # message box with buttons
                    msgBox = self.pick_message_box(params)

                    reply = msgBox.exec()

                    if reply == 0:
                        # accepted
                        if pick in removelist:
                            removelist.remove(pick)
                    elif reply == 3:
                        # cancel
                        break
                    elif reply == 2:
                        # back
                        if i >= 2:
                            i -= 2
                        else:
                            i = -1
                    else:
                        # discard
                        removelist.append(pick)

                    i += 1
                    plt.close()

        # remove picks 
        for pick in removelist:
            self._picks.remove(pick)

        self.n_picks = len(self._picks)

        self.update_pick_info_short()
        self.update_scene()

    @check_pick
    def show_pick(self):
        """
        Lets user select picks based on their 2D scatter.
        Opens self.pick_message_box to display information.
        """

        if self._pick_shape == "Rectangle":
            raise NotImplementedError(
                "Not implemented for rectangular picks"
            )
        channel = self.get_channel3d("Select Channel")

        removelist = [] # picks to be removed

        if channel is not None:
            n_channels = len(self.locs_paths)
            colors = get_colors(n_channels)
            tools_dialog = self.window.tools_settings_dialog
            r = tools_dialog.pick_diameter.value() / 2
            if channel is (len(self.locs_paths)):
                all_picked_locs = []
                for k in range(len(self.locs_paths)):
                    all_picked_locs.append(self.picked_locs(k))
                if self._picks:
                    params = {} # info about selecting
                    params["t0"] = time.time()
                    i = 0
                    while i < len(self._picks):
                        fig = plt.figure(figsize=(5, 5))
                        fig.canvas.manager.set_window_title(
                            "Scatterplot of Pick"
                        )
                        pick = self._picks[i]

                        # plot scatter
                        ax = fig.add_subplot(111)
                        ax.set_title(
                            "Scatterplot of Pick "
                            + str(i + 1)
                            + "  of: "
                            + str(len(self._picks))
                            + "."
                        )
                        for l in range(len(self.locs_paths)):
                            locs = all_picked_locs[l][i]
                            # locs = stack_arrays(
                            #     locs, asrecarray=True, usemask=False
                            # )
                            ax.scatter(locs["x"], locs["y"], c=colors[l], s=2)

                        # adjust x and y lim
                        x_min = pick[0] - r
                        x_max = pick[0] + r
                        y_min = pick[1] - r
                        y_max = pick[1] + r
                        ax.set_xlabel("X [Px]")
                        ax.set_ylabel("Y [Px]")
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                        plt.axis("equal")

                        fig.canvas.draw()

                        width, height = fig.canvas.get_width_height()

                        # scatter will be displayed instead of 
                        # rendered locs
                        im = QtGui.QImage(
                            fig.canvas.buffer_rgba(),
                            width,
                            height,
                            QtGui.QImage.Format_ARGB32,
                        )

                        self.setPixmap((QtGui.QPixmap(im)))
                        self.setAlignment(QtCore.Qt.AlignCenter)

                        # update selection info
                        params["n_removed"] = len(removelist)
                        params["n_kept"] = i - params["n_removed"]
                        params["n_total"] = len(self._picks)
                        params["i"] = i

                        msgBox = self.pick_message_box(params)

                        reply = msgBox.exec()

                        if reply == 0:
                            # acepted
                            if pick in removelist:
                                removelist.remove(pick)
                        elif reply == 3:
                            # cancel
                            break
                        elif reply == 2:
                            # back
                            if i >= 2:
                                i -= 2
                            else:
                                i = -1
                        else:
                            # discard
                            removelist.append(pick)

                        i += 1
                        plt.close()
            else:
                all_picked_locs = self.picked_locs(channel)
                if self._picks:
                    params = {}
                    params["t0"] = time.time()
                    i = 0
                    while i < len(self._picks):
                        pick = self._picks[i]
                        fig = plt.figure(figsize=(5, 5))
                        fig.canvas.manager.set_window_title(
                            "Scatterplot of Pick"
                        )
                        ax = fig.add_subplot(111)
                        ax.set_title(
                            "Scatterplot of Pick "
                            + str(i + 1)
                            + "  of: "
                            + str(len(self._picks))
                            + "."
                        )
                        locs = all_picked_locs[i]
                        locs = stack_arrays(
                            locs, asrecarray=True, usemask=False
                        )
                        x_min = pick[0] - r
                        x_max = pick[0] + r
                        y_min = pick[1] - r
                        y_max = pick[1] + r
                        ax.scatter(
                            locs["x"], locs["y"], c=colors[channel], s=2
                        )
                        ax.set_xlabel("X [Px]")
                        ax.set_ylabel("Y [Px]")
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                        plt.axis("equal")

                        fig.canvas.draw()
                        width, height = fig.canvas.get_width_height()

                        im = QtGui.QImage(
                            fig.canvas.buffer_rgba(),
                            width,
                            height,
                            QtGui.QImage.Format_ARGB32,
                        )

                        self.setPixmap((QtGui.QPixmap(im)))
                        self.setAlignment(QtCore.Qt.AlignCenter)

                        params["n_removed"] = len(removelist)
                        params["n_kept"] = i - params["n_removed"]
                        params["n_total"] = len(self._picks)
                        params["i"] = i

                        msgBox = self.pick_message_box(params)

                        reply = msgBox.exec()

                        if reply == 0:
                            # accepted
                            if pick in removelist:
                                removelist.remove(pick)
                        elif reply == 3:
                            # cancel
                            break
                        elif reply == 2:
                            # back
                            if i >= 2:
                                i -= 2
                            else:
                                i = -1
                        else:
                            # discard
                            removelist.append(pick)

                        i += 1
                        plt.close()

        for pick in removelist:
            self._picks.remove(pick)

        self.n_picks = len(self._picks)

        self.update_pick_info_short()
        self.update_scene()

    @check_pick
    def show_pick_3d(self):
        """
        Lets user select picks based on their 3D scatter.
        Uses PlotDialog for displaying the scatter.
        """

        channel = self.get_channel3d("Show Pick 3D")
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
        removelist = []
        if channel is not None:
            n_channels = len(self.locs_paths)
            colors = get_colors(n_channels)

            if channel is (len(self.locs_paths)):
                # Combined
                all_picked_locs = []
                for k in range(len(self.locs_paths)):
                    all_picked_locs.append(self.picked_locs(k))

                if self._picks:
                    for i, pick in enumerate(self._picks):
                        reply = PlotDialog.getParams(
                            all_picked_locs, i, len(self._picks), 0, colors
                        )
                        if reply == 1:
                            pass  # accepted
                        elif reply == 2:
                            break
                        else:
                            # discard
                            removelist.append(pick)
            else:
                all_picked_locs = self.picked_locs(channel)
                if self._picks:

                    for i, pick in enumerate(self._picks):

                        reply = PlotDialog.getParams(
                            all_picked_locs, i, len(self._picks), 1, 1
                        )
                        if reply == 1:
                            pass  # accepted
                        elif reply == 2:
                            break
                        else:
                            # discard
                            removelist.append(pick)

        for pick in removelist:
            self._picks.remove(pick)
        self.n_picks = len(self._picks)
        self.update_pick_info_short()
        self.update_scene()

    @check_pick
    def show_pick_3d_iso(self):
        """
        Lets user select picks based on their 3D scatter and 
        projections.
        Uses PlotDialogIso for displaying picks.
        """

        channel = self.get_channel3d("Show Pick 3D")
        removelist = []
        if channel is not None:
            n_channels = len(self.locs_paths)
            colors = get_colors(n_channels)

            if channel is (len(self.locs_paths)):
                # combined
                all_picked_locs = []
                for k in range(len(self.locs_paths)):
                    all_picked_locs.append(self.picked_locs(k))

                if self._picks:
                    for i, pick in enumerate(self._picks):
                        reply = PlotDialogIso.getParams(
                            all_picked_locs,
                            i,
                            len(self._picks),
                            0,
                            colors,
                        )
                        if reply == 1:
                            pass  # accepted

                        elif reply == 2:
                            break
                        else:
                            # discard
                            removelist.append(pick)
            else:
                all_picked_locs = self.picked_locs(channel)
                if self._picks:

                    for i, pick in enumerate(self._picks):

                        reply = PlotDialogIso.getParams(
                            all_picked_locs,
                            i,
                            len(self._picks),
                            1,
                            1,
                        )
                        if reply == 1:
                            pass
                            # accepted
                        elif reply == 2:
                            break
                        else:
                            # discard
                            removelist.append(pick)

        for pick in removelist:
            self._picks.remove(pick)
        self.n_picks = len(self._picks)
        self.update_pick_info_short()
        self.update_scene()

    @check_pick
    def analyze_cluster(self):
        """ Clusters picked locs using k-means clustering. """

        channel = self.get_channel3d("Show Pick 3D")
        removelist = []
        saved_locs = []
        clustered_locs = []
        pixelsize = self.window.display_settings_dlg.pixelsize.value()

        if channel is not None:
            n_channels = len(self.locs_paths)
            colors = get_colors(n_channels)

            # combined locs
            if channel is (len(self.locs_paths)):
                all_picked_locs = []
                for k in range(len(self.locs_paths)):
                    all_picked_locs.append(self.picked_locs(k))

                if self._picks:
                    for i, pick in enumerate(self._picks):
                        # 3D
                        if hasattr(all_picked_locs[0], "z"):
                            # k-means clustering
                            reply = ClsDlg3D.getParams(
                                all_picked_locs,
                                i,
                                len(self._picks),
                                0,
                                colors,
                                pixelsize,
                            )
                        # 2D
                        else:
                            # k-means clustering
                            reply = ClsDlg2D.getParams(
                                all_picked_locs, 
                                i, 
                                len(self._picks), 
                                0, 
                                colors,
                            )
                        if reply == 1:
                            # accepted
                            pass
                        elif reply == 2:
                            # canceled
                            break
                        else:
                            # discard
                            removelist.append(pick)
            # one channel
            else:
                all_picked_locs = self.picked_locs(channel)
                if self._picks:
                    n_clusters, ok = QtWidgets.QInputDialog.getInt(
                        self,
                        "Input Dialog",
                        "Enter inital number of clusters:",
                        10,
                    )

                    for i, pick in enumerate(self._picks):
                        reply = 3
                        while reply == 3:
                            # 3D
                            if hasattr(all_picked_locs[0], "z"):
                                # k-means clustering
                                reply, nc, l_locs, c_locs = ClsDlg3D.getParams(
                                    all_picked_locs,
                                    i,
                                    len(self._picks),
                                    n_clusters,
                                    1,
                                    pixelsize,
                                )
                            # 2D
                            else:
                                # k-means clustering
                                reply, nc, l_locs, c_locs = ClsDlg2D.getParams(
                                    all_picked_locs,
                                    i,
                                    len(self._picks),
                                    n_clusters,
                                    1,
                                )
                            n_clusters = nc

                        if reply == 1:
                            # accepted
                            saved_locs.append(l_locs)
                            clustered_locs.extend(c_locs)
                        elif reply == 2:
                            # canceled
                            break
                        else:
                            # discarded
                            removelist.append(pick)
        
        # saved picked locs
        if saved_locs != []:
            base, ext = os.path.splitext(self.locs_paths[channel])
            out_path = base + "_cluster.hdf5"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save picked localizations", out_path, filter="*.hdf5"
            )
            if path:
                saved_locs = stack_arrays(
                    saved_locs, asrecarray=True, usemask=False
                )
                if saved_locs is not None:
                    d = self.window.tools_settings_dialog.pick_diameter.value()
                    pick_info = {
                        "Generated by:": "Picasso Render",
                        "Pick Diameter:": d,
                    }
                    io.save_locs(
                        path, saved_locs, self.infos[channel] + [pick_info]
                    )

            # save pick properties
            base, ext = os.path.splitext(path)
            out_path = base + "_pickprops.hdf5"
            # TODO: save pick properties
            r_max = 2 * max(
                self.infos[channel][0]["Height"],
                self.infos[channel][0]["Width"],
            )
            max_dark, ok = QtWidgets.QInputDialog.getInt(
                self, "Input Dialog", "Enter gap size:", 3
            )
            out_locs = []
            progress = lib.ProgressDialog(
                "Calculating kinetics", 0, len(clustered_locs), self
            )
            progress.set_value(0)
            dark = np.empty(len(clustered_locs))

            datatype = clustered_locs[0].dtype
            for i, pick_locs in enumerate(clustered_locs):
                if not hasattr(pick_locs, "len"):
                    pick_locs = postprocess.link(
                        pick_locs,
                        self.infos[channel],
                        r_max=r_max,
                        max_dark_time=max_dark,
                    )
                pick_locs = postprocess.compute_dark_times(pick_locs)
                dark[i] = estimate_kinetic_rate(pick_locs.dark)
                out_locs.append(pick_locs)
                progress.set_value(i + 1)
            out_locs = stack_arrays(out_locs, asrecarray=True, usemask=False)
            n_groups = len(clustered_locs)
            progress = lib.ProgressDialog(
                "Calculating pick properties", 0, n_groups, self
            )
            pick_props = postprocess.groupprops(
                out_locs, callback=progress.set_value
            )
            n_units = self.window.info_dialog.calculate_n_units(dark)
            pick_props = lib.append_to_rec(pick_props, n_units, "n_units")
            influx = self.window.info_dialog.influx_rate.value()
            info = self.infos[channel] + [
                {"Generated by": "Picasso: Render", "Influx rate": influx}
            ]
            io.save_datasets(out_path, info, groups=pick_props)

        for pick in removelist:
            self._picks.remove(pick)
        self.n_picks = len(self._picks)
        self.update_pick_info_short()
        self.update_scene()

    @check_picks
    def filter_picks(self):
        """ Filters picks by number of locs. """

        channel = self.get_channel("Filter picks by locs")
        if channel is not None:
            locs = self.all_locs[channel]
            info = self.infos[channel]
            d = self.window.tools_settings_dialog.pick_diameter.value()
            r = d / 2
            # index locs in a grid
            index_blocks = self.get_index_blocks(channel)

            if self._picks:
                removelist = [] # picks to remove
                loccount = [] # n_locs in picks
                progress = lib.ProgressDialog(
                    "Counting in picks..", 0, len(self._picks) - 1, self
                )
                progress.set_value(0)
                progress.show()
                for i, pick in enumerate(self._picks):
                    x, y = pick
                    # extract locs at a given region
                    block_locs = postprocess.get_block_locs_at(
                        x, y, index_blocks
                    )
                    # extract the locs around the pick
                    pick_locs = lib.locs_at(x, y, block_locs, r)
                    locs = stack_arrays(
                        pick_locs, asrecarray=True, usemask=False
                    )
                    loccount.append(len(locs))
                    progress.set_value(i)
                progress.close()

                # plot histogram with n_locs in picks
                fig = plt.figure()
                fig.canvas.manager.set_window_title("Localizations in Picks")
                ax = fig.add_subplot(111)
                ax.set_title("Localizations in Picks ")
                n, bins, patches = ax.hist(
                    loccount, 
                    bins='auto', 
                    density=True, 
                    facecolor="green", 
                    alpha=0.75,
                )
                ax.set_xlabel("Number of localizations")
                ax.set_ylabel("Counts")
                fig.canvas.draw()

                width, height = fig.canvas.get_width_height()

                # display the histogram instead of the rendered locs
                im = QtGui.QImage(
                    fig.canvas.buffer_rgba(),
                    width,
                    height,
                    QtGui.QImage.Format_ARGB32,
                )

                self.setPixmap((QtGui.QPixmap(im)))
                self.setAlignment(QtCore.Qt.AlignCenter)

                # filter picks by n_locs
                minlocs, ok = QtWidgets.QInputDialog.getInt(
                    self,
                    "Input Dialog",
                    "Enter minimum number of localizations:",
                )
                if ok:
                    maxlocs, ok2 = QtWidgets.QInputDialog.getInt(
                        self,
                        "Input Dialog",
                        "Enter maximum number of localizations:",
                        max(loccount),
                        minlocs,
                    )
                    if ok2:
                        progress = lib.ProgressDialog(
                            "Removing picks..", 0, len(self._picks) - 1, self
                        )
                        progress.set_value(0)
                        progress.show()
                        for i, pick in enumerate(self._picks):

                            if loccount[i] > maxlocs:
                                removelist.append(pick)
                            elif loccount[i] < minlocs:
                                removelist.append(pick)
                            progress.set_value(i)

                for pick in removelist:
                    self._picks.remove(pick)
                self.n_picks = len(self._picks)
                self.update_pick_info_short()
                progress.close()
                self.update_scene()

    def rmsd_at_com(self, locs):
        """
        Calculates root mean square displacement at center of mass.
        """

        com_x = locs.x.mean()
        com_y = locs.y.mean()
        return np.sqrt(np.mean((locs.x - com_x) ** 2 + (locs.y - com_y) ** 2))

    def index_locs(self, channel, fast_render=False):
        """
        Indexes localizations from a given channel in a grid with grid 
        size equal to the pick radius.
        """

        if fast_render:
            locs = self.locs[channel]
        else:
            locs = self.all_locs[channel]
        info = self.infos[channel]
        d = self.window.tools_settings_dialog.pick_diameter.value()
        size = d / 2
        status = lib.StatusDialog("Indexing localizations...", self.window)
        index_blocks = postprocess.get_index_blocks(
            locs, info, size
        )
        status.close()
        self.index_blocks[channel] = index_blocks

    def get_index_blocks(self, channel, fast_render=False):
        """
        Calls self.index_locs if not calculated earlier.
        Returns indexed locs from a given channel.
        """

        if self.index_blocks[channel] is None or fast_render:
            self.index_locs(channel, fast_render=fast_render)
        return self.index_blocks[channel]

    @check_picks
    def pick_similar(self):
        """
        Searches picks similar to the current picks.

        Focuses on the number of locs and their root mean square
        displacement from center of mass. Std is defined in Tools
        Settings Dialog.

        Raises
        ------
        NotImplementedError
            If pick shape is rectangle
        """

        if self._pick_shape == "Rectangle":
            raise NotImplementedError(
                "Pick similar not implemented for rectangle picks"
            )
        channel = self.get_channel("Pick similar")
        if channel is not None:
            info = self.infos[channel]
            d = self.window.tools_settings_dialog.pick_diameter.value()
            r = d / 2
            d2 = d ** 2
            std_range = (
                self.window.tools_settings_dialog.pick_similar_range.value()
            )
            # extract n_locs and rmsd from current picks
            index_blocks = self.get_index_blocks(channel)
            n_locs = []
            rmsd = []
            for i, pick in enumerate(self._picks):
                x, y = pick
                block_locs = postprocess.get_block_locs_at(x, y, index_blocks)
                pick_locs = lib.locs_at(x, y, block_locs, r)
                n_locs.append(len(pick_locs))
                rmsd.append(self.rmsd_at_com(pick_locs))

            # calculate min and max n_locs and rmsd for picking similar
            mean_n_locs = np.mean(n_locs)
            mean_rmsd = np.mean(rmsd)
            std_n_locs = np.std(n_locs)
            std_rmsd = np.std(rmsd)
            min_n_locs = max(2, mean_n_locs - std_range * std_n_locs)
            max_n_locs = mean_n_locs + std_range * std_n_locs
            min_rmsd = mean_rmsd - std_range * std_rmsd
            max_rmsd = mean_rmsd + std_range * std_rmsd

            # x, y coordinates of found regions:
            x_similar = np.array([_[0] for _ in self._picks])
            y_similar = np.array([_[1] for _ in self._picks])

            # preparations for grid search
            x_range = np.arange(d / 2, info[0]["Width"], np.sqrt(3) * d / 2)
            y_range_base = np.arange(d / 2, info[0]["Height"] - d / 2, d)
            y_range_shift = y_range_base + d / 2
            locs_temp, size, _, _, block_starts, block_ends, K, L = (
                index_blocks
            )
            locs_x = locs_temp.x
            locs_y = locs_temp.y
            locs_xy = np.stack((locs_x, locs_y))
            x_r = np.uint64(x_range / size)
            y_r1 = np.uint64(y_range_shift / size)
            y_r2 = np.uint64(y_range_base / size)
            status = lib.StatusDialog("Picking similar...", self.window)
            # pick similar
            x_similar, y_similar = postprocess.pick_similar(
                    x_range, y_range_shift, y_range_base,
                    min_n_locs, max_n_locs, min_rmsd, max_rmsd, 
                    x_r, y_r1, y_r2,
                    locs_xy, block_starts, block_ends, K, L,        
                    x_similar, y_similar, r, d2,
                )
            # add picks
            similar = list(zip(x_similar, y_similar))
            self._picks = []
            self.add_picks(similar)
            status.close()

    def picked_locs(
        self, 
        channel, 
        add_group=True, 
        fast_render=False,
    ):
        """ 
        Returns picked localizations in the specified channel. 
        
        Parameters
        ----------
        channel : int
            Channel of locs to be processed
        add_group : boolean (default=True)
            True if group id should be added to locs. Each pick will be
            assigned a different id
        fast_render : boolean
            If True, takes self.locs, i.e. after randomly sampling a 
            fraction of self.all_locs. If False, takes self.all_locs

        Returns
        -------
        list 
            List of np.recarrays, each containing locs from one pick
        """

        if len(self._picks):
            picked_locs = []
            progress = lib.ProgressDialog(
                "Creating localization list", 0, len(self._picks), self
            )
            progress.set_value(0)
            if self._pick_shape == "Circle":
                d = self.window.tools_settings_dialog.pick_diameter.value()
                r = d / 2
                index_blocks = self.get_index_blocks(
                    channel, fast_render=fast_render
                )
                for i, pick in enumerate(self._picks):
                    x, y = pick
                    block_locs = postprocess.get_block_locs_at(
                        x, y, index_blocks
                    )
                    group_locs = lib.locs_at(x, y, block_locs, r)
                    if add_group:
                        group = i * np.ones(len(group_locs), dtype=np.int32)
                        group_locs = lib.append_to_rec(
                            group_locs, group, "group"
                        )
                    group_locs.sort(kind="mergesort", order="frame")
                    picked_locs.append(group_locs)
                    progress.set_value(i + 1)
            elif self._pick_shape == "Rectangle":
                w = self.window.tools_settings_dialog.pick_width.value()
                if fast_render:
                    channel_locs = self.locs[channel]
                else:
                    channel_locs = self.all_locs[channel]
                for i, pick in enumerate(self._picks):
                    (xs, ys), (xe, ye) = pick
                    X, Y = self.get_pick_rectangle_corners(xs, ys, xe, ye, w)
                    x_min = min(X)
                    x_max = max(X)
                    y_min = min(Y)
                    y_max = max(Y)
                    group_locs = channel_locs[channel_locs.x > x_min]
                    group_locs = group_locs[group_locs.x < x_max]
                    group_locs = group_locs[group_locs.y > y_min]
                    group_locs = group_locs[group_locs.y < y_max]
                    group_locs = lib.locs_in_rectangle(group_locs, X, Y)
                    # store rotated coordinates in x_rot and y_rot
                    angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
                    x_shifted = group_locs.x - xs
                    y_shifted = group_locs.y - ys
                    x_pick_rot = x_shifted * np.cos(
                        angle
                    ) - y_shifted * np.sin(angle)
                    y_pick_rot = x_shifted * np.sin(
                        angle
                    ) + y_shifted * np.cos(angle)
                    group_locs = lib.append_to_rec(
                        group_locs, x_pick_rot, "x_pick_rot"
                    )
                    group_locs = lib.append_to_rec(
                        group_locs, y_pick_rot, "y_pick_rot"
                    )
                    if add_group:
                        group = i * np.ones(len(group_locs), dtype=np.int32)
                        group_locs = lib.append_to_rec(
                            group_locs, group, "group"
                        )
                    group_locs.sort(kind="mergesort", order="frame")
                    picked_locs.append(group_locs)
                    progress.set_value(i + 1)

            return picked_locs

    def remove_picks(self, position):
        """
        Deletes picks found at a given position. 

        Parameters
        ----------
        position : tuple
            Specifies x and y coordinates
        """

        x, y = position
        new_picks = [] # picks to be kept
        if self._pick_shape == "Circle":
            pick_diameter_2 = (
                self.window.tools_settings_dialog.pick_diameter.value() ** 2
            )
            for x_, y_ in self._picks:
                d2 = (x - x_) ** 2 + (y - y_) ** 2
                if d2 > pick_diameter_2:
                    new_picks.append((x_, y_))
        elif self._pick_shape == "Rectangle":
            width = self.window.tools_settings_dialog.pick_width.value()
            x = np.array([x])
            y = np.array([y])
            for pick in self._picks:
                (start_x, start_y), (end_x, end_y) = pick
                X, Y = self.get_pick_rectangle_corners(
                    start_x, start_y, end_x, end_y, width
                )
                # do not check if rectangle has no size
                if not Y[0] == Y[1]:
                    if not lib.check_if_in_rectangle(
                        x, y, np.array(X), np.array(Y)
                    )[0]:
                        new_picks.append(pick)

        # delete picks and add new_picks
        self._picks = []
        if len(new_picks) == 0: # no picks left
            self.update_pick_info_short()
            self.update_scene(picks_only=True)
        else:
            self.add_picks(new_picks)

    @check_pick
    def remove_picked_locs(self):
        """ Gets channel for removing picked localizations. """

        channel = self.get_channel_all_seq("Remove picked localizations")
        if channel is len(self.locs_paths): # apply to all channels
            for channel in range(len(self.locs)):
                self._remove_picked_locs(channel)
        elif channel is not None: # apply to a single channel
            self._remove_picked_locs(channel)

    def _remove_picked_locs(self, channel):
        """ 
        Deletes localizations in picks in channel. 

        Temporarily adds index to localizations to compare which
        localizations were picked.

        Parameters
        ----------
        channel : int
            Index of the channel were localizations are removed
        """

        index = np.arange(len(self.all_locs[channel]), dtype=np.int32)
        self.all_locs[channel] = lib.append_to_rec(
            self.all_locs[channel], index, "index"
        ) # used for indexing picked localizations

        # if locs were indexed before, they do not have the index 
        # attribute
        if self._pick_shape == "Circle":
            self.index_locs(channel)
        all_picked_locs = self.picked_locs(channel, add_group=False)
        idx = np.array([], dtype=np.int32)
        for picked_locs in all_picked_locs:
            idx = np.concatenate((idx, picked_locs.index))
        self.all_locs[channel] = np.delete(self.all_locs[channel], idx)
        self.all_locs[channel] = lib.remove_from_rec(
            self.all_locs[channel], "index"
        )
        self.locs[channel] = self.all_locs[channel].copy()
        # fast rendering
        self.window.fast_render_dialog.sample_locs()
        self.update_scene()

    def remove_points(self):
        """ Removes all distance measurement points. """

        self._points = []
        self.update_scene()

    def render_scene(
        self, autoscale=False, use_cache=False, cache=True, viewport=None
    ):
        """
        Returns QImage with rendered localizations.

        Parameters
        ----------
        autoscale : boolean (default=False)
            True if optimally adjust contrast
        use_cache : boolean (default=False)
            True if use stored image
        cache : boolena (default=True)
            True if save image
        viewport : tuple (default=None)
            Viewport to be rendered. If None, takes current viewport

        Returns
        -------
        QImage
            Shows rendered locs; 8 bit
        """

        # get oversampling, blur method, etc
        kwargs = self.get_render_kwargs(viewport=viewport)

        n_channels = len(self.locs)
        # render single or multi channel data
        if n_channels == 1:
            self.render_single_channel(
                kwargs, 
                autoscale=autoscale, 
                use_cache=use_cache, 
                cache=cache,
            )
        else:
            self.render_multi_channel(
                kwargs,
                autoscale=autoscale, 
                use_cache=use_cache,
                cache=cache,
            )
        # add alpha channel (no transparency)
        self._bgra[:, :, 3].fill(255)
        # build QImage
        Y, X = self._bgra.shape[:2]
        qimage = QtGui.QImage(
            self._bgra.data, X, Y, QtGui.QImage.Format_RGB32
        )
        return qimage


    def read_colors(self, n_channels=None):
        """
        Finds currently selected colors for multicolor rendering.

        Parameters
        ----------
        n_channels : int
            Number of channels to be rendered. If None, it is taken
            automatically as the number of locs files loaded.

        Returns
        -------
        list
            List of lists with RGB values from 0 to 1 for each channel.
        """

        if n_channels is None:
            n_channels = len(self.locs)
        colors = get_colors(n_channels) # automatic colors
        # color each channel one by one
        for i in range(len(self.locs)):
            # change colors if not automatic coloring
            if not self.window.dataset_dialog.auto_colors.isChecked():
                # get color from Dataset Dialog
                color = (
                    self.window.dataset_dialog.colorselection[i].currentText()
                )
                # if default color
                if color in self.window.dataset_dialog.default_colors:
                    colors_array = np.array(
                        self.window.dataset_dialog.default_colors, 
                        dtype=object,
                    )
                    index = np.where(colors_array == color)[0][0]
                    # assign color
                    colors[i] = tuple(self.window.dataset_dialog.rgbf[index])
                # if hexadecimal is given
                elif is_hexadecimal(color):
                    colorstring = color.lstrip("#")
                    rgbval = tuple(
                        int(colorstring[i: i + 2], 16) / 255 for i in (0, 2, 4)
                    )
                    # assign color
                    colors[i] = rgbval
                else:
                    warning = (
                        "The color selection not recognnised in the channel "
                        " {}Please choose one of the options provided or "
                        " type the hexadecimal code for your color of choice, "
                        " starting with '#', e.g. '#ffcdff' for pink.".format(
                            self.window.dataset_dialog.checks[i].text()
                        )
                    )
                    QtWidgets.QMessageBox.information(self, "Warning", warning)
                    break

            # reverse colors if white background
            if self.window.dataset_dialog.wbackground.isChecked():
                tempcolor = colors[i]
                inverted = tuple([1 - _ for _ in tempcolor])
                colors[i] = inverted

        # use the gist_rainbow colormap for rendering properties
        if self.x_render_state:
            colors = get_render_properties_colors(n_channels)

        return colors

    def render_multi_channel(
        self,
        kwargs,
        locs=None,
        autoscale=False,
        use_cache=False,
        cache=True,
    ):
        """
        Renders and paints multichannel localizations. 

        Also used when other multi-color data is used (clustered or 
        picked locs, render by property)

        Parameters
        ----------
        kwargs : dict
            Contains blur method, etc. See self.get_render_kwargs
        autoscale : boolean (default=False)
            True if optimally adjust contrast
        locs : np.recarray (default=None)
            Locs to be rendered. If None, self.locs is used
        use_cache : boolean (default=False)
            True if use stored image
        cache : boolena (default=True)
            True if save image     

        Returns
        -------
        np.array
            8 bit array with 4 channels (rgb and alpha)
        """

        # get localizations for rendering
        if locs is None:
            # if slicing is used, locs are indexed and changing slices deletes
            # all localizations
            if self.window.slicer_dialog.slicer_radio_button.isChecked():
                locs = copy.copy(self.locs)
            else:
                locs = self.locs

        # if slicing, show only current slice from every channel
        for i in range(len(locs)):
            if hasattr(locs[i], "z"):
                if self.window.slicer_dialog.slicer_radio_button.isChecked():
                    z_min = self.window.slicer_dialog.slicermin
                    z_max = self.window.slicer_dialog.slicermax
                    in_view = (locs[i].z > z_min) & (locs[i].z <= z_max)
                    locs[i] = locs[i][in_view]
        
        if use_cache: # used saved image
            n_locs = self.n_locs
            image = self.image
        else: # render each channel one by one
            # get image shape (to avoid rendering unchecked channels)
            (y_min, x_min), (y_max, x_max) = kwargs["viewport"]
            X, Y = (
                int(np.ceil(kwargs["oversampling"] * (x_max - x_min))),
                int(np.ceil(kwargs["oversampling"] * (y_max - y_min)))
            )
            # if single channel is rendered
            if len(self.locs) == 1:
                renderings = [render.render(_, **kwargs) for _ in locs]
            else:
                renderings = [
                    render.render(_, **kwargs) 
                    if self.window.dataset_dialog.checks[i].isChecked()
                    else [0, np.zeros((Y, X))]
                    for i, _ in enumerate(locs)
                ] # renders only channels that are checked in dataset dialog
            # renderings = [render.render(_, **kwargs) for _ in locs]
            n_locs = sum([_[0] for _ in renderings])
            image = np.array([_[1] for _ in renderings])

        if cache: # store image
            self.n_locs = n_locs
            self.image = image

        # adjust contrast
        image = self.scale_contrast(image, autoscale=autoscale)

        Y, X = image.shape[1:]
        # array with rgb and alpha channels
        bgra = np.zeros((Y, X, 4), dtype=np.float32)

        colors = self.read_colors(n_channels=len(locs))

        # adjust for relative intensity from Dataset Dialog
        for i in range(len(self.locs)):
            iscale = self.window.dataset_dialog.intensitysettings[i].value()
            image[i] = iscale * image[i]

        # color rgb channels and store in bgra
        for color, image in zip(colors, image):
            bgra[:, :, 0] += color[2] * image
            bgra[:, :, 1] += color[1] * image
            bgra[:, :, 2] += color[0] * image

        bgra = np.minimum(bgra, 1) # minimum value of each pixel is 1
        if self.window.dataset_dialog.wbackground.isChecked():
            bgra = -(bgra - 1)
        self._bgra = self.to_8bit(bgra) # convert to 8 bit 
        return self._bgra

    def render_single_channel(
        self, kwargs, autoscale=False, use_cache=False, cache=True,
    ):
        """
        Renders single channel localizations. 

        Calls render_multi_channel in case of clustered or picked locs,
        rendering by property)

        Parameters
        ----------
        kwargs : dict
            Contains blur method, etc. See self.get_render_kwargs
        autoscale : boolean (default=False)
            True if optimally adjust contrast
        use_cache : boolean (default=False)
            True if use stored image
        cache : boolena (default=True)
            True if save image     

        Returns
        -------
        np.array
            8 bit array with 4 channels (rgb and alpha)
        """

        # get np.recarray
        locs = self.locs[0]

        # if render by property
        if self.x_render_state:
            locs = self.x_locs
            return self.render_multi_channel(
                kwargs, locs=locs, autoscale=autoscale, use_cache=use_cache
            )            

        # if locs have group identity (e.g. clusters)
        if hasattr(locs, "group") and locs.group.size:
            locs = [locs[self.group_color == _] for _ in range(N_GROUP_COLORS)]
            return self.render_multi_channel(
                kwargs, locs=locs, autoscale=autoscale, use_cache=use_cache
            )
        # if slicing, show only the current slice
        if hasattr(locs, "z"):
            if self.window.slicer_dialog.slicer_radio_button.isChecked():
                z_min = self.window.slicer_dialog.slicermin
                z_max = self.window.slicer_dialog.slicermax
                in_view = (locs.z > z_min) & (locs.z <= z_max)
                locs = locs[in_view]

        if use_cache: # use saved image
            n_locs = self.n_locs
            image = self.image
        else: # render locs
            n_locs, image = render.render(locs, **kwargs, info=self.infos[0])
        if cache: # store image
            self.n_locs = n_locs
            self.image = image

        # adjust contrast and convert to 8 bits
        image = self.scale_contrast(image, autoscale=autoscale)
        image = self.to_8bit(image)

        # paint locs using the colormap of choice (Display Settings
        # Dialog)
        cmap = self.window.display_settings_dlg.colormap.currentText()
        if cmap == "Custom":
            cmap = np.uint8(
                np.round(255 * self.custom_cmap)
            )
        else:
            cmap = np.uint8(
                np.round(255 * plt.get_cmap(cmap)(np.arange(256)))
            )

        # return a 4 channel (rgb and alpha) array
        Y, X = image.shape
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]

        # invert colors if white background
        if self.window.dataset_dialog.wbackground.isChecked():
            self._bgra = -(self._bgra - 255)
        return self._bgra

    def resizeEvent(self, event):
        """ Defines what happens when window is resized. """

        self.update_scene()

    def save_picked_locs(self, path, channel):
        """ 
        Saves picked locs from a given channel to path as a .hdf5 file. 
        
        Parameters
        ----------
        path : str 
            Path for saving picked localizations
        channel : int
            Channel of locs to be saved
        """

        # extract picked localizations and stack them
        locs = self.picked_locs(channel, add_group=True)
        locs = stack_arrays(locs, asrecarray=True, usemask=False)

        # save picked locs with .yaml 
        if locs is not None:
            pick_info = {
                "Generated by": "Picasso Render : Pick",
                "Pick Shape": self._pick_shape,
            }
            if self._pick_shape == "Circle":
                d = self.window.tools_settings_dialog.pick_diameter.value()
                pick_info["Pick Diameter"] = d
            elif self._pick_shape == "Rectangle":
                w = self.window.tools_settings_dialog.pick_width.value()
                pick_info["Pick Width"] = w
            io.save_locs(path, locs, self.infos[channel] + [pick_info])

    def save_picked_locs_multi(self, path):
        """ 
        Saves picked locs combined from all channels to path.

        Parameters
        ----------
        path : str
            Path for saving localizations
        """

        # for each channel stack locs from all picks and combine them
        for channel in range(len(self.locs_paths)):
            if channel == 0:
                locs = self.picked_locs(channel)
                locs = stack_arrays(locs, asrecarray=True, usemask=False)
            else:
                templocs = self.picked_locs(channel)
                templocs = stack_arrays(
                    templocs, asrecarray=True, usemask=False
                )
                locs = np.append(locs, templocs)

        # save
        locs = locs.view(np.recarray)
        if locs is not None:
            d = self.window.tools_settings_dialog.pick_diameter.value()
            pick_info = {
                "Generated by:": "Picasso Render : Pick",
                "Pick Shape:": self._pick_shape,
            }
            if self._pick_shape == "Circle":
                d = self.window.tools_settings_dialog.pick_diameter.value()
                pick_info["Pick Diameter"] = d
            elif self._pick_shape == "Rectangle":
                w = self.window.tools_settings_dialog.pick_width.value()
                pick_info["Pick Width"] = w
            io.save_locs(path, locs, self.infos[0] + [pick_info])

    def save_pick_properties(self, path, channel):
        """ 
        Saves picks' properties in a given channel to path.

        Properties include number of locs, mean and std of all locs
        dtypes (x, y, photons, etc) and others.
        
        Parameters
        ----------
        path : str 
            Path for saving picks' properties.
        channel : int
            Channel of locs to be saved.
        """

        picked_locs = self.picked_locs(channel)
        pick_diameter = self.window.tools_settings_dialog.pick_diameter.value()
        r_max = min(pick_diameter, 1)
        max_dark = self.window.info_dialog.max_dark_time.value()
        out_locs = []
        progress = lib.ProgressDialog(
            "Calculating kinetics", 0, len(picked_locs), self
        )
        progress.set_value(0)
        dark = np.empty(len(picked_locs)) # estimated mean dark time
        length = np.empty(len(picked_locs)) # estimated mean bright time
        no_locs = np.empty(len(picked_locs)) # number of locs
        for i, pick_locs in enumerate(picked_locs):
            no_locs[i] = len(pick_locs)
            if no_locs[i] > 0:
                if not hasattr(pick_locs, "len"):
                    pick_locs = postprocess.link(
                        pick_locs,
                        self.infos[channel],
                        r_max=r_max,
                        max_dark_time=max_dark,
                    )
                pick_locs = postprocess.compute_dark_times(pick_locs)
                length[i] = estimate_kinetic_rate(pick_locs.len)
                dark[i] = estimate_kinetic_rate(pick_locs.dark)
                out_locs.append(pick_locs)
            progress.set_value(i + 1)
        out_locs = stack_arrays(out_locs, asrecarray=True, usemask=False)
        n_groups = len(picked_locs)
        progress = lib.ProgressDialog(
            "Calculating pick properties", 0, n_groups, self
        )
        progress.show()
        # get mean and std of each dtype (x, y, photons, etc)
        pick_props = postprocess.groupprops(
            out_locs, callback=progress.set_value
        )
        progress.close()
        # QPAINT estimate of number of binding sites 
        n_units = self.window.info_dialog.calculate_n_units(dark)
        pick_props = lib.append_to_rec(pick_props, n_units, "n_units")
        pick_props = lib.append_to_rec(pick_props, no_locs, "locs")
        pick_props = lib.append_to_rec(pick_props, length, "length_cdf")
        pick_props = lib.append_to_rec(pick_props, dark, "dark_cdf")
        influx = self.window.info_dialog.influx_rate.value()
        info = self.infos[channel] + [
            {"Generated by": "Picasso: Render", "Influx rate": influx}
        ]
        io.save_datasets(path, info, groups=pick_props)

    def save_picks(self, path):
        """
        Saves picked regions in .yaml format to path.

        Parameters
        ----------
        path : str
            Path for saving pick regions
        """

        if self._pick_shape == "Circle":
            d = self.window.tools_settings_dialog.pick_diameter.value()
            picks = {
                "Diameter": float(d),
                "Centers": [[float(_[0]), float(_[1])] for _ in self._picks],
            }
        elif self._pick_shape == "Rectangle":
            w = self.window.tools_settings_dialog.pick_width.value()
            picks = {
                "Width": float(w),
                "Center-Axis-Points": [
                    [
                        [float(s[0]), float(s[1])], 
                        [float(e[0]), float(e[1])],
                    ] for s, e in self._picks
                ],
            }
        picks["Shape"] = self._pick_shape
        with open(path, "w") as f:
            yaml.dump(picks, f)

    def scale_contrast(self, image, autoscale=False):
        """
        Scales image based on contrast values from Display Settings
        Dialog.

        Parameters
        ----------
        image : np.array or list of np.arrays
            Array with rendered locs (grayscale)
        autoscale : boolean (default=False)
            If True, finds optimal contrast

        Returns
        -------
        image : np.array or list of np.arrays
            Scaled image(s)
        """

        if autoscale: # find optimum contrast
            if image.ndim == 2:
                max_ = image.max()
            else:
                max_ = min(
                    [
                        _.max() 
                        for _ in image  # single channel locs with only 
                        if _.max() != 0 # one group have 
                    ]                   # N_GROUP_COLORS - 1 images of 
                )                       # only zeroes
            upper = INITIAL_REL_MAXIMUM * max_
            self.window.display_settings_dlg.silent_minimum_update(0)
            self.window.display_settings_dlg.silent_maximum_update(upper)

        upper = self.window.display_settings_dlg.maximum.value()
        lower = self.window.display_settings_dlg.minimum.value()

        if upper == lower:
            upper = lower + 1 / (10 ** 6)
            self.window.display_settings_dlg.silent_maximum_update(upper)

        image = (image - lower) / (upper - lower)
        image[~np.isfinite(image)] = 0
        image = np.minimum(image, 1.0)
        image = np.maximum(image, 0.0)
        return image

    def show_legend(self):
        """ Displays legend for rendering by property. """

        parameter = self.window.display_settings_dlg.parameter.currentText()
        n_colors = self.window.display_settings_dlg.color_step.value()
        min_val = self.window.display_settings_dlg.minimum_render.value()
        max_val = self.window.display_settings_dlg.maximum_render.value()

        colors = get_render_properties_colors(n_colors)

        fig1 = plt.figure(figsize=(5, 1))

        ax1 = fig1.add_subplot(111, aspect="equal")

        color_spacing = 10 / len(colors)
        xpos = 0
        for i in range(len(colors)):
            ax1.add_patch(
                patches.Rectangle((xpos, 0), color_spacing, 1, color=colors[i])
            )
            xpos += color_spacing

        x = np.arange(0, 11, 2.5)
        ax1.set_xlim([0, 10])
        ax1.get_yaxis().set_visible(False)

        labels = np.linspace(min_val, max_val, 5)
        plt.xticks(x, labels)

        plt.title(parameter)
        fig1.show()

    def activate_render_property(self):
        """ Assigns locs by color to render a chosen property. """

        self.deactivate_property_menu() # blocks changing render parameters

        if self.window.display_settings_dlg.render_check.isChecked():
            self.x_render_state = True
            parameter = (
                self.window.display_settings_dlg.parameter.currentText()
            ) # frame or x or y, etc
            colors = self.window.display_settings_dlg.color_step.value()
            min_val = self.window.display_settings_dlg.minimum_render.value()
            max_val = self.window.display_settings_dlg.maximum_render.value()

            x_step = (max_val - min_val) / colors

            # index each loc according to its parameter's value
            self.x_color = np.floor(
                (self.locs[0][parameter] - min_val) / x_step
            ) 
            # values above and below will be fixed:
            self.x_color[self.x_color < 0] = 0
            self.x_color[self.x_color > colors] = colors

            x_locs = []

            # attempt using cached data
            for cached_entry in self.x_render_cache:
                if cached_entry["parameter"] == parameter:
                    if cached_entry["colors"] == colors:
                        if (cached_entry["min_val"] == min_val) & (
                            cached_entry["max_val"] == max_val
                        ):
                            x_locs = cached_entry["locs"]
                        break

            # if no cached data found
            if x_locs == []:
                pb = lib.ProgressDialog(
                    "Indexing " + parameter, 0, colors, self
                )
                pb.set_value(0)
                # assign locs by color
                for i in range(colors + 1):
                    x_locs.append(self.locs[0][self.x_color == i])
                    pb.set_value(i + 1)
                pb.close()

                # cache
                entry = {}
                entry["parameter"] = parameter
                entry["colors"] = colors
                entry["locs"] = x_locs
                entry["min_val"] = min_val
                entry["max_val"] = max_val

                # Do not store too many datasets in cache
                if len(self.x_render_cache) < 10:
                    self.x_render_cache.append(entry)
                else:
                    self.x_render_cache.insert(0, entry)
                    del self.x_render_cache[-1]

            self.x_locs = x_locs

            self.update_scene()

            self.window.display_settings_dlg.show_legend.setEnabled(True)

        else:
            self.x_render_state = False

        self.activate_property_menu() # allows changing render parameters

    def activate_property_menu(self):
        """ Allows changing render parameters. """

        self.window.display_settings_dlg.minimum_render.setEnabled(True)
        self.window.display_settings_dlg.maximum_render.setEnabled(True)
        self.window.display_settings_dlg.color_step.setEnabled(True)

    def deactivate_property_menu(self):
        """ Blocks changing render parameters. """

        self.window.display_settings_dlg.minimum_render.setEnabled(False)
        self.window.display_settings_dlg.maximum_render.setEnabled(False)
        self.window.display_settings_dlg.color_step.setEnabled(False)

    def set_property(self):
        """ Activates rendering by property. """

        self.window.display_settings_dlg.render_check.setEnabled(False)
        parameter = self.window.display_settings_dlg.parameter.currentText()

        min_val = np.min(self.locs[0][parameter])
        max_val = np.max(self.locs[0][parameter])

        if min_val >= 0:
            lower = 0
        else:
            lower = min_val * 100

        if max_val >= 0:
            upper = max_val * 100
        else:
            upper = -min_val * 100

        self.window.display_settings_dlg.maximum_render.blockSignals(True)
        self.window.display_settings_dlg.minimum_render.blockSignals(True)

        self.window.display_settings_dlg.maximum_render.setRange(lower, upper)
        self.window.display_settings_dlg.maximum_render.setValue(max_val)
        self.window.display_settings_dlg.minimum_render.setValue(min_val)

        self.window.display_settings_dlg.maximum_render.blockSignals(False)
        self.window.display_settings_dlg.minimum_render.blockSignals(False)

        self.activate_property_menu()

        self.window.display_settings_dlg.render_check.setEnabled(True)
        self.window.display_settings_dlg.render_check.setCheckState(False)

        self.activate_render_property()

    def set_mode(self, action):
        """
        Sets self._mode for QMouseEvents.

        Activated when Zoom, Pick or Measure is chosen from Tools menu
        in the main window.

        Parameters
        ----------
        action : QAction
            Action defined in Window.__init__: ("Zoom", "Pick" or 
            "Measure")
        """

        self._mode = action.text()
        self.update_cursor()

    def on_pick_shape_changed(self, pick_shape_index):
        """
        If new shape is chosen, asks user to delete current picks,  
        assigns attributes and updates scene.

        Parameters
        ----------
        pick_shape_index : int
            Index of current ToolsSettingsDialog.pick_shape
        """

        t_dialog = self.window.tools_settings_dialog
        current_text = (
            t_dialog.pick_shape.currentText()
        )
        if current_text == self._pick_shape:
            return
        if len(self._picks):
            qm = QtWidgets.QMessageBox()
            qm.setWindowTitle("Changing pick shape")
            ret = qm.question(
                self,
                "",
                "This action will delete any existing picks. Continue?",
                qm.Yes | qm.No,
            )
            if ret == qm.No:
                shape_index = t_dialog.pick_shape.findText(
                    self._pick_shape
                )
                self.window.tools_settings_dialog.pick_shape.setCurrentIndex(
                    shape_index
                )
                return
        self._pick_shape = current_text
        self._picks = []
        self.update_cursor()
        self.update_scene(picks_only=True)
        self.update_pick_info_short()

    def set_zoom(self, zoom):
        """ 
        Zooms in/out to the given value.
        Called by changing zoom in Display Settings Dialog.

        Parameters
        ----------
        zoom : float
            Value of zoom to change to
        """

        current_zoom = self.display_pixels_per_viewport_pixels()
        self.zoom(current_zoom / zoom)

    def sizeHint(self):
        """ Returns recommended window size. """

        return QtCore.QSize(*self._size_hint)

    def to_8bit(self, image):
        """
        Converts image to 8 bit ready to convert to QImage.

        Parameters
        ----------
        image : np.array
            Image to be converted, with values between 0.0 and 1.0

        Returns
        -------
        np.array
            Image converted to 8 bit
        """

        return np.round(255 * image).astype("uint8")

    def to_left(self):
        """ Called on pressing left arrow; moves FOV. """

        self.pan_relative(0, 0.8)

    def to_right(self):
        """ Called on pressing right arrow; moves FOV. """
        
        self.pan_relative(0, -0.8)

    def to_up(self):
        """ Called on pressing up arrow; moves FOV. """
        
        self.pan_relative(0.8, 0)

    def to_down(self):
        """ Called on pressing down arrow; moves FOV. """
        
        self.pan_relative(-0.8, 0)

    def show_drift(self):
        """ Plots current drift. """

        channel = self.get_channel("Show drift")
        if channel is not None:
            drift = self._drift[channel]

            if drift is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "Driftfile error",
                    (
                        "No driftfile found."
                        "  Nothing to display."
                        "  Please perform drift correction first"
                        " or load a .txt drift file."
                    ),
                )
            else:
                self.plot_window = DriftPlotWindow(self)
                if hasattr(self._drift[channel], "z"):
                    self.plot_window.plot_3d(drift)

                else:
                    self.plot_window.plot_2d(drift)

                self.plot_window.show()


    def undrift(self):
        """ 
        Undrifts with RCC. 

        See Wang Y., et al. Optics Express. 2014
        """

        channel = self.get_channel("Undrift")
        if channel is not None:
            info = self.infos[channel]
            n_frames = info[0]["Frames"]
            # get segmentation (number of frames that are considered 
            # in RCC at once)
            if n_frames < 1000:
                default_segmentation = int(n_frames / 4)
            else:
                default_segmentation = 1000
            segmentation, ok = QtWidgets.QInputDialog.getInt(
                self, "Undrift by RCC", "Segmentation:", default_segmentation
            )

            if ok:
                locs = self.all_locs[channel]
                info = self.infos[channel]
                n_segments = postprocess.n_segments(info, segmentation)
                seg_progress = lib.ProgressDialog(
                    "Generating segments", 0, n_segments, self
                )
                n_pairs = int(n_segments * (n_segments - 1) / 2)
                rcc_progress = lib.ProgressDialog(
                    "Correlating image pairs", 0, n_pairs, self
                )
                try:
                    start_time = time.time()
                    # find drift and apply it to locs
                    drift, _ = postprocess.undrift(
                        locs,
                        info,
                        segmentation,
                        False,
                        seg_progress.set_value,
                        rcc_progress.set_value,
                    )
                    finish_time = time.time()
                    print(
                        "RCC drift estimate running time [seconds]: ", 
                        np.round(finish_time-start_time, 1)
                    )
                    # sanity check and assign attributes
                    locs = lib.ensure_sanity(locs, info)
                    self.all_locs[channel] = locs
                    self.locs[channel] = copy.copy(locs)
                    self.index_blocks[channel] = None
                    self.add_drift(channel, drift)
                    self.update_scene()
                    self.show_drift()

                except Exception as e:
                    QtWidgets.QMessageBox.information(
                        self,
                        "RCC Error",
                        (
                            "RCC failed. \nConsider changing segmentation "
                            "and make sure there are enough locs per frame.\n"
                            "The following exception occured:\n\n {}".format(e)
                        ),
                    )
                    rcc_progress.set_value(n_pairs)
                    self.update_scene()

    @check_picks
    def undrift_from_picked(self):
        """ Gets channel for undrifting from picked locs. """

        channel = self.get_channel("Undrift from picked")
        if channel is not None:
            self._undrift_from_picked(channel)

    @check_picks
    def undrift_from_picked2d(self):
        """ 
        Gets channel for undrifting from picked locs in 2D.
        Available when 3D data is loaded.
        """

        channel = self.get_channel("Undrift from picked")
        if channel is not None:
            self._undrift_from_picked2d(channel)

    def _undrift_from_picked_coordinate(
        self, channel, picked_locs, coordinate
    ):
        """
        Calculates drift in a given coordinate.

        Parameters
        ----------
        channel : int
            Channel where locs are being undrifted
        picked_locs : list
            List of np.recarrays with locs for each pick
        coordinate : str
            Spatial coordinate where drift is to be found

        Returns
        -------
        np.array
            Contains average drift across picks for all frames
        """

        n_picks = len(picked_locs)
        n_frames = self.infos[channel][0]["Frames"]

        # Drift per pick per frame
        drift = np.empty((n_picks, n_frames))
        drift.fill(np.nan)

        # Remove center of mass offset
        for i, locs in enumerate(picked_locs):
            coordinates = getattr(locs, coordinate)
            drift[i, locs.frame] = coordinates - np.mean(coordinates)

        # Mean drift over picks
        drift_mean = np.nanmean(drift, 0)
        # Square deviation of each pick's drift to mean drift along frames
        sd = (drift - drift_mean) ** 2
        # Mean of square deviation for each pick
        msd = np.nanmean(sd, 1)
        # New mean drift over picks
        # where each pick is weighted according to its msd
        nan_mask = np.isnan(drift)
        drift = np.ma.MaskedArray(drift, mask=nan_mask)
        drift_mean = np.ma.average(drift, axis=0, weights=1/msd)
        drift_mean = drift_mean.filled(np.nan)

        # Linear interpolation for frames without localizations
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        nans, nonzero = nan_helper(drift_mean)
        drift_mean[nans] = np.interp(
            nonzero(nans), nonzero(~nans), drift_mean[~nans]
        )

        return drift_mean

    def _undrift_from_picked(self, channel):
        """
        Undrifts based on picked locs in a given channel.
        
        Parameters
        ----------
        channel : int
            Channel to be undrifted
        """

        picked_locs = self.picked_locs(channel)
        status = lib.StatusDialog("Calculating drift...", self)

        drift_x = self._undrift_from_picked_coordinate(
            channel, picked_locs, "x"
        ) # find drift in x
        drift_y = self._undrift_from_picked_coordinate(
            channel, picked_locs, "y"
        ) # find drift in y

        # Apply drift
        self.all_locs[channel].x -= drift_x[self.all_locs[channel].frame]
        self.all_locs[channel].y -= drift_y[self.all_locs[channel].frame]
        self.locs[channel].x -= drift_x[self.locs[channel].frame]
        self.locs[channel].y -= drift_y[self.locs[channel].frame]

        # A rec array to store the applied drift
        drift = (drift_x, drift_y)
        drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])

        # If z coordinate exists, also apply drift there
        if all([hasattr(_, "z") for _ in picked_locs]):
            drift_z = self._undrift_from_picked_coordinate(
                channel, picked_locs, "z"
            )
            self.all_locs[channel].z -= drift_z[self.all_locs[channel].frame]
            self.locs[channel].z -= drift_z[self.locs[channel].frame]
            drift = lib.append_to_rec(drift, drift_z, "z")

        # Cleanup
        self.index_blocks[channel] = None
        self.add_drift(channel, drift)
        status.close()
        self.update_scene()

    def _undrift_from_picked2d(self, channel):
        """
        Undrifts in x and y based on picked locs in a given channel.
        
        Parameters
        ----------
        channel : int
            Channel to be undrifted
        """

        picked_locs = self.picked_locs(channel)
        status = lib.StatusDialog("Calculating drift...", self)

        drift_x = self._undrift_from_picked_coordinate(
            channel, picked_locs, "x"
        )
        drift_y = self._undrift_from_picked_coordinate(
            channel, picked_locs, "y"
        )

        # Apply drift
        self.all_locs[channel].x -= drift_x[self.all_locs[channel].frame]
        self.all_locs[channel].y -= drift_y[self.all_locs[channel].frame]
        self.locs[channel].x -= drift_x[self.locs[channel].frame]
        self.locs[channel].y -= drift_y[self.locs[channel].frame]

        # A rec array to store the applied drift
        drift = (drift_x, drift_y)
        drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])

        # Cleanup
        self.index_blocks[channel] = None
        self.add_drift(channel, drift)
        status.close()
        self.update_scene()

    def undo_drift(self):
        """ Gets channel for undoing drift. """

        channel = self.get_channel("Undo drift")
        if channel is not None:
            self._undo_drift(channel)

    def _undo_drift(self, channel):
        """
        Deletes the latest drift in a given channel.

        Parameters
        ----------
        channel : int
            Channel to undo drift
        """

        drift = self.currentdrift[channel]
        drift.x = -drift.x
        drift.y = -drift.y

        self.all_locs[channel].x -= drift.x[self.all_locs[channel].frame]
        self.all_locs[channel].y -= drift.y[self.all_locs[channel].frame]
        self.locs[channel].x -= drift.x[self.locs[channel].frame]
        self.locs[channel].y -= drift.y[self.locs[channel].frame]

        if hasattr(drift, "z"):
            drift.z = -drift.z
            self.all_locs[channel].z -= drift.z[self.all_locs[channel].frame]
            self.locs[channel].z -= drift.z[self.locs[channel].frame]

        self.add_drift(channel, drift)
        self.update_scene()

    def add_drift(self, channel, drift):
        """
        Assigns attributes and saves .txt drift file

        Parameters
        ----------
        channel : int
            Channel where drift is to be added
        drift : np.recarray
            Contains drift in each coordinate
        """

        timestr = time.strftime("%Y%m%d_%H%M%S")[2:]
        base, ext = os.path.splitext(self.locs_paths[channel])
        driftfile = base + "_" + timestr + "_drift.txt"
        self._driftfiles[channel] = driftfile

        if self._drift[channel] is None:
            self._drift[channel] = drift
        else:
            self._drift[channel].x += drift.x
            self._drift[channel].y += drift.y

            if hasattr(drift, "z"):
                if hasattr(self._drift[channel], "z"):
                    self._drift[channel].z += drift.z
                else:
                    self._drift[channel] = lib.append_to_rec(
                        self._drift[channel], drift.z, "z"
                    )

        self.currentdrift[channel] = copy.copy(drift)

        np.savetxt(
            driftfile,
            self._drift[channel],
            newline="\r\n",
        )

    def apply_drift(self):
        """ 
        Applies drift to locs from a .txt file. 
        
        Assigns attributes and shifts self.locs and self.all_locs.
        """

        channel = self.get_channel("Apply drift")
        if channel is not None: 
            path, exe = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load drift file", filter="*.txt", directory=None
            )
            if path:
                drift = np.loadtxt(path, delimiter=' ')
                if hasattr(self.locs[channel], "z"):
                    drift = (drift[:,0], drift[:,1], drift[:,2])
                    drift = np.rec.array(
                        drift, 
                        dtype=[("x", "f"), ("y", "f"), ("z", "f")],
                    )
                    self.all_locs[channel].x -= drift.x[
                        self.all_locs[channel].frame
                    ]
                    self.all_locs[channel].y -= drift.y[
                        self.all_locs[channel].frame
                    ]
                    self.all_locs[channel].z -= drift.z[
                        self.all_locs[channel].frame
                    ]
                    self.locs[channel].x -= drift.x[
                        self.locs[channel].frame
                    ]
                    self.locs[channel].y -= drift.y[
                        self.locs[channel].frame
                    ]
                    self.locs[channel].z -= drift.z[
                        self.locs[channel].frame
                    ]
                else:
                    drift = (drift[:,0], drift[:,1])
                    drift = np.rec.array(
                        drift, 
                        dtype=[("x", "f"), ("y", "f")],
                    )
                    self.all_locs[channel].x -= drift.x[
                        self.all_locs[channel].frame
                    ]
                    self.all_locs[channel].y -= drift.y[
                        self.all_locs[channel].frame
                    ]
                    self.locs[channel].x -= drift.x[
                        self.locs[channel].frame
                    ]
                    self.locs[channel].y -= drift.y[
                        self.locs[channel].frame
                    ]
                self._drift[channel] = drift
                self._driftfiles[channel] = path
                self.currentdrift[channel] = copy.copy(drift)
                self.index_blocks[channel] = None
                self.update_scene()

    def unfold_groups(self):
        """
        Shifts grouped locs across x axis.

        Useful for locs that were processed with Picasso: Average.
        """
        if len(self.all_locs) > 1:
            raise NotImplementedError(
                "Please load only one channel."
            )

        if not hasattr(self, "unfold_status"):
            self.unfold_status = "folded"
        if self.unfold_status == "folded":
            if hasattr(self.all_locs[0], "group"):
                self.all_locs[0].x += self.all_locs[0].group * 2
                groups = np.unique(self.all_locs[0].group)

                if self._picks:
                    if self._pick_shape == "Rectangle":
                        raise NotImplementedError(
                            "Unfolding not implemented for rectangle picks"
                        )
                    for j in range(len(self._picks)):
                        for i in range(len(groups) - 1):
                            position = self._picks[j][:]
                            positionlist = list(position)
                            positionlist[0] += (i + 1) * 2
                            position = tuple(positionlist)
                            self._picks.append(position)
                # Update width information
                self.oldwidth = self.infos[0][0]["Width"]
                minwidth = np.ceil(
                    np.mean(self.all_locs[0].x)
                    + np.max(self.all_locs[0].x)
                    - np.min(self.all_locs[0].x)
                )
                self.infos[0][0]["Width"] = int(
                    np.max([self.oldwidth, minwidth])
                )
                self.locs[0] = copy.copy(self.all_locs[0])
                self.fit_in_view()
                self.unfold_status = "unfolded"
                self.n_picks = len(self._picks)
                self.update_pick_info_short()
        else:
            self.refold_groups()
            self.clear_picks()

    def unfold_groups_square(self):
        """
        Shifts grouped locs onto a rectangular grid of chosen length.

        Useful for locs that were processed with Picasso: Average.
        """

        if len(self.all_locs) > 1:
            raise NotImplementedError(
                "Please load only one channel."
            )

        n_square, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Input Dialog",
            "Set number of elements per row and column:",
            100,
        )
        spacing, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Input Dialog",
            "Set distance between elements:",
            2,
        )
        if hasattr(self.all_locs[0], "group"):

            self.all_locs[0].x += (
                np.mod(self.all_locs[0].group, n_square) 
                * spacing
            )
            self.all_locs[0].y += (
                np.floor(self.all_locs[0].group / n_square) 
                * spacing
            )

            mean_x = np.mean(self.locs[0].x)
            mean_y = np.mean(self.locs[0].y)

            self.all_locs[0].x -= mean_x
            self.all_locs[0].y -= np.mean(self.all_locs[0].y)

            offset_x = np.absolute(np.min(self.all_locs[0].x))
            offset_y = np.absolute(np.min(self.all_locs[0].y))

            self.all_locs[0].x += offset_x
            self.all_locs[0].y += offset_y

            if self._picks:
                if self._pick_shape == "Rectangle":
                    raise NotImplementedError(
                        "Not implemented for rectangle picks"
                    )
                # Also unfold picks
                groups = np.unique(self.all_locs[0].group)

                shift_x = (
                    np.mod(groups, n_square) 
                    * spacing 
                    - mean_x 
                    + offset_x
                )
                shift_y = (
                    np.floor(groups / n_square) 
                    * spacing 
                    - mean_y 
                    + offset_y
                )

                for j in range(len(self._picks)):
                    for k in range(len(groups)):
                        x_pick, y_pick = self._picks[j]
                        self._picks.append(
                            (x_pick + shift_x[k], y_pick + shift_y[k])
                        )

                self.n_picks = len(self._picks)
                self.update_pick_info_short()

        # Update width information
        self.infos[0][0]["Height"] = int(np.ceil(np.max(self.all_locs[0].y)))
        self.infos[0][0]["Width"] = int(np.ceil(np.max(self.all_locs[0].x)))
        self.locs[0] = copy.copy(self.all_locs[0])
        self.fit_in_view()

    def refold_groups(self):
        """ Refolds grouped locs across x axis. """

        if len(self.all_locs) > 1:
            raise NotImplementedError(
                "Please load only one channel."
            )

        if hasattr(self.all_locs[0], "group"):
            self.all_locs[0].x -= self.all_locs[0].group * 2
        self.locs[0] = copy.copy(self.all_locs[0])
        self.fit_in_view()
        self.infos[0][0]["Width"] = self.oldwidth
        self.unfold_status == "folded"

    def update_cursor(self):
        """ Changes cursor according to self._mode. """

        if self._mode == "Zoom" or self._mode == "Measure":
            self.unsetCursor() # normal cursor
        elif self._mode == "Pick":
            if self._pick_shape == "Circle": # circle
                diameter = (
                    self.window.tools_settings_dialog.pick_diameter.value()
                )
                diameter = int(self.width() * diameter / self.viewport_width())
                # remote desktop crashes sometimes for high diameter
                if diameter < 100:
                    pixmap_size = ceil(diameter) + 1
                    pixmap = QtGui.QPixmap(pixmap_size, pixmap_size)
                    pixmap.fill(QtCore.Qt.transparent)
                    painter = QtGui.QPainter(pixmap)
                    painter.setPen(QtGui.QColor("white"))
                    if self.window.dataset_dialog.wbackground.isChecked():
                        painter.setPen(QtGui.QColor("black"))
                    offset = int((pixmap_size - diameter) / 2)
                    painter.drawEllipse(offset, offset, diameter, diameter)
                    painter.end()
                    cursor = QtGui.QCursor(pixmap)
                    self.setCursor(cursor)
                else:
                    self.unsetCursor()
            elif self._pick_shape == "Rectangle":
                self.unsetCursor()

    def update_pick_info_long(self):
        """ Called when evaluating picks statistics in Info Dialog. """

        if len(self._picks) == 0:
            warning = "No picks found. Please pick first."
            QtWidgets.QMessageBox.information(self, "Warning", warning)
            return

        if self._pick_shape == "Rectangle":
            warning = "Not supported for rectangular picks."
            QtWidgets.QMessageBox.information(self, "Warning", warning)
            return

        channel = self.get_channel("Calculate pick info")
        if channel is not None:
            d = self.window.tools_settings_dialog.pick_diameter.value()
            t = self.window.info_dialog.max_dark_time.value()
            r_max = min(d, 1)
            info = self.infos[channel]
            picked_locs = self.picked_locs(channel)
            n_picks = len(picked_locs)
            N = np.empty(n_picks) # number of locs per pick
            rmsd = np.empty(n_picks) # rmsd in each pick
            length = np.empty(n_picks) # estimated mean bright time 
            dark = np.empty(n_picks) # estimated mean dark time
            has_z = hasattr(picked_locs[0], "z")
            if has_z:
                rmsd_z = np.empty(n_picks)
            new_locs = [] # linked locs in each pick
            progress = lib.ProgressDialog(
                "Calculating pick statistics", 0, len(picked_locs), self
            )
            progress.set_value(0)
            for i, locs in enumerate(picked_locs):
                if len(locs) > 0:
                    N[i] = len(locs)
                    com_x = np.mean(locs.x)
                    com_y = np.mean(locs.y)
                    rmsd[i] = np.sqrt(
                        np.mean((locs.x - com_x) ** 2 + (locs.y - com_y) ** 2)
                    )
                    if has_z:
                        rmsd_z[i] = np.sqrt(
                            np.mean((locs.z - np.mean(locs.z)) ** 2)
                        )
                    if not hasattr(locs, "len"):
                        locs = postprocess.link(
                            locs, info, r_max=r_max, max_dark_time=t
                        )
                    locs = postprocess.compute_dark_times(locs)
                    length[i] = estimate_kinetic_rate(locs.len)
                    dark[i] = estimate_kinetic_rate(locs.dark)
                    new_locs.append(locs)
                else:
                    self.remove_picks(self._picks[i])
                progress.set_value(i + 1)

            # update labels in info dialog
            self.window.info_dialog.n_localizations_mean.setText(
                "{:.2f}".format(np.nanmean(N))
            ) # mean number of locs per pick
            self.window.info_dialog.n_localizations_std.setText(
                "{:.2f}".format(np.nanstd(N))
            ) # std number of locs per pick
            self.window.info_dialog.rmsd_mean.setText(
                "{:.2}".format(np.nanmean(rmsd))
            ) # mean rmsd per pick
            self.window.info_dialog.rmsd_std.setText(
                "{:.2}".format(np.nanstd(rmsd))
            ) # std rmsd per pick
            if has_z:
                self.window.info_dialog.rmsd_z_mean.setText(
                    "{:.2f}".format(np.nanmean(rmsd_z))
                ) # mean rmsd in z per pick
                self.window.info_dialog.rmsd_z_std.setText(
                    "{:.2f}".format(np.nanstd(rmsd_z))
                ) # std rmsd in z per pick
            pooled_locs = stack_arrays(
                new_locs, usemask=False, asrecarray=True
            )
            fit_result_len = fit_cum_exp(pooled_locs.len)
            fit_result_dark = fit_cum_exp(pooled_locs.dark)
            self.window.info_dialog.length_mean.setText(
                "{:.2f}".format(np.nanmean(length))
            ) # mean bright time
            self.window.info_dialog.length_std.setText(
                "{:.2f}".format(np.nanstd(length))
            ) # std bright time
            self.window.info_dialog.dark_mean.setText(
                "{:.2f}".format(np.nanmean(dark))
            ) # mean dark time
            self.window.info_dialog.dark_std.setText(
                "{:.2f}".format(np.nanstd(dark))
            ) # std dark time
            self.window.info_dialog.pick_info = {
                "pooled dark": estimate_kinetic_rate(pooled_locs.dark),
                "length": length,
                "dark": dark,
            }
            self.window.info_dialog.update_n_units()
            self.window.info_dialog.pick_hist_window.plot(
                pooled_locs, fit_result_len, fit_result_dark
            )

    def update_pick_info_short(self):
        """ Updates number of picks in Info Dialog. """

        self.window.info_dialog.n_picks.setText(str(len(self._picks)))

    def update_scene(
        self,
        viewport=None,
        autoscale=False,
        use_cache=False,
        picks_only=False,
    ):
        """
        Updates the view of rendered locs as well as cursor.

        Parameters
        ----------
        viewport : tuple (default=None)
            Viewport to be rendered. If None self.viewport is taken
        autoscale : boolean (default=False)
            True if optimally adjust contrast
        use_cache : boolean (default=False)
            True if use stored image
        cache : boolena (default=True)
            True if save image
        """

        # Clear slicer cache
        self.window.slicer_dialog.slicer_cache = {}
        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene(
                viewport,
                autoscale=autoscale,
                use_cache=use_cache,
                picks_only=picks_only,
            )
            self.update_cursor()

    def update_scene_slicer(
        self,
        viewport=None,
        autoscale=False,
        use_cache=False,
        picks_only=False,
    ):
        """
        Updates the view of rendered locs when they are sliced.

        Parameters
        ----------
        viewport : tuple (default=None)
            Viewport to be rendered. If None self.viewport is taken
        autoscale : boolean (default=False)
            True if optimally adjust contrast
        use_cache : boolean (default=False)
            True if use stored image
        cache : boolena (default=True)
            True if save image
        """

        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene_slicer(
                viewport,
                autoscale=autoscale,
                use_cache=use_cache,
                picks_only=picks_only,
            )
            self.update_cursor()

    def viewport_center(self, viewport=None):
        """
        Finds viewport's center (pixels).

        Parameters
        ----------
        viewport: tuple (default=None)
            Viewport to be evaluated. If None self.viewport is taken

        Returns
        tuple
            Contains x and y coordinates of viewport's center (pixels)
        """

        if viewport is None:
            viewport = self.viewport
        return (
            ((viewport[1][0] + viewport[0][0]) / 2),
            ((viewport[1][1] + viewport[0][1]) / 2),
        )

    def viewport_height(self, viewport=None):
        """
        Finds viewport's height.

        Parameters
        ----------
        viewport: tuple (default=None)
            Viewport to be evaluated. If None self.viewport is taken

        Returns
        float
            Viewport's height (pixels)
        """

        if viewport is None:
            viewport = self.viewport
        return viewport[1][0] - viewport[0][0]

    def viewport_size(self, viewport=None):
        """
        Finds viewport's height and width.

        Parameters
        ----------
        viewport: tuple (default=None)
            Viewport to be evaluated. If None self.viewport is taken

        Returns
        tuple
            Viewport's height and width (pixels)
        """
        
        if viewport is None:
            viewport = self.viewport
        return self.viewport_height(viewport), self.viewport_width(viewport)

    def viewport_width(self, viewport=None):
        """
        Finds viewport's width.

        Parameters
        ----------
        viewport: tuple (default=None)
            Viewport to be evaluated. If None self.viewport is taken

        Returns
        float
            Viewport's width (pixels)
        """
        
        if viewport is None:
            viewport = self.viewport
        return viewport[1][1] - viewport[0][1]

    def relative_position(self, viewport_center, cursor_position):
        """ 
        Finds the position of the cursor relative to the viewport's
        center.

        Parameters
        ----------
        viewport_center : tuple
            Specifies the position of viewport's center
        cursor_position : tuple
            Specifies the position of the cursor

        Returns
        -------
        tuple
            Current cursor's position with respect to viewport's 
            center
        """

        rel_pos_x = (
            (cursor_position[0] - viewport_center[1])
            / self.viewport_width()
        )
        rel_pos_y = (
            (cursor_position[1] - viewport_center[0])
            / self.viewport_height()
        )
        return rel_pos_x, rel_pos_y

    def zoom(self, factor, cursor_position=None):
        """
        Changes zoom relatively to factor.

        If zooms via wheelEvent, zooming is centered around cursor's
        position.

        Parameters
        ----------
        factor : float
            Relative zoom magnitude
        cursor_position : tuple (default=None)
            Cursor's position on the screen. If None, zooming is 
            centered around viewport's center
        """

        viewport_height, viewport_width = self.viewport_size()
        new_viewport_height = viewport_height * factor
        new_viewport_width = viewport_width * factor

        if cursor_position is not None: # wheelEvent
            old_viewport_center = self.viewport_center()
            rel_pos_x, rel_pos_y = self.relative_position(
                old_viewport_center, cursor_position
            ) #this stays constant before and after zooming
            new_viewport_center_x = (
                cursor_position[0] - rel_pos_x * new_viewport_width
            )
            new_viewport_center_y = (
                cursor_position[1] - rel_pos_y * new_viewport_height
            )
        else:
            new_viewport_center_y, new_viewport_center_x = (
                self.viewport_center()
            )

        new_viewport = [
            (
                new_viewport_center_y - new_viewport_height/2,
                new_viewport_center_x - new_viewport_width/2,
            ),
            (
                new_viewport_center_y + new_viewport_height/2,
                new_viewport_center_x + new_viewport_width/2,
            ),
        ]
        self.update_scene(new_viewport)

    def zoom_in(self):
        """ Zooms in by a constant factor. """

        self.zoom(1 / ZOOM)

    def zoom_out(self):
        """ Zooms out by a constant factor. """

        self.zoom(ZOOM)

    def wheelEvent(self, QWheelEvent):
        """
        Defines what happens when mouse wheel is used.

        Press Ctrl/Command to zoom in/out.
        """
        
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            direction = QWheelEvent.angleDelta().y()
            position = self.map_to_movie(QWheelEvent.pos())
            if direction > 0:
                self.zoom(1 / ZOOM, cursor_position=position)
            else:
                self.zoom(ZOOM, cursor_position=position)


class Window(QtWidgets.QMainWindow):
    """
    Main Picasso: Render window class.

    ...

    Attributes
    ----------
    actions_3d : list
        specifies actions that are displayed for 3D data only
    dataset_dialog : DatasetDialog
        instance of the dialog for multichannel display
    dialogs : list
        Contains all dialogs that are closed when reseting Render
    display_settings_dlg : DisplaySettingsDialog
        instance of the dialog for display settings
    info_dialog : InfoDialog
        instance of the dialog storing information about data and picks
    fast_render_dialog: FastRenderDialog
        instance of the dialog for sampling a fraction of locs to speed
        up rendering
    mask_settings_dialog : MaskSettingsDialog
        isntance of the dialog for masking image
    menu_bar : QMenuBar
        menu bar with menus: File, View, Tools, Postprocess
    menus : list 
        contains View, Tools and Postprocess menus
    plugins : list
        contains plugins loaded from picasso/gui/plugins
    slicer_dialog : SlicerDialog
        instance of the dialog for slicing 3D data in z axis
    tools_settings_dialog : ToolsSettingsDialog
        instance of the dialog for customising picks
    view : View
        instance of the class for displaying rendered localizations
    window_rot : RotationWindow
        instance of the class for displaying 3D data with rotation
    x_spiral : np.array
        x coordinates before the last spiral action in ApplyDialog
    y_spiral : np.array
        y coordinates before the last spiral action in ApplyDialog

    Methods
    -------
    closeEvent(event)
        Changes user settings and closes all dialogs
    export_complete()
        Exports the whole field of view as .png or .tif
    export_current()
        Exports current view as .png or .tif
    export_current_info()
        Exports info about the current view in .yaml file
    export_multi()
        Asks the user to choose a type of export
    export_fov_ims()
        Exports current FOV to .ims
    export_ts()
        Exports locs as .csv for ThunderSTORM
    export_txt()
        Exports locs as .txt for ImageJ
    export_txt_imaris()
        Exports locs as .txt for IMARIS
    export_txt_nis()
        Exports locs as .txt for NIS
    export_xyz_chimera()
        Exports locs as .xyz for CHIMERA
    export_3d_visp()
        Exports locs as .3d for ViSP
    initUI(plugins_loaded)
        Initializes the main window
    load_picks()
        Loads picks from a .yaml file
    load_user_settings()
        Loads colormap and current directory
    open_apply_dialog()
        Loads expression and applies it to locs
    open_file_dialog()
        Opens localizaitons .hdf5 file(s)
    open_rotated_locs()
        Opens rotated localizations .hdf5 file(s)
    remove_group()
        Displayed locs will have no group information
    resizeEvent(event)
        Updates window size when resizing
    save_locs()
        Saves localizations in a given channel (or all channels)
    save_picked_locs()
        Saves picked localizations in a given channel (or all channels)
    save_picks()
        Saves picks as .yaml
    save_pick_properties()
        Saves pick properties in a given channel
    subtract_picks()
        Subtracts picks from a .yaml file
    remove_locs()
        Resets Window
    rot_win()
        Opens/updates RotationWindow
    update_info()
        Updates Window's size and median loc prec in InfoDialog
    """

    def __init__(self, plugins_loaded=False):
        super().__init__()
        self.initUI(plugins_loaded)

    def initUI(self, plugins_loaded):
        """
        Initializes the main window. Builds dialogs and menu bar.

        Parameters
        ----------
        plugins_loaded : boolean
            If True, plugins have been loaded before.
        """

        # general
        self.setWindowTitle(f"Picasso v{__version__}: Render")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setWindowIcon(icon)
        self.view = View(self) # displays rendered locs
        self.view.setMinimumSize(1, 1)
        self.setCentralWidget(self.view)

        # set up dialogs
        self.display_settings_dlg = DisplaySettingsDialog(self)
        self.tools_settings_dialog = ToolsSettingsDialog(self)
        self.view._pick_shape = (
            self.tools_settings_dialog.pick_shape.currentText()
        )
        self.tools_settings_dialog.pick_shape.currentIndexChanged.connect(
            self.view.on_pick_shape_changed
        )
        self.mask_settings_dialog = MaskSettingsDialog(self)
        self.slicer_dialog = SlicerDialog(self)
        self.info_dialog = InfoDialog(self)
        self.dataset_dialog = DatasetDialog(self)  
        self.fast_render_dialog = FastRenderDialog(self)
        self.window_rot = RotationWindow(self)
        self.test_clusterer_dialog = TestClustererDialog(self)

        self.dialogs = [
            self.display_settings_dlg,
            self.dataset_dialog,
            self.info_dialog,
            self.info_dialog.change_fov,
            self.mask_settings_dialog,
            self.tools_settings_dialog,
            self.slicer_dialog,
            self.window_rot,
            self.fast_render_dialog,
            self.test_clusterer_dialog,
        ]
        
        # menu bar
        self.menu_bar = self.menuBar()

        # menu bar - File
        file_menu = self.menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        open_rot_action = file_menu.addAction("Open rotated localizations")
        open_rot_action.setShortcut("Ctrl+Shift+O")
        open_rot_action.triggered.connect(self.open_rotated_locs)
        save_action = file_menu.addAction("Save localizations")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs)
        save_picked_action = file_menu.addAction("Save picked localizations")
        save_picked_action.setShortcut("Ctrl+Shift+S")
        save_picked_action.triggered.connect(self.save_picked_locs)
        save_pick_properties_action = file_menu.addAction(
            "Save pick properties"
        )
        save_pick_properties_action.triggered.connect(
            self.save_pick_properties
        )
        save_picks_action = file_menu.addAction("Save pick regions")
        save_picks_action.triggered.connect(self.save_picks)
        load_picks_action = file_menu.addAction("Load pick regions")
        load_picks_action.triggered.connect(self.load_picks)

        file_menu.addSeparator()
        export_current_action = file_menu.addAction("Export current view")
        export_current_action.setShortcut("Ctrl+E")
        export_current_action.triggered.connect(self.export_current)
        export_complete_action = file_menu.addAction("Export complete image")
        export_complete_action.setShortcut("Ctrl+Shift+E")
        export_complete_action.triggered.connect(self.export_complete)

        file_menu.addSeparator()
        export_multi_action = file_menu.addAction("Export localizations")
        export_multi_action.triggered.connect(self.export_multi)
        if IMSWRITER:
            export_ims_action = file_menu.addAction("Export ROI for Imaris")
            export_ims_action.triggered.connect(self.export_fov_ims)

        file_menu.addSeparator()
        delete_action = file_menu.addAction("Remove all localizations")
        delete_action.triggered.connect(self.remove_locs)

        # menu bar - View
        view_menu = self.menu_bar.addMenu("View")
        display_settings_action = view_menu.addAction("Display settings")
        display_settings_action.setShortcut("Ctrl+D")
        display_settings_action.triggered.connect(
            self.display_settings_dlg.show
        )
        view_menu.addAction(display_settings_action)
        dataset_action = view_menu.addAction("Files")
        dataset_action.setShortcut("Ctrl+F")
        dataset_action.triggered.connect(self.dataset_dialog.show)

        view_menu.addSeparator()
        to_left_action = view_menu.addAction("Left")
        to_left_action.setShortcuts(["Left", "A"])
        to_left_action.triggered.connect(self.view.to_left)
        to_right_action = view_menu.addAction("Right")
        to_right_action.setShortcuts(["Right", "D"])
        to_right_action.triggered.connect(self.view.to_right)
        to_up_action = view_menu.addAction("Up")
        to_up_action.setShortcuts(["Up", "W"])
        to_up_action.triggered.connect(self.view.to_up)
        to_down_action = view_menu.addAction("Down")
        to_down_action.setShortcuts(["Down", "S"])
        to_down_action.triggered.connect(self.view.to_down)

        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction("Zoom in")
        zoom_in_action.setShortcuts(["Ctrl++", "Ctrl+="])
        zoom_in_action.triggered.connect(self.view.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction("Zoom out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.view.zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction("Fit image to window")
        fit_in_view_action.setShortcut("Ctrl+W")
        fit_in_view_action.triggered.connect(self.view.fit_in_view)
        view_menu.addAction(fit_in_view_action)

        view_menu.addSeparator()
        info_action = view_menu.addAction("Show info")
        info_action.setShortcut("Ctrl+I")
        info_action.triggered.connect(self.info_dialog.show)
        view_menu.addAction(info_action)
        slicer_action = view_menu.addAction("Slice")
        slicer_action.triggered.connect(self.slicer_dialog.initialize)
        rot_win_action = view_menu.addAction("Update rotation window")
        rot_win_action.setShortcut("Ctrl+Shift+R")
        rot_win_action.triggered.connect(self.rot_win)

        # menu bar - Tools
        tools_menu = self.menu_bar.addMenu("Tools")
        tools_actiongroup = QtWidgets.QActionGroup(self.menu_bar)
        zoom_tool_action = tools_actiongroup.addAction(
            QtWidgets.QAction("Zoom", tools_menu, checkable=True)
        )
        zoom_tool_action.setShortcut("Ctrl+Z")
        tools_menu.addAction(zoom_tool_action)
        zoom_tool_action.setChecked(True)
        pick_tool_action = tools_actiongroup.addAction(
            QtWidgets.QAction("Pick", tools_menu, checkable=True)
        )
        pick_tool_action.setShortcut("Ctrl+P")
        tools_menu.addAction(pick_tool_action)
        measure_tool_action = tools_actiongroup.addAction(
            QtWidgets.QAction("Measure", tools_menu, checkable=True)
        )
        measure_tool_action.setShortcut("Ctrl+M")
        tools_menu.addAction(measure_tool_action)
        tools_actiongroup.triggered.connect(self.view.set_mode)

        tools_menu.addSeparator()
        tools_settings_action = tools_menu.addAction("Tools settings")
        tools_settings_action.setShortcut("Ctrl+T")
        tools_settings_action.triggered.connect(
            self.tools_settings_dialog.show
        )

        pick_similar_action = tools_menu.addAction("Pick similar")
        pick_similar_action.setShortcut("Ctrl+Shift+P")
        pick_similar_action.triggered.connect(self.view.pick_similar)

        clear_picks_action = tools_menu.addAction("Clear picks")
        clear_picks_action.triggered.connect(self.view.clear_picks)
        clear_picks_action.setShortcut("Ctrl+C")

        remove_locs_picks_action = tools_menu.addAction(
            "Remove localizations in picks"
        )
        remove_locs_picks_action.triggered.connect(
            self.view.remove_picked_locs
        )

        move_to_pick_action = tools_menu.addAction("Move to pick")
        move_to_pick_action.triggered.connect(self.view.move_to_pick)

        tools_menu.addSeparator()
        show_trace_action = tools_menu.addAction("Show trace")
        show_trace_action.setShortcut("Ctrl+R")
        show_trace_action.triggered.connect(self.view.show_trace)

        tools_menu.addSeparator()
        select_traces_action = tools_menu.addAction("Select picks (trace)")
        select_traces_action.triggered.connect(self.view.select_traces)

        plotpick_action = tools_menu.addAction("Select picks (XY scatter)")
        plotpick_action.triggered.connect(self.view.show_pick)
        plotpick3d_action = tools_menu.addAction("Select picks (XYZ scatter)")
        plotpick3d_action.triggered.connect(self.view.show_pick_3d)
        plotpick3d_iso_action = tools_menu.addAction(
            "Select picks (XYZ scatter, 4 panels)"
        )
        plotpick3d_iso_action.triggered.connect(self.view.show_pick_3d_iso)

        filter_picks_action = tools_menu.addAction(
            "Filter picks by number of locs"
        )
        filter_picks_action.triggered.connect(self.view.filter_picks)

        pickadd_action = tools_menu.addAction("Subtract pick regions")
        pickadd_action.triggered.connect(self.subtract_picks)

        tools_menu.addSeparator()
        cluster_action = tools_menu.addAction("Cluster in pick (k-means)")
        cluster_action.triggered.connect(self.view.analyze_cluster)

        tools_menu.addSeparator()
        mask_action = tools_menu.addAction("Mask image")
        mask_action.triggered.connect(self.mask_settings_dialog.init_dialog)

        tools_menu.addSeparator()
        fast_render_action = tools_menu.addAction("Fast rendering")
        fast_render_action.triggered.connect(self.fast_render_dialog.show)
        
        # menu bar - Postprocess
        postprocess_menu = self.menu_bar.addMenu("Postprocess")
        undrift_action = postprocess_menu.addAction("Undrift by RCC")
        undrift_action.setShortcut("Ctrl+U")
        undrift_action.triggered.connect(self.view.undrift)
        undrift_from_picked_action = postprocess_menu.addAction(
            "Undrift from picked"
        )
        undrift_from_picked_action.setShortcut("Ctrl+Shift+U")
        undrift_from_picked_action.triggered.connect(
            self.view.undrift_from_picked
        )
        undrift_from_picked2d_action = postprocess_menu.addAction(
            "Undrift from picked (2D)"
        )
        undrift_from_picked2d_action.triggered.connect(
            self.view.undrift_from_picked2d
        )
        drift_action = postprocess_menu.addAction("Undo drift")
        drift_action.triggered.connect(self.view.undo_drift)
        drift_action = postprocess_menu.addAction("Show drift")
        drift_action.triggered.connect(self.view.show_drift)
        apply_drift_action = postprocess_menu.addAction(
            "Apply drift from an external file"
        )
        apply_drift_action.triggered.connect(self.view.apply_drift)

        postprocess_menu.addSeparator()
        group_action = postprocess_menu.addAction("Remove group info")
        group_action.triggered.connect(self.remove_group)
        unfold_action = postprocess_menu.addAction("Unfold / Refold groups")
        unfold_action.triggered.connect(self.view.unfold_groups)
        unfold_action_square = postprocess_menu.addAction(
            "Unfold groups (square)"
        )
        unfold_action_square.triggered.connect(self.view.unfold_groups_square)

        postprocess_menu.addSeparator()
        link_action = postprocess_menu.addAction("Link localizations")
        link_action.triggered.connect(self.view.link)
        align_action = postprocess_menu.addAction(
            "Align channels (RCC or from picked)"
        )
        align_action.triggered.connect(self.view.align)
        combine_action = postprocess_menu.addAction("Combine locs in picks")
        combine_action.triggered.connect(self.view.combine)

        postprocess_menu.addSeparator()
        apply_action = postprocess_menu.addAction(
            "Apply expression to localizations"
        )
        apply_action.setShortcut("Ctrl+A")
        apply_action.triggered.connect(self.open_apply_dialog)

        postprocess_menu.addSeparator()
        clustering_menu = postprocess_menu.addMenu("Clustering")
        dbscan_action = clustering_menu.addAction("DBSCAN")
        dbscan_action.triggered.connect(self.view.dbscan)
        hdbscan_action = clustering_menu.addAction("HDBSCAN")
        hdbscan_action.triggered.connect(self.view.hdbscan)
        clusterer_action = clustering_menu.addAction("SMLM clusterer")
        clusterer_action.triggered.connect(self.view.smlm_clusterer)
        test_cluster_action = clustering_menu.addAction("Test clusterer")
        test_cluster_action.triggered.connect(
            self.test_clusterer_dialog.show
        )

        postprocess_menu.addSeparator()
        nn_action = postprocess_menu.addAction("Nearest Neighbor Analysis")
        nn_action.triggered.connect(self.view.nearest_neighbor)

        postprocess_menu.addSeparator()
        resi_action = postprocess_menu.addAction("RESI")
        resi_action.triggered.connect(self.open_resi_dialog)

        self.load_user_settings()        

        # Define 3D entries
        self.actions_3d = [
            plotpick3d_action,
            plotpick3d_iso_action,
            slicer_action,
            undrift_from_picked2d_action,
            rot_win_action
        ]

        # set them invisible; if 3D is loaded later, they can be used
        for action in self.actions_3d:
            action.setVisible(False)

        # De-select all menus until file is loaded
        self.menus = [file_menu, view_menu, tools_menu, postprocess_menu]
        for menu in self.menus[1:]:
            menu.setDisabled(True)

        # add plugins; if it's the first initialization 
        # (plugins_loaded=False), they are not added because they're
        # loaded in __main___. Otherwise, (remove all locs) plugins 
        # need to be added to the menu bar.
        if plugins_loaded:
            try:
                for plugin in self.plugins:
                    plugin.execute()    
            except:
                pass        

    def closeEvent(self, event):
        """
        Changes user settings and closes all dialogs.

        Parameters
        ----------
        event : QCloseEvent
        """

        settings = io.load_user_settings()
        current_colormap = self.display_settings_dlg.colormap.currentText()
        if current_colormap == "Custom":
            try: # change colormap to the one saved the last time
                current_colormap = settings["Render"]["Colormap"]
            except: # otherwise, save magma
                current_colormap = "magma"
        settings["Render"]["Colormap"] = current_colormap
        if self.view.locs_paths != []:
            settings["Render"]["PWD"] = os.path.dirname(
                self.view.locs_paths[0]
            )
        io.save_user_settings(settings)
        QtWidgets.qApp.closeAllWindows()

    def export_current(self):
        """ Exports current view as .png or .tif. """

        try:
            base, ext = os.path.splitext(self.view.locs_paths[0])
        except AttributeError:
            return
        out_path = base + "_view.png"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", out_path, filter="*.png;;*.tif"
        )
        if path:
            scalebar = self.display_settings_dlg.scalebar_groupbox.isChecked()
            if not scalebar:
                self.display_settings_dlg.scalebar_groupbox.setChecked(True)
                qimage_scale = self.view.draw_scalebar(self.view.qimage)
                new_path, ext = os.path.splitext(path)
                new_path = new_path + "_scalebar" + ext
                qimage_scale.save(new_path)
                self.display_settings_dlg.scalebar_groupbox.setChecked(False)
            self.view.qimage.save(path)
            self.export_current_info(path)
        self.view.setMinimumSize(1, 1)

    def export_current_info(self, path):
        """ Exports information about the current file in .yaml 
        format. 
        
        Parameters
        ----------
        path : str
            Path for saving the original image with .png or .tif
            extension
        """

        path, ext = os.path.splitext(path)
        path = path + ".yaml"

        fov_info = [
            self.info_dialog.change_fov.x_box.value(),
            self.info_dialog.change_fov.y_box.value(),
            self.info_dialog.change_fov.w_box.value(),
            self.info_dialog.change_fov.h_box.value(),
        ]
        d = self.display_settings_dlg
        colors = [_.currentText() for _ in self.dataset_dialog.colorselection]

        info = {
            "FOV (X, Y, Width, Height)": fov_info,
            "Zoom": d.zoom.value(),
            "Display Pixel Size (nm)": d.disp_px_size.value(),
            "Min. Density": d.minimum.value(),
            "Max. Density": d.maximum.value(),
            "Colormap": d.colormap.currentText(),
            "Blur Method": d.blur_methods[d.blur_buttongroup.checkedButton()],
            "Scalebar Length (nm)": d.scalebar.value(),
            "Localizations Loaded": self.view.locs_paths,
            "Colors": colors,
        }

        io.save_info(path, [info])

    def export_complete(self):
        """ Exports the whole field of view as .png or .tif. """

        try:
            base, ext = os.path.splitext(self.view.locs_paths[0])
        except AttributeError:
            return
        out_path = base + ".png"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", out_path, filter="*.png;;*.tif"
        )
        if path:
            movie_height, movie_width = self.view.movie_size()
            viewport = [(0, 0), (movie_height, movie_width)]
            qimage = self.view.render_scene(cache=False, viewport=viewport)
            qimage.save(path)

    def export_txt(self):
        """ 
        Exports locs as .txt for ImageJ. 

        Saves frames, x and y.
        """

        channel = self.view.get_channel(
            "Save localizations as txt (frames,x,y)"
        )
        if channel is not None:
            base, ext = os.path.splitext(self.view.locs_paths[channel])
            out_path = base + ".frc.txt"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save localizations as txt (frames,x,y)",
                out_path,
                filter="*.frc.txt",
            )
            if path:
                locs = self.view.all_locs[channel]
                loctxt = locs[["frame", "x", "y"]].copy()
                np.savetxt(
                    path,
                    loctxt,
                    fmt=["%.1i", "%.5f", "%.5f"],
                    newline="\r\n",
                    delimiter="   ",
                )

    def export_txt_nis(self):
        """ Exports locs as .txt for NIS. """

        channel = self.view.get_channel(
            (
                "Save localizations as txt for NIS "
                "(x,y,z,channel,width,bg,length,area,frame)"
            )
        )
        pixelsize = self.display_settings_dlg.pixelsize.value()

        z_header = b"X\tY\tZ\tChannel\tWidth\tBG\tLength\tArea\tFrame\r\n"
        header = b"X\tY\tChannel\tWidth\tBG\tLength\tArea\tFrame\r\n"

        if channel is not None:
            base, ext = os.path.splitext(self.view.locs_paths[channel])
            out_path = base + ".nis.txt"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                (
                    "Save localizations as txt for NIS "
                    "(x,y,z,channel,width,bg,length,area,frame)"
                ),
                out_path,
                filter="*.nis.txt",
            )
            if path:
                locs = self.view.all_locs[channel]
                if hasattr(locs, "z"):
                    loctxt = locs[
                        ["x", "y", "z", "sx", "bg", "photons", "frame"]
                    ].copy()
                    loctxt = [
                        (
                            row[0] * pixelsize,
                            row[1] * pixelsize,
                            row[2] * pixelsize,
                            1,
                            row[3] * pixelsize,
                            row[4],
                            1,
                            row[5],
                            row[6] + 1,
                        )
                        for row in loctxt
                    ]
                    with open(path, "wb") as f:
                        f.write(z_header)
                        np.savetxt(
                            f,
                            loctxt,
                            fmt=[
                                "%.2f",
                                "%.2f",
                                "%.2f",
                                "%.i",
                                "%.2f",
                                "%.i",
                                "%.i",
                                "%.i",
                                "%.i",
                            ],
                            newline="\r\n",
                            delimiter="\t",
                        )
                        print("File saved to {}".format(path))
                else:
                    loctxt = locs[
                        ["x", "y", "sx", "bg", "photons", "frame"]
                    ].copy()
                    loctxt = [
                        (
                            row[0] * pixelsize,
                            row[1] * pixelsize,
                            1,
                            row[2] * pixelsize,
                            row[3],
                            1,
                            row[4],
                            row[5] + 1,
                        )
                        for row in loctxt
                    ]
                    with open(path, "wb") as f:
                        f.write(header)
                        np.savetxt(
                            f,
                            loctxt,
                            fmt=[
                                "%.2f",
                                "%.2f",
                                "%.i",
                                "%.2f",
                                "%.i",
                                "%.i",
                                "%.i",
                                "%.i",
                            ],
                            newline="\r\n",
                            delimiter="\t",
                        )
                        print("File saved to {}".format(path))

    def export_xyz_chimera(self):
        """ 
        Exports locs as .xyz for CHIMERA. Contains only x, y, z.

        Shows a warning if no z coordinate found.
        """

        channel = self.view.get_channel(
            "Save localizations as xyz for chimera (molecule,x,y,z)"
        )
        pixelsize = self.display_settings_dlg.pixelsize.value()
        if channel is not None:
            base, ext = os.path.splitext(self.view.locs_paths[channel])
            out_path = base + ".chi.xyz"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save localizations as xyz for chimera (molecule,x,y,z)",
                out_path,
            )
            if path:
                locs = self.view.all_locs[channel]
                if hasattr(locs, "z"):
                    loctxt = locs[["x", "y", "z"]].copy()
                    loctxt = [
                        (
                            1, 
                            row[0] * pixelsize, 
                            row[1] * pixelsize, 
                            row[2] * pixelsize,
                        )
                        for row in loctxt
                    ]
                    with open(path, "wb") as f:
                        f.write(b"Molecule export\r\n")
                        np.savetxt(
                            f,
                            loctxt,
                            fmt=["%i", "%.5f", "%.5f", "%.5f"],
                            newline="\r\n",
                            delimiter="\t",
                        )
                        print("File saved to {}".format(path))
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Dataset error", "Data has no z. Export skipped."
                    )

    def export_3d_visp(self):
        """
        Exports locs as .3d for ViSP.

        Shows a warning if no z coordinate found.
        """

        channel = self.view.get_channel(
            "Save localizations as xyz for chimera (molecule,x,y,z)"
        )
        pixelsize = self.display_settings_dlg.pixelsize.value()
        if channel is not None:
            base, ext = os.path.splitext(self.view.locs_paths[channel])
            out_path = base + ".visp.3d"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save localizations as xyz for chimera (molecule,x,y,z)",
                out_path,
            )
            if path:
                locs = self.view.all_locs[channel]
                if hasattr(locs, "z"):
                    locs = locs[["x", "y", "z", "photons", "frame"]].copy()
                    locs.x *= pixelsize
                    locs.y *= pixelsize
                    locs.z *= pixelsize
                    with open(path, "wb") as f:
                        np.savetxt(
                            f,
                            locs,
                            fmt=["%.1f", "%.1f", "%.1f", "%.1f", "%d"],
                            newline="\r\n",
                        )
                        print("Saving complete.")
                else:
                    QtWidgets.QMessageBox.information(
                        self, "Dataset error", "Data has no z. Export skipped."
                    )

    def export_multi(self):
        """ Asks the user to choose a type of export. """

        items = [
            ".txt for FRC (ImageJ)",
            ".txt for NIS",
            ".xyz for Chimera",
            ".3d for ViSP",
            ".csv for ThunderSTORM",
        ]
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Export", "Formats", items, 0, False
        )
        if ok and item:
            if item == ".txt for FRC (ImageJ)":
                self.export_txt()
            elif item == ".txt for NIS":
                self.export_txt_nis()
            elif item == ".xyz for Chimera":
                self.export_xyz_chimera()
            elif item == ".3d for ViSP":
                self.export_3d_visp()
            elif item == ".csv for ThunderSTORM":
                self.export_ts()

    def export_ts(self):
        """ Exports locs as .csv for ThunderSTORM. """

        channel = self.view.get_channel(
            "Save localizations as csv for ThunderSTORM"
        )
        pixelsize = self.display_settings_dlg.pixelsize.value()
        if channel is not None:
            base, ext = os.path.splitext(self.view.locs_paths[channel])
            out_path = base + ".csv"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save csv to", out_path, filter="*.csv"
            )
            if path:
                stddummy = 0
                locs = self.view.all_locs[channel]
                if hasattr(locs, "len"):  # Linked locs -> add detections
                    if hasattr(locs, "z"):
                        loctxt = locs[
                            [
                                "frame",
                                "x",
                                "y",
                                "sx",
                                "sy",
                                "photons",
                                "bg",
                                "lpx",
                                "lpy",
                                "z",
                                "len",
                            ]
                        ].copy()
                        loctxt = [
                            (
                                index,
                                row[0],
                                row[1] * pixelsize,
                                row[2] * pixelsize,
                                row[9] * pixelsize,
                                row[3] * pixelsize,
                                row[4] * pixelsize,
                                row[5],
                                row[6],
                                stddummy,
                                (row[7] + row[8]) / 2 * pixelsize,
                                row[10],
                            )
                            for index, row in enumerate(loctxt)
                        ]
                        header = ""
                        for element in [
                            "id",
                            "frame",
                            "x [nm]",
                            "y [nm]",
                            "z [nm]",
                            "sigma1 [nm]",
                            "sigma2 [nm]",
                            "intensity [photon]",
                            "offset [photon]",
                            "bkgstd [photon]",
                            "uncertainty_xy [nm]",
                            "detections",
                        ]:
                            header += '"' + element + '",'
                        header = header[:-1] + "\r\n"
                        with open(path, "wb") as f:
                            f.write(str.encode(header))

                            np.savetxt(
                                f,
                                loctxt,
                                fmt=[
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.i",
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                    "%.i",
                                ],
                                newline="\r\n",
                                delimiter=",",
                            )
                            print("File saved to {}".format(path))
                    else:
                        loctxt = locs[
                            [
                                "frame",
                                "x",
                                "y",
                                "sx",
                                "sy",
                                "photons",
                                "bg",
                                "lpx",
                                "lpy",
                                "len",
                            ]
                        ].copy()
                        loctxt = [
                            (
                                index,
                                row[0],
                                row[1] * pixelsize,
                                row[2] * pixelsize,
                                (row[3] + row[4]) / 2 * pixelsize,
                                row[5],
                                row[6],
                                stddummy,
                                (row[7] + row[8]) / 2 * pixelsize,
                                row[9],
                            )
                            for index, row in enumerate(loctxt)
                        ]
                        header = ""
                        for element in [
                            "id",
                            "frame",
                            "x [nm]",
                            "y [nm]",
                            "sigma [nm]",
                            "intensity [photon]",
                            "offset [photon]",
                            "bkgstd [photon]",
                            "uncertainty_xy [nm]",
                            "detections",
                        ]:
                            header += '"' + element + '",'
                        header = header[:-1] + "\r\n"

                        with open(path, "wb") as f:
                            f.write(str.encode(header))
                            np.savetxt(
                                f,
                                loctxt,
                                fmt=[
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.i",
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                    "%.i",
                                ],
                                newline="\r\n",
                                delimiter=",",
                            )
                            print("File saved to {}".format(path))

                else:
                    if hasattr(locs, "z"):
                        loctxt = locs[
                            [
                                "frame",
                                "x",
                                "y",
                                "sx",
                                "sy",
                                "photons",
                                "bg",
                                "lpx",
                                "lpy",
                                "z",
                            ]
                        ].copy()
                        loctxt = [
                            (
                                index,
                                row[0],
                                row[1] * pixelsize,
                                row[2] * pixelsize,
                                row[9] * pixelsize,
                                row[3] * pixelsize,
                                row[4] * pixelsize,
                                row[5],
                                row[6],
                                stddummy,
                                (row[7] + row[8]) / 2 * pixelsize,
                            )
                            for index, row in enumerate(loctxt)
                        ]
                        header = ""
                        for element in [
                            "id",
                            "frame",
                            "x [nm]",
                            "y [nm]",
                            "z [nm]",
                            "sigma1 [nm]",
                            "sigma2 [nm]",
                            "intensity [photon]",
                            "offset [photon]",
                            "bkgstd [photon]",
                            "uncertainty_xy [nm]",
                        ]:
                            header += '"' + element + '",'
                        header = header[:-1] + "\r\n"

                        with open(path, "wb") as f:
                            f.write(str.encode(header))
                            np.savetxt(
                                f,
                                loctxt,
                                fmt=[
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.i",
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                ],
                                newline="\r\n",
                                delimiter=",",
                            )
                            print("File saved to {}".format(path))
                    else:
                        loctxt = locs[
                            [
                                "frame",
                                "x",
                                "y",
                                "sx",
                                "sy",
                                "photons",
                                "bg",
                                "lpx",
                                "lpy",
                            ]
                        ].copy()
                        loctxt = [
                            (
                                index,
                                row[0],
                                row[1] * pixelsize,
                                row[2] * pixelsize,
                                (row[3] + row[4]) / 2 * pixelsize,
                                row[5],
                                row[6],
                                stddummy,
                                (row[7] + row[8]) / 2 * pixelsize,
                            )
                            for index, row in enumerate(loctxt)
                        ]
                        header = ""
                        for element in [
                            "id",
                            "frame",
                            "x [nm]",
                            "y [nm]",
                            "sigma [nm]",
                            "intensity [photon]",
                            "offset [photon]",
                            "bkgstd [photon]",
                            "uncertainty_xy [nm]",
                        ]:
                            header += '"' + element + '",'
                        header = header[:-1] + "\r\n"

                        with open(path, "wb") as f:
                            f.write(str.encode(header))
                            np.savetxt(
                                f,
                                loctxt,
                                fmt=[
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                    "%.2f",
                                    "%.2f",
                                    "%.i",
                                    "%.i",
                                    "%.i",
                                    "%.2f",
                                ],
                                newline="\r\n",
                                delimiter=",",
                            )
                            print("File saved to {}".format(path))

    def export_fov_ims(self):
        """ Exports current FOV to .ims """

        base, ext = os.path.splitext(self.view.locs_paths[0])
        out_path = base + ".ims"

        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export FOV as ims", out_path, filter="*.ims"
        )

        channel_base, ext_ = os.path.splitext(path)

        if os.path.isfile(path):
            os.remove(path)

        if path:
            status = lib.StatusDialog("Exporting ROIs..", self)

            n_channels = len(self.view.locs_paths)
            viewport = self.view.viewport
            oversampling = (
                self.display_settings_dlg.pixelsize.value()
                / self.display_settings_dlg.disp_px_size.value()
            )
            maximum = self.display_settings_dlg.maximum.value()

            pixelsize = self.display_settings_dlg.pixelsize.value() 

            ims_fields = {
                'ExtMin0':0,
                'ExtMin1':0, 
                'ExtMin2':-0.5, 
                'ExtMax2':0.5,
            }

            for k,v in ims_fields.items():
                try:
                    val = self.view.infos[0][0][k]
                    ims_fields[k] = None
                except KeyError:
                    pass

            (y_min, x_min), (y_max, x_max) = viewport

            z_mins = []
            z_maxs = []
            to_render = []

            has_z = True

            for channel in range(n_channels):
                if self.dataset_dialog.checks[channel].isChecked():
                    locs = self.view.locs[channel]

                    in_view = (
                        (locs.x > x_min) 
                        & (locs.x <= x_max) 
                        & (locs.y > y_min) 
                        & (locs.y <= y_max)
                    )

                    add_dict = {}
                    add_dict["Generated by"] = "Picasso Render (IMS Export)"

                    for k,v in ims_fields.items():
                        if v is not None:
                            add_dict[k] = v

                    info = self.view.infos[channel] + [add_dict]
                    io.save_locs(
                        f"{channel_base}_ch_{channel}.hdf5", 
                        locs[in_view], 
                        info,
                    )

                    if hasattr(locs, "z"):
                        z_min = locs.z[in_view].min()
                        z_max = locs.z[in_view].max()
                        z_mins.append(z_min)
                        z_maxs.append(z_max)
                    else:
                        has_z = False

                    to_render.append(channel)

            if not has_z:
                if len(z_mins) > 0:
                    raise NotImplementedError(
                        "Can't export mixed files with and without z."
                    )

            if has_z:
                z_min = min(z_mins)
                z_max = max(z_maxs)
            else:
                z_min, z_max = 0, 0

            all_img = []
            for idx, channel in enumerate(to_render):
                locs = self.view.locs[channel]
                if has_z:
                    n, image = render.render_hist3d(
                        locs, 
                        oversampling, 
                        y_min, x_min, y_max, x_max, z_min, z_max, 
                        pixelsize,
                    )
                else:
                    n, image = render.render_hist(
                        locs, 
                        oversampling, 
                        y_min, x_min, y_max, x_max,
                    )

                image = image / maximum * 65535
                data = image.astype('uint16')
                data = np.rot90(np.fliplr(data))
                all_img.append(data)

            s_image = np.stack(all_img, axis=-1).T.copy()

            colors = self.view.read_colors()
            colors_ims = [PW.Color(*list(colors[_]), 1) for _ in to_render]

            numpy_to_imaris(
                s_image, 
                path, 
                colors_ims, 
                oversampling, 
                viewport, 
                info, 
                z_min, 
                z_max, 
                pixelsize,
            )
            status.close()

    def load_picks(self):
        """ Loads picks from a .yaml file. """

        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pick regions", filter="*.yaml"
        )
        if path:
            self.view.load_picks(path)

    def subtract_picks(self):
        """ 
        Subtracts picks from a .yaml file. 

        See View.subtract_picks.
        """
        
        if self.view._picks:
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load pick regions", filter="*.yaml"
            )
            if path:
                self.view.subtract_picks(path)
        else:
            warning = "No picks found. Please pick first."
            QtWidgets.QMessageBox.information(self, "Warning", warning)

    def load_user_settings(self):
        """ 
        Loads colormap and current directory (ones used last time). 
        """

        settings = io.load_user_settings()
        colormap = settings["Render"]["Colormap"]
        if len(colormap) == 0:
            colormap = "magma"
        for index in range(self.display_settings_dlg.colormap.count()):
            if self.display_settings_dlg.colormap.itemText(index) == colormap:
                self.display_settings_dlg.colormap.setCurrentIndex(index)
                break
        pwd = []
        try:
            pwd = settings["Render"]["PWD"]
        except Exception as e:
            print(e)
            pass
        if len(pwd) == 0:
            pwd = []
        self.pwd = pwd

    def open_apply_dialog(self):
        """ Loads expression and applies it to locs. """

        cmd, channel, ok = ApplyDialog.getCmd(self)
        if ok:
            input = cmd.split()
            if input[0] == "flip" and len(input) == 3:
                # Distinguish flipping in xy and z
                if "z" in input:
                    var_1 = input[1]
                    var_2 = input[2]
                    if var_1 == "z":
                        var_2 = "z"
                        var_1 = input[2]
                    pixelsize = self.display_settings_dlg.pixelsize.value()
                    templocs = self.view.locs[channel][var_1].copy()
                    movie_height, movie_width = self.view.movie_size()
                    if var_1 == "x":
                        dist = movie_width
                    else:
                        dist = movie_height

                    self.view.locs[channel][var_1] = (
                        self.view.locs[channel][var_2] / pixelsize + dist / 2
                    )  # exchange w. info
                    self.view.all_locs[channel][var_1] = (
                        self.view.all_locs[channel[var_2]]
                        / pixelsize
                        + dist / 2
                    )
                    self.view.locs[channel][var_2] = templocs * pixelsize
                    self.view.all_locs[channel][var_2] = templocs * pixelsize
                else:
                    var_1 = input[1]
                    var_2 = input[2]
                    templocs = self.view.locs[channel][var_1].copy()
                    self.view.locs[channel][var_1] = self.view.locs[channel][
                        var_2
                    ]
                    self.view.all_locs[channel][var_1] = self.view.all_locs[
                        channel
                    ][var_2]
                    self.view.locs[channel][var_2] = templocs
                    self.view.all_locs[channel][var_2] = templocs

            elif input[0] == "spiral" and len(input) == 3:
                # spiral uses radius and turns
                radius = float(input[1])
                turns = int(input[2])
                maxframe = self.view.infos[channel][0]["Frames"]

                self.x_spiral = self.view.locs[channel]["x"].copy()
                self.y_spiral = self.view.locs[channel]["y"].copy()

                scale_time = maxframe / (turns * 2 * np.pi)
                scale_x = turns * 2 * np.pi

                x = self.view.locs[channel]["frame"] / scale_time

                self.view.locs[channel]["x"] = (
                    x * np.cos(x)
                ) / scale_x * radius + self.view.locs[channel]["x"]
                self.view.all_locs[channel]["x"] = (
                    x * np.cos(x)
                ) / scale_x * radius + self.view.all_locs[channel]["x"]
                self.view.locs[channel]["y"] = (
                    x * np.sin(x)
                ) / scale_x * radius + self.view.locs[channel]["y"]
                self.view.all_locs[channel]["y"] = (
                    x * np.sin(x)
                ) / scale_x * radius + self.view.all_locs[channel]["y"]

            elif input[0] == "uspiral":
                try: 
                    self.view.locs[channel]["x"] = self.x_spiral
                    self.view.all_locs[channel]["x"] = self.x_spiral
                    self.view.locs[channel]["y"] = self.y_spiral
                    self.view.all_locs[channel]["y"] = self.y_spiral
                    self.display_settings_dlg.render_check.setChecked(False)
                except:
                    QtWidgets.QMessageBox.information(
                        self, 
                        "Uspiral error", 
                        "Localizations have not been spiraled yet."
                    )
            else:
                vars = self.view.locs[channel].dtype.names
                exec(cmd, {k: self.view.locs[channel][k] for k in vars})
                exec(cmd, {k: self.view.all_locs[channel][k] for k in vars})
            lib.ensure_sanity(
                self.view.locs[channel], self.view.infos[channel]
            )
            lib.ensure_sanity(
                self.view.all_locs[channel], self.view.infos[channel]
            )
            self.view.index_blocks[channel] = None
            self.view.update_scene()

    def open_file_dialog(self):
        """ Opens localizations .hdf5 file(s). """

        if self.pwd == []:
            paths, ext = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add localizations", filter="*.hdf5"
            )
        else:
            paths, ext = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add localizations", directory=self.pwd, filter="*.hdf5"
            )
        if paths:
            self.pwd = paths[0]
            self.view.add_multiple(paths)

    def open_rotated_locs(self):
        """ 
        Opens rotated localizations .hdf5 file(s). 

        In addition to normal file opening, it also requires to load
        info about the pick and rotation.
        """

        # self.remove_locs()
        if self.pwd == []:
            path, ext = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add localizations", filter="*.hdf5" 
            )
        else:
            path, ext = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add localizations", directory=self.pwd, filter="*.hdf5"
            )
        if path:
            self.pwd = path[0]
            self.view.add_multiple(path)
            if "Pick" in self.view.infos[0][-1]:
                self.view._picks = []
                self.view._picks.append(self.view.infos[0][-1]["Pick"])
                self.view._pick_shape = self.view.infos[0][-1]["Pick shape"]
                if self.view._pick_shape == "Circle":
                    self.tools_settings_dialog.pick_diameter.setValue(
                        self.view.infos[0][-1]["Pick size"]
                    )
                else:
                    self.tools_settings_dialog.pick_width.setValue(
                        self.view.infos[0][-1]["Pick size"]
                    )
                self.window_rot.view_rot.angx = self.view.infos[0][-1]["angx"]
                self.window_rot.view_rot.angy = self.view.infos[0][-1]["angy"]
                self.window_rot.view_rot.angz = self.view.infos[0][-1]["angz"]
                self.rot_win()

    def resizeEvent(self, event):
        """ Updates window size when resizing. """

        self.update_info()

    def remove_group(self):
        """ Displayed locs will have no group information. """

        channel = self.view.get_channel("Remove group")
        if channel is not None:
            self.view.locs[channel] = lib.remove_from_rec(
                self.view.locs[channel], "group"
            )
            self.view.all_locs[channel] = lib.remove_from_rec(
                self.view.all_locs[channel], "group"
            )
            self.view.update_scene()

    def save_pick_properties(self):
        """ 
        Saves pick properties in a given channel (or all channels). 
        """

        channel = self.view.get_channel_all_seq("Save localizations")
        if channel is not None:
            if channel == len(self.view.locs_paths):
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_pickprops",
                )
                if ok:
                    for channel in range(len(self.view.locs_paths)):
                        base, ext = os.path.splitext(
                            self.view.locs_paths[channel]
                        )
                        out_path = base + suffix + ".hdf5"
                        self.view.save_pick_properties(out_path, channel)
            else:
                base, ext = os.path.splitext(self.view.locs_paths[channel])
                out_path = base + "_pickprops.hdf5"
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save pick properties", out_path, filter="*.hdf5"
                )
                if path:
                    self.view.save_pick_properties(path, channel)

    def save_locs(self):
        """
        Saves localizations in a given channel (or all channels).
        """

        channel = self.view.save_channel("Save localizations")
        if channel is not None:
            # combine all channels
            if channel is (len(self.view.locs_paths) + 1):
                base, ext = os.path.splitext(self.view.locs_paths[0])
                out_path = base + "_multi.hdf5"
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save localizations",
                    out_path,
                    filter="*.hdf5",
                )
                if path:
                    # combine locs from all channels
                    all_locs = stack_arrays(
                        self.view.all_locs, 
                        asrecarray=True, 
                        usemask=False,
                        autoconvert=True,
                    )
                    all_locs.sort(kind="mergesort", order="frame")
                    info = self.view.infos[0] + [
                        {
                            "Generated by": "Picasso Render Combine",
                            "Paths to combined files": self.view.locs_paths,
                        }
                    ]
                    io.save_locs(path, all_locs, info)

            # save all channels one by one
            elif channel is (len(self.view.locs_paths)):
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_arender",
                )
                if ok:
                    for channel in range(len(self.view.locs_paths)):
                        base, ext = os.path.splitext(
                            self.view.locs_paths[channel]
                        )
                        out_path = base + suffix + ".hdf5"
                        info = self.view.infos[channel] + [
                            {
                                "Generated by": "Picasso Render",
                                "Last driftfile": self.view._driftfiles[
                                    channel
                                ],
                            }
                        ]
                        io.save_locs(
                            out_path, self.view.all_locs[channel], info
                        )
            # save one channel only
            else:
                base, ext = os.path.splitext(self.view.locs_paths[channel])
                out_path = base + "_render.hdf5"
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save localizations", out_path, filter="*.hdf5"
                )
                if path:
                    info = self.view.infos[channel] + [
                        {
                            "Generated by": "Picasso Render",
                            "Last driftfile": self.view._driftfiles[channel],
                        }
                    ]
                    io.save_locs(path, self.view.all_locs[channel], info)

    def save_picked_locs(self):
        """
        Saves picked localizations in a given channel (or all channels).
        """

        channel = self.view.save_channel("Save picked localizations")
        if channel is not None:
            # combine channels to one .hdf5
            if channel is (len(self.view.locs_paths) + 1):
                base, ext = os.path.splitext(self.view.locs_paths[0])
                out_path = base + "_picked_multi.hdf5"
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save picked localizations",
                    out_path,
                    filter="*.hdf5",
                )
                if path:
                    self.view.save_picked_locs_multi(path)
            # save channels one by one
            elif channel is (len(self.view.locs_paths)):
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_apicked",
                )
                if ok:
                    for channel in range(len(self.view.locs_paths)):
                        base, ext = os.path.splitext(
                            self.view.locs_paths[channel]
                        )
                        out_path = base + suffix + ".hdf5"
                        self.view.save_picked_locs(out_path, channel)
            # save one channel only
            else:
                base, ext = os.path.splitext(self.view.locs_paths[channel])
                out_path = base + "_picked.hdf5"
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save picked localizations",
                    out_path,
                    filter="*.hdf5",
                )
                if path:
                    self.view.save_picked_locs(path, channel)

    def save_picks(self):
        """ Saves picks as .yaml. """

        base, ext = os.path.splitext(self.view.locs_paths[0])
        out_path = base + "_picks.yaml"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save pick regions", out_path, filter="*.yaml"
        )
        if path:
            self.view.save_picks(path)

    def remove_locs(self):
        """ Resets Window. """

        for dialog in self.dialogs:
            dialog.close()
        self.menu_bar.clear() #otherwise the menu bar is doubled
        self.setWindowTitle(f"Picasso v{__version__}: Render")
        self.initUI(plugins_loaded=True)

    def rot_win(self):
        """ Opens/updates RotationWindow. """

        if len(self.view._picks) == 0:
            raise ValueError("Pick a region to rotate.")
        elif len(self.view._picks) > 1:
            raise ValueError("Pick only one region.")
        self.window_rot.view_rot.load_locs(update_window=True)
        self.window_rot.show()
        self.window_rot.view_rot.update_scene(autoscale=True)

    def update_info(self):
        """
        Updates Window's size and median loc prec in InfoDialog. 
        """

        self.info_dialog.width_label.setText(
            "{} pixel".format((self.view.width()))
        )
        self.info_dialog.height_label.setText(
            "{} pixel".format((self.view.height()))
        )
        self.info_dialog.locs_label.setText("{:,}".format(self.view.n_locs))
        try:
            self.info_dialog.xy_label.setText(
                "{:.2f} / {:.2f} ".format(
                    self.view.viewport[0][1], self.view.viewport[0][0]
                )
            )
            self.info_dialog.wh_label.setText(
                "{:.2f} / {:.2f} pixel".format(
                    self.view.viewport_width(), self.view.viewport_height()
                )
            )
        except AttributeError:
            pass
        try:
            self.info_dialog.change_fov.x_box.setValue(
                self.view.viewport[0][1]
            )
            self.info_dialog.change_fov.y_box.setValue(
                self.view.viewport[0][0]
            )
            self.info_dialog.change_fov.w_box.setValue(
                self.view.viewport_width()
            )
            self.info_dialog.change_fov.h_box.setValue(
                self.view.viewport_height()
            )
        except AttributeError:
            pass
        try:
            self.info_dialog.fit_precision.setText(
                "{:.3} nm".format(
                    self.view.median_lp 
                    * self.display_settings_dlg.pixelsize.value()
                )
            )
        except AttributeError:
            pass

    def open_resi_dialog(self):
        resi_dialog = RESIDialog(self)
        self.dialogs.append(resi_dialog)
        resi_dialog.show()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.plugins = []

    # load plugins from picasso/gui/plugins
    from . import plugins

    def iter_namespace(pkg):
        return pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")

    plugins = [
        importlib.import_module(name)
        for finder, name, ispkg
        in iter_namespace(plugins)
    ]

    for plugin in plugins:
        p = plugin.Plugin(window)
        if p.name == "render":
            p.execute()
            window.plugins.append(p)

    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        QtCore.QCoreApplication.instance().processEvents()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(
            window, "An error occured", message
        )
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()