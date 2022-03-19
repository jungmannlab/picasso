"""
    gui/render
    ~~~~~~~~~~~~~~~~~~~~
    Graphical user interface for rendering localization images
    :author: Joerg Schnitzbauer & Maximilian Strauss, 2017-2018
    :copyright: Copyright (c) 2017 Jungmann Lab, MPI of Biochemistry
"""
import os
import sys
import traceback
import copy
import time
import os.path
import importlib, pkgutil
from glob import glob
from math import ceil
from icecream import ic
from functools import partial

import lmfit
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import joblib

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
    as NavigationToolbar

from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from numpy.lib.recfunctions import stack_arrays
from PyQt5 import QtCore, QtGui, QtWidgets

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d
from collections import Counter
from h5py import File
from tqdm import tqdm

import colorsys

from .. import imageprocess, io, lib, postprocess, render
from .rotation import RotationWindow

matplotlib.rcParams.update({"axes.titlesize": "large"})

DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5
ZOOM = 9 / 7
N_GROUP_COLORS = 8
N_Z_COLORS = 32


def get_colors(n_channels):
    """ Creates a tuple with rgb channels for each locs channel. """

    hues = np.arange(0, 1, 1 / n_channels)
    colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
    return colors

def is_hexadecimal(text):
    """ 
    True when text is a hexadecimal rgb expression, e.g. #ff02d4,
    False otherwise. 
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
                (
                    "No localizations picked."
                    " Please pick at least twice first."
                ),
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
    figure : plt.Figure
    canvas : FigureCanvas
        PyQt5 backend used for displaying plots
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
    figure : plt.Figure
    canvas : FigureCanvas
        PyQt5 backend used for displaying plots
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
    label : QLabel
        Displays which locs properties can be manipulated
    cmd : QLineEdit
        Enter the expression here

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
    window : Window(QMainWindow)
        Main window instance
    warning : boolean
        Used to memorize if the warning about multiple channels is to
        be displayed
    checks : list
        List with QPushButtons for ticking/unticking each channel
    title : list
        List of QPushButtons to change the title of each channel
    closebuttons : list
        List of QPushButtons to close each channel
    colorselection : list
        List of QComboBoxes specifying the color displayed for each
        channel
    colordisp_all : list
        List of QLabels showing the color selected for each channel
    intensitysettings : list
        List of QDoubleSpinBoxes specifying relative intensity of each
        channel
    legend : QCheckBox
        Used to show/hide legend
    wbackground : QCheckBox
        Used to (de)activate white background for multichannel or
        to invert colors for single channel
    auto_display : QCheckBox
        Tick to automatically adjust the rendered localizations. Untick
        to not change the rendering of localizations
    auto_colors : QCheckBox
        Tick to automatically color each channel. Untick to manually 
        change colors.
    default_colors : list
        List of strings specifying the default 14 colors
    rgbf : list
        List of lists of 3 elements specifying the corresponding colors
        as RGB channels

    Methods
    -------
    add_entry(path)
        Adds the new channel for the given path
    update_colors()
        Changes colors in self.colordisp_all and updates the scene in
        the main window
    change_title(button_name)
        Opens QInputDialog to enter the new title for a given channel
    close_file(i)
        Closes a given channel and delets all corresponding attributes
    update_viewport()
        Updates the scene in the main window
    set_color(n)
        Sets colorsdisp_all and colorselection in the given channel
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
        self.layout.addWidget(self.legend, 0, 0)
        self.layout.addWidget(self.auto_display, 1, 0)
        self.layout.addWidget(self.wbackground, 2, 0)
        self.layout.addWidget(self.auto_colors, 3, 0)
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
            partial(self.close_file, p.objectName())
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
                    " colors.  In case you would like to use your own color, "
                    " please insert the color's hexadecimal expression,"
                    "  starting with '#',  e.g.  '#ffcdff' for pink or choose"
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
                            path = "..." + path[-40:]
                        self.checks[i].setText(path)
                    else:
                        self.checks[i].setText(new_title)
                    self.update_viewport()
                    self.adjustSize()
                break

    def close_file(self, i):
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
            ax.scatter(locs["x"], locs["y"], locs["z"], c=colors, cmap="jet", s=2)
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

            ax.scatter(locs["x"], locs["y"], locs["z"], c=colors, cmap="jet", s=2)
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
            ax3.set_ylabel("Z [Px]")
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
            ax4.set_ylabel("Z [Px]")
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
            ax3.set_ylabel("Z [Px]")
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
            ax4.set_ylabel("Z [Px]")
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


class ClsDlg(QtWidgets.QDialog):
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
        self.layout_grid.addWidget(QtWidgets.QLabel("No clusters:"), 10, 3, 1, 1)

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

        dialog = ClsDlg(None)

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

        print("Mode 1")
        locs = all_picked_locs[current]
        locs = stack_arrays(locs, asrecarray=True, usemask=False)

        est = KMeans(n_clusters=n_clusters)

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

        ax1.scatter(locs["x"], locs["y"], locs["z"], c=labels.astype(np.float), s=2)

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
        ax1.set_zlabel("Z [nm]")

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

        print("Mode 1")
        locs = all_picked_locs[current]
        locs = stack_arrays(locs, asrecarray=True, usemask=False)

        est = KMeans(n_clusters=n_clusters)

        scaled_locs = lib.append_to_rec(locs, locs["x"], "x_scaled")
        scaled_locs = lib.append_to_rec(scaled_locs, locs["y"], "y_scaled")

        X = np.asarray(scaled_locs["x_scaled"])
        Y = np.asarray(scaled_locs["y_scaled"])

        est.fit(np.stack((X, Y), axis=1))

        labels = est.labels_

        counts = list(Counter(labels).items())
        # l_locs = lib.append_to_rec(l_locs,labels,'cluster')

        ax1.scatter(locs["x"], locs["y"], c=labels.astype(np.float), s=2)

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
    max_distance : QDoubleSpinBox
        contains the maximum distance (pixels) between locs to be
        considered as belonging to the same group of linked locs
    max_dark_time : QDoubleSpinBox
        contains the maximum gap between localizations (frames) to be
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
    radius : QDoubleSpinBox
        contains epsilon (pixels) for DBSCAN (see scikit-learn)
    density : QSpinBox
        contains min_samples for DBSCAN (see scikit-learn)

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
        self.radius.setRange(0, 1e6)
        self.radius.setValue(1)
        grid.addWidget(self.radius, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min. samples:"), 1, 0)
        self.density = QtWidgets.QSpinBox()
        self.density.setRange(0, 1e6)
        self.density.setValue(4)
        grid.addWidget(self.density, 1, 1)
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
        Creates the dialog and returns the requested values for DBSCAN.
        """

        dialog = DbscanDialog(parent)
        result = dialog.exec_()
        return (
            dialog.radius.value(),
            dialog.density.value(),
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
    min_cluster : QSpinBox
        contains the minimum number of locs in a cluster
    min_samples : QSpinBox
        contains the number of locs in a neighbourhood for a loc to be 
        considered a core point, see the website
    cluster_eps : QDoubleSpinBox
        contains cluster_selection_epsilon (pixels), see the website

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
        self.min_cluster.setRange(0, 1e6)
        self.min_cluster.setValue(10)
        grid.addWidget(self.min_cluster, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min. samples:"), 1, 0)
        self.min_samples = QtWidgets.QSpinBox()
        self.min_samples.setRange(0, 1e6)
        self.min_samples.setValue(10)
        grid.addWidget(self.min_samples, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Intercluster max. distance: (pixels)"), 2, 0)
        self.cluster_eps = QtWidgets.QDoubleSpinBox()
        self.cluster_eps.setRange(0, 1e6)
        self.cluster_eps.setValue(0.0)
        grid.addWidget(self.cluster_eps, 2, 1)
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
        HDBSCAN.
        """

        dialog = HdbscanDialog(parent)
        result = dialog.exec_()
        return (
            dialog.min_cluster.value(),
            dialog.min_samples.value(),
            dialog.cluster_eps.value(),
            result == QtWidgets.QDialog.Accepted,
        )


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
    plot_3d(drift)
        Creates 3 plots with drift
    plot_2D(drift)
        Creates 2 plots with drift
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
    x_box : QDoubleSpinBox
        contains the minimum x coordinate (pixels) to be displayed
    y_box : QDoubleSpinBox
        contains the minimum y coordinate (pixels) to be displayed
    w_box : QDoubleSpinBox
        contains the width of the viewport (pixels)
    h_box : QDoubleSpinBox
        contains the height of the viewport (pixels)

    Methods
    -------
    save_fov()
        Used for saving the current FOV as a .txt file
    load_fov()
        Used for loading a FOV from a .txt file
    update_scene()
        Updates the scene in the main window and Display section of the
        Info Dialog
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Change field of view")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.x_box = QtWidgets.QDoubleSpinBox()
        self.x_box.setKeyboardTracking(False)
        self.x_box.setRange(-100, 1e6)
        self.layout.addWidget(self.x_box, 0, 1)
        self.layout.addWidget(QtWidgets.QLabel("Y :"), 1, 0)
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

    Attributes
    ----------
    window : Window(QMainWindow)
        main window instance
    change_fov : ChangeFOV(QDialog)
        dialog for changing field of view
    width_label : QLabel
        contains the width of the window (pixels)
    height_label : QLabel
        contains the height of the window (pixels)
    xy_label : QLabel
        shows the minimum y and u coordinates in FOV (pixels) 
    wh_label : QLabel
        shows the width and height of the current FOV (pixels)
    change_display : QPushButton
        opens self.change_fov
    movie_grid : QGridLayout
        contains all the info about the fit precision
    fit_precision : QLabel
        shows median fit precision of the first channel (pixels)
    nena_button : QPushButton
        calculates nearest neighbor based analysis fit precision
    locs_label : QLabel
        shows the number of locs in the current FOV
    picks_grid : QGridLayout
        contains all the info about the picks
    n_picks : QLabel
        shows the number of picks
    n_localization_mean : QLabel
        shows the mean number of locs in all picks
    n_localization_std : QLabel
        shows the std number of locs in all picks
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
    max_dark_time : QSpinBox
        contains the maximum gap between localizations (frames) to be
        considered as belonging to the same group of linked locs
    dark_mean : QLabel
        shows the mean dark time (frames) in all picks
    dark_std : QLabel
        shows the std dark time (frames) in all picks
    units_per_pick : QSpinBox
        contains the number of binding sites per pick
    influx_rate : FloadEdit(QLineEdit)
        contains the calculated or input influx rate (1/frames)
    n_units_mean : QLabel
        shows the calculated mean number of binding sites in all picks
    n_units_std : QLabel
        shows the calculated std number of binding sites in all picks

    Methods
    -------
    calculate_nena_lp()
        Calculates and plots NeNA precision in a given channel
    calibrate_influx()
        Calculates influx rate (1/frames)
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
        self.movie_grid.addWidget(QtWidgets.QLabel("Median fit precision:"), 0, 0)
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
        compute_pick_info_button = QtWidgets.QPushButton("Calculate info below")
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
        self.picks_grid.addWidget(QtWidgets.QLabel("Ignore dark times <="), row, 0)
        self.max_dark_time = QtWidgets.QSpinBox()
        self.max_dark_time.setRange(0, 1e9)
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
        self.picks_grid.addWidget(QtWidgets.QLabel("# Units per pick:"), row, 0)
        self.units_per_pick = QtWidgets.QSpinBox()
        self.units_per_pick.setRange(1, 1e6)
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

            # modify the movie grid
            self.nena_button.setParent(None)
            self.movie_grid.removeWidget(self.nena_button)
            progress = lib.ProgressDialog(
                "Calculating NeNA precision", 0, 100, self
            )
            result_lp = postprocess.nena(locs, info, progress.set_value)
            self.nena_label = QtWidgets.QLabel()
            self.movie_grid.addWidget(self.nena_label, 1, 1)
            self.nena_result, lp = result_lp
            lp *= self.window.display_settings_dlg.pixelsize.value()
            self.nena_label.setText("{:.3} nm".format(lp))
            show_plot_button = QtWidgets.QPushButton("Show plot")
            self.movie_grid.addWidget(
                show_plot_button, self.movie_grid.rowCount() - 1, 2
            )

            # Nena plot
            self.nena_window = NenaPlotWindow(self)
            self.nena_window.plot(self.nena_result)
            show_plot_button.clicked.connect(self.nena_window.show)

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
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Generate Mask")
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)
        mask_groupbox = QtWidgets.QGroupBox("Mask Settings")
        vbox.addWidget(mask_groupbox)
        mask_grid = QtWidgets.QGridLayout(mask_groupbox)

        mask_grid.addWidget(QtWidgets.QLabel("Oversampling"), 0, 0)
        self.mask_oversampling = QtWidgets.QSpinBox()
        self.mask_oversampling.setRange(1, 999999)
        self.mask_oversampling.setValue(1)
        self.mask_oversampling.setSingleStep(1)
        self.mask_oversampling.setKeyboardTracking(False)

        self.mask_oversampling.valueChanged.connect(self.update_plots)

        mask_grid.addWidget(self.mask_oversampling, 0, 1)

        mask_grid.addWidget(QtWidgets.QLabel("Blur"), 1, 0)
        self.mask_blur = QtWidgets.QDoubleSpinBox()
        self.mask_blur.setRange(0, 999999)
        self.mask_blur.setValue(2)
        self.mask_blur.setSingleStep(0.1)
        self.mask_blur.setDecimals(3)
        mask_grid.addWidget(self.mask_blur, 1, 1)

        self.mask_blur.valueChanged.connect(self.update_plots)

        mask_grid.addWidget(QtWidgets.QLabel("Threshold"), 2, 0)
        self.mask_tresh = QtWidgets.QDoubleSpinBox()
        self.mask_tresh.setRange(0, 1)
        self.mask_tresh.setValue(0.5)
        self.mask_tresh.setSingleStep(0.01)
        self.mask_tresh.setDecimals(3)

        self.mask_tresh.valueChanged.connect(self.update_plots)
        mask_grid.addWidget(self.mask_tresh, 2, 1)

        self.figure = plt.figure(figsize=(12, 3))
        self.canvas = FigureCanvas(self.figure)
        mask_grid.addWidget(self.canvas, 3, 0, 1, 2)

        self.maskButton = QtWidgets.QPushButton("Mask")
        mask_grid.addWidget(self.maskButton, 5, 0)
        self.maskButton.clicked.connect(self.mask_locs)

        self.saveButton = QtWidgets.QPushButton("Save")
        self.saveButton.setEnabled(False)
        self.saveButton.clicked.connect(self.save_locs)
        mask_grid.addWidget(self.saveButton, 5, 1)

        self.loadMaskButton = QtWidgets.QPushButton("Load Mask")
        self.loadMaskButton.clicked.connect(self.load_mask)
        mask_grid.addWidget(self.loadMaskButton, 4, 0)

        self.saveMaskButton = QtWidgets.QPushButton("Save Mask")
        self.saveMaskButton.setEnabled(False)
        self.saveMaskButton.clicked.connect(self.save_mask)
        mask_grid.addWidget(self.saveMaskButton, 4, 1)

        self.locs = []
        self.paths = []
        self.infos = []

        self.oversampling = 2
        self.blur = 1
        self.thresh = 0.5

        self.cached_oversampling = 0
        self.cached_blur = 0
        self.cached_thresh = 0

        self.mask_exists = 0

    def init_dialog(self):
        self.show()
        locs = self.locs[0]
        info = self.infos[0][0]
        self.x_min = 0
        self.y_min = 0
        self.x_max = info["Width"]
        self.y_max = info["Height"]
        self.x_min_d, self.x_max_d = [
            np.floor(np.min(locs["x"])),
            np.ceil(np.max(locs["x"])),
        ]
        self.y_min_d, self.y_max_d = [
            np.floor(np.min(locs["y"])),
            np.ceil(np.max(locs["y"])),
        ]
        self.update_plots()

    def generate_image(self):
        locs = self.locs[0]
        self.stepsize = 1 / self.oversampling
        self.xedges = np.arange(self.x_min, self.x_max, self.stepsize)
        self.yedges = np.arange(self.y_min, self.y_max, self.stepsize)
        H, xedges, yedges = np.histogram2d(
            locs["x"], locs["y"], bins=(self.xedges, self.yedges)
        )
        H = H.T  # Let each row list bins with common y range.
        self.H = H

    def blur_image(self):
        H_blur = gaussian_filter(self.H, sigma=self.blur)
        H_blur = H_blur / np.max(H_blur)
        self.H_blur = H_blur

    def save_mask(self):
        # Open dialog to save mask
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save mask to", filter="*.npy"
        )
        if path:
            np.save(path, self.mask)

    def load_mask(self):
        # Save dialog to load mask
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load mask", filter="*.npy"
        )
        if path:
            self.mask = np.load(path)
            # adjust oversampling for mask
            oversampling = int((self.mask.shape[0] + 1) / self.y_max)
            self.oversampling = oversampling
            self.mask_oversampling.setValue(oversampling)
            self.saveMaskButton.setEnabled(True)
            self.generate_image()
            self.blur_image()
            self.update_plots(newMask=False)

    def mask_image(self):
        mask = np.zeros_like(self.H_blur)
        mask[self.H_blur > self.tresh] = 1
        self.mask = mask
        self.saveMaskButton.setEnabled(True)

    def update_plots(self, newMask=True):
        if newMask:
            if (
                self.mask_oversampling.value() == self.oversampling
                and self.cached_oversampling == 1
            ):
                self.cached_oversampling = 1
            else:
                self.oversampling = self.mask_oversampling.value()
                self.cached_oversampling = 0

            if self.mask_blur.value() == self.blur and self.cached_blur == 1:
                self.cached_blur = 1
            else:
                self.blur = self.mask_blur.value()
                self.cached_oversampling = 0

            if (
                self.mask_tresh.value() == self.thresh
                and self.cached_thresh == 1
            ):
                self.cached_thresh = 1
            else:
                self.tresh = self.mask_tresh.value()
                self.cached_thresh = 0

            if self.cached_oversampling:
                pass
            else:
                self.generate_image()
                self.blur_image()
                self.mask_image()
                self.cached_oversampling = 1
                self.cached_blur = 1
                self.cached_thresh = 1

            if self.cached_blur:
                pass
            else:
                self.blur_image()
                self.mask_image()
                self.cached_blur = 1
                self.cached_thresh = 1

            if self.cached_thresh:
                pass
            else:
                self.mask_image()
                self.cached_thresh = 1
        else:
            pass

        ax1 = self.figure.add_subplot(141, title="Original")
        ax1.imshow(
            self.H,
            interpolation="nearest",
            origin="lower",
            extent=[
                self.xedges[0],
                self.xedges[-1],
                self.yedges[0],
                self.yedges[-1],
            ],
        )
        ax1.grid(False)
        ax1.set_xlim(self.x_min_d, self.x_max_d)
        ax1.set_ylim(self.y_min_d, self.y_max_d)
        ax2 = self.figure.add_subplot(142, title="Blurred")
        ax2.imshow(
            self.H_blur,
            interpolation="nearest",
            origin="lower",
            extent=[
                self.xedges[0],
                self.xedges[-1],
                self.yedges[0],
                self.yedges[-1],
            ],
        )
        ax2.grid(False)
        ax2.set_xlim(self.x_min_d, self.x_max_d)
        ax2.set_ylim(self.y_min_d, self.y_max_d)
        ax3 = self.figure.add_subplot(143, title="Mask")
        ax3.imshow(
            self.mask,
            interpolation="nearest",
            origin="lower",
            extent=[
                self.xedges[0],
                self.xedges[-1],
                self.yedges[0],
                self.yedges[-1],
            ],
        )
        ax3.grid(False)
        ax3.set_xlim(self.x_min_d, self.x_max_d)
        ax3.set_ylim(self.y_min_d, self.y_max_d)
        ax4 = self.figure.add_subplot(144, title="Masked image")
        ax4.imshow(
            np.zeros_like(self.H),
            interpolation="nearest",
            origin="lower",
            extent=[
                self.xedges[0],
                self.xedges[-1],
                self.yedges[0],
                self.yedges[-1],
            ],
        )
        ax4.grid(False)
        ax4.set_xlim(self.x_min_d, self.x_max_d)
        ax4.set_ylim(self.y_min_d, self.y_max_d)
        self.canvas.draw()

    def mask_locs(self):
        locs = self.locs[0]
        steps_x = len(self.xedges)
        steps_y = len(self.yedges)

        x_ind = (
            np.floor(
                (locs["x"] - self.x_min) / (self.x_max - self.x_min) * steps_x
            )
            - 1
        )
        y_ind = (
            np.floor(
                (locs["y"] - self.y_min) / (self.y_max - self.y_min) * steps_y
            )
            - 1
        )
        x_ind = x_ind.astype(int)
        y_ind = y_ind.astype(int)

        index = self.mask[y_ind, x_ind].astype(bool)
        self.index_locs = locs[index]
        self.index_locs_out = locs[~index]

        H_new, xedges, yedges = np.histogram2d(
            self.index_locs["x"],
            self.index_locs["y"],
            bins=(self.xedges, self.yedges),
        )
        self.H_new = H_new.T  # Let each row list bins with common y range.

        ax4 = self.figure.add_subplot(144, title="Masked image")
        ax4.imshow(
            self.H_new,
            interpolation="nearest",
            origin="low",
            extent=[
                self.xedges[0],
                self.xedges[-1],
                self.yedges[0],
                self.yedges[-1],
            ],
        )
        ax4.grid(False)
        self.mask_exists = 1
        self.saveButton.setEnabled(True)
        self.canvas.draw()

    def save_locs(self):
        channel = 0
        base, ext = os.path.splitext(self.paths[channel])
        out_path = base + "_mask_in.hdf5"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save localizations within mask", out_path, filter="*.hdf5"
        )
        if path:
            info = self.infos[channel] + [
                {"Generated by": "Picasso Render : Mask in "}
            ]
            clusterfilter_info = {
                "Oversampling": self.oversampling,
                "Blur": self.blur,
                "Threshold": self.tresh,
            }
            info.append(clusterfilter_info)
            io.save_locs(path, self.index_locs, info)
        out_path = base + "_mask_out.hdf5"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save localizations outside of mask",
            out_path,
            filter="*.hdf5",
        )
        if path:
            info = self.infos[channel] + [
                {"Generated by": "Picasso Render : Mask out"}
            ]
            clusterfilter_info = {
                "Oversampling": self.oversampling,
                "Blur": self.blur,
                "Threshold": self.tresh,
            }
            info.append(clusterfilter_info)
            io.save_locs(path, self.index_locs_out, info)


class PickToolCircleSettings(QtWidgets.QWidget):
    """ A class contating information about circular pick. """

    def __init__(self, window, tools_settings_dialog):
        super().__init__()
        self.grid = QtWidgets.QGridLayout(self)
        self.window = window
        self.grid.addWidget(QtWidgets.QLabel("Diameter (cam. pixel):"), 0, 0)
        self.pick_diameter = QtWidgets.QDoubleSpinBox()
        self.pick_diameter.setRange(0, 999999)
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
        self.pick_width.setRange(0, 999999)
        self.pick_width.setValue(1)
        self.pick_width.setSingleStep(0.1)
        self.pick_width.setDecimals(3)
        self.pick_width.setKeyboardTracking(False)
        self.pick_width.valueChanged.connect(
            tools_settings_dialog.on_pick_dimension_changed
        )
        self.grid.addWidget(self.pick_width, 0, 1)
        self.grid.setRowStretch(1, 1)


class ToolsSettingsDialog(QtWidgets.QDialog):
    """
    A dialog class to customize picks - vary shape and size, annotate,
    change std for picking similar.

    ...

    Attributes
    ----------
    pick_shape : QComboBox
        contains the str with the shape of picks (circle or rectangle)
    pick_diameter : QDoubleSpinBox
        contains the diameter of circular picks (pixels)
    pick_width : QDoubleSpinBox
        contains the width of rectangular picks (pixels)
    pick_annotation : QCheckBox
        tick to display picks' indeces 
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

        self.point_picks = QtWidgets.QCheckBox("Display circular picks as points")
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
    A class to change display settings, e.g.: zoom, oversampling, 
    contrast and blur.

    ...

    Attributes
    ----------
    zoom : QDoubleSpinBox
        contains zoom's magnitude
    oversampling : QDoubleSpinBox
        contains the number of super-resolution pixels per camera pixel
    dynamic_oversampling : QCheckBox
        tick to automatically adjust to current window size when zooming.
    minimap : QCheckBox
        tick to display minimap showing current FOV
    minimum : QDoubleSpinBox
        defines at which number of localizations per super-resolution
        pixel the minimum color of the colormap should be applied
    maximum : QDoubleSpinBox
        defines at which number of localizations per super-resolution
        pixel the maximum color of the colormap should be applied
    colormap : QComboBox
        contains strings with available colormaps (single channel only)
    blur_buttongroup : QButtonGroup
        contains available localization blur methods
    min_blur_width : QDoubleSpinBox
        contains the minimum blur for each localization (pixels)
    pixelsize : QDoubleSpinBox
        contains the camera pixel size (nm)
    scalebar_groupbox : QGroupBox
        group with options for customizing scale bar, tick to display
    scalebar : QDoubleSpinBox
        contains the scale bar's length (nm)
    scalebar_text : QCheckBox
        tick to display scale bar's length (nm)
    _silent_oversampling_update : boolean
        True if update oversampling in background
    parameter : QComboBox
        defines what property should be rendered, e.g.: z, photons
    minimum_render : QDoubleSpinBox
        contains the minimum value of the parameter to be rendered
    maximum_render : QDoubleSpinBox
        contains the maximum value of the parameter to be rendered
    color_step : QSpinBox
        defines how many colors are to be rendered
    render_check : QCheckBox
        tick to activate parameter rendering
    show_legend : QPushButton
        click to display parameter rendering's legend

    Methods
    -------
    on_oversampling_changed(value)
        Sets new oversampling, updates contrast and updates scene in
        the main window
    on_zoom_changed(value)
        Zooms the image in the main window
    set_oversampling_silently(oversampling)
        Changes the value of oversampling in the background
    set_zoom_silently(zoom)
        Changes the value of zoom in the background
    silent_minimum_update(value)
        Changes the value of self.minimum in the background
    silent_maximum_update(value)
        Changes the value of self.maximum in the background
    render_scene(*args, **kwargs)
        Updates scene in the main window
    set_dynamic_oversampling(state)
        Updates scene if dynamic oversampling is checked
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
        general_grid.addWidget(QtWidgets.QLabel("Oversampling:"), 1, 0)
        self._oversampling = DEFAULT_OVERSAMPLING
        self.oversampling = QtWidgets.QDoubleSpinBox()
        self.oversampling.setRange(0.001, 1000)
        self.oversampling.setSingleStep(5)
        self.oversampling.setValue(self._oversampling)
        self.oversampling.setKeyboardTracking(False)
        self.oversampling.valueChanged.connect(self.on_oversampling_changed)
        general_grid.addWidget(self.oversampling, 1, 1)
        self.dynamic_oversampling = QtWidgets.QCheckBox("dynamic")
        self.dynamic_oversampling.setChecked(True)
        self.dynamic_oversampling.toggled.connect(
            self.set_dynamic_oversampling
        )
        general_grid.addWidget(self.dynamic_oversampling, 2, 1)
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
        contrast_grid.addWidget(self.colormap, 2, 1)
        self.colormap.currentIndexChanged.connect(self.update_scene)

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
            QtWidgets.QLabel("Min.  Blur (cam.  pixel):"), 5, 0, 1, 1
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
        self.camera_grid.addWidget(QtWidgets.QLabel("Pixel Size:"), 0, 0)
        self.pixelsize = QtWidgets.QDoubleSpinBox()
        self.pixelsize.setRange(1, 1000000000)
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
        scalebar_grid.addWidget(QtWidgets.QLabel("Scale Bar Length (nm):"), 0, 0)
        self.scalebar = QtWidgets.QDoubleSpinBox()
        self.scalebar.setRange(0.0001, 10000000000)
        self.scalebar.setValue(500)
        self.scalebar.setKeyboardTracking(False)
        self.scalebar.valueChanged.connect(self.update_scene)
        scalebar_grid.addWidget(self.scalebar, 0, 1)
        self.scalebar_text = QtWidgets.QCheckBox("Print scale bar length")
        self.scalebar_text.stateChanged.connect(self.update_scene)
        scalebar_grid.addWidget(self.scalebar_text, 1, 0)
        self._silent_oversampling_update = False

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

    def on_oversampling_changed(self, value):
        """
        Sets new oversampling, updates contrast and updates scene in
        the main window.
        """

        contrast_factor = (self._oversampling / value) ** 2
        self._oversampling = value
        self.silent_minimum_update(contrast_factor * self.minimum.value())
        self.silent_maximum_update(contrast_factor * self.maximum.value())
        if not self._silent_oversampling_update:
            self.dynamic_oversampling.setChecked(False)
            self.window.view.update_scene()

    def on_zoom_changed(self, value):
        """ Zooms the image in the main window. """

        self.window.view.set_zoom(value)

    def set_oversampling_silently(self, oversampling):
        """ Changes the value of oversampling in the background. """

        self._silent_oversampling_update = True
        self.oversampling.setValue(oversampling)
        self._silent_oversampling_update = False

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

    def set_dynamic_oversampling(self, state):
        """ Updates scene if dynamic oversampling is checked. """

        if state:
            self.window.view.update_scene()

    def update_scene(self, *args, **kwargs):
        """ Updates scene with cache. """

        self.window.view.update_scene(use_cache=True)


class SlicerDialog(QtWidgets.QDialog):
    """
    A class to customize slicing 3D data in z axis.

    ...

    Attributes
    ----------
    window : QMainWindow
        instance of the main window
    pick_slice : QSpinBox
        contains slice thickness (nm)
    sl : QSlider
        points to the slice to be displayed
    canvas : FigureCanvas
        contains the histogram of number of locs in slices
    slicer_radio_button : QCheckBox
        tick to slice locs
    zcoord : list
        z coordinates of each channel of localization (nm)
    separate_check : QCheckBox
        tick to save channels separately when exporting slice
    full_check : QCheckBox
        tick to save the whole FOV, untick to save only the current
        viewport
    export_button : QPushButton
        click to export slices into .tif files
    slicer_cache : dict
        contains QPixMaps that have been drawn for each slice
    bins : np.array
        contatins bins used in plotting the histogram
    colors : list
        contains rgb channels for each localization channel
    patches : list
        contains plt.artists used in creating histograms
    slicerposition : float
        current position of self.sl
    slicermin : float
        minimum value of self.sl
    slicermax : float
        maximum value of self.sl

    Methods
    -------
    initialize()
        Called when the dialog is open, calculates the histograms and 
        shows the dialog
    calculate_histogram()
        Calculates and histograms z coordintes of each channel
    on_pick_slice_changed()
        Modifies histograms when slice size changes
    toggle_slicer()
        Updates scene in the main window when slicer is moved
    on_slice_position_changed(position)
        Changes some properties and updates scene in the main window
    export_stack(self)
        Saves all slices as .tif files
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("3D Slicer ")
        self.setModal(False)
        vbox = QtWidgets.QVBoxLayout(self)
        slicer_groupbox = QtWidgets.QGroupBox("Slicer Settings")

        vbox.addWidget(slicer_groupbox)
        slicer_grid = QtWidgets.QGridLayout(slicer_groupbox)
        slicer_grid.addWidget(QtWidgets.QLabel("Slice Thickness [nm]:"), 0, 0)
        self.pick_slice = QtWidgets.QSpinBox()
        self.pick_slice.setRange(1, 999999)
        self.pick_slice.setValue(50)
        self.pick_slice.setSingleStep(5)
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

        self.figure = plt.figure(figsize=(3, 3))
        self.canvas = FigureCanvas(self.figure)

        self.slicer_radio_button = QtWidgets.QCheckBox("Slice Dataset")
        self.slicer_radio_button.stateChanged.connect(self.toggle_slicer)

        self.zcoord = []
        self.separate_check = QtWidgets.QCheckBox("Export channels separate")
        self.full_check = QtWidgets.QCheckBox("Export full image")
        self.export_button = QtWidgets.QPushButton("Export Slices")
        self.export_button.setAutoDefault(False)

        self.export_button.clicked.connect(self.export_stack)

        slicer_grid.addWidget(self.canvas, 2, 0, 1, 2)
        slicer_grid.addWidget(self.slicer_radio_button, 3, 0)
        slicer_grid.addWidget(self.separate_check, 4, 0)
        slicer_grid.addWidget(self.full_check, 5, 0)
        slicer_grid.addWidget(self.export_button, 6, 0)

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
        ax = self.figure.add_subplot(111)

        # clear the plot
        plt.cla()
        n_channels = len(self.zcoord)

        # get colors for each channel
        self.colors = get_colors(n_channels)

        # get bins, starting with minimum z and ending with max z
        self.bins = np.arange(
            np.amin(np.hstack(self.zcoord)),
            np.amax(np.hstack(self.zcoord)),
            slice,
        )

        # plot histograms
        self.patches = []
        for i in range(len(self.zcoord)):
            n, bins, patches = plt.hist(
                self.zcoord[i],
                self.bins,
                density=True,
                facecolor=self.colors[i],
                alpha=0.5,
            )
            self.patches.append(patches)

        plt.xlabel("Z-Coordinate [nm]")
        plt.ylabel("Counts")
        plt.title(r"$\mathrm{Histogram\ of\ Z:}$")
        self.canvas.draw()
        self.sl.setMaximum(len(self.bins) - 2)
        self.sl.setValue(len(self.bins) / 2)

        # reset cache
        self.slicer_cache = {}

    def on_pick_slice_changed(self):
        """ Modifies histograms when slice size changes. """

        # reset cache
        self.slicer_cache = {}
        if len(self.bins) < 3:  # in case there should be only 1 bin
            self.calculate_histogram()
        else:
            self.calculate_histogram()
            self.sl.setValue(len(self.bins) / 2)
            self.on_slice_position_changed(self.sl.value())

    def toggle_slicer(self):
        """ Updates scene in the main window when slicer is moved. """

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
                    for i in tqdm(range(self.sl.maximum() + 1)):
                        self.sl.setValue(i)
                        print("Slide: " + str(i))
                        out_path = (
                            base
                            + "_Z"
                            + "{num:03d}".format(num=i)
                            + "_CH"
                            + "{num:03d}".format(num=j + 1)
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

                for i in tqdm(range(self.sl.maximum() + 1)):
                    self.sl.setValue(i)
                    print("Slide: " + str(i))
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
    currentdrift : list
        contains the most up-to-date drift for each channel
    _drift : list
        contains np.recarrays with drift info for each channel, None if 
        no drift found/calculated
    _driftfiles : list
        contains paths to drift .txt files for each channel
    group_color : np.array
        important for single channel data with group info (picked or
        clustered locs); contains an integer index for each loc 
        defining its color
    index_blocks : list
        contains tuples with info about indexed locs for each channel, 
        None if not calculated yet
    infos : list
        contains a dictionary with metadata for each channel
    locs : list
        contains a np.recarray with localizations for each channel
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
    window : QMainWindow
        instance of the main window
    x_render_cache : list
        contains dicts with caches for storing info about locs rendered
        by a property
    x_render_state : boolean
        indicates if rendering by property is used

    Methods
    -------
    activate_property_menu
    activate_render_property
    add(path)
        Loads a .hdf5 and .yaml files
    add_drift
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
    analyze_cluster
    apply_drift
    combine()
        Combines all locs in each pick into one localization
    clear_picks()
        Deletes all current picks
    dbscan()
        Performs DBSCAN with user-defined parameters
    deactivate_property_menu
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
    filter_picks
    fit_in_view()
        Updates scene with all locs shown
    get_channel()
        Opens an input dialog to ask for a channel
    get_channel3d()
        Similar to get_channel, used in selecting 3D picks
    get_group_color(locs)
        Finds group color index for each localization
    get_index_blocks
    get_pick_rectangle_corners(start_x, start_y, end_x, end_y, width)
        Finds the positions of a rectangular pick's corners
    get_pick_rectangle_polygon(start_x, start_y, end_x, end_y, width)
        Finds a PyQt5 object used for drawing a rectangular pick
    get_render_kwargs()
        Returns a dictionary to be used for the kwargs of render.render
    hdscan()
        Performs HDBSCAN with user-defined parameters
    index_locs
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
    on_pick_shape_changed
    pan_relative(dy, dx)
        Moves viewport by a given relative distance
    pick_message_box(params)
        Returns a message box for selecting picks
    pick_similar
    picked_locs
    refold_groups
    relative_position
    remove_points
    remove_picks
    render_multi_channel
    render_scene
    render_single_channel
    render_time
    render_3d
    resizeEvent
    rmsd_at_com
    save_channel_multi()
        Opens an input dialog asking which channel to save
    save_channel()
        Opens an input dialog asking which channel of picked locs to
        save
    save_channel_pickprops()
        Opens an input dialog asking which channel to use in saving 
        pick properties
    save_pick_properties
    save_picked_locs
    save_picked_locs_multi
    save_picks
    scale_contrast
    select_traces()
        Lets user to select picks based on their traces
    set_mode
    set_property
    set_zoom
    shifts_from_picked_coordinate(locs, coordinate)
        Calculates shifts between channels along a given coordinate
    shift_from_picked()
        For each pick, calculate the center of mass and rcc based on 
        shifts
    shift_from_rcc()
        Estimates image shifts based on whole images' rcc
    show_drift
    show_legend
    show_legend_files
    show_pick
    show_pick_3d
    show_pick_3d_iso
    show_trace()
        Displays x and y coordinates of locs in picks in time
    sizeHint
    subtract_picks(path)
        Clears current picks that cover the picks loaded from path
    to_8bit
    to_down
    to_left
    to_right
    to_up
    undrift
    undrift_from_picked
    _undrift_from_picked
    _undrift_from_picked_coordinate
    undrift_from_picked2d
    _undrift_from_picked2d
    undo_drift
    _undo_drift
    unfold_groups
    unfold_groups_square
    update_cursor
    update_pick_info_long
    update_pick_info_short
    update_scene
    update_scene_slicer
    viewport_center
    viewport_height
    viewport_size
    viewport_width
    wheelEvent
    zoom
    zoom_in
    zoom_out
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

        groups = np.unique(locs.group)
        groupcopy = locs.group.copy()

        # check if groups are consecutive
        if set(groups) == set(range(min(groups), max(groups) + 1)):
            if len(groups) > 5000:
                choice = QtWidgets.QMessageBox.question(
                    self,
                    "Group question",
                    (
                        "Groups are not consecutive"
                        " and more than 5000 groups detected."
                        " Re-Index groups? This may take a while."
                    ),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if choice == QtWidgets.QMessageBox.Yes:
                    pb = lib.ProgressDialog(
                        "Re-Indexing groups", 0, len(groups), self
                    )
                    pb.set_value(0)
                    for i in tqdm(range(len(groups))):
                        groupcopy[locs.group == groups[i]] = i
                        pb.set_value(i)
                    pb.close()
            else:
                for i in tqdm(range(len(groups))):
                    groupcopy[locs.group == groups[i]] = i
        else:
            for i in range(len(groups)):
                groupcopy[locs.group == groups[i]] = i
        np.random.shuffle(groups)
        groups %= N_GROUP_COLORS
        return groups[groupcopy]

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
                if len(self.group_color) == 0:
                    self.group_color = self.get_group_color(locs)

        # render the loaded file
        if render:
            self.fit_in_view(autoscale=True)
            self.update_scene()

        # add options to rendering by parameter
        self.window.display_settings_dlg.parameter.addItems(locs.dtype.names)

        # append z coordinates for slicing
        if hasattr(locs, "z"):
            self.window.slicer_dialog.zcoord.append(locs.z)
            # unlock 3D settings
            for action in self.window.actions_3d:
                action.setVisible(True)

        # allow using View, Tools and Postprocess menus
        for menu in self.window.menus:
            menu.setDisabled(False)

        # append data for masking
        self.window.mask_settings_dialog.locs.append(
            locs
        )  # TODO: replace at some point, not very efficient
        self.window.mask_settings_dialog.paths.append(path)
        self.window.mask_settings_dialog.infos.append(info)

        # change current working directory
        os.chdir(os.path.dirname(path))

        # add the locs to the dataset dialog
        self.window.dataset_dialog.add_entry(path)

        self.window.setWindowTitle(
            "Picasso: Render. File: {}".format(os.path.basename(path))
        )

    def add_multiple(self, paths):
        """ Loads several .hdf5 and .yaml files. 
        
        Parameters
        ----------
        paths: list
            Contains the paths to the files to be loaded
        """

        fit_in_view = len(self.locs) == 0
        paths = sorted(paths)
        for path in paths:
            self.add(path, render=False)
        if len(self.locs):  # in case loading was not succesful
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

    def add_point(self, position, update_scene=True):
        """ 
        Adds a point at a given position for measuring distances.
        """

        self._points.append(position)
        if update_scene:
            self.update_scene()

    def add_picks(self, positions):
        """ Adds several picks. """

        for position in positions:
            self.add_pick(position, update_scene=False)
        self.update_scene(picks_only=True)

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
        self.locs[channel] = stack_arrays(
            out_locs, asrecarray=True, usemask=False
        )

        if hasattr(self.locs[channel], "group"):
            groups = np.unique(self.locs[channel].group)
            # In case a group is missing
            groups = np.arange(np.max(groups) + 1)
            np.random.shuffle(groups)
            groups %= N_GROUP_COLORS
            self.group_color = groups[self.locs[channel].group]

        self.update_scene()

    def link(self):
        """
        Link localizations 
        """

        channel = self.get_channel()
        if hasattr(self.locs[channel], "len"):
            QtWidgets.QMessageBox.information(
                self, "Link", "Localizations are already linked. Aborting..."
            )
            return
        else:
            r_max, max_dark, ok = LinkDialog.getParams()
            if ok:
                status = lib.StatusDialog("Linking localizations...", self)
                self.locs[channel] = postprocess.link(
                    self.locs[channel],
                    self.infos[channel],
                    r_max=r_max,
                    max_dark_time=max_dark,
                )
                status.close()
                if hasattr(self.locs[channel], "group"):
                    groups = np.unique(self.locs[channel].group)
                    groups = np.arange(np.max(groups) + 1)
                    np.random.shuffle(groups)
                    groups %= N_GROUP_COLORS
                    self.group_color = groups[self.locs[channel].group]
                self.update_scene()

    def dbscan(self):
        """
        Gets DBSCAN parameters, performs clustering and saves data.
        """

        # get DBSCAN parameters
        radius, min_density, ok = DbscanDialog.getParams()
        if ok:
            status = lib.StatusDialog(
                "Applying DBSCAN. This may take a while...", self
            )

            # perform DBSCAN for each channel
            for locs, locs_info, locs_path in zip(
                self.locs, self.infos, self.locs_paths
            ):
                pixelsize = self.window.display_settings_dlg.pixelsize.value()
                clusters, locs = postprocess.dbscan(
                    locs, radius, min_density, pixelsize
                )
                base, ext = os.path.splitext(locs_path)
                dbscan_info = {
                    "Generated by": "Picasso DBSCAN",
                    "Radius": radius,
                    "Minimum local density": min_density,
                }
                locs_info.append(dbscan_info)
                io.save_locs(base + "_dbscan.hdf5", locs, locs_info)
                with File(base + "_dbclusters.hdf5", "w") as clusters_file:
                    clusters_file.create_dataset("clusters", data=clusters)

                # inform tbe user where clustered locs were saved
                QtWidgets.QMessageBox.information(
                    self, 
                    "DBSCAN", 
                    (
                        "Clustering executed.  Results are saved in: \n" 
                        + base 
                        + "_dbscan.hdf5" 
                        + "\n" 
                        + base 
                        + "_dbclusters.hdf5"
                    ),
                )

            status.close()

    def hdbscan(self):
        """
        Gets DBSCAN parameters, performs clustering and saves data.
        """

        # get HDBSCAN parameters
        min_cluster, min_samples, cluster_eps, ok = HdbscanDialog.getParams()
        if ok:
            status = lib.StatusDialog(
                "Applying HDBSCAN. This may take a while...", self
            )
            # perform HDBSCAN for each channel
            for locs, locs_info, locs_path in zip(
                    self.locs, self.infos, self.locs_paths
            ):
                pixelsize = self.window.display_settings_dlg.pixelsize.value()
                clusters, locs = postprocess.hdbscan(
                    locs, min_cluster, min_samples, cluster_eps, pixelsize
                )
                base, ext = os.path.splitext(locs_path)
                hdbscan_info = {
                    "Generated by": "Picasso HDBSCAN",
                    "Min. cluster": min_cluster,
                    "Min. samples": min_samples,
                    "Intercluster distance": cluster_eps,
                }
                locs_info.append(hdbscan_info)
                io.save_locs(base + "_hdbscan.hdf5", locs, locs_info)
                with File(base + "_hdbclusters.hdf5", "w") as clusters_file:
                    clusters_file.create_dataset("clusters", data=clusters)
                # inform tbe user where clustered locs were saved
                QtWidgets.QMessageBox.information(
                    self, 
                    "HDBSCAN", 
                    (
                        "Clustering executed.  Results are saved in: \n" 
                        + base 
                        + "_hdbscan.hdf5" 
                        + "\n" 
                        + base 
                        + "_hdbclusters.hdf5"
                    ),
                )
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
        x1 = start_x - dx
        x2 = start_x + dx
        x4 = end_x - dx
        x3 = end_x + dx
        y1 = start_y + dy
        y2 = start_y - dy
        y4 = end_y + dy
        y3 = end_y - dy
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

                # yellow is barely visible on white background
                if self.window.dataset_dialog.wbackground.isChecked():
                    painter.setBrush(QtGui.QBrush(QtGui.QColor("red")))

                for i, pick in enumerate(self._picks):

                    # convert from camera units to display units
                    cx, cy = self.map_to_view(*pick)
                    painter.drawEllipse(QtCore.QPoint(cx, cy), 3, 3)

                    # annotate picks
                    if t_dialog.pick_annotation.isChecked():
                        painter.drawText(cx + 2, cy + 2, str(i))
                painter.end()

            # draw circles
            else:
                d = t_dialog.pick_diameter.value()
                d *= self.width() / self.viewport_width()

                painter = QtGui.QPainter(image)
                painter.setPen(QtGui.QColor("yellow"))

                # yellow is barely visible on white background
                if self.window.dataset_dialog.wbackground.isChecked():
                    painter.setPen(QtGui.QColor("red"))

                for i, pick in enumerate(self._picks):

                    # convert from camera units to display units
                    cx, cy = self.map_to_view(*pick)
                    painter.drawEllipse(cx - d / 2, cy - d / 2, d, d)

                    # annotate picks
                    if t_dialog.pick_annotation.isChecked():
                        painter.drawText(cx + d / 2, cy + d / 2, str(i))
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
            painter.drawLine(cx, cy, cx + d / 2, cy)
            painter.drawLine(cx, cy, cx, cy + d / 2)
            painter.drawLine(cx, cy, cx - d / 2, cy)
            painter.drawLine(cx, cy, cx, cy - d / 2)

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
                    (cx + ox) / 2 + d, (cy + oy) / 2 + d, str(distance) + " nm"
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

            # size of drawn squares with colors (display pixels)
            width = 15
            height = 15

            # initial positions
            x = 20
            y = -5

            dy = 25 # space between squares
            for i in range(n_channels):
                if self.window.dataset_dialog.checks[i].isChecked():
                    painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                    colordisp = self.window.dataset_dialog.colordisp_all[i]
                    color = colordisp.palette().color(QtGui.QPalette.Window)
                    painter.setBrush(QtGui.QBrush(color))
                    y += dy
                    painter.drawRect(x, y, height, height)
                    font = painter.font()
                    font.setPixelSize(12)
                    painter.setFont(font)
                    painter.setPen(QtGui.QColor("white"))

                    # white channel name not visible on white background
                    if self.window.dataset_dialog.wbackground.isChecked():
                        painter.setPen(QtGui.QColor("black"))
                    text_spacer = 25
                    text_width = 1000 # in case of long names
                    text_height = 15
                    text = self.window.dataset_dialog.checks[i].text()
                    painter.drawText(
                        x + text_spacer,
                        y,
                        text_width,
                        text_height,
                        QtCore.Qt.AlignLeft,
                        text,
                    )
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
            height_minimap = movie_height / movie_width * 100
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
            length = self.viewport_width() / movie_width * length_minimap
            height = self.viewport_height() / movie_height * height_minimap
            x_vp = self.viewport[0][1] / movie_width * length_minimap
            y_vp = self.viewport[0][0] / movie_height * length_minimap
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
            Viewport defining the current FOV
        autoscale : boolean (default=False)
            True if contrast should be automatically adjusted
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
            # scale image to the window
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
            True if contrast should be automatically adjusted
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
            self.window.slicer_dialog.slicer_cache[
                slicerposition
            ] = self.pixmap
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
        paths = [
            path for path, ext in zip(paths, extensions) if ext == ".hdf5"
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
            else:
                # calculate new viewport
                r = self.window.tools_settings_dialog.pick_diameter.value() / 2
                x, y = self._picks[pick_no]
                x_min = x - 1.4 * r
                x_max = x + 1.4 * r
                y_min = y - 1.4 * r
                y_max = y + 1.4 * r
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

    def save_channel_multi(self, title="Choose a channel"):
        """
        Opens an input dialog to ask which channel to save.
        There is an option to save all channels.

        Returns
        None if no locs found or channel picked, int otherwise
            Index of the chosen channel
        """

        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Save all at once")
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

    def save_channel(self, title="Choose a channel"):
        """
        Opens an input dialog to ask which channel of picked locs to 
        save.
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
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Save all at once")
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

    def save_channel_pickprops(self, title="Choose a channel"):
        """
        Opens an input dialog to ask which channel to use in saving 
        pick properties.
        There is an option to save all channels.

        Returns
        None if no locs found or channel picked, int otherwise
            Index of the chosen channel
        """

        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Save all at once")
            index, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Save pick properties",
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
        blur_button = (
            self.window.display_settings_dlg.blur_buttongroup.checkedButton()
        )

        # oversampling
        optimal_oversampling = (
            self.display_pixels_per_viewport_pixels()
        )
        if self.window.display_settings_dlg.dynamic_oversampling.isChecked():
            oversampling = optimal_oversampling
            self.window.display_settings_dlg.set_oversampling_silently(
                optimal_oversampling
            )
        else:
            oversampling = float(
                self.window.display_settings_dlg.oversampling.value()
            )
            if oversampling > optimal_oversampling:
                QtWidgets.QMessageBox.information(
                    self,
                    "Oversampling too high",
                    (
                        "Oversampling will be adjusted to"
                        " match the display pixel density."
                    ),
                )
                oversampling = optimal_oversampling
                self.window.display_settings_dlg.set_oversampling_silently(
                    optimal_oversampling
                )

        # viewport
        if viewport is None:
            viewport = self.viewport

        return {
            "oversampling": oversampling,
            "viewport": viewport,
            "blur_method": self.window.display_settings_dlg.blur_methods[
                blur_button
            ],
            "min_blur_width": float(
                self.window.display_settings_dlg.min_blur_width.value()
            ),
        }

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
        return cx, cy

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
                self.remove_points((x, y))
                event.accept()
            else:
                event.ignore()

    def movie_size(self):
        """ Returns tuple with movie height and width. """

        movie_height = self.max_movie_height()
        movie_width = self.max_movie_width()
        return (movie_height, movie_width)

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

            xvec = np.arange(max(locs["frame"]) + 1)
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
            ax1.set_xlim(0, (max(locs["frame"]) + 1))
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
                "Picks per Minute: {:.2f}"
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

    def select_traces(self):
        """ 
        Lets user to select picks based on their traces.
        Opens self.pick_message_box to display information.
        """

        print("Showing  traces")

        removelist = [] # picks to be removed

        channel = self.get_channel("Undrift from picked")
        if channel is not None:
            if self._picks: # if there are picks present
                params = {} # stores info about selecting picks
                params["t0"] = time.time()
                all_picked_locs = self.picked_locs(channel)
                i = 0 # index of the currently shown pick
                while i < len(self._picks):
                    fig = plt.figure(figsize=(5, 5))
                    fig.canvas.set_window_title("Trace")
                    pick = self._picks[i]
                    locs = all_picked_locs[i]
                    locs = stack_arrays(locs, asrecarray=True, usemask=False)

                    # essentialy the same plotting as in self.show_trace
                    ax1 = fig.add_subplot(311)
                    ax2 = fig.add_subplot(312, sharex=ax1)
                    ax3 = fig.add_subplot(313, sharex=ax1)

                    xvec = np.arange(max(locs["frame"]) + 1)
                    yvec = xvec[:] * 0
                    yvec[locs["frame"]] = 1
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
                    ax1.scatter(locs["frame"], locs["x"], s=2)
                    ax1.set_ylabel("X-pos [Px]")
                    ax1.set_title("X-pos vs frame")

                    ax1.set_xlim(0, (max(locs["frame"]) + 1))
                    plt.setp(ax1.get_xticklabels(), visible=False)

                    ax2.scatter(locs["frame"], locs["y"], s=2)
                    ax2.set_title("Y-pos vs frame")
                    ax2.set_ylabel("Y-pos [Px]")
                    plt.setp(ax2.get_xticklabels(), visible=False)

                    ax3.plot(xvec, yvec)
                    ax3.set_title("Localizations")
                    ax3.set_xlabel("Frames")
                    ax3.set_ylabel("ON")

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

        print("Showing picks...")
        channel = self.get_channel3d("Select Channel")

        removelist = [] # picks to be removed

        if channel is not None:
            n_channels = len(self.locs_paths)
            colors = get_colors(n_channels)

            if channel is (len(self.locs_paths)):
                all_picked_locs = []
                for k in range(len(self.locs_paths)):
                    all_picked_locs.append(self.picked_locs(k))
                if self._picks:
                    params = {}
                    params["t0"] = time.time()
                    i = 0
                    while i < len(self._picks):
                        fig = plt.figure(figsize=(5, 5))
                        fig.canvas.set_window_title("Scatterplot of Pick")
                        pick = self._picks[i]

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
                            locs = stack_arrays(
                                locs, asrecarray=True, usemask=False
                            )
                            ax.scatter(locs["x"], locs["y"], c=colors[l], s=2)

                        tools_dialog = self.window.tools_settings_dialog
                        r = tools_dialog.pick_diameter.value() / 2
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
                        fig.canvas.set_window_title("Scatterplot of Pick")
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
                        r = self.window.tools_settings_dialog.pick_diameter.value() / 2
                        x_min = pick[0] - r
                        x_max = pick[0] + r
                        y_min = pick[1] - r
                        y_max = pick[1] + r
                        ax.scatter(locs["x"], locs["y"], c=colors[channel], s=2)
                        ax.set_xlabel("X [Px]")
                        ax.set_ylabel("Y [Px]")
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                        # plt.axis("equal")

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
        print("Show pick 3D")
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
        # essentially the same as show_pick_3d
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
        """
        Tool to detect clusters with k-means clustering
        """
        print("Analyzing clusters...")
        channel = self.get_channel3d("Show Pick 3D")
        removelist = []
        saved_locs = []
        clustered_locs = []
        pixelsize = self.window.display_settings_dlg.pixelsize.value()

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
                        if hasattr(all_picked_locs[0], "z"):
                            reply = ClsDlg.getParams(
                                all_picked_locs,
                                i,
                                len(self._picks),
                                0,
                                colors,
                                pixelsize,
                            )
                        else:
                            reply = ClsDlg2D.getParams(
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
                    n_clusters, ok = QtWidgets.QInputDialog.getInt(
                        self,
                        "Input Dialog",
                        "Enter inital number of clusters:",
                        10,
                    )

                    for i, pick in enumerate(self._picks):
                        reply = 3
                        while reply == 3:
                            if hasattr(all_picked_locs[0], "z"):
                                reply, nc, l_locs, c_locs = ClsDlg.getParams(
                                    all_picked_locs,
                                    i,
                                    len(self._picks),
                                    n_clusters,
                                    1,
                                    pixelsize,
                                )
                            else:
                                reply, nc, l_locs, c_locs = ClsDlg2D.getParams(
                                    all_picked_locs,
                                    i,
                                    len(self._picks),
                                    n_clusters,
                                    1,
                                )
                            n_clusters = nc

                        if reply == 1:
                            print("Accepted")
                            saved_locs.append(l_locs)
                            clustered_locs.extend(c_locs)
                        elif reply == 2:
                            break
                        else:
                            print("Discard")
                            removelist.append(pick)
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
            pick_props = postprocess.groupprops(out_locs)
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
        channel = self.get_channel("Pick similar")
        if channel is not None:
            locs = self.locs[channel]
            info = self.infos[channel]
            d = self.window.tools_settings_dialog.pick_diameter.value()
            r = d / 2
            std_range = (
                self.window.tools_settings_dialog.pick_similar_range.value()
            )
            index_blocks = self.get_index_blocks(channel)
            n_locs = []
            rmsd = []

            if self._picks:
                removelist = []
                loccount = []
                progress = lib.ProgressDialog(
                    "Counting in picks..", 0, len(self._picks) - 1, self
                )
                progress.set_value(0)
                progress.show()
                for i, pick in enumerate(self._picks):
                    x, y = pick
                    block_locs = postprocess.get_block_locs_at(
                        x, y, index_blocks
                    )
                    pick_locs = lib.locs_at(x, y, block_locs, r)
                    locs = stack_arrays(
                        pick_locs, asrecarray=True, usemask=False
                    )
                    loccount.append(len(locs))
                    progress.set_value(i)

                progress.close()
                fig = plt.figure()
                fig.canvas.set_window_title("Localizations in Picks")
                ax = fig.add_subplot(111)
                ax.set_title("Localizations in Picks ")
                n, bins, patches = ax.hist(
                    loccount, 20, density=True, facecolor="green", alpha=0.75
                )
                ax.set_xlabel("Number of localizations")
                ax.set_ylabel("Counts")
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
        com_x = locs.x.mean()
        com_y = locs.y.mean()
        return np.sqrt(np.mean((locs.x - com_x) ** 2 + (locs.y - com_y) ** 2))

    def index_locs(self, channel):
        """
        Indexes localizations in a grid
        with grid size equal to the pick radius.
        """
        locs = self.locs[channel]
        info = self.infos[channel]
        d = self.window.tools_settings_dialog.pick_diameter.value()
        size = d / 2
        status = lib.StatusDialog("Indexing localizations...", self.window)
        index_blocks = postprocess.get_index_blocks(
            locs, info, size
        )
        status.close()
        self.index_blocks[channel] = index_blocks

    def get_index_blocks(self, channel):
        if self.index_blocks[channel] is None:
            self.index_locs(channel)
        return self.index_blocks[channel]

    @check_picks
    def pick_similar(self):
        if self._pick_shape == "Rectangle":
            raise NotImplementedError(
                "Pick similar not implemented for rectangle picks"
            )
        channel = self.get_channel("Pick similar")
        if channel is not None:
            locs = self.locs[channel]
            info = self.infos[channel]
            d = self.window.tools_settings_dialog.pick_diameter.value()
            r = d / 2
            d2 = d ** 2
            std_range = (
                self.window.tools_settings_dialog.pick_similar_range.value()
            )
            index_blocks = self.get_index_blocks(channel)
            n_locs = []
            rmsd = []
            for i, pick in enumerate(self._picks):
                x, y = pick
                block_locs = postprocess.get_block_locs_at(x, y, index_blocks)
                pick_locs = lib.locs_at(x, y, block_locs, r)
                n_locs.append(len(pick_locs))
                rmsd.append(self.rmsd_at_com(pick_locs))

            mean_n_locs = np.mean(n_locs)
            mean_rmsd = np.mean(rmsd)
            std_n_locs = np.std(n_locs)
            std_rmsd = np.std(rmsd)
            min_n_locs = mean_n_locs - std_range * std_n_locs
            max_n_locs = mean_n_locs + std_range * std_n_locs
            min_rmsd = mean_rmsd - std_range * std_rmsd
            max_rmsd = mean_rmsd + std_range * std_rmsd

            # x, y coordinates of found similar regions:
            x_similar = np.array([_[0] for _ in self._picks])
            y_similar = np.array([_[1] for _ in self._picks])
            # preparations for hex grid search
            x_range = np.arange(d / 2, info[0]["Width"], np.sqrt(3) * d / 2)
            y_range_base = np.arange(d / 2, info[0]["Height"] - d / 2, d)
            y_range_shift = y_range_base + d / 2
            
            locs_temp, size, _, _, block_starts, block_ends, K, L = index_blocks
            locs_x = locs_temp.x
            locs_y = locs_temp.y
            locs_xy = np.stack((locs_x, locs_y))
            x_r = np.uint64(x_range / size)
            y_r1 = np.uint64(y_range_shift / size)
            y_r2 = np.uint64(y_range_base / size)
            status = lib.StatusDialog("Picking similar...", self.window)
            x_similar, y_similar = postprocess.pick_similar(
                    x_range, y_range_shift, y_range_base,
                    min_n_locs, max_n_locs, min_rmsd, max_rmsd, 
                    x_r, y_r1, y_r2,
                    locs_xy, block_starts, block_ends, K, L,        
                    x_similar, y_similar, r, d2,
                )
            similar = list(zip(x_similar, y_similar))
            self._picks = []
            self.add_picks(similar)
            status.close()

    def picked_locs(self, channel, add_group=True):
        """ Returns picked localizations in the specified channel """
        if len(self._picks):
            picked_locs = []
            progress = lib.ProgressDialog(
                "Creating localization list", 0, len(self._picks), self
            )
            progress.set_value(0)
            if self._pick_shape == "Circle":
                d = self.window.tools_settings_dialog.pick_diameter.value()
                r = d / 2

                index_blocks = self.get_index_blocks(channel)

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
                channel_locs = self.locs[channel]
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
            else:
                raise ValueError("Invalid value for pick shape")

            return picked_locs

    def remove_picks(self, position):
        x, y = position
        new_picks = []
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
        self._picks = []
        if len(new_picks) == 0:
            self.update_pick_info_short()
            self.update_scene(picks_only=True)
        else:
            self.add_picks(new_picks)

    def remove_points(self, position):
        self._points = []
        self.update_scene()

    def render_scene(
        self, autoscale=False, use_cache=False, cache=True, viewport=None
    ):
        kwargs = self.get_render_kwargs(viewport=viewport)
        n_channels = len(self.locs)
        if n_channels == 1:
            self.render_single_channel(
                kwargs, autoscale=autoscale, use_cache=use_cache, cache=cache
            )
        else:
            self.render_multi_channel(
                kwargs,
                autoscale=autoscale,
                use_cache=use_cache,
                cache=cache,
            )
        self._bgra[:, :, 3].fill(255)
        Y, X = self._bgra.shape[:2]
        qimage = QtGui.QImage(
            self._bgra.data, X, Y, QtGui.QImage.Format_RGB32
        )
        return qimage

    def render_multi_channel(
        self,
        kwargs,
        autoscale=False,
        locs=None,
        use_cache=False,
        cache=True,
    ):
        if locs is None:
            locs = self.locs

        # Plot each channel
        for i in range(len(locs)):
            if hasattr(locs[i], "z"):
                if self.window.slicer_dialog.slicer_radio_button.isChecked():
                    z_min = self.window.slicer_dialog.slicermin
                    z_max = self.window.slicer_dialog.slicermax
                    in_view = (locs[i].z > z_min) & (
                        locs[i].z <= z_max
                    )
                    locs[i] = locs[i][in_view]
        n_channels = len(locs)
        colors = get_colors(n_channels)
        if use_cache:
            n_locs = self.n_locs
            image = self.image
        else:
            renderings = [render.render(_, **kwargs) for _ in locs]
            n_locs = sum([_[0] for _ in renderings])
            image = np.array([_[1] for _ in renderings])

        if cache:
            self.n_locs = n_locs
            self.image = image

        image = self.scale_contrast(image)
        Y, X = image.shape[1:]
        bgra = np.zeros((Y, X, 4), dtype=np.float32)

        # Color images
        for i in range(len(self.locs)):
            if not self.window.dataset_dialog.auto_colors.isChecked():
                color = self.window.dataset_dialog.colorselection[i]
                color = color.currentText()
                if color in self.window.dataset_dialog.default_colors:
                    colors_array = np.array(
                        self.window.dataset_dialog.default_colors, 
                        dtype=object,
                    )
                    index = np.where(colors_array == color)[0][0]
                    colors[i] = tuple(self.window.dataset_dialog.rgbf[index])
                elif is_hexadecimal(color):
                    colorstring = color.lstrip("#")
                    rgbval = tuple(
                        int(colorstring[i: i + 2], 16) / 255 for i in (0, 2, 4)
                    )
                    colors[i] = rgbval
                else:
                    warning = (
                        "The color selection not recognnised in the channel {}."
                        "  Please choose one of the options provided or type "
                        " the hexadecimal code for your color of choice,  "
                        " starting with '#', e.g.  '#ffcdff' for pink.".format(
                            self.window.dataset_dialog.checks[i].text()
                        )
                    )
                    QtWidgets.QMessageBox.information(self, "Warning", warning)
                    break

            if self.window.dataset_dialog.wbackground.isChecked():
                tempcolor = colors[i]
                inverted = tuple([1 - _ for _ in tempcolor])
                colors[i] = inverted

            iscale = self.window.dataset_dialog.intensitysettings[i].value()
            image[i] = iscale * image[i]
            if not self.window.dataset_dialog.checks[i].isChecked():
                image[i] = 0 * image[i]

        for color, image in zip(colors, image):
            bgra[:, :, 0] += color[2] * image
            bgra[:, :, 1] += color[1] * image
            bgra[:, :, 2] += color[0] * image

        bgra = np.minimum(bgra, 1)
        if self.window.dataset_dialog.wbackground.isChecked():
            bgra = -(bgra - 1)
        self._bgra = self.to_8bit(bgra)
        return self._bgra

    def render_single_channel(
        self, kwargs, autoscale=False, use_cache=False, cache=True,
    ):
        locs = self.locs[0]

        if self.x_render_state:
            locs = self.x_locs
            return self.render_multi_channel(
                kwargs, autoscale=autoscale, locs=locs, use_cache=use_cache
            )            

        if hasattr(locs, "group"):
            locs = [locs[self.group_color == _] for _ in range(N_GROUP_COLORS)]
            return self.render_multi_channel(
                kwargs, autoscale=autoscale, locs=locs, use_cache=use_cache
            )
        if hasattr(locs, "z"):
            if self.window.slicer_dialog.slicer_radio_button.isChecked():
                z_min = self.window.slicer_dialog.slicermin
                z_max = self.window.slicer_dialog.slicermax
                in_view = (locs.z > z_min) & (locs.z <= z_max)
                locs = locs[in_view]

        if use_cache:
            n_locs = self.n_locs
            image = self.image
        else:
            n_locs, image = render.render(locs, **kwargs, info=self.infos[0])
        if cache:
            self.n_locs = n_locs
            self.image = image
        image = self.scale_contrast(image, autoscale=autoscale)
        image = self.to_8bit(image)
        Y, X = image.shape
        cmap = self.window.display_settings_dlg.colormap.currentText()
        cmap = np.uint8(np.round(255 * plt.get_cmap(cmap)(np.arange(256))))
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        if self.window.dataset_dialog.wbackground.isChecked():
            self._bgra = -(self._bgra - 255)
        return self._bgra

    def resizeEvent(self, event):
        self.update_scene()

    def save_picked_locs(self, path, channel):
        locs = self.picked_locs(channel, add_group=True)
        locs = stack_arrays(locs, asrecarray=True, usemask=False)
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
        for i in range(len(self.locs_paths)):
            channel = self.locs_paths[i]
            if i == 0:
                locs = self.picked_locs(self.locs_paths.index(channel))
                locs = stack_arrays(locs, asrecarray=True, usemask=False)
            else:
                templocs = self.picked_locs(self.locs_paths.index(channel))
                templocs = stack_arrays(
                    templocs, asrecarray=True, usemask=False
                )
                locs = np.append(locs, templocs)
        locs = locs.view(np.recarray)
        if locs is not None:
            d = self.window.tools_settings_dialog.pick_diameter.value()
            pick_info = {
                "Generated by:": "Picasso Render",
                "Pick Diameter:": d,
            }
            io.save_locs(path, locs, self.infos[0] + [pick_info])

    def save_pick_properties(self, path, channel):
        picked_locs = self.picked_locs(channel)
        pick_diameter = self.window.tools_settings_dialog.pick_diameter.value()
        r_max = min(pick_diameter, 1)
        max_dark = self.window.info_dialog.max_dark_time.value()
        out_locs = []
        progress = lib.ProgressDialog(
            "Calculating kinetics", 0, len(picked_locs), self
        )
        progress.set_value(0)
        dark = np.empty(len(picked_locs))
        length = np.empty(len(picked_locs))
        no_locs = np.empty(len(picked_locs))
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
        pick_props = postprocess.groupprops(out_locs)

        progress.set_value(n_groups)
        progress.close()
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
        if self._pick_shape == "Circle":
            d = self.window.tools_settings_dialog.pick_diameter.value()
            picks = {
                "Diameter": float(d),
                "Centers": [[float(_[0]), float(_[1])] for _ in self._picks]}
        elif self._pick_shape == "Rectangle":
            w = self.window.tools_settings_dialog.pick_width.value()
            picks = {
                "Width": float(w),
                "Center-Axis-Points": [
                    [[float(s[0]), float(s[1])], [float(e[0]), float(e[1])]] for s, e in self._picks
                ],
            }
        else:
            raise ValueError("Unrecognized pick shape")
        picks["Shape"] = self._pick_shape
        with open(path, "w") as f:
            yaml.dump(picks, f)

    def scale_contrast(self, image, autoscale=False):
        if autoscale:
            if image.ndim == 2:
                max_ = image.max()
            else:
                max_ = min([_.max() for _ in image])
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

    def render_3d(self):
        if hasattr(self.locs[0], "z"):
            if self.z_render is True:
                self.z_render = False
            else:
                self.z_render = True
                mean_z = np.mean(self.locs[0].z)
                std_z = np.std(self.locs[0].z)
                min_z = mean_z - 3 * std_z
                max_z = mean_z + 3 * std_z
                z_step = (max_z - min_z) / N_Z_COLORS
                self.z_color = np.floor((self.locs[0].z - min_z) / z_step)
                z_locs = []
                if not hasattr(self, "z_locs"):
                    pb = lib.ProgressDialog(
                        "Indexing colors", 0, N_Z_COLORS, self
                    )
                    pb.set_value(0)
                    for i in tqdm(range(N_Z_COLORS)):
                        z_locs.append(self.locs[0][self.z_color == i])
                        pb.set_value(i)
                    pb.close()
                    self.z_locs = z_locs
            self.update_scene()

    def render_time(self):
        if self.t_render is True:
            self.t_render = False
        else:
            self.t_render = True

            min_frames = np.min(self.locs[0].frame)
            max_frames = np.max(self.locs[0].frame)
            t_step = (max_frames - min_frames) / N_Z_COLORS
            self.t_color = np.floor((self.locs[0].frame - min_frames) / t_step)
            t_locs = []
            if not hasattr(self, "t_locs"):
                pb = lib.ProgressDialog("Indexing time", 0, N_Z_COLORS, self)
                pb.set_value(0)
                for i in tqdm(range(N_Z_COLORS)):
                    t_locs.append(self.locs[0][self.t_color == i])
                    pb.set_value(i)
                pb.close()
                self.t_locs = t_locs
        self.update_scene()

    def show_legend(self):
        parameter = self.window.display_settings_dlg.parameter.currentText()
        n_colors = self.window.display_settings_dlg.color_step.value()
        min_val = self.window.display_settings_dlg.minimum_render.value()
        max_val = self.window.display_settings_dlg.maximum_render.value()

        colors = get_colors(n_colors)

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
        self.deactivate_property_menu()

        if self.window.display_settings_dlg.render_check.isChecked():
            self.x_render_state = True
            parameter = (
                self.window.display_settings_dlg.parameter.currentText()
            )
            colors = self.window.display_settings_dlg.color_step.value()
            min_val = self.window.display_settings_dlg.minimum_render.value()
            max_val = self.window.display_settings_dlg.maximum_render.value()

            x_step = (max_val - min_val) / colors

            self.x_color = np.floor(
                (self.locs[0][parameter] - min_val) / x_step
            )

            # values above and below will be fixed:
            self.x_color[self.x_color < 0] = 0
            self.x_color[self.x_color > colors] = colors

            x_locs = []

            for cached_entry in self.x_render_cache:
                if cached_entry["parameter"] == parameter:
                    if cached_entry["colors"] == colors:
                        if (cached_entry["min_val"] == min_val) & (
                            cached_entry["max_val"] == max_val
                        ):
                            x_locs = cached_entry["locs"]
                        break

            if x_locs == []:
                pb = lib.ProgressDialog(
                    "Indexing " + parameter, 0, colors, self
                )
                pb.set_value(0)
                for i in tqdm(range(colors + 1)):
                    x_locs.append(self.locs[0][self.x_color == i])
                    pb.set_value(i + 1)
                pb.close()

                entry = {}
                entry["parameter"] = parameter
                entry["colors"] = colors
                entry["locs"] = x_locs
                entry["min_val"] = min_val
                entry["max_val"] = max_val

                # Do not store to many datasets in cache
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

        self.activate_property_menu()

    def activate_property_menu(self):
        self.window.display_settings_dlg.minimum_render.setEnabled(True)
        self.window.display_settings_dlg.maximum_render.setEnabled(True)
        self.window.display_settings_dlg.color_step.setEnabled(True)

    def deactivate_property_menu(self):
        self.window.display_settings_dlg.minimum_render.setEnabled(False)
        self.window.display_settings_dlg.maximum_render.setEnabled(False)
        self.window.display_settings_dlg.color_step.setEnabled(False)

    def set_property(self):

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
        self._mode = action.text()
        self.update_cursor()

    def on_pick_shape_changed(self, pick_shape_index):
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
        current_zoom = self.display_pixels_per_viewport_pixels()
        self.zoom(current_zoom / zoom)

    def sizeHint(self):
        return QtCore.QSize(*self._size_hint)

    def to_8bit(self, image):
        return np.round(255 * image).astype("uint8")

    def to_left(self):
        self.pan_relative(0, 0.8)

    def to_right(self):
        self.pan_relative(0, -0.8)

    def to_up(self):
        self.pan_relative(0.8, 0)

    def to_down(self):
        self.pan_relative(-0.8, 0)

    def show_drift(self):
        channel = self.get_channel("Show drift")
        if channel is not None:
            drift = self._drift[channel]

            if drift is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "Driftfile error",
                    (
                        "No driftfile found."
                        " Nothing to display."
                        " Please perform drift correction first."
                    ),
                )
            else:
                print('Showing Drift')
                self.plot_window = DriftPlotWindow(self)
                if hasattr(self._drift[channel], "z"):
                    self.plot_window.plot_3d(drift)

                else:
                    self.plot_window.plot_2d(drift)

                self.plot_window.show()


    def undrift(self):
        """ Undrifts with rcc. """
        channel = self.get_channel("Undrift")
        if channel is not None:
            info = self.infos[channel]
            n_frames = info[0]["Frames"]
            if n_frames < 1000:
                default_segmentation = int(n_frames / 4)
            else:
                default_segmentation = 1000
            segmentation, ok = QtWidgets.QInputDialog.getInt(
                self, "Undrift by RCC", "Segmentation:", default_segmentation
            )

            if ok:
                locs = self.locs[channel]
                info = self.infos[channel]
                n_segments = render.n_segments(info, segmentation)
                seg_progress = lib.ProgressDialog(
                    "Generating segments", 0, n_segments, self
                )
                n_pairs = int(n_segments * (n_segments - 1) / 2)
                rcc_progress = lib.ProgressDialog(
                    "Correlating image pairs", 0, n_pairs, self
                )
                try:
                    start_time = time.time()
                    drift, _ = postprocess.undrift(
                        locs,
                        info,
                        segmentation,
                        False,
                        seg_progress.set_value,
                        rcc_progress.set_value,
                    )
                    finish_time = time.time()
                    print("RCC drift estimate running time [seconds]: ", 
                        np.round(finish_time-start_time, 1))
                    self.locs[channel] = lib.ensure_sanity(locs, info)
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
        channel = self.get_channel("Undrift from picked")
        if channel is not None:
            self._undrift_from_picked(channel)

    @check_picks
    def undrift_from_picked2d(self):
        channel = self.get_channel("Undrift from picked")
        if channel is not None:
            self._undrift_from_picked2d(channel)

    def _undrift_from_picked_coordinate(
        self, channel, picked_locs, coordinate
    ):
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
        drift_mean = np.ma.average(drift, axis=0, weights=1 / msd)
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
        picked_locs = self.picked_locs(channel)
        status = lib.StatusDialog("Calculating drift...", self)

        drift_x = self._undrift_from_picked_coordinate(
            channel, picked_locs, "x"
        )
        drift_y = self._undrift_from_picked_coordinate(
            channel, picked_locs, "y"
        )

        # Apply drift
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
            self.locs[channel].z -= drift_z[self.locs[channel].frame]
            drift = lib.append_to_rec(drift, drift_z, "z")

        # Cleanup
        self.index_blocks[channel] = None
        self.add_drift(channel, drift)
        status.close()
        self.update_scene()

    def _undrift_from_picked2d(self, channel):
        picked_locs = self.picked_locs(channel)
        status = lib.StatusDialog("Calculating drift...", self)

        drift_x = self._undrift_from_picked_coordinate(
            channel, picked_locs, "x"
        )
        drift_y = self._undrift_from_picked_coordinate(
            channel, picked_locs, "y"
        )

        # Apply drift
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
        channel = self.get_channel("Undo drift")
        if channel is not None:
            self._undo_drift(channel)

    def _undo_drift(self, channel):
        # Todo undo drift for z
        drift = self.currentdrift[channel]
        drift.x = -drift.x
        drift.y = -drift.y

        self.locs[channel].x -= drift.x[self.locs[channel].frame]
        self.locs[channel].y -= drift.y[self.locs[channel].frame]

        if hasattr(drift, "z"):
            drift.z = -drift.z
            self.locs[channel].z -= drift.z[self.locs[channel].frame]

        self.add_drift(channel, drift)
        self.update_scene()

    def add_drift(self, channel, drift):
        import time

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

        # if hasattr(drift, "z"):
        np.savetxt(
            driftfile,
            self._drift[channel],
            # header="dx\tdy\tdz",
            newline="\r\n",
        )
        # else:
        #     np.savetxt(
        #         driftfile,
        #         self._drift[channel],
        #         # header="dx\tdy",
        #         newline="\r\n",
        #     )

    def apply_drift(self):
        # channel = self.get_channel("Undrift")
        # if channel is not None:
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load drift file", filter="*.txt", directory=None)
        if path:
            drift = np.loadtxt(path, delimiter=' ')
            if hasattr(self.locs[0], "z"):
                drift = (drift[:,0], drift[:,1], drift[:,2])
                drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f"), ("z", "f")])
                self.locs[0].x -= drift.x[self.locs[0].frame]
                self.locs[0].y -= drift.y[self.locs[0].frame]
                self.locs[0].z -= drift.z[self.locs[0].frame]
            else:
                drift = (drift[:,0], drift[:,1])
                drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])
                self.locs[0].x -= drift.x[self.locs[0].frame]
                self.locs[0].y -= drift.y[self.locs[0].frame]
            self._drift = [drift]
            self._driftfiles = [path]
            self.currentdrift = [copy.copy(drift)]
            self.index_blocks[0] = None
            self.update_scene()

    def unfold_groups(self):
        if not hasattr(self, "unfold_status"):
            self.unfold_status = "folded"
        if self.unfold_status == "folded":
            if hasattr(self.locs[0], "group"):
                self.locs[0].x += self.locs[0].group * 2
            groups = np.unique(self.locs[0].group)

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
                np.mean(self.locs[0].x)
                + np.max(self.locs[0].x)
                - np.min(self.locs[0].x)
            )
            self.infos[0][0]["Width"] = np.max([self.oldwidth, minwidth])
            self.fit_in_view()
            self.unfold_status = "unfolded"
            self.n_picks = len(self._picks)
            self.update_pick_info_short()
        else:
            self.refold_groups()
            self.clear_picks()

    def unfold_groups_square(self):
        n_square, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Input Dialog",
            "Set number of elements per row and column:",
            100,
        )
        if hasattr(self.locs[0], "group"):

            self.locs[0].x += np.mod(self.locs[0].group, n_square) * 2
            self.locs[0].y += np.floor(self.locs[0].group / n_square) * 2

            mean_x = np.mean(self.locs[0].x)
            mean_y = np.mean(self.locs[0].y)

            self.locs[0].x -= mean_x
            self.locs[0].y -= np.mean(self.locs[0].y)

            offset_x = np.absolute(np.min(self.locs[0].x))
            offset_y = np.absolute(np.min(self.locs[0].y))

            self.locs[0].x += offset_x
            self.locs[0].y += offset_y

            if self._picks:
                if self._pick_shape == "Rectangle":
                    raise NotImplementedError(
                        "Not implemented for rectangle picks"
                    )
                # Also unfold picks
                groups = np.unique(self.locs[0].group)

                shift_x = np.mod(groups, n_square) * 2 - mean_x + offset_x
                shift_y = np.floor(groups / n_square) * 2 - mean_y + offset_y

                for j in range(len(self._picks)):
                    for k in range(len(groups)):
                        x_pick, y_pick = self._picks[j]
                        self._picks.append(
                            (x_pick + shift_x[k], y_pick + shift_y[k])
                        )

                self.n_picks = len(self._picks)
                self.update_pick_info_short()

        # Update width information
        self.infos[0][0]["Height"] = int(np.ceil(np.max(self.locs[0].y)))
        self.infos[0][0]["Width"] = int(np.ceil(np.max(self.locs[0].x)))
        self.fit_in_view()

    def refold_groups(self):
        if hasattr(self.locs[0], "group"):
            self.locs[0].x -= self.locs[0].group * 2
        self.fit_in_view()
        self.infos[0][0]["Width"] = self.oldwidth
        self.unfold_status == "folded"

    def update_cursor(self):
        if self._mode == "Zoom" or self._mode == "Measure":
            self.unsetCursor()
        elif self._mode == "Pick":
            if self._pick_shape == "Circle":
                diameter = (
                    self.window.tools_settings_dialog.pick_diameter.value()
                )
                diameter = self.width() * diameter / self.viewport_width()
                # remote desktop crashes sometimes
                if diameter < 100:
                    pixmap_size = ceil(diameter)
                    pixmap = QtGui.QPixmap(pixmap_size, pixmap_size)
                    pixmap.fill(QtCore.Qt.transparent)
                    painter = QtGui.QPainter(pixmap)
                    painter.setPen(QtGui.QColor("white"))
                    if self.window.dataset_dialog.wbackground.isChecked():
                        painter.setPen(QtGui.QColor("black"))
                    offset = (pixmap_size - diameter) / 2
                    painter.drawEllipse(offset, offset, diameter, diameter)
                    painter.end()
                    cursor = QtGui.QCursor(pixmap)
                    self.setCursor(cursor)
                else:
                    self.unsetCursor()
            elif self._pick_shape == "Rectangle":
                self.unsetCursor()

    def update_pick_info_long(self, info):
        """ Gets called when "Show info below" """
        if len(self._picks) == 0:
            warning = "No picks found.  Please pick first."
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
            N = np.empty(n_picks)
            rmsd = np.empty(n_picks)
            length = np.empty(n_picks)
            dark = np.empty(n_picks)
            has_z = hasattr(picked_locs[0], "z")
            if has_z:
                rmsd_z = np.empty(n_picks)
            new_locs = []
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

            self.window.info_dialog.n_localizations_mean.setText(
                "{:.2f}".format(np.nanmean(N))
            )
            self.window.info_dialog.n_localizations_std.setText(
                "{:.2f}".format(np.nanstd(N))
            )
            self.window.info_dialog.rmsd_mean.setText(
                "{:.2}".format(np.nanmean(rmsd))
            )
            self.window.info_dialog.rmsd_std.setText(
                "{:.2}".format(np.nanstd(rmsd))
            )
            if has_z:
                self.window.info_dialog.rmsd_z_mean.setText(
                    "{:.2f}".format(np.nanmean(rmsd_z))
                )
                self.window.info_dialog.rmsd_z_std.setText(
                    "{:.2f}".format(np.nanstd(rmsd_z))
                )
            pooled_locs = stack_arrays(
                new_locs, usemask=False, asrecarray=True
            )
            fit_result_len = fit_cum_exp(pooled_locs.len)
            fit_result_dark = fit_cum_exp(pooled_locs.dark)
            self.window.info_dialog.length_mean.setText(
                "{:.2f}".format(np.nanmean(length))
            )
            self.window.info_dialog.length_std.setText(
                "{:.2f}".format(np.nanstd(length))
            )
            self.window.info_dialog.dark_mean.setText(
                "{:.2f}".format(np.nanmean(dark))
            )
            self.window.info_dialog.dark_std.setText(
                "{:.2f}".format(np.nanstd(dark))
            )
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
        self.window.info_dialog.n_picks.setText(str(len(self._picks)))

    def update_scene(
        self,
        viewport=None,
        autoscale=False,
        use_cache=False,
        picks_only=False,
    ):
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
        if viewport is None:
            viewport = self.viewport
        return (
            ((viewport[1][0] + viewport[0][0]) / 2),
            ((viewport[1][1] + viewport[0][1]) / 2),
        )

    def viewport_height(self, viewport=None):
        if viewport is None:
            viewport = self.viewport
        return viewport[1][0] - viewport[0][0]

    def viewport_size(self, viewport=None):
        if viewport is None:
            viewport = self.viewport
        return self.viewport_height(viewport), self.viewport_width(viewport)

    def viewport_width(self, viewport=None):
        if viewport is None:
            viewport = self.viewport
        return viewport[1][1] - viewport[0][1]

    def relative_position(self, viewport_center, cursor_position):
        # finds the position of the cursor relative to the current viewport center;
        # normally takes values between 0 and 1.
        rel_pos_x = (cursor_position[0] - viewport_center[1])/self.viewport_width()
        rel_pos_y = (cursor_position[1] - viewport_center[0])/self.viewport_height()
        return rel_pos_x, rel_pos_y

    def zoom(self, factor, cursor_position=None):
        viewport_height, viewport_width = self.viewport_size()
        new_viewport_height = viewport_height * factor
        new_viewport_width = viewport_width * factor

        if cursor_position is not None:
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
            new_viewport_center_y, new_viewport_center_x = self.viewport_center()

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
        self.zoom(1 / ZOOM)

    def zoom_out(self):
        self.zoom(ZOOM)

    def wheelEvent(self, QWheelEvent):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            direction = QWheelEvent.angleDelta().y()
            position = self.map_to_movie(QWheelEvent.pos())
            if direction > 0:
                self.zoom(1 / ZOOM, cursor_position = position)
            else:
                self.zoom(ZOOM, cursor_position = position)

    def show_legend_files(self, state):
        print(state)


class Window(QtWidgets.QMainWindow):
    def __init__(self, plugins_loaded=False):
        super().__init__()
        self.initUI(plugins_loaded)

    def initUI(self, plugins_loaded):
        self.setWindowTitle("Picasso: Render")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setWindowIcon(icon)
        self.view = View(self)
        self.view.setMinimumSize(1, 1)
        self.setCentralWidget(self.view)
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
        self.window_rot = RotationWindow(self)
        
        self.menu_bar = self.menuBar()
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

        file_menu.addSeparator()
        delete_action = file_menu.addAction("Remove all localizations")
        delete_action.triggered.connect(self.remove_locs)

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

        slicer_action = view_menu.addAction("Slice")
        slicer_action.triggered.connect(self.slicer_dialog.initialize)

        rot_win_action = view_menu.addAction("Update rotation window")
        rot_win_action.setShortcut("Ctrl+Shift+R")
        rot_win_action.triggered.connect(self.rot_win)

        view_menu.addAction(info_action)
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

        filter_picks_action = tools_menu.addAction("Filter picks by locs")
        filter_picks_action.triggered.connect(self.view.filter_picks)

        clear_picks_action = tools_menu.addAction("Clear picks")
        clear_picks_action.triggered.connect(self.view.clear_picks)
        clear_picks_action.setShortcut("Ctrl+C")

        pickadd_action = tools_menu.addAction("Subtract pick regions")
        pickadd_action.triggered.connect(self.subtract_picks)

        tools_menu.addSeparator()
        cluster_action = tools_menu.addAction("Cluster in pick (k-means)")
        cluster_action.triggered.connect(self.view.analyze_cluster)

        mask_action = tools_menu.addAction("Mask image")
        mask_action.triggered.connect(self.mask_settings_dialog.init_dialog)
        # Drift operations
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

        # Group related
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

        self.load_user_settings()

        self.dialogs = [
            self.display_settings_dlg,
            self.dataset_dialog,
            self.info_dialog,
            self.mask_settings_dialog,
            self.tools_settings_dialog,
            self.slicer_dialog,
            self.window_rot,
        ]

        # Define 3D entries

        self.actions_3d = [
            plotpick3d_action,
            plotpick3d_iso_action,
            slicer_action,
            undrift_from_picked2d_action,
            rot_win_action
        ]

        for action in self.actions_3d:
            action.setVisible(False)

        # De-select all menus until file is loaded
        self.menus = [view_menu, postprocess_menu, tools_menu]
        for menu in self.menus:
            menu.setDisabled(True)

        if plugins_loaded:
            try:
                for plugin in self.plugins:
                    plugin.execute()    
            except:
                pass        

    def closeEvent(self, event):
        settings = io.load_user_settings()
        settings["Render"][
            "Colormap"
        ] = self.display_settings_dlg.colormap.currentText()
        if self.view.locs_paths != []:
            settings["Render"]["PWD"] = os.path.dirname(
                self.view.locs_paths[0]
            )
        io.save_user_settings(settings)
        QtWidgets.qApp.closeAllWindows()

    def export_current(self):
        try:
            base, ext = os.path.splitext(self.view.locs_paths[0])
        except AttributeError:
            return
        out_path = base + ".png"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", out_path, filter="*.png;;*.tif"
        )
        if path:
            self.view.qimage.save(path)
        self.view.setMinimumSize(1, 1)

    def export_complete(self):
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
                locs = self.view.locs[channel]
                loctxt = locs[["frame", "x", "y"]].copy()
                np.savetxt(
                    path,
                    loctxt,
                    fmt=["%.1i", "%.5f", "%.5f"],
                    newline="\r\n",
                    delimiter="   ",
                )

    def export_txt_nis(self):
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
                locs = self.view.locs[channel]
                if hasattr(locs, "z"):
                    loctxt = locs[
                        ["x", "y", "z", "sx", "bg", "photons", "frame"]
                    ].copy()
                    loctxt = [
                        (
                            row[0] * pixelsize,
                            row[1] * pixelsize,
                            row[2],
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
                locs = self.view.locs[channel]
                if hasattr(locs, "z"):
                    loctxt = locs[["x", "y", "z"]].copy()
                    loctxt = [
                        (1, row[0] * pixelsize, row[1] * pixelsize, row[2])
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
                locs = self.view.locs[channel]
                if hasattr(locs, "z"):
                    locs = locs[["x", "y", "z", "photons", "frame"]].copy()
                    locs.x *= pixelsize
                    locs.y *= pixelsize
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

    def export_txt_imaris(self):
        channel = self.view.get_channel(
            "Save localizations as txt for IMARIS (x,y,z,frame,channel)"
        )
        pixelsize = self.display_settings_dlg.pixelsize.value()
        if channel is not None:
            base, ext = os.path.splitext(self.view.locs_paths[channel])
            out_path = base + ".imaris.txt"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save localizations as txt for IMARIS (x,y,z,frame,channel)",
                out_path,
                filter="*.imaris.txt",
            )
            if path:
                locs = self.view.locs[channel]
                channel = 0
                tempdata_xyz = locs[["x", "y", "z", "frame"]].copy()
                tempdata_xyz["x"] = tempdata_xyz["x"] * pixelsize
                tempdata_xyz["y"] = tempdata_xyz["y"] * pixelsize
                tempdata = np.array(tempdata_xyz.tolist())
                tempdata = np.array(tempdata_xyz.tolist())
                tempdata_channel = np.hstack(
                    (tempdata, np.zeros((tempdata.shape[0], 1)) + channel)
                )
                np.savetxt(
                    path,
                    tempdata_channel,
                    fmt=["%.1f", "%.1f", "%.1f", "%.1f", "%i"],
                    newline="\r\n",
                    delimiter="\t",
                )

    def export_multi(self):
        items = (
            ".txt for FRC (ImageJ)",
            ".txt for NIS",
            ".txt for IMARIS",
            ".xyz for Chimera",
            ".3d for ViSP",
            ".csv for ThunderSTORM",
        )
        item, ok = QtWidgets.QInputDialog.getItem(
            self, "Select Export", "Formats", items, 0, False
        )
        if ok and item:
            if item == ".txt for FRC (ImageJ)":
                self.export_txt()
            elif item == ".txt for NIS":
                self.export_txt_nis()
            elif item == ".txt for IMARIS":
                self.export_txt_imaris()
            elif item == ".xyz for Chimera":
                self.export_xyz_chimera()
            elif item == ".3d for ViSP":
                self.export_3d_visp()
            elif item == ".csv for ThunderSTORM":
                self.export_ts()
            else:
                print("This should never happen")

    def export_ts(self):
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
                locs = self.view.locs[channel]
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
                                row[9],
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
                        # For 3D: id, frame, x[nm], y[nm], z[nm], sigma1 [nm],
                        #  sigma2 [nm], intensity[Photons], offset[photon]
                        # uncertainty_xy [nm], len
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
                                row[9],
                                row[3] * pixelsize,
                                row[4] * pixelsize,
                                row[5],
                                row[6],
                                stddummy,
                                (row[7] + row[8]) / 2 * pixelsize,
                            )
                            for index, row in enumerate(loctxt)
                        ]
                        # For 3D: id, frame, x[nm], y[nm], z[nm], sigma1 [nm],
                        # sigma2 [nm], intensity[Photons], offset[photon],
                        # uncertainty_xy [nm]
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
                        # For 2D: id, frame, x[nm], y[nm], sigma [nm],
                        # intensity[Photons], offset[photon],
                        # uncertainty_xy [nm]
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

    def load_picks(self):
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pick regions", filter="*.yaml"
        )
        if path:
            self.view.load_picks(path)

    def subtract_picks(self):
        if self.view._picks:
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load pick regions", filter="*.yaml"
            )
            if path:
                self.view.subtract_picks(path)
        else:
            warning = "No picks found.  Please pick first."
            QtWidgets.QMessageBox.information(self, "Warning", warning)

    def load_user_settings(self):
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
        cmd, channel, ok = ApplyDialog.getCmd(self)
        if ok:
            input = cmd.split()
            if input[0] == "flip" and len(input) == 3:
                # Distinguis flipping in xy and z
                if "z" in input:
                    print("xyz")
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
                    self.view.locs[channel][var_2] = templocs * pixelsize
                else:
                    var_1 = input[1]
                    var_2 = input[2]
                    templocs = self.view.locs[channel][var_1].copy()
                    self.view.locs[channel][var_1] = self.view.locs[channel][
                        var_2
                    ]
                    self.view.locs[channel][var_2] = templocs

            elif input[0] == "spiral" and len(input) == 3:
                # spiral uses radius and turns
                radius = float(input[1])
                turns = int(input[2])
                maxframe = self.view.infos[channel][0]["Frames"]
                # Todo: at some point save the spiral in the respective channel

                self.view.x_spiral = self.view.locs[channel]["x"].copy()
                self.view.y_spiral = self.view.locs[channel]["y"].copy()

                scale_time = maxframe / (turns * 2 * np.pi)
                scale_x = turns * 2 * np.pi

                x = self.view.locs[channel]["frame"] / scale_time

                self.view.locs[channel]["x"] = (
                    x * np.cos(x)
                ) / scale_x * radius + self.view.locs[channel]["x"]
                self.view.locs[channel]["y"] = (
                    x * np.sin(x)
                ) / scale_x * radius + self.view.locs[channel]["y"]

            elif input[0] == "uspiral":
                self.view.locs[channel]["x"] = self.view.x_spiral
                self.view.locs[channel]["y"] = self.view.y_spiral
                self.display_settings_dlg.render_check.setChecked(False)
            else:
                vars = self.view.locs[channel].dtype.names
                exec(cmd, {k: self.view.locs[channel][k] for k in vars})
            lib.ensure_sanity(
                self.view.locs[channel], self.view.infos[channel]
            )
            self.view.index_blocks[channel] = None
            self.view.update_scene()

    def open_file_dialog(self):
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

    def open_rotated_locs(self):
        pwd = self.pwd
        self.remove_locs()
        if pwd == []:
            path, ext = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add localizations", filter="*.hdf5" 
            )
        else:
            path, ext = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add localizations", directory=pwd, filter="*.hdf5"
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
        self.update_info()

    def remove_group(self):
        channel = self.view.get_channel("Remove group")
        self.view.locs[channel] = lib.remove_from_rec(
            self.view.locs[channel], "group"
        )
        self.view.update_scene()

    def combine_channels(self):
        print("Combine Channels")
        print(self.view.locs_paths)
        for i in range(len(self.view.locs_paths)):
            channel = self.view.locs_paths[i]
            print(channel)
            if i == 0:
                locs = self.view.locs[i]
                locs = stack_arrays(locs, asrecarray=True, usemask=False)
            else:
                templocs = self.view.locs[i]
                templocs = stack_arrays(
                    templocs, asrecarray=True, usemask=False
                )
                # print(locs)
                # print(templocs)
                locs = np.append(locs, templocs)
            self.view.locs[i] = locs

        self.view.update_scene
        print("Channels combined")
        self.view.zoom_in()
        self.view.zoom_out()

    def save_pick_properties(self):
        channel = self.view.save_channel_pickprops("Save localizations")
        if channel is not None:
            if channel is (len(self.view.locs_paths)):
                print("Save all at once.")
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_apicked",
                )
                if ok:
                    for i in tqdm(range(len(self.view.locs_paths))):
                        channel = i
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
        channel = self.view.save_channel_multi("Save localizations")
        if channel is not None:
            if channel is (len(self.view.locs_paths)):
                print("Save all at once.")
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_arender",
                )
                if ok:
                    for i in tqdm(range(len(self.view.locs_paths))):
                        channel = i
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
                        io.save_locs(out_path, self.view.locs[channel], info)

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
                    io.save_locs(path, self.view.locs[channel], info)

    def save_picked_locs(self):
        channel = self.view.save_channel("Save picked localizations")

        if channel is not None:
            if channel is (len(self.view.locs_paths) + 1):
                print("Multichannel")
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
            elif channel is (len(self.view.locs_paths)):
                print("Save all at once")
                for i in range(len(self.view.locs_paths)):
                    channel = i
                    base, ext = os.path.splitext(self.view.locs_paths[channel])
                    out_path = base + "_apicked.hdf5"
                    self.view.save_picked_locs(out_path, channel)
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
        base, ext = os.path.splitext(self.view.locs_paths[0])
        out_path = base + "_picks.yaml"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save pick regions", out_path, filter="*.yaml"
        )
        if path:
            self.view.save_picks(path)

    def update_info(self):
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

    def remove_locs(self):
        for dialog in self.dialogs:
            dialog.close()
        self.menu_bar.clear() #otherwise the menu bar is doubled
        self.initUI(plugins_loaded=True)

    def rot_win(self):
        if len(self.view._picks) == 0:
            raise ValueError("Pick a region to rotate.")
        elif len(self.view._picks) > 1:
            raise ValueError("Pick only one region.")
        self.window_rot.view_rot.load_locs(update_window=True)
        self.window_rot.show()
        self.window_rot.view_rot.update_scene()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.plugins = []

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
