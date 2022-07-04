"""
    gui/render
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for rendering localization images

    :author: Joerg Schnitzbauer & Maximilian Strauss, 2017-2018
    :copyright: Copyright (c) 2017 Jungmann Lab, MPI of Biochemistry
"""
import os
import os.path
import sys
import traceback
from math import ceil
import copy
import time

import lmfit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
import datetime


from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


from scipy.ndimage.filters import gaussian_filter
from numpy.lib.recfunctions import stack_arrays
from PyQt5 import QtCore, QtGui, QtWidgets

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import axes3d
from collections import Counter
from h5py import File
from tqdm import tqdm
import h5py

import colorsys

from .. import imageprocess, io, lib, postprocess, render

from ..ext.bitplane import IMSWRITER

if IMSWRITER:
    from ..ext.bitplane import numpy_to_imaris


DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5
ZOOM = 10 / 7
N_GROUP_COLORS = 8
N_Z_COLORS = 32

matplotlib.rcParams.update({"axes.titlesize": "large"})

try:
    from PyImarisWriter.ImarisWriterCtypes import *
    from PyImarisWriter import PyImarisWriter as PW

    IMSWRITER = True
except ModuleNotFoundError:
    IMSWRITER = False


def tuple_to_float(tup):
    return tuple(float(_) for _ in tup)


def tuple_to_int(tup):
    return tuple(int(_) for _ in tup)


def get_colors(n_channels):
    hues = np.arange(0, 1, 1 / n_channels)
    colors = [tuple_to_float(colorsys.hsv_to_rgb(_, 1, 1)) for _ in hues]
    return colors


def fit_cum_exp(data):
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


# One for plot pick etc
def check_pick(f):
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


# At least twice for pick_similar etc
def check_picks(f):
    def wrapper(*args):
        if len(args[0]._picks) < 2:
            QtWidgets.QMessageBox.information(
                args[0],
                "Pick Error",
                ("No localizations picked." " Please pick at least twice first."),
            )
        else:
            return f(args[0])

    return wrapper


class FloatEdit(QtWidgets.QLineEdit):

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
        dialog = ApplyDialog(parent)
        result = dialog.exec_()
        cmd = dialog.cmd.text()
        channel = dialog.channel.currentIndex()
        return (cmd, channel, result == QtWidgets.QDialog.Accepted)

    def update_vars(self, index):
        vars = self.window.view.locs[index].dtype.names
        self.label.setText(str(vars))


class MergeDialog(QtWidgets.QDialog):
    """
    A class to handle the Merge Dialog:
    Merge openened localization files into one.

    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Localizations")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

    def init_dialog(self, locs, paths, infos):

        if hasattr(locs[0], "z"):
            use_z = True
        else:
            use_z = False

        self.locs = locs
        self.paths = paths
        self.infos = infos

        self.layout.addWidget(QtWidgets.QLabel("Path"), 1, 0)

        if use_z:
            self.layout.addWidget(QtWidgets.QLabel("z-Offset (nm)"), 1, 1)
            self.offsets = []
        else:
            self.offsets = None

        c = 2

        for idx, path in enumerate(paths):
            dir, filename = os.path.split(path)
            self.layout.addWidget(QtWidgets.QLabel(filename), c + idx, 0)

            if use_z:
                sbox = QtWidgets.QSpinBox()
                sbox.setMinimum(-100_000)
                sbox.setMaximum(+100_000)

                self.layout.addWidget(sbox, c + idx, 1)

                self.offsets.append(sbox)

        merge_button = QtWidgets.QPushButton("Merge")

        self.layout.addWidget(merge_button, c + idx + 1, 1)

        merge_button.clicked.connect(self.merge_action)

        self.show()

    def merge_action(self):
        channel = 0

        if self.offsets is not None:
            locs = []
            parsed_offsets = [_.value() for _ in self.offsets]
            for idx, _ in enumerate(self.locs):
                l = _.copy()
                l["z"] += parsed_offsets[idx]
                locs.append(l)
        else:
            locs = self.locs
            parsed_offsets = None

        merged_locs = stack_arrays(locs, asrecarray=True, usemask=False)

        base, ext = os.path.splitext(self.paths[channel])
        out_path = base + "_merged.hdf5"
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save merged localizations", out_path, filter="*.hdf5"
        )
        if path:
            info = self.infos[channel] + [{"Generated by": "Picasso Render : Merge"}]
            merge_info = {
                "Paths": self.paths,
                "Offsets": parsed_offsets,
            }
            info.append(merge_info)
            io.save_locs(path, merged_locs, info)


class DatasetDialog(QtWidgets.QDialog):
    """
    A class to handle the Dataset Dialog:
    Tick and Untick, set colors and set relative intensity in display

    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Datasets")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.checks = []
        self.closebuttons = []
        self.colorselection = []
        self.colordisp_all = []
        self.intensitysettings = []
        self.setLayout(self.layout)
        self.wbackground = QtWidgets.QCheckBox("White background")
        self.layout.addWidget(self.wbackground, 0, 3)
        self.layout.addWidget(QtWidgets.QLabel("Path"), 1, 0)
        self.layout.addWidget(QtWidgets.QLabel("Color"), 1, 1)
        self.layout.addWidget(QtWidgets.QLabel("#"), 1, 2)
        self.layout.addWidget(QtWidgets.QLabel("Rel. Intensity"), 1, 3)
        self.layout.addWidget(QtWidgets.QLabel("Close"), 1, 4)
        self.wbackground.stateChanged.connect(self.update_viewport)

    def add_entry(self, path):
        c = QtWidgets.QCheckBox(path)
        p = QtWidgets.QPushButton("x")
        currentline = len(self.layout)
        p.setObjectName(str(currentline))

        colordrop = QtWidgets.QComboBox(self)
        colordrop.setEditable(True)
        colordrop.lineEdit().setMaxLength(7)

        # Add default colors
        for default_color in [
            "auto",
            "red",
            "green",
            "blue",
            "gray",
            "cyan",
            "magenta",
            "yellow",
        ]:
            colordrop.addItem(default_color)

        intensity = QtWidgets.QSpinBox(self)
        intensity.setValue(1)
        colordisp = QtWidgets.QLabel("      ")

        palette = colordisp.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        colordisp.setAutoFillBackground(True)
        colordisp.setPalette(palette)

        self.layout.addWidget(c, currentline, 0)
        self.layout.addWidget(colordrop, currentline, 1)
        self.layout.addWidget(colordisp, currentline, 2)
        self.layout.addWidget(intensity, currentline, 3)
        self.layout.addWidget(p, currentline, 4)

        self.intensitysettings.append(intensity)
        self.colorselection.append(colordrop)
        self.colordisp_all.append(colordisp)

        self.checks.append(c)
        self.checks[-1].setChecked(True)
        self.checks[-1].stateChanged.connect(self.update_viewport)
        self.colorselection[-1].currentIndexChanged.connect(self.update_viewport)
        index = len(self.colorselection)
        self.colorselection[-1].currentIndexChanged.connect(
            lambda: self.set_color(index - 1)
        )
        self.intensitysettings[-1].valueChanged.connect(self.update_viewport)

        # update auto colors
        n_channels = len(self.checks)
        colors = get_colors(n_channels)
        for n in range(n_channels):
            palette = self.colordisp_all[n].palette()
            palette.setColor(
                QtGui.QPalette.Window,
                QtGui.QColor.fromRgbF(colors[n][0], colors[n][1], colors[n][2], 1),
            )
            self.colordisp_all[n].setPalette(palette)

        self.closebuttons.append(p)
        p.clicked.connect(self.close_file)

    def close_file(self):
        # TODO call close routine
        raise NotImplementedError("Closing not implemented yet.")
        # print(self.sender.objectName())

    def update_viewport(self):
        if self.window.view.viewport:
            self.window.view.update_scene()

    def set_color(self, n):
        palette = self.colordisp_all[n].palette()
        selectedcolor = self.colorselection[n].currentText()
        if selectedcolor == "auto":
            n_channels = len(self.checks)
            colors = get_colors(n_channels)
            palette.setColor(
                QtGui.QPalette.Window,
                QtGui.QColor.fromRgbF(colors[n][0], colors[n][1], colors[n][2], 1),
            )
        else:
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor(selectedcolor))
        self.colordisp_all[n].setPalette(palette)


class PlotDialog(QtWidgets.QDialog):
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
            colors[colors > np.mean(locs["z"]) + 3 * np.std(locs["z"])] = np.mean(
                locs["z"]
            ) + 3 * np.std(locs["z"])
            colors[colors < np.mean(locs["z"]) - 3 * np.std(locs["z"])] = np.mean(
                locs["z"]
            ) - 3 * np.std(locs["z"])
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
    def getParams(all_picked_locs, current, length, mode, color_sys, pixelsize):

        dialog = PlotDialog(None)
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
            colors[colors > np.mean(locs["z"]) + 3 * np.std(locs["z"])] = np.mean(
                locs["z"]
            ) + 3 * np.std(locs["z"])
            colors[colors < np.mean(locs["z"]) - 3 * np.std(locs["z"])] = np.mean(
                locs["z"]
            ) - 3 * np.std(locs["z"])

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
        self.layout_grid.addWidget(QtWidgets.QLabel(str(x_mean)), self.n_lines, 0, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel(str(y_mean)), self.n_lines, 1, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel(str(z_mean)), self.n_lines, 2, 1, 1)
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
    def getParams(all_picked_locs, current, length, n_clusters, color_sys, pixelsize):

        dialog = ClsDlg(None)

        dialog.start_clusters = n_clusters
        dialog.n_clusters_spin.setValue(n_clusters)

        fig = dialog.figure
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        dialog.label.setText(
            "3D Scatterplot of Pick " + str(current + 1) + "  of: " + str(length) + "."
        )

        print("Mode 1")
        locs = all_picked_locs[current]
        locs = stack_arrays(locs, asrecarray=True, usemask=False)

        est = KMeans(n_clusters=n_clusters)

        scaled_locs = lib.append_to_rec(locs, locs["x"] * pixelsize, "x_scaled")
        scaled_locs = lib.append_to_rec(scaled_locs, locs["y"] * pixelsize, "y_scaled")

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
            l_locs_new_group["group"] * 10**power + l_locs_new_group["cluster"]
        )

        # Combine clustered locs
        clustered_locs = []
        for element in np.unique(labels):
            if element != 0:
                clustered_locs.append(l_locs_new_group[l_locs["cluster"] == element])

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
        self.layout_grid.addWidget(QtWidgets.QLabel(str(x_mean)), self.n_lines, 0, 1, 1)
        self.layout_grid.addWidget(QtWidgets.QLabel(str(y_mean)), self.n_lines, 1, 1, 1)
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
            "2D Scatterplot of Pick " + str(current + 1) + "  of: " + str(length) + "."
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
            l_locs_new_group["group"] * 10**power + l_locs_new_group["cluster"]
        )

        # Combine clustered locs
        clustered_locs = []
        for element in np.unique(labels):
            if element != 0:
                clustered_locs.append(l_locs_new_group[l_locs["cluster"] == element])

        return (
            dialog.result,
            dialog.n_clusters_spin.value(),
            l_locs,
            clustered_locs,
        )


class LinkDialog(QtWidgets.QDialog):
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

    # static method to create the dialog and return input
    @staticmethod
    def getParams(parent=None):
        dialog = LinkDialog(parent)
        result = dialog.exec_()
        return (
            dialog.max_distance.value(),
            dialog.max_dark_time.value(),
            result == QtWidgets.QDialog.Accepted,
        )


class DbscanDialog(QtWidgets.QDialog):
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
        grid.addWidget(QtWidgets.QLabel("Min. density:"), 1, 0)
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

    # static method to create the dialog and return input
    @staticmethod
    def getParams(parent=None):
        dialog = DbscanDialog(parent)
        result = dialog.exec_()
        return (
            dialog.radius.value(),
            dialog.density.value(),
            result == QtWidgets.QDialog.Accepted,
        )


class HdbscanDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter parameters")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Min. cluster:"), 0, 0)
        self.min_cluster = QtWidgets.QSpinBox()
        self.min_cluster.setRange(0, 1e6)
        self.min_cluster.setValue(10)
        grid.addWidget(self.min_cluster, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Min. samples:"), 1, 0)
        self.min_samples = QtWidgets.QSpinBox()
        self.min_samples.setRange(0, 1e6)
        self.min_samples.setValue(10)
        grid.addWidget(self.min_samples, 1, 1)
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

    # static method to create the dialog and return input
    @staticmethod
    def getParams(parent=None):
        dialog = HdbscanDialog(parent)
        result = dialog.exec_()
        return (
            dialog.min_cluster.value(),
            dialog.min_samples.value(),
            result == QtWidgets.QDialog.Accepted,
        )


class DriftPlotWindow(QtWidgets.QTabWidget):
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
            color=list(plt.rcParams["axes.prop_cycle"])[2]["color"],
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
            color=list(plt.rcParams["axes.prop_cycle"])[2]["color"],
        )

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        self.canvas.draw()


class InfoDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Info")
        self.setModal(False)
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
        compute_pick_info_button.clicked.connect(self.window.view.update_pick_info_long)
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
        self.picks_grid.addWidget(QtWidgets.QLabel("Influx rate (1/frames):"), row, 0)
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
        self.picks_grid.addWidget(pick_hists, self.picks_grid.rowCount(), 0, 1, 3)

    def calculate_nena_lp(self):
        channel = self.window.view.get_channel("Calculate NeNA precision")
        if channel is not None:
            locs = self.window.view.locs[channel]
            info = self.window.view.infos[channel]
            self.nena_button.setParent(None)
            self.movie_grid.removeWidget(self.nena_button)
            progress = lib.ProgressDialog("Calculating NeNA precision", 0, 100, self)
            result_lp = postprocess.nena(locs, info, progress.set_value)
            self.nena_label = QtWidgets.QLabel()
            self.movie_grid.addWidget(self.nena_label, 1, 1)
            self.nena_result, lp = result_lp
            self.nena_label.setText("{:.3} pixel".format(lp))
            show_plot_button = QtWidgets.QPushButton("Show plot")
            self.movie_grid.addWidget(
                show_plot_button, self.movie_grid.rowCount() - 1, 2
            )
            # Nena
            self.nena_window = NenaPlotWindow(self)
            self.nena_window.plot(self.nena_result)
            show_plot_button.clicked.connect(self.nena_window.show)

    def calibrate_influx(self):
        influx = 1 / self.pick_info["pooled dark"] / self.units_per_pick.value()
        self.influx_rate.setValue(influx)
        self.update_n_units()

    def calculate_n_units(self, dark):
        influx = self.influx_rate.value()
        return 1 / (influx * dark)

    def update_n_units(self):
        n_units = self.calculate_n_units(self.pick_info["dark"])
        self.n_units_mean.setText("{:,.2f}".format(np.mean(n_units)))
        self.n_units_std.setText("{:,.2f}".format(np.std(n_units)))


class NenaPlotWindow(QtWidgets.QTabWidget):
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

            if self.mask_tresh.value() == self.thresh and self.cached_thresh == 1:
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
            np.floor((locs["x"] - self.x_min) / (self.x_max - self.x_min) * steps_x) - 1
        )
        y_ind = (
            np.floor((locs["y"] - self.y_min) / (self.y_max - self.y_min) * steps_y) - 1
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
            origin="lower",
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
            info = self.infos[channel] + [{"Generated by": "Picasso Render : Mask in "}]
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
            info = self.infos[channel] + [{"Generated by": "Picasso Render : Mask out"}]
            clusterfilter_info = {
                "Oversampling": self.oversampling,
                "Blur": self.blur,
                "Threshold": self.tresh,
            }
            info.append(clusterfilter_info)
            io.save_locs(path, self.index_locs_out, info)


class PickToolCircleSettings(QtWidgets.QWidget):
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
        self.grid.addWidget(QtWidgets.QLabel("Pick similar +/- range (std)"), 1, 0)
        self.pick_similar_range = QtWidgets.QDoubleSpinBox()
        self.pick_similar_range.setRange(0, 100000)
        self.pick_similar_range.setValue(2)
        self.pick_similar_range.setSingleStep(0.1)
        self.pick_similar_range.setDecimals(1)
        self.grid.addWidget(self.pick_similar_range, 1, 1)


class PickToolRectangleSettings(QtWidgets.QWidget):
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

    def on_pick_dimension_changed(self, *args):
        self.window.view.index_blocks = [None for _ in self.window.view.index_blocks]
        self.update_scene_with_cache()

    def update_scene_with_cache(self, *args):
        self.window.view.update_scene(use_cache=True)


class DisplaySettingsDialog(QtWidgets.QDialog):
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
        self.dynamic_oversampling.toggled.connect(self.set_dynamic_oversampling)
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
        self.colormap.addItems(
            sorted(["hot", "viridis", "inferno", "plasma", "magma", "gray"])
        )
        contrast_grid.addWidget(self.colormap, 2, 1)
        self.colormap.currentIndexChanged.connect(self.update_scene)

        contrast_grid.addWidget(QtWidgets.QLabel("Lock Contrast:"), 3, 0)
        self.lock_contrast = QtWidgets.QCheckBox()
        contrast_grid.addWidget(self.lock_contrast, 3, 1)
        self.lock_contrast.stateChanged.connect(self.locked_contrast)

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
        gaussian_button = QtWidgets.QRadioButton("Individual Localization Precision")
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
        blur_grid.addWidget(QtWidgets.QLabel("Min. Blur (cam. pixel):"), 5, 0, 1, 1)
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
        # Scale bar
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
        self.color_step.valueChanged.connect(self.window.view.activate_render_property)
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
        contrast_factor = (self._oversampling / value) ** 2
        self._oversampling = value
        self.silent_minimum_update(contrast_factor * self.minimum.value())
        self.silent_maximum_update(contrast_factor * self.maximum.value())
        if not self._silent_oversampling_update:
            self.dynamic_oversampling.setChecked(False)
            self.window.view.update_scene()

    def on_zoom_changed(self, value):
        self.window.view.set_zoom(value)

    def set_oversampling_silently(self, oversampling):
        if not self.lock_contrast.isChecked():
            self._silent_oversampling_update = True
            self.oversampling.setValue(oversampling)
            self._silent_oversampling_update = False

    def set_zoom_silently(self, zoom):
        self.zoom.blockSignals(True)
        self.zoom.setValue(zoom)
        self.zoom.blockSignals(False)

    def silent_minimum_update(self, value):
        self.minimum.blockSignals(True)
        self.minimum.setValue(value)
        self.minimum.blockSignals(False)

    def silent_maximum_update(self, value):
        self.maximum.blockSignals(True)
        self.maximum.setValue(value)
        self.maximum.blockSignals(False)

    def render_scene(self, *args, **kwargs):
        self.window.view.update_scene()

    def set_dynamic_oversampling(self, state):
        if not self.lock_contrast.isChecked():
            if state:
                self.window.view.update_scene()

    def update_scene(self, *args, **kwargs):
        self.window.view.update_scene(use_cache=True)

    def locked_contrast(self, *args, **kwargs):
        if self.lock_contrast.isChecked():
            self.maximum.setEnabled(False)
            self.minimum.setEnabled(False)
            self.oversampling.setEnabled(False)
            self.dynamic_oversampling.setEnabled(False)
            self.dynamic_oversampling.setChecked(False)
        else:
            self.maximum.setEnabled(True)
            self.minimum.setEnabled(True)
            self.oversampling.setEnabled(True)
            self.dynamic_oversampling.setEnabled(True)
            self.dynamic_oversampling.setChecked(True)


class SlicerDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("3D Slicer ")
        self.setModal(False)
        vbox = QtWidgets.QVBoxLayout(self)
        slicer_groupbox = QtWidgets.QGroupBox("Slicer Settings")

        vbox.addWidget(slicer_groupbox)
        slicer_grid = QtWidgets.QGridLayout(slicer_groupbox)
        slicer_grid.addWidget(QtWidgets.QLabel("Thickness of Slice [nm]:"), 0, 0)
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

        self.slicerRadioButton = QtWidgets.QCheckBox("Slice Dataset")
        self.slicerRadioButton.stateChanged.connect(self.toggle_slicer)

        self.zcoord = []
        self.seperateCheck = QtWidgets.QCheckBox("Export channels separate")
        self.fullCheck = QtWidgets.QCheckBox("Export full image")
        self.exportButton = QtWidgets.QPushButton("Export Slices")
        self.exportButton.setAutoDefault(False)

        self.exportButton.clicked.connect(self.exportStack)

        slicer_grid.addWidget(self.canvas, 2, 0, 1, 2)
        slicer_grid.addWidget(self.slicerRadioButton, 3, 0)
        slicer_grid.addWidget(self.seperateCheck, 4, 0)
        slicer_grid.addWidget(self.fullCheck, 5, 0)
        slicer_grid.addWidget(self.exportButton, 6, 0)

    def initialize(self):
        self.calculate_histogram()
        self.show()

    def calculate_histogram(self):
        slice = self.pick_slice.value()
        ax = self.figure.add_subplot(111)
        plt.cla()
        n_channels = len(self.zcoord)

        self.colors = get_colors(n_channels)

        self.bins = np.arange(
            np.amin(np.hstack(self.zcoord)),
            np.amax(np.hstack(self.zcoord)),
            slice,
        )
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

        self.slicer_cache = {}

    def on_pick_slice_changed(self):
        self.slicer_cache = {}
        if len(self.bins) < 3:  # in case there should be only 1 bin
            self.calculate_histogram()
        else:
            self.calculate_histogram()
            self.sl.setValue(len(self.bins) / 2)
            self.on_slice_position_changed(self.sl.value())

    def toggle_slicer(self):
        self.window.view.update_scene()

    def on_slice_position_changed(self, position):
        for i in range(len(self.zcoord)):
            for patch in self.patches[i]:
                patch.set_facecolor(self.colors[i])
            self.patches[i][position].set_facecolor("black")

        self.slicerposition = position
        self.canvas.draw()
        self.slicermin = self.bins[position]
        self.slicermax = self.bins[position + 1]
        print(
            "Minimum: "
            + str(self.slicermin)
            + " nm, Maxmimum: "
            + str(self.slicermax)
            + " nm"
        )
        self.window.view.update_scene_slicer()

    def exportStack(self):
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

            if self.seperateCheck.isChecked():
                # Uncheck all
                for checks in self.window.dataset_dialog.checks:
                    checks.setChecked(False)
                for j in range(len(self.window.view.locs)):
                    self.window.dataset_dialog.checks[j].setChecked(True)

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
                            + "_CH"
                            + "{num:03d}".format(num=j + 1)
                            + ".tif"
                        )
                        if self.fullCheck.isChecked():
                            movie_height, movie_width = self.window.view.movie_size()
                            viewport = [(0, 0), (movie_height, movie_width)]
                            qimage = self.window.view.render_scene(
                                cache=False, viewport=viewport
                            )
                            gray = qimage.convertToFormat(QtGui.QImage.Format_RGB16)
                        else:
                            gray = self.window.view.qimage.convertToFormat(
                                QtGui.QImage.Format_RGB16
                            )
                        gray.save(out_path)
                        progress.set_value(i)
                    progress.close()
                    self.window.dataset_dialog.checks[j].setChecked(False)
                for checks in self.window.dataset_dialog.checks:
                    checks.setChecked(True)
            else:
                progress = lib.ProgressDialog(
                    "Exporting slices..", 0, self.sl.maximum(), self
                )
                progress.set_value(0)
                progress.show()

                for i in tqdm(range(self.sl.maximum() + 1)):
                    self.sl.setValue(i)
                    print("Slide: " + str(i))
                    out_path = (
                        base + "_Z" + "{num:03d}".format(num=i) + "_CH001" + ".tif"
                    )
                    if self.fullCheck.isChecked():
                        movie_height, movie_width = self.window.view.movie_size()
                        viewport = [(0, 0), (movie_height, movie_width)]
                        qimage = self.window.view.render_scene(
                            cache=False, viewport=viewport
                        )
                        qimage.save(out_path)
                    else:
                        self.window.view.qimage.save(out_path)
                    progress.set_value(i)
                progress.close()


class View(QtWidgets.QLabel):
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
        self.rubberband = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self.rubberband.setStyleSheet("selection-background-color: white")
        self.window = window
        self._pixmap = None
        self.locs = []
        self.infos = []
        self.locs_paths = []
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

    def is_consecutive(l):
        setl = set(l)
        return len(l) == len(setl) and setl == set(range(min(l), max(l) + 1))

    def add(self, path, render=True):
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

        self.locs.append(locs)
        self.infos.append(info)
        self.locs_paths.append(path)
        self.index_blocks.append(None)

        drift = None
        # Try to load a driftfile:
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
                        drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])
                except Exception as e:
                    print(e)
                    # drift already initialized before
                    pass

        self._drift.append(drift)
        self._driftfiles.append(None)
        self.currentdrift.append(None)
        if len(self.locs) == 1:
            self.median_lp = np.mean([np.median(locs.lpx), np.median(locs.lpy)])
            if hasattr(locs, "group"):
                groups = np.unique(locs.group)
                groupcopy = locs.group.copy()
                # check if groups are consecutive
                if set(groups) == set(range(min(groups), max(groups) + 1)):
                    groupcopy = locs.group.copy()
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
                    groupcopy = locs.group.copy()
                    for i in range(len(groups)):
                        groupcopy[locs.group == groups[i]] = i
                np.random.shuffle(groups)
                groups %= N_GROUP_COLORS
                self.group_color = groups[groupcopy]
            if render:
                self.fit_in_view(autoscale=True)
        else:
            if render:
                self.update_scene()

        self.window.display_settings_dlg.parameter.addItems(locs.dtype.names)

        if hasattr(locs, "z"):
            self.window.slicer_dialog.zcoord.append(locs.z)
            # unlock 3D settings
            for action in self.window.actions_3d:
                action.setVisible(True)

        for menu in self.window.menus:
            menu.setDisabled(False)
        self.window.mask_settings_dialog.locs.append(
            locs
        )  # TODO: replace at some point, not very efficient
        self.window.mask_settings_dialog.paths.append(path)
        self.window.mask_settings_dialog.infos.append(info)
        os.chdir(os.path.dirname(path))
        self.window.dataset_dialog.add_entry(path)
        self.window.setWindowTitle(
            "Picasso: Render. File: {}".format(os.path.basename(path))
        )

    def add_multiple(self, paths):
        fit_in_view = len(self.locs) == 0
        paths = sorted(paths)
        for path in paths:
            self.add(path, render=False)
        if len(self.locs):  # In case loading was not succesful.
            if fit_in_view:
                self.fit_in_view(autoscale=True)
            else:
                self.update_scene()

    def add_pick(self, position, update_scene=True):
        self._picks.append(position)
        self.update_pick_info_short()
        if update_scene:
            self.update_scene(picks_only=True)

    def add_point(self, position, update_scene=True):
        self._points.append(position)
        if update_scene:
            self.update_scene(points_only=True)

    def add_picks(self, positions):
        for position in positions:
            self.add_pick(position, update_scene=False)
        self.update_scene(picks_only=True)

    def adjust_viewport_to_view(self, viewport):
        """
        Adds space to a desired viewport
        so that it matches the window aspect ratio.
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
        if len(self._picks) > 0:
            shift = self.shift_from_picked()
            print("Shift {}".format(shift))
            sp = lib.ProgressDialog("Shifting channels", 0, len(self.locs), self)
            sp.set_value(0)
            for i, locs_ in enumerate(self.locs):
                locs_.y -= shift[0][i]
                locs_.x -= shift[1][i]
                if len(shift) == 3:
                    locs_.z -= shift[2][i]
                # Cleanup
                self.index_blocks[i] = None
                sp.set_value(i + 1)

            self.update_scene()
        else:
            max_iterations = 4
            iteration = 0
            convergence = 0.001  # Thhat is 0.001 pixels ~0.13nm
            shift_x = []
            shift_y = []
            shift_z = []
            display = False

            progress = lib.ProgressDialog("Aligning images..", 0, max_iterations, self)
            progress.show()
            progress.set_value(0)

            for iteration in range(max_iterations):
                completed = "True"
                progress.set_value(iteration)
                shift = self.shift_from_rcc()
                sp = lib.ProgressDialog("Shifting channels", 0, len(self.locs), self)
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
            # Plot shift etc
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
        channel = self.get_channel()
        picked_locs = self.picked_locs(channel, add_group=False)
        out_locs = []
        r_max = 2 * max(
            self.infos[channel][0]["Height"], self.infos[channel][0]["Width"]
        )
        max_dark = self.infos[channel][0]["Frames"]
        progress = lib.ProgressDialog(
            "Combining localizations in picks", 0, len(picked_locs), self
        )
        progress.set_value(0)
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
        self.locs[channel] = stack_arrays(out_locs, asrecarray=True, usemask=False)

        if hasattr(self.locs[channel], "group"):
            groups = np.unique(self.locs[channel].group)
            # In case a group is missing
            groups = np.arange(np.max(groups) + 1)
            np.random.shuffle(groups)
            groups %= N_GROUP_COLORS
            self.group_color = groups[self.locs[channel].group]

        self.update_scene()

    def link(self):
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
        radius, min_density, ok = DbscanDialog.getParams()
        if ok:
            status = lib.StatusDialog("Applying DBSCAN. This may take a while..", self)

            for locs_path in self.locs_paths:
                locs, locs_info = io.load_locs(locs_path)
                clusters, locs = postprocess.dbscan(locs, radius, min_density)
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
                print(
                    "Clustering executed. Results are saved in: \n"
                    + base
                    + "_dbscan.hdf5"
                    + "\n"
                    + base
                    + "_dbclusters.hdf5"
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "DBSCAN",
                    "Clustering executed. Results are saved in: \n"
                    + base
                    + "_dbscan.hdf5"
                    + "\n"
                    + base
                    + "_dbclusters.hdf5",
                )

            status.close()

    def hdbscan(self):
        min_cluster, min_samples, ok = HdbscanDialog.getParams()
        if ok:
            status = lib.StatusDialog("Applying HDBSCAN. This may take a while..", self)
            for locs_path in self.locs_paths:
                locs, locs_info = io.load_locs(locs_path)
                clusters, locs = postprocess.hdbscan(locs, min_cluster, min_samples)
                base, ext = os.path.splitext(locs_path)
                hdbscan_info = {
                    "Generated by": "Picasso HDBSCAN",
                    "Min. cluster": min_cluster,
                    "Min. samples": min_samples,
                }
                locs_info.append(hdbscan_info)
                io.save_locs(base + "_hdbscan.hdf5", locs, locs_info)
                with File(base + "_hdbclusters.hdf5", "w") as clusters_file:
                    clusters_file.create_dataset("clusters", data=clusters)
                print(
                    "Clustering executed. Results are saved in: \n"
                    + base
                    + "_hdbscan.hdf5"
                    + "\n"
                    + base
                    + "_hdbclusters.hdf5"
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "HDBSCAN",
                    "Clustering executed. Results are saved in: \n"
                    + base
                    + "_hdbscan.hdf5"
                    + "\n"
                    + base
                    + "_hdbclusters.hdf5",
                )

            status.close()

    def shifts_from_picked_coordinate(self, locs, coordinate):
        """
        Calculates the shift from each channel
        to each other along a given coordinate.
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
                d[i, j] = np.nanmean([cj - ci for ci, cj in zip(coms[i], coms[j])])
        return d

    def shift_from_picked(self):
        """
        Used by align. For each pick, calculate the center
        of mass and does rcc based on shifts
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
        Used by align. Estimates image shifts
        based on image correlation.
        """
        n_channels = len(self.locs)
        rp = lib.ProgressDialog("Rendering images", 0, n_channels, self)
        rp.set_value(0)
        images = []
        for i, (locs_, info_) in enumerate(zip(self.locs, self.infos)):
            _, image = render.render(locs_, info_, blur_method="smooth")
            images.append(image)
            rp.set_value(i + 1)
        n_pairs = int(n_channels * (n_channels - 1) / 2)
        rc = lib.ProgressDialog("Correlating image pairs", 0, n_pairs, self)
        return imageprocess.rcc(images, callback=rc.set_value)

    @check_pick
    def clear_picks(self):
        self._picks = []
        self.update_scene(picks_only=True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def get_pick_rectangle_corners(self, start_x, start_y, end_x, end_y, width):
        drawn_x = end_x - start_x
        if drawn_x == 0:
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
        X, Y = self.get_pick_rectangle_corners(start_x, start_y, end_x, end_y, width)
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
        image = image.copy()
        t_dialog = self.window.tools_settings_dialog
        if self._pick_shape == "Circle":
            d = t_dialog.pick_diameter.value()
            d *= self.width() / self.viewport_width()
            # d = int(round(d))
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QColor("yellow"))
            for i, pick in enumerate(self._picks):
                cx, cy = self.map_to_view(*pick)
                painter.drawEllipse(cx - d / 2, cy - d / 2, d, d)
                if t_dialog.pick_annotation.isChecked():
                    painter.drawText(cx + d / 2, cy + d / 2, str(i))
            painter.end()
        elif self._pick_shape == "Rectangle":
            w = t_dialog.pick_width.value()
            w *= self.width() / self.viewport_width()
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QColor("yellow"))
            for i, pick in enumerate(self._picks):
                start_x, start_y = self.map_to_view(*pick[0])
                end_x, end_y = self.map_to_view(*pick[1])
                painter.drawLine(start_x, start_y, end_x, end_y)
                polygon, most_right = self.get_pick_rectangle_polygon(
                    start_x, start_y, end_x, end_y, w, return_most_right=True
                )
                painter.drawPolygon(polygon)
                if t_dialog.pick_annotation.isChecked():
                    painter.drawText(*most_right, str(i))
            painter.end()
        return image

    def draw_rectangle_pick_ongoing(self, image):
        """Draws an ongoing rectangle pick onto the image"""
        image = image.copy()
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("green"))
        painter.drawLine(
            self.rectangle_pick_start_x,
            self.rectangle_pick_start_y,
            self.rectangle_pick_current_x,
            self.rectangle_pick_current_y,
        )
        w = self.window.tools_settings_dialog.pick_width.value()
        w *= self.width() / self.viewport_width()
        polygon = self.get_pick_rectangle_polygon(
            self.rectangle_pick_start_x,
            self.rectangle_pick_start_y,
            self.rectangle_pick_current_x,
            self.rectangle_pick_current_y,
            w,
        )
        painter.drawPolygon(polygon)
        painter.end()
        return image

    def draw_points(self, image):
        image = image.copy()
        d = 20
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("yellow"))
        cx = []
        cy = []
        ox = []
        oy = []
        oldpoint = []
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
        for point in self._points:
            if oldpoint != []:
                ox, oy = self.map_to_view(*oldpoint)
            cx, cy = self.map_to_view(*point)
            painter.drawPoint(cx, cy)
            painter.drawLine(cx, cy, cx + d / 2, cy)
            painter.drawLine(cx, cy, cx, cy + d / 2)
            painter.drawLine(cx, cy, cx - d / 2, cy)
            painter.drawLine(cx, cy, cx, cy - d / 2)
            if oldpoint != []:
                painter.drawLine(cx, cy, ox, oy)
                font = painter.font()
                font.setPixelSize(20)
                painter.setFont(font)
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
        if self.window.display_settings_dlg.scalebar_groupbox.isChecked():
            pixelsize = self.window.display_settings_dlg.pixelsize.value()
            scalebar = self.window.display_settings_dlg.scalebar.value()
            length_camerapxl = scalebar / pixelsize
            length_displaypxl = int(
                round(self.width() * length_camerapxl / self.viewport_width())
            )
            height = max(int(round(0.15 * length_displaypxl)), 1)
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))
            x = self.width() - length_displaypxl - 35
            y = self.height() - height - 20
            painter.drawRect(x, y, length_displaypxl + 0, height + 0)
            if self.window.display_settings_dlg.scalebar_text.isChecked():
                font = painter.font()
                font.setPixelSize(20)
                painter.setFont(font)
                painter.setPen(QtGui.QColor("white"))
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

    def draw_minimap(self, image):
        if self.window.display_settings_dlg.minimap.isChecked():
            movie_height, movie_width = self.movie_size()
            length_minimap = 100
            height_minimap = movie_height / movie_width * 100
            # draw in the upper right corner, overview rectangle
            x = self.width() - length_minimap - 20
            y = 20
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QColor("white"))
            painter.drawRect(x, y, length_minimap + 0, height_minimap + 0)
            painter.setPen(QtGui.QColor("yellow"))
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
        points_only=False,
    ):
        if not picks_only:
            self.viewport = self.adjust_viewport_to_view(viewport)
            qimage = self.render_scene(autoscale=autoscale, use_cache=use_cache)
            qimage = qimage.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.KeepAspectRatioByExpanding,
            )
            self.qimage_no_picks = self.draw_scalebar(qimage)
            self.qimage_no_picks = self.draw_minimap(self.qimage_no_picks)
            dppvp = self.display_pixels_per_viewport_pixels()
            self.window.display_settings_dlg.set_zoom_silently(dppvp)
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
        points_only=False,
    ):
        slicerposition = self.window.slicer_dialog.slicerposition
        pixmap = self.window.slicer_dialog.slicer_cache.get(slicerposition)

        if pixmap is None:
            self.draw_scene(
                viewport,
                autoscale=autoscale,
                use_cache=use_cache,
                picks_only=picks_only,
                points_only=points_only,
            )
            self.window.slicer_dialog.slicer_cache[slicerposition] = self.pixmap
        else:
            self.setPixmap(pixmap)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [_.toLocalFile() for _ in urls]
        extensions = [os.path.splitext(_)[1].lower() for _ in paths]
        paths = [path for path, ext in zip(paths, extensions) if ext == ".hdf5"]
        self.add_multiple(paths)

    def fit_in_view(self, autoscale=False):
        movie_height, movie_width = self.movie_size()
        viewport = [(0, 0), (movie_height, movie_width)]
        self.update_scene(viewport=viewport, autoscale=autoscale)

    def get_channel(self, title="Choose a channel"):
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
        n_channels = len(self.locs_paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.locs_paths) > 1:
            pathlist = list(self.locs_paths)
            pathlist.append("Exchangerounds by color")
            index, ok = QtWidgets.QInputDialog.getItem(
                self, "Select channel", "Channel:", pathlist, editable=False
            )
            if ok:
                return pathlist.index(index)
            else:
                return None

    def get_render_kwargs(self, viewport=None):
        """
        Returns a dictionary to be used for the
        keyword arguments of render.
        """
        blur_button = self.window.display_settings_dlg.blur_buttongroup.checkedButton()
        optimal_oversampling = self.display_pixels_per_viewport_pixels()
        if self.window.display_settings_dlg.dynamic_oversampling.isChecked():
            oversampling = optimal_oversampling
            self.window.display_settings_dlg.set_oversampling_silently(
                optimal_oversampling
            )
        else:
            oversampling = float(self.window.display_settings_dlg.oversampling.value())
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
        if viewport is None:
            viewport = self.viewport
        return {
            "oversampling": oversampling,
            "viewport": viewport,
            "blur_method": self.window.display_settings_dlg.blur_methods[blur_button],
            "min_blur_width": float(
                self.window.display_settings_dlg.min_blur_width.value()
            ),
        }

    def load_picks(self, path):
        """Loads picks from yaml file."""
        with open(path, "r") as f:
            regions = yaml.load(f)

        # Backwards compatibility for old picked region files
        if "Shape" in regions:
            loaded_shape = regions["Shape"]
        elif "Centers" in regions and "Diameter" in regions:
            loaded_shape = "Circle"
        else:
            raise ValueError("Unrecognized picks file")

        shape_index = self.window.tools_settings_dialog.pick_shape.findText(
            loaded_shape
        )
        self.window.tools_settings_dialog.pick_shape.setCurrentIndex(shape_index)
        if loaded_shape == "Circle":
            self._picks = regions["Centers"]
            self.window.tools_settings_dialog.pick_diameter.setValue(
                regions["Diameter"]
            )
        elif loaded_shape == "Rectangle":
            self._picks = regions["Center-Axis-Points"]
            self.window.tools_settings_dialog.pick_width.setValue(regions["Width"])
        else:
            raise ValueError("Unrecognized pick shape")
        self.update_pick_info_short()
        self.update_scene(picks_only=True)

    def substract_picks(self, path):
        if self._pick_shape == "Rectangle":
            raise NotImplementedError(
                "Subtracting picks not implemented for rectangle picks"
            )
        oldpicks = self._picks.copy()
        with open(path, "r") as f:
            regions = yaml.load(f)
            self._picks = regions["Centers"]
            diameter = regions["Diameter"]

            x_cord = np.array([_[0] for _ in self._picks])
            y_cord = np.array([_[1] for _ in self._picks])
            x_cord_old = np.array([_[0] for _ in oldpicks])
            y_cord_old = np.array([_[1] for _ in oldpicks])

            distances = (
                np.sum(
                    (euclidean_distances(oldpicks, self._picks) < diameter / 2) * 1,
                    axis=1,
                )
                >= 1
            )
            filtered_list = [i for (i, v) in zip(oldpicks, distances) if not v]

            x_cord_new = np.array([_[0] for _ in filtered_list])
            y_cord_new = np.array([_[1] for _ in filtered_list])
            output = False

            if output:
                fig1 = plt.figure()
                plt.title("Old picks and new picks")
                plt.scatter(x_cord, -y_cord, c="r", label="Newpicks", s=2)
                plt.scatter(x_cord_old, -y_cord_old, c="b", label="Oldpicks", s=2)
                plt.scatter(x_cord_new, -y_cord_new, c="g", label="Picks to keep", s=2)
                fig1.show()
            self._picks = filtered_list

            self.update_pick_info_short()
            self.window.tools_settings_dialog.pick_diameter.setValue(
                regions["Diameter"]
            )
            self.update_scene(picks_only=True)

    def map_to_movie(self, position):
        """Converts coordinates from display units to camera units."""
        x_rel = position.x() / self.width()
        x_movie = x_rel * self.viewport_width() + self.viewport[0][1]
        y_rel = position.y() / self.height()
        y_movie = y_rel * self.viewport_height() + self.viewport[0][0]
        return x_movie, y_movie

    def map_to_view(self, x, y):
        """Converts coordinates from camera units to display units."""
        cx = self.width() * (x - self.viewport[0][1]) / self.viewport_width()
        cy = self.height() * (y - self.viewport[0][0]) / self.viewport_height()
        return cx, cy

    def max_movie_height(self):
        """Returns maximum height of all loaded images."""
        return max(info[0]["Height"] for info in self.infos)

    def max_movie_width(self):
        return max([info[0]["Width"] for info in self.infos])

    def mouseMoveEvent(self, event):
        if self._mode == "Zoom":
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(QtCore.QRect(self.origin, event.pos()))
            if self._pan:
                rel_x_move = (event.x() - self.pan_start_x) / self.width()
                rel_y_move = (event.y() - self.pan_start_y) / self.height()
                self.pan_relative(rel_y_move, rel_x_move)
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
        elif self._mode == "Pick":
            if self._pick_shape == "Rectangle":
                if self._rectangle_pick_ongoing:
                    self.rectangle_pick_current_x = event.x()
                    self.rectangle_pick_current_y = event.y()
                    self.update_scene(picks_only=True)

    def mousePressEvent(self, event):
        if self._mode == "Zoom":
            if event.button() == QtCore.Qt.LeftButton:
                if not self.rubberband.isVisible():
                    self.origin = QtCore.QPoint(event.pos())
                    self.rubberband.setGeometry(
                        QtCore.QRect(self.origin, QtCore.QSize())
                    )
                    self.rubberband.show()
            elif event.button() == QtCore.Qt.RightButton:
                self.render_check_was_checked = False
                if self.window.display_settings_dlg.render_check.isChecked():
                    self.window.display_settings_dlg.render_check.setChecked(False)
                    self.render_check_was_checked = True
                self._pan = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
            else:
                event.ignore()
        elif self._mode == "Pick":
            if event.button() == QtCore.Qt.LeftButton:
                if self._pick_shape == "Rectangle":
                    self._rectangle_pick_ongoing = True
                    self.rectangle_pick_start_x = event.x()
                    self.rectangle_pick_start_y = event.y()
                    self.rectangle_pick_start = self.map_to_movie(event.pos())

    def mouseReleaseEvent(self, event):
        if self._mode == "Zoom":
            if event.button() == QtCore.Qt.LeftButton and self.rubberband.isVisible():
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
            elif event.button() == QtCore.Qt.RightButton:
                self._pan = False
                self.setCursor(QtCore.Qt.ArrowCursor)
                event.accept()
                if self.render_check_was_checked:
                    self.window.display_settings_dlg.render_check.setChecked(True)
            else:
                event.ignore()
        elif self._mode == "Pick":
            if self._pick_shape == "Circle":
                if event.button() == QtCore.Qt.LeftButton:
                    x, y = self.map_to_movie(event.pos())
                    self.add_pick((x, y))
                    event.accept()
                elif event.button() == QtCore.Qt.RightButton:
                    x, y = self.map_to_movie(event.pos())
                    self.remove_picks((x, y))
                    event.accept()
                else:
                    event.ignore()
            elif self._pick_shape == "Rectangle":
                if event.button() == QtCore.Qt.LeftButton:
                    rectangle_pick_end = self.map_to_movie(event.pos())
                    self._rectangle_pick_ongoing = False
                    self.add_pick((self.rectangle_pick_start, rectangle_pick_end))
                    event.accept()
                elif event.button() == QtCore.Qt.RightButton:
                    x, y = self.map_to_movie(event.pos())
                    self.remove_picks((x, y))
                    event.accept()
                else:
                    event.ignore()
            else:
                raise ValueError(
                    "`self._pick_shape` must be of ('Circle', 'Rectangle')."
                )
        elif self._mode == "Measure":
            if event.button() == QtCore.Qt.LeftButton:
                x, y = self.map_to_movie(event.pos())
                self.add_point((x, y))
                event.accept()
            elif event.button() == QtCore.Qt.RightButton:
                x, y = self.map_to_movie(event.pos())
                self.remove_points((x, y))
                event.accept()
            else:
                event.ignore()

    def movie_size(self):
        movie_height = self.max_movie_height()
        movie_width = self.max_movie_width()
        return (movie_height, movie_width)

    def display_pixels_per_viewport_pixels(self):
        os_horizontal = self.width() / self.viewport_width()
        os_vertical = self.height() / self.viewport_height()
        # The values should be identical, but just in case, we choose the max:
        return max(os_horizontal, os_vertical)

    def pan_relative(self, dy, dx):
        viewport_height, viewport_width = self.viewport_size()
        x_move = dx * viewport_width
        y_move = dy * viewport_height
        x_min = self.viewport[0][1] - x_move
        x_max = self.viewport[1][1] - x_move
        y_min = self.viewport[0][0] - y_move
        y_max = self.viewport[1][0] - y_move
        viewport = [(y_min, x_min), (y_max, x_max)]
        self.update_scene(viewport)

    def plot3d(self):
        channel = self.get_channel3d("Plot 3D")
        if channel is not None:
            fig = plt.figure()
            fig.canvas.set_window_title("3D - Trace")
            ax = fig.add_subplot(111, projection="3d")
            ax.set_title("3d view of pick")

            if channel is (len(self.locs_paths)):
                n_channels = len(self.locs_paths)
                colors = get_colors(n_channels)

                for i in range(len(self.locs_paths)):
                    locs = self.picked_locs(i)
                    locs = stack_arrays(locs, asrecarray=True, usemask=False)
                    ax.scatter(locs["x"], locs["y"], locs["z"], c=colors[i], s=2)

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

            else:
                locs = self.picked_locs(channel)
                locs = stack_arrays(locs, asrecarray=True, usemask=False)

                colors = locs["z"][:]
                colors[colors > np.mean(locs["z"]) + 3 * np.std(locs["z"])] = np.mean(
                    locs["z"]
                ) + 3 * np.std(locs["z"])
                colors[colors < np.mean(locs["z"]) - 3 * np.std(locs["z"])] = np.mean(
                    locs["z"]
                ) - 3 * np.std(locs["z"])
                ax.scatter(locs["x"], locs["y"], locs["z"], c=colors, cmap="jet", s=2)

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
            ax.set_xlabel("X [Px]")
            ax.set_ylabel("Y [Px]")
            ax.set_zlabel("Z [nm]")
            ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
            ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
            fig.canvas.draw()
            fig.show()

    @check_pick
    def show_trace(self):
        self.current_trace_x = 0
        self.current_trace_y = 0

        channel = self.get_channel("Undrift from picked")
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
            # Three subplots sharing both x/y axes
            ax1, ax2, ax3 = self.canvas.figure.subplots(3, sharex=True)

            ax1.scatter(locs["frame"], locs["x"], s=2)
            ax1.set_title("X-pos vs frame")
            ax2.scatter(locs["frame"], locs["y"], s=2)
            ax2.set_title("Y-pos vs frame")
            ax3.plot(xvec, yvec, linewidth=1)
            ax3.fill_between(xvec, 0, yvec, facecolor="red")
            ax3.set_title("Localizations")

            ax1.set_xlim(0, (max(locs["frame"]) + 1))
            ax3.set_xlabel("Frames")

            ax1.set_ylabel("X-pos [Px]")
            ax2.set_ylabel("Y-pos [Px]")
            ax3.set_ylabel("ON")
            ax3.set_ylim([-0.1, 1.1])

            self.exportTraceButton = QtWidgets.QPushButton("Export (*.csv)")
            self.canvas.toolbar.addWidget(self.exportTraceButton)
            self.exportTraceButton.clicked.connect(self.exportTrace)

            self.canvas.canvas.draw()
            self.canvas.show()

    def exportTrace(self):
        trace = np.array([self.current_trace_x, self.current_trace_y])
        base, ext = os.path.splitext(self.locs_paths[self.channel])
        out_path = base + ".trace.txt"

        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save trace as txt", out_path, filter="*.trace.txt"
        )

        if path:
            np.savetxt(path, trace, fmt="%i", delimiter=",")

    def pick_message_box(self, params):
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

        msgBox.addButton(QtWidgets.QPushButton("Accept"), QtWidgets.QMessageBox.YesRole)
        msgBox.addButton(QtWidgets.QPushButton("Reject"), QtWidgets.QMessageBox.NoRole)
        msgBox.addButton(QtWidgets.QPushButton("Back"), QtWidgets.QMessageBox.ResetRole)
        msgBox.addButton(
            QtWidgets.QPushButton("Cancel"), QtWidgets.QMessageBox.RejectRole
        )

        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        msgBox.move(qr.topLeft())

        return msgBox

    def show_fret(self):
        channel_acceptor = self.get_channel(title="Select acceptor channel")
        channel_donor = self.get_channel(title="Select donor channel")

        removelist = []

        n_channels = len(self.locs_paths)
        acc_picks = self.picked_locs(channel_acceptor)
        don_picks = self.picked_locs(channel_donor)

        if self._picks:
            if self._pick_shape == "Rectangle":
                raise NotImplementedError("Not implemented for rectangle picks")
            params = {}
            params["t0"] = time.time()
            i = 0
            while i < len(self._picks):

                pick = self._picks[i]

                fret_dict, fret_locs = postprocess.calculate_fret(
                    acc_picks[i], don_picks[i]
                )

                fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
                fig.canvas.set_window_title("FRET-trace")
                ax1.plot(fret_dict["frames"], fret_dict["acc_trace"])
                ax1.set_title("Acceptor intensity vs frame")
                ax2.plot(fret_dict["frames"], fret_dict["don_trace"])
                ax2.set_title("Donor intensity vs frame")
                ax3.scatter(fret_dict["fret_timepoints"], fret_dict["fret_events"], s=2)
                ax3.set_title(r"$\frac{I_A}{I_D+I_A}$")

                ax1.set_xlim(0, (fret_dict["maxframes"] + 1))
                ax3.set_xlabel("Frame")

                ax1.set_ylabel("Photons")
                ax2.set_ylabel("Photons")
                ax3.set_ylabel("Ratio")

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
                    # Accepted
                    if pick in removelist:
                        removelist.remove(pick)
                elif reply == 3:
                    # Cancel
                    break
                elif reply == 2:
                    # Back
                    if i >= 2:
                        i -= 2
                    else:
                        i = -1
                else:
                    # Discard
                    removelist.append(pick)

                i += 1
                plt.close()

        for pick in removelist:
            self._picks.remove(pick)

        self.n_picks = len(self._picks)

        self.update_pick_info_short()
        self.update_scene()

    def calculate_fret_dialog(self):
        if self._pick_shape == "Rectangle":
            raise NotImplementedError("Not implemented for rectangle picks")
        print("Calculating FRET")
        fret_events = []

        channel_acceptor = self.get_channel(title="Select acceptor channel")
        channel_donor = self.get_channel(title="Select donor channel")

        acc_picks = self.picked_locs(channel_acceptor)
        don_picks = self.picked_locs(channel_donor)

        K = len(self._picks)
        progress = lib.ProgressDialog("Calculating fret in Picks...", 0, K, self)
        progress.show()

        all_fret_locs = []

        for i in range(K):
            fret_dict, fret_locs = postprocess.calculate_fret(
                acc_picks[i], don_picks[i]
            )
            if fret_dict["fret_events"] != []:
                fret_events.append(fret_dict["fret_events"])
            if fret_locs != []:
                all_fret_locs.append(fret_locs)
            progress.set_value(i + 1)

        progress.close()

        if fret_events == []:
            raise ValueError(
                "No FRET events detected. "
                "Inspect picks with Show FRET Traces "
                "and make sure to have FRET events."
            )

        fig1 = plt.figure()
        plt.hist(np.hstack(fret_events), bins=np.arange(0, 1, 0.02))
        plt.title(r"Distribution of $\frac{I_A}{I_D+I_A}$")
        plt.xlabel("Ratio")
        plt.ylabel("Counts")
        fig1.show()

        base, ext = os.path.splitext(self.locs_paths[channel_acceptor])
        out_path = base + ".fret.txt"

        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save FRET values as txt and picked locs",
            out_path,
            filter="*.fret.txt",
        )

        if path:
            np.savetxt(
                path,
                np.hstack(fret_events),
                fmt="%1.5f",
                newline="\r\n",
                delimiter="   ",
            )

            locs = stack_arrays(all_fret_locs, asrecarray=True, usemask=False)
            if locs is not None:
                base, ext = os.path.splitext(path)
                out_path = base + ".hdf5"
                pick_info = {"Generated by:": "Picasso Render FRET"}
                io.save_locs(out_path, locs, self.infos[channel_acceptor] + [pick_info])

    def select_traces(self):
        print("Showing  traces")

        removelist = []

        channel = self.get_channel("Undrift from picked")
        if channel is not None:
            if self._picks:
                params = {}
                params["t0"] = time.time()
                all_picked_locs = self.picked_locs(channel)
                i = 0
                while i < len(self._picks):
                    fig = plt.figure(figsize=(5, 5))
                    fig.canvas.set_window_title("Trace")
                    pick = self._picks[i]
                    locs = all_picked_locs[i]
                    locs = stack_arrays(locs, asrecarray=True, usemask=False)

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
    def show_pick(self):
        print("Showing picks...")
        channel = self.get_channel3d("Select Channel")

        removelist = []

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
                            locs = stack_arrays(locs, asrecarray=True, usemask=False)
                            ax.scatter(locs["x"], locs["y"], c=colors[l], s=2)

                        ax.set_xlabel("X [Px]")
                        ax.set_ylabel("Y [Px]")
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
                        ax.set_title(
                            "Scatterplot of Pick "
                            + str(i + 1)
                            + "  of: "
                            + str(len(self._picks))
                            + "."
                        )
                        locs = all_picked_locs[i]
                        locs = stack_arrays(locs, asrecarray=True, usemask=False)
                        ax.scatter(locs["x"], locs["y"], c=colors[channel], s=2)
                        ax.set_xlabel("X [Px]")
                        ax.set_ylabel("Y [Px]")
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
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
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
                            pixelsize,
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
                            pixelsize,
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
                saved_locs = stack_arrays(saved_locs, asrecarray=True, usemask=False)
                if saved_locs is not None:
                    d = self.window.tools_settings_dialog.pick_diameter.value()
                    pick_info = {
                        "Generated by:": "Picasso Render",
                        "Pick Diameter:": d,
                    }
                    io.save_locs(path, saved_locs, self.infos[channel] + [pick_info])

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
            std_range = self.window.tools_settings_dialog.pick_similar_range.value()
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
                    block_locs = postprocess.get_block_locs_at(x, y, index_blocks)
                    pick_locs = lib.locs_at(x, y, block_locs, r)
                    locs = stack_arrays(pick_locs, asrecarray=True, usemask=False)
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
        K, L = postprocess.index_blocks_shape(info, size)
        progress = lib.ProgressDialog("Indexing localizations", 0, K, self)
        progress.show()
        progress.set_value(0)
        index_blocks = postprocess.get_index_blocks(
            locs, info, size, progress.set_value
        )
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
            std_range = self.window.tools_settings_dialog.pick_similar_range.value()
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
            d2 = d**2
            nx = len(x_range)
            locs, size, x_index, y_index, block_starts, block_ends, K, L = index_blocks
            progress = lib.ProgressDialog("Pick similar", 0, nx, self)
            progress.set_value(0)
            for i, x_grid in enumerate(x_range):
                # y_grid is shifted for odd columns
                if i % 2:
                    y_range = y_range_shift
                else:
                    y_range = y_range_base
                for y_grid in y_range:
                    n_block_locs = postprocess.n_block_locs_at(
                        x_grid, y_grid, size, K, L, block_starts, block_ends
                    )
                    if n_block_locs > min_n_locs:
                        block_locs = postprocess.get_block_locs_at(
                            x_grid, y_grid, index_blocks
                        )
                        picked_locs = lib.locs_at(x_grid, y_grid, block_locs, r)
                        if len(picked_locs) > 1:
                            # Move to COM peak
                            x_test_old = x_grid
                            y_test_old = y_grid
                            x_test = picked_locs.x.mean()
                            y_test = picked_locs.y.mean()
                            while (
                                np.abs(x_test - x_test_old) > 1e-3
                                or np.abs(y_test - y_test_old) > 1e-3
                            ):
                                x_test_old = x_test
                                y_test_old = y_test
                                picked_locs = lib.locs_at(x_test, y_test, block_locs, r)
                                x_test = picked_locs.x.mean()
                                y_test = picked_locs.y.mean()
                            if np.all(
                                (x_similar - x_test) ** 2 + (y_similar - y_test) ** 2
                                > d2
                            ):
                                if min_n_locs < len(picked_locs) < max_n_locs:
                                    if (
                                        min_rmsd
                                        < self.rmsd_at_com(picked_locs)
                                        < max_rmsd
                                    ):
                                        x_similar = np.append(x_similar, x_test)
                                        y_similar = np.append(y_similar, y_test)
                progress.set_value(i + 1)
            similar = list(zip(x_similar, y_similar))
            self._picks = []
            self.add_picks(similar)

    def picked_locs(self, channel, add_group=True):
        """Returns picked localizations in the specified channel"""

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
                    block_locs = postprocess.get_block_locs_at(x, y, index_blocks)
                    group_locs = lib.locs_at(x, y, block_locs, r)
                    if add_group:
                        group = i * np.ones(len(group_locs), dtype=np.int32)
                        group_locs = lib.append_to_rec(group_locs, group, "group")
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
                    x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
                    y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)
                    group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
                    group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")
                    if add_group:
                        group = i * np.ones(len(group_locs), dtype=np.int32)
                        group_locs = lib.append_to_rec(group_locs, group, "group")
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
                    if not lib.check_if_in_rectangle(x, y, np.array(X), np.array(Y))[0]:
                        new_picks.append(pick)
        self._picks = []
        self.add_picks(new_picks)

    def remove_points(self, position):
        self._points = []
        self.update_scene()

    def render_scene(self, autoscale=False, use_cache=False, cache=True, viewport=None):
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
                plot_channels=True,
            )
        self._bgra[:, :, 3].fill(255)
        Y, X = self._bgra.shape[:2]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        return qimage

    # TODO : check if we still need this function
    def render_scene_hist(
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
                kwargs, autoscale=autoscale, use_cache=use_cache, cache=cache
            )
        self._bgra[:, :, 3].fill(255)
        return self._bgra.data

    def read_colors(self, colors=None):
        if colors is None:
            colors = get_colors(len(self.locs))
        for i in range(len(self.locs)):
            if self.window.dataset_dialog.colorselection[i].currentText() == "red":
                colors[i] = (1.0, 0, 0)
            elif self.window.dataset_dialog.colorselection[i].currentText() == "green":
                colors[i] = (0, 1.0, 0)
            elif self.window.dataset_dialog.colorselection[i].currentText() == "blue":
                colors[i] = (0, 0, 1.0)
            elif self.window.dataset_dialog.colorselection[i].currentText() == "gray":
                colors[i] = (1.0, 1.0, 1.0)
            elif self.window.dataset_dialog.colorselection[i].currentText() == "cyan":
                colors[i] = (0, 1.0, 1.0)
            elif (
                self.window.dataset_dialog.colorselection[i].currentText() == "magenta"
            ):
                colors[i] = (1.0, 0, 1.0)
            elif self.window.dataset_dialog.colorselection[i].currentText() == "yellow":
                colors[i] = (1.0, 1.0, 0)
            elif self.window.dataset_dialog.colorselection[i].currentText() != "auto":
                colorstring = (
                    self.window.dataset_dialog.colorselection[i]
                    .currentText()
                    .lstrip("#")
                )
                rgbval = tuple(
                    float(int(colorstring[i : i + 2], 16) / 255 for i in (0, 2, 4))
                )
                colors[i] = rgbval
        return colors

    def render_multi_channel(
        self,
        kwargs,
        autoscale=False,
        locs=None,
        use_cache=False,
        cache=True,
        plot_channels=False,
    ):
        if locs is None:
            locs = self.locs
        # Plot each channel
        if plot_channels:
            locsall = locs.copy()
            for i in range(len(locs)):
                if hasattr(locs[i], "z"):
                    if self.window.slicer_dialog.slicerRadioButton.isChecked():
                        z_min = self.window.slicer_dialog.slicermin
                        z_max = self.window.slicer_dialog.slicermax
                        in_view = (locsall[i].z > z_min) & (locsall[i].z <= z_max)
                        locsall[i] = locsall[i][in_view]
            n_channels = len(locs)
            colors = get_colors(n_channels)
            if use_cache:
                n_locs = self.n_locs
                image = self.image
            else:
                renderings = []
                for i in range(len(self.locs)):
                    # We render all images first
                    # and later decide to keep them or not
                    renderings.append(render.render(locsall[i], **kwargs))
                renderings = [render.render(_, **kwargs) for _ in locsall]
                n_locs = sum([_[0] for _ in renderings])
                image = np.array([_[1] for _ in renderings])
        else:
            n_channels = len(locs)
            colors = get_colors(n_channels)
            if use_cache:
                n_locs = self.n_locs
                image = self.image
            else:

                pb = lib.ProgressDialog("Rendering.. ", 0, n_channels, self)
                pb.set_value(0)
                renderings = []
                for i in tqdm(range(n_channels)):
                    renderings.append(render.render(locs[i], **kwargs))
                    pb.set_value(i + 1)
                pb.close()

                # renderings = [render.render(_, **kwargs) for _ in locs]
                n_locs = sum([_[0] for _ in renderings])
                image = np.array([_[1] for _ in renderings])

        if cache:
            self.n_locs = n_locs
            self.image = image

        image = self.scale_contrast(image)
        Y, X = image.shape[1:]
        bgra = np.zeros((Y, X, 4), dtype=np.float32)

        # Color images
        colors = self.read_colors(colors)

        for i in range(len(self.locs)):
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
        self, kwargs, autoscale=False, use_cache=False, cache=True
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
            if self.window.slicer_dialog.slicerRadioButton.isChecked():
                z_min = self.window.slicer_dialog.slicermin
                z_max = self.window.slicer_dialog.slicermax
                in_view = (locs.z > z_min) & (locs.z <= z_max)
                locs = locs[in_view]

        if use_cache:
            n_locs = self.n_locs
            image = self.image
        else:
            n_locs, image = render.render(locs, **kwargs)
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
        return self._bgra

    def resizeEvent(self, event):
        self.update_scene()

    def save_picked_locs(self, path, channel):
        locs = self.picked_locs(channel)
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
                templocs = stack_arrays(templocs, asrecarray=True, usemask=False)
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
        progress = lib.ProgressDialog("Calculating kinetics", 0, len(picked_locs), self)
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
        progress = lib.ProgressDialog("Calculating pick properties", 0, n_groups, self)
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
                "Centers": [[float(_[0]), float(_[1])] for _ in self._picks],
            }
        elif self._pick_shape == "Rectangle":
            w = self.window.tools_settings_dialog.pick_width.value()
            picks = {
                "Width": float(w),
                "Center-Axis-Points": [
                    [[float(s[0]), float(s[1])], [float(e[0]), float(e[1])]]
                    for s, e in self._picks
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
            upper = lower + 1 / (10**6)
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
                    pb = lib.ProgressDialog("Indexing colors", 0, N_Z_COLORS, self)
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
            parameter = self.window.display_settings_dlg.parameter.currentText()
            colors = self.window.display_settings_dlg.color_step.value()
            min_val = self.window.display_settings_dlg.minimum_render.value()
            max_val = self.window.display_settings_dlg.maximum_render.value()

            x_step = (max_val - min_val) / colors

            self.x_color = np.floor((self.locs[0][parameter] - min_val) / x_step)

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
                pb = lib.ProgressDialog("Indexing " + parameter, 0, colors, self)
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
        current_text = t_dialog.pick_shape.currentText()
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
                shape_index = t_dialog.pick_shape.findText(self._pick_shape)
                self.window.tools_settings_dialog.pick_shape.setCurrentIndex(
                    shape_index
                )
                return
        self._pick_shape = current_text
        self._picks = []
        self.update_scene(picks_only=True)
        self.update_cursor()
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
        # Todo: Implement a check if there is drift already loaded and load
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
                print("Showing Drift")
                self.plot_window = DriftPlotWindow(self)
                if hasattr(self._drift[channel], "z"):
                    self.plot_window.plot_3d(drift)

                else:
                    self.plot_window.plot_2d(drift)

                self.plot_window.show()

    def undrift(self):
        """Undrifts with rcc."""
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
                    drift, _ = postprocess.undrift(
                        locs,
                        info,
                        segmentation,
                        False,
                        seg_progress.set_value,
                        rcc_progress.set_value,
                    )
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

    def _undrift_from_picked_coordinate(self, channel, picked_locs, coordinate):
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
        drift_mean[nans] = np.interp(nonzero(nans), nonzero(~nans), drift_mean[~nans])

        return drift_mean

    def _undrift_from_picked(self, channel):
        picked_locs = self.picked_locs(channel)
        status = lib.StatusDialog("Calculating drift...", self)

        drift_x = self._undrift_from_picked_coordinate(channel, picked_locs, "x")
        drift_y = self._undrift_from_picked_coordinate(channel, picked_locs, "y")

        # Apply drift
        self.locs[channel].x -= drift_x[self.locs[channel].frame]
        self.locs[channel].y -= drift_y[self.locs[channel].frame]

        # A rec array to store the applied drift
        drift = (drift_x, drift_y)
        drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])

        # If z coordinate exists, also apply drift there
        if all([hasattr(_, "z") for _ in picked_locs]):
            drift_z = self._undrift_from_picked_coordinate(channel, picked_locs, "z")
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

        drift_x = self._undrift_from_picked_coordinate(channel, picked_locs, "x")
        drift_y = self._undrift_from_picked_coordinate(channel, picked_locs, "y")

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

        if hasattr(drift, "z"):
            np.savetxt(
                driftfile,
                self._drift[channel],
                header="dx\tdy\tdz",
                newline="\r\n",
            )
        else:
            np.savetxt(
                driftfile,
                self._drift[channel],
                header="dx\tdy",
                newline="\r\n",
            )

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
                    raise NotImplementedError("Not implemented for rectangle picks")
                # Also unfold picks
                groups = np.unique(self.locs[0].group)

                shift_x = np.mod(groups, n_square) * 2 - mean_x + offset_x
                shift_y = np.floor(groups / n_square) * 2 - mean_y + offset_y

                for j in range(len(self._picks)):
                    for k in range(len(groups)):
                        x_pick, y_pick = self._picks[j]
                        self._picks.append((x_pick + shift_x[k], y_pick + shift_y[k]))

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
        if self._mode == "Zoom":
            self.unsetCursor()
        elif self._mode == "Pick":
            if self._pick_shape == "Circle":
                diameter = self.window.tools_settings_dialog.pick_diameter.value()
                diameter = self.width() * diameter / self.viewport_width()
                if diameter < 100:  # remote desktop crashes if pick is larger than view
                    pixmap_size = ceil(diameter)
                    pixmap = QtGui.QPixmap(pixmap_size, pixmap_size)
                    pixmap.fill(QtCore.Qt.transparent)
                    painter = QtGui.QPainter(pixmap)
                    painter.setPen(QtGui.QColor("white"))
                    offset = (pixmap_size - diameter) / 2
                    painter.drawEllipse(offset, offset, diameter, diameter)
                    painter.end()
                    cursor = QtGui.QCursor(pixmap)
                    self.setCursor(cursor)
            elif self._pick_shape == "Rectangle":
                self.unsetCursor()

    def update_pick_info_long(self, info):
        """Gets called when "Show info below" """
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
                N[i] = len(locs)
                com_x = np.mean(locs.x)
                com_y = np.mean(locs.y)
                rmsd[i] = np.sqrt(
                    np.mean((locs.x - com_x) ** 2 + (locs.y - com_y) ** 2)
                )
                if has_z:
                    rmsd_z[i] = np.sqrt(np.mean((locs.z - np.mean(locs.z)) ** 2))
                if not hasattr(locs, "len"):
                    locs = postprocess.link(locs, info, r_max=r_max, max_dark_time=t)
                locs = postprocess.compute_dark_times(locs)
                length[i] = estimate_kinetic_rate(locs.len)
                dark[i] = estimate_kinetic_rate(locs.dark)
                if N[i] > 0:
                    new_locs.append(locs)
                progress.set_value(i + 1)

            self.window.info_dialog.n_localizations_mean.setText(
                "{:.2f}".format(np.nanmean(N))
            )
            self.window.info_dialog.n_localizations_std.setText(
                "{:.2f}".format(np.nanstd(N))
            )
            self.window.info_dialog.rmsd_mean.setText("{:.2}".format(np.nanmean(rmsd)))
            self.window.info_dialog.rmsd_std.setText("{:.2}".format(np.nanstd(rmsd)))
            if has_z:
                self.window.info_dialog.rmsd_z_mean.setText(
                    "{:.2f}".format(np.nanmean(rmsd_z))
                )
                self.window.info_dialog.rmsd_z_std.setText(
                    "{:.2f}".format(np.nanstd(rmsd_z))
                )
            pooled_locs = stack_arrays(new_locs, usemask=False, asrecarray=True)
            fit_result_len = fit_cum_exp(pooled_locs.len)
            fit_result_dark = fit_cum_exp(pooled_locs.dark)
            self.window.info_dialog.length_mean.setText(
                "{:.2f}".format(np.nanmean(length))
            )
            self.window.info_dialog.length_std.setText(
                "{:.2f}".format(np.nanstd(length))
            )
            self.window.info_dialog.dark_mean.setText("{:.2f}".format(np.nanmean(dark)))
            self.window.info_dialog.dark_std.setText("{:.2f}".format(np.nanstd(dark)))
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
        points_only=False,
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
                points_only=points_only,
            )
            self.update_cursor()

    def update_scene_slicer(
        self,
        viewport=None,
        autoscale=False,
        use_cache=False,
        picks_only=False,
        points_only=False,
    ):
        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene_slicer(
                viewport,
                autoscale=autoscale,
                use_cache=use_cache,
                picks_only=picks_only,
                points_only=points_only,
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

    def zoom(self, factor, custom_center=None):
        viewport_height, viewport_width = self.viewport_size()
        new_viewport_height_half = 0.5 * viewport_height * factor
        new_viewport_width_half = 0.5 * viewport_width * factor

        if custom_center:
            viewport_center_x, viewport_center_y = custom_center
        else:
            viewport_center_y, viewport_center_x = self.viewport_center()
        new_viewport = [
            (
                viewport_center_y - new_viewport_height_half,
                viewport_center_x - new_viewport_width_half,
            ),
            (
                viewport_center_y + new_viewport_height_half,
                viewport_center_x + new_viewport_width_half,
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
            if direction < 0:
                self.zoom(1 / ZOOM, custom_center=position)
            else:
                self.zoom(ZOOM, custom_center=position)


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.view._pick_shape = self.tools_settings_dialog.pick_shape.currentText()
        self.tools_settings_dialog.pick_shape.currentIndexChanged.connect(
            self.view.on_pick_shape_changed
        )
        self.mask_settings_dialog = MaskSettingsDialog(self)
        self.slicer_dialog = SlicerDialog(self)
        self.info_dialog = InfoDialog(self)
        self.merge_dialog = MergeDialog(self)
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        save_action = file_menu.addAction("Save localizations")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs)
        save_picked_action = file_menu.addAction("Save picked localizations")
        save_picked_action.setShortcut("Ctrl+Shift+S")
        save_picked_action.triggered.connect(self.save_picked_locs)
        save_pick_properties_action = file_menu.addAction("Save pick properties")
        save_pick_properties_action.triggered.connect(self.save_pick_properties)
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
            export_ims_action.triggered.connect(self.export_roi_ims)

        view_menu = self.menu_bar.addMenu("View")
        display_settings_action = view_menu.addAction("Display settings")
        display_settings_action.setShortcut("Ctrl+D")
        display_settings_action.triggered.connect(self.display_settings_dlg.show)
        view_menu.addAction(display_settings_action)
        self.dataset_dialog = DatasetDialog(self)
        dataset_action = view_menu.addAction("Files")
        dataset_action.setShortcut("Ctrl+F")
        dataset_action.triggered.connect(self.dataset_dialog.show)
        view_menu.addSeparator()
        to_left_action = view_menu.addAction("Left")
        to_left_action.setShortcut("Left")
        to_left_action.triggered.connect(self.view.to_left)
        to_right_action = view_menu.addAction("Right")
        to_right_action.setShortcut("Right")
        to_right_action.triggered.connect(self.view.to_right)
        to_up_action = view_menu.addAction("Up")
        to_up_action.setShortcut("Up")
        to_up_action.triggered.connect(self.view.to_up)
        to_down_action = view_menu.addAction("Down")
        to_down_action.setShortcut("Down")
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
        tools_settings_action.triggered.connect(self.tools_settings_dialog.show)
        pick_similar_action = tools_menu.addAction("Pick similar")
        pick_similar_action.setShortcut("Ctrl+Shift+P")
        pick_similar_action.triggered.connect(self.view.pick_similar)
        tools_menu.addSeparator()
        show_trace_action = tools_menu.addAction("Show trace")
        show_trace_action.setShortcut("Ctrl+R")
        show_trace_action.triggered.connect(self.view.show_trace)
        plotpick3dsingle_action = tools_menu.addAction("Plot pick (XYZ scatter)")
        plotpick3dsingle_action.triggered.connect(self.view.plot3d)
        plotpick3dsingle_action.setShortcut("Ctrl+3")
        tools_menu.addSeparator()
        select_traces_action = tools_menu.addAction("Select picks (trace)")
        select_traces_action.triggered.connect(self.view.select_traces)

        postprocess_menu = self.menu_bar.addMenu("Postprocess")
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

        pickadd_action = tools_menu.addAction("Substract pick regions")
        pickadd_action.triggered.connect(self.substract_picks)

        tools_menu.addSeparator()
        self.fret_traces_action = tools_menu.addAction("Show FRET traces")
        self.fret_traces_action.triggered.connect(self.view.show_fret)

        self.calculate_fret_action = tools_menu.addAction("Calculate FRET in picks")
        self.calculate_fret_action.triggered.connect(self.view.calculate_fret_dialog)
        tools_menu.addSeparator()
        cluster_action = tools_menu.addAction("Cluster in pick (k-means)")
        cluster_action.triggered.connect(self.view.analyze_cluster)

        mask_action = tools_menu.addAction("Mask image")
        mask_action.triggered.connect(self.mask_settings_dialog.init_dialog)

        merge_action = tools_menu.addAction("Merge localizations")
        merge_action.triggered.connect(self.open_merge_dialog)

        # Drift oeprations
        undrift_action = postprocess_menu.addAction("Undrift by RCC")
        undrift_action.setShortcut("Ctrl+U")
        undrift_action.triggered.connect(self.view.undrift)
        undrift_from_picked_action = postprocess_menu.addAction("Undrift from picked")
        undrift_from_picked_action.setShortcut("Ctrl+Shift+U")
        undrift_from_picked_action.triggered.connect(self.view.undrift_from_picked)
        undrift_from_picked2d_action = postprocess_menu.addAction(
            "Undrift from picked (2D)"
        )
        undrift_from_picked2d_action.triggered.connect(self.view.undrift_from_picked2d)
        drift_action = postprocess_menu.addAction("Undo drift")
        drift_action.triggered.connect(self.view.undo_drift)

        drift_action = postprocess_menu.addAction("Show drift")
        drift_action.triggered.connect(self.view.show_drift)

        # Group related
        postprocess_menu.addSeparator()
        group_action = postprocess_menu.addAction("Remove group info")
        group_action.triggered.connect(self.remove_group)
        unfold_action = postprocess_menu.addAction("Unfold / Refold groups")
        unfold_action.triggered.connect(self.view.unfold_groups)
        unfold_action_square = postprocess_menu.addAction("Unfold groups (square)")
        unfold_action_square.triggered.connect(self.view.unfold_groups_square)

        postprocess_menu.addSeparator()
        link_action = postprocess_menu.addAction("Link localizations")
        link_action.triggered.connect(self.view.link)
        align_action = postprocess_menu.addAction("Align channels (RCC or from picked)")
        align_action.triggered.connect(self.view.align)
        combine_action = postprocess_menu.addAction("Combine locs in picks")
        combine_action.triggered.connect(self.view.combine)

        postprocess_menu.addSeparator()
        apply_action = postprocess_menu.addAction("Apply expression to localizations")
        apply_action.setShortcut("Ctrl+A")
        apply_action.triggered.connect(self.open_apply_dialog)

        postprocess_menu.addSeparator()
        clustering_menu = postprocess_menu.addMenu("Clustering")
        dbscan_action = clustering_menu.addAction("DBSCAN")
        dbscan_action.triggered.connect(self.view.dbscan)
        hdbscan_action = clustering_menu.addAction("HDBSCAN")
        hdbscan_action.triggered.connect(self.view.hdbscan)

        self.load_user_settings()

        # Define 3D entries

        self.actions_3d = [
            plotpick3dsingle_action,
            plotpick3d_action,
            plotpick3d_iso_action,
            slicer_action,
            undrift_from_picked2d_action,
        ]

        for action in self.actions_3d:
            action.setVisible(False)

        # De-select all menus until file is loaded
        self.menus = [view_menu, postprocess_menu, tools_menu]
        for menu in self.menus:
            menu.setDisabled(True)

    def closeEvent(self, event):
        settings = io.load_user_settings()
        settings["Render"][
            "Colormap"
        ] = self.display_settings_dlg.colormap.currentText()
        if self.view.locs_paths != []:
            settings["Render"]["PWD"] = os.path.dirname(self.view.locs_paths[0])
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
        channel = self.view.get_channel("Save localizations as txt (frames,x,y)")
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
                    loctxt = locs[["x", "y", "sx", "bg", "photons", "frame"]].copy()
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
            else:
                print("This should never happen")

    def export_ts(self):
        channel = self.view.get_channel("Save localizations as csv for ThunderSTORM")
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

    def open_merge_dialog(self):
        self.merge_dialog.init_dialog(
            self.view.locs, self.view.locs_paths, self.view.infos
        )

    def estimate_z(self, path_list):
        """
        Helper function to estimate the z_dimensions for multipe locs
        """

        s_x = []
        s_y = []
        nc = len(path_list)
        z_min = []
        z_max = []

        for path in path_list:
            info = io.load_info(path)
            s_x.append(int(info[0]["Width"]))
            s_y.append(int(info[0]["Height"]))

        # Z-needs to be read indiviudally

        for path in path_list:
            locs, info = io.load_locs(path)
            if not hasattr(locs, "z"):
                break
            else:
                z_min.append(locs.z.min())
                z_max.append(locs.z.max())

        if len(z_min) == 0:
            z_min, z_max = 0, 0
        else:
            z_min = np.min(z_min)
            z_max = np.max(z_max)

        return nc, np.max(s_x), np.max(s_y), z_min, z_max

    def convert_qimage_to_numpy(self, qimage):
        """
        Convert qimage to numpy image
        """
        qimage = qimage.convertToFormat(4)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        arr = arr / np.max(arr) * 255
        return arr.astype("|u1")

    def export_roi_ims(self):
        """
        Export for ims
        """
        base, ext = os.path.splitext(self.view.locs_paths[0])
        out_path = base + ".ims"

        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export ROI as ims", out_path, filter="*.ims"
        )

        channel_base, ext_ = os.path.splitext(path)

        if os.path.isfile(path):
            os.remove(path)

        if path:
            status = lib.StatusDialog("Exporting ROIs..", self)

            info = self.view.infos[0]

            n_channels = len(self.view.locs_paths)
            viewport = self.view.viewport
            oversampling = self.view.window.display_settings_dlg.oversampling.value()
            maximum = self.view.window.display_settings_dlg.maximum.value()

            try:
                pixelsize = info[0]["Pixelsize"]
            except KeyError:
                print("Pixelsize not in yaml")
                pixelsize = (
                    self.view.window.display_settings_dlg.pixelsize.value()
                )  # self.pixelsize

            ims_fields = {"ExtMin0": 0, "ExtMin1": 0, "ExtMin2": -0.5, "ExtMax2": 0.5}

            for k, v in ims_fields.items():
                try:
                    val = info[0][k]
                    ims_fields[k] = None
                except KeyError:
                    pass

            x_min = viewport[0][1]
            x_max = viewport[1][1]
            y_min = viewport[0][0]
            y_max = viewport[1][0]

            z_mins = []
            z_maxs = []
            to_render = []

            has_z = True

            for channel in range(n_channels):
                if self.view.window.dataset_dialog.checks[channel].isChecked():
                    locs = self.view.locs[channel]

                    in_view = (
                        (locs.x > x_min)
                        & (locs.x <= x_max)
                        & (locs.y > y_min)
                        & (locs.y <= y_max)
                    )

                    print(f"Exporting locs in view {channel}")

                    add_dict = {}
                    add_dict["Generated by"] = "Picasso Render (IMS Export)"

                    for k, v in ims_fields.items():
                        if v is not None:
                            add_dict[k] = v

                    info = self.view.infos[channel] + [add_dict]
                    io.save_locs(
                        channel_base + f"_ch_{channel}.hdf5", locs[in_view], info
                    )

                    if hasattr(locs, "z"):
                        z_min, z_max = render._render_min_z(
                            locs, x_min, x_max, y_min, y_max
                        )
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

            print(f"Z dimensions {z_min} -> {z_max}")

            COLORS = self.view.read_colors()

            start = time.time()

            all_img = []
            for idx, channel in enumerate(to_render):
                locs = self.view.locs[channel]
                if has_z:
                    n, image = render.render_hist3d(
                        locs,
                        oversampling,
                        y_min,
                        x_min,
                        y_max,
                        x_max,
                        z_min,
                        z_max,
                        pixelsize,
                    )
                else:
                    n, image = render.render_hist(
                        locs, oversampling, y_min, x_min, y_max, x_max
                    )

                image = image / maximum * 65535
                data = image.astype("uint16")

                data = np.rot90(np.fliplr(data))

                base, ext = os.path.splitext(path)
                print(f"{base}")

                all_img.append(data)

            s_image = np.stack(all_img, axis=-1).T.copy()

            print(f"Shape is {s_image.shape}")

            colors = []
            for idx, c in enumerate(to_render):
                color = str(COLORS[c])[1:-1]
                color = color.replace(",", "")
                c_ = color.split(" ")
                colors.append(PW.Color(float(c_[0]), float(c_[1]), float(c_[2]), 1))

            # np.save(base, s_image)

            numpy_to_imaris(
                s_image,
                path,
                colors,
                oversampling,
                viewport,
                info,
                z_min,
                z_max,
                pixelsize,
            )

            print("Wrote to {}".format(path))

            end = time.time()
            print(f"Took {end-start:.2f} seconds.")

            status.close()

    def load_picks(self):
        path, ext = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pick regions", filter="*.yaml"
        )
        if path:
            self.view.load_picks(path)

    def substract_picks(self):
        if self.view._picks:
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load pick regions", filter="*.yaml"
            )
            if path:
                self.view.substract_picks(path)

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
                    self.view.locs[channel][var_1] = self.view.locs[channel][var_2]
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
            lib.ensure_sanity(self.view.locs[channel], self.view.infos[channel])
            self.view.index_blocks[channel] = None
            self.view.update_scene()

    def open_file_dialog(self):
        if self.pwd == []:
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Add localizations", filter="*.hdf5"
            )
        else:
            path, ext = QtWidgets.QFileDialog.getOpenFileName(
                self, "Add localizations", directory=self.pwd, filter="*.hdf5"
            )
        if path:
            self.pwd = path
            self.view.add_multiple([path])

    def resizeEvent(self, event):
        self.update_info()

    def remove_group(self):
        channel = self.view.get_channel("Remove group")
        self.view.locs[channel] = lib.remove_from_rec(self.view.locs[channel], "group")
        self.view.update_scene
        self.view.zoom_in()
        self.view.zoom_out()

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
                templocs = stack_arrays(templocs, asrecarray=True, usemask=False)
                print(locs)
                print(templocs)
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
                        base, ext = os.path.splitext(self.view.locs_paths[channel])
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
                        base, ext = os.path.splitext(self.view.locs_paths[channel])
                        out_path = base + suffix + ".hdf5"
                        info = self.view.infos[channel] + [
                            {
                                "Generated by": "Picasso Render",
                                "Last driftfile": self.view._driftfiles[channel],
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
        self.info_dialog.width_label.setText("{} pixel".format((self.view.width())))
        self.info_dialog.height_label.setText("{} pixel".format((self.view.height())))
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
            self.info_dialog.fit_precision.setText(
                "{:.3} pixel".format(self.view.median_lp)
            )
        except AttributeError:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        QtCore.QCoreApplication.instance().processEvents()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(window, "An error occured", message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
