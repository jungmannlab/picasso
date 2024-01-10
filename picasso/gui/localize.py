#!/usr/bin/env python
"""
    gui/localize
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for localizing single molecules

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2015-2019
    :copyright: Copyright (c) 2015-2019 Jungmann Lab, MPI of Biochemistry
"""

import os.path
import sys
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import numpy as np
import traceback
import importlib, pkgutil
from .. import io, localize, gausslq, gaussmle, zfit, lib, CONFIG, avgroi, __version__
from collections import UserDict

try:
    from pygpufit import gpufit
    gpufit_installed = True
except ImportError:
    gpufit_installed = False


CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
DEFAULT_PARAMETERS = {"Box Size": 7, "Min. Net Gradient": 5000}


class RubberBand(QtWidgets.QRubberBand):
    def __init__(self, parent):
        super().__init__(QtWidgets.QRubberBand.Rectangle, parent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        color = QtGui.QColor(QtCore.Qt.blue)
        painter.setPen(QtGui.QPen(color))
        rect = event.rect()
        rect.setHeight(int(rect.height() - 1))
        rect.setWidth(int(rect.width() - 1))
        painter.drawRect(rect)


class View(QtWidgets.QGraphicsView):
    """The central widget which shows `Scene` objects of individual frames"""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setAcceptDrops(True)
        self.pan = False
        self.hscrollbar = self.horizontalScrollBar()
        self.vscrollbar = self.verticalScrollBar()
        self.rubberband = RubberBand(self)
        self.roi = None
        self.numeric_roi = False

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and not self.numeric_roi:
            self.roi_origin = QtCore.QPoint(event.pos())
            self.rubberband.setGeometry(QtCore.QRect(self.roi_origin, QtCore.QSize()))
            self.rubberband.show()
        elif event.button() == QtCore.Qt.RightButton:
            self.pan = True
            self.pan_start_x = event.x()
            self.pan_start_y = event.y()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton and not self.numeric_roi:
            self.rubberband.setGeometry(QtCore.QRect(self.roi_origin, event.pos()))
        if self.pan:
            self.hscrollbar.setValue(
                self.hscrollbar.value() - event.x() + self.pan_start_x
            )
            self.vscrollbar.setValue(
                self.vscrollbar.value() - event.y() + self.pan_start_y
            )
            self.pan_start_x = event.x()
            self.pan_start_y = event.y()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and not self.numeric_roi:
            self.roi_end = QtCore.QPoint(event.pos())
            dx = abs(self.roi_end.x() - self.roi_origin.x())
            dy = abs(self.roi_end.y() - self.roi_origin.y())
            if dx < 10 or dy < 10:
                self.roi = None
                self.rubberband.hide()
                self.window.parameters_dialog.roi_edit.setText("")
            else:
                roi_points = (
                    self.mapToScene(self.roi_origin),
                    self.mapToScene(self.roi_end),
                )
                self.roi = [[int(_.y()), int(_.x())] for _ in roi_points]
                (y_min, x_min), (y_max, x_max) = self.roi
                self.window.parameters_dialog.roi_edit.setText(
                    f"{y_min},{x_min},{y_max},{x_max}"
                )
                self.numeric_roi = False
            self.window.draw_frame()
        elif event.button() == QtCore.Qt.RightButton:
            self.pan = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
        else:
            event.ignore()

    def wheelEvent(self, event):
        """Implements zoooming with the mouse wheel"""
        scale = 1.008 ** (-event.angleDelta().y())
        self.scale(scale, scale)


class Scene(QtWidgets.QGraphicsScene):
    """
    Scenes render individual frames and can be displayed in a `View` widget
    """

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.dragMoveEvent = self.dragEnterEvent

    def path_from_drop(self, event):
        url = event.mimeData().urls()[0]
        path = url.toLocalFile()
        base, extension = os.path.splitext(path)
        return path, extension

    def drop_has_valid_url(self, event):
        if not event.mimeData().hasUrls():
            return False
        path, extension = self.path_from_drop(event)

        if extension.lower() not in [".raw", ".tif", ".ims", ".nd2"]:
            return False
        return True

    def dragEnterEvent(self, event):
        if self.drop_has_valid_url(event):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Loads when dropped into the scene."""
        path, ext = self.path_from_drop(event)
        self.window.open(path)


class FitMarker(QtWidgets.QGraphicsItemGroup):
    def __init__(self, x, y, size, parent=None):
        super().__init__(parent)
        L = size / 2
        line1 = QtWidgets.QGraphicsLineItem(x - L, y - L, x + L, y + L)
        line1.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line1)
        line2 = QtWidgets.QGraphicsLineItem(x - L, y + L, x + L, y - L)
        line2.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line2)


class OddSpinBox(QtWidgets.QSpinBox):
    """A spinbox that allows only odd numbers"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSingleStep(2)
        self.editingFinished.connect(self.on_editing_finished)

    def on_editing_finished(self):
        value = self.value()
        if value % 2 == 0:
            self.setValue(int(value + 1))


class CamSettingComboBox(QtWidgets.QComboBox):
    """
    Attributes:
        cam_combos : dict
            keys: possible cameras
            values: list of CamSettingComboBoxes; one for each
                sensitivity category, described in the CONFIG entry for the
                respective camera.
        camera : str
            camera name this CamSettingComboBox belongs to
        categories : list of str
            the sensitivity categories of the camera
        index : int
            the index of sensitivity category this CamSettingComboBox belongs to
    """
    def __init__(self, cam_combos, camera, index, sensitivity_categories=[]):
        super().__init__()
        self.cam_combos = cam_combos
        self.camera = camera
        self.index = index
        self.categories = sensitivity_categories

    def change_target_choices(self, index):
        cam_combos = self.cam_combos[self.camera]
        sensitivity = CONFIG["Cameras"][self.camera]["Sensitivity"]
        for i in range(self.index + 1):
            sensitivity = sensitivity[cam_combos[i].currentText()]
        if len(cam_combos) > self.index + 1:
            target = cam_combos[self.index + 1]
            target.blockSignals(True)
            target.clear()
            target.blockSignals(False)
            target.addItems(sorted(list(sensitivity.keys())))


class CamSettingComboBoxDict(UserDict):
    """Dictionary holding CamSettingComboBoxes for different cameras and
    sensitivity categories.
    keys: str
        cameras
    values: List of CamSettingComboBoxes, one for each sensitivity category
        of this camera

    further attributes:
        sensitivity_categories : dict
            keys: cameras
            values: list of str: the categories
    """
    def __init__(self):
        super().__init__()
        self.sensitivity_categories = {}

    def add_categories(self, cam, categories):
        """Call this when setting combo boxes for a new camera, to accompany
        it with the corresponding sensitivity categories.
        """
        self.sensitivity_categories[cam] = categories

    def set_camcombo_value(self, cam, category, value):
        """Sets the value of one combo box

        Args:
            cam : str
                the camera to set
            category : str
                the category combo box to set
            value : str
                the value to set
        """
        cat_idx = self.sensitivity_categories[cam].index(category)
        cam_combo = self.data[cam][cat_idx]
        for index in range(cam_combo.count()):
            if cam_combo.itemText(index) == value:
                cam_combo.setCurrentIndex(index)
                break

    def set_camcombo_values(self, cam, values):
        """Sets the values of all combo boxes of a camera

        Args:
            cam : str
                the camera to set
            values : dict
                keys: sensitivity categories
                values: the values to set
        """
        for i, cat in enumerate(self.sensitivity_categories[cam]):
            if cat in values:
                cam_combo = self.data[cam][i]
                for index in range(cam_combo.count()):
                    if cam_combo.itemText(index) == values[cat]:
                        cam_combo.setCurrentIndex(index)
                        break


class EmissionComboBoxDict(UserDict):
    """Dictionary holding ComboBoxes for different emission wavelenghts.
    keys: str
        cameras
    values: ComboBoxes with wavelengths
    """
    def __init__(self):
        super().__init__()

    def set_emcombo_value(self, cam, wavelength):
        """Sets the value of one combo box

        Args:
            cam : str
                the camera to set
            wavelength : str
                the value to set
        """
        em_combo = self.data[cam]
        for index in range(em_combo.count()):
            if em_combo.itemText(index) == wavelength:
                em_combo.setCurrentIndex(index)
                break


class PromptInfoDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter movie info")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Byte Order:"), 0, 0)
        self.byte_order = QtWidgets.QComboBox()
        self.byte_order.addItems(["Little Endian (loads faster)", "Big Endian"])
        grid.addWidget(self.byte_order, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Data Type:"), 1, 0)
        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(
            [
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "uint8",
                "uint16",
                "uint32",
            ]
        )
        grid.addWidget(self.dtype, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Frames:"), 2, 0)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, int(1e9))
        grid.addWidget(self.frames, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Height:"), 3, 0)
        self.movie_height = QtWidgets.QSpinBox()
        self.movie_height.setRange(1, int(1e9))
        grid.addWidget(self.movie_height, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Width"), 4, 0)
        self.movie_width = QtWidgets.QSpinBox()
        self.movie_width.setRange(1, int(1e9))
        grid.addWidget(self.movie_width, 4, 1)
        self.save = QtWidgets.QCheckBox("Save info to yaml file")
        self.save.setChecked(True)
        grid.addWidget(self.save, 5, 0, 1, 2)
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

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getMovieSpecs(parent=None):
        dialog = PromptInfoDialog(parent)
        result = dialog.exec_()
        info = {}
        info["Byte Order"] = (
            ">" if dialog.byte_order.currentText() == "Big Endian" else "<"
        )
        info["Data Type"] = dialog.dtype.currentText()
        info["Frames"] = dialog.frames.value()
        info["Height"] = dialog.movie_height.value()
        info["Width"] = dialog.movie_width.value()
        save = dialog.save.isChecked()
        return (info, save, result == QtWidgets.QDialog.Accepted)


class PromptChannelDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Select channel")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Channel:"), 0, 0)
        self.byte_order = QtWidgets.QComboBox()

        grid.addWidget(self.byte_order, 0, 1)

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

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getMovieSpecs(parent=None, channels=None):
        dialog = PromptChannelDialog(parent)
        dialog.byte_order.addItems(channels)
        result = dialog.exec_()
        channel = dialog.byte_order.currentText()
        return (channel, result == QtWidgets.QDialog.Accepted)


class ParametersDialog(QtWidgets.QDialog):
    """The dialog showing analysis parameters"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Parameters")
        self.resize(300, 0)
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)
        identification_groupbox = QtWidgets.QGroupBox("Identification")
        vbox.addWidget(identification_groupbox)
        identification_grid = QtWidgets.QGridLayout(identification_groupbox)

        # Box Size
        identification_grid.addWidget(QtWidgets.QLabel("Box side length:"), 0, 0)
        self.box_spinbox = OddSpinBox()
        self.box_spinbox.setKeyboardTracking(False)
        self.box_spinbox.setValue(DEFAULT_PARAMETERS["Box Size"])
        self.box_spinbox.valueChanged.connect(self.on_box_changed)
        identification_grid.addWidget(self.box_spinbox, 0, 1)

        # Min. Net Gradient
        identification_grid.addWidget(QtWidgets.QLabel("Min.  Net Gradient:"), 1, 0)
        self.mng_spinbox = QtWidgets.QSpinBox()
        self.mng_spinbox.setRange(0, int(1e9))
        self.mng_spinbox.setValue(DEFAULT_PARAMETERS["Min. Net Gradient"])
        self.mng_spinbox.setKeyboardTracking(False)
        self.mng_spinbox.valueChanged.connect(self.on_mng_spinbox_changed)
        identification_grid.addWidget(self.mng_spinbox, 1, 1)

        # Slider
        self.mng_slider = QtWidgets.QSlider()
        self.mng_slider.setOrientation(QtCore.Qt.Horizontal)
        self.mng_slider.setRange(0, 10000)
        self.mng_slider.setValue(DEFAULT_PARAMETERS["Min. Net Gradient"])
        self.mng_slider.setSingleStep(1)
        self.mng_slider.setPageStep(20)
        self.mng_slider.valueChanged.connect(self.on_mng_slider_changed)
        identification_grid.addWidget(self.mng_slider, 2, 0, 1, 2)

        hbox = QtWidgets.QHBoxLayout()
        identification_grid.addLayout(hbox, 3, 0, 1, 2)

        # Min SpinBox
        self.mng_min_spinbox = QtWidgets.QSpinBox()
        self.mng_min_spinbox.setRange(0, 999999)
        self.mng_min_spinbox.setKeyboardTracking(False)
        self.mng_min_spinbox.setValue(0)
        self.mng_min_spinbox.valueChanged.connect(self.on_mng_min_changed)
        hbox.addWidget(self.mng_min_spinbox)

        hbox.addStretch(1)

        # Max SpinBox
        self.mng_max_spinbox = QtWidgets.QSpinBox()
        self.mng_max_spinbox.setKeyboardTracking(False)
        self.mng_max_spinbox.setRange(0, 999999)
        self.mng_max_spinbox.setValue(10000)
        self.mng_max_spinbox.valueChanged.connect(self.on_mng_max_changed)
        hbox.addWidget(self.mng_max_spinbox)

        # ROI
        identification_grid.addWidget(
            QtWidgets.QLabel("ROI (y_min,x_min,y_max,x_max):"), 5, 0,
        )
        self.roi_edit = QtWidgets.QLineEdit()
        regex = r"\d+,\d+,\d+,\d+" # regex for 4 integers separated by commas
        validator = QtGui.QRegExpValidator(QtCore.QRegExp(regex))
        self.roi_edit.setValidator(validator)
        self.roi_edit.editingFinished.connect(self.on_roi_edit_finished)
        self.roi_edit.textChanged.connect(self.on_roi_edit_changed)
        identification_grid.addWidget(self.roi_edit, 5, 1)

        self.preview_checkbox = QtWidgets.QCheckBox("Preview")
        self.preview_checkbox.setTristate(False)
        self.preview_checkbox.stateChanged.connect(self.on_preview_changed)
        identification_grid.addWidget(self.preview_checkbox, 4, 0)
        # identification_grid.addWidget(self.preview_checkbox, 5, 0)

        # Camera:
        if "Cameras" in CONFIG:
            # Experiment settings
            exp_groupbox = QtWidgets.QGroupBox("Experiment settings")
            vbox.addWidget(exp_groupbox)
            exp_grid = QtWidgets.QGridLayout(exp_groupbox)
            exp_grid.addWidget(QtWidgets.QLabel("Camera:"), 0, 0)
            self.camera = QtWidgets.QComboBox()
            exp_grid.addWidget(self.camera, 0, 1)
            cameras = sorted(list(CONFIG["Cameras"].keys()))
            self.camera.addItems(cameras)
            self.camera.currentIndexChanged.connect(self.on_camera_changed)

            self.cam_settings = QtWidgets.QStackedWidget()
            exp_grid.addWidget(self.cam_settings, 1, 0, 1, 2)
            self.cam_combos = CamSettingComboBoxDict()
            self.emission_combos = EmissionComboBoxDict()
            for cam in cameras:
                cam_widget = QtWidgets.QWidget()
                cam_grid = QtWidgets.QGridLayout(cam_widget)
                self.cam_settings.addWidget(cam_widget)
                cam_config = CONFIG["Cameras"][cam]
                if "Sensitivity" in cam_config:
                    sensitivity = cam_config["Sensitivity"]
                    if "Sensitivity Categories" in cam_config:
                        self.cam_combos[cam] = []
                        categories = cam_config["Sensitivity Categories"]
                        self.cam_combos.add_categories(cam, categories)
                        for i, category in enumerate(categories):
                            row_count = cam_grid.rowCount()
                            cam_grid.addWidget(
                                QtWidgets.QLabel(category + ":"), row_count, 0
                            )
                            cat_combo = CamSettingComboBox(self.cam_combos, cam, i)
                            cam_grid.addWidget(cat_combo, row_count, 1)
                            self.cam_combos[cam].append(cat_combo)
                        self.cam_combos[cam][0].addItems(
                            sorted(list(sensitivity.keys()))
                        )
                        for cam_combo in self.cam_combos[cam][:-1]:
                            cam_combo.currentIndexChanged.connect(
                                cam_combo.change_target_choices
                            )
                        self.cam_combos[cam][0].change_target_choices(0)
                        self.cam_combos[cam][-1].currentIndexChanged.connect(
                            self.update_sensitivity
                        )
                if "Quantum Efficiency" in cam_config:
                    try:
                        qes = cam_config["Quantum Efficiency"].keys()
                    except AttributeError:
                        pass
                    else:
                        row_count = cam_grid.rowCount()
                        cam_grid.addWidget(
                            QtWidgets.QLabel("Emission Wavelength:"), row_count, 0
                        )
                        emission_combo = QtWidgets.QComboBox()
                        cam_grid.addWidget(emission_combo, row_count, 1)
                        wavelengths = sorted([str(_) for _ in qes])
                        emission_combo.addItems(wavelengths)
                        emission_combo.currentIndexChanged.connect(
                            self.on_emission_changed
                        )
                        self.emission_combos[cam] = emission_combo
                spacer = QtWidgets.QWidget()
                spacer.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
                )
                cam_grid.addWidget(spacer, cam_grid.rowCount(), 0)

        # Photon conversion
        photon_groupbox = QtWidgets.QGroupBox("Photon Conversion")
        vbox.addWidget(photon_groupbox)
        photon_grid = QtWidgets.QGridLayout(photon_groupbox)

        # EM Gain
        photon_grid.addWidget(QtWidgets.QLabel("EM Gain:"), 0, 0)
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(1, int(1e6))
        self.gain.setValue(1)
        photon_grid.addWidget(self.gain, 0, 1)

        # Baseline
        photon_grid.addWidget(QtWidgets.QLabel("Baseline:"), 1, 0)
        self.baseline = QtWidgets.QDoubleSpinBox()
        self.baseline.setRange(0, 1e6)
        self.baseline.setValue(100.0)
        self.baseline.setDecimals(1)
        self.baseline.setSingleStep(0.1)
        photon_grid.addWidget(self.baseline, 1, 1)

        # Sensitivity
        photon_grid.addWidget(QtWidgets.QLabel("Sensitivity:"), 2, 0)
        self.sensitivity = QtWidgets.QDoubleSpinBox()
        self.sensitivity.setRange(0, 1e6)
        self.sensitivity.setValue(1.0)
        self.sensitivity.setDecimals(4)
        self.sensitivity.setSingleStep(0.01)
        photon_grid.addWidget(self.sensitivity, 2, 1)

        # QE
        photon_grid.addWidget(QtWidgets.QLabel("Quantum Efficiency:"), 3, 0)
        self.qe = QtWidgets.QDoubleSpinBox()
        self.qe.setRange(0, 1)
        self.qe.setValue(1)
        self.qe.setDecimals(2)
        self.qe.setSingleStep(0.1)
        photon_grid.addWidget(self.qe, 3, 1)

        # QE
        photon_grid.addWidget(QtWidgets.QLabel("Pixelsize (nm):"), 4, 0)
        self.pixelsize = QtWidgets.QSpinBox()
        self.pixelsize.setRange(0, 10000)
        self.pixelsize.setValue(130)
        self.pixelsize.setSingleStep(1)
        photon_grid.addWidget(self.pixelsize, 4, 1)

        # Fit Settings
        fit_groupbox = QtWidgets.QGroupBox("Fit Settings")
        vbox.addWidget(fit_groupbox)
        fit_grid = QtWidgets.QGridLayout(fit_groupbox)

        fit_grid.addWidget(QtWidgets.QLabel("Method:"), 1, 0)
        self.fit_method = QtWidgets.QComboBox()
        self.fit_method.addItems(
            ["LQ, Gaussian", "MLE, integrated Gaussian", "Average of ROI"]
        )
        self.fit_method.setCurrentIndex(0)
        fit_grid.addWidget(self.fit_method, 1, 1)
        fit_stack = QtWidgets.QStackedWidget()
        fit_grid.addWidget(fit_stack, 2, 0, 1, 2)
        self.fit_method.currentIndexChanged.connect(fit_stack.setCurrentIndex)
        self.fit_method.currentIndexChanged.connect(self.on_fit_method_changed)

        # MLE
        mle_widget = QtWidgets.QWidget()

        mle_grid = QtWidgets.QGridLayout(mle_widget)
        mle_grid.addWidget(QtWidgets.QLabel("Convergence criterion:"), 0, 0)
        self.convergence_criterion = QtWidgets.QDoubleSpinBox()
        self.convergence_criterion.setRange(0, 1e6)
        self.convergence_criterion.setDecimals(6)
        self.convergence_criterion.setValue(0.001)
        mle_grid.addWidget(self.convergence_criterion, 0, 1)
        mle_grid.addWidget(QtWidgets.QLabel("Max. iterations:"), 1, 0)
        self.max_it = QtWidgets.QSpinBox()
        self.max_it.setRange(1, int(1e6))
        self.max_it.setValue(1000)
        mle_grid.addWidget(self.max_it, 1, 1)

        # LQ
        lq_widget = QtWidgets.QWidget()
        lq_grid = QtWidgets.QGridLayout(lq_widget)

        self.gpufit_checkbox = QtWidgets.QCheckBox("Use GPUfit")
        self.gpufit_checkbox.setTristate(False)
        self.gpufit_checkbox.setDisabled(True)
        self.gpufit_checkbox.stateChanged.connect(self.on_gpufit_changed)

        if not gpufit_installed:
            self.gpufit_checkbox.hide()
        else:
            self.gpufit_checkbox.setDisabled(False)
        lq_grid.addWidget(self.gpufit_checkbox)

        fit_stack.addWidget(lq_widget)
        fit_stack.addWidget(mle_widget)
        # lq_grid = QtWidgets.QGridLayout(lq_widget)

        avg_widget = QtWidgets.QWidget()
        fit_stack.addWidget(avg_widget)

        # 3D
        z_groupbox = QtWidgets.QGroupBox("3D via Astigmatism")
        vbox.addWidget(z_groupbox)

        z_grid = QtWidgets.QGridLayout(z_groupbox)
        z_grid.addWidget(
            QtWidgets.QLabel("Non-integrated Gaussian fitting is recommend! (LQ)"),
            0, 0, 1, 2,
        )
        load_z_calib = QtWidgets.QPushButton("Load calibration")
        load_z_calib.setAutoDefault(False)
        load_z_calib.clicked.connect(self.load_z_calib)
        z_grid.addWidget(load_z_calib, 1, 1)
        self.fit_z_checkbox = QtWidgets.QCheckBox("Fit Z")
        self.fit_z_checkbox.setEnabled(False)
        z_grid.addWidget(self.fit_z_checkbox, 3, 1)
        self.z_calib_label = QtWidgets.QLabel("-- no calibration loaded --")
        self.z_calib_label.setAlignment(QtCore.Qt.AlignCenter)
        self.z_calib_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
        z_grid.addWidget(self.z_calib_label, 1, 0)
        z_grid.addWidget(QtWidgets.QLabel("Magnification factor:"), 2, 0)
        self.magnification_factor = QtWidgets.QDoubleSpinBox()
        self.magnification_factor.setRange(0, 1e6)
        self.magnification_factor.setDecimals(4)
        self.magnification_factor.setValue(0.79)
        z_grid.addWidget(self.magnification_factor, 2, 1)

        if "Cameras" in CONFIG:
            camera = self.camera.currentText()
            if camera in CONFIG["Cameras"]:
                self.on_camera_changed(0)
                camera_config = CONFIG["Cameras"][camera]
                if (
                    "Sensitivity" in camera_config
                    and "Sensitivity Categories" in camera_config
                ):
                    self.update_sensitivity()

        astig_groupbox = QtWidgets.QGroupBox("Astigmatism-Correction")
        vbox.addWidget(astig_groupbox)
        astig_grid = QtWidgets.QGridLayout(astig_groupbox)
        load_astig_calib = QtWidgets.QPushButton("Load correction")
        load_astig_calib.setAutoDefault(False)
        load_astig_calib.clicked.connect(self.load_astig_calib)

        astig_groupbox.setVisible(False)

        astig_grid.addWidget(load_astig_calib, 1, 1)
        self.calib_astig_checkbox = QtWidgets.QCheckBox("Correct Astigmatism")
        self.calib_astig_checkbox.setEnabled(False)

        astig_grid.addWidget(self.calib_astig_checkbox, 3, 1)
        self.astig_label = QtWidgets.QLabel("-- no calibration loaded --")
        self.astig_label.setAlignment(QtCore.Qt.AlignCenter)
        self.astig_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
        astig_grid.addWidget(self.astig_label, 1, 0)

        self.vbox = vbox
        self.imsgrid = False

        # Sample quality
        self.quality_groupbox = QtWidgets.QGroupBox("Sample Quality")
        vbox.addWidget(self.quality_groupbox)

        self.quality_grid = QtWidgets.QGridLayout(self.quality_groupbox)
        self.quality_check = QtWidgets.QPushButton(
            "Estimate and add to database"
        )
        self.quality_check.setEnabled(False)
        self.quality_grid.addWidget(self.quality_check, 1, 2)
        self.quality_check.clicked.connect(self.check_quality)

        self.quality_grid_labels = [
            QtWidgets.QLabel("Locs/Frame"),
            QtWidgets.QLabel("NeNA"),
            QtWidgets.QLabel("Mean Drift"),
            QtWidgets.QLabel("Bright Time (Frames)"),
        ]
        for idx, _ in enumerate(self.quality_grid_labels):
            self.quality_grid.addWidget(_, idx + 1, 1)

        self.quality_grid_values = [
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
        ]

        for idx, _ in enumerate(self.quality_grid_values):
            self.quality_grid.addWidget(_, idx + 1, 2)

        self.reset_quality_check()

    def reset_quality_check(self):
        self.quality_check.setEnabled(False)
        self.quality_check.setVisible(True)

        for _ in self.quality_grid_labels:
            _.setVisible(False)

        for _ in self.quality_grid_values:
            _.setVisible(False)
            _.setText("")

    def on_roi_edit_changed(self):
        if self.roi_edit.text() == "":
            self.window.view.numeric_roi = False            
            self.window.view.roi = None
            self.window.view.rubberband.hide()
            self.window.draw_frame()

    def on_roi_edit_finished(self):
        text = self.roi_edit.text().split(",")
        y_min, x_min, y_max, x_max = [int(_) for _ in text]
        # update roi
        self.window.view.roi = [[y_min, x_min], [y_max, x_max]]
        # draw the rectangle roi
        topleft_xy = self.window.view.mapFromScene(x_min, y_min)
        bottomright_xy = self.window.view.mapFromScene(x_max, y_max)
        topleft = QtCore.QPoint(topleft_xy.x(), topleft_xy.y())
        bottomright = QtCore.QPoint(bottomright_xy.x(), bottomright_xy.y())
        self.window.view.rubberband.setGeometry(
            QtCore.QRect(topleft, bottomright)
        )
        self.window.view.rubberband.show()
        self.window.draw_frame()
        self.window.view.numeric_roi = True

    def on_fit_method_changed(self, state):
        if self.fit_method.currentText() == "LQ, Gaussian":
            self.gpufit_checkbox.setDisabled(False)
        else:
            self.gpufit_checkbox.setChecked(False)
            self.gpufit_checkbox.setDisabled(True)

    def load_z_calib(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load 3d calibration", directory=None, filter="*.yaml"
        )
        if path:
            with open(path, "r") as f:
                self.z_calibration = yaml.full_load(f)
                self.z_calibration_path = path
            self.z_calib_label.setAlignment(QtCore.Qt.AlignRight)
            self.z_calib_label.setText(os.path.basename(path))
            self.fit_z_checkbox.setEnabled(True)
            self.fit_z_checkbox.setChecked(True)

    def quality_progress(self, msg, index, result):
        if msg != "":
            self.window.status_bar.showMessage(msg)
        else:
            self.quality_grid_values[index].setText(result)

    def quality_progress_finished(self, msg):
        self.window.status_bar.showMessage(msg)

    def check_quality(self):
        print("Assesing Quality of Sample")
        self.quality_check.setVisible(False)
        for idx, _ in enumerate(self.quality_grid_labels):
            _.setVisible(True)
        for idx, _ in enumerate(self.quality_grid_values):
            _.setVisible(True)

        self.q_worker = QualityWorker(
            self.window.locs, self.window.info, self.window.movie_path, self.pixelsize
        )
        self.q_worker.progressMade.connect(self.quality_progress)
        self.q_worker.finished.connect(self.quality_progress_finished)
        self.q_worker.start()

    def load_astig_calib(self):
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load astigmatism calibration", directory=None, filter="*.pkl"
        )
        if path:
            with open(path, "r") as f:
                # self.astig_calibration = yaml.load(f)
                self.astig_calibration_path = path
            self.astig_label.setAlignment(QtCore.Qt.AlignRight)
            self.astig_label.setText(os.path.basename(path))
            self.calib_astig_checkbox.setEnabled(True)
            self.calib_astig_checkbox.setChecked(True)

    def on_box_changed(self, value):
        self.window.on_parameters_changed()

    def on_camera_changed(self, index):
        self.gain.setValue(1)
        self.cam_settings.setCurrentIndex(index)
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        if "Baseline" in cam_config:
            self.baseline.setValue(cam_config["Baseline"])
        if "DefaultGain" in cam_config:
            self.gain.setValue(cam_config["DefaultGain"])
        if "Pixelsize" in cam_config:
            self.pixelsize.setValue(cam_config["Pixelsize"])
        self.update_sensitivity()
        self.update_qe()

    def update_qe(self):
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        if "Quantum Efficiency" in cam_config:
            qe = cam_config["Quantum Efficiency"]
            try:
                self.qe.setValue(qe)
            except TypeError:
                # qe is not a number
                em_combo = self.emission_combos[camera]
                wavelength = float(em_combo.currentText())
                qe = cam_config["Quantum Efficiency"][wavelength]
                self.qe.setValue(qe)

    def on_emission_changed(self, index):
        self.update_qe()

    def on_mng_spinbox_changed(self, value):
        if value < self.mng_slider.minimum():
            self.mng_min_spinbox.setValue(value)
        if value > self.mng_slider.maximum():
            self.mng_max_spinbox.setValue(value)
        self.mng_slider.setValue(value)

    def on_mng_slider_changed(self, value):
        self.mng_spinbox.setValue(value)
        if self.preview_checkbox.isChecked():
            self.window.on_parameters_changed()

    def on_mng_min_changed(self, value):
        self.mng_slider.setMinimum(value)

    def on_mng_max_changed(self, value):
        self.mng_slider.setMaximum(value)

    def on_preview_changed(self, state):
        self.window.draw_frame()

    def on_gpufit_changed(self, state):
        self.window.draw_frame()

    def set_camera_parameters(self, info):
        if "Cameras" in CONFIG and "Camera" in info:
            cameras = [self.camera.itemText(_) for _ in range(self.camera.count())]
            camera = info["Camera"]
            if camera in cameras:
                index = cameras.index(camera)
                self.camera.setCurrentIndex(index)
                if "Micro-Manager Metadata" in info:
                    mm_info = info["Micro-Manager Metadata"]
                    cam_config = CONFIG["Cameras"][camera]
                    if "Gain Property Name" in cam_config:
                        gain_property_name = cam_config["Gain Property Name"]
                        gain = mm_info[camera + "-" + gain_property_name]
                        if "EM Switch Property" in cam_config:
                            switch_property_name = cam_config["EM Switch Property"][
                                "Name"
                            ]
                            switch_property_value = mm_info[
                                camera + "-" + switch_property_name
                            ]
                            if (
                                switch_property_value
                                == cam_config["EM Switch Property"][True]
                            ):
                                self.gain.setValue(int(gain))
                            else:
                                self.gain.setValue(1)
                    if "Sensitivity Categories" in cam_config:
                        cam_combos = self.cam_combos[camera]
                        categories = cam_config["Sensitivity Categories"]
                        for i, category in enumerate(categories):
                            property_name = camera + "-" + category
                            if property_name in mm_info:
                                exp_setting = mm_info[camera + "-" + category]
                                cam_combo = cam_combos[i]
                                for index in range(cam_combo.count()):
                                    if cam_combo.itemText(index) == exp_setting:
                                        cam_combo.setCurrentIndex(index)
                                        break
                    if "Quantum Efficiency" in cam_config:
                        if "Channel Device" in cam_config:
                            channel_device_name = cam_config["Channel Device"]["Name"]
                            channel = mm_info[channel_device_name]
                            channels = cam_config["Channel Device"][
                                "Emission Wavelengths"
                            ]
                            if channel in channels:
                                wavelength = str(channels[channel])
                                em_combo = self.emission_combos[camera]
                                for index in range(em_combo.count()):
                                    if em_combo.itemText(index) == wavelength:
                                        em_combo.setCurrentIndex(index)
                                        break
                                else:
                                    raise ValueError(
                                        (
                                            "No quantum efficiency found"
                                            " for wavelength " + wavelength
                                        )
                                    )

    def update_sensitivity(self, index=None):
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        sensitivity = cam_config["Sensitivity"]
        if "Sensitivity" in cam_config:
            try:
                self.sensitivity.setValue(sensitivity)
            except TypeError:
                # sensitivity is not a number
                categories = cam_config["Sensitivity Categories"]
                for i, category in enumerate(categories):
                    cat_combo = self.cam_combos[camera][i]
                    sensitivity = sensitivity[cat_combo.currentText()]
                self.sensitivity.setValue(sensitivity)


class ContrastDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Contrast")
        self.resize(200, 0)
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)
        black_label = QtWidgets.QLabel("Black:")
        grid.addWidget(black_label, 0, 0)
        self.black_spinbox = QtWidgets.QSpinBox()
        self.black_spinbox.setKeyboardTracking(False)
        self.black_spinbox.setRange(1, 999999)
        self.black_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.black_spinbox, 0, 1)
        white_label = QtWidgets.QLabel("White:")
        grid.addWidget(white_label, 1, 0)
        self.white_spinbox = QtWidgets.QSpinBox()
        self.white_spinbox.setKeyboardTracking(False)
        self.white_spinbox.setRange(1, 999999)
        self.white_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.white_spinbox, 1, 1)
        self.auto_checkbox = QtWidgets.QCheckBox("Auto")
        self.auto_checkbox.setTristate(False)
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.on_auto_changed)
        grid.addWidget(self.auto_checkbox, 2, 0, 1, 2)
        self.silent_contrast_change = False

    def change_contrast_silently(self, black, white):
        self.silent_contrast_change = True
        self.black_spinbox.setValue(black)
        self.white_spinbox.setValue(white)
        self.silent_contrast_change = False

    def on_contrast_changed(self, value):
        if not self.silent_contrast_change:
            self.auto_checkbox.setChecked(False)
            self.window.draw_frame()

    def on_auto_changed(self, state):
        if state:
            movie = self.window.movie
            frame_number = self.window.curr_frame_number
            frame = movie[frame_number]
            self.change_contrast_silently(frame.min(), frame.max())
            self.window.draw_frame()


class Window(QtWidgets.QMainWindow):
    """The main window"""

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle(f"Picasso v{__version__}: Localize")

        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "localize.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(768, 768)
        self.parameters_dialog = ParametersDialog(self)
        self.contrast_dialog = ContrastDialog(self)
        self.init_menu_bar()
        self.view = View(self)
        self.setCentralWidget(self.view)
        self.scene = Scene(self)
        self.view.setScene(self.scene)
        self.status_bar = self.statusBar()
        self.status_bar_frame_indicator = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.status_bar_frame_indicator)

        #: Holds the curr movie as a numpy
        # memmap in the format (frame, y, x)
        self.movie = None

        #: A dictionary of analysis parameters used for the last operation
        self.last_identification_info = None

        #: A numpy.recarray of identifcations with fields frame, x and y
        self.identifications = None

        self.ready_for_fit = False

        self.locs = None

        self.movie_path = []

        # Load user settings
        self.load_user_settings()

    def load_user_settings(self):
        settings = io.load_user_settings()
        pwd = []
        box_size = []
        gradient = []
        try:
            pwd = settings["Localize"]["PWD"]
            box_size = settings["Localize"]["box_size"]
            gradient = settings["Localize"]["gradient"]
        except Exception as e:
            print(e)
            pass
        if len(pwd) == 0:
            pwd = []
        if type(box_size) is int:
            self.parameters_dialog.box_spinbox.setValue(box_size)
        if type(gradient) is int:
            self.parameters_dialog.mng_slider.setValue(gradient)

        self.pwd = pwd

    def closeEvent(self, event):
        settings = io.load_user_settings()
        if self.movie_path != []:
            settings["Localize"]["PWD"] = os.path.dirname(self.movie_path)
            settings["Localize"][
                "box_size"
            ] = self.parameters_dialog.box_spinbox.value()
            settings["Localize"]["gradient"] = self.parameters_dialog.mng_slider.value()
        io.save_user_settings(settings)
        QtWidgets.qApp.closeAllWindows()

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open movie")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        load_picks_action = file_menu.addAction("Load picks as identifications")
        load_picks_action.triggered.connect(self.open_picks)
        load_locs_action = file_menu.addAction("Load locs as identifications")
        load_locs_action.triggered.connect(self.open_locs)
        save_action = file_menu.addAction("Save localizations")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs_dialog)
        file_menu.addAction(save_action)
        save_spots_action = file_menu.addAction("Save spots")
        save_spots_action.setShortcut("Ctrl+Shift+S")
        save_spots_action.triggered.connect(self.save_spots_dialog)
        file_menu.addAction(save_spots_action)
        # file_menu.addSeparator()
        # open_parameters_action = file_menu.addAction('Load parameters')
        # open_parameters_action.setShortcut('Ctrl+Shift+O')
        # open_parameters_action.triggered.connect(self.open_parameters)
        # file_menu.addAction(open_parameters_action)
        # save_parameters_action = file_menu.addAction('Save parameters')
        # save_parameters_action.setShortcut('Ctrl+Shift+S')
        # save_parameters_action.triggered.connect(self.save_parameters)
        # file_menu.addAction(save_parameters_action)

        """ View """
        view_menu = menu_bar.addMenu("View")
        previous_frame_action = view_menu.addAction("Previous frame")
        previous_frame_action.setShortcut("Left")
        previous_frame_action.triggered.connect(self.previous_frame)
        view_menu.addAction(previous_frame_action)
        next_frame_action = view_menu.addAction("Next frame")
        next_frame_action.setShortcut("Right")
        next_frame_action.triggered.connect(self.next_frame)
        view_menu.addAction(next_frame_action)
        view_menu.addSeparator()
        first_frame_action = view_menu.addAction("First frame")
        first_frame_action.setShortcut("Home")
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction("Last frame")
        last_frame_action.setShortcut("End")
        last_frame_action.triggered.connect(self.last_frame)
        view_menu.addAction(last_frame_action)
        go_to_frame_action = view_menu.addAction("Go to frame")
        go_to_frame_action.setShortcut("Ctrl+G")
        go_to_frame_action.triggered.connect(self.to_frame)
        view_menu.addAction(go_to_frame_action)
        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction("Zoom in")
        zoom_in_action.setShortcuts(["Ctrl++", "Ctrl+="])
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction("Zoom out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction("Fit image to window")
        fit_in_view_action.setShortcut("Ctrl+W")
        fit_in_view_action.triggered.connect(self.fit_in_view)
        view_menu.addAction(fit_in_view_action)
        view_menu.addSeparator()
        display_settings_action = view_menu.addAction("Contrast")
        display_settings_action.setShortcut("Ctrl+C")
        display_settings_action.triggered.connect(self.contrast_dialog.show)
        view_menu.addAction(display_settings_action)

        """ Analyze """
        analyze_menu = menu_bar.addMenu("Analyze")
        parameters_action = analyze_menu.addAction("Parameters")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        analyze_menu.addAction(parameters_action)
        analyze_menu.addSeparator()
        identify_action = analyze_menu.addAction("Identify")
        identify_action.setShortcut("Ctrl+I")
        identify_action.triggered.connect(self.identify)
        analyze_menu.addAction(identify_action)
        fit_action = analyze_menu.addAction("Fit")
        fit_action.setShortcut("Ctrl+F")
        fit_action.triggered.connect(self.fit)
        analyze_menu.addAction(fit_action)
        localize_action = analyze_menu.addAction("Localize (Identify && Fit)")
        localize_action.setShortcut("Ctrl+L")
        localize_action.triggered.connect(self.localize)
        analyze_menu.addAction(localize_action)

        """ 3D """
        threed_menu = menu_bar.addMenu("3D")

        calibrate_z_action = threed_menu.addAction("Calibrate 3D")
        calibrate_z_action.triggered.connect(self.calibrate_z)

    @property
    def camera_info(self):
        camera_info = {}
        camera_info["baseline"] = self.parameters_dialog.baseline.value()
        camera_info["gain"] = self.parameters_dialog.gain.value()
        camera_info["sensitivity"] = self.parameters_dialog.sensitivity.value()
        camera_info["qe"] = self.parameters_dialog.qe.value()
        return camera_info

    def calibrate_z(self):
        self.localize(calibrate_z=True)

    def open_file_dialog(self):
        if self.pwd == []:
            dir = None
        else:
            dir = self.pwd

        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Open image sequence", 
            directory=dir, 
            filter=(
                "All supported formats (*.raw *.tif *.tif *.nd2 *.ims)"
                ";;Raw files (*.raw)"
                ";;Tif images (*.tif)"
                ";;ImaRIS IMS (*.ims)"
                ";;Nd2 files (*.nd2);;"
            )
        )
        if path:
            self.pwd = path
            self.open(path)

    def open(self, path):
        t0 = time.time()

        if path.endswith(".ims"):
            prompt_info = self.prompt_channel
        else:
            prompt_info = self.prompt_info

        result = io.load_movie(path, prompt_info=prompt_info)

        if result is not None:
            self.movie, self.info = result
            dt = time.time() - t0
            self.movie_path = path
            self.identifications = None
            self.locs = None
            self.ready_for_fit = False
            self.set_frame(0)
            self.fit_in_view()
            self.parameters_dialog.set_camera_parameters(self.info[0])
            self.status_bar.showMessage("Opened movie in {:.2f} seconds.".format(dt))

            if "Pixelsize" in self.info[0]:
                self.parameters_dialog.pixelsize.setValue(
                    int(self.info[0]["Pixelsize"])
                )

        self.setWindowTitle(
            "Picasso: Localize. File: {}".format(os.path.basename(path))
        )
        self.parameters_dialog.reset_quality_check()

    def open_picks(self):
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open picks", directory=dir, filter="*.yaml"
        )
        if path:
            self.load_picks(path)

    def load_picks(self, path):
        try:
            with open(path, "r") as f:
                regions = yaml.full_load(f)
            self._picks = regions["Centers"]
            maxframes = int(self.info[0]["Frames"])
            # ask for drift correction
            driftpath, exe = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open drift file",
                directory=os.path.dirname(path),
                filter="*.txt",
            )
            if driftpath:
                drift = np.genfromtxt(driftpath)
            data = []
            n_id = 0

            for element in self._picks:
                # drifted:
                xloc = np.ones((maxframes,), dtype=float) * element[0]
                yloc = np.ones((maxframes,), dtype=float) * element[1]
                if driftpath:
                    xloc += drift[:, 1]
                    yloc += drift[:, 0]
                else:
                    pass

                frames = np.arange(maxframes)
                gradient = np.ones(maxframes) + 100
                n_id_all = np.ones(maxframes) + n_id
                temp = np.array([frames, xloc, yloc, gradient, n_id_all])
                data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
                n_id += 1

            data = [item for sublist in data for item in sublist]
            identifications = np.array(
                data,
                dtype=[
                    ("frame", int),
                    ("x", int),
                    ("y", int),
                    ("net_gradient", float),
                    ("n_id", int),
                ],
            )

            self.identifications = identifications.view(np.recarray)
            self.identifications.sort(kind="mergesort", order="frame")

            # remove all identifications that are oob
            box = self.parameters["Box Size"]
            m_size = self.movie.shape
            r = int(box / 2)

            self.identifications = self.identifications[
                (self.identifications.y - r > 0)
                & (self.identifications.x - r > 0)
                & (self.identifications.x + r < m_size[0])
                & (self.identifications.y + r < m_size[1])
            ]

            self.locs = None

            self.loaded_picks = True

            self.last_identification_info = {
                "Box Size": self.parameters_dialog.box_spinbox.value(),
                "Min. Net Gradient": self.parameters_dialog.mng_slider.value(),
            }
            self.ready_for_fit = True
            self.draw_frame()
            self.status_bar.showMessage(
                "Created a total of {} identifications.".format(
                    len(self.identifications)
                )
            )

        except io.NoMetadataFileError:
            return

    def open_locs(self):
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open locs", directory=dir, filter="*.hdf5"
        )
        if path:
            self.load_locs(path)

    def load_locs(self, path):
        try:
            locs, info = io.load_locs(path)

            max_frames = int(self.info[0]["Frames"])
            n_frames, ok = QtWidgets.QInputDialog.getInteger(
                self,
                "Input Dialog",
                "Enter number of frames around localization event:",
                100,
            )

            # driftpath, exe = QtWidgets.QFileDialog.getOpenFileName(self,
            # 'Open drift file', filter='*.txt')
            # if driftpath:
            #    drift = np.genfromtxt(driftpath)
            data = []
            n_id = 0
            for element in locs:
                currframe = element["frame"]
                if currframe > n_frames and currframe < (max_frames - n_frames):
                    xloc = np.ones((2 * n_frames + 1,), dtype=float) * element["x"]
                    yloc = np.ones((2 * n_frames + 1,), dtype=float) * element["y"]
                    frames = np.arange(currframe - n_frames, currframe + n_frames + 1)
                    gradient = np.ones(2 * n_frames + 1) + 100
                    n_id_all = np.ones(2 * n_frames + 1) + n_id
                    temp = np.array([frames, xloc, yloc, gradient, n_id_all])
                    data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
                n_id += 1

            data = [item for sublist in data for item in sublist]
            identifications = np.array(
                data,
                dtype=[
                    ("frame", int),
                    ("x", int),
                    ("y", int),
                    ("net_gradient", float),
                    ("n_id", int),
                ],
            )
            self.identifications = identifications.view(np.recarray)
            self.identifications.sort(kind="mergesort", order="frame")

            # remove all identifications that are oob
            box = self.parameters["Box Size"]
            m_size = self.movie.shape
            r = int(box / 2)

            self.identifications = self.identifications[
                (self.identifications.y - r > 0)
                & (self.identifications.x - r > 0)
                & (self.identifications.x + r < m_size[0])
                & (self.identifications.y + r < m_size[1])
            ]

            self.locs = None

            self.loaded_picks = True

            self.last_identification_info = {
                "Box Size": self.parameters_dialog.box_spinbox.value(),
                "Min. Net Gradient": self.parameters_dialog.mng_slider.value(),
            }
            self.ready_for_fit = True
            self.draw_frame()
            self.status_bar.showMessage(
                "Created a total of {} identifications.".format(
                    len(self.identifications)
                )
            )

        except io.NoMetadataFileError:
            return

    def prompt_info(self):
        info, save, ok = PromptInfoDialog.getMovieSpecs(self)
        if ok:
            return info, save

    def prompt_channel(self, channels):
        channel, ok = PromptChannelDialog.getMovieSpecs(self, channels)
        if ok:
            return channel

    def previous_frame(self):
        if self.movie is not None:
            if self.curr_frame_number > 0:
                self.set_frame(self.curr_frame_number - 1)

    def next_frame(self):
        if self.movie is not None:
            if self.curr_frame_number + 1 < self.info[0]["Frames"]:
                self.set_frame(self.curr_frame_number + 1)

    def first_frame(self):
        if self.movie is not None:
            self.set_frame(0)

    def last_frame(self):
        if self.movie is not None:
            self.set_frame(self.info[0]["Frames"] - 1)

    def to_frame(self):
        if self.movie is not None:
            frames = self.info[0]["Frames"]
            number, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Go to frame",
                "Frame number:",
                self.curr_frame_number + 1,
                1,
                frames,
            )
            if ok:
                self.set_frame(number - 1)

    def set_frame(self, number):
        self.curr_frame_number = number
        if self.contrast_dialog.auto_checkbox.isChecked():
            black = self.movie[number].min()
            white = self.movie[number].max()
            self.contrast_dialog.change_contrast_silently(black, white)
        self.draw_frame()
        self.status_bar_frame_indicator.setText(
            "{:,}/{:,}".format(number + 1, self.info[0]["Frames"])
        )

    def draw_frame(self):
        if self.movie is not None:
            frame = self.movie[self.curr_frame_number]
            frame = frame.astype("float32")
            if self.contrast_dialog.auto_checkbox.isChecked():
                frame -= frame.min()
                frame /= frame.max()
            else:
                frame -= self.contrast_dialog.black_spinbox.value()
                frame /= self.contrast_dialog.white_spinbox.value()
            frame *= 255.0
            frame = np.maximum(frame, 0)
            frame = np.minimum(frame, 255)
            frame = frame.astype("uint8")
            height, width = frame.shape
            image = QtGui.QImage(
                frame.data, width, height, width, QtGui.QImage.Format_Indexed8
            )
            image.setColorTable(CMAP_GRAYSCALE)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.scene = Scene(self)
            self.scene.addPixmap(pixmap)
            self.view.setScene(self.scene)
            if self.ready_for_fit:
                identifications_frame = self.identifications[
                    self.identifications.frame == self.curr_frame_number
                ]
                box = self.last_identification_info["Box Size"]
                self.draw_identifications(
                    identifications_frame, box, QtGui.QColor("yellow")
                )
            else:
                if self.parameters_dialog.preview_checkbox.isChecked():
                    identifications_frame = localize.identify_by_frame_number(
                        self.movie,
                        self.parameters["Min. Net Gradient"],
                        self.parameters["Box Size"],
                        self.curr_frame_number,
                        self.view.roi,
                    )
                    box = self.parameters["Box Size"]
                    self.status_bar.showMessage(
                        "Found {:,} spots in curr frame.".format(
                            len(identifications_frame)
                        )
                    )
                    self.draw_identifications(
                        identifications_frame, box, QtGui.QColor("red")
                    )
                else:
                    self.status_bar.showMessage("")
            if self.locs is not None:
                locs_frame = self.locs[self.locs.frame == self.curr_frame_number]
                for loc in locs_frame:
                    self.scene.addItem(FitMarker(loc.x + 0.5, loc.y + 0.5, 1))

    def draw_identifications(self, identifications, box, color):
        box_half = int(box / 2)
        for identification in identifications:
            x = identification.x
            y = identification.y
            self.scene.addRect(x - box_half, y - box_half, box, box, color)

    def open_parameters(self):
        if self.pwd == []:
            path, exe = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open parameters", filter="*.yaml"
            )
        else:
            path, exe = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open parameters", directory=self.pwd, filter="*.yaml"
            )
        if path:
            self.load_parameters(path)

    def load_parameters(self, path):
        with open(path, "r") as file:
            parameters = yaml.full_load(file)
            self.parameters_dialog.box_spinbox.setValue(parameters["Box Size"])
            self.parameters_dialog.mng_spinbox.setValue(parameters["Min. Net Gradient"])
            self.status_bar.showMessage("Parameter file {} loaded.".format(path))

    def save_parameters(self):
        path, exe = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save parameters", filter="*.yaml"
        )
        if path:
            with open(path, "w") as file:
                yaml.dump(self.parameters, file)

    @property
    def parameters(self):
        return {
            "Box Size": self.parameters_dialog.box_spinbox.value(),
            "Min. Net Gradient": self.parameters_dialog.mng_slider.value(),
        }

    def on_parameters_changed(self):
        self.locs = None
        self.ready_for_fit = False
        self.draw_frame()

    def identify(self, fit_afterwards=False, calibrate_z=False):
        if self.movie is not None:
            self.status_bar.showMessage("Preparing identification...")
            self.identificaton_worker = IdentificationWorker(
                self, fit_afterwards, calibrate_z
            )
            self.identificaton_worker.progressMade.connect(self.on_identify_progress)
            self.identificaton_worker.finished.connect(self.on_identify_finished)
            self.identificaton_worker.start()

    def on_identify_progress(self, frame_number, parameters):
        n_frames = self.info[0]["Frames"]
        box = parameters["Box Size"]
        mng = parameters["Min. Net Gradient"]
        message = (
            "Identifying in frame {:,} / {:,}"
            " (Box Size: {:,}; Min. Net Gradient: {:,}) ..."
        ).format(frame_number, n_frames, box, mng)
        self.status_bar.showMessage(message)

    def on_identify_finished(
        self, parameters, roi, identifications, fit_afterwards, calibrate_z
    ):
        if len(identifications):
            self.locs = None
            self.last_identification_info = parameters.copy()
            self.last_identification_info["ROI"] = roi
            n_identifications = len(identifications)
            box = parameters["Box Size"]
            mng = parameters["Min. Net Gradient"]
            message = (
                "Identified {:,} spots (Box Size: {:,}; "
                "Min. Net Gradient: {:,}). Ready for fit."
            ).format(n_identifications, box, mng)
            self.status_bar.showMessage(message)
            self.identifications = identifications
            self.ready_for_fit = True
            self.draw_frame()
            if fit_afterwards:
                self.fit(calibrate_z=calibrate_z)

    def fit(self, calibrate_z=False):
        if self.movie is not None and self.ready_for_fit:
            self.status_bar.showMessage("Preparing fit...")
            method = self.parameters_dialog.fit_method.currentText()
            method = {
                "LQ, Gaussian": "lq",
                "MLE, integrated Gaussian": "mle",
                "Average of ROI": "avg",
            }[method]
            eps = self.parameters_dialog.convergence_criterion.value()
            max_it = self.parameters_dialog.max_it.value()
            fit_z = self.parameters_dialog.fit_z_checkbox.isChecked()
            use_gpufit = self.parameters_dialog.gpufit_checkbox.isChecked()
            self.fit_worker = FitWorker(
                self.movie,
                self.camera_info,
                self.identifications,
                self.parameters["Box Size"],
                method,
                eps,
                max_it,
                fit_z,
                calibrate_z,
                use_gpufit,
            )
            self.fit_worker.progressMade.connect(self.on_fit_progress)
            self.fit_worker.finished.connect(self.on_fit_finished)
            self.fit_worker.start()

    def fit_z(self):
        self.status_bar.showMessage("Fitting z position...")
        self.fit_z_worker = FitZWorker(
            self.locs,
            self.info,
            self.parameters_dialog.z_calibration,
            self.parameters_dialog.magnification_factor.value(),
        )
        self.fit_z_worker.progressMade.connect(self.on_fit_z_progress)
        self.fit_z_worker.finished.connect(self.on_fit_z_finished)
        self.fit_z_worker.start()

    def on_fit_progress(self, curr, total):
        if self.parameters_dialog.gpufit_checkbox.isChecked():
            self.status_bar.showMessage("Fitting spots by GPUfit...")
        else:
            message = "Fitting spot {:,} / {:,} ...".format(curr, total)
            self.status_bar.showMessage(message)

    def on_fit_finished(self, locs, elapsed_time, fit_z, calibrate_z):
        self.status_bar.showMessage(
            "Fitted {:,} spots in {:.2f} seconds.".format(len(locs), elapsed_time)
        )
        self.locs = locs
        self.draw_frame()
        base, ext = os.path.splitext(self.movie_path)
        if calibrate_z:
            step, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "3D Calibration",
                "Calibration step size (nm):",
                value=5,
                decimals=2,
            )
            if ok:
                base, ext = os.path.splitext(self.movie_path)
                out_path = base + "_3d_calib.yaml"
                path, exe = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save 3D calibration", out_path, filter="*.yaml"
                )
                if path:
                    zfit.calibrate_z(
                        locs,
                        self.info,
                        step,
                        self.parameters_dialog.magnification_factor.value(),
                        path=path,
                    )
        else:
            if fit_z:
                self.fit_z()
            else:
                self.save_locs_after_fit()

    def on_fit_z_progress(self, curr, total):
        message = "Fitting z coordinate {:,} / {:,} ...".format(curr, total)
        self.status_bar.showMessage(message)

    def on_fit_z_finished(self, locs, elapsed_time):
        self.status_bar.showMessage(
            "Fitted {:,} z coordinates in {:.2f} seconds.".format(
                len(locs), elapsed_time
            )
        )
        self.locs = locs
        self.save_locs_after_fit()

    def save_locs_after_fit(self):
        base, ext = os.path.splitext(self.movie_path)
        self.save_locs(base + "_locs.hdf5")

        if not self.parameters_dialog.quality_check.isEnabled():
            self.parameters_dialog.quality_check.setEnabled(True)

        self.parameters_dialog.gpufit_checkbox.setDisabled(False)

    def fit_in_view(self):
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(10 / 7, 10 / 7)

    def zoom_out(self):
        self.view.scale(7 / 10, 7 / 10)

    def save_spots(self, path):
        box = self.parameters["Box Size"]
        spots = localize.get_spots(
            self.movie, self.identifications, box, self.camera_info
        )
        io.save_datasets(path, self.info, spots=spots)

    def save_spots_dialog(self):
        if self.movie_path == []:
            print("No spots to save.")
        else:
            base, ext = os.path.splitext(self.movie_path)
            path = base + "_spots.hdf5"
            path, exe = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save spots", path, filter="*.hdf5"
            )
            if path:
                self.save_spots(path)

    def save_locs(self, path):
        localize_info = self.last_identification_info.copy()
        localize_info["Generated by"] = "Picasso Localize"
        localize_info["Pixelsize"] = self.parameters_dialog.pixelsize.value()
        localize_info["Fit method"] = self.parameters_dialog.fit_method.currentText()
        if self.parameters_dialog.fit_z_checkbox.isChecked():
            localize_info[
                "Z Calibration Path"
            ] = self.parameters_dialog.z_calibration_path
            localize_info["Z Calibration"] = self.parameters_dialog.z_calibration
        info = self.info + [localize_info | self.camera_info]
        if self.parameters_dialog.calib_astig_checkbox.isChecked():
            print("Correcting astigmatism...")

        io.save_locs(path, self.locs, info)

    def save_locs_dialog(self):
        if self.movie_path == []:
            print("No localizations to save.")
        else:
            base, ext = os.path.splitext(self.movie_path)
            locs_path = base + "_locs.hdf5"
            path, exe = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save localizations", locs_path, filter="*.hdf5"
            )
            if path:
                self.save_locs(path)

    def localize(self, calibrate_z=False):
        self.parameters_dialog.gpufit_checkbox.setDisabled(True)
        if self.identifications is not None and calibrate_z is True:
            self.fit(calibrate_z=calibrate_z)
        else:
            self.identify(fit_afterwards=True, calibrate_z=calibrate_z)
            

class IdentificationWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, dict)
    finished = QtCore.pyqtSignal(dict, object, np.recarray, bool, bool)

    def __init__(self, window, fit_afterwards, calibrate_z):
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.roi = window.view.roi
        self.parameters = window.parameters
        self.fit_afterwards = fit_afterwards
        self.calibrate_z = calibrate_z

    def run(self):
        N = len(self.movie)
        curr, futures = localize.identify_async(
            self.movie,
            self.parameters["Min. Net Gradient"],
            self.parameters["Box Size"],
            self.roi,
        )
        while curr[0] < N:
            self.progressMade.emit(curr[0], self.parameters)
            time.sleep(0.2)
        self.progressMade.emit(curr[0], self.parameters)
        identifications = localize.identifications_from_futures(futures)
        self.finished.emit(
            self.parameters,
            self.roi,
            identifications,
            self.fit_afterwards,
            self.calibrate_z,
        )


class FitWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(np.recarray, float, bool, bool)

    def __init__(
        self,
        movie,
        camera_info,
        identifications,
        box,
        method,
        eps,
        max_it,
        fit_z,
        calibrate_z,
        use_gpufit,
    ):
        super().__init__()
        self.movie = movie
        self.camera_info = camera_info
        self.identifications = identifications
        self.box = box
        self.method = method
        self.eps = eps
        self.max_it = max_it
        self.fit_z = fit_z
        self.calibrate_z = calibrate_z
        self.use_gpufit = use_gpufit

    def run(self):
        N = len(self.identifications)
        t0 = time.time()
        spots = localize.get_spots(
            self.movie, self.identifications, self.box, self.camera_info
        )
        if self.method == "lq":
            if self.use_gpufit:
                self.progressMade.emit(1, 1)
                theta = gausslq.fit_spots_gpufit(spots)
                em = self.camera_info["gain"] > 1
                locs = gausslq.locs_from_fits_gpufit(
                    self.identifications, theta, self.box, em
                )
            else:
                fs = gausslq.fit_spots_parallel(spots, asynch=True)
                n_tasks = len(fs)
                while lib.n_futures_done(fs) < n_tasks:
                    self.progressMade.emit(
                        round(N * lib.n_futures_done(fs) / n_tasks), N
                    )
                    time.sleep(0.2)
                theta = gausslq.fits_from_futures(fs)
                em = self.camera_info["gain"] > 1
                locs = gausslq.locs_from_fits(self.identifications, theta, self.box, em)
        elif self.method == "mle":
            curr, thetas, CRLBs, llhoods, iterations = gaussmle.gaussmle_async(
                spots, self.eps, self.max_it, method="sigmaxy"
            )
            while curr[0] < N:
                self.progressMade.emit(curr[0], N)
                time.sleep(0.2)
            locs = gaussmle.locs_from_fits(
                self.identifications,
                thetas,
                CRLBs,
                llhoods,
                iterations,
                self.box,
            )
        elif self.method == "avg":
            print("Average intensity")
            # just get out the average intensity
            fs = avgroi.fit_spots_parallel(spots, asynch=True)
            n_tasks = len(fs)
            while lib.n_futures_done(fs) < n_tasks:
                self.progressMade.emit(round(N * lib.n_futures_done(fs) / n_tasks), N)
                time.sleep(0.2)
            theta = avgroi.fits_from_futures(fs)
            em = self.camera_info["gain"] > 1
            locs = avgroi.locs_from_fits(self.identifications, theta, self.box, em)
        else:
            print("This should never happen...")
        self.progressMade.emit(N + 1, N)
        dt = time.time() - t0
        self.finished.emit(locs, dt, self.fit_z, self.calibrate_z)


class FitZWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(np.recarray, float)

    def __init__(
        self, 
        locs, 
        info, 
        calibration, 
        magnification_factor,
    ):
        super().__init__()
        self.locs = locs
        self.info = info
        self.calibration = calibration
        self.magnification_factor = magnification_factor

    def run(self):
        t0 = time.time()
        N = len(self.locs)
        fs = zfit.fit_z_parallel(
            self.locs,
            self.info,
            self.calibration,
            self.magnification_factor,
            filter=0,
            asynch=True,
        )
        n_tasks = len(fs)
        while lib.n_futures_done(fs) < n_tasks:
            self.progressMade.emit(round(N * lib.n_futures_done(fs) / n_tasks), N)
            time.sleep(0.2)
        locs = zfit.locs_from_futures(fs, filter=0)
        dt = time.time() - t0
        self.finished.emit(locs, dt)


class QualityWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(str, int, str)
    finished = QtCore.pyqtSignal(str)

    def __init__(self, locs, info, path, pixelsize):
        super().__init__()
        self.locs = locs
        self.info = info
        self.path = path
        self.pixelsize = pixelsize

    def run(self):
        # Sanity of locs.
        sane_locs = lib.ensure_sanity(self.locs, self.info)

        # Locs
        self.progressMade.emit("Checking Quality (1/4) Locs ..", 0, "")
        locs_per_frame = len(sane_locs) / self.info[0]["Frames"]
        self.progressMade.emit("", 0, f"{locs_per_frame:.1f}")

        # NeNA
        self.progressMade.emit("Checking Quality (2/4) NeNA ..", 0, "")

        def nena_callback(x):
            self.progressMade.emit(f"Checking Quality (2/4) NeNA: {x} %", 0, "")

        nena_px = localize.check_nena(sane_locs, self.info, nena_callback)
        nena_nm = float(self.pixelsize.value() * nena_px)
        self.progressMade.emit("", 1, f"{nena_px:.2f} px / {nena_nm:.2f} nm")

        # Drift
        self.progressMade.emit("Checking Quality (3/4) Drift ..", 0, "")

        def drift_callback(x):
            self.progressMade.emit(f"Checking Quality (3/4) Drift {x} %", 0, "")

        drift_x, drift_y = localize.check_drift(
            sane_locs, self.info, callback=drift_callback
        )
        self.progressMade.emit("", 2, f"X: {drift_x:.3f} px / Y: {drift_y:.3f} px")

        # Kinetics
        self.progressMade.emit("Checking Quality (4/4) Kinetics ..", 0, "")
        len_mean = localize.check_kinetics(sane_locs, self.info)
        self.progressMade.emit("", 3, f"{len_mean:.3f}")

        print(f"Quality {nena_px} {drift_x} {drift_y} {len_mean}")

        localize.add_file_to_db(
            self.path, None, drift=(drift_x, drift_y), len_mean=len_mean, nena=nena_px
        )

        self.finished.emit("Quality parameters complete.")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()

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
        if p.name == "localize":
            p.execute()

    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(window, "An error occured", message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook  # #excepthook

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
