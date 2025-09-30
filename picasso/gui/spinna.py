"""
    picasso.gui.spinna
    ~~~~~~~~~~~~~~~~~~

    Graphical user interface for simulating single proteins in
    DNA-PAINT using SPINNA. DOI: 10.1038/s41467-025-59500-z

    :authors: Rafal Kowalewski, Luciano A Masullo, 2022-2025
    :copyright: Copyright (c) 2022-2025 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import re
import importlib
import pkgutil
import io as python_io
from functools import partial
from multiprocessing import cpu_count
from datetime import datetime
from copy import deepcopy
from decimal import Decimal
from math import isclose
from typing import Callable, Literal

import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtSvg import QSvgRenderer

from .. import io, lib, spinna, __version__

matplotlib.use('agg')

MASK_PREVIEW_SIZE = 600
MASK_PREVIEW_ZOOM = 9 / 7
MASK_PREVIEW_PADDING = 0.3
MASK_INFO_OFFSET = 18

STRUCTURE_PREVIEW_SIZE = 512
STRUCTURE_PREVIEW_SCALING = 0.8 * 1.44
STRUCTURE_PREVIEW_MOL_SIZE = 12
STRUCTURE_PREVIEW_COLORS = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44],
    [214, 39, 40], [148, 103, 189], [140, 86, 75],
    [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207],
]

NND_PLOT_SIZE = 470
FIT_RESULT_LIM = 100


def ignore_escape_key(event: QtCore.QEvent) -> None:
    """Ignore the escape key. This function is applied to each of the
    tabs in the main window since we do not want to hide the currently
    viewed tab."""
    if event.key() == QtCore.Qt.Key_Escape:
        event.ignore()


def split_name(name: str) -> tuple[str, int]:
    """Extract str with name (without integer at the end) and the
    integer from name. name is assumed to consist of a few lower
    characters followed by an integer.

    Parameters
    ---------
    name : str
        Name to be processed.

    Returns
    -------
    result : tuple
        Two elements; first is the base of the name (without the
        number), the other is the number.
    """
    split = re.match(r"([a-z]+)(\d+)", name)
    base = split.group(1)
    num = int(split.group(2))
    return base, num


def check_structures_loaded(f: Callable) -> Callable:
    """Decorator that checks if structures are loaded. Displays a
    warning if not."""
    def wrapper(*args, **kwargs):
        if not args[0].structures:
            message = "Please load structures first."
            QtWidgets.QMessageBox.warning(args[0], "", message)
            return
        else:
            return f(*args, **kwargs)
    return wrapper


def check_exp_data_loaded(f: Callable) -> Callable:
    """Decorator that checks if experimental data is loaded. Displays
    a warning if not."""
    def wrapper(*args, **kwargs):
        message = "Please load experimental data first."
        if not args[0].targets:
            QtWidgets.QMessageBox.warning(args[0], "", message)
            return
        for target in args[0].targets:
            if target not in args[0].exp_data.keys():
                QtWidgets.QMessageBox.warning(args[0], "", message)
                return
        return f(*args, **kwargs)
    return wrapper


def check_search_space_loaded(f: Callable) -> Callable:
    """Decorator that checks if the stoichiometry search space is
    loaded. Displays a warning if not."""
    def wrapper(*args, **kwargs):
        if not args[0].N_structures_fit or not args[0].granularity:
            message = "Please generate/load search space."
            QtWidgets.QMessageBox.information(args[0], "Warning", message)
            return
        else:
            return f(*args, **kwargs)
    return wrapper


class ignoreArrowsSpinBox(QtWidgets.QSpinBox):
    """Convenience class that ignores the right and left arrow keys.
    Used in MaskGeneratorTab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            self.clearFocus()
        else:
            super().keyPressEvent(event)


class ignoreArrowsDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Convenience class that ignores the right and left arrow keys.
    Used in MaskGeneratorTab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            self.clearFocus()
        else:
            super().keyPressEvent(event)


class MaskGeneratorLegend(QtWidgets.QLabel):
    """Legend for the mask generator preview, found in the navigation
    box."""

    def __init__(self, mask_tab):
        super().__init__(" ")
        self.mask_tab = mask_tab
        # self.setFixedWidth(314)
        # self.setFixedHeight(35)
        self.fig = None

    def on_preview_updated(self, image: np.ndarray) -> None:
        """Update the legend according to the current field of view.

        Parameters
        ----------
        image : np.ndarray
            Currently shown image of the mask. Values give the
            probability mass function for finding a molecule in the
            pixel/voxel.
        """

        self.plot_legend(image.max())
        buffer = python_io.BytesIO()
        self.fig.savefig(buffer, format='svg')
        buffer.seek(0)
        svg_data = buffer.getvalue().decode()

        # load the data into pyqt5's svg renderer
        svg_renderer = QSvgRenderer()
        svg_renderer.load(svg_data.encode())

        # create the qimage and set pixmap
        qimage = QtGui.QImage(
            QtCore.QSize(290, 70),
            QtGui.QImage.Format_ARGB32_Premultiplied,
        )
        qimage.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(qimage)
        svg_renderer.render(painter)
        painter.end()
        self.setPixmap(QtGui.QPixmap.fromImage(qimage))

    def plot_legend(self, max_value: float) -> None:
        """Plot the legend with the given max value."""

        if self.fig:
            plt.close(self.fig)
        gradient = np.linspace(0, 1, 16)
        gradient = np.vstack((gradient, gradient))
        self.fig, ax = plt.subplots(
            1, figsize=(3, 0.7), constrained_layout=True
        )
        self.fig.patch.set_alpha(0)  # set transparent background
        ax.imshow(gradient, cmap="magma")
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, 15, 5))
        ax.set_xticklabels(["0.00E+0"] + [
            f"{Decimal(str(_)):.2E}"
            for _ in np.linspace(0, max_value, 5)[1:]
        ])


class MaskPreview(QtWidgets.QLabel):
    """Rendering window for masking.

    ...

    Attributes
    ----------
    image : np.ndarray
        Currently shown image of the mask.
    mask_tab : MaskGeneratorTab
        Parent tab, used for generating masks.
    qimage : QtGui.QImage
        Currently shown image of the mask.
    viewport : tuple
        FOV of the mask.
    """

    def __init__(self, mask_tab: MaskGeneratorTab) -> None:
        super().__init__(mask_tab)
        self.mask_tab = mask_tab
        self.qimage = None  # currently shown image of the mask (QImage)
        self.image = None  # currently shown image of the mask (np.ndarray)
        self.viewport = None
        self.setFixedWidth(MASK_PREVIEW_SIZE)
        self.setFixedHeight(MASK_PREVIEW_SIZE)

    def render_image(self) -> None:
        """Render image in the preview."""
        self.mask_tab.legend.on_preview_updated(self.image)
        img = self.image
        img = self.to_2D(img)
        img = self.to_8bit(img)
        self.qimage = self.get_qimage(img)
        self.qimage = self.draw_scalebar(self.qimage)
        self.setPixmap(QtGui.QPixmap.fromImage(self.qimage))

    def on_mask_generated(self, full_fov: bool = True) -> None:
        """Render the whole FOV with the new mask."""
        if self.mask_tab.mask is None:
            return

        if full_fov:
            self.image = self.mask_tab.mask.copy()
            self.viewport = (
                (0, 0), (self.image.shape[1], self.image.shape[0])
            )
        else:
            (y_min, x_min), (y_max, x_max) = self.viewport
            self.image = self.mask_tab.mask.copy()[y_min:y_max, x_min:x_max]
        self.render_image()

    def to_2D(self, image: np.ndarray) -> np.ndarray:
        """Convert mask to 2D that can be displayed (viewed from +z)."""
        if image.ndim == 3:
            image = np.sum(image, axis=2)
        elif image.ndim != 2:
            raise IndexError("Image is neither 3D or 2D.")
        image /= image.max()
        return image

    def to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Convert image (np.ndarray) to 8bit."""
        return np.round(255 * image).astype("uint8")

    def get_qimage(self, image: np.ndarray) -> QtGui.QImage:
        """Apply magma cmap to the image and converts it to QImage."""
        Y, X = image.shape
        bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        cmap = np.uint8(
            np.round(255 * plt.get_cmap("magma")(np.arange(256)))
        )
        bgra[..., 0] = cmap[:, 2][image]
        bgra[..., 1] = cmap[:, 1][image]
        bgra[..., 2] = cmap[:, 0][image]
        bgra[..., 3].fill(255)

        qimage = QtGui.QImage(
            bgra.data, X, Y, QtGui.QImage.Format_RGB32
        ).scaled(
            self.width(),
            self.height(),
            QtCore.Qt.KeepAspectRatioByExpanding,
        )
        return qimage

    def draw_scalebar(self, image: np.ndarray) -> np.ndarray | None:
        """Draw scalebar onto image."""
        if image is None or not self.mask_tab.scalebar_check.isChecked():
            return image

        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        length_nm = self.mask_tab.scalebar_length.value()
        pixelsize = self.mask_tab.mask_generator.binsize
        length_display = int(
            self.width() * length_nm / (pixelsize * self.image.shape[0])
        )
        x = self.width() - length_display - 35
        y = self.height() - 35
        painter.drawRect(x, y, length_display, 10)
        painter.end()
        return image

    def save_current_view(self) -> None:
        """Save self.image (QImage, the current view) as png or tif."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save current view", filter="*.png;;*.tif"
        )
        if path:
            self.qimage.save(path)

    def viewport_size(self) -> tuple[int, int] | None:
        """Get the size of the viewport."""
        if self.viewport is not None:
            width = self.viewport[1][1] - self.viewport[0][1]
            height = self.viewport[1][0] - self.viewport[0][0]
            return height, width

    def viewport_center(self) -> tuple[float, float] | None:
        """Get the center of the viewport."""
        if self.viewport is not None:
            (y_min, x_min), (y_max, x_max) = self.viewport
            yc = (y_max + y_min) / 2
            xc = (x_max + x_min) / 2
            return (yc, xc)

    def zoom_in(self) -> None:
        """Zoom in the viewport."""
        self.zoom(1 / MASK_PREVIEW_ZOOM)

    def zoom_out(self) -> None:
        """Zoom out the viewport."""
        self.zoom(MASK_PREVIEW_ZOOM)

    def zoom(self, factor) -> None:
        """Zoom the viewport by the given factor."""
        vh, vw = self.viewport_size()  # viewport height and width
        yc, xc = self.viewport_center()
        # new viewport height and width
        new_vh, new_vw = [_ * factor for _ in (vh, vw)]
        # new viewport
        y_min = int(yc - new_vh / 2)
        x_min = int(xc - new_vw / 2)
        y_max = int(yc + new_vh / 2)
        x_max = int(xc + new_vw / 2)
        x_min, x_max, y_min, y_max = self.verify_boundaries(
            x_min, x_max, y_min, y_max
        )
        self.viewport = ((y_min, x_min), (y_max, x_max))
        self.image = self.mask_tab.mask.copy()[y_min:y_max, x_min:x_max]
        self.render_image()

    def up(self) -> None:
        """Move viewport one unit up."""
        self.move_viewport(-MASK_PREVIEW_PADDING, 0)

    def down(self) -> None:
        """Move viewport one unit down."""
        self.move_viewport(MASK_PREVIEW_PADDING, 0)

    def left(self) -> None:
        """Move viewport one unit left."""
        self.move_viewport(0, -MASK_PREVIEW_PADDING)

    def right(self) -> None:
        """Move viewport one unit right."""
        self.move_viewport(0, MASK_PREVIEW_PADDING)

    def move_viewport(self, dy: float, dx: float) -> None:
        """Move viewport by proportions given by dy and dx."""
        vh, vw = self.viewport_size()
        x_move = self.get_viewport_shift(int(dx * vw))
        y_move = self.get_viewport_shift(int(dy * vh))
        x_min = self.viewport[0][1] + x_move
        x_max = self.viewport[1][1] + x_move
        y_min = self.viewport[0][0] + y_move
        y_max = self.viewport[1][0] + y_move
        x_min, x_max, y_min, y_max = self.verify_boundaries(
            x_min, x_max, y_min, y_max
        )
        self.viewport = ((y_min, x_min), (y_max, x_max))
        self.image = self.mask_tab.mask.copy()[y_min:y_max, x_min:x_max]
        self.render_image()

    def get_viewport_shift(self, value: int) -> int:
        """Return viewport shift that is at least 3 pixels."""
        if value:  # if non-zero
            if value < 0:
                value = min(value, -3)
            else:
                value = max(value, 3)
        return value

    def verify_boundaries(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
    ) -> tuple[int, int, int, int]:
        """Check if the boundaries lie within the mask boundaries.
        Return the verified boundaries."""
        vh, vw = self.viewport_size()
        vh = y_max - y_min
        vw = x_max - x_min
        if x_min < 0:
            x_min = 0
            x_max = min(vw, self.mask_tab.mask.shape[1])
        if y_min < 0:
            y_min = 0
            y_max = min(vh, self.mask_tab.mask.shape[0])
        bounds_x_y = self.mask_tab.mask.shape
        if x_max > bounds_x_y[1]:
            x_max = bounds_x_y[1]
            x_min = max(0, x_max - vw)
        if y_max > bounds_x_y[0]:
            y_max = bounds_x_y[0]
            y_min = max(0, y_max - vh)

        return x_min, x_max, y_min, y_max


class MaskGeneratorTab(QtWidgets.QDialog):
    """Tab for generating masks for heterogenous density simulations.

    ...

    Attributes
    ----------
    generate_mask_button : QtWidgets.QPushButton
        Button that generates the mask.
    legend : QtWidgets.QLabel
        Displays the legend for the mask.
    load_locs_button : QtWidgets.QPushButton
        Button that loads the molecules.
    locs : np.ndarray
        Localization list to be used for generating the mask.
    locs_path : str
        Path to the molecules.
    mask : np.ndarray
        Generated mask; pixel/voxel values give probability mass
        function for find a molecule in the pixel/voxel.
    mask_binsize : QtWidgets.QSpinBox
        Size of the mask pixel/voxel (nm).
    mask_blur : QtWidgets.QSpinBox
        Size of the Gaussian blur (nm).
    mask_generator : spinna.MaskGenerator
        Mask generator.
    mask_info_display1/2 : QtWidgets.QLabel
        Display the mask info.
    mask_ndim : QtWidgets.QComboBox
        Dimensionality of the mask (2D/3D).
    mask_type : QtWidgets.QComboBox
        Type of the mask (binary or density map).
    navigation_buttons : list
        List of navigation buttons.
    preview : MaskPreview
        Displays the mask.
    save_mask_button : QtWidgets.QPushButton
        Button that saves the mask.
    scalebar_check : QtWidgets.QCheckBox
        Checkbox that enables/disables the scalebar.
    scalebar_length : QtWidgets.QSpinBox
        Length of the scalebar.
    thresholding_check : QtWidgets.QCheckBox
        Checkbox that enables/disables the thresholding for density map.
    thresholding_value : QtWidgets.QDoubleSpinBox
        Value of the threshold for the density map mask type. Gives the
        probability cutoff.
    thresholding_stack : QtWidgets.QStackedWidget
        Stack of widgets that are shown/hidden depending on the mask
        type.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.setAutoFillBackground(True)
        layout = QtWidgets.QGridLayout(self)
        self.setLayout(layout)
        self.preview = MaskPreview(self)

        self.locs_path = ""
        self.locs = None
        self.mask = None
        self.mask_generator = None

        # PREVIEW
        preview_box = QtWidgets.QGroupBox("Preview")
        layout.addWidget(preview_box, 0, 0, 3, 1)
        preview_grid = QtWidgets.QGridLayout(preview_box)
        preview_grid.addWidget(self.preview, 0, 0, 1, 3)

        # scalebar
        self.scalebar_check = QtWidgets.QCheckBox("Show scalebar")
        self.scalebar_check.setChecked(False)
        self.scalebar_check.setEnabled(False)
        self.scalebar_check.stateChanged.connect(self.preview.render_image)
        preview_grid.addWidget(self.scalebar_check, 1, 0)

        label = QtWidgets.QLabel("Scalebar length (nm):")
        label.setAlignment(QtCore.Qt.AlignRight)
        preview_grid.addWidget(label, 1, 1)
        self.scalebar_length = ignoreArrowsSpinBox()
        self.scalebar_length.setEnabled(False)
        self.scalebar_length.setRange(1, 20_000)
        self.scalebar_length.setValue(1_000)
        self.scalebar_length.valueChanged.connect(self.preview.render_image)
        preview_grid.addWidget(self.scalebar_length, 1, 2)

        # MASK PARAMETERS AND LOADING
        mask_box = QtWidgets.QGroupBox("Parameters")
        mask_box.setFixedHeight(340)
        layout.addWidget(mask_box, 0, 1)
        mask_layout = QtWidgets.QGridLayout(mask_box)

        # load molecules
        self.load_locs_button = QtWidgets.QPushButton("Load molecules")
        self.load_locs_button.released.connect(self.load_locs)
        mask_layout.addWidget(self.load_locs_button, 0, 0, 1, 2)

        # mask pixel / voxel size
        mask_layout.addWidget(
            QtWidgets.QLabel("Mask pixel/voxel (nm):"), 1, 0
        )
        self.mask_binsize = ignoreArrowsSpinBox()
        self.mask_binsize.setRange(1, 10_000)
        self.mask_binsize.setSingleStep(1)
        self.mask_binsize.setValue(50)
        mask_layout.addWidget(self.mask_binsize, 1, 1)

        # mask blur
        mask_layout.addWidget(QtWidgets.QLabel("Gaussian blur (nm):"), 2, 0)
        self.mask_blur = ignoreArrowsSpinBox()
        self.mask_blur.setRange(0, 10_000)
        self.mask_blur.setSingleStep(1)
        self.mask_blur.setValue(0)
        mask_layout.addWidget(self.mask_blur, 2, 1)

        # ndimensions:
        mask_layout.addWidget(QtWidgets.QLabel("Mask dimensionality:"), 3, 0)
        self.mask_ndim = QtWidgets.QComboBox()
        self.mask_ndim.addItems(["2D", "3D"])
        mask_layout.addWidget(self.mask_ndim, 3, 1)

        # mask type (binary, loc density)
        mask_layout.addWidget(QtWidgets.QLabel("Mask type:"), 4, 0)
        self.mask_type = QtWidgets.QComboBox()
        self.mask_type.addItems(["Density map", "Binary"])
        self.mask_type.currentIndexChanged.connect(self.on_mask_type_changed)
        mask_layout.addWidget(self.mask_type, 4, 1)

        # generate mask
        self.generate_mask_button = QtWidgets.QPushButton("Generate mask")
        self.generate_mask_button.released.connect(self.generate_mask)
        self.generate_mask_button.setEnabled(False)
        mask_layout.addWidget(self.generate_mask_button, 5, 0, 1, 2)

        # thresholding density map
        self.thresholding_stack = QtWidgets.QStackedWidget()
        mask_layout.addWidget(self.thresholding_stack, 6, 0, 1, 2)
        threshold_widget = QtWidgets.QWidget()
        self.thresholding_stack.addWidget(threshold_widget)
        thresholding_layout = QtWidgets.QHBoxLayout()
        threshold_widget.setLayout(thresholding_layout)

        self.thresholding_check = QtWidgets.QCheckBox("Apply threshold")
        self.thresholding_check.setChecked(False)
        self.thresholding_check.stateChanged.connect(self.apply_threshold)
        thresholding_layout.addWidget(self.thresholding_check)
        self.thresholding_value = ignoreArrowsDoubleSpinBox()
        self.thresholding_value.setRange(0, 1)
        self.thresholding_value.setSingleStep(1e-8)
        self.thresholding_value.setDecimals(8)
        thresholding_layout.addWidget(self.thresholding_value)
        self.thresholding_stack.addWidget(QtWidgets.QLabel("          "))

        # save mask
        self.save_mask_button = QtWidgets.QPushButton("Save mask")
        self.save_mask_button.released.connect(self.save_mask)
        self.save_mask_button.setEnabled(False)
        mask_layout.addWidget(self.save_mask_button, 7, 0, 1, 2)

        # PREVIEW NAVIGATION
        navigation_box = QtWidgets.QGroupBox("Navigation")
        layout.addWidget(navigation_box, 1, 1)
        navigation_layout = QtWidgets.QGridLayout(navigation_box)

        # Full FOV (reset)
        full_fov_button = QtWidgets.QPushButton("Full FOV")
        full_fov_button.released.connect(self.preview.on_mask_generated)
        navigation_layout.addWidget(full_fov_button, 0, 0, 1, 4)

        # Zoom in/out
        zoom_in_button = QtWidgets.QPushButton("Zoom in")
        zoom_in_button.released.connect(self.preview.zoom_in)
        navigation_layout.addWidget(zoom_in_button, 1, 0, 1, 2)
        zoom_out_button = QtWidgets.QPushButton("Zoom out")
        zoom_out_button.released.connect(self.preview.zoom_out)
        navigation_layout.addWidget(zoom_out_button, 1, 2, 1, 2)

        # Padding (move viewport)
        up_button = QtWidgets.QPushButton("Up")
        up_button.setShortcut("Up")
        up_button.released.connect(self.preview.up)
        navigation_layout.addWidget(up_button, 2, 0)
        down_button = QtWidgets.QPushButton("Down")
        down_button.setShortcut("Down")
        down_button.released.connect(self.preview.down)
        navigation_layout.addWidget(down_button, 2, 1)
        left_button = QtWidgets.QPushButton("Left")
        left_button.setShortcut("Left")
        left_button.released.connect(self.preview.left)
        navigation_layout.addWidget(left_button, 2, 2)
        right_button = QtWidgets.QPushButton("Right")
        right_button.setShortcut("Right")
        right_button.released.connect(self.preview.right)
        navigation_layout.addWidget(right_button, 2, 3)

        # Save current view
        save_view_button = QtWidgets.QPushButton("Save current view")
        save_view_button.released.connect(self.preview.save_current_view)
        navigation_layout.addWidget(save_view_button, 3, 0, 1, 4)

        self.legend = MaskGeneratorLegend(self)
        navigation_layout.addWidget(self.legend, 4, 0, 1, 4)

        self.navigation_buttons = [
            full_fov_button,
            zoom_in_button,
            zoom_out_button,
            up_button,
            down_button,
            left_button,
            right_button,
            save_view_button,
        ]
        for button in self.navigation_buttons:
            button.setEnabled(False)

        # MASK INFORMATION
        mask_info_box = QtWidgets.QGroupBox("Mask information")
        mask_info_box.setFixedHeight(100)
        layout.addWidget(mask_info_box, 2, 1)
        mask_info_layout = QtWidgets.QHBoxLayout(mask_info_box)
        self.mask_info_display1 = QtWidgets.QLabel(
            "Area (\u03bcm\u00b2):\n"
            "Dimensions:\n"
            "Size memory:"
        )
        self.mask_info_display1.setAlignment(QtCore.Qt.AlignRight)
        # make sure that the dash symbols are aligned
        self.mask_info_display1.setFixedWidth(
            self.mask_info_display1.fontMetrics().width(
                f"{' '*MASK_INFO_OFFSET}Volume (\u03bcm\u00b3):"
            )
        )
        self.mask_info_display2 = QtWidgets.QLabel("-\n-\n-")
        self.mask_info_display2.setAlignment(QtCore.Qt.AlignLeft)
        mask_info_layout.addWidget(self.mask_info_display1)
        mask_info_layout.addWidget(self.mask_info_display2)

    def load_locs(self) -> None:
        """Load localizations / molecules for mask generation."""
        # get localizations file
        self.locs_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load molecules for mask generation", filter="*.hdf5"
        )
        if self.locs_path:
            self.mask_generator = spinna.MaskGenerator(self.locs_path)
            self.mask_ndim.clear()
            if hasattr(self.mask_generator.locs, "z"):
                self.mask_ndim.addItems(["2D", "3D"])
            else:
                self.mask_ndim.addItems(["2D"])
            self.generate_mask_button.setEnabled(True)
            self.save_mask_button.setEnabled(True)
            self.load_locs_button.setText(
                "Molecules loaded, ready for mask generation"
            )

    def generate_mask(self) -> None:
        """Generate a mask with the currently loaded settings."""
        binsize = self.mask_binsize.value()
        sigma = self.mask_blur.value()
        ndim = int(self.mask_ndim.currentText()[0])
        if self.mask_generator is not None:
            self.mask_generator.binsize = binsize
            self.mask_generator.sigma = sigma
            self.mask_generator.ndim = ndim
            mode = ["loc_den", "binary"][self.mask_type.currentIndex()]
            status = lib.StatusDialog("Generating mask...", self.preview)
            self.mask_generator.generate_mask(
                apply_thresh=False, mode=mode
            )
            status.close()
            self.mask = deepcopy(self.mask_generator.mask)
            self.preview.on_mask_generated()
            self.update_mask_info()
            # set threshold to otsu threhold
            self.thresholding_check.setChecked(False)
            self.thresholding_value.setValue(self.mask_generator.thresh)
            self.scalebar_check.setEnabled(True)
            self.scalebar_length.setEnabled(True)
            for button in self.navigation_buttons:
                button.setEnabled(True)

    def apply_threshold(self, state: int) -> None:
        """Apply the threshold to the density map."""
        if self.mask is None:
            return

        if state == 0:  # unchecked
            self.mask = deepcopy(self.mask_generator.mask)
        elif state == 2:  # checked
            self.mask[self.mask < self.thresholding_value.value()] = 0
            self.mask = self.mask / self.mask.sum()
        self.preview.on_mask_generated(full_fov=False)
        self.update_mask_info()

    def save_mask(self) -> None:
        """Save the generated mask."""
        if self.mask is not None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save mask",
                self.locs_path.replace(".hdf5", "_mask.npy"),
                filter="*.npy",
            )
            if path:
                # if threshold was applied, save the thresholded mask
                if self.thresholding_check.isChecked():
                    self.mask_generator.mask = deepcopy(self.mask)
                    self.mask_generator.thresh = (
                        self.thresholding_value.value()
                    )
                self.mask_generator.save_mask(path)

    def update_mask_info(self) -> None:
        """Update the mask info (area, dimensions, size)."""
        if self.mask is None:
            return

        area_str, area = self.get_mask_area()
        dimensions = self.get_mask_dimensions()
        size = self.get_mask_size()

        self.mask_info_display1.setText(
            f"{area_str}\n Dimensions:\nSize memory:"
        )
        self.mask_info_display2.setText(f"{area}\n{dimensions}\n{size}")

    def get_mask_area(self) -> tuple[str, float]:
        """Find the string with mask area/volume."""
        if self.mask is None:
            area_str = "Area (\u03bcm\u00b2):"
            area = "-"
        else:
            if self.thresholding_check.isChecked():
                count = (self.mask > 0).sum()
            else:
                count = (self.mask > self.mask_generator.thresh).sum()
            if self.mask.ndim == 2:
                area_str = "Area (\u03bcm\u00b2):"
                area = 1e-6 * self.mask_generator.binsize ** 2 * count
            else:
                area_str = "Volume (\u03bcm\u00b3):"
                area = 1e-9 * self.mask_generator.binsize ** 3 * count
        return area_str, np.round(area, 2)

    def get_mask_dimensions(self) -> str:
        """Return the dimensions (pixels/voxels) of the mask."""
        if self.mask is None:
            dimensions = "-"
        else:
            dims = self.mask.shape
            if self.mask.ndim == 2:
                dimensions = f"{dims[0]}x{dims[1]}"
            else:
                dimensions = f"{dims[0]}x{dims[1]}x{dims[2]}"
        return dimensions

    def get_mask_size(self) -> str:
        """Find the string with mask size in MB/GB."""
        if self.mask is None:
            size = "-"
        size_mb = self.mask.nbytes / (1024**2)
        if size_mb > 1024:
            size = f"{np.round(size_mb / 1024, 2)} GB"
        else:
            size = f"{np.round(size_mb, 2)} MB"
        return size

    def on_mask_type_changed(self) -> None:
        """Show/hide the thresholding options for the density map mask
        type."""
        if self.mask_type.currentIndex() == 0:  # density map
            self.thresholding_stack.setCurrentIndex(0)
        elif self.mask_type.currentIndex() == 1:
            self.thresholding_stack.setCurrentIndex(1)


class StructurePreview(QtWidgets.QLabel):
    """Display of the designed structures.

    ...

    Attributes
    ----------
    angx, angy, angz : float
        Rotation angles in radians.
    coords : list of np.2darrays
        Each element contains the coordinates (in pixels) of each
        molecular target species loaded.
    factor : float
        Scaling factor used at drawing the scalebar and spreading
        the molecular targets.
    structure : spinna.Structure
        Currently loaded structure.
    structure_tab : StructureTab
        Parent tab, used for loading structures.
    ORIGIN : np.2darray
        Origin of the coordinate system displaying the molecular
        targets.
    qimage : QtGui.QImage
        Currently shown image of the structure.
    rotating : bool
        Is the displayed structure is being rotated.
    rotation : list
        List of rotation angles. Resets whenever the structure is
        no longer being rotated.
    """

    def __init__(self, structure_tab: StructuresTab) -> None:
        super().__init__(structure_tab)
        self.structure_tab = structure_tab
        self.setFixedWidth(STRUCTURE_PREVIEW_SIZE)
        self.setFixedHeight(STRUCTURE_PREVIEW_SIZE)

        self.structure = None
        self.coords = None
        self.angx = 0  # rotation angles in radians
        self.angy = 0
        self.angz = 0
        self.ORIGIN = np.int32((
            int(STRUCTURE_PREVIEW_SIZE/2),
            int(STRUCTURE_PREVIEW_SIZE/2),
            0,
        ))
        self.factor = 1  # scaling factor used at drawing the scalebar and
        # spreading the molecular targets
        self.rotating = False  # is the displayed structure is being rotated
        self._rotation = []
        self.render()

    def update_scene(self) -> None:
        """Render currently loaded structure."""
        if self.structure is not None and self.structure.targets:
            coords = self.extract_coordinates()
            coords = self.rotate(coords)
            coords = self.scale(coords)
            coords = self.shift(coords)
            self.coords = coords
        else:
            self.coords = None

        self.render()

    def extract_coordinates(self) -> np.ndarray:
        """Extract x, y and z coordinates of the loaded structure (no
        rotation). Also finds the scaling factor (from nm to pixels).

        Returns
        -------
        coords : list of np.2darrays
            Each element contains the x,y,z coordinates of each
            molecular target species in self.structure.
        """
        coords = []
        for target in self.structure.targets:
            x = self.structure.x[target]
            y = self.structure.y[target]
            z = self.structure.z[target]
            coords.append(np.stack((x, y, z)).T)

        # get the approximate max distances between two points
        # (this is not accurate)
        coords_ = np.vstack(coords)  # merge the points from all species
        min_ = coords_.min()
        max_ = coords_.max()
        max_dif = max_ - min_
        factor = STRUCTURE_PREVIEW_SIZE / 2 / STRUCTURE_PREVIEW_SCALING
        if max_dif:  # if non zero
            factor /= max_dif
        self.factor = factor
        return coords

    def rotate(self, coords: list[np.ndarray]) -> list[np.ndarray]:
        """Rotate coordinates of each molecular target species.

        Parameters
        ----------
        coords : list of np.2darrays
            Each element contains the coordinates each molecular target
            species loaded.

        Returns
        -------
        coords_rot : lists of np.2darrays
            Rotated coordinates.
        """
        rot = Rotation.from_euler('zyx', (self.angz, self.angy, self.angx))
        coords_rot = [rot.apply(_) for _ in coords]
        return coords_rot

    def scale(self, coords: list[np.ndarray]) -> list[np.ndarray]:
        """Scale molecular targets' coordinates from nm to display
        pixels.

        Parameters
        ----------
        coords : list of np.2darrays
            Each element contains the x,y,z coordinates (in nm) of each
            molecular target species in self.structure.

        Returns
        -------
        coords_scaled : list of np.2darrays
            Scaled coordinates (in pixels).
        """
        coords_scaled = []
        for coord in coords:
            coord[:, 1] = coord[:, 1] * -1
            coords_scaled.append(coord * self.factor)
        return coords_scaled

    def shift(self, coords: list[np.ndarray]) -> list[np.ndarray]:
        """Shift x and y coordinates towards self.ORIGIN.

        Parameters
        ----------
        coords : lists of np.2darrays
            Each element contains the coordinates (in pixels) of each
            molecular target species loaded.

        Returns
        -------
        coords_shifted : lists of lists
            Shifted coordinates converted to integers.
        """
        coords_shifted = [_ + self.ORIGIN for _ in coords]
        return [_.astype(int) for _ in coords_shifted]

    def render(self) -> None:
        """Render image into self. By default, a black square is
        rendered if no structure is loaded."""
        image = self.generate_background()
        qimage = QtGui.QImage(
            image.data,
            STRUCTURE_PREVIEW_SIZE,
            STRUCTURE_PREVIEW_SIZE,
            QtGui.QImage.Format_RGB32,
        )
        qimage = self.draw_molecular_targets(qimage)
        qimage = self.draw_title(qimage)
        qimage = self.draw_legend(qimage)
        qimage = self.draw_scalebar(qimage)
        self.qimage = self.draw_rotation(qimage)
        self.setPixmap(QtGui.QPixmap.fromImage(self.qimage))

    def generate_background(self) -> np.ndarray:
        """Generate black background for display."""
        image = np.zeros(
            (STRUCTURE_PREVIEW_SIZE, STRUCTURE_PREVIEW_SIZE, 4), dtype=np.uint8
        )
        image[:, :, 3].fill(255)
        return image

    def draw_molecular_targets(self, image: np.ndarray) -> np.ndarray:
        """Draw molecular targets (from self.coords) onto image."""
        if self.coords is None:
            return image

        # image = image.copy()
        colors = self.get_colors()
        painter = QtGui.QPainter(image)
        # iterate over all molecular target species
        for coords, target in zip(self.coords, self.structure.targets):
            color = colors[target]
            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgb(*color)))
            # iterate over each molecular target
            for x, y, z in coords:
                painter.drawEllipse(
                    QtCore.QPoint(x, y),
                    STRUCTURE_PREVIEW_MOL_SIZE,
                    STRUCTURE_PREVIEW_MOL_SIZE,
                )
        painter.end()
        return image

    def draw_title(self, image: np.ndarray) -> np.ndarray:
        """Draw title of the loaded structure onto image."""
        if self.structure is None:
            title = "Please load a structure."
        else:
            title = f"Loaded: {self.structure.title}"

        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("white"))
        font = painter.font()
        font.setPixelSize(18)
        painter.setFont(font)
        painter.drawText(QtCore.QPoint(20, 38), title)
        painter.end()
        return image

    def draw_legend(self, image: np.ndarray) -> np.ndarray:
        """Draw legend onto image."""
        # make sure that a non-empty structure is loaded
        if (
            self.coords is None
            or not self.structure_tab.show_legend_check.isChecked()
        ):
            return image

        painter = QtGui.QPainter(image)
        font = painter.font()
        font.setPixelSize(18)
        painter.setFont(font)
        x = 20
        dy = 28
        y = STRUCTURE_PREVIEW_SIZE - dy * len(self.structure.targets) + 8
        colors = self.get_colors()
        for target in self.structure.targets:
            color = colors[target]
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgb(*color)))
            painter.drawText(QtCore.QPoint(x, y), target)
            y += dy
        painter.end()
        return image

    def draw_scalebar(self, image: np.ndarray) -> np.ndarray:
        """Draw scalebar onto image."""
        if (
            self.coords is None
            or not self.structure_tab.show_scalebar_check.isChecked()
        ):
            return image
        elif self.structure.get_all_targets_count() <= 1:
            return image

        # draw scalebar
        painter = QtGui.QPainter(image)
        length = int(self.structure_tab.scalebar_length.value() * self.factor)
        height = 10
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        x = STRUCTURE_PREVIEW_SIZE - length - 35
        y = STRUCTURE_PREVIEW_SIZE - height - 20
        painter.drawRect(x, y, length, height)
        painter.end()
        return image

    def draw_rotation(self, image: np.ndarray) -> np.ndarray:
        """Draw a small 3 axes icon that rotates with the molecular,
        targets displayed in the bottom left corner."""
        painter = QtGui.QPainter(image)
        length = 30
        x = self.width() - 60
        y = 50
        center = QtCore.QPoint(x, y)

        # set the ends of the x line
        xx = length
        xy = 0
        xz = 0

        # set the ends of the y line
        yx = 0
        yy = length
        yz = 0

        # set the ends of the z line
        zx = 0
        zy = 0
        zz = length

        # rotate these points
        coordinates = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
        rot = Rotation.from_euler('zyx', (self.angz, self.angy, self.angx))
        coordinates = rot.apply(coordinates).astype(int)
        (xx, xy, xz) = coordinates[0]
        (yx, yy, yz) = coordinates[1]
        (zx, zy, zz) = coordinates[2]

        # translate the x and y coordinates of the end points towards
        # bottom right edge of the window
        xx += x
        xy += y
        yx += x
        yy += y
        zx += x
        zy += y

        # set the points at the ends of the lines
        point_x = QtCore.QPoint(xx, xy)
        point_y = QtCore.QPoint(yx, yy)
        point_z = QtCore.QPoint(zx, zy)
        line_x = QtCore.QLine(center, point_x)
        line_y = QtCore.QLine(center, point_y)
        line_z = QtCore.QLine(center, point_z)
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(1, 0, 0, 1)))
        painter.drawLine(line_x)
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 1, 1)))
        painter.drawLine(line_y)
        painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 0, 1)))
        painter.drawLine(line_z)
        painter.end()
        return image

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define the action when mouse is clicked. If left button is
        clicked, the structure starts to be rotated."""
        if event.button() == QtCore.Qt.LeftButton:
            self.rotating = True
            self._rotation.append([event.x(), event.y()])

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define the action when mouse is moved. If self.rotating,
        the rotation angles are updated."""
        if self.rotating:
            self._rotation.append([event.x(), event.y()])
            rel_pos_x = self._rotation[-1][0] - self._rotation[-2][0]
            rel_pos_y = self._rotation[-1][1] - self._rotation[-2][1]
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ControlModifier:
                self.angz += 2 * np.pi * rel_pos_y / STRUCTURE_PREVIEW_SIZE
                self.angy += 2 * np.pi * rel_pos_x / STRUCTURE_PREVIEW_SIZE
            else:
                self.angy += 2 * np.pi * rel_pos_x / STRUCTURE_PREVIEW_SIZE
                self.angx += 2 * np.pi * rel_pos_y / STRUCTURE_PREVIEW_SIZE
            self.update_scene()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define the action when mouse is released. If left button is
        released, the rotation stops. If right button is released, a
        new molecular target is added."""
        if event.button() == QtCore.Qt.LeftButton:
            self.rotating = False
            self._rotation = []
            # event.accept()
        elif event.button() == QtCore.Qt.RightButton:
            self.add_molecular_target_mouse(event.x(), event.y())

    def add_molecular_target_mouse(self, x: float, y: float) -> None:
        """Add a new molecular target at a right mouse button click.

        Parameters
        ----------
        x, y : floats
            Display coordinates where the molecular target is added.
        """
        if (
            self.structure is None
            or self.structure_tab.mol_tar_box.content_layout.count() < 15
        ):  # the function will not work if less than two molecular targets
            # have been loaded
            return

        # extract x and y coordinates and shift towards ORIGIN
        x -= self.ORIGIN[0]
        y -= self.ORIGIN[1]
        # scale the values from pixels to nm
        x /= self.factor
        y /= -self.factor

        rot = Rotation.from_euler('zyx', (self.angz, self.angy, self.angx))
        x, y, z = np.around(rot.apply([x, y, 0], inverse=True), 4)

        # add the widgets
        self.structure_tab.add_molecular_target()
        # set the values of x, y, z coordinates
        widgets = [
            self.structure_tab.mol_tar_box.content_layout.itemAt(i).widget()
            for i in range(
                self.structure_tab.mol_tar_box.content_layout.count()
            )
        ]
        for widget in widgets:
            name = widget.objectName()
            if name == f"x{self.structure_tab.n_mol_tar-1}":
                widget.setValue(x)
            elif name == f"y{self.structure_tab.n_mol_tar-1}":
                widget.setValue(y)
            elif name == f"z{self.structure_tab.n_mol_tar-1}":
                widget.setValue(z)
        self.update_scene()

    def get_colors(self) -> dict:
        """Find colors for each molecular target species."""
        if self.structure is None:
            return {}

        # find unique molecular targets
        all_targets = []
        for structure in self.structure_tab.structures:
            for target in structure.targets:
                if target not in all_targets:
                    all_targets.append(target)

        n = len(all_targets)
        n_ = len(STRUCTURE_PREVIEW_COLORS)
        if n > n_:
            colors_rgb = STRUCTURE_PREVIEW_COLORS * (n // n_ + 1)
        else:
            colors_rgb = STRUCTURE_PREVIEW_COLORS

        colors = {
            target: rgb for target, rgb in zip(all_targets, colors_rgb)
        }
        return colors


class StructuresTab(QtWidgets.QDialog):
    """Tab for creating structures.

    ...

    Attributes
    ----------
    current_structure : str
        Currently chosen structure's title.
    structures : list of spinna.Structure
        List of all structures.
    structures_box : ScrollableGroupBox
        Box with a summary of all structures.
    n_mol_tar : int
        Number of molecular targets in the molecular targets box.
    preview : StructurePreview
        Preview of the currently chosen structure.
    scalebar_length : QtWidgets.QDoubleSpinBox
        Length of the scalebar (nm).
    show_legend_check : QtWidgets.QCheckBox
        Checkbox for showing/hiding legend.
    show_scalebar_check : QtWidgets.QCheckBox
        Checkbox for showing/hiding scalebar.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.setAutoFillBackground(True)
        layout = QtWidgets.QGridLayout(self)
        self.setLayout(layout)

        self.structures = []
        self.current_structure = None
        self.n_mol_tar = 0

        # PREVIEW
        preview_box = QtWidgets.QGroupBox("Preview")
        layout.addWidget(preview_box, 0, 0, 2, 1)
        preview_layout = QtWidgets.QGridLayout(preview_box)
        self.preview = StructurePreview(self)
        preview_layout.addWidget(self.preview, 0, 0, 1, 4)

        # show legend and scalebar
        self.show_legend_check = QtWidgets.QCheckBox("Show legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.stateChanged.connect(self.update_preview)
        preview_layout.addWidget(self.show_legend_check, 1, 0)

        self.show_scalebar_check = QtWidgets.QCheckBox("Show scalebar")
        self.show_scalebar_check.setChecked(True)
        self.show_scalebar_check.stateChanged.connect(self.update_preview)
        preview_layout.addWidget(self.show_scalebar_check, 1, 1)

        length_label = QtWidgets.QLabel("Length (nm):")
        length_label.setAlignment(QtCore.Qt.AlignRight)
        preview_layout.addWidget(length_label, 1, 2)
        self.scalebar_length = QtWidgets.QDoubleSpinBox()
        self.scalebar_length.setDecimals(1)
        self.scalebar_length.setValue(1.0)
        self.scalebar_length.setRange(0.1, 100)
        self.scalebar_length.setSingleStep(0.1)
        self.scalebar_length.valueChanged.connect(self.update_preview)
        preview_layout.addWidget(self.scalebar_length, 1, 3)

        reset_rot_button = QtWidgets.QPushButton("Reset rotation")
        reset_rot_button.released.connect(partial(self.update_preview, True))
        preview_layout.addWidget(reset_rot_button, 2, 0, 1, 2)

        save_view_button = QtWidgets.QPushButton("Save view")
        save_view_button.released.connect(self.save_preview)
        preview_layout.addWidget(save_view_button, 2, 2, 1, 2)

        # STRUCTURES SUMMARY
        self.structures_box = lib.ScrollableGroupBox("Structures summary")
        self.structures_box.setFixedHeight(250)
        layout.addWidget(self.structures_box, 0, 1)

        add_structure_button = QtWidgets.QPushButton("Add a new structure")
        add_structure_button.released.connect(self.add_structure)
        self.structures_box.layout().addWidget(
            add_structure_button, 1, 0, 1, 2,
        )

        save_structures_button = QtWidgets.QPushButton("Save all structures")
        save_structures_button.released.connect(self.save_structures)
        self.structures_box.layout().addWidget(save_structures_button, 2, 0)

        load_structures_button = QtWidgets.QPushButton("Load structures")
        load_structures_button.released.connect(self.load_structures)
        self.structures_box.layout().addWidget(load_structures_button, 2, 1)

        # MOLECULAR TARGETS IN THE CURRENT STRCTURE
        self.mol_tar_box = lib.ScrollableGroupBox("Molecular targets")
        self.mol_tar_box.setFixedHeight(446)
        layout.addWidget(self.mol_tar_box, 1, 1)

        self.mol_tar_box.add_widget(QtWidgets.QLabel("Mol. target"), 0, 0)
        self.mol_tar_box.add_widget(QtWidgets.QLabel("x [nm]"), 0, 1)
        self.mol_tar_box.add_widget(QtWidgets.QLabel("y [nm]"), 0, 2)
        self.mol_tar_box.add_widget(QtWidgets.QLabel("z [nm]"), 0, 3)
        self.mol_tar_box.add_widget(QtWidgets.QLabel("Delete"), 0, 4)

        add_mol_tar_button = QtWidgets.QPushButton("Add a molecular target")
        add_mol_tar_button.released.connect(self.add_molecular_target)
        self.mol_tar_box.layout().addWidget(add_mol_tar_button, 1, 0, 1, 2)

    def update_preview(self, reset_angles: bool = False) -> None:
        """Update information for rendering in self.preview."""
        self.update_current_structure()
        self.preview.structure = (
            self.find_structure_by_title(self.current_structure)
        )
        if isinstance(reset_angles, bool) and reset_angles:
            self.preview.angx = 0
            self.preview.angy = 0
            self.preview.angz = 0
        self.preview.update_scene()

    def add_structure(self) -> None:
        """Add a new structure as the attribute and the corresponding
        widgets."""
        structure_title, ok = QtWidgets.QInputDialog.getText(
            self,
            "",
            "Enter structure's title:",
            QtWidgets.QLineEdit.Normal,
            f"structure_{len(self.structures)+1}",
        )
        if ok:
            if any([structure_title == _.title for _ in self.structures]):
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Structure title already taken."
                )
                return

            # if the title is correct, save the current and add the new
            # structure
            self.update_current_structure()
            structure = spinna.Structure(title=structure_title)
            self.structures.append(structure)
            self.update_structure_box()
            self.current_structure = structure_title
            self.update_mol_tar_box()
            self.update_preview(reset_angles=True)

    def update_structure_box(self) -> None:
        """Remove all widgets from the structures' box and adds the
        currently loaded structures (from self.structures)."""
        self.structures_box.remove_all_widgets()

        for i in range(len(self.structures)):
            row_count = self.structures_box.content_layout.rowCount()
            title = self.structures[i].title
            structure_button = QtWidgets.QPushButton(title)
            structure_button.released.connect(
                partial(self.on_structure_clicked, title)
            )
            self.structures_box.add_widget(structure_button, row_count, 0)

            delete_button = QtWidgets.QPushButton("Delete")
            delete_button.released.connect(
                partial(self.on_structure_deleted, title)
            )
            self.structures_box.add_widget(delete_button, row_count, 1)

    def update_current_structure(self) -> None:
        """Save info about the current structure."""
        if not self.structures:
            return

        structure = self.find_structure_by_title(self.current_structure)
        structure.restart()

        # iterate over all widgets with molecular targets info
        widgets = [
            self.mol_tar_box.content_layout.itemAt(i).widget()
            for i in range(self.mol_tar_box.content_layout.count())
        ]
        targets = []
        xs = []
        ys = []
        zs = []
        for widget in widgets:
            if isinstance(widget, QtWidgets.QLabel):
                continue

            if "target" in widget.objectName():
                targets.append(widget.text())
            elif "x" in widget.objectName():
                xs.append(widget.value())
            elif "y" in widget.objectName():
                ys.append(widget.value())
            elif "z" in widget.objectName():
                zs.append(widget.value())

        for target, x, y, z in zip(targets, xs, ys, zs):
            structure.define_coordinates(target, [x], [y], [z])

    def on_structure_clicked(self, title: str) -> None:
        """Change focus onto the clicked structure."""
        # save the changes to the current structure
        self.update_current_structure()
        # change to the new structure
        self.current_structure = title
        self.update_mol_tar_box()
        self.update_preview(reset_angles=True)

    def on_structure_deleted(self, title: str) -> None:
        """Delete the given structure."""
        structure = self.find_structure_by_title(title)
        self.structures.remove(structure)
        self.update_structure_box()
        if title == self.current_structure:
            if self.structures:
                self.current_structure = self.structures[0].title
            else:
                self.current_structure = None
        self.update_mol_tar_box()
        self.update_preview(reset_angles=True)

    def save_structures(self) -> None:
        """Save current structures as a .yaml file."""
        self.update_current_structure()  # in case it was not saved yet
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save structures", filter="*.yaml"
        )
        if path:
            info = []
            for structure in self.structures:
                m_info = {
                    "Structure title": structure.title,
                    "Molecular targets": structure.targets,
                }
                for target in structure.targets:
                    m_info[f"{target}_x"] = structure.x[target]
                    m_info[f"{target}_y"] = structure.y[target]
                    m_info[f"{target}_z"] = structure.z[target]
                info.append(m_info)
            io.save_info(path, info)

    def load_structures(self) -> None:
        """Load structures from in a .yaml file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load structures", filter="*.yaml"
        )
        if path:
            with open(path, 'r') as file:
                try:
                    info = list(yaml.load_all(file, Loader=yaml.FullLoader))
                except TypeError:
                    raise TypeError(
                        "Incorrect file. Please choose a file that was created"
                        " by SPINNA."
                    )
            if "Structure title" not in info[0]:
                raise TypeError(
                    "Incorrect file. Please choose a file that was created"
                    " by SPINNA."
                )
            # continue if the correct file is loaded
            for m_info in info:
                structure = spinna.Structure(m_info["Structure title"])
                for target in m_info["Molecular targets"]:
                    x = m_info[f"{target}_x"]
                    y = m_info[f"{target}_y"]
                    z = m_info[f"{target}_z"]
                    structure.define_coordinates(target, x, y, z)
                self.structures.append(structure)

            self.current_structure = self.structures[0].title
            self.update_structure_box()
            self.update_mol_tar_box()
            self.update_preview(reset_angles=True)

    def find_structure_by_title(self, title: str) -> spinna.Structure | None:
        """Return the structure with the given title."""
        if title is None:
            return None
        for structure in self.structures:
            if title == structure.title:
                return structure

    def add_molecular_target(self) -> None:
        """Add a new molecular target to the current structure."""
        if self.current_structure is None:
            text = (
                "No structure has been selected. Please click on one of the"
                " structures above or add a new structure."
            )
            QtWidgets.QMessageBox.information(self, "Warning", text)
            return

        # extract the name of the last added molecular target
        count = self.mol_tar_box.content_layout.count()
        if count == 5:
            name = "target"
        else:
            name = self.mol_tar_box.content_layout.itemAt(
                count-5
            ).widget().text()

        name_widget = QtWidgets.QLineEdit(objectName=f"target{self.n_mol_tar}")
        name_widget.setText(name)
        name_widget.editingFinished.connect(self.update_preview)
        x_widget = QtWidgets.QDoubleSpinBox(objectName=f"x{self.n_mol_tar}")
        y_widget = QtWidgets.QDoubleSpinBox(objectName=f"y{self.n_mol_tar}")
        z_widget = QtWidgets.QDoubleSpinBox(objectName=f"z{self.n_mol_tar}")
        for spinbox in (x_widget, y_widget, z_widget):
            spinbox.setRange(-1000, 1000)
            spinbox.setDecimals(2)
            spinbox.setSingleStep(0.1)
            spinbox.setKeyboardTracking(True)
            spinbox.setValue(0)
            spinbox.valueChanged.connect(self.update_preview)
        delete_button = QtWidgets.QPushButton(
            "x", objectName=f"del{self.n_mol_tar}"
        )
        delete_button.released.connect(
            partial(self.delete_molecular_target, delete_button.objectName())
        )

        # add the widgets
        rowcount = self.mol_tar_box.content_layout.rowCount()
        self.mol_tar_box.add_widget(name_widget, rowcount, 0)
        self.mol_tar_box.add_widget(x_widget, rowcount, 1)
        self.mol_tar_box.add_widget(y_widget, rowcount, 2)
        self.mol_tar_box.add_widget(z_widget, rowcount, 3)
        self.mol_tar_box.add_widget(delete_button, rowcount, 4)

        self.n_mol_tar += 1
        self.update_preview()

    def delete_molecular_target(self, name: str) -> None:
        """Delete the widgets in the molecular targets box corresponding
        to the chosen molecular target."""
        name = name[3:]  # extract the number after 'del'
        widgets = [
            self.mol_tar_box.content_layout.itemAt(i).widget()
            for i in range(self.mol_tar_box.content_layout.count())
        ]
        row = int(name)
        for widget in widgets:
            if widget.objectName():  # skip labels
                base, num = split_name(widget.objectName())
                if num == row:  # delete the widget if the same row
                    self.mol_tar_box.content_layout.removeWidget(widget)
                    del widget
                elif num > row:   # if the widget below, lower the object name
                    widget.setObjectName(f"{base}{num-1}")
                    # if this is delete button, connect it to a new function
                    if base == 'del':
                        widget.disconnect()
                        widget.released.connect(
                            partial(
                                self.delete_molecular_target, f"del{num-1}",
                            )
                        )
        self.n_mol_tar -= 1
        self.update_preview()

    def update_mol_tar_box(self) -> None:
        """Delete widgets from the molecular targets box and load the
        widgets corresponding to the currently loaded structure."""

        if self.mol_tar_box.content_layout.count() > 5:
            self.mol_tar_box.remove_all_widgets(keep_labels=True)
            self.n_mol_tar = 0

        if not self.structures:
            return

        structure = deepcopy(
            self.find_structure_by_title(self.current_structure)
        )
        for target in structure.targets:
            for x, y, z in zip(
                structure.x[target],
                structure.y[target],
                structure.z[target],
            ):
                row = self.mol_tar_box.content_layout.rowCount()
                name_widget = QtWidgets.QLineEdit(
                    objectName=f"target{self.n_mol_tar}"
                )
                name_widget.setText(target)
                name_widget.editingFinished.connect(self.update_preview)
                x_widget = QtWidgets.QDoubleSpinBox(
                    objectName=f"x{self.n_mol_tar}"
                )
                y_widget = QtWidgets.QDoubleSpinBox(
                    objectName=f"y{self.n_mol_tar}"
                )
                z_widget = QtWidgets.QDoubleSpinBox(
                    objectName=f"z{self.n_mol_tar}"
                )
                for spinbox in (x_widget, y_widget, z_widget):
                    spinbox.setRange(-1000, 1000)
                    spinbox.setDecimals(2)
                    spinbox.setSingleStep(0.1)
                    spinbox.setKeyboardTracking(True)
                    spinbox.valueChanged.connect(self.update_preview)
                # set value now when negative values are allowed
                x_widget.setValue(x)
                y_widget.setValue(y)
                z_widget.setValue(z)
                delete_button = QtWidgets.QPushButton(
                    "x", objectName=f"del{self.n_mol_tar}"
                )
                delete_button.released.connect(partial(
                    self.delete_molecular_target, delete_button.objectName()
                ))

                self.mol_tar_box.add_widget(name_widget, row, 0)
                self.mol_tar_box.add_widget(x_widget, row, 1)
                self.mol_tar_box.add_widget(y_widget, row, 2)
                self.mol_tar_box.add_widget(z_widget, row, 3)
                self.mol_tar_box.add_widget(delete_button, row, 4)

                self.n_mol_tar += 1

    def save_preview(self) -> None:
        """Save current preview."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save current view", filter="*.png;;.tif"
        )
        if path:
            self.preview.qimage.save(path)


class GenerateSearchSpaceDialog(QtWidgets.QDialog):
    """Input dialog to get the parameters for generating numbers of
    structures (stoichiometries) for SPINNA fitting.

    ...

    Attributes
    ----------
    buttons : QtWidgets.QDialogButtonBox
        Buttons for accepting or rejecting the input.
    n_sim_spin : QtWidgets.QSpinBox
        Number of simulation repeats.
    granularity_spin : QtWidgets.QSpinBox
        Granularity. Controls how many proportions combinations of
        structures are tested.
    save_check : QtWidgets.QCheckBox
        Checkbox for saving the results as a .csv file.
    """

    def __init__(self, sim_tab: QtWidgets.QWidget) -> None:
        super().__init__(sim_tab)
        self.setWindowTitle("Enter parameters")
        vbox = QtWidgets.QVBoxLayout(self)
        layout = QtWidgets.QFormLayout()
        vbox.addLayout(layout)

        self.n_sim_spin = QtWidgets.QSpinBox()
        self.n_sim_spin.setRange(1, 999_999)
        if sim_tab.n_sim_fit is None:
            self.n_sim_spin.setValue(10)
        else:
            self.n_sim_spin.setValue(sim_tab.n_sim_fit)
        layout.addRow(QtWidgets.QLabel("# simulations:"), self.n_sim_spin)

        self.granularity_spin = QtWidgets.QSpinBox()
        self.granularity_spin.setRange(0, 50_000)
        if sim_tab.granularity is None:
            self.granularity_spin.setValue(21)
        else:
            self.granularity_spin.setValue(sim_tab.granularity)
        layout.addRow(QtWidgets.QLabel("Granularity:"), self.granularity_spin)

        self.save_check = QtWidgets.QCheckBox("Save as .csv")
        self.save_check.setChecked(False)
        layout.addRow(self.save_check, QtWidgets.QLabel(" "))

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent: QtWidgets.QWidget) -> list[int | bool]:
        """Create the dialog and returns the numbers of molecular
        targets per simulation, number of simulations, resolution
        factor, check if the results are to be saved."""
        dialog = GenerateSearchSpaceDialog(parent)
        result = dialog.exec_()
        return [
            int(dialog.n_sim_spin.value()),
            int(dialog.granularity_spin.value()),
            dialog.save_check.isChecked(),
            result == QtWidgets.QDialog.Accepted,
        ]


class CompareModelsDialog(QtWidgets.QDialog):
    """Dialog for comparing different models (lists of structures)
    and label uncertainties. Useful for fine-tuning and exploring the
    model structures.

    ...

    Attributes
    ----------
    buttons : QtWidgets.QDialogButtonBox
        Buttons for accepting or rejecting the input.
    label_unc_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling label uncertainties.
    label_unc_from_spins : dict
        Spin boxes for setting the lower bounds of label uncertainties.
    label_unc_to_spins : dict
        Spin boxes for setting the upper bounds of label uncertainties.
    label_unc_step_spins : dict
        Spin boxes for setting the step of label uncertainties.
    models : list of spinna.Structure
        List of all models.
    models_box : ScrollableGroupBox
        Box with a summary of all models.
    model_buttons : list of QtWidgets.QPushButton
        Buttons for removing the models.
    model_paths : list of str
        Paths to the models.
    save_fit_scores : QtWidgets.QCheckBox
        Checkbox for saving the fit scores.
    sim_tab : SimulationsTab
        Simulation tab, parent widget.
    targets : list of str
        Names of the molecular targets in the models.
    """

    def __init__(self, sim_tab: SimulationsTab, targets: list[str]) -> None:
        super().__init__(sim_tab)
        self.setWindowTitle("Compare models")
        self.setModal(True)
        self.sim_tab = sim_tab
        self.targets = targets
        self.models = []
        self.model_buttons = []
        self.model_paths = []
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        # LABEL UNCERTAINTIES
        label_unc_layout = QtWidgets.QGridLayout()
        layout.addLayout(label_unc_layout)
        self.label_unc_checkbox = QtWidgets.QCheckBox("Label uncertainties")
        self.label_unc_checkbox.setChecked(False)
        self.label_unc_checkbox.toggled.connect(self.on_label_unc_toggled)
        label_unc_layout.addWidget(self.label_unc_checkbox, 0, 0, 1, 6)
        # from, to, step
        label_unc_layout.addWidget(QtWidgets.QLabel("Target"), 1, 0)
        label_unc_layout.addWidget(QtWidgets.QLabel("From"), 1, 1)
        label_unc_layout.addWidget(QtWidgets.QLabel("To"), 1, 2)
        label_unc_layout.addWidget(QtWidgets.QLabel("Step"), 1, 3)
        self.label_unc_from_spins = {}
        self.label_unc_to_spins = {}
        self.label_unc_step_spins = {}
        for target in targets:
            label_unc_from_spin = QtWidgets.QDoubleSpinBox()
            label_unc_from_spin.setRange(0, 20)
            label_unc_from_spin.setDecimals(2)
            label_unc_from_spin.setSingleStep(0.1)
            label_unc_from_spin.setValue(3)
            label_unc_to_spin = QtWidgets.QDoubleSpinBox()
            label_unc_to_spin.setRange(0, 20)
            label_unc_to_spin.setDecimals(2)
            label_unc_to_spin.setSingleStep(0.1)
            label_unc_to_spin.setValue(8)
            label_unc_step_spin = QtWidgets.QDoubleSpinBox()
            label_unc_step_spin.setRange(0.1, 10)
            label_unc_step_spin.setDecimals(2)
            label_unc_step_spin.setSingleStep(0.1)
            label_unc_step_spin.setValue(1.0)
            label_unc_from_spin.setEnabled(False)
            label_unc_to_spin.setEnabled(False)
            label_unc_step_spin.setEnabled(False)
            self.label_unc_from_spins[target] = label_unc_from_spin
            self.label_unc_to_spins[target] = label_unc_to_spin
            self.label_unc_step_spins[target] = label_unc_step_spin
            label_unc_layout.addWidget(
                QtWidgets.QLabel(target), 2+targets.index(target), 0
            )
            label_unc_layout.addWidget(
                label_unc_from_spin, 2+targets.index(target), 1
            )
            label_unc_layout.addWidget(
                label_unc_to_spin, 2+targets.index(target), 2
            )
            label_unc_layout.addWidget(
                label_unc_step_spin, 2+targets.index(target), 3
            )

        # models
        self.models_box = lib.ScrollableGroupBox("Models (click to remove)")
        self.models_box.setMinimumHeight(250)
        layout.addWidget(self.models_box)
        add_model_button = QtWidgets.QPushButton("Add a model")
        add_model_button.setStyleSheet("font-weight : bold")
        add_model_button.released.connect(self.on_add_model)
        self.models_box.add_widget(add_model_button, 0, 0)

        # save fit scores
        self.save_fit_scores = QtWidgets.QCheckBox("Save fit scores")
        self.save_fit_scores.setChecked(False)
        layout.addWidget(self.save_fit_scores)

        # cancel/accept buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(
        parent: QtWidgets.QWidget,
        targets: list[str],
    ) -> tuple[list[dict], list[str], dict[str, np.ndarray], bool, bool]:
        dialog = CompareModelsDialog(parent, targets)
        result = dialog.exec_()
        label_unc = {}
        if dialog.label_unc_checkbox.isChecked():
            for target in targets:
                label_unc[target] = np.arange(
                    dialog.label_unc_from_spins[target].value(),
                    dialog.label_unc_to_spins[target].value() + 0.01,
                    dialog.label_unc_step_spins[target].value(),
                )
        else:
            for target, spin in zip(targets, parent.label_unc_spins):
                label_unc[target] = [spin.value()]
        return (
            dialog.models,
            [os.path.basename(path) for path in dialog.model_paths],
            label_unc,
            dialog.save_fit_scores.isChecked(),
            result == QtWidgets.QDialog.Accepted,
        )

    def on_label_unc_toggled(self, state: bool) -> None:
        """Enable/disable the label uncertainties spin boxes."""
        for target in self.targets:
            self.label_unc_from_spins[target].setEnabled(state)
            self.label_unc_to_spins[target].setEnabled(state)
            self.label_unc_step_spins[target].setEnabled(state)

    def on_add_model(self) -> None:
        """Add a new model (list of structures) to the dialog."""
        paths, ext = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Choose model(s)", filter="*.yaml"
        )
        if not paths:
            return

        for path in paths:
            structures, targets = spinna.load_structures(path)
            # check that the loaded targets match the exp. data's targets
            if not set(targets) == set(self.sim_tab.targets):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Warning",
                    (
                        f"Loaded model {path} does not contain all molecular"
                        " targets as specified before."
                    ),
                )
                return

            # add to the models box
            model_button = QtWidgets.QPushButton(
                os.path.splitext(os.path.basename(path))[0]
            )
            model_button.released.connect(partial(self.on_model_clicked, path))
            self.models_box.add_widget(
                model_button, self.models_box.content_layout.rowCount(), 0
            )

            # add attributes
            self.models.append(structures)
            self.model_paths.append(path)
            self.model_buttons.append(model_button)

    def on_model_clicked(self, path: str) -> None:
        """Remove the model from the dialog (model box) and the
        dialog's attributes."""
        index = self.model_paths.index(path)
        self.models_box.content_layout.removeWidget(self.model_buttons[index])
        self.model_buttons[index].setParent(None)
        del self.models[index]
        del self.model_paths[index]
        del self.model_buttons[index]


class OptionalSettingsDialog(QtWidgets.QDialog):
    """Dialog for setting optional parameters in the Simulations Tab.

    ...

    Attributes
    ----------
    asynch_check : QtWidgets.QCheckBox
        Checkbox for using multiprocessing.
    auto_nn_check : QtWidgets.QCheckBox
        Checkbox for auto setting the number of neighbors to consider
        at fitting.
    neighbors_layout : QtWidgets.QFormLayout
        Layout for setting the numbers of neighbors to consider at
        fitting.
    nn_counts : dict
        Numbers of neighbors to consider at fitting for each pair of
        molecular targets. Only used when self.auto_nn_check is
        unchecked.
    rot_dim_widget : QtWidgets.QComboBox
        Widget for choosing the rotations mode (2D, 3D, none).
    sim_tab : spinna.SimulationsTab
        The parent tab.
    """

    def __init__(self, sim_tab: spinna.SimulationsTab) -> None:
        super().__init__(sim_tab)
        self.sim_tab = sim_tab
        self.setWindowTitle("Optional settings")
        self.setModal(True)
        layout = QtWidgets.QVBoxLayout(self)
        self.nn_counts = {}

        # OPTIONAL SETTINGS
        # rotations mode
        self.rot_dim_widget = QtWidgets.QComboBox()
        self.rot_dim_widget.addItems(
            ["random 2D rotations", "random 3D rotations", "No rotations"]
        )
        self.rot_dim_widget.setCurrentIndex(0)
        layout.addWidget(self.rot_dim_widget)

        # use multiprocessing (parallel processing)
        self.asynch_check = QtWidgets.QCheckBox("Use multiprocessing")
        self.asynch_check.setChecked(True)
        layout.addWidget(self.asynch_check)

        # numbers of neighbors to consider at fitting
        self.auto_nn_check = QtWidgets.QCheckBox(
            "Auto set # of NNs for fitting"
        )
        self.auto_nn_check.stateChanged.connect(self.on_auto_nn_checked)
        self.auto_nn_check.setChecked(True)
        layout.addWidget(self.auto_nn_check)

        self.neighbors_layout = QtWidgets.QFormLayout()
        layout.addLayout(self.neighbors_layout)

    def on_auto_nn_checked(self, state: int) -> None:
        """Adjust the number of neighbors to consider at fitting."""
        if state == 0:  # unchecked
            for spin in self.nn_counts.values():
                spin.setEnabled(True)
        elif state == 2:
            for spin in self.nn_counts.values():
                spin.setEnabled(False)
        else:
            raise KeyError("Incorrect state.")

        for spin in self.nn_counts.values():
            spin.setEnabled(not self.auto_nn_check.isChecked())

    def update_neighbors_widgets(self) -> None:
        """Delete the old widgets and add new ones for setting the
        numbers of neighbors to consider at fitting."""
        # delete old widgets
        for i in reversed(range(self.neighbors_layout.count())):
            self.neighbors_layout.itemAt(i).widget().setParent(None)
        self.nn_counts = {}

        # add new widgets
        for i, t1 in enumerate(self.sim_tab.targets):
            for t2 in self.sim_tab.targets[i:]:
                name = f"{t1}-{t2}"
                spin = QtWidgets.QSpinBox(objectName=name)
                spin.setRange(0, 10)
                spin.setValue(1)
                self.neighbors_layout.addRow(
                    QtWidgets.QLabel(f"NN {t1} \u2192 {t2}:"), spin
                )
                self.nn_counts[name] = spin
                if self.auto_nn_check.isChecked():
                    spin.setEnabled(False)


class NNDPlotSettingsDialog(QtWidgets.QDialog):
    """Dialog for adjusting settings for plotting nearest neighbors
    distances.

    ...

    Attributes
    ----------
    alpha : QtWidgets.QDoubleSpinBox
        Transparency of the bins in the histograms.
    binsize_exp : QtWidgets.QDoubleSpinBox
        Bin size for plotting the experimental data.
    binsize_sim : QtWidgets.QDoubleSpinBox
        Bin size for plotting the simulated data.
    colors : list of QtWidgets.QLineEdit
        Colors for the histograms.
    min_dist : QtWidgets.QSpinBox
        Minimum distance to plot.
    max_dist : QtWidgets.QSpinBox
        Maximum distance to plot.
    neighbors_layout : QtWidgets.QFormLayout
        Layout for setting the numbers of neighbors to plot.
    nn_counts : dict
        Numbers of neighbors to plot for each pair of molecular targets.
    nnd_legend_check : QtWidgets.QCheckBox
        Checkbox for showing/hiding the legend.
    sim_tab : spinna.SimulationsTab
        The parent tab.
    title : QtWidgets.QLineEdit
        Title of the plot.
    xlabel : QtWidgets.QLineEdit
        X-axis label.
    ylabel : QtWidgets.QLineEdit
        Y-axis label.
    """

    def __init__(self, sim_tab: spinna.SimulationsTab) -> None:
        super().__init__(sim_tab)
        self.sim_tab = sim_tab
        self.setWindowTitle("Nearest neighbors plots")
        self.setModal(False)
        self.nn_counts = {}
        main_layout = QtWidgets.QVBoxLayout(self)
        const_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(const_layout)

        # update (run a simulation)
        update_button = QtWidgets.QPushButton("Update plot(s)")
        update_button.released.connect(self.update_plots)

        # legend
        self.nnd_legend_check = QtWidgets.QCheckBox("Show legend")
        self.nnd_legend_check.setChecked(False)
        self.nnd_legend_check.stateChanged.connect(
            self.sim_tab.display_current_nnd_plot
        )
        const_layout.addRow(self.nnd_legend_check, update_button)

        # bin sizes (exp and sim)
        self.binsize_sim = QtWidgets.QDoubleSpinBox()
        self.binsize_sim.setRange(0.1, 100)
        self.binsize_sim.setValue(4.0)
        self.binsize_sim.setDecimals(1)
        self.binsize_sim.setSingleStep(0.1)
        const_layout.addRow(
            QtWidgets.QLabel("Bin size sim (nm):"), self.binsize_sim
        )
        self.binsize_exp = QtWidgets.QDoubleSpinBox()
        self.binsize_exp.setRange(0.1, 100)
        self.binsize_exp.setValue(4.0)
        self.binsize_exp.setDecimals(1)
        self.binsize_exp.setSingleStep(0.1)
        const_layout.addRow(
            QtWidgets.QLabel("Bin size exp (nm):"), self.binsize_exp
        )

        # distance limits
        self.min_dist = QtWidgets.QSpinBox()
        self.min_dist.setRange(0, 99999)
        self.min_dist.setValue(0)
        const_layout.addRow(
            QtWidgets.QLabel("Min dist (nm):"), self.min_dist
        )

        self.max_dist = QtWidgets.QSpinBox()
        self.max_dist.setRange(0, 99999)
        self.max_dist.setValue(200)
        const_layout.addRow(
            QtWidgets.QLabel("Max dist (nm):"), self.max_dist
        )

        # title
        self.title = QtWidgets.QLineEdit()
        self.title.setText("Nearest Neighbors Distances:")
        const_layout.addRow(QtWidgets.QLabel("Title:"), self.title)

        # labels
        self.xlabel = QtWidgets.QLineEdit()
        self.xlabel.setText("Distance (nm)")
        const_layout.addRow(QtWidgets.QLabel("X-axis label:"), self.xlabel)
        self.ylabel = QtWidgets.QLineEdit()
        self.ylabel.setText("Norm. frequency")
        const_layout.addRow(QtWidgets.QLabel("Y-axis label:"), self.ylabel)

        # alpha (for histograms only)
        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0, 1)
        self.alpha.setValue(0.6)
        self.alpha.setDecimals(2)
        self.alpha.setSingleStep(0.01)
        const_layout.addRow(
            QtWidgets.QLabel("Transparency (bins):"), self.alpha
        )

        # colors
        const_layout.addRow(QtWidgets.QLabel("Colors:"), QtWidgets.QLabel(" "))
        self.colors = []
        for prefix, i in zip(["1st", "2nd", "3rd", "4th"], range(4)):
            color = QtWidgets.QLineEdit()
            color.setText(spinna.NN_COLORS[i])
            color.editingFinished.connect(self.check_color_labels)
            const_layout.addRow(QtWidgets.QLabel(f"{prefix} NN:"), color)
            self.colors.append(color)
        for i in range(5, 11):
            color = QtWidgets.QLineEdit()
            color.setText("None")
            color.editingFinished.connect(self.check_color_labels)
            const_layout.addRow(QtWidgets.QLabel(f"{i}th NN:"), color)
            self.colors.append(color)

        # numbers of neighbors (similar to OptionalSettingsDialog)
        self.neighbors_layout = QtWidgets.QFormLayout()
        main_layout.addLayout(self.neighbors_layout)

    def update_neighbors_widgets(self) -> None:
        """Delete the old widgets and add new ones for setting the
        numbers of neighbors to plot."""
        # delete old widgets
        for i in reversed(range(self.neighbors_layout.count())):
            self.neighbors_layout.itemAt(i).widget().setParent(None)
        self.nn_counts = {}

        # add new widgets
        for i, t1 in enumerate(self.sim_tab.targets):
            for t2 in self.sim_tab.targets[i:]:
                name = f"{t1}-{t2}"
                spin = QtWidgets.QSpinBox(objectName=name)
                spin.setRange(0, 10)
                spin.setValue(2)
                self.neighbors_layout.addRow(
                    QtWidgets.QLabel(f"NN {t1} \u2192 {t2}:"), spin
                )
                self.nn_counts[name] = spin

                # and reverse
                if t1 != t2:
                    name = f"{t2}-{t1}"
                    spin = QtWidgets.QSpinBox(objectName=name)
                    spin.setRange(0, 10)
                    spin.setValue(2)
                    self.neighbors_layout.addRow(
                        QtWidgets.QLabel(f"NN {t2} \u2192 {t1}:"), spin
                    )
                    self.nn_counts[name] = spin

    def check_color_labels(self) -> None:
        """Check if colors are valid, see:
        https://matplotlib.org/stable/tutorials/colors/colors.html."""
        for color in self.colors:
            if color.text().lower() == "none" or color.text() == "":
                continue
            try:
                matplotlib.colors.to_rgba(color.text())
            except ValueError:
                message = (
                    "Incorrect color. Please choose a color specified by: "
                    "https://matplotlib.org/stable/tutorials/colors/colors."
                    "html. If the color is not needed, leave the space blank "
                    "or type 'None'."
                )
                QtWidgets.QMessageBox.warning(self, "Warning", message)

    def update_plots(self) -> None:
        """Run a single simulation (if data loaded) or update the plots
        of the experimental NNDs only."""
        if self.sim_tab.mixer is None:  # plot experimental data only
            self.sim_tab.plot_exp_nnds()
        else:  # simulation
            self.sim_tab.run_single_sim()

    def extract_params(self) -> dict:
        """Extract the parameters from the dialog.

        Returns
        -------
        params : dict
            Parameters for plotting the nearest neighbors plots.
        """
        mindist = self.min_dist.value()
        maxdist = self.max_dist.value()
        if mindist > maxdist:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Max. distance is lower than min. distance."
            )

        binsize_sim = self.binsize_sim.value()
        binsize_exp = self.binsize_exp.value()
        if max(binsize_sim, binsize_exp) > maxdist-mindist:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Bin size is larger than the distance range."
            )

        title = self.title.text()
        if not title.endswith(" "):
            title += " "

        colors = []
        for color in self.colors:
            if color.text().lower() == "none" or color.text() == "":
                break
            colors.append(color.text())

        params = {
            "binsize_sim": binsize_sim,
            "binsize_exp": binsize_exp,
            "min_dist": mindist,
            "max_dist": maxdist,
            "title": title,
            "xlabel": self.xlabel.text(),
            "ylabel": self.ylabel.text(),
            "alpha": self.alpha.value(),
            "colors": colors,
            "nn_counts": self.nn_counts,
        }
        return params


class SimulationsPlotWindow(QtWidgets.QLabel):
    """Label used for displaying NND plots."""

    def __init__(self, sim_tab):
        super().__init__(sim_tab)
        self.setFixedHeight(int(NND_PLOT_SIZE / 1.3333))
        self.setFixedWidth(NND_PLOT_SIZE)

    def display(self, fig):
        """Display fig - plt.Figure. Uses a somewhat unsual method to
        draw the canvas by saving the .svg format of the figure and
        then creating the QImage isntance. This way, downsampling of
        the image is avoided after drawing on the canvas."""
        # render the figure as .svg
        buffer = python_io.BytesIO()
        fig.savefig(buffer, format='svg')
        buffer.seek(0)
        svg_data = buffer.getvalue().decode()

        # load the data into pyqt5's svg renderer
        svg_renderer = QSvgRenderer()
        svg_renderer.load(svg_data.encode())

        # create the qimage and set pixmap
        qimage = QtGui.QImage(
            QtCore.QSize(NND_PLOT_SIZE, int(NND_PLOT_SIZE / 1.3333)),
            QtGui.QImage.Format_ARGB32_Premultiplied,
        )
        qimage.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(qimage)
        svg_renderer.render(painter)
        painter.end()
        self.setPixmap(QtGui.QPixmap.fromImage(qimage))


class SimulationsTab(QtWidgets.QDialog):
    """Tab for running simulations and finding the proportions of
    structure in the experimental data.

    ...

    Attributes
    ----------
    best_score : float or tuple(float, float)
        Best score of the fitting (KS test statistic), optionally with
        the bootstrap-based uncertainty.
    binsize_exp_spin : QtWidgets.QDoubleSpinBox
        Spin box for setting the bin size for the NND plot for the
        experimental data (nm).
    binsize_sim_spin : QtWidgets.QDoubleSpinBox
        Spin box for setting the bin size for the NND plot for the
        simulated data (nm).
    current_nnd_idx : int
        Index of the currently displayed NND plot.
    current_score : float
        Current score of the fitting (KS 2-sample test statistic).
    densities_spins : list of QtWidgets.QDoubleSpinBox
        Spin boxes for setting densities of observed molecular targets
        (um^-2(or -3)) for each target.
    depth : float
        Depth of the simulation (used for homogenous distribution in 3D
        only).
    depth_stack : QtWidgets.QStackedWidget
        Stack of widgets for setting the depth of the simulation.
        Index == 0 ->
    dim_widget : QtWidgets.QComboBox
        Combo box for setting the dimension of the simulation (2D/3D).
    exp_data : dict
        Experimental data for each target.
    fit_button : QtWidgets.QPushButton
        Button for starting the fitting.
    fit_results_display : QtWidgets.QLabel
        Label for displaying the results of fitting - the proportions
        of best-fit numbers of structures.
    label_unc_box : ScrollableGroupBox
        Box with spin boxes for setting label uncertainty (nm).
    label_unc_spins : list of QtWidgets.QDoubleSpinBox
        Spin boxes for setting label uncertainty (nm) for each target.
    le_box : ScrollableGroupBox
        Box with spin boxes for setting labeling efficiency (%) for
        each target.
    le_fitting_check : QtWidgets.QCheckBox
        Check box for enabling/disabling fitting of labeling efficiency.
    le_spins : list of QtWidgets.QDoubleSpinBox
        Spin boxes for setting labeling efficiency (%) for each target.
    load_exp_data_box : ScrollableGroupBox
        Box with buttons for loading experimental data.
    load_exp_data_buttons : list of QtWidgets.QPushButton
        Buttons for loading experimental data for each target.
    load_mask_buttons : list of QtWidgets.QPushButton
        Buttons for loading masks for each target.
    load_structures_button : QtWidgets.QPushButton
        Button for loading structures.
    mask_button : QtWidgets.QPushButton
        Button for loading masks, switches self.mask_den_stack to the
        mask stack.
    mask_den_stack : QtWidgets.QStackedWidget
        Stack of widgets for setting the mask/density of the simulation.
        Index == 0 -> masks, index == 1 -> homogenous distribution.
    mask_infos : dict
        Information about the masks for each target.
    masks : dict
        Masks for each target.
    structures : list of spinna.Structure's
        List of loaded structures.
    structures_path : str
        Path to loaded structures.
    N_structures_fit : dict
        Number of structures to be simulated for each target when
        fitting.
    nnd_plot_box : SimulationsPlotWindow
        Widget for displaying NN distances.
    nnd_plots : list of SimulationsPlotWindow
        Labels for displaying NND plots.
    nnd_plots_settings_dialog : NNPlotSettingsDialog
        Dialog for adjusting settings of the nearest neighbors plots.
    n_sim_fit : int
        Number of simulations per each combination of numbers of
        structures for fitting.
    n_sim_plot_spin : QtWidgets.QSpinBox
        Spin box for setting the number of simulations to be plotted
        in the NND plot.
    n_total : dict
        Total number of molecules to be simulated for each molecular
        target, i.e., number of observed molecules divided by the
        corresponding labeling efficiency.
    prop_str_input : ScrollableGroupBox
        Box for setting the number of structures to be simulated in a
        single simulation.
    prop_str_input_spins : list of QtWidgets.QDoubleSpinBox
        Spin boxes for setting proportions of structures to be
        simulated in a single simulation.
    rect_roi_button : QtWidgets.QPushButton
        Button for simulating homogenours ROIs, switches
        self.mask_den_stack to the density stack.
    granularity : int
        Granularity used in generating the search space for
        fitting.
    roi_button : QtWidgets.QPushButton
        Buttons for loading area/volume of the ROI. Used for
        homogenous distribution only.
    rot_dim_widget : QtWidgets.QComboBox
        Combo box for setting the dimension of the random rotations
        of simulated structures (2D/3D).
    run_single_sim_button : QtWidgets.QPushButton
        Button for running a single simulation.
    save_fit_results_check : QtWidgets.QCheckBox
        Checkbox for saving the results of fitting (parameter search
        space and the corresponding fitting scores (KS)).
    save_sim_result_check : QtWidgets.QCheckBox
        Checkbox for saving the molecules resulting from a single
        simulation.
    settings_dialog : OptionalSettingsDialog
        Dialog for setting optional parameters (rotations, numbers of
        neighbors to consider at fitting).
    single_sim_mass : float
        Area/volume of a single simulation (um^2(or um^3)). Used for
        homogenous distribution only.
    targets : list of str
        Names of all unique molecular targets in the loaded structures.
    window : QtWidgets.QMainWindow
        Main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.setAutoFillBackground(True)
        layout = QtWidgets.QGridLayout(self)
        left_column = QtWidgets.QGridLayout()
        right_column = QtWidgets.QGridLayout()
        layout.addLayout(left_column, 0, 0)
        layout.addLayout(right_column, 0, 1)

        self.structures = []
        self.targets = []
        self.exp_data = {}
        self.exp_data_paths = {}
        self.masks = {}
        self.mask_infos = {}
        self.mask_paths = {}
        self.N_structures_fit = {}
        self.n_total = {}

        self.load_mask_buttons = []
        self.load_exp_data_buttons = []
        self.densities_spins = []
        self.label_unc_spins = []
        self.le_spins = []
        self.prop_str_input_spins = []
        self.nnd_plots = []

        self.current_nnd_idx = 0
        self.depth = None
        self.granularity = None
        self.n_sim_fit = None
        self.single_sim_mass = None
        self.mixer = None
        self.current_score = 0.0
        self.structures_path = ""
        self.settings_dialog = OptionalSettingsDialog(self)
        self.nn_plot_settings_dialog = NNDPlotSettingsDialog(self)

        # LOAD DATA
        load_data_box = QtWidgets.QGroupBox("Load data")
        load_data_box.setFixedHeight(450)
        left_column.addWidget(load_data_box, 0, 0)
        load_data_layout = QtWidgets.QGridLayout(load_data_box)

        basic_buttons_layout = QtWidgets.QVBoxLayout()
        load_data_layout.addLayout(basic_buttons_layout, 0, 0)
        self.load_structures_button = QtWidgets.QPushButton("Load structures")
        self.load_structures_button.released.connect(self.load_structures)
        basic_buttons_layout.addWidget(self.load_structures_button)

        self.dim_widget = QtWidgets.QComboBox()
        self.dim_widget.addItems(["2D simulation", "3D simulation"])
        self.dim_widget.setCurrentIndex(0)
        self.dim_widget.currentIndexChanged.connect(self.on_dim_changed)
        basic_buttons_layout.addWidget(self.dim_widget)

        optional_settings_button = QtWidgets.QPushButton("Optional settings")
        optional_settings_button.released.connect(self.settings_dialog.show)
        basic_buttons_layout.addWidget(optional_settings_button)

        self.depth_stack = QtWidgets.QStackedWidget()
        basic_buttons_layout.addWidget(self.depth_stack)
        self.depth_stack.addWidget(QtWidgets.QLabel("     "))
        self.depth_button = QtWidgets.QPushButton("Z range (nm)")
        self.depth_button.released.connect(self.on_depth_button_clicked)
        self.depth_stack.addWidget(self.depth_button)
        self.depth_stack.setCurrentIndex(0)

        self.load_exp_data_box = lib.ScrollableGroupBox("Experimental data")
        load_data_layout.addWidget(self.load_exp_data_box, 0, 1)

        self.label_unc_box = lib.ScrollableGroupBox(
            "Label uncertainty (nm)", layout="form"
        )
        load_data_layout.addWidget(self.label_unc_box, 1, 0)

        self.le_box = lib.ScrollableGroupBox(
            "labeling efficiency (%)", layout="form"
        )
        load_data_layout.addWidget(self.le_box, 1, 1)

        mask_den_layout = QtWidgets.QVBoxLayout()
        load_data_layout.addLayout(mask_den_layout, 2, 0)
        self.mask_button = QtWidgets.QPushButton("Masks")
        self.mask_button.released.connect(
            partial(self.set_mask_den_stack, self.mask_button.text())
        )
        self.mask_button.setStyleSheet("background-color : gray")
        mask_den_layout.addWidget(self.mask_button)
        self.rect_roi_button = QtWidgets.QPushButton(
            "Homogeneous\ndistribution"
        )
        self.rect_roi_button.released.connect(
            partial(self.set_mask_den_stack, self.rect_roi_button.text())
        )
        self.rect_roi_button.setStyleSheet("background-color : lightgreen")
        mask_den_layout.addWidget(self.rect_roi_button)

        self.mask_den_stack = QtWidgets.QStackedWidget()
        load_data_layout.addWidget(self.mask_den_stack, 2, 1)
        self.load_mask_box = lib.ScrollableGroupBox("Masks")
        self.mask_den_stack.addWidget(self.load_mask_box)
        self.densities_box = lib.ScrollableGroupBox(
            "Observed densities (\u03bcm\u207b\u00b2)", layout="form"
        )
        self.mask_den_stack.addWidget(self.densities_box)
        self.mask_den_stack.setCurrentIndex(1)

        # NND PLOT
        nnd_plot_box = QtWidgets.QGroupBox("Plotting")
        nnd_plot_box.setFixedHeight(500)
        right_column.addWidget(nnd_plot_box, 0, 0)
        nnd_plot_layout = QtWidgets.QVBoxLayout(nnd_plot_box)
        nnd_plot_layout.setSpacing(8)

        self.nnd_plot_box = SimulationsPlotWindow(self)
        nnd_plot_layout.addWidget(self.nnd_plot_box)

        nnd_buttons_layout = QtWidgets.QGridLayout()
        nnd_plot_layout.addLayout(nnd_buttons_layout)

        left_nnd_button = QtWidgets.QPushButton("<---")
        left_nnd_button.released.connect(self.on_left_nnd_clicked)
        nnd_buttons_layout.addWidget(left_nnd_button, 0, 0, 1, 2)

        right_nnd_button = QtWidgets.QPushButton("--->")
        right_nnd_button.released.connect(self.on_right_nnd_clicked)
        nnd_buttons_layout.addWidget(right_nnd_button, 0, 2, 1, 2)

        save_nnd_png_button = QtWidgets.QPushButton("Save plots")
        save_nnd_png_button.released.connect(self.save_nnd_plots)
        nnd_buttons_layout.addWidget(save_nnd_png_button, 1, 0, 1, 2)

        save_nnd_csv_button = QtWidgets.QPushButton("Save values")
        save_nnd_csv_button.released.connect(self.save_nnd_values)
        nnd_buttons_layout.addWidget(save_nnd_csv_button, 1, 2, 1, 2)

        nnd_buttons_layout.addWidget(
            QtWidgets.QLabel("# simulations:"), 2, 0, 1, 1
        )
        self.n_sim_plot_spin = QtWidgets.QSpinBox()
        self.n_sim_plot_spin.setValue(1)
        self.n_sim_plot_spin.setRange(1, 1000)
        self.n_sim_plot_spin.setSingleStep(1)
        nnd_buttons_layout.addWidget(self.n_sim_plot_spin, 2, 1, 1, 1)

        plot_settings_button = QtWidgets.QPushButton("Plot settings")
        plot_settings_button.released.connect(
            self.nn_plot_settings_dialog.show
        )
        nnd_buttons_layout.addWidget(plot_settings_button, 2, 2, 1, 2)

        # FITTING
        fitting_box = QtWidgets.QGroupBox("Fitting")
        fitting_box.setFixedHeight(250)
        left_column.addWidget(fitting_box, 1, 0)
        fitting_layout = QtWidgets.QGridLayout(fitting_box)

        generate_search_space_button = QtWidgets.QPushButton(
            "Generate parameter\nsearch space"
        )
        generate_search_space_button.setFixedHeight(60)
        generate_search_space_button.released.connect(
            self.generate_search_space
        )
        fitting_layout.addWidget(generate_search_space_button, 0, 0)

        load_search_space_button = QtWidgets.QPushButton(
            "Load parameter\nsearch space"
        )
        load_search_space_button.setFixedHeight(60)
        load_search_space_button.released.connect(self.load_search_space)
        fitting_layout.addWidget(load_search_space_button, 0, 1)

        compare_models_button = QtWidgets.QPushButton("Compare models")
        compare_models_button.setFixedHeight(60)
        compare_models_button.released.connect(self.compare_models)
        fitting_layout.addWidget(compare_models_button, 0, 2)

        self.save_fit_results_check = QtWidgets.QCheckBox(
            "Save fitting scores"
        )
        self.save_fit_results_check.setChecked(False)
        fitting_layout.addWidget(self.save_fit_results_check, 1, 0)

        self.bootstrap_check = QtWidgets.QCheckBox("Bootstrap")
        self.bootstrap_check.setChecked(False)
        fitting_layout.addWidget(self.bootstrap_check, 1, 1)

        self.le_fitting_check = QtWidgets.QCheckBox("Fit labeling efficiency")
        self.le_fitting_check.setChecked(False)
        self.le_fitting_check.setVisible(False)
        self.le_fitting_check.toggled.connect(self.on_le_fitting_toggled)
        fitting_layout.addWidget(self.le_fitting_check, 1, 2)

        self.fit_button = QtWidgets.QPushButton(
            "Find best fitting stoichiometry"
        )
        self.fit_button.released.connect(self.fit_n_str)
        fitting_layout.addWidget(self.fit_button, 2, 0, 1, 3)

        self.fit_results_display = QtWidgets.QLabel("  ")
        self.fit_results_display.setWordWrap(True)
        fitting_layout.addWidget(self.fit_results_display, 3, 0, 1, 3)

        # SINGLE SIMULATION
        single_sim_box = QtWidgets.QGroupBox("Single simulation")
        single_sim_box.setFixedHeight(200)
        right_column.addWidget(single_sim_box, 1, 0)
        single_sim_layout = QtWidgets.QGridLayout(single_sim_box)

        self.prop_str_input = lib.ScrollableGroupBox(
            "Input proportions of structures (%)", layout="grid"
        )
        single_sim_layout.addWidget(self.prop_str_input, 0, 0, 1, 3)

        self.save_sim_result_check = QtWidgets.QCheckBox(
            "Save positions of\nsimulated molecules"
        )
        self.save_sim_result_check.setChecked(False)
        single_sim_layout.addWidget(self.save_sim_result_check, 1, 0)

        self.roi_button = QtWidgets.QPushButton("Area (\u03bcm\u00b2)")
        self.roi_button.released.connect(self.on_roi_button_clicked)
        single_sim_layout.addWidget(self.roi_button, 1, 1)

        self.run_single_sim_button = QtWidgets.QPushButton(
            "Run single simulation"
        )
        self.run_single_sim_button.released.connect(self.run_single_sim)
        single_sim_layout.addWidget(self.run_single_sim_button, 1, 2)

    def load_structures(self) -> None:
        """Load structures from .yaml file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load structures", filter="*.yaml"
        )
        if path:
            self.structures, self.targets = spinna.load_structures(path)

            self.structures_path = path
            self.exp_data = {}
            self.exp_data_paths = {}
            self.masks = {}
            self.mask_infos = {}
            self.mask_paths = {}
            self.N_structures_fit = {}
            self.n_total = {}
            self.granularity = None
            self.n_sim_fit = None
            self.mixer = None

            self.load_densities_widgets()
            self.load_label_unc_widgets()
            self.load_le_widgets()
            self.load_exp_data_widgets()
            self.load_masks_widgets()
            self.load_single_sim_n_str_widgets()
            self.settings_dialog.update_neighbors_widgets()
            self.nn_plot_settings_dialog.update_neighbors_widgets()
            self.load_structures_button.setStyleSheet(
                "background-color : lightgreen"
            )
            self.fit_results_display.setText("  ")
            self.le_fitting_check.setChecked(False)
            if spinna.check_structures_valid_for_fitting(self.structures):
                self.le_fitting_check.setVisible(True)
            else:
                self.le_fitting_check.setVisible(False)

    def load_target_names(self) -> None:
        """Load all unique names of molecular targets in
        self.structures to attribute self.targets."""
        self.targets = []
        for structure in self.structures:
            for target in structure.targets:
                if target not in self.targets:
                    self.targets.append(target)

    def load_exp_data(self, name: str) -> None:
        """Load experimental data for the specified molecular
        target species."""
        target = name[3:]
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load molecules for {target}", filter="*.hdf5"
        )
        if path:
            # load the data
            locs, info = io.load_locs(path)
            pixelsize = None
            for element in info:
                if "Localize" in element.values():
                    if "Pixelsize" in element and pixelsize is None:
                        pixelsize = element["Pixelsize"]
                if "Render : Pick" in element.values():
                    if "Area (um^2)" in element:
                        area = element["Area (um^2)"]
                        # set the observed density of the molecules
                        idx = self.targets.index(target)
                        self.densities_spins[idx].setValue(len(locs) / area)

            if pixelsize is None:
                pixelsize = 130

            if hasattr(locs, "z"):
                coords = np.stack(
                    (locs.x*pixelsize, locs.y*pixelsize, locs.z)
                ).T
            else:
                coords = np.stack((locs.x*pixelsize, locs.y*pixelsize)).T

            self.exp_data[target] = coords
            self.exp_data_paths[target] = path
            # change the color of the button
            for button in self.load_exp_data_buttons:
                if button.objectName() == name:
                    button.setStyleSheet("background-color : lightgreen")
                    button.setText(f"{target} loaded")
                    break

            # check if all channels have been loaded, if yes, plot NNDs
            # for experimental data only
            if self.check_exp_loaded():
                self.plot_exp_nnds()

    def load_mask(self, name: str) -> None:
        """Load mask for the given molecular target species."""
        target = name[4:]
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load mask for {target}", filter="*.npy"
        )
        if path:
            # load the data
            mask, info = io.load_mask(path)
            self.masks[target] = mask
            self.mask_infos[target] = info
            self.mask_paths[target] = path
            # change the color of the button
            for button in self.load_mask_buttons:
                if button.objectName() == name:
                    button.setStyleSheet("background-color : lightgreen")
                    button.setText(f"{target} loaded")
                    break

    def on_dim_changed(self, index: int) -> None:
        """Update widgets for 2D/3D simulation."""
        if index == 0:  # 2D
            self.densities_box.setTitle(
                "Observed densities (\u03bcm\u207b\u00b2)"
            )
            self.depth_stack.setCurrentIndex(0)
            self.roi_button.setText("Area (\u03bcm\u00b2)")
            self.single_sim_mass = None
        elif index == 1:  # 3D
            if self.mask_den_stack.currentIndex() == 1:
                self.depth_stack.setCurrentIndex(1)
            self.densities_box.setTitle(
                "Observed densities (\u03bcm\u207b\u00b3)"
            )
            self.roi_button.setText("Volume (\u03bcm\u00b3)")
            self.single_sim_mass = None

    def on_depth_button_clicked(self) -> None:
        """Ask the user to input the depth (nm) for a homogenous
        distribution simulation."""
        if self.mask_den_stack.currentIndex() == 0:
            return

        depth, ok = QtWidgets.QInputDialog.getInt(
            self, "", "Input z range for simulation (nm)",
            value=100, min=1, max=100_000, step=10
        )
        if ok:
            self.depth = depth
            self.depth_button.setText(f"Z range: {depth} nm")

    def on_le_fitting_toggled(self, state: bool) -> None:
        """If LE fitting box is checked, freeze LE values, else unfreeze
        them."""

        for le_box in self.le_spins:
            if state:
                le_box.setValue(100.0)
            le_box.setEnabled(not state)

    def load_densities_widgets(self) -> None:
        """Load the widgets for inputting observed densities of each
        target."""
        if not self.structures or not self.targets:
            return

        self.densities_box.remove_all_widgets()
        self.densities_spins = []
        for target in self.targets:
            target_spin = QtWidgets.QDoubleSpinBox(objectName=f"den{target}")
            target_spin.setRange(0.00, 10_000.00)
            target_spin.setDecimals(2)
            target_spin.setSingleStep(0.5)
            target_spin.setValue(100.00)
            self.densities_box.content_layout.addRow(
                QtWidgets.QLabel(f"{target}:"), target_spin
            )
            self.densities_spins.append(target_spin)

    def load_exp_data_widgets(self) -> None:
        """Load the widgets to the load experimental data box."""
        if not self.structures or not self.targets:
            return

        self.load_exp_data_box.remove_all_widgets()
        self.load_exp_data_buttons = []
        for target in self.targets:
            target_button = QtWidgets.QPushButton(
                f"Load {target}", objectName=f"exp{target}"
            )
            target_button.released.connect(
                partial(self.load_exp_data, target_button.objectName())
            )
            self.load_exp_data_box.add_widget(
                target_button,
                self.load_exp_data_box.content_layout.rowCount(), 0,
            )
            self.load_exp_data_buttons.append(target_button)

    def load_masks_widgets(self) -> None:
        """Load the widgets to the load masks box."""
        if not self.structures or not self.targets:
            return

        self.load_mask_box.remove_all_widgets()
        self.load_mask_buttons = []
        for target in self.targets:
            row = self.load_mask_box.content_layout.rowCount()
            target_button = QtWidgets.QPushButton(
                f"Load {target}", objectName=f"mask{target}")
            target_button.released.connect(
                partial(self.load_mask, target_button.objectName())
            )
            self.load_mask_box.add_widget(target_button, row, 0)
            self.load_mask_buttons.append(target_button)

    def load_label_unc_widgets(self) -> None:
        """Load the widgets to the load label uncertainty box."""
        if not self.structures or not self.targets:
            return

        self.label_unc_box.remove_all_widgets()
        self.label_unc_spins = []
        for target in self.targets:
            target_spin = QtWidgets.QDoubleSpinBox(objectName=f"lunc{target}")
            target_spin.setRange(0.01, 100.00)
            target_spin.setDecimals(2)
            target_spin.setSingleStep(0.5)
            target_spin.setValue(5.00)
            self.label_unc_box.content_layout.addRow(
                QtWidgets.QLabel(f"{target}:"), target_spin
            )
            self.label_unc_spins.append(target_spin)

    def load_le_widgets(self) -> None:
        """Load the widgets to the load labeling efficiency box."""
        if not self.structures or not self.targets:
            return

        self.le_box.remove_all_widgets()
        self.le_spins = []
        for target in self.targets:
            target_spin = QtWidgets.QDoubleSpinBox(objectName=f"le{target}")
            target_spin.setRange(0.00, 100.00)
            target_spin.setDecimals(2)
            target_spin.setSingleStep(0.5)
            target_spin.setValue(50.00)
            self.le_box.content_layout.addRow(
                QtWidgets.QLabel(f"{target}:"), target_spin
            )
            self.le_spins.append(target_spin)

    @check_structures_loaded
    def load_single_sim_n_str_widgets(self) -> None:
        """Load the widgets to the input numbers of structures for a
        single simulation box."""
        self.prop_str_input.remove_all_widgets()
        self.prop_str_input_spins = []
        for structure in self.structures:
            title = structure.title
            title_spin = QtWidgets.QDoubleSpinBox(objectName=f"pstr{title}")
            title_spin.setRange(0, 100)
            title_spin.setDecimals(2)
            title_spin.setSingleStep(1)
            title_spin.setValue(0)

            label = QtWidgets.QLabel(f"{title}:")
            column = 2 if len(self.prop_str_input_spins) % 2 else 0
            rowcount = self.prop_str_input.content_layout.rowCount()
            row = rowcount - 1 if column else rowcount

            self.prop_str_input.add_widget(label, row, column)
            self.prop_str_input.add_widget(title_spin, row, column + 1)
            self.prop_str_input_spins.append(title_spin)
        self.prop_str_input_spins[0].setValue(100)  # set the last value to 100

    def set_mask_den_stack(self, name: str):
        """Switches self.mask_den_stack to the mask stack or the
        homogenous distribution stack (observed densities)."""
        if name == "Masks":
            self.mask_den_stack.setCurrentIndex(0)
            self.mask_button.setStyleSheet("background-color : lightgreen")
            self.rect_roi_button.setStyleSheet("background-color : gray")
            self.depth_stack.setCurrentIndex(0)
            self.roi_button.setEnabled(False)
        else:
            self.mask_den_stack.setCurrentIndex(1)
            self.rect_roi_button.setStyleSheet("background-color : lightgreen")
            self.mask_button.setStyleSheet("background-color : gray")
            t = f"Z range: {self.depth} nm" if self.depth else "Z range (nm):"
            self.depth_button.setText(t)
            if self.dim_widget.currentIndex() == 0:
                self.depth_stack.setCurrentIndex(0)
            else:
                self.depth_stack.setCurrentIndex(1)
            self.roi_button.setEnabled(True)

    @check_structures_loaded
    @check_exp_data_loaded
    def generate_search_space(self) -> None:
        """Generate combinations numbers of structures for fitting."""
        n_sim_fit, granularity, save, ok = (
            GenerateSearchSpaceDialog.getParams(self)
        )
        if not ok:
            return
        if save:  # get save path for saving search space
            out_path = self.structures_path.replace(
                ".yaml", "_search_space.csv"
            )
            save, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save numbers of structures", out_path, filter="*.csv"
            )
            if not save:
                return
        self.n_sim_fit = n_sim_fit
        self.granularity = granularity
        self.n_sim_plot_spin.setValue(n_sim_fit)  # save in  NND plot settings

        # extract total number of molecules to simulate per target
        n_total = {}
        for target, le_spin in zip(self.targets, self.le_spins):
            n = len(self.exp_data[target])
            n_total[target] = int(n / (le_spin.value() / 100))
        self.n_total = deepcopy(n_total)

        # generate n structures (stoichiometry) search space
        self.N_structures_fit = spinna.generate_N_structures(
            deepcopy(self.structures), n_total, granularity, save=save
        )

        n = len(self.N_structures_fit[self.structures[0].title])
        estimated_time = self.estimate_fit_time(n)
        self.fit_button.setText(
            f"Find best fitting combination (# tested combinations: {n})"
            f"\nEstimated time (hh:mm:ss): {estimated_time}"
        )

    def estimate_fit_time(self, n: int) -> str:
        """Estimate the time it takes to fit n combinations of numbers
        of structures. Assumes that StructureMixer and other necessary
        parameters are set.

        Parameters
        ----------
        n : int
            Number of combinations of numbers of structures to fit.

        Returns
        -------
        estimated_time : str
            Estimated time in hours, minutes and seconds.
        """
        if n < 1:
            return "--:--:--"

        # prepare n structures for a single fit
        N_structures = np.zeros((1, len(self.structures)), dtype=np.int32)
        for i, structure in enumerate(self.structures):
            N_structures[0, i] = self.N_structures_fit[structure.title][0]

        # set up the mixer
        mixer = self.setup_mixer(mode='fit')
        if mixer is None:
            return "--:--:--"

        # fit a single combination of structures' counts and measure
        # the time
        t0 = time.time()
        spinner = spinna.SPINNA(
            mixer=mixer,
            gt_coords=self.exp_data,
            N_sim=self.n_sim_fit,
        )
        _ = spinner.NN_scorer(N_structures, callback=None)
        dt = time.time() - t0

        # estimate the time for n combinations with a certain number of CPUs;
        # 0.8 is a correction factor for delays in multiprocessing
        # (rough approximation)
        n_cpus = cpu_count()
        dt *= n / (n_cpus * 0.8 * 0.8)

        # convert the time to hours, minutes and seconds
        hours = int(dt // 3600)
        minutes = int((dt - hours * 3600) // 60)
        seconds = int(dt - hours * 3600 - minutes * 60)
        estimated_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return estimated_time

    @check_structures_loaded
    def load_search_space(self) -> None:
        """Load combinations of numbers of structures for fitting."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load numbers of structures", filter="*.csv"
        )
        if path:
            df = pd.read_csv(path)
            # check if the structure titles are the same as the ones loaded
            loaded = df.columns
            titles = [_.title for _ in self.structures]
            if (
                len(loaded) == len(titles)
                and all([t in loaded for t in titles])
            ):
                self.N_structures_fit = {
                    column: df[column].values.astype(np.int32)
                    for column in df.columns
                }
                # update the generate n structures button
                n = len(self.N_structures_fit[self.structures[0].title])
                estimated_time = self.estimate_fit_time(n)
                self.fit_button.setText(
                    "Find best fitting combination (# tested combinations:"
                    f" {n})\nEstimated time (hh:mm:ss): {estimated_time}"
                )
            else:  # display a warning
                message = (
                    "The titles of the previously loaded structures do not"
                    " correspond to the titles of the structures in the file."
                )
                QtWidgets.QMessageBox.warning(self, "Warning", message)
                return

    @check_structures_loaded
    @check_exp_data_loaded
    @check_search_space_loaded
    def fit_n_str(self) -> None:
        """Find the best fitting combination of numbers of structures to
        the experimental data."""
        self.mixer = self.setup_mixer(mode='fit')
        if self.mixer is None:
            return

        # update area/volume in case of rectangluar ROI
        if self.mask_den_stack.currentIndex() == 1:  # rect. ROI
            roi_size = self.mixer.roi_size
            if self.dim_widget.currentIndex() == 0:  # 2D
                self.roi_button.setText(f"Area: {roi_size:.0f} \u03bcm\u00b2")
                # discard the z component if 3D data is loaded
                self.exp_data = {
                    target: coords[:, :2]
                    for target, coords in self.exp_data.items()
                }
            else:  # 3D
                self.roi_button.setText(
                    f"Volume: {roi_size:.0f} (\u03bcm\u00b3)"
                )
            self.single_sim_mass = roi_size

        save = ""
        if self.save_fit_results_check.isChecked():
            out_path = self.structures_path.replace(".yaml", "_fit_scores.csv")
            save, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save fitting scores", out_path, filter="*.csv"
            )
            if not save:
                return

        spinner = spinna.SPINNA(
            mixer=self.mixer,
            gt_coords=self.exp_data,
            N_sim=self.n_sim_fit,
        )
        # number of stoichiometries tested
        N = len(list(self.N_structures_fit.values())[0])
        progress = lib.ProgressDialog(
            "Preparing fit, please wait...", 0, N, self
        )
        progress.set_value(0)
        progress.show()
        self.opt_props, self.current_score = spinner.fit_stoichiometry(
            self.N_structures_fit,
            save=save,
            asynch=self.settings_dialog.asynch_check.isChecked(),
            bootstrap=self.bootstrap_check.isChecked(),
            callback=progress,
        )
        progress.close()
        self.best_score = self.current_score

        # update widgets and plot the best fitting stoichiometry
        self.update_prop_str_input_spins(self.opt_props)
        self.display_proportions(self.opt_props)
        self.save_fit_results()
        self.mixer = self.setup_mixer(mode='single_sim')
        self.sim_and_plot_NND()

    def update_prop_str_input_spins(self, prop_str: np.ndarray) -> None:
        """Update the values of the input proportions of structures
        for a single simulation and adds a button to retrieve these
        results."""
        # extract the mean values if bootstrap was used (then a tuple
        # with mean and std is given)
        if len(np.asarray(prop_str).shape) == 2:
            prop_str = prop_str[0]

        # update spin boxes
        for i, ps in enumerate(prop_str):
            spin = self.prop_str_input_spins[i]
            spin.setValue(ps)

        # delete the restart button in the box if there is any
        button = [
            self.prop_str_input.content_layout.itemAt(i).widget()
            for i in range(self.prop_str_input.content_layout.count())
            if isinstance(
                self.prop_str_input.content_layout.itemAt(i).widget(),
                QtWidgets.QPushButton
            )
        ]
        if button:
            button[0].setParent(None)

        # add a push button to retrieve the results
        button = QtWidgets.QPushButton("Best fitting combination")

        def retrieve_results():
            for i, ps in enumerate(prop_str):
                spin = self.prop_str_input_spins[i]
                spin.setValue(ps)
        button.released.connect(retrieve_results)
        self.prop_str_input.add_widget(
            button, self.prop_str_input.content_layout.rowCount(), 0, 1, 2
        )

    def display_proportions(self, prop_str: np.ndarray) -> None:
        """Display the proportions of the best fitting numbers of
        structures."""
        # different display if bootstrap results are to be displayed
        # or not
        text = ""
        if len(np.asarray(prop_str).shape) == 2:  # bootstrap
            for structure, mean, std in zip(self.structures, *prop_str):
                text += f"{structure.title} - {mean:.2f}% +/- {std:.2f}%, "
        else:  # single fit, no bootstraping
            for structure, prop in zip(self.structures, prop_str):
                text += f"{structure.title} - {prop:.2f}%, "
        text = text[:-2]  # remove last comma and space
        if self.le_fitting_check.isChecked():
            # extract the le values based on the recovered proportions
            le_values = spinna.get_le_from_props(
                self.structures, self.opt_props,
            )
            text = (
                f"LE {self.targets[0]}: {le_values[self.targets[0]]:.1f}%,"
                f" LE {self.targets[1]}: {le_values[self.targets[1]]:.1f}%"
            )  # only display the information about LE result
        self.fit_results_display.setText(text)

    def save_fit_results(self) -> None:
        """Save fit results in .txt with all parameters used."""
        metadata = {}
        metadata["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        metadata["File location of structures"] = self.structures_path
        metadata["Molecular targets"] = ", ".join(self.targets)
        metadata["File location of experimental data"] = (
            ", ".join([self.exp_data_paths[target] for target in self.targets])
        )
        if self.mask_den_stack.currentIndex() == 0:
            metadata["File location of masks"] = (
                ", ".join([self.mask_paths[target] for target in self.targets])
            )
            metadata["Number of simulations"] = self.n_sim_fit
        else:
            metadata["Simulated FOV (um)"] = (
                ", ".join(
                    [str(_/1e3) for _ in self.mixer.roi if _ is not None]
                )
            )
        metadata["Label uncertainties (nm)"] = (
            ", ".join([str(_.value()) for _ in self.label_unc_spins])
        )
        metadata["labeling efficiencies (%)"] = (
            ", ".join([str(_.value()) for _ in self.le_spins])
        )
        metadata["Rotations mode"] = (
            self.settings_dialog.rot_dim_widget.currentText()
        )
        metadata["Dimensionality"] = self.dim_widget.currentText()
        metadata["Number of simulation repeats"] = str(self.n_sim_fit)
        metadata["Parameter search space granularity"] = self.granularity
        metadata["Best fitting proportions (%)"] = (
            self.fit_results_display.text().replace("\n", "")
        )
        metadata[
            "Best fitting score (Kolmogorov-Smirnov 2 sample test statistic)"
        ] = self.best_score

        # relative proportions of structures for each target
        if len(self.targets) > 1:
            for target in self.targets:
                if isinstance(self.opt_props, tuple):
                    rel_props = self.mixer.convert_props_for_target(
                        self.opt_props[0], target, self.n_total,
                    )
                    rel_props_sd = self.mixer.convert_props_for_target(
                        self.opt_props[1], target, self.n_total,
                    )
                else:
                    rel_props = self.mixer.convert_props_for_target(
                        self.opt_props, target, self.n_total,
                    )
                idx_valid = np.where(rel_props != np.inf)[0]
                if isinstance(self.opt_props, tuple):
                    value = ", ".join([
                        f"{self.structures[i].title}: {rel_props[i]:.2f}% +/-"
                        f" {rel_props_sd[i]:.2f}%"
                        for i in idx_valid
                    ])
                else:
                    value = ", ".join([
                        f"{self.structures[i].title}: {rel_props[i]:.2f}%"
                        for i in idx_valid
                    ])
                metadata[f"Relative proportions of {target} in"] = value

        # number of neighbors considered at fitting
        for i, t1 in enumerate(self.mixer.targets):
            for t2 in self.mixer.targets[i:]:
                key = f"{t1}-{t2}"
                metadata[f"Number of neighbors at fitting ({key})"] = (
                    self.mixer.get_neighbor_counts(t1, t2)
                )

        # labeling efficiency fitting
        if self.le_fitting_check.isChecked():
            metadata["Labeling efficiency fitting"] = (
                self.fit_results_display.text()
            )

        # save metadata
        out_path = self.structures_path.replace(".yaml", "_fit_summary.txt")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save fitting summary", out_path, filter="*.txt"
        )
        if path:
            with open(path, "w") as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")

    @check_structures_loaded
    @check_exp_data_loaded
    @check_search_space_loaded
    def compare_models(self) -> None:
        """Open the dialog to compare the goodness of fit for
        different models of structures and runs the test."""
        (
            models, model_names, label_unc, save_scores, ok
        ) = CompareModelsDialog.getParams(self, self.targets)
        if not ok or not models:
            return

        # check if the molecules are to be saved
        savedir = ""
        if save_scores:
            savedir = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Choose folder to save scores",
                os.path.dirname(self.structures_path),
            )
            if not savedir:
                return

        # set up the mixer so that it's easier to extract parameters
        # for model comparison
        base_mixer = self.setup_mixer(mode='fit')
        if base_mixer is None:  # if there's an error, return
            return

        progress = lib.ProgressDialog(
            "Comparing models, please wait...", 0, 1, self
        )
        _, idx, best_label_unc, best_mixer, opt_props = spinna.compare_models(
            models=models,
            exp_data=self.exp_data,
            granularity=self.granularity,
            label_unc=label_unc,
            le=base_mixer.le,
            N_sim=self.n_sim_fit,
            mask_dict=base_mixer.mask_dict,
            width=base_mixer.roi[0],
            height=base_mixer.roi[1],
            depth=base_mixer.roi[2],
            random_rot_mode=base_mixer.random_rot_mode,
            asynch=self.settings_dialog.asynch_check.isChecked(),
            savedir=savedir,
            callback=progress,
        )
        progress.close()

        # load the structures and label_unc and display in the NN plot
        # without overwriting other attributes
        for spin, l_unc in zip(self.label_unc_spins, best_label_unc.values()):
            spin.setValue(l_unc)
        self.structures = best_mixer.structures
        self.mixer = best_mixer
        self.mixer.nn_counts = {
            name: self.nn_plot_settings_dialog.nn_counts[name].value()
            for name in self.nn_plot_settings_dialog.nn_counts.keys()
        }
        if self.mask_den_stack.currentIndex() == 1:  # homogeneus dist.
            roi = self.mixer.roi
            self.single_sim_mass = self.mixer.roi_size
            if roi[2] is None:
                self.roi_button.setText(
                    f"Area: {self.single_sim_mass:.0f} \u03bcm\u00b2"
                )
            else:
                self.roi_button.setText(
                    f"Volume: {self.single_sim_mass:.0f} \u03bcm\u00b3"
                )
        self.load_single_sim_n_str_widgets()
        self.settings_dialog.update_neighbors_widgets()
        self.nn_plot_settings_dialog.update_neighbors_widgets()

        # display the results
        self.update_prop_str_input_spins(opt_props)
        self.sim_and_plot_NND()
        text = (
            f"Best fitting model: {model_names[idx]}, already loaded."
        )
        self.fit_results_display.setText(text)

    def on_roi_button_clicked(self) -> None:
        """Ask the user to input the area/volume to be simulated for a
        single simulation."""
        if self.mask_den_stack.currentIndex() == 1:  # rectangular ROI
            # here mass refers to area/volume
            if self.dim_widget.currentIndex() == 0:  # 2D
                mass, ok = QtWidgets.QInputDialog.getInt(
                    self, "", "Area (\u03bcm\u00b2):", 100, 0, 1_000_000,
                )
                if ok:
                    self.single_sim_mass = mass
                    self.roi_button.setText(f"Area: {mass:.0f} \u03bcm\u00b2")
            else:  # 3D
                mass, ok = QtWidgets.QInputDialog.getInt(
                    self, "", "Volume (\u03bcm\u00b3):", 100, 0, 1_000_000,
                )
                if ok:
                    self.single_sim_mass = mass
                    self.roi_button.setText(
                        f"Volume: {mass:.0f} \u03bcm\u00b3"
                    )

    @check_structures_loaded
    def single_sim_n_total(self) -> int:
        """Find the total number of molecules for a single simulation.
        Either take the number of molecules from experimental data
        (masked) or from input observed densities and the
        area / volume. Note that the total number of molecules is
        adjusted for labeling efficiency, i.e., it is the number of
        observed molecules divided by the LE.

        Returns
        -------
        n_total : int
            Total number of molecules to simulate.
        """
        if self.mask_den_stack.currentIndex() == 0:  # mask
            n_total = int(sum([
                len(self.exp_data[t])
                / self.le_spins[i].value() * 100
                for i, t in enumerate(self.targets)
            ]))
        else:
            tot_densities = [
                self.densities_spins[i].value()
                / self.le_spins[i].value() * 100
                for i in range(len(self.densities_spins))
            ]
            n_total = int(self.single_sim_mass * sum(tot_densities))
        return n_total

    @check_structures_loaded
    def run_single_sim(self) -> None:
        """Run a single simulation and plot NNDs."""
        # check input proportions sum to 100%
        ok, sum_ = self.check_input_props()
        if not ok:
            message = (
                "The input proportions do not sum to 100%.\n"
                f"Sum of proportions: {sum_}%"
            )
            QtWidgets.QMessageBox.warning(self, "Warning", message)
            return

        # check if the molecules are to be saved
        if self.save_sim_result_check.isChecked():
            out_path = self.structures_path.replace(".yaml", "_sim.hdf5")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save positions of simulated molecules",
                out_path,
                filter=".hdf5",
            )
            if not path:
                return
        else:
            path = ""

        # create the mixer instance
        self.mixer = self.setup_mixer(mode='single_sim')
        if self.mixer is None:
            return

        # run the simulation
        self.sim_and_plot_NND()
        if path:  # run a single simulation and save the molecules
            props = np.array([_.value() for _ in self.prop_str_input_spins])
            n_total = self.single_sim_n_total()
            n_str = self.mixer.convert_props_to_counts(props, n_total)
            self.mixer.run_simulation(
                N_structures=n_str,
                path=path,
            )

    @check_structures_loaded
    def sim_and_plot_NND(self) -> None:
        """Simulate and plot nearest neighbor distances of simulated
        molecules and optionally experimental data for comparison.

        Takes the numbers of structures from the single simulation box.
        Number of simulations and plotting parameters are taken from
        the NND plotting box.
        """
        # close figures in the NND plotting box
        for fig in self.nnd_plots:
            plt.close(fig)
        self.nnd_plots = []
        self.current_nnd_idx = 0

        n_sim = self.n_sim_plot_spin.value()
        prop_str = np.array([_.value() for _ in self.prop_str_input_spins])
        n_total = self.single_sim_n_total()
        n_str = self.mixer.convert_props_to_counts(prop_str, n_total)
        plot_params = self.nn_plot_settings_dialog.extract_params()
        show_exp = self.check_exp_loaded()

        # extract NNDs from simulations and experimetal data (if loaded)
        dist_sim = spinna.get_NN_dist_simulated(
            n_str,
            n_sim,
            self.mixer,
            duplicate=True,
        )
        if show_exp:
            # get the simulation's fitting score:
            dist_exp = spinna.get_NN_dist_experimental(
                self.exp_data, mixer=self.mixer, duplicate=True
            )
            self.current_score = spinna.NND_score(dist_exp, dist_sim)

        for i, (t1, t2, _) in enumerate(
            self.mixer.get_neighbor_idx(duplicate=True)
        ):
            # plot simulated distances
            fig, ax = spinna.plot_NN(
                dist=dist_sim[i], mode='plot', show_legend=False,
                binsize=plot_params["binsize_sim"],
                xlim=(plot_params["min_dist"], plot_params["max_dist"]),
                return_fig=True, figsize=(4.947, 3.71), alpha=1.0,
                title=f"{plot_params['title']}{t1} \u2192 {t2}",
                xlabel=plot_params["xlabel"], ylabel=plot_params["ylabel"],
                colors=plot_params["colors"],
            )
            if show_exp:  # plot exp. data if loaded
                exp1 = self.exp_data[t1]
                exp2 = self.exp_data[t2]
                # if 2D simulation and 3D experimental data, project the
                # experimental data to 2D
                if self.dim_widget.currentIndex() == 0 and exp1.shape[1] == 3:
                    exp1 = exp1[:, :2]
                if self.dim_widget.currentIndex() == 0 and exp2.shape[1] == 3:
                    exp2 = exp2[:, :2]
                fig, ax = spinna.plot_NN(
                    data1=exp1, data2=exp2,
                    n_neighbors=plot_params["nn_counts"][f"{t1}-{t2}"].value(),
                    show_legend=False, fig=fig, ax=ax, mode='hist',
                    binsize=plot_params["binsize_exp"],
                    xlim=(plot_params["min_dist"], plot_params["max_dist"]),
                    return_fig=True,
                    title=(
                        f"{plot_params['title']}{t1} \u2192 {t2}\n"
                        f"KS2: {self.current_score:.6f}"
                    ),
                    xlabel=plot_params["xlabel"], ylabel=plot_params["ylabel"],
                    alpha=plot_params["alpha"], colors=plot_params["colors"],
                )

            self.nnd_plots.append(fig)

        # display the first plot
        self.display_current_nnd_plot()

    @check_exp_data_loaded
    def plot_exp_nnds(self) -> None:
        """Plot NNDs of experimental data only, according to the
        chosen plot settings."""
        # close figures in the NND plotting box
        for fig in self.nnd_plots:
            plt.close(fig)
        self.nnd_plots = []
        self.current_nnd_idx = 0

        # plot NNDs
        plot_params = self.nn_plot_settings_dialog.extract_params()
        for i, t1 in enumerate(self.targets):
            for t2 in self.targets[i:]:
                exp1 = self.exp_data[t1]
                exp2 = self.exp_data[t2]
                fig, ax = spinna.plot_NN(
                    data1=exp1, data2=exp2,
                    n_neighbors=plot_params["nn_counts"][f"{t1}-{t2}"].value(),
                    show_legend=False,
                    mode='hist',
                    figsize=(4.947, 3.71),
                    binsize=plot_params["binsize_exp"],
                    xlim=(plot_params["min_dist"], plot_params["max_dist"]),
                    return_fig=True,
                    title=f"{plot_params['title']}{t1} \u2192 {t2}",
                    xlabel=plot_params["xlabel"], ylabel=plot_params["ylabel"],
                    alpha=plot_params["alpha"], colors=plot_params["colors"],
                )

                self.nnd_plots.append(fig)

        # display the first plot
        self.display_current_nnd_plot()

    @check_structures_loaded
    def setup_mixer(self, mode: Literal['fit', 'single_sim'] = 'fit') -> None:
        """Initialize the class used for simulations.

        Parameters
        ----------
        mode : {'fit', 'single_sim'}
            Specifies how to find the numbers of structures to be
            considered.
        """
        # extract label uncertainty and LE
        label_unc = {}
        le = {}
        for target, label_spin, le_spin in zip(
            self.targets,
            self.label_unc_spins,
            self.le_spins,
        ):
            label_unc[target] = label_spin.value()
            le[target] = le_spin.value() / 100

        # extract masks/roi
        if self.mask_den_stack.currentIndex() == 0:  # masks
            width, height, depth = [None, None, None]
            # check that all masks are loaded
            ok = self.check_masks_loaded()
            if ok:
                mask_dict = {"mask": self.masks, "info": self.mask_infos}
            else:
                message = "Please load all masks."
                QtWidgets.QMessageBox.information(self, "Warning", message)
                return

        elif self.mask_den_stack.currentIndex() == 1:  # densities
            if self.dim_widget.currentIndex() == 1 and self.depth is None:
                message = (
                    "Please enter depth for the homogeneously distributed"
                    " simulation. To do this, please click the"
                    ' "Depth (nm)" button above.'
                )
                QtWidgets.QMessageBox.information(self, "Warning", message)
                return
            mask_dict = None
            width, height, depth = self.find_roi(mode=mode)
            if width is None:
                return

        # check dimensionalities, rotations, and optionally # of NNs
        ok, message = self.check_dimensionalities()
        if not ok:
            QtWidgets.QMessageBox.information(self, "Warning", message)
            return
        rot_mode = ["2D", "3D", None][
            self.settings_dialog.rot_dim_widget.currentIndex()
        ]
        if mode == 'fit':
            if self.settings_dialog.auto_nn_check.isChecked():
                nn_counts = 'auto'
            else:
                nn_counts = {
                    name: self.settings_dialog.nn_counts[name].value()
                    for name in self.settings_dialog.nn_counts.keys()
                }
        elif mode == "single_sim":
            nn_counts = {
                name: self.nn_plot_settings_dialog.nn_counts[name].value()
                for name in self.nn_plot_settings_dialog.nn_counts.keys()
            }

        mixer = spinna.StructureMixer(
            structures=self.structures,
            label_unc=label_unc,
            le=le,
            mask_dict=mask_dict,
            width=width, height=height, depth=depth,
            random_rot_mode=rot_mode,
            nn_counts=nn_counts,
        )
        return mixer

    def check_masks_loaded(self) -> bool:
        """Verify if all masks have been loaded."""
        if self.targets is None:
            return False
        for target in self.targets:
            if target not in self.masks.keys():
                return False
        return True

    def check_exp_loaded(self) -> bool:
        """Verify if all exp data have been loaded."""
        if self.targets is None:
            return False
        for target in self.targets:
            if target not in self.exp_data.keys():
                return False
        return True

    def check_dimensionalities(self) -> tuple[bool, str]:
        """Check if masks, loaded experimental data and the requested
        dimensionality of the simulation(s) are consistent.

        Returns
        -------
        ok : bool
            If True, the check was passed and the simulation(s) can be
            conducted.
        message : str
            If ok is False, message will show the warning message to
            the user.
        """
        ok = True
        message = ""

        # number of dimensions chosen by the user
        dim = 2 if self.dim_widget.currentIndex() == 0 else 3

        # check if exp data and/or masks are loaded
        exp_loaded = self.check_exp_loaded()
        masks_loaded = (
            self.check_masks_loaded()
            and self.mask_den_stack.currentIndex() == 0
        )

        # check each loaded target
        for target in self.targets:
            if exp_loaded:
                # only throw an error if 2D data is loaded but 3D
                # simulation is requested
                if dim == 3 and self.exp_data[target].shape[1] == 2:
                    ok = False
                    message = (
                        "3D simulation was requested but a 2D experimental"
                        f" data was loaded for {target}."
                    )
                    return ok, message
            if masks_loaded:
                if self.masks[target].ndim != dim:
                    ok = False
                    message = (
                        f"{dim}D simulation was requested but a "
                        f"{2 if dim == 3 else 3}D mask was loaded for "
                        f"{target}."
                    )
                    return ok, message

        return ok, message

    def check_input_props(self) -> tuple[bool, float]:
        """Check if the input proportions of structures sum to 100%.

        Returns
        -------
        ok : bool
            If True, the proportions sum to 100%.
        sum_ : float
            Sum of the input proportions.
        """
        sum_ = sum([_.value() for _ in self.prop_str_input_spins])
        ok = True if isclose(sum_, 100, abs_tol=1e-3) else False
        return ok, sum_

    def find_roi(
        self,
        mode: Literal['fit', 'single_sim'] = 'fit',
    ) -> tuple[float, float, float]:
        """Find width, height, depth to conduct simulation(s) with
        homogeneous distribution.

        Parameters
        ----------
        mode : {'fit' or 'single_sim'}
            Specifies how to find the numbers of structures to be
            considered.

        Returns
        -------
        result : tuple
            Width, height, depth (all nm).
        """
        assert mode in ['fit', 'single_sim']

        if mode == 'fit':
            return self.find_roi_fit()
        elif mode == "single_sim":
            return self.find_roi_single_sim()

    def find_roi_fit(self) -> tuple[float, float, float]:
        """Find width, height, depth to conduct simulation(s) with
        homogeneous distribution for fitting, based on the input
        densities and the exp. data."""
        target = self.targets[0]
        density = self.densities_spins[0].value()
        # convert density from um^-2 to nm^-2
        density /= 1e6
        if self.dim_widget.currentIndex() == 1:  # 3D data
            density /= 1e3
        # density of the molecule before LE
        le = self.le_spins[0].value() / 100
        tot_density = density / le
        n_mol = self.find_n_mol_from_target(target)

        # obtain depth:
        if self.dim_widget.currentIndex() == 0:  # 2D
            depth = None
        else:
            depth = self.depth

        # find width and height - ROI is a square
        if depth is None:
            width = height = np.sqrt(n_mol / tot_density)
        else:
            width = height = np.sqrt(n_mol / tot_density / depth)
        return width, height, depth

    def find_roi_single_sim(self) -> tuple[float, float, float]:
        """Find width, height, depth to conduct a single simulation with
        homogeneous distribution, based on the user-selected
        area/volume."""
        if self.single_sim_mass is None:
            message = "Please input the area/volume of the ROI first."
            QtWidgets.QMessageBox.information(self, "Warning", message)
            return [None, None, None]

        if self.dim_widget.currentIndex() == 0:  # 2D:
            depth = None
            width = height = np.sqrt(self.single_sim_mass * 1e6)
        else:  # 3D
            depth = self.depth
            width = height = np.sqrt(self.single_sim_mass * 1e9 / depth)

        return width, height, depth

    def find_n_mol_from_target(self, target: str) -> int:
        """Find number of molecules of the given molecular target that
        are to be simulated in fitting.

        Parameters
        ----------
        target : str
            Name of the molecular target.

        Returns
        -------
        n_tar : int
            Number of molecules to be simulated.
        """
        # find targets counts per structure:
        t_counts = spinna.find_target_counts(self.targets, self.structures)
        # extract the row from t_counts that specifies number of the
        # target in question across structures
        idx = self.targets.index(target)
        t_counts = t_counts[idx]
        # find the number of structures to be simulated; note that the
        # number of targets is kept constant across all simulations in
        # the fitting, so we can just extract the values from one such
        # simulation
        N_str = [self.N_structures_fit[_.title][0] for _ in self.structures]
        n_tar = (t_counts * N_str).sum()
        return n_tar

    def display_current_nnd_plot(self) -> None:
        """Display currently indexed NND plot."""
        if not self.nnd_plots:
            return

        if self.nn_plot_settings_dialog.nnd_legend_check.isChecked():
            show_legend = True
        else:
            show_legend = False
        fig = self.nnd_plots[self.current_nnd_idx]
        if show_legend:
            fig.axes[0].legend()
        else:
            if fig.axes[0].legend_:
                fig.axes[0].legend_.remove()
        self.nnd_plot_box.display(fig)

    def on_left_nnd_clicked(self) -> None:
        """Display the previous NND plot."""
        N = len(self.nnd_plots)
        if N == 1:
            self.display_current_nnd_plot()
            return

        if self.current_nnd_idx == 0:
            self.current_nnd_idx = N - 1
        else:
            self.current_nnd_idx -= 1
        self.display_current_nnd_plot()

    def on_right_nnd_clicked(self) -> None:
        """Display the next NND plot."""
        N = len(self.nnd_plots)
        if N == 1:
            self.display_current_nnd_plot()
            return

        if self.current_nnd_idx == N - 1:
            self.current_nnd_idx = 0
        else:
            self.current_nnd_idx += 1
        self.display_current_nnd_plot()

    def save_nnd_plots(self) -> None:
        """Save all the loaded NND plots as .png/.svg files."""
        if self.nnd_plots:
            out_path = self.structures_path.replace(".yaml", "_NND_plots")
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save NND plots", out_path, filter="*.png;;*.svg"
            )
            if not path:
                return
        else:
            return

        i = 0
        for (t1, t2, _) in self.mixer.get_neighbor_idx(duplicate=True):
            fig = self.nnd_plots[i]
            outpath = path.replace(ext[1:], f"_{t1}_{t2}{ext[1:]}")
            fig.savefig(outpath)
            i += 1

    def save_nnd_values(self) -> None:
        """Save all NND values (bin centers and bin heights) as .csv
        files."""
        if self.nnd_plots:
            out_path = self.structures_path.replace(".yaml", "_NND_values")
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save NND values", out_path, filter="*.csv"
            )
            if not path:
                return
        else:
            return

        i = 0
        for (t1, t2, n) in self.mixer.get_neighbor_idx(duplicate=True):
            if not n:
                continue
            # extract plotted line (simulation) and histogram (exp, if
            # present) values from each figure
            fig = self.nnd_plots[i]
            ax = fig.axes[0]
            data_sim = {}
            data_sim["bins_sim"] = ax.lines[0].get_xdata()

            # extract simulation data
            for ll, line in enumerate(ax.lines):
                data_sim[f"NN{ll+1}_values_sim"] = line.get_ydata()
            # save simulation data
            outpath_sim = path.replace(".csv", f"_{t1}_{t2}_sim.csv")
            df = pd.DataFrame(data_sim)
            df.to_csv(outpath_sim, index=False)

            # extract experimental data (if available)
            if ax.patches:
                data_exp = {}
                # extract bin centers from experimental data
                n_bins = int(len(ax.patches) / n)
                mindist = self.nn_plot_settings_dialog.min_dist.value()
                maxdist = self.nn_plot_settings_dialog.max_dist.value()
                bin_edges = np.linspace(mindist, maxdist, n_bins+1)
                bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
                data_exp["bins_exp"] = bin_centers
                patches = [
                    ax.patches[ii:ii+n_bins]
                    for ii in range(0, len(ax.patches), n_bins)
                ]
                for p, patch in enumerate(patches):
                    data_exp[f"NN{p+1}_values_exp"] = [
                        patch_.get_height() for patch_ in patch
                    ]
                # save the values
                outpath_exp = path.replace(".csv", f"_{t1}_{t2}_exp.csv")
                df = pd.DataFrame(data_exp)
                df.to_csv(outpath_exp, index=False)
            i += 1


class Window(QtWidgets.QMainWindow):
    """The main window. Constists of three tabs: mask generation,
    heterostructures design and simulations."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Picasso v{__version__}: SPINNA")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "spinna.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setWindowIcon(icon)
        self.resize(1024, 768)
        self.setMinimumSize(1024, 768)
        self.setMaximumSize(1024, 768)

        # TABS
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.structures_tab = StructuresTab(self)
        self.structures_tab.keyPressEvent = ignore_escape_key
        self.tabs.addTab(self.structures_tab, "Structures")

        self.simulations_tab = SimulationsTab(self)
        self.simulations_tab.keyPressEvent = ignore_escape_key
        self.tabs.addTab(self.simulations_tab, "Simulate")

        self.mask_generator_tab = MaskGeneratorTab(self)
        self.mask_generator_tab.keyPressEvent = ignore_escape_key
        self.tabs.addTab(self.mask_generator_tab, "Mask generation")

        self.tabs.setCurrentIndex(0)

        # menu bar
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("File")
        sounds_menu = file_menu.addMenu("Sound notifications")
        sounds_actiongroup = QtWidgets.QActionGroup(self.menu_bar)
        default_sound_path = lib.get_sound_notification_path()  # last used
        default_sound_name = os.path.basename(str(default_sound_path))
        for sound in lib.get_available_sound_notifications():
            sound_name = os.path.splitext(str(sound))[0].replace("_", " ")
            action = sounds_actiongroup.addAction(
                QtWidgets.QAction(sound_name, sounds_menu, checkable=True)
            )
            action.setObjectName(sound)  # store full name
            if default_sound_name == sound:
                action.setChecked(True)
            sounds_menu.addAction(action)
        sounds_actiongroup.triggered.connect(lib.set_sound_notification)


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
        if p.name == "spinna":
            p.execute()

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
