"""
picasso.gui.rotation
~~~~~~~~~~~~~~~~~~~~

Rotation window classes and functions.
Extension of Picasso: Render to visualize 3D data.
Many functions are copied from gui.render.View to avoid circular
import.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2021-2026 Jungmann Lab, MPI of Biochemistry
"""

import os
import threading
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from .. import io, render, lib, lib_qt, __version__, docs_url
from .render_worker import (
    RenderWorker,
    global_precisions_for,
    subsample_request,
)


DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5


def source_key(view, source: str) -> tuple:
    """Identify what the 3D view would show for the main ``view``: its
    single pick (shape and size included) or its field of view. Equal
    keys mean the loaded content would not change."""
    if source == "fov":
        (y_min, x_min), (y_max, x_max) = view.viewport
        return (
            "fov",
            (float(y_min), float(x_min), float(y_max), float(x_max)),
        )
    return ("pick", repr(view._picks[0]), view._pick_shape, view._pick_size)


N_GROUP_COLORS = render.N_GROUP_COLORS  # 8
SHIFT = 0.1
ZOOM = 9 / 7
# suffix of the color bar saved next to an image exported while
# rendering by property
COLORBAR_SUFFIX = "_colorbar"


class DisplaySettingsRotationDialog(lib.Dialog):
    """Class to change display settings, e.g., display pixel size,
    contrast and blur.

    Very similar to its counterpart in ``picasso.gui.render.py`` but
    some functions were deleted.

    ...

    Attributes
    ----------
    blur_buttongroup : QButtonGroup
        Contains available localization blur methods.
    colormap : QComboBox
        Contains strings with available colormaps (single channel only).
    contrast_slider : DensityContrastSlider(RangeSlider)
        Log-scale two-handle slider mirroring the minimum and maximum
        density spin boxes.
    dynamic_disp_px : QCheckBox
        Tick to automatically adjust to current window size when
        zooming.
    maximum : QDoubleSpinBox
        Defines at which number of localizations per super-resolution
        pixel the maximum color of the colormap should be applied.
    min_blur_width : QDoubleSpinBox
        Contains the minimum blur for each localization (nm).
    minimum : QDoubleSpinBox
        Defines at which number of localizations per super-resolution
        pixel the minimum color of the colormap should be applied.
    disp_px_size : QDoubleSpinBox
        Contains the size of super-resolution pixels in nm.
    scalebar : QSpinBox
        Contains the scale bar's length (nm).
    scalebar_groupbox : QGroupBox
        Group with options for customizing scale bar, tick to display.
    scalebar_text : QCheckBox
        Tick to display scale bar's length (nm).
    _silent_disp_px_update : bool
        True if update display pixel size in background.
    """

    def __init__(self, window):
        super().__init__(window)
        self.first_update = True
        self.window = window
        self.setWindowTitle("Display Settings - Rotation Window")
        self.resize(200, 0)
        self.setModal(False)
        vbox = QtWidgets.QVBoxLayout(self)
        # general
        general_groupbox = QtWidgets.QGroupBox("General")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)
        disp_px_label = QtWidgets.QLabel("Display pixel size (nm):")
        disp_px_label.setToolTip("Size of the pixels in the rendered image.")
        general_grid.addWidget(disp_px_label, 1, 0)
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
        self.dynamic_disp_px.toggled.connect(self.set_dynamic_disp_px)
        general_grid.addWidget(self.dynamic_disp_px, 2, 1)

        # contrast
        contrast_groupbox = QtWidgets.QGroupBox("Contrast")
        vbox.addWidget(contrast_groupbox)
        contrast_grid = QtWidgets.QGridLayout(contrast_groupbox)
        minimum_label = QtWidgets.QLabel("Min. density:")
        minimum_label.setToolTip(
            "Minimum density (localizations per super-resolution pixel)"
            " rendered."
        )
        contrast_grid.addWidget(minimum_label, 0, 0)
        self.minimum = lib.LogDoubleSpinBox()
        self.minimum.setRange(0, 999999)
        self.minimum.setSingleStep(5)
        self.minimum.setValue(0)
        self.minimum.setDecimals(6)
        self.minimum.setKeyboardTracking(False)
        self.minimum.valueChanged.connect(self.render_scene)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtWidgets.QLabel("Max. density:")
        maximum_label.setToolTip(
            "Maximum density (localizations per super-resolution pixel)"
            " rendered."
        )
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = lib.LogDoubleSpinBox()
        self.maximum.setRange(0, 999999)
        self.maximum.setSingleStep(5)
        self.maximum.setValue(100)
        self.maximum.setDecimals(6)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.render_scene)
        contrast_grid.addWidget(self.maximum, 1, 1)
        # log-scale slider mirroring the two spin boxes, for dragging the
        # contrast instead of typing it
        self.contrast_slider = lib.DensityContrastSlider(
            self.minimum,
            self.maximum,
            image=lambda: getattr(
                getattr(self.window, "view_rot", None), "image", None
            ),
        )
        contrast_grid.addWidget(self.contrast_slider, 2, 0, 1, 2)
        c_label = QtWidgets.QLabel("Colormap:")
        c_label.setToolTip("Colormap used to render localizations.")
        contrast_grid.addWidget(c_label, 3, 0)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(plt.colormaps())
        contrast_grid.addWidget(self.colormap, 3, 1)
        self.colormap.currentIndexChanged.connect(self.render_scene)

        # blur
        blur_groupbox = QtWidgets.QGroupBox("Blur")
        blur_grid = QtWidgets.QGridLayout(blur_groupbox)
        self.blur_buttongroup = QtWidgets.QButtonGroup()
        points_button = QtWidgets.QRadioButton("None")
        points_button.setToolTip(
            "No blur applied; each localization is rendered as a point."
        )
        self.blur_buttongroup.addButton(points_button)
        smooth_button = QtWidgets.QRadioButton("One-pixel blur")
        smooth_button.setToolTip(
            "Each localization is Gaussian blurred with a \u03c3 of one "
            "rendered pixel."
        )
        self.blur_buttongroup.addButton(smooth_button)
        convolve_button = QtWidgets.QRadioButton(
            "Global localization precision"
        )
        convolve_button.setToolTip(
            "Each localization is Gaussian blurred with a \u03c3 equal to\n"
            "the median localization precision of the dataset."
        )
        self.blur_buttongroup.addButton(convolve_button)
        gaussian_button = QtWidgets.QRadioButton(
            "Individual localization precision"
        )
        gaussian_button.setToolTip(
            "Each localization is Gaussian blurred with a \u03c3 equal to\n"
            "its individual localization precision."
        )
        self.blur_buttongroup.addButton(gaussian_button)
        gaussian_iso_button = QtWidgets.QRadioButton(
            "Individual localization precision, iso"
        )
        gaussian_iso_button.setToolTip(
            "Each localization is Gaussian blurred with a \u03c3 equal to\n"
            "its individual localization precision, isotropic in xy."
        )
        self.blur_buttongroup.addButton(gaussian_iso_button)
        # the same buttons as the main window's dialog, in the same
        # order (the windows sync by button id)
        quadtree_button = QtWidgets.QRadioButton(
            "Adaptive histogram (quad-tree)"
        )
        quadtree_button.setToolTip(
            "Histogram whose bins split while they hold more than the\n"
            "leaf capacity, so every bin has about the same signal-to-noise\n"
            "ratio (Baddeley, Cannell & Soeller, 2010). In 3D the tree is\n"
            "built from the projected localizations for every orientation."
        )
        self.blur_buttongroup.addButton(quadtree_button)
        triangulation_button = QtWidgets.QRadioButton("Jittered triangulation")
        triangulation_button.setToolTip(
            "Delaunay triangles drawn with an intensity inverse to their\n"
            "area, averaged over triangulations of the localizations\n"
            "jittered by their mean distance to their neighbors, so the\n"
            "blur follows the local sampling (Baddeley, Cannell & Soeller,\n"
            "2010). In 3D the projected localizations are triangulated for\n"
            "every orientation. Costly: rendered up to a number of loaded\n"
            "localizations, the histogram is shown above this number instead."
        )
        self.blur_buttongroup.addButton(triangulation_button)

        blur_grid.addWidget(points_button, 0, 0, 1, 2)
        blur_grid.addWidget(smooth_button, 1, 0, 1, 2)
        blur_grid.addWidget(convolve_button, 2, 0, 1, 2)
        blur_grid.addWidget(gaussian_button, 3, 0, 1, 2)
        blur_grid.addWidget(gaussian_iso_button, 4, 0, 1, 2)
        blur_grid.addWidget(quadtree_button, 5, 0, 1, 2)
        blur_grid.addWidget(triangulation_button, 6, 0, 1, 2)
        convolve_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.render_scene_nocache)
        # the minimum blur, shown only for the Gaussian methods that
        # use it (a container, so the grid keeps no empty row otherwise)
        self.min_blur_widgets = QtWidgets.QWidget()
        min_blur_grid = QtWidgets.QGridLayout(self.min_blur_widgets)
        min_blur_grid.setContentsMargins(0, 0, 0, 0)
        min_blur_label = QtWidgets.QLabel("Min. Blur (nm):")
        min_blur_label.setToolTip(
            "Minimum blur applied to all localizations in nm."
        )
        min_blur_grid.addWidget(min_blur_label, 0, 0, 1, 1)
        self.min_blur_width = QtWidgets.QDoubleSpinBox()
        self.min_blur_width.setRange(0, 999999)
        self.min_blur_width.setSingleStep(0.1)
        self.min_blur_width.setValue(0)
        self.min_blur_width.setDecimals(1)
        self.min_blur_width.setKeyboardTracking(False)
        self.min_blur_width.valueChanged.connect(self.render_scene_nocache)
        min_blur_grid.addWidget(self.min_blur_width, 0, 1, 1, 1)
        blur_grid.addWidget(self.min_blur_widgets, 7, 0, 1, 2)
        # the quad-tree's settings, shown only while it is selected
        self.quadtree_widgets = QtWidgets.QWidget()
        quadtree_grid = QtWidgets.QGridLayout(self.quadtree_widgets)
        quadtree_grid.setContentsMargins(0, 0, 0, 0)
        capacity_label = QtWidgets.QLabel("Leaf capacity:")
        capacity_label.setToolTip(
            "Largest number of localizations a bin of the adaptive\n"
            "histogram may hold before it is split into four; every bin\n"
            "then has about the same signal-to-noise ratio,\n"
            "sqrt(capacity / 2) on average."
        )
        quadtree_grid.addWidget(capacity_label, 0, 0, 1, 1)
        self.quadtree_capacity = QtWidgets.QSpinBox()
        self.quadtree_capacity.setRange(1, 100000)
        self.quadtree_capacity.setValue(lib.RENDER_QUADTREE_CAPACITY_DEFAULT)
        self.quadtree_capacity.setKeyboardTracking(False)
        self.quadtree_capacity.setToolTip(capacity_label.toolTip())
        quadtree_grid.addWidget(self.quadtree_capacity, 0, 1, 1, 1)
        self.quadtree_snr = QtWidgets.QLabel()
        quadtree_grid.addWidget(self.quadtree_snr, 1, 0, 1, 2)
        blur_grid.addWidget(self.quadtree_widgets, 8, 0, 1, 2)
        # the triangulation's settings (synced from the main window)
        self.triangulation_widgets = QtWidgets.QWidget()
        triangulation_grid = QtWidgets.QGridLayout(self.triangulation_widgets)
        triangulation_grid.setContentsMargins(0, 0, 0, 0)
        passes_label = QtWidgets.QLabel("Passes:")
        passes_label.setToolTip(
            "Jittered triangulations averaged (the original paper uses 25\n"
            "to 50); 1 shows a single jittered triangulation, more take"
            " longer."
        )
        triangulation_grid.addWidget(passes_label, 0, 0, 1, 1)
        self.triangulation_passes = QtWidgets.QSpinBox()
        self.triangulation_passes.setRange(1, 500)
        self.triangulation_passes.setValue(
            lib.RENDER_TRIANGULATION_PASSES_DEFAULT
        )
        self.triangulation_passes.setKeyboardTracking(False)
        self.triangulation_passes.setToolTip(passes_label.toolTip())
        triangulation_grid.addWidget(self.triangulation_passes, 0, 1, 1, 1)
        jitter_label = QtWidgets.QLabel("Jitter:")
        jitter_label.setToolTip(
            "Width of the random displacement of every localization in\n"
            "units of its mean distance to its neighbors: 1 (the paper's\n"
            "choice) blurs to the local sampling limit, 0.5 keeps more\n"
            "detail for known periodic structures."
        )
        triangulation_grid.addWidget(jitter_label, 1, 0, 1, 1)
        self.triangulation_jitter = QtWidgets.QDoubleSpinBox()
        self.triangulation_jitter.setRange(0.0, 10.0)
        self.triangulation_jitter.setSingleStep(0.1)
        self.triangulation_jitter.setDecimals(2)
        self.triangulation_jitter.setValue(
            lib.RENDER_TRIANGULATION_JITTER_DEFAULT
        )
        self.triangulation_jitter.setKeyboardTracking(False)
        self.triangulation_jitter.setToolTip(jitter_label.toolTip())
        triangulation_grid.addWidget(self.triangulation_jitter, 1, 1, 1, 1)
        max_locs_label = QtWidgets.QLabel("Max. localizations:")
        max_locs_label.setToolTip(
            "Triangulating is computationally expensive.\n"
            "With more localizations loaded than this (every one is\n"
            "projected and triangulated for each orientation) the\n"
            "histogram is rendered instead."
        )
        triangulation_grid.addWidget(max_locs_label, 2, 0, 1, 1)
        self.triangulation_max_locs = QtWidgets.QSpinBox()
        self.triangulation_max_locs.setRange(1000, 100_000_000)
        self.triangulation_max_locs.setSingleStep(10000)
        self.triangulation_max_locs.setValue(
            lib.RENDER_TRIANGULATION_MAX_LOCS_DEFAULT
        )
        self.triangulation_max_locs.setKeyboardTracking(False)
        self.triangulation_max_locs.setToolTip(max_locs_label.toolTip())
        triangulation_grid.addWidget(self.triangulation_max_locs, 2, 1, 1, 1)
        self.triangulation_note = QtWidgets.QLabel()
        self.triangulation_note.setWordWrap(True)
        self.triangulation_note.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        triangulation_grid.addWidget(self.triangulation_note, 3, 0, 1, 2)
        blur_grid.addWidget(self.triangulation_widgets, 9, 0, 1, 2)
        for widget in (
            self.triangulation_passes,
            self.triangulation_jitter,
            self.triangulation_max_locs,
        ):
            widget.valueChanged.connect(self.render_scene_nocache)
        self.triangulation_widgets.setVisible(False)
        self.quadtree_capacity.valueChanged.connect(self._update_quadtree_snr)
        self.quadtree_capacity.valueChanged.connect(self.render_scene_nocache)
        self._update_quadtree_snr()
        self.blur_buttongroup.buttonToggled.connect(self._toggle_blur_widgets)
        self.quadtree_widgets.setVisible(False)

        vbox.addWidget(blur_groupbox)
        self.blur_methods = {
            points_button: None,
            smooth_button: "smooth",
            convolve_button: "convolve",
            gaussian_button: "gaussian",
            gaussian_iso_button: "gaussian_iso",
            quadtree_button: "quadtree",
            triangulation_button: "triangulation",
        }

        # scalebar
        self.scalebar_groupbox = QtWidgets.QGroupBox("Scale bar")
        self.scalebar_groupbox.setCheckable(True)
        self.scalebar_groupbox.setChecked(False)
        self.scalebar_groupbox.toggled.connect(self.render_scene)
        vbox.addWidget(self.scalebar_groupbox)
        scalebar_grid = QtWidgets.QGridLayout(self.scalebar_groupbox)
        scalebar_length_label = QtWidgets.QLabel("Scale bar length (nm):")
        scalebar_length_label.setToolTip("Set the length of the scale bar.")
        scalebar_grid.addWidget(scalebar_length_label, 0, 0)
        self.scalebar = QtWidgets.QSpinBox()
        self.scalebar.setRange(1, 100000)
        self.scalebar.setValue(500)
        self.scalebar.setKeyboardTracking(False)
        self.scalebar.valueChanged.connect(self.render_scene)
        self.scalebar.valueChanged.connect(self._uncheck_optimal_scalebar)
        scalebar_grid.addWidget(self.scalebar, 0, 1)
        self.scalebar_text = QtWidgets.QCheckBox("Print scale bar length")
        self.scalebar_text.setToolTip("Display the length of the scale bar?")
        self.scalebar_text.stateChanged.connect(self.render_scene)
        scalebar_grid.addWidget(self.scalebar_text, 1, 0)
        self.optimal_scalebar_check = QtWidgets.QCheckBox("Automatic length")
        self.optimal_scalebar_check.setToolTip(
            "Set the scale bar length to approximately 1/8 of the current "
            "viewport width."
        )
        self.optimal_scalebar_check.setChecked(True)
        self.optimal_scalebar_check.stateChanged.connect(
            self.window.view_rot.set_optimal_scalebar
        )
        scalebar_grid.addWidget(self.optimal_scalebar_check, 1, 1)

        self._silent_disp_px_update = False

    def blur_method(self) -> str | None:
        """The selected blur method (``render`` name)."""
        return self.blur_methods[self.blur_buttongroup.checkedButton()]

    def _update_quadtree_snr(self, *args) -> None:
        """Show the signal-to-noise ratio the leaf capacity implies
        (Baddeley et al. 2010, sqrt(N / 2))."""
        snr = np.sqrt(self.quadtree_capacity.value() / 2.0)
        self.quadtree_snr.setText(f"Mean SNR per bin \u2248 {snr:.1f}")

    def set_triangulation_note(self, n_loaded: int | None) -> None:
        """Say why the histogram was rendered instead of the
        triangulation (too many localizations loaded), or clear the
        note (None)."""
        if n_loaded is None:
            self.triangulation_note.setText("")
        else:
            self.triangulation_note.setText(
                f"{n_loaded:,} localizations exceed the limit; the "
                "histogram is shown. Open a smaller region or raise the "
                "limit."
            )

    def _toggle_blur_widgets(self, *args) -> None:
        """Show only the settings the selected blur method uses: the
        minimum blur for the Gaussian methods, the leaf capacity for
        the quad-tree. The dialog then grows or shrinks by exactly the
        change of its content (``_follow_content_height``), so it keeps
        the size the user gave it and shows no empty space."""
        method = self.blur_methods[self.blur_buttongroup.checkedButton()]
        content = self
        if getattr(self, "_content_height", None) is None:
            self._content_height = content.sizeHint().height()
        self.min_blur_widgets.setVisible(
            method in ("gaussian", "gaussian_iso", "convolve")
        )
        self.quadtree_widgets.setVisible(method == "quadtree")
        self.triangulation_widgets.setVisible(method == "triangulation")
        # the layouts settle in the event loop; measure afterwards
        QtCore.QTimer.singleShot(0, self._follow_content_height)

    def _follow_content_height(self) -> None:
        """Resize the dialog by the change of its content's height since
        the last measurement (see ``_toggle_blur_widgets``)."""
        content = self
        height = content.sizeHint().height()
        previous = self._content_height
        self._content_height = height
        if self.isVisible() and height != previous:
            self.resize(
                self.width(), max(self.height() + height - previous, 1)
            )

    def on_disp_px_changed(self, value: float) -> None:
        """Set new display pixel size, update contrast and update scene
        in the main window."""
        contrast_factor = (value / self._disp_px_size) ** 2
        self._disp_px_size = value
        self.silent_minimum_update(contrast_factor * self.minimum.value())
        self.silent_maximum_update(contrast_factor * self.maximum.value())
        if not self._silent_disp_px_update:
            self.dynamic_disp_px.setChecked(False)
            self.window.view_rot.update_scene()

    def set_disp_px_silently(self, disp_px_size: float) -> None:
        """Change the value of display pixel size in the background."""
        self._silent_disp_px_update = True
        self.disp_px_size.setValue(disp_px_size)
        self._silent_disp_px_update = False

    def silent_minimum_update(self, value: float) -> None:
        """Change the value of self.minimum in the background."""
        self.minimum.blockSignals(True)
        self.minimum.setValue(value)
        self.minimum.blockSignals(False)
        # the spin box' signals are blocked, so the slider is moved here
        self.contrast_slider.sync()

    def silent_maximum_update(self, value: float) -> None:
        """Change the value of self.maximum in the background."""
        self.maximum.blockSignals(True)
        self.maximum.setValue(value)
        self.maximum.blockSignals(False)
        self.contrast_slider.sync()

    def _uncheck_optimal_scalebar(self, *args) -> None:
        """Uncheck the automatic scale bar checkbox when the user
        manually changes the scale bar length."""
        if self.optimal_scalebar_check.isChecked():
            self.optimal_scalebar_check.blockSignals(True)
            self.optimal_scalebar_check.setChecked(False)
            self.optimal_scalebar_check.blockSignals(False)

    def render_scene(self, *args, **kwargs):
        """Update scene in the rotation window."""
        self.window.view_rot.update_scene(use_cache=True)

    def render_scene_nocache(self, *args, **kwargs):
        """Update scene in the rotation window without using cache."""
        self.window.view_rot.update_scene(use_cache=False)

    def set_dynamic_disp_px(self, state: bool) -> None:
        """Update scene if dynamic display pixel size is checked."""
        if state:
            self.window.view_rot.update_scene()


class AnimationDialog(lib.Dialog):
    """Dialog to prepare 3D animations.

    Position rows live in a scrollable area so the sequence length is
    unbounded. Each call to ``add_position`` instantiates a new row of
    widgets; ``delete_position`` destroys the last one.

    Attributes
    ----------
    add : QPushButton
        Click to add the current view to the animation sequence.
    build : QPushButton
        Click to create an animation.
    current_pos : QLabel
        Shows rotation angles around x, y and z axes in degrees.
    delete : QPushButton
        Click to delete the last saved position.
    fps : QSpinBox
        Contains frames per second used in the animation.
    positions : list of dict
        Contains all positions in the animation sequence. Each holds
        the rotation (scipy Rotation, key ``"R"``), the accumulated
        rotation vector (key ``"rotvec"``, radians, scipy convention,
        keeps full turns), the rotation accumulated since the previous
        position (key ``"segment_rotvec"``, radians, scipy convention;
        may encode turns beyond 180 degrees) and the viewport (key
        ``"viewport"``).
    rows : list of dict
        One entry per saved position, holding the row's widgets:
        ``p_label``, ``angle_label``, ``show_btn``, ``d_label``,
        ``duration``. ``d_label`` and ``duration`` are ``None`` for
        the first row (no transition into it).
    rows_layout : QGridLayout
        Layout inside the scroll area that holds the position rows.
    stay : QPushButton
        Click to copy the exact same position (so that locs do not
        move).
    rot_speed : QDoubleSpinBox
        Contains the default rotation speed calculated when adding a
        position with different angles.
    transition : QComboBox
        Timing of the motion between positions, one of
        ``TRANSITIONS`` (see ``render.build_animation``).
    window : QMainWindow
        Instance of the rotation window.
    """

    DOCS_URL = docs_url("render.html#build-an-animation")

    # display name -> (``render.build_animation`` transition, tooltip)
    TRANSITIONS = {
        "Stop at each position": (
            "ease",
            "Accelerates and decelerates between every two positions, "
            "coming to rest at each of them.",
        ),
        "Smooth": (
            "smooth",
            "Starts and ends at rest and passes through the positions "
            "without abrupt changes of direction or speed.",
        ),
        "Constant speed": (
            "linear",
            "Moves at a constant speed between every two positions; "
            "direction and speed change abruptly at the positions.",
        ),
    }

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setWindowIcon(icon)

        self.window = window
        self.setWindowTitle("Build an animation")
        self.setModal(False)
        # as narrow as the controls allow
        self.resize(0, 420)

        self.positions = []
        self.rows = []

        main_layout = QtWidgets.QVBoxLayout(self)

        # Header: current position
        header = QtWidgets.QHBoxLayout()
        header.addWidget(lib.HelpButton(self.DOCS_URL))
        cp_label = QtWidgets.QLabel("Current position:")
        cp_label.setToolTip(
            "Current rotation in x, y, z (deg). The angles keep track "
            "of full turns, e.g. 720 encodes two full rotations."
        )
        header.addWidget(cp_label)
        angx = np.round(self.window.view_rot.angx * 180 / np.pi, 1)
        angy = np.round(self.window.view_rot.angy * 180 / np.pi, 1)
        angz = np.round(self.window.view_rot.angz * 180 / np.pi, 1)
        self.current_pos = QtWidgets.QLabel(
            "{}, {}, {}".format(angx, angy, angz)
        )
        header.addWidget(self.current_pos)
        header.addStretch(1)
        main_layout.addLayout(header)

        # Scroll area holding the position rows
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        rows_container = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QGridLayout(rows_container)
        self.rows_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # Reserve the duration columns' widths up front so the "Show position"
        # button keeps both its width and its horizontal position when those
        # widgets first appear (with the second position). Slack from the
        # dialog width is absorbed by the angle-label column instead of
        # leaving an empty trailing column on the right.
        template_d_label = QtWidgets.QLabel("Duration (s): ")
        template_duration = QtWidgets.QDoubleSpinBox()
        template_duration.setRange(0.01, 10)
        template_duration.setDecimals(2)
        self.rows_layout.setColumnMinimumWidth(
            3, template_d_label.sizeHint().width()
        )
        self.rows_layout.setColumnMinimumWidth(
            4, template_duration.sizeHint().width()
        )
        template_d_label.deleteLater()
        template_duration.deleteLater()
        self.rows_layout.setColumnStretch(1, 1)
        scroll_area.setWidget(rows_container)
        main_layout.addWidget(scroll_area, 1)

        # Editing the sequence, right below the positions
        sequence_row = QtWidgets.QHBoxLayout()
        self.add = QtWidgets.QPushButton("Add this position")
        self.add.setToolTip(
            "Add the current rotation/view to the animation sequence."
        )
        self.add.clicked.connect(self.add_position)
        self.stay = QtWidgets.QPushButton("Stay in the position")
        self.stay.setToolTip("Add the current position again (no movement).")
        self.stay.clicked.connect(partial(self.add_position, True))
        self.delete = QtWidgets.QPushButton("Remove last position")
        self.delete.setToolTip(
            "Remove the last position from the animation sequence."
        )
        self.delete.clicked.connect(self.delete_position)
        for button in (self.add, self.stay, self.delete):
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            sequence_row.addWidget(button)
        sequence_row.addStretch(1)
        main_layout.addLayout(sequence_row)

        # Settings: two columns of label/field pairs
        settings = QtWidgets.QGridLayout()
        settings.setColumnMinimumWidth(2, 12)  # gap between the pairs
        settings.setColumnStretch(5, 1)  # keep the fields compact

        fps_label = QtWidgets.QLabel("FPS:")
        fps_label.setToolTip("Frames per second used in the animation.")
        settings.addWidget(fps_label, 0, 0)
        self.fps = QtWidgets.QSpinBox()
        self.fps.setValue(30)
        self.fps.setRange(1, 60)
        settings.addWidget(self.fps, 0, 1)

        rs_label = QtWidgets.QLabel("Rotation speed (deg/s):")
        rs_label.setToolTip(
            "Average speed of rotation between positions in the "
            "animation, used to suggest the durations."
        )
        settings.addWidget(rs_label, 1, 0)
        self.rot_speed = QtWidgets.QDoubleSpinBox()
        self.rot_speed.setValue(90)
        self.rot_speed.setDecimals(1)
        self.rot_speed.setRange(0.1, 1000)
        settings.addWidget(self.rot_speed, 1, 1)

        transition_label = QtWidgets.QLabel("Transition:")
        transition_label.setToolTip(
            "How the motion is timed between the positions."
        )
        settings.addWidget(transition_label, 0, 3)
        self.transition = QtWidgets.QComboBox()
        for name, (_, tooltip) in self.TRANSITIONS.items():
            self.transition.addItem(name)
            self.transition.setItemData(
                self.transition.count() - 1,
                tooltip,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        self.transition.setCurrentText("Stop at each position")
        settings.addWidget(self.transition, 0, 4)

        # output resolution, independent of the window's size; follows
        # the window until edited by hand (see ``showEvent``)
        size_label = QtWidgets.QLabel("Resolution (px):")
        size_label.setToolTip(
            "Width and height of the video in pixels (rounded up to a "
            "multiple of 16 for the encoder). Defaults to the window's "
            "size; the frames are rendered at this resolution, whatever "
            "the window's."
        )
        settings.addWidget(size_label, 1, 3)
        size_row = QtWidgets.QHBoxLayout()
        self.width_px = QtWidgets.QSpinBox()
        self.width_px.setRange(16, 8192)
        self.height_px = QtWidgets.QSpinBox()
        self.height_px.setRange(16, 8192)
        self._size_edited = False
        for box in (self.width_px, self.height_px):
            box.setValue(512)
            box.valueChanged.connect(self._mark_size_edited)
        size_row.addWidget(self.width_px)
        size_row.addWidget(QtWidgets.QLabel("x"))
        size_row.addWidget(self.height_px)
        size_row.addStretch(1)
        settings.addLayout(size_row, 1, 4)

        main_layout.addLayout(settings)

        build_row = QtWidgets.QHBoxLayout()
        build_row.addStretch(1)
        self.build = QtWidgets.QPushButton("Build animation")
        self.build.setToolTip("Create the animation as an .mp4 file.")
        self.build.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.build.clicked.connect(self.build_animation)
        build_row.addWidget(self.build)
        main_layout.addLayout(build_row)

        # the build in progress: its thread, worker and the cancel flag
        self._build_thread = None
        self._build_worker = None
        self._build_cancel = None
        self._build_progress = None

    def _mark_size_edited(self) -> None:
        self._size_edited = True

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Default the output resolution to the window's size until the
        user sets one."""
        if not self._size_edited:
            view = self.window.view_rot
            for box, value in (
                (self.width_px, view.width()),
                (self.height_px, view.height()),
            ):
                box.blockSignals(True)
                box.setValue(max(16, value))
                box.blockSignals(False)
        super().showEvent(event)

    def add_position(self, freeze: bool = False) -> None:
        """Add a new position to the animation sequence.

        Parameters
        ----------
        freeze : bool, optional
            True when the new position is the same as the last one,
            i.e., when self.stay is clicked.
        """
        view = self.window.view_rot
        # rotation accumulated since the last position (keeps full
        # turns, e.g. 720 deg); zero when freezing in place
        segment_rotvec = (
            np.zeros(3) if freeze else view.rotation_since_anchor()
        )

        # check that the viewport or the rotation have changed
        if not freeze and self.positions:
            last = self.positions[-1]
            same_rotation = (
                view.rotation * last["R"].inv()
            ).magnitude() < 1e-9
            no_turns = np.linalg.norm(segment_rotvec) < 1e-9
            same_viewport = view.viewport == last["viewport"]
            if same_rotation and no_turns and same_viewport:
                return

        # add a new position to the attribute
        self.positions.append(
            {
                "R": view.rotation,
                "rotvec": view._rotvec.copy(),
                "segment_rotvec": segment_rotvec,
                "viewport": view.viewport,
            }
        )
        # track the next segment's rotation from this position onwards
        view.reset_rotation_anchor()

        index = len(self.positions) - 1
        grid_row = index

        # build the row's widgets
        p_label = QtWidgets.QLabel(f"- Position {index + 1}: ")
        p_label.setToolTip(
            f"Rotation angles in x, y, z (deg) for position {index + 1}."
        )
        self.rows_layout.addWidget(p_label, grid_row, 0)

        angx = np.round(view.angx * 180 / np.pi, 1)
        angy = np.round(view.angy * 180 / np.pi, 1)
        angz = np.round(view.angz * 180 / np.pi, 1)
        angle_label = QtWidgets.QLabel("{}, {}, {}".format(angx, angy, angz))
        self.rows_layout.addWidget(angle_label, grid_row, 1)

        show_btn = QtWidgets.QPushButton("Show position")
        show_btn.setToolTip("Move to this position.")
        show_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        show_btn.clicked.connect(partial(self.retrieve_position, index))
        self.rows_layout.addWidget(show_btn, grid_row, 2)

        d_label = None
        duration = None
        if index > 0:
            d_label = QtWidgets.QLabel("Duration (s): ")
            d_label.setToolTip("Duration of the transition to this position.")
            self.rows_layout.addWidget(d_label, grid_row, 3)
            duration = QtWidgets.QDoubleSpinBox()
            duration.setRange(0.01, 10)
            duration.setValue(1)
            duration.setDecimals(2)
            self.rows_layout.addWidget(duration, grid_row, 4)

            # calculate recommended duration from the total rotation
            # (including full turns) at the requested rotation speed
            total_angle = float(np.linalg.norm(segment_rotvec))
            if not freeze and total_angle > 1e-9:
                rot_speed = self.rot_speed.value() * np.pi / 180
                duration.setValue(total_angle / rot_speed)

        self.rows.append(
            {
                "p_label": p_label,
                "angle_label": angle_label,
                "show_btn": show_btn,
                "d_label": d_label,
                "duration": duration,
            }
        )

    def delete_position(self) -> None:
        """Delete the last position from the animation sequence."""
        if not self.rows:
            return
        row = self.rows.pop()
        for w in (
            row["p_label"],
            row["angle_label"],
            row["show_btn"],
            row["d_label"],
            row["duration"],
        ):
            if w is not None:
                self.rows_layout.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
        del self.positions[-1]

    def retrieve_position(self, i: int) -> None:
        """Move the view to the specified position.

        Parameters
        ----------
        i : int
            Index of the position to be displayed.
        """
        if i <= len(self.positions) - 1:
            position = self.positions[i]
            self.window.view_rot.set_rotation(
                position["R"], rotvec=position["rotvec"]
            )
            self.window.view_rot.update_scene(viewport=position["viewport"])

    def build_animation(self) -> None:
        """Create an animation as an .mp4 file using the positions from
        the animation sequence."""
        if len(self.positions) < 2:
            message = (
                "At least two positions are required to build an "
                "animation. You can add them by clicking 'Add this "
                "position'."
            )
            QtWidgets.QMessageBox.warning(
                self, "Not enough positions", message
            )
            return

        durations = [row["duration"].value() for row in self.rows[1:]]

        # get save file name
        out_path = (
            os.path.splitext(self.window.view_rot.paths[0])[0] + "_video.mp4"
        )
        path, ext = lib.get_save_filename_ext_dialog(
            self, "Save animation", out_path, filter="*.mp4", check_ext=".yaml"
        )
        if not path:
            return
        if self._build_thread is not None:
            QtWidgets.QMessageBox.information(
                self, "Build an animation", "An animation is being built."
            )
            return
        disp_dlg = self.window.display_settings_dlg
        data_dlg = self.window.window.dataset_dialog
        pixelsize = self.window.window.view.pixelsize
        view = self.window.view_rot
        locs, infos = view._prepare_locs_for_rendering()
        if view._pan_z:
            locs = view._apply_pan_z(locs)
        n_frames = int(self.fps.value() * sum(durations))
        adjust_display_pixel = disp_dlg.dynamic_disp_px.isChecked()
        intensities = self.window.window.view.read_relative_intensities()
        positions = [(p["R"], p["viewport"]) for p in self.positions]
        segment_rotations = [p["segment_rotvec"] for p in self.positions[1:]]
        # the frames are rendered at the output resolution: the display
        # pixel size follows from the last position's field of view
        width, height = self.width_px.value(), self.height_px.value()
        last_viewport = positions[-1][1]
        disp_px_size = pixelsize * render.viewport_width(last_viewport) / width
        kwargs = dict(
            positions=positions,
            durations=durations,
            segment_rotations=segment_rotations,
            transition=self.TRANSITIONS[self.transition.currentText()][0],
            disp_px_size=float(disp_px_size),
            image_size=(width, height),
            blur_method=disp_dlg.blur_method(),
            min_blur_width=disp_dlg.min_blur_width.value() / pixelsize,
            quadtree_capacity=disp_dlg.quadtree_capacity.value(),
            triangulation_passes=disp_dlg.triangulation_passes.value(),
            triangulation_jitter=disp_dlg.triangulation_jitter.value(),
            contrast=(disp_dlg.minimum.value(), disp_dlg.maximum.value()),
            invert_colors=data_dlg.wbackground.isChecked(),
            single_channel_colormap=disp_dlg.colormap.currentText(),
            colors=self.window.window.view.read_colors(),
            relative_intensities=intensities,
            fps=self.fps.value(),
            adjust_pixel_size=adjust_display_pixel,
        )
        self._start_build(path, locs, infos, kwargs, n_frames)

    def _start_build(self, path, locs, infos, kwargs, n_frames) -> None:
        """Render and encode the frames on a worker thread, with a
        cancellable, non-modal progress dialog; the windows stay
        usable meanwhile."""
        self._build_cancel = threading.Event()
        progress = QtWidgets.QProgressDialog(
            "Rendering animation frames", "Cancel", 0, n_frames, self
        )
        progress.setWindowTitle("Build an animation")
        progress.setModal(False)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.canceled.connect(self._build_cancel.set)
        progress.show()
        self._build_progress = progress
        self.build.setEnabled(False)

        worker = AnimationBuilder(
            path, locs, infos, kwargs, cancel=self._build_cancel.is_set
        )
        thread = QtCore.QThread()  # unparented, see ViewRotation
        worker.moveToThread(thread)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(self._build_finished)
        self._build_thread = thread
        self._build_worker = worker
        thread.start()
        # the work runs inside the thread's event loop (queued signal),
        # not from ``started``: a build that finished before the loop
        # began would swallow the ``quit()`` and ``wait()`` forever
        worker.start()

    def _build_finished(self, completed: bool, error: str) -> None:
        """Wrap up a build on the GUI thread."""
        thread = self._build_thread
        self._build_thread = None
        self._build_worker = None
        if thread is not None:
            thread.quit()
            thread.wait()
        if self._build_progress is not None:
            self._build_progress.close()
            self._build_progress = None
        self.build.setEnabled(True)
        if error:
            QtWidgets.QMessageBox.warning(
                self, "Build an animation", f"Building failed:\n\n{error}"
            )

    def stop_build(self) -> None:
        """Cancel a build in progress and wait for its thread (a live
        QThread must never be destroyed)."""
        if self._build_cancel is not None:
            self._build_cancel.set()
        thread = self._build_thread
        if thread is not None:
            thread.quit()
            thread.wait()
            self._build_thread = None
            self._build_worker = None
        if self._build_progress is not None:
            self._build_progress.close()
            self._build_progress = None
        self.build.setEnabled(True)


class AnimationBuilder(QtCore.QObject):
    """Runs ``render.build_animation`` on a worker thread.

    Attributes
    ----------
    progress : QtCore.pyqtSignal
        The frame number just rendered (drives the progress dialog).
    finished : QtCore.pyqtSignal
        Emitted with ``(completed, error)``: ``completed`` is False when
        cancelled, ``error`` the message of a failure (empty otherwise).
    """

    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(bool, str)
    _start = QtCore.pyqtSignal()

    def __init__(self, path, locs, infos, kwargs, cancel):
        super().__init__()
        self._path = path
        self._locs = locs
        self._infos = infos
        self._kwargs = kwargs
        self._cancel = cancel
        # auto connection: queued once this object lives on its thread
        self._start.connect(self.run)

    def start(self) -> None:
        """Begin the build on the thread this object was moved to."""
        self._start.emit()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            completed = render.build_animation(
                self._path,
                self._locs,
                self._infos,
                progress_callback=self.progress.emit,
                cancel=self._cancel,
                **self._kwargs,
            )
        except Exception as error:  # reported in the GUI, never lost
            self.finished.emit(False, f"{type(error).__name__}: {error}")
            return
        self.finished.emit(bool(completed), "")


class RotateByAngleDialog(lib.Dialog):
    """Choose rotations angles.

    ...

    Attributes
    ----------
    angx, angy, angz : QtWidgets.QDoubleSpinBox
        Store the rotation angles input by the user.
    frame : QtWidgets.QComboBox
        Selects whether the angles rotate around the data's own axes
        ("Localizations", the axes shown by the axes icon) or the fixed
        screen/camera axes ("World").
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter rotation angles")
        layout = QtWidgets.QFormLayout(self)
        self.angx = QtWidgets.QDoubleSpinBox()
        self.angx.setValue(0)
        self.angx.setRange(-999999, 999999)
        self.angx.setSingleStep(1)
        layout.addRow("Angle x (deg)", self.angx)
        self.angy = QtWidgets.QDoubleSpinBox()
        self.angy.setValue(0)
        self.angy.setRange(-999999, 999999)
        self.angy.setSingleStep(1)
        layout.addRow("Angle y (deg)", self.angy)
        self.angz = QtWidgets.QDoubleSpinBox()
        self.angz.setValue(0)
        self.angz.setRange(-999999, 999999)
        self.angz.setSingleStep(1)
        layout.addRow("Angle z (deg)", self.angz)

        # Coordinate system the angles are applied around. "Localizations"
        # (default) rotates around the data's own axes; "World" rotates
        # around the fixed screen/camera axes.
        self.frame = QtWidgets.QComboBox()
        self.frame.addItems(["Localizations", "World"])
        self.frame.setToolTip(
            "Localizations: rotate around the data's own axes (the axes "
            "shown by the axes icon).\nWorld: rotate around the fixed "
            "screen/camera axes."
        )
        layout.addRow("Rotate around", self.frame)

        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        layout.addRow(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(parent: QtWidgets.QMainWindow | None = None) -> tuple:
        """Create the dialog and return the requested rot. angles and
        the frame ("object" or "world") to rotate around."""
        dialog = RotateByAngleDialog(parent)
        result = dialog.exec()
        frame = "world" if dialog.frame.currentText() == "World" else "object"
        return (
            dialog.angx.value(),
            dialog.angy.value(),
            dialog.angz.value(),
            frame,
            result == QtWidgets.QDialog.DialogCode.Accepted,
        )


class ViewRotation(QtWidgets.QLabel):
    """Display rotated super-resolution datasets.

    Most functions were taken from ``picasso.gui.render.View``.

    ...

    Attributes
    ----------
    angx, angy, angz : float
        Accumulated rotation around the x, y and z axes (radians) in
        the codebase's angle convention (read-only properties; backed
        by ``self._rotvec``). They keep track of full turns, e.g. read
        720 degrees after two full rotations.
    block_x, block_y, block_z : bool
        True if rotate only around x, y, or z axis respectively.
    group_color : lib.IntArray1D
        Important for single channel data with group info (picked or
        clustered locs); contains an integer index for each loc
        defining its color.
    infos : list of dicts
        Contains a dictionary with metadata for each channel.
    locs : list of pd.DataFrame
        Contains a data frame with localizations for each channel.
    _mode : str
        Defines current mode (zoom, pick or measure); important for
        mouseEvents.
    _pan : bool
        Indicates if image is currently panned.
    _pan_z : float
        Z component of the current view target (camera pixels). The
        viewport stores only the X/Y components; this completes it so
        that screen-space panning works at any rotation.
    pan_start_x, pan_start_y : float
        X and Y coordinates of panning's starting position.
    pixmap : QPixmap
        Pixmap currently displayed.
    _points : list
        Coordinates of the points of the measurement set currently
        being drawn (connected by lines with live distances).
    _point_sets : list
        Finalized measurement sets, each a list of point coordinates.
        Kept separate so lines and distances are only drawn within a
        set, not across sets.
    _measure_following : bool
        True while the cursor is followed live and left clicks extend
        the current set. Set to False (frozen) by the first right click
        so a new set can be started; a further right click then deletes
        the last finalized set.
    _measure_cursor : tuple or None
        Live cursor position (camera pixels) in Measure mode while the
        cursor is followed; None otherwise.
    qimage : QImage
        Current image of rendered locs, picks and other drawings.
    _R : scipy.spatial.transform.Rotation
        Source of truth for the current rotation (quaternion-backed).
    _rotvec : np.ndarray
        Accumulated rotation vector (radians, scipy convention) - the
        sum of all applied rotation vectors (a path integral), so full
        turns are preserved, e.g. two full rotations around z read
        (0, 0, 4 pi). For rotations around a single axis it represents
        ``_R`` exactly; in general the exact orientation is ``_R``.
        ``angx/angy/angz`` are derived from this.
    _anchor_rotvec : np.ndarray
        Rotation vector (radians, scipy convention, world frame)
        accumulated since the last anchor (set when an animation
        position is added or shown) - like ``_rotvec``, a path
        integral, so rotations beyond 180 degrees and full turns are
        preserved in animation segments.
    _last_mouse_x, _last_mouse_y : int
        Previous mouse position (Qt coords) during a trackball drag.
    viewport : tuple or None
        Defines current field of view; None until localizations are
        loaded.
    window : QMainWindow
        Instance of the rotation window.
    async_rendering : bool
        Class attribute: full renders run on a worker thread
        (``render_worker.RenderWorker``, latest request wins) and the
        result is shown when it lands, so a drag never blocks the GUI.
        False renders synchronously on the GUI thread (tests, and the
        fallback that stays available).
    """

    async_rendering = True

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.image = None  # raw image of the last full render (cache)
        # asynchronous rendering: completed renders are matched against
        # the newest request id (see ``_on_render_finished``); the
        # worker thread starts with the first asynchronous request
        self._render_request_id = 0
        self._current_request_interactive = False
        self._render_worker = None
        self._render_thread = None
        # interactive requests (drags) render a subsample; the refine
        # timer follows up with a full-quality render on idle
        self._refine_timer = QtCore.QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(150)
        self._refine_timer.timeout.connect(self._refine_render)
        self._R = Rotation.identity()
        self._rotvec = np.zeros(3)
        self._anchor_rotvec = np.zeros(3)
        self._pan_z = 0.0
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self.locs = []
        self.infos = []
        self.paths = []
        self.viewport = None
        # what this window shows, copied from the main window when it is
        # opened (see ``_sync_from_main_window``): the single pick, or
        # the main window's field of view (``_source == "fov"``, pick
        # attributes None). ``_source_key`` identifies the loaded
        # content so opening the view again with nothing changed only
        # raises the window.
        self.pick = None
        self.pick_shape = None
        self.pick_size = None
        self._source = "pick"
        self._source_key = None
        self._fov_viewport = None  # the loaded field of view (fov mode)
        self.group_color = []
        self.x_render_state = False
        self.x_locs = []
        self._z_shift = 0.0
        self._size_hint = (512, 512)
        self._mode = "Rotate"
        self._points = []  # active measurement set being drawn
        self._point_sets = []  # finalized measurement sets
        self._measure_following = True  # cursor followed live while True
        self._measure_cursor = None  # live cursor position in Measure mode
        self._pan = False
        self.block_x = False
        self.block_y = False
        self.block_z = False
        # navigation gestures: the zoom rectangle being dragged (start
        # and current position in widget pixels), and snapped rotation
        # while S is held (rotation accumulates until a step is due)
        self._zoom_rect = None
        # the rectangle's outline, the same widget the 2D window uses
        self.rubberband = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle, self
        )
        self.rubberband.setStyleSheet("selection-background-color: white")
        self._triple_click = lib_qt.TripleClick()
        self._snap = False
        self._snap_accum = np.zeros(3)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        # track the cursor without a pressed button for live measuring
        self.setMouseTracking(True)

    # --- rotation state --- #
    # The codebase's angle convention (see ``render.rotation_matrix``)
    # flips the sign of the x rotation relative to scipy's right-hand
    # rule, i.e., rotation_matrix(a, 0, 0).as_rotvec() == (-a, 0, 0).
    # The angx/angy/angz properties report the accumulated rotation
    # vector in that convention.
    #
    # ``_rotvec`` accumulates the applied rotation vectors (a path
    # integral) rather than re-deriving angles from ``_R``. An absolute
    # orientation cannot count full turns - spinning a tilted structure
    # by 360 degrees returns to the same orientation - so this is the
    # only way the displayed angles can keep track of the number of
    # turns (e.g. read 720 after two full rotations). For rotations
    # around a single axis the accumulated values match the orientation
    # exactly; for composed rotations around different axes they are a
    # record of the applied rotations and the exact orientation is
    # given by ``_R``.
    @property
    def angx(self) -> float:
        return -float(self._rotvec[0])

    @property
    def angy(self) -> float:
        return float(self._rotvec[1])

    @property
    def angz(self) -> float:
        return float(self._rotvec[2])

    @property
    def rotation(self) -> Rotation:
        """Current rotation of the localizations."""
        return self._R

    def set_rotation(
        self,
        R: Rotation,
        rotvec: np.ndarray | None = None,
    ) -> None:
        """Set the absolute rotation and reset the per-segment anchor.

        Parameters
        ----------
        R : scipy.spatial.transform.Rotation
            New rotation.
        rotvec : np.ndarray, optional
            Accumulated rotation vector to restore as the displayed
            angles (radians, scipy convention), e.g. saved earlier
            from ``self._rotvec``; keeps full-turn information. If
            None, the shortest rotation vector of ``R`` is used.
        """
        self._R = R
        if rotvec is None:
            self._rotvec = R.as_rotvec()
        else:
            self._rotvec = np.asarray(rotvec, dtype=float).copy()
        self.reset_rotation_anchor()

    def reset_rotation_anchor(self) -> None:
        """Start tracking the rotation accumulated from the current
        orientation (used for animation segments)."""
        self._anchor_rotvec = np.zeros(3)

    def rotation_since_anchor(self) -> np.ndarray:
        """Rotation vector (radians, scipy convention, world frame)
        accumulated since the last anchor; its magnitude may exceed pi
        and encodes full turns, e.g. 4 pi for two full turns."""
        return self._anchor_rotvec.copy()

    def apply_rotation(
        self,
        rotvec: np.ndarray,
        frame: str = "world",
    ) -> None:
        """Compose a rotation onto the current one and track turns.

        Parameters
        ----------
        rotvec : np.ndarray
            Rotation vector (radians, scipy convention). Its magnitude
            may exceed pi; full turns are preserved in the displayed
            angles and in the per-segment tracking.
        frame : {"world", "object"}, optional
            Frame in which ``rotvec`` is given. "world": the fixed
            world/screen frame (rotation composed as ``delta * R``).
            "object": the data's own (rotated) frame, i.e., rotation
            around the data axes shown by the axes icon (composed as
            ``R * delta``). Default is "world".
        """
        rotvec = np.asarray(rotvec, dtype=float)
        magnitude = float(np.linalg.norm(rotvec))
        if magnitude < 1e-12:
            return
        # accumulate the displayed angles (path integral; an absolute
        # orientation cannot count full turns, so the applied rotation
        # vectors are summed instead)
        self._rotvec = self._rotvec + rotvec
        # accumulate the per-segment rotation the same way, but always
        # in the world frame, since that is how animation segments are
        # interpreted (``render._animation_sequence``). An object-frame
        # delta d applied to R equals the world-frame delta R.apply(d):
        # from_rotvec(R d) * R == R * from_rotvec(d).
        delta = Rotation.from_rotvec(rotvec)
        if frame == "object":
            self._anchor_rotvec = self._anchor_rotvec + self._R.apply(rotvec)
            self._R = self._R * delta
        else:
            self._anchor_rotvec = self._anchor_rotvec + rotvec
            self._R = delta * self._R

    def load_saved_rotation(self, info: dict) -> None:
        """Restore the rotation saved by
        ``RotationWindow.save_locs_rotated``.

        New files store a quaternion together with the accumulated
        rotation angles (keeping full turns); legacy files store Euler
        angles (angx, angy, angz, radians, codebase convention).
        """
        angles = (
            info.get("angx", 0.0),
            info.get("angy", 0.0),
            info.get("angz", 0.0),
        )
        if "Quaternion (x, y, z, w)" in info:
            R = Rotation.from_quat(info["Quaternion (x, y, z, w)"])
            # angx is stored in the codebase convention (x negated
            # relative to scipy's rotation vector)
            rotvec = np.array([-angles[0], angles[1], angles[2]])
            self.set_rotation(R, rotvec=rotvec)
        else:  # legacy: Euler angles
            self.set_rotation(render.rotation_matrix(*angles))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(*self._size_hint)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.update_scene()

    def _sync_from_main_window(self, w):
        # get pixelsize
        self.pixelsize = w.view.pixelsize
        # update blur and colormap
        b = w.display_settings_dlg.blur_buttongroup.checkedId()
        color = w.display_settings_dlg.colormap.currentText()
        self.window.display_settings_dlg.blur_buttongroup.button(b).setChecked(
            True
        )
        self.window.display_settings_dlg.colormap.setCurrentText(color)
        self.window.display_settings_dlg.quadtree_capacity.setValue(
            w.display_settings_dlg.quadtree_capacity.value()
        )
        for name in (
            "triangulation_passes",
            "triangulation_jitter",
            "triangulation_max_locs",
        ):
            getattr(self.window.display_settings_dlg, name).setValue(
                getattr(w.display_settings_dlg, name).value()
            )

        # remove measurement points
        self._points = []
        self._point_sets = []
        self._measure_following = True
        self._measure_cursor = None

        if self._source == "fov":
            # the main window's field of view, rotated about its center
            self.pick = None
            self.pick_shape = None
            self.pick_size = None
            (y_min, x_min), (y_max, x_max) = w.view.viewport
            self.viewport = [(y_min, x_min), (y_max, x_max)]
        else:
            # save the pick information
            self.pick = w.view._picks[0]
            self.pick_shape = w.view._pick_shape
            self.pick_size = w.view._pick_size
            self.viewport = self.fit_in_view_rotated(get_viewport=True)
        self._source_key = source_key(w.view, self._source)

        # update dataset_dialog for multichannel data and paths
        self.window.dataset_dialog = w.dataset_dialog
        self.paths = w.view.locs_paths

        # copy render property state from the main window
        self.x_render_state = w.view.x_render_state
        if self.x_render_state:
            ds = w.display_settings_dlg
            self.x_property = ds.parameter.currentText()
            self.x_n_colors = ds.color_step.value()
            self.x_min_val = ds.minimum_render.value()
            self.x_max_val = ds.maximum_render.value()
            self.x_colormap = ds.colormap_prop.currentText()
        else:
            self.x_locs = []

    def _collect_picked_locs(self, w):
        n_channels = len(self.paths)
        self.locs = []
        self.infos = []
        for i in range(n_channels):
            # only one pick, take the first element
            temp = w.view.picked_locs(i, add_group=False)[0]
            self._append_channel(temp, w.view.infos[i])

    def _collect_fov_locs(self, w):
        """The localizations of each channel inside this window's
        viewport (the main window's field of view when opened, shifted
        with the arrow keys since), copied like a pick's."""
        n_channels = len(self.paths)
        self.locs = []
        self.infos = []
        (y_min, x_min), (y_max, x_max) = self.viewport
        self._fov_viewport = [(y_min, x_min), (y_max, x_max)]  # for fit
        for i in range(n_channels):
            locs = w.view.locs[i]
            # the viewport pyramid's selection where it exists (a slight
            # superset around the edges is harmless here), else a scan
            idx = w.view._viewport_indices(i, self.viewport)
            if idx is None:
                x = locs["x"].to_numpy()
                y = locs["y"].to_numpy()
                idx = np.flatnonzero(
                    (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
                )
            temp = locs.iloc[idx].reset_index(drop=True)
            self._append_channel(temp, w.view.infos[i])

    def _append_channel(self, temp, info):
        """Store one channel's copied localizations with z and lpz in
        camera pixels, as the rotation math expects them."""
        temp["z"] /= self.pixelsize
        if "lpz" in temp.columns:
            temp["lpz"] /= self.pixelsize
        self.locs.append(temp)
        self.infos.append(info)

    def _apply_render_property_split(self):
        if not (self.x_render_state and len(self.locs) == 1):
            return
        if self.x_property in self.locs[0].columns:
            min_value = self.x_min_val
            max_value = self.x_max_val
            # z / lpz are rescaled to camera pixels here (see
            # _collect_picked_locs), and z is additionally mean-shifted (see
            # load_locs), whereas the property limits come from the main
            # window in nm. Map the limits through the same transform so the
            # color binning matches the main view.
            if self.x_property in ("z", "lpz"):
                min_value = min_value / self.pixelsize
                max_value = max_value / self.pixelsize
            if self.x_property == "z":
                min_value -= self._z_shift
                max_value -= self._z_shift
            self.x_locs = render.split_locs_by_property(
                self.locs[0],
                property_name=self.x_property,
                n_colors=self.x_n_colors,
                min_value=min_value,
                max_value=max_value,
            )
        else:
            self.x_render_state = False
            self.x_locs = []

    def load_locs(self, update_window=False, source=None):
        """Load localizations from the main window: those of its single
        pick, or those in its field of view.

        Called when updating the rotation window from there or when
        shifting the pick / the viewport from the rotation window.

        Parameters
        ----------
        update_window : bool, optional
            If True, load attributes, such as blur method, from the
            main window.
        source : {"pick", "fov"}, optional
            What to show: the main window's single pick or its current
            field of view. If None, the source shown last is kept
            (a pick before the window was first opened).
        """
        w = self.window.window  # main window
        if source is not None:
            self._source = source
        if update_window:
            self._sync_from_main_window(w)

        if self._source == "fov":
            self._collect_fov_locs(w)
        else:
            self._collect_picked_locs(w)

        # shift z positions of locs so that the middle of the dataset is
        # at z = 0
        all_locs_z = np.concatenate([_["z"].to_numpy() for _ in self.locs])
        z_shift = all_locs_z.mean()
        # store shift (in camera pixels) so that render-by-property limits,
        # which are given in the main window's z units (nm), can be mapped
        # onto the transformed z used here (see _apply_render_property_split)
        self._z_shift = z_shift
        for i in range(len(self.locs)):
            self.locs[i]["z"] -= z_shift

        # assign self.group_color if single channel and group info present
        if len(self.locs) == 1 and "group" in self.locs[0].columns:
            self.group_color = render.get_group_color(self.locs[0])

        self._apply_render_property_split()

    def render_scene(
        self,
        viewport: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None,
        autoscale: bool = False,
        use_cache: bool = False,
        cache: bool = True,
    ) -> QtGui.QImage:
        """Render QImage of localizations.

        Parameters
        ----------
        viewport : tuple, optional
            Viewport to be rendered ``((y_min, x_min), (y_max, x_max))``.
            If None, takes current viewport.
        autoscale : bool, optional
            If True, optimally adjust contrast.
        use_cache : bool, optional
            If True, use cached image.
        cache : bool, optional
            If True, cache rendered image.

        Returns
        -------
        qimage : QImage
            Shows rendered locs; 8 bit, scaled.
        """
        request = self._build_render_request(
            viewport=viewport, autoscale=autoscale, use_cache=use_cache
        )
        qimage, _, contrast_limits, raw_image = render.render_scene(**request)
        self._adopt_render_result(contrast_limits, raw_image, cache=cache)
        return qimage

    def _build_render_request(
        self,
        viewport: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None,
        autoscale: bool = False,
        use_cache: bool = False,
    ) -> dict:
        """Snapshot everything ``render.render_scene`` needs, on the GUI
        thread (dialog reads and locs preparation are not thread safe).
        The current rotation goes along as ``ang``, so the request
        renders the same on either thread and either backend."""
        # get disp px size, blur method, etc
        kwargs = self.get_render_kwargs(viewport=viewport)
        locs, infos = self._prepare_locs_for_rendering()
        if self._pan_z:
            locs = self._apply_pan_z(locs)
        self._apply_triangulation_limit(kwargs, locs)
        vmin = self.window.display_settings_dlg.minimum.value()
        vmax = self.window.display_settings_dlg.maximum.value()
        cmap = self.window.display_settings_dlg.colormap.currentText()
        contrast = None if autoscale else (vmin, vmax)
        raw_image = self.image if use_cache else None
        intensities = self.window.window.view.read_relative_intensities()
        return dict(
            locs=locs,
            info=infos,
            global_precision=self._global_precisions(
                locs, kwargs["blur_method"]
            ),
            **kwargs,
            ang=self._R,
            contrast=contrast,
            invert_colors=self.window.dataset_dialog.wbackground.isChecked(),
            single_channel_colormap=cmap,
            colors=self.window.window.view.read_colors(),
            relative_intensities=intensities,
            raw_image_cache=raw_image,
            return_contrast_limits=True,
            return_raw_image=True,
        )

    def _apply_triangulation_limit(self, kwargs: dict, locs) -> None:
        """Render the histogram instead of the jittered triangulation
        when more localizations than the dialog's limit are loaded
        (every one is projected and triangulated per frame), and say
        so in the display settings; ``kwargs`` is updated in place."""
        dialog = self.window.display_settings_dlg
        if kwargs.get("blur_method") != "triangulation":
            dialog.set_triangulation_note(None)
            return
        channels = [locs] if isinstance(locs, pd.DataFrame) else locs
        n = sum(len(channel) for channel in channels)
        if n > dialog.triangulation_max_locs.value():
            kwargs["blur_method"] = None
            dialog.set_triangulation_note(n)
        else:
            dialog.set_triangulation_note(None)

    def _global_precisions(self, locs, blur_method: str | None):
        """``render_scene``'s ``global_precision`` for the prepared
        ``locs``: per frame, its channel's median precision, computed
        once per loaded channel (see ``render_worker.global_precision_of``);
        None unless the blur method is 'convolve'."""
        if blur_method != "convolve":
            return None
        cache = getattr(self, "_precision_cache", None)
        if cache is None:
            cache = self._precision_cache = {}
        return global_precisions_for(
            locs,
            self.locs,
            cache,
            checked=lambda i: (
                len(self.locs) == 1
                or self.window.dataset_dialog.checks[i].isChecked()
            ),
        )

    def _adopt_render_result(
        self,
        contrast_limits: tuple[float, float],
        raw_image: np.ndarray,
        cache: bool = True,
    ) -> None:
        """Keep a completed render's raw image as the cache of the
        contrast redraws and show its contrast limits in the display
        settings dialog."""
        if cache:
            self.image = raw_image
        vmin, vmax = contrast_limits
        self.window.display_settings_dlg.silent_minimum_update(vmin)
        self.window.display_settings_dlg.silent_maximum_update(vmax)

    def update_scene(
        self,
        viewport: tuple[float, float, float, float] | None = None,
        autoscale: bool = False,
        use_cache: bool = False,
        interactive: bool = False,
        synchronous: bool = False,
    ) -> None:
        """Update the view of rendered localizations.

        Parameters
        ----------
        viewport : tuple, optional
            Viewport to be rendered ``((y_min, x_min), (y_max, x_max))``.
            If None self.viewport is taken.
        autoscale : bool, optional
            True if optimally adjust contrast.
        use_cache : bool, optional
            True if the rendered scene should be taken from cache.
        interactive : bool, optional
            True during a drag: the render may be a subsampled preview,
            followed by a full-quality render once the drag pauses.
        synchronous : bool, optional
            True to render on the GUI thread and return with the new
            image on screen (exports), whatever ``async_rendering``.
        """
        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene(
                viewport,
                autoscale=autoscale,
                use_cache=use_cache,
                interactive=interactive,
                synchronous=synchronous,
            )

        # update current position in the animation dialog
        angx = np.round(self.angx * 180 / np.pi, 1)
        angy = np.round(self.angy * 180 / np.pi, 1)
        angz = np.round(self.angz * 180 / np.pi, 1)
        self.window.animation_dialog.current_pos.setText(
            f"{angx}, {angy}, {angz}"
        )

    def draw_scene(
        self,
        viewport: tuple[float, float, float, float],
        autoscale: bool = False,
        use_cache: bool = False,
        interactive: bool = False,
        synchronous: bool = False,
    ) -> None:
        """Render localizations in the given viewport and draws legend,
        rotation, etc.

        Parameters
        ----------
        viewport : tuple
            Viewport defining the rendered FOV ``((y_min, x_min),
            (y_max, x_max))``.
        autoscale : bool, optional
            True if contrast should be optimally adjusted.
        use_cache : bool, optional
            True if the rendered scene should be taken from cache.
        interactive, synchronous : bool, optional
            See ``update_scene``.
        """
        # make sure viewport has the same shape as the main window
        self.viewport = self.adjust_viewport_to_view(viewport)
        if not use_cache:
            self.set_optimal_scalebar(silent=True)
        if use_cache or synchronous or not self.async_rendering:
            # cache redraws (contrast, colormap, the live measuring
            # cross) are cheap and stay synchronous for instant feedback
            qimage = self.render_scene(
                autoscale=autoscale, use_cache=use_cache
            )
            self._complete_scene(qimage)
        else:
            # full renders run on the worker thread; the last frame
            # stays on screen until the new one lands
            self._submit_async_render(
                autoscale=autoscale, interactive=interactive
            )

    def _complete_scene(self, qimage: QtGui.QImage) -> None:
        """Second half of ``draw_scene``: scale the rendered frame to
        the window, draw the overlays and show it. Runs on the GUI
        thread, directly for synchronous renders or from
        ``_on_render_finished``."""
        # scale image's size to the window
        self.qimage = qimage.scaled(
            self.width(),
            self.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        )
        # draw scalebar, legend, rotation and measuring points
        self.qimage = self.draw_scalebar(self.qimage)
        self.qimage = self.draw_legend(self.qimage)
        self.qimage = self.draw_rotation(self.qimage)
        self.qimage = self.draw_rotation_angles(self.qimage)
        self.qimage = self.draw_points(self.qimage)

        # convert to pixmap
        self.pixmap = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.pixmap)

    # --- asynchronous rendering --- #
    def _ensure_render_worker(self) -> RenderWorker:
        """The worker and its thread, started on first use so a window
        that never shows 3D data never runs a thread."""
        if self._render_thread is None:
            self._render_worker = RenderWorker()
            # deliberately unparented: a parented QThread would be
            # destroyed by Qt while still running whenever the window
            # is torn down outside closeEvent, which is a hard abort;
            # the Python reference owns it and stop_render_worker()
            # ends it
            self._render_thread = QtCore.QThread()
            self._render_worker.moveToThread(self._render_thread)
            self._render_worker.finished.connect(self._on_render_finished)
            self._render_thread.start()
        return self._render_worker

    def _submit_async_render(
        self, autoscale: bool = False, interactive: bool = False
    ) -> None:
        """Post the newest render request to the worker (latest wins).

        Interactive requests (drags) render a strided subsample with
        compensated contrast, by the main view's ``interaction_subsample``
        rule, and arm the refine timer, which follows up with a
        full-quality render once the drag pauses.
        """
        request = self._build_render_request(autoscale=autoscale)
        if interactive:
            interactive = subsample_request(
                request, self._interaction_subsample_target
            )
        self._render_request_id += 1
        self._current_request_interactive = interactive
        self._ensure_render_worker().submit(
            self._render_request_id, request, self.viewport
        )
        if interactive:
            self._refine_timer.start()
        else:
            self._refine_timer.stop()

    def _refine_render(self) -> None:
        """Follow the last interactive preview with a full render."""
        if len(self.locs) and self.async_rendering:
            self._submit_async_render()

    def _interaction_subsample_target(self, population: int = 0) -> int:
        """Preview target for a request of ``population`` loaded
        localizations, sized by what is in view.

        The main view's rule (one setting for both windows, see
        ``render.View._interaction_subsample_target``) is applied to the
        *visible* population - the requests of this window carry every
        loaded localization, and the preview stride applies to all of
        them alike - and scaled back to the loaded population. Zoomed
        in on a few localizations, a preview therefore renders them
        all; only a view over many localizations is thinned.
        """
        rule = self.window.window.view._interaction_subsample_target
        fraction = self._visible_fraction()
        visible = int(round(fraction * population))
        if visible <= 0:
            return population  # nothing (or no sample) in view: no thinning
        target = rule(visible)
        if target <= 0:
            return 0  # previews disabled
        return int(np.ceil(target * population / visible))

    def _on_render_finished(
        self,
        request_id: int,
        viewport: tuple,
        qimage: QtGui.QImage,
        n_locs: int,
        contrast_limits: tuple[float, float],
        raw_image: np.ndarray,
    ) -> None:
        """Apply a completed worker render on the GUI thread.

        Every frame is shown: a superseded one is still fresher than
        what is on screen and, during a drag, an intermediate
        orientation on the way to the newest request. Only the newest
        full-quality render updates the raw-image cache and the
        contrast spinboxes — a preview's subsampled image would corrupt
        later contrast redraws, and its compensated limits are not the
        user's.
        """
        if (
            request_id == self._render_request_id
            and not self._current_request_interactive
        ):
            self._adopt_render_result(contrast_limits, raw_image)
        self._complete_scene(qimage)

    def stop_render_worker(self) -> None:
        """Stop the render worker thread, if one was started. A render
        in flight is allowed to finish first — destroying a live
        QThread aborts the process."""
        if self._render_thread is not None:
            self._render_thread.quit()
            self._render_thread.wait()
            self._render_thread = None
            self._render_worker = None

    def draw_scalebar(self, image: QtGui.QImage) -> QtGui.QImage:
        """Draw a scalebar.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations.

        Returns
        -------
        QImage
            Image with the drawn scalebar.
        """
        d_dialog = self.window.display_settings_dlg
        if d_dialog.scalebar_groupbox.isChecked():
            color = (
                QtGui.QColor("white")
                if not self.window.dataset_dialog.wbackground.isChecked()
                else QtGui.QColor("black")
            )
            d_dialog = self.window.display_settings_dlg
            image = render.draw_scalebar(
                image=image,
                viewport=self.viewport,
                scalebar_length_nm=d_dialog.scalebar.value(),
                pixelsize=self.window.window.view.pixelsize,
                display_length=d_dialog.scalebar_text.isChecked(),
                color=color,
            )
        return image

    def draw_legend(self, image: QtGui.QImage) -> QtGui.QImage:
        """Draw a legend for multichannel data.

        Displayed in the top left corner, shows the color and the name
        of each channel.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations.

        Returns
        -------
        image : QImage
            Image with the drawn legend.
        """
        if not self.window.legend_action.isChecked():
            return image

        channel_names = []
        channel_colors = []
        for i in range(len(self.locs)):
            if self.window.dataset_dialog.checks[i].isChecked():
                channel_name = self.window.dataset_dialog.checks[i].text()
                channel_names.append(channel_name)
                channel_colors.append(
                    self.window.dataset_dialog.legend_color_8bit(i)
                )
        image = render.draw_legend(
            image=image,
            channel_names=channel_names,
            channel_colors=channel_colors,
        )
        return image

    def draw_rotation(self, image: QtGui.QImage) -> QtGui.QImage:
        """Draw a small 3 axes icon that rotates with locs.

        Displayed in the bottom left corner.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations.

        Returns
        -------
        image : QImage
            Image with the drawn rotation axes icon.
        """
        if self.window.rotation_action.isChecked():
            image = render.draw_rotation(image=image, ang=self._R)
        return image

    def draw_rotation_angles(self, image: QtGui.QImage) -> QtGui.QImage:
        """Draw text displaying current rotation angles in degrees."""
        color = (
            QtGui.QColor("white")
            if not self.window.dataset_dialog.wbackground.isChecked()
            else QtGui.QColor("black")
        )
        if self.window.angles_action.isChecked():
            image = render.draw_rotation_angles(
                image=image, ang=(self.angx, self.angy, self.angz), color=color
            )
        return image

    def draw_points(self, image: QtGui.QImage) -> QtGui.QImage:
        """Draw points and lines and distances between them onto image.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations.

        Returns
        -------
        image : QImage
            Image with the drawn points.
        """
        # the Measure tool looks as in the main window, whose background
        # this window shares
        t_dialog = self.window.window.tools_settings_dialog
        style = t_dialog.measure_overlay_style()
        mark_width = t_dialog.measure_style.value("marker_size")
        # draw all finalized measurement sets (static, no live cursor)
        for point_set in self._point_sets:
            image = render.draw_points(
                image=image,
                viewport=self.viewport,
                points=point_set,
                pixelsize=self.window.window.view.pixelsize,
                mark_width=mark_width,
                style=style,
            )
        # draw the active set; show the live cursor cross and running
        # distance only in Measure mode while the cursor is followed
        cursor = (
            self._measure_cursor
            if self._mode == "Measure" and self._measure_following
            else None
        )
        return render.draw_points(
            image=image,
            viewport=self.viewport,
            points=self._points,
            pixelsize=self.window.window.view.pixelsize,
            mark_width=mark_width,
            cursor=cursor,
            style=style,
        )

    def rotation_input(self) -> None:
        """Ask the user to input 3 rotation angles manually.

        The rotations are applied sequentially around either the data's
        own (rotated) x, y and z axes - the axes shown by the axes icon,
        "object" frame - or the fixed screen/camera axes ("world" frame),
        as chosen in the dialog. With the "object" frame each displayed
        angle changes by exactly the entered amount regardless of the
        current orientation. Angles beyond +/- 180 degrees are preserved,
        e.g. 720 degrees encodes two full turns (relevant for animations).
        """
        angx, angy, angz, frame, ok = RotateByAngleDialog.getParams(self)
        if ok:
            # codebase convention: x angle is the negative of
            # scipy's right-handed x rotation. "object" frame rotates
            # around the data's own axes, "world" around the fixed
            # screen/camera axes.
            self.apply_rotation([-np.radians(angx), 0.0, 0.0], frame=frame)
            self.apply_rotation([0.0, np.radians(angy), 0.0], frame=frame)
            self.apply_rotation([0.0, 0.0, np.radians(angz)], frame=frame)
            self.update_scene()

    def delete_rotation(self) -> None:
        """Reset rotation and any accumulated pan offset."""
        self.set_rotation(Rotation.identity())
        self._pan_z = 0.0
        self.update_scene()

    def fit_in_view_rotated(self, get_viewport: bool = False) -> None:
        """Update viewport to reflect the pick from main window.

        Parameters
        ----------
        get_viewport : bool, optional
            If True, returns the found viewport. Otherwise updates
            scene with the found viewport.

        Returns
        -------
        viewport : list or None
            ``[(y_min, x_min), (y_max, x_max)]`` bounding the pick. Only
            returned if ``get_viewport``; otherwise the scene is updated and
            None is returned. None is also returned when no pick has been
            copied from the main window yet, i.e. before this window has
            been opened for the first time.
        """
        if self.pick_shape is not None:
            x_min, x_max, y_min, y_max = lib.pick_bounds(
                self.pick, self.pick_shape, self.pick_size
            )
            viewport = [(y_min, x_min), (y_max, x_max)]
        elif self._fov_viewport is not None:
            # the field of view that was loaded (see _collect_fov_locs)
            viewport = [tuple(v) for v in self._fov_viewport]
        else:  # never opened; nothing to fit to
            return None
        if get_viewport:
            return viewport
        else:
            self.viewport = viewport
            self._reanchor_pivot()
            self.update_scene()

    def xy_projection(self) -> None:
        """Reset rotation to get XY projection."""
        self.set_rotation(Rotation.identity())
        self.update_scene()

    def xz_projection(self) -> None:
        """Set rotation to get XZ projection."""
        self.set_rotation(render.rotation_matrix(np.pi / 2, 0, 0))
        self.update_scene()

    def yz_projection(self) -> None:
        """Set rotation to get YZ projection."""
        self.set_rotation(render.rotation_matrix(0, np.pi / 2, 0))
        self.update_scene()

    def _arrow_pan(self, sx: float, sy: float) -> None:
        """Arrow-key pan by (sx, sy) world units in the screen frame.

        Mirrors the mouse pan: convert the screen-space shift to a world
        delta via ``R^-1`` so the navigation works at any rotation. The
        X/Y world components shift the pick + viewport (existing path);
        the Z component accumulates into ``self._pan_z``.
        """
        screen_delta = np.array([sx, sy, 0.0], dtype=float)
        world_delta = self._R.inv().apply(screen_delta)
        dx_w = float(world_delta[0])
        dy_w = float(world_delta[1])
        dz_w = float(world_delta[2])
        self.window.move_pick(dx_w, dy_w)
        self._pan_z -= dz_w
        self.shift_viewport(dx_w, dy_w)

    def _arrow_pan_relative(self, fx: float, fy: float) -> None:
        """Arrow-key pan by fractions of the viewport width/height.

        Does nothing if no localizations have been loaded yet, i.e. if
        there is no viewport to pan.

        Parameters
        ----------
        fx, fy : float
            Shift in x and y as fractions of the viewport's width and
            height.
        """
        if self.viewport is None:
            return
        self._arrow_pan(
            fx * render.viewport_width(self.viewport),
            fy * render.viewport_height(self.viewport),
        )

    def to_left_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan_relative(-SHIFT, 0.0)

    def to_right_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan_relative(SHIFT, 0.0)

    def to_up_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan_relative(0.0, -SHIFT)

    def to_down_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan_relative(0.0, SHIFT)

    def set_optimal_scalebar(
        self, force: bool = False, silent: bool = False
    ) -> None:
        """Sets scalebar to approx. 1/8 of the current viewport's
        width."""
        optimal_scalebar = (
            self.window.display_settings_dlg.optimal_scalebar_check
        )
        if force or optimal_scalebar.isChecked():
            width = render.viewport_width(self.viewport)
            scalebar = render.optimal_scalebar_length(self.pixelsize, width)
            scalebar_spinbox = self.window.display_settings_dlg.scalebar
            scalebar_spinbox.blockSignals(True)
            scalebar_spinbox.setValue(scalebar)
            scalebar_spinbox.blockSignals(False)
            if not silent:
                self.update_scene()

    def shift_viewport(self, dx: float, dy: float) -> None:
        """Move viewport by a given amount.

        Parameters
        ----------
        dx, dy : float
            Shift in x and y (camera pixels).
        """
        (y_min, x_min), (y_max, x_max) = self.viewport
        new_viewport = [(y_min + dy, x_min + dx), (y_max + dy, x_max + dx)]
        self.viewport = new_viewport
        self.load_locs()  # the (moved) pick's locs, or the new viewport's
        self._reanchor_pivot()
        self.update_scene(viewport=self.viewport)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Held keys: X, Y, Z lock the rotation axis, S snaps rotation
        to 15 degree steps. Pressed keys: Home fits the loaded region,
        1, 2 and 3 select the XY, XZ and YZ projections."""
        key = event.key()
        Key = QtCore.Qt.Key
        if key == Key.Key_X:
            self.block_x, self.block_y, self.block_z = True, False, False
        elif key == Key.Key_Y:
            self.block_x, self.block_y, self.block_z = False, True, False
        elif key == Key.Key_Z:
            self.block_x, self.block_y, self.block_z = False, False, True
        elif key == Key.Key_S:
            self._snap = True
            self._snap_accum = np.zeros(3)
        elif key == Key.Key_Home and len(self.locs):
            self.fit_in_view_rotated()
        elif key == Key.Key_1 and len(self.locs):
            self.xy_projection()
        elif key == Key.Key_2 and len(self.locs):
            self.xz_projection()
        elif key == Key.Key_3 and len(self.locs):
            self.yz_projection()
        else:
            event.ignore()
            return
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        """Release the axis lock or the rotation snapping."""
        key = event.key()
        Key = QtCore.Qt.Key
        if key in (Key.Key_X, Key.Key_Y, Key.Key_Z):
            self.block_x = self.block_y = self.block_z = False
            event.accept()
        elif key == Key.Key_S:
            self._snap = False
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define actions taken when moving mouse, for example, rotating
        locs, panning or live updating the measuring cross."""
        if self._zoom_rect is not None:
            (x0, y0), _ = self._zoom_rect
            self._zoom_rect = (
                self._zoom_rect[0],
                (event.pos().x(), event.pos().y()),
            )
            # like the 2D window: the rectangle only stretches towards
            # the bottom right; dragging the other way collapses it
            self.rubberband.setGeometry(
                QtCore.QRect(QtCore.QPoint(x0, y0), event.pos())
            )
            return
        if self._pan:
            self._pan_drag(event)
            return

        # live update of the measuring cross and distance
        if self._mode == "Measure" and self._measure_following:
            self._measure_cursor = self.map_to_movie(event.pos())
            if len(self.locs):
                self.update_scene(use_cache=True)
            return

        if self._mode != "Rotate":
            return

        # only rotate while the left button is held; mouse tracking is on
        # so hover events (no button) must not rotate the data
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._rotate_drag(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Hide the live measuring cross when the cursor leaves the
        canvas."""
        if self._mode == "Measure" and self._measure_cursor is not None:
            self._measure_cursor = None
            if len(self.locs):
                self.update_scene(use_cache=True)
        super().leaveEvent(event)

    def _rotate_drag(self, event: QtGui.QMouseEvent) -> None:
        """Trackball-style rotation: incremental rotations are composed
        in the screen frame, so the cursor and the visible data stay in
        sync regardless of any prior rotation."""
        dx_pix = event.pos().x() - self._last_mouse_x
        dy_pix = event.pos().y() - self._last_mouse_y
        self._last_mouse_x = event.pos().x()
        self._last_mouse_y = event.pos().y()
        if dx_pix == 0 and dy_pix == 0:
            return

        # Screen-frame rotation vector. Vertical drag rotates around the
        # screen X axis (tilt), horizontal drag around the screen Y axis
        # (turn).
        ax = 2 * np.pi * dy_pix / self.height()
        ay = 2 * np.pi * dx_pix / self.width()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl = modifiers == QtCore.Qt.KeyboardModifier.ControlModifier

        if self.block_x or self.block_y or self.block_z:
            # Axis locks: pressing X/Y/Z constrains rotation to the
            # corresponding axis. Without Ctrl the rotation is around the
            # *data* axis (the axes shown by the axes icon, "object"
            # frame); with Ctrl it is around the fixed screen/world axis.
            # When Z is locked, vertical drag spins around the screen Z
            # axis so the lock is drivable without Ctrl.
            frame = "world" if ctrl else "object"
            if self.block_z:
                sax, say, saz = 0.0, ay, ax
            else:
                sax, say, saz = ax, ay, 0.0
            v_screen = render.rotation_matrix(sax, say, saz).as_rotvec()
            if frame == "object":
                v = self._R.inv().apply(v_screen)
            else:
                v = v_screen
            keep = np.array(
                [
                    1.0 if self.block_x else 0.0,
                    1.0 if self.block_y else 0.0,
                    1.0 if self.block_z else 0.0,
                ]
            )
            self._apply_drag_rotation(v * keep, frame)
        else:
            # Free trackball. With Ctrl, horizontal drag turns and
            # vertical drag spins in the screen plane (screen Z spin),
            # matching the previous Ctrl semantics.
            az = 0.0
            if ctrl:
                az = ax
                ax = 0.0
            delta_R = render.rotation_matrix(ax, ay, az)
            self._apply_drag_rotation(delta_R.as_rotvec(), "world")
        self.update_scene(interactive=True)

    def _pan_drag(self, event: QtGui.QMouseEvent) -> None:
        """Inverse-rotation panning: convert the screen-space mouse delta
        into a world-space translation via ``R^-1``. The X/Y components
        of the world delta shift the viewport (existing path); the Z
        component accumulates into ``self._pan_z`` (applied to locs.z in
        ``render_scene``). Works at any rotation, including ±90°."""
        dx_pix = event.pos().x() - self.pan_start_x
        dy_pix = event.pos().y() - self.pan_start_y
        self.pan_start_x = event.pos().x()
        self.pan_start_y = event.pos().y()
        if dx_pix == 0 and dy_pix == 0:
            return
        vh, vw = render.viewport_size(self.viewport)
        # the content follows the mouse: the view target moves the
        # other way
        self._shift_view(
            -dx_pix / self.width() * vw, -dy_pix / self.height() * vh
        )
        self._reanchor_pivot()
        self.update_scene(interactive=True)

    def _shift_view(self, screen_dx: float, screen_dy: float) -> None:
        """Move the view target by a shift given in the screen frame
        (camera pixels along the screen axes): converted to world
        coordinates through ``R^-1``, its X/Y components move the
        viewport and its Z component the depth ``_pan_z``, so the shift
        is right at any rotation, including ±90°."""
        world_delta = self._R.inv().apply([screen_dx, screen_dy, 0.0])
        (y_min, x_min), (y_max, x_max) = self.viewport
        dx, dy = float(world_delta[0]), float(world_delta[1])
        self.viewport = [(y_min + dy, x_min + dx), (y_max + dy, x_max + dx)]
        self._pan_z += float(world_delta[2])

    def _zoom_at(self, factor: float, pos: QtCore.QPointF | None) -> None:
        """Zoom the view by ``factor`` (below one zooms in) about the
        widget position ``pos``, which stays under the cursor; about
        the center when ``pos`` is None."""
        if self.viewport is None or not len(self.locs):
            return
        vh, vw = render.viewport_size(self.viewport)
        if pos is not None:
            # the cursor's offset from the center (screen frame) keeps
            # its place: the target moves towards it by (1 - factor)
            rel_x = pos.x() / self.width() - 0.5
            rel_y = pos.y() / self.height() - 0.5
            self._shift_view(
                rel_x * vw * (1 - factor), rel_y * vh * (1 - factor)
            )
        self.viewport = render.zoom_viewport(self.viewport, factor)
        self._reanchor_pivot()
        self.update_scene()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Ctrl (Cmd on macOS) + mouse wheel or trackpad scroll zooms
        about the cursor (smooth: about ten percent per wheel notch),
        as in the main window."""
        delta = event.angleDelta().y()
        ctrl = event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        if delta == 0 or not ctrl or not len(self.locs):
            event.ignore()
            return
        self._zoom_at(1.1 ** (-delta / 120.0), event.position())
        event.accept()

    def event(self, event: QtCore.QEvent) -> bool:
        """Pinch-to-zoom on trackpads (macOS native gesture)."""
        if event.type() == QtCore.QEvent.Type.NativeGesture and (
            event.gestureType()
            == QtCore.Qt.NativeGestureType.ZoomNativeGesture
        ):
            if len(self.locs):
                self._zoom_at(1.0 / (1.0 + event.value()), event.position())
            return True
        return super().event(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Treat the double click as a press, which is what QWidget does
        by default, then remember it: a third click completes a triple
        click (see ``mousePressEvent``)."""
        self.mousePressEvent(event)
        self._triple_click.double_clicked(event)

    def _triple_clicked(self, event: QtGui.QMouseEvent) -> None:
        """Triple click fits the loaded region back into the window;
        with Shift it also resets the rotation."""
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self.set_rotation(Rotation.identity())
            self._pan_z = 0.0
        self.fit_in_view_rotated()
        event.accept()

    def _finish_zoom_rectangle(self) -> None:
        """Zoom to the rectangle dragged with Shift + left button (a
        drag too small to be one is ignored)."""
        (x0, y0), (x1, y1) = self._zoom_rect
        self._zoom_rect = None
        self.rubberband.hide()
        # releasing above or left of the start cancels (as in 2D)
        w_pix, h_pix = x1 - x0, y1 - y0
        if w_pix < 5 or h_pix < 5 or not len(self.locs):
            return
        vh, vw = render.viewport_size(self.viewport)
        # the rectangle's center becomes the view target, its extent
        # (widened to the window's aspect by draw_scene) the field
        cx = (x0 + x1) / 2 / self.width() - 0.5
        cy = (y0 + y1) / 2 / self.height() - 0.5
        self._shift_view(cx * vw, cy * vh)
        (y_min, x_min), (y_max, x_max) = self.viewport
        new_w, new_h = w_pix / self.width() * vw, h_pix / self.height() * vh
        center_y, center_x = render.viewport_center(self.viewport)
        self.viewport = [
            (center_y - new_h / 2, center_x - new_w / 2),
            (center_y + new_h / 2, center_x + new_w / 2),
        ]
        self.viewport = self.adjust_viewport_to_view(self.viewport)
        self._reanchor_pivot()
        self.update_scene()

    def _apply_drag_rotation(self, rotvec: np.ndarray, frame: str) -> None:
        """Apply a drag's rotation increment, or, while S is held,
        accumulate it and apply whole ``SNAP_STEP`` steps only."""
        if not self._snap:
            self.apply_rotation(rotvec, frame=frame)
            return
        self._snap_accum = self._snap_accum + rotvec
        magnitude = float(np.linalg.norm(self._snap_accum))
        if magnitude < self.SNAP_STEP:
            return
        steps = int(magnitude // self.SNAP_STEP)
        quantum = self._snap_accum / magnitude * self.SNAP_STEP * steps
        self._snap_accum = self._snap_accum - quantum
        self.apply_rotation(quantum, frame=frame)

    #: rotation step while S is held (radians): 15 degrees
    SNAP_STEP = np.pi / 12

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define actions taken when pressing mouse buttons, for
        example, starting rotating locs or panning.

        Navigation works in every mode: Shift + left button drags a
        zoom rectangle; the right button, the middle button or Alt
        (Option) + left button pan. In Rotate mode a plain left drag
        rotates. In Measure mode the right button keeps its measuring
        role (freeze a set, delete the last set, see
        ``mouseReleaseEvent``), as in the main window; pan with the
        middle button or Alt + left there."""
        left = event.button() == QtCore.Qt.MouseButton.LeftButton
        right = event.button() == QtCore.Qt.MouseButton.RightButton
        middle = event.button() == QtCore.Qt.MouseButton.MiddleButton
        modifiers = event.modifiers()
        if self._triple_click.is_third(event) and len(self.locs):
            self._triple_clicked(event)
            return
        if left and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            self._zoom_rect = (
                (event.pos().x(), event.pos().y()),
                (event.pos().x(), event.pos().y()),
            )
            self.rubberband.setGeometry(
                QtCore.QRect(event.pos(), QtCore.QSize())
            )
            self.rubberband.show()
            event.accept()
            return
        if (
            middle
            or (right and self._mode != "Measure")
            or (left and modifiers & QtCore.Qt.KeyboardModifier.AltModifier)
        ):
            self._pan = True
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._mode == "Rotate" and left:
            # start rotation
            self._last_mouse_x = event.pos().x()
            self._last_mouse_y = event.pos().y()
            self._snap_accum = np.zeros(3)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define actions taken when releasing mouse buttons, for
        example, stopping rotating locs or panning, add or delete a measure
        point."""
        if self._zoom_rect is not None:
            self._zoom_rect = (
                self._zoom_rect[0],
                (event.pos().x(), event.pos().y()),
            )
            self._finish_zoom_rectangle()
            event.accept()
            return
        if self._pan:
            self._pan = False
            self.update_cursor()
            event.accept()
            return
        if self._mode == "Measure":
            # add a measure point on left click; the first right click
            # freezes the current set so a new one can be started; a
            # further right click then deletes the last finalized set
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                # start a new set if the previous one was frozen
                if not self._measure_following:
                    self._measure_following = True
                    self.update_cursor()
                x, y = self.map_to_movie(event.pos())
                self.add_point((x, y))
                event.accept()
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                if self._measure_following:
                    # freeze the current selection (stop following)
                    self.finalize_measure_set()
                else:
                    # delete the last finalized set of measurements
                    self.remove_last_measure_set()
                event.accept()
            else:
                event.ignore()

        elif self._mode == "Rotate":
            # stop rotation
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                event.accept()

    def map_to_movie(self, position: QtCore.QPoint) -> tuple[float, float]:
        """Convert coordinates from Qt display units to camera units."""
        x_rel = position.x() / self.width()
        x_movie = (
            x_rel * render.viewport_width(self.viewport) + self.viewport[0][1]
        )
        y_rel = position.y() / self.height()
        y_movie = (
            y_rel * render.viewport_height(self.viewport) + self.viewport[0][0]
        )
        return x_movie, y_movie

    def pan_relative(
        self, dy: float, dx: float, interactive: bool = False
    ) -> None:
        """Move viewport by a given relative distance.

        Parameters
        ----------
        dy, dx : float
            Relative displacement of the viewport in y or x axis.
        interactive : bool, optional
            True during a drag, see ``update_scene``.
        """
        viewport_height, viewport_width = render.viewport_size(self.viewport)
        x_move = dx * viewport_width
        y_move = dy * viewport_height
        x_min = self.viewport[0][1] - x_move
        x_max = self.viewport[1][1] - x_move
        y_min = self.viewport[0][0] - y_move
        y_max = self.viewport[1][0] - y_move
        self.viewport = [(y_min, x_min), (y_max, x_max)]
        self.update_scene(interactive=interactive)

    def add_point(
        self,
        position: tuple[float, float],
        update_scene: bool = True,
    ) -> None:
        """Add a point at a given position for measuring distances."""
        self._points.append(position)
        if update_scene:
            self.update_scene()

    def remove_points(self) -> None:
        """Remove all distance measurement points and sets."""
        self._points = []
        self._point_sets = []
        self._measure_following = True
        self._measure_cursor = None
        self.update_cursor()
        self.update_scene()

    def finalize_measure_set(self) -> None:
        """Freeze the current measurement set so a new one can be
        started. The cursor is no longer followed until the next left
        click."""
        if self._points:
            self._point_sets.append(self._points)
            self._points = []
        self._measure_following = False
        self._measure_cursor = None
        self.update_cursor()
        self.update_scene()

    def remove_last_measure_set(self) -> None:
        """Delete the most recently finalized measurement set."""
        if self._point_sets:
            self._point_sets.pop()
            self.update_scene()

    def update_cursor(self) -> None:
        """Change the cursor according to self._mode."""
        if self._mode == "Measure" and self._measure_following:
            # hide the OS cursor; the drawn cross marks the position
            self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        else:
            self.unsetCursor()

    def export_current_view(self) -> None:
        """Export current view as .png or .tif."""
        try:
            base, ext = os.path.splitext(self.paths[0])
        except AttributeError:
            return
        out_path = base + "_rotated_{}_{}_{}.png".format(
            int(self.angx * 180 / np.pi),
            int(self.angy * 180 / np.pi),
            int(self.angz * 180 / np.pi),
        )
        check_ext = [".yaml"]
        scalebar_box = self.window.display_settings_dlg.scalebar_groupbox
        scalebar = scalebar_box.isChecked()
        if scalebar:
            check_ext.append("_scalebar.png")
        if self.x_render_state:
            check_ext.append(COLORBAR_SUFFIX + io.colorbar_export_format())
        path, ext = lib.get_save_filename_ext_dialog(
            self,
            "Save image",
            out_path,
            filter="*.png;;*.tif",
            check_ext=check_ext,
        )
        if path:
            self.qimage.save(path)
            self.save_property_colorbar(path)
            self.export_current_view_info(path)
            if not scalebar:
                self.set_optimal_scalebar(force=True)
                scalebar_box.setChecked(True)
                self.update_scene(synchronous=True)
                self.qimage.save(os.path.splitext(path)[0] + "_scalebar.png")
                scalebar_box.setChecked(False)
                self.update_scene(synchronous=True)

    def save_property_colorbar(self, path: str) -> None:
        """Save the color bar (LUT) of the rendered property next to an
        exported image, as ``*_colorbar.*``.

        The color bar is saved by the main window, which holds the
        colormap and the property values that the rotated view is
        rendered with, and the format it is saved in. Does nothing if
        rendering by property is inactive.

        Parameters
        ----------
        path : str
            Path that the image itself was saved to. The color bar is
            saved next to it.
        """
        if not self.x_render_state:
            return
        self.window.window.view.save_property_colorbar(path)

    def export_current_view_info(self, path: str) -> None:
        """Export current view's information."""
        (y_min, x_min), (y_max, x_max) = self.viewport
        fov = [x_min, y_min, x_max - x_min, y_max - y_min]
        fov = [float(_) for _ in fov]
        d = self.window.display_settings_dlg
        colors = [
            _.currentText() for _ in self.window.dataset_dialog.colorselection
        ]
        pixelsize = self.window.window.view.pixelsize
        rot_angles = [
            int(self.angx * 180 / np.pi),
            int(self.angy * 180 / np.pi),
            int(self.angz * 180 / np.pi),
        ]
        info = {
            "Generated by": f"Picasso v{__version__} Render 3D",
            "Rotation angles (deg)": rot_angles,
            "FOV (X, Y, Width, Height)": fov,
            "Display pixel size (nm)": d.disp_px_size.value(),
            "Min. density": d.minimum.value(),
            "Max. density": d.maximum.value(),
            "Colormap": d.colormap.currentText(),
            "Blur method": d.blur_method(),
            "Scale bar length (nm)": d.scalebar.value(),
            "Min. blur (nm)": d.min_blur_width.value() / pixelsize,
            "Localizations loaded": self.paths,
            "Colors": colors,
        }
        if self.x_render_state:  # rendering by property
            info["Render property"] = self.x_property
            info["Render property min."] = self.x_min_val
            info["Render property max."] = self.x_max_val
            info["Render property colors"] = self.x_n_colors
            info["Colormap property"] = self.x_colormap
        path, ext = os.path.splitext(path)
        path = path + ".yaml"
        io.save_info(path, [info])

    def zoom_in(self) -> None:
        """Zoom in by a constant factor."""
        self.zoom(1 / ZOOM)

    def zoom_out(self) -> None:
        """Zoom out by a constant factor."""
        self.zoom(ZOOM)

    def zoom(self, factor: float) -> None:
        """Change zoom relatively to factor by changing viewport."""
        self.viewport = render.zoom_viewport(self.viewport, factor)
        # what is in view changed: keep the pivot at its depth
        self._reanchor_pivot()
        self.update_scene()

    # --- rotation pivot --- #
    # The render pipeline rotates about the world point at the screen
    # center: the viewport center in x/y at depth ``_pan_z`` (z is
    # shifted by ``-_pan_z`` before rotating, see ``_apply_pan_z``).
    # A screen-space pan while the view is tilted has a z component in
    # world coordinates, which would accumulate in ``_pan_z`` and leave
    # the pivot in front of or behind the structure at the screen
    # center - rotations then make it orbit. Under the orthographic
    # projection the pivot can slide along the viewing direction without
    # changing the image, so after every pan or zoom it is slid to the
    # median depth of the localizations in view.
    _PIVOT_SAMPLE = 200_000  # localizations sampled for the median depth

    def _in_view_sample(self) -> tuple[np.ndarray, np.ndarray] | None:
        """A sample of at most ``_PIVOT_SAMPLE`` localizations (x, y, z
        in the loaded frame) and, per row, whether its rotated position
        falls inside the viewport; None when nothing is loaded."""
        if not self.locs or self.viewport is None:
            return None
        channels = [
            locs
            for locs in self.locs
            if len(locs) and {"x", "y", "z"} <= set(locs.columns)
        ]
        if not channels:
            return None
        total = sum(len(locs) for locs in channels)
        step = max(1, -(-total // self._PIVOT_SAMPLE))
        xyz = np.concatenate(
            [
                locs[["x", "y", "z"]].to_numpy(dtype=float)[::step]
                for locs in channels
            ]
        )
        (y_min, x_min), (y_max, x_max) = self.viewport
        pivot = np.array(
            [
                x_min + (x_max - x_min) / 2,
                y_min + (y_max - y_min) / 2,
                self._pan_z,
            ]
        )
        screen = self._R.apply(xyz - pivot)
        in_view = (np.abs(screen[:, 0]) <= (x_max - x_min) / 2) & (
            np.abs(screen[:, 1]) <= (y_max - y_min) / 2
        )
        return xyz, in_view

    def _in_view_median_z(self) -> float | None:
        """Median z (camera pixels, the loaded frame) of the localizations
        whose rotated position falls inside the viewport; None when no
        localization is in view."""
        sample = self._in_view_sample()
        if sample is None:
            return None
        xyz, in_view = sample
        if not in_view.any():
            return None
        return float(np.median(xyz[in_view, 2]))

    def _visible_fraction(self) -> float:
        """Fraction of the loaded localizations whose rotated position
        falls inside the viewport (estimated from a sample); 1.0 when
        nothing is loaded."""
        sample = self._in_view_sample()
        if sample is None:
            return 1.0
        return float(sample[1].mean())

    def _reanchor_pivot(self) -> None:
        """Slide the rotation pivot along the viewing direction to the
        median depth of the localizations in view (see the note above);
        the viewport and ``_pan_z`` move together, the image does not."""
        if not self.locs or self.viewport is None:
            return
        direction = self._R.inv().apply([0.0, 0.0, 1.0])  # screen normal
        if abs(direction[2]) < 0.2:
            return  # nearly edge-on: no well-defined depth along the ray
        z_ref = self._in_view_median_z()
        if z_ref is None:
            return
        t = (z_ref - self._pan_z) / direction[2]
        dx, dy = t * float(direction[0]), t * float(direction[1])
        (y_min, x_min), (y_max, x_max) = self.viewport
        self.viewport = [(y_min + dy, x_min + dx), (y_max + dy, x_max + dx)]
        self._pan_z = z_ref

    def set_mode(self, action: QtGui.QAction) -> None:
        """Set ``self._mode`` for QMouseEvents.

        Activated when Rotate or Measure is chosen from Tools menu
        in the main window.

        Parameters
        ----------
        action : QAction
            Action defined in Window.__init__: ("Rotate" or "Measure").
        """
        self._mode = action.text()
        self.update_cursor()

    def adjust_viewport_to_view(
        self, viewport: tuple[tuple[float, float], tuple[float, float]]
    ) -> tuple[float, float, float, float]:
        """Add space to a desired viewport, such that it matches the
        window aspect ratio and return the viewport."""
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

    def get_render_kwargs(
        self,
        viewport: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None,
    ) -> dict:
        """
        Returns a dictionary to be used for the keyword arguments of
        render.render.

        Parameters
        ----------
        viewport : list, optional
            Specifies the FOV to be rendered ``((y_min, x_min),
            (y_max, x_max))``. If None, the current viewport is taken.

        Returns
        -------
        kwargs : dict
            Contains blur method, display pixel size, viewport and min
            blur width.
        """
        disp_dlg = self.window.display_settings_dlg
        pixelsize = self.window.window.view.pixelsize

        # oversampling
        opt_oversampling = self.display_pixels_per_viewport_pixels(
            viewport=viewport
        )
        opt_disp_px_size = pixelsize / opt_oversampling
        if disp_dlg.dynamic_disp_px.isChecked():
            disp_px_size = opt_disp_px_size
            disp_dlg.set_disp_px_silently(opt_disp_px_size)
        else:
            if disp_dlg.disp_px_size.value() < opt_disp_px_size:
                QtWidgets.QMessageBox.information(
                    self,
                    "Display pixel size too low",
                    (
                        "Display pixel size will be adjusted to"
                        " match the display pixel density."
                    ),
                )
                disp_px_size = opt_disp_px_size
                disp_dlg.set_disp_px_silently(opt_disp_px_size)
            else:
                disp_px_size = disp_dlg.disp_px_size.value()

        # viewport
        if viewport is None:
            viewport = self.viewport

        kwargs = {
            "disp_px_size": disp_px_size,
            "viewport": viewport,
            "blur_method": disp_dlg.blur_method(),
            "min_blur_width": float(
                disp_dlg.min_blur_width.value() / pixelsize
            ),
            "quadtree_capacity": disp_dlg.quadtree_capacity.value(),
            "triangulation_passes": disp_dlg.triangulation_passes.value(),
            "triangulation_jitter": disp_dlg.triangulation_jitter.value(),
        }
        return kwargs

    def display_pixels_per_viewport_pixels(
        self,
        viewport: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None,
    ) -> float:
        """Return optimal oversampling, i.e., the number of display
        pixels per camera pixel."""
        viewport = viewport or self.viewport
        os_horizontal = self.width() / render.viewport_width(viewport)
        os_vertical = self.height() / render.viewport_height(viewport)
        # The values should be identical, but just in case,
        # we choose the maximum value:
        return max(os_horizontal, os_vertical)

    def _apply_pan_z(
        self,
        locs: pd.DataFrame | list[pd.DataFrame],
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return ``locs`` with z translated by ``-self._pan_z``.

        The render pipeline rotates locs around ``z = 0`` after centering
        in X/Y; subtracting ``_pan_z`` from z shifts the rotation pivot in
        Z without touching ``render.locs_rotation``.
        """
        if isinstance(locs, pd.DataFrame):
            return locs.assign(z=locs["z"] - self._pan_z)
        return [L.assign(z=L["z"] - self._pan_z) for L in locs]

    def _prepare_locs_for_rendering(
        self,
    ) -> tuple[list[pd.DataFrame], list[list[dict]]]:
        """Return locs list and use render-property-colored locs if
        requested."""
        # render by property - use x_locs like multichannel rendering
        if self.x_render_state:
            locs = self.x_locs.copy()
            infos = [self.infos[0]] * len(locs)
        # if group column is present, split locs by group for rendering
        else:
            locs = self.locs
            infos = self.infos
            if "group" in locs[0].columns and len(locs) == 1:
                locs = render.split_locs_by_group(
                    locs[0], group_color=self.group_color
                )
                infos = [self.infos[0]] * len(locs)

        # if multiple channels are loaded, selected only the ones which
        # are checked in the Dataset Dialog
        render_check = (
            self.window.window.display_settings_dlg.render_check.isChecked()
        )
        if len(self.locs) > 1:
            locs_ = []
            info_ = []
            for i in range(len(locs)):
                if self.window.dataset_dialog.checks[i].isChecked():
                    locs_.append(locs[i])
                    info_.append(infos[i])
            locs = locs_
            infos = info_
        elif (
            len(self.locs) == 1
            and "group" not in self.locs[0].columns
            and not render_check
        ):
            locs = locs[0]
            infos = infos[0]
        return locs, infos


class RotationWindow(QtWidgets.QMainWindow):
    """Rotation window.

    ...

    Attributes
    ----------
    angles_action : QtGui.QAction
        Action to toggle the display of current rotation angles.
    animation_dialog : AnimationDialog
        Instance of animation dialog.
    display_settings_dlg : DisplaySettingsRotationDialog
        Instance of display settings rotation dialog.
    legend_action : QtGui.QAction
        Action to toggle the display of the legend.
    menu_bar : QMenuBar
        Menu bar with menus: File, View, Tools.
    menus : list
        Contains File, View and Tools menus, used for plugins.
    rotation_action : QtGui.QAction
        Action to toggle the display of reference axes.
    view_rot : ViewRotation
        Instance of the class for displaying rendered localizations.
    window : QMainWindow
        Instance of the main Picasso: Render window (RotationWindow's
        parent).
    """

    DOCS_URL = docs_url("render.html#d-rotation-window")

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__()
        self.setWindowTitle(f"Picasso v{__version__}: Render 3D")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setWindowIcon(icon)

        self.window = window
        self.view_rot = ViewRotation(self)
        self.setCentralWidget(self.view_rot)
        self.display_settings_dlg = DisplaySettingsRotationDialog(self)
        self.animation_dialog = AnimationDialog(self)

        self.menu_bar = self.menuBar()

        # menu bar - File
        file_menu = self.menu_bar.addMenu("File")
        save_action = file_menu.addAction("Save rotated localizations...")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs_rotated)

        file_menu.addSeparator()
        export_view = file_menu.addAction("Export current view...")
        export_view.setShortcut("Ctrl+E")
        export_view.triggered.connect(self.view_rot.export_current_view)
        animation = file_menu.addAction("Build an animation...")
        animation.setShortcut("Ctrl+Shift+E")
        animation.triggered.connect(self.animation_dialog.show)
        help_action = file_menu.addAction("Help")
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )

        # menu bar - View
        view_menu = self.menu_bar.addMenu("View")
        display_settings_action = view_menu.addAction("Display settings...")
        display_settings_action.setShortcut("Ctrl+D")
        display_settings_action.triggered.connect(
            self.display_settings_dlg.show
        )
        view_menu.addAction(display_settings_action)
        self.legend_action = view_menu.addAction("Show/hide legend")
        self.legend_action.setCheckable(True)
        self.legend_action.setChecked(False)
        self.legend_action.setShortcut("Ctrl+L")
        self.legend_action.triggered.connect(self.update_scene)
        self.rotation_action = view_menu.addAction("Show/hide rotation")
        self.rotation_action.setCheckable(True)
        self.rotation_action.setChecked(True)
        self.rotation_action.setShortcut("Ctrl+P")
        self.rotation_action.triggered.connect(self.update_scene)
        self.angles_action = view_menu.addAction("Show/hide rotation angles")
        self.angles_action.setCheckable(True)
        self.angles_action.setChecked(False)
        self.angles_action.triggered.connect(self.update_scene)

        view_menu.addSeparator()
        rotation_action = view_menu.addAction("Rotate by angle...")
        rotation_action.triggered.connect(self.view_rot.rotation_input)
        rotation_action.setShortcut("Ctrl+Shift+R")

        delete_rotation_action = view_menu.addAction("Reset rotation")
        delete_rotation_action.triggered.connect(self.view_rot.delete_rotation)
        delete_rotation_action.setShortcut("Ctrl+Shift+W")
        fit_in_view_action = view_menu.addAction("Fit image to window")
        fit_in_view_action.setShortcut("Ctrl+W")
        fit_in_view_action.triggered.connect(self.view_rot.fit_in_view_rotated)

        view_menu.addSeparator()
        xy_proj_action = view_menu.addAction("XY projection")
        xy_proj_action.triggered.connect(self.view_rot.xy_projection)
        xz_proj_action = view_menu.addAction("XZ projection")
        xz_proj_action.triggered.connect(self.view_rot.xz_projection)
        yz_proj_action = view_menu.addAction("YZ projection")
        yz_proj_action.triggered.connect(self.view_rot.yz_projection)

        view_menu.addSeparator()
        to_left_action = view_menu.addAction("Left")
        to_left_action.setShortcut("Left")
        to_left_action.triggered.connect(self.view_rot.to_left_rot)
        to_right_action = view_menu.addAction("Right")
        to_right_action.setShortcut("Right")
        to_right_action.triggered.connect(self.view_rot.to_right_rot)
        to_up_action = view_menu.addAction("Up")
        to_up_action.setShortcut("Up")
        to_up_action.triggered.connect(self.view_rot.to_up_rot)
        to_down_action = view_menu.addAction("Down")
        to_down_action.setShortcut("Down")
        to_down_action.triggered.connect(self.view_rot.to_down_rot)

        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction("Zoom in")
        zoom_in_action.setShortcuts(["Ctrl++", "Ctrl+="])
        zoom_in_action.triggered.connect(self.view_rot.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction("Zoom out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.view_rot.zoom_out)
        view_menu.addAction(zoom_out_action)

        # menu bar - Tools
        tools_menu = self.menu_bar.addMenu("Tools")
        tools_actiongroup = QtGui.QActionGroup(self.menu_bar)

        measure_tool_action = tools_actiongroup.addAction(
            QtGui.QAction("Measure", tools_menu, checkable=True)
        )
        measure_tool_action.setShortcut("Ctrl+M")
        tools_menu.addAction(measure_tool_action)
        tools_actiongroup.triggered.connect(self.view_rot.set_mode)

        rotate_tool_action = tools_actiongroup.addAction(
            QtGui.QAction("Rotate", tools_menu, checkable=True)
        )
        rotate_tool_action.setShortcut("Ctrl+R")
        tools_menu.addAction(rotate_tool_action)

        self.menus = [file_menu, view_menu, tools_menu]
        # the window is built here but shown only on demand; see
        # ``hideEvent`` for why its shortcuts must stay inert
        self.menu_bar.setEnabled(False)
        self.setMinimumSize(100, 100)
        self.move(20, 20)

    def move_pick(self, dx: float, dy: float) -> None:
        """Move the pick in the main window by a given amount.

        Parameters
        ----------
        dx, dy : float
            Pick shift in x or y axis (camera pixels).
        """
        if self.view_rot.pick_shape is None:
            return  # the field of view is shown: no pick to move
        if self.view_rot.pick_shape in ["Circle", "Square"]:
            x = self.window.view._picks[0][0]
            y = self.window.view._picks[0][1]
            self.window.view._picks = [(x + dx, y + dy)]  # main window
            self.view_rot.pick = (x + dx, y + dy)  # view rotation
        elif self.view_rot.pick_shape in ["Rectangle", "Box"]:
            # both are defined by two points: the ends of the center
            # axis for a rectangle, two opposite corners for a box
            (xs, ys), (xe, ye) = self.window.view._picks[0]
            self.window.view._picks = [
                (
                    (xs + dx, ys + dy),
                    (xe + dx, ye + dy),
                )
            ]  # main window
            self.view_rot.pick = (
                (xs + dx, ys + dy),
                (xe + dx, ye + dy),
            )  # view rotation
        elif self.view_rot.pick_shape == "Polygon":
            new_pick = []
            for point in self.window.view._picks[0]:
                new_pick.append((point[0] + dx, point[1] + dy))
            self.window.view._picks = [new_pick] + []  # main window
            self.view_rot.pick = new_pick  # view rotation
        elif self.view_rot.pick_shape == "Brush":
            # shift every point of every stroke, keeping their widths
            new_pick = [
                (
                    stroke[0],
                    [(x + dx, y + dy) for x, y in stroke[1]],
                )
                for stroke in self.window.view._picks[0]
            ]
            self.window.view._picks = [new_pick]  # main window
            self.view_rot.pick = new_pick  # view rotation

        self.window.view.update_scene()  # update scene in main window

    def _resolve_picks(self) -> tuple:
        """Obtain pick coordiantes for saving given their shape."""
        pixelsize = self.window.view.pixelsize
        if self.view_rot.pick_shape is None:
            # the field of view: its bounds, like a box pick
            (y0, x0), (y1, x1) = self.view_rot.viewport
            pick = [[float(x0), float(y0)], [float(x1), float(y1)]]
        elif self.view_rot.pick_shape in ["Circle", "Square"]:
            x, y = self.view_rot.pick
            pick = [float(x), float(y)]
        elif self.view_rot.pick_shape in ["Rectangle", "Box"]:
            (x0, y0), (x1, y1) = self.view_rot.pick
            pick = [[float(x0), float(y0)], [float(x1), float(y1)]]
        elif self.view_rot.pick_shape == "Brush":
            # same stroke form as the picks file, widths in nm
            pick = [
                {
                    "Width (nm)": float(stroke[0] * pixelsize),
                    "Path": [[float(x), float(y)] for x, y in stroke[1]],
                }
                for stroke in self.view_rot.pick
            ]
        else:  # polygon - an arbitrary number of vertices
            pick = [[float(x), float(y)] for x, y in self.view_rot.pick]
        return pick

    def save_locs_rotated(self) -> None:
        """Save locs from the main window and provides rotation info for
        later loading."""
        channel = self.window.view.get_channel_save_locs(
            "Save rotated localizations"
        )
        if channel is not None:
            # rotation info
            angx = int(self.view_rot.angx * 180 / np.pi)
            angy = int(self.view_rot.angy * 180 / np.pi)
            angz = int(self.view_rot.angz * 180 / np.pi)
            pixelsize = self.window.view.pixelsize
            pick = self._resolve_picks()
            size = self.view_rot.pick_size
            new_info = [
                {
                    "Generated by": f"Picasso v{__version__} Render 3D",
                    "Pick": pick,
                    "Pick shape": self.view_rot.pick_shape or "Field of view",
                    # polygons and boxes carry their own extent
                    "Pick size (nm)": (
                        size * pixelsize if size is not None else None
                    ),
                    # accumulated rotation angles incl. full turns
                    # (radians, codebase convention); legacy keys
                    "angx": float(self.view_rot.angx),
                    "angy": float(self.view_rot.angy),
                    "angz": float(self.view_rot.angz),
                    "Quaternion (x, y, z, w)": [
                        float(q) for q in self.view_rot.rotation.as_quat()
                    ],
                }
            ]

            # combine all channels
            if channel == len(self.view_rot.paths) + 1:
                base, ext = os.path.splitext(self.view_rot.paths[0])
                out_path = base + "_multi.hdf5"
                path, ext = lib.get_save_filename_ext_dialog(
                    self,
                    "Save picked localizations",
                    out_path,
                    filter="*.hdf5",
                    check_ext=".yaml",
                )
                if path:
                    # combine locs from all channels, which need not all
                    # have the same columns
                    all_locs = lib.concat_locs(self.window.view.locs)
                    all_locs.sort_values(
                        kind="quicksort",
                        by="frame",
                        inplace=True,
                    )
                    info = self.view_rot.infos[0] + new_info
                    io.save_locs(path, all_locs, info)
            # save all channels one by one
            elif channel == len(self.view_rot.paths):  # all channels
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.EchoMode.Normal,
                    f"_arotated_{angx}_{angy}_{angz}",
                )  # get the save file suffix
                if ok:
                    for channel in range(len(self.view_rot.paths)):
                        base, ext = os.path.splitext(
                            self.view_rot.paths[channel]
                        )
                        out_path = base + suffix + ".hdf5"
                        info = self.view_rot.infos[channel] + new_info
                        io.save_locs(
                            out_path, self.window.view.locs[channel], info
                        )
            # save one channel only
            else:
                out_path = (
                    os.path.splitext(self.view_rot.paths[channel])[0]
                    + f"_rotated_{angx}_{angy}_{angz}.hdf5"
                )
                path, ext = lib.get_save_filename_ext_dialog(
                    self,
                    "Save rotated localizations",
                    out_path,
                    filter="*.hdf5",
                    check_ext=".yaml",
                )
                if path:
                    info = self.view_rot.infos[channel] + new_info
                    io.save_locs(path, self.window.view.locs[channel], info)

    def update_scene(self) -> None:
        """Update the scene in ViewRotation."""
        self.view_rot.update_scene()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Arm the menu bar when the window is shown, see ``hideEvent``."""
        self.menu_bar.setEnabled(True)
        QtWidgets.QMainWindow.showEvent(self, event)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        """Disable the menu bar while the window is hidden.

        This window is built together with the main Render window but
        only shown on demand. macOS shares one native menu bar across
        the application, so its shortcuts (Ctrl+W, Ctrl+S, ...) would
        otherwise fire from the main window while this one has never
        been opened - the shortcuts of a disabled menu do not.
        """
        self.menu_bar.setEnabled(False)
        QtWidgets.QMainWindow.hideEvent(self, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Close all children dialogs and self; the render worker
        thread stops too (it restarts with the next asynchronous
        render when the window is opened again)."""
        self.display_settings_dlg.close()
        self.animation_dialog.stop_build()
        self.animation_dialog.close()
        self.view_rot.stop_render_worker()
        QtWidgets.QMainWindow.closeEvent(self, event)
