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
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from .. import io, render, lib, __version__


DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5
N_GROUP_COLORS = render.N_GROUP_COLORS  # 8
SHIFT = 0.1
ZOOM = 9 / 7


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
        c_label = QtWidgets.QLabel("Colormap:")
        c_label.setToolTip("Colormap used to render localizations.")
        contrast_grid.addWidget(c_label, 2, 0)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(plt.colormaps())
        contrast_grid.addWidget(self.colormap, 2, 1)
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

        blur_grid.addWidget(points_button, 0, 0, 1, 2)
        blur_grid.addWidget(smooth_button, 1, 0, 1, 2)
        blur_grid.addWidget(convolve_button, 2, 0, 1, 2)
        blur_grid.addWidget(gaussian_button, 3, 0, 1, 2)
        blur_grid.addWidget(gaussian_iso_button, 4, 0, 1, 2)
        convolve_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.render_scene_nocache)
        min_blur_label = QtWidgets.QLabel("Min. Blur (nm):")
        min_blur_label.setToolTip(
            "Minimum blur applied to all localizations in nm."
        )
        blur_grid.addWidget(min_blur_label, 5, 0, 1, 1)
        self.min_blur_width = QtWidgets.QDoubleSpinBox()
        self.min_blur_width.setRange(0, 999999)
        self.min_blur_width.setSingleStep(0.1)
        self.min_blur_width.setValue(0)
        self.min_blur_width.setDecimals(1)
        self.min_blur_width.setKeyboardTracking(False)
        self.min_blur_width.valueChanged.connect(self.render_scene_nocache)
        blur_grid.addWidget(self.min_blur_width, 5, 1, 1, 1)

        vbox.addWidget(blur_groupbox)
        self.blur_methods = {
            points_button: None,
            smooth_button: "smooth",
            convolve_button: "convolve",
            gaussian_button: "gaussian",
            gaussian_iso_button: "gaussian_iso",
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

    def silent_maximum_update(self, value: float) -> None:
        """Change the value of self.maximum in the background."""
        self.maximum.blockSignals(True)
        self.maximum.setValue(value)
        self.maximum.blockSignals(False)

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
    positions : list
        Contains all positions in the animation sequence; each includes
        3 rotations angles and viewport.
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
    window : QMainWindow
        Instance of the rotation window.
    """

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
        self.resize(600, 500)

        self.positions = []
        self.rows = []

        main_layout = QtWidgets.QVBoxLayout(self)

        # Header: current position
        header = QtWidgets.QHBoxLayout()
        cp_label = QtWidgets.QLabel("Current position:")
        cp_label.setToolTip("Current rotation angles in x, y, z (deg).")
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

        # Controls panel (fixed at the bottom)
        controls = QtWidgets.QGridLayout()

        fps_label = QtWidgets.QLabel("FPS: ")
        fps_label.setToolTip("Frames per second used in the animation.")
        controls.addWidget(fps_label, 0, 0)
        self.fps = QtWidgets.QSpinBox()
        self.fps.setValue(30)
        self.fps.setRange(1, 60)
        controls.addWidget(self.fps, 1, 0)

        rs_label = QtWidgets.QLabel("Rotation speed (deg/s): ")
        rs_label.setToolTip(
            "Speed of rotation between positions in the animation."
        )
        controls.addWidget(rs_label, 0, 1)
        self.rot_speed = QtWidgets.QDoubleSpinBox()
        self.rot_speed.setValue(90)
        self.rot_speed.setDecimals(1)
        self.rot_speed.setRange(0.1, 1000)
        controls.addWidget(self.rot_speed, 1, 1)

        self.add = QtWidgets.QPushButton("Add this position")
        self.add.setToolTip(
            "Add the current rotation/view to the animation sequence."
        )
        self.add.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.add.clicked.connect(self.add_position)
        controls.addWidget(self.add, 0, 2)

        self.delete = QtWidgets.QPushButton("Remove last position")
        self.delete.setToolTip(
            "Remove the last position from the animation sequence."
        )
        self.delete.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.delete.clicked.connect(self.delete_position)
        controls.addWidget(self.delete, 1, 2)

        self.build = QtWidgets.QPushButton("Build\nanimation")
        self.build.setToolTip("Create the animation as an .mp4 file.")
        self.build.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.build.clicked.connect(self.build_animation)
        controls.addWidget(self.build, 0, 3)

        self.stay = QtWidgets.QPushButton("Stay in the\n position")
        self.stay.setToolTip("Add the current position again (no movement).")
        self.stay.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.stay.clicked.connect(partial(self.add_position, True))
        controls.addWidget(self.stay, 1, 3)

        main_layout.addLayout(controls)

    def add_position(self, freeze: bool = False) -> None:
        """Add a new position to the animation sequence.

        Parameters
        ----------
        freeze : bool, optional
            True when the new position is the same as the last one,
            i.e., when self.stay is clicked.
        """
        # check that the viewport or angle(s) have changed
        cond1 = cond2 = cond3 = False
        if not freeze and self.positions:
            cond1 = self.window.view_rot.angx == self.positions[-1][0]
            cond2 = self.window.view_rot.angy == self.positions[-1][1]
            cond3 = self.window.view_rot.angz == self.positions[-1][2]
            cond4 = self.window.view_rot.viewport == self.positions[-1][3]
            if all([cond1, cond2, cond3, cond4]):
                return

        # add a new position to the attribute
        self.positions.append(
            [
                self.window.view_rot.angx,
                self.window.view_rot.angy,
                self.window.view_rot.angz,
                self.window.view_rot.viewport,
            ]
        )

        index = len(self.positions) - 1
        grid_row = index

        # build the row's widgets
        p_label = QtWidgets.QLabel(f"- Position {index + 1}: ")
        p_label.setToolTip(
            f"Rotation angles in x, y, z (deg) for position {index + 1}."
        )
        self.rows_layout.addWidget(p_label, grid_row, 0)

        angx = np.round(self.window.view_rot.angx * 180 / np.pi, 1)
        angy = np.round(self.window.view_rot.angy * 180 / np.pi, 1)
        angz = np.round(self.window.view_rot.angz * 180 / np.pi, 1)
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

            # calculate recommended duration
            if not freeze and not all([cond1, cond2, cond3]):
                dx = self.positions[-1][0] - self.positions[-2][0]
                dy = self.positions[-1][1] - self.positions[-2][1]
                dz = self.positions[-1][2] - self.positions[-2][2]
                dmax = np.max(np.abs([dx, dy, dz]))
                rot_speed = self.rot_speed.value() * np.pi / 180
                duration.setValue(dmax / rot_speed)

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
            self.window.view_rot.angx = self.positions[i][0]
            self.window.view_rot.angy = self.positions[i][1]
            self.window.view_rot.angz = self.positions[i][2]
            self.window.view_rot.update_scene(viewport=self.positions[i][3])

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
        if path:
            disp_dlg = self.window.display_settings_dlg
            data_dlg = self.window.window.dataset_dialog
            pixelsize = self.window.window.view.pixelsize
            locs, infos = self.window.view_rot._prepare_locs_for_rendering()
            n_frames = int(self.fps.value() * sum(durations))
            progress = lib.ProgressDialog(
                "Rendering frames", 0, n_frames, self.window
            )
            adjust_display_pixel = disp_dlg.dynamic_disp_px.isChecked()
            render.build_animation(
                path,
                locs,
                infos,
                positions=self.positions,
                durations=durations,
                disp_px_size=disp_dlg.disp_px_size.value(),
                image_size=(
                    self.window.view_rot.width(),
                    self.window.view_rot.height(),
                ),
                blur_method=disp_dlg.blur_methods[
                    disp_dlg.blur_buttongroup.checkedButton()
                ],
                min_blur_width=disp_dlg.min_blur_width.value() / pixelsize,
                contrast=(disp_dlg.minimum.value(), disp_dlg.maximum.value()),
                invert_colors=data_dlg.wbackground.isChecked(),
                single_channel_colormap=disp_dlg.colormap.currentText(),
                colors=self.window.window.view.read_colors(),
                relative_intensities=self.window.window.view.read_relative_intensities(),
                fps=self.fps.value(),
                adjust_pixel_size=adjust_display_pixel,
                progress_callback=progress.set_value,
            )
            progress.close()


class ViewRotation(QtWidgets.QLabel):
    """Display rotated super-resolution datasets.

    Most functions were taken from ``picasso.gui.render.View``.

    ...

    Attributes
    ----------
    angx, angy, angz : float
        Current rotation angle around x, y, and z axes (read/write
        properties; backed by ``self._R``).
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
        Contains the coordinates of points to measure distances
        between them.
    qimage : QImage
        Current image of rendered locs, picks and other drawings.
    _R : scipy.spatial.transform.Rotation
        Source of truth for the current rotation. ``angx/angy/angz``
        are derived from this via ``_codebase_euler``.
    _last_mouse_x, _last_mouse_y : int
        Previous mouse position (Qt coords) during a trackball drag.
    viewport : tuple
        Defines current field of view.
    window : QMainWindow
        Instance of the rotation window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self._R = Rotation.identity()
        self._pan_z = 0.0
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self.locs = []
        self.infos = []
        self.paths = []
        self.group_color = []
        self.x_render_state = False
        self.x_locs = []
        self._size_hint = (512, 512)
        self._mode = "Rotate"
        self._points = []
        self._pan = False
        self.block_x = False
        self.block_y = False
        self.block_z = False
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    # --- rotation angles --- #
    def _codebase_euler(self) -> tuple[float, float, float]:
        e = self._R.as_euler("XYZ")
        return -float(e[0]), float(e[1]), float(e[2])

    @property
    def angx(self) -> float:
        return self._codebase_euler()[0]

    @angx.setter
    def angx(self, value: float) -> None:
        _, b, c = self._codebase_euler()
        self._R = render.rotation_matrix(float(value), b, c)

    @property
    def angy(self) -> float:
        return self._codebase_euler()[1]

    @angy.setter
    def angy(self, value: float) -> None:
        a, _, c = self._codebase_euler()
        self._R = render.rotation_matrix(a, float(value), c)

    @property
    def angz(self) -> float:
        return self._codebase_euler()[2]

    @angz.setter
    def angz(self, value: float) -> None:
        a, b, _ = self._codebase_euler()
        self._R = render.rotation_matrix(a, b, float(value))

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(*self._size_hint)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.update_scene()

    def load_locs(self, update_window=False):
        """Load localizations from a pick in the main window.

        Called when updating rotation window from there or when
        shifting the pick from rotation window.

        Parameters
        ----------
        update_window : bool, optional
            If True, load attributes, such as blur method, from the
            main window.
        """
        fast_render = False  # should locs be reindexed
        w = self.window.window  # main window
        if update_window:
            fast_render = True
            # get pixelsize
            self.pixelsize = w.view.pixelsize
            # update blur and colormap
            b = w.display_settings_dlg.blur_buttongroup.checkedId()
            color = w.display_settings_dlg.colormap.currentText()
            self.window.display_settings_dlg.blur_buttongroup.button(
                b
            ).setChecked(True)
            self.window.display_settings_dlg.colormap.setCurrentText(color)

            # remove measurement points
            self._points = []

            # save the pick information
            self.pick = w.view._picks[0]
            self.pick_shape = w.view._pick_shape
            self.pick_size = w.view._pick_size

            # update view, dataset_dialog for multichannel data and
            # paths
            self.viewport = self.fit_in_view_rotated(get_viewport=True)
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

        # load locs in the pick and their metadata
        n_channels = len(self.paths)
        self.locs = []
        self.infos = []
        for i in range(n_channels):
            # only one pick, take the first element
            temp = w.view.picked_locs(
                i, add_group=False, fast_render=fast_render
            )[0]
            temp["z"] /= self.pixelsize
            # same for lpz if present
            if "lpz" in temp.columns:
                temp["lpz"] /= self.pixelsize
            self.locs.append(temp)
            self.infos.append(w.view.infos[i])

        # shift z positions of locs so that the middle of the dataset is
        # at z = 0
        all_locs_z = np.concatenate([_["z"].to_numpy() for _ in self.locs])
        z_shift = all_locs_z.mean()
        for i in range(n_channels):
            self.locs[i]["z"] -= z_shift

        # assign self.group_color if single channel and group info
        # present
        if len(self.locs) == 1 and "group" in self.locs[0].columns:
            self.group_color = render.get_group_color(self.locs[0])

        # index locs by property for render property mode
        if self.x_render_state and len(self.locs) == 1:
            if self.x_property in self.locs[0].columns:
                self.x_locs = render.split_locs_by_property(
                    self.locs[0],
                    property_name=self.x_property,
                    n_colors=self.x_n_colors,
                    min_value=self.x_min_val,
                    max_value=self.x_max_val,
                )
            else:
                self.x_render_state = False
                self.x_locs = []

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
        ang : tuple, optional
            Rotation angles to be rendered. If None, takes the current
            angles.
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
        # get disp px size, blur method, etc
        kwargs = self.get_render_kwargs(viewport=viewport)
        locs, infos = self._prepare_locs_for_rendering()
        if self._pan_z:
            locs = self._apply_pan_z(locs)
        vmin = self.window.display_settings_dlg.minimum.value()
        vmax = self.window.display_settings_dlg.maximum.value()
        cmap = self.window.display_settings_dlg.colormap.currentText()
        contrast = None if autoscale else (vmin, vmax)
        raw_image = self.image if use_cache else None

        qimage, n_locs, (vmin, vmax), raw_image = render.render_scene(
            locs=locs,
            info=infos,
            **kwargs,
            ang=(self.angx, self.angy, self.angz),
            contrast=contrast,
            invert_colors=self.window.dataset_dialog.wbackground.isChecked(),
            single_channel_colormap=cmap,
            colors=self.window.window.view.read_colors(),
            relative_intensities=self.window.window.view.read_relative_intensities(),
            raw_image_cache=raw_image,
            return_contrast_limits=True,
            return_raw_image=True,
        )
        if cache:
            self.image = raw_image
        self.window.display_settings_dlg.silent_minimum_update(vmin)
        self.window.display_settings_dlg.silent_maximum_update(vmax)
        return qimage

    def update_scene(
        self,
        viewport: tuple[float, float, float, float] | None = None,
        autoscale: bool = False,
        use_cache: bool = False,
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
        """
        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene(viewport, autoscale=autoscale, use_cache=use_cache)

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
        """
        # make sure viewport has the same shape as the main window
        self.viewport = self.adjust_viewport_to_view(viewport)
        if not use_cache:
            self.set_optimal_scalebar(silent=True)
        # render locs
        qimage = self.render_scene(autoscale=autoscale, use_cache=use_cache)
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
                colordisp = self.window.dataset_dialog.colordisp_all[i]
                color = colordisp.palette().color(
                    QtGui.QPalette.ColorRole.Window
                )
                # Convert QColor to RGB tuple (0-255 range)
                color_rgb = (color.red(), color.green(), color.blue())
                channel_colors.append(color_rgb)
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
            image = render.draw_rotation(
                image=image, ang=(self.angx, self.angy, self.angz)
            )
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
        color = (
            QtGui.QColor("yellow")
            if not self.window.dataset_dialog.wbackground.isChecked()
            else QtGui.QColor("red")
        )
        return render.draw_points(
            image=image,
            viewport=self.viewport,
            points=self._points,
            pixelsize=self.window.window.view.pixelsize,
            color=color,
        )

    def rotation_input(self) -> None:
        """Ask the user to input 3 rotation angles manually."""
        angx, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Rotation angle x",
            "Angle x (degrees):",
            0,
            decimals=2,
        )
        if ok:
            angy, ok2 = QtWidgets.QInputDialog.getDouble(
                self,
                "Rotation angle y",
                "Angle y (degrees):",
                0,
                decimals=2,
            )
            if ok2:
                angz, ok3 = QtWidgets.QInputDialog.getDouble(
                    self,
                    "Rotation angle z",
                    "Angle z (degrees):",
                    0,
                    decimals=2,
                )
                if ok3:
                    self.angx += np.pi * angx / 180
                    self.angy += np.pi * angy / 180
                    self.angz += np.pi * angz / 180

        self.update_scene()

    def delete_rotation(self) -> None:
        """Reset rotation and any accumulated pan offset."""
        self._R = Rotation.identity()
        self._pan_z = 0.0
        self.update_scene()

    def fit_in_view_rotated(self, get_viewport: bool = False) -> None:
        """Update viewport to reflect the pick from main window.

        Parameters
        ----------
        get_viewport : bool, optional
            If True, returns the found viewport. Otherwise updates
            scene with the found viewport.
        """
        if self.pick_shape == "Circle":
            d = self.pick_size
            r = d / 2
            x, y = self.pick
            x_min = x - r
            x_max = x + r
            y_min = y - r
            y_max = y + r
        elif self.pick_shape == "Rectangle":
            w = self.pick_size
            (xs, ys), (xe, ye) = self.pick
            X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, w)
            x_min = min(X)
            x_max = max(X)
            y_min = min(Y)
            y_max = max(Y)
        elif self.pick_shape == "Polygon":
            X, Y = lib.get_pick_polygon_corners(self.pick)
            x_min = min(X)
            x_max = max(X)
            y_min = min(Y)
            y_max = max(Y)
        elif self.pick_shape == "Square":
            s = self.pick_size
            x, y = self.pick
            x_min = x - s / 2
            x_max = x + s / 2
            y_min = y - s / 2
            y_max = y + s / 2

        viewport = [(y_min, x_min), (y_max, x_max)]
        if get_viewport:
            return viewport
        else:
            self.viewport = viewport
            self.update_scene()

    def xy_projection(self) -> None:
        """Set angles to 0 to get XY projection."""
        self.angx = 0
        self.angy = 0
        self.angz = 0
        self.update_scene()

    def xz_projection(self) -> None:
        """Set angles to get XZ projection."""
        self.angx = np.pi / 2
        self.angy = 0
        self.angz = 0
        self.update_scene()

    def yz_projection(self) -> None:
        """Set angles to get YZ projection."""
        self.angx = 0
        self.angy = np.pi / 2
        self.angz = 0
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

    def to_left_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan(-SHIFT * render.viewport_width(self.viewport), 0.0)

    def to_right_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan(SHIFT * render.viewport_width(self.viewport), 0.0)

    def to_up_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan(0.0, -SHIFT * render.viewport_height(self.viewport))

    def to_down_rot(self) -> None:
        """Shift pick in the main window."""
        self._arrow_pan(0.0, SHIFT * render.viewport_height(self.viewport))

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
        self.load_locs()  # pick locs in the new viewport
        self.update_scene(viewport=new_viewport)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Block axes if 'X', 'Y' or 'Z' is pressed on the keyboard."""
        if event.key() == 88:  # x
            self.block_x = True
            self.block_y = False
            self.block_z = False
            event.accept()
        elif event.key() == 89:  # y
            self.block_x = False
            self.block_y = True
            self.block_z = False
            event.accept()
        elif event.key() == 90:  # z
            self.block_x = False
            self.block_y = False
            self.block_z = True
            event.accept()
        else:
            event.ignore()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        """Stop blocking axes if 'X', 'Y' or 'Z' is released on the
        keyboard."""
        if event.key() in [88, 89, 90]:  # x, y or z
            self.block_x = False
            self.block_y = False
            self.block_z = False
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define actions taken when moving mouse, for example, rotating
        locs, panning."""
        if self._mode != "Rotate":
            return

        if self._pan:
            self._pan_drag(event)
        else:
            self._rotate_drag(event)

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
        # (turn). With Ctrl, horizontal drag turns and vertical drag spins
        # in the screen plane, matching the previous Ctrl semantics.
        ax = 2 * np.pi * dy_pix / self.height()
        ay = 2 * np.pi * dx_pix / self.width()
        az = 0.0
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            az = ax
            ax = 0.0

        # Axis locks: pressing X/Y/Z constrains rotation to the
        # corresponding *world* axis. Project the screen-frame rotation
        # vector into world frame, zero the components we don't want, and
        # rotate it back.
        if self.block_x or self.block_y or self.block_z:
            v_screen = np.array([ax, ay, az], dtype=float)
            v_world = self._R.inv().apply(v_screen)
            keep = np.array(
                [
                    1.0 if self.block_x else 0.0,
                    1.0 if self.block_y else 0.0,
                    1.0 if self.block_z else 0.0,
                ]
            )
            v_world = v_world * keep
            v_screen = self._R.apply(v_world)
            ax, ay, az = (
                float(v_screen[0]),
                float(v_screen[1]),
                float(v_screen[2]),
            )

        # The codebase's render.rotation_matrix flips the sign of the X
        # rotation relative to scipy's right-hand-rule convention; using
        # it here keeps screen-aligned tilts feeling identical to the old
        # Euler-update behaviour at R = I.
        delta_R = render.rotation_matrix(ax, ay, az)
        self._R = delta_R * self._R
        self.update_scene()

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
        screen_delta = np.array(
            [dx_pix / self.width() * vw, dy_pix / self.height() * vh, 0.0]
        )
        world_delta = self._R.inv().apply(screen_delta)
        # The viewport stores X/Y of the view target; ``_pan_z`` stores Z.
        # Same sign convention as the viewport (subtract on pan). Update
        # ``_pan_z`` before ``pan_relative`` because ``pan_relative`` calls
        # ``update_scene`` internally and we want both deltas in one frame.
        self._pan_z -= float(world_delta[2])
        # pan_relative takes (dy, dx) in *relative* viewport units and
        # subtracts ``dx * vw`` from the viewport X (and similarly for Y),
        # which is exactly ``viewport_center -= world_delta[:2]``.
        self.pan_relative(
            float(world_delta[1]) / vh, float(world_delta[0]) / vw
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define actions taken when pressing mouse buttons, for
        example, starting rotating locs or panning."""
        if self._mode == "Rotate":
            # start rotation
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._last_mouse_x = event.pos().x()
                self._last_mouse_y = event.pos().y()
                event.accept()

            # start panning
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self._pan = True
                self.pan_start_x = event.pos().x()
                self.pan_start_y = event.pos().y()
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Define actions taken when releasing mouse buttons, for
        example, stopping rotating locs or panning, add or delete a measure
        point."""
        if self._mode == "Measure":
            # add point
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                x, y = self.map_to_movie(event.pos())
                self.add_point((x, y))
                event.accept()
            # remove point
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.remove_points()
                event.accept()
            else:
                event.ignore()

        elif self._mode == "Rotate":
            # stop rotation
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                event.accept()
            # stop panning
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self._pan = False
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

    def pan_relative(self, dy: float, dx: float) -> None:
        """Move viewport by a given relative distance.

        Parameters
        ----------
        dy, dx : float
            Relative displacement of the viewport in y or x axis.
        """
        viewport_height, viewport_width = render.viewport_size(self.viewport)
        x_move = dx * viewport_width
        y_move = dy * viewport_height
        x_min = self.viewport[0][1] - x_move
        x_max = self.viewport[1][1] - x_move
        y_min = self.viewport[0][0] - y_move
        y_max = self.viewport[1][0] - y_move
        self.viewport = [(y_min, x_min), (y_max, x_max)]
        self.update_scene()

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
        """Remove all distance measurement points."""
        self._points = []
        self.update_scene()

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
        path, ext = lib.get_save_filename_ext_dialog(
            self,
            "Save image",
            out_path,
            filter="*.png;;*.tif",
            check_ext=check_ext,
        )
        if path:
            self.qimage.save(path)
            self.export_current_view_info(path)
            if not scalebar:
                self.set_optimal_scalebar(force=True)
                scalebar_box.setChecked(True)
                self.update_scene()
                self.qimage.save(os.path.splitext(path)[0] + "_scalebar.png")
                scalebar_box.setChecked(False)
                self.update_scene()

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
            "Blur method": d.blur_methods[d.blur_buttongroup.checkedButton()],
            "Scale bar length (nm)": d.scalebar.value(),
            "Min. blur (nm)": d.min_blur_width.value() / pixelsize,
            "Localizations loaded": self.paths,
            "Colors": colors,
        }
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
        new_viewport = render.zoom_viewport(self.viewport, factor)
        self.update_scene(new_viewport)

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
            Contains blur method, oversampling, viewport and min blur
            width.
        """
        disp_dlg = self.window.display_settings_dlg
        pixelsize = self.window.window.view.pixelsize

        # blur method
        blur_button = disp_dlg.blur_buttongroup.checkedButton()
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
            "blur_method": disp_dlg.blur_methods[blur_button],
            "min_blur_width": float(
                disp_dlg.min_blur_width.value() / pixelsize
            ),
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
            and not self.window.window.display_settings_dlg.render_check.isChecked()
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

    DOCS_URL = "https://picassosr.readthedocs.io/en/latest/render.html#d-rotation-window"  # noqa: E501

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
        save_action = file_menu.addAction("Save rotated localizations")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs_rotated)

        file_menu.addSeparator()
        export_view = file_menu.addAction("Export current view")
        export_view.setShortcut("Ctrl+E")
        export_view.triggered.connect(self.view_rot.export_current_view)
        animation = file_menu.addAction("Build an animation")
        animation.setShortcut("Ctrl+Shift+E")
        animation.triggered.connect(self.animation_dialog.show)
        help_action = file_menu.addAction("Help")
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )

        # menu bar - View
        view_menu = self.menu_bar.addMenu("View")
        display_settings_action = view_menu.addAction("Display settings")
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
        rotation_action = view_menu.addAction("Rotate by angle")
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
        self.setMinimumSize(100, 100)
        self.move(20, 20)

    def move_pick(self, dx: float, dy: float) -> None:
        """Move the pick in the main window by a given amount.

        Parameters
        ----------
        dx, dy : float
            Pick shift in x or y axis (camera pixels).
        """
        if self.view_rot.pick_shape in ["Circle", "Square"]:
            x = self.window.view._picks[0][0]
            y = self.window.view._picks[0][1]
            self.window.view._picks = [(x + dx, y + dy)]  # main window
            self.view_rot.pick = (x + dx, y + dy)  # view rotation
        elif self.view_rot.pick_shape == "Rectangle":
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

        self.window.view.update_scene()  # update scene in main window

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
            pixelsize = self.window.window.view.pixelsize
            if self.view_rot.pick_shape in ["Circle", "Square"]:
                x, y = self.view_rot.pick
                pick = [float(x), float(y)]
                size = self.view_rot.pick_size
            else:  # rectangle
                (ys, xs), (ye, xe) = self.view_rot.pick
                pick = [[float(ys), float(xs)], [float(ye), float(xe)]]
                size = self.view_rot.pick_size
            new_info = [
                {
                    "Generated by": f"Picasso v{__version__} Render 3D",
                    "Pick": pick,
                    "Pick shape": self.view_rot.pick_shape,
                    "Pick size (nm)": size * pixelsize,
                    "angx": self.view_rot.angx,
                    "angy": self.view_rot.angy,
                    "angz": self.view_rot.angz,
                }
            ]

            # combine all channels
            if channel is (len(self.view_rot.paths) + 1):
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
                    # combine locs from all channels
                    all_locs = pd.concat(
                        self.window.view.all_locs,
                        ignore_index=True,
                    )
                    all_locs.sort_values(
                        kind="quicksort",
                        by="frame",
                        inplace=True,
                    )
                    info = self.view_rot.infos[0] + new_info
                    io.save_locs(path, all_locs, info)
            # save all channels one by one
            elif channel is (len(self.view_rot.paths)):  # all channels
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
                            out_path, self.window.view.all_locs[channel], info
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
                    filter="*hdf5",
                    check_ext=".yaml",
                )
                info = self.view_rot.infos[channel] + new_info
                io.save_locs(path, self.window.view.all_locs[channel], info)

    def update_scene(self) -> None:
        """Update the scene in ViewRotation."""
        self.view_rot.update_scene()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Close all children dialogs and self."""
        self.display_settings_dlg.close()
        self.animation_dialog.close()
        QtWidgets.QMainWindow.closeEvent(self, event)
