"""
    picasso/rotation
    ~~~~~~~~~~~~~~~~~~~~

    Rotation window classes and functions.
    Extension of Picasso: Render to visualize 3D data.
    Many functions are copied from gui.render.View to avoid circular import

    :author: Rafal Kowalewski, 2021-2022
    :copyright: Copyright (c) 2021 Jungmann Lab, MPI of Biochemistry
"""

import os
import colorsys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import imageio
from PyQt5 import QtCore, QtGui, QtWidgets

from numpy.lib.recfunctions import stack_arrays

from .. import io, render
from ..lib import ProgressDialog


DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5
N_GROUP_COLORS = 8
SHIFT = 0.1
ZOOM = 9 / 7


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

class DisplaySettingsRotationDialog(QtWidgets.QDialog):
    """
    A class to change display settings, e.g., display pixel size, 
    contrast and blur.

    Very similar to its counterpart in gui/render.py but some functions
    were deleted.

    ...

    Attributes
    ----------
    blur_buttongroup : QButtonGroup
        contains available localization blur methods
    colormap : QComboBox
        contains strings with available colormaps (single channel only)
    dynamic_disp_px : QCheckBox
        tick to automatically adjust to current window size when zooming.
    maximum : QDoubleSpinBox
        defines at which number of localizations per super-resolution
        pixel the maximum color of the colormap should be applied
    min_blur_width : QDoubleSpinBox
        contains the minimum blur for each localization (pixels)
    minimum : QDoubleSpinBox
        defines at which number of localizations per super-resolution
        pixel the minimum color of the colormap should be applied
    disp_px_size : QDoubleSpinBox
        contains the size of super-resolution pixels in nm
    scalebar : QSpinBox
        contains the scale bar's length (nm)
    scalebar_groupbox : QGroupBox
        group with options for customizing scale bar, tick to display
    scalebar_text : QCheckBox
        tick to display scale bar's length (nm)
    _silent_disp_px_update : boolean
        True if update display pixel size in background

    Methods
    -------
    on_disp_px_changed(value)
        Sets new display pixel size, updates contrast and updates scene 
        in the main window
    render_scene(*args, **kwargs)
        Updates scene in the rotation window and gives warning if 
        needed
    set_dynamic_disp_px(state)
        Updates scene if dynamic disp_px is checked
    set_disp_px_silently(disp_px_size)
        Changes the value of display pixel size in the background
    silent_maximum_update(value)
        Changes the value of self.maximum in the background
    silent_minimum_update(value)
        Changes the value of self.minimum in the background 
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

        # contrast
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
        self.minimum.valueChanged.connect(self.render_scene)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtWidgets.QLabel("Max. Density:")
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = QtWidgets.QDoubleSpinBox()
        self.maximum.setRange(0, 999999)
        self.maximum.setSingleStep(5)
        self.maximum.setValue(100)
        self.maximum.setDecimals(6)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.render_scene)
        contrast_grid.addWidget(self.maximum, 1, 1)
        contrast_grid.addWidget(QtWidgets.QLabel("Colormap:"), 2, 0)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(plt.colormaps())
        contrast_grid.addWidget(self.colormap, 2, 1)
        self.colormap.currentIndexChanged.connect(self.render_scene)
        
        # blur
        blur_groupbox = QtWidgets.QGroupBox("Blur")
        blur_grid = QtWidgets.QGridLayout(blur_groupbox)
        self.blur_buttongroup = QtWidgets.QButtonGroup()
        points_button = QtWidgets.QRadioButton("None")
        self.blur_buttongroup.addButton(points_button)
        smooth_button = QtWidgets.QRadioButton("One-Pixel-Blur")
        self.blur_buttongroup.addButton(smooth_button)
        convolve_button = QtWidgets.QRadioButton(
            "Global Localization Precision"
        )
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

        # camera
        # self.pixelsize = QtWidgets.QDoubleSpinBox()
        # self.pixelsize.setValue(130)
        
        # scalebar
        self.scalebar_groupbox = QtWidgets.QGroupBox("Scale Bar")
        self.scalebar_groupbox.setCheckable(True)
        self.scalebar_groupbox.setChecked(False)
        self.scalebar_groupbox.toggled.connect(self.render_scene)
        vbox.addWidget(self.scalebar_groupbox)
        scalebar_grid = QtWidgets.QGridLayout(self.scalebar_groupbox)
        scalebar_grid.addWidget(
            QtWidgets.QLabel("Scale Bar Length (nm):"), 0, 0
        )
        self.scalebar = QtWidgets.QSpinBox()
        self.scalebar.setRange(1, 100000)
        self.scalebar.setValue(500)
        self.scalebar.setKeyboardTracking(False)
        self.scalebar.valueChanged.connect(self.render_scene)
        scalebar_grid.addWidget(self.scalebar, 0, 1)
        self.scalebar_text = QtWidgets.QCheckBox("Print scale bar length")
        self.scalebar_text.stateChanged.connect(self.render_scene)
        scalebar_grid.addWidget(self.scalebar_text, 1, 0)

        self._silent_disp_px_update = False

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
            self.window.view_rot.update_scene()

    def set_disp_px_silently(self, disp_px_size):
        """ 
        Changes the value of display pixel size in the background. 
        """

        self._silent_disp_px_update = True
        self.disp_px_size.setValue(disp_px_size)
        self._silent_disp_px_update = False

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
        """ Updates scene in the rotation window. """

        self.window.view_rot.update_scene()

    def set_dynamic_disp_px(self, state):
        """ Updates scene if dynamic display pixel size is checked. """

        if state:
            self.window.view_rot.update_scene()

class AnimationDialog(QtWidgets.QDialog):
    """
    A class with a dialog to prepare 3D animations.

    ...

    Attributes
    ----------
    add : QPushButton
        click to add the current view to the animation sequence
    build : QPushButton
        click to create an animation
    count : int
        counts how many positions are currently saved
    current_pos : QLabel
        shows rotation angles around x, y and z axes in degrees
    delete : QPushButton
        click to delete the last saved position
    durations : list
        contains QDoubleSpinBoxes with durations (seconds) of each
        step in the animation sequence
    fps : QSpinBox
        contains frames per second used in the animation
    layout : QGridLayout
        widget storing positions of other widgets in the Dialog
    positions : list
        contains all positions in the animation sequence; each includes
        3 rotations angles and viewport
    positions_labels : list
        displays rotation angles for each saved position in the 
        animation sequence
    show_positions : boolean
        contains buttons to display a given position that has been
        saved in the animation sequence
    stay : QPushButton
        click to copy the exact same position (so that locs do not
        move)
    rot_speed : QDoublesSpinBox
        contains the default rotation speed calculated when adding a
        position with different angles
    window : QMainWindow
        instance of the rotation window
    
    Methods
    -------
    add_position()
        Adds a new position to the animation sequence
    build_animation()
        Creates an animation as an.mp4 file
    delete_position()
        Deletes the last position from the animation sequence
    retrieve_position(i)
        Moves the view to the i'th position
    """

    def __init__(self, window):
        super().__init__(window)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "render.ico")
        icon = QtGui.QIcon(icon_path)
        self.icon = icon
        self.setWindowIcon(icon)

        self.window = window
        self.setWindowTitle("Build an animation")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.positions = []
        self.positions_labels = []
        self.durations = []
        self.show_positions = []
        self.count = 0
        self.layout.addWidget(QtWidgets.QLabel("Current position: "), 0, 0)
        angx = np.round(self.window.view_rot.angx * 180 / np.pi, 1)
        angy = np.round(self.window.view_rot.angy * 180 / np.pi, 1)
        angz = np.round(self.window.view_rot.angz * 180 / np.pi, 1)
        self.current_pos = QtWidgets.QLabel("{}, {}, {}".format(
            angx, angy, angz
        ))
        self.layout.addWidget(self.current_pos, 0, 1)

        for i in range(1, 11):
            self.layout.addWidget(
                QtWidgets.QLabel("- Position {}: ".format(i)), i, 0
            )

            show_position = QtWidgets.QPushButton("Show position")
            show_position.setFocusPolicy(QtCore.Qt.NoFocus)
            show_position.clicked.connect(
                partial(self.retrieve_position, i-1)
            )
            self.show_positions.append(show_position)
            self.layout.addWidget(show_position, i, 2)
            if i > 1:
                self.layout.addWidget(
                    QtWidgets.QLabel("Duration [s]: "), i, 3
                )
                duration = QtWidgets.QDoubleSpinBox()
                duration.setRange(0.01, 10)
                duration.setValue(1)
                duration.setDecimals(2)
                self.durations.append(duration)
                self.layout.addWidget(duration, i, 4)

        self.layout.addWidget(QtWidgets.QLabel("FPS: "), 11, 0)
        self.fps = QtWidgets.QSpinBox()
        self.fps.setValue(30)
        self.fps.setRange(1, 60)
        self.layout.addWidget(self.fps, 12, 0)

        self.layout.addWidget(
            QtWidgets.QLabel("Rotation speed [deg/s]: "), 11, 1
        )
        self.rot_speed = QtWidgets.QDoubleSpinBox()
        self.rot_speed.setValue(90)
        self.rot_speed.setDecimals(1)
        self.rot_speed.setRange(0.1, 1000)
        self.layout.addWidget(self.rot_speed, 12, 1)

        self.add = QtWidgets.QPushButton("+")
        self.add.setFocusPolicy(QtCore.Qt.NoFocus)
        self.add.clicked.connect(self.add_position)
        self.layout.addWidget(self.add, 11, 2)

        self.delete = QtWidgets.QPushButton("-")
        self.delete.setFocusPolicy(QtCore.Qt.NoFocus)
        self.delete.clicked.connect(self.delete_position)
        self.layout.addWidget(self.delete, 12, 2)

        self.build = QtWidgets.QPushButton("Build\nanimation")
        self.build.setFocusPolicy(QtCore.Qt.NoFocus)
        self.build.clicked.connect(self.build_animation)
        self.layout.addWidget(self.build, 11, 3)

        self.stay = QtWidgets.QPushButton("Stay in the\n position")
        self.stay.setFocusPolicy(QtCore.Qt.NoFocus)
        self.stay.clicked.connect(partial(self.add_position, True))
        self.layout.addWidget(self.stay, 12, 3)
    
    def add_position(self, freeze=False):
        """
        Adds a new position to the animation sequence.

        Parameters
        ----------
        freeze : boolean
            True when the new position is the same as the last one
            (i.e. when self.stay is clicked)
        """

        # more than 10 positions are not allowed
        if self.count == 10:
            raise ValueError("More positions are not supported")

        # check that the viewport or angle(s) have changed
        if not freeze:
            if self.count > 0:
                cond1 = self.window.view_rot.angx == self.positions[-1][0]
                cond2 = self.window.view_rot.angy == self.positions[-1][1]
                cond3 = self.window.view_rot.angz == self.positions[-1][2]
                cond4 = self.window.view_rot.viewport == self.positions[-1][3]
                if all([cond1, cond2, cond3, cond4]):
                    return

        # add a new position to the attribute
        self.positions.append([
                self.window.view_rot.angx,
                self.window.view_rot.angy,
                self.window.view_rot.angz,
                self.window.view_rot.viewport,
            ])

        # display the new position
        angx = np.round(self.window.view_rot.angx * 180 / np.pi, 1)
        angy = np.round(self.window.view_rot.angy * 180 / np.pi, 1)
        angz = np.round(self.window.view_rot.angz * 180 / np.pi, 1)
        self.positions_labels.append(QtWidgets.QLabel(
            "{}, {}, {}".format(angx, angy, angz)
        ))
        self.layout.addWidget(self.positions_labels[-1], self.count + 1, 1)

        # calculate recommended duration
        if self.count > 0: # only if it's at least the second position
            if not freeze:
                if not all([cond1, cond2, cond3]):
                    dx = self.positions[-1][0] - self.positions[-2][0]
                    dy = self.positions[-1][1] - self.positions[-2][1]
                    dz = self.positions[-1][2] - self.positions[-2][2]
                    dmax = np.max(np.abs([dx, dy, dz]))
                    rot_speed = self.rot_speed.value() * np.pi / 180
                    dur = dmax / rot_speed
                    self.durations[self.count-1].setValue(dur)

        self.count += 1

    def delete_position(self):
        """ Deletes the last position from the animation sequence. """

        if self.count > 0:
            del self.positions[-1]
            self.layout.removeWidget(self.positions_labels[-1])
            del self.positions_labels[-1]
            self.count -= 1

    def retrieve_position(self, i):
        """
        Moves the view to the specified position.

        Parameters
        ----------
        i : idx
            Index of the position to be displayed
        """

        if i <= len(self.positions) - 1:
            self.window.view_rot.angx = self.positions[i][0]
            self.window.view_rot.angy = self.positions[i][1]
            self.window.view_rot.angz = self.positions[i][2]
            self.window.view_rot.update_scene(viewport=self.positions[i][3])

    def build_animation(self):
        """
        Creates an animation as an .mp4 file using the positions from
        the animation sequence.
        """

        # find the number of frames between each position
        n_frames = [0]
        for i in range(len(self.positions) - 1):
            n_frames.append(int(self.fps.value() * self.durations[i].value()))

        # find rotation angles and viewport for each frame
        angx = np.zeros(np.sum(n_frames))
        angy = np.zeros(np.sum(n_frames))
        angz = np.zeros(np.sum(n_frames))
        ymin = np.zeros(np.sum(n_frames))
        xmin = np.zeros(np.sum(n_frames))
        ymax = np.zeros(np.sum(n_frames))
        xmax = np.zeros(np.sum(n_frames))

        for i in range(len(self.positions) - 1):
            idx_low = np.sum(n_frames[:i+1])
            idx_high = np.sum(n_frames[:i+2])

            # angles
            x1 = self.positions[i][0]
            x2 = self.positions[i+1][0]
            y1 = self.positions[i][1]
            y2 = self.positions[i+1][1]
            z1 = self.positions[i][2]
            z2 = self.positions[i+1][2]
            angx[idx_low:idx_high] = np.linspace(x1, x2, n_frames[i+1])
            angy[idx_low:idx_high] = np.linspace(y1, y2, n_frames[i+1])
            angz[idx_low:idx_high] = np.linspace(z1, z2, n_frames[i+1])

            # viewport
            vp1 = self.positions[i][3]
            vp2 = self.positions[i+1][3]
            ymin[idx_low:idx_high] = np.linspace(
                vp1[0][0], vp2[0][0], n_frames[i+1]
            )
            xmin[idx_low:idx_high] = np.linspace(
                vp1[0][1], vp2[0][1], n_frames[i+1]
            )
            ymax[idx_low:idx_high] = np.linspace(
                vp1[1][0], vp2[1][0], n_frames[i+1]
            )
            xmax[idx_low:idx_high] = np.linspace(
                vp1[1][1], vp2[1][1], n_frames[i+1]
            )

        # get save file name
        out_path = self.window.view_rot.paths[0].replace(
            ".hdf5", "_video.mp4"
        )
        name, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save animation", out_path, filter="*.mp4"
        )
        if name:
            # width and height for building the animation; must be even
            # as many video players do not accept it otherwise
            width = self.window.view_rot.width()
            height = self.window.view_rot.height()
            if width % 2 == 1:
                width += 1
            if height % 2 == 1:
                height += 1

            # render all frames and save in RAM
            video_writer = imageio.get_writer(name, fps=self.fps.value())
            progress = ProgressDialog(
                "Rendering frames", 0, len(angx), self.window
            )
            progress.set_value(0)
            for i in range(len(angx)):
                qimage = self.window.view_rot.render_scene(
                    viewport=[(ymin[i], xmin[i]), (ymax[i], xmax[i])],
                    ang=(angx[i], angy[i], angz[i]),
                    animation=True,
                )
                qimage = qimage.scaled(width, height)

                # convert to a np.array and append
                ptr = qimage.bits()
                ptr.setsize(height * width * 4)
                frame = np.frombuffer(ptr, np.uint8).reshape((width, height, 4))
                frame = frame[:, :, :3]
                frame = frame[:, :, ::-1] # invert RGB to BGR

                video_writer.append_data(frame)
                progress.set_value(i+1)
            progress.close()
            video_writer.close()
            

class ViewRotation(QtWidgets.QLabel):
    """
    A class to displayed rotated super-resolution datasets.

    Most functions were taken from picass/gui/render.py's View class.

    ...

    Attributes
    ----------
    angx : float
        current rotation angle around x axis
    angy : float
        current rotation angle around y axis
    angz : float
        current rotation angle around z axis
    block_x: boolean
        True if rotate only around x axis
    block_y: boolean
        True if rotate only around y axis
    block_z: boolean
        True if rotate only around z axis
    display_angles : boolean
        True if current rotation angles are to be displayed
    display_legend : boolean
        True if legend is to be displayed
    display_rotation : boolean
        True if reference axes is to be displayed
    group_color : np.array
        important for single channel data with group info (picked or
        clustered locs); contains an integer index for each loc 
        defining its color
    infos : list
        contains a dictionary with metadata for each channel
    locs : list
        contains a np.recarray with localizations for each channel
    _mode : str
        defines current mode (zoom, pick or measure); important for 
        mouseEvents
    _pan : boolean
        indicates if image is currently panned
    pan_start_x : float
        x coordinate of panning's starting position
    pan_start_y : float
        y coordinate of panning's starting position
    pixmap : QPixmap
        Pixmap currently displayed
    _points : list
        contains the coordinates of points to measure distances
        between them
    qimage : QImage
        current image of rendered locs, picks and other drawings
    _rotation : list
        contains mouse's positions on the screen while rotating
    viewport : tuple
        defines current field of view
    window : QMainWindow
        instance of the rotation window

    Methods
    -------
    add_angles_view()
        Shows/Hides current rotation in angles.
    add_legend()
        Shows/Hides legend
    add_rotation_view()
        Shows/Hides rotation axes icon
    add_point()
        Adds a point at a given position for measuring distances
    adjust_viewport_view(viewport)
        Adds space to viewport to match self.window's aspect ratio
    delete_rotation()
        Resets rotation angles
    display_pixels_per_viewport_pixels()
        Returns optimal oversampling
    draw_legend(image)
        Draws a legend for multichannel data
    draw_points(image)
        Draws points and lines and distances between them onto image
    draw_rotation(image)
        Draws a small 3 axes icon that rotates with locs
    draw_rotation_angles(image)
        Draws text displaying current rotation angles in degrees 
    draw_scalebar(image)
        Draws a scalebar
    draw_scene(viewport)
        Renders localizations in the given viewport and draws legend,
        rotation, etc
    export_current_view()
        Exports current view as .png or .tif
    fit_in_view_rotated()
        Updates viewport to reflect the pick from main window
    get_render_kwargs()
        Returns a dictionary to be used for the keyword arguments of 
        render.render
    load_locs()
        Loads localizations from a pick in the main window
    map_to_movie()
        Converts coordinates from display units to camera units
    map_to_view()
        Converts coordinates from camera units to display units
    mouseMoveEvent(event)
        Defines actions taken when moving mouse
    mousePressEvent(event)
        Defines actions taken when pressing mouse buttons
    mouseReleaseEvent(event)
        Defines actions taken when releasing mouse buttons
    pan_relative(dy, dx)
        Moves viewport by a given relative distance
    remove_points(position)
        Removes all distance measurement pointss
    render_multi_channel()
        Renders and paints multichannel localizations
    render_scene()
        Returns QImage with rendered localizations
    render_single_channel()
        Renders single channel localizations
    rotation_input()
        Asks user to input 3 rotation angles numerically
    scale_contrast(image)
        Scales image based on contrast values from Display Settings
        Dialog
    set_mode(action)
        Sets self._mode for QMouseEvents
    shift_viewport(dx, dy)
        Moves viewport by dx and dy
    to_down_rot()
        Called on pressing down arrow; shifts pick in the main window
    to_left_rot()
        Called on pressing left arrow; shifts pick in the main window
    to_right_rot()
        Called on pressing right arrow; shifts pick in the main window
    to_up_rot()
        Called on pressing up arrow; shifts pick in the main window
    to_8bit(image)
        Converts image to 8 bit ready to convert to QImage
    update_scene()
        Updates the view of rendered locs
    viewport_center()
        Finds viewport's center (pixels)
    viewport_height()
        Finds viewport's height (pixels)
    viewport_size()
        Finds viewport's height and width (pixels)
    viewport_width()
        Finds viewport's width (pixels)
    zoom(factor)
        Changes zoom by factor by changing viewport
    zoom_in()
        Zooms in by a constant factor
    zoom_out()
        Zooms out by a constant factor
    """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.angx = 0
        self.angy = 0
        self.angz = 0
        self.locs = []
        self.infos = []
        self.paths = []
        self.group_color = []
        self._size_hint = (512, 512)
        self._mode = "Rotate"
        self._rotation = []
        self._points = []
        self._pan = False
        self.display_legend = False
        self.display_rotation = True
        self.display_angles = False
        self.block_x = False
        self.block_y = False
        self.block_z = False
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

    def sizeHint(self):
        return QtCore.QSize(*self._size_hint)

    def resizeEvent(self, event):
        self.update_scene()

    def load_locs(self, update_window=False):
        """
        Loads localizations from a pick in the main window.

        Called when updating rotation window from there or when 
        shifting the pick from rotation window.

        Parameters
        ----------
        update_window : boolean (default=False)
            If True, load attributes, such as blur method, from the 
            main window
        """

        fast_render = False # should locs be reindexed
        if update_window:
            fast_render = True
            # get pixelsize
            self.pixelsize = (
                self.window.window.display_settings_dlg.pixelsize.value()
            )
            # update blur and colormap
            blur = (
                self.window.window.display_settings_dlg.blur_buttongroup. \
                    checkedId()
            )
            color = (
                self.window.window.display_settings_dlg.colormap.currentText()
            )
            self.window.display_settings_dlg.blur_buttongroup.button(
                blur
            ).setChecked(True)
            self.window.display_settings_dlg.colormap.setCurrentText(color)

            # save the pick information
            self.pick = self.window.window.view._picks[0]
            self.pick_shape = self.window.window.view._pick_shape
            if self.pick_shape == "Circle":
                self.pick_size = self.window.window. \
                    tools_settings_dialog.pick_diameter.value()
            else:
                self.pick_size = self.window.window. \
                    tools_settings_dialog.pick_width.value()

            # update view, dataset_dialog for multichannel data and 
            # paths
            self.viewport = self.fit_in_view_rotated(get_viewport=True)
            self.window.dataset_dialog = self.window.window.dataset_dialog
            self.paths = self.window.window.view.locs_paths

        # load locs in the pick and their metadata
        n_channels = len(self.paths)
        self.locs = []
        self.infos = []
        for i in range(n_channels):
            temp = self.window.window.view.picked_locs(
                i, add_group=False, fast_render=fast_render
            )[0] # only one pick, take the first element
            temp.z /= self.pixelsize
            self.locs.append(temp)
            self.infos.append(self.window.window.view.infos[i])

        # assign self.group_color if single channel and group info
        # present
        if len(self.locs) == 1 and hasattr(self.locs[0], "group"):
            self.group_color = self.window.window.view.get_group_color(
                self.locs[0]
            )

    def render_scene(
        self, 
        viewport=None, 
        ang=None, 
        animation=False,
        autoscale=False,
    ):
        """
        Returns QImage with rendered localizations.

        Parameters
        ----------
        viewport : tuple (default=None)
            Viewport to be rendered. If None, takes current viewport
        ang : tuple (default=None)
            Rotation angles to be rendered. If None, takes the current
            angles
        animation : boolean (default=False)
            If True, scenes are rendered for building an animation
        autoscale : boolean
            If True, optimally adjust contrast

        Returns
        -------
        QImage
            Shows rendered locs; 8 bit, scaled
        """

        # get oversampling, blur method, etc
        kwargs = self.get_render_kwargs(
            viewport=viewport, animation=animation
        )
        # render single or multi channel data
        n_channels = len(self.locs)
        if n_channels == 1:
            self.render_single_channel(kwargs, ang=ang, autoscale=autoscale)
        else:
            self.render_multi_channel(kwargs, ang=ang, autoscale=autoscale)
        # add alpha channel (no transparency)
        self._bgra[:, :, 3].fill(255)
        # build QImage
        Y, X = self._bgra.shape[:2]
        qimage = QtGui.QImage(
            self._bgra.data, X, Y, QtGui.QImage.Format_RGB32
        )
        return qimage

    def render_multi_channel(
        self, 
        kwargs, 
        locs=None, 
        ang=None, 
        autoscale=False,
    ):
        """
        Renders and paints multichannel localizations. 

        Also used when other multi-color data is used (clustered or 
        picked locs)

        Parameters
        ----------
        kwargs : dict
            Contains blur method, etc. See self.get_render_kwargs
        locs : np.recarray (default=None)
            Locs to be rendered. If None, self.locs is used
        ang : tuple (default=None)
            Rotation angles to be rendered. If None, takes the current
            angles 
        autoscale : boolean
            If True, optimally adjust contrast

        Returns
        -------
        np.array
            8 bit array with 4 channels (rgb and alpha)
        """

        # get locs to render
        if locs is None:
            locs = self.locs

        # other parameters for rendering
        n_channels = len(locs)
        colors = get_colors(n_channels) # automatic colors

        if ang is None: # no build animation
            renderings = [
                render.render(
                    _, **kwargs, 
                    ang=(self.angx, self.angy, self.angz), 
                ) for _ in locs
            ]
        else: # build animation
            renderings = [
                render.render(
                    _, **kwargs, 
                    ang=ang, 
                ) for _ in locs
            ]
        n_locs = sum([_[0] for _ in renderings])
        image = np.array([_[1] for _ in renderings])
        self.n_locs = n_locs
        self.image = image

        # adjust contrast
        image = self.scale_contrast(image, autoscale=autoscale)

        # set up four channel output
        Y, X = image.shape[1:]
        bgra = np.zeros((Y, X, 4), dtype=np.float32)

        # color each channel one by one
        for i in range(len(self.locs)):
            # change colors if not automatic coloring
            if not self.window.dataset_dialog.auto_colors.isChecked():
                # get color from Dataset Dialog
                color = self.window.dataset_dialog.colorselection[i]
                color = color.currentText()
                # if default color
                if color in self.window.dataset_dialog.default_colors:
                    index = self.window.dataset_dialog.default_colors.index(
                        color
                        )
                    colors[i] = tuple(self.window.dataset_dialog.rgbf[index])
                # if hexadecimal is given
                elif is_hexadecimal(color):
                    colorstring = color.lstrip("#")
                    rgbval = tuple(
                        int(colorstring[i: i + 2], 16) / 255 for i in (0, 2, 4)
                    )
                    colors[i] = rgbval
                else:
                    c = self.window.dataset_dialog.checks[i].text()
                    warning = (
                        f"The color selection not recognnised in the channel" 
                        " {c}.  Please choose one of the options provided or "
                        " type the hexadecimal code for your color of choice, "
                        "starting with '#', e.g.  '#ffcdff' for pink."
                    )
                    QtWidgets.QMessageBox.information(self, "Warning", warning)
                    break

            # reverse colors if white background
            if self.window.dataset_dialog.wbackground.isChecked():
                tempcolor = colors[i]
                inverted = tuple([1 - _ for _ in tempcolor])
                colors[i] = inverted
            
            # adjust for relative intensity from Dataset Dialog
            iscale = self.window.dataset_dialog.intensitysettings[i].value()
            image[i] = iscale * image[i]

            # don't display if channel unchecked in Dataset Dialog
            if not self.window.dataset_dialog.checks[i].isChecked():
                image[i] = 0 * image[i]

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

    def render_single_channel(self, kwargs, ang=None, autoscale=False):
        """
        Renders single channel localizations. 

        Calls render_multi_channel in case of clustered or picked locs,
        rendering by property)

        Parameters
        ----------
        kwargs : dict
            Contains blur method, etc. See self.get_render_kwargs
        ang : tuple (default=None)
            Rotation angles to be rendered. If None, takes the current
            angles 
        autoscale : boolean (default=False)
            True if optimally adjust contrast    

        Returns
        -------
        np.array
            8 bit array with 4 channels (rgb and alpha)
        """

        # get np.recarray
        locs = self.locs[0]

        # if clustered or picked locs
        if hasattr(locs, "group"):
            locs = [
                locs[self.group_color == _] for _ in range(N_GROUP_COLORS)
            ]
            return self.render_multi_channel(
                kwargs, locs=locs, ang=ang, autoscale=autoscale
            )

        if ang is None: # if not build animation
            n_locs, image = render.render(
                locs, 
                **kwargs, 
                info=self.infos[0], 
                ang=(self.angx, self.angy, self.angz), 
            )
        else: # if build animation
            n_locs, image = render.render(
                locs, 
                **kwargs, 
                info=self.infos[0], 
                ang=ang, 
            )
        self.n_locs = n_locs
        self.image = image

        # adjust contrast and convert to 8 bits
        image = self.scale_contrast(image, autoscale=autoscale)
        image = self.to_8bit(image)
        
        # paint locs using the colormap of choice (Display Settings
        # Dialog)
        cmap = self.window.display_settings_dlg.colormap.currentText()
        cmap = np.uint8(np.round(255 * plt.get_cmap(cmap)(np.arange(256))))
        
        # return a 4 channel (rgb and alpha) array
        Y, X = image.shape
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        return self._bgra

    def update_scene(self, viewport=None, autoscale=False):
        """
        Updates the view of rendered locs.

        Parameters
        ----------
        viewport : tuple (default=None)
            Viewport to be rendered. If None self.viewport is taken
        autoscale : boolean (default=False)
            True if optimally adjust contrast
        """

        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene(viewport, autoscale=autoscale)

        # update current position in the animation dialog
        angx = np.round(self.angx * 180 / np.pi, 1)
        angy = np.round(self.angy * 180 / np.pi, 1)
        angz = np.round(self.angz * 180 / np.pi, 1)
        self.window.animation_dialog.current_pos.setText(
            "{}, {}, {}".format(angx, angy, angz)
        )

    def draw_scene(self, viewport, autoscale=False):
        """
        Renders localizations in the given viewport and draws legend,
        rotation, etc.

        Parameters
        ----------
        viewport : tuple
            Viewport defining the rendered FOV
        autoscale : boolean (default=False)
            True if contrast should be optimally adjusted
        """

        # make sure viewport has the same shape as the main window
        self.viewport = self.adjust_viewport_to_view(viewport)
        # render locs
        qimage = self.render_scene(autoscale=autoscale)
        # scale image's size to the window
        self.qimage = qimage.scaled(
            self.width(),
            self.height(),
            QtCore.Qt.KeepAspectRatioByExpanding,
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
            pixelsize = self.window.window.display_settings_dlg.pixelsize.value()
            scalebar = self.window.display_settings_dlg.scalebar.value()
            length_camerapxl = scalebar / pixelsize
            length_displaypxl = int(
                round(self.width() * length_camerapxl / self.viewport_width())
            )
            height = 10
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))
            if self.window.dataset_dialog.wbackground.isChecked():
                painter.setBrush(QtGui.QBrush(QtGui.QColor("black")))
            x = self.width() - length_displaypxl - 35
            y = self.height() - height - 20
            painter.drawRect(x, y, length_displaypxl + 0, height + 0)
            if self.window.display_settings_dlg.scalebar_text.isChecked():
                font = painter.font()
                font.setPixelSize(20)
                painter.setFont(font)
                painter.setPen(QtGui.QColor("white"))
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

        if self.display_legend:
            n_channels = len(self.locs)
            painter = QtGui.QPainter(image)
            x = 12
            y = 20
            dy = 20
            for i in range(n_channels):
                if self.window.dataset_dialog.checks[i].isChecked():
                    palette = self.window.dataset_dialog.colordisp_all[i].palette()
                    color = palette.color(QtGui.QPalette.Window)
                    painter.setPen(QtGui.QColor(color))
                    font = painter.font()
                    font.setPixelSize(16)
                    painter.setFont(font)
                    text = self.window.dataset_dialog.checks[i].text()
                    painter.drawText(QtCore.QPoint(x, y), text)
                    y += dy
        return image

    def draw_rotation(self, image):
        """
        Draws a small 3 axes icon that rotates with locs.

        Displayed in the bottom left corner.

        Parameters
        ----------
        image : QImage
            Image containing rendered localizations

        Returns
        -------
        QImage
            Image with the drawn legend
        """

        if self.display_rotation:
            painter = QtGui.QPainter(image)
            length = 30
            x = 50
            y = self.height() - 50
            center = QtCore.QPoint(x, y)

            #set the ends of the x line
            xx = length
            xy = 0
            xz = 0

            #set the ends of the y line
            yx = 0
            yy = length
            yz = 0

            #set the ends of the z line
            zx = 0
            zy = 0
            zz = length

            #rotate these points
            coordinates = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
            R = render.rotation_matrix(self.angx, self.angy, self.angz)
            coordinates = R.apply(coordinates).astype(int)
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

            #set the points at the ends of the lines
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
        return image

    def draw_rotation_angles(self, image):
        """ 
        Draws text displaying current rotation angles in degrees. 
        """

        if self.display_angles:        
            image = image.copy()
            [angx, angy, angz] = [
                int(np.round(_ * 180 / np.pi, 0)) 
                for _ in [self.angx, self.angy, self.angz]
            ]
            text = f"{angx} {angy} {angz}"
            x = self.width() - len(text) * 8 - 10
            y = self.height() - 20      
            painter = QtGui.QPainter(image)
            font = painter.font()
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("white"))
            if self.window.dataset_dialog.wbackground.isChecked():
                painter.setPen(QtGui.QColor("black"))
            painter.drawText(QtCore.QPoint(x, y), text)
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
        d = 20
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor("yellow"))
        if self.window.dataset_dialog.wbackground.isChecked():
            painter.setPen(QtGui.QColor("red"))
        cx = []
        cy = []
        ox = []
        oy = []
        oldpoint = []
        pixelsize = self.window.window.display_settings_dlg.pixelsize.value()
        for point in self._points:
            if oldpoint != []:
                ox, oy = self.map_to_view(*oldpoint)
            cx, cy = self.map_to_view(*point)
            painter.drawPoint(cx, cy)
            painter.drawLine(cx, cy, int(cx + d / 2), cy)
            painter.drawLine(cx, cy, cx, int(cy + d / 2))
            painter.drawLine(cx, cy, int(cx - d / 2), cy)
            painter.drawLine(cx, cy, cx, int(cy - d / 2))
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
                    int((cx + ox) / 2 + d), 
                    int((cy + oy) / 2 + d), 
                    str(distance) + " nm",
                )
            oldpoint = point
        painter.end()
        return image

    def add_legend(self):
        """ Shows/Hides legend. """

        if self.display_legend:
            self.display_legend = False
        else:
            self.display_legend = True
        self.update_scene()

    def add_rotation_view(self):
        """ Shows/Hides rotation axes icon. """

        if self.display_rotation:
            self.display_rotation = False
        else:
            self.display_rotation = True
        self.update_scene()

    def add_angles_view(self):
        """ Shows/Hides current rotation in angles. """

        if self.display_angles:
            self.display_angles = False
        else:
            self.display_angles = True
        self.update_scene()

    def rotation_input(self): 
        """ Asks user to input 3 rotation angles numerically. """

        angx, ok = QtWidgets.QInputDialog.getDouble(
                self, "Rotation angle x", "Angle x (degrees):",
                0, decimals=2,
            )
        if ok:
            angy, ok2 = QtWidgets.QInputDialog.getDouble(
                    self, "Rotation angle y", "Angle y (degrees):", 
                    0, decimals=2,
                )
            if ok2:
                angz, ok3 = QtWidgets.QInputDialog.getDouble(
                        self, "Rotation angle z", "Angle z (degrees):", 
                        0, decimals=2,
                        )
                if ok3:
                    self.angx += np.pi * angx/180
                    self.angy += np.pi * angy/180
                    self.angz += np.pi * angz/180

        # This is to avoid dividing by zero, cos(90) = 0
        if self.angx == np.pi / 2:
            self.angx += 0.00001
        if self.angy == np.pi / 2:
            self.angy += 0.00001
        if self.angz == np.pi / 2:
            self.angz += 0.00001

        self.update_scene()

    def delete_rotation(self):
        """ Resets rotation angles. """

        self.angx = 0
        self.angy = 0
        self.angz = 0
        self.update_scene()

    def fit_in_view_rotated(self, get_viewport=False):
        """ 
        Updates viewport to reflect the pick from main window. 

        Parameters
        ----------
        get_viewport : boolean
            If True, returns the found viewport. Otherwise updates
            scene with the found viewport
        """

        if self.pick_shape == "Circle":
            d = self.pick_size
            r = d / 2
            x, y = self.pick
            x_min = x - r
            x_max = x + r
            y_min = y - r
            y_max = y + r
        else:
            w = self.pick_size
            (xs, ys), (xe, ye) = self.pick
            X, Y = self.window.window.view.get_pick_rectangle_corners(
                xs, ys, xe, ye, w
            )
            x_min = min(X)
            x_max = max(X)
            y_min = min(Y)
            y_max = max(Y)

        viewport = [(y_min, x_min), (y_max, x_max)]
        if get_viewport:
            return viewport
        else:
            self.viewport = viewport
            self.update_scene()

    def to_left_rot(self):
        """ 
        Called on pressing left arrow; shifts pick in the main window.
        """

        height, width = self.viewport_size()
        dx = -SHIFT * width
        dx /= np.cos(self.angy)
        self.window.move_pick(dx, 0)
        self.shift_viewport(dx, 0)

    def to_right_rot(self):
        """ 
        Called on pressing right arrow; shifts pick in the main window.
        """
        
        height, width = self.viewport_size()
        dx = SHIFT * width
        dx /= np.cos(self.angy)
        self.window.move_pick(dx, 0)
        self.shift_viewport(dx, 0)

    def to_up_rot(self):
        """ 
        Called on pressing up arrow; shifts pick in the main window.
        """
        
        height, width = self.viewport_size()
        dy = -SHIFT * height
        dy /= np.cos(self.angx)
        self.window.move_pick(0, dy)
        self.shift_viewport(0, dy)

    def to_down_rot(self):
        """ 
        Called on pressing down arrow; shifts pick in the main window.
        """
        
        height, width = self.viewport_size()
        dy = SHIFT * height
        dy /= np.cos(self.angx)
        self.window.move_pick(0, dy)
        self.shift_viewport(0, dy)

    def shift_viewport(self, dx, dy):
        """ 
        Moves viewport by a given amount.

        Parameters
        ----------
        dx : float
            shift in x (pixels)
        dy : float
            shift in y (pixels)
        """

        (y_min, x_min), (y_max, x_max) = self.viewport
        new_viewport = [(y_min + dy, x_min + dx), (y_max + dy, x_max + dx)]
        self.load_locs() # pick locs in the new viewport
        self.update_scene(viewport=new_viewport)

    def keyPressEvent(self, event):
        if event.key() == 88: # x
            self.block_x = True
            self.block_y = False
            self.block_z = False
            event.accept()
        elif event.key() == 89: # y
            self.block_x = False
            self.block_y = True
            self.block_z = False
            event.accept()
        elif event.key() == 90: # z
            self.block_x = False
            self.block_y = False
            self.block_z = True
            event.accept()
        else:
            event.ignore()

    def keyReleaseEvent(self, event):
        if event.key() in [88, 89, 90]: # x, y or z
            self.block_x = False
            self.block_y = False
            self.block_z = False
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        """
        Defines actions taken when moving mouse.

        Rotating locs, panning.

        Parameters
        ----------
        event : QMouseEvent
        """

        if self._mode == "Rotate":
            if self._pan: # panning
                rel_x_move = (event.x() - self.pan_start_x) / self.width()
                rel_y_move = (event.y() - self.pan_start_y) / self.height()

                # this partially accounts for rotation of locs
                rel_y_move /= np.cos(self.angx)
                rel_x_move /= np.cos(self.angy)

                self.pan_relative(rel_y_move, rel_x_move)
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()

            else: # rotating
                height, width = self.viewport_size()
                pos = self.map_to_movie(event.pos())

                self._rotation.append([pos[0], pos[1]])

                # calculate the angle of rotation
                rel_pos_x = self._rotation[-1][0] - self._rotation[-2][0]
                rel_pos_y = self._rotation[-1][1] - self._rotation[-2][1]

                # rotate around x and y or y and z axes, depending on
                # whether Ctrl/Command is pressed
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ControlModifier:
                    if not self.block_y:
                        self.angz += float(2 * np.pi * rel_pos_y/height)
                    if not self.block_z:
                        self.angy += float(2 * np.pi * rel_pos_x/width)
                else:
                    if not self.block_x:
                        self.angy += float(2 * np.pi * rel_pos_x/width)
                    if not self.block_y:
                        self.angx += float(2 * np.pi * rel_pos_y/height)

                self.update_scene()

    def mousePressEvent(self, event):
        """
        Defines actions taken when pressing mouse buttons.

        Starting rotating locs or panning.

        Parameters
        ----------
        event : QMouseEvent
        """

        if self._mode == "Rotate":
            # start rotation
            if event.button() == QtCore.Qt.LeftButton:
                pos = self.map_to_movie(event.pos())
                self._rotation.append([float(pos[0]), float(pos[1])])
                event.accept()

            # start panning
            elif event.button() == QtCore.Qt.RightButton:
                self._pan = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()

    def mouseReleaseEvent(self, event):
        """
        Defines actions taken when releasing mouse buttons.

        Stopping rotating locs or panning, add or delete a measure
        point.

        Parameters
        ----------
        event : QMouseEvent
        """

        if self._mode == "Measure":
            # add point
            if event.button() == QtCore.Qt.LeftButton:
                x, y = self.map_to_movie(event.pos())
                self.add_point((x, y))
                event.accept()
            # remove point
            elif event.button() == QtCore.Qt.RightButton:
                x, y = self.map_to_movie(event.pos())
                self.remove_points((x, y))
                event.accept()
            else:
                event.ignore()

        elif self._mode == "Rotate":
            # stop rotation
            if event.button() == QtCore.Qt.LeftButton:
                self._rotation = []
                event.accept()
            # stop panning
            elif event.button() == QtCore.Qt.RightButton:
                self._pan = False
                event.accept()

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
        self.viewport = [(y_min, x_min), (y_max, x_max)]
        self.update_scene()

    def add_point(self, position, update_scene=True):
        """ 
        Adds a point at a given position for measuring distances.
        """

        self._points.append(position)
        if update_scene:
            self.update_scene()

    def remove_points(self, position):
        """ Removes all distance measurement points. """

        self._points = []
        self.update_scene()

    def export_current_view(self):
        """ Exports current view as .png or .tif. """

        try:
            base, ext = os.path.splitext(self.paths[0])
        except AttributeError:
            return
        out_path = base + "_rotated_{}_{}_{}.png".format(
                int(self.angx * 180 / np.pi), 
                int(self.angy * 180 / np.pi), 
                int(self.angz * 180 / np.pi)
            )
        path, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", out_path, filter="*.png;;*.tif"
        )
        if path:
            self.qimage.save(path)

    def zoom_in(self):
        """ Zooms in by a constant factor. """

        self.zoom(1 / ZOOM)

    def zoom_out(self):
        """ Zooms out by a constant factor. """

        self.zoom(ZOOM)

    def zoom(self, factor):
        """
        Changes zoom relatively to factor by changing viewport.

        Parameters
        ----------
        factor : float
            Relative zoom magnitude
        """

        height, width = self.viewport_size()
        new_height = height * factor
        new_width = width * factor

        new_center_y, new_center_x = self.viewport_center()

        new_viewport = [
            (
                new_center_y - new_height/2,
                new_center_x - new_width/2,
            ),
            (
                new_center_y + new_height/2,
                new_center_x + new_width/2,
            ),
        ]
        self.update_scene(new_viewport)

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

    def set_mode(self, action):
        """
        Sets self._mode for QMouseEvents.

        Activated when Rotate or Measure is chosen from Tools menu
        in the main window.

        Parameters
        ----------
        action : QAction
            Action defined in Window.__init__: ("Rotate" or "Measure")
        """

        self._mode = action.text()

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

    def get_render_kwargs(self, viewport=None, animation=False):
        """
        Returns a dictionary to be used for the keyword arguments of 
        render.render.

        Parameters
        ----------
        viewport : list (default=None)
            Specifies the FOV to be rendered. If None, the current 
            viewport is taken.
        animation : boolean
            If True, kwargs are found for building animation

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
        if not animation:
            optimal_oversampling = (
                self.display_pixels_per_viewport_pixels(viewport=viewport)
            )
            if self.window.display_settings_dlg.dynamic_disp_px.isChecked():
                oversampling = optimal_oversampling
                self.window.display_settings_dlg.set_disp_px_silently(
                    self.window.window.display_settings_dlg.pixelsize.value()
                    / optimal_oversampling
                )
            else:
                oversampling = float(
                    self.window.window.display_settings_dlg.pixelsize.value()
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
                        self.window.window.display_settings_dlg.pixelsize.value()
                        / optimal_oversampling
                    )
        else: # if animating, the message box may appear
            oversampling = float(
                self.window.window.display_settings_dlg.pixelsize.value()
                / self.window.display_settings_dlg.disp_px_size.value()
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

    def display_pixels_per_viewport_pixels(self, viewport=None):
        """ Returns optimal oversampling. """

        os_horizontal = self.width() / self.viewport_width()
        os_vertical = self.height() / self.viewport_height()
        # The values should be identical, but just in case, 
        # we choose the maximum value:
        return max(os_horizontal, os_vertical)

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


class RotationWindow(QtWidgets.QMainWindow):
    """
    A class containg the rotation window dialog.

    ...

    Attributes
    ----------
    animation_dialog : AnimationDialog
        instance of animation dialog
    display_settings_dlg : DisplaySettingsRotationDialog
        instance of display settings rotation dialog
    menu_bar : QMenuBar
        menu bar with menus: File, View, Tools
    menus : list
        contains File, View and Tools menus, used for plugins
    view_rot : ViewRotation
        instance of the class for displaying rendered localizations
    window : QMainWindow
        instance of the main Picasso: Render window (RotationWindow's
        parent)

    Methods
    -------
    closeEvent(event)
        Closes all children dialogs and self
    move_pick(dx, dy)
        Moves the pick in the main window by a given amount
    save_channel_multi()
        Opens an input dialog to ask which channel to save
    save_locs_rotated()
        Save locs from the main window and provides rotation info for
        later loading
    """

    def __init__(self, window):
        super().__init__()
        self.setWindowTitle("Picasso: Render 3D")
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

        # menu bar - View
        view_menu = self.menu_bar.addMenu("View")
        display_settings_action = view_menu.addAction("Display settings")
        display_settings_action.setShortcut("Ctrl+D")
        display_settings_action.triggered.connect(
            self.display_settings_dlg.show
        )
        view_menu.addAction(display_settings_action)
        legend_action = view_menu.addAction("Show/hide legend")
        legend_action.setShortcut("Ctrl+L")
        legend_action.triggered.connect(self.view_rot.add_legend)
        rotation_view_action = view_menu.addAction("Show/hide rotation")
        rotation_view_action.setShortcut("Ctrl+P")
        rotation_view_action.triggered.connect(self.view_rot.add_rotation_view)
        angles_display_action = view_menu.addAction(
            "Show/hide rotation angles"
        )
        angles_display_action.triggered.connect(self.view_rot.add_angles_view)
        
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
        tools_actiongroup = QtWidgets.QActionGroup(self.menu_bar)

        measure_tool_action = tools_actiongroup.addAction(
            QtWidgets.QAction("Measure", tools_menu, checkable=True)
        )
        measure_tool_action.setShortcut("Ctrl+M")
        tools_menu.addAction(measure_tool_action)
        tools_actiongroup.triggered.connect(self.view_rot.set_mode)

        rotate_tool_action = tools_actiongroup.addAction(
            QtWidgets.QAction("Rotate", tools_menu, checkable=True)
        )
        rotate_tool_action.setShortcut("Ctrl+R")
        tools_menu.addAction(rotate_tool_action)

        self.menus = [file_menu, view_menu, tools_menu]
        self.setMinimumSize(100, 100)
        self.move(20,20)

    def move_pick(self, dx, dy):
        """
        Moves the pick in the main window by a given amount.

        Parameters
        ----------
        dx : float
            pick shift in x axis (pixels)
        dy : float
            pick shift in x axis (pixels)
        """

        if self.view_rot.pick_shape == "Circle":
            x = self.window.view._picks[0][0]
            y = self.window.view._picks[0][1]
            self.window.view._picks = [(x + dx, y + dy)] # main window
            self.view_rot.pick = (x + dx, y + dy) # view rotation
        else: # rectangle
            (xs, ys), (xe, ye) = self.window.view._picks[0]
            self.window.view._picks = [(
                (xs + dx, ys + dy), 
                (xe + dx, ye + dy),
            )] # main window
            self.view_rot.pick = (
                (xs + dx, ys + dy), 
                (xe + dx, ye + dy),
            ) # view rotation
        self.window.view.update_scene() # update scene in main window

    def save_channel_multi(self, title="Choose a channel"):
        """
        Opens an input dialog to ask which channel to save.
        There is an option to save all channels.

        Returns
        None if no locs found or channel picked, int otherwise
            Index of the chosen channel
        """

        n_channels = len(self.view_rot.paths)
        if n_channels == 0:
            return None
        elif n_channels == 1:
            return 0
        elif len(self.view_rot.paths) > 1:
            pathlist = list(self.view_rot.paths)
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

    def save_locs_rotated(self):
        """
        Save locs from the main window and provides rotation info for
        later loading.
        """

        channel = self.window.view.save_channel("Save rotated localizations")

        if channel is not None:
            # rotation info
            angx = int(self.view_rot.angx * 180 / np.pi)
            angy = int(self.view_rot.angy * 180 / np.pi)
            angz = int(self.view_rot.angz * 180 / np.pi)
            if self.view_rot.pick_shape == "Circle":
                x, y = self.view_rot.pick
                pick = [float(x), float(y)]
                size = self.view_rot.pick_size
            else: # rectangle
                (ys, xs), (ye, xe) = self.view_rot.pick
                pick = [[float(ys), float(xs)], [float(ye), float(xe)]]
                size = self.view_rot.pick_size
            new_info = [{
                "Generated by": "Picasso Render 3D",
                "Pick": pick,
                "Pick shape": self.view_rot.pick_shape,
                "Pick size": size,
                "angx": self.view_rot.angx,
                "angy": self.view_rot.angy,
                "angz": self.view_rot.angz,
            }]

            # combine all channels
            if channel is (len(self.view_rot.paths) + 1):
                base, ext = os.path.splitext(self.view_rot.paths[0])
                out_path = base + "_multi.hdf5"
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save picked localizations",
                    out_path,
                    filter="*.hdf5",
                )
                if path:
                    # combine locs from all channels
                    all_locs = stack_arrays(
                        self.window.view.all_locs, 
                        asrecarray=True, 
                        usemask=False,
                        autoconvert=True,
                    )
                    all_locs.sort(kind="mergesort", order="frame")
                    info = self.view_rot.infos[0] + new_info
                    io.save_locs(path, all_locs, info)
            # save all channels one by one
            elif channel is (len(self.view_rot.paths)): # all channels
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    f"_arotated_{angx}_{angy}_{angz}",
                ) # get the save file suffix
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
                out_path = self.view_rot.paths[channel].replace(
                    ".hdf5", f"_rotated_{angx}_{angy}_{angz}.hdf5"
                )
                path, ext = QtWidgets.QFileDialog.getSaveFileName(
                    self, 
                    "Save rotated localizations", 
                    out_path, 
                    filter="*hdf5",
                )
                info = self.view_rot.infos[channel] + new_info
                io.save_locs(path, self.window.view.all_locs[channel], info)

    def closeEvent(self, event):
        """ Closes all children dialogs and self. """
        
        self.display_settings_dlg.close()
        self.animation_dialog.close()
        QtWidgets.QMainWindow.closeEvent(self, event)