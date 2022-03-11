"""
    picasso/rotation
    ~~~~~~~~~~~~~~~~~~~~

    Rotation window classes and functions.
    Extension of Picasso: Render to visualize 3D data.
    Many functions are copied from gui.render.View to avoid circular import

    :author: Rafal Kowalewski, 2021-2022
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

import os
import colorsys
import re
from functools import partial

import yaml
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PyQt5 import QtCore, QtGui, QtWidgets
from tqdm import tqdm

from numpy.lib.recfunctions import stack_arrays

from .. import io, render

from icecream import ic

DEFAULT_OVERSAMPLING = 1.0
ZOOM = 9 / 7
N_GROUP_COLORS = 8
SHIFT = 0.1


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('([0-9]+)', text) ]

def get_colors(n_channels):
    hues = np.arange(0, 1, 1 / n_channels)
    colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
    return colors

def is_hexadecimal(text):
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
    def __init__(self, window):
        super().__init__(window)
        self.first_update = True
        self.window = window
        self.setWindowTitle("Display Settings - Rotation Window")
        self.resize(200, 0)
        self.setModal(False)
        vbox = QtWidgets.QVBoxLayout(self)
        # General
        general_groupbox = QtWidgets.QGroupBox("General")
        vbox.addWidget(general_groupbox)
        general_grid = QtWidgets.QGridLayout(general_groupbox)
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

        #Camera
        self.pixelsize = QtWidgets.QDoubleSpinBox()
        self.pixelsize.setValue(130)
        
        #Scalebar
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

    def on_oversampling_changed(self, value):
        contrast_factor = (self._oversampling / value) ** 2
        self._oversampling = value
        self.silent_minimum_update(contrast_factor * self.minimum.value())
        self.silent_maximum_update(contrast_factor * self.maximum.value())
        if not self._silent_oversampling_update:
            self.dynamic_oversampling.setChecked(False)
            self.window.view_rot.update_scene()

    def set_oversampling_silently(self, oversampling):
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
        # check if ind loc prec button is checked
        if (self.blur_buttongroup.checkedId() == -5 
            or self.blur_buttongroup.checkedId() == -6):
            if self.ilp_warning:
                self.ilp_warning = False
                warning = (
                    "Rotating with individual localization precision may be "
                    "time consuming.  Therefore, we recommend to firstly "
                    " rotate the object using a different blur method and "
                    " then to apply individual localization precision."
                )
                QtWidgets.QMessageBox.information(self, "Warning", warning)
        
        self.window.view_rot.update_scene()

    def set_dynamic_oversampling(self, state):
        if state:
            self.window.view_rot.update_scene()

    def update_scene(self, *args, **kwargs):
        self.window.view_rot.update_scene()


class AnimationDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Build an animation")
        self.setModal(False)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.positions = []
        self.positions_labels = []
        self.durations = []
        self.show_positions = []
        self.delete = []
        self.count = 0
        self.frames_ready = False
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
        if self.count == 10:
            raise ValueError("More positions are not supported")
        if not freeze:
            if self.count > 0:
                cond1 = self.window.view_rot.angx == self.positions[-1][0]
                cond2 = self.window.view_rot.angy == self.positions[-1][1]
                cond3 = self.window.view_rot.angz == self.positions[-1][2]
                cond4 = self.window.view_rot.viewport == self.positions[-1][3]
                if ((cond1 and cond2) and cond3) and cond4:
                    return

        self.positions.append([
                self.window.view_rot.angx,
                self.window.view_rot.angy,
                self.window.view_rot.angz,
                self.window.view_rot.viewport,
            ])

        angx = np.round(self.window.view_rot.angx * 180 / np.pi, 1)
        angy = np.round(self.window.view_rot.angy * 180 / np.pi, 1)
        angz = np.round(self.window.view_rot.angz * 180 / np.pi, 1)
        self.positions_labels.append(QtWidgets.QLabel("{}, {}, {}".format(
                angx, angy, angz)))
        self.layout.addWidget(self.positions_labels[-1], self.count + 1, 1)

        # calculate recommended duration
        if self.count > 0:
            if not freeze:
                if not ((cond1 and cond2) and cond3):
                    dx = self.positions[-1][0] - self.positions[-2][0]
                    dy = self.positions[-1][1] - self.positions[-2][1]
                    dz = self.positions[-1][2] - self.positions[-2][2]
                    dmax = np.max(np.abs([dx, dy, dz]))
                    rot_speed = self.rot_speed.value() * np.pi / 180
                    dur = dmax / rot_speed
                    self.durations[self.count-1].setValue(dur)

        self.count += 1

    def delete_position(self):
        if self.count > 0:
            del self.positions[-1]
            self.layout.removeWidget(self.positions_labels[-1])
            del self.positions_labels[-1]
            self.count -= 1

    def retrieve_position(self, i):
        if i <= len(self.positions) - 1:
            self.window.view_rot.angx = self.positions[i][0]
            self.window.view_rot.angy = self.positions[i][1]
            self.window.view_rot.angz = self.positions[i][2]
            self.window.view_rot.update_scene(viewport=self.positions[i][3])

    def build_animation(self):
        # get the coordinates for animation
        n_frames = [0]
        for i in range(len(self.positions) - 1):
            n_frames.append(int(self.fps.value() * self.durations[i].value()))

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

            #angles:
            x1 = self.positions[i][0]
            x2 = self.positions[i+1][0]
            y1 = self.positions[i][1]
            y2 = self.positions[i+1][1]
            z1 = self.positions[i][2]
            z2 = self.positions[i+1][2]
            angx[idx_low:idx_high] = np.linspace(x1, x2, n_frames[i+1])
            angy[idx_low:idx_high] = np.linspace(y1, y2, n_frames[i+1])
            angz[idx_low:idx_high] = np.linspace(z1, z2, n_frames[i+1])

            #viewport:
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

        out_path = self.window.view_rot.paths[0] + "_video.mp4"
        name, ext = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save animation", out_path, filter="*.mp4"
        )
        if name:
            # save the images
            base, ext = os.path.splitext(self.window.view_rot.paths[0])
            idx = [i for i, char in enumerate(base) if char == '/'][-1]
            path = base[:idx] + "/animation_frames"
            try:
                os.mkdir(path)
            except:
                m = QtWidgets.QMessageBox()
                m.setWindowTitle("Frames already exist")
                ret = m.question(
                    self,
                    "",
                    "Delete the existing frames folder?",
                    m.Yes | m.No,
                )
                if ret == m.Yes:
                    for file in os.listdir(path):
                        os.remove(os.path.join(path, file))
                elif ret == m.No:
                    self.frames_ready = True

            if not self.frames_ready:
                for i in range(len(angx)):
                        qimage = self.window.view_rot.render_scene(
                            viewport=[(ymin[i], xmin[i]), (ymax[i], xmax[i])],
                            ang=(angx[i], angy[i], angz[i]),
                            animation=True,
                        )
                        qimage = qimage.scaled(500, 500)
                        qimage.save(path + "/frame_{}.png".format(i+1))

            # build a video
            image_files = [
                os.path.join(path,img)
                for img in os.listdir(path)
                if img.endswith(".png")
            ]
            image_files.sort(key=natural_keys)

            video = ImageSequenceClip(image_files, fps=self.fps.value())
            video.write_videofile(name)

            # delete animaiton frames
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            os.rmdir(path)
            

class ViewRotation(QtWidgets.QLabel):
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
        self._mode = "Rotate"
        self._rotation = []
        self._centers = []
        self._points = []
        self._centers_color = ""
        self._pan = False
        self.display_legend = False
        self.display_rotation = True
        self.setMaximumSize(500, 500)

    def load_locs(self, update_window=False):
        if update_window:
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
            self.viewport = self.fit_in_view_rotated(get_viewport=True)
            self.window.dataset_dialog = self.window.window.dataset_dialog
            self.paths = self.window.window.view.locs_paths
        n_channels = len(self.paths)
        self.locs = []
        self.infos = []
        for i in range(n_channels):
            temp = self.window.window.view.picked_locs(i, add_group=False)
            self.locs.append(temp[0])
            self.infos.append(self.window.window.view.infos[i])

        if len(self.locs) == 1 and hasattr(self.locs[0], "group"):
            self.group_color = self.window.window.view.get_group_color(
                self.locs[0]
            )
        self.update_scene()

    def render_scene(self, viewport=None, ang=None, animation=False):
        kwargs = self.get_render_kwargs(viewport=viewport, animation=animation)
        n_channels = len(self.locs)
        if n_channels == 1:
            self.render_single_channel(kwargs, ang=ang)
        else:
            self.render_multi_channel(kwargs, ang=ang)
        self._bgra[:, :, 3].fill(255)
        Y, X = self._bgra.shape[:2]
        qimage = QtGui.QImage(
            self._bgra.data, X, Y, QtGui.QImage.Format_RGB32
        )
        return qimage

    def render_multi_channel(self, kwargs, locs=None, ang=None):
        # rendering all channels
        if locs is None:
            locs = self.locs
        n_channels = len(locs)
        colors = get_colors(n_channels)
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
        if ang is None:
            renderings = [
                render.render(
                    _, **kwargs, 
                    ang=(self.angx, self.angy, self.angz), 
                    pixelsize=pixelsize,
                ) for _ in locs
            ]
        else:
            renderings = [
                render.render(
                    _, **kwargs, 
                    ang=ang, 
                    pixelsize=pixelsize,
                ) for _ in locs
            ]
        n_locs = sum([_[0] for _ in renderings])
        image = np.array([_[1] for _ in renderings])
        self.n_locs = n_locs
        self.image = image
        image = self.scale_contrast(image)
        Y, X = image.shape[1:]
        bgra = np.zeros((Y, X, 4), dtype=np.float32)

        # coloring
        for i in range(len(self.locs)):
            if not self.window.dataset_dialog.auto_colors.isChecked():
                color = self.window.dataset_dialog.colorselection[i].currentText()
                if color in self.window.dataset_dialog.default_colors:
                    index = self.window.dataset_dialog.default_colors.index(color)
                    colors[i] = tuple(self.window.dataset_dialog.rgbf[index])
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

    def render_single_channel(self, kwargs, ang=None):
        locs = self.locs[0]
        if hasattr(locs, "group"):
            locs = [locs[self.group_color == _] for _ in range(N_GROUP_COLORS)]
            return self.render_multi_channel(
                kwargs, locs=locs, ang=ang
            )
        pixelsize = self.window.display_settings_dlg.pixelsize.value()
        if ang is None:
            n_locs, image = render.render(
                locs, **kwargs, 
                info=self.infos[0], 
                ang=(self.angx, self.angy, self.angz), 
                pixelsize=pixelsize,
            )
        else:
            n_locs, image = render.render(
                locs, **kwargs, 
                info=self.infos[0], 
                ang=ang, 
                pixelsize=pixelsize
            )
        self.n_locs = n_locs
        self.image = image
        image = self.scale_contrast(image)
        image = self.to_8bit(image)
        Y, X = image.shape
        cmap = self.window.display_settings_dlg.colormap.currentText()
        cmap = np.uint8(np.round(255 * plt.get_cmap(cmap)(np.arange(256))))
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order="C")
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        return self._bgra

    def update_scene(self, viewport=None):
        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene(viewport)

        # update current position in the animation dialog
        angx = np.round(self.angx * 180 / np.pi, 1)
        angy = np.round(self.angy * 180 / np.pi, 1)
        angz = np.round(self.angz * 180 / np.pi, 1)
        self.window.animation_dialog.current_pos.setText(
            "{}, {}, {}".format(angx, angy, angz)
        )

    def draw_scene(self, viewport):
        self.viewport = self.adjust_viewport_to_view(viewport)
        qimage = self.render_scene()
        self.qimage = qimage.scaled(
            self.width(),
            self.height(),
            QtCore.Qt.KeepAspectRatioByExpanding,
        )
        self.qimage = self.draw_scalebar(self.qimage)
        if self.display_legend:
            self.qimage = self.draw_legend(self.qimage)
        if self.display_rotation:
            self.qimage = self.draw_rotation(self.qimage)
        self.qimage = self.draw_points(self.qimage)
        self.pixmap = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.pixmap)

    def draw_scalebar(self, image):
        if self.window.display_settings_dlg.scalebar_groupbox.isChecked():
            pixelsize = self.window.display_settings_dlg.pixelsize.value()
            scalebar = self.window.display_settings_dlg.scalebar.value()
            length_camerapxl = scalebar / pixelsize
            length_displaypxl = int(
                round(self.width() * length_camerapxl / self.viewport_width())
            )
            # height = max(int(round(0.15 * length_displaypxl)), 1)
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
        n_channels = len(self.locs)
        painter = QtGui.QPainter(image)
        width = 15
        height = 15
        x = 20
        y = -5
        dy = 25
        for i in range(n_channels):
            if self.window.dataset_dialog.checks[i].isChecked():
                painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                palette = self.window.dataset_dialog.colordisp_all[i].palette()
                color = palette.color(QtGui.QPalette.Window)
                painter.setBrush(QtGui.QBrush(color))
                y += dy
                painter.drawRect(x, y, height, height)
                font = painter.font()
                font.setPixelSize(12)
                painter.setFont(font)
                painter.setPen(QtGui.QColor("white"))
                if self.window.dataset_dialog.wbackground.isChecked():
                    painter.setPen(QtGui.QColor("black"))
                text_spacer = 25
                text_width = 1000
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

    def draw_rotation(self, image):
        painter = QtGui.QPainter(image)
        length = 30
        width = 2
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
        coordinates = R.apply(coordinates)
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

    def draw_points(self, image):
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

    def add_legend(self):
        if self.display_legend:
            self.display_legend = False
        else:
            self.display_legend = True
        self.update_scene()

    def add_rotation_view(self):
        if self.display_rotation:
            self.display_rotation = False
        else:
            self.display_rotation = True
        self.update_scene()

    def rotation_input(self, opening=False, ang=None): 
        # asks for rotation angles (3D only)
        if opening:
            self.angx = ang[0]
            self.angy = ang[1]
            self.angz = ang[2]
        else:
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

        # This is to avoid dividing by zero, when the angles are 90 deg 
        # and something is divided by cosines
        if self.angx == np.pi / 2:
            self.angx += 0.00001
        if self.angy == np.pi / 2:
            self.angy += 0.00001
        if self.angz == np.pi / 2:
            self.angz += 0.00001

        self.update_scene()

    def delete_rotation(self):
        self.angx = 0
        self.angy = 0
        self.angz = 0
        self.update_scene()

    def fit_in_view_rotated(self, get_viewport=False):
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
        height, width = self.viewport_size()
        dx = -SHIFT * width
        dx /= np.cos(self.angy)
        self.window.move_pick(dx, 0)
        self.shift_viewport(dx, 0)

    def to_right_rot(self):
        height, width = self.viewport_size()
        dx = SHIFT * width
        dx /= np.cos(self.angy)
        self.window.move_pick(dx, 0)
        self.shift_viewport(dx, 0)

    def to_up_rot(self):
        height, width = self.viewport_size()
        dy = -SHIFT * height
        dy /= np.cos(self.angx)
        self.window.move_pick(0, dy)
        self.shift_viewport(0, dy)

    def to_down_rot(self):
        height, width = self.viewport_size()
        dy = SHIFT * height
        dy /= np.cos(self.angx)
        self.window.move_pick(0, dy)
        self.shift_viewport(0, dy)

    def shift_viewport(self, dx, dy):
        self.load_locs()
        (y_min, x_min), (y_max, x_max) = self.viewport
        new_viewport = [(y_min + dy, x_min + dx), (y_max + dy, x_max + dx)]
        self.update_scene(viewport=new_viewport)

    def mouseMoveEvent(self, event):
        if self._mode == "Rotate":
            height, width = self.viewport_size()
            pos = self.map_to_movie(event.pos())

            if self._pan:
                rel_x_move = (event.x() - self.pan_start_x) / self.width()
                rel_y_move = (event.y() - self.pan_start_y) / self.height()
                rel_y_move /= np.cos(self.angx)
                rel_x_move /= np.cos(self.angy)

                self.pan_relative(rel_y_move, rel_x_move)
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()

            else:
                self._rotation.append([pos[0], pos[1]])

                # calculate the angle of rotation
                rel_pos_x = self._rotation[-1][0] - self._rotation[-2][0]
                rel_pos_y = self._rotation[-1][1] - self._rotation[-2][1]

                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ControlModifier:
                    self.angy += float(2 * np.pi * rel_pos_x/width)
                    self.angz += float(2 * np.pi * rel_pos_y/height)
                else:
                    self.angx += float(2 * np.pi * rel_pos_y/height)
                    self.angy += float(2 * np.pi * rel_pos_x/width)

                self.update_scene()

    def mousePressEvent(self, event):
        if self._mode == "Rotate":
            if event.button() == QtCore.Qt.LeftButton:
                pos = self.map_to_movie(event.pos())
                self._rotation.append([float(pos[0]), float(pos[1])])
                event.accept()

            elif event.button() == QtCore.Qt.RightButton:
                self._pan = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()

    def mouseReleaseEvent(self, event):
        if self._mode == "Measure":
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

        elif self._mode == "Rotate":
            if event.button() == QtCore.Qt.LeftButton:
                self._rotation = []
                event.accept()

            elif event.button() == QtCore.Qt.RightButton:
                self._pan = False
                self.setCursor(QtCore.Qt.ArrowCursor)
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
        return cx, cy

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

    def add_point(self, position, update_scene=True):
        self._points.append(position)
        if update_scene:
            self.update_scene()

    def remove_points(self, position):
        self._points = []
        self.update_scene()

    def export_current_view(self):
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
        self.zoom(1 / ZOOM)

    def zoom_out(self):
        self.zoom(ZOOM)

    def zoom(self, factor, cursor_position=None):
        height, width = self.viewport_size()
        new_height = height * factor
        new_width = width * factor

        if cursor_position is not None:
            old_center = self.viewport_center()
            rel_pos_x, rel_pos_y = self.relative_position(
                old_center, cursor_position
            ) #this stays constant before and after zooming
            new_center_x = (
                cursor_position[0] - rel_pos_x * new_width
            )
            new_center_y = (
                cursor_position[1] - rel_pos_y * new_height
            )
        else:
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

    def relative_position(self, center, cursor_position):
        # finds the position of the cursor relative to the current 
        # viewport center;
        rel_pos_x = (cursor_position[0] - center[1])/self.viewport_width()
        rel_pos_y = (cursor_position[1] - center[0])/self.viewport_height()
        return rel_pos_x, rel_pos_y

    def set_mode(self, action):
        self._mode = action.text()

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

    def get_render_kwargs(self, viewport=None, animation=False):
        """
        Returns a dictionary to be used for the
        keyword arguments of render.
        """
        blur_button = (
            self.window.display_settings_dlg.blur_buttongroup.checkedButton()
        )
        optimal_oversampling = (
            self.display_pixels_per_viewport_pixels(
                viewport=viewport, animation=animation
            )
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
        if viewport is None:
            viewport = self.viewport
        if animation:
            oversampling = optimal_oversampling
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

    def display_pixels_per_viewport_pixels(
        self, viewport=None, animation=False
    ):
        if animation:
            os_horizontal = 500 / self.viewport_width(viewport)
            os_vertical = 500 / self.viewport_height(viewport)
        else:
            os_horizontal = self.width() / self.viewport_width()
            os_vertical = self.height() / self.viewport_height()
        # The values should be identical, but just in case, we choose the max:
        return max(os_horizontal, os_vertical)

    def scale_contrast(self, image):
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
        return np.round(255 * image).astype("uint8")


class RotationWindow(QtWidgets.QMainWindow):
    def __init__(self, window):
        super().__init__()
        self.setWindowTitle("Rotation window")
        self.window = window
        self.view_rot = ViewRotation(self)
        self.setCentralWidget(self.view_rot)

        self.display_settings_dlg = DisplaySettingsRotationDialog(self)
        self.display_settings_dlg.ilp_warning = True

        self.animation_dialog = AnimationDialog(self)

        self.menu_bar = self.menuBar()
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
        
        view_menu.addSeparator()
        rotation_action = view_menu.addAction("Rotate by angle")
        rotation_action.triggered.connect(self.view_rot.rotation_input)
        rotation_action.setShortcut("Ctrl+Shift+R")

        delete_rotation_action = view_menu.addAction("Remove rotation")
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

        self.setMinimumSize(500, 500)
        self.setMaximumSize(500, 500)
        self.move(20,20)

    def move_pick(self, dx, dy):
        """ moves the pick in the main window """
        if self.view_rot.pick_shape == "Circle":
            x = self.window.view._picks[0][0]
            y = self.window.view._picks[0][1]
            self.window.view._picks = [(x + dx, y + dy)]
        else:
            (xs, ys), (xe, ye) = self.window.view._picks[0]
            self.window.view._picks = [(
                (xs + dx, ys + dy), 
                (xe + dx, ye + dy),
            )]
        self.window.view.update_scene()

    def save_channel_multi(self, title="Choose a channel"):
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
        # make sure that the locs in the main window are shown in the 3D window
        self.view_rot.load_locs()

        # save
        channel = self.save_channel_multi("Save rotated localizations")
        if channel is not None:
            angx = self.view_rot.angx * 180 / np.pi
            angy = self.view_rot.angy * 180 / np.pi
            angz = self.view_rot.angz * 180 / np.pi
            if self.view_rot.pick_shape == "Circle":
                x, y = self.pick
                pick = [float(x), float(y)]
                size = self.pick_size
            else:
                (ys, xs), (ye, xe) = self.pick
                pick = [[float(ys), float(xs)], [float(ye), float(xe)]]
                size = self.pick_size

            new_info = [{
                "Generated by": "Picasso Render 3D",
                # "Last driftfile": None,
                "Pick": pick,
                "Pick shape": self.view_rot.pick_shape,
                "Pick size": size,
                "angx": self.view_rot.angx,
                "angy": self.view_rot.angy,
                "angz": self.view_rot.angz,
            }]

            if channel is (len(self.view_rot.paths)):
                suffix, ok = QtWidgets.QInputDialog.getText(
                    self,
                    "Input Dialog",
                    "Enter suffix",
                    QtWidgets.QLineEdit.Normal,
                    "_arender",
                )
                if ok:
                    for channel in tqdm(range(len(self.view_rot.paths))):
                        base, ext = os.path.splitext(
                            self.view_rot.paths[channel]
                        )
                        out_path = (
                            base 
                            + suffix 
                            + "_rotated_{}_{}_{}.hdf5".format(
                                int(angx), int(angy), int(angz)
                            )
                        )
                        info = self.view_rot.infos[channel] + new_info
                        io.save_locs(
                            out_path, self.window.view.locs[channel], info
                        )
            else:
                base, ext = os.path.splitext(self.view_rot.paths[channel])
                out_path = (
                    base 
                    + "_rotated_{}_{}_{}.hdf5".format(
                        int(angx), int(angy), int(angz)
                    )
                )
                info = self.view_rot.infos[channel] + new_info
                io.save_locs(out_path, self.window.view.locs[channel], info)

    def closeEvent(self, event):
        self.display_settings_dlg.close()
        self.animation_dialog.close()
        QtWidgets.QMainWindow.closeEvent(self, event)