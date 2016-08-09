"""
    gui/render
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for rendering localization images

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, Max Planck Institute of Biochemistry
"""
import sys
import os.path
import traceback
from PyQt4 import QtCore, QtGui
import numpy as np
from numpy.lib.recfunctions import stack_arrays
import matplotlib.pyplot as plt
import colorsys
from math import ceil
from .. import io, lib, render


DEFAULT_OVERSAMPLING = 1.0
INITIAL_REL_MAXIMUM = 0.5
ZOOM = 10 / 7


class InfoDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle('Info')
        self.setModal(False)
        layout = QtGui.QGridLayout(self)
        layout.addWidget(QtGui.QLabel('Display Width:'), 0, 0)
        self.width_label = QtGui.QLabel()
        layout.addWidget(self.width_label, 0, 1)
        layout.addWidget(QtGui.QLabel('Display Height:'), 1, 0)
        self.height_label = QtGui.QLabel()
        layout.addWidget(self.height_label, 1, 1)
        layout.addWidget(QtGui.QLabel('# Localizations:'), 2, 0)
        self.locs_label = QtGui.QLabel()
        layout.addWidget(self.locs_label, 2, 1)


class ToolsSettingsDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Tools Settings')
        self.setModal(False)
        grid = QtGui.QGridLayout(self)
        grid.addWidget(QtGui.QLabel('Pick Radius:'), 0, 0)
        self.pick_diameter = QtGui.QDoubleSpinBox()
        self.pick_diameter.setRange(0, 999999)
        self.pick_diameter.setValue(1)
        self.pick_diameter.setSingleStep(0.1)
        self.pick_diameter.setDecimals(3)
        self.pick_diameter.setKeyboardTracking(False)
        self.pick_diameter.valueChanged.connect(self.update_scene)
        grid.addWidget(self.pick_diameter, 0, 1)

    def update_scene(self, diameter):
        self.window.view.update_scene(use_cache=True)


class DisplaySettingsDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Display Settings')
        self.resize(200, 0)
        self.setModal(False)
        vbox = QtGui.QVBoxLayout(self)
        # General
        general_groupbox = QtGui.QGroupBox('General')
        vbox.addWidget(general_groupbox)
        general_grid = QtGui.QGridLayout(general_groupbox)
        general_grid.addWidget(QtGui.QLabel('Oversampling:'), 0, 0)
        self._oversampling = DEFAULT_OVERSAMPLING
        self.oversampling = QtGui.QDoubleSpinBox()
        self.oversampling.setRange(0.001, 1000)
        self.oversampling.setSingleStep(5)
        self.oversampling.setValue(self._oversampling)
        self.oversampling.setKeyboardTracking(False)
        self.oversampling.valueChanged.connect(self.on_oversampling_changed)
        general_grid.addWidget(self.oversampling, 0, 1)
        # Contrast
        contrast_groupbox = QtGui.QGroupBox('Contrast')
        vbox.addWidget(contrast_groupbox)
        contrast_grid = QtGui.QGridLayout(contrast_groupbox)
        minimum_label = QtGui.QLabel('Min. Density:')
        contrast_grid.addWidget(minimum_label, 0, 0)
        self.minimum = QtGui.QDoubleSpinBox()
        self.minimum.setRange(0, 999999)
        self.minimum.setSingleStep(5)
        self.minimum.setValue(0)
        self.minimum.setKeyboardTracking(False)
        self.minimum.valueChanged.connect(self.update_scene)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtGui.QLabel('Max. Density:')
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = QtGui.QDoubleSpinBox()
        self.maximum.setRange(0, 999999)
        self.maximum.setSingleStep(5)
        self.maximum.setValue(100)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.update_scene)
        contrast_grid.addWidget(self.maximum, 1, 1)
        contrast_grid.addWidget(QtGui.QLabel('Colormap:'), 2, 0)
        self.colormap = QtGui.QComboBox()
        self.colormap.addItems(sorted(['hot', 'viridis', 'inferno', 'plasma', 'magma', 'gray']))
        contrast_grid.addWidget(self.colormap, 2, 1)
        self.colormap.currentIndexChanged.connect(self.update_scene)
        # Blur
        blur_groupbox = QtGui.QGroupBox('Blur')
        blur_grid = QtGui.QGridLayout(blur_groupbox)
        self.blur_buttongroup = QtGui.QButtonGroup()
        points_button = QtGui.QRadioButton('None')
        self.blur_buttongroup.addButton(points_button)
        smooth_button = QtGui.QRadioButton('Smooth')
        self.blur_buttongroup.addButton(smooth_button)
        convolve_button = QtGui.QRadioButton('Global Localization Precision')
        self.blur_buttongroup.addButton(convolve_button)
        gaussian_button = QtGui.QRadioButton('Individual Localization Precision')
        self.blur_buttongroup.addButton(gaussian_button)
        blur_grid.addWidget(points_button, 0, 0, 1, 2)
        blur_grid.addWidget(smooth_button, 1, 0, 1, 2)
        blur_grid.addWidget(convolve_button, 2, 0, 1, 2)
        blur_grid.addWidget(gaussian_button, 3, 0, 1, 2)
        smooth_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.render_scene)
        blur_grid.addWidget(QtGui.QLabel('Min. Blur (cam. pixel):'), 4, 0, 1, 1)
        self.min_blur_width = QtGui.QDoubleSpinBox()
        self.min_blur_width.setRange(0, 999999)
        self.min_blur_width.setSingleStep(0.01)
        self.min_blur_width.setValue(0)
        self.min_blur_width.setDecimals(3)
        self.min_blur_width.setKeyboardTracking(False)
        self.min_blur_width.valueChanged.connect(self.render_scene)
        blur_grid.addWidget(self.min_blur_width, 4, 1, 1, 1)
        vbox.addWidget(blur_groupbox)
        self.blur_methods = {points_button: None, smooth_button: 'smooth',
                             convolve_button: 'convolve', gaussian_button: 'gaussian'}
        # Scale bar
        self.scalebar_groupbox = QtGui.QGroupBox('Scale Bar')
        self.scalebar_groupbox.setCheckable(True)
        self.scalebar_groupbox.setChecked(False)
        self.scalebar_groupbox.toggled.connect(self.update_scene)
        vbox.addWidget(self.scalebar_groupbox)
        scalebar_grid = QtGui.QGridLayout(self.scalebar_groupbox)
        scalebar_grid.addWidget(QtGui.QLabel('Pixel Size:'), 0, 0)
        self.pixelsize = QtGui.QDoubleSpinBox()
        self.pixelsize.setRange(1, 1000000000)
        self.pixelsize.setValue(160)
        self.pixelsize.setKeyboardTracking(False)
        self.pixelsize.valueChanged.connect(self.update_scene)
        scalebar_grid.addWidget(self.pixelsize, 0, 1)
        scalebar_grid.addWidget(QtGui.QLabel('Scale Bar Length (nm):'), 1, 0)
        self.scalebar = QtGui.QDoubleSpinBox()
        self.scalebar.setRange(0.0001, 10000000000)
        self.scalebar.setValue(500)
        self.scalebar.setKeyboardTracking(False)
        self.scalebar.valueChanged.connect(self.update_scene)
        scalebar_grid.addWidget(self.scalebar, 1, 1)

    def on_oversampling_changed(self, value):
        contrast_factor = (self._oversampling / value)**2
        self._oversampling = value
        self.silent_minimum_update(contrast_factor * self.minimum.value())
        self.silent_maximum_update(contrast_factor * self.maximum.value())
        self.window.view.update_scene()

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

    def update_scene(self, *args, **kwargs):
        self.window.view.update_scene(use_cache=True)


class View(QtGui.QLabel):

    def __init__(self, window):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.rubberband = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)
        self.rubberband.setStyleSheet('selection-background-color: white')
        self.window = window
        self._pixmap = None
        self.locs = []
        self.infos = []
        self._mode = 'Zoom'
        self._pan = False
        self._size_hint = (768, 768)
        self.n_locs = 0
        self._picks = []

    def add(self, path):
        locs, info = io.load_locs(path)
        self.locs_path = path
        locs = lib.ensure_finite(locs)
        self.locs.append(locs)
        self.infos.append(info)
        if len(self.locs) == 1:
            if hasattr(locs, 'group'):
                self.groups = np.unique(locs.group)
                np.random.shuffle(self.groups)
            self.fit_in_view(autoscale=True)
        else:
            self.update_scene()

    def adjust_viewport_to_view(self, viewport):
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

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def draw_picks(self, image):
        d = self.window.tools_settings_dialog.pick_diameter.value()
        d *= self.width() / self.viewport_width()
        # d = int(round(d))
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QColor('yellow'))
        for pick in self._picks:
            cx, cy = self.map_to_view(pick['x'], pick['y'])
            painter.drawEllipse(cx-d/2, cy-d/2, d, d)
        painter.end()
        return image

    def draw_scalebar(self, image):
        if self.window.display_settings_dialog.scalebar_groupbox.isChecked():
            pixelsize = self.window.display_settings_dialog.pixelsize.value()
            scalebar = self.window.display_settings_dialog.scalebar.value()
            length_camerapxl = scalebar / pixelsize
            length_displaypxl = int(round(self.width() * length_camerapxl / self.viewport_width()))
            height = max(int(round(0.15 * length_displaypxl)), 1)
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            painter.setBrush(QtGui.QBrush(QtGui.QColor('white')))
            x = self.width() - length_displaypxl - 20
            y = self.height() - height - 20
            painter.drawRect(x, y, length_displaypxl + 0, height + 0)
        return image

    def draw_scene(self, viewport, autoscale=False, use_cache=False):
        self.viewport = self.adjust_viewport_to_view(viewport)
        qimage = self.render_scene(autoscale=autoscale, use_cache=use_cache)
        qimage = qimage.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatioByExpanding)
        qimage = self.draw_picks(qimage)
        self.qimage = self.draw_scalebar(qimage)
        pixmap = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(pixmap)
        self.window.update_info()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        extension = os.path.splitext(path)[1].lower()
        if extension == '.hdf5':
            self.add(path)

    def fit_in_view(self, autoscale=False):
        movie_height, movie_width = self.movie_size()
        viewport = [(0, 0), (movie_height, movie_width)]
        self.update_scene(viewport=viewport, autoscale=autoscale)

    def get_render_kwargs(self, viewport):
        blur_button = self.window.display_settings_dialog.blur_buttongroup.checkedButton()
        return {'oversampling': float(self.window.display_settings_dialog.oversampling.value()),
                'viewport': viewport,
                'blur_method': self.window.display_settings_dialog.blur_methods[blur_button],
                'min_blur_width': float(self.window.display_settings_dialog.min_blur_width.value())}

    def map_to_movie(self, position):
        x_rel = position.x() / self.width()
        x_movie = x_rel * self.viewport_width() + self.viewport[0][1]
        y_rel = position.y() / self.height()
        y_movie = y_rel * self.viewport_height() + self.viewport[0][0]
        return x_movie, y_movie

    def map_to_view(self, x, y):
        cx = self.width() * (x - self.viewport[0][1]) / self.viewport_width()
        cy = self.height() * (y - self.viewport[0][0]) / self.viewport_height()
        return cx, cy

    def max_movie_height(self):
        return max(info[0]['Height'] for info in self.infos)

    def max_movie_width(self):
        return max([info[0]['Width'] for info in self.infos])

    def mouseMoveEvent(self, event):
        if self._mode == 'Zoom':
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(QtCore.QRect(self.origin, event.pos()))
            if self._pan:
                rel_x_move = (event.x() - self.pan_start_x) / self.width()
                rel_y_move = (event.y() - self.pan_start_y) / self.height()
                self.pan_relative(rel_y_move, rel_x_move)
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()

    def mousePressEvent(self, event):
        if self._mode == 'Zoom':
            if event.button() == QtCore.Qt.LeftButton:
                if not self.rubberband.isVisible():
                    self.origin = QtCore.QPoint(event.pos())
                    self.rubberband.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
                    self.rubberband.show()
            elif event.button() == QtCore.Qt.RightButton:
                self._pan = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
            else:
                event.ignore()

    def mouseReleaseEvent(self, event):
        if self._mode == 'Zoom':
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
            else:
                event.ignore()
        elif self._mode == 'Pick':
            x, y = self.map_to_movie(event.pos())
            self._picks.append({'x': x, 'y': y})
            self.update_scene(use_cache=True)

    def movie_size(self):
        movie_height = self.max_movie_height()
        movie_width = self.max_movie_width()
        return (movie_height, movie_width)

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

    def render_scene(self, viewport=None, autoscale=False, use_cache=False, cache=True):
        if viewport is None:
            viewport = self.viewport
        kwargs = self.get_render_kwargs(viewport)
        n_channels = len(self.locs)
        if n_channels == 1:
            self.render_single_channel(kwargs, autoscale=autoscale, use_cache=use_cache, cache=cache)
        else:
            self.render_multi_channel(kwargs, autoscale=autoscale, use_cache=use_cache, cache=cache)
        self._bgra[:, :, 3].fill(255)
        Y, X = self._bgra.shape[:2]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        return qimage

    def render_multi_channel(self, kwargs, autoscale=False, locs=None, use_cache=False, cache=True):
        if locs is None:
            locs = self.locs
        n_channels = len(locs)
        hues = np.arange(0, 1, 1/n_channels)
        colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
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
        for color, image in zip(colors, image):
            bgra[:, :, 0] += color[2] * image
            bgra[:, :, 1] += color[1] * image
            bgra[:, :, 2] += color[0] * image
        bgra = np.minimum(bgra, 1)
        self._bgra = self.to_8bit(bgra)
        return self._bgra

    def render_single_channel(self, kwargs, autoscale=False, use_cache=False, cache=True):
        locs = self.locs[0]
        if hasattr(locs, 'group'):
            locs = [locs[locs.group == _] for _ in self.groups]
            return self.render_multi_channel(kwargs, autoscale=autoscale, locs=locs, use_cache=use_cache)
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
        cmap = self.window.display_settings_dialog.colormap.currentText()
        cmap = np.uint8(np.round(255 * plt.get_cmap(cmap)(np.arange(256))))
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = cmap[:, 2][image]
        self._bgra[..., 1] = cmap[:, 1][image]
        self._bgra[..., 2] = cmap[:, 0][image]
        return self._bgra

    def resizeEvent(self, event):
        self.update_scene()

    def save_picked_locs(self, path):
        picked_locs = []
        d = self.window.tools_settings_dialog.pick_diameter.value()
        r = d / 2
        for i, pick in enumerate(self._picks):
            dx = self.locs[0].x - pick['x']
            dy = self.locs[0].y - pick['y']
            is_picked = np.sqrt(dx**2 + dy**2) < r
            group_locs = self.locs[0][is_picked]
            group = i * np.ones(len(group_locs), dtype=np.int32)
            group_locs = lib.append_to_rec(group_locs, group, 'group')
            picked_locs.append(group_locs)
        locs = stack_arrays(picked_locs, asrecarray=True, usemask=False)
        pick_info = {'Generated by:': 'Picasso Render', 'Pick Diameter:': d}
        io.save_locs(path, locs, self.infos[0] + [pick_info])

    def scale_contrast(self, image, autoscale=False):
        if autoscale:
            if image.ndim == 2:
                max_ = image.max()
            else:
                max_ = min([_.max() for _ in image])
            upper = INITIAL_REL_MAXIMUM * max_
            self.window.display_settings_dialog.silent_minimum_update(0)
            self.window.display_settings_dialog.silent_maximum_update(upper)
        upper = self.window.display_settings_dialog.maximum.value()
        lower = self.window.display_settings_dialog.minimum.value()
        image = (image - lower) / (upper - lower)
        image = np.minimum(image, 1.0)
        image = np.maximum(image, 0.0)
        return image

    def set_mode(self, action):
        self._mode = action.text()
        self.update_cursor()

    def sizeHint(self):
        return QtCore.QSize(*self._size_hint)

    def to_8bit(self, image):
        return np.round(255 * image).astype('uint8')

    def to_left(self):
        self.pan_relative(0, 0.8)

    def to_right(self):
        self.pan_relative(0, -0.8)

    def to_up(self):
        self.pan_relative(0.8, 0)

    def to_down(self):
        self.pan_relative(-0.8, 0)

    def update_cursor(self):
        if self._mode == 'Zoom':
            self.unsetCursor()
        elif self._mode == 'Pick':
            diameter = self.window.tools_settings_dialog.pick_diameter.value()
            diameter = self.width() * diameter / self.viewport_width()
            pixmap_size = ceil(diameter)
            pixmap = QtGui.QPixmap(pixmap_size, pixmap_size)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setPen(QtGui.QColor('white'))
            offset = (pixmap_size - diameter) / 2
            painter.drawEllipse(offset, offset, diameter, diameter)
            painter.end()
            cursor = QtGui.QCursor(pixmap)
            self.setCursor(cursor)

    def update_scene(self, viewport=None, autoscale=False, use_cache=False):
        n_channels = len(self.locs)
        if n_channels:
            viewport = viewport or self.viewport
            self.draw_scene(viewport, autoscale=autoscale, use_cache=use_cache)
            self.update_cursor()

    def viewport_center(self):
        return ((self.viewport[1][0] + self.viewport[0][0]) / 2), ((self.viewport[1][1] + self.viewport[0][1]) / 2)

    def viewport_height(self):
        return self.viewport[1][0] - self.viewport[0][0]

    def viewport_size(self):
        return self.viewport_height(), self.viewport_width()

    def viewport_width(self):
        return self.viewport[1][1] - self.viewport[0][1]

    def zoom(self, factor):
        viewport_height, viewport_width = self.viewport_size()
        new_viewport_height_half = 0.5 * viewport_height * factor
        new_viewport_width_half = 0.5 * viewport_width * factor
        viewport_center_y, viewport_center_x = self.viewport_center()
        new_viewport = [(viewport_center_y - new_viewport_height_half,
                         viewport_center_x - new_viewport_width_half),
                        (viewport_center_y + new_viewport_height_half,
                         viewport_center_x + new_viewport_width_half)]
        self.update_scene(new_viewport)

    def zoom_in(self):
        self.zoom(1/ZOOM)

    def zoom_out(self):
        self.zoom(ZOOM)


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Picasso: Render')
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'icons/render.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.view = View(self)
        self.view.setMinimumSize(1, 1)
        self.setCentralWidget(self.view)
        self.display_settings_dialog = DisplaySettingsDialog(self)
        self.tools_settings_dialog = ToolsSettingsDialog(self)
        self.info_dialog = InfoDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        save_picked_action = file_menu.addAction('Save picked localizations')
        save_picked_action.setShortcut('Ctrl+S')
        save_picked_action.triggered.connect(self.save_picked_locs)
        file_menu.addSeparator()
        export_current_action = file_menu.addAction('Export current view')
        export_current_action.setShortcut('Ctrl+E')
        export_current_action.triggered.connect(self.export_current)
        export_complete_action = file_menu.addAction('Export complete image')
        export_complete_action.setShortcut('Ctrl+Shift+E')
        export_complete_action.triggered.connect(self.export_complete)
        view_menu = menu_bar.addMenu('View')
        display_settings_action = view_menu.addAction('Display settings')
        display_settings_action.setShortcut('Ctrl+D')
        display_settings_action.triggered.connect(self.display_settings_dialog.show)
        view_menu.addAction(display_settings_action)
        view_menu.addSeparator()
        to_left_action = view_menu.addAction('Left')
        to_left_action.setShortcut('Left')
        to_left_action.triggered.connect(self.view.to_left)
        to_right_action = view_menu.addAction('Right')
        to_right_action.setShortcut('Right')
        to_right_action.triggered.connect(self.view.to_right)
        to_up_action = view_menu.addAction('Up')
        to_up_action.setShortcut('Up')
        to_up_action.triggered.connect(self.view.to_up)
        to_down_action = view_menu.addAction('Down')
        to_down_action.setShortcut('Down')
        to_down_action.triggered.connect(self.view.to_down)
        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction('Zoom in')
        zoom_in_action.setShortcuts(['Ctrl++', 'Ctrl+='])
        zoom_in_action.triggered.connect(self.view.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction('Zoom out')
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.view.zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction('Fit image to window')
        fit_in_view_action.setShortcut('Ctrl+W')
        fit_in_view_action.triggered.connect(self.view.fit_in_view)
        view_menu.addAction(fit_in_view_action)
        view_menu.addSeparator()
        info_action = view_menu.addAction('Show info')
        info_action.setShortcut('Ctrl+I')
        info_action.triggered.connect(self.info_dialog.show)
        view_menu.addAction(info_action)
        tools_menu = menu_bar.addMenu('Tools')
        tools_actiongroup = QtGui.QActionGroup(menu_bar)
        zoom_tool_action = tools_actiongroup.addAction(QtGui.QAction('Zoom', tools_menu, checkable=True))
        zoom_tool_action.setShortcut('Ctrl+Z')
        tools_menu.addAction(zoom_tool_action)
        zoom_tool_action.setChecked(True)
        pick_tool_action = tools_actiongroup.addAction(QtGui.QAction('Pick', tools_menu, checkable=True))
        pick_tool_action.setShortcut('Ctrl+P')
        tools_menu.addAction(pick_tool_action)
        tools_actiongroup.triggered.connect(self.view.set_mode)
        tools_menu.addSeparator()
        tools_settings_action = tools_menu.addAction('Tools settings')
        tools_settings_action.setShortcut('Ctrl+T')
        tools_settings_action.triggered.connect(self.tools_settings_dialog.show)
        self.load_user_settings()

    def closeEvent(self, event):
        settings = io.load_user_settings()
        settings['Render']['Colormap'] = self.display_settings_dialog.colormap.currentText()
        io.save_user_settings(settings)

    def export_current(self):
        try:
            base, ext = os.path.splitext(self.view.locs_path)
        except AttributeError:
            return
        out_path = base + '.png'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save image', out_path, filter='*.png')
        if path:
            self.view.qimage.save(path)
        self.view.setMinimumSize(1, 1)

    def export_complete(self):
        try:
            base, ext = os.path.splitext(self.view.locs_path)
        except AttributeError:
            return
        out_path = base + '.png'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save image', out_path, filter='*.png')
        if path:
            movie_height, movie_width = self.view.movie_size()
            viewport = [(0, 0), (movie_height, movie_width)]
            qimage = self.view.render_scene(viewport, cache=False)
            qimage.save(path)

    def load_user_settings(self):
        settings = io.load_user_settings()
        try:
            colormap = settings['Render']['Colormap']
        except KeyError:
            colormap = 'viridis'
        for index in range(self.display_settings_dialog.colormap.count()):
            if self.display_settings_dialog.colormap.itemText(index) == colormap:
                self.display_settings_dialog.colormap.setCurrentIndex(index)
                break

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Add localizations', filter='*.hdf5')
        if path:
            self.view.add(path)

    def resizeEvent(self, event):
        self.update_info()

    def save_picked_locs(self):
        base, ext = os.path.splitext(self.view.locs_path)
        out_path = base + '_picked.hdf5'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save picked localizations', out_path, filter='*.hdf5')
        if path:
            self.view.save_picked_locs(path)

    def update_info(self):
        self.info_dialog.width_label.setText(str(self.view.width()))
        self.info_dialog.height_label.setText(str(self.view.height()))
        self.info_dialog.locs_label.setText('{:,}'.format(self.view.n_locs))


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()

    def excepthook(type, value, tback):
        message = ''.join(traceback.format_exception(type, value, tback))
        errorbox = QtGui.QMessageBox.critical(window, 'An error occured', message)
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)
    sys.excepthook = excepthook

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
