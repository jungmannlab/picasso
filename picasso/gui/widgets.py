"""
    picasso.widgets
    ~~~~~~~~~~~~~~~

    GUI widgets for quick re-using

    :author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) 2015 Jungmann Lab, Max Planck Institute of Biochemistry
"""

from PyQt4 import QtCore as _QtCore
from PyQt4 import QtGui as _QtGui
import numpy as _np
import time as _time
from numpy.lib.recfunctions import stack_arrays as _stack_arrays
import matplotlib.pyplot as _plt
from .. import render as _render
from .. import lib as _lib
from .. import io as _io


class LocsRenderer(_QtGui.QLabel):

    rendered = _QtCore.pyqtSignal(int, int, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(_QtGui.QSizePolicy(_QtGui.QSizePolicy.Ignored, _QtGui.QSizePolicy.Ignored))
        self.rubberband = _QtGui.QRubberBand(_QtGui.QRubberBand.Rectangle, self)
        self.rubberband.setStyleSheet('selection-background-color: white')
        self.pan = False
        self.center = None
        self._zoom = None
        self.vmin = 0
        self.vmax = 1
        self.pixelsize = None
        self.scalebar = None
        self.blur_method = None
        self.min_blur_width = None
        self.locs = []
        self.infos = []
        self._mode = 'zoom'
        self._pick_diameter = 1
        self.picked_locs = []
        self.set_colormap('viridis')

    def add_locs(self, locs, info):
        locs = _lib.ensure_finite(locs)
        self.locs.append(locs)
        self.infos.append(info)

    def clear(self):
        self.locs = []

    def fit_in_view(self):
        self.center = [self.infos[0][0]['Height'] / 2, self.infos[0][0]['Width'] / 2]
        view_height = self.height()
        view_width = self.width()
        self._zoom = min(view_height / self.infos[0][0]['Height'], view_width / self.infos[0][0]['Width'])
        return self.render()

    def map_to_movie(self, position):
        left_edge = self.center[1] - (self.width() / self._zoom) / 2
        x_movie_rel = position.x() / self._zoom
        x_movie = left_edge + x_movie_rel
        top_edge = self.center[0] - (self.height() / self._zoom) / 2
        y_movie_rel = position.y() / self._zoom
        y_movie = top_edge + y_movie_rel
        return y_movie, x_movie

    def mousePressEvent(self, event):
        if self._mode == 'zoom':
            if event.button() == _QtCore.Qt.LeftButton:
                if not self.rubberband.isVisible():
                    self.origin = _QtCore.QPoint(event.pos())
                    self.rubberband.setGeometry(_QtCore.QRect(self.origin, _QtCore.QSize()))
                    self.rubberband.show()
            elif event.button() == _QtCore.Qt.RightButton:
                self.pan = True
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()
                self.setCursor(_QtCore.Qt.ClosedHandCursor)
                event.accept()
            else:
                event.ignore()

    def mouseMoveEvent(self, event):
        if self._mode == 'zoom':
            if self.rubberband.isVisible():
                self.rubberband.setGeometry(_QtCore.QRect(self.origin, event.pos()))
            if self.pan:
                self.center[1] -= (event.x() - self.pan_start_x) / self._zoom
                self.center[0] -= (event.y() - self.pan_start_y) / self._zoom
                self.render()
                self.pan_start_x = event.x()
                self.pan_start_y = event.y()

    def mouseReleaseEvent(self, event):
        if self._mode == 'zoom':
            if event.button() == _QtCore.Qt.LeftButton and self.rubberband.isVisible():
                end = _QtCore.QPoint(event.pos())
                if end.x() > self.origin.x() and end.y() > self.origin.y():
                    center_y_view_new = (self.origin.y() + end.y()) / 2
                    center_x_view_new = (self.origin.x() + end.x()) / 2
                    y_min_image_old = self.center[0] - (self.height() / 2) / self._zoom
                    x_min_image_old = self.center[1] - (self.width() / 2) / self._zoom
                    center_y_image_new = center_y_view_new / self._zoom + y_min_image_old
                    center_x_image_new = center_x_view_new / self._zoom + x_min_image_old
                    self.center = [center_y_image_new, center_x_image_new]
                    selection_width = end.x() - self.origin.x()
                    selection_height = end.y() - self.origin.y()
                    self._zoom *= min(self.height() / selection_height, self.width() / selection_width)
                    self.render()
                self.rubberband.hide()
            elif event.button() == _QtCore.Qt.RightButton:
                self.pan = False
                self.setCursor(_QtCore.Qt.ArrowCursor)
                event.accept()
            else:
                event.ignore()
        elif self._mode == 'pick':
            center = self.map_to_movie(event.pos())
            dx = self.locs[0].x - center[1]
            dy = self.locs[0].y - center[0]
            is_picked = _np.sqrt(dx**2 + dy**2) < self._pick_diameter / 2
            self.picked_locs.append(self.locs[0][is_picked])
            self.locs[0] = self.locs[0][~is_picked]
            self.render()

    def resizeEvent(self, event):
        self.render()

    def render(self):
        n_channels = len(self.locs)
        if n_channels:
            view_height = self.height()
            view_width = self.width()
            image_height = view_height / self._zoom
            image_width = view_width / self._zoom
            min_y = self.center[0] - image_height / 2
            max_y = min_y + image_height
            min_x = self.center[1] - image_width / 2
            max_x = min_x + image_width
            viewport = [(min_y, min_x), (max_y, max_x)]
            if n_channels == 1:
                locs = self.locs[0]
                if hasattr(locs, 'group'):
                    colors = locs.group % 3
                    color_locs = [locs[colors == _] for _ in range(3)]
                    T, N, image = self.render_colors(color_locs, viewport)
                    _, Y, X = image.shape
                else:
                    T, N, image = self.render_image(locs, viewport)
                    Y, X = image.shape
            elif n_channels == 2 or n_channels == 3:
                T, N, image = self.render_colors(self.locs, viewport)
                _, Y, X = image.shape
            else:
                raise Exception('Cannot display more than 3 channels.')
            image = self.to_qimage(image)
            self.image = self.draw_scalebar(image)
            pixmap = _QtGui.QPixmap.fromImage(self.image)
            self.setPixmap(pixmap)
            self.rendered.emit(N, X, Y, T)

    def render_colors(self, color_locs, viewport):
        rendering = [self.render_image(locs, viewport) for locs in color_locs]
        image = _np.array([_[2] for _ in rendering])
        N = _np.sum([_[1] for _ in rendering])
        T = _np.sum([_[0] for _ in rendering])
        return T, N, image

    def render_image(self, locs, viewport):
        t0 = _time.time()
        N, image = _render.render(locs, self.infos[0], oversampling=self._zoom, viewport=viewport,
                                  blur_method=self.blur_method, min_blur_width=self.min_blur_width)
        T = _time.time() - t0
        return T, N, image

    def set_colormap(self, name):
        self._cmap = _np.uint8(_np.round(255 * _plt.get_cmap(name)(_np.arange(256))))

    def set_mode(self, mode):
        if mode in ['zoom', 'pick']:
            self._mode = mode
            self.update_cursor()

    def set_pick_diameter(self, diameter):
        self._pick_diameter = diameter
        self.update_cursor()

    def set_zoom(self, zoom):
        self._zoom = zoom
        self.update_cursor()

    def to_8bit(self, image):
        imax = image.max()
        upper = self.vmax * imax
        lower = self.vmin * imax
        image = _np.round(255 * (image - lower) / (upper - lower))
        image = _np.maximum(image, 0)
        image = _np.minimum(image, 255)
        return image.astype('uint8')

    def to_qimage(self, image):
        height, width = image.shape[-2:]
        if image.ndim == 2:
            image = self.to_8bit(image)
            self._bgra = _np.zeros((height, width, 4), dtype=_np.uint8, order='C')
            self._bgra[..., 0] = self._cmap[:, 2][image]
            self._bgra[..., 1] = self._cmap[:, 1][image]
            self._bgra[..., 2] = self._cmap[:, 0][image]
        elif image.ndim == 3:
            self._bgra = _np.zeros((height, width, 4), order='C')
            self._bgra[..., 0] = image[0] + image[1]
            self._bgra[..., 1] = image[0]
            self._bgra[..., 2] = image[1]
            if len(image) > 2:
                self._bgra[..., 1] += image[2]
                self._bgra[..., 2] += image[2]
            self._bgra = self.to_8bit(self._bgra)
        self._bgra[..., 3].fill(255)
        return _QtGui.QImage(self._bgra.data, width, height, _QtGui.QImage.Format_RGB32)

    def draw_scalebar(self, image):
        if self.pixelsize and self.scalebar:
            length_camerapxl = self.scalebar / self.pixelsize
            length_displaypxl = int(round(self._zoom * length_camerapxl))
            height = max(int(round(0.15 * length_displaypxl)), 1)
            painter = _QtGui.QPainter(image)
            painter.setBrush(_QtGui.QBrush(_QtGui.QColor('white')))
            x = self.width() - length_displaypxl - 20
            y = self.height() - height - 20
            painter.drawRect(x, y, length_displaypxl + 1, height + 1)
        return image

    def save(self, path):
        self.image.save(path)

    def save_picked_locs(self, path):
        picked_locs = []
        for i, group_locs in enumerate(self.picked_locs):
            group = i * _np.ones(len(group_locs), dtype=_np.int32)
            picked_locs.append(_lib.append_to_rec(group_locs, group, 'group'))
        locs = _stack_arrays(picked_locs, asrecarray=True, usemask=False)
        pick_info = {'Generated by:': 'Picasso Render', 'Pick Diameter:': self._pick_diameter}
        _io.save_locs(path, locs, self.infos[0] + [pick_info])

    def update_cursor(self):
        if self._mode == 'zoom':
            self.unsetCursor()
        elif self._mode == 'pick':
            pixmap_size = int(round(self._zoom * self._pick_diameter))
            pixmap = _QtGui.QPixmap(pixmap_size, pixmap_size)
            pixmap.fill(_QtCore.Qt.transparent)
            painter = _QtGui.QPainter(pixmap)
            painter.setPen(_QtGui.QColor('white'))
            painter.drawEllipse(0, 0, pixmap.width()-1, pixmap.height()-1)
            painter.end()
            cursor = _QtGui.QCursor(pixmap)
            self.setCursor(cursor)

    def zoom(self):
        return self._zoom
