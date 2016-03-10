"""
    picasso.widgets
    ~~~~~~~~~~~~~~~

    GUI widgets for quick re-using

    :author: Joerg Schnitzbauer, 2015
"""

from PyQt4 import QtCore as _QtCore
from PyQt4 import QtGui as _QtGui
import numpy as _np
import os.path as _ospath
import sys as _sys


_this_file = _ospath.abspath(__file__)
_this_directory = _ospath.dirname(_this_file)
_parent_directory = _ospath.dirname(_this_directory)
_sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import render as _render
from picasso import lib as _lib


class LocsRenderer(_QtGui.QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(_QtGui.QSizePolicy(_QtGui.QSizePolicy.Ignored, _QtGui.QSizePolicy.Ignored))
        self.rubberband = _QtGui.QRubberBand(_QtGui.QRubberBand.Rectangle, self)
        self.rubberband.setStyleSheet('selection-background-color: white')
        self.pan = False
        self.center = None
        self.zoom = None
        self.vmin = 0
        self.vmax = 1
        self.pixelsize = None
        self.scalebar = None
        self.blur_method = None
        self.min_blur_width = None
        self.locs = []
        self.info = None

    def clear(self):
        self.locs = []

    def fit_in_view(self):
        self.center = [self.info[0]['Height'] / 2, self.info[0]['Width'] / 2]
        view_height = self.height()
        view_width = self.width()
        self.zoom = min(view_height / self.info[0]['Height'], view_width / self.info[0]['Width'])
        self.render()

    def mousePressEvent(self, event):
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
        if self.rubberband.isVisible():
            self.rubberband.setGeometry(_QtCore.QRect(self.origin, event.pos()))
        if self.pan:
            self.center[1] -= (event.x() - self.pan_start_x) / self.zoom
            self.center[0] -= (event.y() - self.pan_start_y) / self.zoom
            self.render()
            self.pan_start_x = event.x()
            self.pan_start_y = event.y()

    def mouseReleaseEvent(self, event):
        if event.button() == _QtCore.Qt.LeftButton and self.rubberband.isVisible():
            end = _QtCore.QPoint(event.pos())
            if end.x() > self.origin.x() and end.y() > self.origin.y():
                center_y_view_new = (self.origin.y() + end.y()) / 2
                center_x_view_new = (self.origin.x() + end.x()) / 2
                y_min_image_old = self.center[0] - (self.height() / 2) / self.zoom
                x_min_image_old = self.center[1] - (self.width() / 2) / self.zoom
                center_y_image_new = center_y_view_new / self.zoom + y_min_image_old
                center_x_image_new = center_x_view_new / self.zoom + x_min_image_old
                self.center = [center_y_image_new, center_x_image_new]
                selection_width = end.x() - self.origin.x()
                selection_height = end.y() - self.origin.y()
                self.zoom *= min(self.height() / selection_height, self.width() / selection_width)
                self.render()
            self.rubberband.hide()
        elif event.button() == _QtCore.Qt.RightButton:
            self.pan = False
            self.setCursor(_QtCore.Qt.ArrowCursor)
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        self.render()

    def render(self):
        n_channels = len(self.locs)
        if n_channels:
            view_height = self.height()
            view_width = self.width()
            image_height = view_height / self.zoom
            image_width = view_width / self.zoom
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
                    N, image = self.render_colors(color_locs, viewport)
                else:
                    N, image = self.render_image(locs, viewport)
            elif n_channels == 2 or n_channels == 3:
                N, image = self.render_colors(self.locs, viewport)
            else:
                raise Exception('Cannot display more than 3 channels.')
            # self.status_bar.showMessage('{:,} localizations in FOV'.format(N))    # Needs to happen top-level
            image = self.to_qimage(image)
            self.image = self.draw_scalebar(image)
            pixmap = _QtGui.QPixmap.fromImage(self.image)
            self.setPixmap(pixmap)

    def render_colors(self, color_locs, viewport):
        rendering = [self.render_image(locs, viewport) for locs in color_locs]
        image = _np.array([_[1] for _ in rendering])
        N = _np.sum([_[0] for _ in rendering])
        return N, image

    def render_image(self, locs, viewport):
        return _render.render(locs, self.info, oversampling=self.zoom, viewport=viewport,
                              blur_method=self.blur_method, min_blur_width=self.min_blur_width)

    def to_qimage(self, image):
        imax = image.max()
        image = 255 * (image - imax * self.vmin) / (imax * self.vmax)
        image = _np.minimum(image, 255)
        image = _np.maximum(image, 0)
        image = image.astype('uint8')
        height, width = image.shape[-2:]
        self._bgra = _np.zeros((height, width, 4), _np.uint8, 'C')
        if image.ndim == 2:
            self._bgra[..., 1] = image
        elif image.ndim == 3:
            self._bgra[..., 1] = image[0]
            self._bgra[..., 2] = image[1]
            if len(image) > 2:
                self._bgra[..., 0] = image[2]
        self._bgra[..., 3].fill(255)
        return _QtGui.QImage(self._bgra.data, width, height, _QtGui.QImage.Format_RGB32)

    def draw_scalebar(self, image):
        if self.pixelsize and self.scalebar:
            length_camerapxl = self.scalebar / self.pixelsize
            length_displaypxl = int(round(self.zoom * length_camerapxl))
            height = max(int(round(0.15 * length_displaypxl)), 1)
            painter = _QtGui.QPainter(image)
            painter.setBrush(_QtGui.QBrush(_QtGui.QColor('white')))
            x = self.width() - length_displaypxl - 20
            y = self.height() - height - 20
            painter.drawRect(x, y, length_displaypxl + 1, height + 1)
        return image

    def add_locs(self, locs, info):
        locs = _lib.ensure_finite(locs)
        self.locs.append(locs)
        self.info = info

    def save(self, path):
        self.image.save(path)
