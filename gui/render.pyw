"""
    gui/render
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for rendering localization images

    :author: Joerg Schnitzbauer, 2016
"""


import sys
import os.path
import traceback
from PyQt4 import QtCore, QtGui
import numpy as np

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
_parent_directory = os.path.dirname(_this_directory)
sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import io, render


class View(QtGui.QLabel):

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setAcceptDrops(True)
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored))
        self.rubberband = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle, self)
        self.rubberband.setStyleSheet('selection-background-color: white')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        extension = os.path.splitext(path)[1].lower()
        if extension == '.hdf5':
            self.window.open(path)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if not self.rubberband.isVisible():
                self.origin = QtCore.QPoint(event.pos())
                self.rubberband.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
                self.rubberband.show()

    def mouseMoveEvent(self, event):
        if self.rubberband.isVisible():
            self.rubberband.setGeometry(QtCore.QRect(self.origin, event.pos()))

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.rubberband.isVisible():
            end = QtCore.QPoint(event.pos())
            if end.x() > self.origin.x() and end.y() > self.origin.y():
                center_y_view_new = (self.origin.y() + end.y()) / 2
                center_x_view_new = (self.origin.x() + end.x()) / 2
                y_min_image_old = self.window.center[0] - (self.window.view.height() / 2) / self.window.zoom
                x_min_image_old = self.window.center[1] - (self.window.view.width() / 2) / self.window.zoom
                center_y_image_new = center_y_view_new / self.window.zoom + y_min_image_old
                center_x_image_new = center_x_view_new / self.window.zoom + x_min_image_old
                center = (center_y_image_new, center_x_image_new)
                selection_width = end.x() - self.origin.x()
                selection_height = end.y() - self.origin.y()
                zoom = self.window.zoom * min(self.window.view.height() / selection_height, self.window.view.width() / selection_width)
                self.window.render(center, zoom)
            self.rubberband.hide()

    def resizeEvent(self, event):
        old_size = event.oldSize()
        new_size = event.size()
        zoom = min(new_size.height() / old_size.height(), new_size.width() / old_size.width())
        try:
            self.window.render(self.window.center, zoom * self.window.zoom)
        except AttributeError:
            pass


class DisplaySettingsDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Display Settings')
        self.resize(200, 0)
        self.setModal(False)
        vbox = QtGui.QVBoxLayout(self)
        contrast_groupbox = QtGui.QGroupBox('Contrast')
        vbox.addWidget(contrast_groupbox)
        contrast_grid = QtGui.QGridLayout(contrast_groupbox)
        minimum_label = QtGui.QLabel('Minimum:')
        contrast_grid.addWidget(minimum_label, 0, 0)
        self.minimum = QtGui.QDoubleSpinBox()
        self.minimum.setRange(0, 1)
        self.minimum.setSingleStep(0.1)
        self.minimum.setValue(0)
        self.minimum.setKeyboardTracking(False)
        self.minimum.valueChanged.connect(self.on_contrast_changed)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtGui.QLabel('Maximum:')
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = QtGui.QDoubleSpinBox()
        self.maximum.setRange(0, 1)
        self.maximum.setSingleStep(0.1)
        self.maximum.setValue(0.2)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.on_contrast_changed)
        contrast_grid.addWidget(self.maximum, 1, 1)
        blur_groupbox = QtGui.QGroupBox('Blur Method')
        self.blur_buttongroup = QtGui.QButtonGroup()
        self.points_button = QtGui.QRadioButton('Points (fast)')
        self.blur_buttongroup.addButton(self.points_button)
        self.convolve_button = QtGui.QRadioButton('Gaussian filter (slow for large window)')
        self.blur_buttongroup.addButton(self.convolve_button)
        self.gaussian_button = QtGui.QRadioButton('Individual Gaussians (slow for many locs.)')
        self.blur_buttongroup.addButton(self.gaussian_button)
        blur_vbox = QtGui.QVBoxLayout(blur_groupbox)
        blur_vbox.addWidget(self.points_button)
        blur_vbox.addWidget(self.convolve_button)
        blur_vbox.addWidget(self.gaussian_button)
        self.convolve_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.on_blur_button_released)
        vbox.addWidget(blur_groupbox)

    def on_contrast_changed(self, _):
        self.window.render()

    def on_blur_button_released(self, _):
        self.window.render()


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: Render')
        self.resize(768, 768)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'render.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.view = View(self)
        self.setCentralWidget(self.view)
        self.display_settings_dialog = DisplaySettingsDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        view_menu = menu_bar.addMenu('View')
        display_settings_action = view_menu.addAction('Display Settings')
        display_settings_action.setShortcut('Ctrl+D')
        display_settings_action.triggered.connect(self.display_settings_dialog.show)
        view_menu.addAction(display_settings_action)
        self.locs = None

    def fit_in_view(self):
        center = (self.info[0]['Height'] / 2, self.info[0]['Width'] / 2)
        view_height = self.view.height()
        view_width = self.view.width()
        zoom = min(view_height / self.info[0]['Height'], view_width / self.info[0]['Width'])
        self.render(center, zoom)

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open localizations', filter='*.hdf5')
        if path:
            self.open(path)

    def open(self, path):
        locs, self.info = io.load_locs(path)
        self.locs = locs[np.all(np.array([np.isfinite(locs[_]) for _ in locs.dtype.names]), axis=0)]
        self.color_locs = None
        if hasattr(self.locs, 'group'):
            valid_locs = self.locs[self.locs.group != 1]
            colors = valid_locs.group % 3
            self.color_locs = [valid_locs[colors == _] for _ in range(3)]
        self.fit_in_view()

    def to_qimage(self, image):
        minimum = float(self.display_settings_dialog.minimum.value())
        maximum = float(self.display_settings_dialog.maximum.value())
        imax = image.max()
        image = 255 * (image - imax * minimum) / (imax * maximum)
        image = np.minimum(image, 255)
        image = np.maximum(image, 0)
        image = image.astype('uint8')
        height, width = image.shape[-2:]
        self._bgra = np.zeros((height, width, 4), np.uint8, 'C')
        if image.ndim == 2:
            self._bgra[..., 1] = image
        elif image.ndim == 3:
            self._bgra[..., 0] = image[2]
            self._bgra[..., 1] = image[0]
            self._bgra[..., 2] = image[1]
        self._bgra[..., 3].fill(255)
        return QtGui.QImage(self._bgra.data, width, height, QtGui.QImage.Format_RGB32)

    def render(self, center=None, zoom=None):
        if self.locs is not None:
            if center:
                self.center = center
            if zoom:
                self.zoom = zoom
            view_height = self.view.height()
            view_width = self.view.width()
            image_height = view_height / self.zoom
            image_width = view_width / self.zoom
            min_y = self.center[0] - image_height / 2
            max_y = min_y + image_height
            min_x = self.center[1] - image_width / 2
            max_x = min_x + image_width
            viewport = [(min_y, min_x), (max_y, max_x)]
            if self.color_locs:
                image = np.array([self.render_image(_, viewport) for _ in self.color_locs])
            else:
                image = self.render_image(self.locs, viewport)
            image = self.to_qimage(image)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.view.setPixmap(pixmap)

    def render_image(self, locs, viewport):
        button = self.display_settings_dialog.blur_buttongroup.checkedButton()
        if button == self.display_settings_dialog.points_button:
            blur_method = None
        elif button == self.display_settings_dialog.convolve_button:
            blur_method = 'convolve'
        elif button == self.display_settings_dialog.gaussian_button:
            blur_method = 'gaussian'
        return render.render(locs, self.info, oversampling=self.zoom, viewport=viewport, blur_method=blur_method)


if __name__ == '__main__':
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
