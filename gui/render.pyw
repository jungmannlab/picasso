"""
    gui/render
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for rendering localization images

    :author: Joerg Schnitzbauer, 2016
"""


import sys
import os.path
import traceback
from PyQt4 import QtGui


_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
_parent_directory = os.path.dirname(_this_directory)
sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import io, widgets


class DisplaySettingsDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Display Settings')
        self.resize(200, 0)
        self.setModal(False)
        vbox = QtGui.QVBoxLayout(self)
        # Contrast
        contrast_groupbox = QtGui.QGroupBox('Contrast')
        vbox.addWidget(contrast_groupbox)
        contrast_grid = QtGui.QGridLayout(contrast_groupbox)
        minimum_label = QtGui.QLabel('Minimum:')
        contrast_grid.addWidget(minimum_label, 0, 0)
        self.minimum = QtGui.QDoubleSpinBox()
        self.minimum.setRange(0, 1)
        self.minimum.setSingleStep(0.05)
        self.minimum.setValue(0)
        self.minimum.setDecimals(3)
        self.minimum.setKeyboardTracking(False)
        self.minimum.valueChanged.connect(self.trigger_rendering)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtGui.QLabel('Maximum:')
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = QtGui.QDoubleSpinBox()
        self.maximum.setRange(0, 1)
        self.maximum.setSingleStep(0.05)
        self.maximum.setValue(0.2)
        self.maximum.setDecimals(3)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.trigger_rendering)
        contrast_grid.addWidget(self.maximum, 1, 1)
        # Blur
        blur_groupbox = QtGui.QGroupBox('Blur Method')
        self.blur_buttongroup = QtGui.QButtonGroup()
        points_button = QtGui.QRadioButton('Points (fast)')
        self.blur_buttongroup.addButton(points_button)
        convolve_button = QtGui.QRadioButton('Gaussian filter (slow for large window)')
        self.blur_buttongroup.addButton(convolve_button)
        gaussian_button = QtGui.QRadioButton('Individual Gaussians (slow for many locs.)')
        self.blur_buttongroup.addButton(gaussian_button)
        blur_vbox = QtGui.QVBoxLayout(blur_groupbox)
        blur_vbox.addWidget(points_button)
        blur_vbox.addWidget(convolve_button)
        blur_vbox.addWidget(gaussian_button)
        convolve_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.trigger_rendering)
        vbox.addWidget(blur_groupbox)
        self.blur_methods = {points_button: None, convolve_button: 'convolve', gaussian_button: 'gaussian'}
        # Scale bar
        self.scalebar_groupbox = QtGui.QGroupBox('Scale Bar')
        self.scalebar_groupbox.setCheckable(True)
        self.scalebar_groupbox.setChecked(False)
        self.scalebar_groupbox.toggled.connect(self.trigger_rendering)
        vbox.addWidget(self.scalebar_groupbox)
        scalebar_grid = QtGui.QGridLayout(self.scalebar_groupbox)
        scalebar_grid.addWidget(QtGui.QLabel('Pixel Size:'), 0, 0)
        self.pixelsize_edit = QtGui.QLineEdit('160')
        self.pixelsize_edit.editingFinished.connect(self.trigger_rendering)
        scalebar_grid.addWidget(self.pixelsize_edit, 0, 1)
        scalebar_grid.addWidget(QtGui.QLabel('Scale Bar Length (nm):'), 1, 0)
        self.scalebar_edit = QtGui.QLineEdit('500')
        self.scalebar_edit.editingFinished.connect(self.trigger_rendering)
        scalebar_grid.addWidget(self.scalebar_edit, 1, 1)

    def trigger_rendering(self, *args):
        self.window.set_display_settings()
        self.window.view.render()


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
        self.view = widgets.LocsRenderer(self)
        self.view.setAcceptDrops(True)
        self.view.dragEnterEvent = self.dragEnterEvent
        self.view.dropEvent = self.dropEvent
        self.setCentralWidget(self.view)
        self.display_settings_dialog = DisplaySettingsDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_image_action = file_menu.addAction('Save image')
        save_image_action.setShortcut('Ctrl+Shift+S')
        save_image_action.triggered.connect(self.save_image)
        view_menu = menu_bar.addMenu('View')
        display_settings_action = view_menu.addAction('Display Settings')
        display_settings_action.setShortcut('Ctrl+D')
        display_settings_action.triggered.connect(self.display_settings_dialog.show)
        view_menu.addAction(display_settings_action)
        view_menu.addSeparator()
        to_left_action = view_menu.addAction('Left')
        to_left_action.setShortcut('Left')
        to_left_action.triggered.connect(self.to_left)
        to_right_action = view_menu.addAction('Right')
        to_right_action.setShortcut('Right')
        to_right_action.triggered.connect(self.to_right)
        to_up_action = view_menu.addAction('Up')
        to_up_action.setShortcut('Up')
        to_up_action.triggered.connect(self.to_up)
        to_down_action = view_menu.addAction('Down')
        to_down_action.setShortcut('Down')
        to_down_action.triggered.connect(self.to_down)
        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction('Zoom in')
        zoom_in_action.setShortcuts(['Ctrl++', 'Ctrl+='])
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction('Zoom out')
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction('Fit image to window')
        fit_in_view_action.setShortcut('Ctrl+W')
        fit_in_view_action.triggered.connect(self.view.fit_in_view)
        view_menu.addAction(fit_in_view_action)
        self.status_bar = self.statusBar()
        self.locs = []

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
            self.open(path)

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open localizations', filter='*.hdf5')
        if path:
            self.open(path)

    def open(self, path):
        if len(self.locs) < 3:
            locs, info = io.load_locs(path)
            self.locs_path = path
            self.view.add_locs(locs, info)
            self.set_display_settings()
            if len(self.view.locs) > 1:
                self.view.render()
            else:
                self.view.fit_in_view()
        else:
            raise Exception('Maximum number of channels is 3.')

    def set_display_settings(self):
        self.view.vmin = float(self.display_settings_dialog.minimum.value())
        self.view.vmax = float(self.display_settings_dialog.maximum.value())
        self.view.pixelsize = float(self.display_settings_dialog.pixelsize_edit.text())
        self.view.scalebar = None
        if self.display_settings_dialog.scalebar_groupbox.isChecked():
            self.view.scalebar = float(self.display_settings_dialog.scalebar_edit.text())
        button = self.display_settings_dialog.blur_buttongroup.checkedButton()
        self.view.blur_method = self.display_settings_dialog.blur_methods[button]

    def save_image(self):
        base, ext = os.path.splitext(self.locs_path)
        out_path = base + '.png'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save image', out_path, filter='*.png')
        if path:
            self.view.save(path)

    def to_left(self):
        self.view.center[1] = self.view.center[1] - 0.8 * self.view.width() / self.view.zoom
        self.view.render()

    def to_right(self):
        self.view.center[1] = self.view.center[1] + 0.8 * self.view.width() / self.view.zoom
        self.view.render()

    def to_up(self):
        self.view.center[0] = self.view.center[0] - 0.8 * self.view.height() / self.view.zoom
        self.view.render()

    def to_down(self):
        self.view.center[0] = self.view.center[0] + 0.8 * self.view.height() / self.view.zoom
        self.view.render()

    def zoom_in(self):
        self.view.zoom *= 10/7
        self.view.render()

    def zoom_out(self):
        self.view.zoom *= 7/10
        self.view.render()


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
