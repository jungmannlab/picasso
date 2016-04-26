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


class ToolsSettingsDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle('Tools Settings')
        self.setModal(False)
        grid = QtGui.QGridLayout(self)
        grid.addWidget(QtGui.QLabel('Pick Radius:'), 0, 0)
        pick_diameter = QtGui.QDoubleSpinBox()
        pick_diameter.setRange(0, 999999)
        pick_diameter.setValue(1)
        pick_diameter.setSingleStep(0.1)
        pick_diameter.setDecimals(3)
        pick_diameter.setKeyboardTracking(False)
        pick_diameter.valueChanged.connect(window.view.set_pick_diameter)
        grid.addWidget(pick_diameter, 0, 1)


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
        self.minimum.setRange(0, 999999)
        self.minimum.setSingleStep(5)
        self.minimum.setValue(0)
        self.minimum.setKeyboardTracking(False)
        self.minimum.valueChanged.connect(self.trigger_rendering)
        contrast_grid.addWidget(self.minimum, 0, 1)
        maximum_label = QtGui.QLabel('Maximum:')
        contrast_grid.addWidget(maximum_label, 1, 0)
        self.maximum = QtGui.QDoubleSpinBox()
        self.maximum.setRange(0, 999999)
        self.maximum.setSingleStep(5)
        self.maximum.setValue(20)
        self.maximum.setKeyboardTracking(False)
        self.maximum.valueChanged.connect(self.trigger_rendering)
        contrast_grid.addWidget(self.maximum, 1, 1)
        contrast_grid.addWidget(QtGui.QLabel('Colormap:'), 2, 0)
        self.colormap = QtGui.QComboBox()
        self.colormap.addItems(sorted(['hot', 'viridis', 'inferno', 'plasma', 'magma', 'gray']))
        for i in range(self.colormap.count()):
            if self.colormap.itemText(i) == 'viridis':
                self.colormap.setCurrentIndex(i)
        contrast_grid.addWidget(self.colormap, 2, 1)
        self.colormap.currentIndexChanged.connect(self.trigger_rendering)
        # Blur
        blur_groupbox = QtGui.QGroupBox('Blur')
        blur_grid = QtGui.QGridLayout(blur_groupbox)
        self.blur_buttongroup = QtGui.QButtonGroup()
        points_button = QtGui.QRadioButton('Points (fast)')
        self.blur_buttongroup.addButton(points_button)
        convolve_button = QtGui.QRadioButton('Gaussian filter (slow for large window)')
        self.blur_buttongroup.addButton(convolve_button)
        gaussian_button = QtGui.QRadioButton('Individual Gaussians (slow for many locs.)')
        self.blur_buttongroup.addButton(gaussian_button)
        blur_grid.addWidget(points_button, 0, 0, 1, 2)
        blur_grid.addWidget(convolve_button, 1, 0, 1, 2)
        blur_grid.addWidget(gaussian_button, 2, 0, 1, 2)
        convolve_button.setChecked(True)
        self.blur_buttongroup.buttonReleased.connect(self.trigger_rendering)
        blur_grid.addWidget(QtGui.QLabel('Min. Blur (cam. pixel):'), 3, 0, 1, 1)
        self.min_blur_width = QtGui.QDoubleSpinBox()
        self.min_blur_width.setRange(0, 999999)
        self.min_blur_width.setSingleStep(0.01)
        self.min_blur_width.setValue(0)
        self.min_blur_width.setDecimals(3)
        self.min_blur_width.setKeyboardTracking(False)
        self.min_blur_width.valueChanged.connect(self.trigger_rendering)
        blur_grid.addWidget(self.min_blur_width, 3, 1, 1, 1)
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
        # Window Size
        window_groupbox = QtGui.QGroupBox('Dimensions')
        window_grid = QtGui.QGridLayout(window_groupbox)
        window_grid.addWidget(QtGui.QLabel('Width:'), 0, 0)
        self.width_spinbox = QtGui.QSpinBox()
        self.width_spinbox.setRange(1, 999999)
        self.width_spinbox.setValue(768)
        self.width_spinbox.setKeyboardTracking(False)
        self.width_spinbox.valueChanged.connect(self.resize_view)
        window_grid.addWidget(self.width_spinbox, 0, 1)
        window_grid.addWidget(QtGui.QLabel('Height:'), 1, 0)
        self.height_spinbox = QtGui.QSpinBox()
        self.height_spinbox.setRange(1, 999999)
        self.height_spinbox.setValue(768)
        self.height_spinbox.setKeyboardTracking(False)
        self.height_spinbox.valueChanged.connect(self.resize_view)
        window_grid.addWidget(self.height_spinbox, 1, 1)
        vbox.addWidget(window_groupbox)

    def trigger_rendering(self, *args):
        self.window.set_display_settings()
        self.window.view.render()

    def resize_view(self, *args):
        self.window.update_view_size()


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: Render')
        # self.resize(768, 768)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'render.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.view = widgets.LocsRenderer()
        self.view.setAcceptDrops(True)
        self.view.dragEnterEvent = self.dragEnterEvent
        self.view.dropEvent = self.dropEvent
        self.view.rendered.connect(self.update_status_bar)
        self.setCentralWidget(self.view)
        self.display_settings_dialog = DisplaySettingsDialog(self)
        self.tools_settings_dialog = ToolsSettingsDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_image_action = file_menu.addAction('Save Image')
        save_image_action.setShortcut('Ctrl+Shift+S')
        save_image_action.triggered.connect(self.save_image)
        save_picked_action = file_menu.addAction('Save Picked Localizations')
        save_picked_action.setShortcut('Ctrl+Alt+Shift+S')
        save_picked_action.triggered.connect(self.save_picked_locs)
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
        tools_menu = menu_bar.addMenu('Tools')
        tools_actiongroup = QtGui.QActionGroup(menu_bar)
        zoom_tool_action = tools_actiongroup.addAction(QtGui.QAction('Zoom', tools_menu, checkable=True))
        zoom_tool_action.setShortcut('Ctrl+Z')
        tools_menu.addAction(zoom_tool_action)
        zoom_tool_action.setChecked(True)
        pick_tool_action = tools_actiongroup.addAction(QtGui.QAction('Pick', tools_menu, checkable=True))
        pick_tool_action.setShortcut('Ctrl+P')
        tools_menu.addAction(pick_tool_action)
        tools_actiongroup.triggered.connect(self.on_tool_changed)
        tools_menu.addSeparator()
        tools_settings_action = tools_menu.addAction('Tools Setttings')
        tools_settings_action.setShortcut('Ctrl+T')
        tools_settings_action.triggered.connect(self.tools_settings_dialog.show)
        self.status_bar = self.statusBar()
        self.update_view_size()

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
        if len(self.view.locs) < 3:
            locs, info = io.load_locs(path)
            self.view.add_locs(locs, info)
            self.set_display_settings()
            if len(self.view.locs) > 1:
                self.view.render()
            else:
                self.view.set_tracking(False)
                self.update_view_size()
                self.view.set_tracking(True)
                self.view.fit_in_view()
            if len(self.view.locs) == 1:
                self.locs_path = path
        else:
            raise Exception('Maximum number of channels is 3.')

    def set_display_settings(self):
        self.view.vmin = float(self.display_settings_dialog.minimum.value())/100
        self.view.vmax = float(self.display_settings_dialog.maximum.value())/100
        self.view.pixelsize = float(self.display_settings_dialog.pixelsize_edit.text())
        self.view.scalebar = None
        if self.display_settings_dialog.scalebar_groupbox.isChecked():
            self.view.scalebar = float(self.display_settings_dialog.scalebar_edit.text())
        button = self.display_settings_dialog.blur_buttongroup.checkedButton()
        self.view.blur_method = self.display_settings_dialog.blur_methods[button]
        self.view.min_blur_width = float(self.display_settings_dialog.min_blur_width.value())
        self.view.set_colormap(self.display_settings_dialog.colormap.currentText())

    def save_image(self):
        base, ext = os.path.splitext(self.locs_path)
        out_path = base + '.png'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save image', out_path, filter='*.png')
        if path:
            self.view.save(path)

    def save_picked_locs(self):
        base, ext = os.path.splitext(self.locs_path)
        out_path = base + '_picked.hdf5'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save picked localizations', out_path, filter='*.hdf5')
        if path:
            self.view.save_picked_locs(path)

    def update_view_size(self):
        width = self.display_settings_dialog.width_spinbox.value()
        height = self.display_settings_dialog.height_spinbox.value()
        self.view.setMinimumSize(width, height)
        self.view.setMaximumSize(width, height)
        self.view.resize(width, height)
        self.adjustSize()
        self.view.setMinimumSize(0, 0)
        self.view.setMaximumSize(999999, 999999)

    def to_left(self):
        self.view.center[1] = self.view.center[1] - 0.8 * self.view.width() / self.view.zoom()
        self.view.render()

    def to_right(self):
        self.view.center[1] = self.view.center[1] + 0.8 * self.view.width() / self.view.zoom()
        self.view.render()

    def to_up(self):
        self.view.center[0] = self.view.center[0] - 0.8 * self.view.height() / self.view.zoom()
        self.view.render()

    def to_down(self):
        self.view.center[0] = self.view.center[0] + 0.8 * self.view.height() / self.view.zoom()
        self.view.render()

    def zoom_in(self):
        self.view.set_zoom(self.view.zoom() * 10 / 7)
        self.view.render()

    def zoom_out(self):
        self.view.set_zoom(self.view.zoom() * 7 / 10)
        self.view.render()

    def update_status_bar(self, N, X, Y, T):
        self.status_bar.showMessage('Rendered {:,} localizations on {}x{} pixels in {:.4f} seconds.'.format(N, X, Y, T))

    def on_tool_changed(self, action):
        if action.text() == 'Zoom':
            self.view.set_mode('zoom')
        elif action.text() == 'Pick':
            self.view.set_mode('pick')


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
