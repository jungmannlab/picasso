#!/usr/bin/env python
"""
    gui/localize
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.localize

    :author: Joerg Schnitzbauer, 2015
"""

import os.path
import sys
import yaml
from PyQt4 import QtCore, QtGui
import time
import numpy as np
import traceback


_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
_parent_directory = os.path.dirname(_this_directory)
sys.path.insert(0, _parent_directory)    # We want to use the local picasso instead the system-wide
from picasso import io, localize


CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
DEFAULT_PARAMETERS = {'ROI': 7, 'Minimum LGM': 5000}


class View(QtGui.QGraphicsView):
    """ The central widget which shows `Scene` objects of individual frames """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.setAcceptDrops(True)

    def wheelEvent(self, event):
        """ Implements zoooming with the mouse wheel """
        scale = 1.008 ** (-event.delta())
        self.scale(scale, scale)


class Scene(QtGui.QGraphicsScene):
    """ Scenes render indivdual frames and can be displayed in a `View` widget """

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.dragMoveEvent = self.dragEnterEvent

    def path_from_drop(self, event):
        url = event.mimeData().urls()[0]
        path = url.toLocalFile()
        base, extension = os.path.splitext(path)
        return path, extension

    def drop_has_valid_url(self, event):
        if not event.mimeData().hasUrls():
            return False
        path, extension = self.path_from_drop(event)
        if extension.lower() not in ['.raw', '.yaml']:
            return False
        return True

    def dragEnterEvent(self, event):
        if self.drop_has_valid_url(event):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """ Loads raw movies or yaml parameters when dropped into the scene """
        path, extension = self.path_from_drop(event)
        if extension == '.raw':
            self.window.open(path)
        elif extension == '.yaml':
            self.window.load_parameters(path)
        else:
            pass  # TODO: send message to user


class FitMarker(QtGui.QGraphicsItemGroup):

    def __init__(self, x, y, size, parent=None):
        super().__init__(parent)
        L = size/2
        line1 = QtGui.QGraphicsLineItem(x-L, y-L, x+L, y+L)
        line1.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line1)
        line2 = QtGui.QGraphicsLineItem(x-L, y+L, x+L, y-L)
        line2.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line2)


class OddSpinBox(QtGui.QSpinBox):
    """ A spinbox that allows only odd numbers """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSingleStep(2)
        self.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self, value):
        if value % 2 == 0:
            self.setValue(value + 1)


class ParametersDialog(QtGui.QDialog):
    """ The dialog showing analysis parameters """

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Parameters')
        self.resize(300, 0)
        self.setModal(False)
        grid = QtGui.QGridLayout(self)

        # ROI
        grid.addWidget(QtGui.QLabel('ROI side length:'), 0, 0)
        self.roi_spinbox = OddSpinBox()
        self.roi_spinbox.setValue(DEFAULT_PARAMETERS['ROI'])
        self.roi_spinbox.valueChanged.connect(self.on_roi_changed)
        grid.addWidget(self.roi_spinbox, 0, 1)

        # Minimum LGM
        grid.addWidget(QtGui.QLabel('Minimum LGM:'), 1, 0)
        self.mlgm_spinbox = QtGui.QSpinBox()
        self.mlgm_spinbox.setRange(0, 999999)
        self.mlgm_spinbox.setValue(DEFAULT_PARAMETERS['Minimum LGM'])
        self.mlgm_spinbox.setKeyboardTracking(False)
        self.mlgm_spinbox.valueChanged.connect(self.on_mlgm_spinbox_changed)
        grid.addWidget(self.mlgm_spinbox, 1, 1)

        # Slider
        self.mlgm_slider = QtGui.QSlider()
        self.mlgm_slider.setOrientation(QtCore.Qt.Horizontal)
        self.mlgm_slider.setRange(0, 10000)
        self.mlgm_slider.setValue(DEFAULT_PARAMETERS['Minimum LGM'])
        self.mlgm_slider.setSingleStep(1)
        self.mlgm_slider.setPageStep(20)
        self.mlgm_slider.valueChanged.connect(self.on_mlgm_slider_changed)
        grid.addWidget(self.mlgm_slider, 2, 0, 1, 2)

        hbox = QtGui.QHBoxLayout()
        grid.addLayout(hbox, 3, 0, 1, 2)

        # Min SpinBox
        self.mlgm_min_spinbox = QtGui.QSpinBox()
        self.mlgm_min_spinbox.setRange(0, 999999)
        self.mlgm_min_spinbox.setKeyboardTracking(False)
        self.mlgm_min_spinbox.setValue(0)
        self.mlgm_min_spinbox.valueChanged.connect(self.on_mlgm_min_changed)
        hbox.addWidget(self.mlgm_min_spinbox)

        hbox.addStretch(1)

        # Max SpinBox
        self.mlgm_max_spinbox = QtGui.QSpinBox()
        self.mlgm_max_spinbox.setKeyboardTracking(False)
        self.mlgm_max_spinbox.setRange(0, 999999)
        self.mlgm_max_spinbox.setValue(10000)
        self.mlgm_max_spinbox.valueChanged.connect(self.on_mlgm_max_changed)
        hbox.addWidget(self.mlgm_max_spinbox)

        self.preview_checkbox = QtGui.QCheckBox('Preview')
        self.preview_checkbox.setTristate(False)
        # self.preview_checkbox.setChecked(True)
        self.preview_checkbox.stateChanged.connect(self.on_preview_changed)
        grid.addWidget(self.preview_checkbox, 4, 0)

    def on_roi_changed(self, value):
        self.window.on_parameters_changed()

    def on_mlgm_spinbox_changed(self, value):
        if value < self.mlgm_slider.minimum():
            self.mlgm_min_spinbox.setValue(value)
        if value > self.mlgm_slider.maximum():
            self.mlgm_max_spinbox.setValue(value)
        self.mlgm_slider.setValue(value)

    def on_mlgm_slider_changed(self, value):
        self.mlgm_spinbox.setValue(value)
        if self.preview_checkbox.isChecked():
            self.window.on_parameters_changed()

    def on_mlgm_min_changed(self, value):
        self.mlgm_slider.setMinimum(value)

    def on_mlgm_max_changed(self, value):
        self.mlgm_slider.setMaximum(value)

    def on_preview_changed(self, state):
        self.window.draw_frame()


class ContrastDialog(QtGui.QDialog):

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle('Contrast')
        self.resize(200, 0)
        self.setModal(False)
        grid = QtGui.QGridLayout(self)
        black_label = QtGui.QLabel('Black:')
        grid.addWidget(black_label, 0, 0)
        self.black_spinbox = QtGui.QSpinBox()
        self.black_spinbox.setKeyboardTracking(False)
        self.black_spinbox.setRange(0, 999999)
        self.black_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.black_spinbox, 0, 1)
        white_label = QtGui.QLabel('White:')
        grid.addWidget(white_label, 1, 0)
        self.white_spinbox = QtGui.QSpinBox()
        self.white_spinbox.setKeyboardTracking(False)
        self.white_spinbox.setRange(0, 999999)
        self.white_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.white_spinbox, 1, 1)
        self.auto_checkbox = QtGui.QCheckBox('Auto')
        self.auto_checkbox.setTristate(False)
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.on_auto_changed)
        grid.addWidget(self.auto_checkbox, 2, 0, 1, 2)
        self.silent_contrast_change = False

    def change_contrast_silently(self, black, white):
        self.silent_contrast_change = True
        self.black_spinbox.setValue(black)
        self.white_spinbox.setValue(white)
        self.silent_contrast_change = False

    def on_contrast_changed(self, value):
        if not self.silent_contrast_change:
            self.auto_checkbox.setChecked(False)
            self.window.draw_frame()

    def on_auto_changed(self, state):
        if state:
            movie = self.window.movie
            frame_number = self.window.current_frame_number
            frame = movie[frame_number]
            self.change_contrast_silently(frame.min(), frame.max())
            self.window.draw_frame()


class Window(QtGui.QMainWindow):
    """ The main window """

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: Localize')
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'localize.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(768, 768)
        self.parameters_dialog = ParametersDialog(self)
        self.contrast_dialog = ContrastDialog(self)
        self.init_menu_bar()
        self.view = View()
        self.setCentralWidget(self.view)
        self.scene = Scene(self)
        self.view.setScene(self.scene)
        self.status_bar = self.statusBar()
        self.status_bar_frame_indicator = QtGui.QLabel()
        self.status_bar.addPermanentWidget(self.status_bar_frame_indicator)

        #: Holds the current movie as a numpy memmap in the format (frame, y, x)
        self.movie = None

        #: A dictionary of analysis parameters used for the last operation
        self.last_identification_parameters = None

        #: A numpy.recarray of identifcations with fields frame, x and y
        self.identifications = None

        self.ready_for_fit = False

        self.locs = None

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open movie')
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction('Save localizations')
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_locs)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        open_parameters_action = file_menu.addAction('Load parameters')
        open_parameters_action.setShortcut('Ctrl+Shift+O')
        open_parameters_action.triggered.connect(self.open_parameters)
        file_menu.addAction(open_parameters_action)
        save_parameters_action = file_menu.addAction('Save parameters')
        save_parameters_action.setShortcut('Ctrl+Shift+S')
        save_parameters_action.triggered.connect(self.save_parameters)
        file_menu.addAction(save_parameters_action)

        """ View """
        view_menu = menu_bar.addMenu('View')
        previous_frame_action = view_menu.addAction('Previous frame')
        previous_frame_action.setShortcut('Left')
        previous_frame_action.triggered.connect(self.previous_frame)
        view_menu.addAction(previous_frame_action)
        next_frame_action = view_menu.addAction('Next frame')
        next_frame_action.setShortcut('Right')
        next_frame_action.triggered.connect(self.next_frame)
        view_menu.addAction(next_frame_action)
        view_menu.addSeparator()
        first_frame_action = view_menu.addAction('First frame')
        first_frame_action.setShortcut('Ctrl+Home')
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction('Last frame')
        last_frame_action.setShortcut('Ctrl+End')
        last_frame_action.triggered.connect(self.last_frame)
        view_menu.addAction(last_frame_action)
        go_to_frame_action = view_menu.addAction('Go to frame')
        go_to_frame_action.setShortcut('Ctrl+G')
        go_to_frame_action.triggered.connect(self.to_frame)
        view_menu.addAction(go_to_frame_action)
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
        fit_in_view_action.triggered.connect(self.fit_in_view)
        view_menu.addAction(fit_in_view_action)
        view_menu.addSeparator()
        display_settings_action = view_menu.addAction('Contrast')
        display_settings_action.setShortcut('Ctrl+C')
        display_settings_action.triggered.connect(self.contrast_dialog.show)
        view_menu.addAction(display_settings_action)

        """ Analyze """
        analyze_menu = menu_bar.addMenu('Analyze')
        parameters_action = analyze_menu.addAction('Parameters')
        parameters_action.setShortcut('Ctrl+P')
        parameters_action.triggered.connect(self.parameters_dialog.show)
        analyze_menu.addAction(parameters_action)
        analyze_menu.addSeparator()
        identify_action = analyze_menu.addAction('Identify')
        identify_action.setShortcut('Ctrl+I')
        identify_action.triggered.connect(self.identify)
        analyze_menu.addAction(identify_action)
        fit_action = analyze_menu.addAction('Fit')
        fit_action.setShortcut('Ctrl+F')
        fit_action.triggered.connect(self.fit)
        analyze_menu.addAction(fit_action)
        localize_action = analyze_menu.addAction('Localize (Identify && Fit)')
        localize_action.setShortcut('Ctrl+L')
        localize_action.triggered.connect(self.localize)
        analyze_menu.addAction(localize_action)

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open image sequence', filter='*.raw')
        if path:
            self.open(path)

    def open(self, path):
        try:
            self.movie, self.info = io.load_raw(path, memory_map=True)
            self.movie_path = path
            self.identifications = None
            self.locs = None
            self.ready_for_fit = False
            self.set_frame(0)
            self.fit_in_view()
        except FileNotFoundError:
            pass  # TODO send a message

    def previous_frame(self):
        if self.movie is not None:
            if self.current_frame_number > 0:
                self.set_frame(self.current_frame_number - 1)

    def next_frame(self):
        if self.movie is not None:
            if self.current_frame_number + 1 < self.info['Frames']:
                self.set_frame(self.current_frame_number + 1)

    def first_frame(self):
        if self.movie is not None:
            self.set_frame(0)

    def last_frame(self):
        if self.movie is not None:
            self.set_frame(self.info['Frames'] - 1)

    def to_frame(self):
        if self.movie is not None:
            frames = self.info['Frames']
            number, ok = QtGui.QInputDialog.getInt(self, 'Go to frame', 'Frame number:', self.current_frame_number+1, 1, frames)
            if ok:
                self.set_frame(number - 1)

    def set_frame(self, number):
        self.current_frame_number = number
        if self.contrast_dialog.auto_checkbox.isChecked():
            black = self.movie[number].min()
            white = self.movie[number].max()
            self.contrast_dialog.change_contrast_silently(black, white)
        self.draw_frame()
        self.status_bar_frame_indicator.setText('{:,}/{:,}'.format(number + 1, self.info['Frames']))

    def draw_frame(self):
        if self.movie is not None:
            frame = self.movie[self.current_frame_number]
            frame = frame.astype('float32')
            if self.contrast_dialog.auto_checkbox.isChecked():
                frame -= frame.min()
                frame /= frame.max()
            else:
                frame -= self.contrast_dialog.black_spinbox.value()
                frame /= self.contrast_dialog.white_spinbox.value()
            frame *= 255.0
            frame = np.maximum(frame, 0)
            frame = np.minimum(frame, 255)
            frame = frame.astype('uint8')
            height, width = frame.shape
            image = QtGui.QImage(frame.data, width, height, width, QtGui.QImage.Format_Indexed8)
            image.setColorTable(CMAP_GRAYSCALE)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.scene = Scene(self)
            self.scene.addPixmap(pixmap)
            self.view.setScene(self.scene)
            if self.ready_for_fit:
                identifications_frame = self.identifications[self.identifications.frame == self.current_frame_number]
                roi = self.last_identification_parameters['ROI']
                self.draw_identifications(identifications_frame, roi, QtGui.QColor('yellow'))
            else:
                if self.parameters_dialog.preview_checkbox.isChecked():
                    identifications_frame = localize.identify_frame(self.movie[self.current_frame_number],
                                                                    self.parameters,
                                                                    self.current_frame_number)
                    roi = self.parameters['ROI']
                    self.status_bar.showMessage('Found {:,} spots in current frame.'.format(len(identifications_frame)))
                    self.draw_identifications(identifications_frame, roi, QtGui.QColor('red'))
                else:
                    self.status_bar.showMessage('')
            if self.locs is not None:
                locs_frame = self.locs[self.locs.frame == self.current_frame_number]
                for loc in locs_frame:
                    self.scene.addItem(FitMarker(loc.x+0.5, loc.y+0.5, 1))

    def draw_identifications(self, identifications, roi, color):
        roi_half = int(roi / 2)
        for identification in identifications:
            x = identification.x
            y = identification.y
            self.scene.addRect(x - roi_half, y - roi_half, roi, roi, color)

    def open_parameters(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open parameters', filter='*.yaml')
        if path:
            self.load_parameters(path)

    def load_parameters(self, path):
        with open(path, 'r') as file:
            parameters = yaml.load(file)
            self.parameters_dialog.roi_spinbox.setValue(parameters['ROI'])
            self.parameters_dialog.mlgm_spinbox.setValue(parameters['Minimum LGM'])
            self.status_bar.showMessage('Parameter file {} loaded.'.format(path))

    def save_parameters(self):
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save parameters', filter='*.yaml')
        if path:
            with open(path, 'w') as file:
                yaml.dump(self.parameters, file, default_flow_style=False)

    @property
    def parameters(self):
        return {'ROI': self.parameters_dialog.roi_spinbox.value(),
                'Minimum LGM': self.parameters_dialog.mlgm_slider.value()}

    def on_parameters_changed(self):
        self.locs = None
        self.ready_for_fit = False
        self.draw_frame()

    def identify(self, fit_afterwards=False):
        if self.movie is not None:
            self.status_bar.showMessage('Preparing identification...')
            self.identificaton_worker = IdentificationWorker(self, fit_afterwards)
            self.identificaton_worker.progressMade.connect(self.on_identify_progress)
            self.identificaton_worker.finished.connect(self.on_identify_finished)
            self.identificaton_worker.start()

    def on_identify_progress(self, frame_number, parameters):
        n_frames = self.info['Frames']
        roi = parameters['ROI']
        mmlg = parameters['Minimum LGM']
        message = 'Identifying in frame {:,}/{:,} (ROI: {:,}; Minimum LGM: {:,})...'.format(frame_number, n_frames, roi, mmlg)
        self.status_bar.showMessage(message)

    def on_identify_finished(self, parameters, identifications, fit_afterwards):
        if len(identifications):
            self.locs = None
            self.last_identification_parameters = parameters.copy()
            n_identifications = len(identifications)
            roi = parameters['ROI']
            mmlg = parameters['Minimum LGM']
            message = 'Identified {:,} spots (ROI: {:,}; Minimum LGM: {:,}). Ready for fit.'.format(n_identifications, roi, mmlg)
            self.status_bar.showMessage(message)
            self.identifications = identifications
            self.ready_for_fit = True
            self.draw_frame()
            if fit_afterwards:
                self.fit()

    def fit(self):
        if self.movie is not None and self.ready_for_fit:
            self.status_bar.showMessage('Preparing fit...')
            self.fit_worker = FitWorker(self)
            self.fit_worker.progressMade.connect(self.on_fit_progress)
            self.fit_worker.finished.connect(self.on_fit_finished)
            self.fit_worker.start()

    def on_fit_progress(self, current, n_spots):
        message = 'Fitting spot {:,}/{:,}...'.format(current, n_spots)
        self.status_bar.showMessage(message)

    def on_fit_finished(self, locs):
        self.status_bar.showMessage('Fitted {:,} spots.'.format(len(locs)))
        self.locs = locs
        self.draw_frame()

    def fit_in_view(self):
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(10 / 7, 10 / 7)

    def zoom_out(self):
        self.view.scale(7 / 10, 7 / 10)

    def save_locs(self):
        base, ext = os.path.splitext(self.movie_path)
        locs_path = base + '_locs.hdf5'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save localizations', locs_path, filter='*.hdf5')
        if path:
            localize_info = self.last_identification_parameters.copy()
            localize_info.update({'Generated by': 'Picasso Localize'})
            info = [self.info, localize_info]
            io.save_locs(path, self.locs, info)

    def localize(self):
        self.identify(fit_afterwards=True)


class IdentificationWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, dict)
    finished = QtCore.pyqtSignal(dict, np.recarray, bool)

    def __init__(self, window, fit_afterwards):
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.parameters = window.parameters
        self.fit_afterwards = fit_afterwards

    def run(self):
        result, counter, pool = localize.identify_async(self.movie, self.parameters)
        while not result.ready():
            self.progressMade.emit(int(counter.value), self.parameters)
            time.sleep(0.1)
        identifications = result.get()
        pool.terminate()
        identifications = np.hstack(identifications)
        identifications = identifications.view(np.recarray)
        self.finished.emit(self.parameters, identifications, self.fit_afterwards)


class FitWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(np.recarray)

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.info = window.info
        self.identifications = window.identifications
        self.roi = window.parameters['ROI']

    def run(self):
        thread, fit_info = localize.fit_async(self.movie, self.info, self.identifications, self.roi)
        while thread.is_alive():
            self.progressMade.emit(fit_info.current, fit_info.n_spots)
            time.sleep(0.1)
        thread.join()   # just in case...
        locs = localize.locs_from_fit_info(fit_info, self.identifications, self.roi)
        self.finished.emit(locs)


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
