#!/usr/bin/env python
"""
    gui/localize
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.localize

    :author: Joerg Schnitzbauer, 2015
"""

import sys
import os.path
import yaml
from PyQt4 import QtCore, QtGui
from picasso import io, localize
import time
import numpy as np
import traceback


CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
DEFAULT_MLGM_MAXIMUM = 10000
DEFAULT_PARAMETERS = {'ROI': 5, 'Minimum LGM': 300}


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
        # self.resize(300, 0)
        self.window = window
        self.setWindowTitle('Parameters')
        self.setModal(False)
        grid = QtGui.QGridLayout(self)
        grid.addWidget(QtGui.QLabel('ROI side length:'), 0, 0)
        grid.setColumnStretch(1, 1)
        self.roi_spinbox = OddSpinBox()
        self.roi_spinbox.setValue(DEFAULT_PARAMETERS['ROI'])
        grid.addWidget(self.roi_spinbox, 0, 2)
        grid.addWidget(QtGui.QLabel('Minimum LGM:'), 1, 0)
        mlgm_min_spinbox = QtGui.QSpinBox()
        mlgm_min_spinbox.setRange(1, DEFAULT_MLGM_MAXIMUM)
        mlgm_min_spinbox.setKeyboardTracking(False)
        mlgm_min_spinbox.setValue(1)
        mlgm_min_spinbox.valueChanged.connect(self.on_mlgm_min_changed)
        hbox = QtGui.QHBoxLayout()
        grid.addLayout(hbox, 2, 0, 3, 3)
        hbox.addWidget(mlgm_min_spinbox)
        self.mlgm_slider = QtGui.QSlider()
        self.mlgm_slider.setOrientation(QtCore.Qt.Horizontal)
        self.mlgm_slider.setRange(1, DEFAULT_MLGM_MAXIMUM)
        self.mlgm_slider.setValue(DEFAULT_PARAMETERS['Minimum LGM'])
        self.mlgm_slider.setSingleStep(1)
        self.mlgm_slider.setPageStep(20)
        self.mlgm_slider.valueChanged.connect(self.on_mlgm_changed)
        hbox.addWidget(self.mlgm_slider)
        mlgm_max_spinbox = QtGui.QSpinBox()
        mlgm_max_spinbox.setKeyboardTracking(False)
        mlgm_max_spinbox.setRange(2, 999999)
        mlgm_max_spinbox.setValue(DEFAULT_MLGM_MAXIMUM)
        mlgm_max_spinbox.valueChanged.connect(self.on_mlgm_max_changed)
        hbox.addWidget(mlgm_max_spinbox)
        self.mlgm_label = QtGui.QLabel(str(self.mlgm_slider.value()))
        grid.addWidget(self.mlgm_label, 1, 2)

    def on_mlgm_changed(self, value):
        self.mlgm_label.setText(str(self.mlgm_slider.value()))
        self.window.on_parameters_changed()

    def on_mlgm_min_changed(self, value):
        self.mlgm_slider.setMinimum(value)

    def on_mlgm_max_changed(self, value):
        self.mlgm_slider.setMaximum(value)


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
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction('Save localizations')
        save_action.setShortcut(QtGui.QKeySequence.Save)
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
        first_frame_action.setShortcut(QtGui.QKeySequence.MoveToStartOfLine)
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction('Last frame')
        last_frame_action.setShortcut(QtGui.QKeySequence.MoveToEndOfLine)
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
        view_menu.addSeparator()
        fit_in_view_action = view_menu.addAction('Fit image to window')
        fit_in_view_action.setShortcut('Ctrl+W')
        fit_in_view_action.triggered.connect(self.fit_in_view)
        view_menu.addAction(fit_in_view_action)

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

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open image sequence', filter='*.raw')
        if path:
            self.open(path)

    def open(self, path):
        try:
            self.movie, self.info = io.load_raw(path, memory_map=True)
            self.movie_path = path
            message = 'Loaded {} frames. Ready to go.'.format(self.info['Frames'])
            self.status_bar.showMessage(message)
            self.identifications = None
            self.locs = None
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
        frame = self.movie[number]
        frame = frame.astype('float32')
        frame -= frame.min()
        frame /= frame.ptp()
        frame *= 255.0
        frame = frame.astype('uint8')
        width, height = frame.shape
        image = QtGui.QImage(frame.data, width, height, QtGui.QImage.Format_Indexed8)
        image.setColorTable(CMAP_GRAYSCALE)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.scene = Scene(self)
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.status_bar_frame_indicator.setText('{}/{}'.format(number + 1, self.info['Frames']))
        if self.ready_for_fit:
            identifications_frame = self.identifications[self.identifications.frame == number]
            roi = self.last_identification_parameters['ROI']
            identification_marker_color = QtGui.QColor('yellow')
        else:
            identifications_frame = localize.identify_frame(self.movie[self.current_frame_number],
                                                            self.parameters,
                                                            self.current_frame_number)
            roi = self.parameters['ROI']
            identification_marker_color = QtGui.QColor('red')
            self.status_bar.showMessage('Found {} spots in current frame.'.format(len(identifications_frame)))
        roi_half = int(roi / 2)
        for identification in identifications_frame:
            x = identification.x
            y = identification.y
            self.scene.addRect(x - roi_half, y - roi_half, roi, roi, QtGui.QPen(identification_marker_color))
        if self.locs is not None:
            locs_frame = self.locs[self.locs.frame == number]
            for loc in locs_frame:
                self.scene.addItem(FitMarker(loc.x+0.5, loc.y+0.5, 1))

    def open_parameters(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open parameters', filter='*.yaml')
        if path:
            self.load_parameters(path)

    def load_parameters(self, path):
        with open(path, 'r') as file:
            self.parameters = yaml.load(file)
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
        if self.movie is not None:
            self.locs = None
            self.ready_for_fit = False
            self.set_frame(self.current_frame_number)

    def identify(self):
        if self.movie is not None:
            self.status_bar.showMessage('Preparing identification...')
            self.worker = IdentificationWorker(self)
            self.worker.progressMade.connect(self.on_identify_progress)
            self.worker.finished.connect(self.on_identify_finished)
            self.worker.start()

    def on_identify_progress(self, frame_number, parameters):
        n_frames = self.info['Frames']
        roi = parameters['ROI']
        mmlg = parameters['Minimum LGM']
        message = 'Identifying in frame {}/{} (ROI: {}, Mininum LGM: {})...'.format(frame_number, n_frames, roi, mmlg)
        self.status_bar.showMessage(message)

    def on_identify_finished(self, parameters, identifications):
        if len(identifications):
            self.locs = None
            self.last_identification_parameters = parameters.copy()
            n_identifications = len(identifications)
            roi = parameters['ROI']
            mmlg = parameters['Minimum LGM']
            message = 'Identified {} spots (ROI: {}, Minimum LGM: {}). Ready for fit.'.format(n_identifications, roi, mmlg)
            self.status_bar.showMessage(message)
            self.identifications = identifications
            self.ready_for_fit = True
            self.set_frame(self.current_frame_number)

    def fit(self):
        if self.movie is not None and self.identifications is not None:
            self.status_bar.showMessage('Preparing fit...')
            self.worker = FitWorker(self)
            self.worker.progressMade.connect(self.on_fit_progress)
            self.worker.finished.connect(self.on_fit_finished)
            self.worker.start()

    def on_fit_progress(self, current, n_spots):
        message = 'Fitting spot {}/{}...'.format(current, n_spots)
        self.status_bar.showMessage(message)

    def on_fit_finished(self, locs):
        self.status_bar.showMessage('Fitted {} spots.'.format(len(locs)))
        self.locs = locs
        self.set_frame(self.current_frame_number)

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
            io.save_locs(path, self.locs, self.info, self.last_identification_parameters)


class IdentificationWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, dict)
    finished = QtCore.pyqtSignal(dict, np.recarray)

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.parameters = window.parameters

    def run(self):
        result, counter, pool = localize.identify_async(self.movie, self.parameters)
        while not result.ready():
            self.progressMade.emit(int(counter.value), self.parameters)
            time.sleep(0.1)
        identifications = result.get()
        pool.terminate()
        identifications = np.hstack(identifications)
        identifications = identifications.view(np.recarray)
        self.finished.emit(self.parameters, identifications)


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
