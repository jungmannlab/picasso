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


CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
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

    def dragEnterEvent(self, event):
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        """ Loads raw movies or yaml parameters when dropped into the scene """
        if event.mimeData().urls():
            url = event.mimeData().urls()[0]
            path = url.toLocalFile()
            base, extension = os.path.splitext(path)
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
        line1 = QtGui.QGraphicsLineItem(x-L, y, x+L, y)
        line1.setPen(QtGui.QPen(QtGui.QColor('red')))
        self.addToGroup(line1)
        line2 = QtGui.QGraphicsLineItem(x, y-L, x, y+L)
        line2.setPen(QtGui.QPen(QtGui.QColor('red')))
        self.addToGroup(line2)


class OddSpinBox(QtGui.QSpinBox):
    """ A spinbox that allows only odd numbers """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self, value):
        if value % 2 == 0:
            self.setValue(value + 1)


class ParametersDialog(QtGui.QDialog):
    """ The dialog showing analysis parameters """

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Parameters')
        vbox = QtGui.QVBoxLayout(self)
        grid = QtGui.QGridLayout()
        vbox.addLayout(grid)
        grid.addWidget(QtGui.QLabel('ROI side length:'), 0, 0)
        self.roi_spinbox = OddSpinBox()
        self.roi_spinbox.setValue(parameters['ROI'])
        self.roi_spinbox.setSingleStep(2)
        grid.addWidget(self.roi_spinbox, 0, 1)
        grid.addWidget(QtGui.QLabel('Minimum LGM:'), 1, 0)
        self.mmlg_spinbox = QtGui.QSpinBox()
        self.mmlg_spinbox.setMaximum(999999999)
        self.mmlg_spinbox.setValue(parameters['Minimum LGM'])
        grid.addWidget(self.mmlg_spinbox, 1, 1)
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        vbox.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def parameters(self):
        return {'ROI': self.roi_spinbox.value(), 'Minimum LGM': self.mmlg_spinbox.value()}


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
        self.init_menu_bar()
        self.view = View()
        self.setCentralWidget(self.view)
        self.scene = Scene(self)
        self.view.setScene(self.scene)
        self.status_bar = self.statusBar()
        self.status_bar_frame_indicator = QtGui.QLabel()
        self.status_bar.addPermanentWidget(self.status_bar_frame_indicator)

        #: Holds the current movie as a numpy memmap in the format (frame, x, y)
        self.movie = None

        #: A dictionary of the current analysis parameters
        self.parameters = DEFAULT_PARAMETERS

        #: A dictionary of analysis parameters used for the last operation
        self.last_identification_parameters = None

        #: A numpy.recarray of identifcations with fields frame, x and y
        self.identifications = None

        self.worker = None
        self.worker_interrupt_flag = False
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
        first_frame_action.setShortcut('Ctrl+Left')
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction('Last frame')
        last_frame_action.setShortcut('Ctrl+Right')
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
        parameters_action.triggered.connect(self.open_parameters_dialog)
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
        analyze_menu.addSeparator()
        interrupt_action = analyze_menu.addAction('Interrupt')
        interrupt_action.setShortcut('Ctrl+X')
        interrupt_action.triggered.connect(self.interrupt_worker_if_running)
        analyze_menu.addAction(interrupt_action)

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open image sequence', filter='*.raw')
        if path:
            self.open(path)

    def open(self, path):
        try:
            self.movie, self.info = io.load_raw(path, memory_map=True)
            self.movie_path = path
            message = 'Loaded {} frames. Ready to go.'.format(self.info['frames'])
            self.status_bar.showMessage(message)
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
            if self.current_frame_number + 1 < self.info['frames']:
                self.set_frame(self.current_frame_number + 1)

    def first_frame(self):
        if self.movie is not None:
            self.set_frame(0)

    def last_frame(self):
        if self.movie is not None:
            self.set_frame(self.info['frames'] - 1)

    def to_frame(self):
        if self.movie is not None:
            frames = self.info['frames']
            frames_half = int(frames / 2)
            number, ok = QtGui.QInputDialog.getInt(self, 'Go to frame', 'Frame number:', 1, frames_half, frames)
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
        self.status_bar_frame_indicator.setText('{}/{}'.format(number + 1, self.info['frames']))
        if self.identifications is not None:
            identifications_frame = self.identifications[self.identifications.frame == number]
            roi = self.last_identification_parameters['ROI']
            roi_half = int(roi / 2)
            for identification in identifications_frame:
                x = identification.x
                y = identification.y
                self.scene.addRect(y - roi_half, x - roi_half, roi, roi, QtGui.QPen(QtGui.QColor('red')))
        if self.locs is not None:
            locs_frame = self.locs[self.locs.frame == number]
            L = self.last_identification_parameters['ROI']
            for loc in locs_frame:
                self.scene.addItem(FitMarker(loc.y, loc.x, L))

    def open_parameters(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open parameters', filter='*.yaml')
        if path:
            self.load_parameters(path)

    def load_parameters(self, path):
        with open(path, 'r') as file:
            self.parameters = yaml.load(file)

    def save_parameters(self):
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save parameters', filter='*.yaml')
        if path:
            with open(path, 'w') as file:
                yaml.dump(self.parameters, file, default_flow_style=False)

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self.parameters)
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            self.parameters = dialog.parameters()

    def identify(self):
        if self.movie is not None:
            self.interrupt_worker_if_running()
            self.status_bar.showMessage('Preparing identification...')
            self.worker = IdentificationWorker(self)
            self.worker.progressMade.connect(self.on_identify_progress)
            self.worker.finished.connect(self.on_identify_finished)
            self.worker.interrupted.connect(self.on_identify_interrupted)
            self.worker.start()

    def on_identify_progress(self, frame_number):
        n_frames = self.info['frames']
        roi = self.parameters['ROI']
        mmlg = self.parameters['Minimum LGM']
        message = 'Identifying in frame {}/{} (ROI: {}, Mininum AGS: {})...'.format(frame_number, n_frames, roi, mmlg)
        self.status_bar.showMessage(message)

    def on_identify_finished(self, identifications):
        self.locs = None
        n_identifications = len(identifications)
        roi = self.parameters['ROI']
        mmlg = self.parameters['Minimum LGM']
        message = 'Identified {} spots (ROI: {}, Minimum AGS: {}).'.format(n_identifications, roi, mmlg)
        self.status_bar.showMessage(message)
        self.identifications = identifications
        self.last_identification_parameters = self.parameters.copy()
        self.set_frame(self.current_frame_number)

    def fit(self):
        if self.movie is not None and self.identifications is not None:
            self.interrupt_worker_if_running()
            self.status_bar.showMessage('Preparing fit...')
            self.worker = FitWorker(self)
            self.worker.progressMade.connect(self.on_fit_progress)
            self.worker.finished.connect(self.on_fit_finished)
            self.worker.interrupted.connect(self.on_fit_interrupted)
            self.worker.start()

    def on_fit_progress(self, current, n_spots):
        message = 'Fitting spot {}/{}...'.format(current, n_spots)
        self.status_bar.showMessage(message)

    def on_fit_finished(self, locs):
        self.status_bar.showMessage('Fitted {} spots.'.format(len(locs)))
        self.locs = locs
        self.set_frame(self.current_frame_number)

    def on_fit_interrupted(self):
        self.worker_interrupt_flag = False
        self.status_bar.showMessage('Fitting interrupted.')

    def interrupt_worker_if_running(self):
        if self.worker and self.worker.isRunning():
            self.worker_interrupt_flag = True
            self.worker.wait()
            self.worker_interrupt_flag = False

    def on_identify_interrupted(self):
        self.worker_interrupt_flag = False
        self.status_bar.showMessage('Identification interrupted.')

    def fit_in_view(self):
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(10 / 7, 10 / 7)

    def zoom_out(self):
        self.view.scale(7 / 10, 7 / 10)

    def save_locs(self):
        base, ext = os.path.splitext(self.movie_path)
        locs_path = base + '_locs.txt'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save localizations', locs_path, filter='*.txt')
        if path:
            io.save_locs(path, self.locs, self.info, self.last_identification_parameters)

    def closeEvent(self, event):
        self.interrupt_worker_if_running()


class IdentificationWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(np.recarray)
    interrupted = QtCore.pyqtSignal()

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.movie = window.movie
        self.parameters = window.parameters

    def run(self):
        result, counter, pool = localize.identify_async(self.movie, self.parameters)
        while not result.ready():
            self.progressMade.emit(int(counter.value))
            time.sleep(0.1)
            if self.window.worker_interrupt_flag:
                self.interrupted.emit()
                pool.terminate()
                return
        identifications = result.get()
        identifications = np.hstack(identifications)
        identifications = identifications.view(np.recarray)
        self.finished.emit(identifications)


class FitWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(np.recarray)
    interrupted = QtCore.pyqtSignal()

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
            if self.window.worker_interrupt_flag:
                # Find a way how to stop DLL function
                self.interrupted.emit()
                return
            self.progressMade.emit(fit_info.current, fit_info.n_spots)
            time.sleep(0.1)
        thread.join()   # just in case...
        locs = localize.locs_from_fit_info(fit_info, self.identifications, self.roi)
        self.finished.emit(locs)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
