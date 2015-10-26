#!/usr/bin/env python
"""
    gui/localize
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.localize

    :author: Joerg Schnitzbauer, 2015
"""

import sys
import os.path
import time
from PyQt4 import QtCore, QtGui
from picasso import io, localize


CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
DEFAULT_PARAMETERS = {'roi': 5, 'threshold': 300}
FRAME_STARTED_MESSAGE = 'Identifying: Frame {}/{} (ROI: {}, Mininum AGS: {})'
IDENTIFY_FINISHED_MESSAGE = 'Identifications: {} (ROI: {}, Minimum AGS: {}, Secs/Frame: {:.3f})'


class View(QtGui.QGraphicsView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
        self.setAcceptDrops(True)

    def wheelEvent(self, event):
        scale = 1.01 ** (-event.delta())
        self.scale(scale, scale)


class Scene(QtGui.QGraphicsScene):

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window

    def dragEnterEvent(self, event):
        event.accept()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().urls():
            url = event.mimeData().urls()[0]
            path = url.toLocalFile()
            try:
                self.window.open(path)
            except OSError:
                pass


class OddSpinBox(QtGui.QSpinBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self, value):
        if value % 2 == 0:
            self.setValue(value + 1)


class ParametersDialog(QtGui.QDialog):

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Parameters')
        vbox = QtGui.QVBoxLayout(self)
        grid = QtGui.QGridLayout()
        vbox.addLayout(grid)
        grid.addWidget(QtGui.QLabel('ROI side length:'), 0, 0)
        self.roi_spinbox = OddSpinBox()
        self.roi_spinbox.setValue(parameters['roi'])
        self.roi_spinbox.setSingleStep(2)
        grid.addWidget(self.roi_spinbox, 0, 1)
        grid.addWidget(QtGui.QLabel('Minimum AGS:'), 1, 0)
        self.threshold_spinbox = QtGui.QSpinBox()
        self.threshold_spinbox.setMaximum(999999999)
        self.threshold_spinbox.setValue(parameters['threshold'])
        grid.addWidget(self.threshold_spinbox, 1, 1)
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        vbox.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def parameters(self):
        return {'roi': self.roi_spinbox.value(), 'threshold': self.threshold_spinbox.value()}


class Window(QtGui.QMainWindow):

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
        # Init variables
        self.movie = None
        self.parameters = DEFAULT_PARAMETERS
        self.identifications = []
        self.last_identification_parameters = None
        self.identification_markers = []
        self.worker = None
        self.worker_interrupt_flag = False

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

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
        first_frame_action = view_menu.addAction('First frame')
        first_frame_action.setShortcut('Ctrl+Left')
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction('Last frame')
        last_frame_action.setShortcut('Ctrl+Right')
        last_frame_action.triggered.connect(self.last_frame)
        view_menu.addAction(last_frame_action)
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
        self.movie, self.info = io.load_raw(path)
        self.set_frame(0)
        self.fit_in_view()

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

    def set_frame(self, number):
        if self.identifications:
            self.remove_identification_markers()
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
        if self.identifications:
            self.draw_identification_markers()

    def open_parameters_dialog(self):
        dialog = ParametersDialog(self.parameters)
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            self.parameters = dialog.parameters()

    def identify(self):
        if self.movie is not None:
            self.interrupt_worker_if_running()
            self.worker = IdentificationWorker(self, self.movie, self.parameters)
            self.worker.progressMade.connect(self.on_identify_next_frame_started)
            self.worker.finished.connect(self.on_identify_finished)
            self.worker.interrupted.connect(self.on_identify_interrupted)
            self.worker.start()

    def on_identify_next_frame_started(self, frame_number):
        n_frames = self.info['frames']
        roi = self.parameters['roi']
        threshold = self.parameters['threshold']
        message = FRAME_STARTED_MESSAGE.format(frame_number, n_frames, roi, threshold)
        self.status_bar.showMessage(message)

    def on_identify_finished(self, identifications):
        required_time = self.worker.elapsed_time
        time_per_frame = required_time / self.info['frames']
        n_identifications = 0
        for identifications_frame in identifications:
            n_identifications += len(identifications_frame)
        roi = self.parameters['roi']
        threshold = self.parameters['threshold']
        message = IDENTIFY_FINISHED_MESSAGE.format(n_identifications, roi, threshold, time_per_frame)
        self.status_bar.showMessage(message)
        self.identifications = identifications
        self.last_identification_parameters = self.parameters.copy()
        self.remove_identification_markers()
        self.draw_identification_markers()

    def fit(self):
        pass

    def interrupt_worker_if_running(self):
        if self.worker and self.worker.isRunning():
            self.worker_interrupt_flag = True
            self.worker.wait()
            self.worker_interupt_flag = False

    def on_identify_interrupted(self):
        self.worker_interrupt_flag = False
        self.status_bar.showMessage('Interrupted')

    def remove_identification_markers(self):
        for rect in self.identification_markers:
            self.scene.removeItem(rect)
        self.identification_markers = []

    def draw_identification_markers(self):
        identifications_frame = self.identifications[self.current_frame_number]
        roi = self.last_identification_parameters['roi']
        roi_half = int(roi / 2)
        for y, x in identifications_frame:
            rect = self.scene.addRect(x - roi_half, y - roi_half, roi, roi, QtGui.QPen(QtGui.QColor('red')))
            self.identification_markers.append(rect)

    def fit_in_view(self):
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def zoom_in(self):
        self.view.scale(10 / 7, 10 / 7)

    def zoom_out(self):
        self.view.scale(7 / 10, 7 / 10)


class IdentificationWorker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(list)
    interrupted = QtCore.pyqtSignal()

    def __init__(self, window, movie, parameters):
        super().__init__()
        self.window = window
        self.movie = movie
        self.parameters = parameters

    def run(self):
        start = time.time()
        identifications = []
        for i, frame in enumerate(self.movie):
            self.progressMade.emit(i)
            identifications_frame = localize.identify_frame(frame, self.parameters)
            identifications.append(identifications_frame)
            if self.window.worker_interrupt_flag:
                self.interrupted.emit()
                return
        self.elapsed_time = time.time() - start
        self.finished.emit(identifications)


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
