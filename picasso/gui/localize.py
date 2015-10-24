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
DEFAULT_IDENTIFICATION_PARAMETERS = {'roi': 5, 'threshold': 300}
FRAME_STARTED_MESSAGE = 'ROI: {}, Mininum AGS: {}, Frame: {}/{}'
IDENTIFY_FINISHED_MESSAGE = 'ROI: {}, Minimum AGS: {}, Secs/Frame: {:.3f}, Identifications: {}'
MOVIE_LOADED_MESSAGE = '{} frames loaded in {:.2f} seconds.'


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


class IdentificationParametersDialog(QtGui.QDialog):

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Identification parameters')
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
        self.identification_parameters = DEFAULT_IDENTIFICATION_PARAMETERS
        self.identifications = []
        self.identification_markers = []

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_with_dialog)
        file_menu.addAction(open_action)

        """ View """
        view_menu = menu_bar.addMenu('View')
        next_frame_action = view_menu.addAction('Next frame')
        next_frame_action.setShortcut('Right')
        next_frame_action.triggered.connect(self.on_next_frame)
        view_menu.addAction(next_frame_action)
        previous_frame_action = view_menu.addAction('Previous frame')
        previous_frame_action.setShortcut('Left')
        previous_frame_action.triggered.connect(self.on_previous_frame)
        view_menu.addAction(previous_frame_action)
        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction('Zoom in')
        zoom_in_action.setShortcuts(['Ctrl++', 'Ctrl+='])
        zoom_in_action.triggered.connect(self.on_zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction('Zoom out')
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.on_zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction('Fit image to window')
        fit_in_view_action.setShortcut('Ctrl+W')
        fit_in_view_action.triggered.connect(self.fit_in_view)
        view_menu.addAction(fit_in_view_action)

        """ Analyze """
        analyze_menu = menu_bar.addMenu('Analyze')
        identify_action = analyze_menu.addAction('Identify')
        identify_action.setShortcut('Ctrl+I')
        identify_action.triggered.connect(self.on_identify)
        analyze_menu.addAction(identify_action)
        analyze_menu.addSeparator()
        interrupt_action = analyze_menu.addAction('Interrupt')
        interrupt_action.setShortcut('Ctrl+X')
        interrupt_action.triggered.connect(self.on_interrupt)
        analyze_menu.addAction(interrupt_action)

    def open_with_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open image sequence', filter='*.raw')
        if path:
            self.open(path)

    def open(self, path):
        self.status_bar.showMessage('Loading movie...')
        start = time.time()
        self.movie, self.info = io.load_raw(path)
        n_frames = self.info['frames']
        elapsed_time = time.time() - start
        message = MOVIE_LOADED_MESSAGE.format(n_frames, elapsed_time)
        self.status_bar.showMessage(message)
        self.set_frame(0)
        self.fit_in_view()

    def on_next_frame(self):
        if self.movie is not None and self.current_frame_number + 1 < self.info['frames']:
            self.set_frame(self.current_frame_number + 1)

    def on_previous_frame(self):
        if self.movie is not None and self.current_frame_number > 0:
            self.set_frame(self.current_frame_number - 1)

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

    def on_identify(self):
        if self.movie is not None:
            self.identify()
        else:
            self.open_with_dialog()
            self.identify()

    def identify(self):
        dialog = IdentificationParametersDialog(self.identification_parameters, self)
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            self.identification_parameters = dialog.parameters()
            thread = IdentifyWorker(self, self.movie, self.identification_parameters)
            thread.frameStarted.connect(self.on_identify_frame_started)
            thread.finished.connect(self.on_identify_finished)
            thread.interrupted.connect(self.on_interrupted_thread_done)
            self.interupt_flag = False
            self.identify_start = time.time()
            thread.start()

    def on_identify_frame_started(self, frame_number):
        n_frames = self.info['frames']
        roi = self.identification_parameters['roi']
        threshold = self.identification_parameters['threshold']
        message = FRAME_STARTED_MESSAGE.format(roi, threshold, frame_number, n_frames)
        self.status_bar.showMessage(message)

    def on_identify_finished(self, identifications):
        required_time = time.time() - self.identify_start
        time_per_frame = required_time / self.info['frames']
        n_identifications = 0
        for identifications_frame in identifications:
            n_identifications += len(identifications_frame)
        roi = self.identification_parameters['roi']
        threshold = self.identification_parameters['threshold']
        message = IDENTIFY_FINISHED_MESSAGE.format(roi, threshold, time_per_frame, n_identifications)
        self.status_bar.showMessage(message)
        self.identifications = identifications
        self.remove_identification_markers()
        self.draw_identification_markers()

    def on_interrupt(self):
        """ Gets called when the user chooses the Interrupt action from the menu """
        self.interupt_flag = True

    def on_interrupted_thread_done(self):
        """ Gets called when and interrupted thread returned without finishing """
        self.status_bar.showMessage('')

    def remove_identification_markers(self):
        for rect in self.identification_markers:
            self.scene.removeItem(rect)
        self.identification_markers = []

    def draw_identification_markers(self):
        identifications_frame = self.identifications[self.current_frame_number]
        roi = self.identification_parameters['roi']
        roi_half = int(roi / 2)
        for y, x in identifications_frame:
            rect = self.scene.addRect(x - roi_half, y - roi_half, roi, roi, QtGui.QPen(QtGui.QColor('red')))
            self.identification_markers.append(rect)

    def fit_in_view(self):
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def on_zoom_in(self):
        self.view.scale(10 / 7, 10 / 7)

    def on_zoom_out(self):
        self.view.scale(7 / 10, 7 / 10)


class IdentifyWorker(QtCore.QThread):

    frameStarted = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(list)
    interrupted = QtCore.pyqtSignal()

    def __init__(self, window, movie, parameters):
        super().__init__()
        self.window = window
        self.movie = movie
        self.parameters = parameters

    def run(self):
        identifications = []
        for i, frame in enumerate(self.movie):
            self.frameStarted.emit(i)
            identifications_frame = localize.identify_frame(frame, self.parameters)
            identifications.append(identifications_frame)
            if self.window.interupt_flag:
                self.interrupted.emit()
                return
        self.finished.emit(identifications)


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
