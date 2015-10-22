#!/usr/bin/env python
"""
    gui/localize
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.localize

    :author: Joerg Schnitzbauer, 2015
"""

import sys
from PyQt4 import QtCore, QtGui
from picasso import io, localize


GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
IDENTIFICATION_PARAMETERS_DEFAULTS = {'roi': 5, 'threshold': 100}


class View(QtGui.QGraphicsView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        scale = 1.01 ** (-event.delta())
        self.scale(scale, scale)


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
        vbox = QtGui.QVBoxLayout(self)
        grid = QtGui.QGridLayout()
        vbox.addLayout(grid)
        grid.addWidget(QtGui.QLabel('ROI'), 0, 0)
        self.roi_spinbox = OddSpinBox()
        self.roi_spinbox.setValue(parameters['roi'])
        self.roi_spinbox.setSingleStep(2)
        grid.addWidget(self.roi_spinbox, 0, 1)
        grid.addWidget(QtGui.QLabel('Threshold'), 1, 0)
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
        self.setWindowTitle('Picasso: Localize')
        self.resize(768, 768)
        self.init_menu_bar()
        self.view = View()
        self.setCentralWidget(self.view)
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)
        self.identification_parameters = IDENTIFICATION_PARAMETERS_DEFAULTS
        self.identifications = []
        self.identification_rectangles = []

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.on_file_open)
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

        """ Analyze """
        analyze_menu = menu_bar.addMenu('Analyze')
        identification_parameters_action = analyze_menu.addAction('Identification parameters')
        identification_parameters_action.setShortcut('Ctrl+Shift+I')
        identification_parameters_action.triggered.connect(self.on_identification_parameters)
        analyze_menu.addAction(identification_parameters_action)
        identify_action = analyze_menu.addAction('Identify')
        identify_action.setShortcut('Ctrl+I')
        identify_action.triggered.connect(self.on_identify)
        analyze_menu.addAction(identify_action)

    def on_file_open(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open image sequence', filter='*.raw')
        if path:
            self.movie, self.info = io.load_raw(path)
            self.set_frame(0)
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def on_next_frame(self):
        if self.current_frame_number < self.info['frames']:
            self.set_frame(self.current_frame_number + 1)

    def on_previous_frame(self):
        if self.current_frame_number > 0:
            self.set_frame(self.current_frame_number - 1)

    def set_frame(self, number):
        self.current_frame_number = number
        frame = self.movie[number]
        frame = frame.astype('float32')
        frame -= frame.min()
        frame /= frame.ptp()
        frame *= 255.0
        frame = frame.astype('uint8')
        width, height = frame.shape
        qimage = QtGui.QImage(frame.data, width, height, QtGui.QImage.Format_Indexed8)
        qimage.setColorTable(GRAYSCALE)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.scene = QtGui.QGraphicsScene()
        self.scene.addPixmap(qpixmap)
        self.view.setScene(self.scene)
        if self.identifications:
            self.draw_identifications()

    def on_identification_parameters(self):
        dialog = IdentificationParametersDialog(self.identification_parameters, self)
        result = dialog.exec_()
        if result == QtGui.QDialog.Accepted:
            self.identification_parameters = dialog.parameters()

    def on_identify(self):
        roi_size = self.identification_parameters['roi']
        threshold = self.identification_parameters['threshold']
        self.identifications = []
        for frame in self.movie:
            frame_identificiations = localize.identify(frame, roi_size, threshold)
            self.identifications.append(frame_identificiations)
        self.draw_identifications()

    def draw_identifications(self):
        # Remove old identification
        for rect in self.identification_rectangles:
            self.scene.removeItem(rect)
        frame_identifications = self.identifications[self.current_frame_number]
        self.identification_rectangles = []
        roi_size = self.identification_parameters['roi']
        roi_size_half = int(roi_size/2)
        for y, x in frame_identifications:
            rect = self.scene.addRect(x - roi_size_half, y - roi_size_half, roi_size, roi_size, QtGui.QPen(QtGui.QColor('red')))
            self.identification_rectangles.append(rect)

    def on_zoom_in(self):
        self.view.scale(10 / 7, 10 / 7)

    def on_zoom_out(self):
        self.view.scale(7 / 10, 7 / 10)


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
