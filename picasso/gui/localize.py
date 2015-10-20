#!/usr/bin/env python
"""
    gui/localize
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.localize

    :author: Joerg Schnitzbauer
"""

import sys
from PyQt4 import QtGui
from picasso import io, localize


GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Picasso Localize')
        self.init_menu_bar()
        self.view = QtGui.QGraphicsView()
        self.setCentralWidget(self.view)
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)

    def init_menu_bar(self):
        menu_bar = self.menuBar()

        ## File
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.on_file_open)
        file_menu.addAction(open_action)

        ## View
        view_menu = menu_bar.addMenu('View')

        # Next and previous frames
        next_frame_action = view_menu.addAction('Next frame')
        next_frame_action.setShortcut('Right')
        next_frame_action.triggered.connect(self.on_next_frame)
        view_menu.addAction(next_frame_action)
        previous_frame_action = view_menu.addAction('Previous frame')
        previous_frame_action.setShortcut('Left')
        previous_frame_action.triggered.connect(self.on_previous_frame)
        view_menu.addAction(previous_frame_action)

        # Zooming
        zoom_out_action = view_menu.addAction('Zoom out')
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.on_zoom_out)
        view_menu.addAction(zoom_out_action)
        zoom_in_action = view_menu.addAction('Zoom in')
        zoom_in_action.setShortcuts(['Ctrl++', 'Ctrl+='])
        zoom_in_action.triggered.connect(self.on_zoom_in)
        view_menu.addAction(zoom_in_action)

    def on_file_open(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open image sequence', filter='*.raw')
        if path:
            self.movie, self.info = io.load_raw(path)
            self.set_frame(0)

    def on_next_frame(self):
        if self.frame_number < self.info['frames']:
            self.set_frame(self.frame_number + 1)

    def on_previous_frame(self):
        if self.frame_number > 0:
            self.set_frame(self.frame_number - 1)

    def on_zoom_in(self):
        self.view.scale(4, 4)

    def on_zoom_out(self):
        self.view.scale(0.5, 0.5)

    def set_frame(self, number):
        self.frame_number = number
        frame = self.movie[number]
        frame = frame.astype('float32')
        frame -= frame.min()
        frame /= frame.ptp()
        frame *= 255.0
        frame = frame.astype('uint8')
        image = QtGui.QImage(frame.data, self.info['width'], self.info['height'], QtGui.QImage.Format_Indexed8)
        image.setColorTable(GRAYSCALE)
        #pixmap = QtGui.QPixmap.fromImage(image)
        pixmap = QtGui.QPixmap(2, 2)
        pixmap.fill(QtGui.QColor('white'))
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QColor('black'))
        painter.drawPoint(0, 0)
        painter.drawPoint(1, 1)
        painter.end()
        self.scene = QtGui.QGraphicsScene()
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
