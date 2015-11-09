#!/usr/bin/env python
"""
    gui/toraw
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.io.to_raw

    :author: Joerg Schnitzbauer, 2015
"""

import sys
import os.path
from PyQt4 import QtCore, QtGui
from picasso import io
import traceback


class Window(QtGui.QWidget):

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: ToRaw')
        self.resize(768, 512)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'toraw.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(QtGui.QLabel('Files:'))
        self.path_edit = QtGui.QTextEdit()
        vbox.addWidget(self.path_edit)
        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        self.browse_button = QtGui.QPushButton('Browse')
        self.browse_button.clicked.connect(self.browse)
        hbox.addWidget(self.browse_button)
        hbox.addStretch(1)
        to_raw_button = QtGui.QPushButton('To raw')
        to_raw_button.clicked.connect(self.to_raw)
        hbox.addWidget(to_raw_button)

    def browse(self):
        paths = QtGui.QFileDialog.getOpenFileNames(self, 'Open files to convert', filter='*.tif; **.tiff')
        if paths:
            self.path_edit.clear()
        for path in paths:
            self.path_edit.append(path)

    def to_raw(self):
        self.setEnabled(False)
        text = self.path_edit.toPlainText()
        self.paths = text.splitlines()
        self.update_html(0)
        self.worker = Worker(self.paths)
        self.worker.progressMade.connect(self.update_html)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def update_html(self, done):
        html = ''
        for i, path in enumerate(self.paths):
            if i < done:
                html += '<font color="green">{}</font><br>'.format(path)
            elif i == done:
                html += '<font color="yellow">{}</font><br>'.format(path)
            else:
                html += path + '<br>'
        self.path_edit.setHtml(html)

    def on_finished(self, done):
        self.update_html(done + 1)
        self.setEnabled(True)


class Worker(QtCore.QThread):

    progressMade = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int)
    interrupted = QtCore.pyqtSignal()

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        for i, path in enumerate(self.paths):
            self.progressMade.emit(i)
            io._to_raw_single(path)
        self.finished.emit(i)


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
