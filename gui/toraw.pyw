#!/usr/bin/env python
"""
    gui/toraw
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.io.to_raw

    :author: Joerg Schnitzbauer, 2015
"""

import sys
import os.path
import glob
from PyQt4 import QtCore, QtGui
from picasso import io


class Window(QtGui.QWidget):

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: ToRaw')
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'toraw.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(512, 1)
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        file_label = QtGui.QLabel('Drop or browse a file:')
        file_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.vbox.addWidget(file_label)
        hbox = QtGui.QHBoxLayout()
        self.vbox.addLayout(hbox)
        self.path_edit = QtGui.QLineEdit()
        hbox.addWidget(self.path_edit)
        self.browse_button = QtGui.QPushButton('Browse')
        self.browse_button.released.connect(self.browse)
        hbox.addWidget(self.browse_button)
        hbox2 = QtGui.QHBoxLayout()
        self.vbox.addLayout(hbox2)
        self.convert_button = QtGui.QPushButton('Convert')
        self.convert_button.clicked.connect(self.convert)
        hbox2.addStretch(1)
        hbox2.addWidget(self.convert_button)
        hbox2.addStretch(1)
        self.progress_bar = QtGui.QProgressBar()
        self.progress_bar.setMinimum(0)

    def browse(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open file to convert')
        if path:
            self.path_edit.setText(path)

    def convert(self):
        files = self.path_edit.text()
        paths = glob.glob(files)
        self.progress_bar.setMaximum(len(paths))
        self.vbox.addWidget(self.progress_bar)
        for i, path in enumerate(paths):
            io.to_raw_single(path)
            self.progress_bar.setValue(i + 1)
        self.path_edit.setText('')


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
