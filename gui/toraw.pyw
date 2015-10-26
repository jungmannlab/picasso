#!/usr/bin/env python
"""
    gui/toraw
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.io.to_raw

    :author: Joerg Schnitzbauer, 2015
"""

import sys
import os.path
from PyQt4 import QtGui
from picasso import io


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
        for path in paths:
            self.path_edit.append(path)

    def update_path_edit(self, paths, done):
        html = ''
        for i, path in enumerate(paths):
            if i < done:
                html += '<font color="green">{}</font>\n'.format(path)
            elif i == done:
                html += '<font color="yellow">{}</font>\n'.format(path)
            else:
                html += path + '\n'
        self.path_edit.setHtml(html)

    def to_raw(self):
        self.path_edit.setEnabled(False)
        self.browse_button.setEnabled(False)
        text = self.path_edit.toPlainText()
        paths = text.splitlines()
        for i, path in enumerate(paths):
            self.update_path_edit(paths, i)
            io.to_raw_single(path)
        self.update_path_edit(paths, i + 1)
        self.path_edit.setEnabled(True)
        self.browse_button.setEnabled(True)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
