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
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'toraw.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(512, 1)
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        self.files_edit = QtGui.QLineEdit()
        hbox.addWidget(self.files_edit)
        browse_button = QtGui.QPushButton('Browse')
        hbox.addWidget(browse_button)
        hbox2 = QtGui.QHBoxLayout()
        vbox.addLayout(hbox2)
        go_button = QtGui.QPushButton('Convert')
        hbox2.addWidget(go_button)
        hbox2.addStretch(1)
        hbox2.addWidget(go_button)
        hbox2.addStretch(1)


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
