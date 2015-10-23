#!/usr/bin/env python
"""
    gui/toraw
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.io.to_raw

    :author: Joerg Schnitzbauer, 2015
"""

import sys
from PyQt4 import QtGui
from picasso import io


class Window(QtGui.QWidget):

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: ToRaw')
        self.setWindowIcon(QtGui.QIcon('toraw.ico'))


def main():
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
