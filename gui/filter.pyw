"""
    gui/filter
    ~~~~~~~~~~~~~~~~~~~~

    Graphical user interface for picasso.filter

    :author: Joerg Schnitzbauer, 2015
"""


import sys
import traceback
from PyQt4 import QtCore, QtGui
import h5py
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import numpy as np
import os.path


plt.style.use('ggplot')


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, locs, parent=None):
        super().__init__(parent)
        self.locs = locs

    def columnCount(self, parent):
        return len(self.locs[0])

    def rowCount(self, parent):
        return self.locs.shape[0]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            data = self.locs[index.row()][index.column()]
            return str(data)
        return None

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.locs.dtype.names[section]
            elif orientation == QtCore.Qt.Vertical:
                return section
        return None


class TableView(QtGui.QTableView):

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        extension = os.path.splitext(path)[1].lower()
        if extension == '.hdf5':
            self.window.open(path)


class Window(QtGui.QMainWindow):

    def __init__(self):
        super().__init__()
        # Init GUI
        self.setWindowTitle('Picasso: Filter')
        self.resize(1100, 750)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, 'filter.ico')
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        open_action = file_menu.addAction('Open')
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        plot_menu = menu_bar.addMenu('Plot')
        histogram_action = plot_menu.addAction('Histogram')
        histogram_action.setShortcut('Ctrl+H')
        histogram_action.triggered.connect(self.plot_histogram)
        filter_menu = menu_bar.addMenu('Filter')
        self.table_view = TableView(self, self)
        self.table_view.setAcceptDrops(True)
        self.setCentralWidget(self.table_view)
        self.plot_windows = []

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open localizations', filter='*.hdf5')
        if path:
            self.open(path)

    def open(self, path):
        with h5py.File(path) as hdf:
            self.locs = hdf['locs'][...]
        table_model = TableModel(self.locs, self)
        self.table_view.setModel(table_model)

    def plot_histogram(self):
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) > 0:
            for index in indices:
                index = index.column()
                column = self.locs.dtype.names[index]
                data = self.locs[column]
                data = data[np.isfinite(data)]
                figure = plt.Figure()
                figure.suptitle(column)
                axes = figure.add_subplot(111)
                axes.hist(data, bins=int(len(self.locs)/1000), rwidth=1, linewidth=0)
                canvas = FigureCanvasQTAgg(figure)
                window = PlotWindow(self, canvas)
                self.plot_windows.append(window)
                window.show()


class PlotWindow(QtGui.QWidget):

    def __init__(self, window, canvas):
        super().__init__()
        self.main_window = window
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(canvas)
        vbox.addWidget((NavigationToolbar2QT(canvas, self)))
        self.setWindowTitle('Histogram')

    def closeEvent(self, event):
        self.main_window.plot_windows.remove(self)
        event.accept()

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
