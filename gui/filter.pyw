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
from matplotlib.widgets import SpanSelector
import numpy as np
import os.path
from picasso import io


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


class PlotWindow(QtGui.QWidget):

    def __init__(self, main_window, locs, field):
        super().__init__()
        self.main_window = main_window
        self.locs = locs
        self.field = field
        self.figure = plt.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.plot()
        vbox = QtGui.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        vbox.addWidget((NavigationToolbar2QT(self.canvas, self)))
        self.setWindowTitle('Picasso: Filter')

    def plot(self):
        # Prepare the data
        data = self.locs[self.field]
        data = data[np.isfinite(data)]
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        bin_size = 2 * iqr * len(data)**(-1/3)
        if data.dtype.kind in ('u', 'i') and bin_size < 1:
            bin_size = 1
        bin_min = max(data.min() - bin_size / 2, 0)
        n_bins = min(1000, int(np.ceil((data.max() - bin_min) / bin_size)))
        bins = np.linspace(bin_min, data.max(), n_bins)
        # Prepare the figure
        self.figure.clear()
        self.figure.suptitle(self.field)
        axes = self.figure.add_subplot(111)
        axes.hist(data, bins, rwidth=1, linewidth=0)
        data_range = data.ptp()
        axes.set_xlim([bin_min - 0.05*data_range, data.max() + 0.05*data_range])
        SpanSelector(axes, self.on_span_select, 'horizontal', useblit=True)
        self.canvas.draw()

    def on_span_select(self, xmin, xmax):
        self.locs = self.locs[np.isfinite(self.locs[self.field])]
        self.locs = self.locs[(self.locs[self.field] > xmin) & (self.locs[self.field] < xmax)]
        self.main_window.update_locs(self.locs)
        self.main_window.log_filter(self.field, xmin, xmax)
        self.plot()


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
        save_action = file_menu.addAction('Save')
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.triggered.connect(self.save_file_dialog)
        file_menu.addAction(save_action)
        plot_menu = menu_bar.addMenu('Plot')
        histogram_action = plot_menu.addAction('Histogram')
        histogram_action.setShortcut('Ctrl+H')
        histogram_action.triggered.connect(self.plot_histogram)
        self.table_view = TableView(self, self)
        self.table_view.setAcceptDrops(True)
        self.setCentralWidget(self.table_view)
        self.plot_windows = {}
        self.filter_log = {}

    def open_file_dialog(self):
        path = QtGui.QFileDialog.getOpenFileName(self, 'Open localizations', filter='*.hdf5')
        if path:
            self.open(path)

    def open(self, path):
        locs, self.info = io.load_locs(path)
        self.locs_path = path
        self.update_locs(locs)
        for field in self.locs.dtype.names:
            self.plot_windows[field] = None
            self.filter_log[field] = None

    def plot_histogram(self):
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) > 0:
            for index in indices:
                index = index.column()
                field = self.locs.dtype.names[index]
                if self.plot_windows[field]:
                    self.plot_windows[field].show()
                else:
                    self.plot_windows[field] = PlotWindow(self, self.locs, field)
                    self.plot_windows[field].show()

    def update_locs(self, locs):
        self.locs = locs
        table_model = TableModel(self.locs, self)
        self.table_view.setModel(table_model)

    def log_filter(self, field, xmin, xmax):
        xmin = xmin.item()
        xmax = xmax.item()
        if self.filter_log[field]:
            self.filter_log[field][0] = max(xmin, self.filter_log[field][0])
            self.filter_log[field][1] = min(xmax, self.filter_log[field][1])
        else:
            self.filter_log[field] = [xmin, xmax]

    def save_file_dialog(self):
        base, ext = os.path.splitext(self.locs_path)
        out_path = base + '_filter.hdf5'
        path = QtGui.QFileDialog.getSaveFileName(self, 'Save localizations', out_path, filter='*.hdf5')
        if path:
            filter_info = self.filter_log.copy()
            filter_info.update({'Generated by': 'Picasso Filter'})
            info = self.info + [filter_info]
            io.save_locs(path, self.locs, info)

    def closeEvent(self, event):
        QtGui.qApp.closeAllWindows()


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
