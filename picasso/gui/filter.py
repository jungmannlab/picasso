"""
picasso.gui.filter
~~~~~~~~~~~~~~~~~~

Graphical user interface for filtering localization lists.

:authors: Joerg Schnitzbauer Maximilian Thomas Strauss, 2015-2018
:copyright: Copyright (c) 2015=2018 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os.path
import sys
import traceback
import importlib
import pkgutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.widgets import SpanSelector, RectangleSelector
from matplotlib.colors import LogNorm
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import io, lib, clusterer, __version__

plt.style.use("ggplot")

ROW_HEIGHT = 30


class TableModel(QtCore.QAbstractTableModel):
    """Class for handling the localization data.

    ...

    Attributes
    ----------
    _column_count : int
        Number of columns.
    index : QtCore.QModelIndex
        Row/column.
    locs : pd.DataFrame
        Localizations.
    _row_count : int
        Number of rows.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    index : QtCore.QModelIndex
        Row/column.
    parent : QtWidgets.QWidget, optional
        Parent widget. Can be set to None.
    """

    def __init__(
        self,
        locs: pd.DataFrame,
        index: QtCore.QModelIndex,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.locs = locs
        self.index = index
        try:
            self._column_count = len(locs.columns)
        except IndexError:
            self._column_count = 0
        self._row_count = self.locs.shape[0]

    def columnCount(self, parent: None) -> int:
        return self._column_count

    def rowCount(self, parent: None) -> int:
        return self._row_count

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.DisplayRole,
    ) -> str | None:
        if role == QtCore.Qt.DisplayRole:
            data = self.locs.iloc[index.row(), index.column()]
            return str(data)
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int,
    ) -> str | None:
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.locs.columns[section]
            elif orientation == QtCore.Qt.Vertical:
                return self.index + section
        return None


class TableView(QtWidgets.QTableView):
    """Custom table view for displaying localization data.

    ...

    Attributes
    ----------
    window : QtWidgets.QMainWindow
        Main window.

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        Main window.
    parent : QtWidgets.QWidget, optional
        Parent widget. Can be set to None.
    """

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.window = window
        self.setAcceptDrops(True)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        vertical_header = self.verticalHeader()
        vertical_header.sectionResizeMode(QtWidgets.QHeaderView.Fixed)
        vertical_header.setDefaultSectionSize(ROW_HEIGHT)
        vertical_header.setFixedWidth(70)

    def dragEnterEvent(self, event: QtCore.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtCore.QDragMoveEvent) -> None:
        event.accept()

    def dropEvent(self, event: QtCore.QDropEvent) -> None:
        urls = event.mimeData().urls()
        path = urls[0].toLocalFile()
        extension = os.path.splitext(path)[1].lower()
        if extension == ".hdf5":
            self.window.open(path)


class PlotWindow(QtWidgets.QWidget):
    """Window for displaying 1D/2D histograms.

    ...

    Attributes
    ----------
    main_window : QtWidgets.QMainWindow
        Main window.
    locs : pd.DataFrame
        Localization data.
    figure : plt.Figure
        Matplotlib figure.

    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Main window.
    locs : pd.DataFrame
        Localization data.
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        locs: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.main_window = main_window
        self.locs = locs
        self.figure = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.plot()
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        vbox.addWidget((NavigationToolbar2QT(self.canvas, self)))
        self.setWindowTitle(f"Picasso v{__version__}: Filter")

        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "filter.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)

    def update_locs(self, locs: pd.DataFrame) -> None:
        self.locs = locs
        self.plot()
        self.update()

    def plot(self) -> None:
        pass


class HistWindow(PlotWindow):
    """Window for displaying 1D histograms.

    ...

    Attributes
    ----------
    field : str
        Field name for the histogram.

    Parameters
    ----------
    field : str
        Field name for the histogram.
    main_window : QtWidgets.QMainWindow
        Main window.
    locs : pd.DataFrame
        Localization data.
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        locs: pd.DataFrame,
        field: str,
    ) -> None:
        self.field = field
        super().__init__(main_window, locs)

    def plot(self) -> None:
        # Prepare the data
        data = self.locs[self.field]
        data = data[np.isfinite(data)]
        bins = lib.calculate_optimal_bins(data, 1000)
        # Prepare the figure
        self.figure.clear()
        self.figure.suptitle(self.field)
        axes = self.figure.add_subplot(111)
        axes.hist(data, bins, rwidth=1, linewidth=0)
        data_range = data.max() - data.min()
        axes.set_xlim(
            [bins[0] - 0.05 * data_range, data.max() + 0.05 * data_range]
        )
        self.span = SpanSelector(
            axes,
            self.on_span_select,
            "horizontal",
            useblit=True,
            props=dict(facecolor="green", alpha=0.2),
        )
        self.canvas.draw()

    def on_span_select(self, xmin: float, xmax: float) -> None:
        """Update the localization data based on the selected in the
        histogram plot."""
        x = self.locs[self.field].to_numpy()
        valid_idx = np.isfinite(x) & (x > xmin) & (x < xmax)
        self.locs = self.locs[valid_idx]
        self.main_window.update_locs(self.locs)
        self.main_window.log_filter(self.field, xmin.item(), xmax.item())
        self.plot()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.main_window.hist_windows[self.field] = None
        event.accept()


class Hist2DWindow(PlotWindow):
    """Window for displaying 2D histograms.

    ...

    Attributes
    ----------
    field_x, field_y : str
        Field name for the x- and y-axis.

    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Main window.
    locs : pd.DataFrame
        Localization data.
    field_x : str
        Field name for the x-axis.
    field_y : str
        Field name for the y-axis.
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        locs: pd.DataFrame,
        field_x: str,
        field_y: str,
    ) -> None:
        self.field_x = field_x
        self.field_y = field_y
        super().__init__(main_window, locs)
        self.resize(1000, 800)

    def plot(self) -> None:
        # Prepare the data
        x = self.locs[self.field_x]
        y = self.locs[self.field_y]
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        # Prepare the figure
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        # Start hist2 version
        bins_x = lib.calculate_optimal_bins(x, 1000)
        bins_y = lib.calculate_optimal_bins(y, 1000)
        counts, x_edges, y_edges, image = axes.hist2d(
            x, y, bins=[bins_x, bins_y], norm=LogNorm()
        )
        x_range = x.max() - x.min()
        axes.set_xlim([bins_x[0] - 0.05 * x_range, x.max() + 0.05 * x_range])
        y_range = y.max() - y.min()
        axes.set_ylim([bins_y[0] - 0.05 * y_range, y.max() + 0.05 * y_range])
        self.figure.colorbar(image, ax=axes)
        axes.grid(False)
        axes.get_xaxis().set_label_text(self.field_x)
        axes.get_yaxis().set_label_text(self.field_y)
        self.selector = RectangleSelector(
            axes,
            self.on_rect_select,
            useblit=False,
            props=dict(facecolor="green", alpha=0.2, fill=True),
        )
        self.canvas.draw()

    def on_rect_select(
        self,
        press_event: QtGui.QMouseEvent,
        release_event: QtGui.QMouseEvent,
    ) -> None:
        """Handle rectangle selection over the 2D histogram. Update
        localizations."""
        x1, y1 = press_event.xdata, press_event.ydata
        x2, y2 = release_event.xdata, release_event.ydata
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        x = self.locs[self.field_x].to_numpy()
        y = self.locs[self.field_y].to_numpy()
        finite_idx = np.isfinite(x) & np.isfinite(y)
        valid_idx = (
            (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & finite_idx
        )
        self.locs = self.locs[valid_idx]
        self.main_window.update_locs(self.locs)
        self.main_window.log_filter(self.field_x, xmin.item(), xmax.item())
        self.main_window.log_filter(self.field_y, ymin.item(), ymax.item())
        self.plot()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.main_window.hist2d_windows[self.field_x][self.field_y] = None
        event.accept()


class FilterNum(QtWidgets.QDialog):
    """Dialog for filtering localizations by numeric values.

    ...

    Attributes
    ----------
    attributes : QtWidgets.QComboBox
        Combo box for selecting attributes/fields.
    layout : QtWidgets.QGridLayout
        Layout for the dialog.
    max : QtWidgets.QDoubleSpinBox
        Spin box for maximum value for filtering.
    min : QtWidgets.QDoubleSpinBox
        Spin box for minimum value for filtering.
    window : QtWidgets.QMainWindow
        Main window.

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        Main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Filter by numeric values")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "filter.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        # combox box with all atributes
        self.attributes = QtWidgets.QComboBox(self)
        self.attributes.setEditable(False)
        self.layout.addWidget(self.attributes, 0, 0, 1, 2)

        # lower value
        self.layout.addWidget(QtWidgets.QLabel("Min:"), 1, 0)
        self.min = QtWidgets.QDoubleSpinBox()
        self.min.setValue(10)
        self.min.setDecimals(5)
        self.min.setRange(-9999999, 9999999)
        self.min.setSingleStep(1)
        self.min.setKeyboardTracking(True)
        self.layout.addWidget(self.min, 1, 1)

        # higher value
        self.layout.addWidget(QtWidgets.QLabel("Max:"), 2, 0)
        self.max = QtWidgets.QDoubleSpinBox()
        self.max.setValue(100)
        self.max.setDecimals(5)
        self.max.setRange(-9999999, 9999999)
        self.max.setSingleStep(1)
        self.max.setKeyboardTracking(True)
        self.layout.addWidget(self.max, 2, 1)

        # filter button
        filter_button = QtWidgets.QPushButton("Filter")
        filter_button.setFocusPolicy(QtCore.Qt.NoFocus)
        filter_button.clicked.connect(self.filter)
        self.layout.addWidget(filter_button, 3, 0, 1, 2)

    def filter(self) -> None:
        """Filters locs given the range values."""
        # check that min value < max value
        xmin = self.min.value()
        xmax = self.max.value()
        if xmin < xmax:
            field = self.attributes.currentText()
            locs = self.window.locs
            locs = locs[(locs[field] > xmin) & (locs[field] < xmax)]
            self.window.update_locs(locs)
            self.window.log_filter(field, xmin, xmax)

    def on_locs_loaded(self) -> None:
        """Changes attributes in the dialog according to locs.dtypes."""
        while self.attributes.count():
            self.attributes.removeItem(0)
        names = self.window.locs.columns
        for name in names:
            self.attributes.addItem(name)


class SubclusterNum(QtWidgets.QDialog):
    """Input dialog for specifying the distances used for testing
    for subclustering.

    ...

    Attributes

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        Main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Test subclustering")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "filter.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)

        self.layout = QtWidgets.QFormLayout()
        self.layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.setLayout(self.layout)

        self.distance_clustered = QtWidgets.QDoubleSpinBox()
        self.distance_clustered.setValue(25)
        self.distance_clustered.setDecimals(2)
        self.distance_clustered.setRange(0.01, 99999)
        self.distance_clustered.setSingleStep(1)
        self.distance_clustered.setKeyboardTracking(True)
        self.layout.addRow(
            "Max. dist. between clustered molecules (nm):",
            self.distance_clustered,
        )

        self.distance_sparse = QtWidgets.QDoubleSpinBox()
        self.distance_sparse.setValue(80)
        self.distance_sparse.setDecimals(2)
        self.distance_sparse.setRange(0.01, 99999)
        self.distance_sparse.setSingleStep(1)
        self.distance_sparse.setKeyboardTracking(True)
        self.layout.addRow(
            "Min. dist. between sparse molecules (nm):",
            self.distance_sparse,
        )

        self.save_vals = QtWidgets.QCheckBox("Save histogram values")
        self.save_vals.setChecked(False)
        self.layout.addRow(self.save_vals)
        test_button = QtWidgets.QPushButton("Test subclustering")
        test_button.setFocusPolicy(QtCore.Qt.NoFocus)
        test_button.clicked.connect(self.plot)
        self.layout.addRow(test_button)

    def plot(self) -> None:
        """Plot the subclustering test histogram. Optionally can save
        the histogram values to a .csv file."""
        if not hasattr(self.window.locs, "n_events"):
            raise ValueError("Data must have the 'n_events' attribute.")
        if self.save_vals.isChecked():
            base, ext = os.path.splitext(self.window.locs_path)
            out_path = base + "_subcluster_test.csv"
            path, ext = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save histogram values",
                out_path,
                filter="*.csv",
            )
            if not path:
                return
        else:
            path = None

        dist_clustered = self.distance_clustered.value()
        dist_sparse = self.distance_sparse.value()
        mols = self.window.locs
        clustered_nevents, sparse_nevents = clusterer.test_subclustering(
            mols, self.window.info, dist_clustered, dist_sparse
        )
        if path is not None:
            len_max = max(len(clustered_nevents), len(sparse_nevents))
            df = np.empty((len_max, 2))
            df.fill(np.nan)
            df[: len(clustered_nevents), 0] = clustered_nevents
            df[: len(sparse_nevents), 1] = sparse_nevents
            df = pd.DataFrame(
                df, columns=["clustered_nevents", "sparse_nevents"]
            )
            df.to_csv(path, index=False)
        fig, ax = lib.plot_subclustering_check(
            clustered_nevents, sparse_nevents, return_fig=True
        )
        plt.show()


class Window(QtWidgets.QMainWindow):
    """Main window for the application.

    ...

    Attributes
    ----------
    filter_log : dict
        Dictionary of filter logs, i.e., data on what attribute/field
        of ``self.locs`` was filtered and the corresponding min/max
        values.
    filter_num : FilterNum
        Filter dialog for numeric values.
    hist_windows : dict
        Dictionary of histogram windows.
    hist2d_windows : dict
        Dictionary of 2D histogram windows.
    locs : pd.DataFrame
        Localizations loaded.
    pwd : str
        Current working directory.
    table_view : TableView
        Table view for displaying data.
    """

    def __init__(self) -> None:
        super().__init__()
        # Init GUI
        self.setWindowTitle(f"Picasso v{__version__}: Filter")
        self.resize(1100, 750)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "filter.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.table_view = TableView(self, self)
        self.filter_num = FilterNum(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.triggered.connect(self.save_file_dialog)
        file_menu.addAction(save_action)
        plot_menu = menu_bar.addMenu("Plot")
        histogram_action = plot_menu.addAction("Histogram")
        histogram_action.setShortcut("Ctrl+H")
        histogram_action.triggered.connect(self.plot_histogram)
        scatter_action = plot_menu.addAction("2D Histogram")
        scatter_action.setShortcut("Ctrl+D")
        scatter_action.triggered.connect(self.plot_hist2d)
        test_subcluster_action = plot_menu.addAction("Test subclustering")
        test_subcluster_action.triggered.connect(self.plot_subclustering)
        filter_menu = menu_bar.addMenu("Filter")
        filter_action = filter_menu.addAction("Filter")
        filter_action.setShortcut("Ctrl+F")
        filter_menu.triggered.connect(self.filter_num.show)
        main_widget = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(main_widget)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        self.setCentralWidget(main_widget)
        hbox.addWidget(self.table_view)
        self.vertical_scrollbar = QtWidgets.QScrollBar()
        self.vertical_scrollbar.valueChanged.connect(self.display_locs)
        hbox.addWidget(self.vertical_scrollbar)
        self.hist_windows = {}
        self.hist2d_windows = {}
        self.filter_log = {}
        self.locs = None

        # load user settings (working directory)
        settings = io.load_user_settings()
        pwd = []
        try:
            pwd = settings["Filter"]["PWD"]
        except Exception as e:
            print(e)
            pass
        if len(pwd) == 0:
            pwd = []
        self.pwd = pwd

        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

    def open_file_dialog(self) -> None:
        if self.pwd == []:
            path, exe = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open localizations", filter="*.hdf5"
            )
        else:
            path, exe = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open localizations", directory=self.pwd, filter="*.hdf5"
            )
        if path:
            self.pwd = path
            self.open(path)

    def open(self, path: str) -> None:
        try:
            locs, self.info = io.load_filter(path, qt_parent=self)
        except io.NoMetadataFileError:
            return
        if self.locs is not None:
            for column in self.locs.columns:
                if self.hist_windows[column]:
                    self.hist_windows[column].close()
                for column_y in self.locs.columns:
                    if self.hist2d_windows[column][column_y]:
                        self.hist_windows[column][column_y].close()
        self.locs_path = path
        self.update_locs(locs)
        for column in self.locs.columns:
            self.hist_windows[column] = None
            self.hist2d_windows[column] = {}
            for column_y in self.locs.columns:
                self.hist2d_windows[column][column_y] = None
            self.filter_log[column] = None
        self.filter_num.on_locs_loaded()

        self.setWindowTitle(
            f"Picasso v{__version__}: Filter. File: {os.path.basename(path)}"
        )
        self.pwd = os.path.dirname(path)

    def plot_histogram(self) -> None:
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) > 0:
            for index in indices:
                index = index.column()
                field = self.locs.columns[index]
                if not self.hist_windows[field]:
                    self.hist_windows[field] = HistWindow(
                        self,
                        self.locs,
                        field,
                    )
                self.hist_windows[field].show()

    def plot_hist2d(self) -> None:
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) == 2:
            indices = [index.column() for index in indices]
            field_x, field_y = [self.locs.columns[index] for index in indices]
            if not self.hist2d_windows[field_x][field_y]:
                self.hist2d_windows[field_x][field_y] = Hist2DWindow(
                    self, self.locs, field_x, field_y
                )
            self.hist2d_windows[field_x][field_y].show()

    def plot_subclustering(self) -> None:
        self.subcluster_num = SubclusterNum(self)
        self.subcluster_num.show()

    def update_locs(self, locs: pd.DataFrame) -> None:
        self.locs = locs
        self.vertical_scrollbar.setMaximum(len(locs) - 1)
        self.display_locs(self.vertical_scrollbar.value())
        for field, hist_window in self.hist_windows.items():
            if hist_window:
                hist_window.update_locs(locs)
        for field_x, hist2d_windows in self.hist2d_windows.items():
            for field_y, hist2d_window in hist2d_windows.items():
                if hist2d_window:
                    hist2d_window.update_locs(locs)

    def display_locs(self, index: int) -> None:
        if self.locs is not None:
            view_height = self.table_view.viewport().height()
            n_rows = int(view_height / ROW_HEIGHT) + 2
            table_model = TableModel(
                self.locs[index : index + n_rows],
                index,
                self,
            )
            self.table_view.setModel(table_model)

    def log_filter(self, field: str, xmin: float, xmax: float) -> None:
        if self.filter_log[field]:
            self.filter_log[field][0] = max(xmin, self.filter_log[field][0])
            self.filter_log[field][1] = min(xmax, self.filter_log[field][1])
        else:
            self.filter_log[field] = [xmin, xmax]

    def save_file_dialog(self) -> None:
        if "x" in self.locs.columns:  # Saving only for locs
            base, ext = os.path.splitext(self.locs_path)
            out_path = base + "_filter.hdf5"
            path, exe = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save localizations", out_path, filter="*.hdf5"
            )
            if path:
                filter_info = self.filter_log.copy()
                filter_info.update(
                    {"Generated by": f"Picasso v{__version__} Filter"}
                )
                info = self.info + [filter_info]
                io.save_locs(path, self.locs, info)
        else:
            raise NotImplementedError("Saving only implmented for locs.")

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        new_value = (
            self.vertical_scrollbar.value() - 0.1 * event.angleDelta().y()
        )
        self.vertical_scrollbar.setValue(int(new_value))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.display_locs(self.vertical_scrollbar.value())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        settings = io.load_user_settings()
        if self.locs is not None:
            settings["Filter"]["PWD"] = self.pwd
            io.save_user_settings(settings)
        QtWidgets.qApp.closeAllWindows()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()

    from . import plugins

    def iter_namespace(pkg):
        return pkgutil.iter_modules(pkg.__path__, pkg.__name__ + ".")

    plugins = [
        importlib.import_module(name)
        for finder, name, ispkg in iter_namespace(plugins)
    ]

    for plugin in plugins:
        p = plugin.Plugin(window)
        if p.name == "filter":
            p.execute()

    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(
            window,
            "An error occured",
            message,
        )
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
