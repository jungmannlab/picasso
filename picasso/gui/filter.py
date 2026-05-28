"""
picasso.gui.filter
~~~~~~~~~~~~~~~~~~

Graphical user interface for filtering localization lists.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os.path
import sys
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
from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io, lib, clusterer, __version__

plt.style.use("ggplot")

ROW_HEIGHT = 30


class TableModel(QtCore.QAbstractTableModel):
    """Qt model for the localization table view.

    A single instance is reused for the lifetime of the main window —
    the ``set_data`` method swaps in the small slice of rows currently
    visible.

    Parameters
    ----------
    parent : QtWidgets.QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.locs = pd.DataFrame()
        self.index = 0
        self._column_count = 0
        self._row_count = 0

    def set_data(self, locs: pd.DataFrame, index: int) -> None:
        """Replace the visible slice. Called on every scroll/refresh."""
        self.beginResetModel()
        self.locs = locs
        self.index = index
        self._column_count = len(locs.columns)
        self._row_count = locs.shape[0]
        self.endResetModel()

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return self._column_count

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return self._row_count

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            data = self.locs.iloc[index.row(), index.column()]
            return str(data)
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int,
    ) -> str | None:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return self.locs.columns[section]
            elif orientation == QtCore.Qt.Orientation.Vertical:
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
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        vertical_header = self.verticalHeader()
        vertical_header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Fixed
        )
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

    Holds only a reference to the main window and the field name(s) it
    plots. The localization data is pulled from the main window on
    demand to avoid retaining per-window copies (as before v0.10.1).

    Attributes
    ----------
    main_window : QtWidgets.QMainWindow
        Main window.
    figure : plt.Figure
        Matplotlib figure.

    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Main window.
    """

    def __init__(self, main_window: QtWidgets.QMainWindow) -> None:
        super().__init__()
        self.main_window = main_window
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

    def refresh(self) -> None:
        self.plot()
        self.update()

    def plot(self) -> None:
        pass


class HistWindow(PlotWindow):
    """Window for displaying 1D histograms.

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
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        field: str,
    ) -> None:
        self.field = field
        super().__init__(main_window)

    def plot(self) -> None:
        data = self.main_window.get_column(self.field)
        if data.dtype.kind == "f":
            data = data[np.isfinite(data)]
        self.figure.clear()
        self.figure.suptitle(self.field)
        axes = self.figure.add_subplot(111)
        if len(data) == 0:
            self.canvas.draw()
            return
        bins = lib.calculate_optimal_bins(data, 1000)
        axes.hist(data, bins, rwidth=1, linewidth=0)
        data_max = data.max()
        data_range = data_max - data.min()
        axes.set_xlim(
            [bins[0] - 0.05 * data_range, data_max + 0.05 * data_range]
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
        """Apply the selected range as a filter on the main window."""
        self.main_window.apply_range(self.field, float(xmin), float(xmax))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.main_window.hist_windows[self.field] = None
        event.accept()


class Hist2DWindow(PlotWindow):
    """Window for displaying 2D histograms.

    Attributes
    ----------
    field_x, field_y : str
        Field name for the x- and y-axis.

    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Main window.
    field_x : str
        Field name for the x-axis.
    field_y : str
        Field name for the y-axis.
    """

    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        field_x: str,
        field_y: str,
    ) -> None:
        self.field_x = field_x
        self.field_y = field_y
        super().__init__(main_window)
        self.resize(1000, 800)

    def plot(self) -> None:
        x, y = self.main_window.get_columns([self.field_x, self.field_y])
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        if len(x) == 0:
            axes.get_xaxis().set_label_text(self.field_x)
            axes.get_yaxis().set_label_text(self.field_y)
            self.canvas.draw()
            return
        bins_x = lib.calculate_optimal_bins(x, 1000)
        bins_y = lib.calculate_optimal_bins(y, 1000)
        nx = len(bins_x) - 1
        ny = len(bins_y) - 1
        x_min, x_max = float(bins_x[0]), float(bins_x[-1])
        y_min, y_max = float(bins_y[0]), float(bins_y[-1])
        counts = lib.hist2d_numba(
            np.ascontiguousarray(x),
            np.ascontiguousarray(y),
            x_min,
            x_max,
            y_min,
            y_max,
            nx,
            ny,
        )
        masked = np.ma.masked_equal(counts.T, 0)
        image = axes.pcolormesh(
            bins_x, bins_y, masked, norm=LogNorm(), shading="flat"
        )
        if x.dtype.kind == "f":
            x_data_max = float(np.nanmax(x))
            x_data_min = float(np.nanmin(x))
            y_data_max = float(np.nanmax(y))
            y_data_min = float(np.nanmin(y))
        else:
            x_data_max = float(x.max())
            x_data_min = float(x.min())
            y_data_max = float(y.max())
            y_data_min = float(y.min())
        x_range = x_data_max - x_data_min
        y_range = y_data_max - y_data_min
        axes.set_xlim(
            [bins_x[0] - 0.05 * x_range, x_data_max + 0.05 * x_range]
        )
        axes.set_ylim(
            [bins_y[0] - 0.05 * y_range, y_data_max + 0.05 * y_range]
        )
        self.figure.colorbar(image, ax=axes)
        axes.grid(False)
        axes.get_xaxis().set_label_text(self.field_x)
        axes.get_yaxis().set_label_text(self.field_y)
        self.selector = RectangleSelector(
            axes,
            self.on_rect_select,
            useblit=True,
            props=dict(facecolor="green", alpha=0.2, fill=True),
        )
        self.canvas.draw()

    def on_rect_select(
        self,
        press_event: QtGui.QMouseEvent,
        release_event: QtGui.QMouseEvent,
    ) -> None:
        """Apply the rectangular selection as a 2D filter on the main
        window."""
        x1, y1 = press_event.xdata, press_event.ydata
        x2, y2 = release_event.xdata, release_event.ydata
        xmin = float(min(x1, x2))
        xmax = float(max(x1, x2))
        ymin = float(min(y1, y2))
        ymax = float(max(y1, y2))
        self.main_window.apply_range2d(
            self.field_x, xmin, xmax, self.field_y, ymin, ymax
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.main_window.hist2d_windows[self.field_x][self.field_y] = None
        event.accept()


class FilterNum(lib.Dialog):
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

    DOCS_URL = "https://picassosr.readthedocs.io/en/latest/filter.html"

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.setToolTip(
            "Choose the parameter to filter by.\n"
            "The specified range is inclusive, i.e.,\n"
            "the min/max values are kept."
        )
        self.window = window
        self.setWindowTitle("Filter by numeric values")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "filter.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)

        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        # combox box with all atributes
        self.layout.addWidget(lib.HelpButton(self.DOCS_URL), 0, 0)
        self.attributes = QtWidgets.QComboBox(self)
        self.attributes.setEditable(False)
        self.layout.addWidget(self.attributes, 0, 1)

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
        filter_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        filter_button.clicked.connect(self.filter)
        self.layout.addWidget(filter_button, 3, 0, 1, 2)

    def filter(self) -> None:
        """Filters locs given the range values."""
        xmin = self.min.value()
        xmax = self.max.value()
        if xmin < xmax:
            field = self.attributes.currentText()
            self.window.apply_range(field, xmin, xmax, inclusive=True)

    def on_locs_loaded(self) -> None:
        """Changes attributes in the dialog according to locs.dtypes."""
        while self.attributes.count():
            self.attributes.removeItem(0)
        names = self.window.locs_full.columns
        for name in names:
            self.attributes.addItem(name)


class SubclusterNum(lib.Dialog):
    """Input dialog for specifying the distances used for testing
    for subclustering.

    ...

    Attributes
    ----------
    distance_clustered : QtWidgets.QDoubleSpinBox
        Spin box for maximum distance between clustered molecules (nm).
    distance_sparse : QtWidgets.QDoubleSpinBox
        Spin box for minimum distance between sparse molecules (nm).
    save_vals : QtWidgets.QCheckBox
        Checkbox for saving histogram values.

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
        self.layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        test_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        test_button.clicked.connect(self.plot)
        self.layout.addRow(test_button)

    def plot(self) -> None:
        """Plot the subclustering test histogram. Optionally can save
        the histogram values to a .csv file."""
        if "n_events" not in self.window.locs_full.columns:
            raise ValueError("Data must have the 'n_events' attribute.")
        if self.save_vals.isChecked():
            base, ext = os.path.splitext(self.window.locs_path)
            out_path = base + "_subcluster_test.csv"
            path, ext = lib.get_save_filename_ext_dialog(
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
        mols = self.window.materialize_filtered()
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
            clustered_nevents,
            sparse_nevents,
            return_fig=True,
            clustering_dist=dist_clustered,
            sparse_dist=dist_sparse,
        )
        plt.show()


class Window(QtWidgets.QMainWindow):
    """Main window for the application.

    The localization data is stored once in ``locs_full`` and never
    copied. The currently-visible subset is tracked by ``filtered_idx``,
    an integer index array into ``locs_full``. All filters re-index
    this array, so memory cost is O(N) once (the master) plus O(M)
    for the index (M = current filtered count), independent of how
    many filter steps have been applied.

    Attributes
    ----------
    filter_log : dict
        Dictionary of filter logs, i.e., data on what attribute/field
        was filtered and the corresponding min/max values.
    filter_num : FilterNum
        Filter dialog for numeric values.
    hist_windows : dict
        Dictionary of histogram windows.
    hist2d_windows : dict
        Dictionary of 2D histogram windows.
    locs_full : pd.DataFrame
        Master localizations table, set on load.
    filtered_idx : np.ndarray
        Integer indices into ``locs_full`` of the currently-visible
        rows.
    pwd : str
        Current working directory.
    table_view : TableView
        Table view for displaying data.
    """

    DOCS_URL = "https://picassosr.readthedocs.io/en/latest/filter.html"

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
        self.table_model = TableModel(self)
        self.table_view.setModel(self.table_model)
        self.filter_num = FilterNum(self)
        self.metadata_dialog = lib.MetadataDialog(self)
        self.user_settings_dialog = lib.UserSettingsDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save")
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file_dialog)
        export_csv_action = file_menu.addAction("Export as CSV")
        export_csv_action.triggered.connect(self.export_csv_dialog)
        metadata_action = file_menu.addAction("Show metadata")
        metadata_action.setShortcut("Ctrl+M")
        metadata_action.triggered.connect(self.show_metadata)
        picasso_settings_action = file_menu.addAction("Picasso settings")
        picasso_settings_action.triggered.connect(
            self.user_settings_dialog.show
        )
        help_action = file_menu.addAction("Help")
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )
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
        filter_action = filter_menu.addAction("Filter numerically")
        filter_action.setShortcut("Ctrl+F")
        filter_action.triggered.connect(self.filter_num.show)
        apply_from_metadata_action = filter_menu.addAction(
            "Apply filters from metadata"
        )
        apply_from_metadata_action.triggered.connect(
            self.apply_filters_from_metadata
        )
        remove_columns_action = filter_menu.addAction("Remove columns")
        remove_columns_action.triggered.connect(self.remove_columns)
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
        self.locs_full = None
        self.filtered_idx = None

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

    @property
    def locs(self) -> pd.DataFrame:
        """Materialise the currently-filtered localizations.

        Provided for backward compatibility (e.g. plugins). This
        allocates a fresh DataFrame on every access — prefer
        ``get_column``, ``get_columns`` or ``materialize_filtered``.
        """
        return self.materialize_filtered()

    @property
    def n_filtered(self) -> int:
        """Number of currently-visible rows."""
        if self.locs_full is None:
            return 0
        if self.filtered_idx is None:
            return len(self.locs_full)
        return len(self.filtered_idx)

    def get_column(self, field: str) -> np.ndarray:
        """Return the currently-filtered values of one column as a
        numpy array. When no filter is applied this is a view into the
        master DataFrame (no copy).
        """
        vals = self.locs_full[field].values
        return vals if self.filtered_idx is None else vals[self.filtered_idx]

    def get_columns(
        self, fields: tuple[str]
    ) -> tuple[lib.FloatArray1D | lib.IntArray1D]:
        """Return the currently-filtered values of several columns as
        numpy arrays."""
        if self.filtered_idx is None:
            return tuple(self.locs_full[f].values for f in fields)
        return tuple(
            self.locs_full[f].values[self.filtered_idx] for f in fields
        )

    def materialize_filtered(self) -> pd.DataFrame:
        """Build a DataFrame of the currently-filtered localizations."""
        if self.filtered_idx is None:
            return self.locs_full.reset_index(drop=True)
        return self.locs_full.iloc[self.filtered_idx].reset_index(drop=True)

    def _idx_dtype(self, n: int):
        return np.uint32 if n <= np.iinfo(np.uint32).max else np.int64

    def apply_range(
        self,
        field: str,
        xmin: float,
        xmax: float,
        *,
        inclusive: bool = False,
    ) -> None:
        """Filter the active index to rows with ``field`` in the given
        range. Non-finite values are always dropped.

        ``inclusive=False`` matches the histogram-selection semantics
        (strict ``>`` / ``<``); ``inclusive=True`` matches the numeric
        filter dialog (``>=`` / ``<=``).
        """
        col = self.get_column(field)
        if inclusive:
            keep = (col >= xmin) & (col <= xmax)
        else:
            keep = (col > xmin) & (col < xmax)
        if col.dtype.kind == "f":
            keep &= np.isfinite(col)
        if self.filtered_idx is None:
            self.filtered_idx = np.flatnonzero(keep).astype(
                self._idx_dtype(len(self.locs_full)), copy=False
            )
        else:
            self.filtered_idx = self.filtered_idx[keep]
        self.log_filter(field, xmin, xmax)
        self.refresh()

    def apply_range2d(
        self,
        field_x: str,
        xmin: float,
        xmax: float,
        field_y: str,
        ymin: float,
        ymax: float,
    ) -> None:
        """Apply a 2D rectangular filter in one pass."""
        cx, cy = self.get_columns([field_x, field_y])
        keep = (cx > xmin) & (cx < xmax) & (cy > ymin) & (cy < ymax)
        if cx.dtype.kind == "f":
            keep &= np.isfinite(cx)
        if cy.dtype.kind == "f":
            keep &= np.isfinite(cy)
        if self.filtered_idx is None:
            self.filtered_idx = np.flatnonzero(keep).astype(
                self._idx_dtype(len(self.locs_full)), copy=False
            )
        else:
            self.filtered_idx = self.filtered_idx[keep]
        self.log_filter(field_x, xmin, xmax)
        self.log_filter(field_y, ymin, ymax)
        self.refresh()

    def refresh(self) -> None:
        """Refresh the table view and any open histogram windows."""
        n = self.n_filtered
        self.vertical_scrollbar.setMaximum(max(0, n - 1))
        self.display_locs(self.vertical_scrollbar.value())
        for w in self.hist_windows.values():
            if w:
                w.refresh()
        for d in self.hist2d_windows.values():
            for w in d.values():
                if w:
                    w.refresh()

    def show_metadata(self) -> None:
        """Open the metadata dialog."""
        if self.locs_full is None:
            QtWidgets.QMessageBox.information(
                self, "Metadata", "No file loaded."
            )
            return
        label = os.path.basename(self.locs_path)
        self.metadata_dialog.set_infos(self.info, labels=label)
        self.metadata_dialog.show()
        self.metadata_dialog.raise_()

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
        if self.locs_full is not None:
            for column in self.locs_full.columns:
                if self.hist_windows.get(column):
                    self.hist_windows[column].close()
                for column_y in self.locs_full.columns:
                    if self.hist2d_windows.get(column, {}).get(column_y):
                        self.hist2d_windows[column][column_y].close()
        self.locs_path = path
        self.locs_full = locs
        # No filter applied yet — filtered_idx stays None so that we
        # avoid allocating a 1:1 index of every row.
        self.filtered_idx = None
        self.filter_log = {}
        self.hist_windows = {}
        self.hist2d_windows = {}
        for column in self.locs_full.columns:
            self.hist_windows[column] = None
            self.hist2d_windows[column] = {}
            for column_y in self.locs_full.columns:
                self.hist2d_windows[column][column_y] = None
            self.filter_log[column] = None
        self.filter_num.on_locs_loaded()
        self.refresh()

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
                field = self.locs_full.columns[index]
                if not self.hist_windows[field]:
                    self.hist_windows[field] = HistWindow(self, field)
                self.hist_windows[field].show()

    def plot_hist2d(self) -> None:
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) == 2:
            indices = [index.column() for index in indices]
            field_x, field_y = [
                self.locs_full.columns[index] for index in indices
            ]
            if not self.hist2d_windows[field_x][field_y]:
                self.hist2d_windows[field_x][field_y] = Hist2DWindow(
                    self, field_x, field_y
                )
            self.hist2d_windows[field_x][field_y].show()

    def plot_subclustering(self) -> None:
        self.subcluster_num = SubclusterNum(self)
        self.subcluster_num.show()

    def display_locs(self, index: int) -> None:
        if self.locs_full is None:
            return
        view_height = self.table_view.viewport().height()
        n_rows = int(view_height / ROW_HEIGHT) + 2
        end = min(index + n_rows, self.n_filtered)
        if end <= index:
            visible = self.locs_full.iloc[0:0]
        elif self.filtered_idx is None:
            visible = self.locs_full.iloc[index:end]
        else:
            rows_idx = self.filtered_idx[index:end]
            visible = self.locs_full.iloc[rows_idx]
        self.table_model.set_data(visible, index)

    def log_filter(self, field: str, xmin: float, xmax: float) -> None:
        if self.filter_log[field]:
            self.filter_log[field][0] = max(xmin, self.filter_log[field][0])
            self.filter_log[field][1] = min(xmax, self.filter_log[field][1])
        else:
            self.filter_log[field] = [xmin, xmax]

    def remove_columns(self) -> None:
        """Remove columns from the loaded dataset."""
        if self.locs_full is None:
            return
        columns = self.locs_full.columns.to_list()
        to_remove, ok = lib.RemoveColumnsDialog.getParams(self, columns)
        if not ok or len(to_remove) == 0:
            return
        self.locs_full.drop(columns=to_remove, inplace=True)
        self.refresh()
        if "Removed columns" in self.filter_log:
            self.filter_log["Removed columns"].extend(to_remove)
        else:
            self.filter_log["Removed columns"] = to_remove

    def _build_filter_summary(self, ranges, to_remove, missing) -> str:
        lines = []
        if ranges:
            lines.append("Filters to apply:")
            for field, (xmin, xmax) in ranges.items():
                lines.append(f"  {field}: [{xmin}, {xmax}]")
        if to_remove:
            if lines:
                lines.append("")
            lines.append("Columns to remove:")
            for c in to_remove:
                lines.append(f"  {c}")
        if missing:
            if lines:
                lines.append("")
            lines.append("Not found in current data (will be skipped):")
            for c in missing:
                lines.append(f"  {c}")
        lines.append("")
        lines.append("Apply these steps?")
        return "\n".join(lines)

    def _load_filter_metadata(self):
        directory = self.pwd if self.pwd else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open metadata",
            directory=directory,
            filter="*.yaml",
        )
        if not path:
            return None
        try:
            return io.load_info(path, qt_parent=self)
        except io.NoMetadataFileError:
            return None

    def apply_filters_from_metadata(self) -> None:
        """Replay filter steps recorded in another file's .yaml metadata
        onto the currently loaded localizations."""
        if self.locs_full is None:
            QtWidgets.QMessageBox.information(
                self, "Apply filters from metadata", "No file loaded."
            )
            return

        info = self._load_filter_metadata()
        if info is None:
            return

        ranges, to_remove, missing = lib.extract_filter_steps(
            info, self.locs_full.columns
        )

        if not ranges and not to_remove:
            msg = "No applicable filter steps found in metadata."
            if missing:
                msg += (
                    "\n\nReferenced columns not found in current data:\n  "
                    + "\n  ".join(missing)
                )
            QtWidgets.QMessageBox.information(
                self, "Apply filters from metadata", msg
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Apply filters from metadata",
            self._build_filter_summary(ranges, to_remove, missing),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        for field, (xmin, xmax) in ranges.items():
            self.apply_range(field, xmin, xmax)
        if to_remove:
            self.locs_full.drop(columns=list(to_remove), inplace=True)
            if "Removed columns" in self.filter_log:
                self.filter_log["Removed columns"].extend(to_remove)
            else:
                self.filter_log["Removed columns"] = list(to_remove)
            self.refresh()

    def export_csv_dialog(self) -> None:
        if self.locs_full is None:
            return
        base, ext = os.path.splitext(self.locs_path)
        out_path = base + ".csv"
        path, exe = lib.get_save_filename_ext_dialog(
            self,
            "Export as CSV",
            out_path,
            filter="*.csv",
        )
        if path:
            self.materialize_filtered().to_csv(path, index=False)

    def save_file_dialog(self) -> None:
        if self.locs_full is None:
            return
        if "x" in self.locs_full.columns:  # Saving only for locs
            base, ext = os.path.splitext(self.locs_path)
            out_path = base + "_filter.hdf5"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save localizations",
                out_path,
                filter="*.hdf5",
                check_ext=".yaml",
            )
            if path:
                filter_info = self.filter_log.copy()
                filter_info.update(
                    {"Generated by": f"Picasso v{__version__} Filter"}
                )
                info = self.info + [filter_info]
                io.save_locs(path, self.materialize_filtered(), info)
        else:
            raise NotImplementedError("Saving only implemented for locs.")

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        new_value = (
            self.vertical_scrollbar.value() - 0.1 * event.angleDelta().y()
        )
        self.vertical_scrollbar.setValue(int(new_value))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.display_locs(self.vertical_scrollbar.value())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        settings = io.load_user_settings()
        if self.locs_full is not None:
            settings["Filter"]["PWD"] = self.pwd
            io.save_user_settings(settings)
        QtWidgets.QApplication.instance().closeAllWindows()


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

    from ..updater import setup_gui_update_check

    setup_gui_update_check(window)

    lib.install_excepthook(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
