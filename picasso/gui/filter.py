"""
picasso.gui.filter
~~~~~~~~~~~~~~~~~~

Graphical user interface for filtering localization lists.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
import os.path
import sys
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtGui, QtWidgets

# must come after the PyQt6 import so that matplotlib's qt_compat
# selects the PyQt6 binding (picasso core no longer imports PyQt6)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.widgets import SpanSelector, RectangleSelector

from .. import (
    io,
    lib,
    lib_qt,
    clusterer,
    expression,
    legacy,
    __version__,
    docs_url,
)
from .app import run_gui
from . import idle
from . import theme
from . import toolbars
from . import plot_style
from .plot_style import PlotStyle

plt.style.use("ggplot")

ROW_HEIGHT = 30

_log = logging.getLogger(__name__)


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
        elif extension == ".yaml":
            self.window.apply_filters_from_metadata(path)


class PlotWindow(QtWidgets.QWidget):
    """Window for displaying 1D/2D histograms.

    Holds only a reference to the main window and the field name(s) it
    plots. The localization data is pulled from the main window on
    demand to avoid retaining per-window copies (as before v0.10.1).
    The appearance is the main window's ``plot_style``, shared by all
    histogram windows.

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
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.addSeparator()
        settings_action = toolbar.addAction("Plot settings")
        settings_action.setIcon(theme.icon("plot-settings"))
        settings_action.setToolTip(
            "Appearance of all chart windows and Filter's histograms"
        )
        settings_action.triggered.connect(main_window.show_plot_settings)
        theme.follow_toolbar_style(toolbar, settings_action)
        vbox.addWidget(toolbar)
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


#: Columns in nm that the legacy table does not list (they were never in
#: camera pixels).
_NM_COLUMNS = frozenset({"z", "lpz", "std_z"})
_UNITS = {1: "nm", 2: "nm^2", 3: "nm^3"}


def _axis_label(field: str | None) -> str | None:
    """The axis label of a column: its name, with the unit for lengths,
    areas and volumes (``x (nm)``, ``area (nm^2)``)."""
    if field is None:
        return None
    if field in _NM_COLUMNS:
        return f"{field} (nm)"
    base = field
    for suffix in ("_mean", "_std", "_min", "_max"):
        if field.endswith(suffix):
            base = field[: -len(suffix)]
            break
    power = legacy.LEGACY_PIXEL_COLUMNS.get(base)
    if power is None or (base != field and power != 1):
        return field
    return f"{field} ({_UNITS[power]})"


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
        style = self.main_window.plot_style
        data = self.main_window.get_column(self.field)
        if data.dtype.kind == "f":
            data = data[np.isfinite(data)]
        self.figure.clear()
        style.style_figure(self.figure)
        axes = self.figure.add_subplot(111)
        style.style_axes(axes)
        style.label(axes, _axis_label(self.field), "Counts")
        if len(data) == 0:
            self.canvas.draw()
            return
        bins = lib.calculate_optimal_bins(data, style.max_bins)
        counts, _ = np.histogram(data, bins)
        style.bars(axes, counts, bins)
        style.title(
            self.figure, f"{self.field}  ·  {len(data):,} localizations"
        )
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
            props=style.selection,
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

    The 2D histogram is flanked by the 1D histograms (marginal
    distributions) of both fields, computed from the same bins, unless
    turned off in the plot settings. A rectangle drawn on the 2D
    histogram filters both fields; a span selected on either 1D
    histogram filters only that field.

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
        style = self.main_window.plot_style
        x, y = self.main_window.get_columns([self.field_x, self.field_y])
        self.figure.clear()
        style.style_figure(self.figure)
        self.span_x = self.span_y = None
        if len(x) == 0:
            axes = self.figure.add_subplot(111)
            style.style_axes(axes)
            style.label(
                axes, _axis_label(self.field_x), _axis_label(self.field_y)
            )
            self.canvas.draw()
            return
        bins_x = lib.calculate_optimal_bins(x, style.max_bins)
        bins_y = lib.calculate_optimal_bins(y, style.max_bins)
        nx = len(bins_x) - 1
        ny = len(bins_y) - 1
        x_min, x_max = float(bins_x[0]), float(bins_x[-1])
        y_min, y_max = float(bins_y[0]), float(bins_y[-1])
        counts = lib.hist2d(
            np.ascontiguousarray(x),
            np.ascontiguousarray(y),
            x_min,
            x_max,
            y_min,
            y_max,
            nx,
            ny,
        )
        # optional 1D histograms on top and right, colorbar on the right
        width_ratios = [5]
        height_ratios = [5]
        if style.show_marginals:
            width_ratios.append(1)
            height_ratios.insert(0, 1)
        if style.show_colorbar:
            width_ratios.append(0.12)
        grid = self.figure.add_gridspec(
            len(height_ratios),
            len(width_ratios),
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        axes = self.figure.add_subplot(grid[-1, 0])
        style.style_axes(axes)
        if style.show_marginals:
            axes_x = self.figure.add_subplot(grid[0, 0], sharex=axes)
            axes_y = self.figure.add_subplot(grid[-1, 1], sharey=axes)
            style.style_axes(axes_x, grid_axis="y")
            style.style_axes(axes_y, grid_axis="x")
            # marginals from the 2D counts so that both use identical bins
            style.bars(axes_x, counts.sum(axis=1), bins_x)
            style.bars(
                axes_y, counts.sum(axis=0), bins_y, orientation="horizontal"
            )
            axes_x.tick_params(labelbottom=False)
            axes_y.tick_params(labelleft=False)
            style.label(axes_x, None, "Counts")
            style.label(axes_y, "Counts", None)
        masked = np.ma.masked_equal(counts.T, 0)
        image = axes.pcolormesh(
            bins_x,
            bins_y,
            masked,
            norm=style.norm(),
            cmap=style.cmap,
            shading="flat",
            rasterized=True,
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
        if style.show_colorbar:
            colorbar = self.figure.colorbar(
                image, cax=self.figure.add_subplot(grid[-1, -1])
            )
            style.style_colorbar(colorbar)
        style.label(axes, _axis_label(self.field_x), _axis_label(self.field_y))
        style.title(
            self.figure,
            f"{self.field_x} vs. {self.field_y}  ·  "
            f"{int(counts.sum()):,} localizations",
        )
        self.selector = RectangleSelector(
            axes,
            self.on_rect_select,
            useblit=True,
            props=dict(fill=True, **style.selection),
        )
        if style.show_marginals:
            self.span_x = SpanSelector(
                axes_x,
                self.on_span_select_x,
                "horizontal",
                useblit=True,
                props=style.selection,
            )
            self.span_y = SpanSelector(
                axes_y,
                self.on_span_select_y,
                "vertical",
                useblit=True,
                props=style.selection,
            )
        self.canvas.draw()

    def on_span_select_x(self, xmin: float, xmax: float) -> None:
        """Apply the range selected on the x-field 1D histogram as a
        filter on the main window."""
        self.main_window.apply_range(self.field_x, float(xmin), float(xmax))

    def on_span_select_y(self, ymin: float, ymax: float) -> None:
        """Apply the range selected on the y-field 1D histogram as a
        filter on the main window."""
        self.main_window.apply_range(self.field_y, float(ymin), float(ymax))

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

    DOCS_URL = docs_url("filter.html")

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
        self.attributes.setToolTip("Column to filter by.")
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
        filter_button.setIcon(theme.icon("filter"))
        filter_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        filter_button.clicked.connect(self.filter)
        self.layout.addWidget(filter_button, 3, 0, 1, 2)

    def filter(self) -> None:
        """Filters locs given the range values."""
        if self.window.no_locs_warning("Filter"):
            return
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


EXPRESSION_TOOLTIP = (
    "Condition that the kept localizations satisfy, e.g.,\n"
    "  photons > 500 and 100 < sx < 160\n"
    "  lpx < 10    (lengths in nm)\n"
    "  photons > median(photons)\n\n"
    "Comparisons: < <= > >= == != (also chained: 100 < sx < 160)\n"
    "Logic: and, or, not\n"
    "Arithmetic: + - * / // % **\n"
    "Functions: abs, sqrt, log, log10, exp, floor, ceil, isfinite\n"
    "Statistics: mean, median, std, min, max, sum, percentile(values, q)\n"
    "Statistics are computed over the localizations currently shown,\n"
    "ignoring non-finite values. Press Enter to filter."
)


class FilterExpression(lib.Dialog):
    """Dialog for filtering localizations with a condition on their
    columns, see :mod:`picasso.expression`.

    ...

    Attributes
    ----------
    columns_label : QtWidgets.QLabel
        Names the condition may use.
    expression : QtWidgets.QLineEdit
        Condition to filter by.
    status : QtWidgets.QLabel
        Syntax errors while typing, or the result of the last filter.
    window : QtWidgets.QMainWindow
        Main window.

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        Main window.
    """

    DOCS_URL = docs_url("filter.html#filter-expression")

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Filter by expression")
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "filter.ico")
        self.setWindowIcon(QtGui.QIcon(icon_path))
        self.setMinimumWidth(480)

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(lib.HelpButton(self.DOCS_URL), 0, 0)
        label = QtWidgets.QLabel("Keep localizations where:")
        label.setToolTip(EXPRESSION_TOOLTIP)
        layout.addWidget(label, 0, 1)

        self.expression = QtWidgets.QLineEdit()
        self.expression.setPlaceholderText("photons > 500 and 100 < sx < 160")
        self.expression.setToolTip(EXPRESSION_TOOLTIP)
        self.expression.textChanged.connect(self.check)
        self.expression.returnPressed.connect(self.filter)
        layout.addWidget(self.expression, 1, 0, 1, 2)

        self.columns_label = QtWidgets.QLabel()
        self.columns_label.setWordWrap(True)
        self.columns_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.columns_label, 2, 0, 1, 2)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status, 3, 0, 1, 2)

        filter_button = QtWidgets.QPushButton("Filter")
        filter_button.setIcon(theme.icon("filter"))
        filter_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        filter_button.clicked.connect(self.filter)
        layout.addWidget(filter_button, 4, 0, 1, 2)

    def _available_names(self) -> list[str]:
        """Columns of the loaded file and metadata variables."""
        if not self.window.has_locs:
            return []
        variables = expression.variables_from_info(self.window.info)
        return list(self.window.locs_full.columns) + list(variables)

    def check(self) -> None:
        """Report syntax errors and unknown names while typing."""
        text = self.expression.text()
        if not text.strip() or not self.window.has_locs:
            self.status.clear()
            return
        try:
            unknown = sorted(
                expression.names(text) - set(self._available_names())
            )
        except expression.ExpressionError as e:
            self.status.setText(str(e))
            return
        if unknown:
            self.status.setText(f"Unknown name(s): {', '.join(unknown)}.")
        else:
            self.status.clear()

    def filter(self) -> None:
        """Filter the shown localizations by the condition."""
        if self.window.no_locs_warning("Filter by expression"):
            return
        n_before = self.window.n_filtered
        try:
            self.window.apply_expression(self.expression.text())
        except expression.ExpressionError as e:
            self.status.setText(str(e))
            return
        self.status.setText(
            f"Kept {self.window.n_filtered:,} of {n_before:,}"
            " localizations."
        )

    def on_locs_loaded(self) -> None:
        """List the names a condition may use for the loaded file."""
        self.columns_label.setText(
            "Columns: " + ", ".join(self._available_names())
        )
        self.check()


class _ShownColumns(Mapping):
    """Columns of the localizations shown in the Filter window, read
    on demand (see ``Window.get_column``)."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        self.window = window

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self:
            raise KeyError(name)
        return self.window.get_column(name)

    def __contains__(self, name: object) -> bool:
        return name in self.window.locs_full.columns

    def __iter__(self) -> Iterator[str]:
        return iter(self.window.locs_full.columns)

    def __len__(self) -> int:
        return len(self.window.locs_full.columns)


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
        test_button.setIcon(theme.icon("subclustering"))
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
            mols,
            self.window.info,
            clustering_dist=dist_clustered,
            sparse_dist=dist_sparse,
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
        self.plot_window = lib.GenericPlotWindow(
            "Test subclustering", "filter"
        )
        self.plot_window.resize(700, 500)

        def draw() -> None:
            style = self.plot_window.plot_style
            with self.plot_window.plot_context():
                lib.plot_subclustering_check(
                    clustered_nevents,
                    sparse_nevents,
                    return_fig=True,
                    clustering_dist=dist_clustered,
                    sparse_dist=dist_sparse,
                    fig=self.plot_window.figure,
                    fill=style.hist_fill,
                    outline=style.hist_outline,
                )

        draw()
        self.plot_window.redraw = draw
        self.plot_window.canvas.draw()
        self.plot_window.show()


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
    filter_expression : FilterExpression
        Filter dialog for conditions on the columns.
    filter_log : dict
        Dictionary of filter logs, i.e., data on what attribute/field
        was filtered and the corresponding min/max values, plus the
        conditions applied, in order, under ``lib.FILTER_STEPS_KEY``.
    filter_num : FilterNum
        Filter dialog for numeric values.
    hist_windows : dict
        Dictionary of histogram windows.
    hist2d_windows : dict
        Dictionary of 2D histogram windows.
    locs_full : pd.DataFrame
        Master localizations table, set on load. Moved to disk while
        the GUI is idle and read back on the next access, see
        ``release_idle_memory``.
    filtered_idx : np.ndarray
        Integer indices into ``locs_full`` of the currently-visible
        rows.
    pwd : str
        Current working directory.
    table_view : TableView
        Table view for displaying data.
    """

    DOCS_URL = docs_url("filter.html")

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
        self.filter_expression = FilterExpression(self)
        self.metadata_dialog = lib.MetadataDialog(self)
        self.user_settings_dialog = lib.UserSettingsDialog(self)
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open...")
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        save_action = file_menu.addAction("Save...")
        save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file_dialog)
        export_csv_action = file_menu.addAction("Export as CSV...")
        export_csv_action.triggered.connect(self.export_csv_dialog)
        metadata_action = file_menu.addAction("Show metadata...")
        metadata_action.setIcon(theme.icon("metadata"))
        metadata_action.setShortcut("Ctrl+M")
        metadata_action.triggered.connect(self.show_metadata)
        picasso_settings_action = file_menu.addAction("Picasso settings...")
        picasso_settings_action.setIcon(theme.icon("picasso-settings"))
        picasso_settings_action.triggered.connect(
            self.user_settings_dialog.show
        )
        theme.add_menu_action(file_menu)
        help_action = file_menu.addAction("Help")
        help_action.setIcon(theme.icon("help"))
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
        test_subcluster_action = plot_menu.addAction("Test subclustering...")
        test_subcluster_action.triggered.connect(self.plot_subclustering)
        plot_menu.addSeparator()
        plot_settings_action = plot_menu.addAction("Plot settings...")
        plot_settings_action.triggered.connect(self.show_plot_settings)

        filter_menu = menu_bar.addMenu("Filter")
        filter_action = filter_menu.addAction("Filter numerically...")
        filter_action.setShortcut("Ctrl+F")
        filter_action.triggered.connect(self.filter_num.show)
        filter_expression_action = filter_menu.addAction(
            "Filter by expression..."
        )
        filter_expression_action.setShortcut("Ctrl+E")
        filter_expression_action.triggered.connect(self.filter_expression.show)
        apply_from_metadata_action = filter_menu.addAction(
            "Apply filters from metadata..."
        )
        apply_from_metadata_action.triggered.connect(
            lambda: self.apply_filters_from_metadata()
        )
        remove_columns_action = filter_menu.addAction("Remove columns...")
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
        self.subcluster_num = None  # "Test subclustering" dialog
        self.filter_log = {}
        self._spilled = None  # io.SpilledFrame while moved to disk
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
        self.plot_style = plot_style.current()
        plot_style.hub().changed.connect(self.set_plot_style)

        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

        # toolbar of the most used actions, shared with the menus
        self.toolbar = toolbars.add_toolbar(
            self,
            "Filter toolbar",
            [
                (open_action, "open", "Open"),
                (save_action, "save", "Save"),
                (export_csv_action, "export-csv", "Export CSV"),
                None,
                (histogram_action, "histogram"),
                (scatter_action, "histogram-2d", "2D histogram"),
                (test_subcluster_action, "subclustering", "Subclustering"),
                None,
                (filter_action, "filter", "Filter"),
                (
                    filter_expression_action,
                    "filter-expression",
                    "Expression",
                ),
                (
                    apply_from_metadata_action,
                    "filter-metadata",
                    "From metadata",
                ),
                None,
                (plot_settings_action, "plot-settings", "Plot settings"),
            ],
        )
        # in the menus only, not on the toolbar
        remove_columns_action.setIcon(theme.icon("remove-columns"))

    @property
    def locs(self) -> pd.DataFrame:
        """Materialize the currently-filtered localizations.

        Provided for backward compatibility (e.g. plugins). This
        allocates a fresh DataFrame on every access — prefer
        ``get_column``, ``get_columns`` or ``materialize_filtered``.
        """
        return self.materialize_filtered()

    @property
    def locs_full(self) -> pd.DataFrame | None:
        """Master localizations table, None if no file is loaded. Read
        back from the disk first if it was moved there while the GUI
        was idle, see ``release_idle_memory``."""
        if self._spilled is not None:
            self._restore_locs()
        return self._locs_full

    @locs_full.setter
    def locs_full(self, locs: pd.DataFrame | None) -> None:
        self._discard_spill()
        self._locs_full = locs

    @property
    def has_locs(self) -> bool:
        """Whether a file is loaded, also while its localizations are
        moved to disk (which this does not read back)."""
        return self._locs_full is not None or self._spilled is not None

    def release_idle_memory(self) -> bool:
        """Move the localizations to disk while the GUI is idle (see
        ``picasso.gui.idle``), if they take at least
        ``idle.SPILL_MIN_BYTES``; the next access to ``locs_full`` reads
        them back. The rows shown in the table stay on screen, and the
        filters applied are kept. If the localizations cannot be moved,
        e.g., for lack of disk space, they stay in memory.

        Returns
        -------
        released : bool
            Always True: Filter is never busy.
        """
        locs = self._locs_full
        if locs is None:
            return True
        if locs.memory_usage(index=False).sum() < idle.SPILL_MIN_BYTES:
            return True
        try:
            spilled = io.SpilledFrame(locs)
        except io.SpillError as error:
            _log.warning("Localizations kept in memory: %s", error)
            return True
        # the table shows its own copy of the visible rows, as a slice
        # of the table would keep all of it in memory; replaced without
        # resetting the model, which would clear the selected columns
        self.table_model.locs = self.table_model.locs.copy()
        self._spilled = spilled
        self._locs_full = None
        return True

    def _restore_locs(self) -> None:
        """Read back the localizations moved to disk while idle.

        Raises
        ------
        io.SpillError
            If they cannot be read back; no file is loaded then, and
            the folder on the disk is kept.
        """
        spilled, self._spilled = self._spilled, None
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            self._locs_full = spilled.load()
        except io.SpillError as error:
            self._locs_full = None
            self.filtered_idx = None
            self.table_model.set_data(pd.DataFrame(), 0)
            raise io.SpillError(
                f"The localizations moved to disk while Filter was idle"
                f" could not be read back ({error}). Please open the file"
                f" again."
            ) from error
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        spilled.discard()

    def _discard_spill(self) -> None:
        """Delete the localizations moved to disk, if any, unread."""
        if self._spilled is not None:
            self._spilled.discard()
            self._spilled = None

    @property
    def n_filtered(self) -> int:
        """Number of currently-visible rows."""
        if not self.has_locs:
            return 0
        if self.filtered_idx is None:
            if self._spilled is not None:
                return self._spilled.n_rows
            return len(self._locs_full)
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
        self._keep(keep)
        self.log_filter(field, xmin, xmax)
        self._log_step(
            expression.range_expression(field, xmin, xmax, inclusive)
        )
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
        self._keep(keep)
        self.log_filter(field_x, xmin, xmax)
        self.log_filter(field_y, ymin, ymax)
        self._log_step(expression.range_expression(field_x, xmin, xmax))
        self._log_step(expression.range_expression(field_y, ymin, ymax))
        self.refresh()

    def apply_expression(self, condition: str) -> None:
        """Filter the active index to rows that satisfy a condition,
        e.g. ``"photons > median(photons)"``.

        Statistics in the condition are computed over the rows shown
        before this filter. ``pixelsize`` comes from the metadata.

        Raises
        ------
        expression.ExpressionError
            If the condition is invalid; nothing is filtered then.
        """
        keep = expression.condition(
            condition,
            _ShownColumns(self),
            expression.variables_from_info(self.info),
        )
        self._keep(keep)
        if (column_range := expression.parse_range(condition)) is not None:
            self.log_filter(*column_range)
        self._log_step(condition.strip())
        self.refresh()

    def _keep(self, keep: lib.BoolArray1D) -> None:
        """Narrow the active index to the shown rows where ``keep``."""
        if self.filtered_idx is None:
            self.filtered_idx = np.flatnonzero(keep).astype(
                self._idx_dtype(len(self.locs_full)), copy=False
            )
        else:
            self.filtered_idx = self.filtered_idx[keep]

    def _log_step(self, condition: str) -> None:
        """Record a filter condition, in the order applied."""
        self.filter_log.setdefault(lib.FILTER_STEPS_KEY, []).append(condition)

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
        if not self.has_locs:
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
            loaded = lib_qt.load_or_ask_pixelsize(self, io.load_filter, path)
        except lib.FILE_LOAD_ERRORS as error:
            lib.show_load_error(self, path, error)
            return
        if loaded is None:  # the pixel size prompt was canceled
            return
        locs, self.info = loaded
        # the windows of the previous file (not read back if on disk)
        windows = list(self.hist_windows.values())
        for by_y in self.hist2d_windows.values():
            windows.extend(by_y.values())
        for window in windows:
            if window:
                window.close()
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
        self.filter_log[lib.FILTER_STEPS_KEY] = []
        self.filter_num.on_locs_loaded()
        self.filter_expression.on_locs_loaded()
        self.refresh()

        self.setWindowTitle(
            f"Picasso v{__version__}: Filter. File: {os.path.basename(path)}"
        )
        self.pwd = os.path.dirname(path)

    def no_locs_warning(self, title: str) -> bool:
        """Tell the user that no localizations are loaded, if so.

        Parameters
        ----------
        title : str
            Title of the message box, i.e., the name of the action that
            needs localizations.

        Returns
        -------
        missing : bool
            True if no localizations are loaded (the action should
            stop), False otherwise.
        """
        if self.has_locs:
            return False
        QtWidgets.QMessageBox.information(self, title, "No file loaded.")
        return True

    def plot_histogram(self) -> None:
        if self.no_locs_warning("Histogram"):
            return
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) == 0:
            QtWidgets.QMessageBox.information(
                self,
                "Histogram",
                "Select one or more columns (click their headers) first.",
            )
        else:
            for index in indices:
                index = index.column()
                field = self.locs_full.columns[index]
                if not self.hist_windows[field]:
                    self.hist_windows[field] = HistWindow(self, field)
                self.hist_windows[field].show()

    def plot_hist2d(self) -> None:
        if self.no_locs_warning("2D Histogram"):
            return
        selection_model = self.table_view.selectionModel()
        indices = selection_model.selectedColumns()
        if len(indices) != 2:
            QtWidgets.QMessageBox.information(
                self,
                "2D Histogram",
                "Select two columns (click their headers) first.",
            )
        else:
            indices = [index.column() for index in indices]
            field_x, field_y = [
                self.locs_full.columns[index] for index in indices
            ]
            if not self.hist2d_windows[field_x][field_y]:
                self.hist2d_windows[field_x][field_y] = Hist2DWindow(
                    self, field_x, field_y
                )
            self.hist2d_windows[field_x][field_y].show()

    def show_plot_settings(self) -> None:
        """Show the dialog that sets the appearance of the chart
        windows."""
        plot_style.show_dialog(filter_options=True)

    def set_plot_style(self, style: PlotStyle) -> None:
        """Set the appearance of the histogram windows and redraw the
        open ones."""
        self.plot_style = style
        windows = list(self.hist_windows.values())
        for by_y in self.hist2d_windows.values():
            windows.extend(by_y.values())
        for window in windows:
            if window:
                window.refresh()

    def plot_subclustering(self) -> None:
        if self.no_locs_warning("Test subclustering"):
            return
        # one dialog, shown again: a dialog parented to the window lives
        # as long as the window, with the last plot it made
        if self.subcluster_num is None:
            self.subcluster_num = SubclusterNum(self)
        self.subcluster_num.show()
        self.subcluster_num.raise_()

    def display_locs(self, index: int) -> None:
        if not self.has_locs:
            return
        if self._spilled is not None and index == self.table_model.index:
            return  # e.g., resized while idle: the shown rows stay
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
        if self.no_locs_warning("Remove columns"):
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
        self.filter_num.on_locs_loaded()
        self.filter_expression.on_locs_loaded()

    def _build_filter_summary(self, steps, to_remove, missing) -> str:
        lines = []
        if steps:
            lines.append("Filters to apply, in this order:")
            for step in steps:
                lines.append(f"  {step}")
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

    def _load_filter_metadata(self, path: str | None = None):
        if path is None:
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
            return io.load_info(path)
        except lib.FILE_LOAD_ERRORS as error:
            lib.show_load_error(self, path, error)
            return None

    def apply_filters_from_metadata(self, path: str | None = None) -> None:
        """Replay filter steps recorded in another file's .yaml metadata
        onto the currently loaded localizations.

        Parameters
        ----------
        path : str, optional
            Path to the .yaml metadata, e.g., dropped into the window.
            If None, a file dialog asks for it.
        """
        if not self.has_locs:
            QtWidgets.QMessageBox.information(
                self, "Apply filters from metadata", "No file loaded."
            )
            return

        info = self._load_filter_metadata(path)
        if info is None:
            return

        title = "Apply filters from metadata"
        try:
            steps, to_remove, missing = lib.extract_filter_recipe(
                info, self.locs_full.columns
            )
        except expression.ExpressionError as e:
            QtWidgets.QMessageBox.warning(
                self, title, f"The metadata has an invalid filter:\n{e}"
            )
            return

        if not steps and not to_remove:
            msg = "No applicable filter steps found in metadata."
            if missing:
                msg += (
                    "\n\nReferenced columns not found in current data:\n  "
                    + "\n  ".join(missing)
                )
            QtWidgets.QMessageBox.information(self, title, msg)
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            title,
            self._build_filter_summary(steps, to_remove, missing),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        for step in steps:
            try:
                self.apply_expression(step)
            except expression.ExpressionError as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    title,
                    f"Could not apply\n  {step}\n{e}\n\n"
                    "The steps before it were applied; the rest were not.",
                )
                return
        if to_remove:
            self.locs_full.drop(columns=list(to_remove), inplace=True)
            if "Removed columns" in self.filter_log:
                self.filter_log["Removed columns"].extend(to_remove)
            else:
                self.filter_log["Removed columns"] = list(to_remove)
            self.filter_num.on_locs_loaded()
            self.filter_expression.on_locs_loaded()
            self.refresh()

    def export_csv_dialog(self) -> None:
        if self.no_locs_warning("Export as CSV"):
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
        if self.no_locs_warning("Save"):
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
                filter_info = {
                    key: value
                    for key, value in self.filter_log.items()
                    if key != lib.FILTER_STEPS_KEY or value
                }
                filter_info.update(lib.step_info("Filter"))
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
        if self.has_locs:
            settings["Filter"]["PWD"] = self.pwd
            io.save_user_settings(settings)
        self._discard_spill()
        QtWidgets.QApplication.instance().closeAllWindows()


def main() -> None:
    """Start Picasso: Filter - see ``picasso.gui.app.run_gui``."""
    sys.exit(run_gui(Window, "filter"))


if __name__ == "__main__":
    main()
