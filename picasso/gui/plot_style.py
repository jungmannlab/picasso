"""
picasso.gui.plot_style
~~~~~~~~~~~~~~~~~~~~~~

Appearance of Picasso's chart windows. One ``PlotStyle`` is shared by
the histogram windows of Picasso: Filter (1D and 2D), Filter's
subclustering test, every ``lib.GenericPlotWindow`` (e.g., Render's
drift, NeNA, FRC and trace plots, Nanotron's learning history) and
Simulate's previews, so that they always look the same.

The general settings apply to every chart: theme (background, grid and
text colors; by default light or dark like the windows, see
``picasso.gui.theme``), titles, font and tick label sizes, the data
color (of the first four plotted series, from one of ``PALETTES`` or set
one by one; further series continue the palette), the line width and
whether histograms are filled, filled with an outline or outlined only.
The remaining settings only concern Filter's histograms:
number of bins, count axis, selection color and, for 2D histograms,
colormap, color scale, side 1D histograms and colorbar. The dialog
shows them only when opened from Filter.

``current`` returns the style saved in the user settings
(``settings["PlotStyle"]``) and ``set_current`` saves a new one and
announces it through ``hub().changed``, to which the open windows
listen. ``show_dialog`` opens the one ``PlotStyleDialog`` that edits
it.

Windows that draw their own content apply the style by drawing inside
``PlotStyle.context()``, which sets matplotlib's defaults (theme, fonts,
default colors, line width); colors that carry a meaning, such as
channel colors, are passed explicitly and kept. ``PlotStyle.apply``
restyles a figure that has already been drawn.

:author: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, fields, replace

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
    to_hex,
)
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io, lib
from .overlay_style import PRESET_COLORS, ColorComboBox

#: Colors of each theme: figure and plot background, gridlines, axis
#: lines, text and secondary text (tick labels, colorbar label).
THEMES = {
    "Light": {
        "figure": "#fcfcfb",
        "axes": "#fcfcfb",
        "grid": "#e8e7e3",
        "spine": "#d6d5d0",
        "text": "#0b0b0b",
        "text_muted": "#52514e",
    },
    "Dark": {
        "figure": "#1a1a19",
        "axes": "#1a1a19",
        "grid": "#383835",
        "spine": "#4a4a46",
        "text": "#ffffff",
        "text_muted": "#c3c2b7",
    },
    # the look of Picasso: Filter before the plot settings existed
    "Classic": {
        "figure": "#ffffff",
        "axes": "#e5e5e5",
        "grid": "#ffffff",
        "spine": "#e5e5e5",
        "text": "#000000",
        "text_muted": "#555555",
    },
}

#: Theme that matches the windows: "Dark" on a dark window background
#: (see ``picasso.gui.theme``), "Light" otherwise.
SAME_AS_WINDOWS = "Same as windows"
#: Themes offered in the plot settings.
THEME_CHOICES = (SAME_AS_WINDOWS, *THEMES)

#: Single-hue blue ramp, light (sparse) to dark (dense).
PICASSO_BLUES = LinearSegmentedColormap.from_list(
    "Picasso blues",
    [
        "#b7d3f6",
        "#86b6ef",
        "#5598e7",
        "#2a78d6",
        "#1c5cab",
        "#104281",
        "#0d366b",
    ],
)
#: Color palettes of plotted series: the first ``N_COLORS`` can be set
#: one by one, further series continue the palette.
PALETTES = {
    "Picasso": [
        "#2A78D6",
        "#EB6834",
        "#1BAF7A",
        "#EDA100",
        "#E87BA4",
        "#008300",
        "#4A3AA7",
        "#E34948",
    ],
    "Colorblind-safe": [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#F0E442",
        "#999999",
    ],
    "Matplotlib": [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
    ],
    "ggplot": [
        "#E24A33",
        "#348ABD",
        "#988ED5",
        "#777777",
        "#FBC15E",
        "#8EBA42",
        "#FFB5B8",
    ],
}
#: Palette name of colors that were set one by one.
CUSTOM_PALETTE = "Custom"
#: Number of series colors that can be set one by one.
N_COLORS = 4
#: Colormaps offered for 2D histograms; all but the first are
#: matplotlib's.
COLORMAPS = [
    "Picasso blues",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
    "Greys",
    "hot",
    "turbo",
]
SCALES = ["Linear", "Log"]
#: How histogram bars are drawn, see ``lib.histogram_style``.
HIST_STYLES = ["Filled with outline", "Filled", "Outline"]
SELECTION_OPACITY = 0.25


def _color_code(color: str) -> str:
    """Hexadecimal code of a preset name of ``ColorComboBox`` or of
    any color matplotlib accepts."""
    return to_hex(PRESET_COLORS.get(color, color))


@dataclass(frozen=True)
class PlotStyle:
    """Appearance of the chart windows.

    Attributes
    ----------
    theme : str
        One of ``THEME_CHOICES``: ``SAME_AS_WINDOWS`` or a key of
        ``THEMES``.
    grid : bool
        Whether gridlines are drawn.
    font_size : int
        Size of the axis labels in points; the title is one point
        larger.
    tick_size : int
        Size of the tick labels in points.
    show_title : bool
        Whether titles are shown: in Filter's histogram windows the
        field name(s) and the number of localizations, elsewhere the
        titles of the plots.
    palette : str
        Key of ``PALETTES`` that ``colors`` were taken from, or
        ``CUSTOM_PALETTE``.
    colors : tuple of str
        Colors of the first ``N_COLORS`` plotted series, e.g., 1D
        histogram bars (first), the sparse population of the
        subclustering test (second) or NeNA's fit (second): preset
        names of ``ColorComboBox`` or hexadecimal codes. The first one
        matches the default colormap.
    use_channel_colors : bool
        Whether plots of several channels (e.g., Render's pick profile)
        color each channel as in the rendered image instead of with
        ``colors``.
    line_width : float
        Default width of plotted lines in points, also of histograms
        drawn as outlines.
    hist_style : str
        How histogram bars are drawn, one of ``HIST_STYLES``.
    colormap : str
        Colormap of 2D histograms, one of ``COLORMAPS``.
    reverse_colormap : bool
        Whether the colormap is reversed.
    color_scale : str
        "Log" or "Linear" mapping of 2D histogram counts to colors.
    show_marginals : bool
        Whether 2D histograms are flanked by the 1D histograms of both
        fields.
    show_colorbar : bool
        Whether 2D histograms have a colorbar.
    count_scale : str
        "Linear" or "Log" count axis of 1D histograms.
    max_bins : int
        Maximum number of bins per axis of Filter's histograms.
    selection_color : str
        Color of the span and rectangle used to select a range.
    """

    theme: str = SAME_AS_WINDOWS
    grid: bool = True
    font_size: int = 11
    tick_size: int = 9
    show_title: bool = True
    palette: str = "Picasso"
    colors: tuple = tuple(PALETTES["Picasso"][:N_COLORS])
    use_channel_colors: bool = True
    line_width: float = 1.5
    hist_style: str = "Filled with outline"
    colormap: str = "Picasso blues"
    reverse_colormap: bool = False
    color_scale: str = "Log"
    show_marginals: bool = True
    show_colorbar: bool = True
    count_scale: str = "Linear"
    max_bins: int = 1000
    selection_color: str = "#EB6834"

    @classmethod
    def from_settings(cls, settings: dict | None) -> PlotStyle:
        """Build the style from the user settings, ignoring unknown
        keys and invalid values.

        Parameters
        ----------
        settings : dict or None
            Saved ``PlotStyle`` fields, see ``to_settings``.

        Returns
        -------
        style : PlotStyle
            The style; missing or invalid fields take their default.
        """
        style = cls()
        if not isinstance(settings, dict):
            return style
        values = {}
        # data_color was the first color before colors existed
        if "colors" not in settings and "data_color" in settings:
            settings = dict(settings)
            settings["colors"] = [settings["data_color"]] + list(
                style.colors[1:]
            )
            settings.setdefault("palette", CUSTOM_PALETTE)
        for field in fields(cls):
            if field.name not in settings:
                continue
            value = settings[field.name]
            default = getattr(style, field.name)
            if isinstance(default, bool):
                if isinstance(value, bool):
                    values[field.name] = value
            elif isinstance(default, float):
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    values[field.name] = float(value)
            elif isinstance(default, int):
                if isinstance(value, int) and not isinstance(value, bool):
                    values[field.name] = value
            elif isinstance(default, tuple):
                if isinstance(value, (list, tuple)):
                    values[field.name] = tuple(value)
            elif isinstance(value, str):
                values[field.name] = value
        style = replace(style, **values)
        # replace choices that are no longer offered by their default
        default = cls()
        for name, options in (
            ("theme", THEME_CHOICES),
            ("colormap", COLORMAPS),
            ("color_scale", SCALES),
            ("count_scale", SCALES),
            ("hist_style", HIST_STYLES),
        ):
            if getattr(style, name) not in options:
                style = replace(style, **{name: getattr(default, name)})
        if style.palette not in PALETTES and style.palette != CUSTOM_PALETTE:
            style = replace(style, palette=default.palette)
        try:
            _color_code(style.selection_color)
        except ValueError:
            style = replace(style, selection_color=default.selection_color)
        # exactly N_COLORS valid colors, missing or invalid ones from
        # the palette
        fallback = PALETTES.get(style.palette, PALETTES["Picasso"])
        colors = []
        for i in range(N_COLORS):
            color = style.colors[i] if i < len(style.colors) else None
            try:
                _color_code(color)
            except (ValueError, TypeError):
                color = fallback[i]
            colors.append(str(color))
        return replace(style, colors=tuple(colors))

    def to_settings(self) -> dict:
        """The style as a plain dict for the user settings."""
        settings = asdict(self)
        settings["colors"] = list(self.colors)
        return settings

    @property
    def theme_colors(self) -> dict:
        """Colors of the theme, see ``THEMES``; for ``SAME_AS_WINDOWS``
        those of the windows' current theme."""
        if self.theme == SAME_AS_WINDOWS:
            return THEMES["Dark" if windows_are_dark() else "Light"]
        return THEMES[self.theme]

    @property
    def cmap(self) -> Colormap:
        """The colormap of 2D histograms."""
        if self.colormap == PICASSO_BLUES.name:
            cmap = PICASSO_BLUES
        else:
            cmap = matplotlib.colormaps[self.colormap]
        return cmap.reversed() if self.reverse_colormap else cmap

    def norm(self) -> Normalize:
        """A new normalization of 2D histogram counts (zero-count bins
        are masked and not drawn)."""
        if self.color_scale == "Log":
            return LogNorm()
        return Normalize(vmin=1)

    @property
    def series_colors(self) -> list[str]:
        """Hexadecimal codes of the default colors of plotted series:
        ``colors``, then the rest of the palette (of "Picasso" if the
        colors are custom) except colors already used."""
        first = [_color_code(c) for c in self.colors]
        palette = PALETTES.get(self.palette, PALETTES["Picasso"])
        rest = [
            _color_code(c)
            for c in palette[N_COLORS:]
            if _color_code(c) not in first
        ]
        return first + rest

    @property
    def primary_color(self) -> str:
        """Hexadecimal code of the first series color."""
        return self.series_colors[0]

    def channel_colors(self, colors: list) -> list:
        """Colors of the channels of a multi-channel plot.

        Parameters
        ----------
        colors : list
            Colors of the channels in the rendered image.

        Returns
        -------
        colors : list
            ``colors`` if ``use_channel_colors``, otherwise the series
            colors (repeated if there are more channels).
        """
        if self.use_channel_colors:
            return list(colors)
        series = self.series_colors
        return [series[i % len(series)] for i in range(len(colors))]

    @property
    def hist_fill(self) -> bool:
        """Whether histogram bars are filled."""
        return self.hist_style != "Outline"

    @property
    def hist_outline(self) -> bool:
        """Whether histogram bars are outlined."""
        return self.hist_style != "Filled"

    def hist_kwargs(
        self, color: str | None = None, fill_alpha: float | None = None
    ) -> dict:
        """Matplotlib properties of histogram bars in ``hist_style``,
        see ``lib.histogram_style``.

        Parameters
        ----------
        color : str, optional
            Color of the bars. If None, the data color. Default None.
        fill_alpha : float, optional
            Opacity of the fill, e.g., for overlapping histograms. If
            None, the default of ``lib.histogram_style``. Default None.

        Returns
        -------
        kwargs : dict
            ``fill``, ``facecolor``, ``edgecolor`` and ``linewidth``.
        """
        return lib.histogram_style(
            self.primary_color if color is None else color,
            fill=self.hist_fill,
            outline=self.hist_outline,
            fill_alpha=fill_alpha,
            line_width=self.line_width,
        )

    @property
    def selection(self) -> dict:
        """Matplotlib properties of the range selectors."""
        color = _color_code(self.selection_color)
        return dict(facecolor=color, edgecolor=color, alpha=SELECTION_OPACITY)

    def rc_params(self) -> dict:
        """The style as matplotlib rcParams, see ``context``."""
        colors = self.theme_colors
        classic = self.theme == "Classic"
        return {
            "figure.facecolor": colors["figure"],
            "savefig.facecolor": colors["figure"],
            "figure.titlesize": self.font_size + 1,
            "axes.facecolor": colors["axes"],
            "axes.edgecolor": colors["spine"],
            "axes.labelcolor": colors["text"],
            "axes.titlecolor": colors["text"],
            "axes.labelsize": self.font_size,
            "axes.titlesize": self.font_size + 1,
            "axes.spines.top": classic,
            "axes.spines.right": classic,
            "axes.grid": self.grid,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=self.series_colors),
            "grid.color": colors["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 1.0 if classic else 0.8,
            "xtick.color": colors["text_muted"],
            "ytick.color": colors["text_muted"],
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "text.color": colors["text"],
            "legend.facecolor": colors["axes"],
            "legend.edgecolor": colors["spine"],
            "legend.labelcolor": colors["text"],
            "legend.fontsize": self.tick_size,
            "lines.linewidth": self.line_width,
        }

    def context(self) -> AbstractContextManager:
        """Context in which matplotlib draws with this style by default:
        figures, axes and colorbars created inside it take the theme,
        sizes and grid, data drawn without an explicit color takes
        ``series_colors`` and lines without an explicit width take the
        line width. The background of figures created before entering
        the context is set by ``style_figure``."""
        return matplotlib.rc_context(self.rc_params())

    def apply(self, figure: Figure) -> None:
        """Restyle a figure that has already been drawn: background,
        grid, axis lines, text and legends. Colors of the data are kept
        (redraw inside ``context`` to change them).

        Parameters
        ----------
        figure : Figure
            The figure.
        """
        colors = self.theme_colors
        self.style_figure(figure)
        if figure._suptitle is not None:
            figure._suptitle.set_color(colors["text"])
            figure._suptitle.set_fontsize(self.font_size + 1)
            figure._suptitle.set_visible(self.show_title)
        for axes in figure.axes:
            colorbar = getattr(axes, "_colorbar", None)
            if colorbar is not None:
                self.style_colorbar(colorbar, label=None)
                continue
            self.style_axes(axes)
            for title in (axes.title, axes._left_title, axes._right_title):
                title.set_visible(self.show_title)
            axes.title.set_color(colors["text"])
            axes.title.set_fontsize(self.font_size + 1)
            for label in (axes.xaxis.label, axes.yaxis.label):
                label.set_color(colors["text"])
                label.set_fontsize(self.font_size)
            legend = axes.get_legend()
            if legend is not None:
                frame = legend.get_frame()
                frame.set_facecolor(colors["axes"])
                frame.set_edgecolor(colors["spine"])
                for text in legend.get_texts():
                    text.set_color(colors["text"])
                    text.set_fontsize(self.tick_size)

    def style_figure(self, figure: Figure) -> None:
        """Set the background of ``figure``."""
        figure.set_facecolor(self.theme_colors["figure"])

    def style_axes(self, axes: Axes, grid_axis: str = "both") -> None:
        """Apply the theme, tick label size and grid to ``axes``.

        Parameters
        ----------
        axes : Axes
            The axes.
        grid_axis : {"both", "x", "y"}, optional
            Axis whose gridlines are drawn if the grid is on. Default
            "both".
        """
        colors = self.theme_colors
        axes.set_facecolor(colors["axes"])
        classic = self.theme == "Classic"
        for side, spine in axes.spines.items():
            spine.set_color(colors["spine"])
            spine.set_visible(classic or side in ("left", "bottom"))
        axes.tick_params(
            which="both",
            colors=colors["text_muted"],
            labelsize=self.tick_size,
            length=0,
        )
        axes.grid(False)
        if self.grid:
            axes.grid(
                True,
                axis=grid_axis,
                color=colors["grid"],
                linewidth=1.0 if classic else 0.8,
                linestyle="-",
            )
        axes.set_axisbelow(True)

    def label(self, axes: Axes, x: str | None, y: str | None) -> None:
        """Label the x- and/or y-axis of ``axes``; the "Counts" axis of
        a 1D histogram is labeled in the secondary text color."""
        for text, setter in ((x, axes.set_xlabel), (y, axes.set_ylabel)):
            if text is None:
                continue
            muted = text == "Counts"
            color = self.theme_colors["text_muted" if muted else "text"]
            setter(text, color=color, fontsize=self.font_size)

    def title(self, figure: Figure, text: str) -> None:
        """Set the title of ``figure`` if titles are shown."""
        if self.show_title:
            figure.suptitle(
                text,
                color=self.theme_colors["text"],
                fontsize=self.font_size + 1,
            )

    def bars(
        self,
        axes: Axes,
        counts: np.ndarray,
        bins: np.ndarray,
        orientation: str = "vertical",
    ) -> None:
        """Draw a 1D histogram and set its count scale.

        Parameters
        ----------
        axes : Axes
            The axes.
        counts : np.ndarray
            Counts per bin.
        bins : np.ndarray
            Bin edges, one more than ``counts``.
        orientation : {"vertical", "horizontal"}, optional
            "horizontal" draws the bars along the y-axis. Default
            "vertical".
        """
        log = self.count_scale == "Log"
        # a log axis cannot show the zero baseline, start just below 1
        baseline = 0.5 if log else 0
        axes.stairs(
            counts,
            bins,
            orientation=orientation,
            baseline=baseline,
            **self.hist_kwargs(),
        )
        if orientation == "vertical":
            if log:
                axes.set_yscale("log")
            axes.set_ylim(bottom=baseline)
        else:
            if log:
                axes.set_xscale("log")
            axes.set_xlim(left=baseline)

    def style_colorbar(
        self, colorbar, label: str | None = "Localizations per bin"
    ) -> None:
        """Apply the theme and tick label size to a colorbar and,
        unless ``label`` is None, label it."""
        colors = self.theme_colors
        colorbar.ax.set_facecolor(colors["axes"])
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(
            which="both",
            colors=colors["text_muted"],
            labelsize=self.tick_size,
            length=0,
        )
        if label is not None:
            colorbar.set_label(label)
        colorbar.ax.yaxis.label.set_color(colors["text_muted"])
        colorbar.ax.yaxis.label.set_fontsize(self.font_size)


def windows_are_dark() -> bool:
    """Whether the windows of the running application have a dark
    background, in any theme of ``picasso.gui.theme`` (also "Native")."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    window = app.palette().color(QtGui.QPalette.ColorRole.Window)
    return window.lightness() < 128


def current() -> PlotStyle:
    """The style saved in the user settings."""
    return PlotStyle.from_settings(io.load_user_settings()["PlotStyle"])


def set_current(style: PlotStyle) -> None:
    """Save ``style`` in the user settings and announce it to the open
    windows through ``hub().changed``."""
    settings = io.load_user_settings()
    settings["PlotStyle"] = style.to_settings()
    io.save_user_settings(settings)
    hub().changed.emit(style)


class _Hub(QtCore.QObject):
    """Announces style changes to the open windows."""

    changed = QtCore.pyqtSignal(object)


_hub = None
_dialog = None
# the hub of picasso.gui.theme whose changes restyle the charts
_windows_hub = None


def hub() -> _Hub:
    """The object whose ``changed`` signal carries each new style."""
    global _hub, _windows_hub
    if _hub is None:
        _hub = _Hub()
    # imported here: picasso.gui.theme is not needed before the first
    # chart window
    from . import theme

    if _windows_hub is not theme.hub():
        _windows_hub = theme.hub()
        _windows_hub.changed.connect(_on_windows_theme_changed)
    return _hub


def _on_windows_theme_changed(_appearance) -> None:
    """Restyle the charts that follow the windows' theme."""
    style = current()
    if style.theme == SAME_AS_WINDOWS:
        hub().changed.emit(style)


def show_dialog(filter_options: bool = False) -> PlotStyleDialog:
    """Show the plot settings dialog, creating it on first use. One
    dialog serves all windows; it opens with the saved style.

    Parameters
    ----------
    filter_options : bool, optional
        Whether to show the settings of Filter's histograms besides the
        general ones. Default False.

    Returns
    -------
    dialog : PlotStyleDialog
        The dialog.
    """
    global _dialog
    if _dialog is None:
        _dialog = PlotStyleDialog(current())
        # an open dialog must not keep the app running after the last
        # main window closed
        _dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        _dialog.styleChanged.connect(set_current)
    elif not _dialog.isVisible():
        _dialog.set_style(current(), notify=False)
    _dialog.set_filter_options_visible(filter_options)
    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()
    return _dialog


class PlotStyleDialog(lib.Dialog):
    """Edit the shared ``PlotStyle`` of the chart windows.

    The general settings apply to every chart; the settings of Filter's
    histograms are shown only if requested, see
    ``set_filter_options_visible``. Changes are applied immediately;
    ``styleChanged`` is emitted shortly after the last change so that
    dragging a spin box does not redraw the plots for every step.

    Parameters
    ----------
    style : PlotStyle
        The style shown initially.
    parent : QWidget or None, optional
        Parent widget. Default None.
    """

    styleChanged = QtCore.pyqtSignal(object)

    #: Delay between the last change and ``styleChanged`` (ms).
    DELAY = 150

    def __init__(
        self,
        style: PlotStyle,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot settings")
        self._setting_colors = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.DELAY)
        self._timer.timeout.connect(
            lambda: self.styleChanged.emit(self.style())
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

        general = QtWidgets.QGroupBox("All plots")
        form = QtWidgets.QFormLayout(general)
        self.theme = QtWidgets.QComboBox()
        self.theme.addItems(THEME_CHOICES)
        self.theme.setToolTip(
            "Same as windows: light or dark like the Picasso windows "
            "(File > Appearance).\n"
            "Classic is the look of Filter's histograms in earlier versions."
        )
        form.addRow("Theme:", self.theme)
        self.palette = QtWidgets.QComboBox()
        self.palette.addItems(list(PALETTES) + [CUSTOM_PALETTE])
        self.palette.setToolTip(
            "Fills the colors below; further plotted series continue "
            "the palette."
        )
        form.addRow("Palette:", self.palette)
        self.colors = []
        colors_grid = QtWidgets.QGridLayout()
        colors_grid.setContentsMargins(0, 0, 0, 0)
        for i in range(N_COLORS):
            color = ColorComboBox()
            self.colors.append(color)
            row, column = i // 2, 2 * (i % 2)
            colors_grid.addWidget(QtWidgets.QLabel(f"{i + 1}:"), row, column)
            colors_grid.addWidget(color, row, column + 1)
        colors_widget = QtWidgets.QWidget()
        colors_widget.setLayout(colors_grid)
        colors_widget.setToolTip(
            "Colors of the plotted series, in order. For example:\n"
            "1D histograms and NeNA data: 1; NeNA fit: 2\n"
            "Subclustering test: clustered 1, sparse 2\n"
            "Drift: x 1, y 2, x-y path 3, z 4"
        )
        form.addRow("Colors:", colors_widget)
        self.use_channel_colors = QtWidgets.QCheckBox("Use channel colors")
        self.use_channel_colors.setToolTip(
            "Plots of several channels (e.g., Render's pick profile) color "
            "each channel as in the rendered image. Off: they use the "
            "colors above."
        )
        form.addRow(self.use_channel_colors)
        self.line_width = QtWidgets.QDoubleSpinBox()
        self.line_width.setRange(0.25, 10)
        self.line_width.setSingleStep(0.25)
        self.line_width.setSuffix(" pt")
        self.line_width.setToolTip(
            "Width of plotted curves that do not set their own."
        )
        form.addRow("Line width:", self.line_width)
        self.hist_style = QtWidgets.QComboBox()
        self.hist_style.addItems(HIST_STYLES)
        self.hist_style.setToolTip(
            "How histogram bars are drawn; outlines take the line width."
        )
        form.addRow("Histograms:", self.hist_style)
        self.font_size = QtWidgets.QSpinBox()
        self.font_size.setRange(6, 32)
        self.font_size.setSuffix(" pt")
        form.addRow("Axis label size:", self.font_size)
        self.tick_size = QtWidgets.QSpinBox()
        self.tick_size.setRange(5, 32)
        self.tick_size.setSuffix(" pt")
        form.addRow("Tick label size:", self.tick_size)
        self.grid = QtWidgets.QCheckBox("Show grid")
        form.addRow(self.grid)
        self.show_title = QtWidgets.QCheckBox("Show titles")
        form.addRow(self.show_title)
        layout.addWidget(general)

        hist = QtWidgets.QGroupBox("Filter histograms")
        form = QtWidgets.QFormLayout(hist)
        self.max_bins = QtWidgets.QSpinBox()
        self.max_bins.setRange(10, 10_000)
        self.max_bins.setSingleStep(100)
        self.max_bins.setToolTip(
            "Upper limit of the number of bins per axis; fewer are used "
            "if the data is sparse or integer-valued."
        )
        form.addRow("Max. bins:", self.max_bins)
        self.count_scale = QtWidgets.QComboBox()
        self.count_scale.addItems(SCALES)
        form.addRow("Count axis:", self.count_scale)
        self.selection_color = ColorComboBox()
        self.selection_color.setToolTip(
            "Color of the range selected for filtering."
        )
        form.addRow("Selection color:", self.selection_color)
        layout.addWidget(hist)

        hist2d = QtWidgets.QGroupBox("Filter 2D histograms")
        form = QtWidgets.QFormLayout(hist2d)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(COLORMAPS)
        form.addRow("Colormap:", self.colormap)
        self.reverse_colormap = QtWidgets.QCheckBox("Reverse colormap")
        form.addRow(self.reverse_colormap)
        self.color_scale = QtWidgets.QComboBox()
        self.color_scale.addItems(SCALES)
        form.addRow("Color scale:", self.color_scale)
        self.show_marginals = QtWidgets.QCheckBox("Show 1D histograms")
        self.show_marginals.setToolTip(
            "Distributions of both fields above and to the right of the "
            "2D histogram."
        )
        form.addRow(self.show_marginals)
        self.show_colorbar = QtWidgets.QCheckBox("Show colorbar")
        form.addRow(self.show_colorbar)
        layout.addWidget(hist2d)
        self.filter_groups = (hist, hist2d)

        buttons = QtWidgets.QDialogButtonBox()
        reset = buttons.addButton(
            "Restore defaults",
            QtWidgets.QDialogButtonBox.ButtonRole.ResetRole,
        )
        reset.setToolTip("Restores all settings, also hidden ones.")
        reset.clicked.connect(
            lambda: lib.confirm_restore_defaults(
                self, "all plot settings, also hidden ones,"
            )
            and self.set_style(PlotStyle())
        )
        close = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        close.clicked.connect(self.close)
        layout.addWidget(buttons)

        self.set_style(style)
        for box in (
            self.theme,
            self.hist_style,
            self.count_scale,
            self.colormap,
            self.color_scale,
        ):
            box.currentIndexChanged.connect(self._timer.start)
        for check in (
            self.grid,
            self.show_title,
            self.reverse_colormap,
            self.show_marginals,
            self.show_colorbar,
        ):
            check.toggled.connect(self._timer.start)
        for spin in (
            self.font_size,
            self.tick_size,
            self.max_bins,
            self.line_width,
        ):
            spin.valueChanged.connect(self._timer.start)
        self.selection_color.colorChanged.connect(self._timer.start)
        for color in self.colors:
            color.colorChanged.connect(self._on_color_changed)
        self.palette.currentIndexChanged.connect(self._on_palette_changed)
        self.use_channel_colors.toggled.connect(self._timer.start)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # apply a change still waiting for the delay now, so that the
        # timer does not fire after the dialog is gone
        if self._timer.isActive():
            self._timer.stop()
            self.styleChanged.emit(self.style())
        super().closeEvent(event)

    def _set_colors(self, colors) -> None:
        """Show ``colors`` without marking the palette as custom."""
        # not by blocking signals: a ColorComboBox tracks its value
        # through them
        self._setting_colors = True
        try:
            for i, (box, color) in enumerate(zip(self.colors, colors)):
                if not box.set_value(color):
                    box.set_value(PlotStyle.colors[i])
        finally:
            self._setting_colors = False
        self._timer.start()

    def _set_palette(self, palette: str) -> None:
        """Select ``palette`` without changing the colors."""
        self.palette.blockSignals(True)
        self.palette.setCurrentText(palette)
        self.palette.blockSignals(False)

    def _on_palette_changed(self) -> None:
        palette = self.palette.currentText()
        if palette in PALETTES:
            self._set_colors(PALETTES[palette][:N_COLORS])
        self._timer.start()

    def _on_color_changed(self) -> None:
        if self._setting_colors:
            return
        # a color set by hand no longer follows a palette
        palette = PALETTES.get(self.palette.currentText())
        values = [_color_code(box.value()) for box in self.colors]
        if palette is None or values != [
            _color_code(c) for c in palette[:N_COLORS]
        ]:
            self._set_palette(CUSTOM_PALETTE)
        self._timer.start()

    def set_filter_options_visible(self, visible: bool) -> None:
        """Show or hide the settings of Filter's histograms."""
        for group in self.filter_groups:
            group.setVisible(visible)

    def style(self) -> PlotStyle:
        """The style set in the dialog."""
        return PlotStyle(
            theme=self.theme.currentText(),
            grid=self.grid.isChecked(),
            font_size=self.font_size.value(),
            tick_size=self.tick_size.value(),
            show_title=self.show_title.isChecked(),
            palette=self.palette.currentText(),
            colors=tuple(color.value() for color in self.colors),
            use_channel_colors=self.use_channel_colors.isChecked(),
            line_width=self.line_width.value(),
            hist_style=self.hist_style.currentText(),
            colormap=self.colormap.currentText(),
            reverse_colormap=self.reverse_colormap.isChecked(),
            color_scale=self.color_scale.currentText(),
            show_marginals=self.show_marginals.isChecked(),
            show_colorbar=self.show_colorbar.isChecked(),
            count_scale=self.count_scale.currentText(),
            max_bins=self.max_bins.value(),
            selection_color=self.selection_color.value(),
        )

    def set_style(self, style: PlotStyle, notify: bool = True) -> None:
        """Show ``style`` in the dialog.

        Parameters
        ----------
        style : PlotStyle
            The style.
        notify : bool, optional
            Whether to emit ``styleChanged`` (once, after the delay) if
            ``style`` differs from the shown one. Default True.
        """
        self.theme.setCurrentText(style.theme)
        self.grid.setChecked(style.grid)
        self.font_size.setValue(style.font_size)
        self.tick_size.setValue(style.tick_size)
        self.show_title.setChecked(style.show_title)
        self._set_colors(style.colors)
        self._set_palette(style.palette)
        self.use_channel_colors.setChecked(style.use_channel_colors)
        self.line_width.setValue(style.line_width)
        self.hist_style.setCurrentText(style.hist_style)
        self.colormap.setCurrentText(style.colormap)
        self.reverse_colormap.setChecked(style.reverse_colormap)
        self.color_scale.setCurrentText(style.color_scale)
        self.show_marginals.setChecked(style.show_marginals)
        self.show_colorbar.setChecked(style.show_colorbar)
        self.count_scale.setCurrentText(style.count_scale)
        self.max_bins.setValue(style.max_bins)
        if not self.selection_color.set_value(style.selection_color):
            self.selection_color.set_value(PlotStyle.selection_color)
        if not notify:
            self._timer.stop()
