"""The shared appearance of Picasso's chart windows.

One ``PlotStyle`` (``picasso.gui.plot_style``) is shared by the 1D and
2D histogram windows of ``picasso.gui.filter``, Filter's subclustering
test and every ``lib.GenericPlotWindow``. These tests cover reading it
from the user settings (including invalid values), matplotlib's
defaults inside ``PlotStyle.context``, restyling a drawn figure, the
one plot settings dialog, the live update of open windows, the
optional 1D histograms and colorbar of the 2D window and filtering from
the 1D histograms of the 2D window.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import StepPatch

from picasso import io, lib
from picasso.gui import filter as gui_filter
from picasso.gui import plot_style
from picasso.gui.plot_style import PALETTES, THEMES, PlotStyle

#: The second default series color.
SECOND = to_hex(PALETTES["Picasso"][1])


@pytest.fixture(autouse=True)
def fresh_hub(monkeypatch):
    """Each test gets its own hub and dialog, so windows of earlier
    tests do not receive its style changes."""
    monkeypatch.setattr(plot_style, "_hub", None)
    monkeypatch.setattr(plot_style, "_dialog", None)
    yield
    if plot_style._dialog is not None:
        plot_style._dialog.close()


@pytest.fixture
def window(qt_offscreen):
    """A Filter window holding random photons and sx values."""
    window = gui_filter.Window()
    rng = np.random.default_rng(0)
    n = 5000
    window.locs_full = pd.DataFrame(
        {
            "photons": rng.lognormal(7, 0.5, n).astype(np.float32),
            "sx": rng.normal(1, 0.1, n).astype(np.float32),
        }
    )
    for column in window.locs_full.columns:
        window.filter_log[column] = None
        window.hist_windows[column] = None
        window.hist2d_windows[column] = {
            column_y: None for column_y in window.locs_full.columns
        }
    yield window
    for w in window.hist_windows.values():
        if w:
            w.close()
    for by_y in window.hist2d_windows.values():
        for w in by_y.values():
            if w:
                w.close()


def _open(window):
    hist = gui_filter.HistWindow(window, "photons")
    hist2d = gui_filter.Hist2DWindow(window, "photons", "sx")
    window.hist_windows["photons"] = hist
    window.hist2d_windows["photons"]["sx"] = hist2d
    return hist, hist2d


def test_from_settings_round_trip():
    style = replace(
        PlotStyle(),
        theme="Dark",
        grid=False,
        colormap="magma",
        palette="Custom",
        colors=("#123456", "Red", "#00FF00", "Blue"),
        use_channel_colors=False,
        line_width=2.5,
        max_bins=200,
    )
    assert PlotStyle.from_settings(style.to_settings()) == style


@pytest.mark.parametrize(
    "settings",
    [
        None,
        {},
        {"theme": "Neon", "colormap": "nope", "count_scale": "Cubic"},
        {"grid": 1, "max_bins": "many", "font_size": True},
        {"line_width": "thick", "colors": "red", "palette": "Neon"},
        {"colors": ["nope", 3, None, ""]},
        {"unknown": 3},
    ],
)
def test_from_settings_ignores_invalid(settings):
    assert PlotStyle.from_settings(settings) == PlotStyle()


def test_series_colors():
    style = PlotStyle()
    # the first default color matches the middle of the default colormap
    assert style.primary_color == to_hex("#2A78D6")
    assert replace(style, colormap="magma").primary_color == (
        style.primary_color
    )
    custom = replace(
        style, palette="Custom", colors=("Red",) + style.colors[1:]
    )
    assert custom.series_colors[0] == to_hex("red")
    assert custom.series_colors[1] == SECOND
    # further series continue the palette, without repeating a color
    picasso = [to_hex(c) for c in PALETTES["Picasso"]]
    assert style.series_colors == picasso
    repeated = replace(style, colors=(picasso[5],) + style.colors[1:])
    assert len(set(repeated.series_colors)) == len(repeated.series_colors)
    ggplot = replace(
        style, palette="ggplot", colors=tuple(PALETTES["ggplot"][:4])
    )
    assert ggplot.series_colors == [to_hex(c) for c in PALETTES["ggplot"]]


def test_settings_of_too_few_colors_are_completed():
    style = PlotStyle.from_settings({"colors": ["Red"]})
    assert style.colors == ("Red",) + PlotStyle().colors[1:]
    # the single color of earlier versions becomes the first one
    old = PlotStyle.from_settings({"data_color": "#123456"})
    assert old.colors[0] == "#123456" and old.palette == "Custom"


def test_channel_colors():
    channels = ["#ff0000", "#00ff00", "#0000ff"]
    assert PlotStyle().channel_colors(channels) == channels
    own = replace(PlotStyle(), use_channel_colors=False)
    assert own.channel_colors(channels) == own.series_colors[:3]


@pytest.mark.parametrize("reverse", [False, True])
def test_context_sets_defaults(reverse):
    style = replace(
        PlotStyle(), theme="Dark", reverse_colormap=reverse, line_width=3.0
    )
    with style.context():
        figure = Figure()
        axes = figure.add_subplot()
        first, second = axes.plot([0, 1]), axes.plot([1, 0])
        image = axes.imshow(np.eye(3))
    assert to_hex(figure.get_facecolor()) == THEMES["Dark"]["figure"]
    assert to_hex(axes.get_facecolor()) == THEMES["Dark"]["axes"]
    assert first[0].get_color() == style.primary_color
    assert second[0].get_color() == SECOND
    assert first[0].get_linewidth() == 3.0
    # the colormap is Filter's only, images keep matplotlib's default
    assert image.get_cmap().name != style.cmap.name


def test_apply_restyles_drawn_figure():
    figure = Figure()
    axes = figure.add_subplot()
    (line,) = axes.plot([0, 1], color="red")
    axes.legend(["data"])
    colorbar = figure.colorbar(axes.imshow(np.eye(3)), ax=axes)
    colorbar.set_label("Value")
    style = replace(PlotStyle(), theme="Dark", font_size=15)
    style.apply(figure)
    assert to_hex(axes.get_facecolor()) == THEMES["Dark"]["axes"]
    assert axes.xaxis.label.get_fontsize() == 15
    legend_text = axes.get_legend().get_texts()[0]
    assert to_hex(legend_text.get_color()) == THEMES["Dark"]["text"]
    # data colors and the colorbar label are kept
    assert line.get_color() == "red"
    assert colorbar.ax.get_ylabel() == "Value"


def test_current_round_trip(qt_offscreen):
    assert plot_style.current() == PlotStyle()
    style = replace(PlotStyle(), colormap="cividis", font_size=14)
    received = []
    plot_style.hub().changed.connect(received.append)
    plot_style.set_current(style)
    assert received == [style]
    assert plot_style.current() == style
    settings = io.load_user_settings()
    assert PlotStyle.from_settings(settings["PlotStyle"]) == style


def test_dialog_updates_filter(window):
    hist, _ = _open(window)
    dialog = plot_style.show_dialog()
    assert dialog is plot_style.show_dialog()  # one dialog
    style = replace(
        PlotStyle(),
        theme="Classic",
        reverse_colormap=True,
        color_scale="Linear",
        palette="Custom",
        colors=("#123456",) + PlotStyle().colors[1:],
        selection_color="Lime",
        tick_size=12,
    )
    dialog.set_style(style)
    assert dialog.style() == style
    # the delayed signal saves the style and redraws the open windows
    dialog._timer.timeout.emit()
    assert plot_style.current() == style
    assert window.plot_style == style
    bars = [p for p in hist.figure.axes[0].patches if isinstance(p, StepPatch)]
    assert to_hex(bars[0].get_facecolor()) == "#123456"
    # a new Filter starts with the saved style
    assert gui_filter.Window().plot_style == style


def test_dialog_shows_filter_options_only_from_filter(qt_offscreen):
    dialog = plot_style.show_dialog()
    assert not any(g.isVisible() for g in dialog.filter_groups)
    plot_style.show_dialog(filter_options=True)
    assert all(g.isVisible() for g in dialog.filter_groups)
    # the toolbar action of other windows hides them again
    canvas = lib.GenericPlotWindow("Test", "render")
    settings = [
        a for a in canvas.toolbar.actions() if a.text() == "Plot settings"
    ]
    settings[0].trigger()
    assert not any(g.isVisible() for g in dialog.filter_groups)


def test_nena_data_drawn_as_bars(qt_offscreen):
    _, nena, _, _ = _render_plot_windows()
    axes = nena.figure.axes[0]
    (bars,) = [p for p in axes.patches if isinstance(p, StepPatch)]
    assert bars.get_fill()
    assert to_hex(bars.get_facecolor()) == PlotStyle().primary_color
    (fit,) = axes.get_lines()
    assert fit.get_color() == SECOND


def test_dialog_reopens_with_saved_style(qt_offscreen):
    dialog = plot_style.show_dialog()
    dialog.close()
    style = replace(PlotStyle(), theme="Dark")
    plot_style.set_current(style)  # e.g., from another Picasso app
    assert plot_style.show_dialog().style() == style


def test_style_applies_to_both_windows(window):
    hist, hist2d = _open(window)
    style = replace(PlotStyle(), theme="Dark", grid=False)
    window.set_plot_style(style)
    background = THEMES["Dark"]["axes"]
    for w in (hist, hist2d):
        for axes in w.figure.axes:
            assert to_hex(axes.get_facecolor()) == background
        assert to_hex(w.figure.get_facecolor()) == THEMES["Dark"]["figure"]
    # the 1D histogram and the 1D histograms of the 2D window share bars
    bars = [
        to_hex(patch.get_facecolor())
        for w in (hist, hist2d)
        for axes in w.figure.axes
        for patch in axes.patches
        if isinstance(patch, StepPatch)
    ]
    assert bars and set(bars) == {style.primary_color}


def test_hist2d_layout_options(window):
    _, hist2d = _open(window)
    assert len(hist2d.figure.axes) == 4  # 2D, two 1D, colorbar
    window.set_plot_style(replace(PlotStyle(), show_marginals=False))
    assert len(hist2d.figure.axes) == 2
    assert hist2d.span_x is None and hist2d.span_y is None
    window.set_plot_style(
        replace(PlotStyle(), show_marginals=False, show_colorbar=False)
    )
    assert len(hist2d.figure.axes) == 1


def test_marginal_spans_filter_one_field(window):
    _, hist2d = _open(window)
    hist2d.on_span_select_x(1000.0, 2000.0)
    photons = window.get_column("photons")
    assert photons.min() > 1000.0 and photons.max() < 2000.0
    n = window.n_filtered
    hist2d = window.hist2d_windows["photons"]["sx"]
    hist2d.on_span_select_y(0.9, 1.1)
    sx = window.get_column("sx")
    assert sx.min() > 0.9 and sx.max() < 1.1
    assert window.n_filtered < n


def test_generic_plot_window_redraws(qt_offscreen):
    canvas = lib.GenericPlotWindow("Test", "render")
    calls = []

    def draw():
        calls.append(canvas.plot_style)
        with canvas.plot_context():
            canvas.figure.clear()
            canvas.figure.add_subplot().plot([0, 1])

    draw()
    canvas.redraw = draw
    style = replace(PlotStyle(), theme="Dark", colormap="magma")
    plot_style.set_current(style)
    assert calls[-1] == style
    (line,) = canvas.figure.axes[0].get_lines()
    assert line.get_color() == style.primary_color


def test_generic_plot_window_restyles_without_redraw(qt_offscreen):
    canvas = lib.GenericPlotWindow("Test", "render")
    axes = canvas.figure.add_subplot()
    (line,) = axes.plot([0, 1], color="red")
    plot_style.set_current(replace(PlotStyle(), theme="Dark"))
    assert to_hex(axes.get_facecolor()) == THEMES["Dark"]["axes"]
    assert line.get_color() == "red"


def test_subclustering_draws_on_given_figure():
    rng = np.random.default_rng(0)
    clustered = rng.integers(1, 20, 200)
    sparse = rng.integers(1, 10, 200)
    figure = Figure()
    style = PlotStyle()
    with style.context():
        fig, axes = lib.plot_subclustering_check(
            clustered, sparse, return_fig=True, fig=figure
        )
    assert fig is figure and figure.axes == [axes]
    colors = {to_hex(patch.get_facecolor()) for patch in axes.patches}
    assert style.primary_color in colors
    # the mean lines match the bars even when drawn outside the context
    lines = {line.get_color() for line in axes.get_lines()}
    assert lines == {style.primary_color, SECOND}


def _render_plot_windows():
    """Render's drift, NeNA, FRC and pick kinetics windows, plotted with
    synthetic results."""
    from picasso.gui import render as gui_render

    class Parent:
        pixelsize = 130.0

    frames = np.arange(100)
    drift = gui_render.DriftPlotWindow(Parent())
    drift.plot(pd.DataFrame({"x": np.sin(frames / 20), "y": frames / 100}))
    d = np.linspace(0, 1, 50)
    nena = gui_render.NenaPlotWindow(None)
    nena.plot(
        {
            "d": d,
            "data": np.exp(-d),
            "best_fit": np.exp(-d),
            "pixelsize": 130.0,
            "best_values": {"s": 0.1},
        }
    )
    frc = gui_render.FRCPlotWindow(None)
    frc.plot(
        {
            "frequencies": d,
            "frc_curve": 1 - d,
            "frc_curve_smooth": 1 - d,
            "resolution": 20.0,
        }
    )
    times = pd.Series(np.arange(1.0, 51.0))
    fit = {"best_values": {"a": 50.0, "t": 10.0, "c": 0.0}}
    fit["best_fit"] = 50 * (1 - np.exp(-times.to_numpy() / 10))
    kinetics = gui_render.PickHistWindow()
    kinetics.plot(pd.DataFrame({"len": times, "dark": times}), fit, fit)
    return drift, nena, frc, kinetics


def test_render_plot_windows_follow_style(qt_offscreen):
    windows = _render_plot_windows()
    style = replace(PlotStyle(), theme="Dark", colormap="magma")
    plot_style.set_current(style)
    for window in windows:
        assert window.plot_style == style
        assert to_hex(window.figure.get_facecolor()) == (
            THEMES["Dark"]["figure"]
        )
        for axes in window.figure.axes:
            assert to_hex(axes.get_facecolor()) == THEMES["Dark"]["axes"]
        # redrawn, not only restyled: the first data line takes the
        # style's first color (NeNA's data are bars, its fit the second)
        first = window.figure.axes[0].get_lines()[0]
        assert first.get_color() in (
            style.primary_color,
            style.series_colors[1],
            "gray",
        )
    _, _, frc, kinetics = windows
    assert kinetics.plotted
    assert kinetics.figure.axes == [kinetics.axes1, kinetics.axes2]
    # the FRC threshold stays visible on a dark background
    threshold = frc.figure.axes[0].get_lines()[-1]
    assert to_hex(threshold.get_color()) == THEMES["Dark"]["text"]


def test_ticks_created_at_draw_time_follow_style(qt_offscreen):
    plot_style.set_current(replace(PlotStyle(), theme="Dark"))
    *_, kinetics = _render_plot_windows()
    kinetics.canvas.draw()  # log axes create their ticks here
    for axes in (kinetics.axes1, kinetics.axes2):
        for tick in axes.xaxis.get_major_ticks():
            grid = to_hex(tick.gridline.get_color())
            assert grid == THEMES["Dark"]["grid"]


def test_show_title_hides_plot_titles(qt_offscreen):
    drift, nena, frc, kinetics = _render_plot_windows()
    plot_style.set_current(replace(PlotStyle(), show_title=False))
    for window in (nena, frc, kinetics):
        for axes in window.figure.axes:
            assert not axes.title.get_visible()
    assert not kinetics.figure._suptitle.get_visible()
    plot_style.set_current(PlotStyle())
    assert frc.figure.axes[0].title.get_visible()
    assert kinetics.figure._suptitle.get_visible()


@pytest.mark.parametrize(
    "fill, outline, filled, edge",
    [
        (True, False, True, False),
        (True, True, True, True),
        (False, False, False, True),
    ],
)
def test_histogram_style(fill, outline, filled, edge):
    style = lib.histogram_style("#123456", fill, outline, line_width=2.0)
    assert style["fill"] is filled
    has_edge = style["edgecolor"] != "none" and style["linewidth"] > 0
    assert has_edge is edge
    if not fill:
        assert style["linewidth"] == 2.0
        assert to_hex(style["edgecolor"]) == "#123456"
    if fill and outline:
        assert style["facecolor"][3] == lib.OUTLINED_FILL_ALPHA


@pytest.mark.parametrize("hist_style", plot_style.HIST_STYLES)
def test_hist_style_applies_everywhere(window, hist_style):
    style = replace(PlotStyle(), hist_style=hist_style)
    hist, hist2d = _open(window)
    plot_style.set_current(style)
    _, nena, _, _ = _render_plot_windows()
    rng = np.random.default_rng(0)
    subclustering = Figure()
    with style.context():
        lib.plot_subclustering_check(
            rng.integers(1, 20, 100),
            rng.integers(1, 10, 100),
            fig=subclustering,
            fill=style.hist_fill,
            outline=style.hist_outline,
        )
    patches = [
        p
        for f in (hist.figure, hist2d.figure, nena.figure)
        for axes in f.axes
        for p in axes.patches
        if isinstance(p, StepPatch)
    ]
    patches += list(subclustering.axes[0].patches)
    assert patches
    for patch in patches:
        assert patch.get_fill() is style.hist_fill
        assert (patch.get_linewidth() > 0) is style.hist_outline


def test_dialog_palette_and_colors(qt_offscreen):
    dialog = plot_style.show_dialog()
    dialog.palette.setCurrentText("Colorblind-safe")
    assert dialog.style().colors == tuple(PALETTES["Colorblind-safe"][:4])
    assert dialog.style().palette == "Colorblind-safe"
    # setting one color by hand makes the palette custom
    dialog.colors[1].set_value("#123456")
    style = dialog.style()
    assert style.palette == "Custom"
    assert style.colors[1] == "#123456"
    assert style.colors[0] == PALETTES["Colorblind-safe"][0]
    dialog._timer.timeout.emit()
    assert plot_style.current() == style


def test_drift_uses_four_colors(qt_offscreen):
    from picasso.gui import render as gui_render

    class Parent:
        pixelsize = 130.0

    frames = np.arange(50.0)
    window = gui_render.DriftPlotWindow(Parent())
    window.plot(pd.DataFrame({"x": frames, "y": -frames, "z": frames}))
    lines = [line for ax in window.figure.axes for line in ax.get_lines()]
    colors = [to_hex(line.get_color()) for line in lines]
    assert colors == PlotStyle().series_colors[:4]
