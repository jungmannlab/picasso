"""Text that Picasso draws itself follows the font size of the windows
on screen (``File > Appearance... > Font size``): the charts of
``PlotStyle`` and the scale bar, legend and rotation angles on rendered
images. Saved charts and exported images keep the sizes they always had.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtGui, QtWidgets
from matplotlib.figure import Figure

from picasso import io, render
from picasso.gui import plot_style, theme
from picasso.gui import render as gui_render
from picasso.gui.plot_style import PlotStyle
from picasso.gui.theme import Appearance

from tests.test_gui_rotation_async import (
    HEIGHT_NM,
    PIXELSIZE,
    WIDTH_NM,
    _info,
    _locs,
)


@pytest.fixture(autouse=True)
def fresh_hubs(monkeypatch, tmp_path):
    """Own hubs and user settings (never the developer's)."""
    monkeypatch.setattr(
        io, "_user_settings_filename", lambda: str(tmp_path / "s.yaml")
    )
    monkeypatch.setattr(plot_style, "_hub", None)
    monkeypatch.setattr(plot_style, "_windows_hub", None)
    monkeypatch.setattr(plot_style, "_last_text_scale", 1.0)
    monkeypatch.setattr(theme, "_hub", None)


@pytest.fixture
def scaled(qt_offscreen, restore_theme):
    """Windows at 200 % font size."""
    theme.apply(restore_theme, Appearance(mode="Light", font_scale=200))
    return restore_theme


def test_text_scale(qt_offscreen, restore_theme):
    theme.apply(restore_theme, Appearance(mode="Light", font_scale=150))
    assert theme.text_scale() == 1.5
    assert theme.scaled_text_size(20) == 30
    with theme.unscaled_text():
        with theme.unscaled_text():
            assert theme.text_scale() == 1
        assert theme.text_scale() == 1
    assert theme.text_scale() == 1.5
    # the platform's look keeps its own font
    theme.apply(restore_theme, Appearance(mode="Native", font_scale=150))
    assert theme.text_scale() == 1


def _styled_figure(style: PlotStyle) -> Figure:
    figure = Figure()
    style.style_figure(figure)
    with style.context():
        axes = figure.add_subplot()
        axes.plot([0, 1], [0, 1], label="data")
        axes.legend()
    style.style_axes(axes)
    style.label(axes, "x", "y")
    style.title(figure, "Title")
    # text sized by the chart itself is not the style's
    axes.text(0.5, 0.5, "note", fontsize=7)
    return figure


def _sizes(figure: Figure) -> dict:
    axes = figure.axes[0]
    return {
        "label": axes.xaxis.label.get_fontsize(),
        "title": figure._suptitle.get_fontsize(),
        "ticks": axes.xaxis.get_tick_params(which="major")["labelsize"],
        "tick text": axes.xaxis.get_ticklabels()[0].get_fontsize(),
        "legend": axes.get_legend().get_texts()[0].get_fontsize(),
        "note": axes.texts[0].get_fontsize(),
    }


def test_charts_are_larger_on_screen_but_saved_as_set(scaled, tmp_path):
    style = PlotStyle(font_size=11, tick_size=9, show_title=True)
    figure = _styled_figure(style)
    screen = _sizes(figure)
    assert screen == {
        "label": 22,
        "title": 24,
        "ticks": 18,
        "tick text": 18,
        "legend": 18,
        "note": 7,
    }
    saved = []
    figure.canvas.mpl_connect(
        "draw_event", lambda _event: saved.append(_sizes(figure))
    )
    figure.savefig(tmp_path / "chart.png")
    assert saved[-1] == {
        "label": 11,
        "title": 12,
        "ticks": 9,
        "tick text": 9,
        "legend": 9,
        "note": 7,
    }
    # the screen sizes come back
    assert _sizes(figure) == screen


def test_figures_without_a_plot_style_are_saved_unchanged(scaled, tmp_path):
    figure = Figure()
    axes = figure.add_subplot()
    axes.set_xlabel("x", fontsize=22)
    figure.savefig(tmp_path / "plain.png")
    assert axes.xaxis.label.get_fontsize() == 22


def test_charts_are_restyled_when_the_font_size_changes(
    qt_offscreen, restore_theme
):
    theme.apply(restore_theme, Appearance(mode="Light"))
    received = []
    plot_style.hub().changed.connect(received.append)
    # a chart theme that does not follow the windows' colors
    plot_style.set_current(PlotStyle(theme="Light"))
    received.clear()
    theme.apply(restore_theme, Appearance(mode="Light", font_scale=150))
    assert len(received) == 1
    theme.apply(restore_theme, Appearance(mode="Light", font_scale=150))
    assert len(received) == 1  # same size, nothing to restyle


# -- rendered images ----------------------------------------------------


@pytest.fixture
def window(scaled, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        window.view.add(
            str(tmp_path / f"locs_{i}.hdf5"), _locs(), _info(), render_=False
        )
    window.view.resize(256, 256)
    window.view.viewport = [(0.0, 0.0), (HEIGHT_NM, WIDTH_NM)]
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()


def test_scale_bar_and_legend_text_follow_the_font_size(window, monkeypatch):
    sizes = {}

    def record(name, original):
        def wrapper(**kwargs):
            sizes[name] = kwargs["text_fontsize"]
            return original(**kwargs)

        return wrapper

    monkeypatch.setattr(
        render, "draw_scalebar", record("scalebar", render.draw_scalebar)
    )
    monkeypatch.setattr(
        render, "draw_legend", record("legend", render.draw_legend)
    )
    window.display_settings_dlg.scalebar_groupbox.setChecked(True)
    window.dataset_dialog.legend.setChecked(True)
    image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
    window.view.draw_scalebar(image)
    window.view.draw_legend(image)
    assert sizes == {
        "scalebar": 2 * render.SCALEBAR_FONT_SIZE,
        "legend": 2 * render.LEGEND_FONT_SIZE,
    }
    with theme.unscaled_text():
        window.view.draw_scalebar(image)
        window.view.draw_legend(image)
    assert sizes == {
        "scalebar": render.SCALEBAR_FONT_SIZE,
        "legend": render.LEGEND_FONT_SIZE,
    }


def test_export_current_view_has_the_saved_text_size(
    window, tmp_path, monkeypatch
):
    view = window.view
    view.update_scene()
    path = str(tmp_path / "view.png")
    monkeypatch.setattr(
        gui_render.lib,
        "get_save_filename_ext_dialog",
        lambda *args, **kwargs: (path, ".png"),
    )
    during = []
    original = window._export_current_images

    def export(*args):
        during.append(theme.text_scale())
        original(*args)

    monkeypatch.setattr(window, "_export_current_images", export)
    window.export_current()
    assert during == [1.0]
    assert QtGui.QImage(path).width() == view.qimage.width()
    # the screen is drawn at the windows' size again
    assert theme.text_scale() == 2


def test_rotation_angles_fit_with_large_text(qt_offscreen):
    image = QtGui.QImage(200, 100, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("black"))
    render.draw_rotation_angles(image, (1.0, -2.0, 3.0), text_fontsize=36)
    # the text is drawn right-aligned inside the image
    columns = [
        x
        for x in range(image.width())
        if any(image.pixelColor(x, y).red() > 0 for y in range(100))
    ]
    assert columns and columns[0] > 0 and columns[-1] < image.width() - 5
    QtWidgets.QApplication.processEvents()


def _blank() -> QtGui.QImage:
    image = QtGui.QImage(300, 200, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("black"))
    return image


def test_legend_layout_is_unchanged_at_the_default_size(qt_offscreen):
    names, colors = ["c0", "c1"], [(255, 0, 0), (0, 255, 255)]
    new = render.draw_legend(_blank(), names, colors)
    old = render.draw_legend(_blank(), names, colors, init_pos=(12, 26), dy=24)
    assert new == old


def test_large_scale_bar_text_stays_inside_the_image(qt_offscreen):
    image = render.draw_scalebar(
        _blank(),
        viewport=((0, 0), (64 * PIXELSIZE, 64 * PIXELSIZE)),
        scalebar_length_nm=1000,
        text_fontsize=3 * render.SCALEBAR_FONT_SIZE,
    )
    lit = [
        x
        for x in range(image.width())
        if any(image.pixelColor(x, y).red() > 0 for y in range(150))
    ]
    assert lit and lit[-1] < image.width() - 1
