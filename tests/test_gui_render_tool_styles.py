"""The appearance of the tool overlays of ``picasso.gui.render``.

Tools Settings sets how picks (and a pick being drawn), measured points
and the Move tool's shift label are drawn. These tests cover the
defaults, which must look as before the settings existed, the color
selector (automatic, preset and custom colors), the redraw of the main
and 3D windows and the round trip through the user settings file.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import yaml
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io, render
from picasso.gui import overlay_style
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": np.full(n, 0.1),
            "lpy": np.full(n, 0.1),
            "photons": np.full(n, 1000.0),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding one channel of uniform localizations."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(256, 256)
    yield window
    window.view.stop_render_worker()


def _canvas() -> QtGui.QImage:
    image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("black"))
    return image


def _pixels(image: QtGui.QImage) -> np.ndarray:
    """BGRA pixels of an RGB32 image."""
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    bits = image.constBits()
    bits.setsize(image.height() * image.width() * 4)
    return (
        np.frombuffer(bits, dtype=np.uint8)
        .reshape(image.height(), image.width(), 4)
        .copy()
    )


def test_defaults_match_the_former_look(window):
    t_dialog = window.tools_settings_dialog
    pick = t_dialog.pick_overlay_style()
    assert pick.color == QtGui.QColor("yellow")
    assert pick.line_style == "Solid"
    assert pick.line_width == 1
    assert pick.opacity == 1
    assert pick.fill_opacity is None  # only brush picks are filled
    assert pick.font_size is None
    assert t_dialog.pick_overlay_style(drawing=True).color == QtGui.QColor(
        "green"
    )
    measure = t_dialog.measure_overlay_style()
    assert measure.font_size == 20
    assert t_dialog.measure_style.value("marker_size") == 20
    assert t_dialog.move_overlay_style().color == QtGui.QColor("yellow")

    # the automatic color follows the background
    window.dataset_dialog.wbackground.setChecked(True)
    assert t_dialog.pick_overlay_style().color == QtGui.QColor("red")
    assert t_dialog.measure_overlay_style().color == QtGui.QColor("red")
    # the color while drawing is fixed
    assert t_dialog.pick_overlay_style(drawing=True).color == QtGui.QColor(
        "green"
    )


def test_pick_style_changes_the_drawn_picks(window):
    view = window.view
    view._picks = [(16.0, 16.0)]
    # 10 camera pixels
    window.tools_settings_dialog.pick_diameter.setValue(10 * PIXELSIZE)
    default = _pixels(view.draw_picks(_canvas()))

    style = window.tools_settings_dialog.pick_style
    style.color.set_value("Cyan")
    style.set_value("line_width", 3)
    cyan = _pixels(view.draw_picks(_canvas()))
    assert (cyan[..., :3] > 0).any(-1).sum() > 2 * (
        (default[..., :3] > 0).any(-1).sum()
    )
    assert cyan[..., 2].max() == 0  # no red in cyan
    assert cyan[..., 0].max() == 255

    # filled with the line color
    style.set_value("fill_opacity", 50)
    filled = _pixels(view.draw_picks(_canvas()))
    assert filled[128, 128, 0] == pytest.approx(128, abs=2)

    # the Reset button restores the former look
    style.reset()
    np.testing.assert_array_equal(_pixels(view.draw_picks(_canvas())), default)


def test_style_change_redraws_the_scene(window, monkeypatch):
    calls = []
    monkeypatch.setattr(
        window.view, "update_scene", lambda **kwargs: calls.append(kwargs)
    )
    t_dialog = window.tools_settings_dialog
    t_dialog.pick_style.line_style.setCurrentText("Dotted")
    t_dialog.measure_style.set_value("marker_size", 30)
    assert calls == [{"use_cache": True}, {"use_cache": True}]


def test_drawing_color_is_used_for_a_box_being_dragged(window):
    view = window.view
    view.box_pick_start_x = view.box_pick_start_y = 40
    view.box_pick_current_x = view.box_pick_current_y = 200
    style = window.tools_settings_dialog.pick_style
    style.drawing_color.set_value("Magenta")
    style.set_value("fill_opacity", 100)
    pixels = _pixels(view.draw_box_pick_ongoing(_canvas()))
    np.testing.assert_array_equal(pixels[120, 120, :3], [255, 0, 255])


def test_rectangle_center_line_is_optional(window):
    view = window.view
    view._pick_shape = "Rectangle"
    window.tools_settings_dialog.pick_width.setValue(8 * PIXELSIZE)
    view._picks = [((4.0, 16.0), (28.0, 16.0))]
    view.rectangle_pick_start_x, view.rectangle_pick_start_y = 32, 128
    view.rectangle_pick_current_x, view.rectangle_pick_current_y = 224, 128
    style = window.tools_settings_dialog.pick_style
    assert window.tools_settings_dialog.pick_overlay_style().center_line
    # the center of the pick and of the pick being dragged
    assert _pixels(view.draw_picks(_canvas()))[128, 128, :3].any()
    assert _pixels(view.draw_rectangle_pick_ongoing(_canvas()))[
        128, 128, :3
    ].any()

    style.center_line.setChecked(False)
    assert not _pixels(view.draw_picks(_canvas()))[128, 128, :3].any()
    assert not _pixels(view.draw_rectangle_pick_ongoing(_canvas()))[
        128, 128, :3
    ].any()
    assert style.settings()["Rectangle center line"] is False


def test_measure_style_sets_markers_and_lines(window):
    view = window.view
    view._points = [(4.0, 16.0), (28.0, 16.0)]
    style = window.tools_settings_dialog.measure_style
    style.set_value("marker_size", 40)
    style.line_style.setCurrentText("Dashed")
    lit = (_pixels(view.draw_points(_canvas()))[..., :3] > 0).any(-1)
    # the vertical arm of the first cross, 40 display pixels long,
    # centered at x = 32, y = 128
    assert lit[110:146, 32].all()
    # the dashed line between the crosses has gaps
    assert 0 < lit[128, 60:200].sum() < 140


def test_move_label_uses_its_style(window):
    view = window.view
    view._move_channels = (0,)
    view._move_cursor = QtCore.QPoint(10, 10)
    view._move_shift = (1.0, 2.0)

    def lit(**values):
        style = window.tools_settings_dialog.move_style
        for field, value in values.items():
            style.set_value(field, value)
        pixels = _pixels(view.draw_move_shift(_canvas()))
        return (pixels[..., :3] > 0).any(-1).sum(), pixels

    small, _ = lit(font_size=8)
    large, pixels = lit(font_size=24, color="Lime")
    assert large > 2 * small  # the label is not clipped as it grows
    assert pixels[..., 2].max() == 0 and pixels[..., 1].max() == 255


def test_color_box_custom_colors(qt_offscreen, monkeypatch):
    box = overlay_style.ColorComboBox(auto_colors=("yellow", "red"))
    changes = []
    box.colorChanged.connect(lambda: changes.append(box.value()))
    assert box.value() == overlay_style.AUTO_COLOR
    assert box.color("yellow") == QtGui.QColor("yellow")

    assert box.set_value("#12ab34")
    assert box.value() == "#12AB34"
    assert box.color("yellow") == QtGui.QColor("#12AB34")
    # added once, before "Custom..."
    assert box.set_value("#12AB34")
    assert [box.itemText(i) for i in range(box.count())].count("#12AB34") == 1
    assert box.itemText(box.count() - 1) == overlay_style.CUSTOM_COLOR

    assert not box.set_value("not a color")
    assert not box.set_value(overlay_style.CUSTOM_COLOR)
    assert box.value() == "#12AB34"

    # "Custom..." opens the color picker ...
    monkeypatch.setattr(
        QtWidgets.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QtGui.QColor("#654321"),
    )
    box.setCurrentIndex(box.count() - 1)
    assert box.value() == "#654321"
    # ... and cancelling it keeps the previous color
    monkeypatch.setattr(
        QtWidgets.QColorDialog,
        "getColor",
        lambda *args, **kwargs: QtGui.QColor(),
    )
    box.setCurrentIndex(box.count() - 1)
    assert box.value() == "#654321"
    assert box.currentText() == "#654321"
    assert changes == ["#12AB34", "#654321"]


def test_styles_are_saved_on_close_and_loaded_on_start(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    t_dialog = window.tools_settings_dialog
    t_dialog.pick_style.color.set_value("#ABCDEF")
    t_dialog.pick_style.set_value("line_style", "Dash-dot")
    t_dialog.pick_style.set_value("fill_opacity", 30)
    t_dialog.pick_style.center_line.setChecked(False)
    t_dialog.measure_style.set_value("font_size", 12)
    t_dialog.move_style.set_value("opacity", 50)
    window.close()

    saved = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    styles = saved["Render"]["ToolStyles"]
    assert styles["Pick"]["Color"] == "#ABCDEF"
    assert styles["Pick"]["Line style"] == "Dash-dot"
    assert styles["Pick"]["Fill opacity (%)"] == 30
    assert styles["Pick"]["Label size (px)"] is None  # "Default"
    assert styles["Pick"]["Rectangle center line"] is False
    assert styles["Measure"]["Label size (px)"] == 12
    assert styles["Move"]["Opacity (%)"] == 50
    assert "Line style" not in styles["Move"]  # not a field of the label

    window = gui_render.Window(plugins_loaded=True)
    t_dialog = window.tools_settings_dialog
    pick = t_dialog.pick_overlay_style()
    assert pick.color == QtGui.QColor("#ABCDEF")
    assert pick.line_style == "Dash-dot"
    assert pick.fill_opacity == pytest.approx(0.3)
    assert pick.font_size is None
    assert not pick.center_line
    assert t_dialog.measure_overlay_style().font_size == 12
    assert t_dialog.move_overlay_style().opacity == pytest.approx(0.5)
    window.view.stop_render_worker()


def test_invalid_saved_styles_keep_the_defaults(qt_offscreen):
    io.save_user_settings(
        {
            "Render": {
                "ToolStyles": {
                    "Pick": {
                        "Color": "not a color",
                        "Line style": "Wavy",
                        "Line width (px)": "wide",
                        "Opacity (%)": 40,
                    },
                    "Measure": "nonsense",
                }
            }
        }
    )
    window = gui_render.Window(plugins_loaded=True)
    t_dialog = window.tools_settings_dialog
    pick = t_dialog.pick_overlay_style()
    assert pick.color == QtGui.QColor("yellow")
    assert pick.line_style == "Solid"
    assert pick.line_width == 1
    assert pick.opacity == pytest.approx(0.4)  # the valid entry is kept
    assert t_dialog.measure_overlay_style() == render.OverlayStyle(
        color=QtGui.QColor("yellow"), font_size=20
    )
    window.view.stop_render_worker()


def test_collapsible_group_box(qt_offscreen):
    from picasso import lib

    box = lib.CollapsibleGroupBox("Section", expanded=False, summary="a, b")
    QtWidgets.QVBoxLayout(box.content).addWidget(QtWidgets.QLabel("x"))
    header = box.toggle_button
    # the initial state is shown at once, without turning the chevron
    assert header.progress() == 0
    box.show()
    states = []
    box.expandedChanged.connect(states.append)
    assert not box.isExpanded() and not box.content.isVisible()
    collapsed = header.grab().toImage()

    header.click()
    assert box.isExpanded() and box.content.isVisible()
    # the chevron turns smoothly
    app = QtWidgets.QApplication.instance()
    assert header._animation.state() == header._animation.State.Running
    deadline = time.monotonic() + 5  # a busy event loop can be slow
    while header.progress() < 1 and time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.01)
    assert header.progress() == 1
    assert header.grab().toImage() != collapsed
    box.setExpanded(True)  # no change, no signal
    header.click()
    app.processEvents()
    assert not box.content.isVisible()
    assert states == [True, False]

    # hidden, the chevron is set at once
    box.hide()
    box.setExpanded(True)
    assert header.progress() == 1
    # a click leaves no focus ring behind, the Tab key does
    assert header.focusPolicy() == QtCore.Qt.FocusPolicy.TabFocus
    # the summary widens the header
    assert header.sizeHint().width() > header.minimumSizeHint().width()
    box.close()


def test_tools_dialog_appearance_collapses_and_scrolls(qt_offscreen):
    window = gui_render.Window(plugins_loaded=True)
    dialog = window.tools_settings_dialog
    # one collapsed section with a tab per tool
    assert not dialog.appearance_groupbox.isExpanded()
    tabs = dialog.appearance_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Pick",
        "Measure",
        "Move",
    ]
    for i, style in enumerate(
        (dialog.pick_style, dialog.measure_style, dialog.move_style)
    ):
        assert tabs.widget(i).isAncestorOf(style)
    assert isinstance(dialog.scroll_area, QtWidgets.QScrollArea)

    app = QtWidgets.QApplication.instance()
    dialog.show()
    app.processEvents()
    height = dialog.height()
    dialog.appearance_groupbox.toggle_button.click()
    app.processEvents()
    assert dialog.appearance_groupbox.isExpanded()
    assert dialog.height() > height
    screen_height = dialog.screen().availableGeometry().height()
    assert dialog.height() <= 0.85 * screen_height
    # the contents never scroll horizontally
    assert (
        dialog.scroll_area.viewport().width()
        >= dialog.scroll_area.widget().minimumSizeHint().width()
    )
    dialog.appearance_groupbox.toggle_button.click()
    app.processEvents()
    assert dialog.height() == height

    # on a screen too small for the contents, the dialog scrolls
    dialog.appearance_groupbox.setExpanded(True)
    app.processEvents()
    dialog.resize(dialog.width(), 150)
    app.processEvents()
    assert dialog.scroll_area.verticalScrollBar().maximum() > 0
    dialog.close()
    window.view.stop_render_worker()
