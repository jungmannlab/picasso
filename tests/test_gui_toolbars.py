"""Icons and toolbars of the Picasso windows.

``picasso.gui.theme.icon`` draws single-color SVG icons in the colors of
the theme, and ``theme.add_toolbar`` puts existing menu actions on a
toolbar of Render and Localize. These tests cover the icon colors and
the fallback without an icon file, the toolbar's buttons, tooltips and
style setting, and the toolbars of Render and Localize.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso.gui import localize as gui_localize
from picasso.gui import render as gui_render
from picasso.gui import theme
from picasso.gui.theme import Appearance

#: A filled square, whatever its color, is drawn in the theme's colors.
SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" '
    'fill="red"/></svg>'
)


@pytest.fixture
def icons(tmp_path, monkeypatch):
    """An icon folder holding ``square.svg`` and a broken file."""
    (tmp_path / "square.svg").write_text(SQUARE)
    (tmp_path / "broken.svg").write_text("not an svg")
    monkeypatch.setattr(theme, "ICONS_DIR", str(tmp_path))
    return tmp_path


def _center(icon: QtGui.QIcon, mode, state) -> str:
    image = icon.pixmap(QtCore.QSize(24, 24), mode, state).toImage()
    return image.pixelColor(12, 12).name()


# -- icons ------------------------------------------------------------


def test_missing_or_broken_icon_is_null(qapp, icons):
    assert theme.icon("nonexistent").isNull()
    assert theme.icon("broken").isNull()
    assert not theme.icon("square").isNull()


def test_icon_takes_the_theme_colors(qt_offscreen, icons, restore_theme):
    """Text color, dimmed when disabled, accent color when checked; and
    it follows a new theme."""
    Mode, State = QtGui.QIcon.Mode, QtGui.QIcon.State
    Role, Group = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    icon = theme.icon("square")
    for mode in ("Light", "Dark"):
        theme.apply(restore_theme, Appearance(mode=mode, accent="#123456"))
        palette = restore_theme.palette()
        assert _center(icon, Mode.Normal, State.Off) == (
            palette.color(Role.ButtonText).name()
        )
        assert _center(icon, Mode.Disabled, State.Off) == (
            palette.color(Group.Disabled, Role.ButtonText).name()
        )
        assert _center(icon, Mode.Normal, State.On) == "#123456"


# -- toolbars ---------------------------------------------------------


@pytest.fixture
def main_window(qt_offscreen, restore_theme):
    window = QtWidgets.QMainWindow()
    menu = window.menuBar().addMenu("File")
    open_action = menu.addAction("Open...")
    open_action.setShortcut("Ctrl+O")
    move_action = menu.addAction("Move")
    move_action.setToolTip("Drag the localizations.")
    plain_action = menu.addAction("Plain")
    yield window, open_action, move_action, plain_action
    window.close()


def test_toolbar_shares_the_menu_actions(main_window, icons):
    window, open_action, move_action, plain_action = main_window
    toolbar = theme.add_toolbar(
        window,
        "Test toolbar",
        [(open_action, "square", "Open"), None, (move_action, "missing")],
    )
    assert toolbar.actions()[0] is open_action
    assert toolbar.actions()[1].isSeparator()
    assert toolbar.actions()[2] is move_action
    assert plain_action not in toolbar.actions()
    assert not open_action.icon().isNull()
    # without an icon file, the button shows its text
    assert move_action.icon().isNull()
    assert open_action.iconText() == "Open"
    shortcut = open_action.shortcut().toString(
        QtGui.QKeySequence.SequenceFormat.NativeText
    )
    assert open_action.toolTip() == f"Open ({shortcut})"
    # an action's own tooltip is kept
    assert move_action.toolTip() == "Drag the localizations."


def test_toolbar_style_follows_the_appearance(main_window, restore_theme):
    window, open_action, *_ = main_window
    theme.apply(restore_theme, Appearance(toolbar="Icons and text"))
    toolbar = theme.add_toolbar(window, "Test toolbar", [(open_action, "x")])
    window.show()
    assert toolbar.toolButtonStyle() == (
        QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon
    )
    # an open window changes with the appearance
    theme.apply(restore_theme, Appearance(toolbar="Text", density="Compact"))
    assert toolbar.toolButtonStyle() == (
        QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
    )
    assert toolbar.iconSize() == QtCore.QSize(16, 16)
    theme.apply(restore_theme, Appearance(toolbar="Hidden"))
    assert not toolbar.isVisible()
    theme.apply(restore_theme, Appearance(toolbar="Icons"))
    assert toolbar.isVisible()


def test_toolbar_setting_is_validated_and_in_the_dialog(qt_offscreen):
    assert Appearance().toolbar == "Icons and text"
    assert Appearance.from_settings({"toolbar": "Huge"}).toolbar == (
        "Icons and text"
    )
    dialog = theme.AppearanceDialog(Appearance(toolbar="Text"))
    assert dialog.appearance().toolbar == "Text"
    # the toolbar is offered in every mode, also "Native"
    dialog.mode.setCurrentText("Native")
    assert dialog.form.isRowVisible(dialog.toolbar)
    dialog.close()


# -- Render and Localize ----------------------------------------------


def _locs(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 100, n).astype(np.uint32),
            "x": rng.uniform(0, 32, n).astype(np.float32),
            "y": rng.uniform(0, 32, n).astype(np.float32),
            "photons": np.full(n, 1000, np.float32),
            "sx": np.ones(n, np.float32),
            "sy": np.ones(n, np.float32),
            "bg": np.full(n, 10, np.float32),
            "lpx": np.full(n, 0.1, np.float32),
            "lpy": np.full(n, 0.1, np.float32),
        }
    )


def _info() -> list[dict]:
    return [{"Height": 32, "Width": 32, "Frames": 100, "Pixelsize": 130}]


def test_render_toolbar(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    try:
        toolbar = window.toolbar
        # the tool modes are the Tools menu's exclusive actions
        tools = window.tools_actiongroup.actions()
        assert all(action in toolbar.actions() for action in tools)
        # View and Tools actions wait for data, like their menus
        assert all(not action.isEnabled() for action in window.data_actions)
        window.view.add(
            str(tmp_path / "locs.hdf5"), _locs(), _info(), render_=False
        )
        visible = [a for a in window.data_actions if a.isVisible()]
        assert all(action.isEnabled() for action in visible)
        # 3D view is offered for 3D data only, also on the toolbar
        assert {a.text() for a in window.data_actions} - {
            a.text() for a in visible
        } == {"3D view"}
        # rebuilding the UI replaces the toolbar instead of adding one
        window.remove_locs()
        QtWidgets.QApplication.processEvents()
        toolbars = [
            t
            for t in window.findChildren(QtWidgets.QToolBar)
            if t.property("picassoToolbar") and not t.isHidden()
        ]
        assert toolbars == [window.toolbar]
    finally:
        window.view.stop_render_worker()
        window.close()


def test_localize_toolbar(qt_offscreen):
    window = gui_localize.Window()
    try:
        labels = [
            action.iconText()
            for action in window.toolbar.actions()
            if not action.isSeparator()
        ]
        assert labels[:2] == ["Open movie", "Save"]
        assert {"Identify", "Fit", "Localize", "Abort"} <= set(labels)
        assert window.abort_action in window.toolbar.actions()
    finally:
        window.close()


def test_help_button_is_filled_with_the_accent_color(
    qt_offscreen, icons, restore_theme
):
    """The help icon in the text color on the accent, inverted on
    hover; without the icon file a "?"."""
    from picasso import lib

    theme.apply(restore_theme, Appearance(mode="Light", accent="#123456"))
    Mode, State = QtGui.QIcon.Mode, QtGui.QIcon.State
    button = lib.HelpButton("https://example.org")
    assert button.icon().isNull()
    assert button.text() == "?"
    assert "background: palette(highlight)" in button.styleSheet()
    (icons / "help.svg").write_text(SQUARE)
    button = lib.HelpButton("https://example.org")
    on_accent = (
        restore_theme.palette()
        .color(QtGui.QPalette.ColorRole.HighlightedText)
        .name()
    )
    assert _center(button.icon(), Mode.Normal, State.Off) == on_accent
    point = QtCore.QPointF()
    button.enterEvent(QtGui.QEnterEvent(point, point, point))
    assert _center(button.icon(), Mode.Normal, State.Off) == "#123456"
    button.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert _center(button.icon(), Mode.Normal, State.Off) == on_accent


def test_filter_toolbar(qt_offscreen):
    from picasso.gui import filter as gui_filter

    window = gui_filter.Window()
    try:
        labels = [
            action.iconText()
            for action in window.toolbar.actions()
            if not action.isSeparator()
        ]
        assert labels == [
            "Open",
            "Save",
            "Export CSV",
            "Histogram",
            "2D histogram",
            "Subclustering",
            "Filter",
            "From metadata",
            "Plot settings",
        ]
    finally:
        window.close()


def test_average_toolbar(qt_offscreen):
    from picasso.gui import average as gui_average

    window = gui_average.Window()
    try:
        labels = [
            action.iconText()
            for action in window.toolbar.actions()
            if not action.isSeparator()
        ]
        assert labels == ["Open", "Save", "Parameters", "Average", "Abort"]
        assert not window.abort_action.isEnabled()
    finally:
        window.close()


def test_raster_icon_without_svg(qt_offscreen, icons, restore_theme):
    """Without an SVG, a raster image of the same name (e.g., an
    application icon) is tinted the same way; an SVG wins."""
    image = QtGui.QImage(24, 24, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.fillRect(4, 4, 16, 16, QtGui.QColor("magenta"))
    painter.end()
    image.save(str(icons / "app.png"))
    image.save(str(icons / "square.png"))
    theme.apply(restore_theme, Appearance(mode="Dark"))
    text = (
        restore_theme.palette()
        .color(QtGui.QPalette.ColorRole.ButtonText)
        .name()
    )
    Mode, State = QtGui.QIcon.Mode, QtGui.QIcon.State
    assert _center(theme.icon("app"), Mode.Normal, State.Off) == text
    # the corner stays transparent: only the shape is kept
    corner = theme.icon("app").pixmap(24, 24).toImage().pixelColor(1, 1)
    assert corner.alpha() == 0
    assert _center(theme.icon("square"), Mode.Normal, State.Off) == text


def test_plot_settings_button_follows_the_toolbar_style(
    qt_offscreen, restore_theme
):
    from picasso import lib

    Style = QtCore.Qt.ToolButtonStyle
    window = lib.GenericPlotWindow("Test", "render")
    (action,) = [
        a for a in window.toolbar.actions() if a.text() == "Plot settings"
    ]
    button = window.toolbar.widgetForAction(action)
    theme.apply(restore_theme, Appearance(toolbar="Icons and text"))
    assert button.toolButtonStyle() == Style.ToolButtonTextUnderIcon
    theme.apply(restore_theme, Appearance(toolbar="Text"))
    assert button.toolButtonStyle() == Style.ToolButtonTextOnly
    # the button stays when the toolbars are hidden, as an icon
    theme.apply(restore_theme, Appearance(toolbar="Hidden"))
    assert button.toolButtonStyle() == Style.ToolButtonIconOnly
    window.close()


def test_spinna_help_and_icons(qt_offscreen):
    from picasso.gui import spinna as gui_spinna

    window = gui_spinna.Window()
    try:
        file_menu = window.menuBar().actions()[0].menu()
        assert "Help" in [a.text() for a in file_menu.actions()]
        # the Simulate tab shows the Simulate app icon
        assert not window.tabs.tabIcon(1).isNull()
        buttons = {
            b.text(): b for b in window.findChildren(QtWidgets.QPushButton)
        }
        assert not buttons["Load molecules"].icon().isNull()
    finally:
        window.close()


def test_render_navigation_shortcuts_without_menu_entries(qt_offscreen):
    """Moving and zooming are off the View menu; their shortcuts stay,
    once, also after the UI is rebuilt."""
    window = gui_render.Window(plugins_loaded=True)
    try:
        view_menu = window.menuBar().actions()[1].menu()
        texts = {a.text() for a in view_menu.actions()}
        moves = {"Left", "Right", "Up", "Down", "Zoom in", "Zoom out"}
        assert not moves & texts
        window.remove_locs()
        shortcuts = [
            k.toString() for a in window.actions() for k in a.shortcuts()
        ]
        for key in ("Left", "A", "Ctrl++", "Ctrl+-"):
            assert shortcuts.count(key) == 1
    finally:
        window.view.stop_render_worker()
        window.close()


def test_3d_window_icons_and_navigation_shortcuts(qt_offscreen):
    """The 3D window's actions have icons; moving and zooming are off
    the View menu, their shortcuts work only while the window shows."""
    window = gui_render.Window(plugins_loaded=True)
    rot = window.window_rot
    try:

        def actions(menu_index):
            menu = rot.menuBar().actions()[menu_index].menu()
            return {a.text(): a for a in menu.actions() if a.text()}

        file_actions, view_actions = actions(0), actions(1)
        assert not file_actions["Build an animation..."].icon().isNull()
        assert not view_actions["Reset rotation"].icon().isNull()
        moves = {"Left", "Right", "Up", "Down", "Zoom in", "Zoom out"}
        assert not moves & set(view_actions)
        nav = {a.text(): a for a in rot.actions()}
        assert moves <= set(nav)
        assert not nav["Left"].isEnabled()  # hidden window
        rot.show()
        assert nav["Left"].isEnabled()
        rot.hide()
        assert not nav["Left"].isEnabled()
    finally:
        rot.view_rot.stop_render_worker()
        window.view.stop_render_worker()
        window.close()
