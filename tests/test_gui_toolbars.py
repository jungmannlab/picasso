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
    assert Appearance().toolbar == "Icons"
    assert Appearance.from_settings({"toolbar": "Huge"}).toolbar == "Icons"
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
