"""Customizable keyboard shortcuts of the Picasso GUIs.

``picasso.gui.shortcuts`` lets the user change the shortcuts of the
actions of a GUI's menus and of the actions only the keyboard triggers.
These tests cover identifying the actions, applying and saving the
shortcuts (several per action, none, unknown paths of plugins), the
toolbar tooltips and the dialog.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io
from picasso.gui import shortcuts, toolbars

SCOPE = "Test GUI"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """User settings in ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _build(window: QtWidgets.QMainWindow) -> None:
    """File and Tools menus, a keyboard-only action and a toolbar."""
    file_menu = window.menuBar().addMenu("&File")
    window.open_action = file_menu.addAction("&Open...")
    window.open_action.setShortcut("Ctrl+O")
    window.save_action = file_menu.addAction("Save")
    window.save_action.setShortcuts(["Ctrl+S", "Ctrl+Shift+S"])
    window.shortcuts_action = shortcuts.add_menu_action(file_menu)
    tools_menu = window.menuBar().addMenu("Tools")
    window.pick_action = tools_menu.addAction("Pick")
    window.left_action = QtGui.QAction("Move left", window)
    window.left_action.setShortcut("Left")
    window.addAction(shortcuts.register(window.left_action, "Move left"))
    toolbars.add_toolbar(
        window, "Test toolbar", [(window.open_action, "open")]
    )


@pytest.fixture
def window(qt_offscreen, home):
    window = QtWidgets.QMainWindow()
    _build(window)
    shortcuts.install(window, SCOPE)
    yield window
    toolbars.clear_toolbars(window)
    window.deleteLater()


def _keys(action):
    return shortcuts.action_keys(action)


def test_actions_are_identified_by_path(window):
    actions = shortcuts.window_actions(window)
    assert list(actions) == [
        "File > Open",
        "File > Save",
        "File > Keyboard shortcuts",
        "Tools > Pick",
        "Keyboard only > Move left",
    ]
    assert actions["Keyboard only > Move left"] is window.left_action


def test_defaults_are_the_shortcuts_the_actions_were_built_with(window):
    assert shortcuts.default_keys(window.save_action) == (
        "Ctrl+S",
        "Ctrl+Shift+S",
    )
    assert shortcuts.default_keys(window.pick_action) == ()
    assert shortcuts.saved_shortcuts(SCOPE) == {}


def test_saved_shortcuts_apply_to_the_windows_of_the_gui(window):
    shortcuts.save_shortcuts(
        SCOPE,
        {
            "File > Open": ("Ctrl+K", "F2"),  # several
            "File > Save": (),  # none
            "Tools > Pick": ("P",),  # one where there was none
            "Keyboard only > Move left": ("A",),
            "Plugins > Not loaded": ("Ctrl+J",),  # kept for later
        },
    )
    assert _keys(window.open_action) == ("Ctrl+K", "F2")
    assert _keys(window.save_action) == ()
    assert _keys(window.pick_action) == ("P",)
    assert _keys(window.left_action) == ("A",)
    # the defaults are remembered
    assert shortcuts.default_keys(window.save_action) == (
        "Ctrl+S",
        "Ctrl+Shift+S",
    )
    saved = io.load_user_settings()["Shortcuts"][SCOPE]
    assert saved["File > Open"] == ["Ctrl+K", "F2"]
    assert saved["File > Save"] == []
    assert saved["Plugins > Not loaded"] == ["Ctrl+J"]

    # a window built later, e.g., a rebuilt menu, gets them too
    other = QtWidgets.QMainWindow()
    _build(other)
    shortcuts.install(other, SCOPE)
    assert _keys(other.open_action) == ("Ctrl+K", "F2")
    assert _keys(other.save_action) == ()

    # and back to the defaults; the entry is removed
    shortcuts.save_shortcuts(SCOPE, {})
    assert _keys(window.open_action) == ("Ctrl+O",)
    assert _keys(window.save_action) == ("Ctrl+S", "Ctrl+Shift+S")
    assert _keys(window.pick_action) == ()
    assert "Shortcuts" not in io.load_user_settings()
    toolbars.clear_toolbars(other)
    other.deleteLater()


def test_refresh_gives_actions_added_later_their_shortcuts(window):
    shortcuts.save_shortcuts(SCOPE, {"Plugins > Plugin action": ("F5",)})
    plugin_menu = window.menuBar().addMenu("Plugins")
    action = plugin_menu.addAction("Plugin action")
    assert _keys(action) == ()
    shortcuts.refresh(window)
    assert _keys(action) == ("F5",)


def test_invalid_settings_are_ignored(window):
    settings = io.load_user_settings()
    settings["Shortcuts"][SCOPE] = {
        "File > Open": "Ctrl+K",  # not a list
        "File > Save": ["F3", 4],
        "Tools > Pick": ["F4"],
    }
    io.save_user_settings(settings)
    assert shortcuts.saved_shortcuts(SCOPE) == {"Tools > Pick": ("F4",)}


def test_toolbar_tooltip_follows_the_shortcuts(window):
    native = shortcuts.native_text(["Ctrl+O"])
    assert window.open_action.toolTip() == f"Open ({native})"
    shortcuts.save_shortcuts(SCOPE, {"File > Open": ("Ctrl+K", "F2")})
    keys = shortcuts.native_text(["Ctrl+K", "F2"])
    assert window.open_action.toolTip() == f"Open ({keys})"
    shortcuts.save_shortcuts(SCOPE, {"File > Open": ()})
    assert window.open_action.toolTip() == "Open"
    # an own tooltip stays
    window.open_action.setToolTip("Open a file")
    shortcuts.save_shortcuts(SCOPE, {})
    assert window.open_action.toolTip() == "Open a file"


def test_window_hook_is_called(window):
    calls = []
    window.shortcuts_applied = lambda: calls.append(True)
    shortcuts.refresh(window)
    assert calls == [True]


def test_scope_is_found_from_the_menu(window, monkeypatch):
    opened = []
    monkeypatch.setattr(shortcuts, "show_dialog", opened.append)
    window.shortcuts_action.trigger()
    assert opened == [window]


def test_dialog_edits_and_saves(window, monkeypatch):
    dialog = shortcuts.ShortcutsDialog(window)
    assert dialog.windowTitle() == f"Keyboard shortcuts of {SCOPE}"
    dialog.select("Tools > Pick")
    assert dialog.current_path() == "Tools > Pick"
    assert dialog.add_key("F6")
    assert dialog.add_key("Shift+F6")
    assert not dialog.add_key("F6")  # already there
    assert dialog.keys["Tools > Pick"] == ("F6", "Shift+F6")
    # bold once changed
    assert dialog._item("Tools > Pick").font(0).bold()

    # a shortcut of another action is moved after asking
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_: QtWidgets.QMessageBox.StandardButton.No,
    )
    assert not dialog.add_key("Ctrl+O")
    assert dialog.keys["File > Open"] == ("Ctrl+O",)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    assert dialog.add_key("Ctrl+O")
    assert dialog.keys["File > Open"] == ()
    assert dialog.owners("Ctrl+O") == ["Tools > Pick"]

    # removing the selected shortcut
    dialog.key_list.setCurrentRow(0)
    dialog._remove()
    assert dialog.keys["Tools > Pick"] == ("Shift+F6", "Ctrl+O")

    dialog.accept()
    assert _keys(window.pick_action) == ("Shift+F6", "Ctrl+O")
    assert _keys(window.open_action) == ()
    assert shortcuts.saved_shortcuts(SCOPE) == {
        "File > Open": (),
        "Tools > Pick": ("Shift+F6", "Ctrl+O"),
    }


def test_dialog_records_a_key(window, qt_offscreen):
    dialog = shortcuts.ShortcutsDialog(window)
    dialog.select("Tools > Pick")
    dialog.key_edit.setKeySequence(QtGui.QKeySequence("Ctrl+Shift+Y"))
    dialog.key_edit.editingFinished.emit()
    assert dialog.keys["Tools > Pick"] == ("Ctrl+Shift+Y",)


def test_dialog_keeps_shortcuts_of_absent_actions(window):
    shortcuts.save_shortcuts(SCOPE, {"Plugins > Not loaded": ("Ctrl+J",)})
    dialog = shortcuts.ShortcutsDialog(window)
    dialog.select("File > Save")
    dialog._reset_current()  # unchanged: defaults
    dialog.accept()
    assert shortcuts.saved_shortcuts(SCOPE) == {
        "Plugins > Not loaded": ("Ctrl+J",)
    }


def test_dialog_reset_and_restore_defaults(window):
    shortcuts.save_shortcuts(
        SCOPE,
        {
            "File > Open": ("F2",),
            "Tools > Pick": ("P",),
            "Plugins > Not loaded": ("Ctrl+J",),
        },
    )
    dialog = shortcuts.ShortcutsDialog(window)
    dialog.select("File > Open")
    assert dialog.reset_button.isEnabled()
    dialog._reset_current()
    assert dialog.keys["File > Open"] == ("Ctrl+O",)
    assert not dialog.reset_button.isEnabled()
    dialog.restore_defaults()
    assert dialog.shortcuts() == {}
    dialog.accept()
    assert _keys(window.pick_action) == ()
    assert "Shortcuts" not in io.load_user_settings()


def test_dialog_search(window):
    dialog = shortcuts.ShortcutsDialog(window)
    dialog.search.setText("ctrl+shift+s")  # by shortcut
    shown = [
        item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        for item in dialog._tree_items()
        if not item.isHidden()
    ]
    assert shown == ["File > Save"]
    dialog.search.setText("left")
    shown = [
        item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        for item in dialog._tree_items()
        if not item.isHidden()
    ]
    assert shown == ["Keyboard only > Move left"]
