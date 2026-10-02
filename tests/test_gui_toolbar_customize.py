"""Customizing the toolbars of the Picasso GUIs.

``picasso.gui.toolbars`` adds toolbars of menu actions whose buttons
the user chooses. These tests cover identifying the actions by their
menu path, the defaults, saved layouts (also with actions that are not
in the menus yet), custom labels and icons, the saved side of the
window, the entry in the appearance dialog and the dialog.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io
from picasso.gui import theme, toolbars
from picasso.gui.toolbars import NO_ICON, SEPARATOR, Layout

TITLE = "Test toolbar"
Area = QtCore.Qt.ToolBarArea


@pytest.fixture
def home(tmp_path, monkeypatch):
    """User settings in ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture
def window(qt_offscreen, home):
    """A main window with File (with "Appearance..." and a submenu)
    and Tools menus."""
    window = QtWidgets.QMainWindow()
    file_menu = window.menuBar().addMenu("&File")
    window.open_action = file_menu.addAction("&Open...")
    window.open_action.setShortcut("Ctrl+O")
    window.save_action = file_menu.addAction("Save")
    export_menu = file_menu.addMenu("Export")
    window.export_action = export_menu.addAction("Image...")
    file_menu.addSeparator()
    window.appearance_action = theme.add_menu_action(file_menu)
    window.help_action = file_menu.addAction("Help")
    window.tools_menu = window.menuBar().addMenu("Tools")
    window.pick_action = window.tools_menu.addAction("Pick")
    return window


@pytest.fixture
def appearance_dialog(monkeypatch):
    """The appearance dialog is created afresh."""
    monkeypatch.setattr(theme, "_dialog", None)
    yield
    if theme._dialog is not None:
        theme._dialog.close()


def _add(window):
    return toolbars.add_toolbar(
        window,
        TITLE,
        [
            (window.open_action, "open", "Open"),
            (window.save_action, "save"),
            None,
            (window.pick_action, "tool-pick"),
        ],
    )


def test_actions_are_identified_by_their_menu_path(window):
    actions = toolbars.menu_actions(window.menuBar())
    assert list(actions) == [
        "File > Open",
        "File > Save",
        "File > Export > Image",
        "File > Appearance",
        "File > Help",
        "Tools > Pick",
    ]
    assert actions["File > Export > Image"] is window.export_action


def test_defaults_are_shown(window):
    toolbar = _add(window)
    assert toolbar.defaults == Layout(
        ("File > Open", "File > Save", SEPARATOR, "Tools > Pick")
    )
    actions = toolbar.actions()
    assert actions[:2] == [window.open_action, window.save_action]
    assert actions[2].isSeparator() and actions[3] is window.pick_action
    assert window.open_action.iconText() == "Open"
    assert window.open_action.toolTip().startswith("Open (")
    assert window.toolBarArea(toolbar) == Area.TopToolBarArea


def test_an_action_not_in_the_menus_is_refused(window):
    action = QtWidgets.QWidgetAction(window)
    action.setText("Elsewhere")
    with pytest.raises(ValueError, match="not in the menus"):
        toolbars.add_toolbar(window, TITLE, [(action, "open")])


def test_saved_layout_is_shown_and_keeps_missing_actions(window):
    items = (
        "Tools > Pick",
        SEPARATOR,
        SEPARATOR,
        "Plugins > Later",
        "File > Export > Image",
        SEPARATOR,
    )
    toolbars.save_layout(TITLE, Layout(items))
    toolbar = _add(window)
    assert toolbar.current.items == items
    # no separator twice or last
    actions = toolbar.actions()
    assert actions[0] is window.pick_action
    assert actions[1].isSeparator()
    assert actions[2] is window.export_action
    assert len(actions) == 3
    # the plugin's action is shown once it is there
    later = window.menuBar().addMenu("Plugins").addAction("Later")
    toolbars.refresh(window)
    assert later in toolbar.actions()


def test_saving_updates_the_toolbars_and_none_restores(window):
    toolbar = _add(window)
    toolbars.save_layout(TITLE, Layout(["File > Help"]))
    assert toolbar.actions() == [window.help_action]
    saved = io.load_user_settings()["Toolbars"][TITLE]
    assert saved == {"items": ["File > Help"]}
    toolbars.save_layout(TITLE, None)
    assert toolbar.current == toolbar.defaults
    assert "Toolbars" not in io.load_user_settings()


def test_invalid_saved_layout_gives_the_defaults(window):
    settings = io.load_user_settings()
    settings["Toolbars"][TITLE] = {"items": "File > Open"}
    io.save_user_settings(settings)
    assert toolbars.saved_layout(TITLE) is None
    toolbar = _add(window)
    assert toolbar.current == toolbar.defaults


def test_labels_and_icons_are_kept_for_the_buttons_only():
    layout = Layout(
        ["File > Open", "File > Save"],
        labels={"File > Open": "O", "File > Save": "", "Gone": "G"},
        icons={"File > Save": NO_ICON, "Gone": "open"},
    )
    assert layout.labels == {"File > Open": "O"}
    assert layout.icons == {"File > Save": NO_ICON}
    assert Layout.from_settings(layout.to_settings()) == layout


def test_custom_label_and_icon_and_back(window):
    toolbar = _add(window)
    own_icon = window.open_action.icon()
    toolbars.save_layout(
        TITLE,
        Layout(
            ["File > Open", "File > Help"],
            labels={"File > Help": "?"},
            icons={"File > Open": NO_ICON, "File > Help": "open"},
        ),
    )
    assert window.help_action.iconText() == "?"
    assert window.open_action.icon().isNull()
    # the menus keep the action's text
    assert window.help_action.text() == "Help"
    assert toolbar.own_look(window.help_action)[1] == "Help"
    # the action's own look returns with the defaults
    toolbars.save_layout(TITLE, None)
    assert window.help_action.iconText() == "Help"
    assert window.open_action.iconText() == "Open"
    assert window.open_action.icon().cacheKey() == own_icon.cacheKey()


def test_the_side_of_the_window_is_saved(window):
    toolbar = _add(window)
    window.addToolBar(Area.LeftToolBarArea, toolbar)  # as when dragged
    toolbar.save_area()
    assert toolbars.saved_area(TITLE) == Area.LeftToolBarArea
    # the layout is saved next to it, independently
    toolbars.save_layout(TITLE, Layout(["File > Help"]))
    toolbars.save_layout(TITLE, None)
    assert io.load_user_settings()["Toolbars"][TITLE] == {"area": "Left"}
    window.removeToolBar(toolbar)
    toolbar = _add(window)
    assert window.toolBarArea(toolbar) == Area.LeftToolBarArea


def test_a_released_drag_saves_the_side(window):
    toolbar = _add(window)
    window.addToolBar(Area.BottomToolBarArea, toolbar)
    QtWidgets.QApplication.sendEvent(
        toolbar,
        QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(2, 2),
            QtCore.QPointF(2, 2),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    )
    QtWidgets.QApplication.processEvents()
    assert toolbars.saved_area(TITLE) == Area.BottomToolBarArea


def test_appearance_dialog_customizes_the_window_toolbar(
    window, appearance_dialog
):
    window.appearance_action.trigger()
    dialog = theme._dialog
    # no toolbar yet
    assert not dialog.form.isRowVisible(dialog.customize_button)
    toolbar = _add(window)
    window.appearance_action.trigger()
    assert dialog.form.isRowVisible(dialog.customize_button)
    dialog.customize_button.click()
    (customize,) = window.findChildren(toolbars.CustomizeToolbarDialog)
    assert customize.toolbar is toolbar
    customize.reject()
    # no other menu entry
    texts = [a.text() for a in toolbars.leaf_actions(window.menuBar())]
    assert "Customize toolbar..." not in texts


def _tree_item(dialog, path):
    for item in dialog._tree_items():
        if item.toolTip(0).split("\n")[0] == path:
            return item
    raise KeyError(path)


def test_dialog_edits_and_saves_the_layout(window):
    toolbar = _add(window)
    dialog = toolbars.CustomizeToolbarDialog(toolbar, window)
    assert dialog.layout_() == toolbar.defaults
    # actions on the toolbar cannot be added twice
    assert not _tree_item(dialog, "File > Open").flags()
    dialog.list.setCurrentRow(0)
    dialog._add([_tree_item(dialog, "File > Help")])
    assert dialog.items()[1] == "File > Help"
    dialog._move(-1)
    assert dialog.items()[0] == "File > Help"
    dialog._insert(SEPARATOR)
    assert dialog.items()[1] == SEPARATOR
    dialog.list.setCurrentRow(dialog.items().index("File > Save"))
    dialog._remove()
    assert "File > Save" not in dialog.items()
    assert _tree_item(dialog, "File > Save").flags()
    dialog.accept()
    expected = (
        "File > Help",
        SEPARATOR,
        "File > Open",
        SEPARATOR,
        "Tools > Pick",
    )
    assert toolbars.saved_layout(TITLE) == Layout(expected)
    assert toolbar.current.items == expected


def test_dialog_sets_label_and_icon(window):
    toolbar = _add(window)
    dialog = toolbars.CustomizeToolbarDialog(toolbar, window)
    dialog.list.setCurrentRow(0)  # Open
    assert dialog.label_edit.placeholderText() == "Open"
    dialog.label_edit.textEdited.emit("Load")
    assert dialog.list.item(0).text() == "Load"
    index = dialog.icon_combo.findData(NO_ICON)
    dialog.icon_combo.setCurrentIndex(index)
    dialog.icon_combo.activated.emit(index)
    assert dialog.list.item(0).icon().isNull()
    # a separator has neither
    dialog.list.setCurrentRow(2)
    assert not dialog.label_edit.isEnabled()
    # the selection shows the button's own
    dialog.list.setCurrentRow(0)
    assert dialog.label_edit.text() == "Load"
    assert dialog.icon_combo.currentData() == NO_ICON
    dialog.accept()
    saved = toolbars.saved_layout(TITLE)
    assert saved.labels == {"File > Open": "Load"}
    assert saved.icons == {"File > Open": NO_ICON}
    assert window.open_action.iconText() == "Load"


def test_dialog_restores_the_defaults(window):
    toolbars.save_layout(
        TITLE, Layout(["File > Help"], labels={"File > Help": "?"})
    )
    toolbar = _add(window)
    dialog = toolbars.CustomizeToolbarDialog(toolbar, window)
    assert dialog.items() == ["File > Help"]
    assert dialog.list.item(0).text() == "?"
    dialog.set_layout(toolbar.defaults)
    dialog.accept()
    assert toolbars.saved_layout(TITLE) is None


def test_dialog_search_hides_other_actions(window):
    dialog = toolbars.CustomizeToolbarDialog(_add(window), window)
    dialog.search.setText("image")
    shown = [
        item.toolTip(0) for item in dialog._tree_items() if not item.isHidden()
    ]
    assert shown == ["File > Export > Image"]
    tools = dialog.tree.topLevelItem(1)
    assert tools.text(0) == "Tools" and tools.isHidden()
    dialog.search.clear()
    assert not tools.isHidden()


def test_dialog_marks_missing_actions(window):
    toolbars.save_layout(TITLE, Layout(["Plugins > Later"]))
    dialog = toolbars.CustomizeToolbarDialog(_add(window), window)
    item = dialog.list.item(0)
    assert item.font().italic()
    assert "Not in the menus" in item.toolTip()


# -- the user's icons -------------------------------------------------

#: A filled square, whatever its color, is drawn in the theme's colors.
SQUARE = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" '
    'fill="red"/></svg>'
)


def test_import_icon_copies_once_and_renames_on_a_clash(home, qt_offscreen):
    source = home / "star.svg"
    source.write_text(SQUARE)
    assert toolbars.import_icon(str(source)) == "user:star"
    folder = home / ".picasso" / "icons"
    assert (folder / "star.svg").read_text() == SQUARE
    # the same file again is the same icon
    assert toolbars.import_icon(str(source)) == "user:star"
    # another file of that name gets a new one
    other = home / "other" / "star.svg"
    other.parent.mkdir()
    other.write_text(SQUARE.replace("20", "10"))
    assert toolbars.import_icon(str(other)) == "user:star-2"
    # a file in the folder is used where it is
    assert toolbars.import_icon(str(folder / "star.svg")) == "user:star"
    assert toolbars.user_icon_names() == ["user:star", "user:star-2"]
    assert not toolbars.button_icon("user:star").isNull()
    assert toolbars.button_icon("user:gone").isNull()


def test_import_icon_refuses_other_files(home, qt_offscreen):
    text = home / "notes.txt"
    text.write_text("hello")
    broken = home / "broken.svg"
    broken.write_text("not an svg")
    for path in (text, broken):
        with pytest.raises(ValueError, match="not an SVG"):
            toolbars.import_icon(str(path))
    assert toolbars.user_icon_names() == []


def test_dialog_takes_an_icon_file(window, home):
    toolbar = _add(window)
    dialog = toolbars.CustomizeToolbarDialog(toolbar, window)
    assert not [
        dialog.icon_combo.itemData(i)
        for i in range(dialog.icon_combo.count())
        if str(dialog.icon_combo.itemData(i)).startswith("user:")
    ]
    (home / "star.svg").write_text(SQUARE)
    dialog.list.setCurrentRow(0)
    # offered with the action's own icon only
    assert dialog.icon_combo.currentData() == ""
    assert dialog.icon_file_button.isEnabled()
    index = dialog.icon_combo.findData(NO_ICON)
    dialog.icon_combo.setCurrentIndex(index)
    dialog.icon_combo.activated.emit(index)
    assert not dialog.icon_file_button.isEnabled()
    dialog.icon_combo.setCurrentIndex(0)
    dialog.icon_combo.activated.emit(0)
    assert dialog.icon_file_button.isEnabled()
    dialog.use_icon_file(str(home / "star.svg"))
    assert dialog.icon_combo.currentData() == "user:star"
    assert not dialog.icon_file_button.isEnabled()
    # nor for a separator
    dialog.list.setCurrentRow(dialog.items().index(SEPARATOR))
    assert not dialog.icon_file_button.isEnabled()
    dialog.list.setCurrentRow(0)
    assert not dialog.list.item(0).icon().isNull()
    dialog.accept()
    assert toolbars.saved_layout(TITLE).icons == {"File > Open": "user:star"}
    assert not window.open_action.icon().isNull()


def test_dialog_shows_a_missing_icon(window):
    toolbars.save_layout(
        TITLE, Layout(["File > Open"], icons={"File > Open": "user:gone"})
    )
    toolbar = _add(window)
    # the button shows its label
    assert window.open_action.icon().isNull()
    dialog = toolbars.CustomizeToolbarDialog(toolbar, window)
    dialog.list.setCurrentRow(0)
    assert dialog.icon_combo.currentData() == "user:gone"
    assert dialog.icon_combo.currentText() == "gone (missing)"
