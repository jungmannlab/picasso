"""
picasso.gui.toolbars
~~~~~~~~~~~~~~~~~~~~

Toolbars of the Picasso GUIs, which hold actions of their menus, e.g.,
opening a file or Render's Pick tool. ``add_toolbar`` adds one to a
main window with the most used actions; the user chooses other actions
of the menus, their order, labels and icons in
``CustomizeToolbarDialog``, opened by right-clicking the toolbar or
from the appearance dialog (``File > Appearance...``).

An action is identified by its path in the menus, e.g.,
``"File > Open"``, the texts as the menus show them without the
ellipsis and the mnemonic ampersands (see ``action_path``). The layout
(``Layout``) thus applies to every window of a GUI and to menus that
are built anew, e.g., by Render's "Remove all localizations". A path
whose action is not in the menus, e.g., of a plugin that is loaded
after the window is built or not at all, is kept, and the button shows
once the action is there and the toolbar is refreshed (``refresh``,
done after the plugins are loaded).

The toolbar of each title is saved in ``settings["Toolbars"][<title>]``:

- ``items``: the paths and ``SEPARATOR`` for a separator,
- ``labels``: the text of a button instead of the action's,
- ``icons``: the icon of a button by its name (see ``button_icon``):
  one of Picasso's, one of the user's in ``~/.picasso/icons`` or
  ``NO_ICON``,
- ``area``: the side of the window, one of ``AREAS``, saved when the
  user moves the toolbar.

Without ``items``, the toolbar shows its defaults. Labels and icons
belong to the actions, which the menus share, so a custom icon shows in
the menus too; the action's own look returns when it leaves the
toolbar.

How the buttons look (icons, text) is part of the appearance, see
``picasso.gui.theme``.

:author: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from PyQt6 import QtCore, QtGui, QtWidgets, sip

from .. import docs_url, io, lib
from . import theme

#: Item of a layout that stands for a separator.
SEPARATOR = "---"
#: Icon of a layout for a button without an icon, i.e., with its text.
NO_ICON = "none"
#: Start of the name of an icon of the user's, see ``button_icon``.
USER_ICON_PREFIX = "user:"
#: Joins the menu titles and the action's text in ``action_path``.
PATH_SEPARATOR = " > "
#: Section of the user settings holding the toolbars.
SETTINGS_KEY = "Toolbars"
#: Sides of the window a toolbar can be on, by their name in the
#: settings.
AREAS = {
    "Top": QtCore.Qt.ToolBarArea.TopToolBarArea,
    "Bottom": QtCore.Qt.ToolBarArea.BottomToolBarArea,
    "Left": QtCore.Qt.ToolBarArea.LeftToolBarArea,
    "Right": QtCore.Qt.ToolBarArea.RightToolBarArea,
}

_ItemRole = QtCore.Qt.ItemDataRole
#: Data of the items of the dialog's toolbar list.
_PATH_ROLE = _ItemRole.UserRole
_LABEL_ROLE = _ItemRole.UserRole + 1
_ICON_ROLE = _ItemRole.UserRole + 2


def _walk(
    menu: QtWidgets.QMenu | QtWidgets.QMenuBar, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], QtGui.QAction]]:
    """Every action of ``menu`` and its submenus that can be on a
    toolbar, with its path; no separators, submenus, widgets or actions
    without text."""
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            yield from _walk(submenu, path + (theme._stripped(action.text()),))
        elif (
            not action.isSeparator()
            and not isinstance(action, QtWidgets.QWidgetAction)
            and action.text()
        ):
            yield path + (theme._stripped(action.text()),), action


def action_path(path: Iterable[str]) -> str:
    """The identifier of an action in a layout, e.g.,
    ``"File > Open"`` for ``("File", "Open")``."""
    return PATH_SEPARATOR.join(path)


def menu_actions(
    menu: QtWidgets.QMenu | QtWidgets.QMenuBar,
) -> dict[str, QtGui.QAction]:
    """The actions of ``menu`` and its submenus by their path (see
    ``action_path``), in the order of the menus. Of two actions with
    the same path, the first one is taken.

    Parameters
    ----------
    menu : QtWidgets.QMenu or QtWidgets.QMenuBar
        The menu, e.g., a window's menu bar.

    Returns
    -------
    actions : dict
        The actions by their path.
    """
    actions = {}
    for path, action in _walk(menu):
        actions.setdefault(action_path(path), action)
    return actions


def leaf_actions(menu: QtWidgets.QMenu) -> list[QtGui.QAction]:
    """The actions of ``menu`` and its submenus that can be on a
    toolbar (see ``menu_actions``)."""
    return [action for _, action in _walk(menu)]


def icon_names() -> list[str]:
    """The names of Picasso's SVG icons a button can take, see
    ``theme.icon``."""
    try:
        files = os.listdir(theme.ICONS_DIR)
    except OSError:
        return []
    return sorted(
        name[: -len(".svg")] for name in files if name.endswith(".svg")
    )


def user_icon_names() -> list[str]:
    """The names of the user's own icons in
    ``io.user_icons_directory()``, e.g., ``"user:star"`` for
    ``star.svg``, see ``button_icon``."""
    try:
        files = os.listdir(io.user_icons_directory())
    except OSError:
        return []
    stems = {
        os.path.splitext(name)[0]
        for name in files
        if os.path.splitext(name)[1].lower() in theme.ICON_EXTENSIONS
        and not name.startswith(".")
    }
    return [USER_ICON_PREFIX + stem for stem in sorted(stems)]


def button_icon(name: str) -> QtGui.QIcon:
    """The icon of a button by its name in a layout: one of Picasso's
    (see ``theme.icon``), one of the user's (``USER_ICON_PREFIX`` and
    its file name without the extension) or ``NO_ICON``; a null icon
    if the file is missing, in which case the button shows its label.
    """
    if name == NO_ICON:
        return QtGui.QIcon()
    if name.startswith(USER_ICON_PREFIX):
        path = theme.icon_path(
            io.user_icons_directory(), name[len(USER_ICON_PREFIX) :]
        )
        return QtGui.QIcon() if path is None else theme.icon_from_file(path)
    return theme.icon(name)


def import_icon(path: str) -> str:
    """Copy the image ``path`` to the user's icons
    (``io.user_icons_directory()``), unless it is there already, and
    return its name for a layout (see ``button_icon``).

    Parameters
    ----------
    path : str
        An SVG file, or an ``.ico`` or ``.png`` with a transparent
        background. Only its shape is used, drawn in the colors of the
        theme.

    Returns
    -------
    name : str
        The name of the icon, e.g., ``"user:star"``.

    Raises
    ------
    ValueError
        If the file is not an image an icon can be made of.
    """
    stem, extension = os.path.splitext(os.path.basename(path))
    extension = extension.lower()
    if (
        extension not in theme.ICON_EXTENSIONS
        or theme.icon_from_file(path).isNull()
    ):
        raise ValueError(
            f"{os.path.basename(path)!r} is not an SVG, ICO or PNG image."
        )
    folder = io.user_icons_directory()
    if os.path.dirname(os.path.abspath(path)) == os.path.abspath(folder):
        return USER_ICON_PREFIX + stem
    with open(path, "rb") as file:
        content = file.read()
    # a new name if another icon has this one (in any format)
    name, number = stem, 1
    while True:
        existing = theme.icon_path(folder, name)
        if existing is None:
            break
        if existing == os.path.join(folder, name + extension):
            with open(existing, "rb") as file:
                if file.read() == content:  # imported before
                    return USER_ICON_PREFIX + name
        number += 1
        name = f"{stem}-{number}"
    with open(os.path.join(folder, name + extension), "wb") as file:
        file.write(content)
    return USER_ICON_PREFIX + name


@dataclass
class Layout:
    """The buttons of a toolbar.

    Attributes
    ----------
    items : tuple of str
        Paths of the actions (see ``action_path``) and ``SEPARATOR``,
        from left (or top) to right.
    labels : dict
        Text of a button instead of the action's, by path.
    icons : dict
        Icon of a button instead of the action's, by path: its name
        (see ``theme.icon``) or ``NO_ICON``.
    """

    items: tuple[str, ...] = ()
    labels: dict[str, str] = field(default_factory=dict)
    icons: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.items = tuple(self.items)
        # only the buttons of the toolbar
        self.labels = {
            path: label
            for path, label in self.labels.items()
            if path in self.items and label
        }
        self.icons = {
            path: name
            for path, name in self.icons.items()
            if path in self.items and name
        }

    @classmethod
    def from_settings(cls, settings) -> Layout | None:
        """The layout saved in ``settings`` (the toolbar's entry of the
        user settings), or None if none (or an invalid one) is saved."""
        if not isinstance(settings, dict):
            return None
        items = settings.get("items")
        if not isinstance(items, list) or not all(
            isinstance(item, str) for item in items
        ):
            return None

        def strings(value) -> dict[str, str]:
            if not isinstance(value, dict):
                return {}
            return {
                k: v
                for k, v in value.items()
                if isinstance(k, str) and isinstance(v, str)
            }

        return cls(
            items,
            strings(settings.get("labels")),
            strings(settings.get("icons")),
        )

    def to_settings(self) -> dict:
        """The layout as saved in the user settings."""
        settings = {"items": list(self.items)}
        if self.labels:
            settings["labels"] = dict(self.labels)
        if self.icons:
            settings["icons"] = dict(self.icons)
        return settings


def _saved(title: str) -> dict:
    """The entry of the toolbar ``title`` in the user settings."""
    entry = io.load_user_settings()[SETTINGS_KEY].get(title)
    return dict(entry) if isinstance(entry, dict) else {}


def _save(title: str, entry: dict) -> None:
    """Save ``entry`` as the entry of the toolbar ``title`` in the user
    settings; an empty one is removed."""
    settings = io.load_user_settings()
    if entry:
        settings[SETTINGS_KEY][title] = entry
    else:
        settings[SETTINGS_KEY].pop(title, None)
        if not settings[SETTINGS_KEY]:
            settings.pop(SETTINGS_KEY)
    io.save_user_settings(settings)


def saved_layout(title: str) -> Layout | None:
    """The layout of the toolbar ``title`` saved in the user settings,
    or None if there is none."""
    return Layout.from_settings(_saved(title))


def save_layout(title: str, layout: Layout | None) -> None:
    """Save the layout of the toolbar ``title`` in the user settings
    and show it on every toolbar of that title in the running
    application.

    Parameters
    ----------
    title : str
        Title of the toolbar, e.g., "Render toolbar".
    layout : Layout or None
        The layout; None for the defaults, which removes the saved one.
    """
    entry = _saved(title)
    for key in ("items", "labels", "icons"):
        entry.pop(key, None)
    if layout is not None:
        entry.update(layout.to_settings())
    _save(title, entry)
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for window in app.topLevelWidgets():
        for toolbar in window.findChildren(ToolBar):
            if toolbar.objectName() == title:
                toolbar.set_layout(
                    toolbar.defaults if layout is None else layout
                )


def saved_area(title: str) -> QtCore.Qt.ToolBarArea | None:
    """The side of the window saved for the toolbar ``title``, or None
    if none is saved."""
    return AREAS.get(_saved(title).get("area"))


def save_area(title: str, area: QtCore.Qt.ToolBarArea) -> None:
    """Save ``area`` as the side of the window of the toolbar
    ``title``."""
    names = {value: name for name, value in AREAS.items()}
    if area not in names:
        return
    entry = _saved(title)
    if entry.get("area") != names[area]:
        entry["area"] = names[area]
        _save(title, entry)


def _set_tooltip(action: QtGui.QAction) -> None:
    """Give ``action`` a tooltip with its shortcut, unless it has its
    own tooltip."""
    text = theme._stripped(action.text())
    if action.toolTip() != text:  # own tooltip, or already set
        return
    shortcut = action.shortcut().toString(
        QtGui.QKeySequence.SequenceFormat.NativeText
    )
    action.setToolTip(f"{text} ({shortcut})" if shortcut else text)


class ToolBar(QtWidgets.QToolBar):
    """Toolbar holding actions of the menus of its window, see
    ``add_toolbar``.

    Parameters
    ----------
    title : str
        Title of the toolbar, also its object name and the key of its
        entry in the user settings.
    window : QtWidgets.QMainWindow
        The window whose menus hold the actions.

    Attributes
    ----------
    defaults : Layout
        The layout shown without a saved one.
    current : Layout
        The layout shown, also with the paths whose action is not in
        the menus.
    """

    def __init__(self, title: str, window: QtWidgets.QMainWindow) -> None:
        super().__init__(title, window)
        self.setObjectName(title)
        self.setProperty(theme._TOOLBAR_PROPERTY, True)
        self.defaults = Layout()
        self.current = Layout()
        # the actions' own icon and label, while the layout sets others
        self._own_looks = {}
        self.topLevelChanged.connect(self._save_area_later)

    def available_actions(self) -> dict[str, QtGui.QAction]:
        """The actions of the window's menus by their path."""
        menu_bar = self.parentWidget().menuBar()
        return menu_actions(menu_bar) if menu_bar is not None else {}

    def own_look(self, action: QtGui.QAction) -> tuple[QtGui.QIcon, str]:
        """The icon and the label of ``action`` without the layout's."""
        if action in self._own_looks:
            return self._own_looks[action]
        return action.icon(), action.iconText()

    def set_layout(self, layout: Layout) -> None:
        """Show the buttons of ``layout``; paths whose action is not in
        the menus are kept for later, see ``refresh``."""
        self.current = layout
        for action, (icon, label) in self._own_looks.items():
            if not sip.isdeleted(action):
                action.setIcon(icon)
                action.setIconText(label)
        self._own_looks = {}
        actions = self.available_actions()
        shown = []
        for item in layout.items:
            action = actions.get(item)
            if action is not None:
                shown.append((item, action))
            elif item == SEPARATOR and shown and shown[-1] is not None:
                shown.append(None)  # no separator first or twice
        if shown and shown[-1] is None:
            shown.pop()
        for action in self.actions():
            self.removeAction(action)
            if action.isSeparator():  # made by addSeparator
                action.deleteLater()
        for entry in shown:
            if entry is None:
                self.addSeparator()
                continue
            path, action = entry
            label = layout.labels.get(path)
            name = layout.icons.get(path)
            if label or name:
                self._own_looks[action] = (action.icon(), action.iconText())
            if label:
                action.setIconText(label)
            if name:
                action.setIcon(button_icon(name))
            _set_tooltip(action)
            self.addAction(action)

    def refresh(self) -> None:
        """Show the layout again, e.g., with the actions of plugins
        loaded since."""
        self.set_layout(self.current)

    def customize(self) -> CustomizeToolbarDialog:
        """Open the dialog for choosing the buttons of the toolbar."""
        dialog = CustomizeToolbarDialog(self, self.parentWidget())
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.open()
        return dialog

    def save_area(self) -> None:
        """Save the side of the window the toolbar is on, unless it
        floats."""
        window = self.parentWidget()
        if self.isFloating() or not isinstance(window, QtWidgets.QMainWindow):
            return
        save_area(self.objectName(), window.toolBarArea(self))

    def _save_area_later(self, *_) -> None:
        # once the window has docked the toolbar where it was dropped
        QtCore.QTimer.singleShot(
            0, lambda: None if sip.isdeleted(self) else self.save_area()
        )

    def event(self, event: QtCore.QEvent) -> bool:
        result = super().event(event)
        # the end of dragging the toolbar by its handle
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            self._save_area_later()
        return result

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction("Customize toolbar...", lambda: self.customize())
        menu.exec(event.globalPos())
        menu.deleteLater()
        event.accept()


def add_toolbar(
    window: QtWidgets.QMainWindow,
    title: str,
    items: Iterable[tuple | None],
) -> ToolBar:
    """Add a toolbar of actions of the window's menus to ``window``.

    ``items`` are its defaults; a layout the user saved (see
    ``CustomizeToolbarDialog``) is shown instead, on the side of the
    window it was moved to last. The actions get their icon (see
    ``theme.icon``), which the menus show too, and, unless they have
    their own, a tooltip with the shortcut. The toolbar is shown as set
    in the appearance (``theme.Appearance.toolbar``).

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        The window, with its menus built.
    title : str
        Name of the toolbar, shown in the window's context menu and
        the key of its entry in the user settings.
    items : iterable of tuple or None
        ``(action, icon_name)`` or ``(action, icon_name, label)``,
        where ``action`` is in the window's menus and ``label`` is a
        short text for the button, or None for a separator.

    Returns
    -------
    toolbar : ToolBar
        The toolbar.

    Raises
    ------
    ValueError
        If an action is not in the window's menus.
    """
    toolbar = ToolBar(title, window)
    area = saved_area(title) or QtCore.Qt.ToolBarArea.TopToolBarArea
    window.addToolBar(area, toolbar)
    paths = {
        action: path for path, action in toolbar.available_actions().items()
    }
    defaults = []
    for item in items:
        if item is None:
            defaults.append(SEPARATOR)
            continue
        action, name, *label = item
        if action not in paths:
            raise ValueError(
                f"{theme._stripped(action.text())!r} is not in the menus "
                f"of the window of {title!r}."
            )
        action.setIcon(theme.icon(name))
        if label:
            action.setIconText(label[0])
        defaults.append(paths[action])
    toolbar.defaults = Layout(defaults)
    toolbar.set_layout(saved_layout(title) or toolbar.defaults)
    theme._style_toolbar(toolbar, theme.applied() or theme.Appearance())
    return toolbar


def refresh(window: QtWidgets.QWidget) -> None:
    """Show the layouts of the toolbars of ``window`` again, e.g., with
    the actions of the plugins loaded since it was built."""
    for toolbar in window.findChildren(ToolBar):
        toolbar.refresh()


class CustomizeToolbarDialog(lib.Dialog):
    """Dialog for choosing the buttons of a toolbar, their order,
    labels and icons.

    The left side lists the actions of the window's menus, the right
    side the toolbar. Actions are added with "Add" or a double-click,
    reordered with "Up", "Down" or by dragging. Below the toolbar, the
    selected button gets a label and an icon other than its action's.
    "OK" saves the layout (see ``save_layout``) and shows it on every
    window of the GUI; "Restore Defaults" shows the toolbar's defaults.

    Parameters
    ----------
    toolbar : ToolBar
        The toolbar.
    parent : QtWidgets.QWidget or None, optional
        Parent widget. Default None.
    """

    DOCS_URL = docs_url("others.html#toolbars")

    def __init__(
        self, toolbar: ToolBar, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.toolbar = toolbar
        self.setWindowTitle(f"Customize {toolbar.windowTitle()}")
        self.resize(680, 540)
        self.actions_by_path = toolbar.available_actions()

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search the menus")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        header.addWidget(self.search, 1)
        header.addWidget(lib.HelpButton(self.DOCS_URL))
        layout.addLayout(header)

        lists = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Menus:"))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.tree.itemDoubleClicked.connect(lambda item, _: self._add([item]))
        left.addWidget(self.tree)
        lists.addLayout(left, 1)

        buttons = QtWidgets.QVBoxLayout()
        buttons.addStretch()
        self.add_button = QtWidgets.QPushButton("Add")
        self.add_button.setIcon(theme.icon("add"))
        self.add_button.setToolTip("Add the actions selected in the menus.")
        self.add_button.clicked.connect(
            lambda: self._add(self.tree.selectedItems())
        )
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setIcon(theme.icon("delete"))
        self.remove_button.clicked.connect(self._remove)
        self.separator_button = QtWidgets.QPushButton("Separator")
        self.separator_button.setToolTip(
            "Add a separator after the selected button."
        )
        self.separator_button.clicked.connect(lambda: self._insert(SEPARATOR))
        self.up_button = QtWidgets.QPushButton("Up")
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QtWidgets.QPushButton("Down")
        self.down_button.clicked.connect(lambda: self._move(1))
        for button in (
            self.add_button,
            self.remove_button,
            self.separator_button,
            self.up_button,
            self.down_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch()
        lists.addLayout(buttons)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Toolbar:"))
        self.list = QtWidgets.QListWidget()
        self.list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        self.list.currentRowChanged.connect(self._on_row_changed)
        right.addWidget(self.list)
        look = QtWidgets.QFormLayout()
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setClearButtonEnabled(True)
        self.label_edit.setToolTip(
            "Text of the selected button; empty for the action's own.\n"
            "Short labels keep the toolbar compact."
        )
        self.label_edit.textEdited.connect(self._set_label)
        look.addRow("Label:", self.label_edit)
        self.icon_combo = QtWidgets.QComboBox()
        self.icon_combo.setToolTip(
            "Icon of the selected button. The menus show it too.\n"
            "Without an icon, the button shows its label."
        )
        self.icon_combo.setMaxVisibleItems(16)
        self.icon_combo.activated.connect(self._set_icon)
        self._fill_icons()
        self.icon_file_button = QtWidgets.QPushButton("Choose file...")
        self.icon_file_button.setIcon(theme.icon("open"))
        self.icon_file_button.setToolTip(
            "Use an image of your own: an SVG, or a PNG or ICO with a\n"
            "transparent background. Only its shape is used, drawn in\n"
            "the colors of the theme. It is copied to\n"
            f"{io.user_icons_directory()}, where more icons can be put."
        )
        self.icon_file_button.clicked.connect(self._choose_icon_file)
        icon_row = QtWidgets.QHBoxLayout()
        icon_row.addWidget(self.icon_combo, 1)
        icon_row.addWidget(self.icon_file_button)
        look.addRow("Icon:", icon_row)
        right.addLayout(look)
        lists.addLayout(right, 1)
        layout.addLayout(lists)

        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        box.button(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(lambda: self.set_layout(toolbar.defaults))
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._muted = self.palette().color(
            QtGui.QPalette.ColorRole.PlaceholderText
        )
        self._build_tree()
        self.set_layout(toolbar.current)

    def _build_tree(self) -> None:
        """List the actions of the menus, grouped by menu."""
        nodes = {}
        for path, action in self.actions_by_path.items():
            names = path.split(PATH_SEPARATOR)
            parent = self.tree.invisibleRootItem()
            for depth in range(1, len(names)):
                key = tuple(names[:depth])
                if key not in nodes:
                    node = QtWidgets.QTreeWidgetItem(
                        parent, [names[depth - 1]]
                    )
                    node.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                    nodes[key] = node
                parent = nodes[key]
            item = QtWidgets.QTreeWidgetItem(parent, [names[-1]])
            item.setIcon(0, self.toolbar.own_look(action)[0])
            item.setToolTip(0, path)
            item.setData(0, _PATH_ROLE, path)
        self.tree.expandToDepth(0)

    def _tree_items(self) -> Iterator[QtWidgets.QTreeWidgetItem]:
        """The items of the actions in the tree."""
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value() is not None:
            item = iterator.value()
            if item.data(0, _PATH_ROLE) is not None:
                yield item
            iterator += 1

    def items(self) -> list[str]:
        """The paths shown on the right side, see ``Layout.items``."""
        return [
            self.list.item(row).data(_PATH_ROLE)
            for row in range(self.list.count())
        ]

    def layout_(self) -> Layout:
        """The layout shown on the right side."""
        labels, icons = {}, {}
        for row in range(self.list.count()):
            item = self.list.item(row)
            path = item.data(_PATH_ROLE)
            labels[path] = item.data(_LABEL_ROLE) or ""
            icons[path] = item.data(_ICON_ROLE) or ""
        return Layout(self.items(), labels, icons)

    def set_layout(self, layout: Layout) -> None:
        """Show ``layout`` on the right side."""
        self.list.clear()
        for path in layout.items:
            self.list.addItem(
                self._list_item(
                    path,
                    layout.labels.get(path, ""),
                    layout.icons.get(path, ""),
                )
            )
        self._update_tree()
        self._on_row_changed()

    def _own_look(self, path: str) -> tuple[QtGui.QIcon, str]:
        """The icon and the label of the action of ``path`` without
        the layout's."""
        action = self.actions_by_path.get(path)
        if action is None:
            return QtGui.QIcon(), path.split(PATH_SEPARATOR)[-1]
        return self.toolbar.own_look(action)

    def _list_item(
        self, path: str, label: str = "", icon_name: str = ""
    ) -> QtWidgets.QListWidgetItem:
        """The item showing ``path`` in the toolbar list."""
        item = QtWidgets.QListWidgetItem()
        item.setData(_PATH_ROLE, path)
        if path == SEPARATOR:
            item.setText("―――  separator  ―――")
            item.setForeground(self._muted)
            return item
        item.setData(_LABEL_ROLE, label)
        item.setData(_ICON_ROLE, icon_name)
        if path not in self.actions_by_path:
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            item.setForeground(self._muted)
            item.setToolTip(
                f"{path}\nNot in the menus now, e.g., of a plugin that is "
                "not loaded; shown once it is there."
            )
        else:
            item.setToolTip(path)
        self._show_look(item)
        return item

    def _show_look(self, item: QtWidgets.QListWidgetItem) -> None:
        """Show the label and the icon of the button of ``item``."""
        path = item.data(_PATH_ROLE)
        own_icon, own_label = self._own_look(path)
        item.setText(item.data(_LABEL_ROLE) or theme._stripped(own_label))
        name = item.data(_ICON_ROLE)
        if not name:
            item.setIcon(own_icon)
        else:
            item.setIcon(button_icon(name))

    def _current_button(self) -> QtWidgets.QListWidgetItem | None:
        """The selected item, unless it is a separator."""
        item = self.list.currentItem()
        if item is None or item.data(_PATH_ROLE) == SEPARATOR:
            return None
        return item

    def _set_label(self, text: str) -> None:
        item = self._current_button()
        if item is not None:
            item.setData(_LABEL_ROLE, text.strip())
            self._show_look(item)

    def _set_icon(self, index: int) -> None:
        item = self._current_button()
        if item is not None:
            item.setData(_ICON_ROLE, self.icon_combo.itemData(index))
            self._show_look(item)
        self._update_icon_file_button()

    def _update_icon_file_button(self) -> None:
        """Offer a file of the user's only while the selected button
        has its action's own icon."""
        self.icon_file_button.setEnabled(
            self._current_button() is not None
            and not self.icon_combo.currentData()
        )

    def _fill_icons(self) -> None:
        """List the icons: Picasso's, then the user's."""
        self.icon_combo.clear()
        self.icon_combo.addItem("Action's own", "")
        self.icon_combo.addItem("No icon", NO_ICON)
        self.icon_combo.insertSeparator(self.icon_combo.count())
        for name in icon_names():
            self.icon_combo.addItem(theme.icon(name), name, name)
        user_names = user_icon_names()
        if user_names:
            self.icon_combo.insertSeparator(self.icon_combo.count())
        for name in user_names:
            self.icon_combo.addItem(
                button_icon(name), name[len(USER_ICON_PREFIX) :], name
            )

    def _show_icon(self, name: str) -> None:
        """Select the icon ``name`` in the list, also if its file is
        missing."""
        index = self.icon_combo.findData(name)
        if index < 0:
            label = name.removeprefix(USER_ICON_PREFIX)
            self.icon_combo.addItem(f"{label} (missing)", name)
            index = self.icon_combo.count() - 1
        self.icon_combo.setCurrentIndex(index)

    def _choose_icon_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose an icon",
            io.user_icons_directory(),
            "Icons (*.svg *.png *.ico)",
        )
        if path:
            self.use_icon_file(path)

    def use_icon_file(self, path: str) -> None:
        """Give the selected button the image ``path`` as its icon,
        see ``import_icon``."""
        try:
            name = import_icon(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Icon", str(error))
            return
        self._fill_icons()
        self._show_icon(name)
        self._set_icon(self.icon_combo.currentIndex())

    def _insert(self, path: str) -> None:
        """Insert ``path`` after the selected button, or at the end."""
        row = self.list.currentRow()
        row = self.list.count() if row < 0 else row + 1
        self.list.insertItem(row, self._list_item(path))
        self.list.setCurrentRow(row)
        self._update_tree()

    def _add(self, tree_items: list[QtWidgets.QTreeWidgetItem]) -> None:
        """Add the actions of ``tree_items`` that are not on the
        toolbar yet."""
        on_toolbar = set(self.items())
        for item in tree_items:
            path = item.data(0, _PATH_ROLE)
            if path is not None and path not in on_toolbar:
                self._insert(path)

    def _remove(self) -> None:
        """Remove the selected button."""
        row = self.list.currentRow()
        if row >= 0:
            self.list.takeItem(row)
            self._update_tree()
            self._on_row_changed()

    def _move(self, step: int) -> None:
        """Move the selected button by ``step``."""
        row = self.list.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(target, item)
        self.list.setCurrentRow(target)

    def _filter(self, text: str) -> None:
        """Show the actions whose path contains ``text``."""
        text = text.strip().lower()
        for item in self._tree_items():
            path = item.data(0, _PATH_ROLE)
            item.setHidden(bool(text) and text not in path.lower())
        # hide the menus without a shown action
        iterator = QtWidgets.QTreeWidgetItemIterator(
            self.tree, QtWidgets.QTreeWidgetItemIterator.IteratorFlag.All
        )
        menus = []
        while iterator.value() is not None:
            item = iterator.value()
            if item.data(0, _PATH_ROLE) is None:
                menus.append(item)
            iterator += 1
        for menu in reversed(menus):  # submenus first
            menu.setHidden(
                all(menu.child(i).isHidden() for i in range(menu.childCount()))
            )
        if text:
            self.tree.expandAll()

    def _update_tree(self) -> None:
        """Gray out the actions on the toolbar."""
        on_toolbar = set(self.items())
        selectable = (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        for item in self._tree_items():
            path = item.data(0, _PATH_ROLE)
            if path in on_toolbar:
                item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                item.setToolTip(0, f"{path}\nOn the toolbar.")
            else:
                item.setFlags(selectable)
                item.setToolTip(0, path)

    def _on_row_changed(self, *_) -> None:
        row = self.list.currentRow()
        self.remove_button.setEnabled(row >= 0)
        self.up_button.setEnabled(row > 0)
        self.down_button.setEnabled(0 <= row < self.list.count() - 1)
        # the label and the icon of the selected button
        item = self._current_button()
        for widget in (self.label_edit, self.icon_combo):
            widget.setEnabled(item is not None)
        if item is None:
            self.label_edit.clear()
            self.label_edit.setPlaceholderText("")
            self.icon_combo.setCurrentIndex(0)
        else:
            own_label = theme._stripped(
                self._own_look(item.data(_PATH_ROLE))[1]
            )
            self.label_edit.setText(item.data(_LABEL_ROLE) or "")
            self.label_edit.setPlaceholderText(own_label)
            self._show_icon(item.data(_ICON_ROLE) or "")
        self._update_icon_file_button()

    def accept(self) -> None:
        layout = self.layout_()
        save_layout(
            self.toolbar.objectName(),
            None if layout == self.toolbar.defaults else layout,
        )
        super().accept()
