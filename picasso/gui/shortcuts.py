"""
picasso.gui.shortcuts
~~~~~~~~~~~~~~~~~~~~~

Keyboard shortcuts of the Picasso GUIs, which the user changes in
``ShortcutsDialog`` (``File > Keyboard shortcuts...``). Every action of
a window's menus can have any number of shortcuts, also one that has
none by default; so can the actions that only the keyboard triggers,
e.g., moving the view in Render, which ``register`` names.

An action is identified by its path in the menus, as for the toolbars
(see ``toolbars.action_path``), e.g., ``"File > Open"``; the actions
that are not in the menus by their path under ``KEYBOARD_ONLY``, e.g.,
``"Keyboard only > Zoom in"``. The shortcuts of a path thus apply to
every window of a GUI and to menus that are built anew, e.g., by
Render's "Remove all localizations". ``install`` applies the saved
shortcuts to a window once its menus are built; ``refresh`` again,
e.g., after the plugins added their actions.

The shortcuts that differ from the defaults are saved in
``settings["Shortcuts"][<GUI>][<path>]`` as a list of key sequences in
Qt's portable text, e.g., ``["Ctrl+O", "Ctrl+Shift+O"]`` ("Ctrl" is
Command on macOS), an empty list for none. The defaults are the
shortcuts an action has when the window is built.

:author: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from PyQt6 import QtCore, QtGui, QtWidgets

from .. import docs_url, io, lib
from . import theme, toolbars

#: Section of the user settings holding the shortcuts.
SETTINGS_KEY = "Shortcuts"
#: Group of the actions that are not in the menus, see ``register``.
KEYBOARD_ONLY = "Keyboard only"

#: Dynamic property of a window holding the name of its GUI.
_SCOPE_PROPERTY = "picasso_shortcut_scope"
#: Dynamic property of an action that is not in the menus holding its
#: path.
_PATH_PROPERTY = "picasso_shortcut_path"
#: Dynamic property of an action holding its default shortcuts.
_DEFAULTS_PROPERTY = "picasso_default_shortcuts"

_PORTABLE = QtGui.QKeySequence.SequenceFormat.PortableText
_NATIVE = QtGui.QKeySequence.SequenceFormat.NativeText

_ItemRole = QtCore.Qt.ItemDataRole
#: Path of the action of an item of the dialog's tree.
_PATH_ROLE = _ItemRole.UserRole
#: Shortcut (portable text) of an item of the dialog's key list.
_KEY_ROLE = _ItemRole.UserRole + 1


def _portable(sequence: QtGui.QKeySequence | str) -> str:
    """``sequence`` in Qt's portable text, e.g., ``"Ctrl+O"``."""
    return QtGui.QKeySequence(sequence).toString(_PORTABLE)


def native_text(keys: Iterable[str]) -> str:
    """The shortcuts ``keys`` (portable text) as the platform shows
    them, separated by commas, e.g., ``"⌘O, ⇧⌘O"`` on macOS."""
    return ", ".join(QtGui.QKeySequence(key).toString(_NATIVE) for key in keys)


def action_keys(action: QtGui.QAction) -> tuple[str, ...]:
    """The shortcuts of ``action`` in portable text."""
    keys = (_portable(sequence) for sequence in action.shortcuts())
    return tuple(dict.fromkeys(key for key in keys if key))


def default_keys(action: QtGui.QAction) -> tuple[str, ...]:
    """The shortcuts of ``action`` before the user's were applied, i.e.,
    those it was built with."""
    saved = action.property(_DEFAULTS_PROPERTY)
    if saved is None:
        return action_keys(action)
    return tuple(saved)


def _remember_defaults(action: QtGui.QAction) -> None:
    """Keep the shortcuts ``action`` was built with, see
    ``default_keys``."""
    if action.property(_DEFAULTS_PROPERTY) is None:
        action.setProperty(_DEFAULTS_PROPERTY, list(action_keys(action)))


def register(action: QtGui.QAction, name: str) -> QtGui.QAction:
    """Make the shortcuts of ``action``, which is not in the menus but
    added to its window (``QWidget.addAction``), customizable under the
    path ``"Keyboard only > <name>"``.

    Parameters
    ----------
    action : QtGui.QAction
        The action, e.g., moving the view to the left.
    name : str
        Its name in ``ShortcutsDialog``, e.g., ``"Move left"``.

    Returns
    -------
    action : QtGui.QAction
        The action.
    """
    action.setProperty(
        _PATH_PROPERTY, toolbars.action_path((KEYBOARD_ONLY, name))
    )
    return action


def _menu_bar(window: QtWidgets.QWidget) -> QtWidgets.QMenuBar | None:
    """The menu bar of ``window``: its own if it is a main window, else
    its child, e.g., in Design."""
    if isinstance(window, QtWidgets.QMainWindow):
        return window.menuBar()
    return window.findChild(QtWidgets.QMenuBar)


def window_actions(window: QtWidgets.QWidget) -> dict[str, QtGui.QAction]:
    """The actions of ``window`` whose shortcuts can be changed, by
    their path: those of its menus (see ``toolbars.menu_actions``),
    then those named by ``register``.

    Parameters
    ----------
    window : QtWidgets.QWidget
        The window, usually a main window.

    Returns
    -------
    actions : dict
        The actions by their path.
    """
    menu_bar = _menu_bar(window)
    actions = toolbars.menu_actions(menu_bar) if menu_bar is not None else {}
    for action in window.actions():
        path = action.property(_PATH_PROPERTY)
        if isinstance(path, str):
            actions.setdefault(path, action)
    return actions


def _valid_keys(value) -> tuple[str, ...] | None:
    """The shortcuts of a path as saved, or None if invalid."""
    if not isinstance(value, list) or not all(
        isinstance(key, str) for key in value
    ):
        return None
    keys = (_portable(key) for key in value)
    return tuple(dict.fromkeys(key for key in keys if key))


def saved_shortcuts(scope: str) -> dict[str, tuple[str, ...]]:
    """The shortcuts the user set in the GUI ``scope``, by path; only
    those that differ from the defaults (see the module docstring).

    Parameters
    ----------
    scope : str
        Name of the GUI, e.g., ``"Render"``.

    Returns
    -------
    shortcuts : dict
        The shortcuts (portable text) by path; an empty tuple for an
        action without shortcuts.
    """
    entry = io.load_user_settings()[SETTINGS_KEY].get(scope)
    if not isinstance(entry, dict):
        return {}
    shortcuts = {}
    for path, value in entry.items():
        keys = _valid_keys(value)
        if isinstance(path, str) and keys is not None:
            shortcuts[path] = keys
    return shortcuts


def save_shortcuts(scope: str, shortcuts: dict[str, Iterable[str]]) -> None:
    """Save the shortcuts the user set in the GUI ``scope`` and apply
    them to all of its windows in the running application.

    Parameters
    ----------
    scope : str
        Name of the GUI, e.g., ``"Render"``.
    shortcuts : dict
        The shortcuts (portable text) that differ from the defaults, by
        path; an empty one for an action without shortcuts. Without
        any, the defaults are used and the entry is removed.
    """
    settings = io.load_user_settings()
    if shortcuts:
        settings[SETTINGS_KEY][scope] = {
            path: list(keys) for path, keys in shortcuts.items()
        }
    else:
        settings[SETTINGS_KEY].pop(scope, None)
        if not settings[SETTINGS_KEY]:
            settings.pop(SETTINGS_KEY)
    io.save_user_settings(settings)
    for window in scope_windows(scope):
        apply(window, {path: tuple(keys) for path, keys in shortcuts.items()})


def scope_windows(scope: str) -> list[QtWidgets.QWidget]:
    """The windows of the GUI ``scope`` in the running application, see
    ``install``."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return []
    windows = []
    for widget in app.topLevelWidgets():
        candidates = [widget, *widget.findChildren(QtWidgets.QMainWindow)]
        for window in candidates:
            if window.property(_SCOPE_PROPERTY) == scope and (
                window not in windows
            ):
                windows.append(window)
    return windows


def scope_of(window: QtWidgets.QWidget | None) -> str | None:
    """The name of the GUI of ``window`` (see ``install``), or None."""
    scope = None if window is None else window.property(_SCOPE_PROPERTY)
    return scope if isinstance(scope, str) else None


def _scope_window(
    widget: QtWidgets.QWidget | None,
) -> QtWidgets.QWidget | None:
    """The window holding ``widget`` that was installed (see
    ``install``), e.g., of a menu of its menu bar; None if none."""
    while widget is not None and scope_of(widget) is None:
        widget = widget.parentWidget()
    return widget


def set_action_keys(action: QtGui.QAction, keys: Iterable[str]) -> None:
    """Give ``action`` the shortcuts ``keys`` (portable text) and update
    its tooltip on the toolbars."""
    _remember_defaults(action)
    action.setShortcuts([QtGui.QKeySequence(key) for key in keys])
    toolbars._set_tooltip(action)


def apply(
    window: QtWidgets.QWidget,
    shortcuts: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """Give the actions of ``window`` the user's shortcuts and the
    others their defaults.

    A window that wants to know, e.g., to show the shortcuts in a
    tooltip of its own, has a method ``shortcuts_applied``, called
    afterward.

    Parameters
    ----------
    window : QtWidgets.QWidget
        The window, see ``install``.
    shortcuts : dict or None, optional
        The user's shortcuts by path; None for those saved for the GUI
        of the window. Default None.
    """
    if shortcuts is None:
        scope = scope_of(window)
        shortcuts = {} if scope is None else saved_shortcuts(scope)
    for path, action in window_actions(window).items():
        keys = shortcuts.get(path, default_keys(action))
        if keys != action_keys(action):
            set_action_keys(action, keys)
        else:
            _remember_defaults(action)
    hook = getattr(window, "shortcuts_applied", None)
    if callable(hook):
        hook()


def install(window: QtWidgets.QWidget, scope: str) -> None:
    """Apply the shortcuts the user saved for the GUI ``scope`` to
    ``window``, once its menus are built; ``refresh`` applies them
    again, e.g., to the actions of plugins loaded since.

    Parameters
    ----------
    window : QtWidgets.QWidget
        The window, usually a main window, with its menus built.
    scope : str
        Name of the GUI, which keys the shortcuts in the user settings,
        e.g., ``"Render"``.
    """
    window.setProperty(_SCOPE_PROPERTY, scope)
    apply(window)


def refresh(window: QtWidgets.QWidget | None) -> None:
    """Apply the saved shortcuts to ``window`` again, e.g., with the
    actions of the plugins loaded since it was built; nothing if it
    was not installed (see ``install``)."""
    if window is not None and scope_of(window):
        apply(window)


def show_dialog(window: QtWidgets.QWidget | None) -> ShortcutsDialog | None:
    """Open the dialog for the shortcuts of ``window``'s GUI.

    Parameters
    ----------
    window : QtWidgets.QWidget or None
        The window, see ``install``.

    Returns
    -------
    dialog : ShortcutsDialog or None
        The dialog; None if ``window`` was not installed.
    """
    if scope_of(window) is None:
        return None
    dialog = ShortcutsDialog(window, window)
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.open()
    return dialog


def add_menu_action(menu: QtWidgets.QMenu) -> QtGui.QAction:
    """Add "Keyboard shortcuts...", which opens ``ShortcutsDialog`` for
    the window of ``menu``, to ``menu``.

    Parameters
    ----------
    menu : QtWidgets.QMenu
        The menu, usually File.

    Returns
    -------
    action : QtGui.QAction
        The added action.
    """
    action = menu.addAction("Keyboard shortcuts...")
    action.setIcon(theme.icon("keyboard"))
    action.triggered.connect(lambda: show_dialog(_scope_window(menu)))
    return action


class KeyEdit(QtWidgets.QKeySequenceEdit):
    """Records one key combination, e.g., Ctrl+Shift+O; also those that
    finish recording in ``QKeySequenceEdit``, i.e., Tab and Return."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMaximumSequenceLength(1)
        self.setFinishingKeyCombinations([])
        self.setClearButtonEnabled(True)


class ShortcutsDialog(lib.Dialog):
    """Dialog for the keyboard shortcuts of the actions of a GUI.

    The tree lists the actions of the window's menus, grouped by menu,
    and those only the keyboard triggers (``KEYBOARD_ONLY``), with
    their shortcuts; changed ones are in bold. Below, the shortcuts of
    the selected action are added by pressing a key combination in the
    field, removed or reset to the action's defaults. A shortcut that
    another action has is moved after asking, as two actions with one
    shortcut block each other. "OK" saves the shortcuts (see
    ``save_shortcuts``) and applies them to every window of the GUI;
    "Restore Defaults" shows the defaults of all actions.

    Parameters
    ----------
    window : QtWidgets.QWidget
        The window whose actions are listed, see ``install``.
    parent : QtWidgets.QWidget or None, optional
        Parent widget. Default None.
    """

    DOCS_URL = docs_url("others.html#keyboard-shortcuts")

    def __init__(
        self,
        window: QtWidgets.QWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.scope = scope_of(window) or ""
        self.setWindowTitle(f"Keyboard shortcuts of {self.scope}")
        self.resize(640, 600)
        self.actions_by_path = window_actions(window)
        self.defaults = {
            path: default_keys(action)
            for path, action in self.actions_by_path.items()
        }
        saved = saved_shortcuts(self.scope)
        # saved shortcuts of actions not in the window, e.g., of a
        # plugin that is not loaded, are kept
        self._absent = {
            path: keys
            for path, keys in saved.items()
            if path not in self.actions_by_path
        }
        self.keys = {
            path: saved.get(path, self.defaults[path])
            for path in self.actions_by_path
        }

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search actions and shortcuts")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        header.addWidget(self.search, 1)
        header.addWidget(lib.HelpButton(self.DOCS_URL))
        layout.addLayout(header)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Action", "Shortcuts"])
        self.tree.setAlternatingRowColors(True)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.tree, 1)

        self.edit_box = QtWidgets.QGroupBox("Shortcuts of the selected action")
        edit_layout = QtWidgets.QGridLayout(self.edit_box)
        self.key_list = QtWidgets.QListWidget()
        self.key_list.setMaximumHeight(
            5 * self.key_list.fontMetrics().height() + 12
        )
        self.key_list.currentRowChanged.connect(self._update_buttons)
        edit_layout.addWidget(self.key_list, 0, 0, 3, 1)
        self.key_edit = KeyEdit()
        self.key_edit.setToolTip(
            "Click here and press a key combination to add it as a\n"
            "shortcut of the selected action."
        )
        self.key_edit.editingFinished.connect(self._add_recorded)
        edit_layout.addWidget(self.key_edit, 0, 1)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.setIcon(theme.icon("delete"))
        self.remove_button.setToolTip("Remove the selected shortcut.")
        self.remove_button.clicked.connect(self._remove)
        edit_layout.addWidget(self.remove_button, 1, 1)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setIcon(theme.icon("reload"))
        self.reset_button.setToolTip(
            "Give the selected action its default shortcuts."
        )
        self.reset_button.clicked.connect(self._reset_current)
        edit_layout.addWidget(self.reset_button, 2, 1)
        self.default_label = QtWidgets.QLabel()
        self.default_label.setForegroundRole(
            QtGui.QPalette.ColorRole.PlaceholderText
        )
        edit_layout.addWidget(self.default_label, 3, 0, 1, 2)
        layout.addWidget(self.edit_box)

        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        box.button(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(
            lambda: lib.confirm_restore_defaults(self, "all shortcuts")
            and self.restore_defaults()
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        self._build_tree()
        self._on_current_changed()

    # tree
    def _build_tree(self) -> None:
        """List the actions, grouped by menu."""
        nodes = {}
        for path, action in self.actions_by_path.items():
            names = path.split(toolbars.PATH_SEPARATOR)
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
            item.setIcon(0, action.icon())
            item.setData(0, _PATH_ROLE, path)
            self._show_keys(item)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

    def _tree_items(self) -> Iterator[QtWidgets.QTreeWidgetItem]:
        """The items of the actions in the tree."""
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value() is not None:
            item = iterator.value()
            if item.data(0, _PATH_ROLE) is not None:
                yield item
            iterator += 1

    def _item(self, path: str) -> QtWidgets.QTreeWidgetItem | None:
        """The tree item of the action of ``path``."""
        for item in self._tree_items():
            if item.data(0, _PATH_ROLE) == path:
                return item
        return None

    def _show_keys(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """Show the shortcuts of the action of ``item``, in bold if
        changed."""
        path = item.data(0, _PATH_ROLE)
        keys = self.keys[path]
        item.setText(1, native_text(keys))
        changed = keys != self.defaults[path]
        for column in (0, 1):
            font = item.font(column)
            font.setBold(changed)
            item.setFont(column, font)
        tooltip = path
        if changed:
            default = native_text(self.defaults[path]) or "none"
            tooltip += f"\nChanged; default: {default}"
        item.setToolTip(0, tooltip)
        item.setToolTip(1, tooltip)

    def _filter(self, text: str) -> None:
        """Show the actions whose path or shortcuts contain ``text``."""
        text = text.strip().lower()
        for item in self._tree_items():
            path = item.data(0, _PATH_ROLE)
            haystack = " ".join([path, item.text(1), *self.keys[path]]).lower()
            item.setHidden(bool(text) and text not in haystack)
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

    # selected action
    def current_path(self) -> str | None:
        """The path of the selected action, or None."""
        item = self.tree.currentItem()
        return None if item is None else item.data(0, _PATH_ROLE)

    def select(self, path: str) -> None:
        """Select the action of ``path``."""
        item = self._item(path)
        if item is not None:
            self.tree.setCurrentItem(item)

    def _on_current_changed(self, *_) -> None:
        path = self.current_path()
        self.key_list.clear()
        self.edit_box.setEnabled(path is not None)
        if path is None:
            self.edit_box.setTitle("Shortcuts of the selected action")
            self.default_label.clear()
            self._update_buttons()
            return
        self.edit_box.setTitle(f"Shortcuts of {path}")
        for key in self.keys[path]:
            item = QtWidgets.QListWidgetItem(native_text([key]))
            item.setData(_KEY_ROLE, key)
            self.key_list.addItem(item)
        if not self.keys[path]:
            item = QtWidgets.QListWidgetItem("No shortcut")
            item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.key_list.addItem(item)
        default = native_text(self.defaults[path]) or "none"
        self.default_label.setText(f"Default: {default}")
        self._update_buttons()

    def _update_buttons(self, *_) -> None:
        path = self.current_path()
        item = self.key_list.currentItem()
        self.remove_button.setEnabled(
            item is not None and item.data(_KEY_ROLE) is not None
        )
        self.reset_button.setEnabled(
            path is not None and self.keys[path] != self.defaults[path]
        )

    def _set_keys(self, path: str, keys: Iterable[str]) -> None:
        """Give the action of ``path`` the shortcuts ``keys``."""
        self.keys[path] = tuple(dict.fromkeys(keys))
        item = self._item(path)
        if item is not None:
            self._show_keys(item)
        if path == self.current_path():
            self._on_current_changed()

    # editing
    def owners(self, key: str, path: str | None = None) -> list[str]:
        """The paths of the actions other than ``path`` with the
        shortcut ``key``."""
        return [
            other
            for other, keys in self.keys.items()
            if other != path and key in keys
        ]

    def add_key(self, key: str, ask: bool = True) -> bool:
        """Add the shortcut ``key`` (portable text) to the selected
        action. If other actions have it, it is taken from them after
        asking (unless ``ask`` is False).

        Returns
        -------
        added : bool
            Whether the shortcut was added.
        """
        path = self.current_path()
        key = _portable(key)
        if path is None or not key:
            return False
        if key in self.keys[path]:
            return False
        owners = self.owners(key, path)
        if owners and ask:
            names = "\n".join(f"  {owner}" for owner in owners)
            reply = QtWidgets.QMessageBox.question(
                self,
                "Shortcut in use",
                f"{native_text([key])} is the shortcut of\n{names}\n\n"
                f"Use it for {path} instead?",
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        for owner in owners:
            self._set_keys(
                owner, (other for other in self.keys[owner] if other != key)
            )
        self._set_keys(path, (*self.keys[path], key))
        return True

    def _add_recorded(self) -> None:
        sequence = self.key_edit.keySequence()
        if sequence.isEmpty():
            return
        key = _portable(sequence)
        # cleared before asking, which would record its own keys
        QtCore.QTimer.singleShot(0, self.key_edit.clear)
        self.add_key(key)

    def _remove(self) -> None:
        """Remove the selected shortcut of the selected action."""
        path = self.current_path()
        item = self.key_list.currentItem()
        if path is None or item is None or item.data(_KEY_ROLE) is None:
            return
        key = item.data(_KEY_ROLE)
        self._set_keys(path, (k for k in self.keys[path] if k != key))

    def _reset_current(self) -> None:
        path = self.current_path()
        if path is not None:
            self._set_keys(path, self.defaults[path])

    def restore_defaults(self) -> None:
        """Show the default shortcuts of all actions; also those of
        actions not in the window are dropped on "OK"."""
        for path in self.keys:
            self._set_keys(path, self.defaults[path])
        self._absent = {}

    def shortcuts(self) -> dict[str, tuple[str, ...]]:
        """The shortcuts that differ from the defaults, by path, as
        ``save_shortcuts`` takes them."""
        shortcuts = dict(self._absent)
        for path, keys in self.keys.items():
            if keys != self.defaults[path]:
                shortcuts[path] = keys
        return shortcuts

    def accept(self) -> None:
        save_shortcuts(self.scope, self.shortcuts())
        super().accept()
