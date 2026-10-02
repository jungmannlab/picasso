"""
picasso.gui.plugins_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qt side of the user plugin system: loading the plugins that extend a
Picasso GUI, the Plugins menu, and the store dialog for browsing,
installing and enabling plugins from the online registry.

Everything that does not need Qt — discovery, the enable-state sidecar,
the registry client and the hooks that let a plugin extend the Python
API and the command line — lives in :mod:`picasso.plugins` and is
re-exported here, so existing code importing it from this module keeps
working.

A GUI plugin is a single ``.py`` file defining a ``Plugin`` class with an
``__init__(self, window)`` that sets ``self.name`` (the target GUI app,
e.g. ``"render"``) and ``self.window``, plus an ``execute(self)`` method
that adds actions to ``window.plugin_menu``. The same file may also
export ``PICASSO_API`` and ``register_cli`` to extend Picasso outside the
GUI; see :mod:`picasso.plugins` and ``plugin_template.py``.

Plugins are executed as ordinary Python inside the Picasso process, so
they cannot be sandboxed: a loaded plugin can do anything Picasso can.
The safeguards therefore aim at *provenance and reviewability* rather
than containment — nothing loads until it has been enabled, registry
downloads are pinned to a ``sha256``, and the dialogs below can show a
plugin's source before installing it and a diff before updating it.

Loading is deliberately tolerant: a broken plugin prints a traceback and
is skipped so that it can never crash app startup.

:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import traceback

from PyQt6 import QtCore, QtGui, QtWidgets

from .. import io
from . import theme
from . import toolbars

# Re-exported for backwards compatibility with code and tests that
# imported these from this module before they moved to picasso.plugins.
from ..plugins import (  # noqa: F401
    API_APP,
    BRANCH,
    INCOMPATIBLE,
    LOCAL_ONLY,
    MANIFEST_URL,
    NOT_INSTALLED,
    ORPHAN,
    RAW_BASE,
    REPO,
    REPO_URL,
    UNMANAGED,
    UNVERIFIED,
    UPDATE_AVAILABLE,
    UP_TO_DATE,
    _discover_plugin_files,
    _get,
    _is_usable_entry,
    _load_module_from_path,
    _local_filename,
    _sidecar_path,
    _SIDECAR,
    _TIMEOUT,
    clear_module_cache,
    compare_versions,
    diff_sources,
    disabled_plugin_files,
    download_source,
    entry_filename,
    fetch_manifest,
    install,
    is_compatible,
    is_enabled,
    is_safe_filename,
    is_safe_id,
    is_safe_repo_path,
    is_valid_digest,
    load_state,
    merged_entries,
    parse_version,
    plugin_path,
    read_local_source,
    save_state,
    set_enabled,
    sha256_bytes,
    status_for,
    uninstall,
)


# =============================================================================
# Loading GUI plugins
# =============================================================================


def load_plugins(window, app_name: str) -> list:
    """Discover, instantiate and execute enabled plugins matching ``app_name``.

    Only files that have been explicitly enabled are executed; a ``.py``
    file that has merely been copied into the plugins folder is discovered
    but not run until the user enables it (via the plugin store dialog),
    because loading it means running it.

    Always sets ``window.plugins`` to the (possibly empty) list of
    ``Plugin`` instances whose ``name`` matches ``app_name`` (after a
    successful ``execute``), and returns that list. A failure in any single
    plugin is logged and the plugin skipped.

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        The GUI main window; must expose ``plugin_menu``.
    app_name : str
        The GUI app name to filter by (e.g. ``"render"``).

    Returns
    -------
    plugins : list
        The ``Plugin`` instances that matched ``app_name`` and executed
        successfully; also assigned to ``window.plugins``.
    """
    plugins: list = []
    state = load_state()
    for path in _discover_plugin_files():
        if not is_enabled(state, os.path.basename(path)):
            continue
        try:
            module = _load_module_from_path(path)
            plugin_class = getattr(module, "Plugin", None)
            if plugin_class is None:
                # A plugin that only extends the API or the command line
                # has no GUI half; that is not an error.
                continue
            plugin = plugin_class(window)
            if getattr(plugin, "name", None) == app_name:
                plugin.execute()
                plugins.append(plugin)
        except Exception:
            print(f"Failed to load plugin {path!r}:")
            traceback.print_exc()
    window.plugins = plugins
    return plugins


def execute_plugins(window) -> None:
    """Re-run ``execute`` on the plugins already loaded into ``window``.

    Used when the menu bar is rebuilt (e.g. "Remove all localizations") to
    re-add plugin actions to the new ``window.plugin_menu`` without
    re-discovering plugins from disk. Tolerant: a failing plugin is logged
    and skipped.
    """
    for plugin in getattr(window, "plugins", []):
        try:
            plugin.execute()
        except Exception:
            print(f"Failed to execute plugin {plugin!r}:")
            traceback.print_exc()


def add_plugins_menu_actions(window, app_name: str) -> None:
    """Append a separator plus the plugin management actions to
    ``window.plugin_menu``.

    If plugin files are present but not enabled, a leading entry says so
    and opens the store dialog, where they can be reviewed and enabled.
    """
    menu = window.plugin_menu
    menu.addSeparator()

    pending = disabled_plugin_files()
    if pending:
        count = len(pending)
        noun = "file" if count == 1 else "files"
        pending_action = menu.addAction(
            f"{count} plugin {noun} found but not enabled..."
        )
        pending_action.setIcon(theme.icon("plugins-browse"))
        pending_action.triggered.connect(lambda: _open_store(window, app_name))

    store_action = menu.addAction("Browse online plugins...")
    store_action.setIcon(theme.icon("plugins-browse"))
    store_action.triggered.connect(lambda: _open_store(window, app_name))

    open_action = menu.addAction("Open plugins folder...")
    open_action.setIcon(theme.icon("open"))
    open_action.triggered.connect(
        lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(io.plugins_directory())
        )
    )

    reload_action = menu.addAction("Reload plugins")
    reload_action.setIcon(theme.icon("reload"))
    reload_action.triggered.connect(lambda: reload_plugins(window, app_name))

    # show the plugins' actions on a toolbar customized to have them
    toolbars.refresh(window)


def _open_store(window, app_name: str) -> None:
    """Open the online plugin store dialog for ``app_name``."""
    PluginStoreDialog(window, app_name).exec()


def reload_plugins(window, app_name: str) -> None:
    """Clear the plugins menu and re-load plugins from disk.

    Drops the imported-module cache first, so a plugin file edited while
    Picasso is running is picked up.
    """
    clear_module_cache()
    window.plugin_menu.clear()
    load_plugins(window, app_name)
    add_plugins_menu_actions(window, app_name)


# Backwards-compatible alias for the previous private name.
_reload = reload_plugins


# =============================================================================
# Online plugin store
# =============================================================================

# --- Dialogs -----------------------------------------------------------------

_STATUS_TEXT = {
    NOT_INSTALLED: "Not installed",
    UP_TO_DATE: "Installed",
    UPDATE_AVAILABLE: "Update available",
    INCOMPATIBLE: "Requires newer Picasso",
    UNVERIFIED: "No integrity hash",
    UNMANAGED: "On disk, not installed by Picasso",
    LOCAL_ONLY: "Local file",
    ORPHAN: "Installed (unlisted)",
}


class SourceViewDialog(QtWidgets.QDialog):
    """Read-only viewer for a plugin's source or an update's diff."""

    def __init__(self, parent, title: str, header: str, text: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 600)

        layout = QtWidgets.QVBoxLayout(self)

        label = QtWidgets.QLabel(header)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(label)

        view = QtWidgets.QPlainTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        view.setFont(
            QtGui.QFontDatabase.systemFont(
                QtGui.QFontDatabase.SystemFont.FixedFont
            )
        )
        view.setPlainText(text)
        layout.addWidget(view)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)


class PluginStoreDialog(QtWidgets.QDialog):
    """Browse and manage online plugins for a single Picasso app."""

    def __init__(self, window, app_name: str):
        super().__init__(window)
        self.window = window
        self.app_name = app_name
        self.state = load_state()
        self.manifest: list[dict] = []

        self.setWindowTitle("Online plugins")
        self.resize(900, 500)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.show_all = QtWidgets.QCheckBox(
            "Show plugins for all Picasso apps"
        )
        self.show_all.stateChanged.connect(self._populate)
        layout.addWidget(self.show_all)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Plugin", "App", "Description", "Status", "Enabled", "Action"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        repo_btn = QtWidgets.QPushButton("Open repository...")
        repo_btn.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(REPO_URL))
        )
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(repo_btn)
        buttons.addStretch()
        buttons.addWidget(refresh_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._refresh()

    # -- data -----------------------------------------------------------------

    def _refresh(self) -> None:
        """(Re)download the manifest and rebuild the table."""
        self.state = load_state()
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            self.manifest = fetch_manifest()
            self.status_label.setText(
                f"Plugins available from {REPO}. Downloads are checked "
                "against the hash published in the registry, but a plugin "
                "still runs with full access to your computer once enabled — "
                "use 'View source' and only enable plugins you trust."
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.manifest = []
            self.status_label.setText(
                "Could not reach the online plugin registry "
                f"({exc}). Check your connection and press Refresh."
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._populate()

    def _visible_entries(self) -> list[dict]:
        """The entries to show, filtered to this app unless 'show all' is on.

        Plugins marked ``app: "api"`` extend the Python API or the command
        line rather than one GUI, so they are relevant from wherever the
        dialog was opened and are always shown.
        """
        entries = merged_entries(self.manifest, self.state)
        if not self.show_all.isChecked():
            entries = [
                e
                for e in entries
                if e.get("app") in (self.app_name, API_APP)
                or e.get("_orphan")
                or e.get("_local")
            ]
        return entries

    def _populate(self) -> None:
        entries = self._visible_entries()
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            status = status_for(entry, self.state)

            name_item = QtWidgets.QTableWidgetItem(
                str(entry.get("display_name", entry["id"]))
            )
            author = entry.get("author")
            version = entry.get("version")
            tip = []
            if author:
                tip.append(f"Author: {author}")
            if version:
                tip.append(f"Latest version: {version}")
            if entry.get("sha256"):
                tip.append(f"sha256: {entry['sha256']}")
            if tip:
                name_item.setToolTip("\n".join(tip))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(entry.get("app") or "")
            )
            self.table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem(entry.get("description", "")),
            )

            status_text = _STATUS_TEXT.get(status, status)
            installed = self.state["plugins"].get(entry["id"])
            if status == UPDATE_AVAILABLE and installed:
                status_text += f" ({installed.get('version')} → {version})"
            status_item = QtWidgets.QTableWidgetItem(status_text)
            if status == UNVERIFIED:
                status_item.setToolTip(
                    "This registry entry does not publish a sha256 hash, so "
                    "the download cannot be verified. Installing is disabled."
                )
            self.table.setItem(row, 3, status_item)

            self.table.setCellWidget(row, 4, self._enabled_widget(entry))
            self.table.setCellWidget(
                row, 5, self._action_widget(entry, status)
            )
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

    # -- enable / disable -----------------------------------------------------

    def _enabled_widget(self, entry: dict) -> QtWidgets.QWidget:
        """A checkbox controlling whether the file is loaded at startup.

        Only meaningful for files that exist on disk; otherwise a placeholder
        is shown.
        """
        container = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(container)
        box.setContentsMargins(2, 2, 2, 2)

        filename = entry_filename(entry)
        on_disk = False
        if filename:
            try:
                on_disk = os.path.exists(plugin_path(filename))
            except ValueError:
                on_disk = False
        if not on_disk:
            box.addWidget(QtWidgets.QLabel("—"))
            return container

        check = QtWidgets.QCheckBox()
        check.setChecked(is_enabled(self.state, filename))
        check.setToolTip(
            "Enabled plugins are imported and run every time the app starts."
        )
        check.clicked.connect(
            lambda checked, f=filename: self._toggle_enabled(f, checked)
        )
        box.addWidget(check)
        return container

    def _toggle_enabled(self, filename: str, enable: bool) -> None:
        if enable and not self._confirm_enable(filename):
            self._populate()
            return
        set_enabled(self.state, filename, enable)
        self._reload_app()
        self._populate()

    def _confirm_enable(self, filename: str) -> bool:
        """Confirm enabling a file that Picasso did not install itself.

        Registry installs are already confirmed at install time, so only
        untracked files ask again — those are the ones that arrived by hand.
        """
        tracked = any(
            rec.get("file") == filename
            for rec in self.state["plugins"].values()
        )
        if tracked:
            return True
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Enable plugin?",
            f"'{filename}' did not come from the online plugin registry, so "
            "its contents cannot be verified.\n\nEnabling it will run this "
            "Python file with full access to your computer every time the app "
            "starts. Only enable files you have reviewed or whose author you "
            "trust.\n\nEnable it?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    # -- per-row actions ------------------------------------------------------

    def _action_widget(self, entry: dict, status: str) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(container)
        box.setContentsMargins(2, 2, 2, 2)
        box.setSpacing(4)

        def add(label, slot):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            box.addWidget(btn)
            return btn

        if status == NOT_INSTALLED:
            add("Install", lambda: self._install(entry))
            add("View source", lambda: self._view_remote(entry))
        elif status == UNMANAGED:
            add("Install", lambda: self._install(entry))
            add("View source", lambda: self._view_local(entry))
        elif status == UPDATE_AVAILABLE:
            add("Update", lambda: self._install(entry))
            add("View changes", lambda: self._view_diff(entry))
            add("Uninstall", lambda: self._uninstall(entry))
        elif status in (UP_TO_DATE, ORPHAN):
            add("View source", lambda: self._view_local(entry))
            add("Uninstall", lambda: self._uninstall(entry))
        elif status == LOCAL_ONLY:
            add("View source", lambda: self._view_local(entry))
        elif status == UNVERIFIED:
            label = QtWidgets.QLabel("—")
            label.setToolTip(
                "This plugin cannot be installed until the registry "
                "publishes a sha256 hash for it."
            )
            box.addWidget(label)
        elif status == INCOMPATIBLE:
            label = QtWidgets.QLabel("—")
            label.setToolTip(
                "This plugin requires a newer version of Picasso "
                f"(>= {entry.get('min_picasso_version')})."
            )
            box.addWidget(label)
        box.addStretch()
        return container

    # -- source viewing -------------------------------------------------------

    def _view_remote(self, entry: dict) -> None:
        """Show the source that would be installed, hash-verified."""
        try:
            content = download_source(entry)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QtWidgets.QMessageBox.critical(
                self, "Could not show source", str(exc)
            )
            return
        self._show_source(
            f"Source of {entry.get('display_name', entry['id'])}",
            f"{RAW_BASE}/{entry['file']}\nsha256 {sha256_bytes(content)} "
            "(matches the registry)",
            content.decode("utf-8", errors="replace"),
        )

    def _view_local(self, entry: dict) -> None:
        """Show the source of the file currently on disk."""
        filename = entry_filename(entry)
        try:
            text = read_local_source(filename)
            path = plugin_path(filename)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not show source", str(exc)
            )
            return
        self._show_source(f"Source of {filename}", path, text)

    def _view_diff(self, entry: dict) -> None:
        """Show what an update would change in the installed file."""
        filename = entry_filename(entry)
        try:
            new = download_source(entry).decode("utf-8", errors="replace")
            old = read_local_source(filename)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QtWidgets.QMessageBox.critical(
                self, "Could not show changes", str(exc)
            )
            return
        installed = self.state["plugins"].get(entry["id"], {})
        self._show_source(
            f"Changes to {filename}",
            f"{installed.get('version')} → {entry.get('version')}",
            diff_sources(old, new, filename),
        )

    def _show_source(self, title: str, header: str, text: str) -> None:
        SourceViewDialog(self, title, header, text).exec()

    # -- install / uninstall --------------------------------------------------

    def _confirm_trust(self) -> bool:
        """Show the one-time trust warning before the first install."""
        if self.state.get("trust_acknowledged"):
            return True
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Install plugin?",
            "Plugins are Python files that run with full access to your "
            "computer every time the app starts. Downloads are verified "
            "against the hash published in the registry, but that only proves "
            "the file is the published one — not that it is safe. Only "
            "install plugins from sources you trust; 'View source' shows the "
            "code first.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        self.state["trust_acknowledged"] = True
        save_state(self.state)
        return True

    def _confirm_overwrite(self, entry: dict, status: str) -> bool:
        """Confirm replacing a file Picasso did not install itself."""
        if status != UNMANAGED:
            return True
        filename = entry_filename(entry)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Replace existing file?",
            f"'{filename}' already exists in your plugins folder but was not "
            "installed by Picasso. Installing will overwrite it with the "
            "version from the registry.\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _install(self, entry: dict) -> None:
        status = status_for(entry, self.state)
        if not self._confirm_trust():
            return
        if not self._confirm_overwrite(entry, status):
            return
        try:
            install(entry, self.state)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QtWidgets.QMessageBox.critical(
                self, "Installation failed", str(exc)
            )
            return
        self._reload_app()
        self._populate()

    def _uninstall(self, entry: dict) -> None:
        uninstall(entry["id"], self.state)
        self._reload_app()
        self._populate()

    def _reload_app(self) -> None:
        """Re-discover plugins in the host window so changes take effect."""
        reload_plugins(self.window, self.app_name)
