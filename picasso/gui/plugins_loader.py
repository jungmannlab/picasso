"""
picasso.gui.plugins_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discovery and loading of user plugins from ``~/.picasso/plugins``.

A plugin is a single ``.py`` file defining a ``Plugin`` class with an
``__init__(self, window)`` that sets ``self.name`` (the target GUI app,
e.g. ``"render"``) and ``self.window``, plus an ``execute(self)`` method
that adds actions to ``window.plugin_menu``. See ``plugin_template.py``.

Loading is deliberately tolerant: a broken plugin prints a traceback and
is skipped so that it can never crash app startup.

:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import importlib.util
import os
import traceback

from .. import io


def _discover_plugin_files() -> list[str]:
    """Return sorted ``.py`` files in the user plugins directory, skipping
    files whose name starts with ``_``."""
    directory = io.plugins_directory()
    files = sorted(glob.glob(os.path.join(directory, "*.py")))
    return [f for f in files if not os.path.basename(f).startswith("_")]


def _load_module_from_path(path: str):
    """Import a standalone ``.py`` file that is not part of any package."""
    name = "picasso_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plugins(window, app_name: str) -> list:
    """Discover, instantiate and execute plugins matching ``app_name``.

    Always sets ``window.plugins`` to the (possibly empty) list of
    ``Plugin`` instances whose ``name`` matches ``app_name`` (after a
    successful ``execute``), and returns that list. A failure in any single
    plugin is logged and the plugin skipped.

    Parameters
    ----------
    window
        The GUI main window; must expose ``plugin_menu``.
    app_name : str
        The GUI app name to filter by (e.g. ``"render"``).
    """
    plugins: list = []
    for path in _discover_plugin_files():
        try:
            module = _load_module_from_path(path)
            plugin = module.Plugin(window)
            if getattr(plugin, "name", None) == app_name:
                plugin.execute()
                plugins.append(plugin)
        except Exception:
            print(f"Failed to load plugin {path!r}:")
            traceback.print_exc()
    window.plugins = plugins
    return plugins


def add_plugins_menu_actions(window, app_name: str) -> None:
    """Append a separator plus 'Open plugins folder...' and 'Reload plugins'
    actions to ``window.plugin_menu``."""
    from PyQt6 import QtCore, QtGui

    menu = window.plugin_menu
    menu.addSeparator()

    open_action = menu.addAction("Open plugins folder...")
    open_action.triggered.connect(
        lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(io.plugins_directory())
        )
    )

    reload_action = menu.addAction("Reload plugins")
    reload_action.triggered.connect(lambda: _reload(window, app_name))


def _reload(window, app_name: str) -> None:
    """Clear the plugins menu and re-load plugins from disk."""
    window.plugin_menu.clear()
    load_plugins(window, app_name)
    add_plugins_menu_actions(window, app_name)
