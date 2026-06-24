"""Test ``picasso.gui.plugins_loader`` — discovery and loading of user
plugins from ``~/.picasso/plugins``.

Plugins are written as temporary ``.py`` files into a ``tmp_path`` and the
loader is pointed there by monkeypatching ``picasso.io.plugins_directory``.
A tiny fake window stands in for the GUI so the loader is exercised without
a ``QApplication`` (the loader keeps Qt imports inside the menu helper).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import types

from picasso import io
from picasso.gui import plugins_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMenu:
    """Minimal stand-in for ``window.plugin_menu`` used by plugins."""

    def __init__(self):
        self.actions = []

    def addAction(self, label):
        self.actions.append(label)
        return types.SimpleNamespace(
            triggered=types.SimpleNamespace(connect=lambda *a, **k: None)
        )

    def addSeparator(self):
        pass

    def clear(self):
        self.actions = []


def _make_window():
    return types.SimpleNamespace(plugin_menu=FakeMenu())


def _write_plugin(directory, filename, app_name, body=""):
    """Write a plugin ``.py`` file whose ``execute`` records the menu label
    'loaded-<name>' so the test can assert it ran."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(
            "class Plugin:\n"
            "    def __init__(self, window):\n"
            f"        self.name = {app_name!r}\n"
            "        self.window = window\n"
            "    def execute(self):\n"
            "        label = 'loaded-' + self.name\n"
            "        self.window.plugin_menu.addAction(label)\n"
            f"{body}"
        )
    return path


def _point_loader_at(monkeypatch, directory):
    monkeypatch.setattr(io, "plugins_directory", lambda: directory)


# ---------------------------------------------------------------------------
# plugins_directory
# ---------------------------------------------------------------------------


class TestPluginsDirectory:
    def test_created_under_dot_picasso(self, tmp_path, monkeypatch):
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
        directory = io.plugins_directory()
        assert directory == os.path.join(str(tmp_path), ".picasso", "plugins")
        assert os.path.isdir(directory)


# ---------------------------------------------------------------------------
# load_plugins
# ---------------------------------------------------------------------------


class TestLoadPlugins:
    def test_loads_and_executes_matching_plugin(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "myplugin.py", "render")

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert len(result) == 1
        assert window.plugins is result
        assert "loaded-render" in window.plugin_menu.actions

    def test_filters_by_app_name(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "render_plugin.py", "render")

        window = _make_window()
        plugins_loader.load_plugins(window, "localize")

        assert window.plugins == []
        assert "loaded-render" not in window.plugin_menu.actions

    def test_skips_underscore_files(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))
        _write_plugin(str(tmp_path), "_helper.py", "render")

        window = _make_window()
        plugins_loader.load_plugins(window, "render")

        assert window.plugins == []

    def test_broken_plugin_is_skipped(self, tmp_path, monkeypatch, capsys):
        _point_loader_at(monkeypatch, str(tmp_path))
        # execute raises -> must be caught, logged, and skipped
        _write_plugin(
            str(tmp_path),
            "broken.py",
            "render",
            body="        raise RuntimeError('boom')\n",
        )

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert result == []
        assert "Failed to load plugin" in capsys.readouterr().out

    def test_empty_directory_returns_empty_list(self, tmp_path, monkeypatch):
        _point_loader_at(monkeypatch, str(tmp_path))

        window = _make_window()
        result = plugins_loader.load_plugins(window, "render")

        assert result == []
        assert window.plugins == []
