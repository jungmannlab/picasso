"""Test ``picasso.gui.app``, the shared startup sequence of the GUIs.

The point of the module is that errors are reported from the very first
step - in particular while the main window is still being built, which is
where a one-click build fails and used to die without a word.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import sys
import threading

import pytest

from picasso import diagnostics, lib_qt
from picasso.gui import app as gui_app


@pytest.fixture
def isolated(tmp_path, monkeypatch, qapp, restore_theme):
    """Log into ``tmp_path``, skip the update check, and restore the
    global hooks (pytest installs its own) and the look of the
    application afterwards."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_log_file", None, raising=False)

    from picasso import updater

    monkeypatch.setattr(updater, "setup_gui_update_check", lambda window: None)

    hooks = (sys.excepthook, threading.excepthook, sys.unraisablehook)
    try:
        yield
    finally:
        file = diagnostics._log_file
        if file is not None:
            file.close()
        diagnostics._log_file = None
        sys.excepthook, threading.excepthook, sys.unraisablehook = hooks


@pytest.fixture
def shown_errors(monkeypatch):
    """Record the message boxes instead of showing them."""
    shown = []

    class _FakeBox:
        class Icon:
            Critical = None

        def __init__(self, parent=None):
            self.parent = parent
            shown.append(self)

        def setIcon(self, icon):
            pass

        def setWindowTitle(self, title):
            pass

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            pass

        def setDetailedText(self, text):
            self.detailed = text

        def exec(self):
            pass

    monkeypatch.setattr(lib_qt.QtWidgets, "QMessageBox", _FakeBox)
    return shown


def _log_text() -> str:
    return open(diagnostics.log_path(), encoding="utf-8").read()


def test_a_window_that_fails_to_build_is_reported(isolated, shown_errors):
    """The failure the one-click builds used to swallow entirely: an
    error before there is a window to hang a message box on."""

    def window_factory():
        raise RuntimeError("could not load the camera config")

    assert gui_app.run_gui(window_factory, "render") == 1

    assert len(shown_errors) == 1
    assert "could not load the camera config" in shown_errors[0].text
    assert shown_errors[0].parent is None  # no window exists yet
    assert "could not load the camera config" in _log_text()


def _stub_event_loop(monkeypatch, qapp, inspect):
    """Replace ``app.exec`` with ``inspect``, which returns the exit code.

    The tests are about the startup sequence, not about Qt: running the
    real loop and quitting it from a timer would race with the
    ``processEvents`` call inside the error reporting.
    """
    monkeypatch.setattr(qapp, "exec", inspect, raising=False)


def test_the_gui_starts_and_the_hook_is_parented(isolated, qapp, monkeypatch):
    """The whole sequence: window built, plugins loaded, window shown,
    error box parented on it, event loop entered."""
    from PyQt6 import QtWidgets

    windows = []

    def window_factory():
        window = QtWidgets.QMainWindow()
        windows.append(window)
        return window

    loaded = []
    monkeypatch.setattr(
        gui_app, "_load_plugins", lambda w, name: loaded.append(name)
    )

    seen = {}

    def inspect_running_gui():
        # the state Picasso is in when the event loop takes over
        seen["visible"] = windows[0].isVisible()
        seen["signaler"] = lib_qt._error_signaler
        return 0

    _stub_event_loop(monkeypatch, qapp, inspect_running_gui)

    assert gui_app.run_gui(window_factory, "render") == 0
    assert loaded == ["render"]
    assert seen["visible"]
    assert seen["signaler"] is not None  # the excepthook survived startup
    windows[0].close()


def test_a_broken_plugin_does_not_stop_the_gui(
    isolated, qapp, monkeypatch, shown_errors
):
    """A plugin that fails to load is reported and skipped, not fatal."""
    from PyQt6 import QtWidgets

    from picasso.gui import plugins_loader

    def boom(window, name):
        raise ImportError("no module named 'my_plugin'")

    monkeypatch.setattr(plugins_loader, "load_plugins", boom)

    window = QtWidgets.QMainWindow()
    seen = {}

    def inspect_running_gui():
        seen["visible"] = window.isVisible()
        return 0

    _stub_event_loop(monkeypatch, qapp, inspect_running_gui)

    assert gui_app.run_gui(lambda: window, "render") == 0
    assert seen["visible"]  # started anyway
    assert "no module named 'my_plugin'" in _log_text()
    assert len(shown_errors) == 1  # and the user was told
    window.close()


def test_the_saved_theme_is_applied_before_the_window_is_built(
    isolated, qapp, monkeypatch
):
    """The window is laid out in the theme's style and colors."""
    from PyQt6 import QtWidgets

    from picasso import io
    from picasso.gui import theme

    settings = io.load_user_settings()
    settings["Appearance"] = theme.Appearance(mode="Dark").to_settings()
    io.save_user_settings(settings)
    monkeypatch.setattr(gui_app, "_load_plugins", lambda w, name: None)
    seen = {}

    def window_factory():
        seen["window"] = qapp.palette().window().color().name()
        return QtWidgets.QMainWindow()

    _stub_event_loop(monkeypatch, qapp, lambda: 0)

    assert gui_app.run_gui(window_factory, "render") == 0
    assert seen["window"] == theme.NEUTRALS["Dark"]["window"]
    assert theme.applied() == theme.Appearance(mode="Dark")


def test_a_failing_theme_does_not_stop_the_gui(
    isolated, qapp, monkeypatch, shown_errors
):
    """An error while applying the look is reported, not fatal."""
    from PyQt6 import QtWidgets

    from picasso.gui import theme

    def boom():
        raise RuntimeError("the theme broke")

    monkeypatch.setattr(theme, "current", boom)
    monkeypatch.setattr(gui_app, "_load_plugins", lambda w, name: None)
    window = QtWidgets.QMainWindow()
    _stub_event_loop(monkeypatch, qapp, lambda: 0)

    assert gui_app.run_gui(lambda: window, "render") == 0
    assert "the theme broke" in _log_text()
    assert len(shown_errors) == 1
    window.close()
