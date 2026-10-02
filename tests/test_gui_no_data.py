"""Menu and toolbar actions of the Picasso GUIs without data.

Every action of every GUI is triggered on a freshly opened window, with
nothing loaded; none may raise. An action that needs data tells the
user so in a message box instead. The file and message dialogs are
answered right away (cancel), so nothing waits for input and nothing is
written.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib

import pytest
from PyQt6 import QtGui, QtWidgets

from picasso import io
from picasso.gui import toolbars

APPS = {
    "render": "Window",
    "localize": "Window",
    "filter": "Window",
    "average": "Window",
    "average3": "Window",
    "design": "MainWindow",
    "simulate": "Window",
    "spinna": "Window",
}

#: actions that end the session rather than act on data
SKIP = ("quit", "exit", "close")

LOCS_PATH = "./tests/data/testdata_locs.hdf5"


@pytest.fixture
def messages(qt_offscreen, tmp_path, monkeypatch):
    """Answer every modal dialog at once, recording the message boxes
    as ``(title, text)``; user settings go to ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    shown = []

    def box(kind):
        def show(parent, title, text, *args, **kwargs):
            shown.append((title, text))
            if kind == "question":
                return QtWidgets.QMessageBox.StandardButton.No
            return QtWidgets.QMessageBox.StandardButton.Ok

        return staticmethod(show)

    def box_exec(self):
        shown.append((self.windowTitle(), self.text()))
        return QtWidgets.QMessageBox.StandardButton.Ok

    MessageBox = QtWidgets.QMessageBox
    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(MessageBox, kind, box(kind))
    monkeypatch.setattr(MessageBox, "exec", box_exec)
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: 0)
    FileDialog = QtWidgets.QFileDialog
    for name, value in (
        ("getOpenFileName", ("", "")),
        ("getOpenFileNames", ([], "")),
        ("getSaveFileName", ("", "")),
        ("getExistingDirectory", ""),
    ):
        monkeypatch.setattr(
            FileDialog, name, staticmethod(lambda *a, v=value, **k: v)
        )
    InputDialog = QtWidgets.QInputDialog
    for name, value in (
        ("getInt", 0),
        ("getDouble", 0.0),
        ("getText", ""),
        ("getItem", ""),
    ):
        monkeypatch.setattr(
            InputDialog,
            name,
            staticmethod(lambda *a, v=value, **k: (v, False)),
        )
    monkeypatch.setattr(
        QtGui.QDesktopServices, "openUrl", staticmethod(lambda *a: True)
    )
    return shown


def _window(name: str) -> QtWidgets.QWidget:
    module = importlib.import_module(f"picasso.gui.{name}")
    window = getattr(module, APPS[name])()
    window.show()
    return window


def _actions(window: QtWidgets.QWidget) -> dict[str, QtGui.QAction]:
    """The enabled actions of the window's menus and toolbars by their
    path."""
    bar = window.findChild(QtWidgets.QMenuBar)
    actions = toolbars.menu_actions(bar) if bar is not None else {}
    for toolbar in window.findChildren(QtWidgets.QToolBar):
        for action in toolbar.actions():
            if action.isSeparator() or action in actions.values():
                continue
            path = f"[{toolbar.windowTitle()}] {action.text()}"
            actions.setdefault(path, action)
    return {
        path: action
        for path, action in actions.items()
        if action.isEnabled()
        and not any(word in path.lower() for word in SKIP)
    }


def _trigger_all(
    qapp: QtWidgets.QApplication,
    window: QtWidgets.QWidget,
    skip: tuple[str, ...] = (),
) -> None:
    """Trigger every action, closing what each one opens."""
    keep = set(qapp.topLevelWidgets())
    actions = _actions(window)
    assert actions
    for path, action in actions.items():
        if any(word in path.lower() for word in skip):
            continue
        action.trigger()
        qapp.processEvents()
        for widget in qapp.topLevelWidgets():
            if widget not in keep:
                widget.close()
        qapp.processEvents()


@pytest.mark.parametrize("name", list(APPS))
def test_actions_without_data_do_not_raise(name, messages, qt_offscreen):
    window = _window(name)
    _trigger_all(qt_offscreen, window)
    if name == "render":  # its 3D window has menus of its own
        _trigger_all(qt_offscreen, window.window_rot)
    window.close()


def test_render_file_actions_say_no_files_loaded(messages, qt_offscreen):
    window = _window("render")
    _trigger_all(qt_offscreen, window)
    assert ("Export current view", "No files loaded.") in messages
    assert ("Save pick regions", "No files loaded.") in messages
    window.close()


def test_localize_analysis_asks_for_a_movie(messages, qt_offscreen):
    window = _window("localize")
    window.identify()
    window.localize()
    window.calibrate_z()
    assert [title for title, _ in messages] == [
        "Identify",
        "Localize",
        "Calibrate astigmatism",
    ]
    window.close()


def test_localize_identifications_ask_for_a_movie(
    messages, qt_offscreen, monkeypatch
):
    # a file is chosen, so a missing check would get to loading it
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (LOCS_PATH, "")),
    )
    window = _window("localize")
    window.open_identifications()
    window.open_picks()
    window.open_locs()
    assert [title for title, _ in messages] == [
        "Load identifications",
        "Load picks as identifications",
        "Load locs as identifications",
    ]
    assert window.identifications is None
    window.close()


def test_filter_actions_say_no_file_loaded(messages, qt_offscreen):
    window = _window("filter")
    window.save_file_dialog()
    window.export_csv_dialog()
    window.plot_histogram()
    window.plot_hist2d()
    window.plot_subclustering()
    window.remove_columns()
    window.filter_num.filter()
    assert messages == [
        (title, "No file loaded.")
        for title in (
            "Save",
            "Export as CSV",
            "Histogram",
            "2D Histogram",
            "Test subclustering",
            "Remove columns",
            "Filter",
        )
    ]
    window.close()


def test_filter_histograms_ask_for_columns(messages, qt_offscreen):
    window = _window("filter")
    window.open(LOCS_PATH)
    window.plot_histogram()
    window.plot_hist2d()
    assert [title for title, _ in messages] == ["Histogram", "2D Histogram"]
    window.close()


def test_render_load_picks_without_data(messages, qt_offscreen):
    window = _window("render")
    window.load_picks()
    assert messages == [("Load pick regions", "No files loaded.")]
    window.close()


def test_average_does_not_keep_ungrouped_locs(messages, qt_offscreen):
    window = _window("average")
    window.view.open(LOCS_PATH)  # has no group column
    assert not hasattr(window.view, "locs")
    window.view.average()
    assert messages[-1] == ("Average", "No file loaded.")
    window.close()


@pytest.fixture
def render_window(messages, qt_offscreen):
    """Render with one channel loaded, no picks and no drift."""
    window = _window("render")
    locs, info = io.load_locs(LOCS_PATH)
    window.view.add(LOCS_PATH, locs, info, render_=False)
    window.view.fit_in_view(autoscale=True)
    qt_offscreen.processEvents()
    yield window
    window.close()


def test_render_actions_with_data_do_not_raise(render_window, qt_offscreen):
    # removing the data rebuilds the menus being walked
    _trigger_all(qt_offscreen, render_window, skip=("remove all",))


def test_render_move_to_pick_without_picks(render_window, messages):
    render_window.view.move_to_pick()
    assert messages[-1][0] == "Pick Error"


def test_render_undo_drift_without_drift(render_window, messages):
    render_window.view.undo_drift()
    assert messages[-1][0] == "Undo drift"
