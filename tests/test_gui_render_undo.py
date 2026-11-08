"""Undo and Redo in Picasso: Render (``picasso.gui.render_undo``).

Covers the steps recorded for the actions that change the
localizations (the Move tool, Apply expression, removing localizations
in picks, applying drift, removing columns, synchronizing and unfolding
groups, linking) and the picks and measured points, what a step keeps
(only the columns that changed, arrays that keep neither changing nor
the current localizations in memory, the disk for large steps), picks,
points and the view following
an undone translation of the canvas, when the history is forgotten or
remapped, linked windows, the actions in the File menu, and the guards
that every method of Render that changes the localizations, the picks
or the measured points records a step.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import ast
import gc
import inspect
import time
import weakref

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io, lib, lib_qt, postprocess
from picasso.gui import idle, render_link, render_undo, toolbars
from picasso.gui import render as gui_render
from tests.test_gui_render_index import _EXEMPT, _mutating_methods

WIDTH = HEIGHT = 32
PIXELSIZE = 130.0
#: the view is 256 display pixels wide and shows the whole 32 pixel
#: FOV, i.e., 8 display pixels per camera pixel
SCALE = 8
#: float32 coordinates of a few thousand nm resolve about 0.001 nm
ATOL = 1e-2


def _locs(n: int = 500, seed: int = 0, groups: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locs = pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.uint32),
            "x": (rng.uniform(1.0, WIDTH - 1, size=n) * PIXELSIZE).astype(
                np.float32
            ),
            "y": (rng.uniform(1.0, HEIGHT - 1, size=n) * PIXELSIZE).astype(
                np.float32
            ),
            "lpx": np.full(n, 0.1 * PIXELSIZE, dtype=np.float32),
            "lpy": np.full(n, 0.1 * PIXELSIZE, dtype=np.float32),
            "photons": np.full(n, 1000.0, dtype=np.float32),
        }
    )
    if groups:
        locs["group"] = rng.integers(0, groups, size=n).astype(np.int32)
    return locs


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


class _Event:
    """The parts of a Qt mouse event the handlers read."""

    def __init__(self, x, y, button=QtCore.Qt.MouseButton.LeftButton):
        self._pos = QtCore.QPoint(int(x), int(y))
        self._button = button

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def buttons(self):
        return self._button

    def modifiers(self):
        return QtCore.Qt.KeyboardModifier.NoModifier

    def accept(self):
        pass

    def ignore(self):
        pass


def _drag(view, dx, dy, start=(128, 128)):
    """Drag with the Move tool by ``(dx, dy)`` camera pixels."""
    mode = view._mode
    view._mode = "Move"
    x0, y0 = start
    x1, y1 = x0 + dx * SCALE, y0 + dy * SCALE
    view.mousePressEvent(_Event(x0, y0))
    view.mouseMoveEvent(_Event(x1, y1))
    view.mouseReleaseEvent(_Event(x1, y1))
    view._mode = mode


def _add(window, path, locs) -> None:
    window.view.add(str(path), locs, _info(), render_=False)


def _close(window) -> None:
    window.view.stop_render_worker()
    window.window_rot.view_rot.stop_render_worker()
    window.close()


def _show(window) -> None:
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    window.view.resize(256, 256)


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding two channels."""
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        _add(window, tmp_path / f"locs{i}.hdf5", _locs(seed=i))
    _show(window)
    yield window
    _close(window)


@pytest.fixture
def single(qt_offscreen, tmp_path):
    """A Render window holding one channel."""
    window = gui_render.Window(plugins_loaded=True)
    _add(window, tmp_path / "locs.hdf5", _locs())
    _show(window)
    yield window
    _close(window)


def _wait_for_tasks(qapp, timeout: float = 60.0) -> None:
    start = time.monotonic()
    while lib_qt._running_tasks:
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not end")
    qapp.processEvents()


def _steps(view) -> tuple[list, list]:
    return view.history.steps()


def _copies(view) -> list[pd.DataFrame]:
    return [locs.copy() for locs in view.locs]


def _assert_locs(view, expected: list[pd.DataFrame]) -> None:
    assert len(view.locs) == len(expected)
    for locs, frame in zip(view.locs, expected):
        pd.testing.assert_frame_equal(locs, frame)


@pytest.fixture
def messages(monkeypatch) -> list[str]:
    """Texts of the information and warning boxes shown."""
    texts = []

    def record(parent, title, text, *args, **kwargs):
        texts.append(text)

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", record)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", record)
    return texts


# ---------------------------------------------------------------------------
# The actions
# ---------------------------------------------------------------------------


def test_nothing_to_undo_after_loading(window):
    assert window.view.history.undo_label is None
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    before = _copies(window.view)
    window.view.undo()
    window.view.redo()
    _assert_locs(window.view, before)


def test_file_menu_toolbar_and_shortcuts(window):
    menus = toolbars.menu_actions(window.menu_bar)
    assert menus["File > Undo"] is window.undo_action
    assert menus["File > Redo"] is window.redo_action
    # no menu of their own
    assert "Edit" not in [a.text() for a in window.menu_bar.actions()]
    for action, key in (
        (window.undo_action, QtGui.QKeySequence.StandardKey.Undo),
        (window.redo_action, QtGui.QKeySequence.StandardKey.Redo),
    ):
        # the platform's first binding, e.g., Ctrl+Z (Cmd+Z on macOS)
        assert action.shortcut() == QtGui.QKeySequence.keyBindings(key)[0]
    # Ctrl+Z was the Zoom tool's, which now has Z alone
    zoom = next(
        a for a in window.tools_actiongroup.actions() if a.text() == "Zoom"
    )
    assert zoom.shortcut() == QtGui.QKeySequence("Z")
    toolbar = window.toolbar.actions()
    assert window.undo_action in toolbar and window.redo_action in toolbar
    # the old undos are gone
    assert not any(path.endswith("Undo drift") for path in menus)
    assert not hasattr(window.tools_settings_dialog, "move_undo_button")


def test_single_keys_select_the_tools(window, qt_offscreen):
    from PyQt6 import QtTest

    tools = {a.text(): a for a in window.tools_actiongroup.actions()}
    window.show()
    window.activateWindow()
    qt_offscreen.processEvents()
    for key, tool in (
        (QtCore.Qt.Key.Key_P, "Pick"),
        (QtCore.Qt.Key.Key_M, "Measure"),
        (QtCore.Qt.Key.Key_G, "Move"),
        (QtCore.Qt.Key.Key_Z, "Zoom"),
    ):
        QtTest.QTest.keyClick(window.view, key)
        assert tools[tool].isChecked()
        assert window.view._mode == tool
    # the older Ctrl shortcuts stay
    assert tools["Pick"].shortcuts()[1] == QtGui.QKeySequence("Ctrl+P")
    # a text field takes the key instead
    field = QtWidgets.QLineEdit(window)
    tools["Pick"].trigger()
    field.setFocus()
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Z)
    assert field.text() == "z"
    assert window.view._mode == "Pick"


class _Shown(list):
    """Message boxes shown; ``answer`` is called with each, e.g., to
    tick its check box or click a button."""

    @staticmethod
    def answer(box):
        pass


@pytest.fixture
def notices(monkeypatch) -> _Shown:
    """The message boxes shown with ``exec``, see ``_Shown``."""
    shown = _Shown()

    def exec_(box):
        shown.append(box)
        shown.answer(box)
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)
    return shown


def _press_ctrl_z(window, qapp) -> None:
    from PyQt6 import QtTest

    window.show()
    window.activateWindow()
    qapp.processEvents()
    ctrl = QtCore.Qt.KeyboardModifier.ControlModifier
    QtTest.QTest.keyClick(window.view, QtCore.Qt.Key.Key_Z, ctrl)
    # let go of Ctrl, which _undo asks for
    QtTest.QTest.keyRelease(window.view, QtCore.Qt.Key.Key_Control)
    qapp.processEvents()


def test_ctrl_z_explains_itself_until_turned_off(
    window, qt_offscreen, notices
):
    view = window.view
    before = _copies(view)
    _drag(view, 2, 0)
    _drag(view, 0, 2)
    window.undo_action.trigger()  # from the menu: no notice
    assert not notices
    _press_ctrl_z(window, qt_offscreen)
    _assert_locs(view, before)  # it also undid
    (box,) = notices
    assert 'It undid "Move"' in box.text()
    assert "now selected with Z" in box.text()
    # without "Don't show again", the next Ctrl+Z explains again
    view.redo()
    _press_ctrl_z(window, qt_offscreen)
    assert len(notices) == 2
    _assert_locs(view, before)


def test_ctrl_z_from_the_native_menu_bar_explains_itself(
    window, notices, monkeypatch
):
    """The menu bar of macOS triggers File > Undo itself for Cmd+Z,
    without a shortcut event; Cmd (Ctrl to Qt) is held then."""
    _drag(window.view, 2, 0)
    monkeypatch.setattr(
        QtGui.QGuiApplication,
        "queryKeyboardModifiers",
        staticmethod(lambda: QtCore.Qt.KeyboardModifier.ControlModifier),
    )
    window.undo_action.trigger()
    (box,) = notices
    assert 'It undid "Move"' in box.text()


def test_ctrl_z_with_nothing_to_undo_explains_itself(
    window, qt_offscreen, notices
):
    tools = {a.text(): a for a in window.tools_actiongroup.actions()}
    tools["Pick"].trigger()

    def select_zoom(box):
        (zoom,) = [b for b in box.buttons() if b.text() == "Select Zoom tool"]
        zoom.click()

    notices.answer = select_zoom
    _press_ctrl_z(window, qt_offscreen)
    (box,) = notices
    assert "There was nothing to undo." in box.text()
    assert window.view._mode == "Zoom"


def test_ctrl_z_notice_can_be_turned_off(
    window, qt_offscreen, notices, tmp_path
):
    notices.answer = lambda box: box.checkBox().setChecked(True)
    _press_ctrl_z(window, qt_offscreen)
    assert len(notices) == 1
    settings = io.load_user_settings()
    assert settings["Render"][gui_render.CTRL_Z_NOTICE_KEY] is False
    _press_ctrl_z(window, qt_offscreen)
    assert len(notices) == 1
    other = gui_render.Window(plugins_loaded=True)
    try:
        _add(other, tmp_path / "other.hdf5", _locs())
        _show(other)
        _press_ctrl_z(other, qt_offscreen)
        assert len(notices) == 1
    finally:
        _close(other)


def test_tooltips_name_the_step(window):
    _drag(window.view, 2, 0)
    assert window.undo_action.isEnabled()
    assert window.undo_action.toolTip().startswith("Undo: Move (")
    window.view.undo()
    assert window.redo_action.toolTip().startswith("Redo: Move (")
    assert window.undo_action.toolTip().startswith("Undo (")
    assert not window.undo_action.isEnabled()


# ---------------------------------------------------------------------------
# The Move tool and the canvas
# ---------------------------------------------------------------------------


def test_undo_and_redo_a_move(window):
    view = window.view
    before = _copies(view)
    _drag(view, 2, 1)
    moved = _copies(view)
    (step,), _ = _steps(view)
    record = step.records[0]
    # only the moved coordinates are kept, not the whole table
    assert set(step.records) == {0}
    assert not record.full and record.table.columns == ["x", "y"]
    view.undo()
    _assert_locs(view, before)
    assert gui_render.MANUAL_SHIFT_KEYS[0] not in view.infos[0][-1]
    view.redo()
    _assert_locs(view, moved)
    key_x = gui_render.MANUAL_SHIFT_KEYS[0]
    assert view.infos[0][-1][key_x] == pytest.approx(2 * PIXELSIZE)


def test_undone_translation_takes_picks_points_and_view_back(window):
    view = window.view
    window.tools_settings_dialog.set_move_channels((0,))
    view._set_picks([(650.0, 780.0)], "Circle", 200.0)
    view._points = [(130.0, 130.0)]
    viewport = np.array(view.viewport)
    before = _copies(view)
    _drag(view, -10, -3)  # beyond the top left edge
    ox, oy = (view.infos[1][-1][key] for key in lib.CANVAS_OFFSET_KEYS)
    assert ox > 0 and oy > 0
    moved = _copies(view)
    view.undo()
    _assert_locs(view, before)
    for info in view.infos:
        assert info[0]["Width"] == WIDTH and info[0]["Height"] == HEIGHT
        for key in lib.CANVAS_OFFSET_KEYS:
            assert not lib.get_from_metadata(info, key, 0)
    assert view._picks == [pytest.approx((650.0, 780.0))]
    assert view._points == [pytest.approx((130.0, 130.0))]
    np.testing.assert_allclose(view.viewport, viewport)
    view.redo()
    _assert_locs(view, moved)
    assert view._picks == [pytest.approx((650.0 + ox, 780.0 + oy))]
    assert view._points == [pytest.approx((130.0 + ox, 130.0 + oy))]


def test_a_click_without_a_drag_records_nothing(window):
    _drag(window.view, 0, 0)
    assert window.view.history.undo_label is None


# ---------------------------------------------------------------------------
# The other actions
# ---------------------------------------------------------------------------


def _apply(window, monkeypatch, cmd: str, channel: str | None = None):
    def exec_(dialog):
        dialog.cmd.setText(cmd)
        if channel is not None:
            dialog.channel.setCurrentText(channel)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(gui_render.ApplyDialog, "exec", exec_)
    window.open_apply_dialog()


def test_apply_expression_keeps_the_changed_column_only(window, monkeypatch):
    view = window.view
    x = view.locs[0]["x"].to_numpy()  # the array itself, not a copy
    x_values = x.copy()
    before = _copies(view)
    _apply(window, monkeypatch, "x += 100")
    # the expression did not write into the array the step keeps
    np.testing.assert_array_equal(x, x_values)
    (step,), _ = _steps(view)
    assert step.label == "Apply expression: x += 100"
    assert step.records[0].table.columns == ["x"]
    view.undo()
    _assert_locs(view, before)


def test_apply_expression_to_all_channels_is_one_step(window, monkeypatch):
    view = window.view
    before = _copies(view)
    _apply(window, monkeypatch, "y += 50", "Apply to all sequentially")
    (step,), _ = _steps(view)
    assert set(step.records) == {0, 1}
    view.undo()
    _assert_locs(view, before)


def test_removing_picked_localizations_keeps_the_old_table(window):
    view = window.view
    original = view.locs[0]
    before = _copies(view)
    n_info = len(view.infos[0])
    view._set_picks([(WIDTH * PIXELSIZE / 2,) * 2], "Circle", 1500.0)
    view._remove_picked_locs(0)
    assert len(view.locs[0]) < len(before[0])
    removed = view.locs[0].copy()
    (step,), _ = _steps(view)
    record = step.records[0]
    assert record.full
    # the table itself, not a copy, and nothing shared with the new one
    assert record.table.load() is original
    view.undo()
    assert view.locs[0] is original  # linked windows recognize it
    _assert_locs(view, before)
    assert len(view.infos[0]) == n_info
    view.redo()
    pd.testing.assert_frame_equal(view.locs[0], removed)
    assert len(view.infos[0]) == n_info + 1


def test_undo_and_redo_drift(window):
    view = window.view
    x = view.locs[0]["x"].to_numpy()
    x_values = x.copy()
    before = _copies(view)
    drift = pd.DataFrame({"x": np.full(1000, 10.0), "y": np.full(1000, -5.0)})
    view._apply_drift(0, drift)
    np.testing.assert_array_equal(x, x_values)  # not written into
    np.testing.assert_allclose(view.locs[0]["x"], x_values - 10, atol=ATOL)
    assert view._drift[0] is drift
    (step,), _ = _steps(view)
    assert step.label == "Apply drift"
    assert set(step.records[0].table.columns) == {"x", "y"}
    view.undo()
    _assert_locs(view, before)
    assert view._drift[0] is None
    view.redo()
    pd.testing.assert_frame_equal(view._drift[0], drift)
    np.testing.assert_allclose(view.locs[0]["x"], x_values - 10, atol=ATOL)


def test_applying_drift_replaces_the_columns():
    locs = _locs()
    x = locs["x"].to_numpy()
    x_values = x.copy()
    drift = pd.DataFrame({"x": np.ones(1000), "y": np.ones(1000)})
    shifted, _ = postprocess.apply_drift(locs, _info(), drift=drift)
    assert shifted is locs  # the localizations are changed in place
    np.testing.assert_allclose(locs["x"], x_values - 1)
    np.testing.assert_array_equal(x, x_values)  # the old array is not


def test_undo_removed_columns(window, monkeypatch):
    view = window.view
    before = _copies(view)
    monkeypatch.setattr(view, "get_channel", lambda *a, **k: 1)
    monkeypatch.setattr(
        lib.RemoveColumnsDialog,
        "getParams",
        staticmethod(lambda *a, **k: (["frame", "photons"], True)),
    )
    window.remove_columns()
    assert list(view.locs[1].columns) == ["x", "y", "lpx", "lpy"]
    view.undo()
    _assert_locs(view, before)  # also the order of the columns


def test_undo_synchronized_groups(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    try:
        _add(window, tmp_path / "a.hdf5", _locs(seed=0, groups=40))
        locs = _locs(seed=1, groups=40)
        _add(window, tmp_path / "b.hdf5", locs[locs["group"] < 20])
        _show(window)
        before = _copies(window.view)
        window.view.sync_groups()
        assert len(window.view.locs[0]) < len(before[0])
        window.view.undo()
        _assert_locs(window.view, before)
    finally:
        _close(window)


def test_undo_unfolding_brings_the_picks_back(single, monkeypatch):
    view = single.view
    before = _copies(view)
    picks = [(1000.0, 1000.0), (3000.0, 3000.0)]
    view._set_picks(picks, "Circle", 1500.0)
    answers = iter([(2, True), (250, True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getInt", lambda *a, **k: next(answers)
    )
    view.unfold_groups_square()
    assert not view._picks  # used up
    unfolded = _copies(view)
    view.undo()
    _assert_locs(view, before)
    assert view._picks == picks
    view.redo()
    _assert_locs(view, unfolded)
    assert not view._picks


def test_undo_linking(single, qt_offscreen, monkeypatch):
    view = single.view
    before = _copies(view)
    monkeypatch.setattr(
        gui_render.LinkDialog, "getParams", lambda *a, **k: (300.0, 3, True)
    )
    view.link()
    _wait_for_tasks(qt_offscreen)
    assert "len" in view.locs[0].columns
    linked = _copies(view)
    assert view.history.undo_label == "Link localizations"
    view.undo()
    _assert_locs(view, before)
    view.redo()
    _assert_locs(view, linked)


# ---------------------------------------------------------------------------
# Picks and measurements
# ---------------------------------------------------------------------------


def test_undo_and_redo_picks(window):
    view = window.view
    view.add_pick((650.0, 780.0), shape="Circle", size=200.0)
    view.add_pick((1300.0, 1300.0), shape="Square", size=300.0)
    assert view.history.undo_label == "Add pick"
    view.undo()
    assert view._picks == [(650.0, 780.0)]
    assert view._pick_shapes == ["Circle"]
    assert view._pick_sizes == [200.0]
    assert window.info_dialog.n_picks.text() == "1"
    view.redo()
    assert view._picks == [(650.0, 780.0), (1300.0, 1300.0)]
    assert view._pick_shapes == ["Circle", "Square"]
    assert view._pick_sizes == [200.0, 300.0]


def test_undo_clearing_picks(window):
    view = window.view
    picks = [(650.0, 780.0), ((100.0, 100.0), (900.0, 400.0))]
    view._set_picks(picks, ["Circle", "Rectangle"], [200.0, 80.0])
    view.clear_picks()
    assert view.history.undo_label == "Clear picks"
    view.undo()
    assert view._picks == picks
    assert view._pick_shapes == ["Circle", "Rectangle"]
    assert view._pick_sizes == [200.0, 80.0]
    view.redo()
    assert view._picks == []


def test_undo_polygon_vertices_one_by_one(window):
    """Drawing a polygon adds its vertices to the pick in place; each
    step keeps the polygon as it was."""
    view = window.view
    vertices = [(100.0, 100.0), (1000.0, 100.0), (1000.0, 1000.0)]
    for vertex in vertices:
        view.add_polygon_point(vertex, QtCore.QPoint(0, 0))
    assert view._picks == [vertices]
    assert view.history.undo_label == "Add polygon vertex"
    view.undo()
    assert view._picks == [vertices[:2]]
    view.undo()
    view.undo()
    assert view._picks == []
    for _ in vertices:
        view.redo()
    assert view._picks == [vertices]
    view.remove_polygon_point()
    assert view.history.undo_label == "Remove polygon vertex"
    view.undo()
    assert view._picks == [vertices]


def test_undo_measurements(window):
    view = window.view
    a, b, c = (130.0, 130.0), (260.0, 130.0), (500.0, 500.0)
    view.add_point(a)
    view.add_point(b)
    assert view.history.undo_label == "Add measure point"
    view.finalize_measure_set()  # not a step of its own
    view.add_point(c)
    view.remove_points()
    assert view.history.undo_label == "Clear measurements"
    assert view._points == [] and view._point_sets == []
    view.undo()
    assert view._points == [c] and view._point_sets == [[a, b]]
    view.undo()  # c
    assert view._points == [] and view._point_sets == [[a, b]]
    view.undo()  # b
    assert view._points == [a] and view._point_sets == []
    view.redo()
    view.redo()
    view.finalize_measure_set()
    view.remove_last_measure_set()
    assert view.history.undo_label == "Remove measurement"
    assert view._point_sets == [[a, b]]
    view.undo()
    assert view._point_sets == [[a, b], [c]]


def test_undoing_picks_leaves_the_localizations(window, monkeypatch):
    view = window.view
    locs = list(view.locs)
    view.add_pick((650.0, 780.0))
    (step,), _ = _steps(view)
    assert not step.records and set(step.overlays) == {"picks"}
    invalidated = []
    monkeypatch.setattr(view, "invalidate_locs_index", invalidated.append)
    view.undo()
    view.redo()
    assert all(a is b for a, b in zip(view.locs, locs))
    assert not invalidated


def test_picks_and_measurements_have_their_own_limit(window, monkeypatch):
    """Clicking picks does not push out the steps that changed the
    localizations; forgetting one of those forgets the steps before
    it."""
    monkeypatch.setattr(render_undo, "UNDO_LIMIT", 2)
    monkeypatch.setattr(render_undo, "OVERLAY_UNDO_LIMIT", 3)
    view = window.view
    _apply(window, monkeypatch, "x += 1", view.locs_paths[0])
    for i in range(5):
        view.add_pick((100.0 * (i + 1), 100.0))
    undo, _ = _steps(view)
    labels = [step.label for step in undo]
    assert labels == ["Apply expression: x += 1"] + ["Add pick"] * 3
    _apply(window, monkeypatch, "x += 2", view.locs_paths[0])
    view.add_point((130.0, 130.0))
    _apply(window, monkeypatch, "x += 3", view.locs_paths[0])
    undo, _ = _steps(view)
    assert [step.label for step in undo] == [
        "Apply expression: x += 2",
        "Add measure point",
        "Apply expression: x += 3",
    ]


# ---------------------------------------------------------------------------
# The history
# ---------------------------------------------------------------------------


def test_a_new_step_drops_the_steps_undone(window, monkeypatch):
    view = window.view
    _apply(window, monkeypatch, "x += 10", view.locs_paths[0])
    view.undo()
    assert window.redo_action.isEnabled()
    _apply(window, monkeypatch, "y += 10", view.locs_paths[0])
    assert view.history.redo_label is None
    assert not window.redo_action.isEnabled()
    assert view.history.undo_label == "Apply expression: y += 10"


def test_the_oldest_steps_are_forgotten(window, monkeypatch):
    monkeypatch.setattr(render_undo, "UNDO_LIMIT", 3)
    view = window.view
    for i in range(5):
        _apply(window, monkeypatch, f"x += {i + 1}", view.locs_paths[0])
    undo, _ = _steps(view)
    assert [step.label for step in undo] == [
        f"Apply expression: x += {i}" for i in (3, 4, 5)
    ]


def test_adding_or_closing_a_channel_forgets_the_steps(window, tmp_path):
    view = window.view
    _drag(view, 2, 0)
    _add(window, tmp_path / "more.hdf5", _locs(seed=5))
    assert view.history.undo_label is None
    _drag(view, 2, 0)
    window.dataset_dialog.close_file(2, render=False)
    assert view.history.undo_label is None
    assert not window.undo_action.isEnabled()


def test_reordering_the_channels_keeps_the_steps(window):
    view = window.view
    window.tools_settings_dialog.set_move_channels((1,))
    moved_channel = view.locs_paths[1]
    before = view.locs[1].copy()
    _drag(view, 2, 0)
    window.dataset_dialog.move_channel(1, 0)
    assert view.locs_paths[0] == moved_channel
    view.undo()
    pd.testing.assert_frame_equal(view.locs[0], before)


def test_a_step_that_no_longer_fits_forgets_the_history(window, messages):
    view = window.view
    _drag(view, 2, 0)
    # e.g., a plugin removed localizations without recording a step
    view.locs[0] = view.locs[0].iloc[:10].copy()
    view.undo()
    assert len(view.locs[0]) == 10
    assert view.history.undo_label is None
    assert "forgotten" in messages[-1]


# ---------------------------------------------------------------------------
# Memory and disk
# ---------------------------------------------------------------------------


def test_kept_arrays_do_not_share_memory_with_the_localizations(window):
    view = window.view
    _drag(view, 2, 0)
    (step,), _ = _steps(view)
    kept = step.records[0].table.load()
    current = [view.locs[0][c].to_numpy() for c in view.locs[0].columns]
    for name in kept.columns:
        array = kept[name].to_numpy()
        assert not any(np.may_share_memory(array, c) for c in current)


@pytest.fixture
def spill_folder(tmp_path, monkeypatch):
    folder = tmp_path / "spill"
    folder.mkdir()
    monkeypatch.setattr(io, "spill_directory", lambda: str(folder))
    return folder


def test_a_large_step_is_moved_to_disk(window, spill_folder, monkeypatch):
    monkeypatch.setattr(idle, "SPILL_MIN_BYTES", 0)
    view = window.view
    before = _copies(view)
    view._set_picks([(WIDTH * PIXELSIZE / 2,) * 2], "Circle", 1500.0)
    view._remove_picked_locs(0)
    removed = view.locs[0].copy()
    render_undo.wait_for_writes()
    (step,), _ = _steps(view)
    assert step.records[0].table.on_disk
    assert len(list(spill_folder.iterdir())) == 1
    view.undo()
    _assert_locs(view, before)
    render_undo.wait_for_writes()  # the step to redo
    assert len(list(spill_folder.iterdir())) == 1
    view.redo()
    pd.testing.assert_frame_equal(view.locs[0], removed)
    render_undo.wait_for_writes()
    view.clear_undo_history()
    assert list(spill_folder.iterdir()) == []


def test_undo_after_the_localizations_moved_to_disk_while_idle(
    window, spill_folder, monkeypatch
):
    monkeypatch.setattr(idle, "SPILL_MIN_BYTES", 0)
    view = window.view
    before = _copies(view)
    _drag(view, 2, 1)
    render_undo.wait_for_writes()
    moved = view.locs[0]["x"].to_numpy()
    storage = weakref.ref(moved if moved.base is None else moved.base)
    del moved
    assert window.release_idle_memory()
    assert view.has_spilled_locs
    gc.collect()
    # the history does not keep the localizations in memory
    assert storage() is None
    view.undo()
    _assert_locs(view, before)


# ---------------------------------------------------------------------------
# Linked windows
# ---------------------------------------------------------------------------


@pytest.fixture
def shared(qt_offscreen, tmp_path):
    """The primary window holds channels a and b; the linked window
    shares b and holds its own channel c."""
    primary = gui_render.Window(plugins_loaded=True)
    primary.view.async_rendering = False
    _add(primary, tmp_path / "a.hdf5", _locs(seed=0))
    _add(primary, tmp_path / "b.hdf5", _locs(seed=1))
    _show(primary)
    primary.view.update_scene()
    group = render_link.LinkGroup(primary)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    secondary.view.async_rendering = False
    secondary.view.add_channels_from(primary.view, [1], share=True)
    _add(secondary, tmp_path / "c.hdf5", _locs(seed=2))
    _show(secondary)
    secondary.view.update_scene()
    yield primary, secondary
    for window in (primary, secondary):
        _close(window)


def test_linked_window_takes_an_undo_and_forgets_its_steps(
    shared, monkeypatch
):
    primary, secondary = shared
    b = primary.view.locs[1].copy()
    secondary._apply_cmd("x += 10", 1)  # its own channel c
    assert secondary.view.history.undo_label is not None
    primary._apply_cmd("x += 100", 1)  # the shared channel b
    assert secondary.view.locs[0] is primary.view.locs[1]
    # the linked window's steps no longer match its channels
    assert secondary.view.history.undo_label is None
    assert not secondary.undo_action.isEnabled()
    primary.view.undo()
    pd.testing.assert_frame_equal(primary.view.locs[1], b)
    assert secondary.view.locs[0] is primary.view.locs[1]
    pd.testing.assert_frame_equal(secondary.view.locs[0], b)
    # the window that undid keeps its steps
    assert primary.view.history.redo_label == "Apply expression: x += 100"


# ---------------------------------------------------------------------------
# Every action that changes the localizations records a step
# ---------------------------------------------------------------------------

#: Methods that change the localizations without recording a step, on
#: top of those that do not change any channel (``_EXEMPT``).
_NOT_RECORDED = {
    ("View", "_apply_history_step"),  # puts a step back
    # helpers of the steps of their callers
    ("View", "locs_moved"),
    ("Window", "_apply_cmd_unrecorded"),
}


@pytest.mark.parametrize(
    "cls_name,fn_name,node",
    [pytest.param(c, f, n, id=f"{c}.{f}") for c, f, n in _mutating_methods()],
)
def test_locs_mutation_is_recorded_for_undo(cls_name, fn_name, node):
    """An action that changes the localizations without recording a
    step cannot be undone, and it leaves the steps recorded before it
    out of step with the channels. Add to ``_NOT_RECORDED`` only a
    method that is part of another one's step."""
    if (cls_name, fn_name) in _EXEMPT | _NOT_RECORDED:
        pytest.skip("not an action of its own")
    source = ast.unparse(node)
    assert "undo_step(" in source or "begin_undo_step(" in source, (
        f"{cls_name}.{fn_name} changes the localizations without "
        "recording a step to undo: wrap the change in "
        "`with self.undo_step(<name of the action>)`"
    )


# ---------------------------------------------------------------------------
# Every action that changes the picks or the measured points records a step
# ---------------------------------------------------------------------------

#: Attributes holding the picks and the measured points.
_OVERLAY_ATTRIBUTES = {
    "_pick_list",
    "_pick_shapes",
    "_pick_sizes",
    "_picks",
    "_points",
    "_point_sets",
}
#: Methods that change the picks without recording a step.
_PICK_HELPERS = {"_set_picks", "_keep_picks", "_remove_picks_at"}
_LIST_CHANGES = {"append", "pop", "extend", "insert", "remove", "clear"}

#: Methods that change the picks or the measured points without
#: recording a step.
_OVERLAYS_NOT_RECORDED = {
    ("View", "__init__"),
    ("View", "_apply_history_step"),  # puts a step back
    # helpers of the steps of their callers
    ("View", "_picks"),  # the setter
    ("View", "_set_picks"),
    ("View", "_keep_picks"),
    ("View", "_remove_picks_at"),
    ("View", "_translate_overlays"),  # follows a translated canvas
    # freezing the set of points being measured only regroups them
    ("View", "finalize_measure_set"),
    # drops the polygon being drawn when the pick tool changes; undoing
    # the step of its last vertex brings it back
    ("View", "on_pick_shape_changed"),
    # part of opening a file, which forgets the steps
    ("Window", "_apply_saved_rotation"),
}


def _changed_attribute(node) -> str | None:
    while isinstance(node, ast.Subscript):  # e.g., self._picks[-1]
        node = node.value
    return node.attr if isinstance(node, ast.Attribute) else None


def _changes_overlays(node) -> bool:
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        return any(
            isinstance(sub, ast.Attribute)
            and sub.attr in _OVERLAY_ATTRIBUTES
            and isinstance(sub.ctx, ast.Store)
            for target in targets
            for sub in ast.walk(target)
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        name = node.func.attr
        return name in _PICK_HELPERS or (
            name in _LIST_CHANGES
            and _changed_attribute(node.func.value) in _OVERLAY_ATTRIBUTES
        )
    return False


def _overlay_methods():
    """``(class, method, node)`` for every method in ``gui.render`` that
    changes the picks or the measured points."""
    tree = ast.parse(inspect.getsource(gui_render))
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_changes_overlays(node) for node in ast.walk(fn)):
                yield cls.name, fn.name, fn


def test_the_overlay_guard_finds_the_pick_actions():
    found = {(c, f) for c, f, _ in _overlay_methods()}
    assert {
        ("View", "add_pick"),
        ("View", "add_polygon_point"),
        ("View", "clear_picks"),
        ("View", "add_point"),
        ("View", "remove_points"),
    } <= found
    assert _OVERLAYS_NOT_RECORDED <= found  # no stale entries


@pytest.mark.parametrize(
    "cls_name,fn_name,node",
    [pytest.param(c, f, n, id=f"{c}.{f}") for c, f, n in _overlay_methods()],
)
def test_overlay_change_is_recorded_for_undo(cls_name, fn_name, node):
    """An action that changes the picks or the measured points without
    recording a step cannot be undone. Add to ``_OVERLAYS_NOT_RECORDED``
    only a method that is part of another one's step or not an action
    of the user's."""
    if (cls_name, fn_name) in _OVERLAYS_NOT_RECORDED:
        pytest.skip("not an action of its own")
    source = ast.unparse(node)  # with the decorators
    assert "undo_step(" in source or "records_undo_step(" in source, (
        f"{cls_name}.{fn_name} changes the picks or the measured points "
        "without recording a step to undo: wrap the change in "
        "`with self.undo_step(<name of the action>, picks=True)` (or "
        "points=True)"
    )
