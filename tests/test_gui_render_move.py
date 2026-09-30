"""The "Move" tool of ``picasso.gui.render``: dragging the localizations
of one channel with the mouse.

These tests drive the real mouse handlers of a ``View`` and cover the
live preview, the growth of the canvas (so no localization is removed
when saving), the translation of every channel, the picks and the
measured points when a channel is dragged beyond the top left edge,
the undo, and the Apply expression dialog, which shares the canvas
handling.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore

from picasso import io, lib
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32
PIXELSIZE = 130.0
#: the view is 256 display pixels wide and shows the whole 32 pixel
#: FOV, i.e., 8 display pixels per camera pixel
SCALE = 8


def _locs(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.uint32),
            "x": rng.uniform(0.0, WIDTH, size=n).astype(np.float32),
            "y": rng.uniform(0.0, HEIGHT, size=n).astype(np.float32),
            "lpx": np.full(n, 0.1, dtype=np.float32),
            "lpy": np.full(n, 0.1, dtype=np.float32),
            "photons": np.full(n, 1000.0, dtype=np.float32),
        }
    )


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

    def modifiers(self):
        return QtCore.Qt.KeyboardModifier.NoModifier

    def accept(self):
        pass

    def ignore(self):
        pass


def _drag(view, dx, dy, start=(128, 128)):
    """Drag by ``(dx, dy)`` camera pixels."""
    x0, y0 = start
    x1, y1 = x0 + dx * SCALE, y0 + dy * SCALE
    view.mousePressEvent(_Event(x0, y0))
    view.mouseMoveEvent(_Event(x1, y1))
    view.mouseReleaseEvent(_Event(x1, y1))


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding two channels, in Move mode."""
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        path = str(tmp_path / f"locs{i}.hdf5")
        window.view.add(path, _locs(seed=i), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(256, 256)
    window.view._mode = "Move"
    return window


def test_move_is_a_tool(window):
    actions = [a.text() for a in window.menu_bar.actions()]
    tools = window.menu_bar.actions()[actions.index("Tools")].menu()
    assert "Move" in [a.text() for a in tools.actions()]
    # only the first channel is dragged by default
    tools_dlg = window.tools_settings_dialog
    assert tools_dlg.move_channels() == (0,)
    assert tools_dlg.move_channels_label.text() == "locs0.hdf5"


def test_channel_selection_dialog(window):
    names = ["a.hdf5", "b.hdf5", "c.hdf5"]
    dialog = gui_render.MoveChannelsDialog(window, names, (0, 2))
    assert dialog.selected() == (0, 2)
    ok = dialog.buttons.button(
        gui_render.QtWidgets.QDialogButtonBox.StandardButton.Ok
    )
    dialog._set_all(False)
    assert dialog.selected() == () and not ok.isEnabled()
    dialog.checks[1].setChecked(True)
    assert dialog.selected() == (1,) and ok.isEnabled()
    dialog._set_all(True)
    assert dialog.selected() == (0, 1, 2)


def test_selection_summary(window, tmp_path):
    tools_dlg = window.tools_settings_dialog
    window.view.add(
        str(tmp_path / "locs2.hdf5"), _locs(seed=2), _info(), render_=False
    )
    assert tools_dlg.move_channels() == (0,)
    tools_dlg.set_move_channels((0, 2))
    assert tools_dlg.move_channels_label.text() == "2 of 3 channels"
    tools_dlg.set_move_channels((0, 1, 2))
    assert tools_dlg.move_channels_label.text() == "All 3 channels"
    # closing a channel keeps the selection of the others
    tools_dlg.set_move_channels((2,))
    window.dataset_dialog.close_file(0, render=False)
    assert tools_dlg.move_channels() == (1,)
    # the first channel is selected when none remains selected
    window.dataset_dialog.close_file(1, render=False)
    assert tools_dlg.move_channels() == (0,)


def test_preview_keeps_the_press_coordinates(window):
    view = window.view
    x = view.locs[0]["x"].to_numpy().copy()
    view.mousePressEvent(_Event(128, 128))
    view.mouseMoveEvent(_Event(128 + 2 * SCALE, 128))
    np.testing.assert_allclose(view.locs[0]["x"], x + 2, atol=1e-4)
    view.mouseMoveEvent(_Event(128 + 3 * SCALE, 128))
    np.testing.assert_allclose(view.locs[0]["x"], x + 3, atol=1e-4)
    # the press coordinates were not modified in memory
    np.testing.assert_array_equal(view._move_origin[0][0], x)
    view.mouseReleaseEvent(_Event(128 + 3 * SCALE, 128))
    assert not view._move_channels


def test_drag_within_canvas(window):
    view = window.view
    x0 = view.locs[0]["x"].to_numpy().copy()
    x1 = view.locs[1]["x"].to_numpy().copy()
    _drag(view, 2, 0)
    np.testing.assert_allclose(view.locs[0]["x"], x0 + 2, atol=1e-4)
    np.testing.assert_array_equal(view.locs[1]["x"], x1)
    key_x = gui_render.MANUAL_SHIFT_KEYS[0]
    assert view.infos[0][-1][key_x] == pytest.approx(2)
    assert key_x not in view.infos[1][-1]
    # grown to contain the locs moved beyond the right edge
    assert view.infos[0][0]["Width"] == int(np.floor(x0.max() + 2)) + 1
    assert view.infos[1][0]["Width"] == WIDTH
    assert view._move_undo == [((0,), pytest.approx(2), pytest.approx(0))]
    assert window.tools_settings_dialog.move_undo_button.isEnabled()


def test_drag_beyond_top_left_translates_everything(window, tmp_path):
    view = window.view
    x0 = view.locs[0]["x"].to_numpy().copy()
    x1 = view.locs[1]["x"].to_numpy().copy()
    y1 = view.locs[1]["y"].to_numpy().copy()
    view._pick_shape = "Circle"
    view._picks = [(5.0, 6.0)]
    view._points = [(1.0, 1.0)]
    viewport = view.viewport
    _drag(view, -10, -3)
    ox, oy = (view.infos[1][-1][key] for key in lib.CANVAS_OFFSET_KEYS)
    assert ox == int(np.ceil(10 - x0.min()))
    assert oy >= 1
    for locs in view.locs:
        assert locs["x"].min() >= 0 and locs["y"].min() >= 0
    # the other channel, picks and points moved along
    np.testing.assert_allclose(view.locs[1]["x"], x1 + ox, atol=1e-4)
    np.testing.assert_allclose(view.locs[1]["y"], y1 + oy, atol=1e-4)
    assert view._picks == [(5.0 + ox, 6.0 + oy)]
    assert view._points == [(1.0 + ox, 1.0 + oy)]
    # so did the view, so the display does not jump
    assert view.viewport[0][1] > viewport[0][1]
    assert view.infos[1][0]["Width"] == WIDTH + ox
    assert view.infos[1][-1][lib.CAMERA_SIZE_KEYS[0]] == WIDTH
    # saving keeps every localization
    for i, (locs, info) in enumerate(zip(view.locs, view.infos)):
        path = str(tmp_path / f"moved{i}.hdf5")
        io.save_locs(path, locs, info)
        loaded, _ = io.load_locs(path)
        assert len(loaded) == len(locs)


def test_undo_restores_the_registration(window):
    view = window.view
    diff = (view.locs[0]["x"] - view.locs[1]["x"]).to_numpy().copy()
    _drag(view, -10, 0)
    view.undo_move()
    np.testing.assert_allclose(
        view.locs[0]["x"] - view.locs[1]["x"], diff, atol=1e-4
    )
    assert view.infos[0][-1][gui_render.MANUAL_SHIFT_KEYS[0]] == 0
    assert not view._move_undo
    assert not window.tools_settings_dialog.move_undo_button.isEnabled()
    for locs in view.locs:
        assert locs["x"].min() >= 0
    # the canvas is back to the camera image
    assert view.infos[0][-1][lib.CANVAS_OFFSET_KEYS[0]] == 0
    assert view.infos[0][0]["Width"] == WIDTH


def test_dragging_back_restores_the_canvas(window):
    view = window.view
    x0 = view.locs[0]["x"].to_numpy().copy()
    x1 = view.locs[1]["x"].to_numpy().copy()
    viewport = view.viewport
    _drag(view, -10, 0)
    assert view.infos[0][0]["Width"] > WIDTH
    # the view followed the translation; drag back from the same spot
    # on screen, i.e., by the same distance in the other direction
    _drag(view, 10, 0)
    for info in view.infos:
        assert info[-1][lib.CANVAS_OFFSET_KEYS[0]] == 0
        assert info[0]["Width"] == WIDTH
    np.testing.assert_allclose(view.locs[0]["x"], x0, atol=1e-4)
    np.testing.assert_allclose(view.locs[1]["x"], x1, atol=1e-4)
    np.testing.assert_allclose(view.viewport, viewport)


def test_several_channels_are_dragged_together(window):
    view = window.view
    x0 = view.locs[0]["x"].to_numpy().copy()
    x1 = view.locs[1]["x"].to_numpy().copy()
    window.tools_settings_dialog.set_move_channels((0, 1))
    _drag(view, 2, 1)
    np.testing.assert_allclose(view.locs[0]["x"], x0 + 2, atol=1e-4)
    np.testing.assert_allclose(view.locs[1]["x"], x1 + 2, atol=1e-4)
    for info in view.infos:
        assert info[-1][gui_render.MANUAL_SHIFT_KEYS[1]] == pytest.approx(1)
    assert view._move_undo[-1][0] == (0, 1)
    view.undo_move()
    np.testing.assert_allclose(view.locs[0]["x"], x0, atol=1e-4)
    np.testing.assert_allclose(view.locs[1]["x"], x1, atol=1e-4)
    for info in view.infos:
        assert info[0]["Width"] == WIDTH and info[0]["Height"] == HEIGHT


def test_selected_channel_is_dragged(window):
    view = window.view
    x0 = view.locs[0]["x"].to_numpy().copy()
    window.tools_settings_dialog.set_move_channels((1,))
    _drag(view, 1, 0)
    np.testing.assert_array_equal(view.locs[0]["x"], x0)
    assert view._move_undo[-1][0] == (1,)


def test_click_without_drag_changes_nothing(window):
    view = window.view
    x0 = view.locs[0]["x"].to_numpy().copy()
    _drag(view, 0, 0)
    np.testing.assert_array_equal(view.locs[0]["x"], x0)
    assert not view._move_undo


def test_adding_a_channel_joins_the_translated_frame(window, tmp_path):
    view = window.view
    _drag(view, -40, 0)
    ox = view.infos[0][-1][lib.CANVAS_OFFSET_KEYS[0]]
    locs = _locs(seed=5)
    x = locs["x"].to_numpy().copy()
    view.add(str(tmp_path / "new.hdf5"), locs, _info(), render_=False)
    np.testing.assert_allclose(view.locs[2]["x"], x + ox, atol=1e-4)
    assert view.infos[2][-1][lib.CANVAS_OFFSET_KEYS[0]] == ox
    assert window.tools_settings_dialog.move_channels() == (0,)


def test_apply_expression_grows_the_canvas(window, tmp_path):
    view = window.view
    n = len(view.locs[0])
    before = view.locs[0]["x"].to_numpy()
    window._apply_cmd("x -= 5", 0)
    assert len(view.locs[0]) == n
    assert view.locs[0]["x"].min() >= 0
    # new memory, the GPU uploads are keyed on it
    assert not np.shares_memory(before, view.locs[0]["x"].to_numpy())
    path = str(tmp_path / "applied.hdf5")
    io.save_locs(path, view.locs[0], view.infos[0])
    loaded, _ = io.load_locs(path)
    assert len(loaded) == n
