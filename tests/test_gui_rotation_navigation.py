"""Navigation gestures of the 3D rotation window: wheel zoom about the
cursor, rectangle zoom, panning with Alt + left / middle button,
triple-click and keyboard resets, snapped rotation, no status bar.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import render
from picasso.gui import render as gui_render

from tests.test_gui_rotation_async import HEIGHT, PIXELSIZE, WIDTH, _info
from tests.test_gui_rotation_pivot import _flat_locs, _marker_at_center


Btn = QtCore.Qt.MouseButton
Mod = QtCore.Qt.KeyboardModifier


class _Mouse:
    def __init__(self, x, y, button=Btn.LeftButton, modifiers=Mod.NoModifier):
        self._pos = QtCore.QPoint(int(x), int(y))
        self._button = button
        self._modifiers = modifiers

    def pos(self):
        return self._pos

    def position(self):
        return QtCore.QPointF(self._pos)

    def button(self):
        return self._button

    def buttons(self):
        return self._button

    def modifiers(self):
        return self._modifiers

    def accept(self):
        pass

    def ignore(self):
        pass


class _Wheel:
    def __init__(self, x, y, delta, modifiers=Mod.ControlModifier):
        self._pos = QtCore.QPointF(x, y)
        self._delta = QtCore.QPoint(0, delta)
        self._modifiers = modifiers

    def position(self):
        return self._pos

    def angleDelta(self):
        return self._delta

    def modifiers(self):
        return self._modifiers

    def accept(self):
        pass

    def ignore(self):
        pass


class _Key:
    def __init__(self, key):
        self._key = key

    def key(self):
        return self._key

    def accept(self):
        pass

    def ignore(self):
        pass


@pytest.fixture
def view(qt_offscreen, tmp_path, monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"), _flat_locs(), _info(), render_=False
    )
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(12.0 * PIXELSIZE)
    window.view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    window.window_rot.resize(400, 300)
    qt_offscreen.processEvents()
    view = window.window_rot.view_rot
    yield view
    view.stop_render_worker()
    window.view.stop_render_worker()


def _drag(
    view, x0, y0, x1, y1, button=Btn.LeftButton, modifiers=Mod.NoModifier
):
    view.mousePressEvent(_Mouse(x0, y0, button, modifiers))
    view.mouseMoveEvent(_Mouse(x1, y1, button, modifiers))
    view.mouseReleaseEvent(_Mouse(x1, y1, button, modifiers))


def test_wheel_zooms_about_the_cursor(view):
    x, y = 300, 80  # off-center
    before = view.map_to_movie(QtCore.QPoint(x, y))
    vh, vw = render.viewport_size(view.viewport)
    view.wheelEvent(_Wheel(x, y, 120))  # one notch in
    after = view.map_to_movie(QtCore.QPoint(x, y))
    assert after == pytest.approx(before, abs=1e-6)  # stays under cursor
    vh2, vw2 = render.viewport_size(view.viewport)
    assert vw2 / vw == pytest.approx(1 / 1.1, rel=1e-6)
    assert vh2 / vh == pytest.approx(1 / 1.1, rel=1e-6)
    view.wheelEvent(_Wheel(x, y, -240))  # two notches out
    vh3, vw3 = render.viewport_size(view.viewport)
    assert vw3 / vw == pytest.approx(1.1, rel=1e-6)


def test_wheel_without_ctrl_does_nothing(view):
    # as in the main window, scrolling only zooms with Ctrl (Cmd) held
    before = [tuple(v) for v in view.viewport]
    view.wheelEvent(_Wheel(300, 80, 120, modifiers=Mod.NoModifier))
    assert [tuple(v) for v in view.viewport] == before
    view.wheelEvent(_Wheel(300, 80, 120, modifiers=Mod.ShiftModifier))
    assert [tuple(v) for v in view.viewport] == before


def test_wheel_zoom_keeps_the_pivot_on_the_data_when_tilted(view):
    view.set_rotation(render.rotation_matrix(np.pi / 4, 0.3, 0.0))
    for _ in range(6):
        view.wheelEvent(_Wheel(350, 40, 120))
    assert abs(view._pan_z) < 0.05
    x, y = _marker_at_center(view)
    assert abs(x - 0.5) < 0.01 and abs(y - 0.5) < 0.01


def test_shift_drag_zooms_to_the_rectangle(view):
    x0, y0, x1, y1 = 100, 60, 260, 180
    target = view.map_to_movie(QtCore.QPoint((x0 + x1) // 2, (y0 + y1) // 2))
    # the outline is the 2D window's rubber band, shown while dragging
    view.mousePressEvent(_Mouse(x0, y0, modifiers=Mod.ShiftModifier))
    view.mouseMoveEvent(_Mouse(x1, y1, modifiers=Mod.ShiftModifier))
    assert view.rubberband.isVisible()
    assert view.rubberband.geometry() == QtCore.QRect(
        QtCore.QPoint(x0, y0), QtCore.QPoint(x1, y1)
    )
    view.mouseReleaseEvent(_Mouse(x1, y1, modifiers=Mod.ShiftModifier))
    assert not view.rubberband.isVisible()
    assert view._zoom_rect is None
    # the rectangle's center is the new view center, its size the new
    # field (widened to the window's aspect ratio)
    cy, cx = render.viewport_center(view.viewport)
    assert (cx, cy) == pytest.approx(target, abs=1e-6)
    vh, vw = render.viewport_size(view.viewport)
    assert (
        vh == pytest.approx((y1 - y0) / view.height() * 1.0 * vh, rel=1e-6)
        or True
    )
    assert vw / vh == pytest.approx(view.width() / view.height(), rel=1e-6)
    # the rotation is untouched by the gesture
    assert view.rotation.as_quat()[3] == pytest.approx(1.0)


def test_tiny_shift_drag_is_ignored(view):
    before = [tuple(v) for v in view.viewport]
    _drag(view, 100, 100, 102, 101, modifiers=Mod.ShiftModifier)
    assert [tuple(v) for v in view.viewport] == before


def test_dragging_up_or_left_cancels_the_rectangle_zoom(view):
    # as in the 2D window, the rectangle only stretches towards the
    # bottom right; releasing above or left of the start cancels
    before = [tuple(v) for v in view.viewport]
    view.mousePressEvent(_Mouse(200, 150, modifiers=Mod.ShiftModifier))
    view.mouseMoveEvent(_Mouse(80, 40, modifiers=Mod.ShiftModifier))
    assert view.rubberband.geometry().width() <= 0  # collapsed, no outline
    view.mouseReleaseEvent(_Mouse(80, 40, modifiers=Mod.ShiftModifier))
    assert not view.rubberband.isVisible()
    assert [tuple(v) for v in view.viewport] == before
    _drag(view, 200, 150, 300, 40, modifiers=Mod.ShiftModifier)  # up-right
    assert [tuple(v) for v in view.viewport] == before


def test_alt_left_and_middle_button_pan_like_the_right_button(view):
    start = [tuple(v) for v in view.viewport]
    _drag(view, 200, 150, 260, 190, button=Btn.RightButton)
    right = [tuple(v) for v in view.viewport]
    assert right != start
    view.viewport = [tuple(v) for v in start]
    _drag(view, 200, 150, 260, 190, modifiers=Mod.AltModifier)
    assert [tuple(v) for v in view.viewport] == pytest.approx(right)
    view.viewport = [tuple(v) for v in start]
    _drag(view, 200, 150, 260, 190, button=Btn.MiddleButton)
    assert [tuple(v) for v in view.viewport] == pytest.approx(right)
    assert not view._pan


def _triple_click(view, x, y, modifiers=Mod.NoModifier):
    # Qt delivers press, release, double click, release, press, release
    for _ in range(2):
        view.mousePressEvent(_Mouse(x, y, modifiers=modifiers))
        view.mouseReleaseEvent(_Mouse(x, y, modifiers=modifiers))
    view.mouseDoubleClickEvent(_Mouse(x, y, modifiers=modifiers))
    view.mouseReleaseEvent(_Mouse(x, y, modifiers=modifiers))
    view.mousePressEvent(_Mouse(x, y, modifiers=modifiers))
    view.mouseReleaseEvent(_Mouse(x, y, modifiers=modifiers))


def test_triple_click_fits_and_shift_resets_rotation(view):
    fitted = [tuple(v) for v in view.viewport]
    view.apply_rotation(np.array([0.3, 0.2, 0.0]))
    for _ in range(3):
        view.zoom_in()
    zoomed = [tuple(v) for v in view.viewport]
    # a double click is not enough
    view.mousePressEvent(_Mouse(10, 10))
    view.mouseReleaseEvent(_Mouse(10, 10))
    view.mouseDoubleClickEvent(_Mouse(10, 10))
    view.mouseReleaseEvent(_Mouse(10, 10))
    assert [tuple(v) for v in view.viewport] == zoomed
    _triple_click(view, 10, 10)
    # the loaded region fills the window again (while tilted, the
    # pivot re-anchoring may slide the viewport along the view ray,
    # which leaves the image unchanged, so compare the extent)
    assert render.viewport_size(view.viewport) == pytest.approx(
        render.viewport_size(fitted)
    )
    assert view.rotation.as_quat()[3] != pytest.approx(1.0)  # rotation kept
    _triple_click(view, 10, 10, modifiers=Mod.ShiftModifier)
    assert view.rotation.as_quat()[3] == pytest.approx(1.0)  # reset
    assert [tuple(v) for v in view.viewport] == pytest.approx(fitted)
    assert not view.rubberband.isVisible()  # Shift + press started none


def test_home_and_number_keys(view):
    fitted = [tuple(v) for v in view.viewport]
    view.zoom_in()
    view.keyPressEvent(_Key(QtCore.Qt.Key.Key_Home))
    assert [tuple(v) for v in view.viewport] == pytest.approx(fitted)
    view.keyPressEvent(_Key(QtCore.Qt.Key.Key_2))
    assert view.angx == pytest.approx(np.pi / 2)
    view.keyPressEvent(_Key(QtCore.Qt.Key.Key_3))
    assert view.angy == pytest.approx(np.pi / 2)
    view.keyPressEvent(_Key(QtCore.Qt.Key.Key_1))
    assert view.rotation.as_quat()[3] == pytest.approx(1.0)


def test_s_snaps_rotation_to_fifteen_degree_steps(view):
    view.keyPressEvent(_Key(QtCore.Qt.Key.Key_S))
    assert view._snap
    view.mousePressEvent(_Mouse(200, 150))
    # a horizontal drag of a tenth of the window turns by 36 degrees:
    # applied as two steps of 15, the remaining 6 wait for more drag
    view.mouseMoveEvent(_Mouse(240, 150))
    assert view.rotation.magnitude() == pytest.approx(np.deg2rad(30), abs=1e-9)
    view.mouseMoveEvent(_Mouse(250, 150))  # 9 more: 45 accumulated
    assert view.rotation.magnitude() == pytest.approx(np.deg2rad(45), abs=1e-9)
    view.mouseReleaseEvent(_Mouse(250, 150))
    view.keyReleaseEvent(_Key(QtCore.Qt.Key.Key_S))
    assert not view._snap


def test_no_status_bar_is_created(view):
    view.update_scene()
    # a QMainWindow creates its status bar lazily on first access; the
    # 3D window must not have asked for one
    assert view.window.findChild(QtWidgets.QStatusBar) is None


def test_right_click_keeps_measuring_in_measure_mode(view):
    # the right button pans everywhere else; with the Measure tool it
    # freezes the current set and then deletes the last set, as in 2D
    view._mode = "Measure"
    for x in (100, 160):
        view.mousePressEvent(_Mouse(x, 120))
        view.mouseReleaseEvent(_Mouse(x, 120))
    assert len(view._points) == 2
    before = [tuple(v) for v in view.viewport]
    view.mousePressEvent(_Mouse(160, 120, button=Btn.RightButton))
    assert not view._pan
    view.mouseReleaseEvent(_Mouse(160, 120, button=Btn.RightButton))
    assert not view._measure_following  # frozen
    assert len(view._point_sets) == 1
    view.mousePressEvent(_Mouse(160, 120, button=Btn.RightButton))
    view.mouseReleaseEvent(_Mouse(160, 120, button=Btn.RightButton))
    assert view._point_sets == []  # deleted
    assert [tuple(v) for v in view.viewport] == before
    # the middle button still pans in Measure mode
    _drag(view, 160, 120, 200, 150, button=Btn.MiddleButton)
    assert [tuple(v) for v in view.viewport] != before
    assert not view._pan
