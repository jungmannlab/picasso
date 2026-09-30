"""Navigation gestures of Render's main window that work in every
tool: panning with the middle button or Alt + left, the Shift + left
zoom rectangle, and the Zoom tool's triple click / Home key fit.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtGui

from picasso import render

from tests.test_gui_render_async import window  # noqa: F401 (fixture)
from tests.test_gui_rotation_navigation import Btn, Mod, _Mouse, _triple_click


@pytest.fixture
def view(window):  # noqa: F811
    view = window.view
    view.async_rendering = False
    window.show()  # the rubber band is only visible in a shown window
    view.update_scene()
    return view


def _drag(
    view, x0, y0, x1, y1, button=Btn.LeftButton, modifiers=Mod.NoModifier
):
    view.mousePressEvent(_Mouse(x0, y0, button, modifiers))
    view.mouseMoveEvent(_Mouse(x1, y1, button, modifiers))
    view.mouseReleaseEvent(_Mouse(x1, y1, button, modifiers))


def _viewport(view):
    return [tuple(v) for v in view.viewport]


@pytest.mark.parametrize("tool", ["Zoom", "Pick", "Measure"])
def test_middle_button_and_alt_left_pan_in_every_tool(view, tool):
    view._mode = tool
    start = _viewport(view)
    _drag(view, 40, 40, 70, 60, modifiers=Mod.ControlModifier)  # reference
    ctrl = _viewport(view)
    assert ctrl != start
    view.viewport = list(start)
    _drag(view, 40, 40, 70, 60, button=Btn.MiddleButton)
    assert _viewport(view) == pytest.approx(ctrl)
    view.viewport = list(start)
    _drag(view, 40, 40, 70, 60, modifiers=Mod.AltModifier)
    assert _viewport(view) == pytest.approx(ctrl)
    assert not view._pan
    # the drags added no picks or measure points
    assert view._picks == []
    assert view._points == []


@pytest.mark.parametrize("tool", ["Zoom", "Pick", "Measure"])
def test_shift_left_drags_a_zoom_rectangle_in_every_tool(view, tool):
    view._mode = tool
    x0, y0, x1, y1 = 20, 30, 84, 94
    target = view.map_to_movie(QtCore.QPoint((x0 + x1) // 2, (y0 + y1) // 2))
    corner = view.map_to_movie(QtCore.QPoint(x0, y0))
    view.mousePressEvent(_Mouse(x0, y0, modifiers=Mod.ShiftModifier))
    view.mouseMoveEvent(_Mouse(x1, y1, modifiers=Mod.ShiftModifier))
    assert view.rubberband.isVisible()
    assert view.rubberband.geometry() == QtCore.QRect(
        QtCore.QPoint(x0, y0), QtCore.QPoint(x1, y1)
    )
    view.mouseReleaseEvent(_Mouse(x1, y1, modifiers=Mod.ShiftModifier))
    assert not view.rubberband.isVisible()
    # the rectangle became the field of view: its center is the new
    # center and it is widened to the window's aspect ratio, so one of
    # its corners is on the new field's edge
    cy, cx = render.viewport_center(view.viewport)
    assert (cx, cy) == pytest.approx(target, abs=1e-6)
    (y_min, x_min), _ = view.viewport
    assert x_min == pytest.approx(
        corner[0], abs=1e-6
    ) or y_min == pytest.approx(corner[1], abs=1e-6)
    vh, vw = render.viewport_size(view.viewport)
    assert vw / vh == pytest.approx(view.width() / view.height(), rel=1e-6)
    assert view._picks == []
    assert view._points == []


def test_dragging_up_or_left_cancels_the_rectangle(view):
    view._mode = "Pick"
    before = _viewport(view)
    _drag(view, 80, 80, 20, 20, modifiers=Mod.ShiftModifier)
    assert not view.rubberband.isVisible()
    assert _viewport(view) == before
    assert view._picks == []


def test_triple_click_fits_with_the_zoom_tool_only(view):
    view.fit_in_view()
    fitted = _viewport(view)
    view._mode = "Zoom"
    view.zoom_in()
    view.zoom_in()
    zoomed = _viewport(view)
    # a double click is not enough
    view.mousePressEvent(_Mouse(50, 50))
    view.mouseReleaseEvent(_Mouse(50, 50))
    view.mouseDoubleClickEvent(_Mouse(50, 50))
    view.mouseReleaseEvent(_Mouse(50, 50))
    assert _viewport(view) == zoomed
    _triple_click(view, 50, 50)
    assert _viewport(view) == pytest.approx(fitted)
    assert not view.rubberband.isVisible()
    # picking keeps its clicks: a triple click adds no fit
    view._mode = "Pick"
    view.zoom_in()
    zoomed = _viewport(view)
    _triple_click(view, 50, 50)
    assert _viewport(view) == zoomed


def test_home_fits_the_image_to_the_window(view):
    view.fit_in_view()
    fitted = _viewport(view)
    view.zoom_in()
    action = next(
        a
        for a in view.window.findChildren(QtCore.QObject)
        if hasattr(a, "shortcuts") and a.text() == "Fit image to window"
    )
    assert QtGui.QKeySequence("Home") in action.shortcuts()
    assert QtGui.QKeySequence("Ctrl+W") in action.shortcuts()
    action.trigger()
    assert _viewport(view) == pytest.approx(fitted)
