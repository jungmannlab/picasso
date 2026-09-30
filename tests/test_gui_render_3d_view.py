"""Entering the 3D view from Picasso: Render (``View > 3D view``).

One entry point that always works: with exactly one pick selected it
opens that pick in the rotation window, as ``Update rotation window``
did; otherwise it opens the current field of view, rotated about its
center. Opened again on unchanged content it only raises the window,
keeping the rotation.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtGui, QtWidgets

from picasso.gui import render as gui_render
from picasso.gui import rotation


WIDTH = HEIGHT = 64.0
PIXELSIZE = 130.0


def _locs(n: int = 3000, seed: int = 0, with_z: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locs = pd.DataFrame(
        {
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )
    if with_z:
        locs["z"] = rng.normal(0.0, 20.0, size=n)  # nm
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


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        window.view.add(
            str(tmp_path / f"locs{i}.hdf5"),
            _locs(seed=i),
            _info(),
            render_=False,
        )
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()


def _in_view(locs: pd.DataFrame, viewport) -> int:
    (y_min, x_min), (y_max, x_max) = viewport
    return int(
        (
            (locs["x"] >= x_min)
            & (locs["x"] < x_max)
            & (locs["y"] >= y_min)
            & (locs["y"] < y_max)
        ).sum()
    )


def test_one_pick_opens_that_pick(window):
    view = window.view
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(10.0 * PIXELSIZE)
    view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    view_rot = window.window_rot.view_rot
    assert window.window_rot.isVisible()
    assert view_rot._source == "pick"
    assert view_rot.pick_shape == "Circle"
    assert view_rot.pick == (32.0, 32.0)
    for i in range(2):
        expected = len(view.picked_locs(i, add_group=False)[0])
        assert len(view_rot.locs[i]) == expected
    # the z column arrives in camera pixels
    assert view_rot.locs[0]["z"].abs().max() < 1.0


def test_no_pick_opens_the_field_of_view(window):
    view = window.view
    zoomed = [(10.0, 20.0), (30.0, 40.0)]
    view.viewport = zoomed
    window.open_3d_view()
    view_rot = window.window_rot.view_rot
    assert window.window_rot.isVisible()
    assert view_rot._source == "fov"
    assert view_rot.pick_shape is None and view_rot.pick is None
    # the field of view is the main window's, at most widened to the
    # window's aspect ratio
    (y0, x0), (y1, x1) = view_rot.viewport
    assert y0 <= 10.0 and x0 <= 20.0 and y1 >= 30.0 and x1 >= 40.0
    for i in range(2):
        n = len(view_rot.locs[i])
        assert _in_view(view.locs[i], zoomed) <= n <= len(view.locs[i])
        # nothing far away from the field of view is copied
        assert n < 0.6 * len(view.locs[i])
    assert view_rot.rotation.as_quat()[3] == pytest.approx(1.0)  # identity


def test_several_picks_count_as_none(window):
    view = window.view
    view._pick_shape = "Circle"
    view._picks = [(20.0, 20.0), (40.0, 40.0)]
    window.open_3d_view()
    assert window.window_rot.view_rot._source == "fov"


def test_opening_again_on_the_same_content_only_raises(window):
    view = window.view
    view.viewport = [(10.0, 20.0), (30.0, 40.0)]
    window.open_3d_view()
    view_rot = window.window_rot.view_rot
    loaded = view_rot.locs
    view_rot.apply_rotation(np.array([0.2, 0.3, 0.0]))
    rotation_before = view_rot.rotation.as_quat().copy()
    window.open_3d_view()
    assert view_rot.locs is loaded  # not reloaded
    np.testing.assert_allclose(view_rot.rotation.as_quat(), rotation_before)
    # a changed field of view reloads (and resets the rotation)
    view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.open_3d_view()
    assert view_rot.locs is not loaded
    assert len(view_rot.locs[0]) == len(view.locs[0])
    # so does switching to a pick
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(10.0 * PIXELSIZE)
    view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    assert view_rot._source == "pick"
    assert view_rot.pick_shape == "Circle"


def test_arrow_pan_in_the_field_of_view_keeps_following(window):
    view = window.view
    view.viewport = [(10.0, 20.0), (30.0, 40.0)]
    window.open_3d_view()
    view_rot = window.window_rot.view_rot
    before = [(y0, x0) for (y0, x0), _ in [view_rot.viewport]]
    view_rot._arrow_pan(5.0, 0.0)  # screen x, no rotation: world x
    (y0, x0), _ = view_rot.viewport
    assert x0 == pytest.approx(before[0][1] + 5.0)
    assert y0 == pytest.approx(before[0][0])
    assert view.viewport == [(10.0, 20.0), (30.0, 40.0)]  # main untouched
    assert view._picks == []  # no pick was invented


def test_fit_in_view_works_without_a_pick(window):
    view = window.view
    loaded = [(10.0, 20.0), (30.0, 40.0)]
    view.viewport = loaded
    window.open_3d_view()
    view_rot = window.window_rot.view_rot
    fitted = view_rot.viewport
    for _ in range(3):
        view_rot.zoom_in()
    assert view_rot.viewport != fitted
    view_rot.fit_in_view_rotated()
    # back to the loaded field of view (widened to the window's aspect)
    (y0, x0), (y1, x1) = view_rot.viewport
    assert y0 <= 10.0 and x0 <= 20.0 and y1 >= 30.0 and x1 >= 40.0
    assert y1 - y0 == pytest.approx(fitted[1][0] - fitted[0][0], rel=1e-6)
    # the pick-based fit is unchanged
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(10.0 * PIXELSIZE)
    view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    view_rot.zoom_in()
    view_rot.fit_in_view_rotated()
    (y0, x0), (y1, x1) = view_rot.viewport
    assert y0 <= 27.0 and y1 >= 37.0  # the circle's 10 px diameter


def test_two_dimensional_data_is_refused(qt_offscreen, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: shown.append(a)),
    )
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"),
        _locs(with_z=False),
        _info(),
        render_=False,
    )
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.open_3d_view()
    assert len(shown) == 1 and "z" in shown[0][2]
    assert not window.window_rot.isVisible()
    window.view.stop_render_worker()


def test_menu_entry_and_shortcut(window):
    actions = [
        a for a in window.findChildren(QtGui.QAction) if a.text() == "3D view"
    ]
    assert len(actions) == 1
    assert actions[0].shortcut().toString() == "Ctrl+Shift+R"
    assert rotation.source_key(window.view, "fov")[0] == "fov"
