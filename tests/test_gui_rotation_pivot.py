"""The rotation pivot of the 3D window stays on the data.

Rotate a little, zoom in deep, pan far, rotate again: the structure at
the screen center used to orbit, because a screen-space pan of a tilted
view has a z component that accumulated in the view target's depth and
left the pivot in front of or behind the data. After every pan or zoom
the pivot is slid along the viewing direction (which leaves the image
unchanged) to the median depth of the localizations in view.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtWidgets

from picasso import render
from picasso.gui import render as gui_render

from tests.test_gui_rotation_async import HEIGHT, PIXELSIZE, WIDTH, _info


def _flat_locs(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Localizations in a thin layer (z spread far below their x/y
    extent), so the pivot's depth is unambiguous."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(24.0, 40.0, size=n),
            "y": rng.uniform(24.0, 40.0, size=n),
            "z": rng.normal(0.0, 5.0, size=n),  # nm
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "lpz": rng.uniform(10.0, 60.0, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


class _Event:
    def __init__(self, x, y):
        self._pos = QtCore.QPoint(x, y)

    def pos(self):
        return self._pos


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


def _marker_at_center(view) -> tuple[float, float]:
    """Relative image position of a marker localization placed at the
    world point the view rotates about."""
    request = view._build_render_request()
    (y_min, x_min), (y_max, x_max) = view.viewport
    marker = pd.DataFrame(
        {
            "x": [x_min + (x_max - x_min) / 2],
            "y": [y_min + (y_max - y_min) / 2],
            "z": [0.0],  # the request's z is already shifted by -_pan_z
            "lpx": [0.02],
            "lpy": [0.02],
            "lpz": [0.02],
        }
    )
    request.update(locs=marker, blur_method=None, contrast=(0.0, 1.0))
    _, _, _, raw = render.render_scene(**request)
    iy, ix = np.unravel_index(int(np.argmax(raw)), raw.shape)
    return ix / raw.shape[1], iy / raw.shape[0]


def _tilt_zoom_pan(view):
    view.set_rotation(render.rotation_matrix(np.pi / 4, 0.3, 0.0))
    for _ in range(4):
        view.zoom_in()
    view.pan_start_x, view.pan_start_y = 200, 150
    view._pan_drag(_Event(320, 210))  # 120 x 60 px, far at this zoom


def test_pivot_stays_at_the_depth_of_the_data(view):
    _tilt_zoom_pan(view)
    # the localizations are flat around z = 0 (camera px, mean-shifted):
    # the pivot's depth must be there, not where the pan's z drifted
    assert abs(view._pan_z) < 0.05
    # and the pivot is what the screen center shows
    x, y = _marker_at_center(view)
    assert abs(x - 0.5) < 0.01 and abs(y - 0.5) < 0.01


def test_rotating_after_the_pan_keeps_the_center_fixed(view):
    _tilt_zoom_pan(view)
    before = view.image.copy()
    center_before = _marker_at_center(view)
    view.apply_rotation(np.array([0.0, 0.5, 0.0]))
    view.update_scene()
    assert not np.array_equal(view.image, before)  # it did rotate
    assert _marker_at_center(view) == pytest.approx(center_before, abs=0.01)


def test_reanchoring_does_not_change_the_image(view):
    # a pivot displaced along the viewing direction renders the same
    # image; re-anchoring moves it back onto the data and nothing else
    view.set_rotation(render.rotation_matrix(np.pi / 4, 0.3, 0.0))
    for _ in range(3):
        view.zoom_in()
    direction = view.rotation.inv().apply([0.0, 0.0, 1.0])
    t = 1.5  # camera px along the viewing direction
    (y0, x0), (y1, x1) = view.viewport
    view.viewport = [
        (y0 + t * direction[1], x0 + t * direction[0]),
        (y1 + t * direction[1], x1 + t * direction[0]),
    ]
    view._pan_z += t * direction[2]
    view.update_scene()
    displaced = view.image.copy()
    assert abs(view._pan_z) > 0.5
    view._reanchor_pivot()
    assert abs(view._pan_z) < 0.05
    view.update_scene()
    # the shifted viewport can round to one pixel more or less in the
    # rendered size; the content over the common area is the same
    h = min(view.image.shape[0], displaced.shape[0])
    w = min(view.image.shape[1], displaced.shape[1])
    assert abs(view.image.shape[0] - displaced.shape[0]) <= 1
    assert abs(view.image.shape[1] - displaced.shape[1]) <= 1
    np.testing.assert_allclose(
        view.image[:h, :w], displaced[:h, :w], rtol=1e-4, atol=1e-4
    )


def test_edge_on_views_leave_the_pivot_alone(view):
    view.set_rotation(render.rotation_matrix(np.pi / 2, 0.0, 0.0))
    view._pan_z = 0.7
    viewport = [tuple(v) for v in view.viewport]
    view._reanchor_pivot()  # no depth along a ray parallel to the data
    assert view._pan_z == 0.7
    assert [tuple(v) for v in view.viewport] == viewport


def test_no_rotation_no_drift(view):
    for _ in range(4):
        view.zoom_in()
    view.pan_start_x, view.pan_start_y = 200, 150
    view._pan_drag(_Event(320, 210))
    assert view._pan_z == pytest.approx(0.0, abs=0.05)
    x, y = _marker_at_center(view)
    assert abs(x - 0.5) < 0.01 and abs(y - 0.5) < 0.01
