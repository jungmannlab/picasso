"""The adaptive histogram (quad-tree) blur method in Render's GUI: the
display-settings option, the request it builds (whole channels plus
their spatial index, no preview subsampling) and the 3D window's
substitution of the plain histogram.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import lib, render, spatial_index
from picasso.gui import render as gui_render
from picasso.gui.render_worker import subsample_request

from tests.test_gui_render_async import HEIGHT, WIDTH, _info, _locs


def _select(dialog, method):
    for button, name in dialog.blur_methods.items():
        if name == method:
            button.setChecked(True)
            return
    raise AssertionError(f"no button for {method}")


@pytest.fixture
def window(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    window.view.async_rendering = False
    _select(window.display_settings_dlg, "quadtree")
    yield window
    window.view.stop_render_worker()


def test_settings_show_only_while_the_method_is_selected(window):
    dialog = window.display_settings_dlg
    assert not dialog.quadtree_widgets.isHidden()  # quadtree selected
    assert dialog.min_blur_widgets.isHidden()  # no blur to bound
    _select(dialog, "gaussian")
    assert dialog.quadtree_widgets.isHidden()
    assert not dialog.quadtree_capacity.isVisibleTo(dialog)
    assert not dialog.min_blur_widgets.isHidden()
    _select(dialog, "quadtree")
    assert not dialog.quadtree_widgets.isHidden()


@pytest.mark.parametrize(
    "method,min_blur",
    [
        (None, False),
        ("smooth", False),
        ("convolve", True),
        ("gaussian", True),
        ("gaussian_iso", True),
        ("quadtree", False),
    ],
)
def test_minimum_blur_shows_only_where_it_applies(window, method, min_blur):
    dialog = window.display_settings_dlg
    _select(dialog, method)
    assert dialog.min_blur_widgets.isHidden() is not min_blur
    assert dialog.quadtree_widgets.isHidden() is not (method == "quadtree")


def test_dialog_offers_the_method_with_its_capacity(window):
    dialog = window.display_settings_dlg
    assert "quadtree" in dialog.blur_methods.values()
    assert (
        dialog.quadtree_capacity.value()
        == lib.RENDER_QUADTREE_CAPACITY_DEFAULT
    )
    assert "2.2" in dialog.quadtree_snr.text()
    dialog.quadtree_capacity.setValue(18)
    assert "3.0" in dialog.quadtree_snr.text()
    kwargs = window.view.get_render_kwargs()
    assert kwargs["blur_method"] == "quadtree"
    assert kwargs["quadtree_capacity"] == 18
    # the export dialog's display name maps back to the method
    assert (
        window.view.get_render_kwargs(
            blur_method="Adaptive hist. (quad-tree)"
        )["blur_method"]
        == "quadtree"
    )


def test_request_carries_whole_channels_and_their_index(window):
    view = window.view
    view.viewport = [(8.0, 8.0), (16.0, 16.0)]  # zoomed in: pyramid slices
    request, _ = view._build_render_request()
    assert request["blur_method"] == "quadtree"
    assert request["indices"] is None
    assert len(request["locs"]) == len(view.locs[0])  # not sliced
    assert isinstance(
        request["render_index"], spatial_index.RenderIndexPyramid
    )
    assert request["render_index"] is view.render_index[0]
    # previews render the whole request too
    request["contrast"] = (0.0, 1.0)
    assert not subsample_request(request, lambda population: 10)
    assert len(request["locs"]) == len(view.locs[0])


def test_renders_like_the_library(window):
    view = window.view
    view.update_scene()
    image = view.image
    n, expected = render.render(
        view.locs[0],
        view.infos[0],
        disp_px_size=view.window.display_settings_dlg.disp_px_size.value(),
        viewport=view._image_viewport,
        blur_method="quadtree",
        quadtree_capacity=lib.RENDER_QUADTREE_CAPACITY_DEFAULT,
    )
    np.testing.assert_allclose(image, expected, rtol=1e-5, atol=1e-5)
    assert image.sum() == pytest.approx(n, rel=1e-3)


def test_3d_window_renders_the_quadtree_too(window, monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    view = window.view
    window.display_settings_dlg.quadtree_capacity.setValue(7)
    view.locs[0] = view.locs[0].assign(z=np.zeros(len(view.locs[0])))
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(12.0 * 130.0)
    view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    rot = window.window_rot
    dialog = rot.display_settings_dlg
    # synced by button id, enabled, with the main window's capacity
    assert dialog.blur_method() == "quadtree"
    assert dialog.blur_buttongroup.checkedButton().isEnabled()
    assert not dialog.quadtree_widgets.isHidden()
    assert dialog.min_blur_widgets.isHidden()  # the 3D dialog hides it too
    assert dialog.quadtree_capacity.value() == 7
    kwargs = rot.view_rot.get_render_kwargs()
    assert kwargs["blur_method"] == "quadtree"
    assert kwargs["quadtree_capacity"] == 7
    # a rotated render works and is linear in the count
    rot.view_rot.apply_rotation(np.array([0.4, 0.2, 0.0]))
    rot.view_rot.update_scene(synchronous=True)
    image = rot.view_rot.image
    request = rot.view_rot._build_render_request()
    _, n, _, raw = render.render_scene(**request)
    assert n > 0
    np.testing.assert_allclose(image, raw, rtol=1e-5, atol=1e-5)
    assert image.sum() == pytest.approx(n, rel=1e-3)
    # rotated previews may subsample (the tree is rebuilt per frame)
    request = rot.view_rot._build_render_request()
    request["contrast"] = (0.0, 1.0)
    assert request["ang"] is not None
    assert subsample_request(request, lambda population: 10)
    rot.view_rot.stop_render_worker()
