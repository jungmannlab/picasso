"""The jittered-triangulation blur method in Render's GUI: the
display-settings option and its settings, the limit on the
localizations in view (histogram fallback with a note), previews as
the plain pass, and the 3D window's sync.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtWidgets

from picasso import lib, render
from picasso.gui import render as gui_render
from picasso.gui.render_worker import subsample_request

from tests.test_gui_render_async import HEIGHT, WIDTH, _info, _locs
from tests.test_gui_render_quadtree import _select


@pytest.fixture
def window(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    window.view.async_rendering = False
    _select(window.display_settings_dlg, "triangulation")
    yield window
    window.view.stop_render_worker()


def test_dialog_offers_the_method_with_its_settings(window):
    dialog = window.display_settings_dlg
    assert "triangulation" in dialog.blur_methods.values()
    assert not dialog.triangulation_widgets.isHidden()
    assert dialog.min_blur_widgets.isHidden()
    assert dialog.quadtree_widgets.isHidden()
    assert (
        dialog.triangulation_passes.value()
        == lib.RENDER_TRIANGULATION_PASSES_DEFAULT
    )
    assert (
        dialog.triangulation_jitter.value()
        == lib.RENDER_TRIANGULATION_JITTER_DEFAULT
    )
    assert (
        dialog.triangulation_max_locs.value()
        == lib.RENDER_TRIANGULATION_MAX_LOCS_DEFAULT
    )
    dialog.triangulation_passes.setValue(3)
    dialog.triangulation_jitter.setValue(0.5)
    kwargs = window.view.get_render_kwargs()
    assert kwargs["blur_method"] == "triangulation"
    assert kwargs["triangulation_passes"] == 3
    assert kwargs["triangulation_jitter"] == 0.5
    assert (
        window.view.get_render_kwargs(blur_method="Jittered triangulation")[
            "blur_method"
        ]
        == "triangulation"
    )
    _select(dialog, "gaussian")
    assert dialog.triangulation_widgets.isHidden()


def test_renders_like_the_library_and_previews_the_plain_pass(window):
    view = window.view
    dialog = window.display_settings_dlg
    dialog.triangulation_passes.setValue(3)
    view.update_scene()
    n, expected = render.render(
        view.locs[0],
        view.infos[0],
        disp_px_size=dialog.disp_px_size.value(),
        viewport=view._image_viewport,
        blur_method="triangulation",
        triangulation_passes=3,
    )
    np.testing.assert_allclose(view.image, expected, rtol=1e-5, atol=1e-5)
    assert dialog.triangulation_note.text() == ""
    # a preview keeps every row and renders the single plain pass
    request, _ = view._build_render_request()
    request["contrast"] = (0.0, 1.0)
    assert subsample_request(request, lambda population: 10)
    assert request["triangulation_passes"] == 0
    assert len(request["locs"]) == len(view.locs[0])


def test_too_many_localizations_fall_back_to_the_histogram(window):
    view = window.view
    dialog = window.display_settings_dlg
    dialog.triangulation_max_locs.setValue(1000)  # fewer than loaded
    view.update_scene()
    _, hist = render.render(
        view.locs[0],
        view.infos[0],
        disp_px_size=dialog.disp_px_size.value(),
        viewport=view._image_viewport,
    )
    np.testing.assert_array_equal(view.image, hist)
    assert "exceed the limit" in dialog.triangulation_note.text()
    assert f"{len(view.locs[0]):,}" in dialog.triangulation_note.text()
    dialog.triangulation_max_locs.setValue(100_000)
    view.update_scene()
    assert dialog.triangulation_note.text() == ""


def test_3d_window_syncs_and_renders(window, monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    view = window.view
    dialog = window.display_settings_dlg
    dialog.triangulation_passes.setValue(2)
    dialog.triangulation_jitter.setValue(0.5)
    view.locs[0] = view.locs[0].assign(z=np.zeros(len(view.locs[0])))
    view._picks = []  # the field of view: all 2000 rows are loaded
    window.open_3d_view()
    rot = window.window_rot
    rdialog = rot.display_settings_dlg
    assert rdialog.blur_method() == "triangulation"
    assert not rdialog.triangulation_widgets.isHidden()
    assert rdialog.triangulation_passes.value() == 2
    assert rdialog.triangulation_jitter.value() == 0.5
    kwargs = rot.view_rot.get_render_kwargs()
    assert kwargs["triangulation_passes"] == 2
    rot.view_rot.apply_rotation(np.array([0.3, 0.2, 0.0]))
    rot.view_rot.update_scene(synchronous=True)
    request = rot.view_rot._build_render_request()
    assert request["blur_method"] == "triangulation"
    _, n, _, raw = render.render_scene(**request)
    assert n > 0
    np.testing.assert_allclose(rot.view_rot.image, raw, rtol=1e-5, atol=1e-5)
    # the 3D limit counts the loaded rows (2000 here)
    rdialog.triangulation_max_locs.setValue(1000)
    request = rot.view_rot._build_render_request()
    assert request["blur_method"] is None
    assert "exceed the limit" in rdialog.triangulation_note.text()
    rot.view_rot.stop_render_worker()
