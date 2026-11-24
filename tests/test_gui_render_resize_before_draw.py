"""Regression test for an early resize of Render's ``View`` that
fires before the first draw.

Localizations load on a background thread (``LocsLoadWorker``). Once
they populate, ``n_channels`` becomes truthy, but ``self.viewport`` is
only set lazily by the first ``draw_scene``. If a ``resizeEvent`` (->
``update_scene``) fires in the gap before that first draw, the viewport
must not be read as a missing/None attribute: that used to raise
``AttributeError: 'View' object has no attribute 'viewport'``. The view
now falls back to the full field of view instead.

:author: Heinrich Grabmayr, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest

from picasso.gui import render as gui_render

from tests._render_data import HEIGHT_NM, WIDTH_NM, make_info, make_locs


@pytest.fixture
def view_before_draw(qt_offscreen, tmp_path):
    """A ``View`` with one loaded channel but no prior draw, i.e.
    ``self.viewport`` is still ``None`` (the state right after a
    background locs load, before the first full render)."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, make_locs(), make_info(), render_=False)
    # a square canvas over the square field of view, so the viewport the
    # draw settles on equals the full FOV exactly (no aspect padding, see
    # ``adjust_viewport_to_view``).
    window.view.resize(128, 128)
    yield window.view
    window.view.stop_render_worker()


def test_update_scene_before_first_draw_uses_full_fov(view_before_draw):
    """``update_scene`` from an early resize must not raise and must
    render the full field of view when no viewport has been set yet."""
    view = view_before_draw
    # the loaded channel makes ``n_channels`` truthy ...
    assert view.n_channels > 0
    # ... yet no draw has run, so the viewport is still uninitialised.
    assert view.viewport is None

    # a resizeEvent triggers update_scene() with no explicit viewport;
    # this must not raise AttributeError.
    view.update_scene()

    assert view.viewport == view.full_fov_viewport()
    assert view.viewport == [(0, 0), (HEIGHT_NM, WIDTH_NM)]


def test_update_scene_slicer_before_first_draw_does_not_raise(
    view_before_draw,
):
    """``update_scene_slicer`` shares the same fallback path."""
    view = view_before_draw
    assert view.viewport is None
    view.update_scene_slicer()
    assert view.viewport == view.full_fov_viewport()
