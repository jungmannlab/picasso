"""Linked Render windows (``picasso.gui.render_link``): two windows with
separate channels mirror the attributes enabled in the link group.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtWidgets

from picasso import io, render
from picasso.gui import render as gui_render
from picasso.gui import render_link

from tests.test_gui_render_async import (
    HEIGHT,
    WIDTH,
    _info,
    _locs,
    _wait_until,
)
from tests.test_gui_rotation_navigation import _Mouse


def _load(window, tmp_path, name, size, seed=0, z=False):
    locs = _locs(seed=seed)
    if z:
        rng = np.random.default_rng(seed)
        locs["z"] = rng.uniform(-300.0, 300.0, size=len(locs))
    path = str(tmp_path / f"{name}.hdf5")
    window.view.add(path, locs, _info(), render_=False)
    window.view.resize(*size)
    window.view.async_rendering = False
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.update_scene()


def _close(window):
    window.view.stop_render_worker()
    window.window_rot.view_rot.stop_render_worker()


@pytest.fixture
def linked(qt_offscreen, tmp_path):
    """Two linked windows of different sizes, each with its own
    channel; only pan and zoom are linked (the defaults)."""
    primary = gui_render.Window(plugins_loaded=True)
    _load(primary, tmp_path, "a", (128, 128), seed=0)
    group = render_link.LinkGroup(primary)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    _load(secondary, tmp_path, "b", (200, 100), seed=1)
    yield group, primary, secondary
    for window in (primary, secondary):
        _close(window)


def _center_scale(view):
    center = render.viewport_center(view.viewport)
    scale = render.viewport_height(view.viewport) / view.height()
    return center, scale


def _count_calls(monkeypatch, view, name):
    calls = []
    original = getattr(view, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(view, name, wrapper)
    return calls


def test_defaults(linked):
    group, primary, secondary = linked
    assert group.enabled == {
        "localizations",
        "viewport_center",
        "viewport_zoom",
    }
    assert group.windows == [primary, secondary]
    assert primary.link_group is group and secondary.link_group is group
    assert secondary.windowTitle().endswith("(linked view 2)")


def test_pan_and_zoom_follow_across_window_sizes(linked):
    _, primary, secondary = linked
    primary.view.pan_relative(0.2, 0.1)
    primary.view.zoom_in()
    (cy, cx), scale = _center_scale(primary.view)
    (sy, sx), s_scale = _center_scale(secondary.view)
    assert (sy, sx) == pytest.approx((cy, cx))
    assert s_scale == pytest.approx(scale)
    # the other window keeps its own aspect ratio
    s_height, s_width = render.viewport_size(secondary.view.viewport)
    assert s_width / s_height == pytest.approx(200 / 100)

    # and the other way round
    secondary.view.pan_relative(-0.3, 0.0)
    assert _center_scale(primary.view)[0] == pytest.approx(
        _center_scale(secondary.view)[0]
    )


def test_center_only_keeps_the_own_scale(linked):
    group, primary, secondary = linked
    group.set_enabled("viewport_zoom", False)
    _, own_scale = _center_scale(secondary.view)
    primary.view.zoom_in()
    primary.view.pan_relative(0.2, 0.2)
    center, _ = _center_scale(primary.view)
    s_center, s_scale = _center_scale(secondary.view)
    assert s_center == pytest.approx(center)
    assert s_scale == pytest.approx(own_scale)


def test_unlinked_categories_do_not_propagate(linked):
    group, primary, secondary = linked
    group.set_enabled("viewport_center", False)
    group.set_enabled("viewport_zoom", False)
    before = [tuple(v) for v in secondary.view.viewport]
    primary.view.pan_relative(0.2, 0.1)
    primary.display_settings_dlg.colormap.setCurrentText("viridis")
    assert [tuple(v) for v in secondary.view.viewport] == before
    assert secondary.display_settings_dlg.colormap.currentText() != "viridis"


def test_a_change_propagates_exactly_once(linked, monkeypatch):
    _, primary, secondary = linked
    primary_calls = _count_calls(monkeypatch, primary.view, "update_scene")
    secondary_calls = _count_calls(monkeypatch, secondary.view, "update_scene")
    primary.view.pan_relative(0.2, 0.1)
    assert len(primary_calls) == 1  # no echo back
    assert len(secondary_calls) == 1
    assert secondary_calls[0][1]["interactive"] is True


def test_picks_are_mirrored(linked):
    group, primary, secondary = linked
    group.set_enabled("picks", True)
    primary.view.add_pick((10.0, 12.0))
    assert secondary.view._picks == [(10.0, 12.0)]
    secondary.view.add_pick((30.0, 31.0))
    assert primary.view._picks == [(10.0, 12.0), (30.0, 31.0)]
    primary.view.clear_picks()
    assert secondary.view._picks == []


def test_polygons_extended_in_place_are_mirrored_as_copies(linked):
    group, primary, secondary = linked
    group.set_enabled("picks", True)
    view = primary.view
    view._pick_shape = "Polygon"
    view._picks = [[(1.0, 1.0), (5.0, 1.0)]]
    view.update_scene(picks_only=True)
    assert secondary.view._pick_shape == "Polygon"
    assert secondary.view._picks == view._picks
    assert secondary.view._picks[0] is not view._picks[0]
    view._picks[0].append((5.0, 5.0))
    view.update_scene(picks_only=True)
    assert secondary.view._picks == [[(1.0, 1.0), (5.0, 1.0), (5.0, 5.0)]]


def test_pick_shape_and_size_are_mirrored(linked):
    group, primary, secondary = linked
    group.set_enabled("pick_geometry", True)
    tools = primary.tools_settings_dialog
    tools.pick_shape.setCurrentText("Square")
    tools.pick_side_length.setValue(321.0)
    s_tools = secondary.tools_settings_dialog
    assert s_tools.pick_shape.currentText() == "Square"
    assert secondary.view._pick_shape == "Square"
    assert s_tools.pick_side_length.value() == pytest.approx(321.0)


def test_contrast_is_mirrored_including_autoscale(linked):
    group, primary, secondary = linked
    group.set_enabled("contrast", True)
    d = primary.display_settings_dlg
    s = secondary.display_settings_dlg
    d.maximum.setValue(d.maximum.value() * 3)
    assert s.maximum.value() == pytest.approx(d.maximum.value())
    # an automatic adjustment is written back silently, and mirrored
    primary.view.update_scene(autoscale=True)
    assert s.minimum.value() == pytest.approx(d.minimum.value())
    assert s.maximum.value() == pytest.approx(d.maximum.value())


def test_display_and_render_settings_are_mirrored(linked):
    group, primary, secondary = linked
    for key in ("colormap", "render_settings", "scalebar", "minimap"):
        group.set_enabled(key, True)
    d = primary.display_settings_dlg
    s = secondary.display_settings_dlg
    d.colormap.setCurrentText("viridis")
    d.blur_buttongroup.buttons()[0].click()
    d.min_blur_width.setValue(0.5)
    d.scalebar_groupbox.setChecked(True)
    d.minimap.setChecked(True)
    assert s.colormap.currentText() == "viridis"
    assert s.blur_buttongroup.buttons()[0].isChecked()
    assert s.min_blur_width.value() == pytest.approx(0.5)
    assert s.scalebar_groupbox.isChecked()
    assert s.minimap.isChecked()


def test_tool_mode_is_mirrored(linked):
    group, primary, secondary = linked
    group.set_enabled("tool_mode", True)
    pick = next(
        a for a in primary.tools_actiongroup.actions() if a.text() == "Pick"
    )
    pick.trigger()
    assert secondary.view._mode == "Pick"


def test_crosshair_follows_the_cursor(linked):
    group, primary, secondary = linked
    group.set_enabled("crosshair", True)
    primary.view.mouseMoveEvent(_Mouse(40, 50))
    expected = primary.view.map_to_movie(QtCore.QPoint(40, 50))
    assert secondary.view._link_crosshair == pytest.approx(expected)
    assert primary.view._link_crosshair is None
    primary.view.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert secondary.view._link_crosshair is None


def test_slice_position_is_matched_in_nm(qt_offscreen, tmp_path):
    primary = gui_render.Window(plugins_loaded=True)
    _load(primary, tmp_path, "a", (128, 128), seed=0, z=True)
    group = render_link.LinkGroup(primary)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    _load(secondary, tmp_path, "b", (128, 128), seed=1, z=True)
    try:
        group.set_enabled("slicer", True)
        slicer = primary.slicer_dialog
        slicer.initialize()
        slicer.sl.setValue(2)
        s_slicer = secondary.slicer_dialog
        assert s_slicer.slicer_radio_button.isChecked()
        assert s_slicer.pick_slice.value() == slicer.pick_slice.value()
        z_min = slicer.slicermin
        nearest = np.min(np.abs(s_slicer.bins[:-1] - z_min))
        assert abs(s_slicer.slicermin - z_min) == pytest.approx(nearest)
        slicer.close()
    finally:
        for window in (primary, secondary):
            _close(window)


def test_first_load_adopts_the_group(linked, tmp_path):
    group, primary, secondary = linked
    primary.view.pan_relative(0.2, 0.1)
    primary.view.zoom_in()
    third = gui_render.Window(plugins_loaded=True, link_group=group)
    try:
        third.view.async_rendering = False
        third.view.add(
            str(tmp_path / "c.hdf5"), _locs(seed=2), _info(), render_=False
        )
        third.view.resize(90, 150)
        group.adopt(third)
        center, scale = _center_scale(primary.view)
        t_center, t_scale = _center_scale(third.view)
        assert t_center == pytest.approx(center)
        assert t_scale == pytest.approx(scale)
        # the first render did not move the other windows
        assert _center_scale(primary.view)[0] == pytest.approx(center)
    finally:
        _close(third)


def test_remove_locs_reattaches_the_rebuilt_view(linked, tmp_path):
    group, primary, secondary = linked
    secondary.remove_locs()
    assert secondary.link_group is group
    _load(secondary, tmp_path, "b2", (128, 128), seed=3)
    primary.view.pan_relative(0.25, 0.0)
    assert _center_scale(secondary.view)[0] == pytest.approx(
        _center_scale(primary.view)[0]
    )


def test_closing_a_linked_window_leaves_the_others_open(linked):
    group, primary, secondary = linked
    primary.show()
    secondary.show()
    secondary.close()
    assert group.windows == [primary]
    assert secondary.link_group is None
    assert primary.isVisible()
    assert primary.windowTitle() == primary._base_title


def test_unlinked_window_stops_mirroring(linked):
    group, primary, secondary = linked
    group.remove(secondary)
    before = [tuple(v) for v in secondary.view.viewport]
    primary.view.pan_relative(0.2, 0.1)
    assert [tuple(v) for v in secondary.view.viewport] == before
    center = _center_scale(primary.view)[0]
    secondary.view.pan_relative(0.3, 0.3)
    assert _center_scale(primary.view)[0] == pytest.approx(center)


def test_linked_categories_are_saved(linked):
    group, primary, _ = linked
    group.set_enabled("contrast", True)
    group.set_enabled("viewport_zoom", False)
    assert render_link.LinkGroup._load_enabled() == {
        "localizations",
        "viewport_center",
        "contrast",
    }


def test_categories_missing_from_the_settings_take_their_defaults(
    qt_offscreen,
):
    settings = io.load_user_settings()
    settings["Render"]["LinkCategories"] = {"contrast": True}
    io.save_user_settings(settings)
    assert render_link.LinkGroup._load_enabled() == {
        "localizations",
        "viewport_center",
        "viewport_zoom",
        "contrast",
    }


def test_async_pan_and_first_load_adoption(linked, qapp, tmp_path):
    group, primary, secondary = linked
    group.set_enabled("contrast", True)
    for window in (primary, secondary):
        window.view.async_rendering = True
    primary.view.pan_relative(0.2, 0.1)  # interactive previews
    primary.view.update_scene()  # the full render ending the pan
    center = _center_scale(primary.view)[0]
    assert _center_scale(secondary.view)[0] == pytest.approx(center)

    third = gui_render.Window(plugins_loaded=True, link_group=group)
    try:
        third.view.async_rendering = True
        third.view.add(
            str(tmp_path / "c.hdf5"), _locs(seed=2), _info(), render_=False
        )
        third.view.resize(90, 150)
        group.adopt(third)
        _wait_until(
            qapp, lambda: getattr(third.view, "image", None) is not None
        )
        qapp.processEvents()
        # the new window shows the group's view and contrast, and its
        # first render moved nothing in the other windows
        assert _center_scale(third.view)[0] == pytest.approx(center)
        assert _center_scale(primary.view)[0] == pytest.approx(center)
        d = primary.display_settings_dlg
        t = third.display_settings_dlg
        assert t.maximum.value() == pytest.approx(d.maximum.value())
    finally:
        _close(third)


def test_new_linked_window_from_the_menu(qt_offscreen, qapp, tmp_path):
    paths = []
    for seed, name in enumerate(("a", "b")):
        path = str(tmp_path / f"{name}.hdf5")
        io.save_locs(path, _locs(seed=seed), _info(), render_index=False)
        paths.append(path)
    primary = gui_render.Window(plugins_loaded=True)
    primary.resize(400, 300)
    primary.show()
    primary.view.add_multiple(paths)
    _wait_until(qapp, lambda: len(primary.view.locs) == 2)
    _wait_until(qapp, lambda: primary.view._load_thread is None)
    primary.view.async_rendering = False
    primary.view.zoom_in()
    # an unsaved change, which the new window must show
    source = primary.view.locs[1]
    primary.view.locs[1] = source[source["x"] < WIDTH / 2].reset_index(
        drop=True
    )
    primary.view.invalidate_locs_index(1)
    primary.view._drift[1] = pd.DataFrame({"x": [0.5], "y": [0.25]})

    def accept_second(dialog):
        dialog.checks[1].setChecked(True)
        return QtWidgets.QDialog.DialogCode.Accepted

    new = None
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                render_link.NewLinkedWindowDialog, "exec", accept_second
            )
            primary.open_linked_window()
        group = primary.link_group
        assert group is not None and len(group.windows) == 2
        new = group.windows[1]
        assert new in gui_render.Window._secondary_windows
        # copied from memory right away, no file is read
        assert new.view._load_thread is None
        assert new.view.locs_paths == [paths[1]]
        pd.testing.assert_frame_equal(new.view.locs[0], primary.view.locs[1])
        pd.testing.assert_frame_equal(
            new.view._drift[0], primary.view._drift[1]
        )
        assert new.view.infos[0] == primary.view.infos[1]
        # shared by default: one DataFrame in memory
        assert new.view.locs[0] is primary.view.locs[1]
        assert new.view.infos[0] is primary.view.infos[1]
        assert new.view._drift[0] is primary.view._drift[1]
        # the new window shows the group's view
        assert _center_scale(new.view)[1] == pytest.approx(
            _center_scale(primary.view)[1]
        )
        new.close()
        assert new not in gui_render.Window._secondary_windows
        assert group.windows == [primary]
        assert primary.isVisible()
    finally:
        if new is not None:
            _close(new)
        _close(primary)


# ---------------------------------------------------------------------------
# Shared localizations
# ---------------------------------------------------------------------------


def _add(window, tmp_path, name, seed):
    path = str(tmp_path / f"{name}.hdf5")
    window.view.add(path, _locs(seed=seed), _info(), render_=False)


@pytest.fixture
def shared(qt_offscreen, tmp_path):
    """The primary window holds channels a and b; the linked window
    shares b and holds its own channel c."""
    primary = gui_render.Window(plugins_loaded=True)
    primary.view.async_rendering = False
    _add(primary, tmp_path, "a", 0)
    _add(primary, tmp_path, "b", 1)
    primary.view.resize(128, 128)
    primary.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    primary.view.update_scene()
    group = render_link.LinkGroup(primary)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    secondary.view.async_rendering = False
    secondary.view.resize(128, 128)
    secondary.view.add_channels_from(primary.view, [1], share=True)
    _add(secondary, tmp_path, "c", 2)
    secondary.view.update_scene()
    yield group, primary, secondary
    for window in (primary, secondary):
        _close(window)


def _half(locs):
    return locs[locs["x"] < WIDTH / 2].reset_index(drop=True)


def test_a_shared_channel_is_one_dataframe(shared):
    _, primary, secondary = shared
    assert secondary.view.locs[0] is primary.view.locs[1]
    assert secondary.view.infos[0] is primary.view.infos[1]
    assert secondary.view.locs_paths[0] == primary.view.locs_paths[1]


def test_replaced_channels_are_adopted_both_ways(shared, monkeypatch):
    _, primary, secondary = shared
    own_a, own_c = primary.view.locs[0], secondary.view.locs[1]
    calls = _count_calls(monkeypatch, secondary.view, "update_scene")
    # e.g. filtering assigns a new DataFrame
    primary.view.locs[1] = _half(primary.view.locs[1])
    primary.view.update_scene(resample_locs=True)
    assert secondary.view.locs[0] is primary.view.locs[1]
    assert len(calls) >= 1  # redrawn
    index = secondary.view.render_index[0]  # nothing stale
    assert index is None or len(index.perm) == len(primary.view.locs[1])
    # the channels that are not shared are untouched
    assert primary.view.locs[0] is own_a
    assert secondary.view.locs[1] is own_c

    secondary.view.locs[0] = (
        secondary.view.locs[0].iloc[::2].reset_index(drop=True)
    )
    secondary.view.update_scene(resample_locs=True)
    assert primary.view.locs[1] is secondary.view.locs[0]
    assert primary.view.locs[0] is own_a


def test_drift_applied_in_place_is_adopted(shared, monkeypatch):
    _, primary, secondary = shared
    calls = _count_calls(monkeypatch, primary.view, "update_scene")
    n_frames = _info()[0]["Frames"]
    # shifts towards larger x and y, such that the canvas stays put
    drift = pd.DataFrame(
        {"x": np.full(n_frames, -0.5), "y": np.full(n_frames, -0.25)}
    )
    x_before = primary.view.locs[1]["x"].to_numpy().copy()
    secondary.view._apply_drift(0, drift)
    assert primary.view.locs[1] is secondary.view.locs[0]
    np.testing.assert_allclose(primary.view.locs[1]["x"], x_before + 0.5)
    assert primary.view._drift[1] is secondary.view._drift[0]
    assert len(calls) >= 1


def test_moved_channels_are_redrawn(shared, monkeypatch):
    _, primary, secondary = shared
    calls = _count_calls(monkeypatch, primary.view, "update_scene")
    locs = secondary.view.locs[0]
    secondary.view._set_channel_xy(
        0, locs["x"].to_numpy() + 1.0, locs["y"].to_numpy()
    )
    secondary.view.locs_moved(0)
    assert primary.view.locs[1] is secondary.view.locs[0]
    assert len(calls) >= 1


def test_a_canvas_shift_moves_the_other_windows_channels(shared):
    _, primary, secondary = shared
    own_c_x = secondary.view.locs[1]["x"].to_numpy().copy()
    own_a_x = primary.view.locs[0]["x"].to_numpy().copy()
    # move the shared channel beyond the left edge: the canvas grows by
    # a whole number of pixels and every channel is translated with it
    locs = primary.view.locs[1]
    x = locs["x"].to_numpy()
    primary.view._set_channel_xy(1, x - x.min() - 2.5, locs["y"].to_numpy())
    primary.view.locs_moved(1)
    shift = primary.view.locs[0]["x"].to_numpy() - own_a_x
    assert shift[0] == pytest.approx(3.0)
    np.testing.assert_allclose(
        secondary.view.locs[1]["x"].to_numpy() - own_c_x, shift[0]
    )
    assert primary.view.locs[1]["x"].min() == pytest.approx(0.5)


def test_the_render_index_is_built_once(shared):
    _, primary, secondary = shared
    primary.view.render_index[1] = None
    secondary.view.render_index[0] = None
    pyramid = primary.view._ensure_render_index(1)
    assert pyramid is not None
    assert secondary.view._ensure_render_index(0) is pyramid


def test_switching_sharing_off_gives_copies(shared):
    group, primary, secondary = shared
    group.set_enabled("localizations", False)
    assert secondary.view.locs[0] is not primary.view.locs[1]
    pd.testing.assert_frame_equal(secondary.view.locs[0], primary.view.locs[1])
    assert secondary.view.infos[0] is not primary.view.infos[1]
    before = secondary.view.locs[0]
    primary.view.locs[1] = _half(primary.view.locs[1])
    primary.view.update_scene(resample_locs=True)
    assert secondary.view.locs[0] is before


def test_an_unlinked_window_gets_copies(shared):
    group, primary, secondary = shared
    group.remove(secondary)
    assert secondary.view.locs[0] is not primary.view.locs[1]
    pd.testing.assert_frame_equal(secondary.view.locs[0], primary.view.locs[1])


def test_channels_taken_without_sharing_are_copies(shared, tmp_path):
    _, primary, _ = shared
    third = gui_render.Window(plugins_loaded=True)
    try:
        third.view.async_rendering = False
        third.view.resize(128, 128)
        third.view.add_channels_from(primary.view, [0, 1], share=False)
        for j in (0, 1):
            assert third.view.locs[j] is not primary.view.locs[j]
            assert third.view.infos[j] is not primary.view.infos[j]
            pd.testing.assert_frame_equal(
                third.view.locs[j], primary.view.locs[j]
            )
    finally:
        _close(third)
