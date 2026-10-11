"""The reference grid of Render's 2D and 3D windows.

Display Settings show a grid fixed to the data in the main window and
grid planes rotating with the data in the 3D window. These tests cover
drawing in both windows, the automatic spacing, the settings grayed
out while the grid is off, the animation, the user settings file and
linked windows.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
import yaml
from PyQt6 import QtGui, QtWidgets

from picasso import io, render, lib
from picasso.gui import render as gui_render
from picasso.gui import render_link, rotation

from tests.test_gui_rotation_async import (
    HEIGHT,
    PIXELSIZE,
    WIDTH,
    _drain,
    _info,
    _locs,
    _wait_until,
)


def _rgb(image: QtGui.QImage) -> np.ndarray:
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    bgra = np.frombuffer(bits, dtype=np.uint8).reshape(
        image.height(), image.width(), 4
    )
    return bgra[:, :, 2::-1].copy()


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """A Render window showing one channel on a 256 x 256 canvas."""
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"), _locs(), _info(), render_=False
    )
    window.view.resize(256, 256)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()


def _show_grid(box, spacing_nm: float) -> None:
    box.setChecked(True)
    box.auto_spacing.setChecked(False)
    box.spacing.setValue(int(spacing_nm))


def test_2d_grid_is_drawn_at_multiples_of_the_spacing(window):
    view = window.view
    view.update_scene()
    plain = _rgb(view.qimage)
    box = window.display_settings_dlg.grid_box
    box.labels.setChecked(False)
    # one line every 8 camera pixels = 32 display pixels
    _show_grid(box, 8 * PIXELSIZE)
    grid = _rgb(view.qimage)
    # a row outside the data (the localizations span x, y = 24..40)
    row = np.abs(grid[5].astype(int) - plain[5]).sum(axis=1)
    assert list(np.flatnonzero(row)) == [0, 32, 64, 96, 128, 160, 192, 224]
    box.setChecked(False)
    assert (_rgb(view.qimage) == plain).all()


def test_2d_labels_are_printed_along_the_top_and_left_edges(window):
    view = window.view
    box = window.display_settings_dlg.grid_box
    _show_grid(box, 8 * PIXELSIZE)
    box.labels.setChecked(False)
    without = _rgb(view.qimage).astype(int)
    box.labels.setChecked(True)
    assert box.grid().labels
    labeled = np.abs(_rgb(view.qimage).astype(int) - without).sum(axis=2)
    rows, cols = np.nonzero(labeled)
    # only in the bands along the top and the left edge
    assert len(rows)
    assert np.all((rows < 30) | (cols < 80))
    # and not in the corner both bands share
    assert not labeled[:10, :10].any()


def test_3d_dialog_scrolls_like_the_main_one(view_rot):
    dialog = view_rot.window.display_settings_dlg
    assert (
        dialog.scroll_area.widget().findChild(type(dialog.grid_box))
        is dialog.grid_box
    )
    screen = QtWidgets.QApplication.primaryScreen()
    if screen is not None:
        assert dialog.height() <= screen.size().height() - 200


def test_settings_are_grayed_out_while_the_grid_is_off(window):
    box = window.display_settings_dlg.grid_box
    window.display_settings_dlg.show()
    widgets = [box.spacing, box.auto_spacing, box.blend, box.style_widget]
    assert all(w.isVisible() and not w.isEnabled() for w in widgets)
    box.setChecked(True)
    assert all(w.isVisible() and w.isEnabled() for w in widgets)
    assert not box.planes  # the planes are for the 3D window only


def test_automatic_spacing_follows_the_zoom_until_edited(window):
    view = window.view
    box = window.display_settings_dlg.grid_box
    box.setChecked(True)
    view.update_scene()
    expected = render.optimal_scalebar_length(
        PIXELSIZE, WIDTH, fraction=1 / 10
    )
    assert box.spacing.value() == expected
    view.zoom(2.0)
    zoomed = box.spacing.value()
    assert zoomed != expected
    box.spacing.setValue(1234)  # typed by the user
    assert not box.auto_spacing.isChecked()
    view.zoom(0.5)
    assert box.spacing.value() == 1234
    # ticking "Automatic" again adjusts the spacing at once
    box.auto_spacing.setChecked(True)
    assert box.spacing.value() == view.optimal_grid_spacing() == expected


def test_ticking_automatic_adjusts_the_3d_spacing_at_once(view_rot):
    box = view_rot.window.display_settings_dlg.grid_box
    box.setChecked(True)
    box.spacing.setValue(1234)
    assert not box.auto_spacing.isChecked()
    box.auto_spacing.setChecked(True)
    expected = view_rot.optimal_grid_spacing()
    assert expected is not None and expected != 1234
    assert box.spacing.value() == expected


def test_grid_appears_in_the_exported_info(window):
    box = window.display_settings_dlg.grid_box
    assert window.export_current_info(path=None)["Grid spacing (nm)"] is None
    _show_grid(box, 500)
    assert window.export_current_info(path=None)["Grid spacing (nm)"] == 500


@pytest.fixture
def view_rot(window, qt_offscreen):
    window.view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(12.0 * PIXELSIZE)
    window.view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    window.window_rot.resize(256, 256)
    _drain(qt_offscreen, 0.2)
    return window.window_rot.view_rot


def test_3d_grid_planes_rotate_with_the_data(view_rot):
    view_rot.update_scene()
    plain = _rgb(view_rot.qimage)
    box = view_rot.window.display_settings_dlg.grid_box
    assert set(box.planes) == set(render.GRID_PLANES)
    _show_grid(box, 2 * PIXELSIZE)
    front = _rgb(view_rot.qimage)
    assert (front != plain).any()
    view_rot.apply_rotation(np.array([0.6, 0.4, 0.0]))
    view_rot.update_scene()
    assert (_rgb(view_rot.qimage) != front).any()
    # without planes there is nothing to draw
    for check in box.planes.values():
        check.setChecked(False)
    view_rot.update_scene()
    view_rot.window.display_settings_dlg.grid_box.setChecked(False)
    unrotated_without = _rgb(view_rot.qimage)
    box.setChecked(True)
    assert (_rgb(view_rot.qimage) == unrotated_without).all()


def test_3d_grid_keeps_z_0_nm_on_a_grid_line(view_rot):
    view_rot._pan_z = 0.7
    bounds, origin = view_rot.grid_geometry()
    # z = 0 nm in the frame of the rendered localizations
    z0 = 0.0 / PIXELSIZE - view_rot._z_shift - view_rot._pan_z
    assert origin[2] == pytest.approx(z0)
    lower, upper = view_rot._grid_bounds
    assert bounds[0][2] == pytest.approx(lower[2] - 0.7)
    assert bounds[1][2] == pytest.approx(upper[2] - 0.7)


def test_grid_goes_into_the_animation(view_rot, qapp, monkeypatch, tmp_path):
    received = []

    def record_build(path, locs, info, **kwargs):
        received.append(kwargs)
        return True

    monkeypatch.setattr(rotation.render, "build_animation", record_build)
    out = tmp_path / "anim.mp4"
    monkeypatch.setattr(
        lib, "get_save_filename_ext_dialog", lambda *a, **k: (str(out), ".mp4")
    )
    dialog = view_rot.window.animation_dialog
    dialog.add_position()
    view_rot.apply_rotation(np.array([0.0, np.pi / 2, 0.0]))
    dialog.add_position()
    dialog.build_animation()
    _wait_until(qapp, lambda: dialog._build_thread is None)
    assert "grid" not in received[-1]
    _show_grid(view_rot.window.display_settings_dlg.grid_box, 300)
    dialog.build_animation()
    _wait_until(qapp, lambda: dialog._build_thread is None)
    kwargs = received[-1]
    assert kwargs["grid"].spacing_nm == 300
    bounds, origin = view_rot.grid_geometry()
    np.testing.assert_allclose(kwargs["grid_bounds"], bounds)
    assert kwargs["grid_origin"] == origin


def test_animation_frames_show_the_grid(qapp, tmp_path):
    rng = np.random.default_rng(0)
    locs = _locs()
    locs["z"] = rng.uniform(-3.0, 3.0, size=len(locs))  # camera pixels
    positions = [
        (render.to_rotation((0, 0, 0)), ((24.0, 24.0), (40.0, 40.0))),
        (
            render.to_rotation((0, np.pi / 4, 0)),
            ((24.0, 24.0), (40.0, 40.0)),
        ),
    ]
    frames = {}
    for name, grid in (("plain", None), ("grid", render.GridOverlay(260))):
        path = str(tmp_path / f"{name}.mp4")
        assert render.build_animation(
            path,
            locs,
            _info(),
            positions=positions,
            durations=[1.0],
            disp_px_size=PIXELSIZE / 4,
            image_size=(64, 64),
            fps=2,
            grid=grid,
        )
        frames[name] = np.asarray(imageio.mimread(path), dtype=int)
        settings = io.load_info(path.replace(".mp4", ".yaml"))[0]
        assert settings["Grid spacing (nm)"] == (grid and 260.0)
    assert np.abs(frames["grid"] - frames["plain"]).max() > 50


def test_grid_settings_are_saved_on_close_and_loaded_on_start(
    qt_offscreen, tmp_path
):
    window = gui_render.Window(plugins_loaded=True)
    box_2d = window.display_settings_dlg.grid_box
    box_3d = window.window_rot.display_settings_dlg.grid_box
    _show_grid(box_2d, 750)
    box_2d.blend.setCurrentText("Over localizations")
    box_2d.labels.setChecked(False)
    box_2d.style_widget.set_value("font_size", 16)
    box_2d.style_widget.color.set_value("#ABCDEF")
    box_3d.planes["xz"].setChecked(False)
    box_3d.position.setCurrentText("Through the center")
    box_3d.style_widget.set_value("line_width", 2.5)
    window.close()

    saved = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    grid = saved["Render"]["Grid"]
    assert grid["Spacing (nm)"] == 750
    assert grid["Automatic spacing"] is False
    assert grid["Draw"] == "Over localizations"
    assert grid["Labels"] is False
    assert grid["Label size (px)"] == 16
    assert grid["Color"] == "#ABCDEF"
    assert "Planes" not in grid
    grid_3d = saved["Render"]["Grid3D"]
    assert grid_3d["Planes"] == ["xy", "yz"]
    assert grid_3d["Position"] == "Through the center"

    window = gui_render.Window(plugins_loaded=True)
    box_2d = window.display_settings_dlg.grid_box
    box_3d = window.window_rot.display_settings_dlg.grid_box
    assert not box_2d.isChecked()  # shown or not is not saved
    box_2d.setChecked(True)
    box_3d.setChecked(True)
    grid = box_2d.grid()
    assert grid.spacing_nm == 750
    assert grid.blend == "Over localizations"
    assert not grid.labels
    assert grid.style.font_size == 16
    assert grid.style.color == QtGui.QColor("#ABCDEF")
    grid_3d = box_3d.grid()
    assert grid_3d.planes == ("xy", "yz")
    assert grid_3d.position == "center"
    assert grid_3d.style.line_width == 2.5
    assert grid_3d.style.color is None  # "Auto" follows the background
    window.view.stop_render_worker()


def test_invalid_saved_grid_settings_keep_the_defaults(qt_offscreen):
    io.save_user_settings(
        {
            "Render": {
                "Grid": {"Spacing (nm)": "wide", "Draw": "Sideways"},
                "Grid3D": "nonsense",
            }
        }
    )
    window = gui_render.Window(plugins_loaded=True)
    box = window.display_settings_dlg.grid_box
    box.setChecked(True)
    grid = box.grid()
    assert grid.blend == "Behind localizations"
    assert grid.spacing_nm > 0
    window.view.stop_render_worker()


def test_grid_is_mirrored_to_linked_windows(qt_offscreen, tmp_path):
    from tests.test_gui_render_link import _close, _load

    primary = gui_render.Window(plugins_loaded=True)
    _load(primary, tmp_path, "a", (128, 128), seed=0)
    group = render_link.LinkGroup(primary)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    _load(secondary, tmp_path, "b", (200, 100), seed=1)
    try:
        assert "grid" in render_link.CATEGORY_KEYS
        src = primary.display_settings_dlg.grid_box
        dst = secondary.display_settings_dlg.grid_box
        src.setChecked(True)
        assert not dst.isChecked()  # not linked by default
        group.set_enabled("grid", True)
        src.setChecked(False)
        src.setChecked(True)
        assert dst.isChecked()
        src.spacing.setValue(640)
        src.style_widget.set_value("opacity", 70)
        assert dst.spacing.value() == 640
        assert not dst.auto_spacing.isChecked()
        assert dst.grid().style.opacity == pytest.approx(0.7)
    finally:
        for window in (primary, secondary):
            _close(window)
