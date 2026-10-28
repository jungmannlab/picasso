"""The scale bar of Render's 2D and 3D windows.

Display Settings set the scale bar's length and its appearance: where
it is drawn, its thickness, color and background, and the unit,
placement and font of the printed length. These tests cover the
settings reaching the drawing, the automatic length, the settings
grayed out when unused, exports, the animation, the user settings file
and linked windows.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PyQt6 import QtGui, QtWidgets

from picasso import io, render
from picasso.gui import render as gui_render
from picasso.gui import render_link, rotation, theme
from picasso.gui.scalebar_settings import ScaleBarGroupBox

from tests.test_gui_animation_export import dialog  # noqa: F401
from tests.test_gui_rotation_async import (
    HEIGHT,
    PIXELSIZE,
    WIDTH,
    _info,
    _locs,
    _wait_until,
)


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
    window.view.async_rendering = False
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()


def _record_draws(monkeypatch) -> list[dict]:
    """Record the arguments of every ``render.draw_scalebar`` call."""
    calls = []
    original = render.draw_scalebar

    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(render, "draw_scalebar", record)
    return calls


def _image() -> QtGui.QImage:
    image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("black"))
    return image


def test_appearance_reaches_the_drawing(window, monkeypatch):
    calls = _record_draws(monkeypatch)
    box = window.display_settings_dlg.scalebar_groupbox
    box.setChecked(True)
    box.print_length.setChecked(True)
    box.position.setCurrentText("Top left")
    box.margin_x.setValue(12)
    box.margin_y.setValue(8)
    box.thickness.setValue(3)
    box.color.set_value("Cyan")
    box.background.setValue(50)
    box.units.setCurrentText("µm")
    box.label_position.setCurrentText("Below the bar")
    box.font_size.setValue(15)
    box.bold.setChecked(True)
    box.italic.setChecked(True)
    calls.clear()
    with theme.unscaled_text():
        window.view.draw_scalebar(_image())
    (kwargs,) = calls
    assert kwargs["display_length"] is True
    assert kwargs["position"] == "top left"
    assert kwargs["margin"] == (12, 8)
    assert kwargs["display_height"] == 3
    assert kwargs["color"] == QtGui.QColor("#00FFFF")
    assert kwargs["background_opacity"] == pytest.approx(0.5)
    assert kwargs["units"] == "um"
    assert kwargs["label_position"] == "below"
    assert kwargs["text_fontsize"] == 15
    assert kwargs["bold"] is True
    assert kwargs["italic"] is True
    assert kwargs["font_family"] is None  # the windows' font


def test_automatic_color_follows_the_background(window, monkeypatch):
    calls = _record_draws(monkeypatch)
    box = window.display_settings_dlg.scalebar_groupbox
    box.setChecked(True)
    window.view.draw_scalebar(_image())
    assert calls[-1]["color"] == QtGui.QColor("white")
    window.dataset_dialog.wbackground.setChecked(True)
    window.view.draw_scalebar(_image())
    assert calls[-1]["color"] == QtGui.QColor("black")


def test_hidden_scale_bar_is_not_drawn_unless_forced(window, monkeypatch):
    calls = _record_draws(monkeypatch)
    window.view.draw_scalebar(_image())
    assert calls == []
    window.view.draw_scalebar(_image(), force=True)
    assert len(calls) == 1


def test_automatic_length_follows_the_zoom_until_edited(window):
    view = window.view
    box = window.display_settings_dlg.scalebar_groupbox
    box.setChecked(True)
    view.update_scene()
    view.set_optimal_scalebar()
    expected = render.optimal_scalebar_length(WIDTH * PIXELSIZE)
    assert box.length.value() == expected
    box.length.setValue(1234)  # typed by the user
    assert not box.auto_length.isChecked()
    box.auto_length.setChecked(True)  # adjusts at once
    assert box.length.value() == expected


def test_unused_settings_are_grayed_out(window):
    box = window.display_settings_dlg.scalebar_groupbox
    window.display_settings_dlg.show()
    box.setChecked(True)
    text_widgets = [
        box.units,
        box.label_position,
        box.font_family,
        box.font_size,
        box.bold,
        box.italic,
    ]
    box.print_length.setChecked(False)
    assert not any(w.isEnabled() for w in text_widgets)
    box.print_length.setChecked(True)
    assert all(w.isEnabled() for w in text_widgets)
    assert box.margin_x.isEnabled()
    box.position.setCurrentText("Bottom center")
    assert not box.margin_x.isEnabled()  # centered horizontally
    assert box.margin_y.isEnabled()


def test_reset_keeps_the_length(qt_offscreen, monkeypatch):
    monkeypatch.setattr(
        gui_render.lib, "confirm_restore_defaults", lambda *a, **k: True
    )
    box = ScaleBarGroupBox()
    box.auto_length.setChecked(False)
    box.length.setValue(321)
    box.position.setCurrentText("Top center")
    box.thickness.setValue(2)
    box.bold.setChecked(True)
    box.italic.setChecked(True)
    box.reset_appearance()
    assert box.length.value() == 321
    assert box.position.currentText() == "Bottom right"
    assert box.thickness.value() == 10
    assert not box.bold.isChecked()
    assert not box.italic.isChecked()


def test_export_with_a_hidden_scale_bar_saves_a_copy_with_it(
    window, tmp_path, monkeypatch
):
    """The extra ``_scalebar`` image is drawn as set up, and the view
    and the exported image stay without the scale bar."""
    box = window.display_settings_dlg.scalebar_groupbox
    box.position.setCurrentText("Top left")
    box.thickness.setValue(6)
    window.view.update_scene()
    path = tmp_path / "view.png"
    monkeypatch.setattr(
        gui_render.lib,
        "get_save_filename_ext_dialog",
        lambda *a, **k: (str(path), ".png"),
    )
    window.export_current()
    assert not box.isChecked()
    plain = QtGui.QImage(str(path))
    with_bar = QtGui.QImage(str(tmp_path / "view_scalebar.png"))
    # the top left corner holds the bar only in the copy
    corner = (35 + 2, 20 + 2)
    assert plain.pixelColor(*corner) != with_bar.pixelColor(*corner)
    assert with_bar.pixelColor(*corner) == QtGui.QColor("white")


def test_settings_are_saved_and_restored(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    box_2d = window.display_settings_dlg.scalebar_groupbox
    box_3d = window.window_rot.display_settings_dlg.scalebar_groupbox
    box_2d.length.setValue(750)  # ends the automatic length
    box_2d.print_length.setChecked(True)
    box_2d.position.setCurrentText("Top center")
    box_2d.margin_y.setValue(5)
    box_2d.color.set_value("#ABCDEF")
    box_2d.units.setCurrentText("Automatic")
    box_2d.bold.setChecked(True)
    box_2d.italic.setChecked(True)
    box_3d.thickness.setValue(4)
    window.close()

    saved = yaml.safe_load((tmp_path / "settings.yaml").read_text())
    scalebar = saved["Render"]["ScaleBar"]
    # each session starts with the automatic length
    assert "Length (nm)" not in scalebar
    assert "Automatic length" not in scalebar
    assert scalebar["Position"] == "Top center"
    assert scalebar["Margins (px)"] == [35, 5]
    assert scalebar["Color"] == "#ABCDEF"
    assert saved["Render"]["ScaleBar3D"]["Thickness (px)"] == 4

    window = gui_render.Window(plugins_loaded=True)
    box_2d = window.display_settings_dlg.scalebar_groupbox
    box_3d = window.window_rot.display_settings_dlg.scalebar_groupbox
    assert not box_2d.isChecked()  # shown or not is not saved
    kwargs = box_2d.style_kwargs()
    assert box_2d.length.value() == 500
    assert box_2d.auto_length.isChecked()
    assert box_2d.print_length.isChecked()
    assert kwargs["position"] == "top center"
    assert kwargs["margin"] == (35, 5)
    assert kwargs["color"] == QtGui.QColor("#ABCDEF")
    assert kwargs["units"] == "auto"
    assert kwargs["bold"] is True
    assert kwargs["italic"] is True
    assert not box_3d.italic.isChecked()
    assert box_3d.thickness.value() == 4
    assert box_3d.style_kwargs()["position"] == "bottom right"
    window.view.stop_render_worker()


def test_a_saved_length_is_ignored(qt_offscreen):
    io.save_user_settings(
        {
            "Render": {
                "ScaleBar": {
                    "Length (nm)": 750,
                    "Automatic length": False,
                    "Position": "Top left",
                },
            }
        }
    )
    window = gui_render.Window(plugins_loaded=True)
    box = window.display_settings_dlg.scalebar_groupbox
    assert box.length.value() == 500
    assert box.auto_length.isChecked()
    assert box.position.currentText() == "Top left"
    window.view.stop_render_worker()


def test_invalid_saved_settings_keep_the_defaults(qt_offscreen):
    io.save_user_settings(
        {
            "Render": {
                "ScaleBar": {
                    "Length (nm)": "long",
                    "Position": "Middle",
                    "Margins (px)": [1],
                    "Font": "No such font",
                    "Color": "not a color",
                    "Bold": "yes",
                },
                "ScaleBar3D": "nonsense",
            }
        }
    )
    window = gui_render.Window(plugins_loaded=True)
    box = window.display_settings_dlg.scalebar_groupbox
    kwargs = box.style_kwargs()
    assert box.length.value() == 500
    assert kwargs["position"] == "bottom right"
    assert kwargs["margin"] == (35, 20)
    assert kwargs["font_family"] is None
    assert kwargs["color"] == QtGui.QColor("white")
    assert kwargs["bold"] is False
    window.view.stop_render_worker()


def test_scale_bar_is_mirrored_to_linked_windows(qt_offscreen, tmp_path):
    from tests.test_gui_render_link import _close, _load

    primary = gui_render.Window(plugins_loaded=True)
    _load(primary, tmp_path, "a", (128, 128), seed=0)
    group = render_link.LinkGroup(primary)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    _load(secondary, tmp_path, "b", (200, 100), seed=1)
    try:
        src = primary.display_settings_dlg.scalebar_groupbox
        dst = secondary.display_settings_dlg.scalebar_groupbox
        src.setChecked(True)
        assert not dst.isChecked()  # not linked by default
        group.set_enabled("scalebar", True)
        src.setChecked(False)
        src.setChecked(True)
        assert dst.isChecked()
        src.position.setCurrentText("Top right")
        src.background.setValue(40)
        assert dst.position.currentText() == "Top right"
        assert dst.background.value() == 40
        # an automatic length follows each window's own zoom
        assert dst.auto_length.isChecked()
        assert dst.length.value() == secondary.view.optimal_scalebar_length()
        src.length.setValue(640)
        assert dst.length.value() == 640
        assert not dst.auto_length.isChecked()
    finally:
        for window in (primary, secondary):
            _close(window)


def test_animation_gets_the_appearance(
    dialog, qapp, monkeypatch  # noqa: F811
):
    """The 3D window's scale bar style reaches the animation, with the
    text size of exported images whatever the windows' font size."""
    received = []

    def record_build(path, locs, info, **kwargs):
        received.append(kwargs)
        return True

    monkeypatch.setattr(rotation.render, "build_animation", record_build)
    monkeypatch.setitem(
        theme._state,
        "applied",
        SimpleNamespace(mode="Light", font_scale=200),
    )
    box = dialog.window.display_settings_dlg.scalebar_groupbox
    box.setChecked(True)
    box.position.setCurrentText("Top left")
    box.font_size.setValue(18)
    dialog.build_animation()
    _wait_until(qapp, lambda: dialog._build_thread is None)
    style = received[-1]["scalebar_style"]
    assert style["position"] == "top left"
    assert style["text_fontsize"] == 18
    assert box.style_kwargs()["text_fontsize"] == 36  # on screen


def test_animation_frames_follow_the_style(qt_offscreen, tmp_path):
    """``build_animation`` draws the scale bar where ``scalebar_style``
    puts it."""
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation

    locs = _locs()
    locs["z"] = 0.0
    viewport = ((0.0, 0.0), (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE))

    def last_frame(name, **kwargs):
        out = tmp_path / name
        render.build_animation(
            str(out),
            locs,
            [_info()],
            positions=[(Rotation.identity(), viewport)] * 2,
            durations=[1.0],
            disp_px_size=PIXELSIZE,
            image_size=(96, 64),
            fps=2,
            **kwargs,
        )
        return np.asarray(imageio.mimread(str(out))[-1]).astype(int)

    plain = last_frame("plain.mp4")
    frame = last_frame(
        "bar.mp4",
        scalebar_length_nm=int(5 * PIXELSIZE),
        scalebar_display_length=False,
        scalebar_style={"position": "top left", "margin": (10, 8)},
    )
    assert frame[8 + 3, 10 + 3].min() > 200  # inside the bar
    # the frames differ only around the bar (video compression blurs)
    rows, columns = np.nonzero(np.abs(frame - plain).max(axis=2) > 60)
    assert len(rows)
    assert rows.max() < 8 + 10 + 4 and columns.min() >= 10 - 4
