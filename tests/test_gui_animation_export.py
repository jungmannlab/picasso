"""Building 3D animations in the background (``AnimationDialog``).

The frames are rendered and encoded on a worker thread at the
resolution set in the dialog, with a cancellable, non-modal progress
dialog; a cancelled build leaves no partial video behind.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import imageio.v2 as imageio
import numpy as np
import pytest
from PyQt6 import QtWidgets

from picasso import lib
from picasso.gui import render as gui_render
from picasso.gui import rotation

from tests.test_gui_rotation_async import (
    HEIGHT,
    PIXELSIZE,
    WIDTH,
    _drain,
    _info,
    _locs,
    _wait_until,
)


@pytest.fixture
def dialog(qt_offscreen, tmp_path, monkeypatch):
    """The animation dialog of a rotation window showing one pick, with
    two positions (a quarter turn apart) and the save dialog answered
    with a file in ``tmp_path``."""
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    # a failure report would block an offscreen test: record it instead
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"), _locs(), _info(), render_=False
    )
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(12.0 * PIXELSIZE)
    window.view._picks = [(32.0, 32.0)]
    window.open_3d_view()
    window.window_rot.resize(128, 128)
    _drain(qt_offscreen, 0.2)
    view = window.window_rot.view_rot
    dialog = window.window_rot.animation_dialog
    dialog.add_position()
    view.apply_rotation(np.array([0.0, np.pi / 2, 0.0]))
    dialog.add_position()
    assert len(dialog.positions) == 2
    dialog.rows[1]["duration"].setValue(1.0)
    dialog.fps.setValue(3)
    out = tmp_path / "anim.mp4"
    monkeypatch.setattr(
        lib, "get_save_filename_ext_dialog", lambda *a, **k: (str(out), ".mp4")
    )
    dialog.output = out
    dialog.warnings = warnings
    yield dialog
    dialog.stop_build()
    view.stop_render_worker()
    window.view.stop_render_worker()


def test_builds_in_the_background_at_the_chosen_resolution(
    dialog, qapp, monkeypatch
):
    dialog.width_px.setValue(96)
    dialog.height_px.setValue(64)
    dialog.build_animation()
    assert dialog._build_thread is not None  # running on a thread
    assert not dialog.build.isEnabled()
    assert (
        dialog._build_progress is not None
        and not dialog._build_progress.isModal()
    )
    _wait_until(qapp, lambda: dialog._build_thread is None, timeout=60.0)
    assert dialog.warnings == []
    assert dialog.build.isEnabled()
    assert dialog.output.exists()
    assert dialog.output.with_suffix(".yaml").exists()
    frames = imageio.mimread(str(dialog.output))
    assert len(frames) == 3  # fps 3 x 1 s
    assert frames[0].shape[:2] == (64, 96)  # height x width as chosen


def test_resolution_follows_the_window_until_edited(dialog, qapp):
    view = dialog.window.view_rot
    dialog.hide()
    dialog.show()
    assert (dialog.width_px.value(), dialog.height_px.value()) == (
        max(16, view.width()),
        max(16, view.height()),
    )
    dialog.width_px.setValue(640)
    dialog.hide()
    dialog.show()
    assert dialog.width_px.value() == 640  # edited by hand: kept


def test_cancel_stops_the_build_and_leaves_no_file(dialog, qapp, monkeypatch):
    started = []

    def slow_build(path, locs, info, *, progress_callback, cancel, **kwargs):
        started.append(path)
        for i in range(1000):
            if cancel():
                return False
            progress_callback(i)
            time.sleep(0.01)
        raise AssertionError("never cancelled")

    monkeypatch.setattr(rotation.render, "build_animation", slow_build)
    dialog.build_animation()
    _wait_until(qapp, lambda: started and dialog._build_progress.value() >= 2)
    # the user clicks Cancel: the button emits ``canceled`` (Qt's
    # programmatic ``cancel()`` resets the dialog without emitting it)
    dialog._build_progress.canceled.emit()
    _wait_until(qapp, lambda: dialog._build_thread is None)
    assert dialog.warnings == []
    assert dialog.build.isEnabled()
    assert dialog._build_progress is None
    assert not dialog.output.exists()


def test_failure_is_reported_not_lost(dialog, qapp, monkeypatch):
    def failing(*args, **kwargs):
        raise RuntimeError("no encoder")

    monkeypatch.setattr(rotation.render, "build_animation", failing)
    dialog.build_animation()
    _wait_until(qapp, lambda: dialog._build_thread is None)
    assert len(dialog.warnings) == 1 and "no encoder" in dialog.warnings[0][2]
    assert dialog.build.isEnabled()


def test_closing_the_window_stops_a_build(dialog, qapp, monkeypatch):
    def endless(path, locs, info, *, progress_callback, cancel, **kwargs):
        while not cancel():
            time.sleep(0.01)
        return False

    monkeypatch.setattr(rotation.render, "build_animation", endless)
    dialog.build_animation()
    assert dialog._build_thread is not None
    dialog.window.close()
    assert dialog._build_thread is None
    assert dialog.build.isEnabled()


def test_transition_choice_reaches_the_build(dialog, qapp, monkeypatch):
    """The dialog defaults to stopping at each position and passes the
    selected one on to ``render.build_animation``."""
    received = []

    def record_build(path, locs, info, *, transition, **kwargs):
        received.append(transition)
        return True

    monkeypatch.setattr(rotation.render, "build_animation", record_build)
    assert dialog.transition.currentText() == "Stop at each position"
    for name, expected in (
        ("Smooth", "smooth"),
        ("Stop at each position", "ease"),
        ("Constant speed", "linear"),
    ):
        dialog.transition.setCurrentText(name)
        dialog.build_animation()
        _wait_until(qapp, lambda: dialog._build_thread is None)
        assert received[-1] == expected
    assert dialog.warnings == []


def test_help_button_links_to_the_animation_docs(dialog):
    buttons = dialog.findChildren(lib.HelpButton)
    assert [b.help_url for b in buttons] == [dialog.DOCS_URL]
    assert dialog.DOCS_URL.endswith("render.html#build-an-animation")
