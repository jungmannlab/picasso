"""Single-particle tracking in Picasso: Render (``picasso.gui.render_spt``):
tracking channels in place, analyzing them, drawing their tracks, and
the Tracks settings.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtGui, QtWidgets

from picasso import lib_qt, spt
from picasso.gui import render as gui_render
from picasso.gui import render_spt

WIDTH = HEIGHT = 64.0
PIXELSIZE = 100.0
N_PARTICLES = 30
N_FRAMES = 20
FRAME_TIME = 20.0


def _locs(seed: int = 0) -> pd.DataFrame:
    """Sparse, freely diffusing particles (D = 0.1 um^2/s)."""
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * 0.1 * FRAME_TIME / 1000) * 1000  # nm
    frames, xs, ys = [], [], []
    starts = rng.uniform(8, WIDTH - 8, (N_PARTICLES, 2)) * PIXELSIZE
    for p in range(N_PARTICLES):
        xy = starts[p] + np.cumsum(rng.normal(0, step, (N_FRAMES, 2)), 0)
        first = rng.integers(0, 50)
        frames.append(np.arange(first, first + N_FRAMES))
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
    n = N_PARTICLES * N_FRAMES
    locs = pd.DataFrame(
        {
            "frame": np.concatenate(frames).astype(np.uint32),
            "x": np.clip(
                np.concatenate(xs), 0, (WIDTH - 1) * PIXELSIZE
            ).astype(np.float32),
            "y": np.clip(
                np.concatenate(ys), 0, (HEIGHT - 1) * PIXELSIZE
            ).astype(np.float32),
            "photons": np.full(n, 1000.0, dtype=np.float32),
            "sx": np.full(n, PIXELSIZE, dtype=np.float32),
            "sy": np.full(n, PIXELSIZE, dtype=np.float32),
            "bg": np.full(n, 10.0, dtype=np.float32),
            "lpx": np.full(n, 0.2 * PIXELSIZE, dtype=np.float32),
            "lpy": np.full(n, 0.2 * PIXELSIZE, dtype=np.float32),
        }
    )
    return locs.sort_values("frame", kind="stable").reset_index(drop=True)


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 100,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _wait_for_tasks(qapp, timeout: float = 60.0) -> None:
    start = time.monotonic()
    while lib_qt._running_tasks:
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not end")
    qapp.processEvents()


@pytest.fixture
def window(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"), _locs(), _info(), render_=False
    )
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


TRACK_PARAMS = {
    "max_jump": 400.0,  # nm
    "max_gap": 1,
    "min_length": 2,
    "frame_time": FRAME_TIME,
}


@pytest.fixture
def track_params(monkeypatch):
    monkeypatch.setattr(
        render_spt.TrackDialog,
        "getParams",
        lambda *a, **k: (dict(TRACK_PARAMS), True),
    )


@pytest.fixture
def messages(monkeypatch) -> list[str]:
    """Texts of the information boxes shown."""
    texts = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda parent, title, text, *a, **k: texts.append(text),
    )
    return texts


def _track(window, qapp) -> None:
    window.view.spt.track()
    _wait_for_tasks(qapp)


def test_track_in_place(window, qt_offscreen, track_params):
    view = window.view
    box = window.display_settings_dlg.tracks_box
    assert not box.isChecked()
    _track(window, qt_offscreen)
    locs, info = view.locs[0], view.infos[0]
    assert spt.is_tracked(locs, info)
    assert locs["group"].nunique() == N_PARTICLES
    entry = info[-1]
    # the metadata as spt.link_tracks returns it
    assert entry["Step"] == "SPT"
    assert entry["Max. jump (nm)"] == 400.0
    assert entry[spt.FRAME_TIME_KEY] == FRAME_TIME
    assert len(view.group_color) == len(locs)
    assert box.isChecked()  # the tracks are shown


def test_tracked_and_linked_channels_are_skipped(
    window, qt_offscreen, track_params, messages, tmp_path, monkeypatch
):
    view = window.view
    _track(window, qt_offscreen)
    linked = _locs(1).assign(len=np.int32(1))
    view.add(str(tmp_path / "linked.hdf5"), linked, _info(), render_=False)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *a, **k: ("Apply to all sequentially", True),
    )
    before = [view.locs[0], view.locs[1]]
    view.spt.track()
    _wait_for_tasks(qt_offscreen)
    assert view.locs[0] is before[0] and view.locs[1] is before[1]
    assert len(messages) == 1
    assert "linked.hdf5" in messages[0] and "locs.hdf5" in messages[0]


def test_track_all_channels(
    window, qt_offscreen, track_params, tmp_path, monkeypatch
):
    view = window.view
    view.add(str(tmp_path / "second.hdf5"), _locs(1), _info(), render_=False)
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *a, **k: ("Apply to all sequentially", True),
    )
    _track(window, qt_offscreen)
    assert all(spt.is_tracked(view.locs[c], view.infos[c]) for c in (0, 1))


def test_no_tracks_leaves_the_channel(
    window, qt_offscreen, messages, monkeypatch
):
    params = dict(TRACK_PARAMS, min_length=1000)
    monkeypatch.setattr(
        render_spt.TrackDialog, "getParams", lambda *a, **k: (params, True)
    )
    before = window.view.locs[0]
    _track(window, qt_offscreen)
    assert window.view.locs[0] is before
    assert "No tracks found" in messages[0]


def test_canceled_tracking_leaves_the_channel(
    window, qt_offscreen, track_params, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    link_tracks = spt.link_tracks

    def slow_link_tracks(*args, **kwargs):
        started.set()
        release.wait(10)
        return link_tracks(*args, **kwargs)

    monkeypatch.setattr(spt, "link_tracks", slow_link_tracks)
    before = window.view.locs[0]
    n_infos = len(window.view.infos[0])
    window.view.spt.track()
    assert started.wait(10)
    task = lib_qt._running_tasks[0]
    task.dialog.canceled.emit()  # the Cancel button
    release.set()
    _wait_for_tasks(qt_offscreen)
    assert task.outcome == "canceled"
    assert window.view.locs[0] is before
    assert len(window.view.infos[0]) == n_infos


def test_analysis_window_and_saving(
    window, qt_offscreen, track_params, monkeypatch, tmp_path, messages
):
    _track(window, qt_offscreen)
    analysis = {
        "frame_time": FRAME_TIME,
        "max_lag": 6,
        "n_fit": 3,
        "sigma": "precision",
        "truncate": False,
    }
    monkeypatch.setattr(
        render_spt.TrackAnalysisDialog,
        "getParams",
        lambda *a, **k: (dict(analysis), True),
    )
    window.view.spt.analyze()
    _wait_for_tasks(qt_offscreen)
    analysis_window = window.view.spt.analysis_windows[-1]
    assert analysis_window.tabs.count() == 3
    assert analysis_window.result["n_tracks"] == N_PARTICLES
    assert len(analysis_window.result["msd"]["lag"]) == 6
    assert "tracks" in analysis_window.summary.text()
    # switching the number of populations redraws the fit
    index = analysis_window.populations.findData(2)
    analysis_window.populations.setCurrentIndex(index)
    assert analysis_window.jd_fit()["n_populations"] == 2
    assert analysis_window.summary.text().count("%)") == 2

    out = str(tmp_path / "result_track_stats.csv")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (out, "*.csv"),
    )
    analysis_window.save()
    table = pd.read_csv(out)
    assert len(table) == N_PARTICLES
    assert os.path.exists(str(tmp_path / "result_msd.csv"))
    assert os.path.exists(str(tmp_path / "result_diffusion.yaml"))
    assert "Saved" in messages[-1]


def test_analysis_needs_tracks(window, qt_offscreen, messages, monkeypatch):
    opened = []
    monkeypatch.setattr(
        render_spt.TrackAnalysisDialog,
        "getParams",
        lambda *a, **k: opened.append(1) or ({}, False),
    )
    window.view.spt.analyze()
    assert not opened
    assert "not tracked" in messages[0]


def test_analysis_dialog_shows_only_the_used_error(qt_offscreen):
    dialog = render_spt.TrackAnalysisDialog(
        None, frame_time=15.0, precision=21.3, max_jump=None
    )
    assert dialog.frame_time.value() == 15.0
    assert not dialog.truncate.isEnabled()
    assert dialog.sigma.isHidden()
    assert dialog.params()["sigma"] == 0.0
    dialog.sigma_mode.setCurrentIndex(
        dialog.sigma_mode.findData(render_spt.SIGMA_CUSTOM)
    )
    assert not dialog.sigma.isHidden()
    dialog.sigma.setValue(12.0)
    assert dialog.params()["sigma"] == 12.0
    dialog.sigma_mode.setCurrentIndex(
        dialog.sigma_mode.findData(render_spt.SIGMA_PRECISION)
    )
    assert "21.3 nm" in dialog.sigma_mode.currentText()
    assert dialog.params()["sigma"] == "precision"
    # no precision column: no such option
    dialog = render_spt.TrackAnalysisDialog(None, precision=None)
    assert dialog.sigma_mode.findData(render_spt.SIGMA_PRECISION) < 0


def test_tracks_are_drawn(window, qt_offscreen, track_params):
    view = window.view
    _track(window, qt_offscreen)
    box = window.display_settings_dlg.tracks_box

    def drawn() -> int:
        image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor(0, 0, 0))
        image = view.spt.draw_tracks(image)
        bits = image.constBits()
        bits.setsize(image.sizeInBytes())
        rgb = np.frombuffer(bits, np.uint8).reshape(256, 256, 4)[:, :, :3]
        return int(rgb.any(axis=2).sum())

    assert drawn() > 100
    assert str(N_PARTICLES) in box.status.text()
    box.setChecked(False)
    assert drawn() == 0
    box.setChecked(True)
    # the segments are cached until the localizations change
    assert 0 in view.spt._segments
    view.invalidate_locs_index(0)
    assert 0 not in view.spt._segments
    assert drawn() > 100
    view.release_idle_memory()
    assert not view.spt._segments


def test_untracked_channels_draw_nothing(window, qt_offscreen):
    view = window.view
    window.display_settings_dlg.tracks_box.setChecked(True)
    image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(0, 0, 0))
    assert view.spt.draw_tracks(image) is image


def test_tracks_settings_round_trip(qt_offscreen):
    box = render_spt.TracksGroupBox()
    box.max_tracks.setValue(500)
    box.style_widget.set_value("color", "Green")
    box.style_widget.set_value("line_width", 2.0)
    settings = box.settings()
    other = render_spt.TracksGroupBox()
    other.load_settings(settings)
    assert other.max_tracks.value() == 500
    assert other.style().line_width == 2.0
    assert other.style().color is not None
    assert render_spt.TracksGroupBox().style().color is None  # "Auto"


def test_track_dialog_remembers_parameters(qt_offscreen, monkeypatch):
    monkeypatch.setattr(
        render_spt.TrackDialog,
        "exec",
        lambda self: QtWidgets.QDialog.DialogCode.Accepted,
    )
    params, ok = render_spt.TrackDialog.getParams()
    assert ok and params["max_jump"] == render_spt.DEFAULT_MAX_JUMP
    dialog = render_spt.TrackDialog(None)
    dialog.max_gap.setValue(3)
    monkeypatch.setattr(
        render_spt.TrackDialog,
        "params",
        lambda self: dict(TRACK_PARAMS, max_gap=3),
    )
    render_spt.TrackDialog.getParams()
    assert render_spt.TrackDialog(None).max_gap.value() == 3
    # the frame time of the metadata wins over the remembered one
    assert render_spt.TrackDialog(None, frame_time=7.5).frame_time.value() == (
        7.5
    )
