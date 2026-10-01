"""Long Render operations run as cancelable tasks (``lib.run_task``).

Each operation computes on a worker thread and applies its result on the
GUI thread; canceling must leave the data and the disk untouched.

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
from PyQt6 import QtWidgets

from picasso import lib_qt
from picasso.gui import render as gui_render


WIDTH = HEIGHT = 64.0
PIXELSIZE = 130.0
N_FRAMES = 2000
N_CLUSTERS = 20
LOCS_PER_CLUSTER = 150


def _cluster_centers(seed: int = 0) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(5.0, WIDTH - 5.0, size=(N_CLUSTERS, 2))
    return [tuple(c) for c in centers]


def _locs(seed: int = 0) -> pd.DataFrame:
    """Dense clusters, sampled over the whole acquisition."""
    rng = np.random.default_rng(seed + 1)
    xy = np.repeat(np.array(_cluster_centers(seed)), LOCS_PER_CLUSTER, axis=0)
    xy += rng.normal(0.0, 0.05, size=xy.shape)
    n = len(xy)
    locs = pd.DataFrame(
        {
            "frame": rng.integers(0, N_FRAMES, size=n).astype(np.uint32),
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
            "photons": rng.uniform(500.0, 5000.0, size=n).astype(np.float32),
            "sx": np.full(n, 1.0, dtype=np.float32),
            "sy": np.full(n, 1.0, dtype=np.float32),
            "bg": np.full(n, 10.0, dtype=np.float32),
            "lpx": np.full(n, 0.05, dtype=np.float32),
            "lpy": np.full(n, 0.05, dtype=np.float32),
            "ellipticity": np.zeros(n, dtype=np.float32),
            "net_gradient": np.full(n, 5000.0, dtype=np.float32),
        }
    )
    return locs.sort_values("frame", kind="stable").reset_index(drop=True)


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": N_FRAMES,
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
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


def _dbscan_params() -> dict:
    return dict(
        radius=30.0,  # nm
        min_density=4,
        min_locs=10,
        save_centers=True,
        save_areas=True,
    )


def test_dbscan_saves_results(window, qt_offscreen, tmp_path):
    out = str(tmp_path / "out_dbscan.hdf5")
    window.view._dbscan(0, out, **_dbscan_params())
    _wait_for_tasks(qt_offscreen)
    assert os.path.exists(out)
    assert os.path.exists(str(tmp_path / "out_dbscan_centers.hdf5"))
    # next to the clustered locs, not to the centers
    assert os.path.exists(str(tmp_path / "out_dbscan_areas.csv"))


def test_smlm_clusterer_saves_results(window, qt_offscreen, tmp_path):
    """The clusterer takes the task's progress object itself."""
    out = str(tmp_path / "out_smlm.hdf5")
    window.view._smlm_clusterer(
        0,
        out,
        radius_xy=0.3,  # camera pixels
        radius_z=0.3,
        min_locs=10,
        frame_analysis=False,
        save_centers=True,
    )
    _wait_for_tasks(qt_offscreen)
    centers = pd.read_hdf(str(tmp_path / "out_smlm_centers.hdf5"), "locs")
    assert len(centers) == N_CLUSTERS


def test_canceled_dbscan_saves_nothing(
    window, qt_offscreen, tmp_path, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    dbscan = gui_render.clusterer.dbscan

    def slow_dbscan(*args, **kwargs):
        started.set()
        release.wait(10)
        return dbscan(*args, **kwargs)

    monkeypatch.setattr(gui_render.clusterer, "dbscan", slow_dbscan)
    out = str(tmp_path / "out_dbscan.hdf5")
    window.view._dbscan(0, out, **_dbscan_params())
    assert started.wait(10)
    task = lib_qt._running_tasks[0]
    task.dialog.canceled.emit()  # the Cancel button
    release.set()
    _wait_for_tasks(qt_offscreen)
    assert task.outcome == "canceled"
    assert not os.path.exists(out)


def test_nena_label_and_plot(window, qt_offscreen):
    info_dialog = window.info_dialog
    info_dialog.nena_result = None
    info_dialog.show_nena_plot()  # calculates first, then plots
    _wait_for_tasks(qt_offscreen)
    assert info_dialog.nena_label.text().endswith("nm")
    assert info_dialog.nena_window is not None
    assert "NeNA (nm)" in window.view.infos[0][-1]


def test_nena_button(window, qt_offscreen):
    """The button's checked state must not be taken for ``on_done``."""
    info_dialog = window.info_dialog
    info_dialog.nena_button.click()
    _wait_for_tasks(qt_offscreen)
    assert info_dialog.nena_label.text().endswith("nm")


def test_rcc_applies_drift(window, qt_offscreen, monkeypatch):
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getInt", lambda *a, **k: (500, True)
    )
    window.view.undrift_rcc()
    _wait_for_tasks(qt_offscreen)
    assert window.view._drift[0] is not None
    assert len(window.view._drift[0]) == N_FRAMES


def test_aim_applies_drift(window, qt_offscreen, monkeypatch):
    params = {"segmentation": 500, "intersect_d": 20.0, "roi_r": 60.0}
    monkeypatch.setattr(
        gui_render.AIMDialog, "getParams", lambda *a, **k: (params, True)
    )
    window.view.undrift_aim()
    _wait_for_tasks(qt_offscreen)
    assert window.view._drift[0] is not None
    assert any("AIM" in str(entry) for entry in window.view.infos[0])


def test_canceled_aim_leaves_locs_untouched(window, qt_offscreen, monkeypatch):
    params = {"segmentation": 100, "intersect_d": 20.0, "roi_r": 60.0}
    monkeypatch.setattr(
        gui_render.AIMDialog, "getParams", lambda *a, **k: (params, True)
    )
    before = window.view.locs[0]
    n_infos = len(window.view.infos[0])
    window.view.undrift_aim()
    lib_qt._running_tasks[0].cancel()
    _wait_for_tasks(qt_offscreen)
    assert window.view.locs[0] is before
    assert len(window.view.infos[0]) == n_infos
    assert window.view._drift[0] is None


def _pick_clusters(view) -> None:
    view._pick_shape = "Circle"
    # one camera pixel in diameter
    view.window.tools_settings_dialog.pick_diameter.setValue(PIXELSIZE)
    view._picks = []
    view.add_picks(_cluster_centers())


def test_pick_info_long(window, qt_offscreen):
    _pick_clusters(window.view)
    window.view.update_pick_info_long()
    _wait_for_tasks(qt_offscreen)
    n_mean = float(window.info_dialog.n_localizations_mean.text())
    assert n_mean == pytest.approx(LOCS_PER_CLUSTER, rel=0.05)


def test_save_pick_properties(window, qt_offscreen, tmp_path):
    _pick_clusters(window.view)
    out = str(tmp_path / "props.hdf5")
    window.view.save_pick_properties(out, 0)
    _wait_for_tasks(qt_offscreen)
    props = pd.read_hdf(out, key="groups")
    assert len(props) == N_CLUSTERS


def test_undrift_from_picked_builds_index_off_thread(window, qt_offscreen):
    _pick_clusters(window.view)
    window.view.render_index[0] = None  # as after any edit of the locs
    n_infos = len(window.view.infos[0])
    window.view.undrift_from_picked()
    _wait_for_tasks(qt_offscreen)
    assert window.view._drift[0] is not None
    assert len(window.view.infos[0]) == n_infos + 1
    assert "Undrift from picked" in window.view.infos[0][-1]["Generated by"]


def test_canceled_undrift_from_picked_keeps_index_and_locs(
    window, qt_offscreen, monkeypatch
):
    _pick_clusters(window.view)
    window.view.render_index[0] = None
    started, release = threading.Event(), threading.Event()

    def blocked(*args, progress=None, **kwargs):
        started.set()
        release.wait(10)
        progress.set_value(0)  # the first pick: a cancellation point
        raise AssertionError("not canceled")

    monkeypatch.setattr(
        gui_render.postprocess, "undrift_from_fiducials", blocked
    )
    before = window.view.locs[0]
    window.view.undrift_from_picked()
    assert started.wait(10)
    task = lib_qt._running_tasks[0]
    task.dialog.canceled.emit()
    release.set()
    _wait_for_tasks(qt_offscreen)
    assert task.outcome == "canceled"
    assert window.view.locs[0] is before
    assert window.view._drift[0] is None
    # the index built for the unchanged locs is kept for the next run
    assert window.view.render_index[0] is not None
