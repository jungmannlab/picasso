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
    """Cluster centers (nm), at least 5 camera pixels from the edges."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(5.0, WIDTH - 5.0, size=(N_CLUSTERS, 2))
    return [tuple(c * PIXELSIZE) for c in centers]


def _locs(seed: int = 0) -> pd.DataFrame:
    """Dense clusters, sampled over the whole acquisition."""
    rng = np.random.default_rng(seed + 1)
    xy = np.repeat(np.array(_cluster_centers(seed)), LOCS_PER_CLUSTER, axis=0)
    xy += rng.normal(0.0, 6.5, size=xy.shape)  # nm
    n = len(xy)
    locs = pd.DataFrame(
        {
            "frame": rng.integers(0, N_FRAMES, size=n).astype(np.uint32),
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
            "photons": rng.uniform(500.0, 5000.0, size=n).astype(np.float32),
            "sx": np.full(n, PIXELSIZE, dtype=np.float32),
            "sy": np.full(n, PIXELSIZE, dtype=np.float32),
            "bg": np.full(n, 10.0, dtype=np.float32),
            "lpx": np.full(n, 6.5, dtype=np.float32),
            "lpy": np.full(n, 6.5, dtype=np.float32),
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
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
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
        radius_xy=39.0,  # nm
        radius_z=39.0,
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


def test_rcc_suggests_four_segments(window, monkeypatch):
    """RCC needs at least 4 segments; 1000 frames, suggested for movies
    of 1000 frames and more, gave 2 of the 2000 frames here."""
    asked = []

    def get_int(parent, title, label, value, minimum, maximum, *a, **k):
        asked.append((value, minimum, maximum))
        return value, False  # canceled

    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", get_int)
    window.view.undrift_rcc()
    assert asked == [(N_FRAMES // 4, 1, N_FRAMES)]


def test_rcc_explains_too_few_segments(window, qt_offscreen, monkeypatch):
    messages = []
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getInt", lambda *a, **k: (1000, True)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda parent, title, text, *a, **k: messages.append(text),
    )
    window.view.undrift_rcc()
    _wait_for_tasks(qt_offscreen)
    (message,) = messages
    assert "at most 500 frames" in message
    assert window.view._drift[0] is None


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


def _link_all_channels(view, qapp, monkeypatch) -> list[str]:
    """Choose *Apply to all sequentially* and the default linking
    parameters; return the texts of the information boxes shown."""
    messages = []
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *a, **k: ("Apply to all sequentially", True),
    )
    monkeypatch.setattr(
        gui_render.LinkDialog, "getParams", lambda *a, **k: (30.0, 3, True)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda parent, title, text, *a, **k: messages.append(text),
    )
    view.link()
    _wait_for_tasks(qapp)
    return messages


def test_link_all_channels_sequentially(
    window, qt_offscreen, tmp_path, monkeypatch
):
    view = window.view
    view.add(str(tmp_path / "locs2.hdf5"), _locs(1), _info(), render_=False)
    n_before = [len(locs) for locs in view.locs]
    messages = _link_all_channels(view, qt_offscreen, monkeypatch)
    assert not messages
    for locs, n in zip(view.locs, n_before):
        assert "len" in locs.columns
        assert len(locs) < n


def test_link_all_channels_skips_linked_ones(
    window, qt_offscreen, tmp_path, monkeypatch
):
    view = window.view
    linked, _ = gui_render.postprocess.link(_locs(1), _info(), r_max=26.0)
    view.add(str(tmp_path / "linked.hdf5"), linked, _info(), render_=False)
    messages = _link_all_channels(view, qt_offscreen, monkeypatch)
    assert "len" in view.locs[0].columns
    assert view.locs[1] is linked
    assert len(messages) == 1 and "linked.hdf5" in messages[0]


def _cbc_params(pair, names) -> dict:
    return {
        "pair": pair,
        "names": names,
        "r_max": 200.0,
        "steps": 50,
        "undefined": -0.5,
        "remove_undefined": False,
        "use_z": False,
    }


def _answer_unlinked_warning(monkeypatch, answer) -> list[str]:
    """Answer the warning about unlinked channels; return its texts."""
    texts = []

    def warning(parent, title, text, *args, **kwargs):
        texts.append(text)
        return answer

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", warning)
    return texts


def test_cbc_unlinked_warning_can_abort(
    window, qt_offscreen, tmp_path, monkeypatch
):
    view = window.view
    view.add(str(tmp_path / "partner.hdf5"), _locs(1), _info(), render_=False)
    monkeypatch.setattr(
        gui_render.CBCDialog,
        "getParams",
        lambda *a, **k: (_cbc_params((1, 0), ["A", "B"]), True),
    )
    saved = []
    monkeypatch.setattr(
        gui_render.lib,
        "get_save_filename_ext_dialog",
        lambda *a, **k: saved.append(1) or ("", ""),
    )
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.No
    )
    view.cbc()
    assert len(texts) == 1 and "Link localizations" in texts[0]
    assert not saved  # aborted before asking where to save


def test_cbc_single_pair_saves_scored_channel(
    window, qt_offscreen, tmp_path, monkeypatch
):
    _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    view.add(str(tmp_path / "partner.hdf5"), _locs(1), _info(), render_=False)
    out = str(tmp_path / "scored_cbc.hdf5")
    monkeypatch.setattr(
        gui_render.CBCDialog,
        "getParams",
        lambda *a, **k: (_cbc_params((1, 0), ["Bassoon", "x"]), True),
    )
    monkeypatch.setattr(
        gui_render.lib,
        "get_save_filename_ext_dialog",
        lambda *a, **k: (out, ""),
    )
    view.cbc()
    _wait_for_tasks(qt_offscreen)
    locs, info = gui_render.io.load_locs(out)
    assert len(locs) == len(view.locs[1])
    assert "cbc_Bassoon" in locs.columns
    assert info[-1]["Scored channel"] == view.locs_paths[1]
    assert info[-1]["Partner channels"] == [view.locs_paths[0]]
    assert info[-1]["Undefined CBC value"] == -0.5


@pytest.mark.parametrize("n_channels", [2, 3])
def test_cbc_all_pairs_one_column_per_partner(
    window, qt_offscreen, tmp_path, monkeypatch, n_channels
):
    """Both directions, also with only two channels. Linked channels:
    no warning."""
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.No
    )
    view = window.view
    view.locs[0] = view.locs[0].assign(len=np.int32(1))
    for i in range(1, n_channels):
        locs = _locs(i).assign(len=np.int32(1))
        view.add(str(tmp_path / f"ch{i}.hdf5"), locs, _info(), render_=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    names = ["A", "B", "C"][:n_channels]
    monkeypatch.setattr(
        gui_render.CBCDialog,
        "getParams",
        lambda *a, **k: (_cbc_params(None, names), True),
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(out_dir),
    )
    view.cbc()
    _wait_for_tasks(qt_offscreen)
    files = ["locs", "ch1", "ch2"][:n_channels]
    for file, name in zip(files, names):
        locs, _ = gui_render.io.load_locs(str(out_dir / f"{file}_cbc.hdf5"))
        expected = {f"cbc_{other}" for other in names if other != name}
        assert set(locs.columns) & {f"cbc_{n}" for n in names} == expected
    assert not texts


def test_cbc_dialog_options(window, qt_offscreen, tmp_path, monkeypatch):
    view = window.view
    view.add(str(tmp_path / "ch1.hdf5"), _locs(1), _info(), render_=False)
    dialog = gui_render.CBCDialog(window)
    assert not dialog.all_pairs.isHidden()  # both directions of 2
    assert not dialog.use_z.isEnabled()  # no z in the test data
    assert [n.text() for n in dialog.names] == ["locs", "ch1"]
    dialog.undefined.setValue(5.0)
    assert dialog.undefined.value() == 1.0  # CBC range
    dialog.remove_undefined.setChecked(True)
    assert not dialog.undefined.isEnabled()
    dialog.all_pairs.setChecked(True)
    assert not dialog.scored.isEnabled()
    # names giving the same column are refused
    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda parent, title, text, *a, **k: messages.append(text),
    )
    dialog.names[0].setText("PSD.95")
    dialog.names[1].setText("PSD 95")
    dialog.accept()
    assert messages and "distinct" in messages[0]
    assert dialog.result() != QtWidgets.QDialog.DialogCode.Accepted
    # names are remembered for the session
    dialog.names[1].setText("Rim1")
    dialog.accept()
    assert gui_render.CBCDialog(window).names[1].text() == "Rim1"


def test_cbc_remove_undefined(window, qt_offscreen, tmp_path, monkeypatch):
    _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    # a partner covering only part of the field of view
    partner = _locs(1)
    partner = partner[partner["x"] < WIDTH * PIXELSIZE / 2]
    view.add(str(tmp_path / "half.hdf5"), partner, _info(), render_=False)
    out = str(tmp_path / "scored_cbc.hdf5")
    params = _cbc_params((0, 1), ["A", "B"])
    params["remove_undefined"] = True
    monkeypatch.setattr(
        gui_render.CBCDialog, "getParams", lambda *a, **k: (params, True)
    )
    monkeypatch.setattr(
        gui_render.lib,
        "get_save_filename_ext_dialog",
        lambda *a, **k: (out, ""),
    )
    view.cbc()
    _wait_for_tasks(qt_offscreen)
    locs, info = gui_render.io.load_locs(out)
    n_removed = info[-1]["Localizations without neighbors removed"]
    assert 0 < n_removed < len(view.locs[0])
    assert len(locs) == len(view.locs[0]) - n_removed


def _ripley_params(region: str, **kwargs) -> dict:
    params = {
        "channel": 0,
        "region": region,
        "r_max": 50.0,
        "steps": 10,
        "n_simulations": 2,
        "edge_correction": True,
        "use_z": False,
        "z_range": None,
        "simulated_z": "uniform",
        "partner": None,
        "cross_null": "shift",
    }
    params.update(kwargs)
    return params


def _run_spatial_statistic(
    view, qapp, monkeypatch, params, statistic="ripley_h"
) -> dict | None:
    monkeypatch.setattr(
        gui_render.SpatialStatisticDialog,
        "getParams",
        lambda *a, **k: (params, True),
    )
    view.spatial_statistic(statistic)
    _wait_for_tasks(qapp)
    window = view.spatial_statistic_windows.get(statistic)
    return None if window is None else window.result


def test_ripley_h_dialog_offers_picks_only_with_picks(window, qt_offscreen):
    dialog = gui_render.SpatialStatisticDialog(window, "ripley_h")
    regions = [dialog.region.itemData(i) for i in range(dialog.region.count())]
    assert regions == ["view", "fov"]
    assert dialog.channel.isHidden()  # one channel
    _pick_clusters(window.view)
    dialog = gui_render.SpatialStatisticDialog(window, "ripley_h")
    assert dialog.region.itemData(0) == "picks"
    assert f"({N_CLUSTERS})" in dialog.region.itemText(0)


def test_ripley_h_warning_without_molecules_can_abort(
    window, qt_offscreen, monkeypatch
):
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.No
    )
    result = _run_spatial_statistic(
        window.view, qt_offscreen, monkeypatch, _ripley_params("fov")
    )
    assert len(texts) == 1 and "does not contain molecules" in texts[0]
    assert result is None


def test_ripley_h_does_not_warn_for_molecules(
    window, qt_offscreen, monkeypatch
):
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.No
    )
    view = window.view
    view.locs[0]["n_events"] = np.int32(1)
    result = _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, _ripley_params("fov")
    )
    assert not texts
    assert result is not None
    view.spatial_statistic_windows["ripley_h"].close()


def test_ripley_h_in_picks_plots_and_saves(
    window, qt_offscreen, tmp_path, monkeypatch
):
    _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    _pick_clusters(view)
    result = _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, _ripley_params("picks")
    )
    assert len(result["n_locs"]) == N_CLUSTERS
    assert (result["n_locs"] > 0.9 * LOCS_PER_CLUSTER).all()
    assert result["envelope_low"] is not None
    plot_window = view.spatial_statistic_windows["ripley_h"]
    assert plot_window.figure.axes[0].get_xlabel() == "r (nm)"
    paths = plot_window.write_data(str(tmp_path / "ripley.csv"))
    assert paths[1] == str(tmp_path / "ripley_regions.csv")
    assert len(pd.read_csv(paths[0])) == 10
    assert len(pd.read_csv(paths[1])) == N_CLUSTERS
    plot_window.close()


@pytest.mark.parametrize("region", ["view", "fov"])
def test_ripley_h_view_and_field_of_view(
    window, qt_offscreen, monkeypatch, region
):
    _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    result = _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, _ripley_params(region)
    )
    # the view covers the whole field of view
    assert list(result["n_locs"]) == [len(view.locs[0])]
    assert result["areas"][0] == pytest.approx((WIDTH * PIXELSIZE) ** 2)
    assert result["h_mean"][0] > 0  # dense clusters
    view.spatial_statistic_windows["ripley_h"].close()


def test_ripley_h_reports_picks_without_localizations(
    window, qt_offscreen, monkeypatch
):
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    view._pick_shape = "Circle"
    view.window.tools_settings_dialog.pick_diameter.setValue(PIXELSIZE)
    view._picks = []
    view.add_picks([(PIXELSIZE, PIXELSIZE)])  # clusters start 5 px in
    result = _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, _ripley_params("picks")
    )
    assert result is None
    assert "at least two localizations" in texts[-1]


def test_ripley_h_dialog_shows_3d_settings_only_in_3d(window, qt_offscreen):
    dialog = gui_render.SpatialStatisticDialog(window, "ripley_h")
    assert not dialog.use_z.isEnabled()  # no z
    locs = window.view.locs[0]
    locs["z"] = np.linspace(-200.4, 300.6, len(locs))
    dialog = gui_render.SpatialStatisticDialog(window, "ripley_h")
    assert dialog.use_z.isEnabled() and not dialog.use_z.isChecked()
    assert dialog.z_limits.isHidden() and dialog.simulated_z.isHidden()
    assert (dialog.z_min.value(), dialog.z_max.value()) == (-201, 301)
    dialog.use_z.setChecked(True)
    assert not dialog.z_limits.isHidden()
    assert not dialog.simulated_z.isHidden()
    assert dialog.z_min.isHidden()  # z range from the localizations
    dialog.z_limits.setCurrentIndex(1)  # manual
    assert not dialog.z_min.isHidden()
    dialog.simulations.setValue(0)
    assert dialog.simulated_z.isHidden()


def test_ripley_h_in_3d_warns_about_a_thin_z_range(
    window, qt_offscreen, monkeypatch
):
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    rng = np.random.default_rng(0)
    view.locs[0]["z"] = rng.uniform(-50, 50, len(view.locs[0]))
    params = _ripley_params(
        "fov", use_z=True, z_range=(-60.0, 60.0), simulated_z="observed"
    )
    result = _run_spatial_statistic(view, qt_offscreen, monkeypatch, params)
    assert result["use_z"] and result["simulated_z"] == "observed"
    np.testing.assert_array_equal(result["z_ranges"], [[-60, 60]])
    assert "third of the z range" in texts[-1]
    view.spatial_statistic_windows["ripley_h"].close()


def test_rdf_plots_and_saves(window, qt_offscreen, tmp_path, monkeypatch):
    _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    _pick_clusters(view)
    result = _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, _ripley_params("picks"), "rdf"
    )
    assert len(result["n_locs"]) == N_CLUSTERS
    assert result["g"].shape == (N_CLUSTERS, 10)
    plot_window = view.spatial_statistic_windows["rdf"]
    assert plot_window.windowTitle().startswith("Radial distribution")
    assert plot_window.figure.axes[0].get_ylabel() == "g(r)"
    paths = plot_window.write_data(str(tmp_path / "rdf.csv"))
    assert "g_mean" in pd.read_csv(paths[0]).columns
    assert len(pd.read_csv(paths[1])) == N_CLUSTERS
    # both statistics keep their own window
    _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, _ripley_params("picks")
    )
    assert set(view.spatial_statistic_windows) == {"ripley_h", "rdf"}
    for plot_window in view.spatial_statistic_windows.values():
        plot_window.close()


def test_spatial_statistic_dialogs_default_steps(window, qt_offscreen):
    for statistic, steps in (("ripley_h", 100), ("rdf", 50)):
        dialog = gui_render.SpatialStatisticDialog(window, statistic)
        assert dialog.steps.value() == steps
        title = gui_render.SPATIAL_STATISTICS[statistic]["title"]
        assert dialog.windowTitle() == title


def test_spatial_statistic_dialog_partner_settings(window, qt_offscreen):
    dialog = gui_render.SpatialStatisticDialog(window, "rdf")
    assert dialog.partner.isHidden()  # one channel
    view = window.view
    view.add(
        view.locs_paths[0][:-5] + "_b.hdf5", _locs(1), _info(), render_=False
    )
    dialog = gui_render.SpatialStatisticDialog(window, "rdf")
    assert not dialog.partner.isHidden()
    assert dialog.partner.currentData() is None
    assert dialog.cross_null.isHidden()
    dialog.partner.setCurrentIndex(2)  # the second channel
    assert not dialog.cross_null.isHidden()
    assert dialog.cross_null.currentData() == "shift"
    dialog.simulations.setValue(0)
    assert dialog.cross_null.isHidden()
    # random shifts need rectangular or circular picks
    view._pick_shape = "Polygon"
    corners = [(5, 5), (30, 8), (20, 40), (5, 5)]  # camera pixels
    view._picks = [[(x * PIXELSIZE, y * PIXELSIZE) for x, y in corners]]
    dialog = gui_render.SpatialStatisticDialog(window, "rdf")
    dialog.partner.setCurrentIndex(2)
    assert dialog.region.currentData() == "picks"
    assert not dialog.cross_null.model().item(0).isEnabled()
    assert dialog.cross_null.currentData() == "labels"
    dialog.region.setCurrentIndex(1)  # current view
    assert dialog.cross_null.model().item(0).isEnabled()


def test_cross_ripley_h_of_two_channels(window, qt_offscreen, monkeypatch):
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    view.add(
        view.locs_paths[0][:-5] + "_b.hdf5", _locs(0), _info(), render_=False
    )
    params = _ripley_params("fov", partner=1, cross_null="labels")
    result = _run_spatial_statistic(view, qt_offscreen, monkeypatch, params)
    # neither channel holds molecules
    assert "and" in texts[0] and "do not contain molecules" in texts[0]
    assert result["cross_null"] == "labels"
    assert list(result["n_locs_partner"]) == [len(view.locs[1])]
    assert result["h_mean"][0] > 0  # the same clusters in both
    plot_window = view.spatial_statistic_windows["ripley_h"]
    assert " and " in plot_window.windowTitle()
    plot_window.close()


def test_nnd_dialog_lists_the_channels(window, qt_offscreen):
    view = window.view
    dialog = gui_render.SpatialStatisticDialog(window, "nnd")
    assert dialog.channels_list.isHidden()  # one channel
    assert dialog.edge_correction.isHidden()
    assert dialog.r_max.value() == 200 and dialog.steps.value() == 40
    view.add(
        view.locs_paths[0][:-5] + "_b.hdf5", _locs(1), _info(), render_=False
    )
    dialog = gui_render.SpatialStatisticDialog(window, "nnd")
    assert not dialog.channels_list.isHidden()
    assert dialog.channel.isHidden() and dialog.partner.isHidden()
    assert dialog.channels() == [0, 1]
    assert not dialog.cross_null.isHidden()  # two channels checked
    dialog.channels_list.item(1).setCheckState(
        gui_render.QtCore.Qt.CheckState.Unchecked
    )
    assert dialog.channels() == [0]
    assert dialog.cross_null.isHidden()


def test_nnd_of_two_channels(window, qt_offscreen, tmp_path, monkeypatch):
    texts = _answer_unlinked_warning(
        monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
    )
    view = window.view
    view.add(
        view.locs_paths[0][:-5] + "_b.hdf5", _locs(1), _info(), render_=False
    )
    params = _ripley_params(
        "fov", channels=[0, 1], r_max=100.0, steps=20, cross_null="labels"
    )
    result = _run_spatial_statistic(
        view, qt_offscreen, monkeypatch, params, "nnd"
    )
    assert "do not contain molecules" in texts[0]
    titles = [check.text() for check in window.dataset_dialog.checks]
    assert result["names"] == titles
    assert result["histograms"].shape == (2, 2, 20)
    assert result["cross_null"] == "labels"
    # dense clusters: short NNDs within each channel
    assert result["medians"][0, 0] < result["simulated_medians"][0, 0]
    plot_window = view.spatial_statistic_windows["nnd"]
    assert len(plot_window.figure.axes) == 4
    paths = plot_window.write_data(str(tmp_path / "nnd.csv"))
    assert paths[1] == str(tmp_path / "nnd_summary.csv")
    assert len(pd.read_csv(paths[1])) == 4
    plot_window.close()
