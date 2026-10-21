"""Render's analysis and export dialogs, run as a user would: from the
menu entry, with the modal dialogs answered by the test.

- clustering: DBSCAN, HDBSCAN and the SMLM clusterer, the Test
  Clusterer dialog and the k-means clustering of picks (``Analyze
  cluster``);
- FRC resolution, in the current field of view and in ROIs;
- RESI;
- exports: grayscale images, localizations in other formats, the
  view with given parameters, Imaris, z slices, traces and profiles.

The worker of a cancelable task runs on a Qt thread, which coverage
does not trace; the files these tests check are its result.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import render as gui_render
from tests.test_gui_render_tasks import (
    HEIGHT,
    N_CLUSTERS,
    PIXELSIZE,
    WIDTH,
    _cluster_centers,
    _info,
    _locs,
    _wait_for_tasks,
)

APPLY_TO_ALL = "Apply to all sequentially"


def _saved(folder) -> list[str]:
    """The files in ``folder``, without the test's settings file."""
    return sorted(
        name for name in os.listdir(folder) if not name.startswith("settings")
    )


def _window(tmp_path, n_channels: int = 1) -> gui_render.Window:
    """A Render window with ``n_channels`` channels of the clustered
    localizations of ``test_gui_render_tasks``, saved as
    ``ch<k>.hdf5`` in ``tmp_path``."""
    window = gui_render.Window(plugins_loaded=True)
    for k in range(n_channels):
        path = str(tmp_path / f"ch{k}.hdf5")
        window.view.add(path, _locs(seed=k), _info(), render_=False)
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    window.view.resize(256, 256)
    return window


@pytest.fixture
def window(qt_offscreen, tmp_path):
    window = _window(tmp_path)
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


@pytest.fixture
def window2(qt_offscreen, tmp_path):
    """Two channels."""
    window = _window(tmp_path, n_channels=2)
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


class _Answers:
    """Answers to the modal dialogs of Render, in the order they are
    asked; records the questions."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.messages = []

    def save_path(self, *paths):
        """Answer the save dialogs with ``paths`` ("" cancels)."""
        answers = list(paths)

        def ask(*args, **kwargs):
            path = str(answers.pop(0))
            return path, os.path.splitext(path)[1]

        self.monkeypatch.setattr(
            gui_render.lib, "get_save_filename_ext_dialog", ask
        )

    def item(self, *items):
        """Answer ``QInputDialog.getItem`` (None cancels)."""
        answers = list(items)

        def ask(*args, **kwargs):
            item = answers.pop(0)
            return ("", False) if item is None else (item, True)

        self.monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", ask)

    def text(self, *texts):
        """Answer ``QInputDialog.getText`` (None cancels)."""
        answers = list(texts)

        def ask(*args, **kwargs):
            text = answers.pop(0)
            return ("", False) if text is None else (text, True)

        self.monkeypatch.setattr(QtWidgets.QInputDialog, "getText", ask)

    def integer(self, *values):
        """Answer ``QInputDialog.getInt`` (None cancels)."""
        answers = list(values)

        def ask(*args, **kwargs):
            value = answers.pop(0)
            return (0, False) if value is None else (value, True)

        self.monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", ask)

    def record_messages(self):
        """Record the texts of information and warning boxes."""

        def show(parent, title, text, *args, **kwargs):
            self.messages.append(text)
            return QtWidgets.QMessageBox.StandardButton.Ok

        for kind in ("information", "warning"):
            self.monkeypatch.setattr(QtWidgets.QMessageBox, kind, show)


@pytest.fixture
def answers(monkeypatch):
    return _Answers(monkeypatch)


def _params(dialog, monkeypatch, params, ok=True):
    """Answer ``dialog.getParams`` with ``params``."""
    monkeypatch.setattr(
        dialog, "getParams", staticmethod(lambda *a, **k: (params, ok))
    )


# ---------------------------------------------------------------------------
# Clustering from the menu
# ---------------------------------------------------------------------------

DBSCAN = dict(
    radius=30.0, min_density=4, min_locs=10, save_centers=True, save_areas=True
)
HDBSCAN = dict(
    min_cluster=50,
    min_samples=10,
    cluster_eps=0.0,
    save_centers=True,
    save_areas=False,
)
SMLM = dict(
    radius_xy=39.0,
    radius_z=39.0,
    min_locs=10,
    frame_analysis=False,
    save_centers=True,
    save_areas=False,
)


@pytest.mark.parametrize(
    "method, dialog, params, suffix",
    [
        ("dbscan", gui_render.DbscanDialog, DBSCAN, "_dbscan"),
        ("hdbscan", gui_render.HdbscanDialog, HDBSCAN, "_hdbscan"),
        ("smlm_clusterer", gui_render.SMLMDialog, SMLM, "_clustered"),
    ],
)
def test_clustering_saves_the_clusters(
    method,
    dialog,
    params,
    suffix,
    window,
    answers,
    monkeypatch,
    qt_offscreen,
    tmp_path,
):
    _params(dialog, monkeypatch, params)
    out = tmp_path / f"out{suffix}.hdf5"
    answers.save_path(out)
    getattr(window.view, method)()
    _wait_for_tasks(qt_offscreen)
    clustered, info = io.load_locs(str(out))
    centers, _ = io.load_locs(str(out).replace(".hdf5", "_centers.hdf5"))
    assert clustered["group"].nunique() == N_CLUSTERS
    assert len(centers) == N_CLUSTERS
    assert len(info) == 2  # the clustering is recorded
    has_areas = os.path.exists(str(out).replace(".hdf5", "_areas.csv"))
    assert has_areas is params["save_areas"]


def test_clustering_canceled_saves_nothing(
    window, answers, monkeypatch, qt_offscreen, tmp_path
):
    _params(gui_render.DbscanDialog, monkeypatch, DBSCAN)
    answers.save_path("")
    window.view.dbscan()
    _wait_for_tasks(qt_offscreen)
    _params(gui_render.DbscanDialog, monkeypatch, DBSCAN, ok=False)
    window.view.dbscan()
    _wait_for_tasks(qt_offscreen)
    assert _saved(tmp_path) == []


@pytest.mark.parametrize(
    "method, dialog, params",
    [
        ("dbscan", gui_render.DbscanDialog, DBSCAN),
        ("hdbscan", gui_render.HdbscanDialog, HDBSCAN),
        ("smlm_clusterer", gui_render.SMLMDialog, SMLM),
    ],
)
def test_clustering_of_all_channels(
    method,
    dialog,
    params,
    window2,
    answers,
    monkeypatch,
    qt_offscreen,
    tmp_path,
):
    _params(dialog, monkeypatch, params)
    answers.item(APPLY_TO_ALL)
    answers.text("_mine")
    getattr(window2.view, method)()
    _wait_for_tasks(qt_offscreen)
    for k in range(2):
        clustered, _ = io.load_locs(str(tmp_path / f"ch{k}_mine.hdf5"))
        assert clustered["group"].nunique() == N_CLUSTERS


# ---------------------------------------------------------------------------
# Test Clusterer dialog
# ---------------------------------------------------------------------------


# two clusters 107 nm apart, the next one 756 nm from their middle; one
# cluster alone is noise to HDBSCAN
PAIR = (11, 14)


def _pair_center() -> tuple[float, float]:
    centers = np.array(_cluster_centers())
    return tuple(centers[list(PAIR)].mean(axis=0))


@pytest.fixture
def test_dialog(window):
    """The Test Clusterer dialog with one circular pick around two
    clusters."""
    view = window.view
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(600.0)  # nm
    view._picks = [_pair_center()]
    dialog = window.test_clusterer_dialog
    dialog.show()
    yield dialog
    dialog.close()


def _set_test_params(dialog, name: str) -> None:
    dialog.clusterer_name.setCurrentText(name)
    if name == "DBSCAN":
        dialog.test_dbscan_params.radius.setValue(30.0)
        dialog.test_dbscan_params.min_samples.setValue(4)
        dialog.test_dbscan_params.min_locs.setValue(10)
    elif name == "HDBSCAN":
        dialog.test_hdbscan_params.min_cluster_size.setValue(50)
        dialog.test_hdbscan_params.min_samples.setValue(10)
    elif name == "SMLM":
        dialog.test_smlm_params.radius_xy.setValue(39.0)
        dialog.test_smlm_params.min_locs.setValue(10)
    elif name == "G5M":
        dialog.test_g5m_params.dbscan_radius.setValue(30.0)
        dialog.test_g5m_params.dbscan_min_samples.setValue(4)
        dialog.test_g5m_params.min_locs.setValue(10)


@pytest.mark.parametrize("name", ["DBSCAN", "HDBSCAN", "SMLM", "G5M"])
def test_test_clusterer_clusters_the_pick(name, test_dialog):
    _set_test_params(test_dialog, name)
    test_dialog.test_clusterer()
    view = test_dialog.view
    assert view.locs["group"].nunique() == 2  # the clusters in the pick
    assert len(view.centers) == 2
    assert not view.pixmap().isNull()
    # the view frames the picked localizations
    (y_min, x_min), (y_max, x_max) = view.viewport
    x, y = _pair_center()
    assert x_min < x < x_max and y_min < y < y_max


def test_test_clusterer_needs_one_pick(test_dialog, answers):
    answers.record_messages()
    test_dialog.window.view._picks = []
    test_dialog.test_clusterer()
    assert answers.messages == ["Choose only one pick region"]
    assert test_dialog.view.locs is None


def test_test_clusterer_without_clusters(test_dialog):
    _set_test_params(test_dialog, "DBSCAN")
    test_dialog.test_dbscan_params.min_locs.setValue(10_000)
    test_dialog.test_clusterer()
    assert test_dialog.view.text().startswith("No clusters found")


def test_test_clusterer_display_before_testing(test_dialog):
    """The display options and projections may be changed before the
    first test; there is nothing to show yet."""
    test_dialog.display_all_locs.setChecked(True)
    test_dialog.display_centers.setChecked(True)
    test_dialog.on_xy_proj()
    test_dialog.on_xz_proj()
    test_dialog.get_full_fov()
    test_dialog.view.zoom_in()
    assert test_dialog.view.pixmap().isNull()


@pytest.mark.parametrize(
    "all_locs, centers, n_channels",
    [
        (False, False, gui_render.N_GROUP_COLORS),
        (True, False, 2),
        (False, True, 2),
        (True, True, 3),
    ],
)
def test_test_clusterer_display_options(
    all_locs, centers, n_channels, test_dialog
):
    _set_test_params(test_dialog, "DBSCAN")
    test_dialog.test_clusterer()
    test_dialog.display_all_locs.setChecked(all_locs)
    test_dialog.display_centers.setChecked(centers)
    assert len(test_dialog.view.split_locs()) == n_channels
    assert not test_dialog.view.pixmap().isNull()


def test_test_clusterer_navigation(test_dialog):
    _set_test_params(test_dialog, "DBSCAN")
    test_dialog.test_clusterer()
    view = test_dialog.view
    start = np.array(view.viewport, dtype=float)
    view.to_right()
    assert np.array(view.viewport)[0, 1] > start[0, 1]
    view.to_left()
    view.to_down()
    assert np.array(view.viewport)[0, 0] > start[0, 0]
    view.to_up()
    np.testing.assert_allclose(view.viewport, start)
    view.zoom_in()
    assert render_width(view.viewport) < render_width(start)
    view.zoom_out()
    np.testing.assert_allclose(view.viewport, start)
    test_dialog.get_full_fov()
    np.testing.assert_allclose(view.viewport, start)
    # a 2D pick has no side views
    test_dialog.on_xz_proj()
    test_dialog.on_yz_proj()
    assert view.ang is None


def render_width(viewport) -> float:
    (_, x_min), (_, x_max) = viewport
    return x_max - x_min


@pytest.mark.parametrize("name", ["DBSCAN", "HDBSCAN", "SMLM", "G5M"])
def test_test_clusterer_applies_to_the_dataset(
    name, test_dialog, answers, qt_offscreen, tmp_path
):
    _set_test_params(test_dialog, name)
    test_dialog.display_centers.setChecked(True)
    out = tmp_path / "tested.hdf5"
    answers.save_path(out)
    test_dialog.apply_to_all()
    _wait_for_tasks(qt_offscreen)
    clustered, _ = io.load_locs(str(out))
    assert clustered["group"].nunique() == N_CLUSTERS
    assert os.path.exists(tmp_path / "tested_centers.hdf5")


def test_test_clusterer_applies_to_all_channels(
    qt_offscreen, tmp_path, answers
):
    window = _window(tmp_path, n_channels=2)
    dialog = window.test_clusterer_dialog
    _set_test_params(dialog, "DBSCAN")
    answers.item(APPLY_TO_ALL)
    answers.text("tested")
    dialog.apply_to_all()
    _wait_for_tasks(qt_offscreen)
    for k in range(2):
        assert os.path.exists(tmp_path / f"ch{k}_tested.hdf5")
    window.close()


# ---------------------------------------------------------------------------
# k-means clustering of picks (Analyze cluster)
# ---------------------------------------------------------------------------


@pytest.fixture
def two_picks(window):
    """Two circular picks, each around two neighboring clusters."""
    view = window.view
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(1000.0)  # nm
    view._picks = list(_cluster_centers()[:2])
    view.n_picks = 2
    return view


def _kmeans_replies(monkeypatch, *replies):
    """Answer the k-means dialogs (``ClsDlg2D``): "yes" keeps the pick,
    "no" removes it, "cancel" stops, "escape" closes it without a
    button, an int asks for that many clusters."""
    replies = list(replies)
    shown = []

    def exec_(dialog):
        reply = replies.pop(0)
        shown.append(dialog.n_clusters_spin.value())
        if reply == "yes":
            dialog.on_accept()
        elif reply == "no":
            dialog.on_reject()
        elif reply == "cancel":
            dialog.on_cancel()
        elif reply == "escape":
            dialog.reject()  # what Escape and the close button do
        else:
            dialog.n_clusters_spin.setValue(reply)  # closes the dialog
        return 0

    monkeypatch.setattr(gui_render.ClsDlg2D, "exec", exec_)
    return shown


def test_analyze_cluster_saves_kept_picks(
    two_picks, answers, monkeypatch, tmp_path
):
    shown = _kmeans_replies(monkeypatch, 2, "yes", "no")
    answers.integer(1, 3)  # initial number of clusters, gap size
    out = tmp_path / "kmeans.hdf5"
    answers.save_path(out)
    two_picks.analyze_cluster()
    # asked for 2 clusters, so the first pick was clustered again
    assert shown == [1, 2, 2]
    kept, _ = io.load_locs(str(out))
    assert set(kept["cluster"]) == {1, 2}
    assert kept["group"].nunique() == 1  # only the first pick was kept
    properties = pd.read_hdf(
        str(tmp_path / "kmeans_properties.hdf5"), "groups"
    )
    assert len(properties) == 2  # one row per k-means cluster
    # the second pick was rejected
    assert len(two_picks._picks) == 1


def test_analyze_cluster_in_3d(qt_offscreen, answers, monkeypatch, tmp_path):
    window = _window_3d(tmp_path)
    view = window.view
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(2000.0)
    view._picks = [(1000.0, 1000.0)]
    view.n_picks = 1
    shown = []

    def exec_(dialog):
        shown.append(len(dialog.checks))
        dialog.checks[0].setChecked(False)  # drop the first cluster
        dialog.on_accept()
        return 0

    monkeypatch.setattr(gui_render.ClsDlg3D, "exec", exec_)
    answers.item(view.locs_paths[0])  # the channel
    answers.integer(3, 3)
    out = tmp_path / "kmeans3d.hdf5"
    answers.save_path(out)
    view.analyze_cluster()
    assert shown == [3]
    kept, _ = io.load_locs(str(out))
    assert set(kept["cluster"]) == {0, 2, 3}  # 0: the dropped cluster
    properties = pd.read_hdf(
        str(tmp_path / "kmeans3d_properties.hdf5"), "groups"
    )
    assert len(properties) == 2
    assert properties["group"].dtype.kind == "i"
    window.close()


@pytest.mark.parametrize("reply", ["cancel", "escape"])
def test_analyze_cluster_cancel_stops(
    two_picks, answers, monkeypatch, tmp_path, reply
):
    """Closing the dialog without a button is a Cancel, not a reject
    that removes the pick."""
    _kmeans_replies(monkeypatch, reply)
    answers.integer(2)
    two_picks.analyze_cluster()
    assert len(two_picks._picks) == 2
    assert _saved(tmp_path) == []


def test_analyze_cluster_canceled_saving_saves_nothing(
    two_picks, answers, monkeypatch, tmp_path
):
    _kmeans_replies(monkeypatch, "yes", "yes")
    answers.integer(2, 3)
    answers.save_path("")
    monkeypatch.chdir(tmp_path)  # where a path of "" would save
    two_picks.analyze_cluster()
    assert _saved(tmp_path) == []


def test_analyze_cluster_canceled_number_of_clusters(
    two_picks, answers, monkeypatch
):
    shown = _kmeans_replies(monkeypatch)
    answers.integer(None)
    two_picks.analyze_cluster()
    assert shown == []
    assert len(two_picks._picks) == 2


# ---------------------------------------------------------------------------
# FRC resolution
# ---------------------------------------------------------------------------


def _frc_label_is_set(info_dialog) -> bool:
    text = info_dialog.frc_resolution.text()
    return text.endswith(" nm") or text == "n/a"


def test_frc_resolution_of_the_view(window, answers):
    answers.record_messages()
    info_dialog = window.info_dialog
    info_dialog.calculate_frc_resolution()
    assert _frc_label_is_set(info_dialog)
    assert {"resolution", "frc_curve"} <= set(info_dialog.frc_result)
    # a resolution, or the reason why there is none
    assert (info_dialog.frc_result["resolution"] is None) == bool(
        answers.messages
    )


def test_frc_saves_the_rendered_images(window, answers, tmp_path):
    answers.record_messages()
    window.info_dialog.save_images_check.setChecked(True)
    answers.save_path(tmp_path / "frc.tif")
    window.info_dialog.calculate_frc_resolution()
    assert _saved(tmp_path) == ["frc_1.tif", "frc_2.tif"]


@pytest.mark.parametrize("proceed", [False, True])
def test_frc_of_a_large_view_asks_first(proceed, window, answers, monkeypatch):
    answers.record_messages()
    asked = []
    button = QtWidgets.QMessageBox.StandardButton
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: asked.append(a[2])
        or (button.Yes if proceed else button.No),
    )
    window.view.median_lp = 0.1  # nm: a huge image at this pixel size
    window.info_dialog.calculate_frc_resolution()
    assert "The current FOV is large" in asked[0]
    assert bool(window.info_dialog.frc_result) is proceed


def test_frc_plot(window, answers):
    answers.record_messages()
    info_dialog = window.info_dialog
    info_dialog.show_frc_plot()  # calculates first
    assert info_dialog.frc_window.isVisible()
    assert info_dialog.frc_window.figure.axes
    info_dialog.frc_window.close()


def test_frc_plot_without_a_calculation(window2, answers):
    """Canceling the channel selection leaves nothing to plot."""
    answers.item(None)
    window2.info_dialog.show_frc_plot()
    assert not window2.info_dialog.frc_result
    assert getattr(window2.info_dialog, "frc_window", None) is None


@pytest.fixture
def frc_rois(window, answers, qt_offscreen):
    """The review window of FRC in 4 ROIs of 2 um."""
    answers.record_messages()
    info_dialog = window.info_dialog
    info_dialog.frc_n_rois.setValue(4)
    info_dialog.frc_roi_size.setValue(2.0)
    info_dialog.frc_min_locs.setValue(100)
    info_dialog.calculate_frc_rois()
    _wait_for_tasks(qt_offscreen)
    review = info_dialog.frc_rois_window
    yield review
    review.close()


def test_frc_rois_review(frc_rois, window):
    result = frc_rois.result
    n_rois = len(result["rois"])
    assert 1 <= n_rois <= 4
    assert frc_rois.table.rowCount() == n_rois
    assert frc_rois.selected_row() == 0
    assert frc_rois.figure.axes  # overview and FRC curve
    assert window.info_dialog.frc_rois_resolution.text() != ""


def test_frc_rois_unticked_rois_are_not_used(frc_rois, window):
    from PyQt6 import QtCore

    usable = [
        row
        for row in range(frc_rois.table.rowCount())
        if np.isfinite(frc_rois.result["resolutions"][row])
    ]
    if not usable:
        pytest.skip("no ROI with a resolution in the synthetic data")
    for row in usable:
        frc_rois.table.item(row, 0).setCheckState(
            QtCore.Qt.CheckState.Unchecked
        )
    assert not frc_rois.used.any()
    assert "No ROI used" in frc_rois.summary.text()
    assert window.info_dialog.frc_rois_resolution.text() == "-"


def test_frc_rois_click_selects_a_roi(frc_rois):
    from types import SimpleNamespace

    last = len(frc_rois.result["rois"]) - 1
    (y0, x0), (y1, x1) = frc_rois.result["rois"][last]
    event = SimpleNamespace(
        inaxes=frc_rois.figure.axes[0],
        xdata=(x0 + x1) / 2 / 1000,  # um
        ydata=(y0 + y1) / 2 / 1000,
    )
    frc_rois.on_click(event)
    assert frc_rois.selected_row() == last


def test_frc_rois_export(frc_rois, monkeypatch, tmp_path):
    out = tmp_path / "rois.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (str(out), "*.csv"),
    )
    frc_rois.export()
    table = pd.read_csv(out)
    assert len(table) == len(frc_rois.result["rois"])
    assert {"x_min_um", "resolution_nm", "used"} <= set(table.columns)


def test_frc_rois_reopen_the_review(frc_rois, window):
    frc_rois.close()
    window.info_dialog.show_frc_rois()
    assert frc_rois.isVisible()


def test_frc_rois_without_enough_localizations(window, answers, qt_offscreen):
    answers.record_messages()
    info_dialog = window.info_dialog
    info_dialog.frc_roi_size.setValue(2.0)
    info_dialog.frc_min_locs.setValue(10_000_000)
    info_dialog.calculate_frc_rois()
    _wait_for_tasks(qt_offscreen)
    assert info_dialog.frc_rois_window is None
    assert "No ROI with enough localizations" in answers.messages[0]


# ---------------------------------------------------------------------------
# RESI
# ---------------------------------------------------------------------------


def _resi_dialog(window) -> gui_render.RESIDialog:
    window.open_resi_dialog()
    dialog = window.dialogs[-1]
    dialog.radius_xy[0].setValue(39.0)  # nm
    dialog.min_locs[0].setValue(10)
    dialog.on_same_params_clicked()
    return dialog


def test_resi_needs_two_channels(window, answers):
    answers.record_messages()
    _resi_dialog(window).perform_resi()
    assert "requires at least two localization lists" in answers.messages[0]


def test_resi_same_parameters_for_all_channels(window2):
    dialog = _resi_dialog(window2)
    assert [r.value() for r in dialog.radius_xy] == [39.0, 39.0]
    assert [m.value() for m in dialog.min_locs] == [10, 10]
    dialog.close()


def test_resi_saves_the_cluster_centers_of_all_channels(
    window2, answers, qt_offscreen, tmp_path
):
    out = tmp_path / "resi.hdf5"
    answers.save_path(out)
    dialog = _resi_dialog(window2)
    dialog.perform_resi()
    _wait_for_tasks(qt_offscreen)
    resi, info = io.load_locs(str(out))
    assert sorted(set(resi["resi_channel_id"])) == [0, 1]
    assert len(resi) == 2 * N_CLUSTERS
    assert info[-1]["Paths to RESI channels"] == window2.view.locs_paths
    assert _saved(tmp_path) == ["resi.hdf5", "resi.yaml"]
    dialog.close()


def test_resi_saves_each_channel_on_request(
    window2, answers, qt_offscreen, tmp_path
):
    answers.save_path(tmp_path / "resi.hdf5")
    answers.text("_cl", "_cc")
    dialog = _resi_dialog(window2)
    dialog.save_clustered_locs.setChecked(True)
    dialog.save_cluster_centers.setChecked(True)
    dialog.perform_resi()
    _wait_for_tasks(qt_offscreen)
    for k in range(2):
        clustered, _ = io.load_locs(str(tmp_path / f"ch{k}_cl.hdf5"))
        centers, _ = io.load_locs(str(tmp_path / f"ch{k}_cc.hdf5"))
        assert clustered["group"].nunique() == len(centers) == N_CLUSTERS
    dialog.close()


@pytest.mark.parametrize("cancel", ["path", "suffix"])
def test_resi_canceled_saves_nothing(
    cancel, window2, answers, qt_offscreen, tmp_path
):
    if cancel == "path":
        answers.save_path("")
    else:
        answers.save_path(tmp_path / "resi.hdf5")
        answers.text(None)
    dialog = _resi_dialog(window2)
    dialog.save_clustered_locs.setChecked(True)
    dialog.perform_resi()
    _wait_for_tasks(qt_offscreen)
    assert _saved(tmp_path) == []
    dialog.close()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_export_grayscale(window2, answers, tmp_path):
    answers.text("_gray.png")
    window2.export_grayscale()
    # no scale bar is shown, so a copy with one is saved, too
    assert _saved(tmp_path) == [
        "ch0_gray.png",
        "ch0_gray.yaml",
        "ch0_gray_scalebar.png",
        "ch1_gray.png",
        "ch1_gray.yaml",
        "ch1_gray_scalebar.png",
    ]
    assert io.load_info(str(tmp_path / "ch0_gray.yaml"))[0]["Colormap"] == (
        "gray"
    )


@pytest.mark.parametrize("dpi", [150, None])
def test_export_grayscale_pdf(dpi, window, answers, tmp_path):
    answers.text("_gray.pdf")
    answers.integer(dpi)
    window.export_grayscale()
    assert ("ch0_gray.pdf" in _saved(tmp_path)) is (dpi is not None)


def test_export_grayscale_needs_an_image_format(window, answers, tmp_path):
    answers.record_messages()
    answers.text("_gray.jpg")
    window.export_grayscale()
    assert "does not end with .png, .tif, .pdf or .svg" in answers.messages[0]
    assert _saved(tmp_path) == []


@pytest.mark.parametrize(
    "item, check",
    [
        (".txt for ImageJ", np.loadtxt),  # frame, x, y without a header
        (".txt for NIS", lambda path: pd.read_csv(path, sep="\t")),
        (".csv for ThunderSTORM", pd.read_csv),
        (".mat for SMAP", None),
        (".hdf for PyME", None),
    ],
)
def test_export_localizations(item, check, window, answers, tmp_path):
    ext = item.split()[0]
    out = tmp_path / f"exported{ext}"
    answers.item(item)
    answers.save_path(out)
    window.export_multi()
    if ext == ".mat":  # SMAP needs the _sml suffix
        out = tmp_path / "exported_sml.mat"
    assert out.exists()
    if check is not None:
        assert len(check(out)) == len(window.view.locs[0])


def test_export_localizations_of_all_channels(window2, answers, tmp_path):
    answers.item(APPLY_TO_ALL, ".csv for ThunderSTORM")
    answers.text("_ts.csv")
    window2.export_multi()
    assert _saved(tmp_path) == ["ch0_ts.csv", "ch1_ts.csv"]


def test_export_localizations_suffix_needs_the_extension(
    window2, answers, tmp_path
):
    answers.item(APPLY_TO_ALL, ".csv for ThunderSTORM")
    answers.text("_ts.txt")
    window2.export_multi()
    assert _saved(tmp_path) == []


def test_export_pyme_of_all_channels_in_one_file(window2, answers, tmp_path):
    out = tmp_path / "both.hdf"
    answers.item(APPLY_TO_ALL, ".hdf for PyME")
    answers.save_path(out)
    window2.export_multi()
    locs, _ = io.import_pyme(str(out), PIXELSIZE)
    assert sorted(set(locs["probe"])) == [0, 1]


def _export_kwargs(window, monkeypatch, out, answers):
    """Answer the dialog of ``Export view manually`` with half of the
    field of view at 20 nm pixels."""
    kwargs = {
        "viewport": ((0.0, 0.0), (HEIGHT * PIXELSIZE / 2, WIDTH * PIXELSIZE)),
        "disp_px_size": 20.0,
        "min_blur_width": 0.0,
        "blur_method": "One-pixel",
    }
    _params(gui_render.ExportKwargsDialog, monkeypatch, kwargs)
    answers.save_path(out)
    settings = window.display_settings_dlg
    # a redraw would otherwise fit the display pixel size to the screen
    # and scale the contrast with it
    settings.dynamic_disp_px.setChecked(False)
    return settings.minimum.value(), settings.maximum.value()


def test_export_view_manually(window, answers, monkeypatch, tmp_path):
    from PIL import Image

    contrast = _export_kwargs(window, monkeypatch, tmp_path / "v.png", answers)
    window.export_kwargs()
    with Image.open(tmp_path / "v.png") as image:
        # 8.3 x 4.2 um at 20 nm
        assert image.size == (416, 208)
    assert _saved(tmp_path) == ["v.png", "v.yaml", "v_scalebar.png"]
    settings = window.display_settings_dlg
    assert (settings.minimum.value(), settings.maximum.value()) == contrast


def test_export_view_manually_canceled_restores_the_contrast(
    window, answers, monkeypatch, tmp_path
):
    contrast = _export_kwargs(window, monkeypatch, tmp_path / "v.pdf", answers)
    answers.integer(None)  # no DPI
    window.export_kwargs()
    assert _saved(tmp_path) == []
    settings = window.display_settings_dlg
    assert (settings.minimum.value(), settings.maximum.value()) == contrast
    assert not settings.minimum.signalsBlocked()
    assert not settings.maximum.signalsBlocked()


def test_export_fov_for_imaris(window2, answers, monkeypatch, tmp_path):
    from types import SimpleNamespace

    exported = []
    monkeypatch.setattr(
        gui_render, "PW", SimpleNamespace(Color=lambda *c: c), raising=False
    )
    monkeypatch.setattr(
        gui_render,
        "numpy_to_imaris",
        lambda image, path, colors, *args: exported.append(
            (image.shape, path, colors)
        ),
        raising=False,
    )
    # the left half of the field of view, without channel 1
    window2.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE / 2),
    ]
    window2.dataset_dialog.checks[1].setChecked(False)
    out = tmp_path / "roi.ims"
    answers.save_path(out)
    window2.export_fov_ims()
    locs, info = io.load_locs(str(tmp_path / "roi_ch_0.hdf5"))
    assert 0 < len(locs) < len(window2.view.locs[0])
    # the view keeps its aspect ratio, so the viewport was adapted
    (y_min, x_min), (y_max, x_max) = window2.view.viewport
    assert locs["x"].between(x_min, x_max).all()
    assert locs["y"].between(y_min, y_max).all()
    assert "Render (IMS Export)" in info[-1]["Generated by"]
    assert not (tmp_path / "roi_ch_1.hdf5").exists()
    ((shape, path, colors),) = exported
    assert path == str(out)
    assert shape[0] == len(colors) == 1  # one channel


def test_export_fov_for_imaris_canceled(window, answers, tmp_path):
    answers.save_path("")
    window.export_fov_ims()
    assert _saved(tmp_path) == []


def _window_3d(tmp_path) -> gui_render.Window:
    """Two channels of 3D localizations."""
    from tests.test_gui_render_slicer import HEIGHT as height_3d
    from tests.test_gui_render_slicer import WIDTH as width_3d
    from tests.test_gui_render_slicer import _info as info_3d
    from tests.test_gui_render_slicer import _locs as locs_3d

    window = gui_render.Window(plugins_loaded=True)
    for k in range(2):
        locs = locs_3d(z_range=200.0, seed=k)
        window.view.add(
            str(tmp_path / f"z{k}.hdf5"), locs, info_3d(), render_=False
        )
    window.view.viewport = [
        (0.0, 0.0),
        (height_3d * PIXELSIZE, width_3d * PIXELSIZE),
    ]
    window.view.resize(64, 64)
    return window


@pytest.mark.parametrize("separate", [True, False])
def test_export_z_slices(separate, qt_offscreen, answers, tmp_path):
    window = _window_3d(tmp_path)
    window.dataset_dialog.checks[1].setChecked(False)
    dialog = window.slicer_dialog
    dialog.initialize()
    dialog.separate_check.setChecked(separate)
    dialog.full_check.setChecked(True)
    answers.save_path(tmp_path / "stack.tif")
    dialog.export_stack()
    n_slices = dialog.sl.maximum() + 1
    saved = _saved(tmp_path)
    if separate:  # every channel, also the hidden one
        assert len(saved) == 2 * n_slices
        assert "stack_Z000_CH002.tif" in saved
    else:
        assert len(saved) == n_slices
    assert "stack_Z000_CH001.tif" in saved
    # the channels shown before are shown again
    checks = window.dataset_dialog.checks
    assert [check.isChecked() for check in checks] == [True, False]
    dialog.close()
    window.close()


def test_export_trace(window, answers, tmp_path):
    view = window.view
    view._pick_shape = "Circle"
    window.tools_settings_dialog.pick_diameter.setValue(300.0)
    view._picks = [_cluster_centers()[0]]
    view.show_trace()
    out = tmp_path / "trace.csv"
    answers.save_path(out)
    view.export_trace()
    trace = np.loadtxt(out, delimiter=",")
    assert trace.shape[1] == 3
    view.canvas.close()


def test_export_profile(window2, answers, tmp_path):
    view = window2.view
    window2.tools_settings_dialog.pick_shape.setCurrentText("Rectangle")
    window2.tools_settings_dialog.pick_width.setValue(300.0)
    (x0, y0), (x1, y1) = (_cluster_centers()[i] for i in PAIR)
    view._picks = [((x0, y0), (x1, y1))]
    answers.item(APPLY_TO_ALL)
    view.plot_profile()
    out = tmp_path / "profile.csv"
    answers.save_path(out)
    view.export_profile()
    profile = pd.read_csv(out)
    assert list(profile.columns) == ["ch0.hdf5 (nm)", "ch1.hdf5 (nm)"]
    assert profile["ch0.hdf5 (nm)"].notna().sum() > 0
    view.canvas.close()


def test_export_profile_without_a_profile(window, answers):
    answers.record_messages()
    window.view.export_profile()
    assert answers.messages == ["No profile to export."]
