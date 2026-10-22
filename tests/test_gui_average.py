"""The user flows of Picasso: Average.

Opening picked localizations (from the File menu or by dropping them on
the view), averaging them in the background with the pool of worker
processes that opening started, aborting, and saving the result. The
data is a handful of small, rotated copies of one asymmetric particle;
the pool runs two worker processes, which every test shuts down.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import average, io, lib
from picasso.gui import average as gui_average

N_GROUPS = 6
PIXELSIZE = 130
WIDTH = HEIGHT = 32  # camera pixels


def _particles(seed: int = 0) -> pd.DataFrame:
    """Copies of an L-shaped particle (three sites, nm) at random
    angles, side by side in the field of view, ten localizations per
    site."""
    rng = np.random.default_rng(seed)
    template = np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 20.0]])
    frames = []
    for group in range(N_GROUPS):
        angle = rng.uniform(0, 2 * np.pi)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        sites = template @ rotation.T + [500.0 + 600 * group, 2000.0]
        xy = np.repeat(sites, 10, axis=0) + rng.normal(0, 2, (30, 2))
        frames.append(
            pd.DataFrame(
                {
                    "frame": rng.integers(0, 100, 30).astype(np.int32),
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "photons": np.full(30, 1000.0, dtype=np.float32),
                    "sx": np.full(30, 130.0, dtype=np.float32),
                    "sy": np.full(30, 130.0, dtype=np.float32),
                    "bg": np.full(30, 10.0, dtype=np.float32),
                    "lpx": np.full(30, 2.0, dtype=np.float32),
                    "lpy": np.full(30, 2.0, dtype=np.float32),
                    "group": np.full(30, group, dtype=np.int32),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 100,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture
def picked(tmp_path):
    path = tmp_path / "picked.hdf5"
    io.save_locs(str(path), _particles(), _info())
    return str(path)


@pytest.fixture
def messages(monkeypatch):
    """Message boxes shown, as ``(title, text)``."""
    shown = []

    def box(parent, title, text, *args, **kwargs):
        shown.append((title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind, staticmethod(box))
    return shown


@pytest.fixture
def window(qt_offscreen, messages, monkeypatch):
    """An Average window whose pool has two workers; any averaging
    thread and pool are stopped at the end."""
    monkeypatch.setattr(lib, "n_workers", lambda *args, **kwargs: 2)
    notified = []
    monkeypatch.setattr(
        gui_average.notifications, "notify_finished", notified.append
    )
    window = gui_average.Window()
    window.notified = notified
    window.statuses = []
    window.statusBar().messageChanged.connect(window.statuses.append)
    yield window
    view = window.view
    if view.thread is not None:
        view.thread.requestInterruption()
        view.thread.wait()
    # delivers ``finished``, which may start a new pool after an abort
    qt_offscreen.processEvents()
    view.shutdown_pool()
    window.close()


def _choose_file(monkeypatch, path: str) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (path, "")),
    )


def _choose_save_path(monkeypatch, path: str) -> None:
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (path, "")),
    )


def _open(window, monkeypatch, path: str) -> None:
    _choose_file(monkeypatch, path)
    window.open()


def _wait(view, qapp) -> None:
    """Wait for the averaging thread and deliver its signals."""
    view.thread.wait()
    qapp.processEvents()


def _average(window, qapp, disp_px_size=5.0, iterations=2) -> None:
    window.parameters_dialog.disp_px_size.setValue(disp_px_size)
    window.parameters_dialog.iterations.setValue(iterations)
    window.view.average()
    _wait(window.view, qapp)


# -- opening ----------------------------------------------------------


def test_canceling_the_open_dialog_loads_nothing(window, monkeypatch):
    _open(window, monkeypatch, "")
    assert not hasattr(window.view, "locs")
    assert window.view.pool_thread is None


def test_opening_centers_each_particle_and_starts_the_pool(
    window, picked, monkeypatch
):
    _open(window, monkeypatch, picked)

    view = window.view
    assert view.path == picked
    centers = view.locs.groupby("group")[["x", "y"]].mean()
    np.testing.assert_allclose(centers.to_numpy(), 0, atol=1e-3)
    assert view.r > 0
    assert view.pixmap() is not None and not view.pixmap().isNull()
    assert window.statuses[-1] == "Ready for processing!"
    pool = view.wait_for_pool()
    assert isinstance(pool, average.AveragePool)
    assert pool.n_groups == N_GROUPS
    assert view.pool_thread is None


def test_opening_another_file_replaces_the_pool(window, picked, monkeypatch):
    _open(window, monkeypatch, picked)
    first = window.view.wait_for_pool()

    _open(window, monkeypatch, picked)

    assert first.closed
    assert window.view.wait_for_pool() is not first


def test_a_file_that_cannot_be_loaded_is_reported(
    window, messages, tmp_path, monkeypatch
):
    _open(window, monkeypatch, str(tmp_path / "missing.hdf5"))
    assert len(messages) == 1
    assert not hasattr(window.view, "locs")
    assert window.view.pool_thread is None


def test_particles_without_extent_are_not_kept(
    window, messages, monkeypatch, tmp_path
):
    """With one localization per pick, the centered particles are all
    at (0, 0): there is no image to render or align."""
    locs = _particles().groupby("group").head(1).reset_index(drop=True)
    path = str(tmp_path / "single.hdf5")
    io.save_locs(path, locs, _info())

    _open(window, monkeypatch, path)

    ((title, text),) = messages
    assert "no extent" in text
    assert not hasattr(window.view, "locs")
    assert window.view.pool_thread is None


def test_dropping_a_localization_file_opens_it(window, picked, tmp_path):
    view = window.view

    def drop(path):
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(path)])
        event = QtGui.QDropEvent(
            QtCore.QPointF(5, 5),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        view.dropEvent(event)

    drop(str(tmp_path / "movie.raw"))
    assert not hasattr(view, "locs")
    drop(picked)
    assert view.path == picked


def test_only_drags_with_files_are_accepted(window):
    def drag(mime):
        event = QtGui.QDragEnterEvent(
            QtCore.QPoint(5, 5),
            QtCore.Qt.DropAction.CopyAction,
            mime,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        window.view.dragEnterEvent(event)
        return event.isAccepted()

    files = QtCore.QMimeData()
    files.setUrls([QtCore.QUrl.fromLocalFile("/data/picked.hdf5")])
    text = QtCore.QMimeData()
    text.setText("picked.hdf5")
    assert drag(files)
    assert not drag(text)


def test_display_pixel_size_sets_the_image_resolution(
    window, picked, monkeypatch
):
    _open(window, monkeypatch, picked)
    coarse = window.view._pixmap.width()

    window.parameters_dialog.disp_px_size.setValue(5.0)

    # the default is 10 nm, i.e., twice as many pixels at 5 nm
    assert window.view._pixmap.width() == pytest.approx(2 * coarse, abs=2)


def test_metadata_of_the_opened_file(window, picked, monkeypatch):
    _open(window, monkeypatch, picked)
    window.show_metadata()
    assert window.metadata_dialog.isVisible()
    window.metadata_dialog.close()


# -- averaging --------------------------------------------------------


def _average_runs(view) -> list[tuple]:
    """(display pixel size, iterations) of each averaging run recorded in
    the metadata of the average."""
    return [
        (entry["Display pixel size (nm)"], entry["Iterations"])
        for entry in view.info
        if entry.get("Step") == "Average"
    ]


def test_averaging_records_its_parameters_and_reuses_the_pool(
    window, picked, monkeypatch, qt_offscreen
):
    _open(window, monkeypatch, picked)
    before = window.view.locs.copy()

    _average(window, qt_offscreen, disp_px_size=5.0, iterations=2)

    view = window.view
    assert not view.running
    assert not window.abort_action.isEnabled()
    assert window.statuses[-1] == "Done!"
    assert "Iteration 2/2 — group 6/6" in window.statuses
    assert len(window.notified) == 1
    assert _average_runs(view) == [(5.0, 2)]
    assert len(view.locs) == len(before)
    assert not np.allclose(view.locs["x"], before["x"])
    pool = view.pool

    _average(window, qt_offscreen, disp_px_size=8.0, iterations=1)

    assert view.pool is pool and not pool.closed
    assert _average_runs(view) == [(5.0, 2), (8.0, 1)]


def test_average_while_averaging_starts_nothing(
    window, picked, monkeypatch, qt_offscreen
):
    _open(window, monkeypatch, picked)
    window.parameters_dialog.iterations.setValue(1)
    window.view.average()
    thread = window.view.thread

    window.view.average()

    assert window.view.thread is thread
    _wait(window.view, qt_offscreen)
    assert len(_average_runs(window.view)) == 1


def test_abort_stops_averaging_and_starts_a_new_pool(
    window, picked, monkeypatch, qt_offscreen
):
    _open(window, monkeypatch, picked)
    # more iterations than could finish before the abort
    window.parameters_dialog.iterations.setValue(100_000)
    window.view.average()
    pool = window.view.pool
    assert window.abort_action.isEnabled()

    window.view.abort()

    assert window.statuses[-1] == "Aborting..."
    assert not window.abort_action.isEnabled()
    _wait(window.view, qt_offscreen)
    view = window.view
    assert view.thread.was_aborted
    assert window.statuses[-1] == "Aborted."
    assert _average_runs(view) == []
    assert not view.running
    # aborting terminated the pool, so the next run gets a new one
    assert pool.closed
    assert view.pool is None and view.pool_thread is not None
    assert window.notified == []


def test_opening_another_file_while_averaging_is_refused(
    window, messages, picked, monkeypatch, tmp_path, qt_offscreen
):
    """Opening would shut down the pool that the running averaging
    works with."""
    _open(window, monkeypatch, picked)
    window.parameters_dialog.iterations.setValue(100_000)
    window.view.average()
    pool = window.view.pool
    other = str(tmp_path / "other.hdf5")
    io.save_locs(other, _particles(seed=1).iloc[:120], _info())

    _open(window, monkeypatch, other)

    assert messages == [
        (
            "Average",
            "Averaging is running. Abort it before opening another file.",
        )
    ]
    assert window.view.path == picked
    assert len(window.view.locs) == 6 * 30
    assert not pool.closed
    window.view.abort()
    _wait(window.view, qt_offscreen)


# -- saving -----------------------------------------------------------


def test_canceling_the_save_dialog_writes_nothing(
    window, picked, monkeypatch, tmp_path
):
    _open(window, monkeypatch, picked)
    _choose_save_path(monkeypatch, "")
    before = set(tmp_path.iterdir())

    window.save()

    assert set(tmp_path.iterdir()) == before


def test_saved_average_is_centered_in_the_field_of_view(
    window, picked, monkeypatch, tmp_path, qt_offscreen
):
    _open(window, monkeypatch, picked)
    _average(window, qt_offscreen, disp_px_size=5.0, iterations=2)
    window.parameters_dialog.iterations.setValue(4)  # not used yet
    path = str(tmp_path / "picked_avg.hdf5")
    _choose_save_path(monkeypatch, path)

    window.save()

    assert window.statuses[-1] == f"File saved to {path}."
    locs, info = io.load_locs(path)
    center = WIDTH * PIXELSIZE / 2
    np.testing.assert_allclose(
        locs["x"], window.view.locs["x"] + center, atol=1e-3
    )
    np.testing.assert_allclose(
        locs["y"], window.view.locs["y"] + center, atol=1e-3
    )
    assert info[-1]["Step"] == "Average"
    # the run, not the iterations now on the dialog
    assert info[-1]["Display pixel size (nm)"] == 5.0
    assert info[-1]["Iterations"] == 2


def test_saving_twice_writes_the_same_localizations(
    window, picked, monkeypatch, tmp_path, qt_offscreen
):
    """Saving shifts the average to the center of the field of view;
    the shift is applied to the saved copy, not to the average shown,
    which stays centered at (0, 0)."""
    _open(window, monkeypatch, picked)
    _average(window, qt_offscreen, iterations=1)
    shown = window.view.locs.copy()
    paths = [str(tmp_path / f"avg_{i}.hdf5") for i in range(2)]
    for path in paths:
        _choose_save_path(monkeypatch, path)
        window.save()

    first, second = (io.load_locs(path)[0] for path in paths)
    np.testing.assert_allclose(first["x"], second["x"])
    np.testing.assert_allclose(first["y"], second["y"])
    pd.testing.assert_frame_equal(window.view.locs, shown)
