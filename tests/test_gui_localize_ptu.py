"""Picasso: Localize with PicoQuant ``.ptu`` scans.

Opening a photon-counting scan asks how many scan frames to sum and for
microtime gates, over the decay histogram of the file (the choice is
remembered and reaches the loader for ``.ptu`` files only), channels
(detectors, or detectors x gates) are chosen like the channels of other
multichannel formats, and a loaded photon-counting movie sets the camera
parameters to photon counting. The loader itself is tested in ``tests/test_io_ptu.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import localize as gui_localize
from tests.test_gui_localize_calibration import ACCEPTED, REJECTED, _answer


@pytest.fixture
def window(qt_offscreen):
    window = gui_localize.Window()
    yield window
    window.close()


class _RecordingWorker(gui_localize.MovieLoadWorker):
    """The real load worker, recording how it was made and loading
    nothing."""

    created = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _RecordingWorker.created.append(self)

    def run(self):
        self.finished.emit([], [], [])


def _wait_for_load(window, timeout=10.0):
    end = time.time() + timeout
    while window._load_thread is not None and time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)
    assert window._load_thread is None


@pytest.fixture
def recording(monkeypatch):
    _RecordingWorker.created = []
    monkeypatch.setattr(gui_localize, "MovieLoadWorker", _RecordingWorker)
    return _RecordingWorker.created


def _decay():
    """Decay histograms as ``io._ptu_decay_histograms`` returns them."""
    counts = np.zeros((2, 100), np.uint32)
    counts[:, 10:] = (1000 * np.exp(-np.arange(90) / 20)).astype(np.uint32)
    return {
        "names": ["Detector 1", "Detector 2"],
        "counts": counts,
        "bin_ns": 0.5,
    }


Dialog = gui_localize.PTUOptionsDialog


class TestPTUOptionsDialog:
    def test_defaults_keep_every_scan_frame_and_photon(self, window):
        dialog = Dialog(window)
        assert dialog.options() == {
            "frame_sum_window": 1,
            "frame_sum_stride": 1,
            "microtime_gates": None,
        }
        assert dialog.canvas is None  # no file, no decay histogram

    def test_stride_follows_the_window_unless_it_overlaps(self, window):
        dialog = Dialog(window)
        dialog.frame_window.setValue(20)
        # sums that did not overlap keep not overlapping
        assert dialog.frame_stride.value() == 20
        assert dialog.frame_stride.maximum() == 20
        dialog.frame_stride.setValue(1)
        dialog.frame_window.setValue(10)
        # a sliding window stays one
        assert dialog.frame_stride.value() == 1
        dialog.frame_window.setValue(5)
        dialog.frame_stride.setValue(5)
        dialog.frame_window.setValue(3)
        assert dialog.frame_stride.value() == 3

    def test_gates_are_added_and_removed(self, window):
        dialog = Dialog(window)
        dialog.add_gate(0, 12.5)
        dialog.add_gate()  # continues where the last gate ends
        assert dialog.gates() == [(0.0, 12.5), (12.5, 22.5)]
        dialog.remove_gate()
        assert dialog.options()["microtime_gates"] == [(0.0, 12.5)]

    def test_dragging_over_the_decay_adds_a_gate(self, window, monkeypatch):
        monkeypatch.setattr(io, "_ptu_decay_histograms", lambda path: _decay())
        dialog = Dialog(window, "scan.ptu")
        assert dialog.canvas is not None
        assert len(dialog.axes.lines) == 2  # one decay per detector
        dialog.on_span_select(4.123, 30.0)
        dialog.add_gate()  # reaches the end of the decay (100 bins)
        assert dialog.gates() == [(4.12, 30.0), (30.0, 50.0)]

    def test_unreadable_file_shows_a_note(self, window, monkeypatch):
        def fail(path):
            raise ValueError("not a scan")

        monkeypatch.setattr(io, "_ptu_decay_histograms", fail)
        dialog = Dialog(window, "scan.ptu")
        assert dialog.canvas is None
        dialog.add_gate()
        assert dialog.gates() == [(0.0, 10.0)]

    def test_choice_is_remembered(self, window, monkeypatch):
        def setup(dialog):
            dialog.frame_window.setValue(20)
            dialog.frame_stride.setValue(1)
            dialog.add_gate(0, 25)
            dialog.add_gate(25, 50)

        _answer(monkeypatch, Dialog, ACCEPTED, setup)
        options, ok = Dialog.getOptions(window)
        assert ok
        assert options == {
            "frame_sum_window": 20,
            "frame_sum_stride": 1,
            "microtime_gates": [(0.0, 25.0), (25.0, 50.0)],
        }
        stored = io.load_user_settings()["Localize"]["PTU options"]
        assert stored == {
            "window": 20,
            "stride": 1,
            "gates": [[0.0, 25.0], [25.0, 50.0]],
        }
        assert Dialog(window).options() == options

    def test_cancel_is_not_remembered(self, window, monkeypatch):
        _answer(
            monkeypatch,
            Dialog,
            REJECTED,
            lambda dialog: dialog.frame_window.setValue(7),
        )
        _, ok = Dialog.getOptions(window)
        assert not ok
        assert "PTU options" not in io.load_user_settings()["Localize"]

    def test_a_gate_ending_before_it_starts_is_refused(
        self, window, monkeypatch
    ):
        warnings = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **kwargs: warnings.append(args[2]),
        )
        dialog = Dialog(window)
        dialog.add_gate(20, 30)
        dialog.gate_table.cellWidget(0, 1).setValue(10)
        dialog.accept()
        assert "Gate 1 ends" in warnings[0]
        assert dialog.result() != ACCEPTED


class TestOpenPTU:
    def test_ptu_files_prompt_for_a_channel(self, window):
        assert window._prompt_for_path("a.PTU") == window.prompt_channel

    def test_the_options_reach_the_load(self, window, monkeypatch, recording):
        options = {
            "frame_sum_window": 20,
            "frame_sum_stride": 1,
            "microtime_gates": [(0.0, 25.0)],
        }
        asked = []

        def get_options(parent=None, path=None):
            asked.append(path)
            return options, True

        monkeypatch.setattr(Dialog, "getOptions", staticmethod(get_options))
        window._start_movie_load(
            ["movie.tif", "scan.ptu"], window._prompt_for_path
        )
        _wait_for_load(window)
        assert asked == ["scan.ptu"]  # the decay of the first .ptu
        assert recording[0].ptu_options == options

    def test_canceling_the_options_cancels_the_load(
        self, window, monkeypatch, recording
    ):
        monkeypatch.setattr(
            Dialog,
            "getOptions",
            staticmethod(lambda parent=None, path=None: ({}, False)),
        )
        window._start_movie_load(["scan.ptu"], window._prompt_for_path)
        assert window._load_thread is None
        assert recording == []

    def test_camera_movies_are_not_asked(self, window, monkeypatch, recording):
        def fail(parent=None, path=None):
            raise AssertionError("asked for PTU options")

        monkeypatch.setattr(Dialog, "getOptions", staticmethod(fail))
        window._start_movie_load(["movie.tif"], window._prompt_for_path)
        _wait_for_load(window)
        assert recording[0].ptu_options is None

    @pytest.mark.parametrize("load_all", [False, True])
    def test_only_ptu_files_get_the_options(self, monkeypatch, load_all):
        calls = []

        def loader(path, prompt_info=None, progress=None, **kwargs):
            calls.append((path, kwargs))
            return None

        name = "load_movie_all" if load_all else "load_movie"
        monkeypatch.setattr(io, name, loader)
        options = {
            "frame_sum_window": 3,
            "frame_sum_stride": 1,
            "microtime_gates": [(0.0, 25.0)],
        }
        worker = gui_localize.MovieLoadWorker(
            [], None, load_all=load_all, ptu_options=options
        )
        for path in ("scan.ptu", "movie.tif"):
            worker._load_job(path, [path], None, None)
        assert calls == [("scan.ptu", options), ("movie.tif", {})]


class TestPhotonCounting:
    def _load(self, window, tmp_path, photon_counting):
        movie = np.memmap(
            tmp_path / "scan.raw",
            dtype=np.uint32,
            mode="w+",
            shape=(4, 16, 16),
        )
        info = {"Frames": 4, "Height": 16, "Width": 16, "Pixelsize": 100.0}
        if photon_counting:
            info["Photon Counting"] = True
        window._set_channels(
            [movie], [[info]], [str(tmp_path / "scan.ptu")], ["Detector 1"]
        )

    def test_photon_counting_sets_the_camera(self, window, tmp_path):
        dialog = window.parameters_dialog
        dialog.baseline.setValue(100)
        dialog.sensitivity.setValue(0.45)
        dialog.gain.setValue(300)
        self._load(window, tmp_path, photon_counting=True)
        assert dialog.baseline.value() == 0
        assert dialog.sensitivity.value() == 1
        assert dialog.gain.value() == 1
        assert dialog.pixelsize.value() == 100.0
        assert window.camera_info["Baseline"] == 0

    def test_camera_movies_keep_their_parameters(self, window, tmp_path):
        dialog = window.parameters_dialog
        dialog.baseline.setValue(100)
        self._load(window, tmp_path, photon_counting=False)
        assert dialog.baseline.value() == 100
