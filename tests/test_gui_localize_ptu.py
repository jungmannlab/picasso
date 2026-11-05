"""Picasso: Localize with PicoQuant ``.ptu`` scans.

Opening a photon-counting scan asks how many scan frames to sum (the
choice is remembered and reaches the loader for ``.ptu`` files only),
detectors are chosen like the channels of other multichannel formats, and
a loaded photon-counting movie sets the camera parameters to photon
counting. The loader itself is tested in ``tests/test_io_ptu.py``.

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


class TestFrameSumDialog:
    def test_defaults_keep_every_scan_frame(self, window):
        dialog = gui_localize.PromptFrameSumDialog(window)
        assert dialog.frame_window.value() == 1
        assert dialog.frame_stride.value() == 1

    def test_stride_follows_the_window_unless_it_overlaps(self, window):
        dialog = gui_localize.PromptFrameSumDialog(window)
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

    def test_choice_is_remembered(self, window, monkeypatch):
        def setup(dialog):
            dialog.frame_window.setValue(20)
            dialog.frame_stride.setValue(1)

        _answer(
            monkeypatch, gui_localize.PromptFrameSumDialog, ACCEPTED, setup
        )
        assert gui_localize.PromptFrameSumDialog.getFrameSum(window) == (
            20,
            1,
            True,
        )
        stored = io.load_user_settings()["Localize"]["PTU frame sum"]
        assert stored == {"window": 20, "stride": 1}
        dialog = gui_localize.PromptFrameSumDialog(window)
        assert dialog.frame_window.value() == 20
        assert dialog.frame_stride.value() == 1

    def test_cancel_is_not_remembered(self, window, monkeypatch):
        _answer(
            monkeypatch,
            gui_localize.PromptFrameSumDialog,
            REJECTED,
            lambda dialog: dialog.frame_window.setValue(7),
        )
        *_, ok = gui_localize.PromptFrameSumDialog.getFrameSum(window)
        assert not ok
        assert "PTU frame sum" not in io.load_user_settings()["Localize"]


class TestOpenPTU:
    def test_ptu_files_prompt_for_a_detector(self, window):
        assert window._prompt_for_path("a.PTU") == window.prompt_channel

    def test_the_frame_sum_reaches_the_load(
        self, window, monkeypatch, recording
    ):
        monkeypatch.setattr(
            gui_localize.PromptFrameSumDialog,
            "getFrameSum",
            staticmethod(lambda parent=None: (20, 1, True)),
        )
        window._start_movie_load(["scan.ptu"], window._prompt_for_path)
        _wait_for_load(window)
        assert recording[0].frame_sum == (20, 1)

    def test_canceling_the_frame_sum_cancels_the_load(
        self, window, monkeypatch, recording
    ):
        monkeypatch.setattr(
            gui_localize.PromptFrameSumDialog,
            "getFrameSum",
            staticmethod(lambda parent=None: (20, 20, False)),
        )
        window._start_movie_load(["scan.ptu"], window._prompt_for_path)
        assert window._load_thread is None
        assert recording == []

    def test_camera_movies_are_not_asked(self, window, monkeypatch, recording):
        def fail(parent=None):
            raise AssertionError("asked for a frame sum")

        monkeypatch.setattr(
            gui_localize.PromptFrameSumDialog,
            "getFrameSum",
            staticmethod(fail),
        )
        window._start_movie_load(["movie.tif"], window._prompt_for_path)
        _wait_for_load(window)
        assert recording[0].frame_sum is None

    @pytest.mark.parametrize("load_all", [False, True])
    def test_only_ptu_files_get_the_frame_sum(self, monkeypatch, load_all):
        calls = []

        def loader(path, prompt_info=None, progress=None, **kwargs):
            calls.append((path, kwargs))
            return None

        name = "load_movie_all" if load_all else "load_movie"
        monkeypatch.setattr(io, name, loader)
        worker = gui_localize.MovieLoadWorker(
            [], None, load_all=load_all, frame_sum=(3, 1)
        )
        for path in ("scan.ptu", "movie.tif"):
            worker._load_job(path, [path], None, None)
        assert calls == [
            ("scan.ptu", {"frame_sum_window": 3, "frame_sum_stride": 1}),
            ("movie.tif", {}),
        ]


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
