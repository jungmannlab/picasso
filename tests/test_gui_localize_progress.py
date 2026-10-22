"""How Picasso: Localize shows the progress the core functions report.

The core functions take one ``progress`` tracker (``lib.Progress``) and
label and size their own phases; the workers hand them a
``gui.localize._SignalProgress`` that forwards phases and values as Qt
signals. These tests drive the trackers and the window's slots directly,
without threads.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import types

import pandas as pd
import pytest

from picasso import io, lib, localize
from picasso.gui import localize as gui_localize


@pytest.fixture(autouse=True)
def _qt_app(qapp):
    """The workers are QObjects, which need a QApplication."""
    return qapp


def _recording_tracker(**kwargs):
    seen = []
    tracker = gui_localize._SignalProgress(
        on_value=lambda v, m: seen.append(("value", v, m)),
        on_phase=lambda label, m: seen.append(("phase", label, m)),
        interval=0.0,
        **kwargs,
    )
    return tracker, seen


class TestSignalProgress:
    def test_a_phase_is_announced_with_its_label_and_maximum(self):
        """``lib.start_phase`` sets the label first and the maximum after
        it; the phase is announced once both are known."""
        tracker, seen = _recording_tracker()
        lib.start_phase(tracker, "Fitting spots", 5, unit="spot")
        tracker.set_value(2)
        tracker.set_value(5)
        assert seen == [
            ("phase", "Fitting spots", 5),
            ("value", 2, 5),
            ("value", 5, 5),
        ]

    def test_a_maximum_set_mid_phase_reaches_the_values(self):
        """A busy phase (maximum 0) that learns its length at the end, as
        io's frame indexing does, is not announced a second time."""
        tracker, seen = _recording_tracker()
        lib.start_phase(tracker, "Indexing a.tif", 0)
        tracker.set_value(2000)
        tracker.setMaximum(2345)
        tracker.set_value(2345)
        assert seen == [
            ("phase", "Indexing a.tif", 0),
            ("value", 2000, 0),
            ("value", 2345, 2345),
        ]

    def test_values_are_throttled_but_the_last_one_is_kept(self):
        tracker = gui_localize._SignalProgress(
            on_value=lambda v, m: seen.append(v), interval=60.0
        )
        seen = []
        lib.start_phase(tracker, "Cutting spots", 100)
        for i in range(101):
            tracker.set_value(i)
        assert seen == [0, 100]

    def test_cancel_raises_operation_canceled(self):
        canceled = []
        tracker, _ = _recording_tracker(canceled=lambda: bool(canceled))
        lib.start_phase(tracker, "Reading a.tif", 10)
        tracker.set_value(1)
        canceled.append(True)
        with pytest.raises(lib.OperationCanceled):
            tracker.set_value(2)
        with pytest.raises(lib.OperationCanceled):
            tracker.zero_progress("Reading b.tif")
        # io retries a stray keyframe on RuntimeError; a cancellation
        # must not be mistaken for one
        assert not issubclass(lib.OperationCanceled, RuntimeError)

    def test_it_is_a_progress_tracker(self):
        tracker = gui_localize._SignalProgress()
        assert isinstance(tracker, lib.Progress)
        assert lib.normalize_progress(tracker) is tracker

    def test_status_messages_name_the_phase(self):
        messages = []
        tracker = gui_localize._status_progress(
            messages.append, "Channel registration: "
        )
        tracker._interval = 0.0
        lib.start_phase(tracker, "Detecting spots", 2, unit="movie")
        tracker.set_value(1)
        lib.start_phase(tracker, "Busy", 0)
        assert messages == [
            "Channel registration: Detecting spots 0 / 2 ...",
            "Channel registration: Detecting spots 1 / 2 ...",
            "Channel registration: Busy ...",
        ]


def _info():
    return [{"Frames": 1, "Height": 4, "Width": 4}]


class TestMovieLoadWorker:
    def test_phases_and_pages_reach_the_gui(self, monkeypatch):
        def fake_load_movie(path, prompt_info=None, progress=None):
            lib.start_phase(progress, f"Indexing {path}", 0, unit="page")
            progress.setMaximum(3)
            progress.set_value(3)
            lib.start_phase(progress, f"Reading {path}", 3)
            progress.set_value(3)
            return "movie", _info()

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = gui_localize.MovieLoadWorker(
            ["a.tif"], prompt_for_path=lambda p: None
        )
        seen = []
        worker.phase.connect(lambda label, m: seen.append((label, m)))
        worker.subprogress.connect(lambda v, m: seen.append((v, m)))
        worker.run()
        assert seen == [
            ("Indexing a.tif", 0),
            (3, 3),
            ("Reading a.tif", 3),
            (3, 3),
        ]

    def test_cancel_mid_file_aborts_the_loader(self, monkeypatch):
        """The tracker is the mid-file cancellation point: the loader
        stops at its next report and the batch is discarded."""
        reached = []

        def fake_load_movie(path, prompt_info=None, progress=None):
            lib.start_phase(progress, f"Reading {path}", 3)
            worker.cancel()
            progress.set_value(1)
            reached.append(path)  # never reached
            return "movie", _info()

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = gui_localize.MovieLoadWorker(
            ["a.tif", "b.tif"], prompt_for_path=lambda p: None
        )
        out = {}
        worker.finished.connect(lambda *a: out.update(finished=a))
        worker.failed.connect(lambda m: out.update(failed=m))
        worker.run()
        assert reached == []
        assert out == {"finished": ([], [], [])}


class _Dialog:
    def __init__(self):
        self.label = ""
        self.value = 0

    def setLabelText(self, text):
        self.label = text

    def setValue(self, value):
        self.value = value


def _load_window(n_files, concat):
    window = types.SimpleNamespace(_load_progress=_Dialog())
    gui_localize.Window._on_load_progress(
        window, 0, f"{n_files} files" if concat else "a.tif"
    )
    return window


class TestLoadProgressDialog:
    RES = gui_localize.PROGRESS_RESOLUTION

    def test_indexing_shows_the_count(self):
        window = _load_window(1, concat=False)
        gui_localize.Window._on_load_phase(window, "Indexing a.tif", 0)
        assert window._load_progress.label.endswith("Indexing a.tif")
        gui_localize.Window._on_load_subprogress(window, 4000, 0)
        assert "4,000 counted so far" in window._load_progress.label
        # the bar only moves while reading
        gui_localize.Window._on_load_subprogress(window, 4321, 4321)
        assert window._load_progress.value == 0

    def test_reading_advances_the_bar(self):
        window = _load_window(1, concat=False)
        gui_localize.Window._on_load_phase(window, "Reading a.tif", 10)
        gui_localize.Window._on_load_subprogress(window, 5, 10)
        assert window._load_progress.value == self.RES // 2

    def test_a_concatenation_spans_one_bar(self):
        """Each file of a concatenation gets its share of the one bar."""
        window = _load_window(2, concat=True)
        on_phase = gui_localize.Window._on_load_phase
        on_value = gui_localize.Window._on_load_subprogress
        on_phase(window, "Indexing a.tif (1/2)", 0)
        on_phase(window, "Reading a.tif (1/2)", 10)
        on_value(window, 10, 10)
        assert window._load_progress.value == self.RES // 2
        on_phase(window, "Indexing b.tif (2/2)", 0)
        on_value(window, 2000, 0)
        assert window._load_progress.value == self.RES // 2
        on_phase(window, "Reading b.tif (2/2)", 4)
        on_value(window, 2, 4)
        assert window._load_progress.value == 3 * self.RES // 4
        assert window._load_progress.label.endswith("Reading b.tif (2/2)")

    def test_a_multi_file_ome_movie_spans_one_bar(self):
        """A single movie stored in several files (one ``load_movie``
        call) reports one "Reading" phase per file; the bar must not go
        back at each of them."""
        window = _load_window(1, concat=False)
        on_phase = gui_localize.Window._on_load_phase
        on_value = gui_localize.Window._on_load_subprogress
        values = []
        for j in (1, 2, 3):
            on_phase(window, f"Reading a_{j}.ome.tif ({j}/3)", 6)
            for done in (0, 3, 6):
                on_value(window, done, 6)
                values.append(window._load_progress.value)
        assert values == sorted(values)
        assert values[-1] == self.RES

    def test_parts_of_a_concatenated_file(self):
        """A concatenated file that is itself a multi-file OME set gets
        its parts placed within the file's share."""
        window = _load_window(2, concat=True)
        on_phase = gui_localize.Window._on_load_phase
        on_value = gui_localize.Window._on_load_subprogress
        on_phase(window, "Reading b_1.ome.tif (2/2, part 1/2)", 4)
        on_value(window, 4, 4)
        assert window._load_progress.value == 3 * self.RES // 4
        on_phase(window, "Reading b_2.ome.tif (2/2, part 2/2)", 4)
        on_value(window, 4, 4)
        assert window._load_progress.value == self.RES

    def test_a_retried_phase_covers_the_same_share(self):
        window = _load_window(2, concat=True)
        on_phase = gui_localize.Window._on_load_phase
        on_value = gui_localize.Window._on_load_subprogress
        for _ in range(2):  # e.g. the stray-keyframe retry
            on_phase(window, "Reading a.tif (1/2)", 10)
            on_value(window, 10, 10)
            assert window._load_progress.value == self.RES // 2


class TestFitStatus:
    def _window(self, worker=None):
        messages = []
        window = types.SimpleNamespace(
            _fit_phase="",
            fit_worker=worker,
            status_bar=types.SimpleNamespace(showMessage=messages.append),
        )
        window.on_fit_progress = lambda c, t: (
            gui_localize.Window.on_fit_progress(window, c, t)
        )
        return window, messages

    def test_the_status_bar_shows_the_phase(self):
        window, messages = self._window()
        gui_localize.Window.on_fit_phase(window, "Cutting spots", 20)
        gui_localize.Window.on_fit_progress(window, 10, 20)
        gui_localize.Window.on_fit_phase(window, "Fitting spots", 0)
        assert messages == [
            "Cutting spots 0 / 20 ...",
            "Cutting spots 10 / 20 ...",
            "Fitting spots ...",
        ]

    def test_the_fit_worker_reports_the_fits_phases(self, monkeypatch):
        """``localize.fit`` gets one tracker, whose phases arrive as the
        worker's ``phaseStarted`` and values as ``progressMade``."""

        def fake_fit(movie, info=None, **kwargs):
            progress = kwargs["progress"]
            for label in ("Cutting spots", "Fitting spots"):
                lib.start_phase(progress, label, 2, unit="spot")
                progress.set_value(2)
            return kwargs["identifications"], [{}]

        monkeypatch.setattr(localize, "fit", fake_fit)
        ids = pd.DataFrame({"frame": [0, 0], "x": [1, 2], "y": [1, 2]})
        worker = gui_localize.FitWorker(
            None, [{}], {}, ids, 7, "gausslq", None, None,
            False, False, False,
        )  # fmt: skip
        seen = []
        worker.phaseStarted.connect(lambda label, m: seen.append((label, m)))
        worker.progressMade.connect(lambda v, m: seen.append((v, m)))
        worker.finished.connect(lambda *a: seen.append("finished"))
        worker.run()
        assert seen == [
            ("Cutting spots", 2),
            (2, 2),
            ("Fitting spots", 2),
            (2, 2),
            "finished",
        ]


def test_quality_checks_take_a_tracker(monkeypatch):
    """``check_nena`` / ``check_drift`` get a tracker (a bare callable
    raises TypeError there, which ``check_nena`` turns into NaN)."""
    received = {}

    def fake_nena(locs, info, progress=None):
        received["nena"] = progress
        lib.start_phase(progress, "Calculating NeNA", 4)
        progress.set_value(4)
        return 1.0

    def fake_drift(locs, info, progress=None):
        received["drift"] = progress
        return 0.0, 0.0

    monkeypatch.setattr(localize, "check_nena", fake_nena)
    monkeypatch.setattr(localize, "check_drift", fake_drift)
    monkeypatch.setattr(localize, "check_kinetics", lambda *a: 1.0)
    monkeypatch.setattr(localize, "add_file_to_db", lambda *a, **k: None)
    monkeypatch.setattr(lib, "ensure_sanity", lambda locs, info: locs)
    worker = gui_localize.QualityWorker(
        pd.DataFrame({"frame": [0]}), [{"Frames": 1}], "x.hdf5", 130.0
    )
    messages = []
    worker.progressMade.connect(lambda msg, i, res: messages.append(msg))
    worker.run()
    assert isinstance(received["nena"], lib.Progress)
    assert isinstance(received["drift"], lib.Progress)
    assert (
        "Checking Quality (2/4) NeNA: Calculating NeNA 4 / 4 ..." in messages
    )
