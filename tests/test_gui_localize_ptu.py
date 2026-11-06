"""Picasso: Localize with PicoQuant ``.ptu`` scans.

Opening a photon-counting scan asks how many scan frames to sum and for
microtime gates, over the decay histogram of the file (the choice is
remembered and reaches the loader for ``.ptu`` files only), channels
(detectors, or detectors x gates) are chosen like the channels of other
multichannel formats, and a loaded photon-counting movie sets the camera
parameters to photon counting and can be displayed as lifetimes. The loader
itself is tested in ``tests/test_io_ptu.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt6 import QtGui, QtWidgets

from picasso import io
from picasso.gui import localize as gui_localize
from tests.test_gui_localize_calibration import ACCEPTED, REJECTED, _answer
from tests.test_io_ptu import _install_fake_ptufile


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


# ---------------------------------------------------------------------------
# Lifetime display
# ---------------------------------------------------------------------------


def _lifetime_scan():
    """4 scan frames of 6 x 8 pixels in 16 microtime bins of 0.04 ns: one
    photon per pixel and scan frame in bin 2 (the maximum of the decay),
    and ten more in bin 12 in pixel (y, x) = (1, 2)."""
    data = np.zeros((4, 6, 8, 1, 16), np.uint16)
    data[..., 0, 2] = 1
    data[:, 1, 2, 0, 12] = 10
    return data


# mean delay of the photons of pixel (1, 2) after bin 2, in ns
LIFETIME = (0.5 + 10 * 10.5) / 11 * 0.04


@pytest.fixture
def scan(window, tmp_path, monkeypatch):
    """A photon-counting movie of ``_lifetime_scan`` shown in Localize."""
    data = _lifetime_scan()
    _install_fake_ptufile(monkeypatch, data)

    def decay(self, first=0, last=None, *, dtime_bin=1):
        # the fake file has no records: the decay of its pixels
        lo, hi = self._microtime_range
        return data[first:last, ..., 0, lo:hi].sum((0, 1, 2)).astype(np.uint64)

    monkeypatch.setattr(io.PTUMovie, "_decay", decay)
    path = str(tmp_path / "scan.ptu")
    movie = io.PTUMovie(path)
    window._set_channels([movie], [[movie.info()]], [path], ["Detector 1"])
    return movie


def _pixel(window, x, y):
    """``(r, g, b)`` of a pixel of the frame displayed."""
    (item,) = [
        item
        for item in window.scene.items()
        if isinstance(item, QtWidgets.QGraphicsPixmapItem)
    ]
    color = item.pixmap().toImage().pixelColor(x, y)
    return color.red(), color.green(), color.blue()


class TestLifetimeDisplay:
    def test_offered_for_photon_counting_movies_only(
        self, window, scan, tmp_path
    ):
        assert window.lifetime_action.isEnabled()
        assert window.lifetime_settings_action.isEnabled()
        assert not window.lifetime_button.isHidden()
        window.lifetime_button.setChecked(True)
        assert window.lifetime_action.isChecked()
        assert window.lifetime_shown()
        camera = np.memmap(
            tmp_path / "movie.raw", dtype=np.uint16, mode="w+", shape=(2, 6, 8)
        )
        info = {"Frames": 2, "Height": 6, "Width": 8}
        window._set_channels(
            [camera], [[info]], [str(tmp_path / "movie.raw")], ["Channel 1"]
        )
        assert not window.lifetime_action.isEnabled()
        assert not window.lifetime_settings_action.isEnabled()
        assert window.lifetime_button.isHidden()
        assert not window.lifetime_shown()

    def test_lifetimes_are_mean_photon_delays_in_the_window(
        self, window, scan
    ):
        lifetimes, photons = window.lifetimes()
        assert photons[1, 2] == 11
        assert lifetimes[1, 2] == pytest.approx(LIFETIME)
        # a single photon is fewer than the 5 a lifetime needs by default
        assert photons[0, 0] == 1
        assert np.isnan(lifetimes[0, 0])
        # from the maximum (bin 2) to the end of the microtimes (bin 16)
        assert window.lifetime_window_ns() == pytest.approx((0.08, 0.64))
        # starting 5 bins after the maximum leaves out bin 2
        window.lifetime_dialog.start_spinbox.setValue(0.2)
        lifetimes, photons = window.lifetimes()
        assert photons[1, 2] == 10
        assert lifetimes[1, 2] == pytest.approx(5.5 * 0.04)

    def test_switching_colors_the_frame(self, window, scan):
        contrast = window.contrast_dialog
        contrast.auto_checkbox.setChecked(False)
        contrast.change_contrast_silently(0, 11)
        window.draw_frame()
        assert _pixel(window, 2, 1) == (255, 255, 255)
        assert _pixel(window, 0, 0) == (23, 23, 23)
        window.lifetime_action.setChecked(True)
        assert window.lifetime_button.isChecked()
        # the first time, the colors span the lifetimes of the frame
        dialog = window.lifetime_dialog
        assert dialog.range_known
        shortest = dialog.shortest_spinbox.value()
        longest = dialog.longest_spinbox.value()
        assert (shortest, longest) == pytest.approx((0.3, 0.4))
        lut = gui_localize._lifetime_lut(gui_localize.LIFETIME_COLORMAP)
        index = int((LIFETIME - shortest) / (longest - shortest) * 255 + 0.5)
        assert _pixel(window, 2, 1) == tuple(int(c) for c in lut[index])
        # no lifetime: gray, as bright as the intensity
        assert _pixel(window, 0, 0) == (23, 23, 23)
        window.lifetime_button.setChecked(False)
        assert not window.lifetime_action.isChecked()
        assert _pixel(window, 2, 1) == (255, 255, 255)

    def test_a_remembered_range_is_kept(self, window, scan):
        dialog = window.lifetime_dialog
        dialog.set_range(1.0, 2.0)
        window.set_lifetime_shown(True)
        assert dialog.shortest_spinbox.value() == 1.0
        assert dialog.longest_spinbox.value() == 2.0
        dialog.auto_button.click()
        assert dialog.shortest_spinbox.value() == pytest.approx(0.3)
        assert dialog.longest_spinbox.value() == pytest.approx(0.4)

    def test_photon_delays_are_read_once_per_frame(
        self, window, scan, monkeypatch
    ):
        calls = []
        original = io.PTUMovie._delay_sums

        def counted(self, index, dtime_range):
            calls.append((index, dtime_range))
            return original(self, index, dtime_range)

        monkeypatch.setattr(io.PTUMovie, "_delay_sums", counted)
        window.set_lifetime_shown(True)
        window.draw_frame()
        window.set_frame(1)
        window.set_frame(0)
        assert calls == [(0, (2, 16)), (1, (2, 16))]
        # the same photons, counted differently
        window.lifetime_dialog.min_photons_spinbox.setValue(1)
        window.lifetime_dialog.binning_spinbox.setValue(3)
        assert len(calls) == 2
        # other photons
        window.lifetime_dialog.start_spinbox.setValue(0.2)
        assert calls[-1] == (0, (7, 16))

    def test_a_window_without_photons_shows_why(self, window, scan):
        window.set_lifetime_shown(True)
        assert window._lifetime_error is None
        window.lifetime_dialog.start_spinbox.setValue(5.0)
        assert "outside the photons' microtimes" in window._lifetime_error
        assert _pixel(window, 2, 1) == (255, 255, 255)  # intensity
        image = QtGui.QImage(400, 300, QtGui.QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QtGui.QPainter(image)
        window.paint_lifetime_legend(painter)
        painter.end()

    def test_the_color_bar_is_painted_over_the_view(self, window, scan):
        def painted():
            image = QtGui.QImage(
                window.view.viewport().size(),
                QtGui.QImage.Format.Format_ARGB32,
            )
            image.fill(0)
            painter = QtGui.QPainter(image)
            window.paint_lifetime_legend(painter)
            painter.end()
            return image.pixelColor(image.width() - 12, 12).alpha()

        assert painted() == 0
        window.set_lifetime_shown(True)
        assert painted() > 0

    def test_the_dialog_plots_the_decay_and_dragging_sets_the_window(
        self, window, scan
    ):
        dialog = window.lifetime_dialog
        dialog.show()
        times, counts, maximum = dialog._decay
        assert maximum == pytest.approx(0.08)
        assert counts[2] == 4 * 48
        assert times[2] == pytest.approx(0.1)
        dialog.on_span_select(0.2, 0.6)
        assert dialog.start_spinbox.value() == pytest.approx(0.12)
        assert dialog.length_spinbox.value() == pytest.approx(0.4)
        assert window.lifetime_window_ns() == pytest.approx((0.2, 0.6))
        dialog.close()

    def test_the_dialog_without_a_photon_counting_movie(self, window):
        dialog = window.lifetime_dialog
        dialog.show()
        assert dialog._decay is None
        assert dialog._decay_error is None
        dialog.close()

    def test_settings_are_remembered(self, window):
        dialog = window.lifetime_dialog
        dialog.start_spinbox.setValue(0.3)
        dialog.length_spinbox.setValue(4.0)
        dialog.binning_spinbox.setValue(3)
        dialog.min_photons_spinbox.setValue(10)
        dialog.colormap_combo.setCurrentText("viridis")
        dialog.set_range(1.0, 3.5)
        window.close()
        stored = io.load_user_settings()["Localize"]["Lifetime display"]
        assert stored == {
            "start": 0.3,
            "length": 4.0,
            "binning": 3,
            "min_photons": 10,
            "colormap": "viridis",
            "shortest": 1.0,
            "longest": 3.5,
        }
        other = gui_localize.Window()
        try:
            assert other.lifetime_dialog.settings() == stored
            assert other.lifetime_dialog.range_known
        finally:
            other.close()

    def test_an_unknown_range_is_not_remembered(self, window):
        window.close()
        stored = io.load_user_settings()["Localize"]["Lifetime display"]
        assert "shortest" not in stored
        other = gui_localize.Window()
        try:
            assert not other.lifetime_dialog.range_known
        finally:
            other.close()


class TestLifetimeImage:
    def test_lifetimes_are_mean_delays_of_enough_photons(self):
        photons = np.zeros((5, 5))
        delays = np.zeros((5, 5))
        photons[2, 2], delays[2, 2] = 3, 6.0
        photons[2, 3], delays[2, 3] = 3, 12.0
        lifetimes, counted = gui_localize._lifetime_image(
            photons, delays, min_photons=3
        )
        assert lifetimes.dtype == np.float32
        assert lifetimes[2, 2] == 2.0 and lifetimes[2, 3] == 4.0
        assert np.isnan(lifetimes).sum() == 23
        np.testing.assert_array_equal(counted, photons)

    def test_binning_pools_the_photons_of_neighbors(self):
        photons = np.zeros((5, 5))
        delays = np.zeros((5, 5))
        photons[2, 2], delays[2, 2] = 3, 6.0
        photons[2, 3], delays[2, 3] = 3, 12.0
        lifetimes, counted = gui_localize._lifetime_image(
            photons, delays, binning=3, min_photons=6
        )
        assert counted[2, 2] == counted[1, 3] == 6
        assert lifetimes[2, 2] == lifetimes[1, 3] == 3.0
        assert counted[2, 4] == 3 and np.isnan(lifetimes[2, 4])
        assert np.isfinite(lifetimes).sum() == 6

    def test_colors_show_lifetimes_and_brightness_the_intensity(self):
        lut = np.zeros((256, 3), np.uint8)
        lut[:, 0] = np.arange(256)
        lifetimes = np.array([[np.nan, 1.0, 3.0, 2.0]], np.float32)
        brightness = np.array([[255, 255, 128, 255]], np.uint8)
        rgb = gui_localize._lifetime_rgb(lifetimes, brightness, lut, 1.0, 3.0)
        assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous
        assert rgb[0].tolist() == [
            [255, 255, 255],  # no lifetime: gray
            [0, 0, 0],  # the first color
            [128, 0, 0],  # the last color, half as bright
            [128, 0, 0],  # halfway
        ]

    def test_the_range_covers_most_photons(self):
        lifetimes = np.array([1.0, 2.0, 3.0, np.nan])
        photons = np.array([1.0, 98.0, 1.0, 50.0])
        assert gui_localize._lifetime_range(lifetimes, photons) == (2.0, 2.1)
        spread = np.linspace(0.51, 3.49, 100)
        assert gui_localize._lifetime_range(spread, np.ones(100)) == (
            0.5,
            3.5,
        )
        assert (
            gui_localize._lifetime_range(np.full(3, np.nan), photons[:3])
            is None
        )
