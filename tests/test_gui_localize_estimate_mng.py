"""Localize's 'Estimate' button for the minimum net gradient.

``localize.estimate_min_net_gradient`` makes the estimate; the window
decides what it searches - the identification movie, the ROIs, the
selected split-FOV region and the frame range - and puts the value on the
spin box, from which it reaches the slider and the split-FOV regions as a
typed value does.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PyQt6 import QtWidgets

from picasso import lib_qt, localize
from picasso.gui import localize as gui_localize
from tests.test_gui_localize_calibration import _load, _Messages
from tests.test_localize import _noisy_spot_movie

TITLE = "Estimate min. net gradient"
REGIONS = [[[0, 0], [64, 32]], [[0, 32], [64, 64]]]
MOVIE = _noisy_spot_movie(40, 8)


def _wait_for_tasks(timeout: float = 60.0) -> None:
    start = time.monotonic()
    while lib_qt._running_tasks:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not end")
    QtWidgets.QApplication.processEvents()


class _FakeEstimate:
    """Records the calls to ``localize.estimate_min_net_gradient`` and
    returns a fixed estimate (or raises ``error``)."""

    def __init__(self, monkeypatch, value=4321.4, error=None):
        self.calls = []
        self.value = value
        self.error = error
        monkeypatch.setattr(localize, "estimate_min_net_gradient", self)

    def __call__(self, movie, **kwargs):
        self.calls.append((movie, kwargs))
        if self.error is not None:
            raise self.error
        return {
            "min_net_gradient": self.value,
            "background": 100.0,
            "background_width": 50.0,
            "frames": np.arange(40),
        }


def _estimate(window):
    window.parameters_dialog.mng_estimate_button.click()
    _wait_for_tasks()


@pytest.fixture
def window(qt_offscreen):
    window = gui_localize.Window()
    yield window
    window.close()


@pytest.fixture
def loaded(window, tmp_path):
    _load(window, [MOVIE], tmp_path)
    window.parameters_dialog.mng_slider.setValue(2000)
    return window


class TestEstimateButton:
    def test_the_estimate_is_put_on_the_spin_box(self, loaded):
        pdialog = loaded.parameters_dialog
        assert loaded.parameters["Temporal Median Window"] == 0
        assert loaded.parameters["Gaussian Filter Sigma"] == 0
        expected = localize.estimate_min_net_gradient(
            MOVIE, box=pdialog.box_spinbox.value()
        )
        value = round(expected["min_net_gradient"])
        _estimate(loaded)
        assert pdialog.mng_spinbox.value() == value
        assert pdialog.mng_slider.value() == value
        assert loaded.parameters["Min. Net Gradient"] == value
        assert loaded.status_bar.currentMessage().startswith(
            f"Min. net gradient estimated: {value:,} "
        )

    def test_the_box_the_rois_and_the_frame_range_are_used(
        self, loaded, monkeypatch
    ):
        fake = _FakeEstimate(monkeypatch)
        loaded.parameters_dialog.box_spinbox.setValue(9)
        loaded.view.rois = [[[0, 0], [20, 20]], [[30, 30], [60, 60]]]
        loaded.frame_range = [[5, 30]]
        _estimate(loaded)
        ((_, kwargs),) = fake.calls
        assert kwargs["box"] == 9
        assert kwargs["roi"] == loaded.view.rois
        assert kwargs["frame_bounds"] == [[5, 30]]
        assert loaded.parameters_dialog.mng_spinbox.value() == 4321

    def test_the_filtered_movie_is_searched(self, loaded, monkeypatch):
        fake = _FakeEstimate(monkeypatch)
        loaded.parameters_dialog.gaussian_filter_spinbox.setValue(1.0)
        _estimate(loaded)
        ((movie, _),) = fake.calls
        filtered = np.asarray(loaded.identification_movie()[3])
        np.testing.assert_array_equal(np.asarray(movie[3]), filtered)
        assert not np.array_equal(filtered, MOVIE[3])

    def test_a_negative_estimate_is_put_on_the_spin_box_as_zero(
        self, loaded, monkeypatch
    ):
        _FakeEstimate(monkeypatch, value=-250.0)
        _estimate(loaded)
        assert loaded.parameters_dialog.mng_spinbox.value() == 0

    def test_without_a_movie_nothing_is_estimated(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        fake = _FakeEstimate(monkeypatch)
        _estimate(window)
        assert messages.shown == [("information", TITLE, "No movie loaded.")]
        assert fake.calls == []

    def test_an_error_is_shown_and_the_value_kept(self, loaded, monkeypatch):
        messages = _Messages(monkeypatch)
        _FakeEstimate(monkeypatch, error=ValueError("No local maxima."))
        _estimate(loaded)
        assert messages.shown == [("warning", TITLE, "No local maxima.")]
        assert loaded.parameters_dialog.mng_spinbox.value() == 2000


class TestSplitFov:
    @pytest.fixture
    def split(self, loaded):
        loaded.parameters_dialog.split_fov_checkbox.setChecked(True)
        loaded.view.rois = [list(map(list, r)) for r in REGIONS]
        assert loaded.region_mngs() == [2000, 2000]
        return loaded

    def test_the_selected_region_is_searched_and_set_alone(
        self, split, monkeypatch
    ):
        fake = _FakeEstimate(monkeypatch)
        split.view.selected_roi = 1
        split.parameters_dialog.sync_mng_to_selected_region()
        _estimate(split)
        ((_, kwargs),) = fake.calls
        assert kwargs["roi"] == [REGIONS[1]]
        assert split.region_mngs() == [2000, 4321]

    def test_with_none_selected_every_region_is_searched_and_set(
        self, split, monkeypatch
    ):
        fake = _FakeEstimate(monkeypatch)
        _estimate(split)
        ((_, kwargs),) = fake.calls
        assert kwargs["roi"] == REGIONS
        assert split.region_mngs() == [4321, 4321]
