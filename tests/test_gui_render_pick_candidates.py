"""Render's Tools > Pick by localizations and RMSD...
(``picasso.gui.render.PickCandidatesWindow``).

``postprocess.pick_similar_candidates`` and ``structure_classes`` do the
analysis; the window finds the candidates of a channel with the pick
diameter of the Tools settings, keeps the ranges of the spin boxes, the
rectangle drawn on the histogram and the classes in step, and adds the
candidates within the ranges as circular picks, in one undo step. The
bundled localizations hold nine DNA origami.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time
import types

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import lib_qt, postprocess
from picasso.gui import render as gui_render
from tests.conftest import PIXELSIZE

PICK_SIZE = 1.5 * PIXELSIZE  # nm


def _wait_for_tasks(timeout: float = 60.0) -> None:
    start = time.monotonic()
    while lib_qt._running_tasks:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not end")
    QtWidgets.QApplication.processEvents()


@pytest.fixture
def window(qt_offscreen, tmp_path, locs, info):
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(str(tmp_path / "locs.hdf5"), locs, info, render_=False)
    width, height = (
        info[0]["Width"] * PIXELSIZE,
        info[0]["Height"] * PIXELSIZE,
    )
    window.view.viewport = [(0.0, 0.0), (height, width)]
    window.tools_settings_dialog.pick_diameter.setValue(PICK_SIZE)
    yield window
    _wait_for_tasks()
    window.close()


@pytest.fixture
def opened(window):
    """The window of the tool, with the candidates found."""
    window.view.pick_by_n_locs_rmsd()
    _wait_for_tasks()
    return window.view.pick_candidates_window


class TestPickCandidatesWindow:
    def test_the_menu_opens_it(self, window):
        tools = next(
            a.menu() for a in window.menu_bar.actions() if a.text() == "Tools"
        )
        action = next(
            a
            for a in tools.actions()
            if a.text() == f"{gui_render.PickCandidatesWindow.TITLE}..."
        )
        action.trigger()
        _wait_for_tasks()
        assert window.view.pick_candidates_window.isVisible()

    def test_the_candidates_are_found_with_the_tools_pick_diameter(
        self, opened, locs, info
    ):
        expected = postprocess.pick_similar_candidates(
            locs,
            info,
            pick_size=PICK_SIZE,
            min_n_locs=opened.min_n_locs.value(),
        )
        pd.testing.assert_frame_equal(opened.candidates, expected)
        assert opened.diameter == PICK_SIZE
        assert f"{PICK_SIZE:g} nm" in opened.diameter_label.text()
        # the ranges span every candidate
        assert len(opened.selected()) == len(expected) == 9
        assert opened.status_label.text() == "9 of 9 candidates in range"

    def test_the_min_n_locs_is_a_hundredth_of_the_frames(self, opened, info):
        frames = info[0]["Frames"]
        assert opened.min_n_locs.value() == max(2, frames // 100)

    def test_the_ranges_select_the_candidates(self, opened):
        candidates = opened.candidates
        low = int(candidates["n_locs"].median())
        opened.n_locs_min.setValue(low)
        selected = opened.selected()
        assert len(selected) == (candidates["n_locs"] >= low).sum()
        assert opened.status_label.text().startswith(f"{len(selected)} of 9")

    def test_a_rectangle_on_the_histogram_sets_the_ranges(self, opened):
        press = types.SimpleNamespace(xdata=60.4, ydata=25.04)
        release = types.SimpleNamespace(xdata=40.6, ydata=20.06)
        opened.on_rect_select(press, release)
        # the integers within, and the RMSD widened to the decimal shown
        assert opened.ranges() == ((41, 60), (20.0, 25.1))
        patch = opened.range_patch
        assert (patch.get_x(), patch.get_y()) == (41, 20.0)

    def test_the_picks_are_added_in_one_undo_step(self, window, opened):
        opened.add_button.click()
        view = window.view
        assert len(view._picks) == 9
        assert set(view._pick_shapes) == {"Circle"}
        assert set(view._pick_sizes) == {PICK_SIZE}
        np.testing.assert_allclose(
            np.array(view._picks),
            opened.candidates[["x", "y"]].to_numpy(),
        )
        assert opened.status_label.text() == "Added 9 picks."
        view.undo()
        assert view._picks == []

    def test_classes_set_the_ranges(self, opened, monkeypatch):
        classes = pd.DataFrame(
            {
                "n_candidates": [5, 4],
                "n_locs": [60.0, 40.0],
                "rmsd": [21.0, 23.0],
                "n_locs_min": [50.2, 30.0],
                "n_locs_max": [70.9, 49.0],
                "rmsd_min": [20.04, 22.0],
                "rmsd_max": [22.01, 24.0],
            }
        )
        labels = np.zeros(len(opened.candidates), dtype=np.int64)
        monkeypatch.setattr(
            postprocess,
            "structure_classes",
            lambda candidates: (labels, classes),
        )
        opened.classes_button.click()
        assert opened.class_combo.count() == 2
        assert opened.class_combo.itemText(0) == (
            "Class 1: 5 candidates, 60 localizations, 21.0 nm"
        )
        # the first is selected
        assert opened.ranges() == ((51, 70), (20.0, 22.1))
        opened.class_combo.setCurrentIndex(1)
        opened.class_combo.activated.emit(1)
        assert opened.ranges() == ((30, 49), (22.0, 24.0))

    def test_too_few_candidates_for_classes(self, opened):
        opened.set_candidates(opened.candidates.iloc[:1], PICK_SIZE)
        opened.classes_button.click()
        assert opened.status_label.text() == "Too few candidates for classes."

    def test_without_candidates_nothing_can_be_added(self, window, opened):
        opened.min_n_locs.setValue(10**6)
        opened.find_button.click()
        _wait_for_tasks()
        assert len(opened.candidates) == 0
        assert not opened.add_button.isEnabled()
        assert not opened.classes_button.isEnabled()
        assert window.view._picks == []

    def test_reopening_replaces_the_window(self, window, opened):
        window.view.pick_by_n_locs_rmsd()
        _wait_for_tasks()
        assert window.view.pick_candidates_window is not opened
        assert not opened.isVisible()

    def test_a_style_change_redraws_with_the_ranges(self, opened):
        opened.set_ranges((40, 60), (20, 25))
        opened._on_style_changed(opened.plot_style)
        patch = opened.range_patch
        assert (patch.get_x(), patch.get_width()) == (40, 20)
