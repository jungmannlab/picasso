"""Undo and Redo in the Filter window.

Covers taking back the filters (ranges, 2D ranges, expressions, the
filters replayed from metadata) and the removal of columns, redoing
them, the limit of steps, the actions and their tooltips, and undoing
after the localizations were moved to disk while idle.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import expression, io, lib
from picasso.gui import filter as gui_filter
from picasso.gui import idle

LOCS_PATH = "./tests/data/testdata_locs.hdf5"


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """A Filter window with the test localizations open; user settings
    go to ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    window = gui_filter.Window()
    window.open(LOCS_PATH)
    yield window
    window._discard_spill()
    window.close()


@pytest.fixture
def locs():
    locs, _ = io.load_locs(LOCS_PATH)
    return locs


def _steps(window) -> list[str]:
    return window.filter_log[lib.FILTER_STEPS_KEY]


def _remove(window, monkeypatch, columns) -> None:
    monkeypatch.setattr(
        lib.RemoveColumnsDialog,
        "getParams",
        staticmethod(lambda parent, names: (list(columns), True)),
    )
    window.remove_columns()


def test_nothing_to_undo_after_opening(window):
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()
    window.undo()  # does nothing
    window.redo()
    assert window.filtered_idx is None


def test_undo_and_redo_a_range(window, locs):
    window.apply_range("sx", 107.0, 110.0, inclusive=True)
    filtered = window.materialize_filtered()
    log = {"sx": list(window.filter_log["sx"]), "steps": list(_steps(window))}
    assert window.undo_action.isEnabled()

    window.undo()
    assert window.filtered_idx is None
    assert window.n_filtered == len(locs)
    assert window.filter_log["sx"] is None
    assert _steps(window) == []
    assert not window.undo_action.isEnabled()
    assert window.redo_action.isEnabled()

    window.redo()
    pd.testing.assert_frame_equal(window.materialize_filtered(), filtered)
    assert window.filter_log["sx"] == log["sx"]
    assert _steps(window) == log["steps"]
    assert not window.redo_action.isEnabled()


def test_steps_are_undone_one_at_a_time(window):
    window.apply_range("sx", 107.0, 110.0, inclusive=True)
    after_range = window.materialize_filtered()
    window.apply_range2d("photons", 5000.0, 30000.0, "sy", 100.0, 115.0)
    after_2d = window.materialize_filtered()
    window.apply_expression("photons > median(photons)")
    assert len(window._undo_stack) == 3

    window.undo()
    pd.testing.assert_frame_equal(window.materialize_filtered(), after_2d)
    assert len(_steps(window)) == 3  # the 2D range is two steps
    window.undo()
    pd.testing.assert_frame_equal(window.materialize_filtered(), after_range)
    assert _steps(window) == ["107.0 <= sx <= 110.0"]
    assert window.filter_log["photons"] is None
    window.redo()
    pd.testing.assert_frame_equal(window.materialize_filtered(), after_2d)


def test_a_new_filter_drops_the_steps_undone(window):
    window.apply_range("sx", 107.0, 110.0, inclusive=True)
    window.undo()
    window.apply_expression("photons > 10000")
    assert not window.redo_action.isEnabled()
    assert window._redo_stack == []
    assert _steps(window) == ["photons > 10000"]


def test_an_invalid_expression_records_nothing(window):
    with pytest.raises(expression.ExpressionError):
        window.apply_expression("photons >")
    assert window._undo_stack == []


def test_the_rows_are_stored_compactly():
    n = 1000
    many = np.arange(0, n, 2, dtype=np.uint32)
    packed = gui_filter._pack_rows(many, n)
    assert packed.dtype == np.uint8 and packed.nbytes == n // 8
    restored = gui_filter._unpack_rows(packed, n, np.uint32)
    assert restored.dtype == np.uint32
    np.testing.assert_array_equal(restored, many)
    few = np.array([3, 500], dtype=np.uint32)
    assert gui_filter._pack_rows(few, n) is few
    assert gui_filter._unpack_rows(few, n, np.uint32) is few
    assert gui_filter._pack_rows(None, n) is None
    assert gui_filter._unpack_rows(None, n, np.uint32) is None


def test_undo_a_filter_that_keeps_few_rows(window):
    window.apply_range("photons", 2000.0, 3500.0)  # a few rows
    few = window.filtered_idx
    window.apply_range("sx", 0.0, 200.0)
    window.undo()
    np.testing.assert_array_equal(window.filtered_idx, few)


def test_undo_the_removal_of_columns(window, locs, monkeypatch):
    columns = list(window.locs_full.columns)
    window.apply_range("sx", 107.0, 110.0, inclusive=True)
    filtered = window.materialize_filtered()
    _remove(window, monkeypatch, ["photons", "log_likelihood"])
    assert "photons" not in window.locs_full.columns
    assert window.filter_log["Removed columns"] == [
        "photons",
        "log_likelihood",
    ]

    window.undo()
    assert list(window.locs_full.columns) == columns
    pd.testing.assert_frame_equal(window.materialize_filtered(), filtered)
    assert "Removed columns" not in window.filter_log
    assert window.filter_num.attributes.findText("photons") >= 0
    assert "photons" in window.filter_expression.columns_label.text()

    window.redo()
    assert "photons" not in window.locs_full.columns
    assert "log_likelihood" not in window.locs_full.columns
    assert window.filter_num.attributes.findText("photons") == -1
    window.undo()
    window.undo()  # and the range before it
    pd.testing.assert_frame_equal(
        window.materialize_filtered(), locs.reset_index(drop=True)
    )


def test_removing_a_column_closes_its_histograms(window, monkeypatch):
    window.hist_windows["photons"] = gui_filter.HistWindow(window, "photons")
    window.hist2d_windows["sx"]["photons"] = gui_filter.Hist2DWindow(
        window, "sx", "photons"
    )
    window.hist_windows["sx"] = gui_filter.HistWindow(window, "sx")
    _remove(window, monkeypatch, ["photons"])
    assert window.hist_windows["photons"] is None
    assert window.hist2d_windows["sx"]["photons"] is None
    assert window.hist_windows["sx"] is not None
    window.apply_range("sx", 107.0, 110.0)  # refreshes the histograms
    window.undo()
    window.undo()
    assert "photons" in window.locs_full.columns
    window.hist_windows["sx"].close()


def test_filters_from_metadata_are_one_step(window, locs, monkeypatch):
    steps = ["sx < 110", "photons < median(photons)"]
    info = [
        {
            "Generated by": "Picasso Filter",
            "Filter steps": steps,
            "Removed columns": ["log_likelihood"],
        }
    ]
    monkeypatch.setattr(
        window, "_load_filter_metadata", lambda path=None: info
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *args: QtWidgets.QMessageBox.StandardButton.Yes),
    )
    window.apply_filters_from_metadata()
    assert _steps(window) == steps
    assert "log_likelihood" not in window.locs_full.columns
    assert len(window._undo_stack) == 1
    assert "sx < 110 and photons" in window.undo_action.toolTip()

    window.undo()
    assert window.filtered_idx is None
    assert _steps(window) == []
    assert list(window.locs_full.columns) == list(locs.columns)


def test_a_failed_replay_undoes_the_steps_applied(window, monkeypatch):
    info = [{"Generated by": "Picasso Filter", "Filter steps": ["sx < 110"]}]
    monkeypatch.setattr(
        window, "_load_filter_metadata", lambda path=None: info
    )
    monkeypatch.setattr(
        lib,
        "extract_filter_recipe",
        lambda info, columns: (["sx < 110", "sx >"], [], []),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *args: QtWidgets.QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: None)
    window.apply_filters_from_metadata()
    assert _steps(window) == ["sx < 110"]
    assert len(window._undo_stack) == 1
    window.undo()
    assert window.filtered_idx is None


def test_tooltips_name_the_step(window):
    assert window.undo_action.toolTip().startswith("Undo")
    window.apply_expression("photons > 10000")
    assert window.undo_action.toolTip().startswith("Undo: photons > 10000")
    window.undo()
    assert window.redo_action.toolTip().startswith("Redo: photons > 10000")
    assert not window.undo_action.toolTip().startswith("Undo:")


def test_a_long_step_is_shortened(window):
    condition = " and ".join(["photons > 0"] * 10)
    window.apply_expression(condition)
    label = window._undo_stack[-1].label
    assert len(label) == 60 and label.endswith("…")


def test_the_oldest_steps_are_forgotten(window, monkeypatch):
    monkeypatch.setattr(gui_filter, "UNDO_LIMIT", 3)
    _remove(window, monkeypatch, ["log_likelihood"])
    for xmax in (120.0, 115.0, 112.0):
        window.apply_range("sx", 0.0, xmax)
    assert len(window._undo_stack) == 3
    # no state left has the column, so it is not kept any more
    assert "log_likelihood" not in window._removed_columns
    for _ in range(5):
        window.undo()
    assert _steps(window) == []
    assert "Removed columns" in window.filter_log


def test_opening_a_file_forgets_the_steps(window, monkeypatch):
    _remove(window, monkeypatch, ["log_likelihood"])
    window.apply_range("sx", 107.0, 110.0)
    window.undo()
    window.open(LOCS_PATH)
    assert window._undo_stack == window._redo_stack == []
    assert window._removed_columns == {}
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_undo_after_the_localizations_moved_to_disk(
    window, locs, tmp_path, monkeypatch
):
    folder = tmp_path / "spill"
    folder.mkdir()
    monkeypatch.setattr(io, "spill_directory", lambda: str(folder))
    monkeypatch.setattr(idle, "SPILL_MIN_BYTES", 0)
    window.apply_range("sx", 107.0, 110.0, inclusive=True)
    assert window.release_idle_memory()
    assert window._spilled is not None
    window.undo()
    assert window._spilled is None
    pd.testing.assert_frame_equal(
        window.materialize_filtered(), locs.reset_index(drop=True)
    )
