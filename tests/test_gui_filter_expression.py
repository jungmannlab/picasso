"""Filtering by conditions in the Filter window.

Covers ``Window.apply_expression`` on the shown localizations, the
"Filter by expression" dialog (live syntax check, result, errors), the
ordered filter steps written to the metadata, also by the histogram
and numeric filters, and replaying them from the metadata in the
window (menu or a dropped .yaml) and with ``lib.apply_filter_steps``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io, lib
from picasso.gui import filter as gui_filter

LOCS_PATH = "./tests/data/testdata_locs.hdf5"


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """A Filter window with the test localizations (pixel size 130 nm)
    open; user settings go to ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    window = gui_filter.Window()
    window.open(LOCS_PATH)
    yield window
    window.filter_expression.close()
    window.close()


@pytest.fixture
def locs():
    locs, _ = io.load_locs(LOCS_PATH)
    return locs


def _shown(window):
    return window.materialize_filtered()


def test_apply_expression(window, locs):
    window.apply_expression("photons > median(photons) and sx < 1.2")
    median = np.median(locs["photons"].astype(np.float64))
    expected = locs[(locs["photons"] > median) & (locs["sx"] < 1.2)]
    assert window.n_filtered == len(expected) > 0
    np.testing.assert_array_equal(_shown(window)["frame"], expected["frame"])
    assert window.filter_log[lib.FILTER_STEPS_KEY] == [
        "photons > median(photons) and sx < 1.2"
    ]


def test_statistics_use_the_shown_localizations(window, locs):
    window.apply_range("sx", 0.9, 1.1, inclusive=True)
    shown = _shown(window)
    window.apply_expression("photons > mean(photons)")
    mean = shown["photons"].astype(np.float64).mean()
    expected = shown[shown["photons"] > mean]
    np.testing.assert_array_equal(_shown(window)["frame"], expected["frame"])


def test_pixel_size_from_metadata(window, locs):
    window.apply_expression("lpx * pixelsize < 10")
    assert window.n_filtered == int((locs["lpx"] * 130 < 10).sum())


def test_range_conditions_are_logged_as_ranges(window):
    window.apply_expression("0.9 < sx < 1.1")
    window.apply_expression("sx < 1.05")
    assert window.filter_log["sx"] == [0.9, 1.1]
    assert window.filter_log[lib.FILTER_STEPS_KEY] == [
        "0.9 < sx < 1.1",
        "sx < 1.05",
    ]


def test_range_filters_record_steps(window):
    window.apply_range("photons", 500, 5000)
    window.apply_range("sx", 0.8, 1.2, inclusive=True)
    window.apply_range2d("bg", 0, 1e4, "lpx", 0, 0.5)
    assert window.filter_log[lib.FILTER_STEPS_KEY] == [
        "500.0 < photons < 5000.0",
        "0.8 <= sx <= 1.2",
        "0.0 < bg < 10000.0",
        "0.0 < lpx < 0.5",
    ]


def test_invalid_expression_filters_nothing(window):
    n = window.n_filtered
    with pytest.raises(gui_filter.expression.ExpressionError):
        window.apply_expression("photons > 500 & sx < 1.2")
    assert window.n_filtered == n
    assert window.filter_log[lib.FILTER_STEPS_KEY] == []


def test_dialog(window, locs):
    dialog = window.filter_expression
    assert "photons" in dialog.columns_label.text()
    assert "pixelsize" in dialog.columns_label.text()

    dialog.expression.setText("phtons > 1")
    assert dialog.status.text() == "Unknown name(s): phtons."
    dialog.expression.setText("photons >")
    assert dialog.status.text().startswith("Invalid syntax")
    dialog.expression.setText("photons > 1000")
    assert dialog.status.text() == ""

    dialog.filter()
    kept = int((locs["photons"] > 1000).sum())
    assert dialog.status.text() == (
        f"Kept {kept:,} of {len(locs):,} localizations."
    )
    assert window.n_filtered == kept

    dialog.expression.setText("mean(photons) > 1")
    dialog.filter()
    assert "same for every localization" in dialog.status.text()
    assert window.n_filtered == kept


def test_dialog_without_data(qt_offscreen, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda parent, title, text: shown.append(title)),
    )
    window = gui_filter.Window()
    window.filter_expression.expression.setText("photons > 1")
    assert window.filter_expression.status.text() == ""
    window.filter_expression.filter()
    assert shown == ["Filter by expression"]
    window.close()


def test_removed_columns_leave_the_dialog(window, monkeypatch):
    monkeypatch.setattr(
        lib.RemoveColumnsDialog,
        "getParams",
        staticmethod(lambda parent, columns: (["likelihood"], True)),
    )
    window.remove_columns()
    assert "likelihood" not in window.filter_expression.columns_label.text()


def _save(window, path, monkeypatch):
    monkeypatch.setattr(
        lib,
        "get_save_filename_ext_dialog",
        lambda *args, **kwargs: (str(path), None),
    )
    window.save_file_dialog()
    return io.load_locs(str(path))


def test_saved_steps_replay_to_the_same_result(
    window, locs, tmp_path, monkeypatch
):
    window.apply_range("sx", 0.8, 1.2, inclusive=True)
    window.apply_expression("photons > percentile(photons, 30)")
    window.apply_range("lpx", 0.0, 0.3)
    saved, info = _save(window, tmp_path / "f.hdf5", monkeypatch)
    assert info[-1][lib.FILTER_STEPS_KEY] == [
        "0.8 <= sx <= 1.2",
        "photons > percentile(photons, 30)",
        "0.0 < lpx < 0.3",
    ]
    assert info[-1]["sx"] == [0.8, 1.2]
    replayed, *_ = lib.apply_filter_steps(locs, info)
    np.testing.assert_array_equal(replayed["frame"], saved["frame"])
    np.testing.assert_array_equal(replayed["photons"], saved["photons"])


def test_unfiltered_save_has_no_steps(window, tmp_path, monkeypatch):
    _, info = _save(window, tmp_path / "f.hdf5", monkeypatch)
    assert lib.FILTER_STEPS_KEY not in info[-1]


def test_apply_filters_from_metadata(window, locs, monkeypatch):
    steps = ["sx < 1.1", "photons < median(photons)", "z > 0"]
    info = [{"Generated by": "Picasso Filter", "Filter steps": steps}]
    monkeypatch.setattr(
        window, "_load_filter_metadata", lambda path=None: info
    )
    questions = []

    def question(parent, title, text, *args):
        questions.append(text)
        return QtWidgets.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(question)
    )
    window.apply_filters_from_metadata()
    assert "  sx < 1.1\n  photons < median(photons)\n" in questions[0]
    assert "will be skipped):\n  z" in questions[0]
    expected = locs[locs["sx"] < 1.1]
    median = expected["photons"].astype(np.float64).median()
    expected = expected[expected["photons"] < median]
    np.testing.assert_array_equal(_shown(window)["frame"], expected["frame"])
    assert window.filter_log[lib.FILTER_STEPS_KEY] == steps[:2]


def test_apply_filters_from_old_metadata(window, locs, monkeypatch):
    info = [{"Generated by": "Picasso v0.10 Filter", "sx": [0.9, 1.1]}]
    monkeypatch.setattr(
        window, "_load_filter_metadata", lambda path=None: info
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *args: QtWidgets.QMessageBox.StandardButton.Yes),
    )
    window.apply_filters_from_metadata()
    assert window.n_filtered == int(
        ((locs["sx"] > 0.9) & (locs["sx"] < 1.1)).sum()
    )
    assert window.filter_log["sx"] == [0.9, 1.1]
    assert window.filter_log[lib.FILTER_STEPS_KEY] == ["0.9 < sx < 1.1"]


def test_failing_step_stops_the_replay(window, monkeypatch):
    info = [
        {
            "Generated by": "Picasso Filter",
            "Filter steps": ["sx < 1.1", "mean(sx) > 0", "photons > 1"],
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
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        staticmethod(lambda parent, title, text: warnings.append(text)),
    )
    window.apply_filters_from_metadata()
    assert "mean(sx) > 0" in warnings[0]
    assert window.filter_log[lib.FILTER_STEPS_KEY] == ["sx < 1.1"]


def test_drop_yaml_applies_its_filters(window, locs, tmp_path, monkeypatch):
    path = tmp_path / "other_locs_filter.yaml"
    info = [{"Generated by": "Picasso Filter", "Filter steps": ["sx < 1.1"]}]
    io.save_info(str(path), info)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *args: QtWidgets.QMessageBox.StandardButton.Yes),
    )
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(path))])
    event = QtGui.QDropEvent(
        QtCore.QPointF(0, 0),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    window.table_view.dropEvent(event)
    assert window.n_filtered == int((locs["sx"] < 1.1).sum())
    assert window.filter_log[lib.FILTER_STEPS_KEY] == ["sx < 1.1"]
