"""The windows show why a file could not be opened, from the message of
the error ``picasso.io`` raises (``lib_qt.show_load_error``), and ask
before overwriting files (``lib_qt.get_save_filename_ext_dialog``).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtWidgets

from picasso import lib_qt


@pytest.fixture
def shown(monkeypatch):
    """The (title, text) of every critical message box."""
    boxes = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        staticmethod(lambda parent, title, text: boxes.append((title, text))),
    )
    return boxes


def test_message_names_the_file_and_the_reason(qapp, shown):
    lib_qt.show_load_error(None, "/data/movie_locs.hdf5", ValueError("bad"))
    assert shown == [
        (
            "Could not open the file",
            "movie_locs.hdf5 could not be opened:\n\nbad",
        )
    ]


def test_key_error_without_quotes(qapp, shown):
    error = KeyError("File: x.hdf5 does not contain a 'locs' dataset.")
    lib_qt.show_load_error(None, "x.hdf5", error)
    assert shown[0][1].endswith("does not contain a 'locs' dataset.")


def test_filter_shows_missing_metadata(qt_offscreen, tmp_path, shown):
    import numpy as np
    import pandas as pd

    from picasso.gui import filter as gui_filter

    path = str(tmp_path / "locs.hdf5")
    pd.DataFrame({"x": np.zeros(3)}).to_hdf(path, key="locs")
    window = gui_filter.Window()
    try:
        window.open(path)
    finally:
        window.close()
    assert len(shown) == 1
    assert "Could not find metadata" in shown[0][1]


@pytest.mark.parametrize("answers,expected", [([True], "out"), ([False], "")])
def test_overwrite_is_asked_per_existing_file(
    qapp, tmp_path, monkeypatch, answers, expected
):
    (tmp_path / "out.yaml").write_text("x")  # .png is free, .yaml exists
    target = str(tmp_path / "out.png")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args: (target, "*.png")),
    )
    asked = []

    def confirm(parent, path):
        asked.append(path)
        return answers.pop(0)

    monkeypatch.setattr(lib_qt, "_confirm_overwrite", confirm)
    path, _ = lib_qt.get_save_filename_ext_dialog(
        None, "Save", "", "*.png", check_ext=[".png", ".yaml"]
    )
    assert asked == [str(tmp_path / "out.yaml")]
    assert (path != "") is bool(expected)
