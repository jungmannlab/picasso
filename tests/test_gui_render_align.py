"""Keeping or removing the localizations outside the overlap of the
channels after aligning them, in Render and in ``picasso align``.

Render reads ``Keep non-overlapping after alignment`` from the
``Render`` section of the user settings (True, i.e., all localizations
are kept, by default); the command line takes
``-r/--remove-non-overlapping``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0
N_STRAY = 3


def _locs(x_max: float, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Localizations at (x, y) in [4, x_max] x [4, 28] whose positions
    repeat in each channel, so that the channels align with no shift."""
    rng = np.random.default_rng(seed)
    locs = pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(4.0, 28.0, size=n),
            "y": rng.uniform(4.0, 28.0, size=n),
            "lpx": np.full(n, 0.1),
            "lpy": np.full(n, 0.1),
            "photons": np.full(n, 1000.0),
        }
    )
    return locs[locs["x"] <= x_max].reset_index(drop=True)


def _stray(locs: pd.DataFrame, x: float = 30.5) -> pd.DataFrame:
    """``locs`` plus a few localizations at ``x``."""
    stray = locs.iloc[:N_STRAY].copy()
    stray["x"] = x
    return pd.concat([locs, stray], ignore_index=True)


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _set_keep(value) -> None:
    settings = io.load_user_settings()
    settings["Render"][gui_render.KEEP_NON_OVERLAPPING_KEY] = value
    io.save_user_settings(settings)


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window with two channels; the second has a few
    localizations beyond the first one."""
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "a.hdf5"), _locs(28.0), _info(), render_=False
    )
    window.view.add(
        str(tmp_path / "b.hdf5"), _stray(_locs(28.0)), _info(), render_=False
    )
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    return window


def test_setting_is_written_with_the_default():
    assert gui_render.keep_non_overlapping_setting() is True
    render_settings = io.load_user_settings()["Render"]
    assert render_settings[gui_render.KEEP_NON_OVERLAPPING_KEY] is True


def test_invalid_value_falls_back_to_keeping_all():
    _set_keep("no")
    assert gui_render.keep_non_overlapping_setting() is True
    # the user's value is left in the file
    render_settings = io.load_user_settings()["Render"]
    assert render_settings[gui_render.KEEP_NON_OVERLAPPING_KEY] == "no"


def test_render_writes_the_setting_on_start(qt_offscreen):
    gui_render.Window(plugins_loaded=True)
    render_settings = io.load_user_settings()["Render"]
    assert render_settings[gui_render.KEEP_NON_OVERLAPPING_KEY] is True


def test_align_keeps_all_localizations_by_default(window):
    n_before = [len(_) for _ in window.view.locs]
    window.view.align()
    assert [len(_) for _ in window.view.locs] == n_before


def test_align_removes_non_overlapping_localizations(window):
    _set_keep(False)
    n_before = [len(_) for _ in window.view.locs]
    window.view.align()
    locs = window.view.locs
    assert len(locs[1]) <= n_before[1] - N_STRAY
    assert locs[1]["x"].max() <= locs[0]["x"].max()


def test_align_keeps_all_if_the_channels_do_not_overlap(window, monkeypatch):
    """Disjoint channels are aligned but keep their localizations, and
    the user is told why."""
    _set_keep(False)
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )
    monkeypatch.setattr(
        gui_render.postprocess,
        "align",
        lambda *args, **kwargs: [
            _locs(10.0),
            _locs(28.0).query("x > 20"),
        ],
    )
    window.view.align()
    assert len(warnings) == 1
    assert len(window.view.locs[0]) == len(_locs(10.0))


@pytest.mark.parametrize("remove", [False, True])
def test_cli_remove_non_overlapping(tmp_path, monkeypatch, remove):
    from picasso.__main__ import main

    paths = []
    for name, locs in [("a", _locs(28.0)), ("b", _stray(_locs(28.0)))]:
        path = str(tmp_path / f"{name}.hdf5")
        io.save_locs(path, locs, _info())
        paths.append(path)
    argv = ["picasso", "align", *paths]
    if remove:
        argv.append("--remove-non-overlapping")
    monkeypatch.setattr(sys, "argv", argv)
    main()
    aligned_b, info_b = io.load_locs(str(tmp_path / "b_align.hdf5"))
    n_b = len(_locs(28.0)) + N_STRAY
    if remove:
        assert len(aligned_b) <= n_b - N_STRAY
    else:
        assert len(aligned_b) == n_b
    assert info_b[-1]["Remove non-overlapping"] is remove
