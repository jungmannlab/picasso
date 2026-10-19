"""Picks and drift loaded as identifications in Picasso: Localize.

Picks and drift are in nm; files written before Picasso 1.0 are in
camera pixels and do not record the pixel size, so Localize converts
them with the camera pixel size of its Parameters dialog (the movie's
camera). Either way the identifications land on the camera pixel grid.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import localize as gui_localize

PIXELSIZE = 100.0  # nm, set in the Parameters dialog
N_FRAMES = 5
DRIFT_PX = (0.5, -0.25)  # camera pixels per frame, constant


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """Localize without a movie, but with what loading picks needs."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    window = gui_localize.Window()
    window.parameters_dialog.pixelsize.setValue(PIXELSIZE)
    window.info = [{"Frames": N_FRAMES, "Width": 64, "Height": 64}]
    # needs the movie; the identifications are checked as converted
    monkeypatch.setattr(window, "_clean_up_external_ids", lambda: None)
    yield window
    window.close()


def _choose_drift(monkeypatch, path):
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(path), "")),
    )


def _check(identifications: pd.DataFrame) -> None:
    assert len(identifications) == N_FRAMES
    np.testing.assert_allclose(identifications["x"], 10.0 + DRIFT_PX[0])
    np.testing.assert_allclose(identifications["y"], 20.0 + DRIFT_PX[1])


def test_legacy_picks_and_drift_use_the_camera_pixel_size(
    window, tmp_path, monkeypatch
):
    # before Picasso 1.0: camera pixels, no format version
    picks = tmp_path / "old_picks.yaml"
    io.save_yaml(
        str(picks),
        {"Centers": [[10.0, 20.0]], "Diameter": 2.0, "Shape": "Circle"},
    )
    drift = tmp_path / "old_drift.txt"
    np.savetxt(drift, np.tile(DRIFT_PX, (N_FRAMES, 1)))  # no header
    _choose_drift(monkeypatch, drift)

    window.load_picks(str(picks))

    _check(window.identifications)


def test_picks_and_drift_in_nm(window, tmp_path, monkeypatch):
    picks = tmp_path / "picks.yaml"
    io.save_picks(
        str(picks),
        [(10.0 * PIXELSIZE, 20.0 * PIXELSIZE)],
        "Circle",
        size=2.0 * PIXELSIZE,
    )
    drift = tmp_path / "drift.txt"
    io.save_drift(
        str(drift),
        pd.DataFrame(
            {
                "x": np.full(N_FRAMES, DRIFT_PX[0] * PIXELSIZE),
                "y": np.full(N_FRAMES, DRIFT_PX[1] * PIXELSIZE),
            }
        ),
    )
    _choose_drift(monkeypatch, drift)

    window.load_picks(str(picks))

    _check(window.identifications)
