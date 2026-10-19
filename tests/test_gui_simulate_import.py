"""Structures imported into Picasso: Simulate.

``File > Import handles...`` takes the cluster centers of a clusterer
``.hdf5`` file as the structure. Their y axis points down as in Render,
while Simulate's points up, so the import negates y; the simulated
movie must show the clusters in their original orientation.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets
from scipy import ndimage

from picasso import io, simulate
from picasso.gui import simulate as gui_simulate


@pytest.fixture
def window(qt_offscreen, restore_theme):
    window = gui_simulate.Window()
    yield window
    window.close()


def _spot_centers(frame: np.ndarray) -> np.ndarray:
    """Centers (x, y) of the separated spots in a movie frame, in camera
    pixels with y pointing down."""
    labels, n = ndimage.label(frame > 0.2 * frame.max())
    rows_cols = ndimage.center_of_mass(frame, labels, range(1, n + 1))
    return np.array(rows_cols)[:, ::-1]


def _sorted(points: np.ndarray) -> np.ndarray:
    points = np.round(points - points.mean(axis=0), 1)
    return points[np.lexsort((points[:, 1], points[:, 0]))]


def test_imported_clusters_keep_their_orientation(window, monkeypatch):
    # An L: the long arm points down in Render (+y), the short one right.
    # Spots are 3 px apart so that they do not merge in the movie; the
    # cluster centers are in nm.
    com_x = np.array([10.0, 10.0, 10.0, 10.0, 13.0, 16.0])
    com_y = np.array([10.0, 13.0, 16.0, 19.0, 10.0, 10.0])
    pixelsize = window.pixelsizeEdit.value()
    clusters = pd.DataFrame(
        {"com_x": com_x * pixelsize, "com_y": com_y * pixelsize}
    )
    monkeypatch.setattr(io, "load_clusters", lambda path, **kwargs: clusters)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("clusters.hdf5", ""),
    )
    window.structureIncorporationEdit.setValue(100)
    window.structurerandomOrientationEdit.setChecked(False)

    window.importHandles()

    positions = window.newstruct
    assert positions.shape[1] == len(com_x)
    photons = np.full((positions.shape[1], 1), 20_000)
    frame = simulate.convert_movie(
        0,
        photons,
        positions,
        window.camerasizeEdit.value(),
        1,
        0.3,
        0,
        0,
        0,
        False,
        [],
        [],
    )
    spots = _spot_centers(frame)
    np.testing.assert_allclose(
        _sorted(spots), _sorted(np.c_[com_x, com_y]), atol=0.1
    )


def test_imported_yaml_structure_is_centered(window, monkeypatch):
    # The same L stored as a Simulate/Design .yaml structure: nm relative
    # to its own origin, y pointing up, i.e., the long arm points down
    # in the movie.
    pixelsize = window.pixelsizeEdit.value()
    x = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 6.0]) * pixelsize
    y = np.array([0.0, -3.0, -6.0, -9.0, 0.0, 0.0]) * pixelsize
    z = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    info = {
        "Structure.StructureX": ",".join(map(str, x)),
        "Structure.StructureY": ",".join(map(str, y)),
        "Structure.StructureEx": "1,1,1,1,1,1",
        "Structure.Structure3D": ",".join(map(str, z)),
    }
    monkeypatch.setattr(io, "load_info", lambda path: [info])
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("structure.yaml", ""),
    )
    window.structureIncorporationEdit.setValue(100)
    window.structurerandomOrientationEdit.setChecked(False)

    window.importHandles()

    np.testing.assert_allclose(window.handles["z"], z)
    positions = window.newstruct
    assert positions.shape[1] == len(x)
    image_size = window.camerasizeEdit.value()
    np.testing.assert_allclose(positions[:2].mean(axis=1), image_size / 2)
    photons = np.full((positions.shape[1], 1), 20_000)
    frame = simulate.convert_movie(
        0, photons, positions, image_size, 1, 0.3, 0, 0, 0, False, [], []
    )
    spots = _spot_centers(frame)
    np.testing.assert_allclose(
        _sorted(spots), _sorted(np.c_[x, -y] / pixelsize), atol=0.1
    )
