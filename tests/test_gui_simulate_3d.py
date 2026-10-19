"""3D (astigmatic) movies of Picasso: Simulate.

The 3D calibration gives the spot widths in nm; the movie is sampled on
the camera pixel grid, so Simulate passes the camera pixel size to
``picasso.simulate.convert_movie``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso import simulate
from picasso.gui import simulate as gui_simulate
from tests.conftest import CALIB_3D

PIXELSIZE = 130.0  # nm
Z = -300.0  # nm, off focus: the spot is elongated


@pytest.fixture
def window(qt_offscreen, restore_theme):
    window = gui_simulate.Window()
    yield window
    window.close()


def test_3d_movie_has_the_calibrated_spot_widths(window):
    np.random.seed(0)
    window.cx = list(CALIB_3D["X Coefficients"])  # nm
    window.cy = list(CALIB_3D["Y Coefficients"])
    # one binding site at (16, 16) camera pixels and z = Z
    structures = np.zeros((5, 1))
    structures[0, 0] = structures[1, 0] = 16.0
    structures[4, 0] = Z
    photons = np.full((1, 1), 200_000)
    params = {
        "frames": 1,
        "imagesize": 32,
        "psf": 0.82,
        "photonrate": 0,
        "background": 0,
        "noise": 0,
        "pixelsize": PIXELSIZE,
    }
    movie = window._render_movie_frames(photons, structures, params, 1)

    frame = movie[0]
    rows, cols = np.indices(frame.shape)
    total = frame.sum()

    def spread(coordinate):
        mean = (frame * coordinate).sum() / total
        return np.sqrt((frame * (coordinate - mean) ** 2).sum() / total)

    wx, wy = simulate.calculate_zpsf(Z, window.cx, window.cy)  # nm
    # the photons land in whole pixels: their variance gains 1/12
    np.testing.assert_allclose(
        spread(cols), np.sqrt((wx / PIXELSIZE) ** 2 + 1 / 12), rtol=0.03
    )
    np.testing.assert_allclose(
        spread(rows), np.sqrt((wy / PIXELSIZE) ** 2 + 1 / 12), rtol=0.03
    )
    assert abs(wx - wy) > 0.2 * min(wx, wy)  # astigmatic at this z
