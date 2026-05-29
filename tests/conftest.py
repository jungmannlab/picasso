"""Shared fixtures for the picasso test suite.

Provides:
- ``synthetic_spot_factory``: callable that builds a single Gaussian spot
  with known ground-truth parameters (with or without Poisson noise).
- ``synthetic_spots``: a batch of Gaussian spots with their ground truth,
  used by gausslq / gaussmle tests to assert numerical correctness rather
  than just shapes.
- ``locs_data`` / ``locs`` / ``info`` / ``movie_data`` / ``movie`` /
  ``movie_info``: shared loaders for the bundled test data, so individual
  test files don't reload the same files.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import io


# ---------------------------------------------------------------------------
# Loaded test data (shared across files to avoid repeated I/O)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def locs_data():
    """Return ``(locs, info)`` loaded from the bundled HDF5 once."""
    return io.load_locs("./tests/data/testdata_locs.hdf5")


@pytest.fixture(scope="session")
def locs(locs_data):
    return locs_data[0]


@pytest.fixture(scope="session")
def info(locs_data):
    return locs_data[1]


@pytest.fixture(scope="session")
def movie_data():
    """Return ``(movie, info)`` loaded from the bundled .raw once."""
    return io.load_movie("./tests/data/testdata.raw")


@pytest.fixture(scope="session")
def movie(movie_data):
    return movie_data[0]


@pytest.fixture(scope="session")
def movie_info(movie_data):
    return movie_data[1]


# ---------------------------------------------------------------------------
# Synthetic Gaussian spots — used to assert that fitters recover ground truth
# ---------------------------------------------------------------------------


def _make_gaussian_spot(
    box: int,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    photons: float,
    bg: float,
) -> np.ndarray:
    """Build a noiseless 2D Gaussian spot on a (box, box) grid.

    The center of the box is at index ``box // 2`` so ``x0 = y0 = 0``
    places the spot exactly in the middle pixel — matching the convention
    used by ``picasso.gausslq.fit_spot`` (which returns offsets from the
    box center).
    """
    half = box // 2
    grid = np.arange(-half, half + 1, dtype=np.float64)
    gx = np.exp(-0.5 * ((grid - x0) / sx) ** 2) / (sx * np.sqrt(2 * np.pi))
    gy = np.exp(-0.5 * ((grid - y0) / sy) ** 2) / (sy * np.sqrt(2 * np.pi))
    spot = photons * np.outer(gy, gx) + bg
    return spot.astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_spot_factory():
    """Return a callable that builds Gaussian spots with known params.

    Signature: ``factory(box=7, x0=0.0, y0=0.0, sx=1.0, sy=1.0,
    photons=5000.0, bg=10.0, noise=False, seed=0) -> ndarray``.
    """

    def _factory(
        box: int = 7,
        x0: float = 0.0,
        y0: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        photons: float = 5000.0,
        bg: float = 10.0,
        noise: bool = False,
        seed: int = 0,
    ) -> np.ndarray:
        spot = _make_gaussian_spot(box, x0, y0, sx, sy, photons, bg)
        if noise:
            rng = np.random.default_rng(seed)
            # Poisson photon noise — model match for MLE
            spot = rng.poisson(np.maximum(spot, 0.0)).astype(np.float32)
        return spot

    return _factory


@pytest.fixture(scope="module")
def synthetic_spots():
    """Return ``(spots, ground_truth_df)`` for a batch of clean Gaussian spots.

    ``ground_truth_df`` has columns ``x, y, sx, sy, photons, bg``. Spots
    are generated noiseless so fitters should recover ground truth to
    tight tolerance — anything that bends past those tolerances indicates
    a real bug, not a noise artifact.
    """
    box = 7
    n = 64
    rng = np.random.default_rng(42)
    gt = pd.DataFrame(
        {
            "x": rng.uniform(-0.5, 0.5, n),
            "y": rng.uniform(-0.5, 0.5, n),
            "sx": rng.uniform(0.9, 1.4, n),
            "sy": rng.uniform(0.9, 1.4, n),
            "photons": rng.uniform(2000.0, 8000.0, n),
            "bg": rng.uniform(5.0, 30.0, n),
        }
    )
    spots = np.empty((n, box, box), dtype=np.float32)
    for i in range(n):
        spots[i] = _make_gaussian_spot(
            box,
            gt.x[i],
            gt.y[i],
            gt.sx[i],
            gt.sy[i],
            gt.photons[i],
            gt.bg[i],
        )
    return spots, gt


@pytest.fixture(scope="module")
def synthetic_spots_noisy():
    """Return ``(spots, ground_truth_df)`` like ``synthetic_spots`` but with
    Poisson photon noise. Used to test MLE (which models Poisson noise
    explicitly) and the parallel fitting paths."""
    box = 7
    n = 32
    rng = np.random.default_rng(123)
    gt = pd.DataFrame(
        {
            "x": rng.uniform(-0.5, 0.5, n),
            "y": rng.uniform(-0.5, 0.5, n),
            "sx": rng.uniform(0.9, 1.4, n),
            "sy": rng.uniform(0.9, 1.4, n),
            # higher photons so MLE has a clean signal
            "photons": rng.uniform(5000.0, 12000.0, n),
            "bg": rng.uniform(5.0, 20.0, n),
        }
    )
    spots = np.empty((n, box, box), dtype=np.float32)
    for i in range(n):
        clean = _make_gaussian_spot(
            box,
            gt.x[i],
            gt.y[i],
            gt.sx[i],
            gt.sy[i],
            gt.photons[i],
            gt.bg[i],
        )
        spots[i] = rng.poisson(np.maximum(clean, 0.0)).astype(np.float32)
    return spots, gt


# ---------------------------------------------------------------------------
# Convenience: identifications + spots extracted from the bundled movie
# (used by both test_localize and test_gausslq / test_gaussmle).
# ---------------------------------------------------------------------------


# Shared constants — imported by individual test modules so a single change
# here propagates everywhere. Keep this list narrow: only values used in 2+
# test files belong here.
CAMERA_INFO = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
BOX = 7
MIN_NG = 5000
PIXELSIZE = 130  # camera pixel size, nm

# Astigmatism 3D calibration shared by test_postprocess / test_zfit /
# test_localize. Tests that mutate it should pass ``dict(CALIB_3D)``.
CALIB_3D = {
    "X Coefficients": [
        -1.6680708772714857e-18,
        2.4038209829154137e-15,
        2.1771067332017187e-12,
        -3.0324788231238476e-09,
        3.5433326085494675e-06,
        0.0023039289366630425,
        1.2026032603707493,
    ],
    "Y Coefficients": [
        -1.7708672355491796e-18,
        9.808249540501714e-16,
        2.10653248543535e-12,
        2.228026137415219e-11,
        3.628007433361433e-06,
        -0.001646865504353452,
        1.2257249554338714,
    ],
    "Step size in nm": 5.0,
    "Number of frames": 201,
    "Magnification factor": 0.79,
}


@pytest.fixture(scope="session")
def real_identifications(movie):
    """Identifications from the bundled .raw — shared across test files."""
    from picasso import localize

    return localize.identify(movie, MIN_NG, BOX, return_info=False)


@pytest.fixture(scope="session")
def real_spots(movie, real_identifications):
    """Extracted spots from the bundled .raw — shared across test files."""
    from picasso import localize

    return localize.get_spots(movie, real_identifications, BOX, CAMERA_INFO)


# ---------------------------------------------------------------------------
# AbstractPicassoMovie wrapper
# ---------------------------------------------------------------------------
#
# ``localize.fit2D`` / ``localize.localize`` / ``localize.localize_3D`` all
# assert ``isinstance(movie, io.AbstractPicassoMovie)``, but ``io.load_movie``
# returns a plain ``np.memmap`` for ``.raw`` files. To exercise these paths
# without bundling an OME-TIFF, we wrap the memmap in a thin subclass that
# delegates everything to the underlying ndarray.


class _MemmapPicassoMovie(io.AbstractPicassoMovie):
    """Minimal AbstractPicassoMovie subclass backed by an ndarray.

    Implements only what the localize pipeline needs: iteration (for
    ``_cut_spots_framebyframe``), ``__len__``, ``__getitem__``, ``dtype``,
    and the abstract no-op methods.
    """

    def __init__(self, array, info):
        super().__init__()
        self._array = np.asarray(array)
        self._info = info
        self.n_frames = len(self._array)
        self.shape = self._array.shape

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def info(self):
        return self._info[0]

    def camera_parameters(self, config):
        return {
            "gain": [1],
            "qe": [1],
            "wavelength": [0],
            "cam_index": 0,
            "camera": "None",
        }

    def __getitem__(self, it):
        return self._array[it]

    def __iter__(self):
        return iter(self._array)

    def __len__(self):
        return len(self._array)

    def get_frame(self, index):
        return self._array[index]

    def tofile(self, file_handle, byte_order=None):
        self._array.tofile(file_handle)

    @property
    def dtype(self):
        return self._array.dtype


@pytest.fixture(scope="session")
def picasso_movie(movie, movie_info):
    """``AbstractPicassoMovie`` wrapper around the bundled .raw movie.

    Use this for ``localize.fit2D`` / ``localize.localize`` /
    ``localize.localize_3D`` tests — those functions assert their movie
    argument ``isinstance`` of ``AbstractPicassoMovie``."""
    return _MemmapPicassoMovie(movie, movie_info)
