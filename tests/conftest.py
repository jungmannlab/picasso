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
- ``qapp`` / ``qt_offscreen``: the single ``QApplication`` every GUI test
  shares, and a per-test wrapper that closes the widgets a test opened.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from picasso import io, transforms

# Qt must be told to run without a display before the first QApplication is
# built, and the environment is read once at that point - hence a module-level
# default rather than a fixture. ``setdefault`` keeps an explicit
# ``QT_QPA_PLATFORM=xcb pytest`` (to watch the widgets) working.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Unhandled exceptions
# ---------------------------------------------------------------------------

# An exception that escapes a Qt slot (or a ``QThread.run``) never reaches the
# test that caused it: Python hands it to ``sys.excepthook`` and PyQt aborts
# the interpreter, taking pytest's captured output with it - a bare
# "Fatal Python error: Aborted" and no traceback, which is how these went
# unnoticed. Recording them here both keeps the process alive and, through
# ``_no_unhandled_exceptions`` below, fails the test that produced one instead
# of the whole run. Set ``PICASSO_CRASH_LOG`` to also append them (and the Qt
# messages) to a file, for a run that dies anyway.
_CRASH_LOG = os.environ.get("PICASSO_CRASH_LOG")
_unhandled: list[str] = []


def _write_crash_log(text: str) -> None:
    """Append to the crash log, flushed to disk before returning."""
    if not _CRASH_LOG:
        return
    with open(_CRASH_LOG, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _install_exception_recorder() -> None:
    """Record unhandled exceptions instead of letting PyQt abort on them."""
    import sys
    import traceback

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        _unhandled.append(text)
        _write_crash_log("\n=== unhandled exception ===\n" + text)

    sys.excepthook = excepthook

    if not _CRASH_LOG:
        return
    try:
        from PyQt6 import QtCore
    except ImportError:  # pragma: no cover - Qt is optional here
        return

    def message_handler(mode, context, message):
        _write_crash_log(f"\n=== Qt {mode.name}: {message}\n")

    QtCore.qInstallMessageHandler(message_handler)


_install_exception_recorder()


@pytest.fixture(autouse=True)
def _no_unhandled_exceptions():
    """Fail a test that let an exception escape a slot or a worker thread.

    Delivered signals run outside the test's own call stack, so an exception
    in one is invisible to its assertions - and fatal to the interpreter.
    """
    del _unhandled[:]
    yield
    if _unhandled:
        report = "\n".join(_unhandled)
        del _unhandled[:]
        pytest.fail(
            "an exception escaped a Qt slot or a worker thread (PyQt aborts "
            f"the process on one):\n{report}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(items):
    """Mark every test that builds Qt widgets as ``gui``.

    Deriving the marker from the fixtures a test requests keeps it from
    drifting out of date, and lets a headless run skip the lot with
    ``pytest -m "not gui"``.

    Only fixtures named in the test's own signature count. A module that
    keeps a ``QApplication`` alive with an autouse fixture (test_localize
    does) would otherwise mark its every test as a GUI test, including the
    numerical ones that never build a widget.
    """
    for item in items:
        info = getattr(item, "_fixtureinfo", None)
        argnames = getattr(info, "argnames", None)
        if argnames is None:  # pragma: no cover - non-Function items
            argnames = getattr(item, "fixturenames", ())
        if {"qapp", "qt_offscreen"} & set(argnames):
            item.add_marker(pytest.mark.gui)


@pytest.fixture(scope="session")
def qapp():
    """The one ``QApplication`` for the whole session.

    Qt allows a single application object per process and it must outlive
    every widget built from it, so this is session scoped and never torn
    down. Skips rather than errors where Qt cannot start, so the suite stays
    usable on machines without a working Qt platform plugin.
    """
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        try:
            app = QtWidgets.QApplication([])
        except Exception as exc:  # pragma: no cover - environment issue
            pytest.skip(f"Qt could not be initialized: {exc}")
    return app


@pytest.fixture
def qt_offscreen(qapp):
    """``qapp``, plus cleanup of the widgets the test opened.

    A widget that outlives its test keeps receiving events and can crash the
    interpreter during interpreter shutdown, so anything top level the test
    left behind is closed and scheduled for deletion here.
    """
    before = set(qapp.topLevelWidgets())
    yield qapp
    for widget in qapp.topLevelWidgets():
        if widget not in before:
            widget.close()
            widget.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------


def affine(matrix) -> transforms.AffineTransform:
    """An ``AffineTransform`` from a ``(2, 3)`` or ``(3, 3)`` matrix.

    Most tests write channel registrations as the bare ``(2, 3)`` they used to
    be stored as; this lifts them into the transform objects the code now
    passes around.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    return transforms.AffineTransform(matrix=matrix)


def apply_transform(xy, transform):
    """Map ``(n, 2)`` points through a transform, its serialized dict, or a
    bare matrix."""
    if not isinstance(transform, (transforms.Transform, dict)):
        transform = affine(transform)
    return transforms.from_dict(transform).apply(xy)


def affine_matrix(transform) -> np.ndarray:
    """The ``(2, 3)`` matrix of an affine transform (or its serialized dict) -
    what a channel registration used to be stored as."""
    return transforms.from_dict(transform).matrix[:2]


def affine_matrix_3x3(transform) -> np.ndarray:
    """The ``(3, 3)`` homogeneous matrix of an affine or projective transform
    (or its serialized dict)."""
    return transforms.from_dict(transform).matrix


def linear_part(transform) -> np.ndarray:
    """The ``(2, 2)`` local linear part of a transform at its domain center -
    what the old ``transform[:, :2]`` slice used to be."""
    return transforms.from_dict(transform).jacobian([[0.0, 0.0]])[0]


IDENTITY = transforms.identity().to_dict()


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


@pytest.fixture(scope="module")
def synthetic_spots_isotropic():
    """Return ``(spots, ground_truth_df)`` for a batch of *isotropic*
    Gaussian spots (``sx == sy``).

    Used to test the spherical (single-width) fitters — least squares and
    MLE, CPU and GPU. Because the ground truth already has ``sx == sy``,
    a correct spherical fit must recover the shared width and the
    ellipticity of the resulting localizations is exactly 0 (which is why
    the spherical output drops the ``ellipticity`` column altogether).
    """
    box = 7
    n = 48
    rng = np.random.default_rng(7)
    s = rng.uniform(0.9, 1.4, n)
    gt = pd.DataFrame(
        {
            "x": rng.uniform(-0.5, 0.5, n),
            "y": rng.uniform(-0.5, 0.5, n),
            "sx": s,
            "sy": s.copy(),
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


def make_rotated_gaussian_spot(
    box: int,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    photons: float,
    bg: float,
    angle: float,
) -> np.ndarray:
    """Point-sampled rotated elliptical Gaussian spot.

    Matches the model both ``gausslq._compute_model_rotated`` (CPU) and
    Gpufit's ``GAUSS_2D_ROTATED`` (GPU) optimize::

        mu = photons / (2 pi sx sy) * exp(-0.5 (u^2/sx^2 + w^2/sy^2)) + bg
        u = dx cos(a) - dy sin(a),  w = dx sin(a) + dy cos(a)

    where ``dx``/``dy`` are pixel offsets from the spot center (``x0``/``y0``
    are offsets from the box center) and ``x`` varies along columns.
    """
    half = box // 2
    g = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(g, g)  # X varies along columns (x), Y along rows (y)
    dx, dy = X - x0, Y - y0
    ct, st = np.cos(angle), np.sin(angle)
    u = dx * ct - dy * st
    w = dx * st + dy * ct
    e = np.exp(-0.5 * (u**2 / sx**2 + w**2 / sy**2))
    return (photons / (2 * np.pi * sx * sy) * e + bg).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_spots_rotated():
    """Return ``(spots, ground_truth_df)`` for a batch of *rotated*
    elliptical Gaussian spots.

    ``ground_truth_df`` has the usual ``x, y, sx, sy, photons, bg`` columns
    plus ``angle`` (radians). The widths are deliberately anisotropic so
    the rotation angle is well-defined, and the angles span roughly
    ``(-pi/2, pi/2)`` (the range over which the ellipse orientation is
    unique). Used to test the rotated fitters — LQ (CPU/GPU) and MLE (GPU).
    """
    box = 9
    rng = np.random.default_rng(2026)
    angles = np.array(
        [-1.3, -0.9, -0.45, -0.1, 0.15, 0.5, 0.85, 1.2], dtype=np.float64
    )
    n = len(angles)
    gt = pd.DataFrame(
        {
            "x": rng.uniform(-0.3, 0.3, n),
            "y": rng.uniform(-0.3, 0.3, n),
            # Keep the widths well separated so the ellipse orientation is
            # well conditioned — a near-circular spot has an ill-defined
            # angle and is not a meaningful recovery target.
            "sx": rng.uniform(1.6, 1.9, n),
            "sy": rng.uniform(0.8, 1.0, n),
            "photons": rng.uniform(5000.0, 9000.0, n),
            "bg": rng.uniform(5.0, 20.0, n),
            "angle": angles,
        }
    )
    spots = np.empty((n, box, box), dtype=np.float32)
    for i in range(n):
        spots[i] = make_rotated_gaussian_spot(
            box,
            gt.x[i],
            gt.y[i],
            gt.sx[i],
            gt.sy[i],
            gt.photons[i],
            gt.bg[i],
            gt.angle[i],
        )
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

    return localize.identify(movie, MIN_NG, BOX)[0]


@pytest.fixture(scope="session")
def real_spots(movie, real_identifications):
    """Extracted spots from the bundled .raw — shared across test files."""
    from picasso import localize

    return localize.get_spots(movie, real_identifications, BOX, CAMERA_INFO)


# ---------------------------------------------------------------------------
# AbstractPicassoMovie wrapper
# ---------------------------------------------------------------------------
#
# ``localize.fit`` / ``localize.localize`` assert
# ``isinstance(movie, io.AbstractPicassoMovie)``, but ``io.load_movie``
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

    Use this for ``localize.fit`` / ``localize.localize`` tests — those
    functions assert their movie argument ``isinstance`` of
    ``AbstractPicassoMovie``."""
    return _MemmapPicassoMovie(movie, movie_info)


@pytest.fixture(scope="session")
def picasso_movie_factory():
    """Wrap an arbitrary ndarray as an ``AbstractPicassoMovie``.

    As ``picasso_movie``, but for tests that build their own synthetic movie
    (with known ground truth) rather than using the bundled .raw."""
    return _MemmapPicassoMovie


# ---------------------------------------------------------------------------
# sCMOS camera calibration (per-pixel offset / variance / gain)
# ---------------------------------------------------------------------------


def _make_scmos_maps(
    height=16, width=16, n_hot=3, gain=2.13, seed=0
) -> dict[str, np.ndarray]:
    """Ground-truth per-pixel maps resembling a real sCMOS sensor.

    Values follow Huang et al. (2013), Supplementary Fig. 1, measured on a
    Hamamatsu ORCA Flash 4.0: an offset around 100 ADU, a readout variance of
    a couple of ADU squared with a sparse tail of very noisy pixels, and a
    gain of roughly 2 ADU per photoelectron with column-wise structure.
    """
    rng = np.random.default_rng(seed)
    offset = 100.0 + rng.normal(0.0, 1.5, (height, width))
    variance = rng.gamma(shape=4.0, scale=0.5, size=(height, width)) + 0.5
    # The hot tail is what the whole noise model exists for, so it is placed
    # deterministically rather than left to chance at this map size.
    flat = rng.choice(height * width, size=n_hot, replace=False)
    variance.flat[flat] = np.array([40.0, 220.0, 900.0])[:n_hot]
    # Column-wise amplifiers: gain varies mostly along x, weakly along y.
    columns = gain + rng.normal(0.0, 0.12, (1, width))
    gains = np.repeat(columns, height, axis=0) + rng.normal(
        0.0, 0.02, (height, width)
    )
    return {
        "offset": offset,
        "variance": variance,
        "gain": np.abs(gains),
        "hot_pixels": flat[:n_hot],
    }


@pytest.fixture(scope="session")
def scmos_maps():
    """Ground-truth per-pixel offset / variance / gain maps."""
    return _make_scmos_maps()


@pytest.fixture(scope="session")
def scmos_maps_factory():
    """Build ground-truth per-pixel maps at an arbitrary size."""
    return _make_scmos_maps


@pytest.fixture(scope="session")
def dark_movie_factory():
    """Build a dark movie consistent with a set of ground-truth maps.

    ``camera_output`` also serves the bright case: pass a photon level and it
    adds Poisson shot noise amplified by the gain, which is exactly the
    photon-transfer-curve model the gain calibration inverts.
    """

    def camera_output(maps, n_frames, photons=0.0, seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        shape = maps["offset"].shape
        frames = maps["offset"] + rng.normal(
            0.0, np.sqrt(maps["variance"]), (n_frames, *shape)
        )
        if photons:
            frames = frames + maps["gain"] * rng.poisson(
                photons, (n_frames, *shape)
            )
        return frames.astype(dtype)

    return camera_output


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path, monkeypatch):
    """Every test reads and writes its own, initially absent, settings
    file: the GUIs persist defaults on start and save on close, and the
    developer's ``~/.picasso/settings.yaml`` must never be touched by
    the suite. Tests that need a specific file patch the same name."""
    from picasso import io

    monkeypatch.setattr(
        io, "_user_settings_filename", lambda: str(tmp_path / "settings.yaml")
    )
    monkeypatch.setattr(io, "_settings_load_error", None)


@pytest.fixture(autouse=True)
def cpu_render_backend(request, monkeypatch):
    """Rendering tests compare against CPU references, and the GPU
    backend is selected by default wherever a GPU initializes, so every
    test renders on the CPU unless it opts in: ``tests/test_render_gpu.py``
    and tests marked ``gpu_backend`` see the real selection."""
    if request.node.fspath.basename == "test_render_gpu.py":
        return
    if request.node.get_closest_marker("gpu_backend") is not None:
        return
    from picasso.render import backend

    monkeypatch.setattr(backend, "_gpu_backend", lambda adapter, warn: None)


@pytest.fixture(autouse=True)
def synchronous_gui_rendering(monkeypatch):
    """GUI tests assert on images immediately after ``update_scene``,
    so the async render worker is disabled by default whenever the
    render GUI module is loaded; async-specific tests re-enable it
    explicitly. Zero-cost for tests that never import the GUI."""
    gui_render = sys.modules.get("picasso.gui.render")
    if gui_render is not None:
        monkeypatch.setattr(gui_render.View, "async_rendering", False)
    gui_rotation = sys.modules.get("picasso.gui.rotation")
    if gui_rotation is not None:
        monkeypatch.setattr(
            gui_rotation.ViewRotation, "async_rendering", False
        )
    yield
