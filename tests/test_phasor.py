"""Test phasor localization: the estimator (``picasso.fitting.phasor``),
its propagated precision (``precision.phasor_precision``), the astigmatic
phasor calibration (``zfit.calibrate_z_phasor``) and the method in
``localize.fit`` / ``localize.localize`` (``fitting_method="phasor"``).

The estimator is held to the paper's own algorithm (Martens et al. 2018,
Sec. S9) and to noise-free spots; the aperture photometry to the
definitions of the authors' implementation; the precision, which is
Picasso's, to the scatter of the estimates over simulated spots.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math

import matplotlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

# Headless plotting for calibrate_z_phasor, which calls plt.show()
matplotlib.use("Agg")

from picasso import io, localize, zfit  # noqa: E402
from picasso.fitting import phasor, precision  # noqa: E402

from tests._progress_tracking import (  # noqa: E402
    RecordingProgress,
    assert_phases_complete,
)

PIXELSIZE = 130.0
CAMERA_INFO = {
    "Baseline": 0,
    "Sensitivity": 1,
    "Gain": 1,
    "Pixelsize": PIXELSIZE,
}
COL = phasor.COLUMNS.index


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)


def _profile(edges, center, sigma):
    """Pixel-integrated 1D Gaussians: (n, n_pixels)."""
    center = np.atleast_1d(center)[:, None]
    sigma = np.atleast_1d(sigma)[:, None]
    cdf = 0.5 * (1 + erf((edges[None, :] - center) / (np.sqrt(2) * sigma)))
    return np.diff(cdf, axis=1)


def _expected(n, box, photons, bg, sx=1.0, sy=None, jitter=0.5, rng=None):
    """Expected photons of ``n`` pixel-integrated Gaussian spots, and their
    true offsets from the center pixel."""
    rng = np.random.default_rng(0) if rng is None else rng
    sy = sx if sy is None else sy
    x0 = rng.uniform(-jitter, jitter, n)
    y0 = rng.uniform(-jitter, jitter, n)
    edges = np.arange(box + 1) - 0.5 - box // 2
    mu = (
        photons
        * _profile(edges, y0, np.broadcast_to(sy, n))[:, :, None]
        * _profile(edges, x0, np.broadcast_to(sx, n))[:, None, :]
        + bg
    )
    return mu, x0, y0


def _spots(rng, n, box, photons, bg, **kwargs):
    """Poisson spots and their true offsets."""
    mu, x0, y0 = _expected(n, box, photons, bg, rng=rng, **kwargs)
    return rng.poisson(mu).astype(np.float32), x0, y0


def _paper_algorithm(spots):
    """Martens et al., Sec. S9 (MATLAB, 1-based), on ``(n, P, P)`` spots
    indexed [y, x]: offsets from the center pixel and the magnitudes
    normalized by the box sum, as in Figs. S6-S8."""
    size = spots.shape[1]
    f = np.fft.fft2(spots.astype(np.float64))
    out = []
    for coefficient in (f[:, 0, 1], f[:, 1, 0]):
        angle = np.angle(coefficient)
        angle = np.where(angle > 0, angle - 2 * np.pi, angle)
        position = np.abs(angle) / (2 * np.pi / size) + 1
        out.append(position - (size // 2 + 1))
    total = f[:, 0, 0].real
    return (
        out[0],
        out[1],
        np.abs(f[:, 0, 1]) / total,
        np.abs(f[:, 1, 0]) / total,
    )


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


class TestEstimator:
    @pytest.mark.parametrize("box", [5, 7, 11])
    def test_matches_the_paper_algorithm(self, box):
        rng = np.random.default_rng(1)
        spots, _, _ = _spots(rng, 2000, box, 2000, 10)
        theta = phasor.fit_spots(spots)
        x, y, mag_x, mag_y = _paper_algorithm(spots)
        np.testing.assert_allclose(theta[:, COL("x")], x, atol=1e-5)
        np.testing.assert_allclose(theta[:, COL("y")], y, atol=1e-5)
        np.testing.assert_allclose(
            theta[:, COL("magnitude_x")], mag_x, rtol=1e-5
        )
        np.testing.assert_allclose(
            theta[:, COL("magnitude_y")], mag_y, rtol=1e-5
        )

    def test_noise_free_spots_are_found(self):
        """Without noise only the box's truncation of the spot's tails is
        left, which is tiny for a 7 pixel box and a 1 pixel sigma."""
        mu, x0, y0 = _expected(200, 7, 1e4, 0.0)
        theta = phasor.fit_spots(mu)
        np.testing.assert_allclose(theta[:, COL("x")], x0, atol=1e-3)
        np.testing.assert_allclose(theta[:, COL("y")], y0, atol=1e-3)

    def test_uniform_background_changes_no_position(self):
        """The first harmonic of a constant is zero."""
        mu, _, _ = _expected(50, 7, 3000, 0.0)
        a = phasor.fit_spots(mu)
        b = phasor.fit_spots(mu + 40)
        np.testing.assert_allclose(a[:, :2], b[:, :2], atol=1e-5)
        # the magnitudes are normalized by the box sum, which grows
        assert np.all(b[:, COL("magnitude_x")] < a[:, COL("magnitude_x")])

    def test_wider_spots_have_smaller_magnitudes(self):
        narrow, _, _ = _expected(20, 11, 5000, 10, sx=0.8, sy=1.6)
        theta = phasor.fit_spots(narrow)
        assert np.all(
            theta[:, COL("magnitude_x")] > theta[:, COL("magnitude_y")]
        )

    def test_branch_cut_at_the_box_edges(self):
        """Picasso puts the branch cut of the phase at the box edges, the
        paper's code at the center of the first pixel: only a spot at the
        very edge of the box tells them apart."""
        spot = np.zeros((1, 7, 7), np.float32)
        spot[0, 3, 0] = 1.0  # first column
        spot[0, 3, 6] = 0.2  # last column, next to it across the edge
        theta = phasor.fit_spots(spot)
        x_paper = _paper_algorithm(spot)[0]
        assert -3.5 <= theta[0, COL("x")] < -3.0
        assert x_paper[0] == pytest.approx(theta[0, COL("x")] + 7)

    def test_serial_and_parallel_agree(self):
        rng = np.random.default_rng(2)
        spots, _, _ = _spots(rng, 3000, 7, 1000, 10)
        np.testing.assert_array_equal(
            phasor.fit_spots(spots, parallel=True),
            phasor.fit_spots(spots, parallel=False),
        )

    def test_empty_box_has_no_estimate(self):
        theta = phasor.fit_spots(np.zeros((2, 7, 7), np.float32))
        assert np.isnan(theta[:, COL("var_x")]).all()
        assert np.isnan(theta[:, COL("magnitude_x")]).all()

    @pytest.mark.parametrize("box", [1, 4, 6])
    def test_rejects_an_even_or_tiny_box(self, box):
        with pytest.raises(ValueError, match="odd box"):
            phasor.fit_spots(np.ones((2, box, box), np.float32))

    def test_progress_and_abort(self, monkeypatch):
        monkeypatch.setattr(phasor, "_CHUNK", 100)
        rng = np.random.default_rng(3)
        spots, _, _ = _spots(rng, 350, 7, 1000, 10)
        progress = RecordingProgress()
        phasor.fit_spots(spots, progress=progress)
        assert progress.labels == ["Fitting spots"]
        assert progress.phases[0]["values"] == [100, 200, 300, 350, 350]
        assert_phases_complete(progress)
        assert phasor.fit_spots(spots, abort_callback=lambda: True) is None


# ---------------------------------------------------------------------------
# Aperture photometry
# ---------------------------------------------------------------------------


class TestPhotometry:
    @pytest.mark.parametrize(
        "box, n_signal, n_ring", [(3, 5, 4), (5, 13, 12), (7, 29, 20)]
    )
    def test_mask_sizes(self, box, n_signal, n_ring):
        """The counts of the authors' implementation: the signal within
        box // 2 of the center pixel, the ring up to 2 pixels beyond."""
        signal, ring = phasor.aperture_masks(box)
        assert signal.sum() == n_signal
        assert ring.sum() == n_ring
        assert not (signal & ring).any()

    def test_corners_of_large_boxes_are_left_out(self):
        signal, ring = phasor.aperture_masks(15)
        unused = ~(signal | ring)
        assert unused[0, 0] and unused[-1, -1]
        assert unused.sum() > 0

    def test_flat_background_and_photons(self):
        spot = np.full((1, 7, 7), 10.0, np.float32)
        spot[0, 3, 3] += 1000
        theta = phasor.fit_spots(spot)
        assert theta[0, COL("bg")] == pytest.approx(10)
        assert theta[0, COL("photons")] == pytest.approx(1000)

    def test_background_percentile(self):
        """The background is the ring value at the 0-based index
        ceil(0.56 n) of the sorted ring, as in the authors' code."""
        signal, ring = phasor.aperture_masks(7)
        spot = np.zeros((7, 7), np.float32)
        values = np.arange(1, ring.sum() + 1, dtype=np.float32)[::-1]
        spot[ring] = values
        theta = phasor.fit_spots(spot[None])
        rank = math.ceil(phasor.BACKGROUND_PERCENTILE * ring.sum())
        assert theta[0, COL("bg")] == np.sort(values)[rank]

    def test_photons_are_never_negative(self):
        spot = np.zeros((1, 7, 7), np.float32)
        _, ring = phasor.aperture_masks(7)
        spot[0][ring] = 50  # ring brighter than the signal
        assert phasor.fit_spots(spot)[0, COL("photons")] == 0


# ---------------------------------------------------------------------------
# Propagated precision
# ---------------------------------------------------------------------------


def _ratio(estimate, truth, sd):
    """Mean propagated standard error over the observed scatter."""
    return np.nanmean(sd) / np.nanstd(estimate - truth)


class TestPrecision:
    @pytest.mark.parametrize("box", [5, 7, 11])
    def test_matches_the_scatter(self, box):
        rng = np.random.default_rng(box)
        spots, x0, y0 = _spots(rng, 20000, box, 1000, 10)
        theta = phasor.fit_spots(spots)
        for axis, truth in (("x", x0), ("y", y0)):
            sd = np.sqrt(theta[:, COL(f"var_{axis}")])
            assert 0.95 < _ratio(theta[:, COL(axis)], truth, sd) < 1.05

    def test_emccd(self):
        """EM gain: gamma-distributed outputs of Poisson electrons, whose
        variance is twice the mean."""
        rng = np.random.default_rng(4)
        mu, x0, _ = _expected(20000, 7, 1000, 10, rng=rng)
        spots = rng.gamma(rng.poisson(mu)).astype(np.float32)
        lpx = precision.phasor_precision(spots, em=True)[0]
        theta = phasor.fit_spots(spots, em=True)
        assert 0.95 < _ratio(theta[:, COL("x")], x0, lpx) < 1.05
        lpx_poisson = precision.phasor_precision(spots, em=False)[0]
        np.testing.assert_allclose(lpx, lpx_poisson * np.sqrt(2), rtol=1e-6)

    def test_scmos_readout_variance(self):
        rng = np.random.default_rng(5)
        mu, x0, _ = _expected(20000, 7, 1000, 2, rng=rng)
        variance = rng.uniform(0, 60, mu.shape).astype(np.float32)
        spots = (rng.poisson(mu) + rng.normal(0, np.sqrt(variance))).astype(
            np.float32
        )
        theta = phasor.fit_spots(spots, variance=variance)
        lpx = np.sqrt(theta[:, COL("var_x")])
        assert 0.95 < _ratio(theta[:, COL("x")], x0, lpx) < 1.05
        # without the map, the readout noise is missed
        lpx_without = precision.phasor_precision(spots, em=False)[0]
        assert _ratio(theta[:, COL("x")], x0, lpx_without) < 0.9

    def test_log_magnitude_ratio(self):
        rng = np.random.default_rng(6)
        mu, _, _ = _expected(20000, 11, 3000, 10, sx=0.9, sy=1.5, rng=rng)
        spots = rng.poisson(mu).astype(np.float32)
        theta = phasor.fit_spots(spots)
        log_ratio = np.log(
            theta[:, COL("magnitude_x")] / theta[:, COL("magnitude_y")]
        )
        sd = np.sqrt(theta[:, COL("var_log_ratio")])
        assert 0.95 < np.mean(sd) / np.std(log_ratio) < 1.05

    def test_public_function_wraps_the_kernel(self):
        rng = np.random.default_rng(7)
        spots, _, _ = _spots(rng, 500, 7, 1000, 10)
        lpx, lpy, sd_log_ratio = precision.phasor_precision(spots, em=False)
        theta = phasor.fit_spots(spots)
        np.testing.assert_array_equal(lpx, np.sqrt(theta[:, COL("var_x")]))
        np.testing.assert_array_equal(lpy, np.sqrt(theta[:, COL("var_y")]))
        np.testing.assert_array_equal(
            sd_log_ratio, np.sqrt(theta[:, COL("var_log_ratio")])
        )


# ---------------------------------------------------------------------------
# Astigmatic calibration
# ---------------------------------------------------------------------------

CAL_BOX = 11
CAL_STEP = 20.0  # nm
CAL_STEPS = 41


def _widths(z):
    """Astigmatic sigmas (camera pixels) at z (nm)."""
    z = np.asarray(z, dtype=float)
    return (
        np.sqrt(1 + ((z - 200) / 400) ** 2),
        np.sqrt(1 + ((z + 200) / 400) ** 2),
    )


def _astigmatic_spots(rng, z, photons, bg=10.0):
    sx, sy = _widths(z)
    mu, x0, y0 = _expected(len(z), CAL_BOX, photons, bg, sx=sx, sy=sy, rng=rng)
    return rng.poisson(mu).astype(np.float32)


def _stage_z():
    """The stage position of each step, as calibrate_z assigns them."""
    span = (CAL_STEPS - 1) * CAL_STEP
    return -(np.arange(CAL_STEPS) * CAL_STEP - span / 2)


def _calibration_locs(rng, photons=5000, beads=30):
    frames = np.repeat(np.arange(CAL_STEPS), beads)
    theta = phasor.fit_spots(
        _astigmatic_spots(rng, _stage_z()[frames], photons)
    )
    locs = pd.DataFrame(
        {
            "frame": frames.astype(np.uint32),
            "phasor_magnitude_x": theta[:, COL("magnitude_x")],
            "phasor_magnitude_y": theta[:, COL("magnitude_y")],
        }
    )
    info = [
        {"Frames": CAL_STEPS, "Pixelsize": PIXELSIZE},
        {"Step": "Fit 2D", "Fit method": "phasor", "Fit box size": CAL_BOX},
    ]
    return locs, info


@pytest.fixture(scope="module")
def calibration():
    import matplotlib.pyplot as plt

    show = plt.show
    plt.show = lambda *a, **k: None
    try:
        locs, info = _calibration_locs(np.random.default_rng(10))
        return zfit.calibrate_z_phasor(
            locs, info, d=CAL_STEP, magnification_factor=1.0
        )
    finally:
        plt.show = show
        plt.close("all")


class TestCalibration:
    def test_contents(self, calibration):
        assert zfit._is_phasor_calibration(calibration)
        assert len(calibration["Phasor X coefficients"]) == 3
        assert calibration["Fit box size"] == CAL_BOX
        assert calibration["Pixelsize"] == PIXELSIZE
        assert calibration[io.FORMAT_VERSION_ATTR] == io.FORMAT_VERSION
        z_lo, z_hi = calibration["Z range in nm"]
        assert z_lo < -350 and z_hi > 350

    @pytest.mark.parametrize("z", [-300.0, -100.0, 0.0, 150.0, 300.0])
    def test_recovers_z_and_its_precision(self, calibration, z):
        """The paper's quadratic does not follow the curves exactly, so z
        is off by up to ~15 nm here; lpz matches the scatter."""
        rng = np.random.default_rng(int(z) + 1000)
        theta = phasor.fit_spots(
            _astigmatic_spots(rng, np.full(10000, z), 5000)
        )
        ratio = theta[:, COL("magnitude_x")] / theta[:, COL("magnitude_y")]
        z_fit = zfit._phasor_z(ratio, calibration)
        lpz = zfit._phasor_lpz(
            z_fit, np.sqrt(theta[:, COL("var_log_ratio")]), calibration
        )
        assert np.isfinite(z_fit).mean() > 0.99
        assert abs(np.nanmean(z_fit) - z) < 25
        assert 0.9 < np.nanmean(lpz) / np.nanstd(z_fit) < 1.1

    def test_no_z_outside_the_calibrated_range(self, calibration):
        z_lo, z_hi = calibration["Z range in nm"]
        cx = np.array(calibration["Phasor X coefficients"])
        cy = np.array(calibration["Phasor Y coefficients"])
        edge = np.polyval(cx, z_hi) / np.polyval(cy, z_hi)
        inside = np.polyval(cx, 0.0) / np.polyval(cy, 0.0)
        beyond = edge + 2 * (edge - inside)  # past the end of the range
        z = zfit._phasor_z(np.array([inside, beyond, np.nan]), calibration)
        assert z[0] == pytest.approx(0.0, abs=1.0)
        assert np.isnan(z[1]) and np.isnan(z[2])

    def test_linear_curves(self):
        """A magnitude quadratic without curvature still gives z."""
        calibration = {
            "Phasor X coefficients": [0.0, 1e-4, 0.5],
            "Phasor Y coefficients": [0.0, -1e-4, 0.5],
            "Z range in nm": [-400.0, 400.0],
        }
        z_true = np.array([-300.0, 0.0, 250.0])
        ratio = (1e-4 * z_true + 0.5) / (-1e-4 * z_true + 0.5)
        np.testing.assert_allclose(
            zfit._phasor_z(ratio, calibration), z_true, atol=1e-6
        )

    def test_saves_and_loads(self, tmp_path):
        locs, info = _calibration_locs(np.random.default_rng(11))
        path = str(tmp_path / "phasor_3d_calib.yaml")
        calibration = zfit.calibrate_z_phasor(
            locs, info, d=CAL_STEP, magnification_factor=0.8, path=path
        )
        assert (tmp_path / "phasor_3d_calib.png").exists()
        loaded = io.load_calibration(path)
        assert zfit._is_phasor_calibration(loaded)
        assert loaded["Magnification factor"] == 0.8
        assert loaded["Phasor X coefficients"] == pytest.approx(
            calibration["Phasor X coefficients"]
        )

    def test_needs_phasor_localizations(self):
        locs = pd.DataFrame({"frame": [0], "sx": [1.0], "sy": [1.0]})
        with pytest.raises(ValueError, match="phasor"):
            zfit.calibrate_z_phasor(
                locs,
                [{"Frames": 1, "Pixelsize": PIXELSIZE, "Fit box size": 7}],
                d=CAL_STEP,
                magnification_factor=1.0,
            )

    def test_needs_astigmatism(self):
        """Without astigmatism the ratio is flat: no z to be had."""
        rng = np.random.default_rng(12)
        frames = np.repeat(np.arange(CAL_STEPS), 30)
        mu, _, _ = _expected(len(frames), CAL_BOX, 5000, 10, sx=1.2, rng=rng)
        theta = phasor.fit_spots(rng.poisson(mu).astype(np.float32))
        locs = pd.DataFrame(
            {
                "frame": frames,
                "phasor_magnitude_x": theta[:, COL("magnitude_x")],
                # exactly the x magnitude: no slope anywhere
                "phasor_magnitude_y": theta[:, COL("magnitude_x")],
            }
        )
        info = [
            {"Frames": CAL_STEPS, "Pixelsize": PIXELSIZE},
            {"Fit box size": CAL_BOX},
        ]
        with pytest.raises(ValueError, match="monotonic"):
            zfit.calibrate_z_phasor(
                locs, info, d=CAL_STEP, magnification_factor=1.0
            )

    def test_gaussian_z_fit_refuses_it(self, calibration):
        locs = pd.DataFrame({"x": [1.0], "y": [1.0], "sx": [1.0], "sy": [1.0]})
        with pytest.raises(ValueError, match="phasor calibration"):
            zfit.zfit(
                locs, [{"Pixelsize": PIXELSIZE}], calibration=calibration
            )


# ---------------------------------------------------------------------------
# localize.fit / localize.localize
# ---------------------------------------------------------------------------

FRAMES, HEIGHT, WIDTH = 12, 48, 48
SPOTS = [(12.3, 14.6), (33.8, 30.2), (20.5, 38.4)]


def _frames(rng, box_sigma=(1.1, 1.1), z=None, photons=4000):
    """Frames with three spots each (at ``SPOTS``, jittered), with
    ``z`` astigmatic ones at that z, and the true positions (frame, x, y;
    camera pixels)."""
    frames = np.zeros((FRAMES, HEIGHT, WIDTH))
    ex = np.arange(WIDTH + 1) - 0.5
    ey = np.arange(HEIGHT + 1) - 0.5
    truth = []
    sx, sy = box_sigma if z is None else _widths(z)
    for f in range(FRAMES):
        for cx, cy in SPOTS:
            x0 = cx + rng.uniform(-0.5, 0.5)
            y0 = cy + rng.uniform(-0.5, 0.5)
            frames[f] += photons * np.outer(
                _profile(ey, y0, sy)[0], _profile(ex, x0, sx)[0]
            )
            truth.append((f, x0, y0))
    return rng.poisson(frames + 10).astype(np.uint16), np.array(truth)


def _movie(rng, **kwargs):
    """``_frames`` as a movie, its metadata and the true positions."""
    frames, truth = _frames(rng, **kwargs)
    info = [{"Frames": FRAMES, "Height": HEIGHT, "Width": WIDTH}]
    return localize._InMemoryMovie(frames, info), info, truth


def _params(box=7):
    return localize.IdentificationParameters(
        box=box, method=localize.NetGradient(minimum=1500)
    )


def _step(info, step):
    return next(e for e in info if e.get("Step") == step)


class TestFit:
    def test_columns_and_metadata(self):
        movie, info, _ = _movie(np.random.default_rng(20))
        locs, info = localize.localize(
            movie,
            camera_info=CAMERA_INFO,
            identification_parameters=_params(),
            movie_info=info,
            fitting_method="phasor",
        )
        assert len(locs) == FRAMES * len(SPOTS)
        assert list(locs.columns) == [
            "frame",
            "x",
            "y",
            "photons",
            "bg",
            "lpx",
            "lpy",
            "phasor_magnitude_x",
            "phasor_magnitude_y",
            "net_gradient",
        ]
        for column in locs.columns.drop("frame"):
            assert locs[column].dtype == np.float32, column
        assert (locs["lpx"] > 0).all() and (locs["lpy"] > 0).all()
        entry = _step(info, "Fit 2D")
        assert entry["Fit method"] == "phasor"
        assert (
            entry["Localization precision"] == localize.PHASOR_PRECISION_NOTE
        )
        assert "Photometry" in entry
        assert "Convergence criterion" not in entry
        # every column the GUI saves
        saved = set().union(*localize.LOCALIZATION_COLUMNS.values())
        assert set(locs.columns) <= saved

    def test_agrees_with_the_gaussian_fit(self):
        """Same coordinate convention as the Gaussian fit: the two agree
        on average, and both find the true positions."""
        movie, info, truth = _movie(np.random.default_rng(21))
        results = {}
        for method in ("phasor", "gausslq"):
            locs, _ = localize.localize(
                movie,
                camera_info=CAMERA_INFO,
                identification_parameters=_params(),
                movie_info=info,
                fitting_method=method,
            )
            results[method] = locs.sort_values(["frame", "x"]).reset_index(
                drop=True
            )
        p, g = results["phasor"], results["gausslq"]
        assert abs((p["x"] - g["x"]).mean()) < 0.03 * PIXELSIZE
        assert abs((p["y"] - g["y"]).mean()) < 0.03 * PIXELSIZE
        truth = truth[np.lexsort((truth[:, 1], truth[:, 0]))]
        error = p["x"].to_numpy() - truth[:, 1] * PIXELSIZE
        assert abs(error.mean()) < 3 * error.std() / np.sqrt(len(error))
        assert 0.6 < p["lpx"].mean() / error.std() < 1.6

    def test_saves_and_loads(self, tmp_path):
        movie, info, _ = _movie(np.random.default_rng(22))
        locs, info = localize.localize(
            movie,
            camera_info=CAMERA_INFO,
            identification_parameters=_params(),
            movie_info=info,
            fitting_method="phasor",
        )
        path = str(tmp_path / "movie_locs.hdf5")
        io.save_locs(path, locs, info)
        loaded, loaded_info = io.load_locs(path)
        assert len(loaded) == len(locs)
        assert _step(loaded_info, "Fit 2D")["Fit method"] == "phasor"

    def test_3d(self, calibration):
        movie, info, _ = _movie(
            np.random.default_rng(23), z=120.0, photons=6000
        )
        locs, info = localize.localize(
            movie,
            camera_info=CAMERA_INFO,
            identification_parameters=_params(CAL_BOX),
            movie_info=info,
            fitting_method="phasor",
            calibration_3d=dict(calibration, **{"Magnification factor": 0.8}),
        )
        assert {"z", "lpz"} <= set(locs.columns)
        assert locs["z"].dtype == np.float32
        # z of the calibration scaled by the magnification factor
        assert abs(locs["z"].mean() - 0.8 * 120) < 0.8 * 30
        assert (locs["lpz"] > 0).all()
        entry = _step(info, "Fit 3D phasor")
        assert entry["Outside calibrated z range"] == FRAMES * len(
            SPOTS
        ) - len(locs)
        assert entry["Magnification factor"] == 0.8
        assert entry["Z range in nm"] == calibration["Z range in nm"]

    def test_3d_calibration_needs_the_same_box(self, calibration):
        movie, info, _ = _movie(np.random.default_rng(24))
        with pytest.raises(ValueError, match="box"):
            localize.localize(
                movie,
                camera_info=CAMERA_INFO,
                identification_parameters=_params(7),
                movie_info=info,
                fitting_method="phasor",
                calibration_3d=calibration,
            )

    def test_3d_calibration_needs_the_same_pixel_size(self, calibration):
        movie, info, _ = _movie(np.random.default_rng(25))
        with pytest.raises(ValueError, match="pixel size"):
            localize.localize(
                movie,
                camera_info=dict(CAMERA_INFO, Pixelsize=100.0),
                identification_parameters=_params(CAL_BOX),
                movie_info=info,
                fitting_method="phasor",
                calibration_3d=calibration,
            )

    def test_gaussian_calibration_is_refused(self):
        from tests.conftest import CALIB_3D

        movie, info, _ = _movie(np.random.default_rng(26))
        with pytest.raises(ValueError, match="not a phasor calibration"):
            localize.localize(
                movie,
                camera_info=CAMERA_INFO,
                identification_parameters=_params(),
                movie_info=info,
                fitting_method="phasor",
                calibration_3d=dict(CALIB_3D),
            )

    def test_phasor_calibration_is_refused_by_the_gaussian_fit(
        self, calibration
    ):
        movie, info, _ = _movie(np.random.default_rng(27))
        with pytest.raises(ValueError, match="phasor calibration"):
            localize.localize(
                movie,
                camera_info=CAMERA_INFO,
                identification_parameters=_params(CAL_BOX),
                movie_info=info,
                fitting_method="gausslq",
                calibration_3d=calibration,
            )

    def test_fit_takes_phasor_calibration_only_for_phasor(self, calibration):
        movie, info, _ = _movie(np.random.default_rng(28))
        identifications, ids_info = localize.identify(
            movie, info, parameters=_params(CAL_BOX)
        )
        with pytest.raises(ValueError, match="only used by"):
            localize.fit(
                movie,
                ids_info,
                camera_info=CAMERA_INFO,
                identifications=identifications,
                box=CAL_BOX,
                fitting_method="gausslq",
                phasor_calibration=calibration,
            )

    def test_cli_names(self):
        assert localize.resolve_fit_method("phasor") == ("phasor", False)
        assert localize.resolve_fit_method("phasor-3d") == ("phasor", True)
