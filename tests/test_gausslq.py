"""Test ``picasso.gausslq`` — least-squares 2D Gaussian fitting.

Uses synthetic Gaussian spots with known ground truth (see
``tests/conftest.py``) so assertions can verify numerical correctness,
not just shapes.

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import gausslq

from tests.conftest import BOX


# ---------------------------------------------------------------------------
# fit_spot — single-spot least-squares fit
# ---------------------------------------------------------------------------


class TestFitSpot:
    """Numerical correctness checks for ``gausslq.fit_spot``."""

    def test_returns_six_floats_in_correct_order(self, synthetic_spot_factory):
        """Output is a 1D array of length 6 with the documented order
        ``[x, y, photons, bg, sx, sy]``."""
        spot = synthetic_spot_factory()
        result = gausslq.fit_spot(spot)
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))

    def test_recovers_centered_isotropic_spot(self, synthetic_spot_factory):
        """A noiseless centered isotropic spot must be recovered exactly
        within tight tolerances."""
        spot = synthetic_spot_factory(
            x0=0.0, y0=0.0, sx=1.0, sy=1.0, photons=5000.0, bg=10.0
        )
        x, y, photons, bg, sx, sy = gausslq.fit_spot(spot)
        assert abs(x) < 1e-3
        assert abs(y) < 1e-3
        assert sx == pytest.approx(1.0, abs=1e-3)
        assert sy == pytest.approx(1.0, abs=1e-3)
        assert photons == pytest.approx(5000.0, rel=5e-3)
        assert bg == pytest.approx(10.0, rel=5e-3)

    def test_recovers_offset_position(self, synthetic_spot_factory):
        """Spots offset from the box center are recovered with their
        offset reflected in the returned x/y."""
        spot = synthetic_spot_factory(x0=0.3, y0=-0.2)
        x, y, *_ = gausslq.fit_spot(spot)
        assert x == pytest.approx(0.3, abs=0.05)
        assert y == pytest.approx(-0.2, abs=0.05)

    def test_recovers_anisotropic_sigmas(self, synthetic_spot_factory):
        """sx != sy must be recovered correctly (astigmatic spot)."""
        spot = synthetic_spot_factory(sx=1.3, sy=0.9)
        _, _, _, _, sx, sy = gausslq.fit_spot(spot)
        assert sx == pytest.approx(1.3, abs=0.05)
        assert sy == pytest.approx(0.9, abs=0.05)

    def test_higher_bg_recovered(self, synthetic_spot_factory):
        spot = synthetic_spot_factory(photons=3000.0, bg=50.0)
        _, _, photons, bg, _, _ = gausslq.fit_spot(spot)
        assert photons == pytest.approx(3000.0, rel=0.02)
        assert bg == pytest.approx(50.0, rel=0.05)


# ---------------------------------------------------------------------------
# fit_spots — batch of spots
# ---------------------------------------------------------------------------


class TestFitSpots:
    """Batch fitting tests using the ``synthetic_spots`` fixture."""

    def test_shape_dtype_finite(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta = gausslq.fit_spots(spots)
        assert theta.shape == (len(spots), 6)
        assert theta.dtype == np.float32
        assert np.all(np.isfinite(theta))

    def test_recovers_ground_truth(self, synthetic_spots):
        """Every column of the fit matrix matches its ground truth."""
        spots, gt = synthetic_spots
        theta = gausslq.fit_spots(spots)
        # theta cols: x, y, photons, bg, sx, sy
        np.testing.assert_allclose(theta[:, 0], gt.x.values, atol=0.05)
        np.testing.assert_allclose(theta[:, 1], gt.y.values, atol=0.05)
        np.testing.assert_allclose(theta[:, 2], gt.photons.values, rtol=0.02)
        np.testing.assert_allclose(theta[:, 3], gt.bg.values, rtol=0.10)
        np.testing.assert_allclose(theta[:, 4], gt.sx.values, atol=0.03)
        np.testing.assert_allclose(theta[:, 5], gt.sy.values, atol=0.03)

    def test_per_spot_matches_fit_spot(self, synthetic_spots):
        """Batch results equal scalar ``fit_spot`` results spot-by-spot."""
        spots, _ = synthetic_spots
        theta_batch = gausslq.fit_spots(spots)
        for i in [0, 5, len(spots) - 1]:
            single = gausslq.fit_spot(spots[i])
            np.testing.assert_allclose(theta_batch[i], single, atol=1e-5)

    def test_progress_callback_invoked(self, synthetic_spots):
        """The progress callback is invoked once per spot, with the
        running index."""
        spots, _ = synthetic_spots
        calls = []
        gausslq.fit_spots(spots, progress_callback=calls.append)
        assert len(calls) == len(spots)
        # callback receives the running index, monotonically increasing
        assert calls == list(range(len(spots)))


# ---------------------------------------------------------------------------
# fit_spots_parallel + fits_from_futures
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFitSpotsParallel:
    """Multiprocessing path — verify it produces the same answer as
    serial and that the async/futures path collates correctly."""

    def test_parallel_matches_serial(self, synthetic_spots):
        spots, _ = synthetic_spots
        serial = gausslq.fit_spots(spots)
        parallel = gausslq.fit_spots_parallel(spots, asynch=False)
        assert parallel.shape == serial.shape
        np.testing.assert_allclose(parallel, serial, rtol=1e-4, atol=1e-4)

    def test_async_returns_futures_collated(self, synthetic_spots):
        spots, _ = synthetic_spots
        fs = gausslq.fit_spots_parallel(spots, asynch=True)
        assert isinstance(fs, list)
        for f in fs:
            f.result()  # block until done
        collated = gausslq.fits_from_futures(fs)
        serial = gausslq.fit_spots(spots)
        assert collated.shape == serial.shape
        np.testing.assert_allclose(collated, serial, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# locs_from_fits
# ---------------------------------------------------------------------------


class TestLocsFromFits:
    """Conversion of LQ fit theta into a localization DataFrame."""

    @pytest.fixture
    def identifications(self, synthetic_spots):
        spots, _ = synthetic_spots
        n = len(spots)
        return pd.DataFrame(
            {
                "frame": np.zeros(n, dtype=np.uint32),
                "x": np.full(n, 16, dtype=np.int64),
                "y": np.full(n, 16, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )

    @pytest.fixture
    def theta(self, synthetic_spots):
        spots, _ = synthetic_spots
        return gausslq.fit_spots(spots)

    def test_required_columns_present(self, identifications, theta):
        locs = gausslq.locs_from_fits(identifications, theta, BOX, em=False)
        for col in [
            "frame",
            "x",
            "y",
            "photons",
            "sx",
            "sy",
            "bg",
            "lpx",
            "lpy",
            "ellipticity",
            "net_gradient",
        ]:
            assert col in locs.columns

    def test_length_preserved(self, identifications, theta):
        locs = gausslq.locs_from_fits(identifications, theta, BOX, em=False)
        assert len(locs) == len(identifications)

    def test_lp_strictly_positive(self, identifications, theta):
        locs = gausslq.locs_from_fits(identifications, theta, BOX, em=False)
        assert (locs["lpx"] > 0).all()
        assert (locs["lpy"] > 0).all()

    def test_ellipticity_formula(self, identifications, theta):
        """``ellipticity == (max(sx,sy) - min(sx,sy)) / max(sx,sy)``."""
        locs = gausslq.locs_from_fits(identifications, theta, BOX, em=False)
        a = np.maximum(locs["sx"], locs["sy"])
        b = np.minimum(locs["sx"], locs["sy"])
        expected = (a - b) / a
        np.testing.assert_allclose(
            locs["ellipticity"], expected.astype(np.float32)
        )

    def test_em_doubles_precision_variance(self, identifications, theta):
        """EMCCD multiplies the precision variance by 2 -> precision
        scaled by sqrt(2)."""
        locs_no_em = gausslq.locs_from_fits(
            identifications, theta, BOX, em=False
        )
        locs_em = gausslq.locs_from_fits(identifications, theta, BOX, em=True)
        ratio = locs_em["lpx"] / locs_no_em["lpx"]
        np.testing.assert_allclose(ratio, np.sqrt(2.0), rtol=1e-4)

    def test_x_y_offsets_added_to_identifications(self, theta):
        """Final x/y is theta-offset plus the integer identification x/y.

        Use unique per-row frame numbers so the post-sort order is
        deterministic regardless of pandas' sort stability.
        """
        n = len(theta)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.arange(n, dtype=np.int64) + 10,
                "y": np.arange(n, dtype=np.int64) + 20,
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        locs = gausslq.locs_from_fits(ids, theta, BOX, em=False)
        np.testing.assert_array_equal(
            locs["x"].to_numpy(),
            (theta[:, 0] + ids["x"].to_numpy()).astype(np.float32),
        )
        np.testing.assert_array_equal(
            locs["y"].to_numpy(),
            (theta[:, 1] + ids["y"].to_numpy()).astype(np.float32),
        )

    def test_with_n_id_sorts_by_n_id(self, theta):
        """When n_id is present, locs are sorted by n_id (not frame)."""
        n = len(theta)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 16, dtype=np.int64),
                "y": np.full(n, 16, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
                "n_id": np.arange(n - 1, -1, -1, dtype=np.uint32),
            }
        )
        locs = gausslq.locs_from_fits(ids, theta, BOX, em=False)
        assert "n_id" in locs.columns
        assert list(locs["n_id"]) == list(range(n))


# ---------------------------------------------------------------------------
# localization_precision (Mortensen formula)
# ---------------------------------------------------------------------------


class TestLocalizationPrecision:
    """Analytic checks against the Mortensen formula."""

    def test_no_bg_matches_shot_noise_term(self):
        """With bg=0, the formula reduces to ``sqrt(sa^2 * (16/9) /
        photons)`` where ``sa^2 = s^2 + 1/12``."""
        photons = np.array([1000.0, 5000.0])
        s = np.array([1.0, 1.2])
        s_orth = s.copy()
        bg = np.zeros_like(photons)
        result = gausslq.localization_precision(
            photons, s, s_orth, bg, em=False
        )
        sa2 = s**2 + 1 / 12
        expected = np.sqrt(sa2 * (16 / 9) / photons)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_higher_photons_gives_better_precision(self):
        """Doubling photons should improve (decrease) the precision."""
        result = gausslq.localization_precision(
            np.array([1000.0, 2000.0, 5000.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([5.0, 5.0, 5.0]),
            em=False,
        )
        assert result[0] > result[1] > result[2]

    def test_em_scales_by_sqrt2(self):
        photons = np.array([2000.0])
        s = np.array([1.1])
        s_orth = np.array([0.9])
        bg = np.array([10.0])
        no_em = gausslq.localization_precision(
            photons, s, s_orth, bg, em=False
        )
        with_em = gausslq.localization_precision(
            photons, s, s_orth, bg, em=True
        )
        np.testing.assert_allclose(with_em / no_em, np.sqrt(2.0), rtol=1e-6)


# ---------------------------------------------------------------------------
# sigma_uncertainty (Kowalewski et al. 2026)
# ---------------------------------------------------------------------------


class TestSigmaUncertainty:
    """Verify the closed-form sigma uncertainty formula."""

    def _expected(self, sigma, sigma_orth, photons, bg):
        """Direct re-implementation of the formula in the docstring."""
        sa2 = sigma**2 + 1 / 12
        sa4 = sa2**2
        sa = sa2**0.5
        sa2_orth = sigma_orth**2 + 1 / 12
        sa_orth = sa2_orth**0.5
        var_sa2 = (
            sa4
            / photons
            * (512 / 81 + (64 * np.pi * sa * sa_orth * bg) / (3 * photons))
        )
        var_sigma = var_sa2 / (4 * sigma**2)
        return np.sqrt(var_sigma)

    def test_matches_closed_form(self):
        sigma = np.array([1.0, 1.2, 0.9])
        sigma_orth = np.array([1.0, 1.0, 1.1])
        photons = np.array([1000.0, 3000.0, 8000.0])
        bg = np.array([0.0, 5.0, 20.0])
        result = gausslq.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        expected = self._expected(sigma, sigma_orth, photons, bg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_bg_simplifies(self):
        """At bg=0, only the shot-noise term remains: sqrt(sa^4 *
        (512/81) / photons / (4 sigma^2))."""
        sigma = np.array([1.0])
        sigma_orth = np.array([1.0])
        photons = np.array([1000.0])
        bg = np.array([0.0])
        result = gausslq.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        sa4 = (1.0 + 1 / 12) ** 2
        expected = np.sqrt(sa4 * (512 / 81) / 1000.0 / 4.0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_monotonic_in_photons(self):
        """More photons -> lower sigma uncertainty."""
        photons = np.array([500.0, 1500.0, 5000.0, 20000.0])
        sigma = np.full_like(photons, 1.0)
        sigma_orth = np.full_like(photons, 1.0)
        bg = np.full_like(photons, 5.0)
        se = gausslq.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert (np.diff(se) < 0).all()

    def test_monotonic_in_bg(self):
        """Higher background -> higher sigma uncertainty."""
        bg = np.array([0.0, 5.0, 20.0, 100.0])
        sigma = np.full_like(bg, 1.0)
        sigma_orth = np.full_like(bg, 1.0)
        photons = np.full_like(bg, 2000.0)
        se = gausslq.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert (np.diff(se) > 0).all()

    def test_pandas_series_input(self):
        """The function must accept pandas Series (used by zfit downstream)."""
        sigma = pd.Series([1.0, 1.2])
        sigma_orth = pd.Series([1.1, 1.0])
        photons = pd.Series([1000.0, 2000.0])
        bg = pd.Series([5.0, 10.0])
        se = gausslq.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert len(se) == 2
        assert (se > 0).all()


# ---------------------------------------------------------------------------
# Optional GPU backend — skipped if pygpufit is not installed
# ---------------------------------------------------------------------------


class TestGpufit:
    """Tests for the optional GPU codepath. Skipped if pygpufit isn't
    installed (which is true for the typical test environment)."""

    def test_fit_spots_gpufit(self, synthetic_spots):
        pytest.importorskip("pygpufit")
        spots, gt = synthetic_spots
        theta = gausslq.fit_spots_gpufit(spots)
        assert theta.shape == (len(spots), 6)
        # GPU returns parameters as [photons, x, y, sx, sy, bg]
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=0.05)
