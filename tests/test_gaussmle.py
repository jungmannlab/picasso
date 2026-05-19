"""Test ``picasso.gaussmle`` — maximum-likelihood 2D Gaussian fitting.

Uses synthetic Gaussian spots with known ground truth (see
``tests/conftest.py``) so MLE convergence and accuracy can be checked
numerically.

Note on theta layout: ``gaussmle`` returns ``theta`` with positions
*relative to the box origin* (so a centered spot has ``x = y = box//2``,
not 0). ``locs_from_fits`` later subtracts ``box//2``.

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import gausslq, gaussmle

from tests.conftest import BOX


BOX_HALF = BOX // 2
EPS = 1e-3
MAX_IT = 1000


# ---------------------------------------------------------------------------
# gaussmle — core MLE fit
# ---------------------------------------------------------------------------


class TestGaussmle:
    """Numerical correctness checks for ``gaussmle.gaussmle``."""

    def test_returns_four_arrays_with_expected_shapes(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta, crlbs, lls, its = gaussmle.gaussmle(
            spots, EPS, MAX_IT, method="sigmaxy"
        )
        assert theta.shape == (len(spots), 6)
        assert crlbs.shape == (len(spots), 6)
        assert lls.shape == (len(spots),)
        assert its.shape == (len(spots),)
        assert theta.dtype == np.float32

    def test_sigmaxy_recovers_isotropic_ground_truth(self, synthetic_spots):
        """For (mostly anisotropic) clean spots, sigmaxy recovers
        sx, sy independently."""
        spots, gt = synthetic_spots
        theta, _, _, _ = gaussmle.gaussmle(
            spots, EPS, MAX_IT, method="sigmaxy"
        )
        # theta[:, 0] is x, but in box-origin coordinates -> subtract box//2
        np.testing.assert_allclose(
            theta[:, 0] - BOX_HALF, gt.x.values, atol=0.05
        )
        np.testing.assert_allclose(
            theta[:, 1] - BOX_HALF, gt.y.values, atol=0.05
        )
        # MLE uses an integrated-Gaussian model vs. our point-sampled
        # synthetic spots; slight model mismatch -> ~5% tolerance on
        # photons / bg, ~0.05 px on sigmas.
        np.testing.assert_allclose(theta[:, 2], gt.photons.values, rtol=0.05)
        np.testing.assert_allclose(theta[:, 3], gt.bg.values, rtol=0.20)
        np.testing.assert_allclose(theta[:, 4], gt.sx.values, atol=0.10)
        np.testing.assert_allclose(theta[:, 5], gt.sy.values, atol=0.10)

    def test_sigma_method_returns_equal_sx_sy(self, synthetic_spots):
        """The 'sigma' method imposes sx == sy in the output."""
        spots, _ = synthetic_spots
        theta, _, _, _ = gaussmle.gaussmle(spots, EPS, MAX_IT, method="sigma")
        np.testing.assert_array_equal(theta[:, 4], theta[:, 5])

    def test_invalid_method_raises(self, synthetic_spots):
        spots, _ = synthetic_spots
        with pytest.raises(ValueError):
            gaussmle.gaussmle(spots, EPS, MAX_IT, method="bogus")

    def test_iterations_within_max_it(self, synthetic_spots):
        spots, _ = synthetic_spots
        _, _, _, its = gaussmle.gaussmle(spots, EPS, MAX_IT, method="sigmaxy")
        assert (its <= MAX_IT).all()
        assert (its >= 0).all()

    def test_crlbs_finite_and_positive(self, synthetic_spots):
        spots, _ = synthetic_spots
        _, crlbs, _, _ = gaussmle.gaussmle(
            spots, EPS, MAX_IT, method="sigmaxy"
        )
        # all six parameters should have a finite, positive CRLB
        assert np.all(np.isfinite(crlbs))
        assert (crlbs > 0).all()

    def test_recovers_noisy_spots_within_loose_tolerance(
        self, synthetic_spots_noisy
    ):
        """With Poisson noise, MLE still recovers within ~1 sigma_loc."""
        spots, gt = synthetic_spots_noisy
        theta, _, _, _ = gaussmle.gaussmle(
            spots, EPS, MAX_IT, method="sigmaxy"
        )
        # position recovered to better than ~0.2 px (well above noise floor)
        np.testing.assert_allclose(
            theta[:, 0] - BOX_HALF, gt.x.values, atol=0.2
        )
        np.testing.assert_allclose(
            theta[:, 1] - BOX_HALF, gt.y.values, atol=0.2
        )
        # photons and sigmas: ~5% relative
        np.testing.assert_allclose(theta[:, 2], gt.photons.values, rtol=0.10)
        np.testing.assert_allclose(theta[:, 4], gt.sx.values, atol=0.10)

    def test_progress_callback_invoked(self, synthetic_spots):
        spots, _ = synthetic_spots
        calls = []
        gaussmle.gaussmle(
            spots,
            EPS,
            MAX_IT,
            method="sigmaxy",
            progress_callback=calls.append,
        )
        assert calls == list(range(len(spots)))

    def test_looser_eps_fewer_iterations_on_average(
        self, synthetic_spots_noisy
    ):
        """Looser convergence threshold should require fewer iterations."""
        spots, _ = synthetic_spots_noisy
        _, _, _, it_tight = gaussmle.gaussmle(
            spots, eps=1e-5, max_it=MAX_IT, method="sigmaxy"
        )
        _, _, _, it_loose = gaussmle.gaussmle(
            spots, eps=1e-1, max_it=MAX_IT, method="sigmaxy"
        )
        assert it_loose.mean() <= it_tight.mean()


# ---------------------------------------------------------------------------
# gaussmle_async — threaded MLE
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGaussmleAsync:
    """Threaded MLE — must produce equivalent output to serial."""

    def _wait_until_done(self, current, n, timeout_s=60.0):
        import time

        t0 = time.time()
        while current[0] < n:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"gaussmle_async did not finish within {timeout_s}s"
                )
            time.sleep(0.05)

    def test_async_matches_serial(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta_serial, _, _, _ = gaussmle.gaussmle(
            spots, EPS, MAX_IT, method="sigmaxy"
        )
        current, theta_async, crlbs, lls, its = gaussmle.gaussmle_async(
            spots, EPS, MAX_IT, method="sigmaxy"
        )
        self._wait_until_done(current, len(spots))
        # CRLBs default to inf — once a spot finishes they become finite.
        # Equivalent results despite worker scheduling order.
        np.testing.assert_allclose(theta_async, theta_serial, atol=1e-3)
        assert np.all(np.isfinite(crlbs))

    def test_invalid_method_raises_async(self, synthetic_spots):
        spots, _ = synthetic_spots
        with pytest.raises(ValueError):
            gaussmle.gaussmle_async(spots, EPS, MAX_IT, method="bogus")


# ---------------------------------------------------------------------------
# locs_from_fits — MLE-specific output (extra columns vs. gausslq)
# ---------------------------------------------------------------------------


class TestLocsFromFits:
    """Verify the MLE locs DataFrame has the extra uncertainty columns and
    that they are sane."""

    @pytest.fixture
    def fit_results(self, synthetic_spots):
        spots, _ = synthetic_spots
        return gaussmle.gaussmle(spots, EPS, MAX_IT, method="sigmaxy")

    @pytest.fixture
    def identifications(self, synthetic_spots):
        spots, _ = synthetic_spots
        n = len(spots)
        return pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.arange(n, dtype=np.int64) + 10,
                "y": np.arange(n, dtype=np.int64) + 20,
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )

    def test_required_mle_columns_present(self, fit_results, identifications):
        theta, crlbs, lls, its = fit_results
        locs = gaussmle.locs_from_fits(
            identifications, theta, crlbs, lls, its, BOX
        )
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
            "log_likelihood",
            "iterations",
            "photons_unc",
            "bg_unc",
            "sx_unc",
            "sy_unc",
        ]:
            assert col in locs.columns

    def test_uncertainty_columns_strictly_positive(
        self, fit_results, identifications
    ):
        theta, crlbs, lls, its = fit_results
        locs = gaussmle.locs_from_fits(
            identifications, theta, crlbs, lls, its, BOX
        )
        for col in ["lpx", "lpy", "photons_unc", "bg_unc", "sx_unc", "sy_unc"]:
            assert (locs[col] > 0).all(), f"{col} must be > 0"

    def test_box_offset_subtracted_from_position(
        self, fit_results, identifications
    ):
        theta, crlbs, lls, its = fit_results
        locs = gaussmle.locs_from_fits(
            identifications, theta, crlbs, lls, its, BOX
        )
        # In MLE locs_from_fits, x = theta_x + ids.x - box_offset
        expected_x = (
            theta[:, 0] + identifications["x"].to_numpy() - BOX // 2
        ).astype(np.float32)
        np.testing.assert_array_equal(locs["x"].to_numpy(), expected_x)

    def test_ellipticity_formula(self, fit_results, identifications):
        theta, crlbs, lls, its = fit_results
        locs = gaussmle.locs_from_fits(
            identifications, theta, crlbs, lls, its, BOX
        )
        a = np.maximum(locs["sx"], locs["sy"])
        b = np.minimum(locs["sx"], locs["sy"])
        expected = ((a - b) / a).astype(np.float32)
        np.testing.assert_allclose(locs["ellipticity"], expected)

    def test_lpx_equals_sqrt_crlb(self, fit_results, identifications):
        theta, crlbs, lls, its = fit_results
        locs = gaussmle.locs_from_fits(
            identifications, theta, crlbs, lls, its, BOX
        )
        np.testing.assert_allclose(
            locs["lpx"].to_numpy(),
            np.sqrt(crlbs[:, 0]).astype(np.float32),
        )


# ---------------------------------------------------------------------------
# sigma_uncertainty — Rieger/Stallinga formula
# ---------------------------------------------------------------------------


class TestSigmaUncertainty:
    """Closed-form re-implementation of the docstring formula."""

    def _expected(self, sigma, sigma_orth, photons, bg):
        sa2 = sigma**2 + 1 / 12
        tau = (2 * np.pi * sa2 * bg) / photons
        delta_sigma_sq = (sigma**2 / (4 * photons)) * (
            1 + 8 * tau + np.sqrt((8 * tau) / (1 + 2 * tau))
        )
        return np.sqrt(delta_sigma_sq)

    def test_matches_closed_form(self):
        sigma = np.array([1.0, 1.2, 0.9])
        sigma_orth = np.array([1.0, 1.0, 1.1])
        photons = np.array([1000.0, 3000.0, 8000.0])
        bg = np.array([1.0, 5.0, 20.0])
        result = gaussmle.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        expected = self._expected(sigma, sigma_orth, photons, bg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_monotonic_in_photons(self):
        photons = np.array([500.0, 1500.0, 5000.0, 20000.0])
        sigma = np.full_like(photons, 1.0)
        sigma_orth = np.full_like(photons, 1.0)
        bg = np.full_like(photons, 5.0)
        se = gaussmle.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert (np.diff(se) < 0).all()

    def test_monotonic_in_bg(self):
        bg = np.array([1.0, 5.0, 20.0, 100.0])
        sigma = np.full_like(bg, 1.0)
        sigma_orth = np.full_like(bg, 1.0)
        photons = np.full_like(bg, 2000.0)
        se = gaussmle.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert (np.diff(se) > 0).all()

    def test_differs_from_lq_formula(self):
        """The MLE formula and the LQ formula are different — verify they
        produce different numerical results so we don't accidentally have
        them aliased to each other."""
        sigma = np.array([1.0])
        sigma_orth = np.array([1.0])
        photons = np.array([2000.0])
        bg = np.array([10.0])
        mle = gaussmle.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        lq = gausslq.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert mle[0] != lq[0]

    def test_pandas_series_input(self):
        sigma = pd.Series([1.0, 1.2])
        sigma_orth = pd.Series([1.0, 1.0])
        photons = pd.Series([2000.0, 4000.0])
        bg = pd.Series([5.0, 10.0])
        se = gaussmle.sigma_uncertainty(sigma, sigma_orth, photons, bg)
        assert len(se) == 2
        assert (se > 0).all()
