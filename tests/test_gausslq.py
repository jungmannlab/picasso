"""Test ``picasso.gausslq`` — least-squares 2D Gaussian fitting.

Uses synthetic Gaussian spots with known ground truth (see
``tests/conftest.py``) so assertions can verify numerical correctness,
not just shapes.

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from picasso import gausslq, localize
from picasso.fitting import precision

from tests.conftest import BOX, make_rotated_gaussian_spot


def _precision_matrix(sx, sy, angle):
    """Precision (inverse-covariance) matrix of a rotated Gaussian.

    ``(sx, sy, angle)`` and ``(sy, sx, angle +/- pi/2)`` describe the *same*
    ellipse — the rotated fit is free to return either. The precision
    matrix is invariant under that relabeling, so comparing it (rather than
    the raw angle/widths) tests recovery without tripping on the
    degeneracy.
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([1.0 / sx**2, 1.0 / sy**2])
    return R.T @ D @ R


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
        number of spots fitted so far."""
        spots, _ = synthetic_spots
        calls = []
        gausslq.fit_spots(spots, progress_callback=calls.append)
        assert len(calls) == len(spots)
        # the last call reports all spots done
        assert calls == list(range(1, len(spots) + 1))


class TestConvergenceSchedule:
    """``tolerance``/``max_iterations`` are exposed so the GUI and the CLI can
    offer them for every fitting method, this one included."""

    def test_defaults_reproduce_minpacks_own(self):
        """MAX_ITERATIONS must map to exactly ``leastsq``'s default
        ``maxfev``, so passing the default explicitly is a no-op and the
        historical behavior of ``gausslq`` is unchanged."""
        for n_parameters in (5, 6, 7):
            assert gausslq._max_function_evaluations(
                gausslq.MAX_ITERATIONS, n_parameters
            ) == 200 * (n_parameters + 1)

    def test_defaults_match_passing_none(self, synthetic_spots):
        spots, _ = synthetic_spots
        np.testing.assert_array_equal(
            gausslq.fit_spots(spots),
            gausslq.fit_spots(
                spots,
                tolerance=gausslq.TOLERANCE,
                max_iterations=gausslq.MAX_ITERATIONS,
            ),
        )

    def test_iteration_cap_bites(self, synthetic_spots):
        """One iteration cannot reach the optimum a full fit finds."""
        spots, gt = synthetic_spots
        converged = gausslq.fit_spots(spots)
        starved = gausslq.fit_spots(spots, max_iterations=1)
        assert not np.allclose(converged, starved)
        # ...and it is the starved fit that is wrong, not merely different
        assert np.max(np.abs(starved[:, 0] - gt.x.values)) > np.max(
            np.abs(converged[:, 0] - gt.x.values)
        )

    def test_tighter_tolerance_does_not_move_a_converged_fit(
        self, synthetic_spots
    ):
        """The default stop is already at the optimum for clean spots, so
        tightening it changes the position by far less than a pixel."""
        spots, _ = synthetic_spots
        loose = gausslq.fit_spots(spots)
        tight = gausslq.fit_spots(spots, tolerance=1e-8, max_iterations=400)
        assert np.max(np.abs(loose[:, :2] - tight[:, :2])) < 0.05

    @pytest.mark.parametrize(
        "kwargs", [{"spherical": True}, {"rotated": True}, {}]
    )
    def test_every_model_accepts_the_schedule(self, synthetic_spots, kwargs):
        spots, _ = synthetic_spots
        theta = gausslq.fit_spots(
            spots[:3], tolerance=1e-3, max_iterations=50, **kwargs
        )
        assert np.all(np.isfinite(theta))


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
# Spherical (isotropic, single-width) least-squares fit
# ---------------------------------------------------------------------------


class TestFitSpotSpherical:
    """``gausslq.fit_spot(spherical=True)`` fits one shared width."""

    def test_returns_six_floats_with_equal_widths(
        self, synthetic_spot_factory
    ):
        """The spherical fit still returns the standard 6-parameter layout
        ``[x, y, photons, bg, sx, sy]`` with ``sx == sy`` so downstream
        code is unchanged."""
        spot = synthetic_spot_factory(sx=1.1, sy=1.1)
        result = gausslq.fit_spot(spot, spherical=True)
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))
        assert result[4] == result[5]

    def test_recovers_isotropic_ground_truth(self, synthetic_spot_factory):
        """A noiseless isotropic spot is recovered to tight tolerance."""
        spot = synthetic_spot_factory(
            x0=0.2, y0=-0.15, sx=1.2, sy=1.2, photons=5000.0, bg=10.0
        )
        x, y, photons, bg, sx, sy = gausslq.fit_spot(spot, spherical=True)
        assert x == pytest.approx(0.2, abs=5e-3)
        assert y == pytest.approx(-0.15, abs=5e-3)
        assert sx == pytest.approx(1.2, abs=5e-3)
        assert sy == pytest.approx(1.2, abs=5e-3)
        assert photons == pytest.approx(5000.0, rel=5e-3)
        assert bg == pytest.approx(10.0, rel=5e-2)

    def test_single_width_averages_anisotropic_spot(
        self, synthetic_spot_factory
    ):
        """Given an anisotropic spot, the single-width fit lands between
        the two true widths (it cannot represent sx != sy)."""
        spot = synthetic_spot_factory(sx=1.4, sy=0.9)
        _, _, _, _, sx, sy = gausslq.fit_spot(spot, spherical=True)
        assert sx == sy
        assert 0.9 < sx < 1.4


class TestFitSpotsSpherical:
    """Batch spherical fitting via ``gausslq.fit_spots``."""

    def test_shape_and_equal_widths(self, synthetic_spots_isotropic):
        spots, _ = synthetic_spots_isotropic
        theta = gausslq.fit_spots(spots, spherical=True)
        assert theta.shape == (len(spots), 6)
        assert theta.dtype == np.float32
        assert np.all(np.isfinite(theta))
        np.testing.assert_array_equal(theta[:, 4], theta[:, 5])

    def test_recovers_ground_truth(self, synthetic_spots_isotropic):
        spots, gt = synthetic_spots_isotropic
        theta = gausslq.fit_spots(spots, spherical=True)
        # theta cols: x, y, photons, bg, sx, sy
        np.testing.assert_allclose(theta[:, 0], gt.x.values, atol=0.02)
        np.testing.assert_allclose(theta[:, 1], gt.y.values, atol=0.02)
        np.testing.assert_allclose(theta[:, 2], gt.photons.values, rtol=0.02)
        np.testing.assert_allclose(theta[:, 4], gt.sx.values, atol=0.02)

    def test_per_spot_matches_scalar(self, synthetic_spots_isotropic):
        spots, _ = synthetic_spots_isotropic
        batch = gausslq.fit_spots(spots, spherical=True)
        for i in [0, 5, len(spots) - 1]:
            single = gausslq.fit_spot(spots[i], spherical=True)
            np.testing.assert_allclose(batch[i], single, atol=1e-5)


# ---------------------------------------------------------------------------
# Rotated elliptical least-squares fit
# ---------------------------------------------------------------------------


class TestFitSpotRotated:
    """``gausslq.fit_spot(rotated=True)`` recovers the orientation."""

    def test_returns_seven_floats(self):
        spot = make_rotated_gaussian_spot(
            9, 0.1, -0.1, 1.6, 0.9, 6000.0, 10.0, 0.4
        )
        result = gausslq.fit_spot(spot, rotated=True)
        assert result.shape == (7,)
        assert np.all(np.isfinite(result))

    def test_recovers_ellipse_precision_matrix(self):
        """The recovered ellipse (precision matrix) matches ground truth,
        independent of the sx<->sy / angle+-pi/2 relabeling."""
        box = 9
        for angle in [-1.2, -0.6, -0.2, 0.3, 0.7, 1.1]:
            spot = make_rotated_gaussian_spot(
                box, 0.15, -0.1, 1.7, 0.9, 6000.0, 10.0, angle
            )
            _, _, _, _, sx, sy, ang = gausslq.fit_spot(spot, rotated=True)
            np.testing.assert_allclose(
                _precision_matrix(sx, sy, ang),
                _precision_matrix(1.7, 0.9, angle),
                atol=5e-3,
            )
            # sorted widths recovered regardless of axis labeling
            np.testing.assert_allclose(sorted([sx, sy]), [0.9, 1.7], atol=1e-2)

    def test_recovers_position_and_photons(self):
        spot = make_rotated_gaussian_spot(
            9, 0.25, -0.2, 1.6, 0.95, 7000.0, 12.0, 0.5
        )
        x, y, photons, bg, _, _, _ = gausslq.fit_spot(spot, rotated=True)
        assert x == pytest.approx(0.25, abs=1e-2)
        assert y == pytest.approx(-0.2, abs=1e-2)
        assert photons == pytest.approx(7000.0, rel=1e-2)
        assert bg == pytest.approx(12.0, rel=5e-2)


class TestFitSpotsRotated:
    """Batch rotated fitting via ``gausslq.fit_spots``."""

    def test_shape_is_seven_columns(self, synthetic_spots_rotated):
        spots, _ = synthetic_spots_rotated
        theta = gausslq.fit_spots(spots, rotated=True)
        assert theta.shape == (len(spots), 7)
        assert np.all(np.isfinite(theta))

    def test_recovers_each_ellipse(self, synthetic_spots_rotated):
        spots, gt = synthetic_spots_rotated
        theta = gausslq.fit_spots(spots, rotated=True)
        for i in range(len(spots)):
            np.testing.assert_allclose(
                _precision_matrix(theta[i, 4], theta[i, 5], theta[i, 6]),
                _precision_matrix(gt.sx[i], gt.sy[i], gt.angle[i]),
                atol=1e-2,
            )


# ---------------------------------------------------------------------------
# locs_from_fits — spherical (no ellipticity) and rotated (angle column)
# ---------------------------------------------------------------------------


class TestLocsFromFitsSphericalRotated:
    """The ``spherical`` flag drops the (always-zero) ellipticity column;
    a 7-column theta adds the ``angle`` column."""

    def _ids(self, n):
        return pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 16, dtype=np.int64),
                "y": np.full(n, 16, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )

    def test_spherical_omits_ellipticity(self, synthetic_spots_isotropic):
        spots, _ = synthetic_spots_isotropic
        theta = gausslq.fit_spots(spots, spherical=True)
        ids = self._ids(len(spots))
        locs = gausslq.locs_from_fits(
            ids, theta, BOX, em=False, spherical=True
        )
        assert "ellipticity" not in locs.columns
        # everything else is still present
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
            "net_gradient",
        ]:
            assert col in locs.columns

    def test_non_spherical_keeps_ellipticity(self, synthetic_spots_isotropic):
        """Default (spherical=False) still emits the ellipticity column,
        so only the spherical path drops it."""
        spots, _ = synthetic_spots_isotropic
        theta = gausslq.fit_spots(spots, spherical=True)
        ids = self._ids(len(spots))
        locs = gausslq.locs_from_fits(
            ids, theta, BOX, em=False, spherical=False
        )
        assert "ellipticity" in locs.columns
        # sx == sy -> ellipticity is (numerically) zero
        np.testing.assert_allclose(locs["ellipticity"], 0.0, atol=1e-6)

    def test_spherical_flag_only_changes_ellipticity(
        self, synthetic_spots_isotropic
    ):
        """Dropping ellipticity must not perturb any other column."""
        spots, _ = synthetic_spots_isotropic
        theta = gausslq.fit_spots(spots, spherical=True)
        ids = self._ids(len(spots))
        sph = gausslq.locs_from_fits(ids, theta, BOX, em=False, spherical=True)
        ell = gausslq.locs_from_fits(
            ids, theta, BOX, em=False, spherical=False
        )
        for col in sph.columns:
            np.testing.assert_array_equal(
                sph[col].to_numpy(), ell[col].to_numpy()
            )
        assert set(ell.columns) - set(sph.columns) == {"ellipticity"}

    def test_rotated_adds_normalized_angle_column(
        self, synthetic_spots_rotated
    ):
        spots, _ = synthetic_spots_rotated
        theta = gausslq.fit_spots(spots, rotated=True)
        ids = self._ids(len(spots))
        locs = gausslq.locs_from_fits(ids, theta, BOX, em=False)
        assert "angle" in locs.columns
        # angle stored in degrees, wrapped to [-90, 90)
        assert ((locs["angle"] >= -90.0) & (locs["angle"] < 90.0)).all()
        # ellipticity is present for the (anisotropic) rotated model
        assert "ellipticity" in locs.columns


# ---------------------------------------------------------------------------
# chi-square — the least-squares goodness-of-fit metric
# ---------------------------------------------------------------------------


class TestChiSquare:
    """``return_chi_square`` appends the residual sum of squares at the fit
    optimum. It is the least-squares counterpart of the MLE fits'
    ``log_likelihood``: least squares assumes no noise model, so it has no
    likelihood, but it does have the objective value it minimized."""

    @staticmethod
    def _ids(n):
        return pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 40, dtype=np.int64),
                "y": np.full(n, 60, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )

    @pytest.mark.parametrize(
        "kwargs, n_params",
        [({}, 6), ({"spherical": True}, 6), ({"rotated": True}, 7)],
    )
    def test_fit_spot_appends_one_column(
        self, synthetic_spot_factory, kwargs, n_params
    ):
        """One extra trailing element, and the parameters themselves are
        untouched by asking for it."""
        spot = synthetic_spot_factory()
        plain = gausslq.fit_spot(spot, **kwargs)
        with_chi = gausslq.fit_spot(spot, return_chi_square=True, **kwargs)
        assert plain.shape == (n_params,)
        assert with_chi.shape == (n_params + 1,)
        np.testing.assert_allclose(with_chi[:n_params], plain, rtol=1e-6)
        assert with_chi[-1] >= 0

    def test_matches_residual_sum_of_squares(self, synthetic_spot_factory):
        """The reported value is the sum of squared residuals of the fitted
        model against the spot — recomputed here from the model directly."""
        spot = synthetic_spot_factory()
        theta = gausslq.fit_spot(spot, return_chi_square=True)
        x, y, photons, bg, sx, sy = theta[:6].astype(np.float64)
        half = BOX // 2
        grid = np.arange(-half, half + 1, dtype=np.float64)

        # The fit model is a point-sampled Gaussian PDF, separable in x and y
        # (mirrors gausslq._compute_model / _gaussian).
        def pdf(mu, sigma):
            norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
            return norm * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)

        model = photons * np.outer(pdf(y, sy), pdf(x, sx)) + bg
        expected = np.sum((np.asarray(spot, dtype=np.float64) - model) ** 2)
        np.testing.assert_allclose(theta[-1], expected, rtol=1e-3)

    def test_fit_spots_batch_column(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta = gausslq.fit_spots(spots, return_chi_square=True)
        assert theta.shape == (len(spots), 7)
        assert np.all(theta[:, -1] >= 0)
        assert np.all(np.isfinite(theta[:, -1]))
        plain = gausslq.fit_spots(spots)
        np.testing.assert_allclose(theta[:, :6], plain, rtol=1e-5, atol=1e-5)

    def test_scales_with_noise(self, synthetic_spots, synthetic_spots_noisy):
        """A noisier spot stack leaves larger residuals, so the chi-square
        must be able to tell the two apart (that is the point of saving it)."""
        clean, _ = synthetic_spots
        noisy, _ = synthetic_spots_noisy
        chi_clean = gausslq.fit_spots(clean, return_chi_square=True)[:, -1]
        chi_noisy = gausslq.fit_spots(noisy, return_chi_square=True)[:, -1]
        assert np.median(chi_noisy) > np.median(chi_clean)

    def test_locs_from_fits_adds_column(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta = gausslq.fit_spots(spots, return_chi_square=True)
        theta, chi_square = theta[:, :-1], theta[:, -1]
        ids = self._ids(len(spots))
        locs = gausslq.locs_from_fits(
            ids, theta, BOX, em=False, chi_square=chi_square
        )
        assert locs["chi_square"].dtype == np.float32
        # locs_from_fits sorts by frame; ids are already frame-ordered here
        np.testing.assert_allclose(
            locs["chi_square"].to_numpy(), chi_square, rtol=1e-6
        )

    def test_locs_from_fits_omits_column_by_default(self, synthetic_spots):
        spots, _ = synthetic_spots
        theta = gausslq.fit_spots(spots)
        locs = gausslq.locs_from_fits(self._ids(len(spots)), theta, BOX, False)
        assert "chi_square" not in locs.columns


class TestDeprecation:
    """``picasso.gausslq``'s fitters are deprecated in 0.11 and go in 1.0.

    Two things have to hold: the public names warn, and Picasso's own code
    does not trigger them - a library warning about its own internals is
    noise, not a signal. The internal callers use the private
    implementations (``_fit_spot`` and friends), which is what these tests
    pin."""

    ENTRY_POINTS = ["fit_spot", "fit_spots", "fit_spots_parallel"]

    @pytest.mark.parametrize("name", ENTRY_POINTS + ["fit_spots_gauss_gpu"])
    def test_documented_as_deprecated(self, name):
        doc = getattr(gausslq, name).__doc__
        assert ".. deprecated:: 0.11" in doc
        assert "Picasso 1.0" in doc

    def test_fit_spot_warns(self, synthetic_spots):
        spots, _ = synthetic_spots
        with pytest.warns(DeprecationWarning, match="Picasso 1.0"):
            gausslq.fit_spot(spots[0])

    def test_fit_spots_warns_once_not_per_spot(self, synthetic_spots):
        spots, _ = synthetic_spots
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gausslq.fit_spots(spots)
        deprecations = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecations) == 1

    @pytest.mark.parametrize("name", ["fit_spot", "fit_spots"])
    def test_private_implementation_is_silent(self, name, synthetic_spots):
        """What Picasso itself calls. If these warned, every ordinary fit
        would emit a deprecation notice about Picasso's own code.

        ``_fit_spots_parallel`` is left out only because it needs
        subprocesses; ``localize`` calling it is covered end to end by
        ``test_localize.TestNoSelfDeprecation``."""
        spots, _ = synthetic_spots
        private = getattr(gausslq, "_" + name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            private(spots[0] if name == "fit_spot" else spots)
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_public_and_private_agree(self, name, synthetic_spots):
        """The wrapper must add a warning and nothing else."""
        spots, _ = synthetic_spots
        public = getattr(gausslq, name)
        private = getattr(gausslq, "_" + name)
        if name == "fit_spot":
            args = (spots[0],)
        elif name == "fit_spots":
            args = (spots,)
        else:
            pytest.skip("fit_spots_parallel needs subprocesses")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            np.testing.assert_array_equal(public(*args), private(*args))

    def test_private_locs_from_fits_is_silent(self, synthetic_spots):
        """``_locs_from_fits`` computes ``lpx``/``lpy``, so it must reach the
        *new* home of ``localization_precision`` rather than the shim beside
        it. It did not, once - and every least-squares fit warned about
        Picasso's own code as a result."""
        spots, _ = synthetic_spots
        theta = gausslq._fit_spots(spots)
        identifications = pd.DataFrame(
            {
                "frame": np.zeros(len(spots), np.uint32),
                "x": np.full(len(spots), 10.0),
                "y": np.full(len(spots), 10.0),
                "net_gradient": np.full(len(spots), 5000.0),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gausslq._locs_from_fits(identifications, theta, BOX, em=False)
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]

    def test_replacement_is_named(self):
        """A deprecation without a migration path is just an annoyance."""
        assert "picasso.fitting.gaussfit" in gausslq._DEPRECATION_MESSAGE

    @pytest.mark.parametrize(
        "name",
        [
            "fit_spot",
            "fit_spots",
            "fit_spots_parallel",
            "fit_spots_gauss_gpu",
            "fits_from_futures",
            "locs_from_fits",
            "locs_from_fits_gauss_gpu",
            "localization_precision",
            "sigma_uncertainty",
        ],
    )
    def test_every_public_name_is_deprecated(self, name):
        """The *whole module* goes in 1.0, not just the fitters - so nothing
        may be left here without a documented replacement."""
        doc = getattr(gausslq, name).__doc__ or ""
        assert ".. deprecated:: 0.11" in doc, name
        assert "Picasso 1.0" in doc, name

    def test_no_undeprecated_public_names_remain(self):
        """Catches a *new* public function being added to a module that is
        on its way out."""
        undeprecated = [
            name
            for name in dir(gausslq)
            if not name.startswith("_")
            and callable(getattr(gausslq, name))
            and getattr(getattr(gausslq, name), "__module__", "")
            == "picasso.gausslq"
            and ".. deprecated::" not in (getattr(gausslq, name).__doc__ or "")
        ]
        assert undeprecated == []

    def test_precision_formulas_moved_verbatim(self):
        """``localization_precision`` and ``sigma_uncertainty`` now live in
        ``picasso.fitting.precision``. The numbers must not have moved with
        them - these are published formulas, and ``lpx`` is in every saved
        localization file."""
        photons = np.array([500.0, 1200.0, 8000.0])
        s = np.array([1.2, 1.4, 0.9])
        s_orth = np.array([1.1, 1.5, 1.3])
        bg = np.array([5.0, 12.0, 2.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for em in (False, True):
                np.testing.assert_array_equal(
                    gausslq.localization_precision(photons, s, s_orth, bg, em),
                    precision.localization_precision(
                        photons, s, s_orth, bg, em
                    ),
                )
            np.testing.assert_array_equal(
                gausslq.sigma_uncertainty(s, s_orth, photons, bg),
                precision.sigma_uncertainty_lsq(s, s_orth, photons, bg),
            )

    def test_new_home_does_not_warn(self):
        """The replacement must be usable without tripping the warning the
        old name raises."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            precision.localization_precision(
                np.array([500.0]),
                np.array([1.2]),
                np.array([1.1]),
                np.array([5.0]),
                False,
            )
            precision.sigma_uncertainty_lsq(
                np.array([1.2]),
                np.array([1.1]),
                np.array([500.0]),
                np.array([5.0]),
            )
        assert not [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
