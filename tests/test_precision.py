"""
test_precision
~~~~~~~~~~~~~~

Tests for the closed-form precisions of ``picasso.fitting.precision``: the
localization precision of a least-squares Gaussian fit (Mortensen et al.,
2010) and the uncertainty of the fitted width, for least squares
(Kowalewski, Reinhardt, et al., 2026) and maximum likelihood (Rieger and
Stallinga, 2014).

Each formula is re-implemented here from its docstring, so a change to the
numbers is caught, not only a change of shape. The Fisher-matrix estimators
(``_gauss_crlb``, the spline bounds) are tested in ``test_localize.py``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from picasso.fitting import precision


class TestLocalizationPrecision:
    """Analytic checks against the Mortensen formula."""

    def test_no_bg_matches_shot_noise_term(self):
        """With bg=0, the formula reduces to ``sqrt(sa^2 * (16/9) /
        photons)`` where ``sa^2 = s^2 + 1/12``."""
        photons = np.array([1000.0, 5000.0])
        s = np.array([1.0, 1.2])
        s_orth = s.copy()
        bg = np.zeros_like(photons)
        result = precision.localization_precision(
            photons, s, s_orth, bg, em=False
        )
        sa2 = s**2 + 1 / 12
        expected = np.sqrt(sa2 * (16 / 9) / photons)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_higher_photons_gives_better_precision(self):
        result = precision.localization_precision(
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
        no_em = precision.localization_precision(
            photons, s, s_orth, bg, em=False
        )
        with_em = precision.localization_precision(
            photons, s, s_orth, bg, em=True
        )
        np.testing.assert_allclose(with_em / no_em, np.sqrt(2.0), rtol=1e-6)


class TestSigmaUncertaintyLsq:
    """The least-squares width uncertainty, Kowalewski, Reinhardt, et al.
    2026."""

    @staticmethod
    def _expected(sigma, sigma_orth, photons, bg):
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
        result = precision.sigma_uncertainty_lsq(
            sigma, sigma_orth, photons, bg
        )
        expected = self._expected(sigma, sigma_orth, photons, bg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_bg_simplifies(self):
        """At bg=0, only the shot-noise term remains: sqrt(sa^4 *
        (512/81) / photons / (4 sigma^2))."""
        result = precision.sigma_uncertainty_lsq(
            np.array([1.0]),
            np.array([1.0]),
            np.array([1000.0]),
            np.array([0.0]),
        )
        sa4 = (1.0 + 1 / 12) ** 2
        expected = np.sqrt(sa4 * (512 / 81) / 1000.0 / 4.0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_monotonic_in_photons(self):
        photons = np.array([500.0, 1500.0, 5000.0, 20000.0])
        ones = np.ones_like(photons)
        se = precision.sigma_uncertainty_lsq(ones, ones, photons, 5.0 * ones)
        assert (np.diff(se) < 0).all()

    def test_monotonic_in_bg(self):
        bg = np.array([0.0, 5.0, 20.0, 100.0])
        ones = np.ones_like(bg)
        se = precision.sigma_uncertainty_lsq(ones, ones, 2000.0 * ones, bg)
        assert (np.diff(se) > 0).all()

    def test_pandas_series_input(self):
        """zfit passes the columns of the localizations as they are."""
        se = precision.sigma_uncertainty_lsq(
            pd.Series([1.0, 1.2]),
            pd.Series([1.1, 1.0]),
            pd.Series([1000.0, 2000.0]),
            pd.Series([5.0, 10.0]),
        )
        assert len(se) == 2
        assert (se > 0).all()


class TestSigmaUncertaintyMle:
    """The maximum-likelihood width uncertainty, Rieger and Stallinga
    2014."""

    @staticmethod
    def _expected(sigma, photons, bg):
        """Direct re-implementation of the formula in the docstring."""
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
        result = precision.sigma_uncertainty_mle(
            sigma, sigma_orth, photons, bg
        )
        expected = self._expected(sigma, photons, bg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_monotonic_in_photons(self):
        photons = np.array([500.0, 1500.0, 5000.0, 20000.0])
        ones = np.ones_like(photons)
        se = precision.sigma_uncertainty_mle(ones, ones, photons, 5.0 * ones)
        assert (np.diff(se) < 0).all()

    def test_monotonic_in_bg(self):
        bg = np.array([1.0, 5.0, 20.0, 100.0])
        ones = np.ones_like(bg)
        se = precision.sigma_uncertainty_mle(ones, ones, 2000.0 * ones, bg)
        assert (np.diff(se) > 0).all()

    def test_pandas_series_input(self):
        se = precision.sigma_uncertainty_mle(
            pd.Series([1.0, 1.2]),
            pd.Series([1.0, 1.0]),
            pd.Series([2000.0, 4000.0]),
            pd.Series([5.0, 10.0]),
        )
        assert len(se) == 2
        assert (se > 0).all()


def test_the_two_sigma_uncertainties_are_different_formulas():
    """Same signature, different noise model, materially different
    answers - so the two must never be aliased to each other."""
    args = (
        np.array([1.3]),
        np.array([1.1]),
        np.array([800.0]),
        np.array([10.0]),
    )
    lsq = precision.sigma_uncertainty_lsq(*args)
    mle = precision.sigma_uncertainty_mle(*args)
    assert abs(lsq[0] - mle[0]) / mle[0] > 0.05
