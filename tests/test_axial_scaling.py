"""picasso.axial_scaling: the magnification factor of z under a
refractive-index mismatch (Loginov et al., Optica 2024).

The values are checked against those printed in the paper and its
supplement, against the authors' code (hoogenboom-group/SF on GitHub,
whose re-scaling function is transcribed in ``_authors_factor``) and
against Eq. 13 of the paper, the same relation written for the actual
depth, which is solved numerically here.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from picasso import axial_scaling

#: (NA, n_immersion, n_sample) of the paper's figures and common setups
SETUPS = [
    (0.85, 1.0, 1.336),  # air into water, Fig. 6 and S12
    (0.95, 1.0, 1.33),  # Fig. 5a
    (0.7, 1.0, 1.52),  # Fig. S3
    (1.4, 1.52, 1.336),  # oil into water, Fig. 7 and S13
    (1.49, 1.518, 1.33),  # TIRF objective into water
    (1.25, 1.33, 1.52),  # water objective into oil
]
METHODS = axial_scaling.NEAR_SURFACE_METHODS


def _kwargs(na, n1, n2, **extra):
    return dict(na=na, n_immersion=n1, n_sample=n2, **extra)


def _authors_factor(z_n, na, n1, n2, wavelength, method):
    """``scaling_factor_from_nfp`` of the authors' ``utils.py``,
    transcribed: Eq. 17 with complex square roots and the near-surface
    value as the bound."""
    m = np.emath.sqrt(n2**2 - n1**2)
    delta = (-1 if n2 < n1 else 1) * (wavelength / 4) / (n1 * z_n)
    univ = n2 / n1 * (1 + delta) + m / n1 * np.emath.sqrt(delta * (2 + delta))
    if method == "midpoint":
        top = n1 + np.sqrt(n1**2 - na**2)
        crit = np.real(np.emath.sqrt(4 * (n2**2 - n1**2) + top**2)) / top
    else:
        crit = (n1 - np.emath.sqrt(n1**2 - na**2)) / (
            n2 - np.emath.sqrt(n2**2 - na**2)
        )
    bound = np.maximum if n2 < n1 else np.minimum
    return bound(np.real(univ), np.real(crit))


def _factor_of_actual_depth(z_a, n1, n2, wavelength, crit):
    """Eqs. 3 and 13 of the paper: the factor as a function of the actual
    depth ``z_a``."""
    s = 1 if n2 >= n1 else -1
    eps = s * (wavelength / 4) / (z_a * n2)
    m = np.emath.sqrt(n2**2 - n1**2)
    den = 1 - (n2 / n1) ** 2 * eps * (2 - eps)
    univ = (
        n2 / n1 * ((1 - eps) + m / n1 * np.emath.sqrt(eps * (2 - eps))) / den
    ).real
    if s > 0:
        # past its pole Eq. 13 has broken down: the near-surface value
        return crit if den <= 0 or univ <= 0 else min(univ, crit)
    return max(univ, crit)


class TestNearSurfaceFactor:
    @pytest.mark.parametrize(
        "na,n1,n2,method,expected,tolerance",
        [
            # "the depth-dependent re-scaling factor is 1.532 close to the
            # coverslip" (NA 0.85, air into agarose, Lyakin's value)
            (0.85, 1.0, 1.336, "midpoint", 1.532, 5e-4),
            # Fig. S13a, Lyakin's value
            (1.4, 1.52, 1.336, "midpoint", 0.727, 5e-4),
            # "the analytical expression provides a plateau ... 0.64"
            (1.4, 1.5197, 1.3421, "full-aperture", 0.64, 5e-3),
            # Fig. S11 (NA > n_sample: the real part of Eq. 15)
            (1.4, 1.52, 1.0, "full-aperture", 0.4735, 5e-4),
        ],
    )
    def test_values_of_the_paper(
        self, na, n1, n2, method, expected, tolerance
    ):
        factor = axial_scaling.near_surface_factor(
            **_kwargs(na, n1, n2, near_surface=method)
        )
        assert factor == pytest.approx(expected, abs=tolerance)

    def test_midpoint_is_the_default(self):
        # a TIRF objective from oil into water
        kwargs = _kwargs(1.49, 1.518, 1.33)
        factor = axial_scaling.near_surface_factor(**kwargs)
        assert factor == axial_scaling.near_surface_factor(
            **kwargs, near_surface="midpoint"
        )
        assert factor == pytest.approx(0.5873, abs=1e-4)
        # the full-aperture value is the alternative
        assert axial_scaling.near_surface_factor(
            **kwargs, near_surface="full-aperture"
        ) == pytest.approx(0.7355, abs=1e-4)

    @pytest.mark.parametrize("na,n1,n2", SETUPS)
    @pytest.mark.parametrize("method", METHODS)
    def test_lies_beyond_the_paraxial_value(self, na, n1, n2, method):
        factor = axial_scaling.near_surface_factor(
            **_kwargs(na, n1, n2, near_surface=method)
        )
        if n2 > n1:
            assert factor > n2 / n1
        else:
            assert factor < n2 / n1

    @pytest.mark.parametrize("method", METHODS)
    def test_equal_indices_need_no_correction(self, method):
        factor = axial_scaling.near_surface_factor(
            **_kwargs(1.2, 1.33, 1.33, near_surface=method)
        )
        assert factor == pytest.approx(1.0)

    @pytest.mark.parametrize("na,n1,n2", [(1.4, 1.52, 1.0), (1.7, 1.78, 1.33)])
    def test_midpoint_ray_totally_internally_reflected(self, na, n1, n2):
        with pytest.raises(ValueError, match="totally internally reflected"):
            axial_scaling.near_surface_factor(
                **_kwargs(na, n1, n2, near_surface="midpoint")
            )
        # the full-aperture estimate exists
        factor = axial_scaling.near_surface_factor(
            **_kwargs(na, n1, n2, near_surface="full-aperture")
        )
        assert 0 < factor < n2 / n1

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (_kwargs(1.52, 1.518, 1.33), "smaller than the refractive index"),
            (_kwargs(0.0, 1.518, 1.33), "na must be positive"),
            (_kwargs(1.4, 1.518, -1.0), "n_sample must be positive"),
            (_kwargs(1.4, math.nan, 1.33), "n_immersion must be positive"),
            (_kwargs(1.4, 1.518, 1.33, near_surface="lyakin"), "one of"),
        ],
    )
    def test_invalid_input(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            axial_scaling.near_surface_factor(**kwargs)


class TestMagnificationFactor:
    @pytest.mark.parametrize("na,n1,n2", SETUPS)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("wavelength", [510.0, 670.0])
    def test_matches_the_authors_code(self, na, n1, n2, method, wavelength):
        depths = np.geomspace(1.0, 2e5, 300)
        factor = axial_scaling.magnification_factor(
            depths,
            **_kwargs(na, n1, n2, wavelength=wavelength, near_surface=method),
        )
        np.testing.assert_allclose(
            factor,
            _authors_factor(depths, na, n1, n2, wavelength, method),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("na,n1,n2", SETUPS)
    @pytest.mark.parametrize("method", METHODS)
    def test_matches_eq_13_for_the_actual_depth(self, na, n1, n2, method):
        """The factor at nominal depth z_N equals the factor of Eq. 13 at
        the actual depth z_A = factor * z_N."""
        wavelength = 600.0
        kwargs = _kwargs(
            na, n1, n2, wavelength=wavelength, near_surface=method
        )
        crit = axial_scaling.near_surface_factor(
            **_kwargs(na, n1, n2, near_surface=method)
        )
        for z_a in np.geomspace(20.0, 1e5, 40):

            def nominal_minus(z_n):
                return (
                    z_a
                    / _factor_of_actual_depth(z_a, n1, n2, wavelength, crit)
                    - z_n
                )

            z_n = brentq(nominal_minus, 1e-3, 1e7, xtol=1e-12)
            assert axial_scaling.actual_depth(z_n, **kwargs) == pytest.approx(
                z_a, rel=1e-9
            )

    def test_depth_dependence_of_the_paper(self):
        # Fig. S13 (Lyakin's near-surface value): oil into water, 520 nm;
        # the factor passes Diel's median (0.843) at an actual depth of
        # about 30 um and is 0.86 at 100 um
        kwargs = _kwargs(
            1.4, 1.52, 1.336, wavelength=520.0, near_surface="midpoint"
        )
        z_n = np.linspace(100.0, 120_000.0, 20_000)
        z_a = axial_scaling.actual_depth(z_n, **kwargs)
        factor = z_a / z_n
        median = math.sqrt((1.336**2 - 1.4**2 / 4) / (1.52**2 - 1.4**2 / 4))
        assert z_a[np.argmax(factor > median)] == pytest.approx(
            30_000, rel=0.05
        )
        assert np.interp(100_000, z_a, factor) == pytest.approx(0.86, abs=5e-3)

    @pytest.mark.parametrize("na,n1,n2", SETUPS)
    def test_approaches_the_paraxial_value(self, na, n1, n2):
        factor = axial_scaling.magnification_factor(
            1e9, **_kwargs(na, n1, n2, wavelength=600.0)
        )
        assert factor == pytest.approx(n2 / n1, rel=1e-3)

    @pytest.mark.parametrize("na,n1,n2", SETUPS)
    def test_is_monotonic_in_the_depth(self, na, n1, n2):
        depths = np.geomspace(1.0, 1e7, 2000)
        factor = axial_scaling.magnification_factor(
            depths, **_kwargs(na, n1, n2, wavelength=600.0)
        )
        steps = np.diff(factor)
        assert np.all(steps <= 0) if n2 > n1 else np.all(steps >= 0)

    def test_coverslip_is_the_near_surface_value(self):
        kwargs = _kwargs(1.49, 1.518, 1.33)
        assert axial_scaling.magnification_factor(
            0.0, **kwargs, wavelength=670.0
        ) == axial_scaling.near_surface_factor(**kwargs)

    def test_scalar_and_array_inputs(self):
        kwargs = _kwargs(1.4, 1.515, 1.35, wavelength=670.0)
        assert isinstance(
            axial_scaling.magnification_factor(5000, **kwargs), float
        )
        assert isinstance(
            axial_scaling.magnification_factor(np.float32(5000), **kwargs),
            float,
        )
        grid = axial_scaling.magnification_factor(
            np.full((2, 3), 5000.0), **kwargs
        )
        assert grid.shape == (2, 3)
        assert np.all(
            grid == axial_scaling.magnification_factor(5000, **kwargs)
        )
        assert isinstance(axial_scaling.actual_depth(5000, **kwargs), float)
        np.testing.assert_allclose(
            axial_scaling.actual_depth([0, 5000], **kwargs),
            [0, 5000 * grid[0, 0]],
        )

    def test_equal_indices(self):
        factor = axial_scaling.magnification_factor(
            [0, 10, 1e4], **_kwargs(1.2, 1.33, 1.33, wavelength=600.0)
        )
        np.testing.assert_allclose(factor, 1.0)

    @pytest.mark.parametrize(
        "depth,wavelength,match",
        [
            (-1.0, 600.0, "depth must be finite"),
            ([0.0, math.inf], 600.0, "depth must be finite"),
            (100.0, 0.0, "wavelength must be positive"),
        ],
    )
    def test_invalid_input(self, depth, wavelength, match):
        with pytest.raises(ValueError, match=match):
            axial_scaling.magnification_factor(
                depth, **_kwargs(1.4, 1.518, 1.33, wavelength=wavelength)
            )


class TestNearSurfaceDepth:
    @pytest.mark.parametrize("na,n1,n2", SETUPS)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("wavelength", [510.0, 670.0])
    def test_the_factor_changes_beyond_it(
        self, na, n1, n2, method, wavelength
    ):
        kwargs = _kwargs(
            na, n1, n2, wavelength=wavelength, near_surface=method
        )
        depth = axial_scaling.near_surface_depth(**kwargs)
        crit = axial_scaling.near_surface_factor(
            **_kwargs(na, n1, n2, near_surface=method)
        )
        within, beyond = axial_scaling.magnification_factor(
            [depth * (1 - 1e-6), depth * (1 + 1e-6)], **kwargs
        )
        assert within == crit
        assert beyond != crit
        assert beyond == pytest.approx(crit, rel=1e-5)

    def test_values_of_the_paper(self):
        # Fig. S12: Lyakin's near-surface value of air into water (NA 0.85,
        # 520 nm) holds up to an actual depth of about 11 um
        kwargs = _kwargs(
            0.85, 1.0, 1.336, wavelength=520.0, near_surface="midpoint"
        )
        depth = axial_scaling.near_surface_depth(**kwargs)
        assert axial_scaling.actual_depth(depth, **kwargs) == pytest.approx(
            10_760, rel=1e-3
        )

    def test_scales_with_the_wavelength(self):
        kwargs = _kwargs(1.49, 1.518, 1.33)
        assert axial_scaling.near_surface_depth(
            **kwargs, wavelength=700.0
        ) == pytest.approx(
            2 * axial_scaling.near_surface_depth(**kwargs, wavelength=350.0)
        )

    def test_equal_indices(self):
        assert (
            axial_scaling.near_surface_depth(
                **_kwargs(1.2, 1.33, 1.33, wavelength=600.0)
            )
            == math.inf
        )
