"""Test ``picasso.zfit`` — astigmatic 3D fitting.

Covers the new (non-deprecated) API:

- numerical helpers (``__get_calib_size``, ``_get_prime_calib_size``,
  ``_interpolate_nan``, ``filter_z_fits``);
- the main ``zfit()`` pipeline (serial + multiprocess, with abort hooks
  and argument overrides);
- ``axial_localization_precision`` and ``axial_localization_precision_astig``
  (Kowalewski et al. 2026);
- ``calibrate_z`` driven by synthetic bead-stack data.

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

# Headless plotting for calibrate_z which calls plt.show()
matplotlib.use("Agg")

from picasso import zfit  # noqa: E402

from tests.conftest import CALIB_3D  # noqa: E402


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


class TestGetCalibSize:
    """Polynomial evaluation of the calibration curve."""

    def test_matches_numpy_polyval(self):
        """``_get_calib_size`` is a degree-6 polynomial; should equal
        ``np.polyval`` of the same coefficients."""
        coeffs = np.array(CALIB_3D["X Coefficients"])
        z = np.linspace(-300, 300, 21)
        result = zfit._get_calib_size(coeffs, z)
        expected = np.polyval(coeffs, z)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_at_zero_returns_constant_term(self):
        coeffs = np.array(CALIB_3D["X Coefficients"])
        assert zfit._get_calib_size(coeffs, 0.0) == coeffs[6]

    def test_vectorized_input(self):
        coeffs = np.array(CALIB_3D["X Coefficients"])
        z = np.array([-100.0, 0.0, 100.0])
        result = zfit._get_calib_size(coeffs, z)
        assert result.shape == z.shape


class TestGetPrimeCalibSize:
    """Derivative of the calibration polynomial."""

    def test_matches_polyder(self):
        """``_get_prime_calib_size`` should equal ``np.polyder`` of the
        same coefficients evaluated at the same z."""
        coeffs = np.array(CALIB_3D["X Coefficients"])
        z = np.linspace(-300, 300, 21)
        result = zfit._get_prime_calib_size(coeffs, z)
        expected = np.polyval(np.polyder(coeffs), z)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_for_constant_polynomial(self):
        coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5])
        z = np.linspace(-100, 100, 11)
        np.testing.assert_array_equal(
            zfit._get_prime_calib_size(coeffs, z), np.zeros_like(z)
        )

    def test_finite_difference_consistency(self):
        """Derivative at z must match a centered finite-difference of
        ``_get_calib_size``."""
        coeffs = np.array(CALIB_3D["X Coefficients"])
        z0 = 50.0
        h = 1e-3
        fd = (
            zfit._get_calib_size(coeffs, z0 + h)
            - zfit._get_calib_size(coeffs, z0 - h)
        ) / (2 * h)
        analytical = zfit._get_prime_calib_size(coeffs, z0)
        np.testing.assert_allclose(analytical, fd, rtol=1e-4)


class TestInterpolateNan:
    """Linear interpolation over NaN values."""

    def test_no_nans_identity(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(zfit._interpolate_nan(data.copy()), data)

    def test_interior_nans_filled(self):
        data = np.array([1.0, np.nan, 3.0])
        result = zfit._interpolate_nan(data.copy())
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_multiple_nans_filled(self):
        data = np.array([0.0, np.nan, np.nan, 3.0])
        result = zfit._interpolate_nan(data.copy())
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0])


class TestFilterZFits:
    """Residual-based filtering."""

    def _locs(self, d_zcalib_values):
        return pd.DataFrame(
            {
                "frame": np.arange(len(d_zcalib_values), dtype=np.uint32),
                "z": np.zeros(len(d_zcalib_values), dtype=np.float32),
                "d_zcalib": np.array(d_zcalib_values, dtype=np.float32),
            }
        )

    def test_no_filtering_when_range_zero(self):
        locs = self._locs([0.1, 0.5, 10.0, 100.0])
        result = zfit.filter_z_fits(locs, range=0)
        assert len(result) == len(locs)

    def test_filtering_removes_high_residuals(self):
        """Most rows have small residuals; one outlier far above the
        per-rmsd threshold gets culled at range=2."""
        d_zcalib = [0.1, 0.2, 0.1, 0.2, 100.0]
        locs = self._locs(d_zcalib)
        rmsd = np.sqrt(np.nanmean(np.array(d_zcalib) ** 2))
        result = zfit.filter_z_fits(locs, range=2)
        assert (result["d_zcalib"] <= 2 * rmsd).all()
        assert len(result) == 4  # 100.0 is culled

    def test_no_d_zcalib_column_returns_input(self):
        """Defensive: missing d_zcalib column means the input is passed
        through unchanged."""
        locs = pd.DataFrame({"frame": [0, 1, 2], "z": [0.0, 0.0, 0.0]})
        result = zfit.filter_z_fits(locs, range=2)
        assert len(result) == len(locs)


# ---------------------------------------------------------------------------
# Main pipeline: zfit
# ---------------------------------------------------------------------------


class TestZfit:
    """Tests for the main ``zfit`` pipeline (post v0.11.0 API)."""

    def test_appends_z_d_zcalib_lpz_columns(self, locs, info):
        out, new_info = zfit.zfit(
            locs, info, calibration=dict(CALIB_3D), fitting_method="gausslq"
        )
        for col in ["z", "d_zcalib", "lpz"]:
            assert col in out.columns
        # info gets a new dict appended
        assert isinstance(new_info, list)
        assert len(new_info) == len(info) + 1
        assert "Generated by" in new_info[-1]

    def test_returns_dataframe_with_finite_z(self, locs, info):
        out, _ = zfit.zfit(
            locs, info, calibration=dict(CALIB_3D), fitting_method="gausslq"
        )
        assert len(out) > 0
        assert np.all(np.isfinite(out["z"].to_numpy()))
        assert np.all(np.isfinite(out["lpz"].to_numpy()))
        assert (out["lpz"] > 0).all()

    def test_filter_zero_keeps_all(self, locs, info):
        """``filter=0`` skips residual filtering -> output length == input."""
        out, _ = zfit.zfit(
            locs,
            info,
            calibration=dict(CALIB_3D),
            fitting_method="gausslq",
            filter=0,
        )
        assert len(out) == len(locs)

    def test_pixelsize_argument_is_used(self, locs, info):
        """If ``pixelsize`` is provided as an argument, it is appended to
        info — verify that the function does not raise even when info
        lacks Pixelsize."""
        info_no_px = [
            {k: v for k, v in d.items() if "Pixelsize" not in k} for d in info
        ]
        out, _ = zfit.zfit(
            locs,
            info_no_px,
            calibration=dict(CALIB_3D),
            pixelsize=130.0,
            fitting_method="gausslq",
        )
        assert "z" in out.columns

    def test_invalid_fitting_method_raises(self, locs, info):
        with pytest.raises(AssertionError):
            zfit.zfit(
                locs,
                info,
                calibration=dict(CALIB_3D),
                fitting_method="bogus",
            )

    def test_negative_filter_raises(self, locs, info):
        with pytest.raises(AssertionError):
            zfit.zfit(locs, info, calibration=dict(CALIB_3D), filter=-1)

    def test_calibration_must_be_dict(self, locs, info):
        with pytest.raises(AssertionError):
            zfit.zfit(locs, info, calibration="not a dict")

    def test_magnification_factor_argument_overrides_calibration(
        self, locs, info
    ):
        calib = dict(CALIB_3D)
        out_a, _ = zfit.zfit(
            locs, info, calibration=dict(calib), magnification_factor=0.5
        )
        out_b, _ = zfit.zfit(
            locs, info, calibration=dict(calib), magnification_factor=2.0
        )
        # z scales with magnification factor — so passing different
        # magnifications produces meaningfully different z values.
        assert not np.allclose(out_a["z"].to_numpy(), out_b["z"].to_numpy())

    def test_gausslq_and_gaussmle_paths_both_run(self, locs, info):
        """Both fitting methods should produce a valid output (the lpz
        column is computed differently for each)."""
        out_lq, _ = zfit.zfit(
            locs, info, calibration=dict(CALIB_3D), fitting_method="gausslq"
        )
        out_mle, _ = zfit.zfit(
            locs, info, calibration=dict(CALIB_3D), fitting_method="gaussmle"
        )
        # z is the same regardless of fitting_method (it only affects
        # how lpz is computed)
        assert np.allclose(
            out_lq["z"].to_numpy(), out_mle["z"].to_numpy(), atol=1e-3
        )

    @pytest.mark.slow
    def test_multiprocess_matches_serial(self, locs, info):
        """Multiprocessing must produce equivalent z and lpz values to
        the serial path."""
        out_serial, _ = zfit.zfit(
            locs,
            info,
            calibration=dict(CALIB_3D),
            multiprocess=False,
            filter=0,  # disable filtering so the lengths match exactly
        )
        out_par, _ = zfit.zfit(
            locs,
            info,
            calibration=dict(CALIB_3D),
            multiprocess=True,
            filter=0,
        )
        # both have the same length (filter=0)
        assert len(out_serial) == len(out_par)
        # results align after sorting by frame + integer x/y
        keys = ["frame", "x", "y"]
        s = out_serial.sort_values(keys).reset_index(drop=True)
        p = out_par.sort_values(keys).reset_index(drop=True)
        np.testing.assert_allclose(
            s["z"].to_numpy(), p["z"].to_numpy(), atol=1e-3
        )
        np.testing.assert_allclose(
            s["lpz"].to_numpy(), p["lpz"].to_numpy(), atol=1e-3
        )

    @pytest.mark.slow
    def test_abort_callback_returns_none(self, locs, info):
        """Returning True from ``abort_callback`` aborts and yields
        ``(None, None)``."""
        out, info_out = zfit.zfit(
            locs,
            info,
            calibration=dict(CALIB_3D),
            multiprocess=True,
            abort_callback=lambda: True,
        )
        assert out is None and info_out is None


# ---------------------------------------------------------------------------
# axial_localization_precision and ..._astig
# ---------------------------------------------------------------------------


class TestAxialLocalizationPrecision:
    """Top-level ``axial_localization_precision`` dispatcher."""

    @pytest.fixture
    def fitted_locs(self, locs, info):
        out, _ = zfit.zfit(locs, info, calibration=dict(CALIB_3D), filter=0)
        return out

    def test_returns_one_per_loc(self, fitted_locs, info):
        lpz = zfit.axial_localization_precision(
            fitted_locs, info, dict(CALIB_3D), fitting_method="gausslq"
        )
        assert lpz.shape == (len(fitted_locs),)
        assert (lpz > 0).all()
        assert np.all(np.isfinite(lpz))

    def test_invalid_modality_raises(self, fitted_locs, info):
        with pytest.raises(NotImplementedError):
            zfit.axial_localization_precision(
                fitted_locs,
                info,
                dict(CALIB_3D),
                modality="not_astigmatic",
            )

    def test_gaussmle_with_existing_unc_columns_uses_them(
        self, fitted_locs, info
    ):
        """When sx_unc / sy_unc columns exist, the gaussmle path should
        use them instead of recomputing from gaussmle.sigma_uncertainty."""
        locs2 = fitted_locs.copy()
        # Make the per-loc precomputed uncertainties artificially huge —
        # the resulting lpz must reflect that change.
        locs2["sx_unc"] = 10.0
        locs2["sy_unc"] = 10.0
        lpz_with = zfit.axial_localization_precision(
            locs2, info, dict(CALIB_3D), fitting_method="gaussmle"
        )
        lpz_without = zfit.axial_localization_precision(
            fitted_locs, info, dict(CALIB_3D), fitting_method="gaussmle"
        )
        assert lpz_with.mean() > lpz_without.mean()

    def test_gausslq_and_gaussmle_paths_both_finite(self, fitted_locs, info):
        lpz_lq = zfit.axial_localization_precision(
            fitted_locs, info, dict(CALIB_3D), fitting_method="gausslq"
        )
        lpz_mle = zfit.axial_localization_precision(
            fitted_locs, info, dict(CALIB_3D), fitting_method="gaussmle"
        )
        assert np.all(np.isfinite(lpz_lq))
        assert np.all(np.isfinite(lpz_mle))

    def test_invalid_fitting_method_raises(self, fitted_locs, info):
        with pytest.raises(AssertionError):
            zfit.axial_localization_precision_astig(
                fitted_locs, info, dict(CALIB_3D), fitting_method="bogus"
            )


class TestAxialLocalizationPrecisionAstig:
    """Direct checks on ``axial_localization_precision_astig``."""

    def _make_locs(self, photons_array, n=None):
        """Build a synthetic locs frame with ``photons_array`` and constant
        sigma/bg/z, so we can sweep one variable cleanly."""
        n = n or len(photons_array)
        return pd.DataFrame(
            {
                "x": np.zeros(n, dtype=np.float32),
                "y": np.zeros(n, dtype=np.float32),
                "sx": np.full(n, 1.1, dtype=np.float32),
                "sy": np.full(n, 1.0, dtype=np.float32),
                "photons": np.asarray(photons_array, dtype=np.float32),
                "bg": np.full(n, 5.0, dtype=np.float32),
                "z": np.full(n, 100.0, dtype=np.float32),
            }
        )

    def test_higher_photons_gives_lower_lpz(self, info):
        photons = np.array([500.0, 2000.0, 10000.0])
        locs = self._make_locs(photons)
        lpz = zfit.axial_localization_precision_astig(
            locs, info, dict(CALIB_3D), fitting_method="gausslq"
        )
        # Strict monotonic decrease in lpz as photons grow
        assert lpz[0] > lpz[1] > lpz[2]


# ---------------------------------------------------------------------------
# calibrate_z — synthetic bead-stack
# ---------------------------------------------------------------------------


class TestCalibrateZ:
    """Drive ``calibrate_z`` with a synthetic 'bead stack' DataFrame and
    verify it returns a sensible calibration dict (and writes YAML/PNG
    when a path is given)."""

    @pytest.fixture
    def bead_stack(self):
        """Build a fake bead-stack: 50 frames, 40 beads/frame, with sx/sy
        driven by known polynomials of stage z plus tiny noise.

        ``calibrate_z`` recenters the z axis so the calibration curves
        cross at z=0; we pick polynomials whose curves already cross
        near the middle of the scan to keep the recentering small.
        """
        n_frames = 50
        d = 10.0  # nm step
        rng = np.random.default_rng(0)
        rows = []
        z_total = (n_frames - 1) * d
        for fi in range(n_frames):
            z = -(fi * d - z_total / 2)
            # sx widens with positive z, sy widens with negative z
            sx_mean = 1.5 + 1e-3 * z + 1e-5 * z**2
            sy_mean = 1.5 - 1e-3 * z + 1e-5 * z**2
            for _ in range(40):
                rows.append(
                    {
                        "frame": fi,
                        "x": 16.0,
                        "y": 16.0,
                        "sx": sx_mean + rng.normal(0, 0.02),
                        "sy": sy_mean + rng.normal(0, 0.02),
                        "photons": 5000.0,
                        "bg": 10.0,
                        "lpx": 0.01,
                        "lpy": 0.01,
                    }
                )
        locs = pd.DataFrame(rows)
        info = [
            {
                "Frames": n_frames,
                "Pixelsize": 130,
                "Width": 32,
                "Height": 32,
            }
        ]
        return locs, info, d

    def test_returns_required_keys(self, bead_stack, monkeypatch):
        monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
        locs, info, d = bead_stack
        calib = zfit.calibrate_z(locs, info, d, magnification_factor=0.79)
        for key in [
            "X Coefficients",
            "Y Coefficients",
            "Number of frames",
            "Step size in nm",
            "Magnification factor",
            "Path",
        ]:
            assert key in calib
        assert len(calib["X Coefficients"]) == 7
        assert len(calib["Y Coefficients"]) == 7
        assert calib["Number of frames"] == 50
        assert calib["Step size in nm"] == d
        assert calib["Magnification factor"] == 0.79
        assert calib["Path"] == "N/A"

    def test_recovered_polynomial_resembles_truth(
        self, bead_stack, monkeypatch
    ):
        """Recovered spot-size at z=0 should match the polynomial offset
        we built into the synthetic data (~1.5 px) within ~0.1 px."""
        monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
        locs, info, d = bead_stack
        calib = zfit.calibrate_z(locs, info, d, magnification_factor=0.79)
        cx = np.array(calib["X Coefficients"])
        cy = np.array(calib["Y Coefficients"])
        # at z=0 (after recentering), sx and sy should both be near the
        # crossing point of the synthetic polynomials (~1.5)
        assert zfit._get_calib_size(cx, 0.0) == pytest.approx(1.5, abs=0.1)
        assert zfit._get_calib_size(cy, 0.0) == pytest.approx(1.5, abs=0.1)

    def test_writes_yaml_and_png_when_path_given(
        self, bead_stack, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
        locs, info, d = bead_stack
        out_path = tmp_path / "calib.yaml"
        calib = zfit.calibrate_z(
            locs, info, d, magnification_factor=0.79, path=str(out_path)
        )
        assert out_path.exists()
        png_path = tmp_path / "calib.png"
        assert png_path.exists()
        # Round-trip through YAML
        with open(out_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["X Coefficients"] == calib["X Coefficients"]
        assert loaded["Y Coefficients"] == calib["Y Coefficients"]
        assert loaded["Number of frames"] == calib["Number of frames"]
        assert loaded["Step size in nm"] == calib["Step size in nm"]


class TestCalibrateZFrameBounds:
    """``frame_bounds`` restricts which frames enter the calibration.
    Bounds are inclusive on both ends, matching ``picasso.localize``."""

    N_FRAMES = 50
    D = 10.0  # nm step

    @pytest.fixture(autouse=True)
    def _no_show(self, monkeypatch):
        monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)

    @pytest.fixture
    def bead_stack(self):
        """Same synthetic bead stack as in ``TestCalibrateZ``: sx/sy
        driven by known polynomials of stage z plus tiny noise."""
        rng = np.random.default_rng(0)
        rows = []
        z_total = (self.N_FRAMES - 1) * self.D
        for fi in range(self.N_FRAMES):
            z = -(fi * self.D - z_total / 2)
            sx_mean = 1.5 + 1e-3 * z + 1e-5 * z**2
            sy_mean = 1.5 - 1e-3 * z + 1e-5 * z**2
            for _ in range(40):
                rows.append(
                    {
                        "frame": fi,
                        "x": 16.0,
                        "y": 16.0,
                        "sx": sx_mean + rng.normal(0, 0.02),
                        "sy": sy_mean + rng.normal(0, 0.02),
                        "photons": 5000.0,
                        "bg": 10.0,
                        "lpx": 0.01,
                        "lpy": 0.01,
                    }
                )
        locs = pd.DataFrame(rows)
        info = [
            {
                "Frames": self.N_FRAMES,
                "Pixelsize": 130,
                "Width": 32,
                "Height": 32,
            }
        ]
        return locs, info

    def _polyfit_lengths(self, monkeypatch):
        """Spy on ``np.polyfit`` to record how many frames enter each
        calibration fit."""
        lengths = []
        orig = np.polyfit

        def spy(x, y, deg, **kwargs):
            lengths.append(len(x))
            return orig(x, y, deg, **kwargs)

        monkeypatch.setattr(np, "polyfit", spy)
        return lengths

    def test_none_and_none_none_equivalent(self, bead_stack):
        """``frame_bounds=(None, None)`` must behave exactly like
        ``frame_bounds=None`` (all frames used)."""
        locs, info = bead_stack
        calib_none = zfit.calibrate_z(
            locs, info, self.D, magnification_factor=0.79, frame_bounds=None
        )
        calib_nn = zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(None, None),
        )
        np.testing.assert_allclose(
            calib_none["X Coefficients"], calib_nn["X Coefficients"]
        )
        np.testing.assert_allclose(
            calib_none["Y Coefficients"], calib_nn["Y Coefficients"]
        )

    def test_full_range_bounds_equivalent_to_none(self, bead_stack):
        """``(0, n_frames - 1)`` covers all frames, so it must match the
        unbounded calibration."""
        locs, info = bead_stack
        calib_none = zfit.calibrate_z(
            locs, info, self.D, magnification_factor=0.79
        )
        calib_full = zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(0, self.N_FRAMES - 1),
        )
        np.testing.assert_allclose(
            calib_none["X Coefficients"], calib_full["X Coefficients"]
        )
        np.testing.assert_allclose(
            calib_none["Y Coefficients"], calib_full["Y Coefficients"]
        )

    def test_bounds_are_inclusive(self, bead_stack, monkeypatch):
        """``(10, 39)`` keeps frames 10..39 inclusive -> 30 frames enter
        every polynomial fit."""
        locs, info = bead_stack
        lengths = self._polyfit_lengths(monkeypatch)
        zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(10, 39),
        )
        assert lengths and all(n == 30 for n in lengths)

    def test_one_sided_bounds(self, bead_stack, monkeypatch):
        """``(None, max)`` and ``(min, None)`` each leave the other side
        unbounded."""
        locs, info = bead_stack
        lengths = self._polyfit_lengths(monkeypatch)
        zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(None, 29),  # frames 0..29
        )
        assert lengths and all(n == 30 for n in lengths)
        lengths.clear()
        zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(20, None),  # frames 20..49
        )
        assert lengths and all(n == 30 for n in lengths)

    def test_bounds_including_last_frame(self, bead_stack):
        """Regression test: a nonzero minimum together with the last
        frame as maximum used to raise IndexError (localizations'
        frame numbers were used as indices into the sliced per-frame
        arrays without offsetting)."""
        locs, info = bead_stack
        calib = zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(10, self.N_FRAMES - 1),
        )
        assert len(calib["X Coefficients"]) == 7
        assert len(calib["Y Coefficients"]) == 7

    def test_locs_outside_bounds_are_ignored(self, bead_stack):
        """Passing the full locs with bounds must equal passing locs
        already restricted to those frames."""
        locs, info = bead_stack
        bounds = (10, 39)
        pre_filtered = locs[
            (locs["frame"] >= bounds[0]) & (locs["frame"] <= bounds[1])
        ]
        calib_full = zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=bounds,
        )
        calib_pre = zfit.calibrate_z(
            pre_filtered,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=bounds,
        )
        np.testing.assert_allclose(
            calib_full["X Coefficients"], calib_pre["X Coefficients"]
        )
        np.testing.assert_allclose(
            calib_full["Y Coefficients"], calib_pre["Y Coefficients"]
        )

    def test_bounded_calibration_differs_from_unbounded(self, bead_stack):
        """Restricting the frames must actually change the fit."""
        locs, info = bead_stack
        calib_none = zfit.calibrate_z(
            locs, info, self.D, magnification_factor=0.79
        )
        calib_bounded = zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(10, 39),
        )
        assert not np.allclose(
            calib_none["X Coefficients"], calib_bounded["X Coefficients"]
        )

    def test_frame_bounds_stored_in_calibration(self, bead_stack):
        locs, info = bead_stack
        calib = zfit.calibrate_z(
            locs,
            info,
            self.D,
            magnification_factor=0.79,
            frame_bounds=(10, 39),
        )
        assert calib["Frame bounds"] == (10, 39)
        calib = zfit.calibrate_z(locs, info, self.D, magnification_factor=0.79)
        assert calib["Frame bounds"] is None


# ---------------------------------------------------------------------------
# locs_from_futures
# ---------------------------------------------------------------------------


class TestLocsFromFutures:
    """Combine the per-task outputs of ``zfit`` multiprocessing into one
    DataFrame."""

    def test_concatenates_and_filters(self):
        """Stitching two small frames together should preserve all rows
        (range=0) and apply filtering when range>0."""
        # Many small values, so one outlier sticks out above the rmsd
        df1 = pd.DataFrame(
            {
                "z": np.zeros(10),
                "d_zcalib": np.full(10, 0.1),
                "frame": np.arange(10),
            }
        )
        df2 = pd.DataFrame(
            {
                "z": np.zeros(2),
                "d_zcalib": [0.1, 5.0],  # 5.0 is the outlier
                "frame": [10, 11],
            }
        )

        class _DummyFuture:
            def __init__(self, val):
                self._val = val

            def result(self):
                return self._val

        # No filter -> all 12 rows
        result = zfit.locs_from_futures(
            [_DummyFuture(df1), _DummyFuture(df2)], filter=0
        )
        assert len(result) == 12
        # filter=2 -> the 5.0 outlier is culled
        # rmsd ~= sqrt((11*0.01 + 25)/12) ~= 1.45; threshold = 2*1.45 = 2.9
        result = zfit.locs_from_futures(
            [_DummyFuture(df1), _DummyFuture(df2)], filter=2
        )
        assert len(result) == 11
        assert (result["d_zcalib"] < 5.0).all()
