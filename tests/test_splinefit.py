"""
test_splinefit
~~~~~~~~~~~~~~

Tests for ``picasso.fitting.splinefit``, the CPU cubic-spline PSF fitter.

Everything here runs without a GPU:
the calibrations are built analytically from tensor products of 1D cubic
splines, so the model, its derivatives and the parameters a fit must
recover are all known in closed form.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

import time

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline

from picasso import localize

from tests.conftest import affine, apply_transform
from picasso.fitting import precision, splinefit

BOX = 13
NZ = 21
IDENTITY = np.array([[1.0, 0.0, 0.0, 1.0]])
# A deliberately asymmetric sub-pixel shift. The GPU spline tests use
# symmetric shifts to stay agnostic about the x <-> y convention; these must
# not, because pinning that convention is the point.
DX, DY = 0.30, -0.20


# ---------------------------------------------------------------------------
# Analytic calibrations
# ---------------------------------------------------------------------------
def _gauss_spline_1d(sigma, center, n):
    x = np.arange(n, dtype=np.float64)
    return CubicSpline(x, np.exp(-0.5 * ((x - center) / sigma) ** 2))


def _astigmatic_calibration(box=BOX, nz=NZ, n_channels=1, model="spline-3d"):
    """Analytic 3D spline calibration whose *lateral* shape changes with z.

    A single separable Gaussian ``Gx(x) Gy(y) Gz(z)`` is useless for testing
    the axial fit: its z dependence is a scalar multiplier, so ``N`` and ``z``
    are exactly degenerate and any z fits the data equally well. This sums two
    separable terms whose lateral widths are swapped and whose z weights cross
    over, which is the astigmatic PSF in miniature - narrow in x / wide in y at
    the bottom of the stack, the reverse at the top. The tricubic coefficients
    of a sum are the sum of the coefficients, so the closed form survives.

    Returns ``(calibration, terms)`` where ``terms`` is the list of
    ``(gx, gy, gz)`` 1D splines to hand to :func:`_reference_model`.
    """
    cxy = (box - 1) / 2.0
    # (sigma_x, sigma_y, z-center of this term's weight)
    spec = [(0.9, 1.9, 0.25 * (nz - 1)), (1.9, 0.9, 0.75 * (nz - 1))]
    sigma_z = 0.42 * nz  # wide enough that both terms overlap everywhere
    nix = niy = box - 1
    niz = nz - 1
    coefficients = np.zeros((niz, niy, nix, 4, 4, 4))
    terms = []
    for sx, sy, z0 in spec:
        gx = _gauss_spline_1d(sx, cxy, box)
        gy = _gauss_spline_1d(sy, cxy, box)
        gz = _gauss_spline_1d(sigma_z, z0, nz)
        terms.append((gx, gy, gz))
        # per-interval coefficients, ascending powers:
        # c[i, p] = spline.c[3 - p, i]
        coefficients += np.einsum(
            "zZ,yY,xX->zyxZYX", gz.c[::-1].T, gy.c[::-1].T, gx.c[::-1].T
        )
    # (niz, niy, nix, zp, yp, xp) is the raw flat coefficient buffer (the
    # spline.spline_coefficients layout), which NumPy labels
    # (64, nix, niy, niz).
    raw = coefficients.reshape(64, nix, niy, niz).astype(np.float32)
    if model == "spline-3d-multichannel":
        raw = np.repeat(raw[..., None], n_channels, axis=-1)
    calibration = {
        "model": model,
        "coefficients": raw,
        "n_data": [box, box, nz],
        "n_intervals": [nix, niy, niz],
        "oversampling": 1.0,
        "photon_scale": 1.0,
        "z_center": (nz - 1) / 2.0,
        "z_step_nm": 20.0,
        "magnification_factor": 1.0,
        "box": box,
        "pixelsize": 130.0,
    }
    if model == "spline-3d-multichannel":
        calibration["n_channels"] = n_channels
    return calibration, terms


def _flat_calibration(box=BOX, nz=NZ):
    """The same construction with a single term, i.e. no axial information.

    Useful for the 2D model and wherever z must not matter."""
    cxy = (box - 1) / 2.0
    gx = _gauss_spline_1d(1.0, cxy, box)
    gy = _gauss_spline_1d(1.4, cxy, box)
    nix = niy = box - 1
    coefficients = np.einsum("yY,xX->yxYX", gy.c[::-1].T, gx.c[::-1].T)
    return {
        "model": "spline-2d",
        "coefficients": coefficients.reshape(16, nix, niy).astype(np.float32),
        "n_data": [box, box],
        "n_intervals": [nix, niy],
        "oversampling": 1.0,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": 130.0,
    }, [(gx, gy, None)]


def _reference_model(terms, box, x_shift, y_shift, z_native):
    """Closed-form ``(phi, dphi_dx, dphi_dy, dphi_dz)`` on the box grid.

    Indexed ``[y_pixel, x_pixel]`` - the layout of the spot data itself, which
    is what makes this an independent check of the kernels' x/y convention.
    Derivatives are with respect to the native coordinate."""
    grid = np.arange(box, dtype=np.float64)
    xc = grid - x_shift
    yc = grid - y_shift
    phi = np.zeros((box, box))
    gx_out = np.zeros((box, box))
    gy_out = np.zeros((box, box))
    gz_out = np.zeros((box, box))
    for gx, gy, gz in terms:
        # [y, x] layout: y varies along axis 0.
        vx = gx(xc)[None, :]
        vy = gy(yc)[:, None]
        dvx = gx.derivative()(xc)[None, :]
        dvy = gy.derivative()(yc)[:, None]
        if gz is None:
            vz, dvz = 1.0, 0.0
        else:
            vz = float(gz(z_native))
            dvz = float(gz.derivative()(z_native))
        phi += vx * vy * vz
        gx_out += dvx * vy * vz
        gy_out += vx * dvy * vz
        gz_out += vx * vy * dvz
    return phi, gx_out, gy_out, gz_out


def _spots_from_terms(terms, box, amplitude, offset, x, y, z, n_channels=1):
    """Noiseless spots ``(1, n_channels, box, box)`` from the closed form."""
    spots = np.zeros((1, n_channels, box, box))
    for ch in range(n_channels):
        phi = _reference_model(terms, box, x, y, z)[0]
        spots[0, ch] = offset + amplitude * phi
    return np.ascontiguousarray(spots, dtype=np.float32)


def _run(
    kind,
    calibration,
    spots,
    initial,
    mle,
    seeds=None,
    jacobians=None,
    residuals=None,
    tolerance=None,
    max_iterations=None,
):
    """Serial fit of ``spots`` with the given model."""
    coefficients = precision._spline_coeff_reshaped(calibration)
    n_spots, n_channels = spots.shape[0], spots.shape[1]
    jacobians = IDENTITY if jacobians is None else jacobians
    jacobians = np.asarray(jacobians, dtype=np.float64)
    if jacobians.ndim == 2:
        # (n_channels, 4) -> the per-spot (n_spots, n_channels, 4) the kernels
        # take; an affine registration is the same Jacobian at every spot
        jacobians = np.ascontiguousarray(
            np.tile(jacobians, (max(n_spots, 1), 1, 1))
        )
    if residuals is None:
        residuals = np.zeros((max(n_spots, 1), n_channels, 2))
    apply_seeds = seeds is not None
    z_seeds = np.asarray(seeds, float) if apply_seeds else np.zeros(1)
    if tolerance is None:
        tolerance = (
            splinefit.TOLERANCE_MULTI_START
            if apply_seeds
            else splinefit.TOLERANCE_SINGLE_START
        )
    if max_iterations is None:
        max_iterations = (
            splinefit.MAX_ITERATIONS_MULTI_START
            if apply_seeds
            else splinefit.MAX_ITERATIONS_SINGLE_START
        )
    return splinefit.fit_spots(
        kind,
        spots,
        coefficients,
        jacobians,
        residuals,
        initial,
        z_seeds,
        apply_seeds,
        mle=mle,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _default_seeds(calibration):
    n_seeds = localize._default_n_z_starts(calibration)
    return np.linspace(-(calibration["n_data"][2] - 1), 0.0, n_seeds)


def _drain(fit, timeout=300):
    """Wait for an ``AsyncFit`` to finish, surfacing any worker exception."""
    deadline = time.time() + timeout
    while not fit.finished() and time.time() < deadline:
        fit.raise_errors()
        time.sleep(0.01)
    fit.raise_errors()
    assert fit.finished(), "the fitting workers did not finish"


# ---------------------------------------------------------------------------


class TestSplineEvaluation:
    """The tricubic/bicubic evaluation and its analytic derivatives."""

    @pytest.mark.parametrize(
        "positions",
        [
            pytest.param((0.3, -0.2, 9.0), id="inside"),
            # Out-of-box positions must extrapolate the boundary cubic, not
            # saturate at it: only the interval index is clamped, the
            # fractional coordinate keeps its true value.
            pytest.param((9.5, -8.25, -3.5), id="extrapolating-low"),
            pytest.param((-9.0, 11.0, 26.0), id="extrapolating-high"),
        ],
    )
    def test_matches_closed_form_3d(self, positions):
        x, y, z = positions
        calibration, terms = _astigmatic_calibration()
        coefficients = precision._spline_coeff_reshaped(calibration)
        phi, gx, gy, gz = _reference_model(terms, BOX, x, y, z)
        for j in range(BOX):
            for i in range(BOX):
                got = splinefit._eval_spline_3d(
                    coefficients, 0, i - x, j - y, z
                )
                expected = (phi[j, i], gx[j, i], gy[j, i], gz[j, i])
                scale = max(np.max(np.abs(phi)), 1e-12)
                # The calibration stores float32 coefficients, so the
                # comparison bottoms out at float32 rounding of the table.
                np.testing.assert_allclose(
                    got, expected, atol=5e-7 * scale, rtol=0
                )

    def test_matches_closed_form_2d(self):
        calibration, terms = _flat_calibration()
        coefficients = precision._spline_coeff_reshaped(calibration)
        x, y = 0.3, -0.2
        phi, gx, gy, _ = _reference_model(terms, BOX, x, y, None)
        for j in range(BOX):
            for i in range(BOX):
                got = splinefit._eval_spline_2d(coefficients, 0, i - x, j - y)
                scale = max(np.max(np.abs(phi)), 1e-12)
                np.testing.assert_allclose(
                    got,
                    (phi[j, i], gx[j, i], gy[j, i]),
                    atol=5e-7 * scale,
                    rtol=0,
                )

    def test_matches_closed_form_off_grid_batch(self):
        """The closed form again at several asymmetric, off-grid positions,
        including shifts beyond a pixel. ``test_matches_closed_form_3d`` pins
        one position; this widens the sampling so a layout error that happens
        to be benign at a single point cannot survive."""
        calibration, terms = _astigmatic_calibration()
        coefficients = precision._spline_coeff_reshaped(calibration)
        for x, y, z in ((0.3, -0.2, 9.0), (-0.7, 1.1, 12.4)):
            phi, gx, gy, gz = _reference_model(terms, BOX, x, y, z)
            scale = max(np.max(np.abs(phi)), 1e-12)
            for j in range(BOX):
                for i in range(BOX):
                    got = splinefit._eval_spline_3d(
                        coefficients, 0, i - x, j - y, z
                    )
                    np.testing.assert_allclose(
                        got,
                        (phi[j, i], gx[j, i], gy[j, i], gz[j, i]),
                        atol=5e-7 * scale,
                        rtol=0,
                    )


class TestJacobian:
    """The analytic Jacobian against central finite differences.

    This is what pins the sign of the position derivatives. The CRLB kernels
    can drop that sign because the CRLB diagonal is sign-invariant; a
    Levenberg-Marquardt step cannot - a flipped sign sends x, y and z the wrong
    way. It also covers the multichannel affine chain rule, which uses the
    transpose of the channel affine."""

    JACOBIANS = np.array([[1.0, 0.0, 0.0, 1.0], [0.997, 0.035, -0.031, 1.006]])
    RESIDUALS = np.array([[0.0, 0.0], [0.31, -0.22]])

    @staticmethod
    def _check(kind, calibration, theta, jacobians, residuals):
        coefficients = precision._spline_coeff_reshaped(calibration)
        _, jacobian = splinefit.model_and_jacobian(
            kind, coefficients, jacobians, residuals, theta, BOX
        )
        for p in range(len(theta)):
            step = 1e-6 * max(abs(theta[p]), 1.0)
            up, down = theta.copy(), theta.copy()
            up[p] += step
            down[p] -= step
            mu_up = splinefit.model_and_jacobian(
                kind, coefficients, jacobians, residuals, up, BOX
            )[0]
            mu_down = splinefit.model_and_jacobian(
                kind, coefficients, jacobians, residuals, down, BOX
            )[0]
            numeric = (mu_up - mu_down) / (2 * step)
            scale = max(np.max(np.abs(numeric)), 1e-9)
            np.testing.assert_allclose(
                jacobian[..., p], numeric, atol=1e-5 * scale, rtol=0
            )

    def test_2d(self):
        calibration, _ = _flat_calibration()
        self._check(
            splinefit.KIND_2D,
            calibration,
            np.array([500.0, 0.3, -0.2, 20.0]),
            IDENTITY,
            np.zeros((1, 2)),
        )

    def test_3d(self):
        calibration, _ = _astigmatic_calibration()
        self._check(
            splinefit.KIND_3D,
            calibration,
            np.array([500.0, 0.3, -0.2, -8.4, 20.0]),
            IDENTITY,
            np.zeros((1, 2)),
        )

    def test_3d_multichannel(self):
        calibration, _ = _astigmatic_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        self._check(
            splinefit.KIND_3D,
            calibration,
            np.array([500.0, 0.3, -0.2, -8.4, 20.0]),
            self.JACOBIANS,
            self.RESIDUALS,
        )

    def test_link_xyz(self):
        calibration, _ = _astigmatic_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        self._check(
            splinefit.KIND_LINK_XYZ,
            calibration,
            np.array([0.3, -0.2, -8.4, 400.0, 620.0, 18.0, 25.0]),
            self.JACOBIANS,
            self.RESIDUALS,
        )


class TestSolver:
    """``_solve_gj`` - the one piece that must never raise, because an
    exception inside a nogil worker would stall the shared progress counter and
    hang the driver loop."""

    @staticmethod
    def _solve(matrix, rhs):
        n = len(rhs)
        a = np.array(matrix, dtype=float)
        b = np.array(rhs, dtype=float)
        ok = splinefit._solve_gj(
            a,
            b,
            n,
            np.empty(n, np.int32),
            np.empty(n, np.int32),
            np.empty(n, np.int32),
        )
        return ok, b

    @pytest.mark.parametrize("n", [4, 5, 7, 15])
    def test_matches_numpy(self, n):
        rng = np.random.default_rng(n)
        for _ in range(25):
            a = rng.normal(size=(n, n))
            spd = a @ a.T + n * np.eye(n)
            rhs = rng.normal(size=n)
            ok, solution = self._solve(spd, rhs)
            assert ok
            np.testing.assert_allclose(
                solution, np.linalg.solve(spd, rhs), rtol=1e-9, atol=1e-11
            )

    def test_rejects_singular(self):
        rank_deficient = np.eye(5)
        rank_deficient[3] = rank_deficient[1]
        for matrix in (
            np.ones((5, 5)),
            np.zeros((5, 5)),
            np.full((5, 5), np.nan),
            rank_deficient,
        ):
            ok, _ = self._solve(matrix, np.ones(5))
            assert not ok


class TestSingleChannelFits:
    """Parameter recovery on noiseless spots."""

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_2d_recovers_asymmetric_shift(self, mle):
        calibration, terms = _flat_calibration()
        amplitude, offset = 2000.0, 15.0
        spots = _spots_from_terms(terms, BOX, amplitude, offset, DX, DY, None)
        initial = np.array(
            [[float(spots.max() - spots.min()), 0.0, 0.0, float(spots.min())]]
        )
        theta, _, states, _ = _run(
            splinefit.KIND_2D, calibration, spots, initial, mle
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        assert abs(theta[0, 1] - DX) < 2e-3
        assert abs(theta[0, 2] - DY) < 2e-3
        assert abs(theta[0, 0] / amplitude - 1) < 5e-3
        assert abs(theta[0, 3] - offset) < 0.1

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_3d_recovers_all_parameters(self, mle):
        calibration, terms = _astigmatic_calibration()
        amplitude, offset, z_native = 2000.0, 15.0, 8.4
        spots = _spots_from_terms(
            terms, BOX, amplitude, offset, DX, DY, z_native
        )
        initial = np.zeros((1, 5))
        initial[0, 0] = spots.max() - spots.min()
        initial[0, 3] = -calibration["z_center"]
        initial[0, 4] = spots.min()
        theta, _, states, _ = _run(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            mle,
            seeds=_default_seeds(calibration),
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        assert abs(theta[0, 1] - DX) < 2e-3
        assert abs(theta[0, 2] - DY) < 2e-3
        assert abs(-theta[0, 3] - z_native) < 2e-2
        assert abs(theta[0, 0] / amplitude - 1) < 5e-3
        assert abs(theta[0, 4] - offset) < 0.2

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_converges_from_a_bad_seed(self, mle):
        """A seed 2 px and 3 slices off still converges. This is the check on
        the adaptive damping: Gpufit's scaling vector is the running maximum of
        the Hessian diagonal and must stay monotone across iterations -
        resetting it every iteration makes badly-seeded fits stall.

        Amplitude and background are seeded from the data, as
        ``seeds.initial_parameters_spline`` does, so the test isolates the
        position parameters. Seeding those badly *as well* drives the
        maximum-likelihood estimator into a negative model value and it bails
        out - see :meth:`test_mle_reports_negative_curvature`."""
        calibration, terms = _astigmatic_calibration()
        z_native = 8.4
        spots = _spots_from_terms(terms, BOX, 3000.0, 20.0, DX, DY, z_native)
        initial = np.array(
            [
                [
                    float(spots.max() - spots.min()),
                    2.0,
                    -2.0,
                    -(z_native + 3.0),
                    float(spots.min()),
                ]
            ]
        )
        theta, _, states, _ = _run(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            mle,
            tolerance=splinefit.TOLERANCE_MULTI_START,
            max_iterations=splinefit.MAX_ITERATIONS_MULTI_START,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        assert abs(theta[0, 1] - DX) < 5e-3
        assert abs(theta[0, 2] - DY) < 5e-3
        assert abs(-theta[0, 3] - z_native) < 5e-2

    def test_mle_survives_a_negative_model_value(self):
        """A cubic spline undershoots slightly negative in the tails, so a
        bright, low-background spot has ``amplitude * phi + offset < 0`` in the
        corners at the standard seed. Gpufit aborts such a fit outright; here
        the model value is floored (see ``splinefit.MU_FLOOR``) so the region
        acts as a barrier and the fit still converges."""
        calibration, terms = _astigmatic_calibration()
        amplitude, offset, z_native = 8000.0, 1.0, 8.4
        spots = _spots_from_terms(
            terms, BOX, amplitude, offset, DX, DY, z_native
        )
        # The standard seed of seeds.initial_parameters_spline. It puts the
        # corner model value below zero for this calibration.
        initial = np.zeros((1, 5))
        initial[0, 0] = spots.max() - spots.min()
        initial[0, 3] = -calibration["z_center"]
        initial[0, 4] = spots.min()
        coefficients = precision._spline_coeff_reshaped(calibration)
        mu = splinefit.model_and_jacobian(
            splinefit.KIND_3D,
            coefficients,
            IDENTITY,
            np.zeros((1, 2)),
            initial[0],
            BOX,
        )[0]
        assert mu.min() < 0.0, "seed no longer exercises a negative model"
        theta, _, states, _ = _run(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            True,
            seeds=_default_seeds(calibration),
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        assert abs(theta[0, 1] - DX) < 2e-3
        assert abs(theta[0, 2] - DY) < 2e-3
        assert abs(-theta[0, 3] - z_native) < 2e-2

    def test_diverged_fit_is_marked_not_silently_wrong(self):
        """A fit driven to non-finite parameters reports a non-zero state and
        an infinite chi-square, so ``locs_from_fits_spline`` can NaN it out."""
        calibration, terms = _astigmatic_calibration()
        spots = _spots_from_terms(terms, BOX, 3000.0, 20.0, DX, DY, 8.4)
        initial = np.array([[np.nan, 0.0, 0.0, -8.4, 20.0]])
        theta, chi_squares, states, _ = _run(
            splinefit.KIND_3D, calibration, spots, initial, True
        )
        assert states[0] != splinefit.FIT_STATE_CONVERGED
        assert not np.isfinite(chi_squares[0])
        assert np.all(np.isnan(theta[0]))

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_multistart_recovers_z_away_from_focus(self, mle):
        """Spots from slices across the whole stack recover their z, and the
        recovered z is monotonic in the true z. A single in-focus start leaves
        the fit axially degenerate, which is what the multi-start is for."""
        calibration, terms = _astigmatic_calibration()
        z_planes = np.array([2.5, 6.0, 10.0, 14.0, 18.0])
        spots = np.concatenate(
            [
                _spots_from_terms(terms, BOX, 4000.0, 25.0, DX, DY, z)
                for z in z_planes
            ]
        )
        initial = np.zeros((len(z_planes), 5))
        initial[:, 0] = spots.max(axis=(1, 2, 3)) - spots.min(axis=(1, 2, 3))
        initial[:, 3] = -calibration["z_center"]
        initial[:, 4] = spots.min(axis=(1, 2, 3))
        theta, _, states, _ = _run(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            mle,
            seeds=_default_seeds(calibration),
        )
        recovered = -theta[:, 3]
        np.testing.assert_allclose(recovered, z_planes, atol=0.05)
        assert np.all(np.diff(recovered) > 0)


class TestMultichannelFits:
    """The shared-amplitude and photon-decoupled multichannel models.

    Both are given a non-identity channel affine and non-zero sub-pixel ROI
    residuals, so the test covers the full geometry the CUDA models implement:
    each channel sees the shared lateral shift through its own transform, minus
    its own ROI residual."""

    JACOBIANS = np.array([[1.0, 0.0, 0.0, 1.0], [0.997, 0.035, -0.031, 1.006]])
    RESIDUALS = np.array([[[0.0, 0.0], [0.31, -0.22]]])
    Z_NATIVE = 7.6

    def _spots(self, terms, per_channel):
        """Spots whose channel ``c`` sits at the mapped position, built from
        the closed form - so the fit has to undo the affine and the residual to
        recover the shared reference-frame shift."""
        spots = np.zeros((1, len(per_channel), BOX, BOX))
        for ch, (amplitude, offset) in enumerate(per_channel):
            a00, a01, a10, a11 = self.JACOBIANS[ch]
            x = a00 * DX + a01 * DY + self.RESIDUALS[0, ch, 0]
            y = a10 * DX + a11 * DY + self.RESIDUALS[0, ch, 1]
            phi = _reference_model(terms, BOX, x, y, self.Z_NATIVE)[0]
            spots[0, ch] = offset + amplitude * phi
        return np.ascontiguousarray(spots, dtype=np.float32)

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_shared_amplitude(self, mle):
        calibration, terms = _astigmatic_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        amplitude, offset = 5000.0, 30.0
        spots = self._spots(terms, [(amplitude, offset)] * 2)
        initial = np.array(
            [
                [
                    float(spots.max() - spots.min()),
                    0.0,
                    0.0,
                    -calibration["z_center"],
                    float(spots.min()),
                ]
            ]
        )
        theta, _, states, _ = _run(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            mle,
            seeds=_default_seeds(calibration),
            jacobians=self.JACOBIANS,
            residuals=self.RESIDUALS,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        assert abs(theta[0, 1] - DX) < 2e-3
        assert abs(theta[0, 2] - DY) < 2e-3
        assert abs(-theta[0, 3] - self.Z_NATIVE) < 2e-2
        assert abs(theta[0, 0] / amplitude - 1) < 5e-3

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_link_xyz_decouples_photons(self, mle):
        calibration, terms = _astigmatic_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        calibration = dict(
            calibration, model="spline-3d-multichannel-link-xyz"
        )
        per_channel = [(5000.0, 30.0), (2000.0, 12.0)]
        spots = self._spots(terms, per_channel)
        channel_max = spots.max(axis=(2, 3))[0]
        channel_min = spots.min(axis=(2, 3))[0]
        initial = np.zeros((1, 7))
        initial[0, 2] = -calibration["z_center"]
        initial[0, 3:5] = channel_max - channel_min
        initial[0, 5:7] = channel_min
        theta, _, states, _ = _run(
            splinefit.KIND_LINK_XYZ,
            calibration,
            spots,
            initial,
            mle,
            seeds=_default_seeds(calibration),
            jacobians=self.JACOBIANS,
            residuals=self.RESIDUALS,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        assert abs(theta[0, 0] - DX) < 2e-3
        assert abs(theta[0, 1] - DY) < 2e-3
        assert abs(-theta[0, 2] - self.Z_NATIVE) < 2e-2
        for ch, (amplitude, offset) in enumerate(per_channel):
            assert abs(theta[0, 3 + ch] / amplitude - 1) < 5e-3
            assert abs(theta[0, 5 + ch] - offset) < 0.5


class TestNoisyBatch:
    """Poisson-noisy spots through the threaded driver."""

    @staticmethod
    def _batch(n_spots=200, seed=3):
        rng = np.random.default_rng(seed)
        calibration, terms = _astigmatic_calibration()
        xs = rng.uniform(-1.0, 1.0, n_spots)
        ys = rng.uniform(-1.0, 1.0, n_spots)
        zs = rng.uniform(3.0, NZ - 4.0, n_spots)
        amplitudes = rng.uniform(2000, 6000, n_spots)
        offsets = rng.uniform(5, 30, n_spots)
        clean = np.concatenate(
            [
                _spots_from_terms(terms, BOX, a, o, x, y, z)
                for a, o, x, y, z in zip(amplitudes, offsets, xs, ys, zs)
            ]
        ).astype(np.float64)
        spots = rng.poisson(np.maximum(clean, 0)).astype(np.float32)
        initial = np.zeros((n_spots, 5))
        initial[:, 0] = spots.max(axis=(1, 2, 3)) - spots.min(axis=(1, 2, 3))
        initial[:, 3] = -calibration["z_center"]
        initial[:, 4] = spots.min(axis=(1, 2, 3))
        return calibration, spots, initial, xs, ys, zs

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_threaded_fit_is_unbiased_and_efficient(self, mle):
        """Poisson-noisy spots recover their parameters without bias and with a
        scatter at the theoretical limit.

        The lateral scatter is compared against ``precision._spline_crlb`` for
        the matching estimator, which is a far stronger statement than any
        hand-picked tolerance: it says the fitter actually reaches the bound.
        The axial scatter is checked robustly, because a small tail of spots
        settles in a different axial minimum - inherent to spline fitting, not
        to this implementation, and the reason the multi-start exists."""
        calibration, spots, initial, xs, ys, zs = self._batch()
        n_spots = len(spots)
        coefficients = precision._spline_coeff_reshaped(calibration)
        fit = splinefit.fit_spots_async(
            splinefit.KIND_3D,
            spots,
            coefficients,
            np.tile(IDENTITY, (n_spots, 1, 1)),
            np.zeros((n_spots, 1, 2)),
            initial,
            _default_seeds(calibration),
            True,
            mle=mle,
            tolerance=splinefit.TOLERANCE_MULTI_START,
            max_iterations=splinefit.MAX_ITERATIONS_MULTI_START,
        )
        _drain(fit)
        theta, chi_squares, states, _ = fit.results()
        assert fit.current[0] == n_spots
        assert np.all(np.isfinite(chi_squares))
        assert np.all(states == splinefit.FIT_STATE_CONVERGED)

        crlb = precision._spline_crlb(
            theta.astype(np.float32), calibration, BOX, mle=mle
        )
        sigma = np.sqrt(np.nanmedian(crlb[:, :3], axis=0))
        dx = theta[:, 1] - xs
        dy = theta[:, 2] - ys
        dz = -theta[:, 3] - zs

        # Unbiased. The median is used for z so the axial-minimum tail cannot
        # mask a genuine offset.
        assert abs(np.median(dx)) < 0.2 * sigma[0]
        assert abs(np.median(dy)) < 0.2 * sigma[1]
        assert abs(np.median(dz)) < 0.2 * sigma[2]

        # At the bound laterally. The maximum-likelihood bound is optimistic
        # here because _spline_crlb floors 1/mu where the spline rings
        # negative, so allow more headroom for it.
        headroom = 3.0 if mle else 1.5
        assert dx.std() < headroom * sigma[0]
        assert dy.std() < headroom * sigma[1]

        # Axially, a few percent of spots may land in another minimum.
        robust_dz = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        assert robust_dz < 2.0 * sigma[2]
        outliers = np.mean(np.abs(dz) > 5.0 * sigma[2])
        assert outliers < 0.05, f"{outliers:.1%} of spots missed the z minimum"

    def test_serial_and_threaded_agree(self):
        """The thread pool must not change the result: each spot is fitted
        independently and writes only its own row."""
        calibration, spots, initial, _, _, _ = self._batch(n_spots=64)
        n_spots = len(spots)
        coefficients = precision._spline_coeff_reshaped(calibration)
        args = (
            splinefit.KIND_3D,
            spots,
            coefficients,
            np.tile(IDENTITY, (n_spots, 1, 1)),
            np.zeros((n_spots, 1, 2)),
            initial,
            _default_seeds(calibration),
            True,
        )
        serial = splinefit.fit_spots(*args, mle=False)
        fit = splinefit.fit_spots_async(*args, mle=False)
        _drain(fit)
        theta, chi_squares, states, _ = fit.results()
        np.testing.assert_array_equal(serial[0], theta)
        np.testing.assert_array_equal(serial[1], chi_squares)
        np.testing.assert_array_equal(serial[2], states)

    def test_stop_halts_the_workers(self):
        """``AsyncFit.stop`` must actually stop the fit, so an aborted GUI run
        does not keep burning CPU into arrays nobody will read.

        A single worker on purpose: ``stop`` is set from this thread only once
        the pool is already running, so a full pool on a loaded machine can
        claim every spot before it gets there and the test would fail for a
        reason that has nothing to do with the flag."""
        calibration, spots, initial, _, _, _ = self._batch(n_spots=2000)
        n_spots = len(spots)
        fit = splinefit.fit_spots_async(
            splinefit.KIND_3D,
            spots,
            precision._spline_coeff_reshaped(calibration),
            np.tile(IDENTITY, (n_spots, 1, 1)),
            np.zeros((n_spots, 1, 2)),
            initial,
            _default_seeds(calibration),
            True,
            tolerance=splinefit.TOLERANCE_MULTI_START,
            max_iterations=splinefit.MAX_ITERATIONS_MULTI_START,
            n_threads=1,
        )
        fit.stop()
        _drain(fit)
        assert fit.current[0] < n_spots

    def test_progress_callback_counts_every_spot(self):
        calibration, spots, initial, _, _, _ = self._batch(n_spots=16)
        n_spots = len(spots)
        seen = []
        splinefit.fit_spots(
            splinefit.KIND_3D,
            spots,
            precision._spline_coeff_reshaped(calibration),
            np.tile(IDENTITY, (n_spots, 1, 1)),
            np.zeros((n_spots, 1, 2)),
            initial,
            np.zeros(1),
            False,
            progress_callback=seen.append,
        )
        assert seen == list(range(1, n_spots + 1))


class TestPerSpotJacobians:
    """The channel Jacobian is per spot, like the ROI residual.

    An affine registration gives every spot the same Jacobian, so nothing
    downstream would notice if a kernel silently read row 0 for all of them -
    which is exactly the bug a projective or polynomial registration would hit.
    These tests make the rows genuinely differ.
    """

    def _scene(self, n_spots=6):
        calibration, terms = _astigmatic_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        spots = np.repeat(
            TestMultichannelFits()._spots(terms, [(5000.0, 30.0)] * 2),
            n_spots,
            axis=0,
        ).astype(np.float32)
        initial = np.tile(
            np.array(
                [
                    [
                        float(spots.max() - spots.min()),
                        0.0,
                        0.0,
                        -calibration["z_center"],
                        float(spots.min()),
                    ]
                ]
            ),
            (n_spots, 1),
        )
        residuals = np.tile(TestMultichannelFits.RESIDUALS, (n_spots, 1, 1))
        return calibration, spots, initial, residuals

    @staticmethod
    def _ramped(n_spots):
        """Per-spot Jacobians whose channel-1 x-scale ramps across the field."""
        jac = np.tile(TestMultichannelFits.JACOBIANS, (n_spots, 1, 1)).astype(
            np.float64
        )
        jac[:, 1, 0] += np.linspace(-0.05, 0.05, n_spots)
        return np.ascontiguousarray(jac)

    def test_each_spot_uses_its_own_jacobian(self):
        """Rows that differ must give fits that differ; a kernel indexing
        ``jac[0]`` would return identical results for every spot."""
        n_spots = 6
        calibration, spots, initial, residuals = self._scene(n_spots)
        varying = self._ramped(n_spots)
        constant = np.tile(
            TestMultichannelFits.JACOBIANS, (n_spots, 1, 1)
        ).astype(np.float64)

        def fit(jac):
            theta, _, _, _ = _run(
                splinefit.KIND_3D,
                calibration,
                spots,
                initial.copy(),
                False,
                seeds=_default_seeds(calibration),
                jacobians=jac,
                residuals=residuals,
            )
            return theta

        theta_varying = fit(varying)
        theta_constant = fit(constant)
        # the middle row of the ramp is the constant value, so it must agree;
        # the ends must not
        assert np.allclose(
            theta_varying[n_spots // 2 - 1 : n_spots // 2 + 1, 1],
            theta_constant[n_spots // 2 - 1 : n_spots // 2 + 1, 1],
            atol=5e-3,
        )
        assert abs(theta_varying[0, 1] - theta_constant[0, 1]) > 1e-3
        assert abs(theta_varying[-1, 1] - theta_constant[-1, 1]) > 1e-3
        # and the fitted x must move monotonically with the ramp
        assert np.all(np.diff(theta_varying[:, 1]) < 0)

    def test_serial_and_threaded_agree_with_varying_jacobians(self):
        """The threaded path chunks the spots; a chunk that forgets to slice
        the Jacobians reads another spot's row."""
        n_spots = 6
        calibration, spots, initial, residuals = self._scene(n_spots)
        jac = self._ramped(n_spots)
        coefficients = precision._spline_coeff_reshaped(calibration)
        args = (
            splinefit.KIND_3D,
            spots,
            coefficients,
            jac,
            residuals,
            initial,
            _default_seeds(calibration),
            True,
        )
        serial = splinefit.fit_spots(*args, mle=False)
        fit = splinefit.fit_spots_async(*args, mle=False)
        _drain(fit)
        theta, _, _, _ = fit.results()
        np.testing.assert_array_equal(serial[0], theta)


class TestInputValidation:
    """Shape checks the kernels rely on but cannot make themselves."""

    def _base(self, **overrides):
        calibration, _ = _astigmatic_calibration()
        kwargs = {
            "kind": splinefit.KIND_3D,
            "spots": np.zeros((2, 1, BOX, BOX), np.float32),
            "coefficients": precision._spline_coeff_reshaped(calibration),
            "jacobians": np.tile(IDENTITY, (2, 1, 1)),
            "residuals": np.zeros((2, 1, 2)),
            "initial_parameters": np.zeros((2, 5)),
        }
        kwargs.update(overrides)
        return kwargs

    def test_accepts_valid_input(self):
        splinefit._check_inputs(**self._base())

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"spots": np.zeros((2, BOX, BOX), np.float32)}, "channel-major"),
            ({"spots": np.zeros((2, 1, BOX, BOX + 1), np.float32)}, "square"),
            ({"initial_parameters": np.zeros((2, 4))}, "initial_parameters"),
            ({"jacobians": np.zeros((2, 1, 5))}, "jacobians"),
            ({"residuals": np.zeros((2, 2, 2))}, "residuals"),
        ],
    )
    def test_rejects_bad_shapes(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            splinefit._check_inputs(**self._base(**overrides))

    def test_rejects_coefficients_of_the_wrong_rank(self):
        calibration, _ = _flat_calibration()
        with pytest.raises(ValueError, match="dimensions"):
            splinefit._check_inputs(
                **self._base(
                    coefficients=precision._spline_coeff_reshaped(calibration)
                )
            )

    def test_rejects_multichannel_2d(self):
        calibration, _ = _flat_calibration()
        coefficients = precision._spline_coeff_reshaped(calibration)
        with pytest.raises(ValueError, match="no multichannel 2D"):
            splinefit._check_inputs(
                kind=splinefit.KIND_2D,
                spots=np.zeros((2, 2, BOX, BOX), np.float32),
                coefficients=np.repeat(coefficients, 2, axis=0),
                jacobians=np.tile(IDENTITY, (2, 2, 1)),
                residuals=np.zeros((2, 2, 2)),
                initial_parameters=np.zeros((2, 4)),
            )

    def test_handles_no_spots(self):
        calibration, _ = _astigmatic_calibration()
        theta, chi_squares, states, iterations = splinefit.fit_spots(
            splinefit.KIND_3D,
            np.zeros((0, 1, BOX, BOX), np.float32),
            precision._spline_coeff_reshaped(calibration),
            np.tile(IDENTITY, (1, 1, 1)),
            np.zeros((1, 1, 2)),
            np.zeros((0, 5)),
            np.zeros(1),
            False,
        )
        assert theta.shape == (0, 5)
        assert len(chi_squares) == 0


# ---------------------------------------------------------------------------
# End-to-end: the CPU spline through localize.fit
# ---------------------------------------------------------------------------
def _synthetic_movie(
    calibration,
    terms,
    n_frames=3,
    size=64,
    amplitude=3000.0,
    background=100.0,
    seed=0,
):
    """A small Poisson movie with two spots per frame at known positions, plus
    the identifications and the ground truth."""
    rng = np.random.default_rng(seed)
    clean = np.full((n_frames, size, size), background)
    truth = []
    radius = BOX // 2
    for frame in range(n_frames):
        for k in range(2):
            row, column = 16 + 24 * k, 20 + 22 * k
            z_native = 6.0 + 4.0 * frame
            phi = _reference_model(terms, BOX, DX, DY, z_native)[0]
            clean[
                frame,
                row - radius : row + radius + 1,
                column - radius : column + radius + 1,
            ] += (
                amplitude * phi
            )
            truth.append((frame, column, row, z_native))
    movie = rng.poisson(clean).astype(np.uint16)
    info = [
        {
            "Byte Order": "<",
            "Data Type": "uint16",
            "Frames": n_frames,
            "Height": size,
            "Width": size,
        }
    ]
    identifications = pd.DataFrame(
        {
            "frame": np.array([t[0] for t in truth], np.uint32),
            "x": np.array([t[1] for t in truth], np.int32),
            "y": np.array([t[2] for t in truth], np.int32),
            "net_gradient": np.full(len(truth), 5000.0, np.float32),
        }
    )
    return movie, info, identifications, truth


def _native_z(locs, calibration):
    """Undo locs_from_fits_spline's z reconstruction, back to slices."""
    return calibration["z_center"] - locs.z.values / calibration["z_step_nm"]


class TestFit2DIntegration:
    """``localize.fit`` with the CPU spline codes."""

    @pytest.fixture
    def scene(self, picasso_movie_factory):
        calibration, terms = _astigmatic_calibration()
        movie_array, info, identifications, truth = _synthetic_movie(
            calibration, terms
        )
        movie = picasso_movie_factory(movie_array, info)
        camera_info = {
            "Baseline": 100,
            "Sensitivity": 1.0,
            "Gain": 1,
            "Pixelsize": 130,
            "Qe": 1.0,
        }
        return calibration, movie, info, camera_info, identifications, truth

    @pytest.mark.parametrize("method", ["spline", "spline-mle"])
    def test_recovers_ground_truth(self, scene, method):
        calibration, movie, info, camera_info, identifications, truth = scene
        locs, fit_info = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            fitting_method=method,
            spline_calibration=calibration,
        )
        assert len(locs) == len(truth)
        dx = locs.x.values - np.array([t[1] for t in truth])
        dy = locs.y.values - np.array([t[2] for t in truth])
        dz = _native_z(locs, calibration) - np.array([t[3] for t in truth])
        assert abs(dx.mean() - DX) < 0.02
        assert abs(dy.mean() - DY) < 0.02
        assert np.abs(dz).max() < 0.5
        assert np.all(np.isfinite(locs.lpx)) and np.all(locs.lpx > 0)
        assert np.all(np.isfinite(locs.lpz))
        # per-estimator goodness-of-fit column, as on the GPU
        if method == "spline-mle":
            assert "log_likelihood" in locs and "chi_square" not in locs
        else:
            assert "chi_square" in locs and "log_likelihood" not in locs

    def test_metadata_records_the_device_and_schedule(self, scene):
        calibration, movie, info, camera_info, identifications, _ = scene
        _, fit_info = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            fitting_method="spline",
            spline_calibration=calibration,
        )
        assert fit_info["Fit method"] == "spline"
        assert fit_info["Spline fit device"] == "CPU"
        assert fit_info["Convergence criterion"] == (
            splinefit.TOLERANCE_MULTI_START
        )
        assert fit_info["Max iterations"] == (
            splinefit.MAX_ITERATIONS_MULTI_START
        )
        assert fit_info["Axial seeds"] == localize._default_n_z_starts(
            calibration
        )

    def test_explicit_schedule_is_honored(self, scene):
        calibration, movie, info, camera_info, identifications, _ = scene
        _, fit_info = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            fitting_method="spline",
            spline_calibration=calibration,
            eps=1e-6,
            max_it=7,
        )
        assert fit_info["Convergence criterion"] == 1e-6
        assert fit_info["Max iterations"] == 7

    def test_progress_and_abort(self, scene):
        calibration, movie, info, camera_info, identifications, truth = scene
        seen = []
        locs, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            fitting_method="spline",
            spline_calibration=calibration,
            progress_callback=seen.append,
        )
        # progress is reported at least once and ends at the spot count
        assert seen and seen[-1] == len(truth)
        assert all(0 <= n <= len(truth) for n in seen)

        aborted, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            fitting_method="spline",
            spline_calibration=calibration,
            abort_callback=lambda: True,
        )
        assert aborted is None

    def test_serial_matches_threaded(self, scene):
        calibration, movie, info, camera_info, identifications, _ = scene
        common = dict(fitting_method="spline", spline_calibration=calibration)
        serial, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            multiprocess=False,
            **common,
        )
        threaded, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            multiprocess=True,
            **common,
        )
        for column in ("x", "y", "z", "photons", "bg"):
            np.testing.assert_array_equal(
                serial[column].values, threaded[column].values
            )

    def test_smaller_box_matches_a_pre_cropped_calibration(self, scene):
        """``box`` smaller than the calibration crops it centrally. The fit and
        the CRLB must crop identically, so fitting against an already-cropped
        calibration has to give the same answer."""
        calibration, movie, info, camera_info, identifications, _ = scene
        small = 7
        full, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=small,
            fitting_method="spline",
            spline_calibration=calibration,
        )
        cropped = localize.crop_spline_calibration(calibration, small)
        pre, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=small,
            fitting_method="spline",
            spline_calibration=cropped,
        )
        for column in ("x", "y", "z", "lpx", "lpz"):
            np.testing.assert_allclose(
                full[column].values, pre[column].values, rtol=1e-6
            )

    def test_2d_calibration_yields_no_z(self, scene):
        _, movie, info, camera_info, identifications, _ = scene
        calibration_2d, _ = _flat_calibration()
        locs, _ = localize.fit(
            movie,
            camera_info=camera_info,
            identifications=identifications,
            box=BOX,
            fitting_method="spline",
            spline_calibration=calibration_2d,
        )
        assert "z" not in locs
        assert len(locs) == len(identifications)


class TestMultichannelIntegration:
    """The multichannel entry points routed to the CPU with ``use_gpu=False``.

    These build a real two-channel scene - the emitter is placed in each
    channel through that channel's affine transform, at a sub-pixel offset from
    the integer ROI - so the fit has to undo both the transform and the ROI
    residual to recover the shared position in the reference frame.
    """

    TRANSFORMS = [
        affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        affine([[0.998, 0.02, 7.3], [-0.018, 1.003, -4.6]]),
    ]

    @pytest.fixture
    def scene(self, picasso_movie_factory):
        calibration, terms = _astigmatic_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        calibration["channel_transforms"] = self.TRANSFORMS
        rng = np.random.default_rng(1)
        n_frames, size = 2, 64
        planes = [np.full((n_frames, size, size), 100.0) for _ in range(2)]
        truth = []
        radius = BOX // 2
        for frame in range(n_frames):
            for k in range(3):
                x0, y0 = 18.0 + 14 * k + DX, 20.0 + 9 * k + DY
                z_native = 6.0 + 5.0 * frame
                truth.append((frame, x0, y0, z_native))
                for channel in range(2):
                    xc, yc = apply_transform(
                        np.array([[x0, y0]]), self.TRANSFORMS[channel]
                    )[0]
                    ix, iy = int(round(xc)), int(round(yc))
                    phi = _reference_model(
                        terms, BOX, xc - ix, yc - iy, z_native
                    )[0]
                    planes[channel][
                        frame,
                        iy - radius : iy + radius + 1,
                        ix - radius : ix + radius + 1,
                    ] += (
                        4000.0 * phi
                    )
        info = [
            {
                "Byte Order": "<",
                "Data Type": "uint16",
                "Frames": n_frames,
                "Height": size,
                "Width": size,
            }
        ]
        movies = [
            picasso_movie_factory(rng.poisson(p).astype(np.uint16), info)
            for p in planes
        ]
        camera_infos = [
            {
                "Baseline": 100,
                "Sensitivity": 1.0,
                "Gain": 1,
                "Pixelsize": 130,
                "Qe": 1.0,
            }
            for _ in range(2)
        ]
        identifications = pd.DataFrame(
            {
                "frame": np.array([t[0] for t in truth], np.uint32),
                "x": np.array([int(round(t[1])) for t in truth], np.int32),
                "y": np.array([int(round(t[2])) for t in truth], np.int32),
                "net_gradient": np.full(len(truth), 5000.0, np.float32),
            }
        )
        return calibration, movies, camera_infos, identifications, truth

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    @pytest.mark.parametrize(
        "link_photons", [True, False], ids=["linked", "decoupled"]
    )
    def test_recovers_shared_position(self, scene, mle, link_photons):
        calibration, movies, camera_infos, identifications, truth = scene
        progress = []
        locs = localize.fit_spline_multichannel(
            movies,
            camera_infos,
            identifications,
            BOX,
            calibration,
            mle=mle,
            link_photons=link_photons,
            use_gpu=False,
            progress_callback=progress.append,
        )
        assert len(locs) == len(truth)
        assert progress, "no progress was reported"
        assert np.abs(locs.x.values - [t[1] for t in truth]).max() < 0.05
        assert np.abs(locs.y.values - [t[2] for t in truth]).max() < 0.05
        z_native = calibration["z_center"] - locs.z.values / (
            calibration["z_step_nm"]
        )
        assert np.abs(z_native - [t[3] for t in truth]).max() < 0.5
        # the photon-decoupled model reports per-channel photons
        for channel in range(2):
            assert (f"photons_ch{channel}" in locs) is not link_photons
            assert (f"rel_photons_ch{channel}" in locs) is not link_photons

    def test_ratiometric_assigns_colors(self, scene):
        calibration, movies, camera_infos, identifications, truth = scene
        calibration = dict(
            calibration, photon_ratios=np.array([[1.0, 0.25], [0.25, 1.0]])
        )
        locs = localize.fit_spline_multichannel_ratiometric(
            movies,
            camera_infos,
            identifications,
            BOX,
            calibration,
            use_gpu=False,
        )
        assert len(locs) == len(truth)
        assert "color" in locs
        assert set(np.unique(locs.color.values)) <= {0, 1}

    def test_explicit_gpu_request_without_gpufit_raises(self, scene):
        calibration, movies, camera_infos, identifications, _ = scene
        if localize.CUDA_AVAILABLE:
            pytest.skip("a CUDA device is available")
        with pytest.raises(ImportError, match="use_gpu=False"):
            localize.fit_spline_multichannel(
                movies,
                camera_infos,
                identifications,
                BOX,
                calibration,
                use_gpu=True,
            )


def test_2d_model_ignores_axial_seeds():
    """Parameter 3 of the 2D model is the background, not z. Seeding it would
    silently corrupt the background, so the kernel must ignore axial seeds for
    a 2D fit however it is called."""
    calibration, terms = _flat_calibration()
    amplitude, offset = 2000.0, 15.0
    spots = _spots_from_terms(terms, BOX, amplitude, offset, DX, DY, None)
    initial = np.array(
        [[float(spots.max() - spots.min()), 0.0, 0.0, float(spots.min())]]
    )
    theta, _, states, _ = _run(
        splinefit.KIND_2D,
        calibration,
        spots,
        initial,
        False,
        seeds=np.array([-20.0, -10.0, 0.0]),
    )
    assert states[0] == splinefit.FIT_STATE_CONVERGED
    assert abs(theta[0, 3] - offset) < 0.1
    assert abs(theta[0, 1] - DX) < 2e-3
    assert abs(theta[0, 2] - DY) < 2e-3
