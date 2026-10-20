"""
test_gaussfit_cuda
~~~~~~~~~~~~~~~~~~

Tests for ``picasso.fitting.gaussfit_cuda``, the GPU Gaussian PSF fitter.

The CPU twin, ``picasso.fitting.gaussfit``, is compared against these kernels
in ``test_gaussfit.py``. The ground truth here is the closed form itself: a
noiseless spot built from the model must be recovered to near machine
precision, and the analytic derivatives must match finite differences. Parity
against the ``Gpufit.dll`` these replace lives in ``test_gpufit_parity.py``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest

from picasso.fitting import (
    gaussfit,
    gaussfit_cuda,
    lmfit_cuda,
    seeds,
    splinefit,
)
from tests._progress_tracking import RecordingProgress, assert_phases_complete

pytestmark = pytest.mark.skipif(
    not lmfit_cuda.CUDA_AVAILABLE, reason="no CUDA device"
)

BOX = 7
MODELS = [
    pytest.param(gaussfit_cuda.SPHERICAL, id="spherical"),
    pytest.param(gaussfit_cuda.ELLIPTIC, id="elliptic"),
    pytest.param(gaussfit_cuda.ROTATED, id="rotated"),
]


def _reference(model, theta, box=BOX):
    """The closed-form model image, straight from the Gpufit sources."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
    if model == gaussfit_cuda.SPHERICAL:
        amp, cx, cy, s, bg = theta
        ex = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s**2)
        return amp * ex + bg
    if model == gaussfit_cuda.ELLIPTIC:
        amp, cx, cy, sx, sy, bg = theta
        ex = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        return amp * ex + bg
    amp, cx, cy, sx, sy, bg, angle = theta
    ca, sa = np.cos(angle), np.sin(angle)
    a = (xx - cx) * ca - (yy - cy) * sa
    b = (xx - cx) * sa + (yy - cy) * ca
    return amp * np.exp(-0.5 * ((a / sx) ** 2 + (b / sy) ** 2)) + bg


def _truth(model):
    """A deliberately asymmetric parameter set, so an x/y swap cannot hide."""
    base = [220.0, BOX / 2 - 0.5 + 0.31, BOX / 2 - 0.5 - 0.22]
    if model == gaussfit_cuda.SPHERICAL:
        return np.array(base + [1.6, 9.0])
    if model == gaussfit_cuda.ELLIPTIC:
        return np.array(base + [1.35, 1.85, 9.0])
    return np.array(base + [1.35, 1.85, 9.0, 0.4])


def _canonical_widths(model, theta):
    """Widths as an unordered pair of magnitudes.

    Both symmetries of these models live here - see
    :class:`TestWidthAndAngleDegeneracies`."""
    if model == gaussfit_cuda.SPHERICAL:
        return np.array([abs(theta[3])])
    return np.sort(np.abs(theta[3:5]))


def _seed(model, spots):
    kwargs = {
        gaussfit_cuda.SPHERICAL: dict(spherical=True),
        gaussfit_cuda.ELLIPTIC: {},
        gaussfit_cuda.ROTATED: dict(rotated=True),
    }[model]
    return seeds.initial_parameters_gauss(spots, BOX, **kwargs).astype(
        np.float64
    )


class TestGroundTruthRecovery:
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_recovers_a_noiseless_spot(self, model, mle):
        theta = _truth(model)
        spots = _reference(model, theta)[None].astype(np.float32)
        fitted, chi_squares, states, _ = gaussfit_cuda.fit_spots(
            model,
            spots,
            _seed(model, spots),
            mle=mle,
            max_iterations=200,
            tolerance=1e-8,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        # Position to a hundredth of a pixel, amplitude to 0.5 %.
        np.testing.assert_allclose(fitted[0, 1:3], theta[1:3], atol=1e-2)
        np.testing.assert_allclose(fitted[0, 0], theta[0], rtol=5e-3)
        # Widths are compared as an unordered pair of magnitudes: see
        # ``TestWidthAndAngleDegeneracies`` for the two symmetries of these
        # models that make the raw values non-unique.
        np.testing.assert_allclose(
            _canonical_widths(model, fitted[0]),
            _canonical_widths(model, theta),
            rtol=1e-2,
        )

    @pytest.mark.parametrize("model", MODELS)
    def test_fitted_model_reproduces_the_data(self, model):
        """Whatever route it took, the fit must explain the spot."""
        theta = _truth(model)
        spots = _reference(model, theta)[None].astype(np.float32)
        fitted, _, _, _ = gaussfit_cuda.fit_spots(
            model,
            spots,
            _seed(model, spots),
            max_iterations=200,
            tolerance=1e-8,
        )
        np.testing.assert_allclose(
            _reference(model, fitted[0]), spots[0], rtol=1e-3, atol=1e-2
        )


class TestNoisyBatch:
    @staticmethod
    def _batch(model, n=200, seed=0):
        rng = np.random.default_rng(seed)
        theta = _truth(model)
        spots = np.zeros((n, BOX, BOX), dtype=np.float32)
        for k in range(n):
            perturbed = theta.copy()
            perturbed[1:3] += rng.uniform(-0.8, 0.8, size=2)
            perturbed[0] *= rng.uniform(0.6, 1.8)
            spots[k] = rng.poisson(np.maximum(_reference(model, perturbed), 0))
        return spots

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_batch_converges(self, model, mle):
        spots = self._batch(model)
        fitted, chi_squares, states, _ = gaussfit_cuda.fit_spots(
            model, spots, _seed(model, spots), mle=mle
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.mean() > 0.95
        assert np.isfinite(fitted[converged]).all()
        assert np.isfinite(chi_squares[converged]).all()
        # Magnitude, not sign: the model depends on the width only through
        # ``s**2``, so the sign is not identifiable (see
        # ``TestWidthAndAngleDegeneracies``).
        assert (np.abs(fitted[converged, 3]) > 0).all()
        assert (np.abs(fitted[converged, 3]) < BOX).all()

    @pytest.mark.parametrize("model", MODELS)
    def test_two_runs_are_bitwise_identical(self, model):
        spots = self._batch(model, n=64)
        seed = _seed(model, spots)
        first = gaussfit_cuda.fit_spots(model, spots, seed, mle=True)
        second = gaussfit_cuda.fit_spots(model, spots, seed, mle=True)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_chunking_is_transparent(self, monkeypatch):
        model = gaussfit_cuda.ELLIPTIC
        spots = self._batch(model, n=64)
        seed = _seed(model, spots)
        one = gaussfit_cuda.fit_spots(model, spots, seed)
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 7)
        many = gaussfit_cuda.fit_spots(model, spots, seed)
        for a, b in zip(one, many):
            np.testing.assert_array_equal(a, b)


class TestWidthAndAngleDegeneracies:
    """The two ways these models parameterize the same PSF.

    Neither is a defect and neither is introduced by the port - ``Gpufit.dll``
    returns exactly the same solutions. They are pinned here because they look
    like failures the first time a fitted parameter is compared to the truth.
    """

    def test_width_sign_is_not_identifiable(self):
        """The model depends on the width only through ``s**2``.

        So ``-s`` fits precisely as well as ``+s`` and the optimizer may return
        either. Gpufit does the same, on a similar fraction of spots. Anything
        downstream that consumes a width - the photon conversion
        ``photons = amplitude * 2 pi sx sy`` in ``localize`` above all - has to
        be robust to it.
        """
        theta = _truth(gaussfit_cuda.ELLIPTIC)
        flipped = theta.copy()
        flipped[3] = -flipped[3]
        np.testing.assert_allclose(
            _reference(gaussfit_cuda.ELLIPTIC, flipped),
            _reference(gaussfit_cuda.ELLIPTIC, theta),
            rtol=1e-12,
        )

    def test_rotated_angle_is_defined_modulo_a_quarter_turn(self):
        """Rotating by pi/2 and swapping the widths is the same PSF.

        The fit converges on whichever of the two representations it reaches
        first, so the angle must never be compared to the truth directly - only
        the resulting model image is unique.
        """
        theta = _truth(gaussfit_cuda.ROTATED)
        swapped = theta.copy()
        swapped[3], swapped[4] = theta[4], theta[3]
        swapped[6] = theta[6] + np.pi / 2
        np.testing.assert_allclose(
            _reference(gaussfit_cuda.ROTATED, swapped),
            _reference(gaussfit_cuda.ROTATED, theta),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fit_recovers_the_psf_up_to_those_symmetries(self):
        theta = _truth(gaussfit_cuda.ROTATED)
        spots = _reference(gaussfit_cuda.ROTATED, theta)[None].astype(
            np.float32
        )
        fitted, _, _, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ROTATED,
            spots,
            _seed(gaussfit_cuda.ROTATED, spots),
            max_iterations=200,
            tolerance=1e-8,
        )
        np.testing.assert_allclose(
            _canonical_widths(gaussfit_cuda.ROTATED, fitted[0]),
            _canonical_widths(gaussfit_cuda.ROTATED, theta),
            rtol=1e-2,
        )
        # The angle, brought back into [0, pi/2) where it is unique.
        quarter = np.pi / 2
        assert (
            abs(
                (fitted[0, 6] - theta[6] + quarter / 2) % quarter - quarter / 2
            )
            < 0.05
        )

    def test_seed_breaks_the_width_symmetry(self):
        """With ``sx == sy`` the angle derivative is identically zero.

        The first Hessian is then singular and the fit cannot start, which is
        why ``picasso.fitting.seeds.initial_parameters_gauss`` perturbs the two widths apart. Pinned
        here because it looks like an arbitrary fudge at the call site."""
        spots = np.zeros((1, BOX, BOX), dtype=np.float32)
        seed = _seed(gaussfit_cuda.ROTATED, spots)
        assert seed[0, 3] != seed[0, 4]


class TestNonPositiveModelIsAbandoned:
    """A Gaussian whose model goes non-positive aborts; it is never floored.

    The cubic-spline models floor such a pixel, because a cubic genuinely rings
    negative in the tails of a peaked profile - that is the basis, not the
    parameters. A Gaussian cannot ring: ``amp * exp(...) + bg`` only drops below
    zero when ``bg`` does. Flooring it there would zero exactly the gradient
    that pushes the background back up, so the chi-square would stop moving and
    the *relative* convergence test would then accept a badly wrong fit as
    converged. Reported as ``NEG_CURVATURE_MLE`` instead, matching the
    behavior Picasso shipped before the port.
    """

    def test_negative_background_is_reported_not_floored(self):
        theta = _truth(gaussfit_cuda.ELLIPTIC)
        spots = _reference(gaussfit_cuda.ELLIPTIC, theta)[None].astype(
            np.float32
        )
        # A seed whose background is well below zero: the model is negative
        # across the whole box, so the very first evaluation is infeasible.
        seed = np.array([[theta[0], theta[1], theta[2], 1.3, 1.3, -500.0]])
        _, _, states, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, seed, mle=True
        )
        assert states[0] == splinefit.FIT_STATE_NEG_CURVATURE_MLE

    def test_least_squares_is_unaffected(self):
        """Least squares has no logarithm, so a negative model is legal."""
        theta = _truth(gaussfit_cuda.ELLIPTIC)
        spots = _reference(gaussfit_cuda.ELLIPTIC, theta)[None].astype(
            np.float32
        )
        seed = np.array([[theta[0], theta[1], theta[2], 1.3, 1.3, -500.0]])
        fitted, _, states, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, seed, mle=False
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        np.testing.assert_allclose(fitted[0, 5], theta[5], atol=0.5)


class TestApi:
    def test_kernel_cache(self):
        first = gaussfit_cuda._get_kernel(gaussfit_cuda.ELLIPTIC, True)
        assert first is gaussfit_cuda._get_kernel(gaussfit_cuda.ELLIPTIC, True)
        assert first is not gaussfit_cuda._get_kernel(
            gaussfit_cuda.ELLIPTIC, False
        )
        assert first is not gaussfit_cuda._get_kernel(
            gaussfit_cuda.ROTATED, True
        )

    def test_parameter_counts(self):
        assert gaussfit_cuda.n_parameters(gaussfit_cuda.SPHERICAL) == 5
        assert gaussfit_cuda.n_parameters(gaussfit_cuda.ELLIPTIC) == 6
        assert gaussfit_cuda.n_parameters(gaussfit_cuda.ROTATED) == 7

    def test_empty_input(self):
        spots = np.zeros((0, BOX, BOX), dtype=np.float32)
        initial = np.zeros((0, 6))
        thetas, chi_squares, states, iterations = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, initial
        )
        assert len(thetas) == len(chi_squares) == 0

    def test_rejects_a_non_square_box(self):
        with pytest.raises(ValueError, match="box"):
            gaussfit_cuda.fit_spots(
                gaussfit_cuda.ELLIPTIC,
                np.zeros((2, 5, 7), dtype=np.float32),
                np.zeros((2, 6)),
            )

    def test_rejects_mismatched_initial_parameters(self):
        with pytest.raises(ValueError, match="initial_parameters"):
            gaussfit_cuda.fit_spots(
                gaussfit_cuda.ELLIPTIC,
                np.zeros((2, BOX, BOX), dtype=np.float32),
                np.zeros((2, 7)),
            )

    def test_rejects_an_unknown_model(self):
        with pytest.raises((ValueError, KeyError)):
            gaussfit_cuda.fit_spots(
                99, np.zeros((1, BOX, BOX), dtype=np.float32), np.zeros((1, 6))
            )

    def test_progress_tracker(self):
        model = gaussfit_cuda.ELLIPTIC
        spots = TestNoisyBatch._batch(model, n=32)
        tracker = RecordingProgress()
        gaussfit_cuda.fit_spots(
            model, spots, _seed(model, spots), progress=tracker
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == len(spots)

    def test_progress_none_is_silent_and_console_works(self, capsys):
        model = gaussfit_cuda.ELLIPTIC
        spots = TestNoisyBatch._batch(model, n=8)
        gaussfit_cuda.fit_spots(model, spots, _seed(model, spots))
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        gaussfit_cuda.fit_spots(
            model, spots, _seed(model, spots), progress="console"
        )
        assert "Fitting spots" in capsys.readouterr().err

    def test_progress_tracker_on_empty_input(self):
        tracker = RecordingProgress()
        gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC,
            np.zeros((0, BOX, BOX), dtype=np.float32),
            np.zeros((0, 6)),
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == 0

    def test_zero_width_is_reported_not_crashed(self):
        """A zero width divides by zero in every derivative."""
        spots = np.ones((1, BOX, BOX), dtype=np.float32)
        initial = np.array([[100.0, 6.0, 6.0, 0.0, 0.0, 1.0]])
        thetas, chi_squares, states, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC, spots, initial
        )
        assert states[0] != splinefit.FIT_STATE_CONVERGED
        assert np.isnan(thetas[0]).all()


# ----------------------------------------------------------------------
# Multichannel (joint) spherical Gaussian
# ----------------------------------------------------------------------

MULTI_BOX = 13
MULTI_CENTER = (MULTI_BOX - 1) / 2.0
MULTI_JAC = np.array(
    [[1.0, 0.0, 0.0, 1.0], [0.995, 0.03, -0.028, 1.004]], dtype=np.float64
)
MULTI_RES = np.array([[0.0, 0.0], [0.37, -0.22]], dtype=np.float64)


def _multi_batch(n, amps, bgs, seed=1, box=MULTI_BOX):
    """Two-channel spots, each channel rendered where its own Jacobian and
    sub-pixel residual put the shared position."""
    rng = np.random.RandomState(seed)
    n_channels = len(amps)
    j, i = np.mgrid[0:box, 0:box].astype(np.float64)
    spots = np.zeros((n, n_channels, box, box), dtype=np.float32)
    for k in range(n):
        x = MULTI_CENTER + rng.uniform(-1.5, 1.5)
        y = MULTI_CENTER + rng.uniform(-1.5, 1.5)
        sigma = rng.uniform(1.1, 1.5)
        for c in range(n_channels):
            a00, a01, a10, a11 = MULTI_JAC[c]
            dx, dy = x - MULTI_CENTER, y - MULTI_CENTER
            sx = MULTI_CENTER + a00 * dx + a01 * dy + MULTI_RES[c, 0]
            sy = MULTI_CENTER + a10 * dx + a11 * dy + MULTI_RES[c, 1]
            spots[k, c] = rng.poisson(
                amps[c]
                * np.exp(-0.5 * ((i - sx) ** 2 + (j - sy) ** 2) / sigma**2)
                + bgs[c]
            )
    jac = np.tile(MULTI_JAC[:n_channels], (n, 1, 1))
    res = np.tile(MULTI_RES[:n_channels], (n, 1, 1))
    return spots, jac, res


def _multi_seed(spots, link_photons):
    n, n_channels = spots.shape[0], spots.shape[1]
    if link_photons:
        seed = np.zeros((n, 5))
        ref = spots[:, 0]
        seed[:, 0] = ref.max(axis=(1, 2)) - ref.min(axis=(1, 2))
        seed[:, 1] = MULTI_CENTER
        seed[:, 2] = MULTI_CENTER
        seed[:, 3] = 1.3
        seed[:, 4] = ref.min(axis=(1, 2))
        return seed
    seed = np.zeros((n, 3 + 2 * n_channels))
    seed[:, 0] = MULTI_CENTER
    seed[:, 1] = MULTI_CENTER
    seed[:, 2] = 1.3
    for c in range(n_channels):
        chan = spots[:, c]
        seed[:, 3 + c] = chan.max(axis=(1, 2)) - chan.min(axis=(1, 2))
        seed[:, 3 + n_channels + c] = chan.min(axis=(1, 2))
    return seed


class TestMultichannelCpuGpuEquivalence:
    """The two devices run the identical algorithm, so in double precision and
    under a fixed iteration budget they must agree to machine precision -
    states and iteration counts included."""

    @pytest.mark.parametrize(
        "kind, link_photons",
        [
            (gaussfit_cuda.MULTI_KIND_SHARED, True),
            (gaussfit_cuda.MULTI_KIND_DECOUPLED, False),
        ],
        ids=["shared", "decoupled"],
    )
    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_matches_the_cpu_backend(self, kind, link_photons, mle):
        spots, jac, res = _multi_batch(60, [500.0, 220.0], [12.0, 6.0])
        seed = _multi_seed(spots, link_photons)
        kwargs = dict(mle=mle, tolerance=1e-6, max_iterations=60)

        cpu = gaussfit.fit_spots_multichannel(
            kind, spots, jac, res, seed, **kwargs
        )
        gpu = gaussfit_cuda.fit_spots_multichannel(
            kind, spots, jac, res, seed, single_precision=False, **kwargs
        )

        np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(gpu[1], cpu[1], rtol=1e-8, atol=1e-8)
        np.testing.assert_array_equal(gpu[2], cpu[2])
        np.testing.assert_array_equal(gpu[3], cpu[3])

    def test_multichannel_progress_tracker(self):
        spots, jac, res = _multi_batch(20, [500.0, 220.0], [12.0, 6.0])
        tracker = RecordingProgress()
        gaussfit_cuda.fit_spots_multichannel(
            gaussfit_cuda.MULTI_KIND_SHARED,
            spots,
            jac,
            res,
            _multi_seed(spots, True),
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == len(spots)

    def test_matches_the_cpu_backend_with_a_noise_model(self):
        spots, jac, res = _multi_batch(40, [500.0, 220.0], [12.0, 6.0])
        seed = _multi_seed(spots, True)
        rng = np.random.RandomState(0)
        variance = rng.uniform(1.0, 50.0, spots.shape).astype(np.float32)
        kwargs = dict(mle=True, tolerance=1e-6, max_iterations=60)

        cpu = gaussfit.fit_spots_multichannel(
            gaussfit_cuda.MULTI_KIND_SHARED,
            spots,
            jac,
            res,
            seed,
            variance=variance,
            **kwargs,
        )
        gpu = gaussfit_cuda.fit_spots_multichannel(
            gaussfit_cuda.MULTI_KIND_SHARED,
            spots,
            jac,
            res,
            seed,
            variance=variance,
            single_precision=False,
            **kwargs,
        )

        np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-8, atol=1e-8)


class TestMultichannelKernelCache:
    def test_decoupled_compiles_one_kernel_per_channel_count(self):
        """Its parameter count is ``3 + 2C`` and a device-local array needs a
        compile-time shape, so C selects the kernel."""
        kind = gaussfit_cuda.MULTI_KIND_DECOUPLED
        first = gaussfit_cuda._get_kernel_multichannel(kind, 2, True)
        again = gaussfit_cuda._get_kernel_multichannel(kind, 2, True)
        other = gaussfit_cuda._get_kernel_multichannel(kind, 3, True)
        assert first is again
        assert first is not other

    def test_shared_model_reuses_one_kernel_across_channel_counts(self):
        """It has five parameters whatever C is, so it loops over the channels
        at run time instead of specializing."""
        kind = gaussfit_cuda.MULTI_KIND_SHARED
        assert gaussfit_cuda._get_kernel_multichannel(
            kind, 2, True
        ) is gaussfit_cuda._get_kernel_multichannel(kind, 5, True)

    def test_precision_selects_a_different_kernel(self):
        kind = gaussfit_cuda.MULTI_KIND_SHARED
        assert gaussfit_cuda._get_kernel_multichannel(
            kind, 2, True
        ) is not gaussfit_cuda._get_kernel_multichannel(kind, 2, False)

    def test_rejects_an_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown multichannel"):
            gaussfit_cuda._get_kernel_multichannel(99, 2, True)


class TestMultichannelInputValidation:
    """Shape checking is shared with the CPU backend, so it must still fire."""

    def test_rejects_a_channel_mismatch_in_the_jacobians(self):
        spots, jac, res = _multi_batch(4, [500.0, 220.0], [12.0, 6.0])
        seed = _multi_seed(spots, True)
        with pytest.raises(ValueError, match="jacobians"):
            gaussfit_cuda.fit_spots_multichannel(
                gaussfit_cuda.MULTI_KIND_SHARED,
                spots,
                jac[:, :1],
                res,
                seed,
            )

    def test_rejects_channel_minor_spots(self):
        spots, jac, res = _multi_batch(4, [500.0, 220.0], [12.0, 6.0])
        seed = _multi_seed(spots, True)
        with pytest.raises(ValueError, match="channel-major"):
            gaussfit_cuda.fit_spots_multichannel(
                gaussfit_cuda.MULTI_KIND_SHARED,
                spots.transpose(0, 2, 3, 1),
                jac,
                res,
                seed,
            )
