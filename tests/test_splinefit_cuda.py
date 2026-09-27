"""
test_splinefit_cuda
~~~~~~~~~~~~~~~~~~~

Tests for ``picasso.fitting.splinefit_cuda``, the GPU cubic-spline PSF fitter.

The analytic calibrations, the closed-form reference model and the spot
builders are imported from :mod:`tests.test_splinefit` rather than duplicated:
the CPU fitter is the specification the CUDA kernels are transcribed from, so
both backends must be measured against exactly the same ground truth.

The two devices are compared as follows. The sharp test is a **fixed
iteration budget** - ``tolerance=0`` and a handful of iterations, no
multi-start. That removes the convergence branch from the comparison, so what
is left is pure algebra and it cannot be flaky. In double precision the
kernels then have no rounding excuse and must agree to ~1e-9.

Converged fits are deliberately *not* compared value-for-value. A 1e-6
perturbation of the chi-square flips which Levenberg-Marquardt step is accepted
near the optimum, and that in turn flips which axial seed wins when two minima
are nearly degenerate - so a tight comparison there would be testing luck. They
are compared on properties instead: the GPU fit must not land on a worse
chi-square, and it must pick the same axial basin.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from numba import cuda

from picasso import localize
from picasso.fitting import (
    lmfit_cuda,
    precision,
    splinefit,
    splinefit_cuda,
)

from tests.test_splinefit import (
    BOX,
    DX,
    DY,
    IDENTITY,
    NZ,
    _astigmatic_calibration,
    _flat_calibration,
    _reference_model,
    _spots_from_terms,
)

pytestmark = pytest.mark.skipif(
    not lmfit_cuda.CUDA_AVAILABLE, reason="no CUDA device"
)

# Tolerances for the fixed-iteration comparison. The absolute floor matters:
# a noiseless fit drives the chi-square to ~1e-9 photons squared, where a purely
# relative comparison would amplify a 1e-20 difference into a total failure.
_FP64 = dict(rtol=1e-9, atol=1e-12)
_FP32 = dict(rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _link_xyz_case(n_channels, amplitude=800.0, offset=10.0, z=-6.0):
    """Photon-decoupled calibration, spots, initial parameters and jacobians."""
    calibration, terms = _astigmatic_calibration(
        n_channels=n_channels, model="spline-3d-multichannel"
    )
    calibration = dict(calibration)
    calibration["model"] = "spline-3d-multichannel-link-xyz"
    spots = _spots_from_terms(
        terms, BOX, amplitude, offset, DX, DY, z, n_channels=n_channels
    )
    initial = np.array(
        [
            [0.0, 0.0, -(NZ - 1) / 2.0]
            + [float(spots.max() - spots.min())] * n_channels
            + [float(spots.min())] * n_channels
        ]
    )
    jacobians = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (n_channels, 1))
    return calibration, spots, initial, jacobians


def _case(kind, n_channels=1):
    """``(calibration, spots, initial, jacobians)`` for one model kind."""
    if kind == splinefit.KIND_2D:
        calibration, terms = _flat_calibration()
        spots = _spots_from_terms(terms, BOX, 700.0, 8.0, DX, DY, 0.0)
        initial = np.array(
            [
                [
                    float(spots.max() - spots.min()),
                    0.0,
                    0.0,
                    float(spots.min()),
                ]
            ]
        )
        return calibration, spots, initial, IDENTITY
    if kind == splinefit.KIND_3D:
        calibration, terms = _astigmatic_calibration()
        spots = _spots_from_terms(terms, BOX, 900.0, 12.0, DX, DY, -6.0)
        initial = np.array(
            [
                [
                    float(spots.max() - spots.min()),
                    0.0,
                    0.0,
                    -(NZ - 1) / 2.0,
                    float(spots.min()),
                ]
            ]
        )
        return calibration, spots, initial, IDENTITY
    return _link_xyz_case(n_channels)


def _args(kind, calibration, spots, initial, jacobians, seeds=None):
    coefficients = precision._spline_coeff_reshaped(calibration)
    n_spots, n_channels = spots.shape[0], spots.shape[1]
    jacobians = np.asarray(jacobians, dtype=np.float64)
    if jacobians.ndim == 2:
        # (n_channels, 4) -> the per-spot (n_spots, n_channels, 4) the kernels
        # take; an affine registration is the same Jacobian at every spot
        jacobians = np.ascontiguousarray(
            np.tile(jacobians, (max(n_spots, 1), 1, 1))
        )
    residuals = np.zeros((max(n_spots, 1), n_channels, 2))
    apply_seeds = seeds is not None
    z_seeds = np.asarray(seeds, float) if apply_seeds else np.zeros(1)
    return (
        kind,
        spots,
        coefficients,
        jacobians,
        residuals,
        initial,
        z_seeds,
        apply_seeds,
    )


KINDS = [
    pytest.param(splinefit.KIND_2D, 1, id="2d"),
    pytest.param(splinefit.KIND_3D, 1, id="3d"),
    pytest.param(splinefit.KIND_3D, 2, id="3d-multichannel"),
    pytest.param(splinefit.KIND_LINK_XYZ, 2, id="link-xyz-c2"),
    pytest.param(splinefit.KIND_LINK_XYZ, 6, id="link-xyz-c6"),
]


def _multichannel_3d_case(n_channels):
    calibration, terms = _astigmatic_calibration(
        n_channels=n_channels, model="spline-3d-multichannel"
    )
    spots = _spots_from_terms(
        terms, BOX, 900.0, 12.0, DX, DY, -6.0, n_channels=n_channels
    )
    initial = np.array(
        [
            [
                float(spots.max() - spots.min()),
                0.0,
                0.0,
                -(NZ - 1) / 2.0,
                float(spots.min()),
            ]
        ]
    )
    jacobians = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (n_channels, 1))
    return calibration, spots, initial, jacobians


def _build(kind, n_channels):
    if kind == splinefit.KIND_3D and n_channels > 1:
        return _multichannel_3d_case(n_channels)
    return _case(kind, n_channels)


# ---------------------------------------------------------------------------


class TestSplineEvaluation:
    """The device tricubic against the closed form.

    Pins the coefficient layout, which has no compile-time check anywhere: the
    old Gpufit path packed an axis-reordered blob, and feeding that to these
    kernels would produce a plausible-looking but wrong PSF rather than an
    error.
    """

    @staticmethod
    def _evaluate(coefficients, box, x_shift, y_shift, z_native):
        evaluate = splinefit_cuda._make_eval_spline_3d(np.float64)

        @cuda.jit
        def kernel(coeff, box, x_shift, y_shift, pos_z, out):
            idx = cuda.grid(1)
            if idx >= box * box:
                return
            j = idx // box
            i = idx - j * box
            phi, gx, gy, gz = evaluate(
                coeff, 0, i - x_shift, j - y_shift, pos_z
            )
            out[0, j, i] = phi
            out[1, j, i] = gx
            out[2, j, i] = gy
            out[3, j, i] = gz

        out = np.zeros((4, box, box))
        d_out = cuda.to_device(out)
        n = box * box
        kernel[(n + 31) // 32, 32](
            cuda.to_device(np.ascontiguousarray(coefficients)),
            box,
            x_shift,
            y_shift,
            z_native,
            d_out,
        )
        return d_out.copy_to_host()

    def test_matches_the_closed_form(self):
        calibration, terms = _astigmatic_calibration()
        coefficients = precision._spline_coeff_reshaped(calibration)
        z_native = 6.0
        got = self._evaluate(coefficients, BOX, DX, DY, z_native)
        want = _reference_model(terms, BOX, DX, DY, z_native)
        for k, name in enumerate(("phi", "dphi_dx", "dphi_dy", "dphi_dz")):
            np.testing.assert_allclose(
                got[k], want[k], rtol=1e-6, atol=1e-9, err_msg=name
            )

    def test_matches_the_cpu_evaluator(self):
        """Against ``splinefit._eval_spline_3d``, the CPU kernel these are
        transcribed from.

        Both read ``_spline_coeff_reshaped``'s view, but the CPU kernel is
        scalar (one pixel per call, x and y passed separately) while the device
        kernel fills a ``[y, x]`` image, so the pixel-to-argument mapping is
        spelled out per pixel here; getting it wrong is how an x/y swap slips
        through. Both run in double precision, hence the tight tolerance."""
        calibration, _ = _astigmatic_calibration()
        coefficients = precision._spline_coeff_reshaped(calibration)
        z_native = 6.0
        got = self._evaluate(coefficients, BOX, DX, DY, z_native)
        for j in range(BOX):
            for i in range(BOX):
                want = splinefit._eval_spline_3d(
                    coefficients, 0, i - DX, j - DY, z_native
                )
                np.testing.assert_allclose(
                    [got[k][j, i] for k in range(4)],
                    want,
                    rtol=1e-9,
                    atol=1e-12,
                    err_msg=f"pixel (x={i}, y={j})",
                )


class TestFixedIterationEquivalence:
    """The decisive comparison: identical algebra under a fixed budget."""

    @pytest.mark.parametrize("kind,n_channels", KINDS)
    @pytest.mark.parametrize("mle", [False, True])
    @pytest.mark.parametrize(
        "single_precision,tolerances",
        [(False, _FP64), (True, _FP32)],
        ids=["fp64", "fp32"],
    )
    def test_matches_cpu(
        self, kind, n_channels, mle, single_precision, tolerances
    ):
        args = _args(kind, *_build(kind, n_channels))
        schedule = dict(tolerance=0.0, max_iterations=3)
        cpu = splinefit.fit_spots(*args, mle=mle, **schedule)
        gpu = splinefit_cuda.fit_spots(
            *args, mle=mle, single_precision=single_precision, **schedule
        )
        np.testing.assert_allclose(gpu[0], cpu[0], **tolerances)
        np.testing.assert_allclose(gpu[1], cpu[1], **tolerances)
        np.testing.assert_array_equal(gpu[2], cpu[2])
        np.testing.assert_array_equal(gpu[3], cpu[3])

    @pytest.mark.parametrize("kind,n_channels", KINDS)
    def test_double_precision_is_essentially_exact(self, kind, n_channels):
        """Double precision leaves no rounding excuse for a transcription bug."""
        args = _args(kind, *_build(kind, n_channels))
        cpu = splinefit.fit_spots(
            *args, mle=False, tolerance=0.0, max_iterations=5
        )
        gpu = splinefit_cuda.fit_spots(
            *args,
            mle=False,
            tolerance=0.0,
            max_iterations=5,
            single_precision=False,
        )
        np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-12, atol=1e-12)


class TestConvergedFits:
    """Properties, not values - see the module docstring."""

    @staticmethod
    def _noisy_batch(n=200, seed=0):
        calibration, terms = _astigmatic_calibration()
        rng = np.random.default_rng(seed)
        spots = np.zeros((n, 1, BOX, BOX), dtype=np.float32)
        for k in range(n):
            clean = _spots_from_terms(
                terms,
                BOX,
                rng.uniform(500, 1500),
                rng.uniform(5, 20),
                rng.uniform(-0.7, 0.7),
                rng.uniform(-0.7, 0.7),
                rng.uniform(-(NZ - 2), -1.0),
            )[0, 0]
            spots[k, 0] = rng.poisson(np.maximum(clean, 0))
        initial = np.column_stack(
            [
                spots.max((1, 2, 3)) - spots.min((1, 2, 3)),
                np.zeros(n),
                np.zeros(n),
                np.full(n, -(NZ - 1) / 2.0),
                spots.min((1, 2, 3)),
            ]
        ).astype(np.float64)
        return calibration, spots, initial

    @pytest.mark.parametrize("mle", [False, True])
    def test_chi_square_is_not_worse_than_the_cpu(self, mle):
        """The assertion that actually says "the GPU fit is as good".

        Path-independent, so it survives the two backends taking different
        routes to the optimum. The tolerance is 0.1% rather than something
        tighter because single-precision evaluation moves the accepted-step
        boundary: the observed worst case is ~4e-5 relative, on one spot in 200.
        The median is pinned separately, since a systematic bias would show up
        there long before it showed up in the maximum.
        """
        calibration, spots, initial = self._noisy_batch()
        seeds = np.linspace(-(NZ - 1), 0.0, 5)
        args = _args(
            splinefit.KIND_3D, calibration, spots, initial, IDENTITY, seeds
        )
        cpu = splinefit.fit_spots(*args, mle=mle)
        gpu = splinefit_cuda.fit_spots(*args, mle=mle)
        good = np.isfinite(cpu[1]) & np.isfinite(gpu[1])
        assert good.sum() > 0.9 * len(spots)
        relative = (gpu[1][good] - cpu[1][good]) / cpu[1][good]
        assert relative.max() < 1e-3
        assert abs(np.median(relative)) < 1e-5

    @pytest.mark.parametrize("mle", [False, True])
    def test_multistart_picks_the_same_axial_basin(self, mle):
        """Almost every spot lands in the same axial minimum.

        Not *every* spot: an astigmatic PSF can have two axial minima whose
        chi-squares differ in the seventh significant digit, and single-precision
        evaluation is enough to reverse the ranking. Those are genuinely
        bistable spots rather than a broken fit, which is why the escape hatch
        below is not "allow a few failures" but "when they disagree, the GPU
        must not have chosen the worse minimum" - a stronger statement than the
        blanket tolerance it replaces.
        """
        calibration, spots, initial = self._noisy_batch()
        seeds = np.linspace(-(NZ - 1), 0.0, 5)
        args = _args(
            splinefit.KIND_3D, calibration, spots, initial, IDENTITY, seeds
        )
        cpu = splinefit.fit_spots(*args, mle=mle)
        gpu = splinefit_cuda.fit_spots(*args, mle=mle)
        good = np.isfinite(cpu[1]) & np.isfinite(gpu[1])
        dz = np.abs(gpu[0][good, 3] - cpu[0][good, 3])
        assert np.median(dz) < 1e-3
        same_basin = dz < 0.5
        assert same_basin.mean() > 0.98
        # Where they disagree, the GPU found an equally good or better optimum.
        disagreed = ~same_basin
        if disagreed.any():
            assert np.all(
                gpu[1][good][disagreed] <= cpu[1][good][disagreed] * (1 + 1e-3)
            )
        # Lateral position is far better determined than the axial one.
        assert np.all(
            np.abs(
                gpu[0][good, 1:3][same_basin] - cpu[0][good, 1:3][same_basin]
            )
            < 1e-2
        )


class TestDeviceBehavior:
    def test_two_runs_are_bitwise_identical(self):
        """One thread per spot, no atomics, no reduction - so reproducible."""
        calibration, spots, initial = TestConvergedFits._noisy_batch(n=64)
        args = _args(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            IDENTITY,
            np.linspace(-(NZ - 1), 0.0, 5),
        )
        first = splinefit_cuda.fit_spots(*args, mle=True)
        second = splinefit_cuda.fit_spots(*args, mle=True)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_chunking_is_transparent(self, monkeypatch):
        calibration, spots, initial = TestConvergedFits._noisy_batch(n=64)
        args = _args(
            splinefit.KIND_3D,
            calibration,
            spots,
            initial,
            IDENTITY,
            np.linspace(-(NZ - 1), 0.0, 5),
        )
        one_launch = splinefit_cuda.fit_spots(*args, mle=False)
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 7)
        many_launches = splinefit_cuda.fit_spots(*args, mle=False)
        for a, b in zip(one_launch, many_launches):
            np.testing.assert_array_equal(a, b)

    def test_abort_between_chunks(self, monkeypatch):
        calibration, spots, initial = TestConvergedFits._noisy_batch(n=64)
        args = _args(splinefit.KIND_3D, calibration, spots, initial, IDENTITY)
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 8)
        calls = []

        def abort():
            calls.append(1)
            return len(calls) > 2

        thetas, chi_squares, _, _ = splinefit_cuda.fit_spots(
            *args, mle=False, abort_callback=abort
        )
        # Spots never reached keep their placeholder values.
        assert np.isnan(thetas[-1]).all()
        assert np.isinf(chi_squares[-1])
        assert np.isfinite(thetas[0]).all()

    def test_progress_callback_is_monotone_and_complete(self):
        calibration, spots, initial = TestConvergedFits._noisy_batch(n=32)
        args = _args(splinefit.KIND_3D, calibration, spots, initial, IDENTITY)
        seen = []
        splinefit_cuda.fit_spots(
            *args, mle=False, progress_callback=seen.append
        )
        assert seen == sorted(seen) and seen[-1] == len(spots)

    def test_empty_input(self):
        calibration, spots, initial, jacobians = _build(splinefit.KIND_3D, 1)
        args = _args(
            splinefit.KIND_3D,
            calibration,
            spots[:0],
            initial[:0],
            jacobians,
        )
        thetas, chi_squares, states, iterations = splinefit_cuda.fit_spots(
            *args
        )
        assert len(thetas) == len(chi_squares) == 0
        assert len(states) == len(iterations) == 0

    def test_2d_model_ignores_axial_seeds(self):
        """Parameter 3 of the 2D model is the *background*.

        Seeding it would corrupt the fit rather than move it in z, so the
        multi-start must be a no-op here - the same guard the CPU kernel has.
        """
        calibration, spots, initial, jacobians = _build(splinefit.KIND_2D, 1)
        without = splinefit_cuda.fit_spots(
            *_args(splinefit.KIND_2D, calibration, spots, initial, jacobians)
        )
        with_seeds = splinefit_cuda.fit_spots(
            *_args(
                splinefit.KIND_2D,
                calibration,
                spots,
                initial,
                jacobians,
                seeds=np.linspace(-(NZ - 1), 0.0, 5),
            ),
            tolerance=splinefit.TOLERANCE_SINGLE_START,
            max_iterations=splinefit.MAX_ITERATIONS_SINGLE_START,
        )
        np.testing.assert_allclose(with_seeds[0], without[0], rtol=1e-12)


class TestRunSplinefitAbortContract:
    """``localize._run_splinefit`` must report an aborted GPU fit as None.

    Callers (``_fit2d_spline_cpu`` and friends) treat None as "aborted" and
    anything else as a complete result. The GPU backend always returns its
    arrays - with the spots it never reached left as NaN - so the dispatch layer
    is what has to tell the two apart.
    """

    @staticmethod
    def _run(abort, chunk=8):
        calibration, spots, _ = TestConvergedFits._noisy_batch(n=40)
        spots = spots[:, 0]  # localize takes (n, box, box)
        return localize._run_splinefit(
            spots,
            calibration,
            mle=False,
            n_z_starts=1,
            use_gpu=True,
            abort_callback=abort,
        )

    def test_abort_midway_returns_none(self, monkeypatch):
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 8)
        calls = []

        def abort():
            calls.append(1)
            return len(calls) > 1

        assert self._run(abort) is None

    def test_abort_before_any_work_returns_none(self, monkeypatch):
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 8)
        assert self._run(lambda: True) is None

    def test_a_complete_fit_is_not_discarded(self, monkeypatch):
        """The abort turning True *after* the last chunk must not lose the fit.

        This is why the dispatch records whether the callback actually fired at
        a chunk boundary instead of simply re-checking it once the fit returns.
        """
        monkeypatch.setattr(lmfit_cuda, "chunk_rows", lambda *a, **k: 10_000)
        state = {"done": False}
        result = self._run(lambda: state["done"])
        state["done"] = True
        assert result is not None
        assert np.isfinite(result[0]).all()

    def test_without_a_callback(self):
        result = self._run(None)
        assert result is not None and np.isfinite(result[0]).all()


class TestKernelCache:
    def test_link_xyz_compiles_one_kernel_per_channel_count(self):
        """Its parameter count is ``3 + 2C`` and a device-local array needs a
        compile-time shape, so C selects the kernel."""
        first = splinefit_cuda._get_kernel(splinefit.KIND_LINK_XYZ, 2, True)
        again = splinefit_cuda._get_kernel(splinefit.KIND_LINK_XYZ, 2, True)
        other = splinefit_cuda._get_kernel(splinefit.KIND_LINK_XYZ, 3, True)
        assert first is again
        assert first is not other

    def test_fixed_width_models_share_one_kernel_across_channels(self):
        """The shared-amplitude 3D model has 5 parameters whatever C is, so it
        loops over channels at run time instead of specializing."""
        assert splinefit_cuda._get_kernel(
            splinefit.KIND_3D, 1, True
        ) is splinefit_cuda._get_kernel(splinefit.KIND_3D, 4, True)

    def test_precision_selects_a_different_kernel(self):
        assert splinefit_cuda._get_kernel(
            splinefit.KIND_3D, 1, True
        ) is not splinefit_cuda._get_kernel(splinefit.KIND_3D, 1, False)

    def test_parameter_counts(self):
        assert splinefit_cuda.n_parameters(splinefit.KIND_2D, 1) == 4
        assert splinefit_cuda.n_parameters(splinefit.KIND_3D, 3) == 5
        assert splinefit_cuda.n_parameters(splinefit.KIND_LINK_XYZ, 6) == 15


class TestInputValidation:
    """Shape checking is shared with the CPU backend, so it must still fire."""

    def test_rejects_a_channel_mismatch(self):
        calibration, spots, initial, jacobians = _build(splinefit.KIND_3D, 1)
        args = list(
            _args(splinefit.KIND_3D, calibration, spots, initial, jacobians)
        )
        args[3] = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (2, 1))
        with pytest.raises(ValueError, match="jacobians"):
            splinefit_cuda.fit_spots(*args)

    def test_rejects_wrong_coefficient_rank(self):
        calibration2d, spots, initial, jacobians = _build(splinefit.KIND_2D, 1)
        args = list(
            _args(splinefit.KIND_2D, calibration2d, spots, initial, jacobians)
        )
        calibration3d, _ = _astigmatic_calibration()
        args[2] = precision._spline_coeff_reshaped(calibration3d)
        with pytest.raises(ValueError, match="dimensions"):
            splinefit_cuda.fit_spots(*args)
