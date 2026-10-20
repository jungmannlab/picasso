"""
test_gaussfit
~~~~~~~~~~~~~

Tests for ``picasso.fitting.gaussfit``, the CPU Gaussian PSF fitter.

Unlike ``test_gaussfit_cuda``, this module *does* have a twin to compare
against, in two directions:

- **The GPU backend**, which runs the identical algorithm. Under a fixed
  iteration budget and double precision the two must agree to machine
  precision, states and iteration counts included - there is no convergence
  branch left for them to disagree on, so the comparison pins the algebra
  exactly and cannot be flaky.
- **SciPy's MINPACK**, minimizing the *same objective* (the least-squares
  fit of the former ``picasso.gausslq``, re-implemented here as
  ``_minpack_fit``). Different algorithms cannot be compared step for step,
  but they must land on the same optimum - and where they differ, the fit
  that reaches the lower chi-square is the right one. The reference
  evaluates its model in float32, so it is the one with slack.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from concurrent import futures

import numpy as np
import pytest
from scipy import optimize

from picasso.fitting import (
    gaussfit,
    gaussfit_cuda,
    lmfit_cuda,
    precision,
    seeds,
    splinefit,
)
from tests._progress_tracking import RecordingProgress, assert_phases_complete

BOX = 7
CENTER = (BOX - 1) / 2.0

MODELS = [
    pytest.param(gaussfit.SPHERICAL, id="spherical"),
    pytest.param(gaussfit.ELLIPTIC, id="elliptic"),
    pytest.param(gaussfit.ROTATED, id="rotated"),
]


def _reference(model, theta, box=BOX):
    """The closed-form model image, straight from the source papers."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
    if model == gaussfit.SPHERICAL:
        amp, cx, cy, s, bg = theta
        return (
            amp * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / s**2) + bg
        )
    if model == gaussfit.ELLIPTIC:
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
    base = [220.0, CENTER + 0.31, CENTER - 0.22]
    if model == gaussfit.SPHERICAL:
        return np.array(base + [1.6, 9.0])
    if model == gaussfit.ELLIPTIC:
        return np.array(base + [1.35, 1.85, 9.0])
    return np.array(base + [1.35, 1.85, 9.0, 0.4])


def _canonical_widths(model, theta):
    """Widths as an unordered pair of magnitudes.

    The model depends on the widths only through their squares, and the
    rotated one is invariant under (swap sx/sy, rotate by pi/2), so the raw
    values are not unique. See
    ``test_gaussfit_cuda.TestWidthAndAngleDegeneracies``."""
    if model == gaussfit.SPHERICAL:
        return np.array([abs(theta[3])])
    return np.sort(np.abs(theta[3:5]))


def _seed(model, spots):
    """A starting point of the right shape for ``model``."""
    kwargs = {
        gaussfit.SPHERICAL: dict(spherical=True),
        gaussfit.ELLIPTIC: {},
        gaussfit.ROTATED: dict(rotated=True),
    }[model]
    return seeds.initial_parameters_gauss(spots, BOX, **kwargs).astype(
        np.float64
    )


def _minpack_fit(spots, tolerance, max_iterations):
    """Independent least-squares reference: SciPy's MINPACK ``leastsq`` on
    ``photons * PDF_x * PDF_y + bg``, as the former ``picasso.gausslq`` fit
    it - moment seeds, a float32 model and ``maxfev = max_iterations *
    (n_params + 1)``.

    Returns ``[x, y, photons, bg, sx, sy]`` per spot, x/y relative to the box
    center.
    """
    size = spots.shape[1]
    half = size // 2
    grid = np.arange(-half, half + 1, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    def pdf(mu, sigma):
        norm = 0.3989422804014327 / sigma
        return (norm * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)).astype(
            np.float32
        )

    def residuals(theta, spot):
        x, y, photons, bg, sx, sy = theta
        model = np.float32(photons) * np.outer(pdf(y, sy), pdf(x, sx))
        return (spot - (model + np.float32(bg))).astype(np.float32).ravel()

    out = np.empty((len(spots), 6), np.float32)
    for k, spot in enumerate(spots):
        bg = spot.min()
        signal = spot - bg
        total = signal.sum()
        x0 = (signal * xx).sum() / total
        y0 = (signal * yy).sum() / total
        sx0 = np.sqrt((signal * (xx - x0) ** 2).sum() / total)
        sy0 = np.sqrt((signal * (yy - y0) ** 2).sum() / total)
        theta0 = np.array(
            [x0 - half, y0 - half, max(1.0, total), bg, sx0, sy0], np.float32
        )
        out[k] = optimize.leastsq(
            residuals,
            theta0,
            args=(spot,),
            ftol=tolerance,
            xtol=tolerance,
            maxfev=max_iterations * (len(theta0) + 1),
        )[0]
    return out


@pytest.fixture(scope="module")
def lq_and_lm(noisy_batch):
    """MINPACK and ``gaussfit`` run to convergence on the same spots."""
    lq = _minpack_fit(noisy_batch, tolerance=1e-8, max_iterations=400)
    lm = gaussfit.fit_spots(
        gaussfit.ELLIPTIC,
        noisy_batch,
        seeds.initial_parameters_gauss(noisy_batch, BOX).astype(np.float64),
        mle=False,
        tolerance=1e-10,
        max_iterations=400,
    )[0]
    return lq, lm


@pytest.fixture(scope="module")
def noisy_batch():
    """Poisson-noisy elliptical spots over a realistic parameter range."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:BOX, 0:BOX].astype(np.float64)
    n = 120
    spots = np.empty((n, BOX, BOX), np.float32)
    for k in range(n):
        cx = CENTER + rng.uniform(-0.6, 0.6)
        cy = CENTER + rng.uniform(-0.6, 0.6)
        sx, sy = rng.uniform(1.0, 1.6), rng.uniform(1.0, 1.6)
        amp, bg = rng.uniform(300, 1500), rng.uniform(5, 30)
        mu = (
            amp
            * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
            + bg
        )
        spots[k] = rng.poisson(mu)
    return spots


class TestGroundTruthRecovery:
    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_recovers_a_noiseless_spot(self, model, mle):
        theta = _truth(model)
        spots = _reference(model, theta)[None].astype(np.float32)
        fitted, _, states, _ = gaussfit.fit_spots(
            model,
            spots,
            _seed(model, spots),
            mle=mle,
            max_iterations=200,
            tolerance=1e-8,
        )
        assert states[0] == splinefit.FIT_STATE_CONVERGED
        np.testing.assert_allclose(fitted[0, 1:3], theta[1:3], atol=1e-2)
        np.testing.assert_allclose(fitted[0, 0], theta[0], rtol=5e-3)
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
        fitted, _, _, _ = gaussfit.fit_spots(
            model,
            spots,
            _seed(model, spots),
            max_iterations=200,
            tolerance=1e-8,
        )
        np.testing.assert_allclose(
            _reference(model, fitted[0]), spots[0], rtol=1e-3, atol=1e-2
        )


class TestMinpackParity:
    """Same model, same estimator, different optimizer.

    ``_minpack_fit`` minimizes
    ``sum((data - (photons * PDF_x * PDF_y + bg))^2)``; the elliptical LM
    model is the same surface parameterized by peak height instead of
    integrated photons, so the least-squares optimum is identical. The
    parameterization differs only in how the optimizer walks there."""

    @staticmethod
    def _as_reference_layout(theta):
        """LM ``[peak, x, y, sx, sy, bg]`` -> the reference's
        ``[x, y, photons, bg, sx, sy]``, x/y relative to the box center."""
        peak, cx, cy, sx, sy, bg = theta.T
        photons = peak * 2.0 * np.pi * np.abs(sx * sy)
        return np.stack(
            [cx - CENTER, cy - CENTER, photons, bg, np.abs(sx), np.abs(sy)],
            axis=1,
        )

    @staticmethod
    def _chi_square(theta, spots):
        """Least-squares residual of the LM parameters, in float64."""
        yy, xx = np.mgrid[0:BOX, 0:BOX].astype(np.float64)
        out = np.empty(len(theta))
        for k, (peak, cx, cy, sx, sy, bg) in enumerate(theta):
            mu = (
                peak
                * np.exp(
                    -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
                )
                + bg
            )
            out[k] = ((spots[k].astype(np.float64) - mu) ** 2).sum()
        return out

    def test_lands_on_the_same_optimum(self, lq_and_lm):
        lq, lm = lq_and_lm
        mapped = self._as_reference_layout(lm)
        # Positions and widths in pixels, background in photons.
        assert np.nanmax(np.abs(lq[:, 0] - mapped[:, 0])) < 1e-3
        assert np.nanmax(np.abs(lq[:, 1] - mapped[:, 1])) < 1e-3
        assert np.nanmax(np.abs(np.abs(lq[:, 4]) - mapped[:, 4])) < 1e-3
        assert np.nanmax(np.abs(np.abs(lq[:, 5]) - mapped[:, 5])) < 1e-3
        rel_photons = np.abs(lq[:, 2] - mapped[:, 2]) / np.abs(lq[:, 2])
        assert np.nanmax(rel_photons) < 1e-2

    def test_lm_never_finds_the_worse_optimum(self, lq_and_lm, noisy_batch):
        """The load-bearing assertion. Two different algorithms on one
        objective will not stop at the same floating-point point, so compare
        what actually matters: neither may be the worse fit, and the LM
        backend must not be. The reference evaluates its model in float32,
        so it is the one that cannot converge tighter.
        """
        lq, lm = lq_and_lm
        chi_lm = self._chi_square(lm, noisy_batch)
        # the reference's layout, expressed as LM parameters, so one chi-square
        # function scores both.
        peak = lq[:, 2] / (2.0 * np.pi * np.abs(lq[:, 4] * lq[:, 5]))
        as_lm = np.stack(
            [
                peak,
                lq[:, 0] + CENTER,
                lq[:, 1] + CENTER,
                np.abs(lq[:, 4]),
                np.abs(lq[:, 5]),
                lq[:, 3],
            ],
            axis=1,
        )
        chi_lq = self._chi_square(as_lm, noisy_batch)
        assert np.all(chi_lm <= chi_lq * (1 + 1e-6))

    def test_photon_convention_is_not_accidentally_equal(self, lq_and_lm):
        """Guard against the mapping above being a no-op: the LM amplitude is
        a peak height and must *not* equal the reference's photon count."""
        lq, lm = lq_and_lm
        assert np.nanmedian(np.abs(lq[:, 2] - lm[:, 0])) > 1.0


@pytest.mark.skipif(not lmfit_cuda.CUDA_AVAILABLE, reason="no CUDA device")
class TestCpuGpuEquivalence:
    """The CPU and the GPU run the same algorithm, so they must agree."""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_fixed_iterations_agree_exactly(self, model, mle, noisy_batch):
        """Fixed budget, double precision, no convergence branch: this pins
        the algebra and cannot be flaky."""
        seed = _seed(model, noisy_batch)
        cpu = gaussfit.fit_spots(
            model, noisy_batch, seed, mle=mle, tolerance=0.0, max_iterations=3
        )
        gpu = gaussfit_cuda.fit_spots(
            model,
            noisy_batch,
            seed,
            mle=mle,
            tolerance=0.0,
            max_iterations=3,
            single_precision=False,
        )
        np.testing.assert_allclose(cpu[0], gpu[0], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(cpu[1], gpu[1], rtol=1e-9, atol=1e-9)
        # Same algorithm under a fixed budget: even the bookkeeping matches.
        np.testing.assert_array_equal(cpu[2], gpu[2])
        np.testing.assert_array_equal(cpu[3], gpu[3])

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("mle", [False, True])
    def test_converged_positions_agree(self, model, mle, noisy_batch):
        """With the default single-precision GPU evaluation, positions still
        agree far below the shot-noise floor."""
        seed = _seed(model, noisy_batch)
        cpu = gaussfit.fit_spots(model, noisy_batch, seed, mle=mle)[0]
        gpu = gaussfit_cuda.fit_spots(model, noisy_batch, seed, mle=mle)[0]
        finite = np.isfinite(cpu).all(axis=1) & np.isfinite(gpu).all(axis=1)
        assert finite.sum() > 0.9 * len(noisy_batch)
        assert np.abs(cpu[finite, 1:3] - gpu[finite, 1:3]).max() < 1e-4


class TestThreading:
    def test_threaded_matches_serial_bitwise(self, noisy_batch):
        """One thread per spot with no shared state, so the answer cannot
        depend on the scheduling."""
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        serial = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, noisy_batch, seed, mle=True
        )
        fit = gaussfit.fit_spots_async(
            gaussfit.ELLIPTIC, noisy_batch, seed, mle=True
        )
        while not fit.finished():
            pass
        fit.raise_errors()
        for a, b in zip(serial, fit.results()):
            np.testing.assert_array_equal(a, b)

    def test_async_can_be_stopped(self, noisy_batch):
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        fit = gaussfit.fit_spots_async(
            gaussfit.ELLIPTIC, noisy_batch, seed, n_threads=1
        )
        fit.stop()
        while not fit.finished():
            pass
        fit.raise_errors()
        assert fit.current[0] <= len(noisy_batch)
        # every claimed spot was fitted before the worker exited
        assert fit.n_done == fit.current[0]

    @pytest.mark.parametrize("multichannel", [False, True])
    def test_async_counts_finished_spots(self, noisy_batch, multichannel):
        """``n_done`` counts the spots whose results are written, so it ends
        at the spot count once the fit finished."""
        if multichannel:
            n = 40
            spots, _ = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
            jac, res = _multi_geometry(n)
            fit = gaussfit.fit_spots_multichannel_async(
                gaussfit.MULTI_KIND_SHARED,
                spots,
                jac,
                res,
                _shared_seed(spots),
                n_threads=2,
            )
        else:
            n = len(noisy_batch)
            seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
                np.float64
            )
            fit = gaussfit.fit_spots_async(
                gaussfit.ELLIPTIC, noisy_batch, seed, n_threads=2
            )
        seen = []
        while not fit.finished():
            seen.append(fit.n_done)
        fit.raise_errors()
        assert all(0 <= v <= n for v in seen)
        assert all(b >= a for a, b in zip(seen, seen[1:]))
        assert fit.n_done == fit.done[0] == fit.current[0] == n


class TestSchedule:
    def test_defaults_match_the_gpu_backend(self):
        """One definition of the schedule, imported by the GPU module."""
        assert gaussfit_cuda.TOLERANCE is gaussfit.TOLERANCE
        assert gaussfit_cuda.MAX_ITERATIONS is gaussfit.MAX_ITERATIONS

    def test_model_constants_are_shared(self):
        """The GPU module must not define its own model numbering."""
        assert gaussfit_cuda.SPHERICAL is gaussfit.SPHERICAL
        assert gaussfit_cuda.ELLIPTIC is gaussfit.ELLIPTIC
        assert gaussfit_cuda.ROTATED is gaussfit.ROTATED
        assert gaussfit_cuda.n_parameters is gaussfit.n_parameters

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_iteration_cap_bites(self, noisy_batch, mle):
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        capped = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, noisy_batch, seed, mle=mle, max_iterations=2
        )[3]
        free = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, noisy_batch, seed, mle=mle, max_iterations=40
        )[3]
        assert capped.min() >= 0
        assert capped.max() <= 2
        assert free.max() > 2

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_looser_tolerance_stops_earlier(self, noisy_batch, mle):
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        loose = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            mle=mle,
            tolerance=1e-1,
            max_iterations=60,
        )[3]
        tight = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            mle=mle,
            tolerance=1e-8,
            max_iterations=60,
        )[3]
        assert loose.mean() < tight.mean()

    def test_cpu_least_squares_default_is_converged(self, noisy_batch):
        """The CPU least-squares schedule (``TOLERANCE_LSQ_CPU``, kept from
        MINPACK's ``ftol``/``xtol``) is a relative change of the chi-square
        here, not of the parameters. It must still stop at the optimum:
        tightening it moves no position by more than a twentieth of a
        pixel."""
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        default = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            tolerance=gaussfit.TOLERANCE_LSQ_CPU,
            max_iterations=gaussfit.MAX_ITERATIONS_LSQ_CPU,
        )[0]
        tight = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            tolerance=1e-10,
            max_iterations=400,
        )[0]
        assert np.max(np.abs(default[:, 1:3] - tight[:, 1:3])) < 0.05


class TestProgress:
    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_tracker_counts_every_spot(self, noisy_batch, mle):
        """One "Fitting spots" phase over the spots, reporting the number
        of spots fitted so far after each spot."""
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        tracker = RecordingProgress()
        gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            mle=mle,
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == len(noisy_batch)
        assert tracker.phases[0]["values"] == list(range(len(noisy_batch) + 1))

    def test_multichannel_tracker(self):
        n = 6
        spots, _ = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
        jac, res = _multi_geometry(n)
        tracker = RecordingProgress()
        gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED,
            spots,
            jac,
            res,
            _shared_seed(spots),
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == n

    @pytest.mark.parametrize("multichannel", [False, True])
    def test_none_is_silent_and_console_works(
        self, noisy_batch, multichannel, capsys
    ):
        if multichannel:
            n = 4
            spots, _ = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
            jac, res = _multi_geometry(n)
            args = (
                gaussfit.MULTI_KIND_SHARED,
                spots,
                jac,
                res,
                _shared_seed(spots),
            )
            fit = gaussfit.fit_spots_multichannel
        else:
            spots = noisy_batch[:4]
            args = (gaussfit.ELLIPTIC, spots, _seed(gaussfit.ELLIPTIC, spots))
            fit = gaussfit.fit_spots
        fit(*args, progress=None)
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        fit(*args, progress="console")
        assert "Fitting spots" in capsys.readouterr().err

    def test_rejects_an_invalid_progress_argument(self, noisy_batch):
        spots = noisy_batch[:2]
        with pytest.raises(ValueError, match="Invalid progress"):
            gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                spots,
                _seed(gaussfit.ELLIPTIC, spots),
                progress="tqdm",
            )


class TestChiSquare:
    @pytest.mark.parametrize("model", MODELS)
    def test_least_squares_reports_the_residual_sum_of_squares(
        self, model, noisy_batch
    ):
        """The reported chi-square is the sum of squared residuals of the
        fitted model against the spot - recomputed here from the closed
        form."""
        spots = noisy_batch[:20]
        fitted, chi_square, _, _ = gaussfit.fit_spots(
            model, spots, _seed(model, spots), mle=False
        )
        expected = [
            (
                (spots[k].astype(np.float64) - _reference(model, fitted[k]))
                ** 2
            ).sum()
            for k in range(len(spots))
        ]
        np.testing.assert_allclose(chi_square, expected, rtol=1e-6)


class TestInputValidation:
    def test_rejects_non_square_spots(self):
        with pytest.raises(ValueError, match="n_spots, box, box"):
            gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                np.zeros((2, 5, 7), np.float32),
                np.zeros((2, 6)),
            )

    def test_rejects_mismatched_seed_width(self):
        with pytest.raises(ValueError, match="initial_parameters"):
            gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                np.zeros((2, BOX, BOX), np.float32),
                np.zeros((2, 5)),
            )

    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown Gaussian model"):
            gaussfit.fit_spots(
                99, np.zeros((1, BOX, BOX), np.float32), np.zeros((1, 6))
            )

    def test_async_rejects_unknown_model(self):
        """Raised by the caller's thread, before any worker starts."""
        with pytest.raises(ValueError, match="Unknown Gaussian model"):
            gaussfit.fit_spots_async(
                99, np.zeros((1, BOX, BOX), np.float32), np.zeros((1, 6))
            )

    def test_empty_batch(self):
        thetas, chi, states, iters = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            np.zeros((0, BOX, BOX), np.float32),
            np.zeros((0, 6)),
        )
        assert thetas.shape == (0, 6)
        assert len(chi) == len(states) == len(iters) == 0


def _poisson_spots(box, n=60, seed=3):
    """``noisy_batch``, but at an arbitrary box size.

    The widths stay in the 1.0-1.6 range whatever the box is - a real PSF is
    set by the optics, not by how much padding the box has around it. That
    divergence between the true width and a box-proportional seed is the whole
    point of ``TestWideSigmaSeedDoesNotAbortTheFit``.
    """
    rng = np.random.default_rng(seed)
    center = (box - 1) / 2.0
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
    spots = np.empty((n, box, box), np.float32)
    for k in range(n):
        cx = center + rng.uniform(-0.6, 0.6)
        cy = center + rng.uniform(-0.6, 0.6)
        sx, sy = rng.uniform(1.0, 1.6), rng.uniform(1.0, 1.6)
        amp, bg = rng.uniform(300, 1500), rng.uniform(5, 30)
        mu = (
            amp
            * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
            + bg
        )
        spots[k] = rng.poisson(mu)
    return spots


class TestWideSigmaSeedDoesNotAbortTheFit:
    """A too-wide width seed must not end the fit with ``NEG_CURVATURE_MLE``.

    The regression: the driver used to treat a *trial* step whose model went
    non-positive as terminal, as Gpufit's ``LMFitCPP::calc_chi_square`` does.
    Seeded several times wider than the true PSF, the first (undamped) step
    overshoots and drives the background below zero - so the fit aborted on
    iteration 1, rolled the parameters back and wrote *the seed* out as the
    result: sigma pinned to the seed width, x and y at the exact box center.

    Every other test in this module runs at ``BOX = 7``, where the seed lands
    within 0.4 px of the truth and the first step never overshoots, so none of
    them could see it. These parametrize the box - the axis the bug lives on -
    and drive the width seed directly.
    """

    BOXES = [7, 13, 23, 31]

    @pytest.mark.parametrize("width_seed", [2.0, 3.0, 4.0, 5.0])
    def test_a_wide_seed_is_damped_back_not_abandoned(self, width_seed):
        """The bug in isolation: one knob, the seeded sigma.

        The spots and every other seed parameter are held fixed, so a failure
        here can only be the width. The true widths are ~1.0-1.6 px, so even
        the mildest case here starts well over the truth - enough for the
        first, undamped step to overshoot the background below zero.
        """
        spots = _poisson_spots(31, n=40)
        seed = seeds.initial_parameters_gauss(spots, 31).astype(np.float64)
        seed[:, 3] = seed[:, 4] = width_seed
        fitted, _, states, iterations = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            spots,
            seed,
            mle=True,
            max_iterations=200,
            tolerance=1e-8,
        )
        aborted = states == splinefit.FIT_STATE_NEG_CURVATURE_MLE
        assert not aborted.any(), (
            f"{aborted.sum()}/{len(spots)} fits abandoned at a {width_seed} px "
            "width seed"
        )
        # Recovered, not merely alive: the fit walked back to the true width.
        assert np.median(fitted[:, 3]) < 2.0
        assert (iterations > 1).all()

    def test_an_extreme_seed_costs_iterations_but_still_arrives(self):
        """Where the recovery stops being free.

        At 8 px in a 31 px box the model is nearly flat, so amplitude and
        background are strongly correlated and the damped walk back is slow -
        a median of a few hundred iterations rather than the usual ten. It
        still *arrives*, which is the point: the old driver called this
        situation fatal on iteration 1 and returned the seed. This is also why
        the width seed itself is worth getting right
        (``picasso.fitting.seeds._initial_widths_gauss``) - correctness here is bought with
        an iteration budget no production run would grant.
        """
        spots = _poisson_spots(31, n=40)
        seed = seeds.initial_parameters_gauss(spots, 31).astype(np.float64)
        seed[:, 3] = seed[:, 4] = 8.0
        fitted, _, states, _ = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            spots,
            seed,
            mle=True,
            max_iterations=3000,
            tolerance=1e-8,
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.sum() >= 0.9 * len(spots)
        assert np.median(fitted[converged, 3]) < 2.0

    @pytest.mark.parametrize("box", BOXES)
    @pytest.mark.parametrize("model", MODELS)
    def test_the_production_seed_survives_every_box_size(self, box, model):
        """``size / 5`` is the seed the GUI actually uses, and on a wide box
        it is exactly the too-wide seed above."""
        spots = _poisson_spots(box)
        kwargs = {
            gaussfit.SPHERICAL: dict(spherical=True),
            gaussfit.ELLIPTIC: {},
            gaussfit.ROTATED: dict(rotated=True),
        }[model]
        seed = seeds.initial_parameters_gauss(spots, box, **kwargs).astype(
            np.float64
        )
        _, _, states, _ = gaussfit.fit_spots(
            model, spots, seed, mle=True, max_iterations=200, tolerance=1e-8
        )
        aborted = (states == splinefit.FIT_STATE_NEG_CURVATURE_MLE).sum()
        assert aborted == 0, f"{aborted}/{len(spots)} fits abandoned"

    @pytest.mark.parametrize("box", BOXES)
    def test_the_seed_is_never_echoed_as_the_result(self, box):
        """The downstream signature, asserted directly.

        A fit that aborts on its first step returns its own starting point.
        That is worse than inaccurate, it is silent: the localizations look
        plausible until you notice every width is identical and every
        coordinate is a whole pixel.
        """
        spots = _poisson_spots(box)
        seed = seeds.initial_parameters_gauss(spots, box).astype(np.float64)
        fitted, _, _, iterations = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            spots,
            seed,
            mle=True,
            max_iterations=200,
            tolerance=1e-8,
        )
        untouched = np.all(np.isclose(fitted, seed), axis=1)
        assert not untouched.any()
        assert (iterations > 1).all()
        assert not np.allclose(fitted[:, 1], (box - 1) / 2.0)
        assert np.abs(fitted[:, 1] - np.round(fitted[:, 1])).max() > 0.05

    def test_the_retry_still_costs_an_iteration(self):
        """The retry must stay bounded by ``max_iterations``.

        Turning the abort into a retry would spin forever on a fit that can
        never take a valid step, if a rejected attempt were free.
        """
        spots = _poisson_spots(23, n=20)
        seed = seeds.initial_parameters_gauss(spots, 23).astype(np.float64)
        seed[:, 3] = seed[:, 4] = 8.0
        for budget in (1, 2, 5):
            _, _, _, iterations = gaussfit.fit_spots(
                gaussfit.ELLIPTIC,
                spots,
                seed,
                mle=True,
                max_iterations=budget,
                tolerance=1e-12,
            )
            assert iterations.max() <= budget


class TestNonPositiveModelIsAbandoned:
    """A Gaussian cannot ring negative: the model only goes non-positive when
    the *background* does. Flooring such a pixel (as the spline models do)
    would zero the gradient that pushes the background back up, so the fit
    stalls and the relative convergence test accepts a badly wrong answer.
    This is why ``gaussfit`` uses the strict estimator."""

    def test_negative_background_seed_is_rejected_under_mle(self):
        theta = _truth(gaussfit.ELLIPTIC)
        spots = _reference(gaussfit.ELLIPTIC, theta)[None].astype(np.float32)
        seed = _seed(gaussfit.ELLIPTIC, spots)
        seed[0, 5] = -50.0  # background well below zero
        _, _, states, _ = gaussfit.fit_spots(
            gaussfit.ELLIPTIC, spots, seed, mle=True, max_iterations=50
        )
        assert states[0] == splinefit.FIT_STATE_NEG_CURVATURE_MLE

    def test_every_converged_mle_fit_beats_ground_truth(self, noisy_batch):
        """An MLE that converges must explain the data at least as well as
        the truth does; a floored fit would not."""
        seed = seeds.initial_parameters_gauss(noisy_batch, BOX).astype(
            np.float64
        )
        fitted, chi, states, _ = gaussfit.fit_spots(
            gaussfit.ELLIPTIC,
            noisy_batch,
            seed,
            mle=True,
            max_iterations=200,
            tolerance=1e-8,
        )
        converged = states == splinefit.FIT_STATE_CONVERGED
        assert converged.sum() > 0.8 * len(noisy_batch)
        assert np.all(np.isfinite(chi[converged]))
        assert np.all(fitted[converged, 5] > 0)  # background stayed positive


# ----------------------------------------------------------------------
# Multichannel (joint) spherical Gaussian
# ----------------------------------------------------------------------

MULTI_BOX = 13
MULTI_CENTER = (MULTI_BOX - 1) / 2.0

# Channel 1 carries a real rotation/scale AND a sub-pixel ROI residual, so a
# kernel that dropped either would be caught.
MULTI_JAC = np.array(
    [[1.0, 0.0, 0.0, 1.0], [0.995, 0.03, -0.028, 1.004]], dtype=np.float64
)
MULTI_RES = np.array([[0.0, 0.0], [0.37, -0.22]], dtype=np.float64)


def _multi_spot(box, sx, sy, amp, sigma, bg):
    j, i = np.mgrid[0:box, 0:box].astype(np.float64)
    return amp * np.exp(-0.5 * ((i - sx) ** 2 + (j - sy) ** 2) / sigma**2) + bg


def _multi_batch(n, amps, bgs, seed=1, box=MULTI_BOX):
    """``(spots, truth)`` for a two-channel joint fit.

    Every channel's spot sits where the shared position maps to *through that
    channel's own Jacobian and sub-pixel residual*, which is exactly where the
    kernel evaluates its model."""
    rng = np.random.RandomState(seed)
    n_channels = len(amps)
    spots = np.zeros((n, n_channels, box, box), dtype=np.float32)
    truth = np.zeros((n, 3))
    for k in range(n):
        x = MULTI_CENTER + rng.uniform(-1.5, 1.5)
        y = MULTI_CENTER + rng.uniform(-1.5, 1.5)
        sigma = rng.uniform(1.1, 1.5)
        truth[k] = (x, y, sigma)
        for c in range(n_channels):
            a00, a01, a10, a11 = MULTI_JAC[c]
            dx, dy = x - MULTI_CENTER, y - MULTI_CENTER
            sx = MULTI_CENTER + a00 * dx + a01 * dy + MULTI_RES[c, 0]
            sy = MULTI_CENTER + a10 * dx + a11 * dy + MULTI_RES[c, 1]
            spots[k, c] = rng.poisson(
                _multi_spot(box, sx, sy, amps[c], sigma, bgs[c])
            )
    return spots, truth


def _multi_geometry(n, n_channels=2):
    jac = np.tile(MULTI_JAC[:n_channels], (n, 1, 1))
    res = np.tile(MULTI_RES[:n_channels], (n, 1, 1))
    return jac, res


def _shared_seed(spots):
    n = len(spots)
    seed = np.zeros((n, 5))
    ref = spots[:, 0]
    seed[:, 0] = ref.max(axis=(1, 2)) - ref.min(axis=(1, 2))
    seed[:, 1] = MULTI_CENTER
    seed[:, 2] = MULTI_CENTER
    seed[:, 3] = 1.3
    seed[:, 4] = ref.min(axis=(1, 2))
    return seed


def _decoupled_seed(spots):
    n, n_channels = spots.shape[0], spots.shape[1]
    seed = np.zeros((n, 3 + 2 * n_channels))
    seed[:, 0] = MULTI_CENTER
    seed[:, 1] = MULTI_CENTER
    seed[:, 2] = 1.3
    for c in range(n_channels):
        chan = spots[:, c]
        seed[:, 3 + c] = chan.max(axis=(1, 2)) - chan.min(axis=(1, 2))
        seed[:, 3 + n_channels + c] = chan.min(axis=(1, 2))
    return seed


class TestMultichannelReducesToSingleChannel:
    """The single-channel fit is the identity-Jacobian, zero-residual case of
    the multichannel one, so at one channel the two must agree exactly. This is
    the cheapest possible check on the Jacobian chain rule and its signs."""

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_matches_the_single_channel_spherical_fit(self, mle):
        rng = np.random.RandomState(0)
        n = 30
        spots = np.zeros((n, BOX, BOX), dtype=np.float32)
        for k in range(n):
            spots[k] = rng.poisson(
                _multi_spot(
                    BOX,
                    CENTER + rng.uniform(-1, 1),
                    CENTER + rng.uniform(-1, 1),
                    rng.uniform(200, 500),
                    rng.uniform(1.0, 1.6),
                    rng.uniform(5, 20),
                )
            )
        seed = seeds.initial_parameters_gauss(
            spots, BOX, spherical=True
        ).astype(np.float64)

        single = gaussfit.fit_spots(gaussfit.SPHERICAL, spots, seed, mle=mle)
        multi = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED,
            spots[:, None, :, :],
            np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (n, 1, 1)),
            np.zeros((n, 1, 2)),
            seed,
            mle=mle,
        )

        np.testing.assert_allclose(multi[0], single[0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(multi[1], single[1], rtol=0, atol=1e-8)
        np.testing.assert_array_equal(multi[2], single[2])
        np.testing.assert_array_equal(multi[3], single[3])


class TestMultichannelFits:
    """Joint fits across two registered channels."""

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_shared_model_recovers_the_shared_position(self, mle):
        n = 200
        spots, truth = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
        jac, res = _multi_geometry(n)

        theta, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED,
            spots,
            jac,
            res,
            _shared_seed(spots),
            mle=mle,
        )

        ok = np.isfinite(theta).all(axis=1)
        assert ok.sum() > 0.95 * n
        # unbiased in the reference channel's frame, despite channel 1 sitting
        # at a rotated, sub-pixel-shifted ROI
        assert abs(np.mean(theta[ok, 1] - truth[ok, 0])) < 0.05
        assert abs(np.mean(theta[ok, 2] - truth[ok, 1])) < 0.05
        assert abs(np.mean(theta[ok, 3] - truth[ok, 2])) < 0.05

    def test_dropping_the_sub_pixel_residual_biases_the_position(self):
        """The residual is load-bearing: without it every non-reference
        channel is evaluated at the nearest whole pixel, which pulls the
        shared position off by a systematic fraction of a pixel."""
        n = 200
        spots, truth = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
        jac, res = _multi_geometry(n)
        seed = _shared_seed(spots)

        with_res, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED, spots, jac, res, seed, mle=True
        )
        without, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED,
            spots,
            jac,
            np.zeros_like(res),
            seed,
            mle=True,
        )

        ok = np.isfinite(with_res).all(axis=1) & np.isfinite(without).all(
            axis=1
        )
        bias_with = abs(np.mean(with_res[ok, 1] - truth[ok, 0]))
        bias_without = abs(np.mean(without[ok, 1] - truth[ok, 0]))
        assert bias_with < 0.05
        assert bias_without > 0.1
        assert bias_without > 5 * bias_with

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_decoupled_model_recovers_per_channel_photons(self, mle):
        n = 200
        amps, bgs = [600.0, 220.0], [12.0, 6.0]
        spots, truth = _multi_batch(n, amps, bgs, seed=3)
        jac, res = _multi_geometry(n)

        theta, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_DECOUPLED,
            spots,
            jac,
            res,
            _decoupled_seed(spots),
            mle=mle,
        )

        ok = np.isfinite(theta).all(axis=1)
        assert ok.sum() > 0.95 * n
        assert abs(np.mean(theta[ok, 0] - truth[ok, 0])) < 0.05
        assert abs(np.mean(theta[ok, 1] - truth[ok, 1])) < 0.05
        # the channels are 3x apart in brightness and each is recovered
        assert np.mean(theta[ok, 3]) == pytest.approx(amps[0], rel=0.1)
        assert np.mean(theta[ok, 4]) == pytest.approx(amps[1], rel=0.1)
        assert np.mean(theta[ok, 5]) == pytest.approx(bgs[0], abs=1.5)
        assert np.mean(theta[ok, 6]) == pytest.approx(bgs[1], abs=1.5)

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    def test_mirrored_channel_is_fitted_where_its_spot_is(self, mle):
        """A split field of view flips one half onto the other, so its
        Jacobian has a negative determinant. The channel geometry is a
        linearization of the *displacement* from the box center, not of the
        box coordinate: applied to the coordinate, a mirrored channel is
        evaluated a whole box width away from its own spot and its photon
        count runs off to absurd values while the reference channel still
        looks fine."""
        n = 200
        amps, bgs = [500.0, 500.0], [10.0, 10.0]
        mirrored = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.999, 0.05, 0.049, -0.996]],
            dtype=np.float64,
        )
        rng = np.random.RandomState(11)
        spots = np.zeros((n, 2, MULTI_BOX, MULTI_BOX), dtype=np.float32)
        truth = np.zeros((n, 3))
        for k in range(n):
            x = MULTI_CENTER + rng.uniform(-1.5, 1.5)
            y = MULTI_CENTER + rng.uniform(-1.5, 1.5)
            sigma = rng.uniform(1.1, 1.5)
            truth[k] = (x, y, sigma)
            for c in range(2):
                a00, a01, a10, a11 = mirrored[c]
                dx, dy = x - MULTI_CENTER, y - MULTI_CENTER
                sx = MULTI_CENTER + a00 * dx + a01 * dy + MULTI_RES[c, 0]
                sy = MULTI_CENTER + a10 * dx + a11 * dy + MULTI_RES[c, 1]
                spots[k, c] = rng.poisson(
                    _multi_spot(MULTI_BOX, sx, sy, amps[c], sigma, bgs[c])
                )
        jac = np.tile(mirrored, (n, 1, 1))
        res = np.tile(MULTI_RES[:2], (n, 1, 1))

        theta, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_DECOUPLED,
            spots,
            jac,
            res,
            _decoupled_seed(spots),
            mle=mle,
        )

        ok = np.isfinite(theta).all(axis=1)
        assert ok.sum() > 0.95 * n
        assert abs(np.mean(theta[ok, 0] - truth[ok, 0])) < 0.05
        assert abs(np.mean(theta[ok, 1] - truth[ok, 1])) < 0.05
        # both channels, not just the reference one, keep a sane amplitude
        assert np.mean(theta[ok, 3]) == pytest.approx(amps[0], rel=0.1)
        assert np.mean(theta[ok, 4]) == pytest.approx(amps[1], rel=0.1)
        assert (theta[ok, 4] > 0).all()

    def test_async_matches_the_serial_fit(self):
        n = 40
        spots, _ = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
        jac, res = _multi_geometry(n)
        seed = _shared_seed(spots)

        serial = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED, spots, jac, res, seed, mle=True
        )
        job = gaussfit.fit_spots_multichannel_async(
            gaussfit.MULTI_KIND_SHARED, spots, jac, res, seed, mle=True
        )
        futures.wait(job.futures)
        job.raise_errors()

        np.testing.assert_allclose(job.results()[0], serial[0])


class TestMultichannelApi:
    def test_parameter_counts(self):
        assert (
            gaussfit.n_parameters_multichannel(gaussfit.MULTI_KIND_SHARED, 4)
            == 5
        )
        assert (
            gaussfit.n_parameters_multichannel(
                gaussfit.MULTI_KIND_DECOUPLED, 4
            )
            == 11
        )
        with pytest.raises(ValueError, match="Unknown multichannel"):
            gaussfit.n_parameters_multichannel(99, 2)

    @pytest.mark.parametrize(
        "bad, match",
        [
            ("spots", "channel-major"),
            ("jacobians", "jacobians must have shape"),
            ("residuals", "residuals must have shape"),
            ("initial", "initial_parameters must have shape"),
        ],
    )
    def test_rejects_mismatched_shapes(self, bad, match):
        n = 5
        spots, _ = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
        jac, res = _multi_geometry(n)
        seed = _shared_seed(spots)
        if bad == "spots":
            spots = spots[:, 0]
        elif bad == "jacobians":
            jac = jac[:, :, :3]
        elif bad == "residuals":
            res = res[:, :, :1]
        else:
            seed = seed[:, :4]
        with pytest.raises(ValueError, match=match):
            gaussfit.fit_spots_multichannel(
                gaussfit.MULTI_KIND_SHARED, spots, jac, res, seed
            )

    def test_variance_must_match_the_spots(self):
        n = 5
        spots, _ = _multi_batch(n, [400.0, 400.0], [10.0, 10.0])
        jac, res = _multi_geometry(n)
        with pytest.raises(ValueError, match="same shape as spots"):
            gaussfit.fit_spots_multichannel(
                gaussfit.MULTI_KIND_SHARED,
                spots,
                jac,
                res,
                _shared_seed(spots),
                variance=np.zeros((n, 2, 3, 3), dtype=np.float32),
            )


class TestMultichannelCrlb:
    """The reported precision must describe the fit that actually ran, so the
    CRLB kernels see the same channel Jacobians and sub-pixel residuals the fit
    kernels do. Checked against the empirical scatter of repeated noisy fits -
    an efficient MLE attains its Cramer-Rao bound, so the two must agree."""

    N_TRIALS = 4000
    TRUTH_X, TRUTH_Y, TRUTH_SIGMA = 6.3, 5.8, 1.25

    def _repeats(self, amps, bgs, seed):
        """``N_TRIALS`` independent noisy realizations of one fixed spot."""
        rng = np.random.RandomState(seed)
        n_channels = len(amps)
        spots = np.zeros(
            (self.N_TRIALS, n_channels, MULTI_BOX, MULTI_BOX),
            dtype=np.float32,
        )
        for c in range(n_channels):
            a00, a01, a10, a11 = MULTI_JAC[c]
            dx = self.TRUTH_X - MULTI_CENTER
            dy = self.TRUTH_Y - MULTI_CENTER
            sx = MULTI_CENTER + a00 * dx + a01 * dy + MULTI_RES[c, 0]
            sy = MULTI_CENTER + a10 * dx + a11 * dy + MULTI_RES[c, 1]
            clean = _multi_spot(
                MULTI_BOX, sx, sy, amps[c], self.TRUTH_SIGMA, bgs[c]
            )
            spots[:, c] = rng.poisson(
                np.broadcast_to(clean, (self.N_TRIALS, MULTI_BOX, MULTI_BOX))
            )
        return spots

    def test_shared_model_bound_matches_the_empirical_scatter(self):
        spots = self._repeats([300.0, 300.0], [8.0, 8.0], seed=5)
        jac, res = _multi_geometry(self.N_TRIALS)
        seed = _shared_seed(spots)
        seed[:, 1] = self.TRUTH_X
        seed[:, 2] = self.TRUTH_Y
        seed[:, 3] = self.TRUTH_SIGMA

        theta, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_SHARED,
            spots,
            jac,
            res,
            seed,
            mle=True,
            tolerance=1e-6,
            max_iterations=200,
        )
        ok = np.isfinite(theta).all(axis=1)
        # the CRLB is parametrized in photons, not peak height, and takes the
        # shared parameters first: [x, y, sigma, N, bg]
        photons = theta[:, 0] * 2 * np.pi * theta[:, 3] ** 2
        theta_n = np.column_stack(
            [theta[:, 1], theta[:, 2], theta[:, 3], photons, theta[:, 4]]
        )
        crlb = precision._gauss_crlb_multichannel(
            theta_n, MULTI_BOX, jac, res, mle=True, link_photons=True
        )

        # the variances come back in the order they were given
        for column, empirical in (
            (0, theta[ok, 1].std()),
            (1, theta[ok, 2].std()),
            (2, theta[ok, 3].std()),
            (3, photons[ok].std()),
            (4, theta[ok, 4].std()),
        ):
            predicted = np.sqrt(np.nanmedian(crlb[ok, column]))
            assert predicted == pytest.approx(empirical, rel=0.1)

    def test_decoupled_model_bound_matches_the_empirical_scatter(self):
        amps, bgs = [500.0, 180.0], [10.0, 5.0]
        spots = self._repeats(amps, bgs, seed=9)
        jac, res = _multi_geometry(self.N_TRIALS)
        seed = _decoupled_seed(spots)
        seed[:, 0] = self.TRUTH_X
        seed[:, 1] = self.TRUTH_Y
        seed[:, 2] = self.TRUTH_SIGMA

        theta, _, _, _ = gaussfit.fit_spots_multichannel(
            gaussfit.MULTI_KIND_DECOUPLED,
            spots,
            jac,
            res,
            seed,
            mle=True,
            tolerance=1e-6,
            max_iterations=200,
        )
        ok = np.isfinite(theta).all(axis=1)
        n_channels = len(amps)
        theta_n = theta.copy()
        for c in range(n_channels):
            theta_n[:, 3 + c] = theta[:, 3 + c] * 2 * np.pi * theta[:, 2] ** 2
        crlb = precision._gauss_crlb_multichannel(
            theta_n, MULTI_BOX, jac, res, mle=True, link_photons=False
        )

        for column in (0, 1, 2):  # shared x, y, sigma
            predicted = np.sqrt(np.nanmedian(crlb[ok, column]))
            assert predicted == pytest.approx(theta[ok, column].std(), rel=0.1)
        for c in range(n_channels):
            predicted = np.sqrt(np.nanmedian(crlb[ok, 3 + c]))
            assert predicted == pytest.approx(
                theta_n[ok, 3 + c].std(), rel=0.1
            )
            predicted_bg = np.sqrt(np.nanmedian(crlb[ok, 3 + n_channels + c]))
            assert predicted_bg == pytest.approx(
                theta[ok, 3 + n_channels + c].std(), rel=0.1
            )

    @staticmethod
    def _crlb_theta(n):
        """A plausible ``[x, y, sigma, N, bg]`` row per localization."""
        theta = np.zeros((n, 5))
        theta[:, 0] = MULTI_CENTER
        theta[:, 1] = MULTI_CENTER
        theta[:, 2] = 1.3
        theta[:, 3] = 400.0 * 2 * np.pi * 1.3**2
        theta[:, 4] = 10.0
        return theta

    def test_non_converged_rows_are_nan(self):
        n = 6
        jac, res = _multi_geometry(n)
        theta = self._crlb_theta(n)
        theta[2] = np.nan

        crlb = precision._gauss_crlb_multichannel(
            theta, MULTI_BOX, jac, res, mle=True, link_photons=True
        )

        assert np.all(np.isnan(crlb[2]))
        assert np.all(np.isfinite(crlb[0]))

    def test_em_gain_doubles_the_variance(self):
        n = 20
        jac, res = _multi_geometry(n)
        theta = self._crlb_theta(n)

        plain = precision._gauss_crlb_multichannel(
            theta, MULTI_BOX, jac, res, mle=True, em=False
        )
        em = precision._gauss_crlb_multichannel(
            theta, MULTI_BOX, jac, res, mle=True, em=True
        )

        np.testing.assert_allclose(em, 2.0 * plain, rtol=1e-9)
