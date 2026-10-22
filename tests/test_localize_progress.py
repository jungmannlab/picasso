"""Test the progress reporting of ``picasso.localize`` (and of the
``picasso.avgroi`` / ``picasso.fitting.precision`` helpers it drives):
every public function takes one ``progress`` argument, reports its steps
as labeled phases that end at their maximum, prints nothing with
``progress=None``, and works with ``"console"``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from picasso import avgroi, lib, localize, transforms
from picasso.fitting import precision

from tests._progress_tracking import (
    CancelingProgress,
    RecordingProgress,
    assert_phases_complete,
)
from tests.conftest import CALIB_3D

BOX = 7
CAMERA = {"Baseline": 100, "Sensitivity": 1.0, "Gain": 1, "Pixelsize": 130}
NET_GRADIENT = localize.IdentificationParameters(
    method=localize.NetGradient(2000.0), box=BOX
)
FIT_LABELS = ["Cutting spots", "Fitting spots"]
SPLINE_LABELS = FIT_LABELS + ["Computing spline CRLB"]
MULTI_GAUSS_LABELS = FIT_LABELS + ["Computing Gaussian CRLB"]
MULTI_SPLINE_LABELS = SPLINE_LABELS


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _gauss_frames(
    n_frames=6,
    shape=(32, 32),
    positions=((10.3, 12.6), (21.4, 19.2)),
    amplitude=900.0,
    sigma=1.3,
    seed=0,
):
    """A Poisson movie with Gaussian spots at ``positions`` ((x, y)) in
    every frame, on a background of 10 photons and a baseline of 100, and
    the matching identifications."""
    rng = np.random.default_rng(seed)
    j, i = np.mgrid[0 : shape[0], 0 : shape[1]]
    clean = np.full(shape, 10.0)
    for x, y in positions:
        clean = clean + amplitude * np.exp(
            -0.5 * ((i - x) ** 2 + (j - y) ** 2) / sigma**2
        )
    movie = np.stack(
        [rng.poisson(clean) + 100 for _ in range(n_frames)]
    ).astype(np.uint16)
    ids = pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n_frames), len(positions)),
            "x": np.tile([int(round(x)) for x, _ in positions], n_frames),
            "y": np.tile([int(round(y)) for _, y in positions], n_frames),
            "net_gradient": np.full(n_frames * len(positions), 1e4),
        }
    )
    return movie, ids


def _info(movie):
    return [
        {
            "Byte Order": "<",
            "Data Type": "uint16",
            "Frames": len(movie),
            "Height": movie.shape[1],
            "Width": movie.shape[2],
        }
    ]


def _spline_calibration(model="spline-2d", n_channels=2):
    """A separable Gaussian spline calibration on the fit box."""
    from tests.test_localize import _gauss_spline_calibration

    calib, _ = _gauss_spline_calibration(
        model=model, box=BOX, n_channels=n_channels
    )
    return calib


def _shift(dx, dy=0.0):
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return transforms.AffineTransform(matrix=matrix).to_dict()


IDENTITY = transforms.identity().to_dict()


def _two_channels(n_frames=4):
    """Two registered channel movies (the second shifted by 3 px in x)
    and the reference identifications."""
    a, ids = _gauss_frames(n_frames=n_frames, seed=1)
    b, _ = _gauss_frames(
        n_frames=n_frames,
        positions=((13.3, 12.6), (24.4, 19.2)),
        seed=2,
    )
    return [a, b], [CAMERA, CAMERA], ids


def _split_fov_movie(n_frames=4):
    """One 32 x 64 movie whose halves are two channels, the right half
    being the left one moved by 32 px, and the reference identifications."""
    left, ids = _gauss_frames(n_frames=n_frames, seed=3)
    right, _ = _gauss_frames(n_frames=n_frames, seed=4)
    return np.concatenate([left, right], axis=2), ids


REGIONS = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]


def _split_fov_geometry(calibration=None):
    calibration = dict(calibration or {"model": "channel-registration"})
    calibration.update(
        split_fov=True,
        regions=REGIONS,
        reference=0,
        channel_transforms=[IDENTITY, _shift(32.0)],
    )
    return calibration


def _silent(capsys, function, *args, **kwargs):
    """Run ``function`` with ``progress=None`` and assert it printed
    nothing; return its result."""
    capsys.readouterr()
    result = function(*args, progress=None, **kwargs)
    out, err = capsys.readouterr()
    assert out == "" and err == "", (out, err)
    return result


def _console(capsys, label, function, *args, **kwargs):
    """Run ``function`` with ``progress="console"`` and assert that a bar
    labeled ``label`` was shown; return its result."""
    capsys.readouterr()
    result = function(*args, progress="console", **kwargs)
    assert label in capsys.readouterr().err
    return result


class ThreadRecordingProgress(RecordingProgress):
    """Also record the thread every call came from."""

    def __init__(self):
        super().__init__()
        self.threads = set()

    def set_value(self, value, *args, **kwargs):
        self.threads.add(threading.get_ident())
        super().set_value(value)


class CancelInPhase(RecordingProgress):
    """Raise ``lib.OperationCanceled`` from the first ``set_value`` of the
    phase labeled ``label``."""

    def __init__(self, label):
        super().__init__()
        self.label = label

    def set_value(self, value, *args, **kwargs):
        super().set_value(value)
        if self.phases[-1]["label"] == self.label:
            raise lib.OperationCanceled()


@pytest.fixture
def factory(picasso_movie_factory):
    return picasso_movie_factory


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


class TestIdentify:
    @pytest.mark.parametrize("threaded", [True, False])
    def test_one_phase_over_the_frames(self, threaded):
        movie, _ = _gauss_frames()
        tracker = RecordingProgress()
        ids, _ = localize.identify(
            movie, parameters=NET_GRADIENT, threaded=threaded, progress=tracker
        )
        assert len(ids)
        assert_phases_complete(tracker, ["Identifying spots"])
        # the threaded path used to report claimed frames and no final
        # value; now it ends at the frame count
        assert tracker.phases[0]["maximum"] == len(movie)
        assert tracker.phases[0]["unit"] == "frame"

    @pytest.mark.parametrize("threaded", [True, False])
    def test_silent_and_console(self, capsys, threaded):
        movie, _ = _gauss_frames()
        kwargs = dict(parameters=NET_GRADIENT, threaded=threaded)
        _silent(capsys, localize.identify, movie, **kwargs)
        _console(
            capsys, "Identifying spots", localize.identify, movie, **kwargs
        )

    def test_cancel_propagates(self):
        movie, _ = _gauss_frames(n_frames=40)
        with pytest.raises(lib.OperationCanceled):
            localize.identify(
                movie,
                parameters=NET_GRADIENT,
                threaded=False,
                progress=CancelingProgress(after=2),
            )

    def test_async_reports_finished_frames(self):
        movie, _ = _gauss_frames()
        current, fs = localize.identify_async(
            movie, localize.NetGradient(2000.0), BOX
        )
        localize.identifications_from_futures(fs)
        assert current == [len(movie), len(movie)]

    def test_multichannel_sum(self):
        movies, cameras, _ = _two_channels()
        tracker = RecordingProgress()
        localize.identify_multichannel_sum(
            movies,
            [IDENTITY, _shift(3.0)],
            parameters=NET_GRADIENT,
            camera_infos=cameras,
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Identifying spots"])


# ---------------------------------------------------------------------------
# Cutting spots
# ---------------------------------------------------------------------------


class TestGetSpots:
    @pytest.mark.parametrize("wrap", [False, True], ids=["array", "movie"])
    def test_one_phase_over_the_spots(self, factory, wrap):
        movie, ids = _gauss_frames()
        if wrap:
            movie = factory(movie, _info(movie))
        tracker = RecordingProgress()
        spots = localize.get_spots(movie, ids, BOX, CAMERA, progress=tracker)
        assert len(spots) == len(ids)
        assert_phases_complete(tracker, ["Cutting spots"])
        assert tracker.phases[0]["maximum"] == len(ids)

    def test_concurrent_reads_report_from_the_calling_thread(self, factory):
        # regression: the I/O worker threads used to call the progress
        # callback themselves, which a Qt tracker must never see
        class ConcurrentMovie(factory):
            supports_concurrent_reads = True

        frames, ids = _gauss_frames(n_frames=12)
        movie = ConcurrentMovie(frames, _info(frames))
        tracker = ThreadRecordingProgress()
        spots = localize.get_spots(movie, ids, BOX, CAMERA, progress=tracker)
        expected = localize.get_spots(frames, ids, BOX, CAMERA)
        np.testing.assert_array_equal(spots, expected)
        assert_phases_complete(tracker, ["Cutting spots"])
        assert tracker.threads == {threading.get_ident()}

    def test_concurrent_reads_cancel(self, factory):
        class ConcurrentMovie(factory):
            supports_concurrent_reads = True

        frames, ids = _gauss_frames(n_frames=12)
        movie = ConcurrentMovie(frames, _info(frames))
        with pytest.raises(lib.OperationCanceled):
            localize.get_spots(
                movie, ids, BOX, CAMERA, progress=CancelingProgress(after=1)
            )

    def test_silent_and_console(self, capsys):
        movie, ids = _gauss_frames()
        _silent(capsys, localize.get_spots, movie, ids, BOX, CAMERA)
        _console(
            capsys,
            "Cutting spots",
            localize.get_spots,
            movie,
            ids,
            BOX,
            CAMERA,
        )

    def test_multichannel_one_phase_over_all_channels(self):
        movies, cameras, ids = _two_channels()
        tracker = RecordingProgress()
        localize.get_spots_multichannel(
            movies,
            ids,
            BOX,
            cameras,
            [IDENTITY, _shift(3.0)],
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Cutting spots"])
        assert tracker.phases[0]["maximum"] == 2 * len(ids)


# ---------------------------------------------------------------------------
# Single-channel fits
# ---------------------------------------------------------------------------


class TestFit:
    def _fit(self, factory, method, progress, multiprocess=False, **kwargs):
        frames, ids = _gauss_frames()
        movie = factory(frames, _info(frames))
        locs, _ = localize.fit(
            movie,
            camera_info=CAMERA,
            identifications=ids,
            box=BOX,
            fitting_method=method,
            multiprocess=multiprocess,
            progress=progress,
            **kwargs,
        )
        return locs, ids

    @pytest.mark.parametrize(
        "method", ["gausslq", "gaussmle", "gausslq-spherical", "avg"]
    )
    @pytest.mark.parametrize("multiprocess", [False, True])
    def test_gauss_and_avg_phases(self, factory, method, multiprocess):
        if method == "avg" and multiprocess:
            pytest.skip("covered by test_avg_multiprocess (process pool)")
        tracker = RecordingProgress()
        locs, ids = self._fit(factory, method, tracker, multiprocess)
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, FIT_LABELS)
        assert [p["maximum"] for p in tracker.phases] == [len(ids)] * 2

    @pytest.mark.slow
    def test_avg_multiprocess(self, factory):
        # regression: the process pool reported round(N * tasks / n_tasks)
        # with no final value
        tracker = RecordingProgress()
        locs, ids = self._fit(factory, "avg", tracker, multiprocess=True)
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, FIT_LABELS)

    @pytest.mark.parametrize("method", ["spline", "spline-mle"])
    @pytest.mark.parametrize("multiprocess", [False, True])
    def test_spline_phases(self, factory, method, multiprocess):
        # regression: the threaded CPU fit reported claimed spots and the
        # CRLB was not reported at all
        tracker = RecordingProgress()
        locs, ids = self._fit(
            factory,
            method,
            tracker,
            multiprocess,
            spline_calibration=_spline_calibration(),
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, SPLINE_LABELS)
        assert [p["maximum"] for p in tracker.phases] == [len(ids)] * 3

    @pytest.mark.skipif(not localize.CUDA_AVAILABLE, reason="needs CUDA")
    @pytest.mark.parametrize(
        "method, labels",
        [("gausslq-gpu", FIT_LABELS), ("spline-gpu", SPLINE_LABELS)],
    )
    def test_gpu_phases(self, factory, method, labels):
        # regression: the GPU fits reported a sentinel 1 instead of counts
        tracker = RecordingProgress()
        self._fit(
            factory,
            method,
            tracker,
            spline_calibration=_spline_calibration(),
        )
        assert_phases_complete(tracker, labels)

    @pytest.mark.parametrize("method", ["gausslq", "spline"])
    def test_silent_and_console(self, capsys, factory, method):
        kwargs = dict(spline_calibration=_spline_calibration())
        capsys.readouterr()
        self._fit(factory, method, None, multiprocess=True, **kwargs)
        assert capsys.readouterr() == ("", "")
        self._fit(factory, method, "console", multiprocess=True, **kwargs)
        err = capsys.readouterr().err
        assert "Cutting spots" in err and "Fitting spots" in err

    def test_cancel_stops_the_threaded_spline_fit(self, factory):
        tracker = CancelInPhase("Fitting spots")
        with pytest.raises(lib.OperationCanceled):
            self._fit(
                factory,
                "spline",
                tracker,
                multiprocess=True,
                spline_calibration=_spline_calibration(),
            )

    def test_abort_still_returns_none(self, factory):
        locs, _ = self._fit(
            factory,
            "spline",
            RecordingProgress(),
            multiprocess=True,
            spline_calibration=_spline_calibration(),
            abort_callback=lambda: True,
        )
        assert locs is None


class TestFitSpots:
    def _spots(self):
        movie, ids = _gauss_frames()
        return localize.get_spots(movie, ids, BOX, CAMERA), ids

    @pytest.mark.parametrize("mle", [False, True])
    def test_gauss(self, capsys, mle):
        spots, _ = self._spots()
        tracker = RecordingProgress()
        localize.fit_spots_gauss(spots, mle=mle, progress=tracker)
        assert_phases_complete(tracker, ["Fitting spots"])
        _silent(capsys, localize.fit_spots_gauss, spots, mle=mle)
        _console(capsys, "Fitting spots", localize.fit_spots_gauss, spots)

    @pytest.mark.parametrize("multiprocess", [False, True])
    def test_splinefit(self, capsys, multiprocess):
        spots, _ = self._spots()
        calib = _spline_calibration()
        tracker = RecordingProgress()
        localize.fit_spots_splinefit(
            spots, calib, multiprocess=multiprocess, progress=tracker
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == len(spots)
        _silent(
            capsys,
            localize.fit_spots_splinefit,
            spots,
            calib,
            multiprocess=multiprocess,
        )
        _console(
            capsys,
            "Fitting spots",
            localize.fit_spots_splinefit,
            spots,
            calib,
            multiprocess=multiprocess,
        )

    def test_spline_and_its_crlb(self, capsys):
        spots, ids = self._spots()
        calib = _spline_calibration()
        tracker = RecordingProgress()
        theta = localize.fit_spots_spline(
            spots, calib, use_gpu=False, progress=tracker
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        tracker = RecordingProgress()
        locs = localize.locs_from_fits_spline(
            ids, theta, BOX, False, calib, pixelsize=130, progress=tracker
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, ["Computing spline CRLB"])
        assert tracker.phases[0]["unit"] == "loc"
        kwargs = dict(pixelsize=130)
        args = (ids, theta, BOX, False, calib)
        _silent(capsys, localize.locs_from_fits_spline, *args, **kwargs)
        _console(
            capsys,
            "Computing spline CRLB",
            localize.locs_from_fits_spline,
            *args,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Joint multichannel and split-FOV fits
# ---------------------------------------------------------------------------


class TestGaussMultichannel:
    REGISTRATION = {
        "model": "channel-registration",
        "n_channels": 2,
        "channel_transforms": [IDENTITY, _shift(3.0)],
    }

    @pytest.mark.parametrize("multiprocess", [False, True])
    def test_fit_spots_reports_every_path(self, capsys, multiprocess):
        movies, cameras, ids = _two_channels()
        spots, residuals, jacobians = localize.get_spots_multichannel(
            movies,
            ids,
            BOX,
            cameras,
            self.REGISTRATION["channel_transforms"],
            return_residuals=True,
            return_jacobians=True,
        )
        args = (spots, residuals, jacobians)
        kwargs = dict(use_gpu=False, multiprocess=multiprocess)
        tracker = RecordingProgress()
        localize.fit_spots_gauss_multichannel(
            *args, progress=tracker, **kwargs
        )
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["maximum"] == len(ids)
        _silent(capsys, localize.fit_spots_gauss_multichannel, *args, **kwargs)
        # regression: the default threaded CPU path showed no console bar
        _console(
            capsys,
            "Fitting spots",
            localize.fit_spots_gauss_multichannel,
            *args,
            **kwargs,
        )

    @pytest.mark.parametrize("link_photons", [False, True])
    def test_fit_gauss_multichannel_phases(self, capsys, link_photons):
        # regression: the cutting and the fit both restarted at 0 on one
        # callback, and the CRLB was unreported
        movies, cameras, ids = _two_channels()
        args = (movies, cameras, ids, BOX, self.REGISTRATION)
        kwargs = dict(link_photons=link_photons, use_gpu=False)
        tracker = RecordingProgress()
        locs, _ = localize.fit_gauss_multichannel(
            *args, progress=tracker, **kwargs
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, MULTI_GAUSS_LABELS)
        assert [p["maximum"] for p in tracker.phases] == [
            2 * len(ids),
            len(ids),
            len(ids),
        ]
        _silent(capsys, localize.fit_gauss_multichannel, *args, **kwargs)
        _console(
            capsys,
            "Computing Gaussian CRLB",
            localize.fit_gauss_multichannel,
            *args,
            **kwargs,
        )

    def test_fit_gauss_split_fov_phases(self, capsys):
        movie, ids = _split_fov_movie()
        args = (movie, CAMERA, ids, BOX, _split_fov_geometry())
        tracker = RecordingProgress()
        locs, _ = localize.fit_gauss_split_fov(
            *args, use_gpu=False, progress=tracker
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, MULTI_GAUSS_LABELS)
        _silent(capsys, localize.fit_gauss_split_fov, *args, use_gpu=False)

    def test_locs_from_fits_crlb_phase(self):
        movies, cameras, ids = _two_channels()
        spots, residuals, jacobians = localize.get_spots_multichannel(
            movies,
            ids,
            BOX,
            cameras,
            self.REGISTRATION["channel_transforms"],
            return_residuals=True,
            return_jacobians=True,
        )
        theta = localize.fit_spots_gauss_multichannel(
            spots, residuals, jacobians, use_gpu=False
        )
        tracker = RecordingProgress()
        localize.locs_from_fits_gauss_multichannel(
            ids,
            theta,
            BOX,
            False,
            jacobians,
            residuals,
            pixelsize=130,
            progress=tracker,
        )
        assert_phases_complete(tracker, ["Computing Gaussian CRLB"])
        assert tracker.phases[0]["unit"] == "loc"


class TestSplineMultichannel:
    def _calibration(self, **extra):
        calib = _spline_calibration(model="spline-3d-multichannel")
        calib["channel_transforms"] = [IDENTITY, _shift(3.0)]
        calib.update(extra)
        return calib

    @pytest.mark.parametrize("link_photons", [False, True])
    def test_fit_spline_multichannel_phases(self, capsys, link_photons):
        movies, cameras, ids = _two_channels()
        args = (movies, cameras, ids, BOX, self._calibration())
        kwargs = dict(link_photons=link_photons, use_gpu=False)
        tracker = RecordingProgress()
        locs, _ = localize.fit_spline_multichannel(
            *args, progress=tracker, **kwargs
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, MULTI_SPLINE_LABELS)
        _silent(capsys, localize.fit_spline_multichannel, *args, **kwargs)

    def test_ratiometric_reports_every_hypothesis(self, capsys):
        # regression: only the cutting was reported; the hypothesis fits
        # and the CRLB ran silently
        movies, cameras, ids = _two_channels()
        calib = self._calibration(photon_ratios=[[0.5, 0.5], [0.8, 0.2]])
        args = (movies, cameras, ids, BOX, calib)
        tracker = RecordingProgress()
        locs, _ = localize.fit_spline_multichannel_ratiometric(
            *args, use_gpu=False, progress=tracker
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker)
        labels = tracker.labels
        assert labels[:3] == [
            "Cutting spots",
            "Hypothesis 1/2: fitting spots",
            "Hypothesis 2/2: fitting spots",
        ]
        crlb = labels[3:]
        assert crlb and set(crlb) <= {
            "Hypothesis 1/2: computing spline CRLB",
            "Hypothesis 2/2: computing spline CRLB",
        }
        # every localization is in exactly one winning group
        assert sum(p["maximum"] for p in tracker.phases[3:]) == len(ids)
        _silent(
            capsys,
            localize.fit_spline_multichannel_ratiometric,
            *args,
            use_gpu=False,
        )

    def test_fit_spline_split_fov_phases(self):
        movie, ids = _split_fov_movie()
        calib = _split_fov_geometry(
            _spline_calibration(model="spline-3d-multichannel")
        )
        tracker = RecordingProgress()
        locs, _ = localize.fit_spline_split_fov(
            movie, CAMERA, ids, BOX, calib, use_gpu=False, progress=tracker
        )
        assert len(locs) == len(ids)
        assert_phases_complete(tracker, MULTI_SPLINE_LABELS)


class TestLinking:
    def test_filter_linked_identifications(self, capsys):
        movies, cameras, ids = _two_channels()
        _, ids_b = _gauss_frames(positions=((13.3, 12.6), (24.4, 19.2)))
        args = ([ids, ids_b], [IDENTITY, _shift(3.0)], BOX)
        tracker = RecordingProgress()
        linked, n_kept, n_total = localize.filter_linked_identifications(
            *args, progress=tracker
        )
        assert n_kept == n_total == len(ids)
        assert_phases_complete(tracker, ["Linking channels"])
        _silent(capsys, localize.filter_linked_identifications, *args)

    def test_split_fov(self):
        movie, ids = _split_fov_movie()
        ids_all = pd.concat(
            [ids, ids.assign(x=ids["x"] + 32)], ignore_index=True
        )
        tracker = RecordingProgress()
        _, n_kept, n_total = localize.filter_linked_identifications_split_fov(
            ids_all, _split_fov_geometry(), BOX, progress=tracker
        )
        assert n_kept == n_total == len(ids)
        assert_phases_complete(tracker, ["Linking channels"])


# ---------------------------------------------------------------------------
# Independent fits
# ---------------------------------------------------------------------------


class TestIndependent:
    def test_fit_independent_labels_the_channels(self, capsys, factory):
        frames, _, ids = _two_channels()
        movies = [factory(f, _info(f)) for f in frames]
        args = (movies, CAMERA, [ids, ids], BOX)
        tracker = RecordingProgress()
        results = localize.fit_independent(
            *args, multiprocess=False, progress=tracker
        )
        assert len(results) == 2
        assert [info[-1]["Channel"] for _, info in results] == [0, 1]
        assert_phases_complete(
            tracker,
            [
                "Channel 1/2: cutting spots",
                "Channel 1/2: fitting spots",
                "Channel 2/2: cutting spots",
                "Channel 2/2: fitting spots",
            ],
        )
        _silent(capsys, localize.fit_independent, *args, multiprocess=False)
        _console(
            capsys,
            "Channel 2/2: fitting spots",
            localize.fit_independent,
            *args,
            multiprocess=False,
        )

    def test_fit_split_fov_independent_labels_the_regions(self, factory):
        frames, ids = _split_fov_movie()
        movie = factory(frames, _info(frames))
        ids_all = pd.concat(
            [ids, ids.assign(x=ids["x"] + 32)], ignore_index=True
        )
        tracker = RecordingProgress()
        results = localize.fit_split_fov_independent(
            movie,
            CAMERA,
            ids_all,
            BOX,
            REGIONS,
            multiprocess=False,
            progress=tracker,
        )
        assert [len(locs) for locs, _ in results] == [len(ids)] * 2
        assert_phases_complete(
            tracker,
            [
                "Region 1/2: cutting spots",
                "Region 1/2: fitting spots",
                "Region 2/2: cutting spots",
                "Region 2/2: fitting spots",
            ],
        )


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


class TestLocalize:
    def _movie(self, factory):
        frames, _ = _gauss_frames()
        return factory(frames, _info(frames))

    @pytest.mark.parametrize("threaded", [False, True])
    def test_phases(self, factory, threaded):
        tracker = RecordingProgress()
        locs, _ = localize.localize(
            self._movie(factory),
            camera_info=CAMERA,
            identification_parameters=NET_GRADIENT,
            threaded=threaded,
            progress=tracker,
        )
        assert len(locs)
        assert_phases_complete(tracker, ["Identifying spots"] + FIT_LABELS)

    def test_astigmatic_z_adds_the_z_phases(self, factory):
        tracker = RecordingProgress()
        locs, _ = localize.localize(
            self._movie(factory),
            camera_info=CAMERA,
            identification_parameters=NET_GRADIENT,
            calibration_3d=dict(CALIB_3D),
            movie_info=_info(_gauss_frames()[0]),
            threaded=False,
            progress=tracker,
        )
        assert "z" in locs
        assert tracker.labels[:3] == ["Identifying spots"] + FIT_LABELS
        assert len(tracker.labels) > 3  # zfit's own phase(s)
        assert_phases_complete(tracker)

    def test_silent_and_console(self, capsys, factory):
        kwargs = dict(
            camera_info=CAMERA, identification_parameters=NET_GRADIENT
        )
        _silent(capsys, localize.localize, self._movie(factory), **kwargs)
        capsys.readouterr()
        localize.localize(self._movie(factory), progress="console", **kwargs)
        err = capsys.readouterr().err
        for label in ["Identifying spots"] + FIT_LABELS:
            assert label in err

    def test_localize_frames(self, capsys):
        frames, _ = _gauss_frames()
        tracker = RecordingProgress()
        locs = localize.localize_frames(
            frames,
            _info(frames),
            NET_GRADIENT,
            camera_info=CAMERA,
            progress=tracker,
        )
        assert len(locs)
        assert_phases_complete(tracker, ["Identifying spots"] + FIT_LABELS)
        _silent(
            capsys,
            localize.localize_frames,
            frames,
            _info(frames),
            NET_GRADIENT,
            camera_info=CAMERA,
        )


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


class TestChecks:
    def test_check_nena_console(self, capsys, locs, info):
        # regression: "console" raised a TypeError inside, which was caught
        # and turned into a NaN result
        nena = localize.check_nena(locs, info, progress="console")
        assert np.isfinite(nena) and nena > 0
        _silent(capsys, localize.check_nena, locs, info)

    def test_check_nena_tracker(self, locs, info):
        tracker = RecordingProgress()
        assert np.isfinite(localize.check_nena(locs, info, progress=tracker))
        assert_phases_complete(tracker)

    def test_check_drift(self, capsys, locs, info):
        # regression: "console" raised a TypeError, and None printed the
        # segmentation and correlation bars
        drift = _silent(capsys, localize.check_drift, locs, info)
        assert np.isfinite(drift).all()
        localize.check_drift(locs, info, progress="console")
        tracker = RecordingProgress()
        localize.check_drift(locs, info, progress=tracker)
        assert_phases_complete(tracker)

    def test_get_file_summary_is_silent(self, capsys):
        path = "./tests/data/testdata_locs.hdf5"
        summary = _silent(capsys, localize.get_file_summary, path, path)
        assert np.isfinite(summary["nena_nm"])


# ---------------------------------------------------------------------------
# avgroi and the CRLB helpers
# ---------------------------------------------------------------------------


class TestAvgroi:
    def test_fit_spots_counts(self, capsys):
        spots = np.ones((25, BOX, BOX), dtype=np.float32)
        tracker = RecordingProgress()
        avgroi.fit_spots(spots, progress=tracker)
        # regression: it reported 0-based indices, never the spot count
        assert_phases_complete(tracker, ["Fitting spots"])
        assert tracker.phases[0]["values"][-1] == len(spots)
        _silent(capsys, avgroi.fit_spots, spots)
        _console(capsys, "Fitting spots", avgroi.fit_spots, spots)

    @pytest.mark.slow
    def test_fit_spots_parallel_is_silent(self, capsys):
        # regression: the synchronous pool printed a tqdm bar regardless
        spots = np.ones((50, BOX, BOX), dtype=np.float32)
        theta = _silent(capsys, avgroi.fit_spots_parallel, spots)
        assert theta.shape == (50, 6)
        tracker = RecordingProgress()
        avgroi.fit_spots_parallel(spots, progress=tracker)
        assert_phases_complete(tracker, ["Fitting spots"])


class TestPrecisionHelpers:
    def _theta(self, n=40):
        rng = np.random.default_rng(0)
        theta = np.zeros((n, 4))
        theta[:, 0] = 3000.0
        theta[:, 1:3] = rng.uniform(-0.5, 0.5, (n, 2))
        theta[:, 3] = 15.0
        return theta

    def test_spline_crlb_empty(self):
        tracker = RecordingProgress()
        crlb = precision._spline_crlb(
            np.zeros((0, 4)), _spline_calibration(), BOX, progress=tracker
        )
        assert crlb.shape[0] == 0
        assert_phases_complete(tracker, ["Computing spline CRLB"])

    def test_gpu_failure_restarts_on_the_cpu_in_a_new_phase(self, monkeypatch):
        monkeypatch.setattr(precision, "CUDA_AVAILABLE", True)
        monkeypatch.setattr(precision, "_crlb_gpu_fallback_warned", True)

        def boom(*args, progress=None, **kwargs):
            progress.set_value(10)  # part of the way, then the device dies
            raise RuntimeError("device on fire")

        monkeypatch.setattr(precision, "_spline_crlb_cuda", boom)
        theta = self._theta()
        tracker = RecordingProgress()
        precision._spline_crlb(
            theta, _spline_calibration(), BOX, progress=tracker
        )
        assert tracker.labels == [
            "Computing spline CRLB",
            "Computing spline CRLB on the CPU",
        ]
        cpu = tracker.phases[1]
        assert cpu["values"][-1] == cpu["maximum"] == len(theta)
        assert cpu["values"] == sorted(cpu["values"])

    def test_cancel_is_not_mistaken_for_a_device_failure(self, monkeypatch):
        monkeypatch.setattr(precision, "CUDA_AVAILABLE", True)

        def cancel(*args, **kwargs):
            raise lib.OperationCanceled()

        monkeypatch.setattr(precision, "_spline_crlb_cuda", cancel)
        with pytest.raises(lib.OperationCanceled):
            precision._spline_crlb(self._theta(), _spline_calibration(), BOX)


def test_old_arguments_are_gone():
    import inspect

    old = {
        "progress_callback",
        "callback",
        "identification_progress_callback",
        "fit_progress_callback",
        "fit_z_progress_callback",
        "cut_progress_callback",
    }
    for module in (localize, avgroi):
        for name in dir(module):
            fn = getattr(module, name)
            if inspect.isfunction(fn) and fn.__module__ == module.__name__:
                params = set(inspect.signature(fn).parameters)
                assert not old & params, (name, old & params)
