"""
test_scmos
~~~~~~~~~~

Tests for ``picasso.scmos``, the per-pixel sCMOS camera characterization.

Every estimator here has an independent oracle, so the tests check numbers
rather than shapes:

- The **mean and variance** are checked against a plain two-pass NumPy
  reference, and separately against the literal ``<s^2> - o^2`` of the paper's
  Eq. 2.2 on a large-offset sensor, where that form is expected to lose
  precision. That second comparison is the point of the Chan merge, so it is
  asserted as a *difference*, not an agreement.
- The **gain** is checked against the value used to synthesize a
  photon-transfer curve.
- The **reliability test** is checked in both directions: a matched test movie
  must pass, and a movie from a drifted camera must fail.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from picasso import lib, scmos
from tests._progress_tracking import RecordingProgress, assert_phases_complete


def _moments(movie, progress=None, **kwargs):
    """``scmos._streaming_moments`` with a silent tracker by default, since
    the private helper takes a normalized one."""
    if progress is None:
        progress = lib.MockProgress()
    return scmos._streaming_moments(movie, progress=progress, **kwargs)


# ---- streaming moments ----------------------------------------------------


class TestStreamingMoments:
    def test_matches_numpy_two_pass(self, scmos_maps, dark_movie_factory):
        frames = dark_movie_factory(scmos_maps, 500, seed=1)
        n, mean, variance = _moments(frames)
        assert n == 500
        np.testing.assert_allclose(mean, frames.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(
            variance, frames.var(axis=0, ddof=1), rtol=1e-10
        )

    def test_beats_the_literal_eq_2_2_on_a_large_offset(self):
        """The reason Eq. 2.2 is not implemented as written.

        A 16-bit sensor parked at 32768 ADU with a readout variance of a few
        ADU squared is the cancellation worst case: ``<s^2>`` and ``o^2`` agree
        to eleven digits and their difference is the answer.
        """
        rng = np.random.default_rng(3)
        offset, sigma, n = 32768.0, 1.5, 4000
        frames = rng.normal(offset, sigma, (n, 4, 4))
        reference = frames.var(axis=0, ddof=0)

        _, _, chan = _moments(frames)
        chan_ddof0 = chan * (n - 1) / n
        literal32 = (frames.astype(np.float32) ** 2).mean(
            axis=0, dtype=np.float32
        ) - frames.astype(np.float32).mean(axis=0, dtype=np.float32) ** 2

        np.testing.assert_allclose(chan_ddof0, reference, rtol=1e-9)
        # The float32 literal form does not merely lose digits here, it goes
        # negative - which would make the noise model meaningless.
        assert literal32.min() < 0.0

    @pytest.mark.parametrize(
        "chunk_bytes", [8 * 16 * 16, 8 * 16 * 16 * 7, 1 << 30]
    )
    def test_chunking_is_invariant(
        self, scmos_maps, dark_movie_factory, monkeypatch, chunk_bytes
    ):
        frames = dark_movie_factory(scmos_maps, 300, seed=2)
        _, mean_ref, var_ref = _moments(frames)
        monkeypatch.setattr(scmos, "_CHUNK_BYTES", chunk_bytes)
        _, mean, variance = _moments(frames)
        np.testing.assert_allclose(mean, mean_ref, rtol=1e-12)
        np.testing.assert_allclose(variance, var_ref, rtol=1e-10)

    def test_works_through_an_abstract_movie(
        self, scmos_maps, dark_movie_factory, picasso_movie_factory
    ):
        """The frame-by-frame path, used by every non-ndarray movie type."""
        frames = dark_movie_factory(scmos_maps, 200, seed=4, dtype=np.uint16)
        _, mean_ref, var_ref = _moments(frames)
        wrapped = picasso_movie_factory(frames, [{"Frames": len(frames)}])
        _, mean, variance = _moments(wrapped)
        np.testing.assert_allclose(mean, mean_ref, rtol=1e-12)
        np.testing.assert_allclose(variance, var_ref, rtol=1e-10)

    def test_progress_and_abort(self, scmos_maps, dark_movie_factory):
        frames = dark_movie_factory(scmos_maps, 300, seed=5)
        tracker = RecordingProgress()
        lib.start_phase(tracker, "Characterizing", len(frames), unit="frame")
        _moments(frames, progress=tracker)
        assert_phases_complete(tracker)

        assert _moments(frames, abort_callback=lambda: True) is None

    def test_rejects_an_empty_or_non_2d_movie(self):
        with pytest.raises(ValueError, match="empty movie"):
            _moments(np.zeros((0, 4, 4)))
        with pytest.raises(ValueError, match="2D frames"):
            _moments(np.zeros((5, 4)))


# ---- offset and variance --------------------------------------------------


class TestOffsetAndVariance:
    def test_recovers_the_ground_truth_maps(
        self, scmos_maps, dark_movie_factory
    ):
        n = 20_000
        frames = dark_movie_factory(scmos_maps, n, seed=6)
        calibration = scmos.calibrate_scmos(frames)

        offset_tol = 4.0 * np.sqrt(scmos_maps["variance"] / n)
        assert np.all(
            np.abs(calibration["offset"] - scmos_maps["offset"]) < offset_tol
        )
        # Relative standard error of a variance estimate is sqrt(2/(n-1)).
        rel = np.abs(calibration["variance"] / scmos_maps["variance"] - 1.0)
        assert rel.max() < 5.0 * np.sqrt(2 / (n - 1))

    def test_raises_below_the_minimum_frame_count(
        self, scmos_maps, dark_movie_factory
    ):
        frames = dark_movie_factory(scmos_maps, scmos.MIN_DARK_FRAMES - 1)
        with pytest.raises(ValueError, match="at least"):
            scmos.calibrate_scmos(frames)

    def test_warns_below_the_recommended_frame_count(
        self, scmos_maps, dark_movie_factory
    ):
        frames = dark_movie_factory(scmos_maps, scmos.MIN_DARK_FRAMES + 20)
        with pytest.warns(RuntimeWarning, match="60,000"):
            scmos.calibrate_scmos(frames)

    def test_reports_hot_pixels(self, scmos_maps, dark_movie_factory):
        frames = dark_movie_factory(scmos_maps, 5000, seed=7)
        with pytest.warns(RuntimeWarning):
            calibration = scmos.calibrate_scmos(frames)
        assert calibration["Hot pixels"] >= 3
        assert calibration["Variance max (ADU^2)"] > 500.0
        assert calibration["model"] == "scmos-noise"
        assert "gain" not in calibration

    def test_aborts_cleanly(self, scmos_maps, dark_movie_factory):
        frames = dark_movie_factory(scmos_maps, 300)
        assert (
            scmos.calibrate_scmos(frames, abort_callback=lambda: True) is None
        )


# ---- gain -----------------------------------------------------------------


class TestGain:
    @pytest.fixture(scope="class")
    def calibration(self, scmos_maps_factory, dark_movie_factory):
        """A full calibration from a dark movie plus a bright series.

        Fifteen illumination levels spanning 20-200 photons per pixel, as in
        the paper, but with far fewer frames per level so the test stays fast;
        the tolerance below is set from that frame count.
        """
        maps = scmos_maps_factory(height=12, width=12, seed=11)
        dark = dark_movie_factory(maps, 20_000, seed=12)
        bright = [
            dark_movie_factory(maps, 4_000, photons=p, seed=20 + k)
            for k, p in enumerate(np.linspace(20, 200, 15))
        ]
        return maps, scmos.calibrate_scmos(dark, bright)

    def test_recovers_the_ground_truth_gain(self, calibration):
        maps, calib = calibration
        rel = np.abs(calib["gain"] / maps["gain"] - 1.0)
        assert rel.max() < 0.05
        assert np.median(rel) < 0.01

    def test_records_the_gain_provenance(self, calibration):
        _, calib = calibration
        assert calib["Gain levels"] == 15
        assert calib["Gain frames"] == 15 * 4_000
        assert calib["Gain fallback pixels"] == 0
        assert 1.5 < calib["Gain median (ADU/e-)"] < 3.5

    def test_records_the_response_of_every_level(self, calibration):
        """The per-level medians the linearity panel is drawn from: one entry
        per movie, in the order given, rising with the illumination."""
        _, calib = calibration
        signal = calib["Level median signal (ADU)"]
        variance = calib["Level median variance (ADU^2)"]
        assert len(signal) == len(variance) == 15
        assert calib["Level frames"] == [4_000] * 15
        assert np.all(np.diff(signal) > 0)
        assert np.all(np.diff(variance) > 0)
        # The series spans 20-200 photons, so the last level reads about ten
        # times the first, whatever the gain.
        assert signal[-1] / signal[0] == pytest.approx(10.0, rel=0.05)
        # Signal and excess variance both scale with the photon count, so
        # their ratio is the gain.
        ratio = np.asarray(variance) / np.asarray(signal)
        assert np.allclose(ratio, calib["Gain median (ADU/e-)"], rtol=0.1)

    def test_records_the_illumination_when_it_is_given(
        self, scmos_maps_factory, dark_movie_factory
    ):
        """Unevenly spaced levels, which is what the plot needs them for."""
        maps = scmos_maps_factory(height=8, width=8, seed=61)
        dark = dark_movie_factory(maps, 2_000, seed=62)
        powers = [0.1, 1.0, 5.0, 10.0]
        bright = [
            dark_movie_factory(maps, 1_000, photons=20 * p, seed=70 + k)
            for k, p in enumerate(powers)
        ]
        calib = scmos.calibrate_scmos(
            dark, bright, bright_levels=powers, level_unit="mW"
        )
        assert calib["Level illumination"] == powers
        assert calib["Level unit"] == "mW"
        # The signal follows the power, not the level index.
        signal = np.asarray(calib["Level median signal (ADU)"])
        assert signal[3] / signal[1] == pytest.approx(10.0, rel=0.05)

    def test_illumination_defaults_to_unknown(self, calibration):
        _, calib = calibration
        assert calib["Level illumination"] == []
        assert calib["Level unit"] == ""

    def test_one_illumination_per_bright_movie(
        self, scmos_maps, dark_movie_factory
    ):
        dark = dark_movie_factory(scmos_maps, 200, seed=63)
        bright = [dark_movie_factory(scmos_maps, 100, photons=50, seed=64)]
        with pytest.raises(ValueError, match="one level per movie"):
            scmos.calibrate_scmos(dark, bright, bright_levels=[1.0, 2.0])

    def test_falls_back_for_an_unresponsive_pixel(
        self, scmos_maps_factory, dark_movie_factory
    ):
        """A dead pixel has no slope; it must take the chip median, not NaN."""
        maps = scmos_maps_factory(height=8, width=8, seed=13)
        dark = dark_movie_factory(maps, 20_000, seed=14)
        bright = [
            dark_movie_factory(maps, 2_000, photons=p, seed=30 + k)
            for k, p in enumerate((30.0, 90.0, 180.0))
        ]
        for frames in bright:  # pixel (0, 0) sees no light at any level
            frames[:, 0, 0] = dark[: len(frames), 0, 0]

        calib = scmos.calibrate_scmos(dark, bright)
        assert np.isfinite(calib["gain"]).all()
        assert (calib["gain"] > 0).all()
        assert calib["Gain fallback pixels"] >= 1
        assert calib["gain"][0, 0] == pytest.approx(
            np.median(calib["gain"]), rel=0.2
        )

    def test_warns_on_a_single_illumination_level(
        self, scmos_maps, dark_movie_factory
    ):
        dark = dark_movie_factory(scmos_maps, 20_000, seed=15)
        bright = [dark_movie_factory(scmos_maps, 1_000, photons=100.0)]
        with pytest.warns(RuntimeWarning, match="single illumination level"):
            scmos.calibrate_scmos(dark, bright)

    def test_rejects_a_mismatched_bright_movie(
        self, scmos_maps, scmos_maps_factory, dark_movie_factory
    ):
        dark = dark_movie_factory(scmos_maps, 20_000, seed=16)
        other = dark_movie_factory(
            scmos_maps_factory(height=8, width=8), 500, photons=50.0
        )
        with pytest.raises(ValueError, match="same camera ROI"):
            scmos.calibrate_scmos(dark, [other])


# ---- reliability test -----------------------------------------------------


class TestValidateCalibration:
    @pytest.fixture(scope="class")
    def calibration(self, scmos_maps_factory, dark_movie_factory):
        maps = scmos_maps_factory(height=32, width=32, seed=40)
        dark = dark_movie_factory(maps, 20_000, seed=41)
        return maps, scmos.calibrate_scmos(dark)

    def test_accepts_a_matched_camera(self, calibration, dark_movie_factory):
        maps, calib = calibration
        test = dark_movie_factory(maps, scmos.VALIDATION_FRAMES, seed=42)
        report = scmos.validate_calibration(calib, test)
        assert report["valid"]
        assert report["mean p-value"] == pytest.approx(0.5, abs=0.1)

    def test_rejects_a_drifted_camera(self, calibration, dark_movie_factory):
        """A global change in readout noise is what the test is for."""
        maps, calib = calibration
        drifted = dict(maps, variance=maps["variance"] * 1.6)
        test = dark_movie_factory(drifted, scmos.VALIDATION_FRAMES, seed=43)
        report = scmos.validate_calibration(calib, test)
        assert not report["valid"]
        assert report["mean p-value"] < 0.4

    def test_rejects_a_mismatched_frame_size(
        self, calibration, scmos_maps_factory, dark_movie_factory
    ):
        _, calib = calibration
        test = dark_movie_factory(scmos_maps_factory(height=8, width=8), 100)
        with pytest.raises(ValueError, match="same camera ROI"):
            scmos.validate_calibration(calib, test)

    def test_aborts_cleanly(self, calibration, dark_movie_factory):
        maps, calib = calibration
        test = dark_movie_factory(maps, 200, seed=44)
        assert (
            scmos.validate_calibration(
                calib, test, abort_callback=lambda: True
            )
            is None
        )


class TestCalibrationPlot:
    """The diagnostic PNG written alongside a calibration."""

    @staticmethod
    def _calibration(rng, *, gain=True, uniform=False):
        shape = (16, 16)
        if uniform:
            calibration = {
                "offset": np.full(shape, 100.0, np.float32),
                "variance": np.full(shape, 4.0, np.float32),
            }
            if gain:
                calibration["gain"] = np.full(shape, 2.0, np.float32)
            return calibration
        variance = rng.lognormal(1.0, 0.8, shape)
        variance[0, 0] = 2500.0  # a hot pixel
        calibration = {
            "offset": rng.normal(100, 2, shape).astype(np.float32),
            "variance": variance.astype(np.float32),
        }
        if gain:
            calibration["gain"] = rng.normal(2.13, 0.1, shape).astype(
                np.float32
            )
        return calibration

    def test_plot_path_sits_next_to_the_calibration(self):
        assert (
            scmos.plot_path("/data/mycam_scmos_calib.hdf5")
            == "/data/mycam_scmos_calib_maps.png"
        )

    def test_writes_a_png(self, tmp_path):
        rng = np.random.default_rng(0)
        path = str(tmp_path / "calib_maps.png")
        returned = scmos.save_calibration_plot(self._calibration(rng), path)
        assert returned == path
        # A PNG, not an empty file or another format.
        with open(path, "rb") as file:
            assert file.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_writes_a_png_without_a_gain_map(self, tmp_path):
        rng = np.random.default_rng(1)
        path = str(tmp_path / "nogain_maps.png")
        scmos.save_calibration_plot(self._calibration(rng, gain=False), path)
        assert os.path.getsize(path) > 0

    def test_a_uniform_map_does_not_break_the_scaling(self, tmp_path):
        """A simulated or hand-made calibration can be perfectly uniform, and
        then every percentile coincides."""
        path = str(tmp_path / "uniform_maps.png")
        scmos.save_calibration_plot(
            self._calibration(None, uniform=True), path
        )
        assert os.path.getsize(path) > 0

    def test_a_real_calibration_round_trips_through_the_plot(
        self, scmos_maps_factory, dark_movie_factory, tmp_path
    ):
        """What ``calibrate_scmos`` returns must be plottable as it is."""
        maps = scmos_maps_factory(height=16, width=16, seed=7)
        calib = scmos.calibrate_scmos(dark_movie_factory(maps, 200, seed=8))
        path = str(tmp_path / "real_maps.png")
        scmos.save_calibration_plot(calib, path)
        assert os.path.getsize(path) > 0

    def test_plots_against_the_illumination_when_it_is_known(self, tmp_path):
        """An unevenly spaced series must still be plottable, both with the
        illumination recorded and without it."""
        rng = np.random.default_rng(3)
        base = dict(
            self._calibration(rng),
            **{
                "Level median signal (ADU)": [4.2, 41.5, 207.3, 414.1],
                "Level median variance (ADU^2)": [8.7, 86.5, 432.0, 863.0],
                "Level frames": [1000] * 4,
            },
        )
        with_power = dict(
            base,
            **{
                "Level illumination": [0.1, 1.0, 5.0, 10.0],
                "Level unit": "mW",
            },
        )
        for name, calibration in (
            ("power", with_power),
            ("nopower", base),
        ):
            path = str(tmp_path / f"{name}_maps.png")
            scmos.save_calibration_plot(calibration, path)
            assert os.path.getsize(path) > 0

    def test_the_level_row_only_appears_with_a_bright_series(self, tmp_path):
        """The linearity row is worth a taller figure only when there is a
        series to draw on it."""
        rng = np.random.default_rng(2)
        bare = self._calibration(rng)
        with_levels = dict(
            bare,
            **{
                "Level median signal (ADU)": [40.0, 120.0, 200.0],
                "Level median variance (ADU^2)": [85.0, 255.0, 425.0],
                "Level frames": [1000, 1000, 1000],
            },
        )
        one_level = dict(
            bare,
            **{
                "Level median signal (ADU)": [40.0],
                "Level median variance (ADU^2)": [85.0],
                "Level frames": [1000],
            },
        )
        sizes = {}
        for name, calibration in (
            ("bare", bare),
            ("one", one_level),
            ("levels", with_levels),
        ):
            path = str(tmp_path / f"{name}_maps.png")
            scmos.save_calibration_plot(calibration, path)
            with Image.open(path) as image:
                sizes[name] = image.size
        assert sizes["one"] == sizes["bare"]
        assert sizes["levels"][1] > sizes["bare"][1]


# ---- progress ---------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Only .* dark frames:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:The gain map was estimated:RuntimeWarning")
class TestProgress:
    """``progress``: one phase per movie, each ending at its frame count."""

    @pytest.fixture
    def movies(self, scmos_maps_factory, dark_movie_factory):
        maps = scmos_maps_factory(height=6, width=6, seed=60)
        dark = dark_movie_factory(maps, 150, seed=61)
        bright = [
            dark_movie_factory(maps, n, photons=p, seed=62 + k)
            for k, (n, p) in enumerate([(40, 50.0), (60, 150.0)])
        ]
        return dark, bright

    def test_one_phase_per_movie(self, movies):
        """Regression: one running count across all movies was reported, so
        a tracker saw the dark frames' bar overrun into the bright ones."""
        dark, bright = movies
        tracker = RecordingProgress()
        scmos.calibrate_scmos(dark, bright, progress=tracker)
        assert_phases_complete(
            tracker,
            [
                "Characterizing dark movie",
                "Characterizing bright movie 1/2",
                "Characterizing bright movie 2/2",
            ],
        )
        assert [phase["maximum"] for phase in tracker.phases] == [150, 40, 60]
        assert {phase["unit"] for phase in tracker.phases} == {"frame"}

    def test_dark_movie_only(self, movies):
        dark, _ = movies
        tracker = RecordingProgress()
        scmos.calibrate_scmos(dark, progress=tracker)
        assert_phases_complete(tracker, ["Characterizing dark movie"])

    def test_validation(self, movies):
        dark, bright = movies
        calibration = scmos.calibrate_scmos(dark)
        tracker = RecordingProgress()
        scmos.validate_calibration(calibration, bright[0], progress=tracker)
        assert_phases_complete(tracker, ["Validating"])
        assert tracker.phases[0]["maximum"] == len(bright[0])

    def test_none_is_silent_and_console_works(self, movies, capsys):
        dark, bright = movies
        calibration = scmos.calibrate_scmos(dark, bright)
        scmos.validate_calibration(calibration, bright[0])
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        calibration = scmos.calibrate_scmos(dark, bright, progress="console")
        err = capsys.readouterr().err
        assert "Characterizing dark movie" in err
        assert "Characterizing bright movie 2/2" in err
        scmos.validate_calibration(calibration, bright[0], progress="console")
        assert "Validating" in capsys.readouterr().err
