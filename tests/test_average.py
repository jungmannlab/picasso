"""Test picasso.average functions for particle averaging.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import time

import numpy as np
import pandas as pd
import pytest

from picasso import average, lib, render
from tests._progress_tracking import (
    CancelingProgress,
    RecordingProgress,
    assert_phases_complete,
)


class TestComputeXcorr:
    """Tests for compute_xcorr function."""

    def test_autocorr_peak_at_center(self):
        """Autocorrelation of a delta image peaks at the fftshift center."""
        image = np.zeros((5, 5), dtype=np.float32)
        image[1, 2] = 1.0
        CF = np.conj(np.fft.fft2(image))

        xcorr = average.compute_xcorr(CF, image)

        peak = np.unravel_index(xcorr.argmax(), xcorr.shape)
        # fftshift places zero-shift at index N // 2.
        assert peak == (2, 2)
        assert np.isclose(xcorr[peak], 1.0)

    def test_xcorr_recovers_translation(self):
        """Peak location encodes the translation between two delta images."""
        a = np.zeros((7, 7), dtype=np.float32)
        b = np.zeros((7, 7), dtype=np.float32)
        a[2, 3] = 1.0
        b[4, 5] = 1.0  # shifted from a by (dy=2, dx=2)

        CF_a = np.conj(np.fft.fft2(a))
        xcorr = average.compute_xcorr(CF_a, b)

        peak = np.unravel_index(xcorr.argmax(), xcorr.shape)
        # Center is at (3, 3) for N=7; peak should land at center + shift.
        assert peak == (5, 5)

    @pytest.mark.parametrize("shape", [(3, 3), (5, 5), (8, 8)])
    def test_xcorr_shape(self, shape):
        rng = np.random.default_rng(0)
        image = rng.random(shape, dtype=np.float32)
        CF = np.conj(np.fft.fft2(image))
        assert average.compute_xcorr(CF, image).shape == shape


class TestAlignGroupCore:
    """Tests for align_group_core function."""

    @staticmethod
    def _avg_image_inputs(x, y, oversampling, t_min, t_max):
        _, image_avg = render.render_hist(x, y, oversampling, t_min, t_max)
        CF_image_avg = np.conj(np.fft.fft2(image_avg))
        image_half = image_avg.shape[0] / 2
        return CF_image_avg, image_half

    def test_no_shift_when_group_equals_average(self):
        """A group identical to the average should not be moved."""
        index = np.array([0, 1, 2])
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        angles = np.array([0.0], dtype=np.float32)
        oversampling, t_min, t_max = 1.0, -2.0, 2.0

        CF, ih = self._avg_image_inputs(x, y, oversampling, t_min, t_max)

        x_aligned, y_aligned = average.align_group_core(
            index,
            x.copy(),
            y.copy(),
            angles,
            oversampling,
            t_min,
            t_max,
            CF,
            ih,
        )

        np.testing.assert_allclose(x_aligned, x)
        np.testing.assert_allclose(y_aligned, y)

    def test_subset_returns_subset_size(self):
        """Only the indexed subset is aligned and returned."""
        index = np.array([0, 2])
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        angles = np.array([0.0], dtype=np.float32)
        oversampling, t_min, t_max = 1.0, -2.0, 2.0

        CF, ih = self._avg_image_inputs(x, y, oversampling, t_min, t_max)

        x_aligned, y_aligned = average.align_group_core(
            index,
            x.copy(),
            y.copy(),
            angles,
            oversampling,
            t_min,
            t_max,
            CF,
            ih,
        )

        assert x_aligned.shape == (2,)
        assert y_aligned.shape == (2,)
        assert np.isfinite(x_aligned).all()
        assert np.isfinite(y_aligned).all()


class TestBuildGroupIndex:
    """Tests for build_group_index."""

    def test_maps_groups_to_loc_indices(self):
        locs = pd.DataFrame(
            {
                "x": np.zeros(5, dtype=np.float32),
                "y": np.zeros(5, dtype=np.float32),
                "group": [0, 1, 0, 2, 1],
            }
        )
        gi = average.build_group_index(locs)
        assert gi.shape == (3, 5)
        assert gi.dtype == bool
        np.testing.assert_array_equal(gi[0].nonzero()[1], [0, 2])
        np.testing.assert_array_equal(gi[1].nonzero()[1], [1, 4])
        np.testing.assert_array_equal(gi[2].nonzero()[1], [3])

    def test_single_group(self):
        locs = pd.DataFrame(
            {
                "x": np.zeros(4, dtype=np.float32),
                "y": np.zeros(4, dtype=np.float32),
                "group": [7, 7, 7, 7],
            }
        )
        gi = average.build_group_index(locs)
        assert gi.shape == (1, 4)
        np.testing.assert_array_equal(gi[0].nonzero()[1], [0, 1, 2, 3])


class TestComAlign:
    """Tests for com_align."""

    def test_each_group_centered_at_origin(self):
        locs = pd.DataFrame(
            {
                "x": [10.0, 12.0, 100.0, 102.0],
                "y": [-5.0, -3.0, 50.0, 52.0],
                "group": [0, 0, 1, 1],
            }
        )
        gi = average.build_group_index(locs)
        out = average.com_align(locs, gi)

        for g in (0, 1):
            mask = out["group"] == g
            assert np.isclose(out.loc[mask, "x"].mean(), 0.0)
            assert np.isclose(out.loc[mask, "y"].mean(), 0.0)

    def test_does_not_mutate_input(self):
        locs = pd.DataFrame(
            {"x": [10.0, 12.0], "y": [-5.0, -3.0], "group": [0, 0]}
        )
        original = locs.copy()
        gi = average.build_group_index(locs)
        average.com_align(locs, gi)
        pd.testing.assert_frame_equal(locs, original)


class TestPrepareLocsForSave:
    """Tests for prepare_locs_for_save."""

    def test_shifts_to_positive_coords(self):
        """The averaged particle is moved to the center of the field of
        view (nm)."""
        locs = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [-2.0, 0.0, 2.0]})
        info = [{"Width": 100, "Height": 200, "Pixelsize": 10.0}]

        out, _ = average.prepare_locs_for_save(locs.copy(), info)

        np.testing.assert_allclose(out["x"], [499.0, 500.0, 501.0])
        np.testing.assert_allclose(out["y"], [998.0, 1000.0, 1002.0])

    def test_appends_metadata_entry(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10, "Pixelsize": 10.0}]

        _, new_info = average.prepare_locs_for_save(locs.copy(), info)

        assert len(new_info) == len(info) + 1
        assert "Generated by" in new_info[-1]

    def test_params_added_to_metadata(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10, "Pixelsize": 10.0}]
        params = {"disp_px_size": 5.0, "it": 3}

        _, new_info = average.prepare_locs_for_save(locs.copy(), info, params)

        assert new_info[-1]["Display pixel size (nm)"] == 5.0
        assert new_info[-1]["Iterations"] == 3

    def test_params_partial(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10, "Pixelsize": 10.0}]
        params = {"disp_px_size": 2.5}

        _, new_info = average.prepare_locs_for_save(locs.copy(), info, params)

        assert new_info[-1]["Display pixel size (nm)"] == 2.5
        assert "Iterations" not in new_info[-1]

    def test_params_ignores_unknown_keys(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10, "Pixelsize": 10.0}]
        params = {"unknown_key": "value"}

        _, new_info = average.prepare_locs_for_save(locs.copy(), info, params)

        assert "unknown_key" not in new_info[-1]
        assert "Display pixel size (nm)" not in new_info[-1]
        assert "Iterations" not in new_info[-1]

    def test_empty_params_default(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10, "Pixelsize": 10.0}]

        _, new_info = average.prepare_locs_for_save(locs.copy(), info)

        assert "Display pixel size (nm)" not in new_info[-1]
        assert "Iterations" not in new_info[-1]


class TestAverageParticles:
    """Tests for the top-level average function."""

    @pytest.fixture
    def sample_locs(self):
        rng = np.random.default_rng(42)
        n_locs = 100
        # particles 65 nm wide (nm), rendered at 65 nm pixels below to
        # keep the images small for a fast test
        return pd.DataFrame(
            {
                "x": rng.standard_normal(n_locs).astype(np.float32) * 65.0,
                "y": rng.standard_normal(n_locs).astype(np.float32) * 65.0,
                "group": np.repeat(np.arange(10), 10),
            }
        )

    @pytest.fixture
    def sample_info(self):
        return [{"Width": 256, "Height": 256, "Pixelsize": 130}]

    def test_returns_dataframe_of_same_length(self, sample_locs, sample_info):
        result = average.average(
            sample_locs,
            sample_info,
            display_pixel_size=65,
            iterations=1,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_locs)
        assert {"x", "y", "group"}.issubset(result.columns)

    def test_output_centered_around_origin(self, sample_locs, sample_info):
        result = average.average(
            sample_locs,
            sample_info,
            display_pixel_size=65,
            iterations=1,
        )

        assert np.abs(result["x"].mean()) < 13.0  # nm
        assert np.abs(result["y"].mean()) < 13.0

    def test_iterations_reduce_per_group_spread(
        self, sample_locs, sample_info
    ):
        """More iterations should not increase the per-group spread."""
        spread = lambda df: (
            df.groupby("group")[["x", "y"]].std().to_numpy().mean()
        )

        result1 = average.average(
            sample_locs.copy(),
            sample_info,
            display_pixel_size=65,
            iterations=1,
        )
        result3 = average.average(
            sample_locs.copy(),
            sample_info,
            display_pixel_size=65,
            iterations=3,
        )

        assert spread(result3) <= spread(result1) + 1e-6

    def test_progress_phase_per_iteration(self, sample_locs, sample_info):
        tracker = RecordingProgress()
        average.average(
            sample_locs,
            sample_info,
            display_pixel_size=65,
            iterations=2,
            progress=tracker,
        )
        assert_phases_complete(
            tracker,
            ["Averaging (iteration 1/2)", "Averaging (iteration 2/2)"],
        )
        n_groups = sample_locs["group"].nunique()
        assert [phase["maximum"] for phase in tracker.phases] == [n_groups] * 2
        assert {phase["unit"] for phase in tracker.phases} == {"group"}

    def test_iteration_callback_gets_the_locs_after_each_iteration(
        self, sample_locs, sample_info
    ):
        calls = []

        def on_iteration(done, locs):
            calls.append((done, locs.copy()))

        result = average.average(
            sample_locs,
            sample_info,
            display_pixel_size=65,
            iterations=2,
            iteration_callback=on_iteration,
        )
        assert [done for done, _ in calls] == [1, 2]
        assert all(len(locs) == len(sample_locs) for _, locs in calls)
        # the last iteration's locs are the result
        np.testing.assert_array_equal(calls[-1][1]["x"], result["x"])
        # and every iteration's locs are centered
        for _, locs in calls:
            assert abs(locs["x"].mean()) < 1e-3
            assert abs(locs["y"].mean()) < 1e-3

    def test_progress_none_is_silent_and_console_works(
        self, sample_locs, sample_info, capsys
    ):
        kwargs = dict(display_pixel_size=65, iterations=1)
        average.average(sample_locs.copy(), sample_info, **kwargs)
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        average.average(
            sample_locs.copy(), sample_info, progress="console", **kwargs
        )
        assert "Averaging (iteration 1/1)" in capsys.readouterr().err

    def test_cancel_propagates_and_keeps_an_idle_pool(
        self, sample_locs, sample_info
    ):
        """Canceled before any work was queued, the cancellation propagates
        and the caller's pool stays usable (only a pool with queued work is
        terminated)."""
        pool = average.AveragePool(sample_locs)
        try:
            with pytest.raises(lib.OperationCanceled):
                average.average(
                    sample_locs,
                    sample_info,
                    display_pixel_size=65,
                    iterations=2,
                    progress=CancelingProgress(after=1),
                    pool=pool,
                )
            assert not pool.closed
            result = average.average(
                sample_locs,
                sample_info,
                display_pixel_size=65,
                iterations=1,
                pool=pool,
            )
            assert len(result) == len(sample_locs)
        finally:
            pool.terminate()


class TestWaitForAlignment:
    """Regression: the wait for a round of alignment spun on
    ``result.ready()`` without sleeping, copying the localizations on every
    pass."""

    class _Result:
        def __init__(self, duration):
            self.t_done = time.monotonic() + duration
            self.n_ready = 0

        def ready(self):
            self.n_ready += 1
            return time.monotonic() >= self.t_done

        def wait(self, timeout):
            time.sleep(min(timeout, max(0.0, self.t_done - time.monotonic())))

        def get(self):
            return None

    class _Counter:
        value = 3

    def test_polls_instead_of_spinning(self):
        result = self._Result(0.3)
        tracker = RecordingProgress()
        lib.start_phase(tracker, "Averaging", 5, unit="group")
        aborted = average._wait_for_alignment(
            result, None, self._Counter(), 5, tracker
        )
        assert not aborted
        # about 0.3 s / 0.05 s polls, not thousands
        assert result.n_ready < 30
        assert_phases_complete(tracker)

    def test_abort(self):
        result = self._Result(10.0)
        aborted = average._wait_for_alignment(
            result, lambda: True, self._Counter(), 5, lib.MockProgress()
        )
        assert aborted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
