"""Test picasso.average functions for particle averaging.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pandas as pd
import pytest

from picasso import average, render


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
        _, image_avg = render.render_hist_numba(
            x, y, oversampling, t_min, t_max
        )
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
        locs = pd.DataFrame({"x": [-1.0, 0.0, 1.0], "y": [-2.0, 0.0, 2.0]})
        info = [{"Width": 100, "Height": 200}]

        out, _ = average.prepare_locs_for_save(locs.copy(), info)

        np.testing.assert_allclose(out["x"], [49.0, 50.0, 51.0])
        np.testing.assert_allclose(out["y"], [98.0, 100.0, 102.0])

    def test_appends_metadata_entry(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10}]

        _, new_info = average.prepare_locs_for_save(locs.copy(), info)

        assert len(new_info) == len(info) + 1
        assert "Generated by" in new_info[-1]

    def test_params_added_to_metadata(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10}]
        params = {"disp_px_size": 5.0, "it": 3}

        _, new_info = average.prepare_locs_for_save(locs.copy(), info, params)

        assert new_info[-1]["Display pixel size (nm)"] == 5.0
        assert new_info[-1]["Iterations"] == 3

    def test_params_partial(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10}]
        params = {"disp_px_size": 2.5}

        _, new_info = average.prepare_locs_for_save(locs.copy(), info, params)

        assert new_info[-1]["Display pixel size (nm)"] == 2.5
        assert "Iterations" not in new_info[-1]

    def test_params_ignores_unknown_keys(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10}]
        params = {"unknown_key": "value"}

        _, new_info = average.prepare_locs_for_save(locs.copy(), info, params)

        assert "unknown_key" not in new_info[-1]
        assert "Display pixel size (nm)" not in new_info[-1]
        assert "Iterations" not in new_info[-1]

    def test_empty_params_default(self):
        locs = pd.DataFrame({"x": [0.0], "y": [0.0]})
        info = [{"Width": 10, "Height": 10}]

        _, new_info = average.prepare_locs_for_save(locs.copy(), info)

        assert "Display pixel size (nm)" not in new_info[-1]
        assert "Iterations" not in new_info[-1]


class TestAverageParticles:
    """Tests for the top-level average function."""

    @pytest.fixture
    def sample_locs(self):
        rng = np.random.default_rng(42)
        n_locs = 100
        return pd.DataFrame(
            {
                "x": rng.standard_normal(n_locs).astype(np.float32) * 0.5,
                "y": rng.standard_normal(n_locs).astype(np.float32) * 0.5,
                "group": np.repeat(np.arange(10), 10),
            }
        )

    @pytest.fixture
    def sample_info(self):
        # Pixelsize is required by average(); display_pixel_size=65 keeps
        # oversampling=2 and the rendered image small for a fast test.
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

        assert np.abs(result["x"].mean()) < 0.1
        assert np.abs(result["y"].mean()) < 0.1

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

    def test_progress_callback_invoked_with_full_signature(
        self, sample_locs, sample_info
    ):
        progress_calls = []

        def cb(it, total_it, locs, current, total):
            progress_calls.append((it, total_it, current, total))

        average.average(
            sample_locs,
            sample_info,
            display_pixel_size=65,
            iterations=2,
            progress_callback=cb,
        )

        assert len(progress_calls) >= 2
        # Final call always reports the completed iteration count and
        # all groups processed.
        last_it, last_total, last_current, last_n = progress_calls[-1]
        assert (last_it, last_total) == (2, 2)
        assert last_current == last_n

    def test_return_shifted_locs(self, sample_locs, sample_info):
        """return_shifted_locs=True returns a (locs, info) tuple shifted
        into positive coordinates and with appended metadata."""
        locs_out, info_out = average.average(
            sample_locs,
            sample_info,
            display_pixel_size=65,
            iterations=1,
            return_shifted_locs=True,
        )

        assert (locs_out["x"] > 0).all()
        assert (locs_out["y"] > 0).all()
        assert len(info_out) == len(sample_info) + 1

    def test_missing_group_column_raises(self, sample_info):
        locs = pd.DataFrame(
            {
                "x": np.zeros(3, dtype=np.float32),
                "y": np.zeros(3, dtype=np.float32),
            }
        )
        with pytest.raises(AssertionError):
            average.average(
                locs,
                sample_info,
                display_pixel_size=65,
                iterations=1,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
