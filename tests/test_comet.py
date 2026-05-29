"""Tests for picasso.comet — drift-correction with COMET.

Tests are split into two groups:
- Pure-Python helpers (segmentation, pairing, interpolation): always run when
  numba is installed, no GPU required.
- Full pipeline (comet.comet): requires CUDA hardware; skipped automatically
  on CPU-only machines, but the no-GPU error path IS tested on any machine.

:author: Lenny Reinkensmeier, 2026
"""

import numpy as np
import pandas as pd
import pytest

# The whole picasso package requires numba (lib.py imports it).
# Skip this module cleanly when numba is absent (e.g. CI without GPU stack).
pytest.importorskip("numba")

from picasso.ext import comet  # noqa: E402 — import after guard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_info():
    """Minimal Picasso info list with required Pixelsize entry."""
    return [{"Pixelsize": 130, "Frames": 200}]


@pytest.fixture
def minimal_locs_df():
    """Small synthetic localisation DataFrame spread over 200 frames."""
    rng = np.random.default_rng(0)
    n_frames = 200
    n_per_frame = 5
    frames = np.repeat(np.arange(n_frames), n_per_frame)
    x = rng.uniform(5, 25, size=len(frames)).astype(np.float32)
    y = rng.uniform(5, 25, size=len(frames)).astype(np.float32)
    return pd.DataFrame({"frame": frames, "x": x, "y": y})


@pytest.fixture
def minimal_locs_structured(minimal_locs_df):
    """Same data as a numpy structured array (Picasso's native format)."""
    df = minimal_locs_df
    dtype = np.dtype(
        [("frame", np.uint32), ("x", np.float32), ("y", np.float32)]
    )
    arr = np.empty(len(df), dtype=dtype)
    arr["frame"] = df["frame"].to_numpy(np.uint32)
    arr["x"] = df["x"].to_numpy(np.float32)
    arr["y"] = df["y"].to_numpy(np.float32)
    return arr


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------


class TestSegmentationByLocs:
    def test_produces_correct_segment_count(self):
        frames = np.repeat(np.arange(100), 5)  # 500 locs, 5/frame
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=50,
            segmentation_mode=1,
            return_param_dict=True,
        )
        assert result.n_segments == 10

    def test_all_locs_assigned(self):
        frames = np.repeat(np.arange(50), 10)
        result = comet.segmentation_wrapper(
            frames, segmentation_var=100, segmentation_mode=1
        )
        assert (result.loc_segments >= 0).all()

    def test_loc_valid_subset_of_locs(self):
        frames = np.repeat(np.arange(60), 8)
        result = comet.segmentation_wrapper(
            frames, segmentation_var=40, segmentation_mode=1
        )
        assert result.loc_valid.shape == frames.shape
        assert result.loc_valid.dtype == bool

    def test_center_frames_length_matches_n_segments(self):
        frames = np.repeat(np.arange(80), 5)
        result = comet.segmentation_wrapper(
            frames, segmentation_var=50, segmentation_mode=1
        )
        assert len(result.center_frames) == result.n_segments

    def test_max_locs_per_segment_respected(self):
        frames = np.repeat(np.arange(20), 100)  # 2000 locs, 100/frame
        cap = 30
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=200,
            segmentation_mode=1,
            max_locs_per_segment=cap,
            return_param_dict=True,
        )
        assert result.loc_valid.sum() <= result.n_segments * cap


class TestSegmentationByFrameWindows:
    def test_frame_window_mode(self):
        frames = np.repeat(np.arange(100), 3)
        result = comet.segmentation_wrapper(
            frames, segmentation_var=10, segmentation_mode=2
        )
        assert result.n_segments == 10  # 100 frames / 10 per window

    def test_single_window(self):
        frames = np.arange(50)
        result = comet.segmentation_wrapper(
            frames, segmentation_var=100, segmentation_mode=2
        )
        assert result.n_segments == 1


class TestSegmentationByNumWindows:
    def test_num_windows_mode_creates_requested_count(self):
        frames = np.repeat(np.arange(100), 5)  # 500 locs
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=5,
            segmentation_mode=0,
            return_param_dict=True,
        )
        # mode 0 derives min-locs-per-window from total/n_windows; the resulting
        # segment count should be approximately n_windows (off-by-one tolerated
        # because the last segment is folded into the previous if too small).
        assert abs(result.n_segments - 5) <= 1

    def test_all_locs_assigned(self):
        frames = np.repeat(np.arange(40), 4)
        result = comet.segmentation_wrapper(
            frames, segmentation_var=4, segmentation_mode=0
        )
        assert (result.loc_segments >= 0).all()


class TestSegmentationOutDict:
    """The param dict reports per-segment bookkeeping; verify the values, not
    just that segmentation runs."""

    def test_valid_plus_invalid_equals_total(self):
        frames = np.repeat(np.arange(60), 8)
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=40,
            segmentation_mode=1,
            return_param_dict=True,
        )
        d = result.out_dict
        assert d["n_locs"] == len(frames)
        assert d["n_locs_valid"] + d["n_locs_invalid"] == d["n_locs"]

    def test_per_segment_arrays_have_consistent_length(self):
        frames = np.repeat(np.arange(80), 5)
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=50,
            segmentation_mode=1,
            return_param_dict=True,
        )
        d = result.out_dict
        assert len(d["start_frames"]) == result.n_segments
        assert len(d["end_frames"]) == result.n_segments
        assert len(d["locs_per_segment"]) == result.n_segments

    def test_start_frames_not_after_end_frames(self):
        frames = np.repeat(np.arange(80), 5)
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=50,
            segmentation_mode=1,
            return_param_dict=True,
        )
        d = result.out_dict
        assert (d["start_frames"] <= d["end_frames"]).all()

    def test_center_frames_within_segment_bounds(self):
        frames = np.repeat(np.arange(80), 5)
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=50,
            segmentation_mode=1,
            return_param_dict=True,
        )
        d = result.out_dict
        assert (result.center_frames >= d["start_frames"]).all()
        assert (result.center_frames <= d["end_frames"]).all()

    def test_locs_per_segment_sums_to_valid_count(self):
        frames = np.repeat(np.arange(60), 6)
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=40,
            segmentation_mode=1,
            return_param_dict=True,
        )
        d = result.out_dict
        # locs_per_segment is rewritten to the post-downsampling (valid) count.
        assert d["locs_per_segment"].sum() == d["n_locs_valid"]

    def test_frame_window_center_frames_within_bounds(self):
        frames = np.repeat(np.arange(100), 3)
        result = comet.segmentation_wrapper(
            frames,
            segmentation_var=10,
            segmentation_mode=2,
            return_param_dict=True,
        )
        d = result.out_dict
        assert d["frames_per_window"] == 10
        assert (result.center_frames >= d["start_frames"]).all()
        assert (result.center_frames <= d["end_frames"]).all()


class TestSegmentationDownsamplingByPercentage:
    """max_locs_per_segment < 1 is interpreted as a fraction of the window size."""

    def test_fractional_cap_reduces_valid_locs(self):
        frames = np.repeat(np.arange(20), 50)  # 1000 locs, 50/frame
        full = comet.segmentation_wrapper(
            frames,
            segmentation_var=200,
            segmentation_mode=1,
            return_param_dict=True,
        )
        half = comet.segmentation_wrapper(
            frames,
            segmentation_var=200,
            segmentation_mode=1,
            max_locs_per_segment=0.5,
            return_param_dict=True,
        )
        assert half.out_dict["n_locs_valid"] < full.out_dict["n_locs_valid"]


# ---------------------------------------------------------------------------
# Pair-finding
# ---------------------------------------------------------------------------


class TestPairIndicesKdtree:
    def test_finds_close_pairs(self):
        coords = np.array(
            [[0, 0, 0], [1, 0, 0], [100, 0, 0]], dtype=np.float32
        )
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=5)
        assert ok
        assert len(idx_i) == 1  # only (0,1) are within distance 5

    def test_no_pairs_when_far_apart(self):
        coords = np.eye(3, dtype=np.float32) * 1000
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=1)
        assert ok
        assert len(idx_i) == 0

    def test_all_pairs_when_all_close(self):
        coords = np.zeros((5, 3), dtype=np.float32)
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=1)
        assert ok
        assert len(idx_i) == 5 * 4 // 2  # C(5,2)

    def test_returns_int32_arrays(self):
        coords = np.random.rand(20, 3).astype(np.float32)
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=0.5)
        assert ok
        assert idx_i.dtype == np.int32
        assert idx_j.dtype == np.int32

    def test_2d_coordinates_supported(self):
        coords = np.array([[0, 0], [1, 0], [100, 0]], dtype=np.float32)
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=5)
        assert ok
        assert len(idx_i) == 1  # only (0,1) within distance 5

    def test_pairs_are_lower_index_first(self):
        coords = np.zeros((6, 3), dtype=np.float32)
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=1)
        assert ok
        # query_pairs returns each pair once with i < j; relied on by the kernel.
        assert (idx_i < idx_j).all()

    def test_distance_is_inclusive(self):
        coords = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
        idx_i, idx_j, ok = comet.pair_indices_kdtree(coords, distance=10)
        assert ok
        assert (
            len(idx_i) == 1
        )  # points exactly `distance` apart count as a pair


# ---------------------------------------------------------------------------
# Pair-count estimation
# ---------------------------------------------------------------------------


class TestEstimatePairs:
    def test_far_apart_points_estimate_zero(self):
        # Each point lands in its own grid cell -> no within-cell pairs.
        coords = (
            (np.arange(5)[:, None] * 1000.0).repeat(3, axis=1).astype(float)
        )
        assert comet.estimate_pairs(coords, distance=10) == 0

    def test_clustered_points_estimate_nonzero(self):
        # 200 coincident points -> 200*199 = 39800 -> rounds to 40000.
        coords = np.zeros((200, 3), dtype=float)
        est = comet.estimate_pairs(coords, distance=10)
        assert est == 40000

    def test_result_rounded_to_nearest_ten_thousand(self):
        coords = np.zeros((150, 3), dtype=float)
        est = comet.estimate_pairs(coords, distance=10)
        assert est % 10000 == 0


# ---------------------------------------------------------------------------
# Drift interpolation
# ---------------------------------------------------------------------------


class TestInterpolateDrift:
    def _center_frames_and_drift(self):
        center_frames = np.linspace(10, 90, 6)
        drift_est = np.column_stack(
            [
                np.sin(center_frames / 10),
                np.cos(center_frames / 10),
                np.zeros_like(center_frames),
            ]
        )
        return center_frames, drift_est

    def test_cubic_output_shape(self):
        cf, de = self._center_frames_and_drift()
        frame_range = np.arange(0, 100)
        out = comet.interpolate_drift(cf, de, frame_range, method="cubic")
        assert out.shape == (100, 3)

    def test_linear_output_shape(self):
        cf, de = self._center_frames_and_drift()
        frame_range = np.arange(0, 100)
        out = comet.interpolate_drift(cf, de, frame_range, method="linear")
        assert out.shape == (100, 3)

    def test_cubic_passes_through_control_points(self):
        cf = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        de = np.column_stack([cf * 0.01, cf * 0.005, np.zeros_like(cf)])
        out = comet.interpolate_drift(cf, de, cf.astype(int), method="cubic")
        np.testing.assert_allclose(out[:, 0], de[:, 0], atol=1e-6)

    def test_unknown_method_raises(self):
        cf, de = self._center_frames_and_drift()
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            comet.interpolate_drift(cf, de, np.arange(100), method="spagetti")

    def test_catmull_rom_output_shape(self):
        cf, de = self._center_frames_and_drift()
        frame_range = np.arange(10, 90)
        out = comet.interpolate_drift(
            cf, de, frame_range, method="catmull-rom"
        )
        assert out.shape == (len(frame_range), 3)

    def test_catmull_rom_requires_min_points(self):
        cf = np.array([0.0, 10.0, 20.0])  # only 3 points
        de = np.zeros((3, 3))
        with pytest.raises(ValueError, match="at least 4 points"):
            comet.interpolate_drift(
                cf, de, np.arange(20), method="catmull-rom"
            )

    def test_linear_passes_through_control_points(self):
        cf = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        de = np.column_stack([cf * 0.01, cf * 0.005, cf * 0.002])
        out = comet.interpolate_drift(cf, de, cf.astype(int), method="linear")
        np.testing.assert_allclose(out, de, atol=1e-6)

    def test_linear_clamps_outside_range(self):
        # np.interp clamps to edge values beyond the control-point range.
        cf = np.array([10.0, 20.0, 30.0])
        de = np.column_stack([cf, cf, cf])
        out = comet.interpolate_drift(
            cf, de, np.array([0, 40]), method="linear"
        )
        np.testing.assert_allclose(out[0], [10, 10, 10])
        np.testing.assert_allclose(out[1], [30, 30, 30])

    def test_z_channel_interpolated_independently(self):
        cf = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        de = np.column_stack([np.zeros_like(cf), np.zeros_like(cf), cf * 0.03])
        out = comet.interpolate_drift(cf, de, cf.astype(int), method="cubic")
        np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[:, 2], de[:, 2], atol=1e-6)


# ---------------------------------------------------------------------------
# comet() public API — no-GPU error path
# ---------------------------------------------------------------------------


class TestCometNoCuda:
    """These tests run on any machine; they verify the failure mode when
    CUDA is not present."""

    def test_raises_runtime_error_without_cuda(
        self, minimal_locs_df, minimal_info
    ):
        if comet._CUDA_AVAILABLE:
            pytest.skip(
                "CUDA is present; testing no-GPU path only on CPU machines"
            )
        with pytest.raises(RuntimeError, match="CUDA"):
            comet.comet(
                minimal_locs_df,
                minimal_info,
                locs_per_segment=100,
                max_drift_nm=200,
            )

    def test_error_message_mentions_alternatives(
        self, minimal_locs_df, minimal_info
    ):
        if comet._CUDA_AVAILABLE:
            pytest.skip(
                "CUDA is present; testing no-GPU path only on CPU machines"
            )
        with pytest.raises(RuntimeError, match="AIM|RCC"):
            comet.comet(
                minimal_locs_df,
                minimal_info,
                locs_per_segment=100,
                max_drift_nm=200,
            )

    def test_structured_array_input_accepted_before_gpu_check(
        self, minimal_locs_structured, minimal_info
    ):
        """Numpy structured arrays should be converted to DataFrame before the
        CUDA check, so the error is still a RuntimeError (not an AttributeError
        from missing .columns)."""
        if comet._CUDA_AVAILABLE:
            pytest.skip(
                "CUDA is present; testing no-GPU path only on CPU machines"
            )
        with pytest.raises(RuntimeError, match="CUDA"):
            comet.comet(
                minimal_locs_structured,
                minimal_info,
                locs_per_segment=100,
                max_drift_nm=200,
            )

    def test_missing_columns_raises_key_error(self, minimal_info):
        """Locs missing required columns should raise KeyError, not crash on .columns."""
        if comet._CUDA_AVAILABLE:
            pytest.skip(
                "CUDA is present; GPU path reached before column check"
            )
        bad_locs = pd.DataFrame({"frame": np.arange(10), "x": np.zeros(10)})
        # No 'y' column — should raise KeyError from the column check
        with pytest.raises((KeyError, RuntimeError)):
            comet.comet(
                bad_locs, minimal_info, locs_per_segment=5, max_drift_nm=100
            )

    def test_z_column_accepted(self, minimal_locs_df, minimal_info):
        """3D input (with z) should not crash before the GPU check."""
        if comet._CUDA_AVAILABLE:
            pytest.skip(
                "CUDA is present; testing no-GPU path only on CPU machines"
            )
        df = minimal_locs_df.copy()
        df["z"] = np.zeros(len(df), dtype=np.float32)
        with pytest.raises(RuntimeError, match="CUDA"):
            comet.comet(
                df, minimal_info, locs_per_segment=100, max_drift_nm=200
            )

    def test_missing_pixelsize_raises_on_gpu(self, minimal_locs_df):
        """info without 'Pixelsize' should fail with an informative error.
        Only meaningful on GPU since the CUDA check fires first on CPU."""
        if not comet._CUDA_AVAILABLE:
            pytest.skip("CUDA check fires before pixelsize check on CPU")
        bad_info = [{"Frames": 200}]
        with pytest.raises((KeyError, RuntimeError, ValueError)):
            comet.comet(
                minimal_locs_df,
                bad_info,
                locs_per_segment=100,
                max_drift_nm=200,
            )


# ---------------------------------------------------------------------------
# comet() public API — full pipeline (GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not comet._CUDA_AVAILABLE,
    reason="CUDA GPU not available",
)
class TestCometFullPipeline:
    """Integration tests that run the full optimisation loop on GPU."""

    def test_output_shapes(self, minimal_locs_df, minimal_info):
        undrifted, new_info, drift = comet.comet(
            minimal_locs_df,
            minimal_info,
            locs_per_segment=100,
            max_drift_nm=300,
            target_sigma_nm=50,
        )
        assert len(undrifted) == len(minimal_locs_df)
        assert set(undrifted.columns) >= {"x", "y", "frame"}
        assert "x" in drift.columns and "y" in drift.columns

    def test_drift_covers_frame_range(self, minimal_locs_df, minimal_info):
        frame0 = minimal_locs_df["frame"] - minimal_locs_df["frame"].min()
        _, _, drift = comet.comet(
            minimal_locs_df,
            minimal_info,
            locs_per_segment=100,
            max_drift_nm=300,
            target_sigma_nm=50,
        )
        assert len(drift) == int(frame0.max()) + 1

    def test_new_info_appended(self, minimal_locs_df, minimal_info):
        _, new_info, _ = comet.comet(
            minimal_locs_df,
            minimal_info,
            locs_per_segment=100,
            max_drift_nm=300,
            target_sigma_nm=50,
        )
        assert len(new_info) == len(minimal_info) + 1
        assert "COMET" in new_info[-1]["Generated by"]

    def test_structured_array_gives_same_result_as_dataframe(
        self, minimal_locs_df, minimal_locs_structured, minimal_info
    ):
        locs_df, _, drift_df = comet.comet(
            minimal_locs_df,
            minimal_info,
            locs_per_segment=100,
            max_drift_nm=300,
            target_sigma_nm=50,
        )
        locs_arr, _, drift_arr = comet.comet(
            minimal_locs_structured,
            minimal_info,
            locs_per_segment=100,
            max_drift_nm=300,
            target_sigma_nm=50,
        )
        np.testing.assert_allclose(
            drift_df["x"].to_numpy(), drift_arr["x"].to_numpy(), rtol=1e-5
        )
