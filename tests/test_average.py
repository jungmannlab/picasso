"""Test picasso.average functions for particle averaging.

:author: AI Assistant, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pandas as pd
import pytest

from picasso import average, render


class TestRenderHist:
    """Tests for render_hist function."""

    def test_render_hist_basic(self):
        """Test basic histogram rendering with simple coordinates."""
        x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        oversampling = 1.0
        t_min = 0.0
        t_max = 3.0

        n, image = render.render_hist_numba(x, y, oversampling, t_min, t_max)

        assert n == 3, "Should render 3 localizations"
        assert image.shape == (3, 3), "Image should be 3x3 pixels"
        assert image.dtype == np.float32, "Image should be float32"
        assert np.sum(image) > 0, "Image should have non-zero values"

    def test_render_hist_out_of_bounds(self):
        """Test that out-of-bounds points are excluded."""
        x = np.array([0.5, 5.5], dtype=np.float32)
        y = np.array([0.5, 5.5], dtype=np.float32)
        oversampling = 1.0
        t_min = 0.0
        t_max = 3.0

        n, image = render.render_hist_numba(x, y, oversampling, t_min, t_max)

        assert n == 1, "Should only render 1 in-bounds localization"

    def test_render_hist_oversampling(self):
        """Test that oversampling increases image size."""
        x = np.array([0.5], dtype=np.float32)
        y = np.array([0.5], dtype=np.float32)
        t_min = 0.0
        t_max = 1.0

        n1, image1 = render.render_hist_numba(x, y, 1.0, t_min, t_max)
        n2, image2 = render.render_hist_numba(x, y, 2.0, t_min, t_max)

        assert image1.shape == (1, 1)
        assert image2.shape == (2, 2), "2x oversampling should give 2x2 image"

    def test_render_hist_empty(self):
        """Test rendering with empty input."""
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        oversampling = 1.0
        t_min = 0.0
        t_max = 3.0

        n, image = render.render_hist_numba(x, y, oversampling, t_min, t_max)

        assert n == 0, "Should render 0 localizations"
        assert np.all(image == 0), "Image should be all zeros"


class TestComputeXcorr:
    """Tests for compute_xcorr function."""

    def test_compute_xcorr_identical_images(self):
        """Test cross-correlation of identical images."""
        image = np.ones((5, 5), dtype=np.float32)
        CF_image = np.conj(np.fft.fft2(image))

        xcorr = average.compute_xcorr(CF_image, image)

        # Maximum should be at center
        center_y, center_x = np.array(xcorr.shape) // 2
        assert xcorr[center_y, center_x] == np.max(xcorr)

    def test_compute_xcorr_orthogonal_images(self):
        """Test cross-correlation of orthogonal patterns."""
        # Create two simple patterns
        image1 = np.zeros((5, 5), dtype=np.float32)
        image1[2, :] = 1  # horizontal line

        image2 = np.zeros((5, 5), dtype=np.float32)
        image2[:, 2] = 1  # vertical line

        CF_image1 = np.conj(np.fft.fft2(image1))
        xcorr = average.compute_xcorr(CF_image1, image2)

        assert xcorr.shape == (5, 5)
        assert np.isfinite(xcorr).all()

    def test_compute_xcorr_shape(self):
        """Test that output shape matches input."""
        for shape in [(3, 3), (5, 5), (7, 7)]:
            image = np.random.rand(*shape).astype(np.float32)
            CF_image = np.conj(np.fft.fft2(image))

            xcorr = average.compute_xcorr(CF_image, image)

            assert xcorr.shape == shape


class TestAlignGroupCore:
    """Tests for align_group_core function."""

    def test_align_group_core_no_rotation(self):
        """Test alignment with no rotation needed."""
        # Create simple data: 3 points that shouldn't need rotation
        index = np.array([0, 1, 2])
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        angles = np.array([0.0], dtype=np.float32)
        oversampling = 1.0
        t_min = -2.0
        t_max = 2.0

        # Render average image for correlation
        n, image_avg = render.render_hist_numba(
            x, y, oversampling, t_min, t_max
        )
        CF_image_avg = np.conj(np.fft.fft2(image_avg))
        image_half = image_avg.shape[0] / 2

        x_aligned, y_aligned = average.align_group_core(
            index,
            x.copy(),
            y.copy(),
            angles,
            oversampling,
            t_min,
            t_max,
            CF_image_avg,
            image_half,
        )

        assert x_aligned.shape == (3,)
        assert y_aligned.shape == (3,)
        assert np.isfinite(x_aligned).all()
        assert np.isfinite(y_aligned).all()

    def test_align_group_core_subset(self):
        """Test alignment with subset of points."""
        index = np.array([0, 2])  # Only points 0 and 2
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        angles = np.array([0.0], dtype=np.float32)
        oversampling = 1.0
        t_min = -2.0
        t_max = 2.0

        n, image_avg = render.render_hist_numba(
            x, y, oversampling, t_min, t_max
        )
        CF_image_avg = np.conj(np.fft.fft2(image_avg))
        image_half = image_avg.shape[0] / 2

        x_aligned, y_aligned = average.align_group_core(
            index,
            x.copy(),
            y.copy(),
            angles,
            oversampling,
            t_min,
            t_max,
            CF_image_avg,
            image_half,
        )

        # Should return aligned coordinates for the subset
        assert x_aligned.shape == (2,)
        assert y_aligned.shape == (2,)


class TestAverageParticles:
    """Tests for average function."""

    @pytest.fixture
    def sample_locs(self):
        """Create sample localizations for testing."""
        np.random.seed(42)
        n_locs = 100
        locs = pd.DataFrame(
            {
                "x": np.random.randn(n_locs) * 0.5,
                "y": np.random.randn(n_locs) * 0.5,
                "group": np.repeat(
                    np.arange(10), 10
                ),  # 10 groups of 10 locs each
            }
        )
        return locs

    @pytest.fixture
    def sample_info(self):
        """Create sample metadata."""
        return [{"Width": 256, "Height": 256}]

    def test_average_particles_basic(self, sample_locs, sample_info):
        """Test basic averaging operation."""
        result = average.average(
            sample_locs,
            sample_info,
            oversampling=1.0,
            iterations=1,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_locs)
        assert "x" in result.columns
        assert "y" in result.columns

    def test_average_particles_coordinates_centered(
        self, sample_locs, sample_info
    ):
        """Test that averaged coordinates are centered."""
        result = average.average(
            sample_locs,
            sample_info,
            oversampling=1.0,
            iterations=1,
        )

        # After averaging, coordinates should be close to origin
        assert np.abs(np.mean(result["x"])) < 0.1
        assert np.abs(np.mean(result["y"])) < 0.1

    def test_average_particles_multiple_iterations(
        self, sample_locs, sample_info
    ):
        """Test averaging with multiple iterations."""
        result1 = average.average(
            sample_locs.copy(),
            sample_info,
            oversampling=1.0,
            iterations=1,
        )
        result2 = average.average(
            sample_locs.copy(),
            sample_info,
            oversampling=1.0,
            iterations=2,
        )

        # Results should be different (more iterations = more averaging)
        # but both should have similar structure
        assert len(result1) == len(result2)

    def test_average_particles_with_callback(self, sample_locs, sample_info):
        """Test averaging with progress callback."""
        progress_calls = []

        def progress_callback(it, total_it, locs):
            progress_calls.append((it, total_it))

        average.average(
            sample_locs,
            sample_info,
            oversampling=1.0,
            iterations=2,
            progress_callback=progress_callback,
        )

        # Should be called at least once per iteration
        assert len(progress_calls) >= 2

    def test_average_particles_preserves_group_column(
        self, sample_locs, sample_info
    ):
        """Test that group column is preserved if it exists."""
        result = average.average(
            sample_locs,
            sample_info,
            oversampling=1.0,
            iterations=1,
        )

        # The 'group' column should still be present (not removed)
        if "group" in sample_locs.columns:
            assert "group" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
