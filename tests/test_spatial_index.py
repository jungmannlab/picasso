"""Tests for ``picasso.spatial_index``.

Correctness goal: ``query_viewport`` must return a superset of the
strictly-inside locs (``x_min < x < x_max`` and same for y), and
``picasso.render`` invoked on the pyramid-filtered subset must produce
the same image as on the full locs DataFrame for every blur method.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import render, spatial_index


def _make_locs(
    n: int, width: float, height: float, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, width, size=n),
            "y": rng.uniform(0.0, height, size=n),
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


def _info(width: float, height: float, n_frames: int = 1000) -> list[dict]:
    return [
        {
            "Width": width,
            "Height": height,
            "Frames": n_frames,
            "Pixelsize": 130.0,
        }
    ]


def _brute_force_in_view(locs: pd.DataFrame, viewport) -> np.ndarray:
    (y_min, x_min), (y_max, x_max) = viewport
    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()
    mask = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    return np.nonzero(mask)[0].astype(np.uint32)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_empty_locs_returns_pyramid(self):
        locs = _make_locs(0, 512, 512)
        pyr = spatial_index.build_render_index(locs, _info(512, 512))
        assert pyr is not None
        assert pyr.perm.shape == (0,)
        # Sub-threshold viewport hits the gather path and returns empty.
        idx = spatial_index.query_viewport(pyr, ((0, 0), (10, 10)))
        assert idx is not None and idx.shape == (0,)

    def test_missing_metadata_returns_none(self):
        locs = _make_locs(100, 512, 512)
        assert spatial_index.build_render_index(locs, [{}]) is None

    def test_perm_is_a_permutation(self):
        n = 10_000
        locs = _make_locs(n, 512, 512)
        pyr = spatial_index.build_render_index(locs, _info(512, 512))
        assert pyr.perm.shape == (n,)
        assert np.array_equal(np.sort(pyr.perm), np.arange(n, dtype=np.uint32))

    def test_levels_partition_total_count(self):
        locs = _make_locs(5_000, 576, 576)
        pyr = spatial_index.build_render_index(locs, _info(576, 576))
        for bs, be in zip(pyr.block_starts, pyr.block_ends):
            # Every loc belongs to exactly one block at each level.
            assert int((be - bs).sum()) == len(locs)
            # Block ranges don't overlap and don't go past N.
            assert (be >= bs).all()
            assert int(be.max()) <= len(locs)

    def test_block_sizes_geometric(self):
        pyr = spatial_index.build_render_index(
            _make_locs(100, 512, 512), _info(512, 512)
        )
        assert len(pyr.block_sizes) == 3
        # 4x ratio between successive levels.
        assert pyr.block_sizes[1] == pytest.approx(4 * pyr.block_sizes[0])
        assert pyr.block_sizes[2] == pytest.approx(4 * pyr.block_sizes[1])


# ---------------------------------------------------------------------------
# Query correctness vs brute force
# ---------------------------------------------------------------------------


class TestQuery:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_query_superset_of_strict_in_view(self, seed):
        rng = np.random.default_rng(seed)
        W, H = 512.0, 512.0
        locs = _make_locs(20_000, W, H, seed=seed)
        pyr = spatial_index.build_render_index(locs, _info(W, H))

        # Random viewport inside the FOV, kept well under the bypass
        # coverage ratio so this test exercises the gather path.
        cx, cy = rng.uniform(50, W - 50), rng.uniform(50, H - 50)
        half_w, half_h = rng.uniform(5, 100), rng.uniform(5, 100)
        viewport = (
            (cy - half_h, cx - half_w),
            (cy + half_h, cx + half_w),
        )

        idx = spatial_index.query_viewport(pyr, viewport)
        truth = _brute_force_in_view(locs, viewport)
        # Superset: every strictly-in-view loc must be returned.
        assert idx is not None
        assert set(int(i) for i in truth).issubset(set(int(i) for i in idx))

    def test_viewport_covering_full_fov_returns_none(self):
        # Above the coverage bypass threshold the pyramid returns None
        # so the caller renders the full locs DataFrame without an
        # iloc copy -- this is what restores the pre-pyramid full-FOV
        # render speed for large N.
        W, H = 512.0, 512.0
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        assert spatial_index.query_viewport(pyr, ((0, 0), (H, W))) is None

    def test_viewport_above_bypass_threshold_returns_none(self):
        # Half the FOV area in each dimension -> 25% coverage, below
        # the default 0.5 threshold, so we still get an array. Doubling
        # one dimension to span the full axis brings coverage to 50%
        # and triggers the bypass.
        W, H = 512.0, 512.0
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        sub = spatial_index.query_viewport(pyr, ((0, 0), (H / 2, W / 2)))
        assert sub is not None and sub.shape[0] > 0
        assert spatial_index.query_viewport(pyr, ((0, 0), (H, W / 2))) is None

    def test_viewport_with_negative_bounds_enclosing_fov_returns_none(self):
        # Zoomed/panned out so the viewport extends past every FOV edge
        # (loc coords are always positive but the viewport isn't).
        # All locs are in view -> bypass via the full-enclose check.
        W, H = 512.0, 512.0
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        assert (
            spatial_index.query_viewport(
                pyr, ((-100.0, -200.0), (H + 50.0, W + 300.0))
            )
            is None
        )

    def test_viewport_overhanging_right_bottom_clips_correctly(self):
        # Locs are bounded in (0, W) x (0, H) but viewports aren't --
        # users can pan/zoom past the bottom-right. The clipped area
        # must drive the bypass decision, not the raw viewport extent.
        W, H = 512.0, 512.0
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        # ((100, 100), (700, 700)) overhangs by 188 on each side.
        # Clipped intersection: 412x412 -> ~65% of FOV -> bypass.
        assert (
            spatial_index.query_viewport(pyr, ((100.0, 100.0), (700.0, 700.0)))
            is None
        )
        # Thin overhanging strip: 412x50 -> ~7.9% of FOV -> gather.
        idx = spatial_index.query_viewport(
            pyr, ((100.0, 100.0), (150.0, 700.0))
        )
        assert idx is not None
        truth = _brute_force_in_view(locs, ((100.0, 100.0), (150.0, 700.0)))
        assert set(int(i) for i in truth).issubset(set(int(i) for i in idx))

    def test_viewport_partially_negative_below_threshold_returns_array(self):
        # Viewport overhangs the FOV on one side only; the FOV-clipped
        # area is well under the bypass threshold, so the gather path
        # still runs and returns a valid array.
        W, H = 512.0, 512.0
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        idx = spatial_index.query_viewport(
            pyr, ((-100.0, -100.0), (100.0, 100.0))
        )
        assert idx is not None
        truth = _brute_force_in_view(locs, ((-100.0, -100.0), (100.0, 100.0)))
        assert set(int(i) for i in truth).issubset(set(int(i) for i in idx))

    def test_viewport_outside_fov_returns_empty(self):
        locs = _make_locs(1000, 512, 512)
        pyr = spatial_index.build_render_index(locs, _info(512, 512))
        # Far outside the FOV in both directions.
        idx = spatial_index.query_viewport(pyr, ((2000, 2000), (3000, 3000)))
        assert idx is not None and idx.shape == (0,)

    def test_tiny_zoomed_viewport_returns_few_locs(self):
        W, H = 512.0, 512.0
        n = 50_000
        locs = _make_locs(n, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        viewport = ((100.0, 100.0), (105.0, 105.0))  # 5x5 patch
        idx = spatial_index.query_viewport(pyr, viewport)
        # Expected ~ n * (5*5) / (W*H) = ~50_000 * 25 / 262144 ~= 4-5 locs
        # plus some overspill from the chosen block size. Cap at a
        # generous but tight bound: still tiny fraction of n.
        assert len(idx) < 200
        # And every strict-inside loc must be present.
        truth = _brute_force_in_view(locs, viewport)
        assert set(int(i) for i in truth).issubset(set(int(i) for i in idx))


# ---------------------------------------------------------------------------
# Renderer parity: pre-filtering by pyramid must not change the image
# ---------------------------------------------------------------------------


class TestRendererParity:
    @pytest.fixture(scope="class")
    def locs_pyr(self):
        W, H = 512.0, 512.0
        n = 30_000
        locs = _make_locs(n, W, H, seed=42)
        info = _info(W, H)
        pyr = spatial_index.build_render_index(locs, info)
        return locs, info, pyr

    @pytest.mark.parametrize(
        "blur_method", [None, "gaussian", "gaussian_iso", "smooth", "convolve"]
    )
    def test_parity_with_full_locs(self, locs_pyr, blur_method):
        locs, info, pyr = locs_pyr
        # Slightly off-center, asymmetric viewport so the chosen pyramid
        # level isn't trivially the coarsest.
        viewport = ((40.0, 60.0), (180.0, 240.0))
        oversampling = 4.0

        idx = spatial_index.query_viewport(pyr, viewport)
        filtered = locs.iloc[idx]

        n_full, img_full = render.render(
            locs,
            info,
            oversampling=oversampling,
            viewport=viewport,
            blur_method=blur_method,
        )
        n_filt, img_filt = render.render(
            filtered,
            info,
            oversampling=oversampling,
            viewport=viewport,
            blur_method=blur_method,
        )

        assert n_full == n_filt
        # ``gaussian``/``gaussian_iso`` use parallel summation across
        # locs; reordering the input via the pyramid query changes the
        # summation order and introduces float32 round-off well below
        # any visual threshold. Histogram modes are exact.
        if blur_method in (None, "smooth", "convolve"):
            np.testing.assert_array_equal(img_full, img_filt)
        else:
            np.testing.assert_allclose(
                img_full, img_filt, rtol=1e-5, atol=1e-6
            )
