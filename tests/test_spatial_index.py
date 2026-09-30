"""Tests for ``picasso.spatial_index``.

Correctness goal: ``query_viewport`` must return a superset of the
strictly-inside locs (``x_min < x < x_max`` and same for y), and
``picasso.render`` invoked on the pyramid-filtered subset must produce
the same image as on the full locs DataFrame for every blur method.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import h5py
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
# Circular picks
# ---------------------------------------------------------------------------


class TestQueryCircle:
    @pytest.mark.parametrize("radius", [0.4, 3.0, 25.0])
    def test_matches_brute_force(self, radius):
        locs = _make_locs(20_000, 128.0, 96.0, seed=3)
        pyramid = spatial_index.build_render_index(locs, _info(128.0, 96.0))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        rng = np.random.default_rng(4)
        for cx, cy in zip(rng.uniform(-5, 133, 30), rng.uniform(-5, 101, 30)):
            got = spatial_index.query_circle(pyramid, x, y, cx, cy, radius)
            expected = np.nonzero((x - cx) ** 2 + (y - cy) ** 2 < radius**2)[0]
            assert np.array_equal(np.sort(got), expected)

    def test_matches_index_block_picking(self):
        from picasso import postprocess

        width, height = 64.0, 64.0
        locs = _make_locs(5_000, width, height, seed=5)
        info = _info(width, height)
        pyramid = spatial_index.build_render_index(locs, info)
        picks = [(10.0, 12.0), (40.5, 33.2), (63.9, 0.1)]
        radius = 2.5
        via_blocks = postprocess.picked_locs(
            locs, info, picks, "Circle", pick_size=radius
        )
        via_pyramid = postprocess.picked_locs(
            locs, info, picks, "Circle", pick_size=radius, index_blocks=pyramid
        )
        assert len(via_blocks) == len(via_pyramid) == len(picks)
        for a, b in zip(via_blocks, via_pyramid):
            assert len(a) > 0
            # the index-block path sanitizes (and so standardizes the
            # dtypes of) its input; the rows are what is compared
            pd.testing.assert_frame_equal(
                a.sort_index(),
                b.sort_index(),
                check_like=True,
                check_dtype=False,
            )

    def test_empty_pyramid_and_far_away_pick(self):
        locs = _make_locs(1_000, 32.0, 32.0)
        pyramid = spatial_index.build_render_index(locs, _info(32.0, 32.0))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        assert (
            len(spatial_index.query_circle(pyramid, x, y, 100.0, 100.0, 3.0))
            == 0
        )
        empty = spatial_index.build_render_index(
            locs.iloc[:0], _info(32.0, 32.0)
        )
        if empty is not None:
            assert (
                len(spatial_index.query_circle(empty, x[:0], y[:0], 1, 1, 3))
                == 0
            )


# ---------------------------------------------------------------------------
# Quad-tree layout
# ---------------------------------------------------------------------------


def _clustered_locs(n=2000, width=64.0, height=64.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [rng.uniform(0, width, n // 2), rng.normal(20, 1.5, n // 2)]
    )
    y = np.concatenate(
        [rng.uniform(0, height, n // 2), rng.normal(40, 1.5, n // 2)]
    )
    keep = (x > 0) & (x < width) & (y > 0) & (y < height)
    return x[keep].astype(np.float32), y[keep].astype(np.float32)


# ---------------------------------------------------------------------------
# Exact counts in a rectangle
# ---------------------------------------------------------------------------


class TestCountRect:
    def test_matches_brute_force(self):
        # locs also beyond the FOV (held by the border blocks) and on a
        # block boundary; viewports from sub-pixel to beyond the FOV
        rng = np.random.default_rng(3)
        n, width = 50_000, 256.0
        locs = pd.DataFrame(
            {
                "x": rng.uniform(-2.0, width + 2.0, n).astype(np.float32),
                "y": rng.uniform(-2.0, width + 2.0, n).astype(np.float32),
            }
        )
        locs.loc[:99, "x"] = np.float32(100.0)
        pyramid = spatial_index.build_render_index(locs, _info(width, width))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        for _ in range(500):
            cx, cy = rng.uniform(-20.0, width + 20.0, 2)
            half = 10 ** rng.uniform(-2.0, 2.5) / 2
            viewport = ((cy - half, cx - half), (cy + half, cx + half))
            expected = len(_brute_force_in_view(locs, viewport))
            assert spatial_index.count_rect(pyramid, x, y, viewport) == (
                expected
            )

    def test_zoomed_in_count_excludes_edge_blocks(self):
        # the pyramid query returns whole edge blocks; the count must not
        locs = _make_locs(200_000, 64.0, 64.0)
        pyramid = spatial_index.build_render_index(locs, _info(64.0, 64.0))
        viewport = ((20.3, 20.3), (20.8, 20.8))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        expected = len(_brute_force_in_view(locs, viewport))
        assert len(spatial_index.query_viewport(pyramid, viewport)) > expected
        assert spatial_index.count_rect(pyramid, x, y, viewport) == expected

    def test_empty(self):
        locs = _make_locs(0, 64.0, 64.0)
        pyramid = spatial_index.build_render_index(locs, _info(64.0, 64.0))
        empty = np.empty(0, dtype=np.float32)
        viewport = ((0.0, 0.0), (10.0, 10.0))
        assert spatial_index.count_rect(pyramid, empty, empty, viewport) == 0


class TestQuadTreeLayout:
    W = H = 64.0

    def test_keys_are_sorted_and_refine_the_blocks(self):
        x, y = _clustered_locs()
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        keys = pyramid.sorted_keys
        assert keys is not None and len(keys) == len(x)
        assert np.all(np.diff(keys.astype(np.int64)) >= 0)
        # the base-level block tables still describe the permutation
        info = _info(self.W, self.H)
        locs = pd.DataFrame({"x": x, "y": y})
        assert spatial_index.validate_render_index(pyramid, locs, info)
        assert pyramid.root_bits == 6 and pyramid.fine_bits == 8

    def test_layout_contract(self):
        x, y = _clustered_locs()
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        keys, perm, root_px, total_bits = spatial_index.quadtree_layout(
            pyramid
        )
        assert root_px == pyramid.block_sizes[0] * 2**pyramid.root_bits
        assert total_bits == pyramid.root_bits + pyramid.fine_bits
        # every node's key range holds exactly the rows inside its square
        rng = np.random.default_rng(1)
        for _ in range(50):
            d = int(rng.integers(1, 8))
            ix, iy = (int(v) for v in rng.integers(0, 2**d, 2))
            # the prefix of (ix, iy): interleave d bits, x in the even bits
            prefix = 0
            for bit in range(d):
                prefix |= ((ix >> bit) & 1) << (2 * bit)
                prefix |= ((iy >> bit) & 1) << (2 * bit + 1)
            shift = 2 * (total_bits - d)
            lo = np.searchsorted(keys, np.uint64(prefix << shift))
            hi = np.searchsorted(keys, np.uint64((prefix + 1) << shift))
            side = root_px / 2**d
            inside = (
                (x >= ix * side)
                & (x < (ix + 1) * side)
                & (y >= iy * side)
                & (y < (iy + 1) * side)
            )
            assert np.array_equal(np.sort(perm[lo:hi]), np.nonzero(inside)[0])

    def test_pyramid_without_keys_is_refused(self):
        x, y = _clustered_locs()
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        pyramid.sorted_keys = None
        with pytest.raises(ValueError):
            spatial_index.quadtree_layout(pyramid)


# ---------------------------------------------------------------------------
# Persistence in the localizations file
# ---------------------------------------------------------------------------


class TestPersistence:
    W, H = 96.0, 64.0

    def _file(self, tmp_path, n=3_000, **save_kwargs):
        from picasso import io

        locs = _make_locs(n, self.W, self.H, seed=7)
        info = _info(self.W, self.H)
        path = str(tmp_path / "locs.hdf5")
        io.save_locs(path, locs, info, **save_kwargs)
        return path, locs, info

    def test_round_trip_is_the_built_pyramid(self, tmp_path):
        path, locs, info = self._file(tmp_path, render_index=True)
        with h5py.File(path, "r") as f:
            assert spatial_index.RENDER_INDEX_GROUP in f
            assert f["locs"].dtype["x"] == np.float32
        stored = spatial_index.read_render_index(path)
        built = spatial_index.build_render_index(locs, info)
        assert np.array_equal(stored.perm, built.perm)
        assert stored.block_sizes == built.block_sizes
        for a, b in zip(stored.block_starts, built.block_starts):
            assert np.array_equal(a, b)
        for a, b in zip(stored.block_ends, built.block_ends):
            assert np.array_equal(a, b)
        assert (stored.width, stored.height) == (self.W, self.H)
        assert (stored.root_bits, stored.fine_bits) == (
            built.root_bits,
            built.fine_bits,
        )
        assert stored.sorted_keys is None  # unchecked: no keys yet
        assert spatial_index.validate_render_index(stored, locs, info)
        assert np.array_equal(stored.sorted_keys, built.sorted_keys)
        loaded = spatial_index.load_render_index(path, locs, info)
        assert loaded is not None
        assert loaded.sorted_keys is not None
        # and it queries like the built one
        vp = ((10.0, 10.0), (30.0, 40.0))
        assert np.array_equal(
            np.sort(spatial_index.query_viewport(loaded, vp)),
            np.sort(spatial_index.query_viewport(built, vp)),
        )

    def test_auto_skips_small_files_and_keeps_large_ones(
        self, tmp_path, monkeypatch
    ):
        path, _, _ = self._file(tmp_path)  # "auto", 3k < threshold
        assert spatial_index.read_render_index(path) is None
        monkeypatch.setattr(spatial_index, "PERSIST_MIN_LOCS", 1_000)
        path, locs, info = self._file(tmp_path)
        assert spatial_index.load_render_index(path, locs, info) is not None
        path, _, _ = self._file(tmp_path, render_index=False)
        assert spatial_index.read_render_index(path) is None

    def test_a_given_pyramid_is_stored_as_is(self, tmp_path):
        from picasso import io

        locs = _make_locs(500, self.W, self.H)
        info = _info(self.W, self.H)
        pyramid = spatial_index.build_render_index(locs, info)
        path = str(tmp_path / "given.hdf5")
        io.save_locs(path, locs, info, render_index=pyramid)
        assert np.array_equal(
            spatial_index.read_render_index(path).perm, pyramid.perm
        )
        with pytest.raises(ValueError):
            io.save_locs(path, locs, info, render_index="always")

    def test_files_without_the_group_load_as_before(self, tmp_path):
        path, locs, info = self._file(tmp_path, render_index=False)
        assert spatial_index.load_render_index(path, locs, info) is None

    @pytest.mark.parametrize(
        "edit",
        ["move_one", "drop_one", "append_one", "reorder", "other_fov"],
    )
    def test_edits_outside_picasso_invalidate_the_index(self, tmp_path, edit):
        path, locs, info = self._file(tmp_path, render_index=True)
        stored = spatial_index.read_render_index(path)
        edited = locs.copy()
        if edit == "move_one":
            edited.loc[edited.index[123], "x"] = self.W - 0.5  # other block
        elif edit == "drop_one":
            edited = edited.drop(edited.index[10]).reset_index(drop=True)
        elif edit == "append_one":
            edited = pd.concat([edited, edited.iloc[:1]], ignore_index=True)
        elif edit == "reorder":
            edited = edited.iloc[::-1].reset_index(drop=True)
        elif edit == "other_fov":
            info = _info(self.W * 2, self.H)
        assert not spatial_index.validate_render_index(stored, edited, info)
        # the loader falls back to building
        # (the file on disk is untouched here; simulate by writing the
        # edited rows without an index via a plain h5py write)
        with h5py.File(path, "r+") as f:
            del f["locs"]
            f.create_dataset("locs", data=edited.to_records(index=False))
        assert spatial_index.load_render_index(path, edited, info) is None

    def test_edits_that_keep_the_index_correct_pass(self, tmp_path):
        path, locs, info = self._file(tmp_path, render_index=True)
        stored = spatial_index.read_render_index(path)
        edited = locs.copy()
        edited["photons"] *= 2.0  # another column
        assert spatial_index.validate_render_index(stored, edited, info)
        # a coordinate moved within its finest block
        base = stored.block_sizes[0]
        i = edited.index[5]
        edited.loc[i, "x"] = np.floor(edited.loc[i, "x"] / base) * base + 0.01
        assert spatial_index.validate_render_index(stored, edited, info)

    def test_unknown_version_is_ignored(self, tmp_path):
        path, locs, info = self._file(tmp_path, render_index=True)
        with h5py.File(path, "r+") as f:
            f[spatial_index.RENDER_INDEX_GROUP].attrs["version"] = 99
        assert spatial_index.read_render_index(path) is None

    def test_empty_file(self, tmp_path):
        path, locs, info = self._file(tmp_path, n=0, render_index=True)
        stored = spatial_index.read_render_index(path)
        assert stored is not None and stored.perm.shape == (0,)
        assert spatial_index.validate_render_index(stored, locs, info)


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
        # Coverage well below the bypass threshold still returns an
        # array; coverage well above triggers the bypass. Sized off the
        # module constant so the test tracks any future re-tuning.
        W, H = 512.0, 512.0
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        ratio = spatial_index._BYPASS_COVERAGE_RATIO
        below = float(np.sqrt(ratio * 0.5)) * W
        sub = spatial_index.query_viewport(pyr, ((0.0, 0.0), (below, below)))
        assert sub is not None and sub.shape[0] > 0
        above = float(np.sqrt(min(1.0, ratio * 2.0))) * W
        assert (
            spatial_index.query_viewport(pyr, ((0.0, 0.0), (above, above)))
            is None
        )

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

        idx = spatial_index.query_viewport(pyr, viewport)
        filtered = locs.iloc[idx]

        # 'convolve' blurs with the channel's global precision, which
        # the GUI computes once over the whole channel and passes with
        # every request; without it the median of the rows given
        # would differ between the two calls
        global_precision = None
        if blur_method == "convolve":
            global_precision = (
                float(np.median(locs["lpx"])),
                float(np.median(locs["lpy"])),
            )
        n_full, img_full = render.render(
            locs,
            info,
            disp_px_size=30,
            viewport=viewport,
            blur_method=blur_method,
            global_precision=global_precision,
        )
        n_filt, img_filt = render.render(
            filtered,
            info,
            disp_px_size=30,
            viewport=viewport,
            blur_method=blur_method,
            global_precision=global_precision,
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
