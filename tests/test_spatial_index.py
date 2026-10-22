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


#: camera pixel size (nm); the fields of view below are given in nm,
#: as multiples of it, so that the metadata holds whole pixel counts
PS = 130.0


def _make_locs(
    n: int, width: float, height: float, seed: int = 0
) -> pd.DataFrame:
    """Uniform localizations on a ``width`` x ``height`` nm field."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, width, size=n),
            "y": rng.uniform(0.0, height, size=n),
            "lpx": rng.uniform(5.0, 40.0, size=n),
            "lpy": rng.uniform(5.0, 40.0, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


def _info(width: float, height: float, n_frames: int = 1000) -> list[dict]:
    """Metadata of a ``width`` x ``height`` nm field of view."""
    return [
        {
            "Width": int(round(width / PS)),
            "Height": int(round(height / PS)),
            "Frames": n_frames,
            "Pixelsize": PS,
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
    @pytest.mark.parametrize("radius", [0.4 * PS, 3.0 * PS, 25.0 * PS])
    def test_matches_brute_force(self, radius):
        W, H = 128.0 * PS, 96.0 * PS
        locs = _make_locs(20_000, W, H, seed=3)
        pyramid = spatial_index.build_render_index(locs, _info(W, H))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        rng = np.random.default_rng(4)
        for cx, cy in zip(
            rng.uniform(-5 * PS, W + 5 * PS, 30),
            rng.uniform(-5 * PS, H + 5 * PS, 30),
        ):
            got = spatial_index.query_circle(pyramid, x, y, cx, cy, radius)
            expected = np.nonzero((x - cx) ** 2 + (y - cy) ** 2 < radius**2)[0]
            assert np.array_equal(np.sort(got), expected)

    def test_matches_index_block_picking(self):
        from picasso import postprocess

        width, height = 64.0 * PS, 64.0 * PS
        locs = _make_locs(5_000, width, height, seed=5)
        info = _info(width, height)
        pyramid = spatial_index.build_render_index(locs, info)
        picks = [
            (10.0 * PS, 12.0 * PS),
            (40.5 * PS, 33.2 * PS),
            (63.9 * PS, 0.1 * PS),
        ]
        diameter = 5.0 * PS
        via_blocks = postprocess.picked_locs(
            locs, info, picks=picks, pick_shape="Circle", pick_size=diameter
        )
        via_pyramid = postprocess.picked_locs(
            locs,
            info,
            picks=picks,
            pick_shape="Circle",
            pick_size=diameter,
            index_blocks=pyramid,
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
        W = 32.0 * PS
        locs = _make_locs(1_000, W, W)
        pyramid = spatial_index.build_render_index(locs, _info(W, W))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        assert (
            len(
                spatial_index.query_circle(pyramid, x, y, 3 * W, 3 * W, 3 * PS)
            )
            == 0
        )
        empty = spatial_index.build_render_index(locs.iloc[:0], _info(W, W))
        if empty is not None:
            assert (
                len(spatial_index.query_circle(empty, x[:0], y[:0], 1, 1, 3))
                == 0
            )


# ---------------------------------------------------------------------------
# Quad-tree layout
# ---------------------------------------------------------------------------


def _clustered_locs(n=2000, width=64.0 * PS, height=64.0 * PS, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [rng.uniform(0, width, n // 2), rng.normal(20 * PS, 1.5 * PS, n // 2)]
    )
    y = np.concatenate(
        [rng.uniform(0, height, n // 2), rng.normal(40 * PS, 1.5 * PS, n // 2)]
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
        n, width = 50_000, 256.0 * PS
        locs = pd.DataFrame(
            {
                "x": rng.uniform(-2.0 * PS, width + 2.0 * PS, n).astype(
                    np.float32
                ),
                "y": rng.uniform(-2.0 * PS, width + 2.0 * PS, n).astype(
                    np.float32
                ),
            }
        )
        locs.loc[:99, "x"] = np.float32(100.0 * PS)
        pyramid = spatial_index.build_render_index(locs, _info(width, width))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        for _ in range(500):
            cx, cy = rng.uniform(-20.0 * PS, width + 20.0 * PS, 2)
            half = 10 ** rng.uniform(-2.0, 2.5) / 2 * PS
            viewport = ((cy - half, cx - half), (cy + half, cx + half))
            expected = len(_brute_force_in_view(locs, viewport))
            assert spatial_index.count_rect(pyramid, x, y, viewport) == (
                expected
            )

    def test_zoomed_in_count_excludes_edge_blocks(self):
        # the pyramid query returns whole edge blocks; the count must not
        W = 64.0 * PS
        locs = _make_locs(200_000, W, W)
        pyramid = spatial_index.build_render_index(locs, _info(W, W))
        viewport = ((20.3 * PS, 20.3 * PS), (20.8 * PS, 20.8 * PS))
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        expected = len(_brute_force_in_view(locs, viewport))
        assert len(spatial_index.query_viewport(pyramid, viewport)) > expected
        assert spatial_index.count_rect(pyramid, x, y, viewport) == expected

    def test_empty(self):
        W = 64.0 * PS
        locs = _make_locs(0, W, W)
        pyramid = spatial_index.build_render_index(locs, _info(W, W))
        empty = np.empty(0, dtype=np.float32)
        viewport = ((0.0, 0.0), (10.0 * PS, 10.0 * PS))
        assert spatial_index.count_rect(pyramid, empty, empty, viewport) == 0


class TestQuadTreeLayout:
    W = H = 64.0 * PS

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
        # 64 px of 130 nm = 8320 nm = 83.2 base blocks of 100 nm -> 128
        assert pyramid.root_bits == 7 and pyramid.fine_bits == 8

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
    W, H = 96.0 * PS, 64.0 * PS

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
        vp = ((10.0 * PS, 10.0 * PS), (30.0 * PS, 40.0 * PS))
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
            edited.loc[edited.index[123], "x"] = (
                self.W - 0.5 * PS
            )  # other block
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
        locs = _make_locs(0, 512 * PS, 512 * PS)
        pyr = spatial_index.build_render_index(locs, _info(512 * PS, 512 * PS))
        assert pyr is not None
        assert pyr.perm.shape == (0,)
        # Sub-threshold viewport hits the gather path and returns empty.
        idx = spatial_index.query_viewport(pyr, ((0, 0), (10, 10)))
        assert idx is not None and idx.shape == (0,)

    def test_missing_metadata_returns_none(self):
        locs = _make_locs(100, 512 * PS, 512 * PS)
        assert spatial_index.build_render_index(locs, [{}]) is None

    def test_perm_is_a_permutation(self):
        n = 10_000
        locs = _make_locs(n, 512, 512)
        pyr = spatial_index.build_render_index(locs, _info(512 * PS, 512 * PS))
        assert pyr.perm.shape == (n,)
        assert np.array_equal(np.sort(pyr.perm), np.arange(n, dtype=np.uint32))

    def test_levels_partition_total_count(self):
        locs = _make_locs(5_000, 576 * PS, 576 * PS)
        pyr = spatial_index.build_render_index(locs, _info(576 * PS, 576 * PS))
        for bs, be in zip(pyr.block_starts, pyr.block_ends):
            # Every loc belongs to exactly one block at each level.
            assert int((be - bs).sum()) == len(locs)
            # Block ranges don't overlap and don't go past N.
            assert (be >= bs).all()
            assert int(be.max()) <= len(locs)

    def test_block_sizes_geometric(self):
        pyr = spatial_index.build_render_index(
            _make_locs(100, 512 * PS, 512 * PS), _info(512 * PS, 512 * PS)
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
        W, H = 512.0 * PS, 512.0 * PS
        locs = _make_locs(20_000, W, H, seed=seed)
        pyr = spatial_index.build_render_index(locs, _info(W, H))

        # Random viewport inside the FOV, kept well under the bypass
        # coverage ratio so this test exercises the gather path.
        cx, cy = rng.uniform(50 * PS, W - 50 * PS), rng.uniform(
            50 * PS, H - 50 * PS
        )
        half_w, half_h = rng.uniform(5 * PS, 100 * PS), rng.uniform(
            5 * PS, 100 * PS
        )
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
        W, H = 512.0 * PS, 512.0 * PS
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        assert spatial_index.query_viewport(pyr, ((0, 0), (H, W))) is None

    def test_viewport_above_bypass_threshold_returns_none(self):
        # Coverage well below the bypass threshold still returns an
        # array; coverage well above triggers the bypass. Sized off the
        # module constant so the test tracks any future re-tuning.
        W, H = 512.0 * PS, 512.0 * PS
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
        W, H = 512.0 * PS, 512.0 * PS
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        assert (
            spatial_index.query_viewport(
                pyr,
                ((-100.0 * PS, -200.0 * PS), (H + 50.0 * PS, W + 300.0 * PS)),
            )
            is None
        )

    def test_viewport_overhanging_right_bottom_clips_correctly(self):
        # Locs are bounded in (0, W) x (0, H) but viewports aren't --
        # users can pan/zoom past the bottom-right. The clipped area
        # must drive the bypass decision, not the raw viewport extent.
        W, H = 512.0 * PS, 512.0 * PS
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        # ((100, 100), (700, 700)) px overhangs by 188 px on each side.
        # Clipped intersection: 412x412 -> ~65% of FOV -> bypass.
        big = ((100.0 * PS, 100.0 * PS), (700.0 * PS, 700.0 * PS))
        assert spatial_index.query_viewport(pyr, big) is None
        # Thin overhanging strip: 412x50 -> ~7.9% of FOV -> gather.
        strip = ((100.0 * PS, 100.0 * PS), (150.0 * PS, 700.0 * PS))
        idx = spatial_index.query_viewport(pyr, strip)
        assert idx is not None
        truth = _brute_force_in_view(locs, strip)
        assert set(int(i) for i in truth).issubset(set(int(i) for i in idx))

    def test_viewport_partially_negative_below_threshold_returns_array(self):
        # Viewport overhangs the FOV on one side only; the FOV-clipped
        # area is well under the bypass threshold, so the gather path
        # still runs and returns a valid array.
        W, H = 512.0 * PS, 512.0 * PS
        locs = _make_locs(5000, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        corner = ((-100.0 * PS, -100.0 * PS), (100.0 * PS, 100.0 * PS))
        idx = spatial_index.query_viewport(pyr, corner)
        assert idx is not None
        truth = _brute_force_in_view(locs, corner)
        assert set(int(i) for i in truth).issubset(set(int(i) for i in idx))

    def test_viewport_outside_fov_returns_empty(self):
        locs = _make_locs(1000, 512 * PS, 512 * PS)
        pyr = spatial_index.build_render_index(locs, _info(512 * PS, 512 * PS))
        # Far outside the FOV in both directions.
        idx = spatial_index.query_viewport(
            pyr, ((2000 * PS, 2000 * PS), (3000 * PS, 3000 * PS))
        )
        assert idx is not None and idx.shape == (0,)

    def test_tiny_zoomed_viewport_returns_few_locs(self):
        W, H = 512.0 * PS, 512.0 * PS
        n = 50_000
        locs = _make_locs(n, W, H)
        pyr = spatial_index.build_render_index(locs, _info(W, H))
        viewport = (
            (100.0 * PS, 100.0 * PS),
            (105.0 * PS, 105.0 * PS),
        )  # 5x5 px
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
        W, H = 512.0 * PS, 512.0 * PS
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
        viewport = ((40.0 * PS, 60.0 * PS), (180.0 * PS, 240.0 * PS))

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
