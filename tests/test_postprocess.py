"""Test ``picasso.postprocess`` and the closely associated
``picasso.g5m``.

Most fixtures (``locs``, ``info``) live in ``tests/conftest.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import clusterer, g5m, postprocess, zfit


# Reused parameters
PICK_SIZE = 1.5  # camera pixels
CALIB_3D = {
    "X Coefficients": [
        -1.6680708772714857e-18,
        2.4038209829154137e-15,
        2.1771067332017187e-12,
        -3.0324788231238476e-09,
        3.5433326085494675e-06,
        0.0023039289366630425,
        1.2026032603707493,
    ],
    "Y Coefficients": [
        -1.7708672355491796e-18,
        9.808249540501714e-16,
        2.10653248543535e-12,
        2.228026137415219e-11,
        3.628007433361433e-06,
        -0.001646865504353452,
        1.2257249554338714,
    ],
    "Step size in nm": 5.0,
    "Number of frames": 201,
    "Magnification factor": 0.79,
}


# 9-pick grid covering each origami in the bundled test movie
ORIGAMI_PICKS = [
    [5.5, 5.5],
    [5.5, 15.5],
    [5.5, 25.5],
    [15.5, 5.5],
    [15.5, 15.5],
    [15.5, 25.5],
    [25.5, 5.5],
    [25.5, 15.5],
    [25.5, 25.5],
]


@pytest.fixture(scope="module")
def origami_picks():
    return ORIGAMI_PICKS


@pytest.fixture
def locs_copy(locs):
    """Defensive copy of the session-scoped locs.

    Use this whenever a function under test mutates the input (e.g.,
    ``align``, ``apply_drift``, ``link``-with-sort) so the session
    fixture is not silently corrupted for downstream tests.
    """
    return locs.copy()


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------


class TestIndexBlocks:
    def test_index_blocks_structure(self, locs, info):
        index_blocks = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        ib_locs, size, x_index, y_index, b_starts, b_ends, K, L = index_blocks
        assert size == PICK_SIZE / 2
        assert len(x_index) == len(locs)
        assert len(y_index) == len(locs)
        assert K > 0 and L > 0
        assert b_starts.ndim == 2 and b_ends.ndim == 2
        assert b_starts.shape == b_ends.shape == (K, L)
        # block_starts[i,j] <= block_ends[i,j] for every cell
        assert (b_starts <= b_ends).all()
        # indices are sorted lexicographically by (y_index, x_index)
        keys = y_index.astype(np.int64) * (L + 1) + x_index.astype(np.int64)
        assert (np.diff(keys) >= 0).all()
        # total locs covered by the blocks equals len(locs)
        assert int((b_ends - b_starts).sum()) == len(locs)
        # the indexing also returns a re-sorted copy of the locs
        assert len(ib_locs) == len(locs)

    def test_index_blocks_shape_matches_field_of_view(self, info):
        size = 2.0
        n_y, n_x = postprocess.index_blocks_shape(info, size)
        assert n_y == int(np.ceil(info[0]["Height"] / size))
        assert n_x == int(np.ceil(info[0]["Width"] / size))

    def test_get_block_locs_at_returns_some_locs(self, locs, info):
        index_blocks = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        locs_at = postprocess.get_block_locs_at(15.5, 15.5, index_blocks)
        assert len(locs_at) > 0
        # Block lookup is conservative — within ~PICK_SIZE
        d = np.hypot(locs_at["x"] - 15.5, locs_at["y"] - 15.5)
        assert (d < 2 * PICK_SIZE).all()

    def test_n_block_locs_at_matches_get_block_locs_at(self, locs, info):
        ib = postprocess.get_index_blocks(locs, info, 1.0)
        _, _, _, _, b_starts, b_ends, K, L = ib
        # Pick a populated cell (away from boundary, where n_block_locs_at
        # uses strict inequality and skips the edges)
        for y_idx in range(2, K - 2):
            for x_idx in range(2, L - 2):
                n = postprocess.n_block_locs_at(
                    x_idx, y_idx, K, L, b_starts, b_ends
                )
                if n == 0:
                    continue
                # Expected: sum over the 3x3 cells around (y_idx, x_idx),
                # excluding boundary rows (n_block_locs_at uses 0 < k < K)
                expected = 0
                for k in range(y_idx - 1, y_idx + 2):
                    if 0 < k < K:
                        for ll in range(x_idx - 1, x_idx + 2):
                            if 0 < ll < L:
                                expected += b_ends[k, ll] - b_starts[k, ll]
                assert int(n) == int(expected)
                return  # one populated cell is enough


class TestRmsdAtCom:
    def test_known_value(self):
        # COM = (1, 0), distances 1, 0, 1 -> RMSD = sqrt(2/3)
        xy = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
        assert postprocess.rmsd_at_com(xy) == pytest.approx(np.sqrt(2 / 3))

    def test_zero_for_identical_points(self):
        xy = np.full((2, 5), 3.0)
        assert postprocess.rmsd_at_com(xy) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------


class TestPickedLocs:
    def test_one_list_per_pick(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        assert len(picked) == len(origami_picks)
        # Each pick has at least one loc (the test data has 9 origamis,
        # one per pick center)
        for p in picked:
            assert len(p) > 0

    def test_picked_locs_within_pick_radius(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        radius = PICK_SIZE / 2
        for (cx, cy), p in zip(origami_picks, picked):
            d = np.hypot(p["x"] - cx, p["y"] - cy)
            assert (d <= radius + 1e-6).all()

    def test_add_group_assigns_unique_ids(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        for i, p in enumerate(picked):
            assert (p["group"] == i).all()

    def test_add_group_false_omits_group(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
            add_group=False,
        )
        assert "group" not in picked[0].columns

    def test_picked_locs_sorted_by_frame(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        for p in picked:
            f = p["frame"].to_numpy()
            assert (np.diff(f) >= 0).all()

    def test_empty_picks_returns_empty_list(self, locs, info):
        out = postprocess.picked_locs(
            locs,
            info,
            [],
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        assert out == []

    def test_invalid_shape_raises(self, locs, info, origami_picks):
        with pytest.raises(AssertionError):
            postprocess.picked_locs(
                locs,
                info,
                origami_picks,
                pick_shape="Hexagon",
                pick_size=PICK_SIZE,
            )

    def test_precomputed_index_blocks_matches_internal(
        self, locs, info, origami_picks
    ):
        # ``_picked_circular_locs`` uses ``pick_size`` as the index-block
        # size, so the precomputed index_blocks must use the same size for
        # the two paths to be equivalent.
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        a = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        b = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
            index_blocks=ib,
        )
        assert len(a) == len(b)
        for pa, pb in zip(a, b):
            assert len(pa) == len(pb)

    def test_square_pick_within_bounds(self, locs, info):
        side = 2.0
        cx, cy = 15.5, 15.5
        out = postprocess.picked_locs(
            locs,
            info,
            [(cx, cy)],
            pick_shape="Square",
            pick_size=side,
        )[0]
        assert len(out) > 0
        assert (out["x"] > cx - side / 2).all()
        assert (out["x"] < cx + side / 2).all()
        assert (out["y"] > cy - side / 2).all()
        assert (out["y"] < cy + side / 2).all()

    def test_rectangle_pick_returns_locs(self, locs, info):
        # A rectangle with width 2.0 around the line from (5,5) to (8,5)
        out = postprocess.picked_locs(
            locs,
            info,
            [((5.0, 5.0), (8.0, 5.0))],
            pick_shape="Rectangle",
            pick_size=2.0,
        )[0]
        assert len(out) > 0
        # rotation columns added by the rectangular helper
        assert "x_pick_rot" in out.columns
        assert "y_pick_rot" in out.columns

    def test_polygon_pick_returns_locs(self, locs, info):
        # Closed polygon (first vertex repeated) around (15.5, 15.5)
        polygon = [
            (14.5, 14.5),
            (16.5, 14.5),
            (16.5, 16.5),
            (14.5, 16.5),
            (14.5, 14.5),
        ]
        out = postprocess.picked_locs(
            locs,
            info,
            [polygon],
            pick_shape="Polygon",
        )[0]
        assert len(out) > 0
        assert (out["x"] > 14.5).all() and (out["x"] < 16.5).all()
        assert (out["y"] > 14.5).all() and (out["y"] < 16.5).all()


class TestPickSimilar:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, PICK_SIZE, std_range=123.0
        )
        assert len(new_picks) == len(ORIGAMI_PICKS)

    def test_precomputed_index_blocks_path(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            seed_picks,
            PICK_SIZE,
            std_range=123.0,
            index_blocks=ib,
        )
        assert len(new_picks) == len(ORIGAMI_PICKS)


class TestRemoveLocsInPicks:
    def test_locs_in_pick_removed(self, locs, info):
        picks = [(15.5, 15.5)]
        # Reference: how many locs lie inside this pick?
        picked = postprocess.picked_locs(
            locs,
            info,
            picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )[0]
        n_inside = len(picked)
        out = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        assert len(out) == len(locs) - n_inside
        # No remaining loc lies inside the pick
        d = np.hypot(out["x"] - 15.5, out["y"] - 15.5)
        assert (d > PICK_SIZE / 2 - 1e-6).all()

    def test_polygon_pick_size_ignored(self, locs, info):
        polygon = [
            (14.5, 14.5),
            (16.5, 14.5),
            (16.5, 16.5),
            (14.5, 16.5),
            (14.5, 14.5),
        ]
        # pick_size is asserted only for non-polygon shapes — not needed
        # here but must not raise
        out = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=[polygon],
            pick_shape="Polygon",
        )
        assert len(out) < len(locs)

    def test_invalid_shape_raises(self, locs, info):
        with pytest.raises(AssertionError):
            postprocess.remove_locs_in_picks(
                locs.copy(),
                info,
                picks=[(15.5, 15.5)],
                pick_shape="Hexagon",
                pick_size=1.0,
            )


# ---------------------------------------------------------------------------
# Statistics on locs distributions
# ---------------------------------------------------------------------------


class TestDistanceHistogram:
    def test_shape_and_dtype(self, locs, info):
        dh = postprocess.distance_histogram(
            locs, info, bin_size=0.1, r_max=1.0
        )
        # 10 bins for r_max=1.0 / bin_size=0.1
        assert dh.shape == (10,)
        assert (dh >= 0).all()

    def test_total_count_is_finite_and_positive(self, locs, info):
        dh = postprocess.distance_histogram(
            locs, info, bin_size=0.1, r_max=1.0
        )
        assert dh.sum() > 0

    def test_count_grows_with_r_max(self, locs, info):
        small = postprocess.distance_histogram(
            locs, info, bin_size=0.1, r_max=0.5
        )
        large = postprocess.distance_histogram(
            locs, info, bin_size=0.1, r_max=1.5
        )
        # The large-r histogram must contain all bins of the small-r one
        # (same bin edges) plus more — sum is monotone in r_max.
        assert large[: len(small)].sum() >= small.sum()


class TestNena:
    def test_returns_positive_resolution(self, locs, info):
        _, nena = postprocess.nena(locs, info)
        assert nena > 0
        # NeNA in pixels is sub-pixel for DNA-PAINT data
        assert nena < 5

    def test_result_keys(self, locs, info):
        res, _ = postprocess.nena(locs, info)
        for key in ["d", "data", "best_fit", "best_values"]:
            assert key in res
        for key in ["delta_a", "s", "ac", "dc", "sc"]:
            assert key in res["best_values"]
        # ``s`` corresponds to localization precision and must be positive
        assert res["best_values"]["s"] > 0

    def test_returned_s_matches_nena_value(self, locs, info):
        res, nena = postprocess.nena(locs, info)
        # Convention in postprocess: nena is the fitted ``s`` parameter
        assert res["best_values"]["s"] == pytest.approx(nena)


class TestNextFrameNeighborDistanceHistogram:
    def test_shape_and_non_negative(self, locs):
        bin_centers, dnfl = postprocess.next_frame_neighbor_distance_histogram(
            locs.copy()
        )
        assert bin_centers.shape == dnfl.shape
        assert (dnfl >= 0).all()
        # bin centers are evenly spaced
        diffs = np.diff(bin_centers)
        assert np.allclose(diffs, diffs[0])

    def test_some_neighbors_present(self, locs):
        _, dnfl = postprocess.next_frame_neighbor_distance_histogram(
            locs.copy()
        )
        # Bundled DNA-PAINT data has many on-events lasting >1 frame, so
        # there should be at least some next-frame neighbors recorded.
        assert dnfl.sum() > 0


class TestFrc:
    def test_resolution_keys(self, locs, info):
        viewport = ((15, 15), (16, 16))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        for key in [
            "frequencies",
            "frc_curve",
            "resolution",
            "images",
            "frc_curve_smooth",
        ]:
            assert key in frc_res
        assert frc_res["resolution"] > 0

    def test_frc_curve_starts_near_one(self, locs, info):
        """At low frequency, the FRC curve should be close to 1."""
        viewport = ((15, 15), (16, 16))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["frc_curve"][0] > 0.7

    def test_frc_curve_and_freq_same_length(self, locs, info):
        viewport = ((15, 15), (16, 16))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["frc_curve"].shape == frc_res["frequencies"].shape
        assert (
            frc_res["frc_curve_smooth"].shape == frc_res["frequencies"].shape
        )

    def test_two_images_returned(self, locs, info):
        viewport = ((15, 15), (16, 16))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        images = frc_res["images"]
        assert len(images) == 2
        assert images[0].shape == images[1].shape
        # images are square and odd-sized after the internal trim
        assert images[0].ndim == 2
        assert images[0].shape[0] == images[0].shape[1]
        assert images[0].shape[0] % 2 == 1

    def test_rectangular_viewport_squared_internally(self, locs, info):
        """A non-square viewport must still yield a square image."""
        viewport = ((15, 13), (16, 17))  # width 4, height 1
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["images"][0].shape[0] == frc_res["images"][0].shape[1]


class TestPairCorrelation:
    def test_shape(self, locs, info):
        bins_lower, pc = postprocess.pair_correlation(
            locs, info, bin_size=0.1, r_max=1.0
        )
        assert bins_lower.shape == pc.shape
        assert bins_lower.shape[0] == 10

    def test_pc_finite_and_non_negative(self, locs, info):
        _, pc = postprocess.pair_correlation(
            locs, info, bin_size=0.1, r_max=1.0
        )
        assert np.all(np.isfinite(pc))
        assert (pc >= 0).all()

    def test_normalisation_against_distance_histogram(self, locs, info):
        bin_size, r_max = 0.1, 1.0
        dh = postprocess.distance_histogram(locs, info, bin_size, r_max)
        bins_lower, pc = postprocess.pair_correlation(
            locs, info, bin_size, r_max
        )
        # pc = dh / (pi * bin_size * (2 * bins_lower + bin_size))
        expected = dh / (np.pi * bin_size * (2 * bins_lower + bin_size))
        np.testing.assert_allclose(pc, expected, rtol=1e-6)


class TestLocalDensity:
    def test_density_column_added_with_proper_dtype(self, locs, info):
        out = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE / 2
        )
        assert "density" in out.columns
        assert out["density"].dtype in (np.uint32, np.uint64)
        # Each loc has at least itself within radius
        assert (out["density"] >= 1).all()

    def test_dense_radius_picks_up_origami_clusters(self, locs, info):
        out = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE / 2
        )
        unique_densities, _ = np.unique(out["density"], return_counts=True)
        # Test data has ~9 origamis, so density should take few values
        assert len(unique_densities) < len(locs) // 5

    def test_density_increases_with_radius(self, locs, info):
        small = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE / 4
        )
        large = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE
        )
        # A larger radius can only see at least as many neighbors.
        assert large["density"].sum() >= small["density"].sum()


# ---------------------------------------------------------------------------
# Linking and dark-time computation
# ---------------------------------------------------------------------------


class TestLinking:
    def test_columns_added(self, locs, info):
        linked = postprocess.link(locs.copy(), info)
        for col in ["len", "n", "photon_rate"]:
            assert col in linked.columns

    def test_length_invariants(self, locs, info):
        """Linked length is <= original (events merge); the sum of the
        ``n`` (locs per linked event) column equals the original count."""
        linked = postprocess.link(locs.copy(), info)
        assert len(linked) <= len(locs)
        assert linked["n"].sum() == len(locs)

    def test_len_within_movie_frame_span(self, locs, info):
        linked = postprocess.link(locs.copy(), info)
        n_frames = info[0]["Frames"]
        assert linked["len"].max() <= n_frames

    def test_compute_dark_times_adds_dark_column(self, locs, info):
        linked = postprocess.link(locs.copy(), info)
        with_dark = postprocess.compute_dark_times(linked)
        assert "dark" in with_dark.columns

    def test_compute_dark_times_requires_link(self, locs):
        with pytest.raises(AttributeError):
            postprocess.compute_dark_times(locs.copy())

    def test_link_empty_locs_returns_empty_with_columns(self, locs, info):
        empty = locs.iloc[0:0].copy()
        out = postprocess.link(empty, info)
        assert len(out) == 0
        for col in ["len", "n", "photon_rate"]:
            assert col in out.columns

    def test_link_refit_not_implemented(self, locs, info):
        with pytest.raises(NotImplementedError):
            postprocess.link(locs.copy(), info, combine_mode="refit")

    def test_link_groups_consistent_with_link(self, locs, info):
        # The number of unique non-(-1) link groups must equal the
        # number of linked events when there are no ambiguities to drop.
        sl = locs.sort_values(by="frame", kind="quicksort")
        frame = sl["frame"].to_numpy()
        x = sl["x"].to_numpy()
        y = sl["y"].to_numpy()
        group = np.zeros(len(sl), dtype=np.int32)
        lg = postprocess.get_link_groups(frame, x, y, 0.05, 3, group)
        assert len(lg) == len(locs)
        # All locs must be assigned to a real link group (>= 0)
        assert (lg >= 0).all()

    def test_get_link_groups_tight_radius_separates_locs(self, locs):
        # With a vanishingly small linking radius, each loc should be in
        # its own group.
        sl = locs.sort_values(by="frame", kind="quicksort")
        frame = sl["frame"].to_numpy()
        x = sl["x"].to_numpy()
        y = sl["y"].to_numpy()
        group = np.zeros(len(sl), dtype=np.int32)
        lg = postprocess.get_link_groups(frame, x, y, 1e-9, 1, group)
        assert len(np.unique(lg)) == len(sl)


class TestDarkTimes:
    def test_dark_times_min_positive(self, locs, info):
        linked = postprocess.link(locs.copy(), info)
        dt = postprocess.dark_times(linked)
        assert dt.shape == (len(linked),)
        # -1 sentinel for events not followed by another in the group;
        # all others must be strictly positive (gap of at least 1 frame).
        assert ((dt > 0) | (dt == -1)).all()

    def test_dark_times_with_explicit_group(self, locs, info):
        linked = postprocess.link(locs.copy(), info)
        # Two halves marked as different groups should not see each other
        # as dark-time neighbors; both get all -1 if singletons in group.
        n = len(linked)
        group_arr = np.zeros(n, dtype=np.int32)
        group_arr[n // 2 :] = 1
        dt_split = postprocess.dark_times(linked, group=group_arr)
        # The split must produce at least as many -1 sentinels as the
        # un-split version (boundary events become unmatched).
        dt_full = postprocess.dark_times(linked)
        assert (dt_split == -1).sum() >= (dt_full == -1).sum()


# ---------------------------------------------------------------------------
# Pick-derived per-pick statistics and combination
# ---------------------------------------------------------------------------


class TestEvaluatePicks:
    def test_returns_per_pick_arrays(self, locs, info, origami_picks):
        pl = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        N, n_events, rmsd, rmsd_z, length, dark, new_locs = (
            postprocess.evaluate_picks(pl, info, max_dark_time=3)
        )
        npicks = len(origami_picks)
        for arr in (N, n_events, rmsd, rmsd_z, length, dark):
            assert arr.shape == (npicks,)
        # Number of locs in each pick matches the picked_locs result
        for i, p in enumerate(pl):
            assert N[i] == len(p)
        # RMSD is in nm; bundled data has Pixelsize=130 and origamis are
        # ~tens of nm wide — RMSD must be positive.
        assert (rmsd > 0).all()
        # n_events <= N (linking only ever merges)
        assert (n_events <= N).all()
        # Returned new_locs must have length and dark columns
        for col in ("len", "dark"):
            assert col in new_locs.columns


class TestCombineLocsInPicks:
    def test_combines_into_one_loc_per_pick(self, locs, info, origami_picks):
        combined = postprocess.combine_locs_in_picks(
            locs.copy(),
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        # Each origami collapses to a single linked event
        assert len(combined) == len(origami_picks)
        # n column tracks how many locs each event came from, and the
        # totals must match the picked-loc count.
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        assert combined["n"].sum() == sum(len(p) for p in picked)


class TestPickKinetics:
    def test_per_pick_arrays_and_out_locs(self, locs, info, origami_picks):
        pl = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        length, dark, no_locs, out_locs = postprocess.pick_kinetics(
            pl, info, max_dark_time=3
        )
        # All four returned arrays are 1D and aligned in length: one
        # entry per successfully-evaluated pick (picks where kinetics
        # could not be estimated are silently dropped).
        assert length.ndim == dark.ndim == no_locs.ndim == 1
        assert length.shape == dark.shape == no_locs.shape
        assert length.shape[0] <= len(origami_picks)
        # Bright/dark times are physical durations in frames — strictly
        # positive whenever they exist.
        assert (length > 0).all()
        assert (dark > 0).all()
        # ``no_locs`` counts events per pick after linking and dark-time
        # computation; must be positive.
        assert (no_locs > 0).all()
        # Returned per-loc dataframe carries the kinetics columns.
        for col in ("len", "n", "dark"):
            assert col in out_locs.columns
        # The number of binding events across surviving picks equals the
        # sum of per-pick counts.
        assert len(out_locs) == int(no_locs.sum())


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------


class TestSegmentation:
    def test_n_segments_round(self, info):
        n_frames = info[0]["Frames"]
        assert postprocess.n_segments(info, n_frames) == 1
        # 5 segments of 200 frames each
        assert postprocess.n_segments(info, 200) == 5

    def test_segment_shapes(self, locs, info):
        segmentation = 200
        n_seg = postprocess.n_segments(info, segmentation)
        bounds, segs = postprocess.segment(locs.copy(), info, segmentation)
        assert bounds.shape == (n_seg + 1,)
        assert segs.shape == (n_seg, info[0]["Height"], info[0]["Width"])
        # bounds are strictly increasing and span the movie
        assert (np.diff(bounds) > 0).all()
        assert bounds[0] == 0
        assert bounds[-1] == info[0]["Frames"] - 1


class TestUndrift:
    def test_drift_has_one_row_per_frame(self, locs, info):
        drift, undrifted = postprocess.undrift(
            locs.copy(),
            info,
            segmentation=200,
            display=False,
        )
        n_frames = info[0]["Frames"]
        assert isinstance(drift, pd.DataFrame)
        assert drift.shape == (n_frames, 2)
        assert {"x", "y"}.issubset(drift.columns)
        assert len(undrifted) == len(locs)

    def test_undrift_from_picked_returns_drift(
        self, locs, info, origami_picks
    ):
        # The origami picks are not real fiducials but they exercise the
        # code path and produce a valid (n_frames, 2) drift table.
        pl = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
            add_group=False,
        )
        drift = postprocess.undrift_from_picked(pl, info)
        n_frames = info[0]["Frames"]
        assert drift.shape == (n_frames, 2)
        assert {"x", "y"} == set(drift.columns)
        # Per-frame mean drift should be finite (NaNs only in frames
        # where no pick contributes — that's allowed)
        assert np.isfinite(drift.dropna()).all().all()

    def test_undrift_from_fiducials_with_user_picks(
        self, locs, info, origami_picks
    ):
        out_locs, new_info, drift = postprocess.undrift_from_fiducials(
            locs.copy(),
            info,
            picks=origami_picks,
            pick_size=PICK_SIZE / 2,
            undrift_z=False,
        )
        assert len(out_locs) == len(locs)
        n_frames = info[0]["Frames"]
        assert drift.shape == (n_frames, 2)
        # New info entry appended with a generator tag
        assert any(
            "Undrift from picked" in str(d.get("Generated by", ""))
            for d in new_info
        )

    def test_undrift_from_fiducials_picks_without_size_raises(
        self, locs, info
    ):
        with pytest.raises(ValueError):
            postprocess.undrift_from_fiducials(
                locs.copy(),
                info,
                picks=[(5.5, 5.5)],
                pick_size=None,
            )


class TestApplyDrift:
    def test_apply_constant_drift_dataframe(self, locs, info):
        n_frames = info[0]["Frames"]
        dy = 0.5
        drift = pd.DataFrame(
            {
                "x": np.zeros(n_frames),
                "y": np.full(n_frames, dy),
            }
        )
        out = postprocess.apply_drift(locs.copy(), info, drift=drift)
        assert out["y"].mean() == pytest.approx(
            locs["y"].mean() - dy, abs=1e-3
        )
        assert out["x"].mean() == pytest.approx(locs["x"].mean(), abs=1e-6)

    def test_apply_drift_ndarray_2d(self, locs, info):
        n_frames = info[0]["Frames"]
        drift = np.zeros((n_frames, 2), dtype=np.float64)
        drift[:, 0] = 0.25
        out = postprocess.apply_drift(locs.copy(), info, drift=drift)
        assert out["x"].mean() == pytest.approx(
            locs["x"].mean() - 0.25, abs=1e-3
        )

    def test_apply_drift_array_wrong_shape_raises(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.apply_drift(
                locs.copy(),
                info,
                drift=np.zeros((10, 5)),
            )

    def test_apply_drift_dataframe_missing_columns_raises(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.apply_drift(
                locs.copy(),
                info,
                drift=pd.DataFrame({"foo": [1, 2]}),
            )

    def test_apply_drift_invalid_type_raises(self, locs, info):
        with pytest.raises(AssertionError):
            postprocess.apply_drift(locs.copy(), info, drift="not a frame")


# ---------------------------------------------------------------------------
# Channel alignment
# ---------------------------------------------------------------------------


class TestAlign:
    def test_channels_aligned_after_known_shift(self, locs_copy, info):
        """Apply a known +5 px shift to a copy and check that align()
        brings the channels back together (residual <0.5 px)."""
        a = locs_copy
        b = a.copy()
        b["x"] += 5.0
        aligned = postprocess.align([a, b], [info, info])
        assert len(aligned) == 2
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.5

    def test_no_shift_is_no_op_within_tolerance(self, locs_copy, info):
        """If channels start aligned, alignment shouldn't drift them."""
        a = locs_copy
        b = a.copy()
        aligned = postprocess.align([a, b], [info, info])
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.1

    def test_apply_shifts_false_does_not_modify_locs(self, locs_copy, info):
        a = locs_copy
        b = a.copy()
        b["x"] += 3.0
        x_before_a = a["x"].copy()
        x_before_b = b["x"].copy()
        out, shifts = postprocess.align(
            [a, b],
            [info, info],
            apply_shifts=False,
            return_shifts=True,
        )
        # No mutation when apply_shifts is False
        np.testing.assert_array_equal(out[0]["x"].to_numpy(), x_before_a)
        np.testing.assert_array_equal(out[1]["x"].to_numpy(), x_before_b)
        # Shifts is a 2-tuple of arrays of length n_channels
        shift_x, shift_y = shifts
        assert len(shift_x) == 2
        assert len(shift_y) == 2

    def test_align_rcc_converges(self, locs_copy, info):
        a = locs_copy
        b = a.copy()
        b["x"] += 2.0
        aligned = postprocess.align_rcc([a, b], [info, info])
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.5

    def test_align_from_picked_recovers_known_shift(
        self, locs_copy, info, origami_picks
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1
        aligned, shifts = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            return_shifts=True,
        )
        # shifts is (shift_y, shift_x) where each is per-channel
        # The second channel should have ~0.1 in the x-shift slot
        # (sign convention: shift to subtract from x).
        assert abs(shifts[1][1] - 0.1) < 0.05
        # First channel is the reference: ~0 shift
        assert abs(shifts[0][0]) < 1e-6
        assert abs(shifts[1][0]) < 1e-6

    def test_align_from_picked_invalid_shape_raises(
        self, locs_copy, info, origami_picks
    ):
        with pytest.raises(AssertionError):
            postprocess.align_from_picked(
                [locs_copy, locs_copy.copy()],
                [info, info],
                picks=origami_picks,
                pick_shape="Hexagon",
                pick_size=1.0,
            )


# ---------------------------------------------------------------------------
# groupprops
# ---------------------------------------------------------------------------


class TestGroupprops:
    @pytest.fixture
    def grouped_locs(self, locs, info, origami_picks):
        """Build per-origami picked + linked + dark-times locs."""
        picked = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        merged = pd.concat(picked, ignore_index=True)
        merged = postprocess.link(merged, info)
        return postprocess.compute_dark_times(merged)

    def test_required_columns(self, grouped_locs):
        old_columns = grouped_locs.columns.tolist()
        out = postprocess.groupprops(grouped_locs)
        expected = [c + "_mean" for c in old_columns]
        expected += [c + "_std" for c in old_columns]
        expected += ["n_events", "qpaint_idx"]
        for col in expected:
            assert col in out.columns

    def test_per_group_means_match_manual(self, grouped_locs):
        out = postprocess.groupprops(grouped_locs)
        for g in grouped_locs["group"].unique()[:3]:
            manual = grouped_locs.loc[grouped_locs["group"] == g, "x"].mean()
            row = (
                out.loc[out["group"] == g]
                if "group" in out.columns
                else (out.iloc[[g]])
            )
            assert row["x_mean"].iloc[0] == pytest.approx(manual, rel=1e-3)

    def test_n_events_matches_group_size(self, grouped_locs):
        out = postprocess.groupprops(grouped_locs)
        for g in grouped_locs["group"].unique():
            n = (grouped_locs["group"] == g).sum()
            row = out.loc[out["group"] == g]
            assert row["n_events"].iloc[0] == n

    def test_qpaint_idx_is_inverse_of_dark_mean(self, grouped_locs):
        out = postprocess.groupprops(grouped_locs)
        finite = out[out["dark_mean"] > 0]
        np.testing.assert_allclose(
            finite["qpaint_idx"].to_numpy(),
            1.0 / finite["dark_mean"].to_numpy(),
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Cluster combination
# ---------------------------------------------------------------------------


class TestClusterCombine:
    @pytest.fixture
    def clustered(self, locs):
        """Run dbscan to assign per-loc cluster ids (in the 'cluster'
        column) but lump every loc into a single group, so that
        ``cluster_combine_dist`` can compute inter-cluster distances
        within that group."""
        out = clusterer.dbscan(locs, radius=2 / 130, min_samples=2)
        out = out.copy()
        out["cluster"] = out["group"].to_numpy()
        out["group"] = 0
        return out

    def test_cluster_combine_one_row_per_cluster(self, clustered):
        combined = postprocess.cluster_combine(clustered)
        n_clusters = len(np.unique(clustered["cluster"]))
        assert len(combined) == n_clusters
        for col in ("group", "cluster", "x", "y", "n", "lpx", "lpy"):
            assert col in combined.columns
        assert (combined["n"] > 0).all()
        # Per-cluster ``n`` totals must equal the input row count
        assert combined["n"].sum() == len(clustered)

    def test_cluster_combine_dist_2d_min_dist(self, clustered):
        combined = postprocess.cluster_combine(clustered)
        if len(combined) < 2:
            pytest.skip("Need at least 2 clusters in the group for cdist")
        out = postprocess.cluster_combine_dist(combined)
        assert "min_dist" in out.columns
        assert (out["min_dist"] >= 0).all()
        # Brute-force the nearest-neighbor distance per cluster and
        # confirm it matches the function's output.
        xy = combined[["x", "y"]].to_numpy()
        for i in range(len(combined)):
            others = np.delete(xy, i, axis=0)
            expected = float(np.min(np.linalg.norm(others - xy[i], axis=1)))
            assert out["min_dist"].iloc[i] == pytest.approx(expected, rel=1e-4)

    def test_cluster_combine_dist_3d_min_dist_xy(self, clustered):
        # Add a synthetic z column so the 3D branch is exercised. The
        # 3D output also reports an xy nearest-neighbor distance under
        # the (existing) 'mind_dist_xy' column name.
        clustered = clustered.copy()
        clustered["z"] = 0.0  # dummy z; xy distance is what we verify
        combined = postprocess.cluster_combine(clustered)
        if len(combined) < 2:
            pytest.skip("Need at least 2 clusters in the group for cdist")
        out = postprocess.cluster_combine_dist(combined)
        assert "min_dist" in out.columns
        assert "mind_dist_xy" in out.columns  # existing typo in the API
        assert (out["min_dist"] >= 0).all()
        assert (out["mind_dist_xy"] >= 0).all()
        # With z=0 everywhere, 3D and xy distances must agree.
        np.testing.assert_allclose(
            out["min_dist"].to_numpy(),
            out["mind_dist_xy"].to_numpy(),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# FRET and nearest-neighbor analysis
# ---------------------------------------------------------------------------


class TestCalculateFret:
    def test_returns_keys_and_no_events_for_disjoint_frames(self, locs):
        # Choose acc and don frames with no overlap so fret_trace is 0
        # everywhere (FRET requires both donor and acceptor in same frame).
        a = locs.iloc[:50].copy()
        b = locs.iloc[50:100].copy()
        a["frame"] = np.arange(0, 50)
        b["frame"] = np.arange(100, 150)
        fret_dict, f_locs = postprocess.calculate_fret(a, b)
        for key in (
            "fret_events",
            "fret_timepoints",
            "acc_trace",
            "don_trace",
            "frames",
            "maxframes",
        ):
            assert key in fret_dict
        # No FRET events when donor/acceptor frames are disjoint
        assert len(fret_dict["fret_events"]) == 0

    def test_fret_events_in_range(self, locs):
        # Force coincident frames so FRET is computed
        a = locs.iloc[:50].copy()
        b = locs.iloc[50:100].copy()
        a["frame"] = np.arange(50)
        b["frame"] = np.arange(50)
        # Ensure positive (photons - bg)
        a["photons"] = 1000.0
        a["bg"] = 10.0
        b["photons"] = 1000.0
        b["bg"] = 10.0
        fret_dict, _ = postprocess.calculate_fret(a, b)
        events = fret_dict["fret_events"]
        # The function only keeps fret values in (0, 1)
        assert ((events > 0) & (events < 1)).all()


class TestNnAnalysis:
    def test_inter_set_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        Y = rng.standard_normal((25, 2))
        out = postprocess.nn_analysis(X, Y, nn_count=3)
        assert out.shape == (20, 3)
        # Distances are sorted ascending along axis=1
        assert (np.diff(out, axis=1) >= 0).all()

    def test_self_excludes_zero_distance(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 3))
        out = postprocess.nn_analysis(X, X, nn_count=2)
        assert out.shape == (30, 2)
        # Self-NN must skip the trivial 0-distance match
        assert (out > 0).all()

    def test_dimension_mismatch_raises(self):
        X = np.zeros((5, 2))
        Y = np.zeros((5, 3))
        with pytest.raises(ValueError):
            postprocess.nn_analysis(X, Y, nn_count=1)


# ---------------------------------------------------------------------------
# RESI
# ---------------------------------------------------------------------------


class TestResi:
    def test_resi_2d_combines_channels(self, locs, info):
        out, new_info = postprocess.resi(
            [locs.copy(), locs.copy()],
            [info, info],
            radius_xy=2 / 130,
            min_locs=2,
        )
        assert len(out) > 0
        # Channel id present and covers both channels
        assert "resi_channel_id" in out.columns
        assert set(out["resi_channel_id"].unique()) == {0, 1}
        # Group renamed to cluster_id
        assert "cluster_id" in out.columns
        assert "group" not in out.columns
        # New info entry holds RESI metadata
        assert any(
            "Clustering radius xy (nm) for each channel" in d for d in new_info
        )

    def test_resi_requires_two_channels(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.resi(
                [locs.copy()],
                [info],
                radius_xy=2 / 130,
                min_locs=2,
            )

    def test_resi_per_channel_list_length_validated(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.resi(
                [locs.copy(), locs.copy()],
                [info, info],
                radius_xy=[2 / 130],
                min_locs=2,
            )
        with pytest.raises(ValueError):
            postprocess.resi(
                [locs.copy(), locs.copy()],
                [info, info],
                radius_xy=2 / 130,
                min_locs=[2, 3, 4],
            )


# ---------------------------------------------------------------------------
# g5m end-to-end (consumes postprocess output)
# ---------------------------------------------------------------------------


class TestG5M:
    @pytest.fixture
    def dbscan_locs(self, locs):
        out = clusterer.dbscan(locs, radius=2 / 130, min_samples=2)
        assert len(out) > 0
        return out

    def test_g5m_2d_with_bootstrap(self, dbscan_locs, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs, info, min_locs=5, bootstrap_check=True, asynch=False
        )
        assert "p_val" in mols.columns
        # p-values must be in [0, 1]
        assert (mols["p_val"] >= 0).all() and (mols["p_val"] <= 1).all()

    def test_g5m_2d_global_loc_prec(self, dbscan_locs, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs,
            info,
            min_locs=5,
            bootstrap_check=False,
            loc_prec_handle="abs",
            sigma_bounds=(1 / 130, 3 / 130),
        )
        assert "p_val" in mols.columns
        assert len(mols) > 0


class TestG5M3D:
    @pytest.fixture
    def dbscan_locs_3d(self, locs, info):
        out = clusterer.dbscan(locs, radius=2 / 130, min_samples=2)
        assert len(out) > 0
        rng = np.random.default_rng(42)
        out = out.copy()
        out["z"] = rng.normal(0, 2, size=len(out))
        out["lpz"] = zfit.axial_localization_precision(
            out, info, calibration=CALIB_3D, fitting_method="gaussmle"
        )
        return out

    def test_g5m_3d_with_bootstrap(self, dbscan_locs_3d, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            bootstrap_check=True,
            calibration=CALIB_3D,
            asynch=False,
        )
        assert len(mols) > 0
        assert "p_val" in mols.columns
        assert (mols["p_val"] >= 0).all() and (mols["p_val"] <= 1).all()

    def test_g5m_3d_global_loc_prec(self, dbscan_locs_3d, info):
        mols, _, _ = g5m.g5m(
            dbscan_locs_3d,
            info,
            min_locs=5,
            bootstrap_check=False,
            calibration=CALIB_3D,
            loc_prec_handle="abs",
            sigma_bounds=(1 / 130, 3 / 130),
        )
        assert len(mols) > 0
        assert "p_val" in mols.columns
