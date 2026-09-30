"""Test ``picasso.postprocess``.

Most fixtures (``locs``, ``info``) live in ``tests/conftest.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import clusterer, io, lib, postprocess


# Reused parameters
PICK_SIZE = 1.5  # camera pixels


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


class TestPyramidAsPickIndex:
    """The load-time render pyramid can stand in for the index blocks
    of circular picks, with identical results."""

    def test_picked_locs(self, locs, info, origami_picks):
        from picasso import spatial_index

        pyramid = spatial_index.build_render_index(locs, info)
        via_blocks = postprocess.picked_locs(
            locs, info, origami_picks, "Circle", pick_size=PICK_SIZE / 2
        )
        via_pyramid = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            "Circle",
            pick_size=PICK_SIZE / 2,
            index_blocks=pyramid,
        )
        assert sum(len(_) for _ in via_pyramid) > 0
        for a, b in zip(via_blocks, via_pyramid):
            # equal-frame rows may come out in a different order
            pd.testing.assert_frame_equal(
                a.sort_index(), b.sort_index(), check_like=True
            )

    def test_pick_similar(self, locs, info, origami_picks):
        from picasso import spatial_index

        pyramid = spatial_index.build_render_index(locs, info)
        kwargs = dict(
            locs=locs,
            info=info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            std_range=2.0,
        )
        via_blocks = postprocess.pick_similar(**kwargs)
        via_pyramid = postprocess.pick_similar(index_blocks=pyramid, **kwargs)
        assert via_pyramid == via_blocks

    def test_other_shapes_ignore_the_pyramid(self, locs, info, origami_picks):
        from picasso import spatial_index

        pyramid = spatial_index.build_render_index(locs, info)
        square = postprocess.pick_similar(
            locs=locs,
            info=info,
            picks=origami_picks,
            pick_shape="Square",
            pick_size=PICK_SIZE,
            std_range=2.0,
            index_blocks=pyramid,
        )
        assert isinstance(square, list)


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
        n_y, n_x = postprocess._index_blocks_shape(info, size)
        assert n_y == int(np.ceil(info[0]["Height"] / size))
        assert n_x == int(np.ceil(info[0]["Width"] / size))

    def test_get_block_locs_at_returns_some_locs(self, locs, info):
        r = PICK_SIZE / 2
        index_blocks = postprocess.get_index_blocks(locs, info, r)
        locs_sorted, size, _, _, block_starts, block_ends, K, L = index_blocks
        locs_xy = locs_sorted[["x", "y"]].to_numpy().T
        x, y = 15.5, 15.5
        locs_at = postprocess.get_block_locs_at_numba(
            int(x / r), int(y / r), locs_xy, block_starts, block_ends, K, L
        )
        assert locs_at.shape[1] > 0
        # Block lookup is conservative — within ~PICK_SIZE
        d = np.hypot(locs_at[0] - x, locs_at[1] - y)
        assert (d < 2 * PICK_SIZE).all()

    def test_n_block_locs_at_matches_get_block_locs_at(self, locs, info):
        ib = postprocess.get_index_blocks(locs, info, 1.0)
        _, _, _, _, b_starts, b_ends, K, L = ib
        # Pick a populated cell (away from boundary, where n_block_locs_at
        # uses strict inequality and skips the edges)
        for y_idx in range(2, K - 2):
            for x_idx in range(2, L - 2):
                n = postprocess._n_block_locs_at(
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

    def test_box_pick_within_bounds(self, locs, info):
        # 3 wide, 1 high, so a square test would give a different answer
        pick = ((14.0, 15.0), (17.0, 16.0))
        out = postprocess.picked_locs(
            locs,
            info,
            [pick],
            pick_shape="Box",
        )[0]
        assert len(out) > 0
        assert (out["x"] > 14.0).all() and (out["x"] < 17.0).all()
        assert (out["y"] > 15.0).all() and (out["y"] < 16.0).all()

    def test_box_pick_ignores_corner_order(self, locs, info):
        ordered = postprocess.picked_locs(
            locs, info, [((14.0, 15.0), (17.0, 16.0))], pick_shape="Box"
        )[0]
        reversed_ = postprocess.picked_locs(
            locs, info, [((17.0, 16.0), (14.0, 15.0))], pick_shape="Box"
        )[0]
        assert len(ordered) == len(reversed_)

    def test_box_pick_ignores_pick_size(self, locs, info):
        pick = ((14.0, 15.0), (17.0, 16.0))
        with_size = postprocess.picked_locs(
            locs, info, [pick], pick_shape="Box", pick_size=99.0
        )[0]
        without = postprocess.picked_locs(
            locs, info, [pick], pick_shape="Box"
        )[0]
        assert len(with_size) == len(without)

    def test_brush_pick_within_the_painted_region(self, locs, info):
        stroke = (1.0, [(14.0, 15.5), (17.0, 15.5)])
        out = postprocess.picked_locs(
            locs, info, [[stroke]], pick_shape="Brush"
        )[0]
        assert len(out) > 0
        # every loc is within half the stroke width of the path
        assert (np.abs(out["y"] - 15.5) <= 0.5 + 1e-9).all()
        assert (out["x"] > 13.5).all() and (out["x"] < 17.5).all()

    def test_brush_pick_width_changes_what_is_picked(self, locs, info):
        path = [(14.0, 15.5), (17.0, 15.5)]
        wide = postprocess.picked_locs(
            locs, info, [[(2.0, path)]], pick_shape="Brush"
        )[0]
        narrow = postprocess.picked_locs(
            locs, info, [[(0.2, path)]], pick_shape="Brush"
        )[0]
        assert len(wide) > len(narrow) > 0

    def test_brush_pick_unions_its_strokes(self, locs, info):
        a = (1.0, [(5.0, 5.5), (6.0, 5.5)])
        b = (1.0, [(15.0, 15.5), (16.0, 15.5)])
        separate = postprocess.picked_locs(
            locs, info, [[a], [b]], pick_shape="Brush"
        )
        together = postprocess.picked_locs(
            locs, info, [[a, b]], pick_shape="Brush"
        )
        assert len(together) == 1
        assert len(together[0]) == len(separate[0]) + len(separate[1])

    def test_brush_pick_ignores_pick_size(self, locs, info):
        pick = [(1.0, [(14.0, 15.5), (17.0, 15.5)])]
        with_size = postprocess.picked_locs(
            locs, info, [pick], pick_shape="Brush", pick_size=99.0
        )[0]
        without = postprocess.picked_locs(
            locs, info, [pick], pick_shape="Brush"
        )[0]
        assert len(with_size) == len(without)

    def test_brush_pick_adds_group(self, locs, info):
        picks = [
            [(1.0, [(5.0, 5.5), (6.0, 5.5)])],
            [(1.0, [(15.0, 15.5), (16.0, 15.5)])],
        ]
        out = postprocess.picked_locs(locs, info, picks, pick_shape="Brush")
        assert out[0]["group"].unique().tolist() == [0]
        assert out[1]["group"].unique().tolist() == [1]

    def test_box_pick_adds_group(self, locs, info):
        picks = [
            ((5.0, 5.0), (6.0, 6.0)),
            ((15.0, 15.0), (16.0, 16.0)),
        ]
        out = postprocess.picked_locs(locs, info, picks, pick_shape="Box")
        assert len(out) == 2
        assert out[0]["group"].unique().tolist() == [0]
        assert out[1]["group"].unique().tolist() == [1]


def _matches_origamis(new_picks, tolerance=0.5):
    """Return True if the found picks map one-to-one onto the known
    origami positions."""
    if len(new_picks) != len(ORIGAMI_PICKS):
        return False
    found = np.array([(_[0], _[1]) for _ in new_picks])
    truth = np.array(ORIGAMI_PICKS)
    distances = np.hypot(
        found[:, None, 0] - truth[None, :, 0],
        found[:, None, 1] - truth[None, :, 1],
    )
    closest = distances.argmin(axis=1)
    return (
        len(set(closest.tolist())) == len(ORIGAMI_PICKS)
        and distances.min(axis=1).max() < tolerance
    )


# Circular pick_similar output for the bundled test data, recorded
# before the function was extended to squares and rectangles. Guards
# against the refactoring silently changing the circular path.
CIRCULAR_PICK_SIMILAR = np.array(
    [
        [5.5, 5.5],
        [5.5, 15.5],
        [5.471064897017046, 25.492258522727273],
        [15.521154952375856, 5.451117946677012],
        [15.500356998083726, 15.521690512603184],
        [15.524771372477213, 25.543294270833332],
        [25.496572921525186, 5.510046660010494],
        [25.468092395413308, 15.52528824344758],
        [25.53900722287736, 25.427370467275942],
    ]
)


class TestPickSimilar:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Circle", PICK_SIZE, std_range=123.0
        )
        assert _matches_origamis(new_picks)

    def test_circular_result_unchanged(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Circle", PICK_SIZE, std_range=123.0
        )
        # the pick centers are centers of mass of float32 coordinates, so
        # the last digits depend on the platform's summation order; the
        # tolerance is still orders of magnitude below any real change
        np.testing.assert_allclose(
            np.array(new_picks), CIRCULAR_PICK_SIMILAR, rtol=1e-6, atol=1e-6
        )

    def test_precomputed_index_blocks_path(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            seed_picks,
            "Circle",
            PICK_SIZE,
            std_range=123.0,
            index_blocks=ib,
        )
        assert _matches_origamis(new_picks)

    def test_empty_picks_returns_empty(self, locs, info):
        for shape in ("Circle", "Square", "Rectangle"):
            assert (
                postprocess.pick_similar(locs, info, [], shape, PICK_SIZE)
                == []
            )

    def test_polygon_is_rejected(self, locs, info):
        with pytest.raises(AssertionError):
            postprocess.pick_similar(
                locs, info, [[(1.0, 1.0), (2.0, 2.0)]], "Polygon"
            )

    def test_grid_spacing_only_for_rectangles(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.pick_similar(
                locs,
                info,
                [[5.5, 5.5], [5.5, 15.5]],
                "Circle",
                PICK_SIZE,
                grid_spacing=1.0,
            )


class TestPickSimilarSquare:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Square", PICK_SIZE, std_range=123.0
        )
        assert _matches_origamis(new_picks)

    def test_found_squares_do_not_overlap(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Square", PICK_SIZE, std_range=123.0
        )
        found = np.array(new_picks)
        for i, (x, y) in enumerate(found):
            others = np.delete(found, i, axis=0)
            chebyshev = np.maximum(
                np.abs(others[:, 0] - x), np.abs(others[:, 1] - y)
            )
            assert (chebyshev > PICK_SIZE).all()

    def test_precomputed_index_blocks_path(self, locs, info):
        # circular index blocks of diameter a are valid for squares of
        # side a: both reach at most a / 2 in x and y
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            seed_picks,
            "Square",
            PICK_SIZE,
            std_range=123.0,
            index_blocks=ib,
        )
        assert _matches_origamis(new_picks)

    def test_wrong_sized_index_blocks_are_rebuilt(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        wrong = postprocess.get_index_blocks(locs, info, 5.0)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            seed_picks,
            "Square",
            PICK_SIZE,
            std_range=123.0,
            index_blocks=wrong,
        )
        assert _matches_origamis(new_picks)


# ---------------------------------------------------------------------------
# Pick similar for boxes
# ---------------------------------------------------------------------------


def _box(cx, cy, w, h):
    """Return a box pick of the given size, centered at (cx, cy)."""
    return ((cx - w / 2, cy - h / 2), (cx + w / 2, cy + h / 2))


def _box_centers(picks):
    """Return the centers of box picks as an (n, 2) array."""
    return np.array(
        [(0.5 * (p[0][0] + p[1][0]), 0.5 * (p[0][1] + p[1][1])) for p in picks]
    )


class TestPickSimilarBox:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = [
            _box(5.5, 5.5, PICK_SIZE, PICK_SIZE),
            _box(5.5, 15.5, PICK_SIZE, PICK_SIZE),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Box", std_range=123.0
        )
        assert _matches_origamis(_box_centers(new_picks))

    def test_matches_square_of_the_same_side(self, locs, info):
        # a box of equal sides must behave exactly like a square
        centers = [[5.5, 5.5], [5.5, 15.5]]
        squares = postprocess.pick_similar(
            locs, info, centers, "Square", PICK_SIZE, std_range=123.0
        )
        boxes = postprocess.pick_similar(
            locs,
            info,
            [_box(x, y, PICK_SIZE, PICK_SIZE) for x, y in centers],
            "Box",
            std_range=123.0,
        )
        np.testing.assert_allclose(
            np.sort(_box_centers(boxes), axis=0),
            np.sort(np.array(squares), axis=0),
            atol=1e-6,
        )

    def test_found_boxes_do_not_overlap(self, locs, info):
        seed_picks = [
            _box(5.5, 5.5, PICK_SIZE, PICK_SIZE),
            _box(5.5, 15.5, PICK_SIZE, PICK_SIZE),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Box", std_range=123.0
        )
        found = _box_centers(new_picks)
        for i, (x, y) in enumerate(found):
            others = np.delete(found, i, axis=0)
            chebyshev = np.maximum(
                np.abs(others[:, 0] - x), np.abs(others[:, 1] - y)
            )
            assert (chebyshev > PICK_SIZE).all()

    def test_new_picks_take_the_median_size(self, locs, info):
        # a 2:1 aspect ratio must survive into every found pick
        seed_picks = [
            _box(5.5, 5.5, 2.0, 1.0),
            _box(5.5, 15.5, 2.0, 1.0),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Box", std_range=123.0
        )
        assert len(new_picks) > len(seed_picks)
        for (x0, y0), (x1, y1) in new_picks[len(seed_picks) :]:
            assert x1 - x0 == pytest.approx(2.0)
            assert y1 - y0 == pytest.approx(1.0)

    def test_seed_picks_are_returned_as_drawn(self, locs, info):
        seed_picks = [
            _box(5.5, 5.5, 3.0, 1.0),  # deliberately larger than the rest
            _box(5.5, 15.5, 1.0, 1.0),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, "Box", std_range=123.0
        )
        assert list(new_picks[: len(seed_picks)]) == seed_picks

    def test_wrong_sized_index_blocks_are_rebuilt(self, locs, info):
        seed_picks = [
            _box(5.5, 5.5, PICK_SIZE, PICK_SIZE),
            _box(5.5, 15.5, PICK_SIZE, PICK_SIZE),
        ]
        wrong = postprocess.get_index_blocks(locs, info, 5.0)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            seed_picks,
            "Box",
            std_range=123.0,
            index_blocks=wrong,
        )
        assert _matches_origamis(_box_centers(new_picks))

    def test_empty_picks(self, locs, info):
        assert postprocess.pick_similar(locs, info, [], "Box") == []

    def test_brush_is_rejected(self, locs, info):
        # a painted region has no size or canonical form to replicate
        pick = [(1.0, [(5.0, 5.5), (6.0, 5.5)])]
        with pytest.raises(AssertionError):
            postprocess.pick_similar(locs, info, [pick], "Brush")


# ---------------------------------------------------------------------------
# Pick similar for rectangles
# ---------------------------------------------------------------------------

# The bundled test data has no elongated structures, so the rectangular
# path is tested against synthetic line segments.
LINE_ANGLES_DEG = [0, 15, 30, 45, 60, 75, 89, -89, -75, -60, -45, -30]
LINE_LENGTH = 8.0
LINE_WIDTH = 0.6


def _make_line_locs():
    """Return ``(locs, info, centers)`` for 12 line segments at
    different angles on a 64x64 px field of view, plus uniform
    background."""
    rng = np.random.default_rng(42)
    centers = [
        (
            8.0 + 16.0 * i + rng.uniform(-2, 2),
            8.0 + 16.0 * j + rng.uniform(-2, 2),
        )
        for i in range(4)
        for j in range(3)
    ]
    xs = []
    ys = []
    for (cx, cy), angle in zip(centers, LINE_ANGLES_DEG):
        theta = np.deg2rad(angle)
        along = rng.uniform(-LINE_LENGTH / 2, LINE_LENGTH / 2, 150)
        across = rng.normal(0, 0.15, 150)
        xs.append(cx + along * np.cos(theta) - across * np.sin(theta))
        ys.append(cy + along * np.sin(theta) + across * np.cos(theta))
    xs.append(rng.uniform(0, 64, 200))
    ys.append(rng.uniform(0, 64, 200))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    locs = pd.DataFrame(
        {
            "frame": np.arange(len(x), dtype=np.int32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": np.full(len(x), 1000.0, dtype=np.float32),
            "lpx": np.full(len(x), 0.1, dtype=np.float32),
            "lpy": np.full(len(x), 0.1, dtype=np.float32),
        }
    )
    info = [
        {
            "Width": 64,
            "Height": 64,
            "Frames": len(x),
            "Generated by": "tests.test_postprocess",
        }
    ]
    return locs, info, centers


@pytest.fixture(scope="module")
def line_data():
    return _make_line_locs()


def _line_pick(centers, index):
    """Return the ground-truth rectangular pick for one line."""
    cx, cy = centers[index]
    theta = np.deg2rad(LINE_ANGLES_DEG[index])
    half_x = 0.5 * LINE_LENGTH * np.cos(theta)
    half_y = 0.5 * LINE_LENGTH * np.sin(theta)
    return (
        (cx - half_x, cy - half_y),
        (cx + half_x, cy + half_y),
    )


def _as_center_angle(picks):
    """Convert rectangular picks to ``(center_x, center_y, angle_deg,
    length)``."""
    out = []
    for (x_start, y_start), (x_end, y_end) in picks:
        out.append(
            (
                0.5 * (x_start + x_end),
                0.5 * (y_start + y_end),
                np.degrees(np.arctan2(y_end - y_start, x_end - x_start)),
                np.hypot(x_end - x_start, y_end - y_start),
            )
        )
    return out


class TestPickSimilarRectangle:
    # three seeds spanning the natural spread of the segments; with only
    # two nearly identical seeds the standard deviation, and hence the
    # acceptance windows, collapse
    SEEDS = (0, 3, 7)
    STD_RANGE = 6.0

    def _run(self, line_data, **kwargs):
        locs, info, centers = line_data
        picks = [_line_pick(centers, i) for i in self.SEEDS]
        kwargs.setdefault("std_range", self.STD_RANGE)
        return postprocess.pick_similar(
            locs, info, picks, "Rectangle", LINE_WIDTH, **kwargs
        )

    def test_recovers_all_segments(self, line_data):
        _, _, centers = line_data
        new_picks = self._run(line_data)
        assert len(new_picks) == len(LINE_ANGLES_DEG)
        found = _as_center_angle(new_picks)
        for (cx, cy), angle in zip(centers, LINE_ANGLES_DEG):
            distances = [np.hypot(cx - f[0], cy - f[1]) for f in found]
            nearest = found[int(np.argmin(distances))]
            assert min(distances) < 1.0
            # the pick axis is a director, so compare modulo 180 deg
            d_angle = (nearest[2] - angle + 90) % 180 - 90
            assert abs(d_angle) < 5.0

    def test_steep_angles_are_not_merged(self, line_data):
        # the +89 deg and -89 deg segments differ by 2 deg, not 178 deg
        _, _, centers = line_data
        new_picks = self._run(line_data)
        found = _as_center_angle(new_picks)
        for index in (6, 7):
            cx, cy = centers[index]
            matches = [
                f for f in found if np.hypot(cx - f[0], cy - f[1]) < 1.0
            ]
            assert len(matches) == 1

    def test_found_rectangles_do_not_overlap(self, line_data):
        new_picks = self._run(line_data)
        found = _as_center_angle(new_picks)
        for i, (x1, y1, angle1, length1) in enumerate(found):
            for x2, y2, angle2, length2 in found[i + 1 :]:
                assert not lib.rectangles_overlap(
                    x1,
                    y1,
                    np.deg2rad(angle1),
                    length1,
                    LINE_WIDTH,
                    0.5 * np.hypot(length1, LINE_WIDTH),
                    x2,
                    y2,
                    np.deg2rad(angle2),
                    length2,
                    LINE_WIDTH,
                    0.5 * np.hypot(length2, LINE_WIDTH),
                )

    def test_new_picks_take_the_median_seed_length(self, line_data):
        _, _, centers = line_data
        picks = [_line_pick(centers, i) for i in self.SEEDS]
        new_picks = self._run(line_data)
        # the input picks are returned exactly as drawn
        assert new_picks[: len(picks)] == picks
        expected = np.median([_[3] for _ in _as_center_angle(picks)])
        for _, _, _, length in _as_center_angle(new_picks[len(picks) :]):
            assert length == pytest.approx(expected)

    def test_deterministic(self, line_data):
        first = np.array(self._run(line_data))
        second = np.array(self._run(line_data))
        np.testing.assert_array_equal(first, second)

    def test_coarser_grid_spacing_still_finds_segments(self, line_data):
        new_picks = self._run(line_data, grid_spacing=LINE_LENGTH / 2)
        assert len(new_picks) >= len(LINE_ANGLES_DEG) - 1

    def test_isotropic_decoy_is_rejected(self, line_data):
        # a dense blob has as many localizations inside a rectangle as a
        # line does, but fails both anisotropic RMSD windows
        locs, info, centers = line_data
        rng = np.random.default_rng(7)
        n_decoy = 1500
        decoy = pd.DataFrame(
            {
                "frame": np.arange(n_decoy, dtype=np.int32) + 10**6,
                "x": rng.normal(40.0, 0.7, n_decoy).astype(np.float32),
                "y": rng.normal(55.0, 0.7, n_decoy).astype(np.float32),
                "photons": np.full(n_decoy, 1000.0, dtype=np.float32),
                "lpx": np.full(n_decoy, 0.1, dtype=np.float32),
                "lpy": np.full(n_decoy, 0.1, dtype=np.float32),
            }
        )
        with_decoy = pd.concat([locs, decoy], ignore_index=True)
        picks = [_line_pick(centers, i) for i in self.SEEDS]
        new_picks = postprocess.pick_similar(
            with_decoy,
            info,
            picks,
            "Rectangle",
            LINE_WIDTH,
            std_range=self.STD_RANGE,
        )
        assert len(new_picks) == len(LINE_ANGLES_DEG)
        for cx, cy, _, _ in _as_center_angle(new_picks):
            assert np.hypot(cx - 40.0, cy - 55.0) > 3.0

    def test_single_seed_does_not_crash(self, line_data):
        locs, info, centers = line_data
        new_picks = postprocess.pick_similar(
            locs,
            info,
            [_line_pick(centers, 0)],
            "Rectangle",
            LINE_WIDTH,
            std_range=2.0,
        )
        assert np.isfinite(np.array(new_picks, dtype=float)).all()

    def test_empty_pick_raises(self, line_data):
        # a pick drawn on empty space cannot define the criteria
        locs, info, _ = line_data
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError):
                postprocess.pick_similar(
                    locs,
                    info,
                    [((62.0, 62.0), (63.0, 63.0))],
                    "Rectangle",
                    0.01,
                )


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

    def test_box_pick_size_ignored(self, locs, info):
        box = ((14.5, 14.5), (16.5, 16.5))
        n_inside = len(
            postprocess.picked_locs(locs, info, [box], pick_shape="Box")[0]
        )
        assert n_inside > 0
        # a box carries its own extent, so pick_size is not required
        out = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=[box],
            pick_shape="Box",
        )
        assert len(out) == len(locs) - n_inside

    def test_brush_pick_size_ignored(self, locs, info):
        pick = [(1.5, [(14.5, 15.5), (16.5, 15.5)])]
        n_inside = len(
            postprocess.picked_locs(locs, info, [pick], pick_shape="Brush")[0]
        )
        assert n_inside > 0
        # a brush stroke carries its own width, so no pick_size
        out = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=[pick],
            pick_shape="Brush",
        )
        assert len(out) == len(locs) - n_inside

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
        bin_centers, dnfl = (
            postprocess._next_frame_neighbor_distance_histogram(locs.copy())
        )
        assert bin_centers.shape == dnfl.shape
        assert (dnfl >= 0).all()
        # bin centers are evenly spaced
        diffs = np.diff(bin_centers)
        assert np.allclose(diffs, diffs[0])

    def test_some_neighbors_present(self, locs):
        _, dnfl = postprocess._next_frame_neighbor_distance_histogram(
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

    def test_given_lp_skips_nena(self, locs, info, monkeypatch):
        viewport = ((15, 15), (16, 16))
        expected = postprocess.frc(locs, info, viewport=viewport)
        lp = postprocess.nena(locs, info)[1]

        def fail(*args, **kwargs):
            raise AssertionError("nena must not run when lp is given")

        monkeypatch.setattr(postprocess, "nena", fail)
        frc_res = postprocess.frc(locs, info, viewport=viewport, lp=lp)
        assert frc_res["resolution"] == pytest.approx(expected["resolution"])

    def test_does_not_modify_locs(self, locs, info):
        ref = locs.copy()
        postprocess.frc(locs, info, viewport=((15, 15), (16, 16)))
        pd.testing.assert_frame_equal(locs, ref)

    def test_plot_without_resolution(self, locs, info):
        frc_res = postprocess.frc(locs, info, viewport=((15, 15), (16, 16)))
        frc_res["resolution"] = None
        fig = postprocess.plot_frc(frc_res)
        assert "n/a" in fig.axes[0].get_title()


class TestFrcRois:
    def test_results_per_roi(self, locs, info):
        viewport = ((10, 10), (20, 20))
        ref = locs.copy()
        progress = []
        result = postprocess.frc_rois(
            locs,
            info,
            viewport,
            n_rois=4,
            roi_size=260,  # 2 camera pixels
            min_locs=20,
            callback=progress.append,
        )
        n = len(result["rois"])
        assert 0 < n <= 4
        assert len(result["frc_results"]) == n
        assert result["resolutions"].shape == (n,)
        assert (result["n_locs"] >= 20).all()
        assert all("images" not in _ for _ in result["frc_results"])
        assert progress[-1] == n
        pd.testing.assert_frame_equal(locs, ref)


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
        lg = postprocess._get_link_groups(frame, x, y, 0.05, 3, group)
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
        lg = postprocess._get_link_groups(frame, x, y, 1e-9, 1, group)
        assert len(np.unique(lg)) == len(sl)


class TestBindingEventCores:
    def test_drops_border_locs_of_each_event(self):
        # Two binding events at distinct positions: one 4 frames long,
        # one 2 frames long (too short to have a core).
        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1, 2, 3, 0, 1], dtype=np.int32),
                "x": np.array([1, 1, 1, 1, 50, 50], dtype=np.float32),
                "y": np.array([1, 1, 1, 1, 50, 50], dtype=np.float32),
            }
        )
        out = postprocess.select_binding_event_cores(locs, r_max=0.05)
        # only frames 1 and 2 of the first event survive
        assert out["frame"].tolist() == [1, 2]
        assert out["group"].tolist() == [0, 0]

    def test_min_n_locs_discards_short_events(self):
        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1, 2, 3], dtype=np.int32),
                "x": np.ones(4, dtype=np.float32),
                "y": np.ones(4, dtype=np.float32),
            }
        )
        out = postprocess.select_binding_event_cores(
            locs, r_max=0.05, min_n_locs=5
        )
        assert len(out) == 0

    def test_groups_are_consecutive(self, locs):
        out = postprocess.select_binding_event_cores(locs.copy())
        assert len(out) < len(locs)
        groups = np.unique(out["group"].to_numpy())
        assert (groups == np.arange(len(groups))).all()
        assert "group_input" not in out.columns

    def test_existing_group_preserved_as_group_input(self, locs):
        sub = locs.copy()
        # two input groups, split spatially so events cannot span both
        sub["group"] = (sub["x"].to_numpy() > sub["x"].median()).astype(
            np.int32
        )
        out = postprocess.select_binding_event_cores(sub)
        assert "group_input" in out.columns
        assert set(np.unique(out["group_input"].to_numpy())) <= {0, 1}
        # binding events must not span two input groups
        assert (out.groupby("group")["group_input"].nunique() == 1).all()

    def test_empty_locs(self, locs):
        out = postprocess.select_binding_event_cores(locs.iloc[0:0].copy())
        assert len(out) == 0
        assert "group" in out.columns


class TestDarkTimes:
    def test_dark_times_min_positive(self, locs, info):
        linked = postprocess.link(locs.copy(), info)
        dt = postprocess.dark_times(linked)
        assert dt.shape == (len(linked),)
        # -1 sentinel for events not preceded by another in the group;
        # all others count the frames without signal (0 = no gap).
        assert ((dt >= 0) | (dt == -1)).all()

    def test_dark_times_count_off_frames(self):
        # frames 3-4 on, 5 off, 6 on (issue example): one dark frame
        locs = pd.DataFrame(
            {"frame": np.array([3, 6], dtype=np.uint32), "len": [2, 1]}
        )
        assert postprocess.dark_times(locs).tolist() == [-1, 1]
        # unlinked events in consecutive frames (e.g., separated by more
        # than the linking radius): no dark frame
        locs["frame"] = np.array([3, 5], dtype=np.uint32)
        assert postprocess.dark_times(locs).tolist() == [-1, 0]

    def test_dark_times_closest_preceding_event(self):
        # unsorted input, one overlapping event (ignored)
        locs = pd.DataFrame(
            {
                "frame": np.array([20, 0, 10, 12], dtype=np.uint32),
                "len": [1, 5, 5, 1],
            }
        )
        # 20 <- 10-14 (5 dark), 10 <- 0-4 (5 dark), 12 overlaps 10-14
        # so it follows 0-4 (7 dark)
        assert postprocess.dark_times(locs).tolist() == [5, -1, 5, 7]

    def test_compute_dark_times_drops_unpreceded(self):
        locs = pd.DataFrame(
            {"frame": np.array([3, 6], dtype=np.uint32), "len": [2, 1]}
        )
        out = postprocess.compute_dark_times(locs)
        assert out["dark"].tolist() == [1]

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

    def test_brush_picks_need_no_size(self, locs, info, origami_picks):
        picks = [
            [(PICK_SIZE, [(x - 0.2, y), (x + 0.2, y)])]
            for x, y in origami_picks
        ]
        combined = postprocess.combine_locs_in_picks(
            locs.copy(),
            info,
            picks=picks,
            pick_shape="Brush",
        )
        assert len(combined) >= len(picks)

    def test_box_picks_need_no_size(self, locs, info, origami_picks):
        boxes = [
            (
                (x - PICK_SIZE / 2, y - PICK_SIZE / 2),
                (x + PICK_SIZE / 2, y + PICK_SIZE / 2),
            )
            for x, y in origami_picks
        ]
        combined = postprocess.combine_locs_in_picks(
            locs.copy(),
            info,
            picks=boxes,
            pick_shape="Box",
        )
        assert len(combined) == len(boxes)


class TestPickKinetics:
    def test_per_pick_arrays_and_out_locs(self, locs, info, origami_picks):
        pl = postprocess.picked_locs(
            locs,
            info,
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        length, dark, no_locs, out_locs, kept = postprocess.pick_kinetics(
            pl, info, max_dark_time=3
        )
        # All returned arrays are 1D and aligned in length: one entry
        # per successfully-evaluated pick (picks where kinetics could
        # not be estimated are silently dropped).
        assert length.ndim == dark.ndim == no_locs.ndim == kept.ndim == 1
        assert length.shape == dark.shape == no_locs.shape == kept.shape
        assert length.shape[0] <= len(origami_picks)
        # ``kept`` indexes back into the picks that were passed in.
        assert len(set(kept.tolist())) == len(kept)
        assert kept.min() >= 0
        assert kept.max() < len(pl)
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
        from picasso import lib

        oversampling = 1
        n_pixel_y = int(np.ceil(oversampling * info[0]["Height"]))
        n_pixel_x = int(np.ceil(oversampling * info[0]["Width"]))
        assert segs.shape == (n_seg, n_pixel_y, n_pixel_x)
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

    def test_undrift_from_fiducials_honours_pick_shape(
        self, locs, info, origami_picks
    ):
        # a box pick must be picked as a box, not silently as a circle:
        # a wide, flat box and a circle over the same center enclose
        # different localizations, so the drift they report differs
        boxes = [
            ((x - 1.0, y - 0.2), (x + 1.0, y + 0.2)) for x, y in origami_picks
        ]
        _, new_info, box_drift = postprocess.undrift_from_fiducials(
            locs.copy(),
            info,
            picks=boxes,
            pick_shape="Box",
            undrift_z=False,
        )
        assert new_info[-1]["Pick shape"] == "Box"
        # boxes carry their own extent, so no radius is reported
        assert "Pick radius (nm)" not in new_info[-1]

        _, _, circle_drift = postprocess.undrift_from_fiducials(
            locs.copy(),
            info,
            picks=origami_picks,
            pick_size=PICK_SIZE / 2,
            undrift_z=False,
        )
        assert not np.allclose(
            box_drift["x"].to_numpy(), circle_drift["x"].to_numpy()
        )

    def test_undrift_from_fiducials_box_needs_no_size(
        self, locs, info, origami_picks
    ):
        boxes = [
            ((x - 0.75, y - 0.75), (x + 0.75, y + 0.75))
            for x, y in origami_picks
        ]
        _, _, drift = postprocess.undrift_from_fiducials(
            locs.copy(),
            info,
            picks=boxes,
            pick_shape="Box",
            pick_size=None,
            undrift_z=False,
        )
        assert len(drift) == info[0]["Frames"]


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
        assert abs(residual) < 0.05

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

    def test_align_from_picked_box_recovers_known_shift(
        self, locs_copy, info, origami_picks
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1
        boxes = [
            (
                (x - PICK_SIZE / 2, y - PICK_SIZE / 2),
                (x + PICK_SIZE / 2, y + PICK_SIZE / 2),
            )
            for x, y in origami_picks
        ]
        _, shifts = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=boxes,
            pick_shape="Box",
            return_shifts=True,
        )
        assert abs(shifts[1][1] - 0.1) < 0.05

    def test_align_from_picked_brush_recovers_known_shift(
        self, locs_copy, info, origami_picks
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1
        picks = [
            [(PICK_SIZE, [(x - 0.2, y), (x + 0.2, y)])]
            for x, y in origami_picks
        ]
        _, shifts = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=picks,
            pick_shape="Brush",
            return_shifts=True,
        )
        assert abs(shifts[1][1] - 0.1) < 0.05

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
        out = clusterer.dbscan(locs, radius=2 / 130, min_samples=2)[0]
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

    def test_channels_with_different_columns_keep_all_centers(
        self, locs, info, tmp_path
    ):
        """E.g. a wavelet-identified channel (no ``net_gradient``) next to a
        net gradient one: every center survives the save, which drops rows
        with NaN."""
        assert "net_gradient" in locs.columns
        no_ng = locs.drop(columns="net_gradient")
        path = str(tmp_path / "resi.hdf5")
        with pytest.warns(UserWarning, match="net_gradient"):
            out, _ = postprocess.resi(
                [locs.copy(), no_ng],
                [info, info],
                radius_xy=2 / 130,
                min_locs=2,
                resi_path=path,
            )
        assert "net_gradient" not in out.columns
        assert set(out["resi_channel_id"]) == {0, 1}
        saved, _ = io.load_locs(path)
        assert len(saved) == len(out)

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
