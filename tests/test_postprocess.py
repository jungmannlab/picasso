"""Test ``picasso.postprocess``.

Most fixtures (``locs``, ``info``) live in ``tests/conftest.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

from picasso import io, lib, postprocess

from tests.conftest import PIXELSIZE
from tests.test_postprocess_progress import RecordingProgress, assert_phases

# nm per camera pixel of the bundled test data; the test geometry was
# designed in camera pixels and is expressed in nm through it
PX = PIXELSIZE

# Reused parameters
PICK_SIZE = 1.5 * PX  # nm


# 9-pick grid covering each origami in the bundled test movie (nm)
ORIGAMI_PICKS = [
    [x * PX, y * PX] for x in (5.5, 15.5, 25.5) for y in (5.5, 15.5, 25.5)
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
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE / 2,
        )
        via_pyramid = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
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
        size = 2.0 * PX
        n_y, n_x = postprocess._index_blocks_shape(info, size)
        width, height = lib.field_of_view(info)
        assert n_y == int(np.ceil(height / size))
        assert n_x == int(np.ceil(width / size))

    def test_get_block_locs_at_returns_some_locs(self, locs, info):
        r = PICK_SIZE / 2
        index_blocks = postprocess.get_index_blocks(locs, info, r)
        locs_sorted, size, _, _, block_starts, block_ends, K, L = index_blocks
        locs_xy = locs_sorted[["x", "y"]].to_numpy().T
        x, y = 15.5 * PX, 15.5 * PX
        locs_at = postprocess.get_block_locs_at(
            int(x / r), int(y / r), locs_xy, block_starts, block_ends, K, L
        )
        assert locs_at.shape[1] > 0
        # Block lookup is conservative — within ~PICK_SIZE
        d = np.hypot(locs_at[0] - x, locs_at[1] - y)
        assert (d < 2 * PICK_SIZE).all()

    def test_n_block_locs_at_matches_get_block_locs_at(self, locs, info):
        ib = postprocess.get_index_blocks(locs, info, 1.0 * PX)
        _, _, _, _, b_starts, b_ends, K, L = ib
        locs_xy = ib[0][["x", "y"]].to_numpy().T
        # every cell, including the first and last rows and columns of
        # the grid, counts the localizations that get_block_locs_at
        # returns there
        for y_idx in range(K):
            for x_idx in range(L):
                n = postprocess._n_block_locs_at(
                    x_idx, y_idx, K, L, b_starts, b_ends
                )
                expected = (
                    b_ends[
                        max(y_idx - 1, 0) : y_idx + 2,
                        max(x_idx - 1, 0) : x_idx + 2,
                    ].sum()
                    - b_starts[
                        max(y_idx - 1, 0) : y_idx + 2,
                        max(x_idx - 1, 0) : x_idx + 2,
                    ].sum()
                )
                assert int(n) == int(expected)
                if n:
                    at = postprocess.get_block_locs_at(
                        x_idx, y_idx, locs_xy, b_starts, b_ends, K, L
                    )
                    assert at.shape[1] == int(n)

    def test_n_block_locs_at_off_grid_is_zero(self):
        # a position whose 3 x 3 neighborhood lies entirely off the grid
        # counts nothing
        starts = np.zeros((2, 2), dtype=np.uint32)
        ends = np.full((2, 2), 3, dtype=np.uint32)
        assert postprocess._n_block_locs_at(5, 5, 2, 2, starts, ends) == 0
        # the first row and column count
        assert postprocess._n_block_locs_at(0, 0, 2, 2, starts, ends) == 12


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------


class TestPickedLocs:
    def test_one_list_per_pick(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
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
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        radius = PICK_SIZE / 2
        for (cx, cy), p in zip(origami_picks, picked):
            d = np.hypot(p["x"] - cx, p["y"] - cy)
            assert (d <= radius + 1e-6).all()

    def test_add_group_assigns_unique_ids(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        for i, p in enumerate(picked):
            assert (p["group"] == i).all()

    def test_add_group_false_omits_group(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            add_group=False,
        )
        assert "group" not in picked[0].columns

    def test_picked_locs_sorted_by_frame(self, locs, info, origami_picks):
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        for p in picked:
            f = p["frame"].to_numpy()
            assert (np.diff(f) >= 0).all()

    def test_empty_picks_returns_empty_list(self, locs, info):
        out = postprocess.picked_locs(
            locs,
            info,
            picks=[],
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        assert out == []

    def test_invalid_shape_raises(self, locs, info, origami_picks):
        with pytest.raises(ValueError, match="Unrecognized pick shape"):
            postprocess.picked_locs(
                locs,
                info,
                picks=origami_picks,
                pick_shape="Hexagon",
                pick_size=PICK_SIZE,
            )

    def test_precomputed_index_blocks_matches_internal(
        self, locs, info, origami_picks
    ):
        # ``_picked_circular_locs`` uses the pick radius as the index-block
        # size, so the precomputed index_blocks must use the same size for
        # the two paths to be equivalent.
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        a = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        b = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            index_blocks=ib,
        )
        assert len(a) == len(b)
        for pa, pb in zip(a, b):
            assert len(pa) == len(pb)

    def test_square_pick_within_bounds(self, locs, info):
        side = 2.0 * PX
        cx, cy = 15.5 * PX, 15.5 * PX
        out = postprocess.picked_locs(
            locs,
            info,
            picks=[(cx, cy)],
            pick_shape="Square",
            pick_size=side,
        )[0]
        assert len(out) > 0
        assert (out["x"] > cx - side / 2).all()
        assert (out["x"] < cx + side / 2).all()
        assert (out["y"] > cy - side / 2).all()
        assert (out["y"] < cy + side / 2).all()

    def test_rectangle_pick_returns_locs(self, locs, info):
        # A rectangle 2 camera pixels wide around the line from (5, 5)
        # to (8, 5) camera pixels
        out = postprocess.picked_locs(
            locs,
            info,
            picks=[((5.0 * PX, 5.0 * PX), (8.0 * PX, 5.0 * PX))],
            pick_shape="Rectangle",
            pick_size=2.0 * PX,
        )[0]
        assert len(out) > 0
        # rotation columns added by the rectangular helper
        assert "x_pick_rot" in out.columns
        assert "y_pick_rot" in out.columns

    def test_polygon_pick_returns_locs(self, locs, info):
        # Closed polygon (first vertex repeated) around (15.5, 15.5)
        # camera pixels
        polygon = [
            (14.5 * PX, 14.5 * PX),
            (16.5 * PX, 14.5 * PX),
            (16.5 * PX, 16.5 * PX),
            (14.5 * PX, 16.5 * PX),
            (14.5 * PX, 14.5 * PX),
        ]
        out = postprocess.picked_locs(
            locs,
            info,
            picks=[polygon],
            pick_shape="Polygon",
        )[0]
        assert len(out) > 0
        assert (out["x"] > 14.5 * PX).all() and (out["x"] < 16.5 * PX).all()
        assert (out["y"] > 14.5 * PX).all() and (out["y"] < 16.5 * PX).all()

    def test_box_pick_within_bounds(self, locs, info):
        # 3 wide, 1 high, so a square test would give a different answer
        pick = ((14.0 * PX, 15.0 * PX), (17.0 * PX, 16.0 * PX))
        out = postprocess.picked_locs(
            locs,
            info,
            picks=[pick],
            pick_shape="Box",
        )[0]
        assert len(out) > 0
        assert (out["x"] > 14.0 * PX).all() and (out["x"] < 17.0 * PX).all()
        assert (out["y"] > 15.0 * PX).all() and (out["y"] < 16.0 * PX).all()

    def test_box_pick_ignores_corner_order(self, locs, info):
        corners = ((14.0 * PX, 15.0 * PX), (17.0 * PX, 16.0 * PX))
        ordered = postprocess.picked_locs(
            locs, info, picks=[corners], pick_shape="Box"
        )[0]
        reversed_ = postprocess.picked_locs(
            locs, info, picks=[corners[::-1]], pick_shape="Box"
        )[0]
        assert len(ordered) == len(reversed_)

    def test_box_pick_ignores_pick_size(self, locs, info):
        pick = ((14.0 * PX, 15.0 * PX), (17.0 * PX, 16.0 * PX))
        with_size = postprocess.picked_locs(
            locs, info, picks=[pick], pick_shape="Box", pick_size=99.0
        )[0]
        without = postprocess.picked_locs(
            locs, info, picks=[pick], pick_shape="Box"
        )[0]
        assert len(with_size) == len(without)

    def test_brush_pick_within_the_painted_region(self, locs, info):
        stroke = (1.0 * PX, [(14.0 * PX, 15.5 * PX), (17.0 * PX, 15.5 * PX)])
        out = postprocess.picked_locs(
            locs, info, picks=[[stroke]], pick_shape="Brush"
        )[0]
        assert len(out) > 0
        # every loc is within half the stroke width of the path
        assert (np.abs(out["y"] - 15.5 * PX) <= 0.5 * PX + 1e-6).all()
        assert (out["x"] > 13.5 * PX).all() and (out["x"] < 17.5 * PX).all()

    def test_brush_pick_width_changes_what_is_picked(self, locs, info):
        path = [(14.0 * PX, 15.5 * PX), (17.0 * PX, 15.5 * PX)]
        wide = postprocess.picked_locs(
            locs, info, picks=[[(2.0 * PX, path)]], pick_shape="Brush"
        )[0]
        narrow = postprocess.picked_locs(
            locs, info, picks=[[(0.2 * PX, path)]], pick_shape="Brush"
        )[0]
        assert len(wide) > len(narrow) > 0

    def test_brush_pick_unions_its_strokes(self, locs, info):
        a = (1.0 * PX, [(5.0 * PX, 5.5 * PX), (6.0 * PX, 5.5 * PX)])
        b = (1.0 * PX, [(15.0 * PX, 15.5 * PX), (16.0 * PX, 15.5 * PX)])
        separate = postprocess.picked_locs(
            locs, info, picks=[[a], [b]], pick_shape="Brush"
        )
        together = postprocess.picked_locs(
            locs, info, picks=[[a, b]], pick_shape="Brush"
        )
        assert len(together) == 1
        assert len(together[0]) == len(separate[0]) + len(separate[1])

    def test_brush_pick_ignores_pick_size(self, locs, info):
        pick = [(1.0 * PX, [(14.0 * PX, 15.5 * PX), (17.0 * PX, 15.5 * PX)])]
        with_size = postprocess.picked_locs(
            locs, info, picks=[pick], pick_shape="Brush", pick_size=99.0
        )[0]
        without = postprocess.picked_locs(
            locs, info, picks=[pick], pick_shape="Brush"
        )[0]
        assert len(with_size) == len(without)

    def test_brush_pick_adds_group(self, locs, info):
        picks = [
            [(1.0 * PX, [(5.0 * PX, 5.5 * PX), (6.0 * PX, 5.5 * PX)])],
            [(1.0 * PX, [(15.0 * PX, 15.5 * PX), (16.0 * PX, 15.5 * PX)])],
        ]
        out = postprocess.picked_locs(
            locs, info, picks=picks, pick_shape="Brush"
        )
        assert out[0]["group"].unique().tolist() == [0]
        assert out[1]["group"].unique().tolist() == [1]

    def test_box_pick_adds_group(self, locs, info):
        picks = [
            ((5.0 * PX, 5.0 * PX), (6.0 * PX, 6.0 * PX)),
            ((15.0 * PX, 15.0 * PX), (16.0 * PX, 16.0 * PX)),
        ]
        out = postprocess.picked_locs(
            locs, info, picks=picks, pick_shape="Box"
        )
        assert len(out) == 2
        assert out[0]["group"].unique().tolist() == [0]
        assert out[1]["group"].unique().tolist() == [1]


def _matches_origamis(new_picks, tolerance=0.5 * PX):
    """Return True if the found picks map one-to-one onto the known
    origami positions (within ``tolerance`` nm)."""
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


# Circular pick_similar output for the bundled test data (camera pixels,
# scaled to nm below), recorded before the function was extended to
# squares and rectangles. Guards against the refactoring silently
# changing the circular path.
CIRCULAR_PICK_SIMILAR = (
    np.array(
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
    * PX
)


class TestPickSimilar:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = ORIGAMI_PICKS[:2]
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            std_range=123.0,
        )
        assert _matches_origamis(new_picks)

    def test_circular_result_unchanged(self, locs, info):
        seed_picks = ORIGAMI_PICKS[:2]
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            std_range=123.0,
        )
        # the pick centers are centers of mass of float32 coordinates, so
        # the last digits depend on the platform's summation order and
        # the iterations stop within PICK_CONVERGENCE; the tolerance is
        # still orders of magnitude below any real change
        np.testing.assert_allclose(
            np.array(new_picks),
            CIRCULAR_PICK_SIMILAR,
            rtol=0,
            atol=2 * postprocess.PICK_CONVERGENCE,
        )

    def test_precomputed_index_blocks_path(self, locs, info):
        seed_picks = ORIGAMI_PICKS[:2]
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            std_range=123.0,
            index_blocks=ib,
        )
        assert _matches_origamis(new_picks)

    def test_empty_picks_returns_empty(self, locs, info):
        for shape in ("Circle", "Square", "Rectangle"):
            assert (
                postprocess.pick_similar(
                    locs, info, picks=[], pick_shape=shape, pick_size=PICK_SIZE
                )
                == []
            )

    def test_polygon_is_rejected(self, locs, info):
        with pytest.raises(AssertionError):
            postprocess.pick_similar(
                locs,
                info,
                picks=[[(1.0 * PX, 1.0 * PX), (2.0 * PX, 2.0 * PX)]],
                pick_shape="Polygon",
            )

    def test_grid_spacing_only_for_rectangles(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.pick_similar(
                locs,
                info,
                picks=ORIGAMI_PICKS[:2],
                pick_shape="Circle",
                pick_size=PICK_SIZE,
                grid_spacing=1.0 * PX,
            )


class TestPickSimilarCandidates:
    @pytest.fixture(scope="class")
    def candidates(self, locs, info):
        return postprocess.pick_similar_candidates(
            locs, info, pick_size=PICK_SIZE
        )

    def test_the_origamis_are_the_candidates(self, candidates):
        assert list(candidates.columns) == ["x", "y", "n_locs", "rmsd"]
        assert candidates["n_locs"].dtype == np.int64
        picks = list(zip(candidates["x"], candidates["y"]))
        assert _matches_origamis(picks)

    def test_n_locs_rmsd_and_center_of_the_localizations_within(self):
        """Clusters far narrower than the pick (no localization near its
        edge): each candidate is one cluster, at its center of mass, with
        its size and RMSD."""
        rng = np.random.default_rng(0)
        centers = [(800.0, 800.0), (2400.0, 800.0), (800.0, 2400.0)]
        sizes = [20, 50, 90]
        xy = np.concatenate(
            [rng.normal(c, 10.0, (n, 2)) for c, n in zip(centers, sizes)]
        )
        locs = pd.DataFrame(
            {
                "frame": np.arange(len(xy), dtype=np.uint32),
                "x": xy[:, 0].astype(np.float32),
                "y": xy[:, 1].astype(np.float32),
            }
        )
        info = [{"Frames": 160, "Width": 32, "Height": 32, "Pixelsize": 130.0}]
        candidates = postprocess.pick_similar_candidates(
            locs, info, pick_size=300.0
        )
        assert len(candidates) == 3
        xy = locs[["x", "y"]].to_numpy(np.float64)
        for start, n in zip(np.cumsum([0] + sizes[:-1]), sizes):
            cluster = xy[start : start + n]
            com = cluster.mean(0)
            (row,) = candidates[
                np.hypot(candidates["x"] - com[0], candidates["y"] - com[1])
                < 1.0
            ].itertuples()
            assert (row.x, row.y) == pytest.approx(tuple(com), abs=1e-3)
            assert row.n_locs == n
            rmsd = np.sqrt(np.mean(np.sum((cluster - com) ** 2, 1)))
            assert row.rmsd == pytest.approx(rmsd, rel=1e-5)

    def test_candidates_do_not_overlap(self, locs, info):
        # a pick size of three origami spacings: many candidates converge
        # onto the same localizations, which are kept once
        candidates = postprocess.pick_similar_candidates(
            locs, info, pick_size=2.5 * PX
        )
        xy = candidates[["x", "y"]].to_numpy()
        distances = np.hypot(*(xy[:, None] - xy[None, :]).transpose(2, 0, 1))
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 2.5 * PX

    def test_min_n_locs(self, locs, info, candidates):
        threshold = int(candidates["n_locs"].median())
        fewer = postprocess.pick_similar_candidates(
            locs, info, pick_size=PICK_SIZE, min_n_locs=threshold
        )
        assert len(fewer) < len(candidates)
        assert fewer["n_locs"].min() >= threshold
        expected = candidates[candidates["n_locs"] >= threshold]
        np.testing.assert_allclose(
            np.sort(fewer["x"]), np.sort(expected["x"]), atol=1e-6
        )

    def test_progress_over_the_grid_columns(self, locs, info):
        tracker = RecordingProgress()
        postprocess.pick_similar_candidates(
            locs, info, pick_size=PICK_SIZE, progress=tracker
        )
        assert_phases(tracker, ["Scanning candidate picks"])

    def test_the_pick_size_must_be_positive(self, locs, info):
        with pytest.raises(ValueError, match="positive"):
            postprocess.pick_similar_candidates(locs, info, pick_size=0)


class TestPickSimilarInRanges:
    def test_the_candidates_in_the_ranges_are_picked(self, locs, info):
        # scanned with the lower bound of n_locs_range
        candidates = postprocess.pick_similar_candidates(
            locs, info, pick_size=PICK_SIZE, min_n_locs=50
        )
        n_locs_range = (50, 80)
        rmsd_range = (20.0, None)
        picks = postprocess.pick_similar(
            locs,
            info,
            pick_size=PICK_SIZE,
            n_locs_range=n_locs_range,
            rmsd_range=rmsd_range,
        )
        expected = candidates.query("50 <= n_locs <= 80 and rmsd >= 20")
        assert 0 < len(expected) < len(candidates)
        assert picks == list(zip(expected["x"], expected["y"]))

    def test_smaller_candidates_do_not_exclude_a_structure(self):
        """A few localizations next to a structure, scanned first, make
        a candidate that overlaps it; scanned with the lower bound of
        n_locs_range, the structure is picked."""
        rng = np.random.default_rng(0)
        xy = np.concatenate(
            [
                rng.normal((1600.0, 1600.0), 10.0, (200, 2)),
                [(1360.0, 1597.0), (1360.0, 1600.0), (1360.0, 1603.0)],
            ]
        )
        locs = pd.DataFrame(
            {
                "frame": np.arange(len(xy), dtype=np.uint32),
                "x": xy[:, 0].astype(np.float32),
                "y": xy[:, 1].astype(np.float32),
            }
        )
        info = [
            {"Frames": 1000, "Width": 32, "Height": 32, "Pixelsize": 100.0}
        ]
        candidates = postprocess.pick_similar_candidates(
            locs, info, pick_size=300.0
        )
        # scanned from 2 localizations, the few exclude the structure
        assert candidates["n_locs"].tolist() == [3]
        picks = postprocess.pick_similar(
            locs, info, pick_size=300.0, n_locs_range=(100, None)
        )
        assert len(picks) == 1
        assert picks[0] == pytest.approx(tuple(xy[:200].mean(axis=0)), abs=1)

    def test_one_range_is_enough(self, locs, info):
        picks = postprocess.pick_similar(
            locs, info, pick_size=PICK_SIZE, rmsd_range=(0, 1e3)
        )
        assert _matches_origamis(picks)

    def test_seeds_and_ranges_are_exclusive(self, locs, info):
        with pytest.raises(ValueError, match="not both"):
            postprocess.pick_similar(
                locs,
                info,
                picks=ORIGAMI_PICKS[:2],
                pick_size=PICK_SIZE,
                n_locs_range=(10, 100),
            )

    def test_seeds_or_ranges_are_needed(self, locs, info):
        with pytest.raises(ValueError, match="seed picks"):
            postprocess.pick_similar(locs, info, pick_size=PICK_SIZE)

    @pytest.mark.parametrize("shape", ["Square", ["Circle", "Rectangle"]])
    def test_circles_only(self, locs, info, shape):
        with pytest.raises(ValueError, match="circular picks only"):
            postprocess.pick_similar(
                locs,
                info,
                pick_shape=shape,
                pick_size=PICK_SIZE,
                n_locs_range=(10, 100),
            )


def _class_candidates(seed=0):
    """Candidates of three structure classes, (n_locs, rmsd) of
    (400, 20 nm), (1500, 45 nm) and (150, 10 nm), and 15 scattered
    ones."""
    rng = np.random.default_rng(seed)
    classes = [(300, 400, 40, 20, 2), (200, 1500, 150, 45, 4)]
    classes.append((150, 150, 20, 10, 1.5))
    n_locs = [rng.normal(n, sn, size) for size, n, sn, _, _ in classes]
    rmsd = [rng.normal(r, sr, size) for size, _, _, r, sr in classes]
    n_locs.append(rng.uniform(100, 2500, 15))
    rmsd.append(rng.uniform(5, 60, 15))
    n_locs = np.concatenate(n_locs)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 1e4, len(n_locs)),
            "y": rng.uniform(0, 1e4, len(n_locs)),
            "n_locs": np.round(n_locs).astype(np.int64),
            "rmsd": np.concatenate(rmsd),
        }
    )


class TestStructureClasses:
    def test_the_classes_are_found_with_their_ranges(self):
        candidates = _class_candidates()
        labels, classes = postprocess.structure_classes(candidates)
        assert len(labels) == len(candidates)
        assert list(classes.columns) == [
            "n_candidates",
            "n_locs",
            "rmsd",
            "n_locs_min",
            "n_locs_max",
            "rmsd_min",
            "rmsd_max",
        ]
        # the medians within a standard deviation of each class
        found = classes.sort_values("n_locs")
        n_locs_off = np.abs(found["n_locs"].to_numpy() - [150, 400, 1500])
        assert np.all(n_locs_off <= [20, 40, 150])
        rmsd_off = np.abs(found["rmsd"].to_numpy() - [10, 20, 45])
        assert np.all(rmsd_off <= [1.5, 2, 4])
        for index, row in classes.iterrows():
            members = candidates[labels == index]
            assert row["n_candidates"] == len(members)
            n_median = np.median(members["n_locs"])
            n_std = members["n_locs"].std(ddof=0)
            assert row["n_locs_min"] == pytest.approx(n_median - n_std)
            assert row["n_locs_max"] == pytest.approx(n_median + n_std)
            r_median = np.median(members["rmsd"])
            r_std = members["rmsd"].std(ddof=0)
            assert row["rmsd_min"] == pytest.approx(r_median - r_std)
            assert row["rmsd_max"] == pytest.approx(r_median + r_std)

    def test_the_units_do_not_matter(self):
        """The inputs are standardized: the RMSD in micrometers gives
        the same classes."""
        candidates = _class_candidates()
        labels, _ = postprocess.structure_classes(candidates)
        in_um = candidates.assign(rmsd=candidates["rmsd"] / 1000)
        labels_um, _ = postprocess.structure_classes(in_um)
        np.testing.assert_array_equal(labels, labels_um)

    def test_std_range(self):
        labels, narrow = postprocess.structure_classes(_class_candidates())
        _, wide = postprocess.structure_classes(
            _class_candidates(), std_range=2.0
        )
        np.testing.assert_allclose(
            wide["n_locs_max"] - wide["n_locs"],
            2 * (narrow["n_locs_max"] - narrow["n_locs"]),
        )

    def test_too_few_candidates(self):
        with pytest.raises(ValueError, match="at least 2 candidates"):
            postprocess.structure_classes(_class_candidates().iloc[:1])


class TestPickSimilarSquare:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = ORIGAMI_PICKS[:2]
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Square",
            pick_size=PICK_SIZE,
            std_range=123.0,
        )
        assert _matches_origamis(new_picks)

    def test_found_squares_do_not_overlap(self, locs, info):
        seed_picks = ORIGAMI_PICKS[:2]
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Square",
            pick_size=PICK_SIZE,
            std_range=123.0,
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
        seed_picks = ORIGAMI_PICKS[:2]
        ib = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Square",
            pick_size=PICK_SIZE,
            std_range=123.0,
            index_blocks=ib,
        )
        assert _matches_origamis(new_picks)

    def test_wrong_sized_index_blocks_are_rebuilt(self, locs, info):
        seed_picks = ORIGAMI_PICKS[:2]
        wrong = postprocess.get_index_blocks(locs, info, 5.0 * PX)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Square",
            pick_size=PICK_SIZE,
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
            _box(5.5 * PX, 5.5 * PX, PICK_SIZE, PICK_SIZE),
            _box(5.5 * PX, 15.5 * PX, PICK_SIZE, PICK_SIZE),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, picks=seed_picks, pick_shape="Box", std_range=123.0
        )
        assert _matches_origamis(_box_centers(new_picks))

    def test_matches_square_of_the_same_side(self, locs, info):
        # a box of equal sides must behave exactly like a square
        centers = ORIGAMI_PICKS[:2]
        squares = postprocess.pick_similar(
            locs,
            info,
            picks=centers,
            pick_shape="Square",
            pick_size=PICK_SIZE,
            std_range=123.0,
        )
        boxes = postprocess.pick_similar(
            locs,
            info,
            picks=[_box(x, y, PICK_SIZE, PICK_SIZE) for x, y in centers],
            pick_shape="Box",
            std_range=123.0,
        )
        np.testing.assert_allclose(
            np.sort(_box_centers(boxes), axis=0),
            np.sort(np.array(squares), axis=0),
            atol=1e-6,
        )

    def test_found_boxes_do_not_overlap(self, locs, info):
        seed_picks = [
            _box(5.5 * PX, 5.5 * PX, PICK_SIZE, PICK_SIZE),
            _box(5.5 * PX, 15.5 * PX, PICK_SIZE, PICK_SIZE),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, picks=seed_picks, pick_shape="Box", std_range=123.0
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
            _box(5.5 * PX, 5.5 * PX, 2.0 * PX, 1.0 * PX),
            _box(5.5 * PX, 15.5 * PX, 2.0 * PX, 1.0 * PX),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, picks=seed_picks, pick_shape="Box", std_range=123.0
        )
        assert len(new_picks) > len(seed_picks)
        for (x0, y0), (x1, y1) in new_picks[len(seed_picks) :]:
            assert x1 - x0 == pytest.approx(2.0 * PX)
            assert y1 - y0 == pytest.approx(1.0 * PX)

    def test_seed_picks_are_returned_as_drawn(self, locs, info):
        seed_picks = [
            # deliberately larger than the rest
            _box(5.5 * PX, 5.5 * PX, 3.0 * PX, 1.0 * PX),
            _box(5.5 * PX, 15.5 * PX, 1.0 * PX, 1.0 * PX),
        ]
        new_picks = postprocess.pick_similar(
            locs, info, picks=seed_picks, pick_shape="Box", std_range=123.0
        )
        assert list(new_picks[: len(seed_picks)]) == seed_picks

    def test_wrong_sized_index_blocks_are_rebuilt(self, locs, info):
        seed_picks = [
            _box(5.5 * PX, 5.5 * PX, PICK_SIZE, PICK_SIZE),
            _box(5.5 * PX, 15.5 * PX, PICK_SIZE, PICK_SIZE),
        ]
        wrong = postprocess.get_index_blocks(locs, info, 5.0 * PX)
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=seed_picks,
            pick_shape="Box",
            std_range=123.0,
            index_blocks=wrong,
        )
        assert _matches_origamis(_box_centers(new_picks))

    def test_empty_picks(self, locs, info):
        assert (
            postprocess.pick_similar(locs, info, picks=[], pick_shape="Box")
            == []
        )

    def test_brush_is_rejected(self, locs, info):
        # a painted region has no size or canonical form to replicate
        pick = [(1.0 * PX, [(5.0 * PX, 5.5 * PX), (6.0 * PX, 5.5 * PX)])]
        with pytest.raises(AssertionError):
            postprocess.pick_similar(
                locs, info, picks=[pick], pick_shape="Brush"
            )


# ---------------------------------------------------------------------------
# Pick similar for rectangles
# ---------------------------------------------------------------------------

# The bundled test data has no elongated structures, so the rectangular
# path is tested against synthetic line segments.
LINE_ANGLES_DEG = [0, 15, 30, 45, 60, 75, 89, -89, -75, -60, -45, -30]
LINE_LENGTH = 8.0 * PX  # nm
LINE_WIDTH = 0.6 * PX  # nm


def _make_line_locs():
    """Return ``(locs, info, centers)`` for 12 line segments at
    different angles on a 64x64 camera pixel field of view (nm), plus
    uniform background."""
    rng = np.random.default_rng(42)
    centers = [
        (
            (8.0 + 16.0 * i + rng.uniform(-2, 2)) * PX,
            (8.0 + 16.0 * j + rng.uniform(-2, 2)) * PX,
        )
        for i in range(4)
        for j in range(3)
    ]
    xs = []
    ys = []
    for (cx, cy), angle in zip(centers, LINE_ANGLES_DEG):
        theta = np.deg2rad(angle)
        along = rng.uniform(-LINE_LENGTH / 2, LINE_LENGTH / 2, 150)
        across = rng.normal(0, 0.15 * PX, 150)
        xs.append(cx + along * np.cos(theta) - across * np.sin(theta))
        ys.append(cy + along * np.sin(theta) + across * np.cos(theta))
    xs.append(rng.uniform(0, 64 * PX, 200))
    ys.append(rng.uniform(0, 64 * PX, 200))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    locs = pd.DataFrame(
        {
            "frame": np.arange(len(x), dtype=np.int32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": np.full(len(x), 1000.0, dtype=np.float32),
            "lpx": np.full(len(x), 0.1 * PX, dtype=np.float32),
            "lpy": np.full(len(x), 0.1 * PX, dtype=np.float32),
        }
    )
    info = [
        {
            "Width": 64,
            "Height": 64,
            "Frames": len(x),
            "Pixelsize": PX,
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
            locs,
            info,
            picks=picks,
            pick_shape="Rectangle",
            pick_size=LINE_WIDTH,
            **kwargs,
        )

    def test_recovers_all_segments(self, line_data):
        _, _, centers = line_data
        new_picks = self._run(line_data)
        assert len(new_picks) == len(LINE_ANGLES_DEG)
        found = _as_center_angle(new_picks)
        for (cx, cy), angle in zip(centers, LINE_ANGLES_DEG):
            distances = [np.hypot(cx - f[0], cy - f[1]) for f in found]
            nearest = found[int(np.argmin(distances))]
            assert min(distances) < 1.0 * PX
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
                f for f in found if np.hypot(cx - f[0], cy - f[1]) < 1.0 * PX
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
                "x": rng.normal(40.0 * PX, 0.7 * PX, n_decoy).astype(
                    np.float32
                ),
                "y": rng.normal(55.0 * PX, 0.7 * PX, n_decoy).astype(
                    np.float32
                ),
                "photons": np.full(n_decoy, 1000.0, dtype=np.float32),
                "lpx": np.full(n_decoy, 0.1 * PX, dtype=np.float32),
                "lpy": np.full(n_decoy, 0.1 * PX, dtype=np.float32),
            }
        )
        with_decoy = pd.concat([locs, decoy], ignore_index=True)
        picks = [_line_pick(centers, i) for i in self.SEEDS]
        new_picks = postprocess.pick_similar(
            with_decoy,
            info,
            picks=picks,
            pick_shape="Rectangle",
            pick_size=LINE_WIDTH,
            std_range=self.STD_RANGE,
        )
        assert len(new_picks) == len(LINE_ANGLES_DEG)
        for cx, cy, _, _ in _as_center_angle(new_picks):
            assert np.hypot(cx - 40.0 * PX, cy - 55.0 * PX) > 3.0 * PX

    def test_single_seed_does_not_crash(self, line_data):
        locs, info, centers = line_data
        new_picks = postprocess.pick_similar(
            locs,
            info,
            picks=[_line_pick(centers, 0)],
            pick_shape="Rectangle",
            pick_size=LINE_WIDTH,
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
                    picks=[((62.0 * PX, 62.0 * PX), (63.0 * PX, 63.0 * PX))],
                    pick_shape="Rectangle",
                    pick_size=0.01 * PX,
                )


class TestRemoveLocsInPicks:
    def test_locs_in_pick_removed(self, locs, info):
        picks = [(15.5 * PX, 15.5 * PX)]
        # Reference: how many locs lie inside this pick?
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )[0]
        n_inside = len(picked)
        locs_in = locs.copy()
        out, new_info = postprocess.remove_locs_in_picks(
            locs_in,
            info,
            picks=picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        assert len(out) == len(locs) - n_inside
        # the input is not modified; the step is recorded
        pd.testing.assert_frame_equal(locs_in, locs)
        assert len(new_info) == len(info) + 1
        assert new_info[-1]["Step"] == "Remove picked localizations"
        # No remaining loc lies inside the pick
        d = np.hypot(out["x"] - 15.5 * PX, out["y"] - 15.5 * PX)
        assert (d > PICK_SIZE / 2 - 1e-6).all()

    def test_polygon_pick_size_ignored(self, locs, info):
        polygon = [
            (14.5 * PX, 14.5 * PX),
            (16.5 * PX, 14.5 * PX),
            (16.5 * PX, 16.5 * PX),
            (14.5 * PX, 16.5 * PX),
            (14.5 * PX, 14.5 * PX),
        ]
        # pick_size is asserted only for non-polygon shapes — not needed
        # here but must not raise
        out, _ = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=[polygon],
            pick_shape="Polygon",
        )
        assert len(out) < len(locs)

    def test_box_pick_size_ignored(self, locs, info):
        box = ((14.5 * PX, 14.5 * PX), (16.5 * PX, 16.5 * PX))
        picked = postprocess.picked_locs(
            locs, info, picks=[box], pick_shape="Box"
        )
        n_inside = len(picked[0])
        assert n_inside > 0
        # a box carries its own extent, so pick_size is not required
        out, _ = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=[box],
            pick_shape="Box",
        )
        assert len(out) == len(locs) - n_inside

    def test_brush_pick_size_ignored(self, locs, info):
        pick = [(1.5 * PX, [(14.5 * PX, 15.5 * PX), (16.5 * PX, 15.5 * PX)])]
        n_inside = len(
            postprocess.picked_locs(
                locs, info, picks=[pick], pick_shape="Brush"
            )[0]
        )
        assert n_inside > 0
        # a brush stroke carries its own width, so no pick_size
        out, _ = postprocess.remove_locs_in_picks(
            locs.copy(),
            info,
            picks=[pick],
            pick_shape="Brush",
        )
        assert len(out) == len(locs) - n_inside

    def test_invalid_shape_raises(self, locs, info):
        with pytest.raises(ValueError, match="Unrecognized pick shape"):
            postprocess.remove_locs_in_picks(
                locs.copy(),
                info,
                picks=[(15.5 * PX, 15.5 * PX)],
                pick_shape="Hexagon",
                pick_size=1.0,
            )


# ---------------------------------------------------------------------------
# Statistics on locs distributions
# ---------------------------------------------------------------------------


class TestDistanceHistogram:
    def test_shape_and_dtype(self, locs, info):
        dh = postprocess.distance_histogram(
            locs, info, bin_size=10.0, r_max=100.0
        )
        # 10 bins for r_max=100 nm / bin_size=10 nm
        assert dh.shape == (10,)
        assert (dh >= 0).all()

    def test_total_count_is_finite_and_positive(self, locs, info):
        dh = postprocess.distance_histogram(
            locs, info, bin_size=10.0, r_max=100.0
        )
        assert dh.sum() > 0

    def test_count_grows_with_r_max(self, locs, info):
        small = postprocess.distance_histogram(
            locs, info, bin_size=10.0, r_max=50.0
        )
        large = postprocess.distance_histogram(
            locs, info, bin_size=10.0, r_max=150.0
        )
        # The large-r histogram must contain all bins of the small-r one
        # (same bin edges) plus more — sum is monotone in r_max.
        assert large[: len(small)].sum() >= small.sum()


class TestNena:
    def test_returns_positive_resolution(self, locs, info):
        _, nena = postprocess.nena(locs, info)
        assert nena > 0
        # NeNA (nm) of DNA-PAINT data is a few nm, far below a camera
        # pixel
        assert nena < 20

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
            postprocess._next_frame_neighbor_distance_histogram(
                locs.copy(), lib.MockProgress()
            )
        )
        assert bin_centers.shape == dnfl.shape
        assert (dnfl >= 0).all()
        # 0.2 nm bins up to 200 nm, evenly spaced
        diffs = np.diff(bin_centers)
        assert np.allclose(diffs, 0.2)
        assert len(bin_centers) == 1000

    def test_some_neighbors_present(self, locs):
        _, dnfl = postprocess._next_frame_neighbor_distance_histogram(
            locs.copy(), lib.MockProgress()
        )
        # Bundled DNA-PAINT data has many on-events lasting >1 frame, so
        # there should be at least some next-frame neighbors recorded.
        assert dnfl.sum() > 0


# one camera pixel around the center of the test data
FRC_VIEWPORT = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))


class TestFrc:
    def test_resolution_keys(self, locs, info):
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
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
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["frc_curve"][0] > 0.7

    def test_frc_curve_and_freq_same_length(self, locs, info):
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["frc_curve"].shape == frc_res["frequencies"].shape
        assert (
            frc_res["frc_curve_smooth"].shape == frc_res["frequencies"].shape
        )

    def test_two_images_returned(self, locs, info):
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
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
        # 4 camera pixels wide, 1 high
        viewport = ((15 * PX, 13 * PX), (16 * PX, 17 * PX))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["images"][0].shape[0] == frc_res["images"][0].shape[1]

    def test_given_lp_skips_nena(self, locs, info, monkeypatch):
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
        expected = postprocess.frc(locs, info, viewport=viewport)
        lp = postprocess.nena(locs, info)[1]

        def fail(*args, **kwargs):
            raise AssertionError("nena must not run when lp is given")

        monkeypatch.setattr(postprocess, "nena", fail)
        frc_res = postprocess.frc(locs, info, viewport=viewport, lp=lp)
        assert frc_res["resolution"] == pytest.approx(expected["resolution"])

    def test_too_large_images_raise(self, locs, info, monkeypatch):
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
        lp = 1.0  # 2 * PX pixels on a side

        def fail(*args, **kwargs):
            raise AssertionError("nothing may be rendered")

        monkeypatch.setattr(postprocess.render, "render", fail)
        with pytest.raises(ValueError, match="max_image_size"):
            postprocess.frc(
                locs, info, viewport=viewport, lp=lp, max_image_size=PX
            )

    def test_max_image_size_none_renders_any_size(self, locs, info):
        viewport = ((15 * PX, 15 * PX), (16 * PX, 16 * PX))
        frc_res = postprocess.frc(
            locs, info, viewport=viewport, max_image_size=None
        )
        assert frc_res["resolution"] > 0

    def test_does_not_modify_locs(self, locs, info):
        ref = locs.copy()
        postprocess.frc(locs, info, viewport=FRC_VIEWPORT)
        pd.testing.assert_frame_equal(locs, ref)

    def test_plot_without_resolution(self, locs, info):
        frc_res = postprocess.frc(locs, info, viewport=FRC_VIEWPORT)
        frc_res["resolution"] = None
        fig = postprocess.plot_frc(frc_res)
        assert "n/a" in fig.axes[0].get_title()


class TestFrcRois:
    def test_results_per_roi(self, locs, info):
        viewport = ((10 * PX, 10 * PX), (20 * PX, 20 * PX))
        ref = locs.copy()
        tracker = RecordingProgress()
        result = postprocess.frc_rois(
            locs,
            info,
            viewport=viewport,
            n_rois=4,
            roi_size=260,  # 2 camera pixels
            min_locs=20,
            progress=tracker,
        )
        n = len(result["rois"])
        assert 0 < n <= 4
        assert len(result["frc_results"]) == n
        assert result["resolutions"].shape == (n,)
        assert (result["n_locs"] >= 20).all()
        assert all("images" not in _ for _ in result["frc_results"])
        assert_phases(tracker, ["Calculating NeNA", "Calculating FRC"])
        assert tracker.phases[1].maximum == n
        pd.testing.assert_frame_equal(locs, ref)

    def test_given_lp_skips_nena(self, locs, info, monkeypatch):
        def no_nena(*args, **kwargs):
            raise AssertionError("NeNA calculated")

        monkeypatch.setattr(postprocess, "nena", no_nena)
        tracker = RecordingProgress()
        result = postprocess.frc_rois(
            locs,
            info,
            viewport=((10 * PX, 10 * PX), (20 * PX, 20 * PX)),
            n_rois=4,
            roi_size=260,
            min_locs=20,
            lp=12.0,
            progress=tracker,
        )
        assert result["lp"] == 12.0
        assert_phases(tracker, ["Calculating FRC"])


class TestPairCorrelation:
    def test_shape(self, locs, info):
        bins_lower, pc = postprocess.pair_correlation(
            locs, info, bin_size=10.0, r_max=100.0
        )
        assert bins_lower.shape == pc.shape
        assert bins_lower.shape[0] == 10

    def test_pc_finite_and_non_negative(self, locs, info):
        _, pc = postprocess.pair_correlation(
            locs, info, bin_size=10.0, r_max=100.0
        )
        assert np.all(np.isfinite(pc))
        assert (pc >= 0).all()

    def test_normalization_against_distance_histogram(self, locs, info):
        bin_size, r_max = 10.0, 100.0
        dh = postprocess.distance_histogram(
            locs, info, bin_size=bin_size, r_max=r_max
        )
        bins_lower, pc = postprocess.pair_correlation(
            locs, info, bin_size=bin_size, r_max=r_max
        )
        # pc = dh / (pi * bin_size * (2 * bins_lower + bin_size))
        expected = dh / (np.pi * bin_size * (2 * bins_lower + bin_size))
        np.testing.assert_allclose(pc, expected, rtol=1e-6)


class TestLocalDensity:
    def test_density_column_added_with_proper_dtype(self, locs, info):
        out, _ = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE / 2
        )
        assert "density" in out.columns
        assert out["density"].dtype in (np.uint32, np.uint64)
        # Each loc has at least itself within radius
        assert (out["density"] >= 1).all()

    def test_dense_radius_picks_up_origami_clusters(self, locs, info):
        out, _ = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE / 2
        )
        unique_densities, _ = np.unique(out["density"], return_counts=True)
        # Test data has ~9 origamis, so density should take few values
        assert len(unique_densities) < len(locs) // 5

    def test_density_increases_with_radius(self, locs, info):
        small, _ = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE / 4
        )
        large, _ = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE
        )
        # A larger radius can only see at least as many neighbors.
        assert large["density"].sum() >= small["density"].sum()


# ---------------------------------------------------------------------------
# Linking and dark-time computation
# ---------------------------------------------------------------------------


class TestLinking:
    def test_columns_added(self, locs, info):
        info_in = copy.deepcopy(info)
        linked, new_info = postprocess.link(locs.copy(), info_in)
        for col in ["len", "n", "photon_rate"]:
            assert col in linked.columns
        # the step is appended to a copy of the input metadata
        assert info_in == info
        assert new_info[:-1] == info
        assert new_info[-1]["Step"] == "Link"

    def test_length_invariants(self, locs, info):
        """Linked length is <= original (events merge); the sum of the
        ``n`` (locs per linked event) column equals the original count."""
        linked, _ = postprocess.link(locs.copy(), info)
        assert len(linked) <= len(locs)
        assert linked["n"].sum() == len(locs)

    def test_len_within_movie_frame_span(self, locs, info):
        linked, _ = postprocess.link(locs.copy(), info)
        n_frames = info[0]["Frames"]
        assert linked["len"].max() <= n_frames

    def test_compute_dark_times_adds_dark_column(self, locs, info):
        linked, linked_info = postprocess.link(locs.copy(), info)
        with_dark, dark_info = postprocess.compute_dark_times(
            linked, linked_info
        )
        assert "dark" in with_dark.columns
        # the input is not modified
        assert "dark" not in linked.columns
        assert dark_info[-1]["Step"] == "Dark"

    def test_compute_dark_times_requires_link(self, locs):
        with pytest.raises(AttributeError):
            postprocess.compute_dark_times(locs.copy())

    def test_link_empty_locs_returns_empty_with_columns(self, locs, info):
        empty = locs.iloc[0:0].copy()
        out, _ = postprocess.link(empty, info)
        assert len(out) == 0
        for col in ["len", "n", "photon_rate"]:
            assert col in out.columns

    def test_link_removes_events_at_movie_edges(self):
        # three events far apart: one starting in the first frame, one in
        # the middle and one ending in the last frame of a 10-frame movie
        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1, 4, 5, 8, 9], dtype=np.int32),
                "x": np.array([0, 0, 500, 500, 1000, 1000], dtype=np.float32),
                "y": np.zeros(6, dtype=np.float32),
                "lpx": np.full(6, 5.0, dtype=np.float32),
                "lpy": np.full(6, 5.0, dtype=np.float32),
            }
        )
        info = [{"Frames": 10, "Width": 1000, "Height": 1000}]
        linked, _ = postprocess.link(locs, info)
        assert linked["frame"].tolist() == [4]
        kept, _ = postprocess.link(locs, info, remove_ambiguous_lengths=False)
        assert kept["frame"].tolist() == [0, 4, 8]

    def test_link_groups_consistent_with_link(self, locs, info):
        # The number of unique non-(-1) link groups must equal the
        # number of linked events when there are no ambiguities to drop.
        sl = locs.sort_values(by="frame", kind="quicksort")
        frame = sl["frame"].to_numpy()
        x = sl["x"].to_numpy()
        y = sl["y"].to_numpy()
        group = np.zeros(len(sl), dtype=np.int32)
        lg = postprocess._get_link_groups(frame, x, y, 10.0, 3, group)
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
        # Two binding events 500 nm apart, their localizations scattered
        # by a few nm like real ones: one 4 frames long, one 2 frames
        # long (too short to have a core).
        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1, 2, 3, 0, 1], dtype=np.int32),
                "x": np.array(
                    [1000, 1003, 998, 1004, 1500, 1502], dtype=np.float32
                ),
                "y": np.array(
                    [1000, 997, 1002, 1001, 1500, 1499], dtype=np.float32
                ),
            }
        )
        out, _ = postprocess.select_binding_event_cores(locs)
        # only frames 1 and 2 of the first event survive
        assert out["frame"].tolist() == [1, 2]
        assert out["x"].tolist() == [1003, 998]
        assert out["group"].tolist() == [0, 0]

    def test_default_r_max_is_10_nm(self):
        # Localizations 9 nm apart in consecutive frames form one event,
        # 11 nm apart they do not (the documented default is 10 nm).
        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
                "x": np.array(
                    [1000, 1009, 1000, 3000, 3011, 3022], dtype=np.float32
                ),
                "y": np.full(6, 1000, dtype=np.float32),
            }
        )
        out, _ = postprocess.select_binding_event_cores(locs)
        assert out["x"].tolist() == [1009]

    def test_min_n_locs_discards_short_events(self):
        locs = pd.DataFrame(
            {
                "frame": np.array([0, 1, 2, 3], dtype=np.int32),
                "x": np.full(4, 1000, dtype=np.float32),
                "y": np.full(4, 1000, dtype=np.float32),
            }
        )
        out, _ = postprocess.select_binding_event_cores(locs, min_n_locs=4)
        assert out["frame"].tolist() == [1, 2]
        out, _ = postprocess.select_binding_event_cores(locs, min_n_locs=5)
        assert len(out) == 0

    def test_matches_link_on_test_data(self, locs, info):
        # The events are those of link with the same distance and dark
        # time: each event of n >= min_n_locs (3) localizations keeps
        # its n - 2 inner ones.
        out, _ = postprocess.select_binding_event_cores(locs.copy())
        linked, _ = postprocess.link(
            locs.copy(), info, r_max=10, max_dark_time=0
        )
        n = linked["n"].to_numpy()
        assert (n >= 3).sum() > 0
        assert len(out) == (n[n >= 3] - 2).sum()
        assert out["group"].nunique() == (n >= 3).sum()

    def test_groups_are_consecutive(self, locs):
        out, _ = postprocess.select_binding_event_cores(locs.copy())
        assert 0 < len(out) < len(locs)
        groups = np.unique(out["group"].to_numpy())
        assert (groups == np.arange(len(groups))).all()
        assert "group_input" not in out.columns

    def test_existing_group_preserved_as_group_input(self, locs):
        sub = locs.copy()
        # two input groups, split spatially so events cannot span both
        sub["group"] = (sub["x"].to_numpy() > sub["x"].median()).astype(
            np.int32
        )
        out, _ = postprocess.select_binding_event_cores(sub)
        assert "group_input" in out.columns
        assert set(np.unique(out["group_input"].to_numpy())) == {0, 1}
        # binding events must not span two input groups
        assert (out.groupby("group")["group_input"].nunique() == 1).all()

    def test_empty_locs(self, locs):
        out, _ = postprocess.select_binding_event_cores(locs.iloc[0:0].copy())
        assert len(out) == 0
        assert "group" in out.columns


def _docking_sites(
    n_sites: int,
    n_frames: int,
    *,
    destroyed: float = 0.0,
    tau_dark: float = 200.0,
    tau_bright: float = 3.0,
    seed: int = 0,
) -> list[pd.DataFrame]:
    """Localizations of DNA-PAINT docking sites, one per list element:
    binding events with exponential dark and bright times (frames); the
    fraction ``destroyed`` of the sites stops binding at a uniformly
    random time."""
    rng = np.random.default_rng(seed)
    sites = []
    for _ in range(n_sites):
        end = n_frames
        if rng.uniform() < destroyed:
            end = rng.uniform(0, n_frames)
        frames = []
        t = rng.exponential(tau_dark)
        while t < end:
            length = rng.exponential(tau_bright)
            frames.extend(range(int(t), int(min(t + length, end)) + 1))
            t += length + rng.exponential(tau_dark)
        frames = np.unique(np.clip(frames, 0, n_frames - 1))
        sites.append(_site_locs(frames, rng))
    return sites


def _site_locs(frames, rng=None) -> pd.DataFrame:
    """Localizations of one docking site in ``frames``."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(frames)
    return pd.DataFrame(
        {
            "frame": np.asarray(frames, dtype=np.uint32),
            "x": rng.normal(0, 3, n).astype(np.float32),
            "y": rng.normal(0, 3, n).astype(np.float32),
            "photons": np.full(n, 1000, dtype=np.float32),
            "sx": np.full(n, 120, dtype=np.float32),
            "sy": np.full(n, 120, dtype=np.float32),
            "bg": np.full(n, 20, dtype=np.float32),
            "lpx": np.full(n, 3, dtype=np.float32),
            "lpy": np.full(n, 3, dtype=np.float32),
        }
    )


class TestDockingSiteDamage:
    N_FRAMES = 10_000
    INFO = [{"Frames": N_FRAMES, "Pixelsize": 130.0}]

    def test_intact_sites(self):
        sites = _docking_sites(600, self.N_FRAMES, seed=1)
        result = postprocess.docking_site_damage(sites, self.INFO)
        assert result["n_sites"] == 600
        # r follows an exponential distribution with mean 1
        assert result["mu"] == pytest.approx(1.0, abs=0.1)
        assert np.nanmean(result["r"]) == pytest.approx(1.0, abs=0.1)
        assert result["r_cutoff"] == pytest.approx(4 * result["mu"])
        assert abs(result["fraction_destroyed"]) < 0.02
        assert result["fraction_destroyed"] == pytest.approx(
            result["fraction_above_cutoff"] - np.exp(-4)
        )
        np.testing.assert_allclose(np.nanmedian(result["dark"]), 200, rtol=0.1)
        assert result["counts"].sum() == 600
        assert len(result["edges"]) == len(result["counts"]) + 1

    def test_destroyed_sites(self):
        sites = _docking_sites(600, self.N_FRAMES, destroyed=0.3, seed=2)
        result = postprocess.docking_site_damage(sites, self.INFO)
        # sites destroyed within the last 4 dark times stay below the
        # cut-off: about 0.3 * (1 - 4 * 200 / 10,000)
        assert result["fraction_destroyed"] == pytest.approx(0.276, abs=0.05)
        drop = postprocess.localization_rate_drop(pd.concat(sites), self.INFO)
        assert drop == pytest.approx(0.3 * 0.8 / 0.9, abs=0.07)

    def test_sites_without_a_mean_dark_time(self):
        sites = _docking_sites(50, self.N_FRAMES, seed=3)
        one_event = _site_locs([10, 11])
        sites += [sites[0].iloc[:0], one_event]
        result = postprocess.docking_site_damage(sites, self.INFO)
        assert result["n_sites"] == 50
        assert np.isnan(result["r"][-2:]).all()
        assert np.isnan(result["t_end"][-2])
        assert result["t_end"][-1] == self.N_FRAMES - 1 - 11
        with pytest.raises(ValueError, match="mean dark time"):
            postprocess.docking_site_damage([one_event], self.INFO)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(max_dark_time=-1), "max_dark_time"),
            (dict(cutoff=0.0), "cutoff"),
            (dict(bin_width=0.0), "bin_width"),
        ],
    )
    def test_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            postprocess.docking_site_damage([], self.INFO, **kwargs)

    def test_needs_the_number_of_frames(self):
        with pytest.raises(KeyError, match="Frames"):
            postprocess.docking_site_damage([], [{}])

    def test_plot(self):
        sites = _docking_sites(100, self.N_FRAMES, destroyed=0.2, seed=4)
        result = postprocess.docking_site_damage(sites, self.INFO)
        fig = postprocess.plot_docking_site_damage(result)
        (ax,) = fig.axes
        # the bulk, not the destroyed sites far beyond
        x_max = max(10, 2.5 * result["r_cutoff"])
        assert ax.get_xlim() == pytest.approx((0, x_max))
        assert x_max < result["edges"][-1]
        labels = ax.get_legend_handles_labels()[1]
        assert labels[0] == f"{result['n_sites']} sites"
        assert "destroyed" in labels[-1]

    def test_cdf_mean_is_unbiased_for_few_events(self):
        rng = np.random.default_rng(5)
        estimates = [
            postprocess._exponential_cdf_mean(rng.exponential(100.0, 20))
            for _ in range(300)
        ]
        assert np.mean(estimates) == pytest.approx(100, rel=0.04)
        assert np.isnan(postprocess._exponential_cdf_mean(np.array([5.0])))
        # missing dark times (-1) are ignored
        assert postprocess._exponential_cdf_mean(
            np.array([-1.0, 4.0, 4.0])
        ) == pytest.approx(4.0)


class TestDyePerformance:
    """``dye_performance``, as in Render's Info dialog and
    ``picasso dye-performance``."""

    N_FRAMES = TestDockingSiteDamage.N_FRAMES
    INFO = TestDockingSiteDamage.INFO

    def test_all_values(self):
        sites = _docking_sites(100, self.N_FRAMES, seed=6)
        result = postprocess.dye_performance(sites, self.INFO)
        assert result["errors"] == {}
        pooled = result["locs"]
        # a group per pick, so that binding events stay within a pick
        assert pooled["group"].nunique() == 100
        # the SBR of the centers of binding events (1000 photons, sx =
        # sy = 120 nm, 130 nm pixels, bg = 20)
        scale = 65 / np.sqrt(2) / 120
        expected = 1000 * special.erf(scale) ** 2 / 20
        np.testing.assert_allclose(result["sbr"], expected, rtol=1e-5)
        cores, _ = postprocess.select_binding_event_cores(pooled, r_max=np.inf)
        assert len(result["sbr"]) == len(cores)
        assert result["damage"]["n_sites"] == 100
        assert result[
            "localization_drop"
        ] == postprocess.localization_rate_drop(pooled, self.INFO)

    def test_without_a_gaussian_fit(self):
        """Without sx and sy (e.g., spline fits) there is no SBR; picks
        without enough binding events have no damage."""
        locs = _site_locs(np.arange(10)).drop(columns=["sx", "sy"])
        info = [{"Frames": 100, "Pixelsize": 130.0}]
        result = postprocess.dye_performance([locs], info, max_dark_time=3)
        assert result["sbr"] is None and result["damage"] is None
        assert result["localization_drop"] == pytest.approx(1.0)
        assert "sx, sy" in result["errors"]["sbr"]
        assert "mean dark time" in result["errors"]["damage"]
        empty = postprocess.dye_performance([locs.iloc[:0]], info)
        assert empty["locs"] is None
        assert set(empty["errors"]) == {"sbr", "localization_drop", "damage"}


class TestLocalizationRateDrop:
    INFO = [{"Frames": 1000}]

    def test_compares_the_first_and_last_fifth(self):
        frames = np.concatenate([np.arange(200), np.arange(800, 1000, 2)])
        locs = pd.DataFrame({"frame": np.concatenate([frames, [500]])})
        drop = postprocess.localization_rate_drop(locs, self.INFO)
        assert drop == pytest.approx(0.5)
        drop = postprocess.localization_rate_drop(
            locs, self.INFO, fraction=0.5
        )
        assert drop == pytest.approx(1 - 101 / 200)  # frame 500 is late

    def test_increase_is_negative(self):
        locs = pd.DataFrame({"frame": [0, 900, 950]})
        assert postprocess.localization_rate_drop(locs, self.INFO) == -1.0

    @pytest.mark.parametrize("fraction", [0.0, 0.6])
    def test_invalid_fraction(self, fraction):
        locs = pd.DataFrame({"frame": [0]})
        with pytest.raises(ValueError, match="fraction"):
            postprocess.localization_rate_drop(
                locs, self.INFO, fraction=fraction
            )

    def test_needs_localizations_at_the_start(self):
        locs = pd.DataFrame({"frame": [900]})
        with pytest.raises(ValueError, match="first 20%"):
            postprocess.localization_rate_drop(locs, self.INFO)

    def test_plot(self):
        frames = np.concatenate([np.arange(200), np.arange(800, 1000, 2)])
        locs = pd.DataFrame({"frame": frames})
        fig = postprocess.plot_localization_rate(locs, self.INFO, n_bins=10)
        (ax,) = fig.axes
        # mean localizations per bin of 100 frames: 100 and 50
        levels = ax.collections[-1].get_segments()
        assert [segment[0][1] for segment in levels] == [100, 50]
        assert "drop 50.0%" in ax.get_legend_handles_labels()[1][-1]
        with pytest.raises(ValueError, match="n_bins"):
            postprocess.plot_localization_rate(locs, self.INFO, n_bins=0)


class TestSignalToBackground:
    INFO = [{"Pixelsize": 130.0}]

    def test_photons_in_one_pixel_over_the_background(self):
        sx, sy = 110.0, 150.0
        # the Gaussian integrated over a pixel centered on its peak
        t = np.linspace(-65, 65, 4001)
        gx = np.exp(-(t**2) / (2 * sx**2)) / (np.sqrt(2 * np.pi) * sx)
        gy = np.exp(-(t**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)
        fraction = np.trapezoid(gx, t) * np.trapezoid(gy, t)
        locs = pd.DataFrame(
            {
                "photons": [1000.0, 1000.0, 1000.0],
                "sx": [sx] * 3,
                "sy": [sy] * 3,
                "bg": [20.0, 0.0, -1.0],
            }
        )
        sbr = postprocess.signal_to_background(locs, self.INFO)
        assert sbr[0] == pytest.approx(1000 * fraction / 20, rel=1e-6)
        assert np.isnan(sbr[1:]).all()

    def test_needs_a_gaussian_fit(self):
        locs = pd.DataFrame({"photons": [1.0], "bg": [1.0]})
        with pytest.raises(ValueError, match="sx, sy"):
            postprocess.signal_to_background(locs, self.INFO)


class TestDarkTimes:
    def test_dark_times_min_positive(self, locs, info):
        linked, _ = postprocess.link(locs.copy(), info)
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
        out, _ = postprocess.compute_dark_times(locs)
        assert out["dark"].tolist() == [1]

    def test_dark_times_with_explicit_group(self, locs, info):
        linked, _ = postprocess.link(locs.copy(), info)
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
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
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

    def test_empty_picks_are_nan(self, locs, info, origami_picks):
        """Empty picks were left as uninitialized memory (np.empty),
        which showed huge numbers in Render's Info dialog."""
        pl = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks[:2],
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        pl.insert(1, pl[0].iloc[:0])
        stats = postprocess.evaluate_picks(pl, info, max_dark_time=3)
        for values in stats[:6]:
            assert np.isnan(values[1])
        assert stats[0][0] == len(pl[0]) and stats[0][2] == len(pl[2])


class TestCombineLocsInPicks:
    def test_combines_into_one_loc_per_pick(self, locs, info, origami_picks):
        combined, _ = postprocess.combine_locs_in_picks(
            locs.copy(),
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        # Each origami collapses to a single linked event
        assert len(combined) == len(origami_picks)
        # n column tracks how many locs each event came from, and the
        # totals must match the picked-loc count.
        picked = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        assert combined["n"].sum() == sum(len(p) for p in picked)

    def test_brush_picks_need_no_size(self, locs, info, origami_picks):
        picks = [
            [(PICK_SIZE, [(x - 0.2 * PX, y), (x + 0.2 * PX, y)])]
            for x, y in origami_picks
        ]
        combined, _ = postprocess.combine_locs_in_picks(
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
        combined, _ = postprocess.combine_locs_in_picks(
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
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
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
        # ``no_locs`` counts the localizations of each surviving pick,
        # before linking.
        assert np.array_equal(no_locs, [len(pl[i]) for i in kept])
        # Returned per-loc dataframe carries the kinetics columns.
        for col in ("len", "n", "dark"):
            assert col in out_locs.columns
        # Linking merges localizations into fewer binding events (and
        # the first event of a pick, without a dark time, is dropped).
        assert len(out_locs) < int(no_locs.sum())

    def test_pick_properties_locs_counts_localizations(self, info):
        # three picks with 60, 80 and 100 isolated localizations: ``locs``
        # is the number of localizations, ``n_events`` the number of
        # binding events with a measured dark time
        rng = np.random.default_rng(0)
        picks = []
        for g, n in enumerate((60, 80, 100)):
            frames = np.sort(rng.choice(5000, n, replace=False))
            picks.append(
                pd.DataFrame(
                    {
                        "frame": frames.astype(np.uint32),
                        "x": rng.normal(10, 0.05, n).astype(np.float32),
                        "y": rng.normal(10, 0.05, n).astype(np.float32),
                        "photons": np.full(n, 1000, np.float32),
                        "sx": np.ones(n, np.float32),
                        "sy": np.ones(n, np.float32),
                        "bg": np.ones(n, np.float32),
                        "lpx": np.full(n, 0.05, np.float32),
                        "lpy": np.full(n, 0.05, np.float32),
                        "group": np.full(n, g, np.int32),
                    }
                )
            )
        props = postprocess.pick_properties(picks, info, max_dark_time=3)
        assert props["locs"].tolist() == [60, 80, 100]
        assert (props["n_events"] < props["locs"]).all()


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
        width, height = lib.field_of_view(info)
        oversampling = 1 / postprocess.UNDRIFT_PIXEL_SIZE
        n_pixel_y = int(np.ceil(oversampling * height))
        n_pixel_x = int(np.ceil(oversampling * width))
        assert segs.shape == (n_seg, n_pixel_y, n_pixel_x)
        # bounds are strictly increasing and span the movie
        assert (np.diff(bounds) > 0).all()
        assert bounds[0] == 0
        assert bounds[-1] == info[0]["Frames"] - 1


class TestUndrift:
    def test_drift_has_one_row_per_frame(self, locs, info):
        undrifted, new_info, drift = postprocess.undrift(
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
        assert new_info[:-1] == info
        assert new_info[-1]["Step"] == "Undrift"
        assert new_info[-1]["Segmentation"] == 200

    def test_undrift_from_picked_returns_drift(
        self, locs, info, origami_picks
    ):
        # The origami picks are not real fiducials but they exercise the
        # code path and produce a valid (n_frames, 2) drift table.
        pl = postprocess.picked_locs(
            locs,
            info,
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
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
            pick_size=PICK_SIZE,
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
                picks=ORIGAMI_PICKS[:1],
                pick_size=None,
            )

    def test_undrift_from_fiducials_honors_pick_shape(
        self, locs, info, origami_picks
    ):
        # a box pick must be picked as a box, not silently as a circle:
        # a wide, flat box and a circle over the same center enclose
        # different localizations, so the drift they report differs
        boxes = [
            ((x - 1.0 * PX, y - 0.2 * PX), (x + 1.0 * PX, y + 0.2 * PX))
            for x, y in origami_picks
        ]
        _, new_info, box_drift = postprocess.undrift_from_fiducials(
            locs.copy(),
            info,
            picks=boxes,
            pick_shape="Box",
            undrift_z=False,
        )
        assert new_info[-1]["Pick shape"] == "Box"
        # boxes carry their own extent, so no diameter is reported
        assert "Pick diameter (nm)" not in new_info[-1]

        _, _, circle_drift = postprocess.undrift_from_fiducials(
            locs.copy(),
            info,
            picks=origami_picks,
            pick_size=PICK_SIZE,
            undrift_z=False,
        )
        assert not np.allclose(
            box_drift["x"].to_numpy(), circle_drift["x"].to_numpy()
        )

    def test_undrift_from_fiducials_box_needs_no_size(
        self, locs, info, origami_picks
    ):
        boxes = [
            ((x - 0.75 * PX, y - 0.75 * PX), (x + 0.75 * PX, y + 0.75 * PX))
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
        out, _ = postprocess.apply_drift(locs.copy(), info, drift=drift)
        assert out["y"].mean() == pytest.approx(
            locs["y"].mean() - dy, abs=1e-3
        )
        assert out["x"].mean() == pytest.approx(locs["x"].mean(), abs=1e-3)

    def test_apply_drift_ndarray_2d(self, locs, info):
        n_frames = info[0]["Frames"]
        drift = np.zeros((n_frames, 2), dtype=np.float64)
        drift[:, 0] = 0.25
        out, _ = postprocess.apply_drift(locs.copy(), info, drift=drift)
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
        """Apply a known shift of 5 camera pixels to a copy and check
        that align() brings the channels back together (residual below
        half a camera pixel)."""
        a = locs_copy
        b = a.copy()
        b["x"] += 5.0 * PX
        aligned, new_infos = postprocess.align([a, b], [info, info])
        assert len(aligned) == 2
        assert [_[-1]["Step"] for _ in new_infos] == ["Align", "Align"]
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.5 * PX

    def test_no_shift_is_no_op_within_tolerance(self, locs_copy, info):
        """If channels start aligned, alignment shouldn't drift them."""
        a = locs_copy
        b = a.copy()
        aligned, _ = postprocess.align([a, b], [info, info])
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.1 * PX

    def test_align_does_not_modify_locs(self, locs_copy, info):
        a = locs_copy
        b = a.copy()
        b["x"] += 3.0 * PX
        x_before_a = a["x"].copy()
        x_before_b = b["x"].copy()
        out, _, shifts = postprocess.align(
            [a, b], [info, info], return_shifts=True
        )
        np.testing.assert_array_equal(a["x"].to_numpy(), x_before_a)
        np.testing.assert_array_equal(b["x"].to_numpy(), x_before_b)
        # shifts is (shift_x, shift_y), one value per channel, and the
        # returned locs are the inputs minus the shifts
        shift_x, shift_y = shifts
        assert len(shift_x) == 2
        assert len(shift_y) == 2
        np.testing.assert_allclose(
            out[1]["x"].to_numpy(), x_before_b.to_numpy() - shift_x[1]
        )
        assert abs((shift_x[1] - shift_x[0]) - 3.0 * PX) < 0.05 * PX

    def test_align_picked_matches_align_from_picked(
        self, locs_copy, info, origami_picks
    ):
        """method="picked" gives the result of align_from_picked."""
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1 * PX
        kwargs = dict(
            picks=origami_picks, pick_shape="Circle", pick_size=PICK_SIZE
        )
        aligned, _, shifts = postprocess.align(
            [a, b], [info, info], method="picked", return_shifts=True, **kwargs
        )
        expected, _, expected_shifts = postprocess.align_from_picked(
            [a, b], [info, info], return_shifts=True, **kwargs
        )
        for out, exp in zip(aligned, expected):
            pd.testing.assert_frame_equal(out, exp)
        for shift, exp in zip(shifts, expected_shifts, strict=True):
            np.testing.assert_array_equal(shift, exp)
        assert abs(shifts[0][1] - 0.1 * PX) < 0.05 * PX

    def test_align_invalid_arguments_raise(
        self, locs_copy, info, origami_picks
    ):
        locs = [locs_copy, locs_copy.copy()]
        infos = [info, info]
        with pytest.raises(ValueError, match="Unknown alignment method"):
            postprocess.align(locs, infos, method="affine")
        with pytest.raises(ValueError, match="requires picks"):
            postprocess.align(locs, infos, method="picked")
        with pytest.raises(ValueError, match="only used with"):
            postprocess.align(locs, infos, picks=origami_picks)

    @staticmethod
    def _halves(locs, shift_x, shift_y):
        """The localizations split at random into two channels, the
        second shifted by ``(shift_x, shift_y)`` nm."""
        rng = np.random.default_rng(0)
        second = rng.uniform(size=len(locs)) < 0.5
        a = locs[~second].copy()
        b = locs[second].copy()
        b["x"] += shift_x
        b["y"] += shift_y
        return a, b

    def test_align_from_points(self, info):
        rng = np.random.default_rng(2)
        sites = rng.uniform(0, 20_000, (1000, 2))
        a = self._sites_channel(sites, (0.0, 0.0), rng)
        b = self._sites_channel(sites, (25.3, -8.1), rng)
        b_before = b.copy()
        aligned, infos, (shift_x, shift_y) = postprocess.align_from_points(
            [a, b], [info, info], return_shifts=True
        )
        assert shift_x[0] == shift_y[0] == 0
        entry = infos[1][-1]
        error_x = entry["Shift x uncertainty (nm)"]
        error_y = entry["Shift y uncertainty (nm)"]
        assert 0 < error_x < 0.3 and 0 < error_y < 0.3
        assert shift_x[1] == pytest.approx(25.3, abs=4 * error_x)
        assert shift_y[1] == pytest.approx(-8.1, abs=4 * error_y)
        np.testing.assert_allclose(
            aligned[1]["x"].to_numpy(), b["x"].to_numpy() - shift_x[1]
        )
        assert entry["Method"] == "Points"
        assert entry["Max. shift (nm)"] == 500.0
        assert entry["Shift x (nm)"] == pytest.approx(shift_x[1])
        assert infos[0][-1]["Shift x uncertainty (nm)"] == 0
        pd.testing.assert_frame_equal(b, b_before)  # not modified

    def test_align_points_matches_align_from_points(self, locs_copy, info):
        a, b = self._halves(locs_copy, -40.0, 12.0)
        aligned, infos, shifts = postprocess.align(
            [a, b],
            [info, info],
            method="points",
            max_shift=300.0,
            return_shifts=True,
        )
        expected, expected_infos, expected_shifts = (
            postprocess.align_from_points(
                [a, b], [info, info], max_shift=300.0, return_shifts=True
            )
        )
        for out, exp in zip(aligned, expected):
            pd.testing.assert_frame_equal(out, exp)
        for shift, exp in zip(shifts, expected_shifts, strict=True):
            np.testing.assert_array_equal(shift, exp)
        assert infos[1][-1]["Max. shift (nm)"] == 300.0

    @staticmethod
    def _sites_channel(sites, shift, rng):
        """Ten localizations (6 nm precision) at each of ``sites``,
        shifted by ``shift`` nm, and random ones."""
        points = np.repeat(sites, 10, axis=0) + rng.normal(
            0, 6, (10 * len(sites), 2)
        )
        points = np.vstack([points, rng.uniform(0, 20_000, (2000, 2))])
        points = points + shift
        return pd.DataFrame(
            {
                "frame": np.zeros(len(points), dtype=np.uint32),
                "x": points[:, 0].astype(np.float32),
                "y": points[:, 1].astype(np.float32),
            }
        )

    def test_align_from_points_without_common_structures(self, info):
        """A pair of channels without common structures is left out; the
        shifts are found through the other pairs."""
        rng = np.random.default_rng(1)
        sites, other = rng.uniform(0, 20_000, (2, 1000, 2))
        channels = [
            self._sites_channel(sites, (0.0, 0.0), rng),
            pd.concat(
                [
                    self._sites_channel(sites, (20.0, 5.0), rng),
                    self._sites_channel(other, (20.0, 5.0), rng),
                ]
            ),
            self._sites_channel(other, (-15.0, 30.0), rng),
        ]
        _, infos, (shift_x, shift_y) = postprocess.align_from_points(
            channels, [info] * 3, return_shifts=True
        )
        np.testing.assert_allclose(shift_x, [0, 20, -15], atol=0.5)
        np.testing.assert_allclose(shift_y, [0, 5, 30], atol=0.5)
        # the third channel inherits the uncertainty of the second
        errors = [entry[-1]["Shift x uncertainty (nm)"] for entry in infos]
        assert errors[2] > errors[1] > 0
        with pytest.raises(ValueError, match="could not be found"):
            postprocess.align_from_points(
                [channels[0], channels[2]], [info] * 2
            )

    def test_align_from_points_invalid_arguments(self, locs_copy, info):
        with pytest.raises(ValueError, match="max_shift"):
            postprocess.align_from_points(
                [locs_copy, locs_copy], [info, info], max_shift=0
            )
        with pytest.raises(ValueError, match="two channels"):
            postprocess.align_from_points([locs_copy], [info])
        with pytest.raises(ValueError, match="only used with"):
            postprocess.align(
                [locs_copy, locs_copy],
                [info, info],
                method="points",
                picks=[(0.0, 0.0)],
            )

    def test_align_rcc_converges(self, locs_copy, info):
        a = locs_copy
        b = a.copy()
        b["x"] += 2.0 * PX
        aligned, _, (shift_x, shift_y) = postprocess.align_rcc(
            [a, b], [info, info], return_shifts=True
        )
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.05 * PX
        # the returned shifts are the per-channel totals over all
        # iterations
        assert len(shift_x) == len(shift_y) == 2
        assert abs((shift_x[1] - shift_x[0]) - 2.0 * PX) < 0.05 * PX

    def test_align_rcc_channels_with_different_canvases(self, locs_copy, info):
        """Channels whose canvases differ (e.g., one was moved in Render
        and lib.fit_canvas enlarged only its Width/Height) are rendered
        on one canvas instead of failing with mismatched image shapes."""
        a = locs_copy
        b = a.copy()
        b["x"] += 2.0 * PX
        info_b = copy.deepcopy(info)
        info_b[0]["Height"] = 2 * info[0]["Height"]
        info_b[0]["Width"] = info[0]["Width"] + 3
        aligned, _ = postprocess.align_rcc([a, b], [info, info_b])
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.05 * PX

    def test_align_from_picked_recovers_known_shift(
        self, locs_copy, info, origami_picks
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1 * PX
        aligned, _, shifts = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            return_shifts=True,
        )
        # shifts is (shift_x, shift_y), one value per channel. The
        # second channel should have ~0.1 camera pixels in the x shift
        # (sign convention: shift to subtract from x).
        shift_x, shift_y = shifts
        assert abs(shift_x[1] - 0.1 * PX) < 0.05 * PX
        # First channel is the reference: ~0 shift
        assert abs(shift_x[0]) < 1e-6
        assert abs(shift_y[0]) < 1e-6

    def test_align_from_picked_box_recovers_known_shift(
        self, locs_copy, info, origami_picks
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1 * PX
        boxes = [
            (
                (x - PICK_SIZE / 2, y - PICK_SIZE / 2),
                (x + PICK_SIZE / 2, y + PICK_SIZE / 2),
            )
            for x, y in origami_picks
        ]
        _, _, shifts = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=boxes,
            pick_shape="Box",
            return_shifts=True,
        )
        assert abs(shifts[0][1] - 0.1 * PX) < 0.05 * PX  # x shift of channel 2

    def test_align_from_picked_brush_recovers_known_shift(
        self, locs_copy, info, origami_picks
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1 * PX
        picks = [
            [(PICK_SIZE, [(x - 0.2 * PX, y), (x + 0.2 * PX, y)])]
            for x, y in origami_picks
        ]
        _, _, shifts = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=picks,
            pick_shape="Brush",
            return_shifts=True,
        )
        assert abs(shifts[0][1] - 0.1 * PX) < 0.05 * PX  # x shift of channel 2

    def test_align_from_picked_invalid_shape_raises(
        self, locs_copy, info, origami_picks
    ):
        with pytest.raises(ValueError, match="Unrecognized pick shape"):
            postprocess.align_from_picked(
                [locs_copy, locs_copy.copy()],
                [info, info],
                picks=origami_picks,
                pick_shape="Hexagon",
                pick_size=1.0,
            )

    @staticmethod
    def _with_stray_locs(locs: pd.DataFrame, x: float) -> pd.DataFrame:
        """``locs`` plus three localizations at ``x`` (nm), beyond the x
        range of the test data (about 5 to 26 camera pixels)."""
        stray = locs.iloc[:3].copy()
        stray["x"] = x
        return pd.concat([locs, stray], ignore_index=True)

    @pytest.mark.parametrize("keep", [True, False])
    def test_align_rcc_non_overlapping_locs(self, locs_copy, info, keep):
        """Localizations of one channel that lie beyond the other
        channel after alignment are kept by default and removed with
        keep_non_overlapping=False."""
        a = locs_copy
        b = a.copy()
        b["x"] += 2.0 * PX
        b = self._with_stray_locs(b, 30.5 * PX)
        aligned, _ = postprocess.align_rcc(
            [a, b], [info, info], keep_non_overlapping=keep
        )
        if keep:
            assert [len(_) for _ in aligned] == [len(a), len(b)]
        else:
            assert len(aligned[0]) <= len(a)
            assert len(aligned[1]) <= len(b) - 3
            assert aligned[1]["x"].max() <= aligned[0]["x"].max()
            assert aligned[1].index.equals(pd.RangeIndex(len(aligned[1])))

    @pytest.mark.parametrize("keep", [True, False])
    def test_align_from_picked_non_overlapping_locs(
        self, locs_copy, info, origami_picks, keep
    ):
        a = locs_copy
        b = a.copy()
        b["x"] += 0.1 * PX
        b = self._with_stray_locs(b, 29.0 * PX)
        n_a, n_b = len(a), len(b)
        aligned, _ = postprocess.align_from_picked(
            [a, b],
            [info, info],
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
            keep_non_overlapping=keep,
        )
        if keep:
            assert [len(_) for _ in aligned] == [n_a, n_b]
        else:
            assert len(aligned[1]) <= n_b - 3
            assert aligned[1]["x"].max() <= aligned[0]["x"].max()

    def test_align_non_overlapping_locs(self, locs_copy, info):
        a = locs_copy
        b = self._with_stray_locs(a.copy(), 30.5 * PX)
        n_b = len(b)
        aligned, _ = postprocess.align(
            [a, b], [info, info], keep_non_overlapping=False
        )
        assert len(aligned[1]) <= n_b - 3


class TestOverlap:
    @staticmethod
    def _locs(x_min: float, x_max: float, y_min: float, y_max: float):
        return pd.DataFrame(
            {"x": [x_min, x_max, x_min], "y": [y_min, y_max, y_max]}
        )

    def test_overlap_region_is_the_intersection(self):
        locs = [self._locs(0, 10, 0, 8), self._locs(5, 15, 2, 20)]
        assert postprocess.overlap_region(locs) == ((5, 10), (2, 8))

    def test_overlap_region_none_if_disjoint(self):
        locs = [self._locs(0, 4, 0, 4), self._locs(5, 9, 0, 4)]
        assert postprocess.overlap_region(locs) is None

    def test_overlap_region_none_if_a_channel_is_empty(self):
        locs = [self._locs(0, 4, 0, 4), self._locs(0, 4, 0, 4).iloc[:0]]
        assert postprocess.overlap_region(locs) is None

    def test_remove_non_overlapping_locs(self):
        a = pd.DataFrame({"x": [0.0, 5.0, 10.0], "y": [0.0, 5.0, 10.0]})
        b = pd.DataFrame({"x": [4.0, 6.0, 14.0], "y": [4.0, 6.0, 14.0]})
        out = postprocess.remove_non_overlapping_locs([a, b])
        # overlap is x, y in [4, 10] (borders included)
        assert out[0]["x"].tolist() == [5.0, 10.0]
        assert out[1]["x"].tolist() == [4.0, 6.0]
        assert out[0].index.tolist() == [0, 1]
        # the input is not modified
        assert len(a) == 3 and len(b) == 3

    def test_remove_non_overlapping_locs_raises_if_disjoint(self):
        locs = [self._locs(0, 4, 0, 4), self._locs(5, 9, 0, 4)]
        with pytest.raises(ValueError):
            postprocess.remove_non_overlapping_locs(locs)


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
            picks=origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        merged = pd.concat(picked, ignore_index=True)
        merged, merged_info = postprocess.link(merged, info)
        return postprocess.compute_dark_times(merged, merged_info)[0]

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

    def test_one_neighbor_in_another_set_is_a_column(self):
        """``KDTree.query`` with k=1 returns a flat array."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 2))
        Y = rng.standard_normal((25, 2))
        out = postprocess.nn_analysis(X, Y, nn_count=1)
        assert out.shape == (20, 1)


# ---------------------------------------------------------------------------
# Coordinate-based colocalization (CBC)
# ---------------------------------------------------------------------------


def _cbc_locs(centers, n, seed, sigma=26.0, z=False):
    """Localizations scattered around the given centers (nm)."""
    rng = np.random.default_rng(seed)
    xy = centers[rng.integers(len(centers), size=n)]
    xy = xy + rng.normal(0.0, sigma, size=xy.shape)
    locs = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.uint32),
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
        }
    )
    if z:
        locs["z"] = rng.normal(0.0, 30.0, size=n).astype(np.float32)
    return locs


@pytest.mark.filterwarnings("ignore:.*not linked")
class TestCbc:
    INFO = [{"Pixelsize": 130.0}]

    @pytest.fixture
    def channels(self):
        rng = np.random.default_rng(0)
        centers = rng.uniform(0, 13_000, size=(100, 2))
        a = _cbc_locs(centers, 4000, 1)
        b = _cbc_locs(centers[:50], 2000, 2)  # half of a's clusters
        c = _cbc_locs(rng.uniform(0, 13_000, size=(30, 2)), 1000, 3)
        return a, b, c

    def test_colocalized_above_uncorrelated(self, channels):
        a, b, c = channels
        out, _ = postprocess.cbc(
            a, self.INFO, partner_locs=[b, c], partner_names=["b", "c"]
        )
        assert out["cbc_b"].mean() > out["cbc_c"].mean() + 0.2
        assert abs(out["cbc_c"].median()) < 0.05
        assert out["cbc_b"].dtype == np.float32
        assert out["cbc_b"].between(-1, 1).all()

    def test_not_symmetric(self, channels):
        """b sits on half of a's clusters: every b localization is
        colocalized with a, but only half of a's with b."""
        a, b, _ = channels
        ab, _ = postprocess.cbc(a, self.INFO, partner_locs=b)
        ba, _ = postprocess.cbc(b, self.INFO, partner_locs=a)
        assert ba["cbc"].median() > ab["cbc"].median() + 0.2

    def test_returns_copy_and_metadata(self, channels):
        a, b, _ = channels
        info = list(self.INFO)
        out, new_info = postprocess.cbc(
            a, info, partner_locs=b, r_max=150, steps=50
        )
        assert "cbc" not in a.columns
        assert list(out.columns) == list(a.columns) + ["cbc"]
        assert len(info) == 1 and len(new_info) == 2
        entry = new_info[-1]
        assert entry["CBC columns"] == ["cbc"]
        assert entry["Max. radius (nm)"] == 150
        assert entry["Radius steps"] == 50
        assert entry["Dimensions"] == 2

    def test_default_columns_for_partner_list(self, channels):
        a, b, c = channels
        out, _ = postprocess.cbc(a, self.INFO, partner_locs=[b, c])
        assert {"cbc_0", "cbc_1"} <= set(out.columns)

    def test_partners_and_chunks_do_not_change_values(self, channels):
        """Shared self-counts and chunking are only optimizations."""
        a, b, c = channels
        both, _ = postprocess.cbc(
            a,
            self.INFO,
            partner_locs=[b, c],
            partner_names=["b", "c"],
            chunk_size=777,
        )
        single, _ = postprocess.cbc(a, self.INFO, partner_locs=c)
        np.testing.assert_array_equal(both["cbc_c"], single["cbc"])

    def test_3d_used_when_all_have_z(self):
        rng = np.random.default_rng(4)
        centers = rng.uniform(0, 6500, size=(40, 2))
        a = _cbc_locs(centers, 1000, 5, z=True)
        b = _cbc_locs(centers, 1000, 6, z=True)
        _, info3 = postprocess.cbc(a, self.INFO, partner_locs=b)
        assert info3[-1]["Dimensions"] == 3
        _, info2 = postprocess.cbc(a, self.INFO, partner_locs=b, use_z=False)
        assert info2[-1]["Dimensions"] == 2
        with pytest.raises(ValueError):
            postprocess.cbc(
                a, self.INFO, partner_locs=b.drop(columns="z"), use_z=True
            )

    def test_partner_names_give_short_columns(self, channels):
        a, b, c = channels
        out, info = postprocess.cbc(
            a,
            self.INFO,
            partner_locs=[b, c],
            partner_names=["Cav2.1", "PSD 95"],
        )
        assert {"cbc_Cav2_1", "cbc_PSD_95"} <= set(out.columns)
        assert info[-1]["CBC partner names"] == ["Cav2.1", "PSD 95"]
        assert info[-1]["CBC columns"] == ["cbc_Cav2_1", "cbc_PSD_95"]

    def test_cbc_column(self):
        assert postprocess._cbc_column("Cav2.1") == "cbc_Cav2_1"
        assert postprocess._cbc_column(" a--b ") == "cbc_a_b"
        with pytest.raises(ValueError):
            postprocess._cbc_column(".")

    def test_all_pairs_matches_single_calls(self, channels):
        """Both directions of every pair, identical to ``cbc``."""
        a, b, c = channels
        tracker = RecordingProgress()
        out, infos = postprocess.cbc_all_pairs(
            [a, b, c],
            [self.INFO] * 3,
            names=["A", "B", "C"],
            undefined=-2.0,
            progress=tracker,
        )
        assert len(out) == len(infos) == 3
        assert set(out[0].columns) - set(a.columns) == {"cbc_B", "cbc_C"}
        assert set(out[1].columns) - set(b.columns) == {"cbc_A", "cbc_C"}
        assert infos[1][-1]["CBC name"] == "B"
        assert infos[1][-1]["Undefined CBC value"] == -2.0
        single, _ = postprocess.cbc(
            b,
            self.INFO,
            partner_locs=[a, c],
            partner_names=["A", "C"],
            undefined=-2.0,
        )
        for column in ("cbc_A", "cbc_C"):
            np.testing.assert_array_equal(out[1][column], single[column])
        # both directions differ (not symmetric)
        assert out[1]["cbc_A"].median() > out[0]["cbc_B"].median() + 0.2
        # cumulative progress over all channels
        assert_phases(tracker, ["Calculating CBC"])
        assert tracker.phases[0].maximum == len(a) + len(b) + len(c)

    def test_all_pairs_invalid_arguments(self, channels):
        a, b, _ = channels
        with pytest.raises(ValueError):
            postprocess.cbc_all_pairs([a], [self.INFO])
        with pytest.raises(ValueError):
            postprocess.cbc_all_pairs([a, b], [self.INFO])
        with pytest.raises(ValueError):
            postprocess.cbc_all_pairs([a, b], [self.INFO] * 2, names=["x"])
        with pytest.raises(ValueError):
            postprocess.cbc_all_pairs(
                [a, b], [self.INFO] * 2, names=["x", "x"]
            )

    def test_undefined_without_neighbors(self):
        a = pd.DataFrame({"x": [0.0, 1.3, 6500.0], "y": [0.0, 0.0, 6500.0]})
        b = pd.DataFrame({"x": [2.6, 3.9], "y": [0.0, 1.3]})
        out, _ = postprocess.cbc(
            a, self.INFO, partner_locs=b, undefined=np.nan
        )
        assert np.isnan(out["cbc"].iloc[2])  # isolated localization
        assert np.isfinite(out["cbc"].iloc[:2]).all()
        out, _ = postprocess.cbc(a, self.INFO, partner_locs=b.iloc[:0])
        assert (out["cbc"] == 0).all()  # empty partner
        out, _ = postprocess.cbc(a, self.INFO, partner_locs=b, undefined=-1.5)
        assert out["cbc"].iloc[2] == -1.5

    def test_remove_undefined(self):
        a = pd.DataFrame(
            {"x": [0.0, 1.3, 6500.0, 2.6], "y": [0.0, 0.0, 6500.0, 1.3]}
        )
        b = pd.DataFrame({"x": [2.6, 3.9], "y": [0.0, 1.3]})
        c = pd.DataFrame({"x": [0.0], "y": [2.6]})
        kept, info = postprocess.cbc(
            a, self.INFO, partner_locs=b, remove_undefined=True
        )
        assert len(kept) == 3  # the isolated localization is removed
        assert 6500.0 not in kept["x"].to_numpy()
        assert list(kept.index) == [0, 1, 2]
        assert info[-1]["Localizations without neighbors removed"] == 1
        assert "Undefined CBC value" not in info[-1]
        # removed if undefined against any partner
        kept, _ = postprocess.cbc(
            a, self.INFO, partner_locs=[b, c.iloc[:0]], remove_undefined=True
        )
        assert len(kept) == 0
        all_locs, all_infos = postprocess.cbc_all_pairs(
            [a, b, c], [self.INFO] * 3, remove_undefined=True
        )
        assert len(all_locs[0]) == 3
        assert all_infos[0][-1]["Localizations without neighbors removed"] == 1

    def test_warns_if_not_linked(self, channels):
        a, b, _ = channels
        with pytest.warns(UserWarning, match="not linked"):
            postprocess.cbc(a, self.INFO, partner_locs=b)
        linked_a = a.assign(len=np.ones(len(a), dtype=np.int32))
        linked_b = b.assign(len=np.ones(len(b), dtype=np.int32))
        with pytest.warns(UserWarning, match="not linked"):
            # partner unlinked
            postprocess.cbc(linked_a, self.INFO, partner_locs=b)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            postprocess.cbc(linked_a, self.INFO, partner_locs=linked_b)
        with pytest.warns(UserWarning, match="not linked"):
            postprocess.cbc_all_pairs([linked_a, b], [self.INFO] * 2)

    def test_invalid_arguments(self, channels):
        a, b, c = channels
        with pytest.raises(ValueError):
            postprocess.cbc(
                a, self.INFO, partner_locs=[b, c], partner_names=["x"]
            )
        with pytest.raises(ValueError):  # same column cbc_x_y
            postprocess.cbc(
                a, self.INFO, partner_locs=[b, c], partner_names=["x.y", "x y"]
            )
        with pytest.raises(ValueError):
            postprocess.cbc(a, self.INFO, partner_locs=b, partner_names="...")
        with pytest.raises(ValueError):
            postprocess.cbc(a, self.INFO, partner_locs=b, steps=1)
        with pytest.raises(ValueError):
            postprocess.cbc(a, self.INFO, partner_locs=[])


# ---------------------------------------------------------------------------
# RESI
# ---------------------------------------------------------------------------


class TestResi:
    def test_resi_2d_combines_channels(self, locs, info):
        out, new_info = postprocess.resi(
            [locs.copy(), locs.copy()],
            [info, info],
            radius_xy=2.0,
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
                radius_xy=2.0,
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
                radius_xy=2.0,
                min_locs=2,
            )

    def test_resi_per_channel_list_length_validated(self, locs, info):
        with pytest.raises(ValueError):
            postprocess.resi(
                [locs.copy(), locs.copy()],
                [info, info],
                radius_xy=[2.0],
                min_locs=2,
            )
        with pytest.raises(ValueError):
            postprocess.resi(
                [locs.copy(), locs.copy()],
                [info, info],
                radius_xy=2.0,
                min_locs=[2, 3, 4],
            )


# ---------------------------------------------------------------------------
# Ripley's H function
# ---------------------------------------------------------------------------


def _ripley_locs(x_nm, y_nm):
    """Molecules at the given positions in nm."""
    return pd.DataFrame(
        {
            "frame": np.zeros(len(x_nm), dtype=np.uint32),
            "x": np.asarray(x_nm, dtype=np.float64),
            "y": np.asarray(y_nm, dtype=np.float64),
            "n_events": np.ones(len(x_nm), dtype=np.int32),
        }
    )


class TestRipleyH:
    SIZE = 10_000.0  # nm, field of view (100 camera pixels of 100 nm)
    INFO = [{"Pixelsize": 100.0, "Width": 100, "Height": 100, "Frames": 1}]

    @pytest.fixture
    def random_locs(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, self.SIZE, size=(2, 20_000))
        return _ripley_locs(x, y)

    @pytest.fixture
    def clustered_locs(self):
        """10 molecules within 50 nm of each of 200 random centers."""
        rng = np.random.default_rng(1)
        centers = rng.uniform(500, self.SIZE - 500, size=(200, 2))
        r = 50 * np.sqrt(rng.uniform(0, 1, 2000))
        angle = rng.uniform(0, 2 * np.pi, 2000)
        x = np.repeat(centers[:, 0], 10) + r * np.cos(angle)
        y = np.repeat(centers[:, 1], 10) + r * np.sin(angle)
        return _ripley_locs(x, y)

    def test_random_is_zero_within_envelope(self, random_locs):
        result = postprocess.ripley_h(
            random_locs, self.INFO, r_max=1000, steps=50, n_simulations=19
        )
        assert result["r"][0] == 20 and result["r"][-1] == 1000
        assert np.abs(result["h_mean"]).max() < 5  # nm
        # the envelope brackets 0 and is narrow for 20,000 points
        assert (result["envelope_low"] < 0.5).all()
        assert (result["envelope_high"] > -0.5).all()
        assert (result["envelope_high"] - result["envelope_low"]).max() < 10

    def test_edge_correction_removes_the_downward_bias(self, random_locs):
        kwargs = dict(r_max=1000, steps=10, n_simulations=0)
        corrected = postprocess.ripley_h(random_locs, self.INFO, **kwargs)
        raw = postprocess.ripley_h(
            random_locs, self.INFO, edge_correction=False, **kwargs
        )
        assert raw["h_mean"][-1] < -30
        assert abs(corrected["h_mean"][-1]) < 5
        assert corrected["edge_correction"] and not raw["edge_correction"]

    def test_clusters_peak_between_radius_and_diameter(self, clustered_locs):
        result = postprocess.ripley_h(
            clustered_locs, self.INFO, r_max=500, steps=50, n_simulations=5
        )
        assert 50 <= result["r_h_max"][0] <= 100
        assert result["h_mean"].max() > 100
        assert (result["h_mean"][:10] > result["envelope_high"][:10]).all()

    @pytest.mark.parametrize(
        "picks, shape, size",
        [
            ([(5000, 5000), (3000, 3000)], "Circle", 4000),
            ([(5000, 5000)], "Square", 4000),
            ([((2000, 3000), (8000, 7000))], "Rectangle", 3000),
            ([((1000, 2000), (6000, 9000))], "Box", None),
            (
                [
                    [
                        (1000, 1000),
                        (9000, 1500),
                        (6000, 5000),
                        (8500, 9000),
                        (1500, 8000),
                        (1000, 1000),
                    ]
                ],
                "Polygon",
                None,
            ),
            (
                [
                    [
                        (2000.0, [(2000, 2000), (5000, 6000), (8000, 3000)]),
                        (1500.0, [(5000, 8000)]),
                    ]
                ],
                "Brush",
                None,
            ),
        ],
    )
    def test_random_is_zero_in_every_pick_shape(
        self, random_locs, picks, shape, size
    ):
        result = postprocess.ripley_h(
            random_locs,
            self.INFO,
            picks=picks,
            pick_shape=shape,
            pick_size=size,
            r_max=1000,
            steps=10,
            n_simulations=0,
        )
        np.testing.assert_allclose(
            result["areas"], lib.pick_areas(picks, shape, size), rtol=0.02
        )
        # the localizations counted are the picked ones
        picked = postprocess.picked_locs(
            random_locs,
            self.INFO,
            picks=picks,
            pick_shape=shape,
            pick_size=size,
            add_group=False,
        )
        np.testing.assert_allclose(
            result["n_locs"], [len(p) for p in picked], atol=2
        )
        assert np.abs(result["h_mean"]).max() < 10

    def test_regions_without_locs_or_area_are_skipped(self, random_locs):
        picks = [
            [(1000, 1000), (4000, 1000), (4000, 4000), (1000, 1000)],
            [(100, 100), (200, 200)],  # open polygon
            [(2e4, 2e4), (2.1e4, 2e4), (2.1e4, 2.1e4), (2e4, 2e4)],  # outside
        ]
        result = postprocess.ripley_h(
            random_locs,
            self.INFO,
            picks=picks,
            pick_shape="Polygon",
            n_simulations=2,
        )
        assert result["n_locs"][0] > 0
        assert list(result["n_locs"][1:]) == [0, 0]
        assert np.isnan(result["h"][1:]).all()
        assert np.isnan(result["r_h_max"][1:]).all()
        np.testing.assert_array_equal(result["h_mean"], result["h"][0])
        with pytest.raises(ValueError, match="at least two"):
            postprocess.ripley_h(
                random_locs, self.INFO, picks=picks[1:], pick_shape="Polygon"
            )

    def test_mean_over_picks(self, random_locs):
        picks = [(3000, 3000), (7000, 7000)]
        result = postprocess.ripley_h(
            random_locs,
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=3000,
            n_simulations=0,
        )
        np.testing.assert_allclose(result["h_mean"], result["h"].mean(axis=0))
        assert result["envelope_low"] is None
        assert result["h"].shape == (2, 100)

    def test_envelope_is_reproducible(self, clustered_locs):
        kwargs = dict(r_max=300, steps=10, n_simulations=3)
        a = postprocess.ripley_h(clustered_locs, self.INFO, **kwargs)
        b = postprocess.ripley_h(clustered_locs, self.INFO, **kwargs)
        c = postprocess.ripley_h(
            clustered_locs, self.INFO, random_seed=1, **kwargs
        )
        np.testing.assert_array_equal(a["envelope_low"], b["envelope_low"])
        assert not np.array_equal(a["envelope_low"], c["envelope_low"])

    def test_polygon_edge_fraction_is_exact(self):
        # unit square [0, 1000]^2, circle of radius 300 around a point
        # 100 nm from the left edge: arccos(100 / 300) / pi outside
        region = postprocess._polygon_region(
            np.array([0, 1000, 1000, 0.0]), np.array([0, 0, 1000, 1000.0])
        )
        geometry = (region.kind, region.circle, region.segments)
        fraction = postprocess._circle_fraction_inside(
            100.0, 500.0, 300.0, *geometry
        )
        assert fraction == pytest.approx(1 - np.arccos(1 / 3) / np.pi)
        # corner: a quarter of the circle inside
        fraction = postprocess._circle_fraction_inside(
            0.0, 0.0, 300.0, *geometry
        )
        assert fraction == pytest.approx(0.25)
        assert postprocess._circle_fraction_inside(
            500.0, 500.0, 300.0, *geometry
        ) == pytest.approx(1.0)

    def test_zero_radius_fraction_in_a_circle(self):
        region = postprocess._pick_region((0.0, 0.0), "Circle", 10.0)
        geometry = (region.kind, region.circle, region.segments)
        fractions = [
            postprocess._circle_fraction_inside(x, 0.0, 0.0, *geometry)
            for x in (0.0, 2.0, 7.0)
        ]
        assert fractions == [1.0, 1.0, 0.0]

    def test_no_uniform_points(self):
        region = postprocess._pick_region((0.0, 0.0), "Circle", 10.0)
        x, y = postprocess._uniform_in_region(
            0, region, np.random.default_rng(0)
        )
        assert x.shape == y.shape == (0,)

    def test_localizations_that_are_not_molecules_warn(self, random_locs):
        locs = random_locs.drop(columns="n_events")
        locs["len"] = 1  # linked localizations are still not molecules
        with pytest.warns(UserWarning, match="not molecules"):
            postprocess.ripley_h(locs, self.INFO, steps=5, n_simulations=0)
        # cluster centers, G5M molecules and combined clusters pass
        for column in ("n_events", "std_frame"):
            locs = random_locs.drop(columns="n_events")
            locs[column] = 1
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                postprocess.ripley_h(locs, self.INFO, steps=5, n_simulations=0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(r_max=0),
            dict(steps=1),
            dict(n_simulations=-1),
            dict(picks=[(100, 100)], pick_shape="Hexagon"),
            dict(picks=[(100, 100)], pick_shape="Circle"),  # no size
        ],
    )
    def test_invalid_parameters(self, random_locs, kwargs):
        with pytest.raises(ValueError):
            postprocess.ripley_h(random_locs, self.INFO, **kwargs)

    def test_plot_and_tables(self, clustered_locs):
        picks = [(3000, 3000), (7000, 7000), (50_000, 50_000)]
        result = postprocess.ripley_h(
            clustered_locs,
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=4000,
            r_max=300,
            steps=30,
            n_simulations=2,
        )
        fig = postprocess.plot_ripley_h(result)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "r (nm)"
        labels = ax.get_legend_handles_labels()[1]
        assert "Regions (2)" in labels and "Mean" in labels
        curves, regions = postprocess._ripley_h_tables(result)
        assert list(curves.columns) == [
            "r_nm",
            "h_mean_nm",
            "envelope_low_nm",
            "envelope_high_nm",
            "h_region_0_nm",
            "h_region_1_nm",
            "h_region_2_nm",
        ]
        assert len(curves) == 30
        assert list(regions["n_locs"]) == list(result["n_locs"])
        # 4 um diameter
        assert regions["area_um2"].iloc[0] == pytest.approx(np.pi * 2**2)


@pytest.mark.filterwarnings("ignore:r_max.*third of the z range")
class TestRipleyH3D:
    INFO = TestRipleyH.INFO
    DEPTH = 1000.0  # nm

    @pytest.fixture
    def random_locs(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 10_000, size=(2, 20_000))
        locs = _ripley_locs(x, y)
        locs["z"] = rng.uniform(-self.DEPTH / 2, self.DEPTH / 2, len(locs))
        return locs

    def test_sphere_fraction_follows_archimedes(self):
        region = postprocess._polygon_region(
            np.array([0, 1e4, 1e4, 0.0]), np.array([0, 0, 1e4, 1e4])
        )
        geometry = (region.kind, region.circle, region.segments)
        # between two planes 300 nm above and below: 300 / 400 inside
        fraction = postprocess._sphere_fraction_inside(
            5e3, 5e3, 0.0, 400.0, -300.0, 300.0, 5e3, *geometry
        )
        assert fraction == pytest.approx(0.75)
        # corner of the box on the bottom plane: 1/4 in xy, 1/2 in z
        fraction = postprocess._sphere_fraction_inside(
            0.0, 0.0, -300.0, 200.0, -300.0, 300.0, 0.0, *geometry
        )
        assert fraction == pytest.approx(0.125)

    def test_random_is_zero_and_needs_the_edge_correction(self, random_locs):
        kwargs = dict(r_max=300, steps=30, use_z=True)
        result = postprocess.ripley_h(
            random_locs, self.INFO, n_simulations=5, **kwargs
        )
        # few pairs at the shortest radii, so they are noisier
        assert np.abs(result["h_mean"][:3]).max() < 5
        assert np.abs(result["h_mean"][3:]).max() < 1
        assert result["use_z"] and result["simulated_z"] == "uniform"
        z = random_locs["z"]
        np.testing.assert_allclose(result["z_ranges"], [[z.min(), z.max()]])
        np.testing.assert_allclose(
            result["volumes"], result["areas"] * (z.max() - z.min())
        )
        raw = postprocess.ripley_h(
            random_locs,
            self.INFO,
            n_simulations=0,
            edge_correction=False,
            **kwargs,
        )
        assert raw["h_mean"][-1] < -10

    def test_3d_clusters_peak(self):
        rng = np.random.default_rng(3)
        centers = np.column_stack(
            [
                rng.uniform(500, 9500, size=(300, 2)),
                rng.uniform(-300, 300, size=300),
            ]
        )
        offsets = rng.normal(size=(3000, 3))
        offsets *= (
            50
            * np.cbrt(rng.uniform(0, 1, 3000))
            / np.linalg.norm(offsets, axis=1)
        )[:, None]
        xyz = np.repeat(centers, 10, axis=0) + offsets
        locs = _ripley_locs(xyz[:, 0], xyz[:, 1])
        locs["z"] = xyz[:, 2]
        with pytest.warns(UserWarning, match="third of the z range"):
            result = postprocess.ripley_h(
                locs,
                self.INFO,
                r_max=300,
                steps=30,
                n_simulations=0,
                use_z=True,
            )
        assert 50 <= result["r_h_max"][0] <= 100
        # in 2D, the clusters are smeared out by the projection
        flat = postprocess.ripley_h(
            locs, self.INFO, r_max=300, steps=30, n_simulations=0
        )
        assert result["h_mean"].max() > flat["h_mean"].max()

    @pytest.mark.parametrize("simulated_z", ["uniform", "observed"])
    @pytest.mark.parametrize(
        "picks, shape, size",
        [
            ([(5000, 5000), (3000, 3000)], "Circle", 4000),
            ([((2000, 3000), (8000, 7000))], "Rectangle", 3000),
            (
                [
                    [
                        (1000, 1000),
                        (9000, 1500),
                        (6000, 5000),
                        (8500, 9000),
                        (1500, 8000),
                        (1000, 1000),
                    ]
                ],
                "Polygon",
                None,
            ),
            (
                [
                    [
                        (2000.0, [(2000, 2000), (5000, 6000), (8000, 3000)]),
                        (1500.0, [(5000, 8000)]),
                    ]
                ],
                "Brush",
                None,
            ),
        ],
    )
    def test_random_is_zero_in_picks_with_a_manual_z_range(
        self, random_locs, picks, shape, size, simulated_z
    ):
        result = postprocess.ripley_h(
            random_locs,
            self.INFO,
            picks=picks,
            pick_shape=shape,
            pick_size=size,
            r_max=300,
            steps=10,
            n_simulations=3,
            use_z=True,
            z_range=(-400, 400),
            simulated_z=simulated_z,
        )
        np.testing.assert_array_equal(
            result["z_ranges"], [[-400, 400]] * len(picks)
        )
        assert np.abs(result["h_mean"]).max() < 5
        assert (result["envelope_low"] < result["envelope_high"]).all()

    def test_manual_z_range_ignores_localizations_outside(self, random_locs):
        result = postprocess.ripley_h(
            random_locs,
            self.INFO,
            n_simulations=0,
            steps=5,
            use_z=True,
            z_range=(0, 400),
        )
        z = random_locs["z"]
        assert result["n_locs"][0] == ((z >= 0) & (z <= 400)).sum()

    def test_tables_and_plot_in_3d(self, random_locs):
        result = postprocess.ripley_h(
            random_locs, self.INFO, n_simulations=0, steps=5, use_z=True
        )
        _, regions = postprocess._ripley_h_tables(result)
        columns = {"z_min_nm", "z_max_nm", "volume_um3", "density_per_um3"}
        assert columns <= set(regions.columns)
        assert "density_per_um2" not in regions.columns
        assert regions["density_per_um3"].iloc[0] == pytest.approx(
            len(random_locs) / regions["volume_um3"].iloc[0]
        )
        title = postprocess.plot_ripley_h(result).axes[0].get_title()
        assert title.endswith("(3D)")

    def test_regions_without_depth_are_skipped(self, random_locs):
        locs = random_locs.copy()
        locs["z"] = 0.0
        with pytest.raises(ValueError, match="different z"):
            postprocess.ripley_h(locs, self.INFO, use_z=True)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(z_range=(100, 100)),
            dict(simulated_z="gaussian"),
        ],
    )
    def test_invalid_3d_parameters(self, random_locs, kwargs):
        with pytest.raises(ValueError):
            postprocess.ripley_h(random_locs, self.INFO, use_z=True, **kwargs)
        with pytest.raises(ValueError, match="requires z"):
            postprocess.ripley_h(
                random_locs.drop(columns="z"), self.INFO, use_z=True
            )


@pytest.mark.filterwarnings("ignore:r_max.*third of the z range")
class TestRdf:
    INFO = TestRipleyH.INFO

    @pytest.fixture
    def random_locs(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 10_000, size=(2, 20_000))
        locs = _ripley_locs(x, y)
        locs["z"] = rng.uniform(-500, 500, len(locs))
        return locs

    def test_random_is_one(self, random_locs):
        result = postprocess.rdf(random_locs, self.INFO, n_simulations=9)
        # ring centers of 10 nm wide rings
        np.testing.assert_allclose(result["r"][[0, -1]], [5, 495])
        assert np.abs(result["g_mean"] - 1).max() < 0.05
        assert (result["envelope_low"] < 1).all()
        assert (result["envelope_high"] > 1).all()
        raw = postprocess.rdf(
            random_locs, self.INFO, n_simulations=0, edge_correction=False
        )
        assert raw["g_mean"][-1] < 0.96

    def test_random_is_one_in_3d(self, random_locs):
        result = postprocess.rdf(
            random_locs,
            self.INFO,
            r_max=300,
            steps=30,
            n_simulations=0,
            use_z=True,
        )
        assert np.abs(result["g_mean"][5:] - 1).max() < 0.05
        assert result["use_z"] and result["volumes"] is not None

    @pytest.mark.parametrize("use_z", [False, True])
    def test_integrates_to_ripleys_k(self, random_locs, use_z):
        kwargs = dict(r_max=300, steps=30, n_simulations=0, use_z=use_z)
        h = postprocess.ripley_h(random_locs, self.INFO, **kwargs)["h"][0]
        g = postprocess.rdf(random_locs, self.INFO, **kwargs)["g"][0]
        edges = np.arange(31) * 10.0
        if use_z:
            k = np.cumsum(g * 4 / 3 * np.pi * np.diff(edges**3))
            h_from_g = np.cbrt(3 * k / (4 * np.pi)) - edges[1:]
        else:
            k = np.cumsum(g * np.pi * np.diff(edges**2))
            h_from_g = np.sqrt(k / np.pi) - edges[1:]
        np.testing.assert_allclose(h_from_g, h, atol=1e-9)

    def test_clusters_raise_g_at_short_distances(self):
        rng = np.random.default_rng(1)
        centers = rng.uniform(500, 9500, size=(200, 2))
        r = 50 * np.sqrt(rng.uniform(0, 1, 2000))
        angle = rng.uniform(0, 2 * np.pi, 2000)
        x = np.repeat(centers[:, 0], 10) + r * np.cos(angle)
        y = np.repeat(centers[:, 1], 10) + r * np.sin(angle)
        result = postprocess.rdf(
            _ripley_locs(x, y), self.INFO, n_simulations=3
        )
        # within the clusters (diameter 100 nm), far above random
        assert (result["g_mean"][:5] > 10).all()
        assert (result["g_mean"][:5] > result["envelope_high"][:5]).all()

    def test_plot_and_tables(self, random_locs):
        picks = [(3000, 3000), (7000, 7000)]
        result = postprocess.rdf(
            random_locs,
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=4000,
            n_simulations=2,
        )
        ax = postprocess.plot_rdf(result).axes[0]
        assert ax.get_ylabel() == "g(r)"
        assert ax.get_xlim() == (0, 500)
        curves, regions = postprocess._rdf_tables(result)
        assert list(curves.columns) == [
            "r_nm",
            "g_mean",
            "envelope_low",
            "envelope_high",
            "g_region_0",
            "g_region_1",
        ]
        assert "h_max_nm" not in regions.columns
        assert list(regions["n_locs"]) == list(result["n_locs"])


class TestCrossPairStatistics:
    INFO = TestRipleyH.INFO
    SIZE = 10_000.0

    def _random(self, n, seed, z=False):
        rng = np.random.default_rng(seed)
        locs = _ripley_locs(*rng.uniform(0, self.SIZE, size=(2, n)))
        if z:
            locs["z"] = rng.uniform(-500, 500, n)
        return locs

    def _clustered(self, seed):
        """10 molecules within 50 nm of each of 300 random centers."""
        rng = np.random.default_rng(seed)
        centers = rng.uniform(0, self.SIZE, size=(300, 2))
        r = 50 * np.sqrt(rng.uniform(0, 1, 3000))
        angle = rng.uniform(0, 2 * np.pi, 3000)
        x = np.repeat(centers[:, 0], 10) + r * np.cos(angle)
        y = np.repeat(centers[:, 1], 10) + r * np.sin(angle)
        inside = (x > 0) & (x < self.SIZE) & (y > 0) & (y < self.SIZE)
        return _ripley_locs(x[inside], y[inside])

    @pytest.mark.parametrize("cross_null", ["shift", "labels"])
    def test_independent_random_channels(self, cross_null):
        a, b = self._random(5000, 0), self._random(4000, 1)
        h = postprocess.ripley_h(
            a,
            self.INFO,
            partner_locs=b,
            cross_null=cross_null,
            steps=50,
            n_simulations=9,
        )
        g = postprocess.rdf(a, self.INFO, partner_locs=b, n_simulations=0)
        assert np.abs(h["h_mean"]).max() < 2
        assert np.abs(g["g_mean"][5:] - 1).max() < 0.1
        assert h["cross_null"] == cross_null
        assert list(h["n_locs"]) == [5000]
        assert list(h["n_locs_partner"]) == [4000]
        assert h["simulated_z"] is None

    def test_symmetric_in_the_channels(self):
        a, b = self._random(3000, 0), self._random(2000, 1)
        picks = [(3000, 3000), (7000, 7000)]
        kwargs = dict(n_simulations=0, steps=20)
        ab = postprocess.ripley_h(
            a,
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=4000,
            partner_locs=b,
            **kwargs,
        )
        ba = postprocess.ripley_h(
            b,
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=4000,
            partner_locs=a,
            **kwargs,
        )
        np.testing.assert_allclose(ab["h"], ba["h"], atol=1e-9)

    def test_colocalized_channels(self):
        rng = np.random.default_rng(2)
        a = self._random(3000, 0)
        b = a.copy()
        b[["x", "y"]] += rng.normal(0, 10.0, size=(len(b), 2))  # nm
        result = postprocess.rdf(
            a, self.INFO, partner_locs=b, r_max=200, steps=20, n_simulations=5
        )
        assert result["g_mean"][0] > 10
        assert (result["g_mean"][:2] > result["envelope_high"][:2]).all()

    def test_null_models_of_independent_clustered_channels(self):
        # independent, but each clustered: random shifts keep each
        # channel's clusters, random labels pool them
        a, b = self._clustered(1), self._clustered(2)
        kwargs = dict(partner_locs=b, steps=25, n_simulations=19)
        shift = postprocess.ripley_h(a, self.INFO, **kwargs)
        inside = (shift["h_mean"] >= shift["envelope_low"]) & (
            shift["h_mean"] <= shift["envelope_high"]
        )
        assert inside.mean() > 0.9
        labels = postprocess.ripley_h(
            a, self.INFO, cross_null="labels", **kwargs
        )
        assert (labels["h_mean"] < labels["envelope_low"]).all()

    @pytest.mark.parametrize(
        "pick, shape, size",
        [
            (((2000, 3000), (8000, 7000)), "Rectangle", 3000),
            (((1000, 2000), (6000, 9000)), "Box", None),
            ((5000, 5000), "Square", 4000),
            ((5000, 5000), "Circle", 4000),
            (
                [
                    (1000, 1000),
                    (6000, 1000),
                    (6000, 4000),
                    (1000, 4000),
                    (1000, 1000),
                ],
                "Polygon",
                None,
            ),
        ],
    )
    def test_random_shift_stays_in_the_region(self, pick, shape, size):
        region = postprocess._pick_region(pick, shape, size)
        assert region.rectangle is not None or shape == "Circle"
        rng = np.random.default_rng(0)
        x, y = postprocess._uniform_in_region(500, region, rng)
        z = rng.uniform(-100, 100, 500)
        x2, y2, z2 = postprocess._random_shift(
            x, y, z, region, (-100, 100), rng
        )
        geometry = (region.kind, region.circle, region.segments)
        assert postprocess._in_region_many(x2, y2, *geometry).mean() > 0.99
        assert z2.min() >= -100 and z2.max() <= 100
        assert not np.allclose(x2, x)

    def test_random_shift_needs_rectangles_or_circles(self):
        a, b = self._random(2000, 0), self._random(2000, 1)
        picks = [[(1000, 1000), (9000, 1500), (6000, 5000), (1000, 1000)]]
        with pytest.raises(ValueError, match="cross_null='labels'"):
            postprocess.ripley_h(
                a, self.INFO, picks=picks, pick_shape="Polygon", partner_locs=b
            )
        result = postprocess.ripley_h(
            a,
            self.INFO,
            picks=picks,
            pick_shape="Polygon",
            partner_locs=b,
            cross_null="labels",
            n_simulations=2,
        )
        assert result["envelope_low"] is not None

    def test_cross_in_3d(self):
        a = self._random(8000, 0, z=True)
        b = self._random(8000, 1, z=True)
        result = postprocess.rdf(
            a,
            self.INFO,
            partner_locs=b,
            use_z=True,
            r_max=300,
            steps=15,
            n_simulations=3,
        )
        assert np.abs(result["g_mean"][2:] - 1).max() < 0.1
        z = np.concatenate([a["z"], b["z"]])
        np.testing.assert_allclose(result["z_ranges"], [[z.min(), z.max()]])
        with pytest.raises(ValueError, match="requires z"):
            postprocess.rdf(
                a, self.INFO, partner_locs=b.drop(columns="z"), use_z=True
            )

    def test_regions_need_both_channels(self):
        a, b = self._random(2000, 0), self._random(2000, 1)
        b = b[b["x"] > 5000]  # the right half only
        result = postprocess.ripley_h(
            a,
            self.INFO,
            picks=[(2500, 5000), (7500, 5000)],
            pick_shape="Circle",
            pick_size=3000,
            partner_locs=b,
            n_simulations=0,
        )
        assert np.isnan(result["h"][0]).all()
        assert not np.isnan(result["h"][1]).any()
        assert result["n_locs_partner"][0] == 0

    def test_partner_without_molecules_warns(self):
        a, b = self._random(500, 0), self._random(500, 1)
        with pytest.warns(UserWarning, match="partner are not molecules"):
            postprocess.ripley_h(
                a,
                self.INFO,
                partner_locs=b.drop(columns="n_events"),
                steps=5,
                n_simulations=0,
            )

    def test_plot_and_tables(self):
        a, b = self._random(2000, 0), self._random(2000, 1)
        result = postprocess.rdf(
            a,
            self.INFO,
            partner_locs=b,
            cross_null="labels",
            n_simulations=2,
        )
        ax = postprocess.plot_rdf(result).axes[0]
        assert ax.get_title() == "Cross radial distribution function"
        labels = ax.get_legend_handles_labels()[1]
        assert "Random labeling envelope (2 simulations)" in labels
        _, regions = postprocess._rdf_tables(result)
        assert {"n_locs_partner", "partner_density_per_um2"} <= set(
            regions.columns
        )
        h = postprocess.ripley_h(a, self.INFO, partner_locs=b, n_simulations=2)
        ax = postprocess.plot_ripley_h(h).axes[0]
        assert ax.get_title() == "Ripley's cross H function"
        assert "Independence envelope (2 random shifts)" in (
            ax.get_legend_handles_labels()[1]
        )

    def test_invalid_cross_null(self):
        a, b = self._random(500, 0), self._random(500, 1)
        with pytest.raises(ValueError, match="cross_null"):
            postprocess.ripley_h(
                a, self.INFO, partner_locs=b, cross_null="toroidal"
            )


class TestNndDistributions:
    INFO = TestRipleyH.INFO
    SIZE = 10_000.0

    def _random(self, n, seed, z=False):
        rng = np.random.default_rng(seed)
        locs = _ripley_locs(*rng.uniform(0, self.SIZE, size=(2, n)))
        if z:
            locs["z"] = rng.uniform(-400, 400, n)
        return locs

    def test_random_channel_matches_the_csr_distribution(self):
        locs = self._random(10_000, 0)
        result = postprocess.nnd_distributions(
            locs, self.INFO, n_simulations=5
        )
        assert result["histograms"].shape == (1, 1, 40)
        assert result["names"] == ["0"] and result["cross_null"] is None
        # P(r_1 < NND <= r_2) = exp(-pi lambda r_1^2) - exp(-pi lambda r_2^2)
        density = 10_000 / self.SIZE**2
        edges = result["edges"]
        expected = np.exp(-np.pi * density * edges[:-1] ** 2) - np.exp(
            -np.pi * density * edges[1:] ** 2
        )
        np.testing.assert_allclose(
            result["histograms"][0, 0], expected, atol=0.01
        )
        np.testing.assert_allclose(
            result["simulated_mean"][0, 0], expected, atol=0.005
        )
        assert result["medians"][0, 0] == pytest.approx(
            np.sqrt(np.log(2) / (np.pi * density)), rel=0.05
        )

    @pytest.mark.parametrize("cross_null", ["shift", "labels"])
    def test_independent_channels_match_the_simulations(self, cross_null):
        a, b = self._random(5000, 0), self._random(3000, 1)
        result = postprocess.nnd_distributions(
            [a, b],
            self.INFO,
            names=["A", "B"],
            n_simulations=19,
            cross_null=cross_null,
        )
        assert result["histograms"].shape == (2, 2, 40)
        np.testing.assert_array_equal(
            result["n_molecules"], [[5000, 5000], [3000, 3000]]
        )
        np.testing.assert_allclose(
            result["medians"], result["simulated_medians"], rtol=0.05
        )
        inside = (result["histograms"] >= result["envelope_low"]) & (
            result["histograms"] <= result["envelope_high"]
        )
        assert inside.mean() > 0.8

    def test_colocalized_channels_have_short_cross_nnds(self):
        rng = np.random.default_rng(2)
        a = self._random(3000, 0)
        b = a.copy()
        b[["x", "y"]] += rng.normal(0, 10.0, size=(len(b), 2))  # nm
        result = postprocess.nnd_distributions(
            [a, b], self.INFO, n_simulations=3
        )
        assert result["medians"][0, 1] < 20
        assert result["simulated_medians"][0, 1] > 50
        assert result["histograms"][0, 1, :3].sum() > 0.5

    def test_nnds_are_within_regions(self):
        a, b = self._random(4000, 0), self._random(4000, 1)
        b = b[b["x"] > 5000]  # the right half only
        picks = [(2500, 5000), (7500, 5000)]
        result = postprocess.nnd_distributions(
            [a, b],
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=3000,
            n_simulations=0,
        )
        n_a = postprocess.picked_locs(
            a, self.INFO, picks=picks, pick_shape="Circle", pick_size=3000
        )
        # molecules of a in the left pick have no b to point to
        assert result["n_molecules"][0, 1] == len(n_a[1])
        assert result["n_molecules"][0, 0] == len(n_a[0]) + len(n_a[1])
        assert result["simulated_mean"] is None

    @pytest.mark.parametrize("cross_null", ["shift", "labels"])
    def test_simulations_with_a_channel_missing_from_a_region(
        self, cross_null
    ):
        # b has no molecules in the left pick, which is simulated with
        # no points of b instead of failing
        a, b = self._random(4000, 0), self._random(4000, 1)
        b = b[b["x"] > 5000]
        result = postprocess.nnd_distributions(
            [a, b],
            self.INFO,
            picks=[(2500, 5000), (7500, 5000)],
            pick_shape="Circle",
            pick_size=3000,
            n_simulations=3,
            cross_null=cross_null,
        )
        assert np.isfinite(result["simulated_medians"]).all()
        assert np.isfinite(result["envelope_high"]).all()

    def test_3d(self):
        a, b = self._random(8000, 0, z=True), self._random(8000, 1, z=True)
        result = postprocess.nnd_distributions(
            [a, b], self.INFO, use_z=True, n_simulations=5
        )
        np.testing.assert_allclose(
            result["medians"], result["simulated_medians"], rtol=0.05
        )
        flat = postprocess.nnd_distributions(
            [a, b], self.INFO, n_simulations=0
        )
        assert (result["medians"] > flat["medians"]).all()
        with pytest.raises(ValueError, match="requires z"):
            postprocess.nnd_distributions(
                [a, b.drop(columns="z")], self.INFO, use_z=True
            )

    def test_random_shifts_need_rectangles_or_circles(self):
        a, b = self._random(1000, 0), self._random(1000, 1)
        picks = [[(1000, 1000), (9000, 1500), (6000, 5000), (1000, 1000)]]
        with pytest.raises(ValueError, match="cross_null='labels'"):
            postprocess.nnd_distributions(
                [a, b], self.INFO, picks=picks, pick_shape="Polygon"
            )
        # one channel needs no shifts
        postprocess.nnd_distributions(
            a, self.INFO, picks=picks, pick_shape="Polygon"
        )

    def test_plot_and_tables(self):
        a, b = self._random(2000, 0), self._random(2000, 1)
        result = postprocess.nnd_distributions(
            [a, b], self.INFO, names=["A", "B"], n_simulations=2
        )
        fig = postprocess.plot_nnd_distributions(result)
        assert len(fig.axes) == 4
        assert fig.axes[1].get_title() == "A → B"
        curves, summary = postprocess._nnd_tables(result)
        assert {"A_to_B", "A_to_B_simulated_mean"} <= set(curves.columns)
        assert len(curves) == 40
        assert list(summary["from"]) == ["A", "A", "B", "B"]
        assert list(summary["to"]) == ["A", "B", "A", "B"]

    def test_invalid_parameters(self):
        a = self._random(500, 0)
        with pytest.raises(ValueError, match="names"):
            postprocess.nnd_distributions([a, a], self.INFO, names=["A"])
        with pytest.raises(ValueError, match="cross_null"):
            postprocess.nnd_distributions(
                [a, a], self.INFO, cross_null="toroidal"
            )
        with pytest.raises(ValueError, match="r_max"):
            postprocess.nnd_distributions(a, self.INFO, r_max=0)


class TestKnnCsrDistributions:
    @pytest.mark.parametrize("dim, density", [(2, 1e-4), (3, 1e-6)])
    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_pdf_is_the_derivative_of_the_cdf(self, k, dim, density):
        r = np.linspace(0.0, 400.0, 40_001)
        pdf = postprocess.knn_csr_pdf(r, density, k=k, dim=dim)
        cdf = postprocess.knn_csr_cdf(r, density, k=k, dim=dim)
        integral = np.concatenate(
            [[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(r))]
        )
        np.testing.assert_allclose(integral, cdf, atol=1e-6)
        assert cdf[-1] == pytest.approx(1.0, abs=1e-6)

    def test_nearest_neighbor(self):
        r = np.array([10.0, 50.0, 100.0])
        density = 1e-4  # 100 per um^2
        np.testing.assert_allclose(
            postprocess.knn_csr_cdf(r, density),
            1 - np.exp(-density * np.pi * r**2),
        )
        np.testing.assert_allclose(
            postprocess.knn_csr_pdf(r, density),
            2 * np.pi * density * r * np.exp(-density * np.pi * r**2),
        )
        np.testing.assert_allclose(
            postprocess.knn_csr_cdf(r, 1e-6, dim=3),
            1 - np.exp(-1e-6 * 4 / 3 * np.pi * r**3),
        )

    def test_scalars_and_negative_distances(self):
        assert isinstance(postprocess.knn_csr_pdf(20.0, 1e-4), float)
        assert postprocess.knn_csr_pdf(-1.0, 1e-4) == 0.0
        assert postprocess.knn_csr_cdf(-1.0, 1e-4) == 0.0
        # no overflow for distant neighbors
        assert postprocess.knn_csr_pdf(1e6, 1e-4, k=3) == 0.0

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="density"):
            postprocess.knn_csr_pdf(1.0, 0.0)
        with pytest.raises(ValueError, match="k must"):
            postprocess.knn_csr_cdf(1.0, 1e-4, k=0)
        with pytest.raises(ValueError, match="k must"):
            postprocess.knn_csr_cdf(1.0, 1e-4, k=1.5)
        with pytest.raises(ValueError, match="dim"):
            postprocess.knn_csr_pdf(1.0, 1e-4, dim=1)


class TestKnnDensity:
    INFO = TestRipleyH.INFO
    SIZE = 10_000.0

    def _random(self, n, seed, z=False):
        rng = np.random.default_rng(seed)
        locs = _ripley_locs(*rng.uniform(0, self.SIZE, size=(2, n)))
        if z:
            locs["z"] = rng.uniform(-400, 400, n)
        return locs

    def test_random_molecules_give_the_count_density(self):
        locs = self._random(10_000, 0)
        result = postprocess.knn_density(locs, self.INFO, n_simulations=19)
        count_density = 10_000 / self.SIZE**2
        assert result["count_density"] == pytest.approx(count_density)
        np.testing.assert_allclose(
            result["densities"], count_density, rtol=0.03
        )
        assert result["density"] == pytest.approx(count_density, rel=0.02)
        np.testing.assert_array_equal(result["k"], [1, 2, 3, 4, 5])
        assert result["n_molecules"] == 10_000
        # the distances of each order are exact or censored
        np.testing.assert_array_equal(
            result["n_distances"] + result["n_censored"], 10_000
        )
        assert (result["n_censored"] > 0).all()
        # standard errors of about 1 / sqrt(k n)
        relative_se = result["densities_se"] / result["densities"]
        assert ((relative_se > 0.002) & (relative_se < 0.02)).all()
        assert np.median(result["p_values"]) > 0.2
        # the histograms follow the fitted distributions
        edges = result["edges"]
        for i, k in enumerate(result["k"]):
            cumulative = np.cumsum(result["histograms"][i] * np.diff(edges))
            expected = postprocess.knn_csr_cdf(
                edges[1:], result["densities"][i], k=k
            )
            np.testing.assert_allclose(cumulative, expected, atol=0.01)

    def test_edge_correction_removes_the_bias_in_small_picks(self):
        locs = self._random(20_000, 1)
        centers = np.linspace(1000, 9000, 5)
        picks = [(x, y) for x in centers for y in centers]
        kwargs = dict(
            picks=picks, pick_shape="Circle", pick_size=1500, n_simulations=0
        )
        corrected = postprocess.knn_density(locs, self.INFO, **kwargs)
        uncorrected = postprocess.knn_density(
            locs, self.INFO, edge_correction=False, **kwargs
        )
        count_density = corrected["count_density"]
        np.testing.assert_allclose(
            corrected["densities"], count_density, rtol=0.05
        )
        # neighbors outside the picks are missed, more so for larger k
        assert uncorrected["densities"][0] < 0.96 * count_density
        assert (np.diff(uncorrected["densities"]) < 0).all()
        assert not uncorrected["n_censored"].any()
        assert not uncorrected["edge_correction"]
        n_picked = sum(
            len(pick)
            for pick in postprocess.picked_locs(
                locs,
                self.INFO,
                picks=picks,
                pick_shape="Circle",
                pick_size=1500,
            )
        )
        assert corrected["n_molecules"] == n_picked
        assert corrected["n_regions"] == 25

    def test_clustered_molecules(self):
        rng = np.random.default_rng(2)
        centers = rng.uniform(500, self.SIZE - 500, size=(1000, 2))
        xy = np.repeat(centers, 5, axis=0) + rng.normal(0, 20, (5000, 2))
        result = postprocess.knn_density(
            _ripley_locs(*xy.T), self.INFO, n_simulations=19
        )
        densities = result["densities"]
        assert densities[0] > 5 * result["count_density"]
        # the fifth neighbor lies in another cluster
        assert densities[4] < 0.2 * densities[3]
        np.testing.assert_allclose(result["p_values"], 1 / 20)

    def test_r_min_leaves_out_short_distances(self):
        locs = self._random(10_000, 3)
        result = postprocess.knn_density(
            locs, self.INFO, r_min=30, n_simulations=0
        )
        np.testing.assert_allclose(
            result["densities"], 10_000 / self.SIZE**2, rtol=0.03
        )
        # about 25 % of the nearest neighbors are closer than 30 nm
        assert 0.65 < result["n_distances"][0] / 10_000 < 0.8
        assert result["r_min"] == 30
        assert (result["histograms"][:, result["edges"][1:] <= 30] == 0).all()

    def test_pooled_orders(self):
        rng = np.random.default_rng(4)
        centers = rng.uniform(500, self.SIZE - 500, size=(2000, 2))
        xy = np.repeat(centers, 2, axis=0) + rng.normal(0, 10, (4000, 2))
        locs = _ripley_locs(*xy.T)
        all_orders = postprocess.knn_density(locs, self.INFO, n_simulations=0)
        without_pairs = postprocess.knn_density(
            locs, self.INFO, k_min=2, n_simulations=0
        )
        # the pooled fit lies between the fits of its orders
        assert (
            without_pairs["densities"][1:].min()
            <= without_pairs["density"]
            <= without_pairs["densities"][1:].max()
        )
        assert without_pairs["density"] < all_orders["density"]
        assert without_pairs["k_min"] == 2

    def test_3d(self):
        locs = self._random(10_000, 5, z=True)
        result = postprocess.knn_density(
            locs, self.INFO, use_z=True, n_simulations=0
        )
        count_density = 10_000 / (self.SIZE**2 * 800)
        assert result["count_density"] == pytest.approx(
            count_density, rel=0.01
        )
        np.testing.assert_allclose(
            result["densities"], count_density, rtol=0.04
        )
        uncorrected = postprocess.knn_density(
            locs,
            self.INFO,
            use_z=True,
            edge_correction=False,
            n_simulations=0,
        )
        assert (uncorrected["densities"] < 0.95 * count_density).all()
        with pytest.raises(ValueError, match="requires z"):
            postprocess.knn_density(
                locs.drop(columns="z"), self.INFO, use_z=True
            )

    def test_simulations(self):
        locs = self._random(2000, 6)
        first = postprocess.knn_density(locs, self.INFO, n_simulations=9)
        second = postprocess.knn_density(locs, self.INFO, n_simulations=9)
        np.testing.assert_array_equal(first["p_values"], second["p_values"])
        assert first["n_simulations"] == 9
        assert ((first["p_values"] >= 0.1) & (first["p_values"] <= 1)).all()
        none = postprocess.knn_density(locs, self.INFO, n_simulations=0)
        assert none["p_values"] is None
        assert none["densities_se"] is None and none["density_se"] is None
        np.testing.assert_array_equal(none["densities"], first["densities"])

    def test_regions_without_neighbors(self):
        locs = _ripley_locs([1000.0, 1050.0, 5000.0], [1000.0, 1000.0, 5000.0])
        picks = [(1000, 1000), (5000, 5000), (8000, 8000)]
        result = postprocess.knn_density(
            locs,
            self.INFO,
            picks=picks,
            pick_shape="Circle",
            pick_size=1000,
            k_max=2,
            n_simulations=0,
        )
        assert result["n_molecules"] == 3
        assert result["n_distances"][0] == 2
        assert np.isnan(result["densities"][1])  # no second neighbors
        with pytest.raises(ValueError, match="No region contains"):
            postprocess.knn_density(
                locs,
                self.INFO,
                picks=[(8000, 8000)],
                pick_shape="Circle",
                pick_size=1000,
            )
        with pytest.raises(ValueError, match="r_min"):
            postprocess.knn_density(
                locs,
                self.INFO,
                picks=picks,
                pick_shape="Circle",
                pick_size=1000,
                r_min=100,
            )

    def test_warns_for_localizations(self):
        locs = self._random(500, 7).drop(columns="n_events")
        with pytest.warns(UserWarning, match="not molecules"):
            postprocess.knn_density(locs, self.INFO, n_simulations=0)

    def test_plot_and_tables(self):
        result = postprocess.knn_density(
            self._random(2000, 8), self.INFO, r_min=10, n_simulations=4
        )
        fig = postprocess.plot_knn_density(result)
        assert len(fig.axes) == 2
        assert "4 simulations" in fig.get_suptitle()
        curves, summary = postprocess._knn_density_tables(result)
        assert len(curves) == 50
        assert {"k1_per_nm", "k1_csr_per_nm", "k5_csr_per_nm"} <= set(
            curves.columns
        )
        assert list(summary["estimate"]) == [
            "k=1",
            "k=2",
            "k=3",
            "k=4",
            "k=5",
            "pooled k=1-5",
            "molecules/area",
        ]
        np.testing.assert_allclose(
            summary["density_per_um2"][:5], result["densities"] * 1e6
        )
        assert summary["n_distances"].dtype == "Int64"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(k_max=0), "k_max"),
            (dict(k_min=3, k_max=2), "k_min"),
            (dict(r_min=-1.0), "r_min"),
            (dict(r_min=50.0, r_max=40.0), "r_max"),
            (dict(steps=0), "steps"),
            (dict(n_simulations=-1), "n_simulations"),
            (dict(use_z=True, z_range=(1.0, 0.0)), "z_range"),
        ],
    )
    def test_invalid_parameters(self, kwargs, match):
        locs = self._random(100, 9, z=True)
        with pytest.raises(ValueError, match=match):
            postprocess.knn_density(locs, self.INFO, **kwargs)


def _filaments(precision: float, seed: int = 0, size: float = 5000.0):
    """Localizations (nm) along 40 random straight filaments in a
    ``size`` x ``size`` field, 0.6 molecules per nm, each localized 8
    times with ``precision``."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(40):
        x0, y0 = rng.uniform(0, size, 2)
        angle = rng.uniform(0, np.pi)
        t = rng.uniform(-0.75 * size, 0.75 * size, int(0.9 * size))
        x = np.repeat(x0 + t * np.cos(angle), 8)
        y = np.repeat(y0 + t * np.sin(angle), 8)
        xs.append(x + rng.normal(0, precision, len(x)))
        ys.append(y + rng.normal(0, precision, len(y)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    inside = (x > 0) & (x < size) & (y > 0) & (y < size)
    n = int(inside.sum())
    locs = pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, n).astype(np.uint32),
            "x": x[inside].astype(np.float32),
            "y": y[inside].astype(np.float32),
            "lpx": np.full(n, precision, np.float32),
            "lpy": np.full(n, precision, np.float32),
        }
    )
    info = [{"Width": 50, "Height": 50, "Pixelsize": 100.0, "Frames": 1000}]
    return locs, info


class TestDecorrelation:
    @pytest.fixture(scope="class")
    def sharp(self):
        locs, info = _filaments(4.0)
        return locs, info, postprocess.decorrelation(locs, info)

    def test_resolution_follows_the_precision_and_the_frc(self, sharp):
        locs, info, result = sharp
        blurred_locs, _ = _filaments(12.0)
        blurred = postprocess.decorrelation(blurred_locs, info)
        assert blurred["resolution"] / result["resolution"] == pytest.approx(
            3.0, rel=0.3
        )
        for data, precision, res in (
            (locs, 4.0, result),
            (blurred_locs, 12.0, blurred),
        ):
            frc = postprocess.frc(
                data, info, viewport=((0, 0), (5000, 5000)), lp=precision
            )
            assert res["resolution"] == pytest.approx(
                frc["resolution"], rel=0.15
            )

    def test_smallest_resolution_over_the_pixel_sizes(self, sharp):
        locs, info, result = sharp
        np.testing.assert_allclose(
            result["pixel_sizes"], postprocess.DECORRELATION_PIXEL_SIZES
        )
        assert result["resolution"] == np.nanmin(result["resolutions"])
        best = list(result["pixel_sizes"]).index(result["pixel_size"])
        assert result["resolutions"][best] == result["resolution"]
        assert len(result["results"]) == len(result["pixel_sizes"])
        # large pixels: about twice the pixel size (sampling limited)
        assert result["resolutions"][-1] == pytest.approx(
            2 * result["pixel_sizes"][-1], rel=0.3
        )
        assert result["n_locs"] == len(locs)
        assert not len(result["skipped_pixel_sizes"])
        # smaller pixels than the best were rendered
        assert not result["pixel_size_limited"]

    def test_too_large_pixels_limit_the_resolution(self, sharp):
        locs, info, _ = sharp
        result = postprocess.decorrelation(locs, info, pixel_size=16.0)
        assert result["resolution"] == pytest.approx(32.0, rel=0.3)
        assert result["pixel_size_limited"]
        fig = postprocess.plot_decorrelation(result)
        assert "limited by the pixel size" in fig.get_suptitle()

    def test_one_pixel_size(self, sharp):
        locs, info, _ = sharp
        result = postprocess.decorrelation(
            locs,
            info,
            viewport=((0, 0), (2500, 2500)),
            pixel_size=4.0,
            n_radii=30,
            n_filters=5,
        )
        np.testing.assert_array_equal(result["pixel_sizes"], [4.0])
        assert result["results"][0]["curves"].shape == (6, 30)
        assert result["pixel_size"] == 4.0
        assert result["n_locs"] < len(locs)

    def test_pixel_sizes_outside_the_image_limits_are_skipped(self, sharp):
        locs, info, _ = sharp
        result = postprocess.decorrelation(
            locs, info, pixel_size=[2.0, 8.0, 200.0], max_image_size=1024
        )
        # 5000 nm / 2 nm > 1024 pixels; 5000 nm / 200 nm < 32 pixels
        np.testing.assert_array_equal(result["pixel_sizes"], [8.0])
        np.testing.assert_array_equal(
            result["skipped_pixel_sizes"], [2.0, 200.0]
        )
        with pytest.raises(ValueError, match="No pixel size"):
            postprocess.decorrelation(locs, info, pixel_size=200.0)
        with pytest.raises(ValueError, match="positive"):
            postprocess.decorrelation(locs, info, pixel_size=[4.0, -1.0])
        with pytest.raises(ValueError, match="empty"):
            postprocess.decorrelation(locs, info, viewport=((0, 0), (0, 1000)))

    def test_bilinear_rendering(self):
        x = np.array([10.0, 13.0])
        y = np.array([10.0, 10.0])
        image = postprocess._render_bilinear(x, y, 0.0, 0.0, 4.0, (8, 8))
        assert image.sum() == pytest.approx(2.0)
        # at a pixel center (10 nm = center of pixel 2): one pixel
        assert image[2, 2] == pytest.approx(1.0 + 0.25)
        # 13 nm lies 3/4 of the way from pixel 2 to pixel 3
        assert image[2, 3] == pytest.approx(0.75)

    def test_sample_area_and_density(self):
        # a dense 1 x 1 um square of localizations
        grid = np.arange(0.5, 1000.0, 1.0)
        x, y = (v.ravel() for v in np.meshgrid(grid, grid))
        assert postprocess._sample_area(x, y) == pytest.approx(1e6, rel=0.03)
        locs = pd.DataFrame(
            {"frame": np.zeros(len(x), np.uint32), "x": x, "y": y}
        )
        info = [{"Width": 10, "Height": 10, "Pixelsize": 100.0}]
        result = postprocess.decorrelation(locs, info, pixel_size=16.0)
        assert result["density"] == pytest.approx(1e6, rel=0.03)

    def test_no_resolution_without_structure(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 5000, (2, 2000))
        locs = pd.DataFrame(
            {"frame": np.zeros(2000, np.uint32), "x": x, "y": y}
        )
        info = [{"Width": 50, "Height": 50, "Pixelsize": 100.0}]
        result = postprocess.decorrelation(locs, info, pixel_size=[16.0])
        assert np.isnan(result["resolution"])
        assert np.isnan(result["pixel_size"])
        fig = postprocess.plot_decorrelation(result)
        assert "n/a" in fig.get_suptitle()

    def test_plot_and_tables(self, sharp):
        _, _, result = sharp
        fig = postprocess.plot_decorrelation(result)
        assert len(fig.axes) == 2  # curves and resolution by pixel size
        assert f"{result['resolution']:.1f} nm" in fig.get_suptitle()
        sizes, curves = postprocess._decorrelation_tables(result)
        assert list(sizes.columns) == [
            "pixel_size_nm",
            "resolution_nm",
            "cutoff",
            "amplitude",
        ]
        assert len(sizes) == len(result["pixel_sizes"])
        assert curves.shape == (50, 12)  # frequency, unfiltered, 10 filters
        # an image result plots on one panel
        image_result = result["results"][0]
        assert len(postprocess.plot_decorrelation(image_result).axes) == 1
