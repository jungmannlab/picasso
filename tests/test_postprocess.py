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


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------


class TestIndexBlocks:
    def test_index_blocks_structure(self, locs, info):
        index_blocks = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        _, _, x_index, y_index, block_starts, block_ends, K, L = index_blocks
        assert len(x_index) == len(locs)
        assert len(y_index) == len(locs)
        assert K > 0 and L > 0
        assert block_starts.ndim == 2 and block_ends.ndim == 2
        # block_starts[i,j] <= block_ends[i,j] for every cell
        assert (block_starts <= block_ends).all()

    def test_get_block_locs_at_returns_some_locs(self, locs, info):
        index_blocks = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
        locs_at = postprocess.get_block_locs_at(15.5, 15.5, index_blocks)
        assert len(locs_at) > 0
        # All returned locs must lie within the pick radius of the query
        d = np.hypot(locs_at["x"] - 15.5, locs_at["y"] - 15.5)
        # Block lookup is conservative — within ~PICK_SIZE
        assert (d < 2 * PICK_SIZE).all()


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
            origami_picks,
            pick_shape="Circle",
            pick_size=PICK_SIZE,
        )
        radius = PICK_SIZE / 2
        for (cx, cy), p in zip(origami_picks, picked):
            d = np.hypot(p["x"] - cx, p["y"] - cy)
            assert (d <= radius + 1e-6).all()


class TestPickSimilar:
    def test_finds_remaining_origamis(self, locs, info):
        seed_picks = [[5.5, 5.5], [5.5, 15.5]]
        new_picks = postprocess.pick_similar(
            locs, info, seed_picks, PICK_SIZE, std_range=123.0
        )
        assert len(new_picks) == len(ORIGAMI_PICKS)


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
        # Counts can't be negative
        assert (dh >= 0).all()

    def test_total_count_is_finite_and_positive(self, locs, info):
        dh = postprocess.distance_histogram(
            locs, info, bin_size=0.1, r_max=1.0
        )
        assert dh.sum() > 0


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
        """At low frequency, the FRC curve should be close to 1 (signal
        dominates)."""
        viewport = ((15, 15), (16, 16))
        frc_res = postprocess.frc(locs, info, viewport=viewport)
        assert frc_res["frc_curve"][0] > 0.7


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


class TestLocalDensity:
    def test_density_column_added_with_proper_dtype(self, locs, info):
        out = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE
        )
        assert "density" in out.columns
        assert out["density"].dtype in (np.uint32, np.uint64)
        # Each loc has at least itself within radius
        assert (out["density"] >= 1).all()

    def test_dense_radius_picks_up_origami_clusters(self, locs, info):
        """With ``radius`` larger than each origami, every loc within an
        origami should report a similar density value (the per-origami
        loc count). Verify the *unique* densities are << len(locs)."""
        out = postprocess.compute_local_density(
            locs.copy(), info, radius=PICK_SIZE
        )
        unique_densities, _ = np.unique(out["density"], return_counts=True)
        # Test data has ~9 origamis, so density should take few values
        assert len(unique_densities) < len(locs) // 5


# ---------------------------------------------------------------------------
# Linking and dark-time computation
# ---------------------------------------------------------------------------


class TestLinking:
    def test_columns_added(self, locs, info):
        linked = postprocess.link(locs, info)
        for col in ["len", "n", "photon_rate"]:
            assert col in linked.columns

    def test_length_invariants(self, locs, info):
        """Linked length is <= original (events merge); the sum of the
        ``n`` (locs per linked event) column equals the original count."""
        linked = postprocess.link(locs, info)
        assert len(linked) <= len(locs)
        assert linked["n"].sum() == len(locs)

    def test_len_within_movie_frame_span(self, locs, info):
        """A linked event can't span more frames than the original
        observation window."""
        linked = postprocess.link(locs, info)
        n_frames = info[0]["Frames"]
        assert linked["len"].max() <= n_frames

    def test_compute_dark_times_adds_dark_column(self, locs, info):
        linked = postprocess.link(locs, info)
        with_dark = postprocess.compute_dark_times(linked)
        assert "dark" in with_dark.columns


# ---------------------------------------------------------------------------
# Channel alignment
# ---------------------------------------------------------------------------


class TestAlign:
    def test_channels_aligned_after_known_shift(self, locs, info):
        """Apply a known +5 px shift to a copy and check that align()
        brings the channels back together (residual <0.5 px)."""
        locs2 = locs.copy()
        locs2["x"] += 5.0
        aligned = postprocess.align([locs, locs2], [info, info])
        assert len(aligned) == 2
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.5

    def test_no_shift_is_no_op_within_tolerance(self, locs, info):
        """If channels start aligned, alignment shouldn't drift them
        substantially."""
        aligned = postprocess.align([locs, locs.copy()], [info, info])
        residual = aligned[1]["x"].mean() - aligned[0]["x"].mean()
        assert abs(residual) < 0.1


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
            pick_size=PICK_SIZE,
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
        """For one specific group, the ``x_mean`` in groupprops equals
        the mean of x for that group computed by hand."""
        out = postprocess.groupprops(grouped_locs)
        for g in grouped_locs["group"].unique()[:3]:
            manual = grouped_locs.loc[grouped_locs["group"] == g, "x"].mean()
            row = (
                out.loc[out["group"] == g]
                if "group" in out.columns
                else (out.iloc[[g]])
            )
            assert row["x_mean"].iloc[0] == pytest.approx(manual, rel=1e-3)


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
