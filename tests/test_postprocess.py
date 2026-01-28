"""Test picasso.postprocess functions.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pandas as pd
import pytest
from picasso import io, lib, postprocess

# parameters
PICK_SIZE = 1.5  # in camera pixels


@pytest.fixture(scope="module")
def locs_data():
    """Load localization data once per test module."""
    locs_data = io.load_locs("./tests/data/testdata_locs.hdf5")
    return locs_data


@pytest.fixture(scope="module")
def locs(locs_data):
    """Get locs for testing clusterers."""
    locs = locs_data[0]
    return locs


@pytest.fixture(scope="module")
def info(locs_data):
    """Get info for testing clusterers."""
    info = locs_data[1]
    return info


def test_index_blocks(locs, info):
    index_blocks = postprocess.get_index_blocks(locs, info, PICK_SIZE / 2)
    _, _, x_index, y_index, block_starts, block_ends, K, L = index_blocks
    assert len(x_index) == len(locs), "x_index length mismatch"
    assert len(y_index) == len(locs), "y_index length mismatch"
    assert K > 0, "K should be positive"
    assert L > 0, "L should be positive"
    assert block_starts.ndim == 2, "block_starts should be 2D"
    assert block_ends.ndim == 2, "block_ends should be 2D"

    # get locs at index blocks
    locs_at = postprocess.get_block_locs_at(15.5, 15.5, index_blocks)
    assert len(locs_at) > 0, "No localizations found at specified block center"


def test_picked_locs(locs, info):
    picks = [
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
    picked_locs = postprocess.picked_locs(
        locs, info, picks, pick_shape="Circle", pick_size=PICK_SIZE
    )
    assert len(picked_locs) == len(picks), "Number of picked locs mismatch"


def test_pick_similar(locs, info):
    picks = [
        [5.5, 5.5],
        [5.5, 15.5],
    ]
    new_picks = postprocess.pick_similar(
        locs, info, picks, PICK_SIZE, std_range=123.0
    )
    assert len(new_picks) == 9, "Number of similar picks mismatch"


def test_distance_histogram(locs, info):
    dh = postprocess.distance_histogram(locs, info, bin_size=0.1, r_max=1.0)
    assert dh.shape[0] > 0, "Distance histogram should have positive length"


def test_nena(locs, info):
    res, nena = postprocess.nena(locs, info)
    assert nena > 0, "NeNA should be positive"
    assert all(
        [_ in res.keys() for _ in ["d", "data", "best_fit", "best_values"]]
    ), "NeNA result keys mismatch"
    assert all(
        [
            _ in res["best_values"].keys()
            for _ in ["delta_a", "s", "ac", "dc", "sc"]
        ]
    ), "NeNA best_values keys mismatch"


def test_frc(locs, info):
    viewport = ((15, 15), (16, 16))
    frc_res = postprocess.frc(locs, info, viewport=viewport)
    assert all(
        [
            _ in frc_res.keys()
            for _ in [
                "frequencies",
                "frc_curve",
                "resolution",
                "images",
                "frc_curve_smooth",
            ]
        ]
    ), "FRC result keys mismatch"
    assert frc_res["resolution"] > 0, "FRC resolution should be positive"


def test_pair_correlation(locs, info):
    bin_size = 0.1
    r_max = 1.0
    bins_lower, pc = postprocess.pair_correlation(
        locs, info, bin_size=bin_size, r_max=r_max
    )
    assert bins_lower.shape[0] > 0, "Bins lower should have positive length"
    assert pc.shape[0] > 0, "Pair correlation should have positive length"


def test_local_density(locs, info):
    locs_den = locs.copy()
    locs_den = postprocess.compute_local_density(
        locs_den, info, radius=PICK_SIZE
    )
    # pick size is larger than the origamis, so each localization should have
    # the number of localizations in their origami saved as density
    assert "density" in locs_den.columns, "Density column missing in locs"
    locs_per_origami, freq = np.unique(locs_den["density"], return_counts=True)
    assert len(locs_per_origami) == len(
        freq
    ), "Unique densities count mismatch"
    assert locs_den["density"].dtype in [
        np.uint32,
        np.uint64,
    ], "Density dtype mismatch"


def test_linking(locs, info):
    """Test linking and the associated function (compute_dark_times)."""
    linked_locs = postprocess.link(locs, info)
    assert "len" in linked_locs.columns, "'len' column missing in linked locs"
    assert "n" in linked_locs.columns, "'n' column missing in linked locs"

    locs_ = postprocess.compute_dark_times(linked_locs)
    assert (
        "dark" in locs_.columns
    ), "'dark' column missing in locs after compute_dark_times"


def test_align(locs, info):
    # create a dummy shifted channel
    locs2 = locs.copy()
    locs2["x"] += 5.0
    channels = [locs, locs2]
    infos = [info, info]
    aligned_channels = postprocess.align(channels, infos)
    assert len(aligned_channels) == 2, "Number of aligned channels mismatch"
    shift_x = aligned_channels[1]["x"].mean() - aligned_channels[0]["x"].mean()
    assert shift_x < 0.5, "Channels not properly aligned"


def test_groupprops(locs, info):
    # pick each origami
    picks = [
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
    picked_locs = postprocess.picked_locs(
        locs, info, picks, pick_shape="Circle", pick_size=PICK_SIZE
    )
    picked_locs = pd.concat(picked_locs, ignore_index=True)
    picked_locs = postprocess.link(picked_locs, info)
    picked_locs = postprocess.compute_dark_times(picked_locs)
    old_columns = picked_locs.columns.tolist()
    groupprops = postprocess.groupprops(picked_locs)
    expected_columns = [column + "_mean" for column in old_columns]
    expected_columns += [column + "_std" for column in old_columns]
    expected_columns += ["n_events", "qpaint_idx"]
    for col in expected_columns:
        assert (
            col in groupprops.columns
        ), f"Missing column in groupprops: {col}"
