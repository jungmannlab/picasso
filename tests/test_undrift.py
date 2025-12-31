"""Test picasso functions related to undrifting.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import pytest
from picasso import io, postprocess, aim, lib

# parameters for undrifting
SEGMENTATION = 100


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


# undrifting tests
def test_aim(locs, info):
    """Test undrifting by AIM."""
    undrifted_locs, _, drift = aim.aim(locs, info, segmentation=SEGMENTATION)
    assert len(drift) == lib.get_from_metadata(
        info, "Frames"
    ), "Drift length does not match number of frames."
    assert undrifted_locs.shape == locs.shape, "Undrifted locs shape mismatch."


def test_rcc(locs, info):
    """Test undrifting by RCC."""
    drift, undrifted_locs = postprocess.undrift(
        locs, info, segmentation=SEGMENTATION, display=False
    )
    assert len(drift) == lib.get_from_metadata(
        info, "Frames"
    ), "Drift length does not match number of frames."
    assert undrifted_locs.shape == locs.shape, "Undrifted locs shape mismatch."


def test_undrift_from_picked(locs, info):
    """Test undrifting from picked locs."""
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
        locs, info, picks, pick_shape="Circle", pick_size=200 / 130
    )
    drift = postprocess.undrift_from_picked(picked_locs, info)
    assert len(drift) == lib.get_from_metadata(
        info, "Frames"
    ), "Drift length does not match number of frames."
