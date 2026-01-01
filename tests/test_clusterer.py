"""Test picasso.clusterer functions.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pytest
from picasso import io, clusterer

# parameters for clustering
CAMERA_PIXEL_SIZE = 130  # nm
DBSCAN_EPS = 5 / CAMERA_PIXEL_SIZE  # in camera pixels, like localizations
DBSCAN_MIN_SAMPLES = 2
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_CLUSTER_EPS = 10 / CAMERA_PIXEL_SIZE
GA_RADIUS = 7 / CAMERA_PIXEL_SIZE
GA_RADIUS_Z = GA_RADIUS * 2.5
GA_MIN_LOCS = 5


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


@pytest.fixture(scope="module")
def db_locs(locs):
    """Create clustered locs for testing cluster areas and centers."""
    db_locs = clusterer.dbscan(
        locs, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, min_locs=0
    )
    return db_locs


def test_dbscan(db_locs):
    """Test dbscan."""
    assert "group" in db_locs.columns, "DBSCAN did not add 'group' column"
    assert -1 not in db_locs["group"].values, "Invalid DBSCAN group ids found"


def test_hdbscan(locs):
    """Test hdbscan."""
    clustered_locs = clusterer.hdbscan(
        locs,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_eps=HDBSCAN_CLUSTER_EPS,
    )
    assert (
        "group" in clustered_locs.columns
    ), "HDBSCAN did not add 'group' column"
    assert (
        -1 not in clustered_locs["group"].values
    ), "Invalid HDBSCAN group ids found"


def test_ga(locs):
    """Test ga (smlm clusterer)."""
    clustered_locs = clusterer.cluster(
        locs,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=True,
    )
    assert "group" in clustered_locs.columns, "GA did not add 'group' column"
    assert (
        -1 not in clustered_locs["group"].values
    ), "Invalid GA group ids found"


def test_ga_3d(locs):
    """Test ga (smlm clusterer). in 3d"""
    locs_3d = locs.copy()
    locs_3d["z"] = np.random.normal(0, GA_RADIUS_Z / 2, size=len(locs_3d))
    clustered_locs = clusterer.cluster(
        locs_3d,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=True,
        radius_z=GA_RADIUS_Z,
        pixelsize=CAMERA_PIXEL_SIZE,
    )
    assert "group" in clustered_locs.columns, "GA did not add 'group' column"
    assert (
        -1 not in clustered_locs["group"].values
    ), "Invalid GA group ids found"


def test_cluster_areas(db_locs, info):
    """Test cluster areas."""
    areas = clusterer.cluster_areas(db_locs, info)
    assert len(areas) > 0, "No cluster areas calculated"
    assert areas["Area (LP^2)"].min() > 0, "Invalid cluster area calculated"


def test_cluster_centers(db_locs):
    """Test cluster centers."""
    centers = clusterer.find_cluster_centers(db_locs)
    assert len(centers) > 0, "No cluster centers calculated"
