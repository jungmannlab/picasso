"""Test picasso.clusterer functions.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pandas as pd
import pytest
from picasso import io, clusterer

from tests.conftest import PIXELSIZE
from tests.test_postprocess_progress import RecordingProgress, assert_phases

# parameters for clustering (nm, like the localizations)
DBSCAN_EPS = 5.0
DBSCAN_MIN_SAMPLES = 2
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_CLUSTER_EPS = 10.0
GA_RADIUS = 7.0
GA_RADIUS_Z = GA_RADIUS * 2.5
GA_MIN_LOCS = 5


# ---------------------------------------------------------------------
# Real-data fixtures (preserved for smoke coverage of the load + cluster path)
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def locs_data():
    """Load localization data once per test module."""
    return io.load_locs("./tests/data/testdata_locs.hdf5")


@pytest.fixture(scope="module")
def locs(locs_data):
    return locs_data[0]


@pytest.fixture(scope="module")
def info(locs_data):
    return locs_data[1]


@pytest.fixture(scope="module")
def db_locs(locs):
    """DBSCAN-clustered real locs, for downstream area/center tests."""
    return clusterer.dbscan(
        locs, radius=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, min_locs=0
    )[0]


# ---------------------------------------------------------------------
# Synthetic ground-truth fixtures
# ---------------------------------------------------------------------

# Blobs are placed in nm, well-separated relative to GA_RADIUS /
# DBSCAN_EPS (5-7 nm). Inter-blob spacing of 40 camera pixels (5.2 um) is
# ~1000x the cluster radius.
BLOB_CENTERS_2D = [
    (10.0 * PIXELSIZE, 10.0 * PIXELSIZE),
    (50.0 * PIXELSIZE, 50.0 * PIXELSIZE),
    (10.0 * PIXELSIZE, 50.0 * PIXELSIZE),
]
BLOB_CENTERS_3D = [
    (10.0 * PIXELSIZE, 10.0 * PIXELSIZE, 0.0),
    (50.0 * PIXELSIZE, 50.0 * PIXELSIZE, 200.0),
    (10.0 * PIXELSIZE, 50.0 * PIXELSIZE, -200.0),
]
# Same relative geometry as BLOB_CENTERS_3D but translated far from z=0
# to guard against any z-offset dependence in the anisotropic scaling.
BLOB_CENTERS_3D_OFFSET = [
    (10.0 * PIXELSIZE, 10.0 * PIXELSIZE, 10_000.0),
    (50.0 * PIXELSIZE, 50.0 * PIXELSIZE, 10_200.0),
    (10.0 * PIXELSIZE, 50.0 * PIXELSIZE, 9_800.0),
]
LOCS_PER_BLOB = 30
SIGMA_XY_NM = 0.65
SIGMA_Z_NM = 1.0
# tolerance for recovered cluster centers (nm)
CENTER_TOL = 0.5 * PIXELSIZE


def _make_synthetic_locs(centers, seed=42):
    """Build a localization DataFrame with planted Gaussian blobs.

    Centers are (x, y) or (x, y, z) in nm.
    """
    rng = np.random.default_rng(seed)
    is_3d = len(centers[0]) == 3
    rows = []
    frame = 0
    for c in centers:
        for _ in range(LOCS_PER_BLOB):
            row = {
                "frame": frame,
                "x": rng.normal(c[0], SIGMA_XY_NM),
                "y": rng.normal(c[1], SIGMA_XY_NM),
                "photons": 1000.0,
                "sx": PIXELSIZE,
                "sy": PIXELSIZE,
                "bg": 10.0,
                "lpx": 1.3,
                "lpy": 1.3,
                "net_gradient": 5000.0,
            }
            if is_3d:
                row["z"] = rng.normal(c[2], SIGMA_Z_NM)
            rows.append(row)
            frame += 1
    return pd.DataFrame(rows)


@pytest.fixture
def synth_locs_2d():
    return _make_synthetic_locs(BLOB_CENTERS_2D)


@pytest.fixture
def synth_locs_3d():
    return _make_synthetic_locs(BLOB_CENTERS_3D)


@pytest.fixture
def synth_locs_3d_offset():
    return _make_synthetic_locs(BLOB_CENTERS_3D_OFFSET)


@pytest.fixture
def synth_info():
    return [{"Pixelsize": PIXELSIZE, "Width": 100, "Height": 100}]


# ---------------------------------------------------------------------
# Clusterer call wrappers — let us parametrize across the three algorithms
# ---------------------------------------------------------------------


def _run_dbscan_2d(locs):
    return clusterer.dbscan(
        locs, radius=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, min_locs=0
    )[0]


def _run_hdbscan_2d(locs):
    return clusterer.hdbscan(
        locs,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_eps=HDBSCAN_CLUSTER_EPS,
    )[0]


def _run_smlm_2d(locs):
    return clusterer.cluster(
        locs,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
    )[0]


def _run_dbscan_3d(locs):
    return clusterer.dbscan(
        locs,
        radius=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        min_locs=0,
        radius_z=DBSCAN_EPS * 2.5,
    )[0]


def _run_hdbscan_3d(locs):
    return clusterer.hdbscan(
        locs,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_eps=HDBSCAN_CLUSTER_EPS,
    )[0]


def _run_smlm_3d(locs):
    return clusterer.cluster(
        locs,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
        radius_z=GA_RADIUS_Z,
    )[0]


# With NumPy >= 2.4, scikit-learn's HDBSCAN fails for
# cluster_selection_epsilon > 0 on this real dataset (scikit-learn issue
# 33219, see picasso/clusterer.py); synthetic data passes, so HDBSCAN's
# correctness is still covered. Strict: once a scikit-learn release fixes
# the bug, this test fails as a reminder to remove the workaround (the
# NumPy pin of the installers and the error in clusterer._hdbscan).
_HDBSCAN_REAL_DATA_XFAIL = pytest.mark.xfail(
    clusterer._HDBSCAN_EPSILON_BUG,
    reason="scikit-learn issue 33219: HDBSCAN epsilon with NumPy >= 2.4",
    raises=RuntimeError,
    strict=True,
)

CLUSTERERS_2D = [
    pytest.param(_run_dbscan_2d, id="dbscan"),
    pytest.param(_run_hdbscan_2d, id="hdbscan"),
    pytest.param(_run_smlm_2d, id="smlm"),
]

CLUSTERERS_2D_REAL_DATA = [
    pytest.param(_run_dbscan_2d, id="dbscan"),
    pytest.param(
        _run_hdbscan_2d, id="hdbscan", marks=_HDBSCAN_REAL_DATA_XFAIL
    ),
    pytest.param(_run_smlm_2d, id="smlm"),
]

CLUSTERERS_3D = [
    pytest.param(_run_dbscan_3d, id="dbscan"),
    pytest.param(_run_hdbscan_3d, id="hdbscan"),
    pytest.param(_run_smlm_3d, id="smlm"),
]


# ---------------------------------------------------------------------
# Smoke tests on real data: each clusterer runs and labels something
# ---------------------------------------------------------------------


@pytest.mark.parametrize("run_clusterer", CLUSTERERS_2D_REAL_DATA)
def test_real_data_smoke(locs, run_clusterer):
    """Each clusterer runs on real data and adds a non-empty 'group' column."""
    out = run_clusterer(locs)
    assert "group" in out.columns
    assert len(out) > 0


# ---------------------------------------------------------------------
# Ground-truth correctness on synthetic blobs
# ---------------------------------------------------------------------


def _match_truth_to_recovered(recovered_centers, truth_centers, tol):
    """Each truth center has a unique recovered center within tol distance."""
    recovered = np.asarray(recovered_centers)
    used = np.zeros(len(recovered), dtype=bool)
    for truth in truth_centers:
        diffs = recovered - np.asarray(truth)
        dists = np.linalg.norm(diffs, axis=1)
        dists[used] = np.inf
        nearest = int(np.argmin(dists))
        assert dists[nearest] < tol, (
            f"No recovered cluster within {tol} of truth {truth}; "
            f"nearest distance was {dists[nearest]:.4f}"
        )
        used[nearest] = True


@pytest.mark.parametrize("run_clusterer", CLUSTERERS_2D)
def test_recovers_known_clusters_2d(synth_locs_2d, run_clusterer):
    """Each clusterer recovers exactly the planted 2D blobs."""
    out = run_clusterer(synth_locs_2d)
    assert out["group"].nunique() == len(BLOB_CENTERS_2D)
    centers = out.groupby("group")[["x", "y"]].mean().to_numpy()
    _match_truth_to_recovered(centers, BLOB_CENTERS_2D, tol=CENTER_TOL)


@pytest.mark.parametrize("run_clusterer", CLUSTERERS_3D)
def test_recovers_known_clusters_3d(synth_locs_3d, run_clusterer):
    """Each clusterer recovers exactly the planted 3D blobs."""
    out = run_clusterer(synth_locs_3d)
    assert out["group"].nunique() == len(BLOB_CENTERS_3D)
    centers = out.groupby("group")[["x", "y", "z"]].mean().to_numpy()
    _match_truth_to_recovered(centers, BLOB_CENTERS_3D, tol=CENTER_TOL)


@pytest.mark.parametrize("run_clusterer", CLUSTERERS_3D)
def test_recovers_known_clusters_3d_offset(
    synth_locs_3d_offset, run_clusterer
):
    """Anisotropic clustering still works when z is far from zero."""
    out = run_clusterer(synth_locs_3d_offset)
    assert out["group"].nunique() == len(BLOB_CENTERS_3D_OFFSET)
    centers = out.groupby("group")[["x", "y", "z"]].mean().to_numpy()
    _match_truth_to_recovered(centers, BLOB_CENTERS_3D_OFFSET, tol=CENTER_TOL)


# ---------------------------------------------------------------------
# Error paths: 3D without required parameters
# ---------------------------------------------------------------------


class _FailingHDBSCAN:
    """Fails like scikit-learn's HDBSCAN with NumPy >= 2.4."""

    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        raise TypeError(
            "only 0-dimensional arrays can be converted to Python scalars"
        )


@pytest.mark.parametrize(
    "bug,cluster_eps,expected",
    [
        (True, 0.1, RuntimeError),  # explained
        (True, 0.0, TypeError),  # another TypeError: unchanged
        (False, 0.1, TypeError),  # NumPy < 2.4: unchanged
    ],
)
def test_hdbscan_epsilon_bug_is_explained(
    monkeypatch, synth_locs_2d, bug, cluster_eps, expected
):
    monkeypatch.setattr(clusterer, "HDBSCAN", _FailingHDBSCAN)
    monkeypatch.setattr(clusterer, "_HDBSCAN_EPSILON_BUG", bug)
    with pytest.raises(expected) as caught:
        clusterer.hdbscan(
            synth_locs_2d,
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            cluster_eps=cluster_eps,
        )
    if expected is RuntimeError:
        message = str(caught.value)
        assert "issues/33219" in message
        assert "epsilon to 0" in message
        assert isinstance(caught.value.__cause__, TypeError)


@pytest.mark.parametrize("run_clusterer", [_run_dbscan_2d, _run_hdbscan_2d])
def test_dbscan_3d_without_radius_z_is_isotropic(synth_locs_3d, run_clusterer):
    """3D localizations need no extra parameter: x, y and z are all in
    nm, so the 2D settings cluster them isotropically."""
    out = run_clusterer(synth_locs_3d)
    assert out["group"].nunique() == len(BLOB_CENTERS_3D)


def test_smlm_3d_requires_radius_z(synth_locs_3d):
    with pytest.raises(ValueError):
        clusterer.cluster(
            synth_locs_3d,
            radius_xy=GA_RADIUS,
            min_locs=GA_MIN_LOCS,
            frame_analysis=False,
        )


# ---------------------------------------------------------------------
# (locs, info) return value
# ---------------------------------------------------------------------


def test_dbscan_returns_info(synth_locs_2d):
    out, info = clusterer.dbscan(
        synth_locs_2d,
        radius=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        min_locs=0,
    )
    assert "group" in out.columns
    assert isinstance(info, list)
    info_dict = info[-1]
    assert isinstance(info_dict, dict)
    assert info_dict["Number of clusters"] == len(BLOB_CENTERS_2D)
    assert "Generated by" in info_dict


def test_hdbscan_returns_info(synth_locs_2d):
    out, info = clusterer.hdbscan(
        synth_locs_2d,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_eps=HDBSCAN_CLUSTER_EPS,
    )
    assert isinstance(info, list)
    info_dict = info[-1]
    assert isinstance(info_dict, dict)
    assert info_dict["Number of clusters"] == len(BLOB_CENTERS_2D)


def test_cluster_returns_info(synth_locs_2d):
    out, info = clusterer.cluster(
        synth_locs_2d,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
    )
    assert isinstance(info, list)
    info_dict = info[-1]
    assert isinstance(info_dict, dict)
    assert info_dict["Number of clusters"] == len(BLOB_CENTERS_2D)
    assert info_dict["Clustering radius (nm)"] == GA_RADIUS


def test_info_records_radii_in_nm(synth_locs_3d):
    _, info = clusterer.cluster(
        synth_locs_3d,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
        radius_z=GA_RADIUS_Z,
    )
    info_dict = info[-1]
    assert info_dict["Clustering radius xy (nm)"] == GA_RADIUS
    assert info_dict["Clustering radius z (nm)"] == GA_RADIUS_Z
    _, info = clusterer.dbscan(
        synth_locs_3d,
        radius=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        radius_z=2 * DBSCAN_EPS,
    )
    info_dict = info[-1]
    assert info_dict["Radius (nm)"] == DBSCAN_EPS
    assert info_dict["Radius z (nm)"] == 2 * DBSCAN_EPS
    _, info = clusterer.hdbscan(
        synth_locs_3d,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_eps=HDBSCAN_CLUSTER_EPS,
    )
    info_dict = info[-1]
    assert info_dict["Intercluster distance (nm)"] == HDBSCAN_CLUSTER_EPS


# ---------------------------------------------------------------------
# find_cluster_centers
# ---------------------------------------------------------------------


def test_find_cluster_centers_2d(synth_locs_2d):
    """2D centers have 'area' column, no z, and land near the truth."""
    db_locs = _run_dbscan_2d(synth_locs_2d)
    centers, _ = clusterer.find_cluster_centers(db_locs)

    expected_cols = {
        "x",
        "y",
        "area",
        "convexhull",
        "n_locs",
        "n_events",
        "group",
    }
    assert expected_cols.issubset(centers.columns)
    assert "z" not in centers.columns
    assert "volume" not in centers.columns
    assert len(centers) == len(BLOB_CENTERS_2D)
    _match_truth_to_recovered(
        centers[["x", "y"]].to_numpy(), BLOB_CENTERS_2D, tol=CENTER_TOL
    )
    # std_x / std_y and the precisions are in nm, like the blobs
    np.testing.assert_allclose(centers["std_x"], SIGMA_XY_NM, rtol=0.5)
    np.testing.assert_allclose(
        centers["lpx"], centers["std_x"] / np.sqrt(centers["n_locs"])
    )
    # area (nm^2) assumes a radius of two standard deviations
    np.testing.assert_allclose(
        centers["area"], np.pi * (centers["std_x"] + centers["std_y"]) ** 2
    )
    assert (centers["convexhull"] < centers["area"]).all()


@pytest.mark.parametrize("synth", ["synth_locs_2d", "synth_locs_3d"])
def test_find_cluster_centers_of_no_clusters(synth, request):
    """No clusters (e.g., parameters too strict) give no centers, with
    the columns of the centers of clusters."""
    db_locs = _run_dbscan_2d(request.getfixturevalue(synth))
    centers, _ = clusterer.find_cluster_centers(db_locs)
    no_centers, _ = clusterer.find_cluster_centers(db_locs.iloc[:0])
    assert no_centers.empty
    assert list(no_centers.columns) == list(centers.columns)


def test_find_cluster_centers_3d(synth_locs_3d):
    """3D centers expose volume / std_z / z columns, in nm."""
    db_locs = _run_dbscan_3d(synth_locs_3d)
    centers, _ = clusterer.find_cluster_centers(db_locs)

    expected_cols = {"x", "y", "z", "volume", "std_z", "convexhull", "group"}
    assert expected_cols.issubset(centers.columns)
    assert "area" not in centers.columns
    assert len(centers) == len(BLOB_CENTERS_3D)
    np.testing.assert_allclose(centers["std_z"], SIGMA_Z_NM, rtol=0.5)
    # volume (nm^3) of a sphere with a radius of two mean standard
    # deviations
    radius = 2 * (centers["std_x"] + centers["std_y"] + centers["std_z"]) / 3
    np.testing.assert_allclose(
        centers["volume"], 4 / 3 * np.pi * radius**3, rtol=1e-4
    )
    assert (centers["convexhull"] > 0).all()


# Columns that localizations are not required to have; each maps to the
# center columns that can only be computed when it is present.
OPTIONAL_COLUMN_OUTPUTS = {
    "photons": {"photons"},
    "sx": {"sx", "ellipticity"},
    "sy": {"sy", "ellipticity"},
    "bg": {"bg"},
    "net_gradient": {"net_gradient"},
    "lpx": set(),
    "lpy": set(),
}
ESSENTIAL_CENTER_COLS = {
    "frame",
    "std_frame",
    "x",
    "y",
    "std_x",
    "std_y",
    "lpx",
    "lpy",
    "n_locs",
    "n_events",
    "convexhull",
    "group",
}


def test_find_cluster_centers_2d_minimal_columns(synth_locs_2d):
    """Only frame/x/y present: centers are still computed, and columns
    derived from missing non-essential columns are dropped."""
    minimal = synth_locs_2d[["frame", "x", "y"]]
    centers, _ = clusterer.find_cluster_centers(_run_dbscan_2d(minimal))

    assert (ESSENTIAL_CENTER_COLS | {"area"}).issubset(centers.columns)
    for col in ("photons", "sx", "sy", "bg", "net_gradient", "ellipticity"):
        assert col not in centers.columns
    assert len(centers) == len(BLOB_CENTERS_2D)
    _match_truth_to_recovered(
        centers[["x", "y"]].to_numpy(), BLOB_CENTERS_2D, tol=CENTER_TOL
    )


def test_find_cluster_centers_3d_minimal_columns(synth_locs_3d):
    """3D without lpx/lpy: z falls back to the unweighted cluster mean."""
    minimal = synth_locs_3d[["frame", "x", "y", "z"]]
    db_locs = _run_dbscan_3d(minimal)
    centers, _ = clusterer.find_cluster_centers(db_locs)

    assert (ESSENTIAL_CENTER_COLS | {"z", "lpz", "std_z", "volume"}).issubset(
        centers.columns
    )
    assert "ellipticity" not in centers.columns
    assert len(centers) == len(BLOB_CENTERS_3D)
    expected_z = db_locs.groupby("group", sort=True)["z"].mean().to_numpy()
    np.testing.assert_allclose(centers["z"].to_numpy(), expected_z, rtol=1e-5)


@pytest.mark.parametrize("dropped", sorted(OPTIONAL_COLUMN_OUTPUTS))
def test_find_cluster_centers_single_column_missing(synth_locs_2d, dropped):
    """Dropping any one non-essential column must not raise, and only
    the columns derived from it disappear."""
    full, _ = clusterer.find_cluster_centers(_run_dbscan_2d(synth_locs_2d))
    reduced, _ = clusterer.find_cluster_centers(
        _run_dbscan_2d(synth_locs_2d.drop(columns=dropped))
    )

    lost = set(full.columns) - set(reduced.columns)
    assert lost == OPTIONAL_COLUMN_OUTPUTS[dropped]
    assert len(reduced) == len(full)


def test_find_cluster_centers_averages_custom_columns(synth_locs_2d):
    """Any extra numeric column is carried over as its cluster mean;
    non-numeric columns are ignored rather than raising."""
    locs = synth_locs_2d.copy()
    locs["my_metric"] = np.arange(len(locs), dtype=np.float32)
    locs["my_label"] = "channel_a"
    db_locs = _run_dbscan_2d(locs)
    centers, _ = clusterer.find_cluster_centers(db_locs)

    assert "my_label" not in centers.columns
    expected = db_locs.groupby("group", sort=True)["my_metric"].mean()
    np.testing.assert_allclose(
        centers["my_metric"].to_numpy(), expected.to_numpy(), rtol=1e-5
    )


def test_find_cluster_centers_recomputes_derived_columns(synth_locs_2d):
    """Re-running on centers recomputes derived columns (e.g. n_locs)
    instead of averaging the input values of the same name."""
    centers, _ = clusterer.find_cluster_centers(_run_dbscan_2d(synth_locs_2d))
    centers["group"] = 0  # merge all centers into one cluster
    recentered, _ = clusterer.find_cluster_centers(centers)

    assert len(recentered) == 1
    assert recentered["n_locs"].iloc[0] == len(centers)
    assert not any(c.endswith("_mean") for c in recentered.columns)
    assert list(recentered.columns).count("n_locs") == 1


# ---------------------------------------------------------------------
# cluster_areas
# ---------------------------------------------------------------------


def test_cluster_areas_2d(synth_locs_2d, synth_info):
    db_locs = _run_dbscan_2d(synth_locs_2d)
    areas = clusterer.cluster_areas(db_locs, synth_info)
    assert "Area (LP^2)" in areas.columns
    assert "Volume (LP^3)" not in areas.columns
    assert len(areas) == len(BLOB_CENTERS_2D)
    assert (areas["Area (LP^2)"] > 0).all()


def test_cluster_areas_3d(synth_locs_3d, synth_info):
    db_locs = _run_dbscan_3d(synth_locs_3d)
    areas = clusterer.cluster_areas(db_locs, synth_info)
    assert "Volume (LP^3)" in areas.columns
    assert "Area (LP^2)" not in areas.columns
    assert len(areas) == len(BLOB_CENTERS_3D)
    assert (areas["Volume (LP^3)"] > 0).all()


def test_cluster_areas_real_data(db_locs, info):
    """Smoke check on real data — preserves prior coverage."""
    areas = clusterer.cluster_areas(db_locs, info)
    assert len(areas) > 0
    assert (areas["Area (LP^2)"] > 0).all()


def test_find_cluster_centers_real_data(db_locs):
    """Smoke check on real data — preserves prior coverage."""
    centers, _ = clusterer.find_cluster_centers(db_locs)
    assert len(centers) > 0
    assert {"x", "y", "group"}.issubset(centers.columns)


# ---------------------------------------------------------------------
# Progress reporting (see picasso.lib.Progress)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("progress", [None, "console"])
def test_cluster_progress_modes(synth_locs_2d, progress):
    out, _ = clusterer.cluster(
        synth_locs_2d,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
        progress=progress,
    )
    assert len(np.unique(out["group"])) == len(BLOB_CENTERS_2D)


def test_cluster_reports_progress_to_tracker(synth_locs_2d):
    tracker = RecordingProgress()
    clusterer.cluster(
        synth_locs_2d,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
        progress=tracker,
    )
    assert_phases(tracker, ["Clustering"])
    assert tracker.phases[0].maximum == len(synth_locs_2d)


def _clustered(synth_locs_2d):
    return clusterer.cluster(
        synth_locs_2d,
        radius_xy=GA_RADIUS,
        min_locs=GA_MIN_LOCS,
        frame_analysis=False,
    )[0]


PROGRESS_CASES = [
    (
        "cluster",
        lambda locs, info, p: clusterer.cluster(
            locs,
            radius_xy=GA_RADIUS,
            min_locs=GA_MIN_LOCS,
            frame_analysis=False,
            progress=p,
        ),
        ["Clustering"],
    ),
    (
        "cluster_3d",
        lambda locs, info, p: clusterer.cluster(
            locs.assign(z=0.0),
            radius_xy=GA_RADIUS,
            min_locs=GA_MIN_LOCS,
            frame_analysis=False,
            radius_z=2 * GA_RADIUS,
            progress=p,
        ),
        ["Clustering"],
    ),
    (
        "find_cluster_centers",
        lambda locs, info, p: clusterer.find_cluster_centers(
            _clustered(locs), progress=p
        ),
        ["Calculating cluster centers"],
    ),
    (
        "cluster_areas",
        lambda locs, info, p: clusterer.cluster_areas(
            _clustered(locs).assign(lpx=5.0, lpy=5.0), info, progress=p
        ),
        ["Calculating cluster areas"],
    ),
]
PROGRESS_IDS = [case[0] for case in PROGRESS_CASES]


@pytest.mark.parametrize("name, run, labels", PROGRESS_CASES, ids=PROGRESS_IDS)
def test_progress_none_reports_nothing(
    capsys, synth_locs_2d, synth_info, name, run, labels
):
    """``cluster_areas`` used to print a tqdm bar without a tracker."""
    run(synth_locs_2d, synth_info, None)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("name, run, labels", PROGRESS_CASES, ids=PROGRESS_IDS)
def test_progress_console(
    capsys, synth_locs_2d, synth_info, name, run, labels
):
    run(synth_locs_2d, synth_info, "console")
    err = capsys.readouterr().err
    for label in labels:
        assert label in err


@pytest.mark.parametrize("name, run, labels", PROGRESS_CASES, ids=PROGRESS_IDS)
def test_progress_tracker(synth_locs_2d, synth_info, name, run, labels):
    tracker = RecordingProgress()
    run(synth_locs_2d, synth_info, tracker)
    assert_phases(tracker, labels)
