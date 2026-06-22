"""Test picasso functions related to undrifting.

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot/picasso imports

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from picasso import aim, io, lib, postprocess  # noqa: E402

from tests.conftest import PIXELSIZE  # noqa: E402

# undrifting parameters
SEGMENTATION = 100

# synthetic-data parameters
N_FRAMES_SYNTH = 1000
SYNTH_FOV = 64
RNG_SEED = 42
SYNTH_PICKS = [
    (10.0, 10.0),
    (10.0, 30.0),
    (10.0, 50.0),
    (30.0, 10.0),
    (30.0, 30.0),
    (30.0, 50.0),
    (50.0, 10.0),
    (50.0, 30.0),
    (50.0, 50.0),
]
SYNTH_PICK_RADIUS = 2.0  # camera pixels — comfortably > drift+noise

# tolerances for ground-truth recovery (px)
TOL_PICKED_PX = 0.1
TOL_RCC_PX = 0.5
TOL_AIM_PX = 0.5
TOL_PICKED_Z_PX = 0.2

# AIM defaults are tuned for tiny drifts (~0.5 px ROI radius). The
# synthetic injected drift goes up to ~1 px, so widen the search ROI
# in tests that exercise AIM on synthetic data.
AIM_ROI_R_SYNTH = 2.5  # camera pixels
AIM_INTERSECT_D_SYNTH = 0.2  # camera pixels

# AIM treats z as nm and converts to camera pixels with Pixelsize, unlike
# the picked/fiducial undrifters which track the raw z column. So the AIM
# 3D fixtures express the injected z drift in nm. The trace must be smooth
# and slow relative to the segmentation — AIM's adaptive tracker estimates
# drift one segment at a time, so a fast z oscillation that swings by more
# than the ROI within a single segment is not recoverable.
AIM_Z_DRIFT_AMP_NM = 50.0  # nm; ~0.38 px at PIXELSIZE, < AIM_ROI_R_SYNTH
TOL_AIM_Z_NM = 20.0  # nm


# ---------------------------------------------------------------------
# Real-data fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def locs_data():
    """Load the shared 564-loc / 1000-frame test file once."""
    return io.load_locs("./tests/data/testdata_locs.hdf5")


@pytest.fixture(scope="module")
def locs(locs_data):
    return locs_data[0]


@pytest.fixture(scope="module")
def info(locs_data):
    return locs_data[1]


@pytest.fixture(scope="module")
def picks_grid():
    """3x3 grid of pick centers matching the simulated origami layout."""
    return [
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


# ---------------------------------------------------------------------
# Synthetic fiducial fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_info():
    return [
        {
            "Width": SYNTH_FOV,
            "Height": SYNTH_FOV,
            "Frames": N_FRAMES_SYNTH,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture(scope="module")
def injected_drift_2d():
    """Smooth analytic xy drift trace, < 1 px in each direction."""
    t = np.arange(N_FRAMES_SYNTH)
    return pd.DataFrame(
        {
            "x": 1.0 * np.sin(2 * np.pi * t / 500),
            "y": 0.7 * (t / N_FRAMES_SYNTH - 0.5),
        }
    )


@pytest.fixture(scope="module")
def injected_drift_3d(injected_drift_2d):
    drift = injected_drift_2d.copy()
    t = np.arange(N_FRAMES_SYNTH)
    drift["z"] = 0.5 * np.cos(2 * np.pi * t / 250)
    return drift


def _build_fiducials(centers, drift_df, with_z=False):
    """Emit one localization per frame per pick, drifted + jittered."""
    rng = np.random.default_rng(RNG_SEED)
    sigma = 0.05
    n_frames = len(drift_df)
    n_per = n_frames * len(centers)
    frame = np.tile(np.arange(n_frames), len(centers)).astype(np.int32)
    pick_x = np.repeat([c[0] for c in centers], n_frames)
    pick_y = np.repeat([c[1] for c in centers], n_frames)
    drift_x_long = np.tile(drift_df["x"].to_numpy(), len(centers))
    drift_y_long = np.tile(drift_df["y"].to_numpy(), len(centers))
    df = pd.DataFrame(
        {
            "frame": frame,
            "x": pick_x + drift_x_long + rng.normal(0, sigma, n_per),
            "y": pick_y + drift_y_long + rng.normal(0, sigma, n_per),
            "photons": np.full(n_per, 1000.0),
            "sx": np.full(n_per, 1.0),
            "sy": np.full(n_per, 1.0),
            "bg": np.full(n_per, 10.0),
            "lpx": np.full(n_per, 0.01),
            "lpy": np.full(n_per, 0.01),
            "net_gradient": np.full(n_per, 5000.0),
        }
    )
    if with_z:
        drift_z_long = np.tile(drift_df["z"].to_numpy(), len(centers))
        df["z"] = drift_z_long + rng.normal(0, sigma, n_per)
        df["lpz"] = np.full(n_per, 0.02)
    return df.sort_values("frame").reset_index(drop=True)


@pytest.fixture(scope="module")
def synthetic_fiducials_2d(synthetic_info, injected_drift_2d):
    """Synthetic locs with `x = pick + injected_drift_2d + noise`."""
    return _build_fiducials(SYNTH_PICKS, injected_drift_2d), synthetic_info


@pytest.fixture(scope="module")
def synthetic_fiducials_3d(synthetic_info, injected_drift_3d):
    return (
        _build_fiducials(SYNTH_PICKS, injected_drift_3d, with_z=True),
        synthetic_info,
    )


@pytest.fixture(scope="module")
def injected_drift_3d_aim(injected_drift_2d):
    """xy drift in camera px (< 1 px) + z drift in nm, for AIM.

    AIM expects z in nm (it converts to camera pixels internally with
    Pixelsize), so the z trace here is scaled to nm — unlike
    ``injected_drift_3d`` whose z is in the same generic units as x/y.
    """
    drift = injected_drift_2d.copy()
    t = np.arange(N_FRAMES_SYNTH)
    drift["z"] = AIM_Z_DRIFT_AMP_NM * np.sin(2 * np.pi * t / N_FRAMES_SYNTH)
    return drift


@pytest.fixture(scope="module")
def synthetic_fiducials_3d_aim(synthetic_info, injected_drift_3d_aim):
    """Synthetic 3D locs with z drift expressed in nm (for AIM)."""
    return (
        _build_fiducials(SYNTH_PICKS, injected_drift_3d_aim, with_z=True),
        synthetic_info,
    )


@pytest.fixture(scope="module")
def synthetic_clean_locs(synthetic_info):
    """Same fiducials with zero drift — for ground-truth comparison."""
    zero_drift = pd.DataFrame(
        {
            "x": np.zeros(N_FRAMES_SYNTH),
            "y": np.zeros(N_FRAMES_SYNTH),
        }
    )
    return _build_fiducials(SYNTH_PICKS, zero_drift), synthetic_info


@pytest.fixture(scope="module")
def synthetic_picked_locs_2d(synthetic_fiducials_2d):
    locs_, info_ = synthetic_fiducials_2d
    return postprocess.picked_locs(
        locs_,
        info_,
        SYNTH_PICKS,
        "Circle",
        pick_size=SYNTH_PICK_RADIUS,
        add_group=False,
    )


@pytest.fixture(scope="module")
def synthetic_picked_locs_3d(synthetic_fiducials_3d):
    locs_, info_ = synthetic_fiducials_3d
    return postprocess.picked_locs(
        locs_,
        info_,
        SYNTH_PICKS,
        "Circle",
        pick_size=SYNTH_PICK_RADIUS,
        add_group=False,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _assert_drift_recovers(recovered, injected, tol, coords=("x", "y")):
    """Assert recovered drift matches injected within tol after demeaning.

    Drift is defined up to a constant offset; subtract per-column means
    before comparing.
    """
    assert isinstance(recovered, pd.DataFrame)
    assert set(coords).issubset(
        recovered.columns
    ), f"Expected columns {coords}, got {list(recovered.columns)}"
    assert len(recovered) == len(injected)
    assert recovered[list(coords)].isna().sum().sum() == 0
    for c in coords:
        rec = recovered[c].to_numpy()
        inj = injected[c].to_numpy()
        diff = (rec - rec.mean()) - (inj - inj.mean())
        assert (
            np.max(np.abs(diff)) < tol
        ), f"Coord '{c}' worst error {np.max(np.abs(diff)):.3f} >= tol {tol}"


def _assert_undrifted_locs_invariants(undrifted, original):
    assert len(undrifted) == len(original)
    assert (
        undrifted["frame"].to_numpy() == original["frame"].to_numpy()
    ).all()
    assert np.isfinite(undrifted["x"].to_numpy()).all()
    assert np.isfinite(undrifted["y"].to_numpy()).all()


def _make_drift_df(n_frames, with_z=False):
    t = np.arange(n_frames)
    df = pd.DataFrame({"x": 0.01 * t, "y": -0.005 * t})
    if with_z:
        df["z"] = 0.002 * t
    return df


# ---------------------------------------------------------------------
# n_segments
# ---------------------------------------------------------------------


def test_n_segments_basic():
    assert postprocess.n_segments([{"Frames": 1000}], 100) == 10


@pytest.mark.parametrize(
    "n_frames,segmentation,expected",
    [(1051, 100, 11), (1049, 100, 10), (1050, 100, 10)],  # banker's rounding
)
def test_n_segments_rounding(n_frames, segmentation, expected):
    assert (
        postprocess.n_segments([{"Frames": n_frames}], segmentation)
        == expected
    )


def test_n_segments_uses_last_info_entry():
    info = [{"Frames": 9}, {"Frames": 1000}]
    assert postprocess.n_segments(info, 100) == 10


# ---------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------


def test_segment_shapes(locs, info):
    bounds, segments = postprocess.segment(locs, info, SEGMENTATION)
    n_seg = postprocess.n_segments(info, SEGMENTATION)
    n_frames = info[0]["Frames"]
    assert bounds.shape == (n_seg + 1,)
    assert bounds.dtype == np.uint32
    assert bounds[0] == 0
    assert bounds[-1] == n_frames - 1
    assert segments.shape == (n_seg, info[0]["Height"], info[0]["Width"])
    assert segments.sum() > 0


def test_segment_callback_invocations(locs, info):
    n_seg = postprocess.n_segments(info, SEGMENTATION)
    events = []
    postprocess.segment(locs, info, SEGMENTATION, callback=events.append)
    assert events == list(range(n_seg + 1))


def test_segment_total_count_matches_locs(synthetic_fiducials_2d):
    locs_, info_ = synthetic_fiducials_2d
    _, segments = postprocess.segment(locs_, info_, SEGMENTATION)
    # Default render is histogram-style. The last segment uses a strict
    # `< bounds[-1]` upper bound so locs at the very last frame are
    # dropped — accept the small undercount but reject any other gap.
    last_frame_locs = int((locs_["frame"] == locs_["frame"].max()).sum())
    assert int(round(segments.sum())) == len(locs_) - last_frame_locs


# ---------------------------------------------------------------------
# undrift / RCC
# ---------------------------------------------------------------------


def test_rcc_smoke(locs, info):
    drift, undrifted = postprocess.undrift(
        locs, info, segmentation=SEGMENTATION, display=False
    )
    n_frames = lib.get_from_metadata(info, "Frames")
    assert isinstance(drift, pd.DataFrame)
    assert {"x", "y"}.issubset(drift.columns)
    assert len(drift) == n_frames
    assert drift.isna().sum().sum() == 0
    assert np.allclose(drift.mean(axis=0), 0, atol=1.0)
    _assert_undrifted_locs_invariants(undrifted, locs)


def test_rcc_recovers_injected_drift(
    synthetic_fiducials_2d, injected_drift_2d
):
    locs_, info_ = synthetic_fiducials_2d
    drift, _ = postprocess.undrift(
        locs_, info_, segmentation=SEGMENTATION, display=False
    )
    _assert_drift_recovers(drift, injected_drift_2d, tol=TOL_RCC_PX)


def test_rcc_display_branch_uses_plt_show(monkeypatch, locs, info):
    """display=True must trigger plot_drift + plt.show exactly once."""
    n_calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **kw: n_calls.append(1))
    postprocess.undrift(locs, info, segmentation=SEGMENTATION, display=True)
    assert sum(n_calls) == 1


# ---------------------------------------------------------------------
# undrift_from_fiducials
# ---------------------------------------------------------------------


def test_undrift_from_fiducials_with_picks(locs, info, picks_grid):
    pick_size = 200 / 130  # camera-pixel radius
    new_locs, new_info, drift = postprocess.undrift_from_fiducials(
        locs,
        info,
        picks=picks_grid,
        pick_size=pick_size,
    )
    n_frames = lib.get_from_metadata(info, "Frames")
    assert len(new_locs) == len(locs)
    assert {"x", "y"} == set(drift.columns)  # 2D source -> no z
    assert len(drift) == n_frames
    assert drift.isna().sum().sum() == 0

    appended = new_info[-1]
    assert appended["Number of picks"] == len(picks_grid)
    assert appended["Pick radius (nm)"] == pytest.approx(pick_size * 130)
    assert appended["Generated by"].startswith("Picasso")


def test_undrift_from_fiducials_recovers_injected_drift_2d(
    synthetic_fiducials_2d, synthetic_clean_locs, injected_drift_2d
):
    locs_, info_ = synthetic_fiducials_2d
    clean_locs, _ = synthetic_clean_locs
    new_locs, _, drift = postprocess.undrift_from_fiducials(
        locs_,
        info_,
        picks=SYNTH_PICKS,
        pick_size=SYNTH_PICK_RADIUS,
    )
    _assert_drift_recovers(drift, injected_drift_2d, tol=TOL_PICKED_PX)
    _assert_undrifted_locs_invariants(new_locs, locs_)
    # After undrifting, positions should match the un-drifted ground truth
    # within picking tolerance + jitter — slightly looser than drift tol.
    assert np.max(np.abs(new_locs["x"] - clean_locs["x"])) < 4 * TOL_PICKED_PX
    assert np.max(np.abs(new_locs["y"] - clean_locs["y"])) < 4 * TOL_PICKED_PX


def test_undrift_from_fiducials_recovers_injected_drift_3d(
    synthetic_fiducials_3d, injected_drift_3d
):
    locs_, info_ = synthetic_fiducials_3d
    _, _, drift = postprocess.undrift_from_fiducials(
        locs_,
        info_,
        picks=SYNTH_PICKS,
        pick_size=SYNTH_PICK_RADIUS,
    )
    assert "z" in drift.columns
    _assert_drift_recovers(
        drift, injected_drift_3d, tol=TOL_PICKED_PX, coords=("x", "y")
    )
    _assert_drift_recovers(
        drift, injected_drift_3d, tol=TOL_PICKED_Z_PX, coords=("z",)
    )


def test_undrift_from_fiducials_undrift_z_false(
    synthetic_fiducials_3d, injected_drift_3d
):
    locs_, info_ = synthetic_fiducials_3d
    new_locs, _, drift = postprocess.undrift_from_fiducials(
        locs_,
        info_,
        picks=SYNTH_PICKS,
        pick_size=SYNTH_PICK_RADIUS,
        undrift_z=False,
    )
    assert "z" not in drift.columns
    # z was not corrected — residual z must still carry the injected
    # z drift (its standard deviation should match within jitter).
    expected_std = injected_drift_3d["z"].std()
    assert new_locs["z"].std() == pytest.approx(expected_std, rel=0.1)


def test_undrift_from_fiducials_auto_detect(
    monkeypatch, synthetic_fiducials_2d, injected_drift_2d
):
    locs_, info_ = synthetic_fiducials_2d
    box = SYNTH_PICK_RADIUS * 2

    def fake_find_fiducials(_locs, _info):
        return SYNTH_PICKS, box

    monkeypatch.setattr(
        "picasso.postprocess.imageprocess.find_fiducials", fake_find_fiducials
    )
    _, _, drift = postprocess.undrift_from_fiducials(locs_, info_, picks=None)
    _assert_drift_recovers(drift, injected_drift_2d, tol=TOL_PICKED_PX)


def test_undrift_from_fiducials_missing_pick_size_raises(
    locs, info, picks_grid
):
    with pytest.raises(ValueError, match="pick_size"):
        postprocess.undrift_from_fiducials(locs, info, picks=picks_grid)


def test_undrift_from_fiducials_empty_picks_direct_raises(locs, info):
    with pytest.raises(ValueError, match="No picks"):
        postprocess.undrift_from_fiducials(locs, info, picks=[], pick_size=1.0)


def test_undrift_from_fiducials_empty_picks_auto_raises(
    monkeypatch, locs, info
):
    monkeypatch.setattr(
        "picasso.postprocess.imageprocess.find_fiducials",
        lambda _l, _i: ([], 1.0),
    )
    with pytest.raises(ValueError, match="No picks"):
        postprocess.undrift_from_fiducials(locs, info, picks=None)


# ---------------------------------------------------------------------
# undrift_from_picked
# ---------------------------------------------------------------------


def test_undrift_from_picked_smoke(locs, info, picks_grid):
    pl = postprocess.picked_locs(
        locs,
        info,
        picks_grid,
        pick_shape="Circle",
        pick_size=200 / 130,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        drift = postprocess.undrift_from_picked(pl, info)
    n_frames = lib.get_from_metadata(info, "Frames")
    assert set(drift.columns) == {"x", "y"}
    assert len(drift) == n_frames
    assert drift.isna().sum().sum() == 0
    assert np.allclose(drift.mean(axis=0), 0, atol=1.0)


def test_undrift_from_picked_recovers_injected_drift_2d(
    synthetic_picked_locs_2d, synthetic_info, injected_drift_2d
):
    drift = postprocess.undrift_from_picked(
        synthetic_picked_locs_2d, synthetic_info
    )
    _assert_drift_recovers(drift, injected_drift_2d, tol=TOL_PICKED_PX)


def test_undrift_from_picked_3d(
    synthetic_picked_locs_3d, synthetic_info, injected_drift_3d
):
    drift = postprocess.undrift_from_picked(
        synthetic_picked_locs_3d, synthetic_info
    )
    assert "z" in drift.columns
    _assert_drift_recovers(
        drift, injected_drift_3d, tol=TOL_PICKED_PX, coords=("x", "y")
    )
    _assert_drift_recovers(
        drift, injected_drift_3d, tol=TOL_PICKED_Z_PX, coords=("z",)
    )


def test_undrift_from_picked_interpolates_missing_frames(
    synthetic_fiducials_2d, injected_drift_2d
):
    """Drop a contiguous frame block from every pick; interpolation
    must fill the resulting NaNs in `_undrift_from_picked_coordinate`."""
    locs_, info_ = synthetic_fiducials_2d
    drop = (200, 250)
    locs_gap = locs_[
        (locs_["frame"] < drop[0]) | (locs_["frame"] >= drop[1])
    ].copy()
    pl = postprocess.picked_locs(
        locs_gap,
        info_,
        SYNTH_PICKS,
        "Circle",
        pick_size=SYNTH_PICK_RADIUS,
        add_group=False,
    )
    drift = postprocess.undrift_from_picked(pl, info_)
    assert drift.isna().sum().sum() == 0
    # Recovered drift should still track the injection — looser tol because
    # the gap range is filled by linear interpolation of a sinusoid.
    _assert_drift_recovers(drift, injected_drift_2d, tol=2 * TOL_PICKED_PX)


# ---------------------------------------------------------------------
# apply_drift
# ---------------------------------------------------------------------


def test_apply_drift_dataframe_2d(synthetic_fiducials_2d, injected_drift_2d):
    locs_, info_ = synthetic_fiducials_2d
    out = postprocess.apply_drift(locs_.copy(), info_, drift=injected_drift_2d)
    # Each loc was generated as pick + drift + noise; subtracting drift
    # leaves pick + noise. SYNTH_PICKS sit at x in {10, 30, 50}, so each
    # x must land within a few sigma of one of those columns.
    x = out["x"].to_numpy()
    nearest = np.clip(np.round((x - 10) / 20) * 20 + 10, 10, 50)
    assert np.max(np.abs(x - nearest)) < 0.3


def test_apply_drift_dataframe_3d(synthetic_fiducials_3d, injected_drift_3d):
    locs_, info_ = synthetic_fiducials_3d
    out = postprocess.apply_drift(locs_.copy(), info_, drift=injected_drift_3d)
    # After undrifting z, residual z should be ~ noise (sigma 0.05).
    assert out["z"].std() < 0.1


def test_apply_drift_ndarray_matches_dataframe(
    synthetic_fiducials_2d, injected_drift_2d
):
    locs_, info_ = synthetic_fiducials_2d
    drift_arr = np.column_stack(
        [injected_drift_2d["x"], injected_drift_2d["y"]]
    )
    out_arr = postprocess.apply_drift(locs_.copy(), info_, drift=drift_arr)
    out_df = postprocess.apply_drift(
        locs_.copy(), info_, drift=injected_drift_2d
    )
    np.testing.assert_allclose(out_arr["x"], out_df["x"])
    np.testing.assert_allclose(out_arr["y"], out_df["y"])


def test_apply_drift_ndarray_3d_applies_z(
    synthetic_fiducials_3d, injected_drift_3d
):
    locs_, info_ = synthetic_fiducials_3d
    drift_arr = np.column_stack(
        [
            injected_drift_3d["x"],
            injected_drift_3d["y"],
            injected_drift_3d["z"],
        ]
    )
    out = postprocess.apply_drift(locs_.copy(), info_, drift=drift_arr)
    assert out["z"].std() < 0.1


def test_apply_drift_does_not_mutate_drift(
    synthetic_fiducials_2d, injected_drift_2d
):
    locs_, info_ = synthetic_fiducials_2d
    snapshot = injected_drift_2d.copy()
    postprocess.apply_drift(locs_.copy(), info_, drift=injected_drift_2d)
    pd.testing.assert_frame_equal(injected_drift_2d, snapshot)


def test_apply_drift_wrong_type_raises(synthetic_fiducials_2d):
    locs_, info_ = synthetic_fiducials_2d
    with pytest.raises(AssertionError):
        postprocess.apply_drift(
            locs_.copy(), info_, drift=[[0.0, 0.0]] * N_FRAMES_SYNTH
        )


def test_apply_drift_missing_columns_raises(synthetic_fiducials_2d):
    locs_, info_ = synthetic_fiducials_2d
    bad = pd.DataFrame({"x": np.zeros(N_FRAMES_SYNTH)})
    with pytest.raises(ValueError, match="columns"):
        postprocess.apply_drift(locs_.copy(), info_, drift=bad)


@pytest.mark.parametrize(
    "shape",
    [
        (N_FRAMES_SYNTH, 1),
        (N_FRAMES_SYNTH, 4),
        (N_FRAMES_SYNTH - 1, 2),
        (N_FRAMES_SYNTH + 1, 2),
    ],
    ids=["1col", "4col", "short", "long"],
)
def test_apply_drift_wrong_ndarray_shape_raises(synthetic_fiducials_2d, shape):
    locs_, info_ = synthetic_fiducials_2d
    with pytest.raises(ValueError, match="shape"):
        postprocess.apply_drift(locs_.copy(), info_, drift=np.zeros(shape))


def test_apply_drift_missing_frames_metadata_raises(synthetic_fiducials_2d):
    locs_, _ = synthetic_fiducials_2d
    with pytest.raises(KeyError, match="Frames"):
        postprocess.apply_drift(
            locs_.copy(), [{}], drift=_make_drift_df(N_FRAMES_SYNTH)
        )


# ---------------------------------------------------------------------
# plot_drift
# ---------------------------------------------------------------------


def test_plot_drift_2d_returns_figure(injected_drift_2d):
    fig = postprocess.plot_drift(injected_drift_2d, PIXELSIZE)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        ax_t, ax_xy = fig.axes
        assert ax_t.get_xlabel() == "Frame"
        assert ax_t.get_ylabel() == "Drift (nm)"
        assert ax_xy.get_xlabel() == "x (nm)"
        assert ax_xy.get_ylabel() == "y (nm)"
    finally:
        plt.close(fig)


def test_plot_drift_3d_returns_figure(injected_drift_3d):
    fig = postprocess.plot_drift(injected_drift_3d, PIXELSIZE)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        assert fig.axes[2].get_ylabel() == "Drift (nm)"
    finally:
        plt.close(fig)


def test_plot_drift_reuses_passed_figure(injected_drift_2d):
    fig = plt.Figure()
    fig.add_subplot(111)  # dummy axis
    out = postprocess.plot_drift(injected_drift_2d, PIXELSIZE, fig=fig)
    try:
        assert out is fig
        # fig.clear() runs first, then 2 subplots are added.
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_plot_drift_wrong_type_raises():
    with pytest.raises(AssertionError):
        postprocess.plot_drift(np.zeros((10, 2)), PIXELSIZE)


def test_plot_drift_missing_columns_raises():
    bad = pd.DataFrame({"a": np.zeros(10)})
    with pytest.raises(AssertionError, match=r"x.*y"):
        postprocess.plot_drift(bad, PIXELSIZE)


# ---------------------------------------------------------------------
# aim
# ---------------------------------------------------------------------


def test_aim_smoke(locs, info):
    new_locs, new_info, drift = aim.aim(locs, info, segmentation=SEGMENTATION)
    n_frames = lib.get_from_metadata(info, "Frames")
    assert isinstance(drift, pd.DataFrame)
    assert {"x", "y"}.issubset(drift.columns)
    assert len(drift) == n_frames
    assert drift.isna().sum().sum() == 0
    assert np.allclose(drift.mean(axis=0), 0, atol=1.0)
    _assert_undrifted_locs_invariants(new_locs, locs)
    assert new_info[-1]["Generated by"].startswith("Picasso")
    assert new_info[-1]["Segmentation"] == SEGMENTATION


def test_aim_recovers_injected_drift(
    synthetic_fiducials_2d, injected_drift_2d
):
    locs_, info_ = synthetic_fiducials_2d
    _, _, drift = aim.aim(
        locs_,
        info_,
        segmentation=SEGMENTATION,
        intersect_d=AIM_INTERSECT_D_SYNTH,
        roi_r=AIM_ROI_R_SYNTH,
    )
    _assert_drift_recovers(drift, injected_drift_2d, tol=TOL_AIM_PX)


def test_aim_3d_smoke(synthetic_fiducials_3d_aim):
    locs_, info_ = synthetic_fiducials_3d_aim
    new_locs, new_info, drift = aim.aim(
        locs_,
        info_,
        segmentation=SEGMENTATION,
        intersect_d=AIM_INTERSECT_D_SYNTH,
        roi_r=AIM_ROI_R_SYNTH,
    )
    n_frames = lib.get_from_metadata(info_, "Frames")
    assert isinstance(drift, pd.DataFrame)
    assert {"x", "y", "z"} == set(drift.columns)
    assert len(drift) == n_frames
    assert drift.isna().sum().sum() == 0
    # drift is demeaned by aim() in every dimension
    assert np.allclose(drift.mean(axis=0), 0, atol=1.0)
    _assert_undrifted_locs_invariants(new_locs, locs_)
    assert np.isfinite(new_locs["z"].to_numpy()).all()
    assert new_info[-1]["Generated by"].startswith("Picasso")
    assert new_info[-1]["Segmentation"] == SEGMENTATION


def test_aim_recovers_injected_drift_3d(
    synthetic_fiducials_3d_aim, injected_drift_3d_aim
):
    locs_, info_ = synthetic_fiducials_3d_aim
    _, _, drift = aim.aim(
        locs_,
        info_,
        segmentation=SEGMENTATION,
        intersect_d=AIM_INTERSECT_D_SYNTH,
        roi_r=AIM_ROI_R_SYNTH,
    )
    assert "z" in drift.columns
    _assert_drift_recovers(
        drift, injected_drift_3d_aim, tol=TOL_AIM_PX, coords=("x", "y")
    )
    _assert_drift_recovers(
        drift, injected_drift_3d_aim, tol=TOL_AIM_Z_NM, coords=("z",)
    )


def test_aim_undrifts_z_residual(
    synthetic_fiducials_3d_aim, injected_drift_3d_aim
):
    """After AIM, the z drift in the locs should be removed: the residual
    z spread must be far smaller than the injected z drift amplitude."""
    locs_, info_ = synthetic_fiducials_3d_aim
    new_locs, _, _ = aim.aim(
        locs_,
        info_,
        segmentation=SEGMENTATION,
        intersect_d=AIM_INTERSECT_D_SYNTH,
        roi_r=AIM_ROI_R_SYNTH,
    )
    injected_std = injected_drift_3d_aim["z"].std()
    assert new_locs["z"].std() < 0.5 * injected_std


# ---------------------------------------------------------------------
# Cross-method ground-truth recovery
# ---------------------------------------------------------------------

# Adapters hide the asymmetric return-tuple ordering in the source.
# Each returns (undrifted_locs, drift) regardless of the underlying API.


def _aim_adapter(locs, info):
    new_locs, _, drift = aim.aim(
        locs,
        info,
        segmentation=SEGMENTATION,
        intersect_d=AIM_INTERSECT_D_SYNTH,
        roi_r=AIM_ROI_R_SYNTH,
    )
    return new_locs, drift


def _rcc_adapter(locs, info):
    drift, new_locs = postprocess.undrift(
        locs, info, segmentation=SEGMENTATION, display=False
    )
    return new_locs, drift


def _from_picked_adapter(locs, info):
    pl = postprocess.picked_locs(
        locs,
        info,
        SYNTH_PICKS,
        "Circle",
        pick_size=SYNTH_PICK_RADIUS,
        add_group=False,
    )
    drift = postprocess.undrift_from_picked(pl, info)
    new_locs = postprocess.apply_drift(locs.copy(), info, drift=drift)
    return new_locs, drift


@pytest.mark.parametrize(
    "adapter,tol",
    [
        pytest.param(_from_picked_adapter, TOL_PICKED_PX, id="from_picked"),
        pytest.param(_rcc_adapter, TOL_RCC_PX, id="rcc"),
        pytest.param(_aim_adapter, TOL_AIM_PX, id="aim"),
    ],
)
def test_recovers_known_drift(
    synthetic_fiducials_2d, injected_drift_2d, adapter, tol
):
    locs_, info_ = synthetic_fiducials_2d
    new_locs, drift = adapter(locs_, info_)
    _assert_drift_recovers(drift, injected_drift_2d, tol=tol)
    _assert_undrifted_locs_invariants(new_locs, locs_)
