"""Tests of single-particle tracking: linking, MSD, jump distances and the
analysis and its outputs (``picasso.spt``), and the ``picasso track`` and
``picasso diffusion`` commands.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import check_grad

from picasso import __main__ as cli
from picasso import io, spt

PIXELSIZE = 100.0  # nm
FRAME_TIME = 20.0  # ms


def _info(n_frames: int = 1000, size: float = 512.0) -> list[dict]:
    return [
        {
            "Width": size,
            "Height": size,
            "Frames": n_frames,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _brownian(
    n_particles: int = 100,
    n_frames: int = 30,
    D: float = 0.1,
    sigma: float = 0.0,
    size: float = 50_000.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Freely diffusing particles (D in um^2/s, sigma in nm) in a field
    of ``size`` nm, each seen in ``n_frames`` consecutive frames starting
    at a random frame; the true particle is in the column ``truth``."""
    rng = np.random.default_rng(seed)
    step = np.sqrt(2 * D * FRAME_TIME / 1000) * 1000  # nm
    frames, xs, ys, truth = [], [], [], []
    for p in range(n_particles):
        xy = rng.uniform(500, size - 500, 2) + np.cumsum(
            rng.normal(0, step, (n_frames, 2)), axis=0
        )
        xy += rng.normal(0, sigma, xy.shape)
        first = rng.integers(0, 200)
        frames.append(np.arange(first, first + n_frames))
        xs.append(xy[:, 0])
        ys.append(xy[:, 1])
        truth.append(np.full(n_frames, p))
    locs = pd.DataFrame(
        {
            "frame": np.concatenate(frames).astype(np.uint32),
            "x": np.concatenate(xs).astype(np.float32),
            "y": np.concatenate(ys).astype(np.float32),
            "lpx": np.float32(20.0),
            "lpy": np.float32(30.0),
            "truth": np.concatenate(truth),
        }
    )
    return locs.sort_values("frame", kind="stable").reset_index(drop=True)


def _locs(frame, x, y=None, **columns) -> pd.DataFrame:
    """Hand-made localizations."""
    x = np.asarray(x, dtype=np.float32)
    data = {
        "frame": np.asarray(frame, dtype=np.uint32),
        "x": x,
        "y": np.zeros_like(x) if y is None else np.asarray(y, np.float32),
    }
    data.update(columns)
    return pd.DataFrame(data)


def _track_of(tracked: pd.DataFrame, frame: int, x: float) -> int:
    row = tracked[(tracked["frame"] == frame) & np.isclose(tracked["x"], x)]
    assert len(row) == 1
    return int(row["group"].iloc[0])


def _jumps(D, n, frame_time=FRAME_TIME, sigma=0.0, seed=0):
    """Jump distances (nm) of a 2D Brownian population: r^2 is
    exponentially distributed with the mean 4 D dt + 4 sigma^2."""
    rng = np.random.default_rng(seed)
    s = 4 * D * frame_time / 1000 + 4 * (sigma / 1000) ** 2  # um^2
    return np.sqrt(rng.exponential(s, n)) * 1000


# --------------------------------------------------------------------
# Linking
# --------------------------------------------------------------------


def test_link_sparse_particles_gives_pure_tracks():
    locs = _brownian(size=200_000.0)
    tracked, info = spt.link_tracks(locs, _info(), max_jump=400.0)
    assert len(tracked) == len(locs)
    assert tracked["group"].nunique() == locs["truth"].nunique()
    assert (tracked.groupby("group")["truth"].nunique() == 1).all()
    # consecutive ids in the order of the tracks' first frames
    first = tracked.groupby("group")["frame"].min()
    assert list(first.index) == list(range(len(first)))
    assert first.is_monotonic_increasing
    assert tracked["frame"].is_monotonic_increasing
    assert info[-1]["Number of tracks"] == tracked["group"].nunique()


@pytest.mark.parametrize(
    "frames, max_gap, n_tracks",
    [
        ([0, 1, 3, 4], 1, 1),
        ([0, 1, 3, 4], 0, 2),
        ([0, 1, 4, 5], 1, 2),
        ([0, 1, 4, 5], 2, 1),
    ],
)
def test_link_bridges_gaps_up_to_max_gap(frames, max_gap, n_tracks):
    locs = _locs(frames, [0.0, 0.1, 0.2, 0.3])
    tracked, _ = spt.link_tracks(locs, _info(), 1.0, max_gap=max_gap)
    assert tracked["group"].nunique() == n_tracks


def test_link_resolves_conflicts_globally():
    """The end at x = 1 is closer to 0.9 than the end at x = 0, so it
    continues there, although the end at x = 0 comes first and has 0.9
    in reach. 1.95 then starts a new track."""
    locs = _locs([0, 0, 1, 1], [0.0, 1.0, 0.9, 1.95])
    tracked, _ = spt.link_tracks(locs, _info(), 1.0, min_length=1)
    assert _track_of(tracked, 1, 0.9) == _track_of(tracked, 0, 1.0)
    assert _track_of(tracked, 1, 1.95) not in (
        _track_of(tracked, 0, 0.0),
        _track_of(tracked, 0, 1.0),
    )
    assert tracked["group"].nunique() == 3


def test_link_tie_goes_to_the_track_that_skipped_fewer_frames():
    locs = _locs([0, 1, 2], [0.0, 10.0, 5.0])
    tracked, _ = spt.link_tracks(locs, _info(), 5.0, max_gap=1, min_length=1)
    assert _track_of(tracked, 2, 5.0) == _track_of(tracked, 1, 10.0)
    assert _track_of(tracked, 2, 5.0) != _track_of(tracked, 0, 0.0)


@pytest.mark.parametrize("min_length", [1, 2])
def test_link_never_links_within_a_frame(min_length):
    """Also not in the last frame (unlike ``postprocess.link``)."""
    locs = _locs([0, 1, 1, 1], [0.0, 0.1, 0.2, 0.25])
    tracked, _ = spt.link_tracks(locs, _info(), 1.0, min_length=min_length)
    assert not tracked.duplicated(["group", "frame"]).any()
    if min_length == 1:
        assert tracked["group"].nunique() == 3
        assert _track_of(tracked, 1, 0.1) == _track_of(tracked, 0, 0.0)
    else:
        assert len(tracked) == 2  # the others are single localizations


def test_link_respects_input_groups():
    """Two particles that cross, but belong to different picks."""
    locs = _locs(
        [0, 0, 1, 1, 2, 2],
        [0.0, 2.0, 0.9, 1.1, 2.0, 0.0],
        group=np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
    )
    tracked, info = spt.link_tracks(locs, _info(), 1.5)
    assert tracked["group"].nunique() == 2
    for _, track in tracked.groupby("group"):
        assert track["group_input"].nunique() == 1
    assert info[-1]["Input groups kept in"] == "group_input"
    # without the groups, the closer localizations are linked instead
    plain, _ = spt.link_tracks(locs.drop(columns="group"), _info(), 1.5)
    assert "group_input" not in plain.columns


def test_link_min_length_removes_short_tracks():
    locs = _locs([0, 1, 2, 5, 6, 9], [0.0, 0.1, 0.2, 50.0, 50.1, 90.0])
    tracked, info = spt.link_tracks(locs, _info(), 1.0, min_length=3)
    assert len(tracked) == 3
    assert tracked["group"].unique().tolist() == [0]
    assert info[-1]["Number of tracks"] == 1
    tracked, _ = spt.link_tracks(locs, _info(), 1.0, min_length=1)
    assert tracked["group"].nunique() == 3


def test_link_empty_and_single_frame():
    empty, info = spt.link_tracks(_locs([], []), _info(), 1.0)
    assert len(empty) == 0 and "group" in empty.columns
    assert info[-1]["Number of tracks"] == 0
    single = _locs([0, 0, 0], [0.0, 0.5, 1.0])
    assert len(spt.link_tracks(single, _info(), 1.0)[0]) == 0
    tracked, _ = spt.link_tracks(single, _info(), 1.0, min_length=1)
    assert tracked["group"].nunique() == 3


def test_link_keeps_input_and_records_parameters():
    locs = _brownian(n_particles=5)
    before = locs.copy()
    info = _info()
    tracked, new_info = spt.link_tracks(
        locs, info, 300.0, max_gap=2, min_length=4, frame_time=FRAME_TIME
    )
    pd.testing.assert_frame_equal(locs, before)
    assert len(info) == 1  # not modified either
    entry = new_info[-1]
    assert entry[spt.TRACKING_KEY] == "nearest neighbor"
    assert entry["Max. jump (nm)"] == 300.0
    assert spt.max_jump_nm(new_info) == 300.0
    assert entry["Max. gap (frames)"] == 2
    assert entry["Min. track length"] == 4
    assert entry[spt.FRAME_TIME_KEY] == FRAME_TIME
    assert spt.tracking_info(new_info) is entry
    assert tracked["group"].dtype == np.int32


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_jump": 0.0},
        {"max_jump": -1.0},
        {"max_jump": 1.0, "max_gap": -1},
        {"max_jump": 1.0, "max_gap": 1.5},
        {"max_jump": 1.0, "min_length": 0},
    ],
)
def test_link_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        spt.link_tracks(_locs([0], [0.0]), _info(), **kwargs)


def test_max_jump_nm():
    info = _info() + [
        {spt.TRACKING_KEY: spt.TRACKING_METHOD, "Max. jump (nm)": 250}
    ]
    assert spt.max_jump_nm(info) == 250.0
    assert spt.max_jump_nm(_info()) is None
    assert spt.max_jump_nm([{spt.TRACKING_KEY: spt.TRACKING_METHOD}]) is None


def test_is_tracked():
    locs = _brownian(n_particles=5)
    assert not spt.is_tracked(locs, _info())
    tracked, info = spt.link_tracks(locs, _info(), 300.0)
    assert spt.is_tracked(tracked, info)
    # group overwritten, e.g., by clustering: several locs per frame
    regrouped = tracked.assign(group=np.int32(0))
    assert not spt.is_tracked(regrouped, info)
    assert not spt.is_tracked(tracked, _info())


def test_track_segments():
    locs = _locs([0, 1, 2, 0, 1], [0.0, 1.0, 2.0, 10.0, 11.0])
    tracked, _ = spt.link_tracks(locs, _info(), 1.5)
    segments, track_id, start_row = spt.track_segments(tracked)
    assert segments.shape == (3, 4)
    assert np.all(np.diff(track_id) >= 0)
    for (x0, _, x1, _), row in zip(segments, start_row):
        assert tracked["x"].iloc[row] == x0
        assert x1 == pytest.approx(x0 + 1.0)
    assert spt.track_segments(tracked.iloc[:0])[0].shape == (0, 4)


# --------------------------------------------------------------------
# MSD
# --------------------------------------------------------------------


def test_msd_of_a_track_with_a_skipped_frame():
    """1 frame = 1 s: frames 0, 1, 3 at x = 0, 1, 3 um give one pair at
    each lag 1, 2 and 3."""
    locs = _locs([0, 1, 3], [0.0, 1000.0, 3000.0], group=np.int32(0))
    result = spt.msd(locs, frame_time=1000.0, max_lag=4)
    assert result["n_pairs"].tolist() == [1, 1, 1, 0]
    np.testing.assert_allclose(result["msd"][:3], [1.0, 4.0, 9.0])
    assert np.isnan(result["msd"][3])
    np.testing.assert_allclose(result["tau"], [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(result["sem"]).all()  # a single track


def test_ensemble_msd_is_pooled_over_pairs():
    locs = pd.concat(
        [
            _locs([0, 1, 2], [0.0, 1000.0, 3000.0], group=np.int32(0)),
            _locs([0, 1], [0.0, 2000.0], group=np.int32(1)),
        ]
    )
    result = spt.msd(locs, frame_time=1000.0, max_lag=2)
    # lag 1: (1 + 4) + 4 over 3 pairs; lag 2: 9 over 1 pair
    np.testing.assert_allclose(result["msd"], [3.0, 9.0])
    assert result["n_tracks"].tolist() == [2, 1]
    np.testing.assert_allclose(result["track_msd"][0], [2.5, 9.0])
    assert np.isnan(result["track_msd"][1, 1])


def test_msd_fit_recovers_D_and_sigma():
    locs = _brownian(n_particles=400, n_frames=40, D=0.1, sigma=30.0)
    tracked, info = spt.link_tracks(locs, _info(), 400.0)
    result = spt.msd(tracked, info, frame_time=FRAME_TIME)
    fit = spt.fit_msd(result["tau"], result["msd"], n_fit=4)
    assert fit["D"] == pytest.approx(0.1, rel=0.05)
    assert fit["sigma"] == pytest.approx(30.0, rel=0.1)


def test_fit_msd_rows_and_invalid_input():
    tau = np.array([1.0, 2.0, 3.0])
    rows = np.array([[4.0, 8.0, 12.0], [5.0, 9.0, 13.0], [4.0, np.nan, 12]])
    fit = spt.fit_msd(tau, rows, n_fit=3)
    np.testing.assert_allclose(fit["D"][:2], [1.0, 1.0])
    np.testing.assert_allclose(fit["intercept"][:2], [0.0, 1.0])
    assert np.isnan(fit["sigma"][0])  # no positive intercept
    assert fit["sigma"][1] == pytest.approx(500.0)  # sqrt(1 / 4) um
    assert np.isnan(fit["D"][2])
    with pytest.raises(ValueError):
        spt.fit_msd(tau, rows, n_fit=4)
    with pytest.raises(ValueError):
        spt.fit_msd(tau, rows, n_fit=1)


def test_msd_needs_tracks_and_frame_time():
    locs = _locs([0, 1], [0.0, 1000.0])
    with pytest.raises(ValueError, match="group"):
        spt.msd(locs, frame_time=1.0)
    locs["group"] = np.int32(0)
    with pytest.raises(ValueError, match="frame time"):
        spt.msd(locs, [{}])
    # from the metadata
    result = spt.msd(locs, [{spt.FRAME_TIME_KEY: 1e3}])
    assert result["msd"][0] == pytest.approx(1.0)


# --------------------------------------------------------------------
# Jump distances
# --------------------------------------------------------------------


def test_jump_distances_skip_gaps():
    locs = _locs([0, 1, 3, 4], [0.0, 100.0, 200.0, 400.0], group=np.int32(0))
    jumps = spt.jump_distances(locs)
    np.testing.assert_allclose(jumps, [100.0, 200.0])


def test_fit_one_population():
    jumps = _jumps(0.2, 20_000)
    fit = spt.fit_jump_distances(jumps, FRAME_TIME)
    assert fit["success"]
    assert fit["D"][0] == pytest.approx(0.2, rel=0.02)
    assert fit["fractions"].tolist() == [1.0]
    assert fit["n_jumps"] == 20_000


def test_fit_two_populations():
    jumps = np.concatenate([_jumps(0.05, 15_000), _jumps(1.0, 35_000, seed=1)])
    fit = spt.fit_jump_distances(jumps, FRAME_TIME, n_populations=2)
    np.testing.assert_allclose(fit["D"], [0.05, 1.0], rtol=0.05)
    np.testing.assert_allclose(fit["fractions"], [0.3, 0.7], atol=0.02)
    assert fit["fractions"].sum() == pytest.approx(1.0)
    one = spt.fit_jump_distances(jumps, FRAME_TIME, n_populations=1)
    three = spt.fit_jump_distances(jumps, FRAME_TIME, n_populations=3)
    assert fit["bic"] < one["bic"]
    assert fit["log_likelihood"] >= one["log_likelihood"]
    assert three["log_likelihood"] >= fit["log_likelihood"] - 1e-3


def test_fit_with_localization_error():
    """Fixed at the true error, the fit recovers D; without it, it gives
    the apparent D + sigma^2 / dt."""
    jumps = _jumps(0.1, 30_000, sigma=30.0)
    fit = spt.fit_jump_distances(jumps, FRAME_TIME, sigma=30.0)
    assert fit["D"][0] == pytest.approx(0.1, rel=0.03)
    apparent = spt.fit_jump_distances(jumps, FRAME_TIME)
    expected = 0.1 + 0.03**2 / (FRAME_TIME / 1000)
    assert apparent["D"][0] == pytest.approx(expected, rel=0.03)


def test_fit_corrects_for_the_max_jump():
    jumps = _jumps(1.0, 30_000)
    r_max = 300.0  # nm, about 1.5 times the RMS jump
    kept = jumps[jumps <= r_max]
    truncated = spt.fit_jump_distances(kept, FRAME_TIME, r_max=r_max)
    assert truncated["D"][0] == pytest.approx(1.0, rel=0.03)
    naive = spt.fit_jump_distances(kept, FRAME_TIME)
    assert naive["D"][0] < 0.85
    # jumps beyond the cut-off are left out of a truncated fit
    assert spt.fit_jump_distances(jumps, FRAME_TIME, r_max=r_max)[
        "n_jumps"
    ] == len(kept)


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("n_populations", [1, 2, 3])
def test_jump_distance_gradient(truncated, n_populations):
    r = _jumps(0.3, 500) / 1000
    r_max_2 = 0.2**2 if truncated else None
    if truncated:
        r = r[r <= 0.2]
    k = n_populations
    theta = np.concatenate(
        (np.log(np.linspace(0.05, 1.0, k)), np.linspace(-0.5, 0.5, k - 1))
    )
    args = (r**2, np.log(2 * r).sum(), k, 0.02, 4 * 0.01**2, r_max_2)
    error = check_grad(
        lambda t: spt._jd_objective(t, *args)[0],
        lambda t: spt._jd_objective(t, *args)[1],
        theta,
    )
    assert error < 1e-5


def test_fit_jump_distances_invalid_input():
    jumps = np.concatenate([[0.0, 0.0], _jumps(0.1, 100)])
    fit = spt.fit_jump_distances(jumps, FRAME_TIME)
    assert fit["n_zero_jumps"] == 2 and fit["n_jumps"] == 100
    with pytest.raises(ValueError, match="too few"):
        spt.fit_jump_distances(jumps[:10], FRAME_TIME, n_populations=2)
    with pytest.raises(ValueError):
        spt.fit_jump_distances(jumps, FRAME_TIME, n_populations=0)
    with pytest.raises(ValueError):
        spt.fit_jump_distances(jumps, FRAME_TIME, sigma=-1.0)
    with pytest.raises(ValueError):
        spt.fit_jump_distances(jumps, 0.0)


def test_jump_distance_pdf_integrates_to_one():
    fit = spt.fit_jump_distances(
        np.concatenate([_jumps(0.05, 2000), _jumps(1.0, 2000, seed=1)]),
        FRAME_TIME,
        n_populations=2,
    )
    r = np.linspace(0, 5000, 20_001)
    total, components = spt.jump_distance_pdf(r, fit)
    assert np.trapezoid(total, r) == pytest.approx(1.0, rel=1e-3)
    np.testing.assert_allclose(components.sum(axis=0), total)
    truncated = dict(fit, r_max=300.0)
    total, _ = spt.jump_distance_pdf(r, truncated)
    assert np.trapezoid(total, r) == pytest.approx(1.0, rel=1e-3)
    assert np.all(total[r > 300.0] == 0)


# --------------------------------------------------------------------
# Per-track table, analysis and saving
# --------------------------------------------------------------------


def test_track_table():
    locs = pd.concat(
        [
            _locs(
                [0, 1, 2, 3, 4, 5], 1000.0 * np.arange(6.0), group=np.int32(0)
            ),
            _locs([10, 12], [5000.0, 6000.0], group=np.int32(1)),
        ]
    )
    locs["group_input"] = np.int32(7)
    table = spt.track_table(locs, frame_time=1000.0)
    assert table["group"].tolist() == [0, 1]
    assert table["group_input"].tolist() == [7, 7]
    assert table["n_locs"].tolist() == [6, 2]
    assert table["n_frames"].tolist() == [6, 3]
    assert table["duration_s"].tolist() == [5.0, 2.0]
    assert table["x_mean"].tolist() == [2500.0, 5500.0]  # nm
    assert table["n_jumps"].tolist() == [5, 0]
    assert table["mean_jump_nm"].iloc[0] == pytest.approx(1000.0)
    assert np.isnan(table["mean_jump_nm"].iloc[1])
    # straight motion: MSD = (n um)^2, not linear, but defined
    assert np.isfinite(table["D_um2_s"].iloc[0])
    assert np.isnan(table["D_um2_s"].iloc[1])  # too short


def test_analyze_tracks_and_save(tmp_path):
    locs = _brownian(n_particles=200, n_frames=20, D=0.1, sigma=20.0)
    tracked, info = spt.link_tracks(
        locs, _info(), 400.0, frame_time=FRAME_TIME
    )
    result = spt.analyze_tracks(tracked, info, n_populations=(1, 2))
    assert result["n_tracks"] == 200
    assert result["parameters"]["Localization error"] == "none"
    assert "Localization error (nm)" not in result["parameters"]
    assert "Max. jump correction (nm)" not in result["parameters"]
    assert set(result["jd_fits"]) == {1, 2}
    assert result["msd_fit"]["D"] == pytest.approx(0.1, rel=0.15)

    paths = spt.save_analysis(str(tmp_path / "sim"), result)
    assert [os.path.basename(p) for p in paths] == [
        "sim_track_stats.csv",
        "sim_msd.csv",
        "sim_diffusion.yaml",
    ]
    table = pd.read_csv(paths[0])
    assert len(table) == 200
    msd_table = pd.read_csv(paths[1])
    assert len(msd_table) == spt.MAX_LAG
    with open(paths[2]) as f:
        summary = yaml.safe_load(f)
    assert summary["Number of tracks"] == 200
    assert summary[spt.FRAME_TIME_KEY] == FRAME_TIME
    assert (
        len(summary["Jump distance fits"]["2 populations"]["D (um^2/s)"]) == 2
    )


def test_analyze_tracks_options():
    locs = _brownian(n_particles=50, n_frames=10)
    tracked, info = spt.link_tracks(locs, _info(), 400.0)
    result = spt.analyze_tracks(
        tracked,
        info,
        frame_time=FRAME_TIME,
        sigma="precision",
        n_populations=1,
        truncate=True,
    )
    params = result["parameters"]
    assert params["Localization error"] == "median precision"
    assert params["Localization error (nm)"] == pytest.approx(25.0)
    assert params["Max. jump correction (nm)"] == pytest.approx(400.0)
    assert result["jd_fits"][1]["r_max"] == pytest.approx(400.0)
    with pytest.raises(ValueError, match="frame time"):
        spt.analyze_tracks(tracked, info)
    with pytest.raises(ValueError, match="truncate"):
        spt.analyze_tracks(
            tracked, _info(), frame_time=FRAME_TIME, truncate=True
        )
    with pytest.raises(ValueError, match="n_fit"):
        spt.analyze_tracks(
            tracked, info, frame_time=FRAME_TIME, max_lag=3, n_fit=4
        )
    # too few jumps for a fit: reported, not raised
    few, few_info = spt.link_tracks(locs.iloc[:6], _info(), 400.0)
    result = spt.analyze_tracks(
        few, few_info, frame_time=FRAME_TIME, n_populations=3
    )
    assert result["jd_fits"][3] is None
    assert "too few" in result["jd_errors"][3]


def test_plots():
    locs = _brownian(n_particles=50, n_frames=10)
    tracked, info = spt.link_tracks(locs, _info(), 400.0)
    result = spt.analyze_tracks(tracked, info, frame_time=FRAME_TIME)
    fig = spt.plot_msd(result["msd"], result["msd_fit"])
    assert "D =" in fig.axes[0].get_title()
    fig = spt.plot_jump_distances(result["jumps"], result["jd_fits"][2])
    assert len(fig.axes[0].lines) == 3  # total and two populations
    reused = spt.plot_jump_distances(result["jumps"], None, fig=fig)
    assert reused is fig and len(fig.axes) == 1
    spt.plot_track_lengths(result["table"])
    spt.plot_track_lengths(result["table"].iloc[:0])
    plt.close("all")


# --------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------


def test_cli_track_and_diffusion(tmp_path, capsys):
    path = str(tmp_path / "sim.hdf5")
    locs = _brownian(n_particles=100, n_frames=15).drop(columns="truth")
    io.save_locs(path, locs, _info())
    cli._track([path], 400.0, 1, 2, FRAME_TIME)
    out = str(tmp_path / "sim_tracks.hdf5")
    tracked, info = io.load_locs(out)
    assert spt.is_tracked(tracked, info)
    assert info[-1]["Max. jump (nm)"] == 400.0
    assert tracked["group"].nunique() == 100

    cli._diffusion([out], None, 10, 4, [1, 2], 0.0, False, True)
    for suffix in (
        "_track_stats.csv",
        "_msd.csv",
        "_diffusion.yaml",
        "_msd.png",
        "_jump_distances.png",
        "_track_lengths.png",
    ):
        assert os.path.exists(str(tmp_path / f"sim_tracks{suffix}"))
    assert "2 populations" in capsys.readouterr().out

    # untracked files are skipped
    cli._diffusion([path], FRAME_TIME, 10, 4, [1], 0.0, False, False)
    assert "not tracked" in capsys.readouterr().out


def test_cli_sigma_argument():
    assert cli._spt_sigma("lp") == "precision"
    assert cli._spt_sigma("12.5") == 12.5
    with pytest.raises(argparse.ArgumentTypeError):
        cli._spt_sigma("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._spt_sigma("abc")
