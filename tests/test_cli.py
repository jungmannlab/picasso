"""Tests of the ``picasso`` commands that have no test file of their own:
each runs the command line as a user would (``picasso <command> ...``
through ``__main__.main``) and checks the files it writes. The analysis
behind a command is tested with its module; these tests cover the
argument parsing, the wiring to the analysis and the output files.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

from picasso import __main__ as cli
from picasso import io, transforms


def _picasso(monkeypatch, *args) -> None:
    """Run ``picasso <args>`` as from the command line."""
    monkeypatch.setattr(sys, "argv", ["picasso", *map(str, args)])
    cli.main()


@pytest.fixture
def locs_path(locs, info, tmp_path):
    """The bundled localizations (nine origami) saved in ``tmp_path``."""
    path = tmp_path / "locs.hdf5"
    io.save_locs(str(path), locs, info)
    return path


def _out(path, suffix: str) -> str:
    """``path`` with its extension replaced by ``suffix``."""
    return os.path.splitext(str(path))[0] + suffix


# ---------------------------------------------------------------------------
# File conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command, suffix",
    [
        ("hdf2csv", ".csv"),
        ("hdf2ts", ".csv"),
        ("hdf2imagej", ".txt"),
        ("hdf2nis", ".nis.txt"),
        ("hdf2smap", "_sml.mat"),
    ],
)
def test_export_writes_one_file_per_input(
    command, suffix, locs_path, monkeypatch
):
    _picasso(monkeypatch, command, locs_path)
    assert os.path.getsize(_out(locs_path, suffix)) > 0


@pytest.mark.parametrize(
    "command, suffix", [("hdf2chimera", ".chi.xyz"), ("hdf2visp", ".visp.3d")]
)
def test_3d_export(command, suffix, locs, info, tmp_path, monkeypatch):
    path = tmp_path / "locs3d.hdf5"
    rng = np.random.default_rng(0)
    locs3d = locs.assign(z=rng.uniform(-200, 200, len(locs)).astype("f4"))
    io.save_locs(str(path), locs3d, info)
    _picasso(monkeypatch, command, path)
    with open(_out(path, suffix)) as f:
        assert len(f.read().splitlines()) >= len(locs)


@pytest.mark.parametrize("command", ["hdf2chimera", "hdf2visp"])
def test_3d_export_of_2d_localizations_warns(command, locs_path, monkeypatch):
    with pytest.warns(UserWarning, match="No z coordinate"):
        _picasso(monkeypatch, command, locs_path)


def test_export_of_a_folder_converts_every_file(
    locs, info, tmp_path, monkeypatch
):
    for name in ("a", "b"):
        io.save_locs(str(tmp_path / f"{name}.hdf5"), locs, info)
    _picasso(monkeypatch, "hdf2csv", tmp_path)
    for name in ("a", "b"):
        assert len(pd.read_csv(tmp_path / f"{name}.csv")) == len(locs)


def test_hdf2csv_keeps_the_columns(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "hdf2csv", locs_path)
    exported = pd.read_csv(_out(locs_path, ".csv"))
    assert list(exported.columns) == list(locs.columns)  # no index column
    assert np.allclose(exported["x"], locs["x"], rtol=1e-6)
    assert np.allclose(exported["photons"], locs["photons"], rtol=1e-6)


def test_thunderstorm_round_trip(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "hdf2ts", locs_path)
    _picasso(monkeypatch, "csv2hdf", _out(locs_path, ".csv"), "-p", 160)
    imported, info = io.load_locs(_out(locs_path, "_locs.hdf5"))
    assert len(imported) == len(locs)
    for column in ("x", "y"):
        assert np.allclose(
            np.sort(imported[column]), np.sort(locs[column]), atol=0.01
        )
    assert info[0]["Pixelsize"] == 160


def test_smap_round_trip(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "hdf2smap", locs_path)
    _picasso(monkeypatch, "smap2hdf", _out(locs_path, "_sml.mat"), "-p", 160)
    imported, _ = io.load_locs(_out(locs_path, "_sml_locs.hdf5"))
    assert len(imported) == len(locs)
    assert np.allclose(np.sort(imported["x"]), np.sort(locs["x"]), atol=0.01)


# ---------------------------------------------------------------------------
# Postprocessing of single files
# ---------------------------------------------------------------------------


def test_link_merges_consecutive_localizations(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "link", locs_path, "-d", 50, "-t", 2)
    linked, info = io.load_locs(_out(locs_path, "_link.hdf5"))
    assert 0 < len(linked) < len(locs)
    assert linked["len"].sum() >= len(linked)
    assert info[-1]["Maximum Distance"] == 50
    assert info[-1]["Maximum Transient Dark Time"] == 2


def test_dark_times_of_linked_localizations(locs_path, monkeypatch):
    _picasso(monkeypatch, "link", locs_path)
    linked_path = _out(locs_path, "_link.hdf5")
    _picasso(monkeypatch, "dark", linked_path)
    dark, info = io.load_locs(_out(linked_path, "_dark.hdf5"))
    assert "dark" in dark.columns
    assert info[-1]["Generated by"].endswith("Dark")


def test_density_adds_a_column(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "density", locs_path, 50)
    dense, info = io.load_locs(_out(locs_path, "_density.hdf5"))
    assert len(dense) == len(locs)
    assert (dense["density"] >= 0).all()
    assert info[-1]["Radius (nm)"] == 50


def test_nearest_neighbor_saves_one_distance_per_localization(
    locs, locs_path, monkeypatch
):
    _picasso(monkeypatch, "nearest-neighbor", locs_path)
    distances = np.loadtxt(_out(locs_path, "_minval.txt"))
    assert len(distances) == len(locs)
    assert (distances >= 0).all()


@pytest.mark.parametrize("keep_index", [False, True])
def test_join_concatenates_files(
    keep_index, locs, info, tmp_path, monkeypatch
):
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.hdf5"
        io.save_locs(str(path), locs, info)
        paths.append(path)
    _picasso(monkeypatch, "join", *paths, *(["-k"] if keep_index else []))
    joined, joined_info = io.load_locs(_out(paths[0], "_join.hdf5"))
    assert len(joined) == 2 * len(locs)
    n_frames = info[0]["Frames"]
    if keep_index:
        assert joined["frame"].max() < n_frames
    else:
        # the second file follows the first in time
        assert joined["frame"].max() >= n_frames
        assert joined_info[0]["Frames"] == joined["frame"].max() + 1


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


N_ORIGAMI = 9  # in the bundled localizations


def test_dbscan_finds_the_origami(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "dbscan", locs_path, 50, 5)
    clustered, info = io.load_locs(_out(locs_path, "_dbscan.hdf5"))
    centers, _ = io.load_locs(_out(locs_path, "_dbclusters.hdf5"))
    assert "group" in clustered.columns
    assert len(centers) == N_ORIGAMI
    assert "DBSCAN" in info[-1]["Generated by"]


def test_hdbscan_finds_the_origami(locs_path, monkeypatch):
    _picasso(monkeypatch, "hdbscan", locs_path, 30, 5)
    clustered, _ = io.load_locs(_out(locs_path, "_hdbscan.hdf5"))
    centers, _ = io.load_locs(_out(locs_path, "_hdbclusters.hdf5"))
    assert "group" in clustered.columns
    assert len(centers) == N_ORIGAMI


def test_smlm_cluster_finds_the_origami(locs_path, monkeypatch):
    _picasso(monkeypatch, "smlm-cluster", locs_path, 50, 10)
    clustered, _ = io.load_locs(_out(locs_path, "_clusters.hdf5"))
    centers, _ = io.load_locs(_out(locs_path, "_cluster_centers.hdf5"))
    assert "group" in clustered.columns
    assert len(centers) == N_ORIGAMI


def test_smlm_cluster_with_every_argument(locs_path, monkeypatch):
    _picasso(monkeypatch, "smlm-cluster", locs_path, 50, 10, "yes", 100)
    centers, _ = io.load_locs(_out(locs_path, "_cluster_centers.hdf5"))
    assert len(centers) == N_ORIGAMI


@pytest.mark.parametrize(
    "value, expected",
    [
        ("True", True),
        ("yes", True),
        ("1", True),
        ("False", False),
        ("no", False),
        ("0", False),
    ],
)
def test_smlm_cluster_reads_the_frame_analysis_switch(
    value, expected, locs_path, monkeypatch
):
    recorded = {}

    def cluster(locs, **params):
        recorded.update(params)
        raise SystemExit  # the arguments are all that is checked

    monkeypatch.setattr("picasso.clusterer.cluster", cluster)
    with pytest.raises(SystemExit):
        _picasso(monkeypatch, "smlm-cluster", locs_path, 50, 10, value, 100)
    assert recorded["frame_analysis"] is expected


def test_smlm_cluster_rejects_an_unclear_switch(
    locs_path, monkeypatch, capsys
):
    with pytest.raises(SystemExit):
        _picasso(monkeypatch, "smlm-cluster", locs_path, 50, 10, "maybe")
    assert "expected yes or no" in capsys.readouterr().err


@pytest.fixture
def dbscan_paths(locs_path, monkeypatch):
    """``(clustered localizations, cluster centers)`` written by DBSCAN."""
    _picasso(monkeypatch, "dbscan", locs_path, 50, 5)
    return _out(locs_path, "_dbscan.hdf5"), _out(locs_path, "_dbclusters.hdf5")


def test_cluster_filter_splits_by_a_cluster_property(
    dbscan_paths, monkeypatch
):
    clustered_path, centers_path = dbscan_paths
    centers, _ = io.load_locs(centers_path)
    threshold = float(np.median(centers["n_locs"]))
    _picasso(
        monkeypatch,
        "cluster-filter",
        clustered_path,
        "-c",
        centers_path,
        "-p",
        "n_locs",
        "--min-value",
        threshold,
        "--max-value",
        1e9,
    )
    inside, info = io.load_locs(_out(clustered_path, "_filter_in.hdf5"))
    outside, _ = io.load_locs(_out(clustered_path, "_filter_out.hdf5"))
    clustered, _ = io.load_locs(clustered_path)
    assert len(inside) + len(outside) == len(clustered)
    kept = centers["group"][centers["n_locs"] > threshold]
    assert set(inside["group"]) == set(kept)
    assert info[-1]["Parameter"] == "n_locs"


def test_cluster_filter_reports_an_unknown_property(
    dbscan_paths, monkeypatch, capsys
):
    clustered_path, centers_path = dbscan_paths
    _picasso(
        monkeypatch,
        "cluster-filter",
        clustered_path,
        "-c",
        centers_path,
        "-p",
        "no_such_column",
        "--min-value",
        0,
        "--max-value",
        1,
    )
    assert "no_such_column not found" in capsys.readouterr().out
    assert not os.path.exists(_out(clustered_path, "_filter_in.hdf5"))


def test_group_properties_saves_one_row_per_group(dbscan_paths, monkeypatch):
    clustered_path, _ = dbscan_paths
    _picasso(monkeypatch, "group-properties", clustered_path)
    with h5py.File(_out(clustered_path, "_groupprops.hdf5"), "r") as f:
        groups = f["groups"][...]
        n_locs = len(f["locs"])
    clustered, _ = io.load_locs(clustered_path)
    assert n_locs == len(clustered)
    assert len(groups) == clustered["group"].nunique()


# ---------------------------------------------------------------------------
# Drift correction
# ---------------------------------------------------------------------------


def test_undrift_rcc_writes_the_drift(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "undrift", locs_path, "-s", 200)
    undrifted, info = io.load_locs(_out(locs_path, "_undrift.hdf5"))
    drift = io.load_drift(_out(locs_path, "_drift.txt"))
    assert len(undrifted) == len(locs)
    assert len(drift) == info[0]["Frames"]
    assert info[-1]["Segmentation"] == 200


def test_undrift_from_a_drift_file(
    locs, info, locs_path, tmp_path, monkeypatch
):
    n_frames = info[0]["Frames"]
    drift = pd.DataFrame(
        {"x": np.full(n_frames, 5.0), "y": np.full(n_frames, -3.0)}
    )
    drift_path = str(tmp_path / "drift.txt")
    io.save_drift(drift_path, drift)
    _picasso(monkeypatch, "undrift", locs_path, "-f", drift_path)
    undrifted, undrift_info = io.load_locs(_out(locs_path, "_undrift.hdf5"))
    assert np.allclose(undrifted["x"], locs["x"] - 5.0, atol=1e-3)
    assert np.allclose(undrifted["y"], locs["y"] + 3.0, atol=1e-3)
    assert undrift_info[-1]["From File"] == drift_path


def test_aim_writes_the_drift(locs, locs_path, monkeypatch):
    _picasso(monkeypatch, "aim", locs_path, "-s", 100)
    undrifted, info = io.load_locs(_out(locs_path, "_aim.hdf5"))
    drift = io.load_drift(_out(locs_path, "_aimdrift.txt"))
    assert len(undrifted) == len(locs)
    assert len(drift) == info[0]["Frames"]


def _fiducials(n_frames: int, drift_x: float) -> pd.DataFrame:
    """Four fiducials, on in every frame, that move by ``drift_x`` nm in
    x over the movie (with 3 nm localization noise)."""
    rng = np.random.default_rng(1)
    frames = np.tile(np.arange(n_frames), 4)
    centers = np.repeat(
        [[1000, 1000], [3000, 1200], [1500, 3000], [3000, 3000]],
        n_frames,
        axis=0,
    )
    n = len(frames)
    x = centers[:, 0] + drift_x * frames / n_frames + rng.normal(0, 3, n)
    y = centers[:, 1] + rng.normal(0, 3, n)
    return pd.DataFrame(
        {
            "frame": frames.astype("u4"),
            "x": x.astype("f4"),
            "y": y.astype("f4"),
            "photons": np.full(n, 5000, "f4"),
            "sx": np.full(n, 130, "f4"),
            "sy": np.full(n, 130, "f4"),
            "bg": np.full(n, 10, "f4"),
            "lpx": np.full(n, 3, "f4"),
            "lpy": np.full(n, 3, "f4"),
        }
    )


def test_undrift_fiducials_removes_a_linear_drift(info, tmp_path, monkeypatch):
    # the command runs RCC on segments of 2000 frames first, which needs
    # at least 4 of them
    n_frames = 8000
    info = [{**info[0], "Frames": n_frames}, *info[1:]]
    path = tmp_path / "fiducials.hdf5"
    io.save_locs(str(path), _fiducials(n_frames, drift_x=40.0), info)
    _picasso(monkeypatch, "undrift-fiducials", path)
    undrifted, undrift_info = io.load_locs(
        _out(path, "_undrift_fiducials.hdf5")
    )
    assert os.path.exists(_out(path, "_drift_fiducials.txt"))
    assert undrift_info[-1]["Number of picks"] == 4
    # the spread of a fiducial in x shrinks from ~12 nm to the noise
    near = (np.abs(undrifted["x"] - 1000) < 200) & (
        np.abs(undrifted["y"] - 1000) < 200
    )
    assert undrifted.loc[near, "x"].std() < 5


def test_undrift_fiducials_of_a_short_movie(info, tmp_path, monkeypatch):
    """Segments of 2000 frames do not fit 4 times into the 1000 frames:
    the RCC before finding the fiducials uses segments of 250."""
    path = tmp_path / "fiducials.hdf5"
    io.save_locs(str(path), _fiducials(info[0]["Frames"], 40.0), info)
    _picasso(monkeypatch, "undrift-fiducials", path)
    undrifted, undrift_info = io.load_locs(
        _out(path, "_undrift_fiducials.hdf5")
    )
    assert undrift_info[-1]["pre-RCC segmentation"] == 250
    near = (np.abs(undrifted["x"] - 1000) < 200) & (
        np.abs(undrifted["y"] - 1000) < 200
    )
    assert undrifted.loc[near, "x"].std() < 5


@pytest.mark.parametrize("segmentation", [400, 1000])
def test_undrift_with_too_few_segments_says_why(
    segmentation, locs_path, monkeypatch
):
    with pytest.raises(SystemExit, match="at most 250 frames"):
        _picasso(monkeypatch, "undrift", locs_path, "-s", segmentation)
    assert not os.path.exists(_out(locs_path, "_undrift.hdf5"))


# ---------------------------------------------------------------------------
# Spatial statistics
# ---------------------------------------------------------------------------


@pytest.fixture
def two_channels(locs, info, tmp_path):
    """Two channels: a copy of the localizations ("same", fully
    colocalized) and a copy shifted by 1 um ("far", not colocalized)."""
    paths = {}
    for name, shift in (("same", 0.0), ("far", 1000.0)):
        path = tmp_path / f"{name}.hdf5"
        io.save_locs(str(path), locs.assign(x=locs["x"] + shift), info)
        paths[name] = path
    return paths


def test_cbc_scores_the_first_file(locs_path, two_channels, monkeypatch):
    _picasso(
        monkeypatch,
        "cbc",
        locs_path,
        two_channels["same"],
        two_channels["far"],
        "-r",
        100,
        "-s",
        10,
    )
    scored, info = io.load_locs(_out(locs_path, "_cbc.hdf5"))
    assert scored["cbc_same"].median() > 0.9
    assert scored["cbc_far"].abs().max() == 0.0  # no neighbors: undefined
    assert info[-1]["Scored channel"] == str(locs_path)
    assert not os.path.exists(_out(two_channels["same"], "_cbc.hdf5"))


def test_cbc_of_all_pairs_scores_every_file(
    locs_path, two_channels, monkeypatch
):
    _picasso(
        monkeypatch,
        "cbc",
        locs_path,
        two_channels["same"],
        "-a",
        "-n",
        "first",
        "second",
        "-u",
        "remove",
    )
    first, _ = io.load_locs(_out(locs_path, "_cbc.hdf5"))
    second, _ = io.load_locs(_out(two_channels["same"], "_cbc.hdf5"))
    assert "cbc_second" in first.columns
    assert "cbc_first" in second.columns


@pytest.mark.parametrize(
    "extra, message",
    [
        ([], "at least two localization files"),
        (["{same}", "-n", "only_one"], "1 names given for 2 files"),
        (["{same}", "-n", "a.b", "a_b"], "same CBC column"),
    ],
)
def test_cbc_rejects_bad_arguments(
    extra, message, locs_path, two_channels, monkeypatch
):
    extra = [e.format(same=two_channels["same"]) for e in extra]
    with pytest.raises(SystemExit, match=message):
        _picasso(monkeypatch, "cbc", locs_path, *extra)


def test_cbc_rejects_an_undefined_value_out_of_range(locs_path, monkeypatch):
    with pytest.raises(SystemExit):
        _picasso(monkeypatch, "cbc", locs_path, locs_path, "-u", "2")


def test_nnd_saves_tables_and_a_plot(locs_path, two_channels, monkeypatch):
    _picasso(
        monkeypatch,
        "nnd",
        locs_path,
        two_channels["far"],
        "-n",
        2,
        "-s",
        10,
    )
    for suffix in ("_nnd.csv", "_nnd_summary.csv", "_nnd.png"):
        assert os.path.getsize(_out(locs_path, suffix)) > 0
    summary = pd.read_csv(_out(locs_path, "_nnd_summary.csv"))
    assert len(summary) == 4  # every ordered pair of the two channels


def test_nnd_requires_files(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="No localization files"):
        _picasso(monkeypatch, "nnd", tmp_path / "missing*.hdf5")


@pytest.mark.parametrize("command", ["rdf", "ripley-h"])
def test_pair_statistic_saves_tables_and_a_plot(
    command, locs_path, monkeypatch
):
    _picasso(monkeypatch, command, locs_path, "-n", 2, "-s", 10)
    base = _out(locs_path, "_" + command.replace("-", "_"))
    curves = pd.read_csv(base + ".csv")
    assert len(curves) == 10
    assert os.path.getsize(base + "_regions.csv") > 0
    assert os.path.getsize(base + ".png") > 0


def test_cross_pair_statistic(locs_path, two_channels, monkeypatch):
    _picasso(
        monkeypatch,
        "rdf",
        locs_path,
        "--partner",
        two_channels["same"],
        "-n",
        2,
        "-s",
        10,
    )
    assert os.path.exists(_out(locs_path, "_rdf_cross.csv"))


def test_pair_statistic_requires_files(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="No localization files"):
        _picasso(monkeypatch, "rdf", tmp_path / "missing*.hdf5")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_saves_a_png(locs_path, monkeypatch):
    import matplotlib.image

    _picasso(monkeypatch, "render", locs_path, "-px", 20, "-c", "magma")
    image = matplotlib.image.imread(_out(locs_path, ".png"))
    assert image.shape[:2] == (208, 208)  # 32 camera pixels of 130 nm
    # the colormap is remembered for the next render
    assert io.load_user_settings()["Render"]["Colormap"] == "magma"


@pytest.mark.parametrize("blur", ["none", "gaussian"])
def test_render_options(blur, locs_path, monkeypatch):
    _picasso(
        monkeypatch,
        "render",
        locs_path,
        "-b",
        blur,
        "--scaling",
        "no",
        "--vmax",
        1,
    )
    assert os.path.exists(_out(locs_path, ".png"))


def test_render_uses_the_colormap_of_the_settings(locs_path, monkeypatch):
    settings = io.load_user_settings()
    settings["Render"]["Colormap"] = "gray"
    io.save_user_settings(settings)
    _picasso(monkeypatch, "render", locs_path)
    import matplotlib.image

    image = matplotlib.image.imread(_out(locs_path, ".png"))
    assert np.allclose(image[..., 0], image[..., 1])  # gray


def test_render_a_folder(locs, info, tmp_path, monkeypatch):
    for name in ("a", "b"):
        io.save_locs(str(tmp_path / f"{name}.hdf5"), locs, info)
    _picasso(monkeypatch, "render", tmp_path)
    assert os.path.exists(tmp_path / "a.png")
    assert os.path.exists(tmp_path / "b.png")


# ---------------------------------------------------------------------------
# Unit conversion of files written before Picasso 1.0
# ---------------------------------------------------------------------------


def test_convert_units(tmp_path, monkeypatch, capsys):
    from tests.test_legacy import PS, _legacy_locs, _write_legacy

    old = _legacy_locs()
    path = _write_legacy(tmp_path / "old_locs.hdf5", old)
    current = tmp_path / "current.hdf5"
    current.write_text("")
    _picasso(monkeypatch, "convert-units", path, current)
    out = capsys.readouterr().out
    converted, _ = io.load_locs(str(tmp_path / "old_locs_nm.hdf5"))
    assert np.allclose(converted["x"], old["x"] * PS, rtol=1e-6)
    assert "Converted 1 of 2 files." in out
    assert f"{current}: skipped" in out


def test_pair_correlation_plots_the_function(locs_path, monkeypatch):
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gca()))
    _picasso(monkeypatch, "pair-correlation", locs_path, "-b", 10, "-r", 200)
    (axes,) = shown
    (line,) = axes.get_lines()
    assert len(line.get_xdata()) == 20  # 200 nm in bins of 10 nm
    plt.close("all")


# ---------------------------------------------------------------------------
# Calibrations
# ---------------------------------------------------------------------------


def _save_movie(path, movie, **extra) -> str:
    """Save ``movie`` as a .raw movie with its .yaml."""
    info = {
        "Byte Order": "<",
        "Data Type": "uint16",
        "Frames": int(movie.shape[0]),
        "Height": int(movie.shape[1]),
        "Width": int(movie.shape[2]),
        **extra,
    }
    io.save_raw(str(path), np.asarray(movie).astype(np.uint16), [info])
    return str(path)


@pytest.fixture(scope="module")
def camera_maps():
    from tests.conftest import _make_scmos_maps

    return _make_scmos_maps(height=16, width=16, seed=3)


def test_camera_calibrate_and_validate(
    camera_maps, dark_movie_factory, tmp_path, monkeypatch, capsys
):
    from picasso import scmos

    dark = _save_movie(
        tmp_path / "dark.raw", dark_movie_factory(camera_maps, 5000, seed=4)
    )
    lights = [
        _save_movie(
            tmp_path / f"light{photons}.raw",
            dark_movie_factory(
                camera_maps, 500, photons=photons, seed=photons
            ),
        )
        for photons in (50, 200)
    ]
    _picasso(
        monkeypatch,
        "camera-calibrate",
        dark,
        "-l",
        lights[0],
        "-l",
        lights[1],
        "-p",
        1,
        "-p",
        4,
    )
    calibration_path = _out(dark, "_scmos_calib.hdf5")
    calibration = io.load_camera_calibration(calibration_path)
    assert calibration["Gain levels"] == 2
    assert np.median(calibration["gain"]) == pytest.approx(2.13, rel=0.1)
    assert os.path.exists(scmos.plot_path(calibration_path))
    assert "Use it with: picasso localize" in capsys.readouterr().out

    fresh = _save_movie(
        tmp_path / "fresh.raw",
        dark_movie_factory(camera_maps, scmos.VALIDATION_FRAMES, seed=5),
    )
    _picasso(monkeypatch, "camera-validate", calibration_path, fresh)
    assert "still describes this camera" in capsys.readouterr().out


def test_camera_calibrate_without_light_movies(
    camera_maps, dark_movie_factory, tmp_path, monkeypatch, capsys
):
    dark = _save_movie(
        tmp_path / "dark.raw", dark_movie_factory(camera_maps, 2000, seed=6)
    )
    output = str(tmp_path / "camera.hdf5")
    _picasso(monkeypatch, "camera-calibrate", dark, "-o", output)
    assert "no gain map" in capsys.readouterr().out
    assert io.load_camera_calibration(output).get("gain") is None


def test_camera_calibrate_needs_one_power_per_light_movie(
    camera_maps, dark_movie_factory, tmp_path, monkeypatch
):
    dark = _save_movie(
        tmp_path / "dark.raw", dark_movie_factory(camera_maps, 200, seed=7)
    )
    light = _save_movie(
        tmp_path / "light.raw",
        dark_movie_factory(camera_maps, 200, photons=50, seed=8),
    )
    with pytest.raises(ValueError, match="one per movie"):
        _picasso(
            monkeypatch,
            "camera-calibrate",
            dark,
            "-l",
            light,
            "-p",
            1,
            "-p",
            2,
        )


def test_camera_validate_flags_a_changed_camera(
    camera_maps, dark_movie_factory, tmp_path, monkeypatch, capsys
):
    from picasso import scmos

    dark = _save_movie(
        tmp_path / "dark.raw", dark_movie_factory(camera_maps, 5000, seed=9)
    )
    _picasso(monkeypatch, "camera-calibrate", dark)
    noisier = dict(camera_maps, variance=camera_maps["variance"] * 1.6)
    fresh = _save_movie(
        tmp_path / "fresh.raw",
        dark_movie_factory(noisier, scmos.VALIDATION_FRAMES, seed=10),
    )
    _picasso(
        monkeypatch,
        "camera-validate",
        _out(dark, "_scmos_calib.hdf5"),
        fresh,
    )
    assert "looks noisier" in capsys.readouterr().out


def test_spline_calibrate(tmp_path, monkeypatch, capsys):
    from picasso import spline
    from tests.test_spline import _synthetic_bead_movie

    movie, _, _ = _synthetic_bead_movie()
    path = _save_movie(tmp_path / "beads.raw", movie)
    _picasso(
        monkeypatch,
        "spline-calibrate",
        path,
        "-s",
        20,
        "-g",
        2000,
        "-b",
        7,
        "-px",
        130,
    )
    calibration_path = _out(path, "_spline_calib.hdf5")
    calibration = io.load_any_calibration(calibration_path)
    assert spline.n_beads_used(calibration) == 3
    out = capsys.readouterr().out
    assert "Camera pixel size: 130 nm" in out
    assert f"saved to {calibration_path}" in out


def test_spline_calibrate_of_two_channels(tmp_path, monkeypatch):
    from tests.test_spline import _synthetic_bead_movie

    beads = [(12, 14), (30, 28), (16, 33), (36, 12)]
    paths = []
    for channel, shift in enumerate((0.0, 1.5)):
        movie, _, _ = _synthetic_bead_movie(
            bead_xy=[(x + shift, y) for x, y in beads]
        )
        paths.append(_save_movie(tmp_path / f"channel{channel}.raw", movie))
    output = str(tmp_path / "two_channels.hdf5")
    _picasso(
        monkeypatch,
        "spline-calibrate",
        *paths,
        "-s",
        20,
        "-g",
        2000,
        "-b",
        7,
        "-px",
        130,
        "-rm",
        "translation",
        "-o",
        output,
    )
    assert os.path.exists(output)


def test_spline_calibrate_rejects_a_bad_split_fov(tmp_path, monkeypatch):
    from tests.test_spline import _synthetic_bead_movie

    movie, _, _ = _synthetic_bead_movie()
    path = _save_movie(tmp_path / "beads.raw", movie)
    with pytest.raises(ValueError, match="4 ints"):
        _picasso(
            monkeypatch,
            "spline-calibrate",
            path,
            "-s",
            20,
            "-px",
            130,
            "-sf",
            "0,0,48;0,24,48,48",
        )


def test_lateral_calibrate(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from picasso import lib
    from tests.test_localize import (
        _affine_bead_grid,
        _affine_bead_image,
    )

    monkeypatch.setattr(plt, "show", lambda: None)
    beads = _affine_bead_grid()
    reference = _save_movie(tmp_path / "ref.raw", _affine_bead_image(beads))
    target = _save_movie(
        tmp_path / "target.raw", _affine_bead_image(beads + [2.0, -1.0])
    )
    output = str(tmp_path / "lateral.yaml")
    plot = str(tmp_path / "lateral.png")
    _picasso(
        monkeypatch,
        "lateral-calibrate",
        reference,
        target,
        "-t",
        "chromatic",
        "-m",
        "translation",
        "-g",
        1000,
        "-px",
        130,
        "-o",
        output,
        "-p",
        plot,
    )
    (entry,) = lib.lateral_transforms(io.load_any_calibration(output))
    assert entry["Type"] == "chromatic"
    # the target is shifted by (2, -1) camera pixels
    shift = transforms.from_dict(entry["Transform"]).apply([[0.0, 0.0]])[0]
    assert shift == pytest.approx([-2.0 * 130, 1.0 * 130], abs=5)
    assert os.path.getsize(plot) > 0
    plt.close("all")


def test_lateral_calibrate_needs_an_output(tmp_path, monkeypatch):
    from tests.test_localize import _affine_bead_grid, _affine_bead_image

    movie = _save_movie(
        tmp_path / "ref.raw", _affine_bead_image(_affine_bead_grid())
    )
    with pytest.raises(Exception, match="Pass --calibration"):
        _picasso(monkeypatch, "lateral-calibrate", movie, movie, "-px", 130)


# ---------------------------------------------------------------------------
# picasso plugins
# ---------------------------------------------------------------------------


GOOD_PLUGIN = """
def _run(args):
    print(f"demo got {args.value}")


def register_cli(subparsers):
    parser = subparsers.add_parser("demo-cmd")
    parser.add_argument("value")
    parser.set_defaults(func=_run)
"""


@pytest.fixture
def plugins_dir(tmp_path, monkeypatch):
    """An empty plugins folder in ``tmp_path``."""
    from picasso import plugins

    directory = tmp_path / "plugins"
    directory.mkdir()
    monkeypatch.setattr(io, "plugins_directory", lambda: str(directory))
    plugins.clear_module_cache()
    yield directory
    plugins.clear_module_cache()


def _plugin_file(directory, name: str, source: str, enabled: bool = False):
    from picasso import plugins

    (directory / name).write_text(source)
    if enabled:
        plugins.set_enabled(plugins.load_state(), name, True)


def _enabled(name: str) -> bool:
    from picasso import plugins

    return plugins.is_enabled(plugins.load_state(), name)


def test_plugins_path_and_usage(plugins_dir, monkeypatch, capsys):
    _picasso(monkeypatch, "plugins", "path")
    assert capsys.readouterr().out.strip() == str(plugins_dir)
    _picasso(monkeypatch, "plugins")
    assert "Usage: picasso plugins" in capsys.readouterr().out


def test_plugins_list_of_an_empty_folder(plugins_dir, monkeypatch, capsys):
    _picasso(monkeypatch, "plugins", "list")
    assert "No plugin files in" in capsys.readouterr().out


def test_plugins_list_describes_each_file(plugins_dir, monkeypatch, capsys):
    _plugin_file(plugins_dir, "good.py", GOOD_PLUGIN, enabled=True)
    _plugin_file(
        plugins_dir, "broken.py", "raise RuntimeError('boom')\n", True
    )
    _plugin_file(plugins_dir, "off.py", GOOD_PLUGIN)
    _picasso(monkeypatch, "plugins", "list")
    out = capsys.readouterr().out
    assert "[x] good.py  (local file)" in out
    assert "commands: demo-cmd" in out
    assert "[ ] off.py" in out
    assert "1 file(s) not enabled" in out
    assert "[x] broken.py" in out
    assert "failed to load" in out and "boom" in out


def test_plugin_command_runs_its_handler(plugins_dir, monkeypatch, capsys):
    _plugin_file(plugins_dir, "good.py", GOOD_PLUGIN, enabled=True)
    _picasso(monkeypatch, "demo-cmd", "hello")
    assert "demo got hello" in capsys.readouterr().out


def test_plugin_command_without_a_handler(plugins_dir, monkeypatch, capsys):
    source = GOOD_PLUGIN.replace("    parser.set_defaults(func=_run)\n", "")
    _plugin_file(plugins_dir, "lazy.py", source, enabled=True)
    _picasso(monkeypatch, "demo-cmd", "hello")
    assert "does not name a handler" in capsys.readouterr().out


@pytest.mark.parametrize(
    "answer, enabled", [("y", True), ("n", False), (EOFError, False)]
)
def test_plugins_enable_asks_first(
    answer, enabled, plugins_dir, monkeypatch, capsys
):
    def ask(prompt):
        if answer is EOFError:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", ask)
    _plugin_file(plugins_dir, "good.py", GOOD_PLUGIN)
    _picasso(monkeypatch, "plugins", "enable", "good.py")
    assert _enabled("good.py") is enabled
    assert "full access to your computer" in capsys.readouterr().out


def test_plugins_enable_and_disable_with_yes(plugins_dir, monkeypatch):
    def ask(prompt):
        raise AssertionError("--yes must not ask")

    monkeypatch.setattr("builtins.input", ask)
    _plugin_file(plugins_dir, "good.py", GOOD_PLUGIN)
    _picasso(monkeypatch, "plugins", "enable", "good.py", "--yes")
    assert _enabled("good.py")
    _picasso(monkeypatch, "plugins", "disable", "good.py")
    assert not _enabled("good.py")


@pytest.mark.parametrize(
    "name, message",
    [
        ("../evil.py", "Not a plugin file name"),
        ("missing.py", "No such plugin"),
    ],
)
def test_plugins_enable_rejects_bad_files(
    name, message, plugins_dir, monkeypatch, capsys
):
    _picasso(monkeypatch, "plugins", "enable", name, "--yes")
    assert message in capsys.readouterr().out


@pytest.fixture
def registry(plugins_dir, monkeypatch):
    """A fake online registry with one plugin, ``demo``; returns the
    manifest entries, which a test may change."""
    import hashlib
    import json

    from picasso import plugins

    source = GOOD_PLUGIN.encode()
    entries = [
        {
            "id": "demo",
            "file": "plugins/demo.py",
            "version": "1.2.0",
            "display_name": "Demo",
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    ]

    def get(url):
        if url == plugins.MANIFEST_URL:
            return json.dumps({"plugins": entries}).encode()
        if url == f"{plugins.RAW_BASE}/plugins/demo.py":
            return source
        raise AssertionError(f"unexpected download: {url}")

    monkeypatch.setattr(plugins, "_get", get)
    return entries


def test_plugins_install_and_uninstall(
    registry, plugins_dir, monkeypatch, capsys
):
    _picasso(monkeypatch, "plugins", "install", "demo", "--yes")
    assert "Installed and enabled 'demo'" in capsys.readouterr().out
    assert (plugins_dir / "demo.py").read_text() == GOOD_PLUGIN
    assert _enabled("demo.py")

    _picasso(monkeypatch, "plugins", "list")
    assert "(registry: demo 1.2.0)" in capsys.readouterr().out

    _picasso(monkeypatch, "plugins", "uninstall", "demo")
    assert "Uninstalled 'demo'" in capsys.readouterr().out
    assert not (plugins_dir / "demo.py").exists()


def test_plugins_install_can_be_declined(registry, plugins_dir, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    _picasso(monkeypatch, "plugins", "install", "demo")
    assert not (plugins_dir / "demo.py").exists()


@pytest.mark.parametrize(
    "change, plugin_id, message",
    [
        ({}, "other", "No plugin with id 'other' in the registry"),
        ({"min_picasso_version": "99.0"}, "demo", "requires Picasso 99.0"),
        ({"sha256": "0" * 64}, "demo", "Install failed: Integrity check"),
    ],
)
def test_plugins_install_reports_problems(
    change, plugin_id, message, registry, plugins_dir, monkeypatch, capsys
):
    registry[0].update(change)
    _picasso(monkeypatch, "plugins", "install", plugin_id, "--yes")
    assert message in capsys.readouterr().out
    assert not (plugins_dir / "demo.py").exists()


def test_plugins_install_without_the_registry(
    plugins_dir, monkeypatch, capsys
):
    from picasso import plugins

    def get(url):
        raise OSError("offline")

    monkeypatch.setattr(plugins, "_get", get)
    _picasso(monkeypatch, "plugins", "install", "demo", "--yes")
    assert "Could not reach the online plugin registry: offline" in (
        capsys.readouterr().out
    )


def test_plugins_uninstall_of_an_unknown_plugin(
    plugins_dir, monkeypatch, capsys
):
    _picasso(monkeypatch, "plugins", "uninstall", "demo")
    assert "'demo' is not installed from the registry" in (
        capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# SPINNA batch analysis
# ---------------------------------------------------------------------------


def _save_points(path, xyz) -> str:
    """Save points (nm), ``(n, 2)`` or ``(n, 3)``, as a localization
    file."""
    n = len(xyz)
    locs = pd.DataFrame(
        {
            "frame": np.zeros(n, "u4"),
            "x": xyz[:, 0].astype("f4"),
            "y": xyz[:, 1].astype("f4"),
            "lpx": np.full(n, 2, "f4"),
            "lpy": np.full(n, 2, "f4"),
        }
    )
    if xyz.shape[1] == 3:
        locs["z"] = xyz[:, 2].astype("f4")
    io.save_locs(
        str(path),
        locs,
        [{"Frames": 1, "Width": 40, "Height": 40, "Pixelsize": 130}],
    )
    return str(path)


@pytest.fixture
def spinna_row(tmp_path):
    """A row of a SPINNA parameters file: monomers and dimers of one
    target in 25 um^2; returns the row as a dict."""
    from picasso.spinna import Structure, StructureMixer

    monomer = Structure("Monomer")
    monomer.define_coordinates("A", [0], [0], [0])
    dimer = Structure("Dimer")
    dimer.define_coordinates("A", [-10, 10], [0, 0], [0, 0])
    structures = str(tmp_path / "structures.yaml")
    io.save_info(structures, [monomer.get_info(), dimer.get_info()])
    mixer = StructureMixer(
        structures=[monomer, dimer],
        label_unc={"A": 5.0},
        le={"A": 1.0},
        width=5000.0,
        height=5000.0,
    )
    xy = np.asarray(mixer.run_simulation([300, 300])["A"])
    return {
        "structures_filename": structures,
        "exp_data_A": _save_points(tmp_path / "A.hdf5", xy),
        "le_A": 100,
        "label_unc_A": 5,
        "area": 25.0,
        "granularity": 5,
        "sim_repeats": 1,
        "save_filename": "run",
        "NND_bin": 4,
        "NND_maxdist": 200,
        "fitting_mode": "brute-force",
    }


def _spinna(monkeypatch, tmp_path, *rows) -> str:
    """Run ``picasso spinna`` on ``rows``; returns the results folder."""
    path = tmp_path / "params.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _picasso(monkeypatch, "spinna", "-p", path, "-a")
    return str(tmp_path / "params__fitting_results")


def test_spinna_batch(spinna_row, tmp_path, monkeypatch):
    results = _spinna(monkeypatch, tmp_path, spinna_row)
    summary = pd.read_csv(os.path.join(results, "summary_results.csv"))
    assert len(summary) == 1
    (row,) = summary.to_dict("records")
    assert row["Fitted structures names"] == "['Monomer', 'Dimer']"
    assert row["File location of experimental data"] == str(
        [spinna_row["exp_data_A"]]
    )
    assert row["Area (um^2)"] == 25.0
    assert row["Rotation mode"] == "2D"
    for name in (
        "run_fit_summary.txt",
        "run_fit_scores.csv",
        "run_NND_A_A.png",
    ):
        assert os.path.exists(os.path.join(results, name))


def test_spinna_batch_keeps_earlier_results(spinna_row, tmp_path, monkeypatch):
    first = _spinna(monkeypatch, tmp_path, spinna_row)
    _spinna(monkeypatch, tmp_path, spinna_row)
    assert os.path.exists(first)
    assert os.path.exists(first + "_1")


def test_spinna_batch_with_a_mask(spinna_row, tmp_path, monkeypatch):
    del spinna_row["area"]
    spinna_row["exp_data_A"] = os.path.abspath("tests/data/testdata_mols.hdf5")
    spinna_row["mask_filename_A"] = os.path.abspath(
        "tests/data/testdata_mask_spinna.npy"
    )
    spinna_row["le_A"] = 50
    spinna_row["granularity"] = 3
    results = _spinna(monkeypatch, tmp_path, spinna_row)
    (row,) = pd.read_csv(os.path.join(results, "summary_results.csv")).to_dict(
        "records"
    )
    assert "testdata_mask_spinna.npy" in row["File location of masks"]


def test_spinna_batch_area_from_the_metadata(
    spinna_row, tmp_path, monkeypatch
):
    del spinna_row["area"]
    locs, info = io.load_locs(spinna_row["exp_data_A"])
    io.save_locs(
        spinna_row["exp_data_A"], locs, info + [{"Area (um^2)": 25.0}]
    )
    results = _spinna(monkeypatch, tmp_path, spinna_row)
    (row,) = pd.read_csv(os.path.join(results, "summary_results.csv")).to_dict(
        "records"
    )
    assert row["Area (um^2)"] == 25.0


def test_spinna_batch_fits_the_labeling_efficiency(tmp_path, monkeypatch):
    """Half of A and half of B form heterodimers 10 nm apart, so both
    labeling efficiencies are 50%."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 5000, (400, 2))
    b = np.vstack([a[:200] + [10, 0], rng.uniform(0, 5000, (200, 2))])
    row = {
        "exp_data_A": _save_points(tmp_path / "A.hdf5", a),
        "exp_data_B": _save_points(tmp_path / "B.hdf5", b),
        "label_unc_A": "5",
        "label_unc_B": "5",
        "le_fitting": 1,
        "distances": "10",
        "area": 25.0,
        "granularity": 3,
        "sim_repeats": 1,
        "save_filename": "le",
        "NND_bin": 4,
        "NND_maxdist": 200,
    }
    results = _spinna(monkeypatch, tmp_path, row)
    (summary,) = pd.read_csv(
        os.path.join(results, "summary_results.csv")
    ).to_dict("records")
    for target in "AB":
        fitted = summary[f"Fitted labeling efficiency (%) for {target}"]
        assert fitted == pytest.approx(50, abs=15)
    assert os.path.exists(os.path.join(results, "le_NND_A_B.png"))


def test_spinna_batch_in_3d(spinna_row, tmp_path, monkeypatch):
    del spinna_row["area"]
    rng = np.random.default_rng(2)
    xyz = rng.uniform([0, 0, -200], [5000, 5000, 200], (600, 3))
    spinna_row["exp_data_A"] = _save_points(tmp_path / "A3d.hdf5", xyz)
    spinna_row["volume"] = 10.0  # um^3: 25 um^2 times 400 nm
    spinna_row["z_range"] = 400
    spinna_row["rotation_mode"] = "3D"
    results = _spinna(monkeypatch, tmp_path, spinna_row)
    (row,) = pd.read_csv(os.path.join(results, "summary_results.csv")).to_dict(
        "records"
    )
    assert row["Dimensionality"] == "3D"
    assert row["Rotation mode"] == "3D"


def test_spinna_batch_3d_needs_a_z_range(spinna_row, tmp_path, monkeypatch):
    del spinna_row["area"]
    rng = np.random.default_rng(2)
    xyz = rng.uniform([0, 0, -200], [5000, 5000, 200], (100, 3))
    spinna_row["exp_data_A"] = _save_points(tmp_path / "A3d.hdf5", xyz)
    spinna_row["volume"] = 10.0
    with pytest.raises(ValueError, match="specify z_range"):
        _spinna(monkeypatch, tmp_path, spinna_row)


def test_spinna_batch_of_two_targets_with_bootstrap(tmp_path, monkeypatch):
    from picasso.spinna import Structure

    structures = []
    for title, targets in (("MonA", "A"), ("MonB", "B"), ("Het", "AB")):
        structure = Structure(title)
        for k, target in enumerate(targets):
            structure.define_coordinates(target, [10.0 * k], [0], [0])
        structures.append(structure)
    structures_path = str(tmp_path / "structures.yaml")
    io.save_info(structures_path, [_.get_info() for _ in structures])
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 5000, (300, 2))
    b = np.vstack([a[:150] + [10, 0], rng.uniform(0, 5000, (150, 2))])
    row = {
        "structures_filename": structures_path,
        "exp_data_A": _save_points(tmp_path / "A.hdf5", a),
        "exp_data_B": _save_points(tmp_path / "B.hdf5", b),
        "le_A": 100,
        "le_B": 100,
        "label_unc_A": 5,
        "label_unc_B": 5,
        "area": 25.0,
        "granularity": 3,
        "sim_repeats": 1,
        "save_filename": "two",
        "NND_bin": 4,
        "NND_maxdist": 200,
        "fitting_mode": "brute-force",
    }
    path = tmp_path / "params.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    _picasso(monkeypatch, "spinna", "-p", path, "-a", "-b")
    results = str(tmp_path / "params__fitting_results")
    (summary,) = pd.read_csv(
        os.path.join(results, "summary_results.csv")
    ).to_dict("records")
    assert "+/-" in summary["Fitted proportions of structures"]
    assert "Het:" in summary["Relative proportions of A in"]


def test_spinna_batch_without_rotations(
    spinna_row, tmp_path, monkeypatch, capsys
):
    from picasso import spinna

    modes = []

    class RecordingMixer(spinna.StructureMixer):
        def __init__(self, *args, **kwargs):
            modes.append(kwargs["random_rot_mode"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(spinna, "StructureMixer", RecordingMixer)
    spinna_row["rotation_mode"] = "None"
    results = _spinna(monkeypatch, tmp_path, spinna_row)
    assert "Invalid rotation_mode" not in capsys.readouterr().out
    assert modes == [None]
    (row,) = pd.read_csv(
        os.path.join(results, "summary_results.csv"), keep_default_na=False
    ).to_dict("records")
    assert row["Rotation mode"] == "None"


@pytest.mark.parametrize(
    "value, message", [("", False), ("4D", True)], ids=["empty", "invalid"]
)
def test_spinna_batch_default_rotations(
    value, message, spinna_row, tmp_path, monkeypatch, capsys
):
    spinna_row["rotation_mode"] = value
    results = _spinna(monkeypatch, tmp_path, spinna_row)
    assert ("Invalid rotation_mode '4D'" in capsys.readouterr().out) is message
    (row,) = pd.read_csv(os.path.join(results, "summary_results.csv")).to_dict(
        "records"
    )
    assert row["Rotation mode"] == "2D"


@pytest.mark.parametrize(
    "missing, message",
    [
        ("granularity", "Column granularity not found"),
        ("le_A", "Column le_A not found"),
        ("structures_filename", "structures_filename is required"),
    ],
)
def test_spinna_batch_reports_missing_columns(
    missing, message, spinna_row, tmp_path, monkeypatch
):
    del spinna_row[missing]
    with pytest.raises(ValueError, match=message):
        _spinna(monkeypatch, tmp_path, spinna_row)


def test_spinna_batch_needs_a_csv(monkeypatch):
    with pytest.raises(TypeError, match="must end with .csv"):
        _picasso(monkeypatch, "spinna", "-p", "params.txt")


def test_spinna_columns_prints_the_reference(monkeypatch, capsys):
    _picasso(monkeypatch, "spinna", "--columns")
    assert '"exp_data_TARGET"' in capsys.readouterr().out
