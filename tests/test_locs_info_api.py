"""Test the conventions of the public functions on localizations: the
localizations and their metadata come first, ``f(locs, info, *, ...)``,
with every later argument keyword-only, and the functions that return
new localizations (processing steps) also return their complete
metadata, i.e., ``info`` with one entry appended (see
``lib.append_metadata``), without modifying ``info``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import copy
import importlib
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from picasso import clusterer, expression, io, lib, localize, postprocess
from tests.regen_stability_snapshots import public_api

#: names of the first argument of functions on localizations
LOCS_NAMES = {"locs", "mols", "picked_locs"}
#: names of the metadata argument that follows it
INFO_NAMES = {"info", "infos"}


def _public_functions():
    """(qualified name, function) of every public function."""
    out = []
    for module_name, names in public_api().items():
        module = importlib.import_module(module_name)
        for name in names:
            obj = getattr(module, name)
            if inspect.isfunction(obj):
                out.append((f"{module_name}.{name}", obj))
    return out


def _locs_info_functions():
    """(name, parameters) of the public functions that take
    localizations followed by their metadata."""
    out = []
    for name, fn in _public_functions():
        params = list(inspect.signature(fn).parameters.values())
        for first, second in zip(params, params[1:]):
            if first.name in LOCS_NAMES and second.name in INFO_NAMES:
                out.append((name, params))
                break
    return out


LOCS_INFO_FUNCTIONS = _locs_info_functions()


def test_public_api_has_locs_info_functions():
    # guards the parametrization below against silently testing nothing
    assert len(LOCS_INFO_FUNCTIONS) > 60


@pytest.mark.parametrize(
    "name, params",
    LOCS_INFO_FUNCTIONS,
    ids=[name for name, _ in LOCS_INFO_FUNCTIONS],
)
def test_arguments_after_info_are_keyword_only(name, params):
    names = [p.name for p in params]
    i_info = next(
        i
        for i, p in enumerate(params[1:], 1)
        if p.name in INFO_NAMES and params[i - 1].name in LOCS_NAMES
    )
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert params[i_info].kind in positional, f"{name}: {names}"
    later = [
        p.name
        for p in params[i_info + 1 :]
        if p.kind
        not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
    ]
    assert not later, f"{name}: {later} must be keyword-only"


def _returned_metadata(fn) -> list[str]:
    """Names in the Returns section of ``fn``'s docstring that are
    metadata (``list of dicts`` or ``list of lists of dicts``)."""
    doc = inspect.getdoc(fn) or ""
    match = re.search(
        r"\nReturns\n-+\n(.*?)(?:\n\n[A-Z][A-Za-z ]+\n-+\n|\Z)", doc, re.S
    )
    if match is None:
        return []
    return re.findall(
        r"^(\w+) : list of (?:lists of )?dicts", match.group(1), re.M
    )


RETURNING_METADATA = [
    (name, fn) for name, fn in _public_functions() if _returned_metadata(fn)
]


def test_public_api_has_functions_returning_metadata():
    # guards the parametrization below against silently testing nothing
    assert len(RETURNING_METADATA) > 30


@pytest.mark.parametrize(
    "name, fn",
    RETURNING_METADATA,
    ids=[name for name, _ in RETURNING_METADATA],
)
def test_returned_metadata_is_annotated(name, fn):
    # the annotation follows the docstring when the return changes
    annotation = str(inspect.signature(fn).return_annotation)
    assert "list[dict]" in annotation, f"{name} -> {annotation}"


# --- lib.append_metadata ---------------------------------------------------


def test_append_metadata_copies_and_appends():
    info = [{"Width": 32, "Frames": 10}]
    entry = {"Step": "Test", "Values": [1, 2]}
    new_info = lib.append_metadata(info, entry)
    assert new_info == [{"Width": 32, "Frames": 10}, entry]
    assert info == [{"Width": 32, "Frames": 10}]
    # the entries are copies: setting a key does not change the input
    assert new_info[0] is not info[0] and new_info[1] is not entry
    lib.overwrite_metadata(new_info, "Width", 64)
    assert info[0]["Width"] == 32


@pytest.mark.parametrize(
    "info, expected",
    [
        (None, [{"a": 1}]),
        ({"b": 2}, [{"b": 2}, {"a": 1}]),
        ([], [{"a": 1}]),
    ],
)
def test_append_metadata_accepts_none_and_dict(info, expected):
    assert lib.append_metadata(info, {"a": 1}) == expected


def test_append_metadata_appends_several_entries():
    new_info = lib.append_metadata([{"a": 1}], [{"b": 2}, {"c": 3}])
    assert new_info == [{"a": 1}, {"b": 2}, {"c": 3}]


@pytest.mark.parametrize(
    "info, entry",
    [("x", {"a": 1}), ([{"a": 1}], None), ([{"a": 1}], [1, 2])],
)
def test_append_metadata_rejects_other_types(info, entry):
    with pytest.raises(TypeError):
        lib.append_metadata(info, entry)


# --- processing steps return their complete metadata ------------------------


def _info():
    return [
        {
            "Generated by": "test",
            "Width": 64,
            "Height": 64,
            "Frames": 400,
            "Pixelsize": 130.0,
        }
    ]


def _locs():
    """Five well separated groups of localizations, each a few binding
    events of several consecutive frames."""
    rng = np.random.default_rng(0)
    rows = []
    for k, (cx, cy) in enumerate(
        [(1000, 1000), (3000, 1500), (5000, 5000), (2000, 6000), (7000, 3000)]
    ):
        for start in range(10 + k, 380, 60):
            for frame in range(start, start + 5):
                rows.append(
                    (frame, cx + rng.normal(0, 3), cy + rng.normal(0, 3))
                )
    frame, x, y = np.array(rows).T
    n = len(frame)
    return pd.DataFrame(
        {
            "frame": frame.astype(np.uint32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": np.full(n, 1000, dtype=np.float32),
            "sx": np.full(n, 130, dtype=np.float32),
            "sy": np.full(n, 130, dtype=np.float32),
            "bg": np.full(n, 10, dtype=np.float32),
            "lpx": np.full(n, 3, dtype=np.float32),
            "lpy": np.full(n, 3, dtype=np.float32),
        }
    ).sort_values("frame", kind="stable", ignore_index=True)


def _linked(locs, info):
    return postprocess.link(locs, info, r_max=20)[0]


def _grouped(locs, info):
    return clusterer.dbscan(locs, info, radius=20, min_samples=3)[0]


PICKS = {
    "picks": [(1000.0, 1000.0), (5000.0, 5000.0)],
    "pick_shape": "Circle",
    "pick_size": 100.0,
}

#: step name, function (locs, info) -> (locs, info, ...), the input
#: localizations from the test data and the expected "Step"
STEPS = [
    (
        "link",
        lambda locs, info: postprocess.link(locs, info, r_max=20),
        None,
        "Link",
    ),
    (
        "select_binding_event_cores",
        lambda locs, info: postprocess.select_binding_event_cores(
            locs, info, r_max=20
        ),
        None,
        "Binding event cores",
    ),
    (
        "compute_dark_times",
        lambda locs, info: postprocess.compute_dark_times(locs, info),
        _linked,
        "Dark",
    ),
    (
        "compute_local_density",
        lambda locs, info: postprocess.compute_local_density(
            locs, info, radius=50
        ),
        None,
        "Density",
    ),
    (
        "remove_locs_in_picks",
        lambda locs, info: postprocess.remove_locs_in_picks(
            locs, info, **PICKS
        ),
        None,
        "Remove picked localizations",
    ),
    (
        "combine_locs_in_picks",
        lambda locs, info: postprocess.combine_locs_in_picks(
            locs, info, **PICKS
        ),
        None,
        "Combine picks",
    ),
    (
        "apply_drift",
        lambda locs, info: postprocess.apply_drift(
            locs.copy(), info, drift=np.zeros((400, 2))
        ),
        None,
        "Apply drift",
    ),
    (
        "dbscan",
        lambda locs, info: clusterer.dbscan(
            locs, info, radius=20, min_samples=3
        ),
        None,
        "DBSCAN",
    ),
    (
        "hdbscan",
        lambda locs, info: clusterer.hdbscan(
            locs, info, min_cluster_size=10, min_samples=3
        ),
        None,
        "HDBSCAN",
    ),
    (
        "cluster",
        lambda locs, info: clusterer.cluster(
            locs, info, radius_xy=20, min_locs=5, frame_analysis=False
        ),
        None,
        "SMLM clusterer",
    ),
    (
        "find_cluster_centers",
        lambda locs, info: clusterer.find_cluster_centers(locs, info),
        _grouped,
        "Cluster centers",
    ),
    (
        "filter_locs",
        lambda locs, info: expression.filter_locs(
            locs, info, conditions=["photons > 500", "x < 6000"]
        ),
        None,
        "Filter",
    ),
    (
        "apply_filter_steps",
        lambda locs, info: lib.apply_filter_steps(
            locs,
            info,
            recipe=[{"Step": "Filter", "Filter steps": ["x < 6000"]}],
        ),
        None,
        "Filter",
    ),
    (
        "unfold_localizations_square",
        lambda locs, info: lib.unfold_localizations_square(
            locs, info, n_square=2
        ),
        _grouped,
        "Unfold groups",
    ),
]


@pytest.mark.parametrize(
    "func, prepare, step",
    [(func, prepare, step) for _, func, prepare, step in STEPS],
    ids=[name for name, *_ in STEPS],
)
def test_step_returns_complete_metadata(func, prepare, step):
    locs, info = _locs(), _info()
    if prepare is not None:
        locs = prepare(locs, info)
    locs_before = locs.copy()
    info_before = copy.deepcopy(info)

    out_locs, out_info, *_ = func(locs, info)

    assert isinstance(out_locs, pd.DataFrame) and len(out_locs)
    assert len(out_info) == len(info) + 1
    assert out_info[-1]["Step"] == step
    # the earlier entries are carried over, as copies
    assert all(new is not old for new, old in zip(out_info, info))
    assert out_info[0]["Frames"] == info[0]["Frames"]
    # the input is not modified
    assert info == info_before
    pd.testing.assert_frame_equal(locs, locs_before)


def test_step_without_info_returns_only_its_entry():
    clustered, info = clusterer.dbscan(_locs(), radius=20, min_samples=3)
    assert len(clustered)
    assert [entry["Step"] for entry in info] == ["DBSCAN"]


def test_align_returns_metadata_of_each_channel():
    locs, info = _locs(), _info()
    shifted = locs.copy()
    shifted["x"] += 50
    shifted["y"] -= 30
    infos = [info, copy.deepcopy(info)]
    out_locs, out_infos, shifts = postprocess.align(
        [locs, shifted],
        infos,
        method="picked",
        picks=PICKS["picks"],
        pick_shape="Circle",
        pick_size=200.0,
        return_shifts=True,
    )
    assert len(out_locs) == len(out_infos) == 2
    for i, out_info in enumerate(out_infos):
        entry = out_info[-1]
        assert entry["Step"] == "Align" and entry["Method"] == "Picked"
        assert entry["Shift x (nm)"] == pytest.approx(float(shifts[0][i]))
        assert entry["Shift y (nm)"] == pytest.approx(float(shifts[1][i]))
    assert out_infos[1][-1]["Shift x (nm)"] - out_infos[0][-1][
        "Shift x (nm)"
    ] == pytest.approx(50, abs=1)
    assert [len(i) for i in infos] == [1, 1]


# --- identifying and fitting a movie ----------------------------------------


@pytest.fixture
def movie(tmp_path):
    """A memory-mapped movie with two spots per frame, and its metadata."""
    rng = np.random.default_rng(0)
    n_frames, height, width = 20, 32, 32
    yy, xx = np.mgrid[:height, :width]
    frames = rng.poisson(20, size=(n_frames, height, width)).astype(float)
    for cy, cx in [(10, 10), (20, 22)]:
        frames += 400 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 2.0)
    movie = np.memmap(
        tmp_path / "movie.raw",
        dtype=np.uint16,
        mode="w+",
        shape=frames.shape,
    )
    movie[:] = frames.astype(np.uint16)
    info = [{"Frames": n_frames, "Width": width, "Height": height}]
    return movie, info


CAMERA_INFO = {"Baseline": 0, "Sensitivity": 1, "Gain": 1, "Pixelsize": 130}


def test_identify_and_fit_return_complete_metadata(movie):
    movie, info = movie
    parameters = localize.IdentificationParameters(
        method=localize.NetGradient(1000), box=7
    )
    ids, ids_info = localize.identify(
        movie, info, parameters=parameters, threaded=False
    )
    assert len(ids)
    assert ids_info[0] == info[0]
    assert ids_info[-1]["Step"] == "Identify"
    assert ids_info[-1]["Box Size"] == 7
    locs, locs_info = localize.fit(
        movie,
        ids_info,
        camera_info=CAMERA_INFO,
        identifications=ids,
        box=7,
        multiprocess=False,
    )
    assert len(locs) == len(ids)
    assert [entry.get("Step") for entry in locs_info] == [
        None,
        "Identify",
        "Fit 2D",
    ]
    assert locs_info[-1]["Pixelsize"] == 130
    # the input metadata are not modified
    assert len(info) == 1 and len(ids_info) == 2


def test_identify_without_metadata_returns_only_its_entry(movie):
    movie, _ = movie
    parameters = localize.IdentificationParameters(
        method=localize.NetGradient(1000), box=7
    )
    _, ids_info = localize.identify(
        movie, parameters=parameters, threaded=False
    )
    assert [entry["Step"] for entry in ids_info] == ["Identify"]


def test_calibration_loaders_keep_the_path(tmp_path):
    path = str(tmp_path / "calib.yaml")
    io.save_calibration(
        path,
        {"X Coefficients": [0.0] * 7, "Y Coefficients": [0.0] * 7},
    )
    assert io.load_calibration(path)["Path"] == path
    assert io.load_any_calibration(path)["Path"] == path
