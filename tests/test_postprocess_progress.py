"""Test the progress reporting of ``picasso.postprocess``,
``picasso.imageprocess`` and ``picasso.aim``.

Every public function that reports progress takes one ``progress``
argument (see ``picasso.lib.Progress``): None reports nothing,
"console" shows tqdm bars and a tracker sees each phase start through
``zero_progress``/``setMaximum`` and end at its maximum.

``RecordingProgress`` and ``assert_phases`` are shared with the
progress tests of other modules (e.g. ``tests/test_clusterer.py``).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot/picasso imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from picasso import aim, imageprocess, lib, postprocess  # noqa: E402

from tests.conftest import PIXELSIZE  # noqa: E402

PX = PIXELSIZE
PICK_DIAMETER = 3.0 * PX  # nm
#: one circular pick on each origami of the bundled test data (nm)
PICKS = [
    (x * PX, y * PX) for x in (5.5, 15.5, 25.5) for y in (5.5, 15.5, 25.5)
]
SEGMENTATION = 100

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*not linked"),
    pytest.mark.filterwarnings("ignore:.*molecules"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]


# ---------------------------------------------------------------------
# Recording tracker
# ---------------------------------------------------------------------


@dataclass
class Phase:
    """One phase seen by a ``RecordingProgress``."""

    label: str
    maximum: int | None = None
    values: list = field(default_factory=list)


class RecordingProgress(lib.MockProgress):
    """Tracker recording each phase (label, maximum and the values
    reported in it)."""

    def __init__(self):
        super().__init__()
        self.phases: list[Phase] = []
        self.closed = False

    def zero_progress(self, description=None, *args, **kwargs):
        super().zero_progress(description)
        self.phases.append(Phase(self.description_base))

    def setMaximum(self, maximum, *args, **kwargs):
        super().setMaximum(maximum)
        assert self.phases, "setMaximum called before a phase started"
        self.phases[-1].maximum = maximum

    def set_value(self, value, *args, **kwargs):
        assert self.phases, "set_value called before a phase started"
        self.phases[-1].values.append(value)

    def close(self, *args, **kwargs):
        self.closed = True

    @property
    def labels(self) -> list[str]:
        return [phase.label for phase in self.phases]


def assert_phases(tracker: RecordingProgress, labels=None) -> None:
    """Every phase declares its maximum, reports monotone values from 0
    up to it and ends at it; a tracker passed in is never closed."""
    assert tracker.phases, "no phase was started"
    if labels is not None:
        assert tracker.labels == labels
    for phase in tracker.phases:
        assert phase.maximum is not None, f"{phase.label}: no maximum"
        assert phase.maximum >= 0
        values = phase.values
        assert values, f"{phase.label}: no value reported"
        assert all(isinstance(v, (int, np.integer)) for v in values)
        assert values == sorted(values), f"{phase.label}: not monotone"
        assert values[0] >= 0
        assert values[-1] == phase.maximum, (
            f"{phase.label}: ends at {values[-1]}, not at the maximum "
            f"{phase.maximum}"
        )
    assert not tracker.closed, "a tracker passed in must not be closed"


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def _picked(locs, info):
    return postprocess.picked_locs(
        locs, info, picks=PICKS, pick_shape="Circle", pick_size=PICK_DIAMETER
    )


def _molecules(n, seed, size=10_000.0, z=False):
    """Uniformly random molecules (nm) in a ``size`` x ``size`` field."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, size, size=(2, n))
    locs = pd.DataFrame(
        {
            "frame": np.zeros(n, dtype=np.uint32),
            "x": x,
            "y": y,
            "n_events": np.ones(n, dtype=np.int32),
        }
    )
    if z:
        locs["z"] = rng.uniform(-400, 400, n)
    return locs


STAT_INFO = [{"Pixelsize": 100.0, "Width": 100, "Height": 100, "Frames": 1}]
CBC_INFO = [{"Pixelsize": 130.0}]


def _cbc_channel(n, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 5_000, size=(20, 2))
    xy = centers[rng.integers(len(centers), size=n)]
    xy = xy + rng.normal(0.0, 26.0, size=xy.shape)
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.uint32),
            "x": xy[:, 0].astype(np.float32),
            "y": xy[:, 1].astype(np.float32),
        }
    )


def _shifted_gaussians(n=4, size=32, sigma=2.0):
    """Images of a Gaussian spot shifted by one pixel per image."""
    yy, xx = np.mgrid[:size, :size]
    return [
        np.exp(-((xx - 14 - k) ** 2 + (yy - 16) ** 2) / (2 * sigma**2))
        for k in range(n)
    ]


def _with_z(locs):
    locs = locs.copy()
    rng = np.random.default_rng(0)
    locs["z"] = rng.normal(0, 50, len(locs)).astype(np.float32)
    return locs


# ---------------------------------------------------------------------
# Cases: (id, run(locs, info, progress), expected phase labels)
# ---------------------------------------------------------------------

STAT_LABELS = ["Calculating Ripley's H", "Simulating envelope"]

CASES = [
    (
        "picked_locs",
        lambda l, i, p: postprocess.picked_locs(
            l,
            i,
            picks=PICKS,
            pick_shape="Circle",
            pick_size=PICK_DIAMETER,
            progress=p,
        ),
        ["Picking localizations"],
    ),
    (
        "picked_locs_square",
        lambda l, i, p: postprocess.picked_locs(
            l,
            i,
            picks=PICKS,
            pick_shape="Square",
            pick_size=PICK_DIAMETER,
            progress=p,
        ),
        ["Picking localizations"],
    ),
    (
        "evaluate_picks",
        lambda l, i, p: postprocess.evaluate_picks(
            _picked(l, i), i, progress=p
        ),
        ["Evaluating picks"],
    ),
    (
        "pick_kinetics",
        lambda l, i, p: postprocess.pick_kinetics(
            _picked(l, i), i, progress=p
        ),
        ["Calculating kinetics"],
    ),
    (
        "pick_properties",
        lambda l, i, p: postprocess.pick_properties(
            _picked(l, i), i, progress=p
        ),
        ["Calculating kinetics", "Calculating group properties"],
    ),
    (
        "groupprops",
        lambda l, i, p: postprocess.groupprops(
            pd.concat(_picked(l, i), ignore_index=True), progress=p
        ),
        ["Calculating group properties"],
    ),
    (
        "combine_locs_in_picks",
        lambda l, i, p: postprocess.combine_locs_in_picks(
            l,
            i,
            picks=PICKS,
            pick_shape="Circle",
            pick_size=PICK_DIAMETER,
            progress=p,
        ),
        ["Picking localizations", "Combining picks"],
    ),
    (
        "segment",
        lambda l, i, p: postprocess.segment(l, i, SEGMENTATION, progress=p),
        ["Generating segments"],
    ),
    (
        "undrift",
        lambda l, i, p: postprocess.undrift(
            l, i, segmentation=SEGMENTATION, display=False, progress=p
        ),
        ["Generating segments", "Correlating image pairs"],
    ),
    (
        "undrift_from_fiducials",
        lambda l, i, p: postprocess.undrift_from_fiducials(
            l, i, picks=PICKS, pick_size=PICK_DIAMETER, progress=p
        ),
        ["Picking localizations", "Calculating drift"],
    ),
    (
        "nena",
        lambda l, i, p: postprocess.nena(l, i, progress=p),
        ["Calculating NeNA"],
    ),
    (
        "frc_rois",
        lambda l, i, p: postprocess.frc_rois(
            l,
            i,
            viewport=((10 * PX, 10 * PX), (20 * PX, 20 * PX)),
            n_rois=4,
            roi_size=2 * PX,
            min_locs=20,
            progress=p,
        ),
        ["Calculating NeNA", "Calculating FRC"],
    ),
    (
        "cbc",
        lambda l, i, p: postprocess.cbc(
            _cbc_channel(300, 1),
            CBC_INFO,
            partner_locs=_cbc_channel(200, 2),
            chunk_size=100,
            progress=p,
        ),
        ["Calculating CBC"],
    ),
    (
        "cbc_all_pairs",
        lambda l, i, p: postprocess.cbc_all_pairs(
            [_cbc_channel(300, 1), _cbc_channel(200, 2), _cbc_channel(50, 3)],
            [CBC_INFO] * 3,
            chunk_size=100,
            progress=p,
        ),
        ["Calculating CBC"],
    ),
    (
        "resi",
        lambda l, i, p: postprocess.resi(
            [l.copy(), l.copy()],
            [i, i],
            radius_xy=2.0,
            min_locs=2,
            progress=p,
        ),
        ["Clustering", "Calculating cluster centers"] * 2,
    ),
    (
        "ripley_h",
        lambda l, i, p: postprocess.ripley_h(
            _molecules(500, 0),
            STAT_INFO,
            r_max=500,
            steps=10,
            n_simulations=2,
            progress=p,
        ),
        STAT_LABELS,
    ),
    (
        "ripley_h_no_simulations",
        lambda l, i, p: postprocess.ripley_h(
            _molecules(500, 0),
            STAT_INFO,
            picks=[((0, 0), (5000, 5000)), ((5000, 5000), (10000, 10000))],
            pick_shape="Box",
            r_max=500,
            steps=10,
            n_simulations=0,
            progress=p,
        ),
        ["Calculating Ripley's H"],
    ),
    (
        "rdf",
        lambda l, i, p: postprocess.rdf(
            _molecules(500, 0),
            STAT_INFO,
            r_max=500,
            steps=10,
            n_simulations=2,
            progress=p,
        ),
        ["Calculating RDF", "Simulating envelope"],
    ),
    (
        "nnd_distributions",
        lambda l, i, p: postprocess.nnd_distributions(
            [_molecules(500, 0), _molecules(300, 1)],
            STAT_INFO,
            n_simulations=2,
            progress=p,
        ),
        ["Calculating NNDs", "Simulating envelope"],
    ),
    (
        "rcc",
        lambda l, i, p: imageprocess.rcc(
            _shifted_gaussians(), max_shift=10, progress=p
        ),
        ["Correlating image pairs"],
    ),
    (
        "aim",
        lambda l, i, p: aim.aim(l, i, segmentation=SEGMENTATION, progress=p),
        ["Undrifting xy by AIM (1/2)", "Undrifting xy by AIM (2/2)"],
    ),
    (
        "aim_3d",
        lambda l, i, p: aim.aim(
            _with_z(l), i, segmentation=SEGMENTATION, progress=p
        ),
        [
            "Undrifting xy by AIM (1/2)",
            "Undrifting xy by AIM (2/2)",
            "Undrifting z by AIM (1/2)",
            "Undrifting z by AIM (2/2)",
        ],
    ),
    (
        "intersection_max",
        lambda l, i, p: aim.intersection_max(
            l["x"],
            l["y"],
            l["x"],
            l["y"],
            l["frame"] + 1,
            np.arange(0, 1001, SEGMENTATION),
            20.0,
            60.0,
            lib.field_of_view(i)[0],
            aim_round=2,
            progress=p,
        ),
        ["Undrifting xy by AIM (2/2)"],
    ),
    (
        "intersection_max_z",
        lambda l, i, p: aim.intersection_max_z(
            l["x"],
            l["y"],
            _with_z(l)["z"],
            l["x"],
            l["y"],
            _with_z(l)["z"],
            l["frame"] + 1,
            np.arange(0, 1001, SEGMENTATION),
            20.0,
            60.0,
            *lib.field_of_view(i),
            progress=p,
        ),
        ["Undrifting z by AIM (1/2)"],
    ),
]
CASE_IDS = [case[0] for case in CASES]


@pytest.mark.parametrize("name, run, labels", CASES, ids=CASE_IDS)
def test_none_reports_nothing(capsys, locs, info, name, run, labels):
    run(locs.copy(), info, None)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize("name, run, labels", CASES, ids=CASE_IDS)
def test_console_shows_each_phase(capsys, locs, info, name, run, labels):
    run(locs.copy(), info, "console")
    err = capsys.readouterr().err
    for label in labels:
        assert label in err


@pytest.mark.parametrize("name, run, labels", CASES, ids=CASE_IDS)
def test_tracker_sees_each_phase_end(locs, info, name, run, labels):
    tracker = RecordingProgress()
    run(locs.copy(), info, tracker)
    assert_phases(tracker, labels)


# ---------------------------------------------------------------------
# Maxima set by the functions
# ---------------------------------------------------------------------


def test_picked_locs_counts_picks(locs, info):
    tracker = RecordingProgress()
    postprocess.picked_locs(
        locs,
        info,
        picks=PICKS,
        pick_shape="Circle",
        pick_size=PICK_DIAMETER,
        progress=tracker,
    )
    assert tracker.phases[0].maximum == len(PICKS)
    assert tracker.phases[0].values == list(range(len(PICKS) + 1)) + [
        len(PICKS)
    ]


def test_undrift_maxima(locs, info):
    tracker = RecordingProgress()
    postprocess.undrift(
        locs, info, segmentation=SEGMENTATION, display=False, progress=tracker
    )
    n_seg = postprocess.n_segments(info, SEGMENTATION)
    assert [p.maximum for p in tracker.phases] == [
        n_seg,
        n_seg * (n_seg - 1) // 2,
    ]


def test_aim_maxima(locs, info):
    """The first round aligns all but the reference segment."""
    tracker = RecordingProgress()
    aim.aim(locs, info, segmentation=SEGMENTATION, progress=tracker)
    n_frames = lib.get_from_metadata(info, "Frames")
    n_segments = len(np.arange(0, n_frames, SEGMENTATION))
    assert [p.maximum for p in tracker.phases] == [
        n_segments - 1,
        n_segments,
    ]


def test_undrift_from_fiducials_auto_detection_phase(monkeypatch, locs, info):
    monkeypatch.setattr(
        "picasso.postprocess.imageprocess.find_fiducials",
        lambda _locs, _info: (PICKS, PICK_DIAMETER),
    )
    tracker = RecordingProgress()
    postprocess.undrift_from_fiducials(locs, info, progress=tracker)
    assert_phases(
        tracker,
        ["Finding fiducials", "Picking localizations", "Calculating drift"],
    )
    # auto-detection and drift calculation have no item count
    assert tracker.phases[0].maximum == 0
    assert tracker.phases[2].maximum == 0


def test_cbc_all_pairs_counts_all_channels():
    channels = [_cbc_channel(300, 1), _cbc_channel(200, 2)]
    tracker = RecordingProgress()
    postprocess.cbc_all_pairs(
        channels, [CBC_INFO] * 2, chunk_size=100, progress=tracker
    )
    assert tracker.phases[0].maximum == 500
    assert tracker.phases[0].values == [0, 100, 200, 300, 400, 500, 500]


def test_nested_calls_keep_the_tracker_open(locs, info):
    """``undrift`` passes its tracker to ``segment`` and ``rcc``, which
    must not close it."""
    tracker = RecordingProgress()
    postprocess.undrift(
        locs, info, segmentation=SEGMENTATION, display=False, progress=tracker
    )
    assert not tracker.closed


def test_invalid_progress_raises(locs, info):
    with pytest.raises(ValueError):
        postprocess.nena(locs, info, progress="bogus")
    with pytest.raises(TypeError):
        postprocess.nena(locs, info, progress=print)


# ---------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------


def _nfndh_reference(frame, x, y, group, d_max, bin_size):
    """Brute-force next frame neighbor distance histogram."""
    bins = np.arange(0, d_max, bin_size)
    dnfl = np.zeros(len(bins))
    for i in range(len(frame)):
        for j in range(len(frame)):
            if frame[j] != frame[i] + 1 or group[j] != group[i]:
                continue
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < d_max and int(d / bin_size) < len(dnfl):
                dnfl[int(d / bin_size)] += 1
    return dnfl


@pytest.mark.parametrize("n", [7, 99, 100, 250, 1001])
def test_nfndh_processes_every_localization(n):
    """All localizations count, also when there are fewer than 100 or
    their number is not a multiple of 100 (previously the trailing
    ``n % 100`` were skipped, i.e. all of them for ``n < 100``)."""
    rng = np.random.default_rng(n)
    # one or two localizations per frame, some frames left out
    frame = np.cumsum(rng.integers(0, 2, n))
    x = rng.uniform(0, 150, n)
    y = rng.uniform(0, 150, n)
    group = rng.integers(0, 2, n).astype(np.int32)
    tracker = RecordingProgress()
    _, dnfl = postprocess._nfndh(frame, x, y, group, 200.0, 0.2, tracker)
    reference = _nfndh_reference(frame, x, y, group, 200.0, 0.2)
    np.testing.assert_array_equal(dnfl, reference)
    assert dnfl.sum() > 0
    assert_phases(tracker, ["Calculating NeNA"])
    assert tracker.phases[0].maximum == n
    assert len(tracker.phases[0].values) <= 102


def test_nfndh_counts_neighbors_at_the_end_of_the_list():
    """Next-frame neighbors that are the last localizations of the
    (frame-sorted) list are counted, and the last localization has no
    neighbors of its own."""
    frame = np.array([0, 1, 1])
    x = np.array([0.0, 1.0, 0.0])
    y = np.zeros(3)
    group = np.zeros(3, dtype=np.int32)
    _, dnfl = postprocess._nfndh(
        frame, x, y, group, 200.0, 0.2, lib.MockProgress()
    )
    expected = np.zeros_like(dnfl)
    expected[0] = 1  # distance 0 to loc 2
    expected[5] = 1  # distance 1 nm to loc 1
    np.testing.assert_array_equal(dnfl, expected)


def test_nfndh_ignores_a_distance_of_d_max():
    """A neighbor exactly ``d_max`` away lies past the last bin; numba
    does not check the index, so it was written outside the
    histogram."""
    frame = np.array([0, 1, 2])
    x = np.array([0.0, 200.0, 199.9])
    y = np.zeros(3)
    group = np.zeros(3, dtype=np.int32)
    _, dnfl = postprocess._nfndh(
        frame, x, y, group, 200.0, 0.2, lib.MockProgress()
    )
    expected = np.zeros_like(dnfl)
    expected[int(0.1 / 0.2)] = 1  # 0.1 nm between locs 1 and 2
    np.testing.assert_array_equal(dnfl, expected)


def test_groupprops_without_dark_column(locs, info):
    """Previously a KeyError: only an AttributeError was caught."""
    group_locs = pd.concat(_picked(locs, info), ignore_index=True)
    assert "dark" not in group_locs.columns
    groups = postprocess.groupprops(group_locs)
    assert len(groups) == group_locs["group"].nunique()
    assert "qpaint_idx" not in groups.columns


def test_groupprops_excludes_events_without_dark_time(locs, info):
    group_locs = pd.concat(_picked(locs, info), ignore_index=True)
    group_locs["dark"] = 5.0
    group_locs.loc[group_locs.index[:3], "dark"] = -1
    groups = postprocess.groupprops(group_locs)
    assert groups["n_events"].sum() == len(group_locs) - 3


@pytest.mark.parametrize("progress", [None, "console"])
def test_pick_kinetics_without_picks(progress):
    """Previously a NameError (with a callable) and a ValueError from
    concatenating nothing."""
    tracker = RecordingProgress() if progress is None else progress
    length, dark, no_locs, out_locs, kept = postprocess.pick_kinetics(
        [], [{"Frames": 10}], progress=tracker
    )
    assert len(length) == len(dark) == len(no_locs) == len(kept) == 0
    assert len(out_locs) == 0
    if progress is None:
        assert_phases(tracker, ["Calculating kinetics"])


def test_nnd_distributions_ends_at_maximum_with_invalid_regions():
    """A 3D region without localizations (no depth) is skipped; every
    phase still ends at its maximum."""
    locs = _molecules(500, 0, size=5000.0, z=True)
    picks = [((0, 0), (5000, 5000)), ((6000, 6000), (9000, 9000))]
    tracker = RecordingProgress()
    postprocess.nnd_distributions(
        locs,
        STAT_INFO,
        picks=picks,
        pick_shape="Box",
        use_z=True,
        n_simulations=2,
        progress=tracker,
    )
    assert_phases(tracker, ["Calculating NNDs", "Simulating envelope"])
    assert tracker.phases[0].maximum == 2
    assert tracker.phases[1].maximum == 2


def test_rcc_without_progress_is_silent(capsys):
    """Previously, ``callback=None`` printed a tqdm bar."""
    rng = np.random.default_rng(0)
    segments = [rng.random((16, 16)) for _ in range(3)]
    imageprocess.rcc(segments)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_align_rcc_is_silent(capsys, locs, info):
    postprocess.align_rcc([locs.copy(), locs.copy()], [info, info])
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
