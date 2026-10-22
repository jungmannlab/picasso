"""Filtering localizations by conditions written as text.

Covers ``picasso.expression`` (syntax, statistics, NaN and unsigned
integer handling, error messages, range conditions), the ordered
replay of Filter steps in ``lib.extract_filter_recipe`` and
``lib.apply_filter_steps`` (including metadata saved before the
steps were recorded) and the ``filter`` command with arguments.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from picasso import __main__ as cli
from picasso import expression, io, lib
from picasso.expression import ExpressionError


@pytest.fixture
def locs():
    return pd.DataFrame(
        {
            "frame": np.array([0, 5, 10, 20, 30], dtype=np.uint32),
            "photons": np.array(
                [100, 600, 900, np.nan, 2000], dtype=np.float32
            ),
            "sx": np.array([1.0, 0.9, 1.5, 1.0, 1.1], dtype=np.float32),
            "lpx": np.array([0.2, 0.1, 0.05, 0.1, 0.02], dtype=np.float32),
        }
    )


def _kept(condition, locs, variables=None):
    return np.flatnonzero(expression.condition(condition, locs, variables))


@pytest.mark.parametrize(
    "condition, kept",
    [
        ("photons > 500", [1, 2, 4]),
        ("photons > 500 and 0.8 < sx < 1.2", [1, 4]),
        ("(photons > 500) & (sx < 1.2)", [1, 4]),
        ("sx > 1.4 or frame == 20", [2, 3]),
        ("(sx > 1.4) | ~(frame < 30)", [2, 4]),
        ("not photons > 500", [0, 3]),
        ("abs(sx - 1) < 0.05", [0, 3]),
        ("sqrt(lpx**2 + lpx**2) < 0.1", [2, 4]),
        ("floor(frame / 10) % 2 == 1", [2, 4]),
        ("isfinite(photons)", [0, 1, 2, 4]),
        ("photons >= 2e3", [4]),
        ("-sx < -1.2", [2]),
    ],
)
def test_condition(locs, condition, kept):
    assert _kept(condition, locs).tolist() == kept


def test_unsigned_columns_do_not_wrap(locs):
    """frame is uint32; frame - 10 must be negative for early frames."""
    assert _kept("frame - 10 < 0", locs).tolist() == [0, 1]


def test_statistics_ignore_non_finite_values(locs):
    # finite photons: 100, 600, 900, 2000 -> median 750, mean 900
    assert _kept("photons > median(photons)", locs).tolist() == [2, 4]
    assert _kept("photons >= mean(photons)", locs).tolist() == [2, 4]
    assert _kept("photons == max(photons)", locs).tolist() == [4]
    assert _kept("photons == min(photons)", locs).tolist() == [0]
    assert _kept("photons < percentile(photons, 25)", locs).tolist() == [0]
    assert _kept("lpx < mean(lpx) - std(lpx) / 2", locs).tolist() == [2, 4]
    assert _kept("photons / sum(photons) > 0.5", locs).tolist() == [4]


def test_statistics_of_expressions(locs):
    """A statistic takes any per-localization expression."""
    assert _kept("sx * 2 > median(sx * 2)", locs).tolist() == [2, 4]


def test_constants_and_variables(locs):
    pixelsize = {"pixelsize": 130}
    assert _kept("lpx * pixelsize < 10", locs, pixelsize).tolist() == [2, 4]
    assert _kept("photons < inf", locs).tolist() == [0, 1, 2, 4]
    # a column wins over a variable of the same name
    assert _kept("sx > 1.2", locs, {"sx": 0.0}).tolist() == [2]


def test_evaluate_returns_values(locs):
    np.testing.assert_allclose(
        expression.evaluate("sx * 2", locs), locs["sx"] * 2
    )
    assert expression.evaluate("2 * pi", {}) == pytest.approx(2 * np.pi)


@pytest.mark.parametrize(
    "condition, message",
    [
        ("", "empty"),
        ("photons >", "Invalid syntax"),
        ("phtons > 1", "Unknown name 'phtons'. Available: frame, photons"),
        ("photons > 500 & sx < 1.2", "parentheses"),
        ("lpx ^ 2 < 1", "use ** for powers"),
        ("photons * 2", "must be a condition"),
        ("mean(photons) > 500", "same for every localization"),
        ("photons > 'a'", "Only numbers"),
        ("photons.max() > 0", "not a supported function"),
        ("photons[0] > 0", "not supported"),
        ("open('x') > 0", "Unknown function 'open'"),
        ("percentile(photons) > 0", "two arguments"),
        ("photons > percentile(photons, 101)", "0 to 100"),
        ("median(photons, axis=0) > 0", "keyword"),
        ("photons > 500 and sx", "'sx' is a number"),
        ("not sx", "needs a condition"),
        ("photons in 5", "not supported"),
        ("median > 0", "is a function"),
        ("lpx * pixelsize < 10", "pixel size"),
    ],
)
def test_errors(locs, condition, message):
    with pytest.raises(ExpressionError, match=None) as error:
        expression.condition(condition, locs)
    assert message in str(error.value)


def test_names():
    assert expression.names(
        "photons > median(photons) and lpx * pixelsize < pi"
    ) == {"photons", "lpx", "pixelsize"}


@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize(
    "vmin, vmax", [(0.75, 1.05), (-np.inf, 2.0), (1e-5, np.inf), (-3, 7)]
)
def test_range_round_trip(vmin, vmax, inclusive):
    condition = expression.range_expression("sx", vmin, vmax, inclusive)
    assert expression.parse_range(condition) == ("sx", vmin, vmax)


def test_range_expression_matches_range_filter(locs):
    sx = locs["sx"].to_numpy()
    for inclusive, keep in (
        (False, (sx > 0.9) & (sx < 1.1)),
        (True, (sx >= 0.9) & (sx <= 1.1)),
    ):
        condition = expression.range_expression(
            "sx", np.float32(0.9), np.float64(1.1), inclusive
        )
        np.testing.assert_array_equal(
            expression.condition(condition, locs), keep
        )


@pytest.mark.parametrize(
    "condition",
    ["sx > 1", "0.8 < sx > 1", "0.8 < sx < x0 ", "0 < pi < 4", "sx < 1 < 2"],
)
def test_parse_range_rejects_other_conditions(condition):
    assert expression.parse_range(condition) is None


def test_filter_locs_uses_metadata_pixel_size(locs):
    info = [{"Generated by": "test"}, {"Pixelsize": 100}]
    filtered, _ = expression.filter_locs(
        locs, info, conditions="lpx * pixelsize < 6"
    )
    assert filtered["lpx"].tolist() == pytest.approx([0.05, 0.02])
    assert filtered.index.tolist() == [0, 1]


def test_variables_from_info():
    assert expression.variables_from_info(None) == {}
    assert expression.variables_from_info([{"Width": 10}]) == {}
    assert expression.variables_from_info({"Pixelsize": 130}) == {
        "pixelsize": 130.0
    }


# --- replay of Filter steps from the metadata -----------------------------


def test_recipe_from_metadata_before_steps_were_recorded(locs):
    """Old Filter metadata (ranges only) replays as strict ranges."""
    info = [
        {"Generated by": "Picasso Localize"},
        {
            "Generated by": "Picasso v0.11 Filter",
            "sx": [0.8, 1.2],
            "bg": [0, 10],
            "Removed columns": ["lpx", "len"],
        },
        {"Generated by": "Picasso v0.11 Filter", "photons": [500.0, 3e3]},
    ]
    steps, to_remove, missing = lib.extract_filter_recipe(info, locs.columns)
    assert steps == ["0.8 < sx < 1.2", "500.0 < photons < 3000.0"]
    assert to_remove == ["lpx"]
    assert missing == ["bg", "len"]
    filtered, new_info, missing = lib.apply_filter_steps(locs, recipe=info)
    assert filtered["frame"].tolist() == [5, 30]
    assert "lpx" not in filtered
    entry = new_info[-1]
    ranges = {col: entry[col] for col in locs.columns if col in entry}
    assert ranges == {"sx": [0.8, 1.2], "photons": [500.0, 3000.0]}
    assert entry["Removed columns"] == ["lpx"]


def test_recorded_steps_replay_in_order(locs):
    """Statistics use the localizations left by the previous steps, so
    the recorded order decides the result."""
    for steps, frames in (
        # sx: photons 100, 600, NaN, 2000 -> median 600 -> 100
        (["sx < 1.2", "photons < median(photons)"], [0]),
        # median 750 -> 100, 600 -> sx keeps both
        (["photons < median(photons)", "sx < 1.2"], [0, 5]),
    ):
        info = [{"Generated by": "Picasso Filter", "Filter steps": steps}]
        filtered, *_ = lib.apply_filter_steps(locs, recipe=info)
        assert filtered["frame"].tolist() == frames


def test_recorded_steps_replace_the_ranges_of_their_dict(locs):
    """A dict with steps also holds its ranges for older Picasso; the
    ranges must not be applied a second time, nor change the
    statistics."""
    info = [
        {
            "Generated by": "Picasso Filter",
            "photons": [0.0, 1000.0],
            "Filter steps": [
                "photons < median(photons)",
                "0.0 < photons < 1000.0",
            ],
        }
    ]
    steps, _, _ = lib.extract_filter_recipe(info, locs.columns)
    assert steps == info[0]["Filter steps"]
    # the median of all photons is 750; after the range first it is 600
    filtered, new_info, _ = lib.apply_filter_steps(locs, recipe=info)
    assert filtered["frame"].tolist() == [0, 5]
    entry = new_info[-1]
    ranges = {col: entry[col] for col in locs.columns if col in entry}
    assert ranges == {"photons": [0.0, 1000.0]}


def test_recipe_from_old_and_new_sessions(locs):
    """A file filtered by an old Picasso, then by a new one; Filter
    writes None for the columns it did not filter."""
    info = [
        {"Generated by": "Picasso Localize", "Pixelsize": 130},
        {
            "Generated by": "Picasso v0.11.4 Filter",
            "frame": None,
            "photons": [500, 3000],
            "sx": None,
        },
        {
            "Generated by": "Picasso v1.0.0 Filter",
            "frame": None,
            "sx": [0.0, 1.2],
            "Filter steps": ["sx < 1.2", "photons < median(photons)"],
        },
    ]
    steps, to_remove, missing = lib.extract_filter_recipe(info, locs.columns)
    assert steps == [
        "500.0 < photons < 3000.0",
        "sx < 1.2",
        "photons < median(photons)",
    ]
    assert to_remove == [] and missing == []
    # photons: 600, 900, 2000 -> sx: 600, 2000 -> median 1300 -> 600
    filtered, *_ = lib.apply_filter_steps(locs, recipe=info)
    assert filtered["frame"].tolist() == [5]


def test_steps_with_missing_columns_are_skipped(locs):
    info = [
        {
            "Generated by": "Picasso Filter",
            "Filter steps": ["z > 0", "photons > 500", "d_zcalib < sx"],
        }
    ]
    steps, _, missing = lib.extract_filter_recipe(info, locs.columns)
    assert steps == ["photons > 500"]
    assert missing == ["z", "d_zcalib"]


def test_replay_takes_pixel_size_of_the_filtered_locs(locs):
    info = [
        {"Pixelsize": 100},
        {
            "Generated by": "Picasso Filter",
            "Filter steps": ["lpx * pixelsize < 6"],
        },
    ]
    filtered, *_ = lib.apply_filter_steps(locs, recipe=info)
    assert filtered["frame"].tolist() == [10, 30]
    filtered, *_ = lib.apply_filter_steps(
        locs, [{"Pixelsize": 200}], recipe=info
    )
    assert filtered["frame"].tolist() == [30]


def test_invalid_recorded_step_raises(locs):
    info = [{"Generated by": "Picasso Filter", "Filter steps": ["sx >"]}]
    with pytest.raises(ExpressionError):
        lib.extract_filter_recipe(info, locs.columns)


def test_extract_filter_steps_ignores_recorded_steps(locs):
    info = [
        {
            "Generated by": "Picasso Filter",
            "sx": [0.5, 1.5],
            "Filter steps": ["photons > 1", "0.5 < sx < 1.5"],
        }
    ]
    ranges, to_remove, missing = lib.extract_filter_steps(info, locs.columns)
    assert ranges == {"sx": [0.5, 1.5]}
    assert to_remove == [] and missing == []


# --- command line ---------------------------------------------------------


@pytest.fixture
def locs_path(locs, tmp_path):
    """``locs`` saved with positions, precisions and finite photons."""
    locs = locs.assign(
        photons=locs["photons"].fillna(300), x=1.0, y=2.0, lpy=locs["lpx"]
    )
    path = tmp_path / "movie_locs.hdf5"
    info = [
        {
            "Generated by": "test",
            "Pixelsize": 100,
            "Width": 32,
            "Height": 32,
            "Frames": 40,
        }
    ]
    io.save_locs(str(path), locs, info)
    return path


def test_command_line(locs_path):
    cli._filter(
        str(locs_path),
        [" 0.8 < sx < 1.2 ", "lpx * pixelsize < 15", "sx < 1.15"],
    )
    out_path = str(locs_path).replace(".hdf5", "_filter.hdf5")
    filtered, info = io.load_locs(out_path)
    assert filtered["frame"].tolist() == [5, 20, 30]
    assert info[-1]["Filter steps"] == [
        "0.8 < sx < 1.2",
        "lpx * pixelsize < 15",
        "sx < 1.15",
    ]
    assert info[-1]["sx"] == [0.8, 1.2]
    locs, _ = io.load_locs(str(locs_path))
    replayed, *_ = lib.apply_filter_steps(locs, recipe=info)
    assert replayed["frame"].tolist() == [5, 20, 30]


def test_command_line_reports_invalid_condition(locs_path):
    with pytest.raises(SystemExit, match="Unknown name 'phtons'"):
        cli._filter(str(locs_path), ["phtons > 1"])
    out_path = str(locs_path).replace(".hdf5", "_filter.hdf5")
    assert not os.path.exists(out_path)


def test_command_line_arguments(locs_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["picasso", "filter", str(locs_path), "sx < 1.05"]
    )
    cli.main()
    out_path = str(locs_path).replace(".hdf5", "_filter.hdf5")
    filtered, _ = io.load_locs(out_path)
    assert filtered["frame"].tolist() == [0, 5, 20]


def test_command_line_needs_a_condition(locs_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["picasso", "filter", str(locs_path)])
    with pytest.raises(SystemExit):
        cli.main()
    assert "give at least one condition" in capsys.readouterr().err
