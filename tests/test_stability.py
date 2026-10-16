"""Test the stability policy (``docs/versioning.rst``): the public Python
API and the command line against their snapshots, the deprecation cycle
of ``picasso.deprecation`` and the names renamed in Picasso 1.0.

A failing snapshot test means a public name was added, renamed or
removed. Removing or renaming one needs a deprecation cycle first; once
the change is deliberate, record it with
``python tests/regen_stability_snapshots.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from picasso import __version__, deprecation, lib
from picasso import __main__ as cli

from tests.regen_stability_snapshots import (
    API_SNAPSHOT,
    CLI_SNAPSHOT,
    command_line,
    public_api,
)

REGEN = "python tests/regen_stability_snapshots.py"


def _differences(recorded: dict, current: dict, what: str) -> list[str]:
    """What changed between two ``{group: [names]}`` mappings."""
    lines = []
    for group in sorted(set(recorded) | set(current)):
        before = set(recorded.get(group, []))
        now = set(current.get(group, []))
        if group not in current:
            lines.append(f"{what} {group} is gone")
        elif group not in recorded:
            lines.append(f"{what} {group} is new")
        for name in sorted(before - now):
            lines.append(f"{group}: {name} was removed")
        for name in sorted(now - before):
            lines.append(f"{group}: {name} is new")
    return lines


class TestSnapshots:
    def test_public_api_is_unchanged(self):
        recorded = json.loads(API_SNAPSHOT.read_text())
        changes = _differences(recorded, public_api(), "module")
        assert not changes, (
            "The public API changed (removing or renaming a public name "
            "needs a deprecation cycle, see docs/versioning.rst):\n  "
            + "\n  ".join(changes)
            + f"\nIf this is deliberate, run {REGEN}."
        )

    def test_command_line_is_unchanged(self):
        recorded = json.loads(CLI_SNAPSHOT.read_text())
        changes = _differences(recorded, command_line(), "command")
        assert not changes, (
            "The command line changed (removing or renaming a command or "
            "an option needs a deprecation cycle, see docs/versioning.rst):"
            "\n  " + "\n  ".join(changes) + f"\nIf this is deliberate, run "
            f"{REGEN}."
        )

    def test_snapshot_lists_the_modules_of_the_api_pages(self):
        recorded = json.loads(API_SNAPSHOT.read_text())
        assert "picasso.io" in recorded
        assert "picasso.fitting.gaussfit" in recorded  # recursive
        assert "load_locs" in recorded["picasso.io"]


class TestDeprecationCycle:
    def test_earliest_removal_is_two_minor_releases_later(self):
        assert deprecation.earliest_removal("1.1") == "1.3"
        assert deprecation.removal_allowed("1.1", "1.3")
        assert not deprecation.removal_allowed("1.1", "1.2")
        assert deprecation.removal_allowed("1.9", "2.0")  # a major release
        assert deprecation.removal_allowed("0.11", "1.0")

    def test_too_early_a_removal_is_refused(self):
        with pytest.raises(ValueError, match="1.3 at the earliest"):
            deprecation.register("x", since="1.1", remove_in="1.2")

    @pytest.mark.parametrize(
        "version,due",
        [
            ("1.2.5", False),
            ("1.3.0a1", False),  # its pre-releases may still contain it
            ("1.3.0rc2", False),
            ("1.3.0", True),
            ("1.4.1", True),
            ("2.0.0", True),
        ],
    )
    def test_removal_due(self, version, due):
        entry = deprecation.Deprecation("x", since="1.1", remove_in="1.3")
        assert deprecation.removal_due(entry, version) is due

    def test_no_removal_is_overdue(self):
        # the modules that register deprecations, imported so that their
        # entries are in the registry
        from picasso import gausslq, gaussmle, localize  # noqa: F401

        overdue = [
            f"{entry.name} (deprecated in {entry.since}, removed in "
            f"{entry.remove_in})"
            for entry in deprecation.REGISTRY
            if deprecation.removal_due(entry, __version__)
        ]
        assert (
            not overdue
        ), f"Picasso {__version__} must no longer contain:\n  " + "\n  ".join(
            overdue
        )

    def test_every_deprecation_follows_the_cycle(self):
        from picasso import gausslq, gaussmle, localize  # noqa: F401

        assert deprecation.REGISTRY
        for entry in deprecation.REGISTRY:
            assert deprecation.removal_allowed(entry.since, entry.remove_in)


class TestDeprecatedDecorator:
    @staticmethod
    def _function():
        @deprecation.deprecated(
            since="1.1", remove_in="1.3", alternative="new_function"
        )
        def old_function(x):
            """Double x.

            Parameters
            ----------
            x : float
                A number.
            """
            return 2 * x

        return old_function

    def test_warns_at_the_caller(self):
        old_function = self._function()
        with pytest.warns(FutureWarning) as caught:
            assert old_function(2) == 4
        assert caught[0].filename == __file__
        message = str(caught[0].message)
        assert "old_function is deprecated since Picasso 1.1" in message
        assert "removed in Picasso 1.3" in message
        assert "use new_function instead" in message

    def test_docstring_note_follows_the_summary(self):
        doc = self._function().__doc__
        assert doc.startswith("Double x.\n\n.. deprecated:: 1.1\n")
        assert "Removed in Picasso 1.3. Use new_function instead." in doc
        assert "Parameters\n----------" in doc

    def test_class(self):
        @deprecation.deprecated(since="1.1", remove_in="1.3")
        class Old:
            """An old class."""

            def __init__(self, value):
                self.value = value

        with pytest.warns(FutureWarning, match="Old is deprecated"):
            assert Old(3).value == 3
        assert ".. deprecated:: 1.1" in Old.__doc__

    def test_registered(self):
        old_function = self._function()
        assert any(
            entry.name.endswith("old_function")
            for entry in deprecation.REGISTRY
        )
        assert old_function.__name__ == "old_function"


class TestExistingDeprecations:
    def test_shown_to_scripts_at_the_caller(self):
        # a FutureWarning, which Python shows by default (unlike a
        # DeprecationWarning raised inside a library)
        rec = pd.DataFrame({"x": [0.0, 1.0]}).to_records(index=False)
        with pytest.warns(FutureWarning) as caught:
            lib.append_to_rec(rec, np.array([1.0, 2.0]), "y")
        assert caught[0].filename == __file__

    def test_legacy_helper_warns_at_the_caller(self):
        def deprecated_function():
            lib.deprecation_warning("gone soon")

        with pytest.warns(FutureWarning, match="gone soon") as caught:
            deprecated_function()
        assert caught[0].filename == __file__


class TestRenamedCommandLine:
    def test_old_command_is_renamed_with_a_notice(self, capsys):
        argv = cli._current_names(["pc", "locs.hdf5", "--binsize", "2"])
        assert argv == ["pair-correlation", "locs.hdf5", "--bin-size", "2"]
        err = capsys.readouterr().err
        assert "picasso pc is deprecated since Picasso 1.0" in err
        assert "use picasso pair-correlation instead" in err
        assert "use --bin-size instead" in err

    def test_option_with_equals_sign(self, capsys):
        argv = cli._current_names(["cbc", "f.hdf5", "--rmax=300"])
        assert argv == ["cbc", "f.hdf5", "--r-max=300"]

    def test_arguments_after_double_dash_are_kept(self, capsys):
        argv = cli._current_names(["cbc", "--", "--rmax"])
        assert argv == ["cbc", "--", "--rmax"]

    def test_current_names_print_nothing(self, capsys):
        argv = ["cbc", "f.hdf5", "--r-max", "300"]
        assert cli._current_names(argv) == argv
        assert capsys.readouterr().err == ""

    def test_old_and_new_names_parse_alike(self, capsys):
        parser, _ = cli._build_parser()
        old = parser.parse_args(
            cli._current_names(
                ["clusterfilter", "f.hdf5", "-c", "c.hdf5", "--minval", "1"]
            )
        )
        new = parser.parse_args(
            ["cluster-filter", "f.hdf5", "-c", "c.hdf5", "--min-value", "1"]
        )
        assert vars(old) == vars(new)

    def test_old_names_are_no_current_names(self):
        commands = command_line()
        for old, (new, _) in cli._RENAMED_COMMANDS.items():
            assert old not in commands
            assert new in commands
        for command, options in cli._RENAMED_OPTIONS.items():
            for old, (new, _) in options.items():
                assert old not in commands[command]
                assert new in commands[command]

    def test_main_accepts_an_old_name(self, monkeypatch, capsys):
        called = []
        monkeypatch.setattr(
            cli, "_pair_correlation", lambda *args: called.append(args)
        )
        monkeypatch.setattr(
            "sys.argv",
            ["picasso", "pc", "f.hdf5", "--binsize", "2", "--rmax", "50"],
        )
        monkeypatch.setattr(
            "picasso.updater.check_and_notify", lambda *args: None
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cli.main()
        assert called == [("f.hdf5", 2.0, 50.0)]
