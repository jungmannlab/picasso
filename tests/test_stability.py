"""Test the stability policy (``docs/versioning.rst``): the public Python
API and the command line against their snapshots, the deprecation cycle
of ``picasso.deprecation`` and the names renamed in Picasso 1.0.

A failing snapshot test means a public name was added, renamed or
removed. Removing or renaming one needs a deprecation cycle first; once
the change is deliberate, record it with
``python tests/regen_stability_snapshots.py``, which refuses to record
a removal that skipped the cycle.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import json
import sys
import types
import warnings

import pytest

from picasso import __version__, deprecation, lib
from picasso import __main__ as cli

from tests import regen_stability_snapshots as regen
from tests.regen_stability_snapshots import (
    API_SNAPSHOT,
    CLI_SNAPSHOT,
    DEPRECATED_SNAPSHOT,
    command_line,
    deprecated_names,
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

    def test_deprecations_are_recorded(self):
        # the snapshot of the deprecated names is what lets
        # regen_stability_snapshots record their removal later
        recorded = json.loads(DEPRECATED_SNAPSHOT.read_text())
        current = deprecated_names(public_api(), command_line())
        assert recorded == current, (
            "The deprecated public names changed: recorded "
            f"{recorded}, now {current}.\nRun {REGEN}."
        )


class TestRecordingRemovals:
    """``regen_stability_snapshots`` records the removal of a public
    name only once its deprecation cycle is over."""

    RECORDED = {
        "api": {
            "picasso.io": ["load_locs", "old_function"],
            "picasso.old": ["f"],
        },
        "cli": {"render": ["--old", "-s"], "old-command": ["-x"]},
        "deprecated": {
            "picasso.io.old_function": "1.3",
            "picasso render --old": "1.3",
        },
    }

    def refused(self, version, api=None, cli=None):
        recorded = self.RECORDED
        return regen.refused_removals(
            recorded,
            recorded["api"] if api is None else api,
            recorded["cli"] if cli is None else cli,
            version,
        )

    def test_additions_are_recorded(self):
        api = dict(self.RECORDED["api"], **{"picasso.new": ["g"]})
        cli = dict(self.RECORDED["cli"], **{"new-command": []})
        assert self.refused("1.1.0", api, cli) == []

    def test_removal_without_deprecation_is_refused(self):
        api = dict(self.RECORDED["api"], **{"picasso.io": ["old_function"]})
        assert self.refused("1.3.0", api) == [
            "picasso.io.load_locs was removed without a deprecation"
        ]

    @pytest.mark.parametrize(
        "version,allowed",
        [
            ("1.2.4", False),
            ("1.3.0a1", True),  # during the release cycle of 1.3
            ("1.3.0", True),
            ("2.0.0", True),
        ],
    )
    def test_deprecated_name_waits_for_its_version(self, version, allowed):
        api = dict(self.RECORDED["api"], **{"picasso.io": ["load_locs"]})
        refused = self.refused(version, api)
        if allowed:
            assert refused == []
        else:
            assert refused == [
                "picasso.io.old_function is deprecated, but may be removed "
                f"in Picasso 1.3, not {version}"
            ]

    def test_renamed_option(self):
        cli = dict(self.RECORDED["cli"], render=["--new", "-s"])
        assert self.refused("1.3.0", cli=cli) == []
        assert len(self.refused("1.2.0", cli=cli)) == 1

    def test_removed_module_or_command_is_one_removal(self):
        api = {"picasso.io": self.RECORDED["api"]["picasso.io"]}
        cli = {"render": self.RECORDED["cli"]["render"]}
        assert self.refused("1.3.0", api, cli) == [
            "picasso.old was removed without a deprecation",
            "picasso old-command was removed without a deprecation",
        ]

    def test_deprecations_are_found_by_public_name(self, monkeypatch):
        monkeypatch.setattr(deprecation, "REGISTRY", [])

        @deprecation.deprecated(since="1.1", remove_in="1.3")
        def old_function():
            pass

        deprecation.register(
            "picasso render --old", since="1.1", remove_in="1.3"
        )
        # public in another module than the one defining it, as the
        # functions of picasso.render's submodules
        package = types.ModuleType("picasso_test_package")
        package.old_function = old_function
        monkeypatch.setitem(sys.modules, "picasso_test_package", package)
        api = {"picasso_test_package": ["old_function"]}
        cli = {"render": ["--old", "-s"]}
        assert deprecated_names(api, cli) == {
            "picasso_test_package.old_function": "1.3",
            "picasso render --old": "1.3",
        }

    def test_refused_removal_records_nothing(self, tmp_path, monkeypatch):
        api, cli = public_api(), command_line()
        api["picasso.io"] = api["picasso.io"] + ["gone"]
        paths = {
            "API_SNAPSHOT": tmp_path / "api.json",
            "CLI_SNAPSHOT": tmp_path / "cli.json",
            "DEPRECATED_SNAPSHOT": tmp_path / "deprecated.json",
        }
        for (name, path), data in zip(paths.items(), (api, cli, {})):
            path.write_text(json.dumps(data))
            monkeypatch.setattr(regen, name, path)
        monkeypatch.setattr(regen, "SNAPSHOTS", tmp_path)
        recorded = {path: path.read_text() for path in paths.values()}
        monkeypatch.setattr(sys, "argv", ["regen_stability_snapshots.py"])
        with pytest.raises(SystemExit, match="picasso.io.gone was removed"):
            regen.main()
        assert {path: path.read_text() for path in paths.values()} == recorded


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
        # the registry holds the entries of the imported modules; the
        # command line (``cli``, imported above) registers its renames
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
        old_helper_deprecation = deprecation.register(
            "picasso.lib.old_helper", since="1.1", remove_in="1.3"
        )

        def old_helper():
            old_helper_deprecation.warn()

        with pytest.warns(FutureWarning) as caught:
            old_helper()
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
        argv = cli._current_names(["pair-correlation", "f.hdf5", "--rmax=300"])
        assert argv == ["pair-correlation", "f.hdf5", "--r-max=300"]

    def test_arguments_after_double_dash_are_kept(self, capsys):
        argv = cli._current_names(["pair-correlation", "--", "--rmax"])
        assert argv == ["pair-correlation", "--", "--rmax"]

    def test_current_names_print_nothing(self, capsys):
        argv = ["pair-correlation", "f.hdf5", "--r-max", "300"]
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cli.main()
        assert called == [("f.hdf5", 2.0, 50.0)]
