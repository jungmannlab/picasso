"""Test ``picasso.plugins`` — the Qt-free half of the plugin system: the
hooks that let a plugin extend Picasso's Python API (``PICASSO_API``) and
its command line (``register_cli``), and the enable gate that governs both.

Plugins are written as temporary ``.py`` files into a ``tmp_path`` and the
loader is pointed there by monkeypatching ``picasso.io.plugins_directory``,
matching the approach in ``test_plugins_loader.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import types

import pytest

from picasso import io
from picasso import plugins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_plugins_dir(tmp_path, monkeypatch):
    """Point the loader at an empty temporary plugins folder.

    Also clears the module cache around every test so that files reused
    across tests are never served from a previous test's import.
    """
    directory = tmp_path / "plugins"
    directory.mkdir()
    monkeypatch.setattr(io, "plugins_directory", lambda: str(directory))
    plugins.clear_module_cache()
    yield directory
    plugins.clear_module_cache()


def _write(directory, filename, body):
    path = os.path.join(str(directory), filename)
    with open(path, "w") as f:
        f.write(textwrap.dedent(body))
    return path


def _enable(filename):
    plugins.set_enabled(plugins.load_state(), filename, True)


API_PLUGIN = """
    def double(x):
        return 2 * x

    PICASSO_API = {"double": double}
"""

CLI_PLUGIN = """
    def _run(args):
        print(args.value)

    def register_cli(subparsers):
        parser = subparsers.add_parser("demo-cmd", help="demo")
        parser.add_argument("value")
        parser.set_defaults(func=_run)
"""


def _parser():
    """A stand-in for the ``picasso`` parser with one built-in command."""
    parser = argparse.ArgumentParser("picasso")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("localize")
    return parser, subparsers


# ---------------------------------------------------------------------------
# The enable gate applies outside the GUI too
# ---------------------------------------------------------------------------


class TestEnableGate:
    def test_disabled_plugin_contributes_nothing(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _write(_isolated_plugins_dir, "cli.py", CLI_PLUGIN)

        parser, subparsers = _parser()

        assert plugins.load_api_plugins() == {}
        assert plugins.register_cli_plugins(subparsers) == set()
        assert "demo-cmd" not in subparsers.choices

    def test_disabled_plugin_is_never_imported(self, _isolated_plugins_dir):
        """The gate is about *executing* code, not about hiding names."""
        marker = os.path.join(str(_isolated_plugins_dir), "ran.txt")
        _write(
            _isolated_plugins_dir,
            "sideeffect.py",
            f"""
            with open({marker!r}, "w") as f:
                f.write("ran")
            """,
        )

        plugins.load_api_plugins()

        assert not os.path.exists(marker)

    def test_enabling_makes_it_load(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _enable("api.py")

        assert plugins.load_api_plugins()["double"](3) == 6

    def test_disabled_files_are_reported(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _write(_isolated_plugins_dir, "cli.py", CLI_PLUGIN)
        _enable("api.py")

        assert plugins.disabled_plugin_files() == ["cli.py"]
        assert [
            os.path.basename(p) for p in plugins.enabled_plugin_files()
        ] == ["api.py"]


# ---------------------------------------------------------------------------
# Python API plugins
# ---------------------------------------------------------------------------


class TestApiPlugins:
    def test_exports_are_merged(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "a.py", API_PLUGIN)
        _write(
            _isolated_plugins_dir,
            "b.py",
            """
            PICASSO_API = {"triple": lambda x: 3 * x}
            """,
        )
        _enable("a.py")
        _enable("b.py")

        exports = plugins.load_api_plugins()

        assert sorted(exports) == ["double", "triple"]

    def test_callable_picasso_api_is_called(self, _isolated_plugins_dir):
        _write(
            _isolated_plugins_dir,
            "lazy.py",
            """
            def PICASSO_API():
                return {"lazy": 42}
            """,
        )
        _enable("lazy.py")

        assert plugins.load_api_plugins()["lazy"] == 42

    def test_first_plugin_wins_a_name_collision(self, _isolated_plugins_dir):
        _write(
            _isolated_plugins_dir,
            "a.py",
            """
            PICASSO_API = {"shared": "from a"}
            """,
        )
        _write(
            _isolated_plugins_dir,
            "b.py",
            """
            PICASSO_API = {"shared": "from b"}
            """,
        )
        _enable("a.py")
        _enable("b.py")

        assert plugins.load_api_plugins()["shared"] == "from a"

    @pytest.mark.parametrize(
        "body",
        [
            'PICASSO_API = ["not", "a", "dict"]',
            'PICASSO_API = {1: "not a name"}',
            'PICASSO_API = {"not an identifier": 1}',
            'def PICASSO_API():\n    raise RuntimeError("boom")',
        ],
    )
    def test_malformed_exports_are_skipped(
        self, _isolated_plugins_dir, body, caplog
    ):
        _write(_isolated_plugins_dir, "bad.py", body)
        _write(_isolated_plugins_dir, "good.py", API_PLUGIN)
        _enable("bad.py")
        _enable("good.py")

        exports = plugins.load_api_plugins()

        assert sorted(exports) == ["double"]
        assert "bad.py" in caplog.text

    def test_broken_plugin_does_not_stop_the_others(
        self, _isolated_plugins_dir, caplog
    ):
        _write(_isolated_plugins_dir, "boom.py", "raise RuntimeError('boom')")
        _write(_isolated_plugins_dir, "good.py", API_PLUGIN)
        _enable("boom.py")
        _enable("good.py")

        assert sorted(plugins.load_api_plugins()) == ["double"]
        assert "boom.py" in caplog.text

    def test_api_namespace_resolves_and_refreshes(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _enable("api.py")
        namespace = plugins._Api()

        assert namespace.double(4) == 8
        assert "double" in namespace
        with pytest.raises(AttributeError, match="double"):
            namespace.nosuchthing

        _write(
            _isolated_plugins_dir,
            "more.py",
            'PICASSO_API = {"extra": 1}',
        )
        _enable("more.py")

        assert "extra" not in namespace  # cached
        namespace.refresh()
        assert namespace.extra == 1


# ---------------------------------------------------------------------------
# Command line plugins
# ---------------------------------------------------------------------------


class TestCliPlugins:
    def test_command_is_registered_and_routed(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "cli.py", CLI_PLUGIN)
        _enable("cli.py")
        parser, subparsers = _parser()

        commands = plugins.register_cli_plugins(subparsers)

        assert commands == {"demo-cmd"}
        args = parser.parse_args(["demo-cmd", "hello"])
        assert args.command == "demo-cmd"
        assert args.value == "hello"
        assert callable(args.func)

    def test_builtin_commands_cannot_be_shadowed(
        self, _isolated_plugins_dir, caplog
    ):
        _write(
            _isolated_plugins_dir,
            "greedy.py",
            """
            def register_cli(subparsers):
                subparsers.add_parser("localize")
            """,
        )
        _write(_isolated_plugins_dir, "cli.py", CLI_PLUGIN)
        _enable("greedy.py")
        _enable("cli.py")
        parser, subparsers = _parser()

        # The built-in survives, the well-behaved plugin still registers.
        assert plugins.register_cli_plugins(subparsers) == {"demo-cmd"}
        assert "already exists" in caplog.text
        assert parser.parse_args(["localize"]).command == "localize"

    def test_a_failing_plugin_leaves_no_half_command(
        self, _isolated_plugins_dir, caplog
    ):
        _write(
            _isolated_plugins_dir,
            "half.py",
            """
            def register_cli(subparsers):
                subparsers.add_parser("half-done")
                raise RuntimeError("boom")
            """,
        )
        _write(_isolated_plugins_dir, "cli.py", CLI_PLUGIN)
        _enable("half.py")
        _enable("cli.py")
        parser, subparsers = _parser()

        assert plugins.register_cli_plugins(subparsers) == {"demo-cmd"}
        assert "half-done" not in subparsers.choices
        assert "boom" in caplog.text

    @pytest.mark.parametrize(
        "name", ["Bad-Name", "--flag", "has space", "", "/etc/passwd"]
    )
    def test_implausible_command_names_are_rejected(
        self, _isolated_plugins_dir, name, caplog
    ):
        _write(
            _isolated_plugins_dir,
            "odd.py",
            f"""
            def register_cli(subparsers):
                subparsers.add_parser({name!r})
            """,
        )
        _enable("odd.py")
        _, subparsers = _parser()

        assert plugins.register_cli_plugins(subparsers) == set()
        assert "odd.py" in caplog.text

    def test_plugin_without_register_cli_is_fine(self, _isolated_plugins_dir):
        _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _enable("api.py")
        _, subparsers = _parser()

        assert plugins.register_cli_plugins(subparsers) == set()

    def test_reported_commands_do_not_touch_the_real_parser(
        self, _isolated_plugins_dir
    ):
        path = _write(_isolated_plugins_dir, "cli.py", CLI_PLUGIN)
        _enable("cli.py")
        module = plugins._load_module_from_path(path)
        _, subparsers = _parser()

        assert plugins.plugin_cli_commands(module) == ["demo-cmd"]
        assert "demo-cmd" not in subparsers.choices


# ---------------------------------------------------------------------------
# Reporting a plugin that will not load
# ---------------------------------------------------------------------------


class TestFailureReporting:
    """A broken plugin must not bury every command's output in a traceback."""

    def test_default_is_one_warning(
        self, _isolated_plugins_dir, capsys, caplog
    ):
        _write(_isolated_plugins_dir, "boom.py", "raise RuntimeError('boom')")
        _enable("boom.py")

        plugins.load_enabled_modules()

        captured = capsys.readouterr()
        assert captured.out == ""  # stdout stays clean for piping
        (record,) = caplog.records
        assert record.levelname == "WARNING"
        message = record.getMessage()
        assert "boom.py" in message
        assert "RuntimeError: boom" in message
        assert "picasso plugins list" in message
        assert record.exc_info is None  # no traceback

    def test_reported_once_per_process(self, _isolated_plugins_dir, caplog):
        """A failed import is not cached, so the report must dedup itself."""
        _write(_isolated_plugins_dir, "boom.py", "raise RuntimeError('boom')")
        _enable("boom.py")

        plugins.load_enabled_modules()
        caplog.clear()
        plugins.load_enabled_modules()

        assert caplog.records == []

    def test_verbose_prints_the_traceback(self, _isolated_plugins_dir, capsys):
        _write(_isolated_plugins_dir, "boom.py", "raise RuntimeError('boom')")
        _enable("boom.py")

        plugins.load_enabled_modules(verbose=True)

        captured = capsys.readouterr()
        assert "Failed to load plugin" in captured.out
        assert "Traceback" in captured.err

    def test_a_reload_reports_again(self, _isolated_plugins_dir, caplog):
        _write(_isolated_plugins_dir, "boom.py", "raise RuntimeError('boom')")
        _enable("boom.py")
        plugins.load_enabled_modules()
        caplog.clear()

        plugins.clear_module_cache()
        plugins.load_enabled_modules()

        assert "boom.py" in caplog.text


# ---------------------------------------------------------------------------
# All three hooks in one file
# ---------------------------------------------------------------------------


class TestCombinedPlugin:
    """One .py file may extend the GUI, the API and the CLI at once."""

    BODY = """
        def work(x):
            return x + 1

        class Plugin:
            def __init__(self, window):
                self.name = "render"
                self.window = window

            def execute(self):
                self.window.plugin_menu.addAction("combined")

        PICASSO_API = {"work": work}

        def register_cli(subparsers):
            parser = subparsers.add_parser("combined-cmd")
            parser.set_defaults(func=lambda args: None)
    """

    def test_one_file_serves_all_three(self, _isolated_plugins_dir):
        path = _write(_isolated_plugins_dir, "combined.py", self.BODY)
        _enable("combined.py")
        _, subparsers = _parser()

        assert plugins.load_api_plugins()["work"](1) == 2
        assert plugins.register_cli_plugins(subparsers) == {"combined-cmd"}

        # ...and the GUI half still loads, through the Qt-free code path the
        # GUI loader shares with the rest.
        module = plugins._load_module_from_path(path)
        assert module.Plugin.__name__ == "Plugin"
        assert plugins.plugin_cli_commands(module) == ["combined-cmd"]

    def test_gui_loader_accepts_a_plugin_with_no_gui_half(
        self, _isolated_plugins_dir, capsys
    ):
        """An API/CLI-only plugin is not an error for a GUI to load."""
        from picasso.gui import plugins_loader

        _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _enable("api.py")
        window = types.SimpleNamespace(plugin_menu=None)

        assert plugins_loader.load_plugins(window, "render") == []
        assert window.plugins == []
        captured = capsys.readouterr()
        assert "Failed" not in captured.out + captured.err


# ---------------------------------------------------------------------------
# Module cache
# ---------------------------------------------------------------------------


class TestModuleCache:
    def test_same_file_is_executed_once(self, _isolated_plugins_dir):
        marker = os.path.join(str(_isolated_plugins_dir), "count.txt")
        path = _write(
            _isolated_plugins_dir,
            "counter.py",
            f"""
            with open({marker!r}, "a") as f:
                f.write("x")
            PICASSO_API = {{"nothing": None}}
            """,
        )
        _enable("counter.py")

        plugins._load_module_from_path(path)
        plugins._load_module_from_path(path)

        assert open(marker).read() == "x"

    def test_module_is_registered_in_sys_modules(self, _isolated_plugins_dir):
        """Classes must be able to find their own module while being built.

        ``@dataclass``, ``typing.get_type_hints``, ``pickle`` and ``enum``
        all look ``sys.modules[cls.__module__]`` up, and fail on ``None``
        if the plugin module was never registered there.
        """
        path = _write(
            _isolated_plugins_dir,
            "dc.py",
            """
            from __future__ import annotations

            import dataclasses
            import typing

            @dataclasses.dataclass
            class Point:
                x: float
                y: float = 0.0

            PICASSO_API = {
                "make": lambda: Point(1.0, 2.0),
                "hints": lambda: sorted(typing.get_type_hints(Point)),
            }
            """,
        )
        _enable("dc.py")

        module = plugins._load_module_from_path(path)

        assert sys.modules[module.__name__] is module
        exports = plugins.load_api_plugins()
        assert exports["make"]() == module.Point(1.0, 2.0)
        assert exports["hints"]() == ["x", "y"]

    def test_a_failed_load_leaves_no_module_behind(
        self, _isolated_plugins_dir
    ):
        path = _write(
            _isolated_plugins_dir,
            "boom.py",
            """
            SOMETHING = 1
            raise RuntimeError("boom")
            """,
        )
        _enable("boom.py")

        with pytest.raises(RuntimeError, match="boom"):
            plugins._load_module_from_path(path)

        assert "picasso_plugin_boom" not in sys.modules

    def test_clearing_the_cache_unregisters_the_module(
        self, _isolated_plugins_dir
    ):
        path = _write(_isolated_plugins_dir, "api.py", API_PLUGIN)
        _enable("api.py")
        name = plugins._load_module_from_path(path).__name__
        assert name in sys.modules

        plugins.clear_module_cache()

        assert name not in sys.modules

    def test_edited_file_is_reloaded(self, _isolated_plugins_dir):
        """Same length, same second: only a real recompile picks this up."""
        _write(_isolated_plugins_dir, "v.py", 'PICASSO_API = {"v": 1}')
        _enable("v.py")
        assert plugins.load_api_plugins()["v"] == 1

        _write(_isolated_plugins_dir, "v.py", 'PICASSO_API = {"v": 2}')

        assert plugins.load_api_plugins()["v"] == 2


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


def test_importing_plugins_does_not_import_qt():
    """The CLI and scripts must be able to use plugins without a GUI stack."""
    code = (
        "import sys; import picasso.plugins; " "print('PyQt6' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
