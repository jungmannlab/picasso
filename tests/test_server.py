"""``picasso server`` listens only on this computer unless ``--network``
is given: the server has no login and its watcher runs custom commands.
Also, the watcher's custom command reaches the program it starts as
typed, with the path of the processed file in place of ``$FILENAME``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from picasso import __main__ as cli


@pytest.fixture
def streamlit_argv(monkeypatch, tmp_path):
    """Run ``_start_server`` without starting Streamlit and return the
    arguments it would have started Streamlit with."""
    pytest.importorskip("streamlit")
    from streamlit.web import cli as stcli

    # _start_server writes ~/.streamlit/credentials.toml
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    started = []
    monkeypatch.setattr(stcli, "main", lambda: started.append(sys.argv) or 0)

    def run(*args):
        with pytest.raises(SystemExit):
            cli._start_server(*args)
        return started[-1]

    return run


def test_listens_on_localhost_by_default(streamlit_argv):
    assert "--server.address=localhost" in streamlit_argv()


def test_network_listens_on_all_interfaces(streamlit_argv, capsys):
    argv = streamlit_argv(True)
    assert not any(arg.startswith("--server.address") for arg in argv)
    assert "anyone who can reach this computer" in capsys.readouterr().out


def test_command_line_passes_the_flag(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "_start_server", called.append)
    monkeypatch.setattr("picasso.updater.check_and_notify", lambda *args: None)
    for argv, network in (
        (["picasso", "server"], False),
        (["picasso", "server", "--network"], True),
    ):
        monkeypatch.setattr("sys.argv", argv)
        cli.main()
        assert called[-1] is network


@pytest.fixture
def watcher(monkeypatch):
    """``picasso.server.watcher``, which imports its sibling modules as
    Streamlit runs it, i.e., with the server folder on the path."""
    pytest.importorskip("streamlit")
    import picasso.server

    monkeypatch.syspath_prepend(os.path.dirname(picasso.server.__file__))
    return importlib.import_module("picasso.server.watcher")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX splitting")
def test_custom_command_gets_each_path_as_one_argument(watcher):
    movie = '/data/my movie\'s "first" run.ome.tif'
    locs = "/data/my movie_locs.hdf5"
    command = (
        f"{shlex.quote(sys.executable)} -c "
        "'import sys; print(sys.argv[1:])' $FILENAME $LOCS"
    )
    (to_run,) = watcher.custom_commands(command, movie, [locs])
    result = subprocess.run(to_run, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == repr([movie, locs])


def test_custom_command_runs_once_per_localization_file(watcher):
    locs = ["m_pg_1_locs.hdf5", "m_pg_2_locs.hdf5"]
    commands = watcher.custom_commands("undrift $LOCS", "m.raw", locs)
    if sys.platform == "win32":
        assert commands == [
            "undrift m_pg_1_locs.hdf5",
            "undrift m_pg_2_locs.hdf5",
        ]
    else:
        assert commands == [["undrift", locs[0]], ["undrift", locs[1]]]
    # nothing to run it on
    assert watcher.custom_commands("undrift $LOCS", "m.raw", []) == []


def test_custom_command_without_placeholders_runs_once(watcher):
    locs = ["m_pg_1_locs.hdf5", "m_pg_2_locs.hdf5"]
    if sys.platform == "win32":
        assert watcher.custom_commands("notify.exe", "m.raw", locs) == [
            "notify.exe"
        ]
    else:
        assert watcher.custom_commands("echo done", "m.raw", locs) == [
            ["echo", "done"]
        ]


def test_custom_command_on_windows_keeps_backslashes(watcher, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(watcher.sys, "platform", "win32")
        (command,) = watcher.custom_commands(
            r"C:\Miniconda3\python.exe C:\Users\me\test.py $FILENAME $LOCS",
            r"C:\data\my movie.ome.tif",
            [r"C:\data\my movie_locs.hdf5"],
        )
    assert command == (
        r"C:\Miniconda3\python.exe C:\Users\me\test.py "
        r'"C:\data\my movie.ome.tif" "C:\data\my movie_locs.hdf5"'
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX quoting")
def test_watcher_runs_the_custom_command_after_a_file(
    watcher, monkeypatch, tmp_path
):
    movie = tmp_path / "my movie.raw"
    movie.write_bytes(b"")
    received = tmp_path / "received.txt"
    logfile = tmp_path / "watcher.log"

    class Stop(Exception):
        """Ends the watcher at its first pause."""

    def stop(minutes):
        raise Stop

    def localize(args):
        # what _localize returns: the localization files it saved
        return [args.files.replace(".raw", f"{args.suffix}_locs.hdf5")]

    monkeypatch.setattr(watcher, "_localize", localize)
    monkeypatch.setattr(watcher, "wait_for_completion", lambda file: [])
    monkeypatch.setattr(watcher, "time", SimpleNamespace(sleep=stop))
    script = (
        "import sys; open(sys.argv[1], 'a').write(sys.argv[2] + '|' "
        "+ sys.argv[3] + chr(10)); sys.exit(3)"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} "
        f"{shlex.quote(str(received))} $FILENAME $LOCS"
    )
    with pytest.raises(Stop):
        watcher.check_new_and_process(
            [{"suffix": "_a"}, {"suffix": "_b"}],
            str(tmp_path),
            command,
            str(logfile),
            [],
            1,
        )
    assert received.read_text().splitlines() == [
        f"{movie}|{tmp_path / 'my movie_a_locs.hdf5'}",
        f"{movie}|{tmp_path / 'my movie_b_locs.hdf5'}",
    ]
    log = logfile.read_text()
    assert f"files in {tmp_path}." in log
    assert "Processing group _b" in log
    assert log.count("exited with code 3") == 2
