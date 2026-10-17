"""``picasso server`` listens only on this computer unless ``--network``
is given: the server has no login and its watcher runs custom commands.
Also, the watcher's custom command reaches the program it starts as
typed, with the path of the processed file in place of ``$FILENAME``,
and the local database of the localized files is written and read with
the standard library's ``sqlite3``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shlex
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def test_without_the_server_extra_says_how_to_install_it(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            None if name == "streamlit" else real_find_spec(name, *args)
        ),
    )
    with pytest.raises(SystemExit, match=r"picassosr\[server\]"):
        cli._start_server()


@pytest.fixture
def database(monkeypatch, tmp_path):
    """The local database in an empty home folder, written and read
    without SQLAlchemy, which Picasso does not depend on."""
    from picasso import localize

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)  # import fails
    return localize.db_filename()


@pytest.fixture
def helper(monkeypatch):
    """``picasso.server.helper``, imported as Streamlit runs it."""
    pytest.importorskip("streamlit")
    import picasso.server

    monkeypatch.syspath_prepend(os.path.dirname(picasso.server.__file__))
    return importlib.import_module("helper")


def _summary(filename, created, **columns):
    return {
        "filename": filename,
        "n_locs": 10,
        "photons_mean": 1000.5,
        "file_created": created,
        "entry_created": datetime(2026, 10, 17, 12, 0, 0, 123456),
        **columns,
    }


def test_summaries_are_read_back_with_their_types(database, helper):
    from picasso import localize

    localize._save_file_summary(_summary("b.raw", datetime(2026, 2, 1)))
    # a column that the table does not have yet, e.g., a new statistic
    localize._save_file_summary(
        _summary("a.raw", datetime(2026, 1, 1, 8, 30, 0, 5), nena_nm=4.5)
    )
    df = helper.fetch_db()
    assert df["filename"].tolist() == ["a.raw", "b.raw"]  # by date
    assert str(df["file_created"].dtype) == "datetime64[ns]"
    assert str(df["entry_created"].dtype) == "datetime64[ns]"
    assert df["file_created"].iloc[0] == datetime(2026, 1, 1, 8, 30, 0, 5)
    assert df["n_locs"].tolist() == [10, 10]
    assert df["nena_nm"].iloc[0] == 4.5
    assert np.isnan(df["nena_nm"].iloc[1])


def test_reads_and_extends_a_database_of_older_versions(database, helper):
    # as SQLAlchemy (Picasso < 1.0) created it: DATETIME columns, stored
    # as text with microseconds
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TABLE files (filename TEXT, n_locs BIGINT, "
            "photons_mean FLOAT, file_created DATETIME, "
            "entry_created DATETIME)"
        )
        connection.execute(
            "INSERT INTO files VALUES ('old.raw', 3, 2.5, "
            "'2025-05-06 07:08:09.101112', '2025-05-07 00:00:00.000000')"
        )
    from picasso import localize

    localize._save_file_summary(
        _summary("new.raw", datetime(2026, 3, 1), nena_nm=4.5)
    )
    df = helper.fetch_db()
    assert df["filename"].tolist() == ["old.raw", "new.raw"]
    assert df["file_created"].iloc[0] == datetime(2025, 5, 6, 7, 8, 9, 101112)
    assert df["entry_created"].iloc[0] == datetime(2025, 5, 7)
    assert np.isnan(df["nena_nm"].iloc[0])


def test_empty_database(database, helper):
    assert helper.fetch_db().empty
    assert helper.fetch_watcher().empty


def test_watchers_are_added_and_replaced(database, helper):
    for pid in (1, 2):
        watcher = pd.DataFrame(
            {
                "process id": [pid],
                "started": [datetime(2026, 10, 1, 9, pid)],
                "folder": ["/data"],
                "logfile": ["/data/log.txt"],
            }
        )
        helper.write_table(watcher, "watcher", if_exists="append")
    df = helper.fetch_watcher()
    assert df["process id"].tolist() == [1, 2]
    assert df["started"].iloc[1] == datetime(2026, 10, 1, 9, 2)
    helper.write_table(df.iloc[1:], "watcher", if_exists="replace")
    assert helper.fetch_watcher()["process id"].tolist() == [2]
