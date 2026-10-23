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
    # datetimes in any unit (ns in pandas 2, us inferred in pandas 3)
    assert df["file_created"].dtype.kind == "M"
    assert df["entry_created"].dtype.kind == "M"
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


# ---------------------------------------------------------------------------
# The data functions of the pages (no Streamlit app is run)
# ---------------------------------------------------------------------------


@pytest.fixture
def page(helper):
    """Import a page of Picasso Server (``history``, ``compare``,
    ``status``, ...) as Streamlit runs it."""
    return importlib.import_module


def _touch(folder, *names):
    for name in names:
        (folder / name).write_text("")


class TestFileFamily:
    def test_files_derived_from_the_movie(self, helper, tmp_path):
        _touch(
            tmp_path,
            "movie.raw",
            "movie_locs.hdf5",
            "movie_locs_undrift.hdf5",
            "movie_pg_1_locs.hdf5",
            "movie_locs.yaml",
            "movie10_locs.hdf5",  # another movie
            "other_locs.hdf5",
        )
        files, folder = helper.get_file_family(str(tmp_path / "movie.raw"))
        assert files == [
            "movie_locs.hdf5",
            "movie_locs_undrift.hdf5",
            "movie_pg_1_locs.hdf5",
        ]
        assert folder == str(tmp_path)

    def test_ome_tif(self, helper, tmp_path):
        # Picasso names the localizations after the movie without .tif
        _touch(tmp_path, "Pos0.ome_locs.hdf5", "Pos0_1.ome_locs.hdf5")
        files, _ = helper.get_file_family(str(tmp_path / "Pos0.ome.tif"))
        assert files == ["Pos0.ome_locs.hdf5"]

    def test_dots_in_the_name(self, helper, tmp_path):
        _touch(
            tmp_path, "2026.01.05_cell_locs.hdf5", "2026.01.05_dish_locs.hdf5"
        )
        files, _ = helper.get_file_family(
            str(tmp_path / "2026.01.05_cell.raw")
        )
        assert files == ["2026.01.05_cell_locs.hdf5"]

    def test_a_localization_file_itself(self, helper, tmp_path):
        _touch(tmp_path, "locs.hdf5", "locs_undrift.hdf5", "locs2.hdf5")
        files, _ = helper.get_file_family(str(tmp_path / "locs.hdf5"))
        assert files == ["locs.hdf5", "locs_undrift.hdf5"]

    def test_moved_folder(self, helper, tmp_path):
        with pytest.raises(FileNotFoundError):
            helper.get_file_family(str(tmp_path / "gone" / "movie.raw"))


class TestTagFilter:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("a", (["a"], [])),
            ("a&b", (["a", "b"], [])),
            ("a&b|c", (["a", "b"], ["c"])),
            ("a|b|c", (["a"], ["b", "c"])),
            ("a|b&c", (["a", "c"], ["b"])),
            ("|c", ([], ["c"])),
            ("a|", (["a"], [])),
            ("&", ([], [])),
        ],
    )
    def test_parse_input(self, page, text, expected):
        assert page("history").parse_input(text) == expected

    NAMES = ["cell_r1.raw", "cell_r2.raw", "dish_r1.raw", "dish_r2.raw"]

    @pytest.mark.parametrize(
        "to_add, to_exclude, expected",
        [
            (["cell"], [], ["cell_r1.raw", "cell_r2.raw"]),
            (
                ["cell", "r1"],
                [],
                ["cell_r1.raw", "cell_r2.raw", "dish_r1.raw"],
            ),
            (["cell"], ["r2"], ["cell_r1.raw"]),
            ([], ["cell"], ["dish_r1.raw", "dish_r2.raw"]),
            ([], [], NAMES),
        ],
    )
    def test_filter_by_tags(self, page, to_add, to_exclude, expected):
        df = pd.DataFrame({"filename": self.NAMES})
        filtered = page("history").filter_by_tags(df, to_add, to_exclude)
        assert filtered["filename"].tolist() == expected


class TestFilterEntries:
    @pytest.fixture
    def entries(self):
        return pd.DataFrame(
            {
                "filename": ["a_r1.raw", "b_r1.raw", "c_r2.raw", "d_r1.raw"],
                "file_created": [
                    datetime(2026, 1, 1, 9),
                    datetime(2026, 1, 2, 23, 59),
                    datetime(2026, 1, 3, 0, 0),
                    datetime(2026, 1, 4, 14),
                ],
            }
        )

    def test_both_days_are_included(self, page, entries):
        from datetime import date

        filtered = page("history").filter_entries(
            entries, date(2026, 1, 2), date(2026, 1, 3)
        )
        assert filtered["filename"].tolist() == ["b_r1.raw", "c_r2.raw"]

    def test_the_whole_range_keeps_everything(self, page, entries):
        filtered = page("history").filter_entries(
            entries,
            entries["file_created"].min().date(),
            entries["file_created"].max().date(),
        )
        assert len(filtered) == len(entries)

    def test_tags_and_dates(self, page, entries):
        from datetime import date

        filtered = page("history").filter_entries(
            entries, date(2026, 1, 2), date(2026, 1, 4), "r1 | b"
        )
        assert filtered["filename"].tolist() == ["d_r1.raw"]


@pytest.mark.parametrize(
    "groups, expected",
    [
        ([""], "None"),  # nothing entered
        (["dish"], "None"),
        (["cell", "r1"], "r1"),  # the last match
        (["cell", ""], "cell"),
    ],
)
def test_check_group(page, groups, expected):
    assert page("history").check_group("cell_r1.raw", groups) == expected


def test_convert_df(page):
    df = pd.DataFrame({"filename": ["a.raw"], "n_locs": [3]})
    csv = page("history").convert_df(df)
    assert csv.decode("utf-8").splitlines() == [
        ",filename,n_locs",
        "0,a.raw,3",
    ]


class TestComparePlotData:
    def test_locs_per_frame_counts_empty_frames(self, page):
        locs = pd.DataFrame({"frame": [0, 0, 0, 2, 5, 5]})
        counts = page("compare").locs_per_frame(locs, smooth=1)
        assert counts["frame"].tolist() == [0, 1, 2, 3, 4, 5]
        assert counts["count"].tolist() == [3, 0, 1, 0, 0, 2]

    def test_locs_per_frame_is_smoothed(self, page):
        locs = pd.DataFrame({"frame": [0, 0, 1, 1, 1, 1, 3, 3]})
        counts = page("compare").locs_per_frame(locs, smooth=2)
        assert np.isnan(counts["count"].iloc[0])
        assert counts["count"].tolist()[1:] == [3.0, 2.0, 1.0]

    def test_locs_per_frame_of_no_localizations(self, page):
        counts = page("compare").locs_per_frame(
            pd.DataFrame({"frame": []}), smooth=10
        )
        assert counts.empty

    def test_histograms_share_the_bins(self, page):
        values = {
            "a.hdf5": np.array([0.5, 1.5, 1.6, 3.9]),
            "b.hdf5": np.array([2.5, 9.0]),  # 9 is outside the range
        }
        table = page("compare").histograms(values, "sx", 0.0, 4.0, 4)
        a = table[table["file"] == "a.hdf5"]
        b = table[table["file"] == "b.hdf5"]
        assert a["sx"].tolist() == [0.5, 1.5, 2.5, 3.5]  # bin centers
        assert a["count"].tolist() == [1, 2, 0, 1]
        assert b["count"].tolist() == [0, 0, 1, 0]

    def test_load_file_names_the_file(self, page, locs, info, tmp_path):
        from picasso import io

        path = tmp_path / "movie_locs.hdf5"
        io.save_locs(str(path), locs, info)
        loaded, _ = page("compare").load_file(str(path))
        assert len(loaded) == len(locs)
        assert (loaded["file"] == "movie_locs.hdf5").all()


class TestStatus:
    def test_check_file(self, page, tmp_path):
        status = page("status")
        _touch(tmp_path, "movie.raw", "movie_locs.hdf5", "other.raw")
        assert status.check_file(str(tmp_path / "movie.raw"))
        assert not status.check_file(str(tmp_path / "other.raw"))
        assert status.check_file(str(tmp_path / "movie.ome.tif")) is False

    def test_check_file_of_an_ome_tif(self, page, tmp_path):
        _touch(tmp_path, "Pos0.ome_locs.hdf5")
        assert page("status").check_file(str(tmp_path / "Pos0.ome.tif"))

    def test_escape_markdown(self, page):
        escape = page("status").escape_markdown
        assert escape("C:\\Users\\me") == "C:\\\\Users\\\\me"
        assert escape("a_b*[c](d).e") == "a\\_b\\*\\[c\\]\\(d\\)\\.e"
        assert escape("plain") == "plain"
