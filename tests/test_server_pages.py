"""The pages of Picasso Server (Status, History, Compare, Preview and
Watcher), run with Streamlit's testing harness on a database in an empty
home folder and tiny localization files, and the watcher's scanning of a
folder and localizing of new movies.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import shutil
import sys
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from picasso import io, localize  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def home(monkeypatch, tmp_path):
    """An empty home folder, so that the pages read and write a fresh
    ``~/.picasso/app_0410.db``, and the server folder on the path, as
    Streamlit runs the pages."""
    import streamlit as st

    import picasso.server

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.syspath_prepend(os.path.dirname(picasso.server.__file__))
    # Streamlit installs the page it runs as ``__main__`` and leaves it
    # there; a process pool started later (spawn: Windows, macOS) re-runs
    # ``__main__`` in each worker, which then dies on ``import watcher``
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    st.cache_data.clear()
    yield home
    st.cache_data.clear()


@pytest.fixture
def run_page(home):
    """Run a page of Picasso Server and return the app."""

    def run(page: str) -> AppTest:
        app = AppTest.from_string(
            f"import {page}\n{page}.{page}()\n", default_timeout=60
        )
        return app.run()

    return run


@pytest.fixture
def module(home):
    """A module of Picasso Server, imported as Streamlit runs it."""
    return importlib.import_module


def _entry(movie, created: datetime, **columns) -> dict:
    """A database entry as Localize writes it."""
    return {
        "filename": os.path.normpath(str(movie)),
        "filename_hdf": os.path.splitext(str(movie))[0] + "_locs.hdf5",
        "n_locs": 100,
        "locs_frame": 1.0,
        "photons_mean": 1000.0,
        "frame_std": 10.0,
        "bg_mean": 50.0,
        "drift_x": 1.0,
        "drift_y": 2.0,
        "nena_nm": 5.0,
        "file_created": created,
        "entry_created": datetime(2026, 10, 17, 12),
        **columns,
    }


def _add_entries(*entries: dict) -> None:
    localize.save_file_summaries(pd.DataFrame(list(entries)))


def _widget(widgets, label: str):
    """The widget with a label starting with ``label``."""
    (widget,) = [_ for _ in widgets if _.label.startswith(label)]
    return widget


def _texts(app: AppTest) -> list[str]:
    return [_.value for _ in app.text]


def _charts(app: AppTest) -> list[dict]:
    return [json.loads(_.proto.spec) for _ in app.get("plotly_chart")]


def _title(chart: dict) -> str:
    return chart["layout"]["title"]["text"]


def _values(array) -> list:
    """The values of a plotted array, which Plotly may send encoded."""
    if isinstance(array, dict):
        array = np.frombuffer(
            base64.b64decode(array["bdata"]), dtype=array["dtype"]
        )
    return list(array)


@pytest.fixture
def movie(tmp_path, locs, info):
    """A movie with two localization files, and another movie whose name
    starts the same, in their folder."""
    folder = tmp_path / "data"
    folder.mkdir()
    movie = folder / "cell.raw"
    movie.write_bytes(b"")
    (folder / "cell10.raw").write_bytes(b"")
    io.save_locs(str(folder / "cell_locs.hdf5"), locs, info)
    io.save_locs(str(folder / "cell_locs_undrift.hdf5"), locs[::2], info)
    io.save_locs(str(folder / "cell10_locs.hdf5"), locs, info)
    return movie


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


def test_the_sidebar_opens_each_page(home):
    import picasso
    import picasso.server

    app = AppTest.from_file(
        os.path.join(os.path.dirname(picasso.server.__file__), "app.py"),
        default_timeout=60,
    ).run()
    assert not app.exception
    assert f"Version {picasso.__version__}" in app.sidebar.code[0].value
    assert app.sidebar.radio[0].options[0].endswith("Status")
    assert app.markdown.values[0] == "# Status"
    for name in ("History", "Compare", "Watcher", "Preview"):
        app.sidebar.radio[0].set_value(name)
        app.run()
        assert not app.exception
        assert app.markdown.values[0] == f"# {name}"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_an_empty_database_is_reported(self, run_page, home):
        app = run_page("status")
        assert not app.exception
        assert "Database is empty." in app.markdown.values
        assert app.warning.values == ["Path is not valid or no locs found."]
        # the database path is shown, escaped for markdown
        (getting_started, *_) = app.expander
        assert "app\\_0410\\.db" in getting_started.markdown.values[0]

    def test_the_last_entries_are_previewed(self, run_page):
        _add_entries(
            *(
                _entry(f"/data/m{i}.raw", datetime(2026, 1, 1 + i))
                for i in range(12)
            )
        )
        app = run_page("status")
        assert not app.exception
        assert "The database currently contains 12 entries." in (
            app.markdown.values
        )
        (preview,) = app.dataframe
        assert len(preview.value) == 10
        assert preview.value.columns.tolist() == [
            "entry_created",
            "filename",
            "nena_nm",
            "file_created",
        ]

    def test_a_localized_movie_is_added_to_the_database(
        self, run_page, module, movie
    ):
        app = run_page("status")
        _widget(app.text_input, "Enter file path").set_value(str(movie))
        app.run()
        assert not app.exception
        assert app.json  # the summary
        assert module("helper").fetch_db().empty
        _widget(app.button, "Add to database").click()
        app.run()
        assert app.success.values == ["Submitted to DB. Please refresh page."]
        db = module("helper").fetch_db()
        assert db["filename"].tolist() == [str(movie)]
        assert db["n_locs"].tolist() == [
            len(io.load_locs(str(movie.with_name("cell_locs.hdf5")))[0])
        ]

    def test_a_movie_already_in_the_database_is_not_added_again(
        self, run_page, movie
    ):
        _add_entries(_entry(movie, datetime(2026, 1, 1)))
        app = run_page("status")
        _widget(app.text_input, "Enter file path").set_value(str(movie))
        app.run()
        assert app.error.values == ["File already in database."]
        assert not app.button

    def test_the_localized_movies_of_a_folder_are_added(
        self, run_page, module, movie
    ):
        (movie.parent / "dark.raw").write_bytes(b"")  # not localized
        app = run_page("status")
        _widget(app.text_input, "Enter file path").set_value(str(movie.parent))
        app.run()
        assert "A total of 3 files in folder." in _texts(app)
        _widget(app.button, "Add files").click()
        app.run()
        assert not app.exception
        (error,) = app.error.values
        assert "dark_locs.hdf5 does not exist" in error
        assert app.success.values[0] == "Submitted 2 entries to the DB."
        assert sorted(module("helper").fetch_db()["filename"]) == [
            str(movie),
            str(movie.with_name("cell10.raw")),
        ]

    def test_a_folder_without_localized_movies_adds_nothing(
        self, run_page, module, tmp_path
    ):
        (tmp_path / "dark.raw").write_bytes(b"")
        app = run_page("status")
        _widget(app.text_input, "Enter file path").set_value(str(tmp_path))
        app.run()
        _widget(app.button, "Add files").click()
        app.run()
        assert app.warning.values == ["No files found in folder."]
        assert module("helper").fetch_db().empty


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@pytest.fixture
def history_entries(home):
    _add_entries(
        _entry("/data/cell_r1.raw", datetime(2026, 1, 1, 9), nena_nm=4.0),
        _entry("/data/cell_r2.raw", datetime(2026, 1, 5, 9), nena_nm=6.0),
        _entry("/data/beads.raw", datetime(2026, 1, 9, 9), nena_nm=8.0),
    )


class TestHistory:
    def test_an_empty_database_is_reported(self, run_page):
        app = run_page("history")
        assert not app.exception
        assert app.warning.values == ["Database empty. Process files first."]

    def test_the_table_lists_the_newest_files_first(
        self, run_page, history_entries
    ):
        app = run_page("history")
        assert not app.exception
        assert "From 3 Database entries, 3 are remaining." in _texts(app)
        table = app.dataframe[-1].value
        assert table["filename"].tolist() == [
            os.path.normpath("/data/beads.raw"),
            os.path.normpath("/data/cell_r2.raw"),
            os.path.normpath("/data/cell_r1.raw"),
        ]
        (download,) = app.get("download_button")
        assert download.proto.label == "Download as CSV"

    @pytest.mark.parametrize(
        "tags, message, remaining",
        [
            (
                "cell|r2",
                "Filtering for filenames containing ['cell'] but not "
                "['r2'].",
                1,
            ),
            ("cell", "Filtering for filenames containing ['cell'].", 2),
            ("|cell", "Filtering for filenames not containing ['cell'].", 1),
        ],
    )
    def test_the_tag_filter_says_what_it_keeps(
        self, run_page, history_entries, tags, message, remaining
    ):
        app = run_page("history")
        _widget(app.text_input, "Filter for tag").set_value(tags)
        app.run()
        assert message in _texts(app)
        assert (
            f"From 3 Database entries, {remaining} are remaining."
            in _texts(app)
        )
        assert len(app.dataframe[-1].value) == remaining

    def test_the_date_range_drops_the_files_acquired_outside_it(
        self, run_page, history_entries
    ):
        app = run_page("history")
        _widget(app.date_input, "Minimum acquisition date").set_value(
            date(2026, 1, 5)
        )
        _widget(app.date_input, "Maximum acquisition date").set_value(
            date(2026, 1, 5)
        )
        app.run()
        assert app.dataframe[-1].value["filename"].tolist() == [
            os.path.normpath("/data/cell_r2.raw")
        ]

    def test_groups_are_counted_and_added_to_the_table(
        self, run_page, history_entries
    ):
        app = run_page("history")
        _widget(app.text_input, "Enter tags to group").set_value("r1,r2")
        app.run()
        assert app.dataframe[-1].value["group"].tolist() == [
            "None",
            "r2",
            "r1",
        ]
        counts = app.dataframe[0].value
        assert counts.iloc[:, 0].to_dict() == {"None": 1, "r1": 1, "r2": 1}

    def test_the_bar_chart_table_is_html(self, run_page, history_entries):
        app = run_page("history")
        _widget(app.checkbox, "Barchart in column").check()
        app.run()
        assert not app.exception
        assert any("<table" in _ for _ in app.markdown.values)

    def test_scatter_plots_each_field_with_its_median_and_unit(
        self, run_page, history_entries
    ):
        app = run_page("history")
        _widget(app.selectbox, "Plotmode").set_value("Scatter")
        _widget(app.multiselect, "Fields to plot").set_value(
            ["nena_nm", "n_locs"]
        )
        app.run()
        assert not app.exception
        charts = _charts(app)
        assert [_title(_) for _ in charts] == [
            "nena_nm - median 6.00 (nm)",
            "n_locs - median 100.00",
        ]
        assert sorted(_values(charts[0]["data"][0]["y"])) == [4, 6, 8]

    def test_scatter_plots_show_trendlines(self, run_page, history_entries):
        pytest.importorskip("statsmodels")
        app = run_page("history")
        _widget(app.selectbox, "Plotmode").set_value("Scatter")
        _widget(app.multiselect, "Fields to plot").set_value(["nena_nm"])
        app.run()
        _widget(app.checkbox, "Show trendlines").check()
        app.run()
        assert not app.exception
        (chart,) = _charts(app)
        assert len(chart["data"]) == 2  # the points and their trendline

    def test_box_plots_each_field_per_day(self, run_page, history_entries):
        app = run_page("history")
        _widget(app.selectbox, "Plotmode").set_value("Box")
        app.run()
        assert not app.exception
        charts = _charts(app)
        assert len(charts) == 8  # the default fields
        assert charts[0]["data"][0]["type"] == "box"


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_an_empty_database_is_reported(self, run_page):
        app = run_page("compare")
        assert not app.exception
        assert app.warning.values == ["Database empty. Process files first."]

    def test_the_localization_files_of_the_movie_are_offered(
        self, run_page, movie
    ):
        _add_entries(_entry(movie, datetime(2026, 1, 1)))
        app = run_page("compare")
        _widget(app.multiselect, "Select files").select(str(movie))
        app.run()
        assert not app.exception
        assert _widget(app.multiselect, "Select hdf file").options == [
            "cell_locs.hdf5",
            "cell_locs_undrift.hdf5",
        ]
        assert app.warning.values == ["Please select HDF files."]

    def test_a_missing_movie_is_reported(self, run_page, tmp_path):
        missing = tmp_path / "moved" / "cell.raw"
        _add_entries(_entry(missing, datetime(2026, 1, 1)))
        app = run_page("compare")
        _widget(app.multiselect, "Select files").select(str(missing))
        app.run()
        assert not app.exception
        assert "was not found" in app.error.values[0]

    @pytest.fixture
    def compared(self, run_page, movie):
        """The page comparing the two localization files of the movie."""
        _add_entries(_entry(movie, datetime(2026, 1, 1)))
        app = run_page("compare")
        _widget(app.multiselect, "Select files").select(str(movie))
        app.run()
        hdf = _widget(app.multiselect, "Select hdf file")
        hdf.select("cell_locs.hdf5").select("cell_locs_undrift.hdf5")
        app.run()
        assert not app.exception
        return app

    def test_localizations_per_frame_are_plotted_per_file(self, compared):
        (chart,) = _charts(compared)
        assert _title(chart) == "Locs per Frame"
        assert [os.path.basename(_["name"]) for _ in chart["data"]] == [
            "cell_locs.hdf5",
            "cell_locs_undrift.hdf5",
        ]
        _widget(compared.number_input, "Smooth").set_value(1)
        compared.run()
        (chart,) = _charts(compared)
        assert max(_values(chart["data"][0]["y"])) >= 1

    def test_histograms_of_a_field_are_plotted_per_file(self, compared):
        _widget(compared.selectbox, "Select plot").set_value("Histogram")
        compared.run()
        _widget(compared.selectbox, "Select field").set_value("photons")
        _widget(compared.number_input, "Number of bins").set_value(10)
        compared.run()
        assert not compared.exception
        assert not compared.warning
        (chart,) = _charts(compared)
        assert _title(chart) == "photons"
        assert len(chart["data"]) == 2
        assert len(_values(chart["data"][0]["x"])) == 10

    def test_a_field_that_cannot_be_plotted_is_reported(self, compared):
        _widget(compared.selectbox, "Select plot").set_value("Histogram")
        compared.run()
        _widget(compared.selectbox, "Select field").set_value("file")
        compared.run()
        assert not compared.exception
        assert "plotting field **file**" in compared.warning.values[0]


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


class TestPreview:
    def test_an_empty_database_is_reported(self, run_page):
        app = run_page("preview")
        assert not app.exception
        assert "Database empty. Process files first." in app.markdown.values

    def test_a_moved_folder_is_reported(self, run_page, tmp_path):
        moved = tmp_path / "moved" / "a.raw"
        _add_entries(_entry(moved, datetime(2026, 1, 1)))
        app = run_page("preview")
        assert not app.exception
        assert "Couldn't find folder" in app.error.values[0]

    def test_a_localization_file_is_rendered(self, run_page, movie):
        _add_entries(_entry(movie, datetime(2026, 1, 1)))
        app = run_page("preview")
        hdf = _widget(app.selectbox, "Select hdf file")
        assert hdf.options == [
            "None",
            "cell_locs.hdf5",
            "cell_locs_undrift.hdf5",
        ]
        hdf.set_value("cell_locs.hdf5")
        app.run()
        assert not app.exception
        assert app.info.values[0].startswith("Performance Warning")
        # st.pyplot adds an "imgs" element up to Streamlit 1.56 and an
        # "image" one in later versions
        assert len(app.get("imgs")) + len(app.get("image")) == 1
        assert app.json[0].value  # the database entry
        assert json.loads(app.json[0].value)["filename"] == str(movie)

        def dimensions():
            (text,) = [
                _ for _ in app.markdown.values if _.startswith("Image with")
            ]
            return text

        coarse = dimensions()
        _widget(app.number_input, "Display pixel size").set_value(500.0)
        app.run()
        assert not app.exception
        assert dimensions() != coarse


# ---------------------------------------------------------------------------
# Watcher page
# ---------------------------------------------------------------------------


def _free_pid() -> int:
    """A process id that no process has."""
    import psutil

    pid = 999_999
    while psutil.pid_exists(pid):
        pid -= 1
    return pid


class FakeProcess:
    """Stands in for ``multiprocessing.Process``: records the watcher
    instead of starting it."""

    started: list[FakeProcess] = []

    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.pid = _free_pid()

    def start(self):
        FakeProcess.started.append(self)


@pytest.fixture
def watcher_page(run_page, module, monkeypatch, tmp_path):
    """The Watcher page on a folder, starting no process and not
    waiting."""
    watcher = module("watcher")
    FakeProcess.started = []
    monkeypatch.setattr(watcher, "Process", FakeProcess)
    monkeypatch.setattr(
        watcher, "time", SimpleNamespace(sleep=lambda seconds: None)
    )
    folder = tmp_path / "acquisition"
    folder.mkdir()
    app = run_page("watcher")
    _widget(app.text_input, "Enter folder to watch").set_value(str(folder))
    app.run()
    app.folder = folder
    return app


def _submit(app: AppTest) -> FakeProcess:
    _widget(app.button, "Submit").click()
    app.run()
    assert not app.exception
    (process,) = FakeProcess.started
    return process


class TestWatcherPage:
    def test_without_watchers_none_are_listed(self, run_page):
        app = run_page("watcher")
        assert not app.exception
        assert "None" in app.markdown.values

    def test_a_folder_that_does_not_exist_is_reported(
        self, run_page, tmp_path
    ):
        app = run_page("watcher")
        _widget(app.text_input, "Enter folder to watch").set_value(
            str(tmp_path / "missing")
        )
        app.run()
        assert app.error.values == ["Not a valid path."]

    def test_a_submitted_watcher_is_started_and_registered(
        self, watcher_page, module, home
    ):
        _add_entries(_entry("/data/old.raw", datetime(2026, 1, 1)))
        process = _submit(watcher_page)
        settings, folder, command, logfile, existing, update_time = (
            process.args
        )
        assert process.target is module("watcher").check_new_and_process
        assert settings == [
            {
                "box_side_length": 7,
                "gradient": 5000,
                "gain": 1,
                "baseline": 100,
                "sensitivity": 1.0,
                "fit_method": "lq",
                "drift": 1000,
                "database": True,
                "suffix": "",
            }
        ]
        assert folder == str(watcher_page.folder)
        assert command == ""
        assert os.path.dirname(logfile) == str(home / ".picasso")
        assert existing == [os.path.normpath("/data/old.raw")]
        assert update_time == 1
        assert watcher_page.success.values[0] == "Started watcher."
        watchers = module("helper").fetch_watcher()
        assert watchers["process id"].tolist() == [process.pid]
        assert watchers["logfile"].tolist() == [logfile]

    def test_the_pixel_size_comes_from_the_movie_unless_entered(
        self, watcher_page, module
    ):
        """Not a fixed 130 nm: like ``picasso localize``, the movie's
        metadata or the camera config give it."""
        settings = _submit(watcher_page).args[0][0]
        assert "pixelsize" not in settings
        args = module("watcher").localize_args({"files": "a.raw", **settings})
        assert args.pixelsize is None

        _widget(watcher_page.number_input, "Pixelsize (nm):").set_value(160)
        _widget(watcher_page.button, "Submit").click()
        watcher_page.run()
        assert FakeProcess.started[-1].args[0][0]["pixelsize"] == 160

    def test_parameter_groups_get_a_suffix_each(self, watcher_page):
        _widget(watcher_page.number_input, "Number of Parameter").set_value(2)
        watcher_page.run()
        assert any("_pg_1_locs.hdf5" in _ for _ in _texts(watcher_page))
        min_gradients = [
            _
            for _ in watcher_page.number_input
            if _.label == "Min. Net Gradient:"
        ]
        min_gradients[1].set_value(8000)
        process = _submit(watcher_page)
        settings = process.args[0]
        assert [_["suffix"] for _ in settings] == ["_pg_1", "_pg_2"]
        assert [_["gradient"] for _ in settings] == [5000, 8000]

    def test_3d_fits_with_the_calibration(self, watcher_page, tmp_path):
        calibration = tmp_path / "3d_calib.yaml"
        _widget(watcher_page.checkbox, "3D").check()
        watcher_page.run()
        assert watcher_page.error.values == ["Not a valid file."]
        calibration.write_text("")
        path = _widget(watcher_page.text_input, "Enter path to calibration")
        path.set_value(str(calibration))
        watcher_page.run()
        assert not watcher_page.error
        (settings,) = _submit(watcher_page).args[0]
        assert settings["fit_method"] == "lq-3d"
        assert settings["zc"] == str(calibration)
        assert settings["mf"] == 0.79

    def test_3d_in_two_parameter_groups(self, watcher_page):
        _widget(watcher_page.number_input, "Number of Parameter").set_value(2)
        watcher_page.run()
        for checkbox in watcher_page.checkbox:
            if checkbox.label == "3D":
                checkbox.check()
        watcher_page.run()
        assert not watcher_page.exception
        settings = _submit(watcher_page).args[0]
        assert [_["fit_method"] for _ in settings] == ["lq-3d", "lq-3d"]

    def test_the_custom_command_is_chained(self, watcher_page):
        _widget(watcher_page.checkbox, "Chain Custom command").check()
        watcher_page.run()
        _widget(watcher_page.text_input, "Command").set_value(
            "picasso undrift $LOCS"
        )
        process = _submit(watcher_page)
        assert process.args[2] == "picasso undrift $LOCS"


@pytest.fixture
def watchers(home, module, tmp_path):
    """A running watcher (this process) and one that stopped, each with
    its logfile."""
    rows = []
    for pid, name in ((os.getpid(), "running"), (_free_pid(), "stopped")):
        logfile = tmp_path / f"{name}.log"
        logfile.write_text(f"{name} watcher\n")
        rows.append(
            {
                "process id": pid,
                "started": datetime(2026, 10, 1, 9),
                "folder": str(tmp_path),
                "logfile": str(logfile),
            }
        )
    module("helper").write_table(
        pd.DataFrame(rows), "watcher", if_exists="append"
    )
    return rows


class TestExistingWatchers:
    def test_watchers_are_listed_with_whether_they_run(
        self, run_page, watchers
    ):
        app = run_page("watcher")
        assert not app.exception
        assert app.dataframe[0].value["running"].tolist() == [True, False]

    def test_the_log_of_the_latest_watcher_is_shown(self, run_page, watchers):
        app = run_page("watcher")
        logfile = _widget(app.selectbox, "Select logfile")
        assert logfile.value == watchers[1]["logfile"]
        assert "stopped watcher" in app.expander[0].text[0].value
        logfile.set_value(watchers[0]["logfile"])
        app.run()
        assert "running watcher" in app.expander[0].text[0].value

    def test_a_missing_logfile_is_reported(self, run_page, watchers):
        os.remove(watchers[1]["logfile"])
        app = run_page("watcher")
        assert not app.exception
        assert any("not found" in _ for _ in app.warning.values)

    def test_watchers_that_stopped_are_removed(
        self, run_page, module, watchers
    ):
        app = run_page("watcher")
        _widget(app.button, "Remove non-running watchers").click()
        app.run()
        assert app.success.values == ["Removed. Please refresh this page."]
        assert module("helper").fetch_watcher()["process id"].tolist() == [
            os.getpid()
        ]


# ---------------------------------------------------------------------------
# The watcher at work
# ---------------------------------------------------------------------------


class Stop(Exception):
    """Ends the watcher at its pause after a scan."""


def _stop(minutes):
    raise Stop


def _touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


@pytest.fixture
def folder(tmp_path):
    """An empty folder to watch."""
    folder = tmp_path / "watched"
    folder.mkdir()
    return folder


class TestCheckNew:
    def test_new_movies_are_found(self, module, folder, tmp_path):
        _touch(folder, "a.raw", "b.ome.tif", "c.ims", "notes.txt")
        logfile = tmp_path / "watcher.log"
        new, _ = module("watcher").check_new(str(folder), {}, logfile)
        assert sorted(os.path.basename(_) for _ in new) == [
            "a.raw",
            "b.ome.tif",
            "c.ims",
        ]
        assert "Checking: 4 files, 3 unprocessed" in logfile.read_text()

    def test_processed_movies_are_skipped(self, module, folder, tmp_path):
        _touch(folder, "a.raw", "b.raw")
        processed = {str(folder / "a.raw"): True}
        new, _ = module("watcher").check_new(
            str(folder), processed, tmp_path / "log"
        )
        assert new == [str(folder / "b.raw")]

    def test_localized_movies_count_as_processed(
        self, module, folder, tmp_path
    ):
        _touch(
            folder,
            "a.raw",
            "a_locs.hdf5",
            "b.raw",
            "b_pg_1_locs.hdf5",
            "Pos0.ome.tif",
            "Pos0.ome_locs.hdf5",
            "c.raw",
            "c10_locs.hdf5",  # another movie
        )
        new, processed = module("watcher").check_new(
            str(folder), {}, tmp_path / "log"
        )
        assert new == [str(folder / "c.raw")]
        assert sorted(os.path.basename(_) for _ in processed) == [
            "Pos0.ome.tif",
            "a.raw",
            "b.raw",
        ]

    def test_a_relative_folder_finds_the_processed_movies(
        self, module, folder, tmp_path, monkeypatch
    ):
        _touch(folder, "a.raw")
        monkeypatch.chdir(tmp_path)
        processed = {str(folder / "a.raw"): True}
        new, _ = module("watcher").check_new(
            folder.name, processed, tmp_path / "log"
        )
        assert new == []


class TestWaitForCompletion:
    def test_waits_until_the_file_stops_growing(
        self, module, monkeypatch, tmp_path
    ):
        watcher = module("watcher")
        movie = tmp_path / "a.raw"
        movie.write_bytes(b"0")
        sizes = []

        def sleep(seconds):
            sizes.append(movie.stat().st_size)
            if len(sizes) < 3:  # the camera writes two more frames
                with open(movie, "ab") as f:
                    f.write(b"0")

        monkeypatch.setattr(watcher, "time", SimpleNamespace(sleep=sleep))
        assert watcher.wait_for_completion(str(movie)) == []
        assert sizes == [1, 2, 3]

    def test_waits_for_the_continuation_files_of_an_ome_tif(
        self, module, monkeypatch, tmp_path
    ):
        watcher = module("watcher")
        monkeypatch.setattr(
            watcher, "time", SimpleNamespace(sleep=lambda seconds: None)
        )
        _touch(
            tmp_path,
            "cell_MMStack_Pos0.ome.tif",
            "cell_MMStack_Pos0_1.ome.tif",
            "other_MMStack_Pos0.ome.tif",
        )
        first = str(tmp_path / "cell_MMStack_Pos0.ome.tif")
        assert watcher.wait_for_completion(first) == [
            first,
            str(tmp_path / "cell_MMStack_Pos0_1.ome.tif"),
        ]


def test_localize_args_complete_the_settings_with_the_defaults(module):
    args = module("watcher").localize_args(
        {"files": "/data/a.raw", "fit_method": "mle", "gradient": 4000}
    )
    assert args.files == "/data/a.raw"
    assert args.fit_method == ["mle"]
    assert args.gradient == [4000]
    assert args.box_side_length == 7  # the command line default
    assert args.roi is None


@pytest.fixture
def acquisition(tmp_path):
    """A folder with a movie in it."""
    folder = tmp_path / "movies"
    folder.mkdir()
    for name in ("testdata.raw", "testdata.yaml"):
        shutil.copy(os.path.join(DATA, name), folder / name)
    return folder


def test_a_submitted_watcher_localizes_new_movies(
    watcher_page, module, monkeypatch, acquisition, tmp_path
):
    """From the page to the localization file and the database entry."""
    watcher = module("watcher")
    _widget(watcher_page.text_input, "Enter folder to watch").set_value(
        str(acquisition)
    )
    watcher_page.run()
    _widget(watcher_page.selectbox, "Method").set_value("mle")
    _widget(watcher_page.number_input, "Min. Net Gradient").set_value(2000)
    # the test movie's metadata does not record its pixel size
    _widget(watcher_page.number_input, "Pixelsize (nm):").set_value(160)
    args = _submit(watcher_page).args
    monkeypatch.setattr(watcher, "wait_for_completion", lambda file: [])
    monkeypatch.setattr(watcher, "time", SimpleNamespace(sleep=_stop))
    with pytest.raises(Stop):
        watcher.check_new_and_process(*args)
    movie = acquisition / "testdata.raw"
    locs, info = io.load_locs(str(acquisition / "testdata_locs.hdf5"))
    assert len(locs)
    assert module("helper").fetch_db()["filename"].tolist() == [str(movie)]
    log = open(args[3]).read()
    assert f"File {movie} processed." in log
    # the 100 frames of the test movie are too few to undrift with the
    # smallest segmentation the page offers; localize reports that and
    # keeps the localizations, so the watcher logs no error
    assert "Exception" not in log


def test_a_failing_movie_does_not_stop_the_watcher(
    module, monkeypatch, tmp_path
):
    watcher = module("watcher")
    _touch(tmp_path, "a.raw", "b.raw")
    tried = []

    def localize(args):
        tried.append(os.path.basename(args.files))
        raise SystemExit("too few frames")  # as picasso localize exits

    monkeypatch.setattr(watcher, "_localize", localize)
    monkeypatch.setattr(watcher, "wait_for_completion", lambda file: [])
    scans = []

    def sleep(minutes):
        scans.append(minutes)
        if len(scans) == 2:
            raise Stop

    monkeypatch.setattr(watcher, "time", SimpleNamespace(sleep=sleep))
    logfile = tmp_path / "watcher.log"
    with pytest.raises(Stop):
        watcher.check_new_and_process(
            [{"suffix": ""}], str(tmp_path), "", str(logfile), [], 1
        )
    assert sorted(tried) == ["a.raw", "b.raw"]
    assert logfile.read_text().count("too few frames") == 2


def test_movies_in_the_database_are_not_localized_again(
    module, monkeypatch, tmp_path
):
    watcher = module("watcher")
    _touch(tmp_path, "a.raw")
    monkeypatch.setattr(
        watcher, "_localize", lambda args: pytest.fail("localized")
    )
    monkeypatch.setattr(watcher, "time", SimpleNamespace(sleep=_stop))
    with pytest.raises(Stop):
        watcher.check_new_and_process(
            [{"suffix": ""}],
            str(tmp_path),
            "",
            str(tmp_path / "log"),
            [os.path.normpath(str(tmp_path / "a.raw"))],
            1,
        )


def test_a_command_with_locs_but_no_localization_file_is_not_run(
    module, monkeypatch, tmp_path
):
    watcher = module("watcher")
    _touch(tmp_path, "a.raw")
    monkeypatch.setattr(watcher, "_localize", lambda args: [])
    monkeypatch.setattr(watcher, "wait_for_completion", lambda file: [])
    monkeypatch.setattr(watcher, "time", SimpleNamespace(sleep=_stop))
    monkeypatch.setattr(
        watcher.subprocess, "run", lambda *a, **k: pytest.fail("ran")
    )
    logfile = tmp_path / "watcher.log"
    with pytest.raises(Stop):
        watcher.check_new_and_process(
            [{"suffix": ""}],
            str(tmp_path),
            "picasso undrift $LOCS",
            str(logfile),
            [],
            1,
        )
    assert "No localization file saved" in logfile.read_text()
