"""The update check of the command line: due once a day unless switched
off, asked about only in a terminal and only once the command is done,
so that batch jobs never wait for an answer.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta

import pytest

from picasso import io, updater
from picasso import __main__ as cli

LATEST = "99.0"


@pytest.fixture(autouse=True)
def latest_release(monkeypatch):
    """GitHub is never asked: the latest release is ``LATEST``. Returns
    the list of requests made."""
    requests = []

    def get_latest_version():
        requests.append(threading.current_thread())
        return LATEST

    monkeypatch.setattr(updater, "get_latest_version", get_latest_version)
    return requests


def _set_updates(**entries):
    settings = io.load_user_settings()
    settings["Updates"].update(entries)
    io.save_user_settings(settings)


def _updates() -> dict:
    return dict(io.load_user_settings()["Updates"])


def test_a_check_is_due_on_fresh_settings():
    # no check recorded yet, which used to count as "checked today"
    assert updater.should_check_today() is True
    assert updater.update_check_due() is True


@pytest.mark.parametrize(
    "entries",
    [
        {"Disabled": True},
        {"Snoozed until": (datetime.now() + timedelta(days=3)).isoformat()},
        {"Last update check": datetime.now().isoformat()},
    ],
)
def test_no_request_unless_due(entries, latest_release):
    _set_updates(**entries)
    assert not updater.update_check_due()
    updater.CliUpdateCheck().finish()
    assert latest_release == []


def test_due_again_after_a_day():
    _set_updates(
        **{
            "Last update check": (
                datetime.now() - timedelta(hours=25)
            ).isoformat()
        }
    )
    assert updater.update_check_due()


def test_batch_job_is_told_without_waiting_for_an_answer(monkeypatch, capsys):
    monkeypatch.setattr(updater, "_interactive", lambda: False)

    def no_input(prompt=""):
        raise AssertionError("asked without a terminal")

    monkeypatch.setattr("builtins.input", no_input)
    updater.CliUpdateCheck().finish()
    err = capsys.readouterr().err
    assert f"Picasso update available: v{LATEST}" in err
    assert "set Disabled: true" in err
    # the next command does not check again today
    assert not updater.update_check_due()


def test_terminal_user_is_asked_in_the_calling_thread(
    monkeypatch, latest_release
):
    monkeypatch.setattr(updater, "_interactive", lambda: True)
    asked_in = []

    def answer(prompt=""):
        asked_in.append(threading.current_thread())
        return "2"  # skip this version

    monkeypatch.setattr("builtins.input", answer)
    updater.CliUpdateCheck().finish()
    # the request ran in the background, the question did not
    assert latest_release[0] is not threading.main_thread()
    assert asked_in == [threading.main_thread()]
    assert _updates()["Skipped version"] == LATEST


def test_closed_stdin_at_the_prompt(monkeypatch):
    monkeypatch.setattr(updater, "_interactive", lambda: True)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    updater.CliUpdateCheck().finish()  # does not raise


def test_read_only_settings_do_not_fail_the_command(monkeypatch, capsys):
    def read_only(settings):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(io, "save_user_settings", read_only)
    monkeypatch.setattr(updater, "_interactive", lambda: False)
    updater.CliUpdateCheck().finish()  # does not raise
    assert "update available" in capsys.readouterr().err


def test_skipped_version_is_not_shown(capsys):
    _set_updates(**{"Skipped version": LATEST})
    updater.CliUpdateCheck().finish()
    assert "update available" not in capsys.readouterr().err


def test_offline_check_waits_a_day(monkeypatch, capsys):
    monkeypatch.setattr(updater, "get_latest_version", lambda: None)
    updater.CliUpdateCheck().finish()
    assert "update available" not in capsys.readouterr().err
    # an unreachable server does not slow down every following command
    assert not updater.update_check_due()


def test_command_line_tells_after_the_command(monkeypatch):
    events = []
    monkeypatch.setattr(
        cli, "_pair_correlation", lambda *args: events.append("command")
    )
    monkeypatch.setattr(
        updater, "cli_notify_update", lambda latest: events.append(latest)
    )
    monkeypatch.setattr("sys.argv", ["picasso", "pair-correlation", "f"])
    cli.main()
    assert events == ["command", LATEST]


def _wait_until(condition, timeout=5.0):
    """Process Qt events until ``condition()`` holds."""
    import time

    from PyQt6 import QtCore

    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        QtCore.QCoreApplication.processEvents()
        time.sleep(0.01)


def test_gui_does_not_check_when_switched_off(qt_offscreen, latest_release):
    _set_updates(Disabled=True)
    updater.setup_gui_update_check()
    assert latest_release == []


def test_gui_records_the_check_and_skips_skipped_versions(
    qt_offscreen, monkeypatch
):
    from PyQt6 import QtWidgets

    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec", lambda self: shown.append(self)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(args),
    )
    _set_updates(**{"Skipped version": LATEST})
    updater.setup_gui_update_check()
    _wait_until(lambda: "Last update check" in _updates())
    assert shown == []
