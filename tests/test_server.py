"""``picasso server`` listens only on this computer unless ``--network``
is given: the server has no login and its watcher runs custom commands.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import sys

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
