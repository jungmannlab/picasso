"""The update check of the command line: due once a day unless switched
off, asked about only in a terminal and only once the command is done,
so that batch jobs never wait for an answer. And "Install Now" in the
one-click builds: finding the installer among the release assets,
checking the download, the helper scripts that install it once Picasso
has quit, and the report at the next start.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, timedelta

import pytest

from picasso import __version__, io, updater
from picasso import __main__ as cli

LATEST = "99.0"

#: before the autouse fixture below replaces it
_get_latest_version = updater.get_latest_version


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


#: the assets of v0.11.3, as published
ASSETS_0_11 = [
    {"name": "Picasso-Windows-64bit-0.11.3.zip"},
    {"name": "Picasso-Windows-64bit-GPU-0.11.3.zip"},
    {"name": "Picasso_v0.11.3_installer_macOS_apple_silicon.zip"},
]

#: the assets as the release scripts of 1.0 name them
ASSETS_1_0 = [
    {"name": "Picasso-Windows-64bit-1.0.1.zip"},
    {"name": "Picasso-Windows-64bit-CUDA-1.0.1.zip"},
    {"name": "Picasso-Apple-Silicon-1.0.1.zip"},
]

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="the helper is a bash script on macOS"
)
macos_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="needs hdiutil and open"
)


def _names(assets, platform, cuda=False):
    asset = updater.find_installer(
        {"assets": assets}, platform=platform, cuda=cuda
    )
    return None if asset is None else asset["name"]


@pytest.mark.parametrize(
    "assets, windows, cuda, macos",
    [
        (
            ASSETS_0_11,
            "Picasso-Windows-64bit-0.11.3.zip",
            "Picasso-Windows-64bit-GPU-0.11.3.zip",
            "Picasso_v0.11.3_installer_macOS_apple_silicon.zip",
        ),
        (
            ASSETS_1_0,
            "Picasso-Windows-64bit-1.0.1.zip",
            "Picasso-Windows-64bit-CUDA-1.0.1.zip",
            "Picasso-Apple-Silicon-1.0.1.zip",
        ),
    ],
)
def test_each_edition_finds_its_installer(assets, windows, cuda, macos):
    assert _names(assets, "win32") == windows
    assert _names(assets, "win32", cuda=True) == cuda
    assert _names(assets, "darwin") == macos
    assert _names(assets, "linux") is None


def test_the_installer_is_preferred_to_a_zip_of_it():
    assets = [
        {"name": "Picasso-Windows-64bit-1.0.1.zip"},
        {"name": "Picasso-Windows-64bit-1.0.1.exe"},
        {"name": "Picasso-v1.0.1-macOS-Apple-Silicon.dmg"},
        {"name": "Picasso-Apple-Silicon-1.0.1.zip"},
    ]
    assert _names(assets, "win32") == "Picasso-Windows-64bit-1.0.1.exe"
    assert _names(assets, "darwin") == "Picasso-v1.0.1-macOS-Apple-Silicon.dmg"


def test_no_installer_in_the_release():
    assets = [{"name": "picassosr-1.0.1.tar.gz"}, {"name": "notes.txt"}]
    assert _names(assets, "win32") is None
    assert _names(assets, "darwin") is None
    assert updater.find_installer({}, platform="darwin") is None


class _Response:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.content), chunk_size):
            end = start + chunk_size
            yield self.content[start:end]


@pytest.fixture
def served(monkeypatch):
    """Serve ``content`` for every download; returns its asset."""
    content = os.urandom(3 * updater.DOWNLOAD_CHUNK + 5)
    monkeypatch.setattr(
        updater.requests, "get", lambda *a, **k: _Response(content)
    )
    return {
        "name": "Picasso-Windows-64bit-1.0.1.exe",
        "size": len(content),
        "browser_download_url": "https://example.org/installer",
        "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
    }


def test_download_is_checked_against_its_digest(served, tmp_path):
    path = updater.download_installer(served, str(tmp_path))
    assert path == str(tmp_path / served["name"])
    assert os.path.getsize(path) == served["size"]


def test_damaged_download_is_refused(served, tmp_path):
    served["digest"] = "sha256:" + "0" * 64
    with pytest.raises(updater.UpdateError, match="damaged"):
        updater.download_installer(served, str(tmp_path))


def test_incomplete_download_is_refused(served, tmp_path):
    served["size"] += 1
    with pytest.raises(updater.UpdateError, match="incomplete"):
        updater.download_installer(served, str(tmp_path))


def test_download_without_a_digest(served, tmp_path):
    # assets uploaded before GitHub published digests
    del served["digest"]
    updater.download_installer(served, str(tmp_path))


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return str(path)


def test_installer_is_taken_out_of_the_zip(tmp_path):
    path = _zip(
        tmp_path / "Picasso-Apple-Silicon-1.0.1.zip",
        {"Picasso-v1.0.1-macOS-Apple-Silicon.dmg": b"dmg", "readme.txt": b""},
    )
    installer = updater.extract_installer(path, platform="darwin")
    assert installer == str(
        tmp_path / "Picasso-v1.0.1-macOS-Apple-Silicon.dmg"
    )
    assert open(installer, "rb").read() == b"dmg"
    assert not os.path.exists(path)


def test_a_path_in_the_zip_stays_in_the_folder(tmp_path):
    folder = tmp_path / "download"
    folder.mkdir()
    path = _zip(folder / "installer.zip", {"../../setup.exe": b"exe"})
    installer = updater.extract_installer(path, platform="win32")
    assert installer == str(folder / "setup.exe")


def test_zip_without_an_installer(tmp_path):
    path = _zip(tmp_path / "installer.zip", {"readme.txt": b""})
    with pytest.raises(updater.UpdateError):
        updater.extract_installer(path, platform="win32")


def test_installer_that_is_no_zip_is_kept(tmp_path):
    path = str(tmp_path / "setup.exe")
    assert updater.extract_installer(path, platform="win32") == path


def test_windows_script_quotes_paths():
    update = updater.PreparedUpdate(
        "1.0.1", r"C:\Users\O'Neil\Temp\upd", r"C:\Users\O'Neil\Temp\upd\s.exe"
    )
    script = updater.windows_script(
        update, 4321, r"C:\Picasso", [r"C:\Picasso\picassow.exe", "render"]
    )
    assert "Wait-Process -Id 4321" in script
    # single quotes doubled inside PowerShell literals
    assert r"-FilePath 'C:\Users\O''Neil\Temp\upd\s.exe'" in script
    assert r'/DIR="C:\Picasso"' in script
    assert r"Start-Process -FilePath 'C:\Picasso\picassow.exe'" in script
    assert "-ArgumentList 'render'" in script


def _app(path, content):
    """A stand-in app bundle holding one file."""
    os.makedirs(path / "Contents", exist_ok=True)
    (path / "Contents" / "version.txt").write_text(content)


def _version(path):
    return (path / "Contents" / "version.txt").read_text()


def _run_helper(update, tmp_path, monkeypatch, relaunch):
    log = tmp_path / "update.log"
    monkeypatch.setattr(updater, "_update_log", lambda: str(log))
    # a process that has quit already, as Picasso has
    done = subprocess.Popen(["true"])
    done.wait()
    script = tmp_path / "install_update.sh"
    script.write_text(updater.macos_script(update, done.pid, relaunch))
    subprocess.run(["/bin/bash", str(script)], check=True, timeout=60)
    return log.read_text()


@posix_only
def test_macos_helper_swaps_the_apps_in(tmp_path, monkeypatch):
    applications = tmp_path / "Applications"
    workdir = tmp_path / "picasso-update"
    workdir.mkdir()
    staged = []
    for name in ["Picasso.app", "Picasso Render.app"]:
        _app(applications / name, "old")
        staging = applications / f".{name}.update"
        _app(staging, "new")
        staged.append((str(staging), str(applications / name)))
    update = updater.PreparedUpdate(
        "1.0.1", str(workdir), str(workdir / "x.dmg"), staged
    )
    marker = tmp_path / "relaunched"
    log = _run_helper(update, tmp_path, monkeypatch, ["touch", str(marker)])
    assert _version(applications / "Picasso.app") == "new"
    assert _version(applications / "Picasso Render.app") == "new"
    # nothing hidden left behind
    assert sorted(os.listdir(applications)) == [
        "Picasso Render.app",
        "Picasso.app",
    ]
    assert not workdir.exists()
    assert marker.exists()
    assert "Installed" in log


@macos_only
def test_macos_helper_keeps_the_old_app_if_the_swap_fails(
    tmp_path, monkeypatch
):
    applications = tmp_path / "Applications"
    _app(applications / "Picasso.app", "old")
    workdir = tmp_path / "picasso-update"
    workdir.mkdir()
    # the staged copy is missing, so moving it in fails
    staged = [
        (
            str(applications / ".Picasso.app.update"),
            str(applications / "Picasso.app"),
        )
    ]
    update = updater.PreparedUpdate(
        "1.0.1", str(workdir), str(workdir / "missing.dmg"), staged
    )
    marker = tmp_path / "relaunched"
    log = _run_helper(update, tmp_path, monkeypatch, ["touch", str(marker)])
    assert _version(applications / "Picasso.app") == "old"
    assert os.listdir(applications) == ["Picasso.app"]
    # kept, for installing the disk image by hand
    assert workdir.exists()
    assert marker.exists()
    assert "was not installed" in log


@macos_only
def test_apps_are_staged_from_the_disk_image(tmp_path):
    source = tmp_path / "dmg"
    for name in ["Picasso.app", "Picasso Render.app", "Picasso Average.app"]:
        _app(source / name, "new")
    disk_image = tmp_path / "Picasso.dmg"
    subprocess.run(
        [
            "hdiutil", "create", "-quiet", "-fs", "HFS+", "-srcfolder",
            str(source), "-volname", "Picasso", str(disk_image),
        ],
        check=True,
    )  # fmt: skip
    applications = tmp_path / "Applications"
    _app(applications / "Picasso.app", "old")
    # installed: Render; left out: Average
    _app(applications / "Picasso Render.app", "old")

    staged = updater.stage_macos_apps(
        str(disk_image), str(applications / "Picasso.app")
    )
    assert [target for _, target in staged] == [
        str(applications / "Picasso.app"),
        str(applications / "Picasso Render.app"),
    ]
    for staging, _ in staged:
        assert os.path.basename(staging).startswith(".")
        assert _version(type(applications)(staging)) == "new"
    # the installed apps are untouched until Picasso has quit
    assert _version(applications / "Picasso.app") == "old"
    # the disk image was detached
    info = subprocess.run(
        ["hdiutil", "info"], capture_output=True, text=True
    ).stdout
    assert str(disk_image) not in info


@macos_only
def test_disk_image_without_picasso_is_refused(tmp_path):
    source = tmp_path / "dmg"
    _app(source / "Other.app", "new")
    disk_image = tmp_path / "Other.dmg"
    subprocess.run(
        [
            "hdiutil", "create", "-quiet", "-fs", "HFS+", "-srcfolder",
            str(source), "-volname", "Other", str(disk_image),
        ],
        check=True,
    )  # fmt: skip
    applications = tmp_path / "Applications"
    _app(applications / "Picasso.app", "old")
    with pytest.raises(updater.UpdateError, match="no Picasso.app"):
        updater.stage_macos_apps(
            str(disk_image), str(applications / "Picasso.app")
        )
    assert os.listdir(applications) == ["Picasso.app"]


def test_unfinished_install_is_reported_once():
    _set_updates(Installing="99.0")
    assert updater.unfinished_install() == "99.0"
    assert updater.unfinished_install() is None


def test_finished_install_is_not_reported():
    _set_updates(Installing=__version__)
    assert updater.unfinished_install() is None
    assert "Installing" not in io.load_user_settings()["Updates"]


def test_nothing_to_report():
    assert updater.unfinished_install() is None


class _Process:
    def __init__(self, pid, exe, cmdline):
        self.info = {"pid": pid, "exe": exe, "cmdline": cmdline}


def test_other_instances_of_this_installation(monkeypatch, tmp_path):
    import psutil

    exe = str(tmp_path / "Picasso" / "picassow.exe")
    other_exe = str(tmp_path / "Other" / "picassow.exe")
    monkeypatch.setattr(sys, "executable", exe)
    processes = [
        _Process(os.getpid(), exe, [exe, "render"]),  # this one
        _Process(1001, exe, [exe, "localize"]),
        _Process(1002, exe, [exe, "--multiprocessing-fork"]),  # a worker
        _Process(1003, other_exe, [other_exe, "filter"]),  # elsewhere
        _Process(1004, None, None),  # not accessible
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: processes)
    assert updater.other_instances() == ["localize"]


def test_get_latest_version_reads_the_release(monkeypatch):
    monkeypatch.setattr(updater, "get_latest_version", _get_latest_version)
    monkeypatch.setattr(
        updater, "get_latest_release", lambda: {"tag_name": "v1.2.3"}
    )
    assert updater.get_latest_version() == "1.2.3"
    monkeypatch.setattr(updater, "get_latest_release", lambda: None)
    assert updater.get_latest_version() is None


@pytest.fixture
def one_click(monkeypatch):
    """Run as a one-click build would: the update notification offers
    to install, and the latest release is newer."""
    monkeypatch.setattr(updater, "can_self_update", lambda: True)
    monkeypatch.setattr(
        updater, "get_update_url", lambda: updater.URL_LATEST_RELEASE
    )
    monkeypatch.setattr(updater, "update_check_due", lambda: True)
    monkeypatch.setattr(updater, "is_update_available", lambda: (True, "99.0"))


def test_install_now_starts_the_installation(
    qt_offscreen, one_click, monkeypatch
):
    from PyQt6 import QtWidgets

    buttons = []

    def click_install(box):
        buttons.extend(button.text() for button in box.buttons())
        [install] = [b for b in box.buttons() if b.text() == "Install Now"]
        install.click()

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", click_install)
    installs = []
    monkeypatch.setattr(
        updater, "install_update_from_gui", lambda parent: installs.append(1)
    )
    updater.setup_gui_update_check()
    _wait_until(lambda: installs)
    assert "Open in Browser" in buttons


def test_unreachable_release_server(qt_offscreen, monkeypatch):
    from PyQt6 import QtWidgets

    warnings = []
    monkeypatch.setattr(updater, "get_latest_release", lambda: None)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, text, *a: warnings.append(text),
    )
    updater.install_update_from_gui()
    assert "could not be reached" in warnings[0]


def test_release_without_installer_opens_the_release_page(
    qt_offscreen, monkeypatch
):
    import webbrowser

    from PyQt6 import QtWidgets

    opened = []
    monkeypatch.setattr(
        updater,
        "get_latest_release",
        lambda: {"tag_name": "v99.0", "assets": []},
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", lambda *args: None
    )
    monkeypatch.setattr(webbrowser, "open", opened.append)
    updater.install_update_from_gui()
    assert opened == [updater.URL_LATEST_RELEASE]


def test_other_programs_must_be_closed_first(
    qt_offscreen, monkeypatch, tmp_path
):
    from PyQt6 import QtWidgets

    workdir = tmp_path / "picasso-update"
    workdir.mkdir()
    update = updater.PreparedUpdate(
        "99.0", str(workdir), str(workdir / "setup.exe")
    )
    monkeypatch.setattr(updater, "other_instances", lambda: ["localize"])
    asked = []

    def cancel(parent, title, text, buttons):
        asked.append(text)
        return QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", cancel)
    started = []
    monkeypatch.setattr(updater, "start_install", started.append)
    updater._install_prepared(update, None)
    assert "localize" in asked[0]
    assert started == []
    assert not workdir.exists()


def test_unfinished_install_is_reported_at_start(qt_offscreen, monkeypatch):
    from PyQt6 import QtWidgets

    monkeypatch.setattr(updater, "can_self_update", lambda: True)
    monkeypatch.setattr(updater, "update_check_due", lambda: False)
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec", lambda box: shown.append(box.text())
    )
    _set_updates(Installing="99.0")
    updater.setup_gui_update_check()
    _wait_until(lambda: shown)
    assert "Picasso v99.0 was not installed" in shown[0]
