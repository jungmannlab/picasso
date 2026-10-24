"""
picasso.updater.py
~~~~~~~~~~~~~~~~~~~

Manage Picasso update notifications and checks, and install updates
from within the one-click builds ("Install Now" in the notification).

Picasso downloads the installer of its edition from the GitHub release,
checks it against the SHA-256 digest GitHub publishes for it, quits and
installs it, then opens the same module again. The files of a running
app cannot be replaced (Windows) or must not be (macOS, where modules
not yet imported would come from the new version), so the installation
runs in a small helper script that Picasso starts just before quitting
and that waits for it to exit:

* Windows: the Inno Setup installer runs with ``/SILENT`` into the
  current installation folder, which keeps the selected modules. It
  asks for administrator rights as when installed by hand.
* macOS: ``Picasso.app`` and the module apps already installed next to
  it (e.g. ``Picasso Render.app``) are copied from the disk image while
  Picasso still runs, under hidden names, so that missing rights to the
  Applications folder are reported before Picasso quits; the helper
  only swaps them in. If the swap fails, the old apps are restored and
  the disk image opens in Finder, to install it by hand.

The helper logs to ``~/.picasso/logs/update.log``. At the next start,
Picasso tells if the update was not installed
(``report_unfinished_install``).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests
from packaging.version import Version

from . import __version__, diagnostics, io, lib

URL_LATEST_RELEASE_API = (
    "https://api.github.com/repos/jungmannlab/picasso/releases/latest"
)
URL_LATEST_RELEASE = "https://github.com/jungmannlab/picasso/releases"
URL_GITHUB_REPO = "https://github.com/jungmannlab/picasso"

#: Bytes read at once while downloading or unpacking an installer; also
#: the unit of the progress bar.
DOWNLOAD_CHUNK = 1 << 20

#: How long the helper that installs an update waits for Picasso to
#: quit, in seconds.
QUIT_TIMEOUT = 600


def get_latest_release() -> dict | None:
    """Fetch the latest release from GitHub.

    Returns
    -------
    release : dict or None
        The release as described by the GitHub API, e.g. with its
        ``tag_name`` and ``assets``, or None if the request failed - an
        update check must never crash the app.
    """
    try:
        response = requests.get(URL_LATEST_RELEASE_API, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None  # never crash the app due to update check


def get_latest_version() -> str | None:
    """Fetch the latest release tag from GitHub.

    Returns
    -------
    version : str or None
        The tag with its leading ``v`` stripped, or None if the request
        failed - an update check must never crash the app.
    """
    try:
        tag = get_latest_release()["tag_name"]
        return tag.lstrip("v")  # normalize "v1.2.3" → "1.2.3"
    except Exception:
        return None  # never crash the app due to update check


def is_update_available() -> tuple[bool, str | None]:
    """Compare the installed version against the latest release.

    Returns
    -------
    update_available : bool
        True if the latest release is newer than the running version.
    latest_version : str or None
        The latest release, or None if it could not be determined.
    """
    latest = get_latest_version()
    if latest is None:
        return False, None
    try:
        return Version(latest) > Version(__version__), latest
    except Exception:
        return False, None


def get_update_url() -> str:
    """Return the appropriate update instructions for how the app is
    running.

    Returns
    -------
    message : str
        The releases page URL for a bundled ``.exe``, the ``pip install
        --upgrade`` command for a PyPI install, or the repository URL when
        running from source.
    """

    # PyInstaller sets this attribute when running as a bundled .exe
    if getattr(sys, "frozen", False):
        # Running as .exe → point to GitHub releases page for new installer
        return URL_LATEST_RELEASE

    # Check if installed as a package (pip/PyPI)
    try:
        import importlib.metadata

        importlib.metadata.distribution("picassosr")
        message = (
            "To update Picasso, run the following command in your"
            " terminal:\n\npip install --upgrade picassosr\n"
        )
        return message
    except importlib.metadata.PackageNotFoundError:
        pass

    # Fallback: running from source / GitHub clone
    message = (
        "\nTo update Picasso, please visit the GitHub repository:\n\n"
        f"{URL_GITHUB_REPO}"
    )
    return message


def should_check_today() -> bool:
    """Whether the daily update check is due.

    Returns
    -------
    should_check : bool
        True if the last check was more than 24 hours ago, or if no check
        has been recorded (including when the settings cannot be read).
    """
    try:
        settings = io.load_user_settings()
        last = settings["Updates"].get("Last update check", False)
        if not last:
            return True
        return datetime.now() - datetime.fromisoformat(last) > timedelta(
            hours=24
        )
    except Exception:
        return True


def skip_version(version: str) -> None:
    """Mark a version as "skipped" so the user won't be notified about it
    again.

    Parameters
    ----------
    version : str
        The version to skip.
    """
    settings = io.load_user_settings()
    settings["Updates"]["Skipped version"] = version
    io.save_user_settings(settings)


def snooze_until(days: int) -> None:
    """User chose 'remind me later' — suppress notifications for a while.

    Parameters
    ----------
    days : int
        How many days from now to stay quiet for.
    """
    settings = io.load_user_settings()
    settings["Updates"]["Snoozed until"] = (
        datetime.now() + timedelta(days=days)
    ).isoformat()
    io.save_user_settings(settings)


def disable_updates() -> None:
    """User chose 'don't check for updates' — disable future checks."""
    settings = io.load_user_settings()
    settings["Updates"]["Disabled"] = True
    io.save_user_settings(settings)


def update_check_due() -> bool:
    """Whether to check for updates now, decided before any request.

    Returns
    -------
    due : bool
        False if update checks are disabled, the "remind me later"
        window is still open or a check ran in the last 24 hours.
    """
    try:
        updates = io.load_user_settings()["Updates"]
        if updates.get("Disabled", False):
            return False
        snoozed = updates.get("Snoozed until")
        if snoozed and datetime.now() < datetime.fromisoformat(snoozed):
            return False
    except Exception:
        pass
    return should_check_today()


def is_skipped(latest_version: str) -> bool:
    """Whether the user chose "skip this version" for this version.

    Parameters
    ----------
    latest_version : str
        The version that is available.

    Returns
    -------
    skipped : bool
        True if notifications about this version are switched off.
    """
    try:
        updates = io.load_user_settings()["Updates"]
        return updates.get("Skipped version") == latest_version
    except Exception:
        return False


def mark_checked():
    """Record that an update check ran now, so the next one waits a day."""
    settings = io.load_user_settings()
    settings["Updates"]["Last update check"] = datetime.now().isoformat()
    io.save_user_settings(settings)


class CliUpdateCheck:
    """Check for updates while a command runs and tell about an update
    once the command is done.

    The check starts on creation if one is due (``update_check_due``).
    Only the request runs in a background thread; the user settings are
    read and written, and the user is asked, in the calling thread, by
    ``finish``, so the notification neither interrupts the output of the
    command nor answers the questions it asks.
    """

    def __init__(self) -> None:
        self._result = None
        self._thread = None
        if update_check_due():
            self._thread = threading.Thread(target=self._fetch, daemon=True)
            self._thread.start()

    def _fetch(self) -> None:
        self._result = is_update_available()

    def finish(self, timeout: float = 6) -> None:
        """Wait for the check and tell about an update, see
        ``cli_notify_update``.

        Parameters
        ----------
        timeout : float, optional
            How long to wait for the release server, in seconds. Default
            is 6.
        """
        if self._thread is None:
            return
        self._thread.join(timeout)
        # also without an answer, e.g., offline: the next check waits a
        # day instead of slowing down every command
        try:
            mark_checked()
        except OSError:  # e.g., a read-only home folder
            pass
        if self._result is None:
            return
        available, latest = self._result
        if available and not is_skipped(latest):
            cli_notify_update(latest)


def _interactive() -> bool:
    """Whether someone can answer at the terminal: stdin and stderr are
    a terminal, unlike in scripts, pipes and batch jobs."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):  # None or closed
        return False


def cli_notify_update(latest_version):
    """Tell the user about an update on stderr and, in a terminal, offer
    to silence future notifications.

    Without a terminal, e.g., in a batch job, it only says how to switch
    the checks off and does not wait for an answer.

    Parameters
    ----------
    latest_version : str
        The version that is available.
    """
    url = get_update_url()
    print(
        f"\n⚡ Picasso update available: v{latest_version}\n\n{url}",
        file=sys.stderr,
    )
    if not _interactive():
        print(
            "   To switch off update checks, set Disabled: true in the "
            f"Updates section of {io._user_settings_filename()}.\n",
            file=sys.stderr,
        )
        return
    print(
        "   Would you like to silence update notifications?", file=sys.stderr
    )
    print("   [1] Remind me in 7 days", file=sys.stderr)
    print("   [2] Skip this version", file=sys.stderr)
    print("   [9] Disable update checks", file=sys.stderr)
    print("   [Enter] Do nothing for now (remind tomorrow)\n", file=sys.stderr)

    try:
        choice = input("   Choice: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return
    if choice == "1":
        snooze_until(days=7)
    elif choice == "2":
        skip_version(latest_version)
    elif choice == "9":
        disable_updates()


def setup_gui_update_check(parent=None):  # noqa: C901
    """Schedule a background update check that shows a QMessageBox
    if an update is available.

    Must be called after QApplication is created (e.g. after
    ``window.show()``). The HTTP request runs in a background thread;
    the dialog is delivered to the main thread via a Qt signal.

    Parameters
    ----------
    parent : QtWidgets.QWidget, optional
        Parent of the message box. Default None.
    """
    from PyQt6 import QtCore, QtWidgets

    if can_self_update():
        # after Picasso quit to install an update, once the window shows
        QtCore.QTimer.singleShot(0, lambda: report_unfinished_install(parent))

    if not update_check_due():
        return

    class _Notifier(QtCore.QObject):
        checked = QtCore.pyqtSignal(bool, str)

    notifier = _Notifier()

    def _on_checked(available, latest_version):
        # in the main thread, like all access to the user settings
        mark_checked()
        if available and not is_skipped(latest_version):
            _show_dialog(latest_version)

    def _show_dialog(latest_version):
        import webbrowser

        msg = get_update_url()
        # if one-click-installer is used, allow the user to open the release
        # page
        if msg == URL_LATEST_RELEASE:
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Information,
                "Update available",
                f"Picasso v{latest_version} is available!\n\n{msg}",
                parent=parent,
            )
            # download and install it from within Picasso, see
            # install_update_from_gui
            install_btn = None
            if can_self_update():
                install_btn = box.addButton(
                    "Install Now", QtWidgets.QMessageBox.ButtonRole.AcceptRole
                )
            open_btn = box.addButton(
                "Open in Browser", QtWidgets.QMessageBox.ButtonRole.ActionRole
            )
            remind_btn = box.addButton(
                "Remind me in 7 days",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
            skip_btn = box.addButton(
                "Skip this version",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
            disable_btn = box.addButton(
                "Don't check for updates",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
            box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
            box.exec()
            if install_btn is not None and box.clickedButton() == install_btn:
                install_update_from_gui(parent)
            elif box.clickedButton() == open_btn:
                webbrowser.open(URL_LATEST_RELEASE)
            elif box.clickedButton() == remind_btn:
                snooze_until(days=7)
            elif box.clickedButton() == skip_btn:
                skip_version(latest_version)
            elif box.clickedButton() == disable_btn:
                disable_updates()
        # if installed via pip, show the pip command
        else:
            QtWidgets.QMessageBox.information(
                parent,
                "Update available",
                f"Picasso v{latest_version} is available!\n\n{msg}",
            )

    notifier.checked.connect(_on_checked)

    def _check():
        available, latest = is_update_available()
        notifier.checked.emit(available, latest or "")

    threading.Thread(target=_check, daemon=True).start()

    # prevent garbage collection of the QObject
    if parent is not None:
        parent._update_notifier = notifier
    else:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app._update_notifier = notifier


class UpdateError(Exception):
    """The update could not be downloaded or prepared; the message says
    why, for the user."""


@dataclass
class PreparedUpdate:
    """A downloaded update, ready to be installed by ``start_install``.

    Attributes
    ----------
    version : str
        Version that is installed.
    workdir : str
        Temporary folder holding the download, removed by the helper
        after a successful installation.
    installer : str
        The Windows installer or the macOS disk image.
    staged : list of (str, str)
        macOS only: (staged copy, installed app) of every app to swap.
    """

    version: str
    workdir: str
    installer: str
    staged: list[tuple[str, str]] = field(default_factory=list)


def can_self_update() -> bool:
    """Whether Picasso runs from a one-click build that can install
    updates by itself.

    Returns
    -------
    possible : bool
        True for the Windows and macOS one-click builds; False for pip,
        conda or source installs, which update with pip.
    """
    if not getattr(sys, "frozen", False):
        return False
    if sys.platform == "win32":
        return True
    return sys.platform == "darwin" and _app_bundle() is not None


def _app_bundle() -> str | None:
    """The running ``Picasso.app`` (macOS), or None outside a bundle."""
    # sys.executable is .../Picasso.app/Contents/MacOS/picasso
    macos_dir = os.path.dirname(os.path.realpath(sys.executable))
    contents = os.path.dirname(macos_dir)
    bundle = os.path.dirname(contents)
    if os.path.basename(contents) != "Contents" or not bundle.endswith(".app"):
        return None
    return bundle


def _is_cuda_build() -> bool:
    """Whether this is the CUDA edition of the Windows installer."""
    return importlib.util.find_spec("numba_cuda") is not None


def find_installer(
    release: dict,
    platform: str | None = None,
    cuda: bool | None = None,
) -> dict | None:
    """Find the installer of this build among the assets of a release.

    The assets are recognized by name, as the release scripts in
    ``release/`` name them: "windows" in the Windows installers, with
    "cuda" (or "gpu", before 1.0) for the CUDA edition, and "mac" or
    "apple" in the macOS one. The installer itself is preferred to a
    zip of it.

    Parameters
    ----------
    release : dict
        A release as returned by the GitHub API, see
        ``get_latest_release``.
    platform : str, optional
        ``sys.platform`` of the build. Default None, the running one.
    cuda : bool, optional
        Whether the CUDA edition is wanted (Windows). Default None, the
        running edition.

    Returns
    -------
    asset : dict or None
        The GitHub asset (``name``, ``size``, ``browser_download_url``
        and ``digest``), or None if the release has none for this build.
    """
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        keywords, suffixes = ("windows",), (".exe", ".zip")
        if cuda is None:
            cuda = _is_cuda_build()
    elif platform == "darwin":
        keywords, suffixes = ("mac", "apple"), (".dmg", ".zip")
    else:
        return None
    candidates = []
    for asset in release.get("assets", []):
        name = str(asset.get("name", "")).lower()
        if not any(keyword in name for keyword in keywords):
            continue
        suffix = os.path.splitext(name)[1]
        if suffix not in suffixes:
            continue
        if platform == "win32" and ("cuda" in name or "gpu" in name) != cuda:
            continue
        candidates.append((suffixes.index(suffix), asset))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def download_installer(
    asset: dict,
    folder: str,
    progress: lib.Progress | None = None,
) -> str:
    """Download a release asset and check it against its size and the
    SHA-256 digest GitHub publishes for it.

    Parameters
    ----------
    asset : dict
        The GitHub asset, see ``find_installer``.
    folder : str
        Folder to save it in.
    progress : lib.Progress, optional
        Tracker of the download, in MB. Default None.

    Returns
    -------
    path : str
        The downloaded file.

    Raises
    ------
    UpdateError
        If the download is incomplete or does not match its digest.
    OperationCanceled
        If the tracker reports that the user canceled the download.
    """
    name = os.path.basename(asset["name"])
    path = os.path.join(folder, name)
    size = int(asset.get("size") or 0)
    digest = hashlib.sha256()
    received = 0
    with lib.tracked_progress(progress) as tracker:
        lib.start_phase(
            tracker,
            f"Downloading {name}",
            math.ceil(size / DOWNLOAD_CHUNK),
            "MB",
        )
        with requests.get(
            asset["browser_download_url"], stream=True, timeout=30
        ) as response:
            response.raise_for_status()
            with open(path, "wb") as file:
                for chunk in response.iter_content(DOWNLOAD_CHUNK):
                    file.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    tracker.set_value(received // DOWNLOAD_CHUNK)
    if size and received != size:
        raise UpdateError(
            f"The download of {name} is incomplete ({received} of {size}"
            " bytes). Please try again."
        )
    expected = str(asset.get("digest") or "")
    if expected.startswith("sha256:") and digest.hexdigest() != expected[7:]:
        raise UpdateError(
            f"The download of {name} is damaged: it does not match the"
            " checksum published with the release. Please try again."
        )
    return path


def extract_installer(
    path: str,
    platform: str | None = None,
    progress: lib.Progress | None = None,
) -> str:
    """Take the installer out of a downloaded zip, next to it, and remove
    the zip. A download that is not a zip is returned as is.

    Parameters
    ----------
    path : str
        The download.
    platform : str, optional
        ``sys.platform`` of the build. Default None, the running one.
    progress : lib.Progress, optional
        Tracker of the unpacking, in MB. Default None.

    Returns
    -------
    installer : str
        The Windows installer (.exe) or the macOS disk image (.dmg).

    Raises
    ------
    UpdateError
        If the zip does not hold exactly one installer.
    """
    if not path.lower().endswith(".zip"):
        return path
    platform = sys.platform if platform is None else platform
    suffix = ".exe" if platform == "win32" else ".dmg"
    with zipfile.ZipFile(path) as archive:
        members = [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith(suffix)
            and not info.filename.startswith("__MACOSX")
        ]
        if len(members) != 1:
            raise UpdateError(
                f"{os.path.basename(path)} does not contain an installer."
            )
        member = members[0]
        # only the file name, never a path from the archive
        target = os.path.join(
            os.path.dirname(path), os.path.basename(member.filename)
        )
        with lib.tracked_progress(progress) as tracker:
            lib.start_phase(
                tracker,
                f"Unpacking {os.path.basename(target)}",
                math.ceil(member.file_size / DOWNLOAD_CHUNK),
                "MB",
            )
            done = 0
            with archive.open(member) as source, open(target, "wb") as dest:
                while chunk := source.read(DOWNLOAD_CHUNK):
                    dest.write(chunk)
                    done += len(chunk)
                    tracker.set_value(done // DOWNLOAD_CHUNK)
    os.remove(path)
    return target


def _run(command: list[str]) -> subprocess.CompletedProcess:
    """Run a system tool, without the library path of the frozen app."""
    return subprocess.run(
        command, capture_output=True, text=True, env=_clean_environment()
    )


def _staging_path(app: str) -> str:
    """Hidden name next to ``app`` under which its update is copied."""
    folder, name = os.path.split(app)
    return os.path.join(folder, f".{name}.update")


def stage_macos_apps(
    disk_image: str,
    bundle: str,
    progress: lib.Progress | None = None,
) -> list[tuple[str, str]]:
    """Copy the apps of a disk image next to the installed ones, under
    hidden names, for the helper to swap in once Picasso has quit.

    ``Picasso.app`` replaces ``bundle``; a module app (e.g.
    ``Picasso Render.app``) is copied only if it is installed in the
    same folder, since the module apps may also have been left out or
    moved elsewhere.

    Parameters
    ----------
    disk_image : str
        The downloaded .dmg.
    bundle : str
        The installed ``Picasso.app``.
    progress : lib.Progress, optional
        Tracker, counting the apps copied. Default None.

    Returns
    -------
    staged : list of (str, str)
        (staged copy, installed app), ``Picasso.app`` first.

    Raises
    ------
    UpdateError
        If the disk image cannot be opened or holds no Picasso.app, or
        if the apps cannot be copied, e.g. without the rights to change
        the Applications folder. Nothing is left behind.
    """
    folder = os.path.dirname(bundle)
    mount = tempfile.mkdtemp(prefix="picasso-update-mount-")
    staged = []
    with lib.tracked_progress(progress) as tracker:
        lib.start_phase(tracker, "Opening the disk image", 0)
        attach = _run(
            [
                "hdiutil", "attach", disk_image, "-nobrowse", "-readonly",
                "-noautoopen", "-mountpoint", mount,
            ]
        )  # fmt: skip
        if attach.returncode != 0:
            os.rmdir(mount)
            raise UpdateError(
                f"The disk image could not be opened:\n{attach.stderr}"
            )
        try:
            apps = sorted(
                name for name in os.listdir(mount) if name.endswith(".app")
            )
            main = [name for name in apps if name.lower() == "picasso.app"]
            if not main:
                raise UpdateError("The disk image holds no Picasso.app.")
            targets = [(main[0], bundle)] + [
                (name, os.path.join(folder, name))
                for name in apps
                if name not in main
                and os.path.isdir(os.path.join(folder, name))
            ]
            lib.start_phase(tracker, "Copying the new version", len(targets))
            for i, (name, target) in enumerate(targets):
                tracker.set_value(i)  # also where a cancel takes effect
                staging = _staging_path(target)
                shutil.rmtree(staging, ignore_errors=True)
                # ditto keeps the symlinks and the signature of the bundle
                copy = _run(["ditto", os.path.join(mount, name), staging])
                if copy.returncode != 0:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise UpdateError(
                        f"{name} could not be copied to {folder}:\n"
                        f"{copy.stderr}\nYou may need administrator rights"
                        " to change this folder."
                    )
                staged.append((staging, target))
            tracker.set_value(len(targets))
        except BaseException:
            for staging, _ in staged:
                shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            _run(["hdiutil", "detach", mount, "-force"])
            shutil.rmtree(mount, ignore_errors=True)
    return staged


def discard_update(update: PreparedUpdate) -> None:
    """Remove a prepared update that will not be installed.

    Parameters
    ----------
    update : PreparedUpdate
        The update.
    """
    for staging, _ in update.staged:
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(update.workdir, ignore_errors=True)


def prepare_update(
    asset: dict,
    version: str,
    progress: lib.Progress | None = None,
) -> PreparedUpdate:
    """Download the installer and, on macOS, copy its apps next to the
    installed ones; everything that can be done while Picasso runs.

    Parameters
    ----------
    asset : dict
        The installer, see ``find_installer``.
    version : str
        Its version.
    progress : lib.Progress, optional
        Tracker. Default None.

    Returns
    -------
    update : PreparedUpdate
        The update, for ``start_install``.

    Raises
    ------
    UpdateError
        See ``download_installer``, ``extract_installer`` and
        ``stage_macos_apps``.
    OperationCanceled
        If the user canceled. Nothing is left behind in either case.
    """
    workdir = tempfile.mkdtemp(prefix="picasso-update-")
    try:
        installer = extract_installer(
            download_installer(asset, workdir, progress), progress=progress
        )
        update = PreparedUpdate(version, workdir, installer)
        if sys.platform == "darwin":
            update.staged = stage_macos_apps(
                installer, _app_bundle(), progress
            )
        return update
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


def other_instances() -> list[str]:
    """Picasso programs of this installation running in other processes,
    which must be closed before it can be updated.

    Returns
    -------
    names : list of str
        Their module (e.g. "render"), or "picasso" if not known; empty
        if none runs or the processes cannot be listed.
    """
    import psutil

    try:
        me = psutil.Process()
        own = {me.pid} | {child.pid for child in me.children(recursive=True)}
        folder = _normalized_dir(sys.executable)
        names = []
        for process in psutil.process_iter(["pid", "exe", "cmdline"]):
            exe = process.info["exe"]
            if not exe or process.info["pid"] in own:
                continue
            if _normalized_dir(exe) != folder:
                continue
            cmdline = process.info["cmdline"] or []
            # multiprocessing workers of another Picasso program run the
            # same executable, with options instead of a module
            if len(cmdline) > 1 and cmdline[1].startswith("-"):
                continue
            names.append(cmdline[1] if len(cmdline) > 1 else "picasso")
        return names
    except Exception:  # noqa: BLE001 - the installer closes them anyway
        return []


def _normalized_dir(path: str) -> str:
    return os.path.normcase(os.path.dirname(os.path.realpath(path)))


def _relaunch_arguments() -> list[str]:
    """The module to open again after the update, e.g. ["render"]; files
    passed on the command line are not opened again."""
    return [arg for arg in sys.argv[1:2] if not arg.startswith("-")]


def _update_log() -> str:
    return os.path.join(diagnostics.log_dir(), "update.log")


def _powershell_quote(text: str) -> str:
    """A PowerShell string literal: single quotes, no expansion."""
    return "'" + text.replace("'", "''") + "'"


def windows_script(
    update: PreparedUpdate,
    pid: int,
    install_dir: str,
    relaunch: list[str],
) -> str:
    """The PowerShell helper that installs an update on Windows.

    Parameters
    ----------
    update : PreparedUpdate
        The update.
    pid : int
        Process ID of Picasso, which the helper waits for.
    install_dir : str
        The installation folder, e.g. ``C:\\Picasso``.
    relaunch : list of str
        Executable and arguments to start once installed.

    Returns
    -------
    script : str
        The script.
    """
    q = _powershell_quote
    log = _update_log()
    # /SILENT shows the progress of the installation but asks nothing;
    # /CLOSEAPPLICATIONS closes Picasso programs still using its files
    setup_arguments = (
        "/SILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS"
        f' /DIR="{install_dir}"'
        f' /LOG="{os.path.join(diagnostics.log_dir(), "update-setup.log")}"'
    )
    start = f"Start-Process -FilePath {q(relaunch[0])}"
    if relaunch[1:]:
        start += " -ArgumentList " + ",".join(q(arg) for arg in relaunch[1:])
    return f"""\
# Installs Picasso v{update.version}; written by picasso.updater
$log = {q(log)}
function Write-Log($text) {{
    "$(Get-Date -Format s) $text" | Out-File -FilePath $log -Append -Encoding utf8
}}
Write-Log 'Waiting for Picasso (process {pid}) to quit'
Wait-Process -Id {pid} -Timeout {QUIT_TIMEOUT} -ErrorAction SilentlyContinue
try {{
    Write-Log {q("Running " + update.installer)}
    $setup = Start-Process -FilePath {q(update.installer)} -ArgumentList {q(setup_arguments)} -Wait -PassThru
    $code = $setup.ExitCode
}} catch {{
    $code = $_.Exception.Message
}}
Write-Log "Installer finished: $code"
{start}
if ($code -eq 0) {{
    Set-Location $env:TEMP
    Remove-Item -LiteralPath {q(update.workdir)} -Recurse -Force -ErrorAction SilentlyContinue
}}
"""  # noqa: E501


def _sh_quote(text: str) -> str:
    """A POSIX shell string literal."""
    return "'" + text.replace("'", "'\\''") + "'"


def macos_script(
    update: PreparedUpdate,
    pid: int,
    relaunch: list[str],
) -> str:
    """The bash helper that swaps the staged apps in on macOS.

    Parameters
    ----------
    update : PreparedUpdate
        The update, with its staged apps.
    pid : int
        Process ID of Picasso, which the helper waits for.
    relaunch : list of str
        Command that opens Picasso again.

    Returns
    -------
    script : str
        The script.
    """
    q = _sh_quote
    (main_staging, main_target), *modules = update.staged
    # a module app only starts Picasso.app, so the update counts as
    # installed once Picasso.app is
    module_swaps = "\n".join(
        f"    swap {q(staging)} {q(target)} || rm -rf {q(staging)}"
        for staging, target in modules
    )
    cleanups = "\n".join(
        f"    rm -rf {q(staging)}" for staging, _ in update.staged
    )
    return f"""\
#!/bin/bash
# Installs Picasso v{update.version}; written by picasso.updater
exec >>{q(_update_log())} 2>&1
echo "$(date '+%Y-%m-%dT%H:%M:%S') Waiting for Picasso (process {pid}) to quit"
waited=0
while kill -0 {pid} 2>/dev/null && [ $waited -lt {QUIT_TIMEOUT * 2} ]; do
    sleep 0.5
    waited=$((waited + 1))
done

# replace the app $2 by its staged copy $1, keeping the old one until
# the new one is in place
swap() {{
    local old
    old="$(dirname "$2")/.$(basename "$2").old"
    rm -rf "$old"
    if [ -e "$2" ]; then
        mv "$2" "$old" || return 1
    fi
    if ! mv "$1" "$2"; then
        mv "$old" "$2"
        return 1
    fi
    rm -rf "$old"
    echo "Installed $2"
}}

if swap {q(main_staging)} {q(main_target)}; then
{module_swaps}
    rm -rf {q(update.workdir)}
else
    echo "The update was not installed; opening the disk image"
{cleanups}
    open {q(update.installer)}
fi
{" ".join(q(arg) for arg in relaunch)}
"""


def _clean_environment() -> dict[str, str]:
    """The environment of Picasso without what the frozen app set for
    itself, so that system tools and the new Picasso start as from the
    desktop."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("_PYI_", "_MEI")) and key != "DYLD_LIBRARY_PATH"
    }
    # a frozen Picasso started by this one must not take itself for its
    # child, see the PyInstaller documentation on subprocesses
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def start_install(update: PreparedUpdate) -> None:
    """Start the helper that installs the update once Picasso has quit,
    see the module docstring. Picasso must quit right after.

    Parameters
    ----------
    update : PreparedUpdate
        The update, see ``prepare_update``.
    """
    os.makedirs(diagnostics.log_dir(), exist_ok=True)
    relaunch_args = _relaunch_arguments()
    pid = os.getpid()
    if sys.platform == "win32":
        import ctypes

        script = os.path.join(update.workdir, "install_update.ps1")
        text = windows_script(
            update,
            pid,
            os.path.dirname(sys.executable),
            [sys.executable] + relaunch_args,
        )
        # with a byte order mark, Windows PowerShell reads it as UTF-8,
        # e.g. for a user name with an umlaut in a path
        with open(script, "w", encoding="utf-8-sig") as file:
            file.write(text)
        # the frozen app points the DLL search at its own folder, which
        # PowerShell would otherwise inherit
        ctypes.windll.kernel32.SetDllDirectoryW(None)
        command = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-WindowStyle", "Hidden", "-File", script,
        ]  # fmt: skip
        flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
        subprocess.Popen(
            command,
            env=_clean_environment(),
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        script = os.path.join(update.workdir, "install_update.sh")
        relaunch = ["open", "-n", _app_bundle()]
        if relaunch_args:
            relaunch += ["--args"] + relaunch_args
        with open(script, "w", encoding="utf-8") as file:
            file.write(macos_script(update, pid, relaunch))
        subprocess.Popen(
            ["/bin/bash", script],
            env=_clean_environment(),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _mark_installing(version: str | None) -> None:
    """Remember the version being installed, for
    ``report_unfinished_install``; None forgets it."""
    settings = io.load_user_settings()
    if version is None:
        settings["Updates"].pop("Installing", None)
    else:
        settings["Updates"]["Installing"] = version
    io.save_user_settings(settings)


def unfinished_install() -> str | None:
    """The version of an update that Picasso quit to install but that is
    not running now, and forget about it.

    Returns
    -------
    version : str or None
        The version that was not installed, or None if there was no
        update or it was installed.
    """
    try:
        version = io.load_user_settings()["Updates"].get("Installing")
        if not version:
            return None
        _mark_installing(None)
        return version if Version(__version__) < Version(version) else None
    except Exception:  # noqa: BLE001 - must not stop Picasso starting
        return None


def report_unfinished_install(parent=None) -> None:
    """Tell the user if the update that Picasso quit to install was not
    installed, e.g. when the administrator rights were refused.

    Parameters
    ----------
    parent : QtWidgets.QWidget, optional
        Parent of the message box. Default None.
    """
    version = unfinished_install()
    if version is None:
        return
    import webbrowser

    from PyQt6 import QtWidgets

    box = QtWidgets.QMessageBox(
        QtWidgets.QMessageBox.Icon.Warning,
        "Update not installed",
        f"Picasso v{version} was not installed, this is still"
        f" v{__version__}.\n\nThe log of the update is in"
        f" {_update_log()}. You can also download the installer from the"
        " release page and install it by hand.",
        parent=parent,
    )
    open_button = box.addButton(
        "Open in Browser", QtWidgets.QMessageBox.ButtonRole.ActionRole
    )
    box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
    box.exec()
    if box.clickedButton() == open_button:
        webbrowser.open(URL_LATEST_RELEASE)


def install_update_from_gui(parent=None) -> None:
    """Download and install the latest release, asking the user on the
    way; the "Install Now" button of the update notification.

    Parameters
    ----------
    parent : QtWidgets.QWidget, optional
        Parent of the dialogs. Default None.
    """
    import webbrowser

    from PyQt6 import QtWidgets

    from . import lib_qt

    message_box = QtWidgets.QMessageBox
    release = lib_qt.run_with_status(
        get_latest_release, "Looking for the installer...", parent
    )
    if release is None:
        message_box.warning(
            parent,
            "Install update",
            "The release server could not be reached. Please check the"
            " internet connection and try again.",
        )
        return
    asset = find_installer(release)
    if asset is None:
        message_box.information(
            parent,
            "Install update",
            "No installer for this computer was found in the latest"
            " release. The release page opens in your browser instead.",
        )
        webbrowser.open(URL_LATEST_RELEASE)
        return
    version = str(release["tag_name"]).lstrip("v")
    size = int(asset.get("size") or 0) / 1e6
    answer = message_box.question(
        parent,
        "Install update",
        f"Download Picasso v{version} ({size:.0f} MB) and install it?\n\n"
        "When the download is done, Picasso closes, installs the update"
        " and opens again. Please save your work first.",
    )
    if answer != message_box.StandardButton.Yes:
        return

    def failed(error):
        if not isinstance(error, (UpdateError, OSError)):
            sys.excepthook(type(error), error, error.__traceback__)
            return
        message_box.warning(
            parent,
            "Update failed",
            f"Picasso v{version} could not be installed:\n\n{error}",
        )

    lib_qt.run_task(
        lambda progress: prepare_update(asset, version, progress),
        f"Downloading Picasso v{version}...",
        parent,
        lambda update: _install_prepared(update, parent),
        on_failed=failed,
        title="Install update",
    )


def _install_prepared(update: PreparedUpdate, parent) -> None:
    """Close Picasso and start the helper, once no other Picasso program
    runs and every window agreed to close."""
    from PyQt6 import QtWidgets

    message_box = QtWidgets.QMessageBox
    while others := other_instances():
        answer = message_box.warning(
            parent,
            "Install update",
            "Please close the other Picasso programs first: "
            + ", ".join(sorted(set(others)))
            + ".",
            message_box.StandardButton.Retry
            | message_box.StandardButton.Cancel,
        )
        if answer != message_box.StandardButton.Retry:
            discard_update(update)
            return
    app = QtWidgets.QApplication.instance()
    app.closeAllWindows()
    if any(widget.isVisible() for widget in app.topLevelWidgets()):
        # a window refused to close, e.g. while it saves
        discard_update(update)
        return
    try:
        _mark_installing(update.version)
    except Exception:  # noqa: BLE001 - only the report at the next start
        pass
    try:
        start_install(update)
    except Exception:
        discard_update(update)
        _mark_installing(None)
        raise
    app.quit()
