"""
picasso.updater.py
~~~~~~~~~~~~~~~~~~~

Manage Picasso update notifications and checks.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import sys
import threading
import requests

from datetime import datetime, timedelta
from packaging.version import Version
from . import io, __version__

URL_LATEST_RELEASE_API = (
    "https://api.github.com/repos/jungmannlab/picasso/releases/latest"
)
URL_LATEST_RELEASE = "https://github.com/jungmannlab/picasso/releases"
URL_GITHUB_REPO = "https://github.com/jungmannlab/picasso"


def get_latest_version() -> str | None:
    """Fetch the latest release tag from GitHub.

    Returns
    -------
    version : str or None
        The tag with its leading ``v`` stripped, or None if the request
        failed - an update check must never crash the app.
    """
    try:
        response = requests.get(URL_LATEST_RELEASE_API, timeout=5)
        response.raise_for_status()
        tag = response.json()["tag_name"]
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
    if not update_check_due():
        return

    from PyQt6 import QtCore, QtWidgets

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
            if box.clickedButton() == open_btn:
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
