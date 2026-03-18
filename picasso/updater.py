"""
picasso.updater.py
~~~~~~~~~~~~~~~~~~~

Manage Picasso update notifications and checks.

:author: Rafal Kowalewski 2026
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
    """Fetch the latest release tag from GitHub."""
    try:
        response = requests.get(URL_LATEST_RELEASE_API, timeout=5)
        response.raise_for_status()
        tag = response.json()["tag_name"]
        return tag.lstrip("v")  # normalize "v1.2.3" → "1.2.3"
    except Exception:
        return None  # never crash the app due to update check


def is_update_available() -> tuple[bool, str | None]:
    """Returns (update_available, latest_version)."""
    latest = get_latest_version()
    if latest is None:
        return False, None
    try:
        return Version(latest) > Version(__version__), latest
    except Exception:
        return False, None


def get_update_url() -> str:
    """Return the appropriate update URL based on how the app is running."""

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
    """Only check once per day."""
    # try:
    #     settings = io.load_user_settings()
    #     if settings["Updates"].get("Last update check", False):
    #         last = datetime.fromisoformat(settings["Updates"]["Last update check"])
    #         return datetime.now() - last > timedelta(hours=24)
    # except Exception:
    #     return True #TODO: uncomment
    return True  # always check for updates (disable once we have a working system)


def skip_version(version: str) -> None:
    """Mark the current latest version as "skipped" so the user won't be
    notified about it again."""
    settings = io.load_user_settings()
    settings["Updates"]["Skipped version"] = version
    io.save_user_settings(settings)


def snooze_until(days: int) -> None:
    """User chose 'remind me later' — suppress for N days."""
    settings = io.load_user_settings()
    settings["Updates"]["Snoozed_until"] = (
        datetime.now() + timedelta(days=days)
    ).isoformat()
    io.save_user_settings(settings)


def disable_updates() -> None:
    """User chose 'don't check for updates' — disable future checks."""
    settings = io.load_user_settings()
    settings["Updates"]["Disabled"] = True
    io.save_user_settings(settings)


def should_notify(latest_version: str) -> bool:
    """Check user settings to decide whether to show update notification
    for the given version."""
    settings = io.load_user_settings()
    if settings["Updates"].get("Disabled", False):
        return False

    if settings["Updates"].get("Skipped version") == latest_version:
        return False

    snoozed = settings["Updates"].get("Snoozed_until")
    if snoozed and datetime.now() < datetime.fromisoformat(snoozed):
        return False  # still within snooze window

    return should_check_today()


def mark_checked():
    settings = io.load_user_settings()
    settings["Updates"]["Last update check"] = datetime.now().isoformat()
    io.save_user_settings(settings)


def check_and_notify(notify_callback):
    """Run update check in background thread, call notify_callback if
    update found. Returns the thread so callers can join it."""

    def _check():
        available, latest = is_update_available()
        if not should_notify(latest):
            return
        mark_checked()
        if available:
            notify_callback(latest)

    t = threading.Thread(target=_check, daemon=True)
    t.start()
    return t


def cli_notify_update(latest_version):
    url = get_update_url()
    print(
        f"\n⚡ Picasso update available: v{latest_version}\n\n{url}",
        file=sys.stderr,
    )
    print(
        f"   Would you like to silence update notifications?", file=sys.stderr
    )
    print(f"   [1] Remind me in 7 days", file=sys.stderr)
    print(f"   [2] Skip this version", file=sys.stderr)
    print(f"   [9] Disable update checks", file=sys.stderr)
    print(
        f"   [Enter] Do nothing for now (remind tomorrow)\n", file=sys.stderr
    )

    choice = input("   Choice: ").strip()
    if choice == "1":
        snooze_until(days=7)
    elif choice == "2":
        skip_version(latest_version)
    elif choice == "9":
        disable_updates()


def setup_gui_update_check(parent=None):
    """Schedule a background update check that shows a QMessageBox
    if an update is available.

    Must be called after QApplication is created (e.g. after
    ``window.show()``). The HTTP request runs in a background thread;
    the dialog is delivered to the main thread via a Qt signal.
    """
    if not should_check_today():
        return

    from PyQt5 import QtCore, QtWidgets

    class _Notifier(QtCore.QObject):
        update_found = QtCore.pyqtSignal(str)

    notifier = _Notifier()

    def _show_dialog(latest_version):
        import webbrowser

        mark_checked()
        msg = get_update_url()
        # if one-click-installer is used, allow the user to open the release
        # page
        if msg == URL_LATEST_RELEASE:
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Information,
                "Update available",
                f"Picasso v{latest_version} is available!\n\n{msg}",
                parent=parent,
            )
            open_btn = box.addButton(
                "Open in Browser", QtWidgets.QMessageBox.ActionRole
            )
            remind_btn = box.addButton(
                "Remind me in 7 days", QtWidgets.QMessageBox.ActionRole
            )
            skip_btn = box.addButton(
                "Skip this version", QtWidgets.QMessageBox.ActionRole
            )
            disable_btn = box.addButton(
                "Don't check for updates", QtWidgets.QMessageBox.ActionRole
            )
            close_btn = box.addButton(QtWidgets.QMessageBox.Close)
            box.exec_()
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

    notifier.update_found.connect(_show_dialog)

    def _check():
        available, latest = is_update_available()
        if available:
            notifier.update_found.emit(latest)

    threading.Thread(target=_check, daemon=True).start()

    # prevent garbage collection of the QObject
    if parent is not None:
        parent._update_notifier = notifier
    else:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app._update_notifier = notifier
