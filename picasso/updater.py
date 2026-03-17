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


def get_update_url(latest_version: str) -> str:
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
    #     if settings.get("Last update check", False):
    #         last = datetime.fromisoformat(settings["Last update check"])
    #         return datetime.now() - last > timedelta(hours=24)
    # except Exception:
    #     return True #TODO: uncomment
    return True  # always check for updates (disable once we have a working system)


def mark_checked():
    settings = io.load_user_settings()
    settings["Last update check"] = datetime.now().isoformat()
    io.save_user_settings(settings)


def check_and_notify(notify_callback):
    """Run update check in background thread, call notify_callback if
    update found. Returns the thread so callers can join it."""

    def _check():
        if not should_check_today():
            return
        mark_checked()
        available, latest = is_update_available()
        if available:
            notify_callback(latest)

    t = threading.Thread(target=_check, daemon=True)
    t.start()
    return t


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
        mark_checked()
        url = get_update_url(latest_version)
        QtWidgets.QMessageBox.information(
            parent,
            "Update Available",
            f"Picasso v{latest_version} is available!\n\n{url}",
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
