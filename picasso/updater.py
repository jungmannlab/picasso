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
import yaml

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
        return f"pip install --upgrade picassosr"
    except importlib.metadata.PackageNotFoundError:
        pass

    # Fallback: running from source / GitHub clone
    return URL_GITHUB_REPO


def should_check_today() -> bool:
    """Only check once per day."""
    try:
        settings = io.load_user_settings()
        if settings.get("last_checked", False):
            last = datetime.fromisoformat(settings["last_checked"])
            return datetime.now() - last > timedelta(hours=24)
    except Exception:
        return True


def mark_checked():
    settings = io.load_user_settings()
    settings["last_checked"] = datetime.now().isoformat()
    io.save_user_settings(settings)


def check_and_notify(notify_callback):
    """Run update check in background thread, call notify_callback if
    update found."""

    def _check():
        if not should_check_today():
            return
        mark_checked()
        available, latest = is_update_available()
        if available:
            notify_callback(latest)

    threading.Thread(target=_check, daemon=True).start()
