"""
picasso.notifications
~~~~~~~~~~~~~~~~~~~~~

Sound notifications played when a long-running job finishes.

The user picks one sound file from ``~/.picasso/notification_sounds``
(see :func:`picasso.io.notification_sounds_directory`); the choice is
stored under ``Sound_notification: filename`` in the user settings and
shared by all Picasso modules. Jobs report their duration to
:func:`notify_finished`, which plays the sound if the job took longer
than :data:`MIN_DURATION`. The ``File > Sound notifications`` submenu
is built by ``lib.add_sound_notifications_menu``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import traceback

from picasso import diagnostics, io

#: Minimum duration (seconds) of a job for its end to play a sound.
MIN_DURATION = 60

#: File extensions of the supported sound files.
SOUND_EXTENSIONS = (".mp3", ".wav")


def sounds_directory() -> str:
    """Return the user notification sounds directory, creating it if
    needed (``~/.picasso/notification_sounds``)."""
    return io.notification_sounds_directory()


def available_sounds() -> list[str]:
    """List the supported sound files in the notification sounds
    directory.

    Returns
    -------
    filenames : list of str
        Sorted file names (with extension) of the ``.mp3`` and ``.wav``
        files in ``~/.picasso/notification_sounds``.
    """
    directory = sounds_directory()
    return sorted(
        name
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and os.path.splitext(name)[1].lower() in SOUND_EXTENSIONS
    )


def selected_sound() -> str | None:
    """Return the file name of the selected sound.

    Returns
    -------
    filename : str or None
        The selected file name, or None if no sound is selected or the
        selected file is no longer available.
    """
    section = io.load_user_settings().get("Sound_notification")
    filename = section.get("filename") if isinstance(section, dict) else None
    if isinstance(filename, str) and filename in available_sounds():
        return filename
    return None


def select_sound(filename: str | None) -> None:
    """Store the selected sound in the user settings.

    Parameters
    ----------
    filename : str or None
        File name (with extension) of a sound in the notification
        sounds directory, or None for no sound.
    """
    settings = io.load_user_settings()
    settings["Sound_notification"]["filename"] = filename
    io.save_user_settings(settings)


def play(filename: str | None) -> None:
    """Play a sound from the notification sounds directory without
    blocking.

    A missing or broken audio backend (which happens in the one-click
    builds) is logged, not raised: a notification must never interrupt
    the job that triggered it.

    Parameters
    ----------
    filename : str or None
        File name of the sound. None plays nothing.
    """
    if filename is None:
        return
    path = os.path.join(sounds_directory(), filename)
    try:
        from playsound3 import playsound

        playsound(path, block=False)
    except Exception:  # noqa: BLE001 - notification only
        diagnostics.log_message(
            f"Could not play the sound notification {path}:\n"
            f"{traceback.format_exc()}"
        )


def notify_finished(elapsed: float) -> None:
    """Play the selected sound if a job ran long enough.

    Parameters
    ----------
    elapsed : float
        Duration of the job in seconds. The sound plays only if it
        exceeds :data:`MIN_DURATION`.
    """
    if elapsed > MIN_DURATION and (filename := selected_sound()) is not None:
        play(filename)
