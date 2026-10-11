"""Sound notifications (``picasso.notifications``) and the shared
``File > Sound notifications`` submenu (``lib.add_sound_notifications_menu``).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest

from picasso import diagnostics, io, notifications


@pytest.fixture
def sounds_dir(tmp_path, monkeypatch):
    """An isolated sounds folder with two supported files and one not."""
    directory = tmp_path / "notification_sounds"
    directory.mkdir()
    for name in ("Bell.wav", "Ding_dong.mp3", "notes.txt"):
        (directory / name).write_bytes(b"")
    monkeypatch.setattr(
        io, "notification_sounds_directory", lambda: str(directory)
    )
    return directory


@pytest.fixture
def played(monkeypatch):
    """Record the files ``notifications.play`` would play."""
    calls = []
    monkeypatch.setattr(notifications, "play", calls.append)
    return calls


def test_available_sounds_lists_supported_files(sounds_dir):
    assert notifications.available_sounds() == ["Bell.wav", "Ding_dong.mp3"]


def test_selected_sound_defaults_to_none(sounds_dir):
    assert notifications.selected_sound() is None


def test_select_sound_roundtrip(sounds_dir):
    notifications.select_sound("Bell.wav")
    assert notifications.selected_sound() == "Bell.wav"
    notifications.select_sound(None)
    assert notifications.selected_sound() is None


@pytest.mark.parametrize("stored", ["None", "gone.wav", "notes.txt", 3])
def test_selected_sound_ignores_unavailable(sounds_dir, stored):
    # "None" is what earlier versions stored for no sound
    settings = io.load_user_settings()
    settings["Sound_notification"]["filename"] = stored
    io.save_user_settings(settings)
    assert notifications.selected_sound() is None


def test_notify_finished_respects_min_duration(sounds_dir, played):
    notifications.select_sound("Bell.wav")
    notifications.notify_finished(notifications.MIN_DURATION - 1)
    notifications.notify_finished(notifications.MIN_DURATION + 1)
    assert played == ["Bell.wav"]


def test_notify_finished_without_selection_plays_nothing(sounds_dir, played):
    notifications.notify_finished(notifications.MIN_DURATION + 1)
    assert played == []


def test_play_logs_a_broken_backend(sounds_dir, monkeypatch):
    import playsound3

    def broken(*args, **kwargs):
        raise RuntimeError("no audio backend")

    monkeypatch.setattr(playsound3, "playsound", broken)
    messages = []
    monkeypatch.setattr(diagnostics, "log_message", messages.append)
    notifications.play("Bell.wav")  # must not raise
    assert len(messages) == 1 and "no audio backend" in messages[0]


def test_play_none_plays_nothing(monkeypatch):
    import playsound3

    monkeypatch.setattr(
        playsound3, "playsound", lambda *a, **k: pytest.fail("played")
    )
    notifications.play(None)


def _sound_actions(menu):
    return [a for a in menu.actions() if a.isCheckable()]


def test_menu_lists_and_selects_sounds(qapp, sounds_dir, played):
    from PyQt6 import QtWidgets

    from picasso import lib

    file_menu = QtWidgets.QMenu("File")
    menu = lib.add_sound_notifications_menu(file_menu)
    actions = _sound_actions(menu)
    assert [a.text() for a in actions] == ["None", "Bell", "Ding dong"]
    assert actions[0].isChecked()

    actions[1].trigger()
    assert notifications.selected_sound() == "Bell.wav"
    assert played == ["Bell.wav"]  # preview
    assert actions[1].isChecked() and not actions[0].isChecked()


def test_menu_updates_in_place_on_show(qapp, sounds_dir, played):
    from PyQt6 import QtWidgets

    from picasso import lib

    file_menu = QtWidgets.QMenu("File")  # keep the parent alive
    menu = lib.add_sound_notifications_menu(file_menu)
    before = _sound_actions(menu)
    # another window changes the choice; a sound is added and one removed
    notifications.select_sound("Ding_dong.mp3")
    (sounds_dir / "Chime.wav").write_bytes(b"")
    (sounds_dir / "Bell.wav").unlink()
    menu.aboutToShow.emit()

    after = _sound_actions(menu)
    assert after[:3] == before  # kept, e.g. for toolbars holding them
    assert [a.text() for a in after] == ["None", "Bell", "Ding dong", "Chime"]
    bell, ding = after[1], after[2]
    assert not bell.isEnabled()
    assert ding.isChecked()
    # the open-folder action stays last, after the separator
    assert menu.actions()[-2].isSeparator()
