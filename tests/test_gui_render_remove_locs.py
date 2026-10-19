"""Confirmation before Render's File > Remove all localizations.

Render reads ``Confirm removing all localizations`` from the ``Render``
section of the user settings (False, i.e., no question, by default);
when it is on, ticking
"Don't ask again" in the question turns it off.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import render as gui_render

Button = QtWidgets.QMessageBox.StandardButton


def _set_confirm(value) -> None:
    settings = io.load_user_settings()
    settings["Render"][gui_render.CONFIRM_REMOVE_LOCS_KEY] = value
    io.save_user_settings(settings)


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """A Render window with one channel; ``remove_locs`` only records
    that it was called."""
    window = gui_render.Window(plugins_loaded=True)
    rng = np.random.default_rng(0)
    locs = pd.DataFrame(
        {
            "frame": np.arange(100, dtype=np.uint32),
            "x": rng.uniform(0, 32 * 130.0, 100).astype(np.float32),
            "y": rng.uniform(0, 32 * 130.0, 100).astype(np.float32),
            "photons": np.full(100, 1000, dtype=np.float32),
            "sx": np.full(100, 130.0, dtype=np.float32),
            "sy": np.full(100, 130.0, dtype=np.float32),
            "lpx": np.full(100, 13.0, dtype=np.float32),
            "lpy": np.full(100, 13.0, dtype=np.float32),
        }
    )
    info = [{"Frames": 100, "Width": 32, "Height": 32, "Pixelsize": 130.0}]
    window.view.add(str(tmp_path / "a.hdf5"), locs, info, render_=False)
    window.removed = 0

    def remove_locs():
        window.removed += 1

    monkeypatch.setattr(window, "remove_locs", remove_locs)
    return window


@pytest.fixture
def asking(window):
    """``window``, with the confirmation turned on."""
    _set_confirm(True)
    return window


def _answer(monkeypatch, button, dont_ask=False) -> list:
    """Answer the question with ``button``; returns the list of the
    message boxes shown."""
    shown = []

    def exec_(box):
        shown.append(box)
        box.checkBox().setChecked(dont_ask)
        return button

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)
    return shown


def test_setting_is_written_with_the_default():
    assert gui_render.confirm_remove_locs_setting() is False
    render_settings = io.load_user_settings()["Render"]
    assert render_settings[gui_render.CONFIRM_REMOVE_LOCS_KEY] is False


def test_render_writes_the_setting_on_start(qt_offscreen):
    gui_render.Window(plugins_loaded=True)
    render_settings = io.load_user_settings()["Render"]
    assert render_settings[gui_render.CONFIRM_REMOVE_LOCS_KEY] is False


def test_invalid_value_falls_back_to_not_asking():
    _set_confirm("yes")
    assert gui_render.confirm_remove_locs_setting() is False


def test_no_question_by_default(window, monkeypatch):
    shown = _answer(monkeypatch, Button.Cancel)
    window.ask_remove_locs()
    assert shown == []
    assert window.removed == 1


def test_cancel_keeps_the_localizations(asking, monkeypatch):
    window = asking
    shown = _answer(monkeypatch, Button.Cancel)
    window.ask_remove_locs()
    assert len(shown) == 1
    assert window.removed == 0


def test_yes_removes_the_localizations(asking, monkeypatch):
    window = asking
    _answer(monkeypatch, Button.Yes)
    window.ask_remove_locs()
    assert window.removed == 1
    render_settings = io.load_user_settings()["Render"]
    assert render_settings[gui_render.CONFIRM_REMOVE_LOCS_KEY] is True


def test_dont_ask_again_turns_the_setting_off(asking, monkeypatch):
    window = asking
    _answer(monkeypatch, Button.Yes, dont_ask=True)
    window.ask_remove_locs()
    assert window.removed == 1
    assert gui_render.confirm_remove_locs_setting() is False
    shown = _answer(monkeypatch, Button.Cancel)
    window.ask_remove_locs()
    assert shown == []
    assert window.removed == 2


def test_dont_ask_again_ignored_on_cancel(asking, monkeypatch):
    window = asking
    _answer(monkeypatch, Button.Cancel, dont_ask=True)
    window.ask_remove_locs()
    assert window.removed == 0
    assert gui_render.confirm_remove_locs_setting() is True


def test_no_question_without_localizations(qt_offscreen, monkeypatch):
    window = gui_render.Window(plugins_loaded=True)
    _set_confirm(True)
    calls = []
    monkeypatch.setattr(window, "remove_locs", lambda: calls.append(1))
    shown = _answer(monkeypatch, Button.Cancel)
    window.ask_remove_locs()
    assert shown == []
    assert calls == [1]
