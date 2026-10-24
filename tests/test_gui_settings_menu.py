"""Test that every Picasso window opens the user settings from its File
menu, as the documentation says (``File > Picasso settings`` in any
module).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib

import pytest
from PyQt6 import QtWidgets

from picasso import lib

#: the window class each module's ``main`` passes to ``run_gui``
WINDOWS = {
    "average": "Window",
    "design": "MainWindow",
    "filter": "Window",
    "localize": "Window",
    "nanotron": "Window",
    "render": "Window",
    "simulate": "Window",
    "spinna": "Window",
}


@pytest.mark.parametrize("module", sorted(WINDOWS))
def test_file_menu_opens_the_user_settings(qt_offscreen, module):
    gui = importlib.import_module(f"picasso.gui.{module}")
    window = getattr(gui, WINDOWS[module])()
    try:
        menu_bar = window.findChild(QtWidgets.QMenuBar)
        file_menu = menu_bar.actions()[0].menu()
        assert file_menu.title() == "File"
        actions = {a.text(): a for a in file_menu.actions()}
        assert "Picasso settings..." in actions
        actions["Picasso settings..."].trigger()
        shown = [
            widget
            for widget in QtWidgets.QApplication.topLevelWidgets()
            if isinstance(widget, lib.UserSettingsDialog)
            and widget.isVisible()
        ]
        assert len(shown) == 1
        shown[0].close()
    finally:
        if module == "render":
            window.view.stop_render_worker()
        window.close()
