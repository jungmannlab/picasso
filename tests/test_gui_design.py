"""The design canvas of Picasso: Design follows the windows' theme.

The background and the labels of the design are drawn in the colors of
the windows, light or dark (``picasso.gui.theme``), and change with
them.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from PyQt6 import QtWidgets

from picasso.gui import design as gui_design
from picasso.gui import theme
from picasso.gui.theme import NEUTRALS


def _colors(window):
    scene = window.window.mainscene
    background = scene.backgroundBrush().color().name()
    labels = {
        item.defaultTextColor().name()
        for item in scene.items()
        if isinstance(item, QtWidgets.QGraphicsTextItem)
    }
    return background, labels


def test_design_canvas_follows_the_windows_theme(qt_offscreen, restore_theme):
    theme.apply(restore_theme, theme.Appearance(mode="Dark"))
    window = gui_design.MainWindow()
    assert _colors(window) == (
        NEUTRALS["Dark"]["base"],
        {NEUTRALS["Dark"]["text"]},
    )
    # an open window changes with the theme
    theme.apply(restore_theme, theme.Appearance(mode="Light"))
    assert _colors(window) == (
        NEUTRALS["Light"]["base"],
        {NEUTRALS["Light"]["text"]},
    )
    window.close()
