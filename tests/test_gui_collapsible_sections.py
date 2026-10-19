"""Collapsible, remembered sections of Picasso's long dialogs.

The sections of Render's Display Settings (2D and 3D), Info and Tools
Settings and of Localize's Parameters collapse by clicking their
header; which ones are expanded is kept in the user settings, so that
users keep the layout they set up. These tests cover the checkable
sections (scale bar, grid), the round trip through the settings file
and the resizing of a dialog when a section opens or closes.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
import yaml
from PyQt6 import QtCore, QtWidgets

from picasso import io, lib
from picasso.gui import localize as gui_localize
from picasso.gui import render as gui_render

KEY = "Expanded sections"


def _settle() -> None:
    """Let the layouts settle and the deferred resizes run."""
    app = QtWidgets.QApplication.instance()
    for _ in range(5):
        app.processEvents()
        QtCore.QThread.msleep(10)


def _sections(dialog: QtWidgets.QWidget) -> dict:
    return {
        box.title(): box
        for box in dialog.findChildren(lib.CollapsibleGroupBox)
    }


def _dialog(name: str, parent: QtWidgets.QWidget | None = None):
    """A dialog with three sections in a scroll area, set up like
    Picasso's long dialogs."""
    dialog = QtWidgets.QDialog(parent)
    scroll_area = QtWidgets.QScrollArea()
    scroll_area.setWidgetResizable(True)
    container = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout(container)
    scroll_area.setWidget(container)
    QtWidgets.QVBoxLayout(dialog).addWidget(scroll_area)
    for title in ("First", "Second", "Third"):
        box = lib.CollapsibleGroupBox(title)
        grid = QtWidgets.QGridLayout(box.content)
        for row in range(4):
            grid.addWidget(QtWidgets.QLabel(f"{title} {row}"), row, 0)
        vbox.addWidget(box)
    vbox.addStretch(1)
    lib.remember_expanded_sections(dialog, "Test", name)
    lib.resize_with_sections(dialog, scroll_area)
    dialog.scroll_area = scroll_area
    return dialog


def _saved() -> dict:
    with open(io._user_settings_filename()) as file:
        return yaml.safe_load(file)


def test_checkable_section(qt_offscreen):
    box = lib.CollapsibleGroupBox("Scale bar")
    QtWidgets.QVBoxLayout(box.content).addWidget(QtWidgets.QLabel("x"))
    # like a QGroupBox, a box that is not checkable is not checked
    assert not box.isCheckable() and not box.isChecked()
    box.setChecked(True)
    assert not box.isChecked()
    assert box.title() == "Scale bar"

    width = box.toggle_button.minimumSizeHint().width()
    box.setCheckable(True)
    assert box.isCheckable() and box.isChecked()
    # the checkbox sits between the chevron and the title
    assert box.toggle_button.minimumSizeHint().width() > width
    states = []
    box.toggled.connect(states.append)
    box.setChecked(False)
    assert states == [False] and not box.isChecked()
    # the contents are grayed out, the header still collapses them
    assert not box.content.isEnabled()
    assert box.toggle_button.isEnabled()
    box.setChecked(False)
    assert states == [False]

    # clicking the checkbox checks the box without collapsing it
    box.show()
    box.toggle_button.checkbox.click()
    assert box.isChecked() and box.content.isEnabled()
    assert box.isExpanded()
    assert states == [False, True]
    box.toggle_button.click()
    assert not box.isExpanded() and box.isChecked()

    box.setTitle("Scale")
    assert box.title() == "Scale"
    box.close()


def test_collapsed_section_keeps_its_width(qt_offscreen):
    box = lib.CollapsibleGroupBox("A")
    QtWidgets.QVBoxLayout(box.content).addWidget(
        QtWidgets.QLabel("a rather long label that sets the width")
    )
    hint = box.sizeHint()
    box.setExpanded(False)
    assert box.sizeHint().width() == hint.width()
    assert box.sizeHint().height() < hint.height()


def test_states_are_remembered(qt_offscreen):
    dialog = _dialog("Dialog")
    sections = _sections(dialog)
    # the defaults are written, so the key is visible in the file
    assert _saved()["Test"][KEY]["Dialog"] == {
        "First": True,
        "Second": True,
        "Third": True,
    }
    sections["Second"].toggle_button.click()
    assert _saved()["Test"][KEY]["Dialog"]["Second"] is False

    # a new dialog starts as the user left it, other dialogs do not
    sections = _sections(_dialog("Dialog"))
    assert not sections["Second"].isExpanded()
    assert sections["First"].isExpanded()
    assert sections["Second"].toggle_button.progress() == 0
    assert all(
        box.isExpanded() for box in _sections(_dialog("Other")).values()
    )


def test_invalid_states_are_ignored(qt_offscreen):
    io.save_user_settings(
        {"Test": {KEY: {"Dialog": {"First": "no", "Second": False}}}}
    )
    sections = _sections(_dialog("Dialog"))
    assert sections["First"].isExpanded()
    assert not sections["Second"].isExpanded()
    # the invalid entry is replaced by the default
    assert _saved()["Test"][KEY]["Dialog"]["First"] is True

    io.save_user_settings({"Test": {KEY: "not a mapping"}})
    sections = _sections(_dialog("Dialog"))
    assert all(box.isExpanded() for box in sections.values())


def test_broken_settings_file_is_not_overwritten(qt_offscreen):
    path = io._user_settings_filename()
    with open(path, "w") as file:
        file.write("Render: [unclosed\n")
    sections = _sections(_dialog("Dialog"))
    sections["First"].toggle_button.click()
    assert not sections["First"].isExpanded()
    with open(path) as file:
        assert file.read() == "Render: [unclosed\n"


def test_dialog_follows_its_sections(qt_offscreen):
    dialog = _dialog("Dialog")
    sections = _sections(dialog)
    dialog.show()
    _settle()
    content = dialog.scroll_area.widget()
    dialog.resize(dialog.width(), content.sizeHint().height() + 40)
    _settle()
    height = dialog.height()

    # a collapsed section leaves no empty space behind
    sections["Second"].toggle_button.click()
    _settle()
    assert dialog.height() < height
    assert dialog.scroll_area.verticalScrollBar().maximum() == 0
    # and is shown in full when expanded again
    sections["Second"].toggle_button.click()
    _settle()
    assert dialog.height() == height

    # a dialog the user made too short keeps its height
    dialog.resize(dialog.width(), height // 2)
    _settle()
    sections["Third"].toggle_button.click()
    _settle()
    assert dialog.height() == height // 2
    dialog.close()


@pytest.fixture
def render_window(qt_offscreen):
    window = gui_render.Window(plugins_loaded=True)
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()
    window.close()


def test_render_dialogs_have_remembered_sections(render_window):
    window = render_window
    dialogs = {
        "Display Settings": window.display_settings_dlg,
        "3D Display Settings": window.window_rot.display_settings_dlg,
        "Info": window.info_dialog,
        "Tools Settings": window.tools_settings_dialog,
    }
    saved = _saved()["Render"][KEY]
    for name, dialog in dialogs.items():
        sections = _sections(dialog)
        assert len(sections) >= 3, name
        assert saved[name] == {
            title: box.isExpanded() for title, box in sections.items()
        }
    # the sections that were collapsed before stay so by default
    assert saved["Info"]["FRC in several ROIs"] is False
    assert saved["Tools Settings"]["Appearance"] is False
    assert saved["Display Settings"]["Contrast"] is True

    _sections(window.display_settings_dlg)["Contrast"].toggle_button.click()
    other = gui_render.Window(plugins_loaded=True)
    try:
        sections = _sections(other.display_settings_dlg)
        assert not sections["Contrast"].isExpanded()
    finally:
        other.window_rot.view_rot.stop_render_worker()
        other.view.stop_render_worker()
        other.close()


def test_scale_bar_and_grid_sections_are_checkable(render_window):
    for dialog in (
        render_window.display_settings_dlg,
        render_window.window_rot.display_settings_dlg,
    ):
        for box in (dialog.scalebar_groupbox, dialog.grid_box):
            assert box.isCheckable() and not box.isChecked()
            assert not box.content.isEnabled()
            states = []
            box.toggled.connect(states.append)
            box.toggle_button.checkbox.click()
            assert box.isChecked() and box.content.isEnabled()
            assert states == [True]
            box.setChecked(False)


def test_render_properties_collapse_while_disabled(render_window):
    box = render_window.display_settings_dlg.render_groupbox
    # unavailable without a single channel, but it can still be collapsed
    assert not box.content.isEnabled()
    assert box.toggle_button.isEnabled()


def test_localize_parameters_sections_are_remembered(qt_offscreen):
    window = gui_localize.Window()
    try:
        sections = _sections(window.parameters_dialog)
        assert {
            "Identification",
            "Photon Conversion",
            "Fit Settings",
            "Postprocessing and Sample Quality",
        } <= set(sections)
        sections["Photon Conversion"].toggle_button.click()
        saved = _saved()["Localize"][KEY]["Parameters"]
        assert saved["Photon Conversion"] is False
        assert saved["Identification"] is True
    finally:
        window.close()
    window = gui_localize.Window()
    try:
        sections = _sections(window.parameters_dialog)
        assert not sections["Photon Conversion"].isExpanded()
    finally:
        window.close()
