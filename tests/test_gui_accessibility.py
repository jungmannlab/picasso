"""Names of input widgets and icon-only buttons for screen readers
(``picasso.gui.accessibility``).

Qt names an input after the label whose buddy it is; these tests cover
finding that label in the layouts Picasso uses, the fallbacks (table
column, tooltip) and the event filter that names widgets when shown,
also in a real Render window.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import accessibility
from picasso.gui import render as gui_render

from tests.test_gui_rotation_async import HEIGHT, WIDTH, _info, _locs


def _name(widget: QtWidgets.QWidget) -> str:
    """The name Qt gives ``widget`` (accessible name, else the text of
    its buddy label)."""
    if widget.accessibleName():
        return widget.accessibleName()
    for child in widget.parentWidget().children():
        if isinstance(child, QtWidgets.QLabel) and child.buddy() is widget:
            return child.text()
    return ""


@pytest.fixture
def parent(qt_offscreen):
    widget = QtWidgets.QWidget()
    yield widget
    widget.deleteLater()


def test_grid_label_to_the_left_becomes_the_buddy(parent):
    grid = QtWidgets.QGridLayout(parent)
    label = QtWidgets.QLabel("Box size:")
    spin = QtWidgets.QSpinBox()
    grid.addWidget(label, 0, 0)
    grid.addWidget(spin, 0, 1)
    accessibility.name_widget(spin)
    assert label.buddy() is spin
    assert _name(spin) == "Box size:"
    assert spin.accessibleName() == ""


def test_grid_column_header_but_not_a_spanning_title(parent):
    grid = QtWidgets.QGridLayout(parent)
    grid.addWidget(QtWidgets.QLabel("Advanced"), 0, 0, 1, 3)
    grid.addWidget(QtWidgets.QLabel("Min"), 1, 0)
    low = QtWidgets.QSpinBox()
    grid.addWidget(low, 2, 0)
    title_only = QtWidgets.QSpinBox()
    grid.addWidget(title_only, 1, 2)
    accessibility.name_widget(low)
    accessibility.name_widget(title_only)
    assert _name(low) == "Min"
    assert _name(title_only) == ""


def test_box_label_before_the_input_skipping_stretches(parent):
    box = QtWidgets.QHBoxLayout(parent)
    box.addWidget(QtWidgets.QLabel("Width:"))
    box.addStretch()
    spin = QtWidgets.QDoubleSpinBox()
    box.addWidget(spin)
    box.addWidget(QtWidgets.QLabel("nm"))
    accessibility.name_widget(spin)
    assert _name(spin) == "Width:"


def test_label_next_to_a_nested_layout(parent):
    grid = QtWidgets.QGridLayout(parent)
    grid.addWidget(QtWidgets.QLabel("Range:"), 0, 0)
    row = QtWidgets.QHBoxLayout()
    low = QtWidgets.QSpinBox()
    high = QtWidgets.QSpinBox()
    row.addWidget(low)
    row.addWidget(QtWidgets.QLabel("-"))  # no words, not a name
    row.addWidget(high)
    grid.addLayout(row, 0, 1)
    for spin in (low, high):
        accessibility.name_widget(spin)
    # the label is the buddy of one input, the other one takes its text
    assert _name(low) == "Range:"
    assert high.accessibleName() == "Range"


def test_name_instead_of_buddy_where_a_buddy_would_change_the_label(
    parent,
):
    grid = QtWidgets.QGridLayout(parent)
    ampersand = QtWidgets.QLabel("Photons & background:")
    rich = QtWidgets.QLabel("<b>Threshold</b>:")
    taken = QtWidgets.QLabel("Size:")
    spins = [QtWidgets.QSpinBox() for _ in range(4)]
    taken.setBuddy(spins[3])
    for row, label in enumerate((ampersand, rich, taken)):
        grid.addWidget(label, row, 0)
        grid.addWidget(spins[row], row, 1)
    grid.addWidget(spins[3], 3, 1)
    for spin in spins[:3]:
        accessibility.name_widget(spin)
    assert ampersand.buddy() is None and rich.buddy() is None
    assert ampersand.text() == "Photons & background:"
    assert spins[0].accessibleName() == "Photons & background"
    assert spins[1].accessibleName() == "Threshold"
    assert taken.buddy() is spins[3]
    assert spins[2].accessibleName() == "Size"


def test_existing_names_and_form_layouts_are_kept(parent):
    form = QtWidgets.QFormLayout(parent)
    in_form = QtWidgets.QLineEdit()
    form.addRow("Path:", in_form)
    named = QtWidgets.QSpinBox()
    named.setAccessibleName("Own name")
    form.addRow("Other:", named)
    for widget in (in_form, named):
        accessibility.name_widget(widget)
    assert _name(in_form) == "Path:"
    assert named.accessibleName() == "Own name"


def test_icon_only_buttons_and_unlabeled_inputs_use_the_tooltip(parent):
    box = QtWidgets.QHBoxLayout(parent)
    icon_button = QtWidgets.QPushButton()
    icon_button.setToolTip("Close this dataset.\nThe file is kept.")
    text_button = QtWidgets.QPushButton("Load")
    text_button.setToolTip("Load a file.")
    combo = QtWidgets.QComboBox()
    combo.setToolTip("Column to filter by.")
    for widget in (icon_button, text_button, combo):
        box.addWidget(widget)
        accessibility.name_widget(widget)
    assert icon_button.accessibleName() == "Close this dataset"
    assert text_button.accessibleName() == ""  # Qt reads its text
    assert combo.accessibleName() == "Column to filter by"


def test_inputs_in_table_cells_are_named_after_the_column(qt_offscreen):
    table = QtWidgets.QTableWidget(1, 2)
    table.setHorizontalHeaderLabels(["File", "Color"])
    cell = QtWidgets.QWidget()
    QtWidgets.QHBoxLayout(cell).addWidget(combo := QtWidgets.QComboBox())
    table.setCellWidget(0, 1, cell)
    table.resize(400, 200)
    table.show()
    accessibility.name_widget(combo)
    assert combo.accessibleName() == "Color"
    table.deleteLater()


def test_layout_shadowed_by_an_attribute(parent):
    parent.layout = QtWidgets.QGridLayout()  # as some Picasso dialogs do
    QtWidgets.QWidget.setLayout(parent, parent.layout)
    label = QtWidgets.QLabel("Min:")
    spin = QtWidgets.QDoubleSpinBox()
    parent.layout.addWidget(label, 0, 0)
    parent.layout.addWidget(spin, 0, 1)
    accessibility.name_widget(spin)
    assert label.buddy() is spin


def test_install_names_widgets_when_shown(qt_offscreen, parent):
    accessibility.install(qt_offscreen)
    accessibility.install(qt_offscreen)
    filters = qt_offscreen.findChildren(accessibility._NameOnShow)
    assert len(filters) == 1
    grid = QtWidgets.QGridLayout(parent)
    label = QtWidgets.QLabel("Box size:")
    spin = QtWidgets.QSpinBox()
    grid.addWidget(label, 0, 0)
    grid.addWidget(spin, 0, 1)
    assert label.buddy() is None
    parent.show()
    qt_offscreen.processEvents()
    assert label.buddy() is spin


def test_render_window_inputs_have_names(qt_offscreen, tmp_path, monkeypatch):
    monkeypatch.setattr(
        io, "_user_settings_filename", lambda: str(tmp_path / "s.yaml")
    )
    accessibility.install(qt_offscreen)
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        window.view.add(
            str(tmp_path / f"locs_{i}.hdf5"), _locs(), _info(), render_=False
        )
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    try:
        dialogs = [window.display_settings_dlg, window.dataset_dialog]
        for dialog in dialogs:
            dialog.show()
        qt_offscreen.processEvents()
        assert _name(window.display_settings_dlg.disp_px_size).startswith(
            "Display pixel size"
        )
        dataset = window.dataset_dialog
        assert dataset.colorselection[1].accessibleName() == "Color"
        assert (
            dataset.title[0]
            .accessibleName()
            .startswith("Change the displayed name")
        )
        # every visible input is named
        for dialog in dialogs:
            for widget in dialog.findChildren(QtWidgets.QWidget):
                if (
                    isinstance(widget, accessibility.INPUT_TYPES)
                    and widget.isVisible()
                    and not accessibility._is_part_of_input(widget)
                ):
                    assert _name(widget), (dialog.windowTitle(), widget)
    finally:
        window.window_rot.view_rot.stop_render_worker()
        window.view.stop_render_worker()
