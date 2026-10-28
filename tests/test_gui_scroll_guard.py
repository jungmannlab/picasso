"""The wheel scrolls a dialog instead of changing its inputs
(``picasso.gui.app.install_scroll_guard``).

Qt passes an ignored wheel event on to the parents of the widget, up
to the scroll area; synthesized events do not propagate, so these tests
check that the input keeps its value and leaves the event ignored.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso.gui import app as gui_app

Policy = QtCore.Qt.FocusPolicy


@pytest.fixture
def guard(qt_offscreen):
    """The event filter, removed again so that other tests see the
    default wheel behavior."""
    app = QtWidgets.QApplication.instance()
    gui_app.install_scroll_guard(app)
    yield app.findChild(gui_app._ScrollGuard)
    for watcher in app.findChildren(gui_app._ScrollGuard):
        app.removeEventFilter(watcher)
        watcher.setParent(None)
        watcher.deleteLater()


@pytest.fixture
def area(guard):
    """A scroll area, shown, whose contents do not fit."""
    widget = QtWidgets.QScrollArea()
    widget.setWidgetResizable(True)
    contents = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(contents)
    contents.setMinimumHeight(2000)
    widget.setWidget(contents)
    widget.resize(200, 200)
    yield widget
    widget.close()
    widget.deleteLater()


def _add(area: QtWidgets.QScrollArea, widget: QtWidgets.QWidget):
    """Show ``widget`` in ``area``, without the focus, which the only
    input of a window takes when it is shown."""
    area.widget().layout().addWidget(widget)
    area.show()
    QtWidgets.QApplication.processEvents()
    widget.clearFocus()
    return widget


def _wheel(widget: QtWidgets.QWidget, notches: int = 1) -> QtGui.QWheelEvent:
    """Send a wheel event of ``notches`` up to the center of ``widget``
    and return it, to read whether it was accepted."""
    pos = QtCore.QPointF(widget.rect().center())
    event = QtGui.QWheelEvent(
        pos,
        QtCore.QPointF(widget.mapToGlobal(pos)),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, 120 * notches),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    return event


def _spin_box() -> QtWidgets.QSpinBox:
    spin = QtWidgets.QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(50)
    return spin


def _combo_box() -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    return combo


def _slider() -> QtWidgets.QSlider:
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)
    return slider


def _value(widget: QtWidgets.QWidget):
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentIndex()
    return widget.value()


@pytest.mark.parametrize("make", [_spin_box, _combo_box, _slider])
def test_wheel_passes_an_input_without_focus(area, make):
    widget = _add(area, make())
    before = _value(widget)
    event = _wheel(widget)
    assert _value(widget) == before
    assert not event.isAccepted()  # goes on to the scroll area


@pytest.mark.parametrize("make", [_spin_box, _slider])
def test_wheel_changes_an_input_with_focus(area, make):
    widget = _add(area, make())
    widget.setFocus()
    assert widget.hasFocus()
    before = _value(widget)
    _wheel(widget)
    assert _value(widget) != before


def test_wheel_changes_an_input_outside_a_scroll_area(guard):
    parent = QtWidgets.QWidget()
    spin = _spin_box()
    QtWidgets.QVBoxLayout(parent).addWidget(spin)
    parent.show()
    try:
        _wheel(spin)
        assert spin.value() == 51
    finally:
        parent.close()
        parent.deleteLater()


def test_wheel_still_moves_a_scroll_bar(area):
    bar = area.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    before = bar.value()
    _wheel(bar)
    assert bar.value() != before


def test_inputs_in_a_table_are_guarded(guard):
    table = QtWidgets.QTableWidget(1, 1)
    spin = _spin_box()
    table.setCellWidget(0, 0, spin)
    table.show()
    try:
        QtWidgets.QApplication.processEvents()
        spin.clearFocus()
        assert gui_app._in_scroll_area(spin)
        _wheel(spin)
        assert spin.value() == 50
    finally:
        table.close()
        table.deleteLater()


def test_the_wheel_does_not_give_the_focus(area):
    spin = _add(area, _spin_box())
    assert spin.focusPolicy() == Policy.StrongFocus
    # a policy set on purpose stays
    slider = _slider()
    slider.setFocusPolicy(Policy.NoFocus)
    _add(area, slider)
    assert slider.focusPolicy() == Policy.NoFocus


def test_widgets_built_before_install_lose_wheel_focus(qt_offscreen):
    spin = _spin_box()
    assert spin.focusPolicy() == Policy.WheelFocus
    app = QtWidgets.QApplication.instance()
    gui_app.install_scroll_guard(app)
    try:
        assert spin.focusPolicy() == Policy.StrongFocus
    finally:
        for watcher in app.findChildren(gui_app._ScrollGuard):
            app.removeEventFilter(watcher)
            watcher.setParent(None)
            watcher.deleteLater()
        spin.deleteLater()


def test_install_twice_installs_one_filter(guard):
    app = QtWidgets.QApplication.instance()
    gui_app.install_scroll_guard(app)
    assert len(app.findChildren(gui_app._ScrollGuard)) == 1
