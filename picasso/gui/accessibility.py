"""
picasso.gui.accessibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Names for screen readers (VoiceOver, Narrator, Orca) of the input
widgets and icon-only buttons of all Picasso windows.

A screen reader announces an input widget, e.g., a spin box, by its
accessible name, which Qt takes from a ``QLabel`` whose buddy the input
is. Picasso's dialogs place each label next to its input in a layout
without making it the buddy, so most inputs were announced without a
name. ``install`` names them when they are first shown, from their
place in the layout (see ``find_label``): the label is made the buddy
of the input (so that the name follows changes of the label's text) or,
where that is not possible, its text becomes the accessible name of the
input. Inputs in the cells of a table are named after the column, and
inputs without a label as well as icon-only buttons after the first
line of their tooltip. Widgets that have an accessible name or a buddy
label are left as they are.

The application-wide event filter reacts only to show events, so it
costs well below a microsecond per event.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import re

from PyQt6 import QtCore, QtWidgets

#: Widgets that take input and are named after their label.
INPUT_TYPES = (
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QLineEdit,
    QtWidgets.QAbstractSlider,
    QtWidgets.QPlainTextEdit,
    QtWidgets.QTextEdit,
)
#: Dynamic property set on the widgets that were looked at, so that each
#: widget is named once.
_DONE_PROPERTY = "picassoAccessibleNamed"
_TAGS = re.compile(r"<[^>]*>")


def _is_part_of_input(widget: QtWidgets.QWidget) -> bool:
    """Whether ``widget`` is a part of another widget, e.g., the line
    edit of a spin box or a scroll bar, which Qt names itself."""
    if isinstance(widget, QtWidgets.QScrollBar):
        return True
    parent = widget.parentWidget()
    return isinstance(
        parent, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)
    )


def _has_buddy_label(widget: QtWidgets.QWidget) -> bool:
    """Whether a label names ``widget`` already (as Qt looks it up:
    among the children of the widget's parent)."""
    parent = widget.parentWidget()
    if parent is None:
        return False
    return any(
        isinstance(child, QtWidgets.QLabel) and child.buddy() is widget
        for child in parent.children()
    )


def label_text(label: QtWidgets.QLabel) -> str:
    """The text of ``label`` as a name: without markup and a trailing
    colon. (Without a buddy, a label shows ampersands as they are.)

    Parameters
    ----------
    label : QtWidgets.QLabel
        The label.

    Returns
    -------
    text : str
        The cleaned text, empty if the label shows no words.
    """
    text = label.text()
    if QtCore.Qt.mightBeRichText(text):
        text = _TAGS.sub(" ", text)
    text = " ".join(text.split()).rstrip(":").strip()
    if not any(c.isalpha() for c in text):
        return ""
    return text


def tooltip_name(widget: QtWidgets.QWidget) -> str:
    """The first line of the tooltip of ``widget`` as a name, without
    markup and a closing period or question mark.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        The widget.

    Returns
    -------
    name : str
        The name, empty without a tooltip.
    """
    tooltip = widget.toolTip()
    if QtCore.Qt.mightBeRichText(tooltip):
        tooltip = _TAGS.sub(" ", tooltip)
    lines = tooltip.strip().split("\n")
    return " ".join(lines[0].split()).rstrip(".?").strip()


def _label_of(item: QtWidgets.QLayoutItem | None) -> QtWidgets.QLabel | None:
    """The label in a layout item, if it holds a label showing words."""
    if item is None:
        return None
    widget = item.widget()
    if isinstance(widget, QtWidgets.QLabel) and label_text(widget):
        return widget
    return None


def _locate(
    layout: QtWidgets.QLayout, widget: QtWidgets.QWidget
) -> list[tuple[QtWidgets.QLayout, int]] | None:
    """The path from ``layout`` to ``widget`` through nested layouts:
    the layouts and the index of the item leading to ``widget`` in each,
    outermost first; None if ``widget`` is not in ``layout``."""
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is widget:
            return [(layout, index)]
        sub = item.layout()
        if sub is not None:
            path = _locate(sub, widget)
            if path is not None:
                return [(layout, index), *path]
    return None


def _label_next_to(
    layout: QtWidgets.QLayout, index: int
) -> QtWidgets.QLabel | None:
    """The label describing item ``index`` of ``layout``: the label of
    its row in a form layout, the label to its left in a grid layout
    (else a label above it that spans one column, i.e., a column
    header) and the label before it in a box layout."""
    if isinstance(layout, QtWidgets.QFormLayout):
        row, role = layout.getItemPosition(index)
        if role == QtWidgets.QFormLayout.ItemRole.FieldRole:
            return _label_of(
                layout.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            )
        return None
    if isinstance(layout, QtWidgets.QGridLayout):
        row, column, _, _ = layout.getItemPosition(index)
        if column > 0:
            label = _label_of(layout.itemAtPosition(row, column - 1))
            if label is not None:
                return label
        if row > 0:
            above = layout.itemAtPosition(row - 1, column)
            label = _label_of(above)
            if label is not None:
                _, _, _, span = layout.getItemPosition(layout.indexOf(above))
                if span == 1:
                    return label
        return None
    if isinstance(layout, QtWidgets.QBoxLayout):
        # skip spacers and stretches between the label and the input
        for previous in range(index - 1, -1, -1):
            item = layout.itemAt(previous)
            if item.spacerItem() is not None:
                continue
            return _label_of(item)
    return None


def find_label(widget: QtWidgets.QWidget) -> QtWidgets.QLabel | None:
    """The label that describes ``widget`` by its place in the layout
    of its parent, see ``_label_next_to``. If its own (innermost) layout
    has none, the label next to that layout in the enclosing one is
    taken, e.g., for a spin box and its unit in a box layout in the cell
    of a grid layout.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        The input widget.

    Returns
    -------
    label : QtWidgets.QLabel or None
        The label, or None if there is none.
    """
    parent = widget.parentWidget()
    # some widgets shadow the layout method with an attribute
    layout = QtWidgets.QWidget.layout(parent) if parent is not None else None
    if layout is None:
        return None
    path = _locate(layout, widget)
    if path is None:
        return None
    for layout, index in reversed(path):
        label = _label_next_to(layout, index)
        if label is not None and label is not widget:
            return label
    return None


def _table_column(widget: QtWidgets.QWidget) -> str:
    """The header of the table column holding ``widget`` in a cell
    (also nested in a cell widget); empty if it is in no table."""
    child = widget
    parent = widget.parentWidget()
    while parent is not None:
        table = parent.parentWidget()
        if (
            isinstance(table, QtWidgets.QTableWidget)
            and parent is table.viewport()
        ):
            column = table.columnAt(child.geometry().center().x())
            header = table.horizontalHeaderItem(column)
            return header.text() if header is not None else ""
        child = parent
        parent = parent.parentWidget()
    return ""


def name_widget(widget: QtWidgets.QWidget) -> None:
    """Give ``widget`` a name for screen readers if it has none: an
    input widget from its label (see ``find_label``) or its table
    column, an icon-only button from its tooltip.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        Any widget; only unnamed inputs and icon-only buttons are
        changed.
    """
    if widget.accessibleName():
        return
    if isinstance(widget, QtWidgets.QAbstractButton):
        if not widget.text() and tooltip_name(widget):
            widget.setAccessibleName(tooltip_name(widget))
        return
    if not isinstance(widget, INPUT_TYPES) or _is_part_of_input(widget):
        return
    if _has_buddy_label(widget):
        return
    label = find_label(widget)
    if label is not None:
        text = label.text()
        if (
            label.buddy() is None
            and label.parentWidget() is widget.parentWidget()
            and "&" not in text
            and not QtCore.Qt.mightBeRichText(text)
        ):
            label.setBuddy(widget)
        else:
            widget.setAccessibleName(label_text(label))
        return
    name = _table_column(widget) or tooltip_name(widget)
    if name:
        widget.setAccessibleName(name)


def name_widgets(root: QtWidgets.QWidget) -> None:
    """Name ``root`` and all its descendants, see ``name_widget``.

    Parameters
    ----------
    root : QtWidgets.QWidget
        E.g., a dialog.
    """
    for widget in [root, *root.findChildren(QtWidgets.QWidget)]:
        name_widget(widget)


class _NameOnShow(QtCore.QObject):
    """Application event filter that names each widget when it is
    shown for the first time."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Show and isinstance(
            obj, QtWidgets.QWidget
        ):
            if not obj.property(_DONE_PROPERTY):
                obj.setProperty(_DONE_PROPERTY, True)
                try:
                    name_widget(obj)
                except Exception:  # never let naming break a window
                    pass
        return False


def install(app: QtWidgets.QApplication) -> None:
    """Name the widgets of all windows of ``app`` for screen readers
    when they are first shown (see the module docstring). Installing
    twice has no further effect.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application.
    """
    if app.findChild(_NameOnShow) is None:
        app.installEventFilter(_NameOnShow(app))
