"""
picasso.gui.overlay_style
~~~~~~~~~~~~~~~~~~~~~~~~~

Widgets that set how the tool overlays of Picasso: Render (picks,
measured points, the Move tool's label) are drawn: color, line style,
line width, opacity, fill and label size. The chosen appearance is
turned into a ``picasso.render.OverlayStyle`` for the drawing functions
and saved in the user settings.

:author: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6 import QtCore, QtGui, QtWidgets

from .. import render

#: Color that follows the background: set by the caller, e.g., yellow
#: on a black and red on a white background.
AUTO_COLOR = "Auto"
#: Last item of a color box, opens a color picker.
CUSTOM_COLOR = "Custom..."
PRESET_COLORS = {
    "Yellow": "#FFFF00",
    "Red": "#FF0000",
    "Orange": "#FFA500",
    "Green": "#008000",
    "Lime": "#00FF00",
    "Cyan": "#00FFFF",
    "Blue": "#0000FF",
    "Magenta": "#FF00FF",
    "White": "#FFFFFF",
    "Black": "#000000",
}
# special values of the spin boxes, shown as "Default"
DEFAULT_FILL = -1
DEFAULT_FONT_SIZE = 0

#: Fields of ``OverlayStyleWidget``: attribute name -> (label, key in
#: the user settings).
FIELDS = {
    "color": ("Color:", "Color"),
    "line_style": ("Line:", "Line style"),
    "line_width": ("Width:", "Line width (px)"),
    "opacity": ("Opacity:", "Opacity (%)"),
    "fill_opacity": ("Fill:", "Fill opacity (%)"),
    "font_size": ("Label size:", "Label size (px)"),
    "marker_size": ("Marker size:", "Marker size (px)"),
    "drawing_color": ("While drawing:", "Color while drawing"),
}
#: Values of the fields when not given otherwise.
DEFAULTS = {
    "color": AUTO_COLOR,
    "line_style": "Solid",
    "line_width": 1.0,
    "opacity": 100,
    "fill_opacity": None,
    "font_size": None,
    "marker_size": 20,
    "drawing_color": "Green",
}


def _swatch(colors: list[QtGui.QColor], size: int = 12) -> QtGui.QIcon:
    """Icon filled with one color, or split diagonally into two."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(colors[0])
    painter = QtGui.QPainter(pixmap)
    if len(colors) > 1:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(colors[1])
        painter.drawPolygon(
            QtGui.QPolygon(
                [
                    QtCore.QPoint(size, 0),
                    QtCore.QPoint(size, size),
                    QtCore.QPoint(0, size),
                ]
            )
        )
    painter.setPen(QtGui.QColor("gray"))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawRect(0, 0, size - 1, size - 1)
    painter.end()
    return QtGui.QIcon(pixmap)


class ColorComboBox(QtWidgets.QComboBox):
    """Choose a preset color, a custom one from a color picker or,
    optionally, the automatic color that follows the background.

    Parameters
    ----------
    auto_colors : tuple of str or None, optional
        The automatic colors on a dark and a light background, shown in
        the icon of the "Auto" item. None offers no "Auto" item.
        Default None.
    parent : QWidget or None, optional
        Parent widget. Default None.
    """

    colorChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        auto_colors: tuple[str, str] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if auto_colors is not None:
            self.addItem(
                _swatch([QtGui.QColor(c) for c in auto_colors]), AUTO_COLOR
            )
            self.setItemData(
                0,
                f"{auto_colors[0]} on dark, {auto_colors[1]} on white "
                "background",
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        for name, code in PRESET_COLORS.items():
            self.addItem(_swatch([QtGui.QColor(code)]), name)
        self.addItem(CUSTOM_COLOR)
        self._last_index = 0
        self.currentIndexChanged.connect(self._on_index_changed)

    def _on_index_changed(self, index: int) -> None:
        if self.itemText(index) != CUSTOM_COLOR:
            self._last_index = index
            self.colorChanged.emit()
            return
        # "Custom..." is an action, not a color: open the picker and
        # select what it returns, or go back to the previous color
        previous = self._last_index
        chosen = QtWidgets.QColorDialog.getColor(
            self.color(QtGui.QColor("yellow"), index=previous),
            self,
            "Pick color",
        )
        self.blockSignals(True)
        self.setCurrentIndex(previous)
        self.blockSignals(False)
        if chosen.isValid():
            self.set_value(chosen.name().upper())

    def value(self) -> str:
        """The selected item: "Auto", a preset name or a hexadecimal
        code."""
        return self.itemText(self._last_index)

    def set_value(self, value: str) -> bool:
        """Select ``value`` ("Auto", a preset name or any color name
        ``QColor`` accepts), adding custom colors as their hexadecimal
        code.

        Returns
        -------
        valid : bool
            False if ``value`` is not a valid color, in which case the
            selection is kept.
        """
        value = str(value)
        index = self.findText(value, QtCore.Qt.MatchFlag.MatchFixedString)
        if index < 0 or value == CUSTOM_COLOR:
            color = QtGui.QColor(value)
            if not color.isValid() or value == CUSTOM_COLOR:
                return False
            value = color.name().upper()
            index = self.findText(value)
            if index < 0:
                index = self.count() - 1  # before "Custom..."
                self.insertItem(index, _swatch([color]), value)
        self.setCurrentIndex(index)
        return True

    def color(
        self,
        auto_color: QtGui.QColor | str,
        index: int | None = None,
    ) -> QtGui.QColor:
        """The selected color.

        Parameters
        ----------
        auto_color : QColor or str
            Color returned if "Auto" is selected.
        index : int or None, optional
            Item to read instead of the selected one. Default None.

        Returns
        -------
        color : QColor
            The color.
        """
        text = self.value() if index is None else self.itemText(index)
        if text == AUTO_COLOR:
            return QtGui.QColor(auto_color)
        return QtGui.QColor(PRESET_COLORS.get(text, text))


class OverlayStyleWidget(QtWidgets.QWidget):
    """Set the appearance of one tool overlay, see ``FIELDS``.

    Parameters
    ----------
    fields : tuple of str
        The fields shown, in this order, see ``FIELDS``.
    defaults : dict, optional
        Values of the fields that differ from ``DEFAULTS``, restored by
        the reset button. Default None.
    auto_colors : tuple of str or None, optional
        The automatic colors on a dark and a light background, see
        ``ColorComboBox``. None offers no automatic color. Default is
        yellow and red.
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    changed : pyqtSignal
        Emitted when any field changes.
    color, drawing_color : ColorComboBox
        Color of the lines and labels, and of shapes still being drawn
        (for example, a rectangular pick being dragged).
    line_style : QComboBox
        Line pattern, see ``render.LINE_STYLES``.
    line_width : QDoubleSpinBox
        Line width in display pixels.
    opacity : QSpinBox
        Opacity of the lines and labels (%).
    fill_opacity : QSpinBox
        Opacity of the fill of closed shapes (%), or "Default".
    font_size : QSpinBox
        Pixel size of the labels, or "Default".
    marker_size : QSpinBox
        Size of the point markers in display pixels.
    """

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        fields: tuple[str, ...],
        defaults: dict | None = None,
        auto_colors: tuple[str, str] | None = ("yellow", "red"),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.fields = fields
        self.defaults = {**DEFAULTS, **(defaults or {})}
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)

        for i, field in enumerate(fields):
            widget = self._make(field, auto_colors)
            setattr(self, field, widget)
            label = QtWidgets.QLabel(FIELDS[field][0])
            label.setToolTip(widget.toolTip())
            # two fields per row
            row, column = divmod(i, 2)
            grid.addWidget(label, row, 2 * column)
            grid.addWidget(widget, row, 2 * column + 1)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setToolTip("Restore the default appearance.")
        reset_button.clicked.connect(self.reset)
        grid.addWidget(
            reset_button,
            (len(fields) + 1) // 2,
            3,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight,
        )
        self.reset()

    def _make(
        self,
        field: str,
        auto_colors: tuple[str, str] | None,
    ) -> QtWidgets.QWidget:
        """Create the widget of one field and connect it to
        ``changed``."""
        if field in ("color", "drawing_color"):
            widget = ColorComboBox(
                auto_colors=auto_colors if field == "color" else None
            )
            widget.colorChanged.connect(self.changed)
            widget.setToolTip(
                "Color of the lines and labels."
                if field == "color"
                else "Color of a shape while it is being drawn."
            )
            return widget
        if field == "line_style":
            widget = QtWidgets.QComboBox()
            widget.addItems(render.LINE_STYLES)
            widget.currentIndexChanged.connect(self.changed)
            widget.setToolTip("Pattern of the lines.")
            return widget
        if field == "line_width":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(1.0, 20.0)
            widget.setSingleStep(0.5)
            widget.setDecimals(1)
            widget.setSuffix(" px")
            widget.setToolTip(
                "Width of the lines in screen pixels. Lines wider than\n"
                "one pixel are antialiased."
            )
        elif field == "opacity":
            widget = QtWidgets.QSpinBox()
            widget.setRange(5, 100)
            widget.setSingleStep(10)
            widget.setSuffix(" %")
            widget.setToolTip("Opacity of the lines and labels.")
        elif field == "fill_opacity":
            widget = QtWidgets.QSpinBox()
            widget.setRange(DEFAULT_FILL, 100)
            widget.setSingleStep(5)
            widget.setSuffix(" %")
            widget.setSpecialValueText("Default")
            widget.setToolTip(
                "Opacity of the fill of closed shapes in the line color;\n"
                "0% draws outlines only.\n\n"
                f"Default: only brush picks are filled "
                f"({round(render.BRUSH_FILL_ALPHA / 2.55)}%)."
            )
        elif field == "font_size":
            widget = QtWidgets.QSpinBox()
            widget.setRange(DEFAULT_FONT_SIZE, 200)
            widget.setSuffix(" px")
            widget.setSpecialValueText("Default")
            widget.setToolTip("Size of the labels in screen pixels.")
        elif field == "marker_size":
            widget = QtWidgets.QSpinBox()
            widget.setRange(2, 200)
            widget.setSingleStep(2)
            widget.setSuffix(" px")
            widget.setToolTip("Size of the point markers in screen pixels.")
        else:
            raise ValueError(f"Unknown field: {field}")
        widget.setKeyboardTracking(False)
        widget.valueChanged.connect(self.changed)
        return widget

    def value(self, field: str):
        """Value of a field as saved in the user settings: None for
        "Default"."""
        widget = getattr(self, field)
        if field in ("color", "drawing_color"):
            return widget.value()
        if field == "line_style":
            return widget.currentText()
        value = widget.value()
        if field == "fill_opacity" and value == DEFAULT_FILL:
            return None
        if field == "font_size" and value == DEFAULT_FONT_SIZE:
            return None
        return value

    def set_value(self, field: str, value) -> None:
        """Set a field to ``value``, see ``value``. Invalid values are
        ignored."""
        widget = getattr(self, field)
        if field in ("color", "drawing_color"):
            widget.set_value(value)
        elif field == "line_style":
            if value in render.LINE_STYLES:
                widget.setCurrentText(value)
        else:
            if value is None:
                value = widget.minimum()  # "Default"
            try:
                value = float(value)
            except (TypeError, ValueError):
                return
            if isinstance(widget, QtWidgets.QSpinBox):
                value = round(value)
            widget.setValue(value)

    def reset(self) -> None:
        """Restore the defaults of all fields."""
        self.blockSignals(True)
        for field in self.fields:
            self.set_value(field, self.defaults[field])
        self.blockSignals(False)
        self.changed.emit()

    def settings(self) -> dict:
        """Values of all fields, keyed as in the user settings."""
        return {FIELDS[f][1]: self.value(f) for f in self.fields}

    def load_settings(self, settings: dict) -> None:
        """Set the fields from the user settings; missing or invalid
        values keep the current ones."""
        if not isinstance(settings, dict):
            return
        self.blockSignals(True)
        for field in self.fields:
            key = FIELDS[field][1]
            if key in settings:
                self.set_value(field, settings[key])
        self.blockSignals(False)
        self.changed.emit()

    def style(
        self,
        auto_color: QtGui.QColor | str = "yellow",
    ) -> render.OverlayStyle:
        """The chosen appearance; fields not shown keep their defaults.

        Parameters
        ----------
        auto_color : QColor or str, optional
            Color used if "Auto" is selected. Default "yellow".

        Returns
        -------
        style : render.OverlayStyle
            The appearance.
        """
        kwargs = {}
        if "color" in self.fields:
            kwargs["color"] = self.color.color(auto_color)
        if "line_style" in self.fields:
            kwargs["line_style"] = self.value("line_style")
        if "line_width" in self.fields:
            kwargs["line_width"] = self.value("line_width")
        if "opacity" in self.fields:
            kwargs["opacity"] = self.value("opacity") / 100
        if "fill_opacity" in self.fields:
            fill = self.value("fill_opacity")
            kwargs["fill_opacity"] = None if fill is None else fill / 100
        if "font_size" in self.fields:
            kwargs["font_size"] = self.value("font_size")
        return render.OverlayStyle(**kwargs)

    def drawing_style(
        self,
        auto_color: QtGui.QColor | str = "yellow",
    ) -> render.OverlayStyle:
        """The appearance of a shape still being drawn: ``style`` in
        the color chosen for drawing."""
        return replace(
            self.style(auto_color),
            color=self.drawing_color.color(auto_color),
        )
