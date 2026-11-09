"""
picasso.gui.scalebar_settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Settings of the scale bars of Picasso: where a scale bar is drawn, its
thickness and color, an optional background box, and the unit,
placement and font of the printed length. They are turned into the
arguments of ``picasso.render.draw_scalebar``.

Render's 2D and 3D windows set their scale bars, with the length, in
a ``ScaleBarGroupBox`` of their display settings. Localize and SPINNA
share one look of their scale bars (``settings["ScaleBar"]`` in the
user settings), set in a small ``ScaleBarDialog`` opened from each
place that shows a scale bar; whether the length is printed is set for
each place, see ``ScaleBarOptions``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore, QtGui, QtWidgets, sip

from .. import io, lib, render
from . import theme
from .overlay_style import AUTO_COLOR, PRESET_COLORS, ColorComboBox

#: Labels of the positions -> ``draw_scalebar``'s ``position``.
POSITIONS = {
    position.capitalize(): position for position in render.SCALEBAR_POSITIONS
}
#: Labels of where the length is printed -> ``label_position``.
LABEL_POSITIONS = {
    "Above the bar": "above",
    "Below the bar": "below",
}
#: Labels of the units of the printed length -> ``units``.
UNITS = {
    "nm": "nm",
    "µm": "um",
    "Automatic": "auto",
}
#: The application's font, see ``font_family``.
DEFAULT_FONT = "Default"
#: The look of a scale bar when not given otherwise, keyed as in the
#: user settings, see ``ScaleBarStyleWidget``.
LOOK_DEFAULTS = {
    "Position": "Bottom right",
    "Margins (px)": [35, 20],
    "Thickness (px)": 10,
    "Color": AUTO_COLOR,
    "Background opacity (%)": 0,
    "Unit": "nm",
    "Label": "Above the bar",
    "Font": DEFAULT_FONT,
    "Font size (px)": render.SCALEBAR_FONT_SIZE,
    "Bold": False,
    "Italic": False,
}
#: Settings of Render's scale bars when not given otherwise, see
#: ``ScaleBarGroupBox``.
DEFAULTS = {
    "Length (nm)": 500,
    "Automatic length": True,
    "Print length": False,
    **LOOK_DEFAULTS,
}
#: Settings of the length, not remembered across sessions: each
#: session starts with the automatic length.
LENGTH_KEYS = ("Length (nm)", "Automatic length")
#: Key of the look shared by Localize and SPINNA in the user settings.
SHARED_SETTINGS_KEY = "ScaleBar"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_margins(margins) -> bool:
    """Whether ``margins`` are the two numbers (x and y) of the margins
    of a scale bar."""
    return (
        isinstance(margins, (list, tuple))
        and len(margins) == 2
        and all(_is_number(m) for m in margins)
    )


def _clamped_sizes(settings: dict) -> dict:
    """The thickness, background opacity and font size in ``settings``
    that are numbers, rounded and clipped to their ranges."""
    sizes = {}
    for key, low, high in (
        ("Thickness (px)", 1, 200),
        ("Background opacity (%)", 0, 100),
        ("Font size (px)", 4, 200),
    ):
        value = settings.get(key)
        if _is_number(value):
            sizes[key] = min(high, max(low, int(round(value))))
    return sizes


def look_from_settings(settings: dict | None) -> dict:
    """The look of a scale bar in ``settings``, keyed as in
    ``LOOK_DEFAULTS``; missing or invalid values are the defaults.

    Parameters
    ----------
    settings : dict or None
        Settings, e.g., from the user settings.

    Returns
    -------
    look : dict
        The look, with every key of ``LOOK_DEFAULTS``.
    """
    look = {
        key: list(value) if isinstance(value, list) else value
        for key, value in LOOK_DEFAULTS.items()
    }
    if not isinstance(settings, dict):
        return look
    look.update(_clamped_sizes(settings))
    margins = settings.get("Margins (px)")
    if _valid_margins(margins):
        look["Margins (px)"] = [
            min(2000, max(0, int(round(m)))) for m in margins
        ]
    for key in ("Bold", "Italic"):
        if isinstance(settings.get(key), bool):
            look[key] = settings[key]
    for key, items in (
        ("Position", POSITIONS),
        ("Unit", UNITS),
        ("Label", LABEL_POSITIONS),
    ):
        if settings.get(key) in items:
            look[key] = settings[key]
    if isinstance(settings.get("Font"), str) and settings["Font"]:
        look["Font"] = settings["Font"]
    color = settings.get("Color")
    if isinstance(color, str) and (
        color == AUTO_COLOR
        or color in PRESET_COLORS
        or QtGui.QColor(color).isValid()
    ):
        look["Color"] = color
    return look


def style_kwargs_from_settings(
    settings: dict | None, white_background: bool = False
) -> dict:
    """Arguments of ``picasso.render.draw_scalebar`` that set the look
    in ``settings``, i.e., all but the image, viewport, length and
    whether it is printed.

    The font size follows the font size of the windows, see
    ``theme.scaled_text_size``; draw within ``theme.unscaled_text``
    for exported images.

    Parameters
    ----------
    settings : dict or None
        The look, see ``look_from_settings``.
    white_background : bool, optional
        Whether the image has a white background, which turns the
        automatic color black. Default False.

    Returns
    -------
    kwargs : dict
        Keyword arguments of ``draw_scalebar``.
    """
    look = look_from_settings(settings)
    color = look["Color"]
    if color == AUTO_COLOR:
        color = "black" if white_background else "white"
    return {
        "color": QtGui.QColor(PRESET_COLORS.get(color, color)),
        "display_height": look["Thickness (px)"],
        "margin": tuple(look["Margins (px)"]),
        "text_fontsize": theme.scaled_text_size(look["Font size (px)"]),
        "position": POSITIONS[look["Position"]],
        "label_position": LABEL_POSITIONS[look["Label"]],
        "units": UNITS[look["Unit"]],
        "font_family": None if look["Font"] == DEFAULT_FONT else look["Font"],
        "bold": look["Bold"],
        "italic": look["Italic"],
        "background_opacity": look["Background opacity (%)"] / 100,
    }


class ScaleBarStyleWidget(QtWidgets.QWidget):
    """Set the look of a scale bar, see ``LOOK_DEFAULTS``.

    The widgets are laid out in a four-column grid (``grid``) starting
    at row ``first_row``, so that the caller can add its own widgets in
    the rows above, aligned with them.

    Parameters
    ----------
    first_row : int, optional
        First row of the grid used by the widget. Default 0.
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    changed : pyqtSignal
        Emitted when any setting changes.
    grid : QGridLayout
        Layout of the widget.
    position : QComboBox
        Where the scale bar is drawn, see ``POSITIONS``.
    margin_x, margin_y : QSpinBox
        Distances from the nearest vertical and horizontal edges in
        screen pixels.
    thickness : QSpinBox
        Thickness of the bar in screen pixels.
    color : ColorComboBox
        Color of the bar and its length; "Auto" is white on a dark and
        black on a white background.
    background : QSpinBox
        Opacity (%) of a box behind the scale bar, 0 for none.
    units : QComboBox
        Unit of the printed length, see ``UNITS``.
    label_position : QComboBox
        Whether the length is printed above or below the bar.
    font_family : QComboBox
        Font of the printed length, or the application's font.
    font_size : QSpinBox
        Pixel size of the printed length.
    bold, italic : QCheckBox
        Whether the printed length is bold or italic.
    """

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        first_row: int = 0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        row = first_row

        def add(row, column, label, widget, tooltip, span=1):
            label = QtWidgets.QLabel(label)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            self.grid.addWidget(label, row, column)
            self.grid.addWidget(widget, row, column + 1, 1, span)
            return label

        self.position = QtWidgets.QComboBox()
        self.position.addItems(POSITIONS)
        self.position.currentIndexChanged.connect(self._on_position_changed)
        add(
            row, 0, "Position:", self.position, "Where the scale bar is drawn."
        )
        self.thickness = self._spin_box(1, 200, " px")
        add(
            row,
            2,
            "Thickness:",
            self.thickness,
            "Thickness of the bar in screen pixels.",
        )
        row += 1

        self.margin_x = self._spin_box(0, 2000, " px")
        self._margin_x_label = add(
            row,
            0,
            "Margin x:",
            self.margin_x,
            "Distance from the left or right edge in screen pixels.\n"
            "Not used by the centered positions.",
        )
        self.margin_y = self._spin_box(0, 2000, " px")
        add(
            row,
            2,
            "Margin y:",
            self.margin_y,
            "Distance from the top or bottom edge in screen pixels,\n"
            "including the printed length.",
        )
        row += 1

        self.color = ColorComboBox(auto_colors=("white", "black"))
        self.color.colorChanged.connect(self.changed)
        add(row, 0, "Color:", self.color, "Color of the bar and its length.")
        self.background = self._spin_box(0, 100, " %")
        self.background.setSingleStep(10)
        self.background.setSpecialValueText("None")
        add(
            row,
            2,
            "Background:",
            self.background,
            "Opacity of a box behind the scale bar that keeps it\n"
            "readable over bright localizations: black behind a light\n"
            "and white behind a dark scale bar.",
        )
        row += 1

        self.units = QtWidgets.QComboBox()
        self.units.addItems(UNITS)
        self.units.currentIndexChanged.connect(self.changed)
        self.label_position = QtWidgets.QComboBox()
        self.label_position.addItems(LABEL_POSITIONS)
        self.label_position.currentIndexChanged.connect(self.changed)
        self.font_family = QtWidgets.QComboBox()
        self.font_family.addItem(DEFAULT_FONT)
        self.font_family.setItemData(
            0,
            "The font of the windows.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.font_family.addItems(QtGui.QFontDatabase.families())
        self.font_family.setMaxVisibleItems(20)
        self.font_family.currentIndexChanged.connect(self.changed)
        self.font_size = self._spin_box(4, 200, " px")
        self.bold = QtWidgets.QCheckBox("Bold")
        self.bold.setToolTip("Print the length in bold.")
        self.bold.toggled.connect(self.changed)
        self.italic = QtWidgets.QCheckBox("Italic")
        self.italic.setToolTip("Print the length in italics.")
        self.italic.toggled.connect(self.changed)
        self._text_widgets = [
            add(
                row,
                0,
                "Unit:",
                self.units,
                "Unit of the printed length. Automatic: µm from 1 µm on,\n"
                "nm below.",
            ),
            self.units,
            add(
                row,
                2,
                "Label:",
                self.label_position,
                "Whether the length is printed above or below the bar.",
            ),
            self.label_position,
            add(
                row + 1,
                0,
                "Font:",
                self.font_family,
                "Font of the printed length.",
                span=3,
            ),
            self.font_family,
            add(
                row + 2,
                0,
                "Font size:",
                self.font_size,
                "Size of the printed length in screen pixels. Exported\n"
                "images use it whatever the font size of the windows.",
            ),
            self.font_size,
            self.bold,
            self.italic,
        ]
        style_row = QtWidgets.QHBoxLayout()
        style_row.addWidget(self.bold)
        style_row.addWidget(self.italic)
        style_row.addStretch()
        self.grid.addLayout(style_row, row + 2, 2, 1, 2)
        row += 3

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setToolTip("Restore the default look of the scale bar.")
        reset_button.clicked.connect(
            lambda: lib.confirm_restore_defaults(
                self, "the look of the scale bar"
            )
            and self.reset()
        )
        self.grid.addWidget(
            reset_button, row, 3, alignment=QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.load_settings(LOOK_DEFAULTS, notify=False)

    def _spin_box(
        self, low: int, high: int, suffix: str
    ) -> QtWidgets.QSpinBox:
        """Spin box connected to ``changed``."""
        spin_box = QtWidgets.QSpinBox()
        spin_box.setRange(low, high)
        spin_box.setSuffix(suffix)
        spin_box.setKeyboardTracking(False)
        spin_box.valueChanged.connect(self.changed)
        return spin_box

    def _on_position_changed(self, *args) -> None:
        self._update_margin_x()
        self.changed.emit()

    def _update_margin_x(self) -> None:
        """Gray out the horizontal margin of the centered positions."""
        centered = POSITIONS[self.position.currentText()].endswith("center")
        self.margin_x.setEnabled(not centered)
        self._margin_x_label.setEnabled(not centered)

    def set_text_enabled(self, enabled: bool) -> None:
        """Gray out the settings of the printed length, e.g., while it
        is not printed."""
        for widget in self._text_widgets:
            widget.setEnabled(enabled)

    def style_kwargs(self, white_background: bool = False) -> dict:
        """Arguments of ``picasso.render.draw_scalebar`` that set the
        look, see ``style_kwargs_from_settings``."""
        return style_kwargs_from_settings(self.settings(), white_background)

    def settings(self) -> dict:
        """The look, keyed as in the user settings."""
        return {
            "Position": self.position.currentText(),
            "Margins (px)": [self.margin_x.value(), self.margin_y.value()],
            "Thickness (px)": self.thickness.value(),
            "Color": self.color.value(),
            "Background opacity (%)": self.background.value(),
            "Unit": self.units.currentText(),
            "Label": self.label_position.currentText(),
            "Font": self.font_family.currentText(),
            "Font size (px)": self.font_size.value(),
            "Bold": self.bold.isChecked(),
            "Italic": self.italic.isChecked(),
        }

    def load_settings(
        self, settings: dict | None, notify: bool = True
    ) -> None:
        """Set the look from ``settings``; missing or invalid values
        keep the current ones.

        Parameters
        ----------
        settings : dict or None
            The look, see ``settings``.
        notify : bool, optional
            Whether to emit ``changed`` once. Default True.
        """
        if not isinstance(settings, dict):
            return
        widgets = [
            self.position,
            self.margin_x,
            self.margin_y,
            self.thickness,
            self.background,
            self.units,
            self.label_position,
            self.font_family,
            self.font_size,
            self.bold,
            self.italic,
        ]
        # the color box updates its selection on its own signal, so
        # ``changed`` is blocked instead
        self.blockSignals(True)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self._set_widgets(settings)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self.blockSignals(False)
        self._update_margin_x()
        if notify:
            self.changed.emit()

    def _set_widgets(self, settings: dict) -> None:
        """Set the widgets to the valid values in ``settings``, see
        ``load_settings``."""
        for key, spin_box in (
            ("Thickness (px)", self.thickness),
            ("Background opacity (%)", self.background),
            ("Font size (px)", self.font_size),
        ):
            if _is_number(settings.get(key)):
                spin_box.setValue(int(round(settings[key])))
        margins = settings.get("Margins (px)")
        if _valid_margins(margins):
            self.margin_x.setValue(int(round(margins[0])))
            self.margin_y.setValue(int(round(margins[1])))
        for key, check in (("Bold", self.bold), ("Italic", self.italic)):
            if isinstance(settings.get(key), bool):
                check.setChecked(settings[key])
        for key, combo, items in (
            ("Position", self.position, POSITIONS),
            ("Unit", self.units, UNITS),
            ("Label", self.label_position, LABEL_POSITIONS),
        ):
            if settings.get(key) in items:
                combo.setCurrentText(settings[key])
        font = settings.get("Font")
        # a font missing on this computer keeps the current one
        if isinstance(font, str) and self.font_family.findText(font) >= 0:
            self.font_family.setCurrentText(font)
        if "Color" in settings:
            self.color.set_value(settings["Color"])

    def reset(self) -> None:
        """Restore the default look."""
        self.load_settings(LOOK_DEFAULTS)


class ScaleBarGroupBox(lib.CollapsibleGroupBox):
    """Checkable, collapsible group box that shows a scale bar in
    Render and sets it up.

    Its settings are grayed out while it is unchecked.

    Parameters
    ----------
    optimal_length : callable or None, optional
        Returns the automatic length in nm for the current view, or
        None if there is no view yet. Called when "Automatic length" is
        ticked, so the length adjusts at once. Default None.
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    changed : pyqtSignal
        Emitted when the scale bar is shown or hidden or any setting
        changes, except for automatic length updates
        (``set_auto_length``).
    length : QSpinBox
        Length of the scale bar in nm.
    auto_length : QCheckBox
        Whether the length follows the zoom (set by the window).
    print_length : QCheckBox
        Whether the length is printed.
    style_widget : ScaleBarStyleWidget
        The look of the scale bar.
    """

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        optimal_length: Callable[[], int | None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Scale bar", parent=parent)
        self._optimal_length = optimal_length
        self.setCheckable(True)
        self.setChecked(False)
        # an unchecked group box disables (grays out) its widgets
        layout = QtWidgets.QVBoxLayout(self.content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.style_widget = ScaleBarStyleWidget(first_row=2)
        self.style_widget.changed.connect(self.changed)
        layout.addWidget(self.style_widget)
        # the length in the rows above the look, aligned with it
        grid = self.style_widget.grid

        length_label = QtWidgets.QLabel("Length (nm):")
        length_label.setToolTip("Length of the scale bar.")
        grid.addWidget(length_label, 0, 0)
        self.length = QtWidgets.QSpinBox()
        self.length.setRange(1, 100_000)
        self.length.setKeyboardTracking(False)
        self.length.setToolTip(length_label.toolTip())
        self.length.valueChanged.connect(self._on_length_changed)
        grid.addWidget(self.length, 0, 1)
        self.auto_length = QtWidgets.QCheckBox("Automatic length")
        self.auto_length.setToolTip(
            "Set the scale bar length to approximately 1/8 of the current\n"
            "viewport width."
        )
        self.auto_length.toggled.connect(self._on_auto_length_toggled)
        grid.addWidget(self.auto_length, 0, 2, 1, 2)
        self.print_length = QtWidgets.QCheckBox("Print length")
        self.print_length.setToolTip("Print the length of the scale bar.")
        self.print_length.toggled.connect(self._on_print_length_toggled)
        grid.addWidget(self.print_length, 1, 0, 1, 2)

        self.toggled.connect(self.changed)
        self._load({**DEFAULTS})

    def _on_length_changed(self, *args) -> None:
        """A length typed by the user ends the automatic length."""
        if self.auto_length.isChecked():
            self.auto_length.blockSignals(True)
            self.auto_length.setChecked(False)
            self.auto_length.blockSignals(False)
        self.changed.emit()

    def _on_auto_length_toggled(self, checked: bool) -> None:
        """Ticking "Automatic length" adjusts the length at once."""
        if checked:
            self._apply_optimal_length()
        self.changed.emit()

    def _apply_optimal_length(self) -> None:
        if self._optimal_length is not None:
            length = self._optimal_length()
            if length is not None:
                self.set_auto_length(length)

    def _on_print_length_toggled(self, checked: bool) -> None:
        self.style_widget.set_text_enabled(checked)
        self.changed.emit()

    def set_auto_length(self, length_nm: int | float) -> None:
        """Set the length without emitting ``changed``, if the length
        is automatic."""
        if self.auto_length.isChecked():
            self.length.blockSignals(True)
            self.length.setValue(int(round(length_nm)))
            self.length.blockSignals(False)

    def style_kwargs(self, white_background: bool = False) -> dict:
        """Arguments of ``picasso.render.draw_scalebar`` that set the
        look, see ``style_kwargs_from_settings``."""
        return self.style_widget.style_kwargs(white_background)

    def draw(
        self,
        image: QtGui.QImage,
        viewport: tuple[tuple[float, float], tuple[float, float]],
        white_background: bool = False,
        force: bool = False,
    ) -> QtGui.QImage:
        """Draw the scale bar onto ``image`` if it is shown.

        Parameters
        ----------
        image : QImage
            Image of the rendered localizations.
        viewport : tuple
            Field of view shown in ``image`` (nm).
        white_background : bool, optional
            Whether the image has a white background, see
            ``style_kwargs``. Default False.
        force : bool, optional
            If True, draw the scale bar also while it is hidden.
            Default False.

        Returns
        -------
        image : QImage
            Image with the scale bar.
        """
        if not (force or self.isChecked()):
            return image
        return render.draw_scalebar(
            image=image,
            viewport=viewport,
            scalebar_length_nm=self.length.value(),
            display_length=self.print_length.isChecked(),
            **self.style_kwargs(white_background),
        )

    def settings(self, include_length: bool = True) -> dict:
        """All settings except whether the scale bar is shown, keyed as
        in the user settings.

        Parameters
        ----------
        include_length : bool, optional
            If False, leave out the length and whether it is automatic
            (``LENGTH_KEYS``), which the user settings do not keep.
            Default True.

        Returns
        -------
        settings : dict
            The settings.
        """
        settings = {
            "Length (nm)": self.length.value(),
            "Automatic length": self.auto_length.isChecked(),
            "Print length": self.print_length.isChecked(),
            **self.style_widget.settings(),
        }
        if not include_length:
            for key in LENGTH_KEYS:
                del settings[key]
        return settings

    def load_settings(
        self, settings: dict | None, include_length: bool = True
    ) -> None:
        """Set the settings from the user settings; missing or invalid
        values keep the current ones. An automatic length adjusts to
        the current view. Emits ``changed`` once.

        Parameters
        ----------
        settings : dict or None
            The settings, see ``settings``.
        include_length : bool, optional
            If False, ignore the length and whether it is automatic
            (``LENGTH_KEYS``). Default True.
        """
        if not isinstance(settings, dict):
            return
        if not include_length:
            settings = {
                key: value
                for key, value in settings.items()
                if key not in LENGTH_KEYS
            }
        self._load(settings)
        if self.auto_length.isChecked():
            self._apply_optimal_length()
        self.changed.emit()

    def _load(self, settings: dict) -> None:
        """Set the widgets from ``settings`` without emitting
        ``changed``; see ``load_settings``."""
        widgets = [self.length, self.auto_length, self.print_length]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if _is_number(settings.get("Length (nm)")):
                self.length.setValue(int(round(settings["Length (nm)"])))
            for key, check in (
                ("Automatic length", self.auto_length),
                ("Print length", self.print_length),
            ):
                if isinstance(settings.get(key), bool):
                    check.setChecked(settings[key])
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.style_widget.load_settings(settings, notify=False)
        self.style_widget.set_text_enabled(self.print_length.isChecked())


class _Hub(QtCore.QObject):
    """Holds the look shared by Localize and SPINNA and announces its
    changes to the open windows."""

    changed = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.look = None


_hub = None


def hub() -> _Hub:
    """The object whose ``changed`` signal announces a new shared
    look."""
    global _hub
    if _hub is None or sip.isdeleted(_hub):
        _hub = _Hub()
    return _hub


def shared_look() -> dict:
    """The look of the scale bars of Localize and SPINNA, see
    ``look_from_settings``; read from the user settings on first use."""
    if hub().look is None:
        reload_shared_look()
    return dict(hub().look)


def reload_shared_look() -> None:
    """Read the shared look from the user settings again, e.g., after
    another Picasso program changed it, announcing it if it changed."""
    look = look_from_settings(io.load_user_settings()[SHARED_SETTINGS_KEY])
    if look != hub().look:
        hub().look = look
        hub().changed.emit()


def set_shared_look(look: dict) -> None:
    """Save ``look`` as the look of the scale bars of Localize and
    SPINNA in the user settings and announce it to the open windows."""
    look = look_from_settings(look)
    if look == hub().look:
        return
    hub().look = look
    settings = io.load_user_settings()
    settings[SHARED_SETTINGS_KEY] = look
    io.save_user_settings(settings)
    hub().changed.emit()


class ScaleBarOptions(QtCore.QObject):
    """The scale bar of one place of Localize or SPINNA: the shared look
    (``shared_look``) and whether the length is printed there, saved
    in ``settings[section][print_length_key]``.

    Parameters
    ----------
    section : str
        Section of the user settings, e.g., "Localize".
    print_length_key : str
        Key of whether the length is printed in ``section``.
    print_length_default : bool
        Whether the length is printed if the user settings do not say.
    title : str, optional
        Title of the dialog, see ``show_dialog``. Default "Scale bar".
    parent : QObject or None, optional
        Parent object. Default None.

    Attributes
    ----------
    changed : pyqtSignal
        Emitted when the look or whether the length is printed changes.
    print_length : bool
        Whether the length is printed.
    """

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        section: str,
        print_length_key: str,
        print_length_default: bool,
        title: str = "Scale bar",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.section = section
        self.print_length_key = print_length_key
        self.title = title
        saved = io.load_user_settings()[section].get(print_length_key)
        self.print_length = (
            saved if isinstance(saved, bool) else print_length_default
        )
        self._dialog = None
        hub().changed.connect(self.changed)

    def set_print_length(self, print_length: bool) -> None:
        """Set and save whether the length is printed."""
        if print_length == self.print_length:
            return
        self.print_length = print_length
        settings = io.load_user_settings()
        settings[self.section][self.print_length_key] = print_length
        io.save_user_settings(settings)
        self.changed.emit()

    def draw_kwargs(self, white_background: bool = False) -> dict:
        """Arguments of ``picasso.render.draw_scalebar`` besides the
        image, viewport and length: whether the length is printed and
        the shared look, see ``style_kwargs_from_settings``."""
        return {
            "display_length": self.print_length,
            **style_kwargs_from_settings(shared_look(), white_background),
        }

    def settings_button(
        self, parent: QtWidgets.QWidget | None = None
    ) -> QtWidgets.QToolButton:
        """A small button that opens the dialog, to sit next to the
        control that shows the scale bar.

        Parameters
        ----------
        parent : QWidget or None, optional
            Parent of the button and the dialog. Default None.

        Returns
        -------
        button : QToolButton
            The button.
        """
        button = QtWidgets.QToolButton(parent)
        button.setIcon(theme.icon("display-settings"))
        button.setToolTip(
            "Scale bar settings: the look, shared with Localize and\n"
            "SPINNA, and whether the length is printed."
        )
        button.setAutoRaise(True)
        button.clicked.connect(lambda: self.show_dialog(button.window()))
        return button

    def show_dialog(
        self, parent: QtWidgets.QWidget | None = None
    ) -> ScaleBarDialog:
        """Show the dialog that sets the scale bar, creating it on
        first use.

        Parameters
        ----------
        parent : QWidget or None, optional
            Parent of the dialog when it is created. Default None.

        Returns
        -------
        dialog : ScaleBarDialog
            The dialog.
        """
        if self._dialog is None or sip.isdeleted(self._dialog):
            self._dialog = ScaleBarDialog(self, parent)
        reload_shared_look()
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()
        return self._dialog


class ScaleBarDialog(lib.Dialog):
    """Set whether the length of one place's scale bar is printed and
    the look shared by the scale bars of Localize and SPINNA.

    Changes are applied immediately; the look is saved shortly after
    the last change, so that dragging a spin box does not save it for
    every step.

    Parameters
    ----------
    options : ScaleBarOptions
        The scale bar set.
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    print_length : QCheckBox
        Whether the length is printed.
    style_widget : ScaleBarStyleWidget
        The shared look.
    """

    #: Delay between the last change and saving the look (ms).
    DELAY = 150

    def __init__(
        self,
        options: ScaleBarOptions,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.options = options
        self.setWindowTitle(options.title)
        # an open dialog must not keep the app running after the last
        # main window closed
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.DELAY)
        self._timer.timeout.connect(self._save_look)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

        self.style_widget = ScaleBarStyleWidget(first_row=1)
        self.style_widget.load_settings(shared_look(), notify=False)
        self.style_widget.changed.connect(self._on_look_changed)
        self.print_length = QtWidgets.QCheckBox("Print length")
        self.print_length.setToolTip("Print the length of the scale bar.")
        self.print_length.setChecked(options.print_length)
        self.print_length.toggled.connect(self._on_print_length_toggled)
        self.style_widget.grid.addWidget(self.print_length, 0, 0, 1, 2)
        self.style_widget.set_text_enabled(options.print_length)
        layout.addWidget(self.style_widget)

        note = QtWidgets.QLabel(
            "The look is shared by the scale bars of Localize and SPINNA."
        )
        note.setWordWrap(True)
        note.setEnabled(False)  # muted, like a hint
        layout.addWidget(note)
        hub().changed.connect(self._on_shared_look_changed)

    def _on_print_length_toggled(self, checked: bool) -> None:
        self.style_widget.set_text_enabled(checked)
        self.options.set_print_length(checked)

    def _on_look_changed(self) -> None:
        self._timer.start()

    def _save_look(self) -> None:
        set_shared_look(self.style_widget.settings())

    def _on_shared_look_changed(self) -> None:
        """Show a look set in another dialog."""
        if sip.isdeleted(self):
            return
        look = shared_look()
        if look_from_settings(self.style_widget.settings()) != look:
            self.style_widget.load_settings(look, notify=False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # save a change still waiting for the delay now, so that the
        # timer does not fire after the dialog is gone
        if self._timer.isActive():
            self._timer.stop()
            self._save_look()
        super().closeEvent(event)
