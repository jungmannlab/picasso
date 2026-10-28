"""
picasso.gui.scalebar_settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group box with the settings of the scale bar of Picasso: Render's 2D
and 3D windows: its length, where it is drawn, its thickness and color,
an optional background box, and the unit, placement and font of the
printed length. The settings are turned into the arguments of
``picasso.render.draw_scalebar`` and saved in the user settings.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore, QtGui, QtWidgets

from .. import lib, render
from . import theme
from .overlay_style import ColorComboBox

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
#: Settings when not given otherwise, keyed as in the user settings.
DEFAULTS = {
    "Length (nm)": 500,
    "Automatic length": True,
    "Print length": False,
    "Position": "Bottom right",
    "Margins (px)": [35, 20],
    "Thickness (px)": 10,
    "Color": "Auto",
    "Background opacity (%)": 0,
    "Unit": "nm",
    "Label": "Above the bar",
    "Font": DEFAULT_FONT,
    "Font size (px)": render.SCALEBAR_FONT_SIZE,
    "Bold": False,
    "Italic": False,
}
#: Settings of the length, not remembered across sessions: each
#: session starts with the automatic length.
LENGTH_KEYS = ("Length (nm)", "Automatic length")
#: Settings restored by the reset button: all but the length.
APPEARANCE_KEYS = tuple(
    key for key in DEFAULTS if key not in (*LENGTH_KEYS, "Print length")
)


class ScaleBarGroupBox(lib.CollapsibleGroupBox):
    """Checkable, collapsible group box that shows a scale bar and sets
    it up.

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
    print_length : QCheckBox
        Whether the length is printed.
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
        optimal_length: Callable[[], int | None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Scale bar", parent=parent)
        self._optimal_length = optimal_length
        self.setCheckable(True)
        self.setChecked(False)
        # an unchecked group box disables (grays out) its widgets
        grid = QtWidgets.QGridLayout(self.content)

        def add(row, column, label, widget, tooltip, span=1):
            label = QtWidgets.QLabel(label)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            grid.addWidget(label, row, column)
            grid.addWidget(widget, row, column + 1, 1, span)
            return label

        self.length = QtWidgets.QSpinBox()
        self.length.setRange(1, 100_000)
        self.length.setKeyboardTracking(False)
        self.length.valueChanged.connect(self._on_length_changed)
        add(0, 0, "Length (nm):", self.length, "Length of the scale bar.")
        self.auto_length = QtWidgets.QCheckBox("Automatic length")
        self.auto_length.setToolTip(
            "Set the scale bar length to approximately 1/8 of the current\n"
            "viewport width."
        )
        self.auto_length.toggled.connect(self._on_auto_length_toggled)
        grid.addWidget(self.auto_length, 0, 2, 1, 2)

        self.position = QtWidgets.QComboBox()
        self.position.addItems(POSITIONS)
        self.position.currentIndexChanged.connect(self._on_position_changed)
        add(1, 0, "Position:", self.position, "Where the scale bar is drawn.")
        self.thickness = self._spin_box(1, 200, " px")
        add(
            1,
            2,
            "Thickness:",
            self.thickness,
            "Thickness of the bar in screen pixels.",
        )

        self.margin_x = self._spin_box(0, 2000, " px")
        self._margin_x_label = add(
            2,
            0,
            "Margin x:",
            self.margin_x,
            "Distance from the left or right edge in screen pixels.\n"
            "Not used by the centered positions.",
        )
        self.margin_y = self._spin_box(0, 2000, " px")
        add(
            2,
            2,
            "Margin y:",
            self.margin_y,
            "Distance from the top or bottom edge in screen pixels,\n"
            "including the printed length.",
        )

        self.color = ColorComboBox(auto_colors=("white", "black"))
        self.color.colorChanged.connect(self.changed)
        add(
            3,
            0,
            "Color:",
            self.color,
            "Color of the bar and its length.",
        )
        self.background = self._spin_box(0, 100, " %")
        self.background.setSingleStep(10)
        self.background.setSpecialValueText("None")
        add(
            3,
            2,
            "Background:",
            self.background,
            "Opacity of a box behind the scale bar that keeps it\n"
            "readable over bright localizations: black behind a light\n"
            "and white behind a dark scale bar.",
        )

        self.print_length = QtWidgets.QCheckBox("Print length")
        self.print_length.setToolTip("Print the length of the scale bar.")
        self.print_length.toggled.connect(self._on_print_length_toggled)
        grid.addWidget(self.print_length, 4, 0, 1, 2)
        self.units = QtWidgets.QComboBox()
        self.units.addItems(UNITS)
        self.units.currentIndexChanged.connect(self.changed)
        self._text_widgets = [
            add(
                4,
                2,
                "Unit:",
                self.units,
                "Unit of the printed length. Automatic: µm from 1 µm on,\n"
                "nm below.",
            ),
            self.units,
        ]

        self.label_position = QtWidgets.QComboBox()
        self.label_position.addItems(LABEL_POSITIONS)
        self.label_position.currentIndexChanged.connect(self.changed)
        self.font_size = self._spin_box(4, 200, " px")
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
        self.bold = QtWidgets.QCheckBox("Bold")
        self.bold.setToolTip("Print the length in bold.")
        self.bold.toggled.connect(self.changed)
        self.italic = QtWidgets.QCheckBox("Italic")
        self.italic.setToolTip("Print the length in italics.")
        self.italic.toggled.connect(self.changed)
        self._text_widgets += [
            add(
                5,
                0,
                "Label:",
                self.label_position,
                "Whether the length is printed above or below the bar.",
            ),
            self.label_position,
            add(
                5,
                2,
                "Font size:",
                self.font_size,
                "Size of the printed length in screen pixels. Exported\n"
                "images use it whatever the font size of the windows.",
            ),
            self.font_size,
            add(
                6,
                0,
                "Font:",
                self.font_family,
                "Font of the printed length.",
                span=3,
            ),
            self.font_family,
            self.bold,
            self.italic,
        ]
        grid.addWidget(self.bold, 7, 0)
        grid.addWidget(self.italic, 7, 1)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.setToolTip(
            "Restore the default appearance; the length is kept."
        )
        reset_button.clicked.connect(
            lambda: lib.confirm_restore_defaults(
                self, "the appearance of the scale bar"
            )
            and self.reset_appearance()
        )
        grid.addWidget(
            reset_button, 7, 3, alignment=QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.toggled.connect(self.changed)
        self._load({**DEFAULTS})

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

    def _on_position_changed(self, *args) -> None:
        self._update_enabled()
        self.changed.emit()

    def _on_print_length_toggled(self, *args) -> None:
        self._update_enabled()
        self.changed.emit()

    def _update_enabled(self) -> None:
        """Gray out the settings of the printed length while it is not
        printed and the horizontal margin of the centered positions."""
        printed = self.print_length.isChecked()
        for widget in self._text_widgets:
            widget.setEnabled(printed)
        centered = POSITIONS[self.position.currentText()].endswith("center")
        self.margin_x.setEnabled(not centered)
        self._margin_x_label.setEnabled(not centered)

    def set_auto_length(self, length_nm: int | float) -> None:
        """Set the length without emitting ``changed``, if the length
        is automatic."""
        if self.auto_length.isChecked():
            self.length.blockSignals(True)
            self.length.setValue(int(round(length_nm)))
            self.length.blockSignals(False)

    def style_kwargs(self, white_background: bool = False) -> dict:
        """Arguments of ``picasso.render.draw_scalebar`` that set the
        appearance, i.e., all but the image, viewport, length and
        whether it is printed.

        The font size follows the font size of the windows, see
        ``theme.scaled_text_size``; draw within
        ``theme.unscaled_text`` for exported images.

        Parameters
        ----------
        white_background : bool, optional
            Whether the image has a white background, which turns the
            automatic color black. Default False.

        Returns
        -------
        kwargs : dict
            Keyword arguments of ``draw_scalebar``.
        """
        family = self.font_family.currentText()
        return {
            "color": self.color.color(
                "black" if white_background else "white"
            ),
            "display_height": self.thickness.value(),
            "margin": (self.margin_x.value(), self.margin_y.value()),
            "text_fontsize": theme.scaled_text_size(self.font_size.value()),
            "position": POSITIONS[self.position.currentText()],
            "label_position": LABEL_POSITIONS[
                self.label_position.currentText()
            ],
            "units": UNITS[self.units.currentText()],
            "font_family": None if family == DEFAULT_FONT else family,
            "bold": self.bold.isChecked(),
            "italic": self.italic.isChecked(),
            "background_opacity": self.background.value() / 100,
        }

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

    def reset_appearance(self) -> None:
        """Restore the default appearance; the length is kept."""
        self._load({key: DEFAULTS[key] for key in APPEARANCE_KEYS})
        self.changed.emit()

    def _load(self, settings: dict) -> None:
        """Set the widgets from ``settings`` without emitting
        ``changed``; see ``load_settings``."""
        widgets = [
            self.length,
            self.auto_length,
            self.print_length,
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
            for key, spin_box in (
                ("Length (nm)", self.length),
                ("Thickness (px)", self.thickness),
                ("Background opacity (%)", self.background),
                ("Font size (px)", self.font_size),
            ):
                value = settings.get(key)
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    spin_box.setValue(int(round(value)))
            margins = settings.get("Margins (px)")
            if (
                isinstance(margins, (list, tuple))
                and len(margins) == 2
                and all(
                    isinstance(m, (int, float)) and not isinstance(m, bool)
                    for m in margins
                )
            ):
                self.margin_x.setValue(int(round(margins[0])))
                self.margin_y.setValue(int(round(margins[1])))
            for key, check in (
                ("Automatic length", self.auto_length),
                ("Print length", self.print_length),
                ("Bold", self.bold),
                ("Italic", self.italic),
            ):
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
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self.blockSignals(False)
        self._update_enabled()
