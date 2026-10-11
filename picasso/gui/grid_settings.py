"""
picasso.gui.grid_settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Group box with the settings of the reference grid of Picasso: Render's
2D and 3D windows: spacing, the planes and their placement (3D only),
whether the grid is drawn behind or over the localizations, and the
look of its lines. The settings are turned into a
``picasso.render.GridOverlay`` for the drawing functions and saved in
the user settings.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from PyQt6 import QtCore, QtWidgets

from .. import render
from .overlay_style import AUTO_COLOR, OverlayStyleWidget

#: Labels of the 3D grid positions -> ``GridOverlay.position``.
POSITIONS = {
    "Back of the data": "back",
    "Through the center": "center",
}
#: Fields of the line appearance, see ``OverlayStyleWidget``.
STYLE_FIELDS = ("color", "line_style", "line_width", "opacity")
#: Appearance that differs from the ``OverlayStyleWidget`` defaults.
STYLE_DEFAULTS = {"opacity": 40}
DEFAULT_SPACING = 500  # nm


class GridGroupBox(QtWidgets.QGroupBox):
    """Checkable group box that shows a reference grid and sets it up.

    Its settings are grayed out while it is unchecked.

    Parameters
    ----------
    three_d : bool, optional
        If True, also offer the grid planes and their position, for
        the 3D window. Default False.
    optimal_spacing : callable or None, optional
        Returns the automatic spacing in nm for the current view, or
        None if there is no view yet. Called when "Automatic" is ticked,
        so the spacing adjusts at once. Default None.
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    changed : pyqtSignal
        Emitted when the grid is shown or hidden or any setting changes,
        except for automatic spacing updates (``set_auto_spacing``).
    spacing : QSpinBox
        Distance between the grid lines in nm.
    auto_spacing : QCheckBox
        Whether the spacing follows the zoom (set by the window).
    planes : dict
        3D only: plane name ("xy", "xz", "yz") -> QCheckBox.
    position : QComboBox
        3D only: where the planes are placed, see ``POSITIONS``.
    blend : QComboBox
        Whether the grid is drawn behind or over the localizations.
    style_widget : OverlayStyleWidget
        Color, line style, line width and opacity of the lines.
    """

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        three_d: bool = False,
        optimal_spacing: Callable[[], float | None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Grid", parent)
        self.three_d = three_d
        self._optimal_spacing = optimal_spacing
        self.setCheckable(True)
        self.setChecked(False)
        self.setToolTip(
            "Draw a reference grid fixed to the data"
            + (", rotating with it." if three_d else ".")
        )
        # an unchecked group box disables (grays out) its widgets
        grid = QtWidgets.QGridLayout(self)
        row = 0

        spacing_label = QtWidgets.QLabel("Spacing (nm):")
        spacing_label.setToolTip("Distance between neighboring grid lines.")
        grid.addWidget(spacing_label, row, 0)
        self.spacing = QtWidgets.QSpinBox()
        self.spacing.setRange(1, 1_000_000)
        self.spacing.setValue(DEFAULT_SPACING)
        self.spacing.setKeyboardTracking(False)
        self.spacing.setToolTip(spacing_label.toolTip())
        self.spacing.valueChanged.connect(self._on_spacing_changed)
        grid.addWidget(self.spacing, row, 1)
        self.auto_spacing = QtWidgets.QCheckBox("Automatic")
        self.auto_spacing.setChecked(True)
        self.auto_spacing.setToolTip(
            "Set the spacing to approximately 1/8 of the viewport width,\n"
            "like the automatic scale bar length."
        )
        self.auto_spacing.toggled.connect(self._on_auto_spacing_toggled)
        grid.addWidget(self.auto_spacing, row, 2, 1, 2)
        row += 1

        self.planes = {}
        if three_d:
            planes_label = QtWidgets.QLabel("Planes:")
            planes_label.setToolTip("Grid planes drawn.")
            grid.addWidget(planes_label, row, 0)
            planes_row = QtWidgets.QHBoxLayout()
            for plane in render.GRID_PLANES:
                check = QtWidgets.QCheckBox(plane.upper())
                check.setChecked(True)
                check.setToolTip(
                    f"Draw the grid plane spanned by the {plane[0]} and "
                    f"{plane[1]} axes."
                )
                check.toggled.connect(self.changed)
                planes_row.addWidget(check)
                self.planes[plane] = check
            planes_row.addStretch()
            grid.addLayout(planes_row, row, 1, 1, 3)
            row += 1

            position_label = QtWidgets.QLabel("Position:")
            position_label.setToolTip(
                "Back of the data: each plane lies on the side of the\n"
                "data's bounding box facing away from you, so the planes\n"
                "form the back walls of a box around the data and swap\n"
                "sides as it rotates.\n"
                "Through the center: each plane passes through the\n"
                "center of the bounding box."
            )
            grid.addWidget(position_label, row, 0)
            self.position = QtWidgets.QComboBox()
            self.position.addItems(POSITIONS)
            self.position.setToolTip(position_label.toolTip())
            self.position.currentIndexChanged.connect(self.changed)
            grid.addWidget(self.position, row, 1, 1, 3)
            row += 1

        blend_label = QtWidgets.QLabel("Draw:")
        blend_label.setToolTip(
            "Behind localizations: the grid shows where there are no\n"
            "localizations and is covered the more, the brighter they\n"
            "are.\n"
            "Over localizations: the grid is drawn on top."
        )
        grid.addWidget(blend_label, row, 0)
        self.blend = QtWidgets.QComboBox()
        self.blend.addItems(render.GRID_BLEND_MODES)
        self.blend.setToolTip(blend_label.toolTip())
        self.blend.currentIndexChanged.connect(self.changed)
        grid.addWidget(self.blend, row, 1, 1, 3)
        row += 1

        self.style_widget = OverlayStyleWidget(
            STYLE_FIELDS,
            defaults=STYLE_DEFAULTS,
            auto_colors=("white", "black"),
        )
        self.style_widget.changed.connect(self.changed)
        grid.addWidget(self.style_widget, row, 0, 1, 4)

        self.toggled.connect(self.changed)

    def _on_spacing_changed(self, *args) -> None:
        """A spacing typed by the user ends the automatic spacing."""
        if self.auto_spacing.isChecked():
            self.auto_spacing.blockSignals(True)
            self.auto_spacing.setChecked(False)
            self.auto_spacing.blockSignals(False)
        self.changed.emit()

    def _on_auto_spacing_toggled(self, checked: bool) -> None:
        """Ticking "Automatic" adjusts the spacing at once."""
        if checked and self._optimal_spacing is not None:
            spacing = self._optimal_spacing()
            if spacing is not None:
                self.set_auto_spacing(spacing)
        self.changed.emit()

    def set_auto_spacing(self, spacing_nm: int | float) -> None:
        """Set the spacing without emitting ``changed``, if the spacing
        is automatic."""
        if self.auto_spacing.isChecked():
            self.spacing.blockSignals(True)
            self.spacing.setValue(int(round(spacing_nm)))
            self.spacing.blockSignals(False)

    def grid(self) -> render.GridOverlay | None:
        """The grid to draw, or None if it is hidden. An automatic
        color is left to the drawing functions (None), which follow
        the background."""
        if not self.isChecked():
            return None
        style = self.style_widget.style()
        if self.style_widget.value("color") == AUTO_COLOR:
            style = replace(style, color=None)
        kwargs = {}
        if self.three_d:
            kwargs["planes"] = tuple(
                plane
                for plane, check in self.planes.items()
                if check.isChecked()
            )
            kwargs["position"] = POSITIONS[self.position.currentText()]
        return render.GridOverlay(
            spacing_nm=float(self.spacing.value()),
            style=style,
            blend=self.blend.currentText(),
            **kwargs,
        )

    def export_info(self) -> dict:
        """Grid settings saved with exported images."""
        grid = self.grid()
        info = {"Grid spacing (nm)": None if grid is None else grid.spacing_nm}
        if grid is not None and self.three_d:
            info["Grid planes"] = list(grid.planes)
            info["Grid position"] = grid.position
        return info

    def settings(self) -> dict:
        """All settings except whether the grid is shown, keyed as in
        the user settings."""
        settings = {
            "Spacing (nm)": self.spacing.value(),
            "Automatic spacing": self.auto_spacing.isChecked(),
            "Draw": self.blend.currentText(),
            **self.style_widget.settings(),
        }
        if self.three_d:
            settings["Planes"] = [
                plane
                for plane, check in self.planes.items()
                if check.isChecked()
            ]
            settings["Position"] = self.position.currentText()
        return settings

    def load_settings(self, settings: dict | None) -> None:
        """Set the settings from the user settings; missing or invalid
        values keep the current ones. Emits ``changed`` once."""
        if not isinstance(settings, dict):
            return
        widgets = [self.spacing, self.auto_spacing, self.blend]
        widgets += list(self.planes.values())
        if self.three_d:
            widgets.append(self.position)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            spacing = settings.get("Spacing (nm)")
            if isinstance(spacing, (int, float)) and spacing >= 1:
                self.spacing.setValue(int(round(spacing)))
            auto = settings.get("Automatic spacing")
            if isinstance(auto, bool):
                self.auto_spacing.setChecked(auto)
            if settings.get("Draw") in render.GRID_BLEND_MODES:
                self.blend.setCurrentText(settings["Draw"])
            planes = settings.get("Planes")
            if self.three_d and isinstance(planes, list):
                for plane, check in self.planes.items():
                    check.setChecked(plane in planes)
            if self.three_d and settings.get("Position") in POSITIONS:
                self.position.setCurrentText(settings["Position"])
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.style_widget.blockSignals(True)
        self.style_widget.load_settings(settings)
        self.style_widget.blockSignals(False)
        self.changed.emit()
