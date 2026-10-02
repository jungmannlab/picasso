"""
picasso.gui.theme
~~~~~~~~~~~~~~~~~

Look of the windows of every Picasso GUI. ``run_gui`` applies the theme
to the application before the main window is built.

The theme is Qt's Fusion style, which draws the same widgets on every
platform, with a palette of neutral grays and one accent color, a small
stylesheet that rounds buttons and group boxes and, for each density,
the spacing of the layouts. Colors of image content (rendered
localizations, tool overlays, charts) are not part of it; the charts
have their own ``PlotStyle``.

The user chooses in ``AppearanceDialog`` (File > Appearance...):

- the mode: "System" follows the light or dark appearance of the
  operating system, also when it changes while Picasso runs; "Light"
  and "Dark" fix it; "Native" is the platform's own style, the look of
  Picasso before themes existed;
- the accent color of selections, checked and default buttons, sliders
  and focus frames;
- the font size, in percent of the system's;
- the density, i.e., the spacing of controls;
- how the toolbars of Render and Localize show their buttons.

Icons are single-color SVG files in ``ICONS_DIR`` (``picasso/gui/icons``),
named as the callers of ``icon`` and ``add_toolbar`` use them, e.g.,
``open.svg``. ``icon`` draws them in the colors of the theme, so any
color in the files is ignored; a missing file leaves the button with its
text. Without an SVG, an ``.ico`` or ``.png`` of the same name is used
the same way, e.g., Average's application icon. The SVG icons are from
Lucide (https://lucide.dev, ISC license, see
``LICENSES/Lucide-LICENSE.txt``).

``current`` returns the appearance saved in the user settings
(``settings["Appearance"]``); ``set_current`` saves a new one and
applies it to the running application. ``apply`` announces every
appearance it applies, also when the operating system switches between
light and dark, through ``hub().changed``; charts in the "Same as
windows" theme (``picasso.gui.plot_style``) follow it. Other Picasso
GUIs that are running take a new appearance on when they are started
next.

Widgets that paint themselves take their colors from their palette and
their stylesheets refer to it (e.g., ``palette(mid)``), so that they
follow the theme. ``set_button_state`` marks a button, e.g., as done
("ok"), in the colors of the theme.

:author: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace

from PyQt6 import QtCore, QtGui, QtSvg, QtWidgets

from .. import docs_url, io, lib

#: Modes of the theme; "System" follows the operating system and
#: "Native" is the platform's own style.
MODES = ("System", "Light", "Dark", "Native")
#: Spacing of the controls.
DENSITIES = ("Comfortable", "Compact")
#: How the toolbars of Render and Localize show their buttons.
TOOLBAR_STYLES = {
    "Icons": QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly,
    "Icons and text": QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon,
    "Text": QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly,
    "Hidden": None,
}
#: Folder of the icons, ``<name>.svg``, see ``icon``.
ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
#: Preset accent colors.
ACCENTS = {
    "Blue": "#2A78D6",
    "Purple": "#6A55C2",
    "Green": "#1F9468",
    "Orange": "#E0662F",
    "Red": "#D9443F",
    "Pink": "#D9628F",
    "Gray": "#6E6D68",
}
#: Range of the font size in percent of the system's.
FONT_SCALE_RANGE = (80, 150)

#: Colors of the light and the dark theme, warm grays like those of the
#: "Light" and "Dark" chart themes (``plot_style.THEMES``), so that
#: charts sit well in the windows. In the dark theme, input fields are
#: lighter than the window so that they stand out without a frame.
NEUTRALS = {
    "Light": {
        "window": "#f3f3f1",
        "base": "#fcfcfb",
        "alternate": "#efeeeb",
        "button": "#fbfbfa",
        "hover": "#ebeae7",
        "pressed": "#dfded9",
        "text": "#1b1b19",
        "text_muted": "#62615d",
        "text_disabled": "#a3a29d",
        "border": "#cfcec9",
        "light": "#ffffff",
        "midlight": "#e8e7e3",
        "mid": "#bdbcb7",
        "dark": "#8c8b86",
        "shadow": "#4a4a46",
        "success": "#1f9468",
        "warning": "#c48600",
        "danger": "#d9443f",
    },
    "Dark": {
        "window": "#1f1f1e",
        "base": "#292927",
        "alternate": "#30302e",
        "button": "#363633",
        "hover": "#40403d",
        "pressed": "#4a4a46",
        "text": "#ececea",
        "text_muted": "#a3a29b",
        "text_disabled": "#6b6a65",
        "border": "#52524d",
        "light": "#585852",
        "midlight": "#3a3a37",
        "mid": "#5c5c56",
        "dark": "#151514",
        "shadow": "#000000",
        "success": "#3cba87",
        "warning": "#e3a72a",
        "danger": "#ec6560",
    },
}
#: Button states of ``set_button_state`` -> color of ``NEUTRALS``.
BUTTON_STATES = {"ok": "success", "warning": "warning", "danger": "danger"}

#: Per density: layout margins of windows and of other widgets, layout
#: spacing, padding of push buttons (vertical, horizontal) and of tool
#: buttons, corner radius (all in pixels). None keeps Fusion's value.
_DENSITY_METRICS = {
    "Comfortable": {
        "window_margin": None,
        "child_margin": None,
        "spacing": None,
        "button_padding": (4, 12),
        "tool_padding": 3,
        "radius": 5,
    },
    "Compact": {
        "window_margin": 6,
        "child_margin": 4,
        "spacing": 4,
        "button_padding": (2, 8),
        "tool_padding": 1,
        "radius": 4,
    },
}


@dataclass(frozen=True)
class Appearance:
    """Look of the windows.

    Attributes
    ----------
    mode : str
        One of ``MODES``.
    accent : str
        Accent color, a hexadecimal code.
    font_scale : int
        Font size in percent of the system's, within
        ``FONT_SCALE_RANGE``.
    density : str
        One of ``DENSITIES``.
    toolbar : str
        One of ``TOOLBAR_STYLES``.
    """

    mode: str = "System"
    accent: str = ACCENTS["Blue"]
    font_scale: int = 100
    density: str = "Comfortable"
    toolbar: str = "Icons and text"

    @classmethod
    def from_settings(cls, settings: dict | None) -> Appearance:
        """Build the appearance from the user settings, ignoring
        unknown keys and invalid values.

        Parameters
        ----------
        settings : dict or None
            Saved ``Appearance`` fields, see ``to_settings``.

        Returns
        -------
        appearance : Appearance
            The appearance; missing or invalid fields take their
            default.
        """
        default = cls()
        if not isinstance(settings, dict):
            return default
        values = {}
        for field in fields(cls):
            value = settings.get(field.name)
            if isinstance(getattr(default, field.name), int):
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    values[field.name] = int(round(value))
            elif isinstance(value, str):
                values[field.name] = value
        appearance = replace(default, **values)
        if appearance.mode not in MODES:
            appearance = replace(appearance, mode=default.mode)
        if appearance.toolbar not in TOOLBAR_STYLES:
            appearance = replace(appearance, toolbar=default.toolbar)
        if appearance.density not in DENSITIES:
            appearance = replace(appearance, density=default.density)
        accent = QtGui.QColor(appearance.accent)
        if accent.isValid():
            appearance = replace(appearance, accent=accent.name().upper())
        else:
            appearance = replace(appearance, accent=default.accent)
        low, high = FONT_SCALE_RANGE
        return replace(
            appearance, font_scale=min(high, max(low, appearance.font_scale))
        )

    def to_settings(self) -> dict:
        """The appearance as a plain dict for the user settings."""
        return asdict(self)


def _blend(color: str, other: str, t: float) -> QtGui.QColor:
    """``color`` mixed with the fraction ``t`` of ``other``."""
    a, b = QtGui.QColor(color), QtGui.QColor(other)
    return QtGui.QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
    )


def luminance(color: QtGui.QColor | str) -> float:
    """Relative luminance of ``color`` as defined by WCAG 2."""

    def channel(value: float) -> float:
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    color = QtGui.QColor(color)
    return (
        0.2126 * channel(color.redF())
        + 0.7152 * channel(color.greenF())
        + 0.0722 * channel(color.blueF())
    )


def contrast_ratio(
    color: QtGui.QColor | str, other: QtGui.QColor | str
) -> float:
    """WCAG 2 contrast ratio of two colors, from 1 to 21."""
    high, low = sorted((luminance(color), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _on_color(color: str) -> str:
    """White text on ``color`` if it reaches the contrast that WCAG 2
    asks of controls (3:1), as on most accent colors, else black."""
    if contrast_ratio(color, "#ffffff") >= 3:
        return "#ffffff"
    return "#000000"


def build_palette(appearance: Appearance, dark: bool) -> QtGui.QPalette:
    """The palette of the light or dark theme with the accent color of
    ``appearance``. Every role of every color group is set, so that
    nothing of the platform's palette shows through.

    Parameters
    ----------
    appearance : Appearance
        Gives the accent color.
    dark : bool
        Whether the dark theme is built.

    Returns
    -------
    palette : QtGui.QPalette
        The palette.
    """
    colors = NEUTRALS["Dark" if dark else "Light"]
    accent = QtGui.QColor(appearance.accent)
    link = accent.lighter(140) if dark and luminance(accent) < 0.2 else accent
    Role = QtGui.QPalette.ColorRole
    roles = {
        Role.Window: colors["window"],
        Role.WindowText: colors["text"],
        Role.Base: colors["base"],
        Role.AlternateBase: colors["alternate"],
        Role.ToolTipBase: colors["base"],
        Role.ToolTipText: colors["text"],
        Role.PlaceholderText: colors["text_muted"],
        Role.Text: colors["text"],
        Role.Button: colors["button"],
        Role.ButtonText: colors["text"],
        Role.BrightText: colors["danger"],
        Role.Light: colors["light"],
        Role.Midlight: colors["midlight"],
        Role.Mid: colors["mid"],
        Role.Dark: colors["dark"],
        Role.Shadow: colors["shadow"],
        Role.Highlight: accent,
        Role.HighlightedText: _on_color(appearance.accent),
        Role.Link: link,
        Role.LinkVisited: link.darker(120),
    }
    # Qt 6.6 and later
    if hasattr(Role, "Accent"):
        roles[Role.Accent] = accent
    palette = QtGui.QPalette()
    Group = QtGui.QPalette.ColorGroup
    for group in (Group.Active, Group.Inactive, Group.Disabled):
        for role, color in roles.items():
            palette.setColor(group, role, QtGui.QColor(color))
    for role in (Role.WindowText, Role.Text, Role.ButtonText):
        palette.setColor(
            Group.Disabled, role, QtGui.QColor(colors["text_disabled"])
        )
    palette.setColor(
        Group.Disabled, Role.Highlight, QtGui.QColor(colors["mid"])
    )
    palette.setColor(
        Group.Disabled, Role.HighlightedText, QtGui.QColor(colors["text"])
    )
    palette.setColor(Group.Disabled, Role.Base, QtGui.QColor(colors["window"]))
    palette.setColor(
        Group.Disabled, Role.Button, QtGui.QColor(colors["window"])
    )
    return palette


def _state_rules(dark: bool) -> str:
    """Stylesheet rules of the button states of ``set_button_state``."""
    colors = NEUTRALS["Dark" if dark else "Light"]
    rules = []
    for state, key in BUTTON_STATES.items():
        color = colors[key]
        soft = _blend(colors["button"], color, 0.3 if dark else 0.22).name()
        rules.append(
            f'QPushButton[state="{state}"] {{\n'
            f"    background-color: {soft};\n"
            f"    border: 1px solid {color};\n"
            f"    color: {colors['text']};\n"
            "}"
        )
    return "\n".join(rules)


def stylesheet(appearance: Appearance, dark: bool) -> str:
    """The application stylesheet of ``appearance``: buttons, tool
    buttons, group boxes and the button states of ``set_button_state``.
    In the "Native" mode, only the button states.

    Parameters
    ----------
    appearance : Appearance
        The appearance.
    dark : bool
        Whether the dark theme is used.

    Returns
    -------
    stylesheet : str
        The stylesheet.
    """
    if appearance.mode == "Native":
        return _state_rules(dark)
    colors = NEUTRALS["Dark" if dark else "Light"]
    metrics = _DENSITY_METRICS[appearance.density]
    pad_v, pad_h = metrics["button_padding"]
    radius = metrics["radius"]
    accent = appearance.accent
    on_accent = _on_color(accent)
    accent_hover = QtGui.QColor(accent).lighter(110).name()
    accent_pressed = QtGui.QColor(accent).darker(115).name()
    accent_soft = _blend(colors["button"], accent, 0.3 if dark else 0.2).name()
    return f"""
QPushButton {{
    background-color: {colors["button"]};
    color: {colors["text"]};
    border: 1px solid {colors["border"]};
    border-radius: {radius}px;
    padding: {pad_v}px {pad_h}px;
}}
QPushButton:hover {{
    background-color: {colors["hover"]};
}}
QPushButton:pressed {{
    background-color: {colors["pressed"]};
}}
QPushButton:focus {{
    border-color: {accent};
}}
QPushButton:checked {{
    background-color: {accent_soft};
    border-color: {accent};
}}
QPushButton:default {{
    background-color: {accent};
    color: {on_accent};
    border-color: {accent_pressed};
}}
QPushButton:default:hover {{
    background-color: {accent_hover};
}}
QPushButton:default:pressed {{
    background-color: {accent_pressed};
}}
QPushButton:flat {{
    background-color: transparent;
    border-color: transparent;
}}
QPushButton:disabled {{
    background-color: {colors["window"]};
    color: {colors["text_disabled"]};
    border-color: {colors["midlight"]};
}}
QDialogButtonBox QPushButton {{
    min-width: 4.5em;
}}
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {radius}px;
    padding: {metrics["tool_padding"]}px;
}}
QToolButton:hover {{
    background-color: {colors["hover"]};
    border-color: {colors["border"]};
}}
QToolButton:pressed {{
    background-color: {colors["pressed"]};
}}
QToolButton:checked {{
    background-color: {accent_soft};
    border-color: {accent};
}}
QToolButton:disabled {{
    color: {colors["text_disabled"]};
}}
QGroupBox {{
    border: 1px solid {colors["border"]};
    border-radius: {radius + 1}px;
    margin-top: 0.6em;
    padding-top: 0.5em;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0px 3px;
}}
QGroupBox:disabled {{
    color: {colors["text_disabled"]};
}}
{_state_rules(dark)}
"""


class _PicassoStyle(QtWidgets.QProxyStyle):
    """Fusion with the layout margins and spacing of a density.

    A stylesheet cannot change the spacing of layouts, so it is set
    here.

    Parameters
    ----------
    density : str
        One of ``DENSITIES``.
    """

    _MARGINS = (
        QtWidgets.QStyle.PixelMetric.PM_LayoutLeftMargin,
        QtWidgets.QStyle.PixelMetric.PM_LayoutTopMargin,
        QtWidgets.QStyle.PixelMetric.PM_LayoutRightMargin,
        QtWidgets.QStyle.PixelMetric.PM_LayoutBottomMargin,
    )
    _SPACINGS = (
        QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing,
        QtWidgets.QStyle.PixelMetric.PM_LayoutVerticalSpacing,
    )

    def __init__(self, density: str) -> None:
        super().__init__("Fusion")
        self.density = density
        metrics = _DENSITY_METRICS[density]
        self._window_margin = metrics["window_margin"]
        self._child_margin = metrics["child_margin"]
        self._spacing = metrics["spacing"]

    def pixelMetric(self, metric, option=None, widget=None):
        if metric in self._MARGINS:
            is_window = widget is not None and widget.isWindow()
            margin = self._window_margin if is_window else self._child_margin
            if margin is not None:
                return margin
        elif metric in self._SPACINGS and self._spacing is not None:
            return self._spacing
        return super().pixelMetric(metric, option, widget)


#: What the application looked like before the theme was first applied,
#: to return to in the "Native" mode, and the applied appearance.
_state = {
    "original": None,
    "applied": None,
    "listening": False,
}


def is_dark(appearance: Appearance, app: QtGui.QGuiApplication) -> bool:
    """Whether ``appearance`` is shown in the dark theme: in the
    "System" and "Native" modes, if the operating system's appearance
    is dark."""
    if appearance.mode in ("Light", "Dark"):
        return appearance.mode == "Dark"
    return app.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark


def _scaled_font(font: QtGui.QFont, scale: int) -> QtGui.QFont:
    """A copy of ``font`` with ``scale`` percent of its size."""
    font = QtGui.QFont(font)
    if font.pointSizeF() > 0:
        font.setPointSizeF(round(font.pointSizeF() * scale / 100, 1))
    elif font.pixelSize() > 0:
        font.setPixelSize(max(1, round(font.pixelSize() * scale / 100)))
    return font


def _on_color_scheme_changed(_scheme: QtCore.Qt.ColorScheme) -> None:
    """Follow the operating system to its new appearance."""
    app = QtWidgets.QApplication.instance()
    applied = _state["applied"]
    if app is not None and applied is not None:
        if applied.mode in ("System", "Native"):
            apply(app, applied)


def apply(app: QtWidgets.QApplication, appearance: Appearance) -> None:
    """Apply ``appearance`` to the application and its open windows.

    The first call remembers the platform's style, font and stylesheet;
    the "Native" mode returns to them. In the "System" mode the theme
    follows later changes of the operating system's appearance. Each
    applied appearance is announced through ``hub().changed``, e.g., to
    the charts that follow the windows' theme.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application.
    appearance : Appearance
        The appearance.
    """
    if _state["original"] is None:
        _state["original"] = {
            "style": app.style().name(),
            "font": QtGui.QFont(app.font()),
            "stylesheet": app.styleSheet(),
        }
    original = _state["original"]
    was_themed = _state["applied"] is not None and (
        _state["applied"].mode != "Native"
    )
    _state["applied"] = appearance
    dark = is_dark(appearance, app)
    if appearance.mode == "Native":
        if was_themed:
            style = QtWidgets.QStyleFactory.create(original["style"])
            if style is not None:
                app.setStyle(style)
            # an empty palette resolves to the platform's, which also
            # follows its later changes
            app.setPalette(QtGui.QPalette())
            app.setFont(original["font"])
        app.setStyleSheet(
            original["stylesheet"] + stylesheet(appearance, dark)
        )
    else:
        app.setStyle(_PicassoStyle(appearance.density))
        app.setPalette(build_palette(appearance, dark))
        app.setFont(_scaled_font(original["font"], appearance.font_scale))
        app.setStyleSheet(stylesheet(appearance, dark))
    for window in app.topLevelWidgets():
        for toolbar in window.findChildren(QtWidgets.QToolBar):
            if toolbar.property(_TOOLBAR_PROPERTY):
                _style_toolbar(toolbar, appearance)
    _listen(app)
    hub().changed.emit(appearance)


def _listen(app: QtWidgets.QApplication) -> None:
    """Follow changes of the operating system's appearance."""
    if not _state["listening"]:
        app.styleHints().colorSchemeChanged.connect(_on_color_scheme_changed)
        _state["listening"] = True


def applied() -> Appearance | None:
    """The appearance applied last, or None if none was applied."""
    return _state["applied"]


class _TintedIconEngine(QtGui.QIconEngine):
    """Draws a single-color icon in the colors of the palette: the text
    color, dimmed when disabled, and the accent color when checked
    (e.g., the active tool). The colors are read whenever the icon is
    drawn, so that it follows the theme. Only the shape of the image
    (its opacity) is kept.

    Parameters
    ----------
    path : str
        The image: an SVG file or a raster image with a transparent
        background (e.g., ``.ico`` or ``.png``).
    role : QtGui.QPalette.ColorRole or None, optional
        Palette color of the enabled, unchecked icon instead of the text
        color, e.g., ``HighlightedText`` on an accent background.
        Default None.
    """

    def __init__(
        self, path: str, role: QtGui.QPalette.ColorRole | None = None
    ) -> None:
        super().__init__()
        self._path = path
        self._role = role
        if path.lower().endswith(".svg"):
            self._renderer = QtSvg.QSvgRenderer(path)
            self._image = None
        else:
            self._renderer = None
            self._image = QtGui.QIcon(path)
        self._cache = {}

    def clone(self) -> QtGui.QIconEngine:
        return _TintedIconEngine(self._path, self._role)

    def is_valid(self) -> bool:
        """Whether the file could be read."""
        if self._renderer is not None:
            return self._renderer.isValid()
        return not self._image.isNull()

    def color(
        self, mode: QtGui.QIcon.Mode, state: QtGui.QIcon.State
    ) -> QtGui.QColor:
        """The color of the icon in ``mode`` and ``state``."""
        palette = QtWidgets.QApplication.palette()
        Role = QtGui.QPalette.ColorRole
        if mode == QtGui.QIcon.Mode.Disabled:
            return palette.color(
                QtGui.QPalette.ColorGroup.Disabled, Role.ButtonText
            )
        if mode == QtGui.QIcon.Mode.Selected:
            return palette.color(Role.HighlightedText)
        if state == QtGui.QIcon.State.On:
            return palette.color(Role.Highlight)
        return palette.color(self._role or Role.ButtonText)

    def scaledPixmap(self, size, mode, state, scale):
        color = self.color(mode, state)
        key = (size.width(), size.height(), scale, mode, state, color.rgba())
        pixmap = self._cache.get(key)
        if pixmap is None:
            width = max(1, round(size.width() * scale))
            height = max(1, round(size.height() * scale))
            image = QtGui.QImage(
                width, height, QtGui.QImage.Format.Format_ARGB32_Premultiplied
            )
            image.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(image)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            if self._renderer is not None:
                self._renderer.render(
                    painter, QtCore.QRectF(0, 0, width, height)
                )
            else:
                self._image.paint(painter, QtCore.QRect(0, 0, width, height))
            # keep the shape, replace its color
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(image.rect(), color)
            painter.end()
            pixmap = QtGui.QPixmap.fromImage(image)
            pixmap.setDevicePixelRatio(scale)
            self._cache[key] = pixmap
        return pixmap

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, 1.0)

    def paint(self, painter, rect, mode, state):
        device = painter.device()
        scale = device.devicePixelRatioF() if device is not None else 1.0
        painter.drawPixmap(
            rect, self.scaledPixmap(rect.size(), mode, state, scale)
        )


def icon(
    name: str, role: QtGui.QPalette.ColorRole | None = None
) -> QtGui.QIcon:
    """The icon ``ICONS_DIR/<name>.svg`` in the colors of the theme.

    The SVG is drawn in a single color: the text color of the palette,
    dimmed when disabled and the accent color when checked; it changes
    with the theme. Any color in the file is ignored. Without an SVG, an
    image ``<name>.ico`` or ``<name>.png`` with a transparent background
    is used the same way, e.g., an application icon.

    Parameters
    ----------
    name : str
        File name of the icon without the extension, e.g., "open".
    role : QtGui.QPalette.ColorRole or None, optional
        Palette color of the enabled, unchecked icon instead of the text
        color, e.g., ``HighlightedText`` on an accent background.
        Default None.

    Returns
    -------
    icon : QtGui.QIcon
        The icon, or a null icon if the file is missing or invalid, in
        which case buttons show their text instead.
    """
    for extension in (".svg", ".ico", ".png"):
        path = os.path.join(ICONS_DIR, name + extension)
        if os.path.isfile(path):
            break
    else:
        return QtGui.QIcon()
    engine = _TintedIconEngine(path, role)
    if not engine.is_valid():
        return QtGui.QIcon()
    return QtGui.QIcon(engine)


#: Dynamic property that marks the toolbars styled by ``apply``.
_TOOLBAR_PROPERTY = "picassoToolbar"


def _style_toolbar(
    toolbar: QtWidgets.QToolBar, appearance: Appearance
) -> None:
    """Show ``toolbar`` as set in ``appearance``."""
    style = TOOLBAR_STYLES.get(appearance.toolbar)
    toolbar.setVisible(style is not None)
    if style is not None:
        toolbar.setToolButtonStyle(style)
    compact = appearance.density == "Compact" and appearance.mode != "Native"
    size = 16 if compact else 20
    toolbar.setIconSize(QtCore.QSize(size, size))


def _stripped(text: str) -> str:
    """``text`` as Qt shows it on a button: without the ellipsis and
    the mnemonic ampersands."""
    return re.sub(r"&(.)", r"\1", text.replace("...", ""))


def add_toolbar(
    window: QtWidgets.QMainWindow,
    title: str,
    items: Iterable[tuple | None],
) -> QtWidgets.QToolBar:
    """Add a toolbar of existing actions (e.g., those of the menus) to
    ``window``. The actions get their icon (see ``icon``), which the
    menus show too, and, unless they have their own, a tooltip with the
    shortcut. The toolbar is shown as set in the appearance
    (``Appearance.toolbar``).

    Parameters
    ----------
    window : QtWidgets.QMainWindow
        The window.
    title : str
        Name of the toolbar, shown in the window's context menu.
    items : iterable of tuple or None
        ``(action, icon_name)`` or ``(action, icon_name, label)``,
        where ``label`` is a short text for the button, or None for a
        separator.

    Returns
    -------
    toolbar : QtWidgets.QToolBar
        The toolbar.
    """
    toolbar = window.addToolBar(title)
    toolbar.setObjectName(title)
    toolbar.setProperty(_TOOLBAR_PROPERTY, True)
    for item in items:
        if item is None:
            toolbar.addSeparator()
            continue
        action, name, *label = item
        has_own_tooltip = action.toolTip() != _stripped(action.text())
        action.setIcon(icon(name))
        if label:
            action.setIconText(label[0])
        shortcut = action.shortcut().toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        if not has_own_tooltip:
            text = _stripped(action.text())
            action.setToolTip(f"{text} ({shortcut})" if shortcut else text)
        toolbar.addAction(action)
    _style_toolbar(toolbar, applied() or Appearance())
    return toolbar


def set_button_state(
    button: QtWidgets.QPushButton, state: str | None = None
) -> None:
    """Show ``button`` in the color of a state, e.g., green once the
    file it loads was loaded.

    Parameters
    ----------
    button : QtWidgets.QPushButton
        The button.
    state : str or None, optional
        One of ``BUTTON_STATES`` ("ok", "warning", "danger"), or None
        (default) for the normal look.
    """
    if state is not None and state not in BUTTON_STATES:
        raise ValueError(
            f"Unknown button state {state!r}, expected one of "
            f"{tuple(BUTTON_STATES)} or None."
        )
    button.setProperty("state", state or "")
    # dynamic properties are matched when the widget is polished
    style = button.style()
    style.unpolish(button)
    style.polish(button)
    button.update()


def current() -> Appearance:
    """The appearance saved in the user settings."""
    return Appearance.from_settings(io.load_user_settings()["Appearance"])


def set_current(appearance: Appearance) -> None:
    """Save ``appearance`` in the user settings and apply it to the
    running application, which announces it through ``hub().changed``."""
    settings = io.load_user_settings()
    settings["Appearance"] = appearance.to_settings()
    io.save_user_settings(settings)
    app = QtWidgets.QApplication.instance()
    if app is not None:
        apply(app, appearance)


class _Hub(QtCore.QObject):
    """Announces appearance changes to the open windows."""

    changed = QtCore.pyqtSignal(object)


_hub = None
_dialog = None


def hub() -> _Hub:
    """The object whose ``changed`` signal carries each new
    appearance."""
    global _hub
    if _hub is None:
        _hub = _Hub()
    return _hub


def show_dialog() -> AppearanceDialog:
    """Show the appearance dialog, creating it on first use. One dialog
    serves all windows; it opens with the saved appearance.

    Returns
    -------
    dialog : AppearanceDialog
        The dialog.
    """
    global _dialog
    if _dialog is None:
        _dialog = AppearanceDialog(current())
        # an open dialog must not keep the app running after the last
        # main window closed
        _dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_QuitOnClose, False)
        _dialog.appearanceChanged.connect(set_current)
    elif not _dialog.isVisible():
        _dialog.set_appearance(current(), notify=False)
    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()
    return _dialog


def add_menu_action(menu: QtWidgets.QMenu) -> QtGui.QAction:
    """Add "Appearance...", which opens the appearance dialog, to
    ``menu``.

    Parameters
    ----------
    menu : QtWidgets.QMenu
        The menu, usually File.

    Returns
    -------
    action : QtGui.QAction
        The added action.
    """
    action = menu.addAction("Appearance...")
    action.setIcon(icon("plot-settings"))
    action.triggered.connect(lambda: show_dialog())
    return action


class AppearanceDialog(lib.Dialog):
    """Edit the ``Appearance`` of the Picasso windows.

    Changes are applied immediately; ``appearanceChanged`` is emitted
    shortly after the last change so that dragging the font size does
    not restyle the windows for every step.

    Parameters
    ----------
    appearance : Appearance
        The appearance shown initially.
    parent : QWidget or None, optional
        Parent widget. Default None.
    """

    appearanceChanged = QtCore.pyqtSignal(object)

    DOCS_URL = docs_url("others.html#appearance")

    #: Delay between the last change and ``appearanceChanged`` (ms).
    DELAY = 250

    def __init__(
        self,
        appearance: Appearance,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        # imported here to keep the startup of the GUIs light:
        # overlay_style imports picasso.render
        from .overlay_style import ColorComboBox

        super().__init__(parent)
        self.setWindowTitle("Appearance")
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.DELAY)
        self._timer.timeout.connect(
            lambda: self.appearanceChanged.emit(self.appearance())
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

        self.form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.mode.setItemData(
            0,
            "Light or dark like the operating system",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.mode.setItemData(
            3,
            "The platform's own style, as in earlier Picasso versions",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.form.addRow("Theme:", self.mode)
        self.accent = ColorComboBox(presets=ACCENTS)
        self.accent.setToolTip(
            "Color of selections, checked and default buttons, sliders "
            "and focus frames."
        )
        self.form.addRow("Accent color:", self.accent)
        self.font_scale = QtWidgets.QSpinBox()
        self.font_scale.setRange(*FONT_SCALE_RANGE)
        self.font_scale.setSingleStep(5)
        self.font_scale.setSuffix(" %")
        self.font_scale.setToolTip(
            "Size of the text in percent of the system's.\n"
            "Open windows keep their size; reopen them to fit."
        )
        self.form.addRow("Font size:", self.font_scale)
        self.density = QtWidgets.QComboBox()
        self.density.addItems(DENSITIES)
        self.density.setToolTip("Compact fits more controls on small screens.")
        self.form.addRow("Density:", self.density)
        self.toolbar = QtWidgets.QComboBox()
        self.toolbar.addItems(TOOLBAR_STYLES)
        self.toolbar.setToolTip(
            "Buttons of the toolbars of Render and Localize.\n"
            "Without an icon, a button shows its text."
        )
        self.form.addRow("Toolbar:", self.toolbar)
        layout.addLayout(self.form)

        buttons = QtWidgets.QDialogButtonBox()
        reset = buttons.addButton(
            "Restore defaults",
            QtWidgets.QDialogButtonBox.ButtonRole.ResetRole,
        )
        reset.clicked.connect(lambda: self.set_appearance(Appearance()))
        close = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        close.clicked.connect(self.close)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(lib.HelpButton(self.DOCS_URL))
        button_row.addWidget(buttons)
        layout.addLayout(button_row)

        self.set_appearance(appearance, notify=False)
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        self.accent.colorChanged.connect(self._timer.start)
        self.font_scale.valueChanged.connect(self._timer.start)
        self.density.currentIndexChanged.connect(self._timer.start)
        self.toolbar.currentIndexChanged.connect(self._timer.start)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # apply a change still waiting for the delay now, so that the
        # timer does not fire after the dialog is gone
        if self._timer.isActive():
            self._timer.stop()
            self.appearanceChanged.emit(self.appearance())
        super().closeEvent(event)

    def _on_mode_changed(self) -> None:
        self._update_visibility()
        self._timer.start()

    def _update_visibility(self) -> None:
        """Show only the settings that the selected mode uses."""
        themed = self.mode.currentText() != "Native"
        for widget in (self.accent, self.font_scale, self.density):
            self.form.setRowVisible(widget, themed)

    def appearance(self) -> Appearance:
        """The appearance set in the dialog."""
        accent = self.accent.value()
        return Appearance(
            mode=self.mode.currentText(),
            accent=ACCENTS.get(accent, accent),
            font_scale=self.font_scale.value(),
            density=self.density.currentText(),
            toolbar=self.toolbar.currentText(),
        )

    def set_appearance(
        self, appearance: Appearance, notify: bool = True
    ) -> None:
        """Show ``appearance`` in the dialog.

        Parameters
        ----------
        appearance : Appearance
            The appearance.
        notify : bool, optional
            Whether to emit ``appearanceChanged`` (once, after the
            delay) if ``appearance`` differs from the shown one.
            Default True.
        """
        self.mode.setCurrentText(appearance.mode)
        names = {code.upper(): name for name, code in ACCENTS.items()}
        accent = names.get(appearance.accent.upper(), appearance.accent)
        if not self.accent.set_value(accent):
            self.accent.set_value("Blue")
        self.font_scale.setValue(appearance.font_scale)
        self.density.setCurrentText(appearance.density)
        self.toolbar.setCurrentText(appearance.toolbar)
        self._update_visibility()
        if not notify:
            self._timer.stop()
