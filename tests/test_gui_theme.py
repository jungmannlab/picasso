"""The theme of the Picasso windows.

``picasso.gui.theme`` gives every GUI the same look: Fusion with a
light or dark palette and an accent color, a stylesheet for buttons and
group boxes, a font size and a density. These tests cover reading the
appearance from the user settings (including invalid values), the
contrast of the palettes, applying the theme and returning to the
platform's look, the button states and the appearance dialog.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from PyQt6 import QtGui, QtWidgets

from picasso import io
from picasso.gui import theme
from picasso.gui.theme import ACCENTS, NEUTRALS, Appearance

Role = QtGui.QPalette.ColorRole
Group = QtGui.QPalette.ColorGroup


@pytest.fixture
def home(tmp_path, monkeypatch):
    """User settings in ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def fresh_hub(monkeypatch):
    """Each test gets its own hub and dialog."""
    monkeypatch.setattr(theme, "_hub", None)
    monkeypatch.setattr(theme, "_dialog", None)
    yield
    if theme._dialog is not None:
        theme._dialog.close()


# -- settings ---------------------------------------------------------


def test_defaults():
    appearance = Appearance()
    assert appearance.mode == "System"
    assert appearance.accent == ACCENTS["Blue"]
    assert appearance.font_scale == 100
    assert appearance.density == "Comfortable"


@pytest.mark.parametrize("settings", [None, {}, "Dark", [1, 2]])
def test_missing_settings_give_the_defaults(settings):
    assert Appearance.from_settings(settings) == Appearance()


def test_settings_round_trip():
    appearance = Appearance(
        mode="Dark", accent="#123456", font_scale=120, density="Compact"
    )
    assert Appearance.from_settings(appearance.to_settings()) == appearance


def test_invalid_values_take_their_default():
    appearance = Appearance.from_settings(
        {
            "mode": "Neon",
            "accent": "not a color",
            "font_scale": "big",
            "density": 3,
            "unknown": True,
        }
    )
    assert appearance == Appearance()


def test_font_scale_is_clamped_and_accent_normalized():
    low, high = theme.FONT_SCALE_RANGE
    assert Appearance.from_settings({"font_scale": 10}).font_scale == low
    assert Appearance.from_settings({"font_scale": 999}).font_scale == high
    assert Appearance.from_settings({"font_scale": 112.6}).font_scale == 113
    assert Appearance.from_settings({"font_scale": True}).font_scale == 100
    # color names and lowercase codes are stored as uppercase codes
    assert Appearance.from_settings({"accent": "red"}).accent == "#FF0000"
    assert Appearance.from_settings({"accent": "#abcdef"}).accent == (
        "#ABCDEF"
    )


def test_current_reads_the_user_settings(home):
    assert theme.current() == Appearance()
    settings = io.load_user_settings()
    settings["Appearance"] = {"mode": "Light", "density": "Compact"}
    io.save_user_settings(settings)
    assert theme.current() == Appearance(mode="Light", density="Compact")


def test_set_current_saves_applies_and_announces(home, restore_theme):
    received = []
    theme.hub().changed.connect(received.append)
    appearance = Appearance(mode="Dark", accent=ACCENTS["Green"])
    theme.set_current(appearance)
    assert theme.current() == appearance
    assert theme.applied() == appearance
    assert received == [appearance]
    window = restore_theme.palette().color(Role.Window).name()
    assert window == NEUTRALS["Dark"]["window"]


# -- palette ----------------------------------------------------------


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("accent", list(ACCENTS.values()) + ["#FFFF00"])
def test_palettes_are_readable(dark, accent):
    """Text reaches WCAG AA on its background; text on the accent
    reaches the contrast asked of controls."""
    palette = theme.build_palette(Appearance(accent=accent), dark)

    def ratio(fg, bg, group=Group.Active):
        return theme.contrast_ratio(
            palette.color(group, fg), palette.color(group, bg)
        )

    assert ratio(Role.Text, Role.Base) >= 7
    assert ratio(Role.WindowText, Role.Window) >= 7
    assert ratio(Role.ButtonText, Role.Button) >= 7
    assert ratio(Role.PlaceholderText, Role.Base) >= 4.5
    assert ratio(Role.ToolTipText, Role.ToolTipBase) >= 7
    assert ratio(Role.HighlightedText, Role.Highlight) >= 3
    # disabled text is dimmer than enabled text, but still visible
    disabled = ratio(Role.WindowText, Role.Window, Group.Disabled)
    assert 1.5 <= disabled < ratio(Role.WindowText, Role.Window)


def test_white_text_on_the_default_accent():
    palette = theme.build_palette(Appearance(), dark=False)
    assert palette.color(Role.HighlightedText).name() == "#ffffff"
    yellow = theme.build_palette(Appearance(accent="#FFFF00"), dark=False)
    assert yellow.color(Role.HighlightedText).name() == "#000000"


def test_every_color_group_is_set():
    """Nothing of the platform's palette may show through."""
    palette = theme.build_palette(Appearance(), dark=True)
    for group in (Group.Active, Group.Inactive, Group.Disabled):
        assert palette.color(group, Role.Window).name() == (
            NEUTRALS["Dark"]["window"]
        )
    assert palette.color(Role.Highlight).name().upper() == ACCENTS["Blue"]


def test_dark_mode_is_resolved_from_the_mode(qapp):
    assert theme.is_dark(Appearance(mode="Dark"), qapp)
    assert not theme.is_dark(Appearance(mode="Light"), qapp)


# -- stylesheet and style ---------------------------------------------


def test_stylesheet_uses_accent_and_density():
    accent = "#123456"
    comfortable = theme.stylesheet(Appearance(accent=accent), dark=False)
    compact = theme.stylesheet(
        Appearance(accent=accent, density="Compact"), dark=False
    )
    assert accent in comfortable
    assert "QGroupBox" in comfortable
    assert "padding: 4px 12px" in comfortable
    assert "padding: 2px 8px" in compact
    assert 'QPushButton[state="ok"]' in comfortable


def test_native_stylesheet_has_only_the_button_states():
    sheet = theme.stylesheet(Appearance(mode="Native"), dark=False)
    assert 'QPushButton[state="ok"]' in sheet
    assert "QGroupBox" not in sheet
    assert "QToolButton" not in sheet


def test_compact_density_tightens_layouts(qapp):
    metric = QtWidgets.QStyle.PixelMetric
    compact = theme._PicassoStyle("Compact")
    comfortable = theme._PicassoStyle("Comfortable")
    fusion = QtWidgets.QStyleFactory.create("Fusion")
    parent = QtWidgets.QWidget()
    child = QtWidgets.QWidget(parent)
    assert compact.pixelMetric(metric.PM_LayoutHorizontalSpacing) == 4
    assert compact.pixelMetric(metric.PM_LayoutLeftMargin, None, child) == 4
    for m in (metric.PM_LayoutHorizontalSpacing, metric.PM_LayoutTopMargin):
        assert comfortable.pixelMetric(m) == fusion.pixelMetric(m)


# -- applying ---------------------------------------------------------


def test_apply_sets_style_palette_font_and_stylesheet(restore_theme):
    app = restore_theme
    base_size = app.font().pointSizeF()
    appearance = Appearance(mode="Dark", font_scale=130, density="Compact")
    theme.apply(app, appearance)
    assert app.palette().color(Role.Window).name() == (
        NEUTRALS["Dark"]["window"]
    )
    assert app.font().pointSizeF() == pytest.approx(round(base_size * 1.3, 1))
    assert app.styleSheet() == theme.stylesheet(appearance, dark=True)
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    assert layout.spacing() == 4
    # a second appearance scales the original font, not the scaled one
    theme.apply(app, Appearance(mode="Light", font_scale=100))
    assert app.font().pointSizeF() == pytest.approx(base_size)


def test_native_returns_to_the_platform_look(restore_theme):
    app = restore_theme
    style = app.style().name()
    font = QtGui.QFont(app.font())
    window = app.palette().color(Role.Window).name()
    theme.apply(app, Appearance(mode="Dark", font_scale=150))
    assert app.palette().color(Role.Window).name() != window
    theme.apply(app, Appearance(mode="Native"))
    assert app.font() == font
    assert app.palette().color(Role.Window).name() == window
    assert app.styleSheet() == theme.stylesheet(
        Appearance(mode="Native"), dark=False
    )
    # with a stylesheet, Qt wraps the style; look at the style itself
    app.setStyleSheet("")
    assert app.style().name() == style
    assert not isinstance(app.style(), theme._PicassoStyle)


def test_native_first_leaves_the_platform_style(restore_theme):
    app = restore_theme
    style = app.style().name()
    theme.apply(app, Appearance(mode="Native"))
    app.setStyleSheet("")
    assert app.style().name() == style
    assert not isinstance(app.style(), theme._PicassoStyle)


# -- button states ----------------------------------------------------


def test_set_button_state(qt_offscreen, restore_theme):
    theme.apply(restore_theme, Appearance(mode="Light"))
    button = QtWidgets.QPushButton("Load")
    theme.set_button_state(button, "ok")
    assert button.property("state") == "ok"
    theme.set_button_state(button, None)
    assert button.property("state") == ""
    with pytest.raises(ValueError, match="Unknown button state"):
        theme.set_button_state(button, "great")


# -- dialog -----------------------------------------------------------


def test_dialog_shows_and_returns_the_appearance(qt_offscreen):
    appearance = Appearance(
        mode="Dark",
        accent=ACCENTS["Purple"],
        font_scale=110,
        density="Compact",
    )
    dialog = theme.AppearanceDialog(appearance)
    assert dialog.appearance() == appearance
    # preset accents are shown by name
    assert dialog.accent.value() == "Purple"
    custom = Appearance(accent="#123456")
    dialog.set_appearance(custom)
    assert dialog.appearance() == custom


def test_dialog_links_to_its_documentation(qt_offscreen):
    from picasso import lib

    dialog = theme.AppearanceDialog(Appearance())
    help_buttons = dialog.findChildren(lib.HelpButton)
    assert len(help_buttons) == 1
    assert help_buttons[0].help_url == dialog.DOCS_URL
    assert dialog.DOCS_URL.endswith("others.html#appearance")


def test_dialog_hides_settings_native_does_not_use(qt_offscreen):
    dialog = theme.AppearanceDialog(Appearance(mode="Native"))
    for widget in (dialog.accent, dialog.font_scale, dialog.density):
        assert not dialog.form.isRowVisible(widget)
    dialog.mode.setCurrentText("Light")
    for widget in (dialog.accent, dialog.font_scale, dialog.density):
        assert dialog.form.isRowVisible(widget)


def test_dialog_emits_once_after_changes(qt_offscreen):
    dialog = theme.AppearanceDialog(Appearance())
    received = []
    dialog.appearanceChanged.connect(received.append)
    dialog.mode.setCurrentText("Dark")
    dialog.font_scale.setValue(120)
    assert dialog._timer.isActive()
    dialog._timer.timeout.emit()
    assert received == [Appearance(mode="Dark", font_scale=120)]


def test_menu_action_opens_the_one_dialog(qt_offscreen, home, restore_theme):
    menu = QtWidgets.QMenu()
    action = theme.add_menu_action(menu)
    assert action.text() == "Appearance..."
    action.trigger()
    dialog = theme._dialog
    assert dialog is not None and dialog.isVisible()
    action.trigger()
    assert theme._dialog is dialog
