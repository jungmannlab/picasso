"""The scale bars of Localize and SPINNA.

They share one look (``settings["ScaleBar"]``), set in a small dialog
opened from View > Scale bar settings... in Localize and from a button
next to each scale bar in SPINNA; whether the length is printed is set
for each place. These tests cover the shared look, the defaults of
each place, Localize's scale bar painted over its view and exported,
and SPINNA's mask view and structure preview.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PyQt6 import QtGui

from picasso import render
from picasso.gui import localize as gui_localize
from picasso.gui import scalebar_settings, spinna as gui_spinna
from picasso.gui.scalebar_settings import ScaleBarOptions

# fixtures of the Localize and SPINNA tests
from tests.test_gui_localize_dialogs import (  # noqa: F401
    loaded,
    shown,
    window as localize_window,
)
from tests.test_gui_spinna import (  # noqa: F401
    add_structure,
    dialogs,
    load_mask_locs,
    out,
    save_molecules,
    set_row,
    target_rows,
    window as spinna_window,
)


@pytest.fixture
def window(localize_window):  # noqa: F811
    """The Localize window, for the fixtures ``loaded`` and ``shown``."""
    return localize_window


@pytest.fixture
def mask_tab(spinna_window):  # noqa: F811
    tab = spinna_window.mask_generator_tab
    tab.mask_binsize_xy.setValue(100)
    tab.mask_blur_xy.setValue(100)
    return tab


@pytest.fixture
def structures_tab(spinna_window):  # noqa: F811
    return spinna_window.structures_tab


def _saved(tmp_path) -> dict:
    return yaml.safe_load((tmp_path / "settings.yaml").read_text())


def test_invalid_looks_fall_back_to_the_defaults():
    look = scalebar_settings.look_from_settings(
        {
            "Position": "Middle",
            "Margins (px)": [1],
            "Thickness (px)": -5,
            "Color": "not a color",
            "Bold": "yes",
            "Font": "",
        }
    )
    assert look == {
        **scalebar_settings.LOOK_DEFAULTS,
        "Thickness (px)": 1,  # clipped to the range
    }
    assert scalebar_settings.look_from_settings(None) == (
        scalebar_settings.LOOK_DEFAULTS
    )


def test_the_look_set_in_one_place_is_used_by_all(qt_offscreen, tmp_path):
    first = ScaleBarOptions("Localize", "Print scale bar length", True)
    second = ScaleBarOptions("SPINNA", "Print scale bar length (mask)", True)
    notified = []
    second.changed.connect(lambda: notified.append(True))
    dialog = first.show_dialog()
    other_dialog = second.show_dialog()
    dialog.style_widget.position.setCurrentText("Top left")
    dialog.style_widget.italic.setChecked(True)
    dialog.close()  # saves the change waiting for the delay

    kwargs = second.draw_kwargs()
    assert kwargs["position"] == "top left"
    assert kwargs["italic"] is True
    assert notified
    # an open dialog shows it, too
    assert other_dialog.style_widget.position.currentText() == "Top left"
    look = _saved(tmp_path)["ScaleBar"]
    assert look["Position"] == "Top left"
    assert look["Italic"] is True
    other_dialog.close()


def test_whether_the_length_is_printed_is_set_for_each_place(
    qt_offscreen, tmp_path
):
    localize = ScaleBarOptions("Localize", "Print scale bar length", True)
    mask = ScaleBarOptions("SPINNA", "Print scale bar length (mask)", True)
    dialog = localize.show_dialog()
    assert dialog.print_length.isChecked()
    dialog.print_length.setChecked(False)
    assert localize.draw_kwargs()["display_length"] is False
    assert mask.draw_kwargs()["display_length"] is True
    assert not dialog.style_widget.font_size.isEnabled()
    assert _saved(tmp_path)["Localize"]["Print scale bar length"] is False
    # remembered
    again = ScaleBarOptions("Localize", "Print scale bar length", True)
    assert again.print_length is False
    dialog.close()


def test_the_defaults_of_each_place(qt_offscreen, spinna_window):  # noqa: F811
    localize = gui_localize.Window()
    try:
        assert localize.scalebar_options.print_length is True
    finally:
        localize.close()
    mask = spinna_window.mask_generator_tab.scalebar_options
    structures = spinna_window.structures_tab.scalebar_options
    assert mask.print_length is True
    assert structures.print_length is False
    # the look of Render's default scale bar: white, bottom right
    kwargs = structures.draw_kwargs()
    assert kwargs["position"] == "bottom right"
    assert kwargs["margin"] == (35, 20)
    assert kwargs["display_height"] == 10
    assert kwargs["color"] == QtGui.QColor("white")


def test_localize_has_a_settings_entry_in_the_view_menu(shown):  # noqa: F811
    actions = {
        action.text(): action
        for action in shown.menuBar().actions()[1].menu().actions()
    }
    assert "Scale bar settings..." in actions
    actions["Scale bar settings..."].trigger()
    dialog = shown.scalebar_options._dialog
    assert dialog.isVisible()
    dialog.close()


def test_localize_follows_a_new_look(shown, monkeypatch):  # noqa: F811
    calls = []
    monkeypatch.setattr(
        gui_localize.render,
        "_paint_scalebar",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    shown.scalebar_action.trigger()
    scalebar_settings.set_shared_look({"Position": "Top center"})
    shown.view.viewport().grab()
    assert calls[-1]["position"] == "top center"


def test_localize_exports_the_scale_bar(
    shown, tmp_path, monkeypatch  # noqa: F811
):
    paths = iter([tmp_path / "plain.png", tmp_path / "bar.png"])
    monkeypatch.setattr(
        gui_localize.lib,
        "get_save_filename_ext_dialog",
        lambda *args, **kwargs: (str(next(paths)), "*.png"),
    )
    shown.export_current()
    shown.scalebar_action.trigger()
    shown.export_current()
    plain = QtGui.QImage(str(tmp_path / "plain.png"))
    bar = QtGui.QImage(str(tmp_path / "bar.png"))
    assert plain.size() == bar.size()
    # in the bar 20 px above the bottom, 35 px from the right edge
    x, y = bar.width() - 35 - 3, bar.height() - 20 - 3
    assert bar.pixelColor(x, y) == QtGui.QColor("white")
    assert plain.pixelColor(x, y) != bar.pixelColor(x, y)


def test_spinna_mask_view_uses_the_shared_look(
    mask_tab, dialogs, out, monkeypatch  # noqa: F811
):
    load_mask_locs(mask_tab, dialogs, save_molecules(out / "A.hdf5"))
    mask_tab.generate_mask()
    mask_tab.preview.resize(300, 300)
    mask_tab.preview.fit_view()
    mask_tab.scalebar_check.setChecked(True)
    calls = []
    original = render.draw_scalebar

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(gui_spinna.render, "draw_scalebar", record)
    scalebar_settings.set_shared_look({"Background opacity (%)": 50})
    mask_tab.preview.render_view()
    assert calls[-1]["background_opacity"] == pytest.approx(0.5)
    assert calls[-1]["display_length"] is True


def test_spinna_structure_preview_draws_the_scale_bar(
    structures_tab, dialogs, monkeypatch  # noqa: F811
):
    add_structure(structures_tab, dialogs, "AB")
    structures_tab.add_molecular_target()
    structures_tab.add_molecular_target()
    rows = target_rows(structures_tab)
    set_row(rows[0], "A", -5.0, 0.0, 0.0)
    set_row(rows[1], "B", 5.0, 0.0, 0.0)
    structures_tab.scalebar_length.setValue(4.0)
    preview = structures_tab.preview
    calls = []
    original = render.draw_scalebar

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(gui_spinna.render, "draw_scalebar", record)
    preview.render()
    (kwargs,) = calls
    assert kwargs["display_length"] is False  # as before
    # 4 nm are as long as before: 4 nm times the pixels per nm
    width_nm = render.viewport_width(kwargs["viewport"])
    length_px = gui_spinna.STRUCTURE_PREVIEW_SIZE * 4.0 / width_nm
    assert length_px == pytest.approx(4.0 * preview.factor)

    structures_tab.scalebar_options.set_print_length(True)  # redraws
    assert calls[-1]["display_length"] is True
    image = preview.qimage
    rgb = np.array(
        [
            [image.pixelColor(x, y).red() for x in range(image.width())]
            for y in range(image.height() - 60, image.height())
        ]
    )
    assert (rgb == 255).any()
