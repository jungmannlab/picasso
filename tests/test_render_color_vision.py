"""Simulation of color vision deficiencies (``picasso.render.color_vision``)
and Render's View > Color vision preview, which shows the main and 3D
windows as people with color blindness see them while saving and
exporting keep the colors.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtGui, QtWidgets

from picasso import render
from picasso.gui import render as gui_render

from tests.test_gui_rotation_async import HEIGHT, WIDTH, _info, _locs

DEFICIENCIES = list(render.COLOR_VISION_DEFICIENCIES)


def _reference(image: np.ndarray, deficiency: str) -> np.ndarray:
    """Float simulation without lookup tables."""
    srgb = image / 255
    linear = np.where(
        srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    )
    matrix = render.COLOR_VISION_DEFICIENCIES[deficiency].astype(float)
    out = np.clip(np.einsum("ij,...j->...i", matrix, linear), 0, 1)
    encoded = np.where(
        out <= 0.0031308, 12.92 * out, 1.055 * out ** (1 / 2.4) - 0.055
    )
    return np.round(encoded * 255)


def _rgb(image: QtGui.QImage) -> np.ndarray:
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    bgra = np.frombuffer(bits, dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4
    )
    return bgra[:, : image.width(), 2::-1].copy()


# --- simulation ---------------------------------------------------------


@pytest.mark.parametrize("deficiency", DEFICIENCIES)
def test_simulation_matches_the_float_reference(deficiency):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    simulated = render.simulate_color_vision(image, deficiency)
    assert simulated.shape == image.shape
    assert simulated.dtype == np.uint8
    error = np.abs(simulated.astype(int) - _reference(image, deficiency))
    assert error.max() <= 1


@pytest.mark.parametrize("deficiency", DEFICIENCIES)
def test_grays_are_unchanged(deficiency):
    grays = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    simulated = render.simulate_color_vision(grays, deficiency)
    assert np.abs(simulated.astype(int) - grays).max() <= 1


@pytest.mark.parametrize("deficiency", ["protanopia", "deuteranopia"])
def test_red_and_green_lose_their_difference(deficiency):
    red_green = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    simulated = render.simulate_color_vision(red_green, deficiency)
    # both become yellowish: no blue, red and green in similar ratios
    ratios = simulated[:, 0] / simulated[:, 1]
    assert abs(ratios[0] - ratios[1]) < 0.2
    assert (simulated[:, 2] < 0.3 * simulated[:, 0]).all()


def test_unknown_deficiency_or_shape_raises():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unknown color vision"):
        render.simulate_color_vision(image, "achromatopsia")
    with pytest.raises(ValueError, match="shape"):
        render.simulate_color_vision(np.zeros((2, 2, 4)), "protanopia")


def test_qimage_simulation_keeps_the_input_and_alpha(qt_offscreen):
    image = QtGui.QImage(8, 4, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor(255, 0, 0, 100))
    simulated = render.simulate_color_vision_qimage(image, "deuteranopia")
    assert image.pixelColor(0, 0) == QtGui.QColor(255, 0, 0, 100)
    color = simulated.pixelColor(0, 0)
    assert color.alpha() == 100
    expected = render.simulate_color_vision(
        np.array([255, 0, 0], dtype=np.uint8), "deuteranopia"
    )
    assert (color.red(), color.green(), color.blue()) == tuple(expected)


# --- Render ---------------------------------------------------------------


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """A Render window showing one channel on a 256 x 256 canvas."""
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"), _locs(), _info(), render_=False
    )
    window.view.resize(256, 256)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()


def _menu_actions(window) -> dict:
    view_menu = next(
        a.menu() for a in window.menu_bar.actions() if a.text() == "View"
    )
    submenu = next(
        a.menu()
        for a in view_menu.actions()
        if a.text() == "Color vision preview"
    )
    return {a.data(): a for a in submenu.actions()}


def test_preview_changes_only_the_display(window):
    view = window.view
    view.update_scene()
    saved = _rgb(view.qimage)
    assert (saved.max(axis=2) > 0).any()  # localizations are visible

    _menu_actions(window)["deuteranopia"].trigger()
    assert window.color_vision == "deuteranopia"
    displayed = _rgb(view.pixmap.toImage())
    # the image that saving and exporting use keeps its colors
    assert np.array_equal(_rgb(view.qimage), saved)
    # below the label at the top, the display is the simulation
    expected = render.simulate_color_vision(saved, "deuteranopia")
    below = slice(view.qimage.height() // 2, None)
    assert np.abs(displayed[below].astype(int) - expected[below]).max() <= 1
    assert not np.array_equal(displayed, saved)

    window.set_color_vision(None)
    assert np.array_equal(_rgb(view.pixmap.toImage()), saved)


def test_menu_ticks_follow_the_preview(window):
    window.set_color_vision("tritanopia")
    actions = _menu_actions(window)
    # the ticks are updated when the submenu opens
    actions["tritanopia"].parent().aboutToShow.emit()
    assert actions["tritanopia"].isChecked()
    assert not actions[None].isChecked()
    with pytest.raises(ValueError):
        window.set_color_vision("achromatopsia")


def test_3d_window_follows_the_preview(window):
    view_rot = window.window_rot.view_rot
    view_rot.qimage = QtGui.QImage(200, 100, QtGui.QImage.Format.Format_RGB32)
    view_rot.qimage.fill(QtGui.QColor(255, 0, 0))
    window.set_color_vision("protanopia")
    displayed = view_rot.pixmap.toImage().pixelColor(5, 90)
    expected = render.simulate_color_vision(
        np.array([255, 0, 0], dtype=np.uint8), "protanopia"
    )
    assert (displayed.red(), displayed.green(), displayed.blue()) == tuple(
        expected
    )
    assert view_rot.qimage.pixelColor(5, 90) == QtGui.QColor(255, 0, 0)
    # the 3D window's View menu has the same submenu
    view_menu = next(
        a.menu()
        for a in window.window_rot.menu_bar.actions()
        if a.text() == "View"
    )
    assert "Color vision preview" in [a.text() for a in view_menu.actions()]
