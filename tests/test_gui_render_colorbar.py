"""The color bar (LUT) exported next to an image rendered by property.

Color-coding by z is how 3D data is shown in a figure, and the figure
needs the LUT itself, with the z values it stands for. Render therefore
saves a ``*_colorbar.png`` next to every image exported while rendering
by property - from the main window and from the 3D (rotation) window.
These tests cover that wiring: the file is written only when rendering
by property is on, and it shows the colors that the localizations are
rendered with.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtGui

from picasso import io, render
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0
Z_MIN, Z_MAX = -300.0, 300.0
N_COLORS = 16


def _locs(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "z": rng.uniform(Z_MIN, Z_MAX, size=n),
            "lpx": np.full(n, 0.1),
            "lpy": np.full(n, 0.1),
            "photons": np.full(n, 1000.0),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture(autouse=True)
def _settings_file(tmp_path, monkeypatch):
    """Never touch the developer's ~/.picasso/settings.yaml."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setattr(io, "_user_settings_filename", lambda: str(path))
    return path


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding one 3D channel, color-coded by z."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    d_dialog = window.display_settings_dlg
    d_dialog.parameter.setCurrentText("z")
    d_dialog.minimum_render.setValue(Z_MIN)
    d_dialog.maximum_render.setValue(Z_MAX)
    d_dialog.color_step.setValue(N_COLORS)
    d_dialog.colormap_prop.setCurrentText("gist_rainbow")
    return window


def _enable_property_rendering(window) -> None:
    window.display_settings_dlg.render_check.setChecked(True)


def _qimage_to_array(qimage):
    """Convert a QImage to an HxWx4 uint8 numpy array (BGRA)."""
    qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    width, height = qimage.width(), qimage.height()
    bits = qimage.bits()
    bits.setsize(height * width * 4)
    return np.frombuffer(bits, dtype=np.uint8).reshape(height, width, 4).copy()


class TestPropertyColorbar:
    def test_none_without_property_rendering(self, window):
        assert window.view.property_colorbar() is None

    def test_image_with_property_rendering(self, window):
        _enable_property_rendering(window)
        assert isinstance(window.view.property_colorbar(), QtGui.QImage)

    def test_holds_the_rendered_colors(self, window):
        """The bands are the colors the localizations are rendered
        with, i.e., those of the chosen colormap."""
        _enable_property_rendering(window)
        image = window.view.property_colorbar()
        array = _qimage_to_array(image)
        expected = render.get_colors_from_colormap(N_COLORS, "gist_rainbow")
        expected = {
            tuple(int(round(255 * c)) for c in color) for color in expected
        }
        found = {tuple(pixel) for pixel in array[:, :, :3].reshape(-1, 3)}
        found = {pixel[::-1] for pixel in found}  # the buffer is BGRA
        assert expected <= found

    def test_inverted_with_white_background(self, window):
        """With white background the image itself is inverted, so the
        color bar is inverted too and thus matches it."""
        _enable_property_rendering(window)
        window.dataset_dialog.wbackground.setChecked(True)
        array = _qimage_to_array(window.view.property_colorbar())
        assert list(array[0, 0, :3]) == [255, 255, 255]


class TestExportColorbar:
    def test_saved_next_to_the_image(self, window, tmp_path):
        _enable_property_rendering(window)
        window.view.save_property_colorbar(str(tmp_path / "view.png"))
        colorbar = str(tmp_path / "view_colorbar.png")
        assert os.path.exists(colorbar)
        assert os.path.getsize(colorbar) > 0

    def test_format_follows_the_user_setting(self, window, tmp_path):
        """The image keeps its own format; the color bar is saved in the
        one the user settings ask for."""
        _enable_property_rendering(window)
        io.save_user_settings({"Render": {"Colorbar format": ".svg"}})
        window.view.save_property_colorbar(str(tmp_path / "view.tif"))
        assert os.path.exists(str(tmp_path / "view_colorbar.svg"))
        assert not os.path.exists(str(tmp_path / "view_colorbar.tif"))

    def test_svg_holds_the_bar_itself(self, window, tmp_path):
        """The SVG is drawn, not an embedded image, so the bands and the
        text stay editable in figure software."""
        _enable_property_rendering(window)
        io.save_user_settings({"Render": {"Colorbar format": ".svg"}})
        window.view.save_property_colorbar(str(tmp_path / "view.png"))
        svg = (tmp_path / "view_colorbar.svg").read_text()
        assert "<text" in svg
        assert "image/png" not in svg  # no rasterized bar embedded

    def test_not_saved_without_property_rendering(self, window, tmp_path):
        window.view.save_property_colorbar(str(tmp_path / "view.png"))
        assert not os.path.exists(str(tmp_path / "view_colorbar.png"))

    def test_property_is_recorded_in_the_info(self, window):
        _enable_property_rendering(window)
        info = window.export_current_info(path=None)
        assert info["Render property"] == "z"
        assert info["Render property min."] == Z_MIN
        assert info["Render property max."] == Z_MAX
        assert info["Render property colors"] == N_COLORS
        assert info["Colormap property"] == "gist_rainbow"

    def test_no_property_in_the_info_without_property_rendering(self, window):
        assert "Render property" not in window.export_current_info(path=None)


class TestExportColorbarRotation:
    """The 3D window exports the same color bar; it renders with the
    colors of the main window, so it asks the main window for it."""

    def test_saved_next_to_the_image(self, window, tmp_path):
        _enable_property_rendering(window)
        view_rot = window.window_rot.view_rot
        view_rot.x_render_state = True
        path = str(tmp_path / "view_rotated.png")
        view_rot.save_property_colorbar(path)
        assert os.path.exists(str(tmp_path / "view_rotated_colorbar.png"))

    def test_not_saved_without_property_rendering(self, window, tmp_path):
        view_rot = window.window_rot.view_rot
        view_rot.x_render_state = False
        view_rot.save_property_colorbar(str(tmp_path / "view_rotated.png"))
        assert not os.path.exists(str(tmp_path / "view_rotated_colorbar.png"))
