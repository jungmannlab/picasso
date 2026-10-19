"""The contrast slider below the image of Render and Localize.

Whether it is shown is set by ``contrast_slider`` in the module's
section of the user settings: off by default in Render, on in Localize.
Render's slider is bound to the minimum and maximum density of the
Display Settings dialog.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import io, lib_qt
from picasso.gui import localize as gui_localize
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n) * PIXELSIZE,
            "y": rng.uniform(0.0, HEIGHT, size=n) * PIXELSIZE,
            "lpx": np.full(n, 13.0),
            "lpy": np.full(n, 13.0),
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


def _set(section: str, value) -> None:
    settings = io.load_user_settings()
    settings[section]["contrast_slider"] = value
    io.save_user_settings(settings)


@pytest.fixture
def render_window(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    window.view.add(
        str(tmp_path / "locs.hdf5"), _locs(), _info(), render_=False
    )
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    window.view.resize(128, 128)
    return window


@pytest.mark.parametrize("section", ["Render", "Localize"])
@pytest.mark.parametrize("default", [True, False])
def test_missing_key_is_written_with_the_default(section, default):
    assert lib_qt.contrast_slider_setting(section, default) is default
    assert io.load_user_settings()[section]["contrast_slider"] is default


def test_invalid_value_falls_back_to_the_default():
    _set("Render", "yes")
    assert lib_qt.contrast_slider_setting("Render", False) is False
    # the user's value is left in the file
    assert io.load_user_settings()["Render"]["contrast_slider"] == "yes"


def test_render_hides_the_slider_by_default(render_window):
    assert render_window.contrast_bar.isHidden()


def test_render_applies_the_setting_once_saved(render_window):
    _set("Render", True)
    render_window.user_settings_dialog.saved.emit()
    assert not render_window.contrast_bar.isHidden()
    _set("Render", False)
    render_window.user_settings_dialog.saved.emit()
    assert render_window.contrast_bar.isHidden()


def test_render_slider_is_bound_to_the_density_boxes(render_window):
    dialog = render_window.display_settings_dlg
    slider = render_window.contrast_slider
    # the re-render may rescale the density (display pixel size), so the
    # slider is compared with the spin boxes, not the value typed
    dialog.maximum.setValue(5.0)
    values = (dialog.minimum.value(), dialog.maximum.value())
    assert slider.values() == pytest.approx(values)
    # dragging writes into the spin boxes; the dialog's slider follows
    low, high = 0.1 * values[1], 0.5 * values[1]
    slider.setValues(low, high)
    values = (dialog.minimum.value(), dialog.maximum.value())
    assert values == pytest.approx((low, high), rel=1e-3)
    assert slider.values() == pytest.approx(values)
    assert dialog.contrast_slider.values() == pytest.approx(values)


def test_render_slider_follows_silent_updates(render_window):
    render_window.display_settings_dlg.silent_maximum_update(7.0)
    assert render_window.contrast_slider.values()[1] == pytest.approx(7.0)


def test_render_auto_rescales(render_window):
    render_window.display_settings_dlg.maximum.setValue(1e4)
    render_window.view.async_rendering = False
    render_window._autoscale_contrast()
    assert render_window.display_settings_dlg.maximum.value() < 1e4


def test_localize_shows_the_slider_by_default(qt_offscreen):
    window = gui_localize.Window()
    assert not window.contrast_bar.isHidden()
    _set("Localize", False)
    window.user_settings_dialog.saved.emit()
    assert window.contrast_bar.isHidden()


def test_render_bar_follows_the_image_background(render_window):
    # theme colors until something is rendered
    assert render_window.contrast_bar.styleSheet() == ""
    render_window.view.async_rendering = False
    render_window.view.update_scene(autoscale=True)
    empty = tuple(int(c) for c in render_window.view._color_range[0])
    assert render_window._contrast_bar_color == empty
    assert "rgb({}, {}, {})".format(*empty) in (
        render_window.contrast_bar.styleSheet()
    )
    # light text on the dark (magma) background, dark on a white one
    assert "color: rgb(255, 255, 255)" in (
        render_window._contrast_auto_button.styleSheet()
    )
    render_window.dataset_dialog.wbackground.setChecked(True)
    assert "color: rgb(0, 0, 0)" in (
        render_window._contrast_auto_button.styleSheet()
    )
