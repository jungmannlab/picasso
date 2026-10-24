"""Per-channel contrast in the Files dialog of Render.

Each channel has a lower and an upper contrast limit in percent of the
global contrast window (Display settings), set with two spin boxes and
a slider bound to them. These tests cover the widgets, the values the
renderer is given, re-coloring from the cached render, closing a
channel, the single-channel mode and the .yaml saved with exported
images.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import io
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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


@pytest.fixture(autouse=True)
def _settings_file(tmp_path, monkeypatch):
    """Never touch the developer's ~/.picasso/settings.yaml."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setattr(io, "_user_settings_filename", lambda: str(path))
    return path


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding two channels."""
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        path = str(tmp_path / f"locs_{i}.hdf5")
        window.view.add(path, _locs(seed=i), _info(), render_=False)
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    window.view.resize(128, 128)
    return window


def test_defaults_use_the_global_contrast(window):
    dialog = window.dataset_dialog
    assert len(dialog.contrast_widgets) == 2
    assert [box.value() for box in dialog.contrast_low] == [0.0, 0.0]
    assert [box.value() for box in dialog.contrast_high] == [100.0, 100.0]
    assert window.view.read_channel_contrast() == [(0.0, 1.0), (0.0, 1.0)]


def test_spin_boxes_and_slider_are_bound(window):
    dialog = window.dataset_dialog
    dialog.contrast_high[1].setValue(40.0)
    assert dialog.contrast_sliders[1].values() == pytest.approx((0.0, 40.0))
    # the lower limit cannot pass the upper one
    assert dialog.contrast_low[1].maximum() == pytest.approx(39.9)
    dialog.contrast_sliders[1].setValues(10.0, 40.0)
    assert dialog.contrast_low[1].value() == pytest.approx(10.0)
    assert window.view.read_channel_contrast() == [
        (0.0, 1.0),
        pytest.approx((0.1, 0.4)),
    ]


def test_unticked_channels_are_skipped(window):
    dialog = window.dataset_dialog
    dialog.contrast_high[1].setValue(50.0)
    dialog.checks[0].blockSignals(True)
    dialog.checks[0].setChecked(False)
    dialog.checks[0].blockSignals(False)
    assert window.view.read_channel_contrast() == [(0.0, 0.5)]


def test_change_recolors_from_the_cache(window, monkeypatch):
    calls = []
    monkeypatch.setattr(
        window.view, "update_scene", lambda **kwargs: calls.append(kwargs)
    )
    window.dataset_dialog.contrast_high[0].setValue(50.0)
    assert calls == [{"use_cache": True}]


def test_brightens_the_rendered_channel(window):
    window.view.render_scene(autoscale=True)
    vmax = window.display_settings_dlg.maximum.value()
    before = window.view.render_scene(use_cache=True)
    window.dataset_dialog.set_channel_contrast(0, 0.0, 20.0, update=False)
    after = window.view.render_scene(use_cache=True)
    assert before != after
    # the global contrast is left alone
    assert window.display_settings_dlg.maximum.value() == vmax


def test_reset(window):
    dialog = window.dataset_dialog
    dialog.set_channel_contrast(0, 5.0, 30.0, update=False)
    dialog.set_channel_contrast(1, 0.0, 300.0, update=False)
    dialog.reset_channel_contrast()
    assert window.view.read_channel_contrast() == [(0.0, 1.0), (0.0, 1.0)]
    assert dialog.contrast_sliders[0].values() == pytest.approx((0.0, 100.0))


def test_invalid_window_is_ignored(window):
    dialog = window.dataset_dialog
    dialog.set_channel_contrast(0, 50.0, 50.0, update=False)
    assert window.view.read_channel_contrast()[0] == (0.0, 1.0)


def test_closing_a_channel_removes_its_widgets(window):
    dialog = window.dataset_dialog
    dialog.set_channel_contrast(1, 0.0, 60.0, update=False)
    dialog.close_file(0, render=False)
    assert len(dialog.contrast_widgets) == 1
    assert len(dialog.contrast_low) == len(dialog.contrast_high) == 1
    assert dialog.contrast_high[0].value() == pytest.approx(60.0)


def test_single_channel_mode_disables_the_column(window):
    dialog = window.dataset_dialog
    dialog.set_single_channel_mode("colormap")
    assert not any(w.isEnabled() for w in dialog.contrast_widgets)
    dialog.set_single_channel_mode(None)
    assert all(w.isEnabled() for w in dialog.contrast_widgets)


def test_saved_and_restored_with_exported_images(window):
    dialog = window.dataset_dialog
    dialog.set_channel_contrast(0, 10.0, 50.0, update=False)
    dialog.set_channel_contrast(1, 0.0, 250.0, update=False)
    info = window.export_current_info(path=None)
    assert info[gui_render.CHANNEL_CONTRAST_KEY] == [
        [10.0, 50.0],
        [0.0, 250.0],
    ]
    dialog.reset_channel_contrast()
    # no render happened, so the FOV boxes of the Info dialog are empty
    info[gui_render.FOV_KEY] = [
        0.0,
        0.0,
        WIDTH * PIXELSIZE,
        HEIGHT * PIXELSIZE,
    ]
    window.view.load_screenshot(info)
    assert window.view.read_channel_contrast() == [
        pytest.approx((0.1, 0.5)),
        pytest.approx((0.0, 2.5)),
    ]
