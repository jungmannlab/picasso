"""The charts of Picasso: Simulate.

Simulate's previews of the generated positions and of the structure,
and the statistics of imported experimental data, take the shared chart
style (``picasso.gui.plot_style``), i.e., by default the light or dark
theme of the windows.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import plot_style, theme
from picasso.gui import simulate as gui_simulate
from picasso.gui.plot_style import THEMES, PlotStyle


@pytest.fixture(autouse=True)
def fresh_hub(monkeypatch):
    """Each test gets its own chart style hub."""
    monkeypatch.setattr(plot_style, "_hub", None)
    monkeypatch.setattr(plot_style, "_dialog", None)


@pytest.fixture
def window(qt_offscreen, restore_theme):
    window = gui_simulate.Window()
    yield window
    window.close()


def _frame(window) -> Rectangle:
    """The dashed frame around the positions, the only patch drawn."""
    (frame,) = window.ax1.patches
    assert isinstance(frame, Rectangle)
    return frame


def test_previews_follow_the_windows_theme(window, restore_theme):
    theme.apply(restore_theme, theme.Appearance(mode="Dark"))
    dark = THEMES["Dark"]
    for figure in (window.figure1, window.figure2):
        assert to_hex(figure.get_facecolor()) == dark["figure"]
    assert to_hex(window.ax1.get_facecolor()) == dark["axes"]
    # the frame is drawn in the theme's color, not black
    assert to_hex(_frame(window).get_edgecolor()) == dark["text_muted"]

    theme.apply(restore_theme, theme.Appearance(mode="Light"))
    assert to_hex(window.figure1.get_facecolor()) == (
        THEMES["Light"]["figure"]
    )
    assert to_hex(_frame(window).get_edgecolor()) == (
        THEMES["Light"]["text_muted"]
    )


def test_previews_take_the_series_colors(window):
    style = replace(
        PlotStyle(),
        theme="Classic",
        palette="Custom",
        colors=("#123456", "#654321", "#00FF00", "#0000FF"),
    )
    plot_style.set_current(style)
    assert to_hex(window.ax1.get_facecolor()) == THEMES["Classic"]["axes"]
    (positions,) = window.ax1.get_lines()
    assert to_hex(positions.get_color()) == "#123456"
    # one color per exchange round of the structure
    rounds = window.ax2.get_lines()
    assert len(rounds) == len(window.noexchangecolors)
    assert to_hex(rounds[0].get_color()) == "#123456"


def test_experiment_statistics_window(qt_offscreen, monkeypatch):
    """Importing experimental data (offered in the advanced mode, which
    has the noise model) draws normalized histograms of the photons, PSF
    width and background with their normal fits."""
    monkeypatch.setattr(gui_simulate, "ADVANCEDMODE", 1)
    window = gui_simulate.Window()
    rng = np.random.default_rng(0)
    n = 2000
    locs = pd.DataFrame(
        {
            "photons": rng.normal(2000, 300, n),
            "sx": rng.normal(1.0, 0.1, n),
            "sy": rng.normal(1.0, 0.1, n),
            "bg": rng.normal(200, 20, n),
        }
    )
    monkeypatch.setattr(io, "load_locs", lambda path, qt_parent: (locs, [{}]))
    answers = iter(["100", "5", "50", "100"])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: (next(answers), True),
    )
    window.readhdf5("experiment.hdf5")

    plots = window._experiment_window
    axes = plots.figure.axes
    assert [a.get_title().split(":")[0] for a in axes] == [
        "Photons",
        "PSF",
        "Background",
    ]
    for a in axes:
        assert a.patches  # the histogram
        (fit,) = a.get_lines()
        # normalized: the fitted density integrates to about 1
        x, y = fit.get_data()
        assert np.trapezoid(y, x) == pytest.approx(1, abs=0.1)
    # the chart style applies to the window as to any chart window; it
    # redraws on new axes
    plot_style.set_current(replace(PlotStyle(), theme="Dark"))
    axes = plots.figure.axes
    assert len(axes) == 3
    assert to_hex(axes[0].get_facecolor()) == THEMES["Dark"]["axes"]
    plots.close()
    window.close()
