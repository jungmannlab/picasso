"""The z slicer (``SlicerDialog``) of ``picasso.gui.render``.

Slices are half-open, ``[slicermin, slicermax)``. The slice bounds
used to be set only by the slider's ``valueChanged`` signal, which Qt
does not emit when the new middle slice equals the slider's previous
value (e.g., 50 or 51 bins on a fresh dialog), nor did they survive a
z range thinner than one slice. Opening the slicer
then failed with ``AttributeError: 'SlicerDialog' object has no
attribute 'slicermin'``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0
N_LOCS = 2000


def _locs(z_range: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=N_LOCS).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=N_LOCS),
            "y": rng.uniform(0.0, HEIGHT, size=N_LOCS),
            "z": np.linspace(-z_range / 2, z_range / 2, N_LOCS),
            "lpx": np.full(N_LOCS, 0.1),
            "lpy": np.full(N_LOCS, 0.1),
            "photons": np.full(N_LOCS, 1000.0),
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


def _window(tmp_path, z_range: float) -> gui_render.Window:
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(z_range), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(256, 256)
    return window


# 2500 nm at 50 nm gives 51 bin edges, i.e., the middle slice is the
# slider's initial value of 25 and valueChanged is not emitted
@pytest.mark.parametrize("z_range", [2500.0, 1600.0, 10.0, 0.0])
def test_slicer_opens(qt_offscreen, tmp_path, z_range):
    window = _window(tmp_path, z_range)
    dialog = window.slicer_dialog
    dialog.initialize()

    position = dialog.sl.value()
    assert dialog.slicerposition == position
    assert dialog.slicermin == dialog.bins[position]
    assert dialog.slicermax == dialog.bins[position + 1]
    z = window.view.locs[0]["z"]
    assert ((z >= dialog.slicermin) & (z < dialog.slicermax)).any()
    dialog.close()
    window.close()


def test_slicer_bins_cover_z_range(qt_offscreen, tmp_path):
    window = _window(tmp_path, 1234.0)
    dialog = window.slicer_dialog
    dialog.initialize()
    z = window.view.locs[0]["z"]
    assert dialog.bins[0] <= z.min()
    assert dialog.bins[-1] > z.max()

    # a slice thicker than the z range leaves a single slice
    dialog.pick_slice.setValue(5000.0)
    assert dialog.sl.maximum() == 0
    locs, _ = window.view._prepare_locs_for_rendering()
    assert len(locs) == len(z)
    assert dialog.slicermin == dialog.bins[0]
    assert dialog.slicermax == dialog.bins[1]
    dialog.close()
    window.close()


def test_slicer_before_histogram_keeps_all_locs(qt_offscreen, tmp_path):
    """Ticking "Slice Dataset" before any histogram renders all
    localizations instead of failing."""
    window = _window(tmp_path, 1000.0)
    dialog = window.slicer_dialog
    dialog.slicer_radio_button.setChecked(True)
    locs, _ = window.view._prepare_locs_for_rendering()
    assert len(locs) == len(window.view.locs[0])
    window.close()
