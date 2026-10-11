"""Reordering the channels and the default channel colors in the Files
dialog of Render.

Channels are reordered by dragging their index in the channel table or
from the index's right click menu; everything that refers to a channel
by its index moves along. The colors given to newly loaded channels are
kept in the user settings.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import io
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": np.full(n, 0.1 + 0.01 * seed),
            "lpy": np.full(n, 0.1 + 0.01 * seed),
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
def settings_file(tmp_path, monkeypatch):
    """Never touch the developer's ~/.picasso/settings.yaml."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setattr(io, "_user_settings_filename", lambda: str(path))
    return path


def _add(window, tmp_path, i: int) -> None:
    path = str(tmp_path / f"locs_{i}.hdf5")
    window.view.add(path, _locs(seed=i), _info(), render_=False)


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding three channels."""
    window = gui_render.Window(plugins_loaded=True)
    for i in range(3):
        _add(window, tmp_path, i)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    return window


def _names(window) -> list[str]:
    return [p.rsplit("_", 1)[-1] for p in window.view.locs_paths]


def _row_checks(dialog) -> list:
    """The check box shown in each row of the channel table."""
    return [
        dialog.table.cellWidget(row, 0).findChild(QtWidgets.QCheckBox)
        for row in range(dialog.table.rowCount())
    ]


# --- reordering -------------------------------------------------------


def test_reorder_moves_channels_and_widgets(window):
    dialog = window.dataset_dialog
    view = window.view
    locs = list(view.locs)
    checks = list(dialog.checks)
    colors = [c.currentText() for c in dialog.colorselection]
    dialog.contrast_high[2].setValue(50.0)

    dialog.reorder_channels([2, 0, 1])

    assert _names(window) == ["2.hdf5", "0.hdf5", "1.hdf5"]
    assert [id(x) for x in view.locs] == [id(locs[i]) for i in (2, 0, 1)]
    assert dialog.checks == [checks[i] for i in (2, 0, 1)]
    assert _row_checks(dialog) == dialog.checks
    assert [c.currentText() for c in dialog.colorselection] == [
        colors[i] for i in (2, 0, 1)
    ]
    assert view.read_channel_contrast()[0] == (0.0, 0.5)
    assert len(view.index_blocks) == len(view.render_index) == 3


def test_reorder_keeps_the_lists_shared_by_other_dialogs(window):
    view = window.view
    lists = (view.locs, view.infos, view.locs_paths)
    window.dataset_dialog.reorder_channels([1, 2, 0])
    assert (view.locs, view.infos, view.locs_paths) == lists
    assert all(
        a is b for a, b in zip(lists, (view.locs, view.infos, view.locs_paths))
    )


def test_reorder_updates_channel_indices_elsewhere(window):
    view = window.view
    tools = window.tools_settings_dialog
    tools.set_move_channels((2,))
    view._drift = ["d0", "d1", "d2"]
    view._move_undo = [((0, 2), 1.0, 2.0)]
    window.slicer_dialog.zcoord = ["z0", "z1", "z2"]
    window.test_clusterer_dialog.channels.setCurrentIndex(2)

    window.dataset_dialog.move_channel(2, 0)

    assert tools.move_channels() == (0,)
    assert view._drift == ["d2", "d0", "d1"]
    assert view._move_undo == [((1, 0), 1.0, 2.0)]
    assert window.slicer_dialog.zcoord == ["z2", "z0", "z1"]
    combo = window.test_clusterer_dialog.channels
    assert [combo.itemText(i) for i in range(3)] == [
        "locs_2.hdf5",
        "locs_0.hdf5",
        "locs_1.hdf5",
    ]
    assert combo.currentIndex() == 0


def test_move_channel(window):
    dialog = window.dataset_dialog
    dialog.move_channel(0, 2)
    assert _names(window) == ["1.hdf5", "2.hdf5", "0.hdf5"]
    dialog.move_channel(2, 1)
    assert _names(window) == ["1.hdf5", "0.hdf5", "2.hdf5"]


def test_dragging_the_index_reorders(window):
    dialog = window.dataset_dialog
    header = dialog.table.verticalHeader()
    header.moveSection(0, 2)
    assert _names(window) == ["1.hdf5", "2.hdf5", "0.hdf5"]
    # the header is back in order, so rows and indices agree
    assert [header.logicalIndex(i) for i in range(3)] == [0, 1, 2]
    assert _row_checks(dialog) == dialog.checks


def test_closing_after_reorder(window):
    dialog = window.dataset_dialog
    dialog.reorder_channels([2, 1, 0])
    dialog.close_file(0, render=False)
    assert _names(window) == ["1.hdf5", "0.hdf5"]
    assert _row_checks(dialog) == dialog.checks


def test_invalid_order_raises(window):
    with pytest.raises(ValueError):
        window.dataset_dialog.reorder_channels([0, 0, 1])


def test_close_removes_drift_data(window):
    view = window.view
    view._drift = ["d0", "d1", "d2"]
    view._driftfiles = ["f0", "f1", "f2"]
    view.currentdrift = ["c0", "c1", "c2"]
    window.dataset_dialog.close_file(1, render=False)
    assert view._drift == ["d0", "d2"]
    assert view._driftfiles == ["f0", "f2"]
    assert view.currentdrift == ["c0", "c2"]


# --- default colors ---------------------------------------------------


def test_missing_default_colors_are_written(window):
    settings = io.load_user_settings()
    dialog = window.dataset_dialog
    assert settings["Render"][gui_render.CHANNEL_COLORS_KEY] == list(
        dialog.default_colors
    )
    assert [c.currentText() for c in dialog.colorselection] == [
        "red",
        "cyan",
        "green",
    ]


def test_saved_default_colors_are_used(window, tmp_path):
    dialog = window.dataset_dialog
    dialog.colorselection[0].setCurrentText("blue")
    dialog.colorselection[1].setCurrentText("green_gradient")
    dialog.colorselection[2].setCurrentText("#FF5733")
    dialog.save_default_colors()

    stored = io.load_user_settings()["Render"][gui_render.CHANNEL_COLORS_KEY]
    assert stored[:4] == ["blue", "green_gradient", "#FF5733", "red"]
    assert "blue" not in stored[3:]

    second = gui_render.Window(plugins_loaded=True)
    for i in range(4):
        _add(second, tmp_path, i)
    assert [c.currentText() for c in second.dataset_dialog.colorselection] == [
        "blue",
        "green_gradient",
        "#FF5733",
        "red",
    ]


def test_invalid_default_colors_are_skipped(window):
    settings = io.load_user_settings()
    settings["Render"][gui_render.CHANNEL_COLORS_KEY] = [
        "not a color",
        "yellow",
    ]
    io.save_user_settings(settings)
    colors = window.dataset_dialog.default_channel_colors()
    assert colors[0] == "yellow"
    assert colors[1:4] == ["red", "cyan", "green"]


def test_restore_default_colors(window):
    dialog = window.dataset_dialog
    dialog.colorselection[0].setCurrentText("blue")
    dialog.save_default_colors()
    dialog.restore_default_colors()
    assert dialog.default_channel_colors() == list(dialog.default_colors)


def test_load_colors_with_fewer_names(window, tmp_path, monkeypatch):
    path = tmp_path / "colors.txt"
    path.write_text("blue\nyellow\n")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    dialog = window.dataset_dialog
    dialog.load_colors()
    assert [c.currentText() for c in dialog.colorselection] == [
        "blue",
        "yellow",
        "green",
    ]


# --- automatic coloring -----------------------------------------------


def test_automatic_coloring_grays_out_the_color_selection(window, tmp_path):
    dialog = window.dataset_dialog

    def enabled():
        return [
            (combo.isEnabled(), picker.isEnabled())
            for combo, picker in zip(
                dialog.colorselection, dialog.colorpickers
            )
        ]

    assert enabled() == [(True, True)] * 3
    dialog.auto_colors.setChecked(True)
    assert enabled() == [(False, False)] * 3
    _add(window, tmp_path, 3)  # a new channel follows
    assert enabled() == [(False, False)] * 4
    # the gradient previews show the automatic colors
    assert all(preview.isEnabled() for preview in dialog.colorpreviews)
    dialog.auto_colors.setChecked(False)
    assert enabled() == [(True, True)] * 4


def test_single_channel_mode_grays_out_the_color_selection(window):
    dialog = window.dataset_dialog
    dialog.set_single_channel_mode("colormap")
    assert not any(combo.isEnabled() for combo in dialog.colorselection)
    dialog.set_single_channel_mode(None)
    assert all(combo.isEnabled() for combo in dialog.colorselection)
