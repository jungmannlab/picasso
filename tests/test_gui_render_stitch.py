"""Postprocess > Stitch tiles in ``picasso.gui.render``: the dialog
and the replacement of the tile channels by the stitched ones.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from picasso.gui import render as gui_render
from tests.test_gui_render_tasks import _wait_for_tasks
from tests.test_stitch import _grid


@pytest.fixture
def tiles():
    """Four tiles of a simulated sample: ``(locs, infos, origins)``."""
    rng = np.random.default_rng(0)
    locs, infos, origins, _ = _grid(rng, n_cols=2, n_rows=2)
    return locs, infos, origins


@pytest.fixture
def window(qt_offscreen, tmp_path, tiles):
    window = gui_render.Window(plugins_loaded=True)
    for t, (locs, info) in enumerate(zip(*tiles[:2])):
        path = str(tmp_path / f"tile{t}.hdf5")
        window.view.add(path, locs.copy(), info, render_=False)
    yield window
    _wait_for_tasks(qt_offscreen)
    window.close()


def _messages(monkeypatch) -> list[str]:
    messages = []
    monkeypatch.setattr(
        gui_render.QtWidgets.QMessageBox,
        "information",
        lambda parent, title, text, *args: messages.append(text),
    )
    return messages


def _run(window, qapp, monkeypatch, **changes) -> list[str]:
    """Stitch with the dialog's defaults, updated by ``changes``."""
    accepted = gui_render.QtWidgets.QDialog.DialogCode.Accepted
    monkeypatch.setattr(gui_render.StitchDialog, "exec", lambda _: accepted)
    params, ok = gui_render.StitchDialog.getParams(window)
    assert ok
    params.update(changes)
    monkeypatch.setattr(
        gui_render.StitchDialog,
        "getParams",
        staticmethod(lambda parent: (params, True)),
    )
    messages = _messages(monkeypatch)
    window.view.stitch()
    _wait_for_tasks(qapp)
    return messages


def test_dialog_defaults(window):
    dialog = gui_render.StitchDialog(window)
    np.testing.assert_array_equal(dialog.tile_indices(), [0, 1, 2, 3])
    assert [target.value() for target in dialog.targets] == [1, 1, 1, 1]
    assert dialog.initial.currentIndex() == 0  # stage positions
    assert dialog.orientation.isEnabled()
    assert dialog.use_z.isEnabled() and dialog.use_z.isChecked()
    dialog.initial.setCurrentIndex(1)
    assert not dialog.orientation.isEnabled()


def test_dialog_rejects_target_twice_in_a_tile(window, monkeypatch):
    messages = _messages(monkeypatch)
    dialog = gui_render.StitchDialog(window)
    dialog.tiles[1].setValue(1)  # channels 0 and 1 in tile 1, target 1
    dialog.accept()
    assert dialog.result() != gui_render.QtWidgets.QDialog.DialogCode.Accepted
    assert "only once" in messages[0]
    dialog.targets[1].setValue(2)
    dialog.accept()
    assert dialog.result() == gui_render.QtWidgets.QDialog.DialogCode.Accepted


def _tile_offsets(info, paths):
    """Stitched x, y offset of each tile in the order of ``paths``."""
    entry = info[-1]
    order = [entry["Stitched files"].index(path) for path in paths]
    return np.column_stack(
        [
            np.array(entry["Tile offsets x (cam. px)"])[order],
            np.array(entry["Tile offsets y (cam. px)"])[order],
        ]
    )


def test_stitch_replaces_tiles(window, qt_offscreen, monkeypatch, tiles):
    paths = list(window.view.locs_paths)
    n_total = sum(len(locs) for locs in window.view.locs)
    messages = _run(window, qt_offscreen, monkeypatch)
    view = window.view
    assert len(view.locs) == 1
    assert os.path.basename(view.locs_paths[0]) == "tile0_stitched.hdf5"
    assert 0 < len(view.locs[0]) < n_total  # overlaps cut
    offsets = _tile_offsets(view.infos[0], paths)
    origins = tiles[2]
    np.testing.assert_allclose(
        offsets - offsets[0], origins - origins[0], atol=0.05
    )
    assert "Stitched 4 tiles into 1 channel(s)" in messages[-1]
    assert "Stage orientation: -x, y" in messages[-1]


def test_stitch_from_current_positions(
    window, qt_offscreen, monkeypatch, tiles
):
    """Tiles placed roughly by hand (Move tool shifts in the
    metadata), without stage positions."""
    view = window.view
    paths = list(view.locs_paths)
    origins = tiles[2]
    rough = origins + np.random.default_rng(1).uniform(-15, 15, origins.shape)
    for i, (dx, dy) in enumerate(rough):
        view._set_channel_xy(
            i,
            view.locs[i]["x"] + np.float32(dx),
            view.locs[i]["y"] + np.float32(dy),
        )
        view._record_manual_shift(i, dx, dy)
        view.infos[i][0].pop("Micro-Manager Metadata")
    view.fit_canvas()
    _run(window, qt_offscreen, monkeypatch, stage=False)
    assert len(view.locs) == 1
    offsets = _tile_offsets(view.infos[0], paths)
    np.testing.assert_allclose(
        offsets - offsets[0], origins - origins[0], atol=0.05
    )
    entry = view.infos[0][-1]
    assert entry["Initial positions"] == "given"
    assert entry["Stage orientation"] is None
