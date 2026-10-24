"""Picasso: Localize's dialogs, menu actions and window workflows.

The user-facing paths of the Localize window end to end: opening movies
(the metadata prompts, the concatenation review, background loading and
its cancel), drawing ROIs and split-FOV regions with the mouse, the scale
bar and zoom, identification of one channel, of every channel and of the
channel sum, the single, separate and joint fits with what they save
(localizations, drift, 3D calibrations), identifications from picks,
localizations and saved files, the camera config and the calibration
files, and the refusals and cancel paths along the way.

Movies are tiny (20 frames of 32 x 32 pixels with two still spots) and
memory mapped, as the fit requires of a loaded movie; the identification
and fit threads run for real and are waited for (``_wait``). Modal dialogs
are answered by patching their ``exec`` or static getters, and message
boxes are recorded instead of shown.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io, lib, transforms
from picasso.gui import localize as gui_localize
from tests.test_gui_localize_calibration import (
    ACCEPTED,
    REJECTED,
    _answer,
    _load,
    _Messages,
)
from tests.conftest import affine

FRAMES = 20
SIZE = 32
SPOTS = ((8.3, 9.6), (22.4, 20.2))


def _spot_movie(n_frames=FRAMES, size=SIZE, spots=SPOTS, seed=0):
    """A short movie of bright, still Gaussian spots on a dim noisy
    background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size), 100.0)
    for y, x in spots:
        image += 1500 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.4)
    movie = rng.poisson(np.broadcast_to(image, (n_frames, size, size)))
    return movie.astype(np.uint16)


def _wait(window, timeout=120.0):
    """Let every worker the window started finish and deliver its
    signals - also the ones a finished worker starts in turn.

    A batch fit (``_multi_fit``) starts its next fit from the previous
    fit's signal, so no worker running is not enough: under load the next
    fit can finish before its signal is delivered, and the batch is only
    done once ``_multi_fit`` is cleared."""
    names = ("identification_worker", "fit_worker", "fit_z_worker")
    deadline = time.monotonic() + timeout
    while True:
        for name in names:
            worker = getattr(window, name, None)
            if isinstance(worker, QtCore.QThread):
                worker.wait()
        QtWidgets.QApplication.processEvents()
        running = getattr(window, "_multi_fit", None) is not None
        for name in names:
            worker = getattr(window, name, None)
            if isinstance(worker, QtCore.QThread) and worker.isRunning():
                running = True
        if not running:
            QtWidgets.QApplication.processEvents()
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"the window's workers ran past {timeout} s")
        QtCore.QThread.msleep(5)


def _step(info, step):
    """The last entry of the metadata ``info`` recording ``step``."""
    return next(e for e in reversed(info) if e.get("Step") == step)


@pytest.fixture
def window(qt_offscreen):
    window = gui_localize.Window()
    yield window
    _wait(window)
    window.close()


def _memmap(movie, path):
    """``movie`` as the memory map the fit requires of a loaded movie."""
    mapped = np.memmap(path, dtype=movie.dtype, mode="w+", shape=movie.shape)
    mapped[:] = movie
    return mapped


@pytest.fixture
def loaded(window, tmp_path):
    _load(window, [_memmap(_spot_movie(), tmp_path / "movie.raw")], tmp_path)
    window.parameters_dialog.mng_slider.setValue(2000)
    return window


class TestLocalize:
    def test_identify_and_fit_saves_the_localizations(
        self, loaded, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        loaded.localize()
        _wait(loaded)
        assert messages.shown == []
        locs, info = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
        assert len(locs) == 2 * FRAMES
        # the metadata the identification and the fit returned
        assert _step(info, "Identify")["Box Size"] == (
            loaded.parameters["Box Size"]
        )
        assert info[-1]["Step"] == "Fit 2D"
        assert _status(loaded) == f"Saved {2 * FRAMES} localizations."
        # fit markers on screen
        assert loaded.locs_display is not None


# ---------------------------------------------------------------------------
# Movie metadata prompts
# ---------------------------------------------------------------------------


class TestPromptInfoDialog:
    def test_the_entered_metadata_is_read_back(self, window, monkeypatch):
        def fill(dialog):
            dialog.byte_order.setCurrentText("Big Endian")
            dialog.dtype.setCurrentText("uint16")
            dialog.frames.setValue(500)
            dialog.movie_height.setValue(64)
            dialog.movie_width.setValue(48)
            dialog.pixelsize.setValue(108.5)
            dialog.save.setChecked(False)

        _answer(monkeypatch, gui_localize.PromptInfoDialog, ACCEPTED, fill)
        info, save = window.prompt_info()
        assert info == {
            "Byte Order": ">",
            "Data Type": "uint16",
            "Frames": 500,
            "Height": 64,
            "Width": 48,
            "Pixelsize": 108.5,
        }
        assert not save

    def test_little_endian_is_the_default(self, window, monkeypatch):
        _answer(monkeypatch, gui_localize.PromptInfoDialog, ACCEPTED)
        info, save = window.prompt_info()
        assert info["Byte Order"] == "<"
        assert save

    def test_the_pixel_size_defaults_to_the_parameters(
        self, window, monkeypatch
    ):
        window.parameters_dialog.pixelsize.setValue(97.0)
        _answer(monkeypatch, gui_localize.PromptInfoDialog, ACCEPTED)
        info, _ = window.prompt_info()
        assert info["Pixelsize"] == 97.0

    def test_cancel_returns_nothing(self, window, monkeypatch):
        _answer(monkeypatch, gui_localize.PromptInfoDialog, REJECTED)
        assert window.prompt_info() is None


class TestPromptMovieInfoDialog:
    def test_prefilled_with_what_could_be_read(self, window, monkeypatch):
        partial = {"Frames": 300, "Height": 40, "Width": 50, "File": "a.tif"}
        info, save = None, None

        def check(dialog):
            assert dialog.frames.value() == 300
            assert dialog.movie_height.value() == 40
            assert dialog.movie_width.value() == 50
            dialog.pixelsize.setValue(160.0)

        _answer(
            monkeypatch, gui_localize.PromptMovieInfoDialog, ACCEPTED, check
        )
        info, save = window.prompt_movie_info(partial)
        assert info == {
            "Frames": 300,
            "Height": 40,
            "Width": 50,
            "File": "a.tif",
            "Pixelsize": 160.0,
            "Generated by": "Picasso Localize (manual metadata)",
        }
        assert save
        # the caller's dict is not changed
        assert "Pixelsize" not in partial

    def test_a_read_pixel_size_is_kept(self, window, monkeypatch):
        _answer(monkeypatch, gui_localize.PromptMovieInfoDialog, ACCEPTED)
        info, _ = window.prompt_movie_info({"Pixelsize": 65.0})
        assert info["Pixelsize"] == 65.0

    def test_nothing_read_starts_from_one_frame(self, window, monkeypatch):
        _answer(monkeypatch, gui_localize.PromptMovieInfoDialog, ACCEPTED)
        info, _ = window.prompt_movie_info()
        assert (info["Frames"], info["Height"], info["Width"]) == (1, 1, 1)

    def test_cancel_returns_nothing(self, window, monkeypatch):
        _answer(monkeypatch, gui_localize.PromptMovieInfoDialog, REJECTED)
        assert window.prompt_movie_info({"Frames": 3}) is None


class TestPromptChannelDialog:
    def test_the_selected_channel_is_returned(self, window, monkeypatch):
        _answer(
            monkeypatch,
            gui_localize.PromptChannelDialog,
            ACCEPTED,
            lambda dialog: dialog.byte_order.setCurrentIndex(1),
        )
        assert window.prompt_channel(["488", "561", "640"]) == "561"

    def test_cancel_returns_nothing(self, window, monkeypatch):
        _answer(monkeypatch, gui_localize.PromptChannelDialog, REJECTED)
        assert window.prompt_channel(["488"]) is None


class TestPromptForPath:
    @pytest.mark.parametrize(
        "name, prompt",
        [
            ("a.ims", "prompt_channel"),
            ("a.CZI", "prompt_channel"),
            ("a.lif", "prompt_channel"),
            ("a.tif", "prompt_movie_info"),
            ("a.nd2", "prompt_movie_info"),
            ("a.stk", "prompt_movie_info"),
            ("a.raw", "prompt_info"),
        ],
    )
    def test_each_format_gets_its_prompt(self, window, name, prompt):
        assert window._prompt_for_path(name) == getattr(window, prompt)


# ---------------------------------------------------------------------------
# Opening movies
# ---------------------------------------------------------------------------


def _tif_files(root, names):
    paths = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        paths.append(str(path))
    return paths


def _concatenate_dialog(paths, root=""):
    return gui_localize.ConcatenateMoviesDialog(None, paths, root=str(root))


def _select(dialog, *rows):
    dialog.file_list.clearSelection()
    for row in rows:
        dialog.file_list.item(row).setSelected(True)


class TestConcatenateMoviesDialog:
    def test_files_are_listed_relative_to_the_searched_folder(
        self, qt_offscreen, tmp_path
    ):
        paths = _tif_files(tmp_path, ["a/1.tif", "b/2.tif"])
        dialog = _concatenate_dialog(paths, tmp_path)
        item = dialog.file_list.item(0)
        assert item.text() == os.path.join("a", "1.tif")
        assert item.toolTip() == paths[0]
        assert dialog.paths() == paths
        assert dialog.count_label.text() == "2 file(s)"

    def test_without_a_folder_the_full_paths_are_shown(
        self, qt_offscreen, tmp_path
    ):
        paths = _tif_files(tmp_path, ["1.tif"])
        dialog = gui_localize.ConcatenateMoviesDialog(None, paths)
        assert dialog.file_list.item(0).text() == paths[0]

    def test_the_selection_moves_up_and_down(self, qt_offscreen, tmp_path):
        paths = _tif_files(tmp_path, ["1.tif", "2.tif", "3.tif", "4.tif"])
        dialog = _concatenate_dialog(paths, tmp_path)

        _select(dialog, 2, 3)
        dialog.move_up()
        assert dialog.paths() == [paths[0], paths[2], paths[3], paths[1]]
        # still selected, so it keeps moving
        dialog.move_up()
        assert dialog.paths() == [paths[2], paths[3], paths[0], paths[1]]
        # the top row cannot go higher: nothing moves
        dialog.move_up()
        assert dialog.paths() == [paths[2], paths[3], paths[0], paths[1]]

        dialog.move_down()
        dialog.move_down()
        assert dialog.paths() == [paths[0], paths[1], paths[2], paths[3]]
        dialog.move_down()
        assert dialog.paths() == [paths[0], paths[1], paths[2], paths[3]]

    def test_nothing_selected_moves_nothing(self, qt_offscreen, tmp_path):
        paths = _tif_files(tmp_path, ["1.tif", "2.tif"])
        dialog = _concatenate_dialog(paths, tmp_path)
        dialog.file_list.clearSelection()
        dialog.move_down()
        assert dialog.paths() == paths

    def test_removing_every_file_disables_ok(self, qt_offscreen, tmp_path):
        paths = _tif_files(tmp_path, ["1.tif", "2.tif"])
        dialog = _concatenate_dialog(paths, tmp_path)
        ok = dialog.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        _select(dialog, 0)
        dialog.remove_selected()
        assert dialog.paths() == [paths[1]]
        assert ok.isEnabled()
        _select(dialog, 0)
        dialog.remove_selected()
        assert dialog.count_label.text() == "0 file(s)"
        assert not ok.isEnabled()

    def test_added_files_are_appended_once(
        self, qt_offscreen, tmp_path, monkeypatch
    ):
        paths = _tif_files(tmp_path, ["1.tif", "2.tif"])
        other = _tif_files(tmp_path, ["elsewhere/3.tif"])
        dialog = _concatenate_dialog(paths[:1], tmp_path)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileNames",
            lambda *args, **kwargs: (paths + other, ""),
        )
        dialog.add_files()
        assert dialog.paths() == paths + other

    def test_get_paths_returns_the_order_and_the_answer(
        self, qt_offscreen, tmp_path, monkeypatch
    ):
        paths = _tif_files(tmp_path, ["1.tif", "2.tif"])

        def reverse(dialog):
            _select(dialog, 1)
            dialog.move_up()

        _answer(
            monkeypatch,
            gui_localize.ConcatenateMoviesDialog,
            ACCEPTED,
            reverse,
        )
        assert gui_localize.ConcatenateMoviesDialog.getPaths(
            None, paths, str(tmp_path)
        ) == (paths[::-1], True)
        _answer(monkeypatch, gui_localize.ConcatenateMoviesDialog, REJECTED)
        assert gui_localize.ConcatenateMoviesDialog.getPaths(None, paths) == (
            paths,
            False,
        )


class _LoadRecorder:
    """Stands in for ``Window._start_movie_load``."""

    def __init__(self, window, monkeypatch):
        self.calls = []
        monkeypatch.setattr(window, "_start_movie_load", self)

    def __call__(self, paths, prompt_for_path, **kwargs):
        self.calls.append((paths, prompt_for_path, kwargs))


def _choose_folder(monkeypatch, directory):
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: directory,
    )


class TestOpenConcatenated:
    def test_cancel_loads_nothing(self, window, monkeypatch):
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, "")
        window.open_concatenated_dialog()
        assert loads.calls == []

    def test_an_empty_folder_is_reported(self, window, tmp_path, monkeypatch):
        messages = _Messages(monkeypatch)
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, str(tmp_path))
        window.open_concatenated_dialog()
        assert messages.kinds == ["warning"]
        assert "No TIFF movies were found" in messages.text("warning")
        assert loads.calls == []

    def test_a_canceled_review_loads_nothing(
        self, window, tmp_path, monkeypatch
    ):
        _tif_files(tmp_path, ["1.tif", "2.tif"])
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, str(tmp_path))
        _answer(monkeypatch, gui_localize.ConcatenateMoviesDialog, REJECTED)
        window.open_concatenated_dialog()
        assert loads.calls == []

    def test_the_confirmed_order_is_loaded_as_one_movie(
        self, window, tmp_path, monkeypatch
    ):
        paths = _tif_files(tmp_path, ["a/1.tif", "b/2.tif"])
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, str(tmp_path))

        def reverse(dialog):
            _select(dialog, 1)
            dialog.move_up()

        _answer(
            monkeypatch,
            gui_localize.ConcatenateMoviesDialog,
            ACCEPTED,
            reverse,
        )
        window.open_concatenated_dialog()
        assert loads.calls == [
            (paths[::-1], window._prompt_for_path, {"concat": True})
        ]
        assert window.pwd == paths[1]


class TestOpenDialogs:
    def test_an_empty_micromanager_folder_is_reported(
        self, window, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, str(tmp_path))
        window.open_mm_folder_dialog()
        assert messages.kinds == ["warning"]
        assert loads.calls == []

    def test_a_micromanager_folder_opens_its_first_frame(
        self, window, tmp_path, monkeypatch
    ):
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, str(tmp_path))
        first = str(tmp_path / "img_000000000.tif")
        monkeypatch.setattr(io, "find_mm_separate_first", lambda d: first)
        window.open_mm_folder_dialog()
        assert [paths for paths, _, _ in loads.calls] == [[first]]

    def test_a_canceled_micromanager_folder_loads_nothing(
        self, window, monkeypatch
    ):
        loads = _LoadRecorder(window, monkeypatch)
        _choose_folder(monkeypatch, "")
        window.open_mm_folder_dialog()
        assert loads.calls == []

    def test_a_movie_file_is_opened(self, window, tmp_path, monkeypatch):
        loads = _LoadRecorder(window, monkeypatch)
        path = str(tmp_path / "movie.tif")
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (path, ""),
        )
        window.open_file_dialog()
        assert loads.calls == [([path], window._prompt_for_path, {})]
        assert window.pwd == path

    def test_a_multichannel_file_loads_every_channel(
        self, window, tmp_path, monkeypatch
    ):
        loads = _LoadRecorder(window, monkeypatch)
        path = str(tmp_path / "movie.czi")
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (path, ""),
        )
        window.open_multichannel_file_dialog()
        ((paths, prompt_for_path, kwargs),) = loads.calls
        assert paths == [path]
        assert kwargs == {"load_all": True}
        # every channel of the file is loaded: no channel is asked for
        assert prompt_for_path(path) == window.prompt_movie_info

    def test_several_files_load_as_channels(
        self, window, tmp_path, monkeypatch
    ):
        loads = _LoadRecorder(window, monkeypatch)
        paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileNames",
            lambda *args, **kwargs: (paths, ""),
        )
        window.open_channels_from_files_dialog()
        assert loads.calls == [
            (paths, window._prompt_for_path, {"multi_file": True})
        ]

    @pytest.mark.parametrize(
        "action, getter, empty",
        [
            ("open_file_dialog", "getOpenFileName", ("", "")),
            ("open_multichannel_file_dialog", "getOpenFileName", ("", "")),
            ("open_channels_from_files_dialog", "getOpenFileNames", ([], "")),
        ],
    )
    def test_cancel_loads_nothing(
        self, window, monkeypatch, action, getter, empty
    ):
        loads = _LoadRecorder(window, monkeypatch)
        monkeypatch.setattr(
            QtWidgets.QFileDialog, getter, lambda *args, **kwargs: empty
        )
        getattr(window, action)()
        assert loads.calls == []


# ---------------------------------------------------------------------------
# Drawing and moving ROIs with the mouse
# ---------------------------------------------------------------------------

LEFT = QtCore.Qt.MouseButton.LeftButton
RIGHT = QtCore.Qt.MouseButton.RightButton
MIDDLE = QtCore.Qt.MouseButton.MiddleButton
NO_BUTTON = QtCore.Qt.MouseButton.NoButton


def _mouse(view, kind, scene_xy, button, buttons=None):
    """Send a mouse event at scene position ``scene_xy`` straight to the
    view's handler for ``kind`` ("Press", "Move", "Release" or
    "DblClick")."""
    pos = QtCore.QPointF(view.mapFromScene(QtCore.QPointF(*scene_xy)))
    event_type = getattr(QtCore.QEvent.Type, f"MouseButton{kind}", None)
    if kind == "Move":
        event_type = QtCore.QEvent.Type.MouseMove
    event = QtGui.QMouseEvent(
        event_type,
        pos,
        view.viewport().mapToGlobal(pos),
        button,
        button if buttons is None else buttons,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    handler = {
        "Press": view.mousePressEvent,
        "Move": view.mouseMoveEvent,
        "Release": view.mouseReleaseEvent,
        "DblClick": view.mouseDoubleClickEvent,
    }[kind]
    handler(event)
    return event


def _drag(view, start, end, button=LEFT):
    _mouse(view, "Press", start, button)
    _mouse(view, "Move", end, button)
    return _mouse(view, "Release", end, button, buttons=NO_BUTTON)


@pytest.fixture
def shown(loaded):
    """The loaded window on screen, zoomed so that the 32 x 32 frame is
    several hundred device pixels wide."""
    loaded.resize(900, 700)
    loaded.show()
    QtWidgets.QApplication.processEvents()
    loaded.fit_in_view()
    return loaded


class TestDrawingRois:
    def test_a_drag_adds_an_roi(self, shown):
        _drag(shown.view, (4, 5), (20, 25))
        (((y0, x0), (y1, x1)),) = shown.view.rois
        assert (y0, x0) == pytest.approx((5, 4), abs=1)
        assert (y1, x1) == pytest.approx((25, 20), abs=1)
        assert shown.parameters_dialog.roi_field.text() == (
            f"{y0},{x0},{y1},{x1}"
        )

    def test_a_click_adds_nothing(self, shown):
        _drag(shown.view, (4, 5), (4.1, 5.1))
        assert shown.view.rois == []

    def test_overlapping_rois_are_clipped(self, shown):
        _drag(shown.view, (2, 2), (20, 20))
        _drag(shown.view, (10, 2), (30, 20))
        assert len(shown.view.rois) == 2
        (_, (_, first_x1)), ((_, second_x0), _) = sorted(shown.view.rois)
        assert first_x1 <= second_x0

    def test_a_double_click_removes_the_roi_under_it(self, shown):
        _drag(shown.view, (4, 4), (20, 20))
        _mouse(shown.view, "DblClick", (10, 10), LEFT)
        assert shown.view.rois == []
        # the release closing the double click does not add one back
        _mouse(shown.view, "Release", (10, 10), LEFT, buttons=NO_BUTTON)
        assert shown.view.rois == []
        assert not shown.view._suppress_release

    def test_a_double_click_beside_the_rois_keeps_them(self, shown):
        _drag(shown.view, (4, 4), (12, 12))
        event = _mouse(shown.view, "DblClick", (25, 25), LEFT)
        assert len(shown.view.rois) == 1
        assert not event.isAccepted()

    def test_a_right_drag_pans_the_view(self, shown):
        shown.zoom(4)
        before = shown.view.hscrollbar.value()
        _mouse(shown.view, "Press", (16, 16), RIGHT)
        assert shown.view.pan
        _mouse(shown.view, "Move", (12, 16), NO_BUTTON, buttons=RIGHT)
        _mouse(shown.view, "Release", (12, 16), RIGHT, buttons=NO_BUTTON)
        assert not shown.view.pan
        assert shown.view.hscrollbar.value() != before
        assert shown.view.rois == []

    def test_other_buttons_are_ignored(self, shown):
        for kind in ("Press", "Release"):
            event = _mouse(shown.view, kind, (16, 16), MIDDLE)
            assert not event.isAccepted()
        assert shown.view.rois == []


class TestSplitFovRegions:
    @pytest.fixture
    def split(self, shown):
        shown.parameters_dialog.split_fov_checkbox.setChecked(True)
        assert shown.view.split_fov_mode
        return shown

    def test_the_first_drag_sets_the_region_size(self, split):
        _drag(split.view, (1, 2), (13, 14))
        (((y0, x0), (y1, x1)),) = split.view.rois
        size = (y1 - y0, x1 - x0)
        assert size == pytest.approx((12, 12), abs=1)
        # a click drops a second region of the same size centered on it
        _drag(split.view, (24, 20), (24, 20))
        (y0, x0), (y1, x1) = split.view.rois[1]
        assert (y1 - y0, x1 - x0) == size
        assert ((y0 + y1) / 2, (x0 + x1) / 2) == pytest.approx((20, 24), abs=1)
        assert split.view.selected_roi == 1

    def test_a_first_click_sets_no_size(self, split):
        _drag(split.view, (5, 5), (5, 5))
        assert split.view.rois == []

    def test_a_region_is_kept_inside_the_frame(self, split):
        _drag(split.view, (1, 1), (11, 11))
        _drag(split.view, (31, 31), (31, 31))
        _, (y1, x1) = split.view.rois[1]
        assert (y1, x1) == (SIZE, SIZE)

    def test_dragging_a_region_moves_it(self, split):
        _drag(split.view, (2, 2), (12, 12))
        before = [list(map(list, r)) for r in split.view.rois]
        _mouse(split.view, "Press", (5, 5), LEFT)
        assert split.view._moving_roi == 0
        _mouse(split.view, "Move", (15, 9), LEFT)
        _mouse(split.view, "Release", (15, 9), LEFT, buttons=NO_BUTTON)
        assert split.view._moving_roi is None
        (((y0, x0), (y1, x1)),) = split.view.rois
        (by0, bx0), (by1, bx1) = before[0]
        assert (y0 - by0, x0 - bx0) == pytest.approx((4, 10), abs=1)
        assert (y1 - y0, x1 - x0) == (by1 - by0, bx1 - bx0)

    def test_arrow_keys_nudge_the_selected_region(self, split):
        _drag(split.view, (2, 2), (12, 12))
        (y0, x0), _ = split.view.rois[0]
        for key in (
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Up,
        ):
            split.view.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    key,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
            )
        assert split.view.rois[0][0] == [y0 - 1, x0 + 1]

    def test_a_double_click_removes_a_region_and_frees_the_size(self, split):
        _drag(split.view, (2, 2), (12, 12))
        _mouse(split.view, "DblClick", (5, 5), LEFT)
        _mouse(split.view, "Release", (5, 5), LEFT, buttons=NO_BUTTON)
        assert split.view.rois == []
        _drag(split.view, (2, 2), (22, 8))
        (((y0, x0), (y1, x1)),) = split.view.rois
        assert (y1 - y0, x1 - x0) == pytest.approx((6, 20), abs=1)


# ---------------------------------------------------------------------------
# The view: scale bar, zoom, export
# ---------------------------------------------------------------------------


def _scalebar_text(window):
    texts = [
        item.toPlainText()
        for item in window.scene.items()
        if isinstance(item, QtWidgets.QGraphicsTextItem)
    ]
    return texts[0] if len(texts) == 1 else texts


def _visible_width_nm(window):
    rect = window.view.mapToScene(window.view.viewport().rect()).boundingRect()
    return rect.width() * window.parameters_dialog.pixelsize.value()


class TestScaleBar:
    def test_hidden_by_default(self, shown):
        assert _scalebar_text(shown) == []

    @pytest.mark.parametrize(
        "pixelsize, step", [(0.5, 1), (5, 10), (50, 100), (500, 1000)]
    )
    def test_about_an_eighth_of_the_view_in_round_numbers(
        self, shown, pixelsize, step
    ):
        shown.parameters_dialog.pixelsize.setValue(pixelsize)
        shown.scalebar_action.setChecked(True)
        shown.draw_frame()
        text = _scalebar_text(shown)
        length = int(text.removesuffix(" nm"))
        assert length % step == 0
        assert length == pytest.approx(_visible_width_nm(shown) / 8, rel=0.5)

    def test_capped_at_ten_micrometers(self, shown):
        shown.parameters_dialog.pixelsize.setValue(9000)
        shown.scalebar_action.trigger()
        assert _scalebar_text(shown) == "10000 nm"

    def test_follows_the_zoom(self, shown):
        shown.scalebar_action.trigger()
        before = int(_scalebar_text(shown).removesuffix(" nm"))
        shown.zoom_in()
        shown.zoom_in()
        after = int(_scalebar_text(shown).removesuffix(" nm"))
        assert after < before

    def test_not_drawn_when_rounded_to_nothing(self, shown):
        """Zoomed far in, the bar would be 0 nm long."""
        shown.parameters_dialog.pixelsize.setValue(0.01)
        shown.scalebar_action.trigger()
        shown.zoom(8)
        assert _scalebar_text(shown) == []

    def test_not_drawn_before_the_view_is_laid_out(self, loaded, monkeypatch):
        loaded.scalebar_action.setChecked(True)
        monkeypatch.setattr(
            loaded.view.viewport(), "rect", lambda: QtCore.QRect()
        )
        loaded.draw_frame()
        assert _scalebar_text(loaded) == []


class TestZoom:
    def _width(self, window):
        return (
            window.view.mapToScene(window.view.viewport().rect())
            .boundingRect()
            .width()
        )

    def test_zooming_in_and_out(self, shown):
        fitted = self._width(shown)
        shown.zoom_in()
        assert self._width(shown) == pytest.approx(fitted * 0.7, rel=0.02)
        shown.zoom_out()
        assert self._width(shown) == pytest.approx(fitted, rel=0.02)

    def test_cannot_zoom_out_past_the_frame(self, shown):
        fitted = self._width(shown)
        shown.zoom_out()
        shown.zoom_out()
        assert self._width(shown) == pytest.approx(fitted, rel=0.02)

    def test_the_wheel_zooms_at_the_cursor(self, shown):
        view = shown.view
        anchor = view.mapFromScene(QtCore.QPointF(8, 8))
        event = QtGui.QWheelEvent(
            QtCore.QPointF(anchor),
            QtCore.QPointF(view.viewport().mapToGlobal(anchor)),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, -120),
            NO_BUTTON,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        fitted = self._width(shown)
        view.wheelEvent(event)
        assert self._width(shown) < fitted
        under = view.mapToScene(anchor)
        assert (under.x(), under.y()) == pytest.approx((8, 8), abs=0.5)

    def test_nothing_to_zoom_without_a_movie(self, window):
        transform = window.view.transform()
        window.zoom_in()
        window.fit_in_view()
        assert window.view.transform() == transform


class TestExportCurrentView:
    def test_needs_a_movie(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.export_current()
        assert messages.kinds == ["information"]

    def test_the_visible_part_is_saved(self, shown, tmp_path, monkeypatch):
        path = str(tmp_path / "view.png")
        saved_as = []

        def save_dialog(parent, title, default, **kwargs):
            saved_as.append(default)
            return path, "*.png"

        monkeypatch.setattr(
            gui_localize.lib, "get_save_filename_ext_dialog", save_dialog
        )
        shown.export_current()
        assert saved_as == [str(tmp_path / "Channel 0_view.png")]
        image = QtGui.QImage(path)
        assert not image.isNull()
        assert image.width() > SIZE

    def test_cancel_saves_nothing(self, shown, tmp_path, monkeypatch):
        monkeypatch.setattr(
            gui_localize.lib,
            "get_save_filename_ext_dialog",
            lambda *args, **kwargs: ("", ""),
        )
        shown.export_current()
        assert list(tmp_path.glob("*.png")) == []


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


class _Calls:
    """Stands in for a window method, recording its calls."""

    def __init__(self, window, monkeypatch, name):
        self.calls = []
        monkeypatch.setattr(window, name, self)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _blank(movie):
    """A movie of ``movie``'s shape with no spots in it."""
    return np.zeros_like(movie) + 100


def _status(window):
    return window.status_bar.currentMessage()


def _two_channels(window, tmp_path, second=None):
    movies = [
        _memmap(_spot_movie(), tmp_path / "a.raw"),
        _memmap(
            _spot_movie(seed=1) if second is None else second,
            tmp_path / "b.raw",
        ),
    ]
    _load(window, movies, tmp_path)
    window.parameters_dialog.mng_slider.setValue(2000)
    for channel in range(2):
        window.set_current_channel(channel)
        window.parameters_dialog.mng_slider.setValue(2000)
    window.set_current_channel(0)
    return window


class TestIdentify:
    def test_needs_a_movie(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.identify()
        assert messages.kinds == ["warning"]
        assert window.identifications is None

    def test_the_spots_are_counted(self, loaded):
        loaded.identify()
        _wait(loaded)
        assert len(loaded.identifications) == 2 * FRAMES
        assert loaded.ready_for_fit
        assert not loaded.identifications_outdated()
        assert _status(loaded).startswith(f"Identified {2 * FRAMES} spots")
        assert _status(loaded).endswith("Ready for fit.")
        assert not loaded.abort_action.isEnabled()

    def test_the_filters_are_named(self, loaded):
        parameters = dict(loaded.parameters)
        parameters["Temporal Median Window"] = 11
        parameters["Gaussian Filter Sigma"] = 1.5
        ids = pd.DataFrame({"frame": [0], "x": [1.0], "y": [1.0]})
        loaded.on_identify_finished(
            parameters, [], 0.5, ids, list(loaded.info), False, False, False
        )
        assert "; Temporal median: 11; Gaussian sigma: 1.5)" in _status(loaded)

    def test_finding_nothing_says_so_and_drops_the_last_spots(self, loaded):
        """Not the progress message of the identification, and no boxes
        of an earlier one, which were found with other settings."""
        loaded.identify()
        _wait(loaded)
        loaded.movie = _blank(loaded.movie)
        loaded.identify()
        _wait(loaded)
        assert loaded.identifications is None
        assert not loaded.ready_for_fit
        assert _status(loaded).startswith("No spots identified (Box Size: ")
        assert "Lower the minimum net gradient" in _status(loaded)

    def test_a_spline_calibration_follows(self, loaded, monkeypatch):
        build = _Calls(loaded, monkeypatch, "build_spline_calibration")
        loaded.identify(calibrate_spline=True)
        _wait(loaded)
        assert len(build.calls) == 1

    def test_a_calibration_without_beads_is_reported(
        self, loaded, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        build = _Calls(loaded, monkeypatch, "build_spline_calibration")
        loaded.movie = _blank(loaded.movie)
        loaded.identify(calibrate_spline=True)
        _wait(loaded)
        assert build.calls == []
        assert messages.kinds == ["information"]
        assert "No beads were identified" in messages.text("information")


class TestIdentifyAllChannels:
    def test_every_channel_is_identified(self, window, tmp_path):
        _two_channels(window, tmp_path)
        window.set_current_channel(1)
        window.identify()
        _wait(window)
        assert window.current_channel == 1
        counts = [len(c.identifications) for c in window.channels]
        assert counts == [2 * FRAMES, 2 * FRAMES]
        assert _status(window) == (
            f"Identified {4 * FRAMES} spots across 2 channels. Ready for fit."
        )
        assert not window.abort_action.isEnabled()

    def test_a_channel_without_spots_is_not_ready(self, window, tmp_path):
        _two_channels(window, tmp_path, second=_spot_movie(spots=()))
        window.identify()
        _wait(window)
        assert len(window.channels[0].identifications) == 2 * FRAMES
        assert window.channels[1].identifications is None
        assert not window.channels[1].ready_for_fit

    def test_the_fit_follows(self, window, tmp_path, monkeypatch):
        _two_channels(window, tmp_path)
        fit = _Calls(window, monkeypatch, "fit")
        window.identify(fit_afterwards=True)
        _wait(window)
        assert fit.calls == [((), {"calibrate_z": False})]

    def test_a_spline_calibration_follows(self, window, tmp_path, monkeypatch):
        _two_channels(window, tmp_path)
        build = _Calls(window, monkeypatch, "build_spline_calibration")
        window.identify(calibrate_spline=True)
        _wait(window)
        assert len(build.calls) == 1

    def test_no_beads_in_any_channel_is_reported(
        self, window, tmp_path, monkeypatch
    ):
        blank = _spot_movie(spots=())
        _load(
            window,
            [
                _memmap(blank, tmp_path / "a.raw"),
                _memmap(blank, tmp_path / "b.raw"),
            ],
            tmp_path,
        )
        messages = _Messages(monkeypatch)
        build = _Calls(window, monkeypatch, "build_spline_calibration")
        fit = _Calls(window, monkeypatch, "fit")
        window.identify(calibrate_spline=True, fit_afterwards=True)
        _wait(window)
        assert build.calls == fit.calls == []
        assert messages.kinds == ["information"]
        assert "in any channel" in messages.text("information")

    def test_an_abort_restores_the_active_channel(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        started = []
        monkeypatch.setattr(
            gui_localize.IdentificationWorker,
            "start",
            lambda worker: started.append(worker),
        )
        window.set_current_channel(1)
        window.identify()
        assert window.current_channel == 0
        assert window.abort_action.isEnabled()
        started[-1].aborted.emit()
        assert window.current_channel == 1
        assert window._multi_identify is None
        assert _status(window) == "Aborted."
        assert not window.abort_action.isEnabled()


# ---------------------------------------------------------------------------
# Multichannel spline fit
# ---------------------------------------------------------------------------

SPLINE_MODEL = "Experimental PSF (cubic spline)"


class _Signal:
    def connect(self, slot):
        pass


class _FakeFitWorker:
    """Stands in for a fit worker, recording what it was built with."""

    built = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        _FakeFitWorker.built.append(self)

    def __getattr__(self, name):
        return _Signal()

    def start(self):
        self.started = True

    def isRunning(self):
        return False


@pytest.fixture
def fake_spline_worker(monkeypatch):
    _FakeFitWorker.built = []
    monkeypatch.setattr(
        gui_localize, "MultichannelSplineFitWorker", _FakeFitWorker
    )
    return _FakeFitWorker.built


def _ids(xy, frames=(0,)):
    """Identifications at camera pixels ``xy`` in every frame of
    ``frames``."""
    rows = [(f, x, y) for f in frames for x, y in xy]
    return pd.DataFrame(
        {
            "frame": np.array([r[0] for r in rows], dtype=np.uint32),
            "x": np.array([r[1] for r in rows], dtype=np.int32),
            "y": np.array([r[2] for r in rows], dtype=np.int32),
            "net_gradient": np.full(len(rows), 5000.0, dtype=np.float32),
        }
    )


def _multichannel_calibration(n_channels=2, **extra):
    return {
        "model": "spline-3d-multichannel",
        "n_channels": n_channels,
        "n_data": [7, 7, 21],
        **extra,
    }


def _identified(window, tmp_path, second=True):
    """Two loaded channels, the reference identified and - if ``second`` -
    the other one too."""
    _two_channels(window, tmp_path)
    window.set_current_channel(1)
    window.identifications = _ids([(10, 8)]) if second else None
    window.ready_for_fit = second
    window.set_current_channel(0)
    window.identifications = _ids([(10, 8), (20, 22)])
    window.ready_for_fit = True
    return window


class TestMultichannelSplineFit:
    def test_needs_every_channel_of_the_calibration(
        self, window, tmp_path, monkeypatch, fake_spline_worker
    ):
        _identified(window, tmp_path)
        messages = _Messages(monkeypatch)
        window._start_multichannel_spline_fit(
            _multichannel_calibration(3), "spline"
        )
        assert messages.kinds == ["warning"]
        assert "expects 3 channels, but 2 are loaded" in messages.text(
            "warning"
        )
        assert fake_spline_worker == []

    def test_needs_the_reference_identified(
        self, window, tmp_path, monkeypatch, fake_spline_worker
    ):
        _two_channels(window, tmp_path)
        messages = _Messages(monkeypatch)
        window._start_multichannel_spline_fit(
            _multichannel_calibration(), "spline"
        )
        assert messages.kinds == ["warning"]
        assert "reference channel" in messages.text("warning")
        assert fake_spline_worker == []

    def test_every_channel_is_fitted_together(
        self, window, tmp_path, monkeypatch, fake_spline_worker
    ):
        _identified(window, tmp_path)
        messages = _Messages(monkeypatch)
        calibration = _multichannel_calibration()
        window._start_multichannel_spline_fit(
            calibration, "spline-mle", eps=1e-3, max_it=50
        )
        assert messages.shown == []
        (worker,) = fake_spline_worker
        assert worker.started
        movies, camera_infos, reference, box, used = worker.args
        assert [m is c.movie for m, c in zip(movies, window.channels)] == [
            True,
            True,
        ]
        assert len(reference) == 2
        assert used is calibration
        assert worker.kwargs["mle"]
        assert not worker.kwargs["use_gpu"]
        assert worker.kwargs["link_identifications"]
        assert [len(i) for i in worker.kwargs["identifications_per_channel"]]
        assert (worker.kwargs["eps"], worker.kwargs["max_it"]) == (1e-3, 50)
        assert window.abort_action.isEnabled()

    def test_an_unidentified_channel_is_reported(
        self, window, tmp_path, monkeypatch, fake_spline_worker
    ):
        _identified(window, tmp_path, second=False)
        messages = _Messages(monkeypatch)
        window._start_multichannel_spline_fit(
            _multichannel_calibration(), "spline"
        )
        assert messages.kinds == ["information"]
        assert "1 of the 1 non-reference channels" in messages.text(
            "information"
        )
        # the fit still runs, on the channels that are identified
        assert fake_spline_worker[0].started

    def test_spots_found_on_the_channel_sum_are_not_linked(
        self, window, tmp_path, monkeypatch, fake_spline_worker
    ):
        _identified(window, tmp_path, second=False)
        window.sum_identifications = window.identifications
        messages = _Messages(monkeypatch)
        window._start_multichannel_spline_fit(
            _multichannel_calibration(), "spline"
        )
        assert messages.shown == []
        assert not fake_spline_worker[0].kwargs["link_identifications"]

    def test_affine_corrections_are_not_applied_on_top(
        self, window, tmp_path, monkeypatch, fake_spline_worker
    ):
        _identified(window, tmp_path)
        window.parameters_dialog.lateral_transforms = [{"type": "affine"}]
        messages = _Messages(monkeypatch)
        window._start_multichannel_spline_fit(
            _multichannel_calibration(), "spline"
        )
        assert messages.kinds == ["information"]
        assert "single-channel data only" in messages.text("information")
        assert fake_spline_worker[0].started


class TestSplineBoxSize:
    @pytest.fixture
    def spline(self, window, tmp_path):
        _identified(window, tmp_path)
        dialog = window.parameters_dialog
        dialog.fit_model.setCurrentText(SPLINE_MODEL)
        dialog.spline_calibration = _multichannel_calibration()
        return window

    def _answer_question(self, monkeypatch, button):
        asked = []

        def question(parent, title, text, *args, **kwargs):
            asked.append(text)
            return button

        monkeypatch.setattr(QtWidgets.QMessageBox, "question", question)
        return asked

    def test_a_box_that_fits_the_calibration_is_fitted(
        self, spline, monkeypatch, fake_spline_worker
    ):
        spline.parameters_dialog.box_spinbox.setValue(5)
        spline.identifications = _ids([(10, 8)])
        spline.last_identification_info = None
        spline.ready_for_fit = True
        spline.fit()
        assert fake_spline_worker[0].started

    def test_a_larger_box_is_offered_the_calibration_box(
        self, spline, monkeypatch, fake_spline_worker
    ):
        spline.parameters_dialog.box_spinbox.setValue(9)
        spline.ready_for_fit = True
        asked = self._answer_question(
            monkeypatch, QtWidgets.QMessageBox.StandardButton.Yes
        )
        spline.fit()
        assert "larger than this spline calibration" in asked[0]
        assert spline.parameters_dialog.box_spinbox.value() == 7
        assert fake_spline_worker == []

    def test_declining_keeps_the_box(
        self, spline, monkeypatch, fake_spline_worker
    ):
        spline.parameters_dialog.box_spinbox.setValue(9)
        spline.ready_for_fit = True
        self._answer_question(
            monkeypatch, QtWidgets.QMessageBox.StandardButton.No
        )
        spline.fit()
        assert spline.parameters_dialog.box_spinbox.value() == 9
        assert fake_spline_worker == []
        assert _status(spline) == ""


class TestSplitFovSplineFit:
    REGIONS = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]

    @pytest.fixture
    def split(self, loaded):
        loaded.parameters_dialog.split_fov_checkbox.setChecked(True)
        loaded.view.rois = [list(map(list, r)) for r in self.REGIONS]
        return loaded

    def _calibration(self):
        return _multichannel_calibration(
            split_fov=True, regions=self.REGIONS, reference=0
        )

    def test_needs_identifications(
        self, split, monkeypatch, fake_spline_worker
    ):
        messages = _Messages(monkeypatch)
        split._start_multichannel_spline_fit(self._calibration(), "spline")
        assert messages.kinds == ["warning"]
        assert "Identify spots first" in messages.text("warning")
        assert fake_spline_worker == []

    def test_needs_spots_in_the_reference_region(
        self, split, monkeypatch, fake_spline_worker
    ):
        split.identifications = _ids([(20, 10), (25, 12)])
        messages = _Messages(monkeypatch)
        split._start_multichannel_spline_fit(self._calibration(), "spline")
        assert messages.kinds == ["warning"]
        assert "2 spots were identified but none fall inside" in (
            messages.text("warning")
        )
        assert fake_spline_worker == []

    def test_the_drawn_regions_place_the_channels(
        self, split, monkeypatch, fake_spline_worker
    ):
        split.identifications = _ids([(8, 10)])
        messages = _Messages(monkeypatch)
        split._start_multichannel_spline_fit(
            self._calibration(), "spline-mle-gpu"
        )
        assert messages.shown == []
        (worker,) = fake_spline_worker
        assert worker.kwargs["split_fov"]
        assert worker.kwargs["regions"] == self.REGIONS
        assert worker.kwargs["use_gpu"] and worker.kwargs["mle"]

    def test_the_calibration_places_the_channels_otherwise(
        self, split, monkeypatch, fake_spline_worker
    ):
        split.view.rois = [[[0, 0], [32, 16]]]
        split.identifications = _ids([(8, 10)])
        split._start_multichannel_spline_fit(self._calibration(), "spline")
        assert fake_spline_worker[0].kwargs["regions"] is None


# ---------------------------------------------------------------------------
# Drift correction after the fit
# ---------------------------------------------------------------------------


def _drift_frame(n_frames=FRAMES):
    return pd.DataFrame(
        {
            "x": np.linspace(0, 10, n_frames),
            "y": np.linspace(0, -5, n_frames),
        }
    )


class TestDriftCorrection:
    @pytest.fixture
    def undrifting(self, loaded, monkeypatch):
        """AIM replaced by a known drift: the movie is far too short for
        the real one."""
        calls = []

        def fake_aim(locs, info, segmentation, progress=None):
            calls.append(segmentation)
            locs = locs.copy()
            locs["x"] -= 1.0
            return locs, info, _drift_frame()

        monkeypatch.setattr(gui_localize.aim, "aim", fake_aim)
        loaded.aim_calls = calls
        return loaded

    def _localize(self, window, aim=False, fiducials=False):
        dialog = window.parameters_dialog
        dialog.aim_undrift_checkbox.setChecked(aim)
        dialog.aim_segmentation.setValue(10)
        dialog.fiducial_check.setChecked(fiducials)
        window.localize()
        _wait(window)

    def test_aim_saves_the_undrifted_locs_and_the_drift(
        self, undrifting, tmp_path
    ):
        self._localize(undrifting, aim=True)
        assert undrifting.aim_calls == [10]
        locs, _ = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
        undrifted, _ = io.load_locs(
            str(tmp_path / "Channel 0_locs_undrifted.hdf5")
        )
        assert np.allclose(undrifted["x"], locs["x"] - 1.0)
        drift = np.loadtxt(str(tmp_path / "Channel 0_locs_drift.txt"))
        assert np.allclose(drift[:, 0], _drift_frame()["x"])
        assert _status(undrifting) == (
            "Saved drift-corrected localizations and drift file."
        )
        # the fit markers stay where the spots are on screen
        assert undrifting.locs_display is not undrifting.locs

    def test_the_still_spots_are_used_as_fiducials(self, loaded, tmp_path):
        self._localize(loaded, fiducials=True)
        undrifted, info = io.load_locs(
            str(tmp_path / "Channel 0_locs_undrifted.hdf5")
        )
        assert len(undrifted) == 2 * FRAMES
        drift = np.loadtxt(str(tmp_path / "Channel 0_locs_drift.txt"))
        assert drift.shape == (FRAMES, 2)

    def test_aim_and_fiducial_drifts_add_up(
        self, undrifting, tmp_path, monkeypatch
    ):
        fiducial_drift = _drift_frame() * 0 + 2.0
        monkeypatch.setattr(
            gui_localize.postprocess,
            "undrift_from_fiducials",
            lambda locs, info, picks, pick_size: (
                locs,
                info,
                fiducial_drift,
            ),
        )
        self._localize(undrifting, aim=True, fiducials=True)
        drift = np.loadtxt(str(tmp_path / "Channel 0_locs_drift.txt"))
        assert np.allclose(drift[:, 0], _drift_frame()["x"] + 2.0)
        assert np.allclose(drift[:, 1], _drift_frame()["y"] + 2.0)

    def test_no_fiducials_saves_no_undrifted_locs(
        self, loaded, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            gui_localize.imageprocess,
            "find_fiducials",
            lambda locs, info: ([], 900.0),
        )
        self._localize(loaded, fiducials=True)
        assert (tmp_path / "Channel 0_locs.hdf5").exists()
        assert not (tmp_path / "Channel 0_locs_undrifted.hdf5").exists()
        assert _status(loaded) == (
            "No fiducials found. Skipping fiducial-based drift correction."
        )

    def test_no_fiducials_still_saves_the_aim_result(
        self, undrifting, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            gui_localize.imageprocess,
            "find_fiducials",
            lambda locs, info: ([], 900.0),
        )
        self._localize(undrifting, aim=True, fiducials=True)
        assert (tmp_path / "Channel 0_locs_undrifted.hdf5").exists()
        drift = np.loadtxt(str(tmp_path / "Channel 0_locs_drift.txt"))
        assert np.allclose(drift[:, 0], _drift_frame()["x"])
        assert _status(undrifting).startswith(
            "Saved AIM drift-corrected localizations. No fiducials found"
        )


# ---------------------------------------------------------------------------
# Loading movies in the background
# ---------------------------------------------------------------------------


def _raw(path, movie=None):
    """Write ``movie`` (default: the spot movie) as a .raw file without
    metadata, so that opening it asks for the movie info."""
    movie = _spot_movie() if movie is None else movie
    movie.astype("<u2").tofile(str(path))
    return str(path)


def _describe_raw(monkeypatch, save=True, movie_shape=(FRAMES, SIZE, SIZE)):
    """Answer the movie info prompt of a .raw file."""
    asked = []

    def fill(dialog):
        asked.append(dialog)
        dialog.dtype.setCurrentText("uint16")
        dialog.frames.setValue(movie_shape[0])
        dialog.movie_height.setValue(movie_shape[1])
        dialog.movie_width.setValue(movie_shape[2])
        dialog.pixelsize.setValue(117.0)
        dialog.save.setChecked(save)

    _answer(monkeypatch, gui_localize.PromptInfoDialog, ACCEPTED, fill)
    return asked


def _wait_for_load(window):
    """Serve the load's queued signals (and prompts) until it is done."""
    for _ in range(2000):
        QtWidgets.QApplication.processEvents()
        if window._load_thread is None and not window._detached_loads:
            return
        QtCore.QThread.msleep(5)
    raise AssertionError("the movie load did not finish")


class TestLoadingMovies:
    def test_a_raw_movie_asks_for_its_info(
        self, window, tmp_path, monkeypatch
    ):
        path = _raw(tmp_path / "movie.raw")
        asked = _describe_raw(monkeypatch)
        window.open(path)
        _wait_for_load(window)
        assert len(asked) == 1
        assert window.movie.shape == (FRAMES, SIZE, SIZE)
        assert np.array_equal(window.movie[3], _spot_movie()[3])
        assert window.parameters_dialog.pixelsize.value() == 117.0
        assert window.channels[0].name == "Channel 0"
        assert _status(window).startswith("Opened movie in")
        # entered once, reused next time
        assert io.load_info(path)[0]["Frames"] == FRAMES
        assert window._load_progress is None

    def test_the_info_is_not_saved_on_request(
        self, window, tmp_path, monkeypatch
    ):
        path = _raw(tmp_path / "movie.raw")
        _describe_raw(monkeypatch, save=False)
        window.open(path)
        _wait_for_load(window)
        assert window.movie is not None
        assert not (tmp_path / "movie.yaml").exists()

    def test_a_canceled_prompt_loads_nothing(
        self, window, tmp_path, monkeypatch
    ):
        path = _raw(tmp_path / "movie.raw")
        _answer(monkeypatch, gui_localize.PromptInfoDialog, REJECTED)
        window.open(path)
        _wait_for_load(window)
        assert window.movie is None
        assert window.channels == []

    def test_files_opened_as_channels_are_named_after_them(
        self, window, tmp_path, monkeypatch
    ):
        paths = [
            _raw(tmp_path / "green.raw"),
            _raw(tmp_path / "red.raw", _spot_movie(seed=3)),
        ]
        _describe_raw(monkeypatch)
        window.open_channels_from_files(paths)
        _wait_for_load(window)
        assert [c.name for c in window.channels] == ["green", "red"]
        assert window.channel_combo.isVisibleTo(window)
        assert _status(window).startswith("Opened 2 channel(s) in")

    def test_concatenated_files_make_one_movie(
        self, window, tmp_path, monkeypatch
    ):
        import tifffile

        movie = _spot_movie()
        paths = []
        for i, part in enumerate((movie[:12], movie[12:])):
            path = str(tmp_path / f"part{i}.tif")
            tifffile.imwrite(path, part)
            paths.append(path)
        _answer(
            monkeypatch,
            gui_localize.PromptMovieInfoDialog,
            ACCEPTED,
            lambda dialog: dialog.pixelsize.setValue(117.0),
        )
        window._start_movie_load(paths, window._prompt_for_path, concat=True)
        _wait_for_load(window)
        assert window.movie.shape == (FRAMES, SIZE, SIZE)
        assert np.array_equal(window.movie[15], movie[15])

    def test_an_unreadable_file_is_reported(
        self, window, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        window.open(str(tmp_path / "missing.tif"))
        _wait_for_load(window)
        assert messages.kinds == ["warning"]
        assert messages.shown[0][1] == "Could not load file"
        assert window.movie is None

    def test_cancel_detaches_the_load(self, window, tmp_path, monkeypatch):
        """Canceling while the loader waits for an answer gives the window
        back at once and drops what the loader still delivers."""
        path = _raw(tmp_path / "movie.raw")

        def cancel(dialog):
            window._load_progress.canceled.emit()

        _answer(monkeypatch, gui_localize.PromptInfoDialog, ACCEPTED, cancel)
        window.open(path)
        _wait_for_load(window)
        assert window._load_progress is None
        assert window._load_worker is None
        assert window.movie is None
        assert _status(window).startswith("Load canceled")

    def test_a_second_load_waits_for_the_first(
        self, window, tmp_path, monkeypatch
    ):
        first = _raw(tmp_path / "first.raw")
        second = _raw(tmp_path / "second.raw")
        opened = []

        def fill(dialog):
            opened.append(dialog)
            if len(opened) == 1:
                window.open(second)  # ignored: a load is running
            dialog.dtype.setCurrentText("uint16")
            dialog.frames.setValue(FRAMES)
            dialog.movie_height.setValue(SIZE)
            dialog.movie_width.setValue(SIZE)

        _answer(monkeypatch, gui_localize.PromptInfoDialog, ACCEPTED, fill)
        window.open(first)
        _wait_for_load(window)
        assert len(opened) == 1
        assert window.channels[0].path == first


class TestChannelNames:
    def test_the_metadata_channel_comes_first(self, window):
        info = [{"Channel": "Cy3B"}]
        assert window._channel_name(info, "a.tif", 1, True) == "Cy3B"

    def test_separate_files_are_named_after_the_file(self, window):
        assert window._channel_name([{}], "/x/red.tif", 1, True) == "red"

    def test_otherwise_numbered(self, window):
        assert window._channel_name([{}], "/x/a.ims", 2) == "Channel 2"
        assert window._channel_name(None, "/x/a.ims", 0) == "Channel 0"


# ---------------------------------------------------------------------------
# Identifications from files: picks, localizations, saved identifications
# ---------------------------------------------------------------------------

PIXELSIZE = 130.0


def _open_file(monkeypatch, *paths):
    """Answer the open file dialogs in turn with ``paths``."""
    answers = list(paths)
    asked = []

    def get_open_file_name(parent, caption, *args, **kwargs):
        asked.append(caption)
        return answers.pop(0), ""

    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getOpenFileName", get_open_file_name
    )
    return asked


def _save_file(monkeypatch, path):
    """Answer the save dialog with ``path``, recording the suggestion."""
    suggested = []

    def save_dialog(parent, title, default, **kwargs):
        suggested.append(default)
        return path, ""

    monkeypatch.setattr(
        gui_localize.lib, "get_save_filename_ext_dialog", save_dialog
    )
    return suggested


@pytest.fixture
def spotted(loaded):
    loaded.parameters_dialog.pixelsize.setValue(PIXELSIZE)
    return loaded


class TestNeedsAMovie:
    @pytest.mark.parametrize(
        "action", ["open_picks", "open_locs", "open_identifications"]
    )
    def test_identifications_need_a_movie(self, window, monkeypatch, action):
        messages = _Messages(monkeypatch)
        getattr(window, action)()
        assert messages.kinds == ["warning"]
        assert "Load a movie" in messages.text("warning")

    @pytest.mark.parametrize(
        "action", ["save_spots_dialog", "save_locs_dialog"]
    )
    def test_saving_needs_a_movie(self, window, monkeypatch, action):
        messages = _Messages(monkeypatch)
        getattr(window, action)()
        assert messages.kinds == ["information"]


class TestPicksAsIdentifications:
    def _picks(self, tmp_path, shape="Circle"):
        path = str(tmp_path / "picks.yaml")
        centers = [(x * PIXELSIZE, y * PIXELSIZE) for y, x in SPOTS]
        io.save_picks(path, centers, shape, size=500.0)
        return path

    def test_every_frame_of_each_pick_is_identified(
        self, spotted, tmp_path, monkeypatch
    ):
        picks = self._picks(tmp_path)
        asked = _open_file(monkeypatch, picks, "")
        spotted.open_picks()
        assert asked == ["Open picks", "Open drift file"]
        ids = spotted.identifications
        assert len(ids) == 2 * FRAMES
        # at the pick centers, in camera pixels
        assert sorted(set(zip(ids["y"], ids["x"]))) == pytest.approx(
            sorted(SPOTS)
        )
        assert spotted.ready_for_fit
        assert not spotted.identifications_outdated()
        assert _status(spotted) == (
            f"Created a total of {2 * FRAMES} identifications."
        )

    def test_the_drift_moves_the_picks(self, spotted, tmp_path, monkeypatch):
        picks = self._picks(tmp_path)
        drift_path = str(tmp_path / "drift.txt")
        drift = pd.DataFrame(
            {"x": np.full(FRAMES, 2 * PIXELSIZE), "y": np.zeros(FRAMES)}
        )
        io.save_drift(drift_path, drift)
        _open_file(monkeypatch, picks, drift_path)
        spotted.open_picks()
        xs = sorted(set(spotted.identifications["x"]))
        assert xs == pytest.approx(sorted(x + 2 for _, x in SPOTS))

    def test_an_unreadable_drift_file_is_skipped(
        self, spotted, tmp_path, monkeypatch
    ):
        picks = self._picks(tmp_path)
        broken = tmp_path / "drift.txt"
        broken.write_text("not a drift file")
        messages = _Messages(monkeypatch)
        _open_file(monkeypatch, picks, str(broken))
        spotted.open_picks()
        assert messages.kinds == ["warning"]
        assert "No drift correction will be applied" in messages.text(
            "warning"
        )
        assert len(spotted.identifications) == 2 * FRAMES

    def test_only_circles_are_supported(self, spotted, tmp_path, monkeypatch):
        picks = self._picks(tmp_path, shape="Square")
        messages = _Messages(monkeypatch)
        _open_file(monkeypatch, picks, "")
        spotted.open_picks()
        assert messages.kinds == ["warning"]
        assert "Only circle picks" in messages.text("warning")
        assert spotted.identifications is None

    def test_cancel_loads_nothing(self, spotted, monkeypatch):
        asked = _open_file(monkeypatch, "")
        spotted.open_picks()
        assert asked == ["Open picks"]
        assert spotted.identifications is None


class TestLocsAsIdentifications:
    def _locs(self, tmp_path):
        path = str(tmp_path / "old_locs.hdf5")
        locs = pd.DataFrame(
            {
                "frame": np.array([10], dtype=np.uint32),
                "x": np.array([22.0 * PIXELSIZE], dtype=np.float32),
                "y": np.array([20.0 * PIXELSIZE], dtype=np.float32),
                "lpx": np.array([10.0], dtype=np.float32),
                "lpy": np.array([10.0], dtype=np.float32),
            }
        )
        info = [
            {
                "Frames": FRAMES,
                "Height": SIZE,
                "Width": SIZE,
                "Pixelsize": PIXELSIZE,
            }
        ]
        io.save_locs(path, locs, info)
        return path

    def _frames_around(self, monkeypatch, value, ok=True):
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getInt",
            lambda *args, **kwargs: (value, ok),
        )

    def test_the_frames_around_each_event_are_identified(
        self, spotted, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        _open_file(monkeypatch, self._locs(tmp_path))
        self._frames_around(monkeypatch, 2)
        spotted.open_locs()
        assert messages.shown == []
        ids = spotted.identifications
        assert list(ids["frame"]) == [8, 9, 10, 11, 12]
        assert set(ids["x"]) == {22}
        assert spotted.ready_for_fit

    def test_cancel_loads_nothing(self, spotted, tmp_path, monkeypatch):
        _Messages(monkeypatch)
        _open_file(monkeypatch, self._locs(tmp_path))
        self._frames_around(monkeypatch, 2, ok=False)
        spotted.open_locs()
        assert spotted.identifications is None

    def test_an_unreadable_file_is_reported(
        self, spotted, tmp_path, monkeypatch
    ):
        broken = tmp_path / "broken.hdf5"
        broken.write_text("not hdf5")
        messages = _Messages(monkeypatch)
        _open_file(monkeypatch, str(broken))
        spotted.open_locs()
        assert messages.kinds
        assert spotted.identifications is None


class TestSavedIdentifications:
    def test_saved_and_opened_again(self, loaded, tmp_path, monkeypatch):
        loaded.parameters_dialog.box_spinbox.setValue(9)
        loaded.identify()
        _wait(loaded)
        found = loaded.identifications
        path = str(tmp_path / "ids.hdf5")
        suggested = _save_file(monkeypatch, path)
        loaded.save_identifications_dialog()
        assert suggested == [str(tmp_path / "Channel 0_identifications.hdf5")]

        loaded.parameters_dialog.box_spinbox.setValue(5)
        loaded.parameters_dialog.mng_slider.setValue(100)
        loaded.identifications = None
        _open_file(monkeypatch, path)
        loaded.open_identifications()
        assert loaded.parameters_dialog.box_spinbox.value() == 9
        assert loaded.parameters_dialog.mng_slider.value() == 2000
        assert len(loaded.identifications) == len(found)
        assert not loaded.identifications_outdated()

    def test_nothing_to_save(self, loaded, monkeypatch):
        messages = _Messages(monkeypatch)
        loaded.save_identifications_dialog()
        assert messages.kinds == ["warning"]

    def test_an_unreadable_file_is_reported(
        self, loaded, tmp_path, monkeypatch
    ):
        broken = tmp_path / "broken.hdf5"
        broken.write_text("not hdf5")
        messages = _Messages(monkeypatch)
        _open_file(monkeypatch, str(broken))
        loaded.open_identifications()
        assert messages.kinds
        assert loaded.identifications is None


class TestSavingSpotsAndLocs:
    @pytest.fixture
    def identified(self, loaded):
        loaded.identify()
        _wait(loaded)
        return loaded

    def test_spots_need_identifications(self, loaded, monkeypatch):
        messages = _Messages(monkeypatch)
        _save_file(monkeypatch, "spots.tif")
        loaded.save_spots_dialog()
        assert messages.kinds == ["warning"]

    @pytest.mark.parametrize("ext", [".tif", ".npy"])
    def test_spots_are_saved_with_their_info(
        self, identified, tmp_path, monkeypatch, ext
    ):
        path = str(tmp_path / f"spots{ext}")
        suggested = _save_file(monkeypatch, path)
        identified.save_spots_dialog()
        assert suggested == [str(tmp_path / "Channel 0_spots.tif")]
        if ext == ".npy":
            spots = np.load(path)
        else:
            import imageio

            spots = np.asarray(imageio.mimread(path))
        box = identified.parameters["Box Size"]
        assert spots.shape == (2 * FRAMES, box, box)
        info = io.load_info(path)
        assert _step(info, "Identify")["Box Size"] == box

    def test_spots_canceled(self, identified, tmp_path, monkeypatch):
        _save_file(monkeypatch, "")
        identified.save_spots_dialog()
        assert list(tmp_path.glob("*spots*")) == []

    def test_locs_are_saved_where_chosen(self, loaded, tmp_path, monkeypatch):
        loaded.localize()
        _wait(loaded)
        path = str(tmp_path / "chosen_locs.hdf5")
        suggested = _save_file(monkeypatch, path)
        loaded.save_locs_dialog()
        assert suggested == [str(tmp_path / "Channel 0_locs.hdf5")]
        locs, info = io.load_locs(path)
        assert len(locs) == 2 * FRAMES
        # the fit's code for "2D elliptical Gaussian, Least squares"
        assert _step(info, "Fit 2D")["Fit method"] == "gausslq"

    def test_locs_canceled(self, loaded, tmp_path, monkeypatch):
        _save_file(monkeypatch, "")
        loaded.save_locs_dialog()
        assert list(tmp_path.glob("*.hdf5")) == []


# ---------------------------------------------------------------------------
# The parameters dialog: frame range and ROI field
# ---------------------------------------------------------------------------


class TestFrameRange:
    def _enter(self, window, text):
        edit = window.parameters_dialog.frames_edit
        edit.setText(text)
        edit.editingFinished.emit()

    def test_segments_are_stored_zero_indexed_and_clamped(self, loaded):
        """The movie is a memory map (a .raw file), whose truth value
        numpy refuses to give."""
        self._enter(loaded, "2,10; 15,500")
        assert loaded.frame_range == [[1, 9], [14, FRAMES - 1]]
        assert loaded.parameters_dialog.frames_edit.text() == (
            f"2,10; 15,{FRAMES}"
        )

    def test_an_emptied_field_means_every_frame(self, loaded):
        self._enter(loaded, "2,10")
        self._enter(loaded, "")
        assert loaded.frame_range is None

    def test_identification_keeps_to_the_range(self, loaded):
        self._enter(loaded, "3,7")
        loaded.identify()
        _wait(loaded)
        assert sorted(set(loaded.identifications["frame"])) == list(
            range(2, 7)
        )
        assert loaded.last_identification_info["Frame bounds"] == [[2, 6]]

    def test_ignored_without_a_movie(self, window):
        self._enter(window, "2,10")
        assert window.frame_range is None


class TestRoiField:
    def _enter(self, window, text):
        field = window.parameters_dialog.roi_field
        field.setText(text)
        field.editingFinished.emit()

    def test_a_typed_roi_is_used(self, loaded):
        self._enter(loaded, "2,3,20,25")
        assert loaded.view.rois == [[[2, 3], [20, 25]]]

    def test_incomplete_input_is_waited_for(self, loaded):
        self._enter(loaded, "2,3,20")
        assert loaded.view.rois == []

    def test_emptying_the_field_clears_the_rois(self, loaded):
        self._enter(loaded, "2,3,20,25")
        self._enter(loaded, "")
        assert loaded.view.rois == []

    def test_several_rois_are_only_counted(self, shown):
        _drag(shown.view, (1, 1), (12, 12))
        _drag(shown.view, (18, 18), (30, 30))
        field = shown.parameters_dialog.roi_field
        assert field.text() == "2 ROIs"
        assert field.isReadOnly()


# ---------------------------------------------------------------------------
# Re-aligning the channels of a spline calibration from the signal
# ---------------------------------------------------------------------------


def _registered_calibration(n_channels=2, **extra):
    from picasso import transforms

    identity = transforms.identity("affine").to_dict()
    return _multichannel_calibration(
        n_channels, channel_transforms=[identity] * n_channels, **extra
    )


def _frame_specs(monkeypatch, ok=True):
    asked = []

    def get_frame_specs(parent, n_frames, frame_range=None, model=None):
        asked.append((n_frames, frame_range, model))
        return [[0, n_frames - 1]], 5, "affine", ok

    monkeypatch.setattr(
        gui_localize.RefineRegistrationDialog,
        "getFrameSpecs",
        staticmethod(get_frame_specs),
    )
    return asked


REG_INFO = [
    {"channel": 1, "n_matches": 42, "rms": 0.123, "model": "affine"},
    {
        "channel": 2,
        "n_matches": 7,
        "rms": 0.5,
        "model": "translation",
        "model_requested": "affine",
    },
]


class TestReregisterFromSignal:
    TITLE = "Re-align channels (signal)"

    def test_needs_a_registered_calibration(self, loaded, monkeypatch):
        messages = _Messages(monkeypatch)
        loaded.reregister_channels_from_signal()
        assert messages.kinds == ["warning"]
        assert "Load a multichannel or split-FOV" in messages.text("warning")

    def test_needs_every_channel_movie(self, window, tmp_path, monkeypatch):
        _two_channels(window, tmp_path)
        window.parameters_dialog.spline_calibration = _registered_calibration(
            3
        )
        messages = _Messages(monkeypatch)
        asked = _frame_specs(monkeypatch)
        window.reregister_channels_from_signal()
        assert messages.kinds == ["warning"]
        assert "has 3 channels, but 2 movies" in messages.text("warning")
        assert asked == []

    def test_cancel_changes_nothing(self, window, tmp_path, monkeypatch):
        _two_channels(window, tmp_path)
        window.parameters_dialog.spline_calibration = _registered_calibration()
        refined = []
        monkeypatch.setattr(
            gui_localize.spline,
            "refine_multichannel_transforms_from_signal",
            lambda *args, **kwargs: refined.append(args),
        )
        asked = _frame_specs(monkeypatch, ok=False)
        window.reregister_channels_from_signal()
        assert asked == [(FRAMES, None, "affine")]
        assert refined == []

    def test_the_channels_are_refined_and_summarized(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        calibration = _registered_calibration()
        window.parameters_dialog.spline_calibration = calibration
        refined = []

        def refine(movies, calibration, **kwargs):
            refined.append((movies, calibration, kwargs))
            return calibration, REG_INFO

        monkeypatch.setattr(
            gui_localize.spline,
            "refine_multichannel_transforms_from_signal",
            refine,
        )
        _frame_specs(monkeypatch)
        messages = _Messages(monkeypatch)
        window.reregister_channels_from_signal()
        ((movies, used, kwargs),) = refined
        assert [m is c.movie for m, c in zip(movies, window.channels)] == [
            True,
            True,
        ]
        assert used is calibration
        assert kwargs["frame_bounds"] == [[0, FRAMES - 1]]
        assert kwargs["max_frames"] == 5
        assert messages.kinds == ["information"]
        text = messages.text("information")
        assert "ch1: 42 paired signals, RMS 0.12 camera px, affine" in text
        assert "translation (fell back from affine)" in text

    def test_a_failure_is_reported(self, window, tmp_path, monkeypatch):
        _two_channels(window, tmp_path)
        window.parameters_dialog.spline_calibration = _registered_calibration()

        def refine(*args, **kwargs):
            raise ValueError("too few signals")

        monkeypatch.setattr(
            gui_localize.spline,
            "refine_multichannel_transforms_from_signal",
            refine,
        )
        _frame_specs(monkeypatch)
        messages = _Messages(monkeypatch)
        window.reregister_channels_from_signal()
        assert messages.kinds == ["critical"]
        assert "Re-alignment failed: too few signals" in messages.text(
            "critical"
        )
        assert QtWidgets.QApplication.overrideCursor() is None
        assert _status(window) == ""


class TestReregisterSplitFovFromSignal:
    REGIONS = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]

    @pytest.fixture
    def split(self, loaded):
        loaded.parameters_dialog.split_fov_checkbox.setChecked(True)
        loaded.parameters_dialog.spline_calibration = _registered_calibration(
            split_fov=True, regions=self.REGIONS, reference=0
        )
        return loaded

    def test_needs_a_movie(self, window, monkeypatch):
        window.parameters_dialog.spline_calibration = _registered_calibration(
            split_fov=True, regions=self.REGIONS
        )
        messages = _Messages(monkeypatch)
        window.reregister_channels_from_signal()
        assert messages.shown == [
            ("information", "Re-align channels (signal)", "No movie loaded.")
        ]

    def test_needs_one_region_per_channel(self, split, monkeypatch):
        split.parameters_dialog.spline_calibration["regions"] = self.REGIONS[
            :1
        ]
        messages = _Messages(monkeypatch)
        split.reregister_channels_from_signal()
        assert messages.kinds == ["warning"]
        assert "2 regions are needed" in messages.text("warning")

    def test_the_regions_follow_the_new_registration(self, split, monkeypatch):
        split.view.rois = [list(map(list, r)) for r in self.REGIONS]
        swapped = self.REGIONS[::-1]
        seen = []

        def refine(movie, calibration, regions, **kwargs):
            seen.append(regions)
            calibration["regions"] = swapped
            return calibration, REG_INFO[:1]

        monkeypatch.setattr(
            gui_localize.spline,
            "refine_split_fov_transforms_from_signal",
            refine,
        )
        _frame_specs(monkeypatch)
        messages = _Messages(monkeypatch)
        split.reregister_channels_from_signal()
        assert seen == [self.REGIONS]
        assert split.view.rois == swapped
        assert messages.kinds == ["information"]


# ---------------------------------------------------------------------------
# Cameras from the config
# ---------------------------------------------------------------------------

CAMERA_CONFIG = {
    "CameraPriority": ["iXon", "Missing"],
    "Cameras": {
        "Zyla": {
            "Pixelsize": 108,
            "Baseline": 100,
            "Sensitivity Categories": ["ReadoutRate", "DynamicRange"],
            "Sensitivity": {
                "540 MHz": {"12-bit": 7.18, "16-bit": 0.46},
                "200 MHz": {"12-bit": 7.0, "16-bit": 0.45},
            },
            "Channel Device": {
                "Name": "Turret-Label",
                "Emission Wavelengths": {"1-488": 525, "2-640": 700},
            },
        },
        "iXon": {
            "Pixelsize": 160,
            "Baseline": 200,
            "DefaultGain": 50,
            "Gain Property Name": "Gain",
            "EM Switch Property": {"Name": "EMSwitch", True: "On"},
            "Sensitivity": 4.5,
        },
        # a camera configured without a sensitivity
        "Prime": {"Pixelsize": 65, "Baseline": 50},
    },
}


@pytest.fixture
def cameras(qt_offscreen, monkeypatch):
    monkeypatch.setattr(gui_localize, "CONFIG", CAMERA_CONFIG)
    window = gui_localize.Window()
    yield window.parameters_dialog
    window.close()


def _select_camera(dialog, name):
    dialog.camera.setCurrentIndex(dialog.camera.findText(name))


def _camera_info(camera, **mm):
    return {
        "Camera": camera,
        "Micro-Manager Metadata": {
            f"{camera}-{key}" if key != "Turret-Label" else key: value
            for key, value in mm.items()
        },
    }


class TestCameraConfig:
    def test_the_priority_cameras_come_first(self, cameras):
        names = [cameras.camera.itemText(i) for i in range(3)]
        assert names[0] == "iXon"
        assert sorted(names[1:]) == ["Prime", "Zyla"]

    def test_the_first_camera_sets_the_photon_conversion(self, cameras):
        assert cameras.baseline.value() == 200
        assert cameras.gain.value() == 50
        assert cameras.sensitivity.value() == 4.5
        assert cameras.pixelsize.value() == 160

    def test_a_camera_without_a_sensitivity_keeps_the_last_one(self, cameras):
        _select_camera(cameras, "Prime")
        assert cameras.baseline.value() == 50
        assert cameras.pixelsize.value() == 65
        assert cameras.sensitivity.value() == 4.5
        # no EM gain configured
        assert cameras.gain.value() == 1

    def test_the_sensitivity_follows_the_categories(self, cameras):
        _select_camera(cameras, "Zyla")
        rate, dynamic = cameras.cam_combos["Zyla"]
        rate.setCurrentText("540 MHz")
        dynamic.setCurrentText("16-bit")
        assert cameras.sensitivity.value() == pytest.approx(0.46)
        rate.setCurrentText("200 MHz")
        dynamic.setCurrentText("12-bit")
        assert cameras.sensitivity.value() == pytest.approx(7.0)
        assert [dynamic.itemText(i) for i in range(dynamic.count())] == [
            "12-bit",
            "16-bit",
        ]

    def test_the_wavelengths_of_the_channel_device_are_offered(self, cameras):
        combo = cameras.emission_combos["Zyla"]
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "525",
            "700",
        ]
        assert "iXon" not in cameras.emission_combos


class TestCameraFromMetadata:
    def test_an_unknown_camera_changes_nothing(self, cameras):
        cameras.set_camera_parameters({"Camera": "Other"})
        cameras.set_camera_parameters({})
        assert cameras.camera.currentText() == "iXon"

    def test_the_em_gain_is_read_when_switched_on(self, cameras):
        cameras.set_camera_parameters(
            _camera_info("iXon", Gain="300", EMSwitch="On")
        )
        assert cameras.gain.value() == 300

    def test_no_em_gain_when_switched_off(self, cameras):
        cameras.set_camera_parameters(
            _camera_info("iXon", Gain="300", EMSwitch="Off")
        )
        assert cameras.gain.value() == 1

    def test_missing_gain_properties_leave_the_gain(self, cameras):
        cameras.set_camera_parameters(_camera_info("iXon", EMSwitch="On"))
        assert cameras.gain.value() == 50
        cameras.set_camera_parameters(_camera_info("iXon", Gain="300"))
        assert cameras.gain.value() == 50

    def test_the_readout_and_the_wavelength_are_read(self, cameras):
        cameras.set_camera_parameters(
            _camera_info(
                "Zyla",
                ReadoutRate="200 MHz",
                DynamicRange="16-bit",
                **{"Turret-Label": "2-640"},
            )
        )
        assert cameras.camera.currentText() == "Zyla"
        assert cameras.sensitivity.value() == pytest.approx(0.45)
        assert cameras.emission_combos["Zyla"].currentText() == "700"
        assert cameras.baseline.value() == 100
        assert cameras.pixelsize.value() == 108

    def test_an_unknown_filter_leaves_the_wavelength(self, cameras):
        cameras.set_camera_parameters(
            _camera_info("Zyla", **{"Turret-Label": "9-UV"})
        )
        assert cameras.emission_combos["Zyla"].currentText() == "525"
        cameras.set_camera_parameters(_camera_info("Zyla"))
        assert cameras.emission_combos["Zyla"].currentText() == "525"

    def test_a_loaded_movie_selects_its_camera(self, cameras, tmp_path):
        window = cameras.window
        movie = _spot_movie()
        info = [
            {
                "Frames": FRAMES,
                "Height": SIZE,
                "Width": SIZE,
                **_camera_info("Zyla", ReadoutRate="540 MHz"),
            }
        ]
        window._set_channels([movie], [info], [str(tmp_path / "a.tif")], ["a"])
        assert cameras.camera.currentText() == "Zyla"
        assert cameras.pixelsize.value() == 108


# ---------------------------------------------------------------------------
# The ROI table
# ---------------------------------------------------------------------------


def _roi_dialog(window):
    window.parameters_dialog.on_edit_rois()
    return window.parameters_dialog.roi_dialog


def _row(dialog, row):
    return [
        dialog.table.item(row, col).text()
        for col in range(dialog.table.columnCount())
    ]


class TestRoiDialog:
    def test_add_puts_a_quarter_frame_roi_below_the_last(self, loaded):
        dialog = _roi_dialog(loaded)
        dialog.on_add()
        dialog.on_add()
        assert loaded.view.rois == [
            [[0, 0], [8, 8]],
            [[10, 10], [18, 18]],
        ]
        assert _row(dialog, 1) == ["10", "10", "18", "18"]
        assert loaded.parameters_dialog.roi_field.text() == "2 ROIs"

    def test_an_edited_cell_moves_the_roi(self, loaded):
        dialog = _roi_dialog(loaded)
        dialog.on_add()
        dialog.table.item(0, 2).setText("20")
        assert loaded.view.rois == [[[0, 0], [20, 8]]]

    def test_a_cell_that_is_not_a_number_is_waited_for(self, loaded):
        dialog = _roi_dialog(loaded)
        dialog.on_add()
        dialog.table.item(0, 2).setText("2x")
        assert loaded.view.rois == [[[0, 0], [8, 8]]]

    def test_selecting_a_row_highlights_its_roi(self, loaded):
        dialog = _roi_dialog(loaded)
        dialog.on_add()
        dialog.on_add()
        dialog.table.selectRow(1)
        assert loaded.view.selected_roi == 1
        dialog.table.clearSelection()
        assert loaded.view.selected_roi is None

    def test_remove_deletes_the_selected_rows(self, loaded):
        dialog = _roi_dialog(loaded)
        for _ in range(3):
            dialog.on_add()
        dialog.table.selectRow(1)
        dialog.on_remove()
        assert loaded.view.rois == [
            [[0, 0], [8, 8]],
            [[20, 20], [28, 28]],
        ]
        assert loaded.view.selected_roi is None
        assert dialog.table.rowCount() == 2

    def test_clear_means_the_whole_frame(self, loaded):
        dialog = _roi_dialog(loaded)
        dialog.on_add()
        dialog.on_clear()
        assert loaded.view.rois == []
        assert loaded.parameters_dialog.roi_field.text() == ""

    def test_without_a_movie_the_roi_is_50_pixels(self, window):
        dialog = _roi_dialog(window)
        dialog.on_add()
        assert window.view.rois == [[[0, 0], [50, 50]]]


class TestSplitFovRoiDialog:
    @pytest.fixture
    def regions(self, loaded):
        loaded.parameters_dialog.split_fov_checkbox.setChecked(True)
        loaded.view.rois = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]
        loaded.parameters_dialog.mng_slider.setValue(1500)
        return _roi_dialog(loaded)

    def test_each_region_shows_its_threshold(self, regions):
        assert regions.table.columnCount() == 5
        assert _row(regions, 1)[4] == "1500"

    def test_an_edited_threshold_is_the_region_s(self, regions):
        window = regions.window
        regions.table.item(1, 4).setText("3000")
        assert window.region_mngs() == [1500, 3000]

    def test_removing_a_region_drops_its_threshold(self, regions):
        window = regions.window
        regions.table.item(1, 4).setText("3000")
        regions.table.selectRow(0)
        regions.on_remove()
        assert window.view.rois == [[[0, 16], [32, 32]]]
        assert window.region_mngs() == [3000]


# ---------------------------------------------------------------------------
# Navigation, metadata, config folder
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_go_to_a_frame(self, loaded, monkeypatch):
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getInt",
            lambda *args, **kwargs: (7, True),
        )
        loaded.to_frame()
        assert loaded.curr_frame_number == 6

    def test_go_to_canceled(self, loaded, monkeypatch):
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getInt",
            lambda *args, **kwargs: (7, False),
        )
        loaded.to_frame()
        assert loaded.curr_frame_number == 0

    def test_steps_stay_inside_the_movie(self, loaded):
        loaded.previous_frame()
        assert loaded.curr_frame_number == 0
        loaded.next_frame(5)
        assert loaded.curr_frame_number == 5
        loaded.next_frame(100)
        assert loaded.curr_frame_number == FRAMES - 1
        loaded.next_frame()
        assert loaded.curr_frame_number == FRAMES - 1
        loaded.previous_frame(100)
        assert loaded.curr_frame_number == 0
        loaded.last_frame()
        assert loaded.curr_frame_number == FRAMES - 1
        loaded.first_frame()
        assert loaded.curr_frame_number == 0

    def test_nothing_to_navigate_without_a_movie(self, window, monkeypatch):
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getInt",
            lambda *args, **kwargs: pytest.fail("asked without a movie"),
        )
        window.to_frame()
        window.next_frame()
        window.last_frame()
        assert window.movie is None


class TestMetadata:
    def test_needs_a_movie(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.show_metadata()
        assert messages.kinds == ["information"]

    def test_shows_the_movie_info(self, loaded):
        loaded.show_metadata()
        assert loaded.metadata_dialog.isVisible()

    def test_shows_the_saved_info_after_a_fit(self, loaded):
        loaded.localize()
        _wait(loaded)
        shown = []
        loaded.metadata_dialog.set_infos = (
            lambda infos, labels=None: shown.append(infos)
        )
        loaded.show_metadata()
        assert shown == [loaded.extra_info]


class TestConfigLocation:
    def test_the_config_folder_is_opened(self, window, tmp_path, monkeypatch):
        import picasso

        config = tmp_path / "settings" / "config.yaml"
        monkeypatch.setattr(picasso, "resolve_config_path", lambda: None)
        monkeypatch.setattr(picasso, "config_filename", lambda: str(config))
        opened = []
        monkeypatch.setattr(
            QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url)
        )
        window.open_config_location()
        # Qt reports local files with forward slashes, also on Windows
        assert [os.path.normpath(url.toLocalFile()) for url in opened] == [
            str(config.parent)
        ]
        # created, so that the user can put the config there
        assert config.parent.is_dir()


# ---------------------------------------------------------------------------
# Multichannel Gaussian fit
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gauss_worker(monkeypatch):
    _FakeFitWorker.built = []
    monkeypatch.setattr(
        gui_localize, "MultichannelGaussianFitWorker", _FakeFitWorker
    )
    return _FakeFitWorker.built


def _registration(n_channels=2, **extra):
    from picasso import transforms

    identity = transforms.identity("affine").to_dict()
    return {
        "channel_transforms": [identity] * n_channels,
        "n_channels": n_channels,
        **extra,
    }


def _link_photons(window, linked):
    window.parameters_dialog.gauss_link_photons_checkbox.setChecked(linked)


class TestMultichannelGaussFit:
    def test_needs_every_channel_of_the_registration(
        self, window, tmp_path, monkeypatch, fake_gauss_worker
    ):
        _identified(window, tmp_path)
        messages = _Messages(monkeypatch)
        window._start_multichannel_gauss_fit(_registration(3), "gausslq")
        assert messages.kinds == ["warning"]
        assert "expects 3 channels, but 2 are loaded" in messages.text(
            "warning"
        )
        assert fake_gauss_worker == []

    def test_needs_the_reference_identified(
        self, window, tmp_path, monkeypatch, fake_gauss_worker
    ):
        _two_channels(window, tmp_path)
        messages = _Messages(monkeypatch)
        window._start_multichannel_gauss_fit(_registration(), "gausslq")
        assert messages.kinds == ["warning"]
        assert "reference channel" in messages.text("warning")
        assert fake_gauss_worker == []

    def test_decoupled_photons_need_few_channels(
        self, window, tmp_path, monkeypatch, fake_gauss_worker
    ):
        _identified(window, tmp_path)
        _link_photons(window, False)
        messages = _Messages(monkeypatch)
        window._start_multichannel_gauss_fit(_registration(1), "gausslq")
        assert messages.kinds == ["warning"]
        assert "supports 2 to 6 channels" in messages.text("warning")
        assert fake_gauss_worker == []

    def test_every_channel_is_fitted_together(
        self, window, tmp_path, monkeypatch, fake_gauss_worker
    ):
        _identified(window, tmp_path, second=False)
        window.parameters_dialog.lateral_transforms = [{"type": "affine"}]
        _link_photons(window, True)
        messages = _Messages(monkeypatch)
        registration = _registration()
        window._start_multichannel_gauss_fit(
            registration, "gaussmle", use_gpu=True, eps=1e-4, max_it=30
        )
        # the corrections are not applied, and the unidentified channel
        # cannot be linked
        assert messages.kinds == ["information", "information"]
        assert "Lateral corrections" in messages.shown[0][2]
        assert "1 of the 1 non-reference channels" in messages.shown[1][2]
        (worker,) = fake_gauss_worker
        assert worker.started
        assert worker.args[4] is registration
        assert worker.kwargs["mle"] and worker.kwargs["use_gpu"]
        assert worker.kwargs["link_photons"]
        assert worker.kwargs["link_identifications"]


class TestSplitFovGaussFit:
    REGIONS = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]

    @pytest.fixture
    def split(self, loaded):
        loaded.parameters_dialog.split_fov_checkbox.setChecked(True)
        loaded.view.rois = [list(map(list, r)) for r in self.REGIONS]
        return loaded

    def _registration(self, **extra):
        return _registration(
            split_fov=True,
            regions=self.REGIONS,
            reference=0,
            **extra,
        )

    def test_needs_identifications(
        self, split, monkeypatch, fake_gauss_worker
    ):
        messages = _Messages(monkeypatch)
        split._start_multichannel_gauss_fit(self._registration(), "gausslq")
        assert messages.kinds == ["warning"]
        assert "Identify spots first" in messages.text("warning")

    def test_needs_spots_in_the_reference_region(
        self, split, monkeypatch, fake_gauss_worker
    ):
        split.identifications = _ids([(20, 10)])
        messages = _Messages(monkeypatch)
        split._start_multichannel_gauss_fit(self._registration(), "gausslq")
        assert messages.kinds == ["warning"]
        assert "none fall inside the reference region" in messages.text(
            "warning"
        )
        assert fake_gauss_worker == []

    def test_decoupled_photons_need_few_regions(
        self, split, monkeypatch, fake_gauss_worker
    ):
        split.identifications = _ids([(8, 10)])
        _link_photons(split, False)
        registration = self._registration()
        registration["n_channels"] = 7
        messages = _Messages(monkeypatch)
        split._start_multichannel_gauss_fit(registration, "gausslq")
        assert messages.kinds == ["warning"]
        assert "but this registration has 7" in messages.text("warning")

    def test_the_drawn_regions_place_the_channels(
        self, split, monkeypatch, fake_gauss_worker
    ):
        split.identifications = _ids([(8, 10)])
        messages = _Messages(monkeypatch)
        split._start_multichannel_gauss_fit(self._registration(), "gausslq")
        assert messages.shown == []
        (worker,) = fake_gauss_worker
        assert worker.kwargs["split_fov"]
        assert worker.kwargs["regions"] == self.REGIONS
        assert not worker.kwargs["mle"]


# ---------------------------------------------------------------------------
# Single-channel fit: refusals, 3D
# ---------------------------------------------------------------------------


class TestFitRefusals:
    def test_needs_a_movie(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.fit()
        assert messages.shown == [
            ("warning", "Fit", "Load a movie before fitting.")
        ]

    def test_needs_identifications(self, loaded, monkeypatch):
        messages = _Messages(monkeypatch)
        loaded.fit()
        assert messages.kinds == ["warning"]
        assert "No identifications available" in messages.text("warning")

    def test_a_spline_fit_needs_a_calibration(self, loaded, monkeypatch):
        loaded.identify()
        _wait(loaded)
        loaded.parameters_dialog.fit_model.setCurrentText(SPLINE_MODEL)
        messages = _Messages(monkeypatch)
        loaded.fit()
        assert messages.kinds == ["warning"]
        assert "Load a spline PSF calibration first" in messages.text(
            "warning"
        )
        assert _status(loaded) == ""

    def test_a_failed_fit_says_why(self, loaded, monkeypatch):
        """Not only "Aborted.": the reason is in the error the fit
        raised, which used to go to the console only."""
        loaded.identify()
        _wait(loaded)

        def fail(*args, **kwargs):
            raise ValueError("the movie is not a movie")

        monkeypatch.setattr(gui_localize.localize, "fit", fail)
        messages = _Messages(monkeypatch)
        loaded.fit()
        _wait(loaded)
        assert messages.shown == [
            ("warning", "Fit failed", "the movie is not a movie")
        ]
        assert _status(loaded) == "Fit failed."
        assert not loaded.abort_action.isEnabled()

    def test_a_failed_fit_of_one_channel_ends_the_batch_and_says_why(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        _fit_separately(window)
        window.identify()
        _wait(window)

        def fail(*args, **kwargs):
            raise ValueError("the movie is not a movie")

        monkeypatch.setattr(gui_localize.localize, "fit", fail)
        messages = _Messages(monkeypatch)
        window.fit()
        _wait(window)
        assert messages.text("warning") == "the movie is not a movie"
        assert window._multi_fit is None

    def test_localize_needs_a_movie(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.localize()
        window.localize(calibrate_z=True)
        assert [title for _, title, _ in messages.shown] == [
            "Localize",
            "Calibrate astigmatism",
        ]

    def test_astigmatism_needs_the_elliptical_gaussian(
        self, loaded, monkeypatch
    ):
        loaded.parameters_dialog.fit_model.setCurrentText(
            "2D rotated elliptical Gaussian"
        )
        messages = _Messages(monkeypatch)
        loaded.localize(calibrate_z=True)
        assert messages.kinds == ["warning"]
        assert "requires the '2D elliptical Gaussian'" in messages.text(
            "warning"
        )
        assert loaded.identifications is None


class TestAstigmatism:
    def _specs(self, monkeypatch, ok=True):
        monkeypatch.setattr(
            gui_localize.Calibrate3DDialog,
            "getCalibrationSpecs",
            staticmethod(lambda parent=None: (10.0, 2, "", 1, ok)),
        )

    def test_canceled_specs_save_no_calibration(
        self, loaded, tmp_path, monkeypatch
    ):
        self._specs(monkeypatch, ok=False)
        _save_file(monkeypatch, str(tmp_path / "calib.yaml"))
        loaded.localize(calibrate_z=True)
        _wait(loaded)
        assert len(loaded.locs) == 2 * FRAMES
        assert not (tmp_path / "calib.yaml").exists()
        # the localizations of a calibration stack are not saved either
        assert not (tmp_path / "Channel 0_locs.hdf5").exists()

    def test_the_calibration_is_built_from_the_fit(
        self, loaded, tmp_path, monkeypatch
    ):
        self._specs(monkeypatch)
        path = str(tmp_path / "calib.yaml")
        suggested = _save_file(monkeypatch, path)
        built = []

        def calibrate_z(
            locs, info, *, d, magnification_factor, path, **kwargs
        ):
            built.append((len(locs), d, path, kwargs))

        monkeypatch.setattr(gui_localize.zfit, "calibrate_z", calibrate_z)
        loaded.localize(calibrate_z=True)
        _wait(loaded)
        assert suggested == [str(tmp_path / "Channel 0_3d_calib.yaml")]
        ((n_locs, step, used_path, kwargs),) = built
        assert (n_locs, step, used_path) == (2 * FRAMES, 10.0, path)
        assert kwargs["frames_per_step"] == 2
        assert _status(loaded).startswith("3D calibrated in")

    def test_specs_that_do_not_fit_the_stack_are_reported(
        self, loaded, tmp_path, monkeypatch
    ):
        self._specs(monkeypatch)
        _save_file(monkeypatch, str(tmp_path / "calib.yaml"))

        def calibrate_z(*args, **kwargs):
            raise ValueError("Not enough frames per step.")

        monkeypatch.setattr(gui_localize.zfit, "calibrate_z", calibrate_z)
        messages = _Messages(monkeypatch)
        loaded.localize(calibrate_z=True)
        _wait(loaded)
        assert messages.shown == [
            ("warning", "3D calibration", "Not enough frames per step.")
        ]

    def test_identifications_are_reused_for_the_calibration(
        self, loaded, monkeypatch
    ):
        loaded.identify()
        _wait(loaded)
        fit = _Calls(loaded, monkeypatch, "fit")
        identify = _Calls(loaded, monkeypatch, "identify")
        loaded.localize(calibrate_z=True)
        assert fit.calls == [((), {"calibrate_z": True})]
        assert identify.calls == []


class TestFitZ:
    def test_z_is_fitted_and_saved(self, loaded, tmp_path):
        from tests.conftest import CALIB_3D

        path = str(tmp_path / "3d_calib.yaml")
        io.save_yaml(path, dict(CALIB_3D))
        loaded.parameters_dialog.update_z_calib(path)
        assert loaded.parameters_dialog.fit_z_checkbox.isChecked()
        loaded.localize()
        _wait(loaded)
        locs, info = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
        assert "z" in locs.columns
        assert len(locs) == 2 * FRAMES
        assert _step(info, "Fit 3D")["Calibration path"] == path
        assert _status(loaded) == (f"Saved {2 * FRAMES} localizations.")

    def test_a_missing_calibration_file_is_shown(self, loaded, tmp_path):
        dialog = loaded.parameters_dialog
        dialog.update_z_calib(str(tmp_path / "gone.yaml"))
        assert dialog.z_calibration == {}
        assert dialog.z_calib_label.text() == (
            "-- calibration path not found --"
        )
        assert not dialog.fit_z_checkbox.isEnabled()


# ---------------------------------------------------------------------------
# Identifying on the channel sum
# ---------------------------------------------------------------------------


def _identify_on(window, mode):
    window.parameters_dialog.identify_mode_combo.setCurrentText(mode)


class TestChannelSum:
    def test_needs_two_channels(self, loaded, monkeypatch):
        messages = _Messages(monkeypatch)
        loaded.identify_channel_sum()
        assert messages.kinds == ["information"]
        assert "Summing needs at least two channels" in messages.text(
            "information"
        )

    def test_unregistered_channels_are_added_as_they_are(
        self, window, tmp_path
    ):
        _two_channels(window, tmp_path)
        _identify_on(window, gui_localize.IDENTIFY_MODE_UNREGISTERED_SUM)
        window.identify()
        _wait(window)
        ids = window.sum_identifications
        assert ids is not None and len(ids)
        # every channel is fitted at them
        assert all(c.identifications is ids for c in window.channels)
        # the identification records the sum: unregistered, every channel
        # stands where it is
        info = _step(window.identifications_info, "Identify")
        assert info["Identification Mode"] == "sum"
        assert info["Sum Channel Transforms"] == (
            [transforms.identity("affine").to_dict()] * 2
        )
        assert all(
            c.identifications_info is window.identifications_info
            for c in window.channels
        )
        assert _status(window).startswith(
            f"Identified {len(ids)} spots on the sum of 2 channels"
        )

    def test_the_loaded_registration_places_the_channels(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        registration = _registration()
        # a small shift of the second channel, so that the recorded
        # transforms are the registration's rather than identities
        registration["channel_transforms"] = [
            transforms.identity("affine").to_dict(),
            affine([[1.0, 0.0, 13.0], [0.0, 1.0, 0.0]]).to_dict(),
        ]
        window.parameters_dialog.channel_registration_calibration = (
            registration
        )
        fit = _Calls(window, monkeypatch, "fit")
        _identify_on(window, gui_localize.IDENTIFY_MODE_SUM)
        window.set_current_channel(1)
        window.localize()
        _wait(window)
        # the sum is in the reference channel's coordinates
        assert window.current_channel == 0
        assert window.channels[1].identifications is None
        info = _step(window.identifications_info, "Identify")
        assert info["Identification Mode"] == "sum"
        assert info["Sum Channel Transforms"] == (
            registration["channel_transforms"]
        )
        assert fit.calls == [((), {})]

    def test_channels_that_cannot_be_registered_are_reported(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path, second=_spot_movie(spots=()))
        _identify_on(window, gui_localize.IDENTIFY_MODE_SUM)
        messages = _Messages(monkeypatch)
        window.identify()
        _wait(window)
        assert messages.kinds == ["warning"]
        assert "could not be registered" in messages.text("warning")
        assert window.sum_identifications is None
        assert not window.abort_action.isEnabled()

    def test_nothing_found_on_the_sum(self, window, tmp_path):
        blank = _spot_movie(spots=())
        _load(
            window,
            [
                _memmap(blank, tmp_path / "a.raw"),
                _memmap(blank, tmp_path / "b.raw"),
            ],
            tmp_path,
        )
        _identify_on(window, gui_localize.IDENTIFY_MODE_UNREGISTERED_SUM)
        window.identify()
        _wait(window)
        assert window.sum_identifications is None
        assert not window.ready_for_fit
        assert _status(window).startswith("No spots found on the channel sum")

    def test_an_abort_drops_the_sum(self, window, tmp_path, monkeypatch):
        _two_channels(window, tmp_path)
        _identify_on(window, gui_localize.IDENTIFY_MODE_UNREGISTERED_SUM)
        started = []
        monkeypatch.setattr(
            gui_localize.IdentificationWorker,
            "start",
            lambda worker: started.append(worker),
        )
        window.identify()
        assert window.abort_action.isEnabled()
        started[-1].aborted.emit()
        assert window._sum_identify is None
        assert window.sum_identifications is None
        assert not window.abort_action.isEnabled()
        assert _status(window) == "Aborted."


class TestSplitFovChannelSum:
    REGIONS = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]

    def test_the_regions_are_registered_from_their_spots_first(
        self, window, tmp_path, monkeypatch
    ):
        movie = _spot_movie(
            spots=((8.3, 6.4), (20.2, 9.6), (8.3, 22.4), (20.2, 25.6))
        )
        _load(window, [_memmap(movie, tmp_path / "a.raw")], tmp_path)
        window.parameters_dialog.mng_slider.setValue(2000)
        window.parameters_dialog.split_fov_checkbox.setChecked(True)
        window.view.rois = [list(map(list, r)) for r in self.REGIONS]
        _identify_on(window, gui_localize.IDENTIFY_MODE_SUM)
        messages = _Messages(monkeypatch)
        window.identify()
        _wait(window)
        # two spots per region are too few to register them
        assert messages.kinds == ["warning"]
        assert "The regions could not be registered" in messages.text(
            "warning"
        )
        assert window.sum_identifications is None
        # the regions' own detections, made to register them, are kept
        assert len(window.identifications) == 4 * FRAMES

    def test_unregistered_regions_are_added_as_they_are(
        self, window, tmp_path
    ):
        movie = _spot_movie(spots=((8.3, 6.4), (8.3, 22.4)))
        _load(window, [_memmap(movie, tmp_path / "a.raw")], tmp_path)
        window.parameters_dialog.mng_slider.setValue(2000)
        window.parameters_dialog.split_fov_checkbox.setChecked(True)
        window.view.rois = [list(map(list, r)) for r in self.REGIONS]
        _identify_on(window, gui_localize.IDENTIFY_MODE_UNREGISTERED_SUM)
        window.identify()
        _wait(window)
        ids = window.sum_identifications
        # the two copies of the spot fall on each other in the sum
        assert sorted(set(ids["x"])) == [6]
        assert window.ready_for_fit


# ---------------------------------------------------------------------------
# Loading spline calibrations and channel registrations
# ---------------------------------------------------------------------------


def _calibration_file(tmp_path, name, monkeypatch, loader, calibration):
    """An existing file that ``loader`` (an io function) reads as
    ``calibration``."""
    path = tmp_path / name
    path.write_bytes(b"")
    monkeypatch.setattr(
        gui_localize.io, loader, lambda *args, **kwargs: calibration
    )
    return str(path)


class TestSplineCalibrationFile:
    def test_a_missing_file_is_shown(self, loaded, tmp_path):
        dialog = loaded.parameters_dialog
        dialog.update_spline_calib(str(tmp_path / "gone.hdf5"))
        assert dialog.spline_calibration == {}
        assert dialog.spline_calib_label.text() == (
            "-- calibration path not found --"
        )

    def test_an_invalid_file_is_shown_with_the_reason(self, loaded, tmp_path):
        broken = tmp_path / "broken.hdf5"
        broken.write_text("not hdf5")
        dialog = loaded.parameters_dialog
        dialog.update_spline_calib(str(broken))
        assert dialog.spline_calibration == {}
        assert dialog.spline_calib_label.text() == "-- invalid calibration --"
        assert dialog.spline_calib_label.toolTip()

    def test_a_calibration_is_loaded_and_cleared(
        self, loaded, tmp_path, monkeypatch
    ):
        calibration = _multichannel_calibration(link_photons=False)
        path = _calibration_file(
            tmp_path,
            "psf.hdf5",
            monkeypatch,
            "load_spline_calibration",
            calibration,
        )
        dialog = loaded.parameters_dialog
        dialog.update_spline_calib(path)
        assert dialog.spline_calibration is calibration
        assert dialog.spline_calib_label.text() == "psf.hdf5"
        dialog.update_spline_calib(None)
        assert dialog.spline_calibration == {}
        assert dialog.spline_calibration_path is None
        assert dialog.spline_calib_label.text() == (
            "-- no calibration loaded --"
        )

    def test_a_multichannel_calibration_offers_linked_photons(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        dialog = window.parameters_dialog
        dialog.fit_model.setCurrentText(SPLINE_MODEL)
        path = _calibration_file(
            tmp_path,
            "psf.hdf5",
            monkeypatch,
            "load_spline_calibration",
            _multichannel_calibration(link_photons=False),
        )
        dialog.update_spline_calib(path)
        # as the calibration was built
        assert dialog.link_photons_checkbox.isVisibleTo(dialog)
        assert not dialog._link_photons_enabled()

    def test_a_split_fov_calibration_draws_its_regions(
        self, loaded, tmp_path, monkeypatch
    ):
        regions = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]
        path = _calibration_file(
            tmp_path,
            "psf.hdf5",
            monkeypatch,
            "load_spline_calibration",
            _multichannel_calibration(split_fov=True, regions=regions),
        )
        loaded.parameters_dialog.update_spline_calib(path)
        assert loaded.view.split_fov_mode
        assert loaded.parameters_dialog.split_fov_checkbox.isChecked()
        assert loaded.view.rois == regions


class TestChannelRegistrationFile:
    def test_a_missing_file_is_shown(self, loaded, tmp_path):
        dialog = loaded.parameters_dialog
        dialog.update_channel_registration(str(tmp_path / "gone.yaml"))
        assert not dialog.channel_registration_calibration
        assert dialog.channel_registration_label.text() == (
            "-- registration path not found --"
        )

    def test_an_invalid_file_is_shown_with_the_reason(self, loaded, tmp_path):
        broken = tmp_path / "broken.yaml"
        broken.write_text("{ not: yaml")
        dialog = loaded.parameters_dialog
        dialog.update_channel_registration(str(broken))
        assert not dialog.channel_registration_calibration
        assert dialog.channel_registration_label.text() == (
            "-- invalid registration --"
        )

    def test_a_file_without_transforms_is_refused(
        self, loaded, tmp_path, monkeypatch
    ):
        path = _calibration_file(
            tmp_path,
            "z.yaml",
            monkeypatch,
            "load_any_calibration",
            {"X Coefficients": [1.0]},
        )
        dialog = loaded.parameters_dialog
        dialog.update_channel_registration(path)
        assert not dialog.channel_registration_calibration
        assert dialog.channel_registration_label.text() == (
            "-- not a channel registration --"
        )

    def test_a_registration_is_chosen_and_loaded(
        self, loaded, tmp_path, monkeypatch
    ):
        registration = _registration()
        path = _calibration_file(
            tmp_path,
            "reg.yaml",
            monkeypatch,
            "load_any_calibration",
            registration,
        )
        asked = _open_file(monkeypatch, path)
        dialog = loaded.parameters_dialog
        dialog.load_channel_registration()
        assert asked == ["Load channel registration"]
        assert dialog.channel_registration_calibration is registration
        assert dialog.channel_registration_label.text() == "reg.yaml"

    def test_a_split_fov_registration_draws_its_regions(
        self, loaded, tmp_path, monkeypatch
    ):
        regions = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]
        path = _calibration_file(
            tmp_path,
            "reg.yaml",
            monkeypatch,
            "load_any_calibration",
            _registration(split_fov=True, regions=regions),
        )
        loaded.parameters_dialog.update_channel_registration(path)
        assert loaded.view.split_fov_mode
        assert loaded.view.rois == regions


# ---------------------------------------------------------------------------
# How many identified spots the joint fit links
# ---------------------------------------------------------------------------


class TestLinkedCount:
    def test_the_spots_found_in_every_channel_are_counted(
        self, window, tmp_path
    ):
        # the second spot of the reference channel is missing in the other
        _two_channels(window, tmp_path, second=_spot_movie(spots=SPOTS[:1]))
        window.parameters_dialog.channel_registration_calibration = (
            _registration()
        )
        window.identify()
        _wait(window)
        assert _status(window) == (
            f"Identified {FRAMES} spots linked across 2 channels "
            f"({3 * FRAMES} in total). Ready for fit."
        )

    def test_channels_fitted_separately_are_not_linked(self, window, tmp_path):
        _two_channels(window, tmp_path)
        window.parameters_dialog.channel_registration_calibration = (
            _registration()
        )
        window.parameters_dialog.fit_mode_combo.setCurrentText(
            gui_localize.FIT_MODE_SEPARATE
        )
        window.identify()
        _wait(window)
        assert _status(window) == (
            f"Identified {4 * FRAMES} spots across 2 channels. Ready for fit."
        )

    def test_the_regions_of_a_split_fov_movie_are_linked(
        self, window, tmp_path
    ):
        # a molecule seen in both halves, and one in the left half only
        movie = _spot_movie(spots=((8.3, 6.4), (8.3, 22.4), (22.2, 8.6)))
        _load(window, [_memmap(movie, tmp_path / "a.raw")], tmp_path)
        window.parameters_dialog.mng_slider.setValue(2000)
        regions = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]
        registration = _registration(
            split_fov=True, regions=regions, reference=0
        )
        # region-local: each half maps onto the other unchanged
        registration["channel_registration"] = registration[
            "channel_transforms"
        ]
        window.parameters_dialog.channel_registration_calibration = (
            registration
        )
        window.parameters_dialog.split_fov_checkbox.setChecked(True)
        window.view.rois = [list(map(list, r)) for r in regions]
        window.identify()
        _wait(window)
        assert _status(window).startswith(
            f"Identified {FRAMES} spots linked across 2 regions "
            f"({3 * FRAMES} in total)"
        )


class TestSplitFovMode:
    def test_refused_for_separate_channel_movies(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        messages = _Messages(monkeypatch)
        checkbox = window.parameters_dialog.split_fov_checkbox
        checkbox.setChecked(True)
        assert messages.kinds == ["information"]
        assert not checkbox.isChecked()
        assert not window.view.split_fov_mode


# ---------------------------------------------------------------------------
# Sample quality
# ---------------------------------------------------------------------------


class TestQualityCheck:
    def test_offered_after_the_fit_and_filled_in(
        self, loaded, tmp_path, monkeypatch
    ):
        from picasso import localize

        stored = []
        monkeypatch.setattr(localize, "check_nena", lambda *a, **k: 4.25)
        monkeypatch.setattr(
            localize, "check_drift", lambda *a, **k: (1.5, 2.25)
        )
        monkeypatch.setattr(localize, "check_kinetics", lambda *a: 3.5)
        monkeypatch.setattr(
            localize,
            "add_file_to_db",
            lambda path, *args, **kwargs: stored.append((path, kwargs)),
        )
        dialog = loaded.parameters_dialog
        assert not dialog.quality_check.isEnabled()
        loaded.localize()
        _wait(loaded)
        assert dialog.quality_check.isEnabled()

        dialog.quality_check.click()
        dialog.q_worker.wait()
        QtWidgets.QApplication.processEvents()
        assert not dialog.quality_check.isVisibleTo(dialog)
        assert [label.text() for label in dialog.quality_grid_values] == [
            f"{2 * FRAMES / FRAMES:.1f}",
            "4.25 nm",
            "X: 1.50 nm / Y: 2.25 nm",
            "3.500",
        ]
        assert stored == [
            (
                loaded.movie_path,
                {"drift": (1.5, 2.25), "len_mean": 3.5, "nena": 4.25},
            )
        ]
        assert _status(loaded) == "Quality parameters complete."

    def test_a_new_movie_hides_the_results(self, loaded, tmp_path):
        dialog = loaded.parameters_dialog
        dialog.quality_check.setEnabled(True)
        dialog.quality_grid_values[0].setText("1.0")
        _load(loaded, [_spot_movie()], tmp_path)
        assert not dialog.quality_check.isEnabled()
        assert dialog.quality_grid_values[0].text() == ""


# ---------------------------------------------------------------------------
# Fitting every channel / region on its own
# ---------------------------------------------------------------------------


def _fit_separately(window):
    window.parameters_dialog.fit_mode_combo.setCurrentText(
        gui_localize.FIT_MODE_SEPARATE
    )


class TestSeparateFit:
    def test_every_channel_is_saved_to_its_own_file(self, window, tmp_path):
        _two_channels(window, tmp_path)
        _fit_separately(window)
        window.set_current_channel(1)
        window.localize()
        _wait(window)
        saved = sorted(p.name for p in tmp_path.glob("*_locs.hdf5"))
        assert saved == ["Channel 0_locs.hdf5", "Channel 1_locs.hdf5"]
        locs, info = io.load_locs(str(tmp_path / "Channel 1_locs.hdf5"))
        assert len(locs) == 2 * FRAMES
        # the metadata of the channel's own identifications and fit
        assert info[:-1] == window.channels[1].identifications_info
        assert info[-1]["Step"] == "Fit 2D"
        assert _status(window) == (
            f"Fitted {4 * FRAMES} spots in 2 channels, saved to 2 files."
        )
        # back where the user was
        assert window.current_channel == 1

    def test_a_channel_without_spots_is_skipped(self, window, tmp_path):
        _two_channels(window, tmp_path, second=_spot_movie(spots=()))
        _fit_separately(window)
        window.localize()
        _wait(window)
        assert _status(window) == (
            f"Fitted {2 * FRAMES} spots in 1 channels, saved to 1 files. "
            "Skipped Channel 1."
        )

    def test_a_channel_that_cannot_be_fitted_ends_the_batch(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        _fit_separately(window)
        window.identify()
        _wait(window)
        # the second channel is set up for a spline fit it has no
        # calibration for
        window.set_current_channel(1)
        window.parameters_dialog.fit_model.setCurrentText(SPLINE_MODEL)
        window.set_current_channel(0)
        messages = _Messages(monkeypatch)
        window.fit()
        _wait(window)
        assert messages.kinds == ["warning"]
        assert _status(window) == "Aborted."
        assert window._multi_fit is None
        assert window.current_channel == 0

    def test_spots_of_the_registered_sum_are_not_fitted_separately(
        self, window, tmp_path, monkeypatch
    ):
        _two_channels(window, tmp_path)
        window.sum_identifications = _ids([(10, 8)])
        window.identifications = window.sum_identifications
        window.ready_for_fit = True
        _fit_separately(window)
        messages = _Messages(monkeypatch)
        window.fit()
        assert messages.kinds == ["information"]
        assert "sum of the registered channels" in messages.text("information")


class TestSeparateRegionFit:
    REGIONS = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]

    @pytest.fixture
    def regions(self, window, tmp_path):
        # one spot per half
        movie = _spot_movie(spots=((8.3, 6.4), (20.2, 25.6)))
        _load(window, [_memmap(movie, tmp_path / "a.raw")], tmp_path)
        window.parameters_dialog.mng_slider.setValue(2000)
        window.parameters_dialog.split_fov_checkbox.setChecked(True)
        window.view.rois = [list(map(list, r)) for r in self.REGIONS]
        _fit_separately(window)
        return window

    def test_every_region_is_saved_in_its_own_coordinates(
        self, regions, tmp_path
    ):
        regions.localize()
        _wait(regions)
        base = str(tmp_path / "Channel 0_")
        label = gui_localize.localize.region_label
        first, _ = io.load_locs(base + label(0) + "_locs.hdf5")
        second, info = io.load_locs(base + label(1) + "_locs.hdf5")
        assert len(first) == len(second) == FRAMES
        # from the region's origin: 6.4 px in the left half, 25.6 - 16 px
        # in the right one
        offset = np.mean(second["x"]) - np.mean(first["x"])
        assert offset / 130.0 == pytest.approx(9.6 - 6.4, abs=0.1)
        # the region's size and place on the sensor, recorded last
        assert lib.get_from_metadata(info, "Width") == 16
        assert info[-1]["Region"] == [[0, 16], [32, 32]]
        assert _status(regions) == (
            f"Fitted {2 * FRAMES} spots in 2 regions, saved to 2 files."
        )
        # the movie's own metadata is back
        assert regions.info[0]["Width"] == SIZE

    def test_a_region_without_spots_is_skipped(self, regions, tmp_path):
        regions.view.rois.append([[0, 0], [6, 6]])
        regions.localize()
        _wait(regions)
        assert _status(regions).endswith(
            "Skipped " + gui_localize.localize.region_label(2) + "."
        )

    def test_needs_the_regions_drawn(self, regions, monkeypatch):
        regions.identify()
        _wait(regions)
        regions.view.rois = []
        messages = _Messages(monkeypatch)
        regions.fit()
        assert messages.kinds == ["information"]
        assert "no region is drawn" in messages.text("information")


# ---------------------------------------------------------------------------
# Joint Gaussian fit across channels / regions
# ---------------------------------------------------------------------------

SPHERICAL = "2D spherical Gaussian"


class TestJointGaussFit:
    def test_two_channels_are_fitted_as_one(self, window, tmp_path):
        _two_channels(window, tmp_path)
        dialog = window.parameters_dialog
        dialog.fit_model.setCurrentText(SPHERICAL)
        dialog.channel_registration_calibration = _registration()
        window.set_current_channel(1)
        window.localize()
        _wait(window)
        locs, info = io.load_locs(str(tmp_path / "Channel 0_locs.hdf5"))
        # one localization per molecule and frame, linked across both
        assert len(locs) == 2 * FRAMES
        assert not (tmp_path / "Channel 1_locs.hdf5").exists()
        # shown on both channels, saved with the reference
        assert window.current_channel == 0
        assert len(window.channels[1].locs_display) == 2 * FRAMES

    def test_the_regions_of_one_movie_are_fitted_as_one(
        self, window, tmp_path
    ):
        movie = _spot_movie(spots=((8.3, 6.4), (8.3, 22.4)))
        _load(window, [_memmap(movie, tmp_path / "a.raw")], tmp_path)
        dialog = window.parameters_dialog
        dialog.mng_slider.setValue(2000)
        dialog.fit_model.setCurrentText(SPHERICAL)
        regions = [[[0, 0], [32, 16]], [[0, 16], [32, 32]]]
        registration = _registration(
            split_fov=True, regions=regions, reference=0
        )
        registration["channel_registration"] = registration[
            "channel_transforms"
        ]
        dialog.channel_registration_calibration = registration
        dialog.split_fov_checkbox.setChecked(True)
        window.view.rois = [list(map(list, r)) for r in regions]
        window.localize()
        _wait(window)
        saved = list(tmp_path.glob("*_locs.hdf5"))
        assert len(saved) == 1
        locs, _ = io.load_locs(str(saved[0]))
        assert len(locs) == FRAMES


# ---------------------------------------------------------------------------
# Remembered settings, drag and drop, contrast
# ---------------------------------------------------------------------------


class TestRememberedSettings:
    def test_the_last_settings_come_back(self, qt_offscreen, tmp_path):
        first = gui_localize.Window()
        _load(first, [_spot_movie()], tmp_path)
        dialog = first.parameters_dialog
        dialog.box_spinbox.setValue(9)
        dialog.mng_slider.setValue(1234)
        dialog.fit_model.setCurrentText("2D rotated elliptical Gaussian")
        dialog.fit_optimizer.setCurrentText("MLE")
        dialog.gaussian_filter_spinbox.setValue(1.5)
        first.close()

        second = gui_localize.Window()
        dialog = second.parameters_dialog
        assert dialog.box_spinbox.value() == 9
        assert dialog.mng_slider.value() == 1234
        assert dialog.fit_model.currentText() == (
            "2D rotated elliptical Gaussian"
        )
        assert dialog.fit_optimizer.currentText() == "MLE"
        assert dialog.gaussian_filter_spinbox.value() == 1.5
        assert second.pwd == str(tmp_path)
        second.close()

    def test_unknown_stored_choices_keep_the_defaults(
        self, qt_offscreen, monkeypatch
    ):
        settings = io.load_user_settings()
        settings["Localize"]["fit_model"] = "Removed model"
        settings["Localize"]["fit_optimizer"] = "Removed optimizer"
        settings["Localize"]["fit_mode"] = "Removed mode"
        io.save_user_settings(settings)
        window = gui_localize.Window()
        assert window.parameters_dialog.fit_model.currentText() == (
            gui_localize.ASTIGMATISM_FIT_MODEL
        )
        window.close()


def _drag_event(kind, path):
    """A drag-enter or drop event carrying ``path`` (or text)."""
    mime = QtCore.QMimeData()
    if path is None:
        mime.setText("text")
    else:
        mime.setUrls([QtCore.QUrl.fromLocalFile(path)])
    args = (
        QtCore.Qt.DropAction.CopyAction,
        mime,
        LEFT,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    if kind == "enter":
        event = QtGui.QDragEnterEvent(QtCore.QPoint(5, 5), *args)
    else:
        event = QtGui.QDropEvent(QtCore.QPointF(5, 5), *args)
    # the event does not own its mime data, which Python would otherwise
    # free under it
    event.mime = mime
    return event


class TestDragAndDrop:
    @pytest.mark.parametrize(
        "name, accepted",
        [("movie.tif", True), ("movie.ND2", True), ("locs.hdf5", False)],
    )
    def test_only_movies_are_accepted(self, loaded, name, accepted):
        event = _drag_event("enter", f"/data/{name}")
        loaded.scene.dragEnterEvent(event)
        assert event.isAccepted() == accepted

    def test_text_is_not_accepted(self, loaded):
        event = _drag_event("enter", None)
        loaded.scene.dragEnterEvent(event)
        assert not event.isAccepted()

    def test_a_dropped_movie_is_opened(self, loaded, monkeypatch):
        opened = _Calls(loaded, monkeypatch, "open")
        loaded.scene.dropEvent(_drag_event("drop", "/data/movie.tif"))
        assert opened.calls == [(("/data/movie.tif",), {})]


class TestContrast:
    def test_typing_a_value_switches_auto_off(self, loaded):
        contrast = loaded.contrast_dialog
        contrast.auto_checkbox.setChecked(True)
        contrast.white_spinbox.setValue(900)
        assert not contrast.auto_checkbox.isChecked()
        assert contrast.white_spinbox.value() == 900

    def test_auto_takes_the_range_of_the_frame(self, loaded):
        contrast = loaded.contrast_dialog
        contrast.auto_checkbox.setChecked(False)
        contrast.change_contrast_silently(0, 5)
        contrast.auto_checkbox.setChecked(True)
        frame = loaded.movie[loaded.curr_frame_number]
        assert contrast.black_spinbox.value() == frame.min()
        assert contrast.white_spinbox.value() == frame.max()

    def test_the_slider_sets_both_values(self, loaded):
        contrast = loaded.contrast_dialog
        contrast.auto_checkbox.setChecked(True)
        contrast.set_contrast_from_slider(120.0, 800.0)
        assert not contrast.auto_checkbox.isChecked()
        assert (
            contrast.black_spinbox.value(),
            contrast.white_spinbox.value(),
        ) == (120.0, 800.0)

    def test_the_displayed_frame_follows_the_range(self, loaded):
        contrast = loaded.contrast_dialog
        contrast.auto_checkbox.setChecked(False)
        contrast.change_contrast_silently(100, 200)
        frame = np.array([[50, 100, 150, 300]], dtype=np.uint16)
        assert contrast.to_uint8(frame).tolist() == [[0, 0, 127, 255]]


# ---------------------------------------------------------------------------
# The file pickers of the calibration dialogs
# ---------------------------------------------------------------------------


def _answer_files(monkeypatch, getter, answer):
    asked = []

    def get(parent, caption, *args, **kwargs):
        asked.append(caption)
        return answer, ""

    monkeypatch.setattr(QtWidgets.QFileDialog, getter, get)
    return asked


class TestCameraCalibrationPickers:
    @pytest.fixture
    def dialog(self, window):
        return gui_localize.CameraCalibrationDialog(window)

    def test_the_dark_movie_suggests_the_output(self, dialog, monkeypatch):
        ok = dialog.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        assert not ok.isEnabled()
        _answer_files(monkeypatch, "getOpenFileName", "/data/dark.tif")
        dialog.browse_dark()
        assert dialog.dark_edit.text() == "/data/dark.tif"
        assert dialog.out_edit.text() == "/data/dark_scmos_calib.hdf5"
        assert ok.isEnabled()

    def test_a_chosen_output_is_kept(self, dialog, monkeypatch):
        dialog.out_edit.setText("/out/mine.hdf5")
        _answer_files(monkeypatch, "getOpenFileName", "/data/dark.tif")
        dialog.browse_dark()
        assert dialog.out_edit.text() == "/out/mine.hdf5"
        _answer_files(monkeypatch, "getSaveFileName", "/out/other.hdf5")
        dialog.browse_out()
        assert dialog.out_edit.text() == "/out/other.hdf5"

    def test_canceled_pickers_change_nothing(self, dialog, monkeypatch):
        _answer_files(monkeypatch, "getOpenFileName", "")
        _answer_files(monkeypatch, "getSaveFileName", "")
        dialog.browse_dark()
        dialog.browse_out()
        assert dialog.dark_edit.text() == dialog.out_edit.text() == ""

    def test_bright_movies_are_added_once_and_removed(
        self, dialog, monkeypatch
    ):
        paths = ["/data/b1.tif", "/data/b2.tif"]
        _answer_files(monkeypatch, "getOpenFileNames", paths)
        dialog.browse_bright()
        dialog.browse_bright()
        assert dialog.bright_paths() == paths
        dialog.bright_list.selectRow(0)
        dialog.remove_bright()
        assert dialog.bright_paths() == paths[1:]


class TestCalibrateAffinePickers:
    @pytest.fixture
    def dialog(self, loaded):
        return gui_localize.CalibrateAffineDialog(loaded)

    def test_the_loaded_movie_is_the_reference(self, dialog, loaded):
        assert dialog.reference_path == loaded.movie_path

    def test_the_images_and_the_calibration_are_picked(
        self, dialog, monkeypatch
    ):
        asked = _answer_files(monkeypatch, "getOpenFileName", "/data/a.tif")
        dialog._browse_reference()
        dialog._browse_target()
        dialog._browse_calibration()
        assert dialog.reference_path == dialog.target_path == "/data/a.tif"
        assert dialog.calibration_path == "/data/a.tif"
        assert asked[1] == "Select cylindrical lens image"

    def test_a_new_calibration_gets_the_yaml_extension(
        self, dialog, monkeypatch
    ):
        _answer_files(monkeypatch, "getSaveFileName", "/data/lateral")
        dialog._new_calibration()
        assert dialog.calibration_path == "/data/lateral.yaml"

    def test_show_needs_an_image(self, dialog, monkeypatch):
        dialog.target_edit.setText("")
        messages = _Messages(monkeypatch)
        dialog._show_target()
        assert messages.kinds == ["warning"]

    def test_show_opens_the_image(self, dialog, loaded, monkeypatch):
        opened = _Calls(loaded, monkeypatch, "open")
        dialog._show_reference()
        assert opened.calls == [((loaded.movie_path,), {})]
        assert loaded.parameters_dialog.isVisible()


# ---------------------------------------------------------------------------
# Switching channels
# ---------------------------------------------------------------------------


class TestChannels:
    def test_each_channel_keeps_its_own_settings(self, window, tmp_path):
        _two_channels(window, tmp_path)
        dialog = window.parameters_dialog
        dialog.box_spinbox.setValue(9)
        window.next_channel()
        assert window.current_channel == 1
        dialog.box_spinbox.setValue(5)
        window.channel_combo.setCurrentIndex(0)
        assert window.current_channel == 0
        assert dialog.box_spinbox.value() == 9
        window.previous_channel()  # already the first
        assert window.current_channel == 0
        window.next_channel()
        assert dialog.box_spinbox.value() == 5
        window.next_channel()  # already the last
        assert window.current_channel == 1

    def test_separate_movies_turn_split_fov_off(self, loaded, tmp_path):
        checkbox = loaded.parameters_dialog.split_fov_checkbox
        checkbox.setChecked(True)
        assert loaded.view.split_fov_mode
        _two_channels(loaded, tmp_path)
        assert not checkbox.isChecked()
        assert not checkbox.isEnabled()
        assert not loaded.view.split_fov_mode


class TestCalibrationFilePickers:
    @pytest.mark.parametrize(
        "load, update, caption",
        [
            ("load_z_calib", "update_z_calib", "Load 3d calibration"),
            (
                "load_spline_calib",
                "update_spline_calib",
                "Load spline PSF calibration",
            ),
            (
                "load_camera_calib",
                "update_camera_calib",
                "Load sCMOS camera calibration",
            ),
        ],
    )
    def test_the_chosen_file_is_loaded_and_cancel_loads_nothing(
        self, loaded, monkeypatch, load, update, caption
    ):
        dialog = loaded.parameters_dialog
        updates = _Calls(dialog, monkeypatch, update)
        asked = _answer_files(monkeypatch, "getOpenFileName", "/c/a.file")
        getattr(dialog, load)()
        assert asked == [caption]
        assert updates.calls == [(("/c/a.file",), {})]
        _answer_files(monkeypatch, "getOpenFileName", "")
        getattr(dialog, load)()
        assert len(updates.calls) == 1
