"""Releasing memory while a Picasso GUI is open but not used
(``picasso.gui.idle``).

After the set time without input the main windows drop what they
rebuild on demand and the freed memory goes back to the operating
system; nothing closes and nothing on screen changes.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from picasso import io, lib, lib_qt
from picasso.gui import idle

# imported here, not in a fixture: conftest makes the GUI render
# synchronously only once the module is loaded
from picasso.gui import render as gui_render

WIDTH = HEIGHT = 64.0
PIXELSIZE = 130.0


def _set_minutes(value) -> None:
    settings = io.load_user_settings()
    settings[idle.SETTING] = value
    io.save_user_settings(settings)


def _key_press() -> QtGui.QKeyEvent:
    return QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_A,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


# --- the setting --- #


def test_a_missing_setting_is_written_with_the_default():
    assert idle.idle_minutes() == idle.DEFAULT_MINUTES
    assert io.load_user_settings()[idle.SETTING] == idle.DEFAULT_MINUTES


@pytest.mark.parametrize(
    "value, expected",
    [
        (5, 5.0),
        (0.5, 0.5),
        (0, 0.0),
        (-1, idle.DEFAULT_MINUTES),
        ("ten", idle.DEFAULT_MINUTES),
        (True, idle.DEFAULT_MINUTES),
        (None, idle.DEFAULT_MINUTES),
    ],
)
def test_the_setting_is_validated(value, expected):
    _set_minutes(value)
    assert idle.idle_minutes() == expected


# --- the watcher --- #


@pytest.fixture
def releases(qapp, monkeypatch):
    """Count the releases instead of releasing; ``released`` is what the
    windows answer (False: busy)."""

    class _Releases(list):
        released = True  # what the windows answer

    calls = _Releases()

    def release(app):
        calls.append(app)
        return calls.released

    monkeypatch.setattr(idle, "release_idle_memory", release)
    return calls


@pytest.fixture
def watcher(qapp):
    watcher = idle.IdleWatcher(qapp, tick_ms=60_000)
    yield watcher
    watcher.stop()


def test_memory_is_released_once_after_the_idle_time(releases, watcher):
    _set_minutes(3)
    watcher._tick()
    watcher._tick()
    assert releases == []
    watcher._tick()
    assert len(releases) == 1
    for _ in range(5):  # released until the next input
        watcher._tick()
    assert len(releases) == 1


def test_input_restarts_the_count_and_rearms_the_release(
    qt_offscreen, releases, watcher
):
    _set_minutes(2)
    widget = QtWidgets.QLineEdit()
    for _ in range(2):
        watcher._tick()
    assert len(releases) == 1
    # a key press anywhere in the application, through its event filter
    QtWidgets.QApplication.sendEvent(widget, _key_press())
    assert not watcher._filtering  # no filter while the user works
    watcher._tick()  # the minute with input: the filter comes back
    assert watcher._filtering
    watcher._tick()
    assert len(releases) == 1  # one idle minute only
    watcher._tick()
    assert len(releases) == 2


def test_mouse_moves_are_not_activity(qt_offscreen, releases, watcher):
    _set_minutes(1)
    widget = QtWidgets.QLabel()
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(1, 1),
        QtCore.QPointF(1, 1),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, move)
    watcher._tick()
    assert len(releases) == 1


def test_a_running_task_counts_as_activity(releases, watcher, monkeypatch):
    _set_minutes(1)
    monkeypatch.setattr(lib_qt, "gui_busy", lambda: True)
    for _ in range(5):
        watcher._tick()
    assert releases == []
    monkeypatch.setattr(lib_qt, "gui_busy", lambda: False)
    watcher._tick()
    assert len(releases) == 1


def test_a_busy_window_is_asked_again(releases, watcher):
    _set_minutes(1)
    releases.released = False
    watcher._tick()
    watcher._tick()
    assert len(releases) == 2
    releases.released = True
    watcher._tick()
    watcher._tick()
    assert len(releases) == 3


def test_zero_never_releases(releases, watcher):
    _set_minutes(0)
    for _ in range(100):
        watcher._tick()
    assert releases == []


def test_tasks_and_status_steps_make_the_gui_busy(qt_offscreen):
    # both block the input for as long as they run (the blocker is
    # acquired directly: a nested event loop here would also run the
    # deferred deletions of the windows of earlier tests)
    assert not lib_qt.gui_busy()
    lib_qt._acquire_input_blocker()
    try:
        assert lib_qt.gui_busy()
    finally:
        lib_qt._input_blocker.release()
    assert not lib_qt.gui_busy()


def test_stop_removes_the_watcher(qapp, releases):
    watcher = idle.IdleWatcher(qapp)
    watcher.stop()
    assert not watcher._timer.isActive()
    assert not watcher._filtering


# --- the release --- #


class _Window(QtWidgets.QWidget):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer
        self.asked = 0

    def release_idle_memory(self):
        self.asked += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def trims(monkeypatch):
    calls = []
    monkeypatch.setattr(lib, "release_free_memory", lambda: calls.append(1))
    return calls


def test_every_main_window_is_asked_then_memory_returned(qt_offscreen, trims):
    windows = [_Window(True), _Window(None)]  # None: nothing to say
    assert idle.release_idle_memory(qt_offscreen)
    assert [w.asked for w in windows] == [1, 1]
    assert trims == [1]


def test_a_busy_window_defers_returning_the_memory(qt_offscreen, trims):
    windows = [_Window(True), _Window(False)]
    assert not idle.release_idle_memory(qt_offscreen)
    assert [w.asked for w in windows] == [1, 1]
    assert trims == []


def test_a_failing_window_does_not_stop_the_release(
    qt_offscreen, trims, caplog
):
    windows = [_Window(RuntimeError("boom")), _Window(True)]
    assert idle.release_idle_memory(qt_offscreen)
    assert windows[1].asked == 1
    assert trims == [1]
    assert "_Window" in caplog.text


def test_release_free_memory_runs():
    lib.release_free_memory()  # nothing to check but that it is safe


def test_the_contrast_slider_lets_go_of_the_image(qt_offscreen):
    image = {"array": np.arange(12.0).reshape(3, 4)}
    slider = lib_qt.DensityContrastSlider(
        QtWidgets.QDoubleSpinBox(),
        QtWidgets.QDoubleSpinBox(),
        image=lambda: image["array"],
    )
    assert slider._image_maximum() == 11.0
    slider.release_image()
    image["array"] = None  # dropped by its owner, too
    assert slider._last_image is None
    assert slider._image_maximum() == 11.0  # the track keeps its range
    image["array"] = np.full((2, 2), 3.0)  # the next render
    assert slider._image_maximum() == 3.0
    image["array"] = None
    assert slider._image_maximum() == 0.0  # no render and no release


# --- Render --- #


def _locs(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, WIDTH, size=n) * PIXELSIZE,
            "y": rng.uniform(0.0, HEIGHT, size=n) * PIXELSIZE,
            "lpx": rng.uniform(0.05, 0.3, size=n) * PIXELSIZE,
            "lpy": rng.uniform(0.05, 0.3, size=n) * PIXELSIZE,
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "z": rng.normal(0.0, 20.0, size=n),
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


@pytest.fixture
def render_window(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    for i in range(2):
        window.view.add(
            str(tmp_path / f"locs{i}.hdf5"),
            _locs(seed=i),
            _info(),
            render_=False,
        )
    window.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    window.view.resize(128, 128)
    window.view.update_scene()
    yield window
    window.window_rot.view_rot.stop_render_worker()
    window.view.stop_render_worker()


def test_render_drops_its_caches_and_redraws_the_same(render_window):
    view = render_window.view
    image = view.image.copy()
    view.x_locs = [view.locs[0]]  # as split for render by property
    for slider in (
        render_window.contrast_slider,
        render_window.display_settings_dlg.contrast_slider,
    ):
        slider._last_image = view.image
    render_window.slicer_dialog.slicer_cache = {0: QtGui.QPixmap(4, 4)}

    assert render_window.release_idle_memory()

    assert view.image is None and view._image_viewport is None
    assert view.x_locs is None
    assert view.index_blocks == [None, None]
    assert render_window.contrast_slider._last_image is None
    assert render_window.display_settings_dlg.contrast_slider._last_image is (
        None
    )
    assert render_window.slicer_dialog.slicer_cache == {}
    assert len(view.locs) == 2  # the data stays
    # a contrast change renders again, the same
    view.update_scene(use_cache=True)
    np.testing.assert_array_equal(view.image, image)


def test_render_waits_while_busy(render_window):
    view = render_window.view
    view._refine_timer.start()
    try:
        assert render_window.release_idle_memory() is False
        assert view.image is not None
    finally:
        view._refine_timer.stop()
    dialog = render_window.window_rot.animation_dialog
    dialog._build_thread = object()  # an animation is being built
    try:
        assert render_window.release_idle_memory() is False
    finally:
        dialog._build_thread = None
    assert render_window.release_idle_memory()


def test_the_closed_3d_view_drops_its_localizations(render_window):
    render_window.open_3d_view()
    window_rot = render_window.window_rot
    view_rot = window_rot.view_rot
    n_locs = [len(locs) for locs in view_rot.locs]
    assert n_locs and view_rot.image is not None
    rotation = view_rot._rotvec.copy()

    assert render_window.release_idle_memory()
    assert view_rot.image is None
    assert view_rot.locs  # open: still shown, so kept

    window_rot.close()
    assert render_window.release_idle_memory()
    assert view_rot.locs == [] and view_rot.infos == []
    np.testing.assert_array_equal(view_rot._rotvec, rotation)

    render_window.open_3d_view()  # loaded again
    assert [len(locs) for locs in view_rot.locs] == n_locs
    assert view_rot.image is not None
    window_rot.close()


@pytest.fixture
def spill_folder(tmp_path, monkeypatch):
    folder = tmp_path / "spill"
    folder.mkdir()
    monkeypatch.setattr(io, "spill_directory", lambda: str(folder))
    return folder


# --- Render: localizations moved to disk --- #


@pytest.fixture
def spilling(spill_folder, monkeypatch):
    """Move any localizations to disk when idle, into ``spill_folder``."""
    monkeypatch.setattr(idle, "SPILL_MIN_BYTES", 0)
    return spill_folder


def _weak(frames) -> list:
    import weakref

    return [weakref.ref(frame) for frame in frames]


def _alive(refs) -> list[bool]:
    import gc

    gc.collect()
    return [ref() is not None for ref in refs]


def test_render_moves_its_localizations_to_disk_and_back(
    render_window, spilling
):
    view = render_window.view
    expected = [locs.copy() for locs in view.locs]
    frames = _weak(view.locs)
    shown = view.qimage.copy()

    assert render_window.release_idle_memory()

    assert view.has_spilled_locs and view.n_channels == 2
    assert len(list(spilling.iterdir())) == 2
    assert _alive(frames) == [False, False]  # the memory is free
    assert view.qimage == shown  # nothing on screen changed
    # the next access reads them back
    for locs, original in zip(view.locs, expected):
        pd.testing.assert_frame_equal(locs, original)
    assert not view.has_spilled_locs
    assert list(spilling.iterdir()) == []
    view.update_scene()
    assert view.image is not None


def test_render_hovering_reads_nothing_back(render_window, spilling):
    view = render_window.view
    render_window.release_idle_memory()
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(10, 10),
        QtCore.QPointF(10, 10),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    view.mouseMoveEvent(move)
    view._mode = "Measure"  # the live measuring cross follows the cursor
    view._measure_following = True
    view.mouseMoveEvent(move)
    view.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert view.has_spilled_locs


def test_render_keeps_a_channel_held_elsewhere(render_window, spilling):
    view = render_window.view
    held = view.locs[0]  # e.g., by a plugin: moving it frees nothing
    frames = _weak(view.locs)
    assert render_window.release_idle_memory()
    assert view._locs[0] is held
    assert _alive(frames) == [True, False]
    assert len(list(spilling.iterdir())) == 1  # only the other one
    assert view.locs[0] is held


def test_render_keeps_small_data_in_memory(render_window, spill_folder):
    assert render_window.release_idle_memory()  # far below 200 MB
    assert not render_window.view.has_spilled_locs
    assert list(spill_folder.iterdir()) == []


def test_render_waits_while_loading(render_window, spilling):
    view = render_window.view
    view._load_thread = object()
    try:
        assert gui_render.spill_idle_channels(render_window) is False
    finally:
        view._load_thread = None
    assert not view.has_spilled_locs


def test_render_open_3d_view_moves_its_copies_too(render_window, spilling):
    render_window.open_3d_view()
    view_rot = render_window.window_rot.view_rot
    expected = [locs.copy() for locs in view_rot.locs]
    assert render_window.release_idle_memory()
    assert view_rot.has_spilled_locs
    for locs, original in zip(view_rot.locs, expected):
        pd.testing.assert_frame_equal(locs, original)
    render_window.window_rot.close()


def test_render_dialogs_follow_the_channels(render_window, spilling):
    view = render_window.view
    render_window.release_idle_memory()
    assert render_window.mask_settings_dialog.locs is view.locs
    assert not view.has_spilled_locs  # read back by the dialog
    render_window.release_idle_memory()
    zcoord = render_window.slicer_dialog.zcoord
    assert [len(z) for z in zcoord] == [len(locs) for locs in view.locs]


@pytest.fixture
def linked_windows(render_window, tmp_path):
    """A second, linked window sharing the first window's channel 1."""
    from picasso.gui import render_link

    group = render_link.LinkGroup(render_window)
    secondary = gui_render.Window(plugins_loaded=True, link_group=group)
    secondary.view.add_channels_from(render_window.view, [1], share=True)
    secondary.view.resize(100, 100)
    secondary.view.viewport = [
        (0.0, 0.0),
        (HEIGHT * PIXELSIZE, WIDTH * PIXELSIZE),
    ]
    secondary.view.update_scene()
    render_window.view.update_scene()
    yield group, render_window, secondary
    secondary.view.stop_render_worker()
    secondary.window_rot.view_rot.stop_render_worker()


@pytest.mark.parametrize("first", ["primary", "secondary"])
def test_render_linked_windows_are_released_together(
    linked_windows, spilling, monkeypatch, first
):
    """The idle release asks every window in turn, in any order: the
    first moves the channels of all linked windows to disk, and the
    others must neither read them back nor write them out again."""
    group, primary, secondary = linked_windows
    changed = []
    monkeypatch.setattr(
        group, "channels_changed", lambda *args: changed.append(args)
    )
    order = [primary, secondary]
    if first == "secondary":
        order.reverse()
    for window in order:
        assert window.release_idle_memory()

    assert primary.view.has_spilled_locs
    assert secondary.view.has_spilled_locs
    assert len(list(spilling.iterdir())) == 2  # each table once
    primary.view.update_scene()
    secondary.view.update_scene()
    assert secondary.view.locs[0] is primary.view.locs[1]  # still shared
    assert changed == []  # not mistaken for replaced channels
    assert list(spilling.iterdir()) == []


def test_render_linked_windows_share_after_reading_back(
    linked_windows, spilling, monkeypatch
):
    group, primary, secondary = linked_windows
    shared = primary.view.locs[1]
    assert secondary.view.locs[0] is shared
    frames = _weak(primary.view.locs)
    del shared

    assert primary.release_idle_memory()  # moves both windows' data

    assert primary.view.has_spilled_locs
    assert secondary.view.has_spilled_locs
    assert _alive(frames) == [False, False]
    assert len(list(spilling.iterdir())) == 2  # the shared one once
    changed = []
    monkeypatch.setattr(
        group, "channels_changed", lambda *args: changed.append(args)
    )
    # reads the channels back; the linked window follows the redraw
    primary.view.update_scene()
    assert not primary.view.has_spilled_locs
    assert secondary.view.locs[0] is primary.view.locs[1]  # shared again
    secondary.view.update_scene()
    assert changed == []  # not mistaken for replaced channels


# --- Filter --- #


@pytest.fixture
def filter_window(qt_offscreen, tmp_path, spill_folder, monkeypatch):
    """Filter with a file open, moving any table to disk when idle."""
    from picasso.gui import filter as gui_filter

    monkeypatch.setattr(idle, "SPILL_MIN_BYTES", 0)
    path = str(tmp_path / "locs.hdf5")
    io.save_locs(path, _locs(n=5000), _info())
    window = gui_filter.Window()
    window.resize(900, 600)
    window.show()
    window.open(path)
    window.path = path
    yield window
    window._discard_spill()


def _shown_rows(window) -> list[list[str]]:
    model = window.table_model
    # TableModel.index is the first row shown, not QModelIndex lookup
    cell = QtCore.QAbstractTableModel.index
    return [
        [model.data(cell(model, row, column)) for column in range(4)]
        for row in range(model.rowCount())
    ]


def test_filter_moves_its_localizations_to_disk_and_back(
    filter_window, spill_folder
):
    import gc
    import weakref

    window = filter_window
    window.apply_range("photons", 1000.0, 4000.0, inclusive=True)
    window.vertical_scrollbar.setValue(10)
    n_filtered = window.n_filtered
    shown = _shown_rows(window)
    assert shown
    column = window._locs_full["x"].to_numpy()
    storage = weakref.ref(column if column.base is None else column.base)
    del column

    assert window.release_idle_memory()

    assert window._locs_full is None and window.has_locs
    assert len(list(spill_folder.iterdir())) == 1
    gc.collect()
    assert storage() is None  # the memory is free
    # nothing on screen changed, and nothing was read back for it
    assert _shown_rows(window) == shown
    assert window.n_filtered == n_filtered
    window.resize(700, 500)
    window.display_locs(window.vertical_scrollbar.value())
    assert _shown_rows(window) == shown
    assert window._spilled is not None

    # the next filter reads the localizations back
    window.apply_range("x", 0.0, 32.0)
    assert window._spilled is None
    assert list(spill_folder.iterdir()) == []
    locs, _ = io.load_filter(window.path)
    expected = locs[
        (locs["photons"] >= 1000.0)
        & (locs["photons"] <= 4000.0)
        & (locs["x"] > 0.0)
        & (locs["x"] < 32.0)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(window.materialize_filtered(), expected)


def test_filter_scrolling_reads_the_localizations_back(filter_window):
    window = filter_window
    assert window.release_idle_memory()
    window.vertical_scrollbar.setValue(100)
    assert window._spilled is None
    assert _shown_rows(window)[0][0] == str(window.locs_full.iloc[100, 0])


def test_filter_keeps_small_tables_in_memory(filter_window, monkeypatch):
    monkeypatch.setattr(idle, "SPILL_MIN_BYTES", 2**40)
    assert filter_window.release_idle_memory()
    assert filter_window._spilled is None
    assert filter_window._locs_full is not None


def test_filter_keeps_what_cannot_be_moved(filter_window, monkeypatch):
    def refuse(frame):
        raise io.SpillError("no space left")

    monkeypatch.setattr(io, "SpilledFrame", refuse)
    assert filter_window.release_idle_memory()
    assert filter_window._locs_full is not None


def test_filter_drops_the_disk_copy_unread(
    filter_window, spill_folder, monkeypatch, tmp_path
):
    """Opening another file or closing Filter does not read the old
    localizations back."""
    window = filter_window
    window.release_idle_memory()

    def no_reading(self):
        raise AssertionError("read back")

    monkeypatch.setattr(io.SpilledFrame, "load", no_reading)
    other = str(tmp_path / "other.hdf5")
    io.save_locs(other, _locs(n=100, seed=1), _info())
    window.open(other)
    assert window.n_filtered == 100
    assert list(spill_folder.iterdir()) == []
    window.release_idle_memory()
    assert len(list(spill_folder.iterdir())) == 1
    window.close()
    assert list(spill_folder.iterdir()) == []


def test_filter_reports_a_lost_disk_copy(filter_window):
    import shutil

    window = filter_window
    window.release_idle_memory()
    shutil.rmtree(window._spilled.path)
    with pytest.raises(io.SpillError, match="open the file"):
        window.locs_full
    assert not window.has_locs  # as if no file was loaded


def test_filter_reuses_the_subclustering_dialog(qt_offscreen):
    from picasso.gui import filter as gui_filter

    window = gui_filter.Window()
    window.locs_full = _locs(n=10)
    window.plot_subclustering()
    dialog = window.subcluster_num
    dialog.close()
    window.plot_subclustering()
    assert window.subcluster_num is dialog
    dialog.close()


# --- Localize --- #


def test_localize_clears_the_temporal_median_frames():
    from picasso.gui import localize as gui_localize

    class _Temporal:
        cleared = 0

        def clear_cache(self):
            self.cleared += 1

    shown, other = _Temporal(), _Temporal()

    class _Window:
        _active_worker = None
        _temporal_movie = shown
        _channel_filter_cache = {1: [other, None], 2: [None, None]}

    release = gui_localize.Window.release_idle_memory
    assert release(_Window()) is True
    assert (shown.cleared, other.cleared) == (1, 1)
    _Window._active_worker = object()  # identifying or fitting
    assert release(_Window()) is False
    assert (shown.cleared, other.cleared) == (1, 1)
