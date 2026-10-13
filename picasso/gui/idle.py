"""
picasso.gui.idle
~~~~~~~~~~~~~~~~

Release memory while a Picasso GUI is open but not used.

A GUI left open after work - overnight, over a weekend - keeps the RAM
and GPU memory of its last session, although nothing uses it. After
``Release memory after idle (min)`` minutes without user input (a top
level key of ``~/.picasso/settings.yaml``, default 30; 0 never
releases), the main windows drop the memory that is rebuilt on demand
(e.g., Render's cached renders and its GPU device), and the freed memory
is handed back to the operating system. No window closes and nothing
on screen changes; the localizations, picks and settings stay as they
were. The next action that needs a dropped cache rebuilds it, so it may
take a little longer than usual - e.g., the first render after a
release starts the GPU again.

Data that cannot be rebuilt can be moved to disk instead, if it takes
at least ``SPILL_MIN_BYTES``: the localizations are written to
``~/.picasso/spill`` (``io.SpilledFrame``) and read back on the next
action that needs them. Filter keeps the rows shown in its table on
screen; Render's views (``SpillableChannels``) keep the rendered image.
Folders left behind by a crashed process are removed when a GUI starts.

A main window takes part by defining ``release_idle_memory()``, which
returns False when it cannot release now (e.g., it is loading a file),
so that the release is retried a minute later. Input to any window of
the process counts as activity, and so does a running task
(``lib_qt.run_task`` or ``run_with_status``): a long computation is
never interrupted by a release, which waits until it has finished and
the GUI has been idle for the set time again.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import gc
import logging
import sys
import weakref

import numpy as np
from PyQt6 import QtCore, QtWidgets

from .. import io, lib, lib_qt

#: Top-level key of the user settings holding the idle time (minutes)
#: after which memory is released; 0 never releases it.
SETTING = "Release memory after idle (min)"
#: Default of ``SETTING``.
DEFAULT_MINUTES = 30

#: Tables smaller than this (bytes) stay in memory when idle: moving
#: them to disk would free little.
SPILL_MIN_BYTES = 200 * 2**20

#: Interval (ms) in which the idle time is counted. A release happens at
#: most one interval later than the setting asks for.
TICK_MS = 60_000

#: Events that mean the user is working with the GUI. Moving the mouse
#: is not among them, since a mouse resting on a window can still send
#: moves (jitter, remote desktop sessions).
_ACTIVITY = frozenset(
    {
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QEvent.Type.MouseButtonDblClick,
        QtCore.QEvent.Type.Wheel,
        QtCore.QEvent.Type.KeyPress,
        QtCore.QEvent.Type.DragEnter,
        QtCore.QEvent.Type.Drop,
        QtCore.QEvent.Type.TouchBegin,
        QtCore.QEvent.Type.TabletPress,
        QtCore.QEvent.Type.NativeGesture,
    }
)

_log = logging.getLogger(__name__)


def idle_minutes() -> float:
    """Minutes without user input after which a GUI releases memory.

    Read from ``SETTING`` at the top level of the user settings; a
    missing key is written with ``DEFAULT_MINUTES`` (so that it is
    visible in the file), unless the file on disk could not be read.

    Returns
    -------
    minutes : float
        The idle time; 0 if memory is never released. Invalid values
        (not a number, negative) give ``DEFAULT_MINUTES``.
    """
    settings = io.load_user_settings()
    # AutoDict creates an empty dict for a missing key: check membership
    if SETTING not in settings:
        if not io.settings_file_is_broken():
            settings[SETTING] = DEFAULT_MINUTES
            io.save_user_settings(settings)
        return float(DEFAULT_MINUTES)
    value = settings[SETTING]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(DEFAULT_MINUTES)
    if not value >= 0:  # also rejects NaN
        return float(DEFAULT_MINUTES)
    return float(value)


def release_idle_memory(app: QtWidgets.QApplication) -> bool:
    """Ask every main window to drop the memory it can rebuild, then
    return the freed memory to the operating system.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application whose top-level windows are asked, i.e., those
        that define ``release_idle_memory()``.

    Returns
    -------
    released : bool
        False if a window was busy and is to be asked again later; the
        memory is then not yet returned to the operating system.
    """
    import psutil

    process = psutil.Process()
    before = process.memory_info().rss
    released = True
    for widget in app.topLevelWidgets():
        release = getattr(widget, "release_idle_memory", None)
        if not callable(release):
            continue
        try:
            if release() is False:
                released = False
        except Exception:  # noqa: BLE001 - never disturb an idle GUI
            _log.warning(
                "Releasing the idle memory of %s failed",
                type(widget).__name__,
                exc_info=True,
            )
    if not released:
        return False
    lib.release_free_memory()
    _log.info(
        "Released idle memory: resident memory %.0f MB -> %.0f MB",
        before / 1e6,
        process.memory_info().rss / 1e6,
    )
    return True


class IdleWatcher(QtCore.QObject):
    """Release memory after the set idle time, see the module docstring.

    The input of a whole minute is checked by its first event only: the
    application-wide event filter that sees it removes itself until the
    next tick, so the watcher costs nothing while the user works.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application to watch.
    tick_ms : int, optional
        Interval in which the idle time is counted. Default is
        ``TICK_MS``.
    """

    def __init__(
        self, app: QtWidgets.QApplication, tick_ms: int = TICK_MS
    ) -> None:
        super().__init__()
        self._app = app
        self._tick_ms = tick_ms
        self._idle_ms = 0  # time since the last input
        self._released = False  # since the last input
        self._filtering = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(tick_ms)
        self._timer.timeout.connect(self._tick)
        self._watch_input()
        self._timer.start()

    def stop(self) -> None:
        """Stop watching (when the event loop has ended)."""
        self._timer.stop()
        if self._filtering:
            self._app.removeEventFilter(self)
            self._filtering = False

    def _watch_input(self) -> None:
        self._app.installEventFilter(self)
        self._filtering = True

    def eventFilter(self, obj, event):
        if event.type() in _ACTIVITY:
            self._idle_ms = 0
            self._released = False
            # removing the filter while it runs is safe in Qt
            self._app.removeEventFilter(self)
            self._filtering = False
        return False

    def _tick(self) -> None:
        if not self._filtering:  # there was input since the last tick
            self._watch_input()
            return
        if lib_qt.gui_busy():  # a task runs, which counts as activity
            self._idle_ms = 0
            return
        self._idle_ms += self._tick_ms
        if self._released:
            return
        minutes = idle_minutes()
        if minutes > 0 and self._idle_ms >= minutes * 60_000:
            self._released = release_idle_memory(self._app)


def install(app: QtWidgets.QApplication) -> IdleWatcher | None:
    """Start releasing memory when ``app`` is idle (see the module
    docstring).

    A failure here is reported and swallowed: the GUI works the same
    without it.

    Parameters
    ----------
    app : QtWidgets.QApplication
        The application to watch.

    Returns
    -------
    watcher : IdleWatcher or None
        The watcher, to be stopped when the event loop has ended, or
        None if it could not be started.
    """
    try:
        idle_minutes()  # write the default, so that it shows in the file
        io.remove_stale_spills()
        return IdleWatcher(app)
    except Exception:  # noqa: BLE001 - the GUI works without it
        sys.excepthook(*sys.exc_info())
        return None


class SpillableChannels:
    """Mixin of a view holding the localizations of its channels in
    ``locs``, a list of ``pd.DataFrame``, which ``spill_channels``
    moves to disk while the GUI is idle.

    The next access to ``locs`` reads them back (``restore_locs``);
    ``n_channels`` and ``has_spilled_locs`` do not. Assigning ``locs``
    drops what is on disk. ``locs`` must be assigned once (e.g., in
    ``__init__``) before it is read.
    """

    #: per channel, the ``io.SpilledFrame`` it was moved to (None for a
    #: channel kept in memory), or None if nothing is on disk
    _spilled_locs = None

    @property
    def locs(self) -> list:
        """The localizations of the channels, read back from the disk
        first if they were moved there while the GUI was idle."""
        if self._spilled_locs is not None:
            self.restore_locs()
        return self._locs

    @locs.setter
    def locs(self, locs: list) -> None:
        self._spilled_locs = None
        self._locs = locs

    @property
    def n_channels(self) -> int:
        """Number of channels; reads nothing back from the disk."""
        return len(self._locs)

    @property
    def has_spilled_locs(self) -> bool:
        """Whether channels are moved to disk."""
        return self._spilled_locs is not None

    def restore_locs(self) -> None:
        """Read back the channels moved to disk. A channel shared with
        another view is read once, and both share it again
        (``io.SpilledFrame.load``).

        Raises
        ------
        io.SpillError
            If a channel cannot be read back; the channels stay on disk
            then.
        """
        spilled = self._spilled_locs
        if spilled is None:
            return
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            restored = [
                frame if moved is None else moved.load()
                for frame, moved in zip(self._locs, spilled)
            ]
        except io.SpillError as error:
            raise io.SpillError(
                "The localizations moved to disk while Picasso was idle"
                f" could not be read back ({error}). Please open the"
                " files again."
            ) from error
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._locs[:] = restored
        self._spilled_locs = None


def spill_channels(holders: list, references: list = ()) -> int:
    """Move the channels of views to disk, while the GUI is idle.

    Each table is written once (``io.SpilledFrame``), also if several
    views share it, and the views and ``references`` then hold its
    placeholder instead. A table that is still in memory afterwards,
    because something else holds it, is put back at once and its
    copy on the disk is removed: moving it would free nothing. Nothing
    is moved if all tables together take less than ``SPILL_MIN_BYTES``.

    Parameters
    ----------
    holders : list of SpillableChannels
        The views. Views with channels on disk already are skipped.
    references : list of lists
        Further lists that hold tables of the views, e.g., those a
        linked window compares the channels with. Their tables are
        replaced by the placeholders (``io.SpilledFrame``) in place.

    Returns
    -------
    nbytes : int
        Size of the tables moved to disk.
    """
    holders = [h for h in holders if not h.has_spilled_locs and h.n_channels]
    tables = {}
    for holder in holders:
        tables.update((id(frame), frame) for frame in holder._locs)
    if not tables:
        return 0
    total = sum(f.memory_usage(index=True).sum() for f in tables.values())
    if total < SPILL_MIN_BYTES:
        return 0
    spilled = _write(tables)
    # from here on, only weak references: nothing here may keep a
    # table in memory when checking below whether it was freed
    tables = {key: weakref.ref(tables[key]) for key in spilled}
    buffers = {key: _buffers(table()) for key, table in tables.items()}
    _replace(holders, references, spilled)
    gc.collect()
    nbytes = 0
    for key, moved in spilled.items():
        frame = tables[key]()
        if frame is not None:
            _log.info(
                "Localizations kept in memory: still referenced by %s",
                [type(r).__name__ for r in gc.get_referrers(frame)],
            )
            _put_back(holders, references, moved, frame)
            moved.discard()
            continue
        nbytes += moved.nbytes
        if any(buffer() is not None for buffer in buffers[key]):
            _log.info("Localizations moved to disk, but still in memory")
    for holder in holders:
        if all(moved is None for moved in holder._spilled_locs or ()):
            holder._spilled_locs = None
    return nbytes


def _write(tables: dict) -> dict:
    """``io.SpilledFrame`` of each table that could be moved to disk,
    by the same keys."""
    spilled = {}
    for key, frame in tables.items():
        try:
            spilled[key] = io.SpilledFrame(frame)
        except io.SpillError as error:
            _log.warning("Localizations kept in memory: %s", error)
    return spilled


def _buffers(frame) -> list:
    """Weak references to the arrays holding the columns of ``frame``,
    to check that moving it to disk freed them."""
    arrays = {}
    for i in range(frame.shape[1]):
        array = frame.iloc[:, i].to_numpy()
        while isinstance(array.base, np.ndarray):
            array = array.base
        arrays[id(array)] = array
    return [weakref.ref(array) for array in arrays.values()]


def _replace(holders: list, references: list, spilled: dict) -> None:
    """Swap the tables moved to disk for their placeholders."""
    for holder in holders:
        placeholders = [spilled.get(id(frame)) for frame in holder._locs]
        if any(moved is not None for moved in placeholders):
            holder._locs = [
                frame if moved is None else None
                for frame, moved in zip(holder._locs, placeholders)
            ]
            holder._spilled_locs = placeholders
    for reference in references:
        reference[:] = [spilled.get(id(item), item) for item in reference]


def _put_back(holders: list, references: list, moved, frame) -> None:
    """Undo ``_replace`` for one table that is still in memory."""
    for holder in holders:
        for i, placeholder in enumerate(holder._spilled_locs or ()):
            if placeholder is moved:
                holder._locs[i] = frame
                holder._spilled_locs[i] = None
    for reference in references:
        reference[:] = [frame if item is moved else item for item in reference]
