"""
picasso.gui.render_undo
~~~~~~~~~~~~~~~~~~~~~~~

Undo and Redo in Picasso: Render (File > Undo, File > Redo).

Every action that changes the localizations of the loaded channels,
e.g., the Move tool, drift correction, removing localizations or
linking, is one step of the ``History`` of its ``View``, and so is every
action that changes the picks or the measured points (the overlays,
kept whole, as they are small):

- When the step begins, the state of every channel is taken
  (``ChannelState``). It holds references, not copies, so taking it
  costs next to nothing.
- When the step ends, each channel that changed keeps what puts it back
  (``ChannelRecord``). If the rows and columns are the same, e.g., after
  a move or a drift correction, that is only the columns whose memory
  changed (x and y, say); otherwise, the whole table.
- Undoing a step puts its records back and records the current state of
  the same channels in the same way, for Redo; redoing puts back the
  results of the action rather than running it again.

The arrays a step keeps never change: the actions assign new columns
rather than writing into the existing ones (see ``View._set_channel_xy``
and ``lib.fit_canvas``), and an array that still shares memory with the
current localizations is copied. The history thus neither follows later
changes nor keeps the current localizations in memory once they are
moved to disk while idle (see ``picasso.gui.idle``).

The steps keep at most ``idle.SPILL_MIN_BYTES`` in memory. The tables of
a step that does not fit are moved to disk (``io.SpilledFrame``) in the
background and read back when the step is undone, so that the history
does not multiply the memory used by large data sets. At most
``UNDO_LIMIT`` steps that change the localizations are kept, and at most
``OVERLAY_UNDO_LIMIT`` that change only the overlays, so that clicking
picks does not push out a drift correction.

This module does not know the window: the ``View`` takes the states
(``History.take``), applies the records (``View._apply_history_step``)
and redraws.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from .. import io
from . import idle

#: Number of steps that change the localizations that can be undone in
#: a Render window.
UNDO_LIMIT = 10

#: Number of steps that change only the picks or the measured points
#: that can be undone in a Render window.
OVERLAY_UNDO_LIMIT = 50

#: Longest step name shown in the tooltips of Undo and Redo.
LABEL_LENGTH = 60

#: The value of ``Step.group_color`` when the step did not change it.
UNCHANGED = object()

_log = logging.getLogger(__name__)


class StaleHistoryError(Exception):
    """The localizations no longer match the history, e.g., after a
    plugin changed them, so its steps cannot be put back."""


class Channels(NamedTuple):
    """The per-channel lists of a ``View`` that the history reads, see
    ``View._history_channels``."""

    locs: list[pd.DataFrame]
    infos: list[list[dict]]
    drift: list[pd.DataFrame | None]
    driftfiles: list[str | None]


def shorten(label: str) -> str:
    """Cut ``label`` to ``LABEL_LENGTH`` characters."""
    if len(label) <= LABEL_LENGTH:
        return label
    return label[: LABEL_LENGTH - 1] + "…"


def _columns(locs: pd.DataFrame) -> dict[str, np.ndarray]:
    """The columns of ``locs`` as arrays, without copies."""
    return {name: locs[name].to_numpy() for name in locs.columns}


def _same_memory(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether ``a`` and ``b`` are the same values in the same memory,
    i.e., the column was not replaced."""
    return a is b or (
        a.__array_interface__["data"][0] == b.__array_interface__["data"][0]
        and a.shape == b.shape
        and a.strides == b.strides
        and a.dtype == b.dtype
    )


def _same_values(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether ``a`` and ``b`` hold the same values, e.g., a column that
    was copied but not changed (assigning a column copies it); NaN
    equals NaN."""
    if _same_memory(a, b):
        return True
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def _unshared(
    columns: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """``columns`` with copies of the arrays that share memory with the
    ``current`` columns of a channel."""
    return {
        name: (
            array.copy()
            if any(np.may_share_memory(array, c) for c in current.values())
            else array
        )
        for name, array in columns.items()
    }


def _equal(a: object, b: object) -> bool:
    """``a == b``, False if the comparison fails (e.g., for arrays)."""
    try:
        return bool(a == b)
    except Exception:
        return False


def _drift_equal(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.equals(b)


class ChannelState:
    """One channel when a step began: its columns and index (references),
    its metadata (the dictionaries copied, as steps set their keys) and
    its drift (copied, as ``View.add_drift`` adds to it in place).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations of the channel.
    info : list of dicts
        Metadata of the channel.
    drift : pd.DataFrame or None
        Drift applied to the channel, see ``View._drift``.
    driftfile : str or None
        The last drift file of the channel, see ``View._driftfiles``.
    """

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        drift: pd.DataFrame | None,
        driftfile: str | None,
    ) -> None:
        self.frame = locs
        self.columns = _columns(locs)
        self.index = locs.index
        self.info = [dict(entry) for entry in info]
        self.drift = None if drift is None else drift.copy()
        self.driftfile = driftfile

    def record(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        drift: pd.DataFrame | None,
        driftfile: str | None,
    ) -> ChannelRecord | None:
        """What puts the channel back to this state from its current
        state (the arguments), None if it did not change."""
        current = _columns(locs)
        same_table = list(current) == list(self.columns) and (
            locs.index is self.index
            or (
                len(locs.index) == len(self.index)
                and locs.index.equals(self.index)
            )
        )
        if same_table:
            changed = {
                name: array
                for name, array in self.columns.items()
                if not _same_values(array, current[name])
            }
            table = None
            if changed:
                changed = _unshared(changed, current)
                table = _Table(pd.DataFrame(changed, copy=False))
        else:
            table = _Table(self._full_table(current))
        info_changed = not _equal(self.info, info)
        drift_changed = not _drift_equal(self.drift, drift) or not _equal(
            self.driftfile, driftfile
        )
        if table is None and not info_changed and not drift_changed:
            return None
        return ChannelRecord(
            full=not same_table,
            table=table,
            n_rows=len(locs),
            info=self.info if info_changed else None,
            drift=(self.drift, self.driftfile) if drift_changed else None,
        )

    def _full_table(self, current: dict[str, np.ndarray]) -> pd.DataFrame:
        """The whole table of this state. The table itself if nothing
        changed it since and it shares no memory with the current
        localizations, as usual (no copy; linked windows that held it
        recognize it), else a new one with copies of the shared
        columns."""
        frame = self.frame
        untouched = (
            frame.index is self.index
            and list(frame.columns) == list(self.columns)
            and all(
                _same_memory(array, frame[name].to_numpy())
                for name, array in self.columns.items()
            )
        )
        columns = _unshared(self.columns, current)
        if untouched and all(
            columns[name] is array for name, array in self.columns.items()
        ):
            return frame
        return pd.DataFrame(columns, index=self.index, copy=False)


@dataclass(eq=False)
class ChannelRecord:
    """What a step changed in one channel, to put it back.

    Attributes
    ----------
    full : bool
        Whether ``table`` is the whole table of localizations, which
        replaces the channel's, or only the columns that changed, which
        replace the channel's columns of the same name.
    table : _Table or None
        The localizations, None if they did not change.
    n_rows : int
        Number of localizations of the channel that the record is put
        back into; the columns of a partial record must fit them.
    info : list of dicts or None
        The metadata, None if it did not change.
    drift : tuple or None
        The drift and the drift file, None if they did not change.
    """

    full: bool
    table: _Table | None
    n_rows: int
    info: list[dict] | None
    drift: tuple[pd.DataFrame | None, str | None] | None

    def fits(self, locs: pd.DataFrame) -> bool:
        """Whether the record can be put back into ``locs``."""
        if self.full or self.table is None:
            return True
        return len(locs) == self.n_rows and set(self.table.columns) <= set(
            locs.columns
        )

    def opposite(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        drift: pd.DataFrame | None,
        driftfile: str | None,
    ) -> ChannelRecord:
        """The record that puts the channel back to its current state
        (the arguments) once this one was put back."""
        table = None
        if self.table is not None:
            if self.full:
                table = _Table(locs)
            else:
                columns = {
                    name: locs[name].to_numpy() for name in self.table.columns
                }
                table = _Table(pd.DataFrame(columns, copy=False))
        return ChannelRecord(
            full=self.full,
            table=table,
            n_rows=self.table.n_rows if self.table is not None else len(locs),
            info=None if self.info is None else [dict(i) for i in info],
            drift=(
                None
                if self.drift is None
                else (None if drift is None else drift.copy(), driftfile)
            ),
        )

    def unshare(self, locs: pd.DataFrame) -> None:
        """Copy the arrays that share memory with ``locs``, the channel's
        localizations after this record was taken."""
        if self.table is not None:
            self.table.unshare(_columns(locs))


class _Table:
    """The localizations kept by a record, in memory or moved to disk
    (``spill``, run on the writer thread, see ``_write_later``).

    Parameters
    ----------
    frame : pd.DataFrame
        The localizations; nothing changes them.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._lock = threading.Lock()
        self._frame = frame
        self._spilled = None  # io.SpilledFrame once moved to disk
        self._directory = None  # where it is moved, see write_later
        self._discarded = False
        self.columns = list(frame.columns)
        self.n_rows = len(frame)
        index = frame.index
        default_index = isinstance(index, pd.RangeIndex)
        self.nbytes = int(
            sum(frame[name].to_numpy().nbytes for name in frame.columns)
            + (0 if default_index else index.to_numpy().nbytes)
        )

    @property
    def in_memory(self) -> bool:
        """Whether the table is in memory (also while being moved)."""
        with self._lock:
            return self._spilled is None and not self._discarded

    @property
    def on_disk(self) -> bool:
        """Whether the table was moved to disk."""
        with self._lock:
            return self._spilled is not None

    def write_later(self) -> None:
        """Move the table to disk in the background, into the folder of
        ``io.spill_directory`` at this time."""
        with self._lock:
            if self._directory is not None or self._discarded:
                return
            self._directory = io.spill_directory()
        _write_later(self)

    def spill(self) -> None:
        """Move the table to disk; it stays in memory if it cannot be
        moved, e.g., for lack of disk space."""
        with self._lock:
            frame = self._frame
            if frame is None or self._discarded:
                return
        try:
            spilled = io.SpilledFrame(frame, self._directory)
        except io.SpillError as error:
            _log.warning("Undo history kept in memory: %s", error)
            return
        with self._lock:
            if self._discarded:
                spilled.discard()
                return
            self._spilled = spilled
            self._frame = None

    def load(self) -> pd.DataFrame:
        """The localizations, read back if moved to disk.

        Raises
        ------
        io.SpillError
            If they cannot be read back.
        """
        with self._lock:
            frame, spilled = self._frame, self._spilled
        if frame is not None:
            return frame
        return spilled.load()

    def unshare(self, current: dict[str, np.ndarray]) -> None:
        """Copy the columns that share memory with ``current``."""
        with self._lock:
            frame = self._frame
        if frame is None:  # on disk
            return
        columns = _columns(frame)
        unshared = _unshared(columns, current)
        if any(unshared[name] is not columns[name] for name in columns):
            frame = pd.DataFrame(unshared, index=frame.index, copy=False)
            with self._lock:
                if self._frame is not None:
                    self._frame = frame

    def discard(self) -> None:
        """Drop the table and remove it from the disk."""
        with self._lock:
            self._discarded = True
            self._frame = None
            spilled, self._spilled = self._spilled, None
        if spilled is not None:
            spilled.discard()


@dataclass(eq=False)
class Step:
    """One step of the history: the records of the channels it changed
    and, if it changed them, the group colors and the overlays.

    Attributes
    ----------
    label : str
        Name of the action, shown in the tooltips of Undo and Redo.
    records : dict
        ``ChannelRecord`` by channel index.
    group_color : np.ndarray or list
        The group colors (``View.group_color``) to put back, or
        ``UNCHANGED``.
    overlays : dict
        The overlays the step changed, to put back, by name: ``"picks"``
        (the picks, their shapes and sizes) and ``"points"`` (the
        measured points), see ``View._overlay_state``.
    """

    label: str
    records: dict[int, ChannelRecord]
    group_color: object = UNCHANGED
    overlays: dict[str, object] = field(default_factory=dict)

    @property
    def changes_locs(self) -> bool:
        """Whether the step changes the localizations or the group
        colors, rather than only the overlays (see ``UNDO_LIMIT``)."""
        return bool(self.records) or self.group_color is not UNCHANGED

    def tables(self) -> list[_Table]:
        return [r.table for r in self.records.values() if r.table is not None]

    @property
    def nbytes_in_memory(self) -> int:
        """Size of the tables of the step that are in memory."""
        return sum(t.nbytes for t in self.tables() if t.in_memory)

    def write_later(self) -> None:
        """Move the tables of the step to disk in the background."""
        for table in self.tables():
            table.write_later()

    def fits(self, channels: Channels) -> bool:
        """Whether the step can be put back into ``channels``."""
        return all(
            index < len(channels.locs) and record.fits(channels.locs[index])
            for index, record in self.records.items()
        )

    def opposite(
        self, channels: Channels, group_color: object, overlays: dict
    ) -> Step:
        """The step that puts back the current state (the arguments) of
        the channels, group colors and overlays this step changes."""
        return Step(
            self.label,
            {
                index: record.opposite(
                    channels.locs[index],
                    channels.infos[index],
                    channels.drift[index],
                    channels.driftfiles[index],
                )
                for index, record in self.records.items()
            },
            UNCHANGED if self.group_color is UNCHANGED else group_color,
            {name: overlays[name] for name in self.overlays},
        )

    def unshare(self, channels: Channels) -> None:
        """Copy the arrays that share memory with ``channels``."""
        for index, record in self.records.items():
            record.unshare(channels.locs[index])

    def discard(self) -> None:
        """Drop the tables of the step and remove them from the disk."""
        for table in self.tables():
            table.discard()


@dataclass
class _Begun:
    """The state taken when the outermost step began."""

    label: str
    channels: list[ChannelState]
    group_color: object
    overlays: dict[str, object]


class History:
    """The steps to undo and to redo of one ``View``.

    A step is recorded between ``begin`` and ``end``: the caller takes
    the state after the outermost ``begin`` (``take``) and records the
    step after the outermost ``end`` (``record``). Steps begun inside
    another step belong to it, e.g., each channel of an action applied
    to all of them.

    Attributes
    ----------
    busy : bool
        Whether a step is being recorded, undone or redone; changes of
        the localizations then belong to it.
    """

    def __init__(self) -> None:
        self._undo: list[Step] = []
        self._redo: list[Step] = []
        self._depth = 0
        self._label = ""
        self._begun: _Begun | None = None
        self._applying = False

    @property
    def busy(self) -> bool:
        return self._depth > 0 or self._applying

    @property
    def undo_label(self) -> str | None:
        """Name of the step to undo, None if there is none."""
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        """Name of the step to redo, None if there is none."""
        return self._redo[-1].label if self._redo else None

    @property
    def overlays_taken(self) -> tuple[str, ...]:
        """Names of the overlays taken for the step being recorded, whose
        current state ``record`` needs, see ``take``."""
        return () if self._begun is None else tuple(self._begun.overlays)

    def begin(self, label: str) -> bool:
        """Begin a step named ``label``.

        Returns
        -------
        outermost : bool
            True if this is the outermost step, whose state the caller
            then takes (``take``).
        """
        self._depth += 1
        if self._depth == 1:
            self._label = label
            self._begun = None
        return self._depth == 1

    def take(
        self,
        channels: Channels,
        group_color: object,
        overlays: dict | None = None,
    ) -> None:
        """Take the state of ``channels`` and of the ``overlays`` given
        (by name, see ``Step.overlays``) before the outermost step."""
        self._begun = _Begun(
            self._label,
            [
                ChannelState(*state)
                for state in zip(
                    channels.locs,
                    channels.infos,
                    channels.drift,
                    channels.driftfiles,
                )
            ],
            group_color,
            dict(overlays or {}),
        )

    def end(self) -> bool:
        """End a step.

        Returns
        -------
        outermost : bool
            True if the outermost step ended; the caller then records it
            (``record``).
        """
        self._depth = max(0, self._depth - 1)
        return self._depth == 0

    def record(
        self,
        channels: Channels,
        group_color: object,
        overlays: dict | None = None,
    ) -> bool:
        """Record the outermost step from the state taken and the current
        state (the arguments), if anything changed.

        Returns
        -------
        recorded : bool
            Whether a step was recorded.
        """
        begun, self._begun = self._begun, None
        if begun is None or len(begun.channels) != len(channels.locs):
            return False  # channels were added or closed meanwhile
        records = {}
        for index, state in enumerate(begun.channels):
            record = state.record(
                channels.locs[index],
                channels.infos[index],
                channels.drift[index],
                channels.driftfiles[index],
            )
            if record is not None:
                records[index] = record
        group_changed = group_color is not begun.group_color
        overlays = overlays or {}
        changed_overlays = {
            name: state
            for name, state in begun.overlays.items()
            if not _equal(state, overlays.get(name))
        }
        if not records and not group_changed and not changed_overlays:
            return False
        step = Step(
            shorten(begun.label),
            records,
            begun.group_color if group_changed else UNCHANGED,
            changed_overlays,
        )
        self._undo.append(step)
        self._trim()
        for dropped in self._redo:
            dropped.discard()
        self._redo.clear()
        self._store(step)
        return True

    def _trim(self) -> None:
        """Drop the oldest steps beyond the limits. Beyond
        ``UNDO_LIMIT``, the oldest step that changes the localizations
        goes with all steps before it, which may depend on it (e.g.,
        picks before the canvas was translated); a step that changes
        only the overlays depends on none, so beyond
        ``OVERLAY_UNDO_LIMIT``, the oldest one goes alone."""
        locs_steps = [s for s in self._undo if s.changes_locs]
        if len(locs_steps) > UNDO_LIMIT:
            first_kept = self._undo.index(locs_steps[-UNDO_LIMIT])
            for dropped in self._undo[:first_kept]:
                dropped.discard()
            del self._undo[:first_kept]
        overlay_steps = [s for s in self._undo if not s.changes_locs]
        for dropped in overlay_steps[:-OVERLAY_UNDO_LIMIT]:
            self._undo.remove(dropped)

    def undo(
        self,
        channels: Channels,
        group_color: object,
        overlays: dict,
        apply: Callable[[Step], None],
    ) -> bool:
        """Undo the last step: ``apply(step)`` puts back its records,
        after the current state of the channels it changed was recorded
        as the step to redo.

        Parameters
        ----------
        channels : Channels
            The current channels.
        group_color : np.ndarray or list
            The current group colors.
        overlays : dict
            The current overlays by name, see ``Step.overlays``.
        apply : callable
            Puts back the records of the step given and redraws.

        Returns
        -------
        undone : bool
            False if there is nothing to undo or a step is being
            recorded.

        Raises
        ------
        StaleHistoryError
            If the step does not fit the localizations any more; the
            history is cleared then.
        io.SpillError
            If the step moved to disk cannot be read back; the history
            is cleared then.
        """
        return self._swap(
            self._undo, self._redo, channels, group_color, overlays, apply
        )

    def redo(
        self,
        channels: Channels,
        group_color: object,
        overlays: dict,
        apply: Callable[[Step], None],
    ) -> bool:
        """Redo the last step undone, see ``undo``."""
        return self._swap(
            self._redo, self._undo, channels, group_color, overlays, apply
        )

    def _swap(
        self,
        source: list[Step],
        target: list[Step],
        channels: Channels,
        group_color: object,
        overlays: dict,
        apply: Callable[[Step], None],
    ) -> bool:
        if not source or self.busy:
            return False
        step = source[-1]
        if not step.fits(channels):
            self.clear()
            raise StaleHistoryError(step.label)
        source.pop()
        opposite = step.opposite(channels, group_color, overlays)
        self._applying = True
        try:
            apply(step)
        except BaseException:
            # e.g., a step on disk that cannot be read back: what was put
            # back is unknown, so no step fits any more
            opposite.discard()
            self.clear()
            raise
        finally:
            self._applying = False
            step.discard()
        opposite.unshare(channels)
        target.append(opposite)
        self._store(opposite)
        return True

    def _store(self, step: Step) -> None:
        """Move the tables of ``step`` to disk if the history would keep
        more than ``idle.SPILL_MIN_BYTES`` in memory with them."""
        in_memory = sum(
            other.nbytes_in_memory
            for other in self._undo + self._redo
            if other is not step
        )
        if in_memory + step.nbytes_in_memory > idle.SPILL_MIN_BYTES:
            step.write_later()

    def clear(self) -> None:
        """Forget all steps, e.g., when channels are added or closed."""
        for step in self._undo + self._redo:
            step.discard()
        self._undo.clear()
        self._redo.clear()

    def reorder(self, order: list[int]) -> None:
        """Follow reordered channels: the channel at index ``order[i]``
        moved to index ``i``, see ``View.reorder_channels``."""
        new_index = {old: new for new, old in enumerate(order)}
        for step in self._undo + self._redo:
            step.records = {
                new_index[old]: record
                for old, record in step.records.items()
                if old in new_index
            }
        if self._begun is not None:
            self._begun.channels = [self._begun.channels[i] for i in order]

    def steps(self) -> tuple[list[Step], list[Step]]:
        """The steps to undo and to redo, the last ones at the end."""
        return list(self._undo), list(self._redo)


# --- moving tables to disk in the background --- #

_writes: queue.Queue = queue.Queue()
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()


def _write_later(table: _Table) -> None:
    """Move ``table`` to disk on the writer thread."""
    global _writer
    with _writer_lock:
        if _writer is None or not _writer.is_alive():
            # a daemon thread does not delay quitting; a table written
            # halfway is removed at the next start (remove_stale_spills)
            _writer = threading.Thread(
                target=_write_loop, name="picasso-undo-writer", daemon=True
            )
            _writer.start()
    _writes.put(table)


def _write_loop() -> None:
    while True:
        table = _writes.get()
        try:
            table.spill()
        except Exception:  # never stop the thread
            _log.exception("A step of the undo history was not moved to disk")
        finally:
            _writes.task_done()


def wait_for_writes() -> None:
    """Wait until the tables queued for disk are written, e.g., in
    tests."""
    _writes.join()
