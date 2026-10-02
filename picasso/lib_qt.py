"""
picasso.lib_qt
~~~~~~~~~~~~~~

Qt-dependent (PyQt6) handy classes and functions, split out of
``picasso.lib`` so that importing ``picasso.lib`` does not require
PyQt6. All names defined here remain accessible as ``lib.<name>`` -
``picasso.lib`` forwards them lazily via a module ``__getattr__``, so
PyQt6 is only imported on first use.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import time
import traceback
from collections.abc import Callable
from typing import TypeAlias

import yaml
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtWidgets, QtGui, sip
from playsound3 import playsound

from picasso import diagnostics, docs_url, io
from picasso.lib import (
    _dialogs,
    SOUND_NOTIFICATION_DURATION,
    REQUIRED_COLUMNS,
    MockProgress,
    OperationCanceled,
    TqdmProgress,
    get_sound_notification_path,
    is_path_available,
)


class Dialog(QtWidgets.QDialog):
    """Base class for dialogs without 'What's this?' help."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._focus_buttons = ["OK"]
        self.setWindowFlag(
            QtCore.Qt.WindowType.WindowContextHelpButtonHint, False
        )

    def showEvent(self, event):
        """Remove focus from any QPushButton when the dialog is shown,
        so that pressing Enter does not trigger any button by default
        (unless it's called "OK").

        Parameters
        ----------
        event : QtGui.QShowEvent
            The Qt show event.
        """
        super().showEvent(event)
        for button in self.findChildren(QtWidgets.QPushButton):
            if button.text() in self._focus_buttons:
                continue
            button.setDefault(False)
            button.setAutoDefault(False)


class UserSettingsDialog(Dialog):
    """Dialog for inspecting and editing the user settings YAML file."""

    #: Reference listing every settings key, its default and what it
    #: does, grouped by section - see docs/others.rst.
    DOCS_URL = docs_url("others.html#user-settings-file")

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("User Settings")
        self.setModal(False)
        self.resize(600, 500)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        path_label = QtWidgets.QLabel(
            f"Settings file: {io._user_settings_filename()}\n"
            "Warning: editing this file can affect the behavior of Picasso.\n"
            "Clearing the file will reset all settings to their default "
            "values."
        )
        path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        header.addWidget(path_label, 1)
        help_button = HelpButton(self.DOCS_URL)
        header.addWidget(help_button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        self.editor = QtWidgets.QPlainTextEdit()
        # fixed width for the YAML indentation, in the size of the
        # application's font
        font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        )
        if self.font().pointSizeF() > 0:
            font.setPointSizeF(self.font().pointSizeF())
        self.editor.setFont(font)
        layout.addWidget(self.editor)

        button_layout = QtWidgets.QHBoxLayout()
        reload_button = QtWidgets.QPushButton("Reload")
        reload_button.clicked.connect(self.load_settings)
        button_layout.addWidget(reload_button)
        button_layout.addStretch()
        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Re-read the settings from disk each time the dialog opens.

        Parameters
        ----------
        event : QtGui.QShowEvent
            The Qt show event.
        """
        super().showEvent(event)
        self.load_settings()

    def load_settings(self) -> None:
        """Read the settings file and display its contents."""
        filename = io._user_settings_filename()
        try:
            with open(filename, "r") as f:
                self.editor.setPlainText(f.read())
        except FileNotFoundError:
            self.editor.setPlainText(
                "# No settings file found. Edit and save to create one."
            )

    def save_settings(self) -> None:
        """Validate YAML and write back to the settings file."""
        text = self.editor.toPlainText()
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid YAML",
                f"Cannot save — the YAML is invalid:\n\n{e}",
            )
            return
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid settings",
                "Settings must be a YAML mapping (key: value pairs).",
            )
            return
        io.save_user_settings(parsed)
        QtWidgets.QMessageBox.information(
            self, "Saved", "User settings saved successfully."
        )


def notify_settings_load_error(parent=None) -> bool:
    """Tell the user, once, when the settings file could not be read
    (see ``io.settings_load_error``): default settings are in use, the
    unreadable file was kept as a copy and the file will be rewritten by
    the next save. Returns whether a message was shown."""
    error = io.settings_load_error()
    if error is None:
        return False
    message, kept = error
    text = (
        "The user settings file could not be read, so default settings "
        f"are in use:\n\n{message}\n\n"
    )
    if kept:
        text += f"A copy of the unreadable file was kept as\n{kept}\n\n"
    text += (
        "Picasso rewrites the settings file whenever settings change and "
        "keeps the previous version as settings.yaml.bak. To recover your "
        "settings, fix the YAML in the kept copy and paste it into "
        "File > Picasso settings."
    )
    QtWidgets.QMessageBox.warning(parent, "Settings could not be read", text)
    io.dismiss_settings_load_error()
    return True


class MetadataDialog(Dialog):
    """Dialog for inspecting YAML metadata (list of lists of dicts).

    Can be used standalone with any ``infos`` data, making it reusable
    across Picasso modules.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Metadata")
        self.setModal(False)
        self.resize(700, 500)

        layout = QtWidgets.QVBoxLayout(self)

        # channel selector
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.currentIndexChanged.connect(self._on_channel_changed)
        selector_layout.addWidget(self.channel_box)
        selector_layout.addStretch(1)

        # copy button
        copy_button = QtWidgets.QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._copy_to_clipboard)
        selector_layout.addWidget(copy_button)

        layout.addLayout(selector_layout)

        # tree widget for structured metadata display
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setStretchLastSection(True)
        self.tree.setColumnWidth(0, 250)
        layout.addWidget(self.tree)

        self._infos: list[list[dict]] = []
        self._labels: list[str] = []

    def set_infos(
        self,
        infos: list[list[dict]] | list[dict],
        labels: list[str] | str | None = None,
    ) -> None:
        """Set metadata and refresh the display. The user can provide
        the metadata and the label for a single channel as a list of
        dicts and a single string, respectively, or for multiple
        channels as a list of lists of dicts and a list of strings,
        respectively.

        Parameters
        ----------
        infos : list of list of dict or list of dict
            Metadata for each channel. Each element is a list of dicts
            as loaded from a YAML file.
        labels : list of str, optional
            Display labels for each channel (e.g., file paths).
        """
        if isinstance(infos, list) and all(isinstance(i, dict) for i in infos):
            infos = [infos]  # wrap single list of dicts into a list
        if isinstance(labels, str):
            labels = [labels]  # wrap single label into a list
        self._infos = infos
        self._labels = labels or [f"Channel {i}" for i in range(len(infos))]
        self.channel_box.blockSignals(True)
        self.channel_box.clear()
        self.channel_box.addItems(self._labels)
        self.channel_box.blockSignals(False)
        if infos:
            self._on_channel_changed(0)

    def _on_channel_changed(self, index: int) -> None:
        """Populate tree with metadata from the selected channel."""
        self.tree.clear()
        if index < 0 or index >= len(self._infos):
            return
        info_list = self._infos[index]
        for i, info_dict in enumerate(info_list):
            section_label = info_dict.get("Generated by", f"Section {i}")
            section_item = QtWidgets.QTreeWidgetItem(
                [f"[{i}] {section_label}", ""]
            )
            section_item.setExpanded(True)
            font = section_item.font(0)
            font.setBold(True)
            section_item.setFont(0, font)
            self._add_dict_to_tree(section_item, info_dict)
            self.tree.addTopLevelItem(section_item)
        self.tree.expandAll()

    def _add_dict_to_tree(
        self,
        parent: QtWidgets.QTreeWidgetItem,
        data: dict | list | object,
    ) -> None:
        """Recursively add dict/list contents to a tree item."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    child = QtWidgets.QTreeWidgetItem([str(key), ""])
                    self._add_dict_to_tree(child, value)
                    parent.addChild(child)
                else:
                    child = QtWidgets.QTreeWidgetItem([str(key), str(value)])
                    parent.addChild(child)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                if isinstance(value, (dict, list)):
                    child = QtWidgets.QTreeWidgetItem([f"[{i}]", ""])
                    self._add_dict_to_tree(child, value)
                    parent.addChild(child)
                else:
                    child = QtWidgets.QTreeWidgetItem([f"[{i}]", str(value)])
                    parent.addChild(child)

    def _copy_to_clipboard(self) -> None:
        """Copy the current channel's metadata to clipboard as YAML."""

        index = self.channel_box.currentIndex()
        if index < 0 or index >= len(self._infos):
            return
        text = yaml.dump_all(
            self._infos[index], default_flow_style=False, sort_keys=False
        )
        QtWidgets.QApplication.clipboard().setText(text)


class ProgressDialog(QtWidgets.QProgressDialog):
    """ProgressDialog displays a progress dialog with a progress bar."""

    # pump the event loop on every update, such that the dialog repaints
    # while the computation blocks the GUI thread; not needed (and
    # re-entrant) when the computation runs on a worker thread
    _process_events = True
    # play the finish sound when the bar reaches its maximum; a task with
    # several phases plays it once the whole task is done instead
    _sound_on_maximum = True

    def __init__(self, description, minimum, maximum, parent):
        # append time estimate to description
        super().__init__(
            description,
            None,
            minimum,
            maximum,
            parent,
            QtCore.Qt.WindowType.CustomizeWindowHint,
        )
        self.description_base = description  # without time estimate
        self.initalized = None

    def init(self):
        """Arm the dialog: register it, make it modal and start the clock.

        Called lazily on the first :meth:`set_value` so that a fast operation
        never flashes a dialog.
        """
        _dialogs.append(self)
        self.setMinimumDuration(500)
        self.setModal(True)
        self.t0 = time.time()
        self.app = QtCore.QCoreApplication.instance()
        self.initalized = True
        self.count_started = False
        self.finished = False
        # sound notification
        self.sound_notification_path = get_sound_notification_path()

    def set_value(self, value):
        """Advance the bar, arming the dialog and the time estimate as needed.

        Parameters
        ----------
        value : int
            Cumulative progress so far.
        """
        if not self.initalized:
            self.init()
        self.setValue(value)
        if self.count_started:
            # estimate time left
            elapsed = time.time() - self.t0_est
            remaining = int(
                (self.maximum() - value) * elapsed / (value + 1e-6)
            )
            # convert to hh-mm-ss
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            # format time estimate
            if hours > 0:
                hours = min(10, hours)  # limit hours to 10 for display
                time_estimate = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"
            else:
                time_estimate = f"{minutes:02d}m:{seconds:02d}s"
            # set label text with time estimate
            description = (
                f"{self.description_base}"
                f"\nEstimated time remaining: {time_estimate}"
            )
            self.setLabelText(description)
        # sound notification
        if (
            self._sound_on_maximum
            and value >= self.maximum()
            and self.finished is False
        ):
            self.finished = True
            self.play_sound_notification()
        # if value is above zero, count has started, enabling time estimate
        if not self.count_started:
            if value > 0:
                self.count_started = True
                self.t0_est = time.time()
        if self._process_events:
            self.app.processEvents()

    def close(self):
        """Close the dialog for good, cancelling a pending delayed show.

        The first ``set_value`` arms Qt's ``minimumDuration`` timer, which
        calls ``forceShow()`` when it fires. If the task finishes before
        that (which happens with small datasets, while long tasks show the
        dialog on time), the timer would pop the dialog up *after* it was
        closed, leaving it on screen with no one left to close it.
        ``reset()`` stops that timer, hides the bar and clears Qt's
        internal show state, so the dialog cannot re-appear.
        """
        if self.initalized:
            self.reset()
        return super().close()

    def closeEvent(self, event):
        """Unregister the dialog and play the finish sound if it was not
        already played.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event.
        """
        if self in _dialogs:  # not armed yet, or closed twice
            _dialogs.remove(self)
        if self.finished is False:
            self.finished = True
            self.play_sound_notification()

    def zero_progress(self, description=None):
        """Set progress dialog to zero and change the title if given.

        Parameters
        ----------
        description : str, optional
            Label of the new phase. None keeps the current one.
        """
        if description:
            self.setLabelText(description)
            self.description_base = description
        if self.initalized:
            # restart the time-estimate baseline so the next non-zero
            # set_value re-arms the timer for the new phase
            self.count_started = False
        self.set_value(0)

    def play_sound_notification(self):
        """Play a sound notification if a sound file is specified and
        at least a minute has passed since the dialog was opened.

        A missing or broken audio backend (which happens in the one-click
        builds) is logged, not raised: closing a dialog must always
        succeed, especially while an error is being reported.
        """
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                try:
                    playsound(self.sound_notification_path, block=False)
                except Exception:  # noqa: BLE001 - notification only
                    diagnostics.log_message(
                        "Could not play the sound notification"
                        f" {self.sound_notification_path}:\n"
                        f"{traceback.format_exc()}"
                    )

    def get_iterator(self, start=None, end=None):
        """Get an iterator that spans the dialog's remaining progress.

        Parameters
        ----------
        start, end : int, optional
            First and one-past-last value. None uses the dialog's current
            value and maximum.

        Returns
        -------
        iterator : range
        """
        start = self.value() if start is None else start
        end = self.maximum() if end is None else end
        return range(start, end)


class StatusDialog(Dialog):
    """StatusDialog displays the description string in a dialog."""

    def __init__(self, description, parent):
        super(StatusDialog, self).__init__(
            parent,
            QtCore.Qt.WindowType.CustomizeWindowHint,
        )
        _dialogs.append(self)
        vbox = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(description)
        vbox.addWidget(label)
        self.sound_notification_path = get_sound_notification_path()
        self.t0 = time.time()
        self.show()
        self.raise_()
        # Paint the dialog before the caller starts its (blocking) work.
        # A single processEvents() is not enough: the window may be mapped
        # before its paint event is delivered, leaving an empty dialog for
        # the whole computation. repaint() forces the paint synchronously.
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.processEvents()
        self.repaint()
        if app is not None:
            app.processEvents()

    def closeEvent(self, event):
        """Unregister the dialog and play the notification sound if the task
        ran long enough to warrant one.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event.
        """
        if self in _dialogs:  # closed twice
            _dialogs.remove(self)
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                try:
                    playsound(self.sound_notification_path, block=False)
                except Exception:  # noqa: BLE001 - notification only
                    diagnostics.log_message(
                        "Could not play the sound notification"
                        f" {self.sound_notification_path}:\n"
                        f"{traceback.format_exc()}"
                    )


class TaskProgressDialog(ProgressDialog):
    """Progress dialog of a task running on a worker thread, see
    :func:`run_task`.

    Unlike :class:`ProgressDialog`, it has a Cancel button and is driven
    by the worker's progress signals instead of pumping the event loop.
    Cancel (also Escape or closing the dialog) does not hide it: it
    emits ``cancel_requested`` and shows "Canceling..." until the worker
    has reached its next cancellation point and stopped. The task closes
    the dialog with :meth:`finish`.

    A maximum of 0 shows a busy indicator, for steps that report no
    progress.
    """

    _process_events = False
    _sound_on_maximum = False

    cancel_requested = QtCore.pyqtSignal()

    def __init__(self, description, maximum, parent, title=None):
        super().__init__(description, 0, maximum, parent)
        self.setCancelButtonText("Cancel")
        # QProgressDialog hides itself on cancel and, with auto-reset and
        # auto-close, at the maximum; only the task may close this one
        self.canceled.disconnect(self.cancel)
        self.canceled.connect(self.request_cancel)
        self.setAutoReset(False)
        self.setAutoClose(False)
        if title:
            self.setWindowTitle(title)
        self.cancel_was_requested = False
        self._closing = False
        self.set_value(0)  # arm: modal, delayed show, clock

    def set_value(self, value):
        """Advance the bar; see :meth:`ProgressDialog.set_value`.

        Parameters
        ----------
        value : int
            Cumulative progress so far, ignored by the busy indicator.
        """
        if self.maximum() == 0:  # busy indicator, no time estimate
            if not self.initalized:
                self.init()
            self.setValue(0)
            return
        super().set_value(value)

    def request_cancel(self):
        """Show that the task is being canceled and emit
        ``cancel_requested``, once."""
        if self.cancel_was_requested:
            return
        self.cancel_was_requested = True
        self.finished = True  # no finish sound for a canceled task
        self.setRange(0, 0)  # busy until the worker has stopped
        self.setLabelText(
            "Canceling...\nWaiting for the current step to finish."
        )
        button = self.findChild(QtWidgets.QPushButton)
        if button is not None:
            button.setEnabled(False)
        self.cancel_requested.emit()

    def reject(self):
        """Escape cancels the task instead of hiding the dialog."""
        self.canceled.emit()

    def closeEvent(self, event):
        """Cancel the task instead of closing, unless the task closes the
        dialog via :meth:`finish`.

        Parameters
        ----------
        event : QtGui.QCloseEvent
            The Qt close event.
        """
        if self._closing:
            super().closeEvent(event)
        else:
            event.ignore()
            self.canceled.emit()

    def finish(self, completed: bool) -> None:
        """Close the dialog for good.

        Parameters
        ----------
        completed : bool
            Whether the task ran to completion, which plays the finish
            sound if the task took long enough.
        """
        try:
            self.canceled.disconnect()
        except TypeError:  # finished twice
            pass
        if completed and not self.finished:
            self.play_sound_notification()
        self.finished = True
        self._closing = True
        self.close()


class TaskProgress(QtCore.QObject):
    """Progress tracker handed to the function run by :func:`run_task`.

    Implements the ``ProgressDialog`` interface (see
    ``lib.normalize_progress``), so analysis functions that take a
    progress dialog, or a progress callback such as ``progress.set_value``,
    run unchanged on the worker thread. Updates reach the dialog on the
    GUI thread through queued signals, at most every ``INTERVAL`` seconds.

    Once the user canceled the task, the next ``set_value`` (or
    ``zero_progress``, ``check_canceled``) raises
    ``lib.OperationCanceled``, so every progress update is a
    cancellation point.

    Parameters
    ----------
    description : str
        Label of the first phase.
    maximum : int
        Maximum of the first phase; 0 shows a busy indicator.
    """

    #: Minimum time between two value updates sent to the dialog, in s.
    INTERVAL = 0.05

    value_changed = QtCore.pyqtSignal(int)
    maximum_changed = QtCore.pyqtSignal(int)
    label_changed = QtCore.pyqtSignal(str)
    phase_changed = QtCore.pyqtSignal(str)

    def __init__(self, description: str, maximum: int) -> None:
        super().__init__()
        self.description_base = description
        self._maximum = int(maximum)
        self._value = 0
        self._last_emit = 0.0
        self._canceled = False  # set from the GUI thread

    @property
    def canceled(self) -> bool:
        """Whether the user canceled the task."""
        return self._canceled

    def cancel(self) -> None:
        """Request cancellation, which the next update raises."""
        self._canceled = True

    def check_canceled(self) -> None:
        """Raise ``lib.OperationCanceled`` if the task was canceled.

        For explicit cancellation points between steps that report no
        progress, e.g. before saving the result.

        Raises
        ------
        OperationCanceled
            If the user canceled the task.
        """
        if self._canceled:
            raise OperationCanceled

    def set_value(self, value, *args, **kwargs) -> None:
        """Report the cumulative progress of the current phase.

        Parameters
        ----------
        value : int
            Cumulative progress so far.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.

        Raises
        ------
        OperationCanceled
            If the user canceled the task.
        """
        self.check_canceled()
        self._value = int(value)
        now = time.monotonic()
        if (
            now - self._last_emit >= self.INTERVAL
            or self._value >= self._maximum
        ):
            self._last_emit = now
            self.value_changed.emit(self._value)

    def value(self) -> int:
        """The progress last reported."""
        return self._value

    def setMaximum(self, maximum, *args, **kwargs) -> None:
        """Set the maximum of the current phase.

        Parameters
        ----------
        maximum : int
            The value progress runs up to; 0 shows a busy indicator.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        self._maximum = int(maximum)
        self.maximum_changed.emit(self._maximum)

    def maximum(self) -> int:
        """The maximum of the current phase."""
        return self._maximum

    def setLabelText(self, text, *args, **kwargs) -> None:
        """Show ``text`` in the dialog, until the next time estimate.

        Parameters
        ----------
        text : str
            Label text.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        self.label_changed.emit(str(text))

    def zero_progress(self, description=None, *args, **kwargs) -> None:
        """Start a new phase at zero progress.

        Parameters
        ----------
        description : str, optional
            Label of the new phase. None keeps the current one.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.

        Raises
        ------
        OperationCanceled
            If the user canceled the task.
        """
        self.check_canceled()
        if description:
            self.description_base = description
        self._value = 0
        self._last_emit = time.monotonic()
        self.phase_changed.emit(self.description_base)

    def phase(self, description: str, maximum: int) -> None:
        """Start a new phase with its own label and maximum.

        Parameters
        ----------
        description : str
            Label of the new phase.
        maximum : int
            Maximum of the new phase; 0 shows a busy indicator.

        Raises
        ------
        OperationCanceled
            If the user canceled the task.
        """
        self.setMaximum(maximum)
        self.zero_progress(description)

    def callback(self, description: str, maximum: int) -> Callable:
        """A progress callback that starts its own phase when first
        called, for functions that report several steps through
        separate callbacks.

        Parameters
        ----------
        description : str
            Label of the phase.
        maximum : int
            Maximum of the phase.

        Returns
        -------
        callback : callable
            Takes the cumulative progress of the phase, like
            :meth:`set_value`.
        """
        started = False

        def callback(value, *args, **kwargs):
            nonlocal started
            if not started:
                started = True
                self.phase(description, maximum)
            self.set_value(value)

        return callback

    def get_iterator(self, start=None, end=None):
        """Get an iterator that spans the remaining progress.

        Parameters
        ----------
        start, end : int, optional
            First and one-past-last value. None uses the current value
            and maximum.

        Returns
        -------
        iterator : range
        """
        start = self._value if start is None else start
        end = self._maximum if end is None else end
        return range(start, end)

    def init(self, *args, **kwargs) -> None:
        """Do nothing; the dialog is armed by the task."""

    def update(self, *args, **kwargs) -> None:
        """Do nothing."""

    def close(self, *args, **kwargs) -> None:
        """Do nothing; the task closes the dialog when the function
        returns."""

    def closeEvent(self, *args, **kwargs) -> None:
        """Do nothing."""

    def play_sound_notification(self, *args, **kwargs) -> None:
        """Do nothing; the dialog plays it when the task completes."""


class _TaskThread(QtCore.QThread):
    """Runs a task's function and keeps its outcome for the GUI thread."""

    def __init__(self, fn: Callable, progress: TaskProgress) -> None:
        super().__init__()
        self._fn = fn
        self._progress = progress
        self.outcome = None  # "finished", "canceled" or "failed"
        self.result = None
        self.error = None

    def run(self) -> None:
        try:
            self.result = self._fn(self._progress)
            self.outcome = "finished"
        except OperationCanceled:
            self.outcome = "canceled"
        except BaseException as error:  # noqa: BLE001 - reported by Task
            self.error = error
            self.outcome = "failed"


class _InputBlocker(QtCore.QObject):
    """Swallow user input to all windows except modal dialogs, i.e.,
    the task's progress dialog and any message box, while tasks run.

    A computation on the GUI thread froze all input; a task keeps that
    guarantee (the user cannot change the data the worker reads, start a
    second task or close a window under it) while windows still repaint.
    The filter covers the time before the progress dialog shows (which
    is delayed, so quick tasks do not flash a dialog) and non-modal
    windows the modal dialog does not block, e.g. linked windows.
    """

    _BLOCKED = frozenset(
        {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseButtonDblClick,
            QtCore.QEvent.Type.Wheel,
            QtCore.QEvent.Type.KeyPress,
            QtCore.QEvent.Type.KeyRelease,
            QtCore.QEvent.Type.ShortcutOverride,
            QtCore.QEvent.Type.Shortcut,
            QtCore.QEvent.Type.ContextMenu,
            QtCore.QEvent.Type.Close,
            QtCore.QEvent.Type.DragEnter,
            QtCore.QEvent.Type.DragMove,
            QtCore.QEvent.Type.Drop,
            QtCore.QEvent.Type.TouchBegin,
            QtCore.QEvent.Type.TouchUpdate,
            QtCore.QEvent.Type.TouchEnd,
            QtCore.QEvent.Type.TabletPress,
            QtCore.QEvent.Type.TabletRelease,
            QtCore.QEvent.Type.NativeGesture,
            QtCore.QEvent.Type.Gesture,
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    def acquire(self) -> None:
        if self._count == 0:
            QtCore.QCoreApplication.instance().installEventFilter(self)
        self._count += 1

    def release(self) -> None:
        self._count -= 1
        if self._count == 0:
            QtCore.QCoreApplication.instance().removeEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() not in self._BLOCKED:
            return False
        # window-level objects (QWindow) forward input to their widgets,
        # which are filtered below
        if not isinstance(obj, QtWidgets.QWidget):
            return False
        if obj.window().isModal():
            return False
        if event.type() in (
            QtCore.QEvent.Type.Close,
            QtCore.QEvent.Type.ShortcutOverride,
        ):
            # a close event must be ignored to keep the window open; an
            # accepted override stops the key from triggering a shortcut
            if event.type() == QtCore.QEvent.Type.Close:
                event.ignore()
            else:
                event.accept()
        return True


_input_blocker = None
# running tasks, referenced such that a running QThread is never
# garbage-collected (which aborts the process)
_running_tasks = []


def _stop_running_tasks() -> None:
    """Cancel all running tasks and wait for their threads, called when
    the application quits."""
    for task in list(_running_tasks):
        task.cancel()
        task.wait()


class Task(QtCore.QObject):
    """A function running on a worker thread behind a cancelable progress
    dialog. Created and started by :func:`run_task`, which describes the
    behavior.

    Attributes
    ----------
    progress : TaskProgress
        Progress tracker passed to the function.
    dialog : TaskProgressDialog
        The progress dialog.
    outcome : {"finished", "canceled", "failed"} or None
        How the task ended; None while it runs.
    """

    def __init__(
        self,
        fn: Callable,
        description: str,
        parent: QtWidgets.QWidget,
        maximum: int,
        on_finished: Callable | None,
        on_failed: Callable | None,
        on_canceled: Callable | None,
        title: str | None,
    ) -> None:
        super().__init__()
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_canceled = on_canceled
        self.outcome = None
        self.progress = TaskProgress(description, maximum)
        self.dialog = TaskProgressDialog(description, maximum, parent, title)
        self.progress.value_changed.connect(self._on_value)
        self.progress.maximum_changed.connect(self._on_maximum)
        self.progress.label_changed.connect(self._on_label)
        self.progress.phase_changed.connect(self._on_phase)
        self.dialog.cancel_requested.connect(self.cancel)
        self._thread = _TaskThread(fn, self.progress)
        self._thread.finished.connect(self._on_thread_finished)

    def start(self) -> None:
        """Block input and start the worker thread."""
        global _input_blocker
        app = QtCore.QCoreApplication.instance()
        if _input_blocker is None:
            _input_blocker = _InputBlocker()
            app.aboutToQuit.connect(_stop_running_tasks)
        _input_blocker.acquire()
        _running_tasks.append(self)
        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation. The function stops at its next progress
        update; its result, if it still completes, is discarded."""
        self.progress.cancel()

    def is_running(self) -> bool:
        """Whether the task has not ended yet, i.e., its callbacks have
        not been called."""
        return self.outcome is None

    def wait(self, msecs: int | None = None) -> bool:
        """Block until the worker thread has stopped. The callbacks run
        later, from the event loop.

        Parameters
        ----------
        msecs : int, optional
            Timeout in ms. None waits indefinitely.

        Returns
        -------
        stopped : bool
            False if the timeout expired first.
        """
        if msecs is None:
            return self._thread.wait()
        return self._thread.wait(msecs)

    def _dialog_alive(self) -> bool:
        return not sip.isdeleted(self.dialog)

    def _forward(self) -> bool:
        """Whether progress updates should reach the dialog."""
        return self._dialog_alive() and not self.progress.canceled

    def _on_value(self, value: int) -> None:
        if self._forward():
            self.dialog.set_value(value)

    def _on_maximum(self, maximum: int) -> None:
        if self._forward():
            self.dialog.setMaximum(maximum)

    def _on_label(self, text: str) -> None:
        if self._forward():
            self.dialog.setLabelText(text)

    def _on_phase(self, description: str) -> None:
        if self._forward():
            self.dialog.zero_progress(description)

    def _on_thread_finished(self) -> None:
        thread = self._thread
        thread.wait()  # run() has returned; make sure the thread is done
        outcome = thread.outcome
        if self.progress.canceled and outcome != "canceled":
            # the user asked to cancel: discard a late result, and only
            # log an error, which the cancellation may have caused
            if outcome == "failed":
                diagnostics.log_message(
                    "Error in a canceled task:\n"
                    + "".join(
                        traceback.format_exception(
                            type(thread.error),
                            thread.error,
                            thread.error.__traceback__,
                        )
                    )
                )
            outcome = "canceled"
        self.outcome = outcome
        result, error = thread.result, thread.error
        thread.result = thread.error = None

        # tear down before calling back, so a callback can start a task
        if self._dialog_alive():
            self.dialog.finish(completed=outcome == "finished")
            self.dialog.deleteLater()
        _input_blocker.release()
        _running_tasks.remove(self)

        if outcome == "finished":
            if self._on_finished is not None:
                self._on_finished(result)
        elif outcome == "failed":
            if self._on_failed is not None:
                self._on_failed(error)
            else:
                sys.excepthook(type(error), error, error.__traceback__)
        elif self._on_canceled is not None:
            self._on_canceled()


def run_task(
    fn: Callable,
    description: str,
    parent: QtWidgets.QWidget,
    on_finished: Callable | None = None,
    *,
    maximum: int = 0,
    on_failed: Callable | None = None,
    on_canceled: Callable | None = None,
    title: str | None = None,
) -> Task:
    """Run ``fn(progress)`` on a worker thread behind a cancelable
    progress dialog.

    The GUI stays responsive (windows repaint) while user input is
    blocked as during a computation on the GUI thread, except for the
    dialog's Cancel button. ``fn`` receives a :class:`TaskProgress`,
    which can be passed to any function that takes a progress dialog or
    a progress callback (``progress.set_value``). Cancel makes the next
    progress update raise ``lib.OperationCanceled``, which ends the task
    as canceled.

    ``fn`` runs on the worker thread, so it must not touch widgets or
    mutate GUI state: read the inputs before, and apply the result in
    ``on_finished``, which runs on the GUI thread.

    Parameters
    ----------
    fn : callable
        Takes the ``TaskProgress`` and returns the result.
    description : str
        Label of the progress dialog.
    parent : QWidget
        Parent of the progress dialog.
    on_finished : callable, optional
        Called with the result of ``fn`` if it completed and was not
        canceled. Default None.
    maximum : int, optional
        Maximum of the progress bar; 0 (default) shows a busy indicator
        until ``fn`` sets a maximum.
    on_failed : callable, optional
        Called with the exception raised by ``fn``. None (default) shows
        it like any uncaught exception (see :func:`install_excepthook`).
    on_canceled : callable, optional
        Called once the task stopped after the user canceled it. Default
        None.
    title : str, optional
        Window title of the progress dialog. Default None.

    Returns
    -------
    task : Task
        The started task.
    """
    task = Task(
        fn,
        description,
        parent,
        maximum,
        on_finished,
        on_failed,
        on_canceled,
        title,
    )
    task.start()
    return task


# type alias for the progress dialogs
ProgressType: TypeAlias = ProgressDialog | MockProgress | TqdmProgress


class CollapsibleHeader(QtWidgets.QAbstractButton):
    """Clickable header of a ``CollapsibleGroupBox``.

    Shows a chevron that turns from right (collapsed) to down
    (expanded), the title and, while collapsed, a muted summary on the
    right. The header is highlighted on hover and shows a focus ring
    when reached with the Tab key. All colors come from the palette, so
    that light and dark themes work alike.

    Parameters
    ----------
    title : str
        Title, shown in bold.
    summary : str, optional
        Short description of the contents, shown on the right while
        collapsed. Default "".
    parent : QWidget or None, optional
        Parent widget. Default None.
    """

    #: Duration of the chevron's turn in ms.
    ANIMATION_MS = 160
    _PADDING = 8  # horizontal padding in pixels
    _CHEVRON = 5  # half the height of the chevron in pixels

    def __init__(
        self,
        title: str,
        summary: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setText(title)
        self.setAccessibleName(title)
        self._summary = summary
        # fraction of the turn, 0 when collapsed and 1 when expanded
        self._progress = 0.0
        self._animation = QtCore.QVariantAnimation(self)
        self._animation.setDuration(self.ANIMATION_MS)
        self._animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._set_progress)
        font = self.font()
        font.setBold(True)
        self.setFont(font)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # a click does not leave a focus ring behind, the Tab key does
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def progress(self) -> float:
        """How far the chevron has turned: 0 collapsed, 1 expanded."""
        return self._progress

    def set_expanded(self, expanded: bool, animate: bool = True) -> None:
        """Turn the chevron to the expanded or collapsed state.

        Parameters
        ----------
        expanded : bool
            The new state.
        animate : bool, optional
            Turn smoothly if the header is visible, otherwise at once.
            Default True.
        """
        end = 1.0 if expanded else 0.0
        self._animation.stop()
        if animate and self.isVisible():
            self._animation.setStartValue(self._progress)
            self._animation.setEndValue(end)
            self._animation.start()
        else:
            self._set_progress(end)

    def _set_progress(self, value: float) -> None:
        self._progress = float(value)
        self.update()

    def _summary_font(self) -> QtGui.QFont:
        font = self.font()
        font.setBold(False)
        return font

    def minimumSizeHint(self) -> QtCore.QSize:
        metrics = self.fontMetrics()
        width = (
            3 * self._PADDING
            + 2 * self._CHEVRON
            + metrics.horizontalAdvance(self.text())
        )
        return QtCore.QSize(width, max(metrics.height() + 12, 28))

    def sizeHint(self) -> QtCore.QSize:
        hint = self.minimumSizeHint()
        if self._summary:
            metrics = QtGui.QFontMetrics(self._summary_font())
            hint.setWidth(
                hint.width()
                + 2 * self._PADDING
                + metrics.horizontalAdvance(self._summary)
            )
        return hint

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        palette = self.palette()
        highlight = palette.color(QtGui.QPalette.ColorRole.Highlight)
        text_color = palette.color(QtGui.QPalette.ColorRole.WindowText)
        muted = palette.color(QtGui.QPalette.ColorRole.PlaceholderText)
        progress = self._progress
        rect = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # background: a tint of the accent color on hover and press
        alpha = 0
        if self.isDown():
            alpha = 55
        elif self.underMouse():
            alpha = 28
        if alpha:
            tint = QtGui.QColor(highlight)
            tint.setAlpha(alpha)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(tint)
            painter.drawRoundedRect(rect, 5, 5)
        if self.hasFocus():
            ring = QtGui.QColor(highlight)
            ring.setAlpha(180)
            painter.setPen(QtGui.QPen(ring, 1.5))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 5, 5)

        # divider under the header, fading in as the contents open
        if progress > 0:
            divider = QtGui.QColor(muted)
            divider.setAlphaF(0.35 * progress * divider.alphaF())
            painter.setPen(QtGui.QPen(divider, 1))
            y = rect.bottom() - 0.5
            painter.drawLine(
                QtCore.QPointF(rect.left() + self._PADDING, y),
                QtCore.QPointF(rect.right() - self._PADDING, y),
            )

        # chevron, turning from right to down and taking on the accent
        # color as it does
        c = self._CHEVRON
        center = QtCore.QPointF(
            rect.left() + self._PADDING + c, rect.center().y()
        )
        # (the muted color is often translucent, so alpha is blended too)
        chevron_color = QtGui.QColor.fromRgbF(
            *(
                m + (h - m) * progress
                for m, h in zip(muted.getRgbF(), highlight.getRgbF())
            )
        )
        pen = QtGui.QPen(chevron_color, 1.8)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.save()
        painter.translate(center)
        painter.rotate(90 * progress)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPolyline(
            QtGui.QPolygonF(
                [
                    QtCore.QPointF(-c / 2, -c),
                    QtCore.QPointF(c / 2, 0),
                    QtCore.QPointF(-c / 2, c),
                ]
            )
        )
        painter.restore()

        # title
        text_left = rect.left() + 2 * self._PADDING + 2 * c
        text_rect = QtCore.QRectF(
            text_left, rect.top(), rect.right() - text_left, rect.height()
        )
        painter.setPen(text_color)
        painter.setFont(self.font())
        align = (
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        painter.drawText(text_rect, align, self.text())

        # summary, fading out as the contents open
        if self._summary and progress < 1:
            title_width = self.fontMetrics().horizontalAdvance(self.text())
            summary_rect = text_rect.adjusted(
                title_width + 2 * self._PADDING, 0, -self._PADDING, 0
            )
            font = self._summary_font()
            summary = QtGui.QFontMetrics(font).elidedText(
                self._summary,
                QtCore.Qt.TextElideMode.ElideRight,
                int(summary_rect.width()),
            )
            color = QtGui.QColor(muted)
            color.setAlphaF(color.alphaF() * (1 - progress))
            painter.setPen(color)
            painter.setFont(font)
            painter.drawText(
                summary_rect,
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                summary,
            )
        painter.end()


class CollapsibleGroupBox(QtWidgets.QGroupBox):
    """QGroupBox whose contents are shown or hidden by clicking its
    header, which carries the title, see ``CollapsibleHeader``.

    Lay out the contents on ``content``, e.g.,
    ``QtWidgets.QGridLayout(box.content)``.

    Parameters
    ----------
    title : str
        Title shown in the header.
    expanded : bool, optional
        Whether the contents are shown initially. Default True.
    summary : str, optional
        Short description of the contents, shown in the header while
        collapsed. Default "".
    parent : QWidget or None, optional
        Parent widget. Default None.

    Attributes
    ----------
    content : QWidget
        Holds the contents.
    expandedChanged : pyqtSignal
        Emitted with the new state when the contents are shown or
        hidden.
    toggle_button : CollapsibleHeader
        The header.
    """

    expandedChanged = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        summary: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.toggle_button = CollapsibleHeader(title, summary)
        self.toggle_button.setToolTip(f"Show or hide the {title} settings.")
        layout.addWidget(self.toggle_button)
        self.content = QtWidgets.QWidget()
        layout.addWidget(self.content)
        self._expanded = True
        self.toggle_button.clicked.connect(
            lambda: self.setExpanded(not self._expanded)
        )
        self.setExpanded(expanded)
        self.toggle_button.set_expanded(expanded, animate=False)

    def isExpanded(self) -> bool:
        """Whether the contents are shown."""
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:
        """Show or hide the contents."""
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self.toggle_button.set_expanded(expanded)
        self.content.setVisible(expanded)
        self.expandedChanged.emit(expanded)


class ScrollableGroupBox(QtWidgets.QGroupBox):
    """QGroupBox with QScrollArea as the top widget that enables
    scrolling."""

    def __init__(self, title, parent=None, layout="grid"):
        super().__init__(title, parent=parent)

        # Create a layout for the content of the group box
        if layout == "grid":
            self.content_layout = QtWidgets.QGridLayout(self)
        elif layout == "form":
            self.content_layout = QtWidgets.QFormLayout(self)
        self.content_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(10)
        self.content_layout.setContentsMargins(10, 10, 10, 10)

        # Create a scroll area and set its content to the content layout
        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QtWidgets.QWidget(self))
        self.scroll_area.widget().setLayout(self.content_layout)

        # Set the layout of the group box to the scroll area
        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().addWidget(self.scroll_area, 0, 0, 1, 2)

    def add_widget(self, widget, row, column, height=1, width=1):
        """Add a widget to the grid layout inside the scroll area.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            The widget to add.
        row, column : int
            Where to place it in the grid.
        height, width : int, optional
            How many rows and columns it spans. Default 1.
        """
        self.content_layout.addWidget(widget, row, column, height, width)

    def remove_widget(self, widget):
        """Remove a widget from the grid layout inside the scroll area.

        Parameters
        ----------
        widget : QtWidgets.QWidget
            The widget to remove.
        """
        self.content_layout.removeWidget(widget)

    def remove_all_widgets(self, keep_labels=False):
        """Remove all widgets from the grid layout.

        Parameters
        ----------
        keep_labels : bool, optional
            If True, the QLabels are kept. Default False.
        """
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if keep_labels and isinstance(widget, QtWidgets.QLabel):
                continue
            widget.setParent(None)
            widget.deleteLater()


class LogDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox with logarithmic step size."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, factor: float = 1.2
    ) -> None:
        super().__init__(parent)
        self._factor = factor  # multiply/divide by this on each step

    def stepBy(self, steps: int) -> None:
        """Step the value multiplicatively, so the arrows move it by a factor
        rather than by a fixed amount.

        Parameters
        ----------
        steps : int
            Number of steps; negative steps divide instead of multiply.
        """
        if steps > 0:
            if self.value() <= 10 ** (-self.decimals()):
                self.setValue(2 * 10 ** (-self.decimals()))
            else:
                self.setValue(self.value() * (self._factor**steps))
        elif steps < 0:
            self.setValue(self.value() / (self._factor ** abs(steps)))


class RangeSlider(QtWidgets.QWidget):
    """Horizontal slider with two handles, spanning a value range.

    Qt has no two-handle slider, so this is a minimal one: a groove with
    a low and a high handle, painted to match the stylesheet of the plain
    ``QSlider``s used elsewhere in Picasso. Values are floats, so that the
    slider is not limited to the integer positions of ``QSlider``.

    ``valuesChanged`` is emitted whenever the pair changes, both on user
    interaction and on a programmatic ``setValues`` / ``setRange`` (like
    ``QSlider.setValue``); block the signals to update silently.

    The track is linear by default; ``setLogScale`` switches it to a
    logarithmic one, which is what strongly skewed quantities (rendered
    localization densities, for instance) need to be adjustable at all.
    """

    valuesChanged = QtCore.pyqtSignal(float, float)

    HANDLE_WIDTH = 10
    HANDLE_HEIGHT = 12
    GROOVE_HEIGHT = 4

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0.0
        self._maximum = 1.0
        self._low = 0.0
        self._high = 1.0
        # Smallest allowed distance between the handles. 0 lets them meet;
        # consumers that quantize the values (e.g. to integers) set it to
        # the quantum so that low and high cannot collapse onto each other.
        self._min_gap = 0.0
        # Number of decades the track spans when logarithmic; 0 is linear.
        # See ``setLogScale``.
        self._log_decades = 0.0
        self._pressed_handle = None  # None, "low" or "high"
        self._last_handle = "low"  # the one the arrow keys move
        self._value_labels = ("Min", "Max")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._update_tooltip()

    # -- API ---------------------------------------------------------

    def setRange(self, minimum: float, maximum: float) -> None:
        """Set the ends of the track, re-clamping the current values.

        Parameters
        ----------
        minimum, maximum : float
            The new ends; ``maximum`` is raised to ``minimum`` if it is below
            it.
        """
        minimum = float(minimum)
        maximum = max(float(maximum), minimum)
        self._minimum = minimum
        self._maximum = maximum
        self.setValues(self._low, self._high)
        self.update()

    def range(self) -> tuple[float, float]:
        """The ends of the track, as ``(minimum, maximum)``."""
        return self._minimum, self._maximum

    def minimum(self) -> float:
        """The lower end of the track."""
        return self._minimum

    def maximum(self) -> float:
        """The upper end of the track."""
        return self._maximum

    def setValues(
        self, low: float, high: float, moved: str | None = None
    ) -> bool:
        """Set both handles, clamped into the track and kept ``_min_gap``
        apart.

        Parameters
        ----------
        low, high : float
            The requested handle positions.
        moved : str, optional
            Names the handle the user is dragging (``"low"`` or ``"high"``),
            which is the one that gives way when the two would cross. Default
            None.

        Returns
        -------
        changed : bool
            Whether anything changed; ``valuesChanged`` is emitted if so.
        """
        low = min(max(float(low), self._minimum), self._maximum)
        high = min(max(float(high), self._minimum), self._maximum)
        gap = min(self._min_gap, self._maximum - self._minimum)
        if high - low < gap:
            if moved == "low":
                low = high - gap
                if low < self._minimum:
                    low = self._minimum
                    high = low + gap
            else:
                high = low + gap
                if high > self._maximum:
                    high = self._maximum
                    low = high - gap
        if low == self._low and high == self._high:
            return False
        self._low = low
        self._high = high
        self._update_tooltip()
        self.update()
        self.valuesChanged.emit(low, high)
        return True

    def values(self) -> tuple[float, float]:
        """The handle positions, as ``(low, high)``."""
        return self._low, self._high

    def setMinimumGap(self, gap: float) -> None:
        """Set how far apart the two handles must stay.

        Parameters
        ----------
        gap : float
            Minimum distance, in track units; negative values are clipped to
            0.
        """
        self._min_gap = max(0.0, float(gap))
        self.setValues(self._low, self._high)

    def setValueLabels(self, low_label: str, high_label: str) -> None:
        """Name the two handles, for the tooltip.

        Parameters
        ----------
        low_label, high_label : str
            Names of the lower and upper handle.
        """
        self._value_labels = (low_label, high_label)
        self._update_tooltip()

    def setLogScale(self, log: bool, decades: float = 3.0) -> None:
        """Make the track logarithmic (or linear again).

        The mapping is a symmetric-log one, ``log10(1 + f / c)`` of the
        linear fraction ``f`` with ``c = 10 ** -decades``, normalized to
        span the widget. It is scale-free (it only depends on where a
        value sits within the track) and, unlike a plain logarithm, it is
        defined at the lower end of the track, so a track starting at
        zero works.

        Parameters
        ----------
        log : bool
            True for a logarithmic track, False for a linear one.
        decades : float, optional
            Number of decades below the top of the track that stay
            resolvable; everything below is squeezed into the first
            pixels. Default is 3.
        """
        self._log_decades = max(0.0, float(decades)) if log else 0.0
        self.update()

    def logScale(self) -> bool:
        """Whether the track is logarithmic."""
        return bool(self._log_decades)

    # -- Painting ----------------------------------------------------

    def sizeHint(self) -> QtCore.QSize:
        """The size the slider would like to have."""
        return QtCore.QSize(150, self.HANDLE_HEIGHT + 3)

    def minimumSizeHint(self) -> QtCore.QSize:
        """The smallest size at which both handles still fit."""
        return QtCore.QSize(4 * self.HANDLE_WIDTH, self.HANDLE_HEIGHT + 3)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        """Draw the groove, the selected span and the two handles.

        Parameters
        ----------
        _event : QtGui.QPaintEvent
            The Qt paint event; unused.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        # from the palette, so that the slider follows the theme
        palette = self.palette()
        if self.isEnabled():
            groove_color = palette.color(QtGui.QPalette.ColorRole.Mid)
            handle_color = palette.color(QtGui.QPalette.ColorRole.Highlight)
        else:
            groove_color = palette.color(
                QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.Midlight,
            )
            handle_color = palette.color(
                QtGui.QPalette.ColorGroup.Disabled,
                QtGui.QPalette.ColorRole.Mid,
            )
        mid_y = self.height() / 2
        radius = self.GROOVE_HEIGHT / 2
        groove = QtCore.QRectF(
            0.0,
            mid_y - radius,
            float(self.width()),
            float(self.GROOVE_HEIGHT),
        )
        painter.setBrush(groove_color)
        painter.drawRoundedRect(groove, radius, radius)
        # the selected span, so that the covered part of the range reads
        # at a glance
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)
        painter.setBrush(handle_color)
        painter.drawRoundedRect(
            QtCore.QRectF(
                x_low,
                mid_y - radius,
                max(0.0, x_high - x_low),
                groove.height(),
            ),
            radius,
            radius,
        )
        for x in (x_low, x_high):
            painter.drawRoundedRect(
                QtCore.QRectF(
                    x - self.HANDLE_WIDTH / 2,
                    mid_y - self.HANDLE_HEIGHT / 2,
                    float(self.HANDLE_WIDTH),
                    float(self.HANDLE_HEIGHT),
                ),
                3.0,
                3.0,
            )
        painter.end()

    # -- Interaction -------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Grab the handle nearest the click and move it there.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            The Qt mouse event; anything but a left click is passed on.
        """
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        self._pressed_handle = self._handle_at(x)
        self._last_handle = self._pressed_handle
        self._move_handle(self._pressed_handle, x)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Drag the grabbed handle.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            The Qt mouse event; passed on when no handle is grabbed.
        """
        if self._pressed_handle is None:
            super().mouseMoveEvent(event)
            return
        self._move_handle(self._pressed_handle, event.position().x())
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Release the grabbed handle.

        Parameters
        ----------
        event : QtGui.QMouseEvent
            The Qt mouse event; passed on when no handle is grabbed.
        """
        if self._pressed_handle is None:
            super().mouseReleaseEvent(event)
            return
        self._pressed_handle = None
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Move the last-touched handle with the arrow keys.

        Parameters
        ----------
        event : QtGui.QKeyEvent
            The Qt key event; anything but the arrow keys is passed on.
        """
        key = event.key()
        if key in (
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Up,
        ):
            direction = (
                -1
                if key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Down)
                else 1
            )
            if self._last_handle == "high":
                self.setValues(
                    self._low,
                    self._stepped(self._high, direction),
                    moved="high",
                )
            else:
                self.setValues(
                    self._stepped(self._low, direction),
                    self._high,
                    moved="low",
                )
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Ignore the wheel, so that scrolling passes through to the parent.

        The slider sits under a scrollable movie view; scrolling over it
        should not silently change the contrast.

        Parameters
        ----------
        event : QtGui.QWheelEvent
            The Qt wheel event.
        """
        event.ignore()

    # -- Helpers -----------------------------------------------------

    def _step(self) -> float:
        span = self._maximum - self._minimum
        return max(self._min_gap, span / 100.0) if span else 0.0

    def _stepped(self, value: float, direction: int) -> float:
        """The value one arrow-key step away from ``value``.

        On a logarithmic track the step is a constant fraction of the
        widget's width, so that the handle moves by the same distance
        everywhere; on a linear one it is a constant value.

        Parameters
        ----------
        value : float
            The handle's current value.
        direction : int
            +1 to step up, -1 to step down.

        Returns
        -------
        float
            The stepped value.
        """
        if self._log_decades:
            fraction = self._value_to_fraction(value) + direction / 100.0
            return self._fraction_to_value(fraction)
        return value + direction * self._step()

    def _value_to_fraction(self, value: float) -> float:
        """Where ``value`` sits along the track, as a fraction in [0, 1],
        after the log mapping (if any)."""
        span = self._maximum - self._minimum
        fraction = (value - self._minimum) / span if span else 0.0
        fraction = min(max(fraction, 0.0), 1.0)
        if self._log_decades:
            c = 10.0**-self._log_decades
            fraction = math.log10(1.0 + fraction / c) / math.log10(1.0 + 1 / c)
        return fraction

    def _fraction_to_value(self, fraction: float) -> float:
        """Inverse of ``_value_to_fraction``."""
        fraction = min(max(fraction, 0.0), 1.0)
        if self._log_decades:
            c = 10.0**-self._log_decades
            fraction = c * ((1.0 + 1 / c) ** fraction - 1.0)
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _usable_width(self) -> float:
        return max(1.0, self.width() - self.HANDLE_WIDTH)

    def _value_to_x(self, value: float) -> float:
        """Pixel position of a handle's center, inset by half a handle so
        that the handles stay inside the widget at both ends."""
        fraction = self._value_to_fraction(value)
        return self.HANDLE_WIDTH / 2 + fraction * self._usable_width()

    def _x_to_value(self, x: float) -> float:
        fraction = (x - self.HANDLE_WIDTH / 2) / self._usable_width()
        return self._fraction_to_value(fraction)

    def _handle_at(self, x: float) -> str:
        """The handle a click at ``x`` grabs: the one under the cursor, or
        else the one nearer to it (so clicking the bare groove moves the
        closer handle there)."""
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)
        if x < x_low:
            return "low"
        if x > x_high:
            return "high"
        return "low" if (x - x_low) <= (x_high - x) else "high"

    def _move_handle(self, handle: str, x: float) -> None:
        value = self._x_to_value(x)
        if handle == "low":
            self.setValues(value, self._high, moved="low")
        else:
            self.setValues(self._low, value, moved="high")

    def _update_tooltip(self) -> None:
        low_label, high_label = self._value_labels
        low = self._format_value(self._low)
        high = self._format_value(self._high)
        self.setToolTip(f"{low_label}: {low}, {high_label}: {high}")

    def _format_value(self, value: float) -> str:
        """Format a handle value for the tooltip, keeping the decimals
        that a small track (e.g. localization densities below one) needs
        and dropping them for a large one (e.g. camera counts)."""
        span = self._maximum - self._minimum
        if span >= 100:
            return f"{value:,.0f}"
        if span >= 10:
            return f"{value:,.1f}"
        return f"{value:,.4g}"


class DensityContrastSlider(RangeSlider):
    """Log-scale two-handle slider bound to a pair of density spin boxes.

    Used by the Display Settings dialogs of Picasso: Render and of the
    rotation window to drag the minimum and maximum rendered density
    instead of typing them. The spin boxes stay the single source of
    truth: the slider writes into them (so their usual re-render is
    triggered) and is moved back onto their values by ``sync``.

    The track runs from zero to the brightest pixel of the currently
    rendered image, and is logarithmic, because rendered densities are
    strongly skewed - on a linear track the useful part of the range
    would sit within a couple of pixels of the left end.

    Attributes
    ----------
    minimum_box, maximum_box : QDoubleSpinBox
        The spin boxes holding the minimum and maximum density.
    """

    def __init__(
        self,
        minimum_box: QtWidgets.QDoubleSpinBox,
        maximum_box: QtWidgets.QDoubleSpinBox,
        image: Callable[[], object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Bind the slider to the minimum and maximum spin boxes.

        Parameters
        ----------
        minimum_box, maximum_box : QDoubleSpinBox
            The spin boxes the slider is bound to.
        image : callable, optional
            Returns the raw (grayscale) rendered image, whose maximum
            gives the top of the track, or None if nothing is rendered
            yet. Called on every ``sync``. Default is None, which sizes
            the track from the spin box values alone.
        parent : QWidget, optional
            The parent widget. Default is None.
        """
        super().__init__(parent)
        self.minimum_box = minimum_box
        self.maximum_box = maximum_box
        self._image = image
        # guards the two-way binding: while the slider writes into the
        # spin boxes, their signals must not move the slider back
        self._updating = False
        # the last image the track was sized from, and its maximum, so
        # that dragging does not scan the image on every mouse move
        self._last_image = None
        self._last_upper = 0.0
        self.setLogScale(True)
        self.setValueLabels("Min. density", "Max. density")
        # the spin boxes quantize the values, so the handles must not be
        # able to collapse onto each other
        self.setMinimumGap(10 ** -maximum_box.decimals())
        self.setMaximumHeight(15)
        self.valuesChanged.connect(self._on_values_changed)
        minimum_box.valueChanged.connect(self.sync)
        maximum_box.valueChanged.connect(self.sync)
        self.sync()

    def sync(self, *args) -> None:
        """Re-derive the track from the rendered image and move the
        handles onto the spin box values, without emitting
        ``valuesChanged``.

        Parameters
        ----------
        *args
            Ignored; lets the method be connected to value signals.
        """
        if self._updating:  # the slider itself is driving the change
            return
        low = self.minimum_box.value()
        high = self.maximum_box.value()
        # never cut off a value the user (or autoscaling) has set
        upper = max(self._image_maximum(), low, high)
        if upper <= 0:
            upper = 1.0
        self.blockSignals(True)
        try:
            self.setRange(0.0, upper)
            self.setValues(low, high)
        finally:
            self.blockSignals(False)
        self.update()

    def _image_maximum(self) -> float:
        """The brightest pixel of the rendered image, or 0 if there is
        none. Cached, as ``sync`` runs on every drag step."""
        if self._image is None:
            return 0.0
        image = self._image()
        if image is None or not getattr(image, "size", 0):
            return 0.0
        if image is not self._last_image:
            self._last_image = image
            self._last_upper = float(image.max())
        return self._last_upper

    def _on_values_changed(self, low: float, high: float) -> None:
        """Write a dragged handle into its spin box, which re-renders.

        Parameters
        ----------
        low, high : float
            The new handle positions.
        """
        self._updating = True
        try:
            # only the handle that actually moved, so that a drag
            # triggers a single re-render
            if low != self.minimum_box.value():
                self.minimum_box.setValue(low)
            if high != self.maximum_box.value():
                self.maximum_box.setValue(high)
        finally:
            self._updating = False


class GenericPlotWindow(QtWidgets.QTabWidget):
    """Interface for displaying matplotlib plots in a separate
    window.

    The plots take the shared appearance of Picasso's chart windows
    (``picasso.gui.plot_style``) when drawn inside ``plot_context``.
    The toolbar opens the plot settings; when they change, ``redraw``
    is called if set, otherwise the drawn figure is restyled (keeping
    the colors of the data).

    Attributes
    ----------
    figure : plt.Figure
        The figure to draw on.
    canvas : FigureCanvas
        Canvas showing ``figure``.
    toolbar : NavigationToolbar2QT
        Toolbar of the canvas; callers may add widgets.
    plot_style : picasso.gui.plot_style.PlotStyle
        The current appearance.
    redraw : Callable[[], None] or None
        Draws the plot again (inside ``plot_context``); set by the
        caller so that style changes also recolor the data.
    """

    def __init__(self, window_title, app_name):
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvas,
            NavigationToolbar2QT,
        )

        # imported here: picasso.gui.plot_style imports picasso.lib
        from picasso.gui import plot_style, theme

        super().__init__()
        self.setWindowTitle(window_title)
        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", f"{app_name}.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(1000, 500)
        self.figure = plt.Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)
        vbox.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        vbox.addWidget(self.toolbar)
        self.plot_style = plot_style.current()
        self.plot_style.style_figure(self.figure)
        self.redraw = None
        self.toolbar.addSeparator()
        settings_action = self.toolbar.addAction("Plot settings")
        settings_action.setIcon(theme.icon("plot-settings"))
        settings_action.setToolTip("Appearance of all chart windows")
        settings_action.triggered.connect(lambda: plot_style.show_dialog())
        plot_style.hub().changed.connect(self._on_style_changed)

    @contextlib.contextmanager
    def plot_context(self):
        """Context in which to draw on ``figure`` with the current plot
        style, see ``PlotStyle.context``."""
        self.plot_style.style_figure(self.figure)
        with self.plot_style.context():
            yield
        # ticks created at draw time, after the context, would take the
        # global defaults (e.g., on log axes); store the style on the axes
        self.plot_style.apply(self.figure)

    def _on_style_changed(self, style) -> None:
        if sip.isdeleted(self):
            return
        self.plot_style = style
        if self.redraw is not None:
            self.redraw()
        else:
            style.apply(self.figure)
        self.canvas.draw_idle()


class RemoveColumnsDialog(Dialog):
    """Allow the user to select columns to be removed from the locs
    DataFrame."""

    def __init__(
        self, window: QtWidgets.QMainWindow, columns: list[str]
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Remove columns")
        self.setModal(True)
        vbox = QtWidgets.QVBoxLayout(self)
        self.setLayout(vbox)
        self.checks = {}
        for column in columns:
            check = QtWidgets.QCheckBox(column)
            check.setChecked(False)
            if column in REQUIRED_COLUMNS:
                check.setEnabled(False)
            vbox.addWidget(check)
            self.checks[column] = check
        # OK and Cancel buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    @staticmethod
    def getParams(
        parent: QtWidgets.QMainWindow, columns: list[str]
    ) -> tuple[list[str], bool]:
        """Open the dialog and return the columns to be removed.

        Parameters
        ----------
        parent : QMainWindow
            Instance of the main window.
        columns : list of str
            List of column names in the locs DataFrame.

        Returns
        -------
        to_remove : list of str
            List of column names to be removed.
        accepted : bool
            True if the user clicked OK, False if the user clicked
            Cancel.
        """
        dialog = RemoveColumnsDialog(parent, columns)
        result = dialog.exec()
        to_remove = []
        for col in columns:
            if dialog.checks[col].isChecked():
                to_remove.append(col)
        return to_remove, result == QtWidgets.QDialog.DialogCode.Accepted


class HelpButton(QtWidgets.QToolButton):
    """A reusable ? button that opens a URL."""

    def __init__(
        self, url: str, parent=None, size: int | tuple[int, int] = 22
    ) -> None:
        super().__init__(parent)
        self.help_url = url
        self.setText("?")
        if isinstance(size, int):
            size = (size, size)
        self.setFixedSize(*size)
        self.setToolTip("Open documentation")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # filled with the accent color, so that the button stands out;
        # inverted on hover. Shows the help icon, or "?" without it.
        # imported here: picasso.gui.theme imports picasso.lib
        from picasso.gui import theme

        Role = QtGui.QPalette.ColorRole
        self._icons = (
            theme.icon("help", role=Role.HighlightedText),
            theme.icon("help", role=Role.Highlight),  # on hover
        )
        if not self._icons[0].isNull():
            self.setIcon(self._icons[0])
            side = max(8, min(size) - 8)
            self.setIconSize(QtCore.QSize(side, side))
        radius = min(size) // 2
        self.setStyleSheet(
            f"""
            QToolButton {{
                border: 1px solid palette(highlight);
                border-radius: {radius}px;
                padding: 0px;
                font-weight: bold;
                font-size: 12px;
                color: palette(highlighted-text);
                background: palette(highlight);
            }}
            QToolButton:hover {{
                color: palette(highlight);
                background: palette(highlighted-text);
            }}
        """
        )
        self.clicked.connect(self._open_docs)

    def enterEvent(self, event) -> None:
        if not self._icons[1].isNull():
            self.setIcon(self._icons[1])
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if not self._icons[0].isNull():
            self.setIcon(self._icons[0])
        super().leaveEvent(event)

    def _open_docs(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.help_url))


#: Emitter of the excepthook installed by ``install_excepthook``, kept
#: alive here because the hook outlives the call that installed it.
_error_signaler = None


class TripleClick:
    """Recognize a triple click of the left mouse button.

    Qt reports a double click but no triple click; a widget calls
    ``double_clicked`` from its ``mouseDoubleClickEvent`` and
    ``is_third`` from its ``mousePressEvent``, which is true for the
    press that follows the double click within the platform's
    double-click interval and drag distance.
    """

    def __init__(self) -> None:
        self._double: tuple[float, QtCore.QPoint] | None = None

    def double_clicked(self, event: QtGui.QMouseEvent) -> None:
        """Remember a double click so the next press may complete a
        triple click."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._double = (time.monotonic(), QtCore.QPoint(event.pos()))

    def is_third(self, event: QtGui.QMouseEvent) -> bool:
        """Return whether ``event`` (a mouse press) is the third click
        of a triple click; the remembered double click is consumed."""
        if self._double is None:
            return False
        started, pos = self._double
        self._double = None
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        app = QtWidgets.QApplication.instance()
        interval = (app.doubleClickInterval() if app else 400) / 1000.0
        distance = app.startDragDistance() if app else 10
        return (
            time.monotonic() - started <= interval
            and (event.pos() - pos).manhattanLength() <= distance
        )


def cancel_dialogs():
    """Closes all open dialogs (``ProgressDialog`` and ``StatusDialog``)
    in the GUI.

    Called from the excepthook while an error is being reported, so a
    dialog that fails to close is logged and skipped: it must not stop
    the remaining dialogs from closing, nor the error from reaching the
    user.
    """
    dialogs = [_ for _ in _dialogs]
    for dialog in dialogs:
        try:
            if isinstance(dialog, TaskProgressDialog):
                # stop the worker; the task closes its dialog
                dialog.request_cancel()
            elif isinstance(dialog, ProgressDialog):
                dialog.cancel()
            else:
                dialog.close()
        except Exception:  # noqa: BLE001 - cleanup must not raise
            diagnostics.log_message(
                f"Failed to close {type(dialog).__name__}:\n"
                f"{traceback.format_exc()}"
            )
            if dialog in _dialogs:
                _dialogs.remove(dialog)
    QtCore.QCoreApplication.instance().processEvents()  # just in case...


def install_excepthook(window=None) -> None:
    """Install hooks that show uncaught exceptions in a QMessageBox and
    write them to the Picasso log (``~/.picasso/logs/picasso.log``).

    Covers the main thread, worker threads (``threading.excepthook``) and
    unraisable exceptions - see ``picasso.diagnostics``. Safe to call from
    QThread workers because the error signal is queued to the main thread
    by Qt's event loop.

    Call it again once the main window exists to parent the box on it;
    the previous hooks are replaced, not chained.

    Parameters
    ----------
    window : QtWidgets.QWidget, optional
        Parent of the message box. None (default) shows a parentless box,
        which is what ``picasso.gui.app.run_gui`` uses to report failures
        that happen while the main window is still being built.
    """

    # no-op unless the GUI runs without a console (one-click builds),
    # where sys.stdout/sys.stderr are None
    diagnostics.ensure_std_streams()

    class _ErrorSignaler(QtCore.QObject):
        error = QtCore.pyqtSignal(str)

    signaler = _ErrorSignaler()
    showing = []  # non-empty while an error box is open

    def _show_error(message: str) -> None:
        if showing:  # do not stack one box per failed worker
            return
        showing.append(True)
        try:
            try:
                cancel_dialogs()
            except Exception:  # noqa: BLE001 - report the original error
                diagnostics.log_message(traceback.format_exc())
            lines = message.strip().splitlines()
            summary = lines[-1] if lines else "An unknown error occurred."
            box = QtWidgets.QMessageBox(window)
            box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            box.setWindowTitle("An error occurred")
            box.setText(summary)
            box.setInformativeText(
                f"The full traceback was written to\n{diagnostics.log_path()}"
            )
            box.setDetailedText(message)
            box.exec()
        finally:
            showing.clear()

    signaler.error.connect(_show_error)
    # keep the signaler (and hence the connection) alive; module level
    # because ``window`` may not exist yet
    global _error_signaler
    _error_signaler = signaler

    diagnostics.install_excepthooks(report=signaler.error.emit)


def adjust_widget_size(
    widget: QtWidgets.QWidget,
    size_hint: QtCore.QSize,
    width_offset: int = 0,
    height_offset: int = 0,
    max_height: int | None = None,
) -> None:
    """Adjust the size of a QWidget based on its size hint. The user
    can specify the offsets to be added to the width and height of the
    size hint. The user can also specify whether to limit the width
    and height to the screen size.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        The widget to be adjusted.
    size_hint : QtCore.QSize
        The size hint of the widget. Can be obtained with
        widget.sizeHint().
    width_offset : int, optional
        The offset to be added to the width of the size hint. Default is
        0.
    height_offset : int, optional
        The offset to be added to the height of the size hint. Default
        is 0.
    max_height : int or None, optional
        Absolute cap (in pixels) on the resulting height, applied on top
        of the screen-size-based cap below. Use this so a widget with a
        scrollable area does not grow to fill very tall/large screens.
        Default is None, i.e., no additional cap.
    """
    intended_width = size_hint.width() + width_offset
    intended_height = size_hint.height() + height_offset
    # a vertical scroll bar that is not overlaid (as it is on macOS'
    # native style) takes width from the contents
    style = widget.style()
    if widget.findChild(QtWidgets.QScrollArea) is not None and not (
        style.styleHint(QtWidgets.QStyle.StyleHint.SH_ScrollBar_Transient)
    ):
        intended_width += style.pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
        )
    # adjust to the screen size if necessary
    screen = QtWidgets.QApplication.primaryScreen()
    screen_height = 1000 if screen is None else screen.size().height()
    screen_width = 1000 if screen is None else screen.size().width()
    intended_width = min(intended_width, screen_width - 200)
    intended_height = min(intended_height, screen_height - 200)
    if max_height is not None:
        intended_height = min(intended_height, max_height)
    widget.resize(intended_width, intended_height)


def get_save_filename_ext_dialog(
    parent: QtWidgets.QWidget,
    caption: str = "",
    directory: str = "",
    filter: str = "",
    check_ext: str | list[str] = "",
) -> tuple[str, str]:
    """Custom getSaveFileName dialog that can check for the existence of
    files with other extensions (for example, if the user tries to save
    a .yaml file with the same name as an existing .hdf5 file, it will
    ask if the user wants to overwrite the .hdf5 file). The output is
    the same as for QtWidgets.QFileDialog.getSaveFileName.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog.
    caption : str, optional
        Dialog caption. Default is "".
    directory : str, optional
        Initial directory. Default is "".
    filter : str, optional
        File filter, e.g., "YAML files (*.yaml);;HDF5 files (*.hdf5)".
        Default is "".
    check_ext : str or list of str, optional
        Other extension(s) to be checked if they're available. Does not
        have to be a strict ".ext" format, can also include a suffix to
        the path, e.g., "_1.hdf5". If "", extensions are not checked,
        giving the standard getSaveFileName dialog behavior. Default is
        "".

    Returns
    -------
    selected_path : str
        Selected file path.
    selected_filter : str
        Selected file filter.
    """
    # first run the standard dialog to get the initial path and filter
    selected_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
        parent, caption, directory, filter
    )
    # check for the existence of files with other extensions and ask the
    # user if they want to overwrite them
    if selected_path and check_ext:
        paths_available = is_path_available(
            selected_path, check_ext=check_ext, parent=parent
        )
        if not all(paths_available):
            return "", ""
    # if the user selected a .yml file, change the extension to .yaml
    # for consistency
    if selected_path.endswith(".yml"):
        selected_path = selected_path[:-4] + ".yaml"
    return selected_path, selected_filter
