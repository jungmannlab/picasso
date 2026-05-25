"""
picasso.lib
~~~~~~~~~~~

Handy functions and classes.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import collections
import colorsys
import os
import sys
import time
import traceback
import warnings
from copy import deepcopy
from typing import Any, TypeAlias, Literal
from collections.abc import Callable
from asyncio import Future

import yaml
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields, drop_fields
from scipy import stats, optimize
from PyQt6 import QtCore, QtWidgets, QtGui
from playsound3 import playsound
from tqdm import tqdm

from picasso import io

# A global variable where we store all open progress and status dialogs.
# In case of an exception, we close them all,
# so that the GUI remains responsive.
_dialogs = []

# Min. time to use sound notification when ProcessDialog or
# StatusDialog is finished
SOUND_NOTIFICATION_DURATION = 60  # seconds

# Columns that are required for Picasso
REQUIRED_COLUMNS = ["frame", "x", "y", "z", "lpx", "lpy", "lpz"]

# Type alias
IntArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer[Any]]]
IntArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer[Any]]]
IntArray3D: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.integer[Any]]
]
FloatArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
FloatArray2D: TypeAlias = np.ndarray[
    tuple[int, int], np.dtype[np.floating[Any]]
]
FloatArray3D: TypeAlias = np.ndarray[
    tuple[int, int, int], np.dtype[np.floating[Any]]
]
SeriesOrFloatArray1D: TypeAlias = pd.Series | FloatArray1D
SeriesOrIntArray1D: TypeAlias = pd.Series | IntArray1D
BoolArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
BoolArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
Array3x3: TypeAlias = np.ndarray[
    tuple[Literal[3], Literal[3]], np.dtype[np.floating[Any]]
]


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
        (unless it's called "OK")."""
        super().showEvent(event)
        for button in self.findChildren(QtWidgets.QPushButton):
            if button.text() in self._focus_buttons:
                continue
            button.setDefault(False)
            button.setAutoDefault(False)


class UserSettingsDialog(Dialog):
    """Dialog for inspecting and editing the user settings YAML file."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("User Settings")
        self.setModal(False)
        self.resize(600, 500)

        layout = QtWidgets.QVBoxLayout(self)

        path_label = QtWidgets.QLabel(
            f"Settings file: {io._user_settings_filename()}\n"
            "Warning: editing this file can affect the behavior of Picasso.\n"
            "Clearing the file will reset all settings to their default "
            "values."
        )
        path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(path_label)

        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setFont(QtGui.QFont("Helvetica", 12))
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
        if value >= self.maximum() and self.finished is False:
            self.finished = True
            self.play_sound_notification()
        # if value is above zero, count has started, enabling time estimate
        if not self.count_started:
            if value > 0:
                self.count_started = True
                self.t0_est = time.time()
        self.app.processEvents()

    def closeEvent(self, event):
        _dialogs.remove(self)
        if self.finished is False:
            self.finished = True
            self.play_sound_notification()

    def zero_progress(self, description=None):
        """Set progress dialog to zero and changes title if given."""
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
        at least a minute has passed since the dialog was opened."""
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                playsound(self.sound_notification_path, block=False)

    def get_iterator(self, start=None, end=None):
        """Get an iterator for the progress dialog."""
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
        QtCore.QCoreApplication.instance().processEvents()

    def closeEvent(self, event):
        _dialogs.remove(self)
        if self.sound_notification_path is not None:
            if time.time() - self.t0 > SOUND_NOTIFICATION_DURATION:
                playsound(self.sound_notification_path, block=False)


class MockProgress:
    """Class to mock a progress bar or dialog, allowing for calling
    the same methods but not displaying anything."""

    def __init__(self, *args, **kwargs):
        pass

    def init(self, *args, **kwargs):
        pass

    def set_value(self, *args, **kwargs):
        pass

    def setMaximum(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def closeEvent(self, *args, **kwargs):
        pass

    def zero_progress(self, description, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def setLabelText(self, *args, **kwargs):
        pass

    def play_sound_notification(self, *args, **kwargs):
        pass

    def get_iterator(self, start=0, end=100):
        return range(start, end)


class TqdmProgress:
    """Class to absorb calls to ProgressDialog but is used to display
    tqdm progress bar instead."""

    def __init__(self, *args, **kwargs):
        self.description_base = (
            "" if "description" not in kwargs else kwargs["description"]
        )
        self.iterator = None

    def init(self, *args, **kwargs):
        pass

    def set_value(self, value, *args, **kwargs):
        if self.iterator is not None:
            self.iterator.update(value - self.iterator.n)

    def setMaximum(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def closeEvent(self, *args, **kwargs):
        pass

    def zero_progress(self, description, *args, **kwargs):
        self.description_base = description

    def close(self, *args, **kwargs):
        pass

    def setLabelText(self, *args, **kwargs):
        pass

    def play_sound_notification(self, *args, **kwargs):
        pass

    def get_iterator(self, start=0, end=100, unit="segment"):
        """Get an iterator for the progress bar."""
        iterator = tqdm(
            range(start, end),
            desc=self.description_base,
            unit=unit,
        )
        self.iterator = iterator
        return iterator


# type alias for the progress dialogs
ProgressType: TypeAlias = ProgressDialog | MockProgress | TqdmProgress


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
        """Add a widget to the grid layout inside the scroll area."""
        self.content_layout.addWidget(widget, row, column, height, width)

    def remove_widget(self, widget):
        """Remove a widget from the grid layout inside the scroll
        area."""
        self.content_layout.removeWidget(widget)

    def remove_all_widgets(self, keep_labels=False):
        """Remove all widgets. If ``keep_labels`` is True, the QLabels
        are kept."""
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
        if steps > 0:
            if self.value() <= 10 ** (-self.decimals()):
                self.setValue(2 * 10 ** (-self.decimals()))
            else:
                self.setValue(self.value() * (self._factor**steps))
        elif steps < 0:
            self.setValue(self.value() / (self._factor ** abs(steps)))


class GenericPlotWindow(QtWidgets.QTabWidget):
    """Interface for displaying matplotlib plots in a separate
    window."""

    def __init__(self, window_title, app_name):
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvas,
            NavigationToolbar2QT,
        )

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


class AutoDict(collections.defaultdict):
    """A defaultdict whose auto-generated values are defaultdicts
    itself. This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


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
        self.setStyleSheet(
            """
            QToolButton {
                border: 1px solid palette(mid);
                border-radius: 11px;
                font-weight: bold;
                font-size: 12px;
                color: palette(button-text);
                background: palette(button);
            }
            QToolButton:hover {
                background: palette(highlight);
                color: palette(highlighted-text);
                border-color: palette(highlight);
            }
        """
        )
        self.clicked.connect(self._open_docs)

    def _open_docs(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.help_url))


def deprecation_warning(message: str) -> None:
    """Display a deprecation warning message.

    Parameters
    ----------
    message : str
        The deprecation warning message to be displayed.
    """
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def cancel_dialogs():
    """Closes all open dialogs (``ProgressDialog`` and ``StatusDialog``)
    in the GUI."""
    dialogs = [_ for _ in _dialogs]
    for dialog in dialogs:
        if isinstance(dialog, ProgressDialog):
            dialog.cancel()
        else:
            dialog.close()
    QtCore.QCoreApplication.instance().processEvents()  # just in case...


def install_excepthook(window) -> None:
    """Install a thread-safe excepthook that shows uncaught exceptions in a
    QMessageBox. Safe to call from QThread workers because the error signal is
    queued to the main thread by Qt's event loop."""

    class _ErrorSignaler(QtCore.QObject):
        error = QtCore.pyqtSignal(str)

    signaler = _ErrorSignaler()

    def _show_error(message: str) -> None:
        cancel_dialogs()
        QtWidgets.QMessageBox.critical(window, "An error occurred", message)

    signaler.error.connect(_show_error)

    def excepthook(type, value, tback):
        message = "".join(traceback.format_exception(type, value, tback))
        signaler.error.emit(message)
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook


def get_sound_notification_path() -> str | None:
    """Return the path to the sound notification file from the user
    settings file. If the file is not found or not specified, return
    None.

    Returns
    -------
    path : str or None
        Path to the sound notification file or None if not found or not
        specified.
    """
    settings = io.load_user_settings()
    if "Sound_notification" not in settings:  # add default settings (no sound)
        settings["Sound_notification"]["filename"] = None
        io.save_user_settings(settings)
    filename = settings["Sound_notification"]["filename"]
    sounds_dir = _sound_notification_dir()
    if filename is not None and os.path.isfile(
        os.path.join(sounds_dir, filename)
    ):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".mp3", ".wav"]:
            path = None
        else:
            path = os.path.join(sounds_dir, filename)
    else:
        path = None
    return path


def get_available_sound_notifications() -> list[str | None]:
    """Get a list of file names of the available sound notifications in
    the folder ``gui/notification_sounds``.

    Returns
    -------
    filenames : list of strs
        List of file names of the available sound notifications.
    """
    sounds_dir = _sound_notification_dir()
    filenames = [
        _
        for _ in os.listdir(sounds_dir)
        if os.path.isfile(os.path.join(sounds_dir, _))
        and os.path.splitext(_)[1].lower() in [".mp3", ".wav"]
    ]
    filenames = ["None"] + filenames
    return filenames


def set_sound_notification(action: QtGui.QAction) -> None:
    """Save the selected sound notification in the user settings
    file.

    Parameters
    ----------
    action : QtGui.QAction
        The action representing the selected sound notification.
    """
    settings = io.load_user_settings()
    selected_sound = action.objectName()  # file name with extension
    settings["Sound_notification"]["filename"] = selected_sound
    io.save_user_settings(settings)
    # play selected sound as a preview
    play_path = get_sound_notification_path()
    playsound(play_path, block=False) if play_path is not None else None


def _sound_notification_dir() -> str:
    """Return the path to the sound notification folder."""
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "gui",
        "notification_sounds",
    )


def adjust_widget_size(
    widget: QtWidgets.QWidget,
    size_hint: QtCore.QSize,
    width_offset: int = 0,
    height_offset: int = 0,
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
    """
    intended_width = size_hint.width() + width_offset
    intended_height = size_hint.height() + height_offset
    # adjust to the screen size if necessary
    screen = QtWidgets.QApplication.primaryScreen()
    screen_height = 1000 if screen is None else screen.size().height()
    screen_width = 1000 if screen is None else screen.size().width()
    intended_width = min(intended_width, screen_width - 200)
    intended_height = min(intended_height, screen_height - 200)
    widget.resize(intended_width, intended_height)


def get_from_metadata(
    info: list[dict] | dict,
    key: Any,
    default=None,
    *,
    raise_error: bool = False,
) -> Any:
    """Get a value from the localization metadata (list of dictionaries
    or a dictionary). Runs the search from the last to the first element
    of the input list. Returns default or raises an error if the key is
    not found.

    Parameters
    ----------
    info : list of dicts or dict
        Localization metadata.
    key : Any
        Key to be searched in the metadata.
    default : Any, optional
        Value to be returned if the key is not found. Default is None.
    raise_error : bool, optional
        If True, raises a KeyError if the key is not found. Default is
        False.

    Returns
    -------
    value : Any
        Value corresponding to the key in the metadata. If the key is
        not found, default is returned.
    """
    if isinstance(info, dict):
        if raise_error and key not in info:
            raise KeyError(f"Key '{key}' not found in metadata.")
        return info.get(key, default)
    elif isinstance(info, list):
        for inf in info[::-1]:
            if val := inf.get(key):
                return val
        if raise_error:
            raise KeyError(f"Key '{key}' not found in metadata.")
        return default
    else:
        raise ValueError("info must be a dict or a list of dicts.")


def extract_filter_steps(
    info: list[dict],
    current_columns,
) -> tuple[dict[str, list[float]], list[str], list[str]]:  # noqa: C901
    """Parse filter steps out of a Picasso Filter metadata list.

    Iterates ``info`` oldest -> newest. A dict is treated as a filter
    dict when its ``Generated by`` value contains ``"Filter"``. Numeric
    [min, max] ranges for columns present in ``current_columns`` are
    intersected; columns absent from the current data are reported as
    missing instead of being applied.

    Parameters
    ----------
    info : list of dicts
        Localization metadata loaded via ``io.load_info``.
    current_columns : iterable of str
        Columns available in the target localizations DataFrame.

    Returns
    -------
    ranges : dict[str, list[float]]
        Column -> [min, max] to apply.
    to_remove : list[str]
        Columns to drop (present in current data).
    missing : list[str]
        Columns referenced in metadata but absent from current data.
    """
    current = set(current_columns)
    ranges = {}
    to_remove_all = []
    missing = []

    for d in info:
        if not isinstance(d, dict):
            continue
        gen_by = get_from_metadata(d, "Generated by", default="")
        if "Filter" not in str(gen_by):
            continue
        for key, value in d.items():
            if key == "Generated by":
                continue
            if key == "Removed columns" and isinstance(value, (list, tuple)):
                to_remove_all.extend(value)
                continue
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            ):
                xmin, xmax = float(value[0]), float(value[1])
                if key not in current:
                    missing.append(key)
                    continue
                if key in ranges:
                    ranges[key][0] = max(ranges[key][0], xmin)
                    ranges[key][1] = min(ranges[key][1], xmax)
                else:
                    ranges[key] = [xmin, xmax]

    to_remove = [c for c in to_remove_all if c in current]
    for c in to_remove_all:
        if c not in current:
            missing.append(c)

    seen: set = set()
    missing_unique: list[str] = []
    for c in missing:
        if c not in seen:
            seen.add(c)
            missing_unique.append(c)

    return ranges, to_remove, missing_unique


def apply_filter_steps(
    locs: pd.DataFrame,
    info: list[dict],
) -> tuple[pd.DataFrame, dict[str, list[float]], list[str], list[str]]:
    """Apply Picasso Filter steps recorded in ``info`` to ``locs``.

    Thin wrapper around :func:`extract_filter_steps`: it parses the
    filter recipe out of the metadata, intersects each per-column
    [min, max] range against ``locs``, drops any "Removed columns",
    and reports columns that were referenced by the metadata but absent
    from ``locs``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to filter.
    info : list of dicts
        Localization metadata loaded via ``io.load_info``.

    Returns
    -------
    filtered_locs : pd.DataFrame
        ``locs`` with the range filters and column removals applied.
    ranges : dict[str, list[float]]
        Column -> [min, max] that were applied.
    to_remove : list[str]
        Columns that were dropped.
    missing : list[str]
        Columns referenced in ``info`` but not present in ``locs``
        (skipped, not applied).
    """
    ranges, to_remove, missing = extract_filter_steps(info, locs.columns)
    for field, (xmin, xmax) in ranges.items():
        locs = locs[(locs[field] > xmin) & (locs[field] < xmax)]
    if to_remove:
        locs = locs.drop(columns=to_remove)
    return locs, ranges, to_remove, missing


def overwrite_metadata(
    info: list[dict] | dict, key: Any, value: Any
) -> list[dict] | dict:
    """Overwrite a value in the localization metadata (list of
    dictionaries or a dictionary). If the key does not exist an error
    is raised.

    Parameters
    ----------
    info : list of dicts or dict
        Localization metadata.
    key : Any
        Key to be overwritten or added in the metadata.
    value : Any
        Value to be set for the key.

    Returns
    -------
    updated_info : list of dicts or dict
        Metadata with the updated value.

    Raises
    ------
    KeyError
        If the key is not found in the metadata.
    """
    success = False
    if isinstance(info, dict):
        if key in info:
            info[key] = value
            success = True
    elif isinstance(info, list):
        for inf in info[::-1]:
            if key in inf:
                inf[key] = value
                success = True
                break
    if not success:
        raise KeyError(f"Key '{key}' not found in metadata.")
    return info


def get_colors(n_channels):
    """Create a list with rgb channels for each channel.

    Colors go from red to green, blue, pink and red again.

    Parameters
    ----------
    n_channels : int
        Number of locs channels

    Returns
    -------
    list
        Contains tuples with rgb channels
    """
    hues = np.arange(0, 1, 1 / n_channels)
    colors = [colorsys.hsv_to_rgb(_, 1, 1) for _ in hues]
    return colors


def is_hexadecimal(text):
    """Check if text represents a hexadecimal code for rgb, for
    example ``#ff02d4``.

    Parameters
    ----------
    text : str
        String to be checked.

    Returns
    -------
    bool
        True if text represents rgb, False otherwise.
    """
    allowed_characters = "0123456789abcdefABCDEF"
    if isinstance(text, str) and text[0] == "#" and len(text) == 7:
        n_valid = sum(char in allowed_characters for char in text[1:])
        if n_valid == 6:
            return True
    return False


def is_path_available(
    path: str, *, check_ext: str | list[str] = "", parent=None
) -> bool:
    """Check if a file or folder exists at the given path. Returns True
    if there is not such path. Returns False if the path already exists.
    Allows to easily change the extension of the path.

    Parameters
    ----------
    path : str
        Path to be checked.
    check_ext : str or list of str, optional
        Other extension(s) to be checked if they're available. Default
        is "".
    parent : QWidget, optional
        Parent widget for the error message box if raise_error is True.
        A message box will be displayed showing asking if the user wants
        to continue without the file or folder if the path does not
        exist.

    Returns
    -------
    paths_available : list of bools
        For each path generated with the new extension, True if the path
        is available, False if the path already exists.

    Raises
    ------
    ValueError
        If check_ext is not empty and does not start with a dot.
    """
    if check_ext:
        if isinstance(check_ext, str):
            check_ext = [check_ext]
        paths = [os.path.splitext(path)[0] + ext for ext in check_ext]
    else:
        paths = [path]
    paths_available = []
    for path in paths:
        if os.path.exists(path):
            if parent is not None:
                box = QtWidgets.QMessageBox(parent)
                box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                box.setWindowTitle("File or folder already exists")
                box.setText(
                    f"The path '{path}' already exists."
                    "\nDo you wish to overwrite it?"
                )
                box.setStandardButtons(
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No
                )
                result = box.exec()
                if result != QtWidgets.QMessageBox.StandardButton.Yes:
                    paths_available.append(False)
                else:
                    paths_available.append(True)
            else:
                paths_available.append(False)
        else:
            paths_available.append(True)
    return paths_available


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


@numba.njit
def find_local_minima(arr: FloatArray1D) -> IntArray1D:
    """Find positions of the local minima in a 1D numpy array.

    Parameters
    ----------
    arr : FloatArray1D
        1D array.

    Returns
    -------
    local_minima_indices : IntArray1D
        Indices of the local minima in the array.
    """
    # Compare each element with its neighbors
    local_minima_mask = (arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:])
    # Get the indices of local minima (adjust by +1 due to slicing)
    local_minima_indices = np.where(local_minima_mask)[0] + 1
    return local_minima_indices


def cumulative_exponential(
    x: FloatArray1D,
    a: float,
    t: float,
    c: float,
) -> FloatArray1D:
    """Used for binding kinetics estimation."""
    return a * (1 - np.exp(-(x / t))) + c


def fit_cum_exp(data: FloatArray1D) -> dict:
    """Fit a cumulative exponential function to data. Used for binding
    kinetics estimation.

    Parameters
    ----------
    data : FloatArray1D
        Input data to fit, shape (N,).

    Returns
    -------
    result : dict
        Contains the best fit parameters and the fitted data.
    """
    data.sort()
    n = len(data)
    y = np.arange(1, n + 1)
    data_min = data.min()
    data_max = data.max()
    p0 = [n, np.mean(data), data_min]
    bounds = ([0, data_min, 0], [np.inf, data_max, np.inf])
    popt, _ = optimize.curve_fit(
        cumulative_exponential, data, y, p0=p0, bounds=bounds
    )
    result = {
        "best_values": {"a": popt[0], "t": popt[1], "c": popt[2]},
        "data": data,
        "best_fit": cumulative_exponential(data, *popt),
    }
    return result


def estimate_kinetic_rate(data: FloatArray1D) -> float:
    """Find the mean dark/bright time by fitting to a cumulative
    exponential function.

    Parameters
    ----------
    data : FloatArray1D
        Input data to fit, shape (N,).

    Returns
    -------
    rate : float
        Mean dark/bright time from the fitted exponential function.
    """
    if len(data) > 2:
        if data.max() - data.min() == 0:
            rate = np.nanmean(data)
        else:
            result = fit_cum_exp(data)
            rate = result["best_values"]["t"]
    else:
        rate = np.nanmean(data)
    return rate


def plot_cumulative_exponential_fit(
    data: SeriesOrFloatArray1D,
    fit_result: dict,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a histogram for experimental data and the fitted cumulative
    exponential function. Used for binding kinetics fit display.

    Parameters
    ----------
    data : SeriesOrFloatArray1D
        Input data to fit, shape (N,). For example, bright or dark
        times.
    fit_result : dict
        Output of `fit_cum_exp` containing the best fit parameters and
        the fitted data.
    fig, ax : plt.Figure and plt.Axes, optional
        If given, the plot will be drawn on the given figure and axes.
        Otherwise, a new figure and axes will be created.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()

    # Bright
    a = fit_result["best_values"]["a"]
    t = fit_result["best_values"]["t"]
    c = fit_result["best_values"]["c"]

    ax.set_title(
        "Cumulative exponential\n"
        r"$Fit: {:.2f}\cdot(1-exp(-t/{:.2f}))+{:.2f}$".format(a, t, c)
    )
    data = data.copy()
    data.sort_values(inplace=True)
    y = np.arange(1, len(data) + 1)
    ax.semilogx(data, y, label="data")
    ax.semilogx(
        data,
        fit_result["best_fit"],
        label=f"fit ($\\bar \\tau = {t:.2f}$)",
    )
    ax.legend(loc="best")
    ax.set_xlabel("Duration (frames)")
    ax.set_ylabel("Counts")
    return fig


def plot_trace(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    fig: plt.Figure | None = None,
    include_photons: bool = True,
    return_trace: bool = False,
) -> (
    plt.Figure
    | tuple[
        plt.Figure,
        tuple[FloatArray1D, FloatArray1D, FloatArray1D]
        | tuple[FloatArray1D, FloatArray1D],
    ]
):
    """Plot the trace of a localization over time, showing the x and y
    positions and the spot size.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list[dict]
        Additional information for each localization.
    fig : plt.Figure, optional
        If given, the plot will be drawn on the given figure. Otherwise,
        a new figure will be created.
    include_photons : bool, optional
        If True, the photon count will also be plotted as well. Default
        is True.
    return_trace : bool, optional
        If True, the trace data will be returned as well. Default is
        False.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    trace_data : tuple of FloatArray1D, optional
        If return_trace is True, a tuple containing the x vector
        (frames), the y vector (localization ON/OFF) and the photon
        count vector (if include_photons is True) will be returned.
    """
    if fig is None:
        if include_photons:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                4, 1, figsize=(5, 5), constrained_layout=True, sharex=True
            )
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(5, 5), constrained_layout=True, sharex=True
            )
    else:
        fig.clear()
        if include_photons:
            ax1, ax2, ax3, ax4 = fig.subplots(4, sharex=True)
        else:
            ax1, ax2, ax3 = fig.subplots(3, sharex=True)

    n_frames = get_from_metadata(info, "Frames", raise_error=True)
    xvec = np.arange(n_frames)
    yvec = xvec[:] * 0
    yvec[locs["frame"]] = 1
    yvec_ph = xvec[:] * 0
    if "photons" in locs.columns:
        yvec_ph[locs["frame"]] = locs["photons"]
    else:
        yvec_ph = np.zeros_like(xvec)
    trace_data = (xvec, yvec, yvec_ph) if include_photons else (xvec, yvec)

    # frame vs x
    ax1.scatter(locs["frame"], locs["x"], s=2)
    ax1.set_title("X-pos vs frame")
    ax1.set_xlim(0, n_frames)
    ax1.set_ylabel("X-pos [Px]")

    # frame vs y
    ax2.scatter(locs["frame"], locs["y"], s=2)
    ax2.set_title("Y-pos vs frame")
    ax2.set_ylabel("Y-pos [Px]")

    # locs in time
    ax3.plot(xvec, yvec, linewidth=1)
    ax3.fill_between(xvec, 0, yvec, facecolor="red")
    ax3.set_title("Localizations")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("ON")
    ax3.set_yticks([0, 1])
    ax3.set_ylim([-0.1, 1.1])

    if include_photons:
        ax4.plot(xvec, yvec_ph, linewidth=1)
        ax4.set_title("Photons")
        ax4.set_xlabel("Frames")
        ax4.set_ylabel("Photons")
        ax4.set_ylim([0, yvec_ph.max() * 1.1])

    if return_trace:
        return fig, trace_data
    else:
        return fig


def unpack_calibration(
    calibration: dict,
    pixelsize: float,
) -> tuple[FloatArray2D, FloatArray1D, float]:
    """Extract calibration file for 3D G5M. Return spot widths and
    heights and the corresponding z values + magnification factor.

    New in v0.10.0: the function is deprecated and will be removed in
    Picasso 0.11.0.

    Parameters
    ----------
    calibration : dict
        Calibration dictionary with x and y coefficients, z step
        size and the number of frames.
    pixelsize : float
        Camera pixel size in nm.

    Returns
    -------
    spot_size : FloatArray2D
        Spot width and height from the 3D calibration for each z
        position.
    z_range : FloatArray1D
        Z values (in camera pixels) corresponding to the spot ratios.
    mag_factor : float
        Magnification factor for the 3D calibration.
    """
    deprecation_warning(
        "The function 'unpack_calibration' is deprecated and will be"
        " removed in Picasso 0.11.0. 3D G5M, for which this function"
        " was originally implemented, only requires x and y"
        " coefficients."
    )
    cx = calibration["X Coefficients"]
    cy = calibration["Y Coefficients"]
    z_step_size = calibration["Step size in nm"]
    n_frames = calibration["Number of frames"]
    mag_factor = calibration["Magnification factor"]

    frame_range = np.arange(n_frames)
    z_total_range = (n_frames - 1) * z_step_size
    z_range = -(frame_range * z_step_size - z_total_range / 2)

    spot_width = np.polyval(cx, z_range)
    spot_height = np.polyval(cy, z_range)
    spot_size = np.stack((spot_width, spot_height))

    z_range /= pixelsize
    return spot_size, z_range, mag_factor


def calculate_optimal_bins(
    data: FloatArray1D | IntArray1D,
    max_n_bins: int | None = None,
) -> FloatArray1D:
    """Calculate the optimal bins for display, for example, in
    Picasso: Filter.

    Parameters
    ----------
    data : FloatArray1D | IntArray1D
        Data to be binned.
    max_n_bins : int | None, optional
        Maximum number of bins.

    Returns
    -------
    bins : FloatArray1D
        Bins for display.
    """
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return np.array([data[0] - 1.0, data[0] + 1.0])
    bin_size = 2 * iqr * len(data) ** (-1 / 3)
    if data.dtype.kind in ("u", "i") and bin_size < 1:
        bin_size = 1
    bin_min = data.min() - bin_size / 2
    try:
        n_bins = (data.max() - bin_min) / bin_size
        n_bins = int(n_bins)
    except Exception:
        n_bins = 10
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    bins = np.linspace(bin_min, data.max(), n_bins)
    return bins


def append_to_rec(
    rec_array: np.recarray,
    data: FloatArray1D | IntArray1D,
    name: str,
) -> np.recarray:
    """Append a new column to the existing np.recarray.

    Parameters
    ----------
    rec_array : np.recarray
        Recarray to which the new column is appended.
    data : FloatArray1D | IntArray1D
        1D data to be appended.
    name : str
        Name of the new column.

    Returns
    -------
    rec_array : np.recarray
        Recarray with the new column.
    """
    deprecation_warning(
        "Appending to recarrays is deprecated and will be removed in Picasso"
        " 1.0. Since 0.9.0, Picasso uses pandas DataFrames instead of"
        " recarrays. Simply use locs['new_column'] = data to add a new column"
        " to the DataFrame."
    )
    if hasattr(rec_array, name):
        rec_array = remove_from_rec(rec_array, name)
    rec_array = append_fields(
        rec_array,
        name,
        data,
        dtypes=data.dtype,
        usemask=False,
        asrecarray=True,
    )
    return rec_array


def merge_locs(
    locs_list: list[pd.DataFrame],
    increment_frames: bool | list[int] = True,
    increment_groups: bool | list[int] = True,
) -> pd.DataFrame:
    """Merge localization lists into one file. Can increment frames
    to avoid overlapping frames.

    Parameters
    ----------
    locs_list : list of pd.DataFrame's
        List of localization lists to be merged.
    increment_frames : bool or list, optional
        If True, increments frames of each localization list by the
        maximum frame number of the previous localization list. If a
        list is given, each element is an integer increment of the frame
        indices for each localization list. Default is True.
    increment_groups : bool or list, optional
        If True, increments group indices of each localization list by
        the maximum group number of the previous localization list. If a
        list is given, each element is an integer increment of the group
        indices for each localization list. Default is True.

    Returns
    -------
    locs : pd.DataFrame
        Merged localizations.
    """
    assert isinstance(
        increment_frames, (bool, list)
    ), "increment_frames must be a boolean or a list of integers."
    assert isinstance(
        increment_groups, (bool, list)
    ), "increment_groups must be a boolean or a list of integers."
    if isinstance(increment_frames, list):
        assert len(increment_frames) == len(locs_list), (
            "If increment_frames is a list, its length must be the same"
            " as locs_list."
        )
        assert all(isinstance(i, int) for i in increment_frames), (
            "If increment_frames is a list, all its elements must be "
            "integers."
        )
    if isinstance(increment_groups, list):
        assert len(increment_groups) == len(locs_list), (
            "If increment_groups is a list, its length must be the same"
            " as locs_list."
        )
        assert all(isinstance(i, int) for i in increment_groups), (
            "If increment_groups is a list, all its elements must be "
            "integers."
        )
    # convert boolean increments to lists of integers
    if increment_frames is True:
        increment_frames = np.cumsum(
            [0] + [locs["frame"].max() for locs in locs_list[:-1]]
        ).tolist()
    else:
        increment_frames = [0] * len(locs_list)
    if increment_groups is True:
        increment_groups = np.cumsum(
            [0] + [locs["group"].max() for locs in locs_list[:-1]]
        ).tolist()
    else:
        increment_groups = [0] * len(locs_list)
    return _merge_locs(locs_list, increment_frames, increment_groups)


def _merge_locs(
    locs_list: list[pd.DataFrame],
    increment_frames: list[int],
    increment_groups: list[int],
) -> pd.DataFrame:
    """Helper function for merge_locs. Assumes correct input types and
    values."""
    locs_list = locs_list.copy()
    for i, locs in enumerate(locs_list):
        locs["frame"] += increment_frames[i]
        if "group" in locs.columns:
            locs["group"] += increment_groups[i]
        locs_list[i] = locs
    locs = pd.concat(locs_list, ignore_index=True)
    locs.sort_values(by="frame", inplace=True)
    return locs


def ensure_sanity(locs: pd.DataFrame, info: list[dict]) -> pd.DataFrame:
    """Ensure that localizations are within the image dimensions
    and have positive localization precisions and other parameters.

    v0.9.6: check that the info metadata contains the necessary
    information for processing: Width, Height, Pixelsize and Frames.
    Raises a KeyError if any of the required keys is missing.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Localization metadata.

    Returns
    -------
    locs : pd.DataFrame
        Localizations that pass the sanity checks.
    """
    locs = locs.copy()  # pandas SettingWithCopyWarning
    # no inf and nan:
    locs.replace([np.inf, -np.inf], np.nan, inplace=True)
    locs.dropna(axis=0, how="any", inplace=True)
    # other sanity checks:
    required_keys = ["Width", "Height", "Frames"]
    for key in required_keys:
        value = get_from_metadata(info, key)
        if value is None:
            raise KeyError(f"Metadata is missing required key: '{key}'")

    locs = locs[locs["x"] < get_from_metadata(info, "Width")]
    locs = locs[locs["y"] < get_from_metadata(info, "Height")]
    for attr in [
        "x",
        "y",
        "lpx",
        "lpy",
        "lpz",
        "photons",
        "ellipticity",
        "sx",
        "sy",
    ]:
        if attr in locs.columns:
            locs = locs[locs[attr] >= 0]
    return locs


def is_loc_at(x: float, y: float, locs: pd.DataFrame, r: float) -> BoolArray1D:
    """Check which localizations are within radius ``r`` from position
    ``(x, y)``.

    Parameters
    ----------
    x, y : float
        x and y-coordinate of the position.
    locs : pd.DataFrame
        Localizations.
    r : float
        Radius.

    Returns
    -------
    is_picked : BoolArray1D
        Boolean array - True if a localization is within radius r
        of position (x, y).
    """
    dx = locs["x"] - x
    dy = locs["y"] - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked.to_numpy()


def locs_at(x: float, y: float, locs: pd.DataFrame, r: float) -> pd.DataFrame:
    """Return localizations within radius ``r`` from the position
    ``(x, y)``.

    Parameters
    ----------
    x, y : float
        x and y-coordinate of the position.
    locs : pd.DataFrame
        Localizations.
    r : float
        Radius.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations in the specified area.
    """
    is_picked = is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    return picked_locs


@numba.jit(nopython=True)
def check_if_in_polygon(
    x: FloatArray1D,
    y: FloatArray1D,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> BoolArray1D:
    """Check if points ``(x, y)`` are within the polygon defined by
    corners ``(X, Y)``. Uses the ray casting algorithm, see
    ``check_if_in_rectangle`` for details.

    Parameters
    ----------
    x, y : FloatArray1D
        x and y coordinates of points.
    X, Y : FloatArray1D
        x and y coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : BoolArray1D
        Boolean array indicating which points are in the polygon.
    """
    n_locs = len(x)
    n_polygon = len(X)
    is_in_polygon = np.zeros(n_locs, dtype=np.bool_)

    for i in range(n_locs):
        count = 0
        for j in range(n_polygon):
            j_next = (j + 1) % n_polygon
            if ((Y[j] > y[i]) != (Y[j_next] > y[i])) and (
                (
                    x[i]
                    < X[j]
                    + (X[j_next] - X[j]) * (y[i] - Y[j]) / (Y[j_next] - Y[j])
                )
            ):
                count += 1
        if count % 2 == 1:
            is_in_polygon[i] = True

    return is_in_polygon


def locs_in_polygon(
    locs: pd.DataFrame,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> pd.DataFrame:
    """Return localizations within the polygon defined by corners
    ``(X, Y)``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    X, Y : FloatArray1D
        x and y-coordinates of polygon corners.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations in polygon.
    """
    is_in_polygon = check_if_in_polygon(
        locs["x"].to_numpy(), locs["y"].to_numpy(), np.array(X), np.array(Y)
    )
    return locs[is_in_polygon]


@numba.jit(nopython=True)
def check_if_in_rectangle(
    x: FloatArray1D,
    y: FloatArray1D,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> BoolArray1D:
    """Check if locs with coordinates (x, y) are in rectangle with
    corners (X, Y) by counting the number of rectangle sides which are
    hit by a ray originating from each loc to the right. If the number
    of hit rectangle sides is odd, then the loc is in the rectangle.

    Parameters
    ----------
    x, y : FloatArray1D
        x and y coordinates of points.
    X, Y : FloatArray1D
        x and y coordinates of polygon corners.

    Returns
    -------
    is_in_polygon : BoolArray1D
        Boolean array indicating if point is in polygon.
    """
    n_locs = len(x)
    ray_hits_rectangle_side = np.zeros((n_locs, 4))
    for i in range(4):
        # get two y coordinates of corner points forming one rectangle side
        y_corner_1 = Y[i]
        # take the first if we're at the last side:
        y_corner_2 = Y[0] if i == 3 else Y[i + 1]
        y_corners_min = min(y_corner_1, y_corner_2)
        y_corners_max = max(y_corner_1, y_corner_2)
        for j in range(n_locs):
            y_loc = y[j]
            # only if loc is on level of rectangle side, its ray can hit:
            if y_corners_min <= y_loc <= y_corners_max:
                x_corner_1 = X[i]
                # take the first if we're at the last side:
                x_corner_2 = X[0] if i == 3 else X[i + 1]
                # calculate intersection point of ray and side:
                m_inv = (x_corner_2 - x_corner_1) / (y_corner_2 - y_corner_1)
                x_intersect = m_inv * (y_loc - y_corner_1) + x_corner_1
                x_loc = x[j]
                if x_intersect >= x_loc:
                    # ray hits rectangle side on the right side
                    ray_hits_rectangle_side[j, i] = 1
    n_sides_hit = np.sum(ray_hits_rectangle_side, axis=1)
    is_in_rectangle = n_sides_hit % 2 == 1
    return is_in_rectangle


def locs_in_rectangle(
    locs: pd.DataFrame,
    X: FloatArray1D,
    Y: FloatArray1D,
) -> pd.DataFrame:
    """Return localizations within the rectangle defined by corners
    ``(X, Y)``.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    X, Y : FloatArray1D
        x and y coordinates of rectangle corners.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations in rectangle.
    """
    is_in_rectangle = check_if_in_rectangle(
        locs["x"].to_numpy(), locs["y"].to_numpy(), np.array(X), np.array(Y)
    )
    picked_locs = locs[is_in_rectangle]
    return picked_locs


def minimize_shifts(
    shifts_x: FloatArray2D,
    shifts_y: FloatArray2D,
    shifts_z: FloatArray2D | None = None,
) -> tuple[FloatArray1D, FloatArray1D, FloatArray1D | None]:
    """Minimize shifts in x, y, and z directions. Used for drift
    correction.

    Parameters
    ----------
    shifts_x, shifts_y : FloatArray2D
        Shifts in x and y directions, shape (n_channels, n_channels).
    shifts_z : FloatArray2D, optional
        Shifts in z direction, shape (n_channels, n_channels). If None,
        only x and y shifts are minimized.

    Returns
    -------
    shift_y, shift_x : FloatArray1D
        Minimized shifts in y and x direction.
    shift_z : FloatArray1D, optional
        Minimized shifts in z direction if ``shifts_z`` is specified.
    """
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = np.zeros((n_pairs, n_dims))
    A = np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            rij[flag, 0] = shifts_y[i, j]
            rij[flag, 1] = shifts_x[i, j]
            if n_dims == 3:
                rij[flag, 2] = shifts_z[i, j]
            A[flag, i:j] = 1
            flag += 1
    Dj = np.dot(np.linalg.pinv(A), rij)
    shift_y = np.insert(np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = np.insert(np.cumsum(Dj[:, 1]), 0, 0)
    if n_dims == 2:
        return shift_y, shift_x
    else:
        shift_z = np.insert(np.cumsum(Dj[:, 2]), 0, 0)
        return shift_y, shift_x, shift_z


def n_futures_done(futures: list[Future]) -> int:
    """Return the number of finished futures, used in
    multiprocessing."""
    return sum([_.done() for _ in futures])


def remove_from_rec(rec_array: np.recarray, name: str) -> np.recarray:
    """Remove a column from the existing recarray.

    Parameters
    ----------
    rec_array : np.recarray
        Recarray from which the column is removed.
    name : str
        Name of the column to be removed.

    Returns
    -------
    rec_array : np.recarray
        Recarray without the column.
    """
    deprecation_warning(
        "Removing columns from recarrays is deprecated and will be removed in "
        " Picasso 1.0. Since 0.9.0, Picasso uses pandas DataFrames instead of"
        " recarrays. Simply use locs.drop('new_column', axis=1) to remove a"
        " column from the DataFrame."
    )
    rec_array = drop_fields(rec_array, name, usemask=False, asrecarray=True)
    return rec_array


def locs_glob_map(
    func: Callable[
        [pd.DataFrame, dict, str, Any], tuple[pd.DataFrame, list[dict]]
    ],
    pattern: str,
    args: list = [],
    kwargs: dict = {},
    extension: str = "",
) -> None:
    """Map a function to localization files, specified by the unix style
    path pattern.

    The function must take two arguments: ``locs`` and ``info``. It may
    take additional args and kwargs which are supplied to this map
    function. A new locs file will be saved if an extension is provided.
    In that case, the mapped function must return new locs and a new
    info dict.

    Parameters
    ----------
    func : Callable
        Function to be mapped to each locs file. It must take
        locs, info, path, and any additional args and kwargs.
    pattern : str
        Unix style path pattern to match locs files.
    args : list, optional
        Additional positional arguments to be passed to the function.
    kwargs : dict, optional
        Additional keyword arguments to be passed to the function.
    extension : str, optional
        If provided, the mapped function must return new locs and info
        dict, and a new locs file will be saved with this extension.
        If not provided, the function is expected to modify locs and
        info in place.
    """
    paths = glob.glob(pattern)
    for path in paths:
        locs, info = io.load_locs(path)
        result = func(locs, info, path, *args, **kwargs)
        if extension:
            base, ext = os.path.splitext(path)
            out_path = base + "_" + extension + ".hdf5"
            locs, info = result
            io.save_locs(out_path, locs, info)


def get_pick_polygon_corners(
    pick: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Return X and Y coordinates of a pick polygon.
    Return (None, None) if the pick is not a closed polygon.

    Parameters
    ----------
    pick : list of tuples
        List of tuples, each tuple contains x and y coordinates of a
        polygon corner.

    Returns
    -------
    X, Y : list of floats
        Lists of x and y coordinates of the polygon corners.
        Return (None, None) if the pick is not a closed polygon.
    """
    if len(pick) < 3 or pick[0] != pick[-1]:
        return None, None
    else:
        X = [_[0] for _ in pick]
        Y = [_[1] for _ in pick]
        return X, Y


def get_pick_rectangle_corners(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float,
) -> tuple[list[float], list[float]]:
    """Find the positions of corners of a rectangular pick.
    A rectangular pick is defined by:
        [(start_x, start_y), (end_x, end_y)]
    and its width. (all values in camera pixels).

    Parameters
    ----------
    start_x, start_y : float
        Starting point of the pick.
    end_x, end_y : float
        Ending point of the pick.
    width : float
        Width of the pick in camera pixels.

    Returns
    -------
    corners : tuple
        Contains corners' x and y coordinates in two lists.
    """
    if end_x == start_x:
        alpha = np.pi / 2
    else:
        alpha = np.arctan((end_y - start_y) / (end_x - start_x))
    dx = width * np.sin(alpha) / 2
    dy = width * np.cos(alpha) / 2
    x1 = float(start_x - dx)
    x2 = float(start_x + dx)
    x4 = float(end_x - dx)
    x3 = float(end_x + dx)
    y1 = float(start_y + dy)
    y2 = float(start_y - dy)
    y4 = float(end_y + dy)
    y3 = float(end_y - dy)
    corners = ([x1, x2, x3, x4], [y1, y2, y3, y4])
    return corners


def polygon_area(X: FloatArray1D, Y: FloatArray1D) -> float:
    """Find the area of a polygon defined by corners X and Y.

    Parameters
    ----------
    X, Y : FloatArray1D
        x-coordinates and y-coordinates of the polygon corners.

    Returns
    -------
    area : float
        Area of the polygon.
    """
    n_corners = len(X)
    area = 0
    for i in range(n_corners):
        j = (i + 1) % n_corners  # next corner
        area += X[i] * Y[j] - X[j] * Y[i]
    area = abs(area) / 2
    return area


def _pick_areas_polygon(
    picks: list[list[tuple[float, float]]],
) -> FloatArray1D:
    """Return pick areas for each polygonal pick in picks.

    Parameters
    ----------
    picks : list of lists of tuples
        List of picks, each pick is a list of (x, y) coordinates of the
        polygon corners.

    Returns
    -------
    areas : FloatArray1D
        Pick areas.
    """
    areas = []
    for i, pick in enumerate(picks):
        if len(pick) < 3 or pick[0] != pick[-1]:  # not a closed polygon
            continue
        X, Y = get_pick_polygon_corners(pick)
        areas.append(polygon_area(X, Y))
    areas = np.array(areas)
    areas = areas[areas > 0]  # remove open polygons
    return areas


def _pick_areas_rectangle(
    picks: list[list[tuple[float, float]]],
    w: float,
) -> FloatArray1D:
    """Return pick areas for each pick in picks.

    Parameters
    ----------
    picks : list
        List of picks, each pick is a list of coordinates of the
        rectangle corners.
    w : float
        Pick width.

    Returns
    -------
    areas : FloatArray1D
        Pick areas, same units as ``w``.
    """
    areas = np.zeros(len(picks))
    for i, pick in enumerate(picks):
        (xs, ys), (xe, ye) = pick
        areas[i] = w * np.sqrt((xe - xs) ** 2 + (ye - ys) ** 2)
    return areas


def pick_areas(
    picks: list[tuple],
    pick_shape: Literal["Circle", "Rectangle", "Polygon", "Square"],
    pick_size: float | None,
) -> FloatArray1D:
    """Get pick areas for each pick in picks.

    Parameters
    ----------
    picks : list of tuples
        Coordinates of picks in camera pixels.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square"}
        Shape of picks.
    pick_size : float or None
        Size of picks in camera pixels. For circles - diameters. For
        rectangles - width. For squares - side length. For polygons -
        ignored.

    Returns
    -------
    areas : FloatArray1D
        Pick areas in camera pixels squared.
    """
    if pick_shape == "Circle":
        r = pick_size / 2
        # same area for all picks
        areas = np.pi * r**2 * np.ones(len(picks))
    elif pick_shape == "Rectangle":
        areas = _pick_areas_rectangle(picks, pick_size)
    elif pick_shape == "Polygon":
        areas = _pick_areas_polygon(picks)
    elif pick_shape == "Square":
        # same area for all picks
        areas = pick_size**2 * np.ones(len(picks))
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")
    return areas


def permutation_test(
    arr1: FloatArray1D, arr2: FloatArray1D, iterations: int = 1000
) -> tuple[float, float, float]:
    """Perform a permutation test to compare two arrays. The test
    statistic is the Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    arr1, arr2 : FloatArray1D
        Arrays to be compared.
    iterations : int, optional
        Number of permutations to perform. Default is 1000.

    Returns
    -------
    obs_d : float
        Observed KS statistic.
    p_perm : float
        Permutation p-value.
    ks_pval : float
        KS test theoretical p-value.
    """
    combined = np.concatenate([arr1, arr2])
    n1 = len(arr1)

    # observe the real difference
    obs_d, ks_pval = stats.ks_2samp(arr1, arr2)

    # build null distribution by shuffling
    null_dist = []
    for _ in range(iterations):
        shuffled = np.random.permutation(combined)
        d_perm, _ = stats.ks_2samp(shuffled[:n1], shuffled[n1:])
        null_dist.append(d_perm)

    p_perm = np.sum(np.array(null_dist) >= obs_d) / iterations
    return obs_d, p_perm, ks_pval


def plot_subclustering_check(
    clustered_n_events: IntArray1D,
    sparse_n_events: IntArray1D,
    plot_path: str | list[str] = "",
    return_fig: bool = False,
    clustering_dist: float | None = None,
    sparse_dist: float | None = None,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    """Plot the results of subclustering analysis, see
    ``picasso.clusterer.test_subclustering``.

    Parameters
    ----------
    clustered_n_events : IntArray1D
        Number of events for clustered molecules.
    sparse_n_eveents : IntArray1D
        Number of events for sparse molecules.
    plot_path : str or list of strs, optional
        If provided, the plot is saved to this path. If a list of
        strings is given, each is used to save a separate plot. Default
        is "".
    return_fig : bool, optional
        If True, the figure and axes are returned. Default is False.
    clustering_dist, sparse_dist : float, optional
        Clustering and sparse distances that are displayed in the
        legend. If None, distances are not displayed. Default is None.

    Returns
    -------
    fig, ax : (plt.Figure, plt.Axes) or (None, None)
        Figure and axes if ``return_fig`` is True, otherwise
        (None, None).
    """
    m_clustered = clustered_n_events.mean()
    m_sparse = sparse_n_events.mean()
    s_clustered = clustered_n_events.std()
    s_sparse = sparse_n_events.std()

    # create the plot
    fig, ax1 = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
    min_bin, max_bin = np.percentile(clustered_n_events, [2.5, 97.5])
    vals, counts = np.unique(clustered_n_events, return_counts=True)
    if clustering_dist is not None:
        label = (
            f"Clustered (d < {clustering_dist:.1f} nm) "
            f"{m_clustered:.1f} +/- {s_clustered:.1f}"
        )
    else:
        label = f"Clustered {m_clustered:.1f} +/- {s_clustered:.1f}"
    ax1.bar(
        vals,
        counts,
        width=0.8,
        alpha=0.5,
        label=label,
        color="C0",
    )
    ax1.axvline(m_clustered, color="C0", linestyle="--")
    vals, counts = np.unique(sparse_n_events, return_counts=True)
    if sparse_dist is not None:
        label = (
            f"Sparse (d > {sparse_dist:.1f} nm) "
            f"{m_sparse:.1f} +/- {s_sparse:.1f}"
        )
    else:
        label = f"Sparse {m_sparse:.1f} +/- {s_sparse:.1f}"
    ax1.bar(
        vals,
        counts,
        width=0.8,
        alpha=0.5,
        label=label,
        color="C1",
    )
    ax1.axvline(m_sparse, color="C1", linestyle="--")
    ax1.set_xlabel("Number of events")
    ax1.set_ylabel("Counts")
    ax1.set_xlim(min_bin - 1, max_bin + 1)
    # add stat. tests in the title:
    stat, p_perm, p = permutation_test(clustered_n_events, sparse_n_events)
    p_value_str = r"$p_{value}$"
    title = (
        f"KS test: stat={stat:.4f}\n"
        f"permutation {p_value_str}={p_perm:.4f}\n"
        f"theoretical {p_value_str}={p:.4f}"
    )
    ax1.set_title(title, fontsize=10)
    ax1.legend()
    if len(plot_path):
        if isinstance(plot_path, str):
            plot_path = [plot_path]
        for path in plot_path:
            fig.savefig(path, dpi=300)

    if return_fig:
        return fig, ax1
    else:
        plt.close(fig)
        return None, None


def plot_rel_sigma_check(
    mols: pd.DataFrame, info: list[dict], path: str
) -> None:
    """Plot the relative sigma of G5M molecules to inspect if lp values
    reflect the experimental sizes of localization clouds.

    Parameters
    ----------
    mols : pd.DataFrame
        Molecules to be plotted, output of ``picasso.g5m.g5m``.
    info : list of dicts
        Molecuels metadata.
    path : str
        Path to save the plot.
    """
    if "z" in mols.columns:
        # three plots, one for each dimension
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True)
        bins = calculate_optimal_bins(
            np.concatenate(
                (mols["rel_sigma_x"], mols["rel_sigma_y"], mols["rel_sigma_z"])
            )
        )
        for i, dim in enumerate(["x", "y", "z"]):
            ax = axes[i]
            ax.hist(
                mols[f"rel_sigma_{dim}"], bins=bins, color=f"C{i}", alpha=0.7
            )
            ax.set_xlabel(f"Relative sigma {dim}")
            ax.set_ylabel("Counts")
        fig.savefig(path, dpi=300)
        plt.close(fig)
    else:
        # only one plot
        fig, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
        bins = calculate_optimal_bins(mols["rel_sigma"])
        ax.hist(mols["rel_sigma"], bins=bins, color="C0", alpha=0.7)
        ax.set_xlabel("Relative sigma")
        ax.set_ylabel("Counts")
        fig.savefig(path, dpi=300)
        plt.close(fig)


def unfold_localizations_square(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    n_square: int = 10,
    spacing: int | float = 1,
):
    """Shift localizations onto a square grid (tile) based on their
    group indices. The localizations must contain a 'group' column.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be unfolded. Must contain a 'group' column.
    info : list of dicts
        Localization metadata.
    n_square : int, optional
        Number of groups per square side. Default is 10.
    spacing : int or float, optional
        Spacing between groups in camera pixels. Default is 1.

    Returns
    -------
    shifted_locs : pd.DataFrame
        Localizations shifted onto a square grid based on their group
        indices.
    updated_info : list of dicts
        Updated metadata with new FOV dimensions after unfolding.
    """
    assert (
        "group" in locs.columns
    ), "Localizations must contain a 'group' column."
    # ensure groups are consecutive integers starting from 0
    locs = locs.copy()  # pandas SettingWithCopyWarning
    updated_info = deepcopy(info)
    unique_groups = np.unique(locs["group"])
    group_mapping = {old: new for new, old in enumerate(unique_groups)}
    locs["group"] = locs["group"].map(group_mapping)

    # shift localizations to the middle of the FOV and by the COM
    # of each group
    cx = get_from_metadata(updated_info, "Width", raise_error=True) / 2
    cy = get_from_metadata(updated_info, "Height", raise_error=True) / 2
    for group_id in np.unique(locs["group"]):
        mask = locs["group"] == group_id
        mean_x = locs.loc[mask, "x"].mean()
        mean_y = locs.loc[mask, "y"].mean()
        locs.loc[mask, "x"] += cx - mean_x
        locs.loc[mask, "y"] += cy - mean_y

    # unfold onto grid
    locs["x"] += np.mod(locs["group"], n_square) * spacing
    locs["y"] += np.floor(locs["group"] / n_square) * spacing

    locs["x"] -= locs["x"].mean()
    locs["y"] -= locs["y"].mean()
    locs["x"] += np.absolute(locs["x"].min())
    locs["y"] += np.absolute(locs["y"].min())

    # Update FOV and clean up
    updated_info = overwrite_metadata(
        updated_info, "Width", int(np.ceil(locs["x"].max()))
    )
    updated_info = overwrite_metadata(
        updated_info, "Height", int(np.ceil(locs["y"].max()))
    )
    return locs, updated_info


def sync_groups(locs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Sync group indices across multiple localization lists. Can be
    used, for example, for removing clustered localizations after
    the cluster centers were filtered.

    Parameters
    ----------
    locs : list of pd.DataFrame
        List of localization lists to be synced. Each must contain a
        'group' column.

    Returns
    -------
    synced_locs : list of pd.DataFrame
        List of localization lists with synced group indices.
    """
    assert all(
        "group" in loc.columns for loc in locs
    ), "All localization lists must contain a 'group' column."
    unique_groups = [np.unique(loc["group"]) for loc in locs]
    common_groups = set(unique_groups[0]).intersection(*unique_groups)
    for i in range(len(locs)):
        mask = locs[i]["group"].isin(common_groups)
        locs[i] = locs[i][mask].reset_index(drop=True)
    return locs
