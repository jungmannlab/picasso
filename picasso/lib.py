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
import importlib
import multiprocessing
import os
import sys
import warnings
from copy import deepcopy
from typing import Any, TypeAlias, Literal, TYPE_CHECKING
from collections.abc import Callable
from asyncio import Future

import numba
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields, drop_fields
from scipy import stats, optimize
from playsound3 import playsound
from tqdm import tqdm

from picasso import io

if TYPE_CHECKING:
    from PyQt6 import QtGui

# A global variable where we store all open progress and status dialogs.
# In case of an exception, we close them all,
# so that the GUI remains responsive.
_dialogs = []

# Min. time to use sound notification when ProcessDialog or
# StatusDialog is finished
SOUND_NOTIFICATION_DURATION = 60  # seconds

# Columns that are required for Picasso
REQUIRED_COLUMNS = ["frame", "x", "y", "z", "lpx", "lpy", "lpz"]

#: Shapes that a pick region can take. "Circle" and "Square" are placed
#: by a single click and share one global size; "Rectangle" is an
#: oriented band of a global width drawn along its center axis;
#: "Polygon", "Box" and "Brush" carry their own extent and have no
#: global size.
PICK_SHAPES = ("Circle", "Rectangle", "Polygon", "Square", "Box", "Brush")

#: Pick shapes whose extent is defined per pick rather than by a single
#: global size, i.e., those for which ``pick_size`` is None.
PICK_SHAPES_WITHOUT_SIZE = ("Polygon", "Box", "Brush")

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
FloatArray4D: TypeAlias = np.ndarray[
    tuple[int, int, int, int], np.dtype[np.floating[Any]]
]
SeriesOrFloatArray1D: TypeAlias = pd.Series | FloatArray1D
SeriesOrIntArray1D: TypeAlias = pd.Series | IntArray1D
BoolArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
BoolArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
Array3x3: TypeAlias = np.ndarray[
    tuple[Literal[3], Literal[3]], np.dtype[np.floating[Any]]
]


class _LazyQtModule:
    """Placeholder for a PyQt6 submodule that is imported on first
    attribute access, so that importing the surrounding module does not
    require PyQt6."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attr: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, attr)


# Qt-dependent classes and functions live in picasso.lib_qt (which
# imports PyQt6) but stay accessible as lib.<name>; they are resolved
# lazily below so that importing picasso.lib does not require PyQt6.
_QT_NAMES = (
    "Dialog",
    "UserSettingsDialog",
    "MetadataDialog",
    "ProgressDialog",
    "StatusDialog",
    "TaskProgressDialog",
    "TaskProgress",
    "Task",
    "run_task",
    "ProgressType",
    "CollapsibleHeader",
    "CollapsibleGroupBox",
    "ScrollableGroupBox",
    "LogDoubleSpinBox",
    "RangeSlider",
    "DensityContrastSlider",
    "GenericPlotWindow",
    "RemoveColumnsDialog",
    "HelpButton",
    "confirm_restore_defaults",
    "cancel_dialogs",
    "install_excepthook",
    "adjust_widget_size",
    "get_save_filename_ext_dialog",
)


def __getattr__(name: str) -> Any:
    if name in _QT_NAMES:
        from picasso import lib_qt

        value = getattr(lib_qt, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Maximum number of workers on Windows, where ``WaitForMultipleObjects``
#: cannot wait on more than 64 handles at once and Python crashes above it.
WINDOWS_MAX_WORKERS = 61


#: Default fraction of CPU cores available to rendering. Deliberately
#: lower than Localize's 0.8: localization is a one-off batch job,
#: whereas rendering runs continuously while a user interacts with the
#: GUI, often on shared analysis workstations.
RENDER_CPU_UTILIZATION_DEFAULT = 0.5
#: localizations whose precision (lpx or lpy, nm) exceeds this are not
#: rendered by the per-localization blur methods; settable per user in
#: ``settings["Render"]["max_blur_width"]`` (see docs/render.rst)
RENDER_MAX_BLUR_WIDTH_DEFAULT = 100.0
#: default of ``settings["Render"]["interaction_subsample"]``: live
#: previews of max(500,000, a tenth of the visible localizations) while
#: panning and zooming (see ``picasso.gui.render``)
RENDER_INTERACTION_SUBSAMPLE_DEFAULT = "auto"
#: GPU memory (MB) the render backend may keep resident for uploaded
#: localizations; ``settings["Render"]["gpu"]["vram_budget_mb"]``, 0 =
#: unlimited (see docs/render.rst)
RENDER_VRAM_BUDGET_MB_DEFAULT = 8192
#: ``settings["Render"]["gpu"]["enabled"]``: "auto" renders on the GPU
#: whenever one initializes, "on" additionally warns when it does not,
#: "off" never touches the GPU
RENDER_GPU_ENABLED_DEFAULT = "auto"
#: ``settings["Render"]["gpu"]["adapter"]``: "high-performance",
#: "low-power", or a substring of the adapter's name
RENDER_GPU_ADAPTER_DEFAULT = "high-performance"
#: requests with fewer localizations render on the CPU even when the
#: GPU is enabled: the GPU's fixed cost per render (~6 ms) exceeds the
#: CPU's time for them
RENDER_GPU_MIN_LOCS = 20_000
#: Leaf capacity of the quad-tree adaptive histogram (blur method
#: "quadtree"): bins split while they hold more localizations, so every
#: bin has a signal-to-noise ratio of about sqrt(capacity / 2) (10 -> 2.2;
#: Baddeley et al. 2010 used 5)
RENDER_QUADTREE_CAPACITY_DEFAULT = 10
#: Jittered triangulation (blur method "triangulation", Baddeley et al.
#: 2010): passes averaged, jitter width in units of the mean neighbor
#: distance, and the most localizations in view the GUI renders it for
#: (above, the histogram is shown: Qhull costs ~0.3 s per pass per
#: 200k rows)
RENDER_TRIANGULATION_PASSES_DEFAULT = 25
RENDER_TRIANGULATION_JITTER_DEFAULT = 1.0
RENDER_TRIANGULATION_MAX_LOCS_DEFAULT = 100_000
#: how many times a rotated 3D localization counts towards
#: ``RENDER_GPU_MIN_LOCS``: the projected z precision widens every
#: footprint, so the CPU kernels cost 15-20x more per localization
#: than in 2D, and the GPU wins from a few thousand localizations on
RENDER_ROTATED_COST_FACTOR = 20


def n_workers(
    cpu_utilization: float = 0.75,
    settings_section: str | None = None,
) -> int:
    """Number of workers (processes or threads) to use for parallel
    computation.

    Parameters
    ----------
    cpu_utilization : float, optional
        Fraction of the available CPUs to use. Default 0.75. When
        ``settings_section`` is given, this acts as the fallback for an
        invalid or missing setting.
    settings_section : str, optional
        Name of a section in the user settings file
        (``~/.picasso/settings.yaml``, also editable via
        ``File > Picasso settings`` in the GUIs) that overrides
        ``cpu_utilization``. The file is read on every call, so changes
        apply without restarting Picasso:

        .. code-block:: yaml

            Render:
              cpu_utilization: 0.5  # fraction of CPU cores, in (0, 1)
              max_workers: 4        # optional absolute cap

        ``max_workers``, when set to a positive integer, additionally
        caps the result. Rendering uses ``settings_section="Render"``
        with ``RENDER_CPU_UTILIZATION_DEFAULT`` as the fallback.

    Returns
    -------
    n_workers : int
        ``cpu_utilization`` times the CPU count, at least 1 and, on
        Windows only, at most ``WINDOWS_MAX_WORKERS``.
    """
    max_workers = None
    if settings_section is not None:
        settings = io.load_user_settings()
        try:
            value = settings[settings_section]["cpu_utilization"]
        except Exception:
            value = None
        if isinstance(value, float) and 0.0 < value < 1.0:
            cpu_utilization = value
        try:
            max_workers = settings[settings_section]["max_workers"]
        except Exception:
            max_workers = None
    n = max(1, int(cpu_utilization * multiprocessing.cpu_count()))
    if sys.platform == "win32":
        n = min(WINDOWS_MAX_WORKERS, n)
    if (
        isinstance(max_workers, int)
        and not isinstance(max_workers, bool)
        and max_workers >= 1
    ):
        n = min(n, max_workers)
    return n


def normalize_frame_bounds(frame_bounds, n_frames):
    """Normalize ``frame_bounds`` to a list of concrete, inclusive,
    0-indexed ``(lo, hi)`` segments.

    Accepts either the legacy flat form ``(min, max)`` (where either bound
    may be None for an open end) or a list of such segments. ``None``
    bounds are resolved to ``0`` / ``n_frames``. Returns None when
    ``frame_bounds`` is None (i.e., all frames are used).

    Parameters
    ----------
    frame_bounds : tuple, list of tuples, or None
        A single ``(min, max)`` tuple, a list of such tuples, or None.
    n_frames : int
        Number of frames in the movie, used to resolve open upper bounds.

    Returns
    -------
    segments : list of tuple, or None
        List of ``(lo, hi)`` inclusive 0-indexed segments, or None.
    """
    if frame_bounds is None:
        return None
    # detect the legacy flat (min, max) form: the first element is a
    # scalar or None rather than a (lo, hi) segment
    first = frame_bounds[0]
    if first is None or np.isscalar(first):
        segments = [frame_bounds]
    else:
        segments = frame_bounds
    normalized = []
    for lo, hi in segments:
        lo = 0 if lo is None else lo
        hi = n_frames if hi is None else hi
        normalized.append((lo, hi))
    return normalized


def frame_in_bounds(frame_number, frame_bounds, n_frames):
    """Whether a frame falls within any of the requested segments.

    Parameters
    ----------
    frame_number : int
        The frame to test.
    frame_bounds : tuple, list of tuples or None
        The requested frame ranges; None means all frames. Bounds are
        inclusive. See ``normalize_frame_bounds``.
    n_frames : int
        Total number of frames, used to resolve open-ended bounds.

    Returns
    -------
    in_bounds : bool
    """
    segments = normalize_frame_bounds(frame_bounds, n_frames)
    if segments is None:
        return True
    return any(lo <= frame_number <= hi for lo, hi in segments)


def bin_z_steps(
    values: np.ndarray, z_binning: int, axis: int = 0
) -> np.ndarray:
    """Average groups of ``z_binning`` consecutive z (stage) steps.

    Used by the 3D calibrations (``picasso.zfit.calibrate_z`` and
    ``picasso.spline.build_psf_template``) to merge a finely stepped bead
    scan into coarser axial bins before the calibration model is built.
    Bin ``k`` is the mean of steps ``k * z_binning`` to
    ``(k + 1) * z_binning - 1``, so a bin of the stage positions themselves
    lies at the center of its steps. Trailing steps that do not fill a
    whole bin are dropped, which keeps the bins evenly spaced.

    Parameters
    ----------
    values : np.ndarray
        Per-step values, with the steps along ``axis``: e.g. the stage
        position of each step, the mean spot width per step, or a stack of
        bead volumes.
    z_binning : int
        Number of consecutive steps per bin. 1 returns ``values``
        unchanged.
    axis : int, optional
        Axis of ``values`` that runs over the steps. Default 0.

    Returns
    -------
    binned : np.ndarray
        ``values`` with ``axis`` shortened to
        ``values.shape[axis] // z_binning``. NaN entries are ignored within
        a bin; a bin that is NaN throughout stays NaN.

    Raises
    ------
    ValueError
        If ``z_binning`` is smaller than 1 or larger than the number of
        steps.
    """
    z_binning = int(z_binning)
    values = np.asarray(values)
    if z_binning < 1:
        raise ValueError(f"z binning must be at least 1, got {z_binning}.")
    if z_binning == 1:
        return values
    n_steps = values.shape[axis]
    n_bins = n_steps // z_binning
    if n_bins < 1:
        raise ValueError(
            f"z binning ({z_binning}) is larger than the number of z steps "
            f"({n_steps})."
        )
    moved = np.moveaxis(values, axis, 0)[: n_bins * z_binning]
    grouped = moved.reshape((n_bins, z_binning) + moved.shape[1:])
    with warnings.catch_warnings():
        # an all-NaN bin (e.g. a step without localizations on each of its
        # steps) is legitimately NaN; the caller interpolates it
        warnings.simplefilter("ignore", category=RuntimeWarning)
        binned = np.nanmean(grouped, axis=1)
    # keep a floating input's precision; integer input averages to float
    if np.issubdtype(values.dtype, np.floating):
        binned = binned.astype(values.dtype, copy=False)
    return np.moveaxis(binned, 0, axis)


class OperationCanceled(Exception):
    """Raised inside a long computation when the user canceled it.

    The progress tracker of a GUI task (``lib_qt.TaskProgress``) raises it
    from ``set_value`` once Cancel was pressed, so every progress update
    of a computation doubles as a cancellation point, without the
    computation knowing about Qt. Code that must clean up (e.g. shut down
    a process pool) can catch it, clean up and re-raise.
    """


class MockProgress:
    """Class to mock a progress bar or dialog, allowing for calling
    the same methods but not displaying anything.

    Every method accepts (and ignores) whatever the ``ProgressDialog``
    interface passes, so a caller can drive progress unconditionally. See
    :func:`normalize_progress`.
    """

    def __init__(self, *args, **kwargs):
        """Accept and ignore any arguments a real progress dialog takes."""
        self.description_base = ""
        self._maximum = 0

    def init(self, *args, **kwargs):
        """Do nothing."""
        pass

    def set_value(self, *args, **kwargs):
        """Do nothing."""
        pass

    def setMaximum(self, maximum, *args, **kwargs):
        """Record the maximum, so :meth:`maximum` can report it back.

        Parameters
        ----------
        maximum : int
            The value progress runs up to.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        self._maximum = maximum

    def maximum(self):
        """The maximum last set.

        Returns
        -------
        maximum : int
        """
        return self._maximum

    def update(self, *args, **kwargs):
        """Do nothing."""
        pass

    def closeEvent(self, *args, **kwargs):
        """Do nothing."""
        pass

    def zero_progress(self, description=None, *args, **kwargs):
        """Do nothing.

        Parameters
        ----------
        description : str, optional
            Ignored; a real dialog would show it as the new phase's label.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        pass

    def close(self, *args, **kwargs):
        """Do nothing."""
        pass

    def setLabelText(self, *args, **kwargs):
        """Do nothing."""
        pass

    def play_sound_notification(self, *args, **kwargs):
        """Do nothing."""
        pass

    def get_iterator(self, start=0, end=100):
        """A plain iterator, with no progress attached.

        Parameters
        ----------
        start, end : int, optional
            First and one-past-last value. Defaults 0 and 100.

        Returns
        -------
        iterator : range
        """
        return range(start, end)


class TqdmProgress:
    """Class to absorb calls to ProgressDialog but is used to display
    tqdm progress bar instead.

    Implements the same interface as ``ProgressDialog`` (see
    ``normalize_progress``): the bar is armed lazily on the first
    ``set_value`` call, using the description and maximum declared so
    far; ``zero_progress`` closes the current bar so that the next
    ``set_value`` starts a fresh one (a new phase)."""

    def __init__(self, *args, unit="it", **kwargs):
        """Set up the (not yet armed) bar.

        Parameters
        ----------
        unit : str, optional
            Name of one iteration, shown by tqdm. Default "it".
        *args, **kwargs
            ``description`` (keyword) is used as the bar's label; anything
            else a real progress dialog takes is accepted and ignored.
        """
        self.description_base = (
            "" if "description" not in kwargs else kwargs["description"]
        )
        self.iterator = None
        self.unit = unit
        self._maximum = 0

    def init(self, *args, **kwargs):
        """Do nothing; the bar is armed on the first :meth:`set_value`."""
        pass

    def set_value(self, value, *args, **kwargs):
        """Advance the bar to ``value``, arming it on the first call.

        Parameters
        ----------
        value : int
            Cumulative progress so far.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        if self.iterator is None:
            self.iterator = tqdm(
                total=self._maximum or None,
                desc=self.description_base,
                unit=self.unit,
            )
        self.iterator.update(value - self.iterator.n)

    def setMaximum(self, maximum, *args, **kwargs):
        """Set the value progress runs up to, updating a live bar.

        Parameters
        ----------
        maximum : int
            The total the bar counts towards.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        self._maximum = maximum
        if self.iterator is not None:
            self.iterator.total = maximum
            self.iterator.refresh()

    def maximum(self):
        """The maximum last set.

        Returns
        -------
        maximum : int
        """
        return self._maximum

    def update(self, *args, **kwargs):
        """Do nothing; tqdm redraws itself."""
        pass

    def closeEvent(self, *args, **kwargs):
        """Do nothing; accepted for interface compatibility with Qt."""
        pass

    def zero_progress(self, description=None, *args, **kwargs):
        """Close the current bar so the next :meth:`set_value` starts a fresh
        one, i.e. a new phase.

        Parameters
        ----------
        description : str, optional
            Label of the new phase. None keeps the current one.
        *args, **kwargs
            Accepted and ignored, for ``ProgressDialog`` compatibility.
        """
        if description:
            self.description_base = description
        self.close()

    def close(self, *args, **kwargs):
        """Close the bar, if one is running."""
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None

    def setLabelText(self, *args, **kwargs):
        """Do nothing; the label is set through :meth:`zero_progress`."""
        pass

    def play_sound_notification(self, *args, **kwargs):
        """Do nothing; there is no console equivalent."""
        pass

    def get_iterator(self, start=0, end=100, unit="segment"):
        """Get an iterator that drives the progress bar.

        Parameters
        ----------
        start, end : int, optional
            First and one-past-last value of the iteration. Defaults 0 and
            100.
        unit : str, optional
            Name of one iteration, shown by tqdm. Default "segment".

        Returns
        -------
        iterator : tqdm.tqdm
            Iterator over ``range(start, end)``, also stored on the instance.
        """
        self.close()
        iterator = tqdm(
            range(start, end),
            desc=self.description_base,
            unit=unit,
        )
        self.iterator = iterator
        return iterator


def normalize_progress(
    progress: Any,
    description: str = "",
    unit: str = "it",
) -> Any:
    """Normalize a public ``progress``/``callback`` argument to an
    object with the ``ProgressDialog`` interface (``set_value``,
    ``setMaximum``, ``zero_progress``, ...), so that callers can drive
    progress with plain method calls instead of checking which kind of
    tracker they hold (which would also pull in Qt in headless runs):

    * ``None`` -> ``MockProgress`` (reports nothing),
    * ``"console"`` -> ``TqdmProgress`` (tqdm bar in the console),
    * anything else (e.g. a ``lib.ProgressDialog``) is returned
      unchanged, as long as it exposes the interface.

    Parameters
    ----------
    progress : None, "console", or ProgressType
        The public progress argument.
    description : str, optional
        Initial description for a newly created ``TqdmProgress``.
        Default is "".
    unit : str, optional
        tqdm unit label for a newly created ``TqdmProgress``. Default
        is "it".

    Returns
    -------
    progress : ProgressType
        Object implementing the ``ProgressDialog`` interface.
    """
    if progress is None:
        return MockProgress()
    if isinstance(progress, str):
        if progress == "console":
            return TqdmProgress(description=description, unit=unit)
        raise ValueError(
            f"Invalid progress argument: {progress!r}. Must be None, "
            "'console', or an object with the ProgressDialog interface."
        )
    if not callable(getattr(progress, "set_value", None)):
        raise TypeError(
            f"Invalid progress argument: {progress!r}. Must be None, "
            "'console', or an object with the ProgressDialog interface "
            "(set_value, setMaximum, zero_progress, ...)."
        )
    return progress


class AutoDict(collections.defaultdict):
    """A defaultdict whose auto-generated values are defaultdicts
    itself. This allows for auto-generating nested values, e.g.
    a = AutoDict()
    a['foo']['bar']['carrot'] = 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(AutoDict, *args, **kwargs)


def deprecation_warning(message: str) -> None:
    """Display a deprecation warning message.

    Parameters
    ----------
    message : str
        The deprecation warning message to be displayed.
    """
    warnings.warn(message, DeprecationWarning, stacklevel=2)


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
    the folder ``~/.picasso/notification_sounds``.

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
    """Return the path to the user sound notification folder
    (``~/.picasso/notification_sounds``)."""
    return io.notification_sounds_directory()


def open_sound_notifications_folder() -> None:
    """Open the user sound notification folder
    (``~/.picasso/notification_sounds``) in the system file browser."""
    from PyQt6 import QtCore, QtGui

    QtGui.QDesktopServices.openUrl(
        QtCore.QUrl.fromLocalFile(_sound_notification_dir())
    )


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


def _merge_filter_range(
    ranges: dict, missing: list, current: set, key, xmin: float, xmax: float
) -> None:
    """Intersect one column's [min, max] filter range into ``ranges``.

    Appends ``key`` to ``missing`` instead when it is absent from
    ``current``.
    """
    if key not in current:
        missing.append(key)
        return
    if key in ranges:
        ranges[key][0] = max(ranges[key][0], xmin)
        ranges[key][1] = min(ranges[key][1], xmax)
    else:
        ranges[key] = [xmin, xmax]


def _extract_filter_dict(
    d: dict, current: set, ranges: dict, missing: list, to_remove_all: list
) -> None:
    """Fold one filter-step metadata dict into the running totals."""
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
            _merge_filter_range(
                ranges, missing, current, key, float(value[0]), float(value[1])
            )


def _dedupe_preserve_order(items: list) -> list:
    """``items`` with duplicates removed, keeping first-seen order."""
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_filter_steps(
    info: list[dict],
    current_columns,
) -> tuple[dict[str, list[float]], list[str], list[str]]:
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
        _extract_filter_dict(d, current, ranges, missing, to_remove_all)

    to_remove = [c for c in to_remove_all if c in current]
    missing.extend(c for c in to_remove_all if c not in current)

    return ranges, to_remove, _dedupe_preserve_order(missing)


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

    Only the memory copy of the data is modified, the disk storage stays
    untouched.

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


#: Metadata keys holding the translation of the localizations from the
#: camera frame applied by ``fit_canvas`` (camera pixels); subtract it to
#: return to the camera coordinates
CANVAS_OFFSET_KEYS = ("Canvas offset x (cam. px)", "Canvas offset y (cam. px)")
#: Metadata keys holding the size of the camera image (camera pixels),
#: stored by ``fit_canvas`` once ``Width`` or ``Height`` differ from it
CAMERA_SIZE_KEYS = ("Camera Width", "Camera Height")


def fit_canvas(
    locs_list: list[pd.DataFrame], infos_list: list[list[dict]]
) -> tuple[list[pd.DataFrame], list[list[dict]], list[tuple[int, int]]]:
    """Fit the canvas (``Width`` and ``Height`` in the metadata) of one
    or more channels to their localizations, so that none is removed by
    ``ensure_sanity``, e.g., when saving.

    The canvas is recomputed from the camera frame, i.e., from the
    coordinates minus the current offset (``CANVAS_OFFSET_KEYS``, 0 if
    missing):

    - The offset becomes the smallest whole number of camera pixels
      that makes every coordinate of every channel non-negative (0 if
      none is negative). It is the same for all channels, so their
      relative position is preserved.
    - ``Width`` and ``Height`` become the camera size
      (``CAMERA_SIZE_KEYS``, or the current ``Width`` and ``Height`` if
      missing) or the extent of the localizations, whichever is
      larger, plus the offset.

    The canvas thus grows and shrinks with the localizations, but it
    never becomes smaller than the camera image, which stays at the
    offset. For example, moving a channel beyond the left edge and
    back restores the original canvas.

    The localizations and metadata are modified in place (translated
    columns are replaced with new arrays rather than modified in
    memory, so caches keyed on the array memory, e.g., the GPU
    uploads, see the change). The offset and the camera size are
    stored in the last metadata dictionary of each channel once they
    differ from the defaults.

    Parameters
    ----------
    locs_list : list of pd.DataFrames
        Localizations of each channel.
    infos_list : list of lists of dicts
        Metadata of each channel.

    Returns
    -------
    locs_list : list of pd.DataFrames
        The input localizations, translated if needed.
    infos_list : list of lists of dicts
        The input metadata with the updated canvas.
    shifts : list of tuples of ints
        Translation ``(dx, dy)`` applied to each channel in this call
        (camera pixels), e.g., to move picks along with the
        localizations. Channels that shared an offset share the shift.

    Raises
    ------
    KeyError
        If ``Width`` or ``Height`` is missing from the metadata.
    """

    def finite(locs, column):
        # NaN and inf are removed by ensure_sanity, not translated
        values = locs[column].to_numpy()
        return values[np.isfinite(values)]

    extents = [(finite(locs, "x"), finite(locs, "y")) for locs in locs_list]
    offsets = [
        [get_from_metadata(info, key, 0) for key in CANVAS_OFFSET_KEYS]
        for info in infos_list
    ]

    # the new offset, from the minimum in the camera frame
    new_offset = []
    for axis in range(2):
        camera_min = [
            ext[axis].min() - offset[axis]
            for ext, offset in zip(extents, offsets)
            if len(ext[axis])
        ]
        lowest = min(camera_min, default=0)
        new_offset.append(int(np.ceil(-lowest)) if lowest < 0 else 0)

    shifts = []
    for locs, ext, info, offset in zip(
        locs_list, extents, infos_list, offsets
    ):
        shift = tuple(int(new - old) for new, old in zip(new_offset, offset))
        for column, delta in zip(("x", "y"), shift):
            if delta:
                locs[column] = locs[column].to_numpy() + np.float32(delta)
        shifts.append(shift)

        for axis in range(2):
            _fit_canvas_axis(
                info, axis, ext[axis], shift[axis], new_offset[axis]
            )
    return locs_list, infos_list, shifts


def _fit_canvas_axis(
    info: list[dict],
    axis: int,
    values: np.ndarray,
    shift: int,
    offset: int,
) -> None:
    """Update the canvas size, camera size and offset of one channel
    along one axis in place, see ``fit_canvas``.

    Parameters
    ----------
    info : list of dicts
        Metadata of the channel.
    axis : {0, 1}
        0 for x (``Width``), 1 for y (``Height``).
    values : np.ndarray
        Finite coordinates of the channel along ``axis`` before the
        shift (camera pixels).
    shift : int
        Translation applied to the channel along ``axis``.
    offset : int
        The new canvas offset along ``axis``.
    """
    size_key = ("Width", "Height")[axis]
    camera_key = CAMERA_SIZE_KEYS[axis]
    offset_key = CANVAS_OFFSET_KEYS[axis]
    stored = get_from_metadata(info, camera_key)
    camera = stored
    if camera is None:
        camera = get_from_metadata(info, size_key, raise_error=True)
    value = camera + offset
    if len(values):
        # strictly x < Width is kept by ensure_sanity, hence + 1
        top = values.max() + np.float32(shift)
        value = max(value, int(np.floor(top)) + 1)
    # the canvas size is read from the first and from the last
    # dictionary in different places, so every occurrence is set
    for inf in info:
        if size_key in inf:
            inf[size_key] = int(value)
    if value != camera or stored is not None:
        info[-1][camera_key] = int(camera)
    if offset or get_from_metadata(info, offset_key):
        info[-1][offset_key] = offset


def translate_picks(
    picks: list, shape: str | None, dx: float, dy: float
) -> list:
    """Translate picks by ``(dx, dy)`` camera pixels, e.g., to follow
    localizations moved by ``fit_canvas``.

    Parameters
    ----------
    picks : list
        Picks in the format of the Render window (see
        ``gui.render.View._picks``).
    shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", \
            "Brush"} or None
        Shape of the picks. None is allowed only with no picks.
    dx, dy : float
        Translation in camera pixels.

    Returns
    -------
    translated : list
        New list of translated picks; sizes (e.g., brush widths) are
        kept.
    """

    def point(p):
        return (p[0] + dx, p[1] + dy)

    if not len(picks):
        return []
    if shape in ("Circle", "Square"):
        return [point(p) for p in picks]
    if shape in ("Rectangle", "Box"):
        return [(point(a), point(b)) for a, b in picks]
    if shape == "Polygon":
        return [[point(v) for v in pick] for pick in picks]
    if shape == "Brush":
        return [
            [(width, [point(p) for p in path]) for width, path in pick]
            for pick in picks
        ]
    raise ValueError(f"Unrecognized pick shape: {shape}")


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
                from PyQt6 import QtWidgets

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
    """Cumulative exponential ``a * (1 - exp(-x / t)) + c``.

    Used for binding kinetics estimation, see :func:`fit_cum_exp`.

    Parameters
    ----------
    x : FloatArray1D
        Points to evaluate at.
    a : float
        Amplitude, i.e. the plateau above ``c``.
    t : float
        Decay constant, in the units of ``x``.
    c : float
        Offset.

    Returns
    -------
    y : FloatArray1D
        The function evaluated at ``x``.
    """
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


#: Opacity of histogram bars that are filled and outlined, so that the
#: outline in the full color stands out.
OUTLINED_FILL_ALPHA = 0.55


def histogram_style(
    color: str | tuple,
    fill: bool = True,
    outline: bool = False,
    fill_alpha: float | None = None,
    line_width: float | None = None,
) -> dict:
    """Matplotlib properties of histogram bars: filled, filled with an
    outline or outlined only.

    The properties apply to the patches of ``Axes.hist`` (with
    ``histtype="stepfilled"``), ``Axes.stairs`` and ``Axes.bar``.

    Parameters
    ----------
    color : str or tuple
        Color of the bars.
    fill : bool, optional
        Whether the bars are filled. Default True.
    outline : bool, optional
        Whether the bars are outlined in ``color``; always True if
        ``fill`` is False. Default False.
    fill_alpha : float, optional
        Opacity of the fill. If None, 1 without and
        ``OUTLINED_FILL_ALPHA`` with an outline. Default None.
    line_width : float, optional
        Width of the outline of bars that are not filled, in points. If
        None, matplotlib's default line width. Default None.

    Returns
    -------
    style : dict
        ``fill``, ``facecolor``, ``edgecolor`` and ``linewidth``.
    """
    if not fill:
        if line_width is None:
            line_width = plt.rcParams["lines.linewidth"]
        return dict(
            fill=False, facecolor="none", edgecolor=color, linewidth=line_width
        )
    if fill_alpha is None:
        fill_alpha = OUTLINED_FILL_ALPHA if outline else 1.0
    return dict(
        fill=True,
        facecolor=mcolors.to_rgba(color, fill_alpha),
        edgecolor=mcolors.to_rgba(color) if outline else "none",
        linewidth=1.0 if outline else 0.0,
    )


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


def _data_min_max(data: FloatArray1D | IntArray1D) -> tuple:
    """Min/max of ``data``, ignoring NaNs for float dtypes."""
    if data.dtype.kind == "f":
        return np.nanmin(data), np.nanmax(data)
    return data.min(), data.max()


def _iqr_sample(
    data: FloatArray1D | IntArray1D, n: int, sample_size: int
) -> FloatArray1D:
    """Subsample of ``data`` to estimate the IQR from, finite values only."""
    if n > sample_size:
        rng = np.random.default_rng(0)
        sample = data[rng.choice(n, sample_size, replace=False)]
    else:
        sample = data
    if sample.dtype.kind == "f":
        sample = sample[np.isfinite(sample)]
    return sample


def calculate_optimal_bins(
    data: FloatArray1D | IntArray1D,
    max_n_bins: int | None = None,
    sample_size: int = 1_000_000,
) -> FloatArray1D:
    """Calculate the optimal bins for display, for example, in
    Picasso: Filter.

    Parameters
    ----------
    data : FloatArray1D | IntArray1D
        Data to be binned.
    max_n_bins : int | None, optional
        Maximum number of bins.
    sample_size : int, optional
        For large arrays, estimate the IQR from a random sample of this
        size instead of sorting the full array. min/max are still taken
        from the full data (cheap O(N) reductions). Set to a value >=
        ``len(data)`` to disable sampling. Default 1_000_000.

    Returns
    -------
    bins : FloatArray1D
        Bins for display.
    """
    data = np.asarray(data)  # positional indexing below; Series use labels
    n = len(data)
    if n == 0:
        return np.array([0.0, 1.0])
    data_min, data_max = _data_min_max(data)
    sample = _iqr_sample(data, n, sample_size)
    if len(sample) == 0:
        return np.array([data_min - 1.0, data_max + 1.0])
    iqr = np.subtract(*np.percentile(sample, [75, 25]))
    if iqr == 0:
        return np.array([data[0] - 1.0, data[0] + 1.0])
    bin_size = 2 * iqr * n ** (-1 / 3)
    if data.dtype.kind in ("u", "i") and bin_size < 1:
        bin_size = 1
    bin_min = data_min - bin_size / 2
    try:
        n_bins = (data_max - bin_min) / bin_size
        n_bins = int(n_bins)
    except Exception:
        n_bins = 10
    if max_n_bins and n_bins > max_n_bins:
        n_bins = max_n_bins
    bins = np.linspace(bin_min, data_max, n_bins)
    return bins


@numba.njit(parallel=True, nogil=True)
def hist2d_numba(
    x: FloatArray1D,
    y: FloatArray1D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Fast 2D histogram with uniform bin edges.

    Non-finite points are skipped. Edge values (== x_max / == y_max) are
    folded into the last bin to match the inclusive-right behavior of
    ``np.histogram2d``.

    Parameters
    ----------
    x, y : FloatArray1D
        Sample coordinates.
    x_min, x_max, y_min, y_max : float
        Outer edges of the histogram.
    nx, ny : int
        Number of bins along each axis.

    Returns
    -------
    counts : np.ndarray, shape (nx, ny), dtype int64
        Bin counts, indexed as counts[ix, iy].
    """
    n_threads = numba.get_num_threads()
    local = np.zeros((n_threads, nx, ny), dtype=np.int64)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    n = x.shape[0]
    chunk = (n + n_threads - 1) // n_threads
    for t in numba.prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n:
            end = n
        for i in range(start, end):
            xi = x[i]
            yi = y[i]
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            ix = int((xi - x_min) / dx)
            iy = int((yi - y_min) / dy)
            if ix == nx:
                ix -= 1
            if iy == ny:
                iy -= 1
            if 0 <= ix < nx and 0 <= iy < ny:
                local[t, ix, iy] += 1
    return local.sum(axis=0)


def select_frc_rois(
    locs: pd.DataFrame,
    info: list[dict],
    viewport: tuple[tuple[float, float], tuple[float, float]],
    *,
    n_rois: int = 30,
    roi_size: float = 5000.0,
    min_locs: int = 1000,
    random_seed: int = 42,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Randomly place non-overlapping square ROIs, e.g., for FRC in
    several ROIs (``postprocess.frc_rois``).

    ROIs are placed on a grid with a step of a tenth of the ROI side
    length inside ``viewport``, in random order; a candidate is kept if
    it holds at least ``min_locs`` localizations and does not overlap
    an already kept ROI. Placement stops once ``n_rois`` ROIs are kept
    or no candidates are left, so fewer ROIs may be returned.

    Parameters
    ----------
    locs : pd.DataFrame
        Localization list.
    info : list of dicts
        Metadata of the localizations list.
    viewport : tuple of floats
        Region ((y_min, x_min), (y_max, x_max)) in camera pixels in
        which the ROIs are placed.
    n_rois : int, optional
        Maximum number of ROIs. Default is 30.
    roi_size : float, optional
        Side length of the square ROIs in nm. Default is 5000.
    min_locs : int, optional
        Minimum number of localizations per ROI. Default is 1000.
    random_seed : int, optional
        Random seed for the ROI placement. Default is 42.

    Returns
    -------
    rois : list of tuples
        Viewports ((y_min, x_min), (y_max, x_max)) of the selected ROIs
        in camera pixels.

    Raises
    ------
    ValueError
        If ``viewport`` is smaller than one ROI.
    """
    pixelsize = get_from_metadata(info, "Pixelsize", raise_error=True)
    (y_min, x_min), (y_max, x_max) = viewport
    n_sub = 10  # grid steps per ROI side
    step = roi_size / pixelsize / n_sub
    ny = int((y_max - y_min) / step)
    nx = int((x_max - x_min) / step)
    if ny < n_sub or nx < n_sub:
        raise ValueError(
            f"The viewport is smaller than one ROI of {roi_size:.0f} nm."
        )
    # localization counts per grid cell and their integral image, so
    # that the count of any grid-aligned ROI is exact and O(1)
    counts, _, _ = np.histogram2d(
        locs["y"].to_numpy(),
        locs["x"].to_numpy(),
        bins=(ny, nx),
        range=((y_min, y_min + ny * step), (x_min, x_min + nx * step)),
    )
    integral = np.zeros((ny + 1, nx + 1))
    integral[1:, 1:] = counts.cumsum(0).cumsum(1)
    roi_counts = (
        integral[n_sub:, n_sub:]
        - integral[:-n_sub, n_sub:]
        - integral[n_sub:, :-n_sub]
        + integral[:-n_sub, :-n_sub]
    )
    candidates = np.argwhere(roi_counts >= min_locs)
    rng = np.random.default_rng(random_seed)
    candidates = candidates[rng.permutation(len(candidates))]

    occupied = np.zeros((ny, nx), dtype=bool)
    rois = []
    for i, j in candidates:
        if len(rois) == n_rois:
            break
        if occupied[i : i + n_sub, j : j + n_sub].any():
            continue
        occupied[i : i + n_sub, j : j + n_sub] = True
        y0 = y_min + i * step
        x0 = x_min + j * step
        rois.append(((y0, x0), (y0 + n_sub * step, x0 + n_sub * step)))
    return rois


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


def net_gradient_column(identifications: pd.DataFrame) -> dict:
    """The ``net_gradient`` column of identifications, to be copied into
    the localizations fitted from them, or nothing: spots identified by
    wavelet segmentation (see ``picasso.wavelet``) have no net gradient.

    Parameters
    ----------
    identifications : pd.DataFrame
        Identifications the localizations are fitted from.

    Returns
    -------
    dict
        ``{"net_gradient": values}`` (float32) if the column exists,
        otherwise empty.
    """
    if "net_gradient" not in identifications.columns:
        return {}
    return {
        "net_gradient": np.asarray(
            identifications["net_gradient"], dtype=np.float32
        )
    }


def concat_locs(locs_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate localization tables into one with a new index.

    The tables need not have the same columns: which ones a table has
    depends on how it was made, e.g. spots identified by wavelet
    segmentation (see ``picasso.wavelet``) have no ``net_gradient``,
    spherical Gaussian fits no ``ellipticity``, least-squares fits a
    ``chi_square`` instead of the ``log_likelihood`` of maximum likelihood
    ones, and 2D data no ``z``. A column that not every table has is
    dropped, with a warning: ``pd.concat`` would fill it with NaN for the
    tables without it, and ``ensure_sanity`` would then delete all of
    their localizations when the result is saved.

    Tables without localizations are left out, so their columns neither
    remove nor add any.

    Parameters
    ----------
    locs_list : list of pd.DataFrame
        Localization tables.

    Returns
    -------
    locs : pd.DataFrame
        All localizations, with the columns of the first table that every
        (non-empty) table has.
    """
    locs_list = list(locs_list)
    filled = [locs for locs in locs_list if len(locs)]
    if not filled:
        return pd.concat(locs_list, ignore_index=True)
    common = set(filled[0].columns).intersection(
        *(locs.columns for locs in filled[1:])
    )
    partial = list(
        dict.fromkeys(
            column
            for locs in filled
            for column in locs.columns
            if column not in common
        )
    )
    if partial:
        warnings.warn(
            f"Column(s) {', '.join(map(str, partial))} are missing from some "
            "of the localizations being combined and were dropped, so that "
            "no localizations are lost."
        )
        filled = [
            locs.drop(columns=partial, errors="ignore") for locs in filled
        ]
    return pd.concat(filled, ignore_index=True)


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
    locs = concat_locs(locs_list)
    locs.sort_values(by="frame", inplace=True)
    return locs


def append_group(
    locs: pd.DataFrame,
    group: int | SeriesOrIntArray1D,
) -> pd.DataFrame:
    """Assign the ``group`` column of ``locs``, preserving any existing
    grouping. If ``group`` already exists, its contents are moved to
    ``group_input`` to preserve information.

    The DataFrame is modified in place and returned for convenience.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to assign the group column to.
    group : int or array-like of int
        New group id(s). A scalar is broadcast to all localizations; an
        array-like must have the same length as ``locs`` and is assigned
        positionally.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with the ``group`` column set. If a ``group``
        column was already present, its previous values are stored in
        the ``"group_input"`` column.
    """
    if "group" in locs.columns:
        locs["group_input"] = locs["group"].to_numpy()
    locs["group"] = group if np.isscalar(group) else np.asarray(group)
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
    return standardize_dtypes(locs)


def standardize_dtypes(locs: pd.DataFrame) -> pd.DataFrame:
    """Return ``locs`` with float64 columns cast to float32 and an
    integer ``frame`` column cast to uint32 (the dtypes Localize
    writes), leaving every other column as it is.

    float32 resolves 2e-4 camera pixels at 4096 pixels and far below a
    nanometer in z, two orders below any localization precision, while
    halving memory and file size for columns that pandas arithmetic or
    imports from other software promoted to float64, and sparing the
    GPU renderer a converted copy of such columns on every render.
    Called by ``ensure_sanity``, i.e., on loading, saving and in most
    processing functions.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.

    Returns
    -------
    locs : pd.DataFrame
        The same DataFrame if nothing had to change, else a copy.
    """
    casts = {
        name: np.float32
        for name, dtype in locs.dtypes.items()
        if dtype == np.float64
    }
    if "frame" in locs.columns:
        frame_dtype = locs["frame"].dtype
        if np.issubdtype(frame_dtype, np.integer) and frame_dtype != np.uint32:
            casts["frame"] = np.uint32
    if not casts:
        return locs
    return locs.astype(casts, copy=False)


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


@numba.jit(nopython=True, nogil=True, cache=True)
def is_loc_at_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    r: float,
) -> BoolArray1D:
    """Numba implementation of ``is_loc_at``.

    Parameters
    ----------
    x, y : float
        Center of the circular pick.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.
    r : float
        Pick radius.

    Returns
    -------
    is_picked : BoolArray1D
        True where a localization lies within ``r`` of ``(x, y)``.
    """
    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return is_picked


@numba.jit(nopython=True, nogil=True, cache=True)
def locs_at_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    r: float,
) -> FloatArray2D:
    """Numba implementation of ``locs_at``.

    Parameters
    ----------
    x, y : float
        Center of the circular pick.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.
    r : float
        Pick radius.

    Returns
    -------
    picked_xy : FloatArray2D
        ``(2, n_picked)`` coordinates within ``r`` of ``(x, y)``.
    """
    is_picked = is_loc_at_numba(x, y, locs_xy, r)
    return locs_xy[:, is_picked]


@numba.jit(nopython=True, nogil=True)
def rmsd_at_com(locs_xy: FloatArray2D) -> float:
    """RMS distance of the localizations from their center of mass.

    Parameters
    ----------
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.

    Returns
    -------
    rmsd : float
        Root mean square distance from the center of mass.
    """
    com_x = np.mean(locs_xy[0])
    com_y = np.mean(locs_xy[1])
    return np.sqrt(
        np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2)
    )


@numba.jit(nopython=True, nogil=True, cache=True)
def is_loc_in_square_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    a: float,
) -> BoolArray1D:
    """Check which localizations are within the axis-aligned square of
    side length ``a`` centered at ``(x, y)``.

    Square analogue of ``is_loc_at_numba``. The bounds are exclusive,
    matching ``postprocess._picked_square_locs``.

    Parameters
    ----------
    x, y : float
        Center of the square.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.
    a : float
        Side length of the square.

    Returns
    -------
    is_picked : BoolArray1D
        True if a localization is within the square.
    """
    half_a = 0.5 * a
    dx = np.abs(locs_xy[0] - x)
    dy = np.abs(locs_xy[1] - y)
    is_picked = (dx < half_a) & (dy < half_a)
    return is_picked


@numba.jit(nopython=True, nogil=True, cache=True)
def locs_in_square_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    a: float,
) -> FloatArray2D:
    """Return localizations within the axis-aligned square of side
    length ``a`` centered at ``(x, y)``.

    Parameters
    ----------
    x, y : float
        Center of the square.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.
    a : float
        Side length of the square.

    Returns
    -------
    picked_xy : FloatArray2D
        ``(2, n_picked)`` coordinates inside the square. See
        ``is_loc_in_square_numba``.
    """
    is_picked = is_loc_in_square_numba(x, y, locs_xy, a)
    return locs_xy[:, is_picked]


@numba.jit(nopython=True, nogil=True, cache=True)
def is_loc_in_box_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    w: float,
    h: float,
) -> BoolArray1D:
    """Check which localizations are within the axis-aligned box of
    width ``w`` and height ``h`` centered at ``(x, y)``.

    Generalization of ``is_loc_in_square_numba`` to independent side
    lengths. The bounds are exclusive, matching
    ``postprocess._picked_box_locs``.

    Parameters
    ----------
    x, y : float
        Center of the box.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.
    w, h : float
        Width (x) and height (y) of the box.

    Returns
    -------
    is_picked : BoolArray1D
        True if a localization is within the box.
    """
    dx = np.abs(locs_xy[0] - x)
    dy = np.abs(locs_xy[1] - y)
    is_picked = (dx < 0.5 * w) & (dy < 0.5 * h)
    return is_picked


@numba.jit(nopython=True, nogil=True, cache=True)
def locs_in_box_numba(
    x: float,
    y: float,
    locs_xy: FloatArray2D,
    w: float,
    h: float,
) -> FloatArray2D:
    """Return localizations within the axis-aligned box of width ``w``
    and height ``h`` centered at ``(x, y)``.

    Parameters
    ----------
    x, y : float
        Center of the box.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.
    w, h : float
        Width (x) and height (y) of the box.

    Returns
    -------
    picked_xy : FloatArray2D
        ``(2, n_picked)`` coordinates inside the box. See
        ``is_loc_in_box_numba``.
    """
    is_picked = is_loc_in_box_numba(x, y, locs_xy, w, h)
    return locs_xy[:, is_picked]


@numba.jit(nopython=True, nogil=True, cache=True)
def is_loc_in_rectangle_numba(
    xc: float,
    yc: float,
    theta: float,
    length: float,
    width: float,
    locs_xy: FloatArray2D,
) -> BoolArray1D:
    """Check which localizations are within an oriented rectangle.

    The rectangle is centered at ``(xc, yc)``, its long (center) axis
    points along ``theta`` and has the given ``length``, and it extends
    by ``width`` perpendicular to that axis.

    Unlike ``check_if_in_rectangle``, this rotates the localizations
    into the rectangle's frame instead of ray casting, so it does not
    allocate intermediate arrays per side and has no division by zero
    for axis-aligned rectangles. It is the variant used in the picking
    kernels.

    Parameters
    ----------
    xc, yc : float
        Center of the rectangle.
    theta : float
        Angle of the center axis (radians).
    length : float
        Length of the rectangle along ``theta``.
    width : float
        Width of the rectangle perpendicular to ``theta``.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.

    Returns
    -------
    is_picked : BoolArray1D
        True if a localization is within the rectangle.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    dx = locs_xy[0] - xc
    dy = locs_xy[1] - yc
    u = dx * ct + dy * st  # along the center axis
    v = -dx * st + dy * ct  # perpendicular to the center axis
    half_l = 0.5 * length
    half_w = 0.5 * width
    is_picked = (np.abs(u) < half_l) & (np.abs(v) < half_w)
    return is_picked


@numba.jit(nopython=True, nogil=True, cache=True)
def locs_in_rectangle_numba(
    xc: float,
    yc: float,
    theta: float,
    length: float,
    width: float,
    locs_xy: FloatArray2D,
) -> FloatArray2D:
    """Return localizations within an oriented rectangle.

    Parameters
    ----------
    xc, yc : float
        Center of the rectangle.
    theta : float
        Orientation of its center axis, in radians.
    length : float
        Extent along the center axis.
    width : float
        Extent perpendicular to it.
    locs_xy : FloatArray2D
        Localization coordinates, shape ``(2, N)``.

    Returns
    -------
    picked_xy : FloatArray2D
        ``(2, n_picked)`` coordinates inside the rectangle. See
        ``is_loc_in_rectangle_numba``.
    """
    is_picked = is_loc_in_rectangle_numba(
        xc, yc, theta, length, width, locs_xy
    )
    return locs_xy[:, is_picked]


@numba.jit(nopython=True, nogil=True, cache=True)
def wrap_angle_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi/2, pi/2)``.

    The orientation of a rectangle is a director, i.e., it is only
    defined modulo pi (``theta`` and ``theta + pi`` give the same
    rectangle). Wrapping is required whenever two orientations are
    compared, otherwise, e.g., +89 deg and -89 deg appear to differ by
    178 deg instead of 2 deg.

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    angle : float
        The equivalent angle in ``[-pi/2, pi/2)``.
    """
    return angle - np.pi * np.floor(angle / np.pi + 0.5)


@numba.jit(nopython=True, nogil=True, cache=True)
def principal_axis(
    sxx: float,
    sxy: float,
    syy: float,
) -> tuple[float, float, float]:
    """Find the principal axis and the anisotropic RMSDs from the
    central second moments of a set of 2D points.

    Closed-form eigendecomposition of the 2x2 covariance matrix
    ``[[sxx, sxy], [sxy, syy]]``. Note that
    ``rmsd_along ** 2 + rmsd_across ** 2`` equals the squared isotropic
    RMSD at the center of mass (``rmsd_at_com``).

    Parameters
    ----------
    sxx, sxy, syy : float
        Central second moments, i.e., ``mean((x - mx) ** 2)``,
        ``mean((x - mx) * (y - my))`` and ``mean((y - my) ** 2)``.

    Returns
    -------
    theta : float
        Angle of the principal (major) axis, wrapped to
        ``[-pi/2, pi/2)``. For an isotropic point cloud the axis is
        undefined and 0.0 is returned; callers can detect this from
        ``rmsd_along ** 2 - rmsd_across ** 2`` being (nearly) zero.
    rmsd_along, rmsd_across : float
        RMSD along the principal axis and perpendicular to it.
    """
    delta = np.sqrt((sxx - syy) ** 2 + 4 * sxy**2)
    lambda_plus = 0.5 * (sxx + syy + delta)
    lambda_minus = 0.5 * (sxx + syy - delta)
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    rmsd_along = np.sqrt(max(lambda_plus, 0.0))
    rmsd_across = np.sqrt(max(lambda_minus, 0.0))
    return wrap_angle_pi(theta), rmsd_along, rmsd_across


@numba.jit(nopython=True, nogil=True, cache=True)
def rectangles_overlap(  # noqa: C901
    x1: float,
    y1: float,
    theta1: float,
    length1: float,
    width1: float,
    r1: float,
    x2: float,
    y2: float,
    theta2: float,
    length2: float,
    width2: float,
    r2: float,
) -> bool:
    """Check if two oriented rectangles overlap.

    Uses the separating axis theorem: two convex polygons are disjoint
    if and only if their projections onto one of the edge normals do not
    overlap. For rectangles there are only four candidate axes (two per
    rectangle). Two cheap tests based on the circumscribed and inscribed
    circles are performed first.

    Parameters
    ----------
    x1, y1 : float
        Center of the first rectangle.
    theta1 : float
        Angle of the first rectangle's center axis (radians).
    length1 : float
        Length of the first rectangle along its center axis.
    width1 : float
        Width of the first rectangle.
    r1 : float
        Circumscribed circle radius of the first rectangle, i.e.,
        ``sqrt(length1 ** 2 + width1 ** 2) / 2``. Passed in because it
        is usually precomputed.
    x2, y2 : float
        Center of the second rectangle.
    theta2 : float
        Angle of the second rectangle's center axis (radians).
    length2 : float
        Length of the second rectangle along its center axis.
    width2 : float
        Width of the second rectangle.
    r2 : float
        Circumscribed circle radius of the second rectangle, see
        ``r1``.

    Returns
    -------
    overlap : bool
        True if the two rectangles overlap.
    """
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx**2 + dy**2
    # cheap reject: circumscribed circles do not touch
    if d2 > (r1 + r2) ** 2:
        return False
    # cheap accept: inscribed circles overlap
    r_in = 0.5 * min(length1, width1) + 0.5 * min(length2, width2)
    if d2 < r_in**2:
        return True
    # separating axis theorem
    ct1 = np.cos(theta1)
    st1 = np.sin(theta1)
    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)
    # only two distinct dot products exist between the two frames
    c = abs(ct1 * ct2 + st1 * st2)
    s = abs(ct1 * st2 - st1 * ct2)
    half_l1 = 0.5 * length1
    half_w1 = 0.5 * width1
    half_l2 = 0.5 * length2
    half_w2 = 0.5 * width2
    # axes of the first rectangle
    if abs(dx * ct1 + dy * st1) > half_l1 + half_l2 * c + half_w2 * s:
        return False
    if abs(-dx * st1 + dy * ct1) > half_w1 + half_l2 * s + half_w2 * c:
        return False
    # axes of the second rectangle
    if abs(dx * ct2 + dy * st2) > half_l2 + half_l1 * c + half_w1 * s:
        return False
    if abs(-dx * st2 + dy * ct2) > half_w2 + half_l1 * s + half_w1 * c:
        return False
    return True


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


@numba.jit(nopython=True, nogil=True, cache=True)
def _point_segment_distance_sq(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Return the squared distance from the point ``(px, py)`` to the
    segment ``(ax, ay)-(bx, by)``.

    The primitive underlying every brush stroke test. A degenerate
    segment (both ends equal) reduces to the point-to-point distance.

    Parameters
    ----------
    px, py : float
        The point.
    ax, ay : float
        One end of the segment.
    bx, by : float
        The other end of the segment.

    Returns
    -------
    distance_sq : float
        Squared distance from the point to the segment.
    """
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:  # degenerate segment, i.e., a single point
        t = 0.0
    else:
        # projection of the point onto the segment, clamped to its ends
        t = ((px - ax) * dx + (py - ay) * dy) / length_sq
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    ex = px - (ax + t * dx)
    ey = py - (ay + t * dy)
    return ex * ex + ey * ey


@numba.jit(nopython=True, nogil=True, cache=True)
def _segment_segment_distance_sq(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx_: float,
    dy_: float,
) -> float:
    """Return the squared distance between the segments
    ``(ax, ay)-(bx, by)`` and ``(cx, cy)-(dx_, dy_)``.

    Zero if the segments intersect; otherwise the smallest of the four
    endpoint-to-segment distances, since the closest pair of points on
    two disjoint segments always involves at least one endpoint.

    Parameters
    ----------
    ax, ay, bx, by : float
        Ends of the first segment.
    cx, cy, dx_, dy_ : float
        Ends of the second segment.

    Returns
    -------
    distance_sq : float
        Squared distance between the two segments.
    """
    # orientation of each endpoint with respect to the other segment
    r_x = bx - ax
    r_y = by - ay
    s_x = dx_ - cx
    s_y = dy_ - cy
    denom = r_x * s_y - r_y * s_x
    if denom != 0.0:
        t = ((cx - ax) * s_y - (cy - ay) * s_x) / denom
        u = ((cx - ax) * r_y - (cy - ay) * r_x) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0  # the segments cross
    best = _point_segment_distance_sq(ax, ay, cx, cy, dx_, dy_)
    d = _point_segment_distance_sq(bx, by, cx, cy, dx_, dy_)
    if d < best:
        best = d
    d = _point_segment_distance_sq(cx, cy, ax, ay, bx, by)
    if d < best:
        best = d
    d = _point_segment_distance_sq(dx_, dy_, ax, ay, bx, by)
    if d < best:
        best = d
    return best


@numba.jit(nopython=True, nogil=True, cache=True)
def check_if_in_brush_stroke(
    x: FloatArray1D,
    y: FloatArray1D,
    X: FloatArray1D,
    Y: FloatArray1D,
    r: float,
) -> BoolArray1D:
    """Check which points ``(x, y)`` lie within distance ``r`` of the
    polyline ``(X, Y)``.

    This is the region a round-capped, round-joined pen of width ``2r``
    paints when swept along the polyline, so the test matches the drawn
    stroke exactly. A one-point polyline is a disk of radius ``r``.

    Parameters
    ----------
    x, y : FloatArray1D
        x and y coordinates of the points to test.
    X, Y : FloatArray1D
        x and y coordinates of the polyline vertices, in drawing order.
    r : float
        Half the stroke width.

    Returns
    -------
    is_in_stroke : BoolArray1D
        Boolean array indicating which points are within ``r`` of the
        polyline.
    """
    n_points = len(x)
    n_vertices = len(X)
    r2 = r * r
    is_in_stroke = np.zeros(n_points, dtype=np.bool_)
    for i in range(n_points):
        if n_vertices == 1:  # a dot rather than a swept path
            dx = x[i] - X[0]
            dy = y[i] - Y[0]
            is_in_stroke[i] = dx * dx + dy * dy <= r2
            continue
        for j in range(n_vertices - 1):
            if (
                _point_segment_distance_sq(
                    x[i], y[i], X[j], Y[j], X[j + 1], Y[j + 1]
                )
                <= r2
            ):
                is_in_stroke[i] = True
                break
    return is_in_stroke


@numba.jit(nopython=True, nogil=True, cache=True)
def brush_strokes_overlap(
    X1: FloatArray1D,
    Y1: FloatArray1D,
    X2: FloatArray1D,
    Y2: FloatArray1D,
    max_dist: float,
) -> bool:
    """Check whether two brush strokes paint overlapping regions.

    The painted regions touch exactly when the two polylines come within
    ``max_dist`` of each other, i.e., within the sum of their radii.

    Parameters
    ----------
    X1, Y1 : FloatArray1D
        Vertices of the first polyline.
    X2, Y2 : FloatArray1D
        Vertices of the second polyline.
    max_dist : float
        Sum of the two strokes' radii, i.e., half the sum of their
        widths.

    Returns
    -------
    overlap : bool
        True if the painted regions overlap.
    """
    # bounding boxes expanded by max_dist cannot miss an overlap
    if (
        np.min(X1) - max_dist > np.max(X2)
        or np.max(X1) + max_dist < np.min(X2)
        or np.min(Y1) - max_dist > np.max(Y2)
        or np.max(Y1) + max_dist < np.min(Y2)
    ):
        return False

    max_dist_sq = max_dist * max_dist
    n1 = len(X1)
    n2 = len(X2)
    for i in range(max(n1 - 1, 1)):
        # a one-vertex polyline is treated as a degenerate segment
        i_next = i + 1 if n1 > 1 else i
        for j in range(max(n2 - 1, 1)):
            j_next = j + 1 if n2 > 1 else j
            if (
                _segment_segment_distance_sq(
                    X1[i],
                    Y1[i],
                    X1[i_next],
                    Y1[i_next],
                    X2[j],
                    Y2[j],
                    X2[j_next],
                    Y2[j_next],
                )
                <= max_dist_sq
            ):
                return True
    return False


def brush_stroke_arrays(
    stroke: tuple[float, list[tuple[float, float]]],
) -> tuple[float, FloatArray1D, FloatArray1D]:
    """Split a brush stroke into its width and its path arrays.

    A brush stroke is stored as ``(width, path)``, where ``path`` is the
    list of ``(x, y)`` positions the cursor swept, in camera pixels.
    Loading a picks file yields lists rather than tuples, so the stroke
    is indexed rather than unpacked.

    Parameters
    ----------
    stroke : tuple
        One brush stroke, ``(width, path)``.

    Returns
    -------
    width : float
        Width of the stroke in camera pixels.
    X, Y : FloatArray1D
        x and y coordinates of the path.
    """
    path = np.asarray(stroke[1], dtype=np.float64)
    return float(stroke[0]), path[:, 0], path[:, 1]


def locs_in_brush(
    locs: pd.DataFrame,
    pick: list[tuple[float, list[tuple[float, float]]]],
) -> pd.DataFrame:
    """Return localizations painted by any stroke of a brush pick.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    pick : list of tuples
        One brush pick, i.e., a list of ``(width, path)`` strokes.

    Returns
    -------
    picked_locs : pd.DataFrame
        Localizations within the painted region.
    """
    if not len(pick):
        return locs.iloc[:0]
    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()
    is_picked = np.zeros(len(locs), dtype=np.bool_)
    for stroke in pick:
        width, X, Y = brush_stroke_arrays(stroke)
        is_picked |= check_if_in_brush_stroke(x, y, X, Y, width / 2)
    return locs[is_picked]


def merge_brush_strokes(
    strokes: list[tuple[float, list[tuple[float, float]]]],
) -> list[list[tuple[float, list[tuple[float, float]]]]]:
    """Group brush strokes into picks, merging those that overlap.

    Strokes whose painted regions touch belong to the same region of
    interest, and merging is transitive: a stroke bridging two picks
    joins all three into one. Picks are returned in the order of their
    newest stroke, so the stroke drawn last always ends up as
    ``picks[-1][-1]`` - which is what makes "remove the last stroke"
    well defined after adding, undoing or loading alike.

    Parameters
    ----------
    strokes : list of tuples
        Brush strokes, ``(width, path)``, in drawing order.

    Returns
    -------
    picks : list of lists
        Brush picks, each a list of strokes in drawing order.
    """
    picks = []
    pick_arrays = []  # (width, X, Y) per stroke, cached per pick
    for stroke in strokes:
        width, X, Y = brush_stroke_arrays(stroke)
        merged_strokes = []
        merged_arrays = []
        kept_picks = []
        kept_arrays = []
        for pick, arrays in zip(picks, pick_arrays):
            overlaps = False
            for width_, X_, Y_ in arrays:
                if brush_strokes_overlap(X, Y, X_, Y_, (width + width_) / 2):
                    overlaps = True
                    break
            if overlaps:
                merged_strokes.extend(pick)
                merged_arrays.extend(arrays)
            else:
                kept_picks.append(pick)
                kept_arrays.append(arrays)
        # the new stroke goes last, so it stays the newest one
        merged_strokes.append(stroke)
        merged_arrays.append((width, X, Y))
        kept_picks.append(merged_strokes)
        kept_arrays.append(merged_arrays)
        picks = kept_picks
        pick_arrays = kept_arrays
    return picks


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
    """Return the number of finished futures, used in multiprocessing.

    Parameters
    ----------
    futures : list of Future
        The futures to poll.

    Returns
    -------
    n_done : int
        How many of them have completed.
    """
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


def get_pick_box_corners(
    pick: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Find the positions of corners of a box pick.

    A box pick is defined by two opposite corners:
        ``((x0, y0), (x1, y1))``
    (all values in camera pixels). The corners are returned in the same
    order as ``get_pick_rectangle_corners``, i.e., counter-clockwise
    starting from the corner with the smaller x and y.

    Parameters
    ----------
    pick : tuple of tuples
        The two opposite corners of the box.

    Returns
    -------
    corners : tuple
        Contains corners' x and y coordinates in two lists.
    """
    (x0, y0), (x1, y1) = pick
    x_min, x_max = (x0, x1) if x0 <= x1 else (x1, x0)
    y_min, y_max = (y0, y1) if y0 <= y1 else (y1, y0)
    corners = (
        [x_min, x_max, x_max, x_min],
        [y_min, y_min, y_max, y_max],
    )
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


def _pick_areas_box(
    picks: list[tuple[tuple[float, float], tuple[float, float]]],
) -> FloatArray1D:
    """Return pick areas for each box pick in picks.

    Parameters
    ----------
    picks : list of tuples
        List of picks, each pick holds the two opposite corners of the
        box.

    Returns
    -------
    areas : FloatArray1D
        Pick areas.
    """
    areas = np.zeros(len(picks))
    for i, ((x0, y0), (x1, y1)) in enumerate(picks):
        areas[i] = abs(x1 - x0) * abs(y1 - y0)
    return areas


def _pick_areas_brush(
    picks: list[list[tuple[float, list[tuple[float, float]]]]],
    max_cells: int = 1_000_000,
) -> FloatArray1D:
    """Return pick areas for each brush pick in picks.

    The painted region is a union of round-capped strokes that overlap
    by construction - merging two picks means their strokes touch - so
    the analytic ``2 r L + pi r**2`` of a single stroke would
    double-count. The region is rasterized instead, as ``masking`` does
    for image-based regions.

    Parameters
    ----------
    picks : list of lists
        Brush picks, each a list of ``(width, path)`` strokes.
    max_cells : int, optional
        Upper bound on the number of grid cells per pick, which caps the
        cost for a large region drawn with a thin brush. Default is
        1,000,000.

    Returns
    -------
    areas : FloatArray1D
        Pick areas, accurate to about 1%.
    """
    areas = np.zeros(len(picks))
    for i, pick in enumerate(picks):
        if not len(pick):
            continue
        x_min, x_max, y_min, y_max = _brush_bounds(pick)
        r_min = min(float(stroke[0]) for stroke in pick) / 2
        # resolve the thinnest stroke, but keep the grid bounded
        step = max(
            r_min / 4,
            np.sqrt((x_max - x_min) * (y_max - y_min) / max_cells),
        )
        grid_x = np.arange(x_min + step / 2, x_max, step)
        grid_y = np.arange(y_min + step / 2, y_max, step)
        if not len(grid_x) or not len(grid_y):
            continue
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
        mesh_x = mesh_x.ravel()
        mesh_y = mesh_y.ravel()
        is_inside = np.zeros(len(mesh_x), dtype=np.bool_)
        for stroke in pick:
            width, X, Y = brush_stroke_arrays(stroke)
            is_inside |= check_if_in_brush_stroke(
                mesh_x, mesh_y, X, Y, width / 2
            )
        areas[i] = is_inside.sum() * step**2
    return areas


def _brush_bounds(
    pick: list[tuple[float, list[tuple[float, float]]]],
) -> tuple[float, float, float, float]:
    """Return the bounding box of a brush pick, i.e., the union of its
    strokes' paths, each expanded by its own radius.

    Parameters
    ----------
    pick : list of tuples
        One brush pick, i.e., a list of ``(width, path)`` strokes.

    Returns
    -------
    bounds : tuple
        ``(x_min, x_max, y_min, y_max)`` in camera pixels.

    Raises
    ------
    ValueError
        If the pick holds no strokes.
    """
    if not len(pick):
        raise ValueError("Cannot bound an empty brush pick.")
    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    for stroke in pick:
        width, X, Y = brush_stroke_arrays(stroke)
        r = width / 2
        x_min = min(x_min, X.min() - r)
        x_max = max(x_max, X.max() + r)
        y_min = min(y_min, Y.min() - r)
        y_max = max(y_max, Y.max() + r)
    return float(x_min), float(x_max), float(y_min), float(y_max)


def pick_areas(
    picks: list[tuple],
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None,
) -> FloatArray1D:
    """Get pick areas for each pick in picks.

    Parameters
    ----------
    picks : list of tuples
        Coordinates of picks in camera pixels.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"}
        Shape of picks.
    pick_size : float or None
        Size of picks in camera pixels. For circles - diameters. For
        rectangles - width. For squares - side length. For polygons,
        boxes and brush picks - ignored.

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
    elif pick_shape == "Box":
        areas = _pick_areas_box(picks)
    elif pick_shape == "Brush":
        areas = _pick_areas_brush(picks)
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")
    return areas


def pick_bounds(
    pick: tuple,
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None,
) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box of a single pick.

    Used wherever a pick has to be framed rather than tested against,
    e.g., to build a viewport around it or to set plot limits.

    Parameters
    ----------
    pick : tuple
        Coordinates of one pick in camera pixels. Must match the format
        of ``pick_shape``.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"}
        Shape of the pick.
    pick_size : float or None
        Size of the pick in camera pixels. For circles - diameter. For
        rectangles - width. For squares - side length. For polygons,
        boxes and brush picks - ignored.

    Returns
    -------
    bounds : tuple
        ``(x_min, x_max, y_min, y_max)`` in camera pixels. For an open
        polygon (fewer than three vertices or not closed), the bounding
        box of its vertices is returned; an empty polygon or brush pick
        raises ``ValueError``.
    """
    if pick_shape in ("Circle", "Square"):
        # a circle of diameter d and a square of side d have the same
        # bounding box
        half = pick_size / 2
        x, y = pick
        return x - half, x + half, y - half, y + half
    elif pick_shape == "Rectangle":
        (xs, ys), (xe, ye) = pick
        X, Y = get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
    elif pick_shape == "Polygon":
        X, Y = get_pick_polygon_corners(pick)
        if X is None:  # not closed yet, fall back to the drawn vertices
            if not len(pick):
                raise ValueError("Cannot bound an empty polygon pick.")
            X = [_[0] for _ in pick]
            Y = [_[1] for _ in pick]
    elif pick_shape == "Box":
        X, Y = get_pick_box_corners(pick)
    elif pick_shape == "Brush":
        # each stroke reaches half its own width beyond its path
        return _brush_bounds(pick)
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")
    return min(X), max(X), min(Y), max(Y)


def point_in_pick(
    x: float,
    y: float,
    pick: tuple,
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    pick_size: float | None,
) -> bool:
    """Check whether the point ``(x, y)`` lies inside a single pick.

    Parameters
    ----------
    x, y : float
        Position to test, in camera pixels.
    pick : tuple
        Coordinates of one pick in camera pixels. Must match the format
        of ``pick_shape``.
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"}
        Shape of the pick.
    pick_size : float or None
        Size of the pick in camera pixels. For circles - diameter. For
        rectangles - width. For squares - side length. For polygons,
        boxes and brush picks - ignored.

    Returns
    -------
    is_inside : bool
        True if the point is inside the pick. An open polygon (fewer
        than three vertices or not closed) never contains a point.
    """
    point_xy = np.array([[float(x)], [float(y)]])
    if pick_shape == "Circle":
        r = pick_size / 2
        return bool((x - pick[0]) ** 2 + (y - pick[1]) ** 2 < r**2)
    elif pick_shape == "Square":
        return bool(
            is_loc_in_square_numba(pick[0], pick[1], point_xy, pick_size)[0]
        )
    elif pick_shape == "Rectangle":
        (xs, ys), (xe, ye) = pick
        # the oriented test rotates the point into the rectangle's
        # frame, which - unlike ray casting over the corners - is well
        # defined for axis-aligned rectangles
        return bool(
            is_loc_in_rectangle_numba(
                0.5 * (xs + xe),
                0.5 * (ys + ye),
                np.arctan2(ye - ys, xe - xs),
                np.hypot(xe - xs, ye - ys),
                pick_size,
                point_xy,
            )[0]
        )
    elif pick_shape == "Polygon":
        X, Y = get_pick_polygon_corners(pick)
        if X is None:  # not a closed polygon
            return False
        return bool(
            check_if_in_polygon(
                point_xy[0], point_xy[1], np.array(X), np.array(Y)
            )[0]
        )
    elif pick_shape == "Box":
        (x0, y0), (x1, y1) = pick
        return bool(
            is_loc_in_box_numba(
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                point_xy,
                abs(x1 - x0),
                abs(y1 - y0),
            )[0]
        )
    elif pick_shape == "Brush":
        for stroke in pick:
            width, X, Y = brush_stroke_arrays(stroke)
            if check_if_in_brush_stroke(
                point_xy[0], point_xy[1], X, Y, width / 2
            )[0]:
                return True
        return False
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")


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


def _subcluster_label(
    name: str, dist: float | None, comparator: str, mean: float, std: float
) -> str:
    """Legend label for one subclustering population's histogram bar.

    ``comparator`` is "<" for the clustered population, ">" for sparse.
    """
    if dist is not None:
        return (
            f"{name} (d {comparator} {dist:.1f} nm) {mean:.1f} +/- {std:.1f}"
        )
    return f"{name} {mean:.1f} +/- {std:.1f}"


def _plot_subcluster_bar(
    ax: plt.Axes,
    events: IntArray1D,
    label: str,
    color: str,
    fill: bool = True,
    outline: bool = False,
) -> None:
    """Bar histogram + mean line for one subclustering population."""
    # resolve "CN" colors now: lines would otherwise look them up when
    # drawn, outside of any style context active here
    color = mcolors.to_hex(color)
    vals, counts = np.unique(events, return_counts=True)
    # half-transparent so that both populations stay visible
    style = histogram_style(color, fill, outline, fill_alpha=0.5)
    ax.bar(vals, counts, width=0.8, label=label, **style)
    ax.axvline(events.mean(), color=color, linestyle="--")


def _subclustering_title(
    has_clustered: bool,
    has_sparse: bool,
    clustered_n_events: IntArray1D,
    sparse_n_events: IntArray1D,
) -> str:
    """Plot title: KS-test summary, or why no test was performed."""
    if has_clustered and has_sparse:
        stat, p_perm, p = permutation_test(clustered_n_events, sparse_n_events)
        p_value_str = r"$p_{value}$"
        return (
            f"KS test: stat={stat:.4f}\n"
            f"permutation {p_value_str}={p_perm:.4f}\n"
            f"theoretical {p_value_str}={p:.4f}"
        )
    if has_clustered or has_sparse:
        return (
            "Only one population found, no statistical test performed; "
            "adjust distance parameters."
        )
    return (
        "No molecules found in either population, adjust distance"
        " parameters."
    )


def _save_subclustering_plot(
    fig: plt.Figure, plot_path: str | list[str]
) -> None:
    """Save ``fig`` to each path in ``plot_path`` (a str or list of strs)."""
    if not len(plot_path):
        return
    if isinstance(plot_path, str):
        plot_path = [plot_path]
    for path in plot_path:
        fig.savefig(path, dpi=300)


def plot_subclustering_check(
    clustered_n_events: IntArray1D,
    sparse_n_events: IntArray1D,
    plot_path: str | list[str] = "",
    return_fig: bool = False,
    clustering_dist: float | None = None,
    sparse_dist: float | None = None,
    fig: plt.Figure | None = None,
    fill: bool = True,
    outline: bool = False,
) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
    """Plot the results of subclustering analysis, see
    ``picasso.clusterer.test_subclustering``.

    Parameters
    ----------
    clustered_n_events : IntArray1D
        Number of events for clustered molecules.
    sparse_n_events : IntArray1D
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
    fig : plt.Figure, optional
        If given, the plot is drawn on this figure (which is cleared
        first). Otherwise, a new figure is created. Default is None.
    fill, outline : bool, optional
        Whether the bars are filled and/or outlined, see
        ``histogram_style``. Default is filled without an outline.

    Returns
    -------
    fig, ax : (plt.Figure, plt.Axes) or (None, None)
        Figure and axes if ``return_fig`` is True, otherwise
        (None, None).
    """
    has_clustered = len(clustered_n_events) > 0
    has_sparse = len(sparse_n_events) > 0

    m_clustered = clustered_n_events.mean()
    m_sparse = sparse_n_events.mean()
    s_clustered = clustered_n_events.std()
    s_sparse = sparse_n_events.std()

    # create the plot
    if fig is None:
        fig, ax1 = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
    else:
        fig.clear()
        ax1 = fig.subplots()
    if has_clustered or has_sparse:
        all_events = np.concatenate((sparse_n_events, clustered_n_events))
        min_bin, max_bin = np.percentile(all_events, [2.5, 97.5])

    if has_clustered:
        label = _subcluster_label(
            "Clustered", clustering_dist, "<", m_clustered, s_clustered
        )
        _plot_subcluster_bar(
            ax1, clustered_n_events, label, "C0", fill, outline
        )

    if has_sparse:
        label = _subcluster_label(
            "Sparse", sparse_dist, ">", m_sparse, s_sparse
        )
        _plot_subcluster_bar(ax1, sparse_n_events, label, "C1", fill, outline)

    if has_clustered or has_sparse:
        ax1.set_xlabel("Number of events")
        ax1.set_ylabel("Counts")
        ax1.set_xlim(min_bin - 1, max_bin + 1)
        ax1.legend()

    title = _subclustering_title(
        has_clustered, has_sparse, clustered_n_events, sparse_n_events
    )
    ax1.set_title(title, fontsize=10)
    _save_subclustering_plot(fig, plot_path)

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


# ---------------------------------------------------------------------------
# Lateral (x, y) corrections chained onto any calibration
# ---------------------------------------------------------------------------
# A calibration - Gaussian astigmatism (YAML), cubic-spline PSF (HDF5) or a
# standalone lateral calibration - may carry an ORDERED list of lateral
# corrections under LATERAL_TRANSFORMS_KEY. Each entry holds a geometric
# transform (affine, projective or polynomial; see picasso.transforms) mapping
# the coordinates of the movie being localized into a reference frame:
#
#   astigmatism : cylindrical-lens image -> reference (no-lens) image
#   chromatic   : this color channel     -> reference color channel
#
# They compose: a 3D two-color experiment applies the astigmatism correction
# and then the chromatic one, in the order they appear in the list.
#
# Single-channel only

LATERAL_TRANSFORMS_KEY = "Lateral transforms"
LATERAL_TRANSFORM_TYPES = ("astigmatism", "chromatic")


class DuplicateLateralTransformWarning(UserWarning):
    """A lateral correction was loaded separately although the calibration
    used for fitting already carries it, and was skipped.

    Its own category so that the CLI and the GUI can report the skip in
    their own idiom (a console line, a status-bar message) by filtering
    just this warning, instead of the same thing being said twice.
    """


def lateral_transforms(calibration: dict | list | None) -> list[dict]:
    """The ordered lateral-correction entries carried by a calibration.

    Parameters
    ----------
    calibration : dict or list or None
        Any calibration dictionary (Gaussian astigmatism, spline PSF or a
        standalone affine calibration), an already-extracted list of
        entries, or None.

    Returns
    -------
    transforms : list of dicts
        The entries in the order they must be applied. Empty if the
        calibration carries none.
    """
    if calibration is None:
        return []
    if isinstance(calibration, list):
        return [t for t in calibration if t]
    if not isinstance(calibration, dict):
        # e.g. a calibration still given as a path; nothing to read
        return []
    return list(calibration.get(LATERAL_TRANSFORMS_KEY) or [])


def resolve_lateral_transforms(source: dict | list | str | None) -> list:
    """The lateral corrections held by ``source``, reading it from disk if
    it is a path.

    Parameters
    ----------
    source : dict, list, str or None
        A calibration dictionary, a list of entries, the path of a
        calibration file of any kind (see ``io.load_any_calibration``), or
        None.

    Returns
    -------
    transforms : list of dicts
        The entries in the order they must be applied. Empty if `source`
        is None or carries none.

    Raises
    ------
    ValueError
        If `source` is a path to a file that carries no lateral
        corrections, which would otherwise be a silent no-op.
    """
    if not isinstance(source, str):
        return lateral_transforms(source)
    transforms = lateral_transforms(io.load_any_calibration(source))
    if not transforms:
        raise ValueError(
            f"No lateral corrections found in {source}. Build one with "
            "'picasso lateral-calibrate' or the 'Calibrate lateral "
            "transform' dialog in Picasso: Localize."
        )
    return transforms


def lateral_transform_models(calibration: dict | list | None) -> list:
    """The geometric transforms of ``lateral_transforms``, in the order they
    must be applied.

    Parameters
    ----------
    calibration : dict, list or None
        A calibration carrying an ``"Affine transforms"`` list, that list
        itself, or None. Its items may be entries (dicts with a
        ``"Transform"`` key) or bare transforms.

    Returns
    -------
    models : list of picasso.transforms.Transform
        One transform per correction, in application order.
    """
    from picasso import transforms as _tf

    models = []
    for entry in lateral_transforms(calibration):
        if isinstance(entry, dict) and "Transform" in entry:
            entry = entry["Transform"]
        models.append(_tf.from_dict(entry))
    return models


def append_lateral_transform(calibration: dict, entry: dict) -> dict:
    """Append a lateral correction to ``calibration``'s ordered list.

    An existing entry of the same ``"Type"`` is replaced in place (keeping
    its position), so re-running a calibration updates it instead of
    stacking a second copy of the same correction.

    Parameters
    ----------
    calibration : dict
        Calibration to append to; modified in place. May be empty, which
        is how a standalone affine calibration file is started.
    entry : dict
        The transform entry, as built by
        ``picasso.localize.fit_lateral_transform``. Must carry
        ``"Transform"`` and ``"Type"``.

    Returns
    -------
    calibration : dict
        The same dictionary, with the entry appended or replaced.
    """
    transforms = lateral_transforms(calibration)
    kind = entry.get("Type")
    for i, existing in enumerate(transforms):
        if existing.get("Type") == kind:
            transforms[i] = entry
            break
    else:
        transforms.append(entry)
    calibration[LATERAL_TRANSFORMS_KEY] = transforms
    return calibration


def apply_lateral_transforms(
    locs: pd.DataFrame, calibration: dict | list | None
) -> pd.DataFrame:
    """Map ``locs`` x/y through a calibration's lateral corrections.

    The transforms are applied in stored order, in camera-pixel
    coordinates. Returns ``locs`` unchanged (and untouched) when there is
    nothing to apply, so this is safe to call on every fit path.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x`` and ``y`` columns.
    calibration : dict or list or None
        A calibration carrying affine corrections, a list of entries, or
        None.

    Returns
    -------
    locs : pd.DataFrame
        Localizations with corrected ``x`` and ``y``. A copy, if anything
        was applied.
    """
    models = lateral_transform_models(calibration)
    if not models or not len(locs):
        return locs
    locs = locs.copy()
    xy = np.column_stack(
        [
            locs["x"].to_numpy(dtype=np.float64),
            locs["y"].to_numpy(dtype=np.float64),
        ]
    )
    for model in models:
        xy = model.apply(xy)
    locs["x"] = xy[:, 0].astype(np.float32)
    locs["y"] = xy[:, 1].astype(np.float32)
    return locs


def is_same_lateral_transform(
    a: dict | list, b: dict | list, tol: float = 1e-9
) -> bool:
    """Whether two lateral entries hold the same transform (within ``tol``).

    Compared by the transform, not by identity or source path: the same
    correction saved twice under different names must count as one, since
    applying it twice would correct twice.

    Parameters
    ----------
    a, b : dict or list
        Lateral correction entries (dicts with a ``"Transform"`` key) or bare
        transforms.
    tol : float, optional
        Absolute tolerance of the parameter comparison. Default 1e-9.

    Returns
    -------
    is_same : bool
    """
    from picasso import transforms as _tf

    if isinstance(a, dict) and "Transform" in a:
        a = a["Transform"]
    if isinstance(b, dict) and "Transform" in b:
        b = b["Transform"]
    return _tf.from_dict(a).allclose(_tf.from_dict(b), tol)


def drop_duplicate_lateral_transforms(
    transforms: dict | list | None, applied: dict | list | None
) -> tuple[list[dict], list[dict]]:
    """Split ``transforms`` into the ones still to apply and the ones
    ``applied`` already covers.

    A calibration carries its own corrections and applies them itself, so an
    extra correction loaded alongside it - typically the very same file, or a
    standalone copy of the same transform - must not be applied a second
    time.

    Parameters
    ----------
    transforms : dict or list or None
        The extra corrections, as a calibration or a list of entries.
    applied : dict or list or None
        Corrections the fit applies on its own, i.e. those carried by its
        calibration.

    Returns
    -------
    new : list of dicts
        Entries of ``transforms`` not already in ``applied``, in order.
    duplicates : list of dicts
        Entries dropped because ``applied`` already covers them.
    """
    already = lateral_transforms(applied)
    new, duplicates = [], []
    for transform in lateral_transforms(transforms):
        if any(is_same_lateral_transform(transform, a) for a in already):
            duplicates.append(transform)
        else:
            new.append(transform)
    return new, duplicates


def describe_lateral_transforms(calibration: dict | list | None) -> list[str]:
    """One human-readable line per lateral correction.

    For metadata and GUI labels, e.g.
    ``"astigmatism, projective (25 bead pairs)"``.

    Parameters
    ----------
    calibration : dict, list or None
        A calibration carrying an ``"Affine transforms"`` list, that list
        itself, or None.

    Returns
    -------
    described : list of str
        One line per correction, in application order.
    """
    described = []
    for transform in lateral_transforms(calibration):
        purpose = transform.get("Type", "lateral")
        model = (transform.get("Transform") or {}).get("model")
        pairs = transform.get("Bead pairs")
        label = f"{purpose}, {model}" if model else str(purpose)
        described.append(f"{label} ({pairs} bead pairs)" if pairs else label)
    return described
