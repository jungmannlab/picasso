"""
picasso.gui.localize
~~~~~~~~~~~~~~~~~~~~

Graphical user interface for localizing single molecules.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import glob
import os.path
import re
import sys
import threading
import time
import warnings
from collections import UserDict
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
import imageio
from .. import (
    aim,
    CONFIG,
    imageprocess,
    io,
    localize,
    lib,
    postprocess,
    registration,
    scmos,
    spline,
    __version__,
    zfit,
)

# aliased: `transforms` is used as a local name for lists of channel
# transforms throughout this module
from .. import transforms as transforms_mod
from ..fitting import precision, splinefit
from .app import run_gui
from PyQt6 import QtCore, QtGui, QtWidgets
from playsound3 import playsound

CUDA_AVAILABLE = localize.CUDA_AVAILABLE
CMAP_GRAYSCALE = [QtGui.qRgb(_, _, _) for _ in range(256)]
# Frames sampled to size the contrast slider's track. Kept small: with
# the temporal median on, reading one frame reads a whole window.
CONTRAST_SLIDER_SAMPLES = 10
# Extra headroom on the sampled range, so that the white point can be
# pushed past the brightest sampled pixel (and black below the dimmest).
CONTRAST_SLIDER_PADDING = 0.05
DEFAULT_PARAMETERS = {
    "Box Size": 7,
    "Min. Net Gradient": 5000,
    "Gaussian Filter Sigma": 0.0,
    "Temporal Median Window": 51,
}

IDENTIFY_MODE_SEPARATE = "Each channel separately"
IDENTIFY_MODE_SUM = "Sum of channels"
# How multichannel (and split-FOV) data is fitted: all channels tied into one
# global fit, or every channel on its own. Defined in ``picasso.localize``,
# which records the mode in the metadata of an independent fit.
FIT_MODE_JOINT = localize.FIT_MODE_JOINT
FIT_MODE_SEPARATE = localize.FIT_MODE_INDEPENDENT
FIT_MODE_TOOLTIP = (
    "How the loaded channels (or split-FOV regions) are fitted.\n\n"
    f"'{FIT_MODE_JOINT}': the channels are fitted together,\n"
    "sharing one position per molecule (similar to globLoc). Needs a "
    "multichannel\n"
    "spline calibration or a channel registration, is available for the\n"
    "spherical Gaussian and the spline PSF only, and keeps just the\n"
    "molecules detected in every channel - but reaches the best precision.\n"
    "With no calibration loaded the active channel is fitted alone.\n\n"
    f"'{FIT_MODE_SEPARATE}': every channel is fitted on its own,\n"
    "with its own model, optimizer and calibrations, and saved to its own\n"
    "file. No registration (i.e., matching of signal across channels) is\n"
    "needed and every detection is fitted; the channels can be connected\n"
    "afterwards."
)
IMAGE_FILTER = (
    "All supported formats ("
    + " ".join("*" + e for e in io.MOVIE_EXTENSIONS)
    + ")"
    ";;Raw files (*.raw)"
    ";;Tif images (*.tif *.tiff)"
    ";;BigTiff (*.btf *.tf8 *.tf2)"
    ";;Zeiss LSM (*.lsm)"
    ";;Zeiss CZI (*.czi)"
    ";;Leica LIF (*.lif)"
    ";;ImaRIS IMS (*.ims)"
    ";;Nd2 files (*.nd2)"
    ";;STK files (*.stk)"
)

# Distinct box colors for the cross-channel link overlay (gray is kept
# out of the palette so it reads as "unmatched"). tab20-style hues.
LINK_COLORS = [
    QtGui.QColor(*_rgb)
    for _rgb in (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (219, 219, 141),
        (158, 218, 229),
    )
]
LINK_UNMATCHED_COLOR = QtGui.QColor(150, 150, 150)

# Below this many frames the temporal median filter cannot do anything: the
# rolling median over one frame is that frame, so the filtered movie is
# identically zero (see Window.identification_movie).
_TEMPORAL_MEDIAN_MIN_FRAMES = 2

# Frames sampled when the channels are registered from their own detections
# for the channel sum. Generous compared to the link-color preview's default:
# the channel that makes summing worthwhile in the first place is the dim one,
# and it contributes few detections per frame.
_SUM_REGISTRATION_MAX_FRAMES = 200


# The geometric transform models offered wherever Picasso fits one, as
# (label, model). A polynomial's degree is part of the model name rather than
# a separate field, so the combo carries one value - see picasso.transforms.
_TRANSFORM_MODELS = [
    ("Translation (2 DOF, >= 1 pair)", "translation"),
    ("Affine (6 DOF, >= 3 pairs)", "affine"),
    ("Projective (8 DOF, >= 4 pairs)", "projective"),
    ("Polynomial, degree 2 (>= 6 pairs)", "polynomial2"),
    ("Polynomial, degree 3 (>= 10 pairs)", "polynomial3"),
]

_TRANSFORM_MODEL_TOOLTIP = (
    "How the two coordinate frames are related.\n"
    "  Translation: a shift in x and y and nothing else - for frames that\n"
    "    are known to differ only by an offset. Any rotation or scale then\n"
    "    shows up in the residual instead of being absorbed.\n"
    "  Affine: translation, rotation, scale and shear - the default, and\n"
    "    what a well-aligned splitter does to first order.\n"
    "  Projective: adds the perspective/keystone term a tilted dichroic or\n"
    "    an unequal path length introduces.\n"
    "  Polynomial: a smooth warp that follows genuine field distortion.\n"
    "    Not an optical model, and it extrapolates badly outside the field\n"
    "    the beads span, so use it only with many, well-spread beads.\n"
    "The minima above are hard requirements; about three times as many is\n"
    "wanted, otherwise the fit interpolates the noise in them."
)


def _transform_model_combo() -> QtWidgets.QComboBox:
    """A combo box offering the transform models, defaulting to affine."""
    combo = QtWidgets.QComboBox()
    for label, data in _TRANSFORM_MODELS:
        combo.addItem(label, data)
    # The list is ordered by increasing flexibility, so the default is not the
    # first entry; select it explicitly.
    combo.setCurrentIndex(combo.findData("affine"))
    combo.setToolTip(_TRANSFORM_MODEL_TOOLTIP)
    return combo


def _transform_model_of(combo: QtWidgets.QComboBox) -> str:
    """The model currently selected in a model combo."""
    return combo.currentData() or "affine"


def _select_transform_model(
    combo: QtWidgets.QComboBox, model: str | None
) -> None:
    """Pre-select ``model`` in a model combo, if it is offered."""
    index = combo.findData(model)
    if index >= 0:
        combo.setCurrentIndex(index)


def _retain_size_when_hidden(widget: QtWidgets.QWidget) -> None:
    """Keep a widget's space reserved while it is hidden.

    The multichannel-only widgets appear and disappear with the loaded data.
    Reserving their space means the Parameters dialog keeps one shape: nothing
    below them shifts when channels are opened or closed."""
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


def _normalized_path(path: str) -> str:
    """A path in a form that can be compared for identity across the
    dialogs (absolute, and case-insensitive where the platform is)."""
    return os.path.normcase(os.path.abspath(path)) if path else ""


def _nearest_unique_match(
    pred_xy: np.ndarray, target_xy: np.ndarray, tol: float
) -> dict[int, int]:
    """Nearest-neighbor match within ``tol``.

    ``pred_xy`` are reference points predicted into a channel (via the
    calibration transform); ``target_xy`` are that channel's own detections.
    Returns ``{target_index: reference_index}`` with each target and each
    reference used at most once (closest pair wins) - the display counterpart
    of the per-frame pairing used in the signal re-registration.
    """
    pred_xy = np.asarray(pred_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    out: dict[int, int] = {}
    if len(pred_xy) == 0 or len(target_xy) == 0:
        return out
    dists = np.sqrt(
        ((pred_xy[:, None, :] - target_xy[None, :, :]) ** 2).sum(axis=2)
    )  # (n_reference, n_target)
    pairs = []
    for k in range(dists.shape[0]):
        j = int(np.argmin(dists[k]))
        if dists[k, j] <= tol:
            pairs.append((float(dists[k, j]), k, j))
    pairs.sort()
    used_ref: set[int] = set()
    used_target: set[int] = set()
    for _d, k, j in pairs:
        if k in used_ref or j in used_target:
            continue
        used_ref.add(k)
        used_target.add(j)
        out[j] = k
    return out


def _normalize_rect(
    rect: list,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``[[y_a, x_a], [y_b, x_b]]`` -> ``((y_min, x_min), (y_max, x_max))``."""
    (ya, xa), (yb, xb) = rect
    return (min(ya, yb), min(xa, xb)), (max(ya, yb), max(xa, xb))


LINK_PHOTONS_TIP = (
    "2- to 6-channel fitting.\n\n"
    "CHECK to link photons + background across all channels. "
    "Appropriate when the channels share one emission.\n\n"
    "UNCHECK to decouple: each channel fits a free photon count"
    " + background, x/y/z shared. Adds photons_ch<c>, bg_ch<c> and"
    " rel_photons_ch<c> (the channel's share of the total photons)"
    " to the localizations."
)

GAUSS_LINK_PHOTONS_TIP = (
    "How the multichannel 2D Gaussian fit treats brightness.\n\n"
    "UNCHECKED (default): each channel fits its own photon count and"
    " background,\n"
    "with only x, y and the width shared. Makes no assumption about how the"
    "\nlight is split. Adds photons_ch<c>, bg_ch<c> and rel_photons_ch<c>\n"
    "(the channel's share of the total photons).\n\n"
    "CHECKED: one photon count and background shared by every channel."
)

# Fitting models offered in the GUI, decoupled from the optimizer. Each
# model maps its optimizer labels to the internal ``fit`` codes;
# models without an optimizer (e.g. averaging) declare a fixed ``code``
# and ``optimizers=None``. Add a new fitting algorithm by adding an
# entry here.
FIT_MODELS = {
    "2D elliptical Gaussian": {
        "optimizers": {"Least squares": "gausslq", "MLE": "gaussmle"},
    },
    "2D rotated elliptical Gaussian": {
        "optimizers": {
            "Least squares": "gausslq-rotated",
            "MLE": "gaussmle-rotated",
        },
    },
    "2D spherical Gaussian": {
        "optimizers": {
            "Least squares": "gausslq-spherical",
            "MLE": "gaussmle-spherical",
        },
        "needs_channel_registration": True,
    },
    "Experimental PSF (cubic spline)": {
        "optimizers": {
            "Least squares": "spline",
            "MLE": "spline-mle",
        },
        "needs_spline_calibration": True,
    },
    "Average of ROI": {
        "optimizers": None,
        "code": "avg",
    },
}


MODEL_TOOLTIP = (
    "Model fit to each identified spot:\n\n"
    "2D elliptical Gaussian:\n"
    "Gaussian with independent widths in x and y.\n"
    "Standard choice for 2D data and required for 3D via astigmatism.\n\n"
    "2D rotated elliptical Gaussian:\n"
    "As above, plus a fitted rotation angle of the ellipse.\n"
    "Useful for tilted/anisotropic PSFs.\n\n"
    "2D spherical Gaussian:\n"
    "Gaussian with one common width for x and y.\n\n"
    "Experimental PSF (cubic spline):\n"
    "Fits a cubic spline interpolation of a measured 3D PSF\n"
    "(requires a spline calibration file).\n"
    "Most accurate model and yields z directly, also for aberrated PSFs.\n"
    "Runs on the CPU; tick Use GPU to run it on the GPU instead\n"
    "(much faster).\n\n"
    "Average of ROI:\n"
    "Reports the spot's center of mass and integrated intensity\n"
    "in the fit box."
)

CAMERA_CALIB_TOOLTIP = (
    "Per-pixel offset, readout-variance and (optionally) gain maps measured\n"
    "from a dark movie and (optionally) a light movie, as in Huang et al.,"
    " Nat. Methods 2013.\n\n"
    "When loaded, the offset map replaces Baseline and the gain map replaces\n"
    "Sensitivity, and maximum-likelihood fits use the pixel-dependent sCMOS "
    "noise model.\n\n"
    "Least-squares fits are unaffected by the noise model itself but their "
    "reported\n"
    "uncertainty grows. **For sCMOS data prefer an MLE method, whose Cramer-Rao"
    "\nbound is exact under the model.**\n\n"
    "Build one with Calibrate > Compute sCMOS camera calibration."
)

OPTIMIZER_TOOLTIP = (
    "Optimizer used to fit the model to data:\n\n"
    "Least squares:\n"
    "Minimizes the squared residuals between model and data.\n"
    "Fast and robust, but assumes Gaussian noise, so it is slightly\n"
    "biased for the Poisson (shot) noise of low-photon spots.\n\n"
    "MLE:\n"
    "Maximum likelihood estimation with a Poisson noise model.\n"
    "Statistically optimal (precision close to the Cramer-Rao lower\n"
    "bound) and the better choice for dim spots."
)


def _copied_rois(rois: list) -> list:
    """Copy a list of ROI rectangles down to their corners.

    ROIs are edited in place all over the view (a region is moved by
    rewriting its corners, a double click deletes one from the list), so a
    snapshot that shares those lists changes underneath whoever took it -
    a stale identification would keep comparing equal to the edited ROIs
    in ``identifications_outdated``. Corners are plain ints, so copying the
    two corner lists is deep enough.
    """
    return [[list(rect[0]), list(rect[1])] for rect in rois]


def _filtered_identification_movie(
    movie,
    window: int,
    sigma: float,
    cache: list,
) -> tuple[object, list]:
    """Put the identification filters on top of ``movie``: the temporal
    median background subtraction, then the Gaussian smoothing - the same
    two stages ``localize.identify`` puts on top of the movie it searches,
    so the preview and the batch run find the same spots.

    Unlike the batch run, the stack is always built over the *whole* frame,
    never restricted to the drawn ROIs. The display reads its frames from
    here (see ``Window._draw_frame``), and a ROI-restricted temporal median
    leaves everything outside its bounding box at zero: with the filter on,
    drawing the first ROI would black out the rest of the movie and the
    next ROI could not be placed. Identification is unaffected - the median
    is computed per pixel, so inside the ROIs (and the padding around them
    the gradients are read from) the filtered frames are identical either
    way. Restricting the median stays a batch-run optimization, where
    nothing is displayed.

    Both stages are lazy views, so ``cache`` (a ``[temporal, gaussian]``
    pair) keeps them across redraws: a stage is rebuilt only when its
    source movie or its setting changed. Comparing the source by identity
    chains the invalidation, so rebuilding (or dropping) the median also
    rebuilds the Gaussian on top of it. Returns the filtered movie and the
    refreshed cache; ``window`` of 0 (or a movie too short for it, which
    the caller resolves) and ``sigma`` of 0 switch the respective stage
    off.
    """
    temporal, gaussian = cache
    if window:
        if (
            temporal is None
            or temporal.raw is not movie
            or temporal.window != min(window, len(movie))
        ):
            # comparing against the raw movie by identity means loading a
            # movie or switching channel invalidates this for free
            temporal = localize.TemporalMedianMovie(movie, window)
        movie = temporal
    else:
        temporal = None
    if sigma:
        if (
            gaussian is None
            or gaussian.raw is not movie
            or gaussian.sigma != sigma
        ):
            gaussian = localize.GaussianFilteredMovie(movie, sigma)
        movie = gaussian
    else:
        gaussian = None
    return movie, [temporal, gaussian]


def _fit_code(model: str, optimizer: str) -> str:
    """Resolve a (model, optimizer) selection to an internal ``fit`` code."""
    entry = FIT_MODELS[model]
    if entry["optimizers"] is None:
        return entry["code"]
    return entry["optimizers"][optimizer]


# Fit codes with both a CPU and a GPU implementation, selected by the "Use
# GPU" checkbox rather than by the model/optimizer comboboxes.
_GPU_CAPABLE_CODES = frozenset(
    code
    for code in localize.FIT_METHODS
    if not code.endswith("-gpu") and f"{code}-gpu" in localize.FIT_METHODS
)


def _effective_fit_code(code: str, use_gpu: bool) -> str:
    """The ``fit`` code a (code, GPU checkbox) pair actually runs.

    "Use GPU" is a *modifier* on the model and optimizer comboboxes, so the
    code it produces is assembled here rather than enumerated."""
    if use_gpu and code in _GPU_CAPABLE_CODES:
        return code + "-gpu"
    return code


# Fit codes that iterate, and the default convergence schedule of each. Every
# method except "avg" is here: all of them run an iterative solver, so all of
# them honor the convergence criterion and the maximum-iteration count.
#
# The Gaussian defaults come from ``localize.gauss_schedule`` rather than
# being repeated, so the boxes cannot show a schedule the fit does not use.
_SPLINE_SCHEDULE = (
    splinefit.TOLERANCE_MULTI_START,
    splinefit.MAX_ITERATIONS_MULTI_START,
)
_CONVERGENCE_DEFAULTS = {
    code: (
        _SPLINE_SCHEDULE
        if code.startswith("spline")
        else localize.gauss_schedule(
            localize.parse_gauss_code(code)["mle"],
            localize.parse_gauss_code(code)["use_gpu"],
        )
    )
    for code in localize.FIT_METHODS
    if code != "avg"
}
_CONVERGENCE_CODES = frozenset(_CONVERGENCE_DEFAULTS)


# Steps allotted to each file in the load progress dialog, so the bar can
# advance smoothly within a file from the loader's per-page reports.
PROGRESS_RESOLUTION = 1000


@dataclass
class Channel:
    """Per-channel state for multichannel localization.

    Localize keeps all loaded channels in ``Window.channels`` and mirrors
    the *active* channel into the flat ``Window`` attributes (``movie``,
    ``info``, ``identifications`` ...) so the existing single-movie
    identify/fit/save/draw code keeps operating on the active channel
    unchanged. Switching channels snapshots the flat state (plus the
    Parameters dialog values) into the old ``Channel`` and restores the
    new one.

    All channels' ``movie`` objects stay live simultaneously, so a future
    across-channel fitting algorithm can read every channel at once.
    """

    movie: object = None
    info: list = field(default_factory=list)
    path: str = ""
    name: str = "Channel 0"
    identifications: object = None
    locs: object = None
    locs_display: object = None
    ready_for_fit: bool = False
    last_identification_info: dict | None = None
    extra_info: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _region_fit_summary(params: dict) -> list[tuple[str, str]]:
    """One split-FOV region's fit settings as ``(text, tooltip)`` pairs for
    the ROI table: the model (with its optimizer) and its PSF calibration."""
    models = list(FIT_MODELS)
    model = models[min(params.get("fit_model", 0), len(models) - 1)]
    optimizers = FIT_MODELS[model]["optimizers"]
    optimizer = ""
    if optimizers:
        names = list(optimizers)
        optimizer = names[min(params.get("fit_optimizer", 0), len(names) - 1)]
    path = params.get("spline_calibration_path") or ""
    return [
        (model, f"{model}, {optimizer}" if optimizer else model),
        (os.path.basename(path) if path else "--", path or "none loaded"),
    ]


def _sanitize_filename(name: str) -> str:
    """Turn a channel name into a filename-safe suffix."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "channel"


def _format_mng(minimum_ng: float | list) -> str:
    """Render a minimum net gradient for the status bar. Split-FOV data
    carries one threshold per region, shown as ``ref/ch1/...``."""
    if isinstance(minimum_ng, (list, tuple)):
        return "/".join(f"{int(_):,}" for _ in minimum_ng)
    return f"{int(minimum_ng):,}"


class _LoadCanceledError(Exception):
    """Raised inside the loader's progress callback to abort a canceled
    load mid-file (the io calls are otherwise uninterruptible)."""


class MovieLoadWorker(QtCore.QObject):
    """Load several movie files off the GUI thread, one file per channel.

    Loading large movies on the main thread blocks Qt's event loop, so the
    window stops repainting and responding while several files are read.
    This worker runs ``io.load_movie`` for each path on a background
    thread instead.

    The ``prompt_info`` callbacks show modal dialogs and therefore *must*
    run on the GUI thread. When a loader needs one, the worker emits
    ``prompt_requested`` (delivered to the main thread via a queued
    connection) and blocks on a ``threading.Event`` until the main thread
    has filled the answer into the shared ``holder`` dict and released it.
    """

    progress = QtCore.pyqtSignal(int, str)  # index, filename
    # sub-file progress within the current file: (done, total) pages
    subprogress = QtCore.pyqtSignal(int, int)
    # callback, (args, kwargs), holder dict for the return value
    prompt_requested = QtCore.pyqtSignal(object, object, object)
    finished = QtCore.pyqtSignal(list, list, list)  # movies, infos, paths
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        paths: list[str],
        prompt_for_path,
        load_all: bool = False,
        concat: bool = False,
    ) -> None:
        super().__init__()
        self.paths = paths
        self._prompt_for_path = prompt_for_path
        # When True, each path is read with ``io.load_movie_all`` (every
        # channel of one multichannel file); otherwise ``io.load_movie``
        # loads one channel per file.
        self.load_all = load_all
        # When True, all paths are read as *one* movie, concatenated
        # along the frame axis (``io.load_tif_concatenated``).
        self.concat = concat
        self._prompt_event = threading.Event()
        self._canceled = False

    def cancel(self) -> None:
        """Request cancellation; takes effect before the next file."""
        self._canceled = True
        # Release the worker if it is currently blocked on a prompt.
        self._prompt_event.set()

    def _proxy_prompt(self, callback):
        """Wrap a GUI prompt callback so the dialog runs on the main
        thread while the worker thread blocks for the result."""

        def wrapper(*args, **kwargs):
            holder = {}
            self._prompt_event.clear()
            self.prompt_requested.emit(callback, (args, kwargs), holder)
            self._prompt_event.wait()
            return holder.get("result")

        return wrapper

    def run(self) -> None:
        movies, infos, paths = [], [], []
        try:
            # Normally one job per file (one channel each); when
            # concatenating, all files together form a single job that
            # yields one movie.
            jobs = [self.paths] if self.concat else [[_] for _ in self.paths]
            for i, job in enumerate(jobs):
                if self._canceled:
                    break
                path = job[0]
                label = (
                    f"{len(job)} files"
                    if self.concat
                    else os.path.basename(path)
                )
                self.progress.emit(i, label)
                prompt = self._proxy_prompt(self._prompt_for_path(path))

                # Called (queued to the GUI thread) as io scans the
                # file's IFDs, so the bar advances smoothly within a
                # file. It is also the only code of ours that runs
                # *during* the otherwise-blocking io call, so it doubles
                # as the mid-file cancellation point.
                def report(done: int, total: int) -> None:
                    if self._canceled:
                        raise _LoadCanceledError
                    self.subprogress.emit(done, total)

                if self.concat:
                    result = io.load_tif_concatenated(
                        job, prompt_info=prompt, progress=report
                    )
                    if result is None:
                        continue
                    movie, info = result
                    movies.append(movie)
                    infos.append(info)
                    paths.append(path)
                elif self.load_all:
                    result = io.load_movie_all(
                        path, prompt_info=prompt, progress=report
                    )
                    if result is None:
                        continue
                    file_movies, file_infos = result
                    for movie, info in zip(file_movies, file_infos):
                        movies.append(movie)
                        infos.append(info)
                        paths.append(path)
                else:
                    result = io.load_movie(
                        path, prompt_info=prompt, progress=report
                    )
                    if result is None:
                        continue
                    movie, info = result
                    movies.append(movie)
                    infos.append(info)
                    paths.append(path)
        except Exception as e:  # noqa: BLE001 - reported to the GUI
            if not self._canceled:
                self.failed.emit(str(e))
                return
            movies, infos, paths = [], [], []
        if self._canceled:
            # The file being read when the user canceled has still been
            # loaded to completion (the blocking io call cannot be
            # interrupted); discard it instead of delivering it.
            movies, infos, paths = [], [], []
        self.finished.emit(movies, infos, paths)


class RubberBand(QtWidgets.QRubberBand):
    """Red rubber band for selecting ROI."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(QtWidgets.QRubberBand.Shape.Rectangle, parent)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Change the color of the rubber band."""
        painter = QtGui.QPainter(self)
        color = QtGui.QColor(QtCore.Qt.GlobalColor.blue)
        painter.setPen(QtGui.QPen(color))
        rect = event.rect()
        rect.setHeight(int(rect.height() - 1))
        rect.setWidth(int(rect.width() - 1))
        painter.drawRect(rect)


class View(QtWidgets.QGraphicsView):
    """Central widget which shows ``Scene`` objects of individual
    frames.

    ...

    Attributes
    ----------
    hscrollbar, vscrollbar : QtWidgets.QScrollBar
        Horizontal and vertical scroll bars.
    pan : bool
        Whether the view is currently panned.
    pan_start_x, pan_start_y : int
        Starting position of the pan gesture.
    rubberband : QtWidgets.QRubberBand
        Transient rubber band shown while dragging a new ROI.
    rois : list
        Regions of interest (ROIs) selected by the user. Each ROI is
        ``[[y_min, x_min], [y_max, x_max]]``. The ROIs are kept disjoint
        (overlapping selections are clipped via
        ``localize.clip_rois``). An empty list means the whole frame.
    selected_roi : int or None
        Index of the ROI currently highlighted (selected in the
        parameters dialog table), or None.
    roi_mngs : list
        Split-FOV only: the minimum net gradient of each region, parallel
        to ``rois``. The regions are separate channels imaged through
        different optics, so each gets its own detection threshold. Kept
        in step with ``rois`` by ``Window.region_mngs``; empty (and
        ignored) whenever ``split_fov_mode`` is off.
    roi_params : list
        Split-FOV only: the fit settings of each region (model, optimizer,
        convergence, calibrations), parallel to ``rois``. A region fitted on
        its own is a channel of its own, with its own PSF; kept in step with
        ``rois`` by ``Window.region_params`` and used only when the regions
        are fitted separately (see ``Window.fit_mode``).
    roi_end : QtCore.QPoint
        End point of the ROI being dragged.
    window : QtWidgets.QMainWindow
        Reference to the main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setAcceptDrops(True)
        self.pan = False
        self.hscrollbar = self.horizontalScrollBar()
        self.hscrollbar.valueChanged.connect(self.on_scroll)
        self.vscrollbar = self.verticalScrollBar()
        self.vscrollbar.valueChanged.connect(self.on_scroll)
        self.rubberband = RubberBand(self)
        self.rois = []
        self.selected_roi = None
        # per-region min. net gradient in split-FOV mode, see roi_mngs above
        self.roi_mngs = []
        # per-region fit settings in split-FOV mode, see roi_params above
        self.roi_params = []
        # Split-FOV region mode: ROIs are equal-size rectangular channels of one
        # movie. The first region drawn fixes the size (derived live from the
        # existing regions, so clearing them frees the size again); further
        # regions snap to it, and an existing region can be dragged (moved) to
        # fine-tune its registration. Toggled by ``window.set_split_fov_mode``.
        self.split_fov_mode = False
        self._moving_roi = None  # index of the region being dragged
        self._move_anchor = None  # (scene_dy, scene_dx) press offset in region
        # A double click fires press/release/doubleClick/release; this flag lets
        # the trailing release be ignored so deleting a region does not
        # immediately re-add one at the same spot.
        self._suppress_release = False

    def _frame_shape(self) -> tuple[int, int] | None:
        """(`height`, `width`) of the current movie frame, or None."""
        movie = getattr(self.window, "movie", None)
        if movie is None:
            return None
        return int(movie.shape[1]), int(movie.shape[2])

    def _region_size(self) -> tuple[int, int] | None:
        """Shared (`height`, `width`) of the split-FOV regions, taken from the
        first (reference) region, or None when there are no regions yet.

        Derived live rather than stored so that removing every region - by any
        means (double-click, the ROI field, the numeric dialog) - frees the
        size again and the next drawn region can define a new one."""
        if not self.rois:
            return None
        (y_min, x_min), (y_max, x_max) = self.rois[0]
        return (y_max - y_min, x_max - x_min)

    def _roi_at(self, px: float, py: float) -> int | None:
        """Index of the smallest region containing scene point (px, py)."""
        containing = [
            i
            for i, ((y_min, x_min), (y_max, x_max)) in enumerate(self.rois)
            if y_min <= py <= y_max and x_min <= px <= x_max
        ]
        if not containing:
            return None
        return min(
            containing,
            key=lambda i: (
                (self.rois[i][1][0] - self.rois[i][0][0])
                * (self.rois[i][1][1] - self.rois[i][0][1])
            ),
        )

    def _add_region(self, y0: int, x0: int) -> None:
        """Append a region of the established size with top-left (y0, x0),
        clamped inside the frame; selects it. No clipping (channels may abut).
        """
        size = self._region_size()
        if size is None:
            return
        h, w = size
        shape = self._frame_shape()
        if shape is not None:
            fh, fw = shape
            y0 = int(min(max(0, y0), max(0, fh - h)))
            x0 = int(min(max(0, x0), max(0, fw - w)))
        self.rois.append([[int(y0), int(x0)], [int(y0 + h), int(x0 + w)]])
        self.selected_roi = len(self.rois) - 1
        self.window.parameters_dialog.update_roi_display()

    def _move_region(self, idx: int, y0: int, x0: int) -> None:
        """Translate region ``idx`` so its top-left is (y0, x0), clamped."""
        (y_min, x_min), (y_max, x_max) = self.rois[idx]
        h, w = y_max - y_min, x_max - x_min
        shape = self._frame_shape()
        if shape is not None:
            fh, fw = shape
            y0 = int(min(max(0, y0), max(0, fh - h)))
            x0 = int(min(max(0, x0), max(0, fw - w)))
        self.rois[idx] = [[int(y0), int(x0)], [int(y0 + h), int(x0 + w)]]

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Start either a rubber band for selecting a ROI or panning the
        view."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.split_fov_mode:
                scene_pos = self.mapToScene(event.pos())
                idx = self._roi_at(scene_pos.x(), scene_pos.y())
                if idx is not None:
                    # begin moving an existing region
                    self._moving_roi = idx
                    self.selected_roi = idx
                    (y_min, x_min), _ = self.rois[idx]
                    self._move_anchor = (
                        scene_pos.y() - y_min,
                        scene_pos.x() - x_min,
                    )
                    self.window.parameters_dialog.update_roi_display()
                    self.window.draw_frame()
                    return
            self.roi_origin = QtCore.QPoint(event.pos())
            self.rubberband.setGeometry(
                QtCore.QRect(self.roi_origin, QtCore.QSize())
            )
            self.rubberband.show()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.pan = True
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Update the rubber band or pan the view."""
        if (
            self.split_fov_mode
            and self._moving_roi is not None
            and event.buttons() == QtCore.Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(event.pos())
            dy, dx = self._move_anchor
            self._move_region(
                self._moving_roi,
                int(round(scene_pos.y() - dy)),
                int(round(scene_pos.x() - dx)),
            )
            self.window.draw_frame()
            return
        if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.rubberband.setGeometry(
                QtCore.QRect(self.roi_origin, event.pos())
            )
        if self.pan:
            self.hscrollbar.setValue(
                self.hscrollbar.value() - event.pos().x() + self.pan_start_x
            )
            self.vscrollbar.setValue(
                self.vscrollbar.value() - event.pos().y() + self.pan_start_y
            )
            self.pan_start_x = event.pos().x()
            self.pan_start_y = event.pos().y()
            self.window.draw_frame()
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Add the dragged ROI (clipping against existing ones) or stop
        panning the view."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._suppress_release:
                # trailing release of a double click that deleted a region
                self._suppress_release = False
                self.rubberband.hide()
                event.accept()
                return
            if self.split_fov_mode and self._moving_roi is not None:
                # finished dragging an existing region
                self._moving_roi = None
                self._move_anchor = None
                self.window.parameters_dialog.update_roi_display()
                self.window.draw_frame()
                return
            self.roi_end = QtCore.QPoint(event.pos())
            self.rubberband.hide()
            dx = abs(self.roi_end.x() - self.roi_origin.x())
            dy = abs(self.roi_end.y() - self.roi_origin.y())
            if self.split_fov_mode:
                self._release_split_fov_region(dx, dy)
                self.window.draw_frame()
                return
            if dx >= 10 and dy >= 10:
                roi_points = (
                    self.mapToScene(self.roi_origin),
                    self.mapToScene(self.roi_end),
                )
                new_roi = [[int(_.y()), int(_.x())] for _ in roi_points]
                box = self.window.parameters.get("Box Size", 7)
                self.rois = localize.clip_rois(
                    self.rois + [new_roi], min_size=box
                )
                self.window.parameters_dialog.update_roi_display()
            self.window.draw_frame()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self.pan = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            event.ignore()

    def _release_split_fov_region(self, dx: int, dy: int) -> None:
        """Finish a left-drag/click in split-FOV mode: the first region (a real
        drag) fixes the shared size; later releases drop a same-size region
        centered on the release point."""
        self.rubberband.hide()
        origin = self.mapToScene(self.roi_origin)
        end = self.mapToScene(self.roi_end)
        size = self._region_size()
        if size is None:
            # no regions yet: this drag defines the shared size (a real drag)
            if dx < 10 or dy < 10:
                return
            y0, y1 = sorted((int(origin.y()), int(end.y())))
            x0, x1 = sorted((int(origin.x()), int(end.x())))
            self.rois.append([[y0, x0], [y1, x1]])
            self.selected_roi = len(self.rois) - 1
            self.window.parameters_dialog.update_roi_display()
        else:
            h, w = size
            cy = (origin.y() + end.y()) / 2.0
            cx = (origin.x() + end.x()) / 2.0
            self._add_region(
                int(round(cy - h / 2.0)), int(round(cx - w / 2.0))
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Arrow keys nudge the selected split-FOV region by 1 px for
        pixel-precise channel registration."""
        deltas = {
            QtCore.Qt.Key.Key_Left: (0, -1),
            QtCore.Qt.Key.Key_Right: (0, 1),
            QtCore.Qt.Key.Key_Up: (-1, 0),
            QtCore.Qt.Key.Key_Down: (1, 0),
        }
        if (
            self.split_fov_mode
            and self.selected_roi is not None
            and self.selected_roi < len(self.rois)
            and event.key() in deltas
        ):
            idx = self.selected_roi
            (y0, x0), _ = self.rois[idx]
            ddy, ddx = deltas[event.key()]
            self._move_region(idx, y0 + ddy, x0 + ddx)
            self.window.parameters_dialog.update_roi_display()
            self.window.draw_frame()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Remove the ROI under the cursor on a (left) double click. If
        several ROIs contain the point, the smallest one is removed."""
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.rois:
            event.ignore()
            return
        scene_pos = self.mapToScene(event.pos())
        idx = self._roi_at(scene_pos.x(), scene_pos.y())
        if idx is None:
            event.ignore()
            return
        self.window.region_mngs()  # align the thresholds before deleting
        self.window.region_params()  # ... and the per-region fit settings
        del self.rois[idx]
        if idx < len(self.roi_mngs):
            del self.roi_mngs[idx]
        if idx < len(self.roi_params):
            del self.roi_params[idx]
        self.selected_roi = None
        # the release that closes this double click must not re-add a region
        self._suppress_release = True
        # In split-FOV mode the shared size is derived from the remaining
        # regions (see ``_region_size``), so removing all of them frees it.
        self.window.parameters_dialog.update_roi_display()
        self.window.draw_frame()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Zoom in/out with the mouse wheel, centered on the cursor."""
        scale = 1.008 ** (-event.angleDelta().y())
        self.window.zoom(scale, anchor=event.position().toPoint())

    def on_scroll(self) -> None:
        """Redraw the frame if scale bar is shown."""
        # draw_frame() rebuilds the scene, which can change the scrollbar
        # values and re-fire valueChanged; skip while a draw is in progress
        # to avoid unbounded re-entrant recursion.
        if self.window._drawing_frame:
            return
        if self.window.scalebar_action.isChecked():
            self.window.draw_frame()


class Scene(QtWidgets.QGraphicsScene):
    """Render individual frames, displayed in a ``View`` widget."""

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.window = window
        self.dragMoveEvent = self.dragEnterEvent

    def path_from_drop(self, event: QtGui.QDropEvent) -> tuple[str, str]:
        """Extract path of the dropped file."""
        url = event.mimeData().urls()[0]
        path = url.toLocalFile()
        base, extension = os.path.splitext(path)
        return path, extension

    def drop_has_valid_url(self, event: QtGui.QDropEvent) -> bool:
        """Check if the dropped file has a valid extension."""
        if not event.mimeData().hasUrls():
            return False
        path, extension = self.path_from_drop(event)

        if extension.lower() not in io.MOVIE_EXTENSIONS:
            return False
        return True

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept the file dragged over the widget if it has a valid
        extension."""
        if self.drop_has_valid_url(event):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Loads when dropped into the scene."""
        path, ext = self.path_from_drop(event)
        self.window.open(path)


def format_hover_tooltip(row: pd.Series) -> str:
    """Multi-line ``name: value`` text listing all columns of a
    localization / identification row, shown as a hover tooltip."""
    lines = []
    for name, value in row.items():
        if isinstance(value, (float, np.floating)):
            value = f"{value:.6g}"
        lines.append(f"{name}: {value}")
    return "\n".join(lines)


class FitMarker(QtWidgets.QGraphicsItemGroup):
    """Marker showing fitted position."""

    def __init__(
        self,
        x: float,
        y: float,
        size: float,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        L = size / 2
        line1 = QtWidgets.QGraphicsLineItem(x - L, y - L, x + L, y + L)
        line1.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line1)
        line2 = QtWidgets.QGraphicsLineItem(x - L, y + L, x + L, y - L)
        line2.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
        self.addToGroup(line2)


class OddSpinBox(QtWidgets.QSpinBox):
    """Spinbox allowing only odd numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSingleStep(2)
        self.editingFinished.connect(self.on_editing_finished)

    def on_editing_finished(self):
        value = self.value()
        if value % 2 == 0:
            self.setValue(int(value + 1))


class CamSettingComboBox(QtWidgets.QComboBox):
    """Combo box for selecting camera settings which are relevant for
    sensitivity.

    Datasheets for different camera models specify sensitivity at
    different degrees of granularity: Some only specify one overall
    sensitivity, while for others, the sensitivity depends on the
    readout mode (faster readout leads to lower sensitivity), while
    others again have nested dependencies (e.g. depending on both
    readout rate and dynamic range). The sensitivity information is
    saved in picasso.CONFIG. The aspects the sensitivity depends on are
    termed 'Sensitivity Categories', and are listed for each camera in
    CONFIG (if applicable). Another entry for each camera,
    'Sensitivity', specifies the sensitivity as a scalar, a simple
    dict, or a nested dict, depending on the applicable sensitivity
    categories. The keys in the nested dict are the potential values of
    the respective sensitivity categories at that index of nesting.

    An example for a nested case (Andor Zyla):
        Sensitivity Categories:
          - PixelReadoutRate
          - Sensitivity/DynamicRange
        Sensitivity:
          540 MHz - fastest readout:
            12-bit (high well capacity): 7.98
            12-bit (low noise): 0.26
            16-bit (low noise & high well capacity): 0.51
          200 MHz - lowest noise:
            12-bit (high well capacity): 8.2
            12-bit (low noise): 0.24
            16-bit (low noise & high well capacity): 0.53

    This ``CamSettingComboBox`` class allows for selecting the value of
    one sensitivity category (described by its index in the list
    "Sensitivity Categories"). If the user changes the value of the
    ``CamSettingComboBox``, the entries of the lower levels of
    sensitivity categories (potentially) need to be adapted. Therefore,
    this ``CamSettingComboBox`` holds the ``CamSettingComboBoxDict``
    ``cam_combos``, which is a ``CamSettingComboBoxDict`` with
    references to the ``CamSettingComboBox``'s of all sensitivity
    category indices. This way the changed ``CamSettingComboBox`` can
    trigger the next-level ``CamSettingComboBox`` to adapt its options.

    ...

    Attributes
    ----------
    cam_combos : dict
        keys: Available cameras.

        values: list of CamSettingComboBoxes
            one for each sensitivity category, described in the CONFIG
            entry for the respective camera.
    camera : str
        Camera name this CamSettingComboBox belongs to.
    categories : list of str
        Sensitivity categories of the camera.
    index : int
        Index of sensitivity category this CamSettingComboBox belongs
        to.
    """

    def __init__(
        self,
        cam_combos: dict,
        camera: str,
        index: int,
        sensitivity_categories: list[str] = [],
    ) -> None:
        super().__init__()
        self.cam_combos = cam_combos
        self.camera = camera
        self.index = index
        self.categories = sensitivity_categories

    def change_target_choices(self, index: int) -> None:
        """Update the target choices based on the selected camera
        settings."""
        cam_combos = self.cam_combos[self.camera]
        sensitivity = CONFIG["Cameras"][self.camera]["Sensitivity"]
        for i in range(self.index + 1):
            sensitivity = sensitivity[cam_combos[i].currentText()]
        if len(cam_combos) > self.index + 1:
            target = cam_combos[self.index + 1]
            target.blockSignals(True)
            target.clear()
            target.blockSignals(False)
            target.addItems(sorted(list(sensitivity.keys())))


class CamSettingComboBoxDict(UserDict):
    """Dictionary holding ``CamSettingComboBox``'s for different cameras
    and sensitivity categories.

    keys: str
        Camera names.
    values: list of CamSettingComboBoxes
        one for each sensitivity category of this camera.

    Attributes
    ----------
    sensitivity_categories : dict
        keys: str
            Camera names.
        values: list of str
            Sensitivity categories.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sensitivity_categories = {}

    def add_categories(self, cam: str, categories: list[str]) -> None:
        """Call this when setting combo boxes for a new camera, to
        accompany it with the corresponding sensitivity categories."""
        self.sensitivity_categories[cam] = categories

    def set_camcombo_value(self, cam: str, category: str, value: str) -> None:
        """Set the selected value of one combo box.

        Parameters
        ----------
        cam : str
            Camera name to set.
        category : str
            Category combo box to set.
        value : str
            Value to set.
        """
        cat_idx = self.sensitivity_categories[cam].index(category)
        cam_combo = self.data[cam][cat_idx]
        for index in range(cam_combo.count()):
            if cam_combo.itemText(index) == value:
                cam_combo.setCurrentIndex(index)
                break

    def set_camcombo_values(self, cam: str, values: dict) -> None:
        """Set the values of all combo boxes of a camera.

        Parameters
        ----------
        cam : str
            Camera name to set.
        values : dict
            Maps a sensitivity category to the value to set for it.
        """
        for i, cat in enumerate(self.sensitivity_categories[cam]):
            if cat in values:
                cam_combo = self.data[cam][i]
                for index in range(cam_combo.count()):
                    if cam_combo.itemText(index) == values[cat]:
                        cam_combo.setCurrentIndex(index)
                        break


class EmissionComboBoxDict(UserDict):
    """Dictionary holding ``QComboBox``'s for different cameras,
    each having the potential emission wavelengths as options.
    The ComboBox is only shown if the quantum efficiency is
    given in the CONFIG, otherwise it is irrelevant for localizing.

    keys: str
        Camera names.
    values: QtWidgets.QComboBox
        Wavelengths.
    """

    def __init__(self):
        super().__init__()

    def set_emcombo_value(self, cam: str, wavelength: str):
        """Sets the selected value of one combo box

        Parameters
        ----------
        cam : str
            Camera name to set.
        wavelength : str
            Wavelength to set.
        """
        em_combo = self.data[cam]
        for index in range(em_combo.count()):
            if em_combo.itemText(index) == wavelength:
                em_combo.setCurrentIndex(index)
                break


class PromptInfoDialog(lib.Dialog):
    """Enter movie metadata.

    ...

    Attributes
    ----------
    byte_order : QtWidgets.QComboBox
        Combo box for selecting byte order (little or big endian).
    buttons : QtWidgets.QDialogButtonBox
        Button box for the dialog (OK/Cancel).
    dtype : QtWidgets.QComboBox
        Combo box for selecting data type (float/int, number of bytes).
    frames : QtWidgets.QSpinBox
        Spin box for selecting the number of frames.
    movie_height, movie_width : QtWidgets.QSpinBox
        Spin boxes for selecting the height and width of the movie.
    save : QtWidgets.QCheckBox
        Check box for selecting whether to save the info to a YAML file.
    window : QtWidgets.QWidget
        The parent window for the dialog.
    """

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter movie info")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Byte Order:"), 0, 0)
        self.byte_order = QtWidgets.QComboBox()
        self.byte_order.addItems(
            ["Little Endian (loads faster)", "Big Endian"]
        )
        grid.addWidget(self.byte_order, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Data Type:"), 1, 0)
        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(
            [
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "uint8",
                "uint16",
                "uint32",
            ]
        )
        grid.addWidget(self.dtype, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Frames:"), 2, 0)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, int(1e9))
        grid.addWidget(self.frames, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Height:"), 3, 0)
        self.movie_height = QtWidgets.QSpinBox()
        self.movie_height.setRange(1, int(1e9))
        grid.addWidget(self.movie_height, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Width"), 4, 0)
        self.movie_width = QtWidgets.QSpinBox()
        self.movie_width.setRange(1, int(1e9))
        grid.addWidget(self.movie_width, 4, 1)
        self.save = QtWidgets.QCheckBox("Save info to yaml file")
        self.save.setChecked(True)
        grid.addWidget(self.save, 5, 0, 1, 2)
        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
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

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getMovieSpecs(
        parent: QtWidgets.QWidget | None = None,
    ) -> tuple[dict, bool, bool]:
        dialog = PromptInfoDialog(parent)
        result = dialog.exec()
        info = {}
        info["Byte Order"] = (
            ">" if dialog.byte_order.currentText() == "Big Endian" else "<"
        )
        info["Data Type"] = dialog.dtype.currentText()
        info["Frames"] = dialog.frames.value()
        info["Height"] = dialog.movie_height.value()
        info["Width"] = dialog.movie_width.value()
        save = dialog.save.isChecked()
        return info, save, result == QtWidgets.QDialog.DialogCode.Accepted


class PromptMovieInfoDialog(lib.Dialog):
    """Enter movie metadata manually when it cannot be read from the
    file.

    Used as a fallback for ``.tif``/``.stk``/``.nd2`` movies whose pixel
    data could be opened but whose embedded metadata could not be
    parsed. The dialog is pre-filled with whatever dimensions could
    still be read from the file structure; the user supplies the rest
    (notably the pixel size). See ``docs/files.rst`` for the required
    metadata keys (Width, Height, Frames, Pixelsize).

    Attributes
    ----------
    frames : QtWidgets.QSpinBox
        Spin box for the number of frames.
    movie_height, movie_width : QtWidgets.QSpinBox
        Spin boxes for the height and width of the movie.
    pixelsize : QtWidgets.QSpinBox
        Spin box for the effective camera pixel size in nm.
    save : QtWidgets.QCheckBox
        Whether to save the entered metadata to a YAML file next to the
        movie, so it is reused the next time the movie is opened.
    """

    def __init__(
        self,
        window: QtWidgets.QWidget,
        partial_info: dict | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Enter movie info")
        partial_info = partial_info or {}
        vbox = QtWidgets.QVBoxLayout(self)
        message = QtWidgets.QLabel(
            "The movie metadata could not be read from the file.\n"
            "Please enter the required information manually."
        )
        message.setWordWrap(True)
        vbox.addWidget(message)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Frames:"), 0, 0)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(1, int(1e9))
        self.frames.setValue(int(partial_info.get("Frames", 1)))
        grid.addWidget(self.frames, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Height:"), 1, 0)
        self.movie_height = QtWidgets.QSpinBox()
        self.movie_height.setRange(1, int(1e9))
        self.movie_height.setValue(int(partial_info.get("Height", 1)))
        grid.addWidget(self.movie_height, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Width:"), 2, 0)
        self.movie_width = QtWidgets.QSpinBox()
        self.movie_width.setRange(1, int(1e9))
        self.movie_width.setValue(int(partial_info.get("Width", 1)))
        grid.addWidget(self.movie_width, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Pixel size (nm):"), 3, 0)
        self.pixelsize = QtWidgets.QSpinBox()
        self.pixelsize.setRange(1, 10000)
        self.pixelsize.setValue(int(partial_info.get("Pixelsize", 130)))
        grid.addWidget(self.pixelsize, 3, 1)
        self.save = QtWidgets.QCheckBox("Save info to yaml file")
        self.save.setChecked(True)
        grid.addWidget(self.save, 4, 0, 1, 2)
        vbox.addLayout(grid)
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
    def getMovieSpecs(
        parent: QtWidgets.QWidget | None = None,
        partial_info: dict | None = None,
    ) -> tuple[dict, bool, bool]:
        dialog = PromptMovieInfoDialog(parent, partial_info)
        result = dialog.exec()
        # Preserve any extra readable keys (e.g. "File") already present.
        info = dict(partial_info or {})
        info["Frames"] = dialog.frames.value()
        info["Height"] = dialog.movie_height.value()
        info["Width"] = dialog.movie_width.value()
        info["Pixelsize"] = dialog.pixelsize.value()
        info["Generated by"] = "Picasso Localize (manual metadata)"
        save = dialog.save.isChecked()
        return info, save, result == QtWidgets.QDialog.DialogCode.Accepted


class PromptChannelDialog(lib.Dialog):
    """Dialog for selecting a channel. Used for .IMS files."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Select channel")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Channel:"), 0, 0)
        self.byte_order = QtWidgets.QComboBox()

        grid.addWidget(self.byte_order, 0, 1)

        vbox.addLayout(grid)
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
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

    # static method to create the dialog and return (date, time, accepted)
    @staticmethod
    def getMovieSpecs(
        parent: QtWidgets.QWidget | None = None,
        channels: list[str] | None = None,
    ) -> tuple[dict, bool, bool]:
        dialog = PromptChannelDialog(parent)
        dialog.byte_order.addItems(channels)
        result = dialog.exec()
        channel = dialog.byte_order.currentText()
        return channel, result == QtWidgets.QDialog.DialogCode.Accepted


class ConcatenateMoviesDialog(lib.Dialog):
    """Dialog for reviewing and ordering the TIFF files that are opened
    as one concatenated movie.

    The files are found automatically (see ``io.find_tif_movies``) and
    listed in the order their frames will run. That order is sorted by
    folder and file name, which is usually - but not always - the
    acquisition order, and getting it wrong is only noticed after the
    movie has been localized. So the list is shown for confirmation and
    can be reordered by dragging or with the buttons, and files can be
    removed or added from elsewhere.
    """

    def __init__(
        self, window: QtWidgets.QWidget, paths: list[str], root: str = ""
    ) -> None:
        super().__init__(window)
        self.window = window
        self.root = root
        self.setWindowTitle("Concatenate movies")
        self.resize(700, 400)
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(
            QtWidgets.QLabel(
                "The files below are opened as one movie, with their "
                "frames in this order.\nDrag to reorder."
            )
        )

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove
        )
        for path in paths:
            self._add_path(path)
        hbox.addWidget(self.file_list)

        buttons_vbox = QtWidgets.QVBoxLayout()
        hbox.addLayout(buttons_vbox)
        for label, slot in (
            ("Move up", self.move_up),
            ("Move down", self.move_down),
            ("Remove", self.remove_selected),
            ("Add files...", self.add_files),
        ):
            button = QtWidgets.QPushButton(label)
            button.setAutoDefault(False)
            button.clicked.connect(slot)
            buttons_vbox.addWidget(button)
        buttons_vbox.addStretch(1)

        self.count_label = QtWidgets.QLabel()
        vbox.addWidget(self.count_label)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.file_list.model().rowsInserted.connect(self._update_count)
        self.file_list.model().rowsRemoved.connect(self._update_count)
        self._update_count()

    def _add_path(self, path: str) -> None:
        """Append ``path``, shown relative to the searched folder (full
        paths are too long to compare at a glance) with the absolute path
        as the tooltip."""
        display = path
        if self.root:
            try:
                display = os.path.relpath(path, self.root)
            except ValueError:  # different drive on Windows
                pass
        item = QtWidgets.QListWidgetItem(display)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.file_list.addItem(item)

    def _update_count(self) -> None:
        n = self.file_list.count()
        self.count_label.setText(f"{n} file(s)")
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setEnabled(n > 0)

    def _move(self, offset: int) -> None:
        """Move the selected rows by ``offset``, keeping them selected."""
        rows = sorted(
            self.file_list.row(item) for item in self.file_list.selectedItems()
        )
        if not rows:
            return
        if offset < 0 and rows[0] == 0:
            return
        if offset > 0 and rows[-1] == self.file_list.count() - 1:
            return
        for row in rows if offset < 0 else reversed(rows):
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row + offset, item)
            item.setSelected(True)

    def move_up(self) -> None:
        self._move(-1)

    def move_down(self) -> None:
        self._move(1)

    def remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def add_files(self) -> None:
        """Append files from elsewhere, e.g. a folder that the automatic
        search did not cover."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add movies to concatenate",
            directory=self.root or None,
            filter="TIFF (%s)" % " ".join("*" + e for e in io.TIFF_EXTENSIONS),
        )
        existing = self.paths()
        for path in paths:
            path = os.path.abspath(path)
            if path not in existing:
                self._add_path(path)

    def paths(self) -> list[str]:
        """The absolute file paths in their current order."""
        return [
            self.file_list.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
            for row in range(self.file_list.count())
        ]

    @staticmethod
    def getPaths(
        parent: QtWidgets.QWidget,
        paths: list[str],
        root: str = "",
    ) -> tuple[list[str], bool]:
        dialog = ConcatenateMoviesDialog(parent, paths, root=root)
        result = dialog.exec()
        return (
            dialog.paths(),
            result == QtWidgets.QDialog.DialogCode.Accepted,
        )


class Calibrate3DDialog(lib.Dialog):
    """Dialog for entering the parameters of a 3D (astigmatism)
    calibration: the z step size, the number of frames acquired per z
    (stage) position and, if more than one, the order in which those
    frames were acquired."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("3D Calibration")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        # Step size
        grid.addWidget(QtWidgets.QLabel("Calibration step size (nm):"), 0, 0)
        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(0.01, 1e6)
        self.step.setDecimals(2)
        self.step.setValue(5)
        grid.addWidget(self.step, 0, 1)

        # Number of frames per z (stage) position
        frames_label = QtWidgets.QLabel("Number of frames per step size:")
        frames_label.setToolTip(
            "Number of frames acquired at each z (stage) position.\n"
            "Acquiring several frames per position increases the number\n"
            "of localizations per position and thus the confidence of\n"
            "the calibration fit."
        )
        grid.addWidget(frames_label, 1, 0)
        self.frames_per_step = QtWidgets.QSpinBox()
        self.frames_per_step.setRange(1, 1000)
        self.frames_per_step.setValue(1)
        grid.addWidget(self.frames_per_step, 1, 1)

        # Frame order (only relevant when frames_per_step > 1)
        self.order_label = QtWidgets.QLabel("Frame order:")
        grid.addWidget(self.order_label, 2, 0)
        self.frame_order = QtWidgets.QComboBox()
        self.frame_order.addItem("Different FOVs first", userData="fov")
        self.frame_order.addItem("Different z positions first", userData="z")
        self.frame_order.setToolTip(
            "Order in which the frames were acquired when more than one\n"
            "frame per step size is used.\n"
            "'Same z position, different FOVs': the z position is held\n"
            "constant while several fields of view are imaged, i.e.,\n"
            "consecutive frames share the same z position.\n"
            "'Same FOV, z positions sequentially': the full z stack is\n"
            "scanned and then repeated, i.e., frames cycle through all z\n"
            "positions."
        )
        grid.addWidget(self.frame_order, 2, 1)

        # The order only matters when more than one frame per step
        self.frames_per_step.valueChanged.connect(self._update_order_enabled)
        self._update_order_enabled(self.frames_per_step.value())

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

    def _update_order_enabled(self, n_frames: int) -> None:
        """Enable the frame order choice only if more than one frame is
        acquired per z position."""
        enabled = n_frames > 1
        self.order_label.setEnabled(enabled)
        self.frame_order.setEnabled(enabled)

    @staticmethod
    def getCalibrationSpecs(
        parent: QtWidgets.QWidget | None = None,
    ) -> tuple[float, int, str, bool]:
        """Show the dialog and return the chosen step size, number of
        frames per step, frame order and whether the dialog was
        accepted."""
        dialog = Calibrate3DDialog(parent)
        result = dialog.exec()
        step = dialog.step.value()
        frames_per_step = dialog.frames_per_step.value()
        frame_order = dialog.frame_order.currentData()
        accepted = result == QtWidgets.QDialog.DialogCode.Accepted
        return step, frames_per_step, frame_order, accepted


class CameraCalibrationDialog(lib.Dialog):
    """Collect every input of an sCMOS camera characterization up front.

    A chain of bare file dialogs is a poor way to ask for three different
    things: on macOS the picker shows only its own title bar, so "which file
    am I choosing now?" has no answer once the dialog is open. Here the whole
    job is visible at once - which movie is the dark one, which are the bright
    ones, where the result goes - and each file picker is opened from a row
    that already says what it is for.
    """

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("sCMOS Camera Calibration")
        self.setMinimumWidth(620)
        vbox = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Characterize the per-pixel offset, readout variance and "
            "amplification gain of an sCMOS sensor, following Huang et al., "
            "Nature Methods 10:653 (2013).\n\n"
            "The calibration is only valid for the readout mode, bit depth "
            "and gain setting it was acquired in, so disable any automatic "
            "hot-pixel correction the camera offers before recording."
        )
        intro.setWordWrap(True)
        vbox.addWidget(intro)

        dark_group = QtWidgets.QGroupBox("1. Dark movie (required)")
        vbox.addWidget(dark_group)
        dark_layout = QtWidgets.QVBoxLayout(dark_group)
        dark_hint = QtWidgets.QLabel(
            "Frames recorded with no light reaching the sensor (cap on the "
            f"camera). Gives the offset and variance maps. At least "
            f"{scmos.MIN_DARK_FRAMES} frames; "
            f"{scmos.RECOMMENDED_DARK_FRAMES:,} or more is recommended, "
            "Huang et al. used 60,000."
        )
        dark_hint.setWordWrap(True)
        dark_layout.addWidget(dark_hint)
        dark_row = QtWidgets.QHBoxLayout()
        dark_layout.addLayout(dark_row)
        self.dark_edit = QtWidgets.QLineEdit()
        self.dark_edit.setPlaceholderText("no dark movie selected")
        self.dark_edit.setReadOnly(True)
        dark_row.addWidget(self.dark_edit)
        dark_button = QtWidgets.QPushButton("Browse...")
        dark_button.setAutoDefault(False)
        dark_button.clicked.connect(self.browse_dark)
        dark_row.addWidget(dark_button)

        bright_group = QtWidgets.QGroupBox(
            "2. Bright movies for the gain map (optional)"
        )
        vbox.addWidget(bright_group)
        bright_layout = QtWidgets.QVBoxLayout(bright_group)
        bright_hint = QtWidgets.QLabel(
            "One movie per illumination level, each uniformly illuminated at "
            "a different intensity (Huang et al. used 15 levels spanning "
            "20-200 photons per pixel). Without them there is no gain map "
            "and the scalar Sensitivity keeps being used. The movies do not "
            "need to be ordered by intensity.\n\n"
            "The illumination column is optional and changes nothing in the "
            "calibration: it lets the diagnostic plot show the response "
            "against the laser power (or exposure time) that was actually "
            "set and read linearity off it when the levels are not evenly "
            "spaced. Fill in every row or none."
        )
        bright_hint.setWordWrap(True)
        bright_layout.addWidget(bright_hint)
        self.bright_list = QtWidgets.QTableWidget(0, 2)
        self.bright_list.setHorizontalHeaderLabels(
            ["Movie", "Illumination (optional)"]
        )
        self.bright_list.verticalHeader().setVisible(False)
        self.bright_list.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.bright_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        header = self.bright_list.horizontalHeader()
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.bright_list.setMaximumHeight(130)
        bright_layout.addWidget(self.bright_list)
        bright_row = QtWidgets.QHBoxLayout()
        bright_layout.addLayout(bright_row)
        bright_row.addWidget(QtWidgets.QLabel("Illumination unit:"))
        self.unit_edit = QtWidgets.QLineEdit("mW")
        self.unit_edit.setMaximumWidth(70)
        bright_row.addWidget(self.unit_edit)
        bright_row.addStretch(1)
        add_button = QtWidgets.QPushButton("Add...")
        add_button.setAutoDefault(False)
        add_button.clicked.connect(self.browse_bright)
        bright_row.addWidget(add_button)
        remove_button = QtWidgets.QPushButton("Remove selected")
        remove_button.setAutoDefault(False)
        remove_button.clicked.connect(self.remove_bright)
        bright_row.addWidget(remove_button)

        out_group = QtWidgets.QGroupBox("3. Save the calibration to")
        vbox.addWidget(out_group)
        out_row = QtWidgets.QHBoxLayout(out_group)
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("no output file selected")
        out_row.addWidget(self.out_edit)
        out_button = QtWidgets.QPushButton("Browse...")
        out_button.setAutoDefault(False)
        out_button.clicked.connect(self.browse_out)
        out_row.addWidget(out_button)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        vbox.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.dark_edit.textChanged.connect(self._update_ok_enabled)
        self.out_edit.textChanged.connect(self._update_ok_enabled)
        self._update_ok_enabled()

    @staticmethod
    def _movie_filter() -> str:
        return "Movies (%s)" % " ".join(
            "*" + extension for extension in io.MOVIE_EXTENSIONS
        )

    def _update_ok_enabled(self) -> None:
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setEnabled(
            bool(self.dark_edit.text()) and bool(self.out_edit.text())
        )

    def browse_dark(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select the dark movie",
            directory=os.path.dirname(self.dark_edit.text()) or None,
            filter=self._movie_filter(),
        )
        if not path:
            return
        self.dark_edit.setText(path)
        if not self.out_edit.text():
            base, _ = os.path.splitext(path)
            self.out_edit.setText(base + "_scmos_calib.hdf5")

    def browse_bright(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select the bright (uniformly illuminated) movies",
            directory=os.path.dirname(self.dark_edit.text()) or None,
            filter=self._movie_filter(),
        )
        existing = self.bright_paths()
        for path in paths:
            if path not in existing:
                self.add_bright(path)

    def add_bright(self, path: str, illumination: str = "") -> None:
        """Append one bright movie row; its illumination cell stays editable
        while the path cell does not, since the path came from a picker."""
        row = self.bright_list.rowCount()
        self.bright_list.insertRow(row)
        item = QtWidgets.QTableWidgetItem(path)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        self.bright_list.setItem(row, 0, item)
        self.bright_list.setItem(
            row, 1, QtWidgets.QTableWidgetItem(illumination)
        )

    def remove_bright(self) -> None:
        # Descending, so removing one row does not renumber the rest.
        rows = sorted(
            {index.row() for index in self.bright_list.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.bright_list.removeRow(row)

    def browse_out(self) -> None:
        base, _ = os.path.splitext(self.dark_edit.text())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save the sCMOS camera calibration as",
            self.out_edit.text() or (base and base + "_scmos_calib.hdf5"),
            filter="*.hdf5",
        )
        if path:
            self.out_edit.setText(path)

    def bright_paths(self) -> list[str]:
        return [
            self.bright_list.item(row, 0).text()
            for row in range(self.bright_list.rowCount())
        ]

    def bright_levels(self) -> list[float] | None:
        """The illumination of each bright movie, or None if the column was
        left empty.

        Raises
        ------
        ValueError
            If some rows carry a level and others do not, or if one of them
            is not a number. A partly filled column cannot be plotted against
            and is more likely a slip than an intention.
        """
        texts = [
            (
                self.bright_list.item(row, 1).text().strip()
                if self.bright_list.item(row, 1) is not None
                else ""
            )
            for row in range(self.bright_list.rowCount())
        ]
        filled = [text for text in texts if text]
        if not filled:
            return None
        if len(filled) != len(texts):
            raise ValueError(
                f"Only {len(filled)} of {len(texts)} bright movies have an "
                "illumination. Fill in every row or leave the column empty."
            )
        try:
            return [float(text) for text in texts]
        except ValueError:
            raise ValueError(
                "The illumination column must contain numbers only "
                f"(got {', '.join(repr(t) for t in texts)})."
            ) from None

    @staticmethod
    def getCalibrationSpecs(
        parent: QtWidgets.QWidget | None = None,
    ) -> tuple[str, list[str], list[float] | None, str, str, bool]:
        """Show the dialog and return the dark movie, the bright movies,
        their illumination levels and unit, the output path and whether it
        was accepted.

        Re-shown, with the offending cells kept, if the illumination column
        was filled in only partly or not with numbers."""
        dialog = CameraCalibrationDialog(parent)
        while True:
            result = dialog.exec()
            accepted = result == QtWidgets.QDialog.DialogCode.Accepted
            try:
                levels = dialog.bright_levels()
            except ValueError as error:
                if not accepted:
                    levels = None
                    break
                QtWidgets.QMessageBox.warning(
                    parent, "Camera calibration", str(error)
                )
                continue
            break
        return (
            dialog.dark_edit.text(),
            dialog.bright_paths(),
            levels,
            dialog.unit_edit.text().strip(),
            dialog.out_edit.text(),
            accepted,
        )


class CalibrateSplineDialog(lib.Dialog):
    """Dialog for entering the parameters of a cubic-spline PSF calibration
    built from a bead z-stack: the z step size, the number of frames acquired
    per z (stage) position, the acquisition order of those frames, and whether
    to build a 3D (z-recovering) or 2D (single-plane) spline PSF. The box size
    and minimum net gradient are taken from the main parameters."""

    def __init__(
        self, window: QtWidgets.QWidget, multichannel: bool = False
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Spline PSF Calibration")
        vbox = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        # Step size
        grid.addWidget(QtWidgets.QLabel("Calibration step size (nm):"), 0, 0)
        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(0.01, 1e6)
        self.step.setDecimals(2)
        self.step.setValue(5)
        grid.addWidget(self.step, 0, 1)

        # Number of frames per z (stage) position
        frames_label = QtWidgets.QLabel("Number of frames per step size:")
        frames_label.setToolTip(
            "Number of frames acquired at each z (stage) position.\n"
            "Acquiring several frames per position (e.g., different fields\n"
            "of view) increases the number of beads averaged into the PSF."
        )
        grid.addWidget(frames_label, 1, 0)
        self.frames_per_step = QtWidgets.QSpinBox()
        self.frames_per_step.setRange(1, 1000)
        self.frames_per_step.setValue(1)
        grid.addWidget(self.frames_per_step, 1, 1)

        # Frame order (only relevant when frames_per_step > 1)
        self.order_label = QtWidgets.QLabel("Frame order:")
        grid.addWidget(self.order_label, 2, 0)
        self.frame_order = QtWidgets.QComboBox()
        self.frame_order.addItem("Different FOVs first", userData="fov")
        self.frame_order.addItem("Different z positions first", userData="z")
        self.frame_order.setToolTip(
            "Order in which the frames were acquired when more than one\n"
            "frame per step size is used (see the 3D astigmatism dialog)."
        )
        grid.addWidget(self.frame_order, 2, 1)

        # Spline PSF dimensionality / model
        grid.addWidget(QtWidgets.QLabel("Spline PSF model:"), 3, 0)
        self.model = QtWidgets.QComboBox()
        self.model.addItem("3D (recovers z)", userData="spline-3d")
        self.model.addItem("2D (single plane)", userData="spline-2d")
        self.model.setToolTip(
            "3D / 2D: single- or multichannel cubic-spline PSF."
        )
        grid.addWidget(self.model, 3, 1)

        # Magnification factor (applied to the fitted z, as in astigmatism)
        magnification_label = QtWidgets.QLabel("Magnification factor:")
        magnification_label.setToolTip(
            "Factor used to correct for z-position abberation due to\n"
            "refractive index mismatch, see Huang B, et al. Science. 2008."
        )
        grid.addWidget(magnification_label, 4, 0)
        self.magnification_factor = QtWidgets.QDoubleSpinBox()
        self.magnification_factor.setRange(0, 1e6)
        self.magnification_factor.setDecimals(4)
        self.magnification_factor.setValue(0.79)
        grid.addWidget(self.magnification_factor, 4, 1)

        # Optional z-bias correction (astigmatism)
        self.correct_z_bias = QtWidgets.QCheckBox(
            "Set z = 0 at max. intensity"
        )
        self.correct_z_bias.setToolTip(
            "Define z = 0 at the axial intensity peak of the averaged PSF,\n"
            "correcting a potential z bias in the raw stage scan.\n"
            "Only meaningful for a PSF with a single,\n"
            "well-defined intensity focus (e.g. astigmatism)."
        )
        self.correct_z_bias.setChecked(False)
        grid.addWidget(self.correct_z_bias, 5, 0, 1, 2)

        # Multichannel (2- to 6-channel) default fit mode. Stored in the
        # calibration. Only shown for a multichannel build (several channels
        # loaded or split-FOV mode on); meaningless for single-channel data.
        self.link_photons = QtWidgets.QCheckBox(
            "Link photon counts across channels"
        )
        self.link_photons.setToolTip(LINK_PHOTONS_TIP)
        self.link_photons.setChecked(True)
        self.link_photons.setVisible(multichannel)
        grid.addWidget(self.link_photons, 6, 0, 1, 2)

        # How the channels are registered to the reference. Multichannel only:
        # a single-channel calibration has nothing to register.
        self.registration_label = QtWidgets.QLabel("Channel registration:")
        self.registration_model = _transform_model_combo()
        self.registration_label.setVisible(multichannel)
        self.registration_model.setVisible(multichannel)
        grid.addWidget(self.registration_label, 7, 0)
        grid.addWidget(self.registration_model, 7, 1)

        self.frames_per_step.valueChanged.connect(self._update_order_enabled)
        self._update_order_enabled(self.frames_per_step.value())

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

    def _update_order_enabled(self, n_frames: int) -> None:
        """Enable the frame order choice only if more than one frame is
        acquired per z position."""
        enabled = n_frames > 1
        self.order_label.setEnabled(enabled)
        self.frame_order.setEnabled(enabled)

    @staticmethod
    def getCalibrationSpecs(
        parent: QtWidgets.QWidget | None = None,
        multichannel: bool = False,
    ) -> tuple[float, int, str, str, float, bool, bool, str, int | None, bool]:
        """Show the dialog and return the chosen step size, number of frames
        per step, frame order, spline model, magnification factor, whether to
        correct the z bias, whether to link photons across channels, the
        channel-registration model, and whether it was accepted. ``multichannel`` shows the multichannel-only options (link
        photons, registration model); they are hidden for a single-channel
        calibration."""
        dialog = CalibrateSplineDialog(parent, multichannel=multichannel)
        result = dialog.exec()
        step = dialog.step.value()
        frames_per_step = dialog.frames_per_step.value()
        frame_order = dialog.frame_order.currentData()
        model = dialog.model.currentData()
        magnification_factor = dialog.magnification_factor.value()
        correct_z_bias = dialog.correct_z_bias.isChecked()
        link_photons = dialog.link_photons.isChecked()
        registration = _transform_model_of(dialog.registration_model)
        accepted = result == QtWidgets.QDialog.DialogCode.Accepted
        return (
            step,
            frames_per_step,
            frame_order,
            model,
            magnification_factor,
            correct_z_bias,
            link_photons,
            registration,
            accepted,
        )


class BeadInspectionDialog(lib.Dialog):
    """Gallery of the individual beads a spline PSF calibration was built
    from, showing which ones were averaged into the PSF and which were
    rejected as outliers.

    The calibration only averages beads whose shape agrees with the running
    average and discards the rest (see :func:`picasso.spline._keep_inliers`),
    which is a decision worth looking at: a bead can be dropped because it is
    a doublet, an aggregate or out of focus - or because the sample really
    does have a field-dependent PSF, in which case the calibration is being
    built from a biased subset. Each bead is drawn as its xy slice at focus
    plus the two cross-sections, next to the averaged PSF it was compared
    against, and rejected beads are framed in red and annotated with the
    criterion they failed and their position in the movie.
    """

    def __init__(
        self,
        diagnostics: list[dict],
        parent: QtWidgets.QWidget | None = None,
        title: str = "",
    ) -> None:
        super().__init__(parent)
        # Imported here (not at module import time) so the GUI only pays for
        # the Qt matplotlib backend when the inspector is actually opened.
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self.diagnostics = [d for d in diagnostics if d]
        self.setWindowTitle(
            "Calibration beads" + (f" - {title}" if title else "")
        )
        self.setModal(False)
        vbox = QtWidgets.QVBoxLayout(self)

        controls = QtWidgets.QHBoxLayout()
        vbox.addLayout(controls)
        self.channel = QtWidgets.QComboBox()
        for c in range(len(self.diagnostics)):
            self.channel.addItem(
                "Reference channel" if c == 0 else f"Channel {c}"
            )
        self.channel.setCurrentIndex(0)
        self.channel.currentIndexChanged.connect(self.draw)
        if len(self.diagnostics) > 1:
            controls.addWidget(QtWidgets.QLabel("Channel:"))
            controls.addWidget(self.channel)
        self.only_rejected = QtWidgets.QCheckBox("Show only rejected beads")
        self.only_rejected.setToolTip(
            "Hide the beads that were averaged into the PSF, keeping only\n"
            "the ones the calibration discarded as outliers."
        )
        self.only_rejected.stateChanged.connect(self.draw)
        controls.addWidget(self.only_rejected)
        controls.addWidget(QtWidgets.QLabel("Max. beads shown:"))
        # a z-stack can hold hundreds of beads; drawing them all is slow and
        # unreadable, so only the kept ones are capped (every rejected bead is
        # always drawn, since those are what is being checked)
        self.max_beads = QtWidgets.QSpinBox()
        self.max_beads.setRange(1, 1000)
        self.max_beads.setValue(60)
        self.max_beads.setToolTip(
            "How many beads to draw. Rejected beads are always shown; the\n"
            "limit only caps how many of the kept beads are drawn alongside."
        )
        self.max_beads.valueChanged.connect(self.draw)
        controls.addWidget(self.max_beads)
        controls.addWidget(QtWidgets.QLabel("Zoom:"))
        # The gallery is taller than any window, so it is scrolled rather than
        # squeezed. "Fit" scales it to the window width, which keeps the
        # scrolling one-dimensional; the percentages are there to enlarge a
        # cell that is too small to judge.
        self.zoom = QtWidgets.QComboBox()
        self.zoom.addItem("Fit width", userData=None)
        for percent in (75, 100, 150, 200, 300):
            self.zoom.addItem(f"{percent} %", userData=percent)
        self.zoom.setToolTip(
            "Size of the gallery. 'Fit width' scales it to the window, so\n"
            "only vertical scrolling is needed; a fixed zoom lets you look\n"
            "at a bead more closely and scroll in both directions."
        )
        self.zoom.currentIndexChanged.connect(self._fit_canvas)
        controls.addWidget(self.zoom)
        controls.addStretch(1)
        self.summary = QtWidgets.QLabel("")
        controls.addWidget(self.summary)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        # a matplotlib canvas consumes wheel events (it turns them into its own
        # scroll_event), so without this filter the gallery cannot be scrolled
        # with the wheel or a trackpad at all - only by dragging the scrollbar
        self.canvas.installEventFilter(self)
        # keep the keyboard on the scroll area (Page Up/Down, arrows) instead
        # of letting the canvas take focus for matplotlib's own key bindings,
        # which are of no use in a static gallery
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        # the gallery is always taller than the window, so keeping the
        # vertical scrollbar on stops the width from jumping as it appears
        self.scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.scroll.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.scroll.setFocus()
        vbox.addWidget(self.scroll, stretch=1)
        vbox.addWidget(NavigationToolbar2QT(self.canvas, self))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

        self.resize(1000, 800)
        # guards _fit_canvas against re-entering itself: resizing the canvas
        # can add or remove a scrollbar, which resizes the viewport again
        self._fitting = False
        self._figure_inches = ()
        self.draw()

    def draw(self) -> None:
        """(Re)draw the gallery of the selected channel."""
        if not self.diagnostics:
            return
        data = self.diagnostics[
            min(self.channel.currentIndex(), len(self.diagnostics) - 1)
        ]
        self.figure.clear()
        spline.plot_bead_gallery(
            data,
            self.figure,
            only_rejected=self.only_rejected.isChecked(),
            max_beads=self.max_beads.value(),
        )
        # the size the gallery was laid out at. Kept here rather than read
        # back from the figure on every fit: the canvas is sized in whole
        # pixels, so reading it back would shave a fraction off each time and
        # the gallery would creep smaller with every window resize.
        self._figure_inches = tuple(self.figure.get_size_inches())
        self._fit_canvas()
        n_beads = len(data["keep"])
        n_used = int(data["keep"].sum())
        self.summary.setText(
            f"{n_used} of {n_beads} beads averaged into the PSF, "
            f"{n_beads - n_used} rejected"
        )

    def _fit_canvas(self) -> None:
        """Size the canvas to the figure at the selected zoom.

        The figure sizes itself (in inches) to the number of beads shown, so
        the canvas is given that size in pixels and the scroll area scrolls
        through it. At "Fit width" the pixels-per-inch is chosen so the
        gallery is exactly as wide as the viewport.
        """
        if self._fitting or not self._figure_inches:
            return
        self._fitting = True
        try:
            width_in, height_in = self._figure_inches
            percent = self.zoom.currentData()
            if percent is None:
                # the viewport already excludes the (always-on) vertical
                # scrollbar, so fitting to it never adds a horizontal one
                usable = max(self.scroll.viewport().width() - 2, 100)
                dpi = usable / max(width_in, 1e-6)
            else:
                dpi = percent  # 100 % is 100 pixels per figure inch
            # The canvas is sized in logical pixels while the figure renders
            # at the screen's pixel ratio (2 on a Retina display), so the two
            # are related by dpi / ratio - matplotlib's own resize handler
            # then reads back exactly the inch size the gallery was drawn at.
            ratio = self.canvas.device_pixel_ratio or 1
            self.figure.set_dpi(dpi * ratio)
            self.canvas.setFixedSize(
                max(int(width_in * dpi), 1), max(int(height_in * dpi), 1)
            )
            self.canvas.draw_idle()
        finally:
            self._fitting = False

    def resizeEvent(self, event) -> None:
        """Re-fit the gallery when the window is resized (only the canvas is
        rescaled - the figure itself does not have to be redrawn)."""
        super().resizeEvent(event)
        self._fit_canvas()

    def showEvent(self, event) -> None:
        """Fit the gallery once the dialog has its real size - the first draw
        happens before it is shown, when the viewport is still tiny."""
        super().showEvent(event)
        self._fit_canvas()
        # so Page Up/Down and the arrows scroll the gallery straight away
        self.scroll.setFocus()

    def eventFilter(self, obj, event) -> bool:
        """Scroll the gallery on wheel/trackpad input over the canvas.

        The matplotlib canvas handles wheel events itself and does not pass
        them on, so they are translated into scrollbar movement here; without
        this the gallery could only be scrolled by dragging the scrollbar.
        """
        wheel = QtCore.QEvent.Type.Wheel
        if obj is self.canvas and event.type() == wheel:
            pixels, angle = event.pixelDelta(), event.angleDelta()
            if not pixels.isNull():  # trackpads report pixel-exact deltas
                dx, dy = pixels.x(), pixels.y()
            else:  # a wheel notch is 120 eighths of a degree
                step = self.scroll.verticalScrollBar().singleStep() * 3
                dx = int(angle.x() / 120 * step)
                dy = int(angle.y() / 120 * step)
            for bar, delta in (
                (self.scroll.verticalScrollBar(), dy),
                (self.scroll.horizontalScrollBar(), dx),
            ):
                if delta:
                    bar.setValue(bar.value() - delta)
            return True
        return super().eventFilter(obj, event)


class RegisterChannelsDialog(lib.Dialog):
    """Dialog for the bead-based channel registration.

    Only the transform model is asked for: the box size and minimum net
    gradient come from the parameters dialog."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        """Build the dialog.

        Parameters
        ----------
        window : QtWidgets.QWidget
            The parent window, whose parameters dialog supplies the box size
            and minimum net gradient.
        """
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Register channels (beads)")
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(
            QtWidgets.QLabel(
                "Beads are detected in the currently loaded channels\n"
                "(or the drawn regions), with the box size and minimum\n"
                "net gradient set in the Parameters dialog."
            )
        )
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)
        grid.addWidget(QtWidgets.QLabel("Transform model:"), 0, 0)
        self.registration_model = _transform_model_combo()
        grid.addWidget(self.registration_model, 0, 1)
        self.multi_fov = QtWidgets.QCheckBox(
            "Each frame is a different field of view"
        )
        self.multi_fov.setToolTip(
            "CHECK when the bead movie images a different field of view in\n"
            "every frame, e.g. a stage scan over several positions. Beads are\n"
            "then detected frame by frame and only ever paired within the\n"
            "same frame.\n\n"
            "UNCHECK for a plain bead acquisition, where the frames are\n"
            "repeats of one field and are averaged to beat down the noise."
        )
        self.multi_fov.setTristate(False)
        grid.addWidget(self.multi_fov, 1, 0, 1, 2)
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
    def getSpecs(parent: QtWidgets.QWidget) -> tuple[str, bool, bool]:
        """Show the dialog and read the chosen settings back.

        Parameters
        ----------
        parent : QtWidgets.QWidget
            The window the dialog belongs to.

        Returns
        -------
        model : str
            The chosen transform model.
        multi_fov : bool
            Whether each frame of the bead movie is its own field of view.
        accepted : bool
            False if the dialog was canceled, in which case the other two
            should be ignored.
        """
        dialog = RegisterChannelsDialog(parent)
        result = dialog.exec()
        return (
            _transform_model_of(dialog.registration_model),
            dialog.multi_fov.isChecked(),
            result == QtWidgets.QDialog.DialogCode.Accepted,
        )


class RefineRegistrationDialog(lib.Dialog):
    """Dialog for choosing which frames the signal-based channel re-alignment
    considers.

    The refinement pairs shared single-molecule signal frame by frame, so the
    frames it samples matter: early frames of a movie are often too dense (or
    still bleaching) to pair unambiguously, while late frames may be too
    sparse. The user picks the frame window (1-indexed, inclusive) and how many
    frames are evenly sampled from it."""

    def __init__(
        self,
        window: QtWidgets.QWidget,
        n_frames: int,
        frame_range: list | None,
        model: str | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Re-align channels (signal)")
        vbox = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "Frames used to pair the shared single-molecule signal.\n"
            "The sampled frames are spread evenly over the range."
        )
        vbox.addWidget(info)
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)

        # frame window (1-indexed, inclusive), seeded from the identification
        # frame range when a single segment is set
        lo, hi = 1, max(1, int(n_frames))
        if frame_range:
            first = frame_range[0]
            if not np.isscalar(first):
                lo = max(1, int(first[0]) + 1)
                hi = min(hi, int(frame_range[-1][1]) + 1)
        grid.addWidget(QtWidgets.QLabel("First frame:"), 0, 0)
        self.first_frame = QtWidgets.QSpinBox()
        self.first_frame.setRange(1, max(1, int(n_frames)))
        self.first_frame.setValue(lo)
        grid.addWidget(self.first_frame, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Last frame:"), 1, 0)
        self.last_frame = QtWidgets.QSpinBox()
        self.last_frame.setRange(1, max(1, int(n_frames)))
        self.last_frame.setValue(max(lo, hi))
        grid.addWidget(self.last_frame, 1, 1)

        sampled_label = QtWidgets.QLabel("Frames to sample:")
        sampled_label.setToolTip(
            "Number of frames evenly sampled from the range above.\n"
            "Only these frames are detected on, so this bounds the runtime.\n"
            "Increase it for sparse data (fewer blinks per frame)."
        )
        grid.addWidget(sampled_label, 2, 0)
        self.max_frames = QtWidgets.QSpinBox()
        self.max_frames.setRange(2, 10000)
        self.max_frames.setValue(50)
        grid.addWidget(self.max_frames, 2, 1)

        # Re-registration may also change the model; it defaults to the one
        # the calibration already uses, so a plain re-align keeps it.
        grid.addWidget(QtWidgets.QLabel("Transform model:"), 3, 0)
        self.registration_model = _transform_model_combo()
        _select_transform_model(self.registration_model, model)
        grid.addWidget(self.registration_model, 3, 1)

        # keep the window ordered
        self.first_frame.valueChanged.connect(
            lambda v: self.last_frame.setValue(max(v, self.last_frame.value()))
        )
        self.last_frame.valueChanged.connect(
            lambda v: self.first_frame.setValue(
                min(v, self.first_frame.value())
            )
        )

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
    def getFrameSpecs(
        parent: QtWidgets.QWidget,
        n_frames: int,
        frame_range: list | None = None,
        model: str | None = None,
    ) -> tuple[list, int, str, bool]:
        """Show the dialog and return the chosen ``frame_bounds`` (a single
        0-indexed inclusive segment), the number of frames to sample, the
        transform model, and whether the dialog was accepted. ``model``
        pre-selects the calibration's current model."""
        dialog = RefineRegistrationDialog(parent, n_frames, frame_range, model)
        result = dialog.exec()
        bounds = [
            [dialog.first_frame.value() - 1, dialog.last_frame.value() - 1]
        ]
        chosen = _transform_model_of(dialog.registration_model)
        return (
            bounds,
            dialog.max_frames.value(),
            chosen,
            result == QtWidgets.QDialog.DialogCode.Accepted,
        )


class ROIDialog(lib.Dialog):
    """Sub-dialog for managing several regions of interest (ROIs)
    numerically.

    The dialog edits ``window.view.rois`` directly (clipping overlaps via
    ``localize.clip_rois``) and keeps the compact ROI field in the
    parameters dialog in sync. It is modeless, so the user can keep
    drawing ROIs in the preview while it is open.

    Attributes
    ----------
    table : QtWidgets.QTableWidget
        Table listing the ROIs, one rectangle per row.
    window : QtWidgets.QMainWindow
        Reference to the main window.
    """

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Regions of interest")
        self.setModal(False)
        self._updating = False

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        header.addWidget(self.info_label, 1)
        help_button = lib.HelpButton(ParametersDialog.ROI_URL)
        header.addWidget(help_button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["y_min", "x_min", "y_max", "x_max"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.itemChanged.connect(self.on_table_changed)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add")
        self.add_button.clicked.connect(self.on_add)
        self.remove_button = QtWidgets.QPushButton("Remove")
        self.remove_button.clicked.connect(self.on_remove)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.on_clear)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)

        self.resize(360, 280)
        self.update_table()

    def _box(self) -> int:
        """Current box size, used as the minimum ROI side length."""
        return self.window.parameters.get("Box Size", 7)

    def _split_fov(self) -> bool:
        """Whether the ROIs are split-FOV channels, which get their own
        per-region minimum net gradient column."""
        return bool(self.window.view.split_fov_mode)

    def _commit(self, rois: list) -> None:
        """Clip ``rois`` and store them on the view, refreshing the
        compact field in the parameters dialog."""
        self.window.view.rois = localize.clip_rois(rois, min_size=self._box())
        self.window.parameters_dialog.update_roi_display(skip_dialog=True)
        self.window.draw_frame()

    def update_table(self) -> None:
        """Repopulate the table from the view's ROIs."""
        view = self.window.view
        mngs = self.window.region_mngs()
        split_fov = self._split_fov()
        # the fit settings are per region only when the regions are fitted
        # one at a time; the joint fit uses one calibration for all of them
        per_region_fit = (
            split_fov and self.window.fit_mode() == FIT_MODE_SEPARATE
        )
        params = (
            self.window.flush_region_fit_params() if per_region_fit else []
        )
        per_region_fit = per_region_fit and len(params) == len(view.rois)
        self._updating = True
        self.info_label.setText(
            "Each row is a rectangular ROI (y_min, x_min, y_max, x_max, "
            "in camera pixels). Drag a rectangle in the preview or use "
            "Add, then edit the cells. Overlapping ROIs are clipped "
            "automatically so they never cover a pixel twice. Clear the "
            "list to analyze the whole frame."
            + (
                " In split-FOV mode each region is a channel with its own "
                "min. net gradient (last column); selecting a row also puts "
                "its value on the slider in the parameters dialog."
                if split_fov
                else ""
            )
            + (
                " Fitted separately, each region also carries its own fit "
                "settings, shown read-only in the last two columns: select "
                "a region to edit them in the parameters dialog."
                if per_region_fit
                else ""
            )
        )
        self.table.setColumnCount(
            4 + (1 if split_fov else 0) + (2 if per_region_fit else 0)
        )
        self.table.setHorizontalHeaderLabels(
            ["y_min", "x_min", "y_max", "x_max"]
            + (["min_ng"] if split_fov else [])
            + (["model", "PSF calib."] if per_region_fit else [])
        )
        self.table.setRowCount(len(view.rois))
        for row, ((y_min, x_min), (y_max, x_max)) in enumerate(view.rois):
            values = [y_min, x_min, y_max, x_max]
            if split_fov:
                values.append(mngs[row])
            for col, val in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(int(val)))
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
            if per_region_fit:
                for col, (text, tip) in enumerate(
                    _region_fit_summary(params[row]), start=5
                ):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    item.setToolTip(tip)
                    # the fit settings are edited in the parameters dialog,
                    # with the region selected
                    item.setFlags(
                        item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
                    )
                    self.table.setItem(row, col, item)
        if view.selected_roi is not None and view.selected_roi < len(
            view.rois
        ):
            self.table.selectRow(view.selected_roi)
        self._updating = False

    def on_table_changed(self, item: object = None) -> None:
        """Rebuild the view's ROIs (and, in split-FOV mode, their
        thresholds) from the table, clipping overlaps."""
        if self._updating:
            return
        split_fov = self._split_fov()
        rois = []
        mngs = []
        for row in range(self.table.rowCount()):
            try:
                vals = [int(self.table.item(row, c).text()) for c in range(4)]
                if split_fov:
                    mngs.append(int(self.table.item(row, 4).text()))
            except (AttributeError, ValueError):
                return  # incomplete row, wait for the user to finish
            y_min, x_min, y_max, x_max = vals
            rois.append([[y_min, x_min], [y_max, x_max]])
        self._commit(rois)
        # clipping can drop or split rectangles, so only adopt the edited
        # thresholds when the rows still line up with the stored regions
        if split_fov and len(mngs) == len(self.window.view.rois):
            self.window.view.roi_mngs = mngs
            self.window.parameters_dialog.sync_mng_to_selected_region()
            self.window.draw_frame()
        self.update_table()

    def on_add(self) -> None:
        """Add a default ROI row that the user can edit."""
        view = self.window.view
        if self.window.movie is not None:
            height, width = self.window.movie.shape[1:]
            side = int(min(height, width) / 4)
        else:
            side = 50
        offset = 10 * len(view.rois)  # avoid fully overlapping the last one
        new_roi = [[offset, offset], [offset + side, offset + side]]
        self._commit(view.rois + [new_roi])
        self.update_table()

    def on_remove(self) -> None:
        """Remove the ROI(s) selected in the table."""
        view = self.window.view
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()},
            reverse=True,
        )
        self.window.region_mngs()  # align the thresholds before deleting
        self.window.region_params()  # ... and the per-region fit settings
        for row in rows:
            if 0 <= row < len(view.rois):
                del view.rois[row]
                if row < len(view.roi_mngs):
                    del view.roi_mngs[row]
                if row < len(view.roi_params):
                    del view.roi_params[row]
        view.selected_roi = None
        self._commit(view.rois)
        self.update_table()

    def on_clear(self) -> None:
        """Remove all ROIs (analyze the whole frame)."""
        self.window.view.selected_roi = None
        self._commit([])
        self.update_table()

    def on_selection_changed(self) -> None:
        """Highlight the ROI selected in the table (and, in split-FOV mode,
        put its threshold on the min. net gradient slider)."""
        if self._updating:
            return
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        self.window.view.selected_roi = min(rows) if rows else None
        self.window.parameters_dialog.sync_mng_to_selected_region()
        self.window.sync_region_fit_params()
        self.window.draw_frame()


class CalibrateAffineDialog(lib.Dialog):
    """Select the inputs/output for a lateral-transform calibration.

    The same calibration corrects two things, chosen at the top of the
    dialog: the lateral distortion of a cylindrical lens (astigmatism) or
    chromatic aberration between two color channels. Below it are a
    reference bead image, the bead image to be mapped onto it, and the
    calibration file the transform is appended to - an existing Gaussian
    astigmatism (YAML) or spline PSF (HDF5) calibration, or a new
    standalone YAML holding only lateral corrections (for 2D data). Several
    transforms accumulate in one file as an ordered list and are applied
    one after another.
    """

    LATERAL_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#lateral-corrections-of-x-and-y"  # noqa: E501

    # Emitted when "Calibrate" is pressed. Not ``accepted``: the dialog
    # stays open so the pairing can be inspected on both bead images.
    calibrate_requested = QtCore.pyqtSignal()

    # transform type -> (reference label/tooltip, target label/tooltip)
    IMAGE_LABELS = {
        "astigmatism": (
            "Reference image:",
            "Image of in-focus beads WITHOUT a cylindrical lens in"
            " the optical pathway",
            "Cylindrical lens image:",
            "Image of in-focus beads WITH a cylindrical lens in"
            " the optical pathway",
        ),
        "chromatic": (
            "Reference channel image:",
            "Image of in-focus beads in the reference color channel,"
            " i.e. the channel every other channel is mapped onto",
            "Target channel image:",
            "Image of the same in-focus beads in the color channel to be"
            " corrected",
        ),
    }

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Calibrate lateral transform")
        self.setModal(False)

        vbox = QtWidgets.QVBoxLayout(self)

        type_row = QtWidgets.QHBoxLayout()
        type_label = QtWidgets.QLabel("Correct:")
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem(
            "Astigmatism (cylindrical lens)", "astigmatism"
        )
        self.type_combo.addItem("Chromatic aberration", "chromatic")
        self.type_combo.setToolTip(
            "What the transform corrects. Both are fitted the same way,\n"
            "from two bead images, and are stored as an ordered list in\n"
            "the calibration file: for 3D two-color data, calibrate both\n"
            "into the same file and they are applied one after another."
        )
        self.type_combo.currentIndexChanged.connect(self._update_labels)
        type_row.addWidget(type_label)
        type_row.addWidget(self.type_combo)
        type_row.addStretch(1)
        type_row.addWidget(lib.HelpButton(self.LATERAL_URL))
        vbox.addLayout(type_row)

        model_row = QtWidgets.QHBoxLayout()
        self.model_combo = _transform_model_combo()
        model_row.addWidget(QtWidgets.QLabel("Transform model:"))
        model_row.addWidget(self.model_combo)
        model_row.addStretch(1)
        vbox.addLayout(model_row)

        grid = QtWidgets.QGridLayout()

        rows = [
            (self._browse_reference, self._show_reference),
            (self._browse_target, self._show_target),
            (self._browse_calibration, None),
        ]
        self.reference_edit = QtWidgets.QLineEdit()
        self.target_edit = QtWidgets.QLineEdit()
        self.calibration_edit = QtWidgets.QLineEdit()
        edits = [
            self.reference_edit,
            self.target_edit,
            self.calibration_edit,
        ]
        self.reference_label = QtWidgets.QLabel()
        self.target_label = QtWidgets.QLabel()
        calibration_label = QtWidgets.QLabel("Calibration:")
        calibration_tooltip = (
            "Where the transform is stored. Select an existing calibration\n"
            "(a Gaussian 3D .yaml or a spline PSF .hdf5) to append it to,\n"
            "or 'New' to start a standalone lateral calibration .yaml - the\n"
            "option for purely 2D data."
        )
        calibration_label.setToolTip(calibration_tooltip)
        self.calibration_edit.setToolTip(calibration_tooltip)
        labels = [self.reference_label, self.target_label, calibration_label]
        for row_idx, ((slot, show_slot), edit, label) in enumerate(
            zip(rows, edits, labels)
        ):
            edit.setReadOnly(True)
            edit.setMinimumWidth(400)
            button = QtWidgets.QPushButton("Browse")
            button.clicked.connect(slot)
            grid.addWidget(label, row_idx, 0)
            grid.addWidget(edit, row_idx, 1)
            grid.addWidget(button, row_idx, 2)
            if show_slot is not None:
                show_button = QtWidgets.QPushButton("Show")
                show_button.setToolTip(
                    "Load this image in the main window and open the"
                    " parameters dialog to tune the identification"
                    " parameters (box size, min. net gradient) with a"
                    " live preview."
                )
                show_button.clicked.connect(show_slot)
                grid.addWidget(show_button, row_idx, 3)
            else:
                new_button = QtWidgets.QPushButton("New")
                new_button.setToolTip(
                    "Create a new standalone lateral calibration file\n"
                    "(.yaml), holding only lateral corrections. Use it when\n"
                    "there is no 3D calibration to append to, e.g. a\n"
                    "chromatic correction for 2D data."
                )
                new_button.clicked.connect(self._new_calibration)
                grid.addWidget(new_button, row_idx, 3)
        self._update_labels()
        vbox.addLayout(grid)

        # If a movie is already loaded in the main window, use it as the
        # reference image by default.
        if window.movie_path:
            self.reference_edit.setText(window.movie_path)

        # "Calibrate" does not close the dialog: the bead
        # pairing is drawn over whichever of the two images is displayed,
        # so the "Show" buttons above are what the user reaches for right
        # after a fit. Closing here would mean reopening the dialog and
        # re-picking all three paths to look at the other image.
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        self.calibrate_button = self.buttons.addButton(
            "Calibrate",
            QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole,
        )
        self.calibrate_button.setToolTip(
            "Fit the transform and append it to the calibration file.\n"
            "The dialog stays open so the bead pairing can be inspected\n"
            "on both images with 'Show'."
        )
        self.calibrate_button.setDefault(True)
        self.calibrate_button.clicked.connect(self.calibrate_requested.emit)
        vbox.addWidget(self.buttons)
        self.buttons.rejected.connect(self.reject)

    @property
    def transform_type(self) -> str:
        return self.type_combo.currentData()

    @property
    def transform_model(self) -> str:
        """The model the transform is to be fitted with."""
        return _transform_model_of(self.model_combo)

    @property
    def reference_path(self) -> str:
        return self.reference_edit.text()

    @property
    def target_path(self) -> str:
        return self.target_edit.text()

    @property
    def calibration_path(self) -> str:
        return self.calibration_edit.text()

    def _update_labels(self) -> None:
        """Rename the two image rows to match the selected transform
        type; the inputs and the fit itself are the same either way."""
        ref_text, ref_tip, target_text, target_tip = self.IMAGE_LABELS[
            self.transform_type
        ]
        self.reference_label.setText(ref_text)
        self.reference_label.setToolTip(ref_tip)
        self.reference_edit.setToolTip(ref_tip)
        self.target_label.setText(target_text)
        self.target_label.setToolTip(target_tip)
        self.target_edit.setToolTip(target_tip)

    def _pick_file(
        self,
        edit: QtWidgets.QLineEdit,
        title: str,
        filter_str: str,
    ) -> None:
        current = edit.text()
        directory = os.path.split(current)[0] if current else None
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            title,
            directory=directory,
            filter=filter_str,
        )
        if path:
            edit.setText(path)

    def _browse_reference(self) -> None:
        self._pick_file(
            self.reference_edit, "Select reference image", IMAGE_FILTER
        )

    def _browse_target(self) -> None:
        title = (
            "Select cylindrical lens image"
            if self.transform_type == "astigmatism"
            else "Select target channel image"
        )
        self._pick_file(self.target_edit, title, IMAGE_FILTER)

    def _browse_calibration(self) -> None:
        self._pick_file(
            self.calibration_edit,
            "Select calibration file",
            "Calibration files (*.yaml *.hdf5)",
        )

    def _new_calibration(self) -> None:
        """Pick a path for a new standalone lateral calibration (.yaml).
        Existing files are appended to, not overwritten, by the worker."""
        current = self.calibration_edit.text()
        directory = (
            current
            if current
            else os.path.join(
                os.path.split(self.reference_path)[0],
                "lateral_calibration.yaml",
            )
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "New lateral calibration",
            directory=directory,
            filter="*.yaml",
        )
        if path:
            if not os.path.splitext(path)[1]:
                path += ".yaml"
            self.calibration_edit.setText(path)

    def _show(self, edit: QtWidgets.QLineEdit) -> None:
        """Load the image in ``edit`` into the main window and open the
        parameters dialog."""
        path = edit.text()
        if not path:
            QtWidgets.QMessageBox.warning(
                self,
                "Show image",
                "Select an image first.",
            )
            return
        self.window.open(path)
        self.window.parameters_dialog.show()
        self.window.parameters_dialog.raise_()

    def _show_reference(self) -> None:
        self._show(self.reference_edit)

    def _show_target(self) -> None:
        self._show(self.target_edit)


class ParametersDialog(lib.Dialog):
    """Choose analysis parameters.

    ...

    Attributes
    ----------
    baseline : QtWidgets.QDoubleSpinBox
        Spin box for selecting camera baseline (background amplitude).
    box_spinbox : OddSpinBox
        Spin box for selecting the box size.
    camera : QtWidgets.QComboBox
        Combo box for selecting the camera.
    cam_combos : CamSettingComboBoxDict
        Combo boxes for selecting channels.
    convergence_criterion : QtWidgets.QDoubleSpinBox
        Spin box for setting the convergence criterion. Only used for
        MLE fitting.
    emission_combos : EmissionSettingComboBoxDict
        Combo boxes for selecting emission wavelengths.
    fit_model : QtWidgets.QComboBox
        Combo box for selecting the fitting model (e.g. 2D elliptical
        Gaussian or average of ROI).
    fit_mode_combo : QtWidgets.QComboBox
        Combo box for choosing whether multichannel (or split-FOV) data is
        fitted jointly or one channel at a time, see ``FIT_MODE_JOINT`` /
        ``FIT_MODE_SEPARATE``. Shown for multichannel data only.
    fit_optimizer : QtWidgets.QComboBox
        Combo box for selecting the optimizer (Least squares or MLE).
        Hidden for models that do not use an optimizer.
    fit_z_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling fitting in the z-dimension using
        astigmatism.
    gain : QtWidgets.QSpinBox
        Spin box for selecting camera EM gain.
    gaussian_filter_spinbox : QtWidgets.QDoubleSpinBox
        Spin box for the standard deviation (in camera pixels) of the
        spatial Gaussian filter applied before identification. 0 disables
        it; fitting always uses the raw movie.
    gpu_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling GPU fitting. Only shown if a
        CUDA-capable GPU is available. Also selects the device for the
        astigmatism z fit, which has its own CUDA kernel
        (``picasso.zfit``).
    magnification_factor : QtWidgets.QDoubleSpinBox
        Spin box for setting the magnification factor for 3D fitting.
    max_it : QtWidgets.QSpinBox
        Spin box for selecting the max. number of iterations. Only used
        for MLE fitting.
    mng_min_spinbox : QtWidgets.QSpinBox
        Spin box for selecting the minimum net gradient (lower bound).
    mng_max_spinbox : QtWidgets.QSpinBox
        Spin box for selecting the minimum net gradient (upper bound).
    mng_slider : QtWidgets.QSlider
        Slider for selecting the minimum net gradient.
    mng_spinbox : QtWidgets.QSpinBox
        Spin box for selecting the minimum net gradient.
    pixelsize : QtWidgets.QDoubleSpinBox
        Spin box for setting camera pixel size (nm).
    preview_checkbox : QtWidgets.QCheckBox
        Checkbox for enabling/disabling preview of identified spots.
    roi_field : QtWidgets.QLineEdit
        Compact field summarizing the regions of interest (ROIs): empty
        for the whole frame, the four coordinates of the single ROI when
        there is exactly one (editable), or a count when there are
        several.
    roi_dialog : ROIDialog or None
        Sub-dialog (lazily created) for managing several ROIs in a table.
    sensitivity : QtWidgets.QDoubleSpinBox
        Spin box for setting camera sensitivity.
    qe : QtWidgets.QDoubleSpinBox
        Spin box for setting camera quantum efficiency (QE). **Note**:
        QE value is not used in the analysis, only present for
        backward compatibility.
    quality_check : QtWidgets.QCheckBox
        Checkbox for enabling/disabling quality check (mean bright time,
        drift and NeNA estimation).
    quality_grid_labels : list[QtWidgets.QLabel]
        Labels for displaying quality checks.
    quality_grid_values : list[QtWidgets.QLabel]
        Values for displaying quality checks.
    window : QtWidgets.QWidget
        The main window of the application.
    """

    CALIB_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#d-calibration"  # noqa: E501
    GAUSSIAN_FILTER_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#gaussian-filter"  # noqa: E501
    IDENT_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#identification-and-fitting-of-single-molecule-spots"  # noqa: E501
    ROI_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#regions-of-interest-rois"  # noqa: E501
    SPLINE_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#experimental-psf-cubic-spline-fitting"  # noqa: E501
    TEMPORAL_MEDIAN_URL = "https://picassosr.readthedocs.io/en/latest/localize.html#temporal-median-filter"  # noqa: E501

    def __init__(  # noqa: C901
        self, parent: QtWidgets.QMainWindow | None = None
    ) -> None:
        super().__init__(parent)
        self.window = parent
        self.setWindowTitle("Parameters")
        self.setModal(False)

        self.z_calibration = {}
        self.z_calibration_path = None
        self.spline_calibration = {}
        self.spline_calibration_path = None
        # Standalone channel registration for the multichannel (joint) 2D
        # spherical Gaussian fit. Named apart from the split-FOV
        # "channel_registration" key a spline calibration carries, which is a
        # different thing (region-local transforms).
        self.channel_registration_calibration = {}
        self.channel_registration_path = None
        # Standalone lateral corrections applied after the fit; those
        # carried by the 3D / spline calibration are applied by the fit
        # itself (see load_affine_calib).
        self.lateral_transforms = []
        self.affine_calibration_paths = []
        self.camera_calibration = {}
        self.camera_calibration_path = None
        # Scalars to put back when a calibration is cleared, or None while
        # none is loaded.
        self._scalars_before_calib = None
        # calibration group boxes, toggled by the selected fit model; set up
        # further below
        self.z_groupbox = None
        self.spline_groupbox = None
        self.channel_registration_groupbox = None
        # last resolved fit code, so the convergence defaults are only
        # reapplied when the method actually changes
        self._last_fit_code = None
        # guards the two-way binding between the min. net gradient slider and
        # the selected split-FOV region (see sync_mng_to_selected_region)
        self._syncing_mng = False
        # last value the slider settled on. Split-FOV regions drawn later
        # inherit it, so a slider move that edits one region cannot also
        # seed its neighbors with the value being typed.
        self._last_mng = DEFAULT_PARAMETERS["Min. Net Gradient"]

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        self.scroll_area.setWidget(container)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)

        identification_groupbox = QtWidgets.QGroupBox("Identification")
        vbox.addWidget(identification_groupbox)
        identification_grid = QtWidgets.QGridLayout(identification_groupbox)

        # Box Size
        first_row = QtWidgets.QHBoxLayout()
        identification_grid.addLayout(first_row, 0, 0, 1, 2)
        first_row.addWidget(lib.HelpButton(self.IDENT_URL))
        boxsize_label = QtWidgets.QLabel("Box side length:")
        boxsize_label.setToolTip(
            "Box size in camera pixels for identification."
        )
        first_row.addWidget(boxsize_label)
        self.box_spinbox = OddSpinBox()
        self.box_spinbox.setKeyboardTracking(False)
        self.box_spinbox.setValue(DEFAULT_PARAMETERS["Box Size"])
        self.box_spinbox.valueChanged.connect(self.on_box_changed)
        first_row.addWidget(self.box_spinbox)

        # Min. Net Gradient
        mng_label = QtWidgets.QLabel("Min. net gradient:")
        mng_label.setToolTip(
            "Threshold (related to brightness) for spot identification."
        )
        identification_grid.addWidget(mng_label, 1, 0)
        self.mng_spinbox = QtWidgets.QSpinBox()
        self.mng_spinbox.setRange(0, int(1e9))
        self.mng_spinbox.setValue(DEFAULT_PARAMETERS["Min. Net Gradient"])
        self.mng_spinbox.setKeyboardTracking(False)
        self.mng_spinbox.valueChanged.connect(self.on_mng_spinbox_changed)
        identification_grid.addWidget(self.mng_spinbox, 1, 1)

        # Slider
        self.mng_slider = QtWidgets.QSlider()
        self.mng_slider.setToolTip(
            "Adjust the minimum net gradient for spot identification.\n\n"
            "In split-FOV mode ('Regions = channels') each region has its\n"
            "own threshold: the slider shows and edits the selected\n"
            "region's value, and sets every region's when none is\n"
            "selected. Click a region in the preview (or a row in\n"
            "'Edit ROIs...') to select it."
        )
        self.mng_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.mng_slider.setRange(0, 10000)
        self.mng_slider.setValue(DEFAULT_PARAMETERS["Min. Net Gradient"])
        self.mng_slider.setSingleStep(1)
        self.mng_slider.setPageStep(20)
        self.mng_slider.valueChanged.connect(self.on_mng_slider_changed)
        identification_grid.addWidget(self.mng_slider, 2, 0, 1, 2)

        hbox = QtWidgets.QHBoxLayout()
        identification_grid.addLayout(hbox, 3, 0, 1, 2)

        # Min SpinBox
        self.mng_min_spinbox = QtWidgets.QSpinBox()
        self.mng_min_spinbox.setToolTip(
            "Minimum value for the minimum net gradient slider."
        )
        self.mng_min_spinbox.setRange(0, 999999)
        self.mng_min_spinbox.setKeyboardTracking(False)
        self.mng_min_spinbox.setValue(0)
        self.mng_min_spinbox.valueChanged.connect(self.on_mng_min_changed)
        hbox.addWidget(self.mng_min_spinbox)

        hbox.addStretch(1)

        # Max SpinBox
        self.mng_max_spinbox = QtWidgets.QSpinBox()
        self.mng_max_spinbox.setToolTip(
            "Maximum value for the minimum net gradient slider."
        )
        self.mng_max_spinbox.setKeyboardTracking(False)
        self.mng_max_spinbox.setRange(0, 999999)
        self.mng_max_spinbox.setValue(10000)
        self.mng_max_spinbox.valueChanged.connect(self.on_mng_max_changed)
        hbox.addWidget(self.mng_max_spinbox)

        # temporal median background subtraction (identification only)
        tm_row = QtWidgets.QHBoxLayout()
        tm_row.addWidget(lib.HelpButton(self.TEMPORAL_MEDIAN_URL))
        self.temporal_median_checkbox = QtWidgets.QCheckBox(
            "Temporal median filter"
        )
        self.temporal_median_checkbox.setToolTip(
            "Subtract a running per-pixel temporal median before spot\n"
            "identification, which removes inhomogeneous background and\n"
            "static structures.\n\n"
            "Only the identification uses the filtered frames: fitting, spot\n"
            "cutting and photon conversion always use the raw movie.\n\n"
            "Net gradient values change when this is on, so re-tune 'Min.\n"
            "net gradient' with 'Preview' enabled after switching it on or"
            "off.\n\nNot applied to bead stacks (3D / spline calibration)."
        )
        self.temporal_median_checkbox.setTristate(False)
        self.temporal_median_checkbox.setChecked(False)
        # kept so ``Window._update_temporal_median_availability`` can restore
        # them after showing why the filter is unavailable
        self._temporal_median_text = self.temporal_median_checkbox.text()
        self._temporal_median_tip = self.temporal_median_checkbox.toolTip()
        self.temporal_median_checkbox.stateChanged.connect(
            self.on_temporal_median_changed
        )
        tm_row.addWidget(self.temporal_median_checkbox)
        self.temporal_median_spinbox = QtWidgets.QSpinBox()
        self.temporal_median_spinbox.setRange(3, 100_000)
        self.temporal_median_spinbox.setSingleStep(10)
        self.temporal_median_spinbox.setValue(
            DEFAULT_PARAMETERS["Temporal Median Window"]
        )
        self.temporal_median_spinbox.setKeyboardTracking(False)
        self.temporal_median_spinbox.setEnabled(False)
        self.temporal_median_spinbox.setToolTip(
            "Number of frames in the temporal window used for the median."
        )
        self.temporal_median_spinbox.valueChanged.connect(self.on_box_changed)
        tm_row.addWidget(self.temporal_median_spinbox)
        tm_row.addStretch(1)
        identification_grid.addLayout(tm_row, 4, 0, 1, 2)

        # spatial Gaussian smoothing (identification only). Sits right below
        # the temporal median filter, in the order the two are applied.
        gauss_row = QtWidgets.QHBoxLayout()
        gauss_row.addWidget(lib.HelpButton(self.GAUSSIAN_FILTER_URL))
        gaussian_tip = (
            "Smooth every frame with a Gaussian of this standard deviation\n"
            "(in camera pixels) before spot identification. 0 turns the\n"
            "filter off.\n\n"
            "Use it when spots are not Gaussian-shaped and break up into\n"
            "several local maxima: smoothing merges them into one maximum,\n"
            "so one spot yields one robust identification.\n\n"
            "Only the identification uses the smoothed frames: fitting, spot\n"
            "cutting and photon conversion always use the raw movie, so\n"
            "photon counts and localization precisions are unaffected.\n\n"
            "Smoothing requires re-tuning of 'Min. net gradient' with\n"
            "'Preview' enabled after changing this value.\n\n"
            "Note: large values merge neighboring spots into one."
        )
        gaussian_label = QtWidgets.QLabel("Gaussian filter sigma:")
        gaussian_label.setToolTip(gaussian_tip)
        gauss_row.addWidget(gaussian_label)
        self.gaussian_filter_spinbox = QtWidgets.QDoubleSpinBox()
        self.gaussian_filter_spinbox.setRange(0.0, 10.0)
        self.gaussian_filter_spinbox.setDecimals(2)
        self.gaussian_filter_spinbox.setSingleStep(0.1)
        self.gaussian_filter_spinbox.setSuffix(" px")
        # Qt drops the suffix at the special value, so 0 reads as "Off"
        self.gaussian_filter_spinbox.setSpecialValueText("Off")
        self.gaussian_filter_spinbox.setValue(
            DEFAULT_PARAMETERS["Gaussian Filter Sigma"]
        )
        self.gaussian_filter_spinbox.setKeyboardTracking(False)
        self.gaussian_filter_spinbox.setToolTip(gaussian_tip)
        self.gaussian_filter_spinbox.valueChanged.connect(
            self.on_gaussian_filter_changed
        )
        gauss_row.addWidget(self.gaussian_filter_spinbox)
        gauss_row.addStretch(1)
        identification_grid.addLayout(gauss_row, 5, 0, 1, 2)

        # preview identifications + cross-channel link color overlay
        preview_row = QtWidgets.QHBoxLayout()
        self.preview_checkbox = QtWidgets.QCheckBox("Preview")
        self.preview_checkbox.setToolTip(
            "Show identified spots in the current frame?"
        )
        self.preview_checkbox.setTristate(False)
        self.preview_checkbox.stateChanged.connect(self.on_preview_changed)
        preview_row.addWidget(self.preview_checkbox)
        self.link_colors_checkbox = QtWidgets.QCheckBox("Link colors")
        self.link_colors_checkbox.setToolTip(
            "Color-code the identification boxes by their cross-channel link.\n\n"
            "Spots paired across channels share a color; unmatched spots are\n"
            "gray. Pairing uses the loaded multichannel / split-FOV spline\n"
            "calibration's inter-channel transform. With no calibration\n"
            "loaded, the transform is estimated from the identifications "
            "themselves."
        )
        self.link_colors_checkbox.setTristate(False)
        self.link_colors_checkbox.stateChanged.connect(
            self.on_link_colors_changed
        )
        # shown by the window for multichannel / split-FOV data
        # (see Window._update_multichannel_widgets)
        _retain_size_when_hidden(self.link_colors_checkbox)
        self.link_colors_checkbox.hide()
        preview_row.addWidget(self.link_colors_checkbox)
        preview_row.addStretch(1)
        identification_grid.addLayout(preview_row, 6, 0)

        # Multichannel: identify each channel on its own, or on the channels
        # added together. Shown by the window for multichannel / split-FOV
        # data (see Window._update_multichannel_widgets); its space is
        # reserved while hidden, so loading channels does not reflow the
        # dialog.
        mode_row = QtWidgets.QHBoxLayout()
        self.identify_mode_label = QtWidgets.QLabel("Identify on:")
        identify_mode_tip = (
            "What the spots are searched in.\n\n"
            f"'{IDENTIFY_MODE_SEPARATE}': every channel (or split-FOV\n"
            "region) is identified on its own, and the joint spline fit then\n"
            "keeps only the molecules found in all of them.\n\n"
            f"'{IDENTIFY_MODE_SUM}': the channels are mapped onto the\n"
            "reference channel and added up (in photons), and the spots are\n"
            "identified in that sum. Use this when a channel is too dim to\n"
            "detect in by itself - a molecule that is faint everywhere can\n"
            "still stand out in the sum. The fit then uses these detections\n"
            "directly, without requiring a detection in every channel.\n\n"
            "The registration comes from the loaded multichannel / split-FOV\n"
            "spline calibration; without one, every channel is identified\n"
            "first and the transform is estimated from those detections.\n"
            "The sum is shown (and previewed) as soon as it is selected,\n"
            "wherever the channels are already registered.\n"
            "Note that the minimum net gradient has to be re-tuned for the\n"
            "sum: it is in photons and over all channels."
        )
        self.identify_mode_label.setToolTip(identify_mode_tip)
        self.identify_mode_combo = QtWidgets.QComboBox()
        self.identify_mode_combo.addItems(
            [IDENTIFY_MODE_SEPARATE, IDENTIFY_MODE_SUM]
        )
        self.identify_mode_combo.setToolTip(identify_mode_tip)
        self.identify_mode_combo.currentIndexChanged.connect(
            self.on_identify_mode_changed
        )
        mode_row.addWidget(self.identify_mode_label)
        mode_row.addWidget(self.identify_mode_combo)
        mode_row.addStretch(1)
        for widget in (self.identify_mode_label, self.identify_mode_combo):
            _retain_size_when_hidden(widget)
            widget.hide()
        identification_grid.addLayout(mode_row, 6, 1)

        # ROIs
        label = QtWidgets.QLabel("ROIs:")
        label.setToolTip(
            "Restrict the analysis to one or more rectangular regions.\n"
            "Drag a rectangle in the preview to add a ROI, double-click a\n"
            "ROI to remove it, or click 'Edit ROIs...' to enter them\n"
            "numerically. With a single ROI its coordinates\n"
            "(y_min, x_min, y_max, x_max, in camera pixels) can be edited\n"
            "directly here. Leave empty to analyze the whole frame."
        )
        roi_label_layout = QtWidgets.QHBoxLayout()
        roi_label_layout.addWidget(lib.HelpButton(self.ROI_URL))
        roi_label_layout.addWidget(label)
        roi_label_layout.addStretch(1)
        # Split-FOV: treat the drawn ROIs as separate channels of one movie.
        self.split_fov_checkbox = QtWidgets.QCheckBox("Regions = channels")
        self.split_fov_checkbox.setToolTip(
            "Split-FOV mode: treat the drawn ROIs as separate channels imaged\n"
            "side-by-side on one camera (spectral / biplane split\n"
            "optics). The first region is the reference channel and all\n"
            "regions are kept the same size: drag once to set the size, click\n"
            "to drop more regions, drag a region (or use the arrow keys) to\n"
            "fine-tune its registration. 'Calibrate spline PSF' and the\n"
            "spline fit then use these regions as channels of this movie.\n\n"
            "Each region also carries its own min. net gradient, since the\n"
            "channels need not share a brightness scale: select a region\n"
            "and the slider above tunes that region alone."
        )
        self.split_fov_checkbox.setTristate(False)
        self.split_fov_checkbox.stateChanged.connect(self.on_split_fov_changed)
        roi_label_layout.addWidget(self.split_fov_checkbox)
        identification_grid.addLayout(roi_label_layout, 7, 0)

        self._updating_roi_field = False
        self.roi_dialog = None
        roi_layout = QtWidgets.QHBoxLayout()
        self.roi_field = QtWidgets.QLineEdit()
        self.roi_field.setPlaceholderText("Whole frame")
        regex = r"\d+,\d+,\d+,\d+"  # 4 integers separated by commas
        validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(regex)
        )
        self.roi_field.setValidator(validator)
        self.roi_field.editingFinished.connect(self.on_roi_field_finished)
        self.roi_field.textChanged.connect(self.on_roi_field_changed)
        roi_layout.addWidget(self.roi_field)
        self.roi_edit_button = QtWidgets.QPushButton("Edit ROIs...")
        self.roi_edit_button.clicked.connect(self.on_edit_rois)
        roi_layout.addWidget(self.roi_edit_button)
        identification_grid.addLayout(roi_layout, 7, 1)

        # min/max frames
        label = QtWidgets.QLabel("Frames (min,max):")
        label.setToolTip(
            "Specify the first and last frame (inclusive) to be analyzed;\n"
            "by default, all frames are analyzed.\n"
            "Several disjoint segments can be given as min,max pairs\n"
            "separated by semicolons, e.g. '1,100; 200,300'."
        )
        identification_grid.addWidget(label, 8, 0)
        self.frames_edit = QtWidgets.QLineEdit()
        # one or more "min,max" pairs separated by semicolons, with
        # optional surrounding whitespace
        regex = r"\s*\d+\s*,\s*\d+\s*(;\s*\d+\s*,\s*\d+\s*)*"
        validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(regex)
        )
        self.frames_edit.setValidator(validator)
        self.frames_edit.editingFinished.connect(self.on_frames_edit_finished)
        self.frames_edit.textChanged.connect(self.on_frames_edit_changed)
        identification_grid.addWidget(self.frames_edit, 8, 1)

        # Multichannel: optionally use the same key settings for every channel.
        # Shown only when more than one channel is loaded (see the Window's
        # _populate_channel_combo). Checking a box makes that group shared, so
        # switching channels no longer overwrites it.
        self.link_groupbox = QtWidgets.QGroupBox(
            "Same settings across channels"
        )
        vbox.addWidget(self.link_groupbox)
        link_layout = QtWidgets.QHBoxLayout(self.link_groupbox)
        self.link_box_checkbox = QtWidgets.QCheckBox("Box size")
        self.link_box_checkbox.setToolTip(
            "Use the same box size for every channel."
        )
        self.link_mng_checkbox = QtWidgets.QCheckBox("Min. net gradient")
        self.link_mng_checkbox.setToolTip(
            "Use the same minimum net gradient for every channel."
        )
        self.link_camera_checkbox = QtWidgets.QCheckBox("Camera settings")
        self.link_camera_checkbox.setToolTip(
            "Use the same camera and photon-conversion settings (camera,\n"
            "baseline, EM gain, sensitivity, QE, pixel size) for every "
            "channel."
        )
        self.link_calib_checkbox = QtWidgets.QCheckBox("PSF calibration")
        self.link_calib_checkbox.setToolTip(
            "Use the same experimental PSF (spline) calibration for every\n"
            "channel? Untick to give each channel its own calibration."
        )
        self.link_calib_checkbox.setChecked(True)
        for cb in (
            self.link_box_checkbox,
            self.link_mng_checkbox,
            self.link_camera_checkbox,
            self.link_calib_checkbox,
        ):
            cb.setTristate(False)
            cb.stateChanged.connect(self.on_link_params_changed)
            link_layout.addWidget(cb)
        link_layout.addStretch(1)
        self.link_groupbox.hide()  # shown by the window for >1 channel

        # Camera:
        if "Cameras" in CONFIG:
            # Experiment settings
            exp_groupbox = QtWidgets.QGroupBox("Experiment Settings")
            vbox.addWidget(exp_groupbox)
            exp_grid = QtWidgets.QGridLayout(exp_groupbox)
            exp_grid.addWidget(QtWidgets.QLabel("Camera:"), 0, 0)
            self.camera = QtWidgets.QComboBox()
            exp_grid.addWidget(self.camera, 0, 1)
            cameras = sorted(list(CONFIG["Cameras"].keys()))
            if "CameraPriority" in CONFIG:
                cam_prio_list = CONFIG["CameraPriority"]
                # remove the prio cameras from the sorted list
                for cam in cam_prio_list:
                    if cam in cameras:
                        cameras.remove(cam)
                cameras = cam_prio_list + cameras
            self.camera.addItems(cameras)
            self.camera.currentIndexChanged.connect(self.on_camera_changed)

            self.cam_settings = QtWidgets.QStackedWidget()
            self.cam_settings.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            exp_grid.addWidget(self.cam_settings, 1, 0, 1, 2)
            self.cam_combos = CamSettingComboBoxDict()
            self.emission_combos = EmissionComboBoxDict()
            for cam in cameras:
                cam_widget = QtWidgets.QWidget()
                cam_grid = QtWidgets.QGridLayout(cam_widget)
                self.cam_settings.addWidget(cam_widget)
                cam_config = CONFIG["Cameras"][cam]
                if "Sensitivity" in cam_config:
                    sensitivity = cam_config["Sensitivity"]
                    if "Sensitivity Categories" in cam_config:
                        self.cam_combos[cam] = []
                        categories = cam_config["Sensitivity Categories"]
                        self.cam_combos.add_categories(cam, categories)
                        for i, category in enumerate(categories):
                            row_count = cam_grid.rowCount()
                            cam_grid.addWidget(
                                QtWidgets.QLabel(category + ":"), row_count, 0
                            )
                            cat_combo = CamSettingComboBox(
                                self.cam_combos,
                                cam,
                                i,
                            )
                            cam_grid.addWidget(cat_combo, row_count, 1)
                            self.cam_combos[cam].append(cat_combo)
                        self.cam_combos[cam][0].addItems(
                            sorted(list(sensitivity.keys()))
                        )
                        for cam_combo in self.cam_combos[cam][:-1]:
                            cam_combo.currentIndexChanged.connect(
                                cam_combo.change_target_choices
                            )
                        self.cam_combos[cam][0].change_target_choices(0)
                        self.cam_combos[cam][-1].currentIndexChanged.connect(
                            self.update_sensitivity
                        )
                if "Quantum Efficiency" in cam_config:
                    try:
                        qes = cam_config["Quantum Efficiency"].keys()
                    except AttributeError:
                        pass
                    else:
                        row_count = cam_grid.rowCount()
                        cam_grid.addWidget(
                            QtWidgets.QLabel("Emission Wavelength:"),
                            row_count,
                            0,
                        )
                        emission_combo = QtWidgets.QComboBox()
                        cam_grid.addWidget(emission_combo, row_count, 1)
                        wavelengths = sorted([str(_) for _ in qes])
                        emission_combo.addItems(wavelengths)
                        emission_combo.currentIndexChanged.connect(
                            self.on_emission_changed
                        )
                        self.emission_combos[cam] = emission_combo
                spacer = QtWidgets.QWidget()
                spacer.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Preferred,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                )
                cam_grid.addWidget(spacer, cam_grid.rowCount(), 0)

        # Photon conversion
        photon_groupbox = QtWidgets.QGroupBox("Photon Conversion")
        vbox.addWidget(photon_groupbox)
        photon_grid = QtWidgets.QGridLayout(photon_groupbox)

        # EM Gain
        em_label = QtWidgets.QLabel("EM gain:")
        self.em_gain_tooltip = (
            "Electron multiplying gain of an EMCCD camera. Set it to 1 for "
            "an sCMOS sensor, which has no\nelectron multiplication stage; "
            "a value above 1 would apply the EMCCD excess-noise factor on "
            "top\nof the readout noise and inflate the uncertainties."
        )
        em_label.setToolTip(self.em_gain_tooltip)
        photon_grid.addWidget(em_label, 0, 0)
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(1, int(1e6))
        self.gain.setValue(1)
        self.gain.setToolTip(self.em_gain_tooltip)
        photon_grid.addWidget(self.gain, 0, 1)

        # Baseline
        baseline_label = QtWidgets.QLabel("Baseline:")
        baseline_label.setToolTip("Mean pixel value in the absence of light.")
        photon_grid.addWidget(baseline_label, 1, 0)
        self.baseline = QtWidgets.QDoubleSpinBox()
        self.baseline.setRange(0, 1e6)
        self.baseline.setValue(100.0)
        self.baseline.setDecimals(1)
        self.baseline.setSingleStep(0.1)
        photon_grid.addWidget(self.baseline, 1, 1)

        # Sensitivity
        sensitivity_label = QtWidgets.QLabel("Sensitivity:")
        sensitivity_label.setToolTip(
            "Camera sensitivity in counts per photon (conversion factor)."
        )
        photon_grid.addWidget(sensitivity_label, 2, 0)
        self.sensitivity = QtWidgets.QDoubleSpinBox()
        self.sensitivity.setRange(0, 1e6)
        self.sensitivity.setValue(1.0)
        self.sensitivity.setDecimals(4)
        self.sensitivity.setSingleStep(0.01)
        photon_grid.addWidget(self.sensitivity, 2, 1)

        # QE
        qe_label = QtWidgets.QLabel("Quantum efficiency:")
        qe_label.setToolTip(
            "To be deprecated in v1.0; not used in the analysis."
        )
        photon_grid.addWidget(qe_label, 3, 0)
        self.qe = QtWidgets.QDoubleSpinBox()
        self.qe.setRange(0, 1)
        self.qe.setValue(1)
        self.qe.setDecimals(2)
        self.qe.setSingleStep(0.1)
        photon_grid.addWidget(self.qe, 3, 1)

        # Camera pixel size
        px_label = QtWidgets.QLabel("Pixel size (nm):")
        px_label.setToolTip(
            "Effective camera pixel size in nm (after magnification)."
        )
        photon_grid.addWidget(px_label, 4, 0)
        self.pixelsize = QtWidgets.QSpinBox()
        self.pixelsize.setRange(0, 10000)
        self.pixelsize.setValue(130)
        self.pixelsize.setSingleStep(1)
        self.pixelsize.valueChanged.connect(self.on_pixelsize_changed)
        photon_grid.addWidget(self.pixelsize, 4, 1)

        # sCMOS per-pixel camera calibration
        camera_calib_label = QtWidgets.QLabel("sCMOS noise maps:")
        camera_calib_label.setToolTip(CAMERA_CALIB_TOOLTIP)
        photon_grid.addWidget(camera_calib_label, 5, 0)
        self.camera_calib_label = QtWidgets.QLabel(
            "-- no calibration loaded --"
        )
        self.camera_calib_label.setToolTip(CAMERA_CALIB_TOOLTIP)
        self.camera_calib_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        photon_grid.addWidget(self.camera_calib_label, 5, 1)
        # The buttons sit in the value column only, under the path label, and
        # take their natural width - spanning both columns made two small
        # buttons as wide as the whole group box.
        camera_calib_buttons = QtWidgets.QHBoxLayout()
        camera_calib_buttons.setContentsMargins(0, 0, 0, 0)
        load_camera_calib_button = QtWidgets.QPushButton("Load...")
        load_camera_calib_button.setAutoDefault(False)
        load_camera_calib_button.clicked.connect(self.load_camera_calib)
        camera_calib_buttons.addWidget(load_camera_calib_button)
        clear_camera_calib_button = QtWidgets.QPushButton("Clear")
        clear_camera_calib_button.setAutoDefault(False)
        clear_camera_calib_button.clicked.connect(
            lambda: self.update_camera_calib(None)
        )
        camera_calib_buttons.addWidget(clear_camera_calib_button)
        photon_grid.addLayout(camera_calib_buttons, 6, 1)

        # Fit Settings
        fit_groupbox = QtWidgets.QGroupBox("Fit Settings")
        vbox.addWidget(fit_groupbox)
        fit_grid = QtWidgets.QGridLayout(fit_groupbox)

        model_label = QtWidgets.QLabel("Model:")
        model_label.setToolTip(MODEL_TOOLTIP)
        fit_grid.addWidget(model_label, 1, 0)
        self.fit_model = QtWidgets.QComboBox()
        self.fit_model.addItems(list(FIT_MODELS.keys()))
        self.fit_model.setCurrentIndex(0)
        self.fit_model.setToolTip(MODEL_TOOLTIP)
        fit_grid.addWidget(self.fit_model, 1, 1)

        self.optimizer_label = QtWidgets.QLabel("Optimizer:")
        self.optimizer_label.setToolTip(OPTIMIZER_TOOLTIP)
        fit_grid.addWidget(self.optimizer_label, 2, 0)
        self.fit_optimizer = QtWidgets.QComboBox()
        self.fit_optimizer.setToolTip(OPTIMIZER_TOOLTIP)
        fit_grid.addWidget(self.fit_optimizer, 2, 1)

        self.fit_stack = QtWidgets.QStackedWidget()
        fit_grid.addWidget(self.fit_stack, 3, 0, 1, 2)
        fit_stack = self.fit_stack

        # MLE
        mle_widget = QtWidgets.QWidget()

        mle_grid = QtWidgets.QGridLayout(mle_widget)
        cc_label = QtWidgets.QLabel("Convergence criterion:")
        cc_label.setToolTip(
            "Tolerance for testing if fitting has converged: the fit stops\n"
            " once the chi-square changes by less than this, relative to\n"
            " its own magnitude."
        )
        mle_grid.addWidget(cc_label, 0, 0)
        self.convergence_criterion = QtWidgets.QDoubleSpinBox()
        # Never 0: fit requires a positive tolerance, and below ~1e-6 the
        # test is answering noise in the chi-square rather than convergence.
        self.convergence_criterion.setRange(1e-6, 1e6)
        self.convergence_criterion.setDecimals(6)
        self.convergence_criterion.setValue(0.001)
        self.convergence_criterion.setSingleStep(0.0001)
        mle_grid.addWidget(self.convergence_criterion, 0, 1)
        mi_label = QtWidgets.QLabel("Max. iterations:")
        mi_label.setToolTip("Maximum number of iterations per spot.")
        mle_grid.addWidget(mi_label, 1, 0)
        self.max_it = QtWidgets.QSpinBox()
        self.max_it.setRange(1, int(1e6))
        self.max_it.setValue(100)
        mle_grid.addWidget(self.max_it, 1, 1)

        # Shown for the one method that does not iterate ("Average of ROI");
        # an empty page keeps the stack indices stable.
        lq_widget = QtWidgets.QWidget()

        # 0 -> no optimizer parameters, 1 -> convergence criterion/max_it.
        fit_stack.addWidget(lq_widget)
        fit_stack.addWidget(mle_widget)

        # Picasso implements both least-squares and MLE estimators on the
        # GPU, so the checkbox applies to both optimizers and sits below
        # the optimizer parameter stack.
        self.gpu_checkbox = QtWidgets.QCheckBox("Use GPU")
        self.gpu_checkbox.setToolTip(
            "Perform fitting on a CUDA-capable GPU.\n\n"
            "The algorithm is a port of Gpufit; the kernels are compiled "
            "at run time by Numba.\n\n"
            "Przybylski, A., Thiel, B., Keller-Findeisen, J. et al. "
            "Gpufit: An open-source toolkit for GPU-accelerated curve "
            "fitting. Sci Rep 7, 15722 (2017). "
        )
        self.gpu_checkbox.setTristate(False)
        self.gpu_checkbox.setDisabled(True)
        self.gpu_checkbox.stateChanged.connect(self.on_gpu_fitting_changed)

        if not CUDA_AVAILABLE:
            self.gpu_checkbox.hide()
        else:
            self.gpu_checkbox.setDisabled(False)
        fit_grid.addWidget(self.gpu_checkbox, 4, 0, 1, 2)

        # Multichannel: fit the channels together or each on its own. Shown
        # by the window for multichannel / split-FOV data (see
        # Window._update_multichannel_widgets); its space is reserved while
        # hidden, so loading channels does not reflow the dialog.
        self.fit_mode_label = QtWidgets.QLabel("Fit:")
        self.fit_mode_label.setToolTip(FIT_MODE_TOOLTIP)
        fit_grid.addWidget(self.fit_mode_label, 5, 0)
        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItems([FIT_MODE_JOINT, FIT_MODE_SEPARATE])
        self.fit_mode_combo.setToolTip(FIT_MODE_TOOLTIP)
        self.fit_mode_combo.currentIndexChanged.connect(
            self.on_fit_mode_changed
        )
        fit_grid.addWidget(self.fit_mode_combo, 5, 1)
        for widget in (self.fit_mode_label, self.fit_mode_combo):
            _retain_size_when_hidden(widget)
            widget.hide()

        self.fit_model.currentIndexChanged.connect(self.on_fit_model_changed)
        self.fit_optimizer.currentIndexChanged.connect(
            self.on_fit_optimizer_changed
        )
        # Populate the optimizer combobox and set visibility for the default
        # model.
        self.on_fit_model_changed()

        # 3D via astigmatism (Gaussian models); shown for non-spline fits
        self.z_groupbox = z_groupbox = QtWidgets.QGroupBox(
            "3D via Astigmatism"
        )
        vbox.addWidget(z_groupbox)

        z_grid = QtWidgets.QGridLayout(z_groupbox)
        load_z_calib = QtWidgets.QPushButton("Load calibration")
        load_z_calib.setToolTip(
            "Load a 3D calibration file (.yaml).\n"
            "Please visit the documentation:\n"
            "https://picassosr.readthedocs.io/en/latest/\n"
            "for instructions on how to obtain it."
        )
        load_z_calib.setAutoDefault(False)
        load_z_calib.clicked.connect(self.load_z_calib)
        z_grid.addWidget(load_z_calib, 0, 1)
        self.fit_z_checkbox = QtWidgets.QCheckBox("Fit Z")
        self.fit_z_checkbox.setToolTip("Fit z coordinates?")
        self.fit_z_checkbox.setEnabled(False)
        self.z_calib_label = QtWidgets.QLabel("-- no calibration loaded --")
        self.z_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.z_calib_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        z_grid.addWidget(self.z_calib_label, 0, 0)
        magnification_label = QtWidgets.QLabel("Magnification factor:")
        magnification_label.setToolTip(
            "Factor used to correct for z-position abberation due to\n"
            "refractive index mismatch, see Huang B, et al. Science. 2008."
        )
        z_grid.addWidget(magnification_label, 1, 0)
        self.magnification_factor = QtWidgets.QDoubleSpinBox()
        self.magnification_factor.setRange(0, 1e6)
        self.magnification_factor.setDecimals(4)
        self.magnification_factor.setValue(0.79)
        z_grid.addWidget(self.magnification_factor, 1, 1)

        # The astigmatism z fit runs on whichever device the fit box's
        # 'Use GPU' checkbox selects
        fit_z_row = QtWidgets.QHBoxLayout()
        fit_z_row.addWidget(lib.HelpButton(self.CALIB_URL))
        fit_z_row.addWidget(self.fit_z_checkbox)
        z_grid.addLayout(fit_z_row, 2, 0, 1, 2)

        # Experimental PSF (cubic spline). The calibration is loaded here and
        # passed to the fit when the "Experimental PSF (cubic spline)" model
        # is chosen. Always available: the fit runs on the CPU
        # (picasso.fitting.splinefit) and, with a CUDA GPU, on the GPU
        # (picasso.fitting.splinefit_cuda).
        self.spline_groupbox = spline_groupbox = QtWidgets.QGroupBox(
            "Experimental PSF (spline)"
        )
        spline_groupbox.setToolTip(
            "Fit an experimentally measured PSF, modeled as a cubic "
            "spline. Runs on the CPU, or on a CUDA-capable GPU.\n\n"
            "Li, Y., Mund, M., Hoess, P. et al. Real-time 3D "
            "single-molecule localization using experimental point "
            "spread functions. Nat Methods 15, 367-369 (2018). "
            "https://doi.org/10.1038/nmeth.4661\n\n"
            "Babcock, H.P., Zhuang, X. Analyzing Single Molecule "
            "Localization Microscopy Data Using Cubic Splines. Sci Rep "
            "7, 552 (2017). https://doi.org/10.1038/s41598-017-00622-w"
            "\n\n"
            "Przybylski, A., Thiel, B., Keller-Findeisen, J. et al. "
            "Gpufit: An open-source toolkit for GPU-accelerated curve "
            "fitting. Sci Rep 7, 15722 (2017). "
            "https://doi.org/10.1038/s41598-017-15313-9"
            "\n\n"
            "Multichannel (global) fitting follows globLoc:\n"
            "Li, Y., Shi, W., Liu, S. et al. Global fitting for "
            "high-accuracy multi-channel single-molecule localization. "
            "Nat Commun 13, 3133 (2022). "
            "https://doi.org/10.1038/s41467-022-30719-4"
        )
        vbox.addWidget(spline_groupbox)
        spline_grid = QtWidgets.QGridLayout(spline_groupbox)
        load_spline_calib = QtWidgets.QPushButton("Load calibration")
        load_spline_calib.setToolTip(
            "Load a cubic-spline PSF calibration (.hdf5), built via\n"
            "3D > Calibrate spline PSF. Used by the 'Experimental PSF\n"
            "(cubic spline)' fit model."
        )
        load_spline_calib.setAutoDefault(False)
        load_spline_calib.clicked.connect(self.load_spline_calib)
        spline_grid.addWidget(load_spline_calib, 0, 1)
        spline_grid.addWidget(lib.HelpButton(self.SPLINE_URL), 0, 2)
        self.spline_calib_label = QtWidgets.QLabel(
            "-- no calibration loaded --"
        )
        self.spline_calib_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.spline_calib_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        spline_grid.addWidget(self.spline_calib_label, 0, 0)

        self.link_photons_checkbox = QtWidgets.QCheckBox(
            "Link photon counts across channels"
        )
        self.link_photons_checkbox.setToolTip(LINK_PHOTONS_TIP)
        self.link_photons_checkbox.setTristate(False)
        self.link_photons_checkbox.setChecked(True)
        self.link_photons_checkbox.hide()  # shown for 2-6 channel calibs
        spline_grid.addWidget(self.link_photons_checkbox, 1, 0, 1, 3)

        # Channel registration for the multichannel (joint) 2D spherical
        # Gaussian fit. Shown only for that model; unlike the spline path this
        # needs no measured PSF, only where each channel sits.
        self.channel_registration_groupbox = reg_groupbox = (
            QtWidgets.QGroupBox("Multichannel: channel registration")
        )
        reg_groupbox.setToolTip(
            "Fit all loaded channels at once with one shared position and\n"
            "width (a global fit, as in globLoc), instead of fitting the\n"
            "active channel alone.\n\n"
            "Needs a channel registration built with\n"
            "Calibration > Register channels, from beads or from the\n"
            "blinking signal itself. Without one the fit stays\n"
            "single-channel.\n\n"
            "Li, Y., Shi, W., Liu, S. et al. Global fitting for "
            "high-accuracy multi-channel single-molecule localization. "
            "Nat Commun 13, 3133 (2022). "
            "https://doi.org/10.1038/s41467-022-30719-4"
        )
        vbox.addWidget(reg_groupbox)
        reg_grid = QtWidgets.QGridLayout(reg_groupbox)
        load_registration = QtWidgets.QPushButton("Load registration")
        load_registration.setToolTip(
            "Load a channel registration (.yaml) built via\n"
            "Calibration > Register channels."
        )
        load_registration.setAutoDefault(False)
        load_registration.clicked.connect(self.load_channel_registration)
        reg_grid.addWidget(load_registration, 0, 1)
        clear_registration = QtWidgets.QPushButton("Clear")
        clear_registration.setAutoDefault(False)
        clear_registration.clicked.connect(
            lambda: self.update_channel_registration(None)
        )
        reg_grid.addWidget(clear_registration, 0, 2)
        self.channel_registration_label = QtWidgets.QLabel(
            "-- no registration loaded --"
        )
        self.channel_registration_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.channel_registration_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        reg_grid.addWidget(self.channel_registration_label, 0, 0)

        self.gauss_link_photons_checkbox = QtWidgets.QCheckBox(
            "Link photon counts across channels"
        )
        self.gauss_link_photons_checkbox.setToolTip(GAUSS_LINK_PHOTONS_TIP)
        self.gauss_link_photons_checkbox.setTristate(False)
        # Unchecked by default, unlike the spline fit: a Gaussian carries no
        # per-channel brightness scale, so linking assumes every channel
        # collects the same photons (see GAUSS_LINK_PHOTONS_TIP).
        self.gauss_link_photons_checkbox.setChecked(False)
        self.gauss_link_photons_checkbox.hide()  # shown for 2-6 channels
        reg_grid.addWidget(self.gauss_link_photons_checkbox, 1, 0, 1, 3)

        # Lateral (x, y) corrections kept outside the calibration used for
        # fitting, applied after it. For 2D data this is the only route;
        # for 3D it is the alternative to appending the correction to the
        # 3D / spline calibration, and gives the same coordinates.
        affine_groupbox = QtWidgets.QGroupBox("Lateral correction (x, y)")
        affine_groupbox.setToolTip(
            "Corrections applied to the fitted x/y, typically a\n"
            "chromatic-aberration correction. Build one with\n"
            "3D > Calibrate lateral transform and load it here.\n\n"
            "Works with 2D and 3D fits alike. With astigmatic 3D fitting\n"
            "these are applied after the z fit, on top of any correction\n"
            "the 3D calibration carries - so keeping a correction in its\n"
            "own file gives the same result as appending it to the 3D\n"
            "calibration (the recommended route, since the calibration\n"
            "then travels with its correction).\n\n"
            "A correction the loaded 3D or spline calibration already\n"
            "carries is refused here: it is applied during the fit, and\n"
            "applying it twice would correct the coordinates twice.\n\n"
            "Several files are applied in the order listed. Single-channel\n"
            "data only: a multichannel (global) fit registers its\n"
            "channels itself and ignores these."
        )
        vbox.addWidget(affine_groupbox)
        affine_grid = QtWidgets.QGridLayout(affine_groupbox)
        load_affine_calib = QtWidgets.QPushButton("Load correction")
        load_affine_calib.setToolTip(
            "Load a standalone lateral calibration (.yaml) to apply to\n"
            "this measurement, 2D or 3D."
        )
        load_affine_calib.setAutoDefault(False)
        load_affine_calib.clicked.connect(self.load_affine_calib)
        affine_grid.addWidget(load_affine_calib, 0, 1)
        clear_affine_calib = QtWidgets.QPushButton("Clear")
        clear_affine_calib.setAutoDefault(False)
        clear_affine_calib.clicked.connect(
            lambda: self.update_affine_calib(None)
        )
        affine_grid.addWidget(clear_affine_calib, 0, 2)
        self.affine_calib_label = QtWidgets.QLabel(
            "-- no correction loaded --"
        )
        self.affine_calib_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.affine_calib_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        affine_grid.addWidget(self.affine_calib_label, 0, 0)

        # show the calibration box that matches the initial fit model
        self._update_calib_group_visibility()

        if "Cameras" in CONFIG:
            camera = self.camera.currentText()
            if camera in CONFIG["Cameras"]:
                self.on_camera_changed(0)
                camera_config = CONFIG["Cameras"][camera]
                if (
                    "Sensitivity" in camera_config
                    and "Sensitivity Categories" in camera_config
                ):
                    self.update_sensitivity()

        # Sample quality
        quality_groupbox = QtWidgets.QGroupBox(
            "Postprocessing and Sample Quality"
        )
        quality_groupbox.setToolTip(
            "Drift correction + drift, bright time and NeNA estimates."
        )
        vbox.addWidget(quality_groupbox)
        quality_grid = QtWidgets.QGridLayout(quality_groupbox)

        # drift correction
        self.aim_undrift_checkbox = QtWidgets.QCheckBox("Apply AIM")
        self.aim_undrift_checkbox.setToolTip(
            "Apply the AIM for drift correction"
        )
        self.aim_undrift_checkbox.setChecked(False)
        self.aim_undrift_checkbox.stateChanged.connect(
            self.on_aim_undrift_changed
        )
        quality_grid.addWidget(self.aim_undrift_checkbox, 0, 0)

        aim_segmentation_label = QtWidgets.QLabel("Segmentation")
        aim_segmentation_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        aim_segmentation_label.setToolTip(
            "Select the number of frames in a segment for AIM."
        )
        quality_grid.addWidget(aim_segmentation_label, 0, 1)
        self.aim_segmentation = QtWidgets.QSpinBox()
        self.aim_segmentation.setRange(1, 100_000)
        self.aim_segmentation.setValue(1000)
        self.aim_segmentation.setSingleStep(100)
        self.aim_segmentation.setEnabled(False)
        quality_grid.addWidget(self.aim_segmentation, 0, 2)

        self.fiducial_check = QtWidgets.QCheckBox("Use fiducials")
        self.fiducial_check.setToolTip(
            "Use fiducial markers for drift correction?\n"
        )
        self.fiducial_check.setChecked(False)
        quality_grid.addWidget(self.fiducial_check, 0, 3)

        # sample quality
        self.quality_check = QtWidgets.QPushButton(
            "Estimate and add to database"
        )
        self.quality_check.setEnabled(False)
        quality_grid.addWidget(self.quality_check, 1, 0, 1, 4)
        self.quality_check.clicked.connect(self.check_quality)

        self.quality_grid_labels = [
            QtWidgets.QLabel("Locs/frame"),
            QtWidgets.QLabel("NeNA"),
            QtWidgets.QLabel("Mean drift"),
            QtWidgets.QLabel("Bright time (frames)"),
        ]
        for idx, _ in enumerate(self.quality_grid_labels):
            quality_grid.addWidget(_, idx + 1, 1)

        self.quality_grid_values = [
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
            QtWidgets.QLabel(""),
        ]

        for idx, _ in enumerate(self.quality_grid_values):
            quality_grid.addWidget(_, idx + 1, 2)

        self.reset_quality_check()
        vbox.addStretch(1)

        # adjust the size of the dialog to fit its contents, but cap the
        # height so the dialog stays a reasonable size even on very tall
        # screens - the scroll area (set up above) takes care of the rest
        hint = container.sizeHint()
        lib.adjust_widget_size(self, hint, max_height=850)

    def reset_quality_check(self) -> None:
        """Reset the quality check UI elements."""
        self.quality_check.setEnabled(False)
        self.quality_check.setVisible(True)

        for _ in self.quality_grid_labels:
            _.setVisible(False)

        for _ in self.quality_grid_values:
            _.setVisible(False)
            _.setText("")

    def on_pixelsize_changed(self) -> None:
        """If the movie is loaded and scale bar is shown, update it."""
        if (
            hasattr(self.window, "movie")
            and self.window.movie is not None
            and hasattr(self.window, "scalebar_action")
            and self.window.scalebar_action.isChecked()
        ):
            self.window.draw_frame()

    def update_roi_display(self, skip_dialog: bool = False) -> None:
        """Refresh the compact ROI field (and the ROI sub-dialog) from
        the view's ROIs.

        The field shows nothing (placeholder "Whole frame") when there
        are no ROIs, the four editable coordinates when there is exactly
        one, and a read-only count when there are several.

        Parameters
        ----------
        skip_dialog : bool, optional
            If True, do not refresh the ROI sub-dialog's table (used when
            the call originates from that dialog to avoid recursion).
        """
        view = self.window.view
        n = len(view.rois)
        self._updating_roi_field = True
        if n == 1:
            self.roi_field.setReadOnly(False)
            (y_min, x_min), (y_max, x_max) = view.rois[0]
            self.roi_field.setText(
                f"{int(y_min)},{int(x_min)},{int(y_max)},{int(x_max)}"
            )
        elif n == 0:
            self.roi_field.setReadOnly(False)
            self.roi_field.setText("")
        else:
            self.roi_field.setReadOnly(True)
            self.roi_field.setText(f"{n} ROIs")
        self._updating_roi_field = False
        # split-FOV: keep the per-region thresholds aligned with the regions
        # and put the selected region's on the slider
        self.window.region_mngs()
        self.sync_mng_to_selected_region()
        # ... and the selected region's own fit settings on the fit widgets
        self.window.sync_region_fit_params()
        if not skip_dialog and self.roi_dialog is not None:
            self.roi_dialog.update_table()

    def on_split_fov_changed(self) -> None:
        """Toggle split-FOV region mode (ROIs become channels of one movie)."""
        self.window.set_split_fov_mode(self.split_fov_checkbox.isChecked())

    def on_roi_field_changed(self) -> None:
        """Clear the ROIs when the user empties the field."""
        if self._updating_roi_field or self.roi_field.isReadOnly():
            return
        if self.roi_field.text() == "":
            self.window.view.rois = []
            self.window.view.selected_roi = None
            if self.roi_dialog is not None:
                self.roi_dialog.update_table()
            self.window.draw_frame()

    def on_roi_field_finished(self) -> None:
        """Parse the single ROI typed into the compact field."""
        if self._updating_roi_field or self.roi_field.isReadOnly():
            return
        text = self.roi_field.text()
        if text == "":
            return
        try:
            y_min, x_min, y_max, x_max = (int(v) for v in text.split(","))
        except ValueError:
            return  # incomplete input, wait for the user to finish
        box = self.window.parameters.get("Box Size", 7)
        self.window.view.rois = localize.clip_rois(
            [[[y_min, x_min], [y_max, x_max]]], min_size=box
        )
        self.window.view.selected_roi = None
        self.update_roi_display()
        self.window.draw_frame()

    def on_edit_rois(self) -> None:
        """Open (or raise) the ROI management sub-dialog."""
        if self.roi_dialog is None:
            self.roi_dialog = ROIDialog(self.window)
        self.roi_dialog.update_table()
        self.roi_dialog.show()
        self.roi_dialog.raise_()
        self.roi_dialog.activateWindow()

    def on_fit_model_changed(self) -> None:
        """Repopulate the optimizer combobox for the selected model and
        show/hide the optimizer controls. Models without an optimizer
        (e.g. averaging) hide the optimizer row and its parameters."""
        model = self.fit_model.currentText()
        optimizers = FIT_MODELS[model]["optimizers"]
        if optimizers is None:
            self.optimizer_label.hide()
            self.fit_optimizer.hide()
            self.fit_stack.hide()
            self._apply_convergence_defaults()
        else:
            self.optimizer_label.show()
            self.fit_optimizer.show()
            self.fit_stack.show()
            self.fit_optimizer.blockSignals(True)
            self.fit_optimizer.clear()
            self.fit_optimizer.addItems(list(optimizers.keys()))
            self.fit_optimizer.setCurrentIndex(0)
            self.fit_optimizer.blockSignals(False)
            self.on_fit_optimizer_changed()
        self._update_calib_group_visibility()

    def _update_calib_group_visibility(self) -> None:
        """Show only the calibration box relevant to the selected fit model:
        the astigmatism z-calibration for Gaussian models, the spline PSF
        calibration for the experimental-PSF (spline) model, and the channel
        registration for the model that has a multichannel fit. Guarded so the
        initial ``on_fit_model_changed`` call (before the boxes exist) is a
        no-op."""
        entry = FIT_MODELS[self.fit_model.currentText()]
        needs_spline = entry.get("needs_spline_calibration", False)
        needs_registration = entry.get("needs_channel_registration", False)
        if self.z_groupbox is not None:
            self.z_groupbox.setVisible(not needs_spline)
        if self.spline_groupbox is not None:
            self.spline_groupbox.setVisible(needs_spline)
        if self.channel_registration_groupbox is not None:
            self.channel_registration_groupbox.setVisible(needs_registration)

    def current_fit_code(self) -> str:
        """The ``fit`` code the current selection would run."""
        model = self.fit_model.currentText()
        optimizer = self.fit_optimizer.currentText()
        if not optimizer and FIT_MODELS[model]["optimizers"] is not None:
            return ""
        return _effective_fit_code(
            _fit_code(model, optimizer), self.gpu_checkbox.isChecked()
        )

    def _apply_convergence_defaults(self) -> None:
        """Show the selected method's own convergence schedule."""
        code = self.current_fit_code()
        if not code:
            return
        self.fit_stack.setCurrentIndex(1 if code in _CONVERGENCE_CODES else 0)
        if code == self._last_fit_code:
            return
        self._last_fit_code = code
        defaults = _CONVERGENCE_DEFAULTS.get(code)
        if defaults is not None:
            self.convergence_criterion.setValue(defaults[0])
            self.max_it.setValue(defaults[1])

    def on_fit_optimizer_changed(self) -> None:
        """Enable/disable the GPU fitting checkbox based on the selected
        model and optimizer, then show that method's convergence schedule."""
        model = self.fit_model.currentText()
        optimizer = self.fit_optimizer.currentText()
        code = _fit_code(model, optimizer) if optimizer else ""
        if code in _GPU_CAPABLE_CODES:
            self.gpu_checkbox.setDisabled(not CUDA_AVAILABLE)
        else:
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setDisabled(True)
        self.on_gpu_fitting_changed()

    def on_frames_edit_changed(self) -> None:
        """Handle changes when text is deleted."""
        if self.frames_edit.text() == "":
            self.window.frame_range = None

    def on_frames_edit_finished(self) -> None:
        """Handle the completion of frames editing. Parses one or more
        semicolon-separated ``min,max`` segments (1-indexed, inclusive),
        clamps each to the movie length, and stores them as a list of
        0-indexed ``[lo, hi]`` segments in ``window.frame_range``."""
        if not self.window.movie or self.frames_edit.text() == "":
            return
        n_frames = len(self.window.movie)
        segments = []
        for part in self.frames_edit.text().split(";"):
            min_frame, max_frame = [int(_) for _ in part.split(",")]
            min_frame = max(1, min_frame)
            max_frame = min(n_frames, max_frame)
            segments.append([min_frame, max_frame])
        # store as 0-indexed inclusive segments
        self.window.frame_range = [[lo - 1, hi - 1] for lo, hi in segments]
        self.frames_edit.setText(
            "; ".join(f"{lo},{hi}" for lo, hi in segments)
        )

    def load_z_calib(self) -> None:
        """Load the 3D calibration from a user-selected YAML file."""
        if self.z_calibration_path:
            dialog_directory, _ = os.path.split(self.z_calibration_path)
        else:
            dialog_directory = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load 3d calibration",
            directory=dialog_directory,
            filter="*.yaml",
        )
        if path:
            self.update_z_calib(path)

    def load_camera_calib(self) -> None:
        """Load a per-pixel sCMOS camera calibration from an HDF5."""
        if self.camera_calibration_path:
            dialog_directory, _ = os.path.split(self.camera_calibration_path)
        else:
            dialog_directory = None
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load sCMOS camera calibration",
            directory=dialog_directory,
            filter="*.hdf5",
        )
        if path:
            self.update_camera_calib(path)

    def config_calib_path(self, section: str) -> str | None:
        """Path configured in ``section`` for the selected camera and
        emission wavelength, or ``None`` when the config has no entry for it.

        The three calibration sections a camera configuration can carry
        ("z-calibrations", "spline-calibrations" and "camera-calibrations")
        are all keyed by camera and then by emission wavelength. A single
        path in place of the wavelength mapping is also accepted and then
        serves every wavelength - one sensor read out one way needs one set
        of maps, and repeating the same path under each wavelength is only a
        way for them to drift apart.

        ``None`` means *clear*: switching to a camera or wavelength the
        config knows nothing about must not leave the previous camera's
        calibration in place, since it no longer describes the data (see the
        callers).
        """
        section_config = CONFIG.get(section)
        if not isinstance(section_config, dict):
            return None
        if not hasattr(self, "camera"):
            return None
        camera = self.camera.currentText()
        entry = section_config.get(camera)
        if isinstance(entry, dict):
            em_combo = self.emission_combos.get(camera)
            if em_combo is None:
                # wavelength-keyed, but this camera has no emission combo
                # (no Quantum Efficiency mapping): nothing selects an entry
                return None
            text = em_combo.currentText()
            try:
                wavelength = int(text)
            except ValueError:
                wavelength = text
            entry = entry.get(wavelength)
        return entry or None

    def update_camera_calib_with_config_path(self) -> None:
        """Pick up the calibration configured for the selected camera.

        Switching to a camera or wavelength the config has no entry for
        *clears* the calibration rather than leaving the previous one in
        place.
        """
        if "camera-calibrations" not in CONFIG:
            # nothing is configured for any camera, so nothing was
            # auto-loaded: leave a manually loaded calibration alone
            return
        self.update_camera_calib(self.config_calib_path("camera-calibrations"))

    #: Index of each photon-conversion scalar in ``_scalars_before_calib``.
    _SCALARS = {"gain": 0, "baseline": 1, "sensitivity": 2}

    def set_photon_scalar(self, name: str, value: float) -> None:
        """Set a photon-conversion scalar from the camera configuration.

        While a calibration supersedes one of them its spinbox is disabled,
        but the configuration keeps writing to it on every camera and
        wavelength change. Those writes must not land on the widget - it
        shows the map's median - yet they must not be lost either, or
        clearing the calibration would restore a scalar from whichever camera
        happened to be selected when it was loaded. So a superseded scalar is
        recorded as the value a later Clear restores, and only a scalar still
        in force is written to its spinbox.
        """
        widget = getattr(self, name)
        if self._scalars_before_calib is not None:
            self._scalars_before_calib[self._SCALARS[name]] = value
        if widget.isEnabled():
            widget.setValue(value)

    def _apply_camera_calib_scalars(self) -> None:
        """Set the superseded scalars to the calibration's own medians.

        A disabled spinbox still reading 100.0 while the maps say 498.7 is
        misleading, and the value is not inert: it goes into the saved
        ``.yaml`` as the camera information the run used. The median of the
        map that replaced it makes the frozen number both honest and
        informative - the map is what is applied, pixel by pixel, and its
        median is the one number that summarizes it.
        """
        if self._scalars_before_calib is None:
            self._scalars_before_calib = [
                self.gain.value(),
                self.baseline.value(),
                self.sensitivity.value(),
            ]
        self.gain.setValue(1)
        self.baseline.setValue(
            float(np.median(self.camera_calibration["offset"]))
        )
        gain_map = self.camera_calibration.get("gain")
        if gain_map is not None:
            # Picasso's Sensitivity is electrons per count, the reciprocal of
            # the gain in counts per electron that the calibration measures.
            self.sensitivity.setValue(1.0 / float(np.median(gain_map)))

    def _restore_camera_calib_scalars(self) -> None:
        """Put back the scalars that were in place before a calibration."""
        if self._scalars_before_calib is None:
            return
        gain, baseline, sensitivity = self._scalars_before_calib
        self._scalars_before_calib = None
        self.gain.setValue(gain)
        self.baseline.setValue(baseline)
        self.sensitivity.setValue(sensitivity)

    def update_camera_calib(self, path: str | None) -> None:
        """Load, or clear with ``None``, the sCMOS camera calibration.

        While one is loaded the scalars it supersedes are set to the maps'
        medians and disabled, rather than left at a stale value that no
        longer describes the run: Baseline and EM gain always, Sensitivity
        only when the calibration carries a gain map. Clearing restores what
        was there before.
        """
        if path:
            try:
                calibration = io.load_camera_calibration(path)
            except Exception as error:
                QtWidgets.QMessageBox.critical(
                    self, "Camera calibration", str(error)
                )
                return
            self.camera_calibration = calibration
            self.camera_calibration_path = path
            self.camera_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            self.camera_calib_label.setText(os.path.basename(path))
            has_gain = calibration.get("gain") is not None
            self.camera_calib_label.setToolTip(
                f"{path}\n\n"
                f"{calibration.get('Height')} x {calibration.get('Width')} "
                f"pixels, {calibration.get('Frames')} dark frames\n"
                "median variance "
                + format(
                    calibration.get("Variance median (ADU^2)", float("nan")),
                    ".2f",
                )
                + " ADU^2, "
                f"{calibration.get('Hot pixels', 0)} hot pixels\n"
                + (
                    "gain map present: Sensitivity is superseded"
                    if has_gain
                    else "no gain map: the scalar Sensitivity is still used"
                )
            )
        else:
            self.camera_calibration = {}
            self.camera_calibration_path = None
            self.camera_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.camera_calib_label.setText("-- no calibration loaded --")
            self.camera_calib_label.setToolTip(CAMERA_CALIB_TOOLTIP)
            has_gain = False

        loaded = bool(self.camera_calibration)
        if loaded:
            self._apply_camera_calib_scalars()
        else:
            self._restore_camera_calib_scalars()

        self.baseline.setEnabled(not loaded)
        self.baseline.setToolTip(
            "Replaced by the per-pixel offset map; showing its median."
            if loaded
            else "Mean pixel value in the absence of light."
        )
        self.sensitivity.setEnabled(not has_gain)
        self.sensitivity.setToolTip(
            "Replaced by the per-pixel gain map; showing 1 / its median."
            if has_gain
            else "Camera sensitivity in counts per photon (conversion "
            "factor)."
        )
        self.gain.setEnabled(not loaded)
        self.gain.setToolTip(
            "Fixed at 1: an sCMOS sensor has no electron multiplication."
            if loaded
            else self.em_gain_tooltip
        )

    def load_spline_calib(self) -> None:
        """Load a cubic-spline PSF calibration from a user-selected HDF5."""
        if self.spline_calibration_path:
            dialog_directory, _ = os.path.split(self.spline_calibration_path)
        else:
            dialog_directory = None
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load spline PSF calibration",
            directory=dialog_directory,
            filter="*.hdf5",
        )
        if path:
            self.update_spline_calib(path)

    def load_affine_calib(self) -> None:
        """Load one or more standalone lateral calibrations to apply after
        fitting, whether the fit is 2D or 3D.

        Any calibration file works as a carrier - only its lateral
        corrections are read. With astigmatic 3D fitting they are applied
        after the z fit, on top of the ones the 3D calibration carries, so
        a correction kept in its own file lands exactly where an appended
        one would; a correction the loaded 3D / spline calibration already
        carries is rejected here, since that one is applied during the fit.
        Selecting several files applies them in the order chosen.
        """
        if self.affine_calibration_paths:
            dialog_directory, _ = os.path.split(
                self.affine_calibration_paths[-1]
            )
        else:
            dialog_directory = None
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load lateral correction",
            directory=dialog_directory,
            filter="Calibration files (*.yaml *.hdf5)",
        )
        if paths:
            self.update_affine_calib(paths)

    def update_affine_calib(self, paths: list[str] | None) -> None:
        """Load (or clear) the lateral corrections applied after fitting."""
        if not paths:
            self.lateral_transforms = []
            self.affine_calibration_paths = []
            self.affine_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.affine_calib_label.setText("-- no correction loaded --")
            self.affine_calib_label.setToolTip("")
            return

        transforms, loaded, empty, already = [], [], [], []
        for path in paths:
            try:
                calibration = io.load_any_calibration(path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Load lateral correction",
                    f"Could not read {os.path.basename(path)}:\n{e}",
                )
                continue
            found = lib.lateral_transforms(calibration)
            if not found:
                empty.append(os.path.basename(path))
                continue
            # A correction the loaded 3D / spline calibration already carries
            # is applied by the fit itself; taking it here as well would
            # correct the coordinates twice. This catches the common case -
            # picking the 3D calibration file itself - at load time, while
            # the fit-time check (see ``_extra_lateral_transforms``) also
            # catches a copy of the same transform saved elsewhere.
            found, duplicates = lib.drop_duplicate_lateral_transforms(
                found, self._calibration_lateral_transforms()
            )
            already.extend(duplicates)
            if not found:
                continue
            transforms.extend(found)
            loaded.append(path)
        if empty:
            QtWidgets.QMessageBox.warning(
                self,
                "Load lateral correction",
                "No affine corrections found in: "
                + ", ".join(empty)
                + ".\nBuild one with 3D > Calibrate lateral transform.",
            )
        if already:
            QtWidgets.QMessageBox.warning(
                self,
                "Load lateral correction",
                "Not loaded: "
                + ", ".join(lib.describe_lateral_transforms(already))
                + ".\n\nThe loaded 3D / spline calibration already carries "
                "this correction and applies it during the fit. Loading it "
                "here as well would correct the coordinates twice.",
            )
        if not transforms:
            self.update_affine_calib(None)
            return
        self._set_affine_state(transforms, loaded)

    def _calibration_lateral_transforms(self) -> list[dict]:
        """The affine corrections the currently loaded 3D and spline
        calibrations carry, i.e. the ones the fit applies by itself."""
        return lib.lateral_transforms(
            self.z_calibration
        ) + lib.lateral_transforms(self.spline_calibration)

    def _set_affine_state(self, transforms: list, paths: list[str]) -> None:
        """Store already-loaded affine corrections and label them. Kept
        apart from ``update_affine_calib`` so restoring a channel's
        parameters does not re-read the calibration files."""
        if not transforms:
            self.update_affine_calib(None)
            return
        self.lateral_transforms = transforms
        self.affine_calibration_paths = paths
        described = lib.describe_lateral_transforms(transforms)
        self.affine_calib_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.affine_calib_label.setText(
            ", ".join(t.get("Type", "affine") for t in transforms)
        )
        self.affine_calib_label.setToolTip(
            "\n".join(paths)
            + "\n\nApplied in this order after fitting:\n"
            + "\n".join(f"{i + 1}. {d}" for i, d in enumerate(described))
        )

    def _set_spline_state(self, calibration: dict, path: str | None) -> None:
        """Store an already-loaded spline PSF calibration and label it.

        Kept apart from ``update_spline_calib`` (as ``_set_affine_state`` is
        from ``update_affine_calib``) so that restoring a channel's own
        calibration neither re-reads the file nor triggers that function's
        side effects - entering split-FOV mode and dropping the channel sum,
        neither of which a channel switch should do."""
        self.spline_calibration = calibration or {}
        self.spline_calibration_path = path
        if path:
            self.spline_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
            )
            self.spline_calib_label.setText(os.path.basename(path))
            self.spline_calib_label.setToolTip(path)
        else:
            self.spline_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.spline_calib_label.setText("-- no calibration loaded --")
            self.spline_calib_label.setToolTip("")
        self._update_link_photons_visibility()

    def update_spline_calib_with_config_path(self) -> None:
        """Retrieve the spline PSF calibration path that corresponds to the
        selected camera and emission wavelength, from the config.

        Switching to a camera or wavelength the config has no entry for
        *clears* the calibration rather than leaving the previous one in
        place.
        """
        if self.spline_groupbox is None:  # fit UI not built yet
            return
        if "spline-calibrations" not in CONFIG:
            # nothing is configured for any camera, so nothing was
            # auto-loaded: leave a manually loaded calibration alone
            return
        self.update_spline_calib(self.config_calib_path("spline-calibrations"))

    def update_spline_calib(self, path: str | None) -> None:
        """Load (or clear) a cubic-spline PSF calibration from an HDF5 file."""
        if path:
            if os.path.exists(path):
                try:
                    self.spline_calibration = io.load_spline_calibration(path)
                except Exception as e:
                    self.update_spline_calib(None)
                    self.spline_calib_label.setText(
                        "-- invalid calibration --"
                    )
                    self.spline_calib_label.setToolTip(str(e))
                    return
                self.spline_calibration_path = path
            else:
                self.update_spline_calib(None)
                self.spline_calib_label.setText(
                    "-- calibration path not found --"
                )
                self.spline_calib_label.setToolTip("")
                return
            self.spline_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
            )
            self.spline_calib_label.setText(os.path.basename(path))
            self.spline_calib_label.setToolTip(path)
            # split-FOV: drop the calibration's channel regions into the view
            # and enter split-FOV mode, so the registration can be inspected and
            # fine-tuned on this data (arrow-key nudge / drag) and re-drawn if
            # the split moved. The fit then uses whatever regions are shown.
            if self.spline_calibration.get("split_fov"):
                regions = self.spline_calibration.get("regions") or []
                self.window.view.rois = [
                    [
                        [int(r[0][0]), int(r[0][1])],
                        [int(r[1][0]), int(r[1][1])],
                    ]
                    for r in regions
                ]
                self.window.view.selected_roi = None
                self.split_fov_checkbox.blockSignals(True)
                self.split_fov_checkbox.setChecked(True)
                self.split_fov_checkbox.blockSignals(False)
                self.window.set_split_fov_mode(True)
                self.update_roi_display()
                self.window.draw_frame()
        else:
            nothing_loaded = not (
                self.spline_calibration or self.spline_calibration_path
            )
            if nothing_loaded:
                # nothing to clear: a camera or wavelength change must not
                # drop the channel sum and redraw for no reason. The load
                # error paths above still set their own label afterwards.
                return
            self.spline_calibration = {}
            self.spline_calibration_path = None
            self.spline_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.spline_calib_label.setText("-- no calibration loaded --")
            self.spline_calib_label.setToolTip("")
        self._update_link_photons_visibility()
        # The channel sum may have been registered from the calibration that
        # has just been replaced. Dropping it re-registers it from the new one
        # on the next draw, which is also what brings the summed view up when
        # a calibration is loaded while 'Identify on' is set to the sum (see
        # ``Window.ensure_channel_sum``).
        self.window.drop_channel_sum()
        try:
            if self.window.movie is not None:
                self.window._reset_contrast_to_frame()
                self.window.draw_frame()
        except (AttributeError, RuntimeError):
            pass  # called during startup, before the window is fully built

    def _update_link_photons_visibility(self) -> None:
        """Show the 'Link photons across channels' checkbox only for a
        multichannel spline calibration with 2 to
        ``precision._LINK_XYZ_MAX_CHANNELS`` channels - the range for which
        the photon-decoupled (link-XYZ) fit is supported."""
        # A config auto-load may call update_spline_calib during startup before
        # the fit UI (and this checkbox) is built; ignore until it exists.
        if not hasattr(self, "link_photons_checkbox"):
            return
        cal = self.spline_calibration or {}
        # (see _link_photons_enabled for how the state reaches the fit workers)
        n_channels = int(
            cal.get("n_channels", len(cal.get("channel_transforms", []) or []))
        )
        is_multichannel = cal.get("model") == "spline-3d-multichannel"
        show = (
            is_multichannel
            and self.joint_fit_selected()
            and (2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS)
        )
        if show:
            # default the toggle to the mode chosen when the calibration was
            # built (stored as "link_photons"); the user can still flip it.
            self.link_photons_checkbox.setChecked(
                bool(cal.get("link_photons", True))
            )
        self.link_photons_checkbox.setVisible(show)

    def _link_photons_enabled(self) -> bool:
        """Whether the multichannel spline fit should link photons across
        channels (shared amplitude, model 11). True when the checkbox is absent
        (no GPU / non-multichannel) so behavior is unchanged by default."""
        cb = getattr(self, "link_photons_checkbox", None)
        return cb.isChecked() if cb is not None else True

    def load_channel_registration(self) -> None:
        """Load a standalone channel registration from a user-selected file."""
        dialog_directory = (
            os.path.dirname(self.channel_registration_path)
            if self.channel_registration_path
            else (
                os.path.dirname(self.window.movie_path)
                if self.window.movie_path
                else ""
            )
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load channel registration",
            directory=dialog_directory,
            filter="*.yaml",
        )
        if path:
            self.update_channel_registration(path)

    def update_channel_registration(self, path: str | None) -> None:
        """Load (or clear) the standalone channel registration.

        A file that cannot be read, or that carries no ``channel_transforms``,
        is reported in the label and leaves no registration loaded, rather than
        being accepted and failing later at fit time.

        Parameters
        ----------
        path : str or None
            The registration file to load. None clears the loaded one.
        """
        if path:
            if not os.path.exists(path):
                self.update_channel_registration(None)
                self.channel_registration_label.setText(
                    "-- registration path not found --"
                )
                return
            try:
                calibration = io.load_any_calibration(path)
            except Exception as e:
                self.update_channel_registration(None)
                self.channel_registration_label.setText(
                    "-- invalid registration --"
                )
                self.channel_registration_label.setToolTip(str(e))
                return
            if not calibration.get("channel_transforms"):
                self.update_channel_registration(None)
                self.channel_registration_label.setText(
                    "-- not a channel registration --"
                )
                self.channel_registration_label.setToolTip(
                    "This file carries no 'channel_transforms'. Build one "
                    "with Calibration > Register channels."
                )
                return
            self.channel_registration_calibration = calibration
            self.channel_registration_path = path
            self.channel_registration_label.setText(os.path.basename(path))
            self.channel_registration_label.setToolTip(path)
            # split-FOV: drop the registration's regions into the view and
            # enter split-FOV mode, as loading a split-FOV spline calibration
            # does. They are only the defaults - the ROIs can be re-drawn and
            # the channels are re-placed at them.
            if calibration.get("split_fov"):
                regions = calibration.get("regions") or []
                self.window.view.rois = [
                    [
                        [int(r[0][0]), int(r[0][1])],
                        [int(r[1][0]), int(r[1][1])],
                    ]
                    for r in regions
                ]
                self.window.view.selected_roi = None
                self.split_fov_checkbox.blockSignals(True)
                self.split_fov_checkbox.setChecked(True)
                self.split_fov_checkbox.blockSignals(False)
                self.window.set_split_fov_mode(True)
                self.update_roi_display()
                self.window.draw_frame()
        else:
            self.channel_registration_calibration = {}
            self.channel_registration_path = None
            self.channel_registration_label.setText(
                "-- no registration loaded --"
            )
            self.channel_registration_label.setToolTip("")
        self._update_gauss_link_photons_visibility()
        try:
            if self.window.movie is not None:
                self.window.draw_frame()
        except (AttributeError, RuntimeError):
            pass  # called during startup, before the window is fully built

    def link_calibration(self) -> dict:
        """The loaded calibration whose registration the link colors pair by.

        Either box can hold one - a spline PSF calibration or a standalone
        channel registration - and only the one belonging to the selected fit
        model is even shown (see ``_update_calib_group_visibility``), so that
        model decides which comes first. The other is still used when it is
        the only one loaded: it registers the same channels either way, and
        showing its pairing beats falling back to the estimate.
        """
        spline = self.spline_calibration or {}
        registration = self.channel_registration_calibration or {}
        entry = FIT_MODELS[self.fit_model.currentText()]
        if entry.get("needs_channel_registration"):
            return registration or spline
        return spline or registration

    def _update_gauss_link_photons_visibility(self) -> None:
        """Show the Gaussian 'Link photons' checkbox only for a registration
        with 2 to ``precision._LINK_XYZ_MAX_CHANNELS`` channels - the range the
        photon-decoupled model supports."""
        if not hasattr(self, "gauss_link_photons_checkbox"):
            return
        calibration = self.channel_registration_calibration or {}
        n_channels = int(
            calibration.get(
                "n_channels",
                len(calibration.get("channel_transforms", []) or []),
            )
        )
        show = self.joint_fit_selected() and (
            2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS
        )
        self.gauss_link_photons_checkbox.setVisible(show)

    def _gauss_link_photons_enabled(self) -> bool:
        """Whether the multichannel Gaussian fit should share one photon count
        across channels. False when the checkbox is absent, matching the
        decoupled default (see GAUSS_LINK_PHOTONS_TIP)."""
        cb = getattr(self, "gauss_link_photons_checkbox", None)
        return cb.isChecked() if cb is not None else False

    def update_z_calib_with_config_path(self) -> None:
        """Retrieve the z calibration path that corresponds to the
        selected camera and emission wavelength, from the config.

        Switching to a camera or wavelength the config has no entry for
        *clears* the calibration rather than leaving the previous one in
        place.
        """
        if "z-calibrations" not in CONFIG:
            # nothing is configured for any camera, so nothing was
            # auto-loaded: leave a manually loaded calibration alone
            return
        path = self.config_calib_path("z-calibrations")
        self.update_z_calib(path)
        if path:
            # To avoid the situation where the user runs a 3D localization
            # just because the calib file was loaded, uncheck the "Fit Z"
            # checkbox;
            self.fit_z_checkbox.setChecked(False)

    def update_z_calib(self, path: str | None) -> None:
        """Load (or clear, with ``None``) the 3D calibration from a YAML
        file."""
        if path:
            if os.path.exists(path):
                self.z_calibration = io.load_calibration(path)
                self.z_calibration_path = path
            else:
                self.update_z_calib(None)
                self.z_calib_label.setText("-- calibration path not found --")
                self.z_calib_label.setToolTip("")
                return
            self.z_calib_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            self.z_calib_label.setText(os.path.basename(path))
            self.z_calib_label.setToolTip(path)
            self.fit_z_checkbox.setEnabled(True)
            self.fit_z_checkbox.setChecked(True)
        else:
            self.z_calibration = {}
            self.z_calibration_path = None
            self.z_calib_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.z_calib_label.setText("-- no calibration loaded --")
            self.fit_z_checkbox.setChecked(False)
            self.fit_z_checkbox.setEnabled(False)

    def quality_progress(self, msg: str, index: int, result: str) -> None:
        """Update the quality progress UI elements."""
        if msg != "":
            self.window.status_bar.showMessage(msg)
        else:
            self.quality_grid_values[index].setText(result)

    def quality_progress_finished(self, msg: str) -> None:
        """Handle the completion of the quality progress."""
        self.window.status_bar.showMessage(msg)

    def check_quality(self) -> None:
        """Start the quality check worker thread."""
        self.quality_check.setVisible(False)
        for _ in self.quality_grid_labels:
            _.setVisible(True)
        for _ in self.quality_grid_values:
            _.setVisible(True)

        self.q_worker = QualityWorker(
            self.window.locs,
            self.window.info,
            self.window.movie_path,
            self.pixelsize,
        )
        self.q_worker.progressMade.connect(self.quality_progress)
        self.q_worker.finished.connect(self.quality_progress_finished)
        self.q_worker.start()

    def on_aim_undrift_changed(self, state: int) -> None:
        """Enable/disable AIM segmentation spinbox."""
        if state == 0:  # unchecked
            self.aim_segmentation.setEnabled(False)
        else:  # checked
            self.aim_segmentation.setEnabled(True)

    def on_temporal_median_changed(self, state: int) -> None:
        """Enable/disable the temporal window spinbox and refresh the
        display, which shows the filtered frames while this is on.

        Toggling the filter moves the image onto a different intensity
        scale, so the contrast has to be re-derived or the frame would
        come out solid black (filter on) or solid white (filter off).
        """
        self.temporal_median_spinbox.setEnabled(state != 0)
        self._reset_contrast_and_refresh()

    def on_gaussian_filter_changed(self, _value: float = 0.0) -> None:
        """Refresh the display, which shows the smoothed frames whenever
        sigma is non-zero."""
        self._reset_contrast_and_refresh()

    def _reset_contrast_and_refresh(self) -> None:
        """Re-derive the contrast and redraw after an identification
        filter changed, since the filtered frames live on a different
        intensity scale than the raw ones."""
        # this dialog is built before the contrast dialog
        contrast_dialog = getattr(self.window, "contrast_dialog", None)
        if contrast_dialog is not None:
            contrast_dialog.reset_to_frame()
        # the filtered frames live on a different scale, so the slider's
        # track has to be re-derived, not just widened
        update_range = getattr(
            self.window, "update_contrast_slider_range", None
        )
        if update_range is not None:
            update_range()
        self.window.on_parameters_changed()

    def on_link_params_changed(self, _state: int = 0) -> None:
        """A cross-channel link toggle changed: converge every channel to the
        current (shared) settings so it takes effect immediately, not only on
        the next channel switch."""
        self.window.propagate_linked_params()

    def on_box_changed(self) -> None:
        """Handle changes to the parameter boxes."""
        self.window.on_parameters_changed()

    def on_camera_changed(self, index: int) -> None:
        """Handle changes to the camera selection."""
        self.set_photon_scalar("gain", 1)
        self.cam_settings.setCurrentIndex(index)
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        if "Baseline" in cam_config:
            self.set_photon_scalar("baseline", cam_config["Baseline"])
        if "DefaultGain" in cam_config:
            self.set_photon_scalar("gain", cam_config["DefaultGain"])
        if "Pixelsize" in cam_config:
            self.pixelsize.setValue(cam_config["Pixelsize"])
        self.update_sensitivity()
        self.update_qe()

        # load 3D calibration
        self.update_z_calib_with_config_path()
        # load spline PSF calibration
        self.update_spline_calib_with_config_path()
        # load the per-pixel sCMOS calibration
        self.update_camera_calib_with_config_path()

    def update_qe(self) -> None:
        """Update QE. Note that QE is not used in the analysis, the
        method is kept for backward compatibility."""
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        if "Quantum Efficiency" in cam_config:
            qe = cam_config["Quantum Efficiency"]
            try:
                self.qe.setValue(qe)
            except TypeError:
                # qe is not a number
                em_combo = self.emission_combos[camera]
                wavelength = float(em_combo.currentText())
                qe = cam_config["Quantum Efficiency"][wavelength]
                self.qe.setValue(qe)

    def on_emission_changed(self) -> None:
        """Update QE due to change in emission wavelength."""
        self.update_qe()
        self.update_z_calib_with_config_path()
        self.update_spline_calib_with_config_path()
        self.update_camera_calib_with_config_path()

    def on_mng_spinbox_changed(self, value: int) -> None:
        """Handle change to the min. net gradient spinbox."""
        if value < self.mng_slider.minimum():
            self.mng_min_spinbox.setValue(value)
        if value > self.mng_slider.maximum():
            self.mng_max_spinbox.setValue(value)
        self.mng_slider.setValue(value)

    def on_mng_slider_changed(self, value: int) -> None:
        """Handle change to the min. net gradient slider."""
        self.mng_spinbox.setValue(value)
        if self._syncing_mng:
            # only showing the selected region's threshold - nothing about
            # the identification changed, so the locs must survive
            self._last_mng = value
            return
        # in split-FOV mode the slider edits the selected region's own
        # threshold; do this before ``_last_mng`` moves, so any region that
        # has no threshold yet inherits the previous value rather than the
        # one being typed, and before the preview, so it uses the new one
        self._store_mng_for_regions(value)
        self._last_mng = value
        if self.preview_checkbox.isChecked():
            self.window.on_parameters_changed()

    def _store_mng_for_regions(self, value: int) -> None:
        """Write the slider's value into the split-FOV region it belongs to.

        With a region selected the value is that region's own threshold;
        with none selected it is applied to every region, so the slider
        still works as a global control. A no-op outside split-FOV mode.
        """
        window = self.window
        mngs = window.region_mngs()
        if not mngs:
            return
        index = window.view.selected_roi
        if index is None or index >= len(mngs):
            window.view.roi_mngs = [int(value)] * len(mngs)
        else:
            window.view.roi_mngs[index] = int(value)
        if self.roi_dialog is not None:
            self.roi_dialog.update_table()
        if not self.preview_checkbox.isChecked():
            # the region labels carry the value; with the preview on the
            # redraw comes from on_parameters_changed instead
            window.draw_frame()

    def sync_mng_to_selected_region(self) -> None:
        """Show the selected split-FOV region's own threshold on the min.
        net gradient slider/spinbox, so selecting a region tunes that
        region. A no-op outside split-FOV mode or with nothing selected."""
        if self._syncing_mng:
            return
        window = self.window
        mngs = window.region_mngs()
        index = window.view.selected_roi
        if not mngs or index is None or index >= len(mngs):
            return
        value = int(mngs[index])
        if value == self.mng_spinbox.value():
            return
        self._syncing_mng = True
        try:
            # via the spinbox: its handler widens the slider range when the
            # region's value falls outside it
            self.mng_spinbox.setValue(value)
        finally:
            self._syncing_mng = False

    def on_mng_min_changed(self, value: int) -> None:
        self.mng_slider.setMinimum(value)

    def on_mng_max_changed(self, value: int) -> None:
        self.mng_slider.setMaximum(value)

    def on_preview_changed(self) -> None:
        """Update the frame with/without indentification preview."""
        self.window.draw_frame()

    def on_link_colors_changed(self) -> None:
        """Redraw with/without cross-channel link color-coding of the
        identification boxes."""
        self.window.draw_frame()

    def on_identify_mode_changed(self) -> None:
        """Switch between per-channel and channel-sum identification.

        The two modes search different images, so any identification made in
        the other mode is stale - dropping the channel sum here also takes the
        summed view out of the display and the preview. Like the filters, the
        sum lives on a different intensity scale (photons, over all channels),
        so the contrast has to be re-derived with it.

        Selecting the sum builds the summed view straight away wherever the
        channels can be registered without identifying them first, so that the
        display and the identification preview show the image the
        identification will actually search rather than the raw movie."""
        self.window.drop_channel_sum()
        if self.identify_mode_combo.currentText() == IDENTIFY_MODE_SUM:
            self.window.ensure_channel_sum(notify=True)
        self._reset_contrast_and_refresh()

    def joint_fit_selected(self) -> bool:
        """Whether the 'Fit' setting asks for the joint multichannel fit.

        True for single-channel data too: the combo is hidden then, and the
        joint path degrades to the ordinary single-channel fit. Guarded so
        the calls made while the dialog is still being built are a no-op."""
        combo = getattr(self, "fit_mode_combo", None)
        if combo is None:
            return True
        return combo.currentText() == FIT_MODE_JOINT

    def on_fit_mode_changed(self) -> None:
        """Switch between the joint fit and one fit per channel.

        Nothing is linked or shared in the independent mode, so the options
        that only mean something to the joint fit (photon linking) go away
        with it, and the preview stops color-coding the cross-channel links.
        """
        joint = self.joint_fit_selected()
        self._update_link_photons_visibility()
        self._update_gauss_link_photons_visibility()
        checkbox = self.link_colors_checkbox
        if not joint and checkbox.isChecked():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        checkbox.setEnabled(joint)
        # entering the separate mode, the selected split-FOV region's own
        # settings take over the fit widgets
        self.window.sync_region_fit_params()
        self.window.draw_frame()

    def on_gpu_fitting_changed(self) -> None:
        """Handle changes to the GPU fitting option."""
        self._apply_convergence_defaults()
        # this dialog is created before the window's movie attribute
        if getattr(self.window, "movie", None) is not None:
            self.window.draw_frame()

    def get_camera(self, info: dict) -> tuple[str, list[str]]:
        """Get the camera name from the provided camera info."""
        if "Camera" in info and "Cameras" in CONFIG:
            cameras = [
                self.camera.itemText(_) for _ in range(self.camera.count())
            ]
            camera = info["Camera"]
            if camera in cameras:
                return camera, cameras
        return None, None

    def set_gain(self, camera: str, mm_info: dict, cam_config: dict) -> None:
        """Set EM gain if the relevant information is available in the
        config and metadata.

        The properties named in the config need not be present in the
        metadata (e.g. a movie recorded with the device missing from the
        MicroManager hardware config), in which case the gain is left
        untouched instead of raising."""
        if "Gain Property Name" in cam_config:
            gain_property_name = cam_config["Gain Property Name"]
            gain_property = camera + "-" + gain_property_name
            if gain_property not in mm_info:
                return
            gain = mm_info[gain_property]
            if "EM Switch Property" in cam_config:
                switch_property_name = cam_config["EM Switch Property"]["Name"]
                switch_property = camera + "-" + switch_property_name
                if switch_property not in mm_info:
                    return
                switch_property_value = mm_info[switch_property]
                if (
                    switch_property_value
                    == cam_config["EM Switch Property"][True]
                ):
                    self.gain.setValue(int(gain))
                else:
                    self.gain.setValue(1)

    def set_sensitivity(
        self, camera: str, mm_info: dict, cam_config: dict
    ) -> None:
        if "Sensitivity Categories" in cam_config:
            cam_combos = self.cam_combos[camera]
            categories = cam_config["Sensitivity Categories"]
            for i, category in enumerate(categories):
                property_name = camera + "-" + category
                if property_name in mm_info:
                    e_setting = mm_info[camera + "-" + category]
                    cam_combo = cam_combos[i]
                    for index in range(cam_combo.count()):
                        if cam_combo.itemText(index) == e_setting:
                            cam_combo.setCurrentIndex(index)
                            break

    def set_wavelength(
        self, camera: str, mm_info: dict, cam_config: dict
    ) -> None:
        """Set the emission wavelength from the channel device property
        named in the config.

        The property need not be present in the metadata (e.g. a movie
        recorded with the filter turret missing from the MicroManager
        hardware config), in which case the wavelength is left untouched
        instead of raising."""
        if "Channel Device" in cam_config:
            channel_device_name = cam_config["Channel Device"]["Name"]
            if channel_device_name not in mm_info:
                return
            channel = mm_info[channel_device_name]
            channels = cam_config["Channel Device"]["Emission Wavelengths"]
            if channel in channels:
                wavelength = str(channels[channel])
                em_combo = self.emission_combos[camera]
                for index in range(em_combo.count()):
                    if em_combo.itemText(index) == wavelength:
                        em_combo.setCurrentIndex(index)
                        break

    def set_camera_parameters(self, info: dict) -> None:
        """Set the camera parameters based on the provided camera
        info."""
        camera, cameras = self.get_camera(info)
        if camera is None:
            return

        index = cameras.index(camera)
        self.camera.setCurrentIndex(index)
        if "Micro-Manager Metadata" in info:
            mm_info = info["Micro-Manager Metadata"]
            cam_config = CONFIG["Cameras"][camera]
            self.set_gain(camera, mm_info, cam_config)
            self.set_sensitivity(camera, mm_info, cam_config)
            self.set_wavelength(camera, mm_info, cam_config)

    def update_sensitivity(self) -> None:
        """Update the sensitivity settings for the current camera."""
        camera = self.camera.currentText()
        cam_config = CONFIG["Cameras"][camera]
        sensitivity = cam_config["Sensitivity"]
        if "Sensitivity" in cam_config:
            if not isinstance(sensitivity, dict):
                self.set_photon_scalar("sensitivity", sensitivity)
            else:
                categories = cam_config["Sensitivity Categories"]
                for i, category in enumerate(categories):
                    cat_combo = self.cam_combos[camera][i]
                    sensitivity = sensitivity[cat_combo.currentText()]
                self.set_photon_scalar("sensitivity", sensitivity)


class ContrastDialog(lib.Dialog):
    """Choose display contrast."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Contrast")
        self.resize(200, 0)
        self.setModal(False)
        grid = QtWidgets.QGridLayout(self)
        black_label = QtWidgets.QLabel("Black:")
        black_label.setToolTip("Min. intensity rendered")
        grid.addWidget(black_label, 0, 0)
        self.black_spinbox = lib.LogDoubleSpinBox()
        self.black_spinbox.setDecimals(0)
        self.black_spinbox.setKeyboardTracking(False)
        # 0 must be reachable: a temporal median filtered frame has its
        # background subtracted and clipped at zero, so its black point
        # genuinely is 0. (White stays >= 1, it is a divisor in
        # ``_draw_frame``.)
        self.black_spinbox.setRange(0, 999999)
        self.black_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.black_spinbox, 0, 1)
        white_label = QtWidgets.QLabel("White:")
        white_label.setToolTip("Max. intensity rendered")
        grid.addWidget(white_label, 1, 0)
        self.white_spinbox = lib.LogDoubleSpinBox()
        self.white_spinbox.setDecimals(0)
        self.white_spinbox.setKeyboardTracking(False)
        self.white_spinbox.setRange(1, 999999)
        self.white_spinbox.valueChanged.connect(self.on_contrast_changed)
        grid.addWidget(self.white_spinbox, 1, 1)
        self.auto_checkbox = QtWidgets.QCheckBox("Auto")
        self.auto_checkbox.setToolTip(
            "Set the range automatically for each frame?"
        )
        self.auto_checkbox.setTristate(False)
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.on_auto_changed)
        grid.addWidget(self.auto_checkbox, 2, 0, 1, 2)
        self.silent_contrast_change = False
        self.manual_contrast_change = False

    def change_contrast_silently(self, black: int, white: int) -> None:
        """Change the contrast values without emitting signals."""
        self.silent_contrast_change = True
        self.black_spinbox.setValue(black)
        self.white_spinbox.setValue(white)
        self.silent_contrast_change = False
        self.sync_slider()

    def sync_slider(self) -> None:
        """Mirror the spinboxes onto the window's contrast slider.

        The spinboxes stay the single source of truth; the slider is a
        second view onto them.
        """
        sync = getattr(self.window, "sync_contrast_slider", None)
        if sync is not None:
            sync()

    def set_contrast_from_slider(self, black: float, white: float) -> None:
        """Apply a contrast dragged on the window's slider.

        Same outcome as typing into the spinboxes (manual contrast, Auto
        off), but both values are set at once so that a drag redraws the
        frame once per step instead of twice.
        """
        self.silent_contrast_change = True
        try:
            self.black_spinbox.setValue(black)
            self.white_spinbox.setValue(white)
        finally:
            self.silent_contrast_change = False
        self.manual_contrast_change = True
        try:
            self.auto_checkbox.setChecked(False)
        finally:
            self.manual_contrast_change = False
        self.window.draw_frame()

    def reset_to_frame(self) -> None:
        """Re-derive the range from the frame currently on screen.

        Called when the intensity scale changes underneath the display:
        switching the temporal median filter on subtracts the background
        and clips at zero, and smoothing lowers the peaks, so a contrast
        set for the raw camera counts would render the filtered frame as
        an empty black or a nearly flat image.
        """
        if getattr(self.window, "movie", None) is None:
            return
        frame = self.window.identification_movie()[
            getattr(self.window, "curr_frame_number", 0)
        ]
        self.change_contrast_silently(frame.min(), frame.max())

    def to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Map ``frame`` onto the 0-255 display range.

        Auto spans the frame's own min-max; otherwise the black-white range
        set in this dialog. Both use the same mapping, so switching Auto off
        (which leaves the spinboxes at the values ``set_frame`` derived from
        the current frame) does not change how the frame looks.
        """
        frame = frame.astype("float32")
        if self.auto_checkbox.isChecked():
            black = float(frame.min())
            white = float(frame.max())
        else:
            black = float(self.black_spinbox.value())
            white = float(self.white_spinbox.value())
        frame -= black
        frame /= max(white - black, 1e-12)
        frame *= 255.0
        return np.clip(frame, 0, 255).astype("uint8")

    def on_contrast_changed(self, value: int) -> None:
        if not self.silent_contrast_change:
            # editing a value implies manual contrast; flagged so that
            # unchecking Auto here does not re-derive (and thereby discard)
            # the value that was just typed
            self.manual_contrast_change = True
            try:
                self.auto_checkbox.setChecked(False)
            finally:
                self.manual_contrast_change = False
            self.sync_slider()
            self.window.draw_frame()

    def on_auto_changed(self, _state: int) -> None:
        if not self.manual_contrast_change:
            # the displayed movie, which is the filtered view whenever an
            # identification filter is on - not the raw camera counts.
            # Also on unchecking: seeding the range from the frame on
            # screen is what makes turning Auto off freeze the current
            # rendering instead of changing it.
            self.reset_to_frame()
            self.window.draw_frame()


class LocColumnSelectionDialog(lib.Dialog):
    """Dialog for selecting which columns to save in the localization
    file."""

    # Target height of one on-screen column of checkboxes
    _MAX_ROWS_PER_COLUMN = 20

    def __init__(self, parent: QtWidgets.QMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select columns to save")
        self.setModal(True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # one titled group box per LOCALIZATION_COLUMNS key, packed into as
        # many side-by-side columns as it takes to stay under the row target
        self.column_checkboxes = {}
        screen_column = None
        rows_used = 0
        for key, columns in localize.LOCALIZATION_COLUMNS.items():
            if (
                screen_column is None
                or rows_used + len(columns) > self._MAX_ROWS_PER_COLUMN
            ):
                screen_column = QtWidgets.QVBoxLayout()
                screen_column.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
                layout.addLayout(screen_column)
                rows_used = 0
            group = QtWidgets.QGroupBox(key)
            group_layout = QtWidgets.QVBoxLayout(group)
            for column in columns:
                checkbox = QtWidgets.QCheckBox(column)
                checkbox.setChecked(True)
                if column in lib.REQUIRED_COLUMNS:
                    checkbox.setDisabled(True)
                self.column_checkboxes[column] = checkbox
                group_layout.addWidget(checkbox)
            screen_column.addWidget(group)
            rows_used += len(columns)

        self.load_user_settings()

    def load_user_settings(self) -> None:
        """Reads the columns to save from user settings."""
        settings = io.load_user_settings()
        columns = settings["Localize"].get("Columns to save", None)
        if columns is None:
            columns = {}
            for subcols in localize.LOCALIZATION_COLUMNS.values():
                for col in subcols:
                    columns[col] = True
        for column, checkbox in self.column_checkboxes.items():
            is_checked = columns.get(column, True)
            checkbox.setChecked(is_checked)


class Window(QtWidgets.QMainWindow):
    """The main window.

    ...

    Attributes
    ----------
    camera_info : dict
        Camera information, such as gain, sensitivity, etc.
    contrast_dialog : ContrastDialog
        The dialog for adjusting display contrast.
    channels : list of Channel
        Per-channel state for multichannel data. The active channel
        (``current_channel``) is mirrored into the flat attributes
        (``movie``, ``info``, ``movie_path`` ...). A single movie is held
        as a one-element list.
    current_channel : int
        Index of the active channel within ``channels``.
    extra_info : list of dict
        Movie metadata in a format of a dictionary, with Localize info.
    identifications : pd.DataFrame
        Identified spots - frame, position, net gradient.
    last_identification_info : dict
        A dictionary of analysis parameters used for the last operation.
        Used to save user settings when closing the application.
    locs : pd.DataFrame
        Resulting localizations.
    info : list of dict
        Movie metadata in a format of a dictionary, without Localize
        info, see self.extra_info.
    movie : np.memmap or None
        Loaded movie (frame, y, x) of the active channel.
    movie_path : str or list
        Path to the active channel's movie file (empty list when nothing
        is loaded).
    parameters_dialog : ParametersDialog
        The dialog for adjusting parameters.
    ready_for_fit : bool
        If True, spots were identified and are ready to be fitted.
    scene : Scene
        The scene for displaying the image.
    status_bar : QtWidgets.QStatusBar
        Status bar displayed in the bottom of the window.
    view : View
        The main view for displaying the image.
    """

    DOCS_URL = "https://picassosr.readthedocs.io/en/latest/localize.html"

    def __init__(self) -> None:
        super().__init__()
        # Init GUI
        self.setWindowTitle(f"Picasso v{__version__}: Localize")

        this_directory = os.path.dirname(os.path.realpath(__file__))
        icon_path = os.path.join(this_directory, "icons", "localize.ico")
        icon = QtGui.QIcon(icon_path)
        self.setWindowIcon(icon)
        self.resize(768, 768)
        self.parameters_dialog = ParametersDialog(self)
        self.contrast_dialog = ContrastDialog(self)
        self.columns_dialog = LocColumnSelectionDialog(self)
        self.metadata_dialog = lib.MetadataDialog(self)
        self.user_settings_dialog = lib.UserSettingsDialog(self)
        self.init_menu_bar()
        self.view = View(self)
        self.scene = Scene(self)
        self.view.setScene(self.scene)
        # Slider below the movie for quickly navigating between frames.
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMaximumHeight(15)
        self.frame_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: #b0b0b0;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 10px;
                height: 12px;
                margin: -5px 0;
                border-radius: 3px;
                background: #5a5a5a;
            }
            """
        )
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        # Two-handle slider below it for the display contrast, mirroring
        # the black and white points of the contrast dialog.
        self.contrast_slider = lib.RangeSlider()
        self.contrast_slider.setEnabled(False)
        self.contrast_slider.setMaximumHeight(15)
        self.contrast_slider.setValueLabels("Black", "White")
        # the spinboxes the slider writes into round to integers, so the
        # handles must not be able to land on the same value (white is a
        # divisor in ``ContrastDialog.to_uint8``)
        self.contrast_slider.setMinimumGap(1)
        self.contrast_slider.valuesChanged.connect(
            self.on_contrast_slider_changed
        )
        # Channel selector (hidden unless several channels are loaded).
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.setVisible(False)
        self.channel_combo.currentIndexChanged.connect(
            self.on_channel_combo_changed
        )
        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.channel_combo)
        central_layout.addWidget(self.view)
        central_layout.addWidget(self.frame_slider)
        central_layout.addWidget(self.contrast_slider)
        self.setCentralWidget(central_widget)
        self.status_bar = self.statusBar()
        self.status_bar_frame_indicator = QtWidgets.QLabel()
        self.status_bar.addPermanentWidget(self.status_bar_frame_indicator)

        # re-entrancy guard for draw_frame (see on_scroll)
        self._drawing_frame = False
        # Holds the curr movie as a numpy memmap in the format
        # (frame, y, x)
        self.movie = None
        # Cached identification-filtered views of ``self.movie``, see
        # ``identification_movie``. Never used for fitting.
        self._temporal_movie = None
        self._gaussian_movie = None
        # The same filter stacks for the channels that are *not* on screen,
        # built for the link-color preview only (see
        # ``channel_identification_movie``): {channel index: [temporal,
        # gaussian]}.
        self._channel_filter_cache = {}
        # This frame's detections per channel while the identification
        # preview draws them, so the link colors can pair spots that have
        # not been identified yet (see ``_preview_link_identifications``).
        # None whenever the preview is not drawing.
        self._preview_link_ids = None
        # Cached preview detections of the off-screen channels, keyed by
        # channel and by everything they were identified with.
        self._preview_ids_cache = {}
        # (linked, total) boxes the last link-color draw put on screen, for
        # the preview's status message.
        self._link_box_counts = None
        # Dictionary of analysis parameters used for the last operation
        self.last_identification_info = None
        # Dataframe of identifcations with fields frame, x and y
        self.identifications = None
        self.ready_for_fit = False
        self.locs = None
        # Snapshot of ``self.locs`` used to draw the FitMarker overlay.
        # Kept separate so drift correction can mutate ``self.locs``
        # without moving the markers shown on screen.
        self.locs_display = None
        self.movie_path = []
        # None analyzes all frames; otherwise a list of 0-indexed
        # inclusive [lo, hi] segments to restrict identification to
        self.frame_range = None
        self.info = []
        self.extra_info = []
        # Per-bead records of the last spline PSF calibration built in this
        # session (one per channel), shown by the bead inspector.
        self.bead_diagnostics = []
        self.bead_calibration_path = None
        self._active_worker = None
        # Lateral corrections the last fit applied, as descriptions, for the
        # saved metadata (see _record_applied_lateral).
        self._applied_lateral = []
        # Affine-transform (astigmatism) calibration dialog and its worker;
        # both created lazily when the calibration is first opened/run.
        self._affine_dialog = None
        self.affine_calibration_worker = None
        # Bead pairing of the last affine calibration, drawn over whichever
        # of the two bead images is displayed (see draw_affine_pairing).
        self.affine_pairing = None
        # Bookkeeping for a multichannel "Identify" (Ctrl+I) batch that runs
        # identification on every channel in turn; None when not running.
        self._multi_identify = None
        # ... and for a "Fit" batch that fits every channel (or split-FOV
        # region) on its own, see ``_fit_independent``.
        self._multi_fit = None
        # Split-FOV region currently being fitted on its own: its label is
        # appended to every file that fit writes (see channel_output_base).
        self._region_suffix = ""
        # Which split-FOV region the fit settings now on the dialog belong to
        # (see ``sync_region_fit_params``); None when none is selected.
        self._region_params_owner = None
        # Channel-sum identification (IDENTIFY_MODE_SUM), see
        # ``identify_channel_sum``. ``sum_identifications`` marks the current
        # detections as coming from the summed channels, which is what tells
        # the joint fit not to link them across channels again;
        # ``_sum_movie`` is the summed view the display and the preview use.
        self.sum_identifications = None
        self.sum_transforms = None
        self.sum_transform_source = ""
        self._sum_movie = None
        self._sum_identify = None
        # the summed view is built as soon as the mode is selected, so that
        # the display and the preview show it (see ``ensure_channel_sum``);
        # this remembers a failed attempt, so that a registration that cannot
        # succeed is not retried on every redraw
        self._sum_registration_failed = False
        # Multichannel state. ``self.channels[self.current_channel]`` is
        # the active channel, mirrored into the flat attributes above.
        # Single movies are stored as a one-element list.
        self.channels = []
        self.current_channel = 0
        # Guards the parameter/contrast handlers while restoring a
        # channel's dialog values, so they don't wipe the channel's
        # locs/markers.
        self._switching_channel = False

        # Background movie-loading worker state (see open_channels_from_files).
        self._load_thread = None
        self._load_worker = None
        self._load_progress = None
        self._load_filename = ""
        # Canceled loads, winding down on their own (see
        # ``_on_load_canceled``); kept referenced until their thread ends.
        self._detached_loads = []
        self._load_t0 = 0.0
        self._load_multi_file = False
        # Make sure a running loader thread is stopped before the
        # QApplication is destroyed; destroying a live QThread aborts the
        # process. ``aboutToQuit`` fires on every quit path (including
        # sys.exit through the event loop), unlike ``closeEvent``.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_movie_load)

        self.load_user_settings()

    def load_user_settings(self) -> None:
        """Load user settings based on the last-used parameters."""
        settings = io.load_user_settings()
        pwd = []
        box_size = []
        gradient = []
        try:
            pwd = settings["Localize"]["PWD"]
            box_size = settings["Localize"]["box_size"]
            gradient = settings["Localize"]["gradient"]
        except Exception as e:
            print(e)
            pass
        if len(pwd) == 0:
            pwd = []
        if type(box_size) is int:
            self.parameters_dialog.box_spinbox.setValue(box_size)
        if type(gradient) is int:
            self.parameters_dialog.mng_slider.setValue(gradient)

        temporal_median = settings["Localize"].get("temporal_median", None)
        if type(temporal_median) is int and temporal_median > 0:
            self.parameters_dialog.temporal_median_spinbox.setValue(
                temporal_median
            )
        self.parameters_dialog.temporal_median_checkbox.setChecked(
            bool(settings["Localize"].get("temporal_median_on", False))
        )

        gaussian_sigma = settings["Localize"].get(
            "gaussian_filter_sigma", None
        )
        if isinstance(gaussian_sigma, (int, float)) and gaussian_sigma >= 0:
            self.parameters_dialog.gaussian_filter_spinbox.setValue(
                float(gaussian_sigma)
            )

        # Restore the last-used fitting model and optimizer. The model must
        # be set first, since it repopulates the optimizer combobox.
        fit_model = settings["Localize"].get("fit_model", None)
        if fit_model is not None:
            index = self.parameters_dialog.fit_model.findText(fit_model)
            if index >= 0:
                self.parameters_dialog.fit_model.setCurrentIndex(index)
        fit_optimizer = settings["Localize"].get("fit_optimizer", None)
        if fit_optimizer is not None:
            index = self.parameters_dialog.fit_optimizer.findText(
                fit_optimizer
            )
            if index >= 0:
                self.parameters_dialog.fit_optimizer.setCurrentIndex(index)
        fit_mode = settings["Localize"].get("fit_mode", None)
        if fit_mode is not None:
            index = self.parameters_dialog.fit_mode_combo.findText(fit_mode)
            if index >= 0:
                self.parameters_dialog.fit_mode_combo.setCurrentIndex(index)

        self.pwd = pwd

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Close the application, save user settings."""
        settings = io.load_user_settings()
        if self.movie_path != []:
            settings["Localize"]["PWD"] = os.path.dirname(self.movie_path)
            settings["Localize"][
                "box_size"
            ] = self.parameters_dialog.box_spinbox.value()
            settings["Localize"][
                "gradient"
            ] = self.parameters_dialog.mng_slider.value()
        settings["Localize"][
            "temporal_median"
        ] = self.parameters_dialog.temporal_median_spinbox.value()
        settings["Localize"][
            "temporal_median_on"
        ] = self.parameters_dialog.temporal_median_checkbox.isChecked()
        settings["Localize"][
            "gaussian_filter_sigma"
        ] = self.parameters_dialog.gaussian_filter_spinbox.value()
        settings["Localize"][
            "fit_model"
        ] = self.parameters_dialog.fit_model.currentText()
        settings["Localize"][
            "fit_optimizer"
        ] = self.parameters_dialog.fit_optimizer.currentText()
        settings["Localize"][
            "fit_mode"
        ] = self.parameters_dialog.fit_mode_combo.currentText()
        settings["Localize"]["Columns to save"] = {
            column: checkbox.isChecked()
            for column, checkbox in (
                self.columns_dialog.column_checkboxes.items()
            )
        }
        io.save_user_settings(settings)
        # Settings are safely on disk before the loader threads are dealt
        # with, so a slow file read can never cost the user their
        # settings, however the process ends below.
        self._stop_movie_load()
        self._stop_active_worker()
        if self._loads_still_running() or self._worker_still_running():
            # A loader is stuck in an io call that cannot be interrupted
            # (a hung network share, say), or an identification/fit did
            # not come back in time. Waiting for it would freeze the
            # quit, and letting Qt destroy a live QThread aborts the
            # process with a crash report, so leave right here instead.
            # Nothing is being written, so there is nothing to lose.
            os._exit(0)
        QtWidgets.QApplication.instance().closeAllWindows()

    def init_menu_bar(self) -> None:
        """Initialize the menu bar."""
        menu_bar = self.menuBar()

        """ File """
        file_menu = menu_bar.addMenu("File")
        open_action = file_menu.addAction("Open movie...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        open_multichannel_action = file_menu.addAction(
            "Open one multichannel movie..."
        )
        open_multichannel_action.setShortcut("Ctrl+Shift+O")
        open_multichannel_action.triggered.connect(
            self.open_multichannel_file_dialog
        )
        file_menu.addAction(open_multichannel_action)
        open_channels_action = file_menu.addAction(
            "Open channels from several movies..."
        )
        open_channels_action.triggered.connect(
            self.open_channels_from_files_dialog
        )
        file_menu.addAction(open_channels_action)
        open_mm_folder_action = file_menu.addAction(
            "Open MicroManager image folder..."
        )
        open_mm_folder_action.triggered.connect(self.open_mm_folder_dialog)
        file_menu.addAction(open_mm_folder_action)
        open_concat_action = file_menu.addAction("Concatenate movies...")
        open_concat_action.triggered.connect(self.open_concatenated_dialog)
        file_menu.addAction(open_concat_action)
        save_identifications_action = file_menu.addAction(
            "Save identifications..."
        )
        save_identifications_action.triggered.connect(
            self.save_identifications_dialog
        )
        load_identifications_action = file_menu.addAction(
            "Load identifications..."
        )
        load_identifications_action.triggered.connect(
            self.open_identifications
        )
        load_picks_action = file_menu.addAction(
            "Load picks as identifications..."
        )
        load_picks_action.triggered.connect(self.open_picks)
        load_locs_action = file_menu.addAction(
            "Load locs as identifications..."
        )
        load_locs_action.triggered.connect(self.open_locs)
        save_action = file_menu.addAction("Save localizations...")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_locs_dialog)
        file_menu.addAction(save_action)
        save_spots_action = file_menu.addAction("Save spots...")
        save_spots_action.setShortcut("Ctrl+Shift+S")
        save_spots_action.triggered.connect(self.save_spots_dialog)
        file_menu.addAction(save_spots_action)
        select_columns_action = file_menu.addAction(
            "Select columns to save..."
        )
        select_columns_action.triggered.connect(self.columns_dialog.show)
        file_menu.addAction(select_columns_action)
        file_menu.addSeparator()
        export_current_action = file_menu.addAction("Export current view...")
        export_current_action.setShortcut("Ctrl+E")
        export_current_action.triggered.connect(self.export_current)
        metadata_action = file_menu.addAction("Show metadata...")
        metadata_action.setShortcut("Ctrl+M")
        metadata_action.triggered.connect(self.show_metadata)
        file_menu.addAction(metadata_action)

        file_menu.addSeparator()
        sounds_menu = file_menu.addMenu("Sound notifications")
        sounds_actiongroup = QtGui.QActionGroup(file_menu)
        default_sound_path = lib.get_sound_notification_path()  # last used
        default_sound_name = os.path.basename(str(default_sound_path))
        for sound in lib.get_available_sound_notifications():
            sound_name = os.path.splitext(str(sound))[0].replace("_", " ")
            action = sounds_actiongroup.addAction(
                QtGui.QAction(sound_name, sounds_menu, checkable=True)
            )
            action.setObjectName(sound)  # store full name
            if default_sound_name == sound:
                action.setChecked(True)
            sounds_menu.addAction(action)
        sounds_actiongroup.triggered.connect(lib.set_sound_notification)
        sounds_menu.addSeparator()
        open_sounds_action = sounds_menu.addAction(
            "Open notification sounds folder..."
        )
        open_sounds_action.triggered.connect(
            lib.open_sound_notifications_folder
        )
        picasso_settings_action = file_menu.addAction("Picasso settings...")
        picasso_settings_action.triggered.connect(
            self.user_settings_dialog.show
        )
        open_config_action = file_menu.addAction(
            "Open camera config file location"
        )
        open_config_action.triggered.connect(self.open_config_location)
        help_action = file_menu.addAction("Help")
        help_action.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.DOCS_URL))
        )

        """ View """
        view_menu = menu_bar.addMenu("View")
        previous_frame_action = view_menu.addAction("Previous frame")
        previous_frame_action.setShortcut("Left")
        previous_frame_action.triggered.connect(lambda: self.previous_frame(1))
        view_menu.addAction(previous_frame_action)
        next_frame_action = view_menu.addAction("Next frame")
        next_frame_action.setShortcut("Right")
        next_frame_action.triggered.connect(lambda: self.next_frame(1))
        view_menu.addAction(next_frame_action)
        # Jump multiple frames at once using modifier keys with the arrows.
        # Shift -> 10, Ctrl/Cmd -> 100
        for step, modifier in (
            (10, "Shift"),
            (100, "Ctrl"),
        ):
            jump_back_action = view_menu.addAction(
                "Previous {} frames".format(step)
            )
            jump_back_action.setShortcut("{}+Left".format(modifier))
            jump_back_action.triggered.connect(
                lambda *_, s=step: self.previous_frame(s)
            )
            view_menu.addAction(jump_back_action)
            jump_forward_action = view_menu.addAction(
                "Next {} frames".format(step)
            )
            jump_forward_action.setShortcut("{}+Right".format(modifier))
            jump_forward_action.triggered.connect(
                lambda *_, s=step: self.next_frame(s)
            )
            view_menu.addAction(jump_forward_action)
        view_menu.addSeparator()
        first_frame_action = view_menu.addAction("First frame")
        first_frame_action.setShortcut("Home")
        first_frame_action.triggered.connect(self.first_frame)
        view_menu.addAction(first_frame_action)
        last_frame_action = view_menu.addAction("Last frame")
        last_frame_action.setShortcut("End")
        last_frame_action.triggered.connect(self.last_frame)
        view_menu.addAction(last_frame_action)
        go_to_frame_action = view_menu.addAction("Go to frame...")
        go_to_frame_action.setShortcut("Ctrl+G")
        go_to_frame_action.triggered.connect(self.to_frame)
        view_menu.addAction(go_to_frame_action)
        view_menu.addSeparator()
        # Channel navigation (multichannel only). Disabled - and therefore
        # not stealing the Up/Down keys from the split-FOV region nudging in
        # View.keyPressEvent - until several channels are loaded (see
        # _populate_channel_combo).
        self.previous_channel_action = view_menu.addAction("Previous channel")
        self.previous_channel_action.setShortcut("Up")
        self.previous_channel_action.triggered.connect(self.previous_channel)
        self.previous_channel_action.setEnabled(False)
        view_menu.addAction(self.previous_channel_action)
        self.next_channel_action = view_menu.addAction("Next channel")
        self.next_channel_action.setShortcut("Down")
        self.next_channel_action.triggered.connect(self.next_channel)
        self.next_channel_action.setEnabled(False)
        view_menu.addAction(self.next_channel_action)
        view_menu.addSeparator()
        zoom_in_action = view_menu.addAction("Zoom in")
        zoom_in_action.setShortcuts(["Ctrl++", "Ctrl+="])
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = view_menu.addAction("Zoom out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        fit_in_view_action = view_menu.addAction("Fit image to window")
        fit_in_view_action.setShortcut("Ctrl+W")
        fit_in_view_action.triggered.connect(self.fit_in_view)
        view_menu.addAction(fit_in_view_action)
        view_menu.addSeparator()
        constract_action = view_menu.addAction("Contrast...")
        constract_action.setShortcut("Ctrl+C")
        constract_action.triggered.connect(self.contrast_dialog.show)
        view_menu.addAction(constract_action)
        self.scalebar_action = view_menu.addAction("Show scale bar")
        self.scalebar_action.setCheckable(True)
        self.scalebar_action.setChecked(False)
        self.scalebar_action.triggered.connect(self.draw_frame)
        view_menu.addAction(self.scalebar_action)

        """ Analyze """
        analyze_menu = menu_bar.addMenu("Analyze")
        parameters_action = analyze_menu.addAction("Parameters...")
        parameters_action.setShortcut("Ctrl+P")
        parameters_action.triggered.connect(self.parameters_dialog.show)
        analyze_menu.addAction(parameters_action)
        analyze_menu.addSeparator()
        identify_action = analyze_menu.addAction("Identify")
        identify_action.setShortcut("Ctrl+I")
        identify_action.triggered.connect(self.identify)
        analyze_menu.addAction(identify_action)
        fit_action = analyze_menu.addAction("Fit")
        fit_action.setShortcut("Ctrl+F")
        fit_action.triggered.connect(self.fit)
        analyze_menu.addAction(fit_action)
        localize_action = analyze_menu.addAction("Localize (Identify && Fit)")
        localize_action.setShortcut("Ctrl+L")
        localize_action.triggered.connect(self.localize)
        analyze_menu.addAction(localize_action)
        analyze_menu.addSeparator()
        self.abort_action = analyze_menu.addAction("Abort")
        self.abort_action.setShortcut("Ctrl+.")
        self.abort_action.triggered.connect(self.abort)
        self.abort_action.setEnabled(False)
        analyze_menu.addAction(self.abort_action)

        """ Calibration """
        threed_menu = menu_bar.addMenu("Calibration")

        calibrate_z_action = threed_menu.addAction(
            "Calibrate astigmatism (Gaussian)"
        )
        calibrate_z_action.triggered.connect(self.calibrate_z)

        calibrate_affine_action = threed_menu.addAction(
            "Calibrate lateral transform (astigmatism / chromatic)..."
        )
        calibrate_affine_action.setToolTip(
            "Fit a lateral correction from two bead images and append\n"
            "it to any calibration (Gaussian, spline, or a new\n"
            "standalone lateral calibration). Several corrections stack\n"
            "and are applied one after another."
        )
        calibrate_affine_action.triggered.connect(self.calibrate_affine)

        calibrate_spline_action = threed_menu.addAction("Calibrate spline PSF")
        calibrate_spline_action.triggered.connect(self.calibrate_spline)

        register_menu = threed_menu.addMenu("Register channels (2D)")
        register_menu.setToolTip(
            "Measure where each loaded channel sits relative to the first,\n"
            "and save it as a standalone registration for the multichannel\n"
            "2D spherical Gaussian fit."
        )
        register_beads_action = register_menu.addAction("From bead data...")
        register_beads_action.setToolTip(
            "Register from images of fiducial beads, which appear in every\n"
            "channel at once. Most accurate, but needs a bead acquisition:\n"
            "load the bead movies as the channels first."
        )
        register_beads_action.triggered.connect(
            self.register_channels_from_beads
        )
        register_signal_action = register_menu.addAction(
            "From current signal..."
        )
        register_signal_action.setToolTip(
            "Register from the loaded movies themselves: the same molecule\n"
            "blinks in every channel in the same frame, so the detections\n"
            "can be paired frame by frame. Uses the current identification\n"
            "settings. Refines the loaded registration when there is one."
        )
        register_signal_action.triggered.connect(
            self.register_channels_from_signal
        )

        threed_menu.addSeparator()
        camera_calib_action = threed_menu.addAction(
            "Characterize sCMOS camera (dark movie)"
        )
        camera_calib_action.setToolTip(
            "Measure the per-pixel offset and readout variance from a dark\n"
            "movie, and optionally the per-pixel gain from a series of\n"
            "movies at different illumination levels."
        )
        camera_calib_action.triggered.connect(self.calibrate_camera)

        camera_check_action = threed_menu.addAction(
            "Check sCMOS calibration (fresh dark movie)"
        )
        camera_check_action.setToolTip(
            "Test the loaded calibration against a short fresh dark movie.\n"
            "Sensor temperature and readout mode change the maps."
        )
        camera_check_action.triggered.connect(self.check_camera_calibration)

        self.inspect_beads_action = threed_menu.addAction(
            "Inspect spline calibration beads"
        )
        self.inspect_beads_action.setToolTip(
            "Show the individual beads the last spline PSF calibration was\n"
            "built from, and which of them were rejected as outliers."
        )
        self.inspect_beads_action.setEnabled(False)
        self.inspect_beads_action.triggered.connect(
            self.inspect_calibration_beads
        )

        reregister_signal_action = threed_menu.addAction(
            "Multichannel realignment (current signal)"
        )
        reregister_signal_action.triggered.connect(
            self.reregister_channels_from_signal
        )

        self.plugin_menu = menu_bar.addMenu("Plugins")  # do not delete

    def open_config_location(self) -> None:
        """Open the folder holding the camera config file in the system
        file browser."""
        from .. import config_filename, resolve_config_path

        path = resolve_config_path()
        if path is None:
            # No config yet: point the user at the intended location.
            path = config_filename()
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    @property
    def camera_info(self) -> dict[str, float]:
        """Camera information, baseline, EM gain, sensitivity and QE."""
        camera_info = {}
        camera_info["Baseline"] = self.parameters_dialog.baseline.value()
        camera_info["Gain"] = self.parameters_dialog.gain.value()
        camera_info["Sensitivity"] = self.parameters_dialog.sensitivity.value()
        camera_info["Qe"] = self.parameters_dialog.qe.value()
        camera_info["Pixelsize"] = self.parameters_dialog.pixelsize.value()
        return camera_info

    def calibrate_z(self) -> None:
        """Use the loaded movie to obtain z-calibration data for 3D
        fitting using astigmatism."""
        self.localize(calibrate_z=True)

    def calibrate_affine(self) -> None:
        """Open the affine-transform calibration dialog (astigmatism or
        chromatic aberration).

        The dialog is non-modal, and "Calibrate" leaves it open, so the
        user can keep working in the main window - in particular load
        either bead image with "Show" to see the pairing the fit found -
        and re-run the fit without re-picking the paths.
        """
        # Reuse an existing dialog if it is already open, otherwise create
        # one and run the calibration whenever "Calibrate" is pressed.
        dialog = getattr(self, "_affine_dialog", None)
        if dialog is None:
            dialog = CalibrateAffineDialog(self)
            dialog.calibrate_requested.connect(self._run_calibrate_affine)
            self._affine_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _run_calibrate_affine(self) -> None:
        """Fit the affine transform between the two bead movies and append
        the result to the selected calibration file. Called when the
        affine-transform calibration dialog is accepted."""
        dialog = self._affine_dialog
        ref_path = dialog.reference_path
        target_path = dialog.target_path
        calib_path = dialog.calibration_path
        transform_type = dialog.transform_type
        model = dialog.transform_model
        if not (ref_path and target_path and calib_path):
            QtWidgets.QMessageBox.warning(
                self,
                "Calibrate lateral transform",
                "All three paths must be provided.",
            )
            return

        worker = getattr(self, "affine_calibration_worker", None)
        if worker is not None and worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Calibrate lateral transform",
                "A lateral calibration is already running.",
            )
            return

        def _pixelsize_prompt() -> float | None:
            """Ask for the camera pixel size (GUI thread). ``None`` on
            cancel, which aborts the calibration."""
            pixelsize, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Camera pixel size (nm)",
                "Enter camera pixel size in nm:",
                130,
                min=0,
            )
            return float(pixelsize) if ok else None

        worker = AffineCalibrationWorker(
            ref_path=ref_path,
            target_path=target_path,
            calibration_path=calib_path,
            transform_type=transform_type,
            model=model,
            box=self.parameters["Box Size"],
            minimum_ng=self.parameters_dialog.mng_slider.value(),
            prompt_for_path=self._prompt_for_path,
            pixelsize_prompt=_pixelsize_prompt,
        )
        worker.statusChanged.connect(self.status_bar.showMessage)
        worker.promptRequested.connect(self._on_affine_prompt_requested)
        worker.finished.connect(self.on_affine_calibration_finished)
        worker.failed.connect(self.on_affine_calibration_failed)
        worker.canceled.connect(self.on_affine_calibration_canceled)
        self.affine_calibration_worker = worker
        self.status_bar.showMessage("Calibrating lateral transform ...")
        worker.start()

    def _on_affine_prompt_requested(
        self, callback, args_kwargs, holder: dict
    ) -> None:
        """Run a worker-requested prompt on the GUI thread and hand the
        result back, then unblock the worker (see
        ``_on_load_prompt_requested``)."""
        args, kwargs = args_kwargs
        try:
            holder["result"] = callback(*args, **kwargs)
        finally:
            self.affine_calibration_worker._prompt_event.set()

    def on_affine_calibration_finished(
        self, path: str, n_pairs: int, qc: object
    ) -> None:
        """Save the augmented calibration and draw the diagnostic figure.

        The figure is drawn here rather than in the worker because
        matplotlib must be driven from the GUI thread."""
        self.affine_calibration_worker = None
        self.status_bar.showMessage(
            f"Lateral calibration appended to {os.path.basename(path)} "
            f"({n_pairs} bead pairs). Load either bead image to see which "
            "beads were paired."
        )
        # Keep the pairing so it can be inspected in the viewer: load either
        # bead image ("Show" in the calibration dialog) and the matched beads
        # are boxed, sharing a color between the two images.
        try:
            self.set_affine_pairing(qc)
            self.draw_frame()
        except Exception:  # bookkeeping must never lose a finished fit
            self.affine_pairing = None
        transform_type = qc.get("transform_type", "lateral")
        plot_path = (
            os.path.splitext(path)[0] + f"_lateral_{transform_type}.png"
        )
        try:
            localize.plot_lateral_calibration(qc, save_path=plot_path)
        except Exception as e:  # a failed figure must not lose the fit
            QtWidgets.QMessageBox.warning(
                self,
                "Calibrate lateral transform",
                f"The transform was saved to {path}, but the diagnostic "
                f"figure could not be drawn:\n{e}",
            )

    def on_affine_calibration_failed(self, message: str) -> None:
        """Report a failed lateral calibration."""
        self.affine_calibration_worker = None
        self.status_bar.showMessage("Lateral calibration failed.")
        QtWidgets.QMessageBox.warning(
            self, "Calibrate lateral transform", message
        )

    def on_affine_calibration_canceled(self) -> None:
        """The user canceled one of the worker's prompts."""
        self.affine_calibration_worker = None
        self.status_bar.showMessage("Lateral calibration canceled.")

    def calibrate_camera(self) -> None:
        """Characterize the sCMOS camera from a dark movie.

        Optionally also a bright series, which is what makes a per-pixel gain
        map possible; without it the scalar Sensitivity keeps being used.
        The dark movie is a separate acquisition from the data, so this never
        reuses the currently loaded movie.
        """

        (
            dark_path,
            light_paths,
            light_levels,
            level_unit,
            out_path,
            accepted,
        ) = CameraCalibrationDialog.getCalibrationSpecs(self)
        if not accepted or not dark_path or not out_path:
            return

        try:
            dark_movie, _ = io.load_movie(dark_path)
            light_movies = []
            for path in light_paths:
                movie, _ = io.load_movie(path)
                light_movies.append(movie)
            total = len(dark_movie) + sum(len(m) for m in light_movies)
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self, "Camera calibration", str(error)
            )
            return
        progress = lib.ProgressDialog("Characterizing camera", 0, total, self)
        progress.init()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                calibration = scmos.calibrate_scmos(
                    dark_movie,
                    light_movies or None,
                    progress_callback=progress.set_value,
                    dark_path=dark_path,
                    bright_paths=light_paths,
                    bright_levels=light_levels,
                    level_unit=level_unit,
                )
        except Exception as error:
            progress.close()
            QtWidgets.QMessageBox.critical(
                self, "Camera calibration", str(error)
            )
            return
        finally:
            progress.close()
        if calibration is None:
            return

        calibration["Path"] = out_path
        io.save_camera_calibration(out_path, calibration)
        self.update_camera_calib_from_worker(out_path)
        # A few summary numbers cannot show a dead column, a bright corner or
        # a cluster of hot pixels, so the maps go next to the calibration as
        # a diagnostic image.
        plot_path = scmos.plot_path(out_path)
        try:
            scmos.save_calibration_plot(calibration, plot_path)
        except Exception as error:
            plot_path = None
            plot_error = str(error)

        lines = [
            f"Frames used: {calibration['Frames']}",
            "Offset: median " f"{calibration['Offset median (ADU)']:.2f} ADU",
            "Readout variance: median "
            f"{calibration['Variance median (ADU^2)']:.2f}, max "
            f"{calibration['Variance max (ADU^2)']:.1f} ADU^2",
            f"Hot pixels: {calibration['Hot pixels']}",
        ]
        if calibration.get("gain") is not None:
            lines.append(
                "Gain: median "
                f"{calibration['Gain median (ADU/e-)']:.3f} ADU/e- from "
                f"{calibration['Gain levels']} illumination levels"
            )
        else:
            lines.append(
                "No gain map (no light movies given); the scalar "
                "Sensitivity is still used."
            )
        for warning in caught:
            lines.append("")
            lines.append(str(warning.message))
        lines.append("")
        lines.append(f"Saved to {out_path}.")
        if plot_path is None:
            lines.append(
                "The diagnostic plot could not be saved: " + plot_error
            )
        else:
            lines.append(f"Maps and histograms: {plot_path}")
        QtWidgets.QMessageBox.information(
            self, "Camera calibration", "\n".join(lines)
        )

    def update_camera_calib_from_worker(self, path: str) -> None:
        """Load a freshly built calibration into the parameters dialog."""
        self.parameters_dialog.update_camera_calib(path)

    def check_camera_calibration(self) -> None:
        """Test the loaded calibration against a short fresh dark movie."""

        calibration = self.parameters_dialog.camera_calibration
        if not calibration:
            QtWidgets.QMessageBox.information(
                self,
                "Camera calibration",
                "Load an sCMOS camera calibration first (Photon conversion > "
                "sCMOS noise maps).",
            )
            return
        movie_filter = "Movies (%s)" % " ".join(
            "*" + extension for extension in io.MOVIE_EXTENSIONS
        )
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a fresh dark movie", filter=movie_filter
        )
        if not path:
            return
        try:
            movie, _ = io.load_movie(path)
        except Exception as error:
            QtWidgets.QMessageBox.critical(
                self, "Camera calibration", str(error)
            )
            return
        progress = lib.ProgressDialog(
            "Checking calibration", 0, len(movie), self
        )
        progress.init()
        try:
            report = scmos.validate_calibration(
                calibration, movie, progress_callback=progress.set_value
            )
        except Exception as error:
            progress.close()
            QtWidgets.QMessageBox.critical(
                self, "Camera calibration", str(error)
            )
            return
        finally:
            progress.close()
        if report is None:
            return
        verdict = (
            "The calibration still describes this camera."
            if report["valid"]
            else "The camera looks "
            + ("noisier" if report["mean p-value"] < 0.5 else "quieter")
            + " than when it was characterized. Sensor temperature, readout "
            "mode and bit depth all change the maps; recalibrate before "
            "trusting the noise model."
        )
        QtWidgets.QMessageBox.information(
            self,
            "Camera calibration",
            f"Frames tested: {report['Frames']}\n"
            f"Mean p-value: {report['mean p-value']:.3f} "
            "(0.5 +- 0.1 is a match)\n\n" + verdict,
        )

    def _registration_save_path(self) -> str:
        """Ask where to save a channel registration."""
        base = self.movie_path or ""
        if base:
            base = os.path.splitext(base)[0] + "_channel_reg.yaml"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save channel registration",
            base,
            filter="*.yaml",
        )
        return path

    def _start_channel_registration(self, source: str, **kwargs) -> None:
        """Run one of the channel-registration builders in a worker thread."""
        path = self._registration_save_path()
        if not path:
            return
        self._snapshot_current_channel()
        regions = self._registration_regions()
        if regions is not None:
            # split field of view: one movie, its regions are the channels
            movies = [self.movie]
            channel_paths = [self.movie_path]
        else:
            movies = [channel.movie for channel in self.channels]
            channel_paths = [channel.path for channel in self.channels]
        self.registration_worker = ChannelRegistrationWorker(
            source=source,
            movies=movies,
            box=self.parameters["Box Size"],
            minimum_ng=self.parameters["Min. Net Gradient"],
            channel_paths=channel_paths,
            regions=regions,
            path=path,
            **kwargs,
        )
        self.registration_worker.statusChanged.connect(
            self.status_bar.showMessage
        )
        self.registration_worker.finished.connect(
            self.on_channel_registration_finished
        )
        self.registration_worker.failed.connect(
            self.on_channel_registration_failed
        )
        self.status_bar.showMessage("Registering channels...")
        self.registration_worker.start()

    def _registration_regions(self) -> list | None:
        """The drawn ROIs as channel regions, or None for separate movies."""
        if not self.view.split_fov_mode or len(self.view.rois) < 2:
            return None
        return [list(map(list, r)) for r in self.view.rois]

    def _check_registration_channels(self) -> bool:
        """Whether enough channels are loaded to register anything."""
        if self._registration_regions() is not None:
            return True
        if len(self.channels) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Register channels",
                "Load at least two channels first, with "
                "'File > Open channels from several movies' or "
                "'File > Open one multichannel movie' - or tick "
                "'Regions = channels' and draw one ROI per channel "
                "(reference first) to register a split field of view.",
            )
            return False
        return True

    def register_channels_from_beads(self) -> None:
        """Register the channels from the bead movies loaded here.

        The loaded channels (or the drawn regions of the loaded movie) are
        taken to be the bead images themselves, so open the bead movies as
        channels before running this."""
        if not self._check_registration_channels():
            return
        model, multi_fov, ok = RegisterChannelsDialog.getSpecs(self)
        if not ok:
            return
        self._start_channel_registration(
            "beads",
            model=model,
            multi_fov=multi_fov,
        )

    def register_channels_from_signal(self) -> None:
        """Register the channels from the loaded movies' own blinking signal.

        Seeds from the loaded registration when there is one, so this doubles
        as a re-alignment of a registration that has drifted."""
        if not self._check_registration_channels():
            return
        loaded = self.parameters_dialog.channel_registration_calibration or {}
        n_frames = min(int(c.movie.shape[0]) for c in self.channels)
        bounds, max_frames, model, ok = RefineRegistrationDialog.getFrameSpecs(
            self,
            n_frames,
            self.frame_range,
            model=loaded.get("registration_model"),
        )
        if not ok:
            return
        regions = self._registration_regions()
        n_channels = (
            len(regions) if regions is not None else len(self.channels)
        )
        seed_transforms = loaded.get("channel_transforms")
        if regions is not None and loaded.get("split_fov"):
            # place the stored inter-channel registration at the ROIs in use,
            # so a re-alignment starts from the fine offset rather than only
            # from where the regions sit
            try:
                _, _, seed_transforms = localize.split_fov_fit_geometry(
                    loaded, regions
                )
            except (ValueError, KeyError):
                seed_transforms = None
        if seed_transforms is not None and len(seed_transforms) != n_channels:
            seed_transforms = None
        self._start_channel_registration(
            "signal",
            model=model,
            frame_bounds=bounds,
            max_frames=max_frames,
            seed_transforms=seed_transforms,
        )

    def on_channel_registration_finished(self, path: str) -> None:
        """Load the freshly built registration into the parameters dialog.

        Also reports the per-channel pair count and residual RMS, which is what
        tells the user whether the registration is good enough to fit with.

        Parameters
        ----------
        path : str
            The registration file the worker saved.
        """
        self.status_bar.showMessage("")
        self.parameters_dialog.update_channel_registration(path)
        calibration = self.parameters_dialog.channel_registration_calibration
        rms = calibration.get("rms") or []
        pairs = calibration.get("n_pairs") or []
        detail = "\n".join(
            f"Channel {c + 1}: {n} correspondences, {r:.2f} px RMS"
            for c, (n, r) in enumerate(zip(pairs, rms))
        )
        QtWidgets.QMessageBox.information(
            self,
            "Register channels",
            f"Saved {os.path.basename(path)} and loaded it for fitting.\n\n"
            f"{detail}",
        )

    def on_channel_registration_failed(self, message: str) -> None:
        """Report a registration that could not be measured.

        Parameters
        ----------
        message : str
            Why it failed, from the worker. It names the channel, so a partly
            registerable set says which channel to look at.
        """
        self.status_bar.showMessage("")
        QtWidgets.QMessageBox.warning(self, "Register channels", message)

    def calibrate_spline(self) -> None:
        """Build a cubic-spline PSF calibration from the loaded bead z-stack
        movie. Detects beads, averages them into a PSF volume and
        computes the spline coefficients, saved as an HDF5 calibration."""
        if self.movie is None:
            QtWidgets.QMessageBox.information(
                self, "Spline PSF Calibration", "No file loaded."
            )
            return

        # identify spots first so the beads are displayed (like the
        # astigmatic 3D calibration); the calibration is built afterwards.
        self.identify(calibrate_spline=True)

    def set_split_fov_mode(self, enabled: bool) -> None:
        """Enable/disable split-FOV region mode, where the drawn ROIs are the
        channels of one movie. The shared region size is derived live from the
        existing regions (see ``View._region_size``); enabling the mode only
        gives the view keyboard focus for arrow-key nudging.

        Split-FOV (regions = channels of one movie) is mutually exclusive with
        separate-movie multichannel data: when several movies are loaded the
        request is rejected and the checkbox reverted."""
        if enabled and len(self.channels) > 1:
            QtWidgets.QMessageBox.information(
                self,
                "Split-FOV mode",
                "'Regions = channels' treats the ROIs of a single movie as "
                "channels and cannot be used when several movies are loaded "
                "as separate channels.",
            )
            checkbox = self.parameters_dialog.split_fov_checkbox
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            self.view.split_fov_mode = False
            self._update_multichannel_widgets()
            return
        self.view.split_fov_mode = bool(enabled)
        if enabled:
            self.view.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        # seed each region's own min. net gradient from the current slider
        # value on the way in, drop them again on the way out
        self.region_mngs()
        self.parameters_dialog.update_roi_display()
        self._update_multichannel_widgets()
        self.draw_frame()

    def _update_multichannel_widgets(self) -> None:
        """Show the multichannel-only widgets only when the data actually has
        several channels: either several movies loaded as separate channels or
        split-FOV mode, where the drawn regions are the channels of one movie.

        'Link colors' needs channels to pair spots across; the shared-settings
        group links per-channel parameters and therefore applies to separate
        channels only (split-FOV channels share one movie's parameters).
        'Identify on' needs channels to add together, and 'Fit' channels to
        fit one at a time, so both follow the same rule as 'Link colors'.

        All of them are hidden rather than disabled. The row widgets keep
        their space while hidden (see ``_retain_size_when_hidden``), so the
        rows they sit in do not reflow; the shared-settings group box does
        not, since reserving a whole group box leaves a visible gap in the
        single-channel dialog."""
        separate_channels = len(self.channels) > 1
        multichannel = separate_channels or self.view.split_fov_mode
        pdialog = self.parameters_dialog
        pdialog.link_groupbox.setVisible(separate_channels)
        checkbox = pdialog.link_colors_checkbox
        if not multichannel and checkbox.isChecked():
            # leaving it checked while hidden would keep color-coding the
            # identification boxes with no way to turn it off
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        checkbox.setVisible(multichannel)
        combo = pdialog.identify_mode_combo
        if not multichannel and combo.currentText() != IDENTIFY_MODE_SEPARATE:
            # as above: leaving it on the sum while hidden would identify
            # single-channel data in a mode that cannot be turned off
            combo.blockSignals(True)
            combo.setCurrentText(IDENTIFY_MODE_SEPARATE)
            combo.blockSignals(False)
            self.drop_channel_sum()
        pdialog.identify_mode_label.setVisible(multichannel)
        combo.setVisible(multichannel)
        fit_combo = pdialog.fit_mode_combo
        if not multichannel and fit_combo.currentText() != FIT_MODE_JOINT:
            # as above: leaving it on the separate mode while hidden would
            # keep single-channel data in a mode that cannot be turned off
            fit_combo.blockSignals(True)
            fit_combo.setCurrentText(FIT_MODE_JOINT)
            fit_combo.blockSignals(False)
        pdialog.fit_mode_label.setVisible(multichannel)
        fit_combo.setVisible(multichannel)

    def build_spline_calibration(self) -> None:
        """Prompt for the spline PSF calibration parameters and build the
        calibration from the loaded bead z-stack in a background thread."""
        split_fov = self.view.split_fov_mode
        regions = list(self.view.rois) if split_fov else None
        if split_fov and len(regions) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Spline PSF Calibration",
                "Split-FOV mode is on: draw at least 2 equal-size regions "
                "(channels) before calibrating. The first region is the "
                "reference channel.",
            )
            return

        # Separate-channel multichannel: several channels are loaded (separate
        # files or a multichannel file) and split-FOV mode is off. The first
        # loaded channel is the reference.
        multichannel = not split_fov and len(self.channels) > 1
        if multichannel:
            n_frames = {int(c.movie.shape[0]) for c in self.channels}
            if len(n_frames) != 1:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Spline PSF Calibration",
                    "The loaded channels have different frame counts "
                    f"({sorted(n_frames)}). A multichannel calibration needs "
                    "every channel to share the same z-scan layout.",
                )
                return

        specs = CalibrateSplineDialog.getCalibrationSpecs(
            self, multichannel=multichannel or split_fov
        )
        (
            step,
            frames_per_step,
            frame_order,
            model,
            magnification_factor,
            correct_z_bias,
            link_photons,
            registration_model,
            accepted,
        ) = specs
        if not accepted:
            return

        base = os.path.splitext(self.movie_path)[0] if self.movie_path else ""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save spline PSF calibration",
            base + "_spline_calib.hdf5",
            filter="*.hdf5",
        )
        if not path:
            return

        parameters = self.parameters
        # Separate-channel multichannel build: pass every channel's movie/info
        # (reference first) so the worker calls calibrate_spline_multichannel.
        # Camera info is shared (Localize keeps a single camera-parameter set).
        movies = infos = camera_infos = None
        # The frame range is shared across channels (window-level), so it
        # applies to the reference channel and every other channel alike.
        frame_bounds = self.frame_range
        if multichannel:
            movies = [c.movie for c in self.channels]
            infos = [c.info for c in self.channels]
            camera_infos = [self.camera_info for _ in self.channels]
        self.spline_calibration_worker = SplineCalibrationWorker(
            movie=self.movie,
            info=self.info,
            camera_info=self.camera_info,
            box=parameters["Box Size"],
            minimum_ng=parameters["Min. Net Gradient"],
            step=step,
            frames_per_step=frames_per_step,
            frame_order=frame_order,
            frame_bounds=frame_bounds,
            model=model,
            magnification_factor=magnification_factor,
            correct_z_bias=correct_z_bias,
            link_photons=link_photons,
            registration_model=registration_model,
            roi=self.view.rois,
            regions=regions,
            movies=movies,
            infos=infos,
            camera_infos=camera_infos,
            path=path,
        )
        self.spline_calibration_worker.statusChanged.connect(
            self.status_bar.showMessage
        )
        self.spline_calibration_worker.finished.connect(
            self.on_spline_calibration_finished
        )
        self.spline_calibration_worker.failed.connect(
            self.on_spline_calibration_failed
        )
        if multichannel:
            msg = (
                f"Building multichannel spline PSF calibration from "
                f"{len(self.channels)} channels (reference: "
                f"{self.channels[0].name}) ..."
            )
        elif regions:
            msg = (
                f"Building split-FOV spline PSF calibration from "
                f"{len(regions)} regions ..."
            )
        else:
            msg = "Building spline PSF calibration ..."
        self.status_bar.showMessage(msg)
        self.spline_calibration_worker.start()

    def on_spline_calibration_finished(
        self, path: str, n_beads: int, n_used: int
    ) -> None:
        """Report a successful spline PSF calibration, listing the diagnostic
        images that were written next to it (so a missing one is obvious) and
        offering the bead inspector when beads were filtered out."""
        self.status_bar.showMessage("")
        worker = self.spline_calibration_worker
        self.bead_diagnostics = list(
            getattr(worker, "bead_diagnostics", []) or []
        )
        self.bead_calibration_path = path
        self.inspect_beads_action.setEnabled(bool(self.bead_diagnostics))
        base = os.path.splitext(path)[0]
        written = [
            os.path.basename(p) for p in sorted(glob.glob(base + "_*.png"))
        ]
        n_rejected = max(0, n_beads - n_used)
        built_from = f"{n_beads} beads"
        if n_rejected:
            # do not let the bead count imply that every detected bead is in
            # the PSF - point the user at the inspector to check the ones that
            # were dropped
            built_from = (
                f"{n_used} of {n_beads} detected beads ({n_rejected} rejected "
                "as outliers)"
            )
        lines = [
            f"Spline PSF calibration built from {built_from} and saved to:",
            path,
        ]
        if written:
            lines += ["", "Diagnostics written:"] + [f"  {w}" for w in written]
        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Icon.Information)
        message.setWindowTitle("Spline PSF Calibration")
        message.setText("\n".join(lines))
        message.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        if self.bead_diagnostics:
            inspect = message.addButton(
                "Inspect beads...",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
        else:
            inspect = None
        message.exec()
        if inspect is not None and message.clickedButton() is inspect:
            self.inspect_calibration_beads()

    def inspect_calibration_beads(self) -> None:
        """Show the beads of the last spline PSF calibration built in this
        session, marking those that were rejected as outliers."""
        if not self.bead_diagnostics:
            QtWidgets.QMessageBox.information(
                self,
                "Calibration beads",
                "No spline PSF calibration has been built in this session. "
                "Build one (Calibration > Calibrate spline PSF) to inspect "
                "the beads it averaged; the same gallery is also saved next "
                "to every calibration as <name>_beads.png.",
            )
            return
        # the dialog is parented to the window, so an earlier one would linger
        # as a hidden child; drop it before opening the new one
        previous = getattr(self, "bead_inspection_dialog", None)
        if previous is not None:
            previous.close()
            previous.deleteLater()
        self.bead_inspection_dialog = BeadInspectionDialog(
            self.bead_diagnostics,
            self,
            title=os.path.basename(self.bead_calibration_path or ""),
        )
        self.bead_inspection_dialog.show()

    def on_spline_calibration_failed(self, message: str) -> None:
        """Report a failed spline PSF calibration."""
        self.status_bar.showMessage("")
        QtWidgets.QMessageBox.critical(self, "Spline PSF Calibration", message)

    def show_metadata(self) -> None:
        """Open the metadata dialog."""
        if self.movie is None:
            QtWidgets.QMessageBox.information(
                self, "Metadata", "No file loaded."
            )
            return
        infos = self.extra_info if self.extra_info else self.info
        label = os.path.basename(self.movie_path) if self.movie_path else None
        self.metadata_dialog.set_infos(infos, labels=label)
        self.metadata_dialog.show()
        self.metadata_dialog.raise_()

    def open_file_dialog(self) -> None:
        """Open a file dialog to select a movie file to load."""
        if self.pwd == []:
            dir = None
        else:
            dir = self.pwd

        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open image sequence",
            directory=dir,
            filter=IMAGE_FILTER,
        )
        if path:
            self.pwd = path
            self.open(path)

    def open_mm_folder_dialog(self) -> None:
        """Open a MicroManager "separate image files" acquisition folder.

        MicroManager can save each frame of a movie as its own TIFF
        (``img_*.tif``) inside one folder. This picks such a folder and
        loads the whole sequence as a single movie.
        """
        dir = None if self.pwd == [] else self.pwd
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open MicroManager image folder", directory=dir
        )
        if not directory:
            return
        path = io.find_mm_separate_first(directory)
        if path is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No image sequence found",
                "No MicroManager 'separate image files' sequence "
                "(img_*.tif) was found in the selected folder.",
            )
            return
        self.pwd = path
        self.open(path)

    def open_concatenated_dialog(self) -> None:
        """Open the TIFF movies found below one folder as a single movie.

        An acquisition split over several files and folders is loaded as
        one movie, with the frames running through the files in the order
        confirmed in ``ConcatenateMoviesDialog``.
        """
        dir = None if self.pwd == [] else self.pwd
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the folder containing the movies", directory=dir
        )
        if not directory:
            return
        paths = io.find_tif_movies(directory)
        if not paths:
            QtWidgets.QMessageBox.warning(
                self,
                "No movies found",
                f"No TIFF movies were found in {directory} or its "
                "sub-folders.",
            )
            return
        paths, ok = ConcatenateMoviesDialog.getPaths(
            self, paths, root=directory
        )
        if not ok or not paths:
            return
        self.pwd = paths[0]
        self._start_movie_load(paths, self._prompt_for_path, concat=True)

    def _prompt_for_path(self, path: str):
        """Return the metadata prompt callback appropriate for ``path``."""
        if path.lower().endswith((".ims", ".czi", ".lif")):
            # Multi-channel .ims/.czi/.lif files prompt for a channel.
            return self.prompt_channel
        elif path.lower().endswith(io.TIFF_EXTENSIONS + (".stk", ".nd2")):
            # For these formats the metadata may fail to parse; prompt the
            # user to enter it manually as a fallback.
            return self.prompt_movie_info
        return self.prompt_info

    def open(self, path: str) -> None:
        """Open a single movie file as one channel."""
        self._start_movie_load([path], self._prompt_for_path)

    def open_multichannel_file_dialog(self) -> None:
        """Open a single file and load *all* of its channels."""
        dir = None if self.pwd == [] else self.pwd
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open multichannel file",
            directory=dir,
            filter="Multichannel files (*.ims *.czi *.lif *.nd2)",
        )
        if path:
            self.pwd = path
            self.open_multichannel_file(path)

    def open_multichannel_file(self, path: str) -> None:
        """Load every channel of a single multichannel file."""
        self._start_movie_load(
            [path], lambda _p: self.prompt_movie_info, load_all=True
        )

    def open_channels_from_files_dialog(self) -> None:
        """Open several movie files, each loaded as one channel."""
        dir = None if self.pwd == [] else self.pwd
        paths, exe = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open channels from files",
            directory=dir,
            filter=(
                "All supported formats ("
                + " ".join("*" + e for e in io.MOVIE_EXTENSIONS)
                + ")"
            ),
        )
        if paths:
            self.pwd = paths[0]
            self.open_channels_from_files(paths)

    def open_channels_from_files(self, paths: list[str]) -> None:
        """Load several movie files as channels (one file per channel)."""
        self._start_movie_load(paths, self._prompt_for_path, multi_file=True)

    def _start_movie_load(
        self,
        paths: list[str],
        prompt_for_path,
        load_all: bool = False,
        multi_file: bool = False,
        concat: bool = False,
    ) -> None:
        """Load movies on a background thread (see ``MovieLoadWorker``) so
        the GUI keeps repainting and responding while files are read.

        ``load_all`` reads every channel of each file (single multichannel
        file); otherwise one channel is loaded per file. ``multi_file``
        controls channel naming when several separate files are loaded.
        ``concat`` reads all ``paths`` as a single movie whose frames run
        through the files in the given order.
        """
        if self._load_thread is not None:
            # A load is already in progress; ignore re-entrant requests.
            # (A canceled one is detached immediately, see
            # ``_on_load_canceled``, so it never blocks a new load.)
            return

        self._load_t0 = time.time()
        self._load_multi_file = multi_file
        # Each file spans PROGRESS_RESOLUTION steps so the bar can advance
        # smoothly *within* a file from the worker's per-page reports,
        # instead of only ticking once per file.
        self._load_index = 0
        # Concatenation reads every file into one movie, so the whole
        # load is a single step whose sub-progress already spans all
        # files (see ``io._scaled_progress``).
        n_steps = 1 if concat else len(paths)
        progress = QtWidgets.QProgressDialog(
            "Loading movie...",
            "Cancel",
            0,
            n_steps * PROGRESS_RESOLUTION,
            self,
        )
        progress.setWindowTitle("Opening movie")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._load_progress = progress

        thread = QtCore.QThread(self)
        worker = MovieLoadWorker(
            paths, prompt_for_path, load_all=load_all, concat=concat
        )
        worker.moveToThread(thread)
        self._load_thread = thread
        self._load_worker = worker

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_load_progress)
        worker.subprogress.connect(self._on_load_subprogress)
        worker.prompt_requested.connect(self._on_load_prompt_requested)
        worker.finished.connect(self._on_load_finished)
        worker.failed.connect(self._on_load_failed)
        # Direct connection so the flag is set from the GUI thread
        # immediately; the worker thread is busy in run() and would not
        # service a queued slot until the current file finished.
        progress.canceled.connect(
            worker.cancel, QtCore.Qt.ConnectionType.DirectConnection
        )
        progress.canceled.connect(self._on_load_canceled)
        thread.start()

    def _on_load_progress(self, index: int, filename: str) -> None:
        """Update the progress dialog as each file starts loading.

        The bar cannot move yet: before per-page reporting begins, the
        loader indexes the file's frames, and their number - the total
        the bar needs - is only known when that is done. Say so, so the
        user does not think the load has stalled."""
        self._load_index = index
        self._load_filename = filename
        if self._load_progress is not None:
            self._load_progress.setLabelText(
                f"Opening {filename}...\n"
                "Indexing frames — the bar starts once they are counted."
            )
            self._load_progress.setValue(index * PROGRESS_RESOLUTION)

    def _on_load_subprogress(self, done: int, total: int) -> None:
        """Advance the bar within the current file from the worker's
        per-page reports (see ``MovieLoadWorker.subprogress``).

        A total of 0 means the frames are still being counted (see
        ``io._index_pages``), so there is nothing to advance towards -
        report the running count in the label instead."""
        if self._load_progress is None:
            return
        if total > 0:
            fraction = min(done / total, 1.0)
            value = int((self._load_index + fraction) * PROGRESS_RESOLUTION)
            self._load_progress.setValue(value)
        else:
            self._load_progress.setLabelText(
                f"Opening {self._load_filename}...\n"
                f"Indexing frames — {done} counted so far."
            )

    def _on_load_prompt_requested(
        self, callback, args_kwargs, holder: dict
    ) -> None:
        """Run a worker-requested metadata prompt on the GUI thread and
        hand the result back, then unblock the worker."""
        # the asking worker, which is not necessarily the current one
        # anymore (it may have been canceled and detached meanwhile)
        worker = self.sender()
        if not isinstance(worker, MovieLoadWorker):
            worker = self._load_worker
        args, kwargs = args_kwargs
        try:
            holder["result"] = callback(*args, **kwargs)
        finally:
            if worker is not None:
                worker._prompt_event.set()

    def _on_load_canceled(self) -> None:
        """Give the GUI back to the user the moment they cancel.

        The worker stops at its next cancellation check, but the io call
        it is inside cannot be interrupted, so that can still take a
        while on a slow file system. Instead of waiting for it, the
        worker is *detached*: its signals are disconnected (its result is
        no longer wanted) and it is left to wind down on its own, so a
        new movie can be opened right away. Dropping the progress dialog
        is part of that - ``setValue()`` re-shows a canceled one."""
        progress, self._load_progress = self._load_progress, None
        if progress is not None:
            # disconnect first: closing re-emits ``canceled``
            try:
                progress.canceled.disconnect()
            except TypeError:  # nothing connected
                pass
            progress.close()
        thread, worker = self._load_thread, self._load_worker
        self._load_thread = None
        self._load_worker = None
        if worker is None or thread is None:
            return
        # cancel() also releases a worker blocked on a metadata prompt
        worker.cancel()
        for signal in (
            worker.progress,
            worker.subprogress,
            worker.prompt_requested,
            worker.finished,
            worker.failed,
        ):
            try:
                signal.disconnect()  # no more progress/results wanted
            except TypeError:  # nothing connected
                pass
        # Keep a reference until the thread has really stopped: a QThread
        # destroyed while still running aborts the process.
        self._detached_loads.append((thread, worker))
        thread.finished.connect(
            lambda t=thread: self._on_detached_load_done(t)
        )
        # Ask the thread's event loop to exit; it does so as soon as the
        # worker returns from the io call it is in. Requesting it here
        # rather than from ``worker.finished`` also covers the worker
        # having finished already, just before this cancellation.
        thread.quit()
        self.status_bar.showMessage(
            "Load canceled (the file is still being released in the "
            "background)."
        )

    def _on_detached_load_done(self, thread: QtCore.QThread) -> None:
        """Drop a canceled load's worker thread once it has stopped."""
        for i, (detached, worker) in enumerate(self._detached_loads):
            if detached is thread:
                del self._detached_loads[i]
                worker.deleteLater()
                thread.deleteLater()
                break

    def _stop_movie_load(self) -> None:
        """Cancel any in-progress load before the window/app is
        destroyed, because destroying a live QThread aborts the process.

        Waits only briefly: a worker stuck in an uninterruptible io call
        must not hold up quitting. ``_loads_still_running`` then reports
        that one is left over, so ``closeEvent`` can leave the process
        without destroying it."""
        WAIT_MS = 2000
        for thread, worker in [
            (self._load_thread, self._load_worker),
            *self._detached_loads,
        ]:
            if worker is not None:
                worker.cancel()
            if thread is not None:
                thread.quit()
                thread.wait(WAIT_MS)

    def _stop_active_worker(self) -> None:
        """Interrupt the running identification or fit before the window is
        destroyed, because destroying a live QThread aborts the process.

        The workers poll ``isInterruptionRequested`` between frames, so this
        returns as soon as the frame in flight is done. Waits only briefly:
        ``_worker_still_running`` then reports that one is left over, so
        ``closeEvent`` can leave the process without destroying it."""
        WAIT_MS = 5000
        for worker in self._compute_workers():
            worker.requestInterruption()
            worker.wait(WAIT_MS)

    def _compute_workers(self) -> list:
        """The identification/fit threads that are still running."""
        workers = [
            getattr(self, name, None)
            for name in (
                "_active_worker",
                "identification_worker",
                "fit_worker",
                "fit_z_worker",
            )
        ]
        running = []
        for worker in workers:
            if worker is None or not worker.isRunning():
                continue
            if not any(worker is other for other in running):
                running.append(worker)
        return running

    def _worker_still_running(self) -> bool:
        """True if an identification/fit thread has not come back."""
        return bool(self._compute_workers())

    def _loads_still_running(self) -> bool:
        """True if a loader thread is still inside an io call."""
        threads = [self._load_thread, *(t for t, _ in self._detached_loads)]
        return any(t is not None and t.isRunning() for t in threads)

    def _finish_load(self) -> None:
        """Tear down the worker thread and progress dialog."""
        if self._load_progress is not None:
            # closing a QProgressDialog emits ``canceled``, which would
            # detach the worker that just finished (see
            # ``_on_load_canceled``)
            try:
                self._load_progress.canceled.disconnect()
            except TypeError:  # nothing connected
                pass
            self._load_progress.close()
            self._load_progress = None
        if self._load_thread is not None:
            self._load_thread.quit()
            self._load_thread.wait()
            self._load_thread.deleteLater()
            self._load_thread = None
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None

    def _on_load_finished(
        self, movies: list, infos: list, paths: list
    ) -> None:
        """Activate the loaded channels once the worker is done."""
        self._finish_load()
        if not movies:
            return
        names = [
            self._channel_name(
                infos[i], paths[i], i, multi_file=self._load_multi_file
            )
            for i in range(len(movies))
        ]
        self._set_channels(movies, infos, paths, names)
        dt = time.time() - self._load_t0
        if len(movies) == 1:
            self.status_bar.showMessage(f"Opened movie in {dt:.2f} seconds.")
        else:
            self.status_bar.showMessage(
                f"Opened {len(movies)} channel(s) in {dt:.2f} seconds."
            )

    def _on_load_failed(self, message: str) -> None:
        """Report a load error and tear down the worker."""
        self._finish_load()
        QtWidgets.QMessageBox.warning(self, "Could not load file", message)

    def _channel_name(
        self,
        info: list,
        path: str,
        idx: int,
        multi_file: bool = False,
    ) -> str:
        """Pick a display name for a channel: the metadata ``"Channel"``
        key if present, else the filename stem (separate files), else
        ``"Channel N"``."""
        try:
            channel = info[0].get("Channel")
        except (AttributeError, IndexError, TypeError):
            channel = None
        if channel:
            return str(channel)
        if multi_file:
            return os.path.splitext(os.path.basename(path))[0]
        return f"Channel {idx}"

    def _warn_if_channel_lengths_differ(self, infos: list) -> None:
        """Warn (non-blocking) when the channels have different frame
        counts. Multichannel identify/fit pair spots frame-by-frame across
        channels and the shared frame slider / frame range are clamped to
        each channel's length, so unequal lengths are almost always a
        mistake. The load still proceeds so the movies can be inspected."""
        if len(infos) < 2:
            return
        try:
            frame_counts = [
                int(lib.get_from_metadata(info, "Frames")) for info in infos
            ]
        except Exception:
            return
        if len(set(frame_counts)) > 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Channels differ in length",
                "The loaded channels have different numbers of frames "
                f"({', '.join(str(n) for n in frame_counts)}).\n\n"
                "Channels should have the same length: multichannel "
                "identification and fitting pair spots frame-by-frame "
                "across channels, and the shared frame slider and frame "
                "range are clamped to each channel's length. Please load "
                "channels with equal frame counts.",
            )

    def _set_channels(
        self,
        movies: list,
        infos: list,
        paths: list,
        names: list,
    ) -> None:
        """Build the channel list from loaded movies and activate the
        first one. The single funnel used by every load path."""
        if not movies:
            return
        self._warn_if_channel_lengths_differ(infos)
        # the preview's link colors cache the other channels' detections and
        # filter stacks, and these are other channels now
        self.drop_preview_link_cache()
        # Build channels and seed each channel's parameter/contrast
        # snapshot from the current dialog state, applying any camera /
        # pixel-size hints from that channel's metadata. Guarded so the
        # value changes don't wipe localizations.
        self._switching_channel = True
        try:
            self.channels = []
            for movie, info, path, name in zip(movies, infos, paths, names):
                channel = Channel(movie=movie, info=info, path=path, name=name)
                self.parameters_dialog.set_camera_parameters(info[0])
                if "Pixelsize" in info[0]:
                    self.parameters_dialog.pixelsize.setValue(
                        int(info[0]["Pixelsize"])
                    )
                channel.params = self._capture_params()
                self.channels.append(channel)
            self.current_channel = 0
        finally:
            self._switching_channel = False
        self._populate_channel_combo()
        self.frame_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.curr_frame_number = 0
        self._restore_current_channel()
        self.draw_frame()
        self.parameters_dialog.reset_quality_check()

    def _populate_channel_combo(self) -> None:
        """Refresh the channel selector; hidden unless several channels."""
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems([c.name for c in self.channels])
        self.channel_combo.setCurrentIndex(self.current_channel)
        self.channel_combo.blockSignals(False)
        multichannel = len(self.channels) > 1
        self.channel_combo.setVisible(multichannel)
        self.previous_channel_action.setEnabled(multichannel)
        self.next_channel_action.setEnabled(multichannel)
        # Split-FOV (regions = channels of one movie) is incompatible with
        # separate-movie multichannel data
        checkbox = self.parameters_dialog.split_fov_checkbox
        if multichannel and checkbox.isChecked():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            self.view.split_fov_mode = False
            self.draw_frame()
        checkbox.setEnabled(not multichannel)
        self._update_multichannel_widgets()

    def on_channel_combo_changed(self, index: int) -> None:
        """Switch the active channel from the selector."""
        self.set_current_channel(index)

    def previous_channel(self) -> None:
        """Activate the channel above the current one (Up arrow)."""
        self.set_current_channel(self.current_channel - 1)

    def next_channel(self) -> None:
        """Activate the channel below the current one (Down arrow)."""
        self.set_current_channel(self.current_channel + 1)

    def set_current_channel(self, index: int) -> None:
        """Make channel ``index`` the active one, swapping the flat state
        and the Parameters/Contrast dialog values."""
        if (
            index == self.current_channel
            or index < 0
            or index >= len(self.channels)
        ):
            return
        self._snapshot_current_channel()
        self.current_channel = index
        # Keep the selector in sync for switches that did not come from it
        # (keyboard navigation, multichannel identify batches).
        self.channel_combo.blockSignals(True)
        self.channel_combo.setCurrentIndex(index)
        self.channel_combo.blockSignals(False)
        self._restore_current_channel()
        self.parameters_dialog.reset_quality_check()

    def _snapshot_current_channel(self) -> None:
        """Store the active flat state + dialog values into its Channel."""
        if not self.channels:
            return
        channel = self.channels[self.current_channel]
        channel.movie = self.movie
        channel.info = self.info
        channel.path = self.movie_path
        channel.identifications = self.identifications
        channel.locs = self.locs
        channel.locs_display = self.locs_display
        channel.ready_for_fit = self.ready_for_fit
        channel.last_identification_info = self.last_identification_info
        channel.extra_info = self.extra_info
        channel.params = self._capture_params()
        # Contrast, the current frame and the frame range are shared across
        # channels (see _restore_current_channel), so they are not
        # snapshotted per channel.

    def _restore_current_channel(self) -> None:
        """Load the active Channel's state into the flat attrs + dialogs.

        Contrast and the current frame are shared across channels, so they
        are intentionally *not* restored per channel: the contrast dialog is
        left as-is and the shared frame number is reused (only clamped to
        this channel's length)."""
        channel = self.channels[self.current_channel]
        self._switching_channel = True
        try:
            self.movie = channel.movie
            self.info = channel.info
            self.movie_path = channel.path
            self.identifications = channel.identifications
            self.locs = channel.locs
            self.locs_display = channel.locs_display
            self.ready_for_fit = channel.ready_for_fit
            self.last_identification_info = channel.last_identification_info
            self.extra_info = channel.extra_info
            self._apply_params(channel.params)
            # Contrast and the frame range are shared: keep the dialogs
            # as-is (do not restore per-channel values).
        finally:
            self._switching_channel = False
        # The current frame is shared across channels too.
        self._apply_channel_to_ui(getattr(self, "curr_frame_number", 0))

    def _apply_channel_to_ui(self, frame_number: int = 0) -> None:
        """Sync the frame slider, displayed frame, zoom and title to the
        active channel."""
        last_frame = lib.get_from_metadata(self.info, "Frames") - 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, last_frame))
        self.frame_slider.blockSignals(False)
        frame_number = min(max(0, frame_number), max(0, last_frame))
        self.curr_frame_number = frame_number
        # before set_frame: this channel's intensities can differ from the
        # previous one's, and set_frame's auto contrast then lands on a
        # track that already fits
        self.update_contrast_slider_range()
        self.set_frame(frame_number)
        self.fit_in_view()
        self._update_temporal_median_availability()
        base = os.path.basename(self.movie_path) if self.movie_path else ""
        title = f"Picasso v{__version__}: Localize. File: {base}"
        if len(self.channels) > 1:
            title += f" [{self.channels[self.current_channel].name}]"
        self.setWindowTitle(title)

    def _capture_params(self) -> dict:
        """Snapshot the analysis-relevant Parameters dialog values."""
        pd = self.parameters_dialog
        params = {
            "box": pd.box_spinbox.value(),
            "mng": pd.mng_slider.value(),
            "mng_min": pd.mng_min_spinbox.value(),
            "mng_max": pd.mng_max_spinbox.value(),
            "fit_model": pd.fit_model.currentIndex(),
            "fit_optimizer": pd.fit_optimizer.currentIndex(),
            "baseline": pd.baseline.value(),
            "gain": pd.gain.value(),
            "sensitivity": pd.sensitivity.value(),
            "qe": pd.qe.value(),
            "pixelsize": pd.pixelsize.value(),
            "convergence": pd.convergence_criterion.value(),
            "max_it": pd.max_it.value(),
            "magnification": pd.magnification_factor.value(),
            "fit_z": pd.fit_z_checkbox.isChecked(),
            "fit_z_enabled": pd.fit_z_checkbox.isEnabled(),
            "z_calibration": pd.z_calibration,
            "z_calibration_path": pd.z_calibration_path,
            "z_calib_label": pd.z_calib_label.text(),
            # per loaded movie: each can need its own correction
            "lateral_transforms": pd.lateral_transforms,
            "affine_calibration_paths": pd.affine_calibration_paths,
            # the PSF is measured per channel, so a channel fitted on its own
            # can need its own calibration (shared unless the link is off)
            "spline_calibration": pd.spline_calibration,
            "spline_calibration_path": pd.spline_calibration_path,
            "use_gpu": pd.gpu_checkbox.isChecked(),
            "temporal_median_on": pd.temporal_median_checkbox.isChecked(),
            "temporal_median": pd.temporal_median_spinbox.value(),
            "gaussian_filter_sigma": pd.gaussian_filter_spinbox.value(),
        }
        if hasattr(pd, "camera"):
            params["camera"] = pd.camera.currentIndex()
        return params

    def _apply_params(self, params: dict) -> None:
        """Restore a channel's analysis parameters into the dialog. Caller
        sets ``self._switching_channel`` to suppress side effects."""
        if not params:
            return
        pd = self.parameters_dialog
        # Settings whose "Same across channels" box is ticked are shared, so
        # they are not restored per channel (the current dialog value stays).
        link_box = pd.link_box_checkbox.isChecked()
        link_mng = pd.link_mng_checkbox.isChecked()
        link_cam = pd.link_camera_checkbox.isChecked()
        link_calib = pd.link_calib_checkbox.isChecked()
        # Camera selection first: its cascade fills baseline/gain/pixelsize
        # from the config, which we then override with the channel's values.
        if not link_cam and hasattr(pd, "camera") and "camera" in params:
            pd.camera.setCurrentIndex(params["camera"])
        if not link_box:
            pd.box_spinbox.setValue(params["box"])
        if not link_mng:
            pd.mng_min_spinbox.setValue(params["mng_min"])
            pd.mng_max_spinbox.setValue(params["mng_max"])
            pd.mng_slider.setValue(params["mng"])
            pd.mng_spinbox.setValue(params["mng"])
        # Set the model first so its handler repopulates the optimizer list,
        # then restore the optimizer selection.
        pd.fit_model.setCurrentIndex(params.get("fit_model", 0))
        pd.fit_optimizer.setCurrentIndex(params.get("fit_optimizer", 0))
        if not link_cam:
            pd.baseline.setValue(params["baseline"])
            pd.gain.setValue(params["gain"])
            pd.sensitivity.setValue(params["sensitivity"])
            pd.qe.setValue(params["qe"])
            pd.pixelsize.setValue(params["pixelsize"])
        pd.convergence_criterion.setValue(params["convergence"])
        pd.max_it.setValue(params["max_it"])
        # .get() with defaults: parameter sets captured before the
        # identification filters existed must still restore
        pd.temporal_median_spinbox.setValue(
            params.get(
                "temporal_median", DEFAULT_PARAMETERS["Temporal Median Window"]
            )
        )
        pd.temporal_median_checkbox.setChecked(
            params.get("temporal_median_on", False)
        )
        pd.gaussian_filter_spinbox.setValue(
            params.get(
                "gaussian_filter_sigma",
                DEFAULT_PARAMETERS["Gaussian Filter Sigma"],
            )
        )
        pd.magnification_factor.setValue(params["magnification"])
        pd.z_calibration = params["z_calibration"]
        pd.z_calibration_path = params["z_calibration_path"]
        pd.z_calib_label.setText(params["z_calib_label"])
        pd.fit_z_checkbox.setEnabled(params["fit_z_enabled"])
        pd.fit_z_checkbox.setChecked(params["fit_z"])
        # .get(): parameter sets captured before affine corrections existed
        pd._set_affine_state(
            params.get("lateral_transforms", []),
            params.get("affine_calibration_paths", []),
        )
        # .get(): parameter sets captured before per-channel PSF calibrations
        if not link_calib and "spline_calibration" in params:
            pd._set_spline_state(
                params["spline_calibration"],
                params.get("spline_calibration_path"),
            )
        # Saved parameter files written before the Numba CUDA port use the
        # old "gpufit" key.
        pd.gpu_checkbox.setChecked(
            params.get("use_gpu", params.get("gpufit", False))
        )

    def propagate_linked_params(self) -> None:
        """Copy each linked (shared) parameter group from the active channel's
        current dialog state into every channel, so enabling a link takes
        effect immediately rather than only on the next channel switch."""
        if len(self.channels) < 2:
            return
        pd = self.parameters_dialog
        cur = self._capture_params()
        keys: list[str] = []
        if pd.link_box_checkbox.isChecked():
            keys += ["box"]
        if pd.link_mng_checkbox.isChecked():
            keys += ["mng", "mng_min", "mng_max"]
        if pd.link_calib_checkbox.isChecked():
            keys += ["spline_calibration", "spline_calibration_path"]
        if pd.link_camera_checkbox.isChecked():
            keys += [
                "camera",
                "baseline",
                "gain",
                "sensitivity",
                "qe",
                "pixelsize",
            ]
        for channel in self.channels:
            if not channel.params:
                continue
            for key in keys:
                if key in cur:
                    channel.params[key] = cur[key]

    def _channels_share_file(self) -> bool:
        """True when several channels were loaded from one file (so their
        output names need a channel suffix to avoid overwriting)."""
        return len(self.channels) > 1 and all(
            c.path == self.channels[0].path for c in self.channels
        )

    def channel_output_base(self) -> str:
        """Base path (no extension) for the active channel's output files,
        suffixed with the channel name when channels share one file, and
        with the region label while a split-FOV region is being fitted on
        its own (see ``_fit_independent``) - the regions all come from one
        movie, so they would otherwise overwrite each other."""
        base, _ = os.path.splitext(self.movie_path)
        if self._channels_share_file():
            name = self.channels[self.current_channel].name
            base = base + "_" + _sanitize_filename(name)
        if self._region_suffix:
            base = base + "_" + _sanitize_filename(self._region_suffix)
        return base

    def open_picks(self) -> None:
        """Open a file dialog to select a picks (from Picasso: Render)
        file to load."""
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open picks", directory=dir, filter="*.yaml"
        )
        if path:
            self.load_picks(path)

    def load_picks(self, path: str) -> None:
        """Load picks from a YAML file from Picasso: Render."""
        # ask for drift correction
        driftpath, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open drift file",
            directory=os.path.dirname(path),
            filter="*.txt",
        )
        drift = None
        if driftpath:
            try:
                drift = io.load_drift(driftpath)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not load drift file",
                    f"Drift file could not be loaded, error: {e}\n."
                    "No drift correction will be applied.",
                )
                drift = None
        picks, shape, _ = io.load_picks(path)
        if shape != "Circle":
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported shape",
                f"Only circle picks are supported, but got {shape}.",
            )
            return
        # convert
        self.identifications = localize.picks_to_identifications(
            picks,
            n_frames=lib.get_from_metadata(self.info, "Frames"),
            drift=drift,
        )
        self._clean_up_external_ids()

    def open_locs(self) -> None:
        """Open localizations for refitting data. Provide spot
        identifications."""
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open locs", directory=dir, filter="*.hdf5"
        )
        if path:
            self.load_locs(path)

    def load_locs(self, path: str) -> None:
        """Load localizations from a HDF5 file. Provide spot
        identifications."""
        try:
            locs, _ = io.load_locs(path)
            n_frames, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Input Dialog",
                "Enter number of frames around localization event:",
                10,
                min=0,
            )
            if not ok:
                return
        except io.NoMetadataFileError:
            return

        self.identifications = localize.locs_to_identifications(
            locs=locs,
            movie_info=self.info,
            n_frames=n_frames,
        )
        self._clean_up_external_ids()

    def open_identifications(self) -> None:
        """Open identifications previously saved as a HDF5 file."""
        if self.movie_path != []:
            dir = os.path.dirname(self.movie_path)
        else:
            dir = None
        path, exe = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open identifications",
            directory=dir,
            filter="*.hdf5",
        )
        if path:
            self.load_identifications(path)

    def load_identifications(self, path: str) -> None:
        """Load identifications from a HDF5 file."""
        try:
            identifications, info = io.load_identifications(
                path, qt_parent=self
            )
        except io.NoMetadataFileError:
            return
        except KeyError:
            return
        self.identifications = identifications
        box = lib.get_from_metadata(info, "Box Size")
        min_ng = lib.get_from_metadata(info, "Min. Net Gradient")
        temporal_median = lib.get_from_metadata(info, "Temporal Median Window")
        gaussian_sigma = lib.get_from_metadata(info, "Gaussian Filter Sigma")
        if box or min_ng:
            self.last_identification_info = {}
            if box is not None:
                self.last_identification_info["Box Size"] = box
                self.parameters_dialog.box_spinbox.setValue(box)
            if min_ng is not None:
                self.last_identification_info["Min. Net Gradient"] = min_ng
                if isinstance(min_ng, (list, tuple)):
                    # split-FOV identifications carry one threshold per
                    # region; the slider takes the reference region's and
                    # the rest go back onto the regions themselves (when
                    # they are still drawn)
                    self.parameters_dialog.mng_slider.setValue(int(min_ng[0]))
                    if len(min_ng) == len(self.view.rois):
                        self.view.roi_mngs = [int(_) for _ in min_ng]
                        self.parameters_dialog.update_roi_display()
                else:
                    self.parameters_dialog.mng_slider.setValue(min_ng)
        # restore the identification filters too, otherwise the loaded
        # identifications would immediately count as outdated
        self.parameters_dialog.temporal_median_checkbox.setChecked(
            bool(temporal_median)
        )
        if temporal_median:
            self.parameters_dialog.temporal_median_spinbox.setValue(
                temporal_median
            )
        self.parameters_dialog.gaussian_filter_spinbox.setValue(
            float(gaussian_sigma or 0.0)
        )
        self._clean_up_external_ids()

    def _clean_up_external_ids(self) -> None:
        if self.identifications is None:
            return
        # remove all identifications that are oob
        box = self.parameters["Box Size"]
        m_size = self.movie.shape
        r = int(box / 2)
        self.identifications = self.identifications[
            (self.identifications.y - r > 0)
            & (self.identifications.x - r > 0)
            & (self.identifications.x + r < m_size[0])
            & (self.identifications.y + r < m_size[1])
        ]
        # assign gui attributes
        self.locs = None
        self.locs_display = None
        self.loaded_picks = True
        self.last_identification_info = {
            **self.parameters,
            "ROI": self.view.rois,
            "Frame bounds": self.frame_range,
        }
        self.ready_for_fit = True
        self.draw_frame()
        self.status_bar.showMessage(
            f"Created a total of {len(self.identifications):,} "
            "identifications."
        )

    def prompt_info(self) -> tuple[dict, bool] | None:
        """Prompt for movie information."""
        info, save, ok = PromptInfoDialog.getMovieSpecs(self)
        if ok:
            return info, save

    def prompt_movie_info(
        self, partial_info: dict | None = None
    ) -> tuple[dict, bool] | None:
        """Prompt for movie metadata when it cannot be read from the
        file (fallback for .tif/.stk/.nd2 movies). Pre-filled with the
        dimensions that could still be read. Returns ``(info, save)`` or
        None if canceled."""
        info, save, ok = PromptMovieInfoDialog.getMovieSpecs(
            self, partial_info
        )
        if ok:
            return info, save

    def prompt_channel(self, channels: list[str]) -> str | None:
        """Prompt for channel selection for multi-channel movies
        (IMARIS .ims, Zeiss .czi, Leica .lif)."""
        channel, ok = PromptChannelDialog.getMovieSpecs(self, channels)
        if ok:
            return channel

    def previous_frame(self, step: int = 1) -> None:
        """Navigate backwards by ``step`` frames and display the result."""
        if self.movie is not None:
            if self.curr_frame_number > 0:
                self.set_frame(max(0, self.curr_frame_number - step))

    def next_frame(self, step: int = 1) -> None:
        """Navigate forwards by ``step`` frames and display the result."""
        if self.movie is not None:
            last_frame = self.info[0]["Frames"] - 1
            if self.curr_frame_number < last_frame:
                self.set_frame(min(last_frame, self.curr_frame_number + step))

    def first_frame(self) -> None:
        """Navigate to the first frame and display it."""
        if self.movie is not None:
            self.set_frame(0)

    def last_frame(self) -> None:
        """Navigate to the last frame and display it."""
        if self.movie is not None:
            self.set_frame(self.info[0]["Frames"] - 1)

    def to_frame(self) -> None:
        """Navigate to a specific frame and display it."""
        if self.movie is not None:
            frames = self.info[0]["Frames"]
            number, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Go to frame",
                "Frame number:",
                self.curr_frame_number + 1,
                1,
                frames,
            )
            if ok:
                self.set_frame(number - 1)

    def set_frame(self, number: int) -> None:
        """Set the current frame to the specified number."""
        self.curr_frame_number = number
        if self.contrast_dialog.auto_checkbox.isChecked():
            # the same frame the view shows, so the auto contrast matches
            # what is displayed when an identification filter is on
            frame = self.identification_movie()[number]
            self.contrast_dialog.change_contrast_silently(
                frame.min(), frame.max()
            )
        self.draw_frame()
        self.status_bar_frame_indicator.setText(
            "{:,}/{:,}".format(
                number + 1, lib.get_from_metadata(self.info, "Frames")
            )
        )
        # Keep the slider in sync without re-triggering set_frame.
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(number)
        self.frame_slider.blockSignals(False)

    def on_frame_slider_changed(self, value: int) -> None:
        """Navigate to the frame selected with the slider."""
        if self.movie is not None and value != self.curr_frame_number:
            self.set_frame(value)

    def _contrast_slider_bounds(self) -> tuple[float, float] | None:
        """The intensity range the contrast slider should span.

        There is no cheap movie-wide range to read: the raw dtype range is
        useless (camera counts fill a sliver of it, and the displayed
        frames are float32 whenever an identification filter is on), and
        the current frame's range would rescale the track on every frame
        step. So a handful of evenly spaced frames of the *displayed*
        movie are sampled and padded.
        """
        # the parameters dialog is built (and can fire) before the movie
        # attribute and the slider exist
        if getattr(self, "movie", None) is None:
            return None
        movie = self.identification_movie()
        if movie is None:
            return None
        n_frames = len(movie)
        if not n_frames:
            return None
        indices = np.unique(
            np.linspace(
                0, n_frames - 1, min(CONTRAST_SLIDER_SAMPLES, n_frames)
            ).astype(int)
        )
        try:
            lo = min(float(movie[int(i)].min()) for i in indices)
            hi = max(float(movie[int(i)].max()) for i in indices)
        except Exception:
            # a sampled frame could not be read (e.g. a truncated file);
            # fall back to the frame already on screen
            try:
                frame = movie[self.curr_frame_number]
            except Exception:
                return None
            lo, hi = float(frame.min()), float(frame.max())
        pad = CONTRAST_SLIDER_PADDING * max(hi - lo, 1.0)
        return lo - pad, hi + pad

    def _clamp_contrast_bounds(
        self, lo: float, hi: float
    ) -> tuple[float, float]:
        """Fit a candidate track into what the contrast spinboxes accept."""
        black_box = self.contrast_dialog.black_spinbox
        white_box = self.contrast_dialog.white_spinbox
        lo = max(lo, black_box.minimum())
        hi = min(hi, white_box.maximum())
        hi = max(hi, lo + 1.0, white_box.minimum())
        return lo, hi

    def update_contrast_slider_range(self) -> None:
        """Re-derive the contrast slider's track from the movie.

        Only called when the displayed intensities move onto a different
        scale (a movie loaded, a channel switched, an identification
        filter toggled). Browsing frames must not resize the track, or the
        handles would jump around; it grows there instead, in
        ``sync_contrast_slider``.
        """
        if getattr(self, "contrast_slider", None) is None:
            return
        bounds = self._contrast_slider_bounds()
        if bounds is None:
            return
        lo, hi = self._clamp_contrast_bounds(*bounds)
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setRange(lo, hi)
        self.contrast_slider.blockSignals(False)
        self.sync_contrast_slider()

    def sync_contrast_slider(self) -> None:
        """Move the contrast slider's handles onto the spinbox values,
        widening the track if a value falls outside it."""
        if getattr(self, "contrast_slider", None) is None:
            return
        black = self.contrast_dialog.black_spinbox.value()
        white = self.contrast_dialog.white_spinbox.value()
        lo, hi = self.contrast_slider.range()
        self.contrast_slider.blockSignals(True)
        try:
            if black < lo or white > hi:
                self.contrast_slider.setRange(
                    *self._clamp_contrast_bounds(
                        min(lo, black), max(hi, white)
                    )
                )
            self.contrast_slider.setValues(black, white)
        finally:
            self.contrast_slider.blockSignals(False)

    def on_contrast_slider_changed(self, black: float, white: float) -> None:
        """Apply a contrast dragged on the slider below the movie."""
        if getattr(self, "movie", None) is not None:
            self.contrast_dialog.set_contrast_from_slider(black, white)

    def draw_frame(self) -> None:
        """Draw the current frame - show the movie frame, apply
        contrast, add identifications and fit markers, if applicable."""
        if self.movie is None:
            return
        # rebuilding the scene can change scrollbar values and re-fire
        # valueChanged -> on_scroll -> draw_frame; guard against unbounded
        # re-entrant recursion.
        if self._drawing_frame:
            return
        self._drawing_frame = True
        try:
            self._draw_frame()
        finally:
            self._drawing_frame = False

    def _draw_frame(self) -> None:
        """Actual frame-drawing implementation, wrapped by ``draw_frame``
        with a re-entrancy guard."""
        if self.movie is not None:
            # show what the identification sees, so that the contrast and
            # the preview boxes agree with the min. net gradient being set
            frame = self.identification_movie()[self.curr_frame_number]
            frame = self.contrast_dialog.to_uint8(frame)
            height, width = frame.shape
            image = QtGui.QImage(
                frame.data,
                width,
                height,
                width,
                QtGui.QImage.Format.Format_Indexed8,
            )
            image.setColorTable(CMAP_GRAYSCALE)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.scene = Scene(self)
            self.scene.addPixmap(pixmap)
            # pin the scene rect to the image bounds so overlay items that
            # extend past the edge (e.g. the fixed-size scale bar text) do
            # not enlarge the scene and shift/re-center the view
            self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.view.setScene(self.scene)
            # draw the ROI rectangles (in scene/pixel coordinates)
            split_fov = self.view.split_fov_mode
            region_mngs = self.region_mngs()
            for i, ((y_min, x_min), (y_max, x_max)) in enumerate(
                self.view.rois
            ):
                if i == self.view.selected_roi:
                    color = QtGui.QColor("cyan")
                elif split_fov:
                    # split-FOV: highlight the reference channel (index 0)
                    color = (
                        QtGui.QColor("lime")
                        if i == 0
                        else QtGui.QColor("orange")
                    )
                else:
                    color = QtGui.QColor("blue")
                pen = QtGui.QPen(color)
                pen.setCosmetic(True)  # constant width regardless of zoom
                self.scene.addRect(
                    QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min),
                    pen,
                )
                if split_fov:
                    # label each region by its channel index (0 = reference)
                    # and the threshold it is identified with
                    label = localize.region_label(i)
                    if i < len(region_mngs):
                        label += f" ({region_mngs[i]:,})"
                    text = self.scene.addSimpleText(label)
                    text.setBrush(QtGui.QBrush(color))
                    text.setPos(float(x_min), float(y_min))
                    text.setFlag(
                        QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
                    )
            if self.draw_affine_pairing():
                pass
            elif self.ready_for_fit:
                box = (self.last_identification_info or {}).get(
                    "Box Size", self.parameters["Box Size"]
                )
                if not self._draw_linked_identifications(
                    self.curr_frame_number, box
                ):
                    identifications_frame = self.identifications[
                        self.identifications.frame == self.curr_frame_number
                    ]
                    self.draw_identifications(
                        identifications_frame, box, QtGui.QColor("yellow")
                    )
            else:
                if self.parameters_dialog.preview_checkbox.isChecked():
                    # scrubbing into a new temporal window costs one median
                    # (~0.1 s at 512x512, more for larger frames), and the
                    # link colors identify the other channels' frames on top
                    QtWidgets.QApplication.setOverrideCursor(
                        QtCore.Qt.CursorShape.WaitCursor
                    )
                    try:
                        identifications_frame = (
                            localize.identify_by_frame_number(
                                self.identification_movie(),
                                self.parameters["Min. Net Gradient"],
                                self.parameters["Box Size"],
                                self.curr_frame_number,
                                roi=self.identification_rois(),
                                frame_bounds=self.frame_range,
                            )
                        )
                        self.draw_preview_identifications(
                            identifications_frame,
                            self.parameters["Box Size"],
                        )
                    finally:
                        QtWidgets.QApplication.restoreOverrideCursor()
                else:
                    self.status_bar.showMessage("")
            locs_frame = self._current_frame_locs()
            if locs_frame is not None:
                for _, loc in locs_frame.iterrows():
                    marker = FitMarker(loc["x"] + 0.5, loc["y"] + 0.5, 1)
                    marker.setToolTip(format_hover_tooltip(loc))
                    self.scene.addItem(marker)
            self.draw_scalebar()

    def set_affine_pairing(self, qc: dict) -> None:
        """Keep the bead pairing of an affine calibration so it can be drawn
        over the two bead images (see :meth:`draw_affine_pairing`).

        ``qc`` is what ``localize.fit_lateral_transform`` returns: every
        refined detection in each image plus the indices of the matched
        ones, pair ``k`` being ``(idx_ref[k], idx_target[k])``.
        """

        def side(beads, matched) -> dict:
            # pair id per detection; -1 for the ones that stayed unmatched
            beads = np.asarray(beads, dtype=float)
            pair_id = np.full(len(beads), -1, dtype=int)
            matched = np.asarray(matched, dtype=int)
            pair_id[matched] = np.arange(len(matched))
            return {"beads": beads, "pair_id": pair_id}

        self.affine_pairing = {
            "ref": side(qc["beads_ref"], qc["idx_ref"]),
            "target": side(qc["beads_target"], qc["idx_target"]),
            "paths": {
                "ref": _normalized_path(qc.get("ref_path", "")),
                "target": _normalized_path(qc.get("target_path", "")),
            },
            "n_pairs": int(qc["n_pairs"]),
            "box": int(qc.get("box", self.parameters["Box Size"])),
            "transform_type": qc.get("transform_type", "astigmatism"),
        }

    def draw_affine_pairing(self) -> bool:
        """Draw the last affine calibration's bead pairing as color-coded
        identification boxes over the displayed image.

        A bead and the bead it was matched with carry the same color in the
        reference and in the target image, so switching between the two
        (the calibration dialog's "Show" buttons) shows which bead went with
        which; detections that stayed unmatched are gray. This is the same
        reading as the cross-channel link colors
        (:meth:`_draw_linked_identifications`), and it uses the same palette.

        Returns True when it drew, i.e. when a calibration has run and the
        displayed movie is one of its two bead images - False to leave the
        normal identification boxes to the caller.
        """
        pairing = self.affine_pairing
        if pairing is None:
            return False
        current = _normalized_path(self.movie_path or "")
        for name in ("ref", "target"):
            if current and current == pairing["paths"][name]:
                side = pairing[name]
                break
        else:
            return False

        box = pairing["box"]
        box_half = int(box / 2)
        n_pairs = pairing["n_pairs"]
        other = "target" if side is pairing["ref"] else "reference"
        n_unmatched = int(np.count_nonzero(side["pair_id"] < 0))
        for (row, col), pair_id in zip(side["beads"], side["pair_id"]):
            matched = pair_id >= 0
            color = (
                LINK_COLORS[int(pair_id) % len(LINK_COLORS)]
                if matched
                else LINK_UNMATCHED_COLOR
            )
            item = self.scene.addRect(
                col - box_half, row - box_half, box, box, color
            )
            item.setToolTip(
                (
                    f"bead pair {int(pair_id) + 1} of {n_pairs}\n"
                    f"same color in the {other} image"
                    if matched
                    else f"not paired with any bead in the {other} image"
                )
                + f"\nx: {col:.6g}\ny: {row:.6g}"
            )
        self.status_bar.showMessage(
            f"{pairing['transform_type'].capitalize()} calibration: "
            f"{n_pairs:,} bead pairs shown"
            + (f", {n_unmatched:,} unpaired (gray)" if n_unmatched else "")
            + "."
        )
        return True

    def draw_identifications(
        self,
        identifications: pd.DataFrame,
        box: int,
        color: QtGui.QColor,
    ) -> None:
        """Draw identification boxes in the scene. Hovering a box shows
        the properties of the fitted localization inside it (or of the
        identification itself, before fitting)."""
        box_half = int(box / 2)
        locs_frame = self._current_frame_locs()
        for _, identification in identifications.iterrows():
            x = identification["x"]
            y = identification["y"]
            item = self.scene.addRect(
                x - box_half, y - box_half, box, box, color
            )
            loc = self._loc_near(locs_frame, x, y, box_half)
            item.setToolTip(
                format_hover_tooltip(identification if loc is None else loc)
            )

    def draw_preview_identifications(
        self, identifications_frame: pd.DataFrame, box: int
    ) -> None:
        """Draw the identification preview for the current frame and report
        what it found in the status bar.

        With 'Link colors' on the boxes are color-coded by their cross-channel
        link, exactly as they are after an identification run - the pairing is
        made from the detections the preview itself produces, so the channels
        do not have to be identified first. Plain red boxes otherwise, and
        whenever the channels cannot be linked (yet).
        """
        self._link_box_counts = None
        self._preview_link_ids = self._preview_link_identifications(
            identifications_frame
        )
        try:
            linked = self._draw_linked_identifications(
                self.curr_frame_number, box
            )
        finally:
            self._preview_link_ids = None
        if not linked:
            self.draw_identifications(
                identifications_frame, box, QtGui.QColor("red")
            )
        message = (
            f"Found {len(identifications_frame):,} spots in current frame"
        )
        counts = self._link_box_counts
        if counts is not None:
            n_linked, n_total = counts
            where = "regions" if self.view.split_fov_mode else "channels"
            message += (
                f", {n_linked:,} of {n_total:,} linked across all {where}"
            )
        self.status_bar.showMessage(message + ".")

    def _preview_link_identifications(
        self, identifications_frame: pd.DataFrame
    ) -> list | None:
        """This frame's detections in every channel, for the link colors of
        the identification preview. Returns None when there is nothing to
        link, which keeps the preview on its plain boxes.

        Split-FOV data has every channel in the displayed frame, so the
        preview has already searched all of them and its one table is all
        that is needed. Separate channel movies show one channel at a time,
        so the frames of the other channels are identified here as well -
        with their own settings, exactly as the preview identifies the
        displayed one (see :meth:`channel_frame_identifications`).
        """
        pdialog = self.parameters_dialog
        if not getattr(pdialog, "link_colors_checkbox", None):
            return None
        if not pdialog.link_colors_checkbox.isChecked():
            return None
        if self.identify_mode() == IDENTIFY_MODE_SUM:
            # the preview searches the summed channels, so every detection is
            # a cross-channel spot already - there is nothing to pair
            return None
        if self.view.split_fov_mode:
            if len(self.view.rois) < 2:
                return None
            return [identifications_frame]
        if len(self.channels) < 2:
            return None
        return [
            (
                identifications_frame
                if c == self.current_channel
                else self.channel_frame_identifications(c)
            )
            for c in range(len(self.channels))
        ]

    def channel_frame_identifications(
        self, channel: int
    ) -> pd.DataFrame | None:
        """Identify the current frame in a channel that is not on screen, the
        way the preview identifies the displayed one.

        Cached per channel and per everything the detections depend on: the
        preview redraws on every scroll, every parameter change and every
        mouse move over the frame slider, and each of these calls is a full
        frame identification of another movie.
        """
        movie = self.channels[channel].movie
        if movie is None:
            return None
        frame_number = self.curr_frame_number
        if frame_number >= len(movie):
            # the channels can differ in length; a frame the movie does not
            # have simply has no detections to link against
            return None
        parameters = self.channel_parameters(channel)
        rois = self.identification_rois()
        key = (
            frame_number,
            parameters["Box Size"],
            str(parameters["Min. Net Gradient"]),
            parameters["Temporal Median Window"],
            parameters["Gaussian Filter Sigma"],
            tuple(np.asarray(rois).ravel().tolist()) if rois else None,
            None if self.frame_range is None else str(self.frame_range),
        )
        cached = self._preview_ids_cache.get(channel)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            identifications = localize.identify_by_frame_number(
                self.channel_identification_movie(channel),
                parameters["Min. Net Gradient"],
                parameters["Box Size"],
                frame_number,
                roi=rois,
                frame_bounds=self.frame_range,
            )
        except Exception:
            # another channel's movie must never break the display; without
            # its detections the link colors simply fall back
            return None
        self._preview_ids_cache[channel] = (key, identifications)
        return identifications

    def channel_parameters(self, channel: int) -> dict:
        """The identification settings of a channel that is not on screen.

        The dialog only ever holds the active channel's values, so the others
        come from the snapshot taken when they were last displayed - except
        for the settings whose 'Same across channels' box is ticked, which are
        the dialog's current ones by definition. The temporal median needs a
        stack, so it is switched off for a channel whose movie is too short
        for it, exactly as ``parameters`` does for the displayed one.
        """
        parameters = dict(self.parameters)
        if channel == self.current_channel:
            return parameters
        pdialog = self.parameters_dialog
        params = self.channels[channel].params or {}
        if params:
            if not pdialog.link_box_checkbox.isChecked():
                parameters["Box Size"] = params.get(
                    "box", parameters["Box Size"]
                )
            if not pdialog.link_mng_checkbox.isChecked():
                parameters["Min. Net Gradient"] = params.get(
                    "mng", parameters["Min. Net Gradient"]
                )
            # the filters are per channel: they are always restored from the
            # snapshot on a channel switch (see ``_apply_params``)
            parameters["Temporal Median Window"] = (
                params.get(
                    "temporal_median",
                    DEFAULT_PARAMETERS["Temporal Median Window"],
                )
                if params.get("temporal_median_on", False)
                else 0
            )
            parameters["Gaussian Filter Sigma"] = params.get(
                "gaussian_filter_sigma",
                DEFAULT_PARAMETERS["Gaussian Filter Sigma"],
            )
        movie = self.channels[channel].movie
        if movie is not None and len(movie) < _TEMPORAL_MEDIAN_MIN_FRAMES:
            parameters["Temporal Median Window"] = 0
        return parameters

    def channel_identification_movie(self, channel: int):
        """The identification-filtered movie of a channel that is not on
        screen, built like :meth:`identification_movie` builds the displayed
        one but from that channel's own settings.

        Only the link-color preview needs this; the filter stacks are lazy
        views, so they are cached per channel (and dropped with the channels
        themselves) rather than rebuilt on every redraw.
        """
        if channel == self.current_channel:
            return self.identification_movie()
        movie = self.channels[channel].movie
        if movie is None:
            return None
        parameters = self.channel_parameters(channel)
        movie, cache = _filtered_identification_movie(
            movie,
            parameters["Temporal Median Window"],
            parameters["Gaussian Filter Sigma"],
            self._channel_filter_cache.get(channel, [None, None]),
        )
        self._channel_filter_cache[channel] = cache
        return movie

    def drop_preview_link_cache(self) -> None:
        """Forget the off-screen channels' preview detections and filter
        stacks. Called wherever they would no longer describe the channels
        they were built from (channels loaded or closed, a channel switched
        in, an identification run)."""
        self._preview_ids_cache = {}
        self._channel_filter_cache = {}

    def _current_frame_locs(self) -> pd.DataFrame | None:
        """Fitted localizations in the currently displayed frame, or
        None if no localizations are available."""
        if self.locs_display is None:
            return None
        return self.locs_display[
            self.locs_display.frame == self.curr_frame_number
        ]

    @staticmethod
    def _loc_near(
        locs_frame: pd.DataFrame | None,
        x: float,
        y: float,
        radius: float,
    ) -> pd.Series | None:
        """The localization closest to (x, y) that lies within
        ``radius`` pixels of it in both coordinates, or None."""
        if locs_frame is None or not len(locs_frame):
            return None
        dx = locs_frame["x"] - x
        dy = locs_frame["y"] - y
        inside = (dx.abs() <= radius) & (dy.abs() <= radius)
        if not inside.any():
            return None
        d2 = dx[inside] ** 2 + dy[inside] ** 2
        return locs_frame.loc[d2.idxmin()]

    def _draw_linked_identifications(
        self, frame_number: int, box: int
    ) -> bool:
        """Draw this frame's identification boxes color-coded by cross-channel
        link, when 'Link colors' is on and a multichannel / split-FOV spline
        calibration is loaded.

        Spots paired across channels (matched to the reference channel via the
        calibration's inter-channel transform, as the signal re-registration
        does) share a color; unmatched spots are gray. Returns True if it
        handled the drawing, False to fall back to plain single-color boxes.

        Drawn from the identifications of the last Identify run, or - while the
        preview is drawing (``_preview_link_ids``) - from the detections the
        preview has just made in this frame, so the links show up before
        anything has been identified.
        """
        pdialog = self.parameters_dialog
        if not getattr(pdialog, "link_colors_checkbox", None):
            return False
        if not pdialog.link_colors_checkbox.isChecked():
            return False
        if self.sum_identifications is not None:
            # the detections come from the channel sum, so every one of them
            # is a cross-channel spot already - there is nothing to pair
            return False
        cal = pdialog.link_calibration()
        n_channels = int(cal.get("n_channels", 0))
        if n_channels >= 2:
            # a loaded calibration always provides the registration - the
            # colors then show exactly what the fit will pair, so a bad
            # registration is visible and can be re-registered deliberately
            # (Postprocess > re-register from signal)
            cal = self._link_calibration_for_mode(cal, n_channels)
        else:
            # nothing loaded: register the channels from the identifications
            cal = self._estimated_link_calibration(box)
        if cal is None:
            return False
        n_channels = int(cal["n_channels"])
        tol = 1.5 * float(box)
        try:
            if cal.get("split_fov"):
                boxes = self._linked_boxes_split_fov(
                    cal, n_channels, frame_number, tol
                )
            else:
                boxes = self._linked_boxes_multichannel(
                    cal, n_channels, frame_number, tol
                )
        except Exception:
            # a malformed calibration must never break the viewer; fall back
            return False
        if boxes is None:
            return False
        self._link_box_counts = (
            sum(1 for *_, color in boxes if color != LINK_UNMATCHED_COLOR),
            len(boxes),
        )
        box_half = int(box / 2)
        locs_frame = self._current_frame_locs()
        for x, y, color in boxes:
            item = self.scene.addRect(
                x - box_half, y - box_half, box, box, color
            )
            loc = self._loc_near(locs_frame, x, y, box_half)
            if loc is not None:
                item.setToolTip(format_hover_tooltip(loc))
            else:
                item.setToolTip(f"x: {x:.6g}\ny: {y:.6g}")
        return True

    def _link_calibration_for_mode(
        self, cal: dict, n_channels: int
    ) -> dict | None:
        """The loaded calibration's registration, adapted to how the data are
        currently laid out (split-FOV regions vs. separate channels).

        The calibration is *always* the source of the inter-channel transform
        when one is loaded - the link colors then show exactly the pairing the
        fit will use, so a stale registration shows up as gray boxes and can be
        re-registered on purpose rather than being silently papered over. Only
        the placement is adapted when the layout differs from the calibration's:

        * split-FOV mode with a separate-movie calibration: both its channels
          start at the frame origin, so its transforms already *are* the
          region-local registration; they are placed at the drawn ROIs.
        * separate channels with a split-FOV calibration: the region-local
          affines *are* the inter-channel registration, so they apply directly
          to the whole frame of each movie.

        Returns None if the layout cannot be mapped onto the calibration's
        channels at all (e.g. a different number of ROIs).
        """
        split_fov_cal = bool(cal.get("split_fov"))
        if self.view.split_fov_mode == split_fov_cal:
            return cal
        if self.view.split_fov_mode:
            transforms = cal.get("channel_transforms")
            if not transforms or len(self.view.rois) != n_channels:
                return None
            return {
                "n_channels": n_channels,
                "split_fov": True,
                "regions": [list(map(list, r)) for r in self.view.rois],
                "channel_registration": list(transforms[:n_channels]),
            }
        if len(self.channels) < 2:
            return cal  # single movie: keep the calibration's own regions
        affines = cal.get("channel_registration")
        if affines is None:
            regions = cal.get("regions")
            transforms = cal.get("channel_transforms")
            if not regions or not transforms:
                return None
            affines = [
                a.to_dict()
                for a in localize.decompose_region_transforms(
                    [_normalize_rect(r) for r in regions], transforms
                )
            ]
        return {
            "n_channels": n_channels,
            "channel_transforms": list(affines[:n_channels]),
        }

    def link_identifications(self, channel: int = 0) -> pd.DataFrame | None:
        """The detections the link colors pair ``channel`` from.

        While the identification preview draws, these are the current frame's
        detections it has just made (see ``_preview_link_ids``), so the links
        are shown before anything has been identified; otherwise they are the
        identifications of the last Identify run. Split-FOV data keeps every
        region's detections in one table, so ``channel`` is ignored there and
        the caller splits the table by region.
        """
        preview = self._preview_link_ids
        if preview is not None:
            if self.view.split_fov_mode:
                return preview[0]
            return preview[channel] if channel < len(preview) else None
        if self.view.split_fov_mode or channel == self.current_channel:
            return self.identifications
        if channel < len(self.channels):
            # the active channel's live detections are not mirrored back into
            # ``self.channels`` until the channel is switched
            return self.channels[channel].identifications
        return None

    def _channel_identification_inputs(
        self, preview: bool = False
    ) -> tuple[list | None, list | None]:
        """The per-channel detections the channels can be registered from,
        and the regions they sit in.

        Returns ``(ids_per_channel, regions)`` in the order the registration
        expects (reference first), with ``regions`` None for separate channel
        movies, or ``(None, None)`` when there is nothing to register. Shared
        by the link-color preview and the channel-sum identification, so both
        register the channels from exactly the same detections.

        With ``preview`` the preview's current-frame detections stand in for
        the channels while it is drawing (see :meth:`link_identifications`);
        the channel sum registers off the full identifications only.
        """
        ids = (
            self.link_identifications
            if preview
            else lambda c: (
                self.identifications
                if self.view.split_fov_mode or c == self.current_channel
                else self.channels[c].identifications
            )
        )
        if self.view.split_fov_mode:
            regions = [list(map(list, r)) for r in self.view.rois]
            if len(regions) < 2 or ids(0) is None:
                return None, None
            # split-FOV: one table holds every region's detections
            rects = [_normalize_rect(r) for r in regions]
            return [
                self._identifications_in_region(rect, ids(0)) for rect in rects
            ], regions
        if len(self.channels) < 2:
            return None, None
        return [ids(c) for c in range(len(self.channels))], None

    def _frame_shape_for_registration(self) -> tuple[int, int] | None:
        """``(height, width)`` used to seed the mirror orientations when the
        channels are separate movies; None for split-FOV, where the regions
        supply the placement instead."""
        if self.view.split_fov_mode or self.movie is None:
            return None
        return (int(self.movie.shape[1]), int(self.movie.shape[2]))

    def _estimated_link_calibration(self, box: int) -> dict | None:
        """A calibration-shaped dict whose inter-channel transforms are
        estimated from the identifications themselves, for link coloring
        without a loaded spline calibration.

        The transforms come from
        :func:`spline.estimate_transforms_from_identifications`, which searches
        the mirror orientations - so a flipped channel (image splitter, mirrored
        quadrant) links just as it does with a calibration. Channels that cannot
        be registered fall back to the identity, i.e. the plain overlay.
        Estimating is not free, so the result is cached until the detections,
        the box size or the ROIs change. Returns None if there is nothing to
        link.

        In the preview the detections are the current frame's alone, so the
        registration is estimated from that one frame - enough for a dense
        frame, and the channels simply stay gray (unregistered) where it is
        not. The cached result is therefore also keyed by the frame.
        """
        split_fov = self.view.split_fov_mode
        preview = self._preview_link_ids is not None
        ids_per_channel, regions = self._channel_identification_inputs(
            preview=preview
        )
        if ids_per_channel is None:
            return None
        n_channels = len(ids_per_channel)
        key = (
            split_fov,
            int(box),
            n_channels,
            (
                tuple(np.asarray(regions).ravel().tolist())
                if regions is not None
                else None
            ),
            tuple(0 if ids is None else len(ids) for ids in ids_per_channel),
            self.curr_frame_number if preview else None,
        )
        cached = getattr(self, "_link_cal_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        identity = transforms_mod.identity().to_dict()
        if split_fov:
            region_rects = [_normalize_rect(r) for r in regions]
        try:
            transforms = spline.estimate_transforms_from_identifications(
                ids_per_channel,
                box,
                regions=regions,
                frame_shape=self._frame_shape_for_registration(),
            )
        except Exception:
            transforms = None
        if split_fov:
            # region-local affines: the ROI placement is stripped out, so the
            # boxes follow the ROIs if the user nudges them
            if transforms is None:
                affines = [identity for _ in range(n_channels)]
            else:
                affines = [
                    (
                        identity
                        if t is None
                        else localize.decompose_region_transforms(
                            [region_rects[0], region_rects[c]],
                            [transforms[0], t],
                        )[1].to_dict()
                    )
                    for c, t in enumerate(transforms)
                ]
            cal = {
                "n_channels": n_channels,
                "split_fov": True,
                "regions": regions,
                "channel_registration": affines,
            }
        else:
            cal = {
                "n_channels": n_channels,
                "channel_transforms": [
                    (
                        identity
                        if transforms is None or transforms[c] is None
                        else transforms[c].to_dict()
                    )
                    for c in range(n_channels)
                ],
            }
        self._link_cal_cache = (key, cal)
        return cal

    def _identifications_in_region(
        self, rect: tuple, ids: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """The identifications inside a ``((y_min, x_min), (y_max, x_max))``
        region (split-FOV: one region per channel). ``ids`` defaults to the
        window's own identifications."""
        if ids is None:
            ids = self.identifications
        if ids is None or len(ids) == 0:
            return None
        (y0, x0), (y1, x1) = rect
        x = np.asarray(ids["x"], dtype=float)
        y = np.asarray(ids["y"], dtype=float)
        return ids[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)]

    def _linked_boxes_split_fov(
        self, cal: dict, n_channels: int, frame_number: int, tol: float
    ) -> list | None:
        """Color-coded boxes for a split-FOV calibration: every region lives in
        one frame, so paired boxes across regions get the same color. Returns a
        list of ``(x, y, QColor)`` for all this-frame spots, or None to fall
        back."""
        ids = self.link_identifications()
        if ids is None or len(ids) == 0:
            return None
        # place the channels at the drawn ROIs when they match the channel
        # count (reference first), else the calibration's stored regions
        if self.view.split_fov_mode and len(self.view.rois) == n_channels:
            regions = [list(map(list, r)) for r in self.view.rois]
        else:
            regions = cal.get("regions")
        if not regions or len(regions) != n_channels:
            return None
        region_rects = [_normalize_rect(r) for r in regions]
        affines = cal.get("channel_registration")
        if affines is None:
            affines = [
                a.to_dict()
                for a in localize.decompose_region_transforms(
                    cal["regions"], cal["channel_transforms"]
                )
            ]
        transforms = localize.compose_region_transforms(region_rects, affines)
        m = np.asarray(ids["frame"]) == frame_number
        xy = np.column_stack(
            [np.asarray(ids["x"])[m], np.asarray(ids["y"])[m]]
        ).astype(float)
        if len(xy) == 0:
            return []
        # assign each spot to the region that contains it (first match wins)
        region_of = np.full(len(xy), -1, dtype=int)
        for ci, ((y0, x0), (y1, x1)) in enumerate(region_rects):
            inside = (
                (xy[:, 0] >= x0)
                & (xy[:, 0] < x1)
                & (xy[:, 1] >= y0)
                & (xy[:, 1] < y1)
            )
            region_of[(region_of < 0) & inside] = ci
        ref_local = np.where(region_of == 0)[0]
        ref_xy = xy[ref_local]
        colors = [LINK_UNMATCHED_COLOR] * len(xy)
        # A reference spot links only if matched in EVERY other region/channel
        # (the bead is found in all channels). Count per reference spot, then
        # color; spots missing from any channel stay gray.
        n_ref = len(ref_local)
        match_count = np.zeros(n_ref, dtype=int)
        per_channel: list = []
        n_checked = 0
        for c in range(1, n_channels):
            chan_local = np.where(region_of == c)[0]
            n_checked += 1
            if len(chan_local) == 0 or n_ref == 0:
                per_channel.append((chan_local, {}))
                continue
            pred = transforms_mod.from_dict(transforms[c]).apply(ref_xy)
            matches = _nearest_unique_match(pred, xy[chan_local], tol)
            per_channel.append((chan_local, matches))
            for rk in set(matches.values()):
                match_count[rk] += 1
        complete = (
            match_count == n_checked
            if n_checked
            else np.zeros(n_ref, dtype=bool)
        )
        # color non-reference spots that pair with a fully-linked ref spot
        for chan_local, matches in per_channel:
            for tj, rk in matches.items():
                if complete[rk]:
                    colors[chan_local[tj]] = LINK_COLORS[rk % len(LINK_COLORS)]
        # a reference spot is colored only if it links across all channels
        for rk, gi in enumerate(ref_local):
            if complete[rk]:
                colors[gi] = LINK_COLORS[rk % len(LINK_COLORS)]
        return [(xy[i, 0], xy[i, 1], colors[i]) for i in range(len(xy))]

    def _linked_boxes_multichannel(
        self, cal: dict, n_channels: int, frame_number: int, tol: float
    ) -> list | None:
        """Color-coded boxes for a multichannel calibration (separate movies /
        one multichannel file): only the current channel is on screen, so a spot
        keeps its group color as the user switches channels. Returns a list of
        ``(x, y, QColor)`` for the current channel's this-frame spots, or None to
        fall back."""
        transforms = cal.get("channel_transforms")
        if not transforms or len(transforms) < n_channels:
            return None
        if len(self.channels) < 2:
            return None
        reference_ids = self.link_identifications(0)
        if reference_ids is None:
            return None

        def frame_xy(ids) -> np.ndarray:
            if ids is None or len(ids) == 0:
                return np.empty((0, 2), dtype=float)
            m = np.asarray(ids["frame"]) == frame_number
            return np.column_stack(
                [np.asarray(ids["x"])[m], np.asarray(ids["y"])[m]]
            ).astype(float)

        ref_xy = frame_xy(reference_ids)

        # A reference spot counts as linked only if it is matched in EVERY
        # other channel (i.e. the bead is found in all channels). Count, per
        # reference spot, the channels it matches; "complete" requires a match
        # in each channel checked. Spots missing from any channel stay gray.
        n_ref = len(ref_xy)
        match_count = np.zeros(n_ref, dtype=int)
        n_checked = 0
        for c2 in range(1, n_channels):
            if c2 >= len(self.channels):
                continue
            n_checked += 1
            chan_xy = frame_xy(self.link_identifications(c2))
            if n_ref == 0 or len(chan_xy) == 0:
                continue
            pred = transforms_mod.from_dict(transforms[c2]).apply(ref_xy)
            for rk in set(_nearest_unique_match(pred, chan_xy, tol).values()):
                match_count[rk] += 1
        complete = (
            match_count == n_checked
            if n_checked
            else np.zeros(n_ref, dtype=bool)
        )

        c = self.current_channel
        if c == 0:
            # color reference spots only if they pair in ALL other channels
            return [
                (
                    ref_xy[rk, 0],
                    ref_xy[rk, 1],
                    (
                        LINK_COLORS[rk % len(LINK_COLORS)]
                        if complete[rk]
                        else LINK_UNMATCHED_COLOR
                    ),
                )
                for rk in range(n_ref)
            ]
        # a non-reference channel: color its detections by the reference spot
        # they pair with, but only when that reference spot links across ALL
        # channels; otherwise gray (matches that spot's box in channel 0)
        cur_xy = frame_xy(self.link_identifications(c))
        if len(cur_xy) == 0:
            return []
        matches = {}
        if n_ref:
            pred = transforms_mod.from_dict(transforms[c]).apply(ref_xy)
            matches = _nearest_unique_match(pred, cur_xy, tol)
        return [
            (
                cur_xy[j, 0],
                cur_xy[j, 1],
                (
                    LINK_COLORS[matches[j] % len(LINK_COLORS)]
                    if (j in matches and complete[matches[j]])
                    else LINK_UNMATCHED_COLOR
                ),
            )
            for j in range(len(cur_xy))
        ]

    def draw_scalebar(self) -> None:
        """Draw a scale bar if the option is checked."""
        if not self.scalebar_action.isChecked():
            return

        scene_pixelsize = self.parameters_dialog.pixelsize.value()

        # length (nm) - set optimal size (~1/8 of image width)
        rect = self.view.viewport().rect()
        visible_scene_rect = self.view.mapToScene(rect).boundingRect()
        width = visible_scene_rect.width()
        # the view may not be laid out yet (e.g. slow/network loads), in
        # which case ``width`` is 0 and there is nothing to draw
        if width <= 0:
            return
        width_nm = width * scene_pixelsize
        optimal_scalebar = width_nm / 8

        # approximate to the nearest thousands, hundreds, tens or ones
        if optimal_scalebar > 10_000:
            scalebar = 10_000
        elif optimal_scalebar > 1_000:
            scalebar = int(1_000 * round(optimal_scalebar / 1_000))
        elif optimal_scalebar > 100:
            scalebar = int(100 * round(optimal_scalebar / 100))
        elif optimal_scalebar > 10:
            scalebar = int(10 * round(optimal_scalebar / 10))
        else:
            scalebar = int(round(optimal_scalebar))

        # position against the viewport (the drawable area) rather than
        # the whole view, so the bar is not covered by the scrollbar
        # gutters (opaque on Windows, overlaid on macOS)
        viewport_width = self.view.viewport().width()
        viewport_height = self.view.viewport().height()
        length_displaypxl = int(
            round(viewport_width * (scalebar / scene_pixelsize) / width)
        )
        # when zoomed in far enough the scale bar rounds down to 0 nm /
        # 0 display pixels; skip drawing to avoid a division by zero below
        if scalebar <= 0 or length_displaypxl <= 0:
            return
        height_displaypxl = 10

        # draw a rectangle
        x = viewport_width - length_displaypxl - 40
        y = viewport_height - height_displaypxl - 20
        pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        brush = QtGui.QBrush(QtGui.QColor("white"))
        polygon = self.view.mapToScene(
            x,
            y,
            length_displaypxl,
            height_displaypxl,
        )
        x_scene = polygon.boundingRect().x()
        y_scene = polygon.boundingRect().y()
        length_scene = polygon.boundingRect().width()
        height_scene = polygon.boundingRect().height()
        self.scene.addRect(
            x_scene,
            y_scene,
            length_scene,
            height_scene,
            pen,
            brush,
        )

        # add scale bar text
        font = QtGui.QFont()
        font.setPointSize(20)
        text_item = self.scene.addText(f"{scalebar} nm", font)
        text_item.setDefaultTextColor(QtGui.QColor("white"))
        # scene units per device pixel (uniform zoom, but keep x/y
        # separate to be safe)
        scene_per_px_x = length_scene / length_displaypxl
        scene_per_px_y = height_scene / height_displaypxl
        # position the text centered above the scale bar, with a fixed
        # device-pixel gap so the spacing looks the same at every zoom
        text_rect = text_item.boundingRect()
        text_width = text_rect.width() * scene_per_px_x
        text_height = text_rect.height() * scene_per_px_y
        gap = 8 * scene_per_px_y
        text_x = x_scene + (length_scene - text_width) / 2
        text_y = y_scene - gap - text_height
        text_item.setPos(text_x, text_y)
        text_item.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,  # noqa: E501
            True,
        )

    def region_mngs(self) -> list[int]:
        """Per-region minimum net gradients in split-FOV mode, kept in step
        with ``view.rois``.

        The regions are separate channels (different dyes, different
        optical paths), so a single threshold rarely suits all of them.
        Regions added since the last call inherit the value the slider last
        settled on; removed ones drop out. Returns an empty list whenever
        split-FOV mode is off or no region is drawn - that is, whenever the
        single shared threshold on the slider applies.
        """
        # ``parameters`` is also read from a partially built window, where
        # the single shared threshold stands on its own
        try:
            view = self.view
            if not view.split_fov_mode or not view.rois:
                view.roi_mngs = []
                return []
            default = self.parameters_dialog._last_mng
        except (AttributeError, RuntimeError):
            return []
        mngs = [int(_) for _ in view.roi_mngs[: len(view.rois)]]
        mngs += [int(default)] * (len(view.rois) - len(mngs))
        view.roi_mngs = mngs
        return mngs

    def region_params(self) -> list[dict]:
        """Per-region fit settings in split-FOV mode, kept in step with
        ``view.rois``.

        Fitted separately, each region is a channel of its own: its own
        model and optimizer, its own PSF calibration, its own z-calibration.
        This is the split-FOV counterpart of the per-channel ``Channel.params``
        (see ``_capture_params``), restricted to what the fit uses - one movie
        holds every region, so the identification and camera settings stay
        shared. Regions added since the last call inherit the settings now on
        the dialog; removed ones drop out. Empty whenever split-FOV mode is
        off or no region is drawn.
        """
        try:
            view = self.view
            if not view.split_fov_mode or not view.rois:
                view.roi_params = []
                return []
            default = self._capture_region_fit_params()
        except (AttributeError, RuntimeError):
            # read from a partially built window
            return []
        params = list(view.roi_params[: len(view.rois)])
        params += [dict(default) for _ in range(len(view.rois) - len(params))]
        view.roi_params = params
        return params

    def _capture_region_fit_params(self) -> dict:
        """The fit settings now on the dialog, as one region's parameters."""
        pd = self.parameters_dialog
        return {
            "fit_model": pd.fit_model.currentIndex(),
            "fit_optimizer": pd.fit_optimizer.currentIndex(),
            "convergence": pd.convergence_criterion.value(),
            "max_it": pd.max_it.value(),
            "use_gpu": pd.gpu_checkbox.isChecked(),
            "spline_calibration": pd.spline_calibration,
            "spline_calibration_path": pd.spline_calibration_path,
            "z_calibration": pd.z_calibration,
            "z_calibration_path": pd.z_calibration_path,
            "z_calib_label": pd.z_calib_label.text(),
            "fit_z": pd.fit_z_checkbox.isChecked(),
            "fit_z_enabled": pd.fit_z_checkbox.isEnabled(),
        }

    def _apply_region_fit_params(self, params: dict) -> None:
        """Show one region's fit settings on the dialog.

        ``_switching_channel`` is set for the same reason a channel switch
        sets it: the value changes must not be read as the user editing the
        parameters, which would drop the identifications (see
        ``on_parameters_changed``)."""
        if not params:
            return
        pd = self.parameters_dialog
        self._switching_channel = True
        try:
            # the model first: its handler repopulates the optimizer list
            pd.fit_model.setCurrentIndex(params["fit_model"])
            pd.fit_optimizer.setCurrentIndex(params["fit_optimizer"])
            pd.convergence_criterion.setValue(params["convergence"])
            pd.max_it.setValue(params["max_it"])
            pd.gpu_checkbox.setChecked(params["use_gpu"])
            pd._set_spline_state(
                params["spline_calibration"],
                params["spline_calibration_path"],
            )
            pd.z_calibration = params["z_calibration"]
            pd.z_calibration_path = params["z_calibration_path"]
            pd.z_calib_label.setText(params["z_calib_label"])
            pd.fit_z_checkbox.setEnabled(params["fit_z_enabled"])
            pd.fit_z_checkbox.setChecked(params["fit_z"])
        finally:
            self._switching_channel = False

    def sync_region_fit_params(self) -> None:
        """Swap the fit settings over when another split-FOV region is
        selected: what is on the dialog belongs to the region that was
        selected before, and the newly selected region's settings take its
        place.

        A no-op unless the regions are actually fitted separately - in the
        joint fit one calibration describes every region, so the dialog holds
        one set of settings for all of them, exactly as before.
        """
        if self.fit_mode() != FIT_MODE_SEPARATE:
            return
        params = self.region_params()
        index = self.view.selected_roi
        owner = self._region_params_owner
        if owner is not None and owner < len(params):
            params[owner] = self._capture_region_fit_params()
        if index is None or index >= len(params):
            # nothing selected: the dialog sets every region at once, as the
            # min. net gradient slider does
            self._region_params_owner = None
            return
        self._apply_region_fit_params(params[index])
        self._region_params_owner = index

    def flush_region_fit_params(self) -> list[dict]:
        """The per-region fit settings, with the dialog's current state
        written into the region it belongs to.

        The dialog *is* the selected region's settings until another region
        is selected, so anything that reads them (the fit) has to fold them
        back in first. With no region selected the dialog applies to every
        region, mirroring the min. net gradient slider."""
        params = self.region_params()
        if not params:
            return []
        current = self._capture_region_fit_params()
        index = self.view.selected_roi
        if self.fit_mode() != FIT_MODE_SEPARATE or index is None:
            return [dict(current) for _ in params]
        if index < len(params):
            params[index] = current
        self.view.roi_params = params
        return params

    def identify_mode(self) -> str:
        """Whether the channels are identified separately or added together,
        see ``IDENTIFY_MODE_SEPARATE`` / ``IDENTIFY_MODE_SUM``.

        Single-channel data always reports the separate mode: there is nothing
        to sum, and the combo box is hidden then (see
        ``_update_multichannel_widgets``)."""
        try:
            dialog = self.parameters_dialog
            multichannel = len(self.channels) > 1 or self.view.split_fov_mode
        except (AttributeError, RuntimeError):
            # ``parameters`` is also read from a partially built window
            return IDENTIFY_MODE_SEPARATE
        if not multichannel:
            return IDENTIFY_MODE_SEPARATE
        return dialog.identify_mode_combo.currentText()

    def fit_mode(self) -> str:
        """Whether the channels are fitted together or each on its own, see
        ``FIT_MODE_JOINT`` / ``FIT_MODE_SEPARATE``.

        Single-channel data always reports the joint mode: there is nothing
        to fit separately, the combo box is hidden then (see
        ``_update_multichannel_widgets``) and the joint dispatch degrades to
        the ordinary single-channel fit."""
        try:
            dialog = self.parameters_dialog
            multichannel = len(self.channels) > 1 or self.view.split_fov_mode
        except (AttributeError, RuntimeError):
            # read from a partially built window
            return FIT_MODE_JOINT
        if not multichannel:
            return FIT_MODE_JOINT
        return dialog.fit_mode_combo.currentText()

    @property
    def parameters(self) -> dict:
        """Dictionary with the identification settings: box size, min.
        net gradient, the temporal median window (0 when disabled), the
        Gaussian filter sigma (0 when disabled) and the identification mode.

        In split-FOV mode "Min. Net Gradient" is the list of per-region
        thresholds (see ``region_mngs``) rather than a single number; every
        consumer passes it straight to ``localize.identify``, which accepts
        one threshold per ROI. Identifying on the channel sum searches one
        image, so it takes the single shared threshold instead.

        Every key is compared in ``identifications_outdated`` and stored
        in ``last_identification_info``, so adding one here is enough for
        it to invalidate stale identifications and reach the metadata.
        """
        dialog = self.parameters_dialog
        mode = self.identify_mode()
        return {
            "Box Size": dialog.box_spinbox.value(),
            "Min. Net Gradient": (
                dialog.mng_slider.value()
                if mode == IDENTIFY_MODE_SUM
                else (self.region_mngs() or dialog.mng_slider.value())
            ),
            "Identification Mode": mode,
            "Temporal Median Window": (
                dialog.temporal_median_spinbox.value()
                if (
                    dialog.temporal_median_checkbox.isChecked()
                    # a movie too short for the filter reports it as off, so
                    # display, preview and identification all agree
                    and self.temporal_median_applicable()
                )
                else 0
            ),
            "Gaussian Filter Sigma": dialog.gaussian_filter_spinbox.value(),
        }

    def temporal_median_applicable(self) -> bool:
        """Whether the loaded movie is long enough for the temporal median
        filter to mean anything. Over a single frame the rolling median IS
        that frame, so the filtered movie would be identically zero - a
        black view and no identifications."""
        try:
            movie = self.movie
        except (AttributeError, RuntimeError):
            # ``parameters`` is also read from a partially built window,
            # where the filter setting stands on its own
            return True
        return movie is None or len(movie) >= _TEMPORAL_MEDIAN_MIN_FRAMES

    def _update_temporal_median_availability(self) -> None:
        """Show on the checkbox itself whether the temporal median filter
        applies to the loaded movie, so a movie that is too short for it
        does not look as though it were being filtered."""
        dialog = self.parameters_dialog
        checkbox = dialog.temporal_median_checkbox
        if self.temporal_median_applicable():
            checkbox.setEnabled(True)
            checkbox.setText(dialog._temporal_median_text)
            checkbox.setToolTip(dialog._temporal_median_tip)
            return
        checkbox.setEnabled(False)
        checkbox.setText(
            f"{dialog._temporal_median_text} "
            f"(needs >= {_TEMPORAL_MEDIAN_MIN_FRAMES} frames)"
        )
        checkbox.setToolTip(
            "This movie has "
            f"{0 if self.movie is None else len(self.movie)} frame(s). The "
            "filter subtracts a running per-pixel median, which needs a "
            "stack: over a single frame it would subtract the frame from "
            "itself. It is ignored for this movie."
        )

    def identification_rois(self) -> list:
        """The regions identification actually runs in.

        These are the drawn ROIs, except on the channel sum of split-FOV data:
        only the reference region of the summed canvas holds the sum (the other
        regions have been mapped into it), so it alone is searched - and the
        detections are then already in reference-channel coordinates.

        The rectangles are copied out of the view, so that whatever holds on
        to them (a worker's ROIs, ``last_identification_info["ROI"]``) keeps
        the regions as they were and not a live reference: the ROIs are also
        edited in place - a split-FOV region is moved by rewriting its corners
        (see ``View._move_region``) - and an aliased snapshot would silently
        agree with the edited list and never look outdated."""
        summed = self.channel_sum_view()
        if (
            summed is not None
            and summed.regions is not None
            and self.identify_mode() == IDENTIFY_MODE_SUM
        ):
            return _copied_rois([summed.regions[summed.reference]])
        return _copied_rois(self.view.rois)

    def drop_channel_sum(self) -> None:
        """Forget the channel sum: its detections, its registration and the
        summed view the display and the preview run on.

        Called whenever the sum would no longer describe what is on screen -
        the mode is switched off, the channels or the ROIs change, or the
        identification settings go stale."""
        try:
            self.sum_identifications = None
            self.sum_transforms = None
            self.sum_transform_source = ""
            self._sum_movie = None
            # whatever made the sum stale may also have made the registration
            # possible (or possible again), so let it be attempted once more
            self._sum_registration_failed = False
            # the filter stack is built on top of the summed view; comparing
            # the source by identity in ``identification_movie`` rebuilds it,
            # but dropping it here makes that explicit
            self._temporal_movie = None
            self._gaussian_movie = None
        except (AttributeError, RuntimeError):
            pass  # a partially built window has no channel sum to drop

    def _reset_contrast_to_frame(self) -> None:
        """Re-derive the display contrast from the frame now shown.

        Appearing or disappearing, the channel sum moves the image onto a
        different intensity scale (photons, summed over all channels), exactly
        as the identification filters do."""
        contrast_dialog = getattr(self, "contrast_dialog", None)
        if contrast_dialog is not None:
            contrast_dialog.reset_to_frame()

    def channel_sum_view(self) -> object | None:
        """The summed view the display and the identification currently run
        on, or None when there is none.

        ``parameters`` and ``identification_movie`` are also read from a
        partially built window, where no channel sum can exist - hence the
        guard rather than a plain attribute read."""
        try:
            return self._sum_movie
        except (AttributeError, RuntimeError):
            return None

    def validate_channel_sum(self) -> None:
        """Drop the channel sum if it no longer describes the loaded data.

        The sum is built for one particular layout - these movies, these
        regions - so moving a region, loading other channels or leaving
        split-FOV mode invalidates it. Checking it here, where the sum is
        used, keeps every one of those places from having to remember to."""
        summed = self.channel_sum_view()
        if summed is None:
            return
        if summed.regions is not None:
            valid = self.view.split_fov_mode and summed.matches_regions(
                self.view.rois
            )
        else:
            valid = (
                not self.view.split_fov_mode
                and len(self.channels) == len(summed.movies)
                and all(
                    channel.movie is movie
                    for channel, movie in zip(self.channels, summed.movies)
                )
            )
        if not valid:
            self.drop_channel_sum()
            self._reset_contrast_to_frame()

    def ensure_channel_sum(self, notify: bool = False) -> bool:
        """Build the summed view for ``IDENTIFY_MODE_SUM`` without identifying
        anything, so that the display and the identification preview run on it
        as soon as the mode is selected.

        The registration is the one ``identify_channel_sum`` would use, minus
        the bootstrap: a loaded spline calibration or channel registration, or
        the per-channel identifications when they have already been made.
        Neither of those needs a movie pass, so this is cheap enough to attempt
        from the draw path - the sum itself is a lazy view (see
        ``localize.SummedChannelsMovie``). When the channels cannot be
        registered yet the attempt is remembered and not repeated until the
        sum is dropped again, and the summed view only appears once
        ``identify_channel_sum`` has identified the channels to register them.

        Returns whether a summed view is in place.
        """
        try:
            if self.identify_mode() != IDENTIFY_MODE_SUM:
                return False
            if self._sum_movie is not None:
                return True
            if self._sum_registration_failed or self._sum_identify is not None:
                # already tried, or an identification is building one right now
                return False
            self._sum_registration_failed = True
            transforms, regions, source = self._sum_channel_transforms(
                estimate=True
            )
            if transforms is None:
                if notify:
                    where = (
                        "regions" if self.view.split_fov_mode else "channels"
                    )
                    self.status_bar.showMessage(
                        f"The {where} are not registered yet, so the summed "
                        "view cannot be shown. Identify (Ctrl+I) registers "
                        f"the {where} from their own detections first, or "
                        "load a multichannel / split-FOV spline PSF "
                        "calibration to use its registration."
                    )
                return False
            self._build_channel_sum(transforms, regions, source)
        except ValueError:
            self.drop_channel_sum()
            self._sum_registration_failed = True
            return False
        except (AttributeError, RuntimeError):
            return False  # a partially built window has no channels to sum
        self._sum_registration_failed = False
        if notify:
            self.status_bar.showMessage(
                f"Showing the sum of {len(self.sum_transforms)} channels "
                f"(registered from {source}). Note that it is in photons and "
                "over all channels, so the minimum net gradient has to be "
                "re-tuned for it."
            )
        return True

    def identification_movie(self) -> lib.IntArray3D:
        """The movie the display and the identification preview run on:
        the raw movie (or the channel sum, in ``IDENTIFY_MODE_SUM``),
        optionally temporal median filtered and then Gaussian smoothed -
        built the way ``localize.identify`` builds it, so that the preview
        and the batch run agree. The one difference is that the filters
        always cover the whole frame here, since this movie is also what is
        displayed (see ``_filtered_identification_movie``).

        Never used for fitting - ``FitWorker`` and ``save_spots`` always
        cut spots out of ``self.movie``.
        """
        self.validate_channel_sum()
        # rebuild it here rather than only where the mode is switched, so that
        # the summed view survives everything that invalidates it (moving a
        # region, loading a calibration, changing the settings)
        self.ensure_channel_sum()
        movie = self.movie
        # The summed view replaces the raw movie as the *source* of the filter
        # stack, exactly as ``identify_multichannel_sum`` does: it is the sum
        # that is searched for maxima, so it is the sum that is background
        # subtracted and smoothed. It only exists once the channels have been
        # registered (see ``ensure_channel_sum`` / ``identify_channel_sum``).
        summed = self.channel_sum_view()
        if summed is not None and self.identify_mode() == IDENTIFY_MODE_SUM:
            movie = summed
        if movie is None:
            self._temporal_movie = None
            self._gaussian_movie = None
            return None
        parameters = self.parameters
        window = parameters["Temporal Median Window"]
        sigma = parameters["Gaussian Filter Sigma"]
        if not self.temporal_median_applicable():
            window = 0
        try:
            cache = [self._temporal_movie, self._gaussian_movie]
        except (AttributeError, RuntimeError):
            # also called from a partially built window, which has no
            # cached filter stack yet
            cache = [None, None]
        movie, cache = _filtered_identification_movie(
            movie, window, sigma, cache
        )
        self._temporal_movie, self._gaussian_movie = cache
        return movie

    def on_parameters_changed(self) -> None:
        """Reset ``self.locs`` and draw frame."""
        # Ignore the value changes emitted while restoring a channel's
        # stored parameters - otherwise switching channels would wipe the
        # channel's localizations and fit markers.
        if self._switching_channel:
            return
        self.locs = None
        self.locs_display = None
        self.ready_for_fit = False
        self.draw_frame()

    def abort(self) -> None:
        """Abort the currently running async process."""
        if self._active_worker is not None:
            self._active_worker.requestInterruption()

    def on_worker_aborted(self) -> None:
        """Handle the abortion of any worker thread."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        # restore the GPU checkbox state for the selected model
        self.parameters_dialog.on_fit_optimizer_changed()
        if self._multi_fit is not None:
            # one channel aborted: the rest of the batch goes with it
            self._abort_independent_fit()
            return
        self.status_bar.showMessage("Aborted.")

    def identify(
        self,
        fit_afterwards: bool = False,
        calibrate_z: bool = False,
        calibrate_spline: bool = False,
    ) -> None:
        """Identify spots in the loaded movie.

        Parameters
        ----------
        fit_afterwards : bool, optional
            Whether to automatically fit the identified spots
            afterwards. Default is False.
        calibrate_z : bool, optional
            Whether to run z-calibration for 3D fitting after
            identification. Default is False.
        calibrate_spline : bool, optional
            Whether to build a cubic-spline PSF calibration after
            identification (see ``build_spline_calibration``). Default is
            False.
        """
        # The calibrations are built from bead stacks, one channel at a time,
        # so they always identify the channels separately - only the
        # experimental data can be identified on the sum.
        if (
            self.identify_mode() == IDENTIFY_MODE_SUM
            and not calibrate_spline
            and not calibrate_z
        ):
            self.identify_channel_sum(fit_afterwards=fit_afterwards)
            return
        self.drop_channel_sum()
        if len(self.channels) > 1:
            self._identify_all_channels(
                fit_afterwards=fit_afterwards,
                calibrate_z=calibrate_z,
                calibrate_spline=calibrate_spline,
            )
            return
        if self.movie is not None:
            self.status_bar.showMessage("Preparing identification...")
            self.identification_worker = IdentificationWorker(
                self, fit_afterwards, calibrate_z, calibrate_spline
            )
            self.identification_worker.progressMade.connect(
                self.on_identify_progress
            )
            self.identification_worker.finished.connect(
                self.on_identify_finished
            )
            self.identification_worker.aborted.connect(self.on_worker_aborted)
            self._active_worker = self.identification_worker
            self.abort_action.setEnabled(True)
            self.identification_worker.start()

    def on_identify_progress(
        self,
        frame_number: int,
        parameters: dict,
    ) -> None:
        """Update the status bar with the current identification
        progress."""
        n_frames = self.info[0]["Frames"]
        box = parameters["Box Size"]
        mng = _format_mng(parameters["Min. Net Gradient"])
        message = (
            f"Identifying in frame {frame_number:,} / {n_frames:,}"
            f" (Box Size: {box}; Min. Net Gradient: {mng}) ..."
        )
        self.status_bar.showMessage(message)

    def _linked_count_phrase(self, n_detections: int) -> str | None:
        """How many identified spots a joint fit would actually fit.

        A multichannel (or split-FOV) fit - spline or Gaussian - fits one spot
        per molecule: the reference detections that are found in every channel
        / region are fitted jointly, everything else is dropped (see
        ``localize.filter_linked_identifications``). The pairing uses whichever
        registration the fit would (see
        ``ParametersDialog.link_calibration``).
        """
        if self.fit_mode() == FIT_MODE_SEPARATE:
            # every channel is fitted on its own: every detection is fitted
            # and nothing is paired across channels
            return None
        if self.sum_identifications is not None:
            # identified on the channel sum: every detection is fitted, there
            # is no linking step (see Window.identify_channel_sum)
            return None
        cal = self.parameters_dialog.link_calibration()
        box = self.parameters["Box Size"]
        split_fov = bool(cal.get("split_fov"))
        self.status_bar.showMessage(
            "Linking spots across "
            + ("regions" if split_fov else "channels")
            + " ..."
        )
        self.status_bar.repaint()  # not processEvents: no re-entrant slots
        try:
            if split_fov:
                ids = self.identifications
                if ids is None or len(ids) == 0:
                    return None
                n_channels = int(
                    cal.get("n_channels") or len(cal.get("regions") or [])
                )
                regions = None
                if (
                    self.view.split_fov_mode
                    and len(self.view.rois) == n_channels
                ):
                    regions = [list(map(list, r)) for r in self.view.rois]
                _, n_kept, _ = (
                    localize.filter_linked_identifications_split_fov(
                        ids, cal, box, regions=regions
                    )
                )
                where = "regions"
            else:
                transforms = cal.get("channel_transforms")
                n_channels = min(
                    int(cal.get("n_channels", len(self.channels))),
                    len(self.channels),
                )
                if (
                    n_channels < 2
                    or not transforms
                    or len(transforms) < n_channels
                ):
                    return None
                # the flat state holds the active channel's detections
                self._snapshot_current_channel()
                ids_per_channel = [
                    c.identifications for c in self.channels[:n_channels]
                ]
                reference = ids_per_channel[0]
                if reference is None or len(reference) == 0:
                    return None
                if all(i is None or len(i) == 0 for i in ids_per_channel[1:]):
                    return None
                _, n_kept, _ = localize.filter_linked_identifications(
                    ids_per_channel, transforms, box
                )
                where = "channels"
        except (ValueError, KeyError, IndexError):
            return None
        return (
            f"{n_kept:,} spots linked across {n_channels} {where} "
            f"({n_detections:,} in total)"
        )

    def on_identify_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        fit_afterwards: bool,
        calibrate_z: bool,
        calibrate_spline: bool,
    ) -> None:
        """Handle the completion of the identification process. Save
        the parameters used, and localize/calibrate if requested."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        if len(identifications):
            self.locs = None
            self.locs_display = None
            self.last_identification_info = parameters.copy()
            self.last_identification_info["ROI"] = roi
            self.last_identification_info["Frame bounds"] = self.frame_range
            n_identifications = len(identifications)
            box = parameters["Box Size"]
            mng = _format_mng(parameters["Min. Net Gradient"])
            self.identifications = identifications
            self.ready_for_fit = True
            # for split-FOV data the detections of every region sit in this one
            # table, but the joint fit only fits molecules linked across all of
            # them - report that count (see _linked_count_phrase)
            counted = (
                self._linked_count_phrase(n_identifications)
                or f"{n_identifications:,} spots"
            )
            temporal_median = parameters.get("Temporal Median Window", 0)
            gaussian_sigma = parameters.get("Gaussian Filter Sigma", 0)
            filtered = ""
            if temporal_median:
                filtered += f"; Temporal median: {temporal_median}"
            if gaussian_sigma:
                filtered += f"; Gaussian sigma: {gaussian_sigma:g}"
            message = (
                f"Identified {counted} in {elapsed_time:.2f}"
                f" seconds. (Box Size: {box}; Min. Net Gradient: {mng}"
                f"{filtered}). Ready for fit."
            )
            self.status_bar.showMessage(message)
            self.draw_frame()
            # sound notification
            if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
                sound_path = lib.get_sound_notification_path()
                if sound_path is not None:
                    playsound(sound_path, block=False)
            if calibrate_spline:
                self.build_spline_calibration()
            elif fit_afterwards:
                self.fit(calibrate_z=calibrate_z)
        elif calibrate_spline:
            self.status_bar.showMessage("")
            QtWidgets.QMessageBox.information(
                self,
                "Spline PSF Calibration",
                "No beads were identified. Lower the minimum net gradient "
                "or check the selected frame range and try again.",
            )

    def _identify_all_channels(
        self,
        fit_afterwards: bool = False,
        calibrate_z: bool = False,
        calibrate_spline: bool = False,
        then: Callable[[], None] | None = None,
    ) -> None:
        """Identify spots in every channel in turn (multichannel Identify).

        Each channel is activated and identified with its own box / min. net
        gradient (shared when the matching 'Same across channels' link is on)
        and the shared ROI / frame range; the results are stored per channel.
        The originally active channel is restored when the batch finishes, and
        the requested follow-up (fit or calibration, as passed to
        :meth:`identify`) then runs once - so 'Localize (Identify && Fit)'
        behaves like Identify-then-Fit over all channels.

        ``then`` replaces that follow-up altogether: the channel-sum
        identification uses it to continue with the registration once every
        channel has been identified."""
        self._multi_identify = {
            "return_channel": self.current_channel,
            "queue": list(range(len(self.channels))),
            "total": len(self.channels),
            "done": 0,
            "sum": 0,
            "fit_afterwards": bool(fit_afterwards),
            "calibrate_z": bool(calibrate_z),
            "calibrate_spline": bool(calibrate_spline),
            "then": then,
        }
        self.status_bar.showMessage("Identifying all channels...")
        self._identify_run_next()

    def _identify_run_next(self) -> None:
        """Start identification on the next queued channel, or finish the
        multichannel Identify batch and restore the active channel."""
        state = self._multi_identify
        if state is None:
            return
        if not state["queue"]:
            self._multi_identify = None
            self.set_current_channel(state["return_channel"])
            # persist the last channel's results (set_current_channel only
            # snapshots on an actual switch)
            self._snapshot_current_channel()
            self._active_worker = None
            self.abort_action.setEnabled(False)
            self.draw_frame()
            if state["then"] is not None:
                # a caller that drives this batch itself (the channel-sum
                # identification) reports its own result
                state["then"]()
                return
            # the joint multichannel fit only fits molecules found in every
            # channel, so report those rather than the raw detection total
            counted = self._linked_count_phrase(state["sum"]) or (
                f"{state['sum']:,} spots across {state['total']} channels"
            )
            self.status_bar.showMessage(
                f"Identified {counted}. Ready for fit."
            )
            # the follow-up requested by the entry point (Localize / a
            # calibration run), now that every channel is identified
            if state["sum"]:
                if state["calibrate_spline"]:
                    self.build_spline_calibration()
                elif state["fit_afterwards"]:
                    self.fit(calibrate_z=state["calibrate_z"])
            elif state["calibrate_spline"]:
                QtWidgets.QMessageBox.information(
                    self,
                    "Spline PSF Calibration",
                    "No beads were identified in any channel. Lower the "
                    "minimum net gradient or check the selected frame range "
                    "and try again.",
                )
            return
        idx = state["queue"].pop(0)
        self.set_current_channel(idx)
        if self.movie is None:
            self._identify_run_next()
            return
        worker = IdentificationWorker(self, False, False, False)
        worker.progressMade.connect(self.on_identify_progress)
        worker.finished.connect(self._on_multi_identify_finished)
        worker.aborted.connect(self._on_multi_identify_aborted)
        self.identification_worker = worker
        self._active_worker = worker
        self.abort_action.setEnabled(True)
        worker.start()

    def _on_multi_identify_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        *_: object,
    ) -> None:
        """Store one channel's identifications, then start the next channel."""
        self._active_worker = None
        state = self._multi_identify
        if len(identifications):
            self.locs = None
            self.locs_display = None
            self.last_identification_info = parameters.copy()
            self.last_identification_info["ROI"] = roi
            self.last_identification_info["Frame bounds"] = self.frame_range
            self.identifications = identifications
            self.ready_for_fit = True
            if state is not None:
                state["sum"] += len(identifications)
        else:
            self.identifications = None
            self.ready_for_fit = False
        if state is not None:
            state["done"] += 1
            name = self.channels[self.current_channel].name
            self.status_bar.showMessage(
                f"Identified channel {state['done']}/{state['total']} "
                f"({name}): {len(identifications):,} spots ..."
            )
        self._identify_run_next()

    def _on_multi_identify_aborted(self) -> None:
        """Abort the multichannel Identify batch and restore the view."""
        state = self._multi_identify
        self._multi_identify = None
        self._sum_identify = None
        self._active_worker = None
        self.abort_action.setEnabled(False)
        if state is not None:
            self.set_current_channel(state["return_channel"])
        self.draw_frame()
        self.status_bar.showMessage("Aborted.")

    def n_identification_channels(self) -> int:
        """How many channels identification has to deal with: the loaded
        channel movies, or the drawn regions in split-FOV mode."""
        if self.view.split_fov_mode:
            return len(self.view.rois)
        return len(self.channels)

    def _sum_transforms_from_calibration(
        self, n_channels: int, regions: list | None
    ) -> tuple[list | None, str]:
        """A loaded calibration's reference->channel transforms, placed at the
        regions in use, and a phrase naming where they came from.

        Either kind of loaded registration serves: the multichannel spline PSF
        calibration, or the standalone channel registration the multichannel 2D
        Gaussian fit uses. Whichever describes this data is the one the fit will
        use, so the sum is built with exactly those transforms. ``(None, "")``
        if neither describes the data's layout."""
        pdialog = self.parameters_dialog
        cal = pdialog.link_calibration()
        source = (
            "the loaded channel registration"
            if cal is pdialog.channel_registration_calibration
            else "the loaded spline calibration"
        )
        if int(cal.get("n_channels", 0)) == n_channels:
            link_cal = self._link_calibration_for_mode(cal, n_channels)
            if link_cal is not None:
                try:
                    if regions is not None:
                        # re-place the stored affine at the drawn ROIs
                        _, _, transforms = localize.split_fov_fit_geometry(
                            link_cal, regions
                        )
                    else:
                        transforms = link_cal.get("channel_transforms")
                except (ValueError, KeyError, IndexError):
                    transforms = None
                if transforms and len(transforms) >= n_channels:
                    return (
                        [
                            transforms_mod.from_dict(t)
                            for t in transforms[:n_channels]
                        ],
                        source,
                    )
        # Whichever of the two was not tried above, for separate movies: their
        # transforms need no region placement, so a registration that does not
        # name this data's channel count still maps them.
        if regions is None:
            registration = pdialog.channel_registration_calibration or {}
            transforms = registration.get("channel_transforms")
            if transforms and len(transforms) >= n_channels:
                return (
                    [
                        transforms_mod.from_dict(t)
                        for t in transforms[:n_channels]
                    ],
                    "the loaded channel registration",
                )
        return None, ""

    def _sum_channel_transforms(
        self, estimate: bool = True
    ) -> tuple[list | None, list | None, str]:
        """The registration the channel sum is built with.

        The loaded multichannel / split-FOV spline calibration is the source
        whenever one describes this data - the sum is then built with exactly
        the transforms the fit will use. Without one, the channels are
        registered from their own detections (``estimate``), which is why the
        sum mode identifies every channel first.

        Returns ``(transforms, regions, source)``: one ``(2, 3)``
        reference->channel affine per channel, the split-FOV regions (None for
        separate channel movies) and a phrase naming where the transforms came
        from. ``transforms`` is None when the channels could not be registered;
        a channel that could not be registered is never silently replaced by
        the identity, since summing it in at the wrong place would smear the
        very spots this mode is meant to recover.
        """
        n_channels = self.n_identification_channels()
        if n_channels < 2:
            return None, None, ""
        regions = (
            [_normalize_rect(r) for r in self.view.rois]
            if self.view.split_fov_mode
            else None
        )
        transforms, source = self._sum_transforms_from_calibration(
            n_channels, regions
        )
        if transforms is not None:
            return transforms, regions, source
        if not estimate:
            return None, regions, ""
        ids_per_channel, _ = self._channel_identification_inputs()
        if ids_per_channel is None:
            return None, regions, ""
        try:
            transforms, n_pairs = (
                spline.estimate_transforms_from_identifications(
                    ids_per_channel[:n_channels],
                    self.parameters["Box Size"],
                    regions=regions,
                    frame_shape=self._frame_shape_for_registration(),
                    max_frames=_SUM_REGISTRATION_MAX_FRAMES,
                    return_diagnostics=True,
                )
            )
        except Exception:
            return None, regions, ""
        if transforms is None or any(t is None for t in transforms):
            return None, regions, ""
        return (
            list(transforms),
            regions,
            "the per-channel identifications "
            f"({min(n_pairs[1:]):,} pairs or more per channel)",
        )

    def identify_channel_sum(self, fit_afterwards: bool = False) -> None:
        """Identify spots on the channels added together
        (``IDENTIFY_MODE_SUM``).

        The channels are mapped onto the reference channel and summed in
        photons, and the spots are identified in that sum, so a molecule too
        dim to be detected in any single channel is still found (see
        ``localize.SummedChannelsMovie``). The registration comes from the
        loaded spline calibration; without one every channel is identified
        first and the transforms are estimated from those detections, and only
        then is the sum built.

        The summed view shown while the mode is selected (see
        ``ensure_channel_sum``) is registered the same way, so when one is on
        screen it is identified as it stands - the identification then searches
        exactly the image the preview showed.
        """
        n_channels = self.n_identification_channels()
        if n_channels < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Identify on the channel sum",
                "Summing needs at least two channels. Load them with "
                "'File > Open channels from several movies' / 'Open one "
                "multichannel movie', or enable 'Regions = channels' and draw "
                "one region per channel (reference first).",
            )
            return
        self.validate_channel_sum()
        if self.channel_sum_view() is not None:
            # already registered and on screen: identify on it as it stands
            self.sum_identifications = None  # the ones about to be replaced
            self._sum_identify = {"fit_afterwards": bool(fit_afterwards)}
            self._start_sum_identification()
            return
        self.drop_channel_sum()
        self._sum_identify = {"fit_afterwards": bool(fit_afterwards)}
        transforms, regions, source = self._sum_channel_transforms(
            estimate=False
        )
        if transforms is not None:
            self._run_sum_identification(transforms, regions, source)
            return
        # no calibration for this layout: register the channels from their own
        # detections, which have to be made first
        self.status_bar.showMessage(
            "Identifying every channel to register them for the sum..."
        )
        if self.view.split_fov_mode:
            self._identify_regions_for_sum()
        else:
            self._identify_all_channels(then=self._sum_after_bootstrap)

    def _channel_camera_info(self, channel: "Channel") -> dict:
        """One channel's camera info.

        Only the active channel's camera parameters are in the dialog at any
        time, so summing the channels in photons - the whole point of doing it
        in photons - has to read the other channels' stored snapshots (see
        ``_capture_params``). Channels that carry no snapshot fall back to the
        dialog's current values."""
        if self.channels and channel is self.channels[self.current_channel]:
            return self.camera_info
        params = getattr(channel, "params", None) or {}
        keys = {
            "Baseline": "baseline",
            "Gain": "gain",
            "Sensitivity": "sensitivity",
            "Qe": "qe",
            "Pixelsize": "pixelsize",
        }
        if not all(key in params for key in keys.values()):
            return self.camera_info
        return {name: params[key] for name, key in keys.items()}

    def _identify_regions_for_sum(self) -> None:
        """Split-FOV: one ordinary whole-movie identification pass over the
        drawn regions, whose detections register the regions against each
        other before they are summed."""
        parameters = dict(self.parameters)
        # the regions are identified as the channels they are, each with its
        # own threshold - the single summed threshold applies to the sum only
        parameters["Identification Mode"] = IDENTIFY_MODE_SEPARATE
        parameters["Min. Net Gradient"] = (
            self.region_mngs() or self.parameters_dialog.mng_slider.value()
        )
        worker = IdentificationWorker(
            self,
            False,
            False,
            False,
            rois=list(self.view.rois),
            parameters=parameters,
        )
        worker.progressMade.connect(self.on_identify_progress)
        worker.finished.connect(self._on_sum_bootstrap_finished)
        worker.aborted.connect(self._on_sum_identify_aborted)
        self.identification_worker = worker
        self._active_worker = worker
        self.abort_action.setEnabled(True)
        worker.start()

    def _on_sum_bootstrap_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        *_: object,
    ) -> None:
        """Split-FOV: keep the per-region detections and continue with the
        registration and the sum."""
        self._active_worker = None
        self.identifications = identifications
        self.last_identification_info = parameters.copy()
        self.last_identification_info["ROI"] = roi
        self.last_identification_info["Frame bounds"] = self.frame_range
        self.ready_for_fit = bool(len(identifications))
        self._sum_after_bootstrap()

    def _sum_after_bootstrap(self) -> None:
        """Register the channels from the identifications just made, then
        identify on their sum."""
        state = self._sum_identify
        if state is None:  # aborted meanwhile
            return
        transforms, regions, source = self._sum_channel_transforms(
            estimate=True
        )
        if transforms is None:
            self._sum_identify = None
            self._active_worker = None
            self.abort_action.setEnabled(False)
            self.status_bar.showMessage("")
            where = "regions" if self.view.split_fov_mode else "channels"
            QtWidgets.QMessageBox.warning(
                self,
                "Identify on the channel sum",
                f"The {where} could not be registered against each other, so "
                "they cannot be summed. Every channel needs enough detections "
                "for the transform to be estimated: lower the minimum net "
                f"gradient of the dim {where} until spots appear in them, or "
                "load a multichannel / split-FOV spline PSF calibration, "
                "whose registration is used directly.",
            )
            self.draw_frame()
            return
        self._run_sum_identification(transforms, regions, source)

    def _build_channel_sum(
        self, transforms: list, regions: list | None, source: str
    ) -> None:
        """Put the summed view built from ``transforms`` in place of the raw
        movie for the display, the preview and the identification.

        Raises ``ValueError`` (from ``localize.SummedChannelsMovie``) when the
        channels cannot be summed; the caller decides how loudly to say so."""
        if regions is None:
            channels = self.channels[: len(transforms)]
            movies = [c.movie for c in channels]
            camera_infos = [self._channel_camera_info(c) for c in channels]
        else:
            # split-FOV: the regions are channels of the one loaded movie
            movies = [self.movie] * len(transforms)
            camera_infos = [self.camera_info] * len(transforms)
        # As in the multichannel fit, the GUI holds a single sCMOS calibration
        # and applies it to every channel (split-FOV really is one sensor).
        camera_calibration = self.parameters_dialog.camera_calibration or None
        camera_calibrations = (
            None
            if camera_calibration is None
            else [camera_calibration] * len(movies)
        )
        summed = localize.SummedChannelsMovie(
            movies,
            transforms,
            camera_infos=camera_infos,
            regions=regions,
            camera_calibrations=camera_calibrations,
        )
        self._sum_movie = summed
        self.sum_transforms = list(summed.transforms)
        self.sum_transform_source = source
        # the filter stack now sits on top of the sum
        self._temporal_movie = None
        self._gaussian_movie = None
        # ... and so does the display, on the sum's intensity scale
        self._reset_contrast_to_frame()

    def _run_sum_identification(
        self, transforms: list, regions: list | None, source: str
    ) -> None:
        """Build the summed view from ``transforms`` and identify on it."""
        try:
            self._build_channel_sum(transforms, regions, source)
        except ValueError as error:
            self._sum_identify = None
            self._active_worker = None
            self.abort_action.setEnabled(False)
            self.status_bar.showMessage("")
            QtWidgets.QMessageBox.warning(
                self, "Identify on the channel sum", str(error)
            )
            return
        self._start_sum_identification()

    def _start_sum_identification(self) -> None:
        """Identify on the summed view now in place."""
        summed = self._sum_movie
        worker = IdentificationWorker(
            self,
            False,
            False,
            False,
            movie=summed,
            rois=self.identification_rois(),
            parameters=dict(self.parameters),
        )
        worker.progressMade.connect(self.on_identify_progress)
        worker.finished.connect(self._on_sum_identify_finished)
        worker.aborted.connect(self._on_sum_identify_aborted)
        self.identification_worker = worker
        self._active_worker = worker
        self.abort_action.setEnabled(True)
        if self._sum_identify is not None:
            self._sum_identify["n_channels"] = len(summed.movies)
            self._sum_identify["source"] = self.sum_transform_source
        self.status_bar.showMessage(
            f"Identifying on the sum of {len(summed.movies)} channels "
            f"(registered from {self.sum_transform_source})..."
        )
        worker.start()

    def _on_sum_identify_finished(
        self,
        parameters: dict,
        roi: list,
        elapsed_time: float,
        identifications: pd.DataFrame,
        *_: object,
    ) -> None:
        """Store the channel sum's detections as the reference channel's
        identifications and run the queued follow-up.

        They are already in reference-channel coordinates and already the
        cross-channel consensus, so the joint fit takes them as they are - see
        ``MultichannelSplineFitWorker(link_identifications=False)``."""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        state = self._sum_identify
        self._sum_identify = None
        if not len(identifications):
            self.sum_identifications = None
            self.ready_for_fit = False
            self.status_bar.showMessage(
                "No spots identified on the channel sum. Note that the sum is "
                "in photons and over all channels, so the minimum net "
                "gradient has to be re-tuned for it."
            )
            self.draw_frame()
            return
        if not self.view.split_fov_mode and self.current_channel != 0:
            # the sum lives in the reference channel's coordinates
            self.set_current_channel(0)
        self.locs = None
        self.locs_display = None
        self.identifications = identifications
        self.sum_identifications = identifications
        self.ready_for_fit = True
        self.last_identification_info = parameters.copy()
        self.last_identification_info["ROI"] = roi
        self.last_identification_info["Frame bounds"] = self.frame_range
        state = state or {}
        n_summed = state.get("n_channels") or len(self.sum_transforms or [])
        source = state.get("source") or self.sum_transform_source
        self.last_identification_info["Channel sum registration"] = source
        if not self.view.split_fov_mode:
            reference = self.channels[0]
            reference.identifications = identifications
            reference.ready_for_fit = True
            reference.last_identification_info = self.last_identification_info
        box = parameters["Box Size"]
        mng = _format_mng(parameters["Min. Net Gradient"])
        self.status_bar.showMessage(
            f"Identified {len(identifications):,} spots on the sum of "
            f"{n_summed} channels in "
            f"{elapsed_time:.2f} seconds. (Box Size: {box}; Min. Net "
            f"Gradient: {mng}; registered from {source}). "
            "Ready for fit."
        )
        self.draw_frame()
        if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
            sound_path = lib.get_sound_notification_path()
            if sound_path is not None:
                playsound(sound_path, block=False)
        if state.get("fit_afterwards"):
            self.fit()

    def _on_sum_identify_aborted(self) -> None:
        """Abort a channel-sum identification and drop the partial sum."""
        self._sum_identify = None
        self.drop_channel_sum()
        self.on_worker_aborted()
        self.draw_frame()
        self.status_bar.showMessage("Aborted.")

    def _check_spline_box_size(self, spline_calibration: dict) -> bool:
        """Check that the identification box size is not larger than in
        the spline calibration."""
        n_data = spline_calibration.get("n_data")
        if not n_data:
            return True  # nothing to compare against; let the fit proceed
        calib_box = int(n_data[0])
        box = self.parameters["Box Size"]
        # equal or smaller: supported by fitting against a centered crop
        if box <= calib_box:
            return True
        detail = (
            f"The selected box size ({box} px) is larger than this spline "
            f"calibration, which was built with a box size of "
            f"{calib_box} px. Use the box of the same size or smaller."
        )
        reply = QtWidgets.QMessageBox.question(
            self,
            "Spline PSF fit — box size",
            f"{detail}\n\nSet the box size to {calib_box} px now?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.parameters_dialog.box_spinbox.setValue(calib_box)
        return False

    def fit(self, calibrate_z: bool = False) -> None:
        """Fit identified spots (single molecules).

        Parameters
        ----------
        calibrate_z : bool, optional
            Whether to perform z-calibration during fitting. Default is
            False.
        """
        # a sum built for a layout that has since changed must not decide
        # whether the fit links its identifications across channels
        self.validate_channel_sum()
        if self.movie is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Fit",
                "Load a movie before fitting.",
            )
            return
        if not self._ready_for_fit_anywhere():
            QtWidgets.QMessageBox.warning(
                self,
                "Fit",
                "No identifications available. Run Identify (or load "
                "identifications) before fitting.",
            )
            return
        self.status_bar.showMessage("Preparing fit...")
        # Each channel (or split-FOV region) fitted on its own, with its own
        # settings and its own output file. Nothing is registered, linked or
        # shared, so none of the joint dispatch below applies. A z-calibration
        # run is single-channel by nature and keeps the ordinary path.
        if not calibrate_z and self.fit_mode() == FIT_MODE_SEPARATE:
            self._fit_independent()
            return
        model = self.parameters_dialog.fit_model.currentText()
        optimizer = self.parameters_dialog.fit_optimizer.currentText()
        method = _fit_code(model, optimizer)
        # get the convergence criterion and max iterations
        iterates = (
            self.parameters_dialog.current_fit_code() in _CONVERGENCE_CODES
        )
        eps = (
            self.parameters_dialog.convergence_criterion.value()
            if iterates
            else None
        )
        max_it = self.parameters_dialog.max_it.value() if iterates else None
        use_gpu = self.parameters_dialog.gpu_checkbox.isChecked()
        # A loaded channel registration turns the spherical Gaussian into a
        # joint fit across every channel; with none loaded the fit stays
        # single-channel, exactly as before.
        if FIT_MODELS[model].get("needs_channel_registration"):
            registration = (
                self.parameters_dialog.channel_registration_calibration
            )
            # separate movies loaded as channels, or one movie whose drawn
            # regions are the channels
            split_fov = bool(registration.get("split_fov")) and (
                self.view.split_fov_mode
            )
            if registration and (len(self.channels) > 1 or split_fov):
                self._start_multichannel_gauss_fit(
                    registration, method, use_gpu, eps, max_it
                )
                return
        if method.startswith("spline"):
            spline_calibration = self.parameters_dialog.spline_calibration
            if spline_calibration.get("model") == "spline-3d-multichannel":
                if not self._check_spline_box_size(spline_calibration):
                    self.status_bar.showMessage("")
                    return
                self._start_multichannel_spline_fit(
                    spline_calibration, method, eps, max_it
                )
                return
        worker = self._single_channel_fit_worker(calibrate_z=calibrate_z)
        if worker is None:
            self.status_bar.showMessage("")
            return
        self.fit_worker = worker
        self.fit_worker.finished.connect(self.on_fit_finished)
        self._start_fit_worker(self.fit_worker)

    def _single_channel_fit_worker(
        self,
        identifications: pd.DataFrame | None = None,
        calibrate_z: bool = False,
    ) -> FitWorker | None:
        """A ``FitWorker`` for the active channel, from the settings now on
        the Parameters dialog.

        Shared by the ordinary single-channel fit and by every step of an
        independent batch, which is what lets a channel (or a region) be
        fitted with whatever model, optimizer and calibration it carries.
        Returns None - having said why - when those settings cannot fit:
        a spline model with no calibration loaded, a box larger than the
        calibration's, or a joint multichannel calibration where a
        single-channel one is needed.

        Parameters
        ----------
        identifications : pd.DataFrame or None, optional
            The detections to fit. None (the default) fits the active
            channel's own; a split-FOV region passes the detections inside
            it.
        calibrate_z : bool, optional
            Whether the fit feeds a z-calibration. Default False.
        """
        dialog = self.parameters_dialog
        model = dialog.fit_model.currentText()
        method = _fit_code(model, dialog.fit_optimizer.currentText())
        iterates = dialog.current_fit_code() in _CONVERGENCE_CODES
        eps = dialog.convergence_criterion.value() if iterates else None
        max_it = dialog.max_it.value() if iterates else None
        fit_z = dialog.fit_z_checkbox.isChecked()
        use_gpu = dialog.gpu_checkbox.isChecked()
        spline_calibration = None
        if method.startswith("spline"):
            spline_calibration = dialog.spline_calibration
            if not spline_calibration:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Spline PSF fit",
                    "Load a spline PSF calibration first (Experimental "
                    "PSF (spline) > Load calibration), or build one via "
                    "3D > Calibrate spline PSF.",
                )
                return None
            if spline_calibration.get("model") == "spline-3d-multichannel":
                # it describes every channel at once and only means anything
                # to the joint fit; fitting one channel against it would use
                # the reference channel's PSF everywhere
                QtWidgets.QMessageBox.warning(
                    self,
                    "Spline PSF fit",
                    "This is a multichannel spline calibration, which "
                    "belongs to the joint fit. Fitting the channels "
                    "separately needs one single-channel calibration per "
                    "channel: load each channel's own calibration (untick "
                    "'PSF calibration' under 'Same settings across "
                    "channels'), or set 'Fit' back to "
                    f"'{FIT_MODE_JOINT}'.",
                )
                return None
            # The spline fit requires the box size to match the one the
            # calibration was built with.
            if not self._check_spline_box_size(spline_calibration):
                return None
            # A 3D spline fit recovers z directly, so the separate
            # astigmatism z-fitting step must not run.
            fit_z = False
        return FitWorker(
            self.movie,
            self.info,
            self.camera_info,
            (
                self.identifications
                if identifications is None
                else identifications
            ),
            self.parameters["Box Size"],
            method,
            eps,
            max_it,
            fit_z,
            calibrate_z,
            use_gpu,
            spline_calibration=spline_calibration,
            lateral_transforms=self._extra_lateral_transforms(
                spline_calibration
            ),
            camera_calibration=(
                self.parameters_dialog.camera_calibration or None
            ),
        )

    def _start_fit_worker(self, worker: FitWorker) -> None:
        """Wire a ``FitWorker``'s progress/abort signals up and start it.
        Its ``finished`` signal is connected by the caller, which is what
        decides whether the result is one fit or one step of a batch."""
        worker.progressMade.connect(self.on_fit_progress)
        worker.cutProgressMade.connect(self.on_cut_progress)
        worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = worker
        self.abort_action.setEnabled(True)
        worker.start()

    def _ready_for_fit_anywhere(self) -> bool:
        """Whether there is anything to fit.

        The active channel's detections, or - when the channels are fitted
        one at a time - any channel's: a batch that has three identified
        channels must not be refused because the fourth one is empty."""
        if self.ready_for_fit:
            return True
        if self.fit_mode() != FIT_MODE_SEPARATE:
            return False
        return any(channel.ready_for_fit for channel in self.channels)

    def _fit_independent(self) -> None:
        """Fit every channel - or every split-FOV region - on its own.

        The fitting counterpart of ``_identify_all_channels``: the channels
        are worked through one at a time, each with the model, optimizer and
        calibrations it carries, and each saved to its own file. Nothing is
        registered or linked, so every detection is fitted and the channels
        come out independent, to be connected afterwards.
        """
        if self.sum_identifications is not None:
            # the sum's detections are one consensus set in the reference
            # channel's coordinates, made for the joint fit; fitted
            # separately they would put the reference channel's positions
            # into every channel
            QtWidgets.QMessageBox.information(
                self,
                "Fit",
                "These detections were identified on the sum of the "
                "channels, which is made for the joint fit: they are one "
                "set in the reference channel's coordinates. Set "
                f"'Identify on' to '{IDENTIFY_MODE_SEPARATE}' and identify "
                "again to fit the channels on their own.",
            )
            self.status_bar.showMessage("")
            return
        if self.view.split_fov_mode:
            # the drawn regions, not identification_rois(): every region is
            # fitted here, whatever was identified where
            regions = _copied_rois(self.view.rois)
            if not regions:
                QtWidgets.QMessageBox.information(
                    self,
                    "Fit",
                    "'Regions = channels' is on but no region is drawn. "
                    "Draw the regions, or turn split-FOV mode off.",
                )
                self.status_bar.showMessage("")
                return
            state = {
                "regions": regions,
                "params": self.flush_region_fit_params(),
                "queue": list(range(len(regions))),
                "total": len(regions),
                "info": self.info,
                "selected_roi": self.view.selected_roi,
            }
        else:
            state = {
                "regions": None,
                "params": None,
                "queue": list(range(len(self.channels))),
                "total": len(self.channels),
                "info": None,
                "selected_roi": None,
            }
        state |= {
            "done": 0,
            "sum": 0,
            "skipped": [],
            "current": None,
            "return_channel": self.current_channel,
        }
        self._multi_fit = state
        where = "regions" if state["regions"] is not None else "channels"
        self.status_bar.showMessage(f"Fitting {state['total']} {where}...")
        self._fit_run_next()

    def _fit_run_next(self) -> None:
        """Fit the next queued channel / region, or finish the batch."""
        state = self._multi_fit
        if state is None:
            return
        if not state["queue"]:
            self._finish_independent_fit()
            return
        index = state["queue"].pop(0)
        state["current"] = index
        if state["regions"] is not None:
            self._start_region_fit(index)
        else:
            self._start_channel_fit(index)

    def _skip_in_batch(self, name: str, reason: str) -> None:
        """Note a channel / region the batch cannot fit and move on."""
        state = self._multi_fit
        state["skipped"].append(name)
        state["done"] += 1
        self.status_bar.showMessage(f"Skipped {name}: {reason}.")
        self._fit_run_next()

    def _start_channel_fit(self, index: int) -> None:
        """Fit one loaded channel with its own settings.

        Activating the channel restores them (model, optimizer, calibrations,
        camera) exactly as a manual channel switch does, so the fit that runs
        is the one that channel is set up for."""
        self.set_current_channel(index)
        state = self._multi_fit
        name = self.channels[index].name
        if self.movie is None:
            self._skip_in_batch(name, "no movie loaded")
            return
        if self.identifications is None or not len(self.identifications):
            self._skip_in_batch(name, "no identifications")
            return
        worker = self._single_channel_fit_worker()
        if worker is None:  # the dialog has said why
            self._abort_independent_fit()
            return
        self.status_bar.showMessage(
            f"Fitting channel {state['done'] + 1}/{state['total']} "
            f"({name}) ..."
        )
        self.fit_worker = worker
        worker.finished.connect(self.on_fit_finished)
        self._start_fit_worker(worker)

    def _start_region_fit(self, index: int) -> None:
        """Fit one split-FOV region with its own settings.

        One movie holds every region, so the region is selected (which is
        what its own settings hang off) and only the detections inside it are
        fitted."""
        state = self._multi_fit
        region = state["regions"][index]
        label = localize.region_label(index)
        self.view.selected_roi = index
        self._apply_region_fit_params(state["params"][index])
        self._region_params_owner = index
        identifications = localize.confine_to_region(
            self.identifications, region
        )
        if not len(identifications):
            self._skip_in_batch(label, "no identifications in it")
            return
        worker = self._single_channel_fit_worker(
            identifications=identifications
        )
        if worker is None:  # the dialog has said why
            self._abort_independent_fit()
            return
        self.status_bar.showMessage(
            f"Fitting region {state['done'] + 1}/{state['total']} "
            f"({label}) ..."
        )
        self._region_suffix = label
        self.fit_worker = worker
        worker.finished.connect(self._on_region_fit_finished)
        self._start_fit_worker(worker)
        self.draw_frame()

    def _on_region_fit_finished(
        self,
        locs: pd.DataFrame,
        elapsed_time: float,
        fit_z: bool,
        calibrate_z: bool,
    ) -> None:
        """Store and save one region's localizations, in its own coordinates.

        The region is a channel of its own once it is fitted separately, so
        its localizations are moved to its own origin and its metadata gives
        the region's size - which is what makes the regions overlay each
        other when they are loaded side by side in Picasso: Render. The
        on-screen markers stay where the spots were fitted.
        """
        state = self._multi_fit
        if state is None:  # aborted while the fit was finishing
            return
        self._active_worker = None
        self.abort_action.setEnabled(False)
        region = state["regions"][state["current"]]
        self.status_bar.showMessage(
            f"Fitted {len(locs):,} spots in region "
            f"{localize.region_label(state['current'])} in "
            f"{elapsed_time:.2f} seconds."
        )
        self.locs = localize.locs_to_region_coordinates(locs, region)
        self.locs_display = locs
        self.info = localize.region_movie_info(state["info"], region)
        self.draw_frame()
        # the same tail as a single fit: the astigmatism z-fit (when it is
        # asked for), then saving and the optional drift correction, which
        # is where the batch moves on (see save_locs_after_fit)
        if fit_z:
            self.fit_z()
        else:
            self.save_locs_after_fit()

    def _finish_independent_fit(self) -> None:
        """Restore the view after the last channel / region and report."""
        state = self._multi_fit
        self._multi_fit = None
        self._region_suffix = ""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        self._restore_after_independent_fit(state)
        where = "regions" if state["regions"] is not None else "channels"
        n_fitted = state["total"] - len(state["skipped"])
        message = (
            f"Fitted {state['sum']:,} spots in {n_fitted} {where}, saved to "
            f"{n_fitted} files."
        )
        if state["skipped"]:
            message += " Skipped " + ", ".join(state["skipped"]) + "."
        self.status_bar.showMessage(message)

    def _abort_independent_fit(self) -> None:
        """Give up the rest of the batch and restore the view."""
        state = self._multi_fit
        self._multi_fit = None
        self._region_suffix = ""
        self._active_worker = None
        self.abort_action.setEnabled(False)
        if state is not None:
            self._restore_after_independent_fit(state)
        self.status_bar.showMessage("Aborted.")

    def _restore_after_independent_fit(self, state: dict) -> None:
        """Put back what the batch moved: the movie's own metadata and the
        region selection, or the channel that was active before it."""
        if state["regions"] is not None:
            self.info = state["info"]
            self.view.selected_roi = state["selected_roi"]
            self.sync_region_fit_params()
        else:
            self.set_current_channel(state["return_channel"])
            # persist the last channel's results (set_current_channel only
            # snapshots on an actual switch)
            self._snapshot_current_channel()
        self.draw_frame()

    def _start_multichannel_gauss_fit(
        self,
        registration: dict,
        method: str,
        use_gpu: bool = False,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """Fit a spherical Gaussian jointly across all loaded channels.

        The first loaded channel is the reference; its identifications are
        mapped into every channel via the registration's stored transforms.
        The Gaussian counterpart of :meth:`_start_multichannel_spline_fit`."""
        # Lateral corrections are single-channel only - this fit registers its
        # channels itself - so say so rather than silently ignoring them.
        if self.parameters_dialog.lateral_transforms:
            QtWidgets.QMessageBox.information(
                self,
                "Multichannel Gaussian fit",
                "Lateral corrections apply to single-channel data only and "
                "are not used here: the multichannel fit registers its "
                "channels itself, from the loaded channel registration.",
            )
        n_channels = int(
            registration.get(
                "n_channels", len(registration.get("channel_transforms", []))
            )
        )
        if registration.get("split_fov"):
            self._start_split_fov_gauss_fit(
                registration, method, use_gpu, eps, max_it
            )
            return
        # persist the displayed channel's identifications, so the per-channel
        # tables read below are complete even if no channel switch happened
        # since the last Identify run
        self._snapshot_current_channel()
        if len(self.channels) < n_channels:
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel Gaussian fit",
                f"This registration expects {n_channels} channels, but "
                f"{len(self.channels)} are loaded. Load them with "
                "'File > Open channels from several movies' in the same "
                "order as the registration.",
            )
            self.status_bar.showMessage("")
            return
        reference = self.channels[0]
        if reference.identifications is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel Gaussian fit",
                "Identify spots in the reference channel (the first loaded "
                "channel) before running a multichannel fit.",
            )
            self.status_bar.showMessage("")
            return
        link_photons = self.parameters_dialog._gauss_link_photons_enabled()
        if not link_photons and not (
            2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel Gaussian fit",
                "The photon-decoupled model supports 2 to "
                f"{precision._LINK_XYZ_MAX_CHANNELS} channels, but this "
                f"registration has {n_channels}. Tick 'Link photon counts "
                "across channels' to fit them with a shared photon count.",
            )
            self.status_bar.showMessage("")
            return
        movies = [self.channels[c].movie for c in range(n_channels)]
        camera_infos = [
            getattr(self.channels[c], "camera_info", None) or self.camera_info
            for c in range(n_channels)
        ]
        ids_per_channel = [
            getattr(self.channels[c], "identifications", None)
            for c in range(n_channels)
        ]
        # Identifications made on the channel sum are the linked set already
        # (see Window.identify_channel_sum).
        link_identifications = self.sum_identifications is None
        n_missing = sum(
            1 for ids in ids_per_channel[1:] if ids is None or len(ids) == 0
        )
        if n_missing and link_identifications:
            QtWidgets.QMessageBox.information(
                self,
                "Multichannel Gaussian fit",
                f"{n_missing} of the {n_channels - 1} non-reference channels "
                "have no identifications, so localizations cannot be linked "
                "across all channels. Run 'Analyze > Identify' (Ctrl+I) first "
                "for a fully linked fit; continuing with the channels that "
                "are identified.",
            )
        self.fit_worker = MultichannelGaussianFitWorker(
            movies,
            camera_infos,
            reference.identifications,
            self.parameters["Box Size"],
            registration,
            mle=method.startswith("gaussmle"),
            use_gpu=use_gpu,
            link_photons=link_photons,
            identifications_per_channel=ids_per_channel,
            link_identifications=link_identifications,
            eps=eps,
            max_it=max_it,
            camera_calibration=(
                self.parameters_dialog.camera_calibration or None
            ),
        )
        self.fit_worker.linkMade.connect(self.on_link_progress)
        self.fit_worker.linkFinished.connect(self.on_link_finished)
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def _start_split_fov_gauss_fit(
        self,
        registration: dict,
        method: str,
        use_gpu: bool = False,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """Fit a spherical Gaussian across the regions of the single movie.

        The channels are placed at the drawn ROIs when they match the
        registration's channel count - so a moved split is handled by
        re-drawing them - otherwise at the registration's stored regions. The
        Gaussian counterpart of :meth:`_start_split_fov_spline_fit`."""
        if self.identifications is None or len(self.identifications) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV Gaussian fit",
                "Identify spots first (they are confined to the reference "
                "region automatically).",
            )
            self.status_bar.showMessage("")
            return
        n_channels = int(registration.get("n_channels", 0))
        regions = None
        if self.view.split_fov_mode and len(self.view.rois) == n_channels:
            regions = [list(map(list, r)) for r in self.view.rois]
        link_photons = self.parameters_dialog._gauss_link_photons_enabled()
        if not link_photons and not (
            2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV Gaussian fit",
                "The photon-decoupled model supports 2 to "
                f"{precision._LINK_XYZ_MAX_CHANNELS} channels, but this "
                f"registration has {n_channels}. Tick 'Link photon counts "
                "across channels' to fit them with a shared photon count.",
            )
            self.status_bar.showMessage("")
            return
        # A reference region in the wrong place would otherwise fit nothing,
        # so say so rather than returning an empty result.
        ref_rect = (regions or registration.get("regions"))[
            0 if regions else int(registration.get("reference", 0))
        ]
        (ry0, rx0), (ry1, rx1) = ref_rect
        rx = np.asarray(self.identifications["x"], dtype=float)
        ry = np.asarray(self.identifications["y"], dtype=float)
        n_in = int(
            np.count_nonzero(
                (rx >= min(rx0, rx1))
                & (rx < max(rx0, rx1))
                & (ry >= min(ry0, ry1))
                & (ry < max(ry0, ry1))
            )
        )
        if n_in == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV Gaussian fit",
                f"{len(self.identifications)} spots were identified but none "
                "fall inside the reference region, so there is nothing to "
                "fit. The reference region is probably in the wrong place for "
                "this data: enable 'Regions = channels' and drag the ROIs onto "
                "the channels (reference first), or re-register the channels.",
            )
            self.status_bar.showMessage("")
            return
        self.fit_worker = MultichannelGaussianFitWorker(
            [self.movie],
            [self.camera_info],
            self.identifications,
            self.parameters["Box Size"],
            registration,
            mle=method.startswith("gaussmle"),
            use_gpu=use_gpu,
            link_photons=link_photons,
            split_fov=True,
            regions=regions,
            eps=eps,
            max_it=max_it,
            camera_calibration=(
                self.parameters_dialog.camera_calibration or None
            ),
        )
        self.fit_worker.linkMade.connect(self.on_link_progress)
        self.fit_worker.linkFinished.connect(self.on_link_finished)
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def _start_multichannel_spline_fit(
        self,
        calibration: dict,
        method: str,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """Fit a multichannel spline PSF across all loaded channels
        simultaneously. The first loaded channel is the reference; its
        identifications are mapped into every channel via the calibration's
        stored transforms."""
        # Affine corrections are single-channel only: this fit registers its
        # channels itself, so applying one on top would correct twice. Say
        # so rather than silently ignoring what the user loaded.
        if self.parameters_dialog.lateral_transforms:
            QtWidgets.QMessageBox.information(
                self,
                "Multichannel spline fit",
                "Affine corrections apply to single-channel data only and "
                "are not used here: the multichannel fit registers its "
                "channels itself, from the transforms in its calibration.",
            )
        if calibration.get("split_fov"):
            self._start_split_fov_spline_fit(calibration, method, eps, max_it)
            return
        n_channels = int(calibration.get("n_channels", 0))
        # persist the displayed channel's identifications, so the per-channel
        # tables read below are complete even if no channel switch happened
        # since the last Identify run
        self._snapshot_current_channel()
        if len(self.channels) < n_channels:
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel spline fit",
                f"This calibration expects {n_channels} channels, but "
                f"{len(self.channels)} are loaded. Load them with "
                "'File > Open channels from several movies' in the same "
                "order as the calibration.",
            )
            self.status_bar.showMessage("")
            return
        reference = self.channels[0]
        if reference.identifications is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Multichannel spline fit",
                "Identify spots in the reference channel (the first loaded "
                "channel) before running a multichannel spline fit.",
            )
            self.status_bar.showMessage("")
            return
        movies = [self.channels[c].movie for c in range(n_channels)]
        # Use each channel's own camera info when available (needed for correct
        # per-channel photon conversion); fall back to the shared one.
        camera_infos = [
            getattr(self.channels[c], "camera_info", None) or self.camera_info
            for c in range(n_channels)
        ]
        # Use only linked identifications
        ids_per_channel = [
            getattr(self.channels[c], "identifications", None)
            for c in range(n_channels)
        ]
        # Identifications made on the channel sum are the linked set already
        # (see Window.identify_channel_sum), so they go into the fit as they
        # are - requiring a detection in every channel on top of that would
        # drop precisely the dim-channel molecules the sum recovered.
        link_identifications = self.sum_identifications is None
        n_missing = sum(
            1 for ids in ids_per_channel[1:] if ids is None or len(ids) == 0
        )
        if n_missing and link_identifications:
            QtWidgets.QMessageBox.information(
                self,
                "Multichannel spline fit",
                f"{n_missing} of the {n_channels - 1} non-reference channels "
                "have no identifications, so localizations cannot be linked "
                "across all channels. Run 'Analyze > Identify' (Ctrl+I) first - "
                "with several channels loaded it identifies every channel - for "
                "a fully linked fit; continuing with the channels that are "
                "identified.",
            )
        self.fit_worker = MultichannelSplineFitWorker(
            movies,
            camera_infos,
            reference.identifications,
            self.parameters["Box Size"],
            calibration,
            mle=method in ("spline-mle", "spline-mle-gpu"),
            use_gpu=method.endswith("-gpu"),
            link_photons=self.parameters_dialog._link_photons_enabled(),
            identifications_per_channel=ids_per_channel,
            link_identifications=link_identifications,
            eps=eps,
            max_it=max_it,
            camera_calibration=(
                self.parameters_dialog.camera_calibration or None
            ),
        )
        self.fit_worker.linkMade.connect(self.on_link_progress)
        self.fit_worker.linkFinished.connect(self.on_link_finished)
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def _start_split_fov_spline_fit(
        self,
        calibration: dict,
        method: str,
        eps: float | None = None,
        max_it: int | None = None,
    ) -> None:
        """Fit a split-FOV multichannel spline PSF from the single loaded movie.
        The channels are placed at the drawn ROIs when they match the
        calibration's channel count (so a moved split can be re-registered by
        re-drawing), otherwise at the calibration's stored regions. The
        reference region's identifications are mapped into every region via the
        stored inter-channel affine (see ``localize.fit_spline_split_fov``)."""
        if self.identifications is None or len(self.identifications) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV spline fit",
                "Identify spots first (they are confined to the reference "
                "region automatically).",
            )
            self.status_bar.showMessage("")
            return
        n_channels = int(calibration.get("n_channels", 0))
        # use the drawn ROIs to place the channels when they match the channel
        # count (reference first); else fall back to the calibration positions
        regions = None
        if self.view.split_fov_mode and len(self.view.rois) == n_channels:
            regions = [list(map(list, r)) for r in self.view.rois]
        # guard: warn if no identification falls in the reference region (a
        # moved/rescaled split or wrong ROIs would otherwise fit nothing)
        ref_rect = (regions or calibration.get("regions"))[
            0 if regions else int(calibration.get("reference", 0))
        ]
        (ry0, rx0), (ry1, rx1) = ref_rect
        rx = np.asarray(self.identifications["x"], dtype=float)
        ry = np.asarray(self.identifications["y"], dtype=float)
        n_in = int(
            np.count_nonzero(
                (rx >= min(rx0, rx1))
                & (rx < max(rx0, rx1))
                & (ry >= min(ry0, ry1))
                & (ry < max(ry0, ry1))
            )
        )
        if n_in == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Split-FOV spline fit",
                f"{len(self.identifications)} spots were identified but none "
                "fall inside the reference region, so there is nothing to fit. "
                "The reference region is probably in the wrong place for this "
                "data: enable 'Regions = channels' and drag the ROIs onto the "
                "channels (reference first), or run Calibration > Refine "
                "split-FOV registration.",
            )
            self.status_bar.showMessage("")
            return
        self.fit_worker = MultichannelSplineFitWorker(
            [self.movie],
            [self.camera_info],
            self.identifications,
            self.parameters["Box Size"],
            calibration,
            mle=method in ("spline-mle", "spline-mle-gpu"),
            use_gpu=method.endswith("-gpu"),
            split_fov=True,
            regions=regions,
            link_photons=self.parameters_dialog._link_photons_enabled(),
            link_identifications=self.sum_identifications is None,
            eps=eps,
            max_it=max_it,
            camera_calibration=(
                self.parameters_dialog.camera_calibration or None
            ),
        )
        self.fit_worker.linkMade.connect(self.on_link_progress)
        self.fit_worker.linkFinished.connect(self.on_link_finished)
        self.fit_worker.progressMade.connect(self.on_fit_progress)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_worker
        self.abort_action.setEnabled(True)
        self.fit_worker.start()

    def reregister_channels_from_signal(self) -> None:
        """Re-estimate the inter-channel registration of the loaded spline
        calibration directly from the current blinking data.

        Both channel layouts pair shared single-molecule signal frame by frame
        and re-fit the inter-channel affines, updating the loaded calibration;
        this dispatches on the calibration:

        * **Split-FOV** (regions of one movie): the signal is read inside the
          drawn ROIs (or the calibration's regions) and each channel is seeded
          by only the calibration's flip (see
          :func:`spline.refine_split_fov_transforms_from_signal`).
        * **Multichannel** (a separate movie per channel): the loaded channel
          movies are paired frame by frame, seeded from the calibration's
          existing transforms (see
          :func:`spline.refine_multichannel_transforms_from_signal`).

        The frame window and the number of frames sampled from it are asked for
        in :class:`RefineRegistrationDialog` (seeded from the identification
        frame range), since which part of the movie pairs best is
        data-dependent.
        """
        title = "Re-align channels (signal)"
        calibration = self.parameters_dialog.spline_calibration
        if not (calibration and calibration.get("channel_transforms")):
            QtWidgets.QMessageBox.warning(
                self,
                title,
                "Load a multichannel or split-FOV spline calibration first "
                "(Experimental PSF (spline) > Load calibration).",
            )
            return
        split_fov = bool(calibration.get("split_fov"))
        parameters = self.parameters

        # gather the layout-specific inputs and pick the refiner
        if split_fov:
            if self.movie is None:
                QtWidgets.QMessageBox.information(
                    self, title, "No movie loaded."
                )
                return
            n_channels = int(calibration.get("n_channels", 0))
            if self.view.split_fov_mode and len(self.view.rois) == n_channels:
                regions = [list(map(list, r)) for r in self.view.rois]
            else:
                regions = calibration.get("regions")
            if not regions or len(regions) != n_channels:
                QtWidgets.QMessageBox.warning(
                    self,
                    title,
                    f"Draw one ROI per channel (reference first): {n_channels} "
                    "regions are needed for this calibration.",
                )
                return

            n_movie_frames = len(self.movie)
            # per-region thresholds only apply when the regions being refined
            # are the drawn ones; the calibration's own regions get the
            # single slider value
            minimum_ng = parameters["Min. Net Gradient"]
            if not (
                isinstance(minimum_ng, list) and len(minimum_ng) == n_channels
            ):
                minimum_ng = self.parameters_dialog.mng_slider.value()

            def _refine(frame_bounds, max_frames, model):
                return spline.refine_split_fov_transforms_from_signal(
                    self.movie,
                    calibration,
                    regions,
                    minimum_ng=minimum_ng,
                    box=parameters["Box Size"],
                    frame_bounds=frame_bounds,
                    max_frames=max_frames,
                    model=model,
                )

        else:
            n_channels = int(calibration.get("n_channels", len(self.channels)))
            if len(self.channels) < n_channels:
                QtWidgets.QMessageBox.warning(
                    self,
                    title,
                    f"This calibration has {n_channels} channels, but "
                    f"{len(self.channels)} movies are loaded. Load them with "
                    "'File > Open channels from several movies' in the same "
                    "order as the calibration (reference first).",
                )
                return
            # persist the active channel so every channel's movie is current
            self._snapshot_current_channel()
            movies = [self.channels[c].movie for c in range(n_channels)]
            # the channels are frame-synchronized, so only frames present in
            # every movie can be paired
            n_movie_frames = min(len(m) for m in movies)

            def _refine(frame_bounds, max_frames, model):
                return spline.refine_multichannel_transforms_from_signal(
                    movies,
                    calibration,
                    minimum_ng=parameters["Min. Net Gradient"],
                    box=parameters["Box Size"],
                    frame_bounds=frame_bounds,
                    max_frames=max_frames,
                    model=model,
                )

        # let the user pick the frames considered: the first frames of a movie
        # are often too dense (or still bleaching) for unambiguous pairing
        (
            frame_bounds,
            max_frames,
            model,
            ok,
        ) = RefineRegistrationDialog.getFrameSpecs(
            self,
            n_movie_frames,
            self.frame_range,
            spline.registration_model_name(calibration),
        )
        if not ok:
            return

        self.status_bar.showMessage("Re-aligning channels from signal ...")
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            _, reg_info = _refine(frame_bounds, max_frames, model)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.status_bar.showMessage("")
            QtWidgets.QMessageBox.critical(
                self, title, f"Re-alignment failed: {e}"
            )
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.status_bar.showMessage("")

        # A channel sum was built with the registration that has just been
        # replaced
        self.drop_channel_sum()
        self._reset_contrast_to_frame()

        # split-FOV: reflect the (possibly re-ordered) regions in the view
        if split_fov:
            self.view.rois = [
                [[int(r[0][0]), int(r[0][1])], [int(r[1][0]), int(r[1][1])]]
                for r in (calibration.get("regions") or [])
            ]
            self.parameters_dialog.update_roi_display()
            self.draw_frame()

        rows = []
        for r in reg_info:
            row = (
                f"ch{r['channel']}: {r['n_matches']} paired signals, "
                f"RMS {r['rms']:.2f} px, {r.get('model', 'affine')}"
            )
            # say so when too few pairs survived for the chosen model, rather
            # than silently registering with something simpler
            requested = r.get("model_requested")
            if requested and requested != r.get("model"):
                row += f" (fell back from {requested})"
            rows.append(row)
        QtWidgets.QMessageBox.information(
            self,
            title,
            "Channels re-aligned from the current signal:\n\n"
            + "\n".join(rows),
        )

    def fit_z(self) -> None:
        """Fit z coordinates of the fitted localizations based on the
        calibration data."""
        self.status_bar.showMessage("Fitting z position...")
        model = self.parameters_dialog.fit_model.currentText()
        optimizer = self.parameters_dialog.fit_optimizer.currentText()
        fitting_method = _fit_code(model, optimizer)
        # zfit only knows gausslq/gaussmle; map the GPU/rotated/avg
        # codes to the corresponding CPU noise model
        fitting_method = (
            "gaussmle" if fitting_method.startswith("gaussmle") else "gausslq"
        )
        self.fit_z_worker = FitZWorker(
            self.locs,
            self.info + [self.camera_info],  # ensure pixel size in info
            self.parameters_dialog.z_calibration,
            self.parameters_dialog.magnification_factor.value(),
            self.parameters_dialog.pixelsize.value(),
            fitting_method,
            self.parameters_dialog.gpu_checkbox.isChecked(),
            lateral_transforms=self._extra_lateral_transforms(
                self.parameters_dialog.z_calibration
            ),
        )
        self.fit_z_worker.progressMade.connect(self.on_fit_z_progress)
        self.fit_z_worker.finished.connect(self.on_fit_z_finished)
        self.fit_z_worker.aborted.connect(self.on_worker_aborted)
        self._active_worker = self.fit_z_worker
        self.abort_action.setEnabled(True)
        self.fit_z_worker.start()

    def _record_applied_lateral(self, worker) -> None:
        """Keep what the finished fit did with the lateral corrections, so
        that ``save_locs`` can write it into the metadata.

        The workers discard the info their fit returns - it is rebuilt from
        the dialog when saving - so without this a corrected file would
        carry no record of the correction, and the CLI's would.
        """
        self._applied_lateral = list(
            getattr(worker, "applied_lateral_transforms", None) or []
        )
        skipped = list(
            getattr(worker, "skipped_lateral_transforms", None) or []
        )
        if skipped:
            self.status_bar.showMessage(
                "Skipped "
                + ", ".join(skipped)
                + ": already applied by the calibration used for fitting."
            )

    def _extra_lateral_transforms(self, calibration: dict | None) -> list:
        """The loaded affine corrections minus those ``calibration`` carries
        and therefore applies itself.

        The load-time check in the Parameters dialog rejects the calibration
        file itself; this also catches a copy of the same transform saved
        under another name, which only a comparison of the matrices finds.
        """
        extra, duplicates = lib.drop_duplicate_lateral_transforms(
            self.parameters_dialog.lateral_transforms, calibration
        )
        if duplicates:
            self.status_bar.showMessage(
                "Skipping "
                + ", ".join(lib.describe_lateral_transforms(duplicates))
                + ": already applied by the calibration used for fitting."
            )
        return extra

    def on_cut_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the spot cutting progress."""
        message = f"Extracting spot {curr:,} / {total:,} ..."
        self.status_bar.showMessage(message)

    def _link_target_name(self) -> str:
        """'regions' for a split-FOV fit (the channels are regions of the one
        loaded movie), 'channels' otherwise."""
        worker = getattr(self, "fit_worker", None)
        return "regions" if getattr(worker, "split_fov", False) else "channels"

    def on_link_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the cross-channel linking progress."""
        self.status_bar.showMessage(
            f"Linking spots across {self._link_target_name()}: "
            f"{curr:,} / {total:,} ..."
        )

    def on_link_finished(self, n_kept: int, n_total: int) -> None:
        """Report how many detections survived the cross-channel linking."""
        pct = 100 * n_kept / n_total if n_total else 0.0
        self.status_bar.showMessage(
            f"{n_kept:,} of {n_total:,} spots ({pct:.1f}%) linked across all "
            f"{self._link_target_name()}; fitting those ..."
        )

    def on_fit_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the fitting progress."""
        worker = getattr(self, "fit_worker", None)
        if isinstance(worker, _MULTICHANNEL_FIT_WORKERS):
            # extraction, GPU fit and per-spot CRLB share this callback
            model = (
                "spline"
                if isinstance(worker, MultichannelSplineFitWorker)
                else "Gaussian"
            )
            message = f"Fitting multichannel {model}: {curr:,} / {total:,} ..."
            self.status_bar.showMessage(message)
        elif getattr(worker, "method", "").endswith("-gpu") and getattr(
            worker, "method", ""
        ).startswith("spline"):
            # The GPU spline fit is one launch per chunk, so this callback
            # only ever reports the per-spot CRLB pass that follows it. The
            # CPU spline reports the fit itself and falls through to the
            # generic per-spot message below.
            self.status_bar.showMessage("Calculating localization precision")
        elif self.parameters_dialog.gpu_checkbox.isChecked():
            self.status_bar.showMessage("Fitting spots on the GPU...")
        else:
            message = f"Fitting spot {curr:,} / {total:,} ..."
            self.status_bar.showMessage(message)

    def on_fit_finished(
        self,
        locs: pd.DataFrame,
        elapsed_time: float,
        fit_z: bool,
        calibrate_z: bool,
    ) -> None:
        """Handle the completion of the fitting process. Draw fit
        markers, fit/calibration z coordinates, if requested, save
        localizations."""
        self._record_applied_lateral(self._active_worker)
        self._active_worker = None
        self.abort_action.setEnabled(False)
        self.status_bar.showMessage(
            f"Fitted {len(locs):,} spots in {elapsed_time:.2f} seconds."
        )
        self.locs = locs
        self.locs_display = locs
        self._distribute_multichannel_fit(locs)
        self.draw_frame()
        # sound notification
        if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
            sound_path = lib.get_sound_notification_path()
            if sound_path is not None:
                playsound(sound_path, block=False)
        base = self.channel_output_base()
        if calibrate_z:
            # restore the GPU checkbox state for the selected model
            self.parameters_dialog.on_fit_optimizer_changed()
            step, frames_per_step, frame_order, ok = (
                Calibrate3DDialog.getCalibrationSpecs(self)
            )
            if ok:
                base = self.channel_output_base()
                out_path = base + "_3d_calib.yaml"
                path, exe = lib.get_save_filename_ext_dialog(
                    self, "Save 3D calibration", out_path, filter="*.yaml"
                )
                if path:
                    t0 = time.time()
                    zfit.calibrate_z(
                        locs,
                        self.info,
                        step,
                        self.parameters_dialog.magnification_factor.value(),
                        path=path,
                        frame_bounds=self.frame_range,
                        frames_per_step=frames_per_step,
                        frame_order=frame_order,
                    )
                    dt = time.time() - t0
                    if dt > lib.SOUND_NOTIFICATION_DURATION:
                        sound_path = lib.get_sound_notification_path()
                        if sound_path is not None:
                            playsound(sound_path, block=False)
                    self.status_bar.showMessage(
                        f"3D calibrated in {dt:.2f} seconds."
                    )
        else:
            if fit_z:
                self.fit_z()
            else:
                self.save_locs_after_fit()

    def _locs_in_channel(
        self, locs: pd.DataFrame, transform: np.ndarray
    ) -> pd.DataFrame:
        """Map reference-channel localizations into another channel's pixel
        coordinates via the calibration's transform, for cross-channel
        display."""
        xy = transforms_mod.from_dict(transform).apply(
            np.column_stack(
                [
                    np.asarray(locs["x"], dtype=np.float64),
                    np.asarray(locs["y"], dtype=np.float64),
                ]
            )
        )
        mapped = locs.copy()
        mapped["x"] = xy[:, 0].astype(np.float32)
        mapped["y"] = xy[:, 1].astype(np.float32)
        return mapped

    def _distribute_multichannel_fit(self, locs: pd.DataFrame) -> bool:
        """Show a multichannel spline fit on every loaded channel.

        The fit is in the reference channel's (channel 0) coordinates. Each
        other channel's ``locs_display`` gets the same localizations mapped
        into its own pixel frame via the calibration's ``channel_transforms``,
        so switching channels overlays the fit on that channel's movie. The
        fit itself (saveable ``locs``) stays with the reference channel, which
        is made active so it displays and saves in reference coordinates.

        Returns False (a no-op) for single-movie data, including split-FOV,
        and for channels fitted independently - those localizations belong to
        the channel they were fitted in, not to a reference channel.
        """
        if len(self.channels) <= 1 or self.fit_mode() == FIT_MODE_SEPARATE:
            return False
        # Read the transforms off the worker that actually produced these
        # localizations, rather than whichever calibration happens to be
        # loaded - both a spline calibration and a channel registration can be
        # loaded at once, and only one of them ran.
        worker = getattr(self, "fit_worker", None)
        calibration = getattr(worker, "calibration", None) or getattr(
            worker, "channel_registration", None
        )
        transforms = (calibration or {}).get("channel_transforms")
        for c, channel in enumerate(self.channels):
            if c == 0 or not transforms or c >= len(transforms):
                channel.locs_display = locs
            else:
                channel.locs_display = self._locs_in_channel(
                    locs, transforms[c]
                )
        self.channels[0].locs = locs
        if self.current_channel != 0:
            # the fit belongs to the reference channel: activate it so the
            # overlay lands on the reference movie and saving uses reference
            # coordinates. Restore directly (no snapshot) - the just-set
            # per-channel locs_display would otherwise be clobbered by the
            # stale flat state.
            self.current_channel = 0
            self._populate_channel_combo()
            self._restore_current_channel()
        return True

    def on_fit_z_progress(self, curr: int, total: int) -> None:
        """Update the status bar with the fitting progress."""
        message = f"Fitting z coordinate {curr:,} / {total:,} ..."
        self.status_bar.showMessage(message)

    def on_fit_z_finished(
        self,
        locs: pd.DataFrame,
        elapsed_time: float,
    ) -> None:
        """Handle the completion of the z fitting process.

        ``self.locs_display`` is left untouched so the on-screen
        FitMarker overlays stay at their pre-z-fit, pre-affine
        positions (mirroring the drift-correction pattern)."""
        self._record_applied_lateral(self._active_worker)
        self._active_worker = None
        self.abort_action.setEnabled(False)
        self.status_bar.showMessage(
            f"Fitted {len(locs):,} z coordinates in {elapsed_time:.2f} "
            "seconds."
        )
        self.locs = locs
        self.save_locs_after_fit()
        # sound notification
        if elapsed_time > lib.SOUND_NOTIFICATION_DURATION:
            sound_path = lib.get_sound_notification_path()
            if sound_path is not None:
                playsound(sound_path, block=False)

    def attach_roi_id(self) -> None:
        """Name the ROI each localization was found in, adding a
        ``roi_id`` column to ``self.locs`` (see
        ``localize.add_roi_id``).

        Called once per fit, before drift correction moves the
        coordinates the ids are derived from. Split-FOV mode is
        excluded: there the regions are separate channels rather than
        parts of one field, and each is saved to its own file already."""
        if self.locs is None or self.view.split_fov_mode:
            return
        rois = (self.last_identification_info or {}).get("ROI")
        self.locs = localize.add_roi_id(self.locs, rois)

    def save_locs_after_fit(self) -> None:
        """Save localizations after fitting to an .hdf5 file, naming the
        ROI each of them came from (see ``attach_roi_id``)."""
        self.attach_roi_id()
        base = self.channel_output_base()
        self.save_locs(base + "_locs.hdf5")

        if not self.parameters_dialog.quality_check.isEnabled():
            self.parameters_dialog.quality_check.setEnabled(True)

        # restore the GPU checkbox state for the selected model
        self.parameters_dialog.on_fit_optimizer_changed()
        self.status_bar.showMessage(f"Saved {len(self.locs):,} localizations.")

        # apply drift if requested
        aim_check = self.parameters_dialog.aim_undrift_checkbox.isChecked()
        aim_segmentation = self.parameters_dialog.aim_segmentation.value()
        fiducial_check = self.parameters_dialog.fiducial_check.isChecked()
        if aim_check or fiducial_check:
            self.drift_correction(aim_check, aim_segmentation, fiducial_check)
        # this channel is done with, drift correction included: on to the
        # next one (see _fit_independent)
        if self._multi_fit is not None:
            self._multi_fit["done"] += 1
            self._multi_fit["sum"] += len(self.locs)
            self._fit_run_next()

    def drift_correction(
        self, aim_check: bool, aim_segmentation: int, fiducial_check: bool
    ) -> None:
        """Apply drift correction to the fitted localizations and save
        the drift-corrected localizations and the .txt drift files.

        ``self.locs_display`` is left untouched so the on-screen
        FitMarker overlays stay at their pre-drift positions."""
        drift = None

        if aim_check:
            self.status_bar.showMessage("Applying AIM drift correction...")
            undrift_locs, new_info, drift = aim.aim(
                self.locs,
                self.extra_info,
                aim_segmentation,
                progress=None,
            )
            self.locs = undrift_locs
            self.extra_info = new_info
            self.status_bar.showMessage("AIM finished.")

        if fiducial_check:
            self.status_bar.showMessage(
                "Finding fiducials for drift correction..."
            )
            fiducial_picks, box = imageprocess.find_fiducials(
                self.locs, self.extra_info
            )
            if len(fiducial_picks) == 0:
                if drift is not None:
                    # save the AIM drift-corrected localizations
                    base = self.channel_output_base()
                    self.save_locs(base + "_locs_undrifted.hdf5")
                    # save txt drift file
                    np.savetxt(base + "_locs_drift.txt", drift, newline="\r\n")
                    self.status_bar.showMessage(
                        "Saved AIM drift-corrected localizations. No "
                        "fiducials found, only AIM drift correction "
                        "applied."
                    )
                else:
                    self.status_bar.showMessage(
                        "No fiducials found. Skipping fiducial-based "
                        "drift correction."
                    )
                return
            self.status_bar.showMessage(
                f"Found {len(fiducial_picks)} fiducials. Applying drift "
                "correction..."
            )
            self.locs, self.extra_info, fiducials_drift = (
                postprocess.undrift_from_fiducials(
                    locs=self.locs,
                    info=self.extra_info,
                    picks=fiducial_picks,
                    pick_size=box,
                )
            )
            if drift is not None:
                drift["x"] += fiducials_drift["x"]
                drift["y"] += fiducials_drift["y"]
                if "z" in fiducials_drift.columns:
                    drift["z"] += fiducials_drift["z"]
            else:
                drift = fiducials_drift
            self.status_bar.showMessage(
                "Fiducial-based drift correction finished."
            )
        # save the drift-corrected localizations
        base = self.channel_output_base()
        self.save_locs(base + "_locs_undrifted.hdf5")
        # save txt drift file
        np.savetxt(base + "_locs_drift.txt", drift, newline="\r\n")
        self.status_bar.showMessage(
            "Saved drift-corrected localizations and drift file."
        )

    def fit_in_view(self) -> None:
        """Reset the zoom in the scene."""
        rectangle = QtCore.QRectF(
            0,
            0,
            self.movie.shape[2],
            self.movie.shape[1],
        )
        self.view.fitInView(
            rectangle, QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.draw_frame()

    def zoom_in(self) -> None:
        """Zoom in the view."""
        self.zoom(10 / 7)

    def zoom_out(self) -> None:
        """Zoom out the view."""
        self.zoom(7 / 10)

    def zoom(self, factor: float, anchor: QtCore.QPoint | None = None) -> None:
        """Zoom in or out the view by a specific factor. Anchor can
        specify the cursor position, otherwise zooms to/from the
        viewport's center."""
        if not hasattr(self, "movie") or self.movie is None:
            return
        # do not allow zooming out too much
        if factor < 1:
            rect = self.view.viewport().rect()
            visible_scene_rect = self.view.mapToScene(rect).boundingRect()
            if visible_scene_rect.width() / factor > self.movie.shape[2]:
                self.fit_in_view()
                return
        # fall back to the viewport center if no anchor is given
        # (e.g. zoom in/out from the menu or keyboard shortcuts)
        if anchor is None:
            anchor = self.view.viewport().rect().center()
        # adjust the transform directly so the scene point under the
        # anchor stays fixed (NoAnchor avoids scrollbar rounding drift)
        self.view.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor
        )
        old_scene_pos = self.view.mapToScene(anchor)
        self.view.scale(factor, factor)
        new_scene_pos = self.view.mapToScene(anchor)
        delta = new_scene_pos - old_scene_pos
        self.view.translate(delta.x(), delta.y())

        self.draw_frame()

    def save_spots(self, path: str) -> None:
        """Save identified spots as an .hdf5 file."""
        if self.identifications is None:
            message = (
                "No identifications to save. Please run identification first."
            )
            QtWidgets.QMessageBox.warning(self, "No identifications", message)
            return
        box = self.parameters["Box Size"]
        # raw movie on purpose: spots must carry real photon counts, so the
        # identification filters (a detection aid) never apply
        spots = localize.get_spots(
            self.movie, self.identifications, box, self.camera_info
        )
        info = io.strip_mm_metadata(self.info) + [
            self.last_identification_info | self.camera_info
        ]
        info_path = os.path.splitext(path)[0] + ".yaml"
        if path.endswith(".npy"):
            np.save(path, spots)
            io.save_info(info_path, info)
        elif path.endswith(".tif"):
            imageio.mimwrite(path, spots.astype("float32"))
            io.save_info(info_path, info)

    def save_spots_dialog(self) -> None:
        """Get the path for saving identified spots."""
        if self.movie_path != []:
            base = self.channel_output_base()
            path = base + "_spots.tif"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save spots",
                path,
                filter="*.tif;;*.npy",
                check_ext=".yaml",
            )
            if path:
                self.save_spots(path)

    def export_current(self) -> None:
        """Export current view as .png or .tif."""
        try:
            base = self.channel_output_base()
        except AttributeError:
            return
        out_path = base + "_view.png"
        path, ext = lib.get_save_filename_ext_dialog(
            self, "Save image", out_path, filter="*.png;;*.tif"
        )
        if path:
            visible_scene_rect = self.view.mapToScene(
                self.view.viewport().rect()
            ).boundingRect()
            scene_rect = visible_scene_rect.intersected(
                self.scene.itemsBoundingRect()
            )
            scale = self.view.transform().m11()
            size = QtCore.QSize(
                max(1, int(round(scene_rect.width() * scale))),
                max(1, int(round(scene_rect.height() * scale))),
            )
            qimage = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
            qimage.fill(QtGui.QColor("transparent"))
            painter = QtGui.QPainter(qimage)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.scene.render(
                painter,
                QtCore.QRectF(qimage.rect()),
                scene_rect,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            )
            painter.end()
            qimage.save(path)
        self.view.setMinimumSize(1, 1)

    def save_locs(self, path: str) -> None:
        """Save localizations and their metadata."""
        # An unset identification info must not cost the user the fit that was
        # just run: the rest of the metadata is written either way.
        localize_info = (self.last_identification_info or {}).copy()
        localize_info["Generated by"] = f"Picasso v{__version__} Localize"
        model = self.parameters_dialog.fit_model.currentText()
        if FIT_MODELS[model]["optimizers"] is None:
            localize_info["Fit method"] = model
        else:
            optimizer = self.parameters_dialog.fit_optimizer.currentText()
            localize_info["Fit method"] = f"{model}, {optimizer}"
        # A channel fitted on its own says so, so that its file can be told
        # apart from one the joint fit produced.
        if self.fit_mode() == FIT_MODE_SEPARATE:
            localize_info["Fit mode"] = FIT_MODE_SEPARATE
            localize_info["Channel"] = self._region_suffix or (
                self.channels[self.current_channel].name
                if self.channels
                else ""
            )
        if self.parameters_dialog.fit_z_checkbox.isChecked():
            localize_info["Z Calibration Path"] = (
                self.parameters_dialog.z_calibration_path
            )
            localize_info["Z Calibration"] = (
                self.parameters_dialog.z_calibration
            )
        # A correction loaded separately from the calibration leaves no
        # other trace in the metadata (the calibration dicts below only
        # record the ones they carry themselves), so name every correction
        # the fit applied, and where the separate ones came from.
        if self._applied_lateral:
            localize_info["Lateral corrections applied"] = list(
                self._applied_lateral
            )
            if self.parameters_dialog.affine_calibration_paths:
                localize_info["Lateral correction paths"] = list(
                    self.parameters_dialog.affine_calibration_paths
                )
        if FIT_MODELS.get(model, {}).get("needs_spline_calibration"):
            # Record the path and model only; the coefficient table is far
            # too large to embed in the metadata.
            localize_info["Spline Calibration Path"] = (
                self.parameters_dialog.spline_calibration_path
            )
            localize_info["Spline Calibration Model"] = (
                self.parameters_dialog.spline_calibration.get("model")
            )
        # ``fit`` records this itself, but its info is discarded here (see
        # FitWorker.run) and this dict is rebuilt from the dialog, so without
        # this line a GUI run leaves no trace of the noise model it used.
        localize_info.update(
            localize.camera_calibration_info(
                self.parameters_dialog.camera_calibration
            )
        )
        # the MicroManager metadata block of the movie is only carried
        # over if the user settings ask for it (it is by default)
        self.extra_info = io.strip_mm_metadata(self.info) + [
            localize_info | self.camera_info
        ]
        self.select_locs_columns()  # save only selected columns
        io.save_locs(path, self.locs, self.extra_info)

    def save_locs_dialog(self) -> None:
        """Get the path to save localizations."""
        if self.movie_path != []:
            base = self.channel_output_base()
            locs_path = base + "_locs.hdf5"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save localizations",
                locs_path,
                filter="*.hdf5",
                check_ext=".yaml",
            )
            if path:
                self.save_locs(path)

    def save_identifications(self, path: str) -> None:
        """Save identifications and their metadata to an HDF5 file."""
        ids_info = {
            "Generated by": f"Picasso v{__version__} Localize",
            "Box Size": self.parameters_dialog.box_spinbox.value(),
            # a list of per-region thresholds for split-FOV data, which
            # load_identifications puts back onto the regions
            "Min. Net Gradient": self.parameters["Min. Net Gradient"],
            "Temporal Median Window": (
                self.parameters["Temporal Median Window"]
            ),
            "Gaussian Filter Sigma": (
                self.parameters["Gaussian Filter Sigma"]
            ),
        }
        info = io.strip_mm_metadata(self.info) + [ids_info]
        io.save_identifications(path, self.identifications, info)

    def save_identifications_dialog(self) -> None:
        """Get the path to save identifications."""
        if self.identifications is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No identifications",
                "No identifications to save. Run identification first.",
            )
            return
        if self.movie_path != []:
            base = self.channel_output_base()
            ids_path = base + "_identifications.hdf5"
            path, exe = lib.get_save_filename_ext_dialog(
                self,
                "Save identifications",
                ids_path,
                filter="*.hdf5",
                check_ext=".yaml",
            )
            if path:
                self.save_identifications(path)

    def localize(self, calibrate_z: bool = False) -> None:
        """Identify and fit, see ``identify`` and ``fit``.

        Parameters
        ----------
        calibrate_z : bool, optional
            Whether to run z-calibration for 3D fitting afterwards
            Default is False.
        """
        self.parameters_dialog.gpu_checkbox.setDisabled(True)
        if (
            calibrate_z
            and self.identifications is not None
            and self.ready_for_fit
            and not self.identifications_outdated()
        ):
            # reuse existing identifications (e.g., loaded picks/locs)
            self.fit(calibrate_z=calibrate_z)
        else:
            self.identify(fit_afterwards=True, calibrate_z=calibrate_z)

    def identifications_outdated(self) -> bool:
        """Check whether the identification settings (box size, min. net
        gradient, ROI, frame range) have changed since
        ``self.identifications`` was created."""
        last = self.last_identification_info
        if last is None:
            return True
        if any(
            last.get(key) != value for key, value in self.parameters.items()
        ):
            return True
        if last.get("ROI") != self.view.rois:
            return True
        if last.get("Frame bounds") != self.frame_range:
            return True
        return False

    def select_locs_columns(self) -> None:
        """Select only the columns that are checked in the corresponding
        dialog.

        The ROI id has no checkbox - only a fit restricted to ROIs has
        it - and is always kept."""
        to_keep = []
        for column, checkbox in self.columns_dialog.column_checkboxes.items():
            if checkbox.isChecked() and column in self.locs.columns:
                to_keep.append(column)
        if (
            localize.ROI_ID_COLUMN in self.locs.columns
            and localize.ROI_ID_COLUMN not in to_keep
        ):
            to_keep.append(localize.ROI_ID_COLUMN)
        self.locs = self.locs[to_keep]


class IdentificationWorker(QtCore.QThread):
    """Identify spots in the movie using multiprocessing.

    Loads the user parameters and updates the status bar about the
    progress."""

    progressMade = QtCore.pyqtSignal(int, dict)
    finished = QtCore.pyqtSignal(
        dict, object, float, pd.DataFrame, bool, bool, bool
    )
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        window: QtWidgets.QMainWindow,
        fit_afterwards: bool,
        calibrate_z: bool,
        calibrate_spline: bool = False,
        *,
        movie: object | None = None,
        rois: list | None = None,
        parameters: dict | None = None,
    ) -> None:
        """``movie`` / ``rois`` / ``parameters`` default to the window's, i.e.
        to identifying the active channel as displayed. The channel-sum
        identification overrides them to identify the summed view instead
        (see ``Window.identify_channel_sum``)."""
        super().__init__()
        self.window = window
        self.movie = window.movie if movie is None else movie
        # copied down to the corners: the worker's ROIs are handed back with
        # the finished identifications and stored as the settings it ran with
        # (see ``_copied_rois``)
        self.rois = _copied_rois(window.view.rois if rois is None else rois)
        self.frame_range = window.frame_range
        if parameters is None:
            parameters = window.parameters
            # this worker identifies one movie, whatever mode the window is in
            parameters = dict(parameters)
            parameters["Identification Mode"] = IDENTIFY_MODE_SEPARATE
        self.parameters = dict(parameters)
        if calibrate_z or calibrate_spline:
            # bead calibration stacks: the beads are static and do not
            # blink, so a temporal median would subtract them away
            self.parameters["Temporal Median Window"] = 0
        self.fit_afterwards = fit_afterwards
        self.calibrate_z = calibrate_z
        self.calibrate_spline = calibrate_spline

    def on_progress(self, frame_number: int) -> None:
        self.progressMade.emit(frame_number, self.parameters)

    def run(self) -> None:
        t0 = time.time()
        # we ignore info since we will merge the metadata from identification
        # as well when saving localizations
        result = localize.identify(
            movie=self.movie,
            minimum_ng=self.parameters["Min. Net Gradient"],
            box=self.parameters["Box Size"],
            roi=self.rois,
            frame_bounds=self.frame_range,
            threaded=True,
            temporal_median_window=self.parameters["Temporal Median Window"],
            gaussian_filter_sigma=self.parameters["Gaussian Filter Sigma"],
            progress_callback=self.on_progress,
            abort_callback=self.isInterruptionRequested,
        )
        if result is None:  # handle aborted process
            self.aborted.emit()
            return
        identifications, info = result
        elapsed_time = time.time() - t0
        self.finished.emit(
            self.parameters,
            self.rois,
            elapsed_time,
            identifications,
            self.fit_afterwards,
            self.calibrate_z,
            self.calibrate_spline,
        )


class FitWorker(QtCore.QThread):
    """Fit single molecules to the identified spots using
    multiprocessing and update the status bar accordingly."""

    progressMade = QtCore.pyqtSignal(int, int)
    cutProgressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float, bool, bool)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        movie: np.memmap,
        movie_info: list[dict],
        camera_info: dict,
        identifications: pd.DataFrame,
        box: int,
        method: Literal[
            "gausslq",
            "gausslq-spherical",
            "gausslq-rotated",
            "gausslq-rotated-gpu",
            "gausslq-spherical-gpu",
            "gaussmle",
            "gaussmle-spherical",
            "gaussmle-rotated-gpu",
            "gaussmle-spherical-gpu",
            "spline",
            "spline-mle",
            "spline-gpu",
            "spline-mle-gpu",
            "avg",
        ],
        eps: float | None,
        max_it: int | None,
        fit_z: bool,
        calibrate_z: bool,
        use_gpu: bool,
        spline_calibration: dict | None = None,
        lateral_transforms: list | None = None,
        camera_calibration: dict | None = None,
    ) -> None:
        super().__init__()
        self.movie = movie
        self.movie_info = movie_info
        self.camera_info = camera_info
        self.identifications = identifications
        self.box = box
        self.eps = eps
        self.max_it = max_it
        self.fit_z = fit_z
        self.calibrate_z = calibrate_z
        self.spline_calibration = spline_calibration
        self.lateral_transforms = lateral_transforms or []
        # What the fit did with them, for the metadata (see
        # Window._record_applied_lateral). A 3D astigmatism run leaves this
        # empty: FitZWorker applies them after the z fit and reports there.
        self.applied_lateral_transforms = []
        self.camera_calibration = camera_calibration
        self.N = len(identifications)
        self._last_cut_emit = 0
        self.method = _effective_fit_code(method, use_gpu)

    def on_progress(self, n_done: int) -> None:
        self.progressMade.emit(n_done, self.N)

    def on_cut_progress(self, n_done: int) -> None:
        # The underlying cut loop may call back very frequently (e.g. once
        # per frame), so throttle GUI updates to ~1% increments to avoid
        # flooding the main thread's event queue. Always emit the last one.
        step = max(1, self.N // 1000)
        if n_done - self._last_cut_emit >= step or n_done >= self.N:
            self._last_cut_emit = n_done
            self.cutProgressMade.emit(n_done, self.N)

    def run(self) -> None:
        t0 = time.time()
        # we ignore info since we will merge the metadata from identification
        # as well when saving localizations
        try:
            locs, info = localize.fit(
                movie=self.movie,
                camera_info=self.camera_info,
                identifications=self.identifications,
                box=self.box,
                fitting_method=self.method,
                eps=self.eps,
                max_it=self.max_it,
                spline_calibration=self.spline_calibration,
                camera_calibration=self.camera_calibration,
                multiprocess=True,
                progress_callback=self.on_progress,
                abort_callback=self.isInterruptionRequested,
                cut_progress_callback=self.on_cut_progress,
            )
        except Exception as e:
            # letting it escape run() kills the thread without a word - the
            # window would keep waiting for a fit that will never arrive
            print(f"Fit failed: {e}")
            self.aborted.emit()
            return
        if locs is None:  # handle aborted process
            self.aborted.emit()
            return
        if not self.fit_z:
            # With a z fit to follow, FitZWorker applies them after it, so
            # that they land on the final coordinates either way.
            locs = lib.apply_lateral_transforms(locs, self.lateral_transforms)
            self.applied_lateral_transforms = lib.describe_lateral_transforms(
                self.spline_calibration
            ) + lib.describe_lateral_transforms(self.lateral_transforms)
        self.progressMade.emit(self.N + 1, self.N)
        dt = time.time() - t0
        self.finished.emit(locs, dt, self.fit_z, self.calibrate_z)


class MultichannelSplineFitWorker(QtCore.QThread):
    """Fit a multichannel cubic-spline PSF across several registered channels
    simultaneously. The reference channel's identifications are mapped into
    every channel via the calibration's stored affine transforms."""

    progressMade = QtCore.pyqtSignal(int, int)
    linkMade = QtCore.pyqtSignal(int, int)
    linkFinished = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float, bool, bool)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        movies: list,
        camera_infos: list,
        identifications: pd.DataFrame,
        box: int,
        calibration: dict,
        mle: bool = False,
        split_fov: bool = False,
        regions: list | None = None,
        link_photons: bool = True,
        identifications_per_channel: list | None = None,
        link_identifications: bool = True,
        use_gpu: bool | None = None,
        eps: float | None = None,
        max_it: int | None = None,
        camera_calibration: dict | None = None,
    ) -> None:
        super().__init__()
        self.movies = movies
        self.camera_infos = camera_infos
        # One loaded calibration, applied to every channel. Split-FOV really
        # is one sensor; separate cameras per channel would each need their
        # own maps, which the GUI has no way to load yet - the API
        # (``camera_calibrations``) takes a per-channel list.
        self.camera_calibration = camera_calibration
        self.camera_calibrations = (
            None
            if camera_calibration is None
            else [camera_calibration] * len(movies)
        )
        self.identifications = identifications
        self.identifications_per_channel = identifications_per_channel
        # Identifications made on the channel sum are already the
        # cross-channel consensus (every one of them was found in the summed
        # signal of all channels), so pairing them against per-channel
        # detections again would throw away exactly the dim-channel molecules
        # the sum recovered.
        self.link_identifications = link_identifications
        self.box = box
        self.calibration = calibration
        self.mle = mle
        self.use_gpu = use_gpu
        self.eps = eps
        self.max_it = max_it
        # Link photons across channels (shared amplitude, model 11). When False
        # and the calibration has 2 to 6 channels, fit the photon-decoupled
        # link-XYZ model: per-channel
        # free photons/background
        self.link_photons = link_photons
        # Split-FOV: ``movies``/``camera_infos`` hold a single entry (one loaded
        # movie); the channels are regions of that movie, handled by
        # ``fit_spline_split_fov`` (which confines to the reference region).
        # ``regions`` (optional) places the channels at the current ROIs.
        self.split_fov = split_fov
        self.regions = regions
        self.N = len(identifications)

    def on_progress(self, n_done: int) -> None:
        self.progressMade.emit(n_done, self.N)

    def on_link_progress(self, n_done: int) -> None:
        self.linkMade.emit(n_done, self.N)

    def _link_across_channels(self, n_channels: int) -> bool:
        """Restrict the reference detections to those found in every channel.
        Returns False if nothing survives (the caller then aborts).

        ``self.N`` - the denominator of the fit progress - is set to the number
        of spots that are actually fitted, i.e. the linked molecules (one fit
        per molecule across all channels), not the raw detection count.

        Does nothing when the identifications come from the channel sum
        (``link_identifications=False``): they are the linked set already."""
        if not self.link_identifications:
            return True
        if self.split_fov:
            return self._link_across_regions()
        ids_per_channel = self.identifications_per_channel
        if not ids_per_channel or len(ids_per_channel) < 2:
            return True
        ids_per_channel = list(ids_per_channel[:n_channels])
        transforms = self.calibration.get("channel_transforms")
        if not transforms or len(transforms) < len(ids_per_channel):
            return True
        linked, n_kept, n_total = localize.filter_linked_identifications(
            ids_per_channel,
            transforms,
            self.box,
            progress_callback=self.on_link_progress,
        )
        self.linkFinished.emit(n_kept, n_total)
        if n_kept == 0:
            return False
        self.identifications = linked
        self.N = len(linked)
        return True

    def _link_across_regions(self) -> bool:
        """Split-FOV counterpart of :meth:`_link_across_channels`: the single
        identification table holds every region's detections, so it is split by
        region and linked across them. Keeps the reference-region detections
        found in all other regions - one fitted spot per molecule."""
        all_regions = self.identifications
        try:
            fit_regions, reference, _ = localize.split_fov_fit_geometry(
                self.calibration, self.regions
            )
        except (ValueError, KeyError):
            # geometry unavailable (e.g. no channel_registration to re-place at
            # drawn ROIs): fit as before, on the whole identification table
            return True
        # only the reference region is fitted (the other regions are cut from
        # it via the transforms), so it - not the movie-wide detection count -
        # is the denominator of the linking and fit progress
        self.identifications = localize.confine_to_region(
            all_regions, fit_regions[reference]
        )
        self.N = len(self.identifications)
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(
                all_regions,
                self.calibration,
                self.box,
                regions=self.regions,
                progress_callback=self.on_link_progress,
            )
        )
        self.linkFinished.emit(n_kept, n_total)
        if n_kept == 0:
            return False
        self.identifications = linked
        self.N = len(linked)
        return True

    def run(self) -> None:
        t0 = time.time()
        try:
            n_channels = int(
                self.calibration.get("n_channels", len(self.movies))
            )
            if not self._link_across_channels(n_channels):
                where = "regions" if self.split_fov else "channels"
                print(
                    f"Multichannel spline fit: no detection is linked across "
                    f"all {where} - check the channel registration."
                )
                self.aborted.emit()
                return
            if self.split_fov:
                locs = localize.fit_spline_split_fov(
                    self.movies[0],
                    self.camera_infos[0],
                    self.identifications,
                    self.box,
                    self.calibration,
                    regions=self.regions,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    link_photons=self.link_photons,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                    camera_calibration=self.camera_calibration,
                )
            elif not self.link_photons and (
                2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS
            ):
                # Photon decoupling (globLoc link-XYZ): free per-channel photons
                # and background, shared x/y/z. Supersedes the ratiometric scan.
                locs = localize.fit_spline_multichannel(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.calibration,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    link_photons=False,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                    camera_calibrations=self.camera_calibrations,
                )
            elif self.calibration.get("photon_ratios") is not None:
                # Ratiometric color assignment: the calibration carries
                # candidate per-channel photon ratios (one per dye/color). Each
                # localization is assigned the max-likelihood ratio as `color`.
                locs = localize.fit_spline_multichannel_ratiometric(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.calibration,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                    camera_calibrations=self.camera_calibrations,
                )
            else:
                locs = localize.fit_spline_multichannel(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.calibration,
                    mle=self.mle,
                    use_gpu=self.use_gpu,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                    camera_calibrations=self.camera_calibrations,
                )
        except Exception as e:
            print(f"Multichannel spline fit failed: {e}")
            self.aborted.emit()
            return
        self.progressMade.emit(self.N + 1, self.N)
        dt = time.time() - t0
        # fit_z / calibrate_z are always False for the multichannel spline
        # path (z comes from the fit; there is no astigmatism step)
        self.finished.emit(locs, dt, False, False)


class MultichannelGaussianFitWorker(QtCore.QThread):
    """Fit a spherical Gaussian jointly across several registered channels.

    The Gaussian counterpart of :class:`MultichannelSplineFitWorker`, and
    deliberately the same shape - same signals, same cross-channel linking - so
    it plugs into the window's existing fit slots. It needs no PSF calibration,
    only a channel registration (see :mod:`picasso.registration`)."""

    progressMade = QtCore.pyqtSignal(int, int)
    linkMade = QtCore.pyqtSignal(int, int)
    linkFinished = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float, bool, bool)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        movies: list,
        camera_infos: list,
        identifications: pd.DataFrame,
        box: int,
        channel_registration: dict,
        mle: bool = False,
        link_photons: bool = False,
        identifications_per_channel: list | None = None,
        link_identifications: bool = True,
        use_gpu: bool = False,
        eps: float | None = None,
        max_it: int | None = None,
        camera_calibration: dict | None = None,
        split_fov: bool = False,
        regions: list | None = None,
    ) -> None:
        """Set up a joint fit of every channel.

        Parameters
        ----------
        movies : list
            One movie per channel, reference first, in the registration's own
            channel order. For split field of view, the single loaded movie.
        camera_infos : list
            One camera-info dict per channel, for the photon conversion.
        identifications : pd.DataFrame
            Detections in the reference channel; the set that will be fitted.
        box : int
            Box side length (camera pixels).
        channel_registration : dict
            A calibration carrying ``channel_transforms``, from
            :mod:`picasso.registration`.
        mle : bool, optional
            Use the Poisson maximum-likelihood estimator. Default False.
        link_photons : bool, optional
            Share one photon count and background across the channels. Default
            False - see :func:`picasso.localize.fit_gauss_multichannel` for why
            the decoupled model is the safe default here.
        identifications_per_channel : list, optional
            Each channel's own detections, used to keep only the molecules
            found in every channel. Entries may be None for a channel that was
            not identified.
        link_identifications : bool, optional
            Whether to do that cross-channel filtering at all. False when the
            detections came from the channel sum, which is the linked set
            already. Default True.
        use_gpu : bool, optional
            Fit on a CUDA GPU. Default False.
        eps, max_it : optional
            Convergence criterion and iteration cap; None uses the method's own
            schedule.
        camera_calibration : dict, optional
            One per-pixel sCMOS calibration, applied to every channel.
        split_fov : bool, optional
            The channels are regions of one movie rather than separate movies.
            Default False.
        regions : list, optional
            Split-FOV only: the channel ROIs in use, reference first. None uses
            the registration's own.
        """
        super().__init__()
        self.movies = movies
        self.camera_infos = camera_infos
        # One loaded calibration applied to every channel, as for the spline
        # worker; the API takes a per-channel list.
        self.camera_calibration = camera_calibration
        self.camera_calibrations = (
            None
            if camera_calibration is None
            else [camera_calibration] * len(movies)
        )
        self.identifications = identifications
        self.identifications_per_channel = identifications_per_channel
        self.link_identifications = link_identifications
        self.box = box
        self.channel_registration = channel_registration
        self.mle = mle
        self.use_gpu = use_gpu
        self.eps = eps
        self.max_it = max_it
        self.link_photons = link_photons
        # Split-FOV: ``movies`` holds a single entry (one loaded movie); the
        # channels are regions of it, handled by ``fit_gauss_split_fov``.
        # ``regions`` (optional) places the channels at the current ROIs.
        self.split_fov = split_fov
        self.regions = regions
        self.N = len(identifications)

    def on_progress(self, n_done: int) -> None:
        """Report fit progress to the window.

        Parameters
        ----------
        n_done : int
            Spots fitted so far, out of ``self.N``.
        """
        self.progressMade.emit(n_done, self.N)

    def on_link_progress(self, n_done: int) -> None:
        """Report cross-channel linking progress to the window.

        Parameters
        ----------
        n_done : int
            Reference detections checked so far, out of ``self.N``.
        """
        self.linkMade.emit(n_done, self.N)

    def _link_across_channels(self, n_channels: int) -> bool:
        """Restrict the reference detections to those found in every channel.

        As for the spline worker: one fit per molecule across all channels, so
        ``self.N`` becomes the number of linked molecules.

        Parameters
        ----------
        n_channels : int
            Number of channels the registration describes.

        Returns
        -------
        ok : bool
            False if nothing survives the linking, so the caller aborts rather
            than fitting an empty set.
        """
        if not self.link_identifications:
            return True
        if self.split_fov:
            return self._link_across_regions()
        ids_per_channel = self.identifications_per_channel
        if not ids_per_channel or len(ids_per_channel) < 2:
            return True
        ids_per_channel = list(ids_per_channel[:n_channels])
        transforms = self.channel_registration.get("channel_transforms")
        if not transforms or len(transforms) < len(ids_per_channel):
            return True
        linked, n_kept, n_total = localize.filter_linked_identifications(
            ids_per_channel,
            transforms,
            self.box,
            progress_callback=self.on_link_progress,
        )
        self.linkFinished.emit(n_kept, n_total)
        if n_kept == 0:
            return False
        self.identifications = linked
        self.N = len(linked)
        return True

    def _link_across_regions(self) -> bool:
        """Split-FOV counterpart of :meth:`_link_across_channels`.

        One identification table holds every region's detections, so it is
        split by region and linked across them, keeping the reference-region
        detections found in all the others - one fitted spot per molecule.

        Returns
        -------
        ok : bool
            False if nothing links across the regions.
        """
        all_regions = self.identifications
        try:
            fit_regions, reference, _ = localize.split_fov_fit_geometry(
                self.channel_registration, self.regions
            )
        except (ValueError, KeyError):
            # geometry unavailable: fit the whole identification table, as
            # before
            return True
        # only the reference region is fitted (the others are cut from it via
        # the transforms), so it - not the movie-wide count - is the
        # denominator of the linking and fit progress
        self.identifications = localize.confine_to_region(
            all_regions, fit_regions[reference]
        )
        self.N = len(self.identifications)
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(
                all_regions,
                self.channel_registration,
                self.box,
                regions=self.regions,
                progress_callback=self.on_link_progress,
            )
        )
        self.linkFinished.emit(n_kept, n_total)
        if n_kept == 0:
            return False
        self.identifications = linked
        self.N = len(linked)
        return True

    def run(self) -> None:
        """Link the channels, then fit them jointly.

        The ``QThread`` entry point; emits ``finished`` with the localizations
        or ``aborted`` if nothing could be linked, the fit was stopped, or it
        raised.
        """
        t0 = time.time()
        try:
            n_channels = int(
                self.channel_registration.get("n_channels", len(self.movies))
            )
            if not self._link_across_channels(n_channels):
                where = "regions" if self.split_fov else "channels"
                print(
                    "Multichannel Gaussian fit: no detection is linked across "
                    f"all {where} - check the channel registration."
                )
                self.aborted.emit()
                return
            if self.split_fov:
                locs = localize.fit_gauss_split_fov(
                    self.movies[0],
                    self.camera_infos[0],
                    self.identifications,
                    self.box,
                    self.channel_registration,
                    regions=self.regions,
                    mle=self.mle,
                    link_photons=self.link_photons,
                    use_gpu=self.use_gpu,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                    abort_callback=self.isInterruptionRequested,
                    camera_calibration=self.camera_calibration,
                )
            else:
                locs = localize.fit_gauss_multichannel(
                    self.movies,
                    self.camera_infos,
                    self.identifications,
                    self.box,
                    self.channel_registration,
                    mle=self.mle,
                    link_photons=self.link_photons,
                    use_gpu=self.use_gpu,
                    tolerance=self.eps,
                    max_iterations=self.max_it,
                    progress_callback=self.on_progress,
                    abort_callback=self.isInterruptionRequested,
                    camera_calibrations=self.camera_calibrations,
                )
        except Exception as e:
            print(f"Multichannel Gaussian fit failed: {e}")
            self.aborted.emit()
            return
        if locs is None:
            self.aborted.emit()
            return
        self.progressMade.emit(self.N + 1, self.N)
        dt = time.time() - t0
        # this fit has no astigmatism step: it is 2D and its width is fitted
        self.finished.emit(locs, dt, False, False)


#: Workers that fit every channel at once. They share the window's fit slots,
#: which branch on this rather than on either worker individually.
_MULTICHANNEL_FIT_WORKERS = (
    MultichannelSplineFitWorker,
    MultichannelGaussianFitWorker,
)


class FitZWorker(QtCore.QThread):
    """Fit the z coordinates to fitted localizations based on the
    calibration file using multiprocessing."""

    progressMade = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal(pd.DataFrame, float)
    aborted = QtCore.pyqtSignal()

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        calibration: dict,
        magnification_factor: float,
        pixelsize: float,
        fitting_method: Literal["gausslq", "gaussmle"],
        gpu: bool = False,
        lateral_transforms: list | None = None,
    ) -> None:
        super().__init__()
        self.locs = locs
        self.info = info
        self.calibration = calibration
        self.magnification_factor = magnification_factor
        self.pixelsize = pixelsize
        self.fitting_method = fitting_method
        self.gpu = gpu
        self.lateral_transforms = lateral_transforms or []
        # What the fit actually did with the lateral corrections, for the
        # metadata and the status bar (see Window._record_applied_lateral).
        self.applied_lateral_transforms = []
        self.skipped_lateral_transforms = []

    def on_progress(self, n_done: int) -> None:
        self.progressMade.emit(n_done, len(self.locs))

    def run(self) -> None:
        t0 = time.time()
        # we ignore info since we will merge the metadata from 2D
        # localization as well when saving localizations - except for what
        # zfit reports about the lateral corrections, which is kept below
        # because nothing else records it.
        # The separately loaded corrections go through zfit as well, so that
        # they are applied after the ones the 3D calibration carries and a
        # correction present in both is applied once, exactly as when it is
        # appended to the calibration. zfit warns about that skip; the
        # window says it in the status bar instead.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", lib.DuplicateLateralTransformWarning
            )
            locs, info = zfit.zfit(
                locs=self.locs,
                info=self.info,
                calibration=self.calibration,
                magnification_factor=self.magnification_factor,
                pixelsize=self.pixelsize,
                fitting_method=self.fitting_method,
                lateral_transforms=self.lateral_transforms,
                multiprocess=not self.gpu,
                gpu=self.gpu,
                progress_callback=self.on_progress,
                abort_callback=self.isInterruptionRequested,
            )
        if info is not None:
            self.applied_lateral_transforms = (
                info[-1].get("Lateral corrections applied") or []
            )
            self.skipped_lateral_transforms = (
                info[-1].get("Lateral corrections skipped") or []
            )
        dt = time.time() - t0
        self.finished.emit(locs, dt)


class ChannelRegistrationWorker(QtCore.QThread):
    """Measure where each channel sits relative to the reference, in a
    background thread. ``source`` selects the builder: fiducial beads, or the
    loaded movies' own blinking signal."""

    finished = QtCore.pyqtSignal(str)  # path
    failed = QtCore.pyqtSignal(str)
    statusChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        source: str,
        movies: list,
        box: int,
        minimum_ng: float,
        path: str,
        model: str = "affine",
        channel_paths: list | None = None,
        frame_bounds: list | None = None,
        max_frames: int = 50,
        seed_transforms: list | None = None,
        regions: list | None = None,
        multi_fov: bool = False,
    ) -> None:
        """Set up a channel-registration measurement.

        Parameters
        ----------
        source : {"beads", "signal"}
            Which builder to run: fiducial bead images, or the loaded movies'
            own blinking signal.
        movies : list
            The loaded channel movies, reference first. Both builders read
            them: the bead builder takes them as the bead images.
        box : int
            Box side length (camera pixels) used to detect and fit.
        minimum_ng : float
            Minimum net gradient for a detection.
        path : str
            Where to save the resulting registration (YAML).
        model : str, optional
            Transform model, as in :mod:`picasso.transforms`. Default
            ``"affine"``.
        channel_paths : list, optional
            The loaded channels' file paths, recorded for traceability.
        frame_bounds : list, optional
            Frames the signal builder samples from. None uses all of them.
        max_frames : int, optional
            How many frames are evenly sampled from that range. Default 50.
        seed_transforms : list, optional
            One transform per channel to start the pairing from, when
            re-aligning a registration that has drifted. None builds one from
            scratch.
        regions : list, optional
            Split field of view: one ROI per channel, reference first. The
            single movie is then split by region instead of one movie per
            channel.
        multi_fov : bool, optional
            Beads only: each frame of the bead movie is a different field of
            view, so beads are paired within a frame rather than the frames
            being averaged. Default False.
        """
        super().__init__()
        self.source = source
        self.movies = movies
        self.box = box
        self.minimum_ng = minimum_ng
        self.path = path
        self.model = model
        self.channel_paths = channel_paths
        self.frame_bounds = frame_bounds
        self.max_frames = max_frames
        self.seed_transforms = seed_transforms
        self.regions = regions
        self.multi_fov = multi_fov

    def run(self) -> None:
        """Measure and save the registration.

        The ``QThread`` entry point; emits ``finished`` with the saved path, or
        ``failed`` with the message when a channel could not be registered.
        """
        try:
            if self.source == "beads":
                self.statusChanged.emit("Registering channels on beads...")
                registration.calibrate_channel_registration_from_beads(
                    self.movies,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    model=self.model,
                    regions=self.regions,
                    multi_fov=self.multi_fov,
                    channel_paths=self.channel_paths,
                    path=self.path,
                )
            else:
                self.statusChanged.emit(
                    "Detecting spots for channel registration..."
                )
                registration.calibrate_channel_registration_from_signal(
                    self.movies,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    model=self.model,
                    frame_bounds=self.frame_bounds,
                    max_frames=self.max_frames,
                    seed_transforms=self.seed_transforms,
                    regions=self.regions,
                    channel_paths=self.channel_paths,
                    path=self.path,
                    progress_callback=lambda n: self.statusChanged.emit(
                        f"Registered {n} channel(s)..."
                    ),
                )
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished.emit(self.path)


class SplineCalibrationWorker(QtCore.QThread):
    """Build a cubic-spline PSF calibration from a bead z-stack movie in a
    background thread (bead detection, averaging and spline coefficients, all
    on the CPU)."""

    finished = QtCore.pyqtSignal(str, int, int)  # (path, n_beads, n_used)
    failed = QtCore.pyqtSignal(str)
    statusChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        movie,
        info: list[dict],
        camera_info: dict,
        box: int,
        minimum_ng: float,
        step: float,
        frames_per_step: int,
        frame_order: str,
        model: str,
        path: str,
        frame_bounds=None,
        magnification_factor: float = 0.79,
        correct_z_bias: bool = False,
        link_photons: bool = True,
        registration_model: str = "affine",
        roi=None,
        regions=None,
        movies=None,
        infos=None,
        camera_infos=None,
    ) -> None:
        super().__init__()
        self.bead_diagnostics: list[dict] = []
        self.movie = movie
        self.info = info
        self.camera_info = camera_info
        self.box = box
        self.minimum_ng = minimum_ng
        self.step = step
        self.frames_per_step = frames_per_step
        self.frame_order = frame_order
        self.frame_bounds = frame_bounds
        self.model = model
        self.magnification_factor = magnification_factor
        self.correct_z_bias = correct_z_bias
        self.link_photons = link_photons
        self.registration_model = registration_model
        self.roi = roi
        # When set (>= 2 rectangles), the ROIs are treated as channels of one
        # movie (split-FOV) and a multichannel calibration is built instead.
        self.regions = regions
        # When set (>= 2 movies), the channels are separate movies (separate
        # files or a multichannel file) registered from bead correspondences;
        # movies[0]/infos[0] is the reference channel.
        self.movies = movies
        self.infos = infos
        self.camera_infos = camera_infos
        self.path = path

    def run(self) -> None:
        try:
            if self.movies:
                calibration, diagnostics = (
                    spline.calibrate_spline_multichannel(
                        self.movies,
                        infos=self.infos,
                        camera_infos=self.camera_infos,
                        box=self.box,
                        minimum_ng=self.minimum_ng,
                        d=self.step,
                        frames_per_step=self.frames_per_step,
                        frame_bounds=self.frame_bounds,
                        frame_order=self.frame_order,
                        magnification_factor=self.magnification_factor,
                        correct_z_bias=self.correct_z_bias,
                        link_photons=self.link_photons,
                        model=self.registration_model,
                        reference=0,
                        path=self.path,
                        return_diagnostics=True,
                    )
                )
            elif self.regions:
                calibration, diagnostics = spline.calibrate_spline_split_fov(
                    self.movie,
                    info=self.info,
                    camera_info=self.camera_info,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    d=self.step,
                    regions=self.regions,
                    frames_per_step=self.frames_per_step,
                    frame_bounds=self.frame_bounds,
                    frame_order=self.frame_order,
                    magnification_factor=self.magnification_factor,
                    correct_z_bias=self.correct_z_bias,
                    link_photons=self.link_photons,
                    model=self.registration_model,
                    path=self.path,
                    return_diagnostics=True,
                )
            else:
                calibration, diagnostics = spline.calibrate_spline(
                    self.movie,
                    info=self.info,
                    camera_info=self.camera_info,
                    box=self.box,
                    minimum_ng=self.minimum_ng,
                    d=self.step,
                    frames_per_step=self.frames_per_step,
                    frame_bounds=self.frame_bounds,
                    frame_order=self.frame_order,
                    model=self.model,
                    magnification_factor=self.magnification_factor,
                    correct_z_bias=self.correct_z_bias,
                    roi=self.roi,
                    path=self.path,
                    return_diagnostics=True,
                )
        except Exception as e:  # surface any failure to the GUI
            self.failed.emit(str(e))
            return
        # the per-bead inspection records (which beads were averaged into the
        # PSF, which were rejected) are read off the worker by the window when
        # it opens the bead inspector; they are too large for a signal payload
        self.bead_diagnostics = [d for d in diagnostics if d]
        self.finished.emit(
            self.path,
            int(calibration.get("n_beads", 0)),
            spline.n_beads_used(calibration),
        )


class AffineCalibrationWorker(QtCore.QThread):
    """Fit a target -> reference affine transform (a cylindrical-lens or a
    chromatic-aberration correction) in a background thread and append it to
    a calibration file.

    Loading the two bead movies and fitting the transform both block for
    seconds to minutes, so they run here instead of on the GUI thread.
    Two things cannot: the metadata / pixel-size prompts, which are modal
    dialogs, and the diagnostic figure. The prompts are proxied to the GUI
    thread via ``promptRequested`` (the worker blocks on a
    ``threading.Event`` until the main thread fills in ``holder``, exactly
    as ``MovieLoadWorker`` does), and the figure is not drawn here at all -
    ``localize.fit_lateral_transform`` hands back a ``qc`` dict that the
    window plots once ``finished`` arrives.
    """

    # (calibration path, number of matched bead pairs, qc dict for the plot)
    finished = QtCore.pyqtSignal(str, int, object)
    failed = QtCore.pyqtSignal(str)
    canceled = QtCore.pyqtSignal()
    statusChanged = QtCore.pyqtSignal(str)
    # callback, (args, kwargs), holder dict for the return value
    promptRequested = QtCore.pyqtSignal(object, object, object)

    def __init__(
        self,
        ref_path: str,
        target_path: str,
        calibration_path: str,
        box: int,
        minimum_ng: float,
        prompt_for_path,
        pixelsize_prompt,
        transform_type: str = "astigmatism",
        model: str = "affine",
    ) -> None:
        super().__init__()
        self.ref_path = ref_path
        self.target_path = target_path
        self.calibration_path = calibration_path
        self.transform_type = transform_type
        self.model = model
        self.box = box
        self.minimum_ng = minimum_ng
        # Window._prompt_for_path: path -> metadata prompt callback
        self._prompt_for_path = prompt_for_path
        self._pixelsize_prompt = pixelsize_prompt
        self._prompt_event = threading.Event()

    def _proxy(self, callback):
        """Wrap a GUI prompt callback so the dialog runs on the main thread
        while this thread blocks for the result."""

        def wrapper(*args, **kwargs):
            holder = {}
            self._prompt_event.clear()
            self.promptRequested.emit(callback, (args, kwargs), holder)
            self._prompt_event.wait()
            return holder.get("result")

        return wrapper

    def _load(self, path: str, label: str):
        """Load one bead movie, prompting for metadata on the GUI thread.
        Returns ``None`` if the user canceled the prompt."""
        self.statusChanged.emit(f"Loading {label} image ...")
        prompt = self._proxy(self._prompt_for_path(path))
        # io.load_movie returns None when the user cancels the info prompt
        return io.load_movie(path, prompt_info=prompt)

    def run(self) -> None:
        try:
            loaded_ref = self._load(self.ref_path, "reference")
            if loaded_ref is None:
                self.canceled.emit()
                return
            movie_ref, info_ref = loaded_ref
            label = (
                "cylindrical lens"
                if self.transform_type == "astigmatism"
                else "target channel"
            )
            loaded_target = self._load(self.target_path, label)
            if loaded_target is None:
                self.canceled.emit()
                return
            movie_target, _ = loaded_target
            # An existing calibration of any kind is appended to; a path
            # that does not exist yet starts a standalone affine
            # calibration (the 2D case, where there is nothing to append to).
            if os.path.exists(self.calibration_path):
                calibration = io.load_any_calibration(self.calibration_path)
            else:
                calibration = {}
        except Exception as e:  # noqa: BLE001 - reported to the GUI
            self.failed.emit(f"Could not load inputs:\n{e}")
            return

        pixelsize = lib.get_from_metadata(info_ref, "Pixelsize", default=None)
        if pixelsize is None:
            pixelsize = self._proxy(self._pixelsize_prompt)()
            if pixelsize is None:  # prompt canceled
                self.canceled.emit()
                return

        self.statusChanged.emit("Fitting affine transform ...")
        try:
            calibration, qc = localize.fit_lateral_transform(
                movie_ref,
                movie_target,
                calibration,
                box=self.box,
                minimum_ng=self.minimum_ng,
                pixelsize=pixelsize,
                transform_type=self.transform_type,
                ref_path=self.ref_path,
                target_path=self.target_path,
                model=self.model,
            )
            io.save_any_calibration(self.calibration_path, calibration)
        except Exception as e:  # noqa: BLE001 - reported to the GUI
            self.failed.emit(str(e))
            return
        self.finished.emit(self.calibration_path, qc["n_pairs"], qc)


class QualityWorker(QtCore.QThread):
    """Run quality checks on the localized data, i.e., calculate the
    number of localizations, experimental localization precision (NeNA),
    drift and mean bright time."""

    progressMade = QtCore.pyqtSignal(str, int, str)
    finished = QtCore.pyqtSignal(str)

    def __init__(
        self,
        locs: pd.DataFrame,
        info: list[dict],
        path: str,
        pixelsize: QtWidgets.QDoubleSpinBox,
    ) -> None:
        super().__init__()
        self.locs = locs
        self.info = info
        self.path = path
        self.pixelsize = pixelsize

    def run(self) -> None:
        # Sanity of locs
        sane_locs = lib.ensure_sanity(self.locs, self.info)

        # Locs
        self.progressMade.emit("Checking Quality (1/4) Locs ..", 0, "")
        locs_per_frame = len(sane_locs) / self.info[0]["Frames"]
        self.progressMade.emit("", 0, f"{locs_per_frame:.1f}")

        # NeNA
        self.progressMade.emit("Checking Quality (2/4) NeNA ..", 0, "")

        def nena_callback(x):
            self.progressMade.emit(
                f"Checking Quality (2/4) NeNA: {x} %",
                0,
                "",
            )

        nena_px = localize.check_nena(sane_locs, self.info, nena_callback)
        nena_nm = float(self.pixelsize.value() * nena_px)
        self.progressMade.emit("", 1, f"{nena_px:.2f} px / {nena_nm:.2f} nm")

        # Drift
        self.progressMade.emit("Checking Quality (3/4) Drift ..", 0, "")

        def drift_callback(x):
            self.progressMade.emit(
                f"Checking Quality (3/4) Drift {x} %",
                0,
                "",
            )

        drift_x, drift_y = localize.check_drift(
            sane_locs, self.info, callback=drift_callback
        )
        self.progressMade.emit(
            "",
            2,
            f"X: {drift_x:.3f} px / Y: {drift_y:.3f} px",
        )

        # Kinetics
        self.progressMade.emit("Checking Quality (4/4) Kinetics ..", 0, "")
        len_mean = localize.check_kinetics(sane_locs, self.info)
        self.progressMade.emit("", 3, f"{len_mean:.3f}")

        localize.add_file_to_db(
            self.path,
            None,
            drift=(drift_x, drift_y),
            len_mean=len_mean,
            nena=nena_px,
        )
        self.finished.emit("Quality parameters complete.")


def main() -> None:
    """Start Picasso: Localize - see ``picasso.gui.app.run_gui``."""
    sys.exit(run_gui(Window, "localize"))


if __name__ == "__main__":
    main()
