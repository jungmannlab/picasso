"""
picasso.localize
~~~~~~~~~~~~~~~~

Identify and localize fluorescent single molecules in a frame
sequence.

Spot detection and the localization table live here; the fits themselves are
run by :mod:`picasso.fitting` (Gaussian and cubic-spline PSF models on the CPU
and on CUDA GPUs). This module owns the translation between them:
calibration dicts, initial parameters, the device choice, and the
Cramer-Rao lower bounds that become the reported localization precisions.

References
----------
Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
License (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import multiprocessing
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from functools import partial
from itertools import chain
from typing import Literal, Callable, TypeAlias, Union
from datetime import datetime

import numba
import numpy as np
from numba import cuda
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sqlalchemy import create_engine
import matplotlib.gridspec as gridspec
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist

from .ext import bitplane

from . import (
    io,
    lib,
    avgroi,
    postprocess,
    zfit,
    __version__,
)

# aliased: `transforms` is used as a local name for lists of channel
# transforms all over this module
from . import transforms as tform
from .fitting import (
    gaussfit,
    gaussfit_cuda,
    precision,
    seeds,
    splinefit,
    splinefit_cuda,
)

# Check for CUDA availability of the *fitting* backends. Otherwise, CPU is
# used. The CRLB kernels have their own probe in picasso.fitting.precision.
try:
    CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:
    CUDA_AVAILABLE = False


plt.style.use("ggplot")


#: A movie Picasso can read frames from: one loaded by
#: ``picasso.io.load_movie`` (an ``io.AbstractPicassoMovie``, or the
#: ``np.memmap`` of a ``.raw`` file). Not a plain 3D array: the readers are
#: lazy, so a movie is only indexed frame by frame.
LoadedMovie: TypeAlias = Union["io.AbstractPicassoMovie", np.memmap]

#: Anything the identification reads frames from, i.e. a loaded movie or one
#: of the filter wrappers below, which deliberately do not implement
#: ``io.AbstractPicassoMovie`` so that they cannot reach the fit (see
#: ``TemporalMedianMovie``).
MovieLike: TypeAlias = Union[
    LoadedMovie, "TemporalMedianMovie", "GaussianFilteredMovie"
]


MAX_LOCS = int(1e6)

#: How multichannel (or split-FOV) data is fitted: all channels tied into one
#: global fit, or each channel on its own. Recorded in the metadata of every
#: independent fit and shown as the ``Fit`` setting in Picasso: Localize.
FIT_MODE_JOINT = "Jointly (registered channels)"
FIT_MODE_INDEPENDENT = "Each channel separately"

# Axial multi-start. A single in-focus seed leaves a spline fit z-degenerate at
# large |z|. Several seeds spanning the calibration stack are run per spot and the
# one that best explains the data is kept. One seed per ~20 calibration planes,
# bounded; the rule the calibration diagnostic has always used.
_Z_STARTS_PER_PLANES = 20
_Z_STARTS_MIN = 5
_Z_STARTS_MAX = 15


def _default_n_z_starts(calibration: dict) -> int:
    """Number of axial seeds for a spline fit, from the calibration's z depth.

    1 (i.e. no multi-start) for a 2D model, which has no z to be degenerate in,
    and for a calibration whose ``n_data`` does not describe a z axis."""
    if calibration.get("model") == "spline-2d":
        return 1
    n_data = calibration.get("n_data")
    if n_data is None or len(n_data) < 3:
        return 1
    n_z = int(n_data[2])
    return int(
        np.clip(n_z // _Z_STARTS_PER_PLANES, _Z_STARTS_MIN, _Z_STARTS_MAX)
    )


# The columns under base are always available and the keys such as "3D
# only" will be displayed in the save columns dialog in the GUI for
# clarity
LOCALIZATION_COLUMNS = {
    "Base": [
        "frame",
        "x",
        "y",
        "photons",
        "sx",
        "sy",
        "bg",
        "lpx",
        "lpy",
        "ellipticity",
        "net_gradient",
    ],
    "3D only": ["z", "d_zcalib", "lpz"],
    "Rotation only": ["angle", "angle_unc"],
    "Picked spots only": ["n_id"],
    "MLE only": ["log_likelihood", "iterations"],
    "Least squares only": ["chi_square"],
    "Uncertainty": ["photons_unc", "bg_unc", "sx_unc", "sy_unc"],
    "Multichannel only": (
        [f"photons_ch{c}" for c in range(precision._LINK_XYZ_MAX_CHANNELS)]
        + [f"bg_ch{c}" for c in range(precision._LINK_XYZ_MAX_CHANNELS)]
        + [
            f"rel_photons_ch{c}"
            for c in range(precision._LINK_XYZ_MAX_CHANNELS)
        ]
        + ["color"]
    ),
}
# Column naming the ROI a localization was found in, see ``add_roi_id``.
# Only a fit restricted to ROIs has it, so it is not in
# LOCALIZATION_COLUMNS.
ROI_ID_COLUMN = "roi_id"
# ``ROI_ID_COLUMN`` for a localization inside none of the ROIs.
NO_ROI_ID = -1
# For database:
MEAN_COLS = LOCALIZATION_COLUMNS["Base"] + LOCALIZATION_COLUMNS["3D only"]
SET_COLS = [
    "Frames",
    "Height",
    "Width",
    "Box Size",
    "Min. Net Gradient",
    "Pixelsize",
]
# Memory budget for the cached temporal windows of TemporalMedianMovie.
TEMPORAL_MEDIAN_CACHE_BYTES = 512 * 1024**2
# Gaussian filter for spot identification
GAUSSIAN_FILTER_TRUNCATE = 4.0
GAUSSIAN_FILTER_MODE = "nearest"
# Default bead-detection / matching parameters for `calibrate_lateral_transform`.
_LATERAL_MATCH_MAX_DIST_PX = 40.0  # max distance between matched pair
_AFFINE_XCORR_HALF_WIDTH = 18  # half-width of bead crop for xcorr


@numba.jit(nopython=True, nogil=True, cache=False)
def _local_maxima(
    frame: lib.IntArray2D, box: int
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Find pixels with maximum value within a region of interest.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    box : int
        Size of the box to search for local maxima. Should be an odd
        integer.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the local maxima.
    x : lib.IntArray1D
        x-coordinates of the local maxima.
    """
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def _gradient_at(
    frame: lib.IntArray2D,
    y: int,
    x: int,
    i: int,
) -> tuple[float, float]:
    """Calculate the gradient at a specific pixel in the frame.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    y, x : int
        Coordinates of the pixel where the gradient is calculated.
    i : int
        Index of the pixel in the list of maxima. Not used in this
        function.

    Returns
    -------
    gy : float
        Gradient in the y-direction at the pixel (y, x).
    gx : float
        Gradient in the x-direction at the pixel (y, x).
    """
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@numba.jit(nopython=True, nogil=True, cache=False)
def _net_gradient(
    frame: lib.IntArray2D,
    y: lib.IntArray1D,
    x: lib.IntArray1D,
    box: int,
    uy: lib.FloatArray2D,
    ux: lib.FloatArray2D,
) -> lib.FloatArray1D:
    """Calculate the net gradient at the identified maxima in the
    frame.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    y, x : lib.IntArray1D
        Coordinates of the identified maxima in the frame.
    box : int
        Size of the box used for calculating the gradient.
    uy, ux : lib.FloatArray2D
        Arrays of shape (box, box) containing the y and x components
        of the gradient, respectively.

    Returns
    -------
    ng : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    box_half = int(box / 2)
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(
                range(xi - box_half, xi + box_half + 1)
            ):
                if not (k == yi and m == xi):
                    gy, gx = _gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(
    image: lib.IntArray2D,
    minimum_ng: float,
    box: int,
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Identify local maxima in the image and calculate the net gradient
    at those maxima.

    Parameters
    ----------
    image : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float
        Minimum net gradient value to consider a maximum as valid.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the identified maxima.
    x : lib.IntArray1D
        x-coordinates of the identified maxima.
    ng : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    y, x = _local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = _net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def _normalize_rect(
    rect: tuple[tuple[int, int], tuple[int, int]],
) -> list[list[int]]:
    """Return a rectangle as ``[[y_min, x_min], [y_max, x_max]]`` with
    integer, correctly ordered corners (the input corners may be given
    in any order)."""
    (y0, x0), (y1, x1) = rect
    return [
        [int(min(y0, y1)), int(min(x0, x1))],
        [int(max(y0, y1)), int(max(x0, x1))],
    ]


def _subtract_rect(
    a: list[list[int]], b: list[list[int]]
) -> list[list[list[int]]]:
    """Subtract rectangle ``b`` from rectangle ``a``.

    Returns the parts of ``a`` not covered by ``b`` as a list of up to
    four disjoint axis-aligned rectangles (a guillotine split into top,
    bottom, left and right bands around the intersection). If the
    rectangles do not overlap, ``[a]`` is returned unchanged. Both
    rectangles must use the ``[[y_min, x_min], [y_max, x_max]]`` format.
    """
    (ay0, ax0), (ay1, ax1) = a
    (by0, bx0), (by1, bx1) = b
    # intersection
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return [a]  # no overlap
    pieces = []
    if ay0 < iy0:  # top band, full width of a
        pieces.append([[ay0, ax0], [iy0, ax1]])
    if iy1 < ay1:  # bottom band, full width of a
        pieces.append([[iy1, ax0], [ay1, ax1]])
    if ax0 < ix0:  # left band, between the horizontal cuts
        pieces.append([[iy0, ax0], [iy1, ix0]])
    if ix1 < ax1:  # right band, between the horizontal cuts
        pieces.append([[iy0, ix1], [iy1, ax1]])
    return pieces


def clip_rois(
    rois: list[tuple[tuple[int, int], tuple[int, int]]],
    min_size: int = 0,
) -> list[list[list[int]]]:
    """Clip a list of (possibly overlapping) ROIs into a list of
    disjoint rectangles.

    The ROIs are processed in order; each rectangle is trimmed against
    the union of the already-accepted rectangles (via ``_subtract_rect``)
    so that earlier ROIs take precedence and no pixel is covered twice.
    A single ROI may therefore split into several rectangles. Corners
    are normalized and pieces smaller than ``min_size`` in either
    dimension are dropped (pass ``box`` so slivers that cannot hold a
    spot are discarded).

    Parameters
    ----------
    rois : list of rectangles
        Each rectangle is ``((y_min, x_min), (y_max, x_max))`` (corners
        may be in any order).
    min_size : int, optional
        Minimum side length (in pixels) for a clipped piece to be kept.
        Default is 0 (keep any piece with positive area).

    Returns
    -------
    list of rectangles
        Disjoint rectangles in ``[[y_min, x_min], [y_max, x_max]]``
        format whose union equals the union of the inputs (minus dropped
        slivers).
    """
    accepted: list[list[list[int]]] = []
    for rect in rois:
        pieces = [_normalize_rect(rect)]
        for acc in accepted:
            new_pieces: list[list[list[int]]] = []
            for piece in pieces:
                new_pieces.extend(_subtract_rect(piece, acc))
            pieces = new_pieces
        for piece in pieces:
            height = piece[1][0] - piece[0][0]
            width = piece[1][1] - piece[0][1]
            if height > 0 and width > 0:
                if height >= min_size and width >= min_size:
                    accepted.append(piece)
    return accepted


def _as_roi_list(
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None,
) -> list[list[list[int]]] | None:
    """Normalize the ``roi`` argument into a list of rectangles or None.

    Accepts a single rectangle ``((y0, x0), (y1, x1))`` (for backward
    compatibility) or a list of such rectangles. An empty list and
    ``None`` both map to ``None`` (whole frame).
    """
    if roi is None or len(roi) == 0:
        return None
    first = roi[0][0]
    if isinstance(first, (list, tuple, np.ndarray)):
        rois = [_normalize_rect(r) for r in roi]  # list of rectangles
    else:
        rois = [_normalize_rect(roi)]  # single rectangle
    return rois if len(rois) else None


def _as_ng_list(
    minimum_ng: float | list | np.ndarray,
    n_rois: int,
) -> list[float]:
    """Normalize ``minimum_ng`` into one threshold per ROI.

    A scalar (the usual case) applies to every ROI. A sequence gives each
    ROI its own threshold, which is what split-FOV data needs: the regions
    are separate channels imaged through different optics, so their spots
    do not share a brightness scale. A one-element sequence is treated as
    a scalar.

    Parameters
    ----------
    minimum_ng : float or sequence of float
        Minimum net gradient, shared or one per ROI.
    n_rois : int
        Number of ROIs the thresholds have to cover.

    Returns
    -------
    list of float
        ``n_rois`` thresholds.

    Raises
    ------
    ValueError
        If a sequence is given whose length is neither 1 nor ``n_rois``.
    """
    if isinstance(minimum_ng, (list, tuple, np.ndarray, pd.Series)):
        ngs = [float(_) for _ in minimum_ng]
    else:
        ngs = [float(minimum_ng)]
    if len(ngs) == 1:
        return ngs * n_rois
    if len(ngs) != n_rois:
        raise ValueError(
            f"minimum_ng has {len(ngs)} values but there are {n_rois} "
            "ROI(s); give one threshold per ROI or a single shared one."
        )
    return ngs


def add_roi_id(
    locs: pd.DataFrame,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
) -> pd.DataFrame:
    """Name the ROI each localization was found in.

    A fit restricted to several ROIs produces one table, so the ROI a
    localization came from is only recoverable from its coordinates. A
    ``roi_id`` column is appended holding the index of the rectangle the
    localization falls in - indexing ``roi`` as given, which is the order
    ``clip_rois`` returned it in and the order the ``ROI`` entry of a
    saved file's metadata lists. The rectangles being disjoint (see
    :func:`clip_rois`), the index is unambiguous; a localization inside
    none of them gets ``NO_ROI_ID`` (-1), which the sub-pixel result of a
    fit can be when the spot was detected right at an ROI's edge.

    Call this on the localizations as they come out of the fit: drift
    correction, lateral corrections and z fitting all move ``x`` and
    ``y``, which can push a localization near the seam between two ROIs
    into the neighbouring rectangle.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x`` and ``y`` columns, in the coordinate
        frame the ROIs are given in.
    roi : tuple or list of tuples, optional
        Region(s) of interest the localizations were identified in, see
        :func:`identify_in_frame`. None or an empty list means the whole
        frame was used and no column is added. Default is None.

    Returns
    -------
    locs : pd.DataFrame
        A copy with an ``int32`` ``roi_id`` column appended, or ``locs``
        itself when there is no ROI to name.
    """
    rois = _as_roi_list(roi)
    if rois is None:
        return locs
    x = np.asarray(locs["x"], dtype=np.float64)
    y = np.asarray(locs["y"], dtype=np.float64)
    roi_id = np.full(len(locs), NO_ROI_ID, dtype=np.int32)
    for c, ((y0, x0), (y1, x1)) in enumerate(rois):
        inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        # first match wins, so overlapping rectangles (which clip_rois
        # does not produce) still give one index per localization
        roi_id[inside & (roi_id == NO_ROI_ID)] = c
    return locs.assign(**{ROI_ID_COLUMN: roi_id})


def _temporal_median(
    frames: np.ndarray, max_stripe_bytes: int = 64 * 1024**2
) -> lib.FloatArray2D:
    """Per-pixel median of a stack of frames, i.e. the median along
    axis 0.

    ``np.partition`` is used instead of ``np.median`` because only the
    middle order statistic is needed and because it keeps the data in its
    native (usually integer) dtype - ``np.median`` promotes to float64,
    which doubles the memory traffic. The stack is processed in stripes of
    rows so that the copy ``np.partition`` makes internally stays bounded
    regardless of the window length and the frame size.

    Parameters
    ----------
    frames : np.ndarray
        Stack of frames of shape (N, Y, X).
    max_stripe_bytes : int, optional
        Approximate memory budget for one stripe. Default is 64 MB.

    Returns
    -------
    lib.FloatArray2D
        Per-pixel median, 2D array of shape (Y, X) and dtype float32.
    """
    n_frames, height, width = frames.shape
    median = np.empty((height, width), dtype=np.float32)
    lower = (n_frames - 1) // 2
    upper = n_frames // 2  # == lower for an odd number of frames
    kth = lower if lower == upper else (lower, upper)
    stripe_rows = max(
        1, int(max_stripe_bytes // max(n_frames * width * frames.itemsize, 1))
    )
    for y0 in range(0, height, stripe_rows):
        y1 = min(y0 + stripe_rows, height)
        stripe = np.partition(frames[:, y0:y1], kth, axis=0)
        if lower == upper:
            median[y0:y1] = stripe[lower]
        else:  # cast before adding, the sum can overflow the input dtype
            median[y0:y1] = 0.5 * (
                stripe[lower].astype(np.float32)
                + stripe[upper].astype(np.float32)
            )
    return median


class _TemporalMedianBlock:
    """One cached temporal window of a ``TemporalMedianMovie``.

    ``ready`` is set once ``median`` (and possibly ``frames``) has been
    filled in, or once the attempt has failed and ``error`` holds the
    exception. Threads that did not win the race to compute the block wait
    on it.
    """

    __slots__ = ("start", "stop", "median", "frames", "ready", "error")

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop
        self.median: lib.FloatArray2D | None = None
        self.frames: np.ndarray | None = None
        self.ready = threading.Event()
        self.error: BaseException | None = None

    @property
    def nbytes(self) -> int:
        """Memory held by this block."""
        total = 0 if self.median is None else self.median.nbytes
        if self.frames is not None:
            total += self.frames.nbytes
        return total


class TemporalMedianMovie:
    """Lazily evaluated, read-only temporal median filtered view of a
    movie.

    Frame ``t`` is ``max(movie[t] - median(movie[window(t)]), 0)`` as
    float32, where ``window(t)`` is a window of ``window`` frames centered
    on ``t``. At the edges of the movie the window is shifted inwards
    rather than truncated, so it always covers ``window`` frames.

    Computing that median for every single frame is far too slow for real
    movies, so the median is only evaluated at *anchor* frames spaced
    ``stride`` apart and shared by the frames in between. The default
    ``stride=window`` gives ``n_frames / window`` medians; ``stride=1``
    reproduces the exact per-frame filter.

    This class is meant for spot identification only - fitting, spot
    cutting and photon conversion must always use the raw movie, since
    the subtracted background would otherwise corrupt the photon counts.
    It deliberately does not implement the ``io.AbstractPicassoMovie``
    interface so that accidentally handing it to ``fit`` trips that
    function's input assertion instead of silently returning wrong
    photon numbers.

    Note that subtracting a per-pixel background changes the scale of the
    net gradient, so the minimum net gradient has to be re-tuned when the
    filter is switched on or off.

    References
    ----------
    Martens, K. J. A., Turkowyd, B. & Endesfelder, U.
    "Raw Data to Results: A Hands-On Introduction and Overview of
    Computational Analysis for Single-Molecule Localization Microscopy."
    Frontiers in Bioinformatics 1, 817254 (2022).
    https://doi.org/10.3389/fbinf.2021.817254

    Parameters
    ----------
    movie : MovieLike
        The raw movie, i.e. any object supporting ``len()`` and integer
        indexing that returns 2D frames.
    window : int, optional
        Number of frames in the temporal window used for the median.
        Clipped to the length of the movie. Default is 51.
    stride : int or None, optional
        Spacing (in frames) between the anchors at which the median is
        evaluated. Clipped to ``[1, window]``. None (the default) uses
        ``window``, which is the fastest setting; 1 evaluates the median
        for every frame.
    roi : tuple, list of tuples or None, optional
        Region(s) of interest that will be identified in, in the same
        format as ``identify``. When given, the median is only computed
        inside their (padded) bounding box and the returned frames are
        zero outside it. Default is None (whole frame).
    roi_pad : int, optional
        Number of pixels the ROI bounding box is grown by, so that the
        gradients of maxima sitting on the ROI border are still computed
        from filtered pixels. Pass ``int(box / 2) + 1``. Default is 0.
    cache_bytes : int, optional
        Memory budget for the cached temporal windows. Default is
        ``TEMPORAL_MEDIAN_CACHE_BYTES``.

    Attributes
    ----------
    raw : lib.IntArray3D
        The underlying unfiltered movie.
    window, stride, roi : as above
    """

    # The wrapper is internally thread safe (see _read_lock), so it tells
    # identify_by_frame_number not to serialize reads on the identification
    # lock - that lock is also what _identify_worker hands out frame
    # numbers with, so holding it across a median would stall every worker.
    supports_concurrent_reads = True

    def __init__(
        self,
        movie: MovieLike,
        window: int = 51,
        *,
        stride: int | None = None,
        roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
        roi_pad: int = 0,
        cache_bytes: int = TEMPORAL_MEDIAN_CACHE_BYTES,
    ) -> None:
        self.raw = movie
        self.n_frames = len(movie)
        if self.n_frames == 0:
            raise ValueError("Cannot temporally filter an empty movie.")
        self.window = max(1, min(int(window), self.n_frames))
        stride = self.window if stride is None else int(stride)
        self.stride = max(1, min(stride, self.window))
        self.roi = roi
        self.roi_pad = int(roi_pad)
        self.cache_bytes = int(cache_bytes)
        # one read to learn the frame geometry; movies do not agree on
        # whether they expose .shape (TiffMap does not) or a usable .dtype
        probe = np.asarray(movie[0])
        self.frame_shape = probe.shape
        self._raw_dtype = probe.dtype
        self._bbox = self._roi_bbox(roi, self.roi_pad)
        self._cache: OrderedDict[int, _TemporalMedianBlock] = OrderedDict()
        self._lock = threading.Lock()  # guards _cache only, never held long
        concurrent = getattr(
            movie, "supports_concurrent_reads", False
        ) or isinstance(movie, np.memmap)
        self._read_lock = None if concurrent else threading.Lock()

    def _roi_bbox(
        self,
        roi: tuple[tuple[int, int], tuple[int, int]] | list | None,
        pad: int,
    ) -> tuple[slice, slice] | None:
        """Bounding box of the ROIs, grown by ``pad`` and clipped to the
        frame, or None to use the whole frame."""
        rois = _as_roi_list(roi)
        if rois is None:
            return None
        height, width = self.frame_shape
        y0 = max(0, min(int(r[0][0]) for r in rois) - pad)
        x0 = max(0, min(int(r[0][1]) for r in rois) - pad)
        y1 = min(height, max(int(r[1][0]) for r in rois) + pad)
        x1 = min(width, max(int(r[1][1]) for r in rois) + pad)
        if y1 <= y0 or x1 <= x0:
            return None
        # the median costs O(window * area), so restricting it only pays
        # off when the bounding box is substantially smaller than the frame
        if (y1 - y0) * (x1 - x0) > 0.5 * height * width:
            return None
        return slice(y0, y1), slice(x0, x1)

    def _block_index(self, frame_number: int) -> int:
        """Index of the temporal window used to filter ``frame_number``.
        Frames are tiled into groups of ``stride``, each sharing one
        median."""
        return frame_number // self.stride

    def _bounds(self, block_index: int) -> tuple[int, int]:
        """Start (inclusive) and stop (exclusive) frame of a block's
        temporal window.

        The window covers the block's own ``stride`` frames and is grown
        symmetrically around them up to ``window`` frames; at the edges of
        the movie it is shifted inwards rather than truncated, so it
        always spans ``window`` frames. Every frame of the block therefore
        lies inside its own window, which is what lets the block serve raw
        frames straight out of its cached window.

        The two extremes are exactly the ones we care about:
        ``stride == window`` tiles the movie into non-overlapping blocks,
        and ``stride == 1`` reduces to a window centered on the frame,
        i.e. the exact per-frame filter.
        """
        start = block_index * self.stride - (self.window - self.stride) // 2
        stop = start + self.window
        if start < 0:
            stop -= start
            start = 0
        if stop > self.n_frames:
            start -= stop - self.n_frames
            stop = self.n_frames
        return max(start, 0), min(stop, self.n_frames)

    def _read_frame(self, index: int) -> np.ndarray:
        if self._read_lock is None:
            return np.asarray(self.raw[index])
        with self._read_lock:
            return np.asarray(self.raw[index])

    def _read(self, start: int, stop: int) -> np.ndarray:
        """Read frames ``[start, stop)`` into one array. Frames are read
        one by one because not every movie class supports slice
        indexing."""
        frames = np.empty(
            (stop - start, *self.frame_shape), dtype=self._raw_dtype
        )
        if self._read_lock is None:
            for i in range(start, stop):
                frames[i - start] = self.raw[i]
        else:
            with self._read_lock:
                for i in range(start, stop):
                    frames[i - start] = self.raw[i]
        return frames

    def _fill(self, block: _TemporalMedianBlock) -> None:
        """Compute a block's median (and keep its frames if they fit in
        the cache budget)."""
        frames = self._read(block.start, block.stop)
        if self._bbox is None:
            block.median = _temporal_median(frames)
        else:
            sy, sx = self._bbox
            block.median = _temporal_median(frames[:, sy, sx])
        # every frame this block serves lies inside its own window (since
        # stride <= window), so keeping the frames means each frame of the
        # movie is read exactly once overall. Two blocks stay resident, so
        # only keep the frames if two windows still fit in the budget.
        if 2 * frames.nbytes <= self.cache_bytes:
            block.frames = frames

    def _evict(self) -> None:
        """Keep the cache within its memory budget. Must be called with
        ``_lock`` held.

        Raw frames are dropped before whole blocks: a block's median is
        tiny (one float32 image) and re-reading a frame is far cheaper
        than recomputing a median. The two most recent blocks are always
        kept intact, since the worker pool straddles at most two of them.
        """
        if len(self._cache) <= 2:
            return
        total = sum(block.nbytes for block in self._cache.values())
        if total <= self.cache_bytes:
            return
        for block in list(self._cache.values())[:-2]:  # oldest first
            if block.frames is not None:
                total -= block.frames.nbytes
                block.frames = None
                if total <= self.cache_bytes:
                    return
        while len(self._cache) > 2 and total > self.cache_bytes:
            total -= self._cache.popitem(last=False)[1].nbytes

    def _block(self, frame_number: int) -> _TemporalMedianBlock:
        """The cached block used to filter ``frame_number``, computing it
        if necessary. Exactly one thread computes a given block; the
        others wait for it without holding the cache lock."""
        index = self._block_index(frame_number)
        with self._lock:
            block = self._cache.get(index)
            owner = block is None
            if owner:
                block = _TemporalMedianBlock(*self._bounds(index))
                self._cache[index] = block
            else:
                self._cache.move_to_end(index)
        if not owner:
            block.ready.wait()
            if block.error is not None:
                raise block.error
            return block
        try:
            self._fill(block)
        except BaseException as error:
            block.error = error
            with self._lock:
                self._cache.pop(index, None)
            raise
        finally:
            block.ready.set()
        with self._lock:
            self._evict()
        return block

    def clear_cache(self) -> None:
        """Drop all cached temporal windows."""
        with self._lock:
            self._cache.clear()

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if len(it) == 1:
                return self[it[0]]
            return self[it[0]][tuple(it[1:])]
        if isinstance(it, slice):
            return np.stack(
                [self[i] for i in range(*it.indices(self.n_frames))]
            )
        index = int(it)
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Frame {it} is out of range for a movie with "
                f"{self.n_frames} frames."
            )
        block = self._block(index)
        # bind once: another thread's eviction may clear block.frames
        # between the check and the lookup
        frames = block.frames
        if frames is not None:
            raw = frames[index - block.start]
        else:
            raw = self._read_frame(index)
        if self._bbox is None:
            return np.maximum(raw.astype(np.float32) - block.median, 0)
        sy, sx = self._bbox
        filtered = np.zeros(self.frame_shape, dtype=np.float32)
        filtered[sy, sx] = np.maximum(
            raw[sy, sx].astype(np.float32) - block.median, 0
        )
        return filtered

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height, width)``, as a raw movie's."""
        return (self.n_frames, *self.frame_shape)

    @property
    def dtype(self) -> np.dtype:
        """``float32``: the dtype every frame of this view is returned in."""
        return np.dtype(np.float32)

    def close(self) -> None:
        """Drop the cached medians and close the underlying raw movie."""
        self.clear_cache()
        close = getattr(self.raw, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def gaussian_filter_radius(
    sigma: float | None, truncate: float = GAUSSIAN_FILTER_TRUNCATE
) -> int:
    """Half-width (in pixels) of the kernel that
    ``scipy.ndimage.gaussian_filter`` actually uses for ``sigma``.

    This is the same ``int(truncate * sd + 0.5)`` expression as
    ``scipy.ndimage.gaussian_filter1d``, so a filtered pixel depends on
    exactly the pixels within this distance and no further.

    Parameters
    ----------
    sigma : float or None
        Standard deviation of the Gaussian kernel, in pixels. None or 0
        means no filtering.
    truncate : float, optional
        Kernel cut-off in units of sigma. Default is
        ``GAUSSIAN_FILTER_TRUNCATE``.

    Returns
    -------
    radius : int
        Kernel half-width in pixels, 0 if no filtering takes place.
    """
    if not sigma or sigma <= 0:
        return 0
    return int(float(truncate) * float(sigma) + 0.5)


def identification_roi_pad(
    box: int, gaussian_filter_sigma: float | None = None
) -> int:
    """Number of pixels a ROI has to be grown by so that every pixel the
    identification actually reads is validly filtered.

    ``identify_in_frame`` computes gradients up to ``int(box / 2) + 1``
    pixels outside a ROI. A Gaussian filter mixes in everything within
    its kernel radius, so with the filter on the valid region has to
    extend that much further still - otherwise the zeros that a
    ``TemporalMedianMovie`` leaves outside its bounding box get smeared
    into the very pixels those gradients are computed from.

    Parameters
    ----------
    box : int
        Box side length used for identification.
    gaussian_filter_sigma : float or None, optional
        Sigma of the spatial Gaussian filter, see
        ``GaussianFilteredMovie``. Default is None (no filtering).

    Returns
    -------
    pad : int
        Padding in pixels.
    """
    return int(box / 2) + 1 + gaussian_filter_radius(gaussian_filter_sigma)


class GaussianFilteredMovie:
    """Lazily evaluated, read-only spatially Gaussian smoothed view of a
    movie.

    Frame ``t`` is ``gaussian_filter(movie[t], sigma)`` as float32.

    Spot identification looks for a single local maximum per spot. A PSF
    may break up into several local maxima, so one molecule can be
    difficult to find.

    This class is meant for spot identification only - fitting, spot
    cutting and photon conversion must always use the raw movie, since
    the smoothed intensities would otherwise corrupt the photon counts.
    It deliberately does not implement the ``io.AbstractPicassoMovie``
    interface so that accidentally handing it to ``fit`` trips that
    function's input assertion instead of silently returning wrong
    photon numbers.

    Note that smoothing lowers gradient magnitudes, so the minimum net
    gradient has to be re-tuned whenever sigma changes.

    Parameters
    ----------
    movie : MovieLike
        The raw movie, i.e. any object supporting ``len()`` and integer
        indexing that returns 2D frames. May itself be a
        ``TemporalMedianMovie``, in which case the median is subtracted
        before smoothing.
    sigma : float
        Standard deviation of the Gaussian kernel, in camera pixels.
        Must be positive; use the unwrapped movie to identify without
        smoothing.
    truncate : float, optional
        Kernel cut-off in units of sigma. Default is
        ``GAUSSIAN_FILTER_TRUNCATE``.
    mode : str, optional
        How the frame borders are extended, see
        ``scipy.ndimage.gaussian_filter``. Default is
        ``GAUSSIAN_FILTER_MODE``.

    Attributes
    ----------
    raw : lib.IntArray3D
        The underlying unsmoothed movie.
    radius : int
        Kernel half-width in pixels, see ``gaussian_filter_radius``.
    sigma, truncate, mode : as above
    """

    # Filtering a frame is stateless (there is no cache, and scipy.ndimage
    # is re-entrant), so reads never have to be serialized on the
    # identification lock.
    supports_concurrent_reads = True

    def __init__(
        self,
        movie: MovieLike,
        sigma: float,
        *,
        truncate: float = GAUSSIAN_FILTER_TRUNCATE,
        mode: str = GAUSSIAN_FILTER_MODE,
    ) -> None:
        self.raw = movie
        self.n_frames = len(movie)
        if self.n_frames == 0:
            raise ValueError("Cannot filter an empty movie.")
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError(
                "The Gaussian filter sigma must be positive; identify on "
                "the unwrapped movie to not filter at all."
            )
        self.truncate = float(truncate)
        self.mode = mode
        # one read to learn the frame geometry; movies do not agree on
        # whether they expose .shape (TiffMap does not)
        self.frame_shape = np.asarray(movie[0]).shape
        concurrent = getattr(
            movie, "supports_concurrent_reads", False
        ) or isinstance(movie, np.memmap)
        self._read_lock = None if concurrent else threading.Lock()

    @property
    def radius(self) -> int:
        """Half-width (in pixels) of the kernel used."""
        return gaussian_filter_radius(self.sigma, self.truncate)

    def _read_frame(self, index: int) -> np.ndarray:
        if self._read_lock is None:
            return np.asarray(self.raw[index])
        with self._read_lock:
            return np.asarray(self.raw[index])

    def clear_cache(self) -> None:
        """Do nothing; this view caches nothing. Kept so that callers can
        treat both identification filters alike."""

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if len(it) == 1:
                return self[it[0]]
            return self[it[0]][tuple(it[1:])]
        if isinstance(it, slice):
            return np.stack(
                [self[i] for i in range(*it.indices(self.n_frames))]
            )
        index = int(it)
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Frame {it} is out of range for a movie with "
                f"{self.n_frames} frames."
            )
        raw = self._read_frame(index)
        # gaussian_filter keeps the input dtype unless told otherwise, so
        # a uint16 movie would come back rounded to integers
        return gaussian_filter(
            raw.astype(np.float32, copy=False),
            self.sigma,
            output=np.float32,
            mode=self.mode,
            truncate=self.truncate,
        )

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height, width)``, as a raw movie's."""
        return (self.n_frames, *self.frame_shape)

    @property
    def dtype(self) -> np.dtype:
        """``float32``: the dtype every frame of this view is returned in."""
        return np.dtype(np.float32)

    def close(self) -> None:
        """Close the underlying raw movie."""
        close = getattr(self.raw, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SummedChannelsMovie:
    """Lazily evaluated, read-only view of several registered channels added
    together in the reference channel's coordinates.

    Frame ``t`` is ``sum_c warp(photons(movies[c][t]), transforms[c])``: every
    channel is converted to photons, mapped into the reference channel through
    its affine and accumulated. A molecule that is too dim to be detected in
    any single channel can still stand out in that sum, which is the point -
    identifying on the sum finds the molecules a per-channel identification
    would lose, and the resulting detections are already the cross-channel
    consensus that a joint multichannel fit needs (no linking step).

    Photons, not raw counts, are summed: the channels' baselines would
    otherwise pile up as a pedestal and their gains would weight the channels
    against each other, so a dim channel would contribute by its camera's gain
    rather than by its signal.

    This class is meant for spot identification only - fitting, spot cutting
    and photon conversion must always use the raw movie of the channel in
    question. Like ``GaussianFilteredMovie`` it deliberately does not implement
    the ``io.AbstractPicassoMovie`` interface, so that accidentally handing it
    to ``fit`` trips that function's input assertion instead of silently
    returning wrong photon numbers.

    Note that the minimum net gradient has to be re-tuned for the sum: adding
    ``n`` channels scales the net gradient roughly by ``n``, and the counts to
    photons conversion rescales it again.

    Parameters
    ----------
    movies : list of MovieLike
        One movie per channel, reference first (or at ``reference``). For
        split-FOV data the single movie is repeated once per region, exactly
        as :func:`fit_spline_split_fov` does.
    transforms : list of lib.FloatArray2D
        One ``(2, 3)`` affine per channel, mapping reference-channel
        coordinates into that channel (the calibration's
        ``channel_transforms``; the reference's is the identity). None entries
        are rejected: a channel that could not be registered must not be summed
        in at the identity, since that would smear the sum.
    camera_infos : list of dict, optional
        One camera info per channel, used to convert counts to photons. If
        None, the raw counts are summed instead (only sensible when the
        channels share a camera and a baseline).
    regions : list, optional
        Split-FOV: one ``[[y_min, x_min], [y_max, x_max]]`` rectangle per
        channel. Only the reference region of the canvas is filled (the rest
        stays zero), so the identified coordinates remain absolute frame
        coordinates and only the reference region is ever analyzed. If None,
        the channels are separate movies and the whole frame is filled.
    reference : int, optional
        Index of the reference channel in ``movies``/``transforms``/
        ``regions``. Default is 0.
    order : int, optional
        Spline order of the resampling, see ``scipy.ndimage.affine_transform``.
        Default is 1 (bilinear): higher orders overshoot on noisy
        single-molecule data, which shows up directly in the net gradient.
    camera_calibrations : list of dict, optional
        One sCMOS camera calibration per channel (``offset``, ``gain`` full
        frame maps). Applied before warping, since the maps live in each
        channel's own coordinates.

    Attributes
    ----------
    movies, transforms, camera_infos, regions, reference, order : as above
    frame_shape : tuple
        ``(Y, X)`` of the reference channel, i.e. of the summed frames.
    """

    # Summing a frame is stateless (there is no cache, and scipy.ndimage is
    # re-entrant), so reads never have to be serialized on the identification
    # lock. The wrapped movies serialize their own reads if they need to.
    supports_concurrent_reads = True

    def __init__(
        self,
        movies: list,
        transforms: list,
        *,
        camera_infos: list | None = None,
        regions: list | None = None,
        reference: int = 0,
        order: int = 1,
        camera_calibrations: list | None = None,
    ) -> None:
        n_channels = len(movies)
        if n_channels < 2:
            raise ValueError(
                "Summing channels needs at least two channels; identify on "
                "the movie itself to not sum at all."
            )
        if len(transforms) != n_channels:
            raise ValueError(
                f"Got {n_channels} channels but {len(transforms)} channel "
                "transforms."
            )
        unregistered = [c for c, t in enumerate(transforms) if t is None]
        if unregistered:
            raise ValueError(
                f"Channel(s) {unregistered} have no affine transform, so they "
                "cannot be mapped onto the reference channel. Register them "
                "(load a multichannel spline calibration or identify every "
                "channel so the transforms can be estimated) before summing."
            )
        if not 0 <= reference < n_channels:
            raise ValueError(
                f"The reference channel {reference} is not one of the "
                f"{n_channels} channels."
            )
        if regions is not None and len(regions) != n_channels:
            raise ValueError(
                f"Got {n_channels} channels but {len(regions)} regions."
            )
        self.movies = list(movies)
        self.transforms = [tform.from_dict(t) for t in transforms]
        self.camera_infos = (
            None if camera_infos is None else list(camera_infos)
        )
        self.camera_calibrations = (
            None if camera_calibrations is None else list(camera_calibrations)
        )
        self.regions = (
            None if regions is None else [_normalize_rect(r) for r in regions]
        )
        self.reference = int(reference)
        self.order = int(order)
        lengths = [len(m) for m in self.movies]
        self.n_frames = int(min(lengths))
        if self.n_frames == 0:
            raise ValueError("Cannot sum an empty movie.")
        if len(set(lengths)) > 1:
            warnings.warn(
                f"The channels have different lengths ({lengths}); the sum "
                f"covers their common {self.n_frames} frames.",
                stacklevel=2,
            )
        # one read to learn the frame geometry; movies do not agree on
        # whether they expose .shape (TiffMap does not)
        self.frame_shape = np.asarray(self.movies[self.reference][0]).shape
        # the window of the canvas that is filled: the reference region for
        # split-FOV data, the whole frame for separate channel movies
        if self.regions is None:
            self._window = [[0, 0], list(self.frame_shape)]
        else:
            self._window = self.regions[self.reference]

    def clear_cache(self) -> None:
        """Do nothing; this view caches nothing. Kept so that callers can
        treat all identification views alike."""

    def matches_regions(self, regions: list | None) -> bool:
        """Whether this sum was built for these region rectangles.

        The sum is only valid for the layout it was built for, and the corners
        of a rectangle may be given in any order - so callers compare through
        this rather than against ``regions`` directly.

        Parameters
        ----------
        regions : list or None
            The region rectangles to compare against, in any corner order.

        Returns
        -------
        matches : bool
        """
        if self.regions is None or regions is None:
            return self.regions is None and regions is None
        if len(regions) != len(self.regions):
            return False
        return all(
            _normalize_rect(rect) == mine
            for rect, mine in zip(regions, self.regions)
        )

    def _channel_photons(self, channel: int, index: int) -> np.ndarray:
        """One channel's frame in photons, in that channel's own
        coordinates."""
        frame = np.asarray(self.movies[channel][index])
        if self.camera_infos is None:
            return frame.astype(np.float32, copy=False)
        calibration = (
            None
            if self.camera_calibrations is None
            else self.camera_calibrations[channel]
        )
        offset = gain = None
        if calibration is not None:
            offset = calibration.get("offset")
            gain = calibration.get("gain")
        return _to_photons(frame, self.camera_infos[channel], offset, gain)

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if len(it) == 1:
                return self[it[0]]
            return self[it[0]][tuple(it[1:])]
        if isinstance(it, slice):
            return np.stack(
                [self[i] for i in range(*it.indices(self.n_frames))]
            )
        index = int(it)
        if index < 0:
            index += self.n_frames
        if not 0 <= index < self.n_frames:
            raise IndexError(
                f"Frame {it} is out of range for a movie with "
                f"{self.n_frames} frames."
            )
        canvas = np.zeros(self.frame_shape, dtype=np.float32)
        (y0, x0), (y1, x1) = self._window
        out_shape = (y1 - y0, x1 - x0)
        for c in range(len(self.movies)):
            image = self._channel_photons(c, index)
            if self.transforms[c].is_identity():
                # the channel already is in reference coordinates: take its
                # pixels as they are rather than resampling them
                canvas[y0:y1, x0:x1] += image[y0:y1, x0:x1]
                continue
            # The output grid *is* the reference frame, so the reference ->
            # channel transform is already the pull map that resampling needs
            # - no inversion. Sample only the window, i.e. shift the output
            # origin into it.
            canvas[y0:y1, x0:x1] += tform.warp_image(
                image,
                self.transforms[c],
                output_shape=out_shape,
                origin=(y0, x0),
                order=self.order,
                cval=0.0,
                dtype=np.float32,
            )
        return canvas

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self) -> int:
        return self.n_frames

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_frames, height, width)``, as a raw movie's."""
        return (self.n_frames, *self.frame_shape)

    @property
    def dtype(self) -> np.dtype:
        """``float32``: the dtype every frame of this view is returned in."""
        return np.dtype(np.float32)

    def close(self) -> None:
        """Close every underlying channel movie, each one only once."""
        closed = []
        for movie in self.movies:
            # split-FOV repeats one movie per channel; close it once
            if any(movie is other for other in closed):
                continue
            closed.append(movie)
            close = getattr(movie, "close", None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def identify_in_frame(
    frame: lib.IntArray2D,
    minimum_ng: float | list | np.ndarray,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Identify local maxima in a single frame within optionally
    specified subregion(s) (ROI) and calculate the net gradient at those
    maxima.

    Parameters
    ----------
    frame : lib.IntArray2D
        An image frame, 2D array of shape (Y, X).
    minimum_ng : float or sequence of float
        Minimum net gradient value to consider a maximum as valid. A
        sequence gives each ROI in ``roi`` its own threshold (split-FOV
        regions are separate channels and need not share a brightness
        scale); it must have one value per ROI.
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Note that the origin of the image is in the top-left corner.
        Default is None.

    Returns
    -------
    y : lib.IntArray1D
        y-coordinates of the identified maxima.
    x : lib.IntArray1D
        x-coordinates of the identified maxima.
    net_gradient : lib.FloatArray1D
        Net gradient values at the identified maxima. The shape is
        (len(y),).
    """
    rois = _as_roi_list(roi)
    if rois is None:
        image = np.float32(frame)  # otherwise numba goes crazy
        return identify_in_image(image, _as_ng_list(minimum_ng, 1)[0], box)
    minimum_ngs = _as_ng_list(minimum_ng, len(rois))
    height, width = frame.shape
    # pad each ROI to identify at the border
    pad = int(box / 2) + 1
    ys, xs, ngs = [], [], []
    for roi_index, ((y0, x0), (y1, x1)) in enumerate(rois):
        py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
        py1, px1 = min(y1 + pad, height), min(x1 + pad, width)
        image = np.float32(frame[py0:py1, px0:px1])  # numba needs float32!
        y, x, net_gradient = identify_in_image(
            image, minimum_ngs[roi_index], box
        )
        y += py0  # offset back to global frame coordinates
        x += px0
        # keep only maxima centered inside the actual ROI
        inside = (y >= y0) & (y < y1) & (x >= x0) & (x < x1)
        ys.append(y[inside])
        xs.append(x[inside])
        ngs.append(net_gradient[inside])
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(ngs)


def identify_by_frame_number(
    movie: MovieLike,
    minimum_ng: float | list | np.ndarray,
    box: int,
    frame_number: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
    lock: threading.Lock | None = None,
) -> pd.DataFrame:
    """Identify local maxima in a specific frame of a movie and
    calculate the net gradient at those maxima. Optionally, a lock can
    be used to ensure thread safety when accessing the movie data.

    Parameters
    ----------
    movie : MovieLike
        A 3D array representing the movie of shape (N, Y, X), where N is
        the number of frames, Y is the height, and X is the width.
    minimum_ng : float or sequence of float
        Minimum net gradient value to consider a maximum as valid. A
        sequence gives each ROI its own threshold, one value per ROI (see
        :func:`identify_in_frame`).
    box : int
        Size of the box used for calculating the gradient. Should be
        an odd integer.
    frame_number : int
        The index of the frame in the movie sequence to be processed.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Note that the origin of the image is in the top-left corner.
        Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the identification. A single
        ``(min, max)`` tuple restricts identification to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is processed if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    lock : threading.Lock, optional
        If provided, this lock will be used to ensure thread safety when
        accessing the movie data. This is useful in a multithreaded
        environment. Default is None.

    Returns
    -------
    identifications : pd.DataFrame
        DataFrame containing the frame number, x and y coordinates of
        the identified maxima, and their net gradient.
    """
    # check frame bounds before reading, so that frames that are skipped
    # anyway cost nothing (a TemporalMedianMovie would otherwise compute a
    # whole temporal window for them)
    if not lib.frame_in_bounds(frame_number, frame_bounds, len(movie)):
        return pd.DataFrame(
            {
                "frame": pd.Series(dtype=int),
                "x": pd.Series(dtype=int),
                "y": pd.Series(dtype=int),
                "net_gradient": pd.Series(dtype=np.float32),
            }
        )
    # Movies that read each frame through their own per-thread file
    # handle (TiffMap, STKMovie and the multi-file maps) or a memory map
    # are safe to read concurrently, so they skip the shared lock. This
    # lets several frame reads be in flight at once, which hides per-frame
    # I/O latency on network storage. Formats whose readers are not
    # reentrant stay serialized behind the lock.
    concurrent = getattr(
        movie, "supports_concurrent_reads", False
    ) or isinstance(movie, np.memmap)
    if lock is not None and not concurrent:
        with lock:
            frame = movie[frame_number]
    else:
        frame = movie[frame_number]
    # identify
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * np.ones(len(x))
    identifications = pd.DataFrame(
        {
            "frame": frame.astype(int),
            "x": x.astype(int),
            "y": y.astype(int),
            "net_gradient": net_gradient.astype(np.float32),
        }
    )
    return identifications


def _identify_worker(
    movie: MovieLike,
    current: list[int],
    minimum_ng: float | list | np.ndarray,
    box: int,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None,
    frame_bounds: tuple[int, int] | list | None,
    lock: threading.Lock | None,
) -> list[pd.DataFrame]:
    """Worker function for identifying local maxima in a movie. This
    function is designed to be run in a separate thread and processes
    each frame independently."""
    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(
            identify_by_frame_number(
                movie,
                minimum_ng,
                box,
                index,
                roi=roi,
                frame_bounds=frame_bounds,
                lock=lock,
            )
        )


def identifications_from_futures(
    futures: list[multiprocessing.pool.Future],
) -> pd.DataFrame:
    """Collect the results from a list of futures and combines them
    into a single ``DataFrame``.

    Parameters
    ----------
    futures : list of multiprocessing.pool.Future's
        A list of futures representing the asynchronous tasks.

    Returns
    -------
    ids : pd.DataFrame
        Data frame containing the combined results from
        all futures. Contains fields ``frame``, ``x``, ``y``, and
        ``net_gradient``.
    """
    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = list(chain(*ids_list_of_lists))
    ids = pd.concat(ids_list, ignore_index=True)
    # stable sort: frames are processed by racing threads, so rows can
    # arrive in a scheduling-dependent order; a non-stable sort would let
    # that non-determinism leak into the relative order of same-frame spots
    ids.sort_values(by="frame", kind="stable", inplace=True)
    return ids


def identify_async(
    movie: MovieLike,
    minimum_ng: float | list | np.ndarray,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
) -> tuple[list[int], list[multiprocessing.pool.Future]]:
    """Asynchronously (i.e., using multithreading) identify local
    maxima in a movie using multiple threads. This function divides the
    work among a specified number of threads.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    minimum_ng : float or sequence of float
        The minimum net gradient for a spot to be considered. A
        sequence gives each ROI its own threshold, one value per ROI
        (see :func:`identify_in_frame`).
    box : int
        The size of the box to extract around each spot.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the identification. A single
        ``(min, max)`` tuple restricts identification to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is processed if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.

    Returns
    -------
    current : list[int]
        A list of frame indices representing the current processing
        state.
    f : list[multiprocessing.pool.Future]
            A list of futures representing the asynchronous tasks.
    """
    # Use the user settings to define the number of workers that are being used
    settings = io.load_user_settings()

    # avoid the problem when cpu_utilization is not set
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
    except KeyError:
        cpu_utilization = 0.8

    if isinstance(cpu_utilization, float):
        if cpu_utilization >= 1:
            cpu_utilization = 0.8
    else:
        print("CPU utilization was not set. Setting to 0.8")
        cpu_utilization = 0.8
    settings["Localize"]["cpu_utilization"] = cpu_utilization
    io.save_user_settings(settings)

    n_workers = lib.n_workers(cpu_utilization)

    lock = threading.Lock()
    current = [0]
    executor = ThreadPoolExecutor(n_workers)
    f = [
        executor.submit(
            _identify_worker,
            movie,
            current,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            lock,
        )
        for _ in range(n_workers)
    ]
    executor.shutdown(wait=False)
    return current, f


def _identify_threaded(
    movie,
    minimum_ng,
    box,
    roi,
    frame_bounds,
    progress_callback,
    abort_callback,
):
    """Run identify_async and drive its progress loop.

    Returns the identifications, or None if aborted.
    """
    N = len(movie)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(total=N, desc="Identifying spots", unit="frame")
        if use_tqdm
        else None
    )
    current, futures = identify_async(
        movie, minimum_ng, box, roi=roi, frame_bounds=frame_bounds
    )
    last = 0
    while current[0] < N:
        if abort_callback is not None and abort_callback():
            for f in futures:
                f.cancel()
            if use_tqdm:
                iter_range.close()
            return None
        if use_tqdm:
            iter_range.update(current[0] - last)
            last = current[0]
        elif callable(progress_callback):
            progress_callback(current[0])
        time.sleep(0.2)
    if use_tqdm:
        iter_range.update(N - last)
        iter_range.close()
    return identifications_from_futures(futures)


def _identify_serial(
    movie,
    minimum_ng,
    box,
    roi,
    frame_bounds,
    progress_callback,
):
    """Identify spots frame-by-frame in the current thread."""
    N = len(movie)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(range(N), desc="Identifying spots", unit="frame")
        if use_tqdm
        else range(N)
    )
    identifications = []
    for i in iter_range:
        identifications.append(
            identify_by_frame_number(
                movie,
                minimum_ng,
                box,
                i,
                roi=roi,
                frame_bounds=frame_bounds,
            )
        )
        if callable(progress_callback):
            progress_callback(i)
    ids = pd.concat(identifications, ignore_index=True)
    ids.sort_values(by="frame", kind="stable", inplace=True)
    return ids


def identify(
    movie: MovieLike,
    minimum_ng: float | list | np.ndarray,
    box: int,
    *,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
    threaded: bool = True,
    temporal_median_window: int | None = None,
    temporal_median_stride: int | None = None,
    gaussian_filter_sigma: float | None = None,
    progress_callback: (
        Callable[[list[int]], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Identify local maxima in a movie and calculate the net
    gradient at those maxima. This function can run in a threaded or
    non-threaded mode.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    minimum_ng : float or sequence of float
        The minimum net gradient for a spot to be considered. A
        sequence gives each ROI its own threshold, one value per ROI
        (see :func:`identify_in_frame`).
    box : int
        The size of the box to extract around each spot.
    roi : tuple or list of tuples, optional
        Region(s) of interest (ROI). A single ROI is a tuple of two
        tuples, where the first contains the start coordinates
        (y_start, x_start) and the second the end coordinates
        (y_end, x_end). A list of such tuples restricts identification to
        several (disjoint) regions. If None, the entire frame is used.
        Note that the origin of the image is in the top-left corner.
        Default is None.
    frame_bounds : tuple, list of tuples, optional
        Frame numbers to consider for the identification. A single
        ``(min, max)`` tuple restricts identification to one contiguous,
        inclusive range; a list of such tuples restricts it to several
        (disjoint) segments, where a frame is processed if it falls in any
        segment. If None, all frames are used. If only min or max is to be
        specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    threaded : bool, optional
        Whether to use threading for the identification process. Default
        is True.
    temporal_median_window : int or None, optional
        If given (and non-zero), a temporal median background is
        subtracted from every frame before identifying, using a window of
        this many frames, see ``TemporalMedianMovie``. The filter applies
        to the identification only - the returned coordinates refer to
        the raw movie, which is what the spots must be fitted on. Note
        that ``minimum_ng`` has to be re-tuned when this is switched on
        or off, since subtracting a background changes the scale of the
        net gradient. Default is None (no filtering).
    temporal_median_stride : int or None, optional
        Spacing between the frames at which the temporal median is
        evaluated, see ``TemporalMedianMovie``. None (the default) uses
        ``temporal_median_window``, which is the fastest setting.
    gaussian_filter_sigma : float or None, optional
        If given (and non-zero), every frame is smoothed with a Gaussian
        of this standard deviation (in camera pixels) before identifying,
        see ``GaussianFilteredMovie``. This merges the several local
        maxima of a spot that is not Gaussian-shaped into one. Applied
        after the temporal median filter, if both are used. The filter
        applies to the identification only - the returned coordinates
        refer to the raw movie, which is what the spots must be fitted
        on. Note that ``minimum_ng`` has to be re-tuned when this is
        changed, since smoothing lowers gradient magnitudes. Default is
        None (no filtering).
    progress_callback : callable, "console" or None, optional
        A callback function to report the progress of the identification
        process. If "console", progress will be printed to the console.
        If None, no progress will be reported. Default is None.
    abort_callback : callable, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted. Default is
        None.
    return_info : bool, optional
        Whether to return additional information about the fitting
        process. Default is True. If True, a tuple of (locs, info) is
        returned. In v0.12.0 return_info will be removed and the
        function will always return info.

    Returns
    -------
    ids : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    info : dict, optional
        Additional information about the identification process, such as
        the time taken for identification. Only returned if `return_info`
        is True.
    """
    if not return_info:
        # TODO: remove in v0.12.0
        lib.deprecation_warning(
            "In version 0.12, return_info argument will be removed such "
            "that picasso.localize.localize() will always return both "
            "the localizations and the metadata dictionary."
        )
    roi_pad = identification_roi_pad(box, gaussian_filter_sigma)
    if temporal_median_window:
        # note that identify_async() is not wrapped: callers driving the
        # thread pool themselves build the filtered views explicitly
        movie = TemporalMedianMovie(
            movie,
            temporal_median_window,
            stride=temporal_median_stride,
            roi=roi,
            roi_pad=roi_pad,
        )
    # temporal median first, then smoothing: the Gaussian is meant to merge
    # the maxima of one spot, not those of the background it sits on
    if gaussian_filter_sigma:
        movie = GaussianFilteredMovie(movie, gaussian_filter_sigma)
    if threaded:
        ids = _identify_threaded(
            movie,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            progress_callback,
            abort_callback,
        )
        if ids is None:
            return
    else:
        ids = _identify_serial(
            movie,
            minimum_ng,
            box,
            roi,
            frame_bounds,
            progress_callback,
        )
    if return_info:
        info = {
            "Generated by": f"Picasso: v{__version__} Identify",
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
            "ROI": roi,
            "Frame Bounds": frame_bounds,
            "Temporal Median Window": int(temporal_median_window or 0),
            "Gaussian Filter Sigma": float(gaussian_filter_sigma or 0.0),
        }
        return ids, info
    else:
        return ids


def identify_multichannel_sum(
    movies: list,
    minimum_ng: float,
    box: int,
    transforms: list,
    *,
    camera_infos: list | None = None,
    regions: list | None = None,
    reference: int = 0,
    camera_calibrations: list | None = None,
    roi: tuple[tuple[int, int], tuple[int, int]] | list | None = None,
    frame_bounds: tuple[int, int] | list | None = None,
    threaded: bool = True,
    temporal_median_window: int | None = None,
    temporal_median_stride: int | None = None,
    gaussian_filter_sigma: float | None = None,
    order: int = 1,
    progress_callback: (
        Callable[[list[int]], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Identify spots on the *sum* of registered channels rather than in each
    channel on its own.

    Every channel is converted to photons, mapped into the reference channel
    through its affine transform and added up (see
    :class:`SummedChannelsMovie`); the summed movie is then identified exactly
    as a single-channel movie is. This is the mode for data where a channel is
    too dim to detect in by itself: the molecule is found from the combined
    signal, and the resulting detections are the reference-channel positions a
    joint multichannel fit takes directly - they are already the cross-channel
    consensus, so a :func:`filter_linked_identifications` step would only throw
    away exactly the molecules this mode is meant to recover.

    The temporal median and Gaussian filters apply to the *sum*, matching the
    "identify on the sum" semantics: it is the summed image that is being
    searched for maxima, so it is the summed image that is background
    subtracted and smoothed.

    Parameters
    ----------
    movies : list of MovieLike
        One movie per channel, reference first (or at ``reference``). For
        split-FOV data pass the single movie repeated once per region.
    minimum_ng : float
        Minimum net gradient, a single value: there is one summed image. It has
        to be re-tuned relative to the per-channel thresholds, since the sum is
        in photons and over all channels (see :class:`SummedChannelsMovie`).
    box : int
        The size of the box to extract around each spot.
    transforms : list of lib.FloatArray2D
        One ``(2, 3)`` reference->channel affine per channel; see
        :class:`SummedChannelsMovie`.
    camera_infos, regions, reference, camera_calibrations, order
        Passed to :class:`SummedChannelsMovie`.
    roi : tuple or list of tuples, optional
        Region(s) to identify in, in reference-channel coordinates. Defaults to
        the reference region for split-FOV data (the only part of the canvas
        that is filled) and to the whole frame otherwise.
    frame_bounds, threaded, temporal_median_window, temporal_median_stride, \
gaussian_filter_sigma, progress_callback, abort_callback
        As in :func:`identify`.

    Returns
    -------
    ids : pd.DataFrame
        The identified spots in *reference-channel* coordinates, with fields
        `frame`, `x`, `y` and `net_gradient` (the latter measured on the sum).
    info : dict
        Identification metadata, with the summing recorded under
        ``"Identification Mode"``, ``"Sum Channel Transforms"`` and
        ``"Sum Regions"``.
    """
    summed = SummedChannelsMovie(
        movies,
        transforms,
        camera_infos=camera_infos,
        regions=regions,
        reference=reference,
        order=order,
        camera_calibrations=camera_calibrations,
    )
    if roi is None and regions is not None:
        # only the reference region of the canvas holds the sum
        roi = [summed.regions[summed.reference]]
    result = identify(
        summed,
        minimum_ng,
        box,
        roi=roi,
        frame_bounds=frame_bounds,
        threaded=threaded,
        temporal_median_window=temporal_median_window,
        temporal_median_stride=temporal_median_stride,
        gaussian_filter_sigma=gaussian_filter_sigma,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
    )
    if result is None:  # aborted
        return None
    ids, info = result
    info["Generated by"] = f"Picasso: v{__version__} Identify"
    info["Identification Mode"] = "sum"
    info["Sum Channel Transforms"] = [t.to_dict() for t in summed.transforms]
    info["Sum Regions"] = summed.regions
    info["Sum Reference Channel"] = summed.reference
    return ids, info


def picks_to_identifications(
    picks: list[tuple],
    *,
    n_frames: int | None = None,
    drift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert circular picks (from Picasso: Render) to identifications.
    Only circular picks are allowed.

    Parameters
    ----------
    picks : list of tuples
        List of circular picks positions (centers). See
        ``io.load_picks``.
    n_frames : int, optional
        Number of frames in the acquisition movie. If None is given,
        it will be extracted from the drift file (if provided).
        Otherwise, an error is raised.
    drift : pd.DataFrame or None, optional
        A data frame of length n_frames and with columns 'x' and 'y'.
        Used to adjust the positions of identifications throughout
        acquisition. Only x and y drift is used; if 'z' is present, it
        is ignored.

    Returns
    -------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`. Note that `net_gradient`
        is a dummy value.

    Raises
    ------
    ValueError
        If `n_frames` and `drift` are not provided.
    """
    assert isinstance(picks, (list, tuple)), "picks must be a list or a tuple."
    assert all([len(_) == 2 for _ in picks]), (
        "Circular picks are required. Each element in 'picks' must "
        "contain two numbers (x and y coordinates)."
    )
    if isinstance(drift, pd.DataFrame):
        assert all(
            col in drift.columns for col in ["x", "y"]
        ), "Drift data frame must contain 'x' and 'y' columns."
    if n_frames is None:
        if drift is None:
            raise ValueError(
                "n_frames must be given if no drift file is provided"
            )
        else:
            n_frames = len(drift)
    else:
        assert isinstance(n_frames, int), "n_frames must be an integer."
        if drift is not None:
            assert n_frames == len(drift), (
                f"{n_frames} frames were provided but the drift suggests"
                f" {len(drift)} frames."
            )
    return _picks_to_identifications(picks, n_frames, drift)


def _picks_to_identifications(
    picks: list[tuple],
    n_frames: int,
    drift: pd.DataFrame | None,
) -> pd.DataFrame:
    """Convert circular picks to identifications, can be drift-corrected.
    Assumes correct inputs. See ``picks_to_identifications`` for more
    details."""
    data = []
    n_id = 0
    for pick_x, pick_y in picks:
        # drifted:
        xloc = np.ones((n_frames,), dtype=float) * pick_x
        yloc = np.ones((n_frames,), dtype=float) * pick_y
        if drift is not None:
            xloc += drift["x"].to_numpy()
            yloc += drift["y"].to_numpy()

        frames = np.arange(n_frames)
        gradient = np.ones(n_frames) + 100
        n_id_all = np.ones(n_frames) + n_id
        temp = np.array([frames, xloc, yloc, gradient, n_id_all])
        data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
        n_id += 1

    data = [item for sublist in data for item in sublist]
    identifications = pd.DataFrame(
        {
            "frame": [item[0] for item in data],
            "x": [item[1] for item in data],
            "y": [item[2] for item in data],
            "net_gradient": [item[3] for item in data],
            "n_id": [item[4] for item in data],
        }
    )
    identifications.sort_values(
        by="frame",
        inplace=True,
        kind="quicksort",
    )
    return identifications


def locs_to_identifications(
    locs: pd.DataFrame,
    movie_info: list[dict],
    n_frames: int,
) -> pd.DataFrame:
    """Convert localizations to identifications.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    movie_info : list of dicts
        Movie file metadata.
    n_frames : int
        Number of frames around localizations that are to be used for
        extracting identifications.

    Returns
    -------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`. Note that `net_gradient`
        is a dummy value.
    """
    assert isinstance(
        locs, pd.DataFrame
    ), "Localizations must be a pandas data frame"
    assert (
        isinstance(n_frames, int) and n_frames >= 0
    ), "n_frames must be a non-negative integer"
    max_frames = lib.get_from_metadata(movie_info, "Frames", raise_error=True)
    data = []
    n_id = 0
    for _, element in locs.iterrows():
        currframe = element["frame"]
        if currframe > n_frames and currframe < (max_frames - n_frames):
            xloc = np.ones((2 * n_frames + 1,), dtype=float) * element["x"]
            yloc = np.ones((2 * n_frames + 1,), dtype=float) * element["y"]
            frames = np.arange(
                currframe - n_frames,
                currframe + n_frames + 1,
            )
            gradient = np.ones(2 * n_frames + 1) + 100
            n_id_all = np.ones(2 * n_frames + 1) + n_id
            temp = np.array([frames, xloc, yloc, gradient, n_id_all])
            data.append([tuple(temp[:, j]) for j in range(temp.shape[1])])
        n_id += 1
    data = [item for sublist in data for item in sublist]
    identifications = pd.DataFrame(
        {
            "frame": [item[0] for item in data],
            "x": [item[1] for item in data],
            "y": [item[2] for item in data],
            "net_gradient": [item[3] for item in data],
            "n_id": [item[4] for item in data],
        }
    )
    return identifications


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba_into(
    movie: lib.IntArray3D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
    start: int,
) -> None:
    """Extract spots out of a movie directly into a preallocated array.

    Spots are written into `spots[start : start + len(ids_x)]`, avoiding
    an intermediate allocation and copy. Used for chunked, progress-aware
    cutting.
    """
    r = int(box / 2)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[start + id] = movie[
            frame, yc - r : yc + r + 1, xc - r : xc + r + 1
        ]


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba(
    movie: lib.IntArray3D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
) -> lib.IntArray3D:
    """Extract the spots out of a movie using Numba for performance."""
    n_spots = len(ids_x)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    _cut_spots_numba_into(movie, ids_frame, ids_x, ids_y, box, spots, 0)
    return spots


@numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame: lib.IntArray2D,
    frame_number: int,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    r: int,
    start: int,
    N: int,
    spots: lib.IntArray3D,
) -> int:
    """Extract spots from a movie frame."""
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        if ids_frame[j] < frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return j


@numba.jit(nopython=True, nogil=True, cache=False)
def _cut_spots_single_frame_into(
    frame: lib.IntArray2D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    r: int,
    start: int,
    end: int,
    spots: lib.IntArray3D,
) -> None:
    """Cut every spot in ``ids_[start:end]`` out of a single 2D frame
    into ``spots[start:end]``.

    ``nogil=True`` lets several threads cut (and, more importantly, read
    their frame) at the same time. Each call writes a disjoint slice of
    ``spots``, so no locking is needed around the writes."""
    for j in range(start, end):
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]


def _n_io_workers() -> int:
    """Number of threads to use for I/O-bound frame reading, derived from
    the same ``cpu_utilization`` user setting as identification."""
    settings = io.load_user_settings()
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
    except KeyError:
        cpu_utilization = 0.8
    if not isinstance(cpu_utilization, float) or cpu_utilization >= 1:
        cpu_utilization = 0.8
    return lib.n_workers(cpu_utilization)


@numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(
    movie: MovieLike,
    l_mov: lib.IntArray1D,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    l_mov : lib.IntArray1D
        Length of the movie, a 1D array with a single element.
    ids_frame, ids_x, ids_y : lib.IntArray1D
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : lib.IntArray3D
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.

    Returns
    -------
    spots : lib.IntArray3D
        3D array with extracted spots of shape (k, box, box), where k is
        the number of spots identified.
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number in range(l_mov[0]):
        frame = movie[frame_number, :, :]
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots_framebyframe(
    movie: MovieLike,
    ids_frame: lib.IntArray1D,
    ids_x: lib.IntArray1D,
    ids_y: lib.IntArray1D,
    box: int,
    spots: lib.IntArray3D,
    progress_callback: Callable[[int], None] | None = None,
):
    """Extract the spots out of a movie frame by frame.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    ids_frame, ids_x, ids_y : lib.IntArray1D
        1D arrays containing spot positions in the image data.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    spots : lib.IntArray3D
        3D array to store the cut spots, with shape (k, box, box),
        where k is the number of spots identified.
    progress_callback : callable or None, optional
        If a callable is provided, it is called after each frame with
        the cumulative number of spots cut so far. Default is None.

    Returns
    -------
    spots : lib.IntArray3D
        3D array with extracted spots of shape (k, box, box), where k is
        the number of spots identified.

    Notes
    -----
    When the movie supports concurrent reads (its frames are read
    through a per-thread file handle), frames are read and cut in
    parallel, which hides per-frame I/O latency the same way threaded
    identification does. ``ids_frame`` is assumed to be sorted (as
    ``identify`` returns it), so each frame maps to one contiguous slice
    of ``spots``.
    """
    r = int(box / 2)
    N = len(ids_frame)
    n_frames = len(movie)

    # Since ids are frame-sorted, frame f's spots are the contiguous
    # slice spots[starts[f]:ends[f]].
    starts = np.searchsorted(ids_frame, np.arange(n_frames), side="left")
    ends = np.append(starts[1:], N)

    if getattr(movie, "supports_concurrent_reads", False):
        done = [0]
        progress_lock = threading.Lock()

        def _read_and_cut(frame_number: int) -> None:
            start = int(starts[frame_number])
            end = int(ends[frame_number])
            frame = movie[frame_number]
            _cut_spots_single_frame_into(
                frame, ids_x, ids_y, r, start, end, spots
            )
            if callable(progress_callback):
                with progress_lock:
                    done[0] += end - start
                    progress_callback(done[0])

        with ThreadPoolExecutor(_n_io_workers()) as executor:
            # consume the iterator so exceptions propagate
            list(executor.map(_read_and_cut, range(n_frames)))
        return spots

    # Serial fallback for movies whose readers are not reentrant
    # (e.g. ND2/CZI/LIF).
    cum = 0
    for frame_number in range(n_frames):
        start = int(starts[frame_number])
        end = int(ends[frame_number])
        frame = movie[frame_number]
        _cut_spots_single_frame_into(frame, ids_x, ids_y, r, start, end, spots)
        cum += end - start
        if callable(progress_callback):
            progress_callback(cum)
    return spots


def _cut_spots(
    movie: MovieLike,
    ids: pd.DataFrame,
    box: int,
    progress_callback: Callable[[int], None] | None = None,
) -> lib.IntArray3D:
    """Cut out spots from a movie based on the identified positions.

    If a callable `progress_callback` is provided, it is called with the
    cumulative number of spots cut so far, allowing the cutting progress
    to be tracked.
    """
    N = len(ids)
    ids_frame = ids["frame"].to_numpy()
    ids_x = ids["x"].to_numpy()
    ids_y = ids["y"].to_numpy()
    if isinstance(movie, np.ndarray):
        if not callable(progress_callback):
            return _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box)
        # cut in chunks so that progress can be reported; spots are
        # written directly into the output array to avoid an extra copy
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        chunk = max(1, N // 100)
        for chunk_start in range(0, N, chunk):
            chunk_end = min(chunk_start + chunk, N)
            _cut_spots_numba_into(
                movie,
                ids_frame[chunk_start:chunk_end],
                ids_x[chunk_start:chunk_end],
                ids_y[chunk_start:chunk_end],
                box,
                spots,
                chunk_start,
            )
            progress_callback(chunk_end)
        return spots
    elif isinstance(movie, io.ND2Movie) and movie.use_dask:
        """Assumes that identifications are in order of frames!"""
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = da.apply_gufunc(
            _cut_spots_daskmov,
            "(p,n,m),(b),(k),(k),(k),(),(k,l,l)->(k,l,l)",
            movie.data,
            np.array([len(movie)]),
            ids_frame,
            ids_x,
            ids_y,
            box,
            spots,
            output_dtypes=[movie.dtype],
            allow_rechunk=True,
        ).compute()
        if callable(progress_callback):
            progress_callback(N)
        return spots
    else:
        """Assumes that identifications are in order of frames!"""
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie,
            ids_frame,
            ids_x,
            ids_y,
            box,
            spots,
            progress_callback=progress_callback,
        )
        return spots


def _cut_map(
    image: lib.FloatArray2D, ids: pd.DataFrame, box: int
) -> lib.FloatArray3D:
    """Cut ``(box, box)`` patches out of a full-frame map, one per spot.

    The geometry is that of :func:`_cut_spots_numba_into`
    (``image[yc - r : yc + r + 1, xc - r : xc + r + 1]``), so a patch lines up
    pixel for pixel with the corresponding spot. A camera map has no frame
    axis, so this is a plain fancy-index rather than another ``_cut_spots``
    backend.
    """
    r = box // 2
    offsets = np.arange(box)
    rows = ids["y"].to_numpy()[:, None] - r + offsets  # (k, box)
    cols = ids["x"].to_numpy()[:, None] - r + offsets  # (k, box)
    return np.ascontiguousarray(
        image[rows[:, :, None], cols[:, None, :]], dtype=np.float32
    )


def _sensitivity(
    camera_info: dict, gain_patch: lib.FloatArray3D | None
) -> float | lib.FloatArray3D:
    """Counts-to-photoelectrons factor, scalar or per pixel.

    Picasso's ``Sensitivity`` is electrons per A/D count, i.e. the reciprocal
    of the amplification gain ``g`` a camera calibration measures in ADU per
    photoelectron (Huang et al. 2013, Supplementary Note Section 2.3)."""
    if gain_patch is None:
        return camera_info["Sensitivity"]
    return 1.0 / gain_patch


def _to_photons(
    spots: lib.FloatArray3D,
    camera_info: dict,
    offset: lib.FloatArray3D | None = None,
    gain_patch: lib.FloatArray3D | None = None,
) -> lib.FloatArray3D:
    """Convert the cut spots to photon counts based on camera
    information.

    ``offset`` and ``gain_patch`` are per-spot ``(k, box, box)`` patches of an
    sCMOS camera calibration; each overrides the corresponding scalar
    (``Baseline``, ``Sensitivity``) where it is given.
    """
    spots = np.float32(spots)
    baseline = camera_info["Baseline"] if offset is None else offset
    sensitivity = _sensitivity(camera_info, gain_patch)
    gain = camera_info["Gain"]
    # since v0.6.0: remove quantum efficiency to better reflect precision
    # qe = camera_info["Qe"]
    return (spots - baseline) * sensitivity / (gain)


def _variance_to_photons(
    variance: lib.FloatArray3D,
    camera_info: dict,
    gain_patch: lib.FloatArray3D | None = None,
) -> lib.FloatArray3D:
    """Convert a readout variance from ADU squared to photoelectrons squared.

    :func:`_to_photons` scales counts by ``Sensitivity / Gain``, so a variance
    scales by its square. This is Huang et al.'s ``var / g^2``, the quantity
    the noise model adds to both the data and the model mean.
    """
    sensitivity = _sensitivity(camera_info, gain_patch)
    gain = camera_info["Gain"]
    return np.float32(variance) * (sensitivity / gain) ** 2


def _validate_camera_calibration(
    camera_calibration: dict | None, movie, camera_info: dict
) -> None:
    """Check a camera calibration against the movie it will be applied to.

    The maps are indexed with the identifications' absolute frame coordinates,
    so a calibration recorded with a different camera ROI or binning would
    silently read the wrong pixels. Checked once, before any spot is cut.
    """
    if camera_calibration is None:
        return
    for name in ("offset", "variance"):
        if camera_calibration.get(name) is None:
            raise ValueError(
                f"Invalid camera calibration: missing the '{name}' map. Build "
                "one with picasso.scmos.calibrate_scmos or load it with "
                "picasso.io.load_camera_calibration."
            )
    dims = io._readable_movie_dims(movie)
    height, width = dims.get("Height"), dims.get("Width")
    if height is None or width is None:
        shape = getattr(movie, "shape", None)
        if shape is not None and len(shape) == 3:
            height, width = int(shape[1]), int(shape[2])
    map_shape = np.shape(camera_calibration["offset"])
    if height is not None and width is not None:
        if map_shape != (height, width):
            raise ValueError(
                f"The camera calibration was computed on {map_shape[0]}x"
                f"{map_shape[1]} frames but this movie is {height}x{width}. "
                "Compute the offset/variance maps from a dark movie acquired "
                "with the same camera ROI and binning."
            )
    if (
        camera_calibration.get("gain") is not None
        and camera_info.get("Gain", 1) > 1
    ):
        warnings.warn(
            "A per-pixel camera calibration was supplied together with an EM "
            "gain > 1. The sCMOS noise model of Huang et al. (2013) assumes a "
            "non-multiplying sensor; the EMCCD excess-noise factor of 2 is "
            "still applied to every uncertainty, which double-counts the "
            "noise. Set the EM gain to 1 for an sCMOS camera.",
            RuntimeWarning,
        )


def _mean_readout_variance(
    variance: lib.FloatArray3D | None,
) -> lib.FloatArray1D | float:
    """Per-spot mean readout variance, for the closed-form precisions.

    The Mortensen-family formulas describe a spatially uniform background, so
    a per-pixel map can only enter them through its mean over the fitting box.
    Returns 0.0 when there is no calibration, which leaves those formulas
    exactly as they were."""
    if variance is None:
        return 0.0
    return variance.reshape(len(variance), -1).mean(axis=1)


def _clip_for_mle(
    spots: lib.FloatArray3D, variance: lib.FloatArray3D | None
) -> lib.FloatArray3D:
    """Floor the data where the Poisson likelihood is defined.

    Camera offset subtraction pushes dim pixels below zero, and a Poisson
    likelihood is undefined there. Without a noise model the floor is zero, as
    it has always been. With one, the likelihood is evaluated on the *shifted*
    data ``d + var``, so the floor moves to ``-var``: clipping at zero instead
    would discard exactly the negative excursions readout noise creates, and
    bias the fitted background upward on the noisiest pixels - the opposite of
    what the noise model is for.
    """
    if variance is None:
        return np.maximum(spots, 0)
    return np.maximum(spots, -variance)


def camera_calibration_info(camera_calibration: dict | None) -> dict:
    """Provenance of an sCMOS camera calibration, for the saved metadata.

    Every caller that fits with a calibration must record this, or a
    localization file carries no trace of the noise model that produced it and
    two runs become indistinguishable after the fact. It lives here, rather
    than inline in ``fit``, because Picasso Localize rebuilds its own
    metadata when saving instead of using what ``fit`` returns, and the two
    must not drift apart.

    Parameters
    ----------
    camera_calibration : dict or None
        The loaded sCMOS camera calibration.

    Returns
    -------
    info : dict
        Metadata keys describing the calibration's path, frame count and the
        source of the offset and gain maps, plus its median offset, median
        variance and hot-pixel count when they were recorded. An empty dict
        when there is no calibration, so callers can ``update()``
        unconditionally.
    """
    if not camera_calibration:
        return {}
    info = {
        "Camera calibration path": camera_calibration.get("Path", "N/A"),
        "Camera calibration frames": camera_calibration.get("Frames"),
        "Camera offset source": "per-pixel map",
        "Camera gain source": (
            "per-pixel map"
            if camera_calibration.get("gain") is not None
            else "Sensitivity (scalar)"
        ),
    }
    for key in (
        "Offset median (ADU)",
        "Variance median (ADU^2)",
        "Hot pixels",
    ):
        if key in camera_calibration:
            info[f"Camera calibration {key}"] = camera_calibration[key]
    return info


def _seed_spots(
    spots: lib.FloatArray3D, variance: lib.FloatArray3D | None
) -> lib.FloatArray3D:
    """Spots to estimate the initial fit parameters from.

    The seeds take the background from the dimmest pixel of the ROI and the
    amplitude from the brightest minus the dimmest. That is fine on data
    floored at zero, but :func:`_clip_for_mle` floors at ``-var``, so on a
    noisy pixel the dimmest value can be tens of photons *below* zero. Seeding
    a background there makes the model mean negative across the whole ROI, the
    likelihood is then floored everywhere, the first Hessian is singular and
    the fit aborts - which cost about a third of all spots on a sensor with
    realistic hot pixels.

    The seed is only a starting point, and a negative background is never a
    sensible one, so it is estimated from the zero-floored data. The fit
    itself still runs on the ``-var`` floored data, where the shifted Poisson
    likelihood is defined. This also keeps the seed identical with and without
    a calibration, so any difference between the two fits comes from the noise
    model rather than from where they started.
    """
    if variance is None:
        return spots
    return np.maximum(spots, 0)


def get_spots(
    movie: MovieLike,
    identifications: pd.DataFrame,
    box: int,
    camera_info: dict,
    progress_callback: Callable[[int], None] | None = None,
    camera_calibration: dict | None = None,
    return_variance: bool = False,
) -> lib.FloatArray3D | tuple[lib.FloatArray3D, lib.FloatArray3D | None]:
    """Extract the spots from a movie based on the identified positions
    and convert camera signal to photon counts.

    Parameters
    ----------
    movie : MovieLike
        The input movie, read frame by frame.
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    progress_callback : callable or None, optional
        If a callable is provided, it is called with the cumulative
        number of spots cut so far, allowing the cutting progress to be
        tracked. Default is None.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration from ``picasso.scmos``. Its
        ``offset`` map replaces the scalar ``Baseline`` and, if present, its
        ``gain`` map replaces the scalar ``Sensitivity``. Default is None.
    return_variance : bool, optional
        Also return the per-spot readout variance in photoelectrons squared,
        cut from the same ROIs as the spots. None when no calibration was
        given. Default is False.

    Returns
    -------
    spots : lib.FloatArray3D
        A 3D numpy array containing the extracted spots, with shape
        (k, box, box), where k is the number of spots identified.
    variance : lib.FloatArray3D or None
        Only if ``return_variance``.
    """
    spots = _cut_spots(
        movie, identifications, box, progress_callback=progress_callback
    )
    offset = gain_patch = variance = None
    if camera_calibration is not None:
        offset = _cut_map(camera_calibration["offset"], identifications, box)
        if camera_calibration.get("gain") is not None:
            gain_patch = _cut_map(
                camera_calibration["gain"], identifications, box
            )
        if return_variance:
            variance = _variance_to_photons(
                _cut_map(camera_calibration["variance"], identifications, box),
                camera_info,
                gain_patch,
            )
    spots = _to_photons(spots, camera_info, offset, gain_patch)
    if return_variance:
        return spots, variance
    return spots


def locs_from_fits(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    CRLBs: lib.FloatArray2D,
    likelihoods: lib.FloatArray1D,
    iterations: lib.FloatArray1D,
    box: int,
) -> pd.DataFrame:
    """Convert the resulting localizations from the list of Futures
    into a data frame.

    .. deprecated:: 0.11
        Removed in Picasso 1.0. Left over from the old ``gaussmle`` GPU
        pipeline and unused since; use :func:`locs_from_fits_gauss`,
        which takes the fitters' ``theta`` layout directly.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    theta : lib.FloatArray2D
        The fitted Gaussian parameters for each spot (x, y positions,
        photon counts, background, single-emitter image size in x and
        y).
    CRLBs : lib.FloatArray2D
        The Cramer-Rao Lower Bounds for each fitted parameter.
    likelihoods : lib.FloatArray1D
        The log-likelihoods of the fitted models.
    iterations : lib.FloatArray1D
        The number of iterations taken to converge for each spot.
    box : int
        Size of the box used for fitting. Should be an odd integer.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. The fields include
        `frame`, `x`, `y`, `photons`, `sx`, `sy`, `bg`, `lpx`, `lpy`,
        `net_gradient`, `log_likelihood`, and `iterations`.
    """
    lib.deprecation_warning(
        "picasso.localize.locs_from_fits is deprecated and will be removed "
        "in Picasso 1.0. Use picasso.localize.locs_from_fits_gauss."
    )
    # box_offset = int(box / 2)
    y = theta[:, 0] + identifications["y"]  # - box_offset
    x = theta[:, 1] + identifications["x"]  # - box_offset
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])
    locs = pd.DataFrame(
        {
            "frame": identifications["frame"].astype(np.uint32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "photons": theta[:, 2].astype(np.float32),
            "sx": theta[:, 5].astype(np.float32),
            "sy": theta[:, 4].astype(np.float32),
            "bg": theta[:, 3].astype(np.float32),
            "lpx": lpx.astype(np.float32),
            "lpy": lpy.astype(np.float32),
            "net_gradient": (
                identifications["net_gradient"].astype(np.float32)
            ),
            "log_likelihood": likelihoods.astype(np.float32),
            "iterations": iterations.astype(np.int32),
        }
    )
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def fit(
    movie: LoadedMovie,
    *,
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    spline_calibration: dict | None = None,
    camera_calibration: dict | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    cut_progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Fit 2D localizations to a movie, given positions of the detected
    spots (identifications).

    Since v0.11.0: renamed from ``fit2D``, which is deprecated and will
    be removed in v0.12.0, together with its unused ``movie_info`` and
    ``mle_method`` arguments. Only the movie is accepted positionally.

    Parameters
    ----------
    movie : LoadedMovie
        The input movie, as loaded by ``picasso.io.load_movie``.
    camera_info : dict
        A dictionary containing camera information: "Baseline",
        "Sensitivity", "Gain" and "Pixelsize".
    identifications : pd.DataFrame
        Data frame containing the identified spots. Contains fields
        `frame`, `x`, `y`, and `net_gradient`.
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    fitting_method : {"gausslq", "gausslq-spherical", "gausslq-rotated", \
            "gausslq-gpu", "gausslq-rotated-gpu", "gausslq-spherical-gpu", \
            "gaussmle", "gaussmle-spherical", "gaussmle-gpu", \
            "gaussmle-rotated-gpu", "gaussmle-spherical-gpu", "spline-gpu", \
            "spline-mle-gpu" or "avg"}, optional
        Which 2D fitting algorithm to use. "gausslq" for least-squares
        fitting of a 2D Gaussian. "gausslq-gpu" for its GPU
        implemntation (if available). "gaussmle" for MLE 2D Gaussian
        fitting (CPU). "gaussmle-gpu" for MLE fitting of a 2D Gaussian
        on the GPU (the Poisson maximum likelihood estimator).
        "gausslq-rotated" for CPU least-squares fitting, and
        "gausslq-rotated-gpu" and "gaussmle-rotated-gpu" for GPU
        least-squares and MLE fitting, respectively, of a rotated
        elliptical Gaussian, whose fitted rotation angle (in degrees)
        is saved in the column "angle". "gausslq-spherical" and
        "gaussmle-spherical" for CPU least-squares and MLE fitting, and
        "gausslq-spherical-gpu" and "gaussmle-spherical-gpu" for their
        GPU counterparts, of a spherical (isotropic) Gaussian with a
        single width; the saved "sx" and "sy" columns are identical.
        "spline" and "spline-mle" for CPU least-squares /
        maximum-likelihood fitting of an experimentally measured
        cubic-spline PSF, and "spline-gpu" and "spline-mle-gpu" for their
        GPU counterparts. All four
        require ``spline_calibration``, and a 3D spline calibration yields
        the fitted ``z`` directly. "avg" for taking the average of each
        spot.
    eps : float or None, optional
        The convergence criterion, honored by every iterating method on
        either device (all of them except "avg"). None (the default)
        picks the value that suits the method: 0.001 for "gaussmle",
        0.01 for "gausslq" and the GPU Gaussians, and for either spline
        backend 1e-4 with the axial multi-start and 1e-2 without.
    max_it : int or None, optional
        The maximum number of iterations per spot, as ``eps``. None (the
        default) means 100 for "gaussmle", 200 for "gausslq" (MINPACK's
        own default), 20 for the GPU Gaussians and, for either spline
        backend, 100 with the axial multi-start and 20 without.
    spline_calibration : dict or None, optional
        Cubic-spline PSF calibration (see ``io.load_spline_calibration``),
        required for any "spline*" ``fitting_method`` and ignored
        otherwise. For a 3D spline calibration the resulting localizations
        contain the fitted ``z`` directly (no separate z-fitting step is
        needed). Default is None.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration (see
        ``io.load_camera_calibration`` and ``scmos.calibrate_scmos``),
        holding the maps "offset" (ADU), "variance" (ADU^2) and,
        optionally, "gain" (ADU/e-). The maps must match the full frame
        shape of ``movie``. When given, they replace the scalar
        "Baseline" (and, if a gain map is present, "Sensitivity") of
        ``camera_info``, and the per-pixel readout variance enters the
        noise model of Huang et al., Nat. Methods 10:653 (2013): every
        MLE fit and every uncertainty estimate ("lpx", "lpy", CRLB) then
        de-emphasises noisy pixels. Least-squares fits are unaffected by
        the variance term itself (the shift cancels), but their
        uncertainties do grow on noisy pixels; prefer an MLE method for
        sCMOS data. Default is None.
    multiprocess : bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
    progress_callback : callable, "console" or None, optional
        If a callable provided, it must accept one integer input (number
        of localized spots). If "console", tqdm is used to display
        progress. If None, progress is not tracked.
    abort_callback : callable or None, optional
        A callable for aborting multiprocessing in the GUI. If a
        callable provided, it must accept no input and return a boolean
        indicating whether the fitting should be aborted. Default is
        None.
    cut_progress_callback : callable or None, optional
        If a callable is provided, it is called with the cumulative
        number of spots cut so far while extracting the spots from the
        movie (before fitting). It must accept one integer input.
        Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. Returns None if
        fitting was aborted.
    new_info : dict
        New metadata.
    """
    accepted_movie_types = (io.AbstractPicassoMovie, np.memmap)
    if bitplane.IMSWRITER:
        accepted_movie_types += (
            bitplane.MovieMapper,
            bitplane.MovieMapperStack,
        )
    assert isinstance(
        movie, accepted_movie_types
    ), "movie must be a movie loaded by picasso.io.load_movie"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert isinstance(
        identifications, pd.DataFrame
    ), "identifications must be a DataFrame"
    assert isinstance(box, int) and box > 0, "box must be a positive integer"
    assert fitting_method in FIT_METHODS, (
        f"fitting_method '{fitting_method}' is not one of "
        f"{', '.join(FIT_METHODS)}"
    )
    if fitting_method.startswith("spline"):
        assert isinstance(spline_calibration, dict), (
            "spline_calibration (a spline PSF calibration dict, see "
            "io.load_spline_calibration) is required for spline fitting"
        )
    assert eps is None or (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number or None"
    assert max_it is None or (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer or None"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    if "Pixelsize" not in camera_info:
        warnings.warn(
            "Camera info in picasso.localize.fit does not contain "
            "'Pixelsize', i.e., effective camera pixel size in nm. "
            "Assuming 130."
        )
        camera_info["Pixelsize"] = 130

    # ``camera_info`` is merged verbatim into the saved YAML at the end, so an
    # array in it would be dumped element by element into every sidecar. The
    # per-pixel maps travel as ``camera_calibration`` precisely so that cannot
    # happen. Checked here rather than at the merge so the message arrives
    # before _to_photons turns it into an opaque broadcast error.
    for _key, _value in camera_info.items():
        if isinstance(_value, np.ndarray):
            raise ValueError(
                f"camera_info['{_key}'] is an array. Per-pixel camera maps "
                "belong in the camera_calibration argument, not in "
                "camera_info, which is written to the metadata file as-is."
            )
    _validate_camera_calibration(camera_calibration, movie, camera_info)
    spots, variance = get_spots(
        movie,
        identifications,
        box,
        camera_info,
        progress_callback=cut_progress_callback,
        camera_calibration=camera_calibration,
        return_variance=True,
    )
    em = camera_info["Gain"] > 1
    gauss_flags = parse_gauss_code(fitting_method)
    if gauss_flags is not None:
        if gauss_flags["use_gpu"] and callable(progress_callback):
            progress_callback(1)
        locs = _fit2d_gauss(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            tolerance=eps,
            max_iterations=max_it,
            progress_callback=(
                None if gauss_flags["use_gpu"] else progress_callback
            ),
            variance=variance,
            **gauss_flags,
        )
    elif fitting_method in ("spline-gpu", "spline-mle-gpu"):
        if callable(progress_callback):
            progress_callback(1)
        # "spline-mle-gpu" uses the Poisson maximum-likelihood estimator,
        # "spline-gpu" least squares.
        # The GPU fit itself is a single call; progress_callback then tracks
        # the per-spot CRLB / precision computation in locs_from_fits_spline.
        locs = _fit2d_spline_gpu(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            calibration=spline_calibration,
            mle=fitting_method == "spline-mle-gpu",
            progress_callback=progress_callback,
            tolerance=eps,
            max_iterations=max_it,
            variance=variance,
        )
    elif fitting_method in ("spline", "spline-mle"):
        # The CPU cubic-spline fit (picasso.fitting.splinefit). Unlike the
        # GPU path it is a per-spot loop, so progress_callback tracks the fit
        # itself and the fit can be aborted. eps / max_it override the
        # convergence schedule; None picks the one matching the multi-start.
        locs = _fit2d_spline_cpu(
            spots=spots,
            identifications=identifications,
            box=box,
            em=em,
            calibration=spline_calibration,
            mle=fitting_method == "spline-mle",
            tolerance=eps,
            max_iterations=max_it,
            multiprocess=multiprocess,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
            variance=variance,
        )
    elif fitting_method == "avg":
        locs = _fit2d_avg(
            spots,
            identifications,
            box,
            em,
            multiprocess,
            progress_callback,
            abort_callback,
            variance=variance,
        )
    # updated metadata
    localize_info = {
        "Generated by": f"Picasso: v{__version__} Fit 2D",
        "Fit method": fitting_method,
    }
    # Record the schedule the fit actually ran with, per method - each
    # backend has its own defaults, and "None" in the caller means "yours".
    if gauss_flags is not None:
        tolerance, max_iterations = gauss_schedule(
            gauss_flags["mle"], gauss_flags["use_gpu"], eps, max_it
        )
        localize_info["Convergence criterion"] = tolerance
        localize_info["Max iterations"] = max_iterations
    if fitting_method.startswith("spline"):
        localize_info["Spline calibration model"] = spline_calibration.get(
            "model"
        )
        localize_info["Spline calibration path"] = spline_calibration.get(
            "Path", "N/A"
        )
        on_gpu = fitting_method.endswith("-gpu")
        localize_info["Spline fit device"] = "GPU" if on_gpu else "CPU"
        localize_info["Spline CRLB device"] = (
            "GPU" if precision.CUDA_AVAILABLE else "CPU"
        )
        # Record what the fit actually used, not what was requested: the
        # schedule depends on whether the axial multi-start ran. Identical
        # on both devices - they share ``_run_splinefit``.
        n_z_starts = _default_n_z_starts(spline_calibration)
        _, apply_seeds = _spline_z_seeds(spline_calibration, n_z_starts)
        tolerance, max_iterations = _spline_schedule(apply_seeds, eps, max_it)
        localize_info["Convergence criterion"] = tolerance
        localize_info["Max iterations"] = max_iterations
        localize_info["Axial seeds"] = n_z_starts if apply_seeds else 1
    localize_info.update(camera_calibration_info(camera_calibration))
    new_info = localize_info | camera_info
    return locs, new_info


# TODO: remove in v0.12.0
def fit2D(
    movie: LoadedMovie,
    movie_info: list[dict] | None = None,
    camera_info: dict | None = None,
    identifications: pd.DataFrame | None = None,
    box: int | None = None,
    fitting_method: str = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame | None, dict]:
    """Deprecated alias for :func:`fit`.

    .. deprecated:: 0.11.0
        Use ``picasso.localize.fit`` instead. ``fit2D`` will be removed
        in v0.12.0, together with the ``movie_info`` and ``mle_method``
        arguments, neither of which affects the fit.

    Parameters
    ----------
    movie, camera_info, identifications, box, fitting_method, eps, max_it
        As in :func:`fit`.
    movie_info : list of dicts, optional
        Ignored, other than being asserted to be a list. Removed in v0.12.0.
    mle_method : {"sigma", "sigmaxy"}, optional
        Ignored; warns when given. Removed in v0.12.0.
    **kwargs
        Forwarded to :func:`fit`.

    Returns
    -------
    locs : pd.DataFrame or None
        As in :func:`fit`.
    info : dict
        As in :func:`fit`.
    """
    lib.deprecation_warning(
        "picasso.localize.fit2D is deprecated and will be removed in "
        "version 0.12; use picasso.localize.fit instead. Its movie_info "
        "and mle_method arguments will be removed with it - neither has "
        "any effect on the fit."
    )
    if mle_method is not None:
        lib.deprecation_warning(
            "The mle_method argument is ignored and will be removed in "
            "version 0.12."
        )
    assert isinstance(movie_info, list), "movie_info must be a list"
    return fit(
        movie=movie,
        camera_info=camera_info,
        identifications=identifications,
        box=box,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        **kwargs,
    )


# Per-method convergence schedules. Each backend's own defaults, kept here so
# the resolved values reach both the fit and the saved metadata, and so that
# rerouting a method to a new backend cannot silently change where it stops.
#: Tokens a Gaussian fit code may carry after ``gausslq``/``gaussmle``.
_GAUSS_TOKENS = frozenset({"spherical", "rotated", "gpu"})


def parse_gauss_code(fitting_method: str) -> dict | None:
    """Flags of a Gaussian fit code, or None if it is not one.

    The grammar is ``gauss{lq,mle}[-spherical|-rotated][-gpu]``.

    Parameters
    ----------
    fitting_method : str
        The fit code to parse, e.g. ``"gaussmle-spherical-gpu"``.

    Returns
    -------
    flags : dict or None
        ``{"mle", "spherical", "rotated", "use_gpu"}``, all bool. None for
        anything that is not a valid Gaussian code, so callers can use this as
        both the parser and the validator.
    """
    tokens = fitting_method.split("-")
    if tokens[0] not in ("gausslq", "gaussmle"):
        return None
    if len(set(tokens[1:])) != len(tokens[1:]):
        return None  # a repeated token, e.g. "gausslq-gpu-gpu"
    flags = {
        "mle": tokens[0] == "gaussmle",
        "spherical": False,
        "rotated": False,
        "use_gpu": False,
    }
    for token in tokens[1:]:
        if token not in _GAUSS_TOKENS:
            return None
        if token == "spherical":
            flags["spherical"] = True
        elif token == "rotated":
            flags["rotated"] = True
        else:  # "gpu"
            flags["use_gpu"] = True
    if flags["spherical"] and flags["rotated"]:
        return None
    return flags


def gauss_fit_methods() -> list[str]:
    """Every Gaussian fit code :func:`parse_gauss_code` accepts.

    Returns
    -------
    codes : list of str
        Generated from the grammar rather than listed by hand, so a code
        cannot be offered somewhere and rejected here.
    """
    codes = []
    for estimator in ("gausslq", "gaussmle"):
        for shape in ("", "-spherical", "-rotated"):
            for device in ("", "-gpu"):
                code = f"{estimator}{shape}{device}"
                if parse_gauss_code(code) is not None:
                    codes.append(code)
    return codes


#: Every ``fit`` method. Generated for the Gaussians (see
#: :func:`gauss_fit_methods`) and listed for the rest, which have no grammar.
FIT_METHODS = tuple(
    gauss_fit_methods()
    + ["spline", "spline-mle", "spline-gpu", "spline-mle-gpu", "avg"]
)


_GAUSS_SCHEDULES = {
    # (mle, use_gpu) -> (tolerance, max_iterations)
    #
    # On the CPU each estimator gets a schedule that converges it properly;
    # for least squares that is the historical MINPACK schedule, kept in
    # ``gaussfit`` as ``*_LSQ_CPU``. On the GPU both keep Gpufit's, which is
    # looser - deliberate rather than ideal, since it is what every "-gpu"
    # code has always meant and changing it would silently move existing
    # results. These are only *defaults*: pass ``eps``/``max_it``, or use the
    # Localize parameters dialog, to change them.
    (False, False): (
        gaussfit.TOLERANCE_LSQ_CPU,
        gaussfit.MAX_ITERATIONS_LSQ_CPU,
    ),
    (True, False): (1e-5, 100),
    (False, True): (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS),
    (True, True): (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS),
}


def gauss_schedule(
    mle: bool,
    use_gpu: bool,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> tuple:
    """The convergence schedule a Gaussian fit uses, explicit values winning.

    Parameters
    ----------
    mle : bool
        Whether the fit uses the maximum-likelihood estimator.
    use_gpu : bool
        Whether the fit runs on the GPU.
    tolerance : float, optional
        Relative convergence tolerance. ``None`` (the default) picks the
        default of the method, which differs by estimator and device - see
        :data:`_GAUSS_SCHEDULES`.
    max_iterations : int, optional
        Iteration cap. ``None`` as for ``tolerance``.

    Returns
    -------
    tolerance : float
    max_iterations : int
    """
    default = _GAUSS_SCHEDULES[(bool(mle), bool(use_gpu))]
    if tolerance is None:
        tolerance = default[0]
    if max_iterations is None:
        max_iterations = default[1]
    return float(tolerance), int(max_iterations)


def _gauss_model(rotated: bool, spherical: bool) -> int:
    """The :mod:`picasso.fitting.gaussfit` model of a method's flags."""
    if spherical:
        return gaussfit.SPHERICAL
    if rotated:
        return gaussfit.ROTATED
    return gaussfit.ELLIPTIC


def fit_spots_gauss(
    spots: lib.FloatArray3D,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    use_gpu: bool = False,
    return_stats: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray3D | None = None,
) -> (
    lib.FloatArray2D
    | tuple[
        lib.FloatArray2D,
        lib.FloatArray1D | None,
        lib.FloatArray1D,
        lib.FloatArray1D | None,
    ]
):
    """Fit spots with a 2D Gaussian on the CPU or the GPU.

    The one entry point for every Gaussian method Picasso offers. Both devices
    run the identical Levenberg-Marquardt algorithm
    (:mod:`picasso.fitting.gaussfit` and ``gaussfit_cuda``), so ``use_gpu``
    only affects speed - the arrangement :func:`fit_spots_spline` already uses
    for the spline models.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts.
    rotated : bool, optional
        Fit a rotated elliptical Gaussian, whose seventh parameter is the
        rotation angle in radians. Cannot be combined with ``spherical``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    spherical : bool, optional
        Fit a single shared width. The returned parameters still use the
        elliptical layout with ``sx == sy``, so the rest of the pipeline is
        unchanged.
    use_gpu : bool, optional
        Run on a CUDA GPU. Default False.
    return_stats : bool, optional
        Additionally return ``(log_likelihood, iterations, chi_square)``.
    tolerance, max_iterations : optional
        ``None`` uses the method's own schedule, see :func:`gauss_schedule`.
    progress_callback : callable, "console" or None, optional
        Reported per spot on the CPU; the GPU fit is one launch per chunk.
    variance : lib.FloatArray3D, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like ``spots`` (from ``get_spots(..., return_variance=True)``).
        Applies Huang et al.'s noise model to the maximum-likelihood
        estimator; least squares is unaffected by construction. Default is
        None.

    Returns
    -------
    parameters : lib.FloatArray2D
        ``[photons, x, y, sx, sy, bg]``, plus the rotation angle (radians) if
        ``rotated``. Positions are box-local.
    log_likelihood : lib.FloatArray1D or None
        Only if ``return_stats``. None for least squares - each estimator
        reports the goodness of fit that means something for it.
    number_iterations : lib.FloatArray1D
        Only if ``return_stats``. Iterations each spot took.
    chi_square : lib.FloatArray1D or None
        Only if ``return_stats``. The residual sum of squares at the optimum;
        None for maximum likelihood.
    """
    if rotated and spherical:
        raise ValueError("'rotated' and 'spherical' are mutually exclusive.")
    if use_gpu and not CUDA_AVAILABLE:
        raise ImportError(
            "GPU fitting was requested but no CUDA-capable GPU is available."
        )
    model = _gauss_model(rotated, spherical)
    tolerance, max_iterations = gauss_schedule(
        mle, use_gpu, tolerance, max_iterations
    )
    if mle:
        spots = _clip_for_mle(spots, variance)
    size = spots.shape[1]
    initial_parameters = seeds.initial_parameters_gauss(
        _seed_spots(spots, variance) if mle else spots,
        size,
        rotated=rotated,
        spherical=spherical,
    ).astype(np.float64)

    backend = gaussfit_cuda if use_gpu else gaussfit
    parameters, chi_squares, _states, number_iterations = backend.fit_spots(
        model,
        spots,
        initial_parameters,
        mle=mle,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        variance=variance,
    )
    parameters = parameters.astype(np.float32)
    chi_squares = chi_squares.astype(np.float32)

    if spherical:
        # The isotropic models return [amplitude, x, y, s, bg]. Expand to the
        # standard elliptical layout with sx == sy so the rest of the pipeline
        # (CRLB, column building) is unchanged.
        s = parameters[:, 3]
        expanded = np.empty((len(parameters), 6), dtype=parameters.dtype)
        expanded[:, 0] = parameters[:, 0]
        expanded[:, 1] = parameters[:, 1]
        expanded[:, 2] = parameters[:, 2]
        expanded[:, 3] = s
        expanded[:, 4] = s
        expanded[:, 5] = parameters[:, 4]
        expanded[:, 0] *= 2.0 * np.pi * s * s
        parameters = expanded
    else:
        # The models fit a peak height; convert to total photons.
        parameters[:, 0] *= 2.0 * np.pi * parameters[:, 3] * parameters[:, 4]

    if return_stats:
        # The MLE chi-square equals twice the negative Poisson
        # log-likelihood, so -0.5 * chi_square reproduces the CPU MLE
        # fit's log_likelihood (both Stirling-approximated). For least
        # squares the chi-square is the plain residual sum of squares -
        # not a likelihood, since least squares assumes no noise model -
        # and is reported as such, as this fit's goodness-of-fit metric.
        log_likelihood = -0.5 * chi_squares if mle else None
        chi_square = None if mle else chi_squares
        return parameters, log_likelihood, number_iterations, chi_square
    return parameters


def fit_spots_gauss_gpu(
    spots: lib.FloatArray3D,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    return_stats: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> (
    lib.FloatArray2D
    | tuple[
        lib.FloatArray2D,
        lib.FloatArray1D | None,
        lib.FloatArray1D,
        lib.FloatArray1D | None,
    ]
):
    """Fit spots with a 2D Gaussian on the GPU.

    Thin wrapper over :func:`fit_spots_gauss` with ``use_gpu=True``, kept
    because it is the established public name.

    Parameters
    ----------
    spots, rotated, mle, spherical, return_stats, tolerance, max_iterations
        As in :func:`fit_spots_gauss`.
    variance
        As in :func:`fit_spots_gauss`.

    Returns
    -------
    parameters : lib.FloatArray2D
        As in :func:`fit_spots_gauss`.
    log_likelihood : lib.FloatArray1D or None
        Only if ``return_stats``, as in :func:`fit_spots_gauss`.
    number_iterations : lib.FloatArray1D
        Only if ``return_stats``, as in :func:`fit_spots_gauss`.
    chi_square : lib.FloatArray1D or None
        Only if ``return_stats``, as in :func:`fit_spots_gauss`.
    """
    return fit_spots_gauss(
        spots,
        rotated=rotated,
        mle=mle,
        spherical=spherical,
        use_gpu=True,
        return_stats=return_stats,
        tolerance=tolerance,
        max_iterations=max_iterations,
        variance=variance,
    )


def locs_from_fits_gauss(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    mle: bool = False,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    spherical: bool = False,
    chi_square: lib.FloatArray1D | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame:
    """Convert the fit results from a Gaussian fit into a data frame of
    localizations.

    Backend-agnostic: ``picasso.fitting.gaussfit`` (CPU) and
    ``picasso.fitting.gaussfit_cuda`` (GPU) return the same ``theta``
    layout, so both are converted here.

    Parameters
    ----------
    identifications : pd.DataFrame
        Data frame containing the identifications of the spots,
        including frame numbers, x and y coordinates, and net gradient.
    theta : lib.FloatArray2D
        A 2D array with the optimized parameters for each spot, where
        each row corresponds to a spot and the columns are the
        parameters in the following order: [photons, x, y, sx, sy, bg]
        or, for the rotated elliptical Gaussian,
        [photons, x, y, sx, sy, bg, angle (radians)]. In the latter
        case, the resulting data frame contains the column ``angle``
        (in degrees).
    box : int
        The size of the box used for localization, which is used to
        calculate the offsets for the x and y coordinates.
    em : bool
        Whether EMCCD was used for the localization.
    mle : bool, optional
        Whether ``theta`` came from the maximum-likelihood
        estimator. If True, the localization precisions ``lpx`` / ``lpy``
        and the per-parameter uncertainties (``photons_unc``, ``bg_unc``,
        ``sx_unc``, ``sy_unc`` and, for the rotated model, ``angle_unc``)
        are the Poisson Cramer-Rao bound from the Fisher information of
        the fitted Gaussian model (:func:`precision._gauss_crlb`), matching the CPU
        MLE fit output. If False (least squares), ``lpx`` / ``lpy`` use
        the Mortensen et al. closed form and no per-parameter
        uncertainties are added. Default is False.
    log_likelihood : lib.FloatArray1D, optional
        The per-spot Poisson log-likelihood (from an MLE fit). If
        provided together with ``iterations``, the ``log_likelihood``
        and ``iterations`` columns are added, matching the CPU MLE fit
        output. Default is None.
    iterations : lib.FloatArray1D, optional
        The number of iterations taken to converge for each spot.
        Default is None.
    spherical : bool, optional
        If True, the fit was a spherical (isotropic) Gaussian, so
        ``sx == sy`` and the ellipticity is always 0. The
        ``ellipticity`` column is then omitted as it carries no
        information. Default is False.
    chi_square : lib.FloatArray1D, optional
        The per-spot residual sum of squares at the fit optimum (from a
        least-squares fit). If provided, the ``chi_square`` column is
        added. It is the least-squares counterpart of the MLE fits'
        ``log_likelihood``: a goodness-of-fit measure in photons squared,
        so it scales with the spot brightness and the box size and is
        only comparable between fits of the same box size. Default is
        None.
    variance : lib.FloatArray3D, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like the fitted spots. It enters the Cramer-Rao bound of an
        MLE fit pixel by pixel, and the Mortensen closed form of a
        least-squares fit as its mean over the box. Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    """
    box_offset = int(box / 2)
    rotated = theta.shape[1] == 7
    x = theta[:, 1] + identifications["x"] - box_offset
    y = theta[:, 2] + identifications["y"] - box_offset
    if mle:
        # Poisson Cramer-Rao bound from the Fisher information of the
        # point-sampled Gaussian model the fitters optimize. Columns of ``crlb``
        # follow ``theta``: [photons, x, y, sx, sy, bg, (angle)].
        crlb = precision._gauss_crlb(
            theta, box, em, rotated=rotated, variance=variance
        )
        with np.errstate(invalid="ignore"):
            lpx = np.sqrt(crlb[:, 1])
            lpy = np.sqrt(crlb[:, 2])
    else:
        # The closed form has no per-pixel notion, so the readout noise enters
        # as its mean over the box; see precision.localization_precision.
        readout = _mean_readout_variance(variance)
        lpx = precision.localization_precision(
            theta[:, 0],
            theta[:, 3],
            theta[:, 4],
            theta[:, 5],
            em=em,
            readout_variance=readout,
        )
        lpy = precision.localization_precision(
            theta[:, 0],
            theta[:, 4],
            theta[:, 3],
            theta[:, 5],
            em=em,
            readout_variance=readout,
        )
    columns = {
        "frame": identifications["frame"].astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "photons": theta[:, 0].astype(np.float32),
        "sx": theta[:, 3].astype(np.float32),
        "sy": theta[:, 4].astype(np.float32),
        "bg": theta[:, 5].astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
    }
    if not spherical:
        # For a spherical (isotropic) Gaussian sx == sy, so the
        # ellipticity is always 0 and carries no information.
        a = np.maximum(theta[:, 3], theta[:, 4])
        b = np.minimum(theta[:, 3], theta[:, 4])
        ellipticity = (a - b) / a
        columns["ellipticity"] = ellipticity.astype(np.float32)
    columns["net_gradient"] = identifications["net_gradient"].astype(
        np.float32
    )
    if rotated:  # rotated elliptical Gaussian
        # Normalize to [-90, 90) as the ellipse repeats every half turn.
        angle = -np.rad2deg(theta[:, 6])
        angle = np.mod(angle + 90.0, 180.0) - 90.0
        columns["angle"] = angle.astype(np.float32)
    if mle:
        with np.errstate(invalid="ignore"):
            columns["photons_unc"] = np.sqrt(crlb[:, 0]).astype(np.float32)
            columns["bg_unc"] = np.sqrt(crlb[:, 5]).astype(np.float32)
            columns["sx_unc"] = np.sqrt(crlb[:, 3]).astype(np.float32)
            columns["sy_unc"] = np.sqrt(crlb[:, 4]).astype(np.float32)
            if rotated:
                columns["angle_unc"] = np.rad2deg(np.sqrt(crlb[:, 6])).astype(
                    np.float32
                )
    if log_likelihood is not None:
        columns["log_likelihood"] = log_likelihood.astype(np.float32)
    if iterations is not None:
        columns["iterations"] = iterations.astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    if "n_id" in identifications.columns:
        # The cross-channel link index. Carried through and sorted on, as
        # the spline path does - a multichannel fit needs every channel's
        # localizations in the same order to pair them up.
        locs["n_id"] = np.asarray(identifications["n_id"]).astype(np.uint32)
        locs.sort_values(by="n_id", kind="quicksort", inplace=True)
    else:
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def locs_from_fits_gauss_gpu(*args, **kwargs) -> pd.DataFrame:
    """Convert the fit results from a Gaussian fit into a data frame of
    localizations.

    .. deprecated:: 0.11
        Renamed to :func:`locs_from_fits_gauss` and removed under this
        name in Picasso 1.0. The function never was GPU-specific: it
        converts CPU and GPU Gaussian fits alike.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :func:`locs_from_fits_gauss`.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    """
    lib.deprecation_warning(
        "picasso.localize.locs_from_fits_gauss_gpu is deprecated and will "
        "be removed in Picasso 1.0. It handles CPU and GPU fits alike and "
        "was renamed to picasso.localize.locs_from_fits_gauss."
    )
    return locs_from_fits_gauss(*args, **kwargs)


def _fit2d_gauss(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    rotated: bool = False,
    mle: bool = False,
    spherical: bool = False,
    use_gpu: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame:
    """Fit 2D Gaussians with least squares or, if ``mle``, maximum
    likelihood, on the CPU or the GPU. If ``rotated``, a rotated elliptical
    Gaussian is fitted and the resulting localizations contain the fitted
    rotation angle (in degrees) in the column ``angle``. If ``spherical``, an
    isotropic Gaussian with a single width is fitted and the resulting ``sx``
    and ``sy`` columns are identical. See ``fit`` for more details."""
    theta, log_likelihood, iterations, chi_square = fit_spots_gauss(
        spots,
        rotated=rotated,
        mle=mle,
        spherical=spherical,
        use_gpu=use_gpu,
        return_stats=True,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        variance=variance,
    )
    locs = locs_from_fits_gauss(
        identifications,
        theta,
        box,
        em,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        spherical=spherical,
        chi_square=chi_square,
        variance=variance,
    )
    return locs


# ----------------------------------------------------------------------
# Multichannel (joint) spherical Gaussian fitting
#
# Several registered channels are fitted at once with one shared position and
# width, the globLoc arrangement (Li et al., Nat. Commun. 13, 3133, 2022) the
# spline models already use. The channel registration comes from a standalone
# calibration (see ``picasso.registration``) rather than a PSF calibration, so
# this needs no measured PSF - only where each channel sits.
# ----------------------------------------------------------------------


def fit_spots_gauss_multichannel(
    spots: np.ndarray,
    residuals: np.ndarray,
    jacobians: np.ndarray,
    mle: bool = False,
    link_photons: bool = True,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    variance: np.ndarray | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple | None:
    """Jointly fit registered channels with a spherical Gaussian.

    The Gaussian counterpart of :func:`fit_spots_spline`, and arranged the same
    way: the channel-major reshape, the seeds and the convergence schedule are
    computed once *here*, above the device dispatch, so the CPU and GPU
    backends are guaranteed to see byte-identical inputs and a comparison
    between them tests the algebra rather than two translation layers.

    Parameters
    ----------
    spots : np.ndarray
        ``(n_spots, box, box, n_channels)`` photon counts, as
        :func:`get_spots_multichannel` returns them.
    residuals, jacobians : np.ndarray
        The channel geometry from the same call: ``(n_spots, n_channels, 2)``
        sub-pixel ROI offsets and ``(n_spots, n_channels, 4)`` local Jacobians.
    mle : bool, optional
        Poisson maximum likelihood instead of least squares.
    link_photons : bool, optional
        Share one amplitude and background across the channels. This assumes
        every channel collects the same number of photons - see
        :func:`fit_gauss_multichannel`.
    use_gpu : bool or None, optional
        None uses a CUDA GPU when one is available.
    tolerance, max_iterations : optional
        Convergence schedule. None uses the method's own, see
        :func:`gauss_schedule`.
    multiprocess : bool, optional
        Run the CPU kernels on a thread pool (they are ``nogil``). Ignored on
        the GPU, where one launch fits every spot.
    progress_callback : callable, "console" or None, optional
        Called with the cumulative number of spots fitted; ``"console"`` shows
        a progress bar instead.
    abort_callback : callable, optional
        Polled during the fit; returning True stops it.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        like ``spots``. None fits the plain Poisson model.
    return_stats : bool, optional
        Also return ``(log_likelihood, iterations, chi_square)``.

    Returns
    -------
    theta : np.ndarray
        ``[amplitude, x, y, sigma, bg]`` (linked) or
        ``[x, y, sigma, N_0.., bg_0..]`` (decoupled). The amplitude is a peak
        height; :func:`locs_from_fits_gauss_multichannel` converts it.
    log_likelihood, iterations, chi_square
        Only if ``return_stats``. ``log_likelihood`` is None for least squares
        and ``chi_square`` is None for maximum likelihood - each estimator
        reports the goodness of fit that means something for it.
    None
        If ``abort_callback`` asked to stop.
    """
    n_channels = spots.shape[-1] if spots.ndim == 4 else 1
    spots = precision._spline_channel_major(np.asarray(spots), n_channels)
    variance = precision._crlb_variance_channel_major(variance, n_channels)
    box = spots.shape[2]
    kind = (
        gaussfit.MULTI_KIND_SHARED
        if link_photons
        else gaussfit.MULTI_KIND_DECOUPLED
    )
    use_gpu = _spline_use_gpu(use_gpu)
    tolerance, max_iterations = gauss_schedule(
        mle, use_gpu, tolerance, max_iterations
    )
    if mle:
        spots = _clip_for_mle(spots, variance)
    initial_parameters = seeds.initial_parameters_gauss_multichannel(
        _seed_spots(spots, variance) if mle else spots,
        box,
        link_photons=link_photons,
    ).astype(np.float64)

    if use_gpu:
        result = gaussfit_cuda.fit_spots_multichannel(
            kind,
            spots,
            jacobians,
            residuals,
            initial_parameters,
            mle=mle,
            tolerance=tolerance,
            max_iterations=max_iterations,
            progress_callback=progress_callback,
            abort_callback=abort_callback,
            variance=variance,
        )
        if result is None:
            return None
    elif not multiprocess or len(spots) == 0:
        result = gaussfit.fit_spots_multichannel(
            kind,
            spots,
            jacobians,
            residuals,
            initial_parameters,
            mle=mle,
            tolerance=tolerance,
            max_iterations=max_iterations,
            progress_callback=progress_callback,
            variance=variance,
        )
    else:
        job = gaussfit.fit_spots_multichannel_async(
            kind,
            spots,
            jacobians,
            residuals,
            initial_parameters,
            mle=mle,
            tolerance=tolerance,
            max_iterations=max_iterations,
            variance=variance,
        )
        n_spots = len(spots)
        while not job.finished():
            job.raise_errors()
            if abort_callback is not None and abort_callback():
                job.stop()
                return None
            if callable(progress_callback):
                progress_callback(min(job.current[0], n_spots))
            time.sleep(0.1)
        job.raise_errors()
        if callable(progress_callback):
            progress_callback(n_spots)
        result = job.results()

    theta, chi_squares, _states, iterations = result
    theta = theta.astype(np.float32)
    if return_stats:
        # As for every other Picasso fit: the maximum-likelihood chi-square is
        # twice the negative Poisson log-likelihood, the least-squares one the
        # residual sum of squares.
        log_likelihood = (
            (-0.5 * chi_squares).astype(np.float32) if mle else None
        )
        chi_square = None if mle else chi_squares.astype(np.float32)
        return theta, log_likelihood, iterations, chi_square
    return theta


def locs_from_fits_gauss_multichannel(
    identifications: pd.DataFrame,
    theta: np.ndarray,
    box: int,
    em: bool,
    jacobians: np.ndarray,
    residuals: np.ndarray,
    link_photons: bool = True,
    mle: bool = False,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    chi_square: lib.FloatArray1D | None = None,
    variance: np.ndarray | None = None,
) -> pd.DataFrame:
    """Localizations from a multichannel spherical Gaussian fit.

    The positions are in the **reference channel's** coordinates, as for the
    multichannel spline fit: the fit solves for one shared position and the
    reference channel's box sits on the detection itself.

    Parameters
    ----------
    identifications : pd.DataFrame
        The detections that were fitted, in the reference channel, with
        ``frame``, ``x``, ``y`` and ``net_gradient``.
    theta : np.ndarray
        Fitted parameters from :func:`fit_spots_gauss_multichannel`, box-local
        and with the amplitude as a peak height.
    box : int
        Box side length (camera pixels), used to place the box-local positions
        back into the frame.
    em : bool
        Whether an EMCCD was used; its excess noise doubles every reported
        variance.
    jacobians, residuals : np.ndarray
        The channel geometry the fit used: ``(n_locs, n_channels, 4)`` local
        Jacobians and ``(n_locs, n_channels, 2)`` sub-pixel ROI offsets. Passed
        on to the Cramer-Rao bound, so the reported precision describes the
        geometry that was actually fitted.
    link_photons : bool, optional
        Which model produced ``theta``. Default False.
    mle : bool, optional
        Whether ``theta`` came from the maximum-likelihood estimator. Selects
        the Poisson Cramer-Rao bound over the least-squares sandwich for the
        reported uncertainties. Default False.
    log_likelihood, iterations, chi_square : optional
        Per-spot fit statistics; each is written to a column of the same name
        when given.
    variance : np.ndarray, optional
        Channel-major per-pixel sCMOS readout variance, for the uncertainties.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, sorted by ``frame`` (or by ``n_id`` when the
        identifications carry one). ``photons`` and ``bg`` are the totals
        **across all channels** in both models, so the two are directly
        comparable with each other and with the spline path: the decoupled
        model sums its per-channel counts, and the linked model - which fits
        one amplitude shared *literally* by every channel - multiplies by the
        channel count. The decoupled model additionally reports
        ``photons_ch{c}``, ``bg_ch{c}`` and ``rel_photons_ch{c}`` per channel.
        Widths appear as ``sx`` and ``sy``, both the single fitted sigma; no
        ``ellipticity`` column is written, since a spherical fit constrains it
        to zero and it would carry no information.
    """
    n_channels = jacobians.shape[1]
    box_offset = int(box / 2)
    theta = np.asarray(theta, dtype=np.float64)
    if link_photons:
        x_shift, y_shift, sigma = theta[:, 1], theta[:, 2], theta[:, 3]
        # peak height -> photons, as the single-channel Gaussian fit does
        photons_per_channel = theta[:, 0] * 2.0 * np.pi * sigma * sigma
        photons = n_channels * photons_per_channel
        bg = theta[:, 4]
        crlb_theta = np.column_stack(
            [x_shift, y_shift, sigma, photons_per_channel, bg]
        )
    else:
        x_shift, y_shift, sigma = theta[:, 0], theta[:, 1], theta[:, 2]
        scale = 2.0 * np.pi * sigma * sigma
        photons_ch = theta[:, 3 : 3 + n_channels] * scale[:, None]
        bg_ch = theta[:, 3 + n_channels :]
        photons = photons_ch.sum(axis=1)
        bg = bg_ch.sum(axis=1)
        crlb_theta = np.column_stack(
            [x_shift, y_shift, sigma, photons_ch, bg_ch]
        )

    crlb = precision._gauss_crlb_multichannel(
        crlb_theta,
        box,
        jacobians,
        residuals,
        mle=mle,
        em=em,
        link_photons=link_photons,
        variance=variance,
    )
    # Both models put the shared parameters first and the CRLB comes back in
    # the order it was given, so x, y and sigma are always columns 0, 1, 2.
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(crlb[:, 0])
        lpy = np.sqrt(crlb[:, 1])

    x = x_shift + identifications["x"] - box_offset
    y = y_shift + identifications["y"] - box_offset
    columns = {
        "frame": identifications["frame"].astype(np.uint32),
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "photons": photons.astype(np.float32),
        "sx": sigma.astype(np.float32),
        "sy": sigma.astype(np.float32),
        "bg": bg.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "net_gradient": identifications["net_gradient"].astype(np.float32),
    }
    if not link_photons:
        with np.errstate(invalid="ignore", divide="ignore"):
            total = photons_ch.sum(axis=1)
            relative = np.where(
                total[:, None] > 0, photons_ch / total[:, None], np.nan
            )
        for c in range(n_channels):
            columns[f"photons_ch{c}"] = photons_ch[:, c].astype(np.float32)
            columns[f"bg_ch{c}"] = bg_ch[:, c].astype(np.float32)
            columns[f"rel_photons_ch{c}"] = relative[:, c].astype(np.float32)
    with np.errstate(invalid="ignore"):
        if link_photons:
            # ``photons`` is n_channels times the fitted per-channel count, so
            # its uncertainty scales with it.
            columns["photons_unc"] = (n_channels * np.sqrt(crlb[:, 3])).astype(
                np.float32
            )
            columns["bg_unc"] = np.sqrt(crlb[:, 4]).astype(np.float32)
        else:
            photon_var = crlb[:, 3 : 3 + n_channels]
            bg_var = crlb[:, 3 + n_channels :]
            # independent per-channel estimates, so their variances add
            columns["photons_unc"] = np.sqrt(
                np.nansum(photon_var, axis=1)
            ).astype(np.float32)
            columns["bg_unc"] = np.sqrt(np.nansum(bg_var, axis=1)).astype(
                np.float32
            )
        sigma_unc = np.sqrt(crlb[:, 2]).astype(np.float32)
    columns["sx_unc"] = sigma_unc
    columns["sy_unc"] = sigma_unc
    if log_likelihood is not None:
        columns["log_likelihood"] = np.asarray(log_likelihood).astype(
            np.float32
        )
    if iterations is not None:
        columns["iterations"] = np.asarray(iterations).astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    if "n_id" in identifications.columns:
        locs["n_id"] = np.asarray(identifications["n_id"]).astype(np.uint32)
        locs.sort_values(by="n_id", kind="quicksort", inplace=True)
    else:
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def fit_gauss_multichannel(
    movies: list,
    camera_infos: list[dict],
    identifications: pd.DataFrame,
    box: int,
    channel_registration: dict,
    mle: bool = False,
    link_photons: bool = False,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    abort_callback: Callable[[], bool] | None = None,
    camera_calibrations: list[dict | None] | None = None,
) -> pd.DataFrame | None:
    """Fit a spherical Gaussian jointly across several registered channels.

    Global fit in the sense of globLoc (Li et al., Nat. Commun. 13, 3133,
    2022): every channel contributes to one fit with a linked x, y and width.
    The Gaussian counterpart of :func:`fit_spline_multichannel`, needing only a
    channel registration rather than a measured PSF.

    Parameters
    ----------
    movies : list
        One movie per channel; ``movies[0]`` is the reference channel and the
        order must match the registration's channels.
    camera_infos : list of dict
        One camera-info dict per channel.
    identifications : pd.DataFrame
        Detections in the reference channel.
    box : int
        Box side length (camera pixels).
    channel_registration : dict
        A calibration carrying ``channel_transforms``, from
        :mod:`picasso.registration`.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator. Default False.
    link_photons : bool, optional
        Share one photon count and background across the channels. Default
        **False**, unlike the spline fit: this model has no per-channel
        brightness scale (a spline calibration carries one in its coefficient
        table), so linking assumes every channel collects the *same* number of
        photons - true for equally split, redundant channels and wrong for an
        uneven beam splitter. The decoupled default fits each channel its own
        photon count and background, reported as ``photons_ch{c}`` /
        ``bg_ch{c}`` / ``rel_photons_ch{c}``, and shares only x, y and the
        width.
    use_gpu : bool or None, optional
        Fit on the GPU, the CPU, or (None, the default) whichever is available.
    tolerance, max_iterations : optional
        Convergence schedule. None uses the method's own, see
        :func:`gauss_schedule`.
    multiprocess : bool, optional
        Run the CPU kernels on a thread pool (they are ``nogil``). Ignored on
        the GPU. Default True.
    progress_callback : callable, optional
        Called with the cumulative number of spots processed, during both the
        spot extraction and the fit.
    abort_callback : callable, optional
        Polled during the fit; returning True stops it and returns None.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS calibration per channel, or None. Individual
        entries may be None when only some channels sit on a characterized
        camera.

    Returns
    -------
    locs : pd.DataFrame or None
        Localizations in the reference channel's coordinates - see
        :func:`locs_from_fits_gauss_multichannel` for the columns - or None if
        ``abort_callback`` asked to stop.

    Raises
    ------
    ValueError
        If the registration carries no ``channel_transforms``, if the number of
        movies does not match it, or if the photon-decoupled model is asked for
        outside the 2 to 6 channel range it supports.
    """
    transforms = channel_registration.get("channel_transforms")
    if not transforms:
        raise ValueError(
            "fit_gauss_multichannel needs a calibration with "
            "'channel_transforms' (see picasso.registration)."
        )
    if len(movies) != len(transforms):
        raise ValueError(
            f"Got {len(movies)} channels but the registration has "
            f"{len(transforms)} channel transforms."
        )
    if not link_photons:
        n_channels = len(transforms)
        if not (2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS):
            raise ValueError(
                "The photon-decoupled multichannel Gaussian model supports 2 "
                f"to {precision._LINK_XYZ_MAX_CHANNELS} channels, got "
                f"{n_channels}."
            )
    identifications = multichannel_inbounds_ids(
        identifications, box, movies, transforms
    )
    spots, residuals, variance, jacobians = get_spots_multichannel(
        movies,
        identifications,
        box,
        camera_infos,
        transforms,
        progress_callback=progress_callback,
        return_residuals=True,
        camera_calibrations=camera_calibrations,
        return_variance=True,
        return_jacobians=True,
    )
    result = fit_spots_gauss_multichannel(
        spots,
        residuals,
        jacobians,
        mle=mle,
        link_photons=link_photons,
        use_gpu=use_gpu,
        tolerance=tolerance,
        max_iterations=max_iterations,
        multiprocess=multiprocess,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        variance=variance,
        return_stats=True,
    )
    if result is None:
        return None
    theta, log_likelihood, iterations, chi_square = result
    em = camera_infos[0].get("Gain", 1) > 1
    return locs_from_fits_gauss_multichannel(
        identifications,
        theta,
        box,
        em,
        jacobians,
        residuals,
        link_photons=link_photons,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        chi_square=chi_square,
        variance=precision._crlb_variance_channel_major(
            variance, len(transforms)
        ),
    )


def fit_gauss_split_fov(
    movie,
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    channel_registration: dict,
    regions: list | None = None,
    mle: bool = False,
    link_photons: bool = False,
    confine_to_reference: bool = True,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    abort_callback: Callable[[], bool] | None = None,
    camera_calibration: dict | None = None,
) -> pd.DataFrame | None:
    """Fit a spherical Gaussian jointly across the regions of one movie.

    The split-FOV counterpart of :func:`fit_gauss_multichannel`, for channels
    imaged side by side on a single sensor, and the Gaussian counterpart of
    :func:`fit_spline_split_fov`.

    The registration stores the *inter-channel* mapping region-locally (see
    :func:`decompose_region_transforms`), independent of where the channels sit
    on the chip, so the same file applies to data whose split sits elsewhere -
    the regions in use are re-drawn at fit time. This function repeats the one
    ``movie`` / ``camera_info`` once per channel and delegates to the standard
    multichannel fitter.

    Parameters
    ----------
    movie, camera_info
        The single loaded movie and its camera info dict.
    identifications : pd.DataFrame
        Detections. With ``confine_to_reference`` (the default) they are
        filtered to the reference region, so each molecule yields one spot that
        is mapped into the other regions via the transforms.
    box : int
        Box side length (camera pixels).
    channel_registration : dict
        A split-FOV registration, from :mod:`picasso.registration`.
    regions : list, optional
        The channel ROIs *for this data*, one ``[[y_min, x_min], [y_max,
        x_max]]`` per channel, reference first. When given, the absolute
        channel transforms are rebuilt at these positions from the stored
        region-local ones. None uses the registration's own regions.
    confine_to_reference : bool, optional
        Restrict the detections to the reference region first. Default True.
    mle, link_photons, use_gpu, tolerance, max_iterations, multiprocess, \
    progress_callback, abort_callback
        As :func:`fit_gauss_multichannel`.
    camera_calibration : dict, optional
        One per-pixel sCMOS calibration for the single camera. Split-FOV is one
        sensor and the maps are indexed by absolute frame coordinates, so the
        same full-frame calibration serves every region.

    Returns
    -------
    locs : pd.DataFrame or None
        Localizations in the reference region's coordinates, or None if
        ``abort_callback`` asked to stop.
    """
    fit_regions, reference, transforms = split_fov_fit_geometry(
        channel_registration, regions
    )
    n_channels = len(fit_regions)

    ids = identifications
    if confine_to_reference:
        ids = confine_to_region(ids, fit_regions[reference])

    # the fitter reads channel_transforms off the registration; hand it the
    # transforms placed at the regions actually in use
    channel_registration = dict(channel_registration)
    channel_registration["channel_transforms"] = list(transforms)

    return fit_gauss_multichannel(
        [movie] * n_channels,
        [camera_info] * n_channels,
        ids,
        box,
        channel_registration,
        mle=mle,
        link_photons=link_photons,
        use_gpu=use_gpu,
        tolerance=tolerance,
        max_iterations=max_iterations,
        multiprocess=multiprocess,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        # one physical camera, so every region reads the same maps; _cut_map
        # indexes them with absolute frame coordinates, which is what makes one
        # full-frame calibration serve all regions
        camera_calibrations=(
            None
            if camera_calibration is None
            else [camera_calibration] * n_channels
        ),
    )


# ----------------------------------------------------------------------
# Cubic-spline PSF fitting
#
# The spline models fit an experimentally measured PSF (a cubic-spline model
# built from a bead z-stack, see ``spline.calibrate_spline``). Unlike the
# Gaussian models they
# need a coefficient table, which lives inside the calibration dict (see
# ``io.load_spline_calibration``) and is handed to the kernels by
# ``precision._spline_coeff_reshaped``.
# ----------------------------------------------------------------------


def _as_link_xyz_calibration(calibration: dict) -> dict:
    """Shallow copy of a ``spline-3d-multichannel`` calibration
    re-tagged for the photon-decoupled (link-XYZ) fit. Validates the
    supported channel range."""

    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "Photon-decoupled (link-XYZ) fitting requires a "
            "'spline-3d-multichannel' calibration."
        )
    n_channels = precision._spline_n_channels(calibration)
    if not 2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS:
        raise ValueError(
            "Photon decoupling (link-XYZ) supports 2 to "
            f"{precision._LINK_XYZ_MAX_CHANNELS} channels; this calibration has "
            f"{n_channels}. The limit is the per-thread device memory the "
            "fit kernel needs, which grows as the square of the parameter "
            "count. Keep photons linked - the shared-amplitude model works "
            "for any number of channels."
        )
    cal = dict(calibration)
    cal["model"] = precision._LINK_XYZ_MODEL
    return cal


def crop_spline_calibration(calibration: dict, box: int) -> dict:
    """Adapt a spline calibration to a smaller lateral fit box.

    This derives a smaller-box calibration by cropping the coefficient
    interval grid to the **central** ``box x box`` lateral region -
    centered on the PSF, so a spot centered in its ROI still starts
    (and converges) at ``x_shift = y_shift = 0`` and the reconstructed
    x/y carry no global shift. The axial (z) grid is untouched.

    Parameters
    ----------
    calibration : dict
        A spline PSF calibration, of any of the supported models.
    box : int
        The lateral fit box (camera pixels). Must be a positive integer no
        larger than the calibration's own box. A smaller box of the opposite
        parity is allowed - the crop is then off-center by at most half a
        pixel, a harmless constant shift of all localizations.

    Returns
    -------
    cropped : dict
        A copy of the calibration with ``coefficients``, ``n_intervals``,
        ``n_data`` and ``box`` adapted. The calibration itself (not a copy) if
        ``box`` already equals its box.

    Raises
    ------
    ValueError
        If ``box`` is out of range, or the calibration's model is unknown.
    """
    model = calibration["model"]
    n_data = list(calibration["n_data"])
    cal_box = int(n_data[0])
    box = int(box)
    if box == cal_box:
        return calibration
    if box <= 0 or box > cal_box:
        raise ValueError(
            f"Fit box ({box}) must be a positive integer no larger than the "
            f"spline calibration's box ({cal_box})."
        )
    off = (cal_box - box) // 2  # centered offset (floored for odd size diffs)
    ni = box - 1  # lateral intervals after cropping
    lat = slice(off, off + ni)
    coeff = np.ascontiguousarray(calibration["coefficients"], dtype=np.float32)

    if model == "spline-2d":
        _, nix, niy = coeff.shape
        phys = coeff.ravel(order="C").reshape(niy, nix, 4, 4)
        phys_c = np.ascontiguousarray(phys[lat, lat, :, :])
        new_coeff = phys_c.ravel(order="C").reshape(16, ni, ni)
        new_n_intervals = [ni, ni]
        new_n_data = [box, box]
    elif model == "spline-3d":
        _, nix, niy, niz = coeff.shape
        phys = coeff.ravel(order="C").reshape(niz, niy, nix, 4, 4, 4)
        phys_c = np.ascontiguousarray(phys[:, lat, lat, :, :, :])
        new_coeff = phys_c.ravel(order="C").reshape(64, ni, ni, niz)
        new_n_intervals = [ni, ni, int(niz)]
        new_n_data = [box, box, int(n_data[2])]
    elif model in ("spline-3d-multichannel", precision._LINK_XYZ_MODEL):
        _, nix, niy, niz, n_channels = coeff.shape
        new_coeff = np.empty((64, ni, ni, niz, n_channels), dtype=np.float32)
        for c in range(n_channels):
            sub = np.ascontiguousarray(coeff[..., c])
            phys = sub.ravel(order="C").reshape(niz, niy, nix, 4, 4, 4)
            phys_c = np.ascontiguousarray(phys[:, lat, lat, :, :, :])
            new_coeff[..., c] = phys_c.ravel(order="C").reshape(
                64, ni, ni, niz
            )
        new_n_intervals = [ni, ni, int(niz)]
        new_n_data = [box, box, int(n_data[2])]
    else:
        raise ValueError(f"Unknown spline model '{model}'.")

    cropped = dict(calibration)
    cropped["coefficients"] = np.ascontiguousarray(new_coeff, dtype=np.float32)
    cropped["n_intervals"] = new_n_intervals
    cropped["n_data"] = new_n_data
    cropped["box"] = box
    return cropped


# ----------------------------------------------------------------------
# CPU cubic-spline fitting
#
# The numerical core lives in ``picasso.fitting.splinefit``, a numba port of
# Gpufit's
# Levenberg-Marquardt driver and its spline models). What follows is the
# translation layer: calibration dict in, plain arrays out, plus the
# device-agnostic entry points the multichannel fitters call so that a single
# ``use_gpu`` flag selects the backend.
# ----------------------------------------------------------------------


def _spline_kind(model: str) -> int:
    """``picasso.fitting.splinefit`` model kind for a spline calibration
    ``model``."""
    if model == "spline-2d":
        return splinefit.KIND_2D
    if model in ("spline-3d", "spline-3d-multichannel"):
        return splinefit.KIND_3D
    if model == precision._LINK_XYZ_MODEL:
        return splinefit.KIND_LINK_XYZ
    raise ValueError(
        f"Unknown spline calibration model '{model}'. Expected one of "
        "'spline-2d', 'spline-3d', 'spline-3d-multichannel', "
        f"'{precision._LINK_XYZ_MODEL}'."
    )


def _spline_z_seeds(calibration: dict, n_z_starts: int) -> tuple:
    """Axial seeds for the multi-start, as ``(z_seeds, apply_seeds)``.

    The seeds span the calibration z-stack (``z_shift = -z_plane``, so
    ``[-(n_z - 1), 0]``); both devices get the same grid, so they explore
    the same axial minima."""
    n_seeds = max(1, int(n_z_starts))
    if n_seeds == 1 or calibration["model"] == "spline-2d":
        return np.zeros(1), False
    n_z = int(calibration["n_data"][2])
    return np.linspace(-(n_z - 1), 0.0, n_seeds), True


def _spline_schedule(
    apply_seeds: bool,
    tolerance: float | None,
    max_iterations: int | None,
) -> tuple:
    """Resolve a spline fit's convergence schedule. Same on either device.

    A multi-start has to rank its seeds on the chi-square and therefore needs a
    much tighter stop than a single start. ``None`` picks whichever applies;
    explicit values always win."""
    return splinefit.resolve_schedule(apply_seeds, tolerance, max_iterations)


def _run_splinefit(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    use_gpu: bool = False,
    variance: lib.FloatArray3D | None = None,
) -> tuple | None:
    """Low-level spline fit: unpack the calibration and run the kernels.

    Returns ``(theta, chi_squares, states, iterations)``, or None if
    ``abort_callback`` asked to stop. The axial multi-start runs inside the
    per-spot kernel on both devices, so every seed is tried while that spot's
    data is still in cache and progress is reported once per spot rather than
    once per pass.

    ``use_gpu`` selects :mod:`picasso.fitting.splinefit_cuda` over
    :mod:`picasso.fitting.splinefit`. The two backends are driven from *this*
    function
    rather than from separate translation layers deliberately: everything above
    the dispatch - the box crop, the channel-major reshape, the coefficient
    view, the channel Jacobians, the ROI residuals, the initial parameters
    and the
    schedule - is computed once, so both devices are guaranteed to see
    byte-identical inputs. That is what makes a CPU/GPU comparison meaningful
    rather than a test of two translation layers agreeing.

    ``multiprocess`` keeps ``fit``'s argument name, but as for ``gaussmle``
    it selects a **thread** pool: the CPU kernels are ``nogil``, so the workers
    run concurrently while sharing the spots and the coefficient table rather
    than pickling a copy of each into a subprocess. False runs the fit serially
    in the calling thread, which is what the tests use for reproducibility. It
    is ignored on the GPU, where one launch fits every spot.
    """
    box = spots.shape[1]
    # Fit a smaller-than-calibration box against a centered crop, exactly as
    # locs_from_fits_spline crops identically itself, so its CRLB matches
    # the fit geometry.
    calibration = crop_spline_calibration(calibration, box)
    model = calibration["model"]
    kind = _spline_kind(model)
    n_channels = precision._spline_n_channels(calibration)
    if mle:
        # A Poisson likelihood is undefined for negative counts. Same clip as
        # on the GPU; see :func:`_clip_for_mle`.
        spots = _clip_for_mle(spots, variance)
    fit_data = precision._spline_channel_major(spots, n_channels)
    # The variance rides through the same reshape, so it stays aligned with
    # the spots pixel for pixel on both devices.
    fit_variance = (
        None
        if variance is None
        else precision._spline_channel_major(variance, n_channels)
    )
    initial = np.ascontiguousarray(
        seeds.initial_parameters_spline(
            _seed_spots(spots, variance) if mle else spots, calibration
        ),
        dtype=np.float64,
    )
    coefficients = precision._spline_coeff_reshaped(calibration)
    channel_jacobians = precision._spline_channel_jacobians(
        jacobians, len(spots), n_channels, calibration
    )
    roi_residuals = precision._spline_crlb_residuals(
        residuals, len(spots), n_channels
    )
    if n_z_starts is None:
        n_z_starts = _default_n_z_starts(calibration)
    z_seeds, apply_seeds = _spline_z_seeds(calibration, n_z_starts)
    tolerance, max_iterations = _spline_schedule(
        apply_seeds, tolerance, max_iterations
    )

    args = (
        kind,
        fit_data,
        coefficients,
        channel_jacobians,
        roi_residuals,
        initial,
        z_seeds,
        apply_seeds,
    )
    kwargs = {
        "mle": mle,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "variance": fit_variance,
    }
    aborted = callable(abort_callback) and abort_callback()
    if aborted:
        return None
    if use_gpu:
        stopped_early = False

        def _abort() -> bool:
            nonlocal stopped_early
            if abort_callback():
                stopped_early = True
                return True
            return False

        result = splinefit_cuda.fit_spots(
            *args,
            progress_callback=progress_callback,
            abort_callback=_abort if callable(abort_callback) else None,
            **kwargs,
        )
        return None if stopped_early else result
    if not multiprocess or len(spots) == 0:
        return splinefit.fit_spots(
            *args, progress_callback=progress_callback, **kwargs
        )

    n_spots = len(spots)
    fit = splinefit.fit_spots_async(*args, **kwargs)
    use_tqdm = progress_callback == "console"
    iter_range = (
        tqdm(total=n_spots, desc="Fitting", unit="spot") if use_tqdm else None
    )
    last = 0
    while fit.current[0] < n_spots:
        if callable(abort_callback) and abort_callback():
            fit.stop()
            aborted = True
            break
        if use_tqdm:
            iter_range.update(fit.current[0] - last)
            last = fit.current[0]
        elif callable(progress_callback):
            progress_callback(fit.current[0])
        # The workers write into preallocated arrays and nothing ever collects
        # their futures, so a worker that died would leave the counter frozen
        # and this loop spinning forever. Surface the error instead.
        fit.raise_errors()
        if fit.finished() and fit.current[0] < n_spots:
            raise RuntimeError(
                "The spline fitting workers stopped after "
                f"{fit.current[0]} of {n_spots} spots."
            )
        time.sleep(0.2)
    while not fit.finished():
        # A spot is claimed before it is fitted, so the last few may still be
        # in flight once the counter reaches n_spots. Aborted runs wait too, so
        # no thread is left writing into the arrays after this returns.
        fit.raise_errors()
        time.sleep(0.05)
    fit.raise_errors()
    if use_tqdm:
        if not aborted:
            iter_range.update(n_spots - last)
        iter_range.close()
    if aborted:
        return None
    if callable(progress_callback):
        # Report completion explicitly: a fit short enough to finish before the
        # first poll would otherwise never call back at all.
        progress_callback(n_spots)
    return fit.results()


def fit_spots_splinefit(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    return_stats: bool = False,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    use_gpu: bool = False,
    variance: lib.FloatArray3D | None = None,
) -> np.ndarray | tuple | None:
    """Fit multiple spots with a cubic-spline PSF model using the numba kernels.

    Runs on the CPU by default and on the GPU with ``use_gpu``; the two are the
    same algorithm, so the choice only affects speed. Same arguments, parameter
    conventions and return shape as :func:`fit_spots_spline_gpu`, so all
    three are interchangeable (see :func:`fit_spots_spline`, which picks between
    them). Every spline model is supported: ``spline-2d``, ``spline-3d``,
    ``spline-3d-multichannel`` and the photon-decoupled
    ``spline-3d-multichannel-link-xyz``.

    Parameters
    ----------
    spots : lib.FloatArray3D
        ``(n_spots, box, box)`` photon counts, or
        ``(n_spots, n_channels, box, box)`` for the multichannel models.
    calibration : dict
        A spline PSF calibration. Cropped to the spots' box if needed (see
        :func:`crop_spline_calibration`).
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
        Default False.
    n_z_starts : int, optional
        Number of axial seeds of the multi-start. None (the default) uses
        ``_default_n_z_starts``, i.e. the calibration's z depth; a 2D model
        never runs a multi-start.
    return_stats : bool, optional
        Additionally return ``(log_likelihood, iterations, chi_square)``.
        Default False.
    residuals : np.ndarray, optional
        ``(n_spots, n_channels, 2)`` sub-pixel ROI offsets of the multichannel
        models (see :func:`channel_roi_geometry`). None (the default) means
        zeros, which is what a single-channel fit needs.
    jacobians : np.ndarray, optional
        ``(n_spots, n_channels, 4)`` local Jacobians of the channel transforms
        (see :func:`channel_roi_geometry`). None (the default) means the
        identity.
    tolerance, max_iterations : optional
        Convergence schedule. None (the default) uses the one that fit would
        use by default, see ``picasso.fitting.splinefit.convergence_schedule``.
    multiprocess : bool, optional
        Keeps ``fit``'s argument name, but selects a **thread** pool: the CPU
        kernels are ``nogil``. False fits serially in the calling thread.
        Ignored on the GPU. Default True.
    progress_callback : callable, "console" or None, optional
        ``"console"`` shows a tqdm bar; a callable is invoked with the
        cumulative number of spots fitted - per spot on the CPU, per chunk on
        the GPU.
    abort_callback : callable or None, optional
        Polled while the fit runs; returning True stops it.
    use_gpu : bool, optional
        Run on a CUDA GPU. Default False.
    variance : lib.FloatArray3D, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like ``spots``. Default None.

    Returns
    -------
    theta : np.ndarray or None
        ``(n_spots, n_params)`` fitted parameters, in the model's own order
        (see ``picasso.fitting.seeds.initial_parameters_spline``). None if
        ``abort_callback`` asked to stop.
    log_likelihood : np.ndarray or None
        Only if ``return_stats``. None for least squares - each estimator
        reports the goodness of fit that means something for it.
    iterations : np.ndarray
        Only if ``return_stats``. Iterations the winning seed took.
    chi_square : np.ndarray or None
        Only if ``return_stats``. The residual sum of squares at the optimum;
        None for maximum likelihood.
    """
    result = _run_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        residuals=residuals,
        jacobians=jacobians,
        tolerance=tolerance,
        max_iterations=max_iterations,
        multiprocess=multiprocess,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        use_gpu=use_gpu,
        variance=variance,
    )
    if result is None:
        return None
    theta, chi_squares, _states, iterations = result
    theta = theta.astype(np.float32)
    if return_stats:
        # As in fit_spots_spline_gpu: the maximum-likelihood chi-square is
        # twice the negative Poisson log-likelihood, the least-squares one is
        # the residual sum of squares.
        log_likelihood = (
            (-0.5 * chi_squares).astype(np.float32) if mle else None
        )
        chi_square = None if mle else chi_squares.astype(np.float32)
        return theta, log_likelihood, iterations, chi_square
    return theta


def _fit_splinefit_multistart(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int = 1,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    use_gpu: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> tuple:
    """Axial multi-start with the numba kernels, on either device.

    Returns ``(parameters, chi_squares, converged, n_iterations)``.
    :func:`fit_spline_multichannel_ratiometric` uses this form because it ranks
    photon-ratio hypotheses on the chi-square and needs to know which fits
    converged.

    There is no separate seed loop here: both kernels run the multi-start
    per spot internally and return the winning seed, so this only has to
    translate the fit state into the ``converged`` mask."""
    theta, chi_squares, states, iterations = _run_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        residuals=residuals,
        jacobians=jacobians,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_gpu=use_gpu,
        variance=variance,
    )
    finite = np.isfinite(theta).all(axis=1) & np.isfinite(chi_squares)
    converged = finite & (
        (states == splinefit.FIT_STATE_CONVERGED) if mle else True
    )
    return theta.astype(np.float32), chi_squares, converged, iterations


def _spline_use_gpu(use_gpu: bool | None) -> bool:
    """Resolve a ``use_gpu`` flag: None means "whatever is available".

    Raises if the GPU was explicitly asked for but is unusable, so an explicit
    request never silently becomes a (much slower) CPU fit."""
    if use_gpu is None:
        return CUDA_AVAILABLE
    if use_gpu and not CUDA_AVAILABLE:
        raise ImportError(
            "GPU spline fitting was requested but no CUDA-capable GPU is "
            "available. Pass use_gpu=False to fit the spline PSF on the CPU "
            "instead."
        )
    return bool(use_gpu)


def fit_spots_spline(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    return_stats: bool = False,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray3D | None = None,
) -> np.ndarray | tuple | None:
    """Fit spots with a cubic-spline PSF model on the available device.

    A thin wrapper over :func:`fit_spots_splinefit` that resolves the device.
    Both devices run the same algorithm, so the choice only affects speed.

    Parameters
    ----------
    spots, calibration, mle, n_z_starts, return_stats, residuals, jacobians
        As in :func:`fit_spots_splinefit`.
    use_gpu : bool, optional
        None (the default) uses the GPU when one is available; True raises if
        none is.
    tolerance, max_iterations, progress_callback, variance
        As in :func:`fit_spots_splinefit`.

    Returns
    -------
    theta : np.ndarray or None
        As in :func:`fit_spots_splinefit`.
    log_likelihood : np.ndarray or None
        Only if ``return_stats``, as in :func:`fit_spots_splinefit`.
    iterations : np.ndarray
        Only if ``return_stats``, as in :func:`fit_spots_splinefit`.
    chi_square : np.ndarray or None
        Only if ``return_stats``, as in :func:`fit_spots_splinefit`.
    """
    return fit_spots_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        return_stats=return_stats,
        residuals=residuals,
        jacobians=jacobians,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        use_gpu=_spline_use_gpu(use_gpu),
        variance=variance,
    )


def _fit_spline_multistart(
    spots: lib.FloatArray3D,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int = 1,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> tuple:
    """Axial multi-start on whichever device is available.

    See :func:`_fit_splinefit_multistart`, which does the work on either
    device; this only resolves which one."""
    return _fit_splinefit_multistart(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        residuals=residuals,
        jacobians=jacobians,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_gpu=_spline_use_gpu(use_gpu),
    )


def _locs_from_fits_spline_link_xyz(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    calibration: dict,
    mle: bool = False,
    em: bool = False,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    chi_square: lib.FloatArray1D | None = None,
    variance: lib.FloatArray4D | None = None,
) -> pd.DataFrame:
    """Localizations from a photon-decoupled (link-XYZ) multichannel spline fit.

    ``theta`` columns are ``[x_shift, y_shift, z_shift, N_0..N_{c-1},
    bg_0..bg_{c-1}]``. Emits the shared ``x, y, z`` plus per-channel photon and
    background columns ``photons_ch{c}`` / ``bg_ch{c}``, their totals in
    ``photons`` / ``bg``, and ``rel_photons_ch{c}`` = that channel's share of
    the total photons (the continuous ratiometric readout that the free photon
    ratio provides; the shares sum to 1). ``calibration`` is assumed already
    cropped to ``box``."""
    n_channels = precision._spline_n_channels(calibration)
    variance = precision._crlb_variance_channel_major(variance, n_channels)
    oversampling = float(calibration.get("oversampling", 1.0))
    box_offset = int(box / 2)
    center = (box - 1) / 2.0

    theta = np.asarray(theta, dtype=np.float64)
    x_shift = theta[:, 0]
    y_shift = theta[:, 1]
    z_shift = theta[:, 2]
    amp = theta[:, 3 : 3 + n_channels]  # per-channel amplitude
    bg_ch = theta[
        :, 3 + n_channels : 3 + 2 * n_channels
    ]  # per-channel background

    ps = _photon_scales(calibration, n_channels)  # (n_channels,)
    photons_ch = amp * ps[None, :]  # per-channel photon counts
    photons = photons_ch.sum(axis=1)
    bg_total = bg_ch.sum(axis=1)

    ids_x = np.asarray(identifications["x"], dtype=np.float64)
    ids_y = np.asarray(identifications["y"], dtype=np.float64)
    x = x_shift / oversampling + center + ids_x - box_offset
    y = y_shift / oversampling + center + ids_y - box_offset

    crlb = precision._spline_link_xyz_crlb(
        theta,
        calibration,
        box,
        mle=mle,
        em=em,
        progress_callback=progress_callback,
        residuals=residuals,
        jacobians=jacobians,
        variance=variance,
    )  # variances [x, y, z, N_0.., bg_0..]
    var_amp = crlb[:, 3 : 3 + n_channels]
    var_bg = crlb[:, 3 + n_channels : 3 + 2 * n_channels]
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(crlb[:, 0]) / oversampling
        lpy = np.sqrt(crlb[:, 1]) / oversampling
        # total-photon uncertainty: independent per-channel photon variances add
        photons_unc = np.sqrt(np.sum(var_amp * (ps[None, :] ** 2), axis=1))
        bg_unc = np.sqrt(np.sum(var_bg, axis=1))

    z_center = float(calibration.get("z_center", 0.0))
    z_init = float(calibration.get("z_init", z_center))
    z_step_nm = float(calibration.get("z_step_nm", 1.0))
    magnification_factor = float(calibration.get("magnification_factor", 1.0))
    z = (z_shift + z_init) * z_step_nm * magnification_factor + (
        z_center - z_init
    ) * z_step_nm
    with np.errstate(invalid="ignore", divide="ignore"):
        lpz = np.sqrt(crlb[:, 2]) * z_step_nm * magnification_factor
        # Each channel's share of the total photons; sums to 1 per spot.
        rel_photons = np.where(
            photons[:, None] > 0, photons_ch / photons[:, None], np.nan
        )

    columns = {
        "frame": np.asarray(identifications["frame"]).astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": z.astype(np.float32),
        "photons": photons.astype(np.float32),
        "bg": bg_total.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "lpz": lpz.astype(np.float32),
        "net_gradient": np.asarray(
            identifications["net_gradient"], dtype=np.float32
        ),
        "photons_unc": photons_unc.astype(np.float32),
        "bg_unc": bg_unc.astype(np.float32),
    }
    for c in range(n_channels):
        columns[f"photons_ch{c}"] = photons_ch[:, c].astype(np.float32)
        columns[f"bg_ch{c}"] = bg_ch[:, c].astype(np.float32)
        columns[f"rel_photons_ch{c}"] = rel_photons[:, c].astype(np.float32)
    if log_likelihood is not None:
        columns["log_likelihood"] = np.asarray(log_likelihood).astype(
            np.float32
        )
    if iterations is not None:
        columns["iterations"] = np.asarray(iterations).astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def locs_from_fits_spline(
    identifications: pd.DataFrame,
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    calibration: dict,
    mle: bool = True,
    log_likelihood: lib.FloatArray1D | None = None,
    iterations: lib.FloatArray1D | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    residuals: np.ndarray | None = None,
    jacobians: np.ndarray | None = None,
    chi_square: lib.FloatArray1D | None = None,
    variance: lib.FloatArray4D | None = None,
) -> pd.DataFrame:
    """Convert spline fit results into a localizations data frame.

    Localization precisions (``lpx``, ``lpy``, ``lpz``) and the ``photons`` /
    ``bg`` uncertainties come from :func:`precision._spline_crlb`: the Poisson
    Cramer-Rao bound for maximum-likelihood fits or the least-squares sandwich
    covariance for least-squares ones.

    Parameters
    ----------
    identifications : pd.DataFrame
        The identifications the spots were cut from, with ``frame``, ``x``,
        ``y`` and ``net_gradient`` columns.
    theta : lib.FloatArray2D
        The fitted parameters, ``[amplitude, x_shift, y_shift, offset]`` (2D)
        or ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D). The
        photon-decoupled model's ``[x, y, z, N_0.., bg_0..]`` layout is
        handled by ``_locs_from_fits_spline_link_xyz``.
    box : int
        The lateral fit box (camera pixels); sets the box offset of ``x`` /
        ``y`` and crops the calibration.
    em : bool
        Whether EMCCD was used, which doubles the variances for excess noise,
        as in the Gaussian fits.
    calibration : dict
        The spline PSF calibration the fit used.
    mle : bool, optional
        Whether ``theta`` came from the maximum-likelihood estimator. Must
        match the estimator that produced it, since it selects the CRLB
        branch. Default True.
    log_likelihood : lib.FloatArray1D, optional
        Per-spot Poisson log-likelihood from an MLE fit; becomes a column when
        given. Default None.
    iterations : lib.FloatArray1D, optional
        Iterations each spot took; becomes a column when given. Default None.
    progress_callback : callable, "console" or None, optional
        Forwarded to :func:`precision._spline_crlb`, which reports the
        per-spot CRLB computation.
    residuals : np.ndarray, optional
        ``(n_spots, n_channels, 2)`` sub-pixel ROI offsets of a multichannel
        fit (see :func:`channel_roi_geometry`). Default None.
    jacobians : np.ndarray, optional
        ``(n_spots, n_channels, 4)`` local Jacobians of the channel transforms
        (see :func:`channel_roi_geometry`). Default None.
    chi_square : lib.FloatArray1D, optional
        Per-spot residual sum of squares at the optimum of a least-squares
        fit; becomes a column when given. See :func:`locs_from_fits_gauss` for
        how to read it. Default None.
    variance : lib.FloatArray4D, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        like the fitted spots; enters the CRLB pixel by pixel. Default None.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, sorted by frame, with ``frame``, ``x``, ``y``,
        ``photons``, ``bg``, ``lpx``, ``lpy``, ``net_gradient``,
        ``photons_unc`` and ``bg_unc``, plus ``z`` and ``lpz`` for a 3D model
        and whichever of ``log_likelihood``, ``iterations`` and
        ``chi_square`` were given. Single-channel results additionally have
        the calibration's lateral transforms applied
        (``lib.apply_lateral_transforms``).
    """
    calibration = crop_spline_calibration(calibration, box)
    model = calibration["model"]
    if model == precision._LINK_XYZ_MODEL:
        # Photon-decoupled model: 3 + 2*n_channels parameters
        # [x, y, z, N_0.., bg_0..] with per-channel photons/background and a
        # continuous per-channel relative-photon readout.
        return _locs_from_fits_spline_link_xyz(
            identifications,
            theta,
            box,
            calibration,
            mle=mle,
            em=em,
            log_likelihood=log_likelihood,
            iterations=iterations,
            progress_callback=progress_callback,
            residuals=residuals,
            jacobians=jacobians,
            chi_square=chi_square,
            variance=variance,
        )
    is_3d = model != "spline-2d"
    box_offset = int(box / 2)
    oversampling = float(calibration.get("oversampling", 1.0))

    amplitude = np.asarray(theta[:, 0])
    x_shift = np.asarray(theta[:, 1])
    y_shift = np.asarray(theta[:, 2])
    offset = np.asarray(theta[:, -1])
    center = (box - 1) / 2.0
    x = x_shift / oversampling + center + identifications["x"] - box_offset
    y = y_shift / oversampling + center + identifications["y"] - box_offset

    # photon_scale converts the fitted (shared) amplitude to a photon count.
    # A multichannel calibration may store a per-channel array; the shared
    # amplitude then maps to the TOTAL photons across channels (their sum). A
    # scalar (single-channel, or older multichannel calibrations) is unchanged.
    photon_scale_raw = calibration.get("photon_scale", 1.0)
    if np.ndim(photon_scale_raw) > 0:
        photon_scale = float(np.sum(np.asarray(photon_scale_raw, dtype=float)))
    else:
        photon_scale = float(photon_scale_raw)
    photons = amplitude * photon_scale

    # CRLB / LSQ variances
    crlb = precision._spline_crlb(
        theta,
        calibration,
        box,
        mle=mle,
        em=em,
        progress_callback=progress_callback,
        residuals=residuals,
        jacobians=jacobians,
        variance=variance,
    )
    amp_var, off_var = crlb[:, -2], crlb[:, -1]
    with np.errstate(invalid="ignore"):
        lpx = np.sqrt(crlb[:, 0]) / oversampling
        lpy = np.sqrt(crlb[:, 1]) / oversampling
        photons_unc = np.sqrt(amp_var) * photon_scale
        bg_unc = np.sqrt(off_var)

    columns = {
        "frame": identifications["frame"].astype(np.uint32),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "photons": photons.astype(np.float32),
        "bg": offset.astype(np.float32),
        "lpx": lpx.astype(np.float32),
        "lpy": lpy.astype(np.float32),
        "net_gradient": identifications["net_gradient"].astype(np.float32),
    }
    if is_3d:
        z_shift = np.asarray(theta[:, 3])
        z_center = float(calibration.get("z_center", 0.0))
        z_init = float(calibration.get("z_init", z_center))
        z_step_nm = float(calibration.get("z_step_nm", 1.0))
        magnification_factor = float(
            calibration.get("magnification_factor", 1.0)
        )
        z_position = (z_shift + z_init) * z_step_nm * magnification_factor
        z_offset_nm = (z_center - z_init) * z_step_nm  # raw stage nm, no mag
        z = z_position + z_offset_nm
        columns["z"] = z.astype(np.float32)
        with np.errstate(invalid="ignore"):
            # var(z_shift) -> nm via the same z-step scaling used for z
            lpz = np.sqrt(crlb[:, 2]) * z_step_nm * magnification_factor
        columns["lpz"] = lpz.astype(np.float32)
    columns["photons_unc"] = photons_unc.astype(np.float32)
    columns["bg_unc"] = bg_unc.astype(np.float32)
    if log_likelihood is not None:
        columns["log_likelihood"] = log_likelihood.astype(np.float32)
    if iterations is not None:
        columns["iterations"] = iterations.astype(np.int32)
    if chi_square is not None:
        columns["chi_square"] = np.asarray(chi_square).astype(np.float32)
    locs = pd.DataFrame(columns)
    locs.sort_values(by="frame", kind="quicksort", inplace=True)
    if precision._spline_n_channels(calibration) > 1:
        return locs
    return lib.apply_lateral_transforms(locs, calibration)


def _fit2d_spline_gpu(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    calibration: dict,
    mle: bool = False,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    n_z_starts: int | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame:
    """Fit an experimentally measured cubic-spline PSF on the GPU. For a 3D
    calibration the localizations contain the fitted ``z`` directly. See
    ``fit`` for more details. ``progress_callback`` tracks the per-spot CRLB
    computation in ``locs_from_fits_spline``.

    ``n_z_starts`` is the axial multi-start (see
    :func:`fit_spots_splinefit`); ``None`` picks it from the calibration.
    Pass 1 for the single in-focus start."""
    theta, log_likelihood, iterations, chi_square = fit_spots_splinefit(
        spots,
        calibration,
        mle=mle,
        return_stats=True,
        n_z_starts=n_z_starts,
        tolerance=tolerance,
        max_iterations=max_iterations,
        use_gpu=True,
        variance=variance,
    )
    locs = locs_from_fits_spline(
        identifications,
        theta,
        box,
        em,
        calibration,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        progress_callback=progress_callback,
        chi_square=chi_square,
        variance=variance,
    )
    return locs


def _fit2d_spline_cpu(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    calibration: dict,
    mle: bool = False,
    n_z_starts: int | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame | None:
    """Fit an experimentally measured cubic-spline PSF on the CPU. For a 3D
    calibration the localizations contain the fitted ``z`` directly. See
    ``fit`` for more details.

    Unlike the GPU path, whose fit is one launch per chunk,
    ``progress_callback`` here tracks the fit itself, one step per spot. The
    per-spot CRLB pass in ``locs_from_fits_spline`` is a second sweep over the
    same localizations, so it only gets the callback in ``"console"`` mode -
    where it draws its own labeled progress bar - rather than rewinding a
    GUI's counter back to zero.

    Returns None if ``abort_callback`` asked to stop."""
    result = fit_spots_splinefit(
        spots,
        calibration,
        mle=mle,
        n_z_starts=n_z_starts,
        return_stats=True,
        tolerance=tolerance,
        max_iterations=max_iterations,
        multiprocess=multiprocess,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        variance=variance,
    )
    if result is None:
        return None
    theta, log_likelihood, iterations, chi_square = result
    return locs_from_fits_spline(
        identifications,
        theta,
        box,
        em,
        calibration,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        progress_callback=(
            "console" if progress_callback == "console" else None
        ),
        chi_square=chi_square,
        variance=variance,
    )


# ----------------------------------------------------------------------
# Multichannel cubic-spline PSF fitting (shared-amplitude 3D model)
#
# Several spatially-registered channels (separate movies, e.g. biplane
# microscopy) are fit simultaneously with shared x, y, z. A detection in
# the reference channel is mapped into every channel via a per-channel affine
# transform (stored in the calibration), the box ROIs are cut from each
# channel and stacked, and the stack is fitted against the per-channel spline
# coefficients. There is no affine-transform machinery elsewhere in Picasso,
# so the small least-squares helpers below are self-contained.
#
# This is the global-fitting scheme of globLoc: Li, Y., Shi, W., Liu, S.,
# Cavka, I., Wu, Y.-L., Matti, U., Wu, D., Koehler, S. & Ries, J. "Global
# fitting for high-accuracy multi-channel single-molecule localization."
# Nature Communications 13, 3133 (2022). DOI: 10.1038/s41467-022-30719-4.
# Linked parameters (shared x, y, z), the optional per-channel photon /
# background decoupling (``_as_link_xyz_calibration``) and the ratiometric
# color assignment (``fit_spline_multichannel_ratiometric``) all follow that
# work; the calibration side lives in ``picasso.spline``.
# ----------------------------------------------------------------------


def _region_origin_xy(
    rect: tuple[tuple[int, int], tuple[int, int]] | list,
) -> np.ndarray:
    """``(x, y)`` top-left origin of a ``[[y_a, x_a], [y_b, x_b]]`` rectangle."""
    (ya, xa), (yb, xb) = rect
    return np.array([float(min(xa, xb)), float(min(ya, yb))], dtype=np.float64)


def decompose_region_transforms(
    region_rects: list, transforms: list
) -> list[tform.Transform]:
    """Region-local channel registration from absolute channel transforms
    (split-FOV).

    Each absolute ``transform`` maps reference-channel **absolute** chip
    coordinates to channel-``c`` **absolute** coordinates. This strips out the
    region placement and returns the transform ``A_c`` that maps
    reference-**region-local** coordinates (relative to the reference region's
    top-left) to channel-``c``-region-local coordinates - the *inter-channel*
    registration, independent of where the regions sit on the chip (identity for
    a perfectly aligned, same-orientation split; ``A_0`` is the identity).

    This is the ROI-agnostic form stored in the calibration: the coarse region
    offset lives in the ROI positions (chosen at fit time), while ``A_c`` carries
    only the fine sub-pixel/rotation/scale registration. Inverse of
    :func:`compose_region_transforms`.

    Both directions are ``T(x + pre) + post`` with the region origins as the
    two shifts, so they work for every transform model - see
    :meth:`picasso.transforms.Transform.compose_translations`, which is exact
    for affine, projective and polynomial alike.

    Parameters
    ----------
    region_rects : list
        One ``[[y_a, x_a], [y_b, x_b]]`` rectangle per channel, the reference
        first. Only their origins (top-left corners) are used.
    transforms : list
        The absolute reference -> channel transforms, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``.

    Returns
    -------
    affines : list of picasso.transforms.Transform
        The region-local registration ``A_c``, one per channel; ``A_0`` is the
        identity.
    """
    o0 = _region_origin_xy(region_rects[0])
    affines = []
    for rect, transform in zip(region_rects, transforms):
        oc = _region_origin_xy(rect)
        affines.append(
            tform.from_dict(transform).compose_translations(pre=o0, post=-oc)
        )
    return affines


def compose_region_transforms(
    region_rects: list, affines: list
) -> list[tform.Transform]:
    """Absolute channel transforms from region-local registration + region
    positions.

    Inverse of :func:`decompose_region_transforms`: given the region rectangles
    in use (e.g. re-drawn at fit time) and the stored region-local ``affines``,
    rebuild the absolute reference->channel transforms placed at those regions.
    Only the region *origins* enter, so fit-time regions may differ in size from
    the calibration ones - the placement follows their top-left corners.

    A fit-time region that reaches well beyond the field the transform was
    calibrated on is warned about: harmless for an affine, but a polynomial
    diverges quickly outside its ``domain``.

    Parameters
    ----------
    region_rects : list
        One ``[[y_a, x_a], [y_b, x_b]]`` rectangle per channel, the reference
        first, as in use at fit time. Only their origins are used, so they may
        differ in size from the calibration's.
    affines : list
        The region-local registration, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``.

    Returns
    -------
    transforms : list of picasso.transforms.Transform
        The absolute reference -> channel transforms placed at those regions.
    """
    o0 = _region_origin_xy(region_rects[0])
    transforms = []
    for c, (rect, affine) in enumerate(zip(region_rects, affines)):
        oc = _region_origin_xy(rect)
        placed = tform.from_dict(affine).compose_translations(pre=-o0, post=oc)
        _warn_if_extrapolating(placed, rect, c)
        transforms.append(placed)
    return transforms


# A fit-time region may exceed the calibrated field by this fraction of the
# domain's size before it is worth warning about.
_EXTRAPOLATION_TOLERANCE = 0.2


def _warn_if_extrapolating(transform, rect, channel: int) -> None:
    """Warn when ``rect`` reaches outside the field ``transform`` was fitted
    on. Only non-affine models can misbehave there, so only they warn (a
    translation is an affine, and extrapolates just as harmlessly)."""
    domain = getattr(transform, "domain", None)
    if domain is None or isinstance(transform, tform.AffineTransform):
        return
    (y0, x0), (y1, x1) = _normalize_rect(rect)
    corners = np.array(
        [[float(x0), float(y0)], [float(x1), float(y1)]], dtype=np.float64
    )
    span = np.maximum(domain[1] - domain[0], 1.0)
    margin = _EXTRAPOLATION_TOLERANCE * span
    if np.any(corners[0] < domain[0] - margin) or np.any(
        corners[1] > domain[1] + margin
    ):
        warnings.warn(
            f"The region used for channel {channel} reaches outside the field "
            f"its {transform.model} registration was calibrated on. That "
            "model "
            "extrapolates poorly, so the registration may be inaccurate near "
            "the edges; re-draw the region or re-calibrate.",
            stacklevel=3,
        )


def multichannel_inbounds_ids(
    identifications: pd.DataFrame,
    box: int,
    movies: list,
    transforms: list,
) -> pd.DataFrame:
    """Filter reference detections to those whose full ``box`` fits inside
    every channel's frame after mapping through the per-channel transforms.

    Parameters
    ----------
    identifications : pd.DataFrame
        Detections in the reference channel, with ``x`` and ``y`` columns.
    box : int
        Box side length (camera pixels).
    movies : list
        One movie per channel; only their frame shapes are read.
    transforms : list
        One reference -> channel transform per channel, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``. ``transforms[0]`` is the identity.

    Returns
    -------
    identifications : pd.DataFrame
        The rows whose box fits in every channel, re-indexed. The input frame
        itself (not a copy) when nothing is dropped.
    """
    r = box // 2
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    inside = np.ones(len(ref_xy), dtype=bool)
    for c, movie in enumerate(movies):
        xy = ref_xy if c == 0 else tform.from_dict(transforms[c]).apply(ref_xy)
        x = np.rint(xy[:, 0]).astype(np.int64)
        y = np.rint(xy[:, 1]).astype(np.int64)
        height, width = int(movie.shape[1]), int(movie.shape[2])
        inside &= (
            (x - r >= 0) & (x + r < width) & (y - r >= 0) & (y + r < height)
        )
    if inside.all():
        return identifications
    n_total = len(inside)
    n_dropped = int((~inside).sum())
    # Dropping a few edge detections is normal; dropping a large fraction almost
    # always means the inter-channel registration is wrong (detections map off
    # the frame in another channel), so make that visible instead of silently
    # returning a tiny, edge-clustered subset.
    if n_total and n_dropped / n_total >= 0.2:
        warnings.warn(
            f"Multichannel spot extraction dropped {n_dropped} of {n_total} "
            f"detections ({100 * n_dropped / n_total:.0f}%) whose box falls "
            "outside a channel after mapping - the split-FOV registration is "
            "likely off (re-register the channels or check the drawn ROIs).",
            stacklevel=2,
        )
    return identifications.iloc[np.flatnonzero(inside)].reset_index(drop=True)


def _matched_mask_within_tol(
    pred_xy: np.ndarray, target_xy: np.ndarray, tol: float
) -> np.ndarray:
    """Nearest-neighbor pairing within ``tol``, one-to-one.

    ``pred_xy`` are reference detections predicted into a channel (via the
    calibration transform), ``target_xy`` are that channel's own detections.
    Each reference proposes its nearest target; conflicts are resolved in order
    of increasing distance so every reference and every target is used at most
    once. Returns a boolean mask over ``pred_xy`` marking the paired rows - the
    same pairing the viewer's link colors use (``_nearest_unique_match`` in
    ``picasso.gui.localize``), reduced to "was this reference spot matched".
    """
    n_pred = len(pred_xy)
    matched = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or len(target_xy) == 0:
        return matched
    dists = np.sqrt(
        ((pred_xy[:, None, :] - target_xy[None, :, :]) ** 2).sum(axis=2)
    )  # (n_pred, n_target)
    nearest = np.argmin(dists, axis=1)
    best = dists[np.arange(n_pred), nearest]
    candidates = np.flatnonzero(best <= tol)
    if len(candidates) == 0:
        return matched
    # closest pair wins; a target already claimed cannot be reused
    used_target: set[int] = set()
    for k in candidates[np.argsort(best[candidates], kind="stable")]:
        j = int(nearest[k])
        if j in used_target:
            continue
        used_target.add(j)
        matched[k] = True
    return matched


def link_identifications_multichannel(
    identifications_per_channel: list,
    transforms: list,
    tol: float,
    progress_callback: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Boolean mask over the reference channel's detections marking those that
    are also detected in *every* other channel.

    A molecule that the joint multichannel model can describe must be present in
    all channels: the fit ties one shared ``x, y, z`` to the *relative*
    intensities across channels. A reference detection with no counterpart in
    some channel is fitted there against background only, which biases the
    shared parameters. Filtering on this mask keeps only the
    cross-channel-linked molecules.

    Parameters
    ----------
    identifications_per_channel : list of pd.DataFrame
        One identification table per channel, ``[0]`` being the reference. Each
        needs ``frame``, ``x``, ``y``. ``None`` or empty entries mean that
        channel was not identified; it is then skipped (not counted as a
        missing match), so a partially identified set degrades to linking
        against the channels that are available.
    transforms : list
        One ``(2, 3)`` affine per channel mapping reference coordinates into
        that channel (as stored in the calibration's ``channel_transforms``).
    tol : float
        Pairing radius in camera pixels (the GUI preview uses ``1.5 * box``).
    progress_callback : callable, optional
        Called with the cumulative number of reference detections processed.

    Returns
    -------
    linked : np.ndarray
        Boolean mask over ``identifications_per_channel[0]`` rows. All ``False``
        if no other channel carries identifications.
    """
    reference = identifications_per_channel[0]
    n_ref = 0 if reference is None else len(reference)
    if n_ref == 0:
        return np.zeros(0, dtype=bool)
    ref_frames = np.asarray(reference["frame"], dtype=np.int64)
    ref_xy = np.column_stack(
        [
            np.asarray(reference["x"], dtype=np.float64),
            np.asarray(reference["y"], dtype=np.float64),
        ]
    )
    # frame-sorted view of the reference rows, so each frame is a slice
    ref_order = np.argsort(ref_frames, kind="stable")
    ref_frames_sorted = ref_frames[ref_order]
    frames, frame_starts = np.unique(ref_frames_sorted, return_index=True)
    frame_stops = np.append(frame_starts[1:], n_ref)

    # channels that can be linked against (an unidentified channel is skipped)
    check = [
        c
        for c in range(1, len(identifications_per_channel))
        if identifications_per_channel[c] is not None
        and len(identifications_per_channel[c])
    ]
    n_checked = len(check)
    if n_checked == 0:
        return np.zeros(n_ref, dtype=bool)

    match_count = np.zeros(n_ref, dtype=np.int64)
    for i_c, c in enumerate(check):
        ids_c = identifications_per_channel[c]
        pred = tform.from_dict(transforms[c]).apply(ref_xy)[ref_order]
        c_frames = np.asarray(ids_c["frame"], dtype=np.int64)
        c_xy = np.column_stack(
            [
                np.asarray(ids_c["x"], dtype=np.float64),
                np.asarray(ids_c["y"], dtype=np.float64),
            ]
        )
        c_order = np.argsort(c_frames, kind="stable")
        c_frames_sorted = c_frames[c_order]
        c_xy_sorted = c_xy[c_order]
        # this channel's rows for each reference frame, by binary search
        c_lo = np.searchsorted(c_frames_sorted, frames, side="left")
        c_hi = np.searchsorted(c_frames_sorted, frames, side="right")
        done = 0
        for f in range(len(frames)):
            start, stop = frame_starts[f], frame_stops[f]
            lo, hi = c_lo[f], c_hi[f]
            if hi > lo:
                matched = _matched_mask_within_tol(
                    pred[start:stop], c_xy_sorted[lo:hi], tol
                )
                match_count[ref_order[start:stop][matched]] += 1
            done += stop - start
            # one monotone 0 -> n_ref progression across all checked channels
            if callable(progress_callback) and (f % 2000 == 0):
                progress_callback((i_c * n_ref + done) // n_checked)
    if callable(progress_callback):
        progress_callback(n_ref)
    return match_count == n_checked


def filter_linked_identifications(
    identifications_per_channel: list,
    transforms: list,
    box: int,
    tol: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Keep only the reference detections linked across *all* channels.

    Thin wrapper around :func:`link_identifications_multichannel`. If no other
    channel has identifications, the reference table is returned unchanged so
    an un-identified set degrades to the previous behavior instead of fitting
    nothing.

    Parameters
    ----------
    identifications_per_channel : list
        One detection table per channel, the reference first. Entries may be
        None or empty.
    transforms : list
        One reference -> channel transform per channel, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``.
    box : int
        Box side length (camera pixels); sets the default pairing radius.
    tol : float, optional
        Largest distance (camera pixels) at which a mapped reference detection
        and a channel detection count as the same molecule. None (the default)
        uses ``1.5 * box``.
    progress_callback : callable, optional
        Invoked with a monotone 0 -> ``n_total`` progression.

    Returns
    -------
    identifications : pd.DataFrame
        The linked reference detections, re-indexed.
    n_kept : int
        How many were kept.
    n_total : int
        How many reference detections there were. Equal to ``n_kept`` when no
        other channel had identifications.

    Warns
    -----
    UserWarning
        If 5% or less of the reference detections survive, which means the
        registration or the pairing radius is off rather than that the sample
        is empty.
    """
    reference = identifications_per_channel[0]
    n_total = 0 if reference is None else len(reference)
    others = [
        ids
        for ids in identifications_per_channel[1:]
        if ids is not None and len(ids)
    ]
    if n_total == 0 or not others:
        return reference, n_total, n_total
    if tol is None:
        tol = 1.5 * float(box)
    linked = link_identifications_multichannel(
        identifications_per_channel,
        transforms,
        tol,
        progress_callback=progress_callback,
    )
    n_kept = int(linked.sum())
    # Keeping almost nothing means the inter-channel registration (or the
    # pairing radius) is off rather than that the sample is empty - the same
    # failure mode ``multichannel_inbounds_ids`` warns about.
    if n_total and n_kept / n_total <= 0.05:
        warnings.warn(
            f"Cross-channel linking kept only {n_kept} of {n_total} "
            f"reference detections ({100 * n_kept / n_total:.1f}%) - the "
            "channel registration is likely off, or the other channels were "
            "identified with a much higher threshold.",
            stacklevel=2,
        )
    return (
        reference.iloc[np.flatnonzero(linked)].reset_index(drop=True),
        n_kept,
        n_total,
    )


def get_spots_multichannel(
    movies: list,
    identifications: pd.DataFrame,
    box: int,
    camera_infos: list[dict],
    transforms: list,
    progress_callback: Callable[[int], None] | None = None,
    return_residuals: bool = False,
    camera_calibrations: list[dict | None] | None = None,
    return_variance: bool = False,
    return_jacobians: bool = False,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Extract channel-stacked spots for multichannel spline fitting.

    For each identification (given in the reference channel's coordinates),
    the position is mapped into every channel via its affine ``transform`` and
    the box ROI is cut from that channel's movie. The per-channel ROIs are
    stacked along a new trailing axis.

    Parameters
    ----------
    movies : list
        One movie per channel (as loaded by ``io.load_movie``). ``movies[0]``
        is the reference channel.
    identifications : pd.DataFrame
        Detections in the reference channel (``frame``, ``x``, ``y``,
        ``net_gradient``).
    box : int
        Box side length (camera pixels).
    camera_infos : list of dict
        One camera-info dict per channel (for the photon conversion).
    transforms : list
        One reference -> channel transform per channel, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``; ``transforms[0]`` is the identity.
    progress_callback : callable, optional
        Forwarded to ``get_spots`` for the reference channel.
    return_residuals : bool, optional
        Also return the sub-pixel ROI-placement residuals, i.e. the fractional
        part discarded when each channel's box is snapped to an integer pixel.
        The fit models need these to evaluate the spline where the data
        actually is; see :func:`channel_roi_residuals`. Default False.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS camera calibration per channel, or None for no
        calibration at all. Individual entries may be None when only some
        channels are on a characterized camera. Default None.
    return_variance : bool, optional
        Also return the per-spot readout variance in photoelectrons squared,
        channel-stacked exactly like ``spots``. None when no calibration was
        given. Default False.
    return_jacobians : bool, optional
        Also return the local Jacobians of the channel transforms at each
        detection (see below). Default False.

    Returns
    -------
    spots : np.ndarray
        Array of shape ``(n_spots, box, box, n_channels)`` in photon units.
    residuals : np.ndarray
        Only if ``return_residuals``. ``(n_spots, n_channels, 2)`` in ``[x,
        y]`` order; channel 0 is exactly zero.
    variance : np.ndarray or None
        Only if ``return_variance``. Same shape as ``spots``.
    jacobians : np.ndarray
        Only if ``return_jacobians``. ``(n_spots, n_channels, 4)`` local
        Jacobians of the channel transforms at each detection - the other half
        of the same linearization as ``residuals`` (see
        :func:`channel_roi_geometry`). Constant across spots for an affine
        registration.

    The returned tuple is ordered ``spots, residuals, variance, jacobians``,
    each present only if requested.
    """
    n_channels = len(movies)
    if not (len(camera_infos) == len(transforms) == n_channels):
        raise ValueError(
            "movies, camera_infos and transforms must have the same length "
            "(one per channel)."
        )
    if camera_calibrations is None:
        camera_calibrations = [None] * n_channels
    elif len(camera_calibrations) != n_channels:
        raise ValueError(
            "camera_calibrations must have one entry per channel "
            f"({n_channels}), got {len(camera_calibrations)}."
        )
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    channel_spots = []
    channel_variance = []
    residuals = np.zeros((len(ref_xy), n_channels, 2), dtype=np.float32)
    for c in range(n_channels):
        if c == 0:
            ids_c = identifications
        else:
            mapped = tform.from_dict(transforms[c]).apply(ref_xy)
            ids_c = identifications.copy()
            # get_spots/_cut_spots cut an INTEGER-pixel box, so the box origin
            # cannot express the fractional part of the mapped position. Keep
            # it: the fit model subtracts it from the evaluation position (see
            # channel_roi_residuals). Channel 0 is the reference and its box
            # sits on the (integer) detection itself, so its residual is 0.
            rounded = np.rint(mapped)
            residuals[:, c, :] = (mapped - rounded).astype(np.float32)
            ids_c["x"] = rounded[:, 0].astype(np.int64)
            ids_c["y"] = rounded[:, 1].astype(np.int64)
        # ``ids_c`` carries this channel's *rounded* box origins, so cutting
        # the calibration maps from it here - rather than from the reference
        # identifications - is what keeps a channel's variance patch aligned
        # with the spot it accompanies.
        spots_c, variance_c = get_spots(
            movies[c],
            ids_c,
            box,
            camera_infos[c],
            progress_callback=progress_callback if c == 0 else None,
            camera_calibration=camera_calibrations[c],
            return_variance=True,
        )
        channel_spots.append(spots_c)
        if variance_c is None:
            # A channel on an uncharacterized camera keeps the plain Poisson
            # model, which is what a zero readout variance means.
            variance_c = np.zeros_like(spots_c)
        channel_variance.append(variance_c)
    spots = np.stack(channel_spots, axis=-1)
    variance = (
        np.stack(channel_variance, axis=-1)
        if any(c is not None for c in camera_calibrations)
        else None
    )
    result = (spots,)
    if return_residuals:
        result += (residuals,)
    if return_variance:
        result += (variance,)
    if return_jacobians:
        result += (tform.channel_jacobians(transforms, ref_xy),)
    return result[0] if len(result) == 1 else result


def channel_roi_residuals(
    identifications: pd.DataFrame, transforms: list
) -> np.ndarray:
    """Sub-pixel ROI-placement residual per localization and channel.

    :func:`get_spots_multichannel` cuts each channel's box at an integer pixel
    (``rint`` of the mapped position), so the fractional part of the mapping is
    not representable by the box origin. That leftover, ``mapped -
    rint(mapped)``, is what this returns; the multichannel spline models
    subtract it from the position at which they evaluate the spline (see the
    residual block in the multichannel kernels).

    Detections sit on integer pixels, so for a channel transform
    ``A = I + E`` the residual is ``(E@x + b) - rint(E@x + b)``.

    :func:`get_spots_multichannel` computes the same quantity as a by-product
    of cutting the ROIs; prefer its ``return_residuals=True`` form when you are
    extracting spots anyway. This function is for callers that already have
    spots and only need the residuals.

    Parameters
    ----------
    identifications : pd.DataFrame
        Detections in the reference channel, with integer ``x``/``y`` columns -
        the same frame the ROIs are cut in.
    transforms : list
        One reference -> channel transform per channel, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``, ``transforms[0]`` the identity (as stored in
        the calibration's ``channel_transforms``).

    Returns
    -------
    residuals : np.ndarray
        ``(n_spots, n_channels, 2)`` float32 in ``[x, y]`` order, matching the
        model's ``[fit][channel][x, y]`` residual block. Channel 0 is the
        reference (its box is cut on the detection itself) and is exactly zero.
    """
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    residuals = np.zeros((len(ref_xy), len(transforms), 2), dtype=np.float32)
    for c in range(1, len(transforms)):
        mapped = tform.from_dict(transforms[c]).apply(ref_xy)
        residuals[:, c, :] = (mapped - np.rint(mapped)).astype(np.float32)
    return residuals


def channel_roi_geometry(
    identifications: pd.DataFrame, transforms: list
) -> tuple[np.ndarray, np.ndarray]:
    """``(residuals, jacobians)`` describing how each channel's ROI sits.

    The two halves of the same linearization, evaluated at the same points:
    the multichannel models place channel ``c``'s box at ``rint(T_c(x))`` and
    evaluate the spline at ``J_c(x) @ theta + (T_c(x) - rint(T_c(x)))``, which
    is ``T_c(x + theta)`` to first order. So the ROI residual and the local
    Jacobian belong together, and both are per localization.

    ``theta`` there is the emitter's *displacement* from the box center. The
    Gaussian models parameterize the position by its box coordinate instead,
    so their kernels linearize around the center explicitly: channel ``c``
    sits at ``center + J_c @ (theta - center) + residual``. Feeding the box
    coordinate straight into the Jacobian only looks right for a nearly
    identity registration - a mirrored (split field of view) or strongly
    rotated channel then lands outside its own box.

    They are returned separately rather than bundled because the fitters may
    zero the residuals (``apply_roi_residuals=False``) while still needing the
    geometry.

    Parameters
    ----------
    identifications : pd.DataFrame
        Detections in the reference channel, with integer ``x``/``y`` columns.
    transforms : list
        One reference -> channel transform per channel, each a
        ``picasso.transforms.Transform`` or a dict accepted by
        ``transforms.from_dict``, ``transforms[0]`` the identity.

    Returns
    -------
    residuals : np.ndarray
        ``(n_spots, n_channels, 2)`` float32, as
        :func:`channel_roi_residuals`.
    jacobians : np.ndarray
        ``(n_spots, n_channels, 4)`` float64 rows ``[a00, a01, a10, a11]``.
        Constant across spots for an affine registration, position-dependent
        for a projective or polynomial one.
    """
    ref_xy = np.column_stack(
        [
            np.asarray(identifications["x"], dtype=np.float64),
            np.asarray(identifications["y"], dtype=np.float64),
        ]
    )
    return (
        channel_roi_residuals(identifications, transforms),
        tform.channel_jacobians(transforms, ref_xy),
    )


def fit_spline_multichannel(
    movies: list,
    camera_infos: list[dict],
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    mle: bool = False,
    link_photons: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    apply_roi_residuals: bool = True,
    n_z_starts: int | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    camera_calibrations: list[dict | None] | None = None,
) -> pd.DataFrame:
    """Fit a multichannel cubic-spline PSF across several registered channels.

    Global fit in the sense of globLoc (Li et al., Nat. Commun. 13, 3133,
    2022): every channel contributes to one fit with linked x, y and z.

    Ties ``get_spots_multichannel`` (extraction via the calibration's stored
    ``channel_transforms``) to ``fit_spots_spline`` and
    ``locs_from_fits_spline``. The resulting localizations are in the
    reference channel's coordinates and contain the fitted ``z`` directly.

    Parameters
    ----------
    movies : list
        One movie per channel; ``movies[0]`` is the reference channel and its
        order must match the calibration's channels.
    camera_infos : list of dict
        One camera-info dict per channel.
    identifications : pd.DataFrame
        Detections in the reference channel.
    box : int
        Box side length (camera pixels), must match the calibration.
    calibration : dict
        A ``"spline-3d-multichannel"`` calibration (see ``picasso.spline``).
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator. Default False.
    use_gpu : bool or None, optional
        Fit on the GPU (``picasso.fitting.splinefit_cuda``) or on the CPU
        (``picasso.fitting.splinefit``). None (the default) uses the GPU when
        one is
        available. Both compute the same quantity, so this only affects speed.
    link_photons : bool, optional
        If True (default), the shared-amplitude model links one photon
        amplitude and one background across all channels. If False, use
        the photon-decoupled model: x, y, z stay shared but each channel
        gets its own photon count and background, reported as
        ``photons_ch{c}`` / ``bg_ch{c}`` / ``rel_photons_ch{c}``.
        Available for 2 to 6 channels. See :func:`_as_link_xyz_calibration`.
    apply_roi_residuals : bool, optional
        Hand the sub-pixel ROI-placement residuals to the fit model (default
        True), so each channel's spline is evaluated where its data actually
        sits rather than at the nearest whole pixel. See
        :func:`channel_roi_residuals` for why this matters and by how much. Set
        False to reproduce results from before this correction, or to A/B the
        two on the same data.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS camera calibration per channel (from
        ``picasso.scmos`` or ``io.load_camera_calibration``), or None for none
        at all; individual entries may be None when only some channels sit on
        a characterized camera. Each channel's maps are cut at that channel's
        own mapped, rounded box origin, so a calibration follows its channel
        through the affine registration. Default None.
    progress_callback : callable, optional
        Invoked with the cumulative spot count as the extraction, the fit and
        the CRLB computation proceed.
    n_z_starts : int, optional
        Number of axial seeds of the multi-start. None (the default) picks it
        from the calibration; see :func:`fit_spots_splinefit`.
    tolerance, max_iterations : optional
        Convergence schedule. None (the default) uses the one the fit would
        use by default; see :func:`fit_spots_splinefit`.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, in the reference channel's coordinates, containing
        the fitted ``z`` directly (see :func:`locs_from_fits_spline`). With
        ``link_photons=False`` they additionally carry ``photons_ch{c}``,
        ``bg_ch{c}`` and ``rel_photons_ch{c}``.

    Raises
    ------
    ValueError
        If the calibration is not ``"spline-3d-multichannel"``, or if the
        number of movies disagrees with its channel transforms.
    """
    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "fit_spline_multichannel requires a 'spline-3d-multichannel' "
            "calibration."
        )
    if not link_photons:
        calibration = _as_link_xyz_calibration(calibration)
    transforms = calibration["channel_transforms"]
    if len(movies) != len(transforms):
        raise ValueError(
            f"Got {len(movies)} channels but the calibration has "
            f"{len(transforms)} channel transforms."
        )
    identifications = multichannel_inbounds_ids(
        identifications, box, movies, transforms
    )
    spots, residuals, variance, jacobians = get_spots_multichannel(
        movies,
        identifications,
        box,
        camera_infos,
        transforms,
        progress_callback=progress_callback,
        return_residuals=True,
        camera_calibrations=camera_calibrations,
        return_variance=True,
        return_jacobians=True,
    )
    theta, log_likelihood, iterations, chi_square = fit_spots_spline(
        spots,
        calibration,
        mle=mle,
        return_stats=True,
        residuals=residuals if apply_roi_residuals else None,
        jacobians=jacobians,
        n_z_starts=n_z_starts,
        use_gpu=use_gpu,
        tolerance=tolerance,
        max_iterations=max_iterations,
        progress_callback=progress_callback,
        variance=variance,
    )
    em = camera_infos[0].get("Gain", 1) > 1
    return locs_from_fits_spline(
        identifications,
        theta,
        box,
        em,
        calibration,
        mle=mle,
        log_likelihood=log_likelihood,
        iterations=iterations,
        progress_callback=progress_callback,
        residuals=residuals if apply_roi_residuals else None,
        jacobians=jacobians,
        chi_square=chi_square,
        variance=variance,
    )


def scale_channel_blocks(
    coefficients: np.ndarray, ratios: lib.FloatArray1D
) -> np.ndarray:
    """Scale each channel's spline coefficient block by a per-channel factor.

    ``coefficients`` is a multichannel table
    ``(64, n_int_x, n_int_y, n_int_z, n_channels)``. Because the cubic spline is
    linear in its coefficients, multiplying channel ``c``'s block by ``r[c]``
    scales that channel's model exactly: ``mu_c = offset + amplitude * r[c] *
    phi_c``. This is how a fixed per-channel photon **ratio** is imposed for
    ratiometric color assignment (and how an unequal biplane photon split is
    baked in) without changing the model itself. Only the *relative*
    ratios matter - the shared amplitude absorbs any overall scale.

    Parameters
    ----------
    coefficients : np.ndarray
        A ``(64, n_int_x, n_int_y, n_int_z, n_channels)`` multichannel spline
        coefficient table.
    ratios : lib.FloatArray1D
        One scale factor per channel.

    Returns
    -------
    scaled : np.ndarray
        A new ``float32`` table; the input is not modified.

    Raises
    ------
    ValueError
        If ``coefficients`` is not a multichannel table, or if ``ratios`` does
        not have one entry per channel.
    """
    coeff = np.array(coefficients, dtype=np.float32, copy=True)
    if coeff.ndim != 5:
        raise ValueError(
            "scale_channel_blocks expects a multichannel coefficient table "
            "(64, n_int_x, n_int_y, n_int_z, n_channels)."
        )
    ratios = np.asarray(ratios, dtype=np.float32)
    if coeff.shape[-1] != len(ratios):
        raise ValueError(
            f"Got {len(ratios)} ratios but the coefficient table has "
            f"{coeff.shape[-1]} channels."
        )
    for c in range(coeff.shape[-1]):
        coeff[..., c] *= ratios[c]
    return coeff


def _photon_scales(calibration: dict, n_channels: int) -> np.ndarray:
    """Per-channel ``photon_scale`` as a length-``n_channels`` array.

    Accepts either a per-channel array or a single scalar (broadcast to every
    channel, i.e. the equal-brightness assumption of older calibrations)."""
    ps = np.asarray(calibration.get("photon_scale", 1.0), dtype=float).ravel()
    if ps.size == 1:
        ps = np.repeat(ps, n_channels)
    if ps.size != n_channels:
        raise ValueError(
            f"photon_scale has {ps.size} entries but the calibration has "
            f"{n_channels} channels."
        )
    return ps


def fit_spline_multichannel_ratiometric(
    movies: list,
    camera_infos: list[dict],
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    photon_ratios: lib.FloatArray2D | None = None,
    mle: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    apply_roi_residuals: bool = True,
    n_z_starts: int | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    camera_calibrations: list[dict | None] | None = None,
) -> pd.DataFrame:
    """Ratiometric multichannel spline fit with photon-ratio color assignment.

    Implements globLoc's ratiometric scheme (Li et al., Nat. Commun. 13, 3133,
    2022) on top of the existing ``spline-3d-multichannel`` model: for each
    candidate per-channel photon **ratio** (one per dye/color), the per-channel
    coefficient blocks are scaled by that ratio (see
    :func:`scale_channel_blocks`) and the channel-stacked spots are fit. Each
    spot is then assigned the ratio whose fit best explains the data (lowest
    residual / highest likelihood); the winning ratio index is the color.

    Selection uses the **least-squares** residual by default (``mle=False``).
    The maximum-likelihood chi-square can still be unavailable for a spot whose
    fit diverged, so when ``mle=True`` the ranking is restricted to converged
    fits. (Before the Numba CUDA port this was a much bigger effect: Gpufit
    abandoned any fit whose model rang negative, which was a large fraction of
    them - see ``splinefit.MU_FLOOR``.)

    Parameters
    ----------
    movies, camera_infos, identifications, box, calibration
        As in :func:`fit_spline_multichannel`.
    photon_ratios : lib.FloatArray2D, optional
        ``(n_hypotheses, n_channels)`` candidate per-channel photon ratios,
        one row per dye/color. None (the default) takes them from the
        calibration's ``"photon_ratios"`` field. Only the relative per-channel
        values matter (the shared amplitude absorbs the overall scale).
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator. Default False.
    progress_callback : callable, optional
        Invoked with the cumulative spot count as the fit proceeds.
    apply_roi_residuals : bool, optional
        As in :func:`fit_spline_multichannel`. Default True.
    n_z_starts : int, optional
        Number of axial seeds, shared by every hypothesis so their scores stay
        comparable. None (the default) picks it from the calibration.
    use_gpu : bool or None, optional
        As in :func:`fit_spline_multichannel`.
    tolerance, max_iterations : optional
        Convergence schedule. None (the default) uses the fit's own.
    camera_calibrations : list of dict or None, optional
        One per-pixel sCMOS camera calibration per channel (from
        ``picasso.scmos`` or ``io.load_camera_calibration``), or None for none
        at all; individual entries may be None when only some channels sit on
        a characterized camera. Each channel's maps are cut at that channel's
        own mapped, rounded box origin, so a calibration follows its channel
        through the affine registration. Default None.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, in the reference channel's coordinates, with an
        added integer ``color`` column (the winning ratio index) and
        per-channel photon columns ``photons_ch{c}`` (``photons`` is their
        sum).

    Raises
    ------
    ValueError
        If the calibration is not ``"spline-3d-multichannel"``, if no
        ``photon_ratios`` are given or stored, or if the movie count or the
        ratio width disagrees with the calibration's channel count.
    """
    if calibration.get("model") != "spline-3d-multichannel":
        raise ValueError(
            "fit_spline_multichannel_ratiometric requires a "
            "'spline-3d-multichannel' calibration."
        )
    if photon_ratios is None:
        photon_ratios = calibration.get("photon_ratios")
    if photon_ratios is None:
        raise ValueError(
            "No photon_ratios given and none stored in the calibration; "
            "provide a (n_hypotheses, n_channels) array of candidate ratios."
        )
    photon_ratios = np.atleast_2d(np.asarray(photon_ratios, dtype=np.float64))
    transforms = calibration["channel_transforms"]
    n_channels = len(transforms)
    if len(movies) != n_channels:
        raise ValueError(
            f"Got {len(movies)} channels but the calibration has "
            f"{n_channels} channel transforms."
        )
    if photon_ratios.shape[1] != n_channels:
        raise ValueError(
            f"photon_ratios has {photon_ratios.shape[1]} channels but the "
            f"calibration has {n_channels}."
        )

    identifications = multichannel_inbounds_ids(
        identifications, box, movies, transforms
    )
    spots, roi_residuals, variance, jacobians = get_spots_multichannel(
        movies,
        identifications,
        box,
        camera_infos,
        transforms,
        progress_callback=progress_callback,
        return_residuals=True,
        camera_calibrations=camera_calibrations,
        return_variance=True,
        return_jacobians=True,
    )
    if not apply_roi_residuals:
        roi_residuals = None
    n_spots = len(spots)
    n_hyp = len(photon_ratios)
    # Normalize each hypothesis so the shared amplitude keeps a total-photon
    # meaning; the ranking is unaffected by the overall scale.
    ratios_norm = photon_ratios / photon_ratios.sum(axis=1, keepdims=True)

    # Fit every hypothesis; keep per-spot parameters, fit state and score.
    # Every hypothesis gets the same axial multi-start, so the scores that
    # rank them are comparable - a hypothesis must not win by having landed in
    # a better axial minimum by luck.
    if n_z_starts is None:
        n_z_starts = _default_n_z_starts(calibration)
    thetas = []
    chis = []  # raw per-hypothesis chi-squares, kept for the saved column
    scores = np.full((n_hyp, n_spots), np.inf)
    valid = np.zeros((n_hyp, n_spots), dtype=bool)
    for k in range(n_hyp):
        calib_k = dict(calibration)
        calib_k["coefficients"] = scale_channel_blocks(
            calibration["coefficients"], ratios_norm[k]
        )
        params, chi_squares, converged, _n_it = _fit_spline_multistart(
            spots,
            calib_k,
            mle=mle,
            n_z_starts=n_z_starts,
            residuals=roi_residuals,
            jacobians=jacobians,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_gpu=use_gpu,
            variance=variance,
        )
        thetas.append(params)
        chis.append(np.asarray(chi_squares))
        finite = np.isfinite(params).all(axis=1) & np.isfinite(chi_squares)
        # LSE is robust here; for MLE keep only converged fits since its
        # chi-square is unreliable on the frequent negative-curvature exits.
        valid[k] = finite & (converged if mle else True)
        scores[k] = np.where(finite, chi_squares, np.inf)

    # Per spot: the best VALID hypothesis (lowest residual / chi2), falling back
    # to the best finite score if none was flagged valid.
    best_k = np.argmin(np.where(valid, scores, np.inf), axis=0)
    none_valid = ~valid.any(axis=0)
    if none_valid.any():
        best_k[none_valid] = np.argmin(scores[:, none_valid], axis=0)

    # Build localizations per winning-hypothesis group so z-conversion and CRLB
    # use that hypothesis's (scaled) calibration. Index-aligned column
    # assignment keeps per-channel photons correct across the internal
    # frame-sort of locs_from_fits_spline.
    em = camera_infos[0].get("Gain", 1) > 1
    parts = []
    for k in range(n_hyp):
        rows = np.where(best_k == k)[0]
        if len(rows) == 0:
            continue
        calib_k = dict(calibration)
        calib_k["coefficients"] = scale_channel_blocks(
            calibration["coefficients"], ratios_norm[k]
        )
        ids_k = identifications.iloc[rows]
        theta_k = np.asarray(thetas[k])[rows]
        locs_k = locs_from_fits_spline(
            ids_k,
            theta_k,
            box,
            em,
            calib_k,
            mle=mle,
            residuals=(None if roi_residuals is None else roi_residuals[rows]),
            jacobians=jacobians[rows],
            # The winning hypothesis's own score. Only for least squares:
            # under MLE this chi-square is a likelihood, and the frequent
            # negative-curvature exits make it unreliable anyway (see above).
            chi_square=(None if mle else chis[k][rows]),
            variance=(None if variance is None else variance[rows]),
        )
        amp = pd.Series(np.asarray(theta_k[:, 0]), index=ids_k.index)
        ps = _photon_scales(calib_k, n_channels)
        total = None
        for c in range(n_channels):
            pc = amp * float(ratios_norm[k, c]) * float(ps[c])
            locs_k[f"photons_ch{c}"] = pc.astype(np.float32)
            total = pc if total is None else total + pc
        locs_k["photons"] = total.astype(np.float32)
        locs_k["color"] = np.int32(k)
        parts.append(locs_k)

    locs = pd.concat(parts) if parts else pd.DataFrame()
    if len(locs):
        locs.sort_values(by="frame", kind="quicksort", inplace=True)
    return locs


def _split_fov_channel_affines(calibration: dict) -> list | None:
    """Region-local channel registration for a split-FOV calibration.

    Uses the stored ``channel_registration`` when present; otherwise derives it
    from the stored absolute ``channel_transforms`` and default ``regions`` so
    those calibrations can also be re-placed at fit time. Returns None if
    neither is available."""
    affines = calibration.get("channel_registration")
    if affines is not None:
        return [tform.from_dict(a) for a in affines]
    regions = calibration.get("regions")
    transforms = calibration.get("channel_transforms")
    if regions and transforms:
        return decompose_region_transforms(regions, transforms)
    return None


def split_fov_fit_geometry(
    calibration: dict, regions: list | None = None
) -> tuple[list, int, list]:
    """Where a split-FOV fit's channels sit, and how they map onto each other.

    The channels are placed at ``regions`` (e.g. the ROIs drawn for *this*
    data) when given, else at the calibration's own regions; the inter-channel
    affine is the same either way (see :func:`compose_region_transforms`).

    Parameters
    ----------
    calibration : dict
        Split-FOV spline calibration (``spline.calibrate_spline_split_fov``).
    regions : list, optional
        One ``[[y_min, x_min], [y_max, x_max]]`` per channel, reference first.

    Returns
    -------
    fit_regions : list
        Normalized channel rectangles actually in use.
    reference : int
        Index of the reference channel in ``fit_regions``/``transforms``.
    transforms : list
        One ``(2, 3)`` affine per channel mapping reference-channel
        coordinates into that channel, placed at ``fit_regions``.
    """
    if not calibration.get("split_fov"):
        raise ValueError(
            "A split-FOV fit requires a split-FOV calibration (built with "
            "spline.calibrate_spline_split_fov / a 'regions' argument)."
        )
    calib_regions = calibration.get("regions")
    if not calib_regions:
        raise ValueError("Split-FOV calibration is missing 'regions'.")
    n_channels = len(calib_regions)

    if regions is not None:
        if len(regions) != n_channels:
            raise ValueError(
                f"Got {len(regions)} regions but the calibration has "
                f"{n_channels} channels; draw one ROI per channel "
                "(reference first)."
            )
        fit_regions = [_normalize_rect(r) for r in regions]
        reference = 0  # fit-time ROIs are drawn reference-first
        affines = _split_fov_channel_affines(calibration)
        if affines is None:
            raise ValueError(
                "Calibration has no channel_registration to re-place at the "
                "given regions."
            )
        transforms = compose_region_transforms(fit_regions, affines)
    else:
        fit_regions = [_normalize_rect(r) for r in calib_regions]
        reference = int(calibration.get("reference", 0))
        transforms = [
            tform.from_dict(t) for t in calibration["channel_transforms"]
        ]
    if len(transforms) != n_channels:
        raise ValueError(
            f"Calibration has {n_channels} channels but {len(transforms)} "
            "channel transforms."
        )
    return fit_regions, reference, transforms


def confine_to_region(
    identifications: pd.DataFrame, region: list
) -> pd.DataFrame:
    """The detections inside one rectangle.

    Parameters
    ----------
    identifications : pd.DataFrame
        Detections with ``x`` and ``y`` columns.
    region : list
        A ``[[y_min, x_min], [y_max, x_max]]`` rectangle, in any corner order.

    Returns
    -------
    identifications : pd.DataFrame
        The rows inside it (upper bounds exclusive), re-indexed.
    """
    (y0, x0), (y1, x1) = _normalize_rect(region)
    x = np.asarray(identifications["x"], dtype=np.float64)
    y = np.asarray(identifications["y"], dtype=np.float64)
    inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return identifications.iloc[np.flatnonzero(inside)].reset_index(drop=True)


def filter_linked_identifications_split_fov(
    identifications: pd.DataFrame,
    calibration: dict,
    box: int,
    regions: list | None = None,
    tol: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Split-FOV counterpart of :func:`filter_linked_identifications`.

    A split-FOV movie is identified as a whole, so one table holds the
    detections of every channel. They are split by region and paired across
    regions exactly as separate channel movies are, keeping only the
    reference-region detections found in *all* other regions - the molecules
    the joint fit can describe.

    Parameters
    ----------
    identifications : pd.DataFrame
        Movie-wide detections (all regions), as identified in the GUI.
    calibration : dict
        Split-FOV spline calibration.
    box : int
        Box side length; sets the default pairing radius (``1.5 * box``).
    regions : list, optional
        Channel ROIs for this data (reference first); see
        :func:`split_fov_fit_geometry`. None (the default) uses the
        calibration's own.
    tol : float, optional
        Largest distance (camera pixels) at which detections in two regions
        count as the same molecule. None (the default) uses ``1.5 * box``.
    progress_callback : callable, optional
        Invoked with a monotone 0 -> ``n_total`` progression.

    Returns
    -------
    linked : pd.DataFrame
        The reference-region detections that link across all regions.
    n_kept, n_total : int
        Linked and total *reference-region* detections.
    """
    fit_regions, reference, transforms = split_fov_fit_geometry(
        calibration, regions
    )
    # reference first, as ``filter_linked_identifications`` expects; the
    # transforms already map reference coordinates into each region
    order = [reference] + [
        c for c in range(len(fit_regions)) if c != reference
    ]
    ids_per_region = [
        confine_to_region(identifications, fit_regions[c]) for c in order
    ]
    return filter_linked_identifications(
        ids_per_region,
        [transforms[c] for c in order],
        box,
        tol=tol,
        progress_callback=progress_callback,
    )


def fit_spline_split_fov(
    movie,
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    calibration: dict,
    regions: list | None = None,
    photon_ratios: lib.FloatArray2D | None = None,
    mle: bool = False,
    link_photons: bool = True,
    confine_to_reference: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    apply_roi_residuals: bool = True,
    n_z_starts: int | None = None,
    use_gpu: bool | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    camera_calibration: dict | None = None,
) -> pd.DataFrame:
    """Fit a split-FOV multichannel spline PSF from a *single* movie whose
    rectangular sub-regions are the channels.

    Global fit as in globLoc (Li et al., Nat. Commun. 13, 3133, 2022), for the
    single-camera split-FOV geometry.

    The calibration (built by :func:`picasso.spline.calibrate_spline_split_fov`)
    stores the *inter-channel* registration as region-local
    ``channel_registration``
    (see :func:`decompose_region_transforms`), independent of where the channels sit
    on the chip. This function repeats the one ``movie``/``camera_info`` once per
    channel and delegates to the standard multichannel fitters. The model is
    chosen exactly as in the GUI ``MultichannelSplineFitWorker``: ratiometric if
    photon ratios are present, otherwise the plain linked fit.

    Parameters
    ----------
    movie, camera_info
        The single loaded movie and its camera info dict.
    identifications : pd.DataFrame
        Detections; when ``confine_to_reference`` is True (default) they are
        filtered to the reference region so each molecule yields one spot that is
        mapped into the other regions via the transforms.
    regions : list, optional
        The channel ROIs *for this data* (one ``[[y_min, x_min], [y_max,
        x_max]]`` per channel, reference first), e.g. re-drawn in the GUI. When
        given, the absolute channel transforms are rebuilt at these positions via
        the stored region-local affines - so the same calibration can be applied
        to data whose split sits at a different position. When omitted, the
        calibration's own ``regions`` (the calibration-time positions) are used.
    box : int
        Box side length (camera pixels), must match the calibration.
    calibration : dict
        A split-FOV ``"spline-3d-multichannel"`` calibration.
    photon_ratios : lib.FloatArray2D, optional
        Candidate per-channel ratios for the ratiometric path (else taken from
        the calibration).
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator. Default False.
    link_photons : bool, optional
        As in :func:`fit_spline_multichannel`. Default True.
    confine_to_reference : bool, optional
        Filter ``identifications`` to the reference region first. Default
        True.
    progress_callback : callable, optional
        Invoked with the cumulative spot count as the fit proceeds.
    apply_roi_residuals : bool, optional
        As in :func:`fit_spline_multichannel`. Default True.
    n_z_starts : int, optional
        Number of axial seeds. None (the default) picks it from the
        calibration.
    use_gpu : bool or None, optional
        As in :func:`fit_spline_multichannel`.
    tolerance, max_iterations : optional
        Convergence schedule. None (the default) uses the fit's own.
    camera_calibration : dict or None, optional
        A per-pixel sCMOS camera calibration for the (single) camera. All
        split-FOV regions are read from one sensor and the maps are indexed by
        absolute frame coordinates, so the same full-frame calibration serves
        every region. Default None.

    Returns
    -------
    locs : pd.DataFrame
        The localizations, in the reference region's coordinates, as returned
        by whichever multichannel fitter was chosen.
    """
    fit_regions, reference, transforms = split_fov_fit_geometry(
        calibration, regions
    )
    n_channels = len(fit_regions)

    ids = identifications
    if confine_to_reference:
        ids = confine_to_region(ids, fit_regions[reference])

    # downstream fitters read channel_transforms off the calibration; hand them
    # the transforms placed at the regions actually in use
    calibration = dict(calibration)
    calibration["channel_transforms"] = list(transforms)

    movies = [movie] * n_channels
    camera_infos = [camera_info] * n_channels
    # Split-FOV is one physical camera, so every region reads the same maps;
    # _cut_map indexes them with absolute frame coordinates, which is exactly
    # what makes one full-frame calibration serve all regions.
    camera_calibrations = (
        None
        if camera_calibration is None
        else [camera_calibration] * n_channels
    )

    if (
        not link_photons
        and 2 <= n_channels <= precision._LINK_XYZ_MAX_CHANNELS
    ):
        return fit_spline_multichannel(
            movies,
            camera_infos,
            ids,
            box,
            calibration,
            mle=mle,
            link_photons=False,
            progress_callback=progress_callback,
            apply_roi_residuals=apply_roi_residuals,
            n_z_starts=n_z_starts,
            use_gpu=use_gpu,
            tolerance=tolerance,
            max_iterations=max_iterations,
            camera_calibrations=camera_calibrations,
        )
    if (
        photon_ratios is not None
        or calibration.get("photon_ratios") is not None
    ):
        return fit_spline_multichannel_ratiometric(
            movies,
            camera_infos,
            ids,
            box,
            calibration,
            photon_ratios=photon_ratios,
            mle=mle,
            progress_callback=progress_callback,
            apply_roi_residuals=apply_roi_residuals,
            n_z_starts=n_z_starts,
            use_gpu=use_gpu,
            tolerance=tolerance,
            max_iterations=max_iterations,
            camera_calibrations=camera_calibrations,
        )
    return fit_spline_multichannel(
        movies,
        camera_infos,
        ids,
        box,
        calibration,
        mle=mle,
        link_photons=link_photons,
        progress_callback=progress_callback,
        apply_roi_residuals=apply_roi_residuals,
        n_z_starts=n_z_starts,
        use_gpu=use_gpu,
        tolerance=tolerance,
        max_iterations=max_iterations,
        camera_calibrations=camera_calibrations,
    )


# ----------------------------------------------------------------------
# Independent (per channel / per region) fitting
#
# The counterpart of the joint fits above: every channel - a separate movie,
# or one region of a split field of view - is fitted entirely on its own,
# with its own fitting method and its own calibrations, and comes out as its
# own set of localizations. Nothing is shared, nothing is linked and no
# channel registration is needed; the channels are connected afterwards (e.g.
# by aligning them in Picasso: Render).
#
# The trade-off against the global fit is deliberate: a joint fit reaches a
# better precision by tying the channels to one position, but it needs a
# registration, keeps only the molecules detected in *every* channel and
# exists for two fitting models only. Fitting independently costs that
# precision and gives back every detection, every fitting method and a
# per-channel choice of PSF / z calibration.
# ----------------------------------------------------------------------


def _per_channel_arg(value: object, n_channels: int, name: str) -> list:
    """One value of ``value`` per channel.

    A list or tuple is used as-is (and must have one entry per channel);
    anything else - including a calibration ``dict`` - is one setting shared
    by every channel and is broadcast.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_channels:
            raise ValueError(
                f"{name} has {len(value)} entries but there are "
                f"{n_channels} channels; pass one per channel or a single "
                "value for all of them."
            )
        return list(value)
    return [value] * n_channels


def _channel_progress(
    progress_callback: Callable[[int, int], None] | Literal["console"] | None,
    channel: int,
) -> Callable[[int], None] | Literal["console"] | None:
    """One channel's progress callback: the caller's, told which channel is
    reporting. ``"console"`` is passed straight through, so each channel
    gets its own progress bar."""
    if progress_callback is None or progress_callback == "console":
        return progress_callback
    return partial(progress_callback, channel)


def _check_single_channel_calibration(
    spline_calibration: dict | None, channel: int
) -> None:
    """Reject a joint multichannel calibration in an independent fit.

    A ``spline-3d-multichannel`` calibration describes several channels at
    once and only means anything to the global fit; fitting one channel
    against it would silently use the reference channel's PSF everywhere.
    """
    if not isinstance(spline_calibration, dict):
        return
    if spline_calibration.get("model") == "spline-3d-multichannel":
        raise ValueError(
            f"Channel {channel} was given a 'spline-3d-multichannel' "
            "calibration, which describes all channels at once and belongs "
            "to the joint "
            "fit (fit_spline_multichannel / fit_spline_split_fov). Fitting "
            "the channels independently needs one single-channel spline "
            "calibration per channel."
        )


def region_label(index: int) -> str:
    """How a split-FOV region is named on screen and in its output files:
    the first region is the reference channel, the rest are numbered."""
    return "ref" if index == 0 else f"ch{index}"


def region_movie_info(info: list[dict], region: list) -> list[dict]:
    """Movie metadata describing one region of a split field of view.

    The region is a movie in its own right once its localizations have been
    moved into its own coordinates (see
    :func:`locs_to_region_coordinates`), so its width and height are the
    region's rather than the sensor's. Where it sat on the sensor is kept in
    ``"Region"``, so the original position is recoverable.

    Parameters
    ----------
    info : list of dict
        The movie's metadata, as loaded by ``picasso.io.load_movie``.
    region : list
        A ``[[y_min, x_min], [y_max, x_max]]`` rectangle, in any corner
        order.

    Returns
    -------
    info : list of dict
        A copy with the last entry's ``"Width"`` / ``"Height"`` set to the
        region size and ``"Region"`` recording the rectangle.
    """
    (y0, x0), (y1, x1) = _normalize_rect(region)
    new_info = [dict(_) for _ in info]
    new_info[-1] = new_info[-1] | {
        "Width": int(x1 - x0),
        "Height": int(y1 - y0),
        "Region": [[y0, x0], [y1, x1]],
    }
    return new_info


def locs_to_region_coordinates(
    locs: pd.DataFrame, region: list
) -> pd.DataFrame:
    """Move localizations into one region's own coordinate frame.

    Subtracts the region's top-left corner from ``x`` and ``y``, so that a
    region of a split field of view becomes a stand-alone channel starting at
    the origin - which is what makes the regions overlay each other when they
    are loaded side by side in Picasso: Render.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x`` and ``y`` columns, in sensor coordinates.
    region : list
        A ``[[y_min, x_min], [y_max, x_max]]`` rectangle, in any corner
        order.

    Returns
    -------
    locs : pd.DataFrame
        A copy with ``x`` and ``y`` shifted by the region's origin.
    """
    x0, y0 = _region_origin_xy(region)
    shifted = locs.copy()
    shifted["x"] = (np.asarray(locs["x"], dtype=np.float64) - x0).astype(
        np.float32
    )
    shifted["y"] = (np.asarray(locs["y"], dtype=np.float64) - y0).astype(
        np.float32
    )
    return shifted


def split_locs_by_region(
    locs: pd.DataFrame, regions: list, to_local: bool = True
) -> list[pd.DataFrame]:
    """Split localizations by the region of the sensor they fall in.

    The localization counterpart of :func:`confine_to_region`: a fit run over
    a whole split-FOV movie is separated into one table per region. Regions
    are matched in the order given, and a localization in none of them is
    dropped.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x`` and ``y`` columns, in sensor coordinates.
    regions : list
        One ``[[y_min, x_min], [y_max, x_max]]`` rectangle per region.
    to_local : bool, optional
        Move each region's localizations into its own coordinate frame
        (default True, see :func:`locs_to_region_coordinates`). False keeps
        them in sensor coordinates.

    Returns
    -------
    locs_per_region : list of pd.DataFrame
        One table per region, in the order the regions were given.
    """
    per_region = []
    for region in regions:
        region_locs = confine_to_region(locs, region)
        if to_local:
            region_locs = locs_to_region_coordinates(region_locs, region)
        per_region.append(region_locs)
    return per_region


def fit_independent(
    movies: list,
    camera_infos: list[dict] | dict,
    identifications: list[pd.DataFrame],
    box: int,
    fitting_method: str | list[str] = "gausslq",
    *,
    eps: float | list | None = None,
    max_it: int | list | None = None,
    spline_calibration: dict | list | None = None,
    camera_calibration: dict | list | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int, int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> list[tuple[pd.DataFrame, dict]] | None:
    """Fit several channels independently, one single-channel fit each.

    The alternative to the joint fits (:func:`fit_spline_multichannel`,
    :func:`fit_gauss_multichannel`): no registration is needed, every
    detection is fitted, and each channel may use a different fitting method
    and its own calibrations. The channels come out unlinked, in their own
    coordinates, to be connected afterwards.

    Parameters
    ----------
    movies : list
        One movie per channel, as loaded by ``picasso.io.load_movie``.
    camera_infos : list of dict or dict
        One camera-info dict per channel, or a single one shared by all.
    identifications : list of pd.DataFrame
        That channel's own detections, one table per channel (e.g. from
        :func:`identify` run on each movie).
    box : int
        Box side length (camera pixels), shared by the channels.
    fitting_method : str or list of str, optional
        Any of ``FIT_METHODS``, either one for all channels or one per
        channel. Default "gausslq".
    eps, max_it : float, int, list or None, optional
        Convergence criterion and iteration cap, per channel or shared. See
        :func:`fit`.
    spline_calibration : dict, list or None, optional
        A single-channel spline PSF calibration per channel, or one shared by
        all. A joint ``"spline-3d-multichannel"`` calibration is rejected -
        it belongs to :func:`fit_spline_multichannel`.
    camera_calibration : dict, list or None, optional
        Per-pixel sCMOS calibration per channel, or one shared by all. Each
        must match the shape of its own movie.
    multiprocess : bool, optional
        Whether to use multiprocessing. Default True.
    progress_callback : callable, "console" or None, optional
        Called with ``(channel, n_spots_done)`` as each channel is fitted;
        ``"console"`` shows one tqdm progress bar per channel.
    abort_callback : callable or None, optional
        Called with no arguments; returning True aborts. The whole batch then
        returns None.

    Returns
    -------
    results : list of (pd.DataFrame, dict) or None
        One ``(locs, info)`` pair per channel, in the order given. None if
        the fit was aborted.
    """
    n_channels = len(movies)
    if len(identifications) != n_channels:
        raise ValueError(
            f"{len(identifications)} identification tables for "
            f"{n_channels} movies; pass one per channel."
        )
    infos = _per_channel_arg(camera_infos, n_channels, "camera_infos")
    methods = _per_channel_arg(fitting_method, n_channels, "fitting_method")
    epss = _per_channel_arg(eps, n_channels, "eps")
    max_its = _per_channel_arg(max_it, n_channels, "max_it")
    splines = _per_channel_arg(
        spline_calibration, n_channels, "spline_calibration"
    )
    cameras = _per_channel_arg(
        camera_calibration, n_channels, "camera_calibration"
    )
    results = []
    for c in range(n_channels):
        _check_single_channel_calibration(splines[c], c)
        locs, info = fit(
            movies[c],
            camera_info=infos[c],
            identifications=identifications[c],
            box=box,
            fitting_method=methods[c],
            eps=epss[c],
            max_it=max_its[c],
            spline_calibration=splines[c],
            camera_calibration=cameras[c],
            multiprocess=multiprocess,
            progress_callback=_channel_progress(progress_callback, c),
            abort_callback=abort_callback,
        )
        if locs is None:  # aborted
            return None
        info = info | {
            "Fit mode": FIT_MODE_INDEPENDENT,
            "Channel": c,
            "Channels": n_channels,
        }
        results.append((locs, info))
    return results


def fit_split_fov_independent(
    movie,
    camera_info: dict,
    identifications: pd.DataFrame,
    box: int,
    regions: list,
    fitting_method: str | list[str] = "gausslq",
    *,
    movie_info: list[dict] | None = None,
    to_local: bool = True,
    eps: float | list | None = None,
    max_it: int | list | None = None,
    spline_calibration: dict | list | None = None,
    camera_calibration: dict | None = None,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int, int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> list[tuple[pd.DataFrame, list[dict]]] | None:
    """Fit each region of a split field of view independently.

    The split-FOV counterpart of :func:`fit_independent`: the regions of one
    movie are the channels, so each region's detections are fitted on their
    own - with that region's own fitting method and calibration - and come
    out as a stand-alone set of localizations in the region's own
    coordinates.

    Parameters
    ----------
    movie : LoadedMovie
        The movie holding all regions.
    camera_info : dict
        Camera information; one camera images every region.
    identifications : pd.DataFrame
        Detections over the whole sensor (e.g. from :func:`identify` with
        the regions as its ROIs); they are split by region here.
    box : int
        Box side length (camera pixels).
    regions : list
        One ``[[y_min, x_min], [y_max, x_max]]`` rectangle per region, the
        first being the reference region.
    fitting_method : str or list of str, optional
        Any of ``FIT_METHODS``, either one for all regions or one per
        region. Default "gausslq".
    movie_info : list of dict or None, optional
        The movie's metadata, as loaded by ``picasso.io.load_movie``. When
        given, it is adapted to each region (see :func:`region_movie_info`)
        and prepended to that region's fit metadata, so that what comes back
        is ready to save.
    to_local : bool, optional
        Move each region's localizations into its own coordinate frame
        (default True). False keeps them in sensor coordinates.
    eps, max_it, spline_calibration : optional
        Per region or shared, as in :func:`fit_independent`.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS calibration for the (single) camera, covering the
        full frame. Default None.
    multiprocess : bool, optional
        Whether to use multiprocessing. Default True.
    progress_callback : callable, "console" or None, optional
        Called with ``(region, n_spots_done)`` as each region is fitted;
        ``"console"`` shows one tqdm progress bar per region.
    abort_callback : callable or None, optional
        Called with no arguments; returning True aborts. The whole batch then
        returns None.

    Returns
    -------
    results : list of (pd.DataFrame, list of dict) or None
        One ``(locs, info)`` pair per region, in the order the regions were
        given. ``info`` ends with that region's fit metadata and, when
        ``movie_info`` was passed, starts with the region's movie metadata.
        None if the fit was aborted.
    """
    n_regions = len(regions)
    if not n_regions:
        raise ValueError("fit_split_fov_independent needs at least one region")
    methods = _per_channel_arg(fitting_method, n_regions, "fitting_method")
    epss = _per_channel_arg(eps, n_regions, "eps")
    max_its = _per_channel_arg(max_it, n_regions, "max_it")
    splines = _per_channel_arg(
        spline_calibration, n_regions, "spline_calibration"
    )
    results = []
    for c, region in enumerate(regions):
        _check_single_channel_calibration(splines[c], c)
        locs, fit_info = fit(
            movie,
            camera_info=camera_info,
            identifications=confine_to_region(identifications, region),
            box=box,
            fitting_method=methods[c],
            eps=epss[c],
            max_it=max_its[c],
            spline_calibration=splines[c],
            camera_calibration=camera_calibration,
            multiprocess=multiprocess,
            progress_callback=_channel_progress(progress_callback, c),
            abort_callback=abort_callback,
        )
        if locs is None:  # aborted
            return None
        if to_local:
            locs = locs_to_region_coordinates(locs, region)
        fit_info = fit_info | {
            "Fit mode": FIT_MODE_INDEPENDENT,
            "Channel": c,
            "Channels": n_regions,
            "Region": _normalize_rect(region),
            "Region coordinates": "region" if to_local else "sensor",
        }
        info = [fit_info]
        if movie_info is not None:
            info = region_movie_info(movie_info, region) + info
        results.append((locs, info))
    return results


def _fit2d_avg(
    spots: lib.FloatArray3D,
    identifications: pd.DataFrame,
    box: int,
    em: bool,
    multiprocess: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    variance: lib.FloatArray3D | None = None,
) -> pd.DataFrame | None:
    """Take localizations at the average value of the spots, see
    ``fit_2D`` for more details."""
    N = len(identifications)
    if multiprocess:
        fs = avgroi.fit_spots_parallel(spots, asynch=True)
        theta = _process_fitting_futures(
            fs, N, progress_callback, abort_callback
        )
        if theta is None:
            return
    else:
        theta = avgroi.fit_spots(spots, progress_callback)
    locs = avgroi.locs_from_fits(
        identifications,
        theta,
        box,
        em,
        readout_variance=_mean_readout_variance(variance),
    )
    return locs


def _process_fitting_futures(
    fs: list[Future],
    N: int,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> lib.FloatArray2D | None:
    """Convenience function for processing progress of fitting using
    multiprocessing. See ``_fit2d_gauss``, ``_fit2d_avg``."""
    n_tasks = len(fs)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(total=N, desc="Fitting", unit="spot")

    while lib.n_futures_done(fs) < n_tasks:
        # check for abort
        if callable(abort_callback) and abort_callback():
            for f in fs:
                f.cancel()
            if use_tqdm:
                iter_range.close()
            return

        # update progress
        n_finished = round(N * lib.n_futures_done(fs) / n_tasks)
        if use_tqdm:
            iter_range.update(n_finished - iter_range.n)
        elif callable(progress_callback):
            progress_callback(n_finished)
        time.sleep(0.2)
    if use_tqdm:
        iter_range.update(N - iter_range.n)
        iter_range.close()
    theta = avgroi.fits_from_futures(fs)
    return theta


def localize(
    movie: LoadedMovie,
    # TODO: remove in v0.12.0 - only movie may be passed positionally, and
    # camera_info / identification_parameters become keyword-only
    *args,
    camera_info: dict | None = None,
    identification_parameters: dict | None = None,
    parameters: dict | None = None,  # TODO: remove in v0.12.0 (renamed)
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    movie_info: list[dict] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
        "avg",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] | None = None,  # TODO: rm v0.12.0
    spline_calibration: dict | None = None,
    calibration_3d: dict | str | None = None,
    affine_calibration: dict | list | None = None,
    camera_calibration: dict | None = None,
    threaded: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    return_info: bool = True,  # TODO: remove in v0.12.0
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in a movie using the
    specified parameters.

    Fits in 2D, or in 3D with ``calibration_3d`` (astigmatism, see
    ``zfit``) or a 3D ``spline_calibration`` (z from the fit itself).

    Since v0.10.0: support for frame bounds and ROI for identification +
    all fitting methods.

    Since v0.11.0: astigmatic 3D fitting via ``calibration_3d``, which
    replaces the deprecated ``localize_3D``.

    Parameters
    ----------
    movie : LoadedMovie
        The input movie, as loaded by ``picasso.io.load_movie``.
    *args
        Deprecated positional form of ``camera_info`` and
        ``identification_parameters``, removed in v0.12.0. Each one warns when
        used; see ``_localize_legacy_arguments``.
    camera_info : dict
        A dictionary containing camera information such as
        `Baseline`, `Sensitivity`, and `Gain`.
    identification_parameters : dict
        A dictionary containing spot identification parameters,
        including:

        - `Min. Net Gradient`: Minimum net gradient for spot
          identification.
        - `Box Size`: Size of the box to cut out around each spot.
        - `Temporal Median Window`: optional, window length (in frames)
          of the temporal median filter applied before identification;
          0 or missing disables it. Fitting always uses the raw movie.
        - `Gaussian Filter Sigma`: optional, standard deviation (in
          camera pixels) of a spatial Gaussian filter applied to every
          frame before identification, see ``GaussianFilteredMovie``. It
          merges the several local maxima of a spot that is not
          Gaussian-shaped into one. Applied after the temporal median
          filter, if both are used. The filter applies to the
          identification only - the spots are always cut out of and
          fitted on the raw movie. Note that the minimum net gradient
          has to be re-tuned when this is changed, since smoothing
          lowers gradient magnitudes. 0 or missing disables it.
    parameters : dict, optional
        Deprecated alias for ``identification_parameters``, removed in
        v0.12.0.
    threaded : bool, optional
        Whether to use multithreading/multiprocessing. Default is True.
    movie_info : list[dict], optional
        Movie metadata. If None, an empty list is used. Default is None.
    roi : tuple or list of tuples, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). A list of such tuples restricts the
        identification to several (disjoint) regions, and the returned
        localizations then carry a ``roi_id`` column naming the region
        each of them came from (see :func:`add_roi_id`). If None, the
        entire frame is used. Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. Default is None.
    fitting_method : {"gausslq", "gausslq-spherical", "gausslq-rotated", \
            "gausslq-gpu", "gausslq-rotated-gpu", "gausslq-spherical-gpu", \
            "gaussmle", "gaussmle-spherical", "gaussmle-gpu", \
            "gaussmle-rotated-gpu", "gaussmle-spherical-gpu" or "avg"}, \
            optional
        Which 2D fitting algorithm to use, see ``fit``. Default is
        "gausslq".
    eps : float or None, optional
        The convergence criterion, honored by every iterating method on
        either device (all of them except "avg"). None (the default)
        picks the value that suits the method, see ``fit``.
    max_it : int or None, optional
        The maximum number of iterations per spot, as ``eps``. None (the
        default) picks the value that suits the method, see ``fit``.
    mle_method : Literal["sigma", "sigmaxy"] or None, optional
        Deprecated and ignored, removed in v0.12.0. Specify the
        fitting_method instead.
    spline_calibration : dict or None, optional
        Cubic-spline PSF calibration (see ``io.load_spline_calibration``),
        required for any "spline*" ``fitting_method`` and ignored otherwise.
        For a 3D spline calibration the resulting localizations contain the
        fitted ``z`` directly, so no separate z-fitting step is needed.
        Default is None.
    calibration_3d : dict, str or None, optional
        Astigmatism calibration for fitting z on top of the 2D fit,
        either an already loaded calibration dictionary or a path to a
        YAML file holding one, with the keys:

        - "X Coefficients": list of 7 floats, polynomial coefficients
          for the x-axis calibration curve;
        - "Y Coefficients": list of 7 floats, polynomial coefficients
          for the y-axis calibration curve;
        - "Magnification factor": float, magnification factor of the
          microscope, i.e., the ratio between the actual z position of
          the calibration sample and the estimated z position from the
          localization data.

        Ignored for the "spline*" fitting methods, which fit z
        themselves from ``spline_calibration``. Not supported for "avg",
        which fits no Gaussian widths, nor for the "*-spherical" methods,
        which constrain sx == sy and so carry no astigmatism. The
        "*-rotated" methods report sx and sy along the rotated principal
        axes, whereas the astigmatism calibration assumes the camera
        axes, so use them for z fitting with care. Default is None (2D
        localization).
    affine_calibration : dict, list, str or None, optional
        Lateral (x, y) corrections to apply after fitting, held outside
        the calibration used for fitting - e.g. a chromatic-aberration
        calibration kept in its own file, whether or not the fit is 3D.
        A calibration dictionary carrying a ``"Lateral transforms"``
        list, the list itself, or the path of a calibration file to read
        them from; several are applied in list order. With
        ``calibration_3d`` these run after the corrections that
        calibration carries, so keeping one separate gives the same
        coordinates as appending it to the 3D calibration. A correction
        ``calibration_3d`` or ``spline_calibration`` already carries is
        skipped (with a warning) rather than applied twice. Default is
        None.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration (see
        ``io.load_camera_calibration`` and ``scmos.calibrate_scmos``),
        forwarded to ``fit``. When given, its maps replace the scalar
        "Baseline" (and, if a gain map is present, "Sensitivity") of
        ``camera_info``. Default is None.
    identification_progress_callback : callable or "console" or None
        A callback for progress updates during identification. If
        "console", progress will be printed to the console. If None,
        progress is not reported. Default is None.
    fit_progress_callback : callable or "console" or None
        A callback for progress updates during fitting. If "console",
        progress will be printed to the console. If None, progress is
        not reported. Default is None.
    fit_z_progress_callback : callable or "console" or None
        As ``fit_progress_callback``, for the astigmatic z fitting.
        Ignored unless ``calibration_3d`` is given. Default is None.
    return_info : bool, optional
        Whether to return additional information about the fitting
        process. Default is True. If True, a tuple of (locs, info) is
        returned. In v0.12.0 return_info will be removed and the
        function will always return info.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots. With ``roi`` given, a
        ``roi_id`` column naming the ROI each of them came from is
        appended, see :func:`add_roi_id`.
    info : list[dict], optional
        A list of dictionaries containing metadata about the movie and
        the fitting process. Only returned if `return_info` is True.
    """
    if not return_info:
        # TODO: remove in v0.12.0
        lib.deprecation_warning(
            "In version 0.12, return_info argument will be removed such "
            "that picasso.localize.localize() will always return both "
            "the localizations and the metadata dictionary."
        )
    camera_info, identification_parameters = _localize_legacy_arguments(
        args, camera_info, identification_parameters, parameters, mle_method
    )
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert isinstance(
        identification_parameters, dict
    ), "identification_parameters must be a dict"
    fit_z = _validate_calibration_3d(calibration_3d, fitting_method)
    if fit_z and isinstance(calibration_3d, str):
        calibration_3d = io.load_calibration(calibration_3d)

    # Use empty list as default for movie_info
    if movie_info is None:
        movie_info = []

    # Identify spots
    identifications, identify_info = identify(
        movie,
        identification_parameters["Min. Net Gradient"],
        identification_parameters["Box Size"],
        roi=roi,
        frame_bounds=frame_bounds,
        threaded=threaded,
        temporal_median_window=identification_parameters.get(
            "Temporal Median Window", 0
        ),
        gaussian_filter_sigma=identification_parameters.get(
            "Gaussian Filter Sigma", None
        ),
        progress_callback=identification_progress_callback,
    )

    # Fit spots
    locs, fit_info = fit(
        movie=movie,
        camera_info=camera_info,
        identifications=identifications,
        box=identification_parameters["Box Size"],
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        spline_calibration=spline_calibration,
        camera_calibration=camera_calibration,
        multiprocess=threaded,
        progress_callback=fit_progress_callback,
    )
    # Before the z fit and the lateral corrections below: both move x/y,
    # which would blur the ROI a localization was detected in.
    locs = add_roi_id(locs, roi)
    info = io.strip_mm_metadata(movie_info) + [identify_info] + [fit_info]

    if fit_z:
        # Astigmatic z fitting on top of the 2D fit, see Huang, et al.
        # Science, 2008. zfit only knows gausslq/gaussmle; map the GPU /
        # rotated codes to the corresponding CPU noise model.
        locs, info = zfit.zfit(
            locs=locs,
            info=info,
            calibration=calibration_3d,
            fitting_method=(
                "gaussmle"
                if fitting_method.startswith("gaussmle")
                else "gausslq"
            ),
            filter=0,
            lateral_transforms=affine_calibration,
            multiprocess=threaded,
            progress_callback=fit_z_progress_callback,
        )
        # zfit applies both the calibration's own lateral corrections and
        # the separately loaded ones, in that order, and records them - so
        # a correction kept in its own file gives the same coordinates as
        # one appended to the 3D calibration.
        return _localize_return(locs, info, return_info=return_info)

    # Standalone affine corrections (e.g. a chromatic one used without a 3D
    # calibration); those carried by the spline calibration were already
    # applied by the fit, so they are dropped here rather than applied twice.
    extra, duplicates = lib.drop_duplicate_lateral_transforms(
        lib.resolve_lateral_transforms(affine_calibration), spline_calibration
    )
    _warn_duplicate_affine(duplicates)
    if extra:
        locs = lib.apply_lateral_transforms(locs, extra)
        fit_info["Lateral corrections applied"] = (
            lib.describe_lateral_transforms(extra)
        )
    return _localize_return(locs, info, return_info=return_info)


#: Camera keys ``fit`` reads off ``camera_info`` (see ``_to_photons`` /
#: ``_sensitivity``); the rest of a picasso info list is movie metadata.
#: Pulled out of the info list-of-dicts by ``localize_frames`` when no
#: ``camera_info`` is passed explicitly.
_CAMERA_INFO_KEYS = ("Baseline", "Sensitivity", "Gain", "Qe", "Pixelsize")


def _camera_info_from_info(info: list[dict] | None) -> dict:
    """Collect the camera parameters from a picasso info list-of-dicts.

    The streaming contract (S0B-2, contract 3) carries the camera info inside
    ``info`` rather than as a separate argument, so ``localize_frames`` pulls
    the keys ``fit`` needs (``Baseline``, ``Sensitivity``, ``Gain`` and,
    optionally, ``Qe``/``Pixelsize``) out of it. Later dicts win, so a
    fit/identify block appended to a raw-movie block can override it.
    """
    camera_info: dict = {}
    for entry in info or []:
        if isinstance(entry, dict):
            for key in _CAMERA_INFO_KEYS:
                if key in entry:
                    camera_info[key] = entry[key]
    return camera_info


class _InMemoryMovie(io.AbstractPicassoMovie):
    """Minimal ``AbstractPicassoMovie`` backed by an in-memory frame stack.

    ``fit`` asserts its movie is an ``AbstractPicassoMovie`` (or a memmap) so
    that filtered movie views cannot reach it; a raw in-memory stack is a
    legitimate thing to fit, so ``localize_frames`` wraps one in this thin
    adapter that just delegates to the underlying ndarray. Spots are cut with
    the same per-frame slicing as the memmap path (``_cut_spots_framebyframe``
    vs ``_cut_spots_numba``), so the localizations are identical.
    """

    def __init__(self, frames, info: list[dict] | None = None):
        super().__init__()
        self._frames = np.asarray(frames)
        if self._frames.ndim != 3:
            raise ValueError(
                "frames must be a 3D (n_frames, height, width) stack; got "
                f"shape {self._frames.shape!r}"
            )
        self._info = info or []
        self.shape = self._frames.shape

    @property
    def n_frames(self) -> int:
        # Derived so it cannot desync from the backing array; the localize
        # pipeline reads ``len(movie)`` and ``movie.shape``, but other movie
        # classes expose ``n_frames`` and callers may expect it too.
        return self._frames.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def info(self):
        return self._info[0] if self._info else {}

    def camera_parameters(self, config: dict) -> dict:
        return super().camera_parameters(config)

    def __getitem__(self, it):
        return self._frames[it]

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def get_frame(self, index: int):
        return self._frames[index]

    def tofile(self, file_handle, byte_order=None):
        self._frames.tofile(file_handle)

    @property
    def dtype(self):
        return self._frames.dtype


def localize_frames(
    frames,
    info: list[dict] | None,
    params: dict,
    *,
    start_frame: int = 0,
    camera_info: dict | None = None,
    fitting_method: str = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    spline_calibration: dict | None = None,
    camera_calibration: dict | None = None,
    threaded: bool = True,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> pd.DataFrame:
    """Localize an in-memory frame stack, GUI-free (streaming/live input).

    A thin wrapper around :func:`localize` (identify + fit) for batched or
    live acquisition: it takes an in-memory frame stack instead of a movie
    loaded from disk and assigns absolute frame indices, so successive batches
    concatenate into one growing localization table. It runs no new
    localization algorithm - the fit is exactly the one :func:`localize`
    (and Picasso: Localize) runs, so the result matches that path spot for
    spot on the same frames and parameters.

    Implements contract 3 of the S0B-2 shared data contracts.

    Parameters
    ----------
    frames : array-like
        In-memory frame stack, a 3D ``(n_frames, height, width)`` array (or
        anything :func:`numpy.asarray` turns into one). An already-loaded
        movie (a memmap or an ``io.AbstractPicassoMovie``) is used as-is.
    info : list of dict or None
        Picasso info list-of-dicts: movie metadata and, unless
        ``camera_info`` is passed, the camera parameters (``Baseline``,
        ``Sensitivity``, ``Gain`` and, optionally, ``Qe``/``Pixelsize``),
        which are read out of it (see :func:`_camera_info_from_info`).
    params : dict
        Identification parameters, at least ``"Min. Net Gradient"`` and
        ``"Box Size"``; ``"Temporal Median Window"`` and
        ``"Gaussian Filter Sigma"`` are honored if present, as in
        :func:`localize`.
    start_frame : int, optional
        Absolute index of the first frame in ``frames``. The returned
        ``frame`` column is offset by this, so a caller that increments it by
        each batch's frame count gets one table whose frame indices are
        absolute and contiguous across batch boundaries. Default is 0.
    camera_info : dict or None, optional
        Camera parameters, overriding those found in ``info``. Default None
        (read them from ``info``).
    fitting_method : str, optional
        Which fitting algorithm to use, see :func:`fit`. Default ``"gausslq"``
        (GPU variants such as ``"gausslq-gpu"`` run on the GPU if available).
    eps, max_it, spline_calibration, camera_calibration, threaded
        Forwarded to :func:`localize` unchanged.
    identification_progress_callback, fit_progress_callback
        Progress callbacks forwarded to :func:`localize`.

    Returns
    -------
    locs : pd.DataFrame
        The localization table, columns ``frame, x, y, photons, sx, sy, bg,
        lpx, lpy, net_gradient`` (plus ``z``/``lpz`` for a 3D spline fit), as
        :func:`localize` returns them, with ``frame`` shifted by
        ``start_frame``.
    """
    assert isinstance(params, dict), "params must be a dict"
    assert (
        isinstance(start_frame, (int, np.integer)) and start_frame >= 0
    ), "start_frame must be a non-negative integer"
    if camera_info is None:
        camera_info = _camera_info_from_info(info)
    else:
        # ``fit`` fills a missing "Pixelsize" into camera_info in place; copy
        # it so a caller reusing one dict across streaming batches is not
        # silently mutated.
        camera_info = dict(camera_info)

    if isinstance(frames, (io.AbstractPicassoMovie, np.memmap)):
        movie = frames
    else:
        movie = _InMemoryMovie(frames, info)

    locs, _ = localize(
        movie,
        camera_info=camera_info,
        identification_parameters=params,
        movie_info=info if info is not None else [],
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        spline_calibration=spline_calibration,
        camera_calibration=camera_calibration,
        threaded=threaded,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        return_info=True,
    )

    if start_frame:
        # Shift into absolute coordinates without changing the column dtype,
        # so a batch starting at 0 is byte-for-byte the non-streaming table.
        # ``frame`` is uint32, so the absolute index is assumed to stay below
        # 2**32 (~4.3e9 frames), which holds for any real acquisition.
        locs = locs.copy()
        locs["frame"] = (locs["frame"].to_numpy() + start_frame).astype(
            locs["frame"].dtype
        )
    return locs


# TODO: remove in v0.12.0 (return_info is removed, info is always returned)
def _localize_return(
    locs: pd.DataFrame,
    info: list[dict],
    return_info: bool,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    """``localize``'s return value, honoring the deprecated
    ``return_info``."""
    if return_info:
        return locs, info
    return locs


# TODO: remove in v0.12.0, together with the *args, ``parameters`` and
# ``mle_method`` arguments of ``localize`` that it resolves
def _localize_legacy_arguments(
    args: tuple,
    camera_info: dict | None,
    identification_parameters: dict | None,
    parameters: dict | None,
    mle_method: str | None,
) -> tuple[dict | None, dict | None]:
    """Map ``localize``'s pre-v0.11.0 calling conventions onto the current
    arguments, warning about each one, and return the resolved
    ``(camera_info, identification_parameters)``."""
    if args:
        lib.deprecation_warning(
            "In version 0.12, picasso.localize.localize() will only accept "
            "the movie as a positional argument; pass camera_info and "
            "identification_parameters as keyword arguments."
        )
        if len(args) > 2:
            raise TypeError(
                "localize() takes at most 3 positional arguments "
                f"(movie, camera_info, identification_parameters), "
                f"{len(args) + 1} given"
            )
        if camera_info is not None:
            raise TypeError("localize() got multiple values for camera_info")
        camera_info = args[0]
        if len(args) == 2:
            if identification_parameters is not None or parameters is not None:
                raise TypeError(
                    "localize() got multiple values for "
                    "identification_parameters"
                )
            identification_parameters = args[1]
    if parameters is not None:
        lib.deprecation_warning(
            "The parameters argument of picasso.localize.localize() was "
            "renamed to identification_parameters and will be removed in "
            "version 0.12."
        )
        if identification_parameters is None:
            identification_parameters = parameters
    if mle_method is not None:
        lib.deprecation_warning(
            "The mle_method argument of picasso.localize.localize() is "
            "ignored and will be removed in version 0.12."
        )
    return camera_info, identification_parameters


def _validate_calibration_3d(
    calibration_3d: dict | str | None,
    fitting_method: str,
) -> bool:
    """Whether an astigmatic z fit is to be run after the 2D fit, i.e.
    ``calibration_3d`` was given and the fitting method supports it."""
    if calibration_3d is None:
        return False
    assert isinstance(
        calibration_3d, (dict, str)
    ), "calibration_3d must be a dict or a path to a YAML file"
    if fitting_method.startswith("spline"):
        # The spline PSF fit recovers z itself, from spline_calibration.
        warnings.warn(
            "Ignoring calibration_3d: the spline PSF fit recovers z itself, "
            "using spline_calibration.",
            stacklevel=3,
        )
        return False
    assert fitting_method != "avg", (
        "astigmatic z fitting (calibration_3d) requires fitted Gaussian "
        "widths, which 'avg' does not provide"
    )
    assert "spherical" not in fitting_method, (
        "astigmatic z fitting (calibration_3d) is not possible with the "
        "spherical Gaussian methods, which constrain sx == sy and thus "
        "carry no astigmatism"
    )
    return True


# TODO: remove in v0.12.0 - superseded by localize(calibration_3d=...)
def localize_3D(
    movie: LoadedMovie,
    *,
    movie_info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    calibration_3d: dict,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    affine_calibration: dict | list | None = None,
    camera_calibration: dict | None = None,
    multiprocess: bool = True,
    temporal_median_window: int | None = None,
    gaussian_filter_sigma: float | None = None,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Localize (i.e., identify and fit) spots in 3D in a movie using
    the specified parameters.

    .. deprecated:: 0.11.0
        Use ``picasso.localize.localize`` with its ``calibration_3d``
        argument instead - the two functions differ only in the astigmatic
        z fitting. ``localize_3D`` will be removed in v0.12.0.

    For the Gaussian ``fitting_method`` values this first runs 2D
    localizations, followed by z position fitting assuming astigmatism, see
    Huang, et al. Science, 2008 (``calibration_3d`` holds the astigmatism
    polynomials). For ``"spline-gpu"`` a cubic-spline PSF fit recovers z
    directly in the 2D fit, so no separate z-fitting step is run and
    ``spline_calibration`` is used instead of ``calibration_3d``.

    Parameters
    ----------
    movie : LoadedMovie
        The input movie, read frame by frame.
    movie_info : list of dicts
        Movie metadata.
    camera_info : dict
        A dictionary containing camera information: "Baseline",
        "Sensitivity", "Gain" and "Pixelsize".
    box : int
        Size of the box to cut out around each spot. Should be an odd
        integer.
    minimum_ng : float
        Minimum net gradient for spot identification.
    calibration_3d : path or dict
        Either a path to a YAML file containing the calibration data or
        an already loaded calibration dictionary containing the
        following keys:

        - "X Coefficients": list of 7 floats, polynomial coefficients
            for the x-axis calibration curve;
        - "Y Coefficients": list of 7 floats, polynomial coefficients
            for the y-axis calibration curve;
        - "Magnification factor": float, magnification factor of the
            microscope, i.e., the ratio between the actual z position of
            the calibration sample and the estimated z position from the
            localization data.
    roi : tuple, optional
        Region of interest (ROI) defined as a tuple of two tuples,
        where the first tuple contains the start coordinates
        (y_start, x_start) and the second tuple contains the end
        coordinates (y_end, x_end). If None, the entire frame is used.
        Default is None.
    frame_bounds : tuple, optional
        Minimum and maximum frame numbers to consider for the
        identification. If None, all frames are used. If only min or max
        is to be specified, the other is to be set to None, for example,
        ``(5, None)`` sets minimum frame to 5 without maximum frame.
        Default is None.
    fitting_method : {"gausslq", "gausslq-spherical", "gausslq-rotated", \
            "gausslq-gpu", "gausslq-rotated-gpu", "gausslq-spherical-gpu", \
            "gaussmle", "gaussmle-spherical", "gaussmle-gpu", \
            "gaussmle-rotated-gpu" or "gaussmle-spherical-gpu"}, optional
        Which 2D fitting algorithm to use, see ``fit``. "avg" is not
        supported since z fitting requires the fitted Gaussian sigmas.
        Note that the rotated elliptical Gaussian methods report sx and
        sy along the rotated principal axes, whereas the astigmatism
        calibration assumes the camera axes, so use them for z fitting
        with care. The spherical Gaussian methods constrain sx == sy, so
        they carry no astigmatism and are unsuitable for z fitting.
        Default is "gausslq".
    eps : float, optional
        The convergence criterion for CPU MLE fitting. Ignored for
        other methods (GPU fitting uses its own convergence
        settings). Default is 0.001.
    max_it : int, optional
        The maximum number of iterations for CPU MLE fitting. Ignored
        for other methods. Default is 100.
    mle_method : Literal["sigma", "sigmaxy"], optional
        The method used for CPU MLE fitting (impose same sigma in x and
        y or not, respectively). Default is "sigmaxy".
    spline_calibration : dict or None, optional
        Cubic-spline PSF calibration (see ``io.load_spline_calibration``),
        required for the "spline*" fitting methods and ignored otherwise. A 3D
        spline calibration recovers z in the fit itself, so
        ``calibration_3d`` is then unused. Default is None.
    affine_calibration : dict or list or None, optional
        Lateral (x, y) affine corrections to apply on top of those the
        fit's own calibration carries, e.g. a standalone
        chromatic-aberration calibration combined with an astigmatism
        transform stored in ``calibration_3d``. Applied last, in list
        order. Default is None.
    camera_calibration : dict or None, optional
        Per-pixel sCMOS camera calibration (see
        ``io.load_camera_calibration`` and ``scmos.calibrate_scmos``),
        forwarded to ``fit``. When given, its maps replace the scalar
        "Baseline" (and, if a gain map is present, "Sensitivity") of
        ``camera_info``. Default is None.
    multiprocess : bool, optional
        Whether or not to use multiprocessing. Ignored for GPU fitting.
        Default is True.
    temporal_median_window : int or None, optional
        If given (and non-zero), a temporal median background is
        subtracted from every frame before identifying, using a window of
        this many frames, see ``TemporalMedianMovie``. The filter applies
        to the identification only - the spots are always cut out of and
        fitted on the raw movie. Note that ``minimum_ng`` has to be
        re-tuned when this is switched on or off, since subtracting a
        background changes the scale of the net gradient. Default is None
        (no filtering).
    gaussian_filter_sigma : float or None, optional
        Standard deviation (in camera pixels) of a spatial Gaussian
        filter applied to every frame before identifying, see
        ``GaussianFilteredMovie``. It merges the several local maxima of
        a spot that is not Gaussian-shaped into one. Applied after the
        temporal median filter, if both are used. The filter applies to
        the identification only - the spots are always cut out of and
        fitted on the raw movie. Note that ``minimum_ng`` has to be
        re-tuned when this is changed, since smoothing lowers gradient
        magnitudes. Default is None (no filtering).
    identification_progress_callback : callable, "console" or None, optional
        Progress of the identification, called with the number of movie
        frames processed. "console" displays a tqdm bar; None does not track
        progress. Default is None.
    fit_progress_callback : callable, "console" or None, optional
        As ``identification_progress_callback``, for the 2D fit, called with
        the number of spots fitted. Default is None.
    fit_z_progress_callback : callable, "console" or None, optional
        As ``fit_progress_callback``, for the astigmatic z fitting. Unused
        when a 3D ``spline_calibration`` recovers z in the fit itself.
        Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Data frame containing the localized spots in 3D.
    info : list[dict]
        A list of dictionaries containing metadata about the movie and
        the fitting processes.
    """
    lib.deprecation_warning(
        "picasso.localize.localize_3D is deprecated and will be removed in "
        "version 0.12; use picasso.localize.localize with its calibration_3d "
        "argument instead."
    )
    assert isinstance(
        movie, (np.ndarray, io.ND2Movie)
    ), "movie must be a numpy array or ND2Movie"
    assert isinstance(movie_info, list), "movie_info must be a list"
    assert isinstance(camera_info, dict), "camera_info must be a dict"
    assert (
        isinstance(box, int) and box > 0 and box % 2 == 1
    ), "box must be a positive odd integer"
    assert isinstance(
        minimum_ng, (int, float, list, tuple, np.ndarray)
    ), "minimum_ng must be a number or one number per ROI"
    assert fitting_method in [
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
    ], (
        "fitting_method must be one of 'gausslq', 'gausslq-spherical',"
        " 'gausslq-rotated', 'gausslq-gpu', 'gausslq-rotated-gpu',"
        " 'gausslq-spherical-gpu', 'gaussmle', 'gaussmle-spherical',"
        " 'gaussmle-gpu', 'gaussmle-rotated-gpu', 'gaussmle-spherical-gpu',"
        " 'spline-gpu', or 'spline-mle-gpu'"
    )
    if fitting_method.startswith("spline"):
        # The spline PSF fit recovers z itself; it uses a spline calibration
        # instead of the astigmatism polynomials in calibration_3d.
        assert isinstance(spline_calibration, dict), (
            "spline_calibration (a spline PSF calibration dict, see "
            "io.load_spline_calibration) is required for spline 3D "
            "localization"
        )
    else:
        assert isinstance(
            calibration_3d, (dict, str)
        ), "calibration_3d must be a dict or a path to a YAML file"
    assert eps is None or (
        isinstance(eps, (int, float)) and eps > 0
    ), "eps must be a positive number or None"
    assert max_it is None or (
        isinstance(max_it, int) and max_it > 0
    ), "max_it must be a positive integer or None"
    assert mle_method in [
        "sigma",
        "sigmaxy",
    ], "mle_method must be 'sigma' or 'sigmaxy'"
    assert isinstance(multiprocess, bool), "multiprocess must be a boolean"
    return _localize_3D(
        movie=movie,
        movie_info=movie_info,
        camera_info=camera_info,
        box=box,
        minimum_ng=minimum_ng,
        calibration_3d=calibration_3d,
        roi=roi,
        frame_bounds=frame_bounds,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        mle_method=mle_method,
        spline_calibration=spline_calibration,
        affine_calibration=affine_calibration,
        camera_calibration=camera_calibration,
        multiprocess=multiprocess,
        temporal_median_window=temporal_median_window,
        gaussian_filter_sigma=gaussian_filter_sigma,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        fit_z_progress_callback=fit_z_progress_callback,
    )


def _localize_3D(
    movie: LoadedMovie,
    *,
    movie_info: list[dict],
    camera_info: dict,
    box: int,
    minimum_ng: float,
    calibration_3d: dict,
    roi: tuple[tuple[int, int], tuple[int, int]] | None = None,
    frame_bounds: tuple[int, int] | None = None,
    fitting_method: Literal[
        "gausslq",
        "gausslq-spherical",
        "gausslq-rotated",
        "gausslq-gpu",
        "gausslq-rotated-gpu",
        "gausslq-spherical-gpu",
        "gaussmle",
        "gaussmle-spherical",
        "gaussmle-gpu",
        "gaussmle-rotated-gpu",
        "gaussmle-spherical-gpu",
        "spline",
        "spline-mle",
        "spline-gpu",
        "spline-mle-gpu",
    ] = "gausslq",
    eps: float | None = None,
    max_it: int | None = None,
    mle_method: Literal["sigma", "sigmaxy"] = "sigmaxy",
    spline_calibration: dict | None = None,
    affine_calibration: dict | list | None = None,
    multiprocess: bool = True,
    temporal_median_window: int | None = None,
    gaussian_filter_sigma: float | None = None,
    identification_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    fit_z_progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    camera_calibration: dict | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Internal function for `localize_3D`, assumes validated inputs.

    A thin wrapper around ``localize``, which does the astigmatic z
    fitting itself since v0.11.0. ``mle_method`` is not passed on: it has
    no effect on the fit and warns in ``localize``.
    """
    return localize(
        movie=movie,
        camera_info=camera_info,
        identification_parameters={
            "Min. Net Gradient": minimum_ng,
            "Box Size": box,
            "Temporal Median Window": temporal_median_window,
            "Gaussian Filter Sigma": gaussian_filter_sigma,
        },
        roi=roi,
        frame_bounds=frame_bounds,
        movie_info=movie_info,
        fitting_method=fitting_method,
        eps=eps,
        max_it=max_it,
        spline_calibration=spline_calibration,
        # The spline fit recovers z itself, from spline_calibration; the
        # astigmatism polynomials only apply to the Gaussian methods.
        calibration_3d=(
            None if fitting_method.startswith("spline") else calibration_3d
        ),
        affine_calibration=affine_calibration,
        camera_calibration=camera_calibration,
        threaded=multiprocess,
        identification_progress_callback=identification_progress_callback,
        fit_progress_callback=fit_progress_callback,
        fit_z_progress_callback=fit_z_progress_callback,
    )


def _warn_duplicate_affine(duplicates: list) -> None:
    """Warn that corrections the fit's own calibration already applies were
    dropped, so they are not applied twice."""
    if duplicates:
        warnings.warn(
            "Skipping "
            + ", ".join(lib.describe_lateral_transforms(duplicates))
            + ": the calibration used for fitting already carries this "
            "affine correction and applies it itself. Applying it again "
            "would correct the coordinates twice.",
            lib.DuplicateLateralTransformWarning,
            stacklevel=3,
        )


def _apply_extra_affine(
    locs: pd.DataFrame,
    info: list[dict],
    affine_calibration: dict | list | None,
    applied: dict | list | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply affine corrections that are not carried by the fit's own
    calibration (e.g. a standalone chromatic one) and record them in the
    metadata. Corrections ``applied`` already covers are dropped instead of
    applied a second time. A no-op when there are none."""
    extra, duplicates = lib.drop_duplicate_lateral_transforms(
        lib.resolve_lateral_transforms(affine_calibration), applied
    )
    _warn_duplicate_affine(duplicates)
    if not extra:
        return locs, info
    locs = lib.apply_lateral_transforms(locs, extra)
    info = info + [
        {
            "Generated by": f"Picasso: v{__version__} Affine correction",
            "Lateral corrections applied": lib.describe_lateral_transforms(
                extra
            ),
        }
    ]
    return locs, info


def check_nena(
    locs: pd.DataFrame,
    info: None,
    callback: Callable[[int], None] = None,
) -> float:
    """Calculate the NeNA (experimental localization precision) from
    localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : None
        Not used.
    callback : Callable[[int], None], optional
        A callback function that can be used to report progress. It
        should accept an integer argument representing the current
        step or frame number. Default is None.

    Returns
    -------
    nena_px : float
        The NeNA value in pixels, representing the experimental
        localization precision.
    """
    print("Calculating NeNA.. ", end="")
    locs = locs[0:MAX_LOCS]
    try:
        result, nena_px = postprocess.nena(locs, info, callback=callback)
    except Exception as e:
        print(e)
        nena_px = float("nan")
    print(f"{nena_px:.2f} px.")
    return nena_px


def check_kinetics(locs: pd.DataFrame, info: list[dict]) -> float:
    """Calculate the mean length of binding events from localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list of dicts
        A list of dictionaries containing metadata about the movie.

    Returns
    -------
    len_mean : float
        The mean length of binding events in frames.
    """
    print("Linking.. ", end="")
    locs = locs.iloc[0:MAX_LOCS]
    locs = postprocess.link(locs, info=info)
    len_mean = locs.len.mean()
    print(f"Mean length {len_mean:.2f} frames.")
    return len_mean


def check_drift(
    locs: pd.DataFrame,
    info: list[dict],
    callback: Callable[[int], None] = None,
) -> tuple[float, float]:
    """Estimate the drift of localizations in x and y directions.

    Parameters
    ----------
    locs : pd.DataFrame
        Data frame containing the localized spots.
    info : list[dict]
        A list of dictionaries containing metadata about the movie.
    callback : Callable[[int], None], optional
        A callback function that can be used to report progress. It
        should accept an integer argument representing the current
        step or frame number. Default is None.

    Returns
    -------
    drift_x : float
        The estimated drift in the x direction.
    drift_y : float
        The estimated drift in the y direction.
    """
    steps = int(len(locs) // (MAX_LOCS))
    steps = max(1, steps)
    locs = locs[::steps]

    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)
    segmentation = max(1, int(n_frames // 10))

    print(f"Estimating drift with segmentation {segmentation}")
    drift, locs = postprocess.undrift(
        locs,
        info,
        segmentation,
        display=False,
        rcc_callback=callback,
    )
    drift_x = float(drift["x"].mean())
    drift_y = float(drift["y"].mean())

    print(f"Drift is X: {drift_x:.2f}, Y: {drift_y:.2f}.")

    return (drift_x, drift_y)


def get_file_summary(
    file: str,
    file_hdf: str,
    drift: tuple[float, float] | None = None,
    len_mean: float | None = None,
    nena: float | None = None,
) -> dict:
    """Generate a summary of the localization file, including metadata
    and statistics about the localizations.

    Parameters
    ----------
    file : str
        The path to the localization file (HDF5 format).
    file_hdf : str
        The path to the HDF5 file containing localizations.
    drift : tuple[float, float] | None, optional
        A tuple containing the drift in x and y directions. If None,
        the drift will be calculated from the localizations.
    len_mean : float | None, optional
        The mean length of binding events in frames. If None, it will
        be calculated from the localizations.
    nena : float | None, optional
        The NeNA value in pixels. If None, it will be calculated from
        the localizations.

    Returns
    -------
    summary : dict
        A dictionary containing the summary of the localization file,
        including metadata and statistics about the localizations.
    """
    if file_hdf is None:
        base, ext = os.path.splitext(file)
        file_hdf = base + "_locs.hdf5"

    locs, info = io.load_locs(file_hdf)

    summary = {}

    for col in MEAN_COLS:
        try:
            summary[col + "_mean"] = locs[col].mean()
            summary[col + "_std"] = locs[col].std()
        except KeyError:
            summary[col + "_mean"] = float("nan")
            summary[col + "_std"] = float("nan")

    for col in SET_COLS:
        col_ = col.lower()
        for inf in info:
            if col in inf:
                summary[col_] = inf[col]

    for col in SET_COLS:
        col_ = col.lower()
        if col_ not in summary:
            summary[col_] = float("nan")

    nena_px = check_nena(locs, info) if nena is None else nena
    len_mean = check_kinetics(locs, info) if len_mean is None else len_mean
    drift_x, drift_y = check_drift(locs, info) if drift is None else drift

    summary["len_mean"] = len_mean
    summary["n_locs"] = len(locs)
    summary["locs_frame"] = len(locs) / summary["frames"]
    summary["drift_x"] = drift_x
    summary["drift_y"] = drift_y
    summary["nena_px"] = nena_px
    summary["nena_nm"] = nena_px * summary["pixelsize"]
    summary["filename"] = os.path.normpath(file)
    summary["filename_hdf"] = file_hdf
    summary["file_created"] = datetime.fromtimestamp(os.path.getmtime(file))
    summary["entry_created"] = datetime.now()
    return summary


def _db_filename() -> str:  # TODO: remove in 1.0
    """Old alias for `db_filename."""
    return db_filename()


def db_filename() -> str:
    """Return the path to the SQLite database file used for storing
    localization summaries. The database is stored in the user's home
    directory under the ``.picasso`` folder."""
    home = os.path.expanduser("~")
    picasso_dir = os.path.join(home, ".picasso")
    os.makedirs(picasso_dir, exist_ok=True)
    return os.path.abspath(os.path.join(picasso_dir, "app_0410.db"))


def _save_file_summary(summary: dict) -> None:
    """Save the summary of a localization file to a SQLite database."""
    engine = create_engine("sqlite:///" + db_filename(), echo=False)
    s = pd.Series(summary, index=summary.keys()).to_frame().T
    s.to_sql("files", con=engine, if_exists="append", index=False)


def add_file_to_db(
    file: str,
    file_hdf: str,
    drift: tuple[float, float] | None = None,
    len_mean: float | None = None,
    nena: float | None = None,
) -> None:
    """Add a localization file summary to the SQLite database.

    Parameters
    ----------
    file, file_hdf, drift, len_mean, nena
        As in :func:`get_file_summary`, which builds the summary that is
        appended to the ``files`` table of the database (see ``db_filename``).
    """
    summary = get_file_summary(file, file_hdf, drift, len_mean, nena)
    _save_file_summary(summary)


def _movie_to_image(movie) -> np.ndarray:
    """Collapse a picasso movie to a single 2D float32 image on the
    original intensity (raw-count) scale. Multi-frame movies are
    averaged; single-frame movies are passed through. Keeping the raw
    scale means the net gradient computed during bead detection is
    comparable to the "Min. Net Gradient" used for normal localization.
    Frames are read one-at-a-time so the lazy-loading movie classes in
    ``picasso.io`` don't have to materialise the full stack at once."""
    n = len(movie)
    if n == 0:
        raise ValueError("Movie has zero frames.")
    if n == 1:
        return np.asarray(movie[0], dtype=np.float32)
    acc = np.zeros(np.asarray(movie[0]).shape, dtype=np.float64)
    for i in range(n):
        acc += np.asarray(movie[i], dtype=np.float64)
    return (acc / n).astype(np.float32)


def _lateral_detect_beads(
    image: np.ndarray, box: int, minimum_ng: float
) -> np.ndarray:
    """Detect bead candidates using the standard spot identification
    (local maxima above a minimum net gradient).

    Parameters
    ----------
    image : np.ndarray
        2D image to detect beads in.
    box : int
        Box size used by ``identify_in_image`` (also sets the minimum
        distance between two detected beads). Should be an odd integer.
    minimum_ng : float
        Minimum net gradient for a local maximum to be kept.

    Returns
    -------
    np.ndarray
        (N, 2) array of [row, col] integer coordinates.
    """
    y, x, _ = identify_in_image(image, minimum_ng, box)
    return np.column_stack((y, x))


def _lateral_refine_bead_positions(
    image: np.ndarray, coarse: np.ndarray, box: int
) -> np.ndarray:
    """Refine coarse bead positions to sub-pixel accuracy using the
    standard 2D Gaussian least-squares spot fitting.

    Parameters
    ----------
    image : np.ndarray
        2D image the beads were detected in.
    coarse : np.ndarray
        (N, 2) array of integer [row, col] bead positions.
    box : int
        Box size cut out around each bead for fitting. Should match the
        detection box so every spot lies fully inside the image.

    Returns
    -------
    np.ndarray
        (M, 2) array of refined [row, col] coordinates (M <= N; spots
        whose fit did not converge to a finite position are dropped).
    """
    if len(coarse) == 0:
        return np.empty((0, 2))
    ids = pd.DataFrame(
        {
            "frame": np.zeros(len(coarse), dtype=np.int32),
            "x": coarse[:, 1].astype(np.int32),
            "y": coarse[:, 0].astype(np.int32),
            "net_gradient": np.ones(len(coarse), dtype=np.float32),
        }
    )
    # Treat the single image as a one-frame movie; an identity camera
    # leaves the pixel values unchanged (the fit only needs relative
    # intensities to localize each bead).
    camera_info = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1}
    spots = get_spots(image[np.newaxis], ids, box, camera_info)
    theta = fit_spots_gauss(spots.astype(np.float32))
    locs = locs_from_fits_gauss(ids, theta, box, em=False)
    refined = np.column_stack((locs["y"].to_numpy(), locs["x"].to_numpy()))
    return refined[np.isfinite(refined).all(axis=1)]


def _lateral_match_bead_pairs(
    coords_ref: np.ndarray,
    coords_mov: np.ndarray,
    return_indices: bool = False,
) -> tuple:
    """Match beads via mutual nearest-neighbor with a distance threshold.
    Returns (pairs_ref, pairs_mov), each (M, 2).

    With ``return_indices``, the indices the pairs have in ``coords_ref``
    and ``coords_mov`` are returned as well, so a caller can tell which
    detections stayed unmatched - the Localize viewer grays those out when
    it draws the pairing (see ``Window.draw_affine_pairing``)."""
    if len(coords_ref) == 0 or len(coords_mov) == 0:
        empty = np.empty((0, 2))
        if return_indices:
            idx = np.empty(0, dtype=int)
            return empty, empty, idx, idx
        return empty, empty
    D = cdist(coords_ref, coords_mov)
    nn_r2m = np.argmin(D, axis=1)
    nn_m2r = np.argmin(D, axis=0)
    pairs_r, pairs_m, idx_r, idx_m = [], [], [], []
    for i, j in enumerate(nn_r2m):
        if D[i, j] < _LATERAL_MATCH_MAX_DIST_PX and nn_m2r[j] == i:
            pairs_r.append(coords_ref[i])
            pairs_m.append(coords_mov[j])
            idx_r.append(i)
            idx_m.append(j)
    pairs_ref = np.array(pairs_r) if pairs_r else np.empty((0, 2))
    pairs_mov = np.array(pairs_m) if pairs_m else np.empty((0, 2))
    if return_indices:
        return (
            pairs_ref,
            pairs_mov,
            np.asarray(idx_r, dtype=int),
            np.asarray(idx_m, dtype=int),
        )
    return pairs_ref, pairs_mov


# A bead pair whose residual exceeds this many times the median (but at
# least this many pixels) is dropped before the final fit.
_LATERAL_TRIM_FACTOR = 3.0
_LATERAL_TRIM_FLOOR_PX = 2.0


def _estimate_lateral_transform(
    src: np.ndarray,
    dst: np.ndarray,
    model: str = "affine",
) -> tuple[tform.Transform, np.ndarray]:
    """Fit the ``src -> dst`` lateral correction from ``[row, col]``
    correspondences, rejecting outliers.

    Returns ``(transform, keep)``, where ``keep`` is the boolean mask of the
    pairs the returned transform was fitted on.

    The pairs come from mutual-nearest-neighbor matching, which has no
    outlier rejection of its own: a single mismatched bead barely perturbs a
    6-DOF affine but visibly warps an 8-DOF homography and wrecks a degree-3
    polynomial. So the transform is fitted once, the pairs whose residual is
    far from the median are dropped, and it is refitted - the same trim the
    signal re-registration in :mod:`picasso.spline` uses.
    """
    src_xy = np.asarray(src, dtype=np.float64)[:, ::-1]
    dst_xy = np.asarray(dst, dtype=np.float64)[:, ::-1]
    transform = tform.estimate(src_xy, dst_xy, model)
    keep = np.ones(len(src_xy), dtype=bool)
    distance = np.hypot(*(dst_xy - transform.apply(src_xy)).T)
    cutoff = max(
        _LATERAL_TRIM_FLOOR_PX,
        _LATERAL_TRIM_FACTOR * float(np.median(distance)),
    )
    trimmed = distance <= cutoff
    if trimmed.sum() >= tform.min_points(model) and not trimmed.all():
        keep = trimmed
        transform = tform.estimate(src_xy[keep], dst_xy[keep], model)
    return transform, keep


def _lateral_plot_alignment(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    img_cor: np.ndarray,
    pairs_ref: np.ndarray,
    decomp: dict,
    n_pairs: int,
    pixelsize: float | None = None,
    save_path: str = "",
    ref_path: str = "",
    target_path: str = "",
    transform_type: str = "astigmatism",
) -> None:
    """Four-panel QC figure: overlay before/after correction and mean
    per-bead cross-correlation before/after correction.

    If ``pixelsize`` is None, axes are labeled in pixels; otherwise
    they are scaled to nm and labeled accordingly.
    """
    nm = pixelsize if pixelsize is not None else 1.0
    unit = "nm" if pixelsize is not None else "px"

    def norm(img):
        mn, mx = img.min(), img.max()
        return (img - mn) / (mx - mn + 1e-12)

    ref_n = norm(img_ref)
    mov_n = norm(img_mov)
    cor_n = norm(img_cor)

    crop_r = _AFFINE_XCORR_HALF_WIDTH

    def bead_xcorr_mean(frame_a, coords_a, frame_b, coords_b):
        acc, count = None, 0
        ny, nx = frame_a.shape
        for (ry_a, rx_a), (ry_b, rx_b) in zip(
            coords_a.astype(int), coords_b.astype(int)
        ):
            ya0 = max(0, ry_a - crop_r)
            ya1 = min(ny, ry_a + crop_r)
            xa0 = max(0, rx_a - crop_r)
            xa1 = min(nx, rx_a + crop_r)
            yb0 = max(0, ry_b - crop_r)
            yb1 = min(ny, ry_b + crop_r)
            xb0 = max(0, rx_b - crop_r)
            xb1 = min(nx, rx_b + crop_r)
            pa = frame_a[ya0:ya1, xa0:xa1]
            pb = frame_b[yb0:yb1, xb0:xb1]
            if pa.shape[0] < 4 or pb.shape[0] < 4:
                continue

            def prep(x):
                x = x - x.mean()
                s = x.std()
                return x / (s + 1e-12)

            cc = fftconvolve(prep(pa), prep(pb[::-1, ::-1]), mode="full")
            cc -= cc.min()
            if acc is None:
                acc = np.zeros_like(cc)
            if cc.shape == acc.shape:
                acc += cc
                count += 1
        if count == 0 or acc is None:
            s = 4 * crop_r - 1
            return np.zeros((s, s)), 0
        return acc / count, count

    def peak_nm(cc):
        py, px = np.unravel_index(np.argmax(cc), cc.shape)
        cy_cc, cx_cc = cc.shape[0] // 2, cc.shape[1] // 2
        r = 5
        y0 = max(0, py - r)
        y1 = min(cc.shape[0], py + r + 1)
        x0 = max(0, px - r)
        x1 = min(cc.shape[1], px + r + 1)
        patch = cc[y0:y1, x0:x1]
        nyp, nxp = patch.shape
        yg, xg = np.mgrid[0:nyp, 0:nxp].astype(float)

        def g2d(xy, x0, y0, sx, sy, amp, bg):
            x, y = xy
            return bg + amp * np.exp(
                -((x - x0) ** 2 / (2 * sx**2) + (y - y0) ** 2 / (2 * sy**2))
            )

        try:
            popt, _ = curve_fit(
                g2d,
                (xg.ravel(), yg.ravel()),
                patch.ravel(),
                p0=[
                    nxp / 2,
                    nyp / 2,
                    2.0,
                    2.0,
                    patch.max() - patch.min(),
                    patch.min(),
                ],
                maxfev=400,
            )
            sub_px = x0 + popt[0] - cx_cc
            sub_py = y0 + popt[1] - cy_cc
        except Exception:
            sub_px = float(px - cx_cc)
            sub_py = float(py - cy_cc)
        return sub_py * nm, sub_px * nm

    cc_raw, n_raw = bead_xcorr_mean(ref_n, pairs_ref, mov_n, pairs_ref)
    cc_cor, n_cor = bead_xcorr_mean(ref_n, pairs_ref, cor_n, pairs_ref)

    dy_raw, dx_raw = peak_nm(cc_raw) if n_raw > 0 else (0.0, 0.0)
    dy_cor, dx_cor = peak_nm(cc_cor) if n_cor > 0 else (0.0, 0.0)
    off_raw = np.hypot(dy_raw, dx_raw)
    off_cor = np.hypot(dy_cor, dx_cor)

    fig = plt.figure(figsize=(11, 11))
    if pixelsize is not None:
        trans_str = f"Tx={decomp['tx_nm']:.1f} nm  Ty={decomp['ty_nm']:.1f} nm"
    else:
        trans_str = f"Tx={decomp['tx_px']:.3f} px  Ty={decomp['ty_px']:.3f} px"
    title = (
        f"Alignment check  |  {n_pairs} bead pairs  |  "
        f"Scale X={decomp['scale_x']:.5f}  Y={decomp['scale_y']:.5f}  "
        f"Rot={decomp['rotation_deg']:.4f}°  " + trans_str
    )
    title = f"{transform_type.capitalize()}  |  " + title
    if ref_path or target_path:
        title += (
            f"\nref: {os.path.basename(ref_path)}   "
            f"target: {os.path.basename(target_path)}"
        )
    fig.suptitle(title, fontsize=10, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.25)

    ext = [0, img_ref.shape[1] * nm, img_ref.shape[0] * nm, 0]

    ax = fig.add_subplot(gs[0])
    ax.imshow(
        np.clip(np.stack([mov_n, ref_n, mov_n], axis=-1), 0, 1), extent=ext
    )
    ax.set_title(
        "Overlay  BEFORE\nRef (green) | Target (magenta)",
        fontsize=9,
        fontweight="bold",
    )
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    ax.imshow(
        np.clip(np.stack([cor_n, ref_n, cor_n], axis=-1), 0, 1), extent=ext
    )
    ax.set_title(
        "Overlay  AFTER\nRef (green) | Corrected (magenta)",
        fontsize=9,
        fontweight="bold",
    )
    ax.axis("off")

    lag = (np.arange(cc_raw.shape[1]) - cc_raw.shape[1] // 2) * nm
    e_cc = [lag[0], lag[-1], lag[-1], lag[0]]
    vmax = max(cc_raw.max(), cc_cor.max(), 1e-12)

    ax = fig.add_subplot(gs[2])
    im = ax.imshow(
        cc_raw,
        cmap="hot",
        extent=e_cc,
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.grid(False)
    ax.axhline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"Mean bead cross-corr  BEFORE\n"
        f"peak ({dx_raw:.1f}, {dy_raw:.1f}) {unit}  "
        f"|offset| = {off_raw:.1f} {unit}",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel(f"Δx ({unit})")
    ax.set_ylabel(f"Δy ({unit})")

    ax = fig.add_subplot(gs[3])
    im2 = ax.imshow(
        cc_cor,
        cmap="hot",
        extent=e_cc,
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
    )
    ax.grid(False)
    ax.axhline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0, color="cyan", lw=1.0, ls="--", alpha=0.7)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(
        f"Mean bead cross-corr  AFTER\n"
        f"peak ({dx_cor:.1f}, {dy_cor:.1f}) {unit}  "
        f"|offset| = {off_cor:.1f} {unit}",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel(f"Δx ({unit})")
    ax.set_ylabel(f"Δy ({unit})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def fit_lateral_transform(
    movie_ref,
    movie_target,
    calibration: dict,
    box: int,
    minimum_ng: float,
    pixelsize: float | None = None,
    transform_type: str = "astigmatism",
    ref_path: str = "",
    target_path: str = "",
    model: str = "affine",
) -> tuple[dict, dict]:
    """Fit the target -> reference transform and append it to
    ``calibration``'s ordered list of affine corrections.

    This is the computational half of :func:`calibrate_lateral_transform`.
    It touches no matplotlib state, so it is safe to call from a worker
    thread; the returned ``qc`` dict carries everything
    :func:`plot_lateral_calibration` needs to draw the diagnostic figure
    afterwards (on the GUI thread, where matplotlib must be driven from).

    Parameters
    ----------
    movie_ref, movie_target : AbstractPicassoMovie
        As in :func:`calibrate_lateral_transform`.
    calibration : dict
        As in :func:`calibrate_lateral_transform`.
    box, minimum_ng, pixelsize : int, float and float
        As in :func:`calibrate_lateral_transform`.
    transform_type, ref_path, target_path, model
        As in :func:`calibrate_lateral_transform`. ``plot_path`` is the only
        argument of that function not accepted here.

    Returns
    -------
    calibration : dict
        The input calibration, with the transform appended to its
        ``"Affine transforms"`` list (an existing entry of the same type is
        replaced). Save it with ``io.save_any_calibration``.
    qc : dict
        Inputs for :func:`plot_lateral_calibration`: the reference, target
        and corrected images, the matched reference bead positions, the
        decomposition, the number of pairs, the pixel size, the transform
        type and the source paths.

    Raises
    ------
    ValueError
        If ``transform_type`` is unknown, if ``calibration`` is a
        multichannel spline calibration (affine corrections are
        single-channel only), or if fewer bead pairs match than ``model``
        needs (see ``picasso.transforms.min_points``).
    """
    if transform_type not in lib.LATERAL_TRANSFORM_TYPES:
        raise ValueError(
            f"Unknown lateral transform type '{transform_type}'; expected "
            f"one of {lib.LATERAL_TRANSFORM_TYPES}."
        )
    if calibration.get("model") in (
        "spline-3d-multichannel",
        precision._LINK_XYZ_MODEL,
    ):
        raise ValueError(
            "Lateral corrections apply to single-channel data only, but "
            f"this is a '{calibration['model']}' calibration, which "
            "registers its channels itself. Append the transform to a "
            "single-channel calibration, or save it as a standalone "
            "lateral calibration."
        )
    img_ref = _movie_to_image(movie_ref)
    img_target = _movie_to_image(movie_target)

    coarse_ref = _lateral_detect_beads(img_ref, box, minimum_ng)
    coarse_target = _lateral_detect_beads(img_target, box, minimum_ng)
    refined_ref = _lateral_refine_bead_positions(img_ref, coarse_ref, box)
    refined_target = _lateral_refine_bead_positions(
        img_target, coarse_target, box
    )
    pairs_ref, pairs_target, idx_ref, idx_target = _lateral_match_bead_pairs(
        refined_ref, refined_target, return_indices=True
    )

    needed = tform.min_points(model)
    if len(pairs_ref) < needed:
        raise ValueError(
            f"Only {len(pairs_ref)} matched bead pair(s) — a "
            f"{model} transform needs at "
            f"least {needed}. Check the input images / detection parameters, "
            "or choose a simpler model."
        )

    transform, keep = _estimate_lateral_transform(
        pairs_target, pairs_ref, model
    )
    decomp = transform.decompose(pixelsize)

    # What the two images are called depends on what is being corrected;
    # the fit and the way the transform is applied are identical.
    source = (
        "cylindrical" if transform_type == "astigmatism" else "target channel"
    )
    lateral_entry = {
        "Type": transform_type,
        "Transform": transform.to_dict(),
        "Direction": f"{source} -> reference (x = col, y = row)",
        "Reference image": ref_path or "N/A",
        "Target image": target_path or "N/A",
        "Bead pairs": int(keep.sum()),
        "Bead pairs rejected": int(len(keep) - keep.sum()),
        "Decomposition": decomp,
    }
    if pixelsize is not None:
        lateral_entry["Pixelsize (nm)"] = float(pixelsize)
    lib.append_lateral_transform(calibration, lateral_entry)

    qc = {
        "img_ref": img_ref,
        "img_target": img_target,
        # the warp is part of the fit's output, not of the drawing, so it
        # is computed here and only displayed by the plotting function
        "img_cor": tform.warp_image(img_target, transform.inverse()),
        "pairs_ref": pairs_ref,
        # Every detection in each image plus the indices of the matched
        # ones (pair k is (idx_ref[k], idx_target[k])), so the Localize
        # viewer can draw the pairing as color-coded identification boxes.
        "beads_ref": refined_ref,
        "beads_target": refined_target,
        "idx_ref": idx_ref,
        "idx_target": idx_target,
        "box": int(box),
        "decomposition": decomp,
        "n_pairs": int(len(pairs_ref)),
        "pixelsize": pixelsize,
        "transform_type": transform_type,
        "ref_path": ref_path,
        "target_path": target_path,
    }
    return calibration, qc


def plot_lateral_calibration(qc: dict, save_path: str = "") -> None:
    """Draw the affine-calibration diagnostic figure from the ``qc`` dict
    returned by :func:`fit_lateral_transform`.

    Kept separate from the fit so a GUI can run the fit in a worker thread
    and still draw from the main thread.

    Parameters
    ----------
    qc : dict
        The second return value of :func:`fit_lateral_transform`.
    save_path : str, optional
        If given, the figure is written there. It is always shown
        interactively. Default is "".
    """
    _lateral_plot_alignment(
        qc["img_ref"],
        qc["img_target"],
        qc["img_cor"],
        qc["pairs_ref"],
        qc["decomposition"],
        n_pairs=qc["n_pairs"],
        pixelsize=qc["pixelsize"],
        save_path=save_path,
        ref_path=qc.get("ref_path", ""),
        target_path=qc.get("target_path", ""),
        transform_type=qc.get("transform_type", "astigmatism"),
    )


def calibrate_lateral_transform(
    movie_ref,
    movie_target,
    calibration: dict,
    box: int,
    minimum_ng: float,
    pixelsize: float | None = None,
    transform_type: str = "astigmatism",
    ref_path: str = "",
    target_path: str = "",
    model: str = "affine",
    plot_path: str = "",
) -> dict:
    """Fit a transform that maps a bead image into a reference frame and
    append it to any calibration dict.

    The same calibration serves two corrections, selected by
    ``transform_type``:

    - ``"astigmatism"``: the cylindrical-lens image is mapped into the
      reference (no-lens) frame, undoing the lateral distortion the
      cylindrical lens introduces.
    - ``"chromatic"``: one color channel is mapped into the reference
      color channel, correcting chromatic aberration.

    Both are stored as entries of an ordered ``"Affine transforms"`` list
    and applied to ``x``/``y`` in that order after fitting, so a 3D
    two-color experiment can chain the astigmatism correction and the
    chromatic one. The list is read from whatever calibration the fit uses
    - Gaussian astigmatism (YAML) or cubic-spline PSF (HDF5) - and an empty
    ``calibration`` dict starts a standalone affine calibration file, which
    is what a purely 2D chromatic correction needs.

    This is a **single-channel** correction: it maps one movie into a
    reference frame. A multichannel spline calibration registers its
    channels itself, so appending a transform to one raises ``ValueError``.

    The fit is performed in pixel coordinates on a per-pixel mean of
    each movie. Bead candidates are found by Gaussian-blur + local-max,
    refined to sub-pixel accuracy by a 2D Gaussian fit, then matched
    between the two images by mutual nearest neighbor. The transform is
    fitted by least squares (see :func:`picasso.transforms.estimate`) and
    decomposed into rotation / anisotropic scale / shear.

    Parameters
    ----------
    movie_ref, movie_target : AbstractPicassoMovie
        In-focus bead movies of the reference and of the frame to be
        corrected: without / with the cylindrical lens for
        ``"astigmatism"``, reference / other color channel for
        ``"chromatic"``. If a movie has multiple frames they are
        averaged; a single-frame movie is used as-is.
    calibration : dict
        Calibration the transform is appended to; may be a Gaussian
        astigmatism calibration, a single-channel spline PSF calibration,
        or ``{}`` to start a standalone affine calibration. An existing
        entry of the same ``transform_type`` is replaced. A multichannel
        spline calibration is rejected (see above).
    box : int
        Box size used to identify bead candidates (also sets the minimum
        distance between two detected beads). Should be an odd integer.
    minimum_ng : float
        Minimum net gradient for a bead candidate to be kept.
    pixelsize : float, optional
        Camera pixel size in nm. If given, decomposition translations
        and the diagnostic plot are converted from pixels to nm. If
        None (default), values are reported in pixels. Default is None.
    transform_type : {"astigmatism", "chromatic"}, optional
        What the transform corrects; recorded in the entry and used to
        decide which existing entry it replaces. Default is
        "astigmatism".
    ref_path, target_path : str, optional
        Paths to the source images, recorded in the calibration for
        traceability and shown in the diagnostic plot title. Default
        is "".
    model : str, optional
        The transform model, one of ``picasso.transforms.MODELS``. A more
        flexible model needs more matched bead pairs and extrapolates worse
        outside the field they cover. Default is "affine".
    plot_path : str, optional
        If given, the diagnostic figure is saved to this path. The
        figure is always shown interactively. Default is "".

    Returns
    -------
    calibration : dict
        The input calibration with the transform appended to its
        ``"Affine transforms"`` list. Use ``io.save_any_calibration`` to
        save the result to YAML or HDF5, whichever the calibration is.

    Notes
    -----
    Fit and figure are also available separately as
    :func:`fit_lateral_transform` and :func:`plot_lateral_calibration`, for
    callers that must not touch matplotlib from the thread doing the fit.
    """
    calibration, qc = fit_lateral_transform(
        movie_ref,
        movie_target,
        calibration,
        box=box,
        minimum_ng=minimum_ng,
        pixelsize=pixelsize,
        transform_type=transform_type,
        ref_path=ref_path,
        target_path=target_path,
        model=model,
    )
    plot_lateral_calibration(qc, save_path=plot_path)
    return calibration
