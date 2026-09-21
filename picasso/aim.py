"""
picasso.aim
~~~~~~~~~~~

Picasso implementation of Adaptive Intersection Maximization (AIM)
for fast undrifting in 2D and 3D.

Adapted from: Ma, H., et al. Science Advances. 2024.

:authors: Hongqiang Ma, Maomao Chen, Phuong Nguyen, Yang Liu,
    Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Literal

import numba
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

from . import lib, __version__


@numba.njit(cache=True, nogil=True)
def _count_intersections(
    l0_coords: lib.IntArray1D,
    l0_counts: lib.IntArray1D,
    l1_coords: lib.IntArray1D,
    l1_counts: lib.IntArray1D,
    shifts: lib.IntArray1D | lib.FloatArray1D,
) -> lib.IntArray1D:
    """Count the number of intersected localizations between the
    reference and the target dataset for every shift at once.

    We assume the intersection distance is 1 and, since the coordinates
    are expressed in units of intersection distance, two coordinates
    intersect only if they are exactly equal. Coordinates are encoded as
    1D integers (e.g. ``x + y * width``).

    ``l0_coords`` must be sorted and unique (as returned by
    ``np.unique``). Each shift is a constant offset added to the sorted,
    unique target coordinates, so each shifted coordinate is located in
    the reference with a binary search. The whole search region is
    evaluated in a single allocation-free pass (no ``(M, S)`` temporary
    is materialized), which is why this is a numba kernel rather than a
    vectorized numpy expression.

    Parameters
    ----------
    l0_coords : lib.IntArray1D
        Sorted, unique coordinates of the reference localizations, shape
        ``(L,)``.
    l0_counts : lib.IntArray1D
        Counts of the unique reference coordinates, shape ``(L,)``.
    l1_coords : lib.IntArray1D
        Sorted, unique coordinates of the target localizations, shape
        ``(M,)``.
    l1_counts : lib.IntArray1D
        Counts of the unique target coordinates, shape ``(M,)``.
    shifts : lib.IntArray1D | lib.FloatArray1D
        The ``S`` offsets spanning the local search region.

    Returns
    -------
    roi_cc : lib.IntArray1D
        Number of intersections for each of the ``S`` shifts, shape
        ``(S,)``.
    """
    n_ref = l0_coords.size
    n_shifts = shifts.size
    roi_cc = np.zeros(n_shifts, dtype=np.int64)
    for j in range(l1_coords.size):
        base = l1_coords[j]
        cj = l1_counts[j]
        for s in range(n_shifts):
            key = base + shifts[s]
            # leftmost binary search for key in the sorted reference
            lo = 0
            hi = n_ref
            while lo < hi:
                mid = (lo + hi) // 2
                if l0_coords[mid] < key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < n_ref and l0_coords[lo] == key:
                # for a match, add min(reference count, target count)
                c0 = l0_counts[lo]
                roi_cc[s] += c0 if c0 < cj else cj
    return roi_cc


@numba.njit(cache=True, nogil=True)
def _count_intersections_box(
    l0_coords: lib.IntArray1D,
    l0_counts: lib.IntArray1D,
    l1_coords: lib.IntArray1D,
    l1_counts: lib.IntArray1D,
    shifts: lib.IntArray1D,
    box: int,
) -> lib.IntArray1D:
    """Count intersections across a 2D ``box``-by-``box`` search region,
    exploiting the contiguity of each row of shifts.

    The shifts span a 2D box and are encoded into 1D as
    ``shift = dx + dy * width_units`` (see ``intersection_max``), laid
    out row-major as ``shifts[i * box + j]`` with ``dx`` indexed by ``i``
    and ``dy`` by ``j``. For a fixed ``dy`` (a single row), the ``box``
    encoded shifts are ``box`` consecutive integers — incrementing ``dx``
    by one increments the (truncated) encoded shift by exactly one,
    because ``width_units`` is always far larger than the search radius,
    so no row wraps. Therefore, for every target coordinate, the whole
    row is located with a single binary search into the sorted reference
    plus a short forward scan, instead of one binary search per shift.
    This reduces the work from ``M * box**2 * log L`` to roughly
    ``M * box * (log L + box)``.

    ``l0_coords`` must be sorted and unique. Parameters mirror
    ``_count_intersections``; ``box`` is the side length of the search
    region, so ``shifts.size == box * box``.

    Returns
    -------
    roi_cc : lib.IntArray1D
        Number of intersections for each of the ``box * box`` shifts,
        flattened row-major as ``roi_cc[i * box + j]``.
    """
    n_ref = l0_coords.size
    roi_cc = np.zeros(shifts.size, dtype=np.int64)
    for m in range(l1_coords.size):
        base = l1_coords[m]
        cm = l1_counts[m]
        for j in range(box):  # one dy-row at a time
            # the row's encoded shifts are the consecutive integers
            # run_start .. run_start + box - 1 (i = 0 .. box - 1)
            run_start = base + shifts[j]
            run_end = run_start + (box - 1)
            # leftmost binary search for run_start in the reference
            lo = 0
            hi = n_ref
            while lo < hi:
                mid = (lo + hi) // 2
                if l0_coords[mid] < run_start:
                    lo = mid + 1
                else:
                    hi = mid
            # scan the contiguous run, mapping each hit back to its shift
            p = lo
            while p < n_ref and l0_coords[p] <= run_end:
                i = l0_coords[p] - run_start  # dx index, 0 .. box - 1
                c0 = l0_counts[p]
                roi_cc[i * box + j] += c0 if c0 < cm else cm
                p += 1
    return roi_cc


def _run_intersections(
    l0_coords: lib.IntArray1D,
    l0_counts: lib.IntArray1D,
    l1_coords: lib.IntArray1D,
    l1_counts: lib.IntArray1D,
    shifts: lib.IntArray1D,
    box: int,
) -> lib.IntArray2D | lib.IntArray1D:
    """Run intersection counting across the whole local search region.

    Parameters
    ----------
    l0_coords : lib.IntArray1D
        Sorted, unique coordinates of the reference localizations.
    l0_counts : lib.IntArray1D
        Counts of the reference localizations.
    l1_coords : lib.IntArray1D
        Sorted, unique coordinates of the target localizations.
    l1_counts : lib.IntArray1D
        Counts of the target localizations.
    shifts : lib.IntArray1D
        1D array with the shifts spanning the local search region.
    box : int
        Side length of the local search region. ``box == 1`` signals z
        intersections (1D search region) for 3D undrifting.

    Returns
    -------
    roi_cc : lib.IntArray2D | lib.IntArray1D
        2D array with the number of intersections across the local
        search region. 1D array for z intersections in 3D undrifting.
    """
    if box == 1:  # z intersections only (1D search region), for z undrift
        return _count_intersections(
            l0_coords, l0_counts, l1_coords, l1_counts, shifts
        )
    # 2D search region: exploit the row-wise contiguity of the shifts
    roi_cc = _count_intersections_box(
        l0_coords, l0_counts, l1_coords, l1_counts, shifts, box
    )
    return roi_cc.reshape(box, box)


def _point_intersect_2d(
    l0_coords: lib.IntArray1D,
    l0_counts: lib.IntArray1D,
    x1: lib.SeriesOrFloatArray1D,
    y1: lib.SeriesOrFloatArray1D,
    intersect_d: float,
    width_units: float,
    shifts_xy: lib.IntArray1D,
    box: int,
) -> lib.IntArray2D:
    """Convert target coordinates into a 1D array in units of
    ``intersect_d`` and count the number of intersections in the local
    search region.

    Parameters
    ----------
    l0_coords : lib.IntArray1D
        Unique values of the reference localizations.
    l0_counts : lib.IntArray1D
        Counts of the unique values of reference localizations.
    x1, y1 : lib.SeriesOrFloatArray1D
        x and y coordinates of the target (currently undrifted) localizations.
    intersect_d : float
        Intersect distance in camera pixels.
    width_units : float
        Width of the camera image in units of intersect_d.
    shifts_xy : lib.IntArray1D
        1D array with x and y shifts.
    box : int
        Final side length of the local search region.

    Returns
    -------
    roi_cc : lib.IntArray2D
        2D array with numbers of intersections in the local search
        region.
    """
    # convert target coordinates to a 1D array in intersect_d units
    x1_units = np.round(x1 / intersect_d)
    y1_units = np.round(y1 / intersect_d)
    l1 = np.int32(x1_units + y1_units * width_units)  # 1d list
    # get unique values and counts of the target localizations
    l1_coords, l1_counts = np.unique(l1, return_counts=True)
    # run the intersections counting
    roi_cc = _run_intersections(
        l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box
    )
    return roi_cc


def _point_intersect_3d(
    l0_coords: lib.IntArray1D,
    l0_counts: lib.IntArray1D,
    x1: lib.SeriesOrFloatArray1D,
    y1: lib.SeriesOrFloatArray1D,
    z1: lib.SeriesOrFloatArray1D,
    intersect_d: float,
    width_units: float,
    height_units: float,
    shifts_z: lib.IntArray1D,
) -> lib.IntArray1D:
    """Convert target coordinates into a 1D array in units of
    ``intersect_d`` and count the number of intersections in the local
    search region.

    Parameters
    ----------
    l0_coords : lib.IntArray1D
        Unique values of the reference localizations.
    l0_counts : lib.IntArray1D
        Counts of the unique values of reference localizations.
    x1, y1, z1 : lib.SeriesOrFloatArray1D
        x, y, and z coordinates of the target (currently undrifted)
        localizations.
    intersect_d : float
        Intersect distance in camera pixels.
    width_units : float
        Width of the camera image in units of intersect_d.
    height_units : float
        Height of the camera image in units of intersect_d.
    shifts_z : lib.IntArray1D
        1D array with z shifts.

    Returns
    -------
    roi_cc : lib.IntArray1D
        1D array with numbers of intersections in the local search
        region.
    """
    # convert target coordinates to a 1D array in intersect_d units
    x1_units = np.round(x1 / intersect_d)
    y1_units = np.round(y1 / intersect_d)
    z1_units = np.round(z1 / intersect_d)
    l1 = np.int32(
        x1_units
        + y1_units * width_units
        + z1_units * width_units * height_units
    )  # 1d list
    # get unique values and counts of the target localizations
    l1_coords, l1_counts = np.unique(l1, return_counts=True)
    # run the intersections counting
    roi_cc = _run_intersections(
        l0_coords, l0_counts, l1_coords, l1_counts, shifts_z, 1
    )
    return roi_cc


def _get_fft_peak(
    roi_cc: lib.IntArray2D, intersect_d: float
) -> tuple[float, float]:
    """Estimate the precise sub-pixel position of the peak of ``roi_cc``
    with FFT.

    The phase of the first Fourier coefficient gives the peak position in
    units of ``roi_cc`` bins. Adjacent bins are one integer shift of the
    local search region apart, i.e. ``intersect_d`` camera pixels, so the
    peak is scaled by ``intersect_d`` to return camera pixels.

    Parameters
    ----------
    roi_cc : lib.IntArray2D
        2D array with numbers of intersections in the local search region.
    intersect_d : float
        Intersect distance in camera pixels, i.e. the spacing between
        adjacent bins of ``roi_cc``.

    Returns
    -------
    px, py : float
        Estimated x and y coordinates of the peak, in camera pixels.
    """
    fft_values = np.fft.fft2(roi_cc.T)
    ang_x = np.angle(fft_values[0, 1])
    ang_x = ang_x - 2 * np.pi * (ang_x > 0)  # normalize
    px = (
        np.abs(ang_x) / (2 * np.pi / roi_cc.shape[0])
        - (roi_cc.shape[0] - 1) / 2
    )  # peak in x
    px *= intersect_d  # convert from bins to camera pixels
    ang_y = np.angle(fft_values[1, 0])
    ang_y = ang_y - 2 * np.pi * (ang_y > 0)  # normalize
    py = (
        np.abs(ang_y) / (2 * np.pi / roi_cc.shape[1])
        - (roi_cc.shape[1] - 1) / 2
    )  # peak in y
    py *= intersect_d  # convert from bins to camera pixels
    return px, py


def _get_fft_peak_z(roi_cc: lib.IntArray1D, intersect_d: float) -> float:
    """Estimate the precise sub-pixel position of the peak of 1D
    ``roi_cc``.

    See :func:`_get_fft_peak` (its 2D counterpart) for the scaling of the
    peak from ``roi_cc`` bins to camera pixels.

    Parameters
    ----------
    roi_cc : lib.IntArray1D
        1D array with numbers of intersections in the local search
        region.
    intersect_d : float
        Intersect distance in camera pixels, i.e. the spacing between
        adjacent bins of ``roi_cc``.

    Returns
    -------
    pz : float
        Estimated z-coordinate of the peak, in camera pixels.
    """
    fft_values = np.fft.fft(roi_cc)
    ang_z = np.angle(fft_values[1])
    ang_z = ang_z - 2 * np.pi * (ang_z > 0)  # normalize
    pz = (
        np.abs(ang_z) / (2 * np.pi / roi_cc.size) - (roi_cc.size - 1) / 2
    )  # peak in z
    pz *= intersect_d  # convert from bins to camera pixels
    return pz


def _interpolate_drift(
    seg_bounds: lib.IntArray1D, *drifts: lib.FloatArray1D
) -> list[lib.FloatArray1D]:
    """Interpolate per-segment drifts to per-frame drifts (cubic spline).

    The drifts are sampled at the segment midpoints, so the first and the
    last half segment fall outside the knots. A cubic spline extrapolates
    freely there and can swing wildly, which is why one linearly
    extrapolated knot is appended at each end (as in the reference MATLAB
    implementation), bracketing every frame.

    Parameters
    ----------
    seg_bounds : lib.IntArray1D
        Frame indices of the segmentation bounds.
    *drifts : lib.FloatArray1D
        One drift array per dimension, each with one value per segment.

    Returns
    -------
    interpolated : list of lib.FloatArray1D
        The drifts sampled at every frame, in the order given.
    """
    t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
    t_inter = np.arange(seg_bounds[-1]) + 1
    if t.size == 1:  # a single segment holds no drift information
        return [np.full(t_inter.size, drift[0]) for drift in drifts]
    # bracket the frame range with one linearly extrapolated knot per end
    t = np.concatenate(([2 * t[0] - t[1]], t, [2 * t[-1] - t[-2]]))
    interpolated = []
    for drift in drifts:
        drift = np.concatenate(
            ([2 * drift[0] - drift[1]], drift, [2 * drift[-1] - drift[-2]])
        )
        interpolated.append(
            InterpolatedUnivariateSpline(t, drift, k=3)(t_inter)
        )
    return interpolated


def intersection_max(
    x: lib.SeriesOrFloatArray1D,
    y: lib.SeriesOrFloatArray1D,
    ref_x: lib.SeriesOrFloatArray1D,
    ref_y: lib.SeriesOrFloatArray1D,
    frame: lib.SeriesOrIntArray1D,
    seg_bounds: lib.IntArray1D,
    intersect_d: float,
    roi_r: float,
    width: int,
    aim_round: int = 1,
    exclude_self: bool = False,
    progress: lib.ProgressType | None = None,
) -> tuple[
    lib.FloatArray1D, lib.FloatArray1D, lib.FloatArray1D, lib.FloatArray1D
]:
    """Maximize intersection (undrift) for 2D localizations.

    Parameters
    ----------
    x, y : lib.SeriesOrFloatArray1D
        x and y coordinates of the localizations.
    ref_x, ref_y : lib.SeriesOrFloatArray1D
        x and y coordinates of the reference localizations.
    frame : lib.SeriesOrIntArray1D
        Frame indices of localizations, starting at 1.
    seg_bounds : lib.IntArray1D
        Frame indices of the segmentation bounds. Defines temporal
        intervals used to estimate drift.
    intersect_d : float
        Intersect distance in camera pixels.
    roi_r : float
        Radius of the local search region in camera pixels. Should be
        higher than the maximum expected drift within one segment.
    width : int
        Width of the camera image in camera pixels.
    aim_round : {1, 2}
        Round of AIM algorithm. The first round uses the first interval
        as reference, the second round uses the entire dataset as
        reference. The impact is that in the second round, the first
        interval is also undrifted.
    exclude_self : bool, optional
        Leave each segment out of the reference while that segment is
        being aligned. Only meaningful when the reference is the dataset
        itself (the second round), where it requires ``ref_x``/``ref_y``
        to be the same localizations as ``x``/``y``. A segment scored
        against a reference that contains it intersects itself perfectly
        at zero shift, which biases the estimated shift towards zero and
        compresses exactly the large corrections the second round exists
        to make. Default is False.
    progress : lib.ProgressType | None, optional
        Progress dialog. If TqdmProgress, progress is displayed with tqdm.
        If None or MockProgress, progress is not displayed. Default is None.

    Returns
    -------
    x_pdc, y_pdc : lib.FloatArray1D
        Undrifted x and y coordinates.
    drift_x, drift_y : lib.FloatArray1D
        Drift in x and y directions.
    """
    assert aim_round in [1, 2], "aim_round must be 1 or 2."
    if progress is None:
        progress = lib.MockProgress()

    # number of segments
    n_segments = len(seg_bounds) - 1
    rel_drift_x = 0  # adaptive drift (updated at each interval)
    rel_drift_y = 0

    # drift in x and y
    drift_x = np.zeros(n_segments)
    drift_y = np.zeros(n_segments)

    # find shifts for the local search region (in units of intersect_d)
    roi_units = int(np.ceil(roi_r / intersect_d))
    steps = np.arange(-roi_units, roi_units + 1, 1)
    box = len(steps)
    shifts_xy = np.zeros((box, box), dtype=np.int32)
    width_units = width / intersect_d
    for i, shift_x in enumerate(steps):
        for j, shift_y in enumerate(steps):
            shifts_xy[i, j] = shift_x + shift_y * width_units
    shifts_xy = shifts_xy.reshape(box**2)

    # convert reference to a 1D array in units of intersect_d and find
    # unique values and counts
    x0_units = np.round(ref_x / intersect_d)
    y0_units = np.round(ref_y / intersect_d)
    l0 = np.int32(x0_units + y0_units * width_units)  # 1d list
    l0_coords, l0_counts = np.unique(l0, return_counts=True)

    if exclude_self:
        # a segment's own bins are subtracted from the reference counts,
        # which is only defined if the reference holds the same bins
        l1 = np.int32(
            np.round(np.asarray(x) / intersect_d)
            + np.round(np.asarray(y) / intersect_d) * width_units
        )
        if not np.array_equal(l0, l1):
            raise ValueError(
                "exclude_self requires the reference to be the target"
                " itself, but ref_x/ref_y differ from x/y."
            )

    # sort the target localizations by frame so that each segment is a
    # contiguous slice (located with searchsorted). This avoids
    # re-scanning the whole array for every segment, which dominates the
    # runtime for low segmentation and large datasets.
    frame_sorted = np.asarray(frame)
    order = np.argsort(frame_sorted, kind="stable")
    frame_sorted = frame_sorted[order]
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]
    # first index of every segment: segment s spans the localizations
    # with seg_bounds[s] < frame <= seg_bounds[s + 1]
    seg_idx = np.searchsorted(frame_sorted, seg_bounds, side="right")
    # the reference encoded in the same order, so that each segment's own
    # contribution can be taken out of it again (see exclude_self)
    l0_sorted = l0[order] if exclude_self else None

    # initialize progress such that if GUI is used, tqdm is omitted
    start_idx = 1 if aim_round == 1 else 0
    iterator = progress.get_iterator(start_idx, n_segments)

    # run across each segment
    for s in iterator:
        # get the target localizations within the current segment
        lo, hi = seg_idx[s], seg_idx[s + 1]

        # skip if no target localizations
        if hi == lo:
            drift_x[s] = drift_x[s - 1]
            drift_y[s] = drift_y[s - 1]
            continue

        # undrifting from the previous round (new array, not a view)
        x1 = x_sorted[lo:hi] + rel_drift_x
        y1 = y_sorted[lo:hi] + rel_drift_y

        # take this segment out of the reference counts (restored below).
        # Counts that drop to zero are left in place: a zero count
        # contributes min(0, target count) == 0 to every shift.
        if exclude_self:
            own, own_counts = np.unique(l0_sorted[lo:hi], return_counts=True)
            own_pos = np.searchsorted(l0_coords, own)
            l0_counts[own_pos] -= own_counts

        # count the number of intersected localizations
        roi_cc = _point_intersect_2d(
            l0_coords,
            l0_counts,
            x1,
            y1,
            intersect_d,
            width_units,
            shifts_xy,
            box,
        )

        if exclude_self:
            l0_counts[own_pos] += own_counts
            # nothing left to align against (e.g. a single segment)
            if not roi_cc.any():
                drift_x[s] = drift_x[s - 1]
                drift_y[s] = drift_y[s - 1]
                continue

        # estimate the precise sub-pixel position of the peak of roi_cc
        # with FFT
        px, py = _get_fft_peak(roi_cc, intersect_d)

        # update the relative drift reference for the subsequent
        # segmented subset (interval) and save the drifts
        rel_drift_x += px
        rel_drift_y += py
        drift_x[s] = -rel_drift_x
        drift_y[s] = -rel_drift_y

        # update progress
        progress.set_value(s)

    # interpolate the drifts (cubic spline) for all frames
    drift_x, drift_y = _interpolate_drift(seg_bounds, drift_x, drift_y)

    # undrift the localizations
    x_pdc = x - drift_x[frame - 1]
    y_pdc = y - drift_y[frame - 1]

    return x_pdc, y_pdc, drift_x, drift_y


def intersection_max_z(
    x: lib.SeriesOrFloatArray1D,
    y: lib.SeriesOrFloatArray1D,
    z: lib.SeriesOrFloatArray1D,
    ref_x: lib.SeriesOrFloatArray1D,
    ref_y: lib.SeriesOrFloatArray1D,
    ref_z: lib.SeriesOrFloatArray1D,
    frame: lib.SeriesOrIntArray1D,
    seg_bounds: lib.IntArray1D,
    intersect_d: float,
    roi_r: float,
    width: int,
    height: int,
    pixelsize: float,
    aim_round: int = 1,
    exclude_self: bool = False,
    progress: lib.ProgressType | None = None,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Maximize intersection (undrift) for 3D localizations.

    Assumes that x and y coordinates were already undrifted. See
    :func:`intersection_max` for the algorithm and parameters
    explanation (its 2D counterpart).
    """
    # convert z to camera pixels
    z = z.copy() / pixelsize
    ref_z = ref_z.copy() / pixelsize

    # number of segments
    n_segments = len(seg_bounds) - 1
    rel_drift_z = 0  # adaptive drift (updated at each interval)

    # drift in z
    drift_z = np.zeros(n_segments)

    # find shifts for the local search region (in units of intersect_d)
    roi_units = int(np.ceil(roi_r / intersect_d))
    steps = np.arange(-roi_units, roi_units + 1, 1)
    width_units = width / intersect_d
    height_units = height / intersect_d
    shifts_z = steps.astype(np.int32) * width_units * height_units

    # convert reference to a 1D array in units of intersect_d and find
    # unique values and counts
    x0_units = np.round(ref_x / intersect_d)
    y0_units = np.round(ref_y / intersect_d)
    z0_units = np.round(ref_z / intersect_d)
    l0 = np.int32(
        x0_units
        + y0_units * width_units
        + z0_units * width_units * height_units
    )  # 1d list
    l0_coords, l0_counts = np.unique(l0, return_counts=True)

    if exclude_self:
        # see the 2D counterpart
        l1 = np.int32(
            np.round(np.asarray(x) / intersect_d)
            + np.round(np.asarray(y) / intersect_d) * width_units
            + np.round(np.asarray(z) / intersect_d)
            * width_units
            * height_units
        )
        if not np.array_equal(l0, l1):
            raise ValueError(
                "exclude_self requires the reference to be the target"
                " itself, but ref_x/ref_y/ref_z differ from x/y/z."
            )

    # sort the target localizations by frame so that each segment is a
    # contiguous slice (located with searchsorted). This avoids
    # re-scanning the whole array for every segment, which dominates the
    # runtime for low segmentation and large datasets.
    frame_sorted = np.asarray(frame)
    order = np.argsort(frame_sorted, kind="stable")
    frame_sorted = frame_sorted[order]
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]
    z_sorted = np.asarray(z)[order]
    # the reference encoded in the same order, so that each segment's own
    # contribution can be taken out of it again (see exclude_self)
    l0_sorted = l0[order] if exclude_self else None
    # first index of every segment: segment s spans the localizations
    # with seg_bounds[s] < frame <= seg_bounds[s + 1]
    seg_idx = np.searchsorted(frame_sorted, seg_bounds, side="right")

    # initialize progress such that if GUI is used, tqdm is omitted
    start_idx = 1 if aim_round == 1 else 0
    iterator = progress.get_iterator(start_idx, n_segments)

    # run across each segment
    for s in iterator:
        # get the target localizations within the current segment
        lo, hi = seg_idx[s], seg_idx[s + 1]

        # skip if no target localizations
        if hi == lo:
            drift_z[s] = drift_z[s - 1]
            continue

        x1 = x_sorted[lo:hi]
        y1 = y_sorted[lo:hi]
        # undrifting from the previous round (new array, not a view)
        z1 = z_sorted[lo:hi] + rel_drift_z

        # take this segment out of the reference counts (see the 2D
        # counterpart); restored right after the intersection counting
        if exclude_self:
            own, own_counts = np.unique(l0_sorted[lo:hi], return_counts=True)
            own_pos = np.searchsorted(l0_coords, own)
            l0_counts[own_pos] -= own_counts

        # count the number of intersected localizations
        roi_cc = _point_intersect_3d(
            l0_coords,
            l0_counts,
            x1,
            y1,
            z1,
            intersect_d,
            width_units,
            height_units,
            shifts_z,
        )

        if exclude_self:
            l0_counts[own_pos] += own_counts
            # nothing left to align against (e.g. a single segment)
            if not roi_cc.any():
                drift_z[s] = drift_z[s - 1]
                continue

        # estimate the precise sub-pixel position of the peak of roi_cc
        # with FFT
        pz = _get_fft_peak_z(roi_cc, intersect_d)

        # update the relative drift reference for the subsequent
        # segmented subset (interval) and save the drifts
        rel_drift_z += pz
        drift_z[s] = -rel_drift_z

        # update progress
        progress.set_value(s)

    # interpolate the drifts (cubic spline) for all frames
    (drift_z,) = _interpolate_drift(seg_bounds, drift_z)

    # undrift the localizations
    z_pdc = z - drift_z[frame - 1]

    # convert back to nm
    z_pdc *= pixelsize
    drift_z *= pixelsize

    return z_pdc, drift_z


def aim(
    locs: pd.DataFrame,
    info: list[dict],
    segmentation: int = 100,
    intersect_d: float = 20 / 130,
    roi_r: float = 60 / 130,
    progress: lib.ProgressDialog | Literal["console"] | None = None,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Apply AIM undrifting to the localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations list to be undrifted.
    info : list of dicts
        Localizations list's metadata.
    intersect_d : float
        Intersect distance in camera pixels.
    segmentation : int
        Time interval for drift tracking, unit: frames.
    roi_r : float
        Radius of the local search region in camera pixels. Should be
        larger than the  maximum expected drift within segmentation.
    progress : picasso.lib.ProgressDialog or "console" or None, optional
        Progress dialog. If "console", progress is displayed in the
        console. If None, no progress is displayed. Default is None.

    Returns
    -------
    locs : pd.DataFrame
        Undrifted localizations.
    new_info : list of 1 dict
        Updated metadata.
    drift : pd.DataFrame
        Drift in x and y directions (and z if applicable).
    """
    progress = lib.normalize_progress(
        progress, description="Undrifting by AIM (1/2)"
    )

    locs = locs.copy()
    # extract metadata
    width = lib.get_from_metadata(info, "Width", raise_error=True)
    height = lib.get_from_metadata(info, "Height", raise_error=True)
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)

    # frames should start at 1
    frame = locs["frame"] + 1 - locs["frame"].min()  # 1d array

    # find the segmentation bounds (temporal intervals)
    seg_bounds = np.concatenate(
        (np.arange(0, n_frames, segmentation), [n_frames])
    )

    # get the reference localizations (first interval)
    ref_x = locs["x"][frame <= segmentation]
    ref_y = locs["y"][frame <= segmentation]

    # RUN AIM TWICE #
    # the first run is with the first interval as reference
    x_pdc, y_pdc, drift_x1, drift_y1 = intersection_max(
        locs["x"],
        locs["y"],
        ref_x,
        ref_y,
        frame,
        seg_bounds,
        intersect_d,
        roi_r,
        width,
        aim_round=1,
        progress=progress,
    )
    # the second run is with the entire dataset as reference
    progress.zero_progress(description="Undrifting by AIM (2/2)")
    x_pdc, y_pdc, drift_x2, drift_y2 = intersection_max(
        x_pdc,
        y_pdc,
        x_pdc,
        y_pdc,
        frame,
        seg_bounds,
        intersect_d,
        roi_r,
        width,
        aim_round=2,
        exclude_self=True,
        progress=progress,
    )

    # add the drifts together from the two rounds
    drift_x = drift_x1 + drift_x2
    drift_y = drift_y1 + drift_y2

    # shift the drifts by the mean value
    shift_x = np.mean(drift_x)
    shift_y = np.mean(drift_y)
    drift_x -= shift_x
    drift_y -= shift_y
    x_pdc += shift_x
    y_pdc += shift_y

    # 3D undrifting
    if "z" in locs.columns:
        progress.zero_progress(description="Undrifting z (1/2)")
        ref_x = x_pdc[frame <= segmentation]
        ref_y = y_pdc[frame <= segmentation]
        ref_z = locs["z"][frame <= segmentation]
        z_pdc, drift_z1 = intersection_max_z(
            x_pdc,
            y_pdc,
            locs["z"],
            ref_x,
            ref_y,
            ref_z,
            frame,
            seg_bounds,
            intersect_d,
            roi_r,
            width,
            height,
            pixelsize,
            aim_round=1,
            progress=progress,
        )
        progress.zero_progress(description="Undrifting z (2/2)")
        z_pdc, drift_z2 = intersection_max_z(
            x_pdc,
            y_pdc,
            z_pdc,
            x_pdc,
            y_pdc,
            z_pdc,
            frame,
            seg_bounds,
            intersect_d,
            roi_r,
            width,
            height,
            pixelsize,
            aim_round=2,
            exclude_self=True,
            progress=progress,
        )
        drift_z = drift_z1 + drift_z2
        shift_z = np.mean(drift_z)
        drift_z -= shift_z
        z_pdc += shift_z
        # combine
        drift = pd.DataFrame(
            {"x": drift_x, "y": drift_y, "z": drift_z}, dtype="float32"
        )
    else:
        # combine
        drift = pd.DataFrame({"x": drift_x, "y": drift_y}, dtype="float32")

    # apply the drift to localizations
    locs["x"] = x_pdc
    locs["y"] = y_pdc
    if "z" in locs.columns:
        locs["z"] = z_pdc
    new_info = {
        "Generated by": f"Picasso v{__version__} AIM",
        "Intersect distance (nm)": intersect_d * pixelsize,
        "Segmentation": segmentation,
        "Search regions radius (nm)": roi_r * pixelsize,
    }
    new_info = info + [new_info]
    progress.close()
    return locs, new_info, drift
