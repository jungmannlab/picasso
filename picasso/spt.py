"""
picasso.spt
~~~~~~~~~~~

Basic single-particle tracking (SPT): linking localizations into tracks
and measuring how fast the tracked particles diffuse.

The workflow follows Module 7 of Martens, et al. Frontiers in
Bioinformatics, 2021. DOI: 10.3389/fbinf.2021.817254:

1. **Linking.** :func:`link_tracks` links the localizations of
   consecutive frames into tracks by nearest-neighbor tracking. A
   localization continues the track whose last localization is the
   closest one within ``max_jump``, taking blinking into account: a
   track may skip up to ``max_gap`` frames. Conflicts (two tracks
   reaching for the same localization, or one track for two) are
   resolved globally, the closest pair first. The track id of every
   localization is stored in the ``group`` column, and a previous
   grouping (e.g., picked cells) is kept in ``group_input``; tracks
   never cross the input groups.
2. **Mean squared displacement (MSD).** :func:`msd` averages the squared
   displacements of all pairs of localizations of a track that are
   ``n`` frames apart, per track and over all tracks (the ensemble
   MSD). :func:`fit_msd` fits ``MSD(tau) = 4 D tau + 4 sigma^2`` to the
   first lags, which gives the diffusion coefficient D from the slope
   and the apparent localization error sigma from the intercept.
3. **Jump distances.** :func:`jump_distances` collects the distances
   between localizations of a track in consecutive frames.
   :func:`fit_jump_distances` fits them with a mixture of one or more
   freely diffusing populations in 2D, each with the distribution

   ``p(r) = 2 r / s * exp(-r^2 / s)``, with ``s = 4 D dt + 4 sigma^2``,

   which gives the diffusion coefficient and the fraction of jumps of
   each population. Unlike the paper, which fits the histogram of the
   jump distances, the fit maximizes the likelihood of the jump
   distances themselves, so the result does not depend on the binning;
   the histogram is only used for display.

:func:`analyze_tracks` runs the whole analysis and :func:`save_analysis`
saves the results. :func:`track_table` summarizes every track in one
row (length, duration, mean position, mean jump distance and D).

Units
-----
Linking works in nm and frames, like the rest of Picasso's localization
processing. The analysis needs the frame time (ms, the time between two
consecutive frames, not the exposure time); it is read from the metadata
(``"Frame time (ms)"``) unless given. The results are reported as
diffusion coefficients in um^2/s, MSDs in um^2, time lags in s, and
jump distances and localization errors in nm.

Limitations
-----------
Tracking and analysis are 2D (z is ignored). The linking has no motion
model and does not merge or split tracks, so it is meant for sparse
particles, i.e., when the particles are further apart than they move
between frames.

:author: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import io, lib

__all__ = [
    "FRAME_TIME_KEY",
    "TRACKING_KEY",
    "link_tracks",
    "tracking_info",
    "max_jump_nm",
    "is_tracked",
    "median_precision",
    "track_segments",
    "msd",
    "fit_msd",
    "jump_distances",
    "fit_jump_distances",
    "jump_distance_pdf",
    "track_table",
    "analyze_tracks",
    "summarize_analysis",
    "save_analysis",
    "plot_msd",
    "plot_jump_distances",
    "plot_track_lengths",
]

#: Metadata key of the frame time (ms), i.e., the time between the
#: starts of two consecutive frames.
FRAME_TIME_KEY = "Frame time (ms)"
#: Metadata key marking the entry written by :func:`link_tracks`.
TRACKING_KEY = "Tracking"
#: Name of the linking method, stored under :data:`TRACKING_KEY`.
TRACKING_METHOD = "nearest neighbor"
#: Default maximum number of frames a track may skip.
MAX_GAP = 1
#: Default minimum number of localizations in a track.
MIN_LENGTH = 2
#: Default maximum time lag (frames) of the MSD.
MAX_LAG = 10
#: Default number of time lags fitted in the MSD analysis.
N_FIT = 4
#: Bounds of the diffusion coefficients fitted to the jump distances
#: (um^2/s).
D_BOUNDS = (1e-6, 1e3)
#: Minimum number of jump distances per free parameter of the jump
#: distance fit (a fit of K populations has 2K - 1 free parameters).
MIN_JUMPS_PER_PARAMETER = 5


# --------------------------------------------------------------------
# Linking
# --------------------------------------------------------------------


@numba.njit(cache=True, nogil=True)
def _link_tracks_kernel(
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    group: lib.IntArray1D,
    max_jump: float,
    max_gap: int,
) -> lib.IntArray1D:
    """Link localizations into tracks, see ``link_tracks``.

    Assumes that the localizations are sorted by frame and, within each
    frame, by x.

    Parameters
    ----------
    frame : lib.IntArray1D
        Frames of the localizations (int64).
    x, y : lib.FloatArray1D
        Coordinates of the localizations (nm).
    group : lib.IntArray1D
        Input groups; localizations of different groups are never
        linked.
    max_jump : float
        Maximum distance between consecutive localizations of a track
        (nm).
    max_gap : int
        Maximum number of frames a track may skip.

    Returns
    -------
    track : lib.IntArray1D
        Track id of every localization, in order of the tracks' first
        localizations.
    """
    n = len(frame)
    track = np.full(n, -1, dtype=np.int64)
    max_jump_2 = max_jump * max_jump
    # active track ends (indices of the last localization of a track),
    # sorted by frame
    ends = np.empty(n, dtype=np.int64)
    n_ends = 0
    # candidate links (squared distance, index into ends, localization)
    capacity = 1024
    cand_d2 = np.empty(capacity, dtype=np.float64)
    cand_end = np.empty(capacity, dtype=np.int64)
    cand_loc = np.empty(capacity, dtype=np.int64)
    next_id = 0
    start = 0
    while start < n:
        current_frame = frame[start]
        stop = start
        while stop < n and frame[stop] == current_frame:
            stop += 1
        # retire the ends that skipped too many frames
        n_old = 0
        while (
            n_old < n_ends and current_frame - frame[ends[n_old]] > max_gap + 1
        ):
            n_old += 1
        if n_old > 0:
            for k in range(n_old, n_ends):
                ends[k - n_old] = ends[k]
            n_ends -= n_old
        # candidates, the most recent ends first, so that the stable sort
        # below gives ties to the track that skipped fewer frames
        n_cand = 0
        x_frame = x[start:stop]
        for k in range(n_ends - 1, -1, -1):
            end = ends[k]
            x_end = x[end]
            y_end = y[end]
            lo = start + np.searchsorted(
                x_frame, x_end - max_jump, side="left"
            )
            hi = start + np.searchsorted(
                x_frame, x_end + max_jump, side="right"
            )
            for j in range(lo, hi):
                if group[j] != group[end]:
                    continue
                dy = y[j] - y_end
                if dy * dy > max_jump_2:
                    continue
                dx = x[j] - x_end
                d2 = dx * dx + dy * dy
                if d2 > max_jump_2:
                    continue
                if n_cand == capacity:
                    capacity *= 2
                    new_d2 = np.empty(capacity, dtype=np.float64)
                    new_end = np.empty(capacity, dtype=np.int64)
                    new_loc = np.empty(capacity, dtype=np.int64)
                    new_d2[:n_cand] = cand_d2[:n_cand]
                    new_end[:n_cand] = cand_end[:n_cand]
                    new_loc[:n_cand] = cand_loc[:n_cand]
                    cand_d2 = new_d2
                    cand_end = new_end
                    cand_loc = new_loc
                cand_d2[n_cand] = d2
                cand_end[n_cand] = k
                cand_loc[n_cand] = j
                n_cand += 1
        # resolve conflicts globally, the closest pair first
        end_taken = np.zeros(n_ends, dtype=np.bool_)
        if n_cand > 0:
            order = np.argsort(cand_d2[:n_cand], kind="mergesort")
            for o in order:
                k = cand_end[o]
                j = cand_loc[o]
                if end_taken[k] or track[j] != -1:
                    continue
                end_taken[k] = True
                track[j] = track[ends[k]]
        # localizations without a predecessor start new tracks
        for j in range(start, stop):
            if track[j] == -1:
                track[j] = next_id
                next_id += 1
        # the unmatched ends stay active, followed by this frame's
        # localizations, which keeps the ends sorted by frame
        n_kept = 0
        for k in range(n_ends):
            if not end_taken[k]:
                ends[n_kept] = ends[k]
                n_kept += 1
        for j in range(start, stop):
            ends[n_kept] = j
            n_kept += 1
        n_ends = n_kept
        start = stop
    return track


def link_tracks(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    max_jump: float,
    max_gap: int = MAX_GAP,
    min_length: int = MIN_LENGTH,
    frame_time: float | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Link localizations into tracks by nearest-neighbor tracking.

    Frame by frame, every localization continues the track whose last
    localization is the closest one within ``max_jump``, as in Module 7
    of Martens, et al. Frontiers in Bioinformatics, 2021. DOI:
    10.3389/fbinf.2021.817254. A track may skip up to ``max_gap``
    frames, e.g., when the fluorophore blinks or the particle is not
    detected. Each track gets at most one localization per frame and
    each localization belongs to one track: when several tracks reach
    for the same localization (or one track for several), the closest
    pair is linked first, then the next closest, and so on. Of two
    equally close tracks, the one that skipped fewer frames wins.
    Localizations that do not continue any track start a new one.

    Linking is 2D; z is ignored.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations. If they have a ``group`` column (e.g., picked
        cells), tracks never link localizations of different groups.
        Should not be linked by ``postprocess.link`` beforehand, which
        merges the localizations of consecutive frames.
    info : list of dicts
        Metadata of ``locs``.
    max_jump : float
        Maximum distance between consecutive localizations of a track
        (nm). Should be larger than the jumps of the fastest
        particles (e.g., ``3 * sqrt(4 * D * dt)``) and smaller than the
        distance between particles.
    max_gap : int, optional
        Maximum number of frames a track may skip. 0 links consecutive
        frames only. Default is 1.
    min_length : int, optional
        Minimum number of localizations in a track. Localizations of
        shorter tracks are removed. Default is 2.
    frame_time : float, optional
        Time between two consecutive frames (ms). Not used for linking,
        but stored in the metadata for the analysis of the tracks.
        Default is None (not stored).

    Returns
    -------
    tracked_locs : pd.DataFrame
        Localizations of the tracks, sorted by frame, with the track id
        in the ``group`` column (consecutive integers in the order of
        the tracks' first frames). If ``locs`` had a ``group`` column,
        it is kept in ``group_input``.
    info : list of dicts
        Metadata of ``tracked_locs``: ``info`` with an entry describing
        the tracking appended, see ``lib.append_metadata``.

    Raises
    ------
    ValueError
        If ``max_jump`` is not positive, ``max_gap`` is negative or
        ``min_length`` is smaller than 1.
    """
    if not np.isfinite(max_jump) or max_jump <= 0:
        raise ValueError(f"max_jump must be positive, got {max_jump}.")
    if int(max_gap) != max_gap or max_gap < 0:
        raise ValueError(
            f"max_gap must be a non-negative integer, got {max_gap}."
        )
    if int(min_length) != min_length or min_length < 1:
        raise ValueError(
            f"min_length must be a positive integer, got {min_length}."
        )
    max_gap = int(max_gap)
    min_length = int(min_length)

    frame = locs["frame"].to_numpy().astype(np.int64)
    x = locs["x"].to_numpy().astype(np.float64)
    y = locs["y"].to_numpy().astype(np.float64)
    if "group" in locs.columns:
        group = locs["group"].to_numpy().astype(np.int64)
    else:
        group = np.zeros(len(locs), dtype=np.int64)
    order = np.lexsort((x, frame))
    track_sorted = _link_tracks_kernel(
        frame[order],
        x[order],
        y[order],
        group[order],
        float(max_jump),
        max_gap,
    )
    track = np.empty_like(track_sorted)
    track[order] = track_sorted

    # keep the tracks that are long enough, numbered consecutively
    if len(track):
        lengths = np.bincount(track)
        keep = lengths[track] >= min_length
    else:
        keep = np.zeros(0, dtype=bool)
    track_ids = np.unique(track[keep], return_inverse=True)[1]
    new_track = np.full(len(track), -1, dtype=np.int64)
    new_track[keep] = track_ids
    by_frame = np.argsort(frame, kind="stable")
    by_frame = by_frame[keep[by_frame]]
    tracked = locs.iloc[by_frame].reset_index(drop=True)
    tracked = lib.append_group(tracked, new_track[by_frame].astype(np.int32))

    n_tracks = int(track_ids.max()) + 1 if len(track_ids) else 0
    entry = {
        **lib.step_info("SPT"),
        TRACKING_KEY: TRACKING_METHOD,
        "Max. jump (nm)": float(max_jump),
    }
    entry.update(
        {
            "Max. gap (frames)": max_gap,
            "Min. track length": min_length,
            "Number of tracks": n_tracks,
        }
    )
    if "group" in locs.columns:
        entry["Input groups kept in"] = "group_input"
    if frame_time is not None:
        entry[FRAME_TIME_KEY] = float(frame_time)
    return tracked, lib.append_metadata(info, entry)


def tracking_info(info: list[dict] | None) -> dict | None:
    """The metadata entry written by :func:`link_tracks`.

    Parameters
    ----------
    info : list of dicts or None
        Metadata of the localizations.

    Returns
    -------
    entry : dict or None
        The last entry describing a tracking, or None if the
        localizations were not tracked.
    """
    for entry in (info or [])[::-1]:
        if isinstance(entry, dict) and TRACKING_KEY in entry:
            return entry
    return None


def max_jump_nm(info: list[dict] | None) -> float | None:
    """The maximum jump distance of the tracking (nm) from the metadata.

    Parameters
    ----------
    info : list of dicts or None
        Metadata of the localizations.

    Returns
    -------
    max_jump : float or None
        The maximum jump distance in nm, or None if the localizations
        were not tracked or the entry does not record it.
    """
    entry = tracking_info(info)
    if entry is None:
        return None
    if "Max. jump (nm)" in entry:
        return float(entry["Max. jump (nm)"])
    return None


def is_tracked(locs: pd.DataFrame, info: list[dict] | None) -> bool:
    """Whether the ``group`` column of ``locs`` holds tracks.

    True if the metadata describe a tracking and every group has at
    most one localization per frame. The latter fails when ``group``
    was overwritten after tracking, e.g., by picking or clustering.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts or None
        Metadata of the localizations.

    Returns
    -------
    tracked : bool
    """
    if tracking_info(info) is None or "group" not in locs.columns:
        return False
    return not locs.duplicated(["group", "frame"]).any()


def median_precision(locs: pd.DataFrame) -> float:
    """Median localization precision, averaged over x and y.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with the ``lpx`` and ``lpy`` columns (nm).

    Returns
    -------
    precision : float
        Median localization precision (nm).
    """
    lp = np.mean([np.nanmedian(locs["lpx"]), np.nanmedian(locs["lpy"])])
    return float(lp)


def _require_tracks(locs: pd.DataFrame) -> None:
    """Raise a ValueError if ``locs`` have no track ids."""
    if "group" not in locs.columns:
        raise ValueError(
            "The localizations have no 'group' column with track ids; "
            "link them into tracks first (picasso.spt.link_tracks)."
        )


def _sorted_tracks(locs: pd.DataFrame) -> tuple:
    """Sort localizations by track and frame.

    Returns
    -------
    order : lib.IntArray1D
        Positions of the localizations in the sorted order.
    group, frame : lib.IntArray1D
        Track ids and frames, sorted (int64).
    x, y : lib.FloatArray1D
        Coordinates, sorted (float64, nm).
    starts : lib.IntArray1D
        Start of every track in the sorted order, followed by the
        number of localizations.
    """
    _require_tracks(locs)
    group = locs["group"].to_numpy().astype(np.int64)
    frame = locs["frame"].to_numpy().astype(np.int64)
    order = np.lexsort((frame, group))
    group = group[order]
    frame = frame[order]
    x = locs["x"].to_numpy().astype(np.float64)[order]
    y = locs["y"].to_numpy().astype(np.float64)[order]
    boundaries = np.flatnonzero(np.diff(group)) + 1
    starts = np.concatenate(([0], boundaries, [len(group)]))
    if len(group) == 0:
        starts = np.zeros(1, dtype=np.int64)
    return order, group, frame, x, y, starts.astype(np.int64)


def track_segments(
    locs: pd.DataFrame,
) -> tuple[lib.FloatArray2D, lib.IntArray1D, lib.IntArray1D]:
    """Line segments between consecutive localizations of every track,
    e.g., for drawing the tracks.

    Parameters
    ----------
    locs : pd.DataFrame
        Tracked localizations (track ids in ``group``).

    Returns
    -------
    segments : lib.FloatArray2D
        ``(n_segments, 4)`` array of ``x0, y0, x1, y1`` (nm).
    track_id : lib.IntArray1D
        Track id of every segment.
    start_row : lib.IntArray1D
        Position (row) in ``locs`` of the first localization of every
        segment.
    """
    order, group, _, x, y, _ = _sorted_tracks(locs)
    same = group[1:] == group[:-1]
    segments = np.column_stack(
        (x[:-1][same], y[:-1][same], x[1:][same], y[1:][same])
    )
    return segments.reshape(-1, 4), group[:-1][same], order[:-1][same]


# --------------------------------------------------------------------
# Metadata helpers
# --------------------------------------------------------------------


def _positive(value: Any, name: str) -> float:
    """``value`` as a positive float, or a ValueError."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _frame_time(frame_time: float | None, info: list[dict] | None) -> float:
    """The frame time (ms), given or from the metadata."""
    if frame_time is None and info is not None:
        frame_time = lib.get_from_metadata(info, FRAME_TIME_KEY)
    if frame_time is None:
        raise ValueError(
            "The frame time is unknown. Pass frame_time (ms) or add "
            f"'{FRAME_TIME_KEY}' to the metadata."
        )
    return _positive(frame_time, "frame_time")


# --------------------------------------------------------------------
# Mean squared displacement
# --------------------------------------------------------------------


@numba.njit(cache=True, nogil=True)
def _msd_kernel(
    starts: lib.IntArray1D,
    frame: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    max_lag: int,
) -> tuple[lib.FloatArray2D, lib.IntArray2D]:
    """Sums and numbers of the squared displacements of every track at
    the time lags 1 to ``max_lag`` frames. Pairs of localizations are
    counted at their exact frame difference, so a skipped frame leaves
    out the pairs that would include it.

    Returns
    -------
    sums : lib.FloatArray2D
        ``(n_tracks, max_lag)`` sums of squared displacements (nm^2).
    counts : lib.IntArray2D
        ``(n_tracks, max_lag)`` numbers of pairs.
    """
    n_tracks = len(starts) - 1
    sums = np.zeros((n_tracks, max_lag), dtype=np.float64)
    counts = np.zeros((n_tracks, max_lag), dtype=np.int64)
    for t in range(n_tracks):
        for a in range(starts[t], starts[t + 1]):
            for b in range(a + 1, starts[t + 1]):
                lag = frame[b] - frame[a]
                if lag > max_lag:
                    break
                if lag < 1:
                    continue
                dx = x[b] - x[a]
                dy = y[b] - y[a]
                sums[t, lag - 1] += dx * dx + dy * dy
                counts[t, lag - 1] += 1
    return sums, counts


def msd(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    frame_time: float | None = None,
    max_lag: int = MAX_LAG,
) -> dict:
    """Mean squared displacement (MSD) of tracked particles.

    For every time lag of ``n`` frames, the squared displacements of all
    pairs of localizations of a track that are exactly ``n`` frames apart
    are averaged, per track (time-averaged MSD) and over all tracks
    (ensemble MSD, i.e., the mean over all pairs of all tracks).

    Parameters
    ----------
    locs : pd.DataFrame
        Tracked localizations (track ids in ``group``).
    info : list of dicts, optional
        Metadata, used for ``frame_time`` if not given. Default is None.
    frame_time : float, optional
        Time between two consecutive frames (ms). Default is None (from
        the metadata).
    max_lag : int, optional
        Largest time lag (frames). Default is 10.

    Returns
    -------
    result : dict
        ``"lag"`` (frames) and ``"tau"`` (s): the time lags.
        ``"msd"``: ensemble MSD (um^2); NaN at lags without any pair.
        ``"sem"``: standard error of the ensemble MSD (um^2), treating
        tracks as independent; NaN with fewer than two tracks.
        ``"n_pairs"``, ``"n_tracks"``: numbers of pairs and of tracks
        with at least one pair, per lag.
        ``"track_msd"``: ``(n_tracks, max_lag)`` MSD of every track
        (um^2), NaN where a track has no pair.
        ``"track_n_pairs"``: ``(n_tracks, max_lag)`` numbers of pairs.
        ``"groups"``: track ids, one per row of ``"track_msd"``.
        ``"frame_time"`` (ms).

    Raises
    ------
    ValueError
        If ``max_lag`` is smaller than 1, the localizations are not
        tracked, or the frame time is unknown.
    """
    if int(max_lag) != max_lag or max_lag < 1:
        raise ValueError(f"max_lag must be a positive integer, got {max_lag}.")
    max_lag = int(max_lag)
    frame_time = _frame_time(frame_time, info)
    _, group, frame, x, y, starts = _sorted_tracks(locs)
    sums, counts = _msd_kernel(starts, frame, x, y, max_lag)
    sums *= 1e-6  # nm^2 to um^2
    with np.errstate(invalid="ignore", divide="ignore"):
        track_msd = sums / counts
        n_pairs = counts.sum(axis=0)
        ensemble = sums.sum(axis=0) / n_pairs
        # standard error of a ratio (weighted mean) estimator, with the
        # tracks as independent units: pairs of one track are correlated
        has = counts > 0
        n_tracks = has.sum(axis=0)
        deviation = np.where(has, counts * (track_msd - ensemble), 0.0)
        sem = np.sqrt((deviation**2).sum(axis=0)) / n_pairs
    sem[n_tracks < 2] = np.nan
    lag = np.arange(1, max_lag + 1)
    return {
        "lag": lag,
        "tau": lag * frame_time / 1000,
        "msd": ensemble,
        "sem": sem,
        "n_pairs": n_pairs,
        "n_tracks": n_tracks,
        "track_msd": track_msd,
        "track_n_pairs": counts,
        "groups": group[starts[:-1]],
        "frame_time": frame_time,
    }


def fit_msd(
    tau: lib.FloatArray1D,
    msd_values: lib.FloatArray1D | lib.FloatArray2D,
    n_fit: int = N_FIT,
) -> dict:
    """Fit ``MSD(tau) = 4 D tau + 4 sigma^2`` (free diffusion in 2D) to
    the first time lags of an MSD curve, by ordinary least squares.

    The intercept also contains the motion blur during the camera
    exposure (``-4/3 D t_exposure`` for continuous illumination), which
    is not modeled, so ``sigma`` is an apparent localization error.

    Parameters
    ----------
    tau : lib.FloatArray1D
        Time lags (s).
    msd_values : lib.FloatArray1D or lib.FloatArray2D
        MSD (um^2) at ``tau``, or one MSD curve per row (e.g., the
        ``"track_msd"`` of :func:`msd`).
    n_fit : int, optional
        Number of time lags fitted, at least 2. Default is 4.

    Returns
    -------
    result : dict
        ``"D"``: diffusion coefficient (um^2/s), ``"intercept"`` (um^2,
        may be negative), ``"sigma"``: apparent localization error
        (nm, NaN if the intercept is not positive) and ``"n_fit"``.
        Floats for a single curve, arrays for one curve per row; NaN
        where any of the fitted lags is NaN (no pairs).

    Raises
    ------
    ValueError
        If ``n_fit`` is smaller than 2 or larger than the number of
        time lags.
    """
    tau = np.asarray(tau, dtype=np.float64)
    values = np.asarray(msd_values, dtype=np.float64)
    if int(n_fit) != n_fit or n_fit < 2:
        raise ValueError(
            f"n_fit must be an integer of at least 2, got {n_fit}."
        )
    n_fit = int(n_fit)
    if n_fit > len(tau):
        raise ValueError(
            f"Cannot fit {n_fit} time lags of an MSD with {len(tau)} lags."
        )
    t = tau[:n_fit]
    v = values[..., :n_fit]
    t_centered = t - t.mean()
    slope = (v * t_centered).sum(axis=-1) / (t_centered**2).sum()
    intercept = v.mean(axis=-1) - slope * t.mean()
    with np.errstate(invalid="ignore"):
        sigma = np.where(intercept > 0, np.sqrt(intercept / 4) * 1000, np.nan)
    if values.ndim == 1:
        return {
            "D": float(slope / 4),
            "intercept": float(intercept),
            "sigma": float(sigma),
            "n_fit": n_fit,
        }
    return {
        "D": slope / 4,
        "intercept": intercept,
        "sigma": sigma,
        "n_fit": n_fit,
    }


# --------------------------------------------------------------------
# Jump distances
# --------------------------------------------------------------------


def _jumps(locs: pd.DataFrame) -> tuple[lib.FloatArray1D, lib.IntArray1D]:
    """Distances (nm) between localizations of a track in consecutive
    frames, and the index of the track (0 to n_tracks - 1) of every
    jump."""
    _, group, frame, x, y, starts = _sorted_tracks(locs)
    track_index = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    consecutive = (group[1:] == group[:-1]) & (frame[1:] - frame[:-1] == 1)
    d = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])[consecutive]
    return d, track_index[:-1][consecutive]


def jump_distances(locs: pd.DataFrame) -> lib.FloatArray1D:
    """Distances between localizations of a track in consecutive frames.

    Jumps across skipped frames are left out, since they span a longer
    time.

    Parameters
    ----------
    locs : pd.DataFrame
        Tracked localizations (track ids in ``group``).

    Returns
    -------
    jumps : lib.FloatArray1D
        Jump distances (nm).
    """
    return _jumps(locs)[0]


def _softmax(w: lib.FloatArray1D) -> lib.FloatArray1D:
    """Softmax of ``w``."""
    e = np.exp(w - w.max())
    return e / e.sum()


def _jd_objective(
    theta: lib.FloatArray1D,
    r2: lib.FloatArray1D,
    sum_log_2r: float,
    n_populations: int,
    dt: float,
    s0: float,
    r_max_2: float | None,
) -> tuple[float, lib.FloatArray1D]:
    """Negative log-likelihood (per jump) of the jump distances under a
    mixture of freely diffusing populations, and its gradient.

    Parameters
    ----------
    theta : lib.FloatArray1D
        ``log D`` of every population (um^2/s), followed by the softmax
        weights of the fractions of populations 2 to K (that of
        population 1 is 0).
    r2 : lib.FloatArray1D
        Squared jump distances (um^2).
    sum_log_2r : float
        Sum of ``log(2 r)`` over the jumps (um), a constant term of the
        log-likelihood.
    n_populations : int
        Number of populations K.
    dt : float
        Frame time (s).
    s0 : float
        ``4 sigma^2`` (um^2).
    r_max_2 : float or None
        Squared maximum jump distance (um^2) if the distribution is
        truncated there, else None.

    Returns
    -------
    nll : float
        Negative log-likelihood divided by the number of jumps.
    gradient : lib.FloatArray1D
        Gradient of ``nll`` with respect to ``theta``.
    """
    k = n_populations
    n = len(r2)
    D = np.exp(theta[:k])
    a = _softmax(np.concatenate(([0.0], theta[k:])))
    s = 4 * dt * D + s0
    # log of the weighted densities without the common log(2 r) term
    q = (np.log(a) - np.log(s))[:, None] - r2[None, :] / s[:, None]
    q_max = q.max(axis=0)
    e = np.exp(q - q_max[None, :])
    e_sum = e.sum(axis=0)
    log_likelihood = (q_max + np.log(e_sum)).sum() + sum_log_2r
    gamma = e / e_sum[None, :]  # responsibilities
    gamma_sum = gamma.sum(axis=1)
    d_ll_d_s = (gamma * r2[None, :]).sum(axis=1) / s**2 - gamma_sum / s
    grad_log_D = d_ll_d_s * 4 * dt * D
    grad_w = gamma_sum[1:] - n * a[1:]
    if r_max_2 is not None:
        e = np.exp(-r_max_2 / s)
        F = 1 - e  # fraction of each population's jumps within r_max
        T = (a * F).sum()
        log_likelihood -= n * np.log(T)
        d_F_d_s = -(r_max_2 / s**2) * e
        grad_log_D -= n * a * d_F_d_s * 4 * dt * D / T
        grad_w -= n * a[1:] * (F[1:] - T) / T
    gradient = -np.concatenate((grad_log_D, grad_w)) / n
    return -log_likelihood / n, gradient


def fit_jump_distances(
    jumps: lib.FloatArray1D,
    frame_time: float,
    n_populations: int = 1,
    sigma: float = 0.0,
    r_max: float | None = None,
) -> dict:
    """Fit the jump distances with a mixture of freely diffusing
    populations, by maximum likelihood.

    In 2D, the distance ``r`` a particle with the diffusion coefficient
    D moves in one frame time ``dt``, measured with the localization
    error ``sigma``, is distributed as

    ``p(r) = 2 r / s * exp(-r^2 / s)``, with ``s = 4 D dt + 4 sigma^2``.

    The jump distances are fitted with a mixture of ``n_populations``
    such distributions, with free diffusion coefficients and fractions.
    The fit maximizes the likelihood of the jump distances themselves,
    so the result does not depend on a histogram binning (Martens, et
    al. Frontiers in Bioinformatics, 2021, fit the histogram).

    A single time lag cannot tell the localization error from
    diffusion, since both broaden the distribution in the same way.
    ``sigma`` is therefore fixed: with the default of 0, the fitted
    coefficients are apparent ones, ``D + sigma^2 / dt``. The median
    localization precision is a common choice for ``sigma``
    (``median_precision``), but it is a lower bound of the real error
    of moving particles.

    Parameters
    ----------
    jumps : lib.FloatArray1D
        Jump distances (nm), e.g., from :func:`jump_distances`. Zero
        distances are left out, since they cannot come from diffusion.
    frame_time : float
        Time between two consecutive frames (ms).
    n_populations : int, optional
        Number of populations, at least 1. Default is 1.
    sigma : float, optional
        Localization error (nm). Default is 0 (apparent diffusion
        coefficients).
    r_max : float, optional
        Maximum jump distance of the tracking (nm). If given, the
        distribution is truncated there, which corrects for the jumps
        of fast particles that were too long to be linked; the fractions
        then refer to all jumps, including the missed ones. Default is
        None (not truncated).

    Returns
    -------
    fit : dict
        ``"D"``: diffusion coefficients (um^2/s), sorted ascending,
        ``"fractions"``: fractions of the jumps of every population,
        ``"n_populations"``, ``"sigma"`` (nm), ``"frame_time"`` (ms),
        ``"r_max"`` (nm or None), ``"n_jumps"`` (used),
        ``"n_zero_jumps"`` (left out), ``"log_likelihood"`` (of the
        distances in um), ``"aic"`` and ``"bic"`` (for choosing the
        number of populations: lower is better), ``"success"`` and
        ``"message"`` of the optimizer.

    Raises
    ------
    ValueError
        If the parameters are invalid or there are too few jumps for
        the number of populations.
    """
    if int(n_populations) != n_populations or n_populations < 1:
        raise ValueError(
            f"n_populations must be a positive integer, got {n_populations}."
        )
    k = int(n_populations)
    dt = _positive(frame_time, "frame_time") / 1000
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")
    if r_max is not None:
        r_max = _positive(r_max, "r_max")
    r = np.asarray(jumps, dtype=np.float64) / 1000
    r = r[np.isfinite(r)]
    n_zero = int((r <= 0).sum())
    r = r[r > 0]
    if r_max is not None:
        r = r[r <= r_max / 1000]
    n = len(r)
    n_parameters = 2 * k - 1
    if n < MIN_JUMPS_PER_PARAMETER * n_parameters:
        raise ValueError(
            f"{n} jump distances are too few to fit {k} population"
            f"{'s' if k > 1 else ''} (at least "
            f"{MIN_JUMPS_PER_PARAMETER * n_parameters} needed)."
        )
    r2 = r**2
    sum_log_2r = np.log(2 * r).sum()
    s0 = 4 * (sigma / 1000) ** 2
    r_max_2 = None if r_max is None else (r_max / 1000) ** 2
    log_d_bounds = (np.log(D_BOUNDS[0]), np.log(D_BOUNDS[1]))

    def clip_D(D):
        return np.clip(D, D_BOUNDS[0] * 10, D_BOUNDS[1] / 10)

    # starting points: the apparent D of all jumps, spread over
    # populations, and the D of equal-count chunks of the sorted jumps
    D_all = clip_D((r2.mean() / 4 - s0 / 4) / dt)
    starts = []
    if k == 1:
        starts.append(np.log([D_all]))
    else:
        chunks = np.array_split(np.sort(r2), k)
        starts.append(
            np.log(clip_D([(c.mean() / 4 - s0 / 4) / dt for c in chunks]))
        )
        for low, high in ((-1, 1), (-2, 0.5)):
            starts.append(np.log(clip_D(D_all * np.logspace(low, high, k))))
    bounds = [log_d_bounds] * k + [(-20.0, 20.0)] * (k - 1)
    best = None
    for log_D in starts:
        theta0 = np.concatenate((log_D, np.zeros(k - 1)))
        result = minimize(
            _jd_objective,
            theta0,
            args=(r2, sum_log_2r, k, dt, s0, r_max_2),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or result.fun < best.fun:
            best = result
    D = np.exp(best.x[:k])
    fractions = _softmax(np.concatenate(([0.0], best.x[k:])))
    order = np.argsort(D)
    log_likelihood = -float(best.fun) * n
    return {
        "n_populations": k,
        "D": D[order],
        "fractions": fractions[order],
        "sigma": sigma,
        "frame_time": dt * 1000,
        "r_max": r_max,
        "n_jumps": n,
        "n_zero_jumps": n_zero,
        "log_likelihood": log_likelihood,
        "aic": 2 * n_parameters - 2 * log_likelihood,
        "bic": n_parameters * np.log(n) - 2 * log_likelihood,
        "success": bool(best.success),
        "message": str(best.message),
    }


def jump_distance_pdf(
    r: lib.FloatArray1D, fit: dict
) -> tuple[lib.FloatArray1D, lib.FloatArray2D]:
    """Probability density of the jump distances under a fit.

    Parameters
    ----------
    r : lib.FloatArray1D
        Jump distances (nm).
    fit : dict
        Result of :func:`fit_jump_distances`.

    Returns
    -------
    total : lib.FloatArray1D
        Density of the mixture (per nm).
    components : lib.FloatArray2D
        ``(n_populations, len(r))`` density of every population,
        weighted by its fraction (per nm); they sum to ``total``.
    """
    r_um = np.asarray(r, dtype=np.float64) / 1000
    dt = fit["frame_time"] / 1000
    s = 4 * dt * np.asarray(fit["D"]) + 4 * (fit["sigma"] / 1000) ** 2
    a = np.asarray(fit["fractions"])
    components = (
        a[:, None]
        * 2
        * r_um[None, :]
        / s[:, None]
        * np.exp(-(r_um[None, :] ** 2) / s[:, None])
    )
    if fit["r_max"] is not None:
        r_max = fit["r_max"] / 1000
        components /= (a * (1 - np.exp(-(r_max**2) / s))).sum()
        components[:, r_um > r_max] = 0
    components /= 1000  # per um to per nm
    return components.sum(axis=0), components


# --------------------------------------------------------------------
# Per-track table and full analysis
# --------------------------------------------------------------------


def track_table(
    locs: pd.DataFrame,
    info: list[dict] | None = None,
    *,
    frame_time: float | None = None,
    n_fit: int = N_FIT,
) -> pd.DataFrame:
    """Summary of every track, one row per track.

    Parameters
    ----------
    locs : pd.DataFrame
        Tracked localizations (track ids in ``group``).
    info : list of dicts, optional
        Metadata, used for ``frame_time`` if not given. Default is None.
    frame_time : float, optional
        Time between two consecutive frames (ms). Default is None (from
        the metadata).
    n_fit : int, optional
        Number of time lags of the MSD fitted for the diffusion
        coefficient of every track, see :func:`fit_msd`. Default is 4.

    Returns
    -------
    table : pd.DataFrame
        Columns ``group`` (track id), ``group_input`` (if the
        localizations have it), ``n_locs``, ``first_frame``,
        ``last_frame``, ``n_frames`` (from the first to the last frame,
        skipped frames included), ``duration_s``, ``x_mean`` and
        ``y_mean`` (nm), ``n_jumps`` (between consecutive
        frames), ``mean_jump_nm`` and ``D_um2_s`` (from the MSD of the
        track). The last two are NaN where undefined: no jump, or not
        a single pair of localizations at one of the fitted lags (a
        track without skipped frames needs ``n_fit + 1``
        localizations). Save it with ``DataFrame.to_csv``;
        ``io.save_locs`` removes rows with NaN.
    """
    frame_time = _frame_time(frame_time, info)
    order, group, frame, x, y, starts = _sorted_tracks(locs)
    first = starts[:-1]
    n_locs = np.diff(starts)
    n_tracks = len(n_locs)
    first_frame = frame[first]
    last_frame = frame[starts[1:] - 1]
    if n_tracks:
        x_mean = np.add.reduceat(x, first) / n_locs
        y_mean = np.add.reduceat(y, first) / n_locs
    else:
        x_mean = y_mean = np.zeros(0)
    d, track_index = _jumps(locs)
    n_jumps = np.bincount(track_index, minlength=n_tracks)
    sum_jumps = np.bincount(track_index, weights=d, minlength=n_tracks)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_jump = np.where(n_jumps > 0, sum_jumps / n_jumps, np.nan)
    sums, counts = _msd_kernel(starts, frame, x, y, int(n_fit))
    with np.errstate(invalid="ignore", divide="ignore"):
        track_msd = sums / counts * 1e-6  # nm^2 to um^2
    tau = np.arange(1, int(n_fit) + 1) * frame_time / 1000
    D = fit_msd(tau, track_msd.reshape(n_tracks, int(n_fit)), n_fit)["D"]

    columns = {"group": group[first].astype(np.int32)}
    if "group_input" in locs.columns:
        group_input = locs["group_input"].to_numpy()[order]
        columns["group_input"] = group_input[first]
    columns.update(
        {
            "n_locs": n_locs.astype(np.int32),
            "first_frame": first_frame.astype(np.int64),
            "last_frame": last_frame.astype(np.int64),
            "n_frames": (last_frame - first_frame + 1).astype(np.int64),
            "duration_s": (last_frame - first_frame) * frame_time / 1000,
            "x_mean": x_mean.astype(np.float32),
            "y_mean": y_mean.astype(np.float32),
            "n_jumps": n_jumps.astype(np.int32),
            "mean_jump_nm": mean_jump,
            "D_um2_s": np.asarray(D, dtype=np.float64),
        }
    )
    return pd.DataFrame(columns)


def _resolve_sigma(
    sigma: float | Literal["precision"],
    locs: pd.DataFrame,
) -> tuple[float, str]:
    """The localization error (nm) for the jump distance fit and its
    source ("none", "median precision" or "custom")."""
    if isinstance(sigma, str):
        if sigma != "precision":
            raise ValueError(
                f"sigma must be a number or 'precision', got {sigma!r}."
            )
        if "lpx" not in locs.columns or "lpy" not in locs.columns:
            raise ValueError(
                "sigma='precision' needs the localization precision "
                "(lpx and lpy columns)."
            )
        value = median_precision(locs)
        if not np.isfinite(value):
            raise ValueError(
                "The median localization precision is not finite."
            )
        return value, "median precision"
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")
    return sigma, "none" if sigma == 0 else "custom"


def analyze_tracks(
    locs: pd.DataFrame,
    info: list[dict] | None,
    *,
    frame_time: float | None = None,
    max_lag: int = MAX_LAG,
    n_fit: int = N_FIT,
    sigma: float | Literal["precision"] = 0.0,
    n_populations: int | Sequence[int] = (1, 2, 3),
    truncate: bool = False,
    progress: lib.Progress | Literal["console"] | None = None,
) -> dict:
    """Diffusion analysis of tracked localizations: per-track summary,
    MSD and jump distances.

    Parameters
    ----------
    locs : pd.DataFrame
        Tracked localizations (track ids in ``group``).
    info : list of dicts or None
        Metadata of ``locs``, used for ``frame_time`` and, with
        ``truncate``, the maximum jump distance of the tracking.
    frame_time : float, optional
        Time between two consecutive frames (ms). Default is None (from
        the metadata).
    max_lag : int, optional
        Largest time lag of the MSD (frames). Default is 10.
    n_fit : int, optional
        Number of time lags fitted in the MSD analysis, at most
        ``max_lag``. Default is 4.
    sigma : float or "precision", optional
        Localization error (nm) for the jump distance fit, see
        :func:`fit_jump_distances`. "precision" uses the median
        localization precision. Default is 0 (apparent diffusion
        coefficients).
    n_populations : int or sequence of ints, optional
        Number(s) of populations fitted to the jump distances. Default
        is (1, 2, 3).
    truncate : bool, optional
        Whether the jump distance fit corrects for the maximum jump
        distance of the tracking, see ``r_max`` in
        :func:`fit_jump_distances`. Default is False.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Computing the MSD and jump
        distances" over 2 steps and "Fitting jump distances" over the
        numbers of populations. Default is None.

    Returns
    -------
    result : dict
        ``"table"``: :func:`track_table`, ``"msd"``: :func:`msd`,
        ``"msd_fit"``: :func:`fit_msd` of the ensemble MSD, ``"jumps"``:
        jump distances (nm), ``"jd_fits"``: {number of populations:
        :func:`fit_jump_distances` result, or None if it could not be
        fitted}, ``"jd_errors"``: {number of populations: reason},
        ``"parameters"``: the parameters used (for the metadata),
        ``"n_tracks"``, ``"n_locs"``.

    Raises
    ------
    ValueError
        If the localizations are not tracked or a parameter is invalid.
    """
    frame_time = _frame_time(frame_time, info)
    if int(n_fit) != n_fit or n_fit < 2:
        raise ValueError(
            f"n_fit must be an integer of at least 2, got {n_fit}."
        )
    if n_fit > max_lag:
        raise ValueError(
            f"n_fit ({n_fit}) cannot be larger than max_lag ({max_lag})."
        )
    if np.isscalar(n_populations):
        n_populations = [n_populations]
    n_populations = [int(k) for k in n_populations]
    sigma_nm, sigma_source = _resolve_sigma(sigma, locs)
    r_max = None
    if truncate:
        r_max = max_jump_nm(info)
        if r_max is None:
            raise ValueError(
                "truncate=True needs the maximum jump distance of the "
                "tracking in the metadata."
            )

    with lib.tracked_progress(
        progress, "Analyzing tracks", unit="step"
    ) as progress:
        lib.start_phase(
            progress, "Computing the MSD and jump distances", 2, unit="step"
        )
        table = track_table(locs, frame_time=frame_time, n_fit=n_fit)
        msd_result = msd(locs, frame_time=frame_time, max_lag=max_lag)
        msd_fit = fit_msd(msd_result["tau"], msd_result["msd"], n_fit)
        progress.set_value(1)
        jumps = jump_distances(locs)
        progress.set_value(2)
        lib.start_phase(
            progress, "Fitting jump distances", len(n_populations), unit="fit"
        )
        jd_fits = {}
        jd_errors = {}
        for i, k in enumerate(n_populations):
            try:
                jd_fits[k] = fit_jump_distances(
                    jumps, frame_time, k, sigma=sigma_nm, r_max=r_max
                )
            except ValueError as error:
                jd_fits[k] = None
                jd_errors[k] = str(error)
            progress.set_value(i + 1)
        progress.set_value(len(n_populations))

    parameters = {
        FRAME_TIME_KEY: frame_time,
        "Max. lag (frames)": int(max_lag),
        "MSD fit points": int(n_fit),
        "Localization error": sigma_source,
    }
    if sigma_source != "none":
        parameters["Localization error (nm)"] = sigma_nm
    if r_max is not None:
        parameters["Max. jump correction (nm)"] = r_max
    return {
        "table": table,
        "msd": msd_result,
        "msd_fit": msd_fit,
        "jumps": jumps,
        "jd_fits": jd_fits,
        "jd_errors": jd_errors,
        "parameters": parameters,
        "n_tracks": len(table),
        "n_locs": len(locs),
    }


def _plain(value: Any) -> Any:
    """``value`` with numpy scalars and arrays converted to Python
    types, e.g., for YAML."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def summarize_analysis(result: dict) -> dict:
    """Summary of :func:`analyze_tracks` as plain Python types, e.g.,
    for saving as YAML.

    Parameters
    ----------
    result : dict
        Result of :func:`analyze_tracks`.

    Returns
    -------
    summary : dict
        Parameters, numbers of tracks, localizations and jumps, the MSD
        fit and the jump distance fits.
    """
    summary = lib.step_info("SPT analysis")
    summary.update(result["parameters"])
    summary["Number of tracks"] = result["n_tracks"]
    summary["Number of localizations"] = result["n_locs"]
    summary["Number of jumps"] = len(result["jumps"])
    msd_fit = result["msd_fit"]
    summary["MSD fit"] = {
        "D (um^2/s)": msd_fit["D"],
        "Apparent localization error (nm)": msd_fit["sigma"],
    }
    fits = {}
    for k, fit in result["jd_fits"].items():
        if fit is None:
            fits[f"{k} populations"] = result["jd_errors"].get(k, "failed")
            continue
        fits[f"{k} populations"] = {
            "D (um^2/s)": fit["D"],
            "Fractions": fit["fractions"],
            "Jumps used": fit["n_jumps"],
            "Log-likelihood": fit["log_likelihood"],
            "AIC": fit["aic"],
            "BIC": fit["bic"],
            "Converged": fit["success"],
        }
    summary["Jump distance fits"] = fits
    return _plain(summary)


def save_analysis(base: str, result: dict) -> list[str]:
    """Save the results of :func:`analyze_tracks`.

    Parameters
    ----------
    base : str
        Path without extension, e.g., ``"locs_tracks"``.
    result : dict
        Result of :func:`analyze_tracks`.

    Returns
    -------
    paths : list of str
        The files written: ``<base>_track_stats.csv`` (one row per
        track), ``<base>_msd.csv`` (ensemble MSD) and
        ``<base>_diffusion.yaml`` (parameters and fits).
    """
    table_path = f"{base}_track_stats.csv"
    result["table"].to_csv(table_path, index=False)
    m = result["msd"]
    msd_path = f"{base}_msd.csv"
    pd.DataFrame(
        {
            "lag_frames": m["lag"],
            "tau_s": m["tau"],
            "msd_um2": m["msd"],
            "sem_um2": m["sem"],
            "n_pairs": m["n_pairs"],
            "n_tracks": m["n_tracks"],
        }
    ).to_csv(msd_path, index=False)
    summary_path = f"{base}_diffusion.yaml"
    io.save_info(summary_path, [summarize_analysis(result)])
    return [os.path.abspath(p) for p in (table_path, msd_path, summary_path)]


# --------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------


def _new_figure(fig: plt.Figure | None) -> plt.Figure:
    """``fig`` cleared, or a new figure."""
    if fig is None:
        return plt.Figure(constrained_layout=True)
    fig.clear()
    return fig


def _colors() -> list:
    """Series colors of the active matplotlib style."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_msd(
    msd_result: dict,
    msd_fit: dict | None = None,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot the ensemble MSD and its fit.

    Parameters
    ----------
    msd_result : dict
        Result of :func:`msd`.
    msd_fit : dict, optional
        Result of :func:`fit_msd` of the ensemble MSD. Default is None
        (no fit drawn).
    fig : plt.Figure, optional
        Figure to plot on, cleared first. Default is None (a new
        figure).

    Returns
    -------
    fig : plt.Figure
    """
    fig = _new_figure(fig)
    ax = fig.add_subplot(111)
    data_color, fit_color = _colors()[:2]
    tau = msd_result["tau"]
    sem = np.nan_to_num(msd_result["sem"], nan=0.0)
    ax.errorbar(
        tau,
        msd_result["msd"],
        yerr=sem,
        fmt="o",
        color=data_color,
        capsize=3,
        label="Data",
    )
    title = "Ensemble MSD"
    if msd_fit is not None and np.isfinite(msd_fit["D"]):
        t = np.linspace(0, tau[-1], 100)
        ax.plot(
            t,
            4 * msd_fit["D"] * t + msd_fit["intercept"],
            color=fit_color,
            label=f"Fit (first {msd_fit['n_fit']} lags)",
        )
        title += f", D = {msd_fit['D']:.3g} µm²/s"
        if np.isfinite(msd_fit["sigma"]):
            title += f", σ = {msd_fit['sigma']:.1f} nm"
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.set_xlabel("Time lag (s)")
    ax.set_ylabel("MSD (µm²)")
    ax.legend(loc="best")
    return fig


def plot_jump_distances(
    jumps: lib.FloatArray1D,
    fit: dict | None = None,
    fig: plt.Figure | None = None,
    fill: bool = True,
    outline: bool = False,
    bins: lib.FloatArray1D | None = None,
) -> plt.Figure:
    """Plot the histogram of the jump distances and a fit.

    Parameters
    ----------
    jumps : lib.FloatArray1D
        Jump distances (nm).
    fit : dict, optional
        Result of :func:`fit_jump_distances`. Default is None (no fit
        drawn).
    fig : plt.Figure, optional
        Figure to plot on, cleared first. Default is None (a new
        figure).
    fill, outline : bool, optional
        Whether the histogram bars are filled and/or outlined, see
        ``lib.histogram_style``. Default is filled without an outline.
    bins : lib.FloatArray1D, optional
        Bin edges (nm). Default is None (``lib.calculate_optimal_bins``).

    Returns
    -------
    fig : plt.Figure
    """
    fig = _new_figure(fig)
    ax = fig.add_subplot(111)
    colors = _colors()
    jumps = np.asarray(jumps, dtype=np.float64)
    if bins is None:
        bins = lib.calculate_optimal_bins(jumps, max_n_bins=200)
        bins = np.maximum(bins, 0)
    counts, edges = np.histogram(jumps, bins=bins)
    ax.stairs(
        counts,
        edges,
        label="Data",
        **lib.histogram_style(colors[0], fill, outline),
    )
    title = f"Jump distances (1 frame), n = {len(jumps):,}"
    if fit is not None and len(edges) > 1:
        r = np.linspace(0, edges[-1], 400)
        total, components = jump_distance_pdf(r, fit)
        # expected counts per bin, from the jumps the fit used
        scale = fit["n_jumps"] * np.diff(edges).mean()
        k = fit["n_populations"]
        ax.plot(r, scale * total, color=colors[1 % len(colors)], label="Fit")
        if k > 1:
            for i in range(k):
                ax.plot(
                    r,
                    scale * components[i],
                    color=colors[(2 + i) % len(colors)],
                    linestyle="--",
                    label=(
                        f"D = {fit['D'][i]:.3g} µm²/s "
                        f"({100 * fit['fractions'][i]:.0f}%)"
                    ),
                )
        else:
            title += f", D = {fit['D'][0]:.3g} µm²/s"
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.set_xlabel("Jump distance (nm)")
    ax.set_ylabel("Counts")
    ax.legend(loc="best")
    return fig


def plot_track_lengths(
    table: pd.DataFrame,
    fig: plt.Figure | None = None,
    fill: bool = True,
    outline: bool = False,
) -> plt.Figure:
    """Plot the histogram of the numbers of localizations per track.

    Parameters
    ----------
    table : pd.DataFrame
        Result of :func:`track_table`.
    fig : plt.Figure, optional
        Figure to plot on, cleared first. Default is None (a new
        figure).
    fill, outline : bool, optional
        Whether the histogram bars are filled and/or outlined, see
        ``lib.histogram_style``. Default is filled without an outline.

    Returns
    -------
    fig : plt.Figure
    """
    fig = _new_figure(fig)
    ax = fig.add_subplot(111)
    n_locs = table["n_locs"].to_numpy()
    if len(n_locs):
        high = int(n_locs.max())
        step = max(1, int(np.ceil((high - n_locs.min() + 1) / 200)))
        edges = np.arange(n_locs.min() - 0.5, high + step + 0.5, step)
    else:
        edges = np.array([0.5, 1.5])
    counts, edges = np.histogram(n_locs, bins=edges)
    ax.stairs(
        counts, edges, **lib.histogram_style(_colors()[0], fill, outline)
    )
    mean = n_locs.mean() if len(n_locs) else 0
    ax.set_title(
        f"Track lengths, {len(n_locs):,} tracks, mean {mean:.1f} "
        "localizations"
    )
    ax.set_xlabel("Localizations per track")
    ax.set_ylabel("Counts")
    return fig
