"""
picasso.stitch
~~~~~~~~~~~~~~

Stitching localizations of tiles acquired at different stage positions
into one dataset.

A large field of view is often imaged as a grid of tiles: the stage
moves between acquisitions and the same camera records each tile. The
localizations of every tile are therefore in the same camera frame,
``[0, Width) x [0, Height)``, and only the stage position tells where a
tile lies in the sample. Stitching translates every tile to its place
and merges the tiles, such that neighboring tiles overlap by the few
microns they share:

1. **Initial positions.** The Micro-Manager stage position of each file
   (``XPositionUm`` and ``YPositionUm``) is converted to camera pixels.
   The stage axes may be flipped or swapped relative to the camera axes
   (see :data:`ORIENTATIONS`); by default the orientation whose
   overlaps match best is chosen. Positions may instead be given
   directly, e.g., from tiles placed by hand.
2. **Pairwise matching.** For every pair of tiles predicted to overlap,
   the localizations in the overlap are rendered and compared by
   normalized cross-correlation within a search radius (stage positions
   are typically off by a few microns). Of the highest correlation
   peaks, the one where most localizations find a partner in the other
   tile is refined to sub-pixel precision by matching nearest
   neighbors. For 3D data, the matched pairs also give the z offset
   between the tiles.
3. **Global positions.** The pairwise offsets are combined by weighted
   least squares, similar to Preibisch, et al. Bioinformatics, 2009.
   DOI: 10.1093/bioinformatics/btp184. While the offsets disagree, the
   worst-matching offset of the disagreeing ones is dropped.
4. **Merging.** The tiles of each target are translated and merged into
   one localization table. Optionally, each overlap is cut at the seam
   (the localizations of the tile whose center is closest are kept), so
   that no molecule is counted twice, and the frames of consecutive
   tiles are offset, so that they stay unique.

Several files recorded at the same stage position (e.g., the targets of
an Exchange-PAINT experiment) form one tile: they are aligned together
and the tiles of each target are merged separately.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import copy
import datetime
import warnings
from typing import Any, Hashable

import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.spatial import KDTree

from . import lib, __version__

#: Orientations of the camera relative to the stage, which depend on
#: how the camera is mounted: in which direction of the image a tile
#: lies whose stage X (or Y) position is larger. The keys give the image
#: x and y axes in terms of the stage X and Y axes; e.g., ``"-x, -y"``
#: means that a tile at a larger stage X lies to the left (-x) in the
#: image and one at a larger stage Y lies above it (-y; the image y
#: axis points down), and ``"y, x"`` that the stage X axis runs along
#: the image y axis. :func:`describe_orientation` says it in words. Each
#: value is the matrix that maps stage to image coordinates.
ORIENTATIONS = {
    "x, y": ((1, 0), (0, 1)),
    "-x, y": ((-1, 0), (0, 1)),
    "x, -y": ((1, 0), (0, -1)),
    "-x, -y": ((-1, 0), (0, -1)),
    "y, x": ((0, 1), (1, 0)),
    "-y, x": ((0, -1), (1, 0)),
    "y, -x": ((0, 1), (-1, 0)),
    "-y, -x": ((0, -1), (-1, 0)),
}

# image direction of each unit vector (the image y axis points down)
_DIRECTIONS = {(1, 0): "right", (-1, 0): "left", (0, 1): "down", (0, -1): "up"}


def describe_orientation(orientation: str) -> str:
    """Say in words where tiles at larger stage positions lie in the
    image.

    Parameters
    ----------
    orientation : str
        Key of :data:`ORIENTATIONS`.

    Returns
    -------
    description : str
        E.g., "stage X → left, stage Y → up" for ``"-x, -y"``: a tile
        at a larger stage X lies to the left in the image, one at a
        larger stage Y above.
    """
    matrix = np.array(ORIENTATIONS[orientation], dtype=int)
    x, y = (_DIRECTIONS[tuple(matrix[:, k])] for k in range(2))
    return f"stage X → {x}, stage Y → {y}"


#: Keys of the Micro-Manager stage position (micrometers).
STAGE_KEYS = ("XPositionUm", "YPositionUm")

# rendered pixels per camera pixel for the cross-correlation
_OVERSAMPLING = 2
# pairs scored when choosing the orientation (largest overlaps first)
_MAX_ORIENTATION_PAIRS = 8
# nearest neighbor matching radii, in rendered pixels; the last one is
# repeated until the offset converges
_MATCH_RADII = (3.0, 1.5, 0.75)
_MATCH_TOLERANCE = 1e-3  # camera pixels
_MATCH_MAX_ITER = 20
# correlation peaks tried per pair, each scored by the fraction of a
# sample of localizations that find a partner in the other tile
_N_PEAKS = 5
_MATCH_SAMPLE = 20_000
_MIN_OVERLAP_POINTS = 20


def stage_positions(infos: list[list[dict]]) -> np.ndarray | None:
    """Read the Micro-Manager stage position of each file.

    Parameters
    ----------
    infos : list of lists of dicts
        Metadata of each file.

    Returns
    -------
    positions : np.ndarray or None
        Stage position ``(X, Y)`` of each file in micrometers, shape
        ``(n_files, 2)``. None if any file lacks it, e.g., because the
        Micro-Manager metadata was not saved with the localizations.
    """
    positions = []
    for info in infos:
        mm = lib.get_from_metadata(info, "Micro-Manager Metadata")
        if not isinstance(mm, dict):
            return None
        try:
            positions.append([float(mm[key]) for key in STAGE_KEYS])
        except (KeyError, TypeError, ValueError):
            return None
    return np.array(positions, dtype=float).reshape(-1, 2)


def group_tiles(
    positions: np.ndarray | None, n_files: int, tolerance_um: float = 1.0
) -> np.ndarray:
    """Group files recorded at the same stage position into tiles.

    Parameters
    ----------
    positions : np.ndarray or None
        Stage position of each file in micrometers, see
        :func:`stage_positions`. If None, each file is its own tile.
    n_files : int
        Number of files.
    tolerance_um : float, optional
        Files whose positions differ by at most this much along both
        axes are in the same tile. Default is 1.0.

    Returns
    -------
    tiles : np.ndarray
        Tile index of each file, numbered in the order of first
        appearance.
    """
    if positions is None:
        return np.arange(n_files)
    tiles = np.empty(n_files, dtype=int)
    centers = []
    for i, position in enumerate(positions):
        for t, center in enumerate(centers):
            if np.all(np.abs(position - center) <= tolerance_um):
                tiles[i] = t
                break
        else:
            tiles[i] = len(centers)
            centers.append(position)
    return tiles


def stage_to_camera(
    positions_um: np.ndarray, pixelsize: float, orientation: str
) -> np.ndarray:
    """Convert stage positions into tile positions in camera pixels.

    Parameters
    ----------
    positions_um : np.ndarray
        Stage position of each tile in micrometers, shape
        ``(n_tiles, 2)``.
    pixelsize : float
        Camera pixel size in nm.
    orientation : str
        Key of :data:`ORIENTATIONS`.

    Returns
    -------
    positions : np.ndarray
        Position of each tile in camera pixels, relative to the first
        tile, shape ``(n_tiles, 2)``.
    """
    matrix = np.array(ORIENTATIONS[orientation], dtype=float)
    relative = np.asarray(positions_um, dtype=float) - positions_um[0]
    return relative @ matrix.T * 1e3 / pixelsize


def camera_size(info: list[dict]) -> tuple[int, int]:
    """Size ``(width, height)`` of the camera image of a file in camera
    pixels. Render stores it under ``lib.CAMERA_SIZE_KEYS`` once the
    canvas (``Width`` and ``Height``) was fitted to moved
    localizations, see ``lib.fit_canvas``."""
    size = []
    for camera_key, key in zip(lib.CAMERA_SIZE_KEYS, ("Width", "Height")):
        value = lib.get_from_metadata(info, camera_key)
        if value is None:
            value = lib.get_from_metadata(info, key, raise_error=True)
        size.append(int(value))
    return size[0], size[1]


def overlapping_pairs(
    positions: np.ndarray, sizes: np.ndarray, min_overlap: float
) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """Find the pairs of tiles that overlap.

    Parameters
    ----------
    positions : np.ndarray
        Position of each tile in camera pixels, shape ``(n_tiles, 2)``.
    sizes : np.ndarray
        Size ``(width, height)`` of each tile in camera pixels.
    min_overlap : float
        Minimum overlap along both axes in camera pixels.

    Returns
    -------
    pairs : list of tuples
        ``(a, b, (x0, x1, y0, y1))`` with ``a < b`` and the overlap
        rectangle.
    """
    pairs = []
    n = len(positions)
    for a in range(n):
        for b in range(a + 1, n):
            x0 = max(positions[a, 0], positions[b, 0])
            x1 = min(
                positions[a, 0] + sizes[a, 0], positions[b, 0] + sizes[b, 0]
            )
            y0 = max(positions[a, 1], positions[b, 1])
            y1 = min(
                positions[a, 1] + sizes[a, 1], positions[b, 1] + sizes[b, 1]
            )
            if x1 - x0 >= min_overlap and y1 - y0 >= min_overlap:
                pairs.append((a, b, (x0, x1, y0, y1)))
    return pairs


def _in_rect(
    x: np.ndarray, y: np.ndarray, rect: tuple[float, float, float, float]
) -> np.ndarray:
    """Mask of the points inside the rectangle ``(x0, x1, y0, y1)``."""
    x0, x1, y0, y1 = rect
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


def _render(
    x: np.ndarray, y: np.ndarray, x0: float, y0: float, shape: tuple
) -> np.ndarray:
    """Histogram of points in camera pixels starting at ``(x0, y0)``,
    with ``_OVERSAMPLING`` bins per camera pixel, blurred by one bin."""
    image, _, _ = np.histogram2d(
        (y - y0) * _OVERSAMPLING,
        (x - x0) * _OVERSAMPLING,
        bins=shape,
        range=[[0, shape[0]], [0, shape[1]]],
    )
    return ndimage.gaussian_filter(image, 1.0)


def _peak_subpixel(c: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """Parabolic sub-bin refinement of the peak of ``c`` at
    ``(iy, ix)``."""
    offsets = []
    for axis, i in ((0, iy), (1, ix)):
        if 0 < i < c.shape[axis] - 1:
            idx = [iy, ix]
            idx[axis] = i - 1
            left = c[tuple(idx)]
            idx[axis] = i + 1
            right = c[tuple(idx)]
            center = c[iy, ix]
            denominator = left - 2 * center + right
            offsets.append(
                0.5 * (left - right) / denominator if denominator else 0.0
            )
        else:
            offsets.append(0.0)
    return iy + offsets[0], ix + offsets[1]


def _window_sums(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sum of ``image`` in every window of ``shape`` that fits inside
    it (as ``mode="valid"``), from cumulative sums."""
    padded = np.zeros((image.shape[0] + 1, image.shape[1] + 1))
    padded[1:, 1:] = image.cumsum(0).cumsum(1)
    h, w = shape
    return (
        padded[h:, w:] - padded[:-h, w:] - padded[h:, :-w] + padded[:-h, :-w]
    )


def _normalized_xcorr(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of ``image_b`` with every window of
    the larger ``image_a`` it fits in (Lewis, Vision Interface, 1995).

    Unlike the plain cross-correlation, it does not favor placing
    ``image_b`` over dense regions of ``image_a``.

    Returns
    -------
    ncc : np.ndarray
        Correlation coefficient (-1 to 1) of each window; element
        ``k`` compares ``image_b`` with ``image_a`` shifted by ``k``.
        0 where the window of ``image_a`` is empty.
    """
    b = image_b - image_b.mean()
    norm_b = np.sqrt(np.sum(b**2))
    numerator = signal.fftconvolve(image_a, b[::-1, ::-1], mode="valid")
    n = b.size
    sums = _window_sums(image_a, b.shape)
    squares = _window_sums(image_a**2, b.shape)
    variance = np.maximum(squares - sums**2 / n, 0.0)
    denominator = np.sqrt(variance) * norm_b
    # empty windows (and round-off where there is no signal)
    valid = variance > 1e-9 * max(variance.max(), 1e-300)
    ncc = np.zeros_like(numerator)
    ncc[valid] = numerator[valid] / denominator[valid]
    return ncc


def _coarse_offsets(
    a: np.ndarray,
    b: np.ndarray,
    rect: tuple[float, float, float, float],
    search: float,
    n_peaks: int = 1,
) -> list[tuple[float, float, float]]:
    """Cross-correlate the overlap of two tiles within a search radius.

    Parameters
    ----------
    a, b : np.ndarray
        Coordinates of the localizations of both tiles, shape
        ``(n, 2+)``, at their current positions (camera pixels).
    rect : tuple
        Predicted overlap ``(x0, x1, y0, y1)``.
    search : float
        Search radius in camera pixels.
    n_peaks : int, optional
        Number of correlation peaks returned. Default is 1.

    Returns
    -------
    peaks : list of tuples
        ``(dx, dy, score)`` of the highest peaks, highest first:
        correction of the position of ``b`` relative to ``a`` (camera
        pixels) and the normalized cross-correlation (correlation
        coefficient). Empty if either tile has no localizations in the
        overlap or nothing correlates.
    """
    x0, x1, y0, y1 = rect
    rect_a = (x0 - search, x1 + search, y0 - search, y1 + search)
    in_a = _in_rect(a[:, 0], a[:, 1], rect_a)
    in_b = _in_rect(b[:, 0], b[:, 1], rect)
    if not in_a.any() or not in_b.any():
        return []
    shape_b = (
        max(1, int(round((y1 - y0) * _OVERSAMPLING))),
        max(1, int(round((x1 - x0) * _OVERSAMPLING))),
    )
    margin = int(np.ceil(search * _OVERSAMPLING))
    shape_a = (shape_b[0] + 2 * margin, shape_b[1] + 2 * margin)
    image_a = _render(
        a[in_a, 0],
        a[in_a, 1],
        x0 - margin / _OVERSAMPLING,
        y0 - margin / _OVERSAMPLING,
        shape_a,
    )
    image_b = _render(b[in_b, 0], b[in_b, 1], x0, y0, shape_b)
    c = _normalized_xcorr(image_a, image_b)
    # local maxima, at least a camera pixel apart
    size = 2 * _OVERSAMPLING + 1
    is_peak = (c == ndimage.maximum_filter(c, size=size)) & (c > 0)
    iys, ixs = np.nonzero(is_peak)
    order = np.argsort(c[iys, ixs])[::-1][:n_peaks]
    peaks = []
    for iy, ix in zip(iys[order], ixs[order]):
        py, px = _peak_subpixel(c, iy, ix)
        peaks.append(
            (
                (px - margin) / _OVERSAMPLING,
                (py - margin) / _OVERSAMPLING,
                float(c[iy, ix]),
            )
        )
    return peaks


def _fine_offset(
    a: np.ndarray,
    b: np.ndarray,
    rect: tuple[float, float, float, float],
    radii: tuple[float, ...],
) -> tuple[float, float, float | None, int, int]:
    """Refine the offset of two overlapping tiles by matching nearest
    neighbors.

    The localizations of ``b`` are paired with their nearest neighbor
    in ``a`` within each radius in turn, and the median displacement is
    added to the offset; the last radius is repeated until the offset
    converges.

    Parameters
    ----------
    a, b : np.ndarray
        Coordinates of the localizations of both tiles, shape
        ``(n, 2)`` or ``(n, 3)`` with z, at their current positions
        (camera pixels).
    rect : tuple
        Overlap ``(x0, x1, y0, y1)`` at the current positions.
    radii : tuple of floats
        Matching radii in camera pixels, decreasing.

    Returns
    -------
    dx, dy : float
        Correction of the position of ``b`` relative to ``a`` (camera
        pixels).
    dz : float or None
        z offset to add to ``b`` to match ``a`` (in the units of z), or
        None without z.
    n_matches : int
        Number of pairs matched within the last radius.
    n_b : int
        Number of localizations of ``b`` in the overlap.
    """
    margin = radii[0]
    x0, x1, y0, y1 = rect
    a = a[
        _in_rect(
            a[:, 0],
            a[:, 1],
            (x0 - margin, x1 + margin, y0 - margin, y1 + margin),
        )
    ]
    b = b[_in_rect(b[:, 0], b[:, 1], rect)]
    if not len(a) or not len(b):
        return 0.0, 0.0, None, 0, len(b)
    tree = KDTree(a[:, :2])
    shift = np.zeros(2)
    schedule = list(radii[:-1]) + [radii[-1]] * _MATCH_MAX_ITER
    matched = np.zeros(len(b), dtype=bool)
    index = np.zeros(len(b), dtype=int)
    for i, radius in enumerate(schedule):
        distance, index = tree.query(
            b[:, :2] + shift, distance_upper_bound=radius, workers=-1
        )
        matched = np.isfinite(distance)
        if not matched.any():
            return float(shift[0]), float(shift[1]), None, 0, len(b)
        step = np.median(
            a[index[matched], :2] - b[matched, :2] - shift, axis=0
        )
        shift += step
        if i >= len(radii) - 1 and np.all(np.abs(step) < _MATCH_TOLERANCE):
            break
    # the pairs at the final offset
    distance, index = tree.query(
        b[:, :2] + shift, distance_upper_bound=radii[-1], workers=-1
    )
    matched = np.isfinite(distance)
    dz = None
    if a.shape[1] > 2 and b.shape[1] > 2 and matched.any():
        dz = float(np.median(a[index[matched], 2] - b[matched, 2]))
    return float(shift[0]), float(shift[1]), dz, int(matched.sum()), len(b)


def _translated(
    points: np.ndarray,
    position: np.ndarray,
    rect: tuple[float, float, float, float],
    margin: float,
) -> np.ndarray:
    """The ``points`` (camera frame) within ``margin`` of the rectangle
    ``rect`` (x0, x1, y0, y1) once moved to ``position``, moved
    there."""
    x0, x1, y0, y1 = rect
    px, py = position
    inside = _in_rect(
        points[:, 0],
        points[:, 1],
        (
            x0 - margin - px,
            x1 + margin - px,
            y0 - margin - py,
            y1 + margin - py,
        ),
    )
    moved = points[inside].copy()
    moved[:, :2] += position
    return moved


def _count_matches(
    tree: KDTree,
    points: np.ndarray,
    shift: tuple[float, float],
    fov: tuple[float, float, float, float],
    radius: float,
) -> tuple[int, int]:
    """How many ``points``, moved by ``shift`` into the field of view
    ``fov`` (x0, x1, y0, y1) of the tile in ``tree``, have a neighbor
    there within ``radius``.

    Returns
    -------
    n_matches : int
        Points with a neighbor.
    n_inside : int
        Points inside the field of view.
    """
    moved = points[:, :2] + shift
    inside = _in_rect(moved[:, 0], moved[:, 1], fov)
    if not inside.any():
        return 0, 0
    distance, _ = tree.query(
        moved[inside], distance_upper_bound=radius, workers=-1
    )
    return int(np.isfinite(distance).sum()), int(inside.sum())


def match_pairs(
    points: list[np.ndarray],
    positions: np.ndarray,
    sizes: np.ndarray,
    pixelsize: float,
    *,
    search_radius_um: float = 10.0,
    min_overlap_um: float = 1.0,
    precision_nm: float | None = None,
    progress=None,
) -> list[dict]:
    """Measure the offsets between all pairs of overlapping tiles.

    Parameters
    ----------
    points : list of np.ndarray
        Localizations of each tile in its camera frame, shape ``(n, 2)``
        (x and y in camera pixels) or ``(n, 3)`` (and z in nm).
    positions : np.ndarray
        Initial position of each tile in camera pixels, shape
        ``(n_tiles, 2)``.
    sizes : np.ndarray
        Size ``(width, height)`` of each tile in camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    search_radius_um : float, optional
        How far the true offset of two tiles may be from the one given
        by ``positions``, in micrometers. Default is 10.0.
    min_overlap_um : float, optional
        Tiles are matched if their predicted overlap is at least this
        large along both axes, in micrometers. Default is 1.0.
    precision_nm : float, optional
        Typical localization precision in nm. The final matching radius
        is at least three times this. Default is None.
    progress : None, "console" or ProgressType, optional
        Progress tracker, see ``lib.normalize_progress``. Default None.

    Returns
    -------
    pairs : list of dicts
        One per overlapping pair, with the keys ``"tiles"`` (``(a,
        b)``), ``"offset"`` (measured position of ``b`` minus that of
        ``a``, camera pixels), ``"dz"`` (z of ``a`` minus z of ``b`` in
        the overlap in nm, or None), ``"score"`` (normalized
        cross-correlation of the overlap, or None if no localizations
        overlap), ``"matches"`` (localizations of ``b`` in the overlap
        with a partner in ``a`` within the final radius) and
        ``"fraction"`` (``"matches"`` divided by the localizations of
        ``b`` in the overlap).
    """
    progress = lib.normalize_progress(progress, "Matching overlaps")
    pairs = overlapping_pairs(
        positions, sizes, min_overlap_um * 1e3 / pixelsize
    )
    search = search_radius_um * 1e3 / pixelsize
    radii = np.array(_MATCH_RADII) / _OVERSAMPLING
    if precision_nm:
        radii = np.maximum(radii, 3 * precision_nm / pixelsize)
    progress.setMaximum(len(pairs))
    progress.zero_progress("Matching overlaps")
    results = []
    margin = search + 2 * radii[0]
    for i, (a, b, rect) in enumerate(pairs):
        pa = _translated(points[a], positions[a], rect, margin)
        pb = _translated(points[b], positions[b], rect, margin)
        result = {
            "tiles": (a, b),
            "offset": tuple(map(float, positions[b] - positions[a])),
            "dz": None,
            "score": None,
            "matches": 0,
            "fraction": 0.0,
        }
        candidates = _coarse_offsets(pa, pb, rect, search, _N_PEAKS)
        if candidates:
            fov_a = (
                positions[a, 0],
                positions[a, 0] + sizes[a, 0],
                positions[a, 1],
                positions[a, 1] + sizes[a, 1],
            )
            tree = KDTree(pa[:, :2])
            sample = pb
            if len(pb) > _MATCH_SAMPLE:
                rng = np.random.default_rng(0)
                sample = pb[rng.choice(len(pb), _MATCH_SAMPLE, replace=False)]
            counts = np.array(
                [
                    _count_matches(tree, sample, (dx, dy), fov_a, radii[1])
                    for dx, dy, _ in candidates
                ]
            )
            # the peak where the most localizations find a partner;
            # peaks that leave only a sliver of overlap are not trusted
            fractions = counts[:, 0] / np.maximum(counts[:, 1], 1)
            enough = counts[:, 1] >= max(
                _MIN_OVERLAP_POINTS, 0.2 * counts[:, 1].max()
            )
            fractions[~enough] = -1
            best = int(np.argmax(fractions))
            dx, dy, score = candidates[best]
            pb[:, :2] += (dx, dy)
            shifted = positions[b] + (dx, dy)
            # the overlap at the corrected position
            rect = (
                max(positions[a, 0], shifted[0]),
                min(positions[a, 0] + sizes[a, 0], shifted[0] + sizes[b, 0]),
                max(positions[a, 1], shifted[1]),
                min(positions[a, 1] + sizes[a, 1], shifted[1] + sizes[b, 1]),
            )
            fx, fy, dz, n_matches, n_b = _fine_offset(
                pa, pb, rect, tuple(radii)
            )
            offset = shifted + (fx, fy) - positions[a]
            result.update(
                offset=tuple(map(float, offset)),
                dz=dz,
                score=score,
                matches=n_matches,
                fraction=n_matches / n_b if n_b else 0.0,
            )
        results.append(result)
        progress.set_value(i + 1)
    return results


def _components(n: int, links: list[tuple[int, int]]) -> np.ndarray:
    """Connected component (labeled by its smallest tile) of each
    tile."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def _least_squares(
    n: int,
    links: list[tuple[int, int]],
    measured: np.ndarray,
    weights: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Weighted least squares positions from relative measurements
    ``p[b] - p[a] = measured``, with the anchor tiles fixed.

    Parameters
    ----------
    n : int
        Number of tiles.
    links : list of tuples
        ``(a, b)`` of each measurement.
    measured : np.ndarray
        Measurements, shape ``(n_links, n_dims)``.
    weights : np.ndarray
        Weight of each measurement.
    anchors : np.ndarray
        Fixed value of every tile, used for the anchor tiles (those
        whose value is not NaN), shape ``(n, n_dims)``.

    Returns
    -------
    values : np.ndarray
        Solved value of each tile, shape ``(n, n_dims)``.
    """
    fixed = ~np.isnan(anchors[:, 0])
    free = np.flatnonzero(~fixed)
    values = np.where(fixed[:, None], anchors, 0.0)
    if not len(free) or not links:
        return values
    column = -np.ones(n, dtype=int)
    column[free] = np.arange(len(free))
    design = np.zeros((len(links), len(free)))
    rhs = np.array(measured, dtype=float)
    for k, (a, b) in enumerate(links):
        if fixed[b]:
            rhs[k] -= anchors[b]
        else:
            design[k, column[b]] += 1
        if fixed[a]:
            rhs[k] += anchors[a]
        else:
            design[k, column[a]] -= 1
    w = np.sqrt(weights)[:, None]
    solution = np.linalg.lstsq(design * w, rhs * w, rcond=None)[0]
    values[free] = solution
    return values


def solve_positions(
    pairs: list[dict],
    initial: np.ndarray,
    *,
    min_fraction: float = 0.1,
    min_matches: int = 10,
    max_residual: float = 0.25,
    align_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[int]]:
    """Combine pairwise offsets into tile positions.

    The pairs are weighted by their number of matched localizations.
    While a pair disagrees with the solution by more than
    ``max_residual``, the pairs are inconsistent: of the pairs sharing
    the disagreement, the one whose localizations matched worst (lowest
    ``"fraction"``) is dropped and the positions are solved again.
    Tiles not connected to the first tile by any pair keep their
    position relative to the first tile of their group in
    ``initial``.

    Parameters
    ----------
    pairs : list of dicts
        Output of :func:`match_pairs`.
    initial : np.ndarray
        Initial position of each tile in camera pixels, shape
        ``(n_tiles, 2)``.
    min_fraction : float, optional
        Pairs where a smaller fraction of the localizations in the
        overlap found a partner in the other tile are not used. Default
        is 0.1.
    min_matches : int, optional
        Pairs with fewer matched localizations are not used. Default
        is 10.
    max_residual : float, optional
        Largest disagreement of a used pair with the solution in camera
        pixels. Default is 0.25.
    align_z : bool, optional
        If True, the z offsets of the tiles are solved as well, from the
        pairs with a z offset. Default is True.

    Returns
    -------
    positions : np.ndarray
        Position of each tile in camera pixels, shape ``(n_tiles, 2)``.
    z_offsets : np.ndarray
        z offset to add to each tile in nm (zeros if ``align_z`` is
        False or no pair has a z offset).
    pairs : list of dicts
        Copies of the input pairs with the keys ``"used"`` (bool),
        ``"residual"`` (camera pixels, None if not used) and
        ``"status"``: ``"used"``, ``"empty"`` (no localizations in the
        overlap), ``"weak"`` (below ``min_fraction`` or
        ``min_matches``) or
        ``"inconsistent"`` (dropped for disagreeing with the others).
    unconnected : list of ints
        Tiles not connected to the first tile.
    """
    n = len(initial)
    pairs = [dict(pair) for pair in pairs]
    active = [
        k
        for k, pair in enumerate(pairs)
        if pair["score"] is not None
        and pair["fraction"] >= min_fraction
        and pair["matches"] >= min_matches
    ]
    strong = set(active)
    while True:
        links = [pairs[k]["tiles"] for k in active]
        component = _components(n, links)
        anchors = np.full((n, 2), np.nan)
        for root in np.unique(component):
            anchors[root] = initial[root]
        measured = np.array(
            [pairs[k]["offset"] for k in active], dtype=float
        ).reshape(-1, 2)
        weights = np.array([pairs[k]["matches"] for k in active], dtype=float)
        positions = _least_squares(n, links, measured, weights, anchors)
        residuals = {
            k: float(
                np.linalg.norm(
                    positions[b]
                    - positions[a]
                    - np.asarray(pairs[k]["offset"])
                )
            )
            for k, (a, b) in zip(active, links)
        }
        largest = max(residuals.values(), default=0.0)
        if largest <= max_residual:
            break
        # the pairs of an inconsistent loop share its error, weighted
        # such that the most uncertain pair gets the largest share,
        # which need not be the wrong one: drop the pair of the loop
        # whose localizations matched worst
        loop = [k for k in residuals if residuals[k] >= 0.25 * largest]
        active.remove(min(loop, key=lambda k: pairs[k]["fraction"]))

    z_offsets = np.zeros(n)
    if align_z:
        z_active = [k for k in active if pairs[k]["dz"] is not None]
        if z_active:
            z_links = [pairs[k]["tiles"] for k in z_active]
            z_component = _components(n, z_links)
            z_anchors = np.full((n, 1), np.nan)
            z_anchors[np.unique(z_component)] = 0.0
            # dz is z of a minus z of b, i.e., the offset of b
            z_measured = np.array([[pairs[k]["dz"]] for k in z_active])
            z_weights = np.array([pairs[k]["matches"] for k in z_active])
            z_offsets = _least_squares(
                n, z_links, z_measured, z_weights.astype(float), z_anchors
            )[:, 0]

    for k, pair in enumerate(pairs):
        pair["used"] = k in residuals
        pair["residual"] = residuals.get(k)
        if pair["used"]:
            pair["status"] = "used"
        elif pair["score"] is None:
            pair["status"] = "empty"
        elif k in strong:
            pair["status"] = "inconsistent"
        else:
            pair["status"] = "weak"
    unconnected = [int(t) for t in np.flatnonzero(component != component[0])]
    return positions, z_offsets, pairs, unconnected


def seam_mask(
    x: np.ndarray,
    y: np.ndarray,
    tile: int,
    positions: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    """Which localizations of a tile are kept when the overlaps are cut
    at the seams.

    A point is kept by the tile whose center is the closest among the
    tiles whose field of view contains it (ties go to the tile with
    the lower index). On a regular grid this cuts each overlap in its
    middle.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates of the localizations of the tile at its stitched
        position (camera pixels).
    tile : int
        Index of the tile.
    positions : np.ndarray
        Stitched position of each tile in camera pixels.
    sizes : np.ndarray
        Size ``(width, height)`` of each tile in camera pixels.

    Returns
    -------
    keep : np.ndarray
        Boolean mask of the localizations to keep.
    """
    centers = positions + sizes / 2
    own = (x - centers[tile, 0]) ** 2 + (y - centers[tile, 1]) ** 2
    keep = np.ones(len(x), dtype=bool)
    for other in range(len(positions)):
        if other == tile:
            continue
        rect = (
            positions[other, 0],
            positions[other, 0] + sizes[other, 0],
            positions[other, 1],
            positions[other, 1] + sizes[other, 1],
        )
        inside = _in_rect(x, y, rect)
        if not inside.any():
            continue
        distance = (x - centers[other, 0]) ** 2 + (y - centers[other, 1]) ** 2
        closer = distance < own if other > tile else distance <= own
        keep &= ~(inside & closer)
    return keep


def _acquisition_time(info: list[dict]) -> datetime.datetime | None:
    """Start of the acquisition from the Micro-Manager metadata."""
    mm = lib.get_from_metadata(info, "Micro-Manager Metadata")
    if not isinstance(mm, dict) or "ReceivedTime" not in mm:
        return None
    try:
        return datetime.datetime.strptime(
            str(mm["ReceivedTime"]), "%Y-%m-%d %H:%M:%S.%f %z"
        )
    except ValueError:
        return None


def _n_frames(locs: pd.DataFrame, info: list[dict]) -> int:
    """Number of frames of a file, from the metadata or else the
    localizations."""
    frames = lib.get_from_metadata(info, "Frames")
    if frames is None:
        frames = int(locs["frame"].max()) + 1 if len(locs) else 0
    return int(frames)


def _yaml_label(label: Hashable) -> int | str:
    """A target label as a plain int or str for the yaml metadata."""
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        return int(label)
    return str(label)


def _pixelsize(infos: list[list[dict]]) -> float:
    """Common camera pixel size of all files in nm."""
    sizes = {lib.get_from_metadata(info, "Pixelsize") for info in infos}
    if None in sizes:
        raise ValueError("The pixel size is missing from the metadata.")
    if len(sizes) > 1:
        raise ValueError(
            "The files have different pixel sizes: "
            + ", ".join(map(str, sorted(sizes)))
            + " nm."
        )
    return float(sizes.pop())


def _tile_points(
    locs: list[pd.DataFrame],
    tiles: np.ndarray,
    use: np.ndarray,
    n_tiles: int,
    with_z: bool,
) -> list[np.ndarray]:
    """Pooled coordinates of the files used for the alignment, per
    tile."""
    columns = ["x", "y", "z"] if with_z else ["x", "y"]
    points = []
    for t in range(n_tiles):
        files = [i for i in np.flatnonzero(tiles == t) if use[i]]
        if not files:
            points.append(np.empty((0, len(columns))))
            continue
        points.append(
            np.concatenate(
                [locs[i][columns].to_numpy(dtype=float) for i in files]
            )
        )
    return points


def choose_orientation(
    points: list[np.ndarray],
    positions_um: np.ndarray,
    sizes: np.ndarray,
    pixelsize: float,
    *,
    search_radius_um: float = 10.0,
    min_overlap_um: float = 1.0,
    progress=None,
) -> tuple[str, dict[str, float]]:
    """Find the orientation of the camera relative to the stage whose
    overlaps match best.

    Each orientation (see :data:`ORIENTATIONS`) is scored by the median
    correlation peak of the predicted overlaps (up to 8, the largest
    ones).

    Parameters
    ----------
    points : list of np.ndarray
        Localizations of each tile in its camera frame, see
        :func:`match_pairs`.
    positions_um : np.ndarray
        Stage position of each tile in micrometers.
    sizes : np.ndarray
        Size ``(width, height)`` of each tile in camera pixels.
    pixelsize : float
        Camera pixel size in nm.
    search_radius_um : float, optional
        See :func:`match_pairs`. Default is 10.0.
    min_overlap_um : float, optional
        See :func:`match_pairs`. Default is 1.0.
    progress : None, "console" or ProgressType, optional
        Progress tracker, see ``lib.normalize_progress``. Default None.

    Returns
    -------
    orientation : str
        Key of the best orientation in :data:`ORIENTATIONS`.
    scores : dict
        Score of each orientation (NaN if no tiles overlap).

    Raises
    ------
    ValueError
        If no orientation predicts any overlapping tiles.
    """
    progress = lib.normalize_progress(progress, "Finding stage orientation")
    progress.setMaximum(len(ORIENTATIONS))
    progress.zero_progress("Finding the stage orientation")
    search = search_radius_um * 1e3 / pixelsize
    scores = {}
    for i, orientation in enumerate(ORIENTATIONS):
        positions = stage_to_camera(positions_um, pixelsize, orientation)
        pairs = overlapping_pairs(
            positions, sizes, min_overlap_um * 1e3 / pixelsize
        )
        pairs.sort(key=lambda p: -(p[2][1] - p[2][0]) * (p[2][3] - p[2][2]))
        pair_scores = []
        for a, b, rect in pairs[:_MAX_ORIENTATION_PAIRS]:
            peaks = _coarse_offsets(
                _translated(points[a], positions[a], rect, search),
                _translated(points[b], positions[b], rect, 0.0),
                rect,
                search,
            )
            pair_scores.append(peaks[0][2] if peaks else 0.0)
        scores[orientation] = (
            float(np.median(pair_scores)) if pair_scores else np.nan
        )
        progress.set_value(i + 1)
    valid = {key: value for key, value in scores.items() if np.isfinite(value)}
    if not valid:
        raise ValueError(
            "No tiles overlap at their stage positions. Check the pixel "
            "size and the minimum overlap."
        )
    return max(valid, key=valid.get), scores


def stitch(
    locs: list[pd.DataFrame],
    infos: list[list[dict]],
    *,
    tiles: list[int] | np.ndarray | None = None,
    targets: list[Hashable] | None = None,
    use_for_alignment: list[bool] | np.ndarray | None = None,
    positions: np.ndarray | None = None,
    stage_um: np.ndarray | None = None,
    orientation: str = "auto",
    search_radius_um: float = 10.0,
    min_overlap_um: float = 1.0,
    max_residual: float = 0.25,
    align_z: bool = True,
    cut_seams: bool = True,
    offset_frames: bool = True,
    paths: list[str] | None = None,
    progress=None,
) -> tuple[list[pd.DataFrame], list[list[dict]], dict[str, Any]]:
    """Stitch localizations of tiles acquired at different stage
    positions into one dataset per target.

    Parameters
    ----------
    locs : list of pd.DataFrames
        Localizations of each file, in the camera frame (as saved by
        Localize).
    infos : list of lists of dicts
        Metadata of each file.
    tiles : list or np.ndarray of ints, optional
        Tile index (0, 1, ...) of each file; files recorded at the same
        stage position are one tile. Default None groups the files by
        their stage position (see :func:`group_tiles`), or makes each
        file its own tile without stage positions.
    targets : list, optional
        Target (any hashable label, e.g., the imager) of each file. The
        tiles of each target are merged into one dataset. Default None
        numbers the files of each tile in order, i.e., the n-th file of
        every tile is the same target.
    use_for_alignment : list or np.ndarray of bools, optional
        Whether each file is used to measure the overlaps. Default None
        uses all files.
    positions : np.ndarray, optional
        Position of each tile, i.e., where its camera frame starts in
        the stitched dataset, in camera pixels, shape ``(n_tiles, 2)``.
        It only has to be accurate within ``search_radius_um``. Default
        None places the tiles by their stage positions.
    stage_um : np.ndarray, optional
        Stage position ``(X, Y)`` of each file in micrometers, shape
        ``(n_files, 2)``, e.g., if the files do not contain them.
        Default None reads them from the Micro-Manager metadata (see
        :func:`stage_positions`). Ignored if ``positions`` is given,
        except for grouping the files into tiles.
    orientation : str, optional
        Orientation of the camera relative to the stage, a key of
        :data:`ORIENTATIONS`, or ``"auto"`` (default) to find the one
        whose overlaps match best. Ignored if ``positions`` is given.
    search_radius_um : float, optional
        How far the true offset of two tiles may be from the initial
        one, in micrometers. Default is 10.0.
    min_overlap_um : float, optional
        Tiles are matched if their predicted overlap is at least this
        large along both axes, in micrometers. Default is 1.0.
    max_residual : float, optional
        Pairwise offsets that disagree with the stitched positions by
        more than this many camera pixels are dropped, see
        :func:`solve_positions`. Default is 0.25.
    align_z : bool, optional
        If True (default) and all files used for the alignment contain
        z, each tile also gets the z offset that matches its neighbors.
    cut_seams : bool, optional
        If True (default), each overlap keeps the localizations of only
        one tile (see :func:`seam_mask`), so that no molecule is counted
        twice. If False, overlaps contain the localizations of all
        tiles.
    offset_frames : bool, optional
        If True (default), the frames of each tile are offset by the
        frames of the tiles of the same target acquired before it (in
        the order of the Micro-Manager acquisition times, else in the
        order of ``locs``), so that frames are unique. If False, frames
        are kept.
    paths : list of str, optional
        Paths of the files, saved in the metadata. Default None.
    progress : None, "console" or ProgressType, optional
        Progress tracker, see ``lib.normalize_progress``. Default None.

    Returns
    -------
    stitched_locs : list of pd.DataFrames
        Stitched localizations of each target, in the order the targets
        first appear.
    stitched_infos : list of lists of dicts
        Metadata of each target: that of its first tile, with the
        stitched ``Width``, ``Height`` and ``Frames`` and a new entry
        describing the stitching. The drift file and the canvas of
        Render (see ``lib.fit_canvas``) of the first tile are removed,
        as they do not apply to the stitched dataset.
    report : dict
        ``"targets"`` (labels in output order), ``"tiles"`` (tile of
        each file), ``"orientation"`` (str or None),
        ``"orientation_scores"`` (dict or None), ``"positions"``
        (stitched position of each tile in camera pixels, after the
        translation that makes every coordinate non-negative),
        ``"z_offsets"`` (nm), ``"pairs"`` (see :func:`solve_positions`)
        and ``"unconnected"`` (tiles placed by their initial position
        only).

    Raises
    ------
    ValueError
        If the files have different pixel sizes, if there are fewer
        than two tiles or no stage positions while ``positions`` is
        None, or if the inputs have inconsistent lengths.
    """
    n_files = len(locs)
    if len(infos) != n_files:
        raise ValueError("locs and infos must have the same length.")
    pixelsize = _pixelsize(infos)
    if stage_um is None:
        stage = stage_positions(infos)
    else:
        stage = np.asarray(stage_um, dtype=float).reshape(-1, 2)
        if len(stage) != n_files:
            raise ValueError("stage_um must give one position per file.")
    if tiles is None:
        tiles = group_tiles(stage, n_files)
    tiles = np.asarray(tiles, dtype=int)
    n_tiles = int(tiles.max()) + 1 if n_files else 0
    if len(tiles) != n_files or set(tiles) != set(range(n_tiles)):
        raise ValueError(
            "tiles must give each file a tile index, numbered 0, 1, ..."
        )
    if n_tiles < 2:
        raise ValueError("Stitching requires at least two tiles.")
    if targets is None:
        targets = [int(np.sum(tiles[:i] == tiles[i])) for i in range(n_files)]
    if len(targets) != n_files:
        raise ValueError("targets must give one label per file.")
    for t in range(n_tiles):
        labels = [targets[i] for i in np.flatnonzero(tiles == t)]
        if len(set(labels)) < len(labels):
            raise ValueError(
                f"Tile {t} contains the same target more than once."
            )
    use = (
        np.ones(n_files, dtype=bool)
        if use_for_alignment is None
        else np.asarray(use_for_alignment, dtype=bool)
    )
    if len(use) != n_files:
        raise ValueError("use_for_alignment must have one value per file.")
    for t in range(n_tiles):
        if not use[tiles == t].any():
            use[tiles == t] = True  # every tile needs points to match

    sizes = np.zeros((n_tiles, 2))
    for i, info in enumerate(infos):
        sizes[tiles[i]] = np.maximum(sizes[tiles[i]], camera_size(info))
    with_z = align_z and all(
        "z" in locs[i].columns for i in np.flatnonzero(use)
    )
    points = _tile_points(locs, tiles, use, n_tiles, with_z)
    precision = None
    if all("lpx" in locs[i].columns for i in np.flatnonzero(use)):
        lp = np.concatenate(
            [locs[i]["lpx"].to_numpy() for i in np.flatnonzero(use)]
        )
        lp = lp[np.isfinite(lp)]
        if len(lp):
            precision = float(np.median(lp)) * pixelsize

    progress = lib.normalize_progress(progress, "Stitching")
    chosen = None
    orientation_scores = None
    if positions is None:
        if stage is None:
            raise ValueError(
                "The stage positions are missing from the metadata "
                "(Micro-Manager XPositionUm and YPositionUm); please give "
                "them (stage_um) or the tile positions (positions)."
            )
        tile_stage = np.array(
            [stage[np.flatnonzero(tiles == t)[0]] for t in range(n_tiles)]
        )
        if orientation == "auto":
            chosen, orientation_scores = choose_orientation(
                points,
                tile_stage,
                sizes,
                pixelsize,
                search_radius_um=search_radius_um,
                min_overlap_um=min_overlap_um,
                progress=progress,
            )
        elif orientation in ORIENTATIONS:
            chosen = orientation
        else:
            raise ValueError(
                f"Unknown orientation {orientation!r}; use 'auto' or one "
                f"of {', '.join(ORIENTATIONS)}."
            )
        initial = stage_to_camera(tile_stage, pixelsize, chosen)
    else:
        initial = np.asarray(positions, dtype=float).reshape(-1, 2)
        if len(initial) != n_tiles:
            raise ValueError("positions must give one position per tile.")

    pairs = match_pairs(
        points,
        initial,
        sizes,
        pixelsize,
        search_radius_um=search_radius_um,
        min_overlap_um=min_overlap_um,
        precision_nm=precision,
        progress=progress,
    )
    final, z_offsets, pairs, unconnected = solve_positions(
        pairs, initial, max_residual=max_residual, align_z=with_z
    )
    if unconnected:
        warnings.warn(
            "These tiles do not overlap with the others well enough and "
            "were placed by their initial position only: "
            + ", ".join(map(str, unconnected))
            + "."
        )
    # make every coordinate non-negative, including localizations
    # slightly outside their tile's field of view
    lowest = final.min(axis=0)
    for i, locs_ in enumerate(locs):
        if len(locs_):
            lowest = np.minimum(
                lowest,
                locs_[["x", "y"]].to_numpy().min(axis=0) + final[tiles[i]],
            )
    final = final - lowest

    # acquisition order, for the frame offsets
    times = [_acquisition_time(info) for info in infos]
    if all(time is not None for time in times):
        order = sorted(range(n_files), key=lambda i: (times[i], i))
    else:
        order = list(range(n_files))

    target_labels = list(dict.fromkeys(targets))
    stitched_locs = []
    stitched_infos = []
    for label in target_labels:
        files = [i for i in order if targets[i] == label]
        parts = []
        frame_offsets = []
        frame_offset = 0
        group_offset = 0
        for i in files:
            t = tiles[i]
            part = locs[i].copy()
            x = part["x"].to_numpy(dtype=float) + final[t, 0]
            y = part["y"].to_numpy(dtype=float) + final[t, 1]
            part["x"] = x.astype(np.float32)
            part["y"] = y.astype(np.float32)
            if "z" in part.columns and z_offsets[t]:
                part["z"] = (part["z"].to_numpy() + z_offsets[t]).astype(
                    part["z"].dtype
                )
            if offset_frames:
                part["frame"] = part["frame"] + frame_offset
            frame_offsets.append(frame_offset if offset_frames else 0)
            frame_offset += _n_frames(locs[i], infos[i])
            if "group" in part.columns and len(part):
                part["group"] = part["group"] + group_offset
                group_offset = int(part["group"].max()) + 1
            if cut_seams:
                part = part[seam_mask(x, y, t, final, sizes)]
            parts.append(part)
        merged = lib.concat_locs(parts)
        merged = merged.sort_values("frame", kind="stable").reset_index(
            drop=True
        )

        info = copy.deepcopy(infos[files[0]])
        # the canvas and drift of the first tile do not describe the
        # stitched dataset
        for inf in info:
            for key in (
                *lib.CANVAS_OFFSET_KEYS,
                *lib.CAMERA_SIZE_KEYS,
                "Last driftfile",
            ):
                inf.pop(key, None)
        extent = (final[tiles[files]] + sizes[tiles[files]]).max(axis=0)
        if len(merged):
            # strictly x < Width is kept by lib.ensure_sanity, hence + 1
            top = np.floor(merged[["x", "y"]].to_numpy().max(axis=0)) + 1
            extent = np.maximum(extent, top)
        width, height = (int(np.ceil(value)) for value in extent)
        frames = (
            frame_offset
            if offset_frames
            else max(_n_frames(locs[i], infos[i]) for i in files)
        )
        for inf in info:
            if "Width" in inf:
                inf["Width"] = width
            if "Height" in inf:
                inf["Height"] = height
            if "Frames" in inf:
                inf["Frames"] = frames
        entry = {
            "Generated by": f"Picasso v{__version__} Stitch",
            "Target": _yaml_label(label),
            "Stitched files": [
                paths[i] if paths is not None else int(i) for i in files
            ],
            "Tile offsets x (cam. px)": [
                float(final[tiles[i], 0]) for i in files
            ],
            "Tile offsets y (cam. px)": [
                float(final[tiles[i], 1]) for i in files
            ],
            "Tile offsets z (nm)": [float(z_offsets[tiles[i]]) for i in files],
            "Frame offsets": [int(offset) for offset in frame_offsets],
            "Initial positions": ("stage" if positions is None else "given"),
            "Stage orientation": chosen,
            "Search radius (um)": float(search_radius_um),
            "Overlaps cut at the seams": bool(cut_seams),
            "Width": width,
            "Height": height,
            "Frames": int(frames),
        }
        info.append(entry)
        stitched_locs.append(merged)
        stitched_infos.append(info)

    report = {
        "targets": target_labels,
        "tiles": tiles,
        "orientation": chosen,
        "orientation_scores": orientation_scores,
        "positions": final,
        "z_offsets": z_offsets,
        "pairs": pairs,
        "unconnected": unconnected,
    }
    return stitched_locs, stitched_infos, report
