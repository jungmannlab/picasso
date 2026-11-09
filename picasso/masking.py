"""
picasso.masking
~~~~~~~~~~~~~~~

Functions for masking localizations based on binary masks or
thresholding of images, as well as curve smoothing.

Thresholding functions are adapted from scikit-image. The package is
not used directly to avoid extra dependencies. License (BSD-2-Clause):
``LICENSES/scikit-image-LICENSE.txt``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations
from collections.abc import Iterator
from typing import Literal

import numba
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import KDTree
from statsmodels.nonparametric.smoothers_lowess import lowess as loess

from . import lib, render

__all__ = [
    "mask_locs",
    "generate_image",
    "mask_image",
    "binary_mask",
    "label_components",
    "keep_components",
    "fill_holes",
    "erode",
    "dilate",
    "mask_to_polygons",
]

#: Connectivity of the regions of a mask: pixels touching at an edge or
#: a corner belong to the same region
_CONNECTIVITY = np.ones((3, 3), dtype=bool)

_point_segment_distance_sq = lib._point_segment_distance_sq


def mask_locs(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    mask: lib.BoolArray2D,
    field_of_view: tuple[float, float] | None = None,
    mask_info: dict | None = None,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, list[dict]]:
    """Mask localizations given a binary mask.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be masked.
    info : list of dict
        Localization metadata.
    mask : lib.BoolArray2D
        Binary mask where True indicates the area to keep.
    field_of_view : tuple of floats, optional
        Width and height (nm) of the area the mask spans, starting at
        the origin. Pass the field of view of the localizations the
        mask was made from to mask others, e.g., other channels.
        Localizations beyond it are outside the mask. If None, the
        field of view of ``info`` (``lib.field_of_view``). Default is
        None.
    mask_info : dict, optional
        How the mask was made, e.g., ``{"Blur (nm)": 500.0}``, recorded
        in the metadata of both parts. Default is None.

    Returns
    -------
    locs_in : pd.DataFrame
        Localizations inside the mask.
    info_in : list of dicts
        Metadata of ``locs_in``: ``info`` with a ``"Mask in"`` entry
        appended (see ``lib.append_metadata``), recording ``mask_info``
        and the area of the mask under ``"Area (um^2)"``.
    locs_out : pd.DataFrame
        Localizations outside the mask.
    info_out : list of dicts
        Metadata of ``locs_out``: ``info`` with a ``"Mask out"`` entry
        appended, recording ``mask_info`` and the area outside the
        mask, i.e., the area of the localizations (their
        ``"Area (um^2)"``, e.g., of picks, or else their field of view)
        minus the area of the mask.
    """
    if field_of_view is None:
        field_of_view = lib.field_of_view(info)
    width, height = field_of_view
    n_y, n_x = mask.shape
    x_ind = np.floor(locs["x"].to_numpy() / width * n_x).astype(np.int64)
    y_ind = np.floor(locs["y"].to_numpy() / height * n_y).astype(np.int64)
    inside = (x_ind >= 0) & (x_ind < n_x) & (y_ind >= 0) & (y_ind < n_y)
    index = np.zeros(len(locs), dtype=bool)
    index[inside] = mask[y_ind[inside], x_ind[inside]].astype(bool)
    locs_in = locs.iloc[index].sort_values(by="frame", kind="quicksort")
    locs_out = locs.iloc[~index].sort_values(by="frame", kind="quicksort")

    pixel_area = width / n_x * height / n_y * 1e-6  # um^2
    area_in = float(np.count_nonzero(mask)) * pixel_area
    area_total = lib.get_from_metadata(info, "Area (um^2)")
    if area_total is None:
        locs_width, locs_height = lib.field_of_view(info)
        area_total = locs_width * locs_height * 1e-6
    area_out = max(float(area_total) - area_in, 0.0)
    mask_info = {} if mask_info is None else mask_info
    info_in = lib.append_metadata(
        info,
        lib.step_info("Mask in", **(mask_info | {"Area (um^2)": area_in})),
    )
    info_out = lib.append_metadata(
        info,
        lib.step_info("Mask out", **(mask_info | {"Area (um^2)": area_out})),
    )
    return locs_in, info_in, locs_out, info_out


def generate_image(
    locs: pd.DataFrame, info: list[dict], *, disp_px_size: float, blur: float
) -> lib.FloatArray2D:
    """Generate a normalized (values from 0 to 1) image from
    localizations used to make a mask.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered into an image.
    info : list of dict
        Localization metadata.
    disp_px_size : float
        Display pixel size in nm, used to histogram localizations into
        an image.
    blur : float
        Standard deviation of the Gaussian blur in nm applied to the
        histogrammed image.

    Returns
    -------
    image_blur : lib.FloatArray2D
        The blurred image, scaled so that its maximum is 1.
    """
    _, image = render.render(
        locs=locs,
        info=info,
        disp_px_size=disp_px_size,
        blur_method=None,
    )
    blur_px = blur / disp_px_size
    image_blur = ndi.gaussian_filter(image, blur_px)
    image_blur /= image_blur.max()
    return image_blur


def binary_mask(
    image: lib.FloatArray2D,
    threshold: float | lib.FloatArray2D,
) -> lib.BoolArray2D:
    """Create a binary mask from an image given a threshold.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image.
    threshold : float or lib.FloatArray2D
        Threshold value or array of threshold values. If a single float
        is provided, it is used as a global threshold. If an array is
        provided, it should have the same shape as the input image and
        is used as a pixel-wise threshold.

    Returns
    -------
    mask : lib.BoolArray2D
        Binary mask where True indicates pixels above the threshold.
    """
    mask = np.zeros(image.shape, dtype=bool)
    if np.isscalar(threshold):
        mask[image > threshold] = True
    else:
        if threshold.shape != image.shape:
            raise ValueError(
                "Threshold array must have the same shape as the image"
            )
        mask[image > threshold] = True
    return mask


def mask_image(
    image: lib.FloatArray2D,
    method: (
        float
        | Literal[
            "isodata",
            "li",
            "mean",
            "minimum",
            "otsu",
            "triangle",
            "yen",
            "local_gaussian",
            "local_mean",
            "local_median",
        ]
    ) = "otsu",
) -> tuple[lib.BoolArray2D, float] | tuple[lib.BoolArray2D, lib.FloatArray2D]:
    """Create a binary mask from a grayscale image using a specified
    thresholding method or threshold value.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image.
    method : float or str, optional
        One of 'isodata', 'li', 'mean', 'minimum', 'otsu', 'triangle',
        'yen', 'local_gaussian', 'local_mean' and 'local_median'.
        Thresholding method or threshold value. If a float is provided,
        it is used as a global threshold. If a string is provided, it
        specifies the thresholding method to use. Default is 'otsu'.

    Returns
    -------
    mask : lib.BoolArray2D
        Binary mask where True indicates pixels above the threshold.
    threshold : float or lib.FloatArray2D
        Threshold value used to create the mask. Can be a single float
        for global thresholding or an array for pixel-wise thresholding.
    """
    assert image.ndim == 2, "Input image must be 2D"
    if not isinstance(method, str):
        threshold = float(method)
        mask = binary_mask(image, threshold)
        return mask, threshold
    threshold_functions = {
        "isodata": threshold_isodata,
        "li": threshold_li,
        "mean": threshold_mean,
        "minimum": threshold_minimum,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
        "yen": threshold_yen,
        "local_gaussian": threshold_local_gaussian,
        "local_mean": threshold_local_mean,
        "local_median": threshold_local_median,
    }
    function = threshold_functions.get(method)
    if function is None:
        raise ValueError(f"Unknown thresholding method: {method}")
    threshold = function(image)
    mask = binary_mask(image, threshold)
    return mask, threshold


def label_components(
    mask: lib.BoolArray2D,
) -> tuple[lib.IntArray2D, lib.IntArray1D]:
    """Label the connected regions of a binary mask, e.g., the cells,
    largest first. Pixels touching at an edge or a corner belong to the
    same region.

    Parameters
    ----------
    mask : lib.BoolArray2D
        Binary mask, True (or nonzero) inside.

    Returns
    -------
    labels : lib.IntArray2D
        Region of each pixel: 0 outside the mask, 1 for the largest
        region, 2 for the second largest, etc.
    sizes : lib.IntArray1D
        Number of pixels of each region, in the order of the labels
        (``sizes[0]`` of region 1).
    """
    labels, n_regions = ndi.label(np.asarray(mask, dtype=bool), _CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=n_regions + 1)[1:]
    # largest first; ties in the order of the first pixel
    order = np.argsort(-sizes, kind="stable")
    relabel = np.zeros(n_regions + 1, dtype=np.int64)
    relabel[order + 1] = np.arange(1, n_regions + 1)
    return relabel[labels], sizes[order]


def keep_components(
    mask: lib.BoolArray2D,
    *,
    min_area: float = 0.0,
    n_largest: int | None = None,
    pixel_size: float = 1.0,
) -> lib.BoolArray2D:
    """Keep the connected regions of a binary mask (see
    ``label_components``) that are large enough, e.g., to remove specks
    of background or to keep only the largest cell.

    Parameters
    ----------
    mask : lib.BoolArray2D
        Binary mask, True (or nonzero) inside.
    min_area : float, optional
        Smallest area of a region that is kept, in the unit of
        ``pixel_size`` squared: e.g., nm^2 (1e6 per um^2) with the
        pixel size in nm. Default is 0 (all regions).
    n_largest : int or None, optional
        Number of the largest regions kept (among those of at least
        ``min_area``). None keeps all. Default is None.
    pixel_size : float, optional
        Side length of a pixel of the mask, e.g., in nm. Default is 1
        (``min_area`` in pixels).

    Returns
    -------
    mask : lib.BoolArray2D
        The mask with only the regions kept.

    Raises
    ------
    ValueError
        If ``min_area`` is negative, ``n_largest`` is negative or
        ``pixel_size`` is not positive.
    """
    if not min_area >= 0:
        raise ValueError("min_area must not be negative.")
    if n_largest is not None and n_largest < 0:
        raise ValueError("n_largest must not be negative.")
    if not pixel_size > 0:
        raise ValueError("pixel_size must be positive.")
    labels, sizes = label_components(mask)
    kept = sizes * pixel_size**2 >= min_area
    if n_largest is not None:
        kept[int(n_largest) :] = False  # the regions are largest first
    return np.concatenate([[False], kept])[labels]


def fill_holes(mask: lib.BoolArray2D) -> lib.BoolArray2D:
    """Fill the holes of a binary mask, i.e., the areas outside the
    mask that it encloses and that do not reach the image border.

    Parameters
    ----------
    mask : lib.BoolArray2D
        Binary mask, True (or nonzero) inside.

    Returns
    -------
    mask : lib.BoolArray2D
        The mask with its holes filled.
    """
    return ndi.binary_fill_holes(np.asarray(mask, dtype=bool))


def dilate(
    mask: lib.BoolArray2D,
    distance: float,
    *,
    pixel_size: float = 1.0,
) -> lib.BoolArray2D:
    """Grow a binary mask by a distance in all directions alike: add the
    pixels whose centers lie within ``distance`` of the center of a
    pixel of the mask (Euclidean distance transform).

    Parameters
    ----------
    mask : lib.BoolArray2D
        Binary mask, True (or nonzero) inside.
    distance : float
        Distance by which the mask grows, in the unit of
        ``pixel_size``, e.g., nm. 0 leaves the mask unchanged.
    pixel_size : float, optional
        Side length of a pixel of the mask, e.g., in nm. Default is 1
        (``distance`` in pixels).

    Returns
    -------
    mask : lib.BoolArray2D
        The grown mask.

    Raises
    ------
    ValueError
        If ``distance`` is negative or ``pixel_size`` is not positive.

    See Also
    --------
    erode : Shrink a mask.
    """
    mask = np.asarray(mask, dtype=bool)
    _check_morphology(distance, pixel_size)
    if not mask.any():
        return mask.copy()
    # the distance of each pixel to the nearest pixel of the mask
    distances = ndi.distance_transform_edt(~mask, sampling=pixel_size)
    return distances <= distance


def erode(
    mask: lib.BoolArray2D,
    distance: float,
    *,
    pixel_size: float = 1.0,
) -> lib.BoolArray2D:
    """Shrink a binary mask by a distance in all directions alike, e.g.,
    to stay away from the edges of a cell: keep the pixels whose
    centers lie farther than ``distance`` from the center of every
    pixel outside the mask (Euclidean distance transform). The image
    border is not an edge of the mask.

    Parameters
    ----------
    mask : lib.BoolArray2D
        Binary mask, True (or nonzero) inside.
    distance : float
        Distance by which the mask shrinks, in the unit of
        ``pixel_size``, e.g., nm. 0 leaves the mask unchanged.
    pixel_size : float, optional
        Side length of a pixel of the mask, e.g., in nm. Default is 1
        (``distance`` in pixels).

    Returns
    -------
    mask : lib.BoolArray2D
        The shrunk mask.

    Raises
    ------
    ValueError
        If ``distance`` is negative or ``pixel_size`` is not positive.

    See Also
    --------
    dilate : Grow a mask.
    """
    mask = np.asarray(mask, dtype=bool)
    _check_morphology(distance, pixel_size)
    return ~dilate(~mask, distance, pixel_size=pixel_size)


def _check_morphology(distance: float, pixel_size: float) -> None:
    """Raise ValueError for a negative ``distance`` or a pixel size
    that is not positive."""
    if not distance >= 0:
        raise ValueError("distance must not be negative.")
    if not pixel_size > 0:
        raise ValueError("pixel_size must be positive.")


def mask_to_polygons(
    mask: lib.BoolArray2D,
    *,
    field_of_view: tuple[float, float],
    tolerance: float | None = None,
) -> list[list[tuple[float, float]]]:
    """Outline each connected region of a binary mask (see
    ``label_components``), e.g., each cell, as a polygon pick, so that
    the mask can be used wherever picks are, e.g., in
    ``postprocess.picked_locs`` or the spatial statistics such as
    ``postprocess.ripley_h``.

    The outline follows the edges of the pixels of the region and is
    then simplified (Douglas-Peucker algorithm): corners are removed as
    long as the outline moves by at most ``tolerance``. A region with
    holes becomes a single "keyhole" polygon: each hole is joined to
    the outline by a cut of zero width from its nearest corner, i.e.,
    the outline runs along the cut to the hole, around the hole and
    back along the cut. The cut encloses no area, so the localizations
    on both sides of it are inside the pick and those in the hole are
    outside.

    Parameters
    ----------
    mask : lib.BoolArray2D
        Binary mask, True (or nonzero) inside.
    field_of_view : tuple of floats
        Width and height in nm of the area that the mask spans, starting
        at the origin, as in ``mask_locs``: e.g., the field of view of
        the localizations the mask was made from (see
        ``lib.field_of_view``).
    tolerance : float or None, optional
        Largest distance in nm by which the simplified outline may
        deviate from the edges of the pixels. 0 keeps the outline of
        the pixels (only the corners where it runs straight on are
        removed), so that the picks hold the same localizations as the
        mask (``mask_locs``). If None, the larger side of a pixel,
        which smooths the steps between the pixels. Default is None.

    Returns
    -------
    picks : list of lists of tuples
        One closed polygon pick per region, largest region first: its
        corners ``(x, y)`` in nm, the first one repeated at the end. Use
        them with ``pick_shape="Polygon"``. Empty for an empty mask.

    Raises
    ------
    ValueError
        If the mask is not 2D, the field of view is not positive or
        ``tolerance`` is negative.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("The mask must be a 2D array.")
    width, height = (float(value) for value in field_of_view)
    if not (width > 0 and height > 0):
        raise ValueError("field_of_view must be positive.")
    n_y, n_x = mask.shape
    pixel = np.array([width / n_x, height / n_y])
    if tolerance is None:
        tolerance = float(pixel.max())
    elif not tolerance >= 0:
        raise ValueError("tolerance must not be negative.")
    labels, _ = label_components(mask)
    picks = []
    for label, (rows, columns) in enumerate(ndi.find_objects(labels), 1):
        offset = np.array([columns.start, rows.start])
        outer, holes = None, []
        for ring in _pixel_outlines(labels[rows, columns] == label):
            corners = _simplify_ring((ring + offset) * pixel, tolerance)
            # the region is on the left of its outline, which therefore
            # runs counterclockwise (in x, y) and around holes clockwise
            if _signed_area(corners) > 0:
                outer = corners
            else:
                holes.append(corners)
        corners = _join_holes(outer, holes)
        pick = [(float(x), float(y)) for x, y in corners]
        picks.append(pick + pick[:1])
    return picks


def _pixel_outlines(region: lib.BoolArray2D) -> list[lib.FloatArray2D]:
    """The outlines along the pixel edges of a connected region of a
    binary mask (pixels touching at an edge or a corner): the outer
    outline and that of each hole, as the corners ``(column, row)`` in
    pixels (the corner of the first pixel is ``(0, 0)``), with the
    region on the left of each outline. Corners where an outline runs
    straight on are left out."""
    padded = np.pad(region, 1)
    inside = padded[1:-1, 1:-1]
    starts, ends = [], []
    # the sides of the pixels of the region that face a pixel outside,
    # each directed so that the pixel is on its left: top, right,
    # bottom and left side
    for outside, start, end in (
        (~padded[:-2, 1:-1], (0, 0), (1, 0)),
        (~padded[1:-1, 2:], (1, 0), (1, 1)),
        (~padded[2:, 1:-1], (1, 1), (0, 1)),
        (~padded[1:-1, :-2], (0, 1), (0, 0)),
    ):
        rows, columns = np.nonzero(inside & outside)
        starts.append(np.column_stack([columns + start[0], rows + start[1]]))
        ends.append(np.column_stack([columns + end[0], rows + end[1]]))
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    directions = ends - starts
    n_corners_x = region.shape[1] + 1
    start_ids = starts[:, 1] * n_corners_x + starts[:, 0]
    end_ids = ends[:, 1] * n_corners_x + ends[:, 0]
    order = np.argsort(start_ids, kind="stable")
    first = np.searchsorted(start_ids[order], end_ids, side="left")
    n_leaving = (
        np.searchsorted(start_ids[order], end_ids, side="right") - first
    )
    following = order[first]
    other = order[np.minimum(first + 1, len(order) - 1)]
    # where two pixels of the region touch at a corner only, two sides
    # leave the corner: the right turn keeps both pixels in one outline
    turn = (
        directions[:, 0] * directions[following, 1]
        - directions[:, 1] * directions[following, 0]
    )
    following = np.where((n_leaving == 2) & (turn > 0), other, following)
    sequence, bounds = _cycles(following)
    outlines = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        sides = sequence[start:end]
        turns = np.any(
            directions[sides] != np.roll(directions[sides], 1, axis=0), axis=1
        )
        outlines.append(starts[sides][turns].astype(np.float64))
    return outlines


@numba.jit(nopython=True, nogil=True, cache=True)
def _cycles(
    following: lib.IntArray1D,
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """The cycles of the permutation ``following`` (the element that
    follows each element): the elements in the order of the cycles and
    the start of each cycle in it (with the total number at the
    end)."""
    n = len(following)
    sequence = np.empty(n, dtype=np.int64)
    bounds = np.zeros(n + 1, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    m = 0
    n_cycles = 0
    for first in range(n):
        if visited[first]:
            continue
        element = first
        while not visited[element]:
            visited[element] = True
            sequence[m] = element
            m += 1
            element = following[element]
        n_cycles += 1
        bounds[n_cycles] = m
    return sequence, bounds[: n_cycles + 1]


def _signed_area(corners: lib.FloatArray2D) -> float:
    """Area enclosed by the closed outline through ``corners`` (shape
    (n, 2), x and y), positive if it runs counterclockwise."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _simplify_ring(
    corners: lib.FloatArray2D,
    tolerance: float,
) -> lib.FloatArray2D:
    """Simplify the closed outline through ``corners`` (shape (n, 2))
    by the Douglas-Peucker algorithm, which removes corners as long as
    each removed corner lies within ``tolerance`` of the simplified
    outline, starting from the first corner and the one farthest from
    it."""
    n = len(corners)
    if tolerance <= 0 or n <= 4:
        return corners
    farthest = int(np.argmax(np.sum((corners - corners[0]) ** 2, axis=1)))
    closed = np.vstack([corners, corners[:1]])
    keep = np.zeros(n + 1, dtype=bool)
    keep[: farthest + 1] = _douglas_peucker(closed[: farthest + 1], tolerance)
    keep[farthest:] |= _douglas_peucker(closed[farthest:], tolerance)
    simplified = corners[keep[:n]]
    return simplified if len(simplified) >= 3 else corners


@numba.jit(nopython=True, nogil=True, cache=True)
def _douglas_peucker(
    points: lib.FloatArray2D,
    tolerance: float,
) -> lib.BoolArray1D:
    """Which of the ``points`` (shape (n, 2)) of an open line the
    Douglas-Peucker algorithm keeps: the ends, and recursively the
    point farthest from the segment between two kept points if it is
    farther than ``tolerance``."""
    n = len(points)
    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[n - 1] = True
    # pending stretches between kept points; they do not overlap, so
    # there are fewer than n at a time
    stack = np.empty((n, 2), dtype=np.int64)
    stack[0, 0] = 0
    stack[0, 1] = n - 1
    top = 1
    tolerance_sq = tolerance * tolerance
    while top > 0:
        top -= 1
        first = stack[top, 0]
        last = stack[top, 1]
        farthest = -1
        largest = tolerance_sq
        for i in range(first + 1, last):
            distance_sq = _point_segment_distance_sq(
                points[i, 0],
                points[i, 1],
                points[first, 0],
                points[first, 1],
                points[last, 0],
                points[last, 1],
            )
            if distance_sq > largest:
                largest = distance_sq
                farthest = i
        if farthest >= 0:
            keep[farthest] = True
            stack[top, 0] = first
            stack[top, 1] = farthest
            stack[top + 1, 0] = farthest
            stack[top + 1, 1] = last
            top += 2
    return keep


def _join_holes(
    outer: lib.FloatArray2D,
    holes: list[lib.FloatArray2D],
) -> lib.FloatArray2D:
    """Join the ``holes`` (outlines running opposite to ``outer``) to
    the outer outline by cuts of zero width (keyhole polygon). The
    holes are joined largest first, each from its corner nearest to the
    outlines joined before it (the outer one and the larger holes) to
    the nearest corner of them."""
    if not holes:
        return outer
    rings = [outer] + sorted(holes, key=_signed_area)  # largest first
    attached = _hole_attachments(rings)
    # depth first without recursion, as holes may be joined to holes
    joined = []
    walks = [_walk_ring(rings, attached, 0, 0, False)]
    while walks:
        item = next(walks[-1], None)
        if item is None:
            walks.pop()
        elif isinstance(item, tuple):
            walks.append(_walk_ring(rings, attached, *item, True))
        else:
            joined.append(item)
    return np.array(joined)


def _hole_attachments(
    rings: list[lib.FloatArray2D],
) -> list[dict[int, list[tuple[int, int]]]]:
    """For ``_join_holes``: the holes (``rings[1:]``, largest first)
    joined at each corner of each outline, as ``(hole, corner of the
    hole)``. Each hole is joined from its corner nearest to the
    outlines before it to the nearest corner of them."""
    outer = rings[0]
    ring_of = np.concatenate([np.full(len(r), k) for k, r in enumerate(rings)])
    corner_of = np.concatenate([np.arange(len(r)) for r in rings])
    tree = KDTree(np.concatenate(rings))
    n_neighbors = min(16, len(ring_of))
    outer_tree = None
    attached = [{} for _ in rings]
    for k in range(1, len(rings)):
        distances, nearest = tree.query(rings[k], k=n_neighbors)
        distances[ring_of[nearest] >= k] = np.inf
        j, column = np.unravel_index(np.argmin(distances), distances.shape)
        if np.isfinite(distances[j, column]):
            target = nearest[j, column]
            ring, corner = ring_of[target], corner_of[target]
        else:  # no joined outline among the nearest corners
            if outer_tree is None:
                outer_tree = KDTree(outer)
            distances, nearest = outer_tree.query(rings[k])
            j = int(np.argmin(distances))
            ring, corner = 0, nearest[j]
        attached[ring].setdefault(int(corner), []).append((k, int(j)))
    return attached


def _walk_ring(
    rings: list[lib.FloatArray2D],
    attached: list[dict[int, list[tuple[int, int]]]],
    ring: int,
    start: int,
    back: bool,
) -> Iterator:
    """For ``_join_holes``: the corners of an outline from ``start``
    on, each followed by the holes joined at it (yielded as (hole,
    corner) tuples) and by itself again, ending with ``start`` again if
    ``back``."""
    corners = rings[ring]
    for step in range(len(corners)):
        corner = (start + step) % len(corners)
        yield corners[corner]
        for hole in attached[ring].get(corner, ()):
            yield hole
            yield corners[corner]
    if back:
        yield corners[start]


def threshold_isodata(image: lib.FloatArray2D) -> float:
    """Return threshold value based on the isodata method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.

    References
    ----------
    1. Ridler, TW & Calvard, S (1978), "Picture thresholding using an
       iterative selection method"
       IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
       :DOI:`10.1109/TSMC.1978.4310039`
    2. Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
       Techniques and Quantitative Performance Evaluation" Journal of
       Electronic Imaging, 13(1): 146-165,
       http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
       :DOI:`10.1117/1.1631315`
    3. ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold
    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # image only contains one unique value
    if len(bin_centers) == 1:
        return bin_centers[0]

    csuml = np.cumsum(counts)
    csumh = csuml[-1] - csuml
    intensity_sum = counts * bin_centers

    csum_intensity = np.cumsum(intensity_sum)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]

    all_mean = (lower + higher) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]

    distances = all_mean - bin_centers[:-1]
    threshold = bin_centers[:-1][(distances >= 0) & (distances < bin_width)][0]
    return threshold


def threshold_li(image: lib.FloatArray2D) -> float:
    """Return threshold value based on Li's minimum cross entropy method
    for a grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.

    References
    ----------
    1. Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
       Pattern Recognition, 26(4): 617-625
       :DOI:`10.1016/0031-3203(93)90115-D`
    2. Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
       Cross Entropy Thresholding" Pattern Recognition Letters, 18(8):
       771-776 :DOI:`10.1016/S0167-8655(98)00057-9`
    3. Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
       Techniques and Quantitative Performance Evaluation" Journal of
       Electronic Imaging, 13(1): 146-165
       :DOI:`10.1117/1.1631315`
    4. ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold
    """
    # Li's algorithm requires positive image (because of log(mean))
    if np.any(image < 0):
        raise ValueError("Li's method requires a non-negative image")

    # Make sure image has more than one value; otherwise, return that value
    # This works even for np.inf
    if np.all(image == image.flat[0]):
        return image.flat[0]

    tolerance = np.min(np.diff(np.unique(image))) / 2
    t_next = np.mean(image)
    t_curr = -2 * tolerance

    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = image > t_curr
        mean_fore = np.mean(image[foreground])
        mean_back = np.mean(image[~foreground])

        if mean_back == 0.0:
            break

        t_next = (mean_back - mean_fore) / (
            np.log(mean_back) - np.log(mean_fore)
        )
    threshold = t_next
    return threshold


def threshold_mean(image: lib.FloatArray2D) -> float:
    """Return threshold value based on the mean method for a grayscale
    image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    threshold : float
        Mean threshold value.

    References
    ----------
    1. C. A. Glasbey, "An analysis of histogram-based thresholding
       algorithms," CVGIP: Graphical Models and Image Processing,
       vol. 55, pp. 532-537, 1993.
       :DOI:`10.1006/cgip.1993.1040`
    """
    threshold = image.mean()
    return threshold


def threshold_minimum(image: lib.FloatArray2D) -> float:
    """Return threshold value based on minimum method.

    The histogram of the input ``image`` is computed if not provided and
    smoothed until there are only two maxima. Then the minimum in
    between is the threshold value.

    Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    threshold : float
        Mean threshold value.

    References
    ----------
    1. C. A. Glasbey, "An analysis of histogram-based thresholding
       algorithms," CVGIP: Graphical Models and Image Processing,
       vol. 55, pp. 532-537, 1993.
    2. Prewitt, JMS & Mendelsohn, ML (1966), "The analysis of cell
       images", Annals of the New York Academy of Sciences 128: 1035-1053
       :DOI:`10.1111/j.1749-6632.1965.tb11715.x`
    """

    def find_local_maxima_idx(hist):
        maximum_idxs = list()
        direction = 1
        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1
                    maximum_idxs.append(i)
            else:
                if hist[i + 1] > hist[i]:
                    direction = 1
        return maximum_idxs

    counts, bin_edges = np.histogram(image, bins=256)
    smooth_hist = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for counter in range(10000):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError("Unable to find two maxima in histogram")
    elif counter == 10000 - 1:
        raise RuntimeError("Maximum iteration reached for histogram smoothing")

    # Find the lowest point between the maxima
    threshold_idx = np.argmin(
        smooth_hist[maximum_idxs[0] : maximum_idxs[1] + 1]
    )
    threshold = bin_centers[maximum_idxs[0] + threshold_idx]
    return threshold


def threshold_otsu(image: lib.FloatArray2D) -> float:
    """Return threshold value based on Otsu's method for a grayscale
    image. Adapted from scikit-image.

    Based on implementation in scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    threshold : float
        Otsu's threshold value.

    References
    ----------
    1. Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    """
    if image.min() == image.max():
        # a constant image has no two classes to tell apart: keep all of it
        # (its histogram would leave the class means below at 0 / 0)
        return float(image.min()) - 0.5

    # histogram the image and converts bin edges to bin centers
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


def threshold_triangle(image: lib.FloatArray2D) -> float:
    """Return threshold value based on the triangle algorithm for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.

    References
    ----------
    1. Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
       Automatic Measurement of Sister Chromatid Exchange Frequency,
       Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
       :DOI:`10.1177/25.7.70454`
    2. ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold
    """
    hist, bin_edges = np.histogram(image.reshape(-1), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    nbins = len(hist)

    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]

    if arg_low_level == arg_high_level:
        return image.ravel()[0]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    del arg_high_level

    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    threshold = bin_centers[arg_level]
    return threshold


def threshold_yen(image: lib.FloatArray2D) -> float:
    """Return threshold value based on Yen's method for a grayscale
    image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.

    References
    ----------
    1. Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
       for Automatic Multilevel Thresholding" IEEE Trans. on Image
       Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`
    2. Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
       Techniques and Quantitative Performance Evaluation" Journal of
       Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`
       http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    3. ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold

    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = counts.astype("float32", copy=False) / counts.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf**2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(
        ((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2
    )
    threshold = bin_centers[crit.argmax()]
    return threshold


# threshold methods that return pixel-wise thresholds
def threshold_local_gaussian(image: lib.FloatArray2D) -> lib.BoolArray2D:
    """Return threshold value based on the Gaussian local method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    mask : lib.BoolArray2D
        Binary mask. Values of 1 indicate foreground pixels.

    References
    ----------
    1. Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
       (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
       ISBN: 0-201-18075-8
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    sigma = tuple([(b - 1) / 6.0 for b in block_size])
    ndi.gaussian_filter(
        image,
        sigma=sigma,
        output=thresh_image,
        mode="reflect",
    )
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask


def threshold_local_mean(image: lib.FloatArray2D) -> lib.BoolArray2D:
    """Return threshold value based on the mean local method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    mask : lib.BoolArray2D
        Binary mask. Values of 1 indicate foreground pixels.

    References
    ----------
    1. Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
       (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
       ISBN: 0-201-18075-8
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    ndi.uniform_filter(image, block_size, output=thresh_image, mode="reflect")
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask


def threshold_local_median(image: lib.FloatArray2D) -> lib.BoolArray2D:
    """Return threshold value based on the median local method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    mask : lib.BoolArray2D
        Binary mask. Values of 1 indicate foreground pixels.

    References
    ----------
    1. Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
       (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
       ISBN: 0-201-18075-8
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    ndi.median_filter(image, block_size, output=thresh_image, mode="reflect")
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask


def threshold_tukey(image: lib.FloatArray2D) -> lib.BoolArray2D:
    """Find the Tukey's mask, used to avoid FFT artifacts.

    Parameters
    ----------
    image : lib.FloatArray2D
        The input grayscale image.

    Returns
    -------
    mask : lib.BoolArray2D
        Tukey's mask (binary).
    """
    assert image.shape[0] == image.shape[1], "Image must be square"
    nfac = 8
    height, width = image.shape
    x = np.arange(width)
    x_im = (x - (width / 2)) / width
    x_im = np.tile(x_im, (height, 1))
    mask = 0.5 - 0.5 * np.cos(np.pi * nfac * x_im)
    mask[np.abs(x_im) < ((nfac - 2) / (nfac * 2))] = 1
    mask = mask * np.rot90(mask)
    return mask


def loess_smooth(arr: lib.FloatArray1D, span: int = 5) -> lib.FloatArray1D:
    """Smooth an array using LOESS smoothing.

    Parameters
    ----------
    arr : lib.FloatArray1D
        Input array to be smoothed (1D).
    span : int, optional
        Width of the smoothing window in samples; rounded up to the next odd
        number. Default 5.

    Returns
    -------
    smoothed_arr : lib.FloatArray1D
        Smoothed array.
    """
    # smooth the frc curve
    span += 1 - (span % 2)  # make sure span is odd
    x = np.arange(len(arr))
    smoothed_arr = loess(arr, x, span / len(arr), return_sorted=False)
    return smoothed_arr
