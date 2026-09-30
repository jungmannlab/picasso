"""Adaptively jittered, averaged Delaunay triangulation (Baddeley,
Cannell & Soeller, *Microsc. Microanal.* 2010).

Every triangle of a Delaunay triangulation of the localizations is
painted with an intensity inversely proportional to its area, which is
linear in the local density. To take the local sampling into account,
each localization is displaced by a normally distributed jitter whose
width is its mean distance to its Delaunay neighbors, the point set is
triangulated again and the images of ``passes`` such triangulations are
averaged: the blur follows the local sampling-limited resolution
(about twice that distance) and stays small where the data are dense.

The rasterizer is ``picasso.render.kernels._fill_triangles``; this
module holds the geometry (scipy's Qhull) and the passes. It never
imports the GUI.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from concurrent import futures

import numba
import numpy as np
from scipy.spatial import Delaunay, QhullError

from .. import lib
from .kernels import _fill_triangles

#: Passes averaged by default (the paper's figures use 25 to 50).
PASSES_DEFAULT = lib.RENDER_TRIANGULATION_PASSES_DEFAULT
#: Jitter width in units of the neighbor distance (the paper uses 1,
#: or 0.5 for known periodic structures).
JITTER_DEFAULT = lib.RENDER_TRIANGULATION_JITTER_DEFAULT


@numba.njit(cache=True, nogil=True)
def _mean_edge_lengths(
    x: lib.FloatArray1D, y: lib.FloatArray1D, simplices: lib.IntArray2D
) -> lib.FloatArray1D:
    """Mean length of the triangulation edges incident to every vertex
    (the paper's mean distance between a point and its neighbors)."""
    n = x.shape[0]
    total = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for t in range(simplices.shape[0]):
        for k in range(3):
            a = simplices[t, k]
            b = simplices[t, (k + 1) % 3]
            d = np.sqrt((x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2)
            # every interior edge belongs to two triangles: both add
            # it, which weights edges consistently for the mean
            total[a] += d
            total[b] += d
            count[a] += 1
            count[b] += 1
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if count[i]:
            out[i] = total[i] / count[i]
    return out


def _triangulate(points: lib.FloatArray2D) -> lib.IntArray2D | None:
    """Delaunay simplices of ``points`` (``(n, 2)``), or None where
    Qhull cannot triangulate them (fewer than three distinct points,
    all collinear)."""
    if len(points) < 3:
        return None
    try:
        return Delaunay(points).simplices.astype(np.int32)
    except QhullError:
        return None


def _paint_pass(
    image: lib.FloatArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    mass: float,
) -> bool:
    """Triangulate ``(x, y)`` (display pixels) and paint the triangles
    into ``image``, each carrying ``mass`` localizations spread over its
    area. Returns False if the points could not be triangulated."""
    simplices = _triangulate(np.column_stack((x, y)))
    if simplices is None:
        return False
    _fill_triangles(image, x, y, simplices, float(mass))
    return True


def render_triangulation(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    oversampling: float,
    viewport: tuple,
    passes: int = PASSES_DEFAULT,
    jitter: float = JITTER_DEFAULT,
    seed: int | None = 0,
    workers: int = 1,
) -> tuple[int, lib.FloatArray2D]:
    """Jittered, averaged triangulation of the localizations in view.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        Localization coordinates, camera pixels.
    oversampling : float
        Display pixels per camera pixel.
    viewport : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.
    passes : int, optional
        Number of jittered triangulations averaged. 0 paints the
        unjittered triangulation alone. Default ``PASSES_DEFAULT``.
    jitter : float, optional
        Jitter width in units of each localization's mean distance to
        its neighbors. Default ``JITTER_DEFAULT``.
    seed : int or None, optional
        Seed of the jitter, so a view renders the same twice (None:
        fresh randomness). Default 0.
    workers : int, optional
        Threads the passes are spread over. Default 1.

    Returns
    -------
    n : int
        Localizations in the viewport (the histogram's count).
    image : lib.FloatArray2D
        The image in localizations per display pixel, shaped like the
        histogram's for the same viewport and oversampling: every
        triangle carries ``n / T`` localizations, so the total is ``n``
        up to the rasterization of triangles at the image border.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    n_py = int(np.ceil(oversampling * (y_max - y_min)))
    n_px = int(np.ceil(oversampling * (x_max - x_min)))
    image = np.zeros((n_py, n_px), dtype=np.float32)
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    n = int(in_view.sum())
    if n == 0 or n_py <= 0 or n_px <= 0:
        return n, image
    # a margin of localizations around the view keeps the convex hull
    # (whose triangles are long slivers) outside the image
    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.05 * (y_max - y_min)
    near = (
        (x > x_min - margin_x)
        & (y > y_min - margin_y)
        & (x < x_max + margin_x)
        & (y < y_max + margin_y)
    )
    xs = (oversampling * (x[near] - x_min)).astype(np.float64)
    ys = (oversampling * (y[near] - y_min)).astype(np.float64)
    m = len(xs)
    simplices = _triangulate(np.column_stack((xs, ys)))
    if simplices is None:
        # too few or collinear points: the histogram of what is there
        for xi, yi in zip(xs, ys):
            i, j = int(yi), int(xi)
            if 0 <= i < n_py and 0 <= j < n_px:
                image[i, j] += 1.0
        return n, image
    # every triangle carries the same share of the localizations, so
    # the intensity is linear in density and the image sums to the
    # count (the margin's share falls outside the image)
    mass = m / len(simplices)
    if passes <= 0:
        _fill_triangles(image, xs, ys, simplices, float(mass))
        return n, image
    d = _mean_edge_lengths(xs, ys, simplices) * float(jitter)
    root = np.random.default_rng(seed)
    seeds = root.integers(0, 2**63 - 1, size=passes)

    def one_pass(pass_seed):
        rng = np.random.default_rng(int(pass_seed))
        xj = xs + rng.normal(0.0, 1.0, m) * d
        yj = ys + rng.normal(0.0, 1.0, m) * d
        partial = np.zeros_like(image)
        _paint_pass(partial, xj, yj, mass)
        return partial

    if workers > 1 and passes > 1:
        with futures.ThreadPoolExecutor(min(workers, passes)) as pool:
            partials = list(pool.map(one_pass, seeds))
    else:
        partials = [one_pass(s) for s in seeds]
    for partial in partials:
        image += partial
    image /= passes
    return n, image
