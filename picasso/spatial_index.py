"""Multi-resolution spatial index for fast viewport rendering.

Built once when a channel is loaded; queried per redraw to skip the
O(N) viewport scan that ``picasso.render._render_setup`` would otherwise
perform on every pan/zoom.

The pyramid stores three grid resolutions sharing a single permutation
sorted by Morton (Z-order) at the finest level. Because Z-order is
hierarchical, each coarser block at level L corresponds to a contiguous
range in the same sorted permutation -- so all levels reuse one ``perm``
array (~4 N bytes) rather than one per level.

Circular picks query the same pyramid (``query_circle``): the block
sizes do not depend on the pick size, so the index built at load time
serves every pick diameter, whereas the single-resolution
``get_index_blocks`` of :mod:`picasso.postprocess` has to be rebuilt
(sorting and copying the whole DataFrame) whenever the pick size
changes; it remains the fallback where no pyramid could be built.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numba
import numpy as np
import pandas as pd

from . import lib


_log = logging.getLogger(__name__)

#: HDF5 group holding a persisted pyramid, see ``save_render_index``.
RENDER_INDEX_GROUP = "render_index"
#: Files with fewer localizations get no persisted pyramid: theirs
#: builds in milliseconds and the 4 bytes per row would be a large part
#: of a small file.
PERSIST_MIN_LOCS = 100_000
#: version 2 sorts the permutation by fine Morton keys (see
#: ``_FINE_BITS``), which the quad-tree needs; version 1 files are
#: rebuilt
_RENDER_INDEX_VERSION = 2
#: Levels of the implicit quad-tree below the pyramid's base block: the
#: finest cell is ``base / 2**_FINE_BITS`` (2 px / 256 ≈ 0.008 px, about
#: 1 nm at 130 nm pixels, for the usual 512 px field of view).
_FINE_BITS = 8


# Target upper bound on blocks per viewport edge at the chosen level.
# Tunable; ~64 keeps the inner gather loop tight while still letting the
# finest level cover small zoomed-in viewports.
_TARGET_BLOCKS_PER_EDGE = 64

# Viewport-to-FOV area ratio at/above which ``query_viewport`` bypasses
# the pyramid and returns ``None``. The caller then renders the full
# locs DataFrame and lets the renderer's vectorised ``in_view`` mask do
# the filtering -- avoiding a pandas ``iloc`` copy of nearly all rows,
# which dominates redraw cost at full-FOV (see ``query_viewport``).
_BYPASS_COVERAGE_RATIO = 0.1


@dataclass
class RenderIndexPyramid:
    """Multi-resolution spatial index over a single locs DataFrame.

    Attributes
    ----------
    perm : IntArray1D, shape (N,), dtype uint32
        ``perm[i]`` is the original-locs index at sort position ``i``,
        where the sort key is the Morton code of ``(x // base, y // base)``
        at the finest level.
    block_sizes : tuple[float, ...]
        Block side lengths in camera pixels, ascending. ``block_sizes[0]``
        is the finest level.
    block_starts, block_ends : list[IntArray2D]
        Per level, a ``(K_L, L_L)`` uint32 grid where
        ``perm[block_starts[i, j]:block_ends[i, j]]`` are the
        original-locs indices in block ``(i, j)``.
    width, height : float
        FOV size copied from ``info``, used by the query to clip block
        rectangles.
    """

    perm: lib.IntArray1D
    block_sizes: tuple[float, ...]
    block_starts: list[lib.IntArray2D]
    block_ends: list[lib.IntArray2D]
    width: float
    height: float
    #: bits per axis of the quad-tree root above the base block: the
    #: root is a square of ``2**root_bits`` base blocks covering the
    #: field of view (see ``_quadtree_geometry``)
    root_bits: int = 0
    #: levels below the base block (``_FINE_BITS`` at build time)
    fine_bits: int = 0
    #: the fine Morton key of every row in ``perm`` order, ascending;
    #: the quad-tree is implicit in it (``quadtree_layout``). Filled by
    #: ``build_render_index`` and by ``validate_render_index`` for an
    #: index read from a file.
    sorted_keys: lib.IntArray1D | None = None


def _base_block_size(width: float, height: float) -> float:
    """Pick the finest block size based on FOV.

    Targets ~256k blocks at the finest level for the common 512x512 -
    1024x1024 SMLM FOVs. Floor of 1.0 -- sub-pixel blocks would mostly
    hold a single loc each and waste grid memory.
    """
    return float(max(1.0, np.ceil(np.sqrt(width * height / 256_000.0))))


@numba.njit(cache=True)
def _morton_encode_2d(x: lib.IntArray1D, y: lib.IntArray1D) -> lib.IntArray1D:
    """Interleave bits of ``(x, y)`` into a Morton (Z-order) key.

    ``x`` and ``y`` are 32-bit unsigned block coordinates; the returned
    key is uint64. Inputs above 2**16 are still handled because the
    masks below interleave the full 32-bit input -- but typical SMLM
    grids stay well below that.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.uint64)
    M0 = np.uint64(0x0000FFFF0000FFFF)
    M1 = np.uint64(0x00FF00FF00FF00FF)
    M2 = np.uint64(0x0F0F0F0F0F0F0F0F)
    M3 = np.uint64(0x3333333333333333)
    M4 = np.uint64(0x5555555555555555)
    one = np.uint64(1)
    for i in range(n):
        xi = np.uint64(x[i])
        yi = np.uint64(y[i])
        xi = (xi | (xi << np.uint64(16))) & M0
        xi = (xi | (xi << np.uint64(8))) & M1
        xi = (xi | (xi << np.uint64(4))) & M2
        xi = (xi | (xi << np.uint64(2))) & M3
        xi = (xi | (xi << one)) & M4
        yi = (yi | (yi << np.uint64(16))) & M0
        yi = (yi | (yi << np.uint64(8))) & M1
        yi = (yi | (yi << np.uint64(4))) & M2
        yi = (yi | (yi << np.uint64(2))) & M3
        yi = (yi | (yi << one)) & M4
        out[i] = xi | (yi << one)
    return out


def _quadtree_geometry(
    width: float, height: float, base: float
) -> tuple[int, int, int]:
    """``(L, K, root_bits)``: base blocks per row and column and the
    bits per axis of the smallest dyadic square of base blocks that
    covers the ``(K, L)`` grid (the quad-tree root)."""
    L = max(1, int(np.ceil(width / base)))
    K = max(1, int(np.ceil(height / base)))
    root_bits = max(1, int(np.ceil(np.log2(max(K, L)))))
    return L, K, root_bits


@numba.njit(cache=True)
def _fine_cells(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    base: float,
    fine_bits: int,
    L: int,
    K: int,
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Cell coordinates at the finest quad-tree level: the base block
    (clipped to the grid, so out-of-FOV rows stay queryable) times
    ``2**fine_bits`` plus the sub-block cell, so that shifting a fine
    coordinate right by ``fine_bits`` gives the base block exactly."""
    n = x.shape[0]
    ix = np.empty(n, dtype=np.uint32)
    iy = np.empty(n, dtype=np.uint32)
    sub = 1 << fine_bits
    cell = base / sub
    for i in range(n):
        bx = int(np.floor(x[i] / base))
        bx = min(max(bx, 0), L - 1)
        fx = int(np.floor((x[i] - bx * base) / cell))
        fx = min(max(fx, 0), sub - 1)
        ix[i] = bx * sub + fx
        by = int(np.floor(y[i] / base))
        by = min(max(by, 0), K - 1)
        fy = int(np.floor((y[i] - by * base) / cell))
        fy = min(max(fy, 0), sub - 1)
        iy[i] = by * sub + fy
    return ix, iy


def _fine_keys(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    base: float,
    fine_bits: int,
    L: int,
    K: int,
) -> lib.IntArray1D:
    """Fine Morton key of every row (see ``_fine_cells``)."""
    ix, iy = _fine_cells(x, y, base, fine_bits, L, K)
    return _morton_encode_2d(ix, iy)


@numba.njit(cache=True)
def _fill_blocks_from_sorted(
    bx: lib.IntArray1D,
    by: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
) -> None:
    """Fill ``block_starts``/``block_ends`` by single linear scan.

    Expects ``bx``/``by`` to be the block coordinates of each loc in the
    pyramid sort order; because the sort is by Morton at the finest
    level, locs sharing a block at *any* level form one contiguous run.
    """
    n = bx.shape[0]
    if n == 0:
        return
    cur_bx = bx[0]
    cur_by = by[0]
    block_starts[cur_by, cur_bx] = 0
    for k in range(1, n):
        if bx[k] != cur_bx or by[k] != cur_by:
            block_ends[cur_by, cur_bx] = k
            cur_bx = bx[k]
            cur_by = by[k]
            block_starts[cur_by, cur_bx] = k
    block_ends[cur_by, cur_bx] = n


def build_render_index(
    locs: pd.DataFrame,
    info: list[dict],
    n_levels: int = 3,
) -> RenderIndexPyramid | None:
    """Build the pyramid for one channel's locs.

    Parameters
    ----------
    locs : pd.DataFrame
        The localizations to index, with ``x`` and ``y`` columns.
    info : list of dicts
        Localizations metadata; "Width" and "Height" are required.
    n_levels : int, optional
        Number of pyramid levels, each with blocks 4x larger than the last.
        Default 3.

    Returns
    -------
    pyramid : RenderIndexPyramid or None
        ``None`` if required metadata is missing -- callers should fall back
        to the existing brute-force viewport filter in that case.
    """
    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    if width is None or height is None:
        return None
    return build_render_index_arrays(
        locs["x"].to_numpy(),
        locs["y"].to_numpy(),
        float(width),
        float(height),
        n_levels=n_levels,
    )


def build_render_index_arrays(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    width: float,
    height: float,
    n_levels: int = 3,
) -> RenderIndexPyramid:
    """Build the pyramid from coordinate arrays and the FOV size.

    Same as ``build_render_index`` but on coordinate arrays (camera
    pixels) and the field of view size.

    The permutation sorts the rows by their fine Morton key (the
    Morton code of the cell at ``_FINE_BITS`` levels below the base
    block), so every aligned dyadic square at every level, from the
    quad-tree root down to the finest cell, is one contiguous range of
    it: the block tables of the pyramid levels and the implicit
    quad-tree of ``quadtree_layout`` (rendered by ``picasso.render``)
    both read the same permutation.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        Coordinates of the localizations (camera pixels).
    width, height : float
        Size of the field of view (camera pixels).
    n_levels : int, optional
        Number of pyramid levels, each with blocks 4x larger than the
        last. Default 3.

    Returns
    -------
    pyramid : RenderIndexPyramid
        The spatial index of the localizations.
    """
    base = _base_block_size(width, height)
    block_sizes = tuple(base * (4**lvl) for lvl in range(n_levels))
    L0, K0, root_bits = _quadtree_geometry(width, height, base)

    n = x.shape[0]
    if n == 0:
        block_starts = []
        block_ends = []
        for size in block_sizes:
            K = max(1, int(np.ceil(height / size)))
            L = max(1, int(np.ceil(width / size)))
            block_starts.append(np.zeros((K, L), dtype=np.uint32))
            block_ends.append(np.zeros((K, L), dtype=np.uint32))
        return RenderIndexPyramid(
            perm=np.empty(0, dtype=np.uint32),
            block_sizes=block_sizes,
            block_starts=block_starts,
            block_ends=block_ends,
            width=width,
            height=height,
            root_bits=root_bits,
            fine_bits=_FINE_BITS,
            sorted_keys=np.empty(0, dtype=np.uint64),
        )

    # Cell coords at the finest quad-tree level, clipped to the grid.
    # Out-of-FOV locs are pinned to the boundary so they stay queryable
    # -- matches the existing renderer, which just doesn't draw them.
    keys = _fine_keys(x, y, base, _FINE_BITS, L0, K0)
    # Sort by Morton at the finest level -> hierarchical contiguity.
    perm = np.argsort(keys, kind="stable").astype(np.uint32)
    sorted_keys = keys[perm]

    block_starts = []
    block_ends = []
    for size in block_sizes:
        L = max(1, int(np.ceil(width / size)))
        K = max(1, int(np.ceil(height / size)))
        bx_lvl = np.clip(np.floor(x[perm] / size), 0, L - 1).astype(np.uint32)
        by_lvl = np.clip(np.floor(y[perm] / size), 0, K - 1).astype(np.uint32)
        bs = np.zeros((K, L), dtype=np.uint32)
        be = np.zeros((K, L), dtype=np.uint32)
        _fill_blocks_from_sorted(bx_lvl, by_lvl, bs, be)
        block_starts.append(bs)
        block_ends.append(be)

    return RenderIndexPyramid(
        perm=perm,
        block_sizes=block_sizes,
        block_starts=block_starts,
        block_ends=block_ends,
        width=width,
        height=height,
        root_bits=root_bits,
        fine_bits=_FINE_BITS,
        sorted_keys=sorted_keys,
    )


def _select_level(pyramid: RenderIndexPyramid, viewport: tuple) -> int:
    """Pick the smallest level whose blocks per viewport edge <= target.

    Walking from finest to coarsest means we pick the finest level that
    keeps block iteration bounded -- which also minimizes the gathered
    locs count (more blocks per coarse cell at coarser levels).
    """
    (y_min, x_min), (y_max, x_max) = viewport
    vp_dim = max(x_max - x_min, y_max - y_min)
    for lvl, size in enumerate(pyramid.block_sizes):
        if vp_dim / size <= _TARGET_BLOCKS_PER_EDGE:
            return lvl
    return len(pyramid.block_sizes) - 1


@numba.njit(cache=True)
def _gather_blocks(
    perm: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    cy_min: int,
    cy_max: int,
    cx_min: int,
    cx_max: int,
) -> lib.IntArray1D:
    """Collect original-locs indices from all blocks in the rectangle."""
    total = 0
    for y in range(cy_min, cy_max + 1):
        for x in range(cx_min, cx_max + 1):
            total += block_ends[y, x] - block_starts[y, x]
    out = np.empty(total, dtype=np.uint32)
    pos = 0
    for y in range(cy_min, cy_max + 1):
        for x in range(cx_min, cx_max + 1):
            s = block_starts[y, x]
            e = block_ends[y, x]
            for k in range(s, e):
                out[pos] = perm[k]
                pos += 1
    return out


def query_viewport(
    pyramid: RenderIndexPyramid,
    viewport: tuple,
) -> lib.IntArray1D | None:
    """Indices into the original locs DataFrame for locs in the viewport.

    The returned set is a superset of the strictly-inside locs: a block
    at the viewport edge contributes all of its locs (the renderer's
    own ``in_view`` test inside ``_render_setup`` then prunes the
    overspill, on a tiny array).

    Returns ``None`` when the viewport covers (most of) the FOV --
    above ``_BYPASS_COVERAGE_RATIO`` of the FOV area, or fully
    enclosing it. In that regime gathering ~N indices and copying the
    DataFrame via ``iloc`` costs more than letting the renderer scan
    the full locs with its vectorised ``in_view`` mask. The caller
    treats ``None`` as "no pre-filter, use the full locs".

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index`.
    viewport : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.

    Returns
    -------
    indices : lib.IntArray1D or None
        Positions into the original locs DataFrame, or ``None`` for a
        (near-)full-FOV viewport, as described above.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    # Bypass for (near-)full-FOV viewports -- see module-level constant.
    if (
        x_min <= 0.0
        and y_min <= 0.0
        and x_max >= pyramid.width
        and y_max >= pyramid.height
    ):
        return None
    fov_area = pyramid.width * pyramid.height
    if fov_area > 0.0:
        cx0 = max(0.0, x_min)
        cy0 = max(0.0, y_min)
        cx1 = min(pyramid.width, x_max)
        cy1 = min(pyramid.height, y_max)
        clipped_area = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
        if clipped_area / fov_area >= _BYPASS_COVERAGE_RATIO:
            return None

    if pyramid.perm.shape[0] == 0:
        return np.empty(0, dtype=np.uint32)

    return query_rect(pyramid, viewport)


def query_rect(pyramid: RenderIndexPyramid, rect: tuple) -> lib.IntArray1D:
    """Indices into the original locs DataFrame for locs in ``rect``.

    Unlike ``query_viewport`` there is no full-FOV bypass: the result is
    always an index array, a superset of the locs strictly inside the
    rectangle (whole blocks at its edges are included).

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index`.
    rect : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.

    Returns
    -------
    indices : lib.IntArray1D
        Positions into the original locs DataFrame.
    """
    (y_min, x_min), (y_max, x_max) = rect
    if pyramid.perm.shape[0] == 0:
        return np.empty(0, dtype=np.uint32)
    lvl = _select_level(pyramid, rect)
    size = pyramid.block_sizes[lvl]
    bs = pyramid.block_starts[lvl]
    be = pyramid.block_ends[lvl]
    K, L = bs.shape

    cx_min = int(np.floor(x_min / size))
    cy_min = int(np.floor(y_min / size))
    # x_max/y_max are exclusive in the existing renderer (strict ``<``),
    # so a value landing exactly on a block boundary belongs to the
    # previous block.
    cx_max = int(np.floor((x_max - 1e-9) / size))
    cy_max = int(np.floor((y_max - 1e-9) / size))
    cx_min = max(0, cx_min)
    cy_min = max(0, cy_min)
    cx_max = min(L - 1, cx_max)
    cy_max = min(K - 1, cy_max)
    if cx_min > cx_max or cy_min > cy_max:
        return np.empty(0, dtype=np.uint32)

    return _gather_blocks(pyramid.perm, bs, be, cy_min, cy_max, cx_min, cx_max)


@numba.njit(cache=True)
def _count_blocks_in_rect(
    perm: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    cy_min: int,
    cy_max: int,
    cx_min: int,
    cx_max: int,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> int:
    """Count the locs strictly inside the rectangle. Blocks between the
    edge blocks lie wholly inside it and are counted by size; only the
    edge blocks' locs are tested one by one."""
    n = 0
    for by in range(cy_min, cy_max + 1):
        for bx in range(cx_min, cx_max + 1):
            s = block_starts[by, bx]
            e = block_ends[by, bx]
            if cy_min < by < cy_max and cx_min < bx < cx_max:
                n += e - s
                continue
            for k in range(s, e):
                i = perm[k]
                if (
                    x[i] > x_min
                    and y[i] > y_min
                    and x[i] < x_max
                    and y[i] < y_max
                ):
                    n += 1
    return n


def count_rect(
    pyramid: RenderIndexPyramid,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    rect: tuple,
) -> int:
    """Number of locs strictly inside ``rect``, with the renderer's
    ``in_view`` test (``x_min < x < x_max``, likewise for y).

    Unlike the length of ``query_rect``, which includes whole blocks at
    the edges, the count is exact.

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index` for ``x``, ``y``.
    x, y : lib.FloatArray1D
        Coordinates of all the localizations the pyramid indexes (the
        DataFrame's columns), in camera pixels.
    rect : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.

    Returns
    -------
    n : int
        Number of localizations inside ``rect``.
    """
    (y_min, x_min), (y_max, x_max) = rect
    if pyramid.perm.shape[0] == 0:
        return 0
    lvl = _select_level(pyramid, rect)
    size = pyramid.block_sizes[lvl]
    bs = pyramid.block_starts[lvl]
    be = pyramid.block_ends[lvl]
    K, L = bs.shape
    if x_min >= x_max or y_min >= y_max:
        return 0
    # locs outside the FOV are held by the border blocks, so a rect
    # beyond the FOV is clamped onto them rather than found empty
    cx_min = min(max(0, int(np.floor(x_min / size))), L - 1)
    cy_min = min(max(0, int(np.floor(y_min / size))), K - 1)
    cx_max = min(max(0, int(np.floor((x_max - 1e-9) / size))), L - 1)
    cy_max = min(max(0, int(np.floor((y_max - 1e-9) / size))), K - 1)
    return int(
        _count_blocks_in_rect(
            pyramid.perm,
            bs,
            be,
            cy_min,
            cy_max,
            cx_min,
            cx_max,
            np.asarray(x),
            np.asarray(y),
            float(x_min),
            float(y_min),
            float(x_max),
            float(y_max),
        )
    )


@numba.njit(cache=True)
def _filter_circle(
    indices: lib.IntArray1D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    cx: float,
    cy: float,
    r2: float,
) -> lib.IntArray1D:
    """Keep the indices whose coordinates lie strictly within the
    circle (squared radius ``r2``), as ``lib.is_loc_at_numba`` does."""
    keep = np.empty(indices.shape[0], dtype=np.uint32)
    n = 0
    for k in range(indices.shape[0]):
        i = indices[k]
        dx = x[i] - cx
        dy = y[i] - cy
        if dx * dx + dy * dy < r2:
            keep[n] = i
            n += 1
    return keep[:n]


def query_circle(
    pyramid: RenderIndexPyramid,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    cx: float,
    cy: float,
    radius: float,
) -> lib.IntArray1D:
    """Indices into the original locs DataFrame for locs within a
    circular pick, the way ``picasso.postprocess.picked_locs`` selects
    them (``dx**2 + dy**2 < radius**2``).

    The blocks overlapping the circle's bounding box are gathered from
    the pyramid and the distance test is applied to those locs only.

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The index built by :func:`build_render_index` for ``x``, ``y``.
    x, y : lib.FloatArray1D
        Coordinates of all the localizations the pyramid indexes (the
        DataFrame's columns), in camera pixels.
    cx, cy : float
        Center of the pick in camera pixels.
    radius : float
        Radius of the pick in camera pixels.

    Returns
    -------
    indices : lib.IntArray1D
        Positions into the original locs DataFrame, in the pyramid's
        (Morton) order.
    """
    rect = ((cy - radius, cx - radius), (cy + radius, cx + radius))
    indices = query_rect(pyramid, rect)
    return _filter_circle(indices, x, y, float(cx), float(cy), radius**2)


# ---------------------------------------------------------------------------
# Persistence: the pyramid stored in the localizations' HDF5 file
# ---------------------------------------------------------------------------


def save_render_index(
    hdf_file: h5py.File, pyramid: RenderIndexPyramid
) -> None:
    """Write ``pyramid`` into an open HDF5 file as the group
    ``/render_index``: the permutation and every level's block tables
    as datasets, the block sizes, field size and row count as
    attributes. Older Picasso versions read only ``/locs`` and
    ``/metadata`` and are unaffected by the group.

    Parameters
    ----------
    hdf_file : h5py.File
        The localizations file, open for writing.
    pyramid : RenderIndexPyramid
        The index of the ``/locs`` rows of that file, in their order.
    """
    if RENDER_INDEX_GROUP in hdf_file:
        del hdf_file[RENDER_INDEX_GROUP]
    group = hdf_file.create_group(RENDER_INDEX_GROUP)
    group.attrs["version"] = _RENDER_INDEX_VERSION
    group.attrs["n"] = int(pyramid.perm.shape[0])
    group.attrs["width"] = float(pyramid.width)
    group.attrs["height"] = float(pyramid.height)
    group.attrs["block_sizes"] = np.asarray(
        pyramid.block_sizes, dtype=np.float64
    )
    group.attrs["root_bits"] = int(pyramid.root_bits)
    group.attrs["fine_bits"] = int(pyramid.fine_bits)
    group.create_dataset("perm", data=pyramid.perm)
    for lvl, (bs, be) in enumerate(
        zip(pyramid.block_starts, pyramid.block_ends)
    ):
        group.create_dataset(f"block_starts_{lvl}", data=bs)
        group.create_dataset(f"block_ends_{lvl}", data=be)


def read_render_index(path: str) -> RenderIndexPyramid | None:
    """Read the pyramid stored by ``save_render_index`` in the
    localizations file ``path``; None if the file has none (or it
    cannot be read). The result is *unchecked*: use
    ``load_render_index`` to get one that is known to describe the
    localizations.

    Parameters
    ----------
    path : str
        The localizations HDF5 file.

    Returns
    -------
    pyramid : RenderIndexPyramid or None
    """
    try:
        with h5py.File(path, "r") as hdf_file:
            if RENDER_INDEX_GROUP not in hdf_file:
                return None
            group = hdf_file[RENDER_INDEX_GROUP]
            if int(group.attrs.get("version", 0)) != _RENDER_INDEX_VERSION:
                return None
            block_sizes = tuple(float(s) for s in group.attrs["block_sizes"])
            perm = group["perm"][()].astype(np.uint32, copy=False)
            block_starts = []
            block_ends = []
            for lvl in range(len(block_sizes)):
                block_starts.append(
                    group[f"block_starts_{lvl}"][()].astype(
                        np.uint32, copy=False
                    )
                )
                block_ends.append(
                    group[f"block_ends_{lvl}"][()].astype(
                        np.uint32, copy=False
                    )
                )
            return RenderIndexPyramid(
                perm=perm,
                block_sizes=block_sizes,
                block_starts=block_starts,
                block_ends=block_ends,
                width=float(group.attrs["width"]),
                height=float(group.attrs["height"]),
                root_bits=int(group.attrs["root_bits"]),
                fine_bits=int(group.attrs["fine_bits"]),
            )
    except (OSError, KeyError, ValueError, TypeError):
        return None


@numba.njit(cache=True)
def _is_permutation(perm: lib.IntArray1D, n: int) -> bool:
    """Whether ``perm`` lists every index below ``n`` exactly once."""
    if perm.shape[0] != n:
        return False
    seen = np.zeros(n, dtype=np.uint8)
    for k in range(n):
        p = perm[k]
        if p >= n or seen[p]:
            return False
        seen[p] = 1
    return True


@numba.njit(cache=True)
def _is_sorted(keys: lib.IntArray1D) -> bool:
    """Whether ``keys`` is non-decreasing."""
    for k in range(1, keys.shape[0]):
        if keys[k] < keys[k - 1]:
            return False
    return True


@numba.njit(cache=True)
def _blocks_hold_their_locs(
    perm: lib.IntArray1D,
    block_starts: lib.IntArray2D,
    block_ends: lib.IntArray2D,
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    size: float,
) -> bool:
    """Whether every block's range lists only localizations whose
    (clipped) block coordinates are that block, and the ranges cover
    all ``perm`` entries. This is the correctness criterion of the
    index: as long as it holds, every query is right."""
    n = perm.shape[0]
    K, L = block_starts.shape
    total = 0
    for i in range(K):
        for j in range(L):
            s = block_starts[i, j]
            e = block_ends[i, j]
            if e < s or e > n:
                return False
            total += e - s
            for k in range(s, e):
                p = perm[k]
                bx = int(np.floor(x[p] / size))
                by = int(np.floor(y[p] / size))
                bx = min(max(bx, 0), L - 1)
                by = min(max(by, 0), K - 1)
                if bx != j or by != i:
                    return False
    return total == n


def _validate_index_dimensions(
    pyramid: RenderIndexPyramid, info: list[dict]
) -> bool:
    """Whether the pyramid's field-of-view size matches ``info``."""
    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    if width is None or height is None:
        return False
    return float(width) == pyramid.width and float(height) == pyramid.height


def _validate_index_block_shapes(
    pyramid: RenderIndexPyramid, x: np.ndarray, y: np.ndarray
) -> bool:
    """Whether every block level has consistent shapes and holds its locs."""
    if len(pyramid.block_sizes) != len(pyramid.block_starts) or len(
        pyramid.block_starts
    ) != len(pyramid.block_ends):
        return False
    for size, bs, be in zip(
        pyramid.block_sizes, pyramid.block_starts, pyramid.block_ends
    ):
        K = max(1, int(np.ceil(pyramid.height / size)))
        L = max(1, int(np.ceil(pyramid.width / size)))
        if bs.shape != (K, L) or be.shape != (K, L):
            return False
        if not _blocks_hold_their_locs(
            pyramid.perm, bs, be, x, y, float(size)
        ):
            return False
    return True


def validate_render_index(
    pyramid: RenderIndexPyramid, locs: pd.DataFrame, info: list[dict]
) -> bool:
    """Whether ``pyramid`` correctly indexes ``locs``.

    Checked against the index's own correctness criterion rather than a
    checksum: the permutation covers every row exactly once, every
    block, at every level, holds only rows whose coordinates fall in it
    (one pass per level), and the rows' fine Morton keys are ascending
    along the permutation (the quad-tree's requirement; the keys are
    kept on the pyramid, ``sorted_keys``). Any edit of the file that
    changed, dropped, added or reordered coordinates fails; an edit
    that leaves the index correct (say, other columns) passes, which
    is what matters.

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        A pyramid, e.g. read from the file by ``read_render_index``.
    locs : pd.DataFrame
        The localizations it claims to index, in file order.
    info : list of dicts
        Their metadata (the field size must match the pyramid's).

    Returns
    -------
    valid : bool
    """
    if not _validate_index_dimensions(pyramid, info):
        return False
    n = len(locs)
    if not _is_permutation(pyramid.perm, n):
        return False
    if n == 0:
        pyramid.sorted_keys = np.empty(0, dtype=np.uint64)
        return True
    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()
    if not _validate_index_block_shapes(pyramid, x, y):
        return False
    base = pyramid.block_sizes[0]
    L0, K0, root_bits = _quadtree_geometry(pyramid.width, pyramid.height, base)
    if pyramid.root_bits != root_bits or pyramid.fine_bits <= 0:
        return False
    keys = _fine_keys(x, y, base, pyramid.fine_bits, L0, K0)[pyramid.perm]
    if not _is_sorted(keys):
        return False
    pyramid.sorted_keys = keys
    return True


def load_render_index(
    path: str, locs: pd.DataFrame, info: list[dict]
) -> RenderIndexPyramid | None:
    """The pyramid stored in ``path`` if it (still) describes ``locs``,
    else None -- the caller then builds one with
    ``build_render_index``. A stored index that fails the check (the
    file was edited without ``picasso.io.save_locs``) is reported in
    the log at INFO level.

    Parameters
    ----------
    path : str
        The localizations HDF5 file.
    locs : pd.DataFrame
        The localizations loaded from it.
    info : list of dicts
        Their metadata.

    Returns
    -------
    pyramid : RenderIndexPyramid or None
    """
    pyramid = read_render_index(path)
    if pyramid is None:
        return None
    if not validate_render_index(pyramid, locs, info):
        _log.info(
            "The render index stored in %s does not match its localizations "
            "(the file was modified without picasso.io.save_locs); it is "
            "rebuilt.",
            path,
        )
        return None
    return pyramid


def quadtree_layout(
    pyramid: RenderIndexPyramid,
) -> tuple[lib.IntArray1D, lib.IntArray1D, float, int]:
    """Return the implicit quad-tree of a pyramid.

    Used by the adaptive-histogram renderer
    (``picasso.render.kernels._quadtree_fill``).

    The contract a consumer relies on: the node at depth ``d`` with
    Morton prefix ``p`` (``d`` bits per axis interleaved, x in the even
    bits) holds exactly the rows whose keys lie in
    ``[p << 2 * (total_bits - d), (p + 1) << 2 * (total_bits - d))``,
    a contiguous range of ``perm``; its four children are the prefixes
    ``4 * p + c`` for ``c`` in 0..3, ``c & 1`` being the x half and
    ``c >> 1`` the y half, each of side ``root_px / 2 ** (d + 1)``.

    Parameters
    ----------
    pyramid : RenderIndexPyramid
        The spatial index of a channel.

    Returns
    -------
    sorted_keys : lib.IntArray1D
        The fine Morton key of every row in permutation order
        (ascending).
    perm : lib.IntArray1D
        The permutation.
    root_px : float
        The side of the root square in camera pixels (a power-of-two
        number of base blocks covering the field of view, anchored at
        the origin).
    total_bits : int
        The depth of the tree, i.e. the number of levels from the root
        to the finest cell.

    Raises
    ------
    ValueError
        If the pyramid carries no keys (read from a file but not
        validated).
    """
    if pyramid.sorted_keys is None:
        raise ValueError(
            "the render index carries no sorted keys; validate it against "
            "its localizations or rebuild it"
        )
    root_px = pyramid.block_sizes[0] * (1 << pyramid.root_bits)
    return (
        pyramid.sorted_keys,
        pyramid.perm,
        float(root_px),
        int(pyramid.root_bits + pyramid.fine_bits),
    )
