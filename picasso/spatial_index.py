"""Multi-resolution spatial index for fast viewport rendering.

Built once when a channel is loaded; queried per redraw to skip the
O(N) viewport scan that ``picasso.render._render_setup`` would otherwise
perform on every pan/zoom.

The pyramid stores three grid resolutions sharing a single permutation
sorted by Morton (Z-order) at the finest level. Because Z-order is
hierarchical, each coarser block at level L corresponds to a contiguous
range in the same sorted permutation -- so all levels reuse one ``perm``
array (~4 N bytes) rather than one per level.

See :mod:`picasso.postprocess` for the original single-resolution
``get_index_blocks`` used by pick/cluster code; this module is
intentionally separate so the pick code path is unaffected.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd

from . import lib


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

    Returns ``None`` if required metadata is missing -- callers should
    fall back to the existing brute-force viewport filter in that case.
    """
    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    if width is None or height is None:
        return None
    width = float(width)
    height = float(height)

    base = _base_block_size(width, height)
    block_sizes = tuple(base * (4**lvl) for lvl in range(n_levels))

    n = len(locs)
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
        )

    x = locs["x"].to_numpy()
    y = locs["y"].to_numpy()

    # Block coords at the finest level, clipped to the grid. Out-of-FOV
    # locs are pinned to the boundary so they stay queryable -- matches
    # the existing renderer, which just doesn't draw them.
    n_blocks_x0 = max(1, int(np.ceil(width / base)))
    n_blocks_y0 = max(1, int(np.ceil(height / base)))
    bx0 = np.clip(np.floor(x / base), 0, n_blocks_x0 - 1).astype(np.uint32)
    by0 = np.clip(np.floor(y / base), 0, n_blocks_y0 - 1).astype(np.uint32)

    # Sort by Morton at finest level -> hierarchical contiguity.
    keys = _morton_encode_2d(bx0, by0)
    perm = np.argsort(keys, kind="stable").astype(np.uint32)

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
    )


def _select_level(pyramid: RenderIndexPyramid, viewport: tuple) -> int:
    """Pick the smallest level whose blocks per viewport edge <= target.

    Walking from finest to coarsest means we pick the finest level that
    keeps block iteration bounded -- which also minimises the gathered
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

    lvl = _select_level(pyramid, viewport)
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
