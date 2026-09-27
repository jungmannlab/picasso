"""
picasso.render.splat
~~~~~~~~~~~~~~~~~~~~

Raw splat stage: turn localization coordinates into (blurred) count
images. ``_RenderColumns`` carries the per-localization arrays and the
``_render_*`` functions dispatch per blur method (CPU backend).

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from collections import deque
from concurrent import futures
from typing import Literal

import numba
import numpy as np
import pandas as pd
import psutil
from scipy import signal, ndimage
from scipy.spatial.transform import Rotation

from .. import lib, spatial_index
from . import triangulation
from .backend import SplatBackend
from .kernels import (
    _render_setup,
    _render_setup3d,
    _render_setup3d_anisotropic,
    _fill,
    _fill3d,
    _fill_gaussian,
    _fill_gaussian_theta,
    _fill_gaussian_rot,
    _fill_gaussian_rot_theta,
    _quadtree_fill,
)
from .geometry import to_rotation


def render(
    locs: pd.DataFrame,
    info: dict,
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    max_blur_width: float | None = None,
    ang: tuple | Rotation | None = None,
    indices: lib.IntArray1D | None = None,
    global_precision: tuple[float, float] | None = None,
    quadtree_capacity: int | None = None,
    render_index: spatial_index.RenderIndexPyramid | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations given FOV and blur method.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    info : dict
        Contains localizations metadata.
    disp_px_size : float
        Display pixel size in nm.
    viewport : tuple, optional
        Field of view to be rendered (in camera pixels). The input is
        ``((y_min, x_min), (y_max, x_max))``. If None, all localizations
        are rendered.
    blur_method : {"gaussian", "gaussian_iso", "smooth", "convolve"} or None, \
            optional
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'smooth', 'convolve'. If None, no
        blurring is applied. 'gaussian' uses localization precisions
        of each localization to blur it (different in each dimension).
        'gaussian_iso' is similar but averages x and y localization
        precisions, so that blur is isotropic. 'smooth' applies a one
        pixel blur. 'convolve' applies the same blur to all
        localizations: ``global_precision``, or else the median
        localization precision of the rows rendered.
    min_blur_width : float, optional
        Minimum size of blur (camera pixels).
    max_blur_width : float, optional
        Localizations whose ``lpx`` or ``lpy`` exceeds this (camera
        pixels) are not rendered by 'gaussian' and 'gaussian_iso'
        (see ``_extract_render_columns``). If None (default), all
        localizations are rendered.
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of locs; either a scipy Rotation (e.g. built from a
        quaternion) or a tuple of 3 rotation angles around the x, y
        and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, locs are not rotated.
    indices : lib.IntArray1D, optional
        Positions of the rows of ``locs`` to render (e.g. a viewport
        pre-selection from ``spatial_index``); the other rows are
        ignored. If None (default), all rows are rendered.
    global_precision : tuple of float, optional
        ``(lpx, lpy)`` blur of the 'convolve' method in camera pixels,
        e.g. the median precision of the whole dataset computed once
        by the caller, so the blur is the same at every zoom level and
        rotation and nothing is recomputed per render. If None
        (default), the median of the rows rendered is used.
    quadtree_capacity : int, optional
        Leaf capacity of the 'quadtree' method (bins split while they
        hold more localizations; SNR per bin about
        ``sqrt(capacity / 2)``). If None (default),
        ``lib.RENDER_QUADTREE_CAPACITY_DEFAULT``.
    render_index : spatial_index.RenderIndexPyramid, optional
        The spatial index of ``locs`` (all rows, in order), which the
        'quadtree' method renders from; a GUI passes the one built when
        the channel was loaded. If None (default), it is built here,
        which costs a sort of the rows.
    triangulation_passes : int, optional
        Jittered triangulations averaged by the 'triangulation' method
        (see ``picasso.render.triangulation``); 0 paints the plain
        triangulation. If None (default),
        ``triangulation.PASSES_DEFAULT``.
    triangulation_jitter : float, optional
        Its jitter width in units of each localization's mean distance
        to its neighbors. If None (default),
        ``triangulation.JITTER_DEFAULT``.

    Raises
    ------
    Exception
        If blur_method not one of 'gaussian', 'gaussian_iso', 'smooth',
        'convolve', 'quadtree', 'triangulation' or None.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    return _render_arrays(
        _extract_render_columns(
            locs,
            blur_method,
            ang,
            max_blur_width,
            indices,
            global_precision,
            render_index,
        ),
        info,
        disp_px_size=disp_px_size,
        viewport=viewport,
        blur_method=blur_method,
        min_blur_width=min_blur_width,
        ang=ang,
        quadtree_capacity=quadtree_capacity,
        triangulation_passes=triangulation_passes,
        triangulation_jitter=triangulation_jitter,
    )


class _RenderColumns:
    """Numpy views of the localization columns one render needs.

    Chunked parallel rendering slices these arrays instead of
    DataFrames: the per-chunk pandas overhead (``iloc``, ``to_numpy``)
    holds the GIL and measurably caps the thread pool's efficiency.
    Extraction happens once per channel; ``angle`` is stored in radians
    and ``lpz`` with its fallback already applied, so chunk slices are
    plain array views.
    """

    __slots__ = (
        "x",
        "y",
        "lpx",
        "lpy",
        "lpz",
        "angle",
        "z",
        "indices",
        "global_lp",
        "pyramid",
    )

    def __init__(
        self,
        x,
        y,
        lpx=None,
        lpy=None,
        lpz=None,
        angle=None,
        z=None,
        indices=None,
        global_lp=None,
        pyramid=None,
    ):
        self.x = x
        self.y = y
        self.lpx = lpx
        self.lpy = lpy
        self.lpz = lpz
        self.angle = angle
        self.z = z
        #: rows to render (positions into the column arrays), or None
        #: for all of them: the render-index pyramid's viewport
        #: selection travels this way, so a GPU backend keeps the whole
        #: channel resident and reads only the selected rows
        self.indices = indices
        #: ``(lpx, lpy)`` blur of the ``convolve`` method (camera px,
        #: the channel's median precision), or None to take the median
        #: of the rows given; a GUI computes it once per channel
        self.global_lp = global_lp
        #: the ``spatial_index.RenderIndexPyramid`` of the whole channel
        #: (all rows of ``x``, ``y`` in order) for the ``quadtree``
        #: method, or None to build one on the fly; dropped by ``slice``
        #: and ``materialize`` since it describes the whole channel only
        self.pyramid = pyramid

    def __len__(self) -> int:
        """Number of rows to render."""
        if self.indices is not None:
            return len(self.indices)
        return len(self.x)

    def slice(self, start: int, stop: int) -> "_RenderColumns":
        """Row range (of the rows to render) as array views, no copies."""
        if self.indices is not None:
            return _RenderColumns(
                self.x,
                self.y,
                self.lpx,
                self.lpy,
                self.lpz,
                self.angle,
                self.z,
                self.indices[start:stop],
                global_lp=self.global_lp,
            )

        def cut(array):
            return None if array is None else array[start:stop]

        return _RenderColumns(
            self.x[start:stop],
            self.y[start:stop],
            cut(self.lpx),
            cut(self.lpy),
            cut(self.lpz),
            cut(self.angle),
            cut(self.z),
            global_lp=self.global_lp,
        )

    def materialize(self) -> "_RenderColumns":
        """The selected rows gathered into contiguous arrays (a copy,
        like ``DataFrame.iloc`` with the same indices); a no-op without
        ``indices``."""
        if self.indices is None:
            return self

        def take(array):
            return None if array is None else array[self.indices]

        return _RenderColumns(
            self.x[self.indices],
            self.y[self.indices],
            take(self.lpx),
            take(self.lpy),
            take(self.lpz),
            take(self.angle),
            take(self.z),
            global_lp=self.global_lp,
        )


def _extract_render_columns(
    locs: pd.DataFrame,
    blur_method: str | None,
    ang: tuple | Rotation | None,
    max_blur_width: float | None = None,
    indices: lib.IntArray1D | None = None,
    global_precision: tuple[float, float] | None = None,
    render_index: spatial_index.RenderIndexPyramid | None = None,
) -> _RenderColumns:
    """Pull the columns ``blur_method`` (and rotation) needs out of the
    DataFrame, converting angle to radians and applying the lpz
    fallback once per channel.

    With ``max_blur_width`` (camera pixels), the per-localization blur
    methods (``gaussian``, ``gaussian_iso``) drop localizations whose
    ``lpx`` or ``lpy`` exceeds it: such precisions are useless
    artifacts of unfiltered data, their blur would cover a large FOV
    with a negligible intensity, and rendering them costs too much.
    Filtering here keeps every backend in agreement, including the count
    of rendered localizations.

    ``indices`` (positions into ``locs``) restrict the render to those
    rows without copying the columns (see ``_RenderColumns``); the
    filter above is applied to them as well. ``global_precision`` is
    the ``(lpx, lpy)`` blur of the ``convolve`` method (camera pixels),
    see ``render``. ``render_index`` is the channel's pyramid for the
    ``quadtree`` method; it indexes all rows, so it is only kept when
    no row selection restricts the render."""
    need_lp = blur_method in ("gaussian", "gaussian_iso", "convolve")
    lpx = locs["lpx"].to_numpy() if need_lp else None
    lpy = locs["lpy"].to_numpy() if need_lp else None
    angle = None
    if blur_method == "gaussian" and "angle" in locs:
        # the stored column is in degrees, the kernels expect radians
        angle = np.deg2rad(locs["angle"].to_numpy())
    z = None
    lpz = None
    if ang is not None:
        z = locs["z"].to_numpy()
        if blur_method in ("gaussian", "gaussian_iso"):
            if "lpz" in locs:
                lpz = locs["lpz"].to_numpy()
            else:
                # if lpz not found, make it twice the mean of lpx and lpy
                lpz = 2 * locs[["lpx", "lpy"]].to_numpy().mean(axis=1)
    columns = [locs["x"].to_numpy(), locs["y"].to_numpy(), lpx, lpy, lpz]
    columns += [angle, z]
    if max_blur_width is not None and blur_method in (
        "gaussian",
        "gaussian_iso",
    ):
        keep = (lpx <= max_blur_width) & (lpy <= max_blur_width)
        if indices is not None:
            indices = indices[keep[indices]]
        elif not keep.all():
            columns = [None if c is None else c[keep] for c in columns]
    if indices is not None:
        indices = np.ascontiguousarray(indices, dtype=np.uint32)
    if global_precision is not None:
        global_precision = (
            float(global_precision[0]),
            float(global_precision[1]),
        )
    return _RenderColumns(
        *columns,
        indices=indices,
        global_lp=global_precision,
        pyramid=render_index if indices is None else None,
    )


def _render_arrays(
    columns: _RenderColumns,
    info: dict,
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    ang: tuple | Rotation | None = None,
    quadtree_capacity: int | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """``render`` on pre-extracted column arrays (see ``render`` for
    the parameters). The chunked parallel scheduler calls this per row
    slice so no pandas work happens inside worker tasks."""
    # a row selection is gathered first: the kernels take dense arrays
    columns = columns.materialize()
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    oversampling = pixelsize / disp_px_size

    if viewport is None:
        height = lib.get_from_metadata(info, "Height", raise_error=True)
        width = lib.get_from_metadata(info, "Width", raise_error=True)
        viewport = [(0, 0), (height, width)]

    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        # no blur
        return _render_hist_arrays(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            ang=ang,
        )
    elif blur_method == "gaussian":
        # individual localization precision
        return _render_gaussian(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "gaussian_iso":
        # individual localization precision (same for x and y)
        return _render_gaussian_iso(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "smooth":
        # one pixel blur
        return _render_smooth(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            ang=ang,
        )
    elif blur_method == "convolve":
        # global localization precision
        return _render_convolve(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            ang=ang,
        )
    elif blur_method == "quadtree":
        # adaptive histogram
        return _render_quadtree(
            columns,
            info,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            quadtree_capacity,
            ang=ang,
        )
    elif blur_method == "triangulation":
        # jittered, averaged triangulation
        return _render_triangulation(
            columns,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            triangulation_passes,
            triangulation_jitter,
            ang=ang,
        )
    else:
        raise Exception("blur_method not understood.")


def _render_triangulation(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    passes: int | None,
    jitter: float | None,
    ang: tuple | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """The adaptively jittered, averaged triangulation of Baddeley,
    Cannell & Soeller (2010), see ``picasso.render.triangulation``.
    Rotated (``ang``), the rows in view are projected onto the screen
    first, as for the quad-tree. The passes are spread over the render
    thread pool's budget."""
    if passes is None:
        passes = triangulation.PASSES_DEFAULT
    if jitter is None:
        jitter = triangulation.JITTER_DEFAULT
    workers = _render_worker_budget()
    if ang is not None:
        n_py = int(np.ceil(oversampling * (y_max - y_min)))
        n_px = int(np.ceil(oversampling * (x_max - x_min)))
        xs, ys, _, _ = _locs_rotation_arrays(
            columns, oversampling, x_min, x_max, y_min, y_max, ang
        )
        return triangulation.render_triangulation(
            xs,
            ys,
            1.0,
            ((0.0, 0.0), (float(n_py), float(n_px))),
            passes=int(passes),
            jitter=float(jitter),
            workers=workers,
        )
    return triangulation.render_triangulation(
        columns.x,
        columns.y,
        oversampling,
        ((y_min, x_min), (y_max, x_max)),
        passes=int(passes),
        jitter=float(jitter),
        workers=workers,
    )


def _render_quadtree(
    columns: _RenderColumns,
    info: dict,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    capacity: int | None,
    ang: tuple | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """The quad-tree adaptive histogram of Baddeley, Cannell & Soeller
    (2010): bins are split while they hold more than ``capacity``
    localizations, so every bin has about the same signal-to-noise
    ratio (on average the square root of ``capacity / 2``) and the bin
    size shows the local sampling. Bins never split below a display
    pixel (the paper's truncation, the zoom level of detail), so at an
    overview the image is the histogram; a capacity of 0 gives the
    histogram at any zoom. The image is in localizations per display
    pixel, like the histogram's, and ``n`` is the histogram's count in
    view.

    Unrotated, it renders from the channel's spatial index when
    ``columns`` carries one (``pyramid``), else from an index built
    here (a sort of the rows). Rotated (``ang``), it is the same
    method applied to the projected point set: the rows in view are
    projected onto the screen (``_locs_rotation_arrays``) and a
    tree of the projected coordinates is built per render, which costs
    a sort of the rows in view each time.
    """
    if capacity is None:
        capacity = lib.RENDER_QUADTREE_CAPACITY_DEFAULT
    n_py = int(np.ceil(oversampling * (y_max - y_min)))
    n_px = int(np.ceil(oversampling * (x_max - x_min)))
    image = np.zeros((n_py, n_px), dtype=np.float32)
    if ang is not None:
        # the projected screen coordinates (display pixels) of the rows
        # in view, indexed in that frame: the field is the image
        xs, ys, _, _ = _locs_rotation_arrays(
            columns, oversampling, x_min, x_max, y_min, y_max, ang
        )
        n = len(xs)
        if n and n_py > 0 and n_px > 0:
            pyramid = spatial_index.build_render_index_arrays(
                xs, ys, float(n_px), float(n_py)
            )
            sorted_keys, perm, root_px, total_bits = (
                spatial_index.quadtree_layout(pyramid)
            )
            _quadtree_fill(
                image,
                sorted_keys,
                perm,
                xs,
                ys,
                1.0,
                0.0,
                0.0,
                float(n_py),
                float(n_px),
                root_px,
                total_bits,
                int(capacity),
            )
        return n, image
    pyramid = columns.pyramid
    if pyramid is None or pyramid.sorted_keys is None:
        width = lib.get_from_metadata(info, "Width", raise_error=True)
        height = lib.get_from_metadata(info, "Height", raise_error=True)
        pyramid = spatial_index.build_render_index_arrays(
            columns.x, columns.y, float(width), float(height)
        )
    sorted_keys, perm, root_px, total_bits = spatial_index.quadtree_layout(
        pyramid
    )
    x = columns.x
    y = columns.y
    if len(perm) and n_py > 0 and n_px > 0:
        _quadtree_fill(
            image,
            sorted_keys,
            perm,
            x,
            y,
            float(oversampling),
            float(y_min),
            float(x_min),
            float(y_max),
            float(x_max),
            root_px,
            total_bits,
            int(capacity),
        )
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    return int(in_view.sum()), image


def _render_hist(
    locs: pd.DataFrame,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with no blur by assigning them to pixels.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels)
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels)
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of locs; either a scipy Rotation (e.g. built from a
        quaternion) or a tuple of 3 rotation angles around the x, y
        and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, locs are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    return _render_hist_arrays(
        _extract_render_columns(locs, None, ang),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        ang=ang,
    )


def _render_hist_arrays(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """``_render_hist`` on pre-extracted column arrays."""
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    if ang is not None:
        x, y, _, _ = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
    _fill(image, x, y)
    n = len(x)
    return n, image


@numba.jit(nopython=True, nogil=True)
def render_hist3d(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[int, lib.FloatArray3D]:
    """Render localizations in 3D with no blur by assigning them to
    pixels.

    Parameters
    ----------
    x, y : lib.FloatArray1D
        Lateral coordinates of the localizations (camera pixels).
    z : lib.FloatArray1D
        Axial coordinates of the localizations (nm).
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size in nm, used for converting z coordinates.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray3D
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = _render_setup3d(
        x,
        y,
        z,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        z_min,
        z_max,
        pixelsize,
    )
    _fill3d(image, x, y, z)
    n = len(x)
    return n, image


@numba.jit(nopython=True, nogil=True)
def render_hist3d_anisotropic(
    x: lib.FloatArray1D,
    y: lib.FloatArray1D,
    z: lib.FloatArray1D,
    oversampling_x: float,
    oversampling_y: float,
    oversampling_z: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    z_min: float,
    z_max: float,
    pixelsize: float,
) -> tuple[int, lib.FloatArray3D]:
    """Render localizations in 3D with no blur by assigning them to
    pixels. Allows for different pixel sizes in x, y and z
    (oversampling).

    Parameters
    ----------
    x, y : lib.FloatArray1D
        Lateral coordinates of the localizations (camera pixels).
    z : lib.FloatArray1D
        Axial coordinates of the localizations (nm).
    oversampling_x, oversampling_y, oversampling_z : float (default=1)
        Number of super-resolution pixels per camera pixel in x, y, and
        z directions.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    z_min : float
        Minimum z coordinate to be rendered (nm).
    z_max : float
        Maximum z coordinate to be rendered (nm).
    pixelsize : float
        Camera pixel size in nm, used for converting z coordinates.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray3D
        Rendered 3D image.
    """
    z_min = z_min / pixelsize
    z_max = z_max / pixelsize

    image, n_pixel_y, n_pixel_x, n_pixel_z, x, y, z, in_view = (
        _render_setup3d_anisotropic(
            x,
            y,
            z,
            oversampling_x,
            oversampling_y,
            oversampling_z,
            y_min,
            x_min,
            y_max,
            x_max,
            z_min,
            z_max,
            pixelsize,
        )
    )
    _fill3d(image, x, y, z)
    n = len(x)
    return n, image


def _render_gaussian(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with individual precisions in x and y.

    The localization precision of each localization differs in x and y.

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    min_blur_width : float
        Minimum localization precision (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of localizations; either a scipy Rotation (e.g. built
        from a quaternion) or a tuple of 3 rotation angles around the
        x, y and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang is None:  # not rotated
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        sy = blur_height[in_view]
        sx = blur_width[in_view]

        if columns.angle is not None:
            # per-localization in-plane rotation of the precision
            # ellipse (already converted to radians at extraction)
            angle = columns.angle[in_view]
            _fill_gaussian_theta(
                image, x, y, sx, sy, angle, n_pixel_x, n_pixel_y
            )
        else:
            _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        # lpz carries its fallback from extraction already
        blur_depth = oversampling * np.maximum(columns.lpz, min_blur_width)

        sy = blur_height[in_view]
        sx = blur_width[in_view]
        sz = blur_depth[in_view]

        rot_matrix = np.ascontiguousarray(
            to_rotation(ang).as_matrix(), dtype=np.float32
        )
        if columns.angle is not None:
            # per-localization in-plane rotation (radians), composed
            # with the global rotation
            angle = columns.angle[in_view]
            _fill_gaussian_rot_theta(
                image,
                x,
                y,
                sx,
                sy,
                sz,
                angle,
                n_pixel_x,
                n_pixel_y,
                rot_matrix,
            )
        else:
            _fill_gaussian_rot(
                image, x, y, sx, sy, sz, n_pixel_x, n_pixel_y, rot_matrix
            )

    n = len(x)
    return n, image


def _render_gaussian_iso(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Same as ``_render_gaussian``, but uses the same localization
    precision in x and y."""
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang is None:  # not rotated
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy

        _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    else:  # rotated
        x, y, in_view, z = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )
        blur_width = oversampling * np.maximum(columns.lpx, min_blur_width)
        blur_height = oversampling * np.maximum(columns.lpy, min_blur_width)
        # lpz carries its fallback from extraction already
        blur_depth = oversampling * np.maximum(columns.lpz, min_blur_width)

        sy = (blur_height[in_view] + blur_width[in_view]) / 2
        sx = sy
        sz = blur_depth[in_view]

        # isotropic in-plane blur: per-loc rotation about z has no effect
        rot_matrix = np.ascontiguousarray(
            to_rotation(ang).as_matrix(), dtype=np.float32
        )
        _fill_gaussian_rot(
            image, x, y, sx, sy, sz, n_pixel_x, n_pixel_y, rot_matrix
        )

    return len(x), image


def _render_convolve(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    min_blur_width: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with global localization precision,
    i.e. each localization is blurred by the median localization
    precision in x and y.

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    min_blur_width : float
        Minimum localization precision (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of localizations; either a scipy Rotation (e.g. built
        from a quaternion) or a tuple of 3 rotation angles around the
        x, y and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    if ang is not None:  # rotate
        x, y, in_view, _ = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        lpx, lpy = global_blur(columns)
        blur_width = oversampling * max(lpx, min_blur_width)
        blur_height = oversampling * max(lpy, min_blur_width)
        return n, _fftconvolve(image, blur_width, blur_height)


def global_blur(columns: _RenderColumns) -> tuple[float, float]:
    """Find the ``convolve`` blur of a channel in camera pixels.

    The blur is the caller's ``global_precision`` when given (see
    ``render``), else the median ``lpx`` and ``lpy`` of the rows to
    render. It is the same for every backend, zoom level and rotation,
    so an in-view mask (which a 3D rotation would make expensive) is
    never needed.

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the channel's localizations.

    Returns
    -------
    lpx : float
        Blur in x (camera pixels); 0 if there are no rows to render.
    lpy : float
        Blur in y (camera pixels); 0 if there are no rows to render.
    """
    if columns.global_lp is not None:
        return columns.global_lp
    lpx, lpy = columns.lpx, columns.lpy
    if columns.indices is not None:
        lpx, lpy = lpx[columns.indices], lpy[columns.indices]
    if len(lpx) == 0:
        return 0.0, 0.0
    return float(np.median(lpx)), float(np.median(lpy))


def _render_smooth(
    columns: _RenderColumns,
    oversampling: float,
    y_min: float,
    x_min: float,
    y_max: float,
    x_max: float,
    ang: tuple[float, float, float] | Rotation | None = None,
) -> tuple[int, lib.FloatArray2D]:
    """Render localizations with blur of one display pixel (set by
    oversampling).

    Parameters
    ----------
    columns : _RenderColumns
        Column arrays of the localizations to be rendered.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    y_min, x_min : float
        Minimum y and x coordinates to be rendered (camera pixels).
    y_max, x_max : float
        Maximum y and x coordinates to be rendered (camera pixels).
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of localizations; either a scipy Rotation (e.g. built
        from a quaternion) or a tuple of 3 rotation angles around the
        x, y and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, localizations are not rotated.

    Returns
    -------
    n : int
        Number of localizations rendered.
    image : lib.FloatArray2D
        Rendered image.
    """
    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        columns.x,
        columns.y,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    if ang is not None:  # rotate
        x, y, _, _ = _locs_rotation_arrays(
            columns,
            oversampling,
            x_min,
            x_max,
            y_min,
            y_max,
            ang,
        )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(
    image: lib.FloatArray2D,
    blur_width: float,
    blur_height: float,
) -> lib.FloatArray2D:
    """Blur (convolves) 2D image using fast fourier transform or with
    Gaussian filter applied (faster for small kernels).

    Parameters
    ----------
    image : lib.FloatArray2D
        Image with rendered but not blurred localizations.
    blur_width, blur_height : float
        Blur width and height in pixels.

    Returns
    -------
    image : lib.FloatArray2D
        Blurred image.
    """
    kernel_width = 10 * int(np.round(blur_width)) + 1
    kernel_height = 10 * int(np.round(blur_height)) + 1
    # Spatial separable convolution is faster than FFT for the small
    # kernels typical of SMLM precisions (~1-3 px). Switch to FFT only
    # when the kernel is large relative to the image.
    n_y, n_x = image.shape
    spatial = (
        kernel_height < 0.05 * n_y
        and kernel_width < 0.05 * n_x
        and max(kernel_height, kernel_width) <= 101
    )
    if spatial:
        out = np.empty_like(image, dtype=np.float32)
        ndimage.gaussian_filter(
            image,
            sigma=(blur_height, blur_width),
            output=out,
            mode="constant",
            cval=0.0,
            truncate=5.0,
        )
        return out
    kernel_y = signal.windows.gaussian(kernel_height, blur_height)
    kernel_x = signal.windows.gaussian(kernel_width, blur_width)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    image = signal.fftconvolve(image, kernel, mode="same")
    return image.astype(np.float32)


def locs_rotation(
    locs: pd.DataFrame,
    oversampling: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ang: tuple[float, float, float] | Rotation,
) -> tuple[
    lib.FloatArray1D, lib.FloatArray1D, lib.BoolArray1D, lib.FloatArray1D
]:
    """Rotate localizations within a FOV.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be rotated.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    x_min, x_max : float
        Minimum and maximum x coordinates to be rendered (camera
        pixels).
    y_min, y_max : float
        Minimum and maximum y coordinates to be rendered (camera
        pixels).
    ang : tuple or scipy.spatial.transform.Rotation
        Rotation of localizations; either a scipy Rotation or a tuple
        of 3 rotation angles around the x, y and z axes in radians
        (legacy Euler convention, see ``rotation_matrix``).

    Returns
    -------
    x : lib.FloatArray1D
        New (rotated) x coordinates.
    y : lib.FloatArray1D
        New y coordinates.
    in_view : lib.BoolArray1D
        Mask of the locs that are rendered.
    z : lib.FloatArray1D
        New z coordinates.
    """
    return _locs_rotation_arrays(
        _extract_render_columns(locs, None, ang),
        oversampling,
        x_min,
        x_max,
        y_min,
        y_max,
        ang,
    )


def _locs_rotation_arrays(
    columns: _RenderColumns,
    oversampling: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ang: tuple[float, float, float] | Rotation,
) -> tuple[
    lib.FloatArray1D, lib.FloatArray1D, lib.BoolArray1D, lib.FloatArray1D
]:
    """``locs_rotation`` on pre-extracted column arrays."""
    # z is translated to pixels
    locs_coord = np.column_stack((columns.x, columns.y, columns.z))

    # x and y are in range (x_min/y_min, x_max/y_max) so they need to be
    # shifted (scipy rotation is around origin)
    locs_coord[:, 0] -= x_min + (x_max - x_min) / 2
    locs_coord[:, 1] -= y_min + (y_max - y_min) / 2

    # rotate locs
    R = to_rotation(ang)
    locs_coord = R.apply(locs_coord)

    # unshift locs
    locs_coord[:, 0] += x_min + (x_max - x_min) / 2
    locs_coord[:, 1] += y_min + (y_max - y_min) / 2

    # output
    x = locs_coord[:, 0]
    y = locs_coord[:, 1]
    z = locs_coord[:, 2]
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    z = z[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    z *= oversampling
    return x, y, in_view, z


#: Blur methods whose channels may be split into row chunks rendered
#: separately and summed ("pseudo-channels"). Per-localization methods
#: (hist, gaussian, gaussian_iso) are additive by construction. 'smooth'
#: is exact as well: its one-pixel blur kernel is identical for every
#: chunk and convolution is linear; the extra per-chunk image filters
#: are cheap next to the per-loc fill, and ``_MIN_CHUNK_LOCS`` keeps the
#: few-locs/big-image regime whole. 'convolve' must not be chunked: its
#: blur width is the median precision of the rendered locs, and chunk
#: medians differ from the global median.
_CHUNKABLE_BLUR_METHODS = (None, "gaussian", "gaussian_iso", "smooth")

#: Minimum rows per chunk: below this, per-task overhead (image
#: allocation and summing, ~1 ms) stops being negligible against the
#: fill work it parallelizes. Found empirically on real data.
_MIN_CHUNK_LOCS = 100_000


def _render_worker_budget() -> int:
    """Number of worker threads the render pool may use, from the
    ``Render`` section of the user settings file (see
    ``lib.n_workers``)."""
    return lib.n_workers(
        lib.RENDER_CPU_UTILIZATION_DEFAULT, settings_section="Render"
    )


def _image_bytes(
    info: list[list[dict]],
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None,
) -> int:
    """Size in bytes of one channel's float32 image for the render
    (the largest over the channels' pixel sizes; the field of view
    without a viewport)."""
    n_bytes = 0
    for channel_info in info:
        pixelsize = lib.get_from_metadata(channel_info, "Pixelsize")
        if pixelsize is None:
            continue
        oversampling = float(pixelsize) / disp_px_size
        if viewport is None:
            height = lib.get_from_metadata(channel_info, "Height") or 0
            width = lib.get_from_metadata(channel_info, "Width") or 0
        else:
            (y_min, x_min), (y_max, x_max) = viewport
            height = y_max - y_min
            width = x_max - x_min
        n_py = int(np.ceil(oversampling * height))
        n_px = int(np.ceil(oversampling * width))
        n_bytes = max(n_bytes, 4 * max(n_py, 0) * max(n_px, 0))
    return n_bytes


#: Bytes a chunk task needs per row on top of its image: the in-view
#: copies of x, y and the blur widths (float64) and the mask.
_CHUNK_BYTES_PER_ROW = 40
#: Share of the currently available memory a render may take for its
#: chunk images in flight.
_RENDER_MEMORY_SHARE = 0.5
#: Chunk tasks in flight per worker (their images may be alive at
#: once): with one, the pool idles whenever the oldest task, the
#: biggest, is still running while the others finished (measured 2.7x
#: slower on the 12-plex overview); two keeps it busy. Every one costs
#: a full-size image, which ``_memory_bounded_workers`` accounts for.
_CHUNKS_IN_FLIGHT_PER_WORKER = 2


def _memory_bounded_workers(
    n_workers: int, n_channels: int, image_bytes: int, rows_per_task: int
) -> int:
    """Reduce ``n_workers`` so that the chunk images in flight
    (``_CHUNKS_IN_FLIGHT_PER_WORKER`` per worker plus the one being
    summed), their per-row scratch and the channels' summed images fit in
    ``_RENDER_MEMORY_SHARE`` of the memory available right now. Never
    below 1: a single worker renders one chunk at a time, the same
    footprint as the sequential path."""
    try:
        available = psutil.virtual_memory().available
    except Exception:
        return n_workers
    per_task = 2 * image_bytes + _CHUNK_BYTES_PER_ROW * rows_per_task
    if per_task <= 0:
        return n_workers
    room = _RENDER_MEMORY_SHARE * available - n_channels * image_bytes
    tasks_that_fit = room // per_task - 1
    return int(
        max(1, min(n_workers, tasks_that_fit // _CHUNKS_IN_FLIGHT_PER_WORKER))
    )


def _chunk_tasks(
    n_locs_per_channel: list[int], budget: int
) -> list[tuple[int, int, int]]:
    """Split channels into ``(channel, start, stop)`` row-chunk tasks.

    Aims for about two tasks per worker across the whole render so the
    pool stays load-balanced regardless of channel-size skew, while
    keeping every chunk at least ``_MIN_CHUNK_LOCS`` rows.

    Parameters
    ----------
    n_locs_per_channel : list of int
        Number of localizations per channel.
    budget : int
        Worker threads available to the render pool.

    Returns
    -------
    tasks : list of (int, int, int)
        ``(channel index, start row, stop row)`` per task; the chunks of
        each channel tile it exactly, in ascending row order.
    """
    total = sum(n_locs_per_channel)
    target = max(_MIN_CHUNK_LOCS, -(-total // (2 * budget)))
    tasks = []
    for i, n in enumerate(n_locs_per_channel):
        k = min(
            max(1, n // _MIN_CHUNK_LOCS),
            max(1, -(-n // target)),
        )
        bounds = np.linspace(0, n, k + 1).round().astype(np.int64)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            tasks.append((i, int(start), int(stop)))
    return tasks


class CpuBackend(SplatBackend):
    """CPU reference splat backend: the render thread pool.

    For per-localization blur methods (``_CHUNKABLE_BLUR_METHODS``),
    channels are additionally split into row chunks rendered as
    independent tasks whose images are summed per channel — splatting is
    additive, so any row partition yields the same image up to float
    summation order. One flat task pool covers all channels and chunks,
    so the load stays balanced regardless of channel-size skew and a
    single large channel also renders in parallel. Chunk images are
    summed in fixed row order, making the result deterministic for a
    given worker budget; with a budget of 1 the exact legacy sequential
    path runs. Threads give real parallelism because the fill kernels
    release the GIL (``nogil=True``) and never mutate their inputs. The
    pool only lives for the duration of one render, keeping Picasso
    polite on shared workstations; a single channel too small to chunk
    renders on the calling thread without reading the settings file.

    Safe to call concurrently: every render builds its own task list
    and pool, and the kernels never mutate their inputs.
    """

    name = "cpu"

    def render_channels(
        self,
        columns: list[_RenderColumns],
        info: list[list[dict]],
        *,
        disp_px_size: float,
        viewport: tuple[tuple[float, float], tuple[float, float]] | None,
        blur_method: (
            Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
        ),
        min_blur_width: float,
        ang: tuple | Rotation | None,
        quadtree_capacity: int | None = None,
        triangulation_passes: int | None = None,
        triangulation_jitter: float | None = None,
    ) -> list[tuple[int, lib.FloatArray2D]]:
        """See ``backend.SplatBackend.render_channels``."""

        def render_rows(i: int, start: int, stop: int):
            chunk = columns[i]
            if stop - start < len(chunk):
                chunk = chunk.slice(start, stop)
            return _render_arrays(
                chunk,
                info[i],
                disp_px_size=disp_px_size,
                viewport=viewport,
                blur_method=blur_method,
                min_blur_width=min_blur_width,
                ang=ang,
                quadtree_capacity=quadtree_capacity,
                triangulation_passes=triangulation_passes,
                triangulation_jitter=triangulation_jitter,
            )

        n_channels = len(columns)
        sizes = [len(channel) for channel in columns]
        total = sum(sizes)
        chunkable = blur_method in _CHUNKABLE_BLUR_METHODS
        # a single channel too small to chunk renders on the calling
        # thread without touching the settings file
        if n_channels == 1 and (not chunkable or total < 2 * _MIN_CHUNK_LOCS):
            return [render_rows(0, 0, total)]

        tasks, n_workers = self._plan_render_tasks(
            sizes, n_channels, chunkable, info, disp_px_size, viewport
        )
        return self._run_render_tasks(
            tasks, render_rows, n_channels, n_workers
        )

    @staticmethod
    def _plan_render_tasks(
        sizes, n_channels, chunkable, info, disp_px_size, viewport
    ):
        """Build the ``(channel, start, stop)`` task list and worker
        count for ``render_channels``'s thread pool."""
        budget = _render_worker_budget()
        if chunkable and budget > 1:
            tasks = _chunk_tasks(sizes, budget)
        else:
            tasks = [(i, 0, n) for i, n in enumerate(sizes)]
        n_workers = min(len(tasks), budget)
        if n_workers > 1:
            n_workers = _memory_bounded_workers(
                n_workers,
                n_channels,
                _image_bytes(info, disp_px_size, viewport),
                max(stop - start for _, start, stop in tasks),
            )
        return tasks, n_workers

    @staticmethod
    def _run_render_tasks(tasks, render_rows, n_channels, n_workers):
        """Run ``tasks`` on the calling thread (``n_workers == 1``) or a
        bounded thread pool, summing chunk results per channel.

        Every chunk renders a full-size image; they are summed into
        the channel's image in submission order (biggest tasks first,
        so none serializes the tail of the pool; a fixed order, so a
        given worker budget always produces the same float rounding)
        with at most two per worker plus one alive -- never one per
        task, which exhausted the memory of workstations with many
        cores and large windows (Windows does not overcommit); the
        worker count itself is bounded by the memory available.
        """
        accumulated: list[tuple[int, lib.FloatArray2D] | None] = [None] * (
            n_channels
        )

        def accumulate(task, result):
            i = task[0]
            n, image = result
            if accumulated[i] is None:
                accumulated[i] = (n, image)
            else:
                n_total, total_image = accumulated[i]
                total_image += image
                accumulated[i] = (n_total + n, total_image)

        if n_workers == 1:
            for task in tasks:
                accumulate(task, render_rows(*task))
        else:
            order = sorted(tasks, key=lambda t: t[2] - t[1], reverse=True)
            window = _CHUNKS_IN_FLIGHT_PER_WORKER * n_workers
            pending: deque = deque()
            with futures.ThreadPoolExecutor(n_workers) as executor:
                for task in order:
                    pending.append((task, executor.submit(render_rows, *task)))
                    if len(pending) > window:
                        done_task, future = pending.popleft()
                        accumulate(done_task, future.result())
                while pending:
                    done_task, future = pending.popleft()
                    accumulate(done_task, future.result())
        return [result for result in accumulated]
