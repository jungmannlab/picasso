"""
picasso.render.scene
~~~~~~~~~~~~~~~~~~~~

Compose rendered scenes: parallel per-channel rendering, contrast
scaling, colormaps and multi-channel blending.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import logging
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from .. import lib
from .kernels import _compose_multi_lut, _quantize_rgb, _compose_single
from .backend import (
    SplatBackendError,
    _cpu_backend,
    _get_backend,
    note_fallback,
)
from .splat import _extract_render_columns
from .overlays_qt import rgb_to_qimage

if TYPE_CHECKING:
    from PyQt6 import QtGui
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")

_log = logging.getLogger(__name__)


N_GROUP_COLORS = 8


def solid_to_lut(rgb: tuple[float, float, float]) -> lib.FloatArray2D:
    """Build a (256, 3) float32 LUT that linearly ramps from black to
    the given RGB color.

    The returned LUT is the input format expected by
    :func:`_render_multi_channel` (and therefore :func:`render_scene`)
    when colors are passed as per-channel lookup tables. A solid-color
    channel rendered through this LUT is mathematically identical to
    the legacy ``intensity * rgb`` blend.

    Parameters
    ----------
    rgb : sequence of 3 floats
        Target RGB color, each component in range [0, 1].

    Returns
    -------
    lut : lib.FloatArray2D
        LUT with generated colormap of shape (256, 3).

    Examples
    --------
    >>> lut = solid_to_lut((1.0, 0.0, 0.0))   # black -> red
    >>> render_scene(locs=..., info=..., colors=[lut, ...], ...)
    """
    rgb_arr = np.asarray(rgb, dtype=np.float32).reshape(3)
    return np.linspace(
        np.zeros(3, dtype=np.float32), rgb_arr, 256, dtype=np.float32
    )


def stops_to_lut(
    stops: list[tuple[float, float, float, float]],
) -> lib.FloatArray2D:
    """Build a (256, 3) float32 LUT by linearly interpolating between
    color stops.

    Parameters
    ----------
    stops : sequence of (position, r, g, b) tuples
        Each ``position`` must be in [0, 1], strictly increasing, with
        the first stop at 0.0 and the last at 1.0. ``r``, ``g``, ``b``
        are also in [0, 1].

    Returns
    -------
    lut : lib.FloatArray2D
        LUT with generated colormap of shape (256, 3).

    Examples
    --------
    A 3-stop "fire" gradient (black -> red -> yellow):

    >>> lut = stops_to_lut([
    ...     (0.0, 0, 0, 0),
    ...     (0.5, 1, 0, 0),
    ...     (1.0, 1, 1, 0),
    ... ])
    >>> render_scene(locs=..., info=..., colors=[lut], ...)
    """
    arr = np.asarray(stops, dtype=np.float32)
    positions = arr[:, 0]
    rgb = arr[:, 1:4]
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.empty((256, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(x, positions, rgb[:, c])
    return lut


def get_colors_from_colormap(
    n_channels: int,
    cmap: str = "gist_rainbow",
) -> list[tuple[float, float, float]]:
    """Create a list of RGB colors for rendering by property.

    One color is drawn per channel from the colormap (gist_rainbow by
    default), see
    https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Parameters
    ----------
    n_channels : int
        Number of locs channels.
    cmap : str, optional
        Colormap name. Default is 'gist_rainbow'.

    Returns
    -------
    colors : list of tuples
        Contains tuples with RGB channels ranging between 0 and 1.
    """
    # array of shape (256, 3) with RGB channels with 256 colors
    base = plt.get_cmap(cmap)(np.arange(256))[:, :3]
    # indeces to draw from base
    idx = np.linspace(0, 255, n_channels).astype(int)
    # extract the colors of interest
    colors = base[idx]
    return colors  # value ranging between 0 and 1


def get_group_color(
    locs: pd.DataFrame,
    shuffle: bool = False,
) -> lib.IntArray1D:
    """Find group color for each localization in single channel data
    with group info.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations. Must contain a ``group`` column.
    shuffle : bool, optional
        If True, build a lookup of ``np.arange(max(group) + 1)``,
        randomly permute it, and take it mod ``N_GROUP_COLORS`` before
        indexing by ``group``. This scatters adjacent group ids across
        color slots. Default is False (plain ``group % N_GROUP_COLORS``).

    Returns
    -------
    colors : lib.IntArray1D
        Array with integer group color index for each localization.
    """
    groups = locs["group"].to_numpy().astype(int)
    if shuffle:
        lookup = np.arange(groups.max() + 1)
        np.random.shuffle(lookup)
        lookup %= N_GROUP_COLORS
        return lookup[groups]
    return groups % N_GROUP_COLORS


def render_scene(
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    *,
    disp_px_size: float = 100.0,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    max_blur_width: float | None = None,
    ang: tuple | Rotation | None = None,
    indices: list | None = None,
    global_precision: list | tuple[float, float] | None = None,
    quadtree_capacity: int | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
    render_index: list | None = None,
    contrast: tuple[float, float] | None = None,
    invert_colors: bool = False,
    single_channel_colormap: str | lib.FloatArray2D = "magma",
    colors: list | None = None,
    relative_intensities: list[float] | None = None,
    background_color: tuple[float, float, float] | None = None,
    raw_image_cache: lib.FloatArray2D | lib.FloatArray3D | None = None,
    return_contrast_limits: bool = False,
    return_raw_image: bool = False,
) -> (
    tuple[QtGui.QImage, int]
    | tuple[QtGui.QImage, int, tuple[float, float]]
    | tuple[QtGui.QImage, int, lib.FloatArray2D | lib.FloatArray3D]
    | tuple[
        QtGui.QImage,
        int,
        tuple[float, float],
        lib.FloatArray2D | lib.FloatArray3D,
    ]
):
    """Render localizations into a colored image (either QImage or a
    numpy array).

    For single channel images without group info, the colormap is
    specified by `single_channel_colormap`. For single channel images
    with group info, the colormap is determined by `get_group_color` and
    `lib.get_colors`. For multi-channel images, the colors are specified
    by `colors`.

    If `raw_image_cache` is provided (the raw grayscale image of
    localizations, i.e., obtained with ``render.render``; 2D array for
    single-channel data, 3D array for multi-channel data), some of the
    arguments are not used: `locs`, `info`, `disp_px_size`, `viewport`,
    `blur_method`, `min_blur_width`, `ang`.

    Optionally, the user can request the raw grayscale image of
    localizations and/or the contrast limits used for scaling to be
    returned together with the rendered QImage and number of
    localizations rendered.

    Parameters
    ----------
    locs : pd.DataFrame or list of pd.DataFrame
        Localizations to be rendered. Can be either one localization
        file or a list thereof. If a single DataFrame is provided,
        localizations will be rendered in a single channel, i.e., using
        a color map specified by `single_channel_colormap`. If a list of
        DataFrames is provided, localizations will be rendered in
        multiple channels, and the color of each channel can be
        specified by `colors`.
    info : list of dict or list of list of dict
        List of info dictionaries corresponding to the localization
        file(s).
    disp_px_size : float, optional
        Display pixel size in nm. Default is 100.0.
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
        localizations which is the median localization precision.
    min_blur_width : float, optional
        Minimum size of blur (camera pixels).
    max_blur_width : float, optional
        Localizations whose ``lpx`` or ``lpy`` exceeds this (camera
        pixels) are not rendered by 'gaussian' and 'gaussian_iso'.
        If None (default), all localizations are rendered.
    ang : tuple or scipy.spatial.transform.Rotation, optional
        Rotation of locs; either a scipy Rotation (e.g. built from a
        quaternion) or a tuple of 3 rotation angles around the x, y
        and z axes in radians (legacy Euler convention, see
        ``rotation_matrix``). If None, locs are not rotated.
    indices : list of lib.IntArray1D, optional
        For each channel (a single array for a single ``locs``
        DataFrame), the positions of the rows to render, e.g. a
        viewport pre-selection; None entries render every row. If None
        (default), all rows of every channel are rendered.
    global_precision : list of tuple or tuple, optional
        For each channel (a single ``(lpx, lpy)`` for a single ``locs``
        DataFrame), the blur of the 'convolve' method in camera pixels
        (see ``render``); None entries use the median precision of the
        rows rendered.
    quadtree_capacity : int, optional
        Leaf capacity of the 'quadtree' method, see ``render``. If None
        (default), ``lib.RENDER_QUADTREE_CAPACITY_DEFAULT``.
    triangulation_passes : int, optional
        Number of passes averaged by the 'triangulation' method, see
        ``render``. Default is None (the method's default).
    triangulation_jitter : float, optional
        Jitter width of the 'triangulation' method, see ``render``.
        Default is None (the method's default).
    render_index : list of spatial_index.RenderIndexPyramid or None, \
            optional
        Per channel, its spatial index for the 'quadtree' method (see
        ``render``); None entries (or None) build one on the fly.
        Default is None.
    contrast : tuple of float, optional
        Contrast limits for scaling. If None, contrast is automatically
        determined.
    invert_colors : bool, optional
        If True, invert colors of the rendered image. Default is False.
    single_channel_colormap : str | lib.FloatArray2D, optional
        Colormap to use for single channel data. If a str, the
        corresponding pyplot colormap is selected. If a 2D array, a
        256x4  array is expected with values between 0 and 1. Default is
        'magma'.
    colors : list of tuples or list of lib.FloatArray2D, optional
        Colors of the channels, one entry per channel. Each entry is
        either an ``(r, g, b)`` tuple with values between 0 and 1
        (the channel is rendered as ``intensity * rgb``) or a
        ``(256, 3)`` lookup table with values between 0 and 1 (the
        channel intensity is indexed into the LUT, allowing
        per-channel colormaps; see ``solid_to_lut`` and
        ``stops_to_lut``). The two forms cannot be mixed. Channels are
        blended additively. Only needs to be specified for
        multi-channel data. Default is None, in which case colors are
        taken from ``lib.get_colors``.
    relative_intensities : list of float, optional
        List of relative intensities for each channel. Only needs to be
        specified for multi-channel data. Default is None, in which
        case all channels are rendered with the same intensity.
    background_color : tuple of float, optional
        ``(r, g, b)`` background color with values between 0 and 1 that
        the additively-blended channels are composited over (multi-channel
        data only). The background shows through where there are few/no
        localizations, while bright regions keep their true channel
        colors. Default is None (equivalent to a black background, i.e.
        the image is left unchanged).
    raw_image_cache : lib.FloatArray2D or lib.FloatArray3D, optional
        If provided, this raw grayscale image of localizations, i.e.,
        obtained with ``render.render`` (2D array for single-channel
        data, 3D array for multi-channel data) is used instead of
        recomputing it. Some of the arguments are not used if this is
        provided: `locs`, `info`, `disp_px_size`, `viewport`,
        `blur_method`, `min_blur_width`, `ang`.
    return_contrast_limits : bool, optional
        If True, return the contrast limits used for scaling. Default is
        False.
    return_raw_image : bool, optional
        If True, return the raw grayscale image of localizations (2D
        array for single-channel data, 3D array for multi-channel data).
        Default is False.

    Returns
    -------
    qimage : QtGui.QImage
        RGB image of rendered localizations as a QImage object.
    n_locs : int
        Total number of localizations rendered.
    contrast_limits : tuple of float, optional
        The contrast limits used for scaling. Only returned if
        return_contrast_limits is True.
    raw_image : FloatArray2D or FloatArray3D, optional
        Raw grayscale image of localizations (2D array for single-channel
        data, 3D array for multi-channel data). Only returned if
        return_raw_image is True.
    """
    if isinstance(locs, pd.DataFrame):
        n_locs, rgb, contrast_limits, raw_image = _render_single_channel(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            max_blur_width=max_blur_width,
            ang=ang,
            indices=indices,
            global_precision=global_precision,
            quadtree_capacity=quadtree_capacity,
            triangulation_passes=triangulation_passes,
            triangulation_jitter=triangulation_jitter,
            render_index=render_index,
            contrast=contrast,
            invert_colors=invert_colors,
            single_channel_colormap=single_channel_colormap,
            raw_image_cache=raw_image_cache,
        )
    elif len(locs) == 0:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        n_locs = 0
        contrast_limits = contrast if contrast is not None else (0.0, 1.0)
        raw_image = np.zeros((1, 1), dtype=np.float32)
    else:
        if colors is not None:
            assert len(colors) == len(locs) == len(info), (
                f"Mismatch between {len(colors)} colors, {len(locs)} "
                f"localization files, and {len(info)} info dictionaries."
            )
        else:
            assert len(locs) == len(info), (
                f"Mismatch between {len(locs)} localization files and "
                f"{len(info)} info dictionaries."
            )
        n_locs, rgb, contrast_limits, raw_image = _render_multi_channel(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size,
            colors=colors,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            max_blur_width=max_blur_width,
            ang=ang,
            indices=indices,
            global_precision=global_precision,
            quadtree_capacity=quadtree_capacity,
            triangulation_passes=triangulation_passes,
            triangulation_jitter=triangulation_jitter,
            render_index=render_index,
            contrast=contrast,
            relative_intensities=relative_intensities,
            invert_colors=invert_colors,
            background_color=background_color,
            raw_image_cache=raw_image_cache,
        )
    qimage = rgb_to_qimage(rgb)
    if return_raw_image and return_contrast_limits:
        return qimage, n_locs, contrast_limits, raw_image
    elif return_raw_image:
        return qimage, n_locs, raw_image
    elif return_contrast_limits:
        return qimage, n_locs, contrast_limits
    else:
        return qimage, n_locs


def _render_channels(
    locs: list[pd.DataFrame],
    info: list[list[dict]],
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ),
    min_blur_width: float,
    max_blur_width: float | None = None,
    ang: tuple | Rotation | None,
    indices: list | None = None,
    global_precision: list | None = None,
    quadtree_capacity: int | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
    render_index: list | None = None,
) -> list[tuple[int, lib.FloatArray2D]]:
    """Render each channel's raw grayscale image through the selected
    splat backend.

    Column arrays are extracted once per channel (no pandas below this
    point) and handed to the ``backend.SplatBackend`` implementation
    chosen by ``backend._get_backend()`` — the CPU chunk pool today, a
    GPU backend when one is selected. If a non-CPU backend fails with
    ``SplatBackendError``, the request is re-rendered on the CPU
    backend after one logged warning; rendering never crashes on a
    backend problem.

    Parameters
    ----------
    locs : list of pd.DataFrame
        Localizations, one DataFrame per channel.
    info : list of list of dict
        Metadata, one entry per channel.
    disp_px_size : float
        Display pixel size in nm, see ``render``.
    viewport : tuple or None
        Field of view ``((y_min, x_min), (y_max, x_max))`` in camera
        pixels, see ``render``.
    blur_method : {"gaussian", "gaussian_iso", "smooth", "convolve"} or None
        Blur method, see ``render``.
    min_blur_width : float
        Minimum size of blur (camera pixels), see ``render``.
    max_blur_width : float, optional
        Maximum localization precision rendered by the Gaussian
        methods (camera pixels), see ``render``. Default is None.
    ang : tuple or scipy.spatial.transform.Rotation or None
        Rotation of the localizations, see ``render``.
    indices : list of lib.IntArray1D or None, optional
        Per channel, the rows to render (see ``render``); None entries
        (or None) render every row.
    global_precision : list of tuple or None, optional
        Per channel, the 'convolve' blur (see ``render``); None entries
        (or None) use the median precision of the rows rendered.
    quadtree_capacity : int or None, optional
        Leaf capacity of the 'quadtree' method (see ``render``).
    triangulation_passes : int or None, optional
        Passes of the 'triangulation' method (see ``render``).
    triangulation_jitter : float or None, optional
        Jitter width of the 'triangulation' method (see ``render``).
    render_index : list of spatial_index.RenderIndexPyramid or None, \
            optional
        Per channel, its spatial index for the 'quadtree' method (see
        ``render``); None entries (or None) build one on the fly. The
        method renders on the CPU only.

    Returns
    -------
    renderings : list of (int, lib.FloatArray2D)
        ``render``'s ``(n, image)`` result per channel, in input order.
    """
    if indices is None:
        indices = [None] * len(locs)
    if global_precision is None:
        global_precision = [None] * len(locs)
    if render_index is None:
        render_index = [None] * len(locs)
    columns = [
        _extract_render_columns(
            channel,
            blur_method,
            ang,
            max_blur_width,
            channel_indices,
            channel_precision,
            channel_index,
        )
        for channel, channel_indices, channel_precision, channel_index in zip(
            locs, indices, global_precision, render_index
        )
    ]
    kwargs = dict(
        disp_px_size=disp_px_size,
        viewport=viewport,
        blur_method=blur_method,
        min_blur_width=min_blur_width,
        ang=ang,
        quadtree_capacity=quadtree_capacity,
        triangulation_passes=triangulation_passes,
        triangulation_jitter=triangulation_jitter,
    )
    if blur_method in ("quadtree", "triangulation"):
        # tree descent over the sorted index / Qhull passes: CPU methods
        return _cpu_backend().render_channels(columns, info, **kwargs)
    n_locs = sum(len(c) for c in columns)
    if ang is not None:
        # a rotated 3D localization costs the CPU far more than a 2D
        # one (see lib.RENDER_ROTATED_COST_FACTOR), so the small-render
        # cutoff of the selection is reached that much sooner
        n_locs *= lib.RENDER_ROTATED_COST_FACTOR
    chosen = _get_backend(n_locs=n_locs)
    cpu = _cpu_backend()
    if chosen is not cpu:
        try:
            renderings = chosen.render_channels(columns, info, **kwargs)
        except SplatBackendError as error:
            _log.warning(
                "splat backend '%s' failed; re-rendering on the CPU",
                chosen.name,
                exc_info=True,
            )
            note_fallback(str(error))
        else:
            note_fallback(None)
            return renderings
    return cpu.render_channels(columns, info, **kwargs)


def _contrast_limits(
    image: lib.FloatArray2D | lib.FloatArray3D,
    vmin: float | None,
    vmax: float | None,
    autoscale: bool,
) -> tuple[float, float]:
    """Contrast limits exactly as ``scale_contrast`` derives them,
    without scaling the image."""
    if autoscale:
        if image.ndim == 2:
            max_ = image.max()
        else:
            # lowest max value from all channels, given it's not
            # an empty image
            max_ = min([_.max() for _ in image if _.max() > 0])
        vmax = 0.5 * max_
        vmin = 0.0
    vmin = vmin if vmin is not None else image.min()
    vmax = vmax if vmax is not None else image.max()
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def _resolve_cmap(colormap: str | lib.FloatArray2D) -> lib.IntArray2D:
    """The 256-entry uint8 RGB table that ``apply_colormap`` would use
    for ``colormap`` (built identically, alpha dropped)."""
    if isinstance(colormap, str):
        cmap = np.uint8(np.round(255 * plt.get_cmap(colormap)(np.arange(256))))
    else:
        cmap = np.uint8(np.round(255 * colormap))
    return np.ascontiguousarray(cmap[:, :3])


def _render_multi_channel(
    locs: list[pd.DataFrame],
    info: list[list[dict]],
    *,
    disp_px_size: float,
    colors: list[tuple[int, int, int]] | list[np.ndarray],
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    max_blur_width: float | None = None,
    ang: tuple | Rotation | None = None,
    indices: list | None = None,
    global_precision: list | None = None,
    quadtree_capacity: int | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
    render_index: list | None = None,
    contrast: tuple[float, float] | None = None,
    relative_intensities: list[float] | None = None,
    invert_colors: bool = False,
    background_color: tuple[float, float, float] | None = None,
    raw_image_cache: lib.FloatArray3D | None = None,
) -> tuple[int, lib.IntArray3D, tuple[float, float], lib.FloatArray3D]:
    """Render multi-channel localizations into an RGB 8bit image
    (numpy array). See ``render_scene`` for more details.

    ``colors`` may be either a list of ``(r, g, b)`` triplets (legacy
    behavior: each channel rendered as ``intensity × rgb``, additive
    blend) or a list of ``(256, 3)`` LUTs (each channel indexed into
    its LUT before additive blending — supports per-channel
    matplotlib colormaps and user-defined colormaps from the GUI)."""
    if raw_image_cache is not None:
        assert raw_image_cache.ndim == 3, "raw_image_cache must be a 3D array."
        raw_image = raw_image_cache
        n_locs = 0
    else:
        # monochromatic images of localizations, rendered in parallel
        # across channels within the user's render CPU budget
        renderings = _render_channels(
            locs,
            info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            max_blur_width=max_blur_width,
            ang=ang,
            indices=indices,
            global_precision=global_precision,
            quadtree_capacity=quadtree_capacity,
            triangulation_passes=triangulation_passes,
            triangulation_jitter=triangulation_jitter,
            render_index=render_index,
        )
        n_locs = sum([rendering[0] for rendering in renderings])
        raw_image = np.array([rendering[1] for rendering in renderings])

    vmin, vmax = contrast if contrast is not None else (None, None)
    autoscale = True if contrast is None else False
    if colors is None:  # fallback if the user did not specify colors
        colors = lib.get_colors(raw_image.shape[0])
    colors_arr = np.asarray(colors, dtype=np.float32)

    if colors_arr.ndim == 3:
        # LUT path (the GUI's path): one fused kernel does contrast ->
        # intensities -> LUT gather -> blend -> background -> 8 bit in
        # two passes over the stack instead of the ~10-pass numpy chain
        n_channels = raw_image.shape[0]
        contrast_limits = _contrast_limits(raw_image, vmin, vmax, autoscale)
        if relative_intensities is None:
            rel = np.ones(n_channels, dtype=np.float32)
        else:
            assert len(relative_intensities) == n_channels, (
                "Length of relative_intensities must match number of "
                "channels in images."
            )
            rel = np.asarray(relative_intensities, dtype=np.float32)
        has_bg = background_color is not None and any(
            c > 0 for c in background_color
        )
        bg = np.asarray(
            background_color if has_bg else (0.0, 0.0, 0.0), dtype=np.float32
        )
        rgb32, max_value = _compose_multi_lut(
            np.ascontiguousarray(raw_image, dtype=np.float32),
            np.ascontiguousarray(colors_arr),
            contrast_limits[0],
            contrast_limits[1],
            rel,
            bg,
            has_bg,
        )
        rgb = _quantize_rgb(rgb32, max_value)
    else:
        # legacy solid-color path: each channel is a single (r, g, b),
        # rendered as intensity x rgb through the numpy reference chain
        # (kept as-is for public-API compatibility)
        images, contrast_limits = scale_contrast(
            raw_image,
            vmin,
            vmax,
            autoscale=autoscale,
            return_contrast_limits=True,
        )
        images = scale_intensities(
            images, relative_intensities=relative_intensities
        )
        images_f32 = np.ascontiguousarray(images, dtype=np.float32)
        rgb = np.tensordot(images_f32, colors_arr, axes=([0], [0]))
        # clip to max value of 1 (preserves relative brightness)
        np.minimum(rgb, 1.0, out=rgb)
        # composite over a background color (default black = no change).
        # The background shows through where there are few/no
        # localizations, while bright regions keep their true channel
        # colors. Alpha is the total per-pixel coverage summed across
        # channels.
        if background_color is not None and any(
            c > 0 for c in background_color
        ):
            bg = np.asarray(background_color, dtype=np.float32)
            alpha = np.clip(images_f32.sum(axis=0), 0.0, 1.0)[..., None]
            rgb = rgb + bg * (1.0 - alpha)
            np.minimum(rgb, 1.0, out=rgb)
        rgb = to_8bit(rgb)
    if invert_colors:
        rgb = 255 - rgb
    return n_locs, rgb, contrast_limits, raw_image


def _render_single_channel(
    locs: pd.DataFrame,
    info: list[dict],
    *,
    disp_px_size: float,
    viewport: tuple[tuple[float, float], tuple[float, float]] | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    max_blur_width: float | None = None,
    ang: tuple | Rotation | None = None,
    indices: list | None = None,
    global_precision: tuple[float, float] | list | None = None,
    quadtree_capacity: int | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
    render_index: object | list | None = None,
    contrast: tuple[float, float] | None = None,
    invert_colors: bool = False,
    single_channel_colormap: str = "magma",
    raw_image_cache: lib.FloatArray2D | None = None,
) -> tuple[int, lib.IntArray3D, tuple[float, float], lib.FloatArray2D]:
    """Render single-channel localizations into an RGB 8bit image (numpy
    array). See ``render_scene`` for more details."""
    if raw_image_cache is not None:
        assert raw_image_cache.ndim == 2, "raw_image_cache must be a 2D array."
        raw_image = raw_image_cache
        n_locs = 0
    else:
        # route through the channel/chunk scheduler so a large single
        # channel also renders in parallel within the CPU budget
        if indices is not None and not isinstance(indices, (list, tuple)):
            indices = [indices]  # a single channel's own index array
        if global_precision is not None and not (
            isinstance(global_precision, list)
            and len(global_precision) == 1
            and not np.isscalar(global_precision[0])
        ):
            global_precision = [global_precision]  # one channel's pair
        if render_index is not None and not isinstance(
            render_index, (list, tuple)
        ):
            render_index = [render_index]  # a single channel's pyramid
        ((n_locs, raw_image),) = _render_channels(
            [locs],
            [info],
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            max_blur_width=max_blur_width,
            ang=ang,
            indices=indices,
            global_precision=global_precision,
            quadtree_capacity=quadtree_capacity,
            triangulation_passes=triangulation_passes,
            triangulation_jitter=triangulation_jitter,
            render_index=render_index,
        )
    vmin, vmax = contrast if contrast is not None else (None, None)
    autoscale = True if contrast is None else False
    contrast_limits = _contrast_limits(raw_image, vmin, vmax, autoscale)
    # fused kernel replacing scale_contrast -> to_8bit -> apply_colormap
    rgb = _compose_single(
        np.ascontiguousarray(raw_image, dtype=np.float32),
        _resolve_cmap(single_channel_colormap),
        contrast_limits[0],
        contrast_limits[1],
    )
    if invert_colors:
        rgb = 255 - rgb
    return n_locs, rgb, contrast_limits, raw_image


def color_range(
    n_channels: int | None,
    colors: list | None = None,
    relative_intensities: list[float] | None = None,
    invert_colors: bool = False,
    background_color: tuple[float, float, float] | None = None,
    single_channel_colormap: str | lib.FloatArray2D = "magma",
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Find the background and maximum-contrast colors of a rendering.

    These are the colors that ``render_scene`` gives pixels without
    localizations and pixels where every channel is at the maximum
    contrast, e.g., to tell localizations apart from the background.

    The multi-channel path renormalizes each image by its maximum; the
    colors are those of an image containing both kinds of pixels, as a
    typical view does.

    Parameters
    ----------
    n_channels : int or None
        Number of rendered channels; None for single-channel rendering
        (``locs`` passed as a DataFrame).
    colors : list of tuples or list of lib.FloatArray2D, optional
        Colors of the channels, as in ``render_scene``. Default is
        None.
    relative_intensities : list of float, optional
        Relative intensity of each channel, as in ``render_scene``.
        Default is None.
    invert_colors : bool, optional
        If True, colors are inverted, as in ``render_scene``. Default
        is False.
    background_color : tuple of float, optional
        ``(r, g, b)`` background color of multi-channel data, as in
        ``render_scene``. Default is None.
    single_channel_colormap : str or lib.FloatArray2D, optional
        Colormap of single-channel data, as in ``render_scene``.
        Default is 'magma'.

    Returns
    -------
    empty : IntArray1D
        RGB color (uint8) of shape ``(3,)`` of pixels without
        localizations.
    full : IntArray1D
        RGB color (uint8) of shape ``(3,)`` of pixels at the maximum
        contrast.
    """
    if n_channels is None:
        raw = np.array([[0.0, 1.0]], dtype=np.float32)
        _, rgb, _, _ = _render_single_channel(
            None,
            None,
            disp_px_size=1.0,
            contrast=(0.0, 1.0),
            invert_colors=invert_colors,
            single_channel_colormap=single_channel_colormap,
            raw_image_cache=raw,
        )
    elif n_channels == 0:  # render_scene draws an empty black image
        black = np.zeros(3, dtype=np.uint8)
        return black, black.copy()
    else:
        raw = np.zeros((n_channels, 1, 2), dtype=np.float32)
        raw[:, 0, 1] = 1.0
        _, rgb, _, _ = _render_multi_channel(
            None,
            None,
            disp_px_size=1.0,
            colors=colors,
            contrast=(0.0, 1.0),
            relative_intensities=relative_intensities,
            invert_colors=invert_colors,
            background_color=background_color,
            raw_image_cache=raw,
        )
    rgb = np.asarray(rgb, dtype=np.uint8)
    return rgb[0, 0].copy(), rgb[0, 1].copy()


def scale_contrast(
    image: lib.FloatArray2D | lib.FloatArray3D,
    vmin: float | None = None,
    vmax: float | None = None,
    autoscale: bool = False,
    return_contrast_limits: bool = False,
) -> (
    lib.FloatArray2D
    | lib.FloatArray3D
    | tuple[lib.FloatArray2D | lib.FloatArray3D, tuple[float, float]]
):
    """Scale contrast of the image (2D array) or images (3D array)
    according to the given contrast limits or automatically.

    Parameters
    ----------
    image : FloatArray2D or FloatArray3D
        Image (2D array) or images (3D array) to be contrast scaled.
    vmin : float or None, optional
        Minimum contrast limit. If None, the minimum pixel value of the
        image(s) is used. Default is None.
    vmax : float or None, optional
        Maximum contrast limit. If None, the maximum pixel value of the
        image(s) is used. Default is None.
    autoscale : bool, optional
        If True, automatically adjust contrast limits to optimally use
        the full range of pixel values. Default is False.
    return_contrast_limits : bool, optional
        If True, return the contrast limits used for scaling. Default is
        False.

    Returns
    -------
    scaled_images : FloatArray2D or FloatArray3D
        Contrast scaled image(s).
    contrast_limits : tuple of float, optional
        The contrast limits used for scaling. Only returned if
        return_contrast_limits is True.
    """
    vmin, vmax = _contrast_limits(image, vmin, vmax, autoscale)
    scaled_image = (image - vmin) / (vmax - vmin)
    scaled_image[~np.isfinite(scaled_image)] = 0.0
    scaled_image = np.clip(scaled_image, 0.0, 1.0)
    if return_contrast_limits:
        return scaled_image, (vmin, vmax)
    return scaled_image


def scale_intensities(
    images: lib.FloatArray3D,
    relative_intensities: list[float] | None = None,
) -> lib.FloatArray3D:
    """Scale intensities across images.

    Parameters
    ----------
    images : FloatArray3D
        Images to be intensity scaled, one per channel. Scaled in place.
    relative_intensities : list of float, optional
        List of relative intensities for each channel. If None, all
        channels are rendered with the same intensity. Default is None.

    Returns
    -------
    scaled_images : FloatArray3D
        Intensity scaled images.
    """
    if relative_intensities is not None:
        assert len(relative_intensities) == images.shape[0], (
            "Length of relative_intensities must match number of channels "
            "in images."
        )
        for i in range(images.shape[0]):
            images[i] *= relative_intensities[i]
    return images


def to_8bit(
    image: lib.FloatArray2D | lib.FloatArray3D,
) -> lib.IntArray2D | lib.IntArray3D:
    """Convert a float image with values between 0 and 1 to an 8-bit image
    with values between 0 and 255.

    Parameters
    ----------
    image : FloatArray2D or FloatArray3D
        Image(s) with values between 0 and 1. Normalized in place so that the
        maximum is 1 before the conversion.

    Returns
    -------
    image : IntArray2D or IntArray3D
        The image as ``uint8``.
    """
    # normalize to max value of 1 and convert to 8-bit
    image /= image.max() if image.max() > 0 else 1.0
    return np.round(image * 255).astype(np.uint8)


def apply_colormap(
    image: lib.IntArray2D, colormap: str | lib.FloatArray2D
) -> lib.IntArray3D:
    """Apply a colormap to a single-channel image (2D array) and return an
    RGB image (3D array).

    Parameters
    ----------
    image : IntArray2D
        Single-channel image as a 2D numpy array with integer values
        between 0 and 255 (8bit).
    colormap : str or FloatArray2D
        If a str, the corresponding pyplot colormap is selected. If a 2D
        array, a 256x4 or 256x3 array is expected with values between 0
        and 1. Note: the alpha channel (if present) is ignored and the
        colormap is applied as if all values were fully opaque.

    Returns
    -------
    image : IntArray3D
        RGB image of shape ``(height, width, 3)``, ``uint8``.
    """
    if isinstance(colormap, str):
        cmap = np.uint8(np.round(255 * plt.get_cmap(colormap)(np.arange(256))))
    else:
        cmap = np.uint8(np.round(255 * colormap))
    image = cmap[image][:, :, :3]  # drop alpha channel if present
    return image


def split_locs_by_property(
    locs: pd.DataFrame,
    *,
    property_name: str,
    n_colors: int = 32,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[pd.DataFrame]:
    """Split localizations into groups based on a specified property and
    return a list of DataFrames, one for each group.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    property_name : str
        Name of the property to split the localizations by.
    n_colors : int, optional
        Number of color groups to create. Default is 32.
    min_value : float, optional
        Minimum value of the property for scaling. If None, the minimum
        value in the data is used.
    max_value : float, optional
        Maximum value of the property for scaling. If None, the maximum
        value in the data is used.

    Returns
    -------
    locs_groups : list of pd.DataFrame
        Each element corresponds to a group of localizations with
        similar property values.
    """
    assert (
        property_name in locs.columns
    ), f"Property '{property_name}' not found in localizations."
    values = locs[property_name]
    if min_value is None:
        min_value = values.min()
    if max_value is None:
        max_value = values.max()

    step = (max_value - min_value) / n_colors
    color = np.floor((values - min_value) / step).astype(int)
    color = np.clip(color, 0, n_colors - 1)

    locs_groups = []
    for i in range(n_colors):
        locs_groups.append(locs[color == i])
    return locs_groups


def split_locs_by_group(
    locs: pd.DataFrame,
    n_colors: int = N_GROUP_COLORS,
    group_color: lib.IntArray1D | None = None,
) -> list[pd.DataFrame]:
    """Split localizations into groups based on the 'group' column and
    return a list of DataFrames, one for each group.

    If no 'group' column is present, all localizations are returned as
    single-element list.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    n_colors : int, optional
        Number of color groups to create if 'group' column is not present.
        Default is 8.
    group_color : IntArray1D or None, optional
        If provided, specifies the group color ids (up to `n_colors`)
        for each localization.

    Returns
    -------
    locs_groups : list of pd.DataFrame
        One data frame per group; a single-element list when there is no
        grouping to apply.
    """
    if group_color is not None:
        assert len(group_color) == len(
            locs
        ), "Length of group_color must match number of localizations."
        locs_groups = [locs[group_color == _] for _ in range(n_colors)]
    elif "group" in locs.columns:
        groups = locs["group"].unique()
        locs_groups = [locs[locs["group"] == group] for group in groups]
    else:
        locs_groups = [locs]
    return locs_groups
