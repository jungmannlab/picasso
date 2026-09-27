"""
picasso.render.animation
~~~~~~~~~~~~~~~~~~~~~~~~

Build animations (video files) from sequences of rendered scenes.

:authors: Rafal Kowalewski, Joerg Schnitzbauer
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from typing import Literal, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .. import io, lib, __version__
from .geometry import viewport_width
from .scene import render_scene

if TYPE_CHECKING:
    from PyQt6 import QtCore
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtCore = lib._LazyQtModule("PyQt6.QtCore")


def _normalize_animation_positions(
    positions: list,
) -> list[tuple[Rotation, tuple]]:
    """Normalize animation checkpoints to (Rotation, viewport) tuples.

    Each position must be (rotation, viewport) with rotation being a
    scipy Rotation. The legacy format (angle_x, angle_y, angle_z,
    viewport) with Euler angles was removed in v0.12.0; convert such
    angles with ``rotation_matrix``.
    """
    normalized = []
    for p in positions:
        if len(p) == 2 and isinstance(p[0], Rotation):
            normalized.append((p[0], p[1]))
        elif len(p) == 4:
            raise ValueError(
                "Animation positions given as Euler angles (angle_x, "
                "angle_y, angle_z, viewport) are no longer supported since "
                "v0.12.0. Pass (rotation, viewport) with rotation being a "
                "scipy.spatial.transform.Rotation instead, e.g. "
                "picasso.render.rotation_matrix(angle_x, angle_y, angle_z)."
            )
        else:
            raise ValueError(
                "Each position must be a tuple (rotation, viewport) with "
                "rotation being a scipy Rotation."
            )
    return normalized


def _animation_sequence(
    positions: list[tuple[Rotation, tuple]],
    durations: list[float],
    fps: int,
    segment_rotations: list | None = None,
) -> tuple[list[Rotation], list]:
    """Calculate the sequence of rotations and viewports for the
    animation. See ``build_animation`` for more details.

    Each segment is interpolated along the geodesic between the two
    checkpoint rotations at constant angular velocity (slerp). If
    ``segment_rotations`` is given, the corresponding rotation vector
    defines the rotation path of each segment and may include full
    turns (magnitude beyond pi), e.g. 4 pi for two full turns.

    Such a path is followed as
    ``from_rotvec(t * correction) * from_rotvec(t * rotvec) * R1``,
    where ``correction`` is the (shortest) residual rotation left
    between the end of the given path and the next checkpoint. The
    segment therefore always ends exactly at the next checkpoint while
    keeping the requested turns, without having to re-derive the turns
    from the checkpoint orientations - which is impossible, as a full
    turn leaves the orientation unchanged. When the given path already
    matches the relative rotation (e.g. both around the same axis),
    the correction vanishes and this is plain slerp, split into
    sub-180-degree pieces.

    Frames are placed at ``t = 0, 1/n, ...`` for every segment but the
    last, whose final frame lands on the last checkpoint. This way each
    checkpoint is rendered once, rather than once as the end of one
    segment and again as the start of the next."""
    rotations = []
    viewports = []
    for i in range(len(positions) - 1):
        n_frames = max(1, int(fps * durations[i]))
        # only the last segment includes its final checkpoint; the
        # others end where the next segment starts
        endpoint = i == len(positions) - 2

        # rotations
        R1, vp1 = positions[i]
        R2, vp2 = positions[i + 1]
        relative = R2 * R1.inv()
        if segment_rotations is not None:
            rotvec = np.asarray(segment_rotations[i], dtype=float)
        else:
            rotvec = relative.as_rotvec()
        # residual rotation between the end of the requested path and
        # the next checkpoint, spread evenly over the segment
        correction = (
            relative * Rotation.from_rotvec(rotvec).inv()
        ).as_rotvec()
        fractions = np.linspace(0, 1, n_frames, endpoint=endpoint)
        rotations.extend(
            Rotation.from_rotvec(fraction * correction)
            * Rotation.from_rotvec(fraction * rotvec)
            * R1
            for fraction in fractions
        )

        # viewports
        ymin = np.linspace(vp1[0][0], vp2[0][0], n_frames, endpoint=endpoint)
        xmin = np.linspace(vp1[0][1], vp2[0][1], n_frames, endpoint=endpoint)
        ymax = np.linspace(vp1[1][0], vp2[1][0], n_frames, endpoint=endpoint)
        xmax = np.linspace(vp1[1][1], vp2[1][1], n_frames, endpoint=endpoint)
        current_viewports = [
            ((ymin[j], xmin[j]), (ymax[j], xmax[j])) for j in range(len(ymin))
        ]
        viewports.extend(current_viewports)
    return rotations, viewports


def build_animation(
    path: str,
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    *,
    positions: (
        list[tuple[Rotation, tuple]]
        | list[tuple[tuple[float, float, float], tuple]]
    ),
    durations: list[float],
    disp_px_size: int | float,  # nm
    image_size: tuple[int, int],
    segment_rotations: list | None = None,
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ) = None,
    min_blur_width: float = 0.0,
    quadtree_capacity: int | None = None,
    triangulation_passes: int | None = None,
    triangulation_jitter: float | None = None,
    contrast: tuple[float, float] | None = None,
    invert_colors: bool = False,
    single_channel_colormap: str | lib.FloatArray2D = "magma",
    colors: list | None = None,
    relative_intensities: list[float] | None = None,
    fps: int = 30,
    adjust_pixel_size: bool = True,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    cancel: Callable[[], bool] | None = None,
) -> bool:
    """Build an animation of rendered localizations given the
    checkpoints (rotation, viewport, etc) and the time between them.

    Parameters
    ----------
    path : str
        Path to the animation file to be created. Must end with .mp4.
    locs : pd.DataFrame or list of pd.DataFrame
        Localizations to be rendered. Can be either one localization
        file or a list thereof.
    info : list of dict or list of list of dict
        List of info dictionaries corresponding to the localization
        file(s).
    positions : list
        Each element determines a checkpoint of the animation, which
        is a tuple of 2 elements: (rotation, viewport). Rotation is a
        ``scipy.spatial.transform.Rotation`` defining the orientation
        of the localizations at the checkpoint. Viewport is given as
        ((y_min, x_min), (y_max, x_max)) in camera pixels. Since
        v0.12.0, the legacy format (angle_x, angle_y, angle_z, viewport)
        with Euler angles is no longer accepted; convert the angles with
        ``rotation_matrix``.
    durations : list
        List of durations in seconds between the checkpoints. Must have
        the same length as positions - 1.
    disp_px_size : int or float
        Display pixel size in nm. If 'adjust_pixel_size' is True,
        disp_px_size defines the pixel size in the last frame of the
        animation and will be adjusted if the viewport is zoomed in or
        out such that the number of display pixels remains the same.
        If 'adjust_pixel_size' is False, disp_px_size remains the same
        across the animation.
    image_size : tuple of int
        Size of the rendered image in pixels, given as (width, height).
    segment_rotations : list, optional
        One rotation vector (3 floats, radians, scipy convention) per
        segment, i.e., of length len(positions) - 1, describing the
        full rotation path from one checkpoint to the next. The
        magnitude may exceed pi to encode rotations larger than 180
        degrees, e.g. (0, 0, 4 * pi) for two full turns around the z
        axis. Any rotation left between the end of such a path and the
        next checkpoint is spread over the segment, so the segment
        always ends exactly at the next checkpoint. If None, each
        segment follows the shortest path (slerp). Default is None.
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
    quadtree_capacity : int, optional
        Leaf capacity of the 'quadtree' blur method, see ``render``.
        Default is None (``lib.RENDER_QUADTREE_CAPACITY_DEFAULT``).
    triangulation_passes : int, optional
        Number of passes of the 'triangulation' blur method, see
        ``render``. Default is None (the method's default).
    triangulation_jitter : float, optional
        Jitter width of the 'triangulation' blur method, see
        ``render``. Default is None (the method's default).
    contrast : tuple of float, optional
        Contrast limits for scaling. If None, contrast is automatically
        determined. If given, only the last checkpoint is used to
        determine the contrast limits and the limits will be adjusted
        if the viewport is zoomed in or out.
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
    fps : int, optional
        Frames per second of the animation. Default is 30.
    adjust_pixel_size : bool, optional
        If True, adjust disp_px_size on the go such that the number of
        display pixels remains the same if the viewport is zoomed in or
        out. If False, disp_px_size remains the same across the
        animation.
    progress_callback : callable, "console", or None, optional
        If a callable, it is called with the current frame number as an
        argument after each frame is rendered. If "console", a progress
        bar is printed to the console. If None, no progress is reported.
        Default is None.
    cancel : callable, optional
        Polled before each frame; when it returns True the build stops,
        the incomplete video file is removed and no settings file is
        written. Default is None (never canceled).

    Returns
    -------
    completed : bool
        True when the animation and its settings file were written,
        False when the build was canceled.
    """
    assert isinstance(path, str) and path.endswith(
        ".mp4"
    ), "path must be a string ending with '.mp4'."
    assert isinstance(
        locs, (pd.DataFrame, list)
    ), "locs must be a pd.DataFrame or a list of pd.DataFrames."
    if isinstance(locs, list):
        assert all(
            isinstance(locs_, pd.DataFrame) for locs_ in locs
        ), "All elements of locs must be pd.DataFrames."
        assert len(locs) >= 1, "locs must contain at least one DataFrame."
    assert (
        isinstance(info, list) and len(info) >= 1
    ), "info must be a non-empty list."
    assert (
        isinstance(positions, list) and len(positions) >= 2
    ), "positions must be a list with at least 2 elements."
    positions = _normalize_animation_positions(positions)
    if segment_rotations is not None:
        assert (
            isinstance(segment_rotations, list)
            and len(segment_rotations) == len(positions) - 1
        ), "segment_rotations must be a list of length len(positions) - 1."
        assert all(
            np.asarray(rotvec, dtype=float).shape == (3,)
            for rotvec in segment_rotations
        ), "Each segment rotation must be a rotation vector of 3 floats."
    assert (
        isinstance(durations, list) and len(durations) == len(positions) - 1
    ), "durations must be a list of length len(positions) - 1."
    assert all(d > 0 for d in durations), "All durations must be positive."
    assert (
        isinstance(disp_px_size, (int, float)) and disp_px_size > 0
    ), "disp_px_size must be a positive number."
    assert (
        isinstance(image_size, (tuple, list))
        and len(image_size) == 2
        and all(isinstance(s, int) and s > 0 for s in image_size)
    ), "image_size must be a tuple of two positive integers (width, height)."
    assert blur_method in (
        "gaussian",
        "gaussian_iso",
        "smooth",
        "convolve",
        "quadtree",
        "triangulation",
        None,
    ), (
        "blur_method must be one of 'gaussian', 'gaussian_iso', 'smooth', "
        "'convolve', 'quadtree', 'triangulation' or None."
    )
    assert (
        isinstance(min_blur_width, (int, float)) and min_blur_width >= 0
    ), "min_blur_width must be a non-negative number."
    if contrast is not None:
        assert (
            isinstance(contrast, (tuple, list))
            and len(contrast) == 2
            and contrast[0] < contrast[1]
        ), "contrast must be a tuple (vmin, vmax) with vmin < vmax."
    assert isinstance(invert_colors, bool), "invert_colors must be a bool."
    if not isinstance(single_channel_colormap, str):
        assert (
            hasattr(single_channel_colormap, "shape")
            and single_channel_colormap.ndim == 2
            and single_channel_colormap.shape[0] == 256
            and single_channel_colormap.shape[1] in (3, 4)
        ), (
            "single_channel_colormap must be a str or a 256x3 / 256x4 "
            "float array with values between 0 and 1."
        )
    if colors is not None:
        n_channels = len(locs) if isinstance(locs, list) else 1
        assert (
            len(colors) == n_channels
        ), "colors must have one entry per channel."
    if relative_intensities is not None:
        n_channels = len(locs) if isinstance(locs, list) else 1
        assert (
            len(relative_intensities) == n_channels
        ), "relative_intensities must have one entry per channel."
        assert all(
            v >= 0 for v in relative_intensities
        ), "All relative_intensities must be non-negative."
    assert isinstance(fps, int) and fps > 0, "fps must be a positive integer."
    assert isinstance(
        adjust_pixel_size, bool
    ), "adjust_pixel_size must be a bool."
    assert (
        progress_callback is None
        or progress_callback == "console"
        or callable(progress_callback)
    ), "progress_callback must be None, 'console', or a callable."
    assert cancel is None or callable(cancel), "cancel must be a callable."

    return _build_animation(
        path=path,
        locs=locs,
        info=info,
        positions=positions,
        durations=durations,
        segment_rotations=segment_rotations,
        disp_px_size=disp_px_size,
        image_size=image_size,
        blur_method=blur_method,
        min_blur_width=min_blur_width,
        quadtree_capacity=quadtree_capacity,
        triangulation_passes=triangulation_passes,
        triangulation_jitter=triangulation_jitter,
        contrast=contrast,
        invert_colors=invert_colors,
        single_channel_colormap=single_channel_colormap,
        colors=colors,
        relative_intensities=relative_intensities,
        fps=fps,
        adjust_pixel_size=adjust_pixel_size,
        progress_callback=progress_callback,
        cancel=cancel,
    )


def _build_animation(
    path: str,
    locs: pd.DataFrame | list[pd.DataFrame],
    info: list[dict] | list[list[dict]],
    positions: list[tuple[Rotation, tuple]],
    durations: list[float],
    segment_rotations: list | None,
    disp_px_size: int | float,
    image_size: tuple[int, int],
    blur_method: (
        Literal["gaussian", "gaussian_iso", "smooth", "convolve"] | None
    ),
    min_blur_width: float,
    quadtree_capacity: int | None,
    triangulation_passes: int | None,
    triangulation_jitter: float | None,
    contrast: tuple[float, float] | None,
    invert_colors: bool,
    single_channel_colormap: str | lib.FloatArray2D,
    colors: list | None,
    relative_intensities: list[float] | None,
    fps: int,
    adjust_pixel_size: bool,
    progress_callback: Callable[[int], None] | Literal["console"] | None,
    cancel: Callable[[], bool] | None = None,
) -> bool:
    """Internal function to build an animation of rendered localizations
    given the checkpoints. See ``build_animation`` for more details."""
    rotations, viewports = _animation_sequence(
        positions, durations, fps, segment_rotations=segment_rotations
    )

    # width and height for building the animation; must be divisible by 16
    # as ffmpeg codecs require this for proper encoding
    width, height = image_size
    width = ((width + 15) // 16) * 16
    height = ((height + 15) // 16) * 16

    # render all frames and save in RAM
    video_writer = imageio.get_writer(path, fps=fps)
    use_tqdm = progress_callback == "console"
    if use_tqdm:
        iter_range = tqdm(
            range(len(rotations)), desc="Building animation", unit="frame"
        )
    else:
        iter_range = range(len(rotations))

    for i in iter_range:
        if cancel is not None and cancel():
            # leave nothing half-written behind
            video_writer.close()
            try:
                os.remove(path)
            except OSError:
                pass
            return False
        if callable(progress_callback):
            progress_callback(i)

        disp_px_size_ = (
            _adjust_disp_px_size(disp_px_size, viewports[-1], viewports[i])
            if adjust_pixel_size
            else disp_px_size
        )
        contrast_ = _adjust_contrast(contrast, viewports[-1], viewports[i])
        qimage = render_scene(
            locs=locs,
            info=info,
            disp_px_size=disp_px_size_,
            viewport=viewports[i],
            ang=rotations[i],
            blur_method=blur_method,
            min_blur_width=min_blur_width,
            quadtree_capacity=quadtree_capacity,
            triangulation_passes=triangulation_passes,
            triangulation_jitter=triangulation_jitter,
            contrast=contrast_,
            invert_colors=invert_colors,
            single_channel_colormap=single_channel_colormap,
            colors=colors,
            relative_intensities=relative_intensities,
        )[0]
        qimage = qimage.scaled(
            width,
            height,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
        )

        # convert to a np.array and append
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        frame = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        frame = frame[:, :, :3]
        frame = frame[:, :, ::-1]  # invert RGB to BGR
        video_writer.append_data(frame)

    if callable(progress_callback):
        progress_callback(len(rotations))
    video_writer.close()

    # save a yaml with animation settings, note that yaml does not support
    # numpy types and arrays
    quaternions_yaml = [[float(q) for q in R.as_quat()] for R, _ in positions]
    viewports_yaml = [
        (
            (float(vp[0][0]), float(vp[0][1])),
            (float(vp[1][0]), float(vp[1][1])),
        )
        for _, vp in positions
    ]
    if segment_rotations is None:
        segment_rotations = [
            (positions[i + 1][0] * positions[i][0].inv()).as_rotvec()
            for i in range(len(positions) - 1)
        ]
    segments_yaml = [
        [float(np.degrees(v)) for v in rotvec] for rotvec in segment_rotations
    ]
    anim_settings = {
        "Generated by": f"Picasso v{__version__} Render 3D Animation",
        "FPS": fps,
        "Quaternions at checkpoints (x, y, z, w)": quaternions_yaml,
        "Rotations between checkpoints (x, y, z) (deg)": segments_yaml,
        "Viewports at checkpoints (camera pixels)": viewports_yaml,
        "Durations (s)": durations,
    }
    info_path = os.path.splitext(path)[0] + ".yaml"
    io.save_info(info_path, [anim_settings])
    return True


def _adjust_disp_px_size(
    disp_px_size_ref: float,
    viewport_ref: tuple[tuple[float, float], tuple[float, float]],
    new_viewport: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Adjust display pixel size based on the change in viewport to keep
    the number of display pixels the same."""
    ref_width = viewport_width(viewport_ref)
    new_width = viewport_width(new_viewport)
    # below could be ref_height / new_height, should be the same since
    # we assume the shape of the viewport stays the same
    zoom_factor = ref_width / new_width
    return disp_px_size_ref / zoom_factor


def _adjust_contrast(
    contrast_ref: tuple[float, float] | None,
    viewport_ref: tuple[tuple[float, float], tuple[float, float]],
    new_viewport: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """Adjust contrast limits based on the change in viewport to keep the
    same contrast across zoom levels."""
    if contrast_ref is None:
        return None
    ref_width = viewport_width(viewport_ref)
    new_width = viewport_width(new_viewport)
    zoom_factor = ref_width / new_width
    vmin_ref, vmax_ref = contrast_ref
    vmin_new = vmin_ref / zoom_factor**2
    vmax_new = vmax_ref / zoom_factor**2
    return vmin_new, vmax_new
