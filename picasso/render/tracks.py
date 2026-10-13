"""
picasso.render.tracks
~~~~~~~~~~~~~~~~~~~~~

Tracks of single-particle tracking (see ``picasso.spt``) drawn onto
rendered localizations: every track as a line through its
localizations, in the order of their frames.

The segments between consecutive localizations of every track
(``picasso.spt.track_segments``) are culled to the field of view, and
when more tracks are in view than are to be drawn, a fixed subset of
the tracks (every k-th track id) is drawn, so that the subset does not
change while panning. The lines are drawn opaque onto a transparent
layer, which is then painted with the style's opacity, so overlapping
lines do not add up.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .. import lib
from .geometry import viewport_height, viewport_width
from .image_overlay import _as_rgb32
from .overlays_qt import OverlayStyle

if TYPE_CHECKING:
    from PyQt6 import QtCore, QtGui
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")
    QtCore = lib._LazyQtModule("PyQt6.QtCore")

#: Default maximum number of tracks drawn in one view.
MAX_TRACKS_DEFAULT = 10_000


def select_track_segments(
    segments: lib.FloatArray2D,
    track_id: lib.IntArray1D,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    max_tracks: int | None = None,
) -> tuple[lib.BoolArray1D, int, int]:
    """Select the track segments to draw in a field of view.

    Parameters
    ----------
    segments : lib.FloatArray2D
        ``(n_segments, 4)`` array of ``x0, y0, x1, y1`` (camera pixels),
        see ``picasso.spt.track_segments``.
    track_id : lib.IntArray1D
        Track id of every segment, sorted (as returned by
        ``picasso.spt.track_segments``).
    viewport : tuple
        Field of view, ``((y_min, x_min), (y_max, x_max))`` in camera
        pixels.
    max_tracks : int, optional
        Maximum number of tracks drawn. If more tracks are in view,
        every k-th track id is drawn, with the smallest k that keeps
        the number of tracks below the maximum. None draws all. Default
        is None.

    Returns
    -------
    mask : lib.BoolArray1D
        Whether each segment is drawn.
    n_visible : int
        Number of tracks in view.
    n_drawn : int
        Number of tracks drawn.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    segments = np.asarray(segments).reshape(-1, 4)
    x0, y0, x1, y1 = segments.T
    mask = (
        (np.maximum(x0, x1) >= x_min)
        & (np.minimum(x0, x1) <= x_max)
        & (np.maximum(y0, y1) >= y_min)
        & (np.minimum(y0, y1) <= y_max)
    )
    n_visible = _n_tracks(track_id[mask])
    n_drawn = n_visible
    if max_tracks is not None and n_visible > max_tracks:
        k = int(np.ceil(n_visible / max(int(max_tracks), 1)))
        mask &= track_id % k == 0
        n_drawn = _n_tracks(track_id[mask])
    return mask, n_visible, n_drawn


def _n_tracks(track_id: lib.IntArray1D) -> int:
    """Number of distinct ids in a sorted array."""
    if len(track_id) == 0:
        return 0
    return int(np.count_nonzero(np.diff(track_id))) + 1


def _point_pairs(points: lib.FloatArray2D) -> QtGui.QPolygonF:
    """Polygon holding the end points of the segments ``points``
    (``(n, 4)`` display pixels), as ``QPainter.drawLines`` takes them;
    filled through its buffer, which is far faster than creating one
    ``QLineF`` per segment."""
    polygon = QtGui.QPolygonF()
    polygon.resize(2 * len(points))
    if len(points):
        buffer = polygon.data()
        buffer.setsize(4 * len(points) * np.dtype(np.float64).itemsize)
        np.frombuffer(buffer, dtype=np.float64)[:] = points.reshape(-1)
    return polygon


def draw_tracks(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    segments: lib.FloatArray2D,
    color_index: lib.IntArray1D | None = None,
    palette: list[tuple[int, int, int]] | None = None,
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw track segments onto rendered localizations.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations; drawn on in place if
        it is in the RGB32 or ARGB32 format.
    viewport : tuple
        Field of view shown in ``image``, ``((y_min, x_min), (y_max,
        x_max))`` in camera pixels.
    segments : lib.FloatArray2D
        ``(n_segments, 4)`` array of ``x0, y0, x1, y1`` (camera pixels),
        e.g., those selected by ``select_track_segments``.
    color_index : lib.IntArray1D, optional
        Index into ``palette`` of every segment. Default is None (all
        segments in the first color of ``palette``).
    palette : list of tuples, optional
        RGB colors (0-255) of the segments, used if the style has no
        color. Default is None (white).
    style : OverlayStyle, optional
        Line style, width and opacity of the lines, and their color,
        which overrides ``palette`` if set. Default is 1 pixel wide,
        opaque, solid lines.

    Returns
    -------
    image : QImage
        Image with the drawn tracks.
    """
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
    if len(segments) == 0:
        return image
    width = viewport_width(viewport)
    height = viewport_height(viewport)
    if width <= 0 or height <= 0:
        return image
    style = OverlayStyle() if style is None else style
    (y_min, x_min), _ = viewport
    points = np.empty_like(segments)
    scale_x = image.width() / width  # display pixels per camera pixel
    scale_y = image.height() / height
    points[:, 0::2] = (segments[:, 0::2] - x_min) * scale_x
    points[:, 1::2] = (segments[:, 1::2] - y_min) * scale_y

    if style.color is not None or not palette:
        color = style.color if style.color is not None else "white"
        batches = [(QtGui.QColor(color), points)]
    else:
        if color_index is None:
            color_index = np.zeros(len(points), dtype=np.int64)
        color_index = np.asarray(color_index)
        batches = [
            (
                QtGui.QColor(*(int(c) for c in palette[i])),
                points[color_index == i],
            )
            for i in np.unique(color_index)
        ]

    layer = QtGui.QImage(
        image.width(),
        image.height(),
        QtGui.QImage.Format.Format_ARGB32_Premultiplied,
    )
    layer.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(layer)
    # thin lines stay sharp, as the other overlays (``OverlayStyle``)
    painter.setRenderHint(
        QtGui.QPainter.RenderHint.Antialiasing, style.line_width > 1
    )
    for color, batch in batches:
        painter.setPen(replace(style, color=color, opacity=1.0).pen())
        painter.drawLines(_point_pairs(batch))
    painter.end()

    image = _as_rgb32(image)
    painter = QtGui.QPainter(image)
    painter.setOpacity(style.opacity)
    painter.drawImage(0, 0, layer)
    painter.end()
    return image
