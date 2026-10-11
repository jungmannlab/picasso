"""
picasso.render.grid
~~~~~~~~~~~~~~~~~~~

Reference grid drawn onto rendered localizations: a flat grid fixed to
the data in 2D and grid planes that rotate with the data in 3D.

Grid lines sit at whole multiples of the spacing in the localizations'
coordinates, so a line at ``x = 1000 nm`` stays at that position however
the view is panned or zoomed. In 3D the planes span the bounding box of
the data, enlarged to the next grid lines, and are projected exactly
like the localizations (``picasso.render.locs_rotation``): rotated about
the viewport's center, orthographically.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .. import lib
from .geometry import to_rotation, viewport_height, viewport_width
from .image_overlay import _as_rgb32, _bgra_view, composite_behind
from .overlays_qt import OverlayStyle

if TYPE_CHECKING:
    from PyQt6 import QtGui, QtCore
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")
    QtCore = lib._LazyQtModule("PyQt6.QtCore")


#: How the grid is combined with the localizations, see ``GridOverlay``.
GRID_BLEND_MODES = ("Behind localizations", "Over localizations")
#: Planes of the 3D grid, named by the two axes they contain.
GRID_PLANES = ("xy", "xz", "yz")
#: Where the 3D grid planes are placed, see ``GridOverlay``.
GRID_POSITIONS = ("back", "center")
#: Lines closer together than this (display pixels) are not drawn, they
#: would only tint the image.
MIN_GRID_SPACING_PX = 4
#: Largest number of grid lines along one axis that is drawn.
MAX_GRID_LINES = 2000
# axis normal to each grid plane (0: x, 1: y, 2: z)
_PLANE_NORMALS = {"xy": 2, "xz": 1, "yz": 0}


@dataclass(frozen=True)
class GridOverlay:
    """Reference grid drawn onto rendered localizations.

    Parameters
    ----------
    spacing_nm : float
        Distance between neighboring grid lines in nm.
    style : OverlayStyle, optional
        Color, line style, line width and opacity of the lines. A color
        of None follows the background: white on a dark and black on a
        light one. Default is 40% opaque, solid lines, 1 pixel wide.
    blend : {"Behind localizations", "Over localizations"}, optional
        "Behind localizations" draws the grid as if behind the
        localizations: it shows where there are no localizations and is
        covered the more, the brighter they are, see
        ``picasso.render.composite_behind``. "Over localizations" draws
        it on top. Default is "Behind localizations".
    planes : tuple of str, optional
        3D only: the grid planes drawn, any of ``"xy"``, ``"xz"`` and
        ``"yz"``. Default is all three.
    position : {"back", "center"}, optional
        3D only: "back" places each plane on the side of the bounding
        box facing away from the viewer, so the planes form the back
        walls of a box around the data and change sides as it rotates.
        "center" places each plane through the center of the box.
        Default is "back".

    Raises
    ------
    ValueError
        If ``spacing_nm`` is not positive or ``blend``, ``planes`` or
        ``position`` is unknown.
    """

    spacing_nm: float
    style: OverlayStyle = field(
        default_factory=lambda: OverlayStyle(opacity=0.4)
    )
    blend: Literal["Behind localizations", "Over localizations"] = (
        "Behind localizations"
    )
    planes: tuple[str, ...] = GRID_PLANES
    position: Literal["back", "center"] = "back"

    def __post_init__(self) -> None:
        if not self.spacing_nm > 0:
            raise ValueError("spacing_nm must be positive.")
        if self.blend not in GRID_BLEND_MODES:
            raise ValueError(
                f"Unknown blend mode {self.blend!r}; choose one of "
                f"{GRID_BLEND_MODES}."
            )
        unknown = set(self.planes) - set(GRID_PLANES)
        if unknown:
            raise ValueError(
                f"Unknown grid planes {sorted(unknown)}; choose from "
                f"{GRID_PLANES}."
            )
        if self.position not in GRID_POSITIONS:
            raise ValueError(
                f"Unknown position {self.position!r}; choose one of "
                f"{GRID_POSITIONS}."
            )
        # a list would make the frozen dataclass unhashable
        object.__setattr__(self, "planes", tuple(self.planes))


def grid_levels(
    low: float,
    high: float,
    step: float,
    origin: float = 0.0,
) -> np.ndarray:
    """Positions of the grid lines between ``low`` and ``high``.

    Parameters
    ----------
    low, high : float
        Range covered, both ends included.
    step : float
        Grid spacing, positive.
    origin : float, optional
        A position on the grid; the lines sit at ``origin + k * step``
        for whole numbers ``k``. Default is 0.

    Returns
    -------
    levels : np.ndarray
        Ascending positions of the lines; empty if there are more than
        ``MAX_GRID_LINES``.
    """
    # a little slack keeps lines exactly at the ends of the range
    eps = 1e-9
    k_min = int(np.ceil((low - origin) / step - eps))
    k_max = int(np.floor((high - origin) / step + eps))
    if k_max < k_min or k_max - k_min + 1 > MAX_GRID_LINES:
        return np.zeros(0)
    return origin + step * np.arange(k_min, k_max + 1)


def grid_box(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    step: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Enlarge a bounding box to the next grid lines.

    Parameters
    ----------
    bounds : tuple
        ``((x_min, y_min, z_min), (x_max, y_max, z_max))`` of the data.
    step : float
        Grid spacing in the units of ``bounds``, positive.
    origin : tuple of float, optional
        A point on the grid, see ``grid_levels``. Default is the origin.

    Returns
    -------
    lower, upper : np.ndarray
        Corners of the enlarged box, each of shape ``(3,)``. An axis on
        which the data are flat spans one grid cell.
    """
    low = np.asarray(bounds[0], dtype=float)
    high = np.asarray(bounds[1], dtype=float)
    origin = np.asarray(origin, dtype=float)
    eps = 1e-9
    lower = origin + step * np.floor((low - origin) / step + eps)
    upper = origin + step * np.ceil((high - origin) / step - eps)
    upper = np.maximum(upper, lower + step)
    return lower, upper


def grid_segments_3d(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    step: float,
    ang: tuple[float, float, float] | Rotation,
    planes: tuple[str, ...] = GRID_PLANES,
    position: Literal["back", "center"] = "back",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Line segments of the 3D grid planes, before rotation.

    Parameters
    ----------
    bounds : tuple
        ``((x_min, y_min, z_min), (x_max, y_max, z_max))`` of the data,
        enlarged to the next grid lines by ``grid_box``.
    step : float
        Grid spacing in the units of ``bounds``, positive.
    ang : tuple of float or scipy.spatial.transform.Rotation
        Rotation of the view, see ``picasso.render.to_rotation``. Only
        used to find the back of the box (rotated +z points towards the
        viewer).
    planes : tuple of str, optional
        Planes drawn, see ``GridOverlay``. Default is all three.
    position : {"back", "center"}, optional
        Placement of the planes, see ``GridOverlay``. Default is "back".
    origin : tuple of float, optional
        A point on the grid, see ``grid_levels``. Default is the origin.

    Returns
    -------
    segments : np.ndarray
        Start and end points of the lines, shape ``(n, 2, 3)``.
    """
    lower, upper = grid_box(bounds, step, origin)
    depth_row = to_rotation(ang).as_matrix()[2]  # rotated z of a point
    segments = []
    for plane in planes:
        normal = _PLANE_NORMALS[plane]
        if position == "center":
            offset = (lower[normal] + upper[normal]) / 2
        elif depth_row[normal] > 0:
            # the upper face is nearer the viewer; the lower one is back
            offset = lower[normal]
        else:
            offset = upper[normal]
        a, b = (axis for axis in range(3) if axis != normal)
        levels_a = grid_levels(lower[a], upper[a], step, origin[a])
        levels_b = grid_levels(lower[b], upper[b], step, origin[b])
        # lines along b at every level of a, and vice versa
        for line_axis, cross_axis, levels in (
            (b, a, levels_a),
            (a, b, levels_b),
        ):
            n = len(levels)
            if not n:
                continue
            seg = np.empty((n, 2, 3))
            seg[:, :, normal] = offset
            seg[:, :, cross_axis] = levels[:, None]
            seg[:, 0, line_axis] = lower[line_axis]
            seg[:, 1, line_axis] = upper[line_axis]
            segments.append(seg)
    if not segments:
        return np.zeros((0, 2, 3))
    return np.concatenate(segments)


def project_points(
    points: np.ndarray,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    ang: tuple[float, float, float] | Rotation,
) -> np.ndarray:
    """Rotate 3D points as ``picasso.render.locs_rotation`` rotates
    localizations: about the viewport's center (at z = 0).

    Parameters
    ----------
    points : np.ndarray
        Points of shape ``(..., 3)``: x and y in camera pixels, z in
        the same units, shifted like the rendered localizations.
    viewport : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.
    ang : tuple of float or scipy.spatial.transform.Rotation
        Rotation of the view, see ``picasso.render.to_rotation``.

    Returns
    -------
    rotated : np.ndarray
        Rotated points of the same shape; x and y are in camera pixels
        of the unrotated frame, so ``viewport`` maps them onto the
        image.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0])
    points = np.asarray(points, dtype=float)
    flat = points.reshape(-1, 3) - center
    rotated = to_rotation(ang).apply(flat) + center
    return rotated.reshape(points.shape)


def draw_grid(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    grid: GridOverlay,
    pixelsize: float,
    color_range: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (0, 0, 0),
        (255, 255, 255),
    ),
) -> QtGui.QImage:
    """Draw a 2D grid onto rendered localizations.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations; drawn on in place if
        it is in the RGB32 or ARGB32 format.
    viewport : tuple
        Field of view shown in ``image``, ``((y_min, x_min), (y_max,
        x_max))`` in camera pixels.
    grid : GridOverlay
        The grid; ``planes`` and ``position`` are not used.
    pixelsize : float
        Camera pixel size in nm.
    color_range : tuple, optional
        RGB colors (0-255) of the pixels without localizations and of
        the pixels at the maximum contrast, see
        ``picasso.render.color_range``. Used to draw behind the
        localizations and to choose the automatic color. Default is
        black and white.

    Returns
    -------
    image : QImage
        Image with the drawn grid. Unchanged if the lines would be
        closer than ``MIN_GRID_SPACING_PX``.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    v_width = viewport_width(viewport)
    v_height = viewport_height(viewport)
    if v_width <= 0 or v_height <= 0:
        return image
    step = grid.spacing_nm / pixelsize
    px_x = image.width() / v_width  # display pixels per camera pixel
    px_y = image.height() / v_height
    if step * min(px_x, px_y) < MIN_GRID_SPACING_PX:
        return image
    lines = [
        QtCore.QLineF(x, 0.0, x, image.height())
        for x in (grid_levels(x_min, x_max, step) - x_min) * px_x
    ] + [
        QtCore.QLineF(0.0, y, image.width(), y)
        for y in (grid_levels(y_min, y_max, step) - y_min) * px_y
    ]
    return _paint_lines(image, lines, grid, color_range, antialias=False)


def draw_grid_3d(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    grid: GridOverlay,
    pixelsize: float,
    ang: tuple[float, float, float] | Rotation,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color_range: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (0, 0, 0),
        (255, 255, 255),
    ),
) -> QtGui.QImage:
    """Draw 3D grid planes, rotated with the localizations, onto a
    rendering of rotated localizations.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations; drawn on in place if
        it is in the RGB32 or ARGB32 format.
    viewport : tuple
        Field of view shown in ``image``, ``((y_min, x_min), (y_max,
        x_max))`` in camera pixels.
    grid : GridOverlay
        The grid.
    pixelsize : float
        Camera pixel size in nm.
    ang : tuple of float or scipy.spatial.transform.Rotation
        Rotation the localizations were rendered with, see
        ``picasso.render.to_rotation``.
    bounds : tuple
        ``((x_min, y_min, z_min), (x_max, y_max, z_max))`` of the data
        in camera pixels (z included), in the frame of the rendered
        localizations (i.e., with the same z shift).
    origin : tuple of float, optional
        A point on the grid in that frame, see ``grid_levels``; e.g.,
        ``(0, 0, -z_shift)`` keeps the lines at whole multiples of the
        spacing in the original z coordinates. Default is the origin.
    color_range : tuple, optional
        Colors of the empty and the brightest pixels, see
        ``draw_grid``. Default is black and white.

    Returns
    -------
    image : QImage
        Image with the drawn grid. Unchanged if the lines would be
        closer than ``MIN_GRID_SPACING_PX`` before rotation.
    """
    (y_min, x_min), (y_max, x_max) = viewport
    v_width = viewport_width(viewport)
    v_height = viewport_height(viewport)
    if v_width <= 0 or v_height <= 0:
        return image
    step = grid.spacing_nm / pixelsize
    px_x = image.width() / v_width
    px_y = image.height() / v_height
    if step * min(px_x, px_y) < MIN_GRID_SPACING_PX:
        return image
    segments = grid_segments_3d(
        bounds,
        step,
        ang,
        planes=grid.planes,
        position=grid.position,
        origin=origin,
    )
    if not len(segments):
        return image
    rotated = project_points(segments, viewport, ang)
    x = (rotated[:, :, 0] - x_min) * px_x
    y = (rotated[:, :, 1] - y_min) * px_y
    lines = [
        QtCore.QLineF(x0, y0, x1, y1)
        for (x0, x1), (y0, y1) in zip(x.tolist(), y.tolist())
    ]
    return _paint_lines(image, lines, grid, color_range, antialias=True)


def _paint_lines(
    image: QtGui.QImage,
    lines: list[QtCore.QLineF],
    grid: GridOverlay,
    color_range: tuple[tuple[int, int, int], tuple[int, int, int]],
    antialias: bool,
) -> QtGui.QImage:
    """Draw ``lines`` with the grid's style and blend mode.

    The lines are drawn opaque onto a transparent layer which is then
    painted with the style's opacity, so crossing lines do not add up
    to brighter dots."""
    if not lines:
        return image
    empty = tuple(int(c) for c in color_range[0])
    auto_color = "black" if _is_light(empty) else "white"
    style = grid.style
    layer = QtGui.QImage(
        image.width(),
        image.height(),
        QtGui.QImage.Format.Format_ARGB32_Premultiplied,
    )
    layer.fill(QtCore.Qt.GlobalColor.transparent)
    painter = replace(style, opacity=1.0).painter(layer, default=auto_color)
    painter.setRenderHint(
        QtGui.QPainter.RenderHint.Antialiasing,
        antialias or style.line_width > 1,
    )
    painter.drawLines(lines)
    painter.end()

    image = _as_rgb32(image)
    if grid.blend == "Over localizations":
        target = image
    else:
        # the grid over an empty rendering, then the localizations over
        # that
        target = QtGui.QImage(
            image.width(), image.height(), QtGui.QImage.Format.Format_RGB32
        )
        target.fill(QtGui.QColor(*empty))
    painter = QtGui.QPainter(target)
    painter.setOpacity(style.opacity)
    painter.drawImage(0, 0, layer)
    painter.end()
    if grid.blend == "Behind localizations":
        locs_bgra = _bgra_view(image)
        bg_bgra = _bgra_view(target)
        # BGRA byte order -> RGB
        locs_bgra[:, :, 2::-1] = composite_behind(
            locs_bgra[:, :, 2::-1], bg_bgra[:, :, 2::-1], *color_range
        )
    return image


def _is_light(rgb: tuple[int, int, int]) -> bool:
    """Whether a color (0-255) is light, by its perceived luminance."""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b > 127.5
