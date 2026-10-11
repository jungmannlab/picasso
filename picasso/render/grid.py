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

Optional labels print the coordinates of the lines, each just past one
end of its line: the top end of vertical and the left end of horizontal
lines in 2D, and in 3D one end of the lines of each axis, chosen to
keep the labels off the grid. The axial extent of SMLM data is usually
smaller than the lateral one, so the few z lines are preferred when
labels compete for space.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError
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
#: Automatic grid spacing relative to the viewport width, see
#: ``picasso.render.optimal_scalebar_length``.
AUTO_GRID_FRACTION = 1 / 10
#: Planes of the 3D grid, named by the two axes they contain.
GRID_PLANES = ("xy", "xz", "yz")
#: Where the 3D grid planes are placed, see ``GridOverlay``.
GRID_POSITIONS = ("back", "center")
#: Lines closer together than this (display pixels) are not drawn, they
#: would only tint the image.
MIN_GRID_SPACING_PX = 4
#: Largest number of grid lines along one axis that is drawn.
MAX_GRID_LINES = 2000
#: Pixel size of the labels if the style sets none.
DEFAULT_LABEL_SIZE = 12
# gap (display pixels) between a label and its line, edge or neighbor
_LABEL_GAP = 4
# opacity of the background behind each label, in the empty color
_LABEL_BACKGROUND_ALPHA = 160
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
    labels : bool, optional
        If True, the coordinates of the grid lines are printed just
        past one end of each line, in the line color, fully opaque and
        over the localizations, with the style's ``font_size`` (default
        ``DEFAULT_LABEL_SIZE``). In 3D they name the axis, e.g.,
        ``"z = 200 nm"``, and each axis is labeled once, on the plane
        where its lines are seen best. Labels that would overlap are
        thinned out; z lines are labeled first, so their labels are
        kept where they would collide with others. Default is False.

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
    labels: bool = False

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
    families = _line_families(bounds, step, ang, planes, position, origin)
    if not families:
        return np.zeros((0, 2, 3))
    return np.concatenate([seg for _, _, seg in families])


def _line_families(
    bounds: tuple,
    step: float,
    ang: tuple[float, float, float] | Rotation,
    planes: tuple[str, ...],
    position: Literal["back", "center"],
    origin: tuple[float, float, float],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The parallel lines of each grid plane, see ``grid_segments_3d``.

    Returns
    -------
    families : list of tuple
        ``(axis, levels, segments)`` per plane and direction: the lines
        sit at ``levels`` along ``axis`` and run perpendicular to it
        within the plane; segments of shape ``(n, 2, 3)``.
    """
    lower, upper = grid_box(bounds, step, origin)
    offsets = _plane_offsets(lower, upper, ang, planes, position)
    families = []
    for plane in planes:
        normal = _PLANE_NORMALS[plane]
        a, b = (axis for axis in range(3) if axis != normal)
        # lines along b at every level of a, and vice versa
        for line_axis, cross_axis in ((b, a), (a, b)):
            levels = grid_levels(
                lower[cross_axis],
                upper[cross_axis],
                step,
                origin[cross_axis],
            )
            n = len(levels)
            if not n:
                continue
            seg = np.empty((n, 2, 3))
            seg[:, :, normal] = offsets[plane]
            seg[:, :, cross_axis] = levels[:, None]
            seg[:, 0, line_axis] = lower[line_axis]
            seg[:, 1, line_axis] = upper[line_axis]
            families.append((cross_axis, levels, seg))
    return families


def _plane_offsets(
    lower: np.ndarray,
    upper: np.ndarray,
    ang: tuple[float, float, float] | Rotation,
    planes: tuple[str, ...],
    position: Literal["back", "center"],
) -> dict[str, float]:
    """Coordinate of each grid plane along its normal axis, see
    ``grid_segments_3d``."""
    depth_row = to_rotation(ang).as_matrix()[2]  # rotated z of a point
    offsets = {}
    for plane in planes:
        normal = _PLANE_NORMALS[plane]
        if position == "center":
            offsets[plane] = (lower[normal] + upper[normal]) / 2
        elif depth_row[normal] > 0:
            # the upper face is nearer the viewer; the lower one is back
            offsets[plane] = lower[normal]
        else:
            offsets[plane] = upper[normal]
    return offsets


def format_grid_labels(
    values_nm: np.ndarray | list[float],
    spacing_nm: float,
    axis: str | None = None,
) -> list[str]:
    """Label texts of grid lines.

    The values are given in µm if the spacing is at least 1 µm or a
    value reaches 10 µm, else in nm, with as many decimals as the
    spacing needs (at most 3).

    Parameters
    ----------
    values_nm : array-like of float
        Coordinates of the lines in nm.
    spacing_nm : float
        Grid spacing in nm.
    axis : str or None, optional
        Name of the axis put in front, e.g., ``"z"`` gives
        ``"z = 200 nm"``. Default is None.

    Returns
    -------
    texts : list of str
        One label per value, e.g., ``"1.5 µm"``.
    """
    prefix = "" if axis is None else f"{axis} = "
    values = np.asarray(values_nm, dtype=float)
    largest = float(np.abs(values).max()) if values.size else 0.0
    if spacing_nm >= 1000 or largest >= 10_000:
        unit, scale = "\u00b5m", 1e-3
    else:
        unit, scale = "nm", 1.0
    spacing = spacing_nm * scale
    decimals = 0
    while decimals < 3 and abs(round(spacing, decimals) - spacing) > (
        1e-6 * max(1.0, spacing)
    ):
        decimals += 1
    texts = []
    for value in values * scale:
        text = f"{value:.{decimals}f}"
        if float(text) == 0:
            text = f"{0.0:.{decimals}f}"  # no "-0"
        texts.append(f"{prefix}{text} {unit}")
    return texts


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
    levels_x = grid_levels(x_min, x_max, step)
    levels_y = grid_levels(y_min, y_max, step)
    xs = (levels_x - x_min) * px_x
    ys = (levels_y - y_min) * px_y
    lines = [QtCore.QLineF(x, 0.0, x, image.height()) for x in xs] + [
        QtCore.QLineF(0.0, y, image.width(), y) for y in ys
    ]
    image = _paint_lines(image, lines, grid, color_range, antialias=False)
    if grid.labels:
        image = _paint_labels(
            image,
            _grid_labels_2d(grid, levels_x, levels_y, xs, ys, step, pixelsize),
            grid,
            color_range,
        )
    return image


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
        Image with the drawn grid. Unchanged if the lateral lines would
        be closer than ``MIN_GRID_SPACING_PX`` before rotation.
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
    families = _line_families(
        bounds, step, ang, grid.planes, grid.position, origin
    )
    if not families:
        return image
    segments = np.concatenate([seg for _, _, seg in families])

    def to_display(points: np.ndarray) -> np.ndarray:
        rotated = project_points(points, viewport, ang)
        return np.stack(
            (
                (rotated[..., 0] - x_min) * px_x,
                (rotated[..., 1] - y_min) * px_y,
            ),
            axis=-1,
        )

    display = to_display(segments)
    lines = [
        QtCore.QLineF(x0, y0, x1, y1)
        for (x0, y0), (x1, y1) in display.tolist()
    ]
    image = _paint_lines(image, lines, grid, color_range, antialias=True)
    if grid.labels:
        image = _paint_labels(
            image,
            _grid_labels_3d(
                grid,
                families,
                step,
                origin,
                pixelsize,
                to_display,
                QtCore.QRectF(image.rect()),
            ),
            grid,
            color_range,
        )
    return image


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


def _label_font(grid: GridOverlay) -> QtGui.QFont:
    font = QtGui.QFont()
    size = grid.style.font_size
    font.setPixelSize(int(DEFAULT_LABEL_SIZE if size is None else size))
    return font


def _grid_labels_2d(
    grid: GridOverlay,
    levels_x: np.ndarray,
    levels_y: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    step: float,
    pixelsize: float,
) -> list[tuple[str, QtCore.QRectF]]:
    """Labels of the vertical lines along the top edge and of the
    horizontal lines along the left edge, each just past its line.

    Labels are kept at every n-th line counted from the coordinate
    origin, so they do not jump while panning, with n just large enough
    that neighbors do not overlap. Labels running into the corner
    shared by both edges are left out."""
    metrics = QtGui.QFontMetricsF(_label_font(grid))
    height = metrics.height()
    texts_x = format_grid_labels(levels_x * pixelsize, grid.spacing_nm)
    texts_y = format_grid_labels(levels_y * pixelsize, grid.spacing_nm)
    widths_x = [metrics.horizontalAdvance(t) for t in texts_x]
    widths_y = [metrics.horizontalAdvance(t) for t in texts_y]
    spacing_x = xs[1] - xs[0] if len(xs) > 1 else np.inf
    spacing_y = ys[1] - ys[0] if len(ys) > 1 else np.inf
    stride_x = _label_stride(max(widths_x, default=0), spacing_x)
    stride_y = _label_stride(height, spacing_y)
    # the corner: the column of the left labels and row of the top ones
    corner_x = _LABEL_GAP + max(widths_y, default=0)
    corner_y = _LABEL_GAP + height
    labels = []
    for level, x, text, width in zip(levels_x, xs, texts_x, widths_x):
        if round(level / step) % stride_x or x < corner_x:
            continue
        rect = QtCore.QRectF(x + _LABEL_GAP, _LABEL_GAP, width, height)
        labels.append((text, rect))
    for level, y, text, width in zip(levels_y, ys, texts_y, widths_y):
        if round(level / step) % stride_y or y < corner_y:
            continue
        rect = QtCore.QRectF(_LABEL_GAP, y + _LABEL_GAP, width, height)
        labels.append((text, rect))
    return labels


def _grid_labels_3d(
    grid: GridOverlay,
    families: list[tuple[int, np.ndarray, np.ndarray]],
    step: float,
    origin: tuple[float, float, float],
    pixelsize: float,
    to_display,
    bounds: QtCore.QRectF,
) -> list[tuple[str, QtCore.QRectF]]:
    """Labels of the 3D grid, as in 2D: each just past one end of its
    line, on the extension of the line.

    Every axis is labeled once, on one family of its lines (see
    ``_line_families``) and at the same end of all of them. Of all
    planes and both ends, the one placing the most labels clear of the
    grid's outline on screen (so they do not cover the data), inside
    ``bounds`` (the image) and clear of the labels placed before is
    chosen; on a tie, the family appearing longer on screen and the end
    farther up and to the left (the top end of vertical and the left end
    of horizontal lines, like in 2D). The z axis is labeled first, so it
    has the pick of the space: its few lines are preferred over the x
    and y lines. Labels are kept at every n-th line counted from the
    origin, with n just large enough that neighbors do not overlap.

    ``to_display`` maps points ``(..., 3)`` to display pixels
    ``(..., 2)``."""
    metrics = QtGui.QFontMetricsF(_label_font(grid))
    height = metrics.height()
    by_axis = {}
    for axis, levels, segments in families:
        ends = to_display(segments)  # (n, 2, 2)
        length = float(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).mean())
        if length >= 1:  # else the lines point at the viewer
            by_axis.setdefault(axis, []).append((length, levels, ends))
    # the outline of all lines on screen, which labels should not cover
    all_ends = [ends for fam in by_axis.values() for _, _, ends in fam]
    outline = _outline(np.concatenate(all_ends)) if all_ends else None
    labels = []
    for axis in sorted(by_axis, key=lambda a: (a != 2, a)):  # z first
        best = None
        for length, levels, ends in by_axis[axis]:
            texts = format_grid_labels(
                (levels - origin[axis]) * pixelsize,
                grid.spacing_nm,
                axis="xyz"[axis],
            )
            widths = [metrics.horizontalAdvance(t) for t in texts]
            sums = ends.sum(axis=2).mean(axis=0)
            for end in (0, 1) if sums[0] <= sums[1] else (1, 0):
                candidate = _line_end_labels(
                    ends,
                    end,
                    levels,
                    texts,
                    widths,
                    height,
                    (levels - origin[axis]) / step,
                )
                inside = [bounds.contains(rect) for _, rect in candidate]
                clear = [
                    ok
                    and (
                        outline is None
                        or not outline.intersects(QtGui.QPolygonF(rect))
                    )
                    and not any(rect.intersects(r) for _, r in labels)
                    for ok, (_, rect) in zip(inside, candidate)
                ]
                score = (sum(clear), sum(inside), length)
                if best is None or score > best[0]:
                    best = (score, candidate)
        labels += best[1]
    return labels


def _outline(ends: np.ndarray) -> QtGui.QPolygonF | None:
    """Convex outline (display pixels) of the line end points ``ends``
    (..., 2); None if they are collinear."""
    points = ends.reshape(-1, 2)
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None
    return QtGui.QPolygonF([QtCore.QPointF(*points[k]) for k in hull.vertices])


def _line_end_labels(
    ends: np.ndarray,
    end: int,
    levels: np.ndarray,
    texts: list[str],
    widths: list[float],
    height: float,
    indices: np.ndarray,
) -> list[tuple[str, QtCore.QRectF]]:
    """Labels just past the ``end`` (0 or 1) of parallel lines with
    display end points ``ends`` (n, 2, 2), on their extension; every
    n-th line (by ``indices``, the lines counted from the origin) so
    neighbors do not overlap."""
    anchors = ends[:, end]
    along = (anchors - ends[:, 1 - end]).mean(axis=0)
    along = along / np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    if len(anchors) > 1:
        gap = anchors[1] - anchors[0]
        spacing = float(np.linalg.norm(gap - np.dot(gap, along) * along))
    else:
        spacing = np.inf
    stride = _label_stride(
        max(abs(across[0]) * w + abs(across[1]) * height for w in widths),
        spacing,
    )
    labels = []
    for index, anchor, text, width in zip(indices, anchors, texts, widths):
        if round(index) % stride:
            continue
        extent = abs(along[0]) * width + abs(along[1]) * height
        middle = anchor + along * (_LABEL_GAP + extent / 2)
        labels.append((text, _centered_rect(middle, width, height)))
    return labels


def _label_stride(size: float, spacing: float) -> int:
    """Label every n-th line so labels of ``size`` (along the axis)
    spaced ``spacing`` apart do not overlap."""
    if not np.isfinite(spacing) or spacing <= 0:
        return 1
    return max(1, int(np.ceil((size + _LABEL_GAP) / spacing)))


def _centered_rect(
    middle: np.ndarray, width: float, height: float
) -> QtCore.QRectF:
    return QtCore.QRectF(
        middle[0] - width / 2, middle[1] - height / 2, width, height
    )


def _place_labels(
    labels: list[tuple[str, QtCore.QRectF]],
    bounds: QtCore.QRectF,
) -> list[tuple[str, QtCore.QRectF]]:
    """The labels, in order, that lie inside ``bounds`` and do not
    overlap a label placed before them."""
    placed = []
    for text, rect in labels:
        if not bounds.contains(rect):
            continue
        padded = rect.adjusted(
            -_LABEL_GAP / 2, -_LABEL_GAP / 2, _LABEL_GAP / 2, _LABEL_GAP / 2
        )
        if any(padded.intersects(other) for _, other in placed):
            continue
        placed.append((text, rect))
    return placed


def _paint_labels(
    image: QtGui.QImage,
    labels: list[tuple[str, QtCore.QRectF]],
    grid: GridOverlay,
    color_range: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> QtGui.QImage:
    """Print ``labels`` (text and rectangle) in the line color, on a
    translucent background of the empty color so they stay legible
    over localizations.

    Labels are placed in order; one that would overlap a placed label
    (e.g., where the edges of two axes meet in 3D) or reach past the
    image is left out."""
    labels = _place_labels(labels, QtCore.QRectF(image.rect()))
    if not labels:
        return image
    empty = tuple(int(c) for c in color_range[0])
    auto_color = "black" if _is_light(empty) else "white"
    style = grid.style
    color = QtGui.QColor(auto_color if style.color is None else style.color)
    color.setAlpha(255)
    background = QtGui.QColor(*empty, _LABEL_BACKGROUND_ALPHA)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
    painter.setFont(_label_font(grid))
    painter.setPen(color)
    for text, rect in labels:
        painter.fillRect(rect.adjusted(-2, 0, 2, 0), background)
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    return image


def _is_light(rgb: tuple[int, int, int]) -> bool:
    """Whether a color (0-255) is light, by its perceived luminance."""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b > 127.5
