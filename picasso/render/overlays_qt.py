"""
picasso.render.overlays_qt
~~~~~~~~~~~~~~~~~~~~~~~~~~

Qt drawing on rendered images (QImage): picks, points, scale bar,
legend, minimap and rotation widgets, plus PDF/SVG export.

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .. import lib, __version__
from .geometry import (
    to_rotation,
    viewport_height,
    viewport_width,
    adjust_viewport_decorator,
    map_to_view,
)

if TYPE_CHECKING:
    from PyQt6 import QtGui, QtCore, QtSvg
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")
    QtCore = lib._LazyQtModule("PyQt6.QtCore")
    QtSvg = lib._LazyQtModule("PyQt6.QtSvg")


POLYGON_POINTER_SIZE = 16  # must be even
# opacity of the fill of a brush pick, so that the localizations under
# the painted region stay visible
BRUSH_FILL_ALPHA = 70
# line styles of the tool overlays, see ``OverlayStyle``
LINE_STYLES = ("Solid", "Dashed", "Dotted", "Dash-dot")


@dataclass(frozen=True)
class OverlayStyle:
    """Appearance of a tool overlay drawn onto a rendered image, such
    as picks or measured points.

    Every field left at None falls back to the default of the drawing
    function, e.g., a yellow outline for picks, with only brush picks
    filled.

    Parameters
    ----------
    color : QColor, str or None, optional
        Color of the lines and labels: a ``QColor`` or anything it
        accepts, e.g., ``"yellow"`` or ``"#FF8800"``. Default None.
    line_style : {"Solid", "Dashed", "Dotted", "Dash-dot"}, optional
        Pattern of the lines. Default "Solid".
    line_width : float, optional
        Width of the lines in display pixels. Default 1.
    opacity : float, optional
        Opacity of the lines and labels, from 0 (transparent) to 1.
        Default 1.
    fill_opacity : float or None, optional
        Opacity of the fill of closed shapes, from 0 (no fill) to 1,
        filled with ``color``. Default None.
    font_size : int or None, optional
        Pixel size of the labels. Default None.

    Raises
    ------
    ValueError
        If ``line_style`` is unknown, ``line_width`` or ``font_size`` is
        not positive or an opacity is outside [0, 1].
    """

    color: QtGui.QColor | str | None = None
    line_style: Literal["Solid", "Dashed", "Dotted", "Dash-dot"] = "Solid"
    line_width: float = 1.0
    opacity: float = 1.0
    fill_opacity: float | None = None
    font_size: int | None = None

    def __post_init__(self) -> None:
        if self.line_style not in LINE_STYLES:
            raise ValueError(
                f"Unknown line style: {self.line_style}. Expected one of "
                f"{', '.join(LINE_STYLES)}."
            )
        if not self.line_width > 0:
            raise ValueError("line_width must be positive.")
        if self.font_size is not None and not self.font_size > 0:
            raise ValueError("font_size must be positive.")
        for name in ("opacity", "fill_opacity"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

    def qcolor(self, default: QtGui.QColor | str = "yellow") -> QtGui.QColor:
        """Color of the lines and labels, including their opacity.

        Parameters
        ----------
        default : QColor or str, optional
            Color used if ``color`` is None. Default "yellow".

        Returns
        -------
        color : QColor
            The color.
        """
        color = QtGui.QColor(default if self.color is None else self.color)
        color.setAlphaF(color.alphaF() * self.opacity)
        return color

    def pen(self, default: QtGui.QColor | str = "yellow") -> QtGui.QPen:
        """Pen that draws the lines and labels.

        Parameters
        ----------
        default : QColor or str, optional
            Color used if ``color`` is None. Default "yellow".

        Returns
        -------
        pen : QPen
            The pen.
        """
        pen = QtGui.QPen(self.qcolor(default), self.line_width)
        pen.setStyle(
            {
                "Solid": QtCore.Qt.PenStyle.SolidLine,
                "Dashed": QtCore.Qt.PenStyle.DashLine,
                "Dotted": QtCore.Qt.PenStyle.DotLine,
                "Dash-dot": QtCore.Qt.PenStyle.DashDotLine,
            }[self.line_style]
        )
        return pen

    def fill(
        self,
        default: QtGui.QColor | str = "yellow",
        default_opacity: float = 0.0,
    ) -> QtGui.QBrush | None:
        """Brush that fills closed shapes, or None if they are not
        filled.

        Parameters
        ----------
        default : QColor or str, optional
            Color used if ``color`` is None. Default "yellow".
        default_opacity : float, optional
            Opacity used if ``fill_opacity`` is None. Default 0, i.e.,
            no fill.

        Returns
        -------
        brush : QBrush or None
            The brush.
        """
        opacity = (
            default_opacity if self.fill_opacity is None else self.fill_opacity
        )
        if opacity <= 0:
            return None
        color = QtGui.QColor(default if self.color is None else self.color)
        color.setAlphaF(color.alphaF() * opacity)
        return QtGui.QBrush(color)

    def painter(
        self,
        image: QtGui.QImage,
        default: QtGui.QColor | str = "yellow",
        default_font_size: int | None = None,
    ) -> QtGui.QPainter:
        """Open a painter on ``image`` with this style's pen and font.

        Lines wider than one pixel are antialiased; thin lines are not,
        so that they stay sharp.

        Parameters
        ----------
        image : QImage
            Image to paint on.
        default : QColor or str, optional
            Color used if ``color`` is None. Default "yellow".
        default_font_size : int or None, optional
            Pixel size of the labels if ``font_size`` is None; None
            keeps the painter's font. Default None.

        Returns
        -------
        painter : QPainter
            The painter; call its ``end`` once done.
        """
        painter = QtGui.QPainter(image)
        painter.setPen(self.pen(default))
        if self.line_width > 1:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        font_size = (
            default_font_size if self.font_size is None else self.font_size
        )
        if font_size is not None:
            font = painter.font()
            font.setPixelSize(int(font_size))
            painter.setFont(font)
        return painter


def _overlay_style(
    style: OverlayStyle | None,
    color: QtGui.QColor | str | None,
) -> OverlayStyle:
    """Combine the ``style`` and ``color`` arguments of a drawing
    function; ``color``, if given, takes precedence."""
    if style is None:
        style = OverlayStyle()
    if color is not None:
        style = replace(style, color=color)
    return style


def export_qimage_to_pdf(
    image: QtGui.QImage, path: str, dpi: int = 96
) -> None:
    """Write a rendered image to a PDF at its original physical size.

    The page is sized so that one image pixel is 1/96 inch regardless of
    ``dpi``, which only sets the resolution the image is rasterized at.

    Parameters
    ----------
    image : QtGui.QImage
        The rendered image.
    path : str
        Where to write the PDF.
    dpi : int, optional
        Resolution of the PDF writer. Default 96.
    """
    writer = QtGui.QPdfWriter(path)

    # Fixed physical page size (1 image pixel = 1/96 inch, regardless of dpi)
    width_mm = image.width() * 25.4 / 96
    height_mm = image.height() * 25.4 / 96

    page_size = QtGui.QPageSize(
        QtCore.QSizeF(width_mm, height_mm),
        QtGui.QPageSize.Unit.Millimeter,
    )
    writer.setPageSize(page_size)
    writer.setResolution(dpi)

    # Painter coordinates: 1 unit = 1/dpi inch, so full page =
    # (width_mm / 25.4) * dpi = image.width() * dpi / 96
    draw_width = image.width() * dpi / 96
    draw_height = image.height() * dpi / 96

    painter = QtGui.QPainter(writer)
    painter.drawImage(QtCore.QRectF(0, 0, draw_width, draw_height), image)
    painter.end()


def export_qimage_to_svg(image: QtGui.QImage, path: str):
    """Write a rendered image to an SVG, embedded at its pixel size.

    Parameters
    ----------
    image : QtGui.QImage
        The rendered image.
    path : str
        Where to write the SVG.
    """
    generator = QtSvg.QSvgGenerator()
    generator.setFileName(path)
    generator.setSize(image.size())
    generator.setViewBox(QtCore.QRect(0, 0, image.width(), image.height()))

    painter = QtGui.QPainter(generator)
    painter.drawImage(0, 0, image)
    painter.end()


def get_rectangle_pick_polygon(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    width: float,
    return_most_right: bool = False,
) -> QtGui.QPolygonF | tuple[float, float]:
    """Find QtGui.QPolygonF object used for drawing a rectangular
    pick.

    Parameters
    ----------
    start_x, start_y : float
        One end of the rectangle's center line.
    end_x, end_y : float
        The other end of the center line.
    width : float
        Width of the rectangle, perpendicular to the center line.
    return_most_right : bool, optional
        Also return the rightmost corner, where the GUI anchors the pick
        label. Default False.

    Returns
    -------
    p : QtGui.QPolygonF
        The polygon.
    most_right : tuple
        Only if ``return_most_right``: the ``(x, y)`` of the rightmost corner.
    """
    X, Y = lib.get_pick_rectangle_corners(
        start_x, start_y, end_x, end_y, width
    )
    p = QtGui.QPolygonF()
    for x, y in zip(X, Y):
        p.append(QtCore.QPointF(x, y))
    if return_most_right:
        ix_most_right = np.argmax(X)
        x_most_right = X[ix_most_right]
        y_most_right = Y[ix_most_right]
        return p, (x_most_right, y_most_right)
    return p


def _draw_picks_circle(
    image: QtGui.QImage,
    viewport: list[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # pick coords in camera pixels
    pick_size: float,  # diameter in camera pixels
    point_picks: bool = False,
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw circular picks onto the image of rendered localizations.
    See ``draw_picks`` for more details."""
    style = _overlay_style(style, color)
    painter = style.painter(image)
    if point_picks:  # draw circular picks as points
        painter.setBrush(QtGui.QBrush(style.qcolor()))
        for i, pick in enumerate(picks):
            # convert from camera units to display units
            cx, cy = map_to_view(*pick, image.size(), viewport)
            painter.drawEllipse(QtCore.QPoint(cx, cy), 3, 3)
            if annotate_picks:
                painter.drawText(cx + 20, cy + 20, str(i))

    else:  # draw circles
        d = int(pick_size * image.width() / viewport_width(viewport))
        fill = style.fill()
        if fill is not None:
            painter.setBrush(fill)
        for i, pick in enumerate(picks):
            # check that the pick is within the view
            if (
                pick[0] < viewport[0][1]
                or pick[0] > viewport[1][1]
                or pick[1] < viewport[0][0]
                or pick[1] > viewport[1][0]
            ):
                continue

            # convert from camera units to display units
            cx, cy = map_to_view(*pick, image.size(), viewport)
            painter.drawEllipse(int(cx - d / 2), int(cy - d / 2), d, d)
            if annotate_picks:
                painter.drawText(int(cx + d / 2), int(cy + d / 2), str(i))
    painter.end()
    return image


def _draw_picks_rectangle(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    pick_size: float,  # width in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw rectangular picks onto the image of rendered
    localizations. See ``draw_picks`` for more details."""
    style = _overlay_style(style, color)
    w = pick_size * image.width() / viewport_width(viewport)
    painter = style.painter(image)
    fill = style.fill()
    if fill is not None:
        painter.setBrush(fill)
    for i, pick in enumerate(picks):
        # convert from camera units to display units
        start_x, start_y = map_to_view(*pick[0], image.size(), viewport)
        end_x, end_y = map_to_view(*pick[1], image.size(), viewport)
        # draw a rectangle
        polygon, most_right = get_rectangle_pick_polygon(
            start_x, start_y, end_x, end_y, w, return_most_right=True
        )
        painter.drawPolygon(polygon)
        # draw a straight line across the pick, over the fill
        painter.drawLine(start_x, start_y, end_x, end_y)
        if annotate_picks:
            painter.drawText(int(most_right[0]), int(most_right[1]), str(i))
    painter.end()
    return image


def _draw_picks_polygon(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw polygon picks onto the image of rendered localizations. See
    ``draw_picks`` for more details."""
    style = _overlay_style(style, color)
    painter = style.painter(image)
    fill = style.fill()
    for i, pick in enumerate(picks):
        # only closed polygons are filled; the one being drawn is not
        if fill is not None and len(pick) > 3 and pick[0] == pick[-1]:
            polygon = QtGui.QPolygonF(
                [
                    QtCore.QPointF(*map_to_view(*p, image.size(), viewport))
                    for p in pick
                ]
            )
            path = QtGui.QPainterPath()
            path.addPolygon(polygon)
            painter.fillPath(path, fill)
        oldpoint = []
        for point in pick:
            cx, cy = map_to_view(*point, image.size(), viewport)
            painter.drawEllipse(
                QtCore.QPoint(cx, cy),
                int(POLYGON_POINTER_SIZE / 2),
                int(POLYGON_POINTER_SIZE / 2),
            )
            if oldpoint != []:  # draw the line
                ox, oy = map_to_view(*oldpoint, image.size(), viewport)
                painter.drawLine(cx, cy, ox, oy)
            oldpoint = point

        # annotate picks
        if len(pick) and annotate_picks:
            painter.drawText(
                cx + int(POLYGON_POINTER_SIZE / 2) + 10,
                cy + int(POLYGON_POINTER_SIZE / 2) + 10,
                str(i),
            )
    painter.end()
    return image


def _draw_picks_square(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    pick_size: float,  # side length in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw square picks onto the image of rendered localizations."""
    style = _overlay_style(style, color)
    w = int(pick_size * image.width() / viewport_width(viewport))
    painter = style.painter(image)
    fill = style.fill()
    if fill is not None:
        painter.setBrush(fill)
    for i, pick in enumerate(picks):
        # check that the pick is within the view
        if (
            pick[0] < viewport[0][1]
            or pick[0] > viewport[1][1]
            or pick[1] < viewport[0][0]
            or pick[1] > viewport[1][0]
        ):
            continue

        # convert from camera units to display units
        cx, cy = map_to_view(*pick, image.size(), viewport)
        painter.drawRect(int(cx - w / 2), int(cy - w / 2), w, w)

        # annotate picks
        if annotate_picks:
            painter.drawText(
                int(cx + w / 2) + 10, int(cy + w / 2) + 10, str(i)
            )
    painter.end()
    return image


def _draw_picks_box(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw box picks onto the image of rendered localizations. See
    ``draw_picks`` for more details."""
    style = _overlay_style(style, color)
    painter = style.painter(image)
    fill = style.fill()
    if fill is not None:
        painter.setBrush(fill)
    for i, pick in enumerate(picks):
        X, Y = lib.get_pick_box_corners(pick)
        # unlike the click-placed shapes, a box can be larger than the
        # view, so cull on intersection rather than on its center
        if (
            max(X) < viewport[0][1]
            or min(X) > viewport[1][1]
            or max(Y) < viewport[0][0]
            or min(Y) > viewport[1][0]
        ):
            continue

        # convert from camera units to display units
        x0, y0 = map_to_view(min(X), min(Y), image.size(), viewport)
        x1, y1 = map_to_view(max(X), max(Y), image.size(), viewport)
        painter.drawRect(x0, y0, x1 - x0, y1 - y0)

        # annotate picks
        if annotate_picks:
            painter.drawText(x1 + 10, y1 + 10, str(i))
    painter.end()
    return image


def brush_pick_path(
    pick: list[tuple[float, list[tuple[float, float]]]],
    image_size: QtCore.QSize,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
) -> QtGui.QPainterPath:
    """Return the painted outline of a brush pick in display units.

    Each stroke is swept with a round-capped, round-joined pen of its
    own width - the same region ``lib.check_if_in_brush_stroke`` tests
    against - and the strokes of a pick are united into a single path,
    so that the region can be filled once. Filling stroke by stroke
    would darken every overlap, and the strokes of a merged pick always
    overlap.

    Parameters
    ----------
    pick : list of tuples
        One brush pick, i.e., a list of ``(width, path)`` strokes in
        camera pixels.
    image_size : QSize
        Size of the image the pick is drawn onto.
    viewport : tuple
        Current field of view in camera pixels, ``((y_min, x_min),
        (y_max, x_max))``.

    Returns
    -------
    path : QPainterPath
        The painted region in display coordinates.
    """
    scale = image_size.width() / viewport_width(viewport)
    region = QtGui.QPainterPath()
    for stroke in pick:
        width, X, Y = lib.brush_stroke_arrays(stroke)
        path = QtGui.QPainterPath()
        for j, (x, y) in enumerate(zip(X, Y)):
            cx, cy = map_to_view(x, y, image_size, viewport)
            point = QtCore.QPointF(cx, cy)
            if j == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        if len(X) == 1:  # a dot: sweeping a zero-length path
            path.lineTo(path.currentPosition())
        pen = QtGui.QPen(
            QtGui.QColor("black"),
            max(width * scale, 1.0),
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
        stroker = QtGui.QPainterPathStroker(pen)
        region = region.united(stroker.createStroke(path))
    return region.simplified()


def _draw_picks_brush(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    picks: list[tuple],  # picks in camera pixels
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw brush picks onto the image of rendered localizations, by
    default as a translucent highlight with a solid outline. See
    ``draw_picks`` for more details."""
    style = _overlay_style(style, color)
    fill = style.fill(default_opacity=BRUSH_FILL_ALPHA / 255)
    painter = style.painter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    for i, pick in enumerate(picks):
        if not len(pick):
            continue
        x_min, x_max, y_min, y_max = lib.pick_bounds(pick, "Brush", None)
        # a painted region can be larger than the view, so cull on
        # intersection rather than on a center
        if (
            x_max < viewport[0][1]
            or x_min > viewport[1][1]
            or y_max < viewport[0][0]
            or y_min > viewport[1][0]
        ):
            continue

        region = brush_pick_path(pick, image.size(), viewport)
        if fill is not None:
            painter.fillPath(region, fill)
        painter.drawPath(region)

        # annotate picks just outside the end of the last stroke
        if annotate_picks:
            width, X, Y = lib.brush_stroke_arrays(pick[-1])
            cx, cy = map_to_view(X[-1], Y[-1], image.size(), viewport)
            r = int(width / 2 * image.width() / viewport_width(viewport))
            painter.drawText(cx + r + 10, cy + r + 10, str(i))
    painter.end()
    return image


@adjust_viewport_decorator
def draw_picks(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    pick_shape: Literal[
        "Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"
    ],
    picks: list[tuple],  # pick coords in camera pixels
    pick_size: float | None,  # diameter in camera pixels
    point_picks: bool = False,
    annotate_picks: bool = False,
    color: QtGui.QColor | None = None,  # default: yellow
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw all selected picks onto the image (QImage) of rendered
    localizations.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    pick_shape : {"Circle", "Rectangle", "Polygon", "Square", "Box", "Brush"}
        Shape of the picks to be drawn.
    picks : list of tuples
        List of picks, where each pick is a tuple specifying the pick
        coordinates. Note: this must match the format of the given pick
        shape.
    pick_size : float or None
        Size of the picks in camera pixels. For "Circle", this is the
        diameter; for "Rectangle", this is the width; for "Square", this
        is the side length. This parameter is ignored for "Polygon",
        "Box" and "Brush" picks, which carry their own extent.
    point_picks : bool, optional
        If True and pick_shape is "Circle", draw picks as points instead
        of circles. Default is False.
    annotate_picks : bool, optional
        If True, annotate each pick with its index in the picks list.
        Default is False.
    color : QtGui.QColor, optional
        Color of the picks; overrides the color of ``style``. Default is
        yellow.
    style : OverlayStyle, optional
        Line style, width, opacity, fill and label size of the picks.
        By default, picks are drawn with solid 1-pixel lines and only
        brush picks are filled.

    Returns
    -------
    image : QImage
        Image with the drawn picks.

    Raises
    ------
    ValueError
        If ``pick_shape`` is not recognized.
    """
    image = image.copy()
    if pick_shape == "Circle":
        return _draw_picks_circle(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            point_picks=point_picks,
            annotate_picks=annotate_picks,
            color=color,
            style=style,
        )
    elif pick_shape == "Rectangle":
        return _draw_picks_rectangle(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            annotate_picks=annotate_picks,
            color=color,
            style=style,
        )
    elif pick_shape == "Polygon":
        return _draw_picks_polygon(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
            style=style,
        )
    elif pick_shape == "Square":
        return _draw_picks_square(
            image,
            viewport=viewport,
            picks=picks,
            pick_size=pick_size,
            annotate_picks=annotate_picks,
            color=color,
            style=style,
        )
    elif pick_shape == "Box":
        return _draw_picks_box(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
            style=style,
        )
    elif pick_shape == "Brush":
        return _draw_picks_brush(
            image,
            viewport=viewport,
            picks=picks,
            annotate_picks=annotate_picks,
            color=color,
            style=style,
        )
    else:
        raise ValueError(f"Unknown pick shape: {pick_shape}")


@adjust_viewport_decorator
def draw_points(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    points: list[tuple],  # points in camera pixels,
    pixelsize: int | float,  # camera pixel size in nm
    color: QtGui.QColor | None = None,  # default: yellow
    mark_width: int = 20,  # width of the drawn crosses in display pixels
    cursor: tuple | None = None,  # live cursor position in camera pixels
    style: OverlayStyle | None = None,
) -> QtGui.QImage:
    """Draw points, lines and distances between them onto image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    points : list of tuples
        List of points, where each point is a tuple specifying the point
        coordinates in camera pixels.
    pixelsize : int or float
        Camera pixel size in nm.
    color : QtGui.QColor, optional
        Color of the points, lines and text; overrides the color of
        ``style``. Default is yellow.
    mark_width : int, optional
        Width of the drawn crosses in display pixels. Default is 20.
    cursor : tuple or None, optional
        Current cursor position in camera pixels. If given, it is drawn
        as a cross and, when at least one point exists, a line with the
        live distance to the last point is shown. Default is None.
    style : OverlayStyle, optional
        Line style, width and opacity of the lines between the points
        and the size of the distance labels (20 pixels by default).
        The crosses are drawn with the same width and opacity, always
        with solid lines. Its fill is ignored.

    Returns
    -------
    image : QImage
        Image with the drawn points.
    """
    style = _overlay_style(style, color)
    painter = style.painter(image, default_font_size=20)
    line_pen = painter.pen()
    # a dashed cross would lose its center, so crosses stay solid
    cross_pen = QtGui.QPen(line_pen)
    cross_pen.setStyle(QtCore.Qt.PenStyle.SolidLine)

    def draw_cross(x, y):
        """Draw a cross marker centered at display coordinates."""
        painter.setPen(cross_pen)
        painter.drawPoint(x, y)
        painter.drawLine(x, y, int(x + mark_width / 2), y)
        painter.drawLine(x, y, x, int(y + mark_width / 2))
        painter.drawLine(x, y, int(x - mark_width / 2), y)
        painter.drawLine(x, y, x, int(y - mark_width / 2))

    def draw_distance(x1, y1, x2, y2, p1, p2):
        """Draw a line and the distance label between two points."""
        painter.setPen(line_pen)
        painter.drawLine(x1, y1, x2, y2)
        # get distance with 2 decimal places
        distance = (
            float(
                int(
                    np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    * pixelsize
                    * 100
                )
            )
            / 100
        )
        painter.drawText(
            int((x1 + x2) / 2 + mark_width),
            int((y1 + y2) / 2 + mark_width),
            str(distance) + " nm",
        )

    cx = []
    cy = []
    ox = []  # together with oldpoint used for drawing
    oy = []  # lines between points
    oldpoint = []
    for point in points:
        # convert to display units
        if oldpoint != []:
            ox, oy = map_to_view(*oldpoint, image.size(), viewport=viewport)
        cx, cy = map_to_view(*point, image.size(), viewport=viewport)

        # draw a cross
        draw_cross(cx, cy)

        # draw a line between points and show distance
        if oldpoint != []:
            draw_distance(cx, cy, ox, oy, oldpoint, point)
        oldpoint = point

    # draw the live cursor as a cross and the running distance to the
    # last placed point
    if cursor is not None:
        ccx, ccy = map_to_view(*cursor, image.size(), viewport=viewport)
        draw_cross(ccx, ccy)
        if points:
            lx, ly = map_to_view(*points[-1], image.size(), viewport=viewport)
            draw_distance(ccx, ccy, lx, ly, points[-1], cursor)

    painter.end()
    return image


@adjust_viewport_decorator
def draw_scalebar(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    scalebar_length_nm: int | float,
    pixelsize: int | float,
    display_length: bool = True,
    color: QtGui.QColor | None = None,  # default: white
    display_height: int = 10,
    margin: tuple[int, int] = (35, 20),
    text_spacer: int = 40,
    text_fontsize: int = 20,
) -> QtGui.QImage:
    """Draw a scalebar into rendered localizations (QImage).

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    scalebar_length_nm : int or float
        Scale bar length in nm.
    pixelsize : int or float
        Camera pixel size in nm.
    color : QColor, optional
        Color of the scalebar and text. Default is white.
    display_length : bool, optional
        Whether to display scalebar length in nm. Default is True.
    display_height : int, optional
        Thickness of the scalebar in display pixels. Default is 10.
    margin : tuple of int, optional
        Margins from the right and bottom edges in display pixels.
        Default is (35, 20).
    text_spacer : int, optional
        Spacing between the scalebar and the displayed length text in
        display pixels. Only used if display_length is True. Default is
        40.
    text_fontsize : int, optional
        Font size of the displayed length text in display pixels. Only
        used if display_length is True. Default is 20.

    Returns
    -------
    image : QImage
        Image with the drawn scalebar.
    """
    if color is None:
        color = QtGui.QColor("white")
    painter = QtGui.QPainter(image)
    painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    painter.setBrush(QtGui.QBrush(color))

    length_camerapxl = scalebar_length_nm / pixelsize
    length_displaypxl = int(
        round(image.width() * length_camerapxl / viewport_width(viewport))
    )

    # draw a rectangle
    x = image.width() - length_displaypxl - margin[0]
    y = image.height() - display_height - margin[1]
    painter.drawRect(x, y, length_displaypxl, display_height)

    # display scalebar's length
    if display_length:
        font = painter.font()
        font.setPixelSize(text_fontsize)
        painter.setFont(font)
        painter.setPen(color)
        text_width = length_displaypxl + 2 * text_spacer
        text_height = text_spacer
        painter.drawText(
            x - text_spacer,
            y - 25,
            text_width,
            text_height,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
            f"{str(scalebar_length_nm)} nm",
        )
    return image


def draw_legend(
    image: QtGui.QImage,
    channel_names: list[str],
    channel_colors: list[tuple[int, int, int]],
    init_pos: tuple[int, int] = (12, 26),
    dy: int = 24,
    padding: int = 4,
    text_fontsize: int = 16,
) -> QtGui.QImage:
    """Draw a legend for multichannel data in the top left corner over
    rendered localizations (QImage).

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    channel_names : list of str
        List of channel names to be displayed in the legend.
    channel_colors : list of tuples
        List of RGB tuples corresponding to the colors of the channels.
        Must range between 0 and 255.
    init_pos : tuple of int, optional
        Initial position (x, y) of the first channel name in display
        pixels. Default is (12, 26).
    dy : int, optional
        Space between channel names in display pixels. Default is 24.
    padding : int, optional
        Padding around the text in display pixels. Default is 4.
    text_fontsize : int, optional
        Font size of the channel names in display pixels. Default is 16.

    Returns
    -------
    image : QImage
        Image with the drawn legend.
    """
    assert len(channel_names) == len(channel_colors), (
        "Length of channel_names must match number of channels in " "dataset."
    )
    n_channels = len(channel_names)
    painter = QtGui.QPainter(image)
    # initial positions
    x, y = init_pos
    font = painter.font()
    font.setPixelSize(text_fontsize)
    painter.setFont(font)
    fm = QtGui.QFontMetrics(font)
    for i in range(n_channels):
        text = channel_names[i]
        # draw black background
        text_rect = fm.boundingRect(text)
        bg_rect = QtCore.QRect(
            x - padding,
            y - fm.ascent() - padding,
            text_rect.width() + 2 * padding,
            fm.height() + 2 * padding,
        )
        painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.black))
        painter.drawRect(bg_rect)
        # draw colored text
        color_rgb = channel_colors[i]
        color = QtGui.QColor(color_rgb[0], color_rgb[1], color_rgb[2])
        painter.setPen(QtGui.QPen(color))
        painter.drawText(QtCore.QPoint(x, y), text)
        y += dy
    return image


def _format_tick(value: float) -> str:
    """Format a color bar tick value with a sensible number of digits.

    Parameters
    ----------
    value : float
        Value of the rendered property at the tick.

    Returns
    -------
    text : str
        Formatted value.
    """
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e5 or magnitude < 1e-2:
        return f"{value:.1e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _colorbar_layout(
    colors: list[tuple[float, float, float]] | lib.FloatArray2D,
    min_value: float,
    max_value: float,
    label: str = "",
    vertical: bool = True,
    bar_length: int = 400,
    bar_width: int = 40,
    n_ticks: int = 5,
    color: QtGui.QColor | None = None,  # default: white
    background: QtGui.QColor | None = None,  # default: black
    text_fontsize: int = 20,
    margin: int = 12,
    tick_length: int = 8,
    tick_spacer: int = 4,
) -> dict:
    """Lay out a color bar: its size and everything needed to paint it.

    Shared by ``colorbar_image`` and ``colorbar_svg``, so that the two
    draw the same bar onto their different paint devices. See
    ``colorbar_image`` for the parameters.

    Returns
    -------
    layout : dict
        Everything ``_paint_colorbar`` needs, including the ``width``
        and ``height`` of the bar in display pixels.
    """
    colors_arr = np.asarray(colors, dtype=np.float32)
    assert (
        colors_arr.ndim == 2 and colors_arr.shape[1] == 3
    ), "colors must hold one (r, g, b) tuple (0 to 1) per color."
    if color is None:
        color = QtGui.QColor("white")
    if background is None:
        background = QtGui.QColor("black")

    font = QtGui.QFont()
    font.setPixelSize(text_fontsize)
    fm = QtGui.QFontMetrics(font)
    text_height = fm.height()

    # ticks as (fraction along the bar, text) pairs
    if n_ticks < 2 or max_value <= min_value:
        ticks = []
        text_width = 0
    else:
        values = np.linspace(min_value, max_value, n_ticks)
        ticks = [
            (
                (value - min_value) / (max_value - min_value),
                _format_tick(value),
            )
            for value in values
        ]
        text_width = max(fm.horizontalAdvance(text) for _, text in ticks)
    # space taken by the ticks next to the bar
    tick_extent = tick_length + tick_spacer
    tick_space = (
        tick_extent + (text_width if vertical else text_height) if ticks else 0
    )
    label_height = text_height + margin // 2 if label else 0

    # size and position of the bar; the outermost tick labels stick out
    # beyond the ends of the bar, hence the padding
    if vertical:
        pad = text_height // 2 if ticks else 0
        bar_x = margin
        bar_y = margin + label_height + pad
        width = margin + bar_width + tick_space + margin
        height = bar_y + bar_length + pad + margin
    else:
        pad = text_width // 2 if ticks else 0
        bar_x = margin + pad
        bar_y = margin + label_height
        width = bar_x + bar_length + pad + margin
        height = bar_y + bar_width + tick_space + margin
    if label:  # do not cut off the label
        width = max(width, fm.horizontalAdvance(label) + 2 * margin)

    return {
        "colors": colors_arr,
        "label": label,
        "vertical": vertical,
        "bar_length": bar_length,
        "bar_width": bar_width,
        "bar_x": bar_x,
        "bar_y": bar_y,
        "width": width,
        "height": height,
        "ticks": ticks,
        "tick_length": tick_length,
        "tick_extent": tick_extent,
        "text_width": text_width,
        "text_height": text_height,
        "font": font,
        "color": color,
        "background": background,
        "margin": margin,
    }


def _paint_colorbar(painter: QtGui.QPainter, layout: dict) -> None:
    """Paint a color bar laid out by ``_colorbar_layout`` onto any paint
    device (an image or an SVG generator).

    Parameters
    ----------
    painter : QPainter
        Painter active on the paint device, sized as the layout says.
    layout : dict
        As returned by ``_colorbar_layout``.
    """
    colors = layout["colors"]
    vertical = layout["vertical"]
    bar_x, bar_y = layout["bar_x"], layout["bar_y"]
    bar_length, bar_width = layout["bar_length"], layout["bar_width"]
    color = layout["color"]
    text_width, text_height = layout["text_width"], layout["text_height"]

    painter.setFont(layout["font"])
    painter.fillRect(
        0, 0, layout["width"], layout["height"], layout["background"]
    )

    # color bands; integer edges, so that neighboring bands neither
    # overlap nor leave gaps
    n_colors = len(colors)
    edges = np.round(np.linspace(0, bar_length, n_colors + 1)).astype(int)
    painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    for i in range(n_colors):
        rgb = colors[i]
        painter.setBrush(
            QtGui.QBrush(
                QtGui.QColor(
                    int(round(255 * rgb[0])),
                    int(round(255 * rgb[1])),
                    int(round(255 * rgb[2])),
                )
            )
        )
        thickness = int(edges[i + 1] - edges[i])
        if vertical:  # the first color is at the bottom
            y = bar_y + bar_length - int(edges[i + 1])
            painter.drawRect(bar_x, y, bar_width, thickness)
        else:
            x = bar_x + int(edges[i])
            painter.drawRect(x, bar_y, thickness, bar_width)

    # frame, ticks and text
    painter.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    painter.setPen(QtGui.QPen(color))
    if vertical:
        painter.drawRect(bar_x, bar_y, bar_width, bar_length)
    else:
        painter.drawRect(bar_x, bar_y, bar_length, bar_width)
    for fraction, text in layout["ticks"]:
        if vertical:
            y = int(round(bar_y + (1 - fraction) * bar_length))
            painter.drawLine(
                bar_x + bar_width,
                y,
                bar_x + bar_width + layout["tick_length"],
                y,
            )
            text_rect = QtCore.QRect(
                bar_x + bar_width + layout["tick_extent"],
                y - text_height // 2,
                text_width,
                text_height,
            )
            alignment = (
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        else:
            x = int(round(bar_x + fraction * bar_length))
            painter.drawLine(
                x,
                bar_y + bar_width,
                x,
                bar_y + bar_width + layout["tick_length"],
            )
            text_rect = QtCore.QRect(
                x - text_width // 2,
                bar_y + bar_width + layout["tick_extent"],
                text_width,
                text_height,
            )
            alignment = (
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignTop
            )
        painter.drawText(text_rect, alignment, text)
    if layout["label"]:
        margin = layout["margin"]
        painter.drawText(
            QtCore.QRect(
                margin, margin, layout["width"] - 2 * margin, text_height
            ),
            QtCore.Qt.AlignmentFlag.AlignHCenter
            | QtCore.Qt.AlignmentFlag.AlignTop,
            layout["label"],
        )


def colorbar_image(
    colors: list[tuple[float, float, float]] | lib.FloatArray2D,
    min_value: float,
    max_value: float,
    label: str = "",
    vertical: bool = True,
    bar_length: int = 400,
    bar_width: int = 40,
    n_ticks: int = 5,
    color: QtGui.QColor | None = None,  # default: white
    background: QtGui.QColor | None = None,  # default: black
    text_fontsize: int = 20,
    margin: int = 12,
    tick_length: int = 8,
    tick_spacer: int = 4,
) -> QtGui.QImage:
    """Draw a standalone color bar (LUT) of a rendered property.

    One band is drawn per color, i.e., the bar shows the discretized
    colors that localizations are rendered with when rendering by
    property (see ``get_colors_from_colormap`` and
    ``split_locs_by_property``), not the continuous colormap. The bar is
    annotated with the property values at ``n_ticks`` evenly spaced
    positions; the two ends of the bar correspond to `min_value` and
    `max_value`. The first color is drawn at the bottom (vertical bar)
    or at the left (horizontal bar).

    Meant to be saved next to an exported image, e.g., to annotate z
    color-coding in a figure. See ``colorbar_svg`` for the same bar as a
    vector graphic and ``save_colorbar`` to write either.

    Parameters
    ----------
    colors : list of tuples or lib.FloatArray2D
        Colors of the bands, one ``(r, g, b)`` tuple (values between 0
        and 1) per color, as used for rendering, see
        ``get_colors_from_colormap``.
    min_value : float
        Value of the rendered property at the start of the bar.
    max_value : float
        Value of the rendered property at the end of the bar.
    label : str, optional
        Text displayed above the bar, e.g., 'z (nm)'. Default is "",
        i.e., no label.
    vertical : bool, optional
        Whether the bar is drawn vertically (True) or horizontally
        (False). Default is True.
    bar_length : int, optional
        Length of the bar in display pixels. Default is 400.
    bar_width : int, optional
        Thickness of the bar in display pixels. Default is 40.
    n_ticks : int, optional
        Number of annotated positions along the bar. If less than 2, no
        ticks are drawn. Default is 5.
    color : QColor, optional
        Color of the frame, ticks and text. Default is white.
    background : QColor, optional
        Color of the background. Default is black. Pass a fully
        transparent QColor for a transparent background.
    text_fontsize : int, optional
        Font size of the label and tick text in display pixels. Default
        is 20.
    margin : int, optional
        Margin around the drawn elements in display pixels. Default is
        12.
    tick_length : int, optional
        Length of the tick marks in display pixels. Default is 8.
    tick_spacer : int, optional
        Spacing between the tick marks and the tick text in display
        pixels. Default is 4.

    Returns
    -------
    image : QImage
        Image with the drawn color bar.
    """
    layout = _colorbar_layout(
        colors=colors,
        min_value=min_value,
        max_value=max_value,
        label=label,
        vertical=vertical,
        bar_length=bar_length,
        bar_width=bar_width,
        n_ticks=n_ticks,
        color=color,
        background=background,
        text_fontsize=text_fontsize,
        margin=margin,
        tick_length=tick_length,
        tick_spacer=tick_spacer,
    )
    image = QtGui.QImage(
        layout["width"], layout["height"], QtGui.QImage.Format.Format_ARGB32
    )
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    _paint_colorbar(painter, layout)
    painter.end()
    return image


def colorbar_svg(path: str, **kwargs) -> None:
    """Write a color bar (LUT) of a rendered property to an SVG file.

    Unlike ``export_qimage_to_svg``, which embeds a rendered image, this
    draws the bar itself into the SVG: the bands, the frame, the ticks
    and the text stay vector objects that can be scaled and edited in
    figure software.

    Parameters
    ----------
    path : str
        Where to write the SVG.
    **kwargs
        The color bar to draw, see ``colorbar_image``.
    """
    layout = _colorbar_layout(**kwargs)
    generator = QtSvg.QSvgGenerator()
    generator.setFileName(path)
    generator.setSize(QtCore.QSize(layout["width"], layout["height"]))
    generator.setViewBox(QtCore.QRect(0, 0, layout["width"], layout["height"]))
    generator.setTitle(layout["label"] or "Color bar")
    generator.setDescription(f"Color bar generated by Picasso v{__version__}")
    painter = QtGui.QPainter(generator)
    _paint_colorbar(painter, layout)
    painter.end()


def save_colorbar(path: str, **kwargs) -> None:
    """Save a color bar (LUT) of a rendered property, in the format
    given by the extension of `path`.

    ``.svg`` gives a vector graphic (``colorbar_svg``), every other
    extension an image written by Qt (``colorbar_image``).

    Parameters
    ----------
    path : str
        Where to write the color bar; its extension selects the format.
    **kwargs
        The color bar to draw, see ``colorbar_image``.
    """
    if path.lower().endswith(".svg"):
        colorbar_svg(path, **kwargs)
    else:
        colorbar_image(**kwargs).save(path)


@adjust_viewport_decorator
def draw_minimap(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],  # cam. px
    max_viewport_size: tuple[float, float],  # in camera pixels,
    color_main: QtGui.QColor | None = None,  # default: yellow
    color_frame: QtGui.QColor | None = None,  # default: white
    length_minimap: int = 100,
    margin: tuple[int, int] = (20, 20),
) -> QtGui.QImage:
    """Draw a minimap showing the position of current viewport.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    viewport : tuple
        Current field of view in camera pixels, ((y_min, y_max), (x_min,
        x_max)).
    max_viewport_size : tuple
        Maximum viewport size in camera pixels, i.e., the acquired
        movie size (height, width).
    color_main, color_frame : QColor, optional
        Colors of the viewport and the minimap frame. Default is yellow
        and white, respectively.
    length_minimap : int, optional
        Length of the minimap in pixels. Default is 100.
    margin : tuple of int, optional
        Margins from the right and top edges in display pixels.
        Default is (20, 20).

    Returns
    -------
    image : QImage
        Image with the drawn minimap.
    """
    if color_main is None:
        color_main = QtGui.QColor("yellow")
    if color_frame is None:
        color_frame = QtGui.QColor("white")
    movie_height, movie_width = max_viewport_size
    height_minimap = int(movie_height / movie_width * length_minimap)
    # draw in the upper right corner, overview rectangle
    x = image.width() - length_minimap - margin[0]
    y = margin[1]
    painter = QtGui.QPainter(image)
    painter.setPen(color_frame)
    painter.drawRect(x, y, length_minimap, height_minimap)
    painter.setPen(color_main)
    length = int(viewport_width(viewport) / movie_width * length_minimap)
    length = max(5, length)
    height = int(viewport_height(viewport) / movie_height * height_minimap)
    height = max(5, height)
    x_vp = int(viewport[0][1] / movie_width * length_minimap)
    y_vp = int(viewport[0][0] / movie_height * height_minimap)
    painter.drawRect(x + x_vp, y + y_vp, length, height)
    return image


def draw_rotation(
    image: QtGui.QImage,
    ang: tuple[float, float, float] | Rotation,
    axis_length: int = 30,
    axis_center: tuple[int, int] = (50, -50),  # bottom left
) -> QtGui.QImage:
    """Draw rotation axes icon on the image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    ang : tuple of float or scipy.spatial.transform.Rotation
        Rotation of the localizations; either a scipy Rotation or a
        tuple of 3 rotation angles around the x, y and z axes in
        radians (legacy Euler convention, see ``rotation_matrix``).
    axis_length : int, optional
        Length of the rotation axes in display pixels. Default is 30.
    axis_center : tuple of int, optional
        Position of the rotation axes icon in display pixels, with
        origin in the top left corner. Negative values indicated
        counting from the bottom right corner. Default is (50, -50).

    Returns
    -------
    image : QImage
        Image with the drawn rotation axes icon.
    """
    painter = QtGui.QPainter(image)
    x = (
        axis_center[0]
        if axis_center[0] >= 0
        else image.width() + axis_center[0]
    )
    y = (
        axis_center[1]
        if axis_center[1] >= 0
        else image.height() + axis_center[1]
    )
    center = QtCore.QPoint(x, y)

    # set the ends of the x line
    xx = axis_length
    xy = 0
    xz = 0

    # set the ends of the y line
    yx = 0
    yy = axis_length
    yz = 0

    # set the ends of the z line
    zx = 0
    zy = 0
    zz = axis_length

    # rotate these points
    coordinates = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    R = to_rotation(ang)
    coordinates = R.apply(coordinates).astype(int)
    (xx, xy, xz) = coordinates[0]
    (yx, yy, yz) = coordinates[1]
    (zx, zy, zz) = coordinates[2]

    # translate the x and y coordinates of the end points towards
    # bottom right edge of the window
    xx += x
    xy += y
    yx += x
    yy += y
    zx += x
    zy += y

    # set the points at the ends of the lines
    point_x = QtCore.QPoint(xx, xy)
    point_y = QtCore.QPoint(yx, yy)
    point_z = QtCore.QPoint(zx, zy)
    line_x = QtCore.QLine(center, point_x)
    line_y = QtCore.QLine(center, point_y)
    line_z = QtCore.QLine(center, point_z)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(1, 0, 0, 1)))
    painter.drawLine(line_x)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 1, 1)))
    painter.drawLine(line_y)
    painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(0, 1, 0, 1)))
    painter.drawLine(line_z)
    return image


def draw_rotation_angles(
    image: QtGui.QImage,
    ang: tuple[float, float, float],
    color: QtGui.QColor | None = None,  # default: white
) -> QtGui.QImage:
    """Draw rotation angles (numbers in degrees) on the image.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations.
    ang : tuple of float
        Rotation angles (or rotation-vector components) around x, y,
        and z axes in radians, used only for the displayed text.
    color : QColor, optional
        Color of the text. Default is white.

    Returns
    -------
    image : QImage
        Image with the drawn rotation angles.
    """
    if color is None:
        color = QtGui.QColor("white")
    angx, angy, angz = [int(np.round(_ * 180 / np.pi, 0)) for _ in ang]
    text = f"{angx} {angy} {angz}"
    x = image.width() - len(text) * 8 - 10
    y = image.height() - 20
    painter = QtGui.QPainter(image)
    font = painter.font()
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(QtCore.QPoint(x, y), text)
    return image


def rgb_to_qimage(
    image: lib.IntArray3D, return_bgra: bool = False
) -> QtGui.QImage | tuple[QtGui.QImage, lib.IntArray3D]:
    """Convert a numpy array of shape (height, width, 3) with integer
    values between 0 and 255 to a QImage.

    Parameters
    ----------
    image : IntArray3D
        RGB image as a numpy array of shape (height, width, 3) with
        integer values between 0 and 255.
    return_bgra : bool, optional
        If True, return the BGRA numpy array instead of a QImage.
        Default is False.

    Returns
    -------
    qimage : QImage
        The converted QImage.
    bgra : IntArray3D
        The BGRA numpy array. Only returned if return_bgra is True.
    """
    bgra = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    bgra[:, :, 0] = image[:, :, 2]  # R -> B
    bgra[:, :, 1] = image[:, :, 1]  # G -> G
    bgra[:, :, 2] = image[:, :, 0]  # B -> R
    bgra[:, :, 3] = 255  # A -> 255 (opaque)
    Y, X = image.shape[:2]
    qimage = QtGui.QImage(bgra.data, X, Y, QtGui.QImage.Format.Format_RGB32)
    qimage = qimage.copy()  # make a deep copy to own the data DO NOT DELETE
    if return_bgra:
        return qimage, bgra
    return qimage


def optimal_scalebar_length(pixelsize: int | float, width: int | float) -> int:
    """Calculate optimal scale bar length in nm based on the image
    width.

    Parameters
    ----------
    pixelsize : int or float
        Camera pixel size in nm.
    width : int or float
        Image width in camera pixels.

    Returns
    -------
    scalebar : int
        Suggested scale bar length in nm.
    """
    width_nm = width * pixelsize
    optimal_scalebar = width_nm / 8
    # approximate to the nearest thousands, hundreds, tens or ones
    if optimal_scalebar > 10_000:
        scalebar = 10_000
    elif optimal_scalebar > 1_000:
        scalebar = int(1_000 * round(optimal_scalebar / 1_000))
    elif optimal_scalebar > 100:
        scalebar = int(100 * round(optimal_scalebar / 100))
    elif optimal_scalebar > 10:
        scalebar = int(10 * round(optimal_scalebar / 10))
    else:
        scalebar = int(round(optimal_scalebar))
    return scalebar
