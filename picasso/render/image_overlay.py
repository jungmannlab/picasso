"""
picasso.render.image_overlay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overlay of an image (e.g., a widefield or brightfield PNG or TIFF) on
rendered localizations: loading, placement in camera pixels and drawing.

Placement is anchored to the camera chip, not to the rendered field of
view. A localization at ``x = j`` sits at the center of camera pixel
``j``, so the chip spans ``[-0.5, width - 0.5)`` in localization
coordinates. An image with the movie's size, stretched to the chip,
thus puts each of its pixels exactly onto the camera pixel that
recorded it.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from .. import lib
from .geometry import viewport_height, viewport_width

if TYPE_CHECKING:
    from PyQt6 import QtGui, QtCore
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")
    QtCore = lib._LazyQtModule("PyQt6.QtCore")


# how the image is scaled onto the camera chip
SCALING_MODES = (
    "Fit to camera (keep aspect ratio)",
    "Stretch to camera",
    "Image pixel size",
)
# smallest difference (0-255) between the empty and the full color of a
# rendering in a color channel that ``composite_behind`` uses
MIN_COLOR_SPAN = 16
BLEND_MODES = (
    "Additive",
    "Over localizations",
    "Behind localizations",
    "Multiply",
)
# colors that a grayscale image is displayed in, as RGB in [0, 1]
GRAYSCALE_COLORS = {
    "Gray": (1.0, 1.0, 1.0),
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
}


# file extensions that ``load_overlay_image`` reads
IMAGE_EXTENSIONS = (".png", ".tif", ".tiff")


def load_overlay_image(
    path: str, page: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a PNG or TIFF image to overlay on rendered localizations.

    RGB images whose three channels are identical (e.g., a grayscale
    image saved with a palette) are returned as grayscale, so that
    their contrast and color can be adjusted.

    Parameters
    ----------
    path : str
        Path to the image, see ``IMAGE_EXTENSIONS``.
    page : int, optional
        Page of a multi-page TIFF (e.g., a frame of a movie) to load,
        see ``count_image_pages``. Ignored for PNG. Default is 0.

    Returns
    -------
    data : np.ndarray
        Grayscale image of shape ``(height, width)`` in the file's data
        type (e.g., uint8, uint16 or float32) or RGB image of shape
        ``(height, width, 3)`` (uint8).
    alpha : np.ndarray or None
        Alpha channel of shape ``(height, width)`` (uint8), or None if
        the image is opaque.

    Raises
    ------
    ValueError
        If the file extension is not supported or the image is neither
        grayscale nor RGB(A).
    """
    extension = os.path.splitext(path)[1].lower()
    if extension in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            separate = (
                tif.pages[page].planarconfig == tifffile.PLANARCONFIG.SEPARATE
            )
            image = tif.asarray(key=page)
        if separate and image.ndim == 3:  # samples first, (3, H, W)
            image = np.moveaxis(image, 0, -1)
    elif extension == ".png":
        import imageio.v3 as iio

        image = iio.imread(path)
    else:
        raise ValueError(
            f"Unsupported file extension {extension!r}; expected one of "
            f"{IMAGE_EXTENSIONS}."
        )
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    alpha = None
    if image.ndim == 3 and image.shape[2] in (2, 4):  # gray/RGB + alpha
        alpha = _to_uint8(image[:, :, -1])
        image = image[:, :, :-1]
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 3:
        if image.shape[2] != 3:
            raise ValueError(
                f"Unsupported image shape {image.shape}; expected a "
                "grayscale or an RGB(A) image."
            )
        if np.array_equal(image[:, :, 0], image[:, :, 1]) and (
            np.array_equal(image[:, :, 0], image[:, :, 2])
        ):
            image = image[:, :, 0]
        else:
            image = _to_uint8(image)
    elif image.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {image.shape}; expected a "
            "grayscale or an RGB(A) image."
        )
    if alpha is not None and np.all(alpha == 255):
        alpha = None
    return np.ascontiguousarray(image), alpha


def count_image_pages(path: str) -> int:
    """Number of pages of a TIFF image; 1 for a PNG.

    Parameters
    ----------
    path : str
        Path to the image.

    Returns
    -------
    n_pages : int
        Number of pages (e.g., frames of a movie).
    """
    if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
        return 1
    import tifffile

    with tifffile.TiffFile(path) as tif:
        return len(tif.pages)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8: integers by their data type's range
    (e.g., uint16 values are divided by 257), floats by 255 if they lie
    in [0, 1] and by their maximum otherwise."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        top = np.iinfo(image.dtype).max
        scaled = np.clip(image, 0, None).astype(np.float64) * (255 / top)
    else:
        top = float(np.nanmax(image)) if image.size else 1.0
        factor = 255.0 if top <= 1.0 else 255.0 / top
        scaled = np.nan_to_num(image.astype(np.float64)) * factor
    return np.round(np.clip(scaled, 0, 255)).astype(np.uint8)


def overlay_extent(
    image_shape: tuple[int, ...],
    movie_size: tuple[float, float],
    mode: str,
    scale: float = 1.0,
    shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    """Find where an image overlay lies in camera pixels.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the image, ``(height, width, ...)``.
    movie_size : tuple of float
        Height and width of the camera chip in camera pixels (the
        ``"Height"`` and ``"Width"`` metadata of the localizations).
    mode : str
        One of ``SCALING_MODES``. ``"Fit to camera (keep aspect
        ratio)"`` scales the image uniformly to the largest size that
        fits on the chip and centers it; ``"Stretch to camera"`` scales
        width and height independently to cover the chip;
        ``"Image pixel size"`` scales each image pixel to ``scale``
        camera pixels and aligns the top left corners of the image and
        the chip.
    scale : float, optional
        Size of one image pixel in camera pixels (image pixel size
        divided by camera pixel size). Only used by ``"Image pixel
        size"``. Default is 1.0.
    shift : tuple of float, optional
        Shift ``(x, y)`` of the image in camera pixels, added after
        scaling. Default is ``(0.0, 0.0)``.

    Returns
    -------
    extent : tuple of float
        ``(x, y, width, height)`` of the image in camera pixels, in the
        coordinates of the localizations.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``SCALING_MODES``.
    """
    image_height, image_width = image_shape[:2]
    movie_height, movie_width = movie_size
    x0 = y0 = -0.5  # top left corner of the chip, see the module doc
    if mode == "Fit to camera (keep aspect ratio)":
        factor = min(movie_width / image_width, movie_height / image_height)
        width = image_width * factor
        height = image_height * factor
        x0 += (movie_width - width) / 2
        y0 += (movie_height - height) / 2
    elif mode == "Stretch to camera":
        width = movie_width
        height = movie_height
    elif mode == "Image pixel size":
        width = image_width * scale
        height = image_height * scale
    else:
        raise ValueError(
            f"Unknown scaling mode {mode!r}; choose one of {SCALING_MODES}."
        )
    return (x0 + shift[0], y0 + shift[1], width, height)


def overlay_to_qimage(
    data: np.ndarray,
    alpha: np.ndarray | None = None,
    contrast: tuple[float, float] | None = None,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> QtGui.QImage:
    """Convert an overlay image to a QImage ready for drawing.

    Parameters
    ----------
    data : np.ndarray
        Grayscale ``(height, width)`` image of any real data type or
        RGB ``(height, width, 3)`` uint8 image, see
        ``load_overlay_image``.
    alpha : np.ndarray, optional
        Alpha channel ``(height, width)`` (uint8). Default is None
        (opaque).
    contrast : tuple of float, optional
        Values ``(min, max)`` mapped to black and to ``color``; only
        used for grayscale images. If None, the image's minimum and
        maximum are taken. Default is None.
    color : tuple of float, optional
        RGB color in [0, 1] that a grayscale image is displayed in.
        Default is white (grayscale).

    Returns
    -------
    qimage : QImage
        ARGB32 image of the same size as ``data``.
    """
    if data.ndim == 2:
        if contrast is None:
            contrast = (float(data.min()), float(data.max()))
        lo, hi = contrast
        if hi > lo:
            scaled = (data.astype(np.float32) - lo) / (hi - lo)
            np.clip(scaled, 0.0, 1.0, out=scaled)
            np.nan_to_num(scaled, copy=False)  # NaN pixels of floats
        else:  # e.g., a constant image: threshold at the limit
            scaled = (data >= hi).astype(np.float32)
        rgb = scaled[:, :, None] * (255 * np.asarray(color, np.float32))
        rgb = np.round(rgb).astype(np.uint8)
    else:
        rgb = data
    height, width = rgb.shape[:2]
    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[:, :, 0] = rgb[:, :, 2]
    bgra[:, :, 1] = rgb[:, :, 1]
    bgra[:, :, 2] = rgb[:, :, 0]
    bgra[:, :, 3] = 255 if alpha is None else alpha
    qimage = QtGui.QImage(
        bgra.data, width, height, QtGui.QImage.Format.Format_ARGB32
    )
    return qimage.copy()  # own the data, bgra is freed on return


def draw_image_overlay(
    image: QtGui.QImage,
    viewport: tuple[tuple[float, float], tuple[float, float]],
    overlay: QtGui.QImage,
    extent: tuple[float, float, float, float],
    opacity: float = 1.0,
    blend: str = "Additive",
    color_range: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (0, 0, 0),
        (255, 255, 255),
    ),
) -> QtGui.QImage:
    """Draw an image overlay onto rendered localizations.

    The image pixels are drawn as sharp squares while each covers at
    least one display pixel and smoothed when the view is zoomed out
    further.

    Parameters
    ----------
    image : QImage
        Image containing rendered localizations; drawn on in place if
        it is in the RGB32 or ARGB32 format.
    viewport : tuple
        Field of view shown in ``image``, ``((y_min, x_min), (y_max,
        x_max))`` in camera pixels.
    overlay : QImage
        The overlay, see ``overlay_to_qimage``.
    extent : tuple of float
        ``(x, y, width, height)`` of the overlay in camera pixels, see
        ``overlay_extent``.
    opacity : float, optional
        Opacity of the overlay in [0, 1]. Default is 1.0.
    blend : str, optional
        One of ``BLEND_MODES``. ``"Additive"`` adds overlay and
        localizations (as when merging fluorescence channels, suited to
        a black background). ``"Over localizations"`` paints the
        overlay over the localizations. ``"Behind localizations"``
        paints it behind them: pixels without localizations show the
        overlay, which the localizations cover the more, the brighter
        they are, see ``composite_behind``. ``"Multiply"`` multiplies
        overlay and localizations (suited to a white background).
        Default is ``"Additive"``.
    color_range : tuple, optional
        RGB colors (0-255) of the pixels without localizations and of
        the pixels at the maximum contrast, see
        ``picasso.render.color_range``. Only used by ``"Behind
        localizations"``. Default is black and white.

    Returns
    -------
    image : QImage
        Image with the drawn overlay.

    Raises
    ------
    ValueError
        If ``blend`` is not one of ``BLEND_MODES``.
    """
    if blend not in BLEND_MODES:
        raise ValueError(
            f"Unknown blend mode {blend!r}; choose one of {BLEND_MODES}."
        )
    (y_min, x_min), (y_max, x_max) = viewport
    v_width = viewport_width(viewport)
    v_height = viewport_height(viewport)
    if v_width <= 0 or v_height <= 0 or opacity <= 0:
        return image
    x0, y0, width, height = extent
    # the part of the overlay inside the viewport, in camera pixels
    vx0, vx1 = max(x0, x_min), min(x0 + width, x_max)
    vy0, vy1 = max(y0, y_min), min(y0 + height, y_max)
    if vx1 <= vx0 or vy1 <= vy0:
        return image
    px_x = image.width() / v_width  # display pixels per camera pixel
    px_y = image.height() / v_height
    ov_x = overlay.width() / width  # overlay pixels per camera pixel
    ov_y = overlay.height() / height
    source = QtCore.QRectF(
        (vx0 - x0) * ov_x,
        (vy0 - y0) * ov_y,
        (vx1 - vx0) * ov_x,
        (vy1 - vy0) * ov_y,
    )
    target = QtCore.QRectF(
        (vx0 - x_min) * px_x,
        (vy0 - y_min) * px_y,
        (vx1 - vx0) * px_x,
        (vy1 - vy0) * px_y,
    )
    smooth = min(px_x / ov_x, px_y / ov_y) < 1.0
    modes = QtGui.QPainter.CompositionMode
    if blend == "Behind localizations":
        # the overlay over an empty rendering, then the localizations
        # over that
        image = _as_rgb32(image)
        empty = color_range[0]
        background = QtGui.QImage(
            image.width(), image.height(), QtGui.QImage.Format.Format_RGB32
        )
        background.fill(QtGui.QColor(*[int(_) for _ in empty]))
        _paint(
            background,
            overlay,
            target,
            source,
            modes.CompositionMode_SourceOver,
            opacity,
            smooth,
        )
        rect = target.toAlignedRect().intersected(image.rect())
        if rect.isEmpty():
            return image
        rows = slice(rect.top(), rect.bottom() + 1)
        cols = slice(rect.left(), rect.right() + 1)
        locs_bgra = _bgra_view(image)[rows, cols]
        bg_bgra = _bgra_view(background)[rows, cols]
        # BGRA byte order -> RGB
        rgb = composite_behind(
            locs_bgra[:, :, 2::-1], bg_bgra[:, :, 2::-1], *color_range
        )
        locs_bgra[:, :, 2::-1] = rgb
        return image
    composition = {
        "Over localizations": modes.CompositionMode_SourceOver,
        "Additive": modes.CompositionMode_Plus,
        "Multiply": modes.CompositionMode_Multiply,
    }[blend]
    _paint(image, overlay, target, source, composition, opacity, smooth)
    return image


def composite_behind(
    locs_rgb: np.ndarray,
    background_rgb: np.ndarray,
    empty_color: tuple[int, int, int],
    full_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Composite rendered localizations over a new background.

    A rendered pixel is treated as the localizations' color with an
    opacity ``a`` over ``empty_color``: ``a`` is the pixel's largest
    deviation from ``empty_color`` across the color channels, relative
    to the deviation of ``full_color`` in that channel. Pixels without
    localizations (``a = 0``) thus show the new background, pixels at
    the maximum contrast (``a = 1``) are unchanged, and dim ones are
    see-through. This inverts the renderer's own compositing over a
    background color.

    Parameters
    ----------
    locs_rgb : np.ndarray
        Rendered localizations, uint8 of shape ``(height, width, 3)``.
    background_rgb : np.ndarray
        The new background (e.g., the overlay painted over
        ``empty_color``), of the same shape.
    empty_color : tuple of int
        RGB color (0-255) of the pixels without localizations.
    full_color : tuple of int, optional
        RGB color (0-255) of the pixels at the maximum contrast. Color
        channels in which it differs from ``empty_color`` by less than
        ``MIN_COLOR_SPAN`` are not used, since they cannot resolve the
        opacity; if no channel remains, the largest deviation possible
        in each channel is used instead. Default is white.

    Returns
    -------
    rgb : np.ndarray
        uint8 array of shape ``(height, width, 3)``.

    See Also
    --------
    picasso.render.color_range : ``empty_color`` and ``full_color`` of
        a rendering.
    """
    locs = locs_rgb.astype(np.float32)
    empty = np.asarray(empty_color, dtype=np.float32)
    span = np.abs(np.asarray(full_color, dtype=np.float32) - empty)
    used = span >= MIN_COLOR_SPAN
    if not used.any():
        span = np.maximum(empty, 255 - empty)
        used = span > 0
    deviation = np.abs(locs[:, :, used] - empty[used]) / span[used]
    alpha = np.max(deviation, axis=2)
    np.clip(alpha, 0.0, 1.0, out=alpha)
    rgb = locs + (1.0 - alpha[:, :, None]) * (
        background_rgb.astype(np.float32) - empty
    )
    return np.round(np.clip(rgb, 0, 255)).astype(np.uint8)


def _paint(
    image: QtGui.QImage,
    overlay: QtGui.QImage,
    target: QtCore.QRectF,
    source: QtCore.QRectF,
    composition: QtGui.QPainter.CompositionMode,
    opacity: float,
    smooth: bool,
) -> None:
    """Paint the ``source`` part of ``overlay`` into ``target`` of
    ``image``."""
    painter = QtGui.QPainter(image)
    painter.setCompositionMode(composition)
    painter.setOpacity(opacity)
    painter.setRenderHint(
        QtGui.QPainter.RenderHint.SmoothPixmapTransform, smooth
    )
    painter.drawImage(target, overlay, source)
    painter.end()


def _as_rgb32(image: QtGui.QImage) -> QtGui.QImage:
    """``image`` itself if it is RGB32 or ARGB32, else an RGB32 copy."""
    formats = (
        QtGui.QImage.Format.Format_RGB32,
        QtGui.QImage.Format.Format_ARGB32,
    )
    if image.format() in formats:
        return image
    return image.convertToFormat(QtGui.QImage.Format.Format_RGB32)


def _bgra_view(image: QtGui.QImage) -> np.ndarray:
    """Writable ``(height, width, 4)`` uint8 view of an RGB32 or ARGB32
    image, in the BGRA byte order of little-endian machines."""
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    rows = np.frombuffer(ptr, dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )
    return rows[:, : image.width() * 4].reshape(
        image.height(), image.width(), 4
    )
