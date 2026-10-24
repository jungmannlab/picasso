"""Build the Picasso app icon (picasso.icns, picasso.ico) for the
one-click installers from the square logo (picasso-logo-square.svg, the
helix without the lettering).

The line-art logo sits on a white rounded tile (macOS Big Sur grid:
824/1024 tile, ~22% corner radius) so it stays visible on dark docks and
taskbars. Each size is rendered from the vector source with its own
stroke width, so lines never drop below ~1.4 px.

Usage (macOS, needs ``iconutil``), from this folder::

    python make_picasso_icon.py [logo.svg] [out_dir]

which defaults to ``picasso-logo-square.svg`` and this folder.
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtCore, QtGui, QtSvg  # noqa: E402

TILE = 824 / 1024  # tile side as a fraction of the icon
RADIUS = 0.2237  # tile corner radius as a fraction of the tile
FILL = 0.66  # logo extent as a fraction of the tile
MARGIN = 0.08  # minimum clearance between the logo and the tile edge, as a
# fraction of the tile; the helix ends reach into the rounded corners
STROKE_BOOST = 1.6  # stroke width relative to the source
MIN_STROKE = 1.4  # px in the output

_STROKE_WIDTH = re.compile(r'(stroke-width\s*[:=]\s*"?)([\d.]+)')


def with_stroke(svg, width):
    """Return ``svg`` with every stroke width set to ``width``."""
    return _STROKE_WIDTH.sub(lambda m: f"{m.group(1)}{width:.6g}", svg)


def renderer(svg):
    r = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    if not r.isValid():
        raise ValueError("cannot parse the SVG")
    return r


def alpha(image):
    """Alpha channel of an ARGB32 QImage as a uint8 array."""
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    rows = np.frombuffer(ptr, np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )
    return (
        rows[:, : 4 * image.width()]
        .reshape(image.height(), image.width(), 4)[..., 3]
        .copy()
    )


def new_image(size):
    image = QtGui.QImage(
        size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied
    )
    image.fill(QtCore.Qt.GlobalColor.transparent)
    return image


def paint_tile(image, tile):
    """Fill a rounded square of side ``tile`` centered in ``image``."""
    off = (image.width() - tile) / 2
    path = QtGui.QPainterPath()
    path.addRoundedRect(
        QtCore.QRectF(off, off, tile, tile), RADIUS * tile, RADIUS * tile
    )
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.fillPath(path, QtCore.Qt.GlobalColor.white)
    painter.end()


def content_bounds(svg):
    """Bounds of the drawn line (center line), in SVG user units."""
    view = renderer(svg).viewBoxF()
    res = 4096 / max(view.width(), view.height())  # px per unit
    image = QtGui.QImage(
        round(view.width() * res),
        round(view.height() * res),
        QtGui.QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    renderer(with_stroke(svg, 2 / res)).render(painter)
    painter.end()
    ys, xs = np.nonzero(alpha(image))
    # the 2 px line reaches 1 px past its center line
    return QtCore.QRectF(
        view.x() + (xs.min() + 1) / res,
        view.y() + (ys.min() + 1) / res,
        (xs.max() - xs.min() - 1) / res,
        (ys.max() - ys.min() - 1) / res,
    )


def render(svg, bounds, stroke_src, size):
    tile = TILE * size
    scale = FILL * tile / max(bounds.width(), bounds.height())  # px/unit
    stroke = max(stroke_src * scale * STROKE_BOOST, MIN_STROKE) / scale
    logo = renderer(with_stroke(svg, stroke))
    # view the whole icon, centered on the logo
    half = size / 2 / scale
    center = bounds.center()
    logo.setViewBox(
        QtCore.QRectF(center.x() - half, center.y() - half, 2 * half, 2 * half)
    )

    glyph = new_image(size)
    painter = QtGui.QPainter(glyph)
    logo.render(painter)
    painter.end()
    # the logo must stay inside the tile shrunk by MARGIN on every side
    safe = new_image(size)
    paint_tile(safe, (1 - 2 * MARGIN) * tile)
    if (alpha(glyph) > alpha(safe).astype(int) + 32).any():
        raise ValueError(
            f"the logo comes closer than MARGIN to the tile edge at {size} "
            "px; lower FILL"
        )

    icon = new_image(size)
    paint_tile(icon, tile)
    painter = QtGui.QPainter(icon)
    painter.drawImage(0, 0, glyph)
    painter.end()
    return icon


def main(logo_path, out_dir):
    QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(sys.argv[:1])
    with open(logo_path, encoding="utf-8") as f:
        svg = f.read()
    widths = {float(m.group(2)) for m in _STROKE_WIDTH.finditer(svg)}
    if len(widths) != 1:
        raise ValueError(f"expected one stroke width, found {widths}")
    stroke_src = widths.pop()
    bounds = content_bounds(svg)

    sizes = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
    with tempfile.TemporaryDirectory() as tmp:
        pngs = {}
        for s in sizes:
            pngs[s] = os.path.join(tmp, f"{s}.png")
            render(svg, bounds, stroke_src, s).save(pngs[s])

        iconset = os.path.join(tmp, "picasso.iconset")
        os.mkdir(iconset)
        for s in [16, 32, 128, 256, 512]:
            os.link(pngs[s], os.path.join(iconset, f"icon_{s}x{s}.png"))
            os.link(pngs[2 * s], os.path.join(iconset, f"icon_{s}x{s}@2x.png"))
        subprocess.run(
            [
                "iconutil",
                "-c",
                "icns",
                iconset,
                "-o",
                os.path.join(out_dir, "picasso.icns"),
            ],
            check=True,
        )

        ico_sizes = [16, 24, 32, 48, 64, 128, 256]
        imgs = {s: Image.open(pngs[s]) for s in ico_sizes}
        imgs[256].save(
            os.path.join(out_dir, "picasso.ico"),
            format="ICO",
            sizes=[(s, s) for s in ico_sizes],
            append_images=[imgs[s] for s in ico_sizes[:-1]],
        )


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    args = sys.argv[1:]
    logo_path = (
        args[0] if args else os.path.join(here, "picasso-logo-square.svg")
    )
    out_dir = args[1] if len(args) > 1 else here
    main(logo_path, out_dir)
