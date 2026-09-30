"""Overlay of a PNG image (grayscale or RGB) on rendered localizations.

The image is placed on the camera chip described by the localizations'
metadata; a localization at ``x = j`` sits at the center of camera
pixel ``j``, so the chip spans ``[-0.5, width - 0.5)``. These tests
cover the placement for every scaling mode, loading of the PNG
and TIFF
flavors, drawing, and the wiring into the Render window (display,
exports and their metadata).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image
from PyQt6 import QtGui

from picasso import io, render
from picasso.gui import render as gui_render

WIDTH, HEIGHT = 32.0, 16.0
PIXELSIZE = 130.0


def _qimage_rgb(qimage: QtGui.QImage) -> np.ndarray:
    """RGB array ``(height, width, 3)`` of a QImage."""
    qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    ptr = qimage.constBits()
    ptr.setsize(qimage.sizeInBytes())
    rows = np.frombuffer(ptr, np.uint8).reshape(
        qimage.height(), qimage.bytesPerLine()
    )
    return (
        rows[:, : qimage.width() * 3]
        .reshape(qimage.height(), qimage.width(), 3)
        .copy()  # the converted QImage is freed on return
    )


# --- placement --------------------------------------------------------


def test_stretch_covers_the_chip():
    extent = render.overlay_extent(
        (8, 64), (HEIGHT, WIDTH), "Stretch to camera"
    )
    assert extent == (-0.5, -0.5, WIDTH, HEIGHT)


def test_fit_keeps_aspect_ratio_and_centers():
    # 64 x 64 image on a 32 x 16 chip: limited by the height
    x, y, w, h = render.overlay_extent(
        (64, 64), (HEIGHT, WIDTH), "Fit to camera (keep aspect ratio)"
    )
    assert (w, h) == (HEIGHT, HEIGHT)
    assert y == -0.5
    assert x == pytest.approx(-0.5 + (WIDTH - HEIGHT) / 2)


def test_fit_equals_stretch_for_matching_sizes():
    shape = (int(HEIGHT), int(WIDTH))
    fit = render.overlay_extent(
        shape, (HEIGHT, WIDTH), "Fit to camera (keep aspect ratio)"
    )
    stretch = render.overlay_extent(
        shape, (HEIGHT, WIDTH), "Stretch to camera"
    )
    assert fit == stretch


def test_pixel_size_and_shift():
    x, y, w, h = render.overlay_extent(
        (10, 20),
        (HEIGHT, WIDTH),
        "Image pixel size",
        scale=0.5,
        shift=(1.25, -2.0),
    )
    assert (x, y, w, h) == (0.75, -2.5, 10.0, 5.0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        render.overlay_extent((4, 4), (4, 4), "Nope")


# --- loading ----------------------------------------------------------


@pytest.fixture
def gray8():
    return (np.arange(16 * 32) % 256).astype(np.uint8).reshape(16, 32)


def test_load_grayscale_flavors(tmp_path, gray8):
    for mode in ("L", "P", "LA"):
        path = str(tmp_path / f"{mode}.png")
        Image.fromarray(gray8).convert(mode).save(path)
        data, alpha = render.load_overlay_image(path)
        assert data.shape == gray8.shape, mode
        np.testing.assert_array_equal(data, gray8)
        assert alpha is None  # LA with opaque alpha counts as opaque


def test_load_16_bit_grayscale(tmp_path, gray8):
    path = str(tmp_path / "i16.png")
    Image.fromarray(gray8.astype(np.uint16) * 200).save(path)
    data, _ = render.load_overlay_image(path)
    assert data.dtype == np.uint16
    assert data.max() == 255 * 200


def test_load_rgba(tmp_path, gray8):
    rgba = np.dstack([gray8, 255 - gray8, gray8 // 2, gray8])
    path = str(tmp_path / "rgba.png")
    Image.fromarray(rgba).save(path)
    data, alpha = render.load_overlay_image(path)
    np.testing.assert_array_equal(data, rgba[:, :, :3])
    np.testing.assert_array_equal(alpha, gray8)


def test_load_tiff_flavors(tmp_path, gray8):
    for dtype in (np.uint16, np.int16, np.float32):
        path = str(tmp_path / f"{np.dtype(dtype).name}.tif")
        tifffile.imwrite(path, gray8.astype(dtype) - 3)
        data, alpha = render.load_overlay_image(path)
        assert data.dtype == dtype  # grayscale keeps its data type
        np.testing.assert_array_equal(data, gray8.astype(dtype) - 3)
        assert alpha is None


def test_load_rgb_tiff_contiguous_and_planar(tmp_path, gray8):
    rgb = np.dstack([gray8, 255 - gray8, gray8 // 2])
    contig = str(tmp_path / "contig.tiff")
    planar = str(tmp_path / "planar.tif")
    tifffile.imwrite(contig, rgb, photometric="rgb")
    tifffile.imwrite(
        planar,
        np.moveaxis(rgb, -1, 0),
        photometric="rgb",
        planarconfig="separate",
    )
    for path in (contig, planar):
        data, _ = render.load_overlay_image(path)
        np.testing.assert_array_equal(data, rgb)


def test_rgb_16_bit_tiff_is_scaled_to_8_bit(tmp_path):
    rgb = np.zeros((4, 4, 3), np.uint16)
    rgb[..., 0] = 65535
    rgb[..., 1] = 257 * 100
    path = str(tmp_path / "rgb16.tif")
    tifffile.imwrite(path, rgb, photometric="rgb")
    data, _ = render.load_overlay_image(path)
    assert data.dtype == np.uint8
    assert tuple(data[0, 0]) == (255, 100, 0)


def test_multi_page_tiff(tmp_path, gray8):
    path = str(tmp_path / "movie.tif")
    tifffile.imwrite(
        path, np.stack([gray8, gray8 + 1, gray8 + 2]), imagej=True
    )
    assert render.count_image_pages(path) == 3
    data, _ = render.load_overlay_image(path, page=2)
    np.testing.assert_array_equal(data, gray8 + 2)


def test_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError):
        render.load_overlay_image(str(tmp_path / "image.jpg"))


# --- drawing ----------------------------------------------------------


def _black(width: int, height: int) -> QtGui.QImage:
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(0, 0, 0))
    return image


def test_draw_maps_image_pixels_onto_camera_pixels(qt_offscreen):
    """A 2 x 2 image stretched to a 2 x 2 chip: image pixel (i, j)
    covers localization coordinates [j - 0.5, j + 0.5)."""
    data = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    overlay = render.overlay_to_qimage(data)
    extent = render.overlay_extent((2, 2), (2, 2), "Stretch to camera")
    # the view shows [-0.5, 1.5) at 10 display px per camera px
    viewport = ((-0.5, -0.5), (1.5, 1.5))
    image = render.draw_image_overlay(
        _black(20, 20), viewport, overlay, extent
    )
    rgb = _qimage_rgb(image)
    assert (rgb[:10, :10] == 255).all()
    assert (rgb[10:, :] == 0).all() and (rgb[:, 10:] == 0).all()


def test_draw_opacity_color_and_blending(qt_offscreen):
    data = np.full((4, 4), 200, dtype=np.uint8)
    overlay = render.overlay_to_qimage(
        data, contrast=(0, 200), color=(1.0, 0.0, 0.0)
    )
    extent = (-0.5, -0.5, 4.0, 4.0)
    viewport = ((-0.5, -0.5), (3.5, 3.5))
    rgb = _qimage_rgb(
        render.draw_image_overlay(
            _black(8, 8), viewport, overlay, extent, opacity=0.5
        )
    )
    assert abs(int(rgb[4, 4, 0]) - 128) <= 3  # Qt quantizes opacity
    assert rgb[4, 4, 1] == rgb[4, 4, 2] == 0

    base = _black(8, 8)
    base.fill(QtGui.QColor(0, 100, 0))
    rgb = _qimage_rgb(
        render.draw_image_overlay(
            base, viewport, overlay, extent, blend="Additive"
        )
    )
    assert tuple(rgb[4, 4]) == (255, 100, 0)

    base.fill(QtGui.QColor(255, 255, 255))
    rgb = _qimage_rgb(
        render.draw_image_overlay(
            base, viewport, overlay, extent, blend="Multiply"
        )
    )
    assert tuple(rgb[4, 4]) == (255, 0, 0)


MAGMA = ((0, 0, 4), (252, 253, 191))  # empty and full colors


def test_composite_behind():
    empty, full = MAGMA
    locs = np.array([[empty, full, (100, 0, 4)]], dtype=np.uint8)
    background = np.full((1, 3, 3), 80, dtype=np.uint8)
    rgb = render.composite_behind(locs, background, empty, full)
    assert tuple(rgb[0, 0]) == (80, 80, 80)  # the background shows
    assert tuple(rgb[0, 1]) == full  # opaque at the maximum contrast
    a = 100 / 252  # red is the channel that deviates most
    expected = np.round(
        np.array([100, 0, 4]) + (1 - a) * (80 - np.array(empty))
    )
    np.testing.assert_array_equal(rgb[0, 2], expected)


def test_composite_behind_ignores_flat_channels():
    """Inverted rendering of a red channel: only red changes with the
    density, green and blue stay at 255 and cannot give the opacity."""
    empty, full = (255, 255, 255), (0, 255, 255)
    locs = np.array([[empty, full, (128, 255, 255)]], dtype=np.uint8)
    background = np.full((1, 3, 3), 100, dtype=np.uint8)
    rgb = render.composite_behind(locs, background, empty, full)
    assert tuple(rgb[0, 0]) == (100, 100, 100)
    assert tuple(rgb[0, 1]) == full
    a = 127 / 255
    expected = np.round(np.array([128, 255, 255]) + (1 - a) * (100 - 255))
    np.testing.assert_array_equal(rgb[0, 2], expected)


def test_color_range_follows_the_renderer():
    luts = [render.solid_to_lut((1, 0, 0)), render.solid_to_lut((0, 1, 0))]
    empty, full = render.color_range(None)
    assert (tuple(empty), tuple(full)) == MAGMA
    empty, full = render.color_range(None, invert_colors=True)
    assert tuple(empty) == (255, 255, 251)
    assert tuple(full) == (3, 2, 64)
    empty, full = render.color_range(2, luts)
    assert (tuple(empty), tuple(full)) == ((0, 0, 0), (255, 255, 0))
    empty, full = render.color_range(2, luts, background_color=(0.5, 0.5, 0.5))
    assert tuple(empty) == (128, 128, 128)
    assert tuple(full) == (255, 255, 0)


def test_draw_behind_localizations(qt_offscreen):
    image = _black(4, 4)
    image.setPixelColor(0, 0, QtGui.QColor(255, 255, 255))  # a loc
    overlay = render.overlay_to_qimage(np.full((4, 4), 255, np.uint8))
    rgb = _qimage_rgb(
        render.draw_image_overlay(
            image,
            ((-0.5, -0.5), (3.5, 3.5)),
            overlay,
            (-0.5, -0.5, 4.0, 4.0),
            opacity=0.5,
            blend="Behind localizations",
        )
    )
    assert tuple(rgb[0, 0]) == (255, 255, 255)
    assert abs(int(rgb[2, 2, 0]) - 128) <= 3  # Qt quantizes opacity


def test_unknown_blend_raises(qt_offscreen):
    overlay = render.overlay_to_qimage(np.zeros((2, 2), np.uint8))
    with pytest.raises(ValueError):
        render.draw_image_overlay(
            _black(2, 2), ((0, 0), (2, 2)), overlay, (0, 0, 2, 2), blend="?"
        )


def test_draw_outside_the_view_is_a_noop(qt_offscreen):
    overlay = render.overlay_to_qimage(np.full((2, 2), 255, np.uint8))
    image = render.draw_image_overlay(
        _black(10, 10), ((0, 0), (5, 5)), overlay, (10.0, 10.0, 2.0, 2.0)
    )
    assert (_qimage_rgb(image) == 0).all()


# --- Render window ----------------------------------------------------


@pytest.fixture(autouse=True)
def _settings_file(tmp_path, monkeypatch):
    """Never touch the developer's ~/.picasso/settings.yaml."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setattr(io, "_user_settings_filename", lambda: str(path))
    return path


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window with one channel of a few localizations."""
    window = gui_render.Window(plugins_loaded=True)
    locs = pd.DataFrame(
        {
            "frame": np.zeros(3, dtype=np.int32),
            "x": [3.0, 10.0, 20.0],
            "y": [3.0, 8.0, 12.0],
            "lpx": np.full(3, 0.1),
            "lpy": np.full(3, 0.1),
            "photons": np.full(3, 1000.0),
        }
    )
    info = [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1,
            "Pixelsize": PIXELSIZE,
        }
    ]
    window.view.add(str(tmp_path / "locs.hdf5"), locs, info, render_=False)
    window.view.async_rendering = False
    window.view.resize(256, 128)
    window.view.fit_in_view()
    yield window
    window.view.stop_render_worker()
    window.window_rot.view_rot.stop_render_worker()


def _save_png(path, array) -> str:
    Image.fromarray(array).save(str(path))
    return str(path)


def test_window_shows_overlay_and_sizes(window, tmp_path):
    dialog = window.image_overlay_dialog
    before = _qimage_rgb(window.view.qimage)
    path = _save_png(
        tmp_path / "wf.png", np.full((16, 32), 255, dtype=np.uint8)
    )
    dialog.load_image(path)
    assert dialog.isVisible()
    assert "The sizes match." in dialog.sizes_label.text()
    assert dialog.blend.currentText() == "Additive"  # the default
    after = _qimage_rgb(window.view.qimage)
    assert after.mean() > before.mean() + 50  # 50 % white on black

    dialog.show_check.setChecked(False)
    hidden = _qimage_rgb(window.view.qimage)
    assert hidden.mean() < after.mean()


def test_window_behind_localizations(window, tmp_path):
    view = window.view
    dialog = window.image_overlay_dialog
    locs_only = _qimage_rgb(view.qimage)
    brightest = np.unravel_index(locs_only.sum(axis=2).argmax(), (128, 256))
    dialog.load_image(
        _save_png(tmp_path / "wf.png", np.full((16, 32), 128, np.uint8))
    )
    dialog.opacity.setValue(100)
    dialog.blend.setCurrentText("Behind localizations")
    assert tuple(view._color_range[0]) == (0, 0, 4)  # single channel, magma
    rgb = _qimage_rgb(view.qimage)
    # far from the localizations: the overlay (constant, so full
    # contrast); at the brightest localization: the localization
    assert tuple(rgb[5, 128]) == (255, 255, 255)
    np.testing.assert_array_equal(rgb[brightest], locs_only[brightest])


def test_window_hides_widgets_that_do_not_apply(window, tmp_path):
    dialog = window.image_overlay_dialog
    dialog.load_image(
        _save_png(
            tmp_path / "rgb.png",
            np.broadcast_to(np.uint8([1, 2, 3]), (8, 8, 3)).copy(),
        )
    )
    assert "differ" in dialog.sizes_label.text()
    assert dialog.color.isHidden() and dialog.minimum.isHidden()
    assert dialog.pixel_size.isHidden()
    dialog.scaling.setCurrentText("Image pixel size")
    assert not dialog.pixel_size.isHidden()
    info = dialog.export_info()
    assert "Overlay color" not in info
    assert "Overlay image pixel size (nm)" in info
    dialog.scaling.setCurrentText("Stretch to camera")
    assert "Overlay image pixel size (nm)" not in dialog.export_info()


def test_export_complete_includes_overlay(window, tmp_path, monkeypatch):
    dialog = window.image_overlay_dialog
    dialog.load_image(
        _save_png(tmp_path / "wf.png", np.full((16, 32), 255, np.uint8))
    )
    dialog.opacity.setValue(100)
    out = str(tmp_path / "complete.png")
    monkeypatch.setattr(
        gui_render.lib,
        "get_save_filename_ext_dialog",
        lambda *args, **kwargs: (out, ".png"),
    )
    window.export_complete()
    exported = np.asarray(Image.open(out).convert("RGB"))
    # the chip spans [-0.5, width - 0.5), the rendered image [0, width):
    # all but the last half camera pixel on the right and bottom
    height, width = exported.shape[:2]
    half_x = int(np.ceil(0.5 * width / WIDTH))
    half_y = int(np.ceil(0.5 * height / HEIGHT))
    assert (exported[: height - half_y, : width - half_x] == 255).all()
    assert (exported[:, -1] < 255).all() and (exported[-1] < 255).all()
    meta = io.load_info(out.replace(".png", ".yaml"))[0]
    assert meta["Overlay image"] == dialog.path
    assert meta["Overlay extent (X, Y, Width, Height; camera px)"] == [
        -0.5,
        -0.5,
        WIDTH,
        HEIGHT,
    ]


def test_window_float_tiff_and_pages(window, tmp_path):
    dialog = window.image_overlay_dialog
    assert dialog.page.isHidden()
    frames = np.stack(
        [np.full((16, 32), value, np.float32) for value in (0.25, 0.5)]
    )
    frames[0, 0, 0] = 0.0
    path = str(tmp_path / "movie.tif")
    tifffile.imwrite(path, frames, imagej=True)
    dialog.load_image(path)
    assert "32-bit float" in dialog.sizes_label.text()
    assert not dialog.page.isHidden() and dialog.page.maximum() == 2
    assert dialog.minimum.value() == 0.0
    assert dialog.maximum.value() == 0.25
    assert dialog.maximum.decimals() > 0
    dialog.page.setValue(2)
    assert dialog.data.max() == 0.5
    assert dialog.maximum.value() == 0.25  # contrast kept across pages
    assert dialog.export_info()["Overlay image page"] == 2
    dialog.load_image(
        _save_png(tmp_path / "wf.png", np.zeros((16, 32), np.uint8))
    )
    assert dialog.page.isHidden()
    assert "Overlay image page" not in dialog.export_info()


def test_export_metadata_without_overlay(window):
    info = window.export_current_info(None)
    assert not any(key.startswith("Overlay") for key in info)
