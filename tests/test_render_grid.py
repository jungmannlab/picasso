"""The reference grid of ``picasso.render`` (``picasso.render.grid``).

The 2D grid sits at whole multiples of the spacing in the data's
coordinates; the 3D grid planes span the data's bounding box, are
projected exactly like the localizations and, at the back of the box,
always lie on the side facing away from the viewer. Drawn behind the
localizations, the grid never dims them.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtGui
from scipy.spatial.transform import Rotation

from picasso import render

PIXELSIZE = 10.0  # nm
# 200 x 100 display pixels showing 20 x 10 camera pixels
VIEWPORT = ((0.0, 0.0), (10.0, 20.0))
SIZE = (200, 100)


def _image(color=(0, 0, 0)) -> QtGui.QImage:
    image = QtGui.QImage(*SIZE, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(*color))
    return image


def _rgb(image: QtGui.QImage) -> np.ndarray:
    """RGB pixels (height, width, 3) of an RGB32 image."""
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    bgra = np.frombuffer(bits, dtype=np.uint8).reshape(
        image.height(), image.width(), 4
    )
    return bgra[:, :, 2::-1].copy()


def _grid(**kwargs) -> render.GridOverlay:
    return render.GridOverlay(spacing_nm=100.0, **kwargs)  # 10 camera px


def test_levels_include_both_ends_and_follow_the_origin():
    np.testing.assert_allclose(render.grid_levels(0, 10, 5), [0, 5, 10])
    np.testing.assert_allclose(
        render.grid_levels(1, 12, 5, origin=2), [2, 7, 12]
    )
    np.testing.assert_allclose(render.grid_levels(-7, 4, 5), [-5, 0])
    # too many lines to draw
    assert render.grid_levels(0, 1e9, 1).size == 0


def test_box_snaps_to_the_grid_and_spans_a_cell_on_flat_axes():
    lower, upper = render.grid_box(((1, 2, 0), (9, 10, 0)), 5)
    np.testing.assert_allclose(lower, [0, 0, 0])
    np.testing.assert_allclose(upper, [10, 10, 5])
    lower, upper = render.grid_box(((1, 2, 0), (9, 8, 0)), 5, (0, 0, 1))
    np.testing.assert_allclose(lower, [0, 0, -4])
    np.testing.assert_allclose(upper, [10, 10, 1])


def test_unrotated_view_puts_the_xy_plane_at_the_lowest_z():
    """Rotated +z points towards the viewer (the trackball moves the
    front of the data with the cursor), so the back is the lowest z."""
    bounds = ((0, 0, -10), (20, 20, 10))
    segments = render.grid_segments_3d(
        bounds, 5, Rotation.identity(), planes=("xy",)
    )
    assert np.all(segments[:, :, 2] == -10)
    # turned upside down, the back is the highest z
    flipped = Rotation.from_rotvec([np.pi, 0, 0])
    segments = render.grid_segments_3d(bounds, 5, flipped, planes=("xy",))
    assert np.all(segments[:, :, 2] == 10)
    centered = render.grid_segments_3d(
        bounds, 5, flipped, planes=("xy",), position="center"
    )
    assert np.all(centered[:, :, 2] == 0)


@pytest.mark.parametrize("seed", range(5))
def test_back_planes_lie_behind_the_center_of_the_box(seed):
    R = Rotation.random(random_state=seed)
    bounds = ((0, 0, -10), (20, 30, 10))
    center_depth = R.apply([10, 15, 0])[2]
    for plane in render.GRID_PLANES:
        segments = render.grid_segments_3d(bounds, 5, R, planes=(plane,))
        depth = R.apply(segments.reshape(-1, 3))[:, 2]
        assert depth.mean() <= center_depth + 1e-9


def test_planes_hold_every_grid_line_of_the_box():
    segments = render.grid_segments_3d(
        ((0, 0, 0), (20, 10, 5)), 5, Rotation.identity(), planes=("xy",)
    )
    # 5 lines along y (x = 0..20) and 3 along x (y = 0..10)
    assert segments.shape == (8, 2, 3)


def test_projection_matches_the_localizations():
    rng = np.random.default_rng(0)
    points = rng.uniform([0, 0, -5], [20, 10, 5], size=(50, 3))
    R = Rotation.random(random_state=1)
    locs = pd.DataFrame(points, columns=["x", "y", "z"])
    (y_min, x_min), (y_max, x_max) = VIEWPORT
    x, y, in_view, _ = render.locs_rotation(
        locs, 1.0, x_min, x_max, y_min, y_max, R
    )
    projected = render.project_points(points, VIEWPORT, R)[in_view]
    np.testing.assert_allclose(projected[:, 0] - x_min, x)
    np.testing.assert_allclose(projected[:, 1] - y_min, y)


def test_2d_lines_sit_at_multiples_of_the_spacing(qapp):
    viewport = ((0.0, 3.0), (10.0, 23.0))  # x from 3 to 23
    image = render.draw_grid(_image(), viewport, _grid(), PIXELSIZE)
    row = _rgb(image)[55, :, 0]
    # x = 10 and 20 camera px, 10 display px per camera px
    assert list(np.flatnonzero(row)) == [70, 170]
    # 40% opaque white over black
    assert row[70] == pytest.approx(0.4 * 255, abs=1)
    column = _rgb(image)[:, 5, 0]
    assert list(np.flatnonzero(column)) == [0]  # y = 0 (and 10, outside)


def test_lines_too_close_together_are_not_drawn(qapp):
    dense = render.GridOverlay(spacing_nm=3.0)  # 3 display px apart
    image = render.draw_grid(_image(), VIEWPORT, dense, PIXELSIZE)
    assert _rgb(image).max() == 0


def test_crossing_lines_do_not_add_up(qapp):
    image = render.draw_grid(_image(), VIEWPORT, _grid(), PIXELSIZE)
    rgb = _rgb(image)
    assert rgb[0, 100, 0] == rgb[50, 100, 0]  # crossing and plain line


def test_behind_keeps_localizations_and_over_covers_them(qapp):
    def with_locs():
        image = _image()
        painter = QtGui.QPainter(image)
        painter.fillRect(95, 0, 10, 100, QtGui.QColor(255, 255, 255))
        painter.end()
        return image

    red = render.OverlayStyle(color="red")
    behind = _rgb(
        render.draw_grid(with_locs(), VIEWPORT, _grid(style=red), PIXELSIZE)
    )
    over = _rgb(
        render.draw_grid(
            with_locs(),
            VIEWPORT,
            _grid(style=red, blend="Over localizations"),
            PIXELSIZE,
        )
    )
    # a bright localization on the line at x = 100 stays as it is
    # behind, but is tinted by the line over it
    assert tuple(behind[50, 100]) == (255, 255, 255)
    assert tuple(over[50, 100]) == (255, 0, 0)
    # where there are no localizations, both show the line
    assert tuple(behind[50, 0]) == tuple(over[50, 0]) == (255, 0, 0)


def test_automatic_color_follows_the_background(qapp):
    white = (255, 255, 255)
    image = render.draw_grid(
        _image(white),
        VIEWPORT,
        _grid(),
        PIXELSIZE,
        color_range=(white, (0, 0, 0)),
    )
    rgb = _rgb(image)
    assert rgb[50, 100, 0] == pytest.approx(0.6 * 255, abs=1)  # gray
    assert rgb[50, 50, 0] == 255  # untouched between the lines


def test_unrotated_3d_grid_matches_the_2d_grid(qapp):
    flat = render.draw_grid(_image(), VIEWPORT, _grid(), PIXELSIZE)
    planes = render.draw_grid_3d(
        _image(),
        VIEWPORT,
        _grid(planes=("xy",)),
        PIXELSIZE,
        Rotation.identity(),
        ((0, 0, -1), (20, 10, 1)),
    )
    lit_2d = np.flatnonzero(_rgb(flat)[50, :, 0])
    lit_3d = np.flatnonzero(_rgb(planes)[50, :, 0])
    # antialiased lines may spread over the neighboring pixel
    for column in lit_2d:
        assert np.min(np.abs(lit_3d - column)) <= 1


def test_3d_grid_rotates_with_the_view(qapp):
    def draw(R):
        return _rgb(
            render.draw_grid_3d(
                _image(),
                VIEWPORT,
                _grid(),
                PIXELSIZE,
                R,
                ((5, 2, -3), (15, 8, 3)),
            )
        )

    assert (
        draw(Rotation.identity()) != draw(Rotation.random(random_state=0))
    ).any()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spacing_nm=0),
        dict(spacing_nm=10, blend="Additive"),
        dict(spacing_nm=10, planes=("xw",)),
        dict(spacing_nm=10, position="front"),
    ],
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        render.GridOverlay(**kwargs)
