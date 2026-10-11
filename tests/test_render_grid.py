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


def test_label_texts_use_nm_or_um_with_the_spacing_s_decimals():
    assert render.format_grid_labels([0, 500, 1000], 500) == [
        "0 nm",
        "500 nm",
        "1000 nm",
    ]
    assert render.format_grid_labels([-1500, 0, 1500], 1500) == [
        "-1.5 \u00b5m",
        "0.0 \u00b5m",
        "1.5 \u00b5m",
    ]
    # large coordinates switch to µm even for a fine spacing
    assert render.format_grid_labels([12_250, 12_500], 250) == [
        "12.25 \u00b5m",
        "12.50 \u00b5m",
    ]
    assert render.format_grid_labels([-0.0001], 100) == ["0 nm"]
    assert render.format_grid_labels([-200], 100, axis="z") == ["z = -200 nm"]


def test_2d_labels_sit_by_their_lines_and_are_thinned_out(qapp):
    from picasso.render import grid as grid_module

    grid = _grid(labels=True)
    levels = np.arange(0, 21, 10.0)
    xs = levels * 10  # 10 display px per camera px
    labels = grid_module._grid_labels_2d(
        grid, levels, levels[:2], xs, xs[:2], 10.0, PIXELSIZE
    )
    texts = [text for text, _ in labels]
    # x = 0 runs into the corner of the left labels
    assert texts == ["100 nm", "200 nm", "100 nm"]
    top = [rect for _, rect in labels[:2]]
    assert top[0].left() > 100 and top[0].top() < 10
    # one line every 2 display pixels: labels on every n-th line only
    fine = np.arange(0, 200, 1.0)
    labels = grid_module._grid_labels_2d(
        grid, fine, fine[:0], fine * 2, fine[:0], 1.0, PIXELSIZE
    )
    rects = [rect for _, rect in labels]
    assert 1 < len(rects) < 50
    assert not any(a.intersects(b) for a, b in zip(rects, rects[1:]))


def test_labels_that_overlap_or_leave_the_image_are_dropped(qapp):
    from PyQt6 import QtCore

    from picasso.render import grid as grid_module

    rect = QtCore.QRectF
    labels = [
        ("a", rect(10, 10, 30, 12)),
        ("b", rect(20, 15, 30, 12)),  # overlaps a
        ("c", rect(190, 10, 30, 12)),  # past the right edge
        ("d", rect(100, 10, 30, 12)),
    ]
    placed = grid_module._place_labels(labels, rect(0, 0, 200, 100))
    assert [text for text, _ in placed] == ["a", "d"]


def _labels_3d(R, grid, image_size=(2000, 2000), axes=(0, 1, 2)):
    """3D labels of ``grid`` over the box ((2, 2, -3), (18, 8, 3)),
    viewed with rotation R, 10 display px per camera px; only the lines
    of ``axes`` are labeled."""
    from PyQt6 import QtCore

    from picasso.render import grid as grid_module

    bounds = ((2, 2, -3), (18, 8, 3))
    families = grid_module._line_families(
        bounds, 10.0, R, grid.planes, grid.position, (0.0, 0.0, 0.0)
    )
    families = [family for family in families if family[0] in axes]

    def to_display(points):
        # centered in a large image, so no label leaves it
        rotated = render.project_points(points, VIEWPORT, R)
        return rotated[..., :2] * 10 + 900

    return grid_module._grid_labels_3d(
        grid,
        families,
        10.0,
        (0.0, 0.0, 0.0),
        PIXELSIZE,
        to_display,
        QtCore.QRectF(0, 0, *image_size),
    )


@pytest.mark.parametrize("seed", range(4))
def test_3d_labels_name_their_axis_once_each(qapp, seed):
    R = Rotation.random(random_state=seed)
    labels = _labels_3d(R, _grid(labels=True))
    axes = [text.split(" = ")[0] for text, _ in labels]
    assert set(axes) <= {"x", "y", "z"}
    assert all(text.endswith("nm") for text, _ in labels)
    # z comes first, so it wins where labels collide
    if "z" in axes:
        assert axes[0] == "z"
    # each axis on one family: its labels line up along the ends of
    # its lines (within a label's size, which shifts each center)
    for axis in set(axes):
        rects = [rect for (text, rect) in labels if text.startswith(axis)]
        if len(rects) > 2:
            centers = np.array(
                [(r.center().x(), r.center().y()) for r in rects]
            )
            size = max(max(r.width(), r.height()) for r in rects)
            offsets = centers - centers.mean(axis=0)
            _, _, axes_ = np.linalg.svd(offsets)
            across = offsets @ axes_[-1]  # off the best-fitting line
            assert np.abs(across).max() <= size


@pytest.mark.parametrize("seed", range(8))
def test_z_lines_are_preferred_for_labels(qapp, seed):
    """The z lines get the labels they would get if x and y were not
    labeled at all: x and y take only the space z leaves."""
    R = Rotation.random(random_state=seed)
    grid = _grid(labels=True)
    z_only = _labels_3d(R, grid, axes=(2,))
    all_axes = _labels_3d(R, grid)
    z_labels = [(t, r) for t, r in all_axes if t.startswith("z")]
    assert [t for t, _ in z_labels] == [t for t, _ in z_only]
    assert [r for _, r in z_labels] == [r for _, r in z_only]
    assert len(all_axes) > len(z_only)  # x and y are labeled too


def test_3d_labels_sit_just_past_the_ends_of_their_lines(qapp):
    """Seen from above (no rotation), the planes' lines are those of a
    2D grid, and the labels sit past their outer ends."""
    grid = _grid(labels=True, planes=("xy",))
    labels = _labels_3d(Rotation.identity(), grid)
    # x lines run along y (from y = 0 to 10 camera px = 900..1000 px):
    # labeled past one end, outside the grid
    x_labels = [rect for text, rect in labels if text.startswith("x")]
    assert x_labels
    assert all(r.bottom() < 900 or r.top() > 1000 for r in x_labels)
    y_labels = [rect for text, rect in labels if text.startswith("y")]
    assert all(r.right() < 900 or r.left() > 1100 for r in y_labels)
    # no z lines in the xy plane
    assert not any(text.startswith("z") for text, _ in labels)


def test_3d_labels_stay_off_the_grid(qapp):
    R = Rotation.from_euler("xyz", [-60, 0, 30], degrees=True)
    labels = _labels_3d(R, _grid(labels=True))
    from picasso.render import grid as grid_module

    bounds = ((2, 2, -3), (18, 8, 3))
    segments = render.grid_segments_3d(bounds, 10.0, R, origin=(0.0, 0.0, 0.0))
    ends = render.project_points(segments, VIEWPORT, R)[..., :2] * 10 + 900
    outline = grid_module._outline(ends)
    assert labels
    for _, rect in labels:
        assert not outline.intersects(grid_module.QtGui.QPolygonF(rect))


def test_labels_are_drawn_only_when_asked(qapp):
    plain = _rgb(render.draw_grid(_image(), VIEWPORT, _grid(), PIXELSIZE))
    labeled = _rgb(
        render.draw_grid(_image(), VIEWPORT, _grid(labels=True), PIXELSIZE)
    )
    assert (labeled != plain).any()


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
