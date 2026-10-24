"""Tracks of single-particle tracking drawn onto rendered localizations
(``picasso.render.tracks``).

Segments are culled to the field of view; when too many tracks are in
view, a fixed subset of track ids is drawn. Overlapping lines do not add
up, and a style color overrides the per-track colors.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtGui

from picasso import render

# 100 x 100 display pixels showing 50 x 50 nm (segments are in nm)
VIEWPORT = ((0.0, 0.0), (50.0, 50.0))
SIZE = (100, 100)


@pytest.fixture(autouse=True)
def _qt(qapp):
    """QPainter needs the application object."""
    yield


def _image() -> QtGui.QImage:
    image = QtGui.QImage(*SIZE, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(0, 0, 0))
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


def test_segments_outside_the_view_are_culled():
    segments = np.array(
        [
            [10.0, 10.0, 12.0, 10.0],  # inside
            [-5.0, 20.0, 5.0, 20.0],  # crosses the left edge
            [60.0, 60.0, 61.0, 61.0],  # outside
            [-3.0, 70.0, 80.0, 75.0],  # below the view
        ]
    )
    track_id = np.array([0, 1, 2, 3])
    mask, n_visible, n_drawn = render.select_track_segments(
        segments, track_id, VIEWPORT
    )
    assert mask.tolist() == [True, True, False, False]
    assert n_visible == n_drawn == 2


def test_too_many_tracks_draw_a_fixed_subset():
    n = 100
    segments = np.column_stack(
        [
            np.arange(n) * 0.4,
            np.full(n, 5.0),
            np.arange(n) * 0.4,
            np.full(n, 6),
        ]
    )
    track_id = np.arange(n)
    mask, n_visible, n_drawn = render.select_track_segments(
        segments, track_id, VIEWPORT, max_tracks=10
    )
    assert n_visible == n
    assert n_drawn == 10
    assert set(track_id[mask]) == set(range(0, n, 10))
    # a small pan keeps the drawn tracks (ids), it does not reshuffle
    shifted = ((0.0, 0.5), (50.0, 50.5))
    mask2, _, _ = render.select_track_segments(
        segments, track_id, shifted, max_tracks=10
    )
    assert set(track_id[mask2]) <= set(range(0, n, 10))
    # several segments of one track count once
    mask, n_visible, _ = render.select_track_segments(
        segments, np.repeat(np.arange(n // 2), 2), VIEWPORT
    )
    assert n_visible == n // 2


def _line_rows(rgb: np.ndarray, column: int = 50) -> np.ndarray:
    """Rows of ``column`` where something was drawn."""
    return np.flatnonzero(rgb[:, column].any(axis=1))


def test_tracks_are_drawn_in_their_palette_colors():
    segments = np.array([[10.0, 10.0, 40.0, 10.0], [10.0, 30.0, 40.0, 30.0]])
    palette = [(255, 0, 0), (0, 255, 0)]
    image = render.draw_tracks(
        _image(), VIEWPORT, segments, np.array([0, 1]), palette
    )
    rgb = _rgb(image)
    rows = _line_rows(rgb)
    # one sharp line each, at y = 10 and 30 nm
    assert len(rows) == 2
    assert abs(rows[0] - 20) <= 1 and abs(rows[1] - 60) <= 1
    assert tuple(rgb[rows[0], 50]) == (255, 0, 0)
    assert tuple(rgb[rows[1], 50]) == (0, 255, 0)
    assert not rgb[:, 5].any()  # left of the segments


def test_style_color_overrides_the_palette():
    segments = np.array([[10.0, 10.0, 40.0, 10.0]])
    style = render.OverlayStyle(color="#0000FF", line_width=3.0)
    image = render.draw_tracks(
        _image(), VIEWPORT, segments, np.array([0]), [(255, 0, 0)], style
    )
    rgb = _rgb(image)
    rows = _line_rows(rgb)
    assert 3 <= len(rows) <= 4  # 3 px wide, antialiased
    assert not rgb[rows, 50, :2].any()  # blue only
    assert rgb[rows, 50, 2].max() == 255


def test_overlapping_lines_do_not_add_up():
    segments = np.array([[10.0, 10.0, 40.0, 10.0]] * 2)
    style = render.OverlayStyle(opacity=0.5)
    image = render.draw_tracks(
        _image(), VIEWPORT, segments, None, [(255, 255, 255)], style
    )
    rgb = _rgb(image)
    rows = _line_rows(rgb)
    assert len(rows) == 1
    assert rgb[rows[0], 50, 0] == pytest.approx(128, abs=2)


def test_nothing_to_draw():
    image = _image()
    empty = np.zeros((0, 4))
    assert render.draw_tracks(image, VIEWPORT, empty) is image
    flat = ((0.0, 0.0), (0.0, 50.0))
    segments = np.array([[10.0, 10.0, 40.0, 10.0]])
    assert render.draw_tracks(image, flat, segments) is image
