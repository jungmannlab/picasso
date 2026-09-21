"""Trackball rotation in Render's 3D window and the animation segments
it records.

These tests drive the real mouse handlers of a ``ViewRotation`` with a
drag of more than one full turn - the case that cannot be recovered
from the two checkpoint orientations, because a full turn leaves the
orientation unchanged. A hand-held drag is never perfectly horizontal,
so the drag here jitters vertically the way a real one does; that
jitter is what used to make the recorded segment lose a whole
revolution, turning a full spin into a short wiggle in the animation.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6 import QtCore

from picasso import render
from picasso.gui import render as gui_render

CANVAS = 512  # width and height of the canvas in pixels
# a drag across the full width of the canvas turns the data once
DEG_PER_PIXEL = 360.0 / CANVAS
FULL_VIEWPORT = ((0.0, 0.0), (32.0, 32.0))


class _Event:
    """The parts of a Qt mouse event the rotate handlers read."""

    def __init__(self, x, y, button=QtCore.Qt.MouseButton.LeftButton):
        self._pos = QtCore.QPoint(int(x), int(y))
        self._button = button

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def buttons(self):
        return self._button

    def accept(self):
        pass


@pytest.fixture
def view_rot(qt_offscreen):
    """The canvas of the 3D window, which Render builds with itself.

    No localizations are loaded: the drag handlers and the rotation
    state they keep are what these tests cover, not the rendering.
    """
    window = gui_render.Window(plugins_loaded=True)
    view = window.window_rot.view_rot
    view.resize(CANVAS, CANVAS)
    return view


def _drag(view, total_deg, step_px=2, jitter_px=2, seed=0):
    """Spin the data by dragging the mouse horizontally.

    The drag is split into sweeps across the canvas, each one a press,
    a series of moves and a release, as a user has to do for a rotation
    of more than one turn. Every move sits up to ``jitter_px`` pixels
    above or below the line of the drag, the way a held mouse trembles,
    so no sweep is perfectly around one axis.
    """
    rng = np.random.default_rng(seed)
    remaining = int(round(total_deg / DEG_PER_PIXEL))
    baseline = CANVAS // 2
    while remaining > 0:
        sweep = min(remaining, CANVAS - 1)
        remaining -= sweep
        x, y = 0, baseline
        view.mousePressEvent(_Event(x, y))
        while x < sweep:
            x = min(x + step_px, sweep)
            y = baseline + int(rng.integers(-jitter_px, jitter_px + 1))
            view.mouseMoveEvent(_Event(x, y))
        view.mouseReleaseEvent(_Event(x, y))


def _swept(rotations):
    """Total angle (degrees) traveled along a sequence of rotations."""
    return np.degrees(
        sum(
            (rotations[i + 1] * rotations[i].inv()).magnitude()
            for i in range(len(rotations) - 1)
        )
    )


class TestDraggedRotationSegments:
    """What a drag records for an animation segment."""

    # no two drags tremble alike, and the turn has to survive all of
    # them - it used to be lost or kept depending on where in the
    # tremor the drag happened to pass through the full turn
    @pytest.mark.parametrize("seed", range(5))
    def test_drag_beyond_a_full_turn_keeps_the_turn(self, view_rot, seed):
        """A dragged spin of 378 degrees is recorded as 378 degrees,
        not as the 18 degrees the two orientations differ by."""
        view_rot.reset_rotation_anchor()  # as 'Add this position' does
        _drag(view_rot, 378.0, seed=seed)

        segment = view_rot.rotation_since_anchor()
        assert np.degrees(np.linalg.norm(segment)) == pytest.approx(
            378.0, abs=2.0
        )
        # the spin is around y, and the jitter stays small
        assert np.degrees(segment[1]) == pytest.approx(378.0, abs=2.0)
        assert abs(np.degrees(segment[0])) < 5.0

        # the displayed angles keep the turn as well
        assert np.degrees(view_rot.angy) == pytest.approx(378.0, abs=2.0)

        # ... while the orientation alone cannot: the data sits 18
        # degrees away from where it started, which is exactly why the
        # segment has to be recorded along the way
        assert np.degrees(view_rot.rotation.magnitude()) == pytest.approx(
            18.0, abs=3.0
        )

    def test_animation_spins_the_dragged_turn(self, view_rot):
        """The animation built from such a drag plays the full spin and
        still ends exactly at the checkpoint."""
        R1 = view_rot.rotation
        view_rot.reset_rotation_anchor()
        _drag(view_rot, 378.0)
        R2 = view_rot.rotation
        segment = view_rot.rotation_since_anchor()

        rotations, _ = render._animation_sequence(
            positions=[(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT)],
            durations=[4.2],
            fps=30,
            segment_rotations=[segment],
        )
        assert _swept(rotations) == pytest.approx(378.0, abs=2.0)
        assert (rotations[-1] * R2.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )
        # at constant speed, no jump anywhere along the way
        steps = np.degrees(
            [
                (rotations[i + 1] * rotations[i].inv()).magnitude()
                for i in range(len(rotations) - 1)
            ]
        )
        assert steps.max() - steps.min() < 0.1

    def test_two_dragged_turns(self, view_rot):
        """Two full turns are two full turns, in the segment and in the
        animation, although the data ends up where it started."""
        R1 = view_rot.rotation
        view_rot.reset_rotation_anchor()
        _drag(view_rot, 720.0, seed=3)
        R2 = view_rot.rotation
        segment = view_rot.rotation_since_anchor()

        assert np.degrees(np.linalg.norm(segment)) == pytest.approx(
            720.0, abs=3.0
        )
        # the two checkpoints are the same orientation up to the jitter
        assert np.degrees((R2 * R1.inv()).magnitude()) < 10.0
        rotations, _ = render._animation_sequence(
            positions=[(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT)],
            durations=[8.0],
            fps=30,
            segment_rotations=[segment],
        )
        assert _swept(rotations) == pytest.approx(720.0, abs=3.0)

    def test_short_drag_animates_as_slerp(self, view_rot):
        """A drag of less than half a turn holds no turns to keep, so
        its animation stays the plain shortest path between the two
        orientations."""
        R1 = view_rot.rotation
        view_rot.reset_rotation_anchor()
        _drag(view_rot, 90.0, seed=1)
        R2 = view_rot.rotation
        segment = view_rot.rotation_since_anchor()
        assert np.degrees(np.linalg.norm(segment)) == pytest.approx(
            90.0, abs=1.0
        )

        positions = [(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT)]
        rotations, _ = render._animation_sequence(
            positions, [1.0], fps=30, segment_rotations=[segment]
        )
        slerp, _ = render._animation_sequence(positions, [1.0], fps=30)
        assert _swept(rotations) == pytest.approx(90.0, abs=1.0)
        assert (rotations[-1] * R2.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )
        # the recorded path and the shortest path agree frame by frame
        # (they differ only by how the drag's tremor is distributed)
        deviation = max(
            np.degrees((a * b.inv()).magnitude())
            for a, b in zip(rotations, slerp)
        )
        assert deviation < 1.0

    def test_anchor_resets_at_every_position(self, view_rot):
        """Each segment counts from the position before it, so two
        drags in a row record one turn each, not one and two."""
        view_rot.reset_rotation_anchor()
        _drag(view_rot, 370.0, seed=4)
        first = np.degrees(np.linalg.norm(view_rot.rotation_since_anchor()))

        view_rot.reset_rotation_anchor()
        _drag(view_rot, 370.0, seed=5)
        second = np.degrees(np.linalg.norm(view_rot.rotation_since_anchor()))

        assert first == pytest.approx(370.0, abs=2.0)
        assert second == pytest.approx(370.0, abs=2.0)
        # the displayed angles, in contrast, keep accumulating
        assert np.degrees(view_rot.angy) == pytest.approx(740.0, abs=4.0)
