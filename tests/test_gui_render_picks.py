"""The "Box" and "Brush" pick shapes of ``picasso.gui.render``.

Both are drawn by dragging rather than clicked into place, and both
carry their own extent instead of a shared size, so ``_pick_size`` is
None for them. These tests drive the real mouse handlers of a ``View``
to cover the drag, the removal, the YAML round trip and the metadata,
which is where the None size would otherwise surface as a
``TypeError``. For the brush they also cover the merging of overlapping
strokes and the undo of the last one.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from PyQt6 import QtCore

from picasso import io, lib
from picasso.gui import render as gui_render, rotation

WIDTH = HEIGHT = 32.0
PIXELSIZE = 130.0


def _locs(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": np.full(n, 0.1),
            "lpy": np.full(n, 0.1),
            "photons": np.full(n, 1000.0),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window holding one channel of uniform localizations."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(256, 256)
    return window


class _Event:
    """The parts of a Qt mouse event the pick handlers read."""

    def __init__(self, x, y, button=QtCore.Qt.MouseButton.LeftButton):
        self._pos = QtCore.QPoint(int(x), int(y))
        self._button = button

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def buttons(self):
        return self._button

    def modifiers(self):
        return QtCore.Qt.KeyboardModifier.NoModifier

    def accept(self):
        pass

    def ignore(self):
        pass


def _drag(view, x0, y0, x1, y1):
    """Press, move and release, as the canvas would."""
    view.mousePressEvent(_Event(x0, y0))
    view.mouseMoveEvent(_Event(x1, y1))
    view.mouseReleaseEvent(_Event(x1, y1))


def _paint(view, points):
    """Paint a brush stroke through the given screen positions."""
    view.mousePressEvent(_Event(*points[0]))
    for point in points[1:]:
        view.mouseMoveEvent(_Event(*point))
    view.mouseReleaseEvent(_Event(*points[-1]))


@pytest.fixture
def box_view(window):
    """The view of ``window``, in Pick mode with the Box shape."""
    view = window.view
    window.tools_settings_dialog.pick_shape.setCurrentText("Box")
    view._mode = "Pick"
    assert view._pick_shape == "Box"
    return view


class TestBoxPickTool:
    def test_box_is_registered_as_a_pick_shape(self):
        assert "Box" in lib.PICK_SHAPES
        assert "Box" in lib.PICK_SHAPES_WITHOUT_SIZE

    def test_shape_selector_offers_every_shape(self, window):
        combo = window.tools_settings_dialog.pick_shape
        offered = [combo.itemText(i) for i in range(combo.count())]
        assert offered == list(lib.PICK_SHAPES)

    def test_pick_size_is_none(self, box_view):
        assert box_view._pick_size is None

    def test_drag_creates_one_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        assert len(box_view._picks) == 1
        (x0, y0), (x1, y1) = box_view._picks[0]
        # corners come out ordered, whichever way the drag went
        assert x0 < x1 and y0 < y1

    def test_drag_upwards_gives_the_same_box(self, box_view):
        _drag(box_view, 120, 100, 40, 40)
        down = box_view._picks[0]
        box_view.clear_picks()
        _drag(box_view, 40, 40, 120, 100)
        assert box_view._picks[0] == pytest.approx(np.array(down))

    def test_box_spans_the_dragged_screen_region(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        (x0, y0), (x1, y1) = box_view._picks[0]
        assert (x0, y0) == pytest.approx(
            box_view.map_to_movie(QtCore.QPoint(40, 40))
        )
        assert (x1, y1) == pytest.approx(
            box_view.map_to_movie(QtCore.QPoint(120, 100))
        )

    def test_a_bare_click_creates_nothing(self, box_view):
        _drag(box_view, 60, 60, 60, 60)
        assert box_view._picks == []

    def test_drag_shorter_than_the_minimum_creates_nothing(self, box_view):
        short = gui_render.MIN_BOX_PICK_DRAG - 1
        _drag(box_view, 60, 60, 60 + short, 60 + short)
        assert box_view._picks == []

    def test_the_drag_overlay_is_cleared_on_release(self, box_view):
        box_view.mousePressEvent(_Event(40, 40))
        assert box_view._box_pick_ongoing
        box_view.mouseMoveEvent(_Event(120, 100))
        box_view.mouseReleaseEvent(_Event(120, 100))
        assert not box_view._box_pick_ongoing

    def test_right_click_inside_removes_the_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        inside = _Event(80, 70, QtCore.Qt.MouseButton.RightButton)
        box_view.mouseReleaseEvent(inside)
        assert box_view._picks == []

    def test_right_click_outside_keeps_the_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        outside = _Event(200, 200, QtCore.Qt.MouseButton.RightButton)
        box_view.mouseReleaseEvent(outside)
        assert len(box_view._picks) == 1

    def test_picked_locs_are_inside_the_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        (x0, y0), (x1, y1) = box_view._picks[0]
        picked = box_view.picked_locs(0)[0]
        assert len(picked) > 0
        assert (picked["x"] > x0).all() and (picked["x"] < x1).all()
        assert (picked["y"] > y0).all() and (picked["y"] < y1).all()

    def test_pick_bounds_match_the_drawn_box(self, box_view):
        # this is what "Move to pick" and the XY scatter frame on
        _drag(box_view, 40, 40, 120, 100)
        (x0, y0), (x1, y1) = box_view._picks[0]
        bounds = lib.pick_bounds(box_view._picks[0], "Box", None)
        assert bounds == pytest.approx((x0, x1, y0, y1))

    def test_pick_areas_are_per_box(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        _drag(box_view, 140, 140, 200, 160)
        areas = box_view.pick_areas()
        assert len(areas) == 2
        assert areas[0] > areas[1] > 0

    def test_saved_yaml_round_trips(self, box_view, tmp_path):
        _drag(box_view, 40, 40, 120, 100)
        _drag(box_view, 140, 140, 200, 160)
        path = str(tmp_path / "picks.yaml")
        box_view.save_picks(path)

        regions = yaml.full_load(open(path))
        assert regions["Shape"] == "Box"
        assert "Corners" in regions
        # a box has no global size to store
        assert not any("nm" in key for key in regions)

        saved = [np.array(pick) for pick in box_view._picks]
        box_view.clear_picks()
        box_view.load_picks(path)
        assert box_view._pick_shape == "Box"
        assert len(box_view._picks) == 2
        for loaded, original in zip(box_view._picks, saved):
            assert np.array(loaded) == pytest.approx(original)

    def test_pick_metadata_has_no_size_entry(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        pick_info = box_view._build_base_pick_info()
        assert pick_info["Pick Shape"] == "Box"
        assert pick_info["Number of picks"] == 1
        assert not any("Pick Diameter" in key for key in pick_info)
        # per-pick areas, unlike the single repeated value of a circle
        assert len(pick_info["Pick Areas (um^2)"]) == 1

    def test_index_blocks_are_not_built(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        assert box_view.get_index_blocks(0) is None

    def test_pick_similar_accepts_boxes(self, box_view, monkeypatch):
        _drag(box_view, 40, 40, 120, 100)
        _drag(box_view, 140, 40, 220, 100)
        monkeypatch.setattr(
            gui_render.View, "get_channel", lambda self, title: 0
        )
        warned = []
        monkeypatch.setattr(
            gui_render.QtWidgets.QMessageBox,
            "warning",
            lambda *a, **k: warned.append(a),
        )
        box_view.pick_similar()
        # the shape guard must not fire, and pick_size is None here
        assert warned == []
        assert len(box_view._picks) >= 2

    def test_the_scene_draws_with_a_none_pick_size(self, box_view):
        _drag(box_view, 40, 40, 120, 100)
        box_view.update_scene()
        assert box_view.qimage is not None

    def test_the_drag_overlay_paints(self, box_view):
        box_view.mousePressEvent(_Event(40, 40))
        box_view.mouseMoveEvent(_Event(120, 100))
        image = box_view.draw_box_pick_ongoing(box_view.qimage_no_picks.copy())
        assert image is not None


class TestRectanglePickMinimumLength:
    @pytest.fixture
    def rectangle_view(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Rectangle")
        view._mode = "Pick"
        assert view._pick_shape == "Rectangle"
        return view

    def test_a_bare_click_creates_nothing(self, rectangle_view):
        _drag(rectangle_view, 60, 60, 60, 60)
        assert rectangle_view._picks == []
        assert not rectangle_view._rectangle_pick_ongoing

    def test_drag_shorter_than_the_minimum_creates_nothing(
        self, rectangle_view
    ):
        _drag(rectangle_view, 60, 60, 63, 63)  # 4.2 display pixels
        assert rectangle_view._picks == []

    def test_drag_of_the_minimum_length_creates_a_pick(self, rectangle_view):
        _drag(rectangle_view, 60, 60, 63, 64)  # 5 display pixels
        assert len(rectangle_view._picks) == 1


class TestLoaderUsesTheStoredIndex:
    """The loader thread reads the pyramid stored in the file by
    ``io.save_locs`` instead of building one, when it is valid."""

    def _load(self, path):
        worker = gui_render.LocsLoadWorker([(path, None)])
        results = []
        worker.loaded.connect(lambda *args: results.append(args))
        worker.run()
        assert len(results) == 1
        return results[0]

    def test_stored_index_is_used(self, qt_offscreen, tmp_path, monkeypatch):
        from picasso import io, spatial_index

        path = str(tmp_path / "indexed.hdf5")
        io.save_locs(path, _locs(), _info(), render_index=True)
        monkeypatch.setattr(
            spatial_index,
            "build_render_index",
            lambda *a, **k: pytest.fail("the index was rebuilt"),
        )
        _, locs, info, render_index = self._load(path)
        assert render_index is not None
        assert spatial_index.validate_render_index(render_index, locs, info)

    def test_invalid_stored_index_is_rebuilt(self, qt_offscreen, tmp_path):
        import h5py

        from picasso import io, spatial_index

        path = str(tmp_path / "edited.hdf5")
        io.save_locs(path, _locs(), _info(), render_index=True)
        with h5py.File(path, "r+") as f:  # a script rewrote the rows
            rows = f["locs"][()][::-1]
            del f["locs"]
            f.create_dataset("locs", data=rows)
        _, locs, info, render_index = self._load(path)
        assert render_index is not None
        assert spatial_index.validate_render_index(render_index, locs, info)


class TestCircularPicksUseThePyramid:
    """Circular picks query the render pyramid built at load, so no
    pick-size specific index blocks are built (that indexing sorted
    and copied every channel, again after each pick-size change)."""

    @pytest.fixture
    def circle_view(self, window):
        view = window.view
        view._pick_shape = "Circle"
        window.tools_settings_dialog.pick_diameter.setValue(6.0 * PIXELSIZE)
        view._picks = [(20.0, 20.0), (40.0, 44.0), (58.0, 6.0)]
        return view

    def test_no_index_blocks_are_built(self, circle_view, monkeypatch):
        view = circle_view
        assert view.render_index[0] is not None
        monkeypatch.setattr(
            view,
            "index_locs",
            lambda channel: pytest.fail("index blocks were built"),
        )
        picked = view.picked_locs(0)
        assert len(picked) == 3
        counts = view._count_locs_in_picks(0)
        assert list(counts) == [len(p) for p in picked]
        assert view.index_blocks[0] is None

    def test_matches_the_index_block_path(self, circle_view):
        view = circle_view
        via_pyramid = view.picked_locs(0)
        view.render_index[0] = None
        view._ensure_render_index = lambda channel: None  # no pyramid
        via_blocks = view.picked_locs(0)
        assert view.index_blocks[0] is not None  # the fallback indexed
        for a, b in zip(via_pyramid, via_blocks):
            # the index-block path standardizes dtypes (ensure_sanity)
            pd.testing.assert_frame_equal(
                a.sort_index(),
                b.sort_index(),
                check_like=True,
                check_dtype=False,
            )

    def test_pick_size_change_needs_no_reindexing(self, circle_view):
        view = circle_view
        before = view.picked_locs(0)
        view.window.tools_settings_dialog.pick_diameter.setValue(
            12.0 * PIXELSIZE
        )
        after = view.picked_locs(0)
        assert view.index_blocks[0] is None
        assert all(len(a) >= len(b) for a, b in zip(after, before))


class TestPickRemovalAcrossShapes:
    """``remove_picks`` dispatches through ``lib.point_in_pick``."""

    def _polygon_view(self, window, polygons):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Polygon")
        view._mode = "Pick"
        view._picks = polygons
        return view

    def test_polygon_click_removes_only_the_one_clicked(self, window):
        # this used to clear every pick: remove_picks had no polygon arm
        left = [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0), (1.0, 1.0)]
        right = [
            (20.0, 20.0),
            (25.0, 20.0),
            (25.0, 25.0),
            (20.0, 25.0),
            (20.0, 20.0),
        ]
        view = self._polygon_view(window, [left, right])
        view.remove_picks((3.0, 3.0))
        assert len(view._picks) == 1
        assert list(view._picks[0]) == right

    def test_polygon_click_outside_removes_nothing(self, window):
        left = [(1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0), (1.0, 1.0)]
        view = self._polygon_view(window, [left])
        view.remove_picks((10.0, 10.0))
        assert len(view._picks) == 1

    def test_vertical_rectangle_survives_a_click_elsewhere(self, window):
        # a perfectly vertical rectangle used to be deleted by any
        # right click, because its corner ray casting divided by zero
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Rectangle")
        window.tools_settings_dialog.pick_width.setValue(2.0 * PIXELSIZE)
        view._picks = [((5.0, 5.0), (5.0, 15.0))]
        view.remove_picks((20.0, 20.0))
        assert len(view._picks) == 1

    def test_vertical_rectangle_is_removed_from_inside(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Rectangle")
        window.tools_settings_dialog.pick_width.setValue(2.0 * PIXELSIZE)
        view._picks = [((5.0, 5.0), (5.0, 15.0))]
        view.remove_picks((5.5, 10.0))
        assert view._picks == []


class TestRotationWindowShapes:
    """The 3D window frames and moves a pick of any shape."""

    class _Stub:
        """The attributes ``fit_in_view_rotated`` reads."""

        def __init__(self, pick, pick_shape, pick_size, fov_viewport=None):
            self.pick = pick
            self.pick_shape = pick_shape
            self.pick_size = pick_size
            self._fov_viewport = fov_viewport  # field-of-view mode

    def _viewport(self, pick, shape, size):
        return rotation.ViewRotation.fit_in_view_rotated(
            self._Stub(pick, shape, size), get_viewport=True
        )

    def test_box_viewport(self):
        (y_min, x_min), (y_max, x_max) = self._viewport(
            ((1.0, 2.0), (5.0, 6.0)), "Box", None
        )
        assert (x_min, x_max, y_min, y_max) == (1.0, 5.0, 2.0, 6.0)

    def test_circle_viewport_is_unchanged(self):
        (y_min, x_min), (y_max, x_max) = self._viewport(
            (5.0, 5.0), "Circle", 2.0
        )
        assert (x_min, x_max, y_min, y_max) == (4.0, 6.0, 4.0, 6.0)

    def test_polygon_viewport_is_unchanged(self):
        polygon = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]
        (y_min, x_min), (y_max, x_max) = self._viewport(
            polygon, "Polygon", None
        )
        assert (x_min, x_max, y_min, y_max) == (0.0, 4.0, 0.0, 4.0)

    def test_no_pick_yet(self):
        assert self._viewport(None, None, None) is None


class TestFilterPicksAcrossShapes:
    """``filter_picks`` counts through ``picked_locs`` off the circular
    fast path."""

    def test_counts_match_picked_locs_for_boxes(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Box")
        view._picks = [
            ((4.0, 4.0), (10.0, 10.0)),
            ((20.0, 20.0), (24.0, 22.0)),
        ]
        counts = view._count_locs_in_picks(0)
        expected = [len(_) for _ in view.picked_locs(0, add_group=False)]
        assert list(counts) == expected
        assert all(_ > 0 for _ in counts)

    def test_open_polygons_count_as_empty(self, window):
        view = window.view
        window.tools_settings_dialog.pick_shape.setCurrentText("Polygon")
        closed = [
            (4.0, 4.0),
            (10.0, 4.0),
            (10.0, 10.0),
            (4.0, 10.0),
            (4.0, 4.0),
        ]
        # picked_locs skips the open one, so the counts would otherwise
        # be misaligned with the picks
        view._picks = [closed, [(20.0, 20.0), (24.0, 20.0)], closed]
        counts = view._count_locs_in_picks(0)
        assert len(counts) == 3
        assert counts[1] == 0
        assert counts[0] > 0 and counts[0] == counts[2]


# ---------------------------------------------------------------------------
# The brush
# ---------------------------------------------------------------------------


@pytest.fixture
def brush_view(window):
    """The view of ``window``, in Pick mode with the Brush shape."""
    view = window.view
    window.tools_settings_dialog.pick_shape.setCurrentText("Brush")
    view._mode = "Pick"
    assert view._pick_shape == "Brush"
    return view


# two horizontal strokes far apart, and one that bridges them
STROKE_A = [(60, 60), (100, 60), (140, 60)]
STROKE_B = [(60, 200), (100, 200), (140, 200)]
BRIDGE = [(100, 60), (100, 130), (100, 200)]


class TestBrushPickTool:
    def test_brush_is_registered_as_a_pick_shape(self):
        assert "Brush" in lib.PICK_SHAPES
        assert "Brush" in lib.PICK_SHAPES_WITHOUT_SIZE

    def test_pick_size_is_none(self, brush_view):
        assert brush_view._pick_size is None

    def test_brush_width_follows_the_spin_box(self, brush_view, window):
        window.tools_settings_dialog.brush_width.setValue(260.0)
        assert brush_view._brush_width == pytest.approx(
            260.0 / brush_view.pixelsize
        )

    def test_one_stroke_makes_one_pick(self, brush_view):
        _paint(brush_view, STROKE_A)
        assert len(brush_view._picks) == 1
        assert len(brush_view._picks[0]) == 1

    def test_a_stroke_records_the_swept_path(self, brush_view):
        _paint(brush_view, STROKE_A)
        width, path = brush_view._picks[0][0]
        assert width == pytest.approx(brush_view._brush_width)
        assert len(path) >= 2
        assert path[0] == pytest.approx(
            brush_view.map_to_movie(QtCore.QPoint(*STROKE_A[0]))
        )

    def test_a_click_without_a_drag_paints_a_dot(self, brush_view):
        _paint(brush_view, [(80, 80)])
        assert len(brush_view._picks) == 1
        assert len(brush_view._picks[0][0][1]) == 1

    def test_separate_strokes_stay_separate(self, brush_view):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        assert len(brush_view._picks) == 2

    def test_a_bridging_stroke_merges_the_picks(self, brush_view):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        _paint(brush_view, BRIDGE)
        assert len(brush_view._picks) == 1
        assert len(brush_view._picks[0]) == 3

    def test_the_newest_stroke_is_last(self, brush_view):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        _paint(brush_view, BRIDGE)
        first_point = brush_view._picks[-1][-1][1][0]
        assert first_point == pytest.approx(
            brush_view.map_to_movie(QtCore.QPoint(*BRIDGE[0]))
        )

    def test_right_click_undoes_the_last_stroke(self, brush_view):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        brush_view.mouseReleaseEvent(
            _Event(300, 300, QtCore.Qt.MouseButton.RightButton)
        )
        assert len(brush_view._picks) == 1

    def test_undoing_a_bridge_splits_the_pick_again(self, brush_view):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        _paint(brush_view, BRIDGE)
        assert len(brush_view._picks) == 1
        # the position of the right click does not matter for the brush
        brush_view.mouseReleaseEvent(
            _Event(500, 500, QtCore.Qt.MouseButton.RightButton)
        )
        assert len(brush_view._picks) == 2

    def test_right_click_with_no_picks_is_harmless(self, brush_view):
        brush_view.mouseReleaseEvent(
            _Event(300, 300, QtCore.Qt.MouseButton.RightButton)
        )
        assert brush_view._picks == []

    def test_the_stroke_overlay_is_cleared_on_release(self, brush_view):
        brush_view.mousePressEvent(_Event(*STROKE_A[0]))
        assert brush_view._brush_stroke_ongoing
        brush_view.mouseMoveEvent(_Event(*STROKE_A[1]))
        assert brush_view._brush_stroke
        brush_view.mouseReleaseEvent(_Event(*STROKE_A[-1]))
        assert not brush_view._brush_stroke_ongoing
        assert not brush_view._brush_stroke

    def test_the_overlay_paints(self, brush_view):
        brush_view.mousePressEvent(_Event(*STROKE_A[0]))
        brush_view.mouseMoveEvent(_Event(*STROKE_A[1]))
        image = brush_view.draw_brush_stroke_ongoing(
            brush_view.qimage_no_picks.copy()
        )
        assert image is not None

    def test_changing_the_width_leaves_drawn_strokes_alone(
        self, brush_view, window
    ):
        _paint(brush_view, STROKE_A)
        drawn = brush_view._picks[0][0][0]
        window.tools_settings_dialog.brush_width.setValue(30.0)
        assert brush_view._picks[0][0][0] == drawn
        _paint(brush_view, STROKE_B)
        assert brush_view._picks[-1][-1][0] < drawn

    def test_picked_locs_are_inside_the_stroke(self, brush_view):
        _paint(brush_view, STROKE_A)
        picked = brush_view.picked_locs(0)[0]
        assert len(picked) > 0
        width, path = brush_view._picks[0][0]
        X = np.array([p[0] for p in path])
        Y = np.array([p[1] for p in path])
        assert lib.check_if_in_brush_stroke(
            picked["x"].to_numpy(), picked["y"].to_numpy(), X, Y, width / 2
        ).all()

    def test_pick_areas_are_per_pick(self, brush_view):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, [(300, 300), (320, 300)])
        areas = brush_view.pick_areas()
        assert len(areas) == 2
        assert areas[0] > areas[1] > 0

    def test_the_scene_draws_with_a_none_pick_size(self, brush_view):
        _paint(brush_view, STROKE_A)
        brush_view.update_scene()
        assert brush_view.qimage is not None

    def test_saved_yaml_round_trips(self, brush_view, tmp_path, window):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        _paint(brush_view, BRIDGE)  # merges the two into one pick
        path = str(tmp_path / "picks.yaml")
        brush_view.save_picks(path)

        regions = yaml.full_load(open(path))
        assert regions["Shape"] == "Brush"
        # a flat, ordered stroke list, each with its own width
        assert len(regions["Strokes"]) == 3
        assert set(regions["Strokes"][0]) == {"Width (nm)", "Path"}
        assert "Diameter (nm)" not in regions

        brush_view.clear_picks()
        brush_view.load_picks(path)
        # the merged grouping is re-derived from the strokes
        assert len(brush_view._picks) == 1
        assert len(brush_view._picks[0]) == 3
        assert window.tools_settings_dialog.brush_width.value() > 0

    def test_pick_metadata_has_no_size_entry(self, brush_view):
        _paint(brush_view, STROKE_A)
        pick_info = brush_view._build_base_pick_info()
        assert pick_info["Pick Shape"] == "Brush"
        assert pick_info["Number of picks"] == 1
        assert not any("Pick Diameter" in key for key in pick_info)
        assert len(pick_info["Pick Areas (um^2)"]) == 1

    def test_index_blocks_are_not_built(self, brush_view):
        _paint(brush_view, STROKE_A)
        assert brush_view.get_index_blocks(0) is None

    def test_pick_similar_is_refused(self, brush_view, monkeypatch):
        _paint(brush_view, STROKE_A)
        _paint(brush_view, STROKE_B)
        warned = []
        monkeypatch.setattr(
            gui_render.QtWidgets.QMessageBox,
            "warning",
            lambda *a, **k: warned.append(a),
        )
        brush_view.pick_similar()
        assert warned  # a painted region has no template to replicate


class TestBrushRotationWindow:
    def test_viewport_spans_the_painted_region(self):
        pick = [(2.0, [(0.0, 0.0), (10.0, 0.0)])]
        (y_min, x_min), (y_max, x_max) = (
            rotation.ViewRotation.fit_in_view_rotated(
                TestRotationWindowShapes._Stub(pick, "Brush", None),
                get_viewport=True,
            )
        )
        assert (x_min, x_max) == pytest.approx((-1.0, 11.0))
        assert (y_min, y_max) == pytest.approx((-1.0, 1.0))


class TestMultiChannelSavesKeepAllLocs:
    """Saving several channels into one file keeps every localization even
    when the channels differ in their columns (here: only one has a
    ``net_gradient``, as after a wavelet identification). ``pd.concat``
    alone would fill the missing column with NaN, and ``io.save_locs``
    would then drop all of that channel's localizations."""

    LEFT = [(1.0, 1.0), (12.0, 1.0), (12.0, 12.0), (1.0, 12.0), (1.0, 1.0)]
    RIGHT = [
        (18.0, 18.0),
        (30.0, 18.0),
        (30.0, 30.0),
        (18.0, 30.0),
        (18.0, 18.0),
    ]

    @pytest.fixture
    def two_channels(self, window, tmp_path):
        with_ng = _locs(seed=1)
        with_ng["net_gradient"] = np.full(len(with_ng), 5000.0, np.float32)
        window.view.add(
            str(tmp_path / "ng_locs.hdf5"), with_ng, _info(), render_=False
        )
        window.tools_settings_dialog.pick_shape.setCurrentText("Polygon")
        window.view._picks = [self.LEFT, self.RIGHT]
        return window

    @staticmethod
    def _n_picked(view, pick=None):
        return sum(
            len(picks[pick]) if pick is not None else sum(map(len, picks))
            for picks in (
                view.picked_locs(c, add_group=False)
                for c in range(len(view.locs_paths))
            )
        )

    def test_picked_locs_combined(self, two_channels, tmp_path):
        view = two_channels.view
        path = str(tmp_path / "picked_multi.hdf5")
        with pytest.warns(UserWarning, match="net_gradient"):
            view.save_picked_locs_multi(path)
        saved, _ = io.load_locs(path)
        assert len(saved) == self._n_picked(view) > 0
        assert "net_gradient" not in saved.columns

    def test_picked_locs_combined_per_pick(self, two_channels, tmp_path):
        view = two_channels.view
        path = str(tmp_path / "picked_sep.hdf5")
        with pytest.warns(UserWarning, match="net_gradient"):
            view.save_picked_locs_multi_sep(path)
        for i in range(2):
            saved, _ = io.load_locs(str(tmp_path / f"picked_sep_{i}.hdf5"))
            assert len(saved) == self._n_picked(view, i) > 0

    def test_all_channels_combined(self, two_channels, tmp_path, monkeypatch):
        window = two_channels
        path = str(tmp_path / "combined_multi.hdf5")
        n_channels = len(window.view.locs_paths)
        monkeypatch.setattr(
            window.view,
            "get_channel_save_locs",
            lambda *_: n_channels + 1,
        )
        monkeypatch.setattr(
            lib,
            "get_save_filename_ext_dialog",
            lambda *args, **kwargs: (path, ".hdf5"),
        )
        with pytest.warns(UserWarning, match="net_gradient"):
            window.save_locs()
        saved, _ = io.load_locs(path)
        assert len(saved) == sum(len(locs) for locs in window.view.locs)

    def test_the_3d_window_combines_the_same_way(self):
        import inspect

        source = inspect.getsource(rotation.RotationWindow.save_locs_rotated)
        assert "lib.concat_locs(self.window.view.locs)" in source
        assert "pd.concat" not in source
