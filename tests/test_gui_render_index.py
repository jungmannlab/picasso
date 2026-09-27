"""The spatial index caches of ``picasso.gui.render.View``.

``View.index_blocks`` (pick lookups) and ``View.render_index`` (the
viewport pyramid used when zoomed in) both store *positions* into
``View.locs[channel]``, so both go stale the moment a channel's
localizations change - moved coordinates as well as added, removed or
reordered rows. A stale ``render_index`` is invisible zoomed out
(``query_viewport`` bypasses itself on a full-FOV viewport) and empties
the image zoomed in, which is what "apply expression x += 94" used to
do.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from picasso.gui import render as gui_render

WIDTH = HEIGHT = 256.0


def _locs(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": 130.0,
        }
    ]


class _WindowStub:
    """A window without linked windows."""

    def link_channels_changed(self, changes) -> None:
        pass

    def link_render_index(self, locs) -> None:
        return None


class _ViewStub:
    """The parts of ``View`` the display path touches, and nothing else."""

    _display_indices = gui_render.View._display_indices
    _display_locs = gui_render.View._display_locs
    _viewport_indices = gui_render.View._viewport_indices
    _ensure_render_index = gui_render.View._ensure_render_index
    invalidate_locs_index = gui_render.View.invalidate_locs_index

    def __init__(self, locs: pd.DataFrame) -> None:
        self.locs = [locs]
        self.infos = [_info()]
        self.index_blocks = [None]
        self.render_index = [None]
        self._move_channels = ()  # no channel dragged by the Move tool
        self.window = _WindowStub()


def _brute_force(locs: pd.DataFrame, viewport) -> pd.DataFrame:
    (y_min, x_min), (y_max, x_max) = viewport
    inside = (
        (locs["x"] > x_min)
        & (locs["x"] < x_max)
        & (locs["y"] > y_min)
        & (locs["y"] < y_max)
    )
    return locs[inside]


class TestInvalidate:
    def test_single_channel(self):
        view = _ViewStub(_locs())
        view.locs.append(_locs(seed=1))
        view.infos.append(_info())
        view.index_blocks = ["a", "b"]
        view.render_index = ["c", "d"]

        view.invalidate_locs_index(1)

        assert view.index_blocks == ["a", None]
        assert view.render_index == ["c", None]

    def test_all_channels(self):
        view = _ViewStub(_locs())
        view.locs.append(_locs(seed=1))
        view.index_blocks = ["a", "b"]
        view.render_index = ["c", "d"]

        view.invalidate_locs_index()

        assert view.index_blocks == [None, None]
        assert view.render_index == [None, None]


class TestStaleIndex:
    """A shift applied to the locs must reach a zoomed-in viewport."""

    # small enough that ``query_viewport`` does not bypass itself
    VIEWPORT = ((40.0, 140.0), (70.0, 170.0))
    SHIFT = 94.0

    def test_zoomed_in_locs_follow_a_shift(self):
        view = _ViewStub(_locs())
        # build the pyramid on the original coordinates, the way a first
        # render at any zoom level does
        assert view._ensure_render_index(0) is not None

        view.locs[0]["x"] = view.locs[0]["x"] + self.SHIFT
        view.invalidate_locs_index(0)

        shown = view._display_locs(0, viewport=self.VIEWPORT)
        expected = _brute_force(view.locs[0], self.VIEWPORT)
        assert len(expected) > 0  # the test would be vacuous otherwise
        # the pyramid returns a superset - the renderer prunes the
        # overspill - so check containment rather than equality
        assert set(expected.index) <= set(shown.index)

    def test_without_invalidation_the_view_is_wrong(self):
        """Guards the guard: the assertion above fails on a stale index,
        so it really is testing the invalidation."""
        view = _ViewStub(_locs())
        assert view._ensure_render_index(0) is not None

        view.locs[0]["x"] = view.locs[0]["x"] + self.SHIFT
        # deliberately not invalidated

        shown = view._display_locs(0, viewport=self.VIEWPORT)
        expected = _brute_force(view.locs[0], self.VIEWPORT)
        assert not set(expected.index) <= set(shown.index)

    def test_row_removal_is_covered_too(self):
        """The pyramid stores positions, so dropping rows invalidates it
        even though no coordinate moved."""
        view = _ViewStub(_locs())
        assert view._ensure_render_index(0) is not None

        view.locs[0] = view.locs[0].iloc[::2].reset_index(drop=True)
        view.invalidate_locs_index(0)

        shown = view._display_locs(0, viewport=self.VIEWPORT)
        expected = _brute_force(view.locs[0], self.VIEWPORT)
        assert len(expected) > 0
        assert set(expected.index) <= set(shown.index)


# ---------------------------------------------------------------------------
# Every method that mutates the locs must invalidate
# ---------------------------------------------------------------------------

# Methods that assign to ``.locs`` without invalidating on purpose: they
# either build the list or only take a reference to it.
_EXEMPT = {
    ("View", "__init__"),  # creates the empty lists
    ("View", "add"),  # appends a fresh entry to every cache
    ("DatasetDialog", "_close_one_channel"),  # drops the entry everywhere
    # takes a linked window's channel together with its (valid) index
    ("View", "adopt_shared_channels"),
    ("View", "detach_channel"),  # a row-for-row copy: indices stay valid
    ("TestClustererDialog", "test_clusterer"),  # own copy, not the View's
    ("TestClustererView", "__init__"),  # own copy
    ("MaskSettingsDialog", "init_dialog"),  # reference, read only
    ("RESIDialog", "__init__"),  # reference, read only
}


_LIST_MUTATORS = ("append", "pop", "insert", "remove", "extend", "clear")


def _targets(node):
    """The assignment/deletion targets of ``node``, tuples unpacked."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = node.targets
    else:
        return []
    out = []
    for t in targets:
        if isinstance(t, (ast.Tuple, ast.List)):
            out.extend(t.elts)
        else:
            out.append(t)
    return out


def _touches_locs(node) -> bool:
    """True if ``node`` writes to ``<something>.locs``, an element of it,
    or a column of one of its DataFrames."""
    for t in _targets(node):
        if ast.unparse(t).split("[")[0].endswith(".locs"):
            return True
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        head, _, attr = func.rpartition(".")
        if attr in _LIST_MUTATORS and head.endswith(".locs"):
            return True
    return False


def _mutating_methods():
    """``(class, method, node)`` for every method in ``gui.render`` that
    changes ``<something>.locs`` - the list, one of its DataFrames, or a
    column of one of them."""
    tree = ast.parse(inspect.getsource(gui_render))
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_touches_locs(node) for node in ast.walk(fn)):
                yield cls.name, fn.name, fn


@pytest.mark.parametrize(
    "cls_name,fn_name,node",
    [pytest.param(c, f, n, id=f"{c}.{f}") for c, f, n in _mutating_methods()],
)
def test_locs_mutation_invalidates_the_index(cls_name, fn_name, node):
    """Assigning to ``self.locs`` and leaving the cached indices in place
    renders the wrong localizations at any zoom level past the full-FOV
    bypass. Add to ``_EXEMPT`` only for a method that does not change any
    channel's localizations."""
    if (cls_name, fn_name) in _EXEMPT:
        pytest.skip("exempt by construction")
    source = ast.unparse(node)
    # View.locs_moved invalidates the indices (and is checked here too)
    assert (
        "invalidate_locs_index" in source
        or "resample_locs=True" in source
        or "locs_moved(" in source
    ), (
        f"{cls_name}.{fn_name} mutates the localizations without dropping "
        "the cached spatial indices: call invalidate_locs_index(channel), "
        "update_scene(resample_locs=True) or locs_moved(channel)"
    )
