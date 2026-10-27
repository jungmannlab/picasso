"""Selections in the histograms of the Filter window.

Covers filtering by dragging a range in a 1D histogram and a rectangle
in a 2D histogram, and stopping the drag with Esc, which filters
nothing.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import pytest
from matplotlib.backend_bases import MouseEvent
from PyQt6 import QtCore, QtGui

from picasso.gui import filter as gui_filter

LOCS_PATH = "./tests/data/testdata_locs.hdf5"


@pytest.fixture
def window(qt_offscreen, tmp_path, monkeypatch):
    """A Filter window with the test localizations open; user settings
    go to ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    window = gui_filter.Window()
    window.open(LOCS_PATH)
    yield window
    window.close()


def _mouse(selector, name, xdata, ydata) -> None:
    """Send a mouse event at data coordinates of the selector's axes."""
    x, y = selector.ax.transData.transform((xdata, ydata))
    MouseEvent(name, selector.canvas, x, y, button=1)._process()


def _drag(selector, start, end, *, cancel=None) -> None:
    """Drag from ``start`` to ``end``, calling ``cancel`` midway. A y
    of None is the middle of the axes."""
    if start[1] is None:
        middle_y = sum(selector.ax.get_ylim()) / 2
        start, end = (start[0], middle_y), (end[0], middle_y)
    _mouse(selector, "button_press_event", *start)
    middle = [(a + b) / 2 for a, b in zip(start, end)]
    _mouse(selector, "motion_notify_event", *middle)
    if cancel is not None:
        cancel()
    _mouse(selector, "motion_notify_event", *end)
    _mouse(selector, "button_release_event", *end)


def _esc(plot_window):
    """The Esc shortcut of ``plot_window``."""
    (shortcut,) = plot_window.findChildren(QtGui.QShortcut)
    assert shortcut.key() == QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape)
    return shortcut.activated.emit


@pytest.fixture
def hist(window):
    hist = gui_filter.HistWindow(window, "sx")
    window.hist_windows["sx"] = hist
    yield hist
    hist.close()


@pytest.fixture
def hist2d(window):
    hist2d = gui_filter.Hist2DWindow(window, "sx", "photons")
    window.hist2d_windows["sx"]["photons"] = hist2d
    yield hist2d
    hist2d.close()


def test_a_range_drawn_in_a_histogram_filters(window, hist):
    n = window.n_filtered
    _drag(hist.span, (107.0, None), (110.0, None))
    assert 0 < window.n_filtered < n
    sx = window.get_column("sx")
    assert sx.min() > 107.0 and sx.max() < 110.0


def test_esc_stops_drawing_a_range(window, hist):
    n = window.n_filtered
    _drag(hist.span, (107.0, None), (110.0, None), cancel=_esc(hist))
    assert window.n_filtered == n
    assert window._undo_stack == []
    assert not hist.span.artists[0].get_visible()
    # the next selection works as before
    _drag(hist.span, (107.0, None), (110.0, None))
    assert window.n_filtered < n


def test_esc_stops_drawing_a_rectangle(window, hist2d):
    n = window.n_filtered
    rectangle = hist2d.selector
    start, end = (107.0, 10000.0), (110.0, 20000.0)
    _drag(rectangle, start, end, cancel=_esc(hist2d))
    assert window.n_filtered == n
    _drag(rectangle, start, end)
    assert 0 < window.n_filtered < n


def test_esc_without_a_selection_does_nothing(window, hist2d):
    n = window.n_filtered
    _esc(hist2d)()
    assert window.n_filtered == n
