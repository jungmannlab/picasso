"""Localize's wiring of the ``roi_id`` column.

``localize.add_roi_id`` does the naming; the window decides when it
applies (not in split-FOV mode, where each region is a channel of its own
and gets its own file), *when* it runs (before drift correction moves the
coordinates the ids come from) and that the column reaches the file - it
has no checkbox in the column-selection dialog, since only a fit
restricted to ROIs has it, so ``select_locs_columns`` has to keep it
explicitly or it would be dropped on the way to disk.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pandas as pd

from picasso import localize
from picasso.gui import localize as gui_localize


# two disjoint rectangles, [[y_min, x_min], [y_max, x_max]]
ROIS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]


def _locs() -> pd.DataFrame:
    """Two localizations in the left ROI, one in the right, one outside
    both."""
    return pd.DataFrame(
        {
            "frame": np.arange(4, dtype=np.uint32),
            "x": np.array([1.5, 63.4, 64.6, 200.0], dtype=np.float32),
            "y": np.array([1.5, 10.0, 10.0, 10.0], dtype=np.float32),
            "photons": np.full(4, 1000.0, dtype=np.float32),
        }
    )


class _CheckboxStub:
    def isChecked(self) -> bool:
        return True


def _window(
    locs: pd.DataFrame,
    rois: list | None,
    split_fov: bool = False,
    checked: tuple[str, ...] = ("frame", "x", "y", "photons"),
):
    """The parts of ``Window`` the ROI id touches, and nothing else."""
    window = types.SimpleNamespace(
        locs=locs,
        view=types.SimpleNamespace(split_fov_mode=split_fov),
        last_identification_info=None if rois is None else {"ROI": rois},
        columns_dialog=types.SimpleNamespace(
            column_checkboxes={column: _CheckboxStub() for column in checked}
        ),
    )
    for name in ("attach_roi_id", "select_locs_columns"):
        setattr(
            window,
            name,
            getattr(gui_localize.Window, name).__get__(window),
        )
    return window


class TestAttachRoiId:
    def test_the_rois_of_the_identification_are_named(self):
        window = _window(_locs(), ROIS)

        window.attach_roi_id()

        assert list(window.locs["roi_id"]) == [0, 0, 1, localize.NO_ROI_ID]

    def test_no_rois_no_column(self):
        window = _window(_locs(), [])

        window.attach_roi_id()

        assert localize.ROI_ID_COLUMN not in window.locs.columns

    def test_no_identification_info_no_column(self):
        window = _window(_locs(), None)

        window.attach_roi_id()

        assert localize.ROI_ID_COLUMN not in window.locs.columns

    def test_split_fov_regions_are_not_named(self):
        """Split-FOV regions are separate channels saved to separate
        files, so they get no ROI id."""
        window = _window(_locs(), ROIS, split_fov=True)

        window.attach_roi_id()

        assert localize.ROI_ID_COLUMN not in window.locs.columns

    def test_the_id_is_attached_before_the_first_save(self):
        """Drift correction moves x/y by up to several pixels, which
        would push localizations near an ROI seam into the neighboring
        rectangle - so the ids must be derived before it runs."""
        source = inspect.getsource(gui_localize.Window.save_locs_after_fit)
        assert source.index("attach_roi_id") < source.index("drift_correction")


class TestColumnSelection:
    def test_the_id_survives_the_column_selection(self):
        window = _window(_locs(), ROIS)
        window.attach_roi_id()

        window.select_locs_columns()

        assert list(window.locs.columns) == [
            "frame",
            "x",
            "y",
            "photons",
            "roi_id",
        ]

    def test_unchecked_columns_still_go(self):
        """Guards the guard: the id is kept by name, not because the
        selection happens to keep everything."""
        window = _window(_locs(), ROIS, checked=("frame", "x", "y"))
        window.attach_roi_id()

        window.select_locs_columns()

        assert "photons" not in window.locs.columns
        assert "roi_id" in window.locs.columns
