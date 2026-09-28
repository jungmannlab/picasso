"""Test ``picasso.localize`` — spot identification, extraction, and the
high-level ``fit``/``fit_async`` MLE wrapper, plus the diagnostic
helpers.

Tests for ``gausslq``, ``gaussmle`` and ``zfit`` live in their own files
(``test_gausslq.py``, ``test_gaussmle.py``, ``test_zfit.py``).

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import inspect
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from scipy.interpolate import CubicSpline
from PyQt6 import QtCore, QtWidgets

from picasso import gaussmle, gausslq, io, lib, localize, spline, transforms
from picasso import transforms as transforms_mod
from picasso.fitting import gaussfit_cuda, precision, seeds, splinefit
from picasso.gui import localize as localize_gui

from tests.conftest import (
    BOX,
    CALIB_3D,
    CAMERA_INFO,
    IDENTITY,
    MIN_NG,
    PIXELSIZE,
    affine,
    affine_matrix,
    affine_matrix_3x3,
    apply_transform,
    linear_part,
)

CAMERA_INFO_WITH_PIXELSIZE = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}

# Devices a spline-CRLB test can be pinned to (see ``_crlb``). The GPU variant
# needs a real CUDA device; ``NUMBA_ENABLE_CUDASIM=1`` also satisfies it, which
# is how the kernels can be exercised on a machine without one.
SPLINE_CRLB_DEVICES = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not localize.CUDA_AVAILABLE,
            reason="no CUDA device for the spline CRLB kernels",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient_meshgrid(box: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the normalized (uy, ux) direction vectors that
    ``identify_in_image`` constructs internally — needed to drive
    ``net_gradient`` directly. The center pixel is unused by
    ``net_gradient`` (it is skipped inside the loop) but its norm is 0,
    so we patch it to 1.0 to avoid emitting a divide-by-zero warning."""
    box_half = box // 2
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    unorm[box_half, box_half] = 1.0
    ux /= unorm
    uy /= unorm
    return uy, ux


def _gaussian_frame(
    shape: tuple[int, int],
    center: tuple[int, int],
    sigma: float = 1.2,
    amplitude: float = 5000.0,
    background: float = 100.0,
) -> np.ndarray:
    """Build a single-Gaussian-peak frame on a flat background. Returned
    as float32 so it can be passed straight into the numba-jitted
    helpers without numba complaining about the dtype."""
    Y, X = shape
    cy, cx = center
    yy, xx = np.indices((Y, X), dtype=np.float32)
    g = amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return (g + background).astype(np.float32)


# ---------------------------------------------------------------------------
# _local_maxima
# ---------------------------------------------------------------------------


class TestLocalMaxima:
    """Pure local-maxima search inside a sliding box."""

    def test_single_peak_detected(self):
        frame = np.zeros((20, 20), dtype=np.float32)
        frame[10, 12] = 100.0
        y, x = localize._local_maxima(frame, BOX)
        assert list(zip(y.tolist(), x.tolist())) == [(10, 12)]

    def test_multiple_peaks_far_apart_all_found(self):
        frame = np.zeros((30, 30), dtype=np.float32)
        peaks = [(8, 8), (8, 22), (22, 15)]
        for py, px in peaks:
            frame[py, px] = 50.0
        y, x = localize._local_maxima(frame, BOX)
        found = set(zip(y.tolist(), x.tolist()))
        assert found == set(peaks)

    def test_peaks_in_border_band_are_excluded(self):
        """``_local_maxima`` only scans i in [box_half, Y - box_half - 1)
        — peaks placed inside the border band must not be returned."""
        Y = X = 20
        box_half = BOX // 2
        frame = np.zeros((Y, X), dtype=np.float32)
        frame[1, 1] = 100.0  # top-left border
        frame[Y - 2, X - 2] = 100.0  # bottom-right border
        y, x = localize._local_maxima(frame, BOX)
        # All returned coordinates lie strictly inside the scan band
        assert ((y >= box_half) & (y < Y - box_half - 1)).all()
        assert ((x >= box_half) & (x < X - box_half - 1)).all()

    def test_flat_frame_returns_no_maxima(self):
        """A constant frame has no unique local max — the implementation's
        ``argmax`` returns the top-left pixel (index 0), so no local
        window has its max at the center."""
        frame = np.full((20, 20), 42.0, dtype=np.float32)
        y, x = localize._local_maxima(frame, BOX)
        assert len(y) == 0 and len(x) == 0


# ---------------------------------------------------------------------------
# _gradient_at
# ---------------------------------------------------------------------------


class TestGradientAt:
    """Two-point centered finite difference at (y, x)."""

    def test_horizontal_gradient(self):
        # frame[y, x+1] - frame[y, x-1]  along increasing x
        frame = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        gy, gx = localize._gradient_at(frame, 5, 5, 0)
        assert gy == 0.0
        assert gx == 2.0  # 6 - 4

    def test_vertical_gradient(self):
        # frame[y+1, x] - frame[y-1, x]  along increasing y
        frame = np.tile(
            np.arange(10, dtype=np.float32).reshape(-1, 1), (1, 10)
        )
        gy, gx = localize._gradient_at(frame, 5, 5, 0)
        assert gy == 2.0  # 6 - 4
        assert gx == 0.0

    def test_zero_gradient_in_flat_region(self):
        frame = np.full((10, 10), 7.0, dtype=np.float32)
        gy, gx = localize._gradient_at(frame, 5, 5, 0)
        assert gy == 0.0 and gx == 0.0

    def test_i_argument_is_ignored(self):
        """``i`` is documented as unused — different values must not
        affect the returned gradient."""
        frame = _gaussian_frame((15, 15), (7, 7))
        a = localize._gradient_at(frame, 7, 8, 0)
        b = localize._gradient_at(frame, 7, 8, 999)
        assert a == b


# ---------------------------------------------------------------------------
# net_gradient
# ---------------------------------------------------------------------------


class TestNetGradient:
    """Inner-product of the local gradient field with the radial
    direction vectors — peaks point outward, so the dot product is large
    and positive at a true peak."""

    def test_gaussian_peak_has_positive_net_gradient(self):
        frame = _gaussian_frame((15, 15), (7, 7))
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        assert ng.shape == (1,)
        assert ng[0] > 0

    def test_flat_frame_yields_zero(self):
        frame = np.full((15, 15), 50.0, dtype=np.float32)
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        np.testing.assert_allclose(ng, [0.0], atol=1e-6)

    def test_inverted_peak_yields_negative(self):
        """A dip (gradients pointing inward) gives a negative net
        gradient — the sign is the discriminator between peaks and
        troughs."""
        frame = -_gaussian_frame((15, 15), (7, 7), background=0.0)
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        assert ng[0] < 0

    def test_output_length_matches_input(self):
        frame = _gaussian_frame((30, 30), (10, 10))
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([10, 10, 10], dtype=np.int64)
        x = np.array([8, 10, 12], dtype=np.int64)
        ng = localize._net_gradient(frame, y, x, BOX, uy, ux)
        assert ng.shape == (3,)


# ---------------------------------------------------------------------------
# identify_in_image
# ---------------------------------------------------------------------------


class TestIdentifyInImage:
    """``_local_maxima`` + net-gradient threshold, in one shot."""

    def test_single_gaussian_is_identified(self):
        frame = _gaussian_frame((20, 20), (10, 10), amplitude=5000.0)
        y, x, ng = localize.identify_in_image(frame, 1.0, BOX)
        # One detection at the seeded peak
        assert len(y) == 1 == len(x) == len(ng)
        assert y[0] == 10 and x[0] == 10
        assert ng[0] > 1.0

    def test_high_threshold_rejects_all(self):
        frame = _gaussian_frame((20, 20), (10, 10), amplitude=5000.0)
        y, x, ng = localize.identify_in_image(frame, 1e12, BOX)
        assert len(y) == 0 and len(x) == 0 and len(ng) == 0

    def test_arrays_have_consistent_length(self):
        frame = _gaussian_frame((30, 30), (10, 10))
        # Add a second well-separated peak
        frame2 = _gaussian_frame((30, 30), (20, 22))
        combined = np.maximum(frame, frame2)
        y, x, ng = localize.identify_in_image(combined, 1.0, BOX)
        assert len(y) == len(x) == len(ng)
        assert len(y) >= 2

    def test_flat_frame_returns_empty(self):
        frame = np.full((20, 20), 100.0, dtype=np.float32)
        y, x, ng = localize.identify_in_image(frame, 0.0, BOX)
        assert len(y) == 0


# ---------------------------------------------------------------------------
# identify_in_frame
# ---------------------------------------------------------------------------


class TestIdentifyInFrame:
    """Wrapper that casts to float32 and applies an ROI offset."""

    def test_no_roi_matches_identify_in_image(self):
        frame = _gaussian_frame((20, 20), (10, 10)).astype(np.int32)
        y_a, x_a, ng_a = localize.identify_in_frame(frame, 1.0, BOX)
        y_b, x_b, ng_b = localize.identify_in_image(
            np.float32(frame), 1.0, BOX
        )
        np.testing.assert_array_equal(y_a, y_b)
        np.testing.assert_array_equal(x_a, x_b)
        np.testing.assert_allclose(ng_a, ng_b)

    def test_roi_offsets_coordinates_back_to_global(self):
        """When ROI = ((y0, x0), (y1, x1)) is supplied, returned (y, x)
        are in the *original* frame's coordinate system, not the ROI's."""
        # Peak at global (15, 17); ROI starts at (10, 12)
        frame = _gaussian_frame((30, 30), (15, 17)).astype(np.int32)
        roi = ((10, 12), (25, 28))
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=roi)
        assert len(y) == 1
        assert (int(y[0]), int(x[0])) == (15, 17)

    def test_roi_excludes_peaks_outside(self):
        """A peak outside the ROI window is not seen at all."""
        frame = _gaussian_frame((30, 30), (5, 5)).astype(np.int32)
        roi = ((15, 15), (28, 28))
        y, x, ng = localize.identify_in_frame(frame, 1.0, BOX, roi=roi)
        assert len(y) == 0 and len(x) == 0 and len(ng) == 0

    def test_multiple_rois_find_all_peaks(self):
        """A list of disjoint ROIs collects peaks from every region and
        ignores peaks outside all of them."""
        frame = _gaussian_frame((40, 40), (8, 8)).astype(np.int32)
        frame += (_gaussian_frame((40, 40), (30, 30)) - 100).astype(np.int32)
        frame += (_gaussian_frame((40, 40), (8, 30)) - 100).astype(np.int32)
        rois = [((0, 0), (16, 16)), ((24, 24), (38, 38))]
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=rois)
        found = {(int(yi), int(xi)) for yi, xi in zip(y, x)}
        assert (8, 8) in found  # first ROI
        assert (30, 30) in found  # second ROI
        assert (8, 30) not in found  # outside both ROIs

    def test_multiple_rois_no_double_counting(self):
        """Disjoint ROIs never report the same peak twice."""
        frame = _gaussian_frame((40, 40), (8, 8)).astype(np.int32)
        frame += (_gaussian_frame((40, 40), (30, 30)) - 100).astype(np.int32)
        rois = [((0, 0), (16, 16)), ((24, 24), (38, 38))]
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=rois)
        coords = list(zip([int(v) for v in y], [int(v) for v in x]))
        assert len(coords) == len(set(coords))

    def test_peak_near_roi_border_is_found(self):
        """A peak close to the ROI border is detected: the slice is
        padded internally so the gradient box still sees real pixels."""
        # Peak two pixels inside the bottom-right corner of the ROI.
        frame = _gaussian_frame((40, 40), (18, 18)).astype(np.int32)
        roi = ((5, 5), (20, 20))
        y, x, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=roi)
        found = {(int(yi), int(xi)) for yi, xi in zip(y, x)}
        assert (18, 18) in found

    def test_adjacent_rois_have_no_seam_gap(self):
        """Splitting a region into two ROIs that share an edge must find
        the same peaks as the single, undivided region - no gap of
        ~``box`` pixels along the seam (regression test)."""
        # Two peaks straddling the y = 20 seam, each only two pixels away
        # from it (i.e. within ``box`` pixels) and far apart in x so they
        # do not suppress one another.
        centers = [(18, 12), (22, 28)]
        frame = np.full((40, 40), 100, dtype=np.int32)
        for cy, cx in centers:
            frame += (_gaussian_frame((40, 40), (cy, cx)) - 100).astype(
                np.int32
            )
        whole = ((8, 8), (32, 32))
        split = [((8, 8), (20, 32)), ((20, 8), (32, 32))]
        y_w, x_w, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=whole)
        y_s, x_s, _ = localize.identify_in_frame(frame, 1.0, BOX, roi=split)
        found_whole = {(int(a), int(b)) for a, b in zip(y_w, x_w)}
        found_split = {(int(a), int(b)) for a, b in zip(y_s, x_s)}
        # every seeded peak is found in both cases ...
        for c in centers:
            assert c in found_whole
            assert c in found_split
        # ... and the two ROIs together reproduce the single-region result
        assert found_whole == found_split


# ---------------------------------------------------------------------------
# clip_rois
# ---------------------------------------------------------------------------


def _area(rects) -> int:
    """Total area of a list of [[y0, x0], [y1, x1]] rectangles."""
    return sum((y1 - y0) * (x1 - x0) for (y0, x0), (y1, x1) in rects)


def _overlap(a, b) -> int:
    """Area of the overlap between two [[y0, x0], [y1, x1]] rectangles."""
    (ay0, ax0), (ay1, ax1) = a
    (by0, bx0), (by1, bx1) = b
    dy = max(0, min(ay1, by1) - max(ay0, by0))
    dx = max(0, min(ax1, bx1) - max(ax0, bx0))
    return dy * dx


class TestClipRois:
    """Geometric clipping of (possibly overlapping) ROIs into disjoint
    rectangles."""

    def test_disjoint_unchanged(self):
        rois = [((0, 0), (5, 5)), ((10, 10), (15, 15))]
        out = localize.clip_rois(rois)
        assert out == [[[0, 0], [5, 5]], [[10, 10], [15, 15]]]

    def test_overlap_is_disjoint_and_preserves_union(self):
        rois = [((0, 0), (10, 10)), ((5, 5), (15, 15))]
        out = localize.clip_rois(rois)
        # pairwise disjoint
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                assert _overlap(out[i], out[j]) == 0
        # union area = 100 + 100 - 25 (overlap)
        assert _area(out) == 175

    def test_full_containment_drops_inner(self):
        rois = [((0, 0), (20, 20)), ((5, 5), (10, 10))]
        out = localize.clip_rois(rois)
        assert out == [[[0, 0], [20, 20]]]

    def test_corner_overlap(self):
        rois = [((0, 0), (10, 10)), ((8, 8), (18, 18))]
        out = localize.clip_rois(rois)
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                assert _overlap(out[i], out[j]) == 0
        assert _area(out) == 100 + 100 - 4

    def test_min_size_drops_slivers(self):
        # second ROI overlaps the first leaving a 1-pixel-tall sliver
        rois = [((0, 0), (10, 10)), ((9, 0), (20, 20))]
        out = localize.clip_rois(rois, min_size=3)
        assert [[10, 0], [20, 20]] in out
        # the 1-pixel band (y in 9..10) is discarded
        assert all(piece[1][0] - piece[0][0] >= 3 for piece in out)

    def test_normalizes_corner_order(self):
        out = localize.clip_rois([((25, 28), (10, 12))])
        assert out == [[[10, 12], [25, 28]]]


# ---------------------------------------------------------------------------
# add_roi_id
# ---------------------------------------------------------------------------


def _roi_locs() -> pd.DataFrame:
    """Two localizations in the left half, one in the right, one outside
    both halves."""
    return pd.DataFrame(
        {
            "frame": np.arange(4, dtype=np.uint32),
            "x": np.array([1.5, 63.4, 64.6, 200.0], dtype=np.float32),
            "y": np.array([1.5, 10.0, 10.0, 10.0], dtype=np.float32),
            "photons": np.full(4, 1000.0, dtype=np.float32),
        }
    )


class TestAddRoiId:
    """The ``roi_id`` column of a fit restricted to several ROIs."""

    ROIS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]

    def test_the_index_of_the_roi_is_stored(self):
        out = localize.add_roi_id(_roi_locs(), self.ROIS)
        assert out.columns[-1] == localize.ROI_ID_COLUMN
        assert list(out["roi_id"]) == [0, 0, 1, localize.NO_ROI_ID]
        assert out["roi_id"].dtype == np.int32

    def test_the_other_columns_are_untouched(self):
        locs = _roi_locs()
        out = localize.add_roi_id(locs, self.ROIS)
        pd.testing.assert_frame_equal(out[locs.columns], locs)
        assert "roi_id" not in locs.columns  # the input is not mutated

    def test_every_localization_gets_exactly_one_index(self):
        rois = localize.clip_rois([[[0, 0], [64, 128]], [[0, 32], [64, 96]]])
        out = localize.add_roi_id(_roi_locs(), rois)
        ids = out["roi_id"]
        assert ids.between(localize.NO_ROI_ID, len(rois) - 1).all()

    def test_overlapping_rectangles_resolve_to_the_first(self):
        """``clip_rois`` does not produce overlaps, but a hand-written
        ``roi`` argument may - one index per localization either way."""
        out = localize.add_roi_id(
            _roi_locs(), [[[0, 0], [64, 128]], [[0, 0], [64, 64]]]
        )
        assert list(out["roi_id"]) == [0, 0, 0, localize.NO_ROI_ID]

    def test_a_single_rectangle_is_accepted(self):
        """The whole ``roi`` argument of ``identify``/``localize``, not
        just its list form."""
        out = localize.add_roi_id(_roi_locs(), ((0, 0), (64, 64)))
        assert list(out["roi_id"]) == [0, 0, -1, -1]

    def test_corner_order_does_not_matter(self):
        out = localize.add_roi_id(_roi_locs(), [((64, 64), (0, 0))])
        assert list(out["roi_id"]) == [0, 0, -1, -1]

    @pytest.mark.parametrize("roi", [None, []])
    def test_no_roi_no_column(self, roi):
        locs = _roi_locs()
        out = localize.add_roi_id(locs, roi)
        assert list(out.columns) == list(locs.columns)


# ---------------------------------------------------------------------------
# _to_photons
# ---------------------------------------------------------------------------


class TestToPhotons:
    """Camera-signal -> photon-count conversion: (s - baseline) * sens / gain."""

    def test_identity_camera_returns_input(self):
        spots = np.arange(2 * BOX * BOX, dtype=np.float32).reshape(2, BOX, BOX)
        out = localize._to_photons(spots, CAMERA_INFO)
        np.testing.assert_allclose(out, spots)

    def test_baseline_subtracts(self):
        spots = np.full((2, BOX, BOX), 500.0, dtype=np.float32)
        cam = {"Baseline": 100, "Sensitivity": 1, "Gain": 1}
        out = localize._to_photons(spots, cam)
        np.testing.assert_allclose(out, 400.0)

    def test_sensitivity_multiplies(self):
        spots = np.full((2, BOX, BOX), 50.0, dtype=np.float32)
        cam = {"Baseline": 0, "Sensitivity": 3, "Gain": 1}
        out = localize._to_photons(spots, cam)
        np.testing.assert_allclose(out, 150.0)

    def test_gain_divides(self):
        spots = np.full((2, BOX, BOX), 60.0, dtype=np.float32)
        cam = {"Baseline": 0, "Sensitivity": 1, "Gain": 3}
        out = localize._to_photons(spots, cam)
        np.testing.assert_allclose(out, 20.0)

    def test_combined_transform(self):
        spots = np.full((1, BOX, BOX), 1000.0, dtype=np.float32)
        cam = {"Baseline": 100, "Sensitivity": 2, "Gain": 4}
        out = localize._to_photons(spots, cam)
        # (1000 - 100) * 2 / 4 = 450
        np.testing.assert_allclose(out, 450.0)

    def test_output_is_float32(self):
        spots = np.ones((1, BOX, BOX), dtype=np.uint16) * 100
        out = localize._to_photons(spots, CAMERA_INFO)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# identify
# ---------------------------------------------------------------------------


class TestIdentify:
    """Spot identification on the bundled .raw movie."""

    def test_required_columns_and_finite(self, real_identifications, movie):
        ids = real_identifications
        assert not ids.empty
        for col in ["frame", "x", "y", "net_gradient"]:
            assert col in ids.columns
        assert (ids["net_gradient"] >= MIN_NG).all()
        assert ids["frame"].min() >= 0
        assert ids["frame"].max() < len(movie)

    def test_x_y_inside_movie_bounds(self, real_identifications, movie):
        _, height, width = movie.shape
        ids = real_identifications
        assert (ids["x"] >= 0).all() and (ids["x"] < width).all()
        assert (ids["y"] >= 0).all() and (ids["y"] < height).all()

    def test_roi_is_strict_subset(self, movie, real_identifications):
        """ROI restricts identifications to that pixel window only."""
        roi = ((0, 0), (16, 16))  # ((y_start, x_start), (y_end, x_end))
        ids_roi = localize.identify(
            movie, MIN_NG, BOX, roi=roi, return_info=False
        )
        if len(ids_roi):
            assert (ids_roi["x"] < 16).all()
            assert (ids_roi["y"] < 16).all()
        # subset relationship — ROI cannot find more spots than full image
        assert len(ids_roi) <= len(real_identifications)

    def test_threaded_matches_serial_on_record_set(self, movie):
        """The (frame, y, x) sets identified threaded vs. serial must
        match exactly (order-independent)."""
        ids_t = localize.identify(
            movie, MIN_NG, BOX, threaded=True, return_info=False
        )
        ids_s = localize.identify(
            movie, MIN_NG, BOX, threaded=False, return_info=False
        )
        # Compare as set of (frame, y, x) tuples — same spots, possibly
        # different row order
        set_t = set(zip(ids_t["frame"], ids_t["y"], ids_t["x"]))
        set_s = set(zip(ids_s["frame"], ids_s["y"], ids_s["x"]))
        assert set_t == set_s

    def test_frame_bounds_excludes_outside(self, movie):
        """Setting ``frame_bounds`` confines identifications to that
        range of frame indices."""
        ids = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=(20, 50), return_info=False
        )
        if len(ids):
            assert (ids["frame"] >= 20).all()
            assert (ids["frame"] <= 50).all()

    def test_frame_bounds_multiple_segments(self, movie):
        """A list of ``(min, max)`` segments confines identifications to
        the union of those (disjoint) frame ranges."""
        segments = [(10, 20), (40, 50)]
        ids = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=segments, return_info=False
        )
        if len(ids):
            in_any = ((ids["frame"] >= 10) & (ids["frame"] <= 20)) | (
                (ids["frame"] >= 40) & (ids["frame"] <= 50)
            )
            assert in_any.all()
            # nothing in the gap between segments
            assert not ((ids["frame"] > 20) & (ids["frame"] < 40)).any()

    def test_frame_bounds_single_segment_matches_flat_tuple(self, movie):
        """A single-segment list behaves identically to the flat
        ``(min, max)`` tuple form."""
        flat = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=(20, 50), return_info=False
        )
        listed = localize.identify(
            movie, MIN_NG, BOX, frame_bounds=[(20, 50)], return_info=False
        )
        flat_set = set(zip(flat["frame"], flat["y"], flat["x"]))
        listed_set = set(zip(listed["frame"], listed["y"], listed["x"]))
        assert flat_set == listed_set

    def test_return_info_returns_metadata_dict(self, movie):
        ids, info = localize.identify(
            movie, MIN_NG, BOX, return_info=True, threaded=False
        )
        assert isinstance(info, dict)
        for key in [
            "Generated by",
            "Min. Net Gradient",
            "Box Size",
            "ROI",
            "Frame Bounds",
        ]:
            assert key in info
        assert info["Min. Net Gradient"] == MIN_NG
        assert info["Box Size"] == BOX
        # ids itself is still a DataFrame
        assert isinstance(ids, pd.DataFrame)


class TestIdentifyAsync:
    """The thread-pool identification path."""

    @pytest.mark.slow
    def test_async_finishes_and_matches_serial(self, movie):
        current, fs = localize.identify_async(movie, MIN_NG, BOX)
        n_frames = len(movie)
        # Wait for completion
        t0 = time.time()
        while current[0] < n_frames:
            assert time.time() - t0 < 30, "identify_async timed out"
            time.sleep(0.05)
        ids_async = localize.identifications_from_futures(fs)
        ids_serial = localize.identify(
            movie, MIN_NG, BOX, threaded=False, return_info=False
        )
        set_a = set(zip(ids_async["frame"], ids_async["y"], ids_async["x"]))
        set_s = set(zip(ids_serial["frame"], ids_serial["y"], ids_serial["x"]))
        assert set_a == set_s


class TestIdentifyByFrameNumber:
    """Per-frame identification helper."""

    def test_subset_of_full_identify(self, movie, real_identifications):
        """Per-frame call returns the rows of ``identify`` for that
        frame, no more, no less."""
        for frame in [10, 30, 60]:
            single = localize.identify_by_frame_number(
                movie, MIN_NG, BOX, frame
            )
            full_subset = real_identifications[
                real_identifications["frame"] == frame
            ]
            single_set = set(zip(single["y"], single["x"]))
            full_set = set(zip(full_subset["y"], full_subset["x"]))
            assert (
                single_set == full_set
            ), f"frame={frame}: per-frame {single_set} != full {full_set}"

    def test_frame_outside_bounds_returns_empty(self, movie):
        out = localize.identify_by_frame_number(
            movie, MIN_NG, BOX, 5, frame_bounds=(20, 30)
        )
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_frame_in_one_of_several_segments(self, movie):
        """A frame inside any of several segments is processed; a frame
        in the gap between segments returns empty."""
        segments = [(0, 4), (20, 30)]
        inside = localize.identify_by_frame_number(
            movie, MIN_NG, BOX, 25, frame_bounds=segments
        )
        gap = localize.identify_by_frame_number(
            movie, MIN_NG, BOX, 10, frame_bounds=segments
        )
        # frame 25 matches the full per-frame identification...
        full = localize.identify_by_frame_number(movie, MIN_NG, BOX, 25)
        assert set(zip(inside["y"], inside["x"])) == set(
            zip(full["y"], full["x"])
        )
        # ...while frame 10 (in the gap) yields nothing
        assert len(gap) == 0


# ---------------------------------------------------------------------------
# picks_to_identifications / locs_to_identifications
# ---------------------------------------------------------------------------


class TestPicksToIdentifications:
    """Convert circular picks into identification rows."""

    def test_basic(self):
        picks = [(5.5, 5.5), (15.5, 15.5), (25.5, 25.5)]
        n_frames = 10
        ids = localize.picks_to_identifications(picks, n_frames=n_frames)
        # 3 picks * 10 frames = 30 rows
        assert len(ids) == len(picks) * n_frames
        for col in ["frame", "x", "y", "net_gradient", "n_id"]:
            assert col in ids.columns

    def test_each_pick_present_in_all_frames(self):
        picks = [(5.5, 5.5), (15.5, 15.5)]
        n_frames = 4
        ids = localize.picks_to_identifications(picks, n_frames=n_frames)
        # Every frame must contain one row per pick
        for f in range(n_frames):
            assert (ids["frame"] == f).sum() == len(picks)

    def test_drift_applied_to_positions(self):
        picks = [(5.5, 5.5)]
        drift = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, -1.0, -2.0]})
        ids = localize.picks_to_identifications(picks, drift=drift)
        # Per-frame x/y reflects pick + drift
        ids_sorted = ids.sort_values("frame").reset_index(drop=True)
        np.testing.assert_allclose(
            ids_sorted["x"], [5.5 + 0.0, 5.5 + 1.0, 5.5 + 2.0]
        )
        np.testing.assert_allclose(
            ids_sorted["y"], [5.5 + 0.0, 5.5 - 1.0, 5.5 - 2.0]
        )

    def test_no_n_frames_no_drift_raises(self):
        with pytest.raises(ValueError):
            localize.picks_to_identifications([(1.0, 2.0)])

    def test_non_circular_picks_rejected(self):
        # Each pick must contain exactly two coordinates (circular pick);
        # 3-element picks are rejected.
        with pytest.raises(AssertionError):
            localize.picks_to_identifications([(1.0, 2.0, 3.0)], n_frames=5)

    def test_non_list_input_rejected(self):
        with pytest.raises(AssertionError):
            localize.picks_to_identifications("not a list", n_frames=5)


class TestLocsToIdentifications:
    """Round-trip locs back into identifications spanning a window of
    frames around each loc."""

    def test_columns_and_window_size(self, locs, info):
        n_frames = 2  # ±2 around each loc -> 5 rows per kept loc
        ids = localize.locs_to_identifications(
            locs.iloc[:5], info, n_frames=n_frames
        )
        for col in ["frame", "x", "y", "net_gradient", "n_id"]:
            assert col in ids.columns
        # Each kept loc contributes 2*n_frames + 1 rows
        unique_n_id = ids["n_id"].nunique()
        assert len(ids) == unique_n_id * (2 * n_frames + 1)

    def test_locs_near_movie_edges_excluded(self, locs, info):
        """Locs whose frame is within ``n_frames`` of the movie edges
        are skipped."""
        n_frames = 2
        # Build a locs frame with one "near edge" loc and one in the middle
        movie_frames = info[0]["Frames"]
        edge_locs = pd.DataFrame(
            {
                "frame": [0, 1, movie_frames // 2, movie_frames - 1],
                "x": [10.0, 10.0, 10.0, 10.0],
                "y": [10.0, 10.0, 10.0, 10.0],
            }
        )
        ids = localize.locs_to_identifications(
            edge_locs, info, n_frames=n_frames
        )
        # Only the middle loc passes the edge check
        assert ids["n_id"].nunique() == 1


# ---------------------------------------------------------------------------
# save_identifications / load_identifications (picasso.io)
# ---------------------------------------------------------------------------


class TestSaveLoadIdentifications:
    """HDF5 round-trip for the identifications DataFrame.

    The two functions are defined in ``picasso.io`` but exist solely to
    persist what the Localize GUI calls ``identifications`` (a DataFrame
    with columns ``frame``, ``x``, ``y``, ``net_gradient`` and optionally
    ``n_id``). They mirror the ``save_locs`` / ``load_locs`` pattern: an
    HDF5 file containing an ``"identifications"`` dataset and an
    accompanying ``.yaml`` sidecar with metadata.
    """

    @pytest.fixture(autouse=True)
    def _yaml_sidecar_on(self, monkeypatch):
        """Write the sidecar .yaml, whatever the developer's user
        settings say: ``io._save_metadata_in_yaml`` reads
        ``~/.picasso/settings.yaml``, where it can be turned off."""
        monkeypatch.setattr(io, "_save_metadata_in_yaml", lambda: True)

    def _info(self) -> list[dict]:
        return [
            {"Width": 32, "Height": 32, "Frames": 100},
            {
                "Generated by": "test_localize",
                "Box Size": BOX,
                "Min. Net Gradient": MIN_NG,
            },
        ]

    def test_roundtrip_real_identifications(
        self, tmp_path, real_identifications
    ):
        """Save identifications coming out of ``localize.identify``, then
        load them back — columns and content survive intact."""
        path = tmp_path / "test_identifications.hdf5"
        info = self._info()
        io.save_identifications(str(path), real_identifications, info)
        loaded, loaded_info = io.load_identifications(str(path))
        assert len(loaded) == len(real_identifications)
        assert set(loaded.columns) == set(real_identifications.columns)
        for col in ["frame", "x", "y", "net_gradient"]:
            np.testing.assert_array_equal(
                loaded[col].to_numpy(),
                real_identifications[col].to_numpy(),
            )
        assert loaded_info == info

    def test_roundtrip_picks_identifications(self, tmp_path):
        """``picks_to_identifications`` produces a DataFrame that also
        carries an ``n_id`` column — verify that round-trips too."""
        ids = localize.picks_to_identifications(
            [(5.5, 5.5), (15.5, 15.5)], n_frames=4
        )
        path = tmp_path / "picks_identifications.hdf5"
        io.save_identifications(str(path), ids, self._info())
        loaded, _ = io.load_identifications(str(path))
        assert "n_id" in loaded.columns
        assert len(loaded) == len(ids)
        np.testing.assert_array_equal(
            np.sort(loaded["n_id"].to_numpy()),
            np.sort(ids["n_id"].to_numpy()),
        )

    def test_yaml_sidecar_written(self, tmp_path, real_identifications):
        """``save_identifications`` must drop a YAML next to the HDF5 so
        ``load_identifications`` can recover the metadata."""
        path = tmp_path / "ids.hdf5"
        io.save_identifications(str(path), real_identifications, self._info())
        assert path.exists()
        assert (tmp_path / "ids.yaml").exists()

    def test_hdf5_dataset_key_is_identifications(
        self, tmp_path, real_identifications
    ):
        """The on-disk dataset key must be ``"identifications"`` — that's
        what ``load_identifications`` reads, and that's how a file is
        distinguished from a ``_locs.hdf5``."""
        path = tmp_path / "ids.hdf5"
        io.save_identifications(str(path), real_identifications, self._info())
        with h5py.File(path, "r") as f:
            assert "identifications" in f
            assert f["identifications"].shape == (len(real_identifications),)

    def test_load_missing_dataset_raises_keyerror(self, tmp_path):
        """Loading an HDF5 that has no ``identifications`` dataset (e.g.
        a ``_locs.hdf5``) must raise — silent fallback would let bad
        files masquerade as identifications."""
        path = tmp_path / "wrong.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "locs",
                data=pd.DataFrame({"x": [1.0]}).to_records(index=False),
            )
        # accompanying yaml so we hit the KeyError path, not NoMetadataFileError
        io.save_info(str(tmp_path / "wrong.yaml"), self._info())
        with pytest.raises(KeyError):
            io.load_identifications(str(path))

    def test_load_missing_yaml_falls_back_to_embedded(
        self, tmp_path, real_identifications
    ):
        """If the YAML sidecar is removed, ``load_identifications`` falls
        back to the metadata embedded in the HDF5 ``/metadata`` dataset
        (same contract as ``load_locs``)."""
        path = tmp_path / "ids.hdf5"
        info = self._info()
        io.save_identifications(str(path), real_identifications, info)
        (tmp_path / "ids.yaml").unlink()
        _, loaded_info = io.load_identifications(str(path))
        assert loaded_info == info

    def test_save_uses_yaml_sidecar_path(self, tmp_path, real_identifications):
        """The YAML path is derived from the HDF5 path's base name —
        verify that even when the HDF5 has a non-standard suffix, the
        YAML lands at ``<base>.yaml``."""
        path = tmp_path / "custom_name.hdf5"
        io.save_identifications(str(path), real_identifications, self._info())
        assert (tmp_path / "custom_name.yaml").exists()


# ---------------------------------------------------------------------------
# get_spots
# ---------------------------------------------------------------------------


class TestGetSpots:
    """Pixel patches around identified spots."""

    def test_shape_dtype(self, real_spots, real_identifications):
        n = len(real_identifications)
        assert real_spots.shape == (n, BOX, BOX)
        assert real_spots.dtype == np.float32

    def test_baseline_subtraction_via_camera_info(
        self, movie, real_identifications
    ):
        """Increasing the baseline subtracts from every spot pixel."""
        cam_a = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        cam_b = {"Baseline": 100, "Sensitivity": 1, "Gain": 1}
        spots_a = localize.get_spots(movie, real_identifications, BOX, cam_a)
        spots_b = localize.get_spots(movie, real_identifications, BOX, cam_b)
        np.testing.assert_allclose(spots_a - spots_b, 100.0)

    def test_sensitivity_scales_signal(self, movie, real_identifications):
        cam_x1 = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        cam_x2 = {"Baseline": 0, "Sensitivity": 2, "Gain": 1}
        spots_x1 = localize.get_spots(movie, real_identifications, BOX, cam_x1)
        spots_x2 = localize.get_spots(movie, real_identifications, BOX, cam_x2)
        np.testing.assert_allclose(spots_x2, spots_x1 * 2)

    def test_gain_divides_signal(self, movie, real_identifications):
        cam_x1 = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        cam_x2 = {"Baseline": 0, "Sensitivity": 1, "Gain": 2}
        spots_x1 = localize.get_spots(movie, real_identifications, BOX, cam_x1)
        spots_x2 = localize.get_spots(movie, real_identifications, BOX, cam_x2)
        np.testing.assert_allclose(spots_x2, spots_x1 / 2, rtol=1e-5)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """``check_nena``, ``check_kinetics``, ``check_drift``."""

    def test_check_nena_returns_float(self, locs, info):
        # ``check_nena`` currently hard-codes ``info=None`` when calling
        # ``postprocess.nena`` and catches the resulting failure, returning
        # NaN. The contract observable from the outside is "return a float".
        nena = localize.check_nena(locs, info)
        assert isinstance(nena, float)
        assert nena > 0

    def test_check_kinetics_returns_positive_scalar(self, locs, info):
        len_mean = localize.check_kinetics(locs, info)
        assert np.isfinite(len_mean)
        assert len_mean > 0

    def test_check_drift_returns_two_floats(self, locs, info):
        result = localize.check_drift(locs, info)
        assert isinstance(result, tuple)
        assert len(result) == 2
        drift_x, drift_y = result
        assert np.isfinite(drift_x)
        assert np.isfinite(drift_y)


# ---------------------------------------------------------------------------
# identifications_from_futures (low-level helper used by the GUI)
# ---------------------------------------------------------------------------


class TestIdentificationsFromFutures:
    """Stitch the per-thread results back into a sorted DataFrame."""

    def test_concatenates_and_sorts_by_frame(self):
        # Build two fake futures whose .result() returns lists of DFs.
        df_a = pd.DataFrame(
            {
                "frame": [3, 1],
                "x": [10, 20],
                "y": [11, 21],
                "net_gradient": [5000.0, 6000.0],
            }
        )
        df_b = pd.DataFrame(
            {
                "frame": [2, 0],
                "x": [30, 40],
                "y": [31, 41],
                "net_gradient": [7000.0, 8000.0],
            }
        )

        class _DummyFuture:
            def __init__(self, lst):
                self._lst = lst

            def result(self):
                return self._lst

        out = localize.identifications_from_futures(
            [_DummyFuture([df_a]), _DummyFuture([df_b])]
        )
        # All four rows preserved
        assert len(out) == 4
        # Sorted ascending by frame
        assert list(out["frame"]) == sorted(out["frame"])


# ---------------------------------------------------------------------------
# fit2D — high-level wrapper that supports gausslq / gaussmle / avg
# ---------------------------------------------------------------------------
#
# ``fit2D`` and ``localize`` both assert ``isinstance(movie,
# AbstractPicassoMovie)``. The bundled .raw movie loads as a plain
# ``np.memmap`` so we feed in the ``picasso_movie`` fixture from conftest
# (a thin AbstractPicassoMovie wrapper around the same memmap).


class TestFit2D:
    """The 2D fitting dispatcher used by Picasso: Localize."""

    def test_gausslq_returns_locs_and_metadata(
        self, picasso_movie, real_identifications, movie_info
    ):
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq",
            multiprocess=False,
        )
        assert len(locs) == len(real_identifications)
        for col in ["x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy"]:
            assert col in locs.columns
        # metadata reflects the chosen fitting method
        assert new_info["Fit method"] == "gausslq"
        # camera_info keys merged into new_info
        assert new_info["Pixelsize"] == 130

    @pytest.mark.parametrize("multiprocess", [False, True])
    def test_gausslq_saves_chi_square_not_likelihood(
        self, picasso_movie, real_identifications, movie_info, multiprocess
    ):
        """The CPU least-squares methods report their goodness of fit as
        ``chi_square`` (the residual sum of squares at the optimum), the
        least-squares counterpart of the MLE fits' ``log_likelihood``. Both
        the serial and the multiprocessing path must carry it, since the
        multiprocessing path ferries it as an extra ``theta`` column."""
        locs, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq",
            multiprocess=multiprocess,
        )
        assert "chi_square" in locs.columns
        assert "log_likelihood" not in locs.columns
        chi = locs["chi_square"].to_numpy()
        assert np.all(chi >= 0) and np.all(np.isfinite(chi))
        # the chi-square column must not have displaced a parameter column
        for col in ["x", "y", "photons", "sx", "sy", "bg", "ellipticity"]:
            assert col in locs.columns

    @pytest.mark.parametrize(
        "method", ["gausslq", "gausslq-spherical", "gausslq-rotated"]
    )
    def test_all_cpu_lq_variants_save_chi_square(
        self, picasso_movie, real_identifications, movie_info, method
    ):
        """Every CPU least-squares model variant carries the column, and the
        rotated one still recovers its ``angle`` (whose detection keys off the
        parameter count, so the extra column must be split off first)."""
        locs, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method=method,
            multiprocess=False,
        )
        assert "chi_square" in locs.columns
        assert np.all(locs["chi_square"].to_numpy() >= 0)
        assert ("angle" in locs.columns) == (method == "gausslq-rotated")

    def test_gaussmle_returns_locs(
        self, picasso_movie, real_identifications, movie_info
    ):
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gaussmle",
            multiprocess=False,
        )
        assert len(locs) == len(real_identifications)
        # MLE-specific metadata
        assert new_info["Fit method"] == "gaussmle"
        # 1e-5, not 0.001: the CPU MLE now runs on Levenberg-Marquardt,
        # where the criterion is relative in the chi-square rather than a
        # position shift in pixels. See ``localize._GAUSS_SCHEDULES``.
        assert new_info["Convergence criterion"] == 1e-5
        assert new_info["Max iterations"] == 100

    @pytest.mark.parametrize(
        "method", ["gausslq-spherical", "gaussmle-spherical"]
    )
    def test_spherical_methods_drop_ellipticity(
        self, picasso_movie, real_identifications, movie_info, method
    ):
        """The spherical CPU methods (LQ and MLE) fit sx == sy and omit the
        always-zero ellipticity column, while keeping every other column."""
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method=method,
            multiprocess=False,
        )
        assert new_info["Fit method"] == method
        assert len(locs) == len(real_identifications)
        assert "ellipticity" not in locs.columns
        assert (locs["sx"].to_numpy() == locs["sy"].to_numpy()).all()
        for col in ["x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy"]:
            assert col in locs.columns

    def test_rotated_lq_keeps_ellipticity_and_adds_angle(
        self, picasso_movie, real_identifications, movie_info
    ):
        """The rotated CPU LQ method keeps ellipticity (widths differ) and
        adds an ``angle`` column wrapped to [-90, 90)."""
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq-rotated",
            multiprocess=False,
        )
        assert new_info["Fit method"] == "gausslq-rotated"
        assert "ellipticity" in locs.columns
        assert "angle" in locs.columns
        assert ((locs["angle"] >= -90.0) & (locs["angle"] < 90.0)).all()

    def test_avg_returns_locs(
        self, picasso_movie, real_identifications, movie_info
    ):
        """The ``avg`` method takes per-pixel averages — produces a locs
        DataFrame even though it doesn't fit a Gaussian."""
        locs, new_info = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="avg",
            multiprocess=False,
        )
        assert len(locs) == len(real_identifications)
        assert new_info["Fit method"] == "avg"

    def test_invalid_fitting_method_raises(
        self, picasso_movie, real_identifications, movie_info
    ):
        with pytest.raises(AssertionError):
            localize.fit2D(
                picasso_movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="bogus",
                multiprocess=False,
            )

    def test_negative_eps_rejected(
        self, picasso_movie, real_identifications, movie_info
    ):
        with pytest.raises(AssertionError):
            localize.fit2D(
                picasso_movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="gaussmle",
                eps=-1.0,
                multiprocess=False,
            )

    def test_missing_pixelsize_warns_and_defaults(
        self, picasso_movie, real_identifications, movie_info
    ):
        """If ``Pixelsize`` is absent from camera_info, fit2D emits a
        warning and defaults to 130 nm."""
        cam = {"Baseline": 0, "Sensitivity": 1, "Gain": 1}
        with pytest.warns(UserWarning, match="Pixelsize"):
            _, new_info = localize.fit2D(
                picasso_movie,
                movie_info,
                cam,
                real_identifications,
                BOX,
                fitting_method="gausslq",
                multiprocess=False,
            )
        assert new_info["Pixelsize"] == 130


# ---------------------------------------------------------------------------
# localize — monolithic identify + fit2D entry point
# ---------------------------------------------------------------------------


class TestLocalize:
    """The top-level ``localize`` pipeline (identify -> get_spots -> fit)."""

    def test_basic_pipeline_returns_locs(self, picasso_movie, movie_info):
        locs = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=False,
        )
        assert isinstance(locs, pd.DataFrame)
        assert len(locs) > 0
        for col in ["frame", "x", "y", "photons", "sx", "sy", "bg"]:
            assert col in locs.columns

    def test_return_info_returns_full_info_chain(
        self, picasso_movie, movie_info
    ):
        """With ``return_info=True``, returns ``(locs, info)`` where info
        contains the original movie info, the identify metadata, and the
        fit metadata."""
        locs, info = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=True,
        )
        assert isinstance(locs, pd.DataFrame)
        assert isinstance(info, list)
        assert len(info) == len(movie_info) + 2
        # Identify info appears second-to-last; fit info last.
        assert "Min. Net Gradient" in info[-2]
        assert "Fit method" in info[-1]

    def test_mm_metadata_carried_over_unless_disabled(
        self, picasso_movie, movie_info, monkeypatch
    ):
        """The MicroManager block of the movie ends up in the
        localizations' metadata, unless the user switched it off."""
        mm_info = [
            dict(movie_info[0], **{"Micro-Manager Metadata": {"Cam": "Zyla"}})
        ] + list(movie_info[1:])
        kwargs = dict(
            camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            identification_parameters={
                "Min. Net Gradient": MIN_NG,
                "Box Size": BOX,
            },
            movie_info=mm_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=True,
        )
        _, info = localize.localize(picasso_movie, **kwargs)
        assert info[0]["Micro-Manager Metadata"] == {"Cam": "Zyla"}

        monkeypatch.setattr(io, "_save_mm_metadata", lambda: False)
        _, info = localize.localize(picasso_movie, **kwargs)
        assert "Micro-Manager Metadata" not in info[0]
        # the rest of the movie metadata is untouched, and so is the
        # caller's copy
        assert info[0]["Frames"] == mm_info[0]["Frames"]
        assert "Micro-Manager Metadata" in mm_info[0]

    def test_localize_matches_identify_plus_fit2d(
        self, picasso_movie, real_identifications, movie_info
    ):
        """Calling ``localize`` should produce the same result (up to
        ordering) as calling ``identify`` + ``fit2D`` separately, since
        ``localize`` is just glue."""
        # Direct path
        locs_direct, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            CAMERA_INFO_WITH_PIXELSIZE,
            real_identifications,
            BOX,
            fitting_method="gausslq",
            multiprocess=False,
        )
        # Through the high-level entry point
        locs_high = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            return_info=False,
        )
        assert len(locs_direct) == len(locs_high)
        # photons sums match
        np.testing.assert_allclose(
            np.sort(locs_direct["photons"].to_numpy()),
            np.sort(locs_high["photons"].to_numpy()),
            rtol=1e-3,
        )

    def test_roi_is_applied_at_identification(self, picasso_movie, movie_info):
        """Passing an ROI confines the localizations to that pixel
        window."""
        roi = ((0, 0), (16, 16))
        locs = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            roi=roi,
            fitting_method="gausslq",
            threaded=False,
            return_info=False,
        )
        # No localization outside the ROI window
        if len(locs) > 0:
            assert (locs["x"] < 16).all()
            assert (locs["y"] < 16).all()


# ---------------------------------------------------------------------------
# localize_3D — identify + 2D fit + z fitting
# ---------------------------------------------------------------------------


class TestLocalize3D:
    """End-to-end 3D localization pipeline.

    Note: the public ``localize_3D`` validates its movie argument with
    ``isinstance(movie, (np.ndarray, ND2Movie))`` — but the inner
    ``fit2D`` then asserts ``isinstance(movie, AbstractPicassoMovie)``,
    which conflicts. So the public ``localize_3D`` is unusable for
    AbstractPicassoMovie inputs; we exercise the internal
    ``_localize_3D`` (which has no such guard) to verify that the actual
    pipeline produces sensible 3D locs.
    """

    def test_public_localize_3d_rejects_wrapper(
        self, picasso_movie, movie_info
    ):
        """The public function's input check excludes AbstractPicassoMovie."""
        with pytest.raises(AssertionError, match="numpy array or ND2Movie"):
            localize.localize_3D(
                picasso_movie,
                movie_info=movie_info,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                box=BOX,
                minimum_ng=MIN_NG,
                calibration_3d=dict(CALIB_3D),
                fitting_method="gausslq",
                multiprocess=False,
            )

    def test_public_localize_3d_invalid_calibration_type(
        self, movie, movie_info
    ):
        with pytest.raises(AssertionError, match="calibration_3d"):
            localize.localize_3D(
                movie,
                movie_info=movie_info,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                box=BOX,
                minimum_ng=MIN_NG,
                calibration_3d=12345,  # neither dict nor str
                fitting_method="gausslq",
                multiprocess=False,
            )

    def test_underlying_pipeline_produces_z_locs(
        self, picasso_movie, movie_info
    ):
        """Drive the full identify->fit->zfit pipeline through
        ``_localize_3D`` and verify the output has the expected 3D
        columns and finite z values."""
        locs, _ = localize._localize_3D(
            picasso_movie,
            movie_info=movie_info,
            camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            box=BOX,
            minimum_ng=MIN_NG,
            calibration_3d=dict(CALIB_3D),
            fitting_method="gausslq",
            multiprocess=False,
        )
        assert isinstance(locs, pd.DataFrame)
        assert len(locs) > 0
        for col in ["x", "y", "z", "d_zcalib", "lpz", "sx", "sy"]:
            assert col in locs.columns
        assert np.all(np.isfinite(locs["z"].to_numpy()))
        assert (locs["lpz"] > 0).all()


# ---------------------------------------------------------------------------
# The v0.12.0 API changes: ``fit2D`` -> ``fit``, ``localize_3D`` folded into
# ``localize(calibration_3d=...)``, ``parameters`` ->
# ``identification_parameters``, keyword-only arguments, ``mle_method`` gone.
# Every old call must keep working (with a DeprecationWarning) until then.
# ---------------------------------------------------------------------------


class TestDeprecatedLocalizeAPI:
    """The deprecated spellings still run and warn."""

    def test_fit2d_warns_and_matches_fit(
        self, picasso_movie, real_identifications, movie_info
    ):
        with pytest.warns(DeprecationWarning, match="fit2D"):
            old, old_info = localize.fit2D(
                picasso_movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="gausslq",
                mle_method="sigmaxy",
                multiprocess=False,
            )
        new, new_info = localize.fit(
            picasso_movie,
            camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            identifications=real_identifications,
            box=BOX,
            fitting_method="gausslq",
            multiprocess=False,
        )
        pd.testing.assert_frame_equal(old, new)
        assert old_info == new_info

    def test_fit2d_mle_method_warns_separately(
        self, picasso_movie, real_identifications, movie_info
    ):
        with pytest.warns(DeprecationWarning, match="mle_method"):
            localize.fit2D(
                picasso_movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="avg",
                mle_method="sigma",
                multiprocess=False,
            )

    def test_positional_arguments_warn(self, picasso_movie, movie_info):
        with pytest.warns(DeprecationWarning, match="positional"):
            locs, _ = localize.localize(
                picasso_movie,
                CAMERA_INFO_WITH_PIXELSIZE,
                {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
                movie_info=movie_info,
                fitting_method="gausslq",
                threaded=False,
            )
        assert len(locs) > 0

    def test_parameters_keyword_warns_and_still_works(
        self, picasso_movie, movie_info
    ):
        with pytest.warns(DeprecationWarning, match="identification_para"):
            locs, _ = localize.localize(
                picasso_movie,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                parameters={"Min. Net Gradient": MIN_NG, "Box Size": BOX},
                movie_info=movie_info,
                fitting_method="gausslq",
                threaded=False,
            )
        assert len(locs) > 0

    def test_mle_method_warns(self, picasso_movie, movie_info):
        with pytest.warns(DeprecationWarning, match="mle_method"):
            localize.localize(
                picasso_movie,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                identification_parameters={
                    "Min. Net Gradient": MIN_NG,
                    "Box Size": BOX,
                },
                movie_info=movie_info,
                fitting_method="gausslq",
                mle_method="sigmaxy",
                threaded=False,
            )

    def test_duplicate_camera_info_raises(self, picasso_movie):
        with pytest.raises(TypeError, match="camera_info"):
            localize.localize(
                picasso_movie,
                CAMERA_INFO_WITH_PIXELSIZE,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            )

    def test_localize_3d_warns(self, picasso_movie, movie_info):
        with pytest.warns(DeprecationWarning, match="localize_3D"):
            with pytest.raises(AssertionError):
                # the movie type guard fires after the warning
                localize.localize_3D(
                    picasso_movie,
                    movie_info=movie_info,
                    camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                    box=BOX,
                    minimum_ng=MIN_NG,
                    calibration_3d=dict(CALIB_3D),
                )


class TestLocalizeAstigmatism3D:
    """``localize(calibration_3d=...)``, the replacement for
    ``localize_3D``."""

    def _localize(self, movie, movie_info, **kwargs):
        return localize.localize(
            movie,
            camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            identification_parameters={
                "Min. Net Gradient": MIN_NG,
                "Box Size": BOX,
            },
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            **kwargs,
        )

    def test_matches_localize_3d(self, picasso_movie, movie_info):
        """The astigmatic path reproduces the old ``_localize_3D``."""
        old, old_info = localize._localize_3D(
            picasso_movie,
            movie_info=movie_info,
            camera_info=CAMERA_INFO_WITH_PIXELSIZE,
            box=BOX,
            minimum_ng=MIN_NG,
            calibration_3d=dict(CALIB_3D),
            fitting_method="gausslq",
            multiprocess=False,
        )
        new, new_info = self._localize(
            picasso_movie, movie_info, calibration_3d=dict(CALIB_3D)
        )
        pd.testing.assert_frame_equal(old, new)
        assert len(new_info) == len(old_info)
        for col in ["z", "d_zcalib", "lpz"]:
            assert col in new.columns
        assert np.all(np.isfinite(new["z"].to_numpy()))

    def test_no_calibration_stays_2d(self, picasso_movie, movie_info):
        locs, _ = self._localize(picasso_movie, movie_info)
        assert "z" not in locs.columns

    def test_invalid_calibration_type(self, picasso_movie, movie_info):
        with pytest.raises(AssertionError, match="calibration_3d"):
            self._localize(picasso_movie, movie_info, calibration_3d=12345)

    @pytest.mark.parametrize("fitting_method", ["avg", "gausslq-spherical"])
    def test_rejects_methods_without_astigmatism(
        self, picasso_movie, movie_info, fitting_method
    ):
        with pytest.raises(AssertionError, match="calibration_3d"):
            localize.localize(
                picasso_movie,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                identification_parameters={
                    "Min. Net Gradient": MIN_NG,
                    "Box Size": BOX,
                },
                movie_info=movie_info,
                fitting_method=fitting_method,
                calibration_3d=dict(CALIB_3D),
                threaded=False,
            )


# ---------------------------------------------------------------------------
# MovieLoadWorker — background loader for Picasso: Localize
#
# The GUI opens movies on a background QThread so the window stays
# responsive (see picasso.gui.localize.MovieLoadWorker). These tests drive
# the worker's run() directly with monkeypatched io loaders — no QThread is
# started, so signals delivered to same-thread receivers fire synchronously
# (Qt DirectConnection), which is exactly what we rely on to observe them.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _keep_user_settings(monkeypatch):
    """Isolate the tests from the developer's real ``~/.picasso`` settings,
    in both directions.

    Closing a Localize ``Window`` persists the dialog state (last folder, box
    size, min. net gradient, fit model, temporal median, Gaussian filter),
    which a test must not do to the machine it runs on. Building one *restores*
    that same state, so a stored value (a non-zero Gaussian sigma, say) would
    otherwise decide what these tests see - passing on a fresh install and
    failing on the machine of whoever last used the GUI."""
    monkeypatch.setattr(io, "save_user_settings", lambda settings: None)
    monkeypatch.setattr(io, "load_user_settings", lambda: lib.AutoDict())


@pytest.fixture(scope="session", autouse=True)
def _qt_app(qapp):
    """A QApplication must exist before any QObject (the worker) is built.

    Autouse because widgets are built throughout this module; the application
    itself is the shared one from conftest."""
    return qapp


class _Collector:
    """Records the worker's terminal signals for assertions."""

    def __init__(self, worker):
        self.finished = None
        self.failed = None
        self.progress = []
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.progress.connect(
            lambda i, name: self.progress.append((i, name))
        )

    def _on_finished(self, movies, infos, paths):
        self.finished = (movies, infos, paths)

    def _on_failed(self, message):
        self.failed = message


def _info(name="Channel 0"):
    return [{"Frames": 1, "Height": 4, "Width": 4, "Channel": name}]


@pytest.mark.gui
class TestMovieLoadWorker:
    def test_load_movie_per_file(self, monkeypatch):
        """load_all=False loads one channel per path via io.load_movie."""
        loaded = {}

        def fake_load_movie(path, prompt_info=None, progress=None):
            loaded[path] = prompt_info
            return f"movie::{path}", _info(path)

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = localize_gui.MovieLoadWorker(
            ["a.tif", "b.tif"], prompt_for_path=lambda p: None
        )
        out = _Collector(worker)
        worker.run()

        movies, infos, paths = out.finished
        assert movies == ["movie::a.tif", "movie::b.tif"]
        assert paths == ["a.tif", "b.tif"]
        assert infos[1][0]["Channel"] == "b.tif"
        assert out.failed is None
        # One progress tick per file, in order.
        assert [i for i, _ in out.progress] == [0, 1]

    def test_load_all_expands_channels(self, monkeypatch):
        """load_all=True reads every channel of each file via
        io.load_movie_all; each returned movie maps back to its path."""

        def fake_load_movie_all(path, prompt_info=None, progress=None):
            return ["m0", "m1", "m2"], [_info("c0"), _info("c1"), _info("c2")]

        monkeypatch.setattr(io, "load_movie_all", fake_load_movie_all)
        worker = localize_gui.MovieLoadWorker(
            ["multi.lif"], prompt_for_path=lambda p: None, load_all=True
        )
        out = _Collector(worker)
        worker.run()

        movies, infos, paths = out.finished
        assert movies == ["m0", "m1", "m2"]
        # Every channel is attributed to the one source file.
        assert paths == ["multi.lif", "multi.lif", "multi.lif"]
        assert len(infos) == 3

    def test_none_result_is_skipped(self, monkeypatch):
        """A loader returning None (e.g. a canceled prompt) drops that
        file without aborting the rest of the batch."""

        def fake_load_movie(path, prompt_info=None, progress=None):
            if path == "skip.tif":
                return None
            return f"movie::{path}", _info(path)

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = localize_gui.MovieLoadWorker(
            ["skip.tif", "keep.tif"], prompt_for_path=lambda p: None
        )
        out = _Collector(worker)
        worker.run()

        movies, _, paths = out.finished
        assert movies == ["movie::keep.tif"]
        assert paths == ["keep.tif"]

    def test_prompt_is_proxied_and_result_returned(self, monkeypatch):
        """A loader that calls prompt_info gets the value the GUI handler
        supplies: the worker emits prompt_requested, the (same-thread)
        handler fills the holder and releases the worker's event."""

        def fake_load_movie(path, prompt_info=None, progress=None):
            chosen = prompt_info(["DAPI", "GFP"])
            return f"movie::{chosen}", _info(chosen)

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        worker = localize_gui.MovieLoadWorker(
            ["m.czi"], prompt_for_path=lambda p: (lambda chans: "GFP")
        )

        seen = {}

        def handle_prompt(callback, args_kwargs, holder):
            args, kwargs = args_kwargs
            seen["channels"] = args[0]
            holder["result"] = callback(*args, **kwargs)
            worker._prompt_event.set()

        worker.prompt_requested.connect(handle_prompt)
        out = _Collector(worker)
        worker.run()

        assert seen["channels"] == ["DAPI", "GFP"]
        assert out.finished[0] == ["movie::GFP"]

    def test_exception_emits_failed(self, monkeypatch):
        """A loader error is reported through the failed signal rather than
        propagating out of run() (which runs on a worker thread)."""

        def boom(path, prompt_info=None, progress=None):
            raise RuntimeError("bad file")

        monkeypatch.setattr(io, "load_movie", boom)
        worker = localize_gui.MovieLoadWorker(
            ["x.tif"], prompt_for_path=lambda p: None
        )
        out = _Collector(worker)
        worker.run()

        assert out.finished is None
        assert "bad file" in out.failed

    def test_cancel_stops_before_next_file(self, monkeypatch):
        """cancel() set during a load stops the loop before the next
        file, and the whole batch is discarded: the file that was being
        read cannot be interrupted mid-way, so it is dropped rather than
        delivered as a half-loaded channel."""
        seen = []

        def fake_load_movie(path, prompt_info=None, progress=None):
            seen.append(path)
            worker.cancel()  # cancel while the first file is "loading"
            return f"movie::{path}", _info(path)

        worker = localize_gui.MovieLoadWorker(
            ["first.tif", "second.tif"], prompt_for_path=lambda p: None
        )
        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        out = _Collector(worker)
        worker.run()

        # Only the first file was loaded; the second was never attempted.
        assert seen == ["first.tif"]
        # finished still fires (so the GUI can tear the dialog down), but
        # with an empty batch.
        assert out.finished == ([], [], [])


# ---------------------------------------------------------------------------
# Optional GPU backend (Gpufit) — skipped when no CUDA GPU is available
# ---------------------------------------------------------------------------


# The GPU fit reports a per-spot termination code; 0 is a converged fit. The
# MLE (Poisson) estimator additionally emits code 3 (NEG_CURVATURE_MLE) when the
# likelihood Hessian loses positive-definiteness - the returned parameters are
# then the last (unconverged, unreliable) iterate, so tests that assert
# numerical recovery must restrict to converged spots.
_GPUFIT_CONVERGED = 0


def _gpufit_gauss_with_states(spots, rotated=False, mle=False):
    """Run the low-level GPU Gaussian fit while keeping the per-spot fit
    states that :func:`localize.fit_spots_gauss_gpu` drops. Mirrors that function
    exactly (same initial parameters, model, estimator, tolerance and iteration
    cap) and applies the same ``photons = amplitude * 2*pi*sx*sy`` conversion,
    returning ``(theta, states, n_iterations)``."""
    data = np.maximum(spots, 0) if mle else spots
    size = data.shape[1]
    init = seeds.initial_parameters_gauss(data, size, rotated=rotated)
    model = gaussfit_cuda.ROTATED if rotated else gaussfit_cuda.ELLIPTIC
    params, _chi_squares, states, n_iter = gaussfit_cuda.fit_spots(
        model, data, init.astype(np.float64), mle=mle
    )
    params = params.astype(np.float32)
    params[:, 0] *= 2.0 * np.pi * params[:, 3] * params[:, 4]
    return params, states, n_iter


def _make_rotated_spot(box, x0, y0, sx, sy, photons, bg, angle):
    """Point-sampled rotated elliptical Gaussian, matching the model Gpufit's
    GAUSS_2D_ROTATED optimizes (and the reference ``_gauss_model`` used by the
    CRLB tests): ``mu = photons/(2 pi sx sy) * exp(...) + bg``. ``x0``/``y0``
    are offsets from the box center."""
    half = box // 2
    g = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(g, g)  # X varies along columns (x), Y along rows (y)
    dx, dy = X - x0, Y - y0
    ct, st = np.cos(angle), np.sin(angle)
    u = dx * ct - dy * st
    w = dx * st + dy * ct
    e = np.exp(-0.5 * (u**2 / sx**2 + w**2 / sy**2))
    return (photons / (2 * np.pi * sx * sy) * e + bg).astype(np.float32)


@pytest.mark.skipif(not localize.CUDA_AVAILABLE, reason="no CUDA device")
class TestGpufit:
    """Thorough tests for the Gpufit Gaussian codepath (``fit_spots_gauss_gpu``).
    Requires a CUDA-capable GPU, so skipped in the typical test environment.

    Parameter order returned by Gpufit is ``[photons, x, y, sx, sy, bg]`` and,
    for the rotated model, ``[photons, x, y, sx, sy, bg, angle]``. ``x``/``y``
    are box-pixel coordinates, so the ground-truth center offset is
    ``x - box // 2``."""

    # -- least squares: the deterministic, model-exact path ----------------

    def test_lse_recovers_all_parameters_noiseless(self, synthetic_spots):
        """On noiseless spots the model matches the data exactly, so LSE must
        recover every parameter - not just photons - to tight tolerance."""
        spots, gt = synthetic_spots
        theta = localize.fit_spots_gauss_gpu(spots, mle=False)
        assert theta.shape == (len(spots), 6)
        half = spots.shape[1] // 2
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=1e-3)
        np.testing.assert_allclose(theta[:, 1] - half, gt.x.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 2] - half, gt.y.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 3], gt.sx.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 4], gt.sy.values, atol=2e-3)
        np.testing.assert_allclose(theta[:, 5], gt.bg.values, atol=1e-2)

    def test_lse_recovers_parameters_noisy(self, synthetic_spots_noisy):
        """With Poisson noise LSE still recovers ground truth, at looser
        (noise-limited) tolerance."""
        spots, gt = synthetic_spots_noisy
        theta = localize.fit_spots_gauss_gpu(spots, mle=False)
        half = spots.shape[1] // 2
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=0.05)
        np.testing.assert_allclose(theta[:, 1] - half, gt.x.values, atol=0.1)
        np.testing.assert_allclose(theta[:, 2] - half, gt.y.values, atol=0.1)

    def test_photons_are_amplitude_times_2pi_sxsy(self, synthetic_spots):
        """The reported photon count is the raw Gaussian amplitude scaled by
        its integral ``2*pi*sx*sy`` - the conversion fit_spots_gauss_gpu
        applies to the model's peak-height parameter."""
        spots, _ = synthetic_spots
        size = spots.shape[1]
        init = seeds.initial_parameters_gauss(spots, size)
        raw, _, _, _ = gaussfit_cuda.fit_spots(
            gaussfit_cuda.ELLIPTIC,
            spots,
            init.astype(np.float64),
            mle=False,
        )
        theta = localize.fit_spots_gauss_gpu(spots, mle=False)
        expected = raw[:, 0] * 2.0 * np.pi * raw[:, 3] * raw[:, 4]
        np.testing.assert_allclose(theta[:, 0], expected, rtol=1e-5)

    # -- rotated elliptical Gaussian --------------------------------------

    def test_rotated_recovers_angle(self):
        """The rotated model recovers the ground-truth rotation angle (radians)
        for a range of angles."""
        box = 9
        angles = np.array([0.2, 0.6, -0.5, 1.0])
        spots = np.stack(
            [
                _make_rotated_spot(box, 0.1, -0.15, 1.6, 0.9, 5000.0, 10.0, a)
                for a in angles
            ]
        )
        theta = localize.fit_spots_gauss_gpu(spots, rotated=True, mle=False)
        assert theta.shape == (len(angles), 7)
        np.testing.assert_allclose(theta[:, 6], angles, atol=1e-3)
        # widths recovered along the rotated axes
        np.testing.assert_allclose(theta[:, 3], 1.6, atol=5e-3)
        np.testing.assert_allclose(theta[:, 4], 0.9, atol=5e-3)

    def test_rotated_and_spherical_mutually_exclusive(self, synthetic_spots):
        spots, _ = synthetic_spots
        with pytest.raises(ValueError):
            localize.fit_spots_gauss_gpu(spots, rotated=True, spherical=True)

    # -- spherical (isotropic, single-width GAUSS_2D model) ---------------

    def test_spherical_lse_recovers_isotropic(self, synthetic_spots_isotropic):
        """The GAUSS_2D model recovers the shared width and returns the
        expanded elliptical layout with sx == sy."""
        spots, gt = synthetic_spots_isotropic
        theta = localize.fit_spots_gauss_gpu(spots, spherical=True, mle=False)
        assert theta.shape == (len(spots), 6)
        np.testing.assert_array_equal(theta[:, 3], theta[:, 4])
        half = spots.shape[1] // 2
        np.testing.assert_allclose(theta[:, 0], gt.photons.values, rtol=2e-3)
        np.testing.assert_allclose(theta[:, 1] - half, gt.x.values, atol=3e-3)
        np.testing.assert_allclose(theta[:, 2] - half, gt.y.values, atol=3e-3)
        np.testing.assert_allclose(theta[:, 3], gt.sx.values, atol=3e-3)

    def test_spherical_mle_converged_recover(self, synthetic_spots_isotropic):
        spots, gt = synthetic_spots_isotropic
        theta, ll, n_iter, chi2 = localize.fit_spots_gauss_gpu(
            spots, spherical=True, mle=True, return_stats=True
        )
        assert theta.shape == (len(spots), 6)
        np.testing.assert_array_equal(theta[:, 3], theta[:, 4])
        assert ll is not None and ll.shape == (len(spots),)
        assert chi2 is None

    def test_spherical_end_to_end_omits_ellipticity(
        self, synthetic_spots_isotropic
    ):
        """The full GPU spherical path (via ``_fit2d_gauss``) drops the
        ellipticity column."""
        spots, _ = synthetic_spots_isotropic
        n = len(spots)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 50, dtype=np.int64),
                "y": np.full(n, 70, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        locs = localize._fit2d_gauss(
            spots, ids, spots.shape[1], em=False, spherical=True, use_gpu=True
        )
        assert "ellipticity" not in locs.columns
        assert (locs["sx"].to_numpy() == locs["sy"].to_numpy()).all()

    # -- maximum likelihood (Poisson) -------------------------------------

    def test_mle_converged_spots_recover(self, synthetic_spots_noisy):
        """Gpufit's MLE terminates a fraction of fits with NEG_CURVATURE_MLE
        (state 3) and returns their last, unreliable iterate. The fits that DO
        converge (state 0) must recover ground truth tightly. This documents
        the real contract of the vendored MLE estimator."""
        spots, gt = synthetic_spots_noisy
        theta, states, _ = _gpufit_gauss_with_states(spots, mle=True)
        converged = states == _GPUFIT_CONVERGED
        assert converged.any(), "expected at least some MLE fits to converge"
        half = spots.shape[1] // 2
        idx = np.where(converged)[0]
        np.testing.assert_allclose(
            theta[idx, 0], gt.photons.values[idx], rtol=0.08
        )
        np.testing.assert_allclose(
            theta[idx, 1] - half, gt.x.values[idx], atol=0.15
        )
        np.testing.assert_allclose(
            theta[idx, 2] - half, gt.y.values[idx], atol=0.15
        )

    def test_mle_matches_lse_on_converged_spots(self, synthetic_spots_noisy):
        """Where the MLE fit converges, its photon estimate agrees with the
        (always-converging) LSE fit on the same data."""
        spots, _ = synthetic_spots_noisy
        lse = localize.fit_spots_gauss_gpu(spots, mle=False)
        mle, states, _ = _gpufit_gauss_with_states(spots, mle=True)
        idx = np.where(states == _GPUFIT_CONVERGED)[0]
        np.testing.assert_allclose(mle[idx, 0], lse[idx, 0], rtol=0.1)

    def test_mle_clamps_negative_pixels(self, synthetic_spots_noisy):
        """The MLE path clamps negative pixel values (Poisson counts cannot be
        negative) rather than crashing; the public function must accept spots
        with negatives and return finite parameters for converged fits."""
        spots, _ = synthetic_spots_noisy
        spots = spots.copy()
        spots[:, 0, 0] -= 50.0  # inject a negative pixel per spot
        theta, states, _ = _gpufit_gauss_with_states(spots, mle=True)
        idx = states == _GPUFIT_CONVERGED
        assert np.all(np.isfinite(theta[idx]))

    # -- return_stats semantics -------------------------------------------

    def test_return_stats_mle(self, synthetic_spots_noisy):
        spots, _ = synthetic_spots_noisy
        theta, ll, n_iter, chi2 = localize.fit_spots_gauss_gpu(
            spots, mle=True, return_stats=True
        )
        assert theta.shape == (len(spots), 6)
        # MLE: log-likelihood is -0.5 * chi-square, finite, one per spot
        assert ll is not None and ll.shape == (len(spots),)
        assert np.all(np.isfinite(ll))
        assert n_iter.shape == (len(spots),)
        # the chi-square IS the likelihood here, so it is not reported twice
        assert chi2 is None

    def test_return_stats_lse_has_chi_square_not_likelihood(
        self, synthetic_spots
    ):
        spots, _ = synthetic_spots
        theta, ll, n_iter, chi2 = localize.fit_spots_gauss_gpu(
            spots, mle=False, return_stats=True
        )
        # LSE assumes no noise model, so there is no likelihood; what it does
        # report is the residual sum of squares at the optimum.
        assert ll is None
        assert n_iter.shape == (len(spots),)
        assert chi2 is not None and chi2.shape == (len(spots),)
        assert np.all(chi2 >= 0)
        assert np.all(np.isfinite(chi2))

    def test_lse_end_to_end_saves_chi_square(self, synthetic_spots):
        """The full GPU least-squares path emits a chi_square column and no
        log_likelihood."""
        spots, _ = synthetic_spots
        n = len(spots)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 50, dtype=np.int64),
                "y": np.full(n, 70, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        locs = localize._fit2d_gauss(
            spots, ids, spots.shape[1], em=False, mle=False, use_gpu=True
        )
        assert "chi_square" in locs.columns
        assert "log_likelihood" not in locs.columns
        assert np.all(locs["chi_square"].to_numpy() >= 0)

    # -- end-to-end: fit -> localizations ---------------------------------

    def test_end_to_end_locs_absolute_position(self, synthetic_spots):
        """fit_spots_gauss_gpu + locs_from_fits_gauss place each spot at its true
        absolute position (identification pixel + sub-pixel fit offset)."""
        spots, gt = synthetic_spots
        n = len(spots)
        box = spots.shape[1]
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 50, dtype=np.int64),
                "y": np.full(n, 70, dtype=np.int64),
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )
        theta = localize.fit_spots_gauss_gpu(spots, mle=False)
        locs = localize.locs_from_fits_gauss(ids, theta, box, em=False)
        box_offset = int(box / 2)
        # x_abs = x_id + (x_fit - box_offset); x_fit - box//2 == gt.x
        np.testing.assert_allclose(locs["x"], 50 + gt.x.values, atol=3e-3)
        np.testing.assert_allclose(locs["y"], 70 + gt.y.values, atol=3e-3)
        np.testing.assert_allclose(
            locs["photons"], gt.photons.values, rtol=1e-3
        )
        assert np.all(np.isfinite(locs["lpx"])) and (locs["lpx"] > 0).all()


# ---------------------------------------------------------------------------
# GPU-free gpufit helpers — the initial-parameter seed and the fit ->
# localization converter are pure NumPy/pandas and run without a CUDA GPU.
# ---------------------------------------------------------------------------


class TestInitialParametersGpufit:
    """``seeds.initial_parameters_gauss`` seeds the Levenberg-Marquardt
    fit; it is pure NumPy and needs no GPU."""

    def test_elliptic_layout_and_values(self):
        # Two spots with known per-spot max/min so amplitude (max - min) and
        # background (min) are predictable.
        box = 7
        spots = np.zeros((2, box, box), dtype=np.float32)
        spots[0] = 3.0  # flat -> max == min
        spots[0, 3, 3] = 103.0  # peak
        spots[1] = 7.0
        spots[1, 2, 4] = 57.0
        init = seeds.initial_parameters_gauss(spots, box)

        assert init.shape == (2, 6)
        assert init.dtype == np.float32
        center = box / 2.0 - 0.5  # 3.0
        # amplitude = max - min
        np.testing.assert_allclose(init[:, 0], [100.0, 50.0])
        # x, y seeded at the geometric box center
        np.testing.assert_allclose(init[:, 1], center)
        np.testing.assert_allclose(init[:, 2], center)
        # Spot 0's bright pixel sits on the central row and column, so its
        # second moment about them is zero and the width lands on the
        # numerical floor. Spot 1's peak is at (2, 4), off both center lines,
        # so those profiles are empty (0/0) and it falls back to box / 5.
        np.testing.assert_allclose(init[:, 3], [0.5, 1.4])
        np.testing.assert_allclose(init[:, 4], [0.5, 1.4])
        # background = per-spot minimum
        np.testing.assert_allclose(init[:, 5], [3.0, 7.0])

    def test_width_floor_for_small_box(self):
        # A flat spot carries no signal at all: the moment is 0/0, so the seed
        # falls back to max(box / 5, 1.0) and is then capped by it.
        box = 4
        spots = np.ones((1, box, box), dtype=np.float32)
        init = seeds.initial_parameters_gauss(spots, box)
        np.testing.assert_allclose(init[:, 3], 1.0)
        np.testing.assert_allclose(init[:, 4], 1.0)

    def test_the_width_seed_tracks_the_spot_not_the_box(self):
        """The moment seed is what keeps a wide box from starting the fit at a
        blob several times too fat - see
        ``test_gaussfit.TestWideSigmaSeedDoesNotAbortTheFit``."""
        truth = 1.4
        widths = {}
        for box in (13, 23, 31):
            center = (box - 1) / 2.0
            yy, xx = np.mgrid[0:box, 0:box].astype(np.float64)
            mu = (
                500.0
                * np.exp(
                    -0.5
                    * (((xx - center) ** 2 + (yy - center) ** 2) / truth**2)
                )
                + 10.0
            )
            init = seeds.initial_parameters_gauss(
                mu[None].astype(np.float32), box
            )
            widths[box] = init[0, 3]
            # Never wider than the old fixed seed, and close to the truth.
            assert widths[box] <= max(box / 5.0, 1.0)
            assert abs(widths[box] - truth) < 0.5
        # The old seed tripled from box 13 to 31; this one barely moves.
        assert abs(widths[31] - widths[13]) < 0.3

    def test_rotated_breaks_width_symmetry(self):
        # The rotated model gets a 7th (angle) parameter, and the two widths
        # are deliberately made unequal so the angle derivative is non-zero
        # (an isotropic seed makes the first LM Hessian singular).
        box = 7
        spots = np.zeros((3, box, box), dtype=np.float32)
        spots[:, 3, 3] = 100.0
        init = seeds.initial_parameters_gauss(spots, box, rotated=True)
        assert init.shape == (3, 7)
        # A single bright pixel gives a zero second moment, so the widths sit
        # on the floor before the asymmetry nudge is applied.
        np.testing.assert_allclose(init[:, 3], 0.5 * 1.1)
        np.testing.assert_allclose(init[:, 4], 0.5 * 0.9)
        assert (init[:, 3] != init[:, 4]).all()
        np.testing.assert_allclose(init[:, 6], 0.0)

    def test_spherical_single_width_layout(self):
        # Gpufit's isotropic GAUSS_2D model takes 5 parameters:
        # [amplitude, x, y, s, bg] — a single width, unlike the elliptic
        # 6-parameter layout.
        box = 7
        spots = np.zeros((2, box, box), dtype=np.float32)
        spots[0] = 3.0
        spots[0, 3, 3] = 103.0
        spots[1] = 7.0
        spots[1, 2, 4] = 57.0
        init = seeds.initial_parameters_gauss(spots, box, spherical=True)
        assert init.shape == (2, 5)
        assert init.dtype == np.float32
        center = box / 2.0 - 0.5
        np.testing.assert_allclose(init[:, 0], [100.0, 50.0])  # amplitude
        np.testing.assert_allclose(init[:, 1], center)  # x
        np.testing.assert_allclose(init[:, 2], center)  # y
        # As above: spot 0 on the center lines, spot 1 off them.
        np.testing.assert_allclose(init[:, 3], [0.5, 1.4])  # single width
        np.testing.assert_allclose(init[:, 4], [3.0, 7.0])  # background


class TestLocsFromFitsGpufit:
    """``localize.locs_from_fits_gauss`` maps gpufit theta
    ``[photons, x, y, sx, sy, bg, (angle)]`` to a localizations frame. Pure
    pandas/NumPy - no GPU needed."""

    def _ids(self, n, frames=None):
        return pd.DataFrame(
            {
                "frame": (
                    np.arange(n, dtype=np.uint32)
                    if frames is None
                    else np.asarray(frames, dtype=np.uint32)
                ),
                "x": np.arange(n, dtype=np.int64) + 10,
                "y": np.arange(n, dtype=np.int64) + 20,
                "net_gradient": np.full(n, 5000.0, dtype=np.float32),
            }
        )

    def test_xy_offset_and_passthrough_columns(self):
        # x/y are the sub-pixel fit offset plus the integer identification
        # position minus the box half-offset; photons/sx/sy/bg pass through.
        theta = np.array(
            [
                [500.0, 3.2, 3.7, 1.3, 1.1, 5.0],
                [800.0, 3.4, 3.1, 1.2, 1.4, 4.0],
            ],
            dtype=np.float32,
        )
        ids = self._ids(2)
        locs = localize.locs_from_fits_gauss(ids, theta, BOX, em=False)
        box_offset = int(BOX / 2)
        np.testing.assert_allclose(
            locs["x"], theta[:, 1] + ids["x"].to_numpy() - box_offset
        )
        np.testing.assert_allclose(
            locs["y"], theta[:, 2] + ids["y"].to_numpy() - box_offset
        )
        np.testing.assert_allclose(locs["photons"], theta[:, 0])
        np.testing.assert_allclose(locs["sx"], theta[:, 3])
        np.testing.assert_allclose(locs["sy"], theta[:, 4])
        np.testing.assert_allclose(locs["bg"], theta[:, 5])

    def test_ellipticity_formula(self):
        theta = np.array([[500.0, 3.0, 3.0, 1.4, 1.0, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gauss(
            theta=theta, box=BOX, em=False, identifications=self._ids(1)
        )
        # (max - min) / max = (1.4 - 1.0) / 1.4
        np.testing.assert_allclose(
            locs["ellipticity"], (1.4 - 1.0) / 1.4, rtol=1e-6
        )

    def test_spherical_omits_ellipticity_column(self):
        # A spherical fit has sx == sy, so ellipticity is always 0 and is
        # dropped entirely. The rest of the columns are unaffected.
        theta = np.array([[500.0, 3.0, 3.0, 1.2, 1.2, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, spherical=True
        )
        assert "ellipticity" not in locs.columns
        for col in (
            "frame",
            "x",
            "y",
            "photons",
            "sx",
            "sy",
            "bg",
            "lpx",
            "lpy",
            "net_gradient",
        ):
            assert col in locs.columns

    def test_spherical_flag_only_drops_ellipticity(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.2, 1.2, 5.0]], dtype=np.float32)
        full = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, mle=True, spherical=False
        )
        sph = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, mle=True, spherical=True
        )
        assert set(full.columns) - set(sph.columns) == {"ellipticity"}
        for col in sph.columns:
            np.testing.assert_array_equal(
                sph[col].to_numpy(), full[col].to_numpy()
            )

    def test_lse_precision_is_mortensen_no_unc_columns(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, mle=False
        )
        expected_lpx = gausslq.localization_precision(
            theta[:, 0], theta[:, 3], theta[:, 4], theta[:, 5], em=False
        )
        expected_lpy = gausslq.localization_precision(
            theta[:, 0], theta[:, 4], theta[:, 3], theta[:, 5], em=False
        )
        np.testing.assert_allclose(locs["lpx"], expected_lpx, rtol=1e-6)
        np.testing.assert_allclose(locs["lpy"], expected_lpy, rtol=1e-6)
        # least squares does not emit per-parameter uncertainties
        for col in ("photons_unc", "bg_unc", "sx_unc", "sy_unc"):
            assert col not in locs.columns

    def test_mle_precision_is_crlb_with_unc_columns(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        locs = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, mle=True
        )
        crlb = precision._gauss_crlb(theta, BOX, em=False)
        np.testing.assert_allclose(locs["lpx"], np.sqrt(crlb[:, 1]), rtol=1e-6)
        np.testing.assert_allclose(locs["lpy"], np.sqrt(crlb[:, 2]), rtol=1e-6)
        np.testing.assert_allclose(
            locs["photons_unc"], np.sqrt(crlb[:, 0]), rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["bg_unc"], np.sqrt(crlb[:, 5]), rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["sx_unc"], np.sqrt(crlb[:, 3]), rtol=1e-6
        )
        np.testing.assert_allclose(
            locs["sy_unc"], np.sqrt(crlb[:, 4]), rtol=1e-6
        )

    def test_rotated_angle_column_normalized(self):
        # angle is stored in degrees, sign-flipped from the radians theta and
        # wrapped to [-90, 90) since an ellipse repeats every half turn.
        # 100 deg -> -100 deg after the sign flip -> wraps to +80.
        theta = np.array(
            [[500.0, 3.0, 3.0, 1.4, 1.0, 5.0, np.deg2rad(100.0)]],
            dtype=np.float32,
        )
        locs = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, mle=True
        )
        assert "angle" in locs.columns
        np.testing.assert_allclose(locs["angle"], 80.0, atol=1e-4)
        assert -90.0 <= locs["angle"].iloc[0] < 90.0
        assert "angle_unc" in locs.columns

    def test_sorted_by_frame(self):
        theta = np.tile(
            np.array([500.0, 3.0, 3.0, 1.2, 1.2, 5.0], dtype=np.float32),
            (3, 1),
        )
        ids = self._ids(3, frames=[2, 0, 1])
        locs = localize.locs_from_fits_gauss(ids, theta, BOX, em=False)
        assert list(locs["frame"]) == [0, 1, 2]

    def test_stats_columns_optional(self):
        theta = np.array([[500.0, 3.0, 3.0, 1.2, 1.2, 5.0]], dtype=np.float32)
        # without stats, no log_likelihood / iterations
        locs = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False
        )
        assert "log_likelihood" not in locs.columns
        assert "iterations" not in locs.columns
        # with stats they appear, correctly typed
        locs = localize.locs_from_fits_gauss(
            self._ids(1),
            theta,
            BOX,
            em=False,
            mle=True,
            log_likelihood=np.array([-12.0], dtype=np.float32),
            iterations=np.array([7], dtype=np.int32),
        )
        assert locs["log_likelihood"].dtype == np.float32
        assert locs["iterations"].dtype == np.int32
        assert locs["iterations"].iloc[0] == 7

    def test_em_scales_lse_precision_by_sqrt2(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        no_em = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=False, mle=False
        )
        em = localize.locs_from_fits_gauss(
            self._ids(1), theta, BOX, em=True, mle=False
        )
        np.testing.assert_allclose(
            em["lpx"] / no_em["lpx"], np.sqrt(2.0), rtol=1e-5
        )

    def test_gpu_suffixed_alias_warns_and_forwards(self):
        """The old ``_gpu``-suffixed name was a misnomer (the converter is
        backend-agnostic), so it warns but must still return the same frame."""
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]], dtype=np.float32)
        ids = self._ids(1)
        with pytest.warns(DeprecationWarning, match="locs_from_fits_gauss"):
            old = localize.locs_from_fits_gauss_gpu(ids, theta, BOX, em=False)
        new = localize.locs_from_fits_gauss(ids, theta, BOX, em=False)
        pd.testing.assert_frame_equal(old, new)

    def test_legacy_locs_from_fits_is_deprecated(self):
        """The leftover ``gaussmle``-GPU converter goes in 1.0."""
        theta = np.zeros((1, 6), dtype=np.float32)
        with pytest.warns(DeprecationWarning, match="locs_from_fits_gauss"):
            localize.locs_from_fits(
                self._ids(1),
                theta,
                np.ones((1, 6), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.ones(1, dtype=np.int32),
                BOX,
            )


# ---------------------------------------------------------------------------
# Cubic-spline PSF fitting (Gpufit SPLINE_2D / SPLINE_3D)
# ---------------------------------------------------------------------------


def _fake_spline_calibration(model="spline-3d", box=BOX, n_channels=2):
    """Build a small, structurally valid spline calibration dict. The
    coefficient values are arbitrary - these tests exercise the packing,
    parameter mapping and I/O, not a real fit."""
    if model == "spline-2d":
        n_intervals = [box - 1, box - 1]
        n_data = [box, box]
        n_coef = 16
        coefficients = np.arange(
            n_coef * np.prod(n_intervals), dtype=np.float32
        ).reshape([n_coef] + n_intervals)
    elif model == "spline-3d-multichannel":
        nz = 21
        n_intervals = [box - 1, box - 1, nz - 1]
        n_data = [box, box, nz]
        coefficients = np.arange(
            64 * np.prod(n_intervals) * n_channels, dtype=np.float32
        ).reshape([64] + n_intervals + [n_channels])
    else:
        nz = 21
        n_intervals = [box - 1, box - 1, nz - 1]
        n_data = [box, box, nz]
        n_coef = 64
        coefficients = np.arange(
            n_coef * np.prod(n_intervals), dtype=np.float32
        ).reshape([n_coef] + n_intervals)
    calib = {
        "model": model,
        "coefficients": coefficients,
        "n_data": n_data,
        "n_intervals": n_intervals,
        "oversampling": 1.0,
        "z_center": 10.0,
        "z_step_nm": 20.0,
        "effective_sigma": 1.2,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": PIXELSIZE,
        "Path": "test_calibration",
    }
    if model == "spline-3d-multichannel":
        calib["n_channels"] = n_channels
        calib["channel_transforms"] = [IDENTITY] * n_channels
    return calib


# ---------------------------------------------------------------------------
# Known separable Gaussian spline for CRLB tests. Built with scipy CubicSpline
# so the exact model Phi = gx(x) gy(y) [gz(z)] and its analytic derivatives are
# known, giving a closed-form reference for _spline_crlb
# independently of spline.spline_coefficients. sx != sy makes it astigmatic
# (lpx != lpy); gz encodes axial information (finite lpz). The coefficient table
# is written in the raw flat-buffer layout the evaluator/kernels expect.
# ---------------------------------------------------------------------------
def _gauss_spline_1d(sigma, center, n):
    x = np.arange(n, dtype=np.float64)
    return CubicSpline(x, np.exp(-0.5 * ((x - center) / sigma) ** 2))


def _gauss_spline_calibration(
    model="spline-3d",
    box=BOX,
    nz=21,
    sx=1.0,
    sy=1.4,
    sz=3.0,
    n_channels=2,
    photon_scale=1.0,
):
    """Calibration dict for a separable Gaussian spline, plus the reference 1D
    splines ``(gx, gy, gz)`` (``gz`` None for 2D) for the closed-form CRLB."""
    cxy = (box - 1) / 2.0
    gx = _gauss_spline_1d(sx, cxy, box)
    gy = _gauss_spline_1d(sy, cxy, box)
    nix = niy = box - 1
    # per-interval coeffs, ascending powers: c[i, p] = spline.c[3 - p, i]
    cx = gx.c[::-1].T
    cy = gy.c[::-1].T
    calib = {
        "model": model,
        "oversampling": 1.0,
        "photon_scale": photon_scale,
        "box": box,
        "pixelsize": PIXELSIZE,
    }
    if model == "spline-2d":
        # (niy, nix, yp, xp) -> raw (16, nix, niy)
        c = np.einsum("yY,xX->yxYX", cy, cx).reshape(16, nix, niy)
        calib.update(coefficients=c.astype(np.float32), n_data=[box, box])
        return calib, (gx, gy, None)
    gz = _gauss_spline_1d(sz, (nz - 1) / 2.0, nz)
    niz = nz - 1
    cz = gz.c[::-1].T
    # (niz, niy, nix, zp, yp, xp) -> raw (64, nix, niy, niz)
    c = np.einsum("zZ,yY,xX->zyxZYX", cz, cy, cx).reshape(64, nix, niy, niz)
    if model == "spline-3d-multichannel":
        c = np.repeat(c[..., None], n_channels, axis=-1)
        calib["n_channels"] = n_channels
    calib.update(
        coefficients=c.astype(np.float32),
        n_data=[box, box, nz],
        z_center=(nz - 1) / 2.0,
        z_step_nm=20.0,
        magnification_factor=1.0,
    )
    return calib, (gx, gy, gz)


def _ref_model_grad(splines, box, x_shift, y_shift, z_eval):
    """Reference Phi and native x/y/z derivatives on the box grid, ``(M, box,
    box)`` indexed ``[loc, x-pixel, y-pixel]`` (``z_eval`` None for 2D)."""
    gx, gy, gz = splines
    gi = np.arange(box)
    xc = gi[None, :] - np.asarray(x_shift, float)[:, None]
    yc = gi[None, :] - np.asarray(y_shift, float)[:, None]
    Gx = gx(xc)[:, :, None]
    Gy = gy(yc)[:, None, :]
    dGx = gx.derivative()(xc)[:, :, None]
    dGy = gy.derivative()(yc)[:, None, :]
    if gz is None:
        return Gx * Gy, dGx * Gy, Gx * dGy, None
    z = np.asarray(z_eval, float)[:, None, None]
    Gz = gz(z)
    dGz = gz.derivative()(z)
    return Gx * Gy * Gz, dGx * Gy * Gz, Gx * dGy * Gz, Gx * Gy * dGz


def _ref_crlb(splines, box, amp, xs, ys, ze, off):
    """Closed-form Poisson CRLB variances ``[x, y, (z,) amp, off]`` for one
    localization of the separable Gaussian spline."""
    gz = splines[2]
    phi, dx, dy, dz = _ref_model_grad(
        splines, box, [xs], [ys], None if gz is None else [ze]
    )
    cols = [amp * dx, amp * dy]
    if gz is not None:
        cols.append(amp * dz)
    cols += [phi, np.ones_like(phi)]
    deriv = np.stack([c.reshape(-1) for c in cols], axis=1)
    mu = (off + amp * phi).reshape(-1)
    weight = 1.0 / np.maximum(mu, precision._SPLINE_CRLB_MU_FLOOR)
    return np.diag(np.linalg.pinv((deriv * weight[:, None]).T @ deriv))


def _ref_crlb_lsq(splines, box, amp, xs, ys, ze, off):
    """Closed-form unweighted-least-squares sandwich covariance ``J⁻¹ M J⁻¹``
    (Poisson pixel variance ``σ² = μ``) ``[x, y, (z,) amp, off]`` for one
    localization of the separable Gaussian spline."""
    gz = splines[2]
    phi, dx, dy, dz = _ref_model_grad(
        splines, box, [xs], [ys], None if gz is None else [ze]
    )
    cols = [amp * dx, amp * dy]
    if gz is not None:
        cols.append(amp * dz)
    cols += [phi, np.ones_like(phi)]
    deriv = np.stack([c.reshape(-1) for c in cols], axis=1)
    mu = np.maximum(
        (off + amp * phi).reshape(-1), precision._SPLINE_CRLB_MU_FLOOR
    )
    j = deriv.T @ deriv  # Σ g gᵀ (Gauss-Newton normal matrix)
    m = (deriv * mu[:, None]).T @ deriv  # Σ μ g gᵀ (sandwich meat)
    j_inv = np.linalg.pinv(j)
    return np.diag(j_inv @ m @ j_inv)


class TestCropSplineCalibration:
    """``localize.crop_spline_calibration`` — fitting a smaller, PSF-centered
    box against a larger calibration (no GPU needed).

    The core check evaluates the coefficients through ``_spline_coeff_reshaped``,
    the exact physical layout both real consumers read (Gpufit's ``user_info``
    reorder and the CRLB kernel), and asserts the crop equals the *central*
    lateral intervals of the full calibration — i.e. the crop selects the true
    piecewise-cubic pieces of the calibrated PSF (a faithful central slice), not
    a re-interpolation."""

    @staticmethod
    def _central(reshaped, off, ni, is_2d):
        # _spline_coeff_reshaped -> (n_ch, niy, nix, 4, 4) (2D) or
        # (n_ch, niz, niy, nix, 4, 4, 4) (3D); crop the two lateral axes.
        if is_2d:
            return reshaped[:, off : off + ni, off : off + ni]
        return reshaped[:, :, off : off + ni, off : off + ni]

    @pytest.mark.parametrize(
        "model", ["spline-2d", "spline-3d", "spline-3d-multichannel"]
    )
    def test_crop_matches_central_intervals(self, model):
        cal_box, box = 13, 7
        calib, _ = _gauss_spline_calibration(model=model, box=cal_box)
        cropped = localize.crop_spline_calibration(calib, box)

        # metadata is updated consistently with the smaller grid
        assert cropped["box"] == box
        assert cropped["n_data"][0] == box and cropped["n_data"][1] == box
        assert cropped["n_intervals"][0] == box - 1
        assert cropped["n_intervals"][1] == box - 1

        # the coefficients both consumers read == the full calibration's central
        # lateral intervals (exact selection: no numerical error)
        off, ni = (cal_box - box) // 2, box - 1
        full = precision._spline_coeff_reshaped(calib)
        crop = precision._spline_coeff_reshaped(cropped)
        expected = self._central(full, off, ni, model == "spline-2d")
        assert crop.shape == expected.shape
        np.testing.assert_array_equal(crop, expected)

    def test_z_grid_and_photon_scale_preserved(self):
        calib, _ = _gauss_spline_calibration(
            model="spline-3d", box=13, nz=21, photon_scale=3.7
        )
        cropped = localize.crop_spline_calibration(calib, 7)
        # photon_scale is the full-PSF integral (total photons), not rescaled
        assert cropped["photon_scale"] == 3.7
        # the axial grid is untouched (only the lateral box shrinks): the z
        # data size is preserved and its interval count follows from it
        assert cropped["n_data"][2] == calib["n_data"][2]
        assert cropped["n_intervals"][2] == calib["n_data"][2] - 1

    def test_equal_box_returns_calibration_unchanged(self):
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=BOX)
        assert localize.crop_spline_calibration(calib, BOX) is calib

    def test_larger_box_rejected(self):
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=7)
        with pytest.raises(ValueError, match="no larger than"):
            localize.crop_spline_calibration(calib, 9)

    def test_smaller_box_any_parity_is_cropped(self):
        """A smaller box of the opposite parity is allowed - the crop is then
        off-center by at most half a pixel (a harmless constant shift) - and
        still yields a valid calibration of the requested box."""
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=13)
        cropped = localize.crop_spline_calibration(calib, 8)
        assert cropped["box"] == 8
        assert cropped["n_data"][0] == 8 and cropped["n_data"][1] == 8
        assert (
            cropped["n_intervals"][0] == 7 and cropped["n_intervals"][1] == 7
        )
        # the coefficient grid matches the requested box, and stays consumable
        # by the shared reshape both the fit and the CRLB use
        assert cropped["coefficients"].shape[1:3] == (7, 7)
        reshaped = precision._spline_coeff_reshaped(cropped)
        assert reshaped.shape[2:4] == (7, 7)

    def test_locs_from_fits_auto_crops_smaller_box(self):
        """locs_from_fits_spline handles a smaller box than the calibration by
        cropping internally: passing the full calibration with a small box gives
        exactly the same localizations as passing a pre-cropped one (CPU only,
        no GPU). This confirms the fit/CRLB/reconstruction stay consistent."""
        cal_box, box, n = 13, 7, 4
        full, _ = _gauss_spline_calibration(model="spline-3d", box=cal_box)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 20.0),
                "y": np.full(n, 30.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        theta = np.zeros((n, 5), dtype=np.float32)
        theta[:, 0] = 5000.0  # amplitude
        theta[:, 1] = 0.3  # x_shift
        theta[:, 2] = -0.4  # y_shift
        theta[:, 3] = -full["z_center"] + 2.0  # z_shift
        theta[:, 4] = 12.0  # offset

        auto = localize.locs_from_fits_spline(ids, theta, box, False, full)
        pre_cal = localize.crop_spline_calibration(full, box)
        pre = localize.locs_from_fits_spline(ids, theta, box, False, pre_cal)

        for col in ("x", "y", "z", "photons", "bg", "lpx", "lpy", "lpz"):
            np.testing.assert_array_equal(
                auto[col].to_numpy(), pre[col].to_numpy()
            )
        # the CRLB columns are real (finite, positive) at the smaller box
        for col in ("lpx", "lpy", "lpz"):
            assert np.all(np.isfinite(auto[col])) and np.all(auto[col] > 0)


class TestSplineCalibrationIO:
    """Round-trip of the spline PSF calibration through HDF5 (no GPU)."""

    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_save_load_roundtrip(self, tmp_path, model):
        calib = _fake_spline_calibration(model=model)
        path = str(tmp_path / "psf_spline_calib.hdf5")
        io.save_spline_calibration(path, calib)
        loaded = io.load_spline_calibration(path)

        assert loaded["model"] == model
        np.testing.assert_array_equal(
            loaded["coefficients"], calib["coefficients"]
        )
        assert loaded["coefficients"].dtype == np.float32
        assert list(loaded["n_data"]) == list(calib["n_data"])
        assert list(loaded["n_intervals"]) == list(calib["n_intervals"])
        assert loaded["z_step_nm"] == calib["z_step_nm"]
        assert loaded["Path"] == "test_calibration"

    def test_save_requires_coefficients(self, tmp_path):
        with pytest.raises(ValueError):
            io.save_spline_calibration(
                str(tmp_path / "bad.hdf5"), {"model": "spline-3d"}
            )

    def test_load_rejects_non_spline_file(self, tmp_path):
        path = str(tmp_path / "not_a_spline.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("locs", data=np.zeros(3))
        with pytest.raises(ValueError):
            io.load_spline_calibration(path)


class TestSplineHelpers:
    """Pure-logic tests for the spline backend (run without a GPU)."""

    def test_initial_parameters_shape(self, synthetic_spots):
        spots, _ = synthetic_spots
        calib_3d = _fake_spline_calibration(model="spline-3d")
        init_3d = seeds.initial_parameters_spline(spots, calib_3d)
        assert init_3d.shape == (len(spots), 5)
        assert init_3d.dtype == np.float32
        # z_shift is initialized to -z_init (the in-focus slice), so that the
        # Gpufit model's native z = -z_shift starts at the focus (see
        # seeds.initial_parameters_spline / spline.calibrate_spline). z_init defaults
        # to z_center for this calibration.
        np.testing.assert_allclose(init_3d[:, 3], -calib_3d["z_center"])

        calib_2d = _fake_spline_calibration(model="spline-2d")
        init_2d = seeds.initial_parameters_spline(spots, calib_2d)
        assert init_2d.shape == (len(spots), 4)

    def test_locs_from_fits_spline_3d(self):
        # a realistic (Gaussian) calibration so the CRLB columns are meaningful
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        n = 5
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 20.0),
                "y": np.full(n, 30.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        # theta: [amplitude, x_shift, y_shift, z_shift, offset]
        theta = np.zeros((n, 5), dtype=np.float32)
        theta[:, 0] = 5000.0  # amplitude
        theta[:, 1] = 0.5  # x_shift
        theta[:, 2] = -0.5  # y_shift
        # z_shift is initialized to -z_center (focus); 2 slices past focus
        theta[:, 3] = -calib["z_center"] + 2.0
        theta[:, 4] = 12.0  # offset (bg)

        locs = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)

        box_offset = int(BOX / 2)
        center = (BOX - 1) / 2.0
        np.testing.assert_allclose(locs["x"], 0.5 + center + 20.0 - box_offset)
        np.testing.assert_allclose(
            locs["y"], -0.5 + center + 30.0 - box_offset
        )
        np.testing.assert_allclose(locs["photons"], 5000.0)
        np.testing.assert_allclose(locs["bg"], 12.0)
        # z follows the astigmatism (zfit / z_of_step) convention - it rises
        # with stage z - and is scaled by the magnification factor (default 1.0
        # when absent):
        # z = (z_shift + z_center) * z_step_nm = (2) * 20 = 40 nm
        np.testing.assert_allclose(locs["z"], 40.0)
        # lpx/lpy/lpz are now real Cramer-Rao bounds (finite, positive), not
        # the former 0.01 placeholder, and the CRLB uncertainty columns exist.
        for col in ("lpx", "lpy", "lpz", "photons_unc", "bg_unc"):
            assert col in locs.columns
            assert np.all(np.isfinite(locs[col])) and np.all(locs[col] > 0)

    def test_locs_from_fits_spline_applies_affine_transforms(self):
        """A spline calibration carries the same affine corrections as an
        astigmatism one, chained in stored order."""
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        n = 4
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 20.0),
                "y": np.full(n, 30.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        theta = np.zeros((n, 5), dtype=np.float32)
        theta[:, 0] = 5000.0
        theta[:, 3] = -calib["z_center"] + 2.0
        theta[:, 4] = 12.0

        plain = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)
        corrected_calib = dict(calib)
        shifts = (("astigmatism", 3.0, -1.5), ("chromatic", 5.0, 2.0))
        for kind, dx, dy in shifts:
            matrix = [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]
            lib.append_lateral_transform(
                corrected_calib,
                {"Type": kind, "Transform": affine(matrix).to_dict()},
            )
        moved = localize.locs_from_fits_spline(
            ids, theta, BOX, False, corrected_calib
        )
        np.testing.assert_allclose(moved["x"], plain["x"] + 8.0, atol=1e-4)
        np.testing.assert_allclose(moved["y"], plain["y"] + 0.5, atol=1e-4)
        np.testing.assert_allclose(moved["z"], plain["z"], atol=1e-6)

    def test_multichannel_spline_ignores_affine_transforms(self):
        """Single-channel only: a multichannel calibration that somehow
        carries a correction must not have it applied on top of the
        registration the joint fit already does."""
        calib, _ = _gauss_spline_calibration(model="spline-3d-multichannel")
        n = 3
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 20.0),
                "y": np.full(n, 30.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        theta = np.zeros((n, 5), dtype=np.float32)
        theta[:, 0] = 5000.0
        theta[:, 3] = -calib["z_center"]
        theta[:, 4] = 12.0

        plain = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)
        with_affine = dict(calib)
        lib.append_lateral_transform(
            with_affine,
            {
                "Type": "chromatic",
                "Transform": affine(
                    [[1.0, 0.0, 9.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]]
                ).to_dict(),
            },
        )
        moved = localize.locs_from_fits_spline(
            ids, theta, BOX, False, with_affine
        )
        np.testing.assert_allclose(moved["x"], plain["x"])
        np.testing.assert_allclose(moved["y"], plain["y"])

    def test_locs_from_fits_spline_2d_has_no_z(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        n = 3
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 10.0),
                "y": np.full(n, 10.0),
                "net_gradient": np.full(n, 500.0),
            }
        )
        theta = np.zeros((n, 4), dtype=np.float32)
        theta[:, 0] = 3000.0
        theta[:, 3] = 8.0
        locs = localize.locs_from_fits_spline(ids, theta, BOX, False, calib)
        assert "z" not in locs.columns
        assert "lpz" not in locs.columns
        np.testing.assert_allclose(locs["photons"], 3000.0)
        # 2D still reports lateral + photon/bg CRLB uncertainties
        for col in ("lpx", "lpy", "photons_unc", "bg_unc"):
            assert col in locs.columns
            assert np.all(np.isfinite(locs[col])) and np.all(locs[col] > 0)

    @pytest.mark.skipif(
        localize.CUDA_AVAILABLE,
        reason="ImportError only raised when no CUDA device is present",
    )
    def test_fit_spots_spline_without_gpu_raises(self, synthetic_spots):
        # an explicit use_gpu=True never silently falls back to the CPU; the
        # default (use_gpu=None) does fit here, on whatever device exists
        spots, _ = synthetic_spots
        calib = _fake_spline_calibration(model="spline-3d")
        with pytest.raises(ImportError):
            localize.fit_spots_spline(spots, calib, use_gpu=True)

    def test_spline_kind_covers_every_model(self):
        from picasso.fitting import splinefit

        assert localize._spline_kind("spline-2d") == splinefit.KIND_2D
        assert localize._spline_kind("spline-3d") == splinefit.KIND_3D
        assert (
            localize._spline_kind("spline-3d-multichannel")
            == splinefit.KIND_3D
        )
        assert (
            localize._spline_kind(precision._LINK_XYZ_MODEL)
            == splinefit.KIND_LINK_XYZ
        )
        with pytest.raises(ValueError, match="Unknown spline"):
            localize._spline_kind("nonsense")

    def test_single_channel_spots_are_not_copied(self):
        """The CPU kernels want channel-major spots. For a single channel that
        must stay a view - transposing instead would copy the whole spot stack
        (hundreds of MB for a real movie) for no reason."""
        spots = np.zeros((32, BOX, BOX), np.float32)
        reshaped = precision._spline_channel_major(spots, 1)
        assert reshaped.shape == (32, 1, BOX, BOX)
        assert np.shares_memory(reshaped, spots)

    def test_multichannel_spots_become_channel_major(self):
        spots = np.arange(2 * BOX * BOX * 3, dtype=np.float32).reshape(
            2, BOX, BOX, 3
        )
        reshaped = precision._spline_channel_major(spots, 3)
        assert reshaped.shape == (2, 3, BOX, BOX)
        np.testing.assert_array_equal(reshaped[0, 1], spots[0, :, :, 1])
        with pytest.raises(ValueError, match="channels"):
            precision._spline_channel_major(spots, 2)

    def test_cpu_z_seeds_match_the_gpu_grid(self):
        """CPU and GPU must explore the same axial minima, or the two devices
        disagree for reasons that have nothing to do with the fit."""
        calib = _fake_spline_calibration(model="spline-3d")
        n_starts = localize._default_n_z_starts(calib)
        seeds, apply_seeds = localize._spline_z_seeds(calib, n_starts)
        assert apply_seeds
        # the seed grid both devices build
        n_z = int(calib["n_data"][2])
        np.testing.assert_allclose(
            seeds, np.linspace(-(n_z - 1), 0.0, n_starts)
        )
        assert localize._spline_z_seeds(calib, 1)[1] is False
        calib_2d = _fake_spline_calibration(model="spline-2d")
        assert localize._spline_z_seeds(calib_2d, 9)[1] is False

    def test_schedule_defaults_and_overrides(self):
        from picasso.fitting import splinefit

        assert localize._spline_schedule(True, None, None) == (
            splinefit.TOLERANCE_MULTI_START,
            splinefit.MAX_ITERATIONS_MULTI_START,
        )
        assert localize._spline_schedule(False, None, None) == (
            splinefit.TOLERANCE_SINGLE_START,
            splinefit.MAX_ITERATIONS_SINGLE_START,
        )
        assert localize._spline_schedule(True, 1e-7, 3) == (1e-7, 3)

    @pytest.mark.parametrize("apply_seeds", [False, True])
    def test_cpu_and_gpu_use_the_same_schedule(self, apply_seeds, monkeypatch):
        """Both devices must stop in the same place, or a CPU and a GPU fit of
        the same spots differ for reasons unrelated to the fit. The convergence
        test is relative, so a different tolerance is a real difference - and
        the multi-start ranks its axial seeds on that chi-square.

        Asserted behaviorally rather than by reading the source: whatever
        ``splinefit.convergence_schedule`` returns is what *both* backends are
        handed."""
        from picasso.fitting import splinefit, splinefit_cuda

        shared = splinefit.convergence_schedule(apply_seeds)
        assert localize._spline_schedule(apply_seeds, None, None) == shared
        assert splinefit.resolve_schedule(apply_seeds) == shared

        sentinel = (0.1234, 7)
        monkeypatch.setattr(
            splinefit, "convergence_schedule", lambda seeded: sentinel
        )
        calib = _fake_spline_calibration(model="spline-3d")
        box = calib["box"]
        spots = np.zeros((2, box, box), np.float32)
        n_z_starts = localize._default_n_z_starts(calib) if apply_seeds else 1

        for module, use_gpu in (
            (splinefit, False),
            (splinefit_cuda, True),
        ):
            if use_gpu and not localize.CUDA_AVAILABLE:
                continue
            seen = {}

            def fake_fit_spots(*args, _seen=seen, **kwargs):
                _seen.update(kwargs)
                n = len(args[1])
                return (
                    np.zeros((n, args[5].shape[1])),
                    np.zeros(n),
                    np.zeros(n, np.int32),
                    np.zeros(n, np.int32),
                )

            monkeypatch.setattr(module, "fit_spots", fake_fit_spots)
            localize._run_splinefit(
                spots,
                calib,
                n_z_starts=n_z_starts,
                multiprocess=False,
                use_gpu=use_gpu,
            )
            assert (seen["tolerance"], seen["max_iterations"]) == sentinel

    def test_spline_use_gpu_resolution(self):
        # no GPU on this machine unless Gpufit is installed
        assert localize._spline_use_gpu(None) is localize.CUDA_AVAILABLE
        assert localize._spline_use_gpu(False) is False
        if not localize.CUDA_AVAILABLE:
            with pytest.raises(ImportError, match="use_gpu=False"):
                localize._spline_use_gpu(True)

    def test_affine_transform_roundtrip(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [5.0, 7.0]])
        m_true = np.array([[1.02, -0.03, 3.0], [0.01, 0.98, -2.0]])
        dst = apply_transform(src, m_true)
        m_est = transforms.estimate(src, dst)
        np.testing.assert_allclose(affine_matrix(m_est), m_true, atol=1e-9)

    def test_estimate_affine_needs_three_points(self):
        with pytest.raises(ValueError):
            transforms.estimate(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_channel_roi_residuals_pure_translation_is_constant(self):
        # Detections sit on integer pixels, so a pure translation shifts every
        # box by the same fractional amount - the case the calibration absorbs.
        ids = pd.DataFrame(
            {"x": np.arange(40, 340, 7), "y": np.arange(60, 360, 7)}
        )
        identity = affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        shifted = affine([[1.0, 0.0, 12.3], [0.0, 1.0, -4.8]])
        res = localize.channel_roi_residuals(ids, [identity, shifted])
        assert res.shape == (len(ids), 2, 2)
        # channel 0 is the reference: its box is the detection itself
        np.testing.assert_array_equal(res[:, 0, :], 0.0)
        # frac(12.3) = .3 -> residual .3; frac(-4.8) -> .2
        np.testing.assert_allclose(res[:, 1, 0], 0.3, atol=1e-5)
        np.testing.assert_allclose(res[:, 1, 1], 0.2, atol=1e-5)
        assert np.ptp(res[:, 1, :], axis=0).max() < 1e-5

    def test_channel_roi_residuals_rotation_varies_across_field(self):
        # With any linear part the residual is no longer constant: it sweeps
        # the full +-0.5 px as soon as E@x moves by a pixel across the field.
        ids = pd.DataFrame(
            {"x": np.arange(0, 512, 3), "y": np.arange(0, 512, 3)}
        )
        theta = np.deg2rad(0.5)
        rotated = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
            ]
        )
        res = localize.channel_roi_residuals(
            ids, [affine(np.eye(2, 3)), affine(rotated)]
        )
        assert np.all(np.abs(res) <= 0.5 + 1e-6)
        # spans most of the available range rather than sitting at one value
        assert np.ptp(res[:, 1, 0]) > 0.8
        # and it is not noise: it tracks the mapped position deterministically
        mapped = apply_transform(ids[["x", "y"]].to_numpy(float), rotated)
        np.testing.assert_allclose(
            res[:, 1, :], mapped - np.rint(mapped), atol=1e-6
        )

    def test_get_spots_multichannel_residuals_match_helper(
        self, movie, real_identifications
    ):
        # the extractor's by-product and the standalone helper must agree,
        # otherwise the model is told about a shift the ROIs do not have
        identity = affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        skewed = affine([[1.002, 0.004, 1.4], [-0.004, 1.001, -0.6]])
        transforms = [identity, skewed]
        ids = localize.multichannel_inbounds_ids(
            real_identifications, BOX, [movie, movie], transforms
        )
        spots, res = localize.get_spots_multichannel(
            [movie, movie],
            ids,
            BOX,
            [CAMERA_INFO, CAMERA_INFO],
            transforms,
            return_residuals=True,
        )
        assert spots.shape == (len(ids), BOX, BOX, 2)
        np.testing.assert_allclose(
            res, localize.channel_roi_residuals(ids, transforms), atol=1e-6
        )
        # default stays a bare array, so existing callers are unaffected
        assert isinstance(
            localize.get_spots_multichannel(
                [movie, movie],
                ids,
                BOX,
                [CAMERA_INFO, CAMERA_INFO],
                transforms,
            ),
            np.ndarray,
        )

    @pytest.mark.parametrize("gpu", SPLINE_CRLB_DEVICES)
    def test_spline_crlb_residual_equals_a_lateral_shift(self, gpu):
        # A ROI residual just moves where the spline is evaluated, so it must
        # be indistinguishable from moving the lateral parameter by the same
        # amount. This pins the sign and the placement of the residual term -
        # on whichever device computes it.
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        calib = dict(calib)
        calib["n_channels"] = 1
        calib["coefficients"] = calib["coefficients"][..., 1:2]
        calib["channel_transforms"] = [IDENTITY]
        box = calib["box"]
        rng = np.random.default_rng(1)
        theta = np.zeros((4, 5))  # [amplitude, x, y, z, offset]
        theta[:, 0], theta[:, 3], theta[:, 4] = 500.0, -8.0, 5.0
        theta[:, 1] = rng.uniform(-0.3, 0.3, 4)
        theta[:, 2] = rng.uniform(-0.3, 0.3, 4)
        dx, dy = 0.31, -0.24
        res = np.zeros((4, 1, 2))
        res[:, 0, 0], res[:, 0, 1] = dx, dy
        with_residual = _crlb(
            theta, calib, box, gpu=gpu, mle=True, residuals=res
        )
        moved = theta.copy()
        moved[:, 1] += dx
        moved[:, 2] += dy
        np.testing.assert_allclose(
            with_residual,
            _crlb(moved, calib, box, gpu=gpu, mle=True),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("gpu", SPLINE_CRLB_DEVICES)
    def test_spline_crlb_affine_scales_lateral_variance(self, gpu):
        # The shared shift reaches a channel through its affine, so d(mu)/dp
        # carries A^T. For an isotropic A = s*I the x/y variances must come out
        # exactly s^-2 times the identity case evaluated at the same position,
        # and z must be untouched - which is what the chain rule buys.
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        calib = dict(calib)
        calib["n_channels"] = 1
        calib["coefficients"] = calib["coefficients"][..., 1:2]
        identity = IDENTITY
        calib["channel_transforms"] = [identity]
        box = calib["box"]
        rng = np.random.default_rng(2)
        theta = np.zeros((4, 5))
        theta[:, 0], theta[:, 3], theta[:, 4] = 500.0, -8.0, 5.0
        theta[:, 1] = rng.uniform(-0.3, 0.3, 4)
        theta[:, 2] = rng.uniform(-0.3, 0.3, 4)

        s = 1.05
        scaled = dict(calib)
        scaled["channel_transforms"] = [
            affine([[s, 0.0, 0.0], [0.0, s, 0.0]]).to_dict()
        ]
        crlb_scaled = _crlb(theta, scaled, box, gpu=gpu, mle=True)
        # identity affine reaches the same evaluation point at s * p
        moved = theta.copy()
        moved[:, 1] *= s
        moved[:, 2] *= s
        crlb_identity = _crlb(moved, calib, box, gpu=gpu, mle=True)
        np.testing.assert_allclose(
            crlb_scaled[:, 0], crlb_identity[:, 0] / s**2, rtol=1e-8
        )
        np.testing.assert_allclose(
            crlb_scaled[:, 1], crlb_identity[:, 1] / s**2, rtol=1e-8
        )
        np.testing.assert_allclose(
            crlb_scaled[:, 2], crlb_identity[:, 2], rtol=1e-8
        )

    @pytest.mark.parametrize("gpu", SPLINE_CRLB_DEVICES)
    @pytest.mark.parametrize("link_xyz", [False, True])
    def test_spline_crlb_default_geometry_is_unchanged(self, link_xyz, gpu):
        # identity affine + zero residual must reproduce the pre-correction
        # result exactly, so single-channel output cannot drift
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        if link_xyz:
            calib = localize._as_link_xyz_calibration(calib)
            theta = np.zeros((3, 7))  # [x, y, z, N0, N1, bg0, bg1]
            theta[:, 2] = -8.0
            theta[:, 3:5] = 400.0
            theta[:, 5:] = 5.0
        else:
            theta = np.zeros((3, 5))  # [amplitude, x, y, z, offset]
            theta[:, 0], theta[:, 3], theta[:, 4] = 500.0, -8.0, 5.0
        box = calib["box"]
        n_channels = precision._spline_n_channels(calib)
        explicit = _crlb(
            theta,
            calib,
            box,
            gpu=gpu,
            mle=True,
            residuals=np.zeros((len(theta), n_channels, 2)),
        )
        implied = _crlb(theta, calib, box, gpu=gpu, mle=True)
        assert np.isfinite(implied).all()
        np.testing.assert_array_equal(explicit, implied)

    def test_spline_channel_jacobians_default_to_identity(self):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=3
        )
        jac = precision._spline_channel_jacobians(None, 5, 3, calib)
        np.testing.assert_array_equal(
            jac, np.tile([1.0, 0.0, 0.0, 1.0], (5, 3, 1))
        )
        # a calibration without usable transforms falls back to the identity,
        # matching what the CUDA models do with no affine block
        stripped = dict(calib)
        stripped.pop("channel_transforms")
        np.testing.assert_array_equal(
            precision._spline_channel_jacobians(None, 5, 3, stripped), jac
        )

    def test_default_n_z_starts_from_calibration_depth(self):
        # one seed per ~20 calibration planes, bounded; 2D has no z to be
        # degenerate in and must stay on a single start
        calib = _fake_spline_calibration(model="spline-3d")
        assert localize._default_n_z_starts(calib) == 5
        assert (
            localize._default_n_z_starts(
                _fake_spline_calibration(model="spline-2d")
            )
            == 1
        )
        deep = dict(calib)
        deep["n_data"] = [7, 7, 600]  # far past the upper bound
        assert localize._default_n_z_starts(deep) == localize._Z_STARTS_MAX
        shallow = dict(calib)
        shallow["n_data"] = [7, 7, 4]
        assert localize._default_n_z_starts(shallow) == localize._Z_STARTS_MIN
        # a calibration that does not describe a z axis cannot be multi-started
        assert localize._default_n_z_starts({"n_data": [7, 7]}) == 1

    @staticmethod
    def _capture_kernel_call(monkeypatch):
        """Record the arguments ``_run_splinefit`` hands to the kernel.

        The multi-start runs *inside* the per-spot kernel on both devices, so
        it is no longer visible as a repeated call - what has to be asserted is
        the seed grid and the schedule the kernel is given."""
        from picasso.fitting import splinefit

        calls = []

        def fake_fit_spots(*args, **kwargs):
            calls.append((args, kwargs))
            n_spots = len(args[1])
            n_params = args[5].shape[1]
            return (
                np.zeros((n_spots, n_params)),
                np.full(n_spots, 1.0),
                np.zeros(n_spots, np.int32),
                np.ones(n_spots, np.int32),
            )

        monkeypatch.setattr(splinefit, "fit_spots", fake_fit_spots)
        return calls

    @pytest.mark.parametrize("model", ["spline-3d", "spline-3d-multichannel"])
    def test_fit_spots_multistarts_by_default(self, monkeypatch, model):
        calib = _fake_spline_calibration(model=model)
        expected = localize._default_n_z_starts(calib)
        assert expected > 1
        n_channels = precision._spline_n_channels(calib)
        box = calib["box"]
        spots = np.zeros((3, box, box), np.float32)
        if model == "spline-3d-multichannel":
            spots = np.zeros((3, box, box, n_channels), np.float32)

        calls = self._capture_kernel_call(monkeypatch)
        localize.fit_spots_splinefit(
            spots, calib, mle=True, multiprocess=False
        )
        assert len(calls) == 1
        args, kwargs = calls[0]
        z_seeds, apply_seeds = args[6], args[7]
        assert apply_seeds is True
        # the seeds must actually differ, spanning the calibration stack
        assert len(set(z_seeds.tolist())) == expected
        assert z_seeds.min() == pytest.approx(-(calib["n_data"][2] - 1))
        assert z_seeds.max() == pytest.approx(0.0)
        # seeded runs must use the tight convergence, else neighboring axial
        # minima are indistinguishable and the multi-start is pointless
        assert kwargs["tolerance"] == 1e-4
        assert kwargs["max_iterations"] == 100

    def test_ratiometric_multistarts_every_hypothesis(self, monkeypatch):
        # color is decided by comparing hypotheses' residuals, so they all
        # need the same axial search - otherwise a hypothesis can win merely by
        # having stumbled into a better z minimum
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        n_starts = localize._default_n_z_starts(calib)
        n_hyp = 3
        calls = []

        def fake_multistart(spots_, calibration, **kw):
            calls.append(kw["n_z_starts"])
            n = len(spots_)
            return (
                np.zeros((n, 5), np.float32),
                np.full(n, 1.0),
                np.ones(n, bool),
                np.ones(n, np.int32),
            )

        monkeypatch.setattr(
            # the device-agnostic dispatcher, which is what the ratiometric
            # fitter calls (it routes to the GPU or CPU multi-start)
            localize,
            "_fit_spline_multistart",
            fake_multistart,
        )
        # exercise only the hypothesis loop; locs building needs a real fit
        monkeypatch.setattr(
            localize,
            "locs_from_fits_spline",
            lambda ids, theta, box, em, cal, **kw: pd.DataFrame(
                {"frame": np.asarray(ids["frame"])}
            ),
        )
        box = calib["box"]
        n_spots = 4
        spots = np.zeros((n_spots, box, box, 2), np.float32)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n_spots),
                "x": np.full(n_spots, 20),
                "y": np.full(n_spots, 20),
                "net_gradient": np.ones(n_spots),
            }
        )

        def fake_get_spots(*args, **kwargs):
            """Mirror the real return contract, including ``return_variance``.

            No camera calibration is passed here, so the variance is None -
            the plain Poisson model, which is what this test is about."""
            out = (spots,)
            if kwargs.get("return_residuals"):
                out += (np.zeros((n_spots, 2, 2), np.float32),)
            if kwargs.get("return_variance"):
                out += (None,)
            if kwargs.get("return_jacobians"):
                out += (np.tile([1.0, 0.0, 0.0, 1.0], (n_spots, 2, 1)),)
            return out[0] if len(out) == 1 else out

        monkeypatch.setattr(localize, "get_spots_multichannel", fake_get_spots)
        monkeypatch.setattr(
            localize, "multichannel_inbounds_ids", lambda i, *a, **kw: i
        )
        localize.fit_spline_multichannel_ratiometric(
            [None, None],
            [CAMERA_INFO, CAMERA_INFO],
            ids,
            box,
            calib,
            photon_ratios=np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
        )
        assert calls == [n_starts] * n_hyp

    def test_fit_spots_single_start_is_still_reachable(self, monkeypatch):
        calib = _fake_spline_calibration(model="spline-3d")
        box = calib["box"]
        calls = self._capture_kernel_call(monkeypatch)
        localize.fit_spots_splinefit(
            np.zeros((3, box, box), np.float32),
            calib,
            n_z_starts=1,
            multiprocess=False,
        )
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[7] is False  # apply_seeds
        # the single-start path keeps the loose defaults
        assert kwargs["tolerance"] == 1e-2
        assert kwargs["max_iterations"] == 20

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_as_link_xyz_calibration_accepts_supported_channels(
        self, n_channels
    ):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        link = localize._as_link_xyz_calibration(calib)
        assert link["model"] == precision._LINK_XYZ_MODEL
        # shallow copy: the original is untouched and the (large) coefficient
        # table is shared, not duplicated
        assert calib["model"] == "spline-3d-multichannel"
        assert link["coefficients"] is calib["coefficients"]

    @pytest.mark.parametrize("n_channels", [1, 7, 12])
    def test_as_link_xyz_calibration_rejects_unsupported_channels(
        self, n_channels
    ):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        with pytest.raises(ValueError, match="2 to 6 channels"):
            localize._as_link_xyz_calibration(calib)

    def test_as_link_xyz_calibration_requires_multichannel(self):
        with pytest.raises(ValueError):
            localize._as_link_xyz_calibration(
                _fake_spline_calibration(model="spline-3d")
            )

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_initial_parameters_link_xyz_per_channel(self, n_channels):
        calib = localize._as_link_xyz_calibration(
            _fake_spline_calibration(
                model="spline-3d-multichannel", n_channels=n_channels
            )
        )
        rng = np.random.default_rng(0)
        spots = rng.random((4, BOX, BOX, n_channels)).astype(np.float32)
        # scale the channels apart so a transposed or channel-major/minor
        # mix-up cannot pass
        spots = spots * (1.0 + np.arange(n_channels, dtype=np.float32))
        init = seeds.initial_parameters_spline(spots, calib)
        # [x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]
        assert init.shape == (4, 3 + 2 * n_channels)
        np.testing.assert_allclose(init[:, :2], 0.0)
        np.testing.assert_allclose(init[:, 2], -calib["z_center"])
        ch_max = spots.max(axis=(1, 2))
        ch_min = spots.min(axis=(1, 2))
        np.testing.assert_allclose(
            init[:, 3 : 3 + n_channels], ch_max - ch_min, rtol=1e-5
        )
        np.testing.assert_allclose(
            init[:, 3 + n_channels :], ch_min, rtol=1e-5
        )

    def test_initial_parameters_multichannel_stacked(self):
        calib = _fake_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        # channel-stacked spots (n, box, box, n_channels)
        spots = (
            np.random.default_rng(0)
            .random((4, BOX, BOX, 2), dtype=np.float64)
            .astype(np.float32)
        )
        init = seeds.initial_parameters_spline(spots, calib)
        assert init.shape == (4, 5)
        # z_shift initialized to -z_init (= -z_center here); see the
        # single-channel test above.
        np.testing.assert_allclose(init[:, 3], -calib["z_center"])

    def test_get_spots_multichannel_identity(
        self, movie, real_identifications
    ):
        single = localize.get_spots(
            movie, real_identifications, BOX, CAMERA_INFO
        )
        identity = affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        stacked = localize.get_spots_multichannel(
            [movie, movie],
            real_identifications,
            BOX,
            [CAMERA_INFO, CAMERA_INFO],
            [identity, identity],
        )
        assert stacked.shape == (len(real_identifications), BOX, BOX, 2)
        # identity transform => each channel equals the single-movie extraction
        np.testing.assert_array_equal(stacked[..., 0], single)
        np.testing.assert_array_equal(stacked[..., 1], single)

    def test_multichannel_inbounds_ids_drops_edge_spots(self, movie):
        """A detection whose box falls outside the frame in any channel is
        dropped, so the joint extractor never reads an out-of-bounds box."""
        height, width = movie.shape[1], movie.shape[2]
        r = BOX // 2
        ids = pd.DataFrame(
            {
                "frame": [0, 0, 0, 0],
                # centered (ok), top edge (box off the top), left edge, and a
                # spot that only leaves the frame once mapped in channel 1.
                "x": [width // 2, width // 2, 0, width - r - 1],
                "y": [height // 2, 0, height // 2, height // 2],
                "net_gradient": [1.0, 1.0, 1.0, 1.0],
            }
        )
        identity = affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # channel 1 shifts +r in x, pushing the last (near-right-edge) spot out
        shift = affine([[1.0, 0.0, float(r + 1)], [0.0, 1.0, 0.0]])
        kept = localize.multichannel_inbounds_ids(
            ids, BOX, [movie, movie], [identity, shift]
        )
        # only the centered spot survives in both channels
        assert len(kept) == 1
        assert int(kept["x"].iloc[0]) == width // 2
        assert int(kept["y"].iloc[0]) == height // 2
        # extraction on the filtered ids must not raise
        stacked = localize.get_spots_multichannel(
            [movie, movie],
            kept,
            BOX,
            [CAMERA_INFO, CAMERA_INFO],
            [identity, shift],
        )
        assert stacked.shape == (1, BOX, BOX, 2)


class TestCrossChannelLinking:
    """Reference detections must be reduced to those found in every channel:
    the joint multichannel model shares x, y, z across channels, so a spot
    missing from one channel would be fitted there against background only."""

    IDENTITY = affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    SHIFT_X = affine([[1.0, 0.0, 100.0], [0.0, 1.0, 0.0]])

    @staticmethod
    def _ids(frames, xs, ys):
        return pd.DataFrame(
            {
                "frame": np.asarray(frames, dtype=int),
                "x": np.asarray(xs, dtype=float),
                "y": np.asarray(ys, dtype=float),
                "net_gradient": np.ones(len(xs), dtype=float),
            }
        )

    def test_requires_a_match_in_every_channel(self):
        ref = self._ids([0, 0, 0, 1], [10, 20, 30, 40], [10, 20, 30, 40])
        # channel 1 lives 100 px to the right; it sees ref spots 0, 2 and 3
        ch1 = self._ids([0, 0, 1], [110.4, 130.2, 140], [10.2, 30, 40])
        # channel 2 is aligned with the reference; it sees ref spots 0 and 1
        ch2 = self._ids([0, 0], [10.1, 20], [9.9, 20])
        linked = localize.link_identifications_multichannel(
            [ref, ch1, ch2], [self.IDENTITY, self.SHIFT_X, self.IDENTITY], 3.0
        )
        # only ref spot 0 is present in all three channels
        assert list(linked) == [True, False, False, False]

    def test_pairing_is_one_to_one(self):
        """Two reference spots near one detection: only the closer one links."""
        ref = self._ids([0, 0], [10, 11], [10, 10])
        target = self._ids([0], [10.2], [10])
        linked = localize.link_identifications_multichannel(
            [ref, target], [self.IDENTITY, self.IDENTITY], 3.0
        )
        assert list(linked) == [True, False]

    def test_tolerance_and_frame_order(self):
        ref = self._ids([1, 0, 0], [40, 10, 20], [40, 10, 20])
        ch1 = self._ids([0, 1], [10.1, 40], [10, 40])
        args = ([ref, ch1], [self.IDENTITY, self.IDENTITY])
        # unsorted reference frames are handled (matching is per frame)
        assert list(
            localize.link_identifications_multichannel(*args, 3.0)
        ) == [
            True,
            True,
            False,
        ]
        # a tight tolerance rejects the 0.1 px mismatch but keeps the exact one
        assert list(
            localize.link_identifications_multichannel(*args, 0.05)
        ) == [True, False, False]

    def test_unidentified_channels_are_skipped(self):
        ref = self._ids([0, 0], [10, 20], [10, 20])
        ch1 = self._ids([0], [110], [10])
        transforms = [self.IDENTITY, self.SHIFT_X, self.IDENTITY]
        # channel 2 was never identified -> link against channel 1 only
        linked = localize.link_identifications_multichannel(
            [ref, ch1, None], transforms, 3.0
        )
        assert list(linked) == [True, False]
        # no other channel identified at all -> nothing is linked
        linked = localize.link_identifications_multichannel(
            [ref, None, None], transforms, 3.0
        )
        assert not linked.any()

    def test_filter_wrapper_counts_and_passthrough(self):
        ref = self._ids([0, 0], [10, 20], [10, 20])
        ch1 = self._ids([0], [110], [10])
        kept, n_kept, n_total = localize.filter_linked_identifications(
            [ref, ch1], [self.IDENTITY, self.SHIFT_X], box=2
        )
        assert (n_kept, n_total) == (1, 2)
        assert len(kept) == 1 and kept["x"].iloc[0] == 10
        # without identifications in any other channel the table is unchanged,
        # so an un-identified set degrades to the unfiltered behavior
        same, n_kept, n_total = localize.filter_linked_identifications(
            [ref, None], [self.IDENTITY, self.IDENTITY], box=2
        )
        assert same is ref and (n_kept, n_total) == (2, 2)

    def test_progress_is_monotone_across_channels(self):
        ref = self._ids([0, 0, 1], [10, 20, 30], [10, 20, 30])
        ch1 = self._ids([0, 1], [110, 130], [10, 30])
        ch2 = self._ids([0, 1], [10, 30], [10, 30])
        seen = []
        localize.link_identifications_multichannel(
            [ref, ch1, ch2],
            [self.IDENTITY, self.SHIFT_X, self.IDENTITY],
            3.0,
            progress_callback=seen.append,
        )
        # one 0 -> n_ref progression for the whole linking pass, not one per
        # channel (the GUI shows this as a single status count)
        assert seen and seen[-1] == len(ref)
        assert all(b >= a for a, b in zip(seen, seen[1:]))


class TestCrossRegionLinkingSplitFov:
    """Split-FOV: one movie is identified as a whole, so a single table holds
    every region's detections. They must be split by region and linked across
    regions, leaving one fitted spot per molecule - not one per detection."""

    # two side-by-side 32x32 regions of a 32x64 movie; channel 1 sits half a
    # pixel to the right and a quarter pixel up of the perfect split
    REGIONS = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]
    AFFINES = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]],
    ]

    def _calibration(self, reference=0):
        transforms = localize.compose_region_transforms(
            [localize._normalize_rect(r) for r in self.REGIONS],
            [affine(a) for a in self.AFFINES],
        )
        return {
            "model": "spline-3d-multichannel",
            "split_fov": True,
            "n_channels": 2,
            "reference": reference,
            "regions": self.REGIONS,
            "channel_registration": [
                affine(a).to_dict() for a in self.AFFINES
            ],
            "channel_transforms": [t.to_dict() for t in transforms],
        }, transforms

    @staticmethod
    def _ids(xy):
        xy = np.asarray(xy, dtype=float)
        return pd.DataFrame(
            {
                "frame": np.zeros(len(xy), dtype=int),
                "x": xy[:, 0],
                "y": xy[:, 1],
                "net_gradient": np.ones(len(xy), dtype=float),
            }
        )

    def _detections(self, transforms):
        """Three reference-region spots, two of which have a counterpart in
        region 1, plus one unpaired detection in region 1."""
        ref = np.array([[5.0, 5.0], [10.0, 20.0], [25.0, 8.0]])
        partners = apply_transform(ref[:2], transforms[1]) + 0.3
        return self._ids(np.vstack([ref, partners, [[50.0, 30.0]]])), ref

    def test_keeps_reference_spots_found_in_every_region(self):
        calib, transforms = self._calibration()
        ids, ref = self._detections(transforms)
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(ids, calib, box=7)
        )
        # 6 detections over both regions -> 3 in the reference region -> the 2
        # molecules seen in both. The count the fit progress must show.
        assert len(ids) == 6
        assert (n_kept, n_total) == (2, 3)
        assert np.allclose(linked[["x", "y"]].to_numpy(), ref[:2])

    def test_regions_may_be_re_placed(self):
        """Drawn ROIs shifted off the calibration positions link the same."""
        calib, transforms = self._calibration()
        ids, _ = self._detections(transforms)
        shifted = ids.copy()
        shifted["x"] += 3.0
        shifted["y"] += 2.0
        regions = [
            [[y0 + 2, x0 + 3], [y1 + 2, x1 + 3]]
            for (y0, x0), (y1, x1) in self.REGIONS
        ]
        _, n_kept, n_total = localize.filter_linked_identifications_split_fov(
            shifted, calib, box=7, regions=regions
        )
        assert (n_kept, n_total) == (2, 3)

    def test_non_zero_reference_region(self):
        """A calibration whose reference is not region 0 links against it."""
        calib, transforms = self._calibration(reference=1)
        ids, _ = self._detections(transforms)
        # transforms must map the reference region (1) into every region
        inverse = affine([[1.0, 0.0, -32.5], [0.0, 1.0, 0.25]])
        calib["channel_transforms"] = [inverse.to_dict(), IDENTITY]
        linked, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(ids, calib, box=7)
        )
        # region 1 holds the 2 counterparts plus one unpaired detection
        assert (n_kept, n_total) == (2, 3)
        assert (linked["x"] > 32).all()

    def test_misregistration_links_nothing(self):
        calib, transforms = self._calibration()
        ids, _ = self._detections(transforms)
        # the stored transform puts channel 1 40 px off its true position
        calib["channel_transforms"] = [
            IDENTITY,
            affine([[1.0, 0.0, 32.5], [0.0, 1.0, 39.75]]).to_dict(),
        ]
        with pytest.warns(UserWarning, match="registration"):
            _, n_kept, n_total = (
                localize.filter_linked_identifications_split_fov(
                    ids, calib, box=7
                )
            )
        assert (n_kept, n_total) == (0, 3)

    def test_other_regions_unidentified_passes_through(self):
        """Identifying only the reference region (e.g. a restricted ROI) keeps
        every reference detection, as for separately loaded channels."""
        calib, _ = self._calibration()
        ids = self._ids([[5.0, 5.0], [10.0, 20.0]])
        kept, n_kept, n_total = (
            localize.filter_linked_identifications_split_fov(ids, calib, box=7)
        )
        assert (n_kept, n_total) == (2, 2)
        assert len(kept) == 2

    def test_confine_to_region(self):
        ids = self._ids([[5.0, 5.0], [40.0, 5.0], [31.9, 31.9]])
        inside = localize.confine_to_region(ids, self.REGIONS[0])
        # the region is half-open: [y_min, y_max) x [x_min, x_max)
        assert list(inside["x"]) == [5.0, 31.9]

    def test_geometry_requires_a_split_fov_calibration(self):
        with pytest.raises(ValueError, match="split-FOV"):
            localize.split_fov_fit_geometry({"model": "spline-3d"})


def _synthetic_spline_3d_calibration(box=BOX, nz=41):
    """Build a 3D spline calibration from a synthetic astigmatic PSF,
    mirroring the reference pyGpufit splinefit_3d example; returns
    (calibration, template, amplitude, offset)."""
    x = np.arange(box, dtype=np.float32)
    y = np.arange(box, dtype=np.float32)
    amplitude, offset = 100.0, 10.0
    s0 = box / 6.0
    template = np.zeros((box, box, nz), dtype=np.float32)
    for k in range(nz):
        sx = s0 * (1.0 + 0.4 * (k - nz / 2) / nz)
        sy = s0 * (1.0 - 0.4 * (k - nz / 2) / nz)
        gx = np.exp(-0.5 * ((x - (box - 1) / 2) / sx) ** 2)
        gy = np.exp(-0.5 * ((y - (box - 1) / 2) / sy) ** 2)
        template[:, :, k] = np.outer(gx, gy)
    coefficients = spline.spline_coefficients(template)
    n_intervals = np.array(template.shape) - 1
    calib = {
        "model": "spline-3d",
        "coefficients": coefficients,
        "n_data": [box, box, nz],
        "n_intervals": [int(i) for i in n_intervals],
        "oversampling": 1.0,
        "z_center": (nz - 1) / 2,
        "z_step_nm": 20.0,
        "effective_sigma": s0,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": PIXELSIZE,
        "Path": "synthetic",
    }
    return calib, template, amplitude, offset


def _eval_spline_psf(calib, x_shift, y_shift, z_eval=None):
    """The calibration PSF on the box grid, via the production CPU evaluator
    (``splinefit._eval_spline_3d`` / ``_eval_spline_2d`` on
    ``precision._spline_coeff_reshaped``'s view).

    Returns a ``(box, box)`` image indexed ``[x-pixel, y-pixel]`` - the
    orientation of the templates these synthetic calibrations are built from
    (``template[:, :, k] = np.outer(gx, gy)``), not the ``[y, x]`` layout of
    real spot data."""
    box = calib["n_data"][0]
    coeff = precision._spline_coeff_reshaped(calib)
    out = np.empty((box, box))
    for i in range(box):  # i = x-pixel
        for j in range(box):  # j = y-pixel
            if z_eval is None:
                out[i, j] = splinefit._eval_spline_2d(
                    coeff, 0, i - x_shift, j - y_shift
                )[0]
            else:
                out[i, j] = splinefit._eval_spline_3d(
                    coeff, 0, i - x_shift, j - y_shift, z_eval
                )[0]
    return out


class TestSplineCoefficients:
    """Coefficient computation (``spline.spline_coefficients``) and its
    evaluation (plain CPU, no GPU/CUDA)."""

    def test_spline_coefficients_roundtrip(self):
        """Evaluating the computed coefficients at the grid nodes must
        reproduce the source template (validates coefficient layout /
        flatten order)."""
        calib, template, _, _ = _synthetic_spline_3d_calibration()
        nz = calib["n_data"][2]
        values = np.stack(
            [_eval_spline_psf(calib, 0.0, 0.0, float(k)) for k in range(nz)],
            axis=-1,
        )
        np.testing.assert_allclose(values, template, atol=1e-3)

    def test_spline_crlb_real_coefficients(self):
        """_spline_crlb runs on a real calibration and yields finite,
        positive precisions (exercises the actual spline-evaluation path, not
        the analytic fake used by TestSplineCRLB)."""
        calib, _, amplitude, offset = _synthetic_spline_3d_calibration()
        box = calib["n_data"][0]
        z_focus = calib["z_center"]
        # a centered molecule a few slices below focus
        theta = np.array(
            [[amplitude, 0.0, 0.0, -(z_focus - 5.0), offset]], np.float64
        )
        crlb = precision._spline_crlb(theta, calib, box)[0]
        assert np.all(np.isfinite(crlb)) and np.all(crlb > 0)

    def test_evaluation_matches_scipy(self):
        """Authoritative layout check: the fitting kernels' view of a real
        ``spline.spline_coefficients`` table must equal an independent scipy
        evaluation of the tensor-product natural spline at sub-pixel shifts.

        The analytic calibrations in test_splinefit.py build their coefficient
        buffer by hand, so they pin the kernels against that hand-written
        layout; only this test feeds the kernels a genuine
        ``spline_coefficients`` output and so can catch the two disagreeing."""
        calib, template, _, _ = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        rng = np.random.default_rng(1)
        m = 12
        xs = rng.uniform(-0.5, 0.5, m)
        ys = rng.uniform(-0.5, 0.5, m)
        ze = rng.uniform(5, nz - 6, m)
        # Independent reference: the tensor-product natural spline evaluated
        # axis by axis with scipy (a natural interpolant is separable, so
        # interpolating in z first and then laterally is the same function).
        grid = np.arange(box, dtype=np.float64)
        cs_z = CubicSpline(
            np.arange(nz),
            template.astype(np.float64),
            axis=2,
            bc_type="natural",
        )
        for k in range(m):
            slab = cs_z(ze[k])  # (box, box) at native z
            cs_x = CubicSpline(grid, slab, axis=0, bc_type="natural")
            cols = cs_x(grid - xs[k])  # (box, box), rows = x pixels
            cs_y = CubicSpline(grid, cols, axis=1, bc_type="natural")
            ref = cs_y(grid - ys[k])  # (box, box) indexed [x, y]
            got = _eval_spline_psf(calib, xs[k], ys[k], ze[k])
            np.testing.assert_allclose(got, ref, atol=1e-3)


def _synthetic_spline_2d_calibration(box=13, sigma=1.4):
    """Build a 2D (16-coefficient) spline calibration from a single isotropic
    Gaussian slice. Isotropic -> swap-invariant in x/y, so recovery
    assertions are convention-agnostic. Returns
    ``(calibration, amplitude, offset)``."""
    x = np.arange(box, dtype=np.float32)
    g = np.exp(-0.5 * ((x - (box - 1) / 2) / sigma) ** 2)
    template = np.outer(g, g).astype(np.float32)
    n_intervals = np.array(template.shape) - 1
    coefficients = spline.spline_coefficients(template)
    calib = {
        "model": "spline-2d",
        "coefficients": coefficients,
        "n_data": [box, box],
        "n_intervals": [int(i) for i in n_intervals],
        "oversampling": 1.0,
        "z_center": 0.0,
        "z_step_nm": 20.0,
        "effective_sigma": sigma,
        "photon_scale": 1.0,
        "box": box,
        "pixelsize": PIXELSIZE,
        "Path": "synthetic-2d",
    }
    return calib, 100.0, 10.0


class TestSplineFit:
    """End-to-end spline fitting. ``fit_spots_spline`` picks whatever device is
    available (``use_gpu=None``), so this runs on the CPU in a GPU-less
    environment and on the GPU otherwise.

    Spline theta is ``[amplitude, x_shift, y_shift, z_shift, offset]`` for 3D
    and ``[amplitude, x_shift, y_shift, offset]`` for 2D. The exact x/y and z
    conventions of a manually built calibration are subtle (the astigmatic
    PSF couples an x<->y swap with a z mirror away from focus), so recovery
    assertions here stay convention-agnostic: amplitude/offset, symmetric
    (diagonal) sub-pixel shifts, monotonic z, LSE/MLE agreement, and model
    round-trip residuals."""

    def test_fit_spots_spline_3d(self):
        calib, template, amplitude, offset = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amplitude * template[:, :, z_slice] + offset).astype(
            np.float32
        )
        spots = np.stack([spot] * 4)
        theta = localize.fit_spots_spline(spots, calib)
        assert theta.shape == (len(spots), 5)
        # The fitted z_shift maps to the taken native slice with magnitude
        # ~= z_slice; the sign encodes the native_z = -z_shift convention
        # (see test_spline_crlb_native_z_reconstruction), so we check the
        # magnitude here to stay convention-agnostic.
        np.testing.assert_allclose(np.abs(theta[:, 3]), z_slice, atol=2.0)

    def test_spline_crlb_native_z_reconstruction(self):
        """Definitive check of the native_z = -z_shift convention that
        ``_spline_crlb`` relies on for lpz: re-evaluating the spline at
        ``-z_shift`` must reproduce the fitted spot better than at ``+z_shift``.
        """
        calib, template, amplitude, offset = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amplitude * template[:, :, z_slice] + offset).astype(
            np.float32
        )
        amp, xs, ys, zs, off = localize.fit_spots_spline(
            np.stack([spot]), calib
        )[0]
        box, _, nz = calib["n_data"]

        def model(native_z):
            zc = np.clip(native_z, 0.0, nz - 1)
            phi = _eval_spline_psf(calib, xs, ys, zc)
            return off + amp * phi

        res_minus = float(np.mean((model(-zs) - spot) ** 2))
        res_plus = float(np.mean((model(+zs) - spot) ** 2))
        assert res_minus < res_plus

    def test_fit_spots_spline_box_mismatch(self):
        calib, _, _, _ = _synthetic_spline_3d_calibration(box=BOX)
        wrong_box = BOX + 2
        spots = np.zeros((2, wrong_box, wrong_box), dtype=np.float32)
        with pytest.raises(ValueError):
            localize.fit_spots_spline(spots, calib)

    def test_spline_3d_recovers_amplitude_offset_at_focus(self):
        """A centered in-focus spot recovers its amplitude, offset and (near)
        zero lateral shift. At focus the PSF is ~isotropic, so the lateral
        recovery is convention-agnostic."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        z_slice = int(calib["z_center"])
        spot = (amp * template[:, :, z_slice] + off).astype(np.float32)
        theta = localize.fit_spots_spline(np.stack([spot] * 3), calib)
        np.testing.assert_allclose(theta[:, 0], amp, rtol=1e-3)  # amplitude
        np.testing.assert_allclose(theta[:, 4], off, atol=1e-2)  # offset
        np.testing.assert_allclose(theta[:, 1], 0.0, atol=1e-2)  # x_shift
        np.testing.assert_allclose(theta[:, 2], 0.0, atol=1e-2)  # y_shift

    def test_spline_3d_recovers_diagonal_subpixel_shift(self):
        """Sub-pixel shifts recover exactly. Uses symmetric (dx == dy) shifts
        so the result is invariant to the x<->y axis convention."""
        calib, _, amp, off = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        z_focus = np.float32(calib["z_center"])
        shifts = [0.0, 0.3, -0.4, 0.45]
        spots = []
        for d in shifts:
            phi = _eval_spline_psf(calib, d, d, z_focus)
            spots.append((off + amp * phi).astype(np.float32))
        theta = localize.fit_spots_spline(np.stack(spots), calib)
        np.testing.assert_allclose(theta[:, 1], shifts, atol=5e-3)
        np.testing.assert_allclose(theta[:, 2], shifts, atol=5e-3)

    def test_spline_3d_native_z_monotonic(self):
        """The recovered native z (= -z_shift) advances monotonically as the
        input is taken from successive slices of the calibration stack -
        convention-agnostic evidence that the axial fit tracks defocus."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        slices = [12, 16, 20, 24, 28]
        spots = np.stack(
            [
                (amp * template[:, :, k] + off).astype(np.float32)
                for k in slices
            ]
        )
        theta = localize.fit_spots_spline(spots, calib)
        native_z = -theta[:, 3]
        diffs = np.diff(native_z)
        # strictly monotonic (all steps share one sign)
        assert np.all(diffs > 0) or np.all(diffs < 0)

    def test_spline_3d_mle_agrees_with_lse(self):
        """Unlike the Gaussian MLE, the spline MLE estimator converges cleanly
        here; its amplitude/offset must agree with the least-squares fit."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        slices = [14, 20, 26]
        spots = np.stack(
            [
                (amp * template[:, :, k] + off).astype(np.float32)
                for k in slices
            ]
        )
        lse = localize.fit_spots_spline(spots, calib, mle=False)
        mle = localize.fit_spots_spline(spots, calib, mle=True)
        np.testing.assert_allclose(
            mle[:, 0], lse[:, 0], rtol=1e-3
        )  # amplitude
        np.testing.assert_allclose(mle[:, 4], lse[:, 4], atol=1e-2)  # offset
        np.testing.assert_allclose(mle[:, 3], lse[:, 3], atol=0.2)  # z_shift

    def test_spline_3d_roundtrip_reproduces_focus_spot(self):
        """Convention-independent correctness check: re-evaluating the spline
        at the fitted parameters reproduces the input spot to the noise
        floor."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        z_slice = int(calib["z_center"])
        spot = (amp * template[:, :, z_slice] + off).astype(np.float32)
        a, xs, ys, zs, o = localize.fit_spots_spline(np.stack([spot]), calib)[
            0
        ]
        phi = _eval_spline_psf(calib, xs, ys, np.clip(-zs, 0, nz - 1))
        model = o + a * phi
        rms = np.sqrt(np.mean((model - spot) ** 2))
        assert rms < 0.5  # amplitude is 100 -> < 0.5% of peak

    def test_spline_2d_recovers_amplitude_offset_shift(self):
        """The 2D spline model (16 coefficients, no z) recovers amplitude,
        offset and symmetric sub-pixel shift."""
        calib, amp, off = _synthetic_spline_2d_calibration()
        box = calib["n_data"][0]
        shifts = [0.0, 0.3, -0.35]
        spots = []
        for d in shifts:
            phi = _eval_spline_psf(calib, d, d)
            spots.append((off + amp * phi).astype(np.float32))
        theta = localize.fit_spots_spline(np.stack(spots), calib)
        assert theta.shape == (len(shifts), 4)  # [amp, x_shift, y_shift, off]
        np.testing.assert_allclose(theta[:, 0], amp, rtol=1e-3)
        np.testing.assert_allclose(theta[:, 3], off, atol=1e-2)
        np.testing.assert_allclose(theta[:, 1], shifts, atol=5e-3)
        np.testing.assert_allclose(theta[:, 2], shifts, atol=5e-3)

    def test_spline_3d_locs_end_to_end(self):
        """fit_spots_spline + locs_from_fits_spline yields a valid
        localizations frame with finite, positive CRLB precisions and a z
        column for the 3D model."""
        calib, template, amp, off = _synthetic_spline_3d_calibration()
        n = 4
        z_slice = int(calib["z_center"])
        spot = (amp * template[:, :, z_slice] + off).astype(np.float32)
        spots = np.stack([spot] * n)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n, dtype=np.uint32),
                "x": np.full(n, 40.0),
                "y": np.full(n, 60.0),
                "net_gradient": np.full(n, 1000.0),
            }
        )
        theta = localize.fit_spots_spline(spots, calib)
        box = calib["n_data"][0]
        locs = localize.locs_from_fits_spline(ids, theta, box, False, calib)
        assert len(locs) == n
        for col in ("x", "y", "z", "photons", "bg", "lpx", "lpy", "lpz"):
            assert col in locs.columns
        for col in ("lpx", "lpy", "lpz"):
            assert np.all(np.isfinite(locs[col])) and (locs[col] > 0).all()
        np.testing.assert_allclose(locs["photons"], amp, rtol=1e-3)


class TestNoSelfDeprecation:
    """Picasso must not warn about its own use of the deprecated fitters.

    ``picasso.gausslq`` and ``picasso.gaussmle`` are deprecated as of 0.11
    and go in 1.0, but ``localize`` and ``spline`` still call them until the
    method codes are rerouted. They call the *private* implementations for
    exactly this reason: a deprecation notice a user cannot act on - because
    it is Picasso's own internals that triggered it - is noise, and it would
    train people to ignore the ones that do matter."""

    CAMERA_INFO = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}

    @pytest.mark.parametrize(
        "method",
        [
            "gausslq",
            "gausslq-spherical",
            "gausslq-rotated",
            "gaussmle",
            "gaussmle-spherical",
            "avg",
        ],
    )
    def test_fit2d_raises_no_deprecation_warning(
        self, picasso_movie, movie_info, real_identifications, method
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            localize.fit2D(
                picasso_movie,
                movie_info,
                self.CAMERA_INFO,
                real_identifications[:20],
                BOX,
                fitting_method=method,
                multiprocess=False,
            )
        offenders = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and ("gausslq" in str(w.message) or "gaussmle" in str(w.message))
        ]
        assert offenders == []

    def test_multiprocessed_gausslq_is_silent(
        self, picasso_movie, movie_info, real_identifications
    ):
        """The process-pool path goes through ``_fit_spots_parallel``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            localize.fit2D(
                picasso_movie,
                movie_info,
                self.CAMERA_INFO,
                real_identifications[:20],
                BOX,
                fitting_method="gausslq",
                multiprocess=True,
            )
        offenders = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "gausslq" in str(w.message)
        ]
        assert offenders == []

    def test_internal_callers_use_the_private_names(self):
        """Belt and braces: reading the source, so a future edit that
        reintroduces a public call is caught even if the warning filter
        configuration changes."""
        for module in (localize, spline):
            source = inspect.getsource(module)
            for public in (
                "gausslq.fit_spot(",
                "gausslq.fit_spots(",
                "gausslq.fit_spots_parallel(",
                "gaussmle.gaussmle(",
                "gaussmle.gaussmle_async(",
            ):
                assert public not in source, (module.__name__, public)


class TestConvergenceSchedulePlumbing:
    """The convergence criterion and the iteration cap must reach *every*
    fitting method, on either device, and be recorded as what the fit actually
    used. Both used to be grayed out for GPU fitting and unavailable for the
    least-squares Gaussians, so a user could not touch them at all for
    Picasso's default method."""

    CAMERA_INFO = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}

    @pytest.mark.parametrize(
        "method,default",
        [
            ("gausslq", (gausslq.TOLERANCE, gausslq.MAX_ITERATIONS)),
            ("gausslq-spherical", (gausslq.TOLERANCE, gausslq.MAX_ITERATIONS)),
            ("gausslq-rotated", (gausslq.TOLERANCE, gausslq.MAX_ITERATIONS)),
            ("gaussmle", localize.gauss_schedule(True, False)),
            ("gaussmle-spherical", localize.gauss_schedule(True, False)),
            (
                "gausslq-gpu",
                (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS),
            ),
            (
                "gaussmle-rotated-gpu",
                (gaussfit_cuda.TOLERANCE, gaussfit_cuda.MAX_ITERATIONS),
            ),
        ],
    )
    def test_metadata_records_the_schedule(
        self, picasso_movie, movie_info, real_identifications, method, default
    ):
        """Asked for None, the metadata reports that method's own default;
        asked for a value, it reports the value."""
        if method.endswith("-gpu") and not localize.CUDA_AVAILABLE:
            pytest.skip("no CUDA device")

        def run(eps, max_it):
            _, info = localize.fit2D(
                picasso_movie,
                movie_info,
                self.CAMERA_INFO,
                real_identifications[:20],
                BOX,
                fitting_method=method,
                eps=eps,
                max_it=max_it,
                multiprocess=False,
            )
            return info["Convergence criterion"], info["Max iterations"]

        assert run(None, None) == default
        assert run(1e-5, 7) == (1e-5, 7)

    def test_avg_records_no_schedule(
        self, picasso_movie, movie_info, real_identifications
    ):
        """The one method that does not iterate must not claim one."""
        _, info = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications[:20],
            BOX,
            fitting_method="avg",
            multiprocess=False,
        )
        assert "Convergence criterion" not in info
        assert "Max iterations" not in info

    @pytest.mark.parametrize("mle", [False, True])
    def test_gpu_gauss_iteration_cap_bites(self, mle):
        """The cap has to reach the kernel, not just the metadata."""
        if not localize.CUDA_AVAILABLE:
            pytest.skip("no CUDA device")
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:BOX, 0:BOX].astype(float)
        center = (BOX - 1) / 2.0
        spots = np.stack(
            [
                rng.poisson(
                    900.0
                    * np.exp(
                        -0.5
                        * (
                            ((xx - center - dx) / 1.3) ** 2
                            + ((yy - center - dy) / 1.7) ** 2
                        )
                    )
                    + 12.0
                ).astype(np.float32)
                for dx, dy in rng.uniform(-0.5, 0.5, (24, 2))
            ]
        )
        capped = localize.fit_spots_gauss_gpu(
            spots, mle=mle, return_stats=True, max_iterations=2
        )[2]
        free = localize.fit_spots_gauss_gpu(
            spots, mle=mle, return_stats=True, max_iterations=40
        )[2]
        assert capped.max() <= 2
        assert free.max() > 2

    def test_gpu_gauss_tolerance_bites(self):
        """A looser stop must take strictly fewer iterations."""
        if not localize.CUDA_AVAILABLE:
            pytest.skip("no CUDA device")
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:BOX, 0:BOX].astype(float)
        center = (BOX - 1) / 2.0
        spots = np.stack(
            [
                rng.poisson(
                    900.0
                    * np.exp(
                        -0.5
                        * (
                            ((xx - center) / 1.3) ** 2
                            + ((yy - center) / 1.7) ** 2
                        )
                    )
                    + 12.0
                ).astype(np.float32)
                for _ in range(24)
            ]
        )
        loose = localize.fit_spots_gauss_gpu(
            spots, return_stats=True, tolerance=1e-1, max_iterations=60
        )[2]
        tight = localize.fit_spots_gauss_gpu(
            spots, return_stats=True, tolerance=1e-8, max_iterations=60
        )[2]
        assert loose.mean() < tight.mean()

    def test_spline_schedule_reaches_the_gpu_backend(self, monkeypatch):
        """``_fit2d_spline_gpu`` used to drop the caller's schedule on the
        floor - the GPU boxes were grayed out, so there was never one to
        pass."""
        calib = _fake_spline_calibration(model="spline-3d")
        box = calib["box"]
        seen = {}

        def fake(*args, **kwargs):
            seen.update(kwargs)
            n = len(args[0])
            return (
                np.zeros((n, 5), np.float32),
                np.zeros(n, np.float32),
                np.zeros(n, np.float32),
                np.zeros(n, np.float32),
            )

        monkeypatch.setattr(localize, "fit_spots_splinefit", fake)
        monkeypatch.setattr(
            localize, "locs_from_fits_spline", lambda *a, **k: pd.DataFrame()
        )
        localize._fit2d_spline_gpu(
            spots=np.zeros((2, box, box), np.float32),
            identifications=pd.DataFrame(),
            box=box,
            em=False,
            calibration=calib,
            tolerance=1e-7,
            max_iterations=3,
        )
        assert seen["tolerance"] == 1e-7
        assert seen["max_iterations"] == 3


class TestGaussCodeGrammar:
    """``localize.parse_gauss_code`` is both the parser and the validator for
    the Gaussian fit codes, so a code cannot be offered somewhere and
    rejected here."""

    #: The eleven codes that predate 0.11, with the flags they have always
    #: meant. These are what saved metadata and existing CLI scripts contain,
    #: so their meaning is frozen.
    LEGACY = {
        "gausslq": (False, False, False, False),
        "gausslq-spherical": (False, True, False, False),
        "gausslq-rotated": (False, False, True, False),
        "gausslq-gpu": (False, False, False, True),
        "gausslq-spherical-gpu": (False, True, False, True),
        "gausslq-rotated-gpu": (False, False, True, True),
        "gaussmle": (True, False, False, False),
        "gaussmle-spherical": (True, True, False, False),
        "gaussmle-gpu": (True, False, False, True),
        "gaussmle-spherical-gpu": (True, True, False, True),
        "gaussmle-rotated-gpu": (True, False, True, True),
    }

    @pytest.mark.parametrize("code,expected", sorted(LEGACY.items()))
    def test_legacy_codes_keep_their_meaning(self, code, expected):
        flags = localize.parse_gauss_code(code)
        assert flags is not None, code
        assert (
            flags["mle"],
            flags["spherical"],
            flags["rotated"],
            flags["use_gpu"],
        ) == expected

    def test_every_legacy_code_is_still_offered(self):
        assert set(self.LEGACY) <= set(localize.FIT_METHODS)

    @pytest.mark.parametrize(
        "code",
        [
            "gausslq-bogus",
            "gaussmle-gpu-gpu",
            "gausslq-spherical-rotated",
            "spline",
            "avg",
            "",
        ],
    )
    def test_rejects_what_is_not_a_gaussian_code(self, code):
        assert localize.parse_gauss_code(code) is None

    def test_rotated_has_no_integrated_form(self):
        """Not an oversight: the pixel integral of a rotated elliptical
        Gaussian is not separable in the rotated frame."""
        assert localize.parse_gauss_code("gausslq-int-rotated") is None
        assert not any(
            "int" in c and "rotated" in c for c in localize.FIT_METHODS
        )

    def test_fit_methods_round_trip(self):
        """Every generated code parses, and nothing else claims to."""
        for code in localize.FIT_METHODS:
            if code.startswith(("gausslq", "gaussmle")):
                assert localize.parse_gauss_code(code) is not None, code
            else:
                assert localize.parse_gauss_code(code) is None, code

    def test_fit2d_rejects_an_unknown_code(self, tmp_path):
        """The grammar is the validator: anything it does not accept must be
        refused rather than silently fitted as something else."""
        raw = tmp_path / "movie.raw"
        np.zeros((1, 16, 16), np.uint16).tofile(raw)
        movie = np.memmap(raw, dtype=np.uint16, mode="r", shape=(1, 16, 16))
        identifications = pd.DataFrame(
            {"frame": [0], "x": [8.0], "y": [8.0], "net_gradient": [1.0]}
        )
        with pytest.raises(AssertionError, match="not one of"):
            localize.fit2D(
                movie,
                [{"Frames": 1}],
                {"Baseline": 0, "Sensitivity": 1, "Gain": 1, "Pixelsize": 130},
                identifications,
                7,
                fitting_method="gausslq-nonsense",
            )


@pytest.mark.gui
class TestGuiConvergenceDefaults:
    """The GUI's per-method default table has to agree with the values the
    backends use when asked for None; otherwise the boxes show one schedule
    and the fit runs another."""

    def test_every_iterating_method_has_defaults(self):
        """Every ``fit2D`` code except "avg" iterates and must be listed."""
        codes = set()
        for entry in localize_gui.FIT_MODELS.values():
            optimizers = entry["optimizers"]
            if optimizers is None:
                codes.add(entry["code"])
                continue
            for code in optimizers.values():
                codes.add(code)
                codes.add(localize_gui._effective_fit_code(code, True))
        assert codes - localize_gui._CONVERGENCE_CODES == {"avg"}

    def test_defaults_match_the_backends(self):
        """The boxes must show the schedule the fit will actually use, so the
        table is derived from ``localize.gauss_schedule`` rather than
        repeated - this pins that it still resolves to the right values."""
        table = localize_gui._CONVERGENCE_DEFAULTS
        assert table["gausslq"] == (
            gausslq.TOLERANCE,
            gausslq.MAX_ITERATIONS,
        )
        assert table["gausslq-gpu"] == (
            gaussfit_cuda.TOLERANCE,
            gaussfit_cuda.MAX_ITERATIONS,
        )
        # Not 0.001: rerouting the CPU MLE onto Levenberg-Marquardt changed
        # what the criterion *means* (relative in the chi-square, not a
        # position shift in pixels), so the default was re-derived rather
        # than carried across. See ``localize._GAUSS_SCHEDULES``.
        assert table["gaussmle"] == localize.gauss_schedule(True, False)
        assert table["gaussmle"] == (1e-5, 100)
        assert table["spline-gpu"] == table["spline"]
        assert table["spline"] == localize._spline_schedule(True, None, None)

    def test_gpu_capable_codes_are_real_fit2d_codes(self):
        """``_effective_fit_code`` appends "-gpu"; the result has to be a
        method ``fit2D`` accepts."""
        for code in localize_gui._GPU_CAPABLE_CODES:
            assert not code.endswith("-gpu")
            assert code + "-gpu" in localize_gui._CONVERGENCE_CODES

    def test_every_fitted_model_can_use_the_gpu(self):
        """Every PSF model in the dialog runs on both devices now, so ticking
        "Use GPU" must move *all* of them onto the GPU - a code left out of
        ``_GPU_CAPABLE_CODES`` would silently keep fitting on the CPU. Only
        "Average of ROI", which does not fit, is exempt."""
        for entry in localize_gui.FIT_MODELS.values():
            for code in (entry["optimizers"] or {}).values():
                assert code in localize_gui._GPU_CAPABLE_CODES
                assert (
                    localize_gui._effective_fit_code(code, True)
                    == code + "-gpu"
                )

    def test_z_fitting_follows_the_fit_gpu_checkbox(self):
        """The astigmatism box no longer carries its own GPU checkbox, so a
        user cannot put the two devices in disagreement."""
        source = inspect.getsource(localize_gui)
        assert "fit_z_gpu_checkbox" not in source
        assert (
            "self.parameters_dialog.gpu_checkbox.isChecked(),"
            in inspect.getsource(localize_gui.Window.fit_z)
        )

    def test_dialog_refills_the_boxes_per_method_and_device(self):
        """Drive the real dialog: the schedule shown has to follow the model,
        the optimizer *and* the GPU checkbox, and a value the user typed must
        survive an unrelated update."""

        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

        dialog = localize_gui.ParametersDialog(_StubWindow())
        try:
            assert dialog.current_fit_code() == "gausslq"
            assert (
                dialog.convergence_criterion.value(),
                dialog.max_it.value(),
            ) == localize.gauss_schedule(False, False)

            if localize.CUDA_AVAILABLE:
                dialog.gpu_checkbox.setChecked(True)
                assert dialog.current_fit_code() == "gausslq-gpu"
                assert (
                    dialog.convergence_criterion.value(),
                    dialog.max_it.value(),
                ) == localize.gauss_schedule(False, True)
                dialog.gpu_checkbox.setChecked(False)
                assert (
                    dialog.convergence_criterion.value(),
                    dialog.max_it.value(),
                ) == localize.gauss_schedule(False, False)

            dialog.fit_optimizer.setCurrentIndex(
                dialog.fit_optimizer.findText("MLE")
            )
            assert dialog.current_fit_code() == "gaussmle"
            assert (
                dialog.convergence_criterion.value(),
                dialog.max_it.value(),
            ) == localize.gauss_schedule(True, False)

            # An unrelated refresh must not overwrite a typed value.
            dialog.convergence_criterion.setValue(0.05)
            dialog.on_gpu_fitting_changed()
            assert dialog.convergence_criterion.value() == 0.05

            # The one non-iterating method hides the page entirely.
            dialog.fit_model.setCurrentIndex(
                dialog.fit_model.findText("Average of ROI")
            )
            assert dialog.current_fit_code() == "avg"
            assert dialog.fit_stack.currentIndex() == 0
        finally:
            dialog.deleteLater()

    def test_convergence_criterion_cannot_be_zero(self):
        """``fit2D`` asserts a positive tolerance, so the box must not offer
        0 - it used to, which made the fit raise on a valid-looking value."""

        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

        dialog = localize_gui.ParametersDialog(_StubWindow())
        try:
            assert dialog.convergence_criterion.minimum() > 0
            dialog.convergence_criterion.setValue(0.0)
            assert dialog.convergence_criterion.value() > 0
        finally:
            dialog.deleteLater()


@pytest.mark.skipif(not localize.CUDA_AVAILABLE, reason="no CUDA device")
class TestFit2DGpu:
    """End-to-end ``localize.fit2D`` through every GPU fitting method, driven by
    the bundled movie and its real identifications. Verifies the high-level
    dispatch, spot extraction, GPU fit and localization assembly hang together
    and produce a saveable localizations frame."""

    CAMERA_INFO = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}

    @pytest.mark.parametrize(
        "method,has_angle",
        [
            ("gausslq-gpu", False),
            ("gaussmle-gpu", False),
            ("gausslq-rotated-gpu", True),
            ("gaussmle-rotated-gpu", True),
        ],
    )
    def test_gauss_gpu_methods(
        self,
        picasso_movie,
        movie_info,
        real_identifications,
        method,
        has_angle,
    ):
        locs, info = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications,
            BOX,
            fitting_method=method,
        )
        assert len(locs) == len(real_identifications)
        for col in (
            "frame",
            "x",
            "y",
            "photons",
            "sx",
            "sy",
            "bg",
            "lpx",
            "lpy",
        ):
            assert col in locs.columns
        assert ("angle" in locs.columns) == has_angle
        assert info["Fit method"] == method
        # MLE methods attach per-parameter uncertainties + fit diagnostics
        if method.startswith("gaussmle"):
            for col in (
                "photons_unc",
                "bg_unc",
                "log_likelihood",
                "iterations",
            ):
                assert col in locs.columns

    @pytest.mark.parametrize("method", ["spline-gpu", "spline-mle-gpu"])
    def test_spline_gpu_methods(
        self, picasso_movie, movie_info, real_identifications, method
    ):
        calib, _, _, _ = _synthetic_spline_3d_calibration(box=BOX)
        locs, info = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications,
            BOX,
            fitting_method=method,
            spline_calibration=calib,
        )
        assert len(locs) == len(real_identifications)
        for col in (
            "frame",
            "x",
            "y",
            "z",
            "photons",
            "bg",
            "lpx",
            "lpy",
            "lpz",
        ):
            assert col in locs.columns
        assert info["Spline calibration model"] == "spline-3d"

    def test_gpu_matches_direct_fit_path(
        self, picasso_movie, movie_info, real_identifications
    ):
        """fit2D('gausslq-gpu') equals calling the spot extraction + GPU fit +
        localization assembly directly - i.e. the wrapper adds no drift."""
        locs, _ = localize.fit2D(
            picasso_movie,
            movie_info,
            self.CAMERA_INFO,
            real_identifications,
            BOX,
            fitting_method="gausslq-gpu",
        )
        spots = localize.get_spots(
            picasso_movie, real_identifications, BOX, self.CAMERA_INFO
        )
        theta = localize.fit_spots_gauss_gpu(spots, mle=False)
        direct = localize.locs_from_fits_gauss(
            real_identifications, theta, BOX, em=False
        )
        np.testing.assert_allclose(
            locs["x"].to_numpy(), direct["x"].to_numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(
            locs["photons"].to_numpy(), direct["photons"].to_numpy(), rtol=1e-5
        )


class TestCRLBModuleBoundary:
    @pytest.mark.parametrize(
        "name",
        [
            "_LINK_XYZ_MAX_CHANNELS",
            "_LINK_XYZ_MODEL",
            "_crlb_variance_channel_major",
            "_gauss_crlb",
            "_spline_channel_jacobians",
            "_spline_channel_major",
            "_spline_coeff_reshaped",
            "_spline_crlb",
            "_spline_crlb_residuals",
            "_spline_link_xyz_crlb",
            "_spline_n_channels",
        ],
    )
    def test_precision_owns_the_name(self, name):
        assert hasattr(precision, name)
        assert not hasattr(localize, name)

    def test_the_two_cuda_flags_are_independent(self):
        """``localize`` gates the fitting backends, ``precision`` the CRLB
        kernels. They agree on any real machine, but they are separate module
        globals - patching one must not move the other, which is what the
        device-pinning helpers in this file rely on."""
        assert localize.CUDA_AVAILABLE == precision.CUDA_AVAILABLE
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                precision, "CUDA_AVAILABLE", not precision.CUDA_AVAILABLE
            )
            assert localize.CUDA_AVAILABLE != precision.CUDA_AVAILABLE


class TestSplineCRLBReal:
    """The CRLB path on a real (non-separable) calibration (CPU, no GPU
    fit)."""

    def test_crlb_finite_across_z(self):
        calib, _, amplitude, offset = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        z_focus = calib["z_center"]
        thetas = np.array(
            [
                [amplitude, 0.1, -0.1, -(z_focus - dz), offset]
                for dz in (-6.0, 0.0, 6.0)
            ],
            np.float64,
        )
        crlb = precision._spline_crlb(thetas, calib, box)
        assert np.all(np.isfinite(crlb)) and np.all(crlb > 0)

    @pytest.mark.skipif(
        not precision.CUDA_AVAILABLE,
        reason="requires a CUDA-capable GPU (numba-cuda)",
    )
    def test_gpu_matches_cpu_on_real_coefficients(self):
        """Parity on a genuinely non-separable coefficient table - the
        analytic test spline cannot exercise that layout."""
        calib, _, amplitude, offset = _synthetic_spline_3d_calibration()
        box, _, nz = calib["n_data"]
        rng = np.random.default_rng(0)
        n = 500
        thetas = np.zeros((n, 5))
        thetas[:, 0] = amplitude * rng.uniform(0.5, 2.0, n)
        thetas[:, 1] = rng.uniform(-0.7, 0.7, n)
        thetas[:, 2] = rng.uniform(-0.7, 0.7, n)
        thetas[:, 3] = -(calib["z_center"] + rng.uniform(-6.0, 6.0, n))
        thetas[:, 4] = offset * rng.uniform(0.5, 2.0, n)
        for mle in (True, False):
            np.testing.assert_allclose(
                _crlb(thetas, calib, box, mle=mle, gpu=True),
                _crlb(thetas, calib, box, mle=mle, gpu=False),
                rtol=1e-6,
            )


class TestSplineCRLB:
    """Cramer-Rao lower bounds for spline-fitted localizations. Uses a known
    separable Gaussian spline built with scipy (see _gauss_spline_calibration),
    so the exact CRLB reference is available in closed form without a GPU.
    The numba kernel is validated against the closed-form reference; the
    layout check is in TestSplineCoefficients."""

    def test_evaluator_matches_scipy_3d(self):
        # The evaluator the CRLB kernels are built on must reproduce the scipy
        # spline value and its x/y/z derivatives on the very fixture the CRLB
        # assertions below take as ground truth.
        calib, splines = _gauss_spline_calibration(model="spline-3d")
        nz = calib["n_data"][2]
        coeff = precision._spline_coeff_reshaped(calib)
        rng = np.random.default_rng(0)
        # The evaluator is scalar (one call per pixel), so this samples fewer
        # positions than a vectorized reference could afford.
        m = 12
        xs = rng.uniform(-0.7, 0.7, m)
        ys = rng.uniform(-0.7, 0.7, m)
        ze = rng.uniform(5, nz - 6, m)
        ref = _ref_model_grad(splines, BOX, xs, ys, ze)
        # ref is indexed [loc, x-pixel, y-pixel]
        for k in range(m):
            for i in range(BOX):
                for j in range(BOX):
                    got = splinefit._eval_spline_3d(
                        coeff, 0, i - xs[k], j - ys[k], ze[k]
                    )
                    want = tuple(float(r[k, i, j]) for r in ref)
                    np.testing.assert_allclose(
                        got,
                        want,
                        atol=1e-3,
                        rtol=0,
                        err_msg=f"loc {k}, pixel (x={i}, y={j})",
                    )

    def test_crlb_matches_reference_3d(self):
        # sx < sy (astigmatic) so lpx < lpy - also guards the x/y association.
        calib, splines = _gauss_spline_calibration(
            model="spline-3d", sx=1.0, sy=1.4
        )
        # native_z = -z_shift = 6, off the gz focus (=10) so the separable test
        # PSF carries real axial information (dPhi/dz != 0) and lpz is finite.
        amp, off, z_shift = 4000.0, 20.0, -6.0
        theta = np.array([[amp, 0.2, -0.15, z_shift, off]])
        crlb = precision._spline_crlb(theta, calib, BOX)[0]
        ref = _ref_crlb(splines, BOX, amp, 0.2, -0.15, -z_shift, off)
        np.testing.assert_allclose(crlb, ref, rtol=1e-2)
        assert crlb[0] < crlb[1]  # lpx < lpy
        assert np.isfinite(crlb[2]) and crlb[2] > 0  # finite lpz

    def test_crlb_matches_reference_2d(self):
        calib, splines = _gauss_spline_calibration(
            model="spline-2d", sx=1.0, sy=1.3
        )
        amp, off = 3000.0, 15.0
        theta = np.array([[amp, 0.1, -0.1, off]])
        crlb = precision._spline_crlb(theta, calib, BOX)[0]
        ref = _ref_crlb(splines, BOX, amp, 0.1, -0.1, None, off)
        np.testing.assert_allclose(crlb, ref, rtol=1e-2)
        assert crlb[0] < crlb[1]

    def test_lsq_sandwich_matches_reference_3d(self):
        # mle=False must return the unweighted-least-squares sandwich covariance.
        calib, splines = _gauss_spline_calibration(
            model="spline-3d", sx=1.0, sy=1.4
        )
        amp, off, z_shift = 4000.0, 20.0, -6.0
        theta = np.array([[amp, 0.2, -0.15, z_shift, off]])
        var = precision._spline_crlb(theta, calib, BOX, mle=False)[0]
        ref = _ref_crlb_lsq(splines, BOX, amp, 0.2, -0.15, -z_shift, off)
        np.testing.assert_allclose(var, ref, rtol=1e-2)
        assert var[0] < var[1]  # lpx < lpy
        assert np.isfinite(var[2]) and var[2] > 0  # finite lpz

    def test_lsq_sandwich_matches_reference_2d(self):
        calib, splines = _gauss_spline_calibration(
            model="spline-2d", sx=1.0, sy=1.3
        )
        amp, off = 3000.0, 15.0
        theta = np.array([[amp, 0.1, -0.1, off]])
        var = precision._spline_crlb(theta, calib, BOX, mle=False)[0]
        ref = _ref_crlb_lsq(splines, BOX, amp, 0.1, -0.1, None, off)
        np.testing.assert_allclose(var, ref, rtol=1e-2)
        assert var[0] < var[1]

    def test_lsq_variance_geq_crlb(self):
        # Least squares is not efficient for Poisson data: with background the
        # sandwich covariance is strictly above the Cramer-Rao (MLE) bound.
        calib, _ = _gauss_spline_calibration(model="spline-2d", sx=1.0, sy=1.3)
        theta = np.array([[3000.0, 0.1, -0.1, 40.0]])
        crlb = precision._spline_crlb(theta, calib, BOX, mle=True)[0]
        lsq = precision._spline_crlb(theta, calib, BOX, mle=False)[0]
        assert np.all(np.isfinite(lsq))
        # allow tiny numerical slack, then require the x/y positions to be worse
        assert np.all(lsq >= crlb * (1 - 1e-6))
        assert lsq[0] > crlb[0] and lsq[1] > crlb[1]

    def test_lsq_nan_theta_row_isolated(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, 0.0, 15.0], [np.nan, 0.0, 0.0, 10.0]])
        var = precision._spline_crlb(theta, calib, BOX, mle=False)
        assert np.all(np.isfinite(var[0]))
        assert np.all(np.isnan(var[1]))

    def test_multichannel_sums_fisher(self):
        # Two identical channels double the Fisher information -> half variance.
        calib_1, _ = _gauss_spline_calibration(model="spline-3d")
        calib_2, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=2
        )
        theta = np.array([[5000.0, 0.1, -0.1, -8.0, 20.0]])
        crlb_1 = precision._spline_crlb(theta, calib_1, BOX)[0]
        crlb_2 = precision._spline_crlb(theta, calib_2, BOX)[0]
        np.testing.assert_allclose(crlb_2, crlb_1 / 2.0, rtol=1e-3)

    def test_nan_theta_row_isolated(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, 0.0, 15.0], [np.nan, 0.0, 0.0, 10.0]])
        crlb = precision._spline_crlb(theta, calib, BOX)
        assert np.all(np.isfinite(crlb[0]))
        assert np.all(np.isnan(crlb[1]))

    def test_low_signal_stays_finite(self):
        # offset = 0 drives some model pixels to ~0; the MU_FLOOR guard keeps
        # the Fisher weight (1 / mu) finite.
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[500.0, 0.0, 0.0, 0.0]])
        assert np.all(
            np.isfinite(precision._spline_crlb(theta, calib, BOX)[0])
        )

    def test_progress_callback_and_console(self):
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        n = 250
        rng = np.random.default_rng(0)
        theta = np.zeros((n, 4))
        theta[:, 0] = 3000.0
        theta[:, 1:3] = rng.uniform(-0.5, 0.5, (n, 2))
        theta[:, 3] = 15.0
        seen = []
        precision._spline_crlb(
            theta, calib, BOX, progress_callback=seen.append
        )
        assert seen and seen[-1] == n and seen == sorted(seen)
        # the tqdm ("console") path must not raise
        precision._spline_crlb(theta, calib, BOX, progress_callback="console")


def _crlb(theta, calibration, box, *, gpu, **kwargs):
    """``precision._spline_crlb`` pinned to one device.

    Production dispatch is automatic (GPU when present, CPU otherwise), so
    hiding the GPU is the only way to exercise the CPU path deliberately - and
    the only way to compare the two against each other. The flag that decides
    lives with the kernels in ``picasso.fitting.precision``, not in
    ``localize``, whose own ``CUDA_AVAILABLE`` gates the *fitting* backends."""
    with pytest.MonkeyPatch.context() as m:
        m.setattr(precision, "CUDA_AVAILABLE", gpu)
        return precision._spline_crlb(theta, calibration, box, **kwargs)


def _shared_theta(n, rng, n_params=5, nz=21):
    """Well-spread fitted parameters for the shared-amplitude spline models,
    ``[amplitude, x, y, (z,) offset]``."""
    theta = np.zeros((n, n_params))
    theta[:, 0] = rng.uniform(500.0, 8000.0, n)
    theta[:, 1] = rng.uniform(-0.7, 0.7, n)
    theta[:, 2] = rng.uniform(-0.7, 0.7, n)
    if n_params == 5:
        # native z = -z_shift, kept clear of the ends of the calibration stack
        theta[:, 3] = -rng.uniform(4.0, nz - 5.0, n)
    theta[:, -1] = rng.uniform(1.0, 50.0, n)
    return theta


def _with_channel_geometry(calib, n_locs, rng):
    """Give ``calib`` a non-identity per-channel affine and return matching
    sub-pixel ROI residuals.

    Without these, every channel is evaluated at the same place and the
    geometry terms in the CRLB kernels are dead code - so a GPU/CPU comparison
    on a plain calibration cannot tell whether either side applies them. The
    reference channel keeps the identity and a zero residual, as the real
    pipeline does (see :func:`localize.channel_roi_residuals`)."""
    n_channels = precision._spline_n_channels(calib)
    calib = dict(calib)
    calib["channel_transforms"] = [
        (
            IDENTITY
            if c == 0
            else affine(
                [
                    [1.0 + 0.03 * c, 0.02 * c, 0.0],
                    [-0.015 * c, 1.0 - 0.01 * c, 0.0],
                ]
            ).to_dict()
        )
        for c in range(n_channels)
    ]
    residuals = rng.uniform(-0.5, 0.5, (n_locs, n_channels, 2))
    residuals[:, 0, :] = 0.0
    return calib, residuals


def _link_xyz_calib_and_theta(n_channels, n_locs, rng, nz=21):
    """Link-XYZ calibration on the separable Gaussian spline, plus a batch of
    fitted parameters ``[x, y, z, N_0.., bg_0..]`` with unequal per-channel
    photons so a channel mix-up cannot pass unnoticed."""
    calib, _ = _gauss_spline_calibration(
        model="spline-3d-multichannel", n_channels=n_channels, nz=nz
    )
    calib = localize._as_link_xyz_calibration(calib)
    theta = np.zeros((n_locs, 3 + 2 * n_channels))
    theta[:, 0] = rng.uniform(-0.7, 0.7, n_locs)
    theta[:, 1] = rng.uniform(-0.7, 0.7, n_locs)
    theta[:, 2] = -rng.uniform(4.0, nz - 5.0, n_locs)
    theta[:, 3 : 3 + n_channels] = rng.uniform(
        200.0, 3000.0, (n_locs, n_channels)
    )
    theta[:, 3 + n_channels :] = rng.uniform(2.0, 40.0, (n_locs, n_channels))
    return calib, theta


class TestSplineLinkXyzCRLB:
    """Variances of the photon-decoupled (link-XYZ) spline model, the
    ``(3 + 2*n_channels)``-parameter block-sparse kernel. CPU only; the GPU
    parity tests live in ``TestSplineCRLBGPU``."""

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("mle", [True, False])
    def test_shape_and_positivity(self, n_channels, mle):
        rng = np.random.default_rng(0)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 5, rng)
        var = _crlb(theta, calib, BOX, mle=mle, gpu=False)
        assert var.shape == (5, 3 + 2 * n_channels)
        assert np.all(np.isfinite(var)) and np.all(var > 0)

    def test_channel_order_is_carried_through(self):
        """Reordering the channels permutes the per-channel variances and
        leaves the shared x/y/z ones alone. Guards the block-sparse indexing
        (``3 + ch`` for photons, ``3 + n_channels + ch`` for background), which
        is where a channel mix-up would hide."""
        n_channels = 4
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        # Make the channels distinguishable: without this a permutation is a
        # no-op and the test proves nothing.
        coeff = np.asarray(calib["coefficients"], dtype=np.float32).copy()
        for c in range(n_channels):
            coeff[..., c] *= 0.5 + 0.4 * c
        calib["coefficients"] = coeff
        calib = localize._as_link_xyz_calibration(calib)

        rng = np.random.default_rng(1)
        _, theta = _link_xyz_calib_and_theta(n_channels, 6, rng)
        var = _crlb(theta, calib, BOX, gpu=False)

        perm = np.array([2, 0, 3, 1])
        permuted = dict(calib)
        permuted["coefficients"] = coeff[..., perm]
        theta_p = theta.copy()
        theta_p[:, 3 : 3 + n_channels] = theta[:, 3 + perm]
        theta_p[:, 3 + n_channels :] = theta[:, 3 + n_channels + perm]
        var_p = _crlb(theta_p, permuted, BOX, gpu=False)

        np.testing.assert_allclose(var_p[:, :3], var[:, :3], rtol=1e-9)
        np.testing.assert_allclose(
            var_p[:, 3 : 3 + n_channels],
            var[:, 3 + perm],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            var_p[:, 3 + n_channels :],
            var[:, 3 + n_channels + perm],
            rtol=1e-9,
        )

    @pytest.mark.parametrize("n_channels", [2, 4])
    def test_lsq_variance_geq_crlb(self, n_channels):
        # Least squares is not efficient for Poisson data.
        rng = np.random.default_rng(2)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 4, rng)
        crlb = _crlb(theta, calib, BOX, mle=True, gpu=False)
        lsq = _crlb(theta, calib, BOX, mle=False, gpu=False)
        assert np.all(lsq >= crlb * (1 - 1e-6))

    def test_nan_theta_row_isolated(self):
        rng = np.random.default_rng(3)
        calib, theta = _link_xyz_calib_and_theta(2, 2, rng)
        theta[1, 3] = np.nan
        var = _crlb(theta, calib, BOX, gpu=False)
        assert np.all(np.isfinite(var[0]))
        assert np.all(np.isnan(var[1]))

    def test_progress_callback_and_console(self):
        rng = np.random.default_rng(4)
        calib, theta = _link_xyz_calib_and_theta(2, 120, rng)
        seen = []
        _crlb(theta, calib, BOX, progress_callback=seen.append, gpu=False)
        assert seen and seen[-1] == len(theta) and seen == sorted(seen)
        _crlb(theta, calib, BOX, progress_callback="console", gpu=False)


class TestSplineCRLBEMCCD:
    """EMCCD excess noise in the spline uncertainties. Stochastic electron
    multiplication doubles every pixel's variance on top of the Poisson term,
    so every reported variance has to double with it - for both estimators,
    since that is a property of the detector and not of the fit. Matches
    ``gausslq.localization_precision`` and ``_gauss_crlb``."""

    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize(
        "model", ["spline-2d", "spline-3d", "spline-3d-multichannel"]
    )
    def test_em_doubles_the_variance(self, model, mle):
        calib, _ = _gauss_spline_calibration(model=model, n_channels=2)
        rng = np.random.default_rng(0)
        theta = _shared_theta(
            32, rng, n_params=4 if model == "spline-2d" else 5
        )
        plain = precision._spline_crlb(theta, calib, BOX, mle=mle)
        em = precision._spline_crlb(theta, calib, BOX, mle=mle, em=True)
        np.testing.assert_allclose(em, 2.0 * plain, rtol=1e-12)

    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 4])
    def test_em_doubles_the_variance_link_xyz(self, n_channels, mle):
        rng = np.random.default_rng(1)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 16, rng)
        plain = precision._spline_crlb(theta, calib, BOX, mle=mle)
        em = precision._spline_crlb(theta, calib, BOX, mle=mle, em=True)
        np.testing.assert_allclose(em, 2.0 * plain, rtol=1e-12)

    @pytest.mark.parametrize("mle", [True, False])
    def test_em_reaches_the_reported_precisions(self, mle):
        """The end-to-end check: ``em`` must survive the trip through
        ``locs_from_fits_spline`` into lpx/lpy/lpz/photons_unc/bg_unc. It used
        to be accepted there and silently dropped, leaving EMCCD precisions a
        factor sqrt(2) too optimistic."""
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        rng = np.random.default_rng(2)
        theta = _shared_theta(24, rng)
        ids = pd.DataFrame(
            {
                "frame": np.arange(len(theta), dtype=np.uint32),
                "x": np.full(len(theta), 20.0),
                "y": np.full(len(theta), 30.0),
                "net_gradient": np.full(len(theta), 100.0),
            }
        )
        cols = ["lpx", "lpy", "lpz", "photons_unc", "bg_unc"]
        plain = localize.locs_from_fits_spline(
            ids, theta, BOX, False, calib, mle=mle
        )
        em = localize.locs_from_fits_spline(
            ids, theta, BOX, True, calib, mle=mle
        )
        for col in cols:
            np.testing.assert_allclose(
                em[col].to_numpy(),
                np.sqrt(2.0) * plain[col].to_numpy(),
                rtol=1e-5,
            )


requires_crlb_gpu = pytest.mark.skipif(
    not precision.CUDA_AVAILABLE,
    reason="requires a CUDA-capable GPU (numba-cuda)",
)


class TestSplineCRLBGPU:
    """The numba.cuda spline CRLB path.

    The GPU reproduces the CPU kernels plus ``numpy.linalg.pinv`` rather than
    approximating them, so these are parity tests, run through :func:`_crlb` to
    pin each side to one device. The separable Gaussian test PSF makes that a
    demanding comparison: ``dPhi/dz`` is proportional to ``Phi``, so the z and
    amplitude columns are collinear and the information matrix is exactly rank
    deficient - both paths have to truncate the same mode for the answers to
    agree at all.
    """

    # Headroom for the pseudo-inverse of that rank-deficient matrix; on a real
    # (well-conditioned) calibration the two agree to ~1e-8 relative.
    RTOL = 1e-5

    def test_no_cuda_uses_the_cpu_silently(self, monkeypatch):
        """Without a CUDA device the CPU kernels are used, with no warning and
        no way (or need) for the caller to ask for anything else."""
        monkeypatch.setattr(precision, "CUDA_AVAILABLE", False)
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, -0.1, 15.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = precision._spline_crlb(theta, calib, BOX)
        np.testing.assert_array_equal(got, _crlb(theta, calib, BOX, gpu=False))

    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_unsolvable_rows_are_recomputed_on_the_cpu(
        self, monkeypatch, model
    ):
        """Rows the device reports as unsolvable must come back with the CPU's
        pinv numbers, not whatever the kernel left behind. Driven through a stub
        device driver so it runs without a GPU - this fallback is the only
        structural difference between the two paths, so it needs a test.

        The stub takes the real driver's signature rather than swallowing the
        extras with ``**kwargs``: a mismatch would raise, and the host catches
        every exception from the device and quietly recomputes *everything* on
        the CPU - which is also what this test asserts, so it would keep passing
        while exercising the error path instead of the ``failed`` mask. The
        ``simplefilter("error")`` below is what pins the difference: only the
        exception path warns."""
        monkeypatch.setattr(precision, "CUDA_AVAILABLE", True)
        calib, _ = _gauss_spline_calibration(model=model)
        rng = np.random.default_rng(5)
        n_params = 4 if model == "spline-2d" else 5
        theta = _shared_theta(9, rng, n_params=n_params)
        expected = _crlb(theta, calib, BOX, gpu=False)
        called = []

        def stub(
            coeff,
            jac,
            res,
            box,
            amp,
            xs,
            ys,
            ze,
            off,
            finite,
            mu_floor,
            mle,
            progress_callback=None,
            variance=None,
        ):
            called.append(True)
            out = precision._spline_crlb_cpu(
                np.asarray(coeff, dtype=np.float64),
                jac,
                res,
                box,
                amp,
                xs,
                ys,
                ze,
                off,
                finite,
                mle,
                variance=variance,
            )
            failed = np.zeros(len(amp), dtype=bool)
            failed[::2] = True
            out[failed] = 12345.0  # device garbage the host must discard
            return out, failed

        monkeypatch.setattr(precision, "_spline_crlb_cuda", stub)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _crlb(theta, calib, BOX, gpu=True)
        assert called, "the device driver was never reached"
        np.testing.assert_allclose(got, expected)

    def test_device_error_falls_back_and_warns_once(self, monkeypatch):
        """A device that is present but fails still returns the right numbers,
        but must not do it silently - a permanently broken GPU path would
        otherwise never be noticed. (Having no device at all is not an error
        and stays quiet; see test_no_cuda_uses_the_cpu_silently.)"""
        monkeypatch.setattr(precision, "CUDA_AVAILABLE", True)
        monkeypatch.setattr(precision, "_crlb_gpu_fallback_warned", False)

        def boom(*args, **kwargs):
            raise RuntimeError("device on fire")

        monkeypatch.setattr(precision, "_spline_crlb_cuda", boom)
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[3000.0, 0.1, -0.1, 15.0]])
        expected = _crlb(theta, calib, BOX, gpu=False)

        with pytest.warns(RuntimeWarning, match="falling back to the CPU"):
            got = precision._spline_crlb(theta, calib, BOX)
        np.testing.assert_allclose(got, expected)

        # warned once per process, not once per call
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            np.testing.assert_allclose(
                precision._spline_crlb(theta, calib, BOX), expected
            )

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize(
        "model, n_channels",
        [
            ("spline-2d", 1),
            ("spline-3d", 1),
            ("spline-3d-multichannel", 2),
            ("spline-3d-multichannel", 6),
        ],
    )
    def test_matches_cpu(self, model, n_channels, mle):
        calib, _ = _gauss_spline_calibration(
            model=model, n_channels=n_channels
        )
        rng = np.random.default_rng(0)
        theta = _shared_theta(
            300, rng, n_params=4 if model == "spline-2d" else 5
        )
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, mle=mle, gpu=True),
            _crlb(theta, calib, BOX, mle=mle, gpu=False),
            rtol=self.RTOL,
        )

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_link_xyz_matches_cpu(self, n_channels, mle):
        rng = np.random.default_rng(1)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 200, rng)
        gpu = _crlb(theta, calib, BOX, mle=mle, gpu=True)
        cpu = _crlb(theta, calib, BOX, mle=mle, gpu=False)
        assert gpu.shape == (200, 3 + 2 * n_channels)
        np.testing.assert_allclose(gpu, cpu, rtol=self.RTOL)

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 3, 6])
    def test_channel_geometry_matches_cpu(self, n_channels, mle):
        """With a non-identity per-channel affine and non-zero ROI residuals -
        the case the plain parity tests cannot see, since at identity and zero
        the geometry terms drop out of both kernels alike."""
        rng = np.random.default_rng(11)
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        theta = _shared_theta(200, rng)
        calib, res = _with_channel_geometry(calib, len(theta), rng)
        gpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=True)
        cpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=False)
        np.testing.assert_allclose(gpu, cpu, rtol=self.RTOL)
        # and the geometry actually moved the answer, so this is a real test
        plain = _crlb(theta, calib, BOX, mle=mle, gpu=True)
        assert not np.allclose(gpu[:, :2], plain[:, :2], rtol=self.RTOL)

    @requires_crlb_gpu
    @pytest.mark.parametrize("mle", [True, False])
    @pytest.mark.parametrize("n_channels", [2, 3, 6])
    def test_link_xyz_channel_geometry_matches_cpu(self, n_channels, mle):
        rng = np.random.default_rng(12)
        calib, theta = _link_xyz_calib_and_theta(n_channels, 200, rng)
        calib, res = _with_channel_geometry(calib, len(theta), rng)
        gpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=True)
        cpu = _crlb(theta, calib, BOX, mle=mle, residuals=res, gpu=False)
        np.testing.assert_allclose(gpu, cpu, rtol=self.RTOL)
        plain = _crlb(theta, calib, BOX, mle=mle, gpu=True)
        assert not np.allclose(gpu[:, :2], plain[:, :2], rtol=self.RTOL)

    @requires_crlb_gpu
    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_em_doubling_matches_cpu(self, model):
        """The EMCCD factor is applied by the caller, after the device hands
        back its variances - so it must land identically on both paths."""
        calib, _ = _gauss_spline_calibration(model=model)
        rng = np.random.default_rng(13)
        theta = _shared_theta(
            64, rng, n_params=4 if model == "spline-2d" else 5
        )
        gpu = _crlb(theta, calib, BOX, em=True, gpu=True)
        np.testing.assert_allclose(
            gpu, _crlb(theta, calib, BOX, em=True, gpu=False), rtol=self.RTOL
        )
        np.testing.assert_allclose(
            gpu, 2.0 * _crlb(theta, calib, BOX, gpu=True), rtol=1e-12
        )

    @requires_crlb_gpu
    def test_matches_closed_form_3d(self):
        """Against the analytic reference, not just against the CPU - a port
        that is systematically wrong could still match a co-wrong CPU path."""
        calib, splines = _gauss_spline_calibration(
            model="spline-3d", sx=1.0, sy=1.4
        )
        amp, off, z_shift = 4000.0, 20.0, -6.0
        theta = np.array([[amp, 0.2, -0.15, z_shift, off]])
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, gpu=True)[0],
            _ref_crlb(splines, BOX, amp, 0.2, -0.15, -z_shift, off),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, mle=False, gpu=True)[0],
            _ref_crlb_lsq(splines, BOX, amp, 0.2, -0.15, -z_shift, off),
            rtol=1e-2,
        )

    @requires_crlb_gpu
    def test_matches_closed_form_2d(self):
        calib, splines = _gauss_spline_calibration(
            model="spline-2d", sx=1.0, sy=1.3
        )
        amp, off = 3000.0, 15.0
        theta = np.array([[amp, 0.1, -0.1, off]])
        np.testing.assert_allclose(
            _crlb(theta, calib, BOX, gpu=True)[0],
            _ref_crlb(splines, BOX, amp, 0.1, -0.1, None, off),
            rtol=1e-2,
        )

    @requires_crlb_gpu
    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_nan_theta_row_isolated(self, model):
        calib, _ = _gauss_spline_calibration(model=model)
        rng = np.random.default_rng(6)
        n_params = 4 if model == "spline-2d" else 5
        theta = _shared_theta(3, rng, n_params=n_params)
        theta[1, 0] = np.nan
        crlb = _crlb(theta, calib, BOX, gpu=True)
        assert np.all(np.isfinite(crlb[0])) and np.all(np.isfinite(crlb[2]))
        assert np.all(np.isnan(crlb[1]))

    @requires_crlb_gpu
    @pytest.mark.parametrize("model", ["spline-2d", "spline-3d"])
    def test_empty_theta(self, model):
        calib, _ = _gauss_spline_calibration(model=model)
        n_params = 4 if model == "spline-2d" else 5
        crlb = _crlb(np.zeros((0, n_params)), calib, BOX, gpu=True)
        assert crlb.shape == (0, n_params)

    @requires_crlb_gpu
    def test_cropped_box_matches_cpu(self):
        """The GPU must see the same centered crop the fit used."""
        calib, _ = _gauss_spline_calibration(model="spline-3d", box=9)
        calib = localize.crop_spline_calibration(calib, 7)
        rng = np.random.default_rng(7)
        theta = _shared_theta(64, rng)
        np.testing.assert_allclose(
            _crlb(theta, calib, 7, gpu=True),
            _crlb(theta, calib, 7, gpu=False),
            rtol=self.RTOL,
        )

    @requires_crlb_gpu
    def test_low_signal_stays_finite(self):
        # offset = 0 drives some model pixels to ~0; the MU_FLOOR guard keeps
        # the Fisher weight (1 / mu) finite on the device too.
        calib, _ = _gauss_spline_calibration(model="spline-2d")
        theta = np.array([[500.0, 0.0, 0.0, 0.0]])
        crlb = _crlb(theta, calib, BOX, gpu=True)[0]
        assert np.all(np.isfinite(crlb))
        np.testing.assert_allclose(
            crlb,
            _crlb(theta, calib, BOX, gpu=False)[0],
            rtol=self.RTOL,
        )

    @requires_crlb_gpu
    def test_progress_callback_and_console(self):
        calib, _ = _gauss_spline_calibration(model="spline-3d")
        rng = np.random.default_rng(8)
        theta = _shared_theta(250, rng)
        seen = []
        _crlb(theta, calib, BOX, progress_callback=seen.append, gpu=True)
        assert seen and seen[-1] == len(theta) and seen == sorted(seen)
        _crlb(theta, calib, BOX, progress_callback="console", gpu=True)

    @pytest.mark.slow
    def test_simulator_matches_cpu(self):
        """Under Numba's CUDA simulator (no physical GPU needed) the kernels and
        the device pseudo-inverse reproduce the CPU path. Runs in a subprocess
        because the simulator must be enabled before numba is imported. Kept
        tiny - the simulator interprets every thread in Python."""
        import subprocess

        script = (
            "import os\n"
            "os.environ['NUMBA_ENABLE_CUDASIM'] = '1'\n"
            "import numpy as np\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import sys; sys.path.insert(0, '.')\n"
            # a device failure would fall back to the CPU and turn every
            # comparison below into CPU-vs-CPU, i.e. vacuously true
            "import warnings; warnings.simplefilter('error', RuntimeWarning)\n"
            "from picasso.fitting import precision\n"
            "from tests.test_localize import ("
            "_gauss_spline_calibration, _shared_theta, _crlb, BOX,"
            " _link_xyz_calib_and_theta, _with_channel_geometry)\n"
            "assert precision.CUDA_AVAILABLE, 'simulator off'\n"
            "rng = np.random.default_rng(0)\n"
            "for model in ('spline-2d', 'spline-3d'):\n"
            "    calib, _ = _gauss_spline_calibration(model=model)\n"
            "    p = 4 if model == 'spline-2d' else 5\n"
            "    theta = _shared_theta(3, rng, n_params=p)\n"
            "    for mle in (True, False):\n"
            "        g = _crlb("
            "theta, calib, BOX, mle=mle, gpu=True)\n"
            "        c = _crlb("
            "theta, calib, BOX, mle=mle, gpu=False)\n"
            "        np.testing.assert_allclose(g, c, rtol=1e-5)\n"
            # the multichannel kernels, with the affine + ROI residual live
            "calib, _ = _gauss_spline_calibration("
            "model='spline-3d-multichannel', n_channels=2)\n"
            "theta = _shared_theta(2, rng)\n"
            "calib, res = _with_channel_geometry(calib, len(theta), rng)\n"
            "np.testing.assert_allclose(\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=True),\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=False),\n"
            "    rtol=1e-5)\n"
            "calib, theta = _link_xyz_calib_and_theta(2, 2, rng)\n"
            "calib, res = _with_channel_geometry(calib, len(theta), rng)\n"
            "np.testing.assert_allclose(\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=True),\n"
            "    _crlb(theta, calib, BOX, residuals=res, gpu=False),\n"
            "    rtol=1e-5)\n"
            "print('SIMOK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert "SIMOK" in result.stdout, result.stdout + result.stderr


def _gauss_model(theta, box, rotated):
    """Point-sampled Gaussian gpufit optimizes, parametrized by total photons N:
    mu(i, j) = N / (2 pi sx sy) * E + bg, i = x (column), j = y (row)."""
    N, x, y, sx, sy, bg = theta[:6]
    grid = np.arange(box, dtype=np.float64)
    dx = grid[:, None] - x
    dy = grid[None, :] - y
    if rotated:
        ct, st = np.cos(theta[6]), np.sin(theta[6])
        u = dx * ct - dy * st
        w = dx * st + dy * ct
        E = np.exp(-0.5 * (u**2 / sx**2 + w**2 / sy**2))
    else:
        E = np.exp(-0.5 * (dx**2 / sx**2 + dy**2 / sy**2))
    return N / (2 * np.pi * sx * sy) * E + bg


def _ref_gauss_crlb(theta, box, rotated, floor=1e-3):
    """Finite-difference Poisson Fisher CRLB reference for _gauss_crlb: builds
    I = sum g gᵀ / mu with numerical gradients g = d mu / d theta and inverts.
    """

    def model(t):
        return _gauss_model(t, box, rotated)

    n_params = len(theta)
    mu = np.maximum(model(theta), floor)
    g = np.zeros((n_params,) + mu.shape)
    for k in range(n_params):
        h = 1e-6 * max(abs(theta[k]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        g[k] = (model(tp) - model(tm)) / (2 * h)
    return np.diag(np.linalg.pinv(np.einsum("pij,qij->pq", g / mu, g)))


class TestGaussCRLB:
    """Poisson Cramer-Rao lower bounds for gpufit MLE Gaussian fits
    (precision._gauss_crlb). The analytic Fisher matrix of the point-sampled
    Gaussian is validated against a finite-difference reference; no GPU is
    needed since the CRLB is evaluated at given parameters."""

    def test_crlb_matches_reference_elliptic(self):
        # sx > sy so var_x > var_y - also guards the x/y association.
        theta = np.array([[500.0, 3.2, 3.7, 1.4, 1.0, 5.0]])
        crlb = precision._gauss_crlb(theta, BOX, em=False)[0]
        ref = _ref_gauss_crlb(theta[0], BOX, rotated=False)
        np.testing.assert_allclose(crlb, ref, rtol=1e-4)
        assert crlb[1] > crlb[2]  # var_x > var_y

    def test_crlb_matches_reference_rotated(self):
        theta = np.array([[800.0, 3.4, 3.1, 1.5, 0.9, 4.0, 0.6]])
        crlb = precision._gauss_crlb(theta, BOX, em=False, rotated=True)[0]
        ref = _ref_gauss_crlb(theta[0], BOX, rotated=True)
        np.testing.assert_allclose(crlb, ref, rtol=1e-4)
        assert np.isfinite(crlb[6]) and crlb[6] > 0  # finite angle variance

    def test_em_doubles_variance(self):
        theta = np.array([[500.0, 3.2, 3.7, 1.3, 1.1, 5.0]])
        crlb = precision._gauss_crlb(theta, BOX, em=False)[0]
        crlb_em = precision._gauss_crlb(theta, BOX, em=True)[0]
        np.testing.assert_allclose(crlb_em, 2.0 * crlb, rtol=1e-10)

    def test_nan_theta_row_isolated(self):
        theta = np.array(
            [
                [500.0, 3.2, 3.7, 1.3, 1.1, 5.0],
                [np.nan, 3.0, 3.0, 1.0, 1.0, 5.0],
            ]
        )
        crlb = precision._gauss_crlb(theta, BOX, em=False)
        assert np.all(np.isfinite(crlb[0]))
        assert np.all(np.isnan(crlb[1]))

    def test_low_signal_stays_finite(self):
        # bg = 0 drives outer model pixels to ~0; the MU_FLOOR guard keeps the
        # Fisher weight (1 / mu) finite.
        theta = np.array([[300.0, 3.0, 3.0, 1.2, 1.2, 0.0]])
        assert np.all(
            np.isfinite(precision._gauss_crlb(theta, BOX, em=False)[0])
        )

    def test_empty_input(self):
        assert precision._gauss_crlb(
            np.zeros((0, 6)), BOX, em=False
        ).shape == (
            0,
            6,
        )

    def test_more_photons_tightens_bound(self):
        # More photons tighten the position (x, y) and width (sx, sy) bounds.
        # var(N) itself grows with N (absolute photon-count noise increases).
        base = [3.2, 3.7, 1.3, 1.1, 5.0]
        dim = precision._gauss_crlb(np.array([[200.0, *base]]), BOX, em=False)[
            0
        ]
        bright = precision._gauss_crlb(
            np.array([[4000.0, *base]]), BOX, em=False
        )[0]
        assert np.all(bright[1:5] < dim[1:5])


def _link_xyz_theta(n_channels, n_locs=3, z_center=10.0, bg=2.0):
    """``[x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]`` with deliberately unequal
    per-channel photons, so a channel mix-up cannot pass unnoticed."""
    photons = 100.0 * (1.0 + np.arange(n_channels))
    theta = np.zeros((n_locs, 3 + 2 * n_channels))
    theta[:, 2] = -z_center
    theta[:, 3 : 3 + n_channels] = photons
    theta[:, 3 + n_channels :] = bg
    return theta, photons


class TestSplineLinkXyzColumns:
    """Output columns of the photon-decoupled (link-XYZ) multichannel spline
    fit. Pure numpy/numba - the constructor is a function of the fitted
    parameters, so no GPU is needed."""

    @staticmethod
    def _ids(n_locs):
        return pd.DataFrame(
            {
                "frame": np.arange(n_locs, dtype=np.uint32),
                "x": np.full(n_locs, 20.0),
                "y": np.full(n_locs, 30.0),
                "net_gradient": np.full(n_locs, 100.0),
            }
        )

    def _locs(self, n_channels, theta=None):
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        calib = localize._as_link_xyz_calibration(calib)
        photons = None
        if theta is None:
            theta, photons = _link_xyz_theta(
                n_channels, z_center=calib["z_center"]
            )
        locs = localize.locs_from_fits_spline(
            self._ids(len(theta)),
            theta,
            BOX,
            em=False,
            calibration=calib,
            mle=False,
        )
        return locs, photons

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_emits_one_column_per_channel(self, n_channels):
        locs, photons = self._locs(n_channels)
        for c in range(n_channels):
            assert f"photons_ch{c}" in locs.columns
            assert f"bg_ch{c}" in locs.columns
            assert f"rel_photons_ch{c}" in locs.columns
        # nothing for channels this calibration does not have
        assert f"photons_ch{n_channels}" not in locs.columns
        assert f"rel_photons_ch{n_channels}" not in locs.columns
        # the continuous readout replaced the old `color` column, and there is
        # no bare `rel_photons`
        assert "color" not in locs.columns
        assert "rel_photons" not in locs.columns

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_rel_photons_are_the_channel_shares(self, n_channels):
        locs, photons = self._locs(n_channels)
        rel = np.stack(
            [locs[f"rel_photons_ch{c}"].to_numpy() for c in range(n_channels)],
            axis=1,
        )
        # each channel's share of the total, summing to 1 per localization
        np.testing.assert_allclose(rel.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_allclose(
            rel, np.broadcast_to(photons / photons.sum(), rel.shape), rtol=1e-5
        )
        per_ch = np.stack(
            [locs[f"photons_ch{c}"].to_numpy() for c in range(n_channels)],
            axis=1,
        )
        np.testing.assert_allclose(
            locs["photons"].to_numpy(), per_ch.sum(axis=1), rtol=1e-5
        )
        np.testing.assert_allclose(per_ch, rel * photons.sum(), rtol=1e-5)

    def test_rel_photons_nan_without_photons(self):
        n_channels = 3
        theta, _ = _link_xyz_theta(n_channels, n_locs=2)
        theta[1, 3 : 3 + n_channels] = 0.0  # a spot that fitted no photons
        locs, _ = self._locs(n_channels, theta=theta)
        rel = np.stack(
            [locs[f"rel_photons_ch{c}"].to_numpy() for c in range(n_channels)],
            axis=1,
        )
        assert np.isfinite(rel[0]).all()
        assert np.isnan(rel[1]).all()


class TestSaveableColumns:
    """Guard the save whitelist: every column a fit path can emit must be
    registered in ``localize.LOCALIZATION_COLUMNS``. The GUI's
    ``Window.select_locs_columns`` keeps only whitelisted columns, so any
    unregistered column (e.g. the MLE ``*_unc`` uncertainties) is silently
    dropped before the .hdf5 is written. This test builds a representative
    localizations frame from every ``locs_from_fits_*`` constructor and fails
    if it produces a column the whitelist does not cover."""

    IDS = pd.DataFrame(
        {
            "frame": [0, 1],
            "x": [10, 20],
            "y": [15, 25],
            "net_gradient": [100.0, 120.0],
        }
    )

    def _fit_frames(self):
        """One output frame per fit path Picasso can save (GPU-free: the
        constructors are pure functions of the fitted parameters)."""
        ids = self.IDS
        stats = dict(
            log_likelihood=np.array([-10.0, -11.0], dtype=np.float32),
            iterations=np.array([5, 6], dtype=np.int32),
        )
        # Least-squares fits report a chi-square instead of a likelihood.
        lsq_stats = dict(
            chi_square=np.array([120.0, 95.0], dtype=np.float32),
            iterations=np.array([5, 6], dtype=np.int32),
        )
        frames = {}

        # gpufit Gaussian, [photons, x, y, sx, sy, bg] (+ angle if rotated)
        theta_e = np.array(
            [
                [500.0, 3.2, 3.7, 1.3, 1.1, 5.0],
                [800.0, 3.4, 3.1, 1.2, 1.2, 4.0],
            ],
            dtype=np.float32,
        )
        theta_r = np.array(
            [
                [800.0, 3.4, 3.1, 1.5, 0.9, 4.0, 0.6],
                [900.0, 3.0, 3.5, 1.4, 1.0, 3.0, -0.3],
            ],
            dtype=np.float32,
        )
        frames["gpufit-mle"] = localize.locs_from_fits_gauss(
            ids, theta_e, BOX, em=False, mle=True, **stats
        )
        frames["gpufit-mle-rotated"] = localize.locs_from_fits_gauss(
            ids, theta_r, BOX, em=False, mle=True, **stats
        )
        frames["gpufit-lse"] = localize.locs_from_fits_gauss(
            ids, theta_e, BOX, em=False, mle=False, **lsq_stats
        )
        frames["gpufit-lse-rotated"] = localize.locs_from_fits_gauss(
            ids, theta_r, BOX, em=False, mle=False, **lsq_stats
        )

        # CPU least-squares Gaussian, [x, y, photons, bg, sx, sy]
        theta_lq = np.array(
            [
                [3.0, 3.0, 500.0, 5.0, 1.2, 1.1],
                [3.1, 2.9, 600.0, 4.0, 1.0, 1.3],
            ]
        )
        frames["gausslq-cpu"] = gausslq.locs_from_fits(
            ids, theta_lq, BOX, em=False, chi_square=lsq_stats["chi_square"]
        )

        # spline PSF, [amplitude, x_shift, y_shift, (z_shift,) offset]
        calib_2d, _ = _gauss_spline_calibration(model="spline-2d")
        calib_3d, _ = _gauss_spline_calibration(model="spline-3d")
        theta_s2 = np.array(
            [[3000.0, 0.1, -0.1, 15.0], [2800.0, 0.05, -0.05, 14.0]]
        )
        theta_s3 = np.array(
            [[4000.0, 0.2, -0.15, -6.0, 20.0], [3800.0, 0.1, -0.1, -5.0, 18.0]]
        )
        frames["spline-2d"] = localize.locs_from_fits_spline(
            ids,
            theta_s2,
            BOX,
            em=False,
            calibration=calib_2d,
            mle=True,
            **stats,
        )
        frames["spline-3d"] = localize.locs_from_fits_spline(
            ids,
            theta_s3,
            BOX,
            em=False,
            calibration=calib_3d,
            mle=True,
            **stats,
        )

        # CPU MLE Gaussian, [x, y, photons, bg, sx, sy] with its 6-col CRLB
        theta_cpu = np.array(
            [
                [3.0, 3.0, 500.0, 5.0, 1.2, 1.1],
                [3.1, 2.9, 600.0, 4.0, 1.0, 1.3],
            ]
        )
        crlbs = np.full((2, 6), 0.01)
        frames["gaussmle-cpu"] = gaussmle.locs_from_fits(
            ids,
            theta_cpu,
            crlbs,
            stats["log_likelihood"],
            stats["iterations"],
            BOX,
        )
        # Photon-decoupled (link-XYZ) multichannel spline, at the largest
        # channel count there is a fit model for - it emits the widest set of
        # per-channel columns.
        n_channels = precision._LINK_XYZ_MAX_CHANNELS
        calib_mc, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel", n_channels=n_channels
        )
        calib_link = localize._as_link_xyz_calibration(calib_mc)
        theta_link, _ = _link_xyz_theta(
            n_channels, n_locs=2, z_center=calib_link["z_center"]
        )
        frames["spline-3d-link-xyz"] = localize.locs_from_fits_spline(
            ids,
            theta_link,
            BOX,
            em=False,
            calibration=calib_link,
            mle=False,
            **lsq_stats,
        )

        # Ratiometric multichannel spline: the shared-amplitude model plus the
        # per-channel photons and the integer `color` (the assigned channel)
        # that fit_spline_multichannel_ratiometric bolts on afterwards.
        theta_mc = np.array(
            [
                [4000.0, 0.2, -0.15, -6.0, 20.0],
                [3800.0, 0.1, -0.10, -5.0, 18.0],
            ]
        )
        locs_ratio = localize.locs_from_fits_spline(
            ids,
            theta_mc,
            BOX,
            em=False,
            calibration=calib_mc,
            mle=False,
            **lsq_stats,
        )
        ratios = np.arange(1.0, n_channels + 1.0)
        ratios /= ratios.sum()
        for c in range(n_channels):
            locs_ratio[f"photons_ch{c}"] = (
                locs_ratio["photons"] * ratios[c]
            ).astype(np.float32)
        locs_ratio["color"] = np.int32(1)
        frames["spline-3d-ratiometric"] = locs_ratio
        return frames

    def test_all_fit_columns_are_whitelisted(self):
        saveable = {
            col
            for cols in localize.LOCALIZATION_COLUMNS.values()
            for col in cols
        }
        offenders = {
            name: sorted(set(locs.columns) - saveable)
            for name, locs in self._fit_frames().items()
            if set(locs.columns) - saveable
        }
        assert not offenders, (
            "fit paths emit columns missing from LOCALIZATION_COLUMNS (they "
            f"would be dropped on save): {offenders}"
        )

    def test_uncertainty_columns_survive_select_locs_columns(self):
        # Mirror Window.select_locs_columns: keep only whitelisted columns (all
        # checkboxes default checked). The MLE uncertainties must survive.
        saveable = {
            col
            for cols in localize.LOCALIZATION_COLUMNS.values()
            for col in cols
        }
        frames = self._fit_frames()
        for expected in ("photons_unc", "bg_unc", "sx_unc", "sy_unc"):
            locs = frames["gpufit-mle"]
            kept = [c for c in locs.columns if c in saveable]
            assert expected in kept, f"{expected} dropped for gpufit-mle"
        assert "angle_unc" in [
            c for c in frames["gpufit-mle-rotated"].columns if c in saveable
        ]


# ---------------------------------------------------------------------------
# Ratiometric multichannel spline fitting (globLoc-style color assignment)
# ---------------------------------------------------------------------------


def _stack_multichannel(
    calib1, n_channels=2, photon_scale=None, transforms=None
):
    """Turn a single-channel 3D spline calibration into a multichannel one by
    replicating its (real, fit-grade) coefficient block across channels."""
    identity = IDENTITY
    calib = dict(calib1)
    calib["model"] = "spline-3d-multichannel"
    calib["coefficients"] = np.repeat(
        np.asarray(calib1["coefficients"])[..., None], n_channels, axis=-1
    ).astype(np.float32)
    calib["n_channels"] = n_channels
    calib["channel_transforms"] = transforms or [identity] * n_channels
    if photon_scale is not None:
        calib["photon_scale"] = photon_scale
    return calib


class TestScaleChannelBlocks:
    """Pure-Python coefficient scaling / photon_scale helpers (no GPU)."""

    def test_scales_each_channel(self):
        coeff = np.arange(64 * 2 * 2 * 3 * 2, dtype=np.float32).reshape(
            64, 2, 2, 3, 2
        )
        scaled = localize.scale_channel_blocks(coeff, [0.25, 4.0])
        np.testing.assert_allclose(scaled[..., 0], coeff[..., 0] * 0.25)
        np.testing.assert_allclose(scaled[..., 1], coeff[..., 1] * 4.0)

    def test_input_not_modified(self):
        coeff = np.ones((64, 2, 2, 3, 2), dtype=np.float32)
        before = coeff.copy()
        localize.scale_channel_blocks(coeff, [2.0, 3.0])
        np.testing.assert_array_equal(coeff, before)

    def test_ratio_length_mismatch(self):
        coeff = np.ones((64, 2, 2, 3, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            localize.scale_channel_blocks(coeff, [1.0, 2.0, 3.0])

    def test_requires_multichannel_table(self):
        with pytest.raises(ValueError):
            localize.scale_channel_blocks(
                np.ones((64, 2, 2, 3), np.float32), [1.0]
            )

    def test_photon_scales_scalar_broadcast(self):
        ps = localize._photon_scales({"photon_scale": 2.5}, 3)
        np.testing.assert_array_equal(ps, [2.5, 2.5, 2.5])

    def test_photon_scales_array_passthrough(self):
        ps = localize._photon_scales({"photon_scale": [1.0, 2.0, 3.0]}, 3)
        np.testing.assert_array_equal(ps, [1.0, 2.0, 3.0])


class TestSplinePerChannelPhotonScale:
    """locs_from_fits_spline maps the shared amplitude to TOTAL photons when
    photon_scale is a per-channel array (sum). CPU-only (numba CRLB)."""

    def test_total_photons_sum_per_channel(self):
        calib, _ = _gauss_spline_calibration(
            model="spline-3d-multichannel",
            n_channels=2,
            photon_scale=[2.0, 3.0],
        )
        box = calib["n_data"][0]  # box == cal box -> crop is a no-op
        amp = 100.0
        theta = np.array(
            [[amp, 0.0, 0.0, -calib["z_center"], 5.0]], np.float64
        )
        ids = pd.DataFrame(
            {
                "frame": [0],
                "x": [box // 2],
                "y": [box // 2],
                "net_gradient": [1.0],
            }
        )
        locs = localize.locs_from_fits_spline(
            ids, theta, box, em=False, calibration=calib, mle=False
        )
        # photons = amplitude * (2 + 3)
        np.testing.assert_allclose(
            locs["photons"].iloc[0], amp * 5.0, rtol=1e-4
        )


class TestSplineRatiometric:
    """End-to-end ratiometric color assignment on a real PSF,
    stacked into two identical channels. The two channels differ only by a
    known photon split; the fitter must recover it as the winning ratio."""

    CANDS = np.array([[0.9, 0.1], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75]])
    TRUE_IDX = 1  # [0.75, 0.25]

    def _calib_and_base(self):
        calib1, template, _, _ = _synthetic_spline_3d_calibration()
        z_slice = int(calib1["z_center"])
        base = template[:, :, z_slice].astype(np.float32)
        calib = _stack_multichannel(calib1, 2, photon_scale=[1.0, 1.0])
        return calib, base

    def test_core_selection_recovers_true_ratio(self):
        calib, base = self._calib_and_base()
        box = calib["n_data"][0]
        r_true = self.CANDS[self.TRUE_IDX]
        rng = np.random.default_rng(0)
        n = 150
        spots = np.empty((n, box, box, 2), np.float32)
        for c in range(2):
            mu = np.clip(20.0 + 3000.0 * r_true[c] * base, 0, None)
            spots[..., c] = rng.poisson(mu, size=(n, box, box)).astype(
                np.float32
            )
        rn = self.CANDS / self.CANDS.sum(1, keepdims=True)
        scores = np.full((len(rn), n), np.inf)
        for k, r in enumerate(rn):
            ck = dict(calib)
            ck["coefficients"] = localize.scale_channel_blocks(
                calib["coefficients"], r
            )
            params, chi, _states, _n_it = localize._run_splinefit(
                spots, ck, mle=False, use_gpu=localize.CUDA_AVAILABLE
            )
            fin = np.isfinite(params).all(1) & np.isfinite(chi)
            scores[k] = np.where(fin, chi, np.inf)
        best = np.argmin(scores, axis=0)
        assert (best == self.TRUE_IDX).mean() > 0.95

    def test_entry_point_colors_and_photons(self):
        calib, base = self._calib_and_base()
        box = calib["n_data"][0]
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1, "Qe": 1.0}
        r_true = self.CANDS[self.TRUE_IDX]
        rng = np.random.default_rng(1)
        n = 80
        movies = []
        for c in range(2):
            mu = np.clip(20.0 + 3000.0 * r_true[c] * base, 0, None)
            mov = np.stack(
                [rng.poisson(mu).astype(np.float32) for _ in range(n)]
            )
            movies.append(mov)
        ids = pd.DataFrame(
            {
                "frame": np.arange(n),
                "x": np.full(n, box // 2),
                "y": np.full(n, box // 2),
                "net_gradient": np.ones(n),
            }
        )
        locs = localize.fit_spline_multichannel_ratiometric(
            movies,
            [cam, cam],
            ids,
            box,
            calib,
            photon_ratios=self.CANDS,
        )
        assert len(locs) == n
        assert (locs["color"] == self.TRUE_IDX).mean() > 0.9
        # per-channel photons track the 0.75/0.25 split; total ~= 3000
        frac0 = locs["photons_ch0"].median() / (
            locs["photons_ch0"].median() + locs["photons_ch1"].median()
        )
        assert abs(frac0 - 0.75) < 0.06
        for col in ("z", "lpx", "lpy", "lpz", "photons_ch0", "photons_ch1"):
            assert col in locs.columns

    def test_ratios_from_calibration_when_omitted(self):
        calib, base = self._calib_and_base()
        calib["photon_ratios"] = self.CANDS.tolist()
        box = calib["n_data"][0]
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1, "Qe": 1.0}
        r_true = self.CANDS[self.TRUE_IDX]
        rng = np.random.default_rng(2)
        n = 40
        movies = []
        for c in range(2):
            mu = np.clip(20.0 + 3000.0 * r_true[c] * base, 0, None)
            movies.append(
                np.stack(
                    [rng.poisson(mu).astype(np.float32) for _ in range(n)]
                )
            )
        ids = pd.DataFrame(
            {
                "frame": np.arange(n),
                "x": np.full(n, box // 2),
                "y": np.full(n, box // 2),
                "net_gradient": np.ones(n),
            }
        )
        # no photon_ratios argument -> taken from the calibration
        locs = localize.fit_spline_multichannel_ratiometric(
            movies, [cam, cam], ids, box, calib
        )
        assert (locs["color"] == self.TRUE_IDX).mean() > 0.9

    def test_missing_ratios_raises(self):
        calib, _ = self._calib_and_base()
        cam = {"Baseline": 0, "Sensitivity": 1.0, "Gain": 1}
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        box = calib["n_data"][0]
        movie = np.zeros((1, box, box), np.float32)
        with pytest.raises(ValueError):
            localize.fit_spline_multichannel_ratiometric(
                [movie, movie], [cam, cam], ids, box, calib
            )


# ---------------------------------------------------------------------------
# Split-FOV multichannel: regions of ONE movie treated as channels
# ---------------------------------------------------------------------------


def _split_fov_bead_movie(dx=2, dy=-1, n_frames=21):
    """One movie whose left/right 48x48 halves are two channels: the right
    half is the left (reference) half shifted within its region by ``(dx, dy)``.
    Returns ``(movie, regions, bead_xy, focus)``."""
    bead_xy = [(12, 14), (30, 28), (16, 33)]
    s0 = 1.1
    focus = n_frames // 2
    yy, xx = np.mgrid[0:48, 0:48]
    base = np.zeros((n_frames, 48, 48), dtype=np.float32)
    for f in range(n_frames):
        sigma = s0 * (1.0 + 0.07 * abs(f - focus))
        img = np.full((48, 48), 100.0, dtype=np.float32)
        for bx, by in bead_xy:
            img += 3000.0 * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )
        base[f] = img
    base = base.astype(np.uint16)
    movie = np.zeros((n_frames, 48, 96), dtype=np.uint16)
    movie[:, :, :48] = base
    movie[:, :, 48:] = np.roll(base, shift=(dy, dx), axis=(1, 2))
    regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    return movie, regions, bead_xy, focus


class TestSplitFovRegionAffines:
    """Region-local <-> absolute channel-transform conversion (no GPU)."""

    def test_decompose_compose_round_trip(self):
        regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        t0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        tc = [[1.0, 0.0, 50.0], [0.0, 1.0, -1.0]]  # 48 offset + (2, -1) fine
        affines = localize.decompose_region_transforms(
            regions, [affine(t0), affine(tc)]
        )
        # region-local affine carries only the fine (2, -1) registration
        np.testing.assert_allclose(
            affine_matrix(affines[0]), [[1, 0, 0], [0, 1, 0]], atol=1e-9
        )
        np.testing.assert_allclose(
            affine_matrix(affines[1]), [[1, 0, 2], [0, 1, -1]], atol=1e-9
        )
        back = localize.compose_region_transforms(regions, affines)
        np.testing.assert_allclose(affine_matrix(back[0]), t0, atol=1e-9)
        np.testing.assert_allclose(affine_matrix(back[1]), tc, atol=1e-9)

    def test_replace_at_new_positions(self):
        regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        tc = [[1.0, 0.0, 50.0], [0.0, 1.0, -1.0]]
        affines = localize.decompose_region_transforms(
            regions, [affine([[1, 0, 0], [0, 1, 0]]), affine(tc)]
        )
        # channels re-placed: reference at (10, 10), channel at (10, 200)
        new = [[[10, 10], [58, 58]], [[10, 200], [58, 248]]]
        t = localize.compose_region_transforms(new, affines)
        # reference is identity at its new spot; channel = new offset + fine
        np.testing.assert_allclose(
            affine_matrix(t[0]), [[1, 0, 0], [0, 1, 0]], atol=1e-9
        )
        np.testing.assert_allclose(
            affine_matrix(t[1]), [[1, 0, 192], [0, 1, -1]], atol=1e-9
        )

    def test_rotation_is_position_independent(self):
        # a small rotation in the linear part survives re-placement unchanged
        th = np.deg2rad(3.0)
        L = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        regions = [[[0, 0], [40, 40]], [[0, 50], [40, 90]]]
        tc = [[L[0][0], L[0][1], 55.0], [L[1][0], L[1][1], 2.0]]
        affines = localize.decompose_region_transforms(
            regions, [affine([[1, 0, 0], [0, 1, 0]]), affine(tc)]
        )
        new = [[[5, 5], [45, 45]], [[5, 300], [45, 340]]]
        t = localize.compose_region_transforms(new, affines)
        # linear (rotation) part is unchanged by moving the regions
        np.testing.assert_allclose(linear_part(t[1]), L, atol=1e-9)


class TestFitSplineSplitFovValidation:
    """Argument validation that needs no GPU."""

    def test_requires_split_fov_calibration(self):
        calib = {"model": "spline-3d-multichannel"}  # no split_fov flag
        movie = np.zeros((1, 16, 32), np.float32)
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        with pytest.raises(ValueError, match="split-FOV"):
            localize.fit_spline_split_fov(movie, CAMERA_INFO, ids, BOX, calib)


class TestFitSplineSplitFov:
    """End-to-end split-FOV fit on a single movie built + calibrated from the
    same beads (model matches, so the fit recovers the bead positions)."""

    def _movie_and_calib(self):
        dx, dy = 2, -1
        movie, regions, bead_xy, focus = _split_fov_bead_movie(dx, dy)
        info = [{"Frames": int(movie.shape[0])}]
        calib = spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
        )
        return movie, calib, bead_xy, focus

    def test_fit_returns_locs_in_reference_region(self):
        movie, calib, bead_xy, focus = self._movie_and_calib()
        # detect the reference-region beads on the in-focus frames
        ids, _ = localize.identify(
            movie,
            2000.0,
            BOX,
            roi=calib["regions"][0],
            frame_bounds=(focus - 1, focus + 1),
        )
        locs = localize.fit_spline_split_fov(
            movie, CAMERA_INFO, ids, BOX, calib
        )
        assert len(locs) > 0
        # localizations are in the reference region's coordinates
        assert np.all(locs["x"] >= 0) and np.all(locs["x"] < 48)
        assert np.all(locs["y"] >= 0) and np.all(locs["y"] < 48)
        # each fitted spot sits near one of the three reference beads
        true_x = np.array([b[0] for b in bead_xy])
        true_y = np.array([b[1] for b in bead_xy])
        for x, y in zip(locs["x"], locs["y"]):
            d = np.hypot(true_x - x, true_y - y)
            assert d.min() < 2.0

    def test_confines_identifications_to_reference_region(self):
        movie, calib, bead_xy, focus = self._movie_and_calib()
        # identify over the WHOLE frame -> detections in both halves
        ids, _ = localize.identify(
            movie, 2000.0, BOX, frame_bounds=(focus, focus)
        )
        assert (ids["x"] >= 48).any()  # some detections are in region 1
        locs = localize.fit_spline_split_fov(
            movie, CAMERA_INFO, ids, BOX, calib, confine_to_reference=True
        )
        # only the reference-region molecules are fitted (region 1 is the
        # mapped partner, not an independent detection)
        assert len(locs) > 0
        assert np.all(locs["x"] < 48)

    def test_fit_at_repositioned_regions(self):
        """ROI-agnostic: the same calibration fits data whose split sits at a
        different position, when the fit-time ROIs are supplied."""
        movie, calib, bead_xy, focus = self._movie_and_calib()
        assert (
            "channel_registration" in calib
        )  # ROI-agnostic registration stored
        # embed the identical split (same inter-channel geometry) at a global
        # offset in a larger frame
        oy, ox = 20, 60
        n, h, w = movie.shape
        big = np.zeros((n, h + oy + 10, w + ox + 10), dtype=movie.dtype)
        big[:, oy : oy + h, ox : ox + w] = movie
        ref_region = [[oy, ox], [oy + 48, ox + 48]]
        chan_region = [[oy, ox + 48], [oy + 48, ox + 96]]
        fit_regions = [ref_region, chan_region]

        ids, _ = localize.identify(
            big,
            2000.0,
            BOX,
            roi=ref_region,
            frame_bounds=(focus - 1, focus + 1),
        )
        locs = localize.fit_spline_split_fov(
            big, CAMERA_INFO, ids, BOX, calib, regions=fit_regions
        )
        assert len(locs) > 0
        # localizations land near the shifted reference beads
        tx = np.array([ox + b[0] for b in bead_xy])
        ty = np.array([oy + b[1] for b in bead_xy])
        for x, y in zip(locs["x"], locs["y"]):
            assert np.hypot(tx - x, ty - y).min() < 2.0

        # control: without fit-time regions, the calibration's original
        # positions no longer match the data -> nothing in the reference region
        ids_all, _ = localize.identify(
            big, 2000.0, BOX, frame_bounds=(focus, focus)
        )
        locs_orig = localize.fit_spline_split_fov(
            big, CAMERA_INFO, ids_all, BOX, calib
        )
        assert len(locs_orig) < len(locs)


@pytest.mark.gui
class TestMultichannelWorkerRouting:
    """MultichannelSplineFitWorker routes to the ratiometric fitter iff the
    calibration carries photon_ratios (no GPU; the fit fns are monkeypatched).
    """

    def _run(self, calibration, monkeypatch):
        calls = []
        df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0]})
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel_ratiometric",
            lambda *a, **k: (calls.append("ratio"), df)[1],
        )
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel",
            lambda *a, **k: (calls.append("plain"), df)[1],
        )
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelSplineFitWorker(
            [None, None], [{}, {}], ids, BOX, calibration, mle=False
        )
        result = {}
        worker.finished.connect(
            lambda locs, dt, a, b: result.update(locs=locs)
        )
        worker.run()
        return calls, result

    def test_routes_to_ratiometric_with_ratios(self, monkeypatch):
        calls, result = self._run(
            {"model": "spline-3d-multichannel", "photon_ratios": [[0.7, 0.3]]},
            monkeypatch,
        )
        assert calls == ["ratio"]
        assert "locs" in result

    def test_routes_to_plain_without_ratios(self, monkeypatch):
        calls, _ = self._run({"model": "spline-3d-multichannel"}, monkeypatch)
        assert calls == ["plain"]

    def _run_unlinked(self, n_channels, monkeypatch, photon_ratios=None):
        """Route with photons UNLINKED, recording which fitter ran and the
        ``link_photons`` it was called with ("default" = not passed)."""
        calls = []
        df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0]})
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel_ratiometric",
            lambda *a, **k: (calls.append("ratio"), df)[1],
        )
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel",
            lambda *a, **k: (
                calls.append(("plain", k.get("link_photons", "default"))),
                df,
            )[1],
        )
        calibration = {
            "model": "spline-3d-multichannel",
            "n_channels": n_channels,
        }
        if photon_ratios is not None:
            calibration["photon_ratios"] = photon_ratios
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelSplineFitWorker(
            [None] * n_channels,
            [{}] * n_channels,
            ids,
            BOX,
            calibration,
            mle=False,
            link_photons=False,
        )
        worker.run()
        return calls

    @pytest.mark.parametrize("n_channels", [2, 3, 4, 5, 6])
    def test_unlinked_photons_route_to_link_xyz(self, n_channels, monkeypatch):
        assert self._run_unlinked(n_channels, monkeypatch) == [
            ("plain", False)
        ]

    def test_unlinked_photons_supersede_ratiometric(self, monkeypatch):
        # photon decoupling fits the ratio instead of scanning hypotheses, so
        # it wins over the ratiometric branch when both are possible
        calls = self._run_unlinked(
            4, monkeypatch, photon_ratios=[[0.4, 0.3, 0.2, 0.1]]
        )
        assert calls == [("plain", False)]

    def test_unlinked_photons_above_cap_fall_back_to_linked(self, monkeypatch):
        # no link-XYZ fit model is compiled beyond the cap, so the
        # shared-amplitude model (which takes any channel count) is used
        n_channels = precision._LINK_XYZ_MAX_CHANNELS + 1
        assert self._run_unlinked(n_channels, monkeypatch) == [
            ("plain", "default")
        ]

    def test_fits_only_spots_linked_across_channels(self, monkeypatch):
        """Given every channel's identifications, the worker hands the fitter
        only the reference spots detected in all channels."""
        seen = {}
        monkeypatch.setattr(
            localize,
            "fit_spline_multichannel",
            lambda movies, cams, ids, *a, **k: (
                seen.update(ids=ids),
                pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0]}),
            )[1],
        )
        identity = IDENTITY
        ref = pd.DataFrame(
            {
                "frame": [0, 0],
                "x": [6.0, 60.0],
                "y": [6.0, 60.0],
                "net_gradient": [1.0, 1.0],
            }
        )
        # channel 1 only sees the first reference spot
        ch1 = pd.DataFrame(
            {"frame": [0], "x": [6.2], "y": [5.9], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelSplineFitWorker(
            [None, None],
            [{}, {}],
            ref,
            BOX,
            {
                "model": "spline-3d-multichannel",
                "n_channels": 2,
                "channel_transforms": [identity, identity],
            },
            identifications_per_channel=[ref, ch1],
        )
        linked = []
        worker.linkFinished.connect(
            lambda kept, total: linked.append((kept, total))
        )
        worker.run()
        assert linked == [(1, 2)]
        assert len(seen["ids"]) == 1
        assert float(seen["ids"]["x"].iloc[0]) == 6.0
        # progress totals follow the linked subset, not the original count
        assert worker.N == 1


# ---------------------------------------------------------------------------
# Hover tooltips over fit markers / identification boxes
#
# Hovering a FitMarker or an identification box in the localize GUI shows
# the columns of the corresponding localization (or identification, before
# fitting). These tests cover the tooltip text formatting and the matching
# of a box position to the fitted localization inside it.
# ---------------------------------------------------------------------------


@pytest.mark.gui
class TestHoverTooltips:
    def test_format_hover_tooltip_lists_all_columns(self):
        row = pd.Series(
            {"frame": 3, "x": 234.123456, "y": 12.654, "photons": 1234.5678}
        )
        text = localize_gui.format_hover_tooltip(row)
        assert text.splitlines() == [
            "frame: 3",
            "x: 234.123",
            "y: 12.654",
            "photons: 1234.57",
        ]

    def test_loc_near_picks_closest_within_radius(self):
        locs = pd.DataFrame(
            {
                "x": [10.4, 12.0, 30.0],
                "y": [9.8, 11.5, 30.0],
                "photons": [100.0, 200.0, 300.0],
            }
        )
        loc = localize_gui.Window._loc_near(locs, 10, 10, 3)
        assert loc is not None
        assert loc["photons"] == 100.0

    def test_loc_near_none_outside_radius_or_without_locs(self):
        locs = pd.DataFrame({"x": [30.0], "y": [30.0]})
        assert localize_gui.Window._loc_near(locs, 10, 10, 3) is None
        assert localize_gui.Window._loc_near(None, 10, 10, 3) is None
        empty = locs[locs["x"] < 0]
        assert localize_gui.Window._loc_near(empty, 10, 10, 3) is None


# ---------------------------------------------------------------------------
# Temporal median filter (identification-only background subtraction)
# ---------------------------------------------------------------------------


def _notebook_temporal_median_filter(
    stack: np.ndarray, temporal_length: int
) -> np.ndarray:
    """Reference implementation, transcribed from the teaching notebook
    this feature is based on (Endesfelder Lab, SMLMComputational module 1).

    Deliberately kept as the original triple loop so that it is obvious it
    has not been "helpfully" rewritten alongside the code it validates.
    """
    filtered = np.zeros(stack.shape)
    for tt in range(stack.shape[0]):
        for xx in range(stack.shape[1]):
            for yy in range(stack.shape[2]):
                start_t = tt - (temporal_length - 1) / 2
                end_t = tt + (temporal_length - 1) / 2 + 1
                if start_t < 0:
                    end_t = end_t - start_t
                    start_t = 0
                if end_t > stack.shape[0]:
                    start_t = start_t - (end_t - stack.shape[0])
                    end_t = stack.shape[0]
                median = np.median(stack[int(start_t) : int(end_t), xx, yy])
                if stack[tt, xx, yy] >= median:
                    filtered[tt, xx, yy] = stack[tt, xx, yy] - median
                else:
                    filtered[tt, xx, yy] = 0
    return filtered


class _CountingMovie:
    """Movie wrapper that counts how often a frame is read."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.reads = 0

    def __getitem__(self, it):
        self.reads += 1
        return self.data[it]

    def __len__(self) -> int:
        return len(self.data)


class TestTemporalMedian:
    """``localize.TemporalMedianMovie`` and its use from ``identify``."""

    @staticmethod
    def _movie(n_frames=24, height=8, width=6, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 4000, size=(n_frames, height, width), dtype=np.uint16
        )

    @pytest.mark.parametrize("n_frames", [5, 6])
    def test_temporal_median_matches_numpy(self, n_frames):
        stack = self._movie(n_frames=n_frames)
        median = localize._temporal_median(stack)
        assert median.dtype == np.float32
        np.testing.assert_allclose(
            median, np.median(stack, axis=0), rtol=0, atol=1e-3
        )

    def test_stripes_do_not_change_the_result(self):
        stack = self._movie(n_frames=9, height=17, width=5)
        one_stripe = localize._temporal_median(stack)
        many_stripes = localize._temporal_median(stack, max_stripe_bytes=1)
        np.testing.assert_array_equal(one_stripe, many_stripes)

    def test_stride_one_matches_the_notebook(self):
        """``stride=1`` must reproduce the reference filter exactly."""
        stack = self._movie(n_frames=12, height=8, width=8, seed=3)
        filtered = localize.TemporalMedianMovie(stack, 5, stride=1)
        expected = _notebook_temporal_median_filter(stack, 5)
        for frame_number in range(len(stack)):
            np.testing.assert_allclose(
                filtered[frame_number], expected[frame_number], atol=1e-3
            )

    def test_window_is_shifted_not_truncated_at_the_edges(self):
        stack = self._movie(n_frames=20)
        filtered = localize.TemporalMedianMovie(stack, 7, stride=1)
        for frame_number in range(len(stack)):
            start, stop = filtered._bounds(filtered._block_index(frame_number))
            assert stop - start == 7
            assert 0 <= start < stop <= len(stack)
            # the frame being filtered always lies inside its own window
            assert start <= frame_number < stop

    def test_window_longer_than_movie_uses_whole_movie(self):
        stack = self._movie(n_frames=5)
        filtered = localize.TemporalMedianMovie(stack, 51)
        assert filtered.window == 5
        expected = np.maximum(
            stack.astype(np.float32) - np.median(stack, axis=0), 0
        )
        for frame_number in range(len(stack)):
            np.testing.assert_allclose(
                filtered[frame_number], expected[frame_number], atol=1e-3
            )

    def test_every_frame_lies_inside_its_block_window(self):
        """Holds for any stride, and is what lets the block serve raw
        frames straight out of its cached window."""
        stack = self._movie(n_frames=37)
        for window in (3, 8, 11):
            for stride in (1, 2, window):
                filtered = localize.TemporalMedianMovie(
                    stack, window, stride=stride
                )
                for frame_number in range(len(stack)):
                    start, stop = filtered._bounds(
                        filtered._block_index(frame_number)
                    )
                    assert start <= frame_number < stop

    def test_movie_protocol(self):
        stack = self._movie()
        filtered = localize.TemporalMedianMovie(stack, 5)
        assert len(filtered) == len(stack)
        assert filtered.shape == stack.shape
        assert filtered.dtype == np.float32
        assert filtered[-1].shape == stack.shape[1:]
        np.testing.assert_array_equal(filtered[-1], filtered[len(stack) - 1])
        np.testing.assert_array_equal(
            filtered[2:5], np.stack([filtered[i] for i in (2, 3, 4)])
        )
        np.testing.assert_array_equal(
            np.stack(list(iter(filtered))),
            np.stack([filtered[i] for i in range(len(stack))]),
        )
        with pytest.raises(IndexError):
            filtered[len(stack)]

    def test_output_is_non_negative_float32(self):
        filtered = localize.TemporalMedianMovie(self._movie(), 5)
        for frame_number in range(len(filtered)):
            frame = filtered[frame_number]
            assert frame.dtype == np.float32
            assert (frame >= 0).all()

    def test_cache_does_not_change_the_result(self):
        stack = self._movie(n_frames=30)
        cached = localize.TemporalMedianMovie(stack, 7)
        uncached = localize.TemporalMedianMovie(stack, 7, cache_bytes=0)
        first_pass = [cached[i] for i in range(len(stack))]
        second_pass = [cached[i] for i in range(len(stack))]
        for i in range(len(stack)):
            np.testing.assert_array_equal(first_pass[i], second_pass[i])
            np.testing.assert_array_equal(first_pass[i], uncached[i])

    def test_eviction_stays_within_budget_and_keeps_results(self):
        """A tight budget must drop cached frames (and then whole blocks)
        without changing what the filter returns."""
        stack = self._movie(n_frames=60)
        reference = localize.TemporalMedianMovie(stack, 5, stride=1)
        one_median = reference[0].nbytes
        # room for a handful of medians but not for any raw window
        budget = 6 * one_median
        tight = localize.TemporalMedianMovie(
            stack, 5, stride=1, cache_bytes=budget
        )
        for frame_number in range(len(stack)):
            np.testing.assert_array_equal(
                tight[frame_number], reference[frame_number]
            )
            resident = sum(block.nbytes for block in tight._cache.values())
            assert resident <= budget or len(tight._cache) <= 2

    def test_each_frame_is_read_once(self):
        """With the default stride the cached window serves every frame it
        covers, so the movie is read exactly once end to end."""
        stack = self._movie(n_frames=30)
        counting = _CountingMovie(stack)
        filtered = localize.TemporalMedianMovie(counting, 10)
        counting.reads = 0  # ignore the geometry probe read in __init__
        for frame_number in range(len(stack)):
            filtered[frame_number]
        assert counting.reads == len(stack)

    def test_roi_restricts_the_median_but_not_the_geometry(self):
        stack = self._movie(n_frames=20, height=40, width=40)
        roi = ((0, 0), (8, 8))
        filtered = localize.TemporalMedianMovie(stack, 5, roi=roi, roi_pad=1)
        full = localize.TemporalMedianMovie(stack, 5)
        for frame_number in (0, 7, 19):
            frame = filtered[frame_number]
            assert frame.shape == stack.shape[1:]
            # inside the ROI the result is identical to filtering everything
            np.testing.assert_allclose(
                frame[:8, :8], full[frame_number][:8, :8], atol=1e-3
            )
            # outside the padded bounding box nothing is reported
            assert (frame[10:, 10:] == 0).all()

    def test_threaded_matches_serial(self, movie):
        ids_t = localize.identify(
            movie,
            MIN_NG,
            BOX,
            threaded=True,
            temporal_median_window=11,
            return_info=False,
        )
        ids_s = localize.identify(
            movie,
            MIN_NG,
            BOX,
            threaded=False,
            temporal_median_window=11,
            return_info=False,
        )
        as_set = lambda ids: set(  # noqa: E731
            map(tuple, ids[["frame", "y", "x"]].to_numpy())
        )
        assert as_set(ids_t) == as_set(ids_s)

    def test_identify_argument_matches_explicit_wrapper(self, movie):
        ids_arg, info = localize.identify(
            movie, MIN_NG, BOX, threaded=False, temporal_median_window=11
        )
        ids_wrapped = localize.identify(
            localize.TemporalMedianMovie(movie, 11, roi_pad=int(BOX / 2) + 1),
            MIN_NG,
            BOX,
            threaded=False,
            return_info=False,
        )
        pd.testing.assert_frame_equal(ids_arg, ids_wrapped)
        assert info["Temporal Median Window"] == 11

    def test_info_records_zero_when_disabled(self, movie):
        _, info = localize.identify(movie, MIN_NG, BOX, threaded=False)
        assert info["Temporal Median Window"] == 0

    def test_static_structure_is_removed_but_blinking_spots_survive(self):
        """A bright but constant structure must stop being identified,
        while a spot that is only on in a few frames must not."""
        n_frames, size = 40, 32
        movie = np.full((n_frames, size, size), 100, dtype=np.uint16)
        yy, xx = np.mgrid[0:size, 0:size]

        def gaussian(y0, x0, amplitude):
            return amplitude * np.exp(
                -((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.2**2)
            )

        movie += gaussian(8, 8, 3000).astype(np.uint16)  # static structure
        blinking = [5, 6, 20, 21]
        for frame_number in blinking:
            movie[frame_number] += gaussian(24, 24, 3000).astype(np.uint16)

        found = lambda ids, y, x: (  # noqa: E731
            (ids["y"] - y).abs().le(1) & (ids["x"] - x).abs().le(1)
        ).any()

        raw_ids = localize.identify(
            movie, 500, BOX, threaded=False, return_info=False
        )
        assert found(raw_ids, 8, 8)
        assert found(raw_ids, 24, 24)

        filtered_ids = localize.identify(
            movie,
            500,
            BOX,
            threaded=False,
            temporal_median_window=11,
            return_info=False,
        )
        assert not found(filtered_ids, 8, 8)
        assert found(filtered_ids, 24, 24)
        assert set(filtered_ids["frame"]) == set(blinking)

    def test_out_of_bounds_frames_are_not_read(self):
        """``identify_by_frame_number`` must not touch the movie for a
        frame it is going to skip - reading one would make a temporally
        filtered movie compute a whole window for nothing."""
        counting = _CountingMovie(self._movie(n_frames=10))
        ids = localize.identify_by_frame_number(
            counting, MIN_NG, BOX, 0, frame_bounds=(5, 8)
        )
        assert len(ids) == 0
        assert counting.reads == 0


@pytest.mark.gui
class TestTemporalMedianGui:
    """Wiring of the temporal median filter into Picasso: Localize.

    The risk this guards is stale state: every place that records or
    compares identification settings has to know about the new one, or
    identifications are silently reused (or never reused) after toggling
    the filter.
    """

    @staticmethod
    def _dialog():
        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

            def on_parameters_changed(self):
                pass

        return localize_gui.ParametersDialog(_StubWindow())

    def test_spinbox_follows_the_checkbox(self):
        dialog = self._dialog()
        try:
            assert not dialog.temporal_median_checkbox.isChecked()
            assert not dialog.temporal_median_spinbox.isEnabled()
            dialog.temporal_median_checkbox.setChecked(True)
            assert dialog.temporal_median_spinbox.isEnabled()
            dialog.temporal_median_checkbox.setChecked(False)
            assert not dialog.temporal_median_spinbox.isEnabled()
        finally:
            dialog.close()

    def test_parameters_reports_zero_when_unchecked(self):
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            assert window.parameters["Temporal Median Window"] == 0
            window.parameters_dialog.temporal_median_checkbox.setChecked(True)
            window.parameters_dialog.temporal_median_spinbox.setValue(33)
            assert window.parameters["Temporal Median Window"] == 33
        finally:
            window.parameters_dialog.close()

    def test_toggling_the_filter_invalidates_identifications(self):
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            window.view = type("_View", (), {"rois": []})()
            window.frame_range = None
            window.last_identification_info = {
                **window.parameters,
                "ROI": [],
                "Frame bounds": None,
            }
            assert not window.identifications_outdated()
            window.parameters_dialog.temporal_median_checkbox.setChecked(True)
            assert window.identifications_outdated()
        finally:
            window.parameters_dialog.close()

    def test_identification_movie_is_cached_and_invalidated(self):
        rng = np.random.default_rng(0)
        movie = rng.integers(0, 4000, size=(20, 8, 8), dtype=np.uint16)
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            window.view = type("_View", (), {"rois": []})()
            window.movie = movie
            window._temporal_movie = None

            # disabled: the raw movie is handed out untouched
            assert window.identification_movie() is movie

            window.parameters_dialog.temporal_median_checkbox.setChecked(True)
            window.parameters_dialog.temporal_median_spinbox.setValue(5)
            filtered = window.identification_movie()
            assert isinstance(filtered, localize.TemporalMedianMovie)
            assert filtered.raw is movie
            # cached across calls, so scrubbing frames does not rebuild it
            assert window.identification_movie() is filtered

            # changing the window rebuilds it
            window.parameters_dialog.temporal_median_spinbox.setValue(7)
            assert window.identification_movie() is not filtered

            # so does loading another movie, by identity
            other = movie.copy()
            window.movie = other
            assert window.identification_movie().raw is other
        finally:
            window.parameters_dialog.close()

    def test_bead_calibration_runs_ignore_the_filter(self, movie):
        """Beads are static, so a temporal median would subtract them."""
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            window.parameters_dialog.temporal_median_checkbox.setChecked(True)
            window.parameters_dialog.temporal_median_spinbox.setValue(21)
            window.movie = movie
            window.view = type("_View", (), {"rois": []})()
            window.frame_range = None

            def window_length(calibrate_z, calibrate_spline):
                worker = localize_gui.IdentificationWorker(
                    window,
                    fit_afterwards=False,
                    calibrate_z=calibrate_z,
                    calibrate_spline=calibrate_spline,
                )
                return worker.parameters["Temporal Median Window"]

            assert window_length(False, False) == 21
            assert window_length(True, False) == 0
            assert window_length(False, True) == 0
            # the window's own settings must not have been mutated
            assert window.parameters["Temporal Median Window"] == 21
        finally:
            window.parameters_dialog.close()

    def test_contrast_follows_the_filter(self, movie):
        """Filtered frames sit on a completely different intensity scale,
        so a contrast set for the raw camera counts must not survive the
        toggle - it would render the frame solid black."""
        window = gui_window = localize_gui.Window()
        try:
            window.movie = movie
            window.curr_frame_number = 40
            dialog = window.parameters_dialog
            contrast = window.contrast_dialog
            # start from a known state: the dialog restores the last-used
            # setting from the user's settings file
            dialog.temporal_median_checkbox.setChecked(False)
            dialog.temporal_median_spinbox.setValue(11)

            # black must be reachable: filtered frames are clipped at zero
            assert contrast.black_spinbox.minimum() == 0
            # white divides in _draw_frame, so it must stay positive
            assert contrast.white_spinbox.minimum() >= 1

            contrast.auto_checkbox.setChecked(False)
            contrast.change_contrast_silently(210, 300)

            def displayed_range():
                frame = window.identification_movie()[40]
                return float(frame.min()), float(frame.max())

            def contrast_range():
                return (
                    contrast.black_spinbox.value(),
                    contrast.white_spinbox.value(),
                )

            for checked in (True, False, True):
                dialog.temporal_median_checkbox.setChecked(checked)
                assert contrast_range() == displayed_range()

            # 'Auto' must re-read the displayed movie, not the raw one
            contrast.auto_checkbox.setChecked(True)
            assert contrast_range() == displayed_range()
        finally:
            gui_window.close()

    def test_channel_params_round_trip(self):
        window = localize_gui.Window.__new__(localize_gui.Window)
        dialog = self._dialog()
        window.parameters_dialog = dialog
        try:
            dialog.temporal_median_checkbox.setChecked(True)
            dialog.temporal_median_spinbox.setValue(77)
            params = window._capture_params()
            dialog.temporal_median_checkbox.setChecked(False)
            window._apply_params(params)
            assert dialog.temporal_median_checkbox.isChecked()
            assert dialog.temporal_median_spinbox.value() == 77
            # a parameter set from before this feature existed still loads
            legacy = {
                key: value
                for key, value in params.items()
                if not key.startswith("temporal_median")
            }
            window._apply_params(legacy)
            assert not dialog.temporal_median_checkbox.isChecked()
        finally:
            dialog.close()


# ---------------------------------------------------------------------------
# Gaussian filter (identification-only spatial smoothing)
# ---------------------------------------------------------------------------


def _double_lobed_movie(
    n_frames: int = 6,
    size: int = 32,
    separation: int = 4,
    psf_sigma: float = 1.0,
    amplitude: float = 3000.0,
    baseline: int = 100,
) -> np.ndarray:
    """Movie of one non-Gaussian "spot" per frame: two lobes ``separation``
    pixels apart in x, centered on the frame.

    ``_local_maxima`` suppresses maxima within ``+/- int(box / 2)`` of each
    other, so with ``BOX = 7`` the lobes have to be more than 3 px apart to
    both survive as maxima - which is exactly the situation this feature is
    meant to fix.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    frame = np.full((size, size), baseline, dtype=np.float64)
    for dx in (-separation / 2, separation / 2):
        frame += amplitude * np.exp(
            -((yy - center) ** 2 + (xx - (center + dx)) ** 2)
            / (2 * psf_sigma**2)
        )
    return np.tile(frame.astype(np.uint16), (n_frames, 1, 1))


class TestGaussianFilter:
    """``localize.GaussianFilteredMovie``, the kernel-radius arithmetic and
    their use from ``identify``."""

    @staticmethod
    def _movie(n_frames=12, height=8, width=6, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(
            0, 4000, size=(n_frames, height, width), dtype=np.uint16
        )

    def test_matches_scipy_reference(self):
        stack = self._movie()
        filtered = localize.GaussianFilteredMovie(stack, 1.5)
        for frame_number in range(len(stack)):
            expected = ndimage.gaussian_filter(
                stack[frame_number].astype(np.float32),
                1.5,
                mode=localize.GAUSSIAN_FILTER_MODE,
                truncate=localize.GAUSSIAN_FILTER_TRUNCATE,
            )
            np.testing.assert_allclose(
                filtered[frame_number], expected, atol=1e-5
            )

    def test_output_is_float32_not_the_input_dtype(self):
        """``gaussian_filter`` keeps the input dtype unless told otherwise,
        so a uint16 movie would silently come back rounded to integers."""
        filtered = localize.GaussianFilteredMovie(self._movie(), 1.0)
        frame = filtered[0]
        assert frame.dtype == np.float32
        assert filtered.dtype == np.float32
        assert np.any(frame % 1 != 0)

    def test_movie_protocol(self):
        stack = self._movie()
        filtered = localize.GaussianFilteredMovie(stack, 1.0)
        assert len(filtered) == len(stack)
        assert filtered.shape == stack.shape
        assert filtered[-1].shape == stack.shape[1:]
        np.testing.assert_array_equal(filtered[-1], filtered[len(stack) - 1])
        np.testing.assert_array_equal(
            filtered[2:5], np.stack([filtered[i] for i in (2, 3, 4)])
        )
        np.testing.assert_array_equal(
            np.stack(list(iter(filtered))),
            np.stack([filtered[i] for i in range(len(stack))]),
        )
        np.testing.assert_array_equal(filtered[3, 2], filtered[3][2])
        with pytest.raises(IndexError):
            filtered[len(stack)]

    @pytest.mark.parametrize("sigma", [0, -1.0])
    def test_rejects_non_positive_sigma(self, sigma):
        with pytest.raises(ValueError):
            localize.GaussianFilteredMovie(self._movie(), sigma)

    def test_rejects_empty_movie(self):
        with pytest.raises(ValueError):
            localize.GaussianFilteredMovie(
                np.empty((0, 4, 4), dtype=np.uint16), 1.0
            )

    def test_read_lock_follows_the_source(self):
        """A source that is not safe for concurrent reads must be
        serialized; one that is (or a memmap) must not be."""
        plain = localize.GaussianFilteredMovie(
            _CountingMovie(self._movie()), 1.0
        )
        assert plain._read_lock is not None
        concurrent = localize.GaussianFilteredMovie(
            localize.TemporalMedianMovie(self._movie(), 5), 1.0
        )
        assert concurrent._read_lock is None
        assert localize.GaussianFilteredMovie.supports_concurrent_reads

    def test_each_frame_is_read_once(self):
        stack = self._movie(n_frames=10)
        counting = _CountingMovie(stack)
        filtered = localize.GaussianFilteredMovie(counting, 1.0)
        counting.reads = 0  # ignore the geometry probe read in __init__
        for frame_number in range(len(stack)):
            filtered[frame_number]
        assert counting.reads == len(stack)

    def test_threaded_matches_serial(self, movie):
        as_set = lambda ids: set(  # noqa: E731
            map(tuple, ids[["frame", "y", "x"]].to_numpy())
        )
        common = dict(gaussian_filter_sigma=1.0, return_info=False)
        ids_t = localize.identify(movie, MIN_NG, BOX, threaded=True, **common)
        ids_s = localize.identify(movie, MIN_NG, BOX, threaded=False, **common)
        assert as_set(ids_t) == as_set(ids_s)

    def test_identify_argument_matches_explicit_wrapper(self, movie):
        ids_arg, info = localize.identify(
            movie, MIN_NG, BOX, threaded=False, gaussian_filter_sigma=1.0
        )
        ids_wrapped = localize.identify(
            localize.GaussianFilteredMovie(movie, 1.0),
            MIN_NG,
            BOX,
            threaded=False,
            return_info=False,
        )
        pd.testing.assert_frame_equal(ids_arg, ids_wrapped)
        assert info["Gaussian Filter Sigma"] == 1.0

    def test_info_records_zero_when_disabled(self, movie):
        _, info = localize.identify(movie, MIN_NG, BOX, threaded=False)
        assert info["Gaussian Filter Sigma"] == 0.0

    def test_fit2d_rejects_the_filtered_view(self, movie, movie_info):
        """The class deliberately does not implement the movie interface,
        so a filtered view reaching the fit is a loud error rather than
        silently wrong photon numbers."""
        ids = localize.identify(
            movie, MIN_NG, BOX, threaded=False, return_info=False
        )
        with pytest.raises(AssertionError):
            localize.fit2D(
                movie=localize.GaussianFilteredMovie(movie, 1.0),
                movie_info=movie_info,
                camera_info=CAMERA_INFO,
                identifications=ids,
                box=BOX,
                multiprocess=False,
            )

    # -- kernel radius / ROI padding ---------------------------------------

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.0, 3.7])
    def test_radius_matches_the_kernel_scipy_actually_uses(self, sigma):
        """The ROI padding is derived from this number, so it has to be
        the exact support of scipy's kernel, not an estimate."""
        impulse = np.zeros(201)
        impulse[100] = 1.0
        smoothed = ndimage.gaussian_filter(
            impulse,
            sigma,
            mode="constant",
            truncate=localize.GAUSSIAN_FILTER_TRUNCATE,
        )
        support = np.nonzero(smoothed)[0]
        assert 100 - support[0] == localize.gaussian_filter_radius(sigma)
        assert support[-1] - 100 == localize.gaussian_filter_radius(sigma)

    def test_radius_is_zero_without_a_filter(self):
        assert localize.gaussian_filter_radius(0) == 0
        assert localize.gaussian_filter_radius(None) == 0

    def test_identification_roi_pad(self):
        # int(box / 2) + 1, the reach of identify_in_frame
        assert localize.identification_roi_pad(BOX) == 4
        assert localize.identification_roi_pad(BOX, 0) == 4
        # ... plus the kernel radius, int(4 * sigma + 0.5)
        assert localize.identification_roi_pad(BOX, 1.0) == 8
        assert localize.identification_roi_pad(BOX, 1.5) == 10

    def test_roi_border_is_unaffected_by_the_temporal_median_zero_fill(self):
        """A ``TemporalMedianMovie`` zeroes everything outside its padded
        bounding box. Unless that padding accounts for the Gaussian's
        kernel radius, those zeros are smeared into the pixels the
        gradients of a spot on the ROI border are computed from.
        """
        n_frames, size, sigma = 12, 64, 1.5
        yy, xx = np.mgrid[0:size, 0:size]
        rng = np.random.default_rng(5)
        movie = rng.integers(90, 110, size=(n_frames, size, size)).astype(
            np.uint16
        )
        # a blinking spot sitting exactly on the ROI's right border
        roi = ((20, 20), (32, 32))
        for frame_number in (2, 3, 8):
            movie[frame_number] += (
                4000
                * np.exp(-((yy - 26) ** 2 + (xx - 31) ** 2) / (2 * 1.2**2))
            ).astype(np.uint16)

        def gradients(**kwargs):
            ids = localize.identify(
                movie,
                200,
                BOX,
                threaded=False,
                temporal_median_window=5,
                gaussian_filter_sigma=sigma,
                return_info=False,
                **kwargs,
            )
            return ids[ids["x"].between(29, 32)]["net_gradient"].to_numpy()

        whole_frame = gradients()
        with_roi = gradients(roi=[roi])
        assert len(with_roi) and len(with_roi) == len(whole_frame)
        np.testing.assert_allclose(with_roi, whole_frame, rtol=1e-4)

        # ... and the test really does detect the bug it guards: with the
        # pad the identification alone would need, the border gradients
        # come out different
        too_small = localize.GaussianFilteredMovie(
            localize.TemporalMedianMovie(
                movie, 5, roi=[roi], roi_pad=int(BOX / 2) + 1
            ),
            sigma,
        )
        ids = localize.identify(
            too_small, 200, BOX, roi=[roi], threaded=False, return_info=False
        )
        starved = ids[ids["x"].between(29, 32)]["net_gradient"].to_numpy()
        assert not np.allclose(starved, whole_frame, rtol=1e-4)

    # -- what the feature is for -------------------------------------------

    def test_double_peaked_spot_collapses_to_one_identification(self):
        """The motivating case: a spot that is not Gaussian-shaped breaks
        into two local maxima and is identified twice; smoothing merges
        them into a single identification at the true center."""
        movie = _double_lobed_movie()
        center = movie.shape[-1] // 2

        raw_ids = localize.identify(
            movie, 500, BOX, threaded=False, return_info=False
        )
        assert len(raw_ids) == 2 * len(movie)

        # a lower threshold, since smoothing lowers gradient magnitudes -
        # that re-tuning is exactly what the tooltip and docs warn about
        smoothed_ids = localize.identify(
            movie,
            50,
            BOX,
            threaded=False,
            gaussian_filter_sigma=2.0,
            return_info=False,
        )
        assert len(smoothed_ids) == len(movie)
        assert (smoothed_ids["x"] - center).abs().max() <= 1
        assert (smoothed_ids["y"] - center).abs().max() <= 1


@pytest.mark.gui
class TestGaussianFilterGui:
    """Wiring of the Gaussian filter into Picasso: Localize.

    As for the temporal median filter, the risk is stale state: every
    place that records or compares identification settings has to know
    about the sigma, or identifications are silently reused after it
    changes.
    """

    _dialog = staticmethod(TestTemporalMedianGui._dialog)

    def test_parameters_reports_the_sigma(self):
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            assert window.parameters["Gaussian Filter Sigma"] == 0.0
            window.parameters_dialog.gaussian_filter_spinbox.setValue(1.5)
            assert window.parameters["Gaussian Filter Sigma"] == 1.5
        finally:
            window.parameters_dialog.close()

    def test_changing_the_sigma_invalidates_identifications(self):
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            window.view = type("_View", (), {"rois": []})()
            window.frame_range = None
            window.last_identification_info = {
                **window.parameters,
                "ROI": [],
                "Frame bounds": None,
            }
            assert not window.identifications_outdated()
            window.parameters_dialog.gaussian_filter_spinbox.setValue(1.5)
            assert window.identifications_outdated()
        finally:
            window.parameters_dialog.close()

    def test_identification_movie_composes_and_caches(self):
        rng = np.random.default_rng(0)
        movie = rng.integers(0, 4000, size=(20, 8, 8), dtype=np.uint16)
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        dialog = window.parameters_dialog
        try:
            window.view = type("_View", (), {"rois": []})()
            window.movie = movie
            window._temporal_movie = None
            window._gaussian_movie = None

            # disabled: the raw movie is handed out untouched
            assert window.identification_movie() is movie

            dialog.gaussian_filter_spinbox.setValue(1.0)
            smoothed = window.identification_movie()
            assert isinstance(smoothed, localize.GaussianFilteredMovie)
            assert smoothed.raw is movie
            # cached, so scrubbing frames does not rebuild it
            assert window.identification_movie() is smoothed

            # both filters compose, median first
            dialog.temporal_median_checkbox.setChecked(True)
            dialog.temporal_median_spinbox.setValue(5)
            composed = window.identification_movie()
            assert isinstance(composed, localize.GaussianFilteredMovie)
            assert isinstance(composed.raw, localize.TemporalMedianMovie)
            assert composed.raw.raw is movie
            assert window.identification_movie() is composed

            # changing sigma rebuilds only the smoothing stage
            median_stage = composed.raw
            dialog.gaussian_filter_spinbox.setValue(2.0)
            resmoothed = window.identification_movie()
            assert resmoothed is not composed
            assert resmoothed.raw is median_stage

            # changing the window rebuilds both, via the identity chain
            dialog.temporal_median_spinbox.setValue(7)
            rebuilt = window.identification_movie()
            assert rebuilt is not resmoothed
            assert rebuilt.raw is not median_stage

            # sigma back to 0 drops the smoothing stage entirely
            dialog.gaussian_filter_spinbox.setValue(0.0)
            median_only = window.identification_movie()
            assert isinstance(median_only, localize.TemporalMedianMovie)
            assert window._gaussian_movie is None

            # ... and so does unloading the movie
            window.movie = None
            assert window.identification_movie() is None
            assert window._temporal_movie is None
        finally:
            dialog.close()

    def test_the_preview_filters_are_never_restricted_to_the_rois(self):
        """The preview movie is what the window displays, so its filters
        have to cover the whole frame: a ROI-restricted temporal median
        leaves everything outside its bounding box at zero, and with the
        filter on, drawing the first ROI would black out the rest of the
        movie - the next one could not be placed. The detections must
        still match ``localize.identify``, which does restrict it.
        """
        n_frames, size = 12, 64
        yy, xx = np.mgrid[0:size, 0:size]
        rng = np.random.default_rng(5)
        movie = rng.integers(90, 110, size=(n_frames, size, size)).astype(
            np.uint16
        )
        # blinking, or the temporal median would subtract them away
        for frame_number in (2, 3, 8):
            for cy, cx in ((26, 26), (48, 48)):  # inside and outside the ROI
                movie[frame_number] += (
                    4000
                    * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.2**2))
                ).astype(np.uint16)
        rois = [[[20, 20], [32, 32]]]
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        dialog = window.parameters_dialog
        try:
            window.view = type("_View", (), {"rois": rois})()
            window.movie = movie
            window._temporal_movie = None
            window._gaussian_movie = None
            window.frame_range = None
            dialog.temporal_median_checkbox.setChecked(True)
            dialog.temporal_median_spinbox.setValue(5)
            for sigma in (0.0, 1.5):
                dialog.gaussian_filter_spinbox.setValue(sigma)
                filtered = window.identification_movie()
                median = (
                    filtered.raw
                    if isinstance(filtered, localize.GaussianFilteredMovie)
                    else filtered
                )
                # the whole frame stays filtered, and stays visible
                assert median.roi is None
                assert filtered[3][40:, 40:].max() > 0
                # ... and identification still agrees with the batch run,
                # which restricts the median to the ROI
                preview = localize.identify_by_frame_number(
                    filtered, 200, BOX, 3, roi=rois
                )
                batch = localize.identify(
                    movie,
                    200,
                    BOX,
                    roi=rois,
                    threaded=False,
                    temporal_median_window=5,
                    gaussian_filter_sigma=sigma,
                    return_info=False,
                )
                batch = batch[batch["frame"] == 3]
                assert len(preview) == len(batch) == 1
                np.testing.assert_allclose(
                    preview["net_gradient"].to_numpy(),
                    batch["net_gradient"].to_numpy(),
                    rtol=1e-4,
                )

            # the ROIs no longer key the median, so drawing one (or nudging
            # the sigma) must not throw away the expensive cached medians
            median_stage = window.identification_movie().raw
            window.view.rois = rois + [[[40, 40], [56, 56]]]
            dialog.gaussian_filter_spinbox.setValue(2.0)
            assert window.identification_movie().raw is median_stage
        finally:
            dialog.close()

    def test_bead_calibration_runs_keep_the_filter(self, movie):
        """Deliberately unlike the temporal median: smoothing does not
        erase static beads, and defocused beads are exactly the
        multi-peaked PSFs the filter helps with."""
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = self._dialog()
        try:
            window.parameters_dialog.gaussian_filter_spinbox.setValue(1.5)
            window.movie = movie
            window.view = type("_View", (), {"rois": []})()
            window.frame_range = None

            def sigma(calibrate_z, calibrate_spline):
                worker = localize_gui.IdentificationWorker(
                    window,
                    fit_afterwards=False,
                    calibrate_z=calibrate_z,
                    calibrate_spline=calibrate_spline,
                )
                return worker.parameters["Gaussian Filter Sigma"]

            assert sigma(False, False) == 1.5
            assert sigma(True, False) == 1.5
            assert sigma(False, True) == 1.5
        finally:
            window.parameters_dialog.close()

    def test_contrast_follows_the_filter(self, movie):
        """Smoothing lowers the peaks, so a contrast set for the raw
        camera counts must not survive a change of sigma."""
        window = localize_gui.Window()
        try:
            window.movie = movie
            window.curr_frame_number = 40
            dialog = window.parameters_dialog
            contrast = window.contrast_dialog
            # start from a known state: the dialog restores the last-used
            # setting from the user's settings file
            dialog.temporal_median_checkbox.setChecked(False)
            dialog.gaussian_filter_spinbox.setValue(0.0)
            contrast.auto_checkbox.setChecked(False)
            contrast.change_contrast_silently(210, 300)

            def displayed_range():
                frame = window.identification_movie()[40]
                return float(frame.min()), float(frame.max())

            def contrast_range():
                return (
                    contrast.black_spinbox.value(),
                    contrast.white_spinbox.value(),
                )

            for sigma in (2.0, 0.0, 1.0):
                dialog.gaussian_filter_spinbox.setValue(sigma)
                # the spin boxes carry no decimals, so the smoothed
                # frame's float range lands on them rounded
                assert contrast_range() == pytest.approx(
                    displayed_range(), abs=1
                )
        finally:
            window.close()

    def test_channel_params_round_trip(self):
        window = localize_gui.Window.__new__(localize_gui.Window)
        dialog = self._dialog()
        window.parameters_dialog = dialog
        try:
            dialog.gaussian_filter_spinbox.setValue(2.5)
            params = window._capture_params()
            dialog.gaussian_filter_spinbox.setValue(0.0)
            window._apply_params(params)
            assert dialog.gaussian_filter_spinbox.value() == 2.5
            # a parameter set from before this feature existed still loads
            legacy = {
                key: value
                for key, value in params.items()
                if key != "gaussian_filter_sigma"
            }
            window._apply_params(legacy)
            assert dialog.gaussian_filter_spinbox.value() == (
                localize_gui.DEFAULT_PARAMETERS["Gaussian Filter Sigma"]
            )
        finally:
            dialog.close()

    def test_identifications_metadata_round_trip(self, movie, tmp_path):
        """Saving and reloading identifications must restore the sigma,
        otherwise the loaded ones immediately count as outdated."""
        # a real window: load_identifications redraws the frame
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        try:
            window.movie = movie
            window.curr_frame_number = 0
            window.info = []
            dialog.temporal_median_checkbox.setChecked(False)
            dialog.gaussian_filter_spinbox.setValue(1.5)
            window.identifications = localize.identify(
                movie,
                MIN_NG,
                BOX,
                threaded=False,
                gaussian_filter_sigma=1.5,
                return_info=False,
            )
            path = str(tmp_path / "ids.hdf5")
            window.save_identifications(path)

            dialog.gaussian_filter_spinbox.setValue(0.0)
            window.load_identifications(path)
            assert dialog.gaussian_filter_spinbox.value() == 1.5
            assert not window.identifications_outdated()
        finally:
            window.close()


class TestRoiSnapshotsGui:
    """ROIs are edited in place all over Picasso: Localize - a region is
    moved by rewriting its corners, a double click deletes one from the
    list. Anything that snapshots them (the identification worker, the
    settings the identifications were made with) therefore has to copy
    them, or the snapshot changes underneath it and keeps agreeing with
    the edited ROIs.
    """

    _dialog = staticmethod(TestTemporalMedianGui._dialog)

    @staticmethod
    def _window(dialog, rois):
        window = localize_gui.Window.__new__(localize_gui.Window)
        window.parameters_dialog = dialog
        window.view = type("_View", (), {"rois": rois})()
        window.movie = np.zeros((4, 8, 8), np.uint16)
        window.frame_range = None
        return window

    def test_identification_rois_are_copied_out_of_the_view(self):
        dialog = self._dialog()
        rois = [[[0, 0], [8, 8]]]
        try:
            window = self._window(dialog, rois)
            copied = window.identification_rois()
            assert copied == rois
            copied[0][0][0] = 4
            copied.append([[10, 10], [20, 20]])
            assert rois == [[[0, 0], [8, 8]]]
        finally:
            dialog.close()

    def test_moving_a_region_invalidates_identifications(self):
        dialog = self._dialog()
        rois = [[[0, 0], [8, 8]]]
        try:
            window = self._window(dialog, rois)
            window.last_identification_info = {
                **window.parameters,
                "ROI": window.identification_rois(),
                "Frame bounds": None,
            }
            assert not window.identifications_outdated()
            # as ``View._move_region`` moves a split-FOV region
            rois[0] = [[2, 2], [10, 10]]
            assert window.identifications_outdated()
        finally:
            dialog.close()

    def test_worker_rois_do_not_follow_the_view(self):
        dialog = self._dialog()
        rois = [[[0, 0], [8, 8]]]
        try:
            window = self._window(dialog, rois)
            worker = localize_gui.IdentificationWorker(window, False, False)
            rois[0][1][0] = 16
            rois.append([[10, 10], [20, 20]])
            assert worker.rois == [[[0, 0], [8, 8]]]
        finally:
            dialog.close()


def _two_region_frame() -> np.ndarray:
    """One frame with a bright spot in the left half and a ten-times dimmer
    one in the right half - a split-FOV pair that no single threshold can
    detect at once without also detecting the noise floor of the bright
    region."""
    frame = np.full((64, 128), 100.0, np.float32)
    yy, xx = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-(yy**2 + xx**2) / 2.0)
    frame[28:37, 26:35] += 2000.0 * kernel
    frame[28:37, 90:99] += 200.0 * kernel
    return frame


TWO_REGIONS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]


class TestPerRoiMinNetGradient:
    """``identify_in_frame`` with one threshold per ROI (split-FOV: the
    regions are separate channels and need not share a brightness scale)."""

    def test_scalar_applies_to_every_roi(self):
        frame = _two_region_frame()
        _, x, _ = localize.identify_in_frame(frame, 500, 7, TWO_REGIONS)
        assert sorted(x.tolist()) == [30, 94]
        _, x, _ = localize.identify_in_frame(frame, 5000, 7, TWO_REGIONS)
        assert x.tolist() == [30]  # the dim region drops out

    def test_each_roi_uses_its_own_threshold(self):
        frame = _two_region_frame()
        _, x, _ = localize.identify_in_frame(
            frame, [5000, 500], 7, TWO_REGIONS
        )
        assert sorted(x.tolist()) == [30, 94]
        # swapping the thresholds keeps only the bright spot: the dim one is
        # now below its region's threshold
        _, x, _ = localize.identify_in_frame(
            frame, [500, 5000], 7, TWO_REGIONS
        )
        assert x.tolist() == [30]

    def test_length_must_match_the_rois(self):
        frame = _two_region_frame()
        with pytest.raises(ValueError, match="one threshold per ROI"):
            localize.identify_in_frame(frame, [1, 2, 3], 7, TWO_REGIONS)

    def test_single_element_sequence_is_shared(self):
        frame = _two_region_frame()
        _, x, _ = localize.identify_in_frame(frame, [500], 7, TWO_REGIONS)
        assert sorted(x.tolist()) == [30, 94]

    def test_identify_carries_the_thresholds_into_the_metadata(self):
        movie = np.stack([_two_region_frame()] * 3)
        ids, info = localize.identify(
            movie, [5000, 500], 7, roi=TWO_REGIONS, threaded=False
        )
        assert info["Min. Net Gradient"] == [5000, 500]
        assert sorted(set(ids["x"])) == [30, 94]


@pytest.mark.gui
class TestPerRegionMinNetGradientGui:
    """The split-FOV region <-> min. net gradient slider binding.

    The slider is the only place these are tuned interactively, so it has
    to follow the selected region without the selection itself counting as
    a parameter change (which would throw away the localizations).
    """

    @staticmethod
    def _window():
        """A stub window carrying the real ``region_mngs`` / ``parameters``
        over a minimal view, plus the dialog under test."""

        class _View:
            def __init__(self):
                self.rois = []
                self.roi_mngs = []
                self.selected_roi = None
                self.split_fov_mode = False

        class _StubWindow(QtWidgets.QMainWindow):
            movie = None
            region_mngs = localize_gui.Window.region_mngs
            parameters = localize_gui.Window.parameters
            identify_mode = localize_gui.Window.identify_mode

            def __init__(self):
                super().__init__()
                self.view = _View()
                self.draws = 0
                self.invalidations = 0

            def draw_frame(self):
                self.draws += 1

            def on_parameters_changed(self):
                self.invalidations += 1

        window = _StubWindow()
        window.parameters_dialog = localize_gui.ParametersDialog(window)
        return window

    def test_no_thresholds_without_split_fov(self):
        window = self._window()
        try:
            window.view.rois = [r[:] for r in TWO_REGIONS]
            assert window.region_mngs() == []
            assert isinstance(window.parameters["Min. Net Gradient"], int)
        finally:
            window.parameters_dialog.close()
            window.close()

    def test_regions_inherit_the_slider_value(self):
        window = self._window()
        try:
            window.parameters_dialog.mng_slider.setValue(4000)
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            assert window.region_mngs() == [4000, 4000]
            assert window.parameters["Min. Net Gradient"] == [4000, 4000]
        finally:
            window.parameters_dialog.close()
            window.close()

    def test_thresholds_follow_added_and_removed_regions(self):
        window = self._window()
        try:
            window.parameters_dialog.mng_slider.setValue(4000)
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.region_mngs()
            window.view.roi_mngs = [1000, 2000]
            window.view.rois.append([[0, 0], [64, 64]])
            assert window.region_mngs() == [1000, 2000, 4000]
            del window.view.rois[2]
            assert window.region_mngs() == [1000, 2000]
        finally:
            window.parameters_dialog.close()
            window.close()

    def test_slider_edits_only_the_selected_region(self):
        window = self._window()
        dialog = window.parameters_dialog
        try:
            dialog.mng_slider.setValue(4000)
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.view.selected_roi = 1
            dialog.mng_slider.setValue(800)
            assert window.region_mngs() == [4000, 800]
        finally:
            dialog.close()
            window.close()

    def test_slider_sets_every_region_when_none_is_selected(self):
        window = self._window()
        dialog = window.parameters_dialog
        try:
            dialog.mng_slider.setValue(4000)
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.view.roi_mngs = [1000, 2000]
            window.view.selected_roi = None
            dialog.mng_slider.setValue(777)
            assert window.region_mngs() == [777, 777]
        finally:
            dialog.close()
            window.close()

    def test_selecting_a_region_shows_its_threshold(self):
        window = self._window()
        dialog = window.parameters_dialog
        try:
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.view.roi_mngs = [4000, 800]
            window.view.selected_roi = 1
            dialog.sync_mng_to_selected_region()
            assert dialog.mng_slider.value() == 800
            assert dialog.mng_spinbox.value() == 800
            # the other region is untouched by merely looking at this one
            assert window.region_mngs() == [4000, 800]
        finally:
            dialog.close()
            window.close()

    def test_selecting_a_region_does_not_invalidate_the_locs(self):
        """Switching regions moves the slider, but nothing about the
        identification changed - the existing localizations must survive."""
        window = self._window()
        dialog = window.parameters_dialog
        try:
            dialog.preview_checkbox.setChecked(True)
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.view.roi_mngs = [4000, 800]
            window.invalidations = 0
            window.view.selected_roi = 1
            dialog.sync_mng_to_selected_region()
            assert dialog.mng_slider.value() == 800
            assert window.invalidations == 0
            # an actual edit still does invalidate them
            dialog.mng_slider.setValue(900)
            assert window.invalidations == 1
        finally:
            dialog.close()
            window.close()

    def test_identifications_metadata_round_trip(self, tmp_path):
        """The per-region thresholds have to reach the metadata and come
        back onto the regions, or reloaded split-FOV identifications count
        as outdated straight away."""
        movie = np.stack([_two_region_frame()] * 3).astype(np.uint16)
        window = localize_gui.Window()
        try:
            window.movie = movie
            window.curr_frame_number = 0
            window.info = []
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.set_split_fov_mode(True)
            window.view.roi_mngs = [5000, 500]
            assert window.parameters["Min. Net Gradient"] == [5000, 500]
            window.identifications = localize.identify(
                movie,
                [5000, 500],
                BOX,
                roi=TWO_REGIONS,
                threaded=False,
                return_info=False,
            )
            path = str(tmp_path / "ids.hdf5")
            window.save_identifications(path)

            window.view.roi_mngs = [1, 1]
            window.load_identifications(path)
            assert window.view.roi_mngs == [5000, 500]
            assert window.parameters["Min. Net Gradient"] == [5000, 500]
        finally:
            window.close()

    def test_a_region_threshold_outside_the_slider_range_widens_it(self):
        window = self._window()
        dialog = window.parameters_dialog
        try:
            dialog.mng_max_spinbox.setValue(10000)
            window.view.split_fov_mode = True
            window.view.rois = [r[:] for r in TWO_REGIONS]
            window.view.roi_mngs = [4000, 25000]
            window.view.selected_roi = 1
            dialog.sync_mng_to_selected_region()
            assert dialog.mng_slider.value() == 25000
            assert dialog.mng_slider.maximum() >= 25000
            assert window.region_mngs() == [4000, 25000]
        finally:
            dialog.close()
            window.close()


# ---------------------------------------------------------------------------
# Affine calibration: target -> reference (astigmatism / chromatic)
# ---------------------------------------------------------------------------


def _affine_bead_image(
    positions: np.ndarray,
    shape: tuple[int, int] = (256, 256),
    sigma: float = 1.3,
    amplitude: float = 6000.0,
    baseline: float = 100.0,
) -> np.ndarray:
    """One-frame bead movie with Gaussian beads at ``positions`` (``(n, 2)``
    in ``[x, y]``), as ``calibrate_lateral_transform`` expects its inputs."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = np.full(shape, baseline, dtype=np.float64)
    for x, y in positions:
        img += amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
        )
    return img[np.newaxis].astype(np.uint16)


def _affine_bead_grid(
    n: int = 5, spacing: float = 48.0, offset: float = 32.0
) -> np.ndarray:
    """``(n*n, 2)`` grid of bead positions in ``[x, y]``, jittered off the
    pixel grid so the sub-pixel refinement is actually exercised."""
    rng = np.random.RandomState(0)
    grid = np.array(
        [
            (offset + i * spacing, offset + j * spacing)
            for i in range(n)
            for j in range(n)
        ],
        dtype=np.float64,
    )
    return grid + rng.uniform(-0.4, 0.4, size=grid.shape)


def _apply_homogeneous(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous (x, y) transform to ``(n, 2)`` points."""
    return xy @ np.asarray(matrix)[:2, :2].T + np.asarray(matrix)[:2, 2]


class TestLateralTransformMath:
    """The lateral estimator and the decomposition on synthetic point sets.

    The decomposition itself is covered per model in
    ``tests/test_transforms.py``; these pin the ``[row, col]`` convention the
    bead pipeline feeds it and the outlier trim.
    """

    def test_recovers_known_transform(self):
        rng = np.random.RandomState(1)
        src_xy = rng.uniform(10, 200, size=(30, 2))
        truth = np.array(
            [[1.004, -0.011, 3.5], [0.008, 0.997, -2.25], [0.0, 0.0, 1.0]]
        )
        dst_xy = _apply_homogeneous(truth, src_xy)
        # the estimator takes [row, col] = [y, x] correspondences
        transform, keep = localize._estimate_lateral_transform(
            src_xy[:, ::-1], dst_xy[:, ::-1]
        )
        assert keep.all()
        assert np.allclose(affine_matrix_3x3(transform), truth, atol=1e-8)

    def test_too_few_pairs_raise(self):
        src = np.array([[10.0, 20.0], [40.0, 60.0]])  # [row, col]
        with pytest.raises(ValueError, match="at least 3"):
            localize._estimate_lateral_transform(src, src + 1.0)

    def test_decomposition_recovers_rotation_and_scale(self):
        angle = np.radians(1.5)
        scale_x, scale_y = 1.01, 0.98
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        matrix = np.eye(3)
        matrix[:2, :2] = rotation @ np.diag([scale_x, scale_y])
        matrix[:2, 2] = [4.0, -6.0]
        decomposition = affine(matrix).decompose(pixelsize=130)
        assert decomposition["rotation_deg"] == pytest.approx(1.5, abs=1e-6)
        assert decomposition["scale_x"] == pytest.approx(scale_x, abs=1e-9)
        assert decomposition["scale_y"] == pytest.approx(scale_y, abs=1e-9)
        assert decomposition["shear_deg"] == pytest.approx(0.0, abs=1e-6)
        assert decomposition["tx_px"] == pytest.approx(4.0)
        assert decomposition["tx_nm"] == pytest.approx(4.0 * 130)
        assert decomposition["ty_nm"] == pytest.approx(-6.0 * 130)

    def test_decomposition_omits_nm_without_pixelsize(self):
        decomposition = transforms.identity().decompose()
        assert "tx_nm" not in decomposition and "ty_nm" not in decomposition


class TestFitAffineTransform:
    """End-to-end calibration on synthetic bead images (no plotting)."""

    TRUTH = np.array(
        [[1.003, -0.009, 4.0], [0.007, 0.998, -2.5], [0.0, 0.0, 1.0]]
    )

    @pytest.fixture(scope="class")
    def bead_movies(self):
        """Reference and cylindrical bead movies related by ``TRUTH``
        (which maps cylindrical -> reference, the direction that is fit)."""
        ref_xy = _affine_bead_grid()
        inverse = np.linalg.inv(self.TRUTH)
        cyl_xy = _apply_homogeneous(inverse, ref_xy)
        return _affine_bead_image(ref_xy), _affine_bead_image(cyl_xy), ref_xy

    @pytest.mark.parametrize(
        "model", [m for m in transforms.MODELS if m != "translation"]
    )
    def test_every_model_recovers_the_applied_transform(
        self, bead_movies, model
    ):
        """The correction is a pure affine here, which every model except the
        translation can represent - so each must recover it, and record which
        one it is. The translation is covered on a pure shift below."""
        movie_ref, movie_cyl, ref_xy = bead_movies
        calibration, _ = localize.fit_lateral_transform(
            movie_ref,
            movie_cyl,
            {},
            box=BOX,
            minimum_ng=1000,
            model=model,
        )
        (entry,) = lib.lateral_transforms(calibration)
        assert entry["Transform"]["model"] == model
        cyl_xy = _apply_homogeneous(np.linalg.inv(self.TRUTH), ref_xy)
        mapped = transforms.from_dict(entry["Transform"]).apply(cyl_xy)
        assert np.allclose(mapped, ref_xy, atol=0.3)

    def test_a_translation_recovers_a_pure_shift(self):
        """The 2-DOF model on the only correction it claims to fit. It is
        given its own bead pair because the class-wide ``TRUTH`` rotates and
        scales, which a translation must *not* absorb."""
        shift = np.array([3.5, -2.25])
        ref_xy = _affine_bead_grid()
        movie_ref = _affine_bead_image(ref_xy)
        movie_shifted = _affine_bead_image(ref_xy - shift)
        calibration, _ = localize.fit_lateral_transform(
            movie_ref,
            movie_shifted,
            {},
            box=BOX,
            minimum_ng=1000,
            model="translation",
        )
        (entry,) = lib.lateral_transforms(calibration)
        assert entry["Transform"]["model"] == "translation"
        transform = transforms.from_dict(entry["Transform"])
        assert np.allclose(transform.shift, shift, atol=0.1)
        assert np.allclose(transform.apply(ref_xy - shift), ref_xy, atol=0.3)

    def test_too_few_pairs_for_the_model_raises(self):
        """Enough beads for an affine, too few for a degree-3 polynomial: the
        error must name the model's own requirement, not a fixed 3."""
        ref_xy = _affine_bead_grid(n=2)  # 4 beads
        cyl_xy = _apply_homogeneous(np.linalg.inv(self.TRUTH), ref_xy)
        movie_ref = _affine_bead_image(ref_xy)
        movie_cyl = _affine_bead_image(cyl_xy)
        # an affine (needs 3) is fine on the same data
        localize.fit_lateral_transform(
            movie_ref, movie_cyl, {}, box=BOX, minimum_ng=1000
        )
        with pytest.raises(ValueError, match="polynomial3"):
            localize.fit_lateral_transform(
                movie_ref,
                movie_cyl,
                {},
                box=BOX,
                minimum_ng=1000,
                model="polynomial3",
            )

    def test_a_mismatched_pair_is_trimmed(self):
        """Mutual-nearest-neighbor matching has no outlier rejection, so the
        fit must reject the bad pair itself - otherwise one mismatched bead
        visibly warps a projective and wrecks a polynomial."""
        ref_xy = _affine_bead_grid()
        cyl_xy = _apply_homogeneous(np.linalg.inv(self.TRUTH), ref_xy)
        clean, _ = localize._estimate_lateral_transform(
            cyl_xy[:, ::-1], ref_xy[:, ::-1], "affine"
        )
        # drag one target bead far off its true counterpart
        spoiled = ref_xy.copy()
        spoiled[0] += (25.0, -18.0)
        trimmed, keep = localize._estimate_lateral_transform(
            cyl_xy[:, ::-1], spoiled[:, ::-1], "affine"
        )
        assert not keep[0] and keep.sum() == len(ref_xy) - 1
        assert np.allclose(
            trimmed.apply(cyl_xy), clean.apply(cyl_xy), atol=1e-6
        )

    def test_recovers_the_applied_transform(self, bead_movies):
        movie_ref, movie_cyl, ref_xy = bead_movies
        calibration, qc = localize.fit_lateral_transform(
            movie_ref, movie_cyl, {}, box=BOX, minimum_ng=1000
        )
        (entry,) = lib.lateral_transforms(calibration)
        matrix = affine_matrix_3x3(entry["Transform"])
        # compare where it sends points, not the coefficients: a small
        # coefficient error far from the origin is what actually matters
        cyl_xy = _apply_homogeneous(np.linalg.inv(self.TRUTH), ref_xy)
        mapped = _apply_homogeneous(matrix, cyl_xy)
        assert np.allclose(mapped, ref_xy, atol=0.2)
        assert entry["Bead pairs"] == len(ref_xy)
        assert qc["n_pairs"] == len(ref_xy)

    def test_calibration_entry_contents(self, bead_movies):
        movie_ref, movie_cyl, _ = bead_movies
        calibration = {"X Coefficients": [1, 2, 3]}
        calibration, _ = localize.fit_lateral_transform(
            movie_ref,
            movie_cyl,
            calibration,
            box=BOX,
            minimum_ng=1000,
            pixelsize=PIXELSIZE,
            ref_path="ref.tif",
            target_path="cyl.tif",
        )
        (entry,) = lib.lateral_transforms(calibration)
        # the existing 3D calibration is augmented, not replaced
        assert calibration["X Coefficients"] == [1, 2, 3]
        assert entry["Type"] == "astigmatism"
        assert entry["Direction"].startswith("cylindrical -> reference")
        assert entry["Reference image"] == "ref.tif"
        assert entry["Target image"] == "cyl.tif"
        assert entry["Pixelsize (nm)"] == float(PIXELSIZE)
        assert entry["Decomposition"]["rotation_deg"] == pytest.approx(
            0.46, abs=0.1
        )
        # plain floats, so yaml.dump can write the calibration
        assert all(
            isinstance(v, float)
            for row in entry["Transform"]["matrix"]
            for v in row
        )

    def test_identical_movies_give_identity(self, bead_movies):
        movie_ref, _, ref_xy = bead_movies
        calibration, _ = localize.fit_lateral_transform(
            movie_ref, movie_ref, {}, box=BOX, minimum_ng=1000
        )
        matrix = affine_matrix_3x3(
            lib.lateral_transforms(calibration)[0]["Transform"]
        )
        mapped = _apply_homogeneous(matrix, ref_xy)
        assert np.allclose(mapped, ref_xy, atol=0.05)

    def test_survives_a_yaml_round_trip(self, bead_movies, tmp_path):
        movie_ref, movie_cyl, _ = bead_movies
        calibration, _ = localize.fit_lateral_transform(
            movie_ref,
            movie_cyl,
            dict(CALIB_3D),  # io.load_calibration validates the 3D entries
            box=BOX,
            minimum_ng=1000,
            pixelsize=PIXELSIZE,
        )
        path = str(tmp_path / "calib.yaml")
        io.save_calibration(path, calibration)
        loaded = io.load_calibration(path)
        assert lib.lateral_transform_models(loaded)[0].allclose(
            lib.lateral_transform_models(calibration)[0]
        )

    def test_multichannel_calibration_is_rejected(self, bead_movies):
        """Affine corrections are single-channel only: a multichannel
        spline fit registers its channels itself."""
        movie_ref, movie_target, _ = bead_movies
        for model in ("spline-3d-multichannel", precision._LINK_XYZ_MODEL):
            with pytest.raises(ValueError, match="single-channel"):
                localize.fit_lateral_transform(
                    movie_ref,
                    movie_target,
                    {"model": model, "n_channels": 2},
                    box=BOX,
                    minimum_ng=1000,
                )

    def test_too_few_beads_raises(self):
        movie = _affine_bead_image(np.array([[40.0, 40.0], [140.0, 140.0]]))
        with pytest.raises(ValueError, match="matched bead pair"):
            localize.fit_lateral_transform(
                movie, movie, {}, box=BOX, minimum_ng=1000
            )

    def test_qc_carries_the_plot_inputs(self, bead_movies):
        movie_ref, movie_cyl, ref_xy = bead_movies
        _, qc = localize.fit_lateral_transform(
            movie_ref,
            movie_cyl,
            {},
            box=BOX,
            minimum_ng=1000,
            pixelsize=PIXELSIZE,
        )
        assert set(qc) == {
            "img_ref",
            "img_target",
            "img_cor",
            "pairs_ref",
            "beads_ref",
            "beads_target",
            "idx_ref",
            "idx_target",
            "box",
            "decomposition",
            "n_pairs",
            "pixelsize",
            "transform_type",
            "ref_path",
            "target_path",
        }
        # every detection, and which of them were matched, so the viewer can
        # color-code the pairing and gray out the beads that were dropped
        assert len(qc["beads_ref"]) >= len(qc["pairs_ref"])
        assert len(qc["idx_ref"]) == len(qc["pairs_ref"])
        assert len(qc["idx_target"]) == len(qc["pairs_ref"])
        assert qc["img_cor"].shape == qc["img_ref"].shape
        assert qc["pixelsize"] == PIXELSIZE
        # the warp brings the cylindrical image onto the reference: the
        # corrected image must correlate better with the reference than the
        # raw one does

        def _corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            return float((a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum()))

        assert _corr(qc["img_ref"], qc["img_cor"]) > _corr(
            qc["img_ref"], qc["img_target"]
        )

    def test_fit_does_not_plot(self, bead_movies, monkeypatch):
        """The fit half must not touch matplotlib - it runs off the GUI
        thread, where drawing is not allowed."""
        movie_ref, movie_cyl, _ = bead_movies

        def _fail(*args, **kwargs):
            raise AssertionError("fit_lateral_transform must not plot")

        monkeypatch.setattr(localize.plt, "figure", _fail)
        monkeypatch.setattr(localize.plt, "show", _fail)
        localize.fit_lateral_transform(
            movie_ref, movie_cyl, {}, box=BOX, minimum_ng=1000
        )

    def test_calibrate_wrapper_fits_and_plots(self, bead_movies, monkeypatch):
        """The one-call entry point still does both, with the plot fed from
        the fit's qc dict."""
        movie_ref, movie_cyl, _ = bead_movies
        drawn = {}

        def _plot(qc, save_path=""):
            drawn["qc"] = qc
            drawn["save_path"] = save_path

        monkeypatch.setattr(localize, "plot_lateral_calibration", _plot)
        calibration = localize.calibrate_lateral_transform(
            movie_ref,
            movie_cyl,
            {},
            box=BOX,
            minimum_ng=1000,
            pixelsize=PIXELSIZE,
            plot_path="figure.png",
        )
        assert len(lib.lateral_transforms(calibration)) == 1
        assert drawn["save_path"] == "figure.png"
        assert drawn["qc"]["n_pairs"] == 25


@pytest.mark.gui
class TestShortMovieDisplayGUI:
    """A movie too short for the temporal median filter must still be
    visible: the in-focus bead images an affine calibration uses are single
    frames, and the filter would make them identically zero (black)."""

    @pytest.fixture
    def window(self):
        window = localize_gui.Window()
        yield window
        window.close()

    @staticmethod
    def _bead_movie(n_frames=1):
        frame = np.full((64, 64), 100, dtype=np.uint16)
        frame[9:12, 19:22] = 6000  # one bright bead
        return np.repeat(frame[np.newaxis], n_frames, axis=0)

    def _show(self, window, movie):
        """Display ``movie`` the way loading one does, and return the
        brightest pixel actually painted."""
        window.movie = movie
        window.info = [{"Frames": len(movie), "Height": 64, "Width": 64}]
        window.curr_frame_number = 0
        window._apply_channel_to_ui(0)
        item = [
            i for i in window.scene.items() if "Pixmap" in type(i).__name__
        ][0]
        image = item.pixmap().toImage()
        return max(
            image.pixelColor(x, y).red()
            for y in range(image.height())
            for x in range(image.width())
        )

    def test_single_frame_movie_is_not_blank(self, window):
        window.parameters_dialog.temporal_median_checkbox.setChecked(True)
        movie = self._bead_movie(1)
        brightest = self._show(window, movie)
        assert brightest > 0, "single-frame movie displayed as all black"
        # identification and the preview see the same raw movie
        assert window.identification_movie() is movie
        assert window.parameters["Temporal Median Window"] == 0

    def test_the_checkbox_says_why_the_filter_is_unavailable(self, window):
        window.parameters_dialog.temporal_median_checkbox.setChecked(True)
        checkbox = window.parameters_dialog.temporal_median_checkbox
        self._show(window, self._bead_movie(1))
        assert not checkbox.isEnabled()
        assert "frames" in checkbox.text()
        assert "1 frame(s)" in checkbox.toolTip()
        # ... and it comes back for a movie the filter can work on
        self._show(window, self._bead_movie(8))
        assert checkbox.isEnabled()
        assert checkbox.text() == "Temporal median filter"

    def test_the_filter_still_runs_on_a_long_enough_movie(self, window):
        window.parameters_dialog.temporal_median_checkbox.setChecked(True)
        movie = self._bead_movie(8)
        self._show(window, movie)
        assert window.parameters["Temporal Median Window"] > 0
        assert window.identification_movie() is not movie

    def test_a_uniform_frame_does_not_cast_nan(self, window):
        """Auto contrast divides by the frame range; a uniform frame made
        that 0/0, and casting NaN to uint8 painted the view black."""
        window.parameters_dialog.temporal_median_checkbox.setChecked(False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            brightest = self._show(
                window, np.full((3, 32, 32), 100, dtype=np.uint16)
            )
        assert brightest == 0  # uniform data really is featureless


@pytest.mark.gui
class TestAffinePairingOverlayGUI:
    """The bead pairing of an affine calibration, drawn in the Localize
    viewer as color-coded identification boxes."""

    QC = {
        "beads_ref": np.array([[10.0, 20.0], [50.0, 60.0], [90.0, 30.0]]),
        "beads_target": np.array([[12.0, 17.0], [52.0, 57.0]]),
        "idx_ref": np.array([0, 1]),
        "idx_target": np.array([0, 1]),
        "n_pairs": 2,
        "box": 7,
        "ref_path": "/tmp/ref.tif",
        "target_path": "/tmp/target.tif",
        "transform_type": "chromatic",
    }

    @pytest.fixture
    def window(self):
        window = localize_gui.Window()
        window.set_affine_pairing(dict(self.QC))
        yield window
        window.close()

    def _draw(self, window, movie_path):
        """Draw onto a fresh scene and return the box colors, in the order
        they were added."""
        window.movie_path = movie_path
        window.scene = QtWidgets.QGraphicsScene()
        drew = window.draw_affine_pairing()
        colors = [
            item.pen().color().name() for item in window.scene.items()[::-1]
        ]
        return drew, colors

    def test_matched_beads_share_a_color_between_the_images(self, window):
        drew_ref, ref_colors = self._draw(window, self.QC["ref_path"])
        drew_target, target_colors = self._draw(window, self.QC["target_path"])
        assert drew_ref and drew_target
        # one box per detection in each image
        assert len(ref_colors) == 3 and len(target_colors) == 2
        # pair k has the same color on both sides, and pairs differ
        assert ref_colors[:2] == target_colors
        assert target_colors[0] != target_colors[1]

    def test_unmatched_beads_are_gray(self, window):
        _, ref_colors = self._draw(window, self.QC["ref_path"])
        assert ref_colors[2] == localize_gui.LINK_UNMATCHED_COLOR.name()
        assert "1 unpaired" in window.status_bar.currentMessage()

    def test_boxes_use_the_calibration_box_size(self, window):
        window.movie_path = self.QC["ref_path"]
        window.scene = QtWidgets.QGraphicsScene()
        window.draw_affine_pairing()
        rect = window.scene.items()[-1].rect()
        assert rect.width() == self.QC["box"]
        # centered on the bead: [row, col] = [10, 20] with box 7 -> x = 17
        assert rect.x() == 20.0 - 3 and rect.y() == 10.0 - 3

    def test_not_drawn_for_another_movie(self, window):
        drew, colors = self._draw(window, "/tmp/some_other_movie.tif")
        assert not drew and colors == []

    def test_nothing_drawn_before_a_calibration_ran(self):
        window = localize_gui.Window()
        try:
            window.movie_path = self.QC["ref_path"]
            assert window.draw_affine_pairing() is False
        finally:
            window.close()


@pytest.mark.gui
class TestCalibrateAffineDialogGUI:
    """The calibration dialog serves both transform types and can start a
    standalone calibration file."""

    def _dialog(self):
        window = QtWidgets.QWidget()
        window.movie_path = ""
        return localize_gui.CalibrateAffineDialog(window)

    def test_defaults_to_astigmatism_with_lens_labels(self):
        dialog = self._dialog()
        assert dialog.transform_type == "astigmatism"
        assert "Cylindrical" in dialog.target_label.text()

    def test_switching_to_chromatic_relabels_the_images(self):
        dialog = self._dialog()
        dialog.type_combo.setCurrentIndex(1)
        assert dialog.transform_type == "chromatic"
        assert "channel" in dialog.target_label.text().lower()
        assert "channel" in dialog.reference_label.text().lower()

    def test_defaults_to_affine_and_offers_the_other_models(self):
        dialog = self._dialog()
        assert dialog.transform_model == "affine"
        offered = [
            dialog.model_combo.itemData(i)
            for i in range(dialog.model_combo.count())
        ]
        assert offered == list(transforms.MODELS)

    def test_the_chosen_model_reaches_the_worker(self, monkeypatch):
        """The combo is only useful if its value is what gets fitted."""
        seen = {}

        def fake_fit(*args, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop after capturing the arguments")

        monkeypatch.setattr(localize, "fit_lateral_transform", fake_fit)
        worker = localize_gui.AffineCalibrationWorker(
            ref_path="ref.tif",
            target_path="target.tif",
            calibration_path="calib.yaml",
            box=7,
            minimum_ng=1000,
            prompt_for_path=lambda path: (lambda *a, **k: None),
            pixelsize_prompt=lambda: None,
            transform_type="chromatic",
            model="polynomial3",
        )
        assert worker.model == "polynomial3"

    def test_calibrate_requests_a_fit_without_closing(self):
        """The dialog must stay open: the bead pairing is inspected right
        after the fit by loading either image with 'Show', and re-picking
        all three paths to do that would be absurd."""
        dialog = self._dialog()
        dialog.show()
        requested = []
        dialog.calibrate_requested.connect(lambda: requested.append(True))
        dialog.calibrate_button.click()
        assert requested == [True]
        assert dialog.isVisible()

    def test_close_button_closes_it(self):
        dialog = self._dialog()
        dialog.show()
        close = dialog.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        close.click()
        assert not dialog.isVisible()


@pytest.mark.gui
class TestAffineCalibrationWorkerGUI:
    """The worker appends to an existing calibration of any format and
    starts a new standalone one when the path does not exist yet."""

    def _worker(self, tmp_path, calib_path, transform_type, monkeypatch):
        ref_xy = _affine_bead_grid()
        movie_ref = _affine_bead_image(ref_xy)
        movie_target = _affine_bead_image(ref_xy + np.array([2.0, -1.0]))
        movies = {"ref.tif": movie_ref, "target.tif": movie_target}

        def fake_load_movie(path, prompt_info=None, progress=None):
            name = os.path.basename(path)
            return movies[name], [{"Pixelsize": PIXELSIZE}]

        monkeypatch.setattr(io, "load_movie", fake_load_movie)
        return localize_gui.AffineCalibrationWorker(
            ref_path=str(tmp_path / "ref.tif"),
            target_path=str(tmp_path / "target.tif"),
            calibration_path=calib_path,
            box=BOX,
            minimum_ng=1000,
            prompt_for_path=lambda path: None,
            pixelsize_prompt=lambda: PIXELSIZE,
            transform_type=transform_type,
        )

    def test_creates_a_standalone_chromatic_calibration(
        self, tmp_path, monkeypatch
    ):
        calib_path = str(tmp_path / "chromatic.yaml")  # does not exist yet
        worker = self._worker(tmp_path, calib_path, "chromatic", monkeypatch)
        failures = []
        worker.failed.connect(failures.append)
        worker.run()
        assert not failures
        (entry,) = lib.lateral_transforms(io.load_any_calibration(calib_path))
        assert entry["Type"] == "chromatic"

    def test_appends_to_a_spline_calibration(self, tmp_path, monkeypatch):
        calib_path = str(tmp_path / "spline_calib.hdf5")
        io.save_spline_calibration(
            calib_path,
            {
                "coefficients": np.zeros((16, 2, 2), dtype=np.float32),
                "model": "spline-2d",
                "n_data": [3, 3],
            },
        )
        worker = self._worker(tmp_path, calib_path, "chromatic", monkeypatch)
        failures = []
        worker.failed.connect(failures.append)
        worker.run()
        assert not failures
        calibration = io.load_any_calibration(calib_path)
        # the PSF itself survives alongside the new correction
        assert calibration["model"] == "spline-2d"
        assert len(lib.lateral_transforms(calibration)) == 1


@pytest.mark.gui
class TestAffineDuplicateGuardsGUI:
    """A correction the loaded calibration already carries must not be
    taken a second time through the 2D lateral correction box."""

    MATRIX = [[1.0, 0.0, 4.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]]

    def _calibration_file(self, tmp_path, name="calib.yaml"):
        calibration = dict(CALIB_3D)
        lib.append_lateral_transform(
            calibration,
            {
                "Type": "astigmatism",
                "Transform": affine(self.MATRIX).to_dict(),
                "Bead pairs": 20,
            },
        )
        path = str(tmp_path / name)
        io.save_any_calibration(path, calibration)
        return path, calibration

    def test_loading_the_z_calibration_again_is_refused(
        self, tmp_path, monkeypatch
    ):
        path, calibration = self._calibration_file(tmp_path)
        dialog = localize_gui.ParametersDialog(None)
        dialog.z_calibration = calibration
        warnings_shown = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **kwargs: warnings_shown.append(args[2]),
        )
        dialog.update_affine_calib([path])
        assert dialog.lateral_transforms == []
        assert any(
            "would correct the coordinates twice" in w for w in warnings_shown
        )

    def test_an_unrelated_correction_still_loads(self, tmp_path, monkeypatch):
        _, calibration = self._calibration_file(tmp_path)
        other = {}
        lib.append_lateral_transform(
            other,
            {
                "Type": "chromatic",
                "Transform": affine(
                    [[1.0, 0.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ).to_dict(),
                "Bead pairs": 8,
            },
        )
        other_path = str(tmp_path / "chromatic.yaml")
        io.save_any_calibration(other_path, other)
        dialog = localize_gui.ParametersDialog(None)
        dialog.z_calibration = calibration
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning", lambda *a, **k: None
        )
        dialog.update_affine_calib([other_path])
        assert len(dialog.lateral_transforms) == 1
        assert dialog.lateral_transforms[0]["Type"] == "chromatic"


class TestChainedAffineTransforms:
    """The ordered list of corrections that any calibration can carry."""

    ASTIG = [[1.002, -0.01, 3.0], [0.009, 0.998, -1.5], [0.0, 0.0, 1.0]]
    CHROMATIC = [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]

    @staticmethod
    def _entry(kind, matrix):
        return {
            "Type": kind,
            "Transform": affine(matrix).to_dict(),
            "Bead pairs": 12,
        }

    def _both(self):
        calibration = {}
        lib.append_lateral_transform(
            calibration, self._entry("astigmatism", self.ASTIG)
        )
        lib.append_lateral_transform(
            calibration, self._entry("chromatic", self.CHROMATIC)
        )
        return calibration

    def test_locs_are_mapped_into_the_reference_frame(self):
        calibration = {
            lib.LATERAL_TRANSFORMS_KEY: [
                self._entry("astigmatism", self.ASTIG)
            ]
        }
        locs = pd.DataFrame({"x": [10.0, 100.0], "y": [20.0, 200.0]})
        expected = _apply_homogeneous(
            np.asarray(self.ASTIG), locs[["x", "y"]].to_numpy()
        )
        moved = lib.apply_lateral_transforms(locs, calibration)
        assert np.allclose(moved[["x", "y"]].to_numpy(), expected, atol=1e-4)

    def test_two_transforms_apply_in_order(self):
        locs = pd.DataFrame({"x": [10.0, 100.0], "y": [20.0, 200.0]})
        xy = locs[["x", "y"]].to_numpy()
        expected = _apply_homogeneous(
            np.asarray(self.CHROMATIC),
            _apply_homogeneous(np.asarray(self.ASTIG), xy),
        )
        moved = lib.apply_lateral_transforms(locs, self._both())
        assert np.allclose(moved[["x", "y"]].to_numpy(), expected, atol=1e-4)
        # the input frame is left alone
        assert np.allclose(locs[["x", "y"]].to_numpy(), xy)

    def test_same_type_is_replaced_not_stacked(self):
        calibration = self._both()
        updated = self._entry("astigmatism", np.eye(3).tolist())
        lib.append_lateral_transform(calibration, updated)
        transforms = lib.lateral_transforms(calibration)
        assert [t["Type"] for t in transforms] == [
            "astigmatism",
            "chromatic",
        ]
        assert np.allclose(
            affine_matrix_3x3(transforms[0]["Transform"]), np.eye(3)
        )

    def test_no_transforms_is_a_no_op(self):
        locs = pd.DataFrame({"x": [1.0], "y": [2.0]})
        assert lib.apply_lateral_transforms(locs, {}) is locs
        assert lib.apply_lateral_transforms(locs, None) is locs

    def test_duplicates_of_the_calibration_are_dropped(self):
        """A correction the fit's own calibration carries must not be
        applied a second time as an extra."""
        calibration = self._both()
        # the very same file loaded again as an "extra"
        new, duplicates = lib.drop_duplicate_lateral_transforms(
            calibration, calibration
        )
        assert new == []
        assert len(duplicates) == 2

    def test_duplicate_detection_compares_matrices_not_identity(self):
        """A copy of the same transform saved under another name (and with
        different bookkeeping) still counts as a duplicate."""
        calibration = self._both()
        copy = {
            "Type": "chromatic",
            "Transform": affine(self.CHROMATIC).to_dict(),
            "Reference image": "somewhere_else.tif",
            "Bead pairs": 99,
        }
        fresh = {
            "Type": "chromatic",
            "Transform": affine(
                [[1.0, 0.0, 4.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]
            ).to_dict(),
        }
        new, duplicates = lib.drop_duplicate_lateral_transforms(
            [copy, fresh], calibration
        )
        assert new == [fresh]
        assert duplicates == [copy]

    def test_nothing_to_compare_against_keeps_everything(self):
        calibration = self._both()
        new, duplicates = lib.drop_duplicate_lateral_transforms(
            calibration, None
        )
        assert len(new) == 2 and duplicates == []
        # a calibration still given as a path carries nothing to compare
        new, duplicates = lib.drop_duplicate_lateral_transforms(
            calibration, "calib.yaml"
        )
        assert len(new) == 2 and duplicates == []

    def test_a_duplicate_of_the_fits_own_calibration_is_dropped(self):
        """The 2D / spline route: the fit applies what its calibration
        carries, so the same one handed in as an extra must be skipped.
        (The astigmatic 3D route does this inside ``zfit`` - see
        ``TestZfitSeparatelyLoadedLateralCorrections``.)"""
        locs = pd.DataFrame(
            {"x": [10.0, 20.0], "y": [30.0, 40.0], "frame": [0, 1]}
        )
        calibration = self._both()
        with pytest.warns(UserWarning, match="already carries"):
            out, info = localize._apply_extra_affine(
                locs, [], calibration, applied=calibration
            )
        assert out is locs  # untouched
        assert info == []

    def test_a_genuinely_new_correction_is_applied(self):
        locs = pd.DataFrame(
            {"x": [10.0, 20.0], "y": [30.0, 40.0], "frame": [0, 1]}
        )
        extra = [self._entry("chromatic", self.CHROMATIC)]
        # the fit's own calibration carries a different correction
        applied = {
            lib.LATERAL_TRANSFORMS_KEY: [
                self._entry("astigmatism", self.ASTIG)
            ]
        }
        out, info = localize._apply_extra_affine(
            locs, [], extra, applied=applied
        )
        np.testing.assert_allclose(out["x"], locs["x"] + 5.0)
        assert info[-1]["Lateral corrections applied"] == [
            "chromatic, affine (12 bead pairs)"
        ]

    def test_chromatic_only_calibration_round_trips_as_yaml(self, tmp_path):
        calibration = {}
        lib.append_lateral_transform(
            calibration, self._entry("chromatic", self.CHROMATIC)
        )
        path = str(tmp_path / "chromatic.yaml")
        io.save_any_calibration(path, calibration)
        loaded = io.load_any_calibration(path)
        assert lib.lateral_transform_models(loaded)[0].allclose(
            lib.lateral_transform_models(calibration)[0]
        )

    def test_spline_calibration_carries_transforms(self, tmp_path):
        calibration = {
            "coefficients": np.zeros((16, 2, 2), dtype=np.float32),
            "model": "spline-2d",
            "n_data": [3, 3],
        }
        lib.append_lateral_transform(
            calibration, self._entry("chromatic", self.CHROMATIC)
        )
        path = str(tmp_path / "spline_calib.hdf5")
        io.save_any_calibration(path, calibration)
        loaded = io.load_any_calibration(path)
        assert np.allclose(
            affine_matrix_3x3(lib.lateral_transform_models(loaded)[0]),
            np.asarray(self.CHROMATIC),
        )
        # cropping to a smaller fit box must not drop the corrections
        cropped = localize.crop_spline_calibration(loaded, 2)
        assert len(lib.lateral_transforms(cropped)) == 1


class TestRangeSlider:
    """The two-handle slider behind the contrast slider below the movie.
    Qt has no such widget, so every invariant here is hand-rolled."""

    @staticmethod
    def _slider(minimum=0, maximum=1000, gap=1):
        slider = lib.RangeSlider()
        slider.resize(200, 15)
        slider.setMinimumGap(gap)
        slider.setRange(minimum, maximum)
        slider.setValues(100, 900)
        return slider

    def test_values_are_clamped_into_the_track(self):
        slider = self._slider()
        try:
            slider.setValues(-500, 5000)
            assert slider.values() == (0.0, 1000.0)
        finally:
            slider.deleteLater()

    @pytest.mark.parametrize(
        "low, high, moved, expected",
        [
            (500, 500, "low", (499.0, 500.0)),  # the dragged handle gives way
            (500, 500, "high", (500.0, 501.0)),
            (0, 0, "low", (0.0, 1.0)),  # ... unless it is at the end
            (1000, 1000, "high", (999.0, 1000.0)),
        ],
    )
    def test_handles_stay_a_gap_apart(self, low, high, moved, expected):
        """White divides in ``ContrastDialog.to_uint8`` and both values are
        rounded to integers, so the handles must never collapse onto the
        same value."""
        slider = self._slider()
        try:
            slider.setValues(low, high, moved=moved)
            assert slider.values() == expected
        finally:
            slider.deleteLater()

    def test_pixel_mapping_round_trips(self):
        slider = self._slider()
        try:
            for value in (0, 1, 250, 999, 1000):
                x = slider._value_to_x(value)
                assert slider._x_to_value(x) == pytest.approx(value)
            # both extremes stay inside the widget, handle included
            assert slider._value_to_x(0) >= slider.HANDLE_WIDTH / 2
            assert (
                slider._value_to_x(1000)
                <= slider.width() - slider.HANDLE_WIDTH / 2
            )
        finally:
            slider.deleteLater()

    def test_a_degenerate_track_does_not_divide_by_zero(self):
        slider = self._slider()
        try:
            slider.setRange(7, 7)
            assert slider.values() == (7.0, 7.0)
            slider._value_to_x(7)
            slider._x_to_value(0)
        finally:
            slider.deleteLater()

    def test_setting_the_range_reclamps_the_values(self):
        slider = self._slider()
        try:
            slider.setRange(0, 500)
            assert slider.values() == (100.0, 500.0)
        finally:
            slider.deleteLater()

    def test_clicking_grabs_the_nearer_handle(self):
        slider = self._slider()
        try:
            assert slider._handle_at(slider._value_to_x(120)) == "low"
            assert slider._handle_at(slider._value_to_x(880)) == "high"
            # outside the span, the handle on that side
            assert slider._handle_at(slider._value_to_x(10)) == "low"
            assert slider._handle_at(slider._value_to_x(990)) == "high"
        finally:
            slider.deleteLater()

    def test_dragging_emits_the_pair(self):
        slider = self._slider()
        emitted = []
        slider.valuesChanged.connect(lambda lo, hi: emitted.append((lo, hi)))
        try:
            slider._move_handle("high", slider._value_to_x(400))
            assert slider.values()[1] == pytest.approx(400)
            assert emitted[-1] == slider.values()
            # a no-op move must not emit (it would redraw the frame)
            emitted.clear()
            slider._move_handle("high", slider._value_to_x(400))
            assert emitted == []
        finally:
            slider.deleteLater()


@pytest.mark.gui
class TestContrastSliderGUI:
    """The contrast slider below the frame slider is a second view onto the
    contrast dialog's spinboxes; the two must never drift apart."""

    @pytest.fixture
    def window(self):
        window = localize_gui.Window()
        yield window
        window.close()

    @staticmethod
    def _movie(n_frames=20):
        rng = np.random.default_rng(0)
        movie = rng.normal(200, 10, (n_frames, 32, 32)).astype("uint16")
        movie[:, 10, 10] = 3000  # a bright spot to widen the range
        return movie

    def _show(self, window, movie=None):
        movie = self._movie() if movie is None else movie
        window.movie = movie
        window.info = [{"Frames": len(movie), "Height": 32, "Width": 32}]
        window.contrast_slider.setEnabled(True)
        window._apply_channel_to_ui(0)
        return movie

    def test_track_spans_the_sampled_movie(self, window):
        movie = self._show(window)
        lo, hi = window.contrast_slider.range()
        assert lo <= movie.min()
        assert hi >= movie.max()
        # padded, but not by orders of magnitude (the dtype range would be
        # 0-65535 and leave both handles crammed on the left)
        assert hi < 2 * movie.max()

    def test_handles_follow_the_spinboxes(self, window):
        self._show(window)
        contrast = window.contrast_dialog
        assert window.contrast_slider.values() == (
            contrast.black_spinbox.value(),
            contrast.white_spinbox.value(),
        )
        contrast.black_spinbox.setValue(150)
        assert window.contrast_slider.values()[0] == 150

    def test_a_value_beyond_the_track_widens_it(self, window):
        self._show(window)
        window.contrast_dialog.white_spinbox.setValue(50000)
        assert window.contrast_slider.range()[1] >= 50000
        assert window.contrast_slider.values()[1] == 50000

    def test_dragging_sets_a_manual_contrast_and_redraws_once(self, window):
        self._show(window)
        contrast = window.contrast_dialog
        assert contrast.auto_checkbox.isChecked()
        draws = []
        window.draw_frame = lambda: draws.append(1)
        window.contrast_slider.setValues(150, 1000)
        assert not contrast.auto_checkbox.isChecked()
        assert (
            contrast.black_spinbox.value(),
            contrast.white_spinbox.value(),
        ) == (150, 1000)
        # one redraw per drag step, not one per spinbox
        assert len(draws) == 1

    def test_auto_button_mirrors_the_dialog_checkbox(self, window):
        self._show(window)
        contrast = window.contrast_dialog
        button = window.contrast_auto_button
        assert button.isChecked() and contrast.auto_checkbox.isChecked()
        # dialog -> button
        contrast.auto_checkbox.setChecked(False)
        assert not button.isChecked()
        contrast.auto_checkbox.setChecked(True)
        assert button.isChecked()
        # a manual edit unchecks Auto, so the button follows it too
        window.contrast_slider.setValues(150, 1000)
        assert not button.isChecked()
        # button -> dialog: re-derives the range from the frame
        button.setChecked(True)
        assert contrast.auto_checkbox.isChecked()
        frame = window.identification_movie()[window.curr_frame_number]
        assert (
            contrast.black_spinbox.value(),
            contrast.white_spinbox.value(),
        ) == (frame.min(), frame.max())

    def test_the_track_does_not_shrink_while_browsing_frames(self, window):
        self._show(window)
        track = window.contrast_slider.range()
        for number in range(1, 10):
            window.set_frame(number)
            lo, hi = window.contrast_slider.range()
            assert lo <= track[0] and hi >= track[1]

    def test_a_filter_rescales_the_track(self, window):
        """Temporal median subtracts the background, so the displayed
        intensities collapse towards zero; a track left at the raw camera
        counts would pin both handles to its left edge."""
        self._show(window)
        raw_track = window.contrast_slider.range()
        window.parameters_dialog.temporal_median_checkbox.setChecked(True)
        filtered_track = window.contrast_slider.range()
        assert filtered_track[1] < raw_track[1] / 2
        filtered = window.identification_movie()[window.curr_frame_number]
        assert filtered_track[1] >= filtered.max()

    def test_bounds_stay_within_what_the_spinboxes_accept(self, window):
        """The track is what the handles can reach, so it must not run past
        the spinbox limits the values are written into."""
        self._show(window)
        contrast = window.contrast_dialog
        lo, hi = window._clamp_contrast_bounds(-1e6, 1e9)
        assert lo == contrast.black_spinbox.minimum()
        assert hi == contrast.white_spinbox.maximum()

    def test_no_movie_leaves_the_slider_alone(self, window):
        window.update_contrast_slider_range()
        assert not window.contrast_slider.isEnabled()


@pytest.mark.gui
class TestContrastDialog:
    """The contrast mapping is shared by the Auto and the manual branch, so
    unchecking Auto must freeze what is on screen rather than re-scale it on
    the next redraw (which used to happen at the next frame)."""

    class _StubWindow(QtWidgets.QMainWindow):
        """Minimal stand-in for the Localize window: a two-frame movie with
        very different intensity ranges, plus a draw counter."""

        def __init__(self, frames):
            super().__init__()
            self.movie = frames
            self.curr_frame_number = 0
            self.draw_calls = 0

        def identification_movie(self):
            return self.movie

        def draw_frame(self):
            self.draw_calls += 1

    @staticmethod
    def _frames():
        dim = np.linspace(100, 300, 64, dtype="float32").reshape(8, 8)
        bright = np.linspace(100, 900, 64, dtype="float32").reshape(8, 8)
        return np.stack([dim, bright])

    def _dialog(self):
        window = self._StubWindow(self._frames())
        return localize_gui.ContrastDialog(window), window

    def test_manual_matches_auto_for_the_same_range(self):
        """Auto sets black/white to the frame's min/max, so rendering the
        frame with Auto off and those same values has to be pixel-identical
        - otherwise switching Auto off visibly changes the image."""
        dialog, window = self._dialog()
        try:
            frame = window.movie[0]
            auto = dialog.to_uint8(frame)
            dialog.auto_checkbox.setChecked(False)
            assert (
                dialog.black_spinbox.value(),
                dialog.white_spinbox.value(),
            ) == (frame.min(), frame.max())
            np.testing.assert_array_equal(dialog.to_uint8(frame), auto)
            # the full range is used: black maps to 0, white to 255
            assert dialog.to_uint8(frame).min() == 0
            assert dialog.to_uint8(frame).max() == 255
        finally:
            dialog.deleteLater()

    def test_manual_range_does_not_follow_the_frame(self):
        """With Auto off a brighter frame must render brighter (clipping at
        white), not get renormalized back to the full 0-255 range."""
        dialog, window = self._dialog()
        try:
            dialog.auto_checkbox.setChecked(False)
            dim, bright = window.movie
            np.testing.assert_array_equal(
                dialog.to_uint8(bright)[bright > dim.max()], 255
            )
            assert dialog.to_uint8(bright).mean() > dialog.to_uint8(dim).mean()
            # Auto, in contrast, rescales every frame to the same spread
            # (both are linear ramps; atol covers float32 rounding)
            dialog.auto_checkbox.setChecked(True)
            np.testing.assert_allclose(
                dialog.to_uint8(bright).astype(int),
                dialog.to_uint8(dim).astype(int),
                atol=1,
            )
        finally:
            dialog.deleteLater()

    def test_unchecking_auto_keeps_the_values_and_redraws(self):
        """The change has to take effect immediately: before, only the next
        frame change triggered a redraw, which is when the user saw the
        contrast 'adjust itself'."""
        dialog, window = self._dialog()
        try:
            frame = window.movie[0]
            window.draw_calls = 0
            dialog.auto_checkbox.setChecked(False)
            assert window.draw_calls == 1
            assert (
                dialog.black_spinbox.value(),
                dialog.white_spinbox.value(),
            ) == (frame.min(), frame.max())
        finally:
            dialog.deleteLater()

    def test_typing_a_value_unchecks_auto_and_keeps_it(self):
        """Editing a spinbox turns Auto off; that must not bounce back and
        overwrite what was just typed."""
        dialog, window = self._dialog()
        try:
            dialog.white_spinbox.setValue(500)
            assert not dialog.auto_checkbox.isChecked()
            assert dialog.white_spinbox.value() == 500
        finally:
            dialog.deleteLater()

    def test_set_frame_only_retunes_the_range_when_auto_is_on(self):
        """``Window.set_frame`` is what tracks the range per frame; with Auto
        off it must leave the spinboxes alone."""
        source = inspect.getsource(localize_gui.Window.set_frame)
        assert "auto_checkbox.isChecked()" in source
        assert "change_contrast_silently" in source


# ---------------------------------------------------------------------------
# sCMOS per-pixel noise model (Huang et al. 2013)
# ---------------------------------------------------------------------------


class TestCutMap:
    """``_cut_map`` is load-bearing: every map patch is cut by it."""

    def test_matches_cut_spots_exactly(self):
        """Tile a map across frames, then ``_cut_spots`` must reproduce it.

        This is the only check that the map patch and the spot it accompanies
        describe the same pixels. An off-by-one here would put a hot pixel's
        variance on its neighbor, silently.
        """
        rng = np.random.default_rng(0)
        image = rng.normal(100.0, 5.0, (32, 30)).astype(np.float32)
        movie = np.broadcast_to(image, (4, 32, 30)).copy()
        ids = pd.DataFrame(
            {
                "frame": [0, 1, 2, 3, 0],
                "x": [10, 20, 5, 25, 15],
                "y": [12, 8, 20, 16, 4],
            }
        )
        spots = localize._cut_spots(movie, ids, BOX)
        patches = localize._cut_map(image, ids, BOX)
        np.testing.assert_array_equal(patches, spots.astype(np.float32))

    def test_places_a_unique_pixel_where_it_belongs(self):
        image = np.zeros((20, 20), dtype=np.float32)
        image[9, 13] = 1.0
        ids = pd.DataFrame({"frame": [0], "x": [11], "y": [10]})
        patch = localize._cut_map(image, ids, 7)
        # box-local: y = 9 - (10 - 3) = 2, x = 13 - (11 - 3) = 5
        assert patch[0, 2, 5] == 1.0
        assert patch.sum() == 1.0


class TestPhotonConversionWithMaps:
    def test_a_constant_offset_map_equals_the_scalar_baseline(self):
        rng = np.random.default_rng(1)
        spots = rng.normal(500.0, 20.0, (5, BOX, BOX)).astype(np.float32)
        info = {"Baseline": 100.0, "Sensitivity": 0.47, "Gain": 1}
        flat = np.full_like(spots, 100.0)
        np.testing.assert_allclose(
            localize._to_photons(spots, info, offset=flat),
            localize._to_photons(spots, info),
            rtol=0,
            atol=0,
        )

    def test_a_gain_map_overrides_the_scalar_sensitivity(self):
        rng = np.random.default_rng(2)
        spots = rng.normal(500.0, 20.0, (5, BOX, BOX)).astype(np.float32)
        info = {"Baseline": 100.0, "Sensitivity": 0.5, "Gain": 1}
        gain = np.full_like(spots, 2.0)  # sensitivity = 1 / 2.0 = 0.5
        np.testing.assert_allclose(
            localize._to_photons(spots, info, gain_patch=gain),
            localize._to_photons(spots, info),
            rtol=1e-6,
        )

    def test_variance_converts_by_the_squared_factor(self):
        """``var / g^2`` in the paper's notation, ``var * S^2`` in Picasso's."""
        info = {"Baseline": 100.0, "Sensitivity": 0.47, "Gain": 1}
        var_adu = np.full((3, BOX, BOX), 900.0, dtype=np.float32)
        got = localize._variance_to_photons(var_adu, info)
        np.testing.assert_allclose(got, 900.0 * 0.47**2, rtol=1e-6)

        gain = np.full((3, BOX, BOX), 2.13, dtype=np.float32)
        got = localize._variance_to_photons(var_adu, info, gain_patch=gain)
        np.testing.assert_allclose(got, 900.0 / 2.13**2, rtol=1e-6)


class TestClipForMle:
    def test_floors_at_zero_without_a_map(self):
        spots = np.array([[[-5.0, 3.0]]], dtype=np.float32)
        np.testing.assert_array_equal(
            localize._clip_for_mle(spots, None), [[[0.0, 3.0]]]
        )

    def test_floors_at_minus_variance_with_a_map(self):
        """``d + var >= 0`` is the condition; clipping at 0 instead would
        discard exactly the negative excursions readout noise creates."""
        spots = np.array([[[-5.0, -20.0]]], dtype=np.float32)
        var = np.array([[[10.0, 4.0]]], dtype=np.float32)
        np.testing.assert_array_equal(
            localize._clip_for_mle(spots, var), [[[-5.0, -4.0]]]
        )


class TestSeedSpots:
    """The ``-var`` floor must not leak into the initial parameters."""

    def test_is_a_no_op_without_a_map(self):
        spots = np.array([[[-5.0, 3.0]]], dtype=np.float32)
        assert localize._seed_spots(spots, None) is spots

    def test_floors_at_zero_with_a_map(self):
        spots = np.array([[[-5.0, 3.0]]], dtype=np.float32)
        var = np.array([[[10.0, 1.0]]], dtype=np.float32)
        np.testing.assert_array_equal(
            localize._seed_spots(spots, var), [[[0.0, 3.0]]]
        )

    def test_mle_survives_a_deeply_negative_hot_pixel(self):
        """Regression: seeding the background from a ``-var`` floored pixel
        makes the model mean negative over the whole ROI, so the likelihood
        is floored everywhere, the first Hessian is singular and the fit
        aborts with NaN parameters.
        """
        box = 7
        rng = np.random.default_rng(0)
        grid = np.arange(box, dtype=np.float64)
        dx = grid[None, :] - 3.0
        dy = grid[:, None] - 3.0
        mu = (
            300.0
            / (2 * np.pi * 1.3**2)
            * np.exp(-0.5 * (dx**2 + dy**2) / 1.3**2)
            + 5.0
        )
        spots = rng.poisson(mu, (32, box, box)).astype(np.float32)
        var = np.full((32, box, box), 1.0, dtype=np.float32)
        # One hot corner pixel, far from the emitter, with a large negative
        # excursion - exactly what a 2,000 ADU^2 pixel produces.
        var[:, 0, 0] = 400.0
        spots[:, 0, 0] = -80.0

        theta = localize.fit_spots_gauss(
            localize._clip_for_mle(spots, var), mle=True, variance=var
        )

        assert np.isfinite(theta).all()
        assert np.abs(theta[:, 1] - 3.0).max() < 1.0


class TestGaussCrlbVariance:
    def test_respects_the_transposed_axis_order(self):
        """``_gauss_crlb`` builds ``[spot, x, y]``; a patch is ``[spot, y, x]``.

        A single asymmetric hot pixel is the only way to see the difference:
        with a symmetric map, a transpose is invisible.
        """
        theta = np.array([[500.0, 3.0, 3.0, 1.2, 1.6, 5.0]])
        box = 7
        var = np.zeros((1, box, box), dtype=np.float32)
        var[0, 1, 5] = 400.0  # y = 1, x = 5 - deliberately not symmetric

        got = precision._gauss_crlb(theta, box, em=False, variance=var)

        # Reference: rebuild mu in [spot, y, x] and add the patch directly.
        grid = np.arange(box, dtype=np.float64)
        dy = grid[:, None] - theta[0, 2]
        dx = grid[None, :] - theta[0, 1]
        sx, sy, n = theta[0, 3], theta[0, 4], theta[0, 0]
        e = np.exp(-0.5 * (dx**2 / sx**2 + dy**2 / sy**2))
        sig = n / (2 * np.pi * sx * sy) * e
        grads = [
            sig / n,
            sig * dx / sx**2,
            sig * dy / sy**2,
            sig * (dx**2 / sx**3 - 1.0 / sx),
            sig * (dy**2 / sy**3 - 1.0 / sy),
            np.ones_like(sig),
        ]
        mu = sig + theta[0, 5] + var[0]
        g = np.stack(grads)
        fisher = np.einsum("pij,qij,ij->pq", g, g, 1.0 / mu)
        expected = np.diagonal(np.linalg.pinv(fisher))
        np.testing.assert_allclose(got[0], expected, rtol=1e-8)

    def test_a_hot_pixel_widens_the_bound(self):
        theta = np.array([[500.0, 3.0, 3.0, 1.3, 1.3, 5.0]])
        plain = precision._gauss_crlb(theta, 7, em=False)
        var = np.zeros((1, 7, 7), dtype=np.float32)
        var[0, 2, 4] = 500.0
        noisy = precision._gauss_crlb(theta, 7, em=False, variance=var)
        assert np.all(noisy >= plain - 1e-12)
        assert noisy[0, 1] > plain[0, 1]


@pytest.fixture(scope="module")
def scmos_scene():
    """A synthetic sCMOS camera, its calibration, and a single-emitter movie.

    Conditions follow the paper's own simulations - 200 photons per molecule,
    5 background photons per pixel - with one very noisy pixel inside the
    fitting box but off-center and off-axis, so a bias shows up in x and
    cannot be mistaken for a symmetric artifact.
    """
    from picasso import scmos

    rng = np.random.default_rng(11)
    h = w = 40
    offset = 100.0 + rng.normal(0, 1.0, (h, w))
    variance = rng.gamma(4.0, 0.4, (h, w)) + 0.4
    variance[18, 22] = 2000.0
    gain = np.full((h, w), 2.13)

    dark = offset + rng.normal(0, np.sqrt(variance), (20_000, h, w))
    calibration = scmos.calibrate_scmos(dark)

    n_frames = 3000
    x0, y0, sigma, photons, bg = 20.0, 18.0, 1.3, 200.0, 5.0
    yy, xx = np.mgrid[0:h, 0:w]
    mu = (
        photons
        / (2 * np.pi * sigma**2)
        * np.exp(-0.5 * (((xx - x0) ** 2 + (yy - y0) ** 2) / sigma**2))
        + bg
    )
    adu = (
        gain * rng.poisson(mu, (n_frames, h, w))
        + offset
        + rng.normal(0, np.sqrt(variance), (n_frames, h, w))
    )
    identifications = pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "x": np.full(n_frames, 20),
            "y": np.full(n_frames, 18),
            "net_gradient": np.full(n_frames, 1e4, np.float32),
        }
    )
    camera_info = {
        "Baseline": float(np.median(offset)),
        "Sensitivity": 1 / 2.13,
        "Gain": 1,
        "Qe": 1.0,
        "Pixelsize": 130,
    }
    return {
        "movie": adu.astype(np.float32),
        "info": [{"Frames": n_frames, "Height": h, "Width": w}],
        "identifications": identifications,
        "camera_info": camera_info,
        "calibration": calibration,
        "truth": (x0, y0),
    }


class TestScmosFit2DIntegration:
    """The whole feature, end to end through ``fit2D``."""

    @staticmethod
    def _fit(scene, picasso_movie_factory, method, calibration):
        movie = picasso_movie_factory(scene["movie"], scene["info"])
        locs, info = localize.fit2D(
            movie,
            [{}],
            dict(scene["camera_info"]),
            scene["identifications"],
            BOX,
            fitting_method=method,
            camera_calibration=calibration,
        )
        ok = np.isfinite(locs["x"]) & np.isfinite(locs["lpx"])
        return locs[ok], info

    def test_mle_bias_from_a_hot_pixel_is_removed(
        self, scmos_scene, picasso_movie_factory
    ):
        x0, _ = scmos_scene["truth"]
        plain, _ = self._fit(
            scmos_scene, picasso_movie_factory, "gaussmle", None
        )
        modeled, _ = self._fit(
            scmos_scene,
            picasso_movie_factory,
            "gaussmle",
            scmos_scene["calibration"],
        )
        bias_plain = abs(plain["x"].mean() - x0)
        bias_model = abs(modeled["x"].mean() - x0)
        sem = plain["x"].std() / np.sqrt(len(plain))

        # The test is only meaningful if the conventional fit is measurably
        # biased in the first place.
        assert bias_plain > 3 * sem
        assert bias_model < bias_plain / 3
        # ... and the precision must not be paid for it.
        assert modeled["x"].std() < plain["x"].std()

    def test_mle_precision_estimate_becomes_honest(
        self, scmos_scene, picasso_movie_factory
    ):
        """The conventional CRLB ignores readout noise and is optimistic;
        CRLB_sCMOS is attained. This is Fig. 1c,d of the paper."""
        plain, _ = self._fit(
            scmos_scene, picasso_movie_factory, "gaussmle", None
        )
        modeled, _ = self._fit(
            scmos_scene,
            picasso_movie_factory,
            "gaussmle",
            scmos_scene["calibration"],
        )
        ratio_plain = plain["x"].std() / plain["lpx"].mean()
        ratio_model = modeled["x"].std() / modeled["lpx"].mean()
        assert ratio_plain > 1.3
        assert 0.85 < ratio_model < 1.15

    def test_lsq_uncertainty_grows_but_the_fit_barely_moves(
        self, scmos_scene, picasso_movie_factory
    ):
        """Least squares does not use the noise model in its objective.

        Its reported uncertainty does grow, because the readout noise is a
        real contribution to the residual scatter. The fit itself changes only
        through the per-pixel offset and gain maps replacing the two scalars -
        the variance term cancels out of a least-squares objective exactly
        (asserted at the kernel level in ``test_gaussfit``).
        """
        plain, _ = self._fit(
            scmos_scene, picasso_movie_factory, "gausslq", None
        )
        modeled, _ = self._fit(
            scmos_scene,
            picasso_movie_factory,
            "gausslq",
            scmos_scene["calibration"],
        )
        assert modeled["lpx"].mean() > plain["lpx"].mean()
        assert modeled["x"].std() == pytest.approx(plain["x"].std(), rel=0.05)

    def test_metadata_records_provenance_not_arrays(
        self, scmos_scene, picasso_movie_factory
    ):
        _, info = self._fit(
            scmos_scene,
            picasso_movie_factory,
            "gaussmle",
            scmos_scene["calibration"],
        )
        assert info["Camera offset source"] == "per-pixel map"
        assert info["Camera calibration frames"] == 20_000
        # Nothing sensor-sized may reach the YAML sidecar.
        assert not any(
            isinstance(value, np.ndarray) for value in info.values()
        )

    def test_rejects_a_calibration_of_the_wrong_size(
        self, scmos_scene, picasso_movie_factory
    ):
        wrong = dict(scmos_scene["calibration"])
        wrong["offset"] = np.zeros((8, 8), dtype=np.float32)
        wrong["variance"] = np.ones((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="same camera ROI"):
            self._fit(scmos_scene, picasso_movie_factory, "gaussmle", wrong)

    def test_warns_when_an_em_gain_is_combined_with_a_map(
        self, scmos_scene, picasso_movie_factory
    ):
        scene = dict(scmos_scene)
        scene["camera_info"] = dict(scmos_scene["camera_info"], Gain=100)
        calibration = dict(scmos_scene["calibration"])
        calibration["gain"] = np.full_like(calibration["offset"], 2.13)
        with pytest.warns(RuntimeWarning, match="double-counts"):
            self._fit(scene, picasso_movie_factory, "gaussmle", calibration)

    def test_camera_info_with_an_array_is_refused(
        self, scmos_scene, picasso_movie_factory
    ):
        scene = dict(scmos_scene)
        scene["camera_info"] = dict(
            scmos_scene["camera_info"], Baseline=np.zeros((40, 40))
        )
        with pytest.raises(ValueError, match="camera_calibration argument"):
            self._fit(scene, picasso_movie_factory, "gaussmle", None)


class TestScmosSplineIntegration:
    """The spline models, through ``fit2D``, with a camera calibration."""

    @pytest.fixture(scope="class")
    def scene(self):
        from picasso import scmos
        from tests.test_splinefit import _flat_calibration, _reference_model
        import tests.test_splinefit as ts

        psf_calibration, terms = _flat_calibration()
        cal_box = ts.BOX
        phi = _reference_model(terms, cal_box, 0.0, 0.0, 0.0)[0]
        half = cal_box // 2

        rng = np.random.default_rng(3)
        h = w = 40
        offset = 100.0 + rng.normal(0, 1.0, (h, w))
        variance = rng.gamma(4.0, 0.4, (h, w)) + 0.4
        gain = np.full((h, w), 2.13)
        cy, cx = 18, 20
        variance[cy, cx + 2] = 1500.0
        dark = offset + rng.normal(0, np.sqrt(variance), (20_000, h, w))
        calibration = scmos.calibrate_scmos(dark)

        n_frames = 1500
        mu = np.full((h, w), 5.0)
        mu[cy - half : cy + half + 1, cx - half : cx + half + 1] += 250.0 * phi
        adu = (
            gain * rng.poisson(mu, (n_frames, h, w))
            + offset
            + rng.normal(0, np.sqrt(variance), (n_frames, h, w))
        )
        identifications = pd.DataFrame(
            {
                "frame": np.arange(n_frames),
                "x": np.full(n_frames, cx),
                "y": np.full(n_frames, cy),
                "net_gradient": np.full(n_frames, 1e4, np.float32),
            }
        )
        return {
            "movie": adu.astype(np.float32),
            "info": [{"Frames": n_frames, "Height": h, "Width": w}],
            "identifications": identifications,
            "psf_calibration": psf_calibration,
            "calibration": calibration,
            "camera_info": {
                "Baseline": float(np.median(offset)),
                "Sensitivity": 1 / 2.13,
                "Gain": 1,
                "Qe": 1.0,
                "Pixelsize": 130,
            },
            "x0": float(cx),
        }

    @staticmethod
    def _fit(scene, picasso_movie_factory, method, calibration):
        movie = picasso_movie_factory(scene["movie"], scene["info"])
        locs, _ = localize.fit2D(
            movie,
            [{}],
            dict(scene["camera_info"]),
            scene["identifications"],
            7,
            fitting_method=method,
            spline_calibration=scene["psf_calibration"],
            camera_calibration=calibration,
        )
        return locs[np.isfinite(locs["x"]) & np.isfinite(locs["lpx"])]

    def test_mle_crlb_becomes_honest(self, scene, picasso_movie_factory):
        plain = self._fit(scene, picasso_movie_factory, "spline-mle", None)
        modeled = self._fit(
            scene, picasso_movie_factory, "spline-mle", scene["calibration"]
        )
        assert plain["x"].std() / plain["lpx"].mean() > 1.15
        ratio = modeled["x"].std() / modeled["lpx"].mean()
        assert 0.85 < ratio < 1.15
        assert modeled["x"].std() < plain["x"].std()

    def test_lsq_fit_is_stable_but_its_uncertainty_grows(
        self, scene, picasso_movie_factory
    ):
        """The Huber-sandwich meat is the true pixel variance, ``mu + var``.

        So a least-squares spline fit lands in the same place but stops
        under-reporting its own scatter.
        """
        plain = self._fit(scene, picasso_movie_factory, "spline", None)
        modeled = self._fit(
            scene, picasso_movie_factory, "spline", scene["calibration"]
        )
        assert modeled["x"].std() == pytest.approx(plain["x"].std(), rel=0.05)
        assert modeled["lpx"].mean() > plain["lpx"].mean()


class TestScmosMultichannel:
    """Per-channel calibrations must follow their channel's registration.

    ``get_spots_multichannel`` maps each detection through that channel's
    affine and snaps the box to an integer pixel. A camera map has to be cut
    at *that* origin, not at the reference channel's - otherwise every
    non-reference channel's noise is read from the wrong place, off by however
    far the channels are registered apart.
    """

    BOX = 7
    H = W = 40

    @classmethod
    def _scene(cls, shift=(6, 4), n_channels=2):
        """Two channels whose variance maps carry a unique marker each.

        The markers sit exactly where each channel's box center will land, so
        a correctly cut patch puts them both at the center pixel.
        """
        cx, cy = 20, 18
        dx, dy = shift
        variance = [np.zeros((cls.H, cls.W), np.float32) for _ in range(2)]
        variance[0][cy, cx] = 111.0
        variance[1][cy + dy, cx + dx] = 222.0
        calibrations = [
            {
                "offset": np.full((cls.H, cls.W), 100.0, np.float32),
                "variance": variance[c],
            }
            for c in range(n_channels)
        ]
        movies = [
            np.full((3, cls.H, cls.W), 500, np.uint16)
            for _ in range(n_channels)
        ]
        camera_infos = [
            {"Baseline": 100.0, "Sensitivity": 1.0, "Gain": 1}
        ] * n_channels
        transforms = [
            affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            affine([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]]),
        ][:n_channels]
        identifications = pd.DataFrame(
            {
                "frame": [0],
                "x": [cx],
                "y": [cy],
                "net_gradient": np.float32([1e4]),
            }
        )
        return movies, camera_infos, transforms, identifications, calibrations

    def test_each_channel_reads_its_own_map_at_its_own_origin(self):
        movies, cams, transforms, ids, calibs = self._scene()
        _, _, variance = localize.get_spots_multichannel(
            movies,
            ids,
            self.BOX,
            cams,
            transforms,
            camera_calibrations=calibs,
            return_residuals=True,
            return_variance=True,
        )
        center = self.BOX // 2
        assert variance.shape == (1, self.BOX, self.BOX, 2)
        assert variance[0, center, center, 0] == 111.0
        assert variance[0, center, center, 1] == 222.0
        # Nothing leaks between channels.
        assert variance[..., 0].sum() == 111.0
        assert variance[..., 1].sum() == 222.0

    def test_the_reference_position_would_miss_the_shifted_map(self):
        """Guards the whole point: cutting at the reference origin is wrong.

        If this ever stops holding, the test above has become vacuous because
        the channels are no longer actually offset.
        """
        _, _, _, ids, calibs = self._scene()
        naive = localize._cut_map(calibs[1]["variance"], ids, self.BOX)
        assert naive.sum() == 0.0

    def test_a_channel_without_a_calibration_gets_zero_variance(self):
        """Zero variance is exactly the plain Poisson model for that channel."""
        movies, cams, transforms, ids, calibs = self._scene()
        _, _, variance = localize.get_spots_multichannel(
            movies,
            ids,
            self.BOX,
            cams,
            transforms,
            camera_calibrations=[calibs[0], None],
            return_residuals=True,
            return_variance=True,
        )
        assert variance[..., 0].sum() == 111.0
        assert variance[..., 1].sum() == 0.0

    def test_no_calibrations_at_all_yields_none(self):
        movies, cams, transforms, ids, _ = self._scene()
        _, _, variance = localize.get_spots_multichannel(
            movies,
            ids,
            self.BOX,
            cams,
            transforms,
            return_residuals=True,
            return_variance=True,
        )
        assert variance is None

    def test_the_legacy_return_shape_is_unchanged(self):
        """Existing callers ask for spots (+ residuals) and must keep getting
        exactly that tuple."""
        movies, cams, transforms, ids, _ = self._scene()
        spots = localize.get_spots_multichannel(
            movies, ids, self.BOX, cams, transforms
        )
        assert isinstance(spots, np.ndarray)
        pair = localize.get_spots_multichannel(
            movies, ids, self.BOX, cams, transforms, return_residuals=True
        )
        assert len(pair) == 2

    def test_rejects_a_wrong_number_of_calibrations(self):
        movies, cams, transforms, ids, calibs = self._scene()
        with pytest.raises(ValueError, match="one entry per channel"):
            localize.get_spots_multichannel(
                movies,
                ids,
                self.BOX,
                cams,
                transforms,
                camera_calibrations=calibs[:1],
            )

    def test_channel_major_reordering_keeps_channels_apart(self):
        """The CRLB kernels index ``var[m, ch, j, i]``.

        Picasso stacks multichannel patches channel-*last*, so the reordering
        has to happen for the variance exactly as it does for the spots. Left
        undone, a kernel reads another channel's readout noise.
        """
        movies, cams, transforms, ids, calibs = self._scene()
        _, _, variance = localize.get_spots_multichannel(
            movies,
            ids,
            self.BOX,
            cams,
            transforms,
            camera_calibrations=calibs,
            return_residuals=True,
            return_variance=True,
        )
        major = precision._crlb_variance_channel_major(variance, 2)
        center = self.BOX // 2
        assert major.shape == (1, 2, self.BOX, self.BOX)
        assert major[0, 0, center, center] == 111.0
        assert major[0, 1, center, center] == 222.0

    def test_single_channel_patches_still_gain_their_axis(self):
        var = np.zeros((4, self.BOX, self.BOX), np.float32)
        major = precision._crlb_variance_channel_major(var, 1)
        assert major.shape == (4, 1, self.BOX, self.BOX)
        assert precision._crlb_variance_channel_major(None, 1) is None

    def test_the_crlb_kernels_see_a_multichannel_map(self):
        """The variance must reach the spline CRLB and widen it, per channel.

        The fake calibration's coefficients put ``mu`` around 1e5, so the
        variance is deliberately large here: this pins the *plumbing* (does it
        arrive, and in the right channel plane), not a physical magnitude.
        """
        calibration = _fake_spline_calibration(
            model="spline-3d-multichannel", box=self.BOX, n_channels=2
        )
        theta = np.array([[500.0, 0.1, -0.2, 0.0, 8.0]])
        residuals = np.zeros((1, 2, 2))
        big = 1e7

        def crlb(variance):
            return precision._spline_crlb(
                theta,
                calibration,
                self.BOX,
                mle=True,
                residuals=residuals,
                variance=variance,
            )

        base = crlb(None)
        hot_ch1 = np.zeros((1, self.BOX, self.BOX, 2), np.float32)
        hot_ch1[0, 2, 4, 1] = big
        hot_ch0 = np.zeros((1, self.BOX, self.BOX, 2), np.float32)
        hot_ch0[0, 2, 4, 0] = big

        widened = crlb(hot_ch1)
        assert (widened != base).any()
        assert np.all(widened[0] >= base[0] - 1e-9)
        # Putting the same noise on the other channel is a different problem.
        # The two fake channels' coefficients differ by about one part in 1e5,
        # so the gap is small - but it must not be zero, which is what a
        # collapsed channel axis would give.
        assert crlb(hot_ch0)[0, -1] != widened[0, -1]

    @pytest.mark.parametrize("mle", [False, True])
    def test_the_multichannel_fitters_accept_a_calibration(self, mle):
        """End to end through the public entry points.

        The fake calibration cannot recover ground truth, so this asserts that
        the calibration flows all the way through and changes the answer -
        the numbers themselves are pinned by the single-channel integration
        tests, which use a real PSF.
        """
        calibration = _fake_spline_calibration(
            model="spline-3d-multichannel", box=self.BOX, n_channels=2
        )
        dx, dy = 6, 4
        calibration["channel_transforms"] = [
            IDENTITY,
            affine([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]]).to_dict(),
        ]
        rng = np.random.default_rng(4)
        n_frames = 20
        cx, cy = 20, 18
        movies = [
            rng.poisson(300, (n_frames, self.H, self.W)).astype(np.uint16)
            for _ in range(2)
        ]
        camera_infos = [
            {
                "Baseline": 100.0,
                "Sensitivity": 1.0,
                "Gain": 1,
                "Pixelsize": PIXELSIZE,
            }
        ] * 2
        identifications = pd.DataFrame(
            {
                "frame": np.arange(n_frames),
                "x": np.full(n_frames, cx),
                "y": np.full(n_frames, cy),
                "net_gradient": np.full(n_frames, 1e4, np.float32),
            }
        )
        calibrations = []
        for c in range(2):
            variance = np.full((self.H, self.W), 1.0, np.float32)
            if c == 1:
                variance[cy + dy, cx + dx + 2] = 2000.0
            calibrations.append(
                {
                    "offset": np.full((self.H, self.W), 100.0, np.float32),
                    "variance": variance,
                }
            )

        plain = localize.fit_spline_multichannel(
            movies,
            camera_infos,
            identifications,
            self.BOX,
            calibration,
            mle=mle,
            use_gpu=False,
        )
        modeled = localize.fit_spline_multichannel(
            movies,
            camera_infos,
            identifications,
            self.BOX,
            calibration,
            mle=mle,
            use_gpu=False,
            camera_calibrations=calibrations,
        )
        assert len(plain) == len(modeled) == n_frames
        assert np.isfinite(modeled["lpx"]).any()

    def test_the_ratiometric_fitter_accepts_a_calibration(self):
        calibration = _fake_spline_calibration(
            model="spline-3d-multichannel", box=self.BOX, n_channels=2
        )
        rng = np.random.default_rng(5)
        n_frames = 12
        movies = [
            rng.poisson(300, (n_frames, self.H, self.W)).astype(np.uint16)
            for _ in range(2)
        ]
        camera_infos = [
            {
                "Baseline": 100.0,
                "Sensitivity": 1.0,
                "Gain": 1,
                "Pixelsize": PIXELSIZE,
            }
        ] * 2
        identifications = pd.DataFrame(
            {
                "frame": np.arange(n_frames),
                "x": np.full(n_frames, 20),
                "y": np.full(n_frames, 18),
                "net_gradient": np.full(n_frames, 1e4, np.float32),
            }
        )
        calibrations = [
            {
                "offset": np.full((self.H, self.W), 100.0, np.float32),
                "variance": np.full((self.H, self.W), 2.0, np.float32),
            }
        ] * 2
        locs = localize.fit_spline_multichannel_ratiometric(
            movies,
            camera_infos,
            identifications,
            self.BOX,
            calibration,
            photon_ratios=np.array([[0.5, 0.5], [0.8, 0.2]]),
            mle=False,
            use_gpu=False,
            camera_calibrations=calibrations,
        )
        assert len(locs) == n_frames
        assert "color" in locs.columns

    def test_split_fov_serves_every_region_from_one_calibration(self):
        """Split-FOV is one physical sensor, so one full-frame map suffices.

        ``_cut_map`` indexes with absolute frame coordinates, so a region's
        box reads that region's own pixels out of the same map without any
        per-region bookkeeping.
        """
        calibration = _fake_spline_calibration(
            model="spline-3d-multichannel", box=self.BOX, n_channels=2
        )
        calibration["split_fov"] = True
        calibration["regions"] = [((0, 0), (40, 20)), ((0, 20), (40, 40))]
        calibration["channel_transforms"] = [
            IDENTITY,
            affine([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]]).to_dict(),
        ]
        rng = np.random.default_rng(6)
        n_frames = 12
        movie = rng.poisson(300, (n_frames, self.H, self.W)).astype(np.uint16)
        camera_info = {
            "Baseline": 100.0,
            "Sensitivity": 1.0,
            "Gain": 1,
            "Pixelsize": PIXELSIZE,
        }
        identifications = pd.DataFrame(
            {
                "frame": np.arange(n_frames),
                "x": np.full(n_frames, 8),
                "y": np.full(n_frames, 18),
                "net_gradient": np.full(n_frames, 1e4, np.float32),
            }
        )
        calibration_maps = {
            "offset": np.full((self.H, self.W), 100.0, np.float32),
            "variance": np.full((self.H, self.W), 2.0, np.float32),
        }
        locs = localize.fit_spline_split_fov(
            movie,
            camera_info,
            identifications,
            self.BOX,
            calibration,
            mle=False,
            use_gpu=False,
            camera_calibration=calibration_maps,
        )
        assert len(locs) == n_frames
        assert np.isfinite(locs["lpx"]).all()


@pytest.mark.gui
class TestCameraCalibrationConfigLookup:
    """``camera-calibrations`` in ``config.yaml``, keyed camera -> wavelength.

    The same shape as ``z-calibrations`` and ``spline-calibrations``, because
    channels are often recorded in different camera ROIs or readout modes and
    those need different maps.
    """

    @staticmethod
    def _dialog():
        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

        return localize_gui.ParametersDialog(_StubWindow())

    @staticmethod
    def _calibration_file(tmp_path, name, variance):
        path = str(tmp_path / name)
        io.save_camera_calibration(
            path,
            {
                "offset": np.full((8, 8), 100.0, np.float32),
                "variance": np.full((8, 8), variance, np.float32),
                "model": "scmos-noise",
            },
        )
        return path

    def _drive(self, dialog, monkeypatch, config, camera, wavelength):
        """Point the dialog at ``config`` and run the lookup.

        ``camera`` / ``emission_combos`` only exist once the camera-config UI
        has been built from a real ``config.yaml``, so they are stubbed here -
        the lookup itself is what is under test."""

        class _Combo:
            def __init__(self, text):
                self._text = str(text)

            def currentText(self):
                return self._text

        monkeypatch.setattr(localize_gui, "CONFIG", config)
        monkeypatch.setattr(dialog, "camera", _Combo(camera), raising=False)
        monkeypatch.setattr(
            dialog,
            "emission_combos",
            {camera: _Combo(wavelength)},
            raising=False,
        )
        dialog.update_camera_calib_with_config_path()

    def test_picks_the_file_for_the_selected_wavelength(
        self, tmp_path, monkeypatch
    ):
        green = self._calibration_file(tmp_path, "g_scmos_calib.hdf5", 2.0)
        red = self._calibration_file(tmp_path, "r_scmos_calib.hdf5", 9.0)
        config = {"camera-calibrations": {"Cam": {525: green, 595: red}}}
        dialog = self._dialog()
        try:
            self._drive(dialog, monkeypatch, config, "Cam", 525)
            assert dialog.camera_calibration_path == green
            assert dialog.camera_calibration["variance"][0, 0] == 2.0

            # Switching the emission must switch the maps.
            self._drive(dialog, monkeypatch, config, "Cam", 595)
            assert dialog.camera_calibration_path == red
            assert dialog.camera_calibration["variance"][0, 0] == 9.0
        finally:
            dialog.close()

    def test_a_bare_path_serves_every_wavelength(self, tmp_path, monkeypatch):
        """One sensor read out one way needs one set of maps; repeating the
        path under every wavelength is only a way for them to drift apart."""
        shared = self._calibration_file(tmp_path, "s_scmos_calib.hdf5", 3.0)
        config = {"camera-calibrations": {"Cam": shared}}
        dialog = self._dialog()
        try:
            for wavelength in (525, 595, 700):
                self._drive(dialog, monkeypatch, config, "Cam", wavelength)
                assert dialog.camera_calibration_path == shared
        finally:
            dialog.close()

    def test_an_unconfigured_wavelength_clears_the_calibration(
        self, tmp_path, monkeypatch
    ):
        """Leaving the previous maps loaded would apply them to a channel
        they were not measured for."""
        green = self._calibration_file(tmp_path, "g2_scmos_calib.hdf5", 2.0)
        config = {"camera-calibrations": {"Cam": {525: green}}}
        dialog = self._dialog()
        try:
            self._drive(dialog, monkeypatch, config, "Cam", 525)
            assert dialog.camera_calibration_path == green

            self._drive(dialog, monkeypatch, config, "Cam", 595)
            assert dialog.camera_calibration_path is None
            assert dialog.camera_calibration == {}
            # ... and the superseded scalar becomes editable again.
            assert dialog.baseline.isEnabled()
        finally:
            dialog.close()

    def test_an_unconfigured_camera_clears_the_calibration(
        self, tmp_path, monkeypatch
    ):
        green = self._calibration_file(tmp_path, "g3_scmos_calib.hdf5", 2.0)
        config = {"camera-calibrations": {"Cam": {525: green}}}
        dialog = self._dialog()
        try:
            self._drive(dialog, monkeypatch, config, "Cam", 525)
            assert dialog.camera_calibration_path == green
            self._drive(dialog, monkeypatch, config, "Other", 525)
            assert dialog.camera_calibration_path is None
        finally:
            dialog.close()

    def test_no_config_section_leaves_a_manual_load_alone(
        self, tmp_path, monkeypatch
    ):
        """Without the section the lookup must not touch anything."""
        manual = self._calibration_file(tmp_path, "m_scmos_calib.hdf5", 5.0)
        dialog = self._dialog()
        try:
            dialog.update_camera_calib(manual)
            self._drive(dialog, monkeypatch, {}, "Cam", 525)
            assert dialog.camera_calibration_path == manual
        finally:
            dialog.close()

    def test_the_emission_handler_runs_the_lookup(self):
        """Changing the wavelength must re-run it, as for z and spline."""
        body = inspect.getsource(
            localize_gui.ParametersDialog.on_emission_changed
        )
        assert "update_camera_calib_with_config_path" in body
        assert "update_spline_calib_with_config_path" in body
        assert "update_z_calib_with_config_path" in body


@pytest.mark.gui
class TestCameraCalibrationProvenance:
    """A saved file must say whether the noise model was used.

    ``fit2D`` records this, but Picasso Localize throws away the info
    ``fit2D`` returns and rebuilds its own when saving, so the GUI path needs
    its own assertion - without one, a GUI run silently saved localizations
    that gave no hint a calibration had been applied.
    """

    @staticmethod
    def _calibration(gain=True):
        calibration = {
            "offset": np.full((8, 8), 500.0, np.float32),
            "variance": np.full((8, 8), 4.0, np.float32),
            "model": "scmos-noise",
            "Path": "/data/cam_scmos_calib.hdf5",
            "Frames": 20000,
            "Offset median (ADU)": 500.0,
            "Variance median (ADU^2)": 4.0,
            "Hot pixels": 7,
        }
        if gain:
            calibration["gain"] = np.full((8, 8), 2.0, np.float32)
        return calibration

    def test_is_empty_without_a_calibration(self):
        assert localize.camera_calibration_info(None) == {}
        assert localize.camera_calibration_info({}) == {}

    def test_records_the_provenance_but_never_the_maps(self):
        info = localize.camera_calibration_info(self._calibration())
        assert info["Camera calibration path"] == "/data/cam_scmos_calib.hdf5"
        assert info["Camera calibration frames"] == 20000
        assert info["Camera gain source"] == "per-pixel map"
        assert info["Camera calibration Hot pixels"] == 7
        # The maps are sensor-sized and this dict is dumped to YAML verbatim.
        assert not any(
            isinstance(value, np.ndarray) for value in info.values()
        )

    def test_names_the_scalar_when_there_is_no_gain_map(self):
        info = localize.camera_calibration_info(self._calibration(gain=False))
        assert info["Camera gain source"] == "Sensitivity (scalar)"

    def test_the_gui_save_path_records_it(self, tmp_path, monkeypatch):
        """Regression: a GUI run used to save localizations that gave no hint
        a calibration had been applied, because ``Window.save_locs`` rebuilds
        the metadata from the dialog and ``FitWorker`` discards what
        ``fit2D`` returns."""
        path = str(tmp_path / "c_scmos_calib.hdf5")
        io.save_camera_calibration(path, self._calibration())

        window = localize_gui.Window()
        saved = {}
        monkeypatch.setattr(
            localize_gui.io,
            "save_locs",
            lambda p, locs, info: saved.update(info=info),
        )
        try:
            window.parameters_dialog.update_camera_calib(path)
            window.locs = pd.DataFrame({"frame": [0], "x": [1.0], "y": [2.0]})
            window.info = [{"Frames": 1}]
            window.last_identification_info = {"Box Size": 7}
            monkeypatch.setattr(window, "select_locs_columns", lambda: None)

            window.save_locs(str(tmp_path / "locs.hdf5"))

            info = saved["info"][-1]
            assert info["Camera calibration path"] == path
            assert info["Camera gain source"] == "per-pixel map"
        finally:
            window.close()

    def test_the_gui_save_path_stays_silent_without_one(
        self, tmp_path, monkeypatch
    ):
        window = localize_gui.Window()
        saved = {}
        monkeypatch.setattr(
            localize_gui.io,
            "save_locs",
            lambda p, locs, info: saved.update(info=info),
        )
        try:
            window.locs = pd.DataFrame({"frame": [0], "x": [1.0], "y": [2.0]})
            window.info = [{"Frames": 1}]
            window.last_identification_info = {"Box Size": 7}
            monkeypatch.setattr(window, "select_locs_columns", lambda: None)

            window.save_locs(str(tmp_path / "locs.hdf5"))

            assert "Camera noise model" not in saved["info"][-1]
        finally:
            window.close()


@pytest.mark.gui
class TestCameraCalibrationScalars:
    """A loaded calibration supersedes Baseline, Sensitivity and EM gain.

    Freezing them is not enough: a disabled spinbox reading 100.0 while the
    maps say 498.7 misinforms, and the value is saved to the ``.yaml`` as the
    camera information the run used.
    """

    @staticmethod
    def _dialog():
        class _StubWindow(QtWidgets.QMainWindow):
            movie = None

            def draw_frame(self):
                pass

        return localize_gui.ParametersDialog(_StubWindow())

    @staticmethod
    def _calibration_file(tmp_path, name, *, offset=498.7, gain=None):
        path = str(tmp_path / name)
        calibration = {
            "offset": np.full((8, 8), offset, np.float32),
            "variance": np.full((8, 8), 4.0, np.float32),
            "model": "scmos-noise",
        }
        if gain is not None:
            calibration["gain"] = np.full((8, 8), gain, np.float32)
        io.save_camera_calibration(path, calibration)
        return path

    def test_offset_median_becomes_the_baseline(self, tmp_path):
        path = self._calibration_file(tmp_path, "a_scmos_calib.hdf5")
        dialog = self._dialog()
        try:
            dialog.baseline.setValue(100.0)
            dialog.update_camera_calib(path)
            assert dialog.baseline.value() == pytest.approx(498.7, abs=0.05)
            assert not dialog.baseline.isEnabled()
        finally:
            dialog.close()

    def test_sensitivity_is_the_reciprocal_of_the_median_gain(self, tmp_path):
        """Picasso's Sensitivity is electrons per count; the calibration
        measures counts per electron."""
        path = self._calibration_file(tmp_path, "b_scmos_calib.hdf5", gain=2.0)
        dialog = self._dialog()
        try:
            dialog.update_camera_calib(path)
            assert dialog.sensitivity.value() == pytest.approx(0.5, abs=1e-4)
            assert not dialog.sensitivity.isEnabled()
        finally:
            dialog.close()

    def test_sensitivity_is_untouched_without_a_gain_map(self, tmp_path):
        path = self._calibration_file(tmp_path, "c_scmos_calib.hdf5")
        dialog = self._dialog()
        try:
            dialog.sensitivity.setValue(0.47)
            dialog.update_camera_calib(path)
            assert dialog.sensitivity.value() == pytest.approx(0.47)
            assert dialog.sensitivity.isEnabled()
        finally:
            dialog.close()

    def test_em_gain_is_pinned_to_one(self, tmp_path):
        """An sCMOS sensor has no multiplication stage, and a value above 1
        would apply the EMCCD excess-noise factor on top of the readout
        noise."""
        path = self._calibration_file(tmp_path, "d_scmos_calib.hdf5")
        dialog = self._dialog()
        try:
            dialog.gain.setValue(300)
            dialog.update_camera_calib(path)
            assert dialog.gain.value() == 1
            assert not dialog.gain.isEnabled()
        finally:
            dialog.close()

    def test_clearing_restores_the_previous_scalars(self, tmp_path):
        path = self._calibration_file(tmp_path, "e_scmos_calib.hdf5", gain=2.0)
        dialog = self._dialog()
        try:
            dialog.gain.setValue(300)
            dialog.baseline.setValue(100.0)
            dialog.sensitivity.setValue(0.47)
            dialog.update_camera_calib(path)
            dialog.update_camera_calib(None)
            assert dialog.gain.value() == 300
            assert dialog.baseline.value() == pytest.approx(100.0)
            assert dialog.sensitivity.value() == pytest.approx(0.47)
            assert dialog.gain.isEnabled()
            assert dialog.baseline.isEnabled()
            assert dialog.sensitivity.isEnabled()
        finally:
            dialog.close()

    def test_loading_a_second_calibration_keeps_the_original_restore_point(
        self, tmp_path
    ):
        first = self._calibration_file(tmp_path, "f1_scmos_calib.hdf5")
        second = self._calibration_file(
            tmp_path, "f2_scmos_calib.hdf5", offset=203.0
        )
        dialog = self._dialog()
        try:
            dialog.baseline.setValue(100.0)
            dialog.update_camera_calib(first)
            dialog.update_camera_calib(second)
            assert dialog.baseline.value() == pytest.approx(203.0, abs=0.05)
            dialog.update_camera_calib(None)
            assert dialog.baseline.value() == pytest.approx(100.0)
        finally:
            dialog.close()

    def test_a_superseded_scalar_records_the_config_value_for_later(
        self, tmp_path
    ):
        """The camera config keeps writing Baseline on every camera change.
        That write must not land on the frozen spinbox, but it must not be
        lost either - otherwise clearing restores a scalar belonging to
        whichever camera happened to be selected when the maps were loaded.
        """
        path = self._calibration_file(tmp_path, "g_scmos_calib.hdf5")
        dialog = self._dialog()
        try:
            dialog.baseline.setValue(100.0)
            dialog.update_camera_calib(path)
            dialog.set_photon_scalar("baseline", 42.0)
            # Not on the widget, which shows the map's median ...
            assert dialog.baseline.value() == pytest.approx(498.7, abs=0.05)
            # ... but it is what Clear restores.
            dialog.update_camera_calib(None)
            assert dialog.baseline.value() == pytest.approx(42.0)
        finally:
            dialog.close()

    def test_a_scalar_still_in_force_is_written_through(self, tmp_path):
        path = self._calibration_file(tmp_path, "h_scmos_calib.hdf5")
        dialog = self._dialog()
        try:
            dialog.update_camera_calib(path)  # no gain map
            dialog.set_photon_scalar("sensitivity", 0.26)
            assert dialog.sensitivity.value() == pytest.approx(0.26)
            dialog.update_camera_calib(None)
            assert dialog.sensitivity.value() == pytest.approx(0.26)
        finally:
            dialog.close()


@pytest.mark.gui
class TestCameraCalibrationDialogs:
    """The dialog that collects the inputs of a characterization."""

    def test_ok_needs_a_dark_movie_and_an_output(self, tmp_path):
        dialog = localize_gui.CameraCalibrationDialog(None)
        try:
            ok = dialog.buttons.button(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
            )
            assert not ok.isEnabled()
            dialog.dark_edit.setText(str(tmp_path / "dark.raw"))
            assert not ok.isEnabled()
            dialog.out_edit.setText(str(tmp_path / "calib.hdf5"))
            assert ok.isEnabled()
        finally:
            dialog.close()

    def test_bright_movies_are_optional_and_deduplicated(self):
        dialog = localize_gui.CameraCalibrationDialog(None)
        try:
            assert dialog.bright_paths() == []
            dialog.add_bright("/a.raw")
            dialog.add_bright("/b.raw")
            assert dialog.bright_paths() == ["/a.raw", "/b.raw"]
            dialog.bright_list.selectRow(0)
            dialog.remove_bright()
            assert dialog.bright_paths() == ["/b.raw"]
        finally:
            dialog.close()

    def test_illumination_levels_are_read_back(self):
        """The optional column, which only the diagnostic plot uses."""
        dialog = localize_gui.CameraCalibrationDialog(None)
        try:
            dialog.add_bright("/a.raw", "0.1")
            dialog.add_bright("/b.raw", "1")
            dialog.add_bright("/c.raw", "10")
            assert dialog.bright_levels() == [0.1, 1.0, 10.0]
        finally:
            dialog.close()

    def test_no_illumination_levels_is_not_an_error(self):
        dialog = localize_gui.CameraCalibrationDialog(None)
        try:
            dialog.add_bright("/a.raw")
            dialog.add_bright("/b.raw")
            assert dialog.bright_levels() is None
        finally:
            dialog.close()

    def test_a_partly_filled_or_non_numeric_column_is_refused(self):
        """Either is a slip; a level list that does not match the movies
        cannot be plotted against."""
        dialog = localize_gui.CameraCalibrationDialog(None)
        try:
            dialog.add_bright("/a.raw", "0.1")
            dialog.add_bright("/b.raw")
            with pytest.raises(ValueError, match="every row"):
                dialog.bright_levels()
            dialog.bright_list.item(1, 1).setText("bright")
            with pytest.raises(ValueError, match="numbers only"):
                dialog.bright_levels()
        finally:
            dialog.close()


# ---------------------------------------------------------------------------
# Channel-sum identification
# ---------------------------------------------------------------------------

IDENTITY_AFFINE = affine(np.eye(3))
UNIT_CAMERA = {"Baseline": 0.0, "Sensitivity": 1.0, "Gain": 1.0, "Qe": 1.0}


def _spots_frame(shape, spots, sigma=1.2, background=0.0):
    """A frame with a Gaussian at every ``(x, y, amplitude)`` in ``spots``."""
    yy, xx = np.indices(shape, dtype=np.float64)
    frame = np.full(shape, float(background))
    for x, y, amplitude in spots:
        frame += amplitude * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
        )
    return frame.astype(np.float32)


def _channel_pair(
    positions,
    transform,
    shape=(48, 48),
    n_frames=3,
    amplitudes=(200.0, 200.0),
    background=0.0,
):
    """Two synthetic channel movies of the same molecules, the second one
    seen through ``transform`` (reference -> channel)."""
    reference, channel = [], []
    for _ in range(n_frames):
        reference.append(
            _spots_frame(
                shape,
                [(x, y, amplitudes[0]) for x, y in positions],
                background=background,
            )
        )
        mapped = apply_transform(np.asarray(positions, dtype=float), transform)
        channel.append(
            _spots_frame(
                shape,
                [(x, y, amplitudes[1]) for x, y in mapped],
                background=background,
            )
        )
    return np.stack(reference), np.stack(channel)


class TestSummedChannelsMovie:
    """The identification-only view of the registered channels added up."""

    def test_shifted_channel_lands_on_the_reference_spots(self):
        positions = [(12.0, 15.0), (30.0, 22.0)]
        transform = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
        reference, channel = _channel_pair(positions, transform)
        summed = localize.SummedChannelsMovie(
            [reference, channel],
            [IDENTITY_AFFINE, transform],
            camera_infos=[UNIT_CAMERA] * 2,
        )
        frame = summed[0]
        assert frame.shape == reference[0].shape
        assert summed.shape == (len(reference), *reference[0].shape)
        assert summed.dtype == np.float32
        for x, y in positions:
            # both channels contribute their full amplitude at the same place
            assert frame[int(y), int(x)] == pytest.approx(400.0, rel=1e-3)

    def test_mirrored_channel_is_mapped_back(self):
        positions = [(12.0, 15.0), (30.0, 22.0)]
        # flip in x about a 48 px wide frame
        transform = affine([[-1.0, 0.0, 47.0], [0.0, 1.0, 0.0]])
        reference, channel = _channel_pair(positions, transform)
        summed = localize.SummedChannelsMovie(
            [reference, channel],
            [IDENTITY_AFFINE, transform],
            camera_infos=[UNIT_CAMERA] * 2,
        )
        frame = summed[0]
        for x, y in positions:
            assert frame[int(y), int(x)] == pytest.approx(400.0, rel=1e-3)

    def test_reference_channel_is_not_resampled(self):
        positions = [(12.5, 15.5)]
        transform = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
        reference, channel = _channel_pair(positions, transform)
        summed = localize.SummedChannelsMovie(
            [reference, np.zeros_like(channel)],
            [IDENTITY_AFFINE, transform],
            camera_infos=[UNIT_CAMERA] * 2,
        )
        # the identity channel is copied straight through, interpolation and
        # all its smoothing kept out of the reference signal
        np.testing.assert_allclose(summed[0], reference[0])

    def test_photon_conversion_weights_the_channels_equally(self):
        """Two channels carrying the same photons contribute the same amount,
        whatever their baseline and sensitivity - a raw count sum would not."""
        positions = [(20.0, 20.0)]
        reference, channel = _channel_pair(
            positions, IDENTITY_AFFINE, amplitudes=(100.0, 200.0)
        )
        # channel 1 records twice the counts per photon and sits on an offset
        camera_infos = [
            {"Baseline": 0.0, "Sensitivity": 1.0, "Gain": 1.0, "Qe": 1.0},
            {"Baseline": 500.0, "Sensitivity": 0.5, "Gain": 1.0, "Qe": 1.0},
        ]
        summed = localize.SummedChannelsMovie(
            [reference, channel + 500.0],
            [IDENTITY_AFFINE, IDENTITY_AFFINE],
            camera_infos=camera_infos,
        )
        assert summed[0][20, 20] == pytest.approx(200.0, rel=1e-4)
        # without camera infos the raw counts are summed, and the brighter
        # camera dominates by its gain and drags its baseline along
        raw = localize.SummedChannelsMovie(
            [reference, channel + 500.0],
            [IDENTITY_AFFINE, IDENTITY_AFFINE],
        )
        assert raw[0][20, 20] == pytest.approx(800.0, rel=1e-4)

    def test_split_fov_fills_only_the_reference_region(self):
        regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        # the same molecule in both halves of one frame
        transform = affine([[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]])
        positions = [(12.0, 15.0), (30.0, 22.0)]
        frames = []
        for _ in range(3):
            spots = [(x, y, 200.0) for x, y in positions]
            spots += [(x + 48.0, y, 200.0) for x, y in positions]
            frames.append(_spots_frame((48, 96), spots))
        movie = np.stack(frames)
        summed = localize.SummedChannelsMovie(
            [movie, movie],
            [IDENTITY_AFFINE, transform],
            camera_infos=[UNIT_CAMERA] * 2,
            regions=regions,
        )
        frame = summed[0]
        assert frame.shape == (48, 96)
        # the non-reference region has been mapped into the reference one
        assert np.all(frame[:, 48:] == 0)
        for x, y in positions:
            assert frame[int(y), int(x)] == pytest.approx(400.0, rel=1e-3)

    def test_unregistered_channel_is_refused(self):
        movie = np.zeros((2, 8, 8), np.float32)
        with pytest.raises(ValueError, match=r"Channel\(s\) \[1\]"):
            localize.SummedChannelsMovie(
                [movie, movie], [IDENTITY_AFFINE, None]
            )

    def test_needs_at_least_two_channels(self):
        movie = np.zeros((2, 8, 8), np.float32)
        with pytest.raises(ValueError, match="at least two channels"):
            localize.SummedChannelsMovie([movie], [IDENTITY_AFFINE])

    def test_transform_count_must_match(self):
        movie = np.zeros((2, 8, 8), np.float32)
        with pytest.raises(ValueError, match="channel transforms"):
            localize.SummedChannelsMovie(
                [movie, movie, movie], [IDENTITY_AFFINE, IDENTITY_AFFINE]
            )

    def test_channels_of_different_length_warn_and_are_truncated(self):
        long_movie = np.zeros((5, 8, 8), np.float32)
        short_movie = np.zeros((3, 8, 8), np.float32)
        with pytest.warns(UserWarning, match="different lengths"):
            summed = localize.SummedChannelsMovie(
                [long_movie, short_movie], [IDENTITY_AFFINE, IDENTITY_AFFINE]
            )
        assert len(summed) == 3


class TestIdentifyMultichannelSum:
    """Identification on the summed channels."""

    def test_finds_a_molecule_too_dim_for_any_single_channel(self):
        """The regression this mode exists for: a molecule below threshold in
        every channel on its own, but above it in the sum."""
        positions = [(16.0, 20.0), (32.0, 28.0)]
        transform = affine([[1.0, 0.0, 5.0], [0.0, 1.0, -4.0]])
        reference, channel = _channel_pair(
            positions, transform, amplitudes=(300.0, 300.0), n_frames=4
        )
        # a threshold no single channel reaches (one channel's spot has a net
        # gradient of ~5000 here, the sum's ~10000)
        minimum_ng = 7000
        alone, _ = localize.identify(
            reference, minimum_ng, BOX, threaded=False
        )
        assert len(alone) == 0
        ids, info = localize.identify_multichannel_sum(
            [reference, channel],
            minimum_ng,
            BOX,
            [IDENTITY_AFFINE, transform],
            camera_infos=[UNIT_CAMERA] * 2,
            threaded=False,
        )
        assert len(ids) == 2 * len(reference)
        found = set(zip(ids["x"].tolist(), ids["y"].tolist()))
        assert found == {(16, 20), (32, 28)}
        assert info["Identification Mode"] == "sum"
        assert info["Sum Regions"] is None
        assert info["Sum Reference Channel"] == 0
        assert np.allclose(
            affine_matrix(info["Sum Channel Transforms"][1]),
            affine_matrix(transform),
        )

    def test_split_fov_detections_are_in_reference_coordinates(self):
        regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        transform = affine([[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]])
        positions = [(12.0, 15.0), (30.0, 22.0)]
        frames = []
        for _ in range(3):
            spots = [(x, y, 300.0) for x, y in positions]
            spots += [(x + 48.0, y, 300.0) for x, y in positions]
            frames.append(_spots_frame((48, 96), spots))
        movie = np.stack(frames)
        ids, info = localize.identify_multichannel_sum(
            [movie, movie],
            800,
            BOX,
            [IDENTITY_AFFINE, transform],
            camera_infos=[UNIT_CAMERA] * 2,
            regions=regions,
            threaded=False,
        )
        found = set(zip(ids["x"].tolist(), ids["y"].tolist()))
        assert found == {(12, 15), (30, 22)}
        # nothing is reported in the non-reference region
        assert ids["x"].max() < 48
        assert info["Sum Regions"] == regions

    def test_rejects_an_unregistered_channel(self):
        movie = np.zeros((3, 16, 16), np.float32)
        with pytest.raises(ValueError, match="cannot be mapped"):
            localize.identify_multichannel_sum(
                [movie, movie], 100, BOX, [IDENTITY_AFFINE, None]
            )


def _sum_mode_window(reference, channel):
    """A Localize window with two channel movies loaded and the channel-sum
    identification mode selected."""
    window = localize_gui.Window()
    dialog = window.parameters_dialog
    # a unit camera, so the summed photons are the summed counts; set before
    # the channels are built, so every channel's snapshot carries it
    dialog.baseline.setValue(0)
    dialog.sensitivity.setValue(1.0)
    dialog.gain.setValue(1)
    window._set_channels(
        [reference, channel],
        [_info("Channel 0"), _info("Channel 1")],
        ["ref.tif", "ch1.tif"],
        ["Channel 0", "Channel 1"],
    )
    dialog.identify_mode_combo.setCurrentText(localize_gui.IDENTIFY_MODE_SUM)
    return window


@pytest.mark.gui
class TestIdentifyModeParameter:
    """The 'Identify on' setting and how it reaches the identification."""

    def test_hidden_for_single_channel_data(self):
        """Hidden, like every other multichannel-only widget - not merely
        grayed out."""
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        try:
            window._update_multichannel_widgets()
            assert dialog.identify_mode_combo.isHidden()
            assert dialog.identify_mode_label.isHidden()
            assert (
                window.parameters["Identification Mode"]
                == localize_gui.IDENTIFY_MODE_SEPARATE
            )
        finally:
            window.close()

    def test_the_dialog_does_not_reflow_when_channels_are_loaded(self):
        """The complaint this guards against: the identification group used to
        shift everything below it down the moment multichannel data was
        opened, because the 'Identify on' row appeared."""
        movie = np.zeros((2, 8, 8), np.float32)
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        # widget geometry is only laid out once the dialog is shown
        dialog.show()

        def rows_y():
            QtWidgets.QApplication.processEvents()
            return (
                dialog.roi_field.geometry().y(),
                dialog.frames_edit.geometry().y(),
            )

        try:
            window._update_multichannel_widgets()
            before = rows_y()
            window._set_channels(
                [movie, movie],
                [_info("Channel 0"), _info("Channel 1")],
                ["a.tif", "b.tif"],
                ["Channel 0", "Channel 1"],
            )
            window._update_multichannel_widgets()
            assert not dialog.identify_mode_combo.isHidden()
            assert rows_y() == before
            # ... and back again when the channels are closed
            window._set_channels([movie], [_info()], ["a.tif"], ["Channel 0"])
            window._update_multichannel_widgets()
            assert dialog.identify_mode_combo.isHidden()
            assert rows_y() == before

            # the measurement is sensitive: it is the retained size that keeps
            # the rows in place, not the layout being insensitive to the row
            policy = dialog.identify_mode_combo.sizePolicy()
            policy.setRetainSizeWhenHidden(False)
            dialog.identify_mode_combo.setSizePolicy(policy)
            dialog.identify_mode_label.setSizePolicy(policy)
            assert rows_y() < before
        finally:
            dialog.close()
            window.close()

    def test_reaches_the_parameters_for_multichannel_data(self):
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            assert (
                window.parameters["Identification Mode"]
                == localize_gui.IDENTIFY_MODE_SUM
            )
            window.parameters_dialog.identify_mode_combo.setCurrentText(
                localize_gui.IDENTIFY_MODE_SEPARATE
            )
            assert (
                window.parameters["Identification Mode"]
                == localize_gui.IDENTIFY_MODE_SEPARATE
            )
        finally:
            window.close()

    def test_switching_back_to_single_channel_resets_the_mode(self):
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            window._set_channels(
                [movie], [_info()], ["ref.tif"], ["Channel 0"]
            )
            window._update_multichannel_widgets()
            assert (
                window.parameters_dialog.identify_mode_combo.currentText()
                == localize_gui.IDENTIFY_MODE_SEPARATE
            )
        finally:
            window.close()

    def test_the_sum_takes_the_shared_threshold_in_split_fov_mode(self):
        """One summed image means one threshold, not the per-region list."""
        movie = np.zeros((2, 32, 64), np.float32)
        window = localize_gui.Window()
        try:
            window._set_channels([movie], [_info()], ["a.tif"], ["Channel 0"])
            window.set_split_fov_mode(True)
            window.view.rois = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]
            window.view.roi_mngs = [1000, 2000]
            dialog = window.parameters_dialog
            dialog.identify_mode_combo.setCurrentText(
                localize_gui.IDENTIFY_MODE_SEPARATE
            )
            assert window.parameters["Min. Net Gradient"] == [1000, 2000]
            dialog.identify_mode_combo.setCurrentText(
                localize_gui.IDENTIFY_MODE_SUM
            )
            assert (
                window.parameters["Min. Net Gradient"]
                == dialog.mng_slider.value()
            )
        finally:
            window.close()


class TestChannelSumRegistration:
    """Where the channel sum takes its registration from, and what it does
    when a channel cannot be registered."""

    def test_a_loaded_calibration_is_preferred(self):
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            calibration = _fake_spline_calibration(
                model="spline-3d-multichannel"
            )
            calibration["channel_transforms"] = [
                IDENTITY,
                affine([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]]).to_dict(),
            ]
            window.parameters_dialog.spline_calibration = calibration
            transforms, regions, source = window._sum_channel_transforms(
                estimate=False
            )
            assert regions is None
            assert source == "the loaded spline calibration"
            np.testing.assert_allclose(
                affine_matrix(transforms[1]),
                [[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]],
            )
        finally:
            window.close()

    def test_a_loaded_channel_registration_is_used(self):
        """The standalone registration the multichannel 2D Gaussian fit uses
        registers the sum too, so the sum is built with the same transforms
        that fit will use rather than re-estimating them from detections."""
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            window.parameters_dialog.channel_registration_calibration = {
                "model": "channel-registration",
                "n_channels": 2,
                "channel_transforms": [
                    IDENTITY,
                    affine([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]]).to_dict(),
                ],
            }
            transforms, regions, source = window._sum_channel_transforms(
                estimate=False
            )
            assert regions is None
            assert source == "the loaded channel registration"
            np.testing.assert_allclose(
                affine_matrix(transforms[1]),
                [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]],
            )
        finally:
            window.close()

    def test_a_spline_calibration_wins_over_a_channel_registration(self):
        """A spline calibration also describes the PSF, so when both are
        loaded it is the one the fit will use."""
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            calibration = _fake_spline_calibration(
                model="spline-3d-multichannel"
            )
            calibration["channel_transforms"] = [
                IDENTITY,
                affine([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]]).to_dict(),
            ]
            window.parameters_dialog.spline_calibration = calibration
            window.parameters_dialog.channel_registration_calibration = {
                "model": "channel-registration",
                "n_channels": 2,
                "channel_transforms": [
                    IDENTITY,
                    affine([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]]).to_dict(),
                ],
            }
            transforms, _, source = window._sum_channel_transforms(
                estimate=False
            )
            assert source == "the loaded spline calibration"
            np.testing.assert_allclose(
                affine_matrix(transforms[1]),
                [[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]],
            )
        finally:
            window.close()

    def test_estimated_from_the_per_channel_identifications(self):
        """Without a calibration the channels register from their own
        detections - which is why the sum mode identifies them first."""
        transform = affine([[1.0, 0.0, 5.0], [0.0, 1.0, -4.0]])
        rng = np.random.default_rng(3)
        positions = rng.uniform(8, 40, size=(40, 2))
        frames = np.repeat(np.arange(8), 5)
        ref_xy = positions[: len(frames)]
        chan_xy = apply_transform(ref_xy, transform)
        ids_ref = pd.DataFrame(
            {
                "frame": frames,
                "x": ref_xy[:, 0],
                "y": ref_xy[:, 1],
                "net_gradient": 1.0,
            }
        )
        ids_chan = pd.DataFrame(
            {
                "frame": frames,
                "x": chan_xy[:, 0],
                "y": chan_xy[:, 1],
                "net_gradient": 1.0,
            }
        )
        movie = np.zeros((8, 48, 48), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            window.channels[0].identifications = ids_ref
            window.channels[1].identifications = ids_chan
            window.identifications = ids_ref
            transforms, regions, source = window._sum_channel_transforms(
                estimate=True
            )
            assert regions is None
            assert "per-channel identifications" in source
            np.testing.assert_allclose(
                affine_matrix(transforms[1]),
                affine_matrix(transform),
                atol=1e-6,
            )
        finally:
            window.close()

    def test_an_unregisterable_channel_is_reported_not_assumed(self):
        """A channel that cannot be registered must never fall back to the
        identity: summing it in at the wrong place would smear the sum."""
        movie = np.zeros((4, 48, 48), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            ids = pd.DataFrame(
                {
                    "frame": [0, 1, 2],
                    "x": [10.0, 20.0, 30.0],
                    "y": [10.0, 20.0, 30.0],
                    "net_gradient": 1.0,
                }
            )
            window.channels[0].identifications = ids
            window.channels[1].identifications = None
            window.identifications = ids
            transforms, _, source = window._sum_channel_transforms(
                estimate=True
            )
            assert transforms is None
            assert source == ""
        finally:
            window.close()


@pytest.mark.gui
class TestChannelSumState:
    """The summed view behind the display, the preview and the fit."""

    def _windowed_sum(self):
        positions = [(12.0, 15.0), (30.0, 22.0)]
        transform = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
        reference, channel = _channel_pair(positions, transform)
        window = _sum_mode_window(reference, channel)
        window._run_sum_identification(
            [IDENTITY_AFFINE, transform], None, "a test"
        )
        window._active_worker.wait()
        return window, positions

    def test_the_display_and_the_preview_run_on_the_sum(self):
        window, positions = self._windowed_sum()
        try:
            frame = window.identification_movie()[0]
            for x, y in positions:
                # both channels' photons, in reference coordinates
                assert frame[int(y), int(x)] == pytest.approx(400.0, rel=1e-3)
        finally:
            window.close()

    def test_the_sum_is_dropped_when_the_layout_changes(self):
        window, _ = self._windowed_sum()
        try:
            assert window._sum_movie is not None
            movie = np.zeros((2, 8, 8), np.float32)
            window._set_channels(
                [movie, movie],
                [_info("Channel 0"), _info("Channel 1")],
                ["a.tif", "b.tif"],
                ["Channel 0", "Channel 1"],
            )
            window.validate_channel_sum()
            assert window._sum_movie is None
            assert window.sum_identifications is None
        finally:
            window.close()

    def test_split_fov_identifies_the_reference_region_only(self):
        movie = np.zeros((2, 32, 64), np.float32)
        window = localize_gui.Window()
        try:
            window._set_channels([movie], [_info()], ["a.tif"], ["Channel 0"])
            window.set_split_fov_mode(True)
            window.view.rois = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]
            window.parameters_dialog.identify_mode_combo.setCurrentText(
                localize_gui.IDENTIFY_MODE_SUM
            )
            transform = affine([[1.0, 0.0, 32.0], [0.0, 1.0, 0.0]])
            window._run_sum_identification(
                [IDENTITY_AFFINE, transform],
                [[[0, 0], [32, 32]], [[0, 32], [32, 64]]],
                "a test",
            )
            window._active_worker.wait()
            assert window.identification_rois() == [[[0, 0], [32, 32]]]
            # the drawn regions still describe the sum, so it survives
            window.validate_channel_sum()
            assert window._sum_movie is not None
            # nudging a region does not: the sum was built at the old place
            window.view.rois[1] = [[0, 30], [32, 62]]
            window.validate_channel_sum()
            assert window._sum_movie is None
        finally:
            window.close()


@pytest.mark.gui
class TestChannelSumPreview:
    """The summed view is on screen as soon as the mode is selected, so the
    display and the identification preview show the image the identification
    will search - without having to identify first."""

    def _reselect_sum_mode(self, window):
        """Re-pick 'Sum of channels', as the user would after loading a
        calibration or identifying the channels."""
        combo = window.parameters_dialog.identify_mode_combo
        combo.setCurrentText(localize_gui.IDENTIFY_MODE_SEPARATE)
        combo.setCurrentText(localize_gui.IDENTIFY_MODE_SUM)

    def test_a_calibration_puts_the_sum_up_without_identifying(self):
        """The complaint this guards against: the summed movie only appeared
        once identification had run, which made the preview useless."""
        positions = [(12.0, 15.0), (30.0, 22.0)]
        transform = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
        reference, channel = _channel_pair(positions, transform)
        window = _sum_mode_window(reference, channel)
        try:
            calibration = _fake_spline_calibration(
                model="spline-3d-multichannel"
            )
            calibration["channel_transforms"] = [
                IDENTITY,
                transform.to_dict(),
            ]
            window.parameters_dialog.spline_calibration = calibration
            self._reselect_sum_mode(window)
            assert window._sum_movie is not None
            # nothing was identified to get there
            assert window.sum_identifications is None
            assert window.identifications is None
            frame = window.identification_movie()[0]
            for x, y in positions:
                # both channels' photons, in reference coordinates
                assert frame[int(y), int(x)] == pytest.approx(400.0, rel=1e-3)
        finally:
            window.close()

    def test_the_per_channel_identifications_register_the_preview(self):
        """Without a calibration the detections already made are enough: only
        a layout that cannot be registered at all has to wait for Identify."""
        transform = affine([[1.0, 0.0, 5.0], [0.0, 1.0, -4.0]])
        rng = np.random.default_rng(3)
        frames = np.repeat(np.arange(8), 5)
        ref_xy = rng.uniform(8, 40, size=(len(frames), 2))
        chan_xy = apply_transform(ref_xy, transform)
        ids = [
            pd.DataFrame(
                {
                    "frame": frames,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "net_gradient": 1.0,
                }
            )
            for xy in (ref_xy, chan_xy)
        ]
        movie = np.zeros((8, 48, 48), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            window.channels[0].identifications = ids[0]
            window.channels[1].identifications = ids[1]
            window.identifications = ids[0]
            self._reselect_sum_mode(window)
            assert window._sum_movie is not None
            assert "per-channel identifications" in window.sum_transform_source
            np.testing.assert_allclose(
                affine_matrix(window.sum_transforms[1]),
                affine_matrix(transform),
                atol=1e-6,
            )
        finally:
            window.close()

    def test_unregisterable_channels_keep_the_raw_movie_and_are_not_retried(
        self,
    ):
        """Nothing to register the channels with: the display stays on the raw
        movie, and the attempt is not repeated on every redraw."""
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        attempts = []
        original = window._sum_channel_transforms

        def counted(*args, **kwargs):
            attempts.append(1)
            return original(*args, **kwargs)

        window._sum_channel_transforms = counted
        try:
            # selecting the mode already tried once and failed
            window.drop_channel_sum()  # ... and a drop lets it try again
            assert window.ensure_channel_sum() is False
            assert window._sum_movie is None
            # the display falls back to the raw movie, unfiltered
            assert window.identification_movie() is window.movie
            assert window.identification_movie() is window.movie
            assert len(attempts) == 1
            # ... until something that could make it possible happens
            window.drop_channel_sum()
            assert window.ensure_channel_sum() is False
            assert len(attempts) == 2
        finally:
            window.close()

    def test_identifying_searches_the_previewed_sum(self):
        """The identification takes the summed view already on screen rather
        than registering the channels a second time."""
        positions = [(12.0, 15.0), (30.0, 22.0)]
        transform = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
        reference, channel = _channel_pair(positions, transform)
        window = _sum_mode_window(reference, channel)
        try:
            window._build_channel_sum(
                [IDENTITY_AFFINE, transform], None, "a test"
            )
            previewed = window._sum_movie
            window.identify_channel_sum()
            window._active_worker.wait()
            QtWidgets.QApplication.processEvents()
            assert window._sum_movie is previewed
            assert window.sum_transform_source == "a test"
            found = set(
                zip(
                    window.identifications["x"].tolist(),
                    window.identifications["y"].tolist(),
                )
            )
            assert found == {(12, 15), (30, 22)}
        finally:
            window.close()

    def test_a_loaded_calibration_refreshes_the_summed_view(self):
        """Loading (or clearing) a calibration replaces the registration the
        sum was built with, so the view has to be built again."""
        source = inspect.getsource(
            localize_gui.ParametersDialog.update_spline_calib
        )
        assert "self.window.drop_channel_sum()" in source


@pytest.mark.gui
class TestSumIdentificationsSkipLinking:
    """Identifications made on the sum go into the joint fit as they are."""

    def _worker(self, link_identifications):
        ids_ref = pd.DataFrame({"frame": [0, 0], "x": [10, 20], "y": [10, 20]})
        # channel 1 only saw one of the two molecules
        ids_chan = pd.DataFrame({"frame": [0], "x": [10], "y": [10]})
        calibration = _fake_spline_calibration(model="spline-3d-multichannel")
        movie = np.zeros((1, 32, 32), np.float32)
        return localize_gui.MultichannelSplineFitWorker(
            [movie, movie],
            [CAMERA_INFO, CAMERA_INFO],
            ids_ref,
            BOX,
            calibration,
            identifications_per_channel=[ids_ref, ids_chan],
            link_identifications=link_identifications,
        )

    def test_linking_drops_the_unmatched_molecule_by_default(self):
        worker = self._worker(True)
        assert worker._link_across_channels(2)
        assert len(worker.identifications) == 1

    def test_the_channel_sum_keeps_every_detection(self):
        worker = self._worker(False)
        assert worker._link_across_channels(2)
        assert len(worker.identifications) == 2


class TestEstimateTransformsDiagnostics:
    """``spline.estimate_transforms_from_identifications`` reporting how well
    each channel registered - what tells the channel sum which channel it
    could not place."""

    def _identifications(self, transform, n_frames=8, per_frame=5, seed=4):
        rng = np.random.default_rng(seed)
        frames = np.repeat(np.arange(n_frames), per_frame)
        ref_xy = rng.uniform(8, 40, size=(len(frames), 2))
        chan_xy = apply_transform(ref_xy, transform)
        make = lambda xy: pd.DataFrame(  # noqa: E731
            {
                "frame": frames,
                "x": xy[:, 0],
                "y": xy[:, 1],
                "net_gradient": 1.0,
            }
        )
        return [make(ref_xy), make(chan_xy)]

    def test_pair_counts_are_returned_alongside_the_transforms(self):
        transform = affine([[1.0, 0.0, 5.0], [0.0, 1.0, -4.0]])
        ids = self._identifications(transform)
        transforms, n_pairs = spline.estimate_transforms_from_identifications(
            ids, BOX, frame_shape=(48, 48), return_diagnostics=True
        )
        np.testing.assert_allclose(
            affine_matrix(transforms[1]), affine_matrix(transform), atol=1e-6
        )
        assert n_pairs[0] == 0  # the reference registers against itself
        assert n_pairs[1] >= 10

    def test_a_failed_channel_reports_zero_pairs(self):
        ids = self._identifications(IDENTITY_AFFINE)
        ids[1] = None
        transforms, n_pairs = spline.estimate_transforms_from_identifications(
            ids, BOX, frame_shape=(48, 48), return_diagnostics=True
        )
        assert transforms is None
        assert n_pairs == [0, 0]

    def test_the_default_return_value_is_unchanged(self):
        transform = affine([[1.0, 0.0, 5.0], [0.0, 1.0, -4.0]])
        ids = self._identifications(transform)
        transforms = spline.estimate_transforms_from_identifications(
            ids, BOX, frame_shape=(48, 48)
        )
        assert isinstance(transforms, list)
        np.testing.assert_allclose(
            affine_matrix(transforms[1]), affine_matrix(transform), atol=1e-6
        )


@pytest.mark.gui
class TestChannelSumAfterRealignment:
    """``Calibration > Re-align channels (current signal)`` updates the loaded
    calibration in place, so the channel sum has to follow it."""

    def _window_with_calibration(self):
        movie = np.zeros((2, 8, 8), np.float32)
        window = _sum_mode_window(movie, movie)
        calibration = _fake_spline_calibration(model="spline-3d-multichannel")
        calibration["channel_transforms"] = [
            IDENTITY,
            affine([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]]).to_dict(),
        ]
        window.parameters_dialog.spline_calibration = calibration
        return window, calibration

    def test_the_refined_transforms_are_the_ones_summed(self):
        window, calibration = self._window_with_calibration()
        try:
            # what a re-alignment does: mutate the loaded calibration
            calibration["channel_transforms"][1] = affine(
                [[1.0, 0.0, 7.4], [0.0, 1.0, -2.6]]
            ).to_dict()
            transforms, _, source = window._sum_channel_transforms(
                estimate=False
            )
            assert source == "the loaded spline calibration"
            np.testing.assert_allclose(
                affine_matrix(transforms[1]),
                [[1.0, 0.0, 7.4], [0.0, 1.0, -2.6]],
            )
        finally:
            window.close()

    def test_a_sum_built_before_the_realignment_is_dropped(self):
        """The layout need not change when the channels are re-aligned, so the
        stale sum has to be dropped explicitly - otherwise the display, the
        preview and the fit would keep using the old registration."""
        positions = [(12.0, 15.0)]
        transform = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])
        reference, channel = _channel_pair(positions, transform)
        window = _sum_mode_window(reference, channel)
        try:
            window._run_sum_identification(
                [IDENTITY_AFFINE, transform], None, "a test"
            )
            window._active_worker.wait()
            window.sum_identifications = pd.DataFrame(
                {"frame": [0], "x": [12], "y": [15]}
            )
            assert window._sum_movie is not None
            # the layout is untouched, so the layout check alone keeps it
            window.validate_channel_sum()
            assert window._sum_movie is not None
            # a re-alignment must drop it anyway
            window.drop_channel_sum()
            assert window._sum_movie is None
            assert window.sum_identifications is None
        finally:
            window.close()

    def test_the_realignment_drops_the_channel_sum(self):
        """The real call path: re-aligning invalidates the sum."""
        source = inspect.getsource(
            localize_gui.Window.reregister_channels_from_signal
        )
        assert "self.drop_channel_sum()" in source


# the plain (unlinked) identification preview color
_RED = "#ff0000"


def _movie_info(movie, name="Channel 0"):
    """Metadata matching a synthetic movie, as loading one produces."""
    return [
        {
            "Frames": len(movie),
            "Height": int(movie.shape[1]),
            "Width": int(movie.shape[2]),
            "Channel": name,
        }
    ]


def _two_channel_window(reference, channel, box=BOX, mng=800):
    """A Localize window with two channel movies loaded, identified with a
    unit camera and the given box size / threshold, shared across both
    channels ('Same settings across channels')."""
    window = localize_gui.Window()
    dialog = window.parameters_dialog
    dialog.baseline.setValue(0)
    dialog.sensitivity.setValue(1.0)
    dialog.gain.setValue(1)
    window._set_channels(
        [reference, channel],
        [
            _movie_info(reference, "Channel 0"),
            _movie_info(channel, "Channel 1"),
        ],
        ["ref.tif", "ch1.tif"],
        ["Channel 0", "Channel 1"],
    )
    dialog.link_box_checkbox.setChecked(True)
    dialog.link_mng_checkbox.setChecked(True)
    dialog.box_spinbox.setValue(box)
    dialog.mng_slider.setValue(mng)
    return window


def _preview_box_colors(window):
    """Draw the frame and return the identification box colors in the order
    they were added."""
    window.draw_frame()
    return [
        item.pen().color().name()
        for item in window.scene.items()[::-1]
        if isinstance(item, QtWidgets.QGraphicsRectItem)
    ]


@pytest.mark.gui
class TestPreviewLinkColors:
    """'Link colors' on top of the identification preview: the cross-channel
    links are shown for the displayed frame before anything is identified, so
    the registration can be judged while the settings are still being tuned."""

    POSITIONS = [(12.0, 15.0), (30.0, 22.0), (20.0, 36.0)]
    TRANSFORM = affine([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]])

    def _window(self, positions=None, transform=None, **kwargs):
        reference, channel = _channel_pair(
            self.POSITIONS if positions is None else positions,
            self.TRANSFORM if transform is None else transform,
            amplitudes=(300.0, 300.0),
            **kwargs,
        )
        window = _two_channel_window(reference, channel)
        dialog = window.parameters_dialog
        dialog.preview_checkbox.setChecked(True)
        dialog.link_colors_checkbox.setChecked(True)
        return window

    def _with_calibration(self, window, transform=None):
        calibration = _fake_spline_calibration(model="spline-3d-multichannel")
        calibration["channel_transforms"] = [
            IDENTITY_AFFINE.to_dict(),
            (self.TRANSFORM if transform is None else transform).to_dict(),
        ]
        window.parameters_dialog.spline_calibration = calibration
        return window

    def test_the_links_are_drawn_before_anything_is_identified(self):
        """The request this covers: seeing the linked identifications in the
        preview rather than only after a full identification run."""
        window = self._with_calibration(self._window())
        try:
            assert window.identifications is None
            assert not window.ready_for_fit
            colors = _preview_box_colors(window)
            assert len(colors) == len(self.POSITIONS)
            # every spot is in both channels, so none of them stays gray ...
            assert localize_gui.LINK_UNMATCHED_COLOR.name() not in colors
            # ... and none is drawn in the plain preview red
            assert _RED not in colors
            assert len(set(colors)) == len(self.POSITIONS)
            message = window.status_bar.currentMessage()
            assert "3 of 3 linked across all channels" in message
        finally:
            window.close()

    def test_the_colors_follow_a_channel_switch(self):
        """The second channel is drawn in its own coordinates, and a spot
        keeps the color of the reference spot it pairs with."""
        window = self._with_calibration(self._window())
        try:
            reference_colors = _preview_box_colors(window)
            window.set_current_channel(1)
            colors = _preview_box_colors(window)
            assert colors == reference_colors
            boxes = [
                item.rect()
                for item in window.scene.items()[::-1]
                if isinstance(item, QtWidgets.QGraphicsRectItem)
            ]
            mapped = apply_transform(
                np.asarray(self.POSITIONS, dtype=float), self.TRANSFORM
            )
            half = int(window.parameters["Box Size"] / 2)
            assert sorted(
                (rect.x() + half, rect.y() + half) for rect in boxes
            ) == sorted((round(x), round(y)) for x, y in mapped.tolist())
        finally:
            window.close()

    def test_a_spot_missing_from_the_other_channel_stays_gray(self):
        window = self._with_calibration(
            self._window(positions=self.POSITIONS[:1])
        )
        try:
            # a spot the second channel does not have: it cannot link
            window.channels[0].movie = _channel_pair(
                self.POSITIONS[:2], self.TRANSFORM, amplitudes=(300.0, 300.0)
            )[0]
            window.movie = window.channels[0].movie
            colors = _preview_box_colors(window)
            assert len(colors) == 2
            assert colors.count(localize_gui.LINK_UNMATCHED_COLOR.name()) == 1
            assert "1 of 2 linked across all channels" in (
                window.status_bar.currentMessage()
            )
        finally:
            window.close()

    def test_the_other_channels_frame_is_identified_for_it(self):
        """The displayed channel is the only one the preview searches, so the
        others have to be searched here - in their own coordinates."""
        window = self._with_calibration(self._window())
        try:
            ids = window.channel_frame_identifications(1)
            mapped = apply_transform(
                np.asarray(self.POSITIONS, dtype=float), self.TRANSFORM
            )
            found = sorted(zip(ids["x"].tolist(), ids["y"].tolist()))
            assert found == sorted(
                (round(x), round(y)) for x, y in mapped.tolist()
            )
            assert set(ids["frame"]) == {window.curr_frame_number}
        finally:
            window.close()

    def test_the_other_channels_detections_are_cached(self):
        """The preview redraws on every scroll and every parameter change;
        identifying the other channels' frames again each time is what makes
        it too slow to leave on."""
        window = self._with_calibration(self._window())
        try:
            first = window.channel_frame_identifications(1)
            assert window.channel_frame_identifications(1) is first
            # ... but a setting the detections depend on invalidates it
            window.parameters_dialog.mng_slider.setValue(1500)
            assert window.channel_frame_identifications(1) is not first
        finally:
            window.close()

    def test_the_channels_are_registered_from_the_preview_itself(self):
        """No calibration loaded: the transform is estimated from the frame's
        own detections, mirror orientations included."""
        rng = np.random.default_rng(7)
        positions = [
            tuple(xy) for xy in rng.uniform(8.0, 40.0, size=(14, 2)).round(0)
        ]
        window = self._window(positions=positions)
        try:
            colors = _preview_box_colors(window)
            # a couple of the random spots fall too close to resolve
            assert len(colors) >= len(positions) - 2
            gray = localize_gui.LINK_UNMATCHED_COLOR.name()
            assert sum(color != gray for color in colors) >= 10
        finally:
            window.close()

    def test_plain_red_boxes_without_link_colors(self):
        window = self._with_calibration(self._window())
        try:
            window.parameters_dialog.link_colors_checkbox.setChecked(False)
            colors = _preview_box_colors(window)
            assert colors == [_RED] * len(self.POSITIONS)
            assert window.status_bar.currentMessage() == (
                "Found 3 spots in current frame."
            )
        finally:
            window.close()

    def test_nothing_to_link_on_the_channel_sum(self):
        """The sum is searched as one image, so its detections are
        cross-channel spots already."""
        window = self._with_calibration(self._window())
        try:
            window.parameters_dialog.identify_mode_combo.setCurrentText(
                localize_gui.IDENTIFY_MODE_SUM
            )
            assert window._preview_link_identifications(pd.DataFrame()) is None
        finally:
            window.close()

    def test_an_off_screen_channel_keeps_its_own_threshold(self):
        window = self._with_calibration(self._window())
        dialog = window.parameters_dialog
        try:
            dialog.link_mng_checkbox.setChecked(False)
            window.channels[1].params["mng"] = 4321
            assert window.channel_parameters(1)["Min. Net Gradient"] == 4321
            # ... unless the threshold is shared across channels
            dialog.link_mng_checkbox.setChecked(True)
            assert window.channel_parameters(1)["Min. Net Gradient"] == (
                dialog.mng_slider.value()
            )
        finally:
            window.close()

    def test_loading_other_channels_drops_the_cached_detections(self):
        window = self._with_calibration(self._window())
        try:
            window.channel_frame_identifications(1)
            assert window._preview_ids_cache
            movie = np.zeros((2, 16, 16), np.float32)
            window._set_channels(
                [movie, movie],
                [_movie_info(movie), _movie_info(movie, "Channel 1")],
                ["a.tif", "b.tif"],
                ["Channel 0", "Channel 1"],
            )
            # the stale detections are gone: these movies are empty, and a
            # redraw has already cached that
            assert len(window.channel_frame_identifications(1)) == 0
        finally:
            window.close()


@pytest.mark.gui
class TestPreviewLinkColorsSplitFov:
    """The same on split-FOV data, where the channels are regions of the one
    displayed frame - the preview already searches all of them."""

    REGIONS = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    # spaced well apart, so a region shifted by more than the pairing
    # tolerance really has nothing to pair with
    POSITIONS = [(10.0, 8.0), (34.0, 8.0), (10.0, 32.0)]

    def _window(self, shift=(48.0, 0.0)):
        """One movie holding both regions; the right region is the left one
        shifted by ``shift``."""
        spots = [(x, y, 300.0) for x, y in self.POSITIONS]
        spots += [
            (x + shift[0], y + shift[1], 300.0) for x, y in self.POSITIONS
        ]
        movie = np.stack([_spots_frame((48, 96), spots)] * 2)
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        dialog.baseline.setValue(0)
        dialog.sensitivity.setValue(1.0)
        dialog.gain.setValue(1)
        window._set_channels(
            [movie], [_movie_info(movie)], ["split.tif"], ["Channel 0"]
        )
        window.view.rois = [[r[:] for r in region] for region in self.REGIONS]
        window.set_split_fov_mode(True)
        dialog.box_spinbox.setValue(BOX)
        dialog.mng_slider.setValue(800)
        calibration = _fake_spline_calibration(model="spline-3d-multichannel")
        calibration["split_fov"] = True
        calibration["regions"] = self.REGIONS
        calibration["channel_registration"] = [IDENTITY, IDENTITY]
        dialog.spline_calibration = calibration
        dialog.preview_checkbox.setChecked(True)
        dialog.link_colors_checkbox.setChecked(True)
        return window

    def test_the_regions_are_linked_in_the_preview(self):
        window = self._window()
        try:
            assert window.identifications is None
            colors = _preview_box_colors(window)
            # the two region rectangles are drawn as boxes too
            boxes = colors[len(self.REGIONS) :]
            assert len(boxes) == 2 * len(self.POSITIONS)
            assert localize_gui.LINK_UNMATCHED_COLOR.name() not in boxes
            # each spot has the same color in both regions, and the spots
            # differ from one another
            assert boxes[: len(self.POSITIONS)] == boxes[len(self.POSITIONS) :]
            assert len(set(boxes)) == len(self.POSITIONS)
            assert "6 of 6 linked across all regions" in (
                window.status_bar.currentMessage()
            )
        finally:
            window.close()

    def test_a_misregistered_region_stays_gray(self):
        """The registration is the calibration's, so a shift it does not
        know about shows up as unlinked spots rather than being absorbed."""
        window = self._window(shift=(48.0, 12.0))
        try:
            boxes = _preview_box_colors(window)[len(self.REGIONS) :]
            gray = localize_gui.LINK_UNMATCHED_COLOR.name()
            assert len(boxes) > len(self.POSITIONS)
            assert boxes == [gray] * len(boxes)
        finally:
            window.close()


class TestFitGaussMultichannel:
    """End to end through ``localize.fit_gauss_multichannel``: two registered
    channels, a known sub-pixel inter-channel offset, and real ground truth."""

    H = W = 48
    BOX = 9
    N_FRAMES = 60
    SIGMA = 1.3
    BASELINE = 100.0
    # deliberately fractional, so a fit that ignored the sub-pixel ROI
    # residual would be visibly biased
    DX, DY = 7.35, -5.6

    def _dataset(self, amps=(900.0, 340.0), bgs=(12.0, 7.0), seed=2):
        rng = np.random.RandomState(seed)
        truth_x = rng.uniform(16, 32, self.N_FRAMES)
        truth_y = rng.uniform(16, 32, self.N_FRAMES)
        j, i = np.mgrid[0 : self.H, 0 : self.W]
        clean = [np.zeros((self.N_FRAMES, self.H, self.W)) for _ in amps]
        for f in range(self.N_FRAMES):
            for c, amp in enumerate(amps):
                x = truth_x[f] + (self.DX if c else 0.0)
                y = truth_y[f] + (self.DY if c else 0.0)
                clean[c][f] += amp * np.exp(
                    -0.5 * ((i - x) ** 2 + (j - y) ** 2) / self.SIGMA**2
                )
        movies = [
            (rng.poisson(c + b) + self.BASELINE).astype(np.uint16)
            for c, b in zip(clean, bgs)
        ]
        camera_infos = [
            {
                "Baseline": self.BASELINE,
                "Sensitivity": 1.0,
                "Gain": 1,
                "Pixelsize": 130,
            }
        ] * len(amps)
        identifications = pd.DataFrame(
            {
                "frame": np.arange(self.N_FRAMES),
                "x": np.rint(truth_x).astype(np.int64),
                "y": np.rint(truth_y).astype(np.int64),
                "net_gradient": np.full(self.N_FRAMES, 1e4, np.float32),
            }
        )
        return movies, camera_infos, identifications, truth_x, truth_y

    def _registration(self, dx=None, dy=None, n_channels=2):
        dx = self.DX if dx is None else dx
        dy = self.DY if dy is None else dy
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
        return {
            "model": "channel-registration",
            "n_channels": n_channels,
            "channel_transforms": [
                transforms.identity().to_dict(),
                transforms.AffineTransform(matrix=matrix).to_dict(),
            ],
        }

    @pytest.mark.parametrize("mle", [False, True], ids=["lsq", "mle"])
    @pytest.mark.parametrize(
        "link_photons", [False, True], ids=["decoupled", "linked"]
    )
    def test_recovers_ground_truth(self, mle, link_photons):
        movies, camera_infos, ids, truth_x, truth_y = self._dataset()

        locs = localize.fit_gauss_multichannel(
            movies,
            camera_infos,
            ids,
            self.BOX,
            self._registration(),
            mle=mle,
            link_photons=link_photons,
            use_gpu=False,
            multiprocess=False,
        )

        assert len(locs) == self.N_FRAMES
        frame = locs["frame"].to_numpy()
        error_x = locs["x"].to_numpy() - truth_x[frame]
        error_y = locs["y"].to_numpy() - truth_y[frame]
        # the positions are in the reference channel's frame, and unbiased
        # despite the fractional inter-channel offset
        assert abs(np.nanmean(error_x)) < 0.05
        assert abs(np.nanmean(error_y)) < 0.05
        assert np.nanstd(error_x) < 0.1
        assert np.nanstd(error_y) < 0.1
        # a spherical fit has one width, reported in both columns
        np.testing.assert_allclose(locs["sx"], locs["sy"])
        assert "ellipticity" not in locs.columns

    def test_reported_precision_matches_the_scatter(self):
        """``lpx``/``lpy`` are the bound the MLE actually attains."""
        movies, camera_infos, ids, truth_x, truth_y = self._dataset()

        locs = localize.fit_gauss_multichannel(
            movies,
            camera_infos,
            ids,
            self.BOX,
            self._registration(),
            mle=True,
            use_gpu=False,
            multiprocess=False,
        )

        frame = locs["frame"].to_numpy()
        scatter = np.nanstd(locs["x"].to_numpy() - truth_x[frame])
        assert np.nanmedian(locs["lpx"]) == pytest.approx(scatter, rel=0.4)

    def test_decoupled_model_splits_the_photons(self):
        amps = (900.0, 340.0)
        movies, camera_infos, ids, _, _ = self._dataset(amps=amps)

        locs = localize.fit_gauss_multichannel(
            movies,
            camera_infos,
            ids,
            self.BOX,
            self._registration(),
            mle=True,
            link_photons=False,
            use_gpu=False,
            multiprocess=False,
        )

        expected = [a * 2 * np.pi * self.SIGMA**2 for a in amps]
        for c, want in enumerate(expected):
            assert np.nanmedian(locs[f"photons_ch{c}"]) == pytest.approx(
                want, rel=0.1
            )
            assert f"bg_ch{c}" in locs.columns
        # the relative shares sum to one and reflect the true split
        total = sum(expected)
        assert np.nanmedian(locs["rel_photons_ch0"]) == pytest.approx(
            expected[0] / total, abs=0.05
        )
        # and ``photons`` is the total across channels
        assert np.nanmedian(locs["photons"]) == pytest.approx(total, rel=0.1)

    def test_every_column_is_saveable(self):
        """Anything the fit emits must be in ``LOCALIZATION_COLUMNS`` or it is
        silently dropped on save."""
        movies, camera_infos, ids, _, _ = self._dataset()
        allowed = {
            column
            for columns in localize.LOCALIZATION_COLUMNS.values()
            for column in columns
        }

        for link_photons in (False, True):
            locs = localize.fit_gauss_multichannel(
                movies,
                camera_infos,
                ids,
                self.BOX,
                self._registration(),
                mle=True,
                link_photons=link_photons,
                use_gpu=False,
                multiprocess=False,
            )
            assert set(locs.columns) <= allowed

    def test_a_wrong_registration_is_worse_than_the_right_one(self):
        """The registration is load-bearing: pointing channel 1 at the wrong
        place must degrade the joint fit."""
        movies, camera_infos, ids, truth_x, truth_y = self._dataset()

        def bias(registration):
            locs = localize.fit_gauss_multichannel(
                movies,
                camera_infos,
                ids,
                self.BOX,
                registration,
                mle=True,
                use_gpu=False,
                multiprocess=False,
            )
            frame = locs["frame"].to_numpy()
            return abs(np.nanmean(locs["x"].to_numpy() - truth_x[frame]))

        assert bias(self._registration()) < 0.05
        assert bias(self._registration(dx=self.DX + 2.0)) > 0.2

    def test_rejects_a_channel_count_mismatch(self):
        movies, camera_infos, ids, _, _ = self._dataset()
        with pytest.raises(ValueError, match="channel transforms"):
            localize.fit_gauss_multichannel(
                movies[:1],
                camera_infos[:1],
                ids,
                self.BOX,
                self._registration(),
                use_gpu=False,
            )

    def test_rejects_a_registration_without_transforms(self):
        movies, camera_infos, ids, _, _ = self._dataset()
        with pytest.raises(ValueError, match="channel_transforms"):
            localize.fit_gauss_multichannel(
                movies, camera_infos, ids, self.BOX, {}, use_gpu=False
            )

    def test_per_channel_scmos_calibration_reaches_the_fit(self):
        """A hot pixel in one channel's variance map must change the answer,
        or the noise model is not plumbed through."""
        movies, camera_infos, ids, _, _ = self._dataset()
        registration = self._registration()

        def fit(camera_calibrations):
            return localize.fit_gauss_multichannel(
                movies,
                camera_infos,
                ids,
                self.BOX,
                registration,
                mle=True,
                use_gpu=False,
                multiprocess=False,
                camera_calibrations=camera_calibrations,
            )

        maps = []
        for c in range(2):
            variance = np.full((self.H, self.W), 1.0, np.float32)
            if c == 1:
                variance[:] = 400.0
            maps.append(
                {
                    "offset": np.full(
                        (self.H, self.W), self.BASELINE, np.float32
                    ),
                    "variance": variance,
                }
            )

        plain = fit(None)
        modeled = fit(maps)
        assert not np.allclose(
            plain["lpx"].to_numpy(), modeled["lpx"].to_numpy()
        )


@pytest.mark.gui
class TestMultichannelGaussianWorkerRouting:
    """``MultichannelGaussianFitWorker`` passes the GUI's settings through to
    ``localize.fit_gauss_multichannel`` (which is monkeypatched, so no GPU or
    real movies are needed)."""

    def _registration(self, n_channels=2):
        return {
            "model": "channel-registration",
            "n_channels": n_channels,
            "channel_transforms": [transforms.identity().to_dict()]
            * n_channels,
        }

    def _run(self, monkeypatch, **worker_kwargs):
        seen = {}
        df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0]})

        def fake_fit(*args, **kwargs):
            seen.update(kwargs)
            seen["n_movies"] = len(args[0])
            return df

        monkeypatch.setattr(localize, "fit_gauss_multichannel", fake_fit)
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelGaussianFitWorker(
            [None, None],
            [{}, {}],
            ids,
            BOX,
            self._registration(),
            **worker_kwargs,
        )
        result = {}
        worker.finished.connect(
            lambda locs, dt, a, b: result.update(locs=locs)
        )
        worker.aborted.connect(lambda: result.update(aborted=True))
        worker.run()
        return seen, result

    @pytest.mark.parametrize("link_photons", [False, True])
    def test_link_photons_reaches_the_fitter(self, monkeypatch, link_photons):
        seen, result = self._run(monkeypatch, link_photons=link_photons)
        assert seen["link_photons"] is link_photons
        assert "locs" in result

    @pytest.mark.parametrize("mle", [False, True])
    def test_the_estimator_reaches_the_fitter(self, monkeypatch, mle):
        seen, _ = self._run(monkeypatch, mle=mle)
        assert seen["mle"] is mle

    def test_the_registration_is_handed_over(self, monkeypatch):
        seen, _ = self._run(monkeypatch)
        assert seen["n_movies"] == 2

    def test_a_failed_fit_aborts_rather_than_raising(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("no")

        monkeypatch.setattr(localize, "fit_gauss_multichannel", boom)
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelGaussianFitWorker(
            [None, None], [{}, {}], ids, BOX, self._registration()
        )
        aborted = []
        worker.aborted.connect(lambda: aborted.append(True))
        worker.run()
        assert aborted == [True]

    def test_an_aborted_fit_emits_aborted(self, monkeypatch):
        monkeypatch.setattr(
            localize, "fit_gauss_multichannel", lambda *a, **k: None
        )
        ids = pd.DataFrame(
            {"frame": [0], "x": [6], "y": [6], "net_gradient": [1.0]}
        )
        worker = localize_gui.MultichannelGaussianFitWorker(
            [None, None], [{}, {}], ids, BOX, self._registration()
        )
        aborted = []
        worker.aborted.connect(lambda: aborted.append(True))
        worker.run()
        assert aborted == [True]


@pytest.mark.gui
class TestChannelRegistrationParametersDialog:
    """The channel-registration group box in the parameters dialog."""

    def test_only_shown_for_the_spherical_gaussian(self):
        window = localize_gui.Window()
        try:
            dialog = window.parameters_dialog
            dialog.fit_model.setCurrentText("2D spherical Gaussian")
            assert dialog.channel_registration_groupbox.isVisibleTo(dialog)
            assert not dialog.spline_groupbox.isVisibleTo(dialog)

            dialog.fit_model.setCurrentText("2D elliptical Gaussian")
            assert not dialog.channel_registration_groupbox.isVisibleTo(dialog)
            assert dialog.z_groupbox.isVisibleTo(dialog)

            dialog.fit_model.setCurrentText("Experimental PSF (cubic spline)")
            assert not dialog.channel_registration_groupbox.isVisibleTo(dialog)
            assert dialog.spline_groupbox.isVisibleTo(dialog)
        finally:
            window.close()

    def test_loading_a_registration_shows_the_link_photons_box(self, tmp_path):
        window = localize_gui.Window()
        try:
            dialog = window.parameters_dialog
            # the box lives in the registration group, which only the
            # spherical Gaussian shows
            dialog.fit_model.setCurrentText("2D spherical Gaussian")
            assert not dialog.gauss_link_photons_checkbox.isVisibleTo(dialog)

            path = str(tmp_path / "reg.yaml")
            io.save_any_calibration(
                path,
                {
                    "model": "channel-registration",
                    "n_channels": 2,
                    "channel_transforms": [transforms.identity().to_dict()]
                    * 2,
                },
            )
            dialog.update_channel_registration(path)

            assert dialog.channel_registration_calibration["n_channels"] == 2
            assert dialog.gauss_link_photons_checkbox.isVisibleTo(dialog)
            # decoupled is the default: a Gaussian has no per-channel
            # brightness scale, so linking would assume equal brightness
            assert dialog._gauss_link_photons_enabled() is False
        finally:
            window.close()

    def test_a_file_without_transforms_is_rejected(self, tmp_path):
        window = localize_gui.Window()
        try:
            dialog = window.parameters_dialog
            path = str(tmp_path / "not_a_registration.yaml")
            io.save_any_calibration(path, {"model": "something-else"})

            dialog.update_channel_registration(path)

            assert dialog.channel_registration_calibration == {}
            assert "not a channel registration" in (
                dialog.channel_registration_label.text()
            )
        finally:
            window.close()

    def test_clearing_forgets_the_registration(self, tmp_path):
        window = localize_gui.Window()
        try:
            dialog = window.parameters_dialog
            path = str(tmp_path / "reg.yaml")
            io.save_any_calibration(
                path,
                {
                    "model": "channel-registration",
                    "n_channels": 2,
                    "channel_transforms": [transforms.identity().to_dict()]
                    * 2,
                },
            )
            dialog.update_channel_registration(path)
            dialog.update_channel_registration(None)

            assert dialog.channel_registration_calibration == {}
            assert dialog.channel_registration_path is None
            assert not dialog.gauss_link_photons_checkbox.isVisibleTo(dialog)
        finally:
            window.close()


@pytest.mark.gui
class TestMultichannelGaussianFitThroughTheWindow:
    """The whole GUI path: two channels loaded, a registration loaded, and
    ``Window.fit()`` dispatching to the joint Gaussian fit rather than to the
    single-channel worker."""

    H = W = 40
    BOX = 9
    N_FRAMES = 12
    SIGMA = 1.3
    DX, DY = 6.4, -4.35

    def _movies(self):
        rng = np.random.RandomState(11)
        truth_x = rng.uniform(14, 26, self.N_FRAMES)
        truth_y = rng.uniform(14, 26, self.N_FRAMES)
        j, i = np.mgrid[0 : self.H, 0 : self.W]
        movies = []
        for c, amp in enumerate((900.0, 400.0)):
            clean = np.zeros((self.N_FRAMES, self.H, self.W))
            for f in range(self.N_FRAMES):
                x = truth_x[f] + (self.DX if c else 0.0)
                y = truth_y[f] + (self.DY if c else 0.0)
                clean[f] = amp * np.exp(
                    -0.5 * ((i - x) ** 2 + (j - y) ** 2) / self.SIGMA**2
                )
            movies.append(rng.poisson(clean + 10).astype(np.uint16))
        return movies, truth_x, truth_y

    def _registration_file(self, tmp_path):
        matrix = np.array(
            [[1.0, 0.0, self.DX], [0.0, 1.0, self.DY], [0.0, 0.0, 1.0]]
        )
        path = str(tmp_path / "reg.yaml")
        io.save_any_calibration(
            path,
            {
                "model": "channel-registration",
                "n_channels": 2,
                "channel_transforms": [
                    transforms.identity().to_dict(),
                    transforms.AffineTransform(matrix=matrix).to_dict(),
                ],
            },
        )
        return path

    def _window(self, tmp_path, registration=True):
        movies, truth_x, truth_y = self._movies()
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        dialog.baseline.setValue(0)
        dialog.sensitivity.setValue(1.0)
        dialog.gain.setValue(1)
        window.parameters["Box Size"] = self.BOX
        info = [{"Frames": self.N_FRAMES, "Height": self.H, "Width": self.W}]
        # under tmp_path: the fit auto-saves the localizations next to the
        # movie, and a relative path puts them in the working directory
        window._set_channels(
            movies,
            [info, info],
            [str(tmp_path / "ref.tif"), str(tmp_path / "ch1.tif")],
            ["ch0", "ch1"],
        )
        dialog.fit_model.setCurrentText("2D spherical Gaussian")
        dialog.fit_optimizer.setCurrentText("MLE")
        if registration:
            dialog.update_channel_registration(
                self._registration_file(tmp_path)
            )
        ids = pd.DataFrame(
            {
                "frame": np.arange(self.N_FRAMES),
                "x": np.rint(truth_x).astype(np.int64),
                "y": np.rint(truth_y).astype(np.int64),
                "net_gradient": np.full(self.N_FRAMES, 1e4, np.float32),
            }
        )
        for channel in window.channels:
            channel.identifications = ids
            channel.ready_for_fit = True
        window.identifications = ids
        window.ready_for_fit = True
        return window, truth_x, truth_y

    @staticmethod
    def _finish(window):
        """Run the fit and let Qt deliver the worker's queued signals.

        The worker runs on its own thread, so ``finished`` only reaches the
        window once the event loop turns."""
        window.fit()
        worker = window.fit_worker
        worker.wait()
        QtWidgets.QApplication.processEvents()
        return worker

    def test_fit_dispatches_to_the_multichannel_worker(self, tmp_path):
        window, truth_x, truth_y = self._window(tmp_path)
        try:
            worker = self._finish(window)

            assert isinstance(
                worker, localize_gui.MultichannelGaussianFitWorker
            )
            assert window.locs is not None
            frame = window.locs["frame"].to_numpy()
            error = window.locs["x"].to_numpy() - truth_x[frame]
            assert abs(np.nanmean(error)) < 0.1
            # decoupled is the default, so the per-channel columns are there
            assert "photons_ch1" in window.locs.columns
        finally:
            window.close()

    def test_without_a_registration_it_stays_single_channel(self, tmp_path):
        window, _, _ = self._window(tmp_path, registration=False)
        try:
            worker = self._finish(window)
            assert isinstance(worker, localize_gui.FitWorker)
        finally:
            window.close()

    def test_the_fit_is_distributed_to_every_channel(self, tmp_path):
        """Each channel gets the localizations mapped into its own frame, so
        switching channels overlays them on that channel's movie."""
        window, _, _ = self._window(tmp_path)
        try:
            self._finish(window)

            display = [c.locs_display for c in window.channels]
            assert all(d is not None for d in display)
            shift = np.nanmedian(
                display[1]["x"].to_numpy() - display[0]["x"].to_numpy()
            )
            assert shift == pytest.approx(self.DX, abs=0.2)
        finally:
            window.close()


@pytest.mark.gui
class TestRegisterChannelsFlow:
    """Building a channel registration from the GUI, both ways."""

    H = W = 48
    BOX = 7
    N_FRAMES = 60
    DX, DY = 5.4, -3.7

    def _blinking_movies(self):
        rng = np.random.RandomState(3)
        j, i = np.mgrid[0 : self.H, 0 : self.W]
        movies = [np.zeros((self.N_FRAMES, self.H, self.W)) for _ in range(2)]
        for f in range(self.N_FRAMES):
            for _ in range(rng.randint(5, 8)):
                x = rng.uniform(14, 34)
                y = rng.uniform(14, 34)
                amp = rng.uniform(2500, 4000)
                for c, (dx, dy) in enumerate(((0.0, 0.0), (self.DX, self.DY))):
                    movies[c][f] += amp * np.exp(
                        -0.5
                        * ((i - (x + dx)) ** 2 + (j - (y + dy)) ** 2)
                        / 1.2**2
                    )
        return [
            rng.poisson(np.maximum(m, 0) + 100).astype(np.uint16)
            for m in movies
        ]

    def test_the_worker_builds_a_loadable_registration(self, tmp_path):
        """The signal builder, run exactly as the GUI runs it."""
        path = str(tmp_path / "reg.yaml")
        worker = localize_gui.ChannelRegistrationWorker(
            source="signal",
            movies=self._blinking_movies(),
            box=self.BOX,
            minimum_ng=800.0,
            path=path,
            model="affine",
        )
        done, failed = [], []
        worker.finished.connect(done.append)
        worker.failed.connect(failed.append)

        worker.run()

        assert failed == [], failed
        assert done == [path]
        calibration = io.load_any_calibration(path)
        matrix = transforms.from_dict(
            calibration["channel_transforms"][1]
        ).matrix[:2]
        assert matrix[0, 2] == pytest.approx(self.DX, abs=0.4)
        assert matrix[1, 2] == pytest.approx(self.DY, abs=0.4)

    def test_a_failure_is_reported_rather_than_raised(self, tmp_path):
        blank = [
            np.full((10, self.H, self.W), 100, np.uint16) for _ in range(2)
        ]
        worker = localize_gui.ChannelRegistrationWorker(
            source="signal",
            movies=blank,
            box=self.BOX,
            minimum_ng=800.0,
            path=str(tmp_path / "reg.yaml"),
        )
        failed = []
        worker.failed.connect(failed.append)

        worker.run()

        assert len(failed) == 1

    def test_the_menu_action_registers_and_loads(self, tmp_path, monkeypatch):
        """``register_channels_from_signal`` builds the file and loads it into
        the parameters dialog, so a fit can use it straight away."""
        window = localize_gui.Window()
        try:
            info = [
                {"Frames": self.N_FRAMES, "Height": self.H, "Width": self.W}
            ]
            window.parameters["Box Size"] = self.BOX
            window.parameters["Min. Net Gradient"] = 800.0
            window._set_channels(
                self._blinking_movies(),
                [info, info],
                ["a.tif", "b.tif"],
                ["ch0", "ch1"],
            )
            path = str(tmp_path / "reg.yaml")
            monkeypatch.setattr(
                localize_gui.RefineRegistrationDialog,
                "getFrameSpecs",
                staticmethod(
                    lambda *a, **k: (
                        [[0, self.N_FRAMES - 1]],
                        40,
                        "affine",
                        True,
                    )
                ),
            )
            monkeypatch.setattr(
                localize_gui.Window,
                "_registration_save_path",
                lambda self: path,
            )
            shown = []
            monkeypatch.setattr(
                QtWidgets.QMessageBox,
                "information",
                lambda *a, **k: shown.append(a),
            )

            window.register_channels_from_signal()
            window.registration_worker.wait()
            QtWidgets.QApplication.processEvents()

            loaded = window.parameters_dialog.channel_registration_calibration
            assert loaded.get("n_channels") == 2
            assert window.parameters_dialog.channel_registration_path == path
            assert shown, "the user is told what was built"
        finally:
            window.close()

    def test_it_refuses_a_single_channel(self, monkeypatch):
        window = localize_gui.Window()
        try:
            shown = []
            monkeypatch.setattr(
                QtWidgets.QMessageBox,
                "information",
                lambda *a, **k: shown.append(a),
            )
            window.register_channels_from_signal()
            assert shown
        finally:
            window.close()


class TestMultichannelGaussianUncertaintyColumns:
    """Each reported uncertainty must describe its own parameter.

    Position, width and photon variances differ by orders of magnitude, so a
    column mix-up between the fit's parameter order and the CRLB's shows up
    immediately here - unlike in a symmetric precision check, where var(x) and
    var(y) are nearly equal and hide it.
    """

    BOX = 9
    N_FRAMES = 40
    SIGMA = 1.3

    def _locs(self, link_photons):
        rng = np.random.RandomState(17)
        height = width = 44
        truth_x = rng.uniform(16, 28, self.N_FRAMES)
        truth_y = rng.uniform(16, 28, self.N_FRAMES)
        j, i = np.mgrid[0:height, 0:width]
        dx, dy = 6.4, -4.3
        movies = []
        for c, amp in enumerate((900.0, 500.0)):
            clean = np.zeros((self.N_FRAMES, height, width))
            for f in range(self.N_FRAMES):
                x = truth_x[f] + (dx if c else 0.0)
                y = truth_y[f] + (dy if c else 0.0)
                clean[f] = amp * np.exp(
                    -0.5 * ((i - x) ** 2 + (j - y) ** 2) / self.SIGMA**2
                )
            movies.append((rng.poisson(clean + 10) + 100).astype(np.uint16))
        camera_infos = [{"Baseline": 100.0, "Sensitivity": 1.0, "Gain": 1}] * 2
        ids = pd.DataFrame(
            {
                "frame": np.arange(self.N_FRAMES),
                "x": np.rint(truth_x).astype(np.int64),
                "y": np.rint(truth_y).astype(np.int64),
                "net_gradient": np.full(self.N_FRAMES, 1e4, np.float32),
            }
        )
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
        registration = {
            "model": "channel-registration",
            "n_channels": 2,
            "channel_transforms": [
                transforms.identity().to_dict(),
                transforms.AffineTransform(matrix=matrix).to_dict(),
            ],
        }
        return localize.fit_gauss_multichannel(
            movies,
            camera_infos,
            ids,
            self.BOX,
            registration,
            mle=True,
            link_photons=link_photons,
            use_gpu=False,
            multiprocess=False,
        )

    @pytest.mark.parametrize(
        "link_photons", [False, True], ids=["decoupled", "linked"]
    )
    def test_each_uncertainty_is_on_its_own_scale(self, link_photons):
        locs = self._locs(link_photons)

        photons = np.nanmedian(locs["photons"])
        # a position precision is a small fraction of a pixel ...
        assert 0.0 < np.nanmedian(locs["lpx"]) < 0.2
        assert 0.0 < np.nanmedian(locs["lpy"]) < 0.2
        # ... a width uncertainty likewise ...
        assert 0.0 < np.nanmedian(locs["sx_unc"]) < 0.2
        # ... while a photon uncertainty is on the order of sqrt(photons),
        # hundreds of times larger. Reading a position variance into this
        # column (or vice versa) cannot pass both bounds.
        assert np.nanmedian(locs["photons_unc"]) == pytest.approx(
            np.sqrt(photons), rel=0.7
        )
        assert 0.0 < np.nanmedian(locs["bg_unc"]) < 10.0

    def test_the_width_uncertainty_matches_the_width_scatter(self):
        """``sx_unc`` must track the width, not the photon count."""
        locs = self._locs(link_photons=False)
        assert np.nanmedian(locs["sx_unc"]) == pytest.approx(
            np.nanstd(locs["sx"]), rel=0.6
        )
        np.testing.assert_allclose(locs["sx_unc"], locs["sy_unc"])


class TestFitGaussSplitFov:
    """The single-sensor split field of view: the channels are regions of one
    movie rather than separate movies."""

    H, W = 64, 128
    BOX = 9
    N_FRAMES = 40
    SIGMA = 1.25
    REGIONS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]
    FX, FY = 1.4, -0.85  # fine offset on top of the 64 px region step

    def _render(self, frame, x, y, amp):
        j, i = np.mgrid[0 : self.H, 0 : self.W]
        frame += amp * np.exp(
            -0.5 * ((i - x) ** 2 + (j - y) ** 2) / self.SIGMA**2
        )

    def _dataset(self, amps=(900.0, 400.0), seed=9):
        rng = np.random.RandomState(seed)
        truth_x = rng.uniform(14, 50, self.N_FRAMES)
        truth_y = rng.uniform(14, 50, self.N_FRAMES)
        movie = np.zeros((self.N_FRAMES, self.H, self.W))
        for f in range(self.N_FRAMES):
            self._render(movie[f], truth_x[f], truth_y[f], amps[0])
            self._render(
                movie[f],
                truth_x[f] + 64 + self.FX,
                truth_y[f] + self.FY,
                amps[1],
            )
        movie = (rng.poisson(movie + 10) + 100).astype(np.uint16)
        camera_info = {"Baseline": 100.0, "Sensitivity": 1.0, "Gain": 1}
        # a movie-wide identification table, as the GUI produces: both regions
        ids = pd.DataFrame(
            {
                "frame": np.concatenate([np.arange(self.N_FRAMES)] * 2),
                "x": np.concatenate(
                    [
                        np.rint(truth_x),
                        np.rint(truth_x + 64 + self.FX),
                    ]
                ).astype(np.int64),
                "y": np.concatenate(
                    [np.rint(truth_y), np.rint(truth_y + self.FY)]
                ).astype(np.int64),
                "net_gradient": np.full(2 * self.N_FRAMES, 1e4, np.float32),
            }
        )
        return movie, camera_info, ids, truth_x, truth_y

    def _registration(self):
        matrix = np.array(
            [
                [1.0, 0.0, 64 + self.FX],
                [0.0, 1.0, self.FY],
                [0.0, 0.0, 1.0],
            ]
        )
        transforms = [
            transforms_mod.identity(),
            transforms_mod.AffineTransform(matrix=matrix),
        ]
        rects = [localize._normalize_rect(r) for r in self.REGIONS]
        return {
            "model": "channel-registration",
            "n_channels": 2,
            "split_fov": True,
            "reference": 0,
            "regions": rects,
            "channel_transforms": [t.to_dict() for t in transforms],
            "channel_registration": [
                a.to_dict()
                for a in localize.decompose_region_transforms(
                    rects, transforms
                )
            ],
        }

    @pytest.mark.parametrize(
        "link_photons", [False, True], ids=["decoupled", "linked"]
    )
    def test_recovers_ground_truth(self, link_photons):
        movie, camera_info, ids, truth_x, truth_y = self._dataset()

        locs = localize.fit_gauss_split_fov(
            movie,
            camera_info,
            ids,
            self.BOX,
            self._registration(),
            regions=self.REGIONS,
            mle=True,
            link_photons=link_photons,
            use_gpu=False,
            multiprocess=False,
        )

        # confined to the reference region, so one localization per molecule
        assert len(locs) == self.N_FRAMES
        frame = locs["frame"].to_numpy()
        error_x = locs["x"].to_numpy() - truth_x[frame]
        error_y = locs["y"].to_numpy() - truth_y[frame]
        assert abs(np.nanmean(error_x)) < 0.06
        assert abs(np.nanmean(error_y)) < 0.06
        # and they are in the reference region's coordinates
        assert locs["x"].max() < 64

    def test_the_regions_can_be_redrawn(self):
        """The registration is ROI-agnostic, so a split that sits elsewhere is
        handled by moving the regions rather than re-calibrating."""
        movie, camera_info, ids, truth_x, _ = self._dataset()
        registration = self._registration()
        # pretend the calibration was taken with the split 6 px further right
        shifted = [[[0, 0], [64, 64]], [[0, 58], [64, 122]]]
        registration["regions"] = shifted

        locs = localize.fit_gauss_split_fov(
            movie,
            camera_info,
            ids,
            self.BOX,
            registration,
            regions=self.REGIONS,  # the ROIs actually in use
            mle=True,
            use_gpu=False,
            multiprocess=False,
        )

        frame = locs["frame"].to_numpy()
        assert abs(np.nanmean(locs["x"].to_numpy() - truth_x[frame])) < 0.06

    def test_confining_to_the_reference_can_be_turned_off(self):
        """The flag only says who does the confining, not whether it happens:
        pre-confined detections must give the same fit."""
        movie, camera_info, ids, _, _ = self._dataset()
        registration = self._registration()
        confined = localize.confine_to_region(ids, self.REGIONS[0])
        assert len(confined) == self.N_FRAMES

        internally = localize.fit_gauss_split_fov(
            movie,
            camera_info,
            ids,
            self.BOX,
            registration,
            regions=self.REGIONS,
            mle=True,
            use_gpu=False,
            multiprocess=False,
        )
        externally = localize.fit_gauss_split_fov(
            movie,
            camera_info,
            confined,
            self.BOX,
            registration,
            regions=self.REGIONS,
            confine_to_reference=False,
            mle=True,
            use_gpu=False,
            multiprocess=False,
        )

        np.testing.assert_allclose(
            externally["x"].to_numpy(), internally["x"].to_numpy()
        )
        np.testing.assert_allclose(
            externally["photons"].to_numpy(), internally["photons"].to_numpy()
        )

    def test_rejects_a_registration_that_is_not_split_fov(self):
        movie, camera_info, ids, _, _ = self._dataset()
        registration = self._registration()
        del registration["split_fov"]
        with pytest.raises(ValueError, match="split-FOV"):
            localize.fit_gauss_split_fov(
                movie,
                camera_info,
                ids,
                self.BOX,
                registration,
                use_gpu=False,
            )


@pytest.mark.gui
class TestSplitFovGaussianThroughTheWindow:
    """Split-FOV dispatch: one movie with ROIs, no separate channels loaded."""

    H, W = 64, 128
    BOX = 9
    N_FRAMES = 12
    REGIONS = [[[0, 0], [64, 64]], [[0, 64], [64, 128]]]
    FX, FY = 1.4, -0.85

    def _window(self, tmp_path):
        rng = np.random.RandomState(21)
        truth_x = rng.uniform(16, 46, self.N_FRAMES)
        truth_y = rng.uniform(16, 46, self.N_FRAMES)
        j, i = np.mgrid[0 : self.H, 0 : self.W]
        movie = np.zeros((self.N_FRAMES, self.H, self.W))
        for f in range(self.N_FRAMES):
            for x, y, amp in (
                (truth_x[f], truth_y[f], 900.0),
                (truth_x[f] + 64 + self.FX, truth_y[f] + self.FY, 450.0),
            ):
                movie[f] += amp * np.exp(
                    -0.5 * ((i - x) ** 2 + (j - y) ** 2) / 1.25**2
                )
        movie = (rng.poisson(movie + 10) + 100).astype(np.uint16)

        rects = [localize._normalize_rect(r) for r in self.REGIONS]
        matrix = np.array(
            [[1.0, 0.0, 64 + self.FX], [0.0, 1.0, self.FY], [0.0, 0.0, 1.0]]
        )
        tfs = [
            transforms_mod.identity(),
            transforms_mod.AffineTransform(matrix=matrix),
        ]
        path = str(tmp_path / "sfov_reg.yaml")
        io.save_any_calibration(
            path,
            {
                "model": "channel-registration",
                "n_channels": 2,
                "split_fov": True,
                "reference": 0,
                "regions": rects,
                "channel_transforms": [t.to_dict() for t in tfs],
                "channel_registration": [
                    a.to_dict()
                    for a in localize.decompose_region_transforms(rects, tfs)
                ],
            },
        )

        window = localize_gui.Window()
        dialog = window.parameters_dialog
        dialog.baseline.setValue(100)
        dialog.sensitivity.setValue(1.0)
        dialog.gain.setValue(1)
        window.parameters["Box Size"] = self.BOX
        # split-FOV is a single loaded movie whose regions are the channels
        window._set_channels(
            [movie],
            [[{"Frames": self.N_FRAMES, "Height": self.H, "Width": self.W}]],
            [str(tmp_path / "split.tif")],  # the fit saves next to the movie
            ["ch0"],
        )
        dialog.fit_model.setCurrentText("2D spherical Gaussian")
        dialog.fit_optimizer.setCurrentText("MLE")
        # loading the registration should drop its regions in and switch the
        # view into split-FOV mode by itself
        dialog.update_channel_registration(path)

        ids = pd.DataFrame(
            {
                "frame": np.concatenate([np.arange(self.N_FRAMES)] * 2),
                "x": np.concatenate(
                    [np.rint(truth_x), np.rint(truth_x + 64 + self.FX)]
                ).astype(np.int64),
                "y": np.concatenate(
                    [np.rint(truth_y), np.rint(truth_y + self.FY)]
                ).astype(np.int64),
                "net_gradient": np.full(2 * self.N_FRAMES, 1e4, np.float32),
            }
        )
        window.identifications = ids
        window.ready_for_fit = True
        return window, truth_x

    def test_loading_the_registration_enters_split_fov_mode(self, tmp_path):
        window, _ = self._window(tmp_path)
        try:
            assert window.view.split_fov_mode
            assert len(window.view.rois) == 2
        finally:
            window.close()

    def test_fit_dispatches_to_the_split_fov_path(self, tmp_path):
        window, truth_x = self._window(tmp_path)
        try:
            window.fit()
            worker = window.fit_worker
            assert isinstance(
                worker, localize_gui.MultichannelGaussianFitWorker
            )
            assert worker.split_fov is True
            worker.wait()
            QtWidgets.QApplication.processEvents()

            assert window.locs is not None
            # one localization per molecule, in reference-region coordinates
            assert len(window.locs) == self.N_FRAMES
            frame = window.locs["frame"].to_numpy()
            error = window.locs["x"].to_numpy() - truth_x[frame]
            assert abs(np.nanmean(error)) < 0.1
            assert window.locs["x"].max() < 64
        finally:
            window.close()


class _InterruptibleWorker(QtCore.QThread):
    """A worker that runs until it is asked to stop, like the real ones."""

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self.msleep(10)


@pytest.mark.gui
class TestClosingStopsTheWorkers:
    """Qt aborts the process when a running QThread is destroyed, so a window
    must not be closed out from under its identification or fit."""

    def test_a_running_worker_is_interrupted_and_waited_for(self):
        window = localize_gui.Window()
        worker = _InterruptibleWorker()
        worker.start()
        while not worker.isRunning():  # started asynchronously
            pass
        window._active_worker = worker
        try:
            window.close()
            assert not worker.isRunning()
            assert window._worker_still_running() is False
        finally:
            worker.requestInterruption()
            worker.wait()

    def test_the_fit_worker_is_stopped_too(self):
        """Not only the one the abort button points at: the window is
        destroyed with every thread it started."""
        window = localize_gui.Window()
        worker = _InterruptibleWorker()
        worker.start()
        while not worker.isRunning():
            pass
        window.fit_worker = worker
        try:
            window.close()
            assert not worker.isRunning()
        finally:
            worker.requestInterruption()
            worker.wait()


@pytest.mark.gui
class TestSlotsSurviveAStaleWindow:
    """A worker's signals arrive when the worker is done, not when the window
    still looks the way it did at the start - anything it touches may have
    been cleared meanwhile. An exception in a slot aborts the process, so
    these have to hold."""

    def test_the_markers_are_drawn_without_an_identification_info(self):
        """The box falls back to the current parameter."""
        movie = np.zeros((2, 32, 32), np.float32)
        window = localize_gui.Window()
        try:
            window._set_channels([movie], [_info()], ["a.tif"], ["Channel 0"])
            window.identifications = pd.DataFrame(
                {"frame": [0], "x": [10.0], "y": [10.0]}
            )
            window.ready_for_fit = True
            window.last_identification_info = None
            window.draw_frame()  # must not raise
        finally:
            window.close()

    def test_a_dropped_sum_still_reports_what_was_identified(self):
        """Anything that drops the sum (a loaded calibration, a moved region)
        can happen while the identification runs; the detections still came
        from the sum that was searched."""
        movie = np.zeros((2, 32, 32), np.float32)
        window = _sum_mode_window(movie, movie)
        try:
            window._sum_identify = {
                "fit_afterwards": False,
                "n_channels": 2,
                "source": "a test",
            }
            window.drop_channel_sum()
            assert window.sum_transforms is None
            ids = pd.DataFrame(
                {
                    "frame": [0],
                    "x": [10.0],
                    "y": [10.0],
                    "net_gradient": [1.0],
                }
            )
            window._on_sum_identify_finished(
                dict(window.parameters), [], 0.1, ids
            )
            message = window.status_bar.currentMessage()
            assert "sum of 2 channels" in message
            assert "registered from a test" in message
            assert (
                window.last_identification_info["Channel sum registration"]
                == "a test"
            )
        finally:
            window.close()

    def test_the_fit_is_saved_without_an_identification_info(self, tmp_path):
        """The metadata is written from what there is, rather than costing
        the user the fit that has just run."""
        movie = np.zeros((2, 32, 32), np.float32)
        window = localize_gui.Window()
        try:
            window._set_channels([movie], [_info()], ["a.tif"], ["Channel 0"])
            window.last_identification_info = None
            window.locs = pd.DataFrame(
                {"frame": [0], "x": [10.0], "y": [10.0]}
            )
            path = str(tmp_path / "locs.hdf5")
            window.save_locs(path)  # must not raise
            _, info = io.load_locs(path)
            assert info[-1]["Generated by"].startswith("Picasso")
        finally:
            window.close()


# ---------------------------------------------------------------------------
# Independent (per channel / per region) fitting
# ---------------------------------------------------------------------------


def _independent_movie(n_frames=6, height=32, width=64, spots=(), seed=3):
    """A movie with one Gaussian spot per (frame, position) entry.

    ``spots`` is a list of ``(x, y, amplitude)``, each repeated in every
    frame. Returns ``(movie, identifications)``."""
    rng = np.random.RandomState(seed)
    j, i = np.mgrid[0:height, 0:width]
    movie = np.zeros((n_frames, height, width))
    for x, y, amp in spots:
        movie += amp * np.exp(-0.5 * ((i - x) ** 2 + (j - y) ** 2) / 1.2**2)
    movie = rng.poisson(movie + 10).astype(np.uint16)
    ids = pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n_frames), len(spots)),
            "x": np.tile([int(round(s[0])) for s in spots], n_frames),
            "y": np.tile([int(round(s[1])) for s in spots], n_frames),
            "net_gradient": np.full(
                n_frames * len(spots), 1e4, dtype=np.float32
            ),
        }
    )
    return movie, ids


class TestPerChannelArgument:
    """Scalars are shared by every channel, lists are per channel."""

    def test_scalar_is_broadcast(self):
        assert localize._per_channel_arg("gausslq", 3, "m") == ["gausslq"] * 3

    def test_a_dict_is_one_shared_calibration(self):
        # a calibration is a mapping, not a per-channel sequence
        calib = {"model": "spline-2d"}
        assert localize._per_channel_arg(calib, 2, "c") == [calib, calib]

    def test_list_must_match_the_channel_count(self):
        with pytest.raises(ValueError, match="2 entries but there are 3"):
            localize._per_channel_arg(["a", "b"], 3, "fitting_method")


class TestRegionCoordinates:
    """Moving localizations into a region's own frame, and back out."""

    REGION = [[10, 20], [40, 60]]

    def _locs(self):
        return pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "x": np.float32([25.5, 55.0, 5.0]),
                "y": np.float32([12.5, 30.0, 30.0]),
            }
        )

    def test_locs_are_shifted_by_the_region_origin(self):
        shifted = localize.locs_to_region_coordinates(
            self._locs(), self.REGION
        )
        np.testing.assert_allclose(shifted["x"], [5.5, 35.0, -15.0])
        np.testing.assert_allclose(shifted["y"], [2.5, 20.0, 20.0])

    def test_corner_order_does_not_matter(self):
        flipped = localize.locs_to_region_coordinates(
            self._locs(), [[40, 60], [10, 20]]
        )
        expected = localize.locs_to_region_coordinates(
            self._locs(), self.REGION
        )
        np.testing.assert_allclose(flipped["x"], expected["x"])

    def test_region_info_carries_the_region_size(self):
        info = [{"Frames": 5, "Height": 64, "Width": 128}]
        region_info = localize.region_movie_info(info, self.REGION)
        assert region_info[-1]["Width"] == 40
        assert region_info[-1]["Height"] == 30
        assert region_info[-1]["Region"] == [[10, 20], [40, 60]]
        # the original is untouched
        assert info[-1]["Width"] == 128

    def test_split_by_region_matches_confine_to_region(self):
        locs = self._locs()
        regions = [self.REGION, [[10, 60], [40, 100]]]
        per_region = localize.split_locs_by_region(
            locs, regions, to_local=False
        )
        for region, part in zip(regions, per_region):
            expected = localize.confine_to_region(locs, region)
            np.testing.assert_allclose(part["x"], expected["x"])
        # the localization at x=5 is in neither region and is dropped
        assert sum(len(_) for _ in per_region) == 2

    def test_split_by_region_is_local_by_default(self):
        locs = self._locs()
        first = localize.split_locs_by_region(locs, [self.REGION])[0]
        np.testing.assert_allclose(first["x"], [5.5, 35.0])
        np.testing.assert_allclose(first["y"], [2.5, 20.0])


class TestFitIndependent:
    """Several movies, each fitted on its own."""

    def _channels(self, factory):
        movie_a, ids_a = _independent_movie(spots=[(12.4, 15.6, 900.0)])
        movie_b, ids_b = _independent_movie(
            spots=[(40.2, 20.3, 700.0)], seed=7
        )
        info = [{"Frames": len(movie_a)}]
        return (
            [factory(movie_a, info), factory(movie_b, info)],
            [ids_a, ids_b],
        )

    def test_matches_fitting_each_channel_separately(
        self, picasso_movie_factory
    ):
        movies, ids = self._channels(picasso_movie_factory)
        results = localize.fit_independent(
            movies,
            CAMERA_INFO_WITH_PIXELSIZE,
            ids,
            BOX,
            "gausslq",
            multiprocess=False,
        )
        assert len(results) == 2
        for c, (locs, info) in enumerate(results):
            expected, _ = localize.fit(
                movies[c],
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                identifications=ids[c],
                box=BOX,
                fitting_method="gausslq",
                multiprocess=False,
            )
            np.testing.assert_allclose(locs["x"], expected["x"], rtol=1e-6)
            assert info["Channel"] == c
            assert info["Channels"] == 2
            assert info["Fit mode"] == localize.FIT_MODE_INDEPENDENT

    def test_each_channel_may_use_its_own_method(self, picasso_movie_factory):
        movies, ids = self._channels(picasso_movie_factory)
        results = localize.fit_independent(
            movies,
            CAMERA_INFO_WITH_PIXELSIZE,
            ids,
            BOX,
            ["gausslq", "avg"],
            multiprocess=False,
        )
        assert results[0][1]["Fit method"] == "gausslq"
        assert results[1][1]["Fit method"] == "avg"

    def test_channel_positions_are_recovered(self, picasso_movie_factory):
        movies, ids = self._channels(picasso_movie_factory)
        results = localize.fit_independent(
            movies,
            CAMERA_INFO_WITH_PIXELSIZE,
            ids,
            BOX,
            "gausslq",
            multiprocess=False,
        )
        assert abs(results[0][0]["x"].mean() - 12.4) < 0.3
        assert abs(results[1][0]["x"].mean() - 40.2) < 0.3

    def test_one_identification_table_per_movie_is_required(
        self, picasso_movie_factory
    ):
        movies, ids = self._channels(picasso_movie_factory)
        with pytest.raises(ValueError, match="one per channel"):
            localize.fit_independent(
                movies, CAMERA_INFO_WITH_PIXELSIZE, ids[:1], BOX
            )

    def test_a_joint_calibration_is_rejected(self, picasso_movie_factory):
        movies, ids = self._channels(picasso_movie_factory)
        with pytest.raises(ValueError, match="spline-3d-multichannel"):
            localize.fit_independent(
                movies,
                CAMERA_INFO_WITH_PIXELSIZE,
                ids,
                BOX,
                "spline",
                spline_calibration={"model": "spline-3d-multichannel"},
            )

    def test_an_abort_gives_up_the_whole_batch(
        self, picasso_movie_factory, monkeypatch
    ):
        """A channel that aborts mid-fit returns None from ``fit``; the
        batch must not come back half-finished."""
        movies, ids = self._channels(picasso_movie_factory)
        monkeypatch.setattr(localize, "fit", lambda *a, **k: (None, {}))
        assert (
            localize.fit_independent(
                movies,
                CAMERA_INFO_WITH_PIXELSIZE,
                ids,
                BOX,
                "gausslq",
                multiprocess=False,
            )
            is None
        )


class TestFitSplitFovIndependent:
    """Regions of one movie, each fitted on its own."""

    REGIONS = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]

    def _movie(self, factory):
        movie, ids = _independent_movie(
            spots=[(12.4, 15.6, 900.0), (44.7, 18.2, 700.0)]
        )
        return factory(movie, [{"Frames": len(movie)}]), ids

    def _fit(self, factory, **kwargs):
        movie, ids = self._movie(factory)
        results = localize.fit_split_fov_independent(
            movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            ids,
            BOX,
            self.REGIONS,
            "gausslq",
            multiprocess=False,
            **kwargs,
        )
        return movie, ids, results

    def test_each_region_is_fitted_on_its_own(self, picasso_movie_factory):
        movie, ids, results = self._fit(picasso_movie_factory)
        assert len(results) == 2
        for region, (locs, _) in zip(self.REGIONS, results):
            expected, _ = localize.fit(
                movie,
                camera_info=CAMERA_INFO_WITH_PIXELSIZE,
                identifications=localize.confine_to_region(ids, region),
                box=BOX,
                fitting_method="gausslq",
                multiprocess=False,
            )
            expected = localize.locs_to_region_coordinates(expected, region)
            np.testing.assert_allclose(locs["x"], expected["x"], rtol=1e-6)

    def test_localizations_come_out_in_region_coordinates(
        self, picasso_movie_factory
    ):
        _, _, results = self._fit(picasso_movie_factory)
        # both regions are 32 px wide, so both spots land in the same range
        assert abs(results[0][0]["x"].mean() - 12.4) < 0.3
        assert abs(results[1][0]["x"].mean() - 12.7) < 0.3
        assert results[1][0]["x"].max() < 32

    def test_sensor_coordinates_on_request(self, picasso_movie_factory):
        _, _, results = self._fit(picasso_movie_factory, to_local=False)
        assert abs(results[1][0]["x"].mean() - 44.7) < 0.3

    def test_region_metadata_is_returned_with_the_movie_info(
        self, picasso_movie_factory
    ):
        info = [{"Frames": 6, "Height": 32, "Width": 64}]
        _, _, results = self._fit(picasso_movie_factory, movie_info=info)
        first, second = (r[1] for r in results)
        # the movie entry describes the region, the fit entry the fit
        assert first[0]["Width"] == 32 and first[0]["Height"] == 32
        assert second[0]["Region"] == [[0, 32], [32, 64]]
        assert second[-1]["Channel"] == 1
        assert second[-1]["Fit method"] == "gausslq"
        assert second[-1]["Fit mode"] == localize.FIT_MODE_INDEPENDENT
        assert second[-1]["Region coordinates"] == "region"

    def test_the_fit_metadata_comes_back_without_a_movie_info(
        self, picasso_movie_factory
    ):
        _, _, results = self._fit(picasso_movie_factory)
        for c, (_, info) in enumerate(results):
            assert len(info) == 1
            assert info[-1]["Channel"] == c
            assert info[-1]["Fit mode"] == localize.FIT_MODE_INDEPENDENT

    def test_each_region_may_use_its_own_method(self, picasso_movie_factory):
        movie, ids = self._movie(picasso_movie_factory)
        results = localize.fit_split_fov_independent(
            movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            ids,
            BOX,
            self.REGIONS,
            ["gausslq", "avg"],
            multiprocess=False,
        )
        # "avg" has no fitted width, so its localizations differ from a
        # Gaussian fit's - the point is only that both ran
        assert len(results) == 2
        assert all(len(locs) for locs, _ in results)

    def test_at_least_one_region_is_needed(self, picasso_movie_factory):
        movie, ids = self._movie(picasso_movie_factory)
        with pytest.raises(ValueError, match="at least one region"):
            localize.fit_split_fov_independent(
                movie, CAMERA_INFO_WITH_PIXELSIZE, ids, BOX, [], "gausslq"
            )


class TestLocalizeCliSeparateLateralCorrection:
    """``picasso localize -ac`` alongside ``-zc``: a lateral correction kept
    in its own file must land exactly where an appended one would, compose
    with the one the 3D calibration carries, and never be applied twice.
    """

    CHROMATIC = [[1.0, 0.0, 5.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
    ASTIG = [[1.0, 0.0, 3.0], [0.0, 1.0, -1.5], [0.0, 0.0, 1.0]]

    @staticmethod
    def _entry(kind, matrix):
        return {"Type": kind, "Transform": affine(matrix).to_dict()}

    def _movie_path(self, tmp_path):
        movie, _ = _independent_movie(spots=[(12.4, 15.6, 900.0)])
        path = str(tmp_path / "movie.raw")
        io.save_raw(
            path,
            movie.astype(np.uint16),
            [
                {
                    "Byte Order": "<",
                    "Data Type": "uint16",
                    "Frames": int(movie.shape[0]),
                    "Height": int(movie.shape[1]),
                    "Width": int(movie.shape[2]),
                }
            ],
        )
        return path

    def _calibration(self, tmp_path, name, *entries):
        calibration = dict(CALIB_3D)
        for entry in entries:
            lib.append_lateral_transform(calibration, entry)
        path = str(tmp_path / name)
        io.save_any_calibration(path, calibration)
        return path

    def _run(self, tmp_path, monkeypatch, movie, suffix, method, *extra):
        from picasso import __main__ as cli

        # "lq-3d" is what turns --zc into an astigmatic z fit; "lq" is the
        # plain 2D fit.
        argv = [
            "picasso",
            "localize",
            movie,
            "-b",
            "7",
            "-g",
            "100",
            "-a",
            method,
            "-d",
            "0",
            "-mf",
            "0.79",
            "-sf",
            suffix,
            *extra,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        return io.load_locs(str(tmp_path / f"movie{suffix}_locs.hdf5"))

    def test_separate_matches_appended(self, tmp_path, monkeypatch):
        movie = self._movie_path(tmp_path)
        appended = self._calibration(
            tmp_path,
            "both_3d_calib.yaml",
            self._entry("astigmatism", self.ASTIG),
            self._entry("chromatic", self.CHROMATIC),
        )
        astig_only = self._calibration(
            tmp_path,
            "astig_3d_calib.yaml",
            self._entry("astigmatism", self.ASTIG),
        )
        chromatic = str(tmp_path / "chromatic.yaml")
        io.save_any_calibration(
            chromatic,
            lib.append_lateral_transform(
                {}, self._entry("chromatic", self.CHROMATIC)
            ),
        )

        ref, ref_info = self._run(
            tmp_path,
            monkeypatch,
            movie,
            "_appended",
            "lq-3d",
            "-zc",
            appended,
        )
        out, info = self._run(
            tmp_path,
            monkeypatch,
            movie,
            "_separate",
            "lq-3d",
            "-zc",
            astig_only,
            "-ac",
            chromatic,
        )
        assert len(out) == len(ref) and len(out)
        np.testing.assert_allclose(
            out["x"].to_numpy(), ref["x"].to_numpy(), atol=1e-5
        )
        np.testing.assert_allclose(
            out["y"].to_numpy(), ref["y"].to_numpy(), atol=1e-5
        )
        # both routes name both corrections, in application order
        for each in (ref_info, info):
            assert lib.get_from_metadata(
                each, "Lateral corrections applied"
            ) == ["astigmatism, affine", "chromatic, affine"]

    def test_a_duplicate_of_the_z_calibration_is_skipped(
        self, tmp_path, monkeypatch
    ):
        movie = self._movie_path(tmp_path)
        appended = self._calibration(
            tmp_path,
            "astig_3d_calib.yaml",
            self._entry("astigmatism", self.ASTIG),
        )
        # the same transform, saved on its own under another name
        copy = str(tmp_path / "astig_copy.yaml")
        io.save_any_calibration(
            copy,
            lib.append_lateral_transform(
                {}, self._entry("astigmatism", self.ASTIG)
            ),
        )
        ref, _ = self._run(
            tmp_path, monkeypatch, movie, "_once", "lq-3d", "-zc", appended
        )
        out, info = self._run(
            tmp_path,
            monkeypatch,
            movie,
            "_twice",
            "lq-3d",
            "-zc",
            appended,
            "-ac",
            copy,
        )
        np.testing.assert_allclose(
            out["x"].to_numpy(), ref["x"].to_numpy(), atol=1e-5
        )
        assert lib.get_from_metadata(info, "Lateral corrections applied") == [
            "astigmatism, affine"
        ]
        assert lib.get_from_metadata(info, "Lateral corrections skipped") == [
            "astigmatism, affine"
        ]

    def test_applied_to_a_2d_fit_as_well(self, tmp_path, monkeypatch):
        """No -zc at all: the same flag corrects a plain 2D fit."""
        movie = self._movie_path(tmp_path)
        chromatic = str(tmp_path / "chromatic.yaml")
        io.save_any_calibration(
            chromatic,
            lib.append_lateral_transform(
                {}, self._entry("chromatic", self.CHROMATIC)
            ),
        )
        plain, _ = self._run(tmp_path, monkeypatch, movie, "_plain", "lq")
        out, info = self._run(
            tmp_path, monkeypatch, movie, "_corrected", "lq", "-ac", chromatic
        )
        np.testing.assert_allclose(
            out["x"].to_numpy(), plain["x"].to_numpy() + 5.0, atol=1e-5
        )
        np.testing.assert_allclose(
            out["y"].to_numpy(), plain["y"].to_numpy() + 2.0, atol=1e-5
        )
        assert lib.get_from_metadata(info, "Lateral corrections applied") == [
            "chromatic, affine"
        ]


class TestLocalizeCliRegionsSeparately:
    """``picasso localize --regions-separately``: one file per region."""

    REGIONS = ["0", "0", "32", "32"], ["0", "32", "32", "64"]

    def _movie_path(self, tmp_path):
        movie, _ = _independent_movie(
            spots=[(12.4, 15.6, 900.0), (44.7, 18.2, 700.0)]
        )
        path = str(tmp_path / "split.raw")
        io.save_raw(
            path,
            movie.astype(np.uint16),
            [
                {
                    "Byte Order": "<",
                    "Data Type": "uint16",
                    "Frames": int(movie.shape[0]),
                    "Height": int(movie.shape[1]),
                    "Width": int(movie.shape[2]),
                }
            ],
        )
        return path

    def _run(self, tmp_path, monkeypatch, *extra):
        from picasso import __main__ as cli

        path = self._movie_path(tmp_path)
        argv = ["picasso", "localize", path, "-b", "7", "-d", "0"]
        for region in self.REGIONS:
            argv += ["-r", *region]
        argv += ["--regions-separately", *extra]
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
        return path

    def test_each_region_is_saved_on_its_own(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, "-g", "100", "-a", "lq")
        ref = tmp_path / "split_ref_locs.hdf5"
        ch1 = tmp_path / "split_ch1_locs.hdf5"
        assert ref.exists() and ch1.exists()
        # ... and not the single combined file of an ordinary run
        assert not (tmp_path / "split_locs.hdf5").exists()
        for expected_x, path in ((12.4, ref), (12.7, ch1)):
            locs, info = io.load_locs(str(path))
            assert len(locs)
            # each region's localizations are in its own coordinates
            assert abs(locs["x"].mean() - expected_x) < 0.5
            assert lib.get_from_metadata(info, "Width") == 32
            assert (
                lib.get_from_metadata(info, "Fit mode")
                == localize.FIT_MODE_INDEPENDENT
            )

    def test_a_fitting_method_per_region(self, tmp_path, monkeypatch):
        self._run(
            tmp_path,
            monkeypatch,
            "-g",
            "100",
            "-a",
            "lq",
            "-a",
            "avg",
        )
        _, ref_info = io.load_locs(str(tmp_path / "split_ref_locs.hdf5"))
        _, ch1_info = io.load_locs(str(tmp_path / "split_ch1_locs.hdf5"))
        assert lib.get_from_metadata(ref_info, "Fit method") == "gausslq"
        assert lib.get_from_metadata(ch1_info, "Fit method") == "avg"

    def test_a_threshold_per_region(self, tmp_path, monkeypatch):
        """One --gradient per --roi, as the regions need not share a
        brightness scale."""
        self._run(
            tmp_path, monkeypatch, "-g", "100", "-g", "1000000", "-a", "lq"
        )
        ref, _ = io.load_locs(str(tmp_path / "split_ref_locs.hdf5"))
        assert len(ref)
        # nothing in the second region passes a threshold that high, so it
        # is not written at all
        assert not (tmp_path / "split_ch1_locs.hdf5").exists()

    def test_the_counts_must_match_the_regions(self, tmp_path, monkeypatch):
        with pytest.raises(Exception, match="one --fit-method per --roi"):
            self._run(
                tmp_path,
                monkeypatch,
                "-a",
                "lq",
                "-a",
                "avg",
                "-a",
                "mle",
            )

    def test_regions_are_required(self, tmp_path, monkeypatch):
        from picasso import __main__ as cli

        path = self._movie_path(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["picasso", "localize", path, "--regions-separately", "-d", "0"],
        )
        with pytest.raises(Exception, match="at least one --roi is needed"):
            cli.main()

    def test_an_ordinary_run_still_writes_one_file(
        self, tmp_path, monkeypatch
    ):
        from picasso import __main__ as cli

        path = self._movie_path(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["picasso", "localize", path, "-b", "7", "-d", "0", "-a", "lq"],
        )
        cli.main()
        assert (tmp_path / "split_locs.hdf5").exists()
        assert not (tmp_path / "split_ref_locs.hdf5").exists()


@pytest.mark.gui
class TestFitModeParameter:
    """The 'Fit' setting: joint by default, shown for multichannel data."""

    def test_single_channel_data_hides_it_and_fits_jointly(self):
        window = localize_gui.Window()
        try:
            window._set_channels(
                [np.zeros((2, 8, 8), np.float32)], [_info()], ["a.tif"], ["c0"]
            )
            assert not window.parameters_dialog.fit_mode_combo.isVisible()
            assert window.fit_mode() == localize_gui.FIT_MODE_JOINT
        finally:
            window.close()

    def test_several_channels_show_it(self):
        window = localize_gui.Window()
        try:
            movie = np.zeros((2, 8, 8), np.float32)
            window._set_channels(
                [movie, movie],
                [_info("c0"), _info("c1")],
                ["a.tif", "b.tif"],
                ["c0", "c1"],
            )
            dialog = window.parameters_dialog
            assert dialog.fit_mode_combo.isVisibleTo(dialog)
            assert dialog.fit_mode_label.isVisibleTo(dialog)
            # the default keeps every existing analysis as it was
            assert window.fit_mode() == localize_gui.FIT_MODE_JOINT
        finally:
            window.close()

    def test_it_keeps_its_space_while_hidden(self):
        dialog = localize_gui.Window().parameters_dialog
        try:
            for widget in (dialog.fit_mode_label, dialog.fit_mode_combo):
                assert widget.sizePolicy().retainSizeWhenHidden()
        finally:
            dialog.window.close()

    def test_the_mode_is_reset_when_the_channels_go(self):
        """Left on 'separately' with one channel it could not be turned off
        again - the combo is hidden then."""
        window = localize_gui.Window()
        try:
            movie = np.zeros((2, 8, 8), np.float32)
            window._set_channels(
                [movie, movie],
                [_info("c0"), _info("c1")],
                ["a.tif", "b.tif"],
                ["c0", "c1"],
            )
            window.parameters_dialog.fit_mode_combo.setCurrentText(
                localize_gui.FIT_MODE_SEPARATE
            )
            assert window.fit_mode() == localize_gui.FIT_MODE_SEPARATE
            window._set_channels([movie], [_info()], ["a.tif"], ["c0"])
            assert window.fit_mode() == localize_gui.FIT_MODE_JOINT
        finally:
            window.close()

    def test_the_joint_only_options_go_with_it(self):
        """Nothing is shared between channels fitted on their own, so the
        photon-linking checkboxes have nothing to say."""
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        try:
            movie = np.zeros((2, 8, 8), np.float32)
            window._set_channels(
                [movie, movie],
                [_info("c0"), _info("c1")],
                ["a.tif", "b.tif"],
                ["c0", "c1"],
            )
            dialog.channel_registration_calibration = {"n_channels": 2}
            dialog._update_gauss_link_photons_visibility()
            # isHidden, not isVisibleTo: the box it sits in is shown only
            # for the model that has a joint Gaussian fit
            assert not dialog.gauss_link_photons_checkbox.isHidden()
            dialog.fit_mode_combo.setCurrentText(
                localize_gui.FIT_MODE_SEPARATE
            )
            assert dialog.gauss_link_photons_checkbox.isHidden()
        finally:
            window.close()


def _pump(window, timeout=60.0):
    """Run the Qt event loop until an independent fit batch has finished."""
    deadline = time.time() + timeout
    while window._multi_fit is not None and time.time() < deadline:
        QtWidgets.QApplication.processEvents()
        worker = window._active_worker
        if worker is not None:
            worker.wait(50)
    QtWidgets.QApplication.processEvents()


def _spot_movie(n_frames=4, height=32, width=64, spots=(), seed=5):
    """A movie with one Gaussian spot per ``(x, y, amplitude)``, and the
    integer detections that identification would produce for it."""
    return _independent_movie(n_frames, height, width, spots, seed)


@pytest.mark.gui
class TestFitEachChannelSeparately:
    """Several loaded movies, each fitted with its own settings."""

    def _window(self, tmp_path, factory):
        movie_a, ids_a = _spot_movie(spots=[(12.4, 15.6, 900.0)])
        movie_b, ids_b = _spot_movie(spots=[(40.2, 20.3, 700.0)], seed=7)
        info = [{"Frames": 4, "Height": 32, "Width": 64}]
        movie_a = factory(movie_a, info)
        movie_b = factory(movie_b, info)
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        dialog.baseline.setValue(0)
        dialog.sensitivity.setValue(1.0)
        dialog.gain.setValue(1)
        window._set_channels(
            [movie_a, movie_b],
            [list(info), list(info)],
            [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")],
            ["c0", "c1"],
        )
        dialog.box_spinbox.setValue(7)
        dialog.fit_mode_combo.setCurrentText(localize_gui.FIT_MODE_SEPARATE)
        # every channel identified, as Identify (Ctrl+I) leaves them
        for index, ids in ((0, ids_a), (1, ids_b)):
            window.set_current_channel(index)
            window.identifications = ids
            window.ready_for_fit = True
            window.last_identification_info = dict(window.parameters)
        window.set_current_channel(0)
        return window

    def test_every_channel_is_fitted_and_saved_on_its_own(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            window.fit()
            _pump(window)
            assert (tmp_path / "a_locs.hdf5").exists()
            assert (tmp_path / "b_locs.hdf5").exists()
            first, _ = io.load_locs(str(tmp_path / "a_locs.hdf5"))
            second, _ = io.load_locs(str(tmp_path / "b_locs.hdf5"))
            assert abs(first["x"].mean() - 12.4) < 0.5
            assert abs(second["x"].mean() - 40.2) < 0.5
            # each channel keeps its own localizations - the joint fit's
            # reference-channel distribution must not have run
            assert window.channels[0].locs is not None
            assert window.channels[1].locs is not None
            assert len(window.channels[0].locs) == len(first)
        finally:
            window.close()

    def test_each_channel_uses_its_own_model(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            window.set_current_channel(1)
            window.parameters_dialog.fit_model.setCurrentText("Average of ROI")
            window.set_current_channel(0)
            window.fit()
            _pump(window)
            _, first_info = io.load_locs(str(tmp_path / "a_locs.hdf5"))
            _, second_info = io.load_locs(str(tmp_path / "b_locs.hdf5"))
            assert "Gaussian" in lib.get_from_metadata(
                first_info, "Fit method"
            )
            assert (
                lib.get_from_metadata(second_info, "Fit method")
                == "Average of ROI"
            )
            assert (
                lib.get_from_metadata(second_info, "Fit mode")
                == localize_gui.FIT_MODE_SEPARATE
            )
        finally:
            window.close()

    def test_a_channel_without_identifications_is_skipped(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            window.set_current_channel(1)
            window.identifications = None
            window.ready_for_fit = False
            window.set_current_channel(0)
            window.fit()
            _pump(window)
            assert (tmp_path / "a_locs.hdf5").exists()
            assert not (tmp_path / "b_locs.hdf5").exists()
            assert "Skipped c1" in window.status_bar.currentMessage()
        finally:
            window.close()

    def test_the_active_channel_comes_back(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            window.set_current_channel(1)
            window.fit()
            _pump(window)
            assert window.current_channel == 1
        finally:
            window.close()


@pytest.mark.gui
class TestFitEachRegionSeparately:
    """One movie whose regions are the channels, each fitted on its own."""

    REGIONS = [[[0, 0], [32, 32]], [[0, 32], [32, 64]]]

    def _window(self, tmp_path, factory):
        movie, ids = _spot_movie(
            spots=[(12.4, 15.6, 900.0), (44.7, 18.2, 700.0)]
        )
        movie = factory(movie, [{"Frames": 4, "Height": 32, "Width": 64}])
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        dialog.baseline.setValue(0)
        dialog.sensitivity.setValue(1.0)
        dialog.gain.setValue(1)
        window._set_channels(
            [movie],
            [[{"Frames": 4, "Height": 32, "Width": 64}]],
            [str(tmp_path / "split.tif")],
            ["c0"],
        )
        dialog.box_spinbox.setValue(7)
        window.view.rois = [list(r) for r in self.REGIONS]
        window.set_split_fov_mode(True)
        dialog.fit_mode_combo.setCurrentText(localize_gui.FIT_MODE_SEPARATE)
        window.identifications = ids
        window.ready_for_fit = True
        window.last_identification_info = dict(window.parameters)
        return window

    def test_one_file_per_region_in_its_own_coordinates(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            window.fit()
            _pump(window)
            ref = tmp_path / "split_ref_locs.hdf5"
            ch1 = tmp_path / "split_ch1_locs.hdf5"
            assert ref.exists() and ch1.exists()
            first, first_info = io.load_locs(str(ref))
            second, second_info = io.load_locs(str(ch1))
            assert abs(first["x"].mean() - 12.4) < 0.5
            # the second region's spot sits at x = 44.7 on the sensor and at
            # 12.7 in its own frame
            assert abs(second["x"].mean() - 12.7) < 0.5
            assert lib.get_from_metadata(second_info, "Width") == 32
            assert lib.get_from_metadata(second_info, "Region") == [
                [0, 32],
                [32, 64],
            ]
            assert (
                lib.get_from_metadata(first_info, "Fit mode")
                == localize_gui.FIT_MODE_SEPARATE
            )
        finally:
            window.close()

    def test_each_region_keeps_its_own_fit_settings(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        dialog = window.parameters_dialog
        try:
            # select the second region and give it another model; selecting
            # a region swaps the fit settings over, as it does the threshold
            window.view.selected_roi = 1
            window.sync_region_fit_params()
            dialog.fit_model.setCurrentText("Average of ROI")
            window.view.selected_roi = 0
            window.sync_region_fit_params()
            assert dialog.fit_model.currentText() != "Average of ROI"
            window.fit()
            _pump(window)
            _, first_info = io.load_locs(str(tmp_path / "split_ref_locs.hdf5"))
            _, second_info = io.load_locs(
                str(tmp_path / "split_ch1_locs.hdf5")
            )
            assert "Gaussian" in lib.get_from_metadata(
                first_info, "Fit method"
            )
            assert (
                lib.get_from_metadata(second_info, "Fit method")
                == "Average of ROI"
            )
        finally:
            window.close()

    def test_the_movie_metadata_is_restored_afterwards(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            window.fit()
            _pump(window)
            assert lib.get_from_metadata(window.info, "Width") == 64
            assert window._region_suffix == ""
        finally:
            window.close()

    def test_a_region_without_identifications_is_skipped(
        self, tmp_path, picasso_movie_factory
    ):
        window = self._window(tmp_path, picasso_movie_factory)
        try:
            # only the reference region's detections
            window.identifications = localize.confine_to_region(
                window.identifications, self.REGIONS[0]
            )
            window.fit()
            _pump(window)
            assert (tmp_path / "split_ref_locs.hdf5").exists()
            assert not (tmp_path / "split_ch1_locs.hdf5").exists()
            assert "Skipped ch1" in window.status_bar.currentMessage()
        finally:
            window.close()

    def test_a_joint_calibration_is_refused(
        self, tmp_path, monkeypatch, picasso_movie_factory
    ):
        """A multichannel spline calibration describes every region at once;
        fitting one region against it would use the wrong PSF."""
        window = self._window(tmp_path, picasso_movie_factory)
        dialog = window.parameters_dialog
        warned = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **kwargs: warned.append(args[2]),
        )
        try:
            dialog.fit_model.setCurrentText("Experimental PSF (cubic spline)")
            dialog.spline_calibration = {
                "model": "spline-3d-multichannel",
                "n_data": [7, 7, 21],
            }
            window.fit()
            _pump(window)
            assert warned and "multichannel spline calibration" in warned[0]
            assert not (tmp_path / "split_ref_locs.hdf5").exists()
            assert window._multi_fit is None
        finally:
            window.close()


@pytest.mark.gui
class TestIndependentFitGuards:
    """What the independent fit refuses, and why."""

    def test_sum_detections_are_refused(self, tmp_path, monkeypatch):
        """The channel sum's detections are one consensus set in reference
        coordinates - fitting them per channel would fit the reference
        channel's positions everywhere."""
        movie = np.zeros((2, 16, 16), np.float32)
        window = localize_gui.Window()
        told = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            lambda *args, **kwargs: told.append(args[2]),
        )
        try:
            window._set_channels(
                [movie, movie],
                [_info("c0"), _info("c1")],
                [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")],
                ["c0", "c1"],
            )
            window.parameters_dialog.fit_mode_combo.setCurrentText(
                localize_gui.FIT_MODE_SEPARATE
            )
            ids = pd.DataFrame(
                {"frame": [0], "x": [8], "y": [8], "net_gradient": [1e4]}
            )
            window.identifications = ids
            window.sum_identifications = ids
            window.ready_for_fit = True
            window.fit()
            assert told and "sum of the channels" in told[0]
            assert window._multi_fit is None
        finally:
            window.close()

    def test_a_channel_keeps_its_own_psf_calibration(self, tmp_path):
        """With the PSF link off, each channel's calibration follows it."""
        movie = np.zeros((2, 16, 16), np.float32)
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        try:
            window._set_channels(
                [movie, movie],
                [_info("c0"), _info("c1")],
                [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")],
                ["c0", "c1"],
            )
            dialog.link_calib_checkbox.setChecked(False)
            dialog._set_spline_state({"model": "spline-2d"}, "ref_psf.hdf5")
            window.set_current_channel(1)
            assert dialog.spline_calibration_path != "ref_psf.hdf5"
            dialog._set_spline_state({"model": "spline-2d"}, "ch1_psf.hdf5")
            window.set_current_channel(0)
            assert dialog.spline_calibration_path == "ref_psf.hdf5"
            window.set_current_channel(1)
            assert dialog.spline_calibration_path == "ch1_psf.hdf5"
        finally:
            window.close()

    def test_the_link_keeps_one_calibration_for_every_channel(self, tmp_path):
        """Ticked (the default), the calibration is shared - which is what
        the joint multichannel fit needs."""
        movie = np.zeros((2, 16, 16), np.float32)
        window = localize_gui.Window()
        dialog = window.parameters_dialog
        try:
            window._set_channels(
                [movie, movie],
                [_info("c0"), _info("c1")],
                [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")],
                ["c0", "c1"],
            )
            assert dialog.link_calib_checkbox.isChecked()
            dialog._set_spline_state({"model": "spline-2d"}, "shared.hdf5")
            window.set_current_channel(1)
            assert dialog.spline_calibration_path == "shared.hdf5"
        finally:
            window.close()
