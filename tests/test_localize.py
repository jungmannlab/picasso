"""Test ``picasso.localize`` — spot identification, extraction, and the
high-level ``fit``/``fit_async`` MLE wrapper, plus the diagnostic
helpers.

Tests for ``gausslq``, ``gaussmle`` and ``zfit`` live in their own files
(``test_gausslq.py``, ``test_gaussmle.py``, ``test_zfit.py``).

:author: Rafal Kowalewski, 2025-2026
:copyright: Copyright (c) 2025-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from picasso import localize

from tests.conftest import BOX, CALIB_3D, CAMERA_INFO, MIN_NG, PIXELSIZE


CAMERA_INFO_WITH_PIXELSIZE = {**CAMERA_INFO, "Pixelsize": PIXELSIZE}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient_meshgrid(box: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the normalised (uy, ux) direction vectors that
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
# local_maxima
# ---------------------------------------------------------------------------


class TestLocalMaxima:
    """Pure local-maxima search inside a sliding box."""

    def test_single_peak_detected(self):
        frame = np.zeros((20, 20), dtype=np.float32)
        frame[10, 12] = 100.0
        y, x = localize.local_maxima(frame, BOX)
        assert list(zip(y.tolist(), x.tolist())) == [(10, 12)]

    def test_multiple_peaks_far_apart_all_found(self):
        frame = np.zeros((30, 30), dtype=np.float32)
        peaks = [(8, 8), (8, 22), (22, 15)]
        for py, px in peaks:
            frame[py, px] = 50.0
        y, x = localize.local_maxima(frame, BOX)
        found = set(zip(y.tolist(), x.tolist()))
        assert found == set(peaks)

    def test_peaks_in_border_band_are_excluded(self):
        """``local_maxima`` only scans i in [box_half, Y - box_half - 1)
        — peaks placed inside the border band must not be returned."""
        Y = X = 20
        box_half = BOX // 2
        frame = np.zeros((Y, X), dtype=np.float32)
        frame[1, 1] = 100.0  # top-left border
        frame[Y - 2, X - 2] = 100.0  # bottom-right border
        y, x = localize.local_maxima(frame, BOX)
        # All returned coordinates lie strictly inside the scan band
        assert ((y >= box_half) & (y < Y - box_half - 1)).all()
        assert ((x >= box_half) & (x < X - box_half - 1)).all()

    def test_flat_frame_returns_no_maxima(self):
        """A constant frame has no unique local max — the implementation's
        ``argmax`` returns the top-left pixel (index 0), so no local
        window has its max at the center."""
        frame = np.full((20, 20), 42.0, dtype=np.float32)
        y, x = localize.local_maxima(frame, BOX)
        assert len(y) == 0 and len(x) == 0


# ---------------------------------------------------------------------------
# gradient_at
# ---------------------------------------------------------------------------


class TestGradientAt:
    """Two-point centered finite difference at (y, x)."""

    def test_horizontal_gradient(self):
        # frame[y, x+1] - frame[y, x-1]  along increasing x
        frame = np.tile(np.arange(10, dtype=np.float32), (10, 1))
        gy, gx = localize.gradient_at(frame, 5, 5, 0)
        assert gy == 0.0
        assert gx == 2.0  # 6 - 4

    def test_vertical_gradient(self):
        # frame[y+1, x] - frame[y-1, x]  along increasing y
        frame = np.tile(
            np.arange(10, dtype=np.float32).reshape(-1, 1), (1, 10)
        )
        gy, gx = localize.gradient_at(frame, 5, 5, 0)
        assert gy == 2.0  # 6 - 4
        assert gx == 0.0

    def test_zero_gradient_in_flat_region(self):
        frame = np.full((10, 10), 7.0, dtype=np.float32)
        gy, gx = localize.gradient_at(frame, 5, 5, 0)
        assert gy == 0.0 and gx == 0.0

    def test_i_argument_is_ignored(self):
        """``i`` is documented as unused — different values must not
        affect the returned gradient."""
        frame = _gaussian_frame((15, 15), (7, 7))
        a = localize.gradient_at(frame, 7, 8, 0)
        b = localize.gradient_at(frame, 7, 8, 999)
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
        ng = localize.net_gradient(frame, y, x, BOX, uy, ux)
        assert ng.shape == (1,)
        assert ng[0] > 0

    def test_flat_frame_yields_zero(self):
        frame = np.full((15, 15), 50.0, dtype=np.float32)
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize.net_gradient(frame, y, x, BOX, uy, ux)
        np.testing.assert_allclose(ng, [0.0], atol=1e-6)

    def test_inverted_peak_yields_negative(self):
        """A dip (gradients pointing inward) gives a negative net
        gradient — the sign is the discriminator between peaks and
        troughs."""
        frame = -_gaussian_frame((15, 15), (7, 7), background=0.0)
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([7], dtype=np.int64)
        x = np.array([7], dtype=np.int64)
        ng = localize.net_gradient(frame, y, x, BOX, uy, ux)
        assert ng[0] < 0

    def test_output_length_matches_input(self):
        frame = _gaussian_frame((30, 30), (10, 10))
        uy, ux = _gradient_meshgrid(BOX)
        y = np.array([10, 10, 10], dtype=np.int64)
        x = np.array([8, 10, 12], dtype=np.int64)
        ng = localize.net_gradient(frame, y, x, BOX, uy, ux)
        assert ng.shape == (3,)


# ---------------------------------------------------------------------------
# identify_in_image
# ---------------------------------------------------------------------------


class TestIdentifyInImage:
    """``local_maxima`` + net-gradient threshold, in one shot."""

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
# fit + fit_async (MLE wrapper)
# ---------------------------------------------------------------------------


class TestFit:
    """High-level MLE wrapper that combines ``get_spots`` + ``gaussmle``."""

    def test_returns_locs_with_required_columns(
        self, movie, real_identifications
    ):
        locs = localize.fit(
            movie,
            CAMERA_INFO,
            real_identifications,
            BOX,
            method="sigmaxy",
        )
        assert len(locs) == len(real_identifications)
        for col in [
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
        ]:
            assert col in locs.columns

    def test_method_sigma_returns_equal_sx_sy(
        self, movie, real_identifications
    ):
        locs = localize.fit(
            movie, CAMERA_INFO, real_identifications, BOX, method="sigma"
        )
        # In the localize.fit MLE path the method=sigma constrains sx==sy
        np.testing.assert_array_equal(
            locs["sx"].to_numpy(), locs["sy"].to_numpy()
        )

    @pytest.mark.slow
    def test_fit_async_matches_fit_after_completion(
        self, movie, real_identifications
    ):
        """``fit_async`` returns the in-progress state; once
        ``current[0]`` reaches the spot count, the per-spot fit results
        match the synchronous version (set equality up to numerical noise)."""
        locs_sync = localize.fit(
            movie,
            CAMERA_INFO,
            real_identifications,
            BOX,
            method="sigmaxy",
        )
        current, thetas, _, _, iterations = localize.fit_async(
            movie,
            CAMERA_INFO,
            real_identifications,
            BOX,
            method="sigmaxy",
        )
        # `current[0]` is incremented *before* the per-spot fit runs (see
        # ``gaussmle._worker``), so polling it can let the loop exit while
        # the last few worker writes are still in flight. ``iterations``
        # is zero-initialised and only written when a fit completes —
        # poll on that for a race-free completion signal.
        t0 = time.time()
        while (iterations == 0).any():
            assert time.time() - t0 < 30, "fit_async timed out"
            time.sleep(0.05)
        del current  # only kept to document the returned tuple shape
        # The per-spot photon counts should match (with possibly different
        # row ordering across worker scheduling). Compare sorted lists.
        np.testing.assert_allclose(
            np.sort(thetas[:, 2]),
            np.sort(locs_sync["photons"].to_numpy()),
            atol=1e-3,
        )


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

    def test_input_validation_rejects_plain_ndarray(
        self, movie, real_identifications, movie_info
    ):
        """``fit2D`` requires its movie argument to be an
        ``AbstractPicassoMovie`` — a plain ``np.memmap`` (what
        ``io.load_movie`` returns for ``.raw``) is rejected."""
        with pytest.raises(AssertionError):
            localize.fit2D(
                movie,
                movie_info,
                CAMERA_INFO_WITH_PIXELSIZE,
                real_identifications,
                BOX,
                fitting_method="gausslq",
                multiprocess=False,
            )

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
        assert new_info["Convergence criterion"] == 0.001
        assert new_info["Max iterations"] == 100

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

    def test_default_return_info_emits_deprecation(
        self, picasso_movie, movie_info
    ):
        """Passing ``return_info=None`` (the legacy default) triggers a
        deprecation warning. Will be removed in v0.12.0."""
        # ``lib.deprecation_warning`` prints to stderr; just verify the
        # call still returns the DataFrame without crashing.
        result = localize.localize(
            picasso_movie,
            CAMERA_INFO_WITH_PIXELSIZE,
            {"Min. Net Gradient": MIN_NG, "Box Size": BOX},
            movie_info=movie_info,
            fitting_method="gausslq",
            threaded=False,
            # return_info omitted on purpose
        )
        assert isinstance(result, pd.DataFrame)


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

    def test_public_localize_3d_rejects_ndarray_due_to_inner_assert(
        self, movie, movie_info
    ):
        """And feeding the bundled np.memmap (passes the outer check)
        fails the *inner* fit2D AbstractPicassoMovie assert. This
        documents the current incompatibility — both halves of the
        deprecated public API can't agree on a movie type."""
        with pytest.raises(
            AssertionError, match="movie must be a movie loaded by"
        ):
            localize.localize_3D(
                movie,
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
