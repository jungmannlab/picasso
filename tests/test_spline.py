"""Tests for picasso.spline (cubic-spline PSF calibration generation).

The whole calibration (frame binning, PSF-template building, registration,
normalization and the spline-coefficient step) runs on the CPU everywhere.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from picasso import localize, registration, spline, transforms, wavelet
from picasso.fitting import precision

from tests.conftest import (
    BOX,
    CAMERA_INFO,
    IDENTITY,
    affine,
    affine_matrix,
    apply_transform,
    linear_part,
)

# ---------------------------------------------------------------------------
# Synthetic bead z-stack
# ---------------------------------------------------------------------------


def _synthetic_bead_movie(n_frames=21, h=48, w=48, box=BOX, bead_xy=None):
    """A movie of a few static beads with a Gaussian PSF whose width is
    minimal at the central frame (focus) and grows away from it.

    ``bead_xy`` overrides the default three beads - the richer registration
    models need more correspondences than an affine does."""
    bead_xy = bead_xy or [(12, 14), (30, 28), (16, 33)]
    s0 = 1.1
    focus = n_frames // 2
    yy, xx = np.mgrid[0:h, 0:w]
    movie = np.zeros((n_frames, h, w), dtype=np.float32)
    for f in range(n_frames):
        sigma = s0 * (1.0 + 0.07 * abs(f - focus))
        img = np.full((h, w), 100.0, dtype=np.float32)
        for bx, by in bead_xy:
            img += 3000.0 * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )
        movie[f] = img
    return movie.astype(np.uint16), bead_xy, focus


def _synthetic_multifov_movie(
    n_fov=3, n_steps=11, h=48, w=48, order="z", box=BOX
):
    """A genuine multi-FOV bead z-stack: ``n_fov`` fields, each with beads at
    *different* positions, each scanned over ``n_steps`` z positions (focus at
    the center). Frames are laid out in ``order`` ("z": each FOV is a full z
    stack, then the next FOV; "fov": the FOVs are interleaved at each z).

    Returns ``(movie, fov_beads, focus)`` where ``fov_beads[k]`` are the bead
    centers of FOV ``k``. The total number of physical beads is
    ``sum(len(b) for b in fov_beads)`` - more than any single field holds.
    """
    fov_beads = [
        [(12, 14), (30, 28)],
        [(18, 33), (35, 12)],
        [(22, 20), (14, 38), (40, 30)],
        [(38, 16), (11, 25)],
    ][:n_fov]
    s0 = 1.1
    focus = n_steps // 2
    yy, xx = np.mgrid[0:h, 0:w]

    def frame_img(fov, k):
        sigma = s0 * (1.0 + 0.07 * abs(k - focus))
        img = np.full((h, w), 100.0, dtype=np.float32)
        for bx, by in fov_beads[fov]:
            img += 3000.0 * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )
        return img

    frames = []
    if order == "z":
        for fov in range(n_fov):
            for k in range(n_steps):
                frames.append(frame_img(fov, k))
    else:  # "fov": all FOVs at z0, then all FOVs at z1, ...
        for k in range(n_steps):
            for fov in range(n_fov):
                frames.append(frame_img(fov, k))
    movie = np.stack(frames).astype(np.uint16)
    return movie, fov_beads, focus


class TestFovOfFrame:
    def test_z_order(self):
        # 2 FOVs x 5 steps, z order: each FOV is a full z stack
        fov = spline._fov_of_frame(10, 2, "z")
        np.testing.assert_array_equal(fov, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_fov_order(self):
        # 2 FOVs interleaved at each z position
        fov = spline._fov_of_frame(10, 2, "fov")
        np.testing.assert_array_equal(fov, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def test_single_fov(self):
        np.testing.assert_array_equal(
            spline._fov_of_frame(5, 1, "fov"), [0, 0, 0, 0, 0]
        )

    def test_trailing_frames_marked_invalid(self):
        # 7 frames, 2 FOVs -> n_steps=3, frame 6 does not complete a step
        fov = spline._fov_of_frame(7, 2, "z")
        assert fov[-1] == -1

    def test_fov_and_step_pair_uniquely(self):
        # every (fov, step) pair maps to exactly one frame, in both orders
        n_frames, fps = 12, 3
        for order in ("fov", "z"):
            step, _, _ = spline._step_of_frame(
                n_frames, 10.0, fps, order, None
            )
            fov = spline._fov_of_frame(n_frames, fps, order)
            pairs = list(zip(fov.tolist(), step.tolist()))
            assert len(set(pairs)) == len(pairs) == n_frames


class TestMaskToSegments:
    def test_contiguous_run(self):
        mask = np.array([0, 1, 1, 1, 0, 0], dtype=bool)
        assert spline._mask_to_segments(mask) == [(1, 3)]

    def test_multiple_runs(self):
        mask = np.array([1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
        assert spline._mask_to_segments(mask) == [(0, 1), (3, 3), (5, 7)]

    def test_empty(self):
        assert spline._mask_to_segments(np.zeros(5, dtype=bool)) == []

    def test_reference_segments_split_per_fov_in_z_order(self):
        # 3 FOVs x 12 steps, z order: the in-focus middle third of each FOV is
        # a separate segment (not one giant min..max span)
        n_frames, fps = 36, 3
        step, _, step_range = spline._step_of_frame(
            n_frames, 10.0, fps, "z", None
        )
        segments = spline._reference_frame_segments(step, step_range)
        assert len(segments) == fps  # one in-focus block per FOV


class TestStepOfFrame:
    def test_one_frame_per_step(self):
        step, z_of_step, step_range = spline._step_of_frame(
            10, 20.0, 1, "fov", None
        )
        np.testing.assert_array_equal(step, np.arange(10))
        np.testing.assert_array_equal(step_range, np.arange(10))
        assert len(z_of_step) == 10

    def test_fov_order(self):
        step, _, step_range = spline._step_of_frame(10, 20.0, 2, "fov", None)
        np.testing.assert_array_equal(step, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        np.testing.assert_array_equal(step_range, np.arange(5))

    def test_z_order(self):
        step, _, _ = spline._step_of_frame(10, 20.0, 2, "z", None)
        np.testing.assert_array_equal(step, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    def test_frame_bounds_exclude(self):
        step, _, step_range = spline._step_of_frame(10, 20.0, 1, "fov", (2, 5))
        # frames outside [2, 5] are marked -1
        assert np.all(step[:2] == -1)
        assert np.all(step[6:] == -1)
        np.testing.assert_array_equal(step_range, [2, 3, 4, 5])

    def test_too_many_frames_per_step(self):
        with pytest.raises(ValueError):
            spline._step_of_frame(3, 20.0, 10, "fov", None)


class TestTemplateHelpers:
    def test_normalize_template(self):
        box, nz = BOX, 5
        vol = np.full((box, box, nz), 50.0, dtype=np.float32)
        vol[box // 2, box // 2, 2] = 50.0 + 800.0  # peak at focus slice 2
        template, bg, amp, photon_scale = spline._normalize_template(vol, 2)
        assert bg == pytest.approx(50.0, abs=1e-3)
        assert amp == pytest.approx(800.0, rel=1e-3)
        # peak of the normalized in-focus slice is ~1
        assert template[box // 2, box // 2, 2] == pytest.approx(1.0, abs=1e-3)
        assert photon_scale > 0

    def test_normalize_rejects_flat(self):
        vol = np.full((BOX, BOX, 3), 10.0, dtype=np.float32)
        with pytest.raises(ValueError):
            spline._normalize_template(vol, 1)

    def test_focus_step_picks_sharpest(self):
        box, nz = BOX, 5
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        vol = np.zeros((box, box, nz), dtype=np.float32)
        sigmas = [2.4, 1.8, 1.0, 1.8, 2.4]  # sharpest at index 2
        for k, s in enumerate(sigmas):
            vol[:, :, k] = np.exp(
                -((xx - c) ** 2 + (yy - c) ** 2) / (2 * s**2)
            )
        z_center, eff_sigma = spline._focus_step(vol)
        assert z_center == 2
        assert eff_sigma == pytest.approx(1.0, abs=0.4)

    def test_register_and_average_centers_beads(self):
        box, nz = BOX, 3
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        s = 1.1

        def gauss(cx, cy):
            return np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s**2)
            ).astype(np.float32)

        # two beads offset in opposite directions at the focus slice
        volumes = np.zeros((2, box, box, nz), dtype=np.float32)
        for k in range(nz):
            volumes[0, :, :, k] = gauss(c + 1.0, c)
            volumes[1, :, :, k] = gauss(c - 1.0, c)
        mean_vol = spline._register_and_average(volumes, z_center=1)
        # after centering, the averaged PSF peak should be at the box center
        focus = mean_vol[:, :, 1]
        peak_row, peak_col = np.unravel_index(np.argmax(focus), focus.shape)
        assert peak_row == c
        assert peak_col == c

    def test_register_and_average_centers_a_common_offset(self):
        # Every bead offset the SAME way: cross-correlation alignment is a
        # no-op here (they already agree), so only an explicit centering step
        # can put the average back on the box center. Left off-center, the
        # offset is baked into the template - a constant lateral bias that is
        # harmless for one channel but becomes an inter-channel
        # misregistration in a linked multichannel fit, where every channel
        # picks its own anchor bead.
        box, nz = BOX, 3
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        s = 1.1
        dx, dy = 1.0, -1.0
        volumes = np.zeros((3, box, box, nz), dtype=np.float32)
        for b in range(3):
            for k in range(nz):
                volumes[b, :, :, k] = np.exp(
                    -((xx - (c + dx)) ** 2 + (yy - (c + dy)) ** 2) / (2 * s**2)
                )
        mean_vol = spline._register_and_average(volumes, z_center=1)
        focus = mean_vol[:, :, 1]
        peak_row, peak_col = np.unravel_index(np.argmax(focus), focus.shape)
        assert (peak_row, peak_col) == (c, c)

    def test_focus_center_offset_matches_a_known_shift(self):
        box, nz = BOX, 3
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        dx, dy = 0.4, -0.7
        volume = np.zeros((box, box, nz), dtype=np.float32)
        for k in range(nz):
            volume[:, :, k] = np.exp(
                -((xx - (c + dx)) ** 2 + (yy - (c + dy)) ** 2) / (2 * 1.2**2)
            )
        d_row, d_col = spline._focus_center_offset(volume, z_center=1)
        assert d_row == pytest.approx(dy, abs=0.05)
        assert d_col == pytest.approx(dx, abs=0.05)

    def test_focus_center_offset_is_accurate_within_its_bound(self):
        # inside the cap the measurement plus one cubic shift must land on the
        # center; this is what makes a single centering pass enough
        from scipy.ndimage import shift as ndi_shift

        box, nz = BOX, 5
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        for d in (0.3, 0.8, 1.0):
            volume = np.zeros((box, box, nz), dtype=np.float32)
            for k in range(nz):
                volume[:, :, k] = np.exp(
                    -((xx - (c + d)) ** 2 + (yy - (c - d)) ** 2) / (2 * 1.1**2)
                )
            d_row, d_col = spline._focus_center_offset(volume, 2)
            assert d_col == pytest.approx(d, abs=0.02)
            assert d_row == pytest.approx(-d, abs=0.02)
            centered = ndi_shift(
                volume, shift=(-d_row, -d_col, 0.0), order=3, mode="nearest"
            )
            left = spline._focus_center_offset(centered, 2)
            assert max(abs(left[0]), abs(left[1])) < 0.02

    def test_focus_center_offset_refuses_beyond_the_cap(self):
        # a shift this large cannot be delivered on a small box (the PSF runs
        # off the edge), so the volume is left alone rather than half-moved
        box, nz = BOX, 5
        yy, xx = np.mgrid[0:box, 0:box]
        c = box // 2
        d = spline._RECENTER_MAX_SHIFT + 0.5
        volume = np.zeros((box, box, nz), dtype=np.float32)
        for k in range(nz):
            volume[:, :, k] = np.exp(
                -((xx - (c + d)) ** 2 + (yy - c) ** 2) / (2 * 1.1**2)
            )
        assert spline._focus_center_offset(volume, 2) == (0.0, 0.0)

    def test_focus_center_offset_rejects_implausible_shifts(self):
        # a bead average is centered to well under a pixel; a large estimate
        # means the estimate is wrong, and shifting on it would do harm
        box, nz = BOX, 3
        volume = np.zeros((box, box, nz), dtype=np.float32)
        volume[0, 0, :] = 1.0  # all the signal jammed into one corner
        assert spline._focus_center_offset(volume, z_center=1) == (0.0, 0.0)


def _volumes_with_one_outlier(box=BOX, nz=5, n_good=5, seed=0):
    """A stack of near-identical Gaussian beads plus one obvious outlier (a
    doublet), with a little noise so the robust spread is non-zero."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:box, 0:box]
    c = box // 2
    volumes = np.zeros((n_good + 1, box, box, nz), dtype=np.float32)
    for k in range(nz):
        s = 1.1 * (1.0 + 0.3 * abs(k - nz // 2))
        good = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s**2))
        for b in range(n_good):
            volumes[b, :, :, k] = 1000.0 * good + rng.normal(
                0.0, 3.0, (box, box)
            )
        doublet = np.exp(
            -((xx - c - 2) ** 2 + (yy - c) ** 2) / (2 * s**2)
        ) + np.exp(-((xx - c + 2) ** 2 + (yy - c) ** 2) / (2 * s**2))
        volumes[n_good, :, :, k] = 500.0 * doublet + rng.normal(
            0.0, 3.0, (box, box)
        )
    return volumes


class TestBeadFiltering:
    """The outlier filtering that decides which beads enter the PSF, and the
    per-bead record that makes that decision visible in the GUI."""

    def test_keep_inliers_reports_the_thresholds_it_applied(self):
        ncc = np.array([0.98, 0.97, 0.98, 0.99, 0.60])
        mse = np.array([1.0, 1.1, 0.9, 1.0, 50.0])
        keep, limits = spline._keep_inliers(ncc, mse)
        assert keep.tolist() == [True, True, True, True, False]
        # the thresholds are reported so the inspector can show the user
        # where the cut was made
        assert 0.6 < limits["ncc_min"] < 0.97
        assert 1.1 < limits["mse_max"] < 50.0
        assert limits["fallback"] is False

    def test_keep_inliers_flags_the_fallback(self):
        # both criteria together would reject nearly everything, so the
        # lowest-MSE half is kept instead - which is worth telling the user,
        # since the thresholds then say nothing about a rejected bead
        # two beads fail on correlation, two others on the residual: four of
        # six would go, so the lowest-MSE half is kept instead
        ncc = np.array([0.990, 0.991, 0.989, 0.992, 0.50, 0.51])
        mse = np.array([1.0, 1.01, 500.0, 600.0, 1.02, 1.03])
        keep, limits = spline._keep_inliers(ncc, mse)
        assert limits["fallback"] is True
        assert keep.tolist() == [True, True, False, False, True, False]

    def test_register_and_average_reports_every_bead(self):
        volumes = _volumes_with_one_outlier()
        mean_volume, registered, quality = spline._register_and_average(
            volumes, z_center=2, return_registered=True
        )
        # the doublet is dropped from the average ...
        assert quality["keep"][:-1].all()
        assert not quality["keep"][-1]
        assert registered.shape[0] == int(quality["keep"].sum())
        # ... but is still reported, with the measures behind the decision
        assert quality["registered_all"].shape[0] == len(volumes)
        assert quality["ncc"][-1] < quality["ncc"][:-1].min()
        assert quality["mse"][-1] > quality["mse"][:-1].max()

    def test_bead_quality_summary_keeps_the_central_views(self):
        volumes = _volumes_with_one_outlier()
        _, _, quality = spline._register_and_average(
            volumes, z_center=2, return_registered=True
        )
        beads = pd.DataFrame(
            {"x": np.arange(len(volumes)), "y": np.arange(len(volumes)) + 5}
        )
        summary = spline._bead_quality_summary(quality, beads, z_center=2)
        n_beads, box, _, nz = volumes.shape
        assert summary["xy"].shape == (n_beads, box, box)
        assert summary["xz"].shape == (n_beads, nz, box)
        assert summary["yz"].shape == (n_beads, nz, box)
        # each bead is normalized to its own focus peak, so beads are compared
        # by shape rather than by brightness
        assert summary["xy"].max() == pytest.approx(1.0, abs=0.05)
        assert summary["x"].tolist() == beads["x"].tolist()
        # the full volumes are dropped - only the views are kept around
        assert "registered_all" not in summary

    def test_build_template_records_the_filtering(self):
        movie, _, _ = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie, CAMERA_INFO, box=BOX, minimum_ng=2000.0, d=20.0
        )
        quality = built["bead_quality"]
        assert len(quality["keep"]) == built["n_beads"]
        assert built["n_beads_used"] == int(quality["keep"].sum())
        assert len(quality["x"]) == built["n_beads"]

    def test_inspection_data_and_rejection_reasons(self):
        movie, _, _ = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie, CAMERA_INFO, box=BOX, minimum_ng=2000.0, d=20.0
        )
        # pretend the last bead was rejected for both reasons
        quality = built["bead_quality"]
        quality["keep"][-1] = False
        quality["ncc"][-1] = 0.1
        quality["mse"][-1] = 1e9
        quality["ncc_min"], quality["mse_max"] = 0.9, 1.0
        data = spline.bead_inspection_data(
            built, {"z_center": float(built["z_center"]), "pixelsize": 130.0}
        )
        assert data["template_xy"].shape == (BOX, BOX)
        assert len(data["z_nm"]) == movie.shape[0]
        reasons = spline._rejection_reasons(data)
        assert reasons[:-1] == [""] * (len(reasons) - 1)
        assert reasons[-1] == "low correlation, high residual"

    def test_inspection_data_is_none_without_the_record(self):
        # a template built by an older version has no per-bead record; the
        # inspector must degrade rather than raise
        assert spline.bead_inspection_data({}, {}) is None

    def test_plot_bead_gallery_draws_every_bead(self):
        from matplotlib.figure import Figure

        movie, _, _ = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie, CAMERA_INFO, box=BOX, minimum_ng=2000.0, d=20.0
        )
        built["bead_quality"]["keep"][-1] = False
        data = spline.bead_inspection_data(
            built, {"z_center": float(built["z_center"]), "pixelsize": 130.0}
        )
        n_beads = len(data["keep"])
        fig = Figure()
        spline.plot_bead_gallery(data, fig)
        # three panels per bead plus the averaged-PSF cell, plus the scatter
        assert len(fig.axes) == 3 * (n_beads + 1) + 1
        # only the rejected bead (and the averaged PSF) when filtered
        fig = Figure()
        spline.plot_bead_gallery(data, fig, only_rejected=True)
        assert len(fig.axes) == 3 * 2 + 1

    def test_plot_bead_gallery_never_caps_the_rejected_beads(self):
        """The cap keeps a bead-rich z-stack readable, but must only drop
        beads that were kept - hiding a rejection would defeat the point."""
        from matplotlib.figure import Figure

        n, n_rejected, nz = 12, 4, 5
        keep = np.ones(n, dtype=bool)
        keep[:n_rejected] = False
        data = {
            "label": "",
            "keep": keep,
            "ncc": np.linspace(0.5, 1.0, n),
            "mse": np.linspace(10.0, 1.0, n),
            "ncc_min": 0.8,
            "mse_max": 5.0,
            "fallback": False,
            "x": np.arange(n, dtype=float),
            "y": np.arange(n, dtype=float),
            "xy": np.zeros((n, BOX, BOX), dtype=np.float32),
            "xz": np.zeros((n, nz, BOX), dtype=np.float32),
            "yz": np.zeros((n, nz, BOX), dtype=np.float32),
            "template_xy": np.zeros((BOX, BOX), dtype=np.float32),
            "template_xz": np.zeros((nz, BOX), dtype=np.float32),
            "template_yz": np.zeros((nz, BOX), dtype=np.float32),
            "z_nm": np.linspace(200.0, -200.0, nz),
            "pixelsize": 130.0,
        }
        fig = Figure()
        spline.plot_bead_gallery(data, fig, max_beads=6)
        # the averaged PSF, all 4 rejected beads and 2 kept ones
        assert len(fig.axes) == 3 * (1 + n_rejected + 2) + 1
        # a cap below the number of rejected beads still shows all of them
        fig = Figure()
        spline.plot_bead_gallery(data, fig, max_beads=1)
        assert len(fig.axes) == 3 * (1 + n_rejected) + 1

    def test_n_beads_used_reads_either_calibration_layout(self):
        # a multichannel calibration counts per channel; the reference
        # channel's count is the one to report
        assert spline.n_beads_used({"n_beads": 20, "n_beads_used": 17}) == 17
        assert (
            spline.n_beads_used({"n_beads": 20, "n_beads_used": [17, 15]})
            == 17
        )
        # a calibration built before the filtering was recorded
        assert spline.n_beads_used({"n_beads": 20}) == 20

    def test_calibrate_multichannel_records_every_channel(self, tmp_path):
        """Every channel builds its own PSF from its own beads, so each one
        gets its own filtering record and gallery - the reference channel's
        rejections say nothing about the others."""
        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = np.roll(movie_ref, shift=(2, -1), axis=(1, 2))
        info = [{"Frames": int(movie_ref.shape[0])}]
        path = str(tmp_path / "mc_spline_calib.hdf5")
        calib, diagnostics = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            path=path,
            return_diagnostics=True,
        )
        assert len(calib["n_beads_used"]) == 2
        assert spline.n_beads_used(calib) == calib["n_beads_used"][0]
        assert len(diagnostics) == 2
        assert diagnostics[0]["label"].startswith("reference channel")
        assert diagnostics[1]["label"].startswith("channel 1")
        for data in diagnostics:
            assert len(data["keep"]) == calib["n_beads"]
        for channel in (0, 1):
            assert os.path.exists(
                str(tmp_path / f"mc_spline_calib_ch{channel}_beads.png")
            )
        # the galleries must not have cost us the cross-channel diagnostics
        assert os.path.exists(str(tmp_path / "mc_spline_calib_summary.png"))
        assert os.path.exists(
            str(tmp_path / "mc_spline_calib_registration.png")
        )

    def test_calibrate_spline_writes_the_bead_gallery(self, tmp_path):
        movie, _, _ = _synthetic_bead_movie()
        path = str(tmp_path / "bead_spline_calib.hdf5")
        calib, diagnostics = spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            path=path,
            return_diagnostics=True,
        )
        assert os.path.exists(str(tmp_path / "bead_spline_calib_beads.png"))
        assert calib["n_beads_used"] <= calib["n_beads"]
        assert len(diagnostics) == 1
        assert len(diagnostics[0]["keep"]) == calib["n_beads"]


class TestBuildPsfTemplate:
    """End-to-end PSF template building on a synthetic bead movie (no GPU)."""

    def test_build_template(self):
        movie, bead_xy, focus = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie,
            CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
        )
        template = built["template"]
        assert template.shape == (BOX, BOX, movie.shape[0])
        # at least some beads detected
        assert built["n_beads"] >= 2
        # focus recovered near the true central frame
        assert abs(built["z_center"] - focus) <= 2
        # normalized template: focus peak ~1, minimum ~0
        assert template[:, :, built["z_center"]].max() == pytest.approx(
            1.0, abs=0.05
        )
        assert template.min() == pytest.approx(0.0, abs=0.1)
        assert built["effective_sigma"] > 0


def _labeled_bead_movie(n_frames, fov_of=None, h=48, w=48):
    """Beads with distinguishable brightness, so a spot can be traced back to
    the bead it came from by its peak value alone."""
    bead_xy = [(12, 14), (30, 28), (16, 33)]
    amps = [1000.0, 2000.0, 3000.0]
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 100.0, dtype=np.float32)
    for (bx, by), a in zip(bead_xy, amps):
        img += a * np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * 1.1**2))
    movie = np.stack([img] * n_frames).astype(np.uint16)
    return movie, bead_xy, amps


def _peaks(spots):
    return spots.reshape(len(spots), -1).max(axis=1)


class TestSpotBeadIndex:
    """``_bead_volumes`` must report which bead each flattened spot came from -
    the two extraction paths flatten in different orders, so no positional rule
    recovers it for both."""

    def test_single_field_spot_bead_idx(self):
        import pandas as pd

        n = 6
        movie, bead_xy, amps = _labeled_bead_movie(n)
        step_of_frame, _, step_range = spline._step_of_frame(
            n, 20.0, 1, "fov", None
        )
        beads = pd.DataFrame(
            {"x": [b[0] for b in bead_xy], "y": [b[1] for b in bead_xy]}
        )
        _, spots, _, bead_idx = spline._bead_volumes(
            movie,
            CAMERA_INFO,
            beads,
            BOX,
            step_of_frame,
            step_range,
            return_spots=True,
        )
        assert len(bead_idx) == len(spots)
        expected = np.array([amps[b] for b in bead_idx])
        np.testing.assert_allclose(_peaks(spots), expected, atol=250)

    def test_multifov_spot_bead_idx(self):
        import pandas as pd

        # the multi-FOV grid is built bead-major and then re-sorted by frame
        # (lazy movies need frame-sorted identifications), so the flattened
        # order is neither bead-major nor frame-major-with-all-beads
        n_fov, n_steps = 2, 5
        n_frames = n_fov * n_steps
        movie, bead_xy, amps = _labeled_bead_movie(n_frames)
        step_of_frame, _, step_range = spline._step_of_frame(
            n_frames, 20.0, n_fov, "z", None
        )
        fov_of_frame = spline._fov_of_frame(n_frames, n_fov, "z")
        beads = pd.DataFrame(
            {
                "x": [b[0] for b in bead_xy],
                "y": [b[1] for b in bead_xy],
                "fov": [0, 1, 0],
            }
        )
        _, spots, _, bead_idx = spline._bead_volumes(
            movie,
            CAMERA_INFO,
            beads,
            BOX,
            step_of_frame,
            step_range,
            return_spots=True,
            fov_of_frame=fov_of_frame,
        )
        peaks = _peaks(spots)
        np.testing.assert_allclose(
            peaks, np.array([amps[b] for b in bead_idx]), atol=250
        )
        # and the obvious positional shortcut really is wrong here, which is
        # why the index is returned rather than recomputed by callers
        positional = np.array([amps[i % len(amps)] for i in range(len(spots))])
        assert not np.allclose(peaks, positional, atol=250)


class TestSpotRoiGeometry:
    def test_expands_per_bead_geometry_onto_spots(self):
        import pandas as pd

        ref_xy = np.array([[10.0, 20.0], [31.0, 12.0], [7.0, 44.0]])
        transforms = [
            affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            affine([[1.002, 0.004, 1.4], [-0.004, 1.001, -0.6]]),
        ]
        spot_bead_idx = np.array([2, 0, 1, 2, 0, 1])
        per_channel = [{"spot_bead_idx": spot_bead_idx} for _ in range(2)]
        res, jac = spline._spot_roi_geometry(per_channel, ref_xy, transforms)
        assert res.shape == (len(spot_bead_idx), 2, 2)
        assert jac.shape == (len(spot_bead_idx), 2, 4)
        per_bead_res, per_bead_jac = localize.channel_roi_geometry(
            pd.DataFrame({"x": ref_xy[:, 0], "y": ref_xy[:, 1]}), transforms
        )
        np.testing.assert_allclose(res, per_bead_res[spot_bead_idx])
        np.testing.assert_allclose(jac, per_bead_jac[spot_bead_idx])
        # the reference channel's box sits on the detection itself
        np.testing.assert_array_equal(res[:, 0, :], 0.0)

    def test_projective_jacobians_vary_across_the_field(self):
        # the whole point of carrying jacobians per spot: a projective
        # registration has no single per-channel value, so the joint refit
        # cannot run without them (it used to raise and silently drop the
        # summary plot's z-accuracy panels)
        ref_xy = np.array([[10.0, 20.0], [400.0, 380.0]])
        transforms = [
            affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            {
                "model": "projective",
                "matrix": [
                    [1.002, 0.004, 1.4],
                    [-0.004, 1.001, -0.6],
                    [1e-4, -8e-5, 1.0],
                ],
                "domain": None,
            },
        ]
        spot_bead_idx = np.array([0, 1])
        per_channel = [{"spot_bead_idx": spot_bead_idx} for _ in range(2)]
        _, jac = spline._spot_roi_geometry(per_channel, ref_xy, transforms)
        assert not np.allclose(jac[0, 1], jac[1, 1])

    def test_returns_none_when_channels_disagree(self):
        ref_xy = np.array([[10.0, 20.0], [31.0, 12.0]])
        transforms = [np.eye(2, 3), np.eye(2, 3)]
        # the caller stacks row i of every channel as one bead+frame, so a
        # disagreement means that stacking is wrong too - bail rather than
        # attach residuals to the wrong spots
        per_channel = [
            {"spot_bead_idx": np.array([0, 1])},
            {"spot_bead_idx": np.array([1, 0])},
        ]
        assert (
            spline._spot_roi_geometry(per_channel, ref_xy, transforms) is None
        )
        # and missing indices are simply "no information"
        assert (
            spline._spot_roi_geometry(
                [{"spot_bead_idx": np.array([0, 1])}, {}], ref_xy, transforms
            )
            is None
        )


class TestMultiFov:
    """Genuine multi-FOV z-stacks: several fields with *different* beads at
    *different* positions. Beads must be detected and extracted per FOV (each
    from its own field's frames), never averaged across fields."""

    def test_detects_beads_from_all_fovs_with_labels(self):
        movie, fov_beads, _ = _synthetic_multifov_movie(n_fov=3, order="z")
        n_fov = len(fov_beads)
        n_total = sum(len(b) for b in fov_beads)
        n_frames = movie.shape[0]
        step_of_frame, _, step_range = spline._step_of_frame(
            n_frames, 20.0, n_fov, "z", None
        )
        fov_of_frame = spline._fov_of_frame(n_frames, n_fov, "z")
        segments = spline._reference_frame_segments(step_of_frame, step_range)
        beads = spline._detect_bead_positions(
            movie, 2000.0, BOX, segments, fov_of_frame=fov_of_frame
        )
        assert "fov" in beads.columns
        # every field's beads are found (pooling would merge/lose some)
        assert len(beads) == n_total
        assert sorted(beads["fov"].unique()) == list(range(n_fov))
        for k in range(n_fov):
            assert (beads["fov"] == k).sum() == len(fov_beads[k])

    def test_bead_volume_is_isolated_to_its_own_fov(self):
        """A bead's volume must come only from its own FOV's frames: a bright
        contaminant at the same pixel in another FOV must not leak in (the bug
        that corrupted cross-FOV pixel-averaging)."""
        import pandas as pd

        n_steps, h, w = 5, 24, 24
        yy, xx = np.mgrid[0:h, 0:w]
        bx, by = 10, 10

        def blob(amp, sigma):
            return amp * np.exp(
                -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
            )

        frames = []
        # FOV0: real bead, amplitude 3000; FOV1: bright contaminant 12000 at the
        # SAME pixel (z order: FOV0 stack, then FOV1 stack)
        for k in range(n_steps):
            frames.append(blob(3000.0, 1.1 * (1 + 0.1 * abs(k - 2))))
        for k in range(n_steps):
            frames.append(blob(12000.0, 1.1))
        movie = np.stack(frames).astype(np.uint16)

        step_of_frame, _, step_range = spline._step_of_frame(
            2 * n_steps, 20.0, 2, "z", None
        )
        fov_of_frame = spline._fov_of_frame(2 * n_steps, 2, "z")
        beads = pd.DataFrame({"x": [bx], "y": [by], "fov": [0]})
        vols = spline._bead_volumes(
            movie,
            CAMERA_INFO,
            beads,
            BOX,
            step_of_frame,
            step_range,
            fov_of_frame=fov_of_frame,
        )
        peak = vols[0].max()
        # FOV0-only peak ~3000; cross-FOV averaging would give ~7500, and the
        # raw contaminant is 12000. Must be the isolated FOV0 value.
        assert peak == pytest.approx(3000.0, rel=0.15)
        assert peak < 5000.0

    def test_build_template_multifov_is_clean(self):
        """End-to-end: a multi-FOV stack yields a well-focused template built
        from all fields' beads (the per-FOV path), for both frame orders."""
        for order in ("z", "fov"):
            movie, fov_beads, focus = _synthetic_multifov_movie(
                n_fov=3, order=order
            )
            n_fov = len(fov_beads)
            n_total = sum(len(b) for b in fov_beads)
            built = spline.build_psf_template(
                movie,
                CAMERA_INFO,
                box=BOX,
                minimum_ng=2000.0,
                d=20.0,
                frames_per_step=n_fov,
                frame_order=order,
            )
            # beads pooled from all fields
            assert built["n_beads"] == n_total
            # focus recovered near the central step; clean normalized template
            assert abs(built["z_center"] - focus) <= 2
            tpl = built["template"]
            assert tpl[:, :, built["z_center"]].max() == pytest.approx(
                1.0, abs=0.05
            )
            assert tpl.min() == pytest.approx(0.0, abs=0.1)


class TestCalibrateSpline:
    """Full calibration including the spline-coefficient step (CPU)."""

    def test_calibrate_spline_3d_roundtrip(self, tmp_path):
        from picasso import io

        movie, _, _ = _synthetic_bead_movie()
        path = str(tmp_path / "bead_spline_calib.hdf5")
        calib = spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            model="spline-3d",
            path=path,
        )
        assert calib["model"] == "spline-3d"
        assert calib["coefficients"].shape[0] == 64
        assert list(calib["n_data"]) == [BOX, BOX, movie.shape[0]]
        # the saved calibration loads and can drive the fitter
        loaded = io.load_spline_calibration(path)
        coefficients = precision._spline_coeff_reshaped(loaded)
        assert coefficients.ndim == 7  # (C, niz, niy, nix, 4, 4, 4)

    def test_calibrate_spline_2d(self):
        movie, _, _ = _synthetic_bead_movie()
        calib = spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            model="spline-2d",
        )
        assert calib["model"] == "spline-2d"
        assert calib["coefficients"].shape[0] == 16
        assert list(calib["n_data"]) == [BOX, BOX]


# ---------------------------------------------------------------------------
# Multichannel calibration (Session D) - registration/matching without a GPU
# ---------------------------------------------------------------------------


class TestMultichannelCalibration:
    def test_match_points(self):
        ref = np.array([[0, 0], [10, 10], [20, 20]], dtype=float)
        other = np.array([[20.3, 20.1], [0.2, -0.1], [100, 100]], dtype=float)
        ref_idx, other_idx = spline.match_points(ref, other, 1.0)
        # ref[0]->other[1], ref[2]->other[0]; ref[1] has no match within 1 px
        assert ref_idx.tolist() == [0, 2]
        assert other_idx.tolist() == [1, 0]

    def test_match_points_unique_targets(self):
        ref = np.array([[0, 0], [0.5, 0]], dtype=float)
        other = np.array([[0.1, 0.0]], dtype=float)
        ref_idx, other_idx = spline.match_points(ref, other, 5.0)
        # both refs are near the single target; it must be used only once
        assert len(other_idx) == 1
        assert ref_idx.tolist() == [0]  # closest reference wins


class TestRansacMatch:
    """RANSAC bead matching that makes the channel registration robust to the
    coarse (ROI-origin) pre-alignment - the fix for the ROI-placement
    hypersensitivity of split-FOV calibrations."""

    @staticmethod
    def _mirrored_pair():
        # six reference beads and their images under a known y-mirror + small
        # rotation + shift (a realistic biplane registration)
        ref = np.array(
            [[10, 12], [40, 18], [25, 55], [60, 50], [15, 70], [50, 82]],
            dtype=float,
        )
        theta = np.deg2rad(3.0)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        flip = np.array([[1.0, 0.0], [0.0, -1.0]])  # mirror in y
        linear = rot @ flip
        offset = np.array([7.0, 190.0])
        c = ref @ linear.T + offset
        return ref, c, linear, offset

    @pytest.mark.parametrize("overlay_offset", [(0.0, 0.0), (14.0, -11.0)])
    def test_recovers_transform_despite_bad_overlay(self, overlay_offset):
        """A wrong coarse overlay (a misplaced ROI) must not change the result:
        the correct correspondences and transform are recovered regardless."""
        ref, c, linear, offset = self._mirrored_pair()
        # the coarse overlay maps c back near ref but is deliberately off; a plain
        # nearest-neighbor match at this offset would mis-pair some beads
        inv = np.linalg.inv(linear)
        aligned = (c - offset) @ inv.T + np.asarray(overlay_offset)
        ref_idx, c_idx = spline.ransac_match(
            ref, c, aligned, inlier_tol=3.0, radius=40.0
        )
        assert len(ref_idx) == len(ref)  # all beads matched
        # identity correspondence (c[i] is the image of ref[i]) is recovered
        order = np.argsort(ref_idx)
        assert ref_idx[order].tolist() == list(range(len(ref)))
        assert c_idx[order].tolist() == list(range(len(ref)))
        # and the transform fit on them matches the truth (mirror -> det < 0)
        M = transforms.estimate(ref[ref_idx], c[c_idx])
        np.testing.assert_allclose(linear_part(M), linear, atol=0.02)
        assert np.linalg.det(linear_part(M)) < 0

    def test_rejects_decoy_and_is_overlay_independent(self):
        """Extra unmatched channel beads (decoys) are rejected, and two very
        different overlays give the same correspondences."""
        ref, c, _, offset = self._mirrored_pair()
        c_dec = np.vstack([c, [[200.0, 5.0], [5.0, 5.0]]])  # 2 decoys
        results = []
        for off in [(0.0, 0.0), (18.0, 16.0)]:
            aligned = np.vstack(
                [ref + np.asarray(off), [[999.0, 999.0], [-999.0, -999.0]]]
            )
            ri, ci = spline.ransac_match(
                ref, c_dec, aligned, inlier_tol=3.0, radius=45.0
            )
            results.append((ri.tolist(), ci.tolist()))
            assert len(ri) == len(ref)  # decoys excluded
            assert max(ci) < len(c)  # only real channel beads matched
        assert results[0] == results[1]  # overlay-independent

    def test_estimate_channel_transform_recovers_shift(self):
        movie_ref, _, _ = _synthetic_bead_movie()
        dx, dy = 3, -2  # channel is the reference shifted by (dx, dy)
        movie_c = np.roll(movie_ref, shift=(dy, dx), axis=(1, 2))

        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        beads_ref = spline._detect_bead_positions(
            movie_ref, 2000.0, BOX, ref_bounds
        )
        transform, n_matches = spline._estimate_channel_transform(
            movie_ref,
            movie_c,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
        )
        assert n_matches >= 3
        # transform maps reference (x, y) -> channel (x + dx, y + dy)
        np.testing.assert_allclose(
            transform.matrix[:2],
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]]),
            atol=0.6,
        )

    def test_estimate_channel_transform_recovers_flip(self):
        """Separate movies where the channel is a vertically mirrored copy of
        the reference (a reflected optical path). The flip-aware coarse matching
        must still register it: all beads match and the affine is a reflection.
        """
        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = movie_ref[:, ::-1, :]  # mirror in y (up/down)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        beads_ref = spline._detect_bead_positions(
            movie_ref, 2000.0, BOX, ref_bounds
        )
        n_ref = len(beads_ref)
        transform, n_matches = spline._estimate_channel_transform(
            movie_ref,
            movie_c,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
        )
        assert n_matches == n_ref
        # a pure translation could never register a mirror: reflection -> det < 0
        assert np.linalg.det(linear_part(transform)) < 0
        # ref (x, y) -> channel (x, H - 1 - y)
        h = movie_ref.shape[1]
        ref_xy = beads_ref[["x", "y"]].to_numpy(float)
        mapped = apply_transform(ref_xy, transform)
        np.testing.assert_allclose(mapped[:, 0], ref_xy[:, 0], atol=1.0)
        np.testing.assert_allclose(
            mapped[:, 1], h - 1 - ref_xy[:, 1], atol=1.0
        )


class TestCalibrateSplineMultichannel:
    """Full multichannel calibration including the spline-coefficient
    step (CPU)."""

    @pytest.mark.parametrize("model", transforms.MODELS)
    def test_registration_model_is_used_and_recorded(self, tmp_path, model):
        """The chosen model is fitted, stored, and survives the HDF5 JSON
        round-trip - and the fit path can still read it back."""
        from picasso import io

        # a degree-3 polynomial needs 10 well-spread correspondences, so
        # this test uses a denser bead field than the shared fixture
        # jittered off a perfect rectangular grid: an exact grid is rank
        # deficient for a degree-3 polynomial (only 10 distinct monomials)
        rng = np.random.RandomState(3)
        beads = [
            (x + rng.uniform(-3, 3), y + rng.uniform(-3, 3))
            for x in (9, 20, 31, 40)
            for y in (10, 22, 34)
        ]
        movie_ref, _, _ = _synthetic_bead_movie(bead_xy=beads)
        movie_c = np.roll(movie_ref, shift=(2, -1), axis=(1, 2))
        info = [{"Frames": int(movie_ref.shape[0])}]
        path = str(tmp_path / f"mc_{model}.hdf5")
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            path=path,
            model=model,
        )
        stored = calib["channel_transforms"][1]
        assert stored["model"] == model
        assert spline.registration_model_name(calib) == model
        loaded = io.load_spline_calibration(path)
        reloaded = transforms.from_dict(loaded["channel_transforms"][1])
        # the shift the channel was built with, whatever the model
        probe = np.array([[20.0, 20.0], [30.0, 25.0]])
        np.testing.assert_allclose(
            reloaded.apply(probe), probe + [-1.0, 2.0], atol=0.6
        )

    def test_calibrate_multichannel(self, tmp_path):
        from picasso import io

        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = np.roll(movie_ref, shift=(2, -1), axis=(1, 2))
        info = [{"Frames": int(movie_ref.shape[0])}]
        path = str(tmp_path / "mc_spline_calib.hdf5")
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            photon_ratios=[[0.7, 0.3], [0.4, 0.6]],
            path=path,
        )
        assert calib["model"] == "spline-3d-multichannel"
        assert calib["n_channels"] == 2
        # candidate photon ratios stored for ratiometric color assignment,
        # and they survive the HDF5 round-trip (JSON metadata)
        np.testing.assert_allclose(
            calib["photon_ratios"], [[0.7, 0.3], [0.4, 0.6]]
        )
        assert io.load_spline_calibration(path)["photon_ratios"] is not None
        assert calib["coefficients"].shape[0] == 64
        assert calib["coefficients"].shape[-1] == 2
        assert len(calib["channel_transforms"]) == 2
        # round-trips and drives the multichannel fitter
        loaded = io.load_spline_calibration(path)
        coefficients = precision._spline_coeff_reshaped(loaded)
        assert coefficients.shape[0] == 2  # one coefficient block per channel
        # co-focal channels: both planes at the same focus
        np.testing.assert_allclose(
            calib["plane_offsets"], [0.0, 0.0], atol=25.0
        )
        # per-channel photon_scale is stored as a list (one per channel)
        assert len(calib["photon_scale"]) == 2

    def test_calibrate_biplane_recovers_plane_offset(self, tmp_path):
        """Biplane: the second channel is the same z-stack with its focus at a
        different stage step (a frame-axis roll). The calibration must recover
        a non-zero plane offset of the right magnitude, while keeping the two
        channels laterally registered (identity transform)."""
        movie_ref, _, focus = _synthetic_bead_movie(n_frames=21)
        d_nm = 20.0
        delta_steps = 3  # channel 1 focuses 3 steps deeper
        movie_c = np.roll(movie_ref, shift=delta_steps, axis=0)
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=d_nm,
            path=str(tmp_path / "biplane_spline_calib.hdf5"),
        )
        offsets = calib["plane_offsets"]
        assert offsets[0] == 0.0
        # channel 1 focus offset ~ delta_steps * d (within ~1 step)
        np.testing.assert_allclose(
            offsets[1], delta_steps * d_nm, atol=1.5 * d_nm
        )

    def test_calibrate_separate_channels_mirrored(self, tmp_path):
        """Separate-movie channels where channel 1 is a vertical mirror of the
        reference (reflected optical path). The build must register it (all
        beads matched) and store a reflection transform, so the per-channel
        template is built at the real mirrored bead positions."""
        movie_ref, _, _ = _synthetic_bead_movie()
        movie_c = movie_ref[:, ::-1, :]  # mirror in y
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            path=str(tmp_path / "mirrored_spline_calib.hdf5"),
        )
        assert calib["n_channels"] == 2
        # channel-1 transform is a reflection (negative determinant), not the
        # garbage a translation-only match would have produced
        t1 = affine_matrix(calib["channel_transforms"][1])
        assert np.linalg.det(linear_part(calib["channel_transforms"][1])) < 0
        # the transform maps into the mirrored frame: y -> H - 1 - y
        h = movie_ref.shape[1]
        assert abs(t1[1, 1] + 1.0) < 0.1  # y scale ~ -1
        assert abs(t1[1, 2] - (h - 1)) < 2.0  # y offset ~ H - 1

    def test_axial_precision_multichannel_is_joint(self):
        """The multichannel axial-precision diagnostic fits jointly.

        It must fit all channels *jointly* (the real pipeline) rather than
        each plane alone. This checks
        the joint contract: it stacks every channel's per-frame spots, fits them
        against the full calibration, tags the result as a joint N-channel fit
        and returns one bias/precision sample per z-step. (Degeneracy-breaking
        needs realistic aberrated PSFs; the symmetric synthetic Gaussian here is
        z-degenerate even jointly, so no tight bias bound is asserted.)"""
        import pandas as pd

        movie_ref, _, _ = _synthetic_bead_movie()
        # biplane-style: channel 1 focuses at a different stage step
        movie_c = np.roll(movie_ref, shift=2, axis=0)
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
        )
        # rebuild each channel's per-frame spots at its (mapped) bead positions,
        # exactly as calibrate_spline_multichannel does internally
        transforms = calib["channel_transforms"]
        movies = [movie_ref, movie_c]
        step_of_frame, _, step_range = spline._step_of_frame(
            movie_ref.shape[0], 20.0, 1, "fov", None
        )
        rb = spline._reference_frame_segments(step_of_frame, step_range)
        beads_ref = spline._detect_bead_positions(movie_ref, 2000.0, BOX, rb)
        ref_xy = beads_ref[["x", "y"]].to_numpy(float)
        per_channel = []
        for c, m in enumerate(movies):
            if c == 0:
                beads_c = beads_ref
            else:
                mp = apply_transform(ref_xy, transforms[c])
                beads_c = pd.DataFrame(
                    {
                        "x": np.rint(mp[:, 0]).astype(int),
                        "y": np.rint(mp[:, 1]).astype(int),
                    }
                )
            per_channel.append(
                spline.build_psf_template(
                    m,
                    CAMERA_INFO,
                    BOX,
                    2000.0,
                    20.0,
                    beads=beads_c,
                    return_spots=True,
                )
            )
        prec = spline._axial_precision_multichannel(per_channel, calib)
        assert prec is not None
        assert prec["joint"] == 2  # tagged as a joint 2-channel fit
        z = np.asarray(per_channel[0]["z_of_step"], float)
        # one bias/precision sample per z-step, and the joint fit produced
        # finite z estimates for a good fraction of spots
        assert len(prec["bias_z"]) == len(z)
        assert len(prec["precision_z"]) == len(z)
        assert np.any(np.isfinite(prec["bias_z"]))
        assert prec["n_spots"] > 0
        assert len(prec["scatter_fit"]) == len(prec["scatter_stage"]) > 0


def _synthetic_split_fov_movie(dx=2, dy=-1):
    """A single movie whose left and right 48x48 halves are two channels.

    The right half (region 1) is the left half (region 0, reference) shifted
    within its region by ``(dx, dy)`` pixels, so the ref->region-1 affine is a
    pure translation ``[[1, 0, 48 + dx], [0, 1, dy]]`` in absolute chip
    coordinates. Returns ``(movie, regions, bead_xy)`` where ``bead_xy`` are the
    reference-region bead centers.
    """
    base, bead_xy, _focus = _synthetic_bead_movie(h=48, w=48)
    n = base.shape[0]
    movie = np.zeros((n, 48, 96), dtype=np.uint16)
    movie[:, :, :48] = base
    # small shift stays well inside the region, so np.roll wrap-around is moot
    movie[:, :, 48:] = np.roll(base, shift=(dy, dx), axis=(1, 2))
    regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    return movie, regions, bead_xy


def _synthetic_split_fov_movie_flipped(axis="y"):
    """Single movie whose right half is the left half *mirrored* (as biplane /
    spectral splitters do with a reflected optical path).

    ``axis="y"`` flips up/down, ``"x"`` left/right. Returns ``(movie, regions)``.
    A pure-translation coarse alignment cannot register a mirrored channel, so
    this exercises the flip-aware matching in ``_estimate_channel_transform``.
    """
    base, _bead_xy, _focus = _synthetic_bead_movie(h=48, w=48)
    n = base.shape[0]
    movie = np.zeros((n, 48, 96), dtype=np.uint16)
    movie[:, :, :48] = base
    flipped = base[:, ::-1, :] if axis == "y" else base[:, :, ::-1]
    movie[:, :, 48:] = flipped
    regions = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
    return movie, regions


class TestChannelTransformMultiFov:
    """The channel cloud must be de-duplicated by the same rule as the
    reference cloud it is matched against."""

    @staticmethod
    def _movie(n_steps=11):
        """Split-FOV, 2 fields, whose bead sets deliberately land within a box
        of each other *across* fields - the case a global de-duplication
        wrongly merges. Left half is the reference region, right half the
        channel (a pure +48 px translation)."""
        h = w = 48
        # FOV 1's beads sit 4 px from FOV 0's: same physical field would never
        # place them that close, different fields routinely do.
        fov_beads = [[(12, 14), (30, 28)], [(16, 18), (34, 32)]]
        s0, focus = 1.1, n_steps // 2
        yy, xx = np.mgrid[0:h, 0:w]
        frames = []
        for beads in fov_beads:  # "z" order: a full z stack per field
            for k in range(n_steps):
                sigma = s0 * (1.0 + 0.07 * abs(k - focus))
                img = np.full((h, w), 100.0, dtype=np.float32)
                for bx, by in beads:
                    img += 3000.0 * np.exp(
                        -((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma**2)
                    )
                frames.append(img)
        half = np.stack(frames).astype(np.uint16)
        movie = np.zeros((len(frames), h, 2 * w), dtype=np.uint16)
        movie[:, :, :w] = half
        movie[:, :, w:] = half
        regions = [[[0, 0], [h, w]], [[0, w], [h, 2 * w]]]
        return movie, regions, fov_beads

    def test_channel_beads_are_deduped_per_fov(self):
        movie, regions, fov_beads = self._movie()
        n_fov = len(fov_beads)
        n_physical = sum(len(b) for b in fov_beads)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie.shape[0], 20.0, n_fov, "z", None
        )
        fov_of_frame = spline._fov_of_frame(movie.shape[0], n_fov, "z")
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        ref_roi = spline._normalized_region(regions[0])
        chan_roi = spline._normalized_region(regions[1])

        beads_ref = spline._detect_bead_positions(
            movie,
            2000.0,
            BOX,
            ref_bounds,
            roi=ref_roi,
            fov_of_frame=fov_of_frame,
        )
        assert len(beads_ref) == n_physical  # every field's beads survive

        captured = {}
        original = spline._detect_bead_positions

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            captured["beads_c"] = out
            captured["fov_of_frame"] = kwargs.get("fov_of_frame")
            return out

        spline._detect_bead_positions = spy
        try:
            _, n_matches = spline._estimate_channel_transform(
                movie,
                movie,
                beads_ref,
                2000.0,
                BOX,
                ref_bounds,
                mid,
                max_distance=float(BOX),
                channel_roi=chan_roi,
                coarse_shift=(-48.0, 0.0),
                fov_of_frame=fov_of_frame,
            )
        finally:
            spline._detect_bead_positions = original

        # the channel is detected with the very same FOV labels ...
        assert captured["fov_of_frame"] is not None
        np.testing.assert_array_equal(captured["fov_of_frame"], fov_of_frame)
        # ... so it keeps as many beads as the reference, instead of losing
        # the ones a global de-duplication merges across fields
        assert len(captured["beads_c"]) == len(beads_ref) == n_physical
        assert n_matches == n_physical

    def test_matching_pairs_only_within_a_field(self):
        """A reference bead whose own field holds no partner must stay
        unmatched rather than be paired to a nearby *other* field's bead.

        The registration is the identity, so every legitimate pair is exact.
        Reference bead (100, 100) belongs to FOV 0, which has no counterpart
        for it; FOV 1 has an unpaired channel bead 1.4 px away. Pooled, that
        bead is the nearest neighbor and wins - a correspondence between two
        different physical beads in two different fields.
        """
        ref_xy = np.array(
            [
                [10.0, 10.0],
                [40.0, 40.0],
                [70.0, 70.0],
                [100.0, 100.0],
                [11.0, 11.0],
                [41.0, 41.0],
                [71.0, 71.0],
            ]
        )
        ref_fov = np.array([0, 0, 0, 0, 1, 1, 1])
        c_xy = np.array(
            [
                [10.0, 10.0],
                [40.0, 40.0],
                [70.0, 70.0],
                [11.0, 11.0],
                [41.0, 41.0],
                [71.0, 71.0],
                [101.0, 101.0],
            ]
        )
        c_fov = np.array([0, 0, 0, 1, 1, 1, 1])
        lone_ref = 3  # ref (100, 100), FOV 0
        lone_c = 6  # channel (101, 101), FOV 1

        # pooled (no labels): the cross-field pair is made
        ri, ci = spline.ransac_match(
            ref_xy, c_xy, c_xy, inlier_tol=3.0, radius=15.0
        )
        pooled = dict(zip(ri.tolist(), ci.tolist()))
        assert pooled.get(lone_ref) == lone_c
        assert np.any(ref_fov[ri] != c_fov[ci])

        # per field: it is refused, and every other pair is unaffected
        ri, ci = spline.ransac_match(
            ref_xy,
            c_xy,
            c_xy,
            inlier_tol=3.0,
            radius=15.0,
            ref_fov=ref_fov,
            c_fov=c_fov,
        )
        np.testing.assert_array_equal(ref_fov[ri], c_fov[ci])
        assert lone_ref not in ri.tolist()
        assert lone_c not in ci.tolist()
        assert len(ri) == 6  # the six genuine correspondences, all of them
        np.testing.assert_allclose(ref_xy[ri], c_xy[ci])

    def test_fov_groups_drops_fields_missing_on_one_side(self):
        # a field with no counterpart cannot yield a correspondence; letting
        # its beads search other fields is the mis-pairing we are preventing
        groups = registration._fov_groups(
            np.array([0, 0, 2]), np.array([0, 0, 1]), 3, 3
        )
        assert len(groups) == 1
        np.testing.assert_array_equal(groups[0][0], [0, 1])
        np.testing.assert_array_equal(groups[0][1], [0, 1])
        # missing / mismatched labels fall back to one pooled cloud
        assert registration._fov_groups(None, np.array([0]), 1, 1) is None
        assert (
            registration._fov_groups(np.array([0, 1]), np.array([0]), 5, 1)
            is None
        )

    def test_degenerate_registration_is_rejected(self):
        """A consensus of exactly min_points fits the model exactly, so its
        residual is zero however wrong the pairing was. Only the geometry can
        catch it, and it must be an error rather than a silent calibration."""
        movie, regions, _ = self._movie()
        n_fov = 2
        step_of_frame, _, step_range = spline._step_of_frame(
            movie.shape[0], 20.0, n_fov, "z", None
        )
        fov_of_frame = spline._fov_of_frame(movie.shape[0], n_fov, "z")
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        beads_ref = spline._detect_bead_positions(
            movie,
            2000.0,
            BOX,
            ref_bounds,
            roi=spline._normalized_region(regions[0]),
            fov_of_frame=fov_of_frame,
        )
        # Map 3 widely spread reference beads onto 3 nearly coincident channel
        # beads: an exact affine fit, zero residual, and a linear part that
        # collapses the whole field onto a speck - what the real failure did.
        assert len(beads_ref) >= 3
        clustered = pd.DataFrame(
            {
                "x": [60.0, 61.0, 60.0],
                "y": [20.0, 20.0, 21.0],
                "fov": [0, 0, 0],
            }
        )
        orig_detect, orig_match = (
            spline._detect_bead_positions,
            spline.ransac_match,
        )
        spline._detect_bead_positions = lambda *a, **k: clustered
        spline.ransac_match = lambda *a, **k: (
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        )
        try:
            with pytest.raises(ValueError, match="implausible"):
                spline._estimate_channel_transform(
                    movie,
                    movie,
                    beads_ref,
                    2000.0,
                    BOX,
                    ref_bounds,
                    mid,
                    max_distance=float(BOX),
                    channel_roi=spline._normalized_region(regions[1]),
                    coarse_shift=(-48.0, 0.0),
                    fov_of_frame=fov_of_frame,
                )
        finally:
            spline._detect_bead_positions = orig_detect
            spline.ransac_match = orig_match

    def test_global_dedupe_would_have_merged_across_fields(self):
        """Guards the premise of the test above: without the FOV labels the
        channel cloud really is smaller, so the assertion is not vacuous."""
        _, _, fov_beads = self._movie()
        x = np.array([b[0] for f in fov_beads for b in f])
        y = np.array([b[1] for f in fov_beads for b in f])
        fov = np.concatenate(
            [np.full(len(f), k) for k, f in enumerate(fov_beads)]
        )
        per_fov = sum(
            len(spline._dedupe_beads(x[fov == k], y[fov == k], BOX)[0])
            for k in range(len(fov_beads))
        )
        pooled, _ = spline._dedupe_beads(x, y, BOX)
        assert len(pooled) < per_fov == len(x)


class TestSplitFovTransform:
    """Region-aware channel-transform estimation."""

    def test_estimate_region_transform_recovers_shift(self):
        dx, dy = 2, -1
        movie, regions, _ = _synthetic_split_fov_movie(dx, dy)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        ref_roi = spline._normalized_region(regions[0])
        chan_roi = spline._normalized_region(regions[1])
        beads_ref = spline._detect_bead_positions(
            movie, 2000.0, BOX, ref_bounds, roi=ref_roi
        )
        # coarse shift = (x0_ref - x0_c, y0_ref - y0_c) = (0 - 48, 0 - 0)
        transform, n_matches = spline._estimate_channel_transform(
            movie,
            movie,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
            channel_roi=chan_roi,
            coarse_shift=(-48.0, 0.0),
        )
        assert n_matches >= 3
        # ref (x, y) -> region-1 (x + 48 + dx, y + dy)
        np.testing.assert_allclose(
            transform.matrix[:2],
            np.array([[1.0, 0.0, 48 + dx], [0.0, 1.0, dy]]),
            atol=0.6,
        )

    @pytest.mark.parametrize("axis", ["y", "x"])
    def test_estimate_region_transform_recovers_flip(self, axis):
        """A mirrored channel (biplane reflected path) must still register:
        the flip-aware coarse matching finds all correspondences and the affine
        encodes the mirror (negative determinant)."""
        movie, regions = _synthetic_split_fov_movie_flipped(axis)

        step_of_frame, _, step_range = spline._step_of_frame(
            movie.shape[0], 20.0, 1, "fov", None
        )
        ref_bounds = spline._reference_frame_segments(
            step_of_frame, step_range
        )
        mid = spline._reference_mid_frame(step_of_frame, step_range)
        ref_roi = spline._normalized_region(regions[0])
        chan_roi = spline._normalized_region(regions[1])
        beads_ref = spline._detect_bead_positions(
            movie, 2000.0, BOX, ref_bounds, roi=ref_roi
        )
        n_ref = len(beads_ref)
        transform, n_matches = spline._estimate_channel_transform(
            movie,
            movie,
            beads_ref,
            2000.0,
            BOX,
            ref_bounds,
            mid,
            max_distance=float(BOX),
            channel_roi=chan_roi,
            coarse_shift=(-48.0, 0.0),
        )
        # all reference beads are matched (a pure translation would find few)
        assert n_matches == n_ref
        # the linear part is a reflection -> negative determinant
        assert np.linalg.det(linear_part(transform)) < 0
        # applying the transform maps ref beads into the channel region, mirrored
        ref_xy = beads_ref[["x", "y"]].to_numpy(float)
        mapped = apply_transform(ref_xy, transform)
        if axis == "y":
            np.testing.assert_allclose(
                mapped[:, 0], ref_xy[:, 0] + 48, atol=1.0
            )
            np.testing.assert_allclose(
                mapped[:, 1], 47 - ref_xy[:, 1], atol=1.0
            )
        else:
            np.testing.assert_allclose(
                mapped[:, 0], 47 - ref_xy[:, 0] + 48, atol=1.0
            )
            np.testing.assert_allclose(mapped[:, 1], ref_xy[:, 1], atol=1.0)


class TestCalibrateSplitFov:
    """Full split-FOV calibration from one movie with two FOV regions."""

    def test_stores_metadata_and_region_transform(self, tmp_path):
        from picasso import io

        dx, dy = 2, -1
        movie, regions, _ = _synthetic_split_fov_movie(dx, dy)
        info = [{"Frames": int(movie.shape[0])}]
        path = str(tmp_path / "splitfov_spline_calib.hdf5")
        calib = spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
            path=path,
        )
        assert calib["model"] == "spline-3d-multichannel"
        assert calib["n_channels"] == 2
        assert calib["split_fov"] is True
        assert calib["reference"] == 0
        assert len(calib["regions"]) == 2
        # region-1 transform is the known translation (absolute coords)
        np.testing.assert_allclose(
            affine_matrix(calib["channel_transforms"][1]),
            [[1.0, 0.0, 48 + dx], [0.0, 1.0, dy]],
            atol=0.6,
        )
        assert calib["coefficients"].shape[-1] == 2
        # split-FOV metadata survives the HDF5 round-trip
        loaded = io.load_spline_calibration(path)
        assert loaded["split_fov"] is True
        assert len(loaded["regions"]) == 2

    def test_reference_index_is_reordered_first(self, tmp_path):
        dx, dy = 2, -1
        movie, regions, _ = _synthetic_split_fov_movie(dx, dy)
        info = [{"Frames": int(movie.shape[0])}]
        # pick region 1 as the reference; it must become channel 0 (identity)
        calib = spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
            reference=1,
        )
        # reference region stored first, transform now maps region-1 -> region-0
        assert calib["regions"][0] == [[0, 48], [48, 96]]
        np.testing.assert_allclose(
            affine_matrix(calib["channel_transforms"][1]),
            [[1.0, 0.0, -(48 + dx)], [0.0, 1.0, -dy]],
            atol=0.7,
        )

    def test_saves_per_channel_and_registration_diagnostics(self, tmp_path):
        movie, regions, _ = _synthetic_split_fov_movie(2, -1)
        info = [{"Frames": int(movie.shape[0])}]
        path = tmp_path / "splitfov_spline_calib.hdf5"
        spline.calibrate_spline_split_fov(
            movie,
            info=info,
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            regions=regions,
            path=str(path),
        )
        base = str(tmp_path / "splitfov_spline_calib")
        # one PSF diagnostic per channel + one registration diagnostic
        assert os.path.exists(base + "_ch0.png")
        assert os.path.exists(base + "_ch1.png")
        assert os.path.exists(base + "_registration.png")
        assert os.path.getsize(base + "_registration.png") > 5000

    def test_unequal_region_sizes_raise(self):
        movie, regions, _ = _synthetic_split_fov_movie()
        regions = [[[0, 0], [48, 48]], [[0, 48], [40, 90]]]  # different size
        info = [{"Frames": int(movie.shape[0])}]
        with pytest.raises(ValueError, match="same size"):
            spline.calibrate_spline_split_fov(
                movie,
                info=info,
                camera_info=CAMERA_INFO,
                box=BOX,
                minimum_ng=2000.0,
                d=20.0,
                regions=regions,
            )


class TestRefineSplitFovTransformsFromSignal:
    """Data-driven (no-bead) re-registration of a split-FOV calibration: pair
    blinking single-molecule signal across channels frame by frame, seeded by
    only the calibration's flip, over a bounded sample of frames."""

    @staticmethod
    def _blinking_movie(true_affine, n_frames=150, seed=0):
        """Two 48x48 regions in a 48x96 frame; each frame has a few emitters in
        the reference region that also appear in the channel region at the
        region-local ``true_affine`` mapping (shared signal, as in biplane)."""
        rng = np.random.RandomState(seed)
        H, W = 48, 96
        ref_rect = [[0, 0], [48, 48]]
        c_rect = [[0, 48], [48, 96]]
        t_true = localize.compose_region_transforms(
            [ref_rect, c_rect], [affine(np.eye(3)), affine(true_affine)]
        )[1]

        def render(frame, x, y, amp, sigma=1.2):
            xi, yi = int(round(x)), int(round(y))
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        frame[yy, xx] += amp * np.exp(
                            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
                        )

        movie = np.zeros((n_frames, H, W), dtype=np.float32)
        for f in range(n_frames):
            for _ in range(rng.randint(3, 6)):
                x = rng.uniform(10, 38)
                y = rng.uniform(10, 38)
                amp = rng.uniform(2500, 4000)
                render(movie[f], x, y, amp)
                cx, cy = apply_transform(np.array([[x, y]]), t_true)[0]
                render(movie[f], cx, cy, amp)
        movie = rng.poisson(np.maximum(movie, 0) + 100).astype(np.uint16)
        return movie, [ref_rect, c_rect]

    def test_recovers_true_affine_from_signal(self):
        # true fine registration: small rotation + subpixel shift
        theta = 0.02
        true_affine = np.array(
            [
                [np.cos(theta), -np.sin(theta), 1.5],
                [np.sin(theta), np.cos(theta), -1.0],
            ]
        )
        movie, regions = self._blinking_movie(true_affine)
        identity = IDENTITY
        # stale calibration: identity fine registration (drifted from truth)
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": [[[0, 0], [48, 48]], [[0, 48], [48, 96]]],
            "channel_registration": [identity, identity],
            "channel_transforms": [
                affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to_dict(),
                affine([[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        updated, reg_info = spline.refine_split_fov_transforms_from_signal(
            movie, calib, regions, minimum_ng=800.0, box=BOX
        )
        # only ~50 frames are sampled (not all 150), so expect fewer pairs
        assert reg_info[0]["n_matches"] >= 40
        assert reg_info[0]["rms"] < 1.0
        # the region-local affine now matches the true fine registration
        np.testing.assert_allclose(
            affine_matrix(updated["channel_registration"][1]),
            true_affine,
            atol=0.15,
        )

    def test_recovers_true_affine_with_mirror(self):
        # channel is an x-mirror of the reference (as a biplane relay flips it)
        # plus a small sub-pixel shift; the calibration stores only the mirror,
        # which is the coarse seed the re-registration is allowed to trust - the
        # fine rotation/scale/shift must be recovered fresh from the signal
        w = 48
        true_affine = np.array([[-1.0, 0.0, w + 0.6], [0.0, 1.0, -0.4]])
        movie, regions = self._blinking_movie(true_affine)
        identity = IDENTITY
        mirror = [[-1.0, 0.0, float(w)], [0.0, 1.0, 0.0]]
        regs = [[[0, 0], [48, 48]], [[0, 48], [48, 96]]]
        mirror_transform = localize.compose_region_transforms(
            regs, [identity, affine(mirror)]
        )[1].to_dict()
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": regs,
            "channel_registration": [identity, affine(mirror).to_dict()],
            "channel_transforms": [
                IDENTITY,
                mirror_transform,
            ],
        }
        updated, reg_info = spline.refine_split_fov_transforms_from_signal(
            movie, calib, regions, minimum_ng=800.0, box=BOX
        )
        assert reg_info[0]["n_matches"] >= 40
        assert reg_info[0]["rms"] < 1.0
        # the fitted affine stays mirrored (negative determinant) and recovers
        # the true fine registration on top of the flip
        linear = linear_part(updated["channel_registration"][1])
        assert np.linalg.det(linear) < 0
        np.testing.assert_allclose(
            affine_matrix(updated["channel_registration"][1]),
            true_affine,
            atol=0.2,
        )

    def test_raises_without_shared_signal(self):
        # channel region has no correlated signal -> no pairs -> raises
        rng = np.random.RandomState(1)
        movie = rng.poisson(np.full((60, 48, 96), 100.0)).astype(np.uint16)
        # a few emitters only in the reference region
        for f in range(60):
            for _ in range(4):
                x, y = rng.uniform(10, 38), rng.uniform(10, 38)
                xi, yi = int(round(x)), int(round(y))
                movie[f, yi - 1 : yi + 2, xi - 1 : xi + 2] += 3000
        identity = IDENTITY
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": [[[0, 0], [48, 48]], [[0, 48], [48, 96]]],
            "channel_registration": [identity, identity],
            "channel_transforms": [
                affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to_dict(),
                affine([[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        with pytest.raises(ValueError):
            spline.refine_split_fov_transforms_from_signal(
                movie,
                calib,
                [[[0, 0], [48, 48]], [[0, 48], [48, 96]]],
                minimum_ng=800.0,
                box=BOX,
            )


class TestRefineMultichannelTransformsFromSignal:
    """Data-driven re-registration of a separate-movie multichannel calibration:
    pair blinking signal across the (frame-synchronized) channel movies frame by
    frame, seeded from the calibration's existing (slightly stale) transforms.
    """

    @staticmethod
    def _blinking_movies(true_transform, n_frames=150, seed=0):
        """Two 64x64 frame-synchronized movies; each frame has a few emitters in
        the reference movie that also appear in the channel movie at
        ``true_transform`` (shared signal, as in multichannel acquisition)."""
        rng = np.random.RandomState(seed)
        H, W = 64, 64

        def render(frame, x, y, amp, sigma=1.2):
            xi, yi = int(round(x)), int(round(y))
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        frame[yy, xx] += amp * np.exp(
                            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)
                        )

        ref_movie = np.zeros((n_frames, H, W), dtype=np.float32)
        c_movie = np.zeros((n_frames, H, W), dtype=np.float32)
        for f in range(n_frames):
            for _ in range(rng.randint(5, 8)):
                x = rng.uniform(16, 48)
                y = rng.uniform(16, 48)
                amp = rng.uniform(2500, 4000)
                render(ref_movie[f], x, y, amp)
                cx, cy = apply_transform(np.array([[x, y]]), true_transform)[0]
                render(c_movie[f], cx, cy, amp)
        ref_movie = rng.poisson(np.maximum(ref_movie, 0) + 100).astype(
            np.uint16
        )
        c_movie = rng.poisson(np.maximum(c_movie, 0) + 100).astype(np.uint16)
        return [ref_movie, c_movie]

    def test_recovers_true_transform_from_signal(self):
        # true fine registration: small rotation + sub-pixel shift
        theta = 0.02
        true_transform = np.array(
            [
                [np.cos(theta), -np.sin(theta), 2.0],
                [np.sin(theta), np.cos(theta), -1.5],
            ]
        )
        movies = self._blinking_movies(true_transform)
        # stale calibration: the stored transform has drifted from the truth
        calib = {
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "channel_transforms": [
                IDENTITY,
                affine([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        seed = affine_matrix(calib["channel_transforms"][1])
        updated, reg_info = spline.refine_multichannel_transforms_from_signal(
            movies, calib, minimum_ng=800.0, box=BOX
        )
        # only ~50 frames are sampled (not all 150), so expect fewer pairs
        assert reg_info[0]["n_matches"] >= 40
        assert reg_info[0]["rms"] < 1.0
        # the refined transform recovers the truth (a residual sub-pixel bias
        # remains because identify uses a centroid, not a Gaussian fit) ...
        fitted = affine_matrix(updated["channel_transforms"][1])
        np.testing.assert_allclose(fitted, true_transform, atol=0.35)
        # ... and is much closer to the truth than the stale stored seed
        assert (
            np.abs(fitted - true_transform).max()
            < 0.5 * np.abs(seed - true_transform).max()
        )

    def test_rejects_split_fov_calibration(self):
        movies = self._blinking_movies(
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), n_frames=10
        )
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "channel_transforms": [
                IDENTITY,
                affine([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        with pytest.raises(ValueError):
            spline.refine_multichannel_transforms_from_signal(
                movies, calib, minimum_ng=800.0, box=BOX
            )

    def test_raises_without_shared_signal(self):
        # channel movie has no correlated signal -> no pairs -> raises
        rng = np.random.RandomState(1)
        ref_movie = np.zeros((60, 64, 64), dtype=np.float32)
        for f in range(60):
            for _ in range(4):
                x, y = rng.uniform(12, 52), rng.uniform(12, 52)
                xi, yi = int(round(x)), int(round(y))
                ref_movie[f, yi - 1 : yi + 2, xi - 1 : xi + 2] += 3000
        ref_movie = rng.poisson(np.maximum(ref_movie, 0) + 100).astype(
            np.uint16
        )
        c_movie = rng.poisson(np.full((60, 64, 64), 100.0)).astype(np.uint16)
        calib = {
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "channel_transforms": [
                IDENTITY,
                affine([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        with pytest.raises(ValueError):
            spline.refine_multichannel_transforms_from_signal(
                [ref_movie, c_movie], calib, minimum_ng=800.0, box=BOX
            )


# ---------------------------------------------------------------------------
# Session C: GUI + CLI wiring (no GPU required)
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_fit_method_map(self):
        from picasso import __main__ as cli

        # bare names are the CPU fit, "-gpu" the Gpufit one - the convention
        # the Gaussian methods already follow
        assert cli._FIT_METHOD_MAP["spline"] == "spline"
        assert cli._FIT_METHOD_MAP["spline-mle"] == "spline-mle"
        assert cli._FIT_METHOD_MAP["spline-gpu"] == "spline-gpu"
        assert cli._FIT_METHOD_MAP["spline-mle-gpu"] == "spline-mle-gpu"

    def test_only_the_gpu_codes_require_a_gpu(self):
        """The CPU codes must not trip the GPU precheck, and every "-gpu" one
        must. Keyed on the resolved fit code rather than a hand-written list,
        which went stale as the remaining methods gained a GPU backend."""
        import inspect
        from picasso import __main__ as cli

        src = inspect.getsource(cli._localize)
        assert '_FIT_METHOD_MAP[args.fit_method].endswith("-gpu")' in src
        # aliases that hide the suffix still resolve to a GPU code ...
        assert cli._FIT_METHOD_MAP["lq-gpu-3d"].endswith("-gpu")
        # ... and no CPU alias does
        for alias in ("spline", "spline-mle", "lq", "mle", "lq-3d", "avg"):
            assert not cli._FIT_METHOD_MAP[alias].endswith("-gpu")

    def test_spline_calibrate_handler_exists(self):
        from picasso import __main__ as cli

        assert callable(cli._spline_calibrate)

    def test_spline_calibrate_reports_the_filtered_beads(self, tmp_path):
        """The command line has no inspector, so its summary must say how many
        beads were actually used and point at the gallery that was written."""
        import inspect
        from picasso import __main__ as cli

        src = inspect.getsource(cli._spline_calibrate)
        assert "spline.n_beads_used(calibration)" in src
        assert "_beads.png" in src

        # and the gallery really is written next to the calibration
        movie, _, _ = _synthetic_bead_movie()
        path = str(tmp_path / "cli_spline_calib.hdf5")
        spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=2000.0,
            d=20.0,
            path=path,
        )
        assert os.path.exists(str(tmp_path / "cli_spline_calib_beads.png"))

    def test_backend_accepts_both_spline_codes(self):
        # both spline codes must be recognized model ids by the backend
        # (guards the fit / localize dispatch strings)
        import inspect

        src = inspect.getsource(localize.fit)
        for code in ("spline", "spline-mle", "spline-gpu", "spline-mle-gpu"):
            assert f'"{code}"' in src


class TestGuiWiring:
    def test_spline_model_is_offered_without_gpufit(self):
        """The spline model must be in the menu on every machine: it now has a
        CPU backend, so it is no longer deleted when Gpufit is missing."""
        from picasso.gui import localize as glocalize

        assert "Experimental PSF (cubic spline)" in glocalize.FIT_MODELS

    def test_fit_code_resolves_spline(self):
        from picasso.gui import localize as glocalize

        # bare (CPU) codes; FitWorker appends "-gpu" when the GPUfit
        # checkbox is ticked, as for the Gaussian models
        assert (
            glocalize._fit_code(
                "Experimental PSF (cubic spline)", "Least squares"
            )
            == "spline"
        )
        assert (
            glocalize._fit_code("Experimental PSF (cubic spline)", "MLE")
            == "spline-mle"
        )

    def test_spline_honors_the_convergence_controls(self):
        """The CPU spline iterates under both estimators, so both must show
        the convergence page - and with the spline's own schedule, not the
        Gaussian MLE's."""
        from picasso.gui import localize as glocalize
        from picasso.fitting import splinefit

        assert "spline" in glocalize._CONVERGENCE_CODES
        assert "spline-mle" in glocalize._CONVERGENCE_CODES
        assert glocalize._CONVERGENCE_DEFAULTS["spline"] == (
            splinefit.TOLERANCE_MULTI_START,
            splinefit.MAX_ITERATIONS_MULTI_START,
        )

    @staticmethod
    def _fit_worker(method, use_gpufit, calib):
        """Build a FitWorker. Callers must hold the ``qt_offscreen`` fixture -
        a QObject cannot be built before the QApplication exists."""
        import pandas as pd
        from picasso.gui import localize as glocalize

        return glocalize.FitWorker(
            None,
            [],
            {},
            pd.DataFrame({"x": [], "y": [], "frame": []}),
            BOX,
            method,
            0.001,
            100,
            False,
            False,
            use_gpufit,
            spline_calibration=calib,
        )

    def test_fit_worker_preserves_spline_method_and_calibration(
        self, qt_offscreen
    ):
        calib = {"model": "spline-3d"}
        worker = self._fit_worker("spline-mle-gpu", True, calib)
        # the "-gpu" suffix must not be appended to an already-gpu spline code
        assert worker.method == "spline-mle-gpu"
        assert worker.spline_calibration is calib

    @pytest.mark.parametrize(
        "method, use_gpufit, expected",
        [
            ("spline", False, "spline"),
            ("spline-mle", False, "spline-mle"),
            ("spline", True, "spline-gpu"),
            ("spline-mle", True, "spline-mle-gpu"),
        ],
    )
    def test_fit_worker_routes_spline_by_gpufit_checkbox(
        self, qt_offscreen, method, use_gpufit, expected
    ):
        """The GPUfit checkbox selects the device for the spline model exactly
        as it does for the Gaussian ones."""
        worker = self._fit_worker(method, use_gpufit, {"model": "spline-3d"})
        assert worker.method == expected

    def test_bead_inspection_dialog_shows_the_filtering(self, qt_offscreen):
        """The bead inspector must render the calibration's per-bead record,
        so the user can check the outlier filtering instead of trusting it."""
        from picasso.gui import localize as glocalize

        movie, _, _ = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie, CAMERA_INFO, box=BOX, minimum_ng=2000.0, d=20.0
        )
        built["bead_quality"]["keep"][-1] = False
        data = spline.bead_inspection_data(
            built, {"z_center": float(built["z_center"]), "pixelsize": 130.0}
        )
        n_beads = len(data["keep"])

        dialog = glocalize.BeadInspectionDialog([data, data])
        assert f"of {n_beads} beads" in dialog.summary.text()
        assert "1 rejected" in dialog.summary.text()
        # one entry per channel, and the gallery is drawn on the canvas
        assert dialog.channel.count() == 2
        assert len(dialog.figure.axes) == 3 * (n_beads + 1) + 1
        dialog.only_rejected.setChecked(True)
        assert len(dialog.figure.axes) == 3 * 2 + 1
        dialog.close()

    def test_bead_inspection_dialog_scrolls_the_gallery(self, qt_offscreen):
        """The gallery is taller than any window, so it must scroll - and a
        matplotlib canvas swallows wheel events unless they are forwarded."""
        from PyQt6 import QtCore, QtGui, QtWidgets
        from picasso.gui import localize as glocalize

        app = qt_offscreen

        movie, _, _ = _synthetic_bead_movie()
        built = spline.build_psf_template(
            movie, CAMERA_INFO, box=BOX, minimum_ng=2000.0, d=20.0
        )
        data = spline.bead_inspection_data(
            built, {"z_center": float(built["z_center"]), "pixelsize": 130.0}
        )

        dialog = glocalize.BeadInspectionDialog([data])
        dialog.show()
        app.processEvents()

        viewport = dialog.scroll.viewport()
        # "Fit width" leaves nothing to scroll horizontally ...
        assert dialog.zoom.currentData() is None
        assert dialog.canvas.width() <= viewport.width()
        assert dialog.scroll.horizontalScrollBar().maximum() == 0
        # ... while the gallery itself runs past the bottom of the window
        vbar = dialog.scroll.verticalScrollBar()
        assert dialog.canvas.height() > viewport.height()
        assert vbar.maximum() > 0

        wheel = QtGui.QWheelEvent(
            QtCore.QPointF(50, 50),
            dialog.canvas.mapToGlobal(QtCore.QPointF(50, 50)),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, -240),  # two notches down
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QtWidgets.QApplication.sendEvent(dialog.canvas, wheel)
        app.processEvents()
        assert vbar.value() > 0

        # a fixed zoom sizes the gallery to the figure itself, so a larger
        # percentage always draws a larger gallery
        width_in = dialog._figure_inches[0]
        dialog.zoom.setCurrentIndex(dialog.zoom.findText("150 %"))
        app.processEvents()
        assert dialog.canvas.width() == pytest.approx(150 * width_in, abs=2)
        dialog.zoom.setCurrentIndex(dialog.zoom.findText("300 %"))
        app.processEvents()
        assert dialog.canvas.width() == pytest.approx(300 * width_in, abs=2)
        # ... and once it is wider than the window, it scrolls both ways.
        # Which percentage that is depends on the window (the dialog is
        # laid out wider under the offscreen Qt platform), so pick the
        # first one that overflows.
        percent = next(
            (
                dialog.zoom.itemData(i)
                for i in range(dialog.zoom.count())
                if dialog.zoom.itemData(i) is not None
                and dialog.zoom.itemData(i) * width_in > viewport.width()
            ),
            None,
        )
        assert percent is not None, "the window is wider than any zoom"
        dialog.zoom.setCurrentIndex(dialog.zoom.findText(f"{percent} %"))
        app.processEvents()
        assert dialog.canvas.width() > viewport.width()
        assert dialog.scroll.horizontalScrollBar().maximum() > 0
        dialog.close()

    def test_spline_calibration_worker_collects_the_bead_record(self):
        """The worker must hand the per-bead record to the window; without it
        the inspector has nothing to show after a calibration."""
        import inspect
        from picasso.gui import localize as glocalize

        src = inspect.getsource(glocalize.SplineCalibrationWorker.run)
        assert src.count("return_diagnostics=True") == 3  # every entry point
        assert "self.bead_diagnostics" in src
        finished = inspect.getsource(
            glocalize.Window.on_spline_calibration_finished
        )
        assert "bead_diagnostics" in finished
        assert "inspect_beads_action" in finished


# ---------------------------------------------------------------------------
# Wavelet bead detection
# ---------------------------------------------------------------------------


class TestWaveletDetection:
    """Every calibration entry point detects by wavelet segmentation when
    given ``wavelet`` (the method selected for the movie), and then needs no
    minimum net gradient."""

    WAVELET = wavelet.WaveletParameters()

    def test_calibrate_spline(self):
        movie, bead_xy, _ = _synthetic_bead_movie()
        calib = spline.calibrate_spline(
            movie,
            info=[{"Frames": int(movie.shape[0])}],
            camera_info=CAMERA_INFO,
            box=BOX,
            minimum_ng=None,
            d=20.0,
            wavelet=self.WAVELET,
        )
        assert calib["n_beads"] == len(bead_xy)

    def test_detect_bead_positions(self):
        movie, bead_xy, focus = _synthetic_bead_movie()
        beads = spline._detect_bead_positions(
            movie, None, BOX, (focus - 2, focus + 2), wavelet=self.WAVELET
        )
        assert sorted(zip(beads["x"], beads["y"])) == sorted(bead_xy)

    def test_calibrate_spline_multichannel(self):
        movie_ref, bead_xy, _ = _synthetic_bead_movie()
        movie_c = np.roll(movie_ref, shift=(2, -1), axis=(1, 2))
        info = [{"Frames": int(movie_ref.shape[0])}]
        calib = spline.calibrate_spline_multichannel(
            [movie_ref, movie_c],
            infos=[info, info],
            camera_infos=[CAMERA_INFO, CAMERA_INFO],
            box=BOX,
            minimum_ng=None,
            d=20.0,
            wavelet=self.WAVELET,
        )
        probe = np.array([[20.0, 20.0]])
        shifted = transforms.from_dict(calib["channel_transforms"][1])
        np.testing.assert_allclose(
            shifted.apply(probe), probe + [-1.0, 2.0], atol=0.6
        )

    def test_calibrate_spline_split_fov_forwards(self, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return {}

        monkeypatch.setattr(spline, "calibrate_spline_multichannel", spy)
        spline.calibrate_spline_split_fov(
            np.zeros((3, 16, 32)),
            [{}],
            CAMERA_INFO,
            BOX,
            None,
            20.0,
            regions=[[[0, 0], [16, 16]], [[0, 16], [16, 32]]],
            wavelet=self.WAVELET,
        )
        assert seen["wavelet"] is self.WAVELET

    def test_refine_multichannel_transforms_from_signal(self):
        truth = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.5]])
        movies = TestRefineMultichannelTransformsFromSignal._blinking_movies(
            truth
        )
        calib = {
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "channel_transforms": [
                IDENTITY,
                affine([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        updated, reg_info = spline.refine_multichannel_transforms_from_signal(
            movies, calib, minimum_ng=None, box=BOX, wavelet=self.WAVELET
        )
        assert reg_info[0]["n_matches"] >= 40
        np.testing.assert_allclose(
            affine_matrix(updated["channel_transforms"][1]), truth, atol=0.35
        )

    def test_refine_split_fov_transforms_from_signal(self):
        truth = np.array([[1.0, 0.0, 1.5], [0.0, 1.0, -1.0]])
        movie, regions = (
            TestRefineSplitFovTransformsFromSignal._blinking_movie(truth)
        )
        calib = {
            "split_fov": True,
            "n_channels": 2,
            "box": BOX,
            "n_data": [BOX, BOX, 1],
            "regions": regions,
            "channel_registration": [IDENTITY, IDENTITY],
            "channel_transforms": [
                affine([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to_dict(),
                affine([[1.0, 0.0, 48.0], [0.0, 1.0, 0.0]]).to_dict(),
            ],
        }
        updated, reg_info = spline.refine_split_fov_transforms_from_signal(
            movie,
            calib,
            regions,
            minimum_ng=None,
            box=BOX,
            wavelet=self.WAVELET,
        )
        assert reg_info[0]["n_matches"] >= 40
        # the detections are integer box centers, as for the net gradient,
        # which leaves a sub-pixel residual (the tolerance of the
        # multichannel refinement above)
        np.testing.assert_allclose(
            affine_matrix(updated["channel_registration"][1]),
            truth,
            atol=0.35,
        )
