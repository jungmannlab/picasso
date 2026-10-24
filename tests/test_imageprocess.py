"""Test ``picasso.imageprocess`` — FFT cross-correlation and shift detection.

Most fixtures (``locs``, ``info``) live in ``tests/conftest.py``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot/picasso imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from picasso import imageprocess  # noqa: E402

from tests.test_postprocess_progress import (  # noqa: E402
    RecordingProgress,
    assert_phases,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic Gaussian-blob images for cross-correlation tests
# ---------------------------------------------------------------------------


def _gaussian_image(
    size: int,
    cx: float,
    cy: float,
    sigma: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Build a single 2D Gaussian blob image of shape ``(size, size)``."""
    y, x = np.mgrid[0:size, 0:size]
    return (
        amplitude * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# xcorr
# ---------------------------------------------------------------------------


class TestXcorr:
    def test_autocorrelation_peak_at_center(self):
        img = _gaussian_image(64, 32, 32, sigma=3.0)
        cc = imageprocess.xcorr(img, img)
        peak_y, peak_x = np.unravel_index(cc.argmax(), cc.shape)
        # fftshift puts zero-shift peak at the geometric center
        assert peak_y == img.shape[0] // 2
        assert peak_x == img.shape[1] // 2

    @pytest.mark.parametrize("dy,dx", [(0, 0), (3, 0), (0, -4), (5, -5)])
    def test_translation_peak_offset(self, dy, dx):
        size = 64
        img = _gaussian_image(size, 32, 32, sigma=3.0)
        rolled = np.roll(img, (dy, dx), axis=(0, 1))
        cc = imageprocess.xcorr(img, rolled)
        peak_y, peak_x = np.unravel_index(cc.argmax(), cc.shape)
        # peak position relative to fftshift center == (-dy, -dx) modulo size
        assert (peak_y - size // 2) % size == (-dy) % size
        assert (peak_x - size // 2) % size == (-dx) % size

    def test_zero_input_no_nan(self):
        img = np.zeros((32, 32), dtype=np.float64)
        cc = imageprocess.xcorr(img, img)
        assert np.all(np.isfinite(cc))
        assert np.allclose(cc, 0.0)


# ---------------------------------------------------------------------------
# get_image_shift
# ---------------------------------------------------------------------------


class TestGetImageShift:
    def test_zero_input_short_circuits(self):
        # See imageprocess.py:83-84 — either-zero input must return (0, 0).
        img = _gaussian_image(32, 16, 16)
        zero = np.zeros_like(img)
        assert imageprocess.get_image_shift(zero, img, box=5) == (0, 0)
        assert imageprocess.get_image_shift(img, zero, box=5) == (0, 0)

    def test_no_shift_returns_zero(self):
        img = _gaussian_image(64, 32, 32, sigma=3.0)
        yc, xc = imageprocess.get_image_shift(img, img, box=7)
        assert abs(yc) < 0.05
        assert abs(xc) < 0.05

    @pytest.mark.parametrize(
        "dy,dx",
        [
            (3.0, 0.0),
            (0.0, -4.0),
            (2.5, 1.0),
            (-1.5, 2.5),
        ],
    )
    def test_recovers_known_shift(self, dy, dx):
        # Place a Gaussian at center vs shifted by (dy, dx); recovered
        # shift should be close to that displacement to <0.5 px.
        size = 64
        cx = size / 2
        cy = size / 2
        imgA = _gaussian_image(size, cx, cy, sigma=3.0)
        imgB = _gaussian_image(size, cx + dx, cy + dy, sigma=3.0)
        yc, xc = imageprocess.get_image_shift(imgA, imgB, box=7)
        # get_image_shift returns (-yc, -xc), i.e. the shift A->B
        assert abs(yc - dy) < 0.5
        assert abs(xc - dx) < 0.5

    def test_with_roi(self):
        # roi crops the cross-correlation before peak detection
        size = 64
        imgA = _gaussian_image(size, 32, 32, sigma=3.0)
        imgB = _gaussian_image(size, 33, 32, sigma=3.0)
        yc, xc = imageprocess.get_image_shift(imgA, imgB, box=7, roi=20)
        assert abs(yc - 0.0) < 0.5
        assert abs(xc - 1.0) < 0.5


# ---------------------------------------------------------------------------
# rcc
# ---------------------------------------------------------------------------


class TestRCC:
    def test_recovers_known_per_segment_shifts(self):
        # 3 segments with known absolute offsets relative to seg 0.
        size = 64
        cx = size / 2
        cy = size / 2
        offsets = [(0.0, 0.0), (2.0, -1.0), (-1.0, 3.0)]  # (dy, dx)
        segments = [
            _gaussian_image(size, cx + dx, cy + dy, sigma=3.0)
            for dy, dx in offsets
        ]
        shift_y, shift_x = imageprocess.rcc(segments, max_shift=20)

        # rcc returns (shift_y, shift_x) per segment relative to segment 0
        for i, (dy, dx) in enumerate(offsets):
            assert (
                abs(shift_y[i] - dy) < 0.5
            ), f"segment {i}: dy={shift_y[i]:.3f} vs {dy}"
            assert (
                abs(shift_x[i] - dx) < 0.5
            ), f"segment {i}: dx={shift_x[i]:.3f} vs {dx}"

    def test_progress_counts_pairs(self):
        size = 32
        segments = [_gaussian_image(size, 16, 16, sigma=2.0)] * 3
        tracker = RecordingProgress()
        imageprocess.rcc(segments, max_shift=10, progress=tracker)
        assert_phases(tracker, ["Correlating image pairs"])
        # 3 segments -> 3 pairs, counted as they finish
        assert tracker.phases[0].maximum == 3
        assert tracker.phases[0].values == [0, 1, 2, 3, 3]


# ---------------------------------------------------------------------------
# find_fiducials — uses bundled locs/info
# ---------------------------------------------------------------------------


class TestFindFiducials:
    def test_returns_picks_and_box(self, locs, info):
        picks, box = imageprocess.find_fiducials(locs, info)
        assert isinstance(picks, list)
        # the box is an odd number of rendered (100 nm) pixels, about
        # 900 nm, returned in nm
        assert box == pytest.approx(imageprocess.FIDUCIAL_BOX_SIZE)
        assert round(box / imageprocess.FIDUCIAL_RENDER_PIXEL_SIZE) % 2 == 1

    def test_picks_within_image_bounds(self, locs, info):
        from picasso import lib

        picks, _ = imageprocess.find_fiducials(locs, info)
        width, height = lib.field_of_view(info)
        for x, y in picks:
            assert 0 <= x <= width
            assert 0 <= y <= height

    def test_recovers_planted_fiducial_positions(self):
        """find_fiducials must return picks *at* the planted fiducials.

        Regression test for the coordinate-scaling bug where the image
        was rendered at ``disp_px_size=100`` while the maxima (image
        pixel indices) were used as localization coordinates without
        mapping them back by the rendered pixel size, so every detected
        pick was offset and the markers were missed. The shape-only
        tests do not catch this because the offset picks are still
        in-bounds numbers.

        Pixelsize is deliberately 130 (!= the rendered 100 nm pixel) so
        that the field of view is not a whole number of rendered pixels.
        """
        fov = 64  # camera pixels
        pixelsize = 130.0
        n_frames = 200
        centers = [
            (12.0 * pixelsize, 12.0 * pixelsize),
            (12.0 * pixelsize, 48.0 * pixelsize),
            (48.0 * pixelsize, 12.0 * pixelsize),
            (48.0 * pixelsize, 48.0 * pixelsize),
        ]
        info = [
            {
                "Width": fov,
                "Height": fov,
                "Frames": n_frames,
                "Pixelsize": pixelsize,
            }
        ]

        rng = np.random.default_rng(0)
        # one dense, persistent localization per frame per fiducial
        n_per = n_frames * len(centers)
        x = np.repeat([c[0] for c in centers], n_frames) + rng.normal(
            0, 6.5, n_per
        )
        y = np.repeat([c[1] for c in centers], n_frames) + rng.normal(
            0, 6.5, n_per
        )
        locs = pd.DataFrame(
            {
                "frame": np.tile(np.arange(n_frames), len(centers)).astype(
                    np.int32
                ),
                "x": x,
                "y": y,
                "photons": np.full(n_per, 1000.0),
                "sx": np.full(n_per, pixelsize),
                "sy": np.full(n_per, pixelsize),
                "bg": np.full(n_per, 10.0),
                "lpx": np.full(n_per, 1.3),
                "lpy": np.full(n_per, 1.3),
            }
        )

        picks, box = imageprocess.find_fiducials(locs, info)

        # every planted fiducial must be matched by a returned pick, and
        # every returned pick must be near a planted fiducial (no spurious
        # offset picks) — within half a box of the true position
        tol = box / 2
        for cx, cy in centers:
            assert any(
                abs(px - cx) <= tol and abs(py - cy) <= tol for px, py in picks
            ), f"planted fiducial ({cx}, {cy}) not recovered; got {picks}"
        for px, py in picks:
            assert any(
                abs(px - cx) <= tol and abs(py - cy) <= tol
                for cx, cy in centers
            ), f"spurious/offset pick ({px}, {py}); centers={centers}"


# ---------------------------------------------------------------------------
# radial_sum
# ---------------------------------------------------------------------------


class TestRadialSum:
    def test_delta_at_center(self):
        size = 9
        img = np.zeros((size, size), dtype=np.float64)
        img[size // 2, size // 2] = 1.0
        counts = imageprocess.radial_sum(img)
        assert counts[0] == 1.0
        assert np.all(counts[1:] == 0.0)

    def test_uniform_image_monotonic_then_falls(self):
        # On a uniform image, the radial sum increases until the inscribed
        # circle, then drops as the square corners are excluded.
        size = 11
        img = np.ones((size, size), dtype=np.float64)
        counts = imageprocess.radial_sum(img)
        assert len(counts) == size // 2 + 1
        # counts[0] is the single center pixel
        assert counts[0] == 1.0

    def test_total_equals_sum_within_disk(self):
        # Sum of radial-sum values must equal the sum of pixels enclosed
        # by the largest tested radius (center + 1).
        size = 11
        img = np.arange(size * size, dtype=np.float64).reshape(size, size)
        counts = imageprocess.radial_sum(img)
        # Reproduce the disk mask radial_sum walks over.
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        max_radius = center + 1
        mask = dist_sq < max_radius**2
        assert np.isclose(counts.sum(), img[mask].sum())

    @pytest.mark.parametrize(
        "shape",
        [(8, 8), (10, 12), (7,)],  # even-square / non-square / non-2D
    )
    def test_invalid_shape_raises(self, shape):
        img = np.zeros(shape, dtype=np.float64)
        with pytest.raises(AssertionError):
            imageprocess.radial_sum(img)
