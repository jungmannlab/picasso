"""Test ``picasso.wavelet`` — the wavelet spot identification after
Izeddin et al. (2012): the à trous transform, the noise estimates, the
watershed segmentation and the box centers.

The integration into ``picasso.localize.identify`` and the rest of Picasso
is tested in ``test_localize.py`` (``TestWaveletIdentify``).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from picasso import wavelet

G1 = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16
G2 = np.zeros(9)
G2[::2] = G1


def _spots_frame(
    shape: tuple[int, int],
    centers: list[tuple[float, float]],
    amplitude: float = 500.0,
    background: float = 1000.0,
    sigma: float = 1.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Gaussian spots on a flat background, Poisson noise unless ``seed``
    is None."""
    yy, xx = np.indices(shape, dtype=np.float64)
    image = np.full(shape, background, dtype=np.float64)
    for cy, cx in centers:
        image += amplitude * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    if seed is not None:
        image = np.random.default_rng(seed).poisson(image)
    return image.astype(np.float32)


def _reference_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first two wavelet planes with scipy, as the paper writes them:
    V1 = g1 * V0, V2 = g2 * V1 (separable), W1 = V0 - V1, W2 = V1 - V2."""
    image = image.astype(np.float64)

    def smooth(im, taps):
        rows = ndimage.convolve1d(im, taps, axis=1, mode="mirror")
        return ndimage.convolve1d(rows, taps, axis=0, mode="mirror")

    v1 = smooth(image, G1)
    v2 = smooth(v1, G2)
    return image - v1, v1 - v2


class TestKernel:
    def test_b3_kernel_is_the_papers(self):
        # H0 = 3/8, H1 = 1/4, H2 = 1/16
        np.testing.assert_allclose(
            wavelet.B3_KERNEL, [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]
        )
        assert wavelet.B3_KERNEL.sum() == pytest.approx(1.0)

    def test_w1_noise_gain(self):
        # sqrt(sum((delta - g1 g1^T)^2)), Starck & Murtagh's 0.889
        assert wavelet.W1_NOISE_GAIN == pytest.approx(0.8908, abs=1e-4)


class TestWaveletPlanes:
    def test_matches_scipy_mirror(self):
        image = _spots_frame((40, 50), [(10.3, 20.7), (30.0, 35.2)])
        w1, w2 = wavelet.wavelet_planes(image)
        ref_w1, ref_w2 = _reference_planes(image)
        np.testing.assert_allclose(w1, ref_w1, atol=5e-3)
        np.testing.assert_allclose(w2, ref_w2, atol=5e-3)
        assert w1.dtype == np.float32 and w2.dtype == np.float32

    def test_equals_single_13x13_kernel(self):
        # W2 = (K1 - K1 * K2) * V0: one 13 x 13 kernel, 6 px reach
        k1 = np.outer(G1, G1)
        k2 = np.outer(G2, G2)
        kernel = np.pad(k1, 4) - ndimage.convolve(
            np.pad(k1, 4), k2, mode="constant"
        )
        assert kernel.shape == (2 * wavelet.WAVELET_RADIUS + 1,) * 2
        assert kernel.sum() == pytest.approx(0.0, abs=1e-12)
        image = _spots_frame((48, 48), [(20.0, 25.0)])
        direct = ndimage.convolve(
            image.astype(np.float64), kernel, mode="mirror"
        )
        np.testing.assert_allclose(
            wavelet.wavelet_planes(image)[1], direct, atol=5e-3
        )

    def test_constant_image_has_zero_detail(self):
        image = np.full((30, 30), 1234.0, dtype=np.float32)
        w1, w2 = wavelet.wavelet_planes(image)
        np.testing.assert_allclose(w1, 0, atol=1e-3)
        np.testing.assert_allclose(w2, 0, atol=1e-3)

    def test_noise_gains(self):
        noise = np.random.default_rng(1).normal(0, 1, (512, 512))
        w1, w2 = wavelet.wavelet_planes(noise)
        assert w1.std() == pytest.approx(0.891, rel=0.02)
        assert w2.std() == pytest.approx(0.201, rel=0.03)

    @pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 4), (5, 13)])
    def test_images_smaller_than_the_kernel(self, shape):
        image = np.random.default_rng(2).normal(size=shape)
        w1, w2 = wavelet.wavelet_planes(image)
        ref_w1, ref_w2 = _reference_planes(image)
        np.testing.assert_allclose(w1, ref_w1, atol=1e-5)
        np.testing.assert_allclose(w2, ref_w2, atol=1e-5)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            wavelet.wavelet_planes(np.zeros((2, 3, 4)))


class TestNoiseSigma:
    def test_both_estimates_on_pure_noise(self):
        noise = np.random.default_rng(3).normal(100, 7, (256, 256))
        assert wavelet.noise_sigma(noise) == pytest.approx(7, rel=0.02)
        assert wavelet.noise_sigma(
            noise, noise=wavelet.NOISE_W1_MAD
        ) == pytest.approx(7, rel=0.05)

    def test_mad_is_robust_to_dense_spots(self):
        rng = np.random.default_rng(4)
        centers = rng.uniform(5, 123, (60, 2))
        image = _spots_frame((128, 128), centers, amplitude=2000, seed=5)
        # the Poisson noise of the background is sqrt(1000) ~ 31.6; the
        # spots inflate the standard deviation of the frame several times
        std = wavelet.noise_sigma(image)
        mad = wavelet.noise_sigma(image, noise=wavelet.NOISE_W1_MAD)
        assert std > 4 * 31.6
        assert mad == pytest.approx(31.6, rel=0.25)

    def test_unknown_estimate(self):
        with pytest.raises(ValueError):
            wavelet.noise_sigma(np.zeros((5, 5)), noise="nope")


class TestDetectRegions:
    def test_centroids_are_sub_pixel_accurate(self):
        centers = [(20.3, 30.6), (60.8, 15.1), (45.5, 70.25)]
        image = _spots_frame((90, 90), centers, amplitude=3000, seed=None)
        image += np.random.default_rng(6).normal(0, 3, image.shape)
        y, x, area = wavelet.detect_regions(image)
        assert len(y) == 3
        found = sorted(zip(y, x))
        for (fy, fx), (cy, cx) in zip(found, sorted(centers)):
            assert abs(fy - cy) < 0.1 and abs(fx - cx) < 0.1
        assert np.all(area >= wavelet.WaveletParameters().min_area)

    def test_watershed_splits_close_spots(self):
        # 4 px apart: one connected region above the threshold, two basins
        centers = [(30.0, 28.0), (30.0, 32.0)]
        image = _spots_frame((60, 60), centers, amplitude=3000, seed=None)
        image += np.random.default_rng(7).normal(0, 3, image.shape)
        params = wavelet.WaveletParameters()
        _, w2 = wavelet.wavelet_planes(image)
        level = params.threshold * wavelet.noise_sigma(image)
        _, n_components = ndimage.label(w2 > level, np.ones((3, 3)))
        assert n_components == 1
        y, x, _ = wavelet.detect_regions(image, params)
        assert len(y) == 2
        np.testing.assert_allclose(sorted(x), [28, 32], atol=0.5)
        np.testing.assert_allclose(y, 30, atol=0.5)

    def test_min_area_removes_small_regions(self):
        # a single bright pixel yields a W2 blob of a handful of pixels
        image = np.random.default_rng(8).normal(1000, 1, (40, 40))
        image[20, 20] += 50
        image = image.astype(np.float32)
        _, _, area = wavelet.detect_regions(
            image, wavelet.WaveletParameters(threshold=2, min_area=1)
        )
        assert len(area)
        largest = int(area.max())
        kept = wavelet.detect_regions(
            image, wavelet.WaveletParameters(threshold=2, min_area=largest)
        )[2]
        dropped = wavelet.detect_regions(
            image,
            wavelet.WaveletParameters(threshold=2, min_area=largest + 1),
        )[2]
        assert largest in kept
        assert largest not in dropped

    def test_constant_image_has_no_regions(self):
        y, x, area = wavelet.detect_regions(np.full((20, 20), 5.0))
        assert len(y) == len(x) == len(area) == 0

    def test_deterministic(self):
        image = _spots_frame((64, 64), [(20, 20), (40, 45)], seed=9)
        first = wavelet.detect_regions(image)
        second = wavelet.detect_regions(image)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


class TestIdentifyInImage:
    def test_box_centers_are_rounded_centroids(self):
        centers = [(20.3, 30.6), (45.7, 15.2)]
        image = _spots_frame((64, 64), centers, amplitude=3000, seed=10)
        y, x = wavelet.identify_in_image(image, 7)
        assert sorted(zip(y.tolist(), x.tolist())) == [(20, 31), (46, 15)]
        assert y.dtype.kind == "i" and x.dtype.kind == "i"

    def test_border_band_matches_net_gradient(self):
        # the band localize._local_maxima searches:
        # box//2 <= c < n - box//2 - 1
        box = 7
        centers = [(2.0, 30.0), (3.0, 40.0), (60.0, 30.0), (59.0, 40.0)]
        image = _spots_frame((64, 64), centers, amplitude=3000, seed=11)
        y, x = wavelet.identify_in_image(image, box)
        assert set(y.tolist()) == {3, 59}
        assert np.all(y >= box // 2) and np.all(y < 64 - box // 2 - 1)

    def test_raster_order(self):
        centers = [(40.0, 10.0), (10.0, 40.0), (40.0, 40.0), (10.0, 10.0)]
        image = _spots_frame((52, 52), centers, amplitude=3000, seed=12)
        y, x = wavelet.identify_in_image(image, 7)
        order = np.lexsort((x, y))
        np.testing.assert_array_equal(order, np.arange(len(y)))

    def test_higher_threshold_finds_fewer(self):
        rng = np.random.default_rng(13)
        centers = rng.uniform(8, 120, (40, 2))
        amplitudes = rng.uniform(50, 1500, 40)
        image = np.zeros((128, 128))
        for (cy, cx), a in zip(centers, amplitudes):
            image += _spots_frame(
                (128, 128), [(cy, cx)], amplitude=a, background=0, seed=None
            )
        image = rng.poisson(image + 1000).astype(np.float32)
        counts = [
            len(
                wavelet.identify_in_image(
                    image, 7, wavelet.WaveletParameters(threshold=t)
                )[0]
            )
            for t in (0.5, 1.0, 2.0)
        ]
        assert counts[0] >= counts[1] >= counts[2]
        assert counts[0] > counts[2]


class TestWaveletParameters:
    def test_defaults_are_the_papers(self):
        params = wavelet.WaveletParameters()
        assert params.threshold == 0.5
        assert params.noise == wavelet.NOISE_IMAGE_STD
        assert params.min_area == 4

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": -0.1},
            {"threshold": float("nan")},
            {"noise": "median"},
            {"min_area": 0},
            {"min_area": 2.5},
        ],
    )
    def test_validation(self, kwargs):
        with pytest.raises(ValueError):
            wavelet.WaveletParameters(**kwargs)

    def test_types_are_normalized(self):
        params = wavelet.WaveletParameters(threshold=1, min_area=np.int64(3))
        assert type(params.threshold) is float
        assert type(params.min_area) is int

    def test_info_round_trip(self):
        params = wavelet.WaveletParameters(
            threshold=1.25, noise=wavelet.NOISE_W1_MAD, min_area=6
        )
        info = params.to_info()
        assert set(info) == {
            wavelet.INFO_THRESHOLD,
            wavelet.INFO_NOISE,
            wavelet.INFO_MIN_AREA,
        }
        assert wavelet.WaveletParameters.from_info(info) == params
        assert (
            wavelet.WaveletParameters.from_info({})
            == wavelet.WaveletParameters()
        )

    def test_to_dict_is_plain(self):
        params = wavelet.WaveletParameters(threshold=2)
        assert params.to_dict() == {
            "threshold": 2.0,
            "noise": "image_std",
            "min_area": 4,
        }

    def test_frozen(self):
        with pytest.raises(AttributeError):
            wavelet.WaveletParameters().threshold = 1.0
