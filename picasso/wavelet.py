"""
picasso.wavelet
~~~~~~~~~~~~~~~

Spot identification by B-spline wavelet segmentation, after Izeddin et al.
(2012).

An alternative to the net gradient detector of :mod:`picasso.localize`. It
finds the boxes that are fitted afterwards; the localization itself is still
done by the Gaussian or spline fit. Per frame:

1. The frame is decomposed with the undecimated ("à trous") wavelet transform
   using the third-order B-spline low-pass ``[1, 4, 6, 4, 1] / 16``. Level 1
   smooths the frame, ``V1 = g1 * V0``; level 2 smooths again with the same
   taps spaced two pixels apart, ``V2 = g2 * V1``. The second wavelet plane
   ``W2 = V1 - V2`` keeps structures of about the size of a diffraction-limited
   spot and discards both the pixel noise (in ``W1 = V0 - V1``) and the
   background (in ``V2``).
2. ``W2`` is thresholded at ``threshold`` times the standard deviation of the
   noise. The paper chooses the factor between 0.5 and 2 and uses 0.5 in its
   Fig. 2, which is the default here. The noise is estimated per frame, either
   as the standard deviation of the frame itself (``"image_std"``, the paper's
   choice when single molecules are sparse, and the default) or from the first
   wavelet plane (``"w1_mad"``, the median absolute deviation estimate of
   Donoho & Johnstone that the paper cites for an automatic estimate). The
   standard deviation of a frame includes its spots and so overestimates the
   noise, which the low default relies on: where it does not (sparse frames,
   or the ``"w1_mad"`` estimate, which measures the noise itself), 0.5 lets
   noise through as spots, and a factor of about 1 is the better choice.
3. The thresholded map is split into one region per spot with a watershed on
   ``W2``, so that overlapping spots are separated.
4. Regions of less than ``min_area`` pixels (4 in the paper) are removed as
   noise.
5. The centroid of each region, weighted by ``W2``, rounded to the nearest
   pixel, is the center of the box that is fitted.

The paper does not state how the frame borders are extended; this module
mirrors them (``d c b | a b c d | c b a``, i.e. ``scipy.ndimage``'s
``"mirror"`` mode), as the reference implementation of the ``W2`` filter by
Hekrdla & Petráková does. The watershed floods from the regional maxima of
``W2`` in order of decreasing value (Vincent & Soille's immersion) with
8-connected pixels and assigns every pixel of the mask to a region, i.e. it
draws no watershed lines.

The threshold is in units of the noise, so unlike the minimum net gradient it
does not depend on the camera gain or the brightness of the dye, and one value
serves all channels of a multichannel acquisition. The identifications carry no
``net_gradient`` column.

References
----------
Izeddin, I., Boulanger, J., Racine, V., Specht, C. G., Kechkar, A., Nair, D.,
Triller, A., Choquet, D., Dahan, M. & Sibarita, J. B. "Wavelet analysis for
single molecule localization microscopy." Optics Express 20, 2081-2095 (2012).
https://doi.org/10.1364/OE.20.002081

Donoho, D. & Johnstone, I. "Adapting to unknown smoothness via wavelet
shrinkage." Journal of the American Statistical Association 90, 1200-1224
(1995).

Vincent, L. & Soille, P. "Watersheds in digital spaces: an efficient algorithm
based on immersion simulations." IEEE Transactions on Pattern Analysis and
Machine Intelligence 13, 583-598 (1991).

Hekrdla, M. & Petráková, V. "Optimized molecule detection in localization
microscopy with selected false positive probability." (2024).

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from . import lib

#: Low-pass taps of the third-order B-spline, ``[H2, H1, H0, H1, H2]`` with
#: ``H0 = 3/8``, ``H1 = 1/4`` and ``H2 = 1/16``.
B3_KERNEL = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16], dtype=np.float32)
_B3_RADIUS = 2

#: Farthest a pixel of ``W2`` reads from: 2 pixels for the first smoothing and
#: 4 for the second, whose taps are spaced two pixels apart.
WAVELET_RADIUS = 6

#: Standard deviation of the first wavelet plane of unit white noise, i.e. the
#: L2 norm of the kernel ``delta - g1 g1^T`` that produces ``W1``.
W1_NOISE_GAIN = float(
    np.sqrt(
        np.sum(
            (
                np.pad([[1.0]], _B3_RADIUS)
                - np.outer(B3_KERNEL, B3_KERNEL).astype(np.float64)
            )
            ** 2
        )
    )
)
# Standard deviation of a standard normal per unit of its median absolute
# deviation, Phi^{-1}(3/4).
_MAD_TO_SIGMA = 0.6744897501960817

#: Noise estimate from the standard deviation of the frame itself.
NOISE_IMAGE_STD = "image_std"
#: Noise estimate from the median absolute deviation of the first wavelet
#: plane.
NOISE_W1_MAD = "w1_mad"
NOISE_ESTIMATES = (NOISE_IMAGE_STD, NOISE_W1_MAD)

# Keys of the parameters in the identification metadata.
INFO_THRESHOLD = "Wavelet Threshold"
INFO_NOISE = "Wavelet Noise Estimate"
INFO_MIN_AREA = "Wavelet Min. Area"


@dataclass(frozen=True)
class WaveletParameters:
    """Settings of the wavelet spot identification.

    The defaults are those of Izeddin et al. (2012).

    Parameters
    ----------
    threshold : float, optional
        Threshold on the second wavelet plane, in units of the standard
        deviation of the noise. The paper uses values between 0.5 and 2.
        Default is 0.5.
    noise : {"image_std", "w1_mad"}, optional
        How the noise is estimated in each frame: the standard deviation
        of the frame (``"image_std"``), which is a good estimate if the
        spots are sparse, or the median absolute deviation of the first
        wavelet plane (``"w1_mad"``), which is robust to dense spots and
        to an inhomogeneous background. Default is ``"image_std"``.
    min_area : int, optional
        Regions of fewer pixels are discarded as noise. Default is 4.

    Raises
    ------
    ValueError
        If ``threshold`` is negative or not finite, ``noise`` is unknown
        or ``min_area`` is smaller than 1.
    """

    threshold: float = 0.5
    noise: str = NOISE_IMAGE_STD
    min_area: int = 4

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "The wavelet threshold must be a non-negative number, got "
                f"{self.threshold}."
            )
        if self.noise not in NOISE_ESTIMATES:
            raise ValueError(
                f"Unknown wavelet noise estimate {self.noise!r}; use one of "
                f"{', '.join(NOISE_ESTIMATES)}."
            )
        if int(self.min_area) != self.min_area or self.min_area < 1:
            raise ValueError(
                "The minimum region area must be a positive integer, got "
                f"{self.min_area}."
            )
        # frozen: normalize the types through object.__setattr__
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "min_area", int(self.min_area))

    def to_dict(self) -> dict:
        """Return the parameters as a plain dictionary.

        Returns
        -------
        params : dict
            Threshold, noise estimate and minimum area under the field
            names as keys.
        """
        return {
            "threshold": self.threshold,
            "noise": self.noise,
            "min_area": self.min_area,
        }

    def to_info(self) -> dict:
        """Parameters under the keys of the identification metadata."""
        return {
            INFO_THRESHOLD: self.threshold,
            INFO_NOISE: self.noise,
            INFO_MIN_AREA: self.min_area,
        }

    @classmethod
    def from_info(cls, info: dict) -> WaveletParameters:
        """Read the parameters from a dictionary with the keys of
        :meth:`to_info`. Missing keys take the default values.

        Parameters
        ----------
        info : dict
            Identification metadata or parameters.

        Returns
        -------
        WaveletParameters
            The parameters found in ``info``.
        """
        default = cls()
        return cls(
            threshold=info.get(INFO_THRESHOLD, default.threshold),
            noise=info.get(INFO_NOISE, default.noise),
            min_area=info.get(INFO_MIN_AREA, default.min_area),
        )


@numba.jit(nopython=True, nogil=True, cache=False)
def _mirror_index(i: int, n: int) -> int:
    """Index ``i`` reflected into ``[0, n)`` without repeating the edge
    pixel, as often as needed for kernels wider than the image."""
    if 0 <= i < n:
        return i
    if n == 1:
        return 0
    period = 2 * (n - 1)
    i = i % period
    if i >= n:
        i = period - i
    return i


@numba.jit(nopython=True, nogil=True, cache=False)
def _smooth(image: lib.FloatArray2D, step: int) -> lib.FloatArray2D:
    """One level of the à trous transform: the separable B-spline
    low-pass with its taps ``step`` pixels apart, over the rows and then
    the columns, with mirrored borders."""
    n_y, n_x = image.shape
    rows = np.empty((n_y, n_x), dtype=np.float32)
    for y in range(n_y):
        for x in range(n_x):
            acc = 0.0
            for t in range(-_B3_RADIUS, _B3_RADIUS + 1):
                xx = _mirror_index(x + t * step, n_x)
                acc += B3_KERNEL[t + _B3_RADIUS] * image[y, xx]
            rows[y, x] = acc
    out = np.empty((n_y, n_x), dtype=np.float32)
    for y in range(n_y):
        for x in range(n_x):
            acc = 0.0
            for t in range(-_B3_RADIUS, _B3_RADIUS + 1):
                yy = _mirror_index(y + t * step, n_y)
                acc += B3_KERNEL[t + _B3_RADIUS] * rows[yy, x]
            out[y, x] = acc
    return out


@numba.jit(nopython=True, nogil=True, cache=False)
def _wavelet_planes(
    image: lib.FloatArray2D,
) -> tuple[lib.FloatArray2D, lib.FloatArray2D]:
    """First and second wavelet planes of a float32 image."""
    v1 = _smooth(image, 1)
    v2 = _smooth(v1, 2)
    return image - v1, v1 - v2


@numba.jit(nopython=True, nogil=True, cache=False)
def _noise_sigma(
    image: lib.FloatArray2D, w1: lib.FloatArray2D, use_mad: bool
) -> float:
    """Standard deviation of the noise, see ``noise_sigma``."""
    if use_mad:
        return np.median(np.abs(w1)) / _MAD_TO_SIGMA / W1_NOISE_GAIN
    # two passes in double precision: large frames lose digits otherwise
    n = image.size
    total = 0.0
    for value in image.ravel():
        total += value
    mean = total / n
    sq = 0.0
    for value in image.ravel():
        sq += (value - mean) ** 2
    return np.sqrt(sq / n)


@numba.jit(nopython=True, nogil=True, cache=False)
def _watershed_regions(
    w2: lib.FloatArray2D, level: float, min_area: int
) -> tuple[lib.FloatArray1D, lib.FloatArray1D, lib.IntArray1D]:
    """Split the pixels of ``w2`` above ``level`` into regions with a
    watershed and return the ``w2``-weighted centroid (y, x) and the area
    of every region of at least ``min_area`` pixels.

    The pixels are flooded in order of decreasing value. A pixel none of
    whose 8 neighbors is flooded yet is a regional maximum and starts a
    new region; any other pixel joins the region of its highest flooded
    neighbor. The sort is stable, so ties are broken in raster order and
    the result is deterministic.
    """
    ys, xs, values = _pixels_above(w2, level)
    labels, n_labels = _flood_labels(w2, ys, xs, values)
    return _region_centroids(labels, n_labels, ys, xs, values, min_area)


@numba.jit(nopython=True, nogil=True, cache=False)
def _pixels_above(
    w2: lib.FloatArray2D, level: float
) -> tuple[lib.IntArray1D, lib.IntArray1D, lib.FloatArray1D]:
    """Coordinates (y, x) and values of the pixels of ``w2`` above
    ``level``, in raster order."""
    n_y, n_x = w2.shape
    n = 0
    for y in range(n_y):
        for x in range(n_x):
            if w2[y, x] > level:
                n += 1
    ys = np.empty(n, dtype=np.int64)
    xs = np.empty(n, dtype=np.int64)
    values = np.empty(n, dtype=np.float64)
    i = 0
    for y in range(n_y):
        for x in range(n_x):
            if w2[y, x] > level:
                ys[i] = y
                xs[i] = x
                values[i] = w2[y, x]
                i += 1
    return ys, xs, values


@numba.jit(nopython=True, nogil=True, cache=False)
def _flood_labels(
    w2: lib.FloatArray2D,
    ys: lib.IntArray1D,
    xs: lib.IntArray1D,
    values: lib.FloatArray1D,
) -> tuple[lib.IntArray2D, int]:
    """Watershed labels (1, 2, ...; 0: not flooded) of the pixels
    ``(ys, xs)``, flooded in order of decreasing ``values``, see
    ``_watershed_regions``. Returns the labels and their number."""
    order = np.argsort(-values, kind="mergesort")
    labels = np.zeros(w2.shape, dtype=np.int64)  # 0: not flooded
    n_labels = 0
    for index in order:
        y = ys[index]
        x = xs[index]
        label = _highest_neighbor_label(w2, labels, y, x)
        if label == 0:
            n_labels += 1
            label = n_labels
        labels[y, x] = label
    return labels, n_labels


@numba.jit(nopython=True, nogil=True, cache=False)
def _highest_neighbor_label(
    w2: lib.FloatArray2D, labels: lib.IntArray2D, y: int, x: int
) -> int:
    """Label of the highest flooded of the 8 neighbors of ``(y, x)``,
    or 0 if none is flooded."""
    n_y, n_x = w2.shape
    label = 0
    highest = -np.inf
    for dy in range(-1, 2):
        yy = y + dy
        if yy < 0 or yy >= n_y:
            continue
        for dx in range(-1, 2):
            xx = x + dx
            if (dy == 0 and dx == 0) or xx < 0 or xx >= n_x:
                continue
            if labels[yy, xx] > 0 and w2[yy, xx] > highest:
                highest = w2[yy, xx]
                label = labels[yy, xx]
    return label


@numba.jit(nopython=True, nogil=True, cache=False)
def _region_centroids(
    labels: lib.IntArray2D,
    n_labels: int,
    ys: lib.IntArray1D,
    xs: lib.IntArray1D,
    values: lib.FloatArray1D,
    min_area: int,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D, lib.IntArray1D]:
    """Value-weighted centroid (y, x) and area of every labeled region
    of at least ``min_area`` pixels, see ``_watershed_regions``."""
    area = np.zeros(n_labels + 1, dtype=np.int64)
    weight = np.zeros(n_labels + 1, dtype=np.float64)
    sum_y = np.zeros(n_labels + 1, dtype=np.float64)
    sum_x = np.zeros(n_labels + 1, dtype=np.float64)
    for i in range(len(values)):
        label = labels[ys[i], xs[i]]
        area[label] += 1
        weight[label] += values[i]
        sum_y[label] += values[i] * ys[i]
        sum_x[label] += values[i] * xs[i]
    keep = area[1:] >= min_area
    n_keep = keep.sum()
    cy = np.empty(n_keep, dtype=np.float64)
    cx = np.empty(n_keep, dtype=np.float64)
    kept_area = np.empty(n_keep, dtype=np.int64)
    j = 0
    for label in range(1, n_labels + 1):
        if keep[label - 1]:
            cy[j] = sum_y[label] / weight[label]
            cx[j] = sum_x[label] / weight[label]
            kept_area[j] = area[label]
            j += 1
    return cy, cx, kept_area


@numba.jit(nopython=True, nogil=True, cache=False)
def _detect_regions(
    image: lib.FloatArray2D, threshold: float, use_mad: bool, min_area: int
) -> tuple[lib.FloatArray1D, lib.FloatArray1D, lib.IntArray1D]:
    """Wavelet segmentation of a float32 image, see ``detect_regions``."""
    w1, w2 = _wavelet_planes(image)
    sigma = _noise_sigma(image, w1, use_mad)
    if not sigma > 0:
        # a noise-free (e.g. constant) frame has no scale to threshold at
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    return _watershed_regions(w2, threshold * sigma, min_area)


@numba.jit(nopython=True, nogil=True, cache=False)
def _identify_in_image(
    image: lib.FloatArray2D,
    box: int,
    threshold: float,
    use_mad: bool,
    min_area: int,
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Box centers of the wavelet regions, see ``identify_in_image``."""
    cy, cx, _ = _detect_regions(image, threshold, use_mad, min_area)
    n_y, n_x = image.shape
    box_half = box // 2
    # the band localize._local_maxima searches, so that every box can be
    # cut from the frame; a map also merges regions rounding to one pixel
    centers = np.zeros((n_y, n_x), dtype=np.uint8)
    for i in range(len(cy)):
        y = int(np.floor(cy[i] + 0.5))
        x = int(np.floor(cx[i] + 0.5))
        if box_half <= y < n_y - box_half - 1:
            if box_half <= x < n_x - box_half - 1:
                centers[y, x] = 1
    return np.where(centers)


def _as_float32_image(image: np.ndarray) -> lib.FloatArray2D:
    """C-contiguous float32 copy (or view) of a 2D image."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")
    return image


def wavelet_planes(
    image: np.ndarray,
) -> tuple[lib.FloatArray2D, lib.FloatArray2D]:
    """First and second planes of the à trous B-spline wavelet transform.

    Parameters
    ----------
    image : np.ndarray
        2D image of shape (Y, X).

    Returns
    -------
    w1 : lib.FloatArray2D
        First wavelet plane, ``V0 - V1``, holding the pixel noise.
    w2 : lib.FloatArray2D
        Second wavelet plane, ``V1 - V2``, holding structures of about
        the size of a diffraction-limited spot. Float32, same shape as
        ``image``.
    """
    return _wavelet_planes(_as_float32_image(image))


def noise_sigma(
    image: np.ndarray,
    w1: np.ndarray | None = None,
    noise: str = NOISE_IMAGE_STD,
) -> float:
    """Standard deviation of the noise in an image.

    Parameters
    ----------
    image : np.ndarray
        2D image of shape (Y, X).
    w1 : np.ndarray, optional
        First wavelet plane of ``image`` (see :func:`wavelet_planes`).
        Computed if not given and needed.
    noise : {"image_std", "w1_mad"}, optional
        Estimator, see :class:`WaveletParameters`. Default is
        ``"image_std"``.

    Returns
    -------
    sigma : float
        Estimated standard deviation of the noise, in the units of
        ``image``.
    """
    if noise not in NOISE_ESTIMATES:
        raise ValueError(f"Unknown wavelet noise estimate {noise!r}.")
    image = _as_float32_image(image)
    if w1 is None:
        w1 = _wavelet_planes(image)[0]
    w1 = _as_float32_image(w1)
    return float(_noise_sigma(image, w1, noise == NOISE_W1_MAD))


def detect_regions(
    image: np.ndarray,
    parameters: WaveletParameters | None = None,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D, lib.IntArray1D]:
    """Segment the spots of an image and return their sub-pixel centroids.

    Parameters
    ----------
    image : np.ndarray
        2D image of shape (Y, X).
    parameters : WaveletParameters, optional
        Segmentation settings. Default is ``WaveletParameters()``, i.e.
        the settings of Izeddin et al. (2012).

    Returns
    -------
    y, x : lib.FloatArray1D
        Centroids of the regions, weighted by the second wavelet plane,
        in pixels.
    area : lib.IntArray1D
        Number of pixels of each region.
    """
    if parameters is None:
        parameters = WaveletParameters()
    return _detect_regions(
        _as_float32_image(image),
        parameters.threshold,
        parameters.noise == NOISE_W1_MAD,
        parameters.min_area,
    )


def identify_in_image(
    image: np.ndarray,
    box: int,
    parameters: WaveletParameters | None = None,
) -> tuple[lib.IntArray1D, lib.IntArray1D]:
    """Identify spots in an image by wavelet segmentation.

    Each region found by :func:`detect_regions` is one spot, centered at
    its centroid rounded to the nearest pixel. Spots whose box of side
    ``box`` would not fit into the image are dropped, exactly as the net
    gradient identification does, and regions that round to the same
    pixel are reported once.

    Parameters
    ----------
    image : np.ndarray
        2D image of shape (Y, X).
    box : int
        Side length of the box that is fitted around each spot. Should
        be an odd integer.
    parameters : WaveletParameters, optional
        Segmentation settings. Default is ``WaveletParameters()``.

    Returns
    -------
    y, x : lib.IntArray1D
        Pixel coordinates of the box centers, in raster order.
    """
    if parameters is None:
        parameters = WaveletParameters()
    return _identify_in_image(
        _as_float32_image(image),
        int(box),
        parameters.threshold,
        parameters.noise == NOISE_W1_MAD,
        parameters.min_area,
    )
