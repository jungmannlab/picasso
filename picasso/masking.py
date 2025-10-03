"""
    picasso.masking
    ~~~~~~~~~~~~~~~

    Functions for masking localizations based on binary masks or
    thresholding of images.

    Thresholding functions are adapted from scikit-image. The package is
    not used directly to avoid extra dependencies.

    :author: Rafal Kowalewski 2025
    :copyright: Copyright (c) 2015-2025 Jungmann Lab, MPI Biochemistry
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    uniform_filter,
    uniform_filter1d,
    median_filter,
)


def mask_locs(
    locs: np.recarray,
    mask: np.ndarray,
    width: float,
    height: float,
) -> tuple[np.recarray, np.recarray]:
    """Mask localizations given a binary mask.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be masked.
    mask : np.ndarray
        Binary mask where True indicates the area to keep.
    width : float
        Maximum x coordinate of the localizations.
    height : float
        Maximum y coordinate of the localizations.

    Returns
    -------
    locs_in : np.recarray
        Localizations inside the mask.
    locs_out : np.recarray
        Localizations outside the mask.
    """
    x_ind = (np.floor(locs["x"] / width * mask.shape[0])).astype(int)
    y_ind = (np.floor(locs["y"] / height * mask.shape[1])).astype(int)

    index = mask[y_ind, x_ind].astype(bool)
    locs_in = locs[index]
    locs_in.sort(kind="mergesort", order="frame")
    locs_out = locs[~index]
    locs_out.sort(kind="mergesort", order="frame")

    return locs_in, locs_out


def binary_mask(
    image: np.ndarray,
    threshold: float | np.ndarray,
) -> np.ndarray:
    """Create a binary mask from an image given a threshold.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    threshold : float or np.ndarray
        Threshold value or array of threshold values. If a single float
        is provided, it is used as a global threshold. If an array is
        provided, it should have the same shape as the input image and
        is used as a pixel-wise threshold.

    Returns
    -------
    mask : np.ndarray
        Binary mask where True indicates pixels above the threshold.
    """
    mask = np.zeros(image.shape, dtype=bool)
    if np.isscalar(threshold):
        mask[image > threshold] = True
    else:
        if threshold.shape != image.shape:
            raise ValueError(
                "Threshold array must have the same shape as the image"
            )
        mask[image > threshold] = True
    return mask


def mask_image(
    image: np.ndarray,
    method: float | Literal[
        'isodata',
        'li',
        'mean',
        'minimum',
        'otsu',
        'triangle',
        'yen',
        'local_gaussian',
        'local_mean',
        'local_median',
    ] = 'otsu',
) -> np.ndarray:
    """Create a binary mask from a grayscale image using a specified
    thresholding method or threshold value.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    method : float or {'isodata', 'li', 'mean', 'minimum', 'otsu',
            'triangle', 'yen', 'local_gaussian', 'local_mean',
            'local_median'}, optional
        Thresholding method or threshold value. If a float is provided,
        it is used as a global threshold. If a string is provided, it
        specifies the thresholding method to use. Default is 'otsu'.
    """
    assert image.ndim == 2, "Input image must be 2D"
    if np.isscalar(method):
        threshold = float(method)
        mask = binary_mask(image, threshold)
        return mask
    threshold_functions = {
        'isodata': threshold_isodata,
        'li': threshold_li,
        'mean': threshold_mean,
        'minimum': threshold_minimum,
        'otsu': threshold_otsu,
        'triangle': threshold_triangle,
        'yen': threshold_yen,
        'local_gaussian': threshold_local_gaussian,
        'local_mean': threshold_local_mean,
        'local_median': threshold_local_median,
    }
    function = threshold_functions.get(method)
    if function is None:
        raise ValueError(f"Unknown thresholding method: {method}")
    threshold = function(image)
    mask = binary_mask(image, threshold)
    return mask


def threshold_isodata(image: np.ndarray) -> float:
    """Return threshold value based on the isodata method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype('float32', copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # image only contains one unique value
    if len(bin_centers) == 1:
        return bin_centers[0]

    csuml = np.cumsum(counts)
    csumh = csuml[-1] - csuml
    intensity_sum = counts * bin_centers

    csum_intensity = np.cumsum(intensity_sum)
    lower = csum_intensity[:-1] / csuml[:-1]
    higher = (csum_intensity[-1] - csum_intensity[:-1]) / csumh[:-1]

    all_mean = (lower + higher) / 2.0
    bin_width = bin_centers[1] - bin_centers[0]

    distances = all_mean - bin_centers[:-1]
    threshold = bin_centers[:-1][(distances >= 0) & (distances < bin_width)][0]
    return threshold


def threshold_li(image: np.ndarray) -> float:
    """Return threshold value based on Li's minimum cross entropy method
    for a grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    # Li's algorithm requires positive image (because of log(mean))
    if np.any(image < 0):
        raise ValueError("Li's method requires a non-negative image")

    # Make sure image has more than one value; otherwise, return that value
    # This works even for np.inf
    if np.all(image == image.flat[0]):
        return image.flat[0]

    tolerance = np.min(np.diff(np.unique(image))) / 2
    t_next = np.mean(image)
    t_curr = -2 * tolerance

    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = image > t_curr
        mean_fore = np.mean(image[foreground])
        mean_back = np.mean(image[~foreground])

        if mean_back == 0.0:
            break

        t_next = (
            (mean_back - mean_fore)
            / (np.log(mean_back) - np.log(mean_fore))
        )
    threshold = t_next
    return threshold


def threshold_mean(image: np.ndarray) -> float:
    """Return threshold value based on the mean method for a grayscale
    image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    threshold : float
        Mean threshold value.
    """
    threshold = image.mean()
    return threshold


def threshold_minimum(image):
    """Return threshold value based on minimum method.

    The histogram of the input ``image`` is computed if not provided and
    smoothed until there are only two maxima. Then the minimum in
    between is the threshold value.

    Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    threshold : float
        Mean threshold value.
    """

    def find_local_maxima_idx(hist):
        maximum_idxs = list()
        direction = 1
        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1
                    maximum_idxs.append(i)
            else:
                if hist[i + 1] > hist[i]:
                    direction = 1
        return maximum_idxs

    counts, bin_edges = np.histogram(image, bins=256)
    smooth_hist = counts.astype('float32', copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    for counter in range(10000):
        smooth_hist = uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError('Unable to find two maxima in histogram')
    elif counter == 10000 - 1:
        raise RuntimeError('Maximum iteration reached for histogram smoothing')

    # Find the lowest point between the maxima
    threshold_idx = np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1])
    threshold = bin_centers[maximum_idxs[0] + threshold_idx]
    return threshold


def threshold_otsu(image: np.ndarray) -> float:
    """Return threshold value based on Otsu's method for a grayscale
    image. Adapted from scikit-image.

    Based on implementation in scikit-image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    threshold : float
        Otsu's threshold value.
    """
    # histogram the image and converts bin edges to bin centers
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype('float32', copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold


def threshold_triangle(image: np.ndarray) -> float:
    """Return threshold value based on the triangle algorithm for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    hist, bin_edges = np.histogram(image.reshape(-1), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    nbins = len(hist)

    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]

    if arg_low_level == arg_high_level:
        return image.ravel()[0]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    del arg_high_level

    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    threshold = bin_centers[arg_level]
    return threshold


def threshold_yen(image: np.ndarray) -> float:
    """Return threshold value based on Yen's method for a grayscale
    image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    threshold : float
        Threshold value.
    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype('float32', copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = counts.astype('float32', copy=False) / counts.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf**2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(
        ((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2
    )
    threshold = bin_centers[crit.argmax()]
    return threshold


# threshold methods that return pixel-wise thresholds
def threshold_local_gaussian(image: np.ndarray) -> np.ndarray:
    """Return threshold value based on the Gaussian local method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    mask : np.ndarray
        Binary mask. Values of 1 indicate foreground pixels.
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    sigma = tuple([(b - 1) / 6.0 for b in block_size])
    gaussian_filter(image, sigma=sigma, out=thresh_image, mode='reflect')
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask


def threshold_local_mean(image: np.ndarray) -> np.ndarray:
    """Return threshold value based on the mean local method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    mask : np.ndarray
        Binary mask. Values of 1 indicate foreground pixels.
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    uniform_filter(image, block_size, output=thresh_image, mode='reflect')
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask


def threshold_local_median(image: np.ndarray) -> np.ndarray:
    """Return threshold value based on the median local method for a
    grayscale image. Adapted from scikit-image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    mask : np.ndarray
        Binary mask. Values of 1 indicate foreground pixels.
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    median_filter(image, block_size, output=thresh_image, mode='reflect')
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask
