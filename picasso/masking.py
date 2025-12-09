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
import pandas as pd
from scipy import ndimage as ndi


def mask_locs(
    locs: pd.DataFrame,
    mask: np.ndarray,
    width: float,
    height: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask localizations given a binary mask.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations to be masked.
    mask : np.ndarray
        Binary mask where True indicates the area to keep.
    width : float
        Maximum x coordinate of the localizations.
    height : float
        Maximum y coordinate of the localizations.

    Returns
    -------
    locs_in : pd.DataFrame
        Localizations inside the mask.
    locs_out : pd.DataFrame
        Localizations outside the mask.
    """
    x_ind = (np.floor(locs["x"].values / width * mask.shape[0])).astype(int)
    y_ind = (np.floor(locs["y"].values / height * mask.shape[1])).astype(int)

    index = mask[y_ind, x_ind].astype(bool)
    locs_in = locs.iloc[index].sort_values(by="frame", kind="mergesort")
    locs_out = locs.iloc[~index].sort_values(by="frame", kind="mergesort")
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
    method: (
        float
        | Literal[
            "isodata",
            "li",
            "mean",
            "minimum",
            "otsu",
            "triangle",
            "yen",
            "local_gaussian",
            "local_mean",
            "local_median",
        ]
    ) = "otsu",
) -> tuple[np.ndarray, float] | tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    mask : np.ndarray
        Binary mask where True indicates pixels above the threshold.
    threshold : float or np.ndarray
        Threshold value used to create the mask. Can be a single float
        for global thresholding or an array for pixel-wise thresholding.
    """
    assert image.ndim == 2, "Input image must be 2D"
    if not isinstance(method, str):
        threshold = float(method)
        mask = binary_mask(image, threshold)
        return mask
    threshold_functions = {
        "isodata": threshold_isodata,
        "li": threshold_li,
        "mean": threshold_mean,
        "minimum": threshold_minimum,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
        "yen": threshold_yen,
        "local_gaussian": threshold_local_gaussian,
        "local_mean": threshold_local_mean,
        "local_median": threshold_local_median,
    }
    function = threshold_functions.get(method)
    if function is None:
        raise ValueError(f"Unknown thresholding method: {method}")
    threshold = function(image)
    mask = binary_mask(image, threshold)
    return mask, threshold


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

    References
    ----------
    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
           iterative selection method"
           IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
           :DOI:`10.1109/TSMC.1978.4310039`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
           :DOI:`10.1117/1.1631315`
    .. [3] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold
    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

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

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8):
           771-776 :DOI:`10.1016/S0167-8655(98)00057-9`
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           :DOI:`10.1117/1.1631315`
    .. [4] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold
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

        t_next = (mean_back - mean_fore) / (
            np.log(mean_back) - np.log(mean_fore)
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

    References
    ----------
    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding
        algorithms," CVGIP: Graphical Models and Image Processing,
        vol. 55, pp. 532-537, 1993.
        :DOI:`10.1006/cgip.1993.1040`
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

    References
    ----------
    .. [1] C. A. Glasbey, "An analysis of histogram-based thresholding
           algorithms," CVGIP: Graphical Models and Image Processing,
           vol. 55, pp. 532-537, 1993.
    .. [2] Prewitt, JMS & Mendelsohn, ML (1966), "The analysis of cell
           images", Annals of the New York Academy of Sciences 128: 1035-1053
           :DOI:`10.1111/j.1749-6632.1965.tb11715.x`
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
    smooth_hist = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for counter in range(10000):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError("Unable to find two maxima in histogram")
    elif counter == 10000 - 1:
        raise RuntimeError("Maximum iteration reached for histogram smoothing")

    # Find the lowest point between the maxima
    threshold_idx = np.argmin(
        smooth_hist[maximum_idxs[0] : maximum_idxs[1] + 1]
    )
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

    References
    ----------
    .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
    """
    # histogram the image and converts bin edges to bin centers
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

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

    References
    ----------
    .. [1] Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,
       Automatic Measurement of Sister Chromatid Exchange Frequency,
       Journal of Histochemistry and Cytochemistry 25 (7), pp. 741-753
       :DOI:`10.1177/25.7.70454`
    .. [2] ImageJ AutoThresholder code,
       http://fiji.sc/wiki/index.php/Auto_Threshold
    """
    hist, bin_edges = np.histogram(image.reshape(-1), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
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

    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378. :DOI:`10.1109/83.366472`
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165, :DOI:`10.1117/1.1631315`
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold

    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = counts.astype("float32", copy=False) / counts.sum()
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

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
           ISBN: 0-201-18075-8
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    sigma = tuple([(b - 1) / 6.0 for b in block_size])
    ndi.gaussian_filter(
        image,
        sigma=sigma,
        output=thresh_image,
        mode="reflect",
    )
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

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
           ISBN: 0-201-18075-8
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    ndi.uniform_filter(image, block_size, output=thresh_image, mode="reflect")
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

    References
    ----------
    .. [1] Gonzalez, R. C. and Wood, R. E. "Digital Image Processing
           (2nd Edition)." Prentice-Hall Inc., 2002: 600--612.
           ISBN: 0-201-18075-8
    """
    block_size = (3, 3)
    thresh_image = np.zeros(image.shape, dtype=image.dtype)
    ndi.median_filter(image, block_size, output=thresh_image, mode="reflect")
    mask = np.zeros(image.shape, dtype=bool)
    mask[image > thresh_image] = True
    return mask


def threshold_tukey(image: np.ndarray) -> np.ndarray:
    """Find the Tukey's mask, used to avoid FFT artifacts.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    mask : np.ndarray
        Tukey's mask (binary).
    """
    assert image.shape[0] == image.shape[1], "Image must be square"
    nfac = 8
    height, width = image.shape
    x = np.arange(width)
    x_im = (x - (width / 2)) / width
    x_im = np.tile(x_im, (height, 1))
    mask = 0.5 - 0.5 * np.cos(np.pi * nfac * x_im)
    mask[np.abs(x_im) < ((nfac - 2) / (nfac * 2))] = 1
    mask = mask * np.rot90(mask)
    return mask
