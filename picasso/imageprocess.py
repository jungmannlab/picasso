"""
    picasso.imageprocess
    ~~~~~~~~~~~~~~~~~~~~

    Image processing functions

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from . import lib, render, localize, postprocess


plt.style.use("ggplot")


def xcorr(imageA: np.ndarray, imageB: np.ndarray) -> np.ndarray:
    """Compute the cross-correlation of two images using FFT.

    Parameters
    ----------
    imageA, imageB : np.ndarray
        Input images to be cross-correlated. They should have the same
        shape.

    Returns
    -------
    res : np.ndarray
        The cross-correlation result, which is the inverse Fourier
        transform of the product of the Fourier transforms of the two
        images.
    """
    FimageA = np.fft.fft2(imageA)
    CFimageB = np.conj(np.fft.fft2(imageB))
    res = np.fft.fftshift(
        np.real(np.fft.ifft2((FimageA * CFimageB)))
    ) / np.sqrt(imageA.size)
    return res


def get_image_shift(
    imageA: np.ndarray,
    imageB: np.ndarray,
    box: int,
    roi: int | None = None,
    display: bool = False,
) -> tuple[float, float]:
    """Compute the shift from ``imageA`` to ``imageB``.

    Parameters
    ----------
    imageA, imageB : np.ndarray
        Input images to be cross-correlated. They should have the same
        shape.
    box : int
        Size of the box used for fitting the cross-correlation peak.
    roi : int, optional
        Region of interest size to cut out the center of the
        cross-correlation image. If None, the entire cross-correlation
        image is used.
    display : bool, optional
        If True, displays the images and the cross-correlation result.

    Returns
    -------
    yc, xc : float
        The y and x coordinates of the shift from imageA to imageB.
        The coordinates are adjusted to be relative to the center of
        the images.
    """
    if (np.sum(imageA) == 0) or (np.sum(imageB) == 0):
        return 0, 0
    # Compute image correlation
    XCorr = xcorr(imageA, imageB)
    # Cut out center roi
    Y, X = imageA.shape
    if roi is not None:
        Y_ = int((Y - roi) / 2)
        X_ = int((X - roi) / 2)
        if Y_ > 0:
            XCorr = XCorr[Y_:-Y_, :]
        else:
            Y_ = 0
        if X_ > 0:
            XCorr = XCorr[:, X_:-X_]
        else:
            X_ = 0
    else:
        Y_ = X_ = 0
    # A quarter of the fit ROI
    fit_X = int(box / 2)
    # A coordinate grid for the fitting ROI
    y, x = np.mgrid[-fit_X:fit_X + 1, -fit_X:fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X:y_max_ + fit_X + 1,
        x_max_ - fit_X:x_max_ + fit_X + 1,
    ]

    dimensions = FitROI.shape

    if 0 in dimensions or dimensions[0] != dimensions[1]:
        xc, yc = 0, 0
    else:
        def flat_2d_gaussian(coords, a, xc, yc, s, b):
            x, y = coords
            A = a * np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s**2) + b
            return A.flatten()

        p0 = [FitROI.max(), 0, 0, 1, FitROI.min()]
        bounds = (
            [0, -np.inf, -np.inf, 0, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        )
        popt, _ = curve_fit(
            flat_2d_gaussian, (x, y), FitROI.flatten(), p0=p0, bounds=bounds,
        )

        # Get maximum coordinates and add offsets
        xc = popt[1]
        yc = popt[2]
        xc += X_ + x_max_
        yc += Y_ + y_max_

        if display:
            plt.figure(figsize=(17, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(imageA, interpolation="none")
            plt.subplot(1, 3, 2)
            plt.imshow(imageB, interpolation="none")
            plt.subplot(1, 3, 3)
            plt.imshow(XCorr, interpolation="none")
            plt.plot(xc, yc, "x")
            plt.show()

        xc -= np.floor(X / 2)
        yc -= np.floor(Y / 2)

    return -yc, -xc


def rcc(
    segments: list[np.ndarray],
    max_shift: float | None = None,
    callback: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RCC, see Wang, Schnitzbauer, et al. Optics Express,
    2014. Return the shifts in x and y directions for each pair of
    segments.

    Parameters
    ----------
    segments : list of np.ndarray
        List of image segments to be correlated. Each segment should be
        a 2D numpy array representing an image.
    max_shift : float, optional
        Maximum allowed shift in pixels. If None, the default value is
        set to 5 pixels.
    callback : Callable[[int], None], optional
        A callback function that takes an integer argument and is called
        after processing each pair of segments. This can be used to
        update a progress bar or perform other actions during the
        computation.

    Returns
    -------
    shifts_x, shifts_y : np.ndarray
        2D numpy arrays containing the shifts in x and y directions for
        each pair of segments. The shape of the arrays is
        (n_segments, n_segments), where n_segments is the number of
        segments provided.
    """
    n_segments = len(segments)
    shifts_x = np.zeros((n_segments, n_segments))
    shifts_y = np.zeros((n_segments, n_segments))
    n_pairs = int(n_segments * (n_segments - 1) / 2)
    flag = 0
    if callback is None:
        with tqdm(
            total=n_pairs, desc="Correlating image pairs", unit="pairs"
        ) as progress_bar:
            for i in range(n_segments - 1):
                for j in range(i + 1, n_segments):
                    progress_bar.update()
                    shifts_y[i, j], shifts_x[i, j] = get_image_shift(
                        segments[i], segments[j], 5, max_shift
                    )
                    flag += 1
    else:
        callback(0)
        for i in range(n_segments - 1):
            for j in range(i + 1, n_segments):
                shifts_y[i, j], shifts_x[i, j] = get_image_shift(
                    segments[i], segments[j], 5, max_shift
                )
                flag += 1
                callback(flag)

    return lib.minimize_shifts(shifts_x, shifts_y)


def find_fiducials(
    locs: np.recarray,
    info: list[dict],
) -> tuple[list[tuple[int, int]], int]:
    """Find the xy coordinates of regions with high density of
    localizations, likely originating from fiducial markers.

    Uses ``picasso.localize.identify_in_image`` with threshold set to
    99th percentile of the image histogram. The image is rendered using
    one-pixel-blur, see picasso.render.render.

    Parameters
    ----------
    locs : np.recarray
        Localizations.
    info : list of dicts
        Localizations' metadata (from the corresponding .yaml file).

    Returns
    -------
    picks : list of (2,) tuples
        Coordinates of fiducial markers. Each list element corresponds
        to (x, y) coordinates of one fiducial marker.
    box : int
        Size of the box used for the fiducial marker identification.
        Can be set as the pick diameter in pixels for undrifting.
    """
    image = render.render(
        locs=locs,
        info=info,
        oversampling=1,
        viewport=None,
        blur_method="smooth",
    )[1]
    # hist = np.histogram(image.flatten(), bins=256)
    threshold = np.percentile(image.flatten(), 99)
    # box size should be an odd number, corresponding to approximately
    # 900 nm
    pixelsize = 130
    for inf in info:
        if val := inf.get("Pixelsize"):
            pixelsize = val
            break
    box = int(np.round(900 / pixelsize))
    box = box + 1 if box % 2 == 0 else box

    # find the local maxima and translate to pick coordinates
    y, x, _ = localize.identify_in_image(image, threshold, box=box)
    picks = [(xi, yi) for xi, yi in zip(x, y)]

    # select the picks with appropriate number of localizations
    n_frames = 0
    for inf in info:
        if val := inf.get("Frames"):
            n_frames = val
            break
    min_n = 0.8 * n_frames
    picked_locs = postprocess.picked_locs(
        locs, info, picks, "Circle", pick_size=box/2, add_group=False,
    )
    picks = [
        pick for i, pick in enumerate(picks) if len(picked_locs[i]) > min_n
    ]
    return picks, box
