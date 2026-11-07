"""
picasso.imageprocess
~~~~~~~~~~~~~~~~~~~~

Image processing functions

:authors: Joerg Schnitzbauer, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft as sfft
from scipy.optimize import curve_fit
from scipy.signal import windows

from . import lib, render, localize, postprocess

__all__ = ["find_fiducials", "get_image_shift", "rcc", "decorrelation"]


plt.style.use("ggplot")


def xcorr(
    imageA: lib.FloatArray2D, imageB: lib.FloatArray2D
) -> lib.FloatArray2D:
    """Compute the cross-correlation of two images using FFT.

    Parameters
    ----------
    imageA, imageB : lib.FloatArray2D
        Input images to be cross-correlated. They should have the same
        shape.

    Returns
    -------
    res : lib.FloatArray2D
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
    imageA: lib.FloatArray2D,
    imageB: lib.FloatArray2D,
    box: int,
    roi: int | None = None,
    display: bool = False,
) -> tuple[float, float]:
    """Compute the shift from ``imageA`` to ``imageB``.

    Parameters
    ----------
    imageA, imageB : lib.FloatArray2D
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
    y, x = np.mgrid[-fit_X : fit_X + 1, -fit_X : fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X : y_max_ + fit_X + 1,
        x_max_ - fit_X : x_max_ + fit_X + 1,
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
            flat_2d_gaussian,
            (x, y),
            FitROI.flatten(),
            p0=p0,
            bounds=bounds,
        )

        # Get maximum coordinates and add offsets
        xc = popt[1]
        yc = popt[2]
        xc += X_ + x_max_
        yc += Y_ + y_max_

        if display:
            plt.figure(figsize=(17, 10), constrained_layout=True)
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
    segments: list[lib.FloatArray2D],
    max_shift: float | None = None,
    progress: lib.Progress | Literal["console"] | None = None,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Compute RCC, see Wang, Schnitzbauer, et al. Optics Express,
    2014. Return the shift of each segment in y and x relative to the
    first segment.

    Parameters
    ----------
    segments : list of lib.FloatArray2D
        List of image segments to be correlated. Each segment should be
        a 2D numpy array representing an image.
    max_shift : float, optional
        Maximum allowed shift in pixels. If None (default), the shift
        is not limited.
    progress : lib.Progress, "console" or None, optional
        Progress tracker, see ``picasso.lib.Progress``: e.g. a
        ``lib.ProgressDialog``, "console" for a tqdm bar in the console,
        or None for no reporting. Reports "Correlating image pairs" over
        the ``n_segments * (n_segments - 1) / 2`` segment pairs. Default
        is None.

    Returns
    -------
    shift_y, shift_x : lib.FloatArray1D
        Shifts of each segment in y and x (in pixels of the segment
        images), shape (n_segments,), found by least-squares
        minimization of the pairwise shifts (see
        ``lib.minimize_shifts``). The first segment is the reference,
        i.e., ``shift_y[0] == shift_x[0] == 0``.
    """
    n_segments = len(segments)
    shifts_x = np.zeros((n_segments, n_segments))
    shifts_y = np.zeros((n_segments, n_segments))
    n_pairs = n_segments * (n_segments - 1) // 2
    with lib.tracked_progress(
        progress, "Correlating image pairs", unit="pair"
    ) as progress:
        lib.start_phase(
            progress, "Correlating image pairs", n_pairs, unit="pair"
        )
        progress.set_value(0)
        n_done = 0
        for i in range(n_segments - 1):
            for j in range(i + 1, n_segments):
                shifts_y[i, j], shifts_x[i, j] = get_image_shift(
                    segments[i], segments[j], 5, max_shift
                )
                n_done += 1
                progress.set_value(n_done)
        progress.set_value(n_pairs)

    return lib.minimize_shifts(shifts_x, shifts_y)


#: Pixel size (nm) of the image rendered by :func:`find_fiducials`.
FIDUCIAL_RENDER_PIXEL_SIZE = 100.0
#: Approximate side length (nm) of the box used to identify fiducials.
FIDUCIAL_BOX_SIZE = 900.0


def find_fiducials(
    locs: pd.DataFrame,
    info: list[dict],
) -> tuple[list[tuple[float, float]], float]:
    """Find the xy coordinates of regions with high density of
    localizations, likely originating from fiducial markers.

    Renders the localizations at a pixel size of 100 nm (one-pixel
    blur, see ``picasso.render.render``) and uses
    ``picasso.localize.identify_in_image`` with the threshold set to the
    99th percentile of the image histogram. The identified image pixels
    are mapped back to nm (pixel centers).

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Localizations' metadata (from the corresponding .yaml file).

    Returns
    -------
    picks : list of (2,) tuples
        Coordinates of fiducial markers in nm. Each list element
        corresponds to (x, y) coordinates of one fiducial marker.
    box : float
        Side length (nm) of the box used for the fiducial marker
        identification (about 900 nm). Can be used as the pick diameter
        for undrifting.
    """
    disp_px_size = FIDUCIAL_RENDER_PIXEL_SIZE
    image = render.render(
        locs=locs,
        info=info,
        disp_px_size=disp_px_size,
        viewport=None,
        blur_method="smooth",
    )[1]
    threshold = np.percentile(image.flatten(), 99)
    # box size (rendered pixels) should be an odd number, corresponding
    # to approximately 900 nm
    box_px = int(np.round(FIDUCIAL_BOX_SIZE / disp_px_size))
    box_px = box_px + 1 if box_px % 2 == 0 else box_px
    box = box_px * disp_px_size

    # find the local maxima and translate to pick coordinates (nm); the
    # image starts at the origin of the field of view
    y, x, _ = localize.identify_in_image(image, threshold, box=box_px)
    picks = [
        ((xi + 0.5) * disp_px_size, (yi + 0.5) * disp_px_size)
        for xi, yi in zip(x, y)
    ]

    # select the picks with appropriate number of localizations
    n_frames = lib.get_from_metadata(info, "Frames", default=0)
    min_n = 0.8 * n_frames
    picked_locs = postprocess.picked_locs(
        locs,
        info,
        picks=picks,
        pick_shape="Circle",
        pick_size=box,
        add_group=False,
    )
    picks = [
        pick for i, pick in enumerate(picks) if len(picked_locs[i]) > min_n
    ]
    return picks, box


def quick_fiducials_check(
    locs: pd.DataFrame,
    info: list[dict],
    fraction: float = 0.8,
    region_size: float = 1000.0,
) -> int:
    """Quickly estimate whether fiducial markers are likely present.

    Unlike ``find_fiducials``, this does not render an image or run spot
    identification. It bins localizations onto a coarse 2D grid (one bin
    per ``region_size`` nm) and counts the bins that contain more than
    ``fraction * n_frames`` localizations. Such dense, persistent regions
    are characteristic of fiducial markers (e.g. gold beads), which are
    present in nearly every frame, whereas blinking emitters are not.

    This is a cheap heuristic intended as a fast pre-check. It may miss
    fiducials split across bin boundaries; use ``find_fiducials`` for an
    accurate localization of the markers.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations.
    info : list of dicts
        Localizations' metadata (from the corresponding .yaml file).
    fraction : float, optional
        Fraction of the total number of frames a region must exceed (in
        localization count) to be flagged as a likely fiducial. Default
        is 0.8.
    region_size : float, optional
        Approximate bin size in nm. Default is 1,000 nm.

    Returns
    -------
    n_regions : int
        Number of grid regions flagged as likely fiducial markers. A
        value greater than zero suggests fiducials are present.
    """
    if len(locs) == 0:
        return 0
    n_frames = lib.get_from_metadata(info, "Frames", raise_error=True)

    x = np.asarray(locs["x"])
    y = np.asarray(locs["y"])
    n_bins_x = max(1, int(np.ceil((x.max() - x.min()) / region_size)))
    n_bins_y = max(1, int(np.ceil((y.max() - y.min()) / region_size)))

    hist, _, _ = np.histogram2d(x, y, bins=[n_bins_x, n_bins_y])
    return int(np.count_nonzero(hist > fraction * n_frames))


def radial_sum(image: lib.FloatArray2D) -> lib.FloatArray1D:
    """Compute the radial projection of the sum of pixel values.

    If the radial distance of a pixel to center is r, then the sum of the
    intensities of all pixels with n <= r < (n + 1) is stored at
    position n in the output array.

    Parameters
    ----------
    image : lib.FloatArray2D
        Input image array.

    Returns
    -------
    radial_profile : lib.FloatArray1D
        1D array containing the radial sum values.

    Raises
    ------
    ValueError
        If ``max_radius`` is not "inner_radius" or "outer_radius".
    """
    assert image.ndim == 2, "Input image must be 2D."
    assert image.shape[0] == image.shape[1], "Input image must be square."
    assert image.shape[0] % 2 == 1, "Input image size must be odd."

    size = image.shape[0]
    center = size // 2
    # ring index n = floor(r) satisfies n**2 <= r**2 < (n + 1)**2; sqrt
    # of an integer perfect square is exact, so the rings are exact too
    y, x = np.ogrid[:size, :size]
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    ring = np.floor(np.sqrt(dist_sq)).astype(np.intp).ravel()
    keep = ring <= center
    ring = ring[keep]
    values = image.ravel()[keep]
    n_bins = center + 1
    if np.iscomplexobj(values):
        counts = np.bincount(
            ring, weights=values.real, minlength=n_bins
        ) + 1j * np.bincount(ring, weights=values.imag, minlength=n_bins)
    else:
        counts = np.bincount(ring, weights=values, minlength=n_bins)
    return counts.astype(image.dtype, copy=False)


#: Peaks of the decorrelation functions lower than this are noise and
#: ignored, see ``decorrelation``
_DECORRELATION_MIN_AMPLITUDE = 0.05
#: A peak of a decorrelation function must exceed all later values by
#: more than this, so that it is not numerical noise on a flat curve
_DECORRELATION_PEAK_TOLERANCE = 1e-3
#: Width (pixels) of the cosine taper at the image edges
_DECORRELATION_EDGE = 20
#: Widest high-pass filter, in normalized frequency (1 = Nyquist)
_DECORRELATION_MAX_FILTER = 3.0
#: Smallest image side (pixels) for the decorrelation analysis
DECORRELATION_MIN_SIZE = 32


class _Spectrum(NamedTuple):
    """Radial sums of the Fourier transform of an image in bins of
    normalized frequency (1 = Nyquist) of width ``width``, inside the
    unit circle: of the amplitude, the squared amplitude and the number
    of nonzero coefficients; ``centers`` are the bin centers."""

    amplitude: lib.FloatArray1D
    power: lib.FloatArray1D
    count: lib.FloatArray1D
    centers: lib.FloatArray1D
    width: float


def _apodize(image: lib.FloatArray2D, edge: int) -> lib.FloatArray2D:
    """``image`` with its outer ``edge`` pixels blended to its mean by a
    cosine taper (Tukey window), so that the edges do not add high
    frequencies."""
    ny, nx = image.shape
    window_y = windows.tukey(ny, min(1.0, 2 * edge / ny))
    window_x = windows.tukey(nx, min(1.0, 2 * edge / nx))
    mean = image.mean()
    return (image - mean) * np.outer(window_y, window_x) + mean


def _decorrelation_spectrum(image: lib.FloatArray2D) -> _Spectrum:
    """Radial sums of the Fourier transform of ``image``, see
    ``_Spectrum``. The transform of a real image is symmetric, so only
    half of it is computed and the other half counted twice."""
    ny, nx = image.shape
    amplitude = np.abs(sfft.rfft2(image.astype(np.float32), workers=-1))
    # columns of the half-plane that stand for two coefficients
    weight = np.full(amplitude.shape[1], 2.0, dtype=np.float32)
    weight[0] = 1.0
    if nx % 2 == 0:
        weight[-1] = 1.0
    radius = 2 * np.sqrt(
        sfft.fftfreq(ny)[:, None] ** 2 + sfft.rfftfreq(nx)[None, :] ** 2
    )
    n_bins = max(ny, nx)
    width = 1.0 / n_bins
    inside = radius < 1
    bins = np.minimum((radius[inside] / width).astype(np.intp), n_bins - 1)
    weights = np.broadcast_to(weight, amplitude.shape)[inside]
    values = amplitude[inside].astype(np.float64)
    return _Spectrum(
        np.bincount(bins, values * weights, n_bins),
        np.bincount(bins, values**2 * weights, n_bins),
        np.bincount(bins, (values > 0) * weights, n_bins),
        (np.arange(n_bins) + 0.5) * width,
        width,
    )


def _decorrelation_curves(
    spectrum: _Spectrum,
    radii: lib.FloatArray1D,
    filters: list[float | None],
) -> lib.FloatArray2D:
    """Decorrelation function at ``radii`` of the image (filter None)
    and of the image high-pass filtered by each width of ``filters``,
    shape (len(filters), len(radii)).

    The image's transform I correlated with its normalized transform
    I / |I| within a radius r sums to the sum of |I| within r, and the
    normalized transform has unit amplitude, so each curve follows from
    the radial sums of the spectrum."""
    index = np.clip(np.round(radii / spectrum.width).astype(np.intp), 0, None)
    index = np.minimum(index, len(spectrum.amplitude))
    count = np.concatenate([[0.0], np.cumsum(spectrum.count)])[index]
    curves = np.zeros((len(filters), len(radii)))
    for i, sigma in enumerate(filters):
        if sigma is None:
            high_pass = np.ones_like(spectrum.centers)
        else:
            high_pass = -np.expm1(-(spectrum.centers**2) / (2 * sigma**2))
        within = np.concatenate(
            [[0.0], np.cumsum(spectrum.amplitude * high_pass)]
        )[index]
        energy = np.sum(spectrum.power * high_pass**2)
        with np.errstate(invalid="ignore", divide="ignore"):
            curves[i] = within / np.sqrt(energy * count)
    return np.nan_to_num(curves, nan=0.0, posinf=0.0)


def _decorrelation_peaks(
    curves: lib.FloatArray2D,
    radii: lib.FloatArray1D,
    min_radius: float,
) -> tuple[lib.FloatArray1D, lib.FloatArray1D]:
    """Position and amplitude of the peak of each curve, NaN for curves
    without one: the highest value that exceeds all later values by
    more than ``_DECORRELATION_PEAK_TOLERANCE``, if it is at least
    ``_DECORRELATION_MIN_AMPLITUDE`` and at a radius of at least
    ``min_radius``."""
    positions = np.full(len(curves), np.nan)
    amplitudes = np.full(len(curves), np.nan)
    for i, curve in enumerate(curves):
        # the lowest value after each point
        after = np.append(np.minimum.accumulate(curve[::-1])[::-1][1:], np.inf)
        peaks = np.flatnonzero(
            (curve - after > _DECORRELATION_PEAK_TOLERANCE)
            & (radii >= min_radius)
        )
        if not len(peaks):
            continue
        peak = peaks[np.argmax(curve[peaks])]
        if curve[peak] >= _DECORRELATION_MIN_AMPLITUDE:
            positions[i] = radii[peak]
            amplitudes[i] = curve[peak]
    return positions, amplitudes


def decorrelation(
    image: lib.FloatArray2D,
    *,
    pixel_size: float = 1.0,
    n_radii: int = 50,
    n_filters: int = 10,
) -> dict:
    """Resolution of an image by decorrelation analysis (Descloux et
    al., 2019), without parameters to tune.

    The Fourier transform of the image, ``I(k)``, is normalized to
    unit amplitude, ``In(k) = I(k) / |I(k)|``, which weighs the
    frequencies dominated by noise as much as those with signal. The
    decorrelation function::

        d(r) = sum_(|k| < r) Re(I(k) In*(k))
               / sqrt(sum_k |I(k)|^2 * sum_(|k| < r) |In(k)|^2)

    correlates the image with its normalized transform, low-pass
    filtered at the frequency r (normalized: 1 is the Nyquist
    frequency, half the sampling frequency). It peaks where higher
    frequencies add more noise than signal. The analysis is repeated
    on the image high-pass filtered by inverted Gaussians, ``I(k) * (1
    - exp(-|k|^2 / (2 sigma^2)))``, with ``n_filters`` widths sigma
    spaced geometrically from the lowest frequency of the image to 3,
    which
    moves the peak towards the highest frequency with signal. The
    cut-off frequency kc is the highest peak position of all curves,
    and the resolution is its period, 2 * ``pixel_size`` / kc.

    Here, the edges of the image are first blended to its mean over 20
    pixels (cosine taper), so that they add no high frequencies. A peak
    must exceed the rest of its curve by more than 0.001, and peaks
    lower than 0.05 or at frequencies below two radius steps (from the
    mean of the image) are ignored. The curves are computed at
    ``n_radii`` radii from 0 to 1; around the highest peak, they are
    computed again at ``n_radii`` radii and ``n_filters`` widths between
    the neighbors of the best filter, which refines kc.

    Parameters
    ----------
    image : lib.FloatArray2D
        Image, at least 32 x 32 pixels.
    pixel_size : float, optional
        Size of a pixel, which sets the unit of the resolution, e.g., in
        nm. Default is 1 (resolution in pixels).
    n_radii : int, optional
        Number of radii from 0 to 1 (Nyquist), at least 10. Default is
        50.
    n_filters : int, optional
        Number of high-pass filters, at least 2. Default is 10.

    Returns
    -------
    result : dict
        Dictionary with the keys:

        - "resolution": resolution, 2 * ``pixel_size`` / kc, in the
          unit of ``pixel_size``; NaN if no curve has a peak, e.g., if
          the image has signal up to the Nyquist frequency (pixels too
          large) or no signal.
        - "cutoff": kc, normalized to the Nyquist frequency (NaN
          without a peak).
        - "amplitude": height of the peak of the unfiltered curve, which
          grows with the signal-to-noise ratio (NaN without a peak).
        - "radii": radii of the curves (normalized frequencies), shape
          (n_radii,), and "frequencies": the same in cycles per unit of
          ``pixel_size``.
        - "curves": decorrelation function without high-pass filter
          (first row) and with each filter, shape (n_filters + 1,
          n_radii).
        - "filters": widths sigma of the high-pass filters
          (normalized frequencies).
        - "peak_radii", "peak_amplitudes": position and height of the
          peak of each curve, NaN for curves without a peak.
        - "pixel_size".

    Raises
    ------
    ValueError
        If the image is not 2D or smaller than 32 x 32 pixels, or the
        parameters are invalid.

    See Also
    --------
    picasso.postprocess.decorrelation : Resolution of localizations.

    References
    ----------
    Descloux, A., Grussmayer, K. S., Radenovic, A. Parameter-free
    image resolution estimation based on decorrelation analysis. Nat
    Methods 16, 918-924 (2019).
    https://doi.org/10.1038/s41592-019-0515-7

    Examples
    --------
    >>> result = imageprocess.decorrelation(image, pixel_size=65.0)
    >>> print(f"{result['resolution']:.0f} nm")
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"The image must be 2D, not {image.ndim}D.")
    if min(image.shape) < DECORRELATION_MIN_SIZE:
        raise ValueError(
            f"The image must be at least {DECORRELATION_MIN_SIZE} x "
            f"{DECORRELATION_MIN_SIZE} pixels, not {image.shape}."
        )
    if not pixel_size > 0:
        raise ValueError("pixel_size must be positive.")
    if n_radii < 10:
        raise ValueError("n_radii must be at least 10.")
    if n_filters < 2:
        raise ValueError("n_filters must be at least 2.")
    spectrum = _decorrelation_spectrum(_apodize(image, _DECORRELATION_EDGE))
    lowest = 2.0 / max(image.shape)  # lowest frequency of the image
    filters = np.geomspace(lowest, _DECORRELATION_MAX_FILTER, n_filters)
    radii = np.linspace(0.0, 1.0, n_radii)
    step = radii[1]
    curves = _decorrelation_curves(spectrum, radii, [None, *filters])
    peak_radii, peak_amplitudes = _decorrelation_peaks(curves, radii, 2 * step)
    cutoff = np.nan
    if np.isfinite(peak_radii).any():
        cutoff = np.nanmax(peak_radii)
        # refine around the best curve; the unfiltered curve is next to
        # a filter one step weaker than the weakest
        ratio = filters[1] / filters[0]
        widths = np.concatenate([[lowest / ratio], filters])
        best = int(np.nanargmax(peak_radii))
        fine_filters = np.geomspace(
            widths[max(best - 1, 0)],
            widths[min(best + 1, n_filters)],
            n_filters,
        )
        fine_radii = np.linspace(
            max(cutoff - step, 0.0), min(cutoff + 10 * step, 1.0), n_radii
        )
        fine_peaks, _ = _decorrelation_peaks(
            _decorrelation_curves(spectrum, fine_radii, list(fine_filters)),
            fine_radii,
            2 * step,
        )
        if np.isfinite(fine_peaks).any():
            cutoff = max(cutoff, np.nanmax(fine_peaks))
    return {
        "resolution": 2 * pixel_size / cutoff,
        "cutoff": cutoff,
        "amplitude": peak_amplitudes[0],
        "radii": radii,
        "frequencies": radii / (2 * pixel_size),
        "curves": curves,
        "filters": filters,
        "peak_radii": peak_radii,
        "peak_amplitudes": peak_amplitudes,
        "pixel_size": pixel_size,
    }
