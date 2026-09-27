"""
picasso.scmos
~~~~~~~~~~~~~

Per-pixel sCMOS camera characterization: offset, readout variance and
amplification gain maps, following Huang et al. (2013).

An sCMOS sensor has no global readout characteristic. Every pixel carries its
own offset, its own amplification gain and its own readout noise variance, and
those variances span several to thousands of ADU^2 on the same chip. A
localization algorithm that assumes one scalar baseline and one scalar
sensitivity - which is the right model for an EMCCD - therefore reads a noisy
pixel's excursions as signal and pulls localizations towards it, producing
reconstruction artifacts that correlate with the variance map rather than with
the sample.

This module measures the three maps the sCMOS noise model needs. They are
consumed by ``picasso.localize`` (per-pixel photon conversion and the
``picasso.fitting`` noise model), stored by
``picasso.io.save_camera_calibration`` and reloaded by
``picasso.io.load_camera_calibration``.

Two acquisitions feed it:

- A **dark movie** (lens cap on, or a dark room) gives the offset as the
  temporal mean and the readout variance as the temporal variance of each
  pixel. This is the only required input.
- Optionally, a **bright series**: several movies at different quasi-uniform
  illumination levels (e.g., laser powers). Because a pixel's output
  mean is ``g * u + o`` and its output variance is ``g^2 * u + var``,
  the pair ``(mean - o, variance - var)`` traces a photon-transfer curve
  whose slope is the gain, recovered by a least-squares fit through the
  origin. Without a bright series the gain map is absent and the caller
  falls back to the scalar ``Sensitivity``.

All three maps are stored raw and camera-native - offset in ADU, variance in
ADU squared, gain in ADU per photoelectron - so a calibration does not depend
on any Picasso setting and can be reused across analyses. The conversion into
photoelectrons happens where the spots are cut, in ``picasso.localize``.

A calibration is only valid for the camera settings it was acquired with. Bit
depth, readout rate and any selectable gain setting may change the maps, and
so does a large change in sensor temperature - Huang, et al. report that
switching from fan to liquid cooling, around 30 kelvin, was enough to
invalidate theirs. :func:`validate_calibration` checks a calibration against a
short fresh dark movie for exactly this reason.

References
----------
Huang, F., Hartwich, T. M. P., Rivera-Molina, F. E., Lin, Y., Duim, W. C.,
Long, J. J., Uchil, P. D., Myers, J. R., Baird, M. A., Mothes, W.,
Davidson, M. W., Toomre, D. & Bewersdorf, J. "Video-rate nanoscopy using
sCMOS camera-specific single-molecule localization algorithms."
Nature Methods 10, 653-658 (2013). DOI: 10.1038/nmeth.2488.
Sections 2.1-2.3 (characterization) and 5 (reliability test) of the
Supplementary Note.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import warnings
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.stats import chi2
from tqdm import tqdm

from picasso import lib
from picasso.version import __version__

# Below this many dark frames a calibration is refused outright: the variance
# estimate is worth less than the scalar it would replace.
MIN_DARK_FRAMES = 100

# Below this many dark frames a calibration is computed but warned about. The
# relative standard error of a variance estimate is sqrt(2 / (M - 1)): 14% at
# 100 frames, 4.5% at 1,000, 1.4% here, and 0.58% at the 60,000 frames Huang
# et al. used. The variance enters the noise model as a known constant, so its
# error propagates straight into the fit.
RECOMMENDED_DARK_FRAMES = 10_000

# Frames used by the reliability test in the paper. Only a default.
VALIDATION_FRAMES = 1_000

# A pixel is called hot when its readout variance exceeds this multiple of the
# chip median. Reported in the metadata as a headline number; nothing in the
# analysis branches on it, since the noise model handles every pixel on its own
# terms rather than by classifying it.
HOT_PIXEL_FACTOR = 10.0

# Working-set budget for one chunk of frames, in bytes. The accumulator runs
# in float64, so the chunk length is derived from the frame size rather than
# fixed: 256 MiB is seven frames of a 2048 x 2048 sensor but two thousand
# frames of a 128 x 128 crop. A fixed frame count would allocate 3.2 GB on the
# former.
_CHUNK_BYTES = 256 << 20

_MODEL = "scmos-noise"


def _chunk_length(height: int, width: int) -> int:
    """Number of frames per accumulation chunk, from :data:`_CHUNK_BYTES`."""
    return max(1, int(_CHUNK_BYTES // (8 * max(1, height * width))))


def _frame_stack(movie, start: int, stop: int) -> lib.FloatArray3D:
    """Frames ``[start, stop)`` of ``movie`` as one float64 array.

    Plain arrays and memmaps slice directly; every other movie type is read
    frame by frame, which is what ``AbstractPicassoMovie`` guarantees and what
    ``picasso.localize._cut_spots_framebyframe`` already relies on."""
    if isinstance(movie, np.ndarray):
        return np.asarray(movie[start:stop], dtype=np.float64)
    return np.stack(
        [np.asarray(movie[i], dtype=np.float64) for i in range(start, stop)]
    )


def _streaming_moments(
    movie,
    *,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[int, lib.FloatArray2D, lib.FloatArray2D] | None:
    """Per-pixel temporal mean and variance of ``movie``, in one pass.

    Implements Eqs. 2.1 and 2.2 of the Supplementary Note, but **not** in the
    form Eq. 2.2 is written for numerical stability.

    Parameters
    ----------
    movie : array-like or io.AbstractPicassoMovie
        Any Picasso movie. Indexed by frame.
    progress_callback : callable, "console" or None, optional
        Called with the cumulative number of frames processed. ``"console"``
        draws a tqdm bar.
    abort_callback : callable or None, optional
        Polled once per chunk; returning True abandons the pass and returns
        None.
    progress_offset, progress_total : int, optional
        Bias and total for the progress report, so several calls can share one
        bar when a whole calibration is being computed.

    Returns
    -------
    (n_frames, mean, variance) or None
        ``mean`` and ``variance`` are float64 ``(height, width)``, the variance
        with ``ddof=1``. None if the pass was aborted.
    """
    n_frames = int(len(movie))
    if n_frames == 0:
        raise ValueError("Cannot characterize a camera from an empty movie.")
    first = np.asarray(movie[0])
    if first.ndim != 2:
        raise ValueError(
            "Expected a movie of 2D frames, got a frame with shape "
            f"{first.shape}."
        )
    height, width = first.shape

    count = 0
    mean = np.zeros((height, width), dtype=np.float64)
    m2 = np.zeros((height, width), dtype=np.float64)

    use_tqdm = progress_callback == "console"
    pbar = (
        tqdm(
            total=progress_total if progress_total else n_frames,
            initial=progress_offset,
            desc="Characterizing",
            unit="frame",
        )
        if use_tqdm
        else None
    )
    try:
        step = _chunk_length(height, width)
        for start in range(0, n_frames, step):
            if abort_callback is not None and abort_callback():
                return None
            stop = min(start + step, n_frames)
            chunk = _frame_stack(movie, start, stop)
            n_c = chunk.shape[0]
            mean_c = chunk.mean(axis=0)
            m2_c = ((chunk - mean_c) ** 2).sum(axis=0)
            # Chan, Golub & LeVeque (1983), the pairwise-merge update.
            total = count + n_c
            delta = mean_c - mean
            m2 += m2_c + delta * delta * (count * n_c / total)
            mean += delta * (n_c / total)
            count = total
            if use_tqdm:
                pbar.update(n_c)
            elif callable(progress_callback):
                progress_callback(progress_offset + count)
    finally:
        if use_tqdm:
            pbar.close()

    if count < 2:
        raise ValueError(
            f"Need at least 2 frames to estimate a variance, got {count}."
        )
    return count, mean, m2 / (count - 1)


def _gain_map(
    offset: lib.FloatArray2D,
    variance: lib.FloatArray2D,
    means: list[lib.FloatArray2D],
    variances: list[lib.FloatArray2D],
) -> tuple[lib.FloatArray2D, int]:
    """Per-pixel gain from a photon-transfer curve (Eqs. 2.3-2.5).

    A pixel exposed to ``u`` photoelectrons outputs a mean of ``g * u + o`` and
    a variance of ``g^2 * u + var``, so across illumination levels ``k`` the
    points ``(mean_k - o, variance_k - var)`` lie on a line through the origin
    of slope ``g``. Eq. 2.5's pseudo-inverse is, for a scalar gain per pixel,
    the ordinary least-squares slope through the origin.

    Returns the gain map in ADU per photoelectron and the number of pixels
    whose slope was indeterminate and had to be filled with the chip median -
    a dead pixel, or one that saw no light at any level.
    """
    b = np.stack(means) - offset  # (levels, height, width)
    a = np.stack(variances) - variance
    denominator = (b * b).sum(axis=0)
    numerator = (a * b).sum(axis=0)
    # A pixel that never responded has no slope to estimate. Guard before the
    # divide rather than cleaning up NaNs afterwards, so the fallback count is
    # exact.
    usable = denominator > 0.0
    gain = np.ones_like(denominator)
    np.divide(numerator, denominator, out=gain, where=usable)
    # A non-positive slope is as indeterminate as a missing one: gain is a
    # physical amplification and the conversion divides by it.
    usable &= gain > 0.0
    n_fallback = int((~usable).sum())
    if usable.any():
        gain[~usable] = float(np.median(gain[usable]))
    else:
        raise ValueError(
            "Could not estimate the gain of a single pixel. The bright movies "
            "must be recorded at several different, non-zero illumination "
            "levels, with the same camera settings as the dark movie."
        )
    return gain, n_fallback


def _summary(calibration: dict) -> dict:
    """Headline statistics stored alongside the maps."""
    offset, variance = calibration["offset"], calibration["variance"]
    threshold = HOT_PIXEL_FACTOR * float(np.median(variance))
    summary = {
        "Offset median (ADU)": float(np.median(offset)),
        "Offset min (ADU)": float(offset.min()),
        "Offset max (ADU)": float(offset.max()),
        "Variance median (ADU^2)": float(np.median(variance)),
        "Variance min (ADU^2)": float(variance.min()),
        "Variance max (ADU^2)": float(variance.max()),
        "Variance 99.9 percentile (ADU^2)": float(
            np.percentile(variance, 99.9)
        ),
        "Hot pixel threshold (ADU^2)": threshold,
        "Hot pixels": int((variance > threshold).sum()),
    }
    gain = calibration.get("gain")
    if gain is not None:
        summary["Gain median (ADU/e-)"] = float(np.median(gain))
        summary["Gain min (ADU/e-)"] = float(gain.min())
        summary["Gain max (ADU/e-)"] = float(gain.max())
    return summary


def calibrate_scmos(
    dark_movie,
    bright_movies: Sequence | None = None,
    *,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
    dark_path: str | None = None,
    bright_paths: Sequence[str] | None = None,
    bright_levels: Sequence[float] | None = None,
    level_unit: str = "",
) -> dict | None:
    """Characterize an sCMOS camera from a dark movie and an optional
    bright series.

    Parameters
    ----------
    dark_movie : array-like or io.AbstractPicassoMovie
        Frames recorded with no light on the sensor. Huang et al. used 60,000;
        fewer than :data:`RECOMMENDED_DARK_FRAMES` warns and fewer than
        :data:`MIN_DARK_FRAMES` raises.
    bright_movies : sequence of movies, optional
        Movies at several different quasi-uniform illumination levels, all with
        the same camera settings as ``dark_movie``. Huang et al. used 15 levels
        of 20,000 frames spanning roughly 20 to 200 photons per pixel. Omitting
        them omits the gain map, and the caller then keeps using the scalar
        ``Sensitivity``.
    progress_callback : callable, "console" or None, optional
        Called with the cumulative number of frames processed across every
        movie. ``"console"`` draws a tqdm bar.
    abort_callback : callable or None, optional
        Polled once per chunk; returning True abandons the calibration and
        returns None.
    dark_path, bright_paths : str or sequence of str, optional
        Recorded in the metadata so a calibration says where it came from.
    bright_levels : sequence of float, optional
        The illumination each bright movie was recorded at - laser power,
        exposure time, whatever was varied - one per movie, in the same
        order. Nothing in the gain fit needs them: the photon-transfer curve
        is traced by the measured signal itself. They are recorded so the
        diagnostic plot can show the response against the quantity that was
        actually set, which is the only way to see linearity when the levels
        are not evenly spaced.
    level_unit : str, optional
        Unit of ``bright_levels``, for the axis label (e.g. ``"mW"``).

    Returns
    -------
    calibration : dict or None
        Maps under ``"offset"`` (ADU), ``"variance"`` (ADU squared) and, when a
        bright series was given, ``"gain"`` (ADU per photoelectron), plus
        metadata. A bright series also adds one entry per illumination level
        to ``"Level median signal (ADU)"``, ``"Level median variance
        (ADU^2)"`` and ``"Level frames"`` - the chip medians of
        ``mean_k - offset`` and ``variance_k - variance``, in the order the
        movies were given, plus ``"Level illumination"`` and ``"Level unit"``
        when ``bright_levels`` was given, which is what
        :func:`save_calibration_plot` draws its linearity panel from. None if
        the calibration was aborted.

    Raises
    ------
    ValueError
        If the dark movie is too short, if a bright movie's frame size does
        not match the dark movie's, or if ``bright_levels`` has a different
        length than ``bright_movies``.
    """
    bright_movies = list(bright_movies) if bright_movies is not None else []
    if bright_levels is not None and len(bright_levels) != len(bright_movies):
        raise ValueError(
            f"Got {len(bright_levels)} illumination levels for "
            f"{len(bright_movies)} bright movies. Pass one level per movie, "
            "in the same order, or none at all."
        )
    total_frames = int(len(dark_movie)) + sum(
        int(len(m)) for m in bright_movies
    )

    moments = _streaming_moments(
        dark_movie,
        progress_callback=progress_callback,
        abort_callback=abort_callback,
        progress_total=total_frames,
    )
    if moments is None:
        return None
    n_dark, offset, variance = moments

    if n_dark < MIN_DARK_FRAMES:
        raise ValueError(
            f"A camera calibration needs at least {MIN_DARK_FRAMES} dark "
            f"frames, got {n_dark}. The relative uncertainty of a variance "
            "estimate is sqrt(2 / (M - 1)), so at this length the map would "
            "be noisier than the constant it replaces."
        )
    if n_dark < RECOMMENDED_DARK_FRAMES:
        warnings.warn(
            f"Only {n_dark} dark frames: the readout variance is estimated to "
            f"about {100 * np.sqrt(2 / (n_dark - 1)):.1f}% relative "
            "uncertainty per pixel. Huang et al. (2013) used 60,000 frames, "
            f"for 0.6%; {RECOMMENDED_DARK_FRAMES} frames give about 1.4%.",
            RuntimeWarning,
            stacklevel=2,
        )

    height, width = offset.shape
    calibration = {
        "offset": np.ascontiguousarray(offset, dtype=np.float32),
        "variance": np.ascontiguousarray(variance, dtype=np.float32),
        "model": _MODEL,
        "Generated by": f"Picasso v{__version__} sCMOS Calibration",
        "Height": int(height),
        "Width": int(width),
        "Frames": int(n_dark),
        "Dark movie": (os.path.basename(dark_path) if dark_path else "N/A"),
        "Gain levels": 0,
        "Gain frames": 0,
        "Light movies": [],
        "Gain fallback pixels": 0,
        "Path": "N/A",
    }

    if bright_movies:
        means, variances, n_bright = [], [], 0
        level_signal, level_variance, level_frames = [], [], []
        done = n_dark
        for index, movie in enumerate(bright_movies):
            moments = _streaming_moments(
                movie,
                progress_callback=progress_callback,
                abort_callback=abort_callback,
                progress_offset=done,
                progress_total=total_frames,
            )
            if moments is None:
                return None
            n_k, mean_k, var_k = moments
            if mean_k.shape != offset.shape:
                raise ValueError(
                    f"Bright movie {index} has {mean_k.shape[0]}x"
                    f"{mean_k.shape[1]} frames but the dark movie has "
                    f"{height}x{width}. Every movie in a camera calibration "
                    "must be recorded with the same camera ROI and binning."
                )
            means.append(mean_k)
            variances.append(var_k)
            # The medians of this level, kept before the maps are collapsed
            # into a single slope. They are the only record of *how* the
            # sensor responded level by level, and the linearity panel of the
            # diagnostic plot is drawn from them.
            level_signal.append(float(np.median(mean_k - offset)))
            level_variance.append(float(np.median(var_k - variance)))
            level_frames.append(int(n_k))
            n_bright += n_k
            done += n_k
        if len(means) < 2:
            warnings.warn(
                "The gain map was estimated from a single illumination level. "
                "Huang et al. (2013) used 15 levels spanning roughly 20 to "
                "200 photons per pixel; one level cannot separate a genuine "
                "slope from a bad variance estimate.",
                RuntimeWarning,
                stacklevel=2,
            )
        gain, n_fallback = _gain_map(offset, variance, means, variances)
        calibration["gain"] = np.ascontiguousarray(gain, dtype=np.float32)
        calibration["Gain levels"] = len(means)
        calibration["Gain frames"] = int(n_bright)
        calibration["Gain fallback pixels"] = n_fallback
        calibration["Light movies"] = [
            os.path.basename(p) for p in (bright_paths or [])
        ]
        calibration["Level median signal (ADU)"] = level_signal
        calibration["Level median variance (ADU^2)"] = level_variance
        calibration["Level frames"] = level_frames
        calibration["Level illumination"] = (
            [float(value) for value in bright_levels]
            if bright_levels is not None
            else []
        )
        calibration["Level unit"] = level_unit

    calibration.update(_summary(calibration))
    return calibration


def validate_calibration(
    calibration: dict,
    test_movie,
    *,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    abort_callback: Callable[[], bool] | None = None,
) -> dict | None:
    """Check a calibration against a short fresh dark movie (Eq. 5.1).

    If the camera still behaves as it did when it was characterized, each
    pixel's test frames are a fresh sample from the Gaussian described by its
    stored offset and variance, so the sum of squared standardized residuals
    follows a chi-square distribution with ``K`` degrees of freedom and the
    resulting per-pixel p-values are uniform on ``[0, 1]``. A drift in any
    global parameter - most often sensor temperature - pushes the p-value
    distribution towards one tail, so the *mean* p-value moving away from 0.5
    is the signal.

    A thousand frames are plenty; this is not a re-characterization.

    Parameters
    ----------
    calibration : dict
        A calibration from :func:`calibrate_scmos` or
        ``picasso.io.load_camera_calibration``.
    test_movie : array-like or io.AbstractPicassoMovie
        A short, fresh dark movie recorded exactly as the calibration's was.
    progress_callback : callable, "console" or None, optional
        Called with the cumulative number of frames processed. ``"console"``
        draws a tqdm bar. As in :func:`calibrate_scmos`.
    abort_callback : callable or None, optional
        Polled once per chunk; returning True abandons the check and returns
        None.

    Returns
    -------
    report : dict or None
        ``"mean p-value"``, ``"valid"`` (True when the mean is within 0.1 of
        0.5), ``"Frames"``, and the fraction of pixels in each tail. None if
        aborted.

    Raises
    ------
    ValueError
        If the test movie's frame size does not match the calibration.
    """
    offset = np.asarray(calibration["offset"], dtype=np.float64)
    variance = np.asarray(calibration["variance"], dtype=np.float64)

    n_frames = int(len(test_movie))
    first = np.asarray(test_movie[0])
    if first.shape != offset.shape:
        raise ValueError(
            f"The test movie is {first.shape[0]}x{first.shape[1]} but the "
            f"calibration is {offset.shape[0]}x{offset.shape[1]}. Record the "
            "test movie with the same camera ROI and binning."
        )

    # Sum of squared standardized residuals, streamed the same way the
    # calibration itself is. Pixels with a non-positive stored variance cannot
    # be standardized and are excluded rather than allowed to divide by zero.
    usable = variance > 0.0
    statistic = np.zeros_like(offset)
    safe_variance = np.where(usable, variance, 1.0)
    height, width = offset.shape
    step = _chunk_length(height, width)
    use_tqdm = progress_callback == "console"
    pbar = (
        tqdm(total=n_frames, desc="Validating", unit="frame")
        if use_tqdm
        else None
    )
    try:
        for start in range(0, n_frames, step):
            if abort_callback is not None and abort_callback():
                return None
            stop = min(start + step, n_frames)
            chunk = _frame_stack(test_movie, start, stop)
            statistic += ((chunk - offset) ** 2 / safe_variance).sum(axis=0)
            if use_tqdm:
                pbar.update(stop - start)
            elif callable(progress_callback):
                progress_callback(stop)
    finally:
        if use_tqdm:
            pbar.close()

    p_values = chi2.sf(statistic[usable], n_frames)
    mean_p = float(p_values.mean()) if p_values.size else float("nan")
    return {
        "Frames": n_frames,
        "Pixels tested": int(usable.sum()),
        "mean p-value": mean_p,
        "valid": bool(abs(mean_p - 0.5) <= 0.1),
        "fraction p < 0.05": (
            float((p_values < 0.05).mean()) if p_values.size else float("nan")
        ),
        "fraction p > 0.95": (
            float((p_values > 0.95).mean()) if p_values.size else float("nan")
        ),
    }


def _plot_map(figure, ax, values, key, title, unit) -> None:
    """One map panel, on a color scale that shows the sensor rather than
    its outliers."""
    if key == "variance":
        vmin, vmax = 0.0, float(np.percentile(values, 99.5))
    else:
        vmin = float(np.percentile(values, 0.5))
        vmax = float(np.percentile(values, 99.5))
    if not vmax > vmin:
        # A uniform map (a simulation, or a hand-made calibration) collapses
        # every percentile onto the same value.
        vmin, vmax = float(values.min()), float(values.max())
        if not vmax > vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
    image = ax.imshow(
        values, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    ax.set_title(f"{title} map")
    ax.set_xticks([])
    ax.set_yticks([])
    figure.colorbar(image, ax=ax, label=unit, fraction=0.046)


def _plot_histogram(ax, values, title, unit) -> None:
    """One histogram panel, on a log count axis so the tail stays visible."""
    flat = values.ravel()
    lower, upper = float(flat.min()), float(np.percentile(flat, 99.9))
    if not upper > lower:
        upper = float(flat.max())
    if not upper > lower:
        upper = lower + 1.0
    ax.hist(flat, bins=100, range=(lower, upper), color="k")
    ax.set_yscale("log")
    ax.set_xlabel(f"{title} ({unit})")
    ax.set_ylabel("Pixels")
    ax.set_title(f"{title} distribution")


def _plot_linearity(
    ax,
    signal: list[float],
    frames: list[int],
    illumination: list[float] | None = None,
    unit: str = "",
) -> None:
    """Median signal against illumination.

    The gain fit assumes every pixel responds linearly over the range the
    bright series covers; a level that saturates, or a laser that is not
    linear in its setting, breaks that assumption and biases the slope
    without making the fit fail.

    Plotted against the illumination that was actually set, when the caller
    recorded it. Without it there is only the order the movies were given, and
    an index axis silently assumes the levels were evenly spaced - a series at
    0.1, 1, 5 and 10 mW then bends into a curve that says nothing about the
    camera. The axis label says which of the two is being shown, and the
    straight-line fit is only quoted for the first.
    """
    values = np.asarray(signal, dtype=np.float64)
    spaced = illumination is not None and len(illumination) == len(values)
    if spaced:
        x = np.asarray(illumination, dtype=np.float64)
        order = np.argsort(x)
        x, values = x[order], values[order]
        ax.set_xlabel(f"Illumination ({unit})" if unit else "Illumination")
    else:
        x = np.arange(1, len(values) + 1, dtype=np.float64)
        ax.set_xlabel("Illumination level (spacing unknown)")
        ax.set_xticks(x)
    ax.plot(x, values, "o", color="k", markersize=4)

    if spaced and len(values) > 2 and np.ptp(x) > 0:
        # A free line rather than one through the origin: a bright series
        # often carries some stray background, and an intercept is not a
        # nonlinearity.
        slope, intercept = np.polyfit(x, values, 1)
        fitted = slope * x + intercept
        ax.plot(x, fitted, "--", color="tab:red", linewidth=1)
        span = float(np.ptp(values))
        if span > 0:
            worst = 100 * float(np.abs(values - fitted).max()) / span
            ax.set_title(f"Linearity (max {worst:.1f}% off the fit)")
        else:
            ax.set_title("Linearity")
    else:
        # Joining the points is a reading aid, not a claim about the spacing.
        ax.plot(x, values, "-", color="0.6", linewidth=1)
        ax.set_title("Response per level")
    ax.set_ylabel("Median signal, mean $-$ offset (ADU)")
    if frames:
        ax.text(
            0.02,
            0.98,
            f"{min(frames)}-{max(frames)} frames per level",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="0.4",
        )


def _plot_transfer(
    ax, signal: list[float], variance: list[float], gain: float
) -> None:
    """The photon-transfer curve the gain is the slope of.

    Signal and excess variance are both linear in the number of
    photoelectrons, so the median points lie on a line through the origin of
    slope ``gain`` whatever the illumination levels were - unlike the
    linearity panel, this one needs no assumption about their spacing.
    """
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(variance, dtype=np.float64)
    ax.plot(x, y, "o", color="k", markersize=4)
    upper = float(max(x.max(), 0.0))
    if upper > 0:
        line = np.array([0.0, upper])
        ax.plot(
            line,
            gain * line,
            "-",
            color="tab:red",
            linewidth=1,
            label=f"median gain {gain:.3f} ADU/e$^-$",
        )
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("Median signal (ADU)")
    ax.set_ylabel("Median excess variance (ADU$^2$)")
    ax.set_title("Photon transfer curve")


def plot_path(calibration_path: str) -> str:
    """Where the diagnostic plot of a calibration file belongs.

    Parameters
    ----------
    calibration_path : str
        Path of the saved camera calibration.

    Returns
    -------
    path : str
        The same path with its extension replaced by ``_maps.png``.
    """
    base, _ = os.path.splitext(calibration_path)
    return base + "_maps.png"


def save_calibration_plot(calibration: dict, path: str) -> str:
    """Write the maps and their histograms to a PNG, as Supplementary Fig. 1
    of Huang et al. (2013).

    Maps and histograms answer different questions and both are worth having.
    The map shows *structure* - the column stripes of the per-column
    amplifiers in the gain, a bright corner in the offset, a cluster of hot
    pixels - which a histogram averages away. The histogram shows the
    *distribution*, in particular the tail of high-variance pixels the noise
    model exists for, which a map on a linear color scale cannot resolve
    against its own outliers.

    A bright series of more than one level adds a third question, and a row
    for it: whether the sensor's response was linear over the range the gain
    was fitted on. The left panel plots the chip median signal against the
    illumination it was recorded at, when the calibration knows it, and
    otherwise level by level. The right one plots the photon-transfer curve
    those levels trace, with the median gain as its slope; because it has the
    measured signal on both axes it is straight whatever the illumination
    levels were, and is the panel to read when they were not evenly spaced.

    Parameters
    ----------
    calibration : dict
        A calibration as returned by :func:`calibrate_scmos` or
        ``io.load_camera_calibration``.
    path : str
        Where to write the PNG. Use :func:`plot_path` to derive it from the
        calibration's own path.

    Returns
    -------
    str
        ``path``, for convenience when reporting where it went.
    """
    # Imported here, and through the non-interactive Agg backend, so that
    # importing this module costs nothing and so that writing the file never
    # depends on a display being available - this runs from the CLI too.
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    panels = [
        ("variance", "Readout variance", "ADU$^2$"),
        ("offset", "Offset", "ADU"),
    ]
    if calibration.get("gain") is not None:
        panels.append(("gain", "Gain", "ADU/e$^-$"))

    signal = list(calibration.get("Level median signal (ADU)") or [])
    level_variance = list(
        calibration.get("Level median variance (ADU^2)") or []
    )
    # One level is a point, not a curve; there is nothing to show about how
    # the sensor responds across illumination.
    with_levels = len(signal) > 1 and len(level_variance) == len(signal)

    rows = len(panels) + (1 if with_levels else 0)
    figure = Figure(figsize=(9, 3 * rows), tight_layout=True)
    axes = figure.subplots(rows, 2, squeeze=False)
    for row, (key, title, unit) in enumerate(panels):
        values = np.asarray(calibration[key], dtype=np.float64)
        _plot_map(figure, axes[row, 0], values, key, title, unit)
        _plot_histogram(axes[row, 1], values, title, unit)
    if with_levels:
        gain = calibration.get("gain")
        median_gain = (
            float(np.median(np.asarray(gain, dtype=np.float64)))
            if gain is not None
            else 0.0
        )
        _plot_linearity(
            axes[-1, 0],
            signal,
            list(calibration.get("Level frames") or []),
            list(calibration.get("Level illumination") or []) or None,
            str(calibration.get("Level unit") or ""),
        )
        _plot_transfer(axes[-1, 1], signal, level_variance, median_gain)
    figure.savefig(path, dpi=150)
    return path
