"""
    picasso.imageprocess
    ~~~~~~~~~~~~~~~~~~~~

    Image processing functions

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) 2016 Jungmann Lab, MPI of Biochemistry
"""
import matplotlib.pyplot as _plt
import numpy as _np
from numpy import fft as _fft
import lmfit as _lmfit
from tqdm import tqdm as _tqdm
from . import lib as _lib
from . import render as _render
from . import localize as _localize
from . import postprocess as _postprocess

_plt.style.use("ggplot")


def xcorr(imageA, imageB):
    FimageA = _fft.fft2(imageA)
    CFimageB = _np.conj(_fft.fft2(imageB))
    return _fft.fftshift(_np.real(_fft.ifft2((FimageA * CFimageB)))) / _np.sqrt(
        imageA.size
    )


def get_image_shift(imageA, imageB, box, roi=None, display=False):
    """Computes the shift from imageA to imageB"""
    if (_np.sum(imageA) == 0) or (_np.sum(imageB) == 0):
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
    y, x = _np.mgrid[-fit_X : fit_X + 1, -fit_X : fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = _np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X : y_max_ + fit_X + 1,
        x_max_ - fit_X : x_max_ + fit_X + 1,
    ]

    dimensions = FitROI.shape

    if 0 in dimensions or dimensions[0] != dimensions[1]:
        xc, yc = 0, 0
    else:
        # The fit model
        def flat_2d_gaussian(a, xc, yc, s, b):
            A = a * _np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s**2) + b
            return A.flatten()

        gaussian2d = _lmfit.Model(
            flat_2d_gaussian, name="2D Gaussian", independent_vars=[]
        )

        # Set up initial parameters and fit
        params = _lmfit.Parameters()
        params.add("a", value=FitROI.max(), vary=True, min=0)
        params.add("xc", value=0, vary=True)
        params.add("yc", value=0, vary=True)
        params.add("s", value=1, vary=True, min=0)
        params.add("b", value=FitROI.min(), vary=True, min=0)
        results = gaussian2d.fit(FitROI.flatten(), params)

        # Get maximum coordinates and add offsets
        xc = results.best_values["xc"]
        yc = results.best_values["yc"]
        xc += X_ + x_max_
        yc += Y_ + y_max_

        if display:
            _plt.figure(figsize=(17, 10))
            _plt.subplot(1, 3, 1)
            _plt.imshow(imageA, interpolation="none")
            _plt.subplot(1, 3, 2)
            _plt.imshow(imageB, interpolation="none")
            _plt.subplot(1, 3, 3)
            _plt.imshow(XCorr, interpolation="none")
            _plt.plot(xc, yc, "x")
            _plt.show()

        xc -= _np.floor(X / 2)
        yc -= _np.floor(Y / 2)

    return -yc, -xc


def rcc(segments, max_shift=None, callback=None):
    n_segments = len(segments)
    shifts_x = _np.zeros((n_segments, n_segments))
    shifts_y = _np.zeros((n_segments, n_segments))
    n_pairs = int(n_segments * (n_segments - 1) / 2)
    flag = 0
    if callback is None:
        with _tqdm(
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
        
    return _lib.minimize_shifts(shifts_x, shifts_y)


def find_fiducials(locs, info):
    """Finds the xy coordinates of regions with high density of 
    localizations, likely originating from fiducial markers. 

    Uses picasso.localize.identify_in_image with threshold set to 99th
    percentile of the image histogram. The image is rendered using 
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

    image = _render.render(
        locs=locs,
        info=info,
        oversampling=1,
        viewport=None,
        blur_method="smooth",        
    )[1]
    # hist = _np.histogram(image.flatten(), bins=256)
    threshold = _np.percentile(image.flatten(), 99)
    # box size should be an odd number, corresponding to approximately
    # 900 nm 
    pixelsize = 130
    for inf in info:
        if val := inf.get("Pixelsize"):
            pixelsize = val
            break
    box = int(_np.round(900 / pixelsize))
    box = box + 1 if box % 2 == 0 else box

    # find the local maxima and translate to pick coordinates
    y, x, _ = _localize.identify_in_image(image, threshold, box=box)
    picks = [(xi, yi) for xi, yi in zip(x, y)]

    # select the picks with appropriate number of localizations
    n_frames = 0
    for inf in info:
        if val := inf.get("Frames"):
            n_frames = val
            break
    min_n = 0.8 * n_frames
    picked_locs = _postprocess.picked_locs(
        locs, info, picks, "Circle", pick_size=box/2, add_group=False,
    )
    picks = [
        pick for i, pick in enumerate(picks) if len(picked_locs[i]) > min_n
    ]
    return picks, box