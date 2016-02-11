"""
    picasso/imageprocess
    ~~~~~~~~~~~~~~~~~~~~

    Image processing functions

    :author: Joerg Schnitzbauer, 2016
"""
import matplotlib.pyplot as _plt
from matplotlib.widgets import RectangleSelector as _RectangleSelector
import copy as _copy
import numpy as _np
from numpy import fft as _fft
import lmfit as _lmfit


_plt.style.use('ggplot')


def xcorr(imageA, imageB):
    FimageA = _fft.fft2(imageA)
    CFimageB = _np.conj(_fft.fft2(imageB))
    return _fft.fftshift(_np.real(_fft.ifft2((FimageA * CFimageB)))) / _np.sqrt(imageA.size)


def get_image_shift(imageA, imageB, margin, fit_roi):
    # Compute image correlation
    XCorr = xcorr(imageA, imageB)
    # Cut the margins
    Y, X = imageA.shape
    Y_ = int(margin * Y)
    X_ = int(margin * X)
    XCorr_ = XCorr[Y_:-Y_, X_:-X_]
    # A quarter of the fit ROI
    fit_X = int(fit_roi / 2)
    # A coordinate grid for the fitting ROI
    y, x = _np.mgrid[-fit_X:fit_X+1, -fit_X:fit_X+1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = _np.unravel_index(XCorr_.argmax(), XCorr_.shape)
    FitROI = XCorr[y_max_ - fit_X + Y_:y_max_ + fit_X + Y_ + 1, x_max_ - fit_X + X_:x_max_ + fit_X + X_ + 1]

    # The fit model
    def flat_2d_gaussian(a, xc, yc, s, b):
        A = a * _np.exp(-0.5 * ((x - xc)**2 + (y - yc)**2) / s**2) + b
        return A.flatten()
    gaussian2d = _lmfit.Model(flat_2d_gaussian, name='2D Gaussian', independent_vars=[])

    # Set up initial parameters and fit
    params = _lmfit.Parameters()
    params.add('a', value=FitROI.max(), vary=True, min=0)
    params.add('xc', value=0, vary=True)
    params.add('yc', value=0, vary=True)
    params.add('s', value=1, vary=True, min=0)
    params.add('b', value=FitROI.min(), vary=True, min=0)
    results = gaussian2d.fit(FitROI.flatten(), params)

    # Get maximum coordinates and add offsets
    xc = results.best_values['xc']
    yc = results.best_values['yc']
    xc += X_ + x_max_ - X / 2
    yc += Y_ + y_max_ - Y / 2
    return -yc, -xc
