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


def split(movie, info, frame=0, vmax=0.9, rectangle=None):
    if rectangle is None:
        subregions = []
        shifts = []
        infos = []

        def on_split_select(press_event, release_event):
            x1, y1 = press_event.xdata, press_event.ydata
            x2, y2 = release_event.xdata, release_event.ydata
            xmin = int(min(x1, x2) + 0.5)
            xmax = int(max(x1, x2) + 0.5)
            ymin = int(min(y1, y2) + 0.5)
            ymax = int(max(y1, y2) + 0.5)
            subregions.append(movie[:, ymin:ymax+1, xmin:xmax+1])
            shifts.append((ymin, xmin))
            subregion_info = _copy.deepcopy(info)
            subregion_info[0]['Height'] = ymax - ymin + 1
            subregion_info[0]['Width'] = xmax - xmin + 1
            infos.append(subregion_info)

        f = _plt.figure(figsize=(12, 12))
        ax = f.add_subplot(111)
        ax.matshow(movie[frame], cmap='viridis', vmax=vmax*movie[frame].max())
        ax.grid(False)
        selector = _RectangleSelector(ax, on_split_select, useblit=True, rectprops=dict(edgecolor='red', fill=False))
        _plt.show()
        wavelengths = []
        for i in range(len(shifts)):
            wavelengths.append(int(input('Excitation wavelength for region {}? '.format(i))))
        return subregions, shifts, infos, wavelengths
    else:
        pass


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
