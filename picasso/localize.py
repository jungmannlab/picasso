"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in an frame sequence

    :author: Joerg Schnitzbauer, 2015
"""

import numpy as np
import numba


@numba.jit(nopython=True)
def local_maxima_map(frame, roi):
    """ Finds pixels with maximum value within a region of interest """
    X, Y = frame.shape
    maxima_map = np.zeros_like(frame)
    roi_half = int(roi/2)
    for i in range(roi, X-roi):
        for j in range(roi, Y-roi):
            local_frame = frame[i-roi_half:i+roi_half+1, j-roi_half:j+roi_half+1]
            flat_max = np.argmax(local_frame)
            j_local_max = int(flat_max / roi)
            i_local_max = int(flat_max % roi)
            if (i_local_max == roi_half) and (j_local_max == roi_half):
                maxima_map[i, j] = 1
    return maxima_map


@numba.jit(nopython=True)
def local_gradient_magnitude(frame, roi, abs_gradient):
    """ Returns the sum of the absolute gradient within a ROI around each pixel """
    X, Y = frame.shape
    lgm = np.zeros_like(abs_gradient)
    roi_half = int(roi/2)
    for i in range(roi, X-roi):
        for j in range(roi, Y-roi):
            local_gradient = abs_gradient[i-roi_half:i+roi_half+1, j-roi_half:j+roi_half+1]
            lgm[i, j] = np.sum(local_gradient)
    return lgm


def identify_frame(frame, parameters):
    gradient_x, gradient_y = np.gradient(np.float32(frame))
    abs_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    roi = parameters['roi']
    s_map = local_gradient_magnitude(frame, roi, abs_gradient)
    lm_map = local_maxima_map(frame, roi)
    s_map_thesholded = s_map > parameters['threshold']
    combined_map = (lm_map * s_map_thesholded) > 0.5
    return np.vstack(np.where(combined_map)).T


def identify(movie, parameters):
    identifications = [identify_frame(frame, parameters) for frame in movie]
    return identifications
