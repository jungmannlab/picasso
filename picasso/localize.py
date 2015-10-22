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
def spot_map(frame, roi, abs_gradient):
    """ Returns the sum of the absolute gradient within a ROI around each pixel """
    X, Y = frame.shape
    spot_map = np.zeros_like(abs_gradient)
    roi_half = int(roi/2)
    for i in range(roi, X-roi):
        for j in range(roi, Y-roi):
            local_gradient = abs_gradient[i-roi_half:i+roi_half+1, j-roi_half:j+roi_half+1]
            spot_map[i, j] = np.sum(local_gradient)
    return spot_map


def identify_frame(frame, roi, threshold):
    gradient_x, gradient_y = np.gradient(np.float32(frame))
    abs_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    s_map = spot_map(frame, roi, abs_gradient)
    lm_map = local_maxima_map(frame, roi)
    s_map_thesholded = s_map > threshold
    combined_map = (lm_map * s_map_thesholded) > 0.5
    return np.vstack(np.where(combined_map)).T


def identify(movie, roi, threshold):
    identifications = [identify_frame(frame, roi, threshold) for frame in movie]
    return identifications
