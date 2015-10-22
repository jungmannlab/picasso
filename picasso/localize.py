"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in an image sequence

    :author: Joerg Schnitzbauer, 2015
"""

import numpy as np
import numba


@numba.jit(nopython=True)
def local_maxima_map(image, roi_size):
    """ Finds pixels with maximum value within a region of interest """
    X, Y = image.shape
    maxima_map = np.zeros_like(image)
    roi_half_size = int(np.floor(roi_size/2))
    for i in range(roi_half_size+1, X-roi_half_size-2):
        for j in range(roi_half_size+1, Y-roi_half_size-2):
            local_image = image[i-roi_half_size:i+roi_half_size+1, j-roi_half_size:j+roi_half_size+1]
            flat_max = np.argmax(local_image)
            j_local_max = int(flat_max / roi_size)
            i_local_max = int(flat_max % roi_size)
            if (i_local_max == roi_half_size) and (j_local_max == roi_half_size):
                maxima_map[i, j] = 1
    return maxima_map


def local_maxima(image, roi_size):
    maxima_map = local_maxima_map(image, roi_size)
    x, y = np.where(maxima_map)
    return np.vstack((x, y)).T


@numba.jit(nopython=True)
def find_peaks(image, roi_size, abs_gradient):
    X, Y = image.shape
    spot_map = np.zeros_like(abs_gradient)
    roi_size_half = np.int(np.floor(roi_size/2))
    for i in range(roi_size_half+1, X-roi_size_half-2):
        for j in range(roi_size_half+1, Y-roi_size_half-2):
            local_gradient = abs_gradient[i-roi_size_half:i+roi_size_half+1, j-roi_size_half:j+roi_size_half+1]
            spot_map[i, j] = np.sum(local_gradient)
    return spot_map


@numba.jit(nopython=True)
def argmax2d_subset(image, candidate_coordinates):
    max_x = candidate_coordinates[0][0]
    max_y = candidate_coordinates[0][1]
    max = image[max_x, max_y]
    n_candidates = candidate_coordinates.shape[0]
    for i in range(1, n_candidates):
        x_candidate = candidate_coordinates[i, 0]
        y_candidate = candidate_coordinates[i, 1]
        if image[x_candidate, y_candidate] > max:
            max = image[x_candidate, y_candidate]
            max_x = x_candidate
            max_y = y_candidate
    return max_x, max_y


def identify(image, roi_size, threshold):
    gradient_x, gradient_y = np.gradient(np.float32(image))
    abs_gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    spot_map = find_peaks(image, roi_size, abs_gradient)
    lm_map = local_maxima_map(image, roi_size)
    spot_map_thesholded = spot_map > threshold
    combined_map = (lm_map * spot_map_thesholded) > 0.5
    return np.vstack(np.where(combined_map)).T
    """
    labeled_spot_map, n_labels = label(spot_map > threshold)
    x = np.zeros(n_labels, dtype='uint16')
    y = np.zeros(n_labels, dtype='uint16')
    for i in range(n_labels):
        spot_coordinates = np.vstack(np.where(labeled_spot_map == i)).T
        x[i], y[i] = argmax2d_subset(image, spot_coordinates)
    return np.vstack((x, y)).T, spot_map
    """


def localize(files, parameters):
    print('Localizer started')
