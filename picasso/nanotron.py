"""
    picasso.nanotron
    ~~~~~~~~~~~~~~~~

    Deep learning library for classification of picked localizations.

    :author: Alexander Auer, Maximilian Strauss 2020
    :copyright: Copyright (c) 2020 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from scipy import ndimage
from sklearn.neural_network import MLPClassifier

from . import render, lib


def prepare_img(
    img: np.ndarray,
    img_shape: int,
    alpha: float = 1,
    bg: float = 0,
) -> np.ndarray:
    """Prepare image for classification.

    Parameters
    ----------
    img : np.ndarray
        Input image to be prepared.
    img_shape : int
        Shape of the image (assumed to be square).
    alpha : float, optional
        Scaling factor for the image, by default 1.
    bg : float, optional
        Background value to be subtracted, by default 0.

    Returns
    -------
    img : np.ndarray
        Prepared image.
    """
    img = alpha * img - bg
    img = img.astype('float')
    img = img / img.max()
    img = img.clip(min=0)
    img = img.reshape(img_shape**2)

    return img


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by a given angle.

    Parameters
    ----------
    img : np.ndarray
        Input image to be rotated.
    angle : float
        Angle in degrees by which to rotate the image.

    Returns
    -------
    rot_img : np.ndarray
        Rotated image.
    """
    rot_img = ndimage.rotate(img, angle, reshape=False)

    return rot_img


def roi_to_img(
    locs: np.recarray,
    pick: int,
    radius: float,
    oversampling: float,
    picks: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert a region of interest (ROI) defined by localizations to an
    image.

    Parameters
    ----------
    locs : np.recarray
        Localizations from which to create the image.
    pick : int
        The group number of the localizations to be picked.
    radius : float
        Radius around the mean position of the localizations to define
        the ROI.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    picks : tuple[float, float] | None, optional
        Specific coordinates (x, y) to isolate localizations from, by
        default None. If None, uses the mean position of the localizations
        in the specified group.

    Returns
    -------
    pick_img : np.ndarray
        Image of the picked localizations.
    """
    # Isolate locs from pick
    pick_locs = []
    if picks is None:
        pick_locs = locs[(locs["group"] == pick)]
    else:
        x, y = picks
        pick_locs = lib.locs_at(x, y, locs, radius)
        pick_locs.sort(kind="mergesort", order="frame")
    # dirty method to avoid floating point errors with render
    radius -= 0.001

    x_mean = np.mean(pick_locs.x)
    y_mean = np.mean(pick_locs.y)

    x_min = x_mean - radius
    x_max = x_mean + radius
    y_min = y_mean - radius
    y_max = y_mean + radius

    viewport = (y_min, x_min), (y_max, x_max)

    # for debugging
    if False:
        print("mean x: {}".format(np.mean(pick_locs.x)))
        print('length x: {}'.format(x_max - x_min))
        print("mean y: {}".format(np.mean(pick_locs.y)))
        print('length y: {}'.format(y_max - y_min))
        print('radius: {}'.format(radius))
        print('viewport: {}'.format(viewport))

    # Render locs with Picasso render function
    try:
        len_x, pick_img = render.render(pick_locs, viewport=viewport,
                                        oversampling=oversampling,
                                        blur_method='smooth')
    except Exception:
        pass
    return pick_img


def prepare_data(
    locs: np.recarray,
    label: int,
    pick_radius: float,
    oversampling: float,
    alpha: float = 10,
    bg: float = 1,
    export: bool = False,
) -> tuple[list[np.ndarray], list[int]]:
    """Prepare data for classification by extracting images of
    localizations.

    Parameters
    ----------
    locs : np.recarray
        Localizations from which to create the images.
    label : int
        Label for the data, typically the group number.
    pick_radius : float
        Radius around the mean position of the localizations to define
        the ROI.
    oversampling : float
        Number of super-resolution pixels per camera pixel.
    alpha : float, optional
        Scaling factor for the image, by default 10.
    bg : float, optional
        Background value to be subtracted, by default 1.
    export : bool, optional
        If True, saves the images to the './img/' directory, by default
        False.

    Returns
    -------
    data : list[np.ndarray]
        List of prepared images of the localizations.
    labels : list[int]
        List of labels corresponding to the images.
    """
    img_shape = int(2 * pick_radius * oversampling)
    data = []
    labels = []

    for pick in tqdm(range(locs.group.max()), desc='Prepare '+str(label)):

        pick_img = roi_to_img(locs, pick,
                              radius=pick_radius,
                              oversampling=oversampling)

        if export is True and pick < 10:
            filename = 'label' + str(label) + '-' + str(pick)
            plt.imsave(
                './img/' + filename + '.png', (alpha*pick_img-bg),
                cmap='Greys',
                vmax=10,
            )

        pick_img = prepare_img(
            pick_img, img_shape=img_shape, alpha=alpha, bg=bg,
        )

        data.append(pick_img)
        labels.append(label)

    return data, label


def predict_structure(
    mlp: MLPClassifier,
    locs: np.recarray,
    pick: int,
    pick_radius: float,
    oversampling: float,
    picks: tuple[float, float] | None = None,
) -> tuple[int, np.ndarray]:
    """Predict the structure of localizations using a trained MLP
    classifier.

    Parameters
    ----------
    mlp : MLPClassifier
        Trained MLP classifier.
    locs : np.recarray
        Localizations to predict.
    pick : int
        Index of the localizations to predict.
    pick_radius : float
        Radius around the localizations to consider.
    oversampling : float
        Oversampling factor for the image.
    picks : tuple[float, float] | None, optional
        Coordinates of the region of interest (ROI) to predict.

    Returns
    -------
    pred : int
        Predicted label for the image.
    pred_proba : np.ndarray
        Predicted probabilities for each class.
    """
    img_shape = int(2 * pick_radius * oversampling)
    img = roi_to_img(
        locs,
        pick=pick,
        radius=pick_radius,
        oversampling=oversampling,
        picks=picks,
    )
    img = prepare_img(img, img_shape=img_shape, alpha=10, bg=1)
    img = img.reshape(1, img_shape**2)

    pred = mlp.predict(img)
    pred_proba = mlp.predict_proba(img)

    return pred, pred_proba
