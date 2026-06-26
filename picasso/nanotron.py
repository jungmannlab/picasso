"""
picasso.nanotron
~~~~~~~~~~~~~~~~

Deep learning library for classification of picked localizations.

:authors: Alexander Auer, Maximilian Strauss
:copyright: Copyright (c) 2020-2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from scipy import ndimage
from sklearn.neural_network import MLPClassifier

from . import render, lib


def prepare_img(
    img: lib.FloatArray2D,
    img_shape: int,
    alpha: float = 1,
    bg: float = 0,
) -> lib.FloatArray1D:
    """Prepare image for classification.

    Parameters
    ----------
    img : lib.FloatArray2D
        Input image to be prepared.
    img_shape : int
        Shape of the image (assumed to be square).
    alpha : float, optional
        Scaling factor for the image, by default 1.
    bg : float, optional
        Background value to be subtracted, by default 0.

    Returns
    -------
    img : lib.FloatArray1D
        Prepared image.
    """
    img = alpha * img - bg
    img = img.astype("float")
    img = img / img.max()
    img = img.clip(min=0)
    img = img.reshape(img_shape**2)

    return img


def rotate_img(img: lib.FloatArray2D, angle: float) -> lib.FloatArray2D:
    """Rotate image by a given angle.

    Parameters
    ----------
    img : lib.FloatArray2D
        Input image to be rotated.
    angle : float
        Angle in degrees by which to rotate the image.

    Returns
    -------
    rot_img : lib.FloatArray2D
        Rotated image.
    """
    rot_img = ndimage.rotate(img, angle, reshape=False)

    return rot_img


def roi_to_img(
    locs: pd.DataFrame,
    info: list[dict],
    pick: int,
    radius: float,
    disp_px_size: float,
    picks: tuple[float, float] | None = None,
) -> lib.FloatArray2D:
    """Convert a region of interest (ROI) defined by localizations to an
    image.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations from which to create the image.
    info : list of dicts
        Localization metadata.
    pick : int
        The group number of the localizations to be picked.
    radius : float
        Radius around the mean position of the localizations to define
        the ROI.
    disp_px_size : float
        Display pixel size used for rendering images (nm).
    picks : tuple[float, float] | None, optional
        Specific coordinates (x, y) to isolate localizations from, by
        default None. If None, uses the mean position of the localizations
        in the specified group.

    Returns
    -------
    pick_img : lib.FloatArray2D
        Image of the picked localizations.
    """
    # Isolate locs from pick
    pick_locs = []
    if picks is None:
        pick_locs = locs[locs["group"] == pick]
    else:
        x, y = picks
        pick_locs = lib.locs_at(x, y, locs, radius)
        pick_locs.sort_values(by="frame", kind="quicksort", inplace=True)
    # dirty method to avoid floating point errors with render
    radius -= 0.001

    x_mean = np.mean(pick_locs["x"])
    y_mean = np.mean(pick_locs["y"])

    x_min = x_mean - radius
    x_max = x_mean + radius
    y_min = y_mean - radius
    y_max = y_mean + radius

    viewport = (y_min, x_min), (y_max, x_max)

    # Render locs with Picasso render function
    try:
        len_x, pick_img = render.render(
            pick_locs,
            info,
            disp_px_size=disp_px_size,
            viewport=viewport,
            blur_method="smooth",
        )
    except Exception:
        pass
    return pick_img


def prepare_data(
    locs: pd.DataFrame,
    info: list[dict],
    label: int,
    pick_radius: float,
    disp_px_size: float,
    alpha: float = 10,
    bg: float = 1,
    export: bool = False,
) -> tuple[list[lib.FloatArray1D], list[int]]:
    """Prepare data for classification by extracting images of
    localizations.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations from which to create the images.
    info : list of dicts
        Localization metadata. Must contain the camera pixel size.
    label : int
        Label for the data, typically the group number.
    pick_radius : float
        Radius around the mean position of the localizations to define
        the ROI.
    disp_px_size : float
        Display pixel size used for rendering images (nm).
    alpha : float, optional
        Scaling factor for the image, by default 10.
    bg : float, optional
        Background value to be subtracted, by default 1.
    export : bool, optional
        If True, saves the images to the './img/' directory, by default
        False.

    Returns
    -------
    data : list[lib.FloatArray1D]
        List of prepared images of the localizations.
    labels : list[int]
        List of labels corresponding to the images.
    """
    data = []
    labels = []

    for pick in tqdm(range(locs["group"].max()), desc="Prepare " + str(label)):

        pick_img = roi_to_img(
            locs, info, pick, radius=pick_radius, disp_px_size=disp_px_size
        )

        if export is True and pick < 10:
            filename = "label" + str(label) + "-" + str(pick)
            plt.imsave(
                "./img/" + filename + ".png",
                (alpha * pick_img - bg),
                cmap="Greys",
                vmax=10,
            )

        # derive the image shape from the rendered ROI so it always
        # matches render's pixel count for the given display pixel size
        img_shape = pick_img.shape[0]
        pick_img = prepare_img(
            pick_img,
            img_shape=img_shape,
            alpha=alpha,
            bg=bg,
        )

        data.append(pick_img)
        labels.append(label)

    return data, labels


def predict_structure(
    mlp: MLPClassifier,
    locs: pd.DataFrame,
    info: list[dict],
    pick: int,
    pick_radius: float,
    disp_px_size: float,
    picks: tuple[float, float] | None = None,
) -> tuple[int, lib.FloatArray1D]:
    """Predict the structure of localizations using a trained MLP
    classifier.

    Parameters
    ----------
    mlp : MLPClassifier
        Trained MLP classifier.
    locs : pd.DataFrame
        Localizations to predict.
    info : list of dicts
        Localization metadata. Must contain the camera pixel size.
    pick : int
        Index of the localizations to predict.
    pick_radius : float
        Radius around the localizations to consider.
    disp_px_size : float
        Display pixel size used for rendering images (nm).
    picks : tuple[float, float] | None, optional
        Coordinates of the region of interest (ROI) to predict.

    Returns
    -------
    pred : int
        Predicted label for the image.
    pred_proba : lib.FloatArray1D
        Predicted probabilities for each class.
    """
    img = roi_to_img(
        locs,
        info,
        pick=pick,
        radius=pick_radius,
        disp_px_size=disp_px_size,
        picks=picks,
    )
    # derive the image shape from the rendered ROI so it always matches
    # render's pixel count for the given display pixel size
    img_shape = img.shape[0]
    img = prepare_img(img, img_shape=img_shape, alpha=10, bg=1)
    img = img.reshape(1, img_shape**2)

    pred = mlp.predict(img)
    pred_proba = mlp.predict_proba(img)

    return pred, pred_proba
