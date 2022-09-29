"""
    picasso.nanotron
    ~~~~~~~~~~

    Deep learning library for classification of picked localizations

    :author: Alexander Auer, Maximilian Strauss 2020
    :copyright: Copyright (c) 2020 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from scipy import ndimage

from . import render, lib


def prepare_img(img, img_shape, alpha=1, bg=0):

    img = alpha * img - bg
    img = img.astype('float')
    img = img / img.max()
    img = img.clip(min=0)
    img = img.reshape(img_shape**2)

    return img


def rotate_img(img, angle):

    rot_img = ndimage.rotate(img, angle, reshape=False)

    return rot_img

def roi_to_img(locs, pick, radius, oversampling, picks=None):

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
    except:
        pass
    return pick_img


def prepare_data(locs, label, pick_radius,
                 oversampling, alpha=10,
                 bg=1, export=False):

    img_shape = int(2 * pick_radius * oversampling)
    data = []
    labels = []

    for pick in tqdm(range(locs.group.max()), desc='Prepare '+str(label)):

        pick_img = roi_to_img(locs, pick,
                              radius=pick_radius,
                              oversampling=oversampling)

        if export is True and pick < 10:
            filename = 'label' + str(label) + '-' + str(pick)
            plt.imsave('./img/' + filename + '.png', (alpha*pick_img-bg), cmap='Greys', vmax=10)

        pick_img = prepare_img(pick_img, img_shape=img_shape, alpha=alpha, bg=bg)

        data.append(pick_img)
        labels.append(label)

    return data, label


def predict_structure(mlp, locs, pick, pick_radius, oversampling, picks=None):

    img_shape = int(2 * pick_radius * oversampling)
    img = roi_to_img(locs, pick=pick, radius=pick_radius, oversampling=oversampling, picks=picks)
    img = prepare_img(img, img_shape=img_shape, alpha=10, bg=1)
    img = img.reshape(1, img_shape**2)

    pred = mlp.predict(img)
    pred_proba = mlp.predict_proba(img)

    return pred, pred_proba