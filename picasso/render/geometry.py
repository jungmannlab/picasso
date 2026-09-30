"""
picasso.render.geometry
~~~~~~~~~~~~~~~~~~~~~~~

Pure coordinate math: viewport measures and transformations, and 3D
rotation helpers.

:authors: Rafal Kowalewski, Joerg Schnitzbauer
:copyright: Copyright (c) 2015-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .. import lib

if TYPE_CHECKING:
    from PyQt6 import QtGui, QtCore
else:
    # PyQt6 is imported on first attribute access so that importing
    # picasso.render does not require PyQt6.
    QtGui = lib._LazyQtModule("PyQt6.QtGui")
    QtCore = lib._LazyQtModule("PyQt6.QtCore")


def rotation_matrix(angx: float, angy: float, angz: float) -> Rotation:
    """Find rotation matrix given rotation angles around axes.

    Parameters
    ----------
    angx, angy, angz : float
        Rotation angles around x, y and z axes in radians.

    Returns
    -------
    scipy.spatial.transform.Rotation
        Scipy class that can be applied to rotate an Nx3 array.
    """
    rot_mat_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angx), np.sin(angx)],
            [0.0, -np.sin(angx), np.cos(angx)],
        ]
    )  # rotation matrix around x axis
    rot_mat_y = np.array(
        [
            [np.cos(angy), 0.0, np.sin(angy)],
            [0.0, 1.0, 0.0],
            [-np.sin(angy), 0.0, np.cos(angy)],
        ]
    )  # rotation matrix around y axis
    rot_mat_z = np.array(
        [
            [np.cos(angz), -np.sin(angz), 0.0],
            [np.sin(angz), np.cos(angz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )  # rotation matrix around z axis
    rot_mat = rot_mat_x @ rot_mat_y @ rot_mat_z
    return Rotation.from_matrix(rot_mat)


def to_rotation(
    ang: tuple[float, float, float] | Rotation | None,
) -> Rotation | None:
    """Normalize a rotation input to a scipy Rotation.

    Parameters
    ----------
    ang : tuple, Rotation or None
        Rotation to be normalized. Either a scipy Rotation (e.g. built
        from a quaternion; used as is), a tuple of 3 rotation angles
        around the x, y and z axes in radians (legacy Euler convention,
        see ``rotation_matrix``), or None.

    Returns
    -------
    scipy.spatial.transform.Rotation or None
        The rotation as a scipy Rotation, or None if ``ang`` is None.
    """
    if ang is None:
        return None
    if isinstance(ang, Rotation):
        return ang
    return rotation_matrix(*ang)


def closest_rotvec(
    rotation: Rotation,
    reference: lib.FloatArray1D,
) -> lib.FloatArray1D:
    """Find the rotation vector representing ``rotation`` that is
    closest to ``reference``.

    A rotation vector (axis * angle, radians) is not unique: adding
    full turns (2 pi) around the same axis yields the same rotation.
    This function picks the representation closest to ``reference``,
    which allows keeping track of rotations beyond +/- 180 degrees,
    e.g. unwrapping a continuously updated rotation.

    Note that full turns cannot be recovered from an orientation this
    way: near a full turn ``rotation`` is close to the identity and the
    axis of its rotation vector is dominated by whatever small residual
    rotation is left, so the turns held by ``reference`` are dropped.
    Rotation paths that may contain full turns (the displayed angles of
    the rotation window, animation segments) are therefore accumulated
    as a path integral of the applied rotations instead.

    Parameters
    ----------
    rotation : scipy.spatial.transform.Rotation
        The rotation to be represented.
    reference : lib.FloatArray1D
        Rotation vector (radians) to which the result should be
        closest.

    Returns
    -------
    rotvec : lib.FloatArray1D
        Rotation vector (radians) such that
        ``Rotation.from_rotvec(rotvec) == rotation``, with the number
        of full turns chosen to match ``reference``. Its magnitude may
        exceed pi.
    """
    reference = np.asarray(reference, dtype=float)
    base = rotation.as_rotvec()  # magnitude <= pi
    theta = np.linalg.norm(base)
    if theta < 1e-9:
        # identity rotation; keep the full turns along reference's axis
        ref_magnitude = np.linalg.norm(reference)
        if ref_magnitude < 1e-9:
            return np.zeros(3)
        n_turns = np.round(ref_magnitude / (2 * np.pi))
        return reference * (2 * np.pi * n_turns / ref_magnitude)
    axis = base / theta
    n_turns = np.round((axis @ reference - theta) / (2 * np.pi))
    return axis * (theta + 2 * np.pi * n_turns)


def viewport_height(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> float:
    """Calculate viewport height in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    height : float
        Viewport height in camera pixels.
    """
    return viewport[1][0] - viewport[0][0]


def viewport_width(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> float:
    """Calculate viewport width in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    width : float
        Viewport width in camera pixels.
    """
    return viewport[1][1] - viewport[0][1]


def viewport_size(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Calculate viewport size in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    height, width : float
        Viewport height and width in camera pixels.
    """
    height = viewport_height(viewport)
    width = viewport_width(viewport)
    return height, width


def viewport_center(
    viewport: list[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    """Calculate viewport center in camera pixels.

    Parameters
    ----------
    viewport : list of tuples
        Viewport coordinates in camera pixels, [[y_min, y_max], [x_min,
        x_max]].

    Returns
    -------
    center : tuple
        Viewport center coordinates in camera pixels (y, x).
    """
    center = (
        ((viewport[1][0] + viewport[0][0]) / 2),
        ((viewport[1][1] + viewport[0][1]) / 2),
    )
    return center


def shift_viewport(
    viewport: tuple[tuple[float, float], tuple[float, float]],
    dx: float,
    dy: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shift the viewport by the given shift vector (toward the bottom
    right corner).

    Parameters
    ----------
    viewport : tuple
        Current viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    dx, dy : float
        Shifts in camera pixels.

    Returns
    -------
    new_viewport : tuple
        New viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    """
    (ymin, xmin), (ymax, xmax) = viewport
    new_viewport = ((ymin + dy, xmin + dx), (ymax + dy, xmax + dx))
    return new_viewport


def zoom_viewport(
    viewport: tuple[tuple[float, float], tuple[float, float]],
    factor: float,
    cursor_position: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Zoom the viewport by the given factor.

    Parameters
    ----------
    viewport : tuple
        Current viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    factor : float
        Zoom factor. Values > 1 will zoom in, values < 1 will zoom out.
    cursor_position : tuple, optional
        Cursor's position on the screen. If None, zooming is centered
        around viewport's center. Default is None.

    Returns
    -------
    new_viewport : tuple
        New viewport in camera pixels ((ymin, xmin), (ymax, xmax)).
    """
    viewport_height, viewport_width = viewport_size(viewport)
    new_viewport_height = viewport_height * factor
    new_viewport_width = viewport_width * factor

    if cursor_position is not None:  # wheelEvent
        old_viewport_center = viewport_center(viewport)
        rel_pos_x = (
            cursor_position[0] - old_viewport_center[1]
        ) / viewport_width
        rel_pos_y = (
            cursor_position[1] - old_viewport_center[0]
        ) / viewport_height
        new_viewport_center_x = (
            cursor_position[0] - rel_pos_x * new_viewport_width
        )
        new_viewport_center_y = (
            cursor_position[1] - rel_pos_y * new_viewport_height
        )
    else:
        new_viewport_center_y, new_viewport_center_x = viewport_center(
            viewport
        )

    new_viewport = [
        (
            new_viewport_center_y - new_viewport_height / 2,
            new_viewport_center_x - new_viewport_width / 2,
        ),
        (
            new_viewport_center_y + new_viewport_height / 2,
            new_viewport_center_x + new_viewport_width / 2,
        ),
    ]
    return new_viewport


def adjust_viewport_to_aspect_ratio(
    image: QtGui.QImage,
    viewport: list[tuple[float, float], tuple[float, float]],
) -> list[tuple[float, float], tuple[float, float]]:
    """Adjust viewport to match the aspect ratio of the image.

    Parameters
    ----------
    image : QtGui.QImage
        Image of rendered localizations.
    viewport : list of tuples
        Viewport coordinates in camera pixels, ((y_min, y_max), (x_min,
        x_max)).

    Returns
    -------
    viewport : list of tuples
        Adjusted viewport coordinates in camera pixels, ((y_min, y_max),
        (x_min, x_max)).
    """
    viewport_height, viewport_width = viewport_size(viewport)
    view_height = image.height()
    view_width = image.width()
    viewport_aspect = viewport_width / viewport_height
    view_aspect = view_width / view_height
    if view_aspect >= viewport_aspect:
        y_min = viewport[0][0]
        y_max = viewport[1][0]
        x_range = viewport_height * view_aspect
        x_margin = (x_range - viewport_width) / 2
        x_min = viewport[0][1] - x_margin
        x_max = viewport[1][1] + x_margin
    else:
        x_min = viewport[0][1]
        x_max = viewport[1][1]
        y_range = viewport_width / view_aspect
        y_margin = (y_range - viewport_height) / 2
        y_min = viewport[0][0] - y_margin
        y_max = viewport[1][0] + y_margin
    return ((y_min, x_min), (y_max, x_max))


def adjust_viewport_decorator(func):
    """Decorator that adjusts viewport to match image aspect ratio before
    calling the decorated function.

    Note that this assumes image and viewport to be the first two
    arguments of the decorated function

    Parameters
    ----------
    func : callable
        Function that takes `image` and `viewport` as arguments.

    Returns
    -------
    wrapper : callable
        Wrapped function with automatic viewport adjustment.
    """

    def wrapper(image, viewport, *args, **kwargs):
        adjusted_viewport = adjust_viewport_to_aspect_ratio(image, viewport)
        return func(image, adjusted_viewport, *args, **kwargs)

    return wrapper


def map_to_view(
    x: float,
    y: float,
    image_size: QtCore.QSize,
    viewport: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[int, int]:
    """Convert (x, y) from camera pixels to display pixels.

    Parameters
    ----------
    x, y : float
        Coordinates in camera pixels.
    image_size : QtCore.QSize
        Size of the displayed image in display pixels.
    viewport : tuple
        ``((y_min, x_min), (y_max, x_max))`` in camera pixels.

    Returns
    -------
    cx, cy : int
        Coordinates in display pixels.
    """
    image_width = image_size.width()
    image_height = image_size.height()
    cx = image_width * (x - viewport[0][1]) / viewport_width(viewport)
    cy = image_height * (y - viewport[0][0]) / viewport_height(viewport)
    return int(cx), int(cy)
