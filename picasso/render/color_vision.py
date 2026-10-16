"""
picasso.render.color_vision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation of color vision deficiencies, to preview rendered images
as people with color blindness see them.

Each deficiency is a 3×3 matrix applied to linear RGB (Machado,
Oliveira & Fernandes, *IEEE Trans. Vis. Comput. Graph.* 2009, severity
1.0, i.e., dichromacy): protanopia and deuteranopia (no L or M cones;
red-green color blindness) and tritanopia (no S cones; blue-yellow).
The matrices keep grays unchanged. Milder, anomalous forms
(protanomaly, deuteranomaly) lose less of the color differences than
the simulation shows.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from .image_overlay import _as_rgb32, _bgra_view

if TYPE_CHECKING:
    # annotations only: importing picasso.render must not require PyQt6
    from PyQt6 import QtGui

#: Simulation matrices of the color vision deficiencies, applied to
#: linear RGB column vectors (Machado et al. 2009, severity 1.0).
COLOR_VISION_DEFICIENCIES = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        dtype=np.float32,
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        dtype=np.float32,
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ],
        dtype=np.float32,
    ),
}

# number of entries of the table that encodes linear values back to
# 8-bit sRGB; fine enough that dark values (where the sRGB curve is
# steepest) are off by less than one level
_N_ENCODE = 4096


def _srgb_tables() -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables from 8-bit sRGB to linear values (256 entries)
    and from linear values, sampled evenly in [0, 1], back to 8-bit
    sRGB (``_N_ENCODE`` entries)."""
    srgb = np.arange(256) / 255
    to_linear = np.where(
        srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
    )
    linear = np.linspace(0, 1, _N_ENCODE)
    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear ** (1 / 2.4) - 0.055,
    )
    to_srgb = np.round(encoded * 255).astype(np.uint8)
    return to_linear.astype(np.float32), to_srgb


_TO_LINEAR, _TO_SRGB = _srgb_tables()


@numba.njit(cache=True, nogil=True)
def _simulate(
    image: np.ndarray,
    matrix: np.ndarray,
    to_linear: np.ndarray,
    to_srgb: np.ndarray,
    i_r: int,
    i_g: int,
    i_b: int,
) -> None:
    """Apply ``matrix`` in place to the colors of an ``(height, width,
    channels)`` uint8 image whose red, green and blue are the channels
    ``i_r``, ``i_g`` and ``i_b``; other channels (alpha) are kept."""
    n = to_srgb.shape[0] - 1
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = to_linear[image[y, x, i_r]]
            g = to_linear[image[y, x, i_g]]
            b = to_linear[image[y, x, i_b]]
            for c in range(3):
                v = matrix[c, 0] * r + matrix[c, 1] * g + matrix[c, 2] * b
                v = min(max(v, 0.0), 1.0)
                out = to_srgb[int(v * n + 0.5)]
                if c == 0:
                    image[y, x, i_r] = out
                elif c == 1:
                    image[y, x, i_g] = out
                else:
                    image[y, x, i_b] = out


def _matrix(deficiency: str) -> np.ndarray:
    try:
        return COLOR_VISION_DEFICIENCIES[deficiency]
    except KeyError:
        raise ValueError(
            f"Unknown color vision deficiency '{deficiency}'; expected one "
            f"of {', '.join(COLOR_VISION_DEFICIENCIES)}."
        ) from None


def simulate_color_vision(image: np.ndarray, deficiency: str) -> np.ndarray:
    """Simulate how people with a color vision deficiency see an RGB
    image.

    Parameters
    ----------
    image : np.ndarray
        8-bit sRGB image of shape ``(..., 3)``, e.g., ``(height, width,
        3)``.
    deficiency : {"protanopia", "deuteranopia", "tritanopia"}
        The color vision deficiency, see
        ``COLOR_VISION_DEFICIENCIES``.

    Returns
    -------
    simulated : np.ndarray
        The simulated image, uint8 of the same shape as ``image``.

    Raises
    ------
    ValueError
        If ``deficiency`` is unknown or ``image`` does not have 3
        channels in its last axis.
    """
    matrix = _matrix(deficiency)
    image = np.asarray(image)
    if image.ndim < 1 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB image of shape (..., 3), got {image.shape}."
        )
    simulated = np.array(image, dtype=np.uint8, copy=True, order="C")
    flat = simulated.reshape(1, -1, 3)
    _simulate(flat, matrix, _TO_LINEAR, _TO_SRGB, 0, 1, 2)
    return simulated


def simulate_color_vision_qimage(
    image: QtGui.QImage, deficiency: str
) -> QtGui.QImage:
    """Simulate how people with a color vision deficiency see a QImage,
    see ``simulate_color_vision``.

    Parameters
    ----------
    image : QImage
        Image to simulate; it is not changed.
    deficiency : {"protanopia", "deuteranopia", "tritanopia"}
        The color vision deficiency.

    Returns
    -------
    simulated : QImage
        The simulated image, RGB32 or ARGB32 (alpha is kept).
    """
    matrix = _matrix(deficiency)
    simulated = _as_rgb32(image)
    if simulated is image:
        simulated = image.copy()
    # BGRA byte order, see _bgra_view
    _simulate(_bgra_view(simulated), matrix, _TO_LINEAR, _TO_SRGB, 2, 1, 0)
    return simulated
