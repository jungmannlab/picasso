"""Synthetic localizations for the Render GUI tests.

- ``WIDTH`` / ``HEIGHT`` / ``PIXELSIZE``: the camera field of view
  (pixels) and pixel size (nm); ``WIDTH_NM`` / ``HEIGHT_NM`` give the
  field of view in nm.
- ``make_locs``: uniformly distributed localizations over that field of
  view.
- ``make_info``: the matching metadata.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WIDTH = HEIGHT = 64.0  # camera pixels
PIXELSIZE = 130.0
WIDTH_NM = WIDTH * PIXELSIZE
HEIGHT_NM = HEIGHT * PIXELSIZE


def make_locs(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """``n`` localizations spread uniformly over the field of view."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, WIDTH_NM, size=n),
            "y": rng.uniform(0.0, HEIGHT_NM, size=n),
            "lpx": rng.uniform(0.05, 0.3, size=n) * PIXELSIZE,
            "lpy": rng.uniform(0.05, 0.3, size=n) * PIXELSIZE,
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


def make_info() -> list[dict]:
    """Metadata matching ``make_locs``."""
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]
