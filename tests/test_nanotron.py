"""``picasso.nanotron.roi_to_img`` renders one pick, lets the errors of
rendering through and says when a pick holds no localizations.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import nanotron

INFO = [{"Width": 32, "Height": 32, "Frames": 50, "Pixelsize": 130}]


def _locs(groups=(0,)) -> pd.DataFrame:
    """50 localizations per group, the groups 3 camera pixels apart."""
    rng = np.random.default_rng(0)
    n = 50
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "frame": np.arange(n),
                    "x": rng.normal(5 + 3 * group, 0.1, n),
                    "y": rng.normal(5, 0.1, n),
                    "lpx": np.full(n, 0.05),
                    "lpy": np.full(n, 0.05),
                    "group": np.full(n, group),
                }
            )
            for group in groups
        ],
        ignore_index=True,
    )


def _image(locs, **kwargs):
    return nanotron.roi_to_img(
        locs, INFO, radius=1.0, disp_px_size=13, **kwargs
    )


def test_renders_a_pick():
    image = _image(_locs(), pick=0)
    assert image.shape == (20, 20)
    assert image.sum() > 0


def test_render_errors_are_not_hidden(monkeypatch):
    # they used to end in an UnboundLocalError about ``pick_img``
    def broken(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(nanotron.render, "render", broken)
    with pytest.raises(RuntimeError, match="render failed"):
        _image(_locs(), pick=0)


@pytest.mark.parametrize(
    "kwargs, where",
    [
        ({"pick": 7}, "group 7"),
        ({"pick": 0, "picks": (20.0, 20.0)}, "around (20.0, 20.0)"),
    ],
)
def test_an_empty_pick_is_named(kwargs, where):
    # instead of a 0 x 0 image that fails later
    with pytest.raises(ValueError, match="No localizations in") as error:
        _image(_locs(), **kwargs)
    assert where in str(error.value)


def test_prepare_data_covers_every_group():
    # a gap in the group numbers, and the last group
    data, labels = nanotron.prepare_data(
        _locs(groups=(0, 2, 3)),
        INFO,
        label=1,
        pick_radius=1.0,
        disp_px_size=13,
    )
    assert len(data) == 3
    assert labels == [1, 1, 1]
