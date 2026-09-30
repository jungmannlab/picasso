"""The adaptively jittered, averaged triangulation
(``blur_method="triangulation"``, Baddeley, Cannell & Soeller 2010) of
``picasso.render``: linearity, mass conservation, the neighbor
distance, determinism, the histogram limit at coarse zoom, rotation,
and the plumbing through ``render_scene``.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Delaunay

from picasso import render
from picasso.render import triangulation as tri


W = H = 64.0


def _info():
    return [{"Width": W, "Height": H, "Frames": 1, "Pixelsize": 130.0}]


def _clustered(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [rng.uniform(0, W, n // 2), rng.normal(20, 1.5, n // 2)]
    )
    y = np.concatenate(
        [rng.uniform(0, H, n // 2), rng.normal(40, 1.5, n // 2)]
    )
    keep = (x > 0) & (x < W) & (y > 0) & (y < H)
    return x[keep].astype(np.float32), y[keep].astype(np.float32)


def test_neighbor_distance_is_the_mean_incident_edge_length():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 10, (200, 2))
    simplices = Delaunay(pts).simplices.astype(np.int32)
    d = tri._mean_edge_lengths(pts[:, 0], pts[:, 1], simplices)
    # brute force for a few vertices: every edge of every incident
    # triangle that touches the vertex, counted once per triangle
    for v in (0, 17, 100):
        lengths = []
        for a, b, c in simplices:
            for p, q in ((a, b), (b, c), (c, a)):
                if v in (p, q):
                    lengths.append(np.linalg.norm(pts[p] - pts[q]))
        assert d[v] == pytest.approx(np.mean(lengths))


@pytest.mark.parametrize("passes", [0, 5])
def test_linear_and_mass_conserving(passes):
    x, y = _clustered()
    viewport = ((0.0, 0.0), (H, W))
    n, image = tri.render_triangulation(x, y, 4.0, viewport, passes=passes)
    assert n == len(x)
    # every triangle carries the same share; the border loses a little
    assert image.sum() == pytest.approx(n, rel=0.03)
    assert image.shape == (256, 256)
    # the dense cluster is the brightest region
    cy, cx = np.unravel_index(np.argmax(image), image.shape)
    assert abs(cx / 4 - 20) < 4 and abs(cy / 4 - 40) < 4


def test_uniform_field_is_flat():
    rng = np.random.default_rng(3)
    n = 30_000
    x = rng.uniform(0, W, n).astype(np.float32)
    y = rng.uniform(0, H, n).astype(np.float32)
    _, image = tri.render_triangulation(
        x, y, 2.0, ((0.0, 0.0), (H, W)), passes=10
    )
    inner = image[16:-16, 16:-16]  # away from the border
    density = n / (W * H) / 4  # per display pixel
    assert inner.mean() == pytest.approx(density, rel=0.03)
    # the jittered average is far smoother than the plain triangulation
    _, plain = tri.render_triangulation(
        x, y, 2.0, ((0.0, 0.0), (H, W)), passes=0
    )
    assert inner.std() < 0.7 * plain[16:-16, 16:-16].std()


def test_seed_makes_it_repeatable():
    x, y = _clustered()
    viewport = ((10.0, 10.0), (50.0, 50.0))
    a = tri.render_triangulation(x, y, 4.0, viewport, passes=3, seed=5)[1]
    b = tri.render_triangulation(x, y, 4.0, viewport, passes=3, seed=5)[1]
    c = tri.render_triangulation(x, y, 4.0, viewport, passes=3, seed=6)[1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # threads do not change the result
    d = tri.render_triangulation(
        x, y, 4.0, viewport, passes=3, seed=5, workers=3
    )[1]
    assert np.array_equal(a, d)


def test_coarse_zoom_tends_to_the_histogram():
    x, y = _clustered()
    viewport = ((0.0, 0.0), (H, W))
    locs = pd.DataFrame({"x": x, "y": y})
    _, hist = render.render(
        locs, _info(), disp_px_size=130 * 8, viewport=viewport
    )
    _, plain = tri.render_triangulation(x, y, 1 / 8, viewport, passes=0)
    # at 8 camera pixels per display pixel nearly every triangle is
    # within a pixel and drops its share into it: close to the counts
    assert plain.shape == hist.shape
    assert np.abs(plain - hist).mean() < 0.1 * hist.mean()


def test_too_few_points_fall_back_to_the_histogram():
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([1.0, 1.0], dtype=np.float32)
    n, image = tri.render_triangulation(x, y, 1.0, ((0.0, 0.0), (4.0, 4.0)))
    assert n == 2 and image.sum() == 2
    n, image = tri.render_triangulation(
        x[:0], y[:0], 1.0, ((0.0, 0.0), (4.0, 4.0))
    )
    assert n == 0 and image.sum() == 0


def test_through_render_and_render_scene(monkeypatch):
    from picasso.render import scene

    x, y = _clustered()
    locs = pd.DataFrame({"x": x, "y": y, "z": np.zeros_like(x)})
    viewport = ((8.0, 8.0), (56.0, 56.0))
    n, image = render.render(
        locs,
        _info(),
        disp_px_size=130 / 4,
        viewport=viewport,
        blur_method="triangulation",
        triangulation_passes=2,
        triangulation_jitter=0.5,
    )
    assert n > 0 and image.sum() == pytest.approx(n, rel=0.05)
    monkeypatch.setattr(
        scene, "_get_backend", lambda *a, **k: pytest.fail("GPU consulted")
    )
    _, n2, raw = render.render_scene(
        locs,
        _info(),
        disp_px_size=130 / 4,
        viewport=viewport,
        blur_method="triangulation",
        triangulation_passes=2,
        triangulation_jitter=0.5,
        return_raw_image=True,
    )
    assert n2 == n
    np.testing.assert_array_equal(raw, image)
    # rotated: the projected points are triangulated
    n3, rotated = render.render(
        locs,
        _info(),
        disp_px_size=130 / 4,
        viewport=viewport,
        blur_method="triangulation",
        triangulation_passes=2,
        ang=(0.3, 0.2, 0.1),
    )
    assert n3 > 0 and rotated.sum() == pytest.approx(n3, rel=0.05)
