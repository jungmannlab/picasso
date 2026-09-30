"""The quad-tree adaptive histogram (``blur_method="quadtree"``, Baddeley,
Cannell & Soeller 2010) of ``picasso.render``: checked pixel for pixel
against a plain reference tree, against the histogram at capacity 0,
for linearity and for the paper's behavior on uniform and mixed-density
fields.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from picasso import render, spatial_index
from picasso.render import splat


def _info(width: float, height: float) -> list[dict]:
    return [
        {"Width": width, "Height": height, "Frames": 1, "Pixelsize": 130.0}
    ]


def _quadtree(pyramid, x, y, oversampling, viewport, capacity):
    """``render.render`` with the quad-tree, on arrays and a pyramid."""
    (y_min, x_min), (y_max, x_max) = viewport
    columns = splat._RenderColumns(x, y, pyramid=pyramid)
    return splat._render_quadtree(
        columns,
        _info(pyramid.width, pyramid.height),
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
        capacity,
    )


class _RefNode:
    """A plain quad-tree in the paper's formulation (insert, split
    above the capacity), rooted on the field of view like the implicit
    one, with the density of every leaf accumulated over the pixels it
    overlaps. Slow, but obviously right: the oracle for the kernel."""

    def __init__(self, bbox, capacity, depth=0):
        self.bbox = bbox  # (x0, x1, y0, y1)
        self.capacity = capacity
        self.depth = depth
        self.points = []
        self.children = None

    def insert(self, point):
        if self.children is None:
            self.points.append(point)
            if len(self.points) > self.capacity and self.depth < 40:
                self._split()
            return
        for child in self.children:
            x0, x1, y0, y1 = child.bbox
            if x0 <= point[0] < x1 and y0 <= point[1] < y1:
                child.insert(point)
                return

    def _split(self):
        x0, x1, y0, y1 = self.bbox
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        boxes = [
            (x0, xm, y0, ym),
            (xm, x1, y0, ym),
            (x0, xm, ym, y1),
            (xm, x1, ym, y1),
        ]
        self.children = [
            _RefNode(box, self.capacity, self.depth + 1) for box in boxes
        ]
        points, self.points = self.points, []
        for point in points:
            self.insert(point)

    def leaves(self):
        if self.children is None:
            if self.points:
                yield self
        else:
            for child in self.children:
                yield from child.leaves()


def _reference_quadtree(x, y, root, capacity, oversampling, viewport):
    (y_min, x_min), (y_max, x_max) = viewport
    tree = _RefNode((0.0, root, 0.0, root), capacity)
    for point in zip(x, y):
        tree.insert(point)
    n_py = int(np.ceil(oversampling * (y_max - y_min)))
    n_px = int(np.ceil(oversampling * (x_max - x_min)))
    image = np.zeros((n_py, n_px))
    px = 1.0 / oversampling
    for leaf in tree.leaves():
        x0, x1, y0, y1 = leaf.bbox
        rho = len(leaf.points) / ((x1 - x0) * (y1 - y0))
        for i in range(n_py):
            py0 = y_min + i * px
            oy = min(py0 + px, y1) - max(py0, y0)
            if oy <= 0:
                continue
            for j in range(n_px):
                qx0 = x_min + j * px
                ox = min(qx0 + px, x1) - max(qx0, x0)
                if ox > 0:
                    image[i, j] += rho * ox * oy
    return image


def _clustered_locs(n=2000, width=64.0, height=64.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [rng.uniform(0, width, n // 2), rng.normal(20, 1.5, n // 2)]
    )
    y = np.concatenate(
        [rng.uniform(0, height, n // 2), rng.normal(40, 1.5, n // 2)]
    )
    keep = (x > 0) & (x < width) & (y > 0) & (y < height)
    return x[keep].astype(np.float32), y[keep].astype(np.float32)


class TestQuadTree:
    W = H = 64.0

    @pytest.mark.parametrize(
        "capacity,oversampling", [(10, 4.0), (5, 8.0), (30, 2.0), (1, 4.0)]
    )
    def test_matches_the_reference_tree(self, capacity, oversampling):
        x, y = _clustered_locs()
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        root = pyramid.block_sizes[0] * (1 << pyramid.root_bits)
        viewport = ((0.0, 0.0), (self.H, self.W))
        # at these densities no leaf is narrower than a display pixel,
        # so the kernel's per-pixel binning never differs from the
        # reference's area spreading
        n, image = _quadtree(pyramid, x, y, oversampling, viewport, capacity)
        reference = _reference_quadtree(
            x, y, root, capacity, oversampling, viewport
        )
        assert n == len(x)
        np.testing.assert_allclose(image, reference, rtol=1e-5, atol=1e-5)
        assert image.sum() == pytest.approx(len(x), rel=1e-5)  # linearity

    def test_capacity_zero_is_the_histogram(self):
        x, y = _clustered_locs()
        locs = pd.DataFrame({"x": x, "y": y})
        info = _info(self.W, self.H)
        viewport = ((0.0, 0.0), (self.H, self.W))
        n_h, hist = render.render(
            locs, info, disp_px_size=130 / 4, viewport=viewport
        )
        n_q, quad = render.render(
            locs,
            info,
            disp_px_size=130 / 4,
            viewport=viewport,
            blur_method="quadtree",
            quadtree_capacity=0,
        )
        assert n_q == n_h
        assert np.array_equal(hist, quad)

    def test_uniform_field_is_flat_at_high_capacity(self):
        rng = np.random.default_rng(3)
        n = 40_000
        x = rng.uniform(0, self.W, n).astype(np.float32)
        y = rng.uniform(0, self.H, n).astype(np.float32)
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        viewport = ((0.0, 0.0), (self.H, self.W))
        # capacity 2000: bins of ~1000 rows, SNR ~ 30; the image is the
        # density everywhere within a few percent
        _, image = _quadtree(pyramid, x, y, 4.0, viewport, 2000)
        density = n / (self.W * self.H) / 16  # per display pixel
        assert image.mean() == pytest.approx(density, rel=1e-3)
        assert image.std() / image.mean() < 0.1
        # capacity 5: much noisier, same mean
        _, noisy = _quadtree(pyramid, x, y, 4.0, viewport, 5)
        assert noisy.mean() == pytest.approx(density, rel=1e-3)
        assert noisy.std() > image.std()

    def test_bins_grow_where_density_falls(self):
        # the paper's Figure 4: a sparse and a dense half
        rng = np.random.default_rng(4)
        x = np.concatenate(
            [rng.uniform(0, 32, 400), rng.uniform(32, 64, 40_000)]
        ).astype(np.float32)
        y = rng.uniform(0, 64, len(x)).astype(np.float32)
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        _, image = _quadtree(
            pyramid, x, y, 8.0, ((0.0, 0.0), (self.H, self.W)), 10
        )
        sparse, dense = image[:, :256], image[:, 256:]

        # in the sparse half a bin spans many pixels: long runs of
        # equal values along a row; in the dense half almost none
        def mean_run(block):
            runs = []
            for row in block:
                same = np.diff(row) == 0
                runs.append(same.mean())
            return np.mean(runs)

        # (bins hold about five rows in both halves, so the dense half's
        # bins still span pixels; they are just much smaller)
        assert mean_run(sparse) > 0.95
        assert mean_run(dense) < mean_run(sparse) - 0.1

    def test_viewport_culls_and_counts(self):
        x, y = _clustered_locs()
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        viewport = ((30.0, 10.0), (50.0, 30.0))
        n, image = _quadtree(pyramid, x, y, 4.0, viewport, 10)
        in_view = (x > 10) & (x < 30) & (y > 30) & (y < 50)
        assert n == in_view.sum()
        assert image.shape == (80, 80)
        # the leaves overlapping the viewport edge contribute only
        # their overlap, so the total stays near the count in view
        assert image.sum() == pytest.approx(n, rel=0.15)

    @pytest.mark.parametrize("capacity", [0, 10])
    def test_rotated_is_the_tree_of_the_projected_points(self, capacity):
        # in 3D the method is applied to the projected point set: at
        # capacity 0 that is exactly the rotated histogram, at any
        # capacity the total is the count in view
        rng = np.random.default_rng(6)
        x, y = _clustered_locs()
        z = rng.normal(0.0, 2.0, len(x)).astype(np.float32)
        locs = pd.DataFrame({"x": x, "y": y, "z": z})
        info = _info(self.W, self.H)
        viewport = ((8.0, 8.0), (56.0, 56.0))
        ang = (0.4, 0.3, 0.2)
        n_h, hist = render.render(
            locs, info, disp_px_size=130 / 4, viewport=viewport, ang=ang
        )
        n_q, quad = render.render(
            locs,
            info,
            disp_px_size=130 / 4,
            viewport=viewport,
            blur_method="quadtree",
            quadtree_capacity=capacity,
            ang=ang,
        )
        assert n_q == n_h
        assert quad.sum() == pytest.approx(n_q, rel=1e-5)
        if capacity == 0:
            assert np.array_equal(hist, quad)
        else:
            assert not np.array_equal(hist, quad)  # adaptive bins
            assert quad.shape == hist.shape

    def test_animation_passes_the_capacity(self, monkeypatch, tmp_path):
        from picasso.render import animation

        seen = []

        def fake_render_scene(**kwargs):
            seen.append(kwargs.get("quadtree_capacity"))
            from PyQt6 import QtGui

            return QtGui.QImage(4, 4, QtGui.QImage.Format.Format_RGB32), 0

        monkeypatch.setattr(animation, "render_scene", fake_render_scene)

        class _Writer:
            def __init__(self, *a, **k):
                pass

            def append_data(self, *a, **k):
                pass

            def close(self):
                pass

        monkeypatch.setattr(
            animation.imageio, "get_writer", lambda *a, **k: _Writer()
        )
        x, y = _clustered_locs()
        locs = pd.DataFrame({"x": x, "y": y, "z": np.zeros_like(x)})
        from scipy.spatial.transform import Rotation

        viewport = ((0.0, 0.0), (self.H, self.W))
        animation.build_animation(
            str(tmp_path / "a.mp4"),
            locs,
            _info(self.W, self.H),
            positions=[(Rotation.identity(), viewport)] * 2,
            durations=[0.1],
            disp_px_size=130 / 4,
            image_size=(16, 16),
            blur_method="quadtree",
            quadtree_capacity=7,
            fps=10,
        )
        assert seen and all(c == 7 for c in seen)

    def test_scene_uses_the_given_index_on_the_cpu(self, monkeypatch):
        from picasso.render import scene

        x, y = _clustered_locs()
        locs = pd.DataFrame({"x": x, "y": y})
        info = _info(self.W, self.H)
        pyramid = spatial_index.build_render_index_arrays(x, y, self.W, self.H)
        viewport = ((0.0, 0.0), (self.H, self.W))
        built = []
        original = spatial_index.build_render_index_arrays
        monkeypatch.setattr(
            spatial_index,
            "build_render_index_arrays",
            lambda *a, **k: built.append(1) or original(*a, **k),
        )
        monkeypatch.setattr(
            scene,
            "_get_backend",
            lambda *a, **k: pytest.fail("the backend selection was consulted"),
        )
        _, n, raw = render.render_scene(
            locs,
            info,
            disp_px_size=130 / 4,
            viewport=viewport,
            blur_method="quadtree",
            render_index=pyramid,
            return_raw_image=True,
        )
        assert n == len(x) and built == []
        _, _, raw2 = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=130 / 4,
            viewport=viewport,
            blur_method="quadtree",
            return_raw_image=True,
        )
        assert built == [1, 1]  # built on the fly, once per channel
        np.testing.assert_array_equal(raw2[0], raw)
