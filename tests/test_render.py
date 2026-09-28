"""Test picasso.render functions and the associated functions in
picasso.masking.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import logging

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui
from scipy.spatial.transform import Rotation

from picasso import io, lib, masking, render

from tests.conftest import PIXELSIZE

# parameters reused across tests
VIEWPORT = ((15, 15), (16, 16))
FULL_VIEWPORT = ((0, 0), (32, 32))
BLUR_METHODS = ["gaussian", "gaussian_iso", "smooth", "convolve"]
LINEAR_BLUR_METHODS = ["smooth", "convolve"]  # preserve total mass
MASKING_METHODS = [
    "isodata",
    "li",
    "mean",
    "minimum",
    "otsu",
    "triangle",
    "yen",
    "local_gaussian",
    "local_mean",
    "local_median",
    0.01,
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _qt_app(qapp):
    """Ensure an application object exists for QImage / QPainter
    operations.

    Defers to the shared ``qapp`` rather than building a
    ``QGuiApplication`` here: Qt allows one application object per
    process, and a plain ``QGuiApplication`` has no ``topLevelWidgets``,
    which the ``qt_offscreen`` fixture needs. Creating one here first
    would therefore break every widget test that runs afterwards.
    """
    return qapp


@pytest.fixture(scope="module")
def locs_data():
    """Load localization data once per test module."""
    return io.load_locs("./tests/data/testdata_locs.hdf5")


@pytest.fixture(scope="module")
def locs(locs_data):
    return locs_data[0]


@pytest.fixture(scope="module")
def info(locs_data):
    return locs_data[1]


@pytest.fixture(scope="module")
def locs_3d(locs):
    """Synthetic 3D locs (no z column in the test data file)."""
    rng = np.random.default_rng(0)
    locs_z = locs.copy()
    locs_z["z"] = rng.uniform(-100.0, 100.0, size=len(locs)).astype(np.float32)
    return locs_z


@pytest.fixture(scope="module")
def image(locs, info):
    """Rendered image used by masking tests."""
    return render.render(locs, info, disp_px_size=PIXELSIZE / 13)[1]


@pytest.fixture(scope="module")
def small_qimage():
    """Small black QImage used as a canvas for draw_* / export_* tests."""
    img = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(0, 0, 0))
    return img


def _qimage_to_array(qimage):
    """Convert a QImage (Format_RGB32) to an HxWx4 uint8 numpy array."""
    width = qimage.width()
    height = qimage.height()
    bits = qimage.bits()
    bits.setsize(height * width * 4)
    return np.frombuffer(bits, dtype=np.uint8).reshape(height, width, 4).copy()


# ---------------------------------------------------------------------------
# render.render
# ---------------------------------------------------------------------------


class TestRender:
    """Tests for the top-level render.render dispatcher."""

    def test_no_blur_mass_conservation(self, locs, info):
        """Each loc deposits exactly 1, so image.sum() must equal n."""
        n, im = render.render(
            locs, info, disp_px_size=PIXELSIZE / 13, viewport=VIEWPORT
        )
        assert im.sum() == n
        assert n > 0, "Test data should have locs in viewport"

    def test_viewport_exact_shape(self, locs, info):
        """Image shape is exactly oversampling * viewport size."""
        n, im = render.render(
            locs, info, disp_px_size=PIXELSIZE / 130, viewport=VIEWPORT
        )
        assert im.shape == (130, 130)
        assert im.dtype == np.float32

    def test_returned_n_matches_in_view(self, locs, info):
        """`n` returned from render equals the count of locs strictly inside
        the viewport."""
        (y_min, x_min), (y_max, x_max) = VIEWPORT
        x = locs["x"].to_numpy()
        y = locs["y"].to_numpy()
        in_view = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
        expected = int(in_view.sum())
        n, _ = render.render(
            locs, info, disp_px_size=PIXELSIZE / 13, viewport=VIEWPORT
        )
        assert n == expected

    @pytest.mark.parametrize("blur_method", BLUR_METHODS)
    def test_blur_methods(self, locs, info, blur_method):
        """All four blur methods produce correctly-shaped, finite, non-zero
        images."""
        n, im = render.render(
            locs,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method=blur_method,
        )
        assert im.shape == (160, 160)
        assert im.dtype == np.float32
        assert np.isfinite(im).all()
        assert im.sum() > 0
        assert n > 0

    @pytest.mark.parametrize("blur_method", LINEAR_BLUR_METHODS)
    def test_linear_blur_preserves_mass(self, locs, info, blur_method):
        """`smooth` and `convolve` apply mass-preserving operations after
        the histogram fill, so total mass should still equal n."""
        n, im = render.render(
            locs,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method=blur_method,
        )
        assert im.sum() == pytest.approx(n, rel=1e-3)

    def test_min_blur_width_broadens(self, locs, info):
        """Larger min_blur_width spreads mass: peak intensity must drop."""
        _, im_narrow = render.render(
            locs,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            min_blur_width=0.0,
        )
        _, im_wide = render.render(
            locs,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            min_blur_width=2.0,
        )
        assert im_wide.max() < im_narrow.max()

    def test_invalid_blur_raises(self, locs, info):
        with pytest.raises(Exception, match="blur_method"):
            render.render(
                locs,
                info,
                disp_px_size=PIXELSIZE / 5,
                viewport=FULL_VIEWPORT,
                blur_method="not_a_method",
            )

    def test_no_info_no_viewport_raises(self, locs):
        with pytest.raises(Exception):
            render.render(locs, None, disp_px_size=PIXELSIZE / 5)

    def test_empty_locs_gaussian(self, locs, info):
        """Empty input must not crash the parallel _fill_gaussian path."""
        empty = locs.iloc[:0]
        n, im = render.render(
            empty,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
        )
        assert n == 0
        assert im.shape == (160, 160)
        assert (im == 0).all()

    def test_empty_locs_gaussian_rot(self, locs_3d, info):
        """Empty input must not crash the parallel _fill_gaussian_rot path."""
        empty = locs_3d.iloc[:0]
        n, im = render.render(
            empty,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            ang=(0.1, 0.2, 0.3),
        )
        assert n == 0
        assert (im == 0).all()

    def test_3d_rotation_changes_image(self, locs_3d, info):
        """A non-zero rotation must produce a different image."""
        _, im_no_rot = render.render(
            locs_3d,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            ang=(0.0, 0.0, 0.0),
        )
        _, im_rot = render.render(
            locs_3d,
            info,
            disp_px_size=PIXELSIZE / 5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            ang=(0.5, 0.3, 0.2),
        )
        assert not np.array_equal(im_no_rot, im_rot)

    def test_gaussian_angle_zero_matches_unrotated(self):
        """An 'angle' column of 0 reproduces the axis-aligned render."""
        info = {"Pixelsize": 100.0, "Height": 20, "Width": 20}
        base = pd.DataFrame(
            {"x": [10.0], "y": [10.0], "lpx": [0.6], "lpy": [0.2]}
        )
        _, im_plain = render.render(
            base, info, disp_px_size=25.0, blur_method="gaussian"
        )
        _, im_rot0 = render.render(
            base.assign(angle=[0.0]),
            info,
            disp_px_size=25.0,
            blur_method="gaussian",
        )
        assert np.allclose(im_rot0, im_plain, rtol=1e-4, atol=1e-6)

    def test_gaussian_angle_ninety_swaps_precision(self):
        """A 90 degree rotation is equivalent to swapping lpx and lpy."""
        info = {"Pixelsize": 100.0, "Height": 20, "Width": 20}
        rotated = pd.DataFrame(
            {
                "x": [10.0],
                "y": [10.0],
                "lpx": [0.6],
                "lpy": [0.2],
                "angle": [90.0],
            }
        )
        swapped = pd.DataFrame(
            {"x": [10.0], "y": [10.0], "lpx": [0.2], "lpy": [0.6]}
        )
        _, im_rot = render.render(
            rotated, info, disp_px_size=25.0, blur_method="gaussian"
        )
        _, im_swap = render.render(
            swapped, info, disp_px_size=25.0, blur_method="gaussian"
        )
        assert np.allclose(im_rot, im_swap, rtol=1e-4, atol=1e-6)

    def test_gaussian_angle_changes_image_and_keeps_mass(self):
        """A non-zero rotation tilts the ellipse while conserving mass."""
        info = {"Pixelsize": 100.0, "Height": 20, "Width": 20}
        base = pd.DataFrame(
            {"x": [10.0], "y": [10.0], "lpx": [0.6], "lpy": [0.2]}
        )
        _, im0 = render.render(
            base.assign(angle=[0.0]),
            info,
            disp_px_size=25.0,
            blur_method="gaussian",
        )
        _, im45 = render.render(
            base.assign(angle=[45.0]),
            info,
            disp_px_size=25.0,
            blur_method="gaussian",
        )
        assert not np.allclose(im0, im45)
        assert np.isclose(im0.sum(), im45.sum(), rtol=1e-3)

    def test_empty_locs_gaussian_theta(self):
        """Empty input with an 'angle' column must not crash."""
        info = {"Pixelsize": 100.0, "Height": 20, "Width": 20}
        empty = (
            pd.DataFrame(
                {"x": [10.0], "y": [10.0], "lpx": [0.6], "lpy": [0.2]}
            )
            .assign(angle=[0.0])
            .iloc[:0]
        )
        n, im = render.render(
            empty, info, disp_px_size=25.0, blur_method="gaussian"
        )
        assert n == 0
        assert (im == 0).all()

    def test_gaussian_rot_angle_changes_image(self):
        """Per-loc angle must affect the globally-rotated (3D) render."""
        info = {"Pixelsize": 100.0, "Height": 20, "Width": 20}
        base = pd.DataFrame(
            {
                "x": [10.0],
                "y": [10.0],
                "z": [0.0],
                "lpx": [0.6],
                "lpy": [0.2],
                "lpz": [0.4],
            }
        )
        ang = (0.3, 0.2, 0.1)
        _, im0 = render.render(
            base.assign(angle=[0.0]),
            info,
            disp_px_size=25.0,
            blur_method="gaussian",
            ang=ang,
        )
        _, im45 = render.render(
            base.assign(angle=[45.0]),
            info,
            disp_px_size=25.0,
            blur_method="gaussian",
            ang=ang,
        )
        assert not np.allclose(im0, im45)
        # mass is only approximately conserved under a global tilt because
        # the 3-sigma bounding box truncates differently per orientation
        assert np.isclose(im0.sum(), im45.sum(), rtol=1e-2)

    def test_gaussian_rot_angle_matches_2d_under_identity(self):
        """Under identity global rotation the composed 3D path reduces to
        the pure 2D rotated render."""
        info = {"Pixelsize": 100.0, "Height": 20, "Width": 20}
        base = pd.DataFrame(
            {
                "x": [10.0],
                "y": [10.0],
                "z": [0.0],
                "lpx": [0.6],
                "lpy": [0.2],
                "lpz": [0.4],
                "angle": [40.0],
            }
        )
        _, im_2d = render.render(
            base, info, disp_px_size=25.0, blur_method="gaussian"
        )
        _, im_3d = render.render(
            base,
            info,
            disp_px_size=25.0,
            blur_method="gaussian",
            ang=(0.0, 0.0, 0.0),
        )
        assert np.allclose(im_2d, im_3d, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# render_hist_numba
# ---------------------------------------------------------------------------


class TestRenderHistNumba:
    def test_real_data_mass_conservation(self, locs):
        """Integration check on the real test dataset."""
        x = locs["x"].to_numpy().astype(np.float32)
        y = locs["y"].to_numpy().astype(np.float32)
        n, im = render.render_hist_numba(
            x, y, oversampling=4.0, t_min=0.0, t_max=32.0
        )
        assert im.shape == (128, 128)
        assert im.dtype == np.float32
        assert im.sum() == n

    def test_basic_synthetic(self):
        """Three in-bounds points produce a 3x3 image with mass = 3."""
        x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        y = np.array([0.5, 1.5, 2.5], dtype=np.float32)

        n, im = render.render_hist_numba(
            x, y, oversampling=1.0, t_min=0.0, t_max=3.0
        )

        assert n == 3
        assert im.shape == (3, 3)
        assert im.dtype == np.float32
        assert im.sum() == n

    def test_excludes_out_of_bounds(self):
        x = np.array([0.5, 5.5], dtype=np.float32)
        y = np.array([0.5, 5.5], dtype=np.float32)

        n, _ = render.render_hist_numba(
            x, y, oversampling=1.0, t_min=0.0, t_max=3.0
        )

        assert n == 1

    def test_oversampling_scales_image(self):
        x = np.array([0.5], dtype=np.float32)
        y = np.array([0.5], dtype=np.float32)

        _, im1 = render.render_hist_numba(x, y, 1.0, 0.0, 1.0)
        _, im2 = render.render_hist_numba(x, y, 2.0, 0.0, 1.0)

        assert im1.shape == (1, 1)
        assert im2.shape == (2, 2)

    def test_empty_input(self):
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)

        n, im = render.render_hist_numba(
            x, y, oversampling=1.0, t_min=0.0, t_max=3.0
        )

        assert n == 0
        assert np.all(im == 0)


# ---------------------------------------------------------------------------
# 3D rendering
# ---------------------------------------------------------------------------


class TestRenderHist3D:
    def test_basic(self, locs_3d):
        n, im = render.render_hist3d(
            locs_3d["x"].to_numpy(),
            locs_3d["y"].to_numpy(),
            locs_3d["z"].to_numpy(),
            oversampling=2,
            y_min=0,
            x_min=0,
            y_max=32,
            x_max=32,
            z_min=-100,
            z_max=100,
            pixelsize=PIXELSIZE,
        )
        assert im.ndim == 3
        assert im.dtype == np.float32
        assert im.sum() == n

    def test_z_filtering(self, locs_3d):
        """Locs outside [z_min, z_max] must be excluded from n and image."""
        z_min, z_max = -50.0, 50.0
        z = locs_3d["z"].to_numpy()
        x = locs_3d["x"].to_numpy()
        y = locs_3d["y"].to_numpy()
        in_view = (
            (x > 0)
            & (x < 32)
            & (y > 0)
            & (y < 32)
            & (z / PIXELSIZE > z_min / PIXELSIZE)
            & (z / PIXELSIZE < z_max / PIXELSIZE)
        )
        expected = int(in_view.sum())
        n, _ = render.render_hist3d(
            x,
            y,
            z,
            oversampling=2,
            y_min=0,
            x_min=0,
            y_max=32,
            x_max=32,
            z_min=z_min,
            z_max=z_max,
            pixelsize=PIXELSIZE,
        )
        assert n == expected

    def test_anisotropic_axes(self, locs_3d):
        """Different oversampling per axis produces matching axis sizes."""
        n, im = render.render_hist3d_anisotropic(
            locs_3d["x"].to_numpy(),
            locs_3d["y"].to_numpy(),
            locs_3d["z"].to_numpy(),
            oversampling_x=2.0,
            oversampling_y=4.0,
            oversampling_z=1.0,
            y_min=0,
            x_min=0,
            y_max=32,
            x_max=32,
            z_min=-100,
            z_max=100,
            pixelsize=PIXELSIZE,
        )
        # n_pixel_y = ceil(4 * 32) = 128, n_pixel_x = ceil(2 * 32) = 64
        # n_pixel_z = ceil(1 * (100/130 - (-100/130))) = ceil(1.539) = 2
        assert im.shape == (128, 64, 2)
        assert im.sum() == n


# ---------------------------------------------------------------------------
# Viewport math
# ---------------------------------------------------------------------------


class TestViewport:
    @pytest.mark.parametrize(
        "viewport, height, width, center",
        [
            (((0, 0), (10, 20)), 10, 20, (5, 10)),
            (((10, 20), (30, 50)), 20, 30, (20, 35)),
            (((-5, -5), (5, 5)), 10, 10, (0, 0)),
        ],
    )
    def test_height_width_size_center(self, viewport, height, width, center):
        assert render.viewport_height(viewport) == height
        assert render.viewport_width(viewport) == width
        assert render.viewport_size(viewport) == (height, width)
        assert render.viewport_center(viewport) == center

    def test_shift_invariants(self):
        v = ((10.0, 20.0), (30.0, 50.0))
        dx, dy = 3.0, -2.0
        new = render.shift_viewport(v, dx, dy)
        # size is preserved
        assert render.viewport_size(new) == render.viewport_size(v)
        # center moves by (dy, dx)
        old_c = render.viewport_center(v)
        new_c = render.viewport_center(new)
        assert new_c[0] == pytest.approx(old_c[0] + dy)
        assert new_c[1] == pytest.approx(old_c[1] + dx)

    def test_zoom_no_cursor_keeps_center(self):
        v = ((10.0, 20.0), (30.0, 50.0))
        factor = 2.0
        new = render.zoom_viewport(v, factor)
        assert render.viewport_center(new) == pytest.approx(
            render.viewport_center(v)
        )
        h0, w0 = render.viewport_size(v)
        h1, w1 = render.viewport_size(new)
        assert h1 == pytest.approx(h0 * factor)
        assert w1 == pytest.approx(w0 * factor)

    def test_zoom_round_trip(self):
        v = ((10.0, 20.0), (30.0, 50.0))
        roundtrip = render.zoom_viewport(render.zoom_viewport(v, 2.0), 0.5)
        assert np.allclose(np.array(roundtrip), np.array(v))

    def test_zoom_with_cursor_at_center_equals_no_cursor(self):
        v = ((10.0, 20.0), (30.0, 50.0))
        cy, cx = render.viewport_center(v)
        new_with = render.zoom_viewport(v, 2.0, cursor_position=(cx, cy))
        new_without = render.zoom_viewport(v, 2.0)
        assert np.allclose(np.array(new_with), np.array(new_without))

    def test_adjust_aspect_ratio_matching(self, small_qimage):
        """When viewport already matches image aspect ratio, no change."""
        v = ((0.0, 0.0), (64.0, 64.0))
        adjusted = render.adjust_viewport_to_aspect_ratio(small_qimage, v)
        assert np.allclose(np.array(adjusted), np.array(v))

    def test_adjust_aspect_ratio_widens(self):
        """Wider image → x-range expands; y-range stays the same."""
        wide = QtGui.QImage(200, 100, QtGui.QImage.Format.Format_RGB32)
        v = ((0.0, 0.0), (10.0, 10.0))
        adjusted = render.adjust_viewport_to_aspect_ratio(wide, v)
        # y unchanged
        assert adjusted[0][0] == 0.0 and adjusted[1][0] == 10.0
        # x expanded symmetrically
        assert adjusted[0][1] < 0.0 and adjusted[1][1] > 10.0
        new_w = adjusted[1][1] - adjusted[0][1]
        new_h = adjusted[1][0] - adjusted[0][0]
        assert new_w / new_h == pytest.approx(200 / 100)


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------


class TestMapToView:
    def test_origin_maps_to_zero(self):
        size = QtCore.QSize(100, 200)
        v = ((10.0, 20.0), (30.0, 50.0))
        cx, cy = render.map_to_view(v[0][1], v[0][0], size, v)
        assert (cx, cy) == (0, 0)

    def test_known_interior_point(self):
        """Center of viewport maps to image center (within int truncation)."""
        size = QtCore.QSize(100, 200)
        v = ((0.0, 0.0), (10.0, 20.0))
        cy, cx = render.viewport_center(v)
        out_x, out_y = render.map_to_view(cx, cy, size, v)
        assert out_x == 50
        assert out_y == 100


# ---------------------------------------------------------------------------
# Rotation math
# ---------------------------------------------------------------------------


class TestRotation:
    def test_zero_angle_is_identity(self):
        R = render.rotation_matrix(0.0, 0.0, 0.0).as_matrix()
        assert np.allclose(R, np.eye(3))

    def test_orthogonality(self):
        R = render.rotation_matrix(0.4, -0.7, 1.1).as_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)

    def test_z_axis_90_degrees(self):
        R = render.rotation_matrix(0.0, 0.0, np.pi / 2).as_matrix()
        out = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-6)

    def test_locs_rotation_zero_angle_preserves_coords(self, locs_3d):
        x_min, x_max, y_min, y_max = 0.0, 32.0, 0.0, 32.0
        oversampling = 5.0
        x_in = locs_3d["x"].to_numpy()
        y_in = locs_3d["y"].to_numpy()
        in_view_expected = (
            (x_in > x_min) & (x_in < x_max) & (y_in > y_min) & (y_in < y_max)
        )
        x_out, y_out, in_view, _ = render.locs_rotation(
            locs_3d, oversampling, x_min, x_max, y_min, y_max, (0.0, 0.0, 0.0)
        )
        # x_out = oversampling * (x - x_min), in-view subset only
        expected_x = oversampling * (x_in[in_view_expected] - x_min)
        expected_y = oversampling * (y_in[in_view_expected] - y_min)
        assert np.allclose(x_out, expected_x, atol=1e-5)
        assert np.allclose(y_out, expected_y, atol=1e-5)

    def test_locs_rotation_in_view_consistency(self, locs_3d):
        x_out, y_out, in_view, z_out = render.locs_rotation(
            locs_3d, 5.0, 0.0, 32.0, 0.0, 32.0, (0.1, 0.2, 0.3)
        )
        n_in_view = int(in_view.sum())
        assert len(x_out) == n_in_view
        assert len(y_out) == n_in_view
        assert len(z_out) == n_in_view

    def test_to_rotation_none(self):
        assert render.to_rotation(None) is None

    def test_to_rotation_passes_rotation_through(self):
        R = Rotation.from_rotvec([0.1, -0.2, 0.3])
        assert render.to_rotation(R) is R

    def test_to_rotation_legacy_euler_equivalence(self):
        ang = (0.4, -0.7, 1.1)
        R = render.to_rotation(ang)
        expected = render.rotation_matrix(*ang)
        assert np.allclose(R.as_matrix(), expected.as_matrix())

    def test_locs_rotation_accepts_rotation_object(self, locs_3d):
        ang = (0.1, 0.2, 0.3)
        out_tuple = render.locs_rotation(
            locs_3d, 5.0, 0.0, 32.0, 0.0, 32.0, ang
        )
        out_rotation = render.locs_rotation(
            locs_3d, 5.0, 0.0, 32.0, 0.0, 32.0, render.rotation_matrix(*ang)
        )
        for a, b in zip(out_tuple, out_rotation):
            assert np.allclose(a, b)

    def test_render_accepts_rotation_object(self, locs_3d, info):
        """Rendering with a scipy Rotation matches the legacy Euler
        tuple input for all rotation-aware code paths."""
        ang = (0.5, 0.3, 0.2)
        for blur_method in (None, "gaussian"):
            _, im_tuple = render.render(
                locs_3d,
                info,
                disp_px_size=25,
                viewport=FULL_VIEWPORT,
                blur_method=blur_method,
                ang=ang,
            )
            _, im_rotation = render.render(
                locs_3d,
                info,
                disp_px_size=25,
                viewport=FULL_VIEWPORT,
                blur_method=blur_method,
                ang=render.rotation_matrix(*ang),
            )
            assert np.allclose(im_tuple, im_rotation)


class TestClosestRotvec:
    def test_zero_reference_returns_base(self):
        R = Rotation.from_rotvec([0.3, 0.0, 0.0])
        out = render.closest_rotvec(R, np.zeros(3))
        assert np.allclose(out, [0.3, 0.0, 0.0])

    def test_unwraps_full_turns(self):
        """A 10-degree rotation with a reference near 370 degrees must
        unwrap to 370 degrees."""
        axis = np.array([0.0, 0.0, 1.0])
        R = Rotation.from_rotvec(np.radians(10) * axis)
        reference = np.radians(365) * axis
        out = render.closest_rotvec(R, reference)
        assert np.allclose(out, np.radians(370) * axis)

    def test_unwraps_across_pi(self):
        """Crossing 180 degrees must continue counting up instead of
        flipping the axis."""
        axis = np.array([1.0, 0.0, 0.0])
        R = Rotation.from_rotvec(np.radians(181) * axis)  # wraps to -179
        reference = np.radians(180) * axis
        out = render.closest_rotvec(R, reference)
        assert np.allclose(out, np.radians(181) * axis)

    def test_identity_keeps_turns_of_reference(self):
        """The identity rotation with a reference of ~2 turns must keep
        the full turns along the reference axis."""
        axis = np.array([0.0, 1.0, 0.0])
        out = render.closest_rotvec(
            Rotation.identity(), np.radians(719) * axis
        )
        assert np.allclose(out, np.radians(720) * axis)

    def test_identity_zero_reference(self):
        out = render.closest_rotvec(Rotation.identity(), np.zeros(3))
        assert np.allclose(out, np.zeros(3))

    def test_result_represents_same_rotation(self):
        R = Rotation.from_rotvec([0.2, -0.4, 0.6])
        reference = np.array([2.5, -4.0, 6.5])
        out = render.closest_rotvec(R, reference)
        assert np.allclose(
            (Rotation.from_rotvec(out) * R.inv()).magnitude(), 0.0, atol=1e-9
        )


# ---------------------------------------------------------------------------
# 3x3 matrix helpers
# ---------------------------------------------------------------------------


class TestMathUtils:
    def test_inverse_3x3_matches_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3)).astype(np.float32) + 5 * np.eye(
            3, dtype=np.float32
        )
        assert np.allclose(render.inverse_3x3(A), np.linalg.inv(A), atol=1e-4)

    def test_inverse_3x3_identity(self):
        I = np.eye(3, dtype=np.float32)
        assert np.allclose(render.inverse_3x3(I), I, atol=1e-6)

    def test_inverse_3x3_round_trip(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((3, 3)).astype(np.float32) + 5 * np.eye(
            3, dtype=np.float32
        )
        assert np.allclose(A @ render.inverse_3x3(A), np.eye(3), atol=1e-4)

    def test_determinant_3x3_matches_numpy(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((3, 3)).astype(np.float32)
        assert render.determinant_3x3(A) == pytest.approx(
            np.linalg.det(A), rel=1e-4
        )


# ---------------------------------------------------------------------------
# _fftconvolve (spatial / FFT branches)
# ---------------------------------------------------------------------------


class TestFftConvolve:
    """The spatial vs FFT branch is chosen by kernel size relative to image.
    Kernel formula: ``10 * round(blur) + 1``. Spatial branch is taken when
    both kernel dims are < 0.05 * image dim and max(kernel) <= 101.
    """

    def test_spatial_branch_preserves_mass(self):
        """Small kernel → ndimage.gaussian_filter spatial path."""
        im = np.zeros((64, 64), dtype=np.float32)
        im[32, 32] = 1.0
        # blur=0.05 → kernel=1; 1 < 0.05*64=3.2 → spatial
        out = render._fftconvolve(im, 0.05, 0.05)
        assert out.shape == im.shape
        assert out.dtype == np.float32
        assert out.sum() == pytest.approx(1.0, abs=1e-5)

    def test_fft_branch_preserves_mass(self):
        """Kernel large relative to image → fftconvolve path."""
        im = np.zeros((64, 64), dtype=np.float32)
        im[32, 32] = 1.0
        # blur=2.0 → kernel=21; 21 > 0.05*64=3.2 → FFT
        out = render._fftconvolve(im, 2.0, 2.0)
        assert out.shape == im.shape
        assert out.dtype == np.float32
        assert out.sum() == pytest.approx(1.0, abs=1e-3)

    def test_branches_agree_on_intermediate_blur(self):
        """Both branches should produce visually similar results for a
        blur that happens to fall near the threshold."""
        rng = np.random.default_rng(0)
        im = rng.random((256, 256)).astype(np.float32)
        # blur=1.0 → kernel=11. 11 < 0.05*256=12.8 → spatial.
        out_spatial = render._fftconvolve(im, 1.0, 1.0)
        # blur=1.0 with smaller image forces FFT: kernel=11 > 0.05*100=5
        out_fft = render._fftconvolve(im[:100, :100], 1.0, 1.0)
        # Both must be finite and preserve total mass approximately.
        assert np.isfinite(out_spatial).all()
        assert np.isfinite(out_fft).all()
        # Mass approximately preserved; some loss near borders from
        # zero-padding (more pronounced on the smaller crop).
        assert out_spatial.sum() == pytest.approx(im.sum(), rel=1e-2)
        assert out_fft.sum() == pytest.approx(im[:100, :100].sum(), rel=5e-2)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


class TestImageProcessing:
    def test_scale_contrast_basic(self):
        im = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        out = render.scale_contrast(im)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_scale_contrast_with_explicit_limits(self):
        im = np.array([[0.0, 5.0], [10.0, 15.0]], dtype=np.float32)
        out = render.scale_contrast(im, vmin=5.0, vmax=10.0)
        assert (out >= 0.0).all() and (out <= 1.0).all()
        assert out[0, 0] == 0.0  # below vmin clipped
        assert out[1, 1] == 1.0  # above vmax clipped
        assert out[1, 0] == 1.0  # at vmax

    def test_scale_contrast_autoscale(self):
        im = np.array([[0.0, 10.0], [20.0, 100.0]], dtype=np.float32)
        out, limits = render.scale_contrast(
            im, autoscale=True, return_contrast_limits=True
        )
        assert limits == (0.0, 50.0)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_scale_contrast_returns_limits(self):
        im = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        out, limits = render.scale_contrast(im, return_contrast_limits=True)
        assert isinstance(limits, tuple)
        assert len(limits) == 2
        assert limits[0] == 1.0 and limits[1] == 4.0

    def test_scale_contrast_constant_image(self):
        """Exercises the vmin == vmax + 1e-6 branch."""
        im = np.full((4, 4), 5.0, dtype=np.float32)
        out = render.scale_contrast(im)
        assert np.isfinite(out).all()
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_to_8bit_dtype_and_range(self):
        im = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        out = render.to_8bit(im)
        assert out.dtype == np.uint8
        assert out.max() == 255
        assert out.min() >= 0

    def test_to_8bit_zero_image(self):
        """No div-by-zero on all-zero input."""
        im = np.zeros((4, 4), dtype=np.float32)
        out = render.to_8bit(im)
        assert out.dtype == np.uint8
        assert (out == 0).all()

    def test_apply_colormap_str(self):
        im = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = render.apply_colormap(im, "magma")
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_apply_colormap_array(self):
        """Accepts a 256x4 array and drops the alpha channel."""
        im = np.arange(256, dtype=np.uint8).reshape(16, 16)
        cmap = np.zeros((256, 4), dtype=np.float32)
        cmap[:, 0] = np.linspace(0, 1, 256)  # red ramp
        cmap[:, 3] = 1.0
        out = render.apply_colormap(im, cmap)
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_scale_intensities_default_no_op(self):
        images = np.ones((3, 5, 5), dtype=np.float32)
        out = render.scale_intensities(images.copy())
        assert np.array_equal(out, images)

    def test_scale_intensities_relative(self):
        images = np.ones((3, 5, 5), dtype=np.float32)
        out = render.scale_intensities(
            images.copy(), relative_intensities=[0.5, 1.0, 2.0]
        )
        assert np.allclose(out[0], 0.5)
        assert np.allclose(out[1], 1.0)
        assert np.allclose(out[2], 2.0)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


class TestColors:
    def test_get_colors_from_colormap_count(self):
        for n in [1, 3, 8, 16]:
            colors = render.get_colors_from_colormap(n)
            assert len(colors) == n

    def test_get_colors_from_colormap_range(self):
        colors = np.asarray(render.get_colors_from_colormap(5))
        assert (colors >= 0.0).all() and (colors <= 1.0).all()

    def test_get_group_color_modulo(self):
        df = pd.DataFrame({"group": np.arange(20)})
        out = render.get_group_color(df)
        assert (out == np.arange(20) % render.N_GROUP_COLORS).all()


# ---------------------------------------------------------------------------
# Per-channel LUT helpers (solid_to_lut, stops_to_lut)
# ---------------------------------------------------------------------------


class TestSolidToLut:
    """``render.solid_to_lut`` builds a (256, 3) float32 black->color ramp.
    For solid-color channels this LUT path is mathematically identical to
    the legacy ``intensity * rgb`` blend used by ``_render_multi_channel``.
    """

    def test_shape_and_dtype(self):
        lut = render.solid_to_lut((1.0, 0.0, 0.0))
        assert lut.shape == (256, 3)
        assert lut.dtype == np.float32

    def test_endpoints(self):
        """First row is black; last row is the target color."""
        lut = render.solid_to_lut((1.0, 0.5, 0.25))
        assert np.allclose(lut[0], [0.0, 0.0, 0.0])
        assert np.allclose(lut[-1], [1.0, 0.5, 0.25])

    def test_linear_ramp(self):
        """Row i is i / 255 * target_rgb."""
        rgb = np.array([0.8, 0.4, 0.2], dtype=np.float32)
        lut = render.solid_to_lut(rgb)
        expected = np.linspace(np.zeros(3), rgb, 256, dtype=np.float32)
        assert np.allclose(lut, expected)

    def test_accepts_tuple_list_array(self):
        """Helper accepts any 3-element rgb container."""
        for rgb in [
            (1.0, 0.0, 0.0),
            [1.0, 0.0, 0.0],
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ]:
            lut = render.solid_to_lut(rgb)
            assert lut.shape == (256, 3)
            assert np.allclose(lut[-1], [1.0, 0.0, 0.0])

    def test_black_target_is_all_zero(self):
        lut = render.solid_to_lut((0.0, 0.0, 0.0))
        assert (lut == 0).all()


class TestStopsToLut:
    """``render.stops_to_lut`` linearly interpolates a list of color stops
    into the same (256, 3) LUT shape consumed by ``_render_multi_channel``.
    """

    def test_shape_and_dtype(self):
        lut = render.stops_to_lut([(0.0, 0, 0, 0), (1.0, 1.0, 1.0, 1.0)])
        assert lut.shape == (256, 3)
        assert lut.dtype == np.float32

    def test_endpoints_match_first_and_last_stop(self):
        lut = render.stops_to_lut([(0.0, 0.1, 0.2, 0.3), (1.0, 0.7, 0.8, 0.9)])
        assert np.allclose(lut[0], [0.1, 0.2, 0.3])
        assert np.allclose(lut[-1], [0.7, 0.8, 0.9])

    def test_two_stop_matches_linspace(self):
        """A 2-stop gradient is equivalent to a per-channel linspace."""
        lut = render.stops_to_lut([(0.0, 0, 0, 0), (1.0, 1, 0, 0)])
        expected_r = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        assert np.allclose(lut[:, 0], expected_r)
        assert (lut[:, 1] == 0).all()
        assert (lut[:, 2] == 0).all()

    def test_three_stop_midpoint(self):
        """Middle stop at position 0.5 is hit exactly at index 128 (≈0.502)."""
        lut = render.stops_to_lut(
            [(0.0, 0, 0, 0), (0.5, 1, 0, 0), (1.0, 1, 1, 1)]
        )
        # At LUT index where x == 0.5 exactly (no such integer index for
        # 256 samples over [0, 1] inclusive), the closest is index 128
        # (x = 128/255 ≈ 0.502) which is just past the middle stop.
        # Channel R should already be at 1 at the middle stop.
        # Channels G and B should still be very close to 0 at the middle
        # stop and rising linearly toward 1 at the end.
        mid = lut[128]
        assert mid[0] == pytest.approx(1.0, abs=2 / 255)
        assert mid[1] == pytest.approx(0.0, abs=2 / 255)
        assert mid[2] == pytest.approx(0.0, abs=2 / 255)
        # quarter-way past the middle stop -> halfway from red to white
        three_quarter = lut[192]
        assert three_quarter[0] == pytest.approx(1.0, abs=1e-3)
        assert three_quarter[1] == pytest.approx(0.5, abs=2 / 255)
        assert three_quarter[2] == pytest.approx(0.5, abs=2 / 255)

    def test_clamped_to_input_range(self):
        """All LUT values stay within the RGB span of the input stops."""
        stops = [(0.0, 0.1, 0.0, 0.0), (1.0, 0.9, 0.0, 0.0)]
        lut = render.stops_to_lut(stops)
        assert lut[:, 0].min() >= 0.1 - 1e-6
        assert lut[:, 0].max() <= 0.9 + 1e-6

    def test_monotonic_when_stops_are_monotonic(self):
        """Strictly increasing color values produce a strictly non-decreasing
        LUT column."""
        stops = [(0.0, 0.0, 0, 0), (0.5, 0.4, 0, 0), (1.0, 1.0, 0, 0)]
        lut = render.stops_to_lut(stops)
        assert (np.diff(lut[:, 0]) >= 0).all()

    def test_accepts_numpy_array(self):
        stops = np.array(
            [[0.0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32
        )
        lut = render.stops_to_lut(stops)
        assert lut.shape == (256, 3)
        assert np.allclose(lut[-1], [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# _render_multi_channel: LUT path
# ---------------------------------------------------------------------------


class TestRenderSceneLutPath:
    """``_render_multi_channel`` accepts both legacy RGB triplets and the
    new (256, 3) LUT shape. The two paths must agree on solid colors and
    the LUT path must additionally handle non-solid colormaps."""

    def test_accepts_lut_list(self, locs, info):
        """Passing per-channel LUTs returns a valid QImage."""
        lut_red = render.solid_to_lut((1.0, 0.0, 0.0))
        lut_green = render.solid_to_lut((0.0, 1.0, 0.0))
        qimage, n_locs = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[lut_red, lut_green],
        )
        assert isinstance(qimage, QtGui.QImage)
        assert qimage.width() == 32 and qimage.height() == 32
        assert n_locs == 2 * len(locs)

    def test_lut_path_equivalent_to_triplets_for_solid_colors(
        self, locs, info
    ):
        """For solid colors, a black->color LUT must produce the same
        8-bit output (within 1 LSB) as passing the RGB triplet directly."""
        triplets = [(1.0, 0.0, 0.0), (0.0, 0.5, 1.0)]
        luts = [render.solid_to_lut(c) for c in triplets]

        qimage_triplet, _ = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=triplets,
        )
        qimage_lut, _ = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=luts,
        )
        diff = np.abs(
            _qimage_to_array(qimage_triplet).astype(int)
            - _qimage_to_array(qimage_lut).astype(int)
        )
        # LUT path quantizes intensity*255 to 256 entries; expect ≤1 LSB diff.
        assert diff.max() <= 1
        # ...and a strong majority of pixels exactly match.
        assert (diff == 0).mean() > 0.7

    def test_lut_path_isolates_pure_red(self, locs, info):
        """A black->red LUT must leave G and B at zero in the output."""
        lut_red = render.solid_to_lut((1.0, 0.0, 0.0))
        qimage, _ = render.render_scene(
            [locs],
            [info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[lut_red],
        )
        bgra = _qimage_to_array(qimage)
        assert (bgra[..., 0] == 0).all()  # B
        assert (bgra[..., 1] == 0).all()  # G
        assert bgra[..., 2].max() > 0  # R lights up

    def test_lut_path_with_white_endpoint(self, locs, info):
        """A black->color->white gradient (the picasso "built-in colormap"
        shape) produces a saturated output at high intensity, with all
        three channels rising above zero."""
        stops = [
            (0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 1.0),  # blue at midpoint
            (1.0, 1.0, 1.0, 1.0),  # white at peak
        ]
        lut = render.stops_to_lut(stops)
        qimage, _ = render.render_scene(
            [locs],
            [info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[lut],
        )
        bgra = _qimage_to_array(qimage)
        # All three channels should have at least one non-zero pixel:
        # high-intensity samples pull the gradient toward white.
        assert bgra[..., 0].max() > 0  # B
        assert bgra[..., 1].max() > 0  # G
        assert bgra[..., 2].max() > 0  # R

    def test_lut_clipped_at_one(self, locs, info):
        """When two saturated channels overlap, the per-pixel sum is
        clipped to 1.0 → output bytes stay in [0, 255]."""
        lut_red = render.solid_to_lut((1.0, 0.0, 0.0))
        lut_green = render.solid_to_lut((0.0, 1.0, 0.0))
        qimage, _ = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[lut_red, lut_green],
        )
        bgra = _qimage_to_array(qimage)
        # No overflow above uint8 range (trivially true by dtype, but the
        # underlying float32 rgb is clipped to <= 1.0 before to_8bit).
        assert bgra.dtype == np.uint8
        assert bgra[..., :3].max() <= 255

    def test_lut_path_with_raw_image_cache(self, locs, info):
        """LUT shape works with the raw_image_cache fast-redraw path."""
        # First call to populate the cache for two channels.
        _, _, raw = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            return_raw_image=True,
        )
        lut_red = render.solid_to_lut((1.0, 0.0, 0.0))
        lut_green = render.solid_to_lut((0.0, 1.0, 0.0))
        qimage, n_locs = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[lut_red, lut_green],
            raw_image_cache=raw,
        )
        assert isinstance(qimage, QtGui.QImage)
        assert n_locs == 0  # cached path doesn't re-count locs


# ---------------------------------------------------------------------------
# Localization splitting
# ---------------------------------------------------------------------------


class TestSplitLocs:
    def test_by_property_count(self, locs):
        groups = render.split_locs_by_property(
            locs, property_name="photons", n_colors=8
        )
        assert len(groups) == 8

    def test_by_property_total_preserved(self, locs):
        groups = render.split_locs_by_property(
            locs, property_name="photons", n_colors=4
        )
        assert sum(len(g) for g in groups) == len(locs)

    def test_by_property_disjoint(self, locs):
        groups = render.split_locs_by_property(
            locs, property_name="photons", n_colors=4
        )
        index_sets = [set(g.index) for g in groups]
        # pairwise disjoint
        for i in range(len(index_sets)):
            for j in range(i + 1, len(index_sets)):
                assert index_sets[i].isdisjoint(index_sets[j])

    def test_by_property_missing_raises(self, locs):
        with pytest.raises(AssertionError):
            render.split_locs_by_property(
                locs, property_name="not_a_real_column", n_colors=4
            )

    def test_by_group_with_group_column(self, locs):
        df = locs.copy()
        # 3 synthetic groups, round-robin
        df["group"] = np.arange(len(df)) % 3
        groups = render.split_locs_by_group(df)
        assert len(groups) == 3
        assert sum(len(g) for g in groups) == len(df)
        for g in groups:
            unique_groups = g["group"].unique()
            assert len(unique_groups) == 1

    def test_by_group_without_group_column(self, locs):
        groups = render.split_locs_by_group(locs)
        assert len(groups) == 1
        assert len(groups[0]) == len(locs)

    def test_by_group_explicit_array(self, locs):
        n_colors = 4
        rng = np.random.default_rng(0)
        group_color = rng.integers(0, n_colors, size=len(locs))
        groups = render.split_locs_by_group(
            locs, n_colors=n_colors, group_color=group_color
        )
        assert len(groups) == n_colors
        assert sum(len(g) for g in groups) == len(locs)


# ---------------------------------------------------------------------------
# Scalebar
# ---------------------------------------------------------------------------


class TestOptimalScalebar:
    @pytest.mark.parametrize(
        "pixelsize, width, expected",
        [
            (130, 32, 500),  # 130*32/8 = 520 → nearest 100 = 500
            (130, 320, 5000),  # 130*320/8 = 5200 → nearest 1000 = 5000
            (130, 8000, 10000),  # > 10_000
            (1, 240, 30),  # 240/8 = 30 → nearest 10 = 30
            (1, 50, 6),  # 50/8 = 6.25 → 6
        ],
    )
    def test_known_answers(self, pixelsize, width, expected):
        assert render.optimal_scalebar_length(pixelsize, width) == expected


# ---------------------------------------------------------------------------
# render_scene (high-level colored rendering)
# ---------------------------------------------------------------------------


class TestRenderScene:
    def test_single_channel(self, locs, info):
        qimage, n_locs = render.render_scene(
            locs, info, disp_px_size=PIXELSIZE, viewport=FULL_VIEWPORT
        )
        assert isinstance(qimage, QtGui.QImage)
        assert qimage.width() == 32 and qimage.height() == 32
        assert n_locs == len(locs)

    def test_returns_contrast_limits(self, locs, info):
        out = render.render_scene(
            locs,
            info,
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            return_contrast_limits=True,
        )
        assert len(out) == 3
        qimage, n, climits = out
        assert isinstance(qimage, QtGui.QImage)
        assert isinstance(climits, tuple) and len(climits) == 2

    def test_returns_raw_image(self, locs, info):
        out = render.render_scene(
            locs,
            info,
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            return_raw_image=True,
        )
        assert len(out) == 3
        qimage, n, raw = out
        assert raw.ndim == 2
        assert raw.dtype == np.float32
        assert raw.shape == (32, 32)

    def test_multi_channel(self, locs, info):
        qimage, n_locs = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
        assert isinstance(qimage, QtGui.QImage)
        assert n_locs == 2 * len(locs)

    def test_multi_channel_color_isolation(self, locs, info):
        """Pure-red channel must leave G and B at zero in the output."""
        qimage, _ = render.render_scene(
            [locs],
            [info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[(1.0, 0.0, 0.0)],
        )
        bgra = _qimage_to_array(qimage)
        assert (bgra[..., 0] == 0).all()  # B
        assert (bgra[..., 1] == 0).all()  # G
        assert bgra[..., 2].max() > 0  # R lights up

    def test_multi_channel_green_isolation(self, locs, info):
        """Pure-green channel must leave R and B at zero in the output."""
        qimage, _ = render.render_scene(
            [locs],
            [info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[(0.0, 1.0, 0.0)],
        )
        bgra = _qimage_to_array(qimage)
        assert (bgra[..., 0] == 0).all()  # B
        assert (bgra[..., 2] == 0).all()  # R
        assert bgra[..., 1].max() > 0  # G lights up

    def test_empty_locs_list(self, info):
        qimage, n_locs = render.render_scene(
            [], [], disp_px_size=PIXELSIZE, viewport=FULL_VIEWPORT
        )
        assert isinstance(qimage, QtGui.QImage)
        assert n_locs == 0
        assert qimage.width() == 1 and qimage.height() == 1

    def test_with_raw_image_cache(self, locs, info):
        """Passing a raw_image_cache skips rendering; n_locs == 0."""
        _, _, raw = render.render_scene(
            locs,
            info,
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            return_raw_image=True,
        )
        qimage, n_locs = render.render_scene(
            locs,
            info,
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            raw_image_cache=raw,
        )
        assert isinstance(qimage, QtGui.QImage)
        assert n_locs == 0

    def test_background_color_none_equals_black(self, locs, info):
        """``background_color=None`` and an explicit black background must
        be byte-for-byte identical (black is the no-op default)."""
        kwargs = dict(
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        )
        q_none, _ = render.render_scene(
            [locs, locs], [info, info], **kwargs, background_color=None
        )
        q_black, _ = render.render_scene(
            [locs, locs],
            [info, info],
            **kwargs,
            background_color=(0.0, 0.0, 0.0),
        )
        assert np.array_equal(
            _qimage_to_array(q_none), _qimage_to_array(q_black)
        )

    def test_background_color_fills_empty_pixels(self, locs, info):
        """With a white background, pixels with no localizations become
        white while pixels containing localizations do not."""
        qimage, _, raw = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            background_color=(1.0, 1.0, 1.0),
            return_raw_image=True,
        )
        bgra = _qimage_to_array(qimage)
        empty = raw.sum(axis=0) == 0  # (H, W) mask of pixels with no locs
        assert empty.any(), "Sparse test data should leave empty pixels"
        # every empty pixel is fully white
        assert (bgra[empty][:, :3] == 255).all()
        # at least one occupied pixel keeps its channel color (not white)
        occupied = ~empty
        assert occupied.any()
        assert not (bgra[occupied][:, :3] == 255).all()

    def test_background_color_composite_exact(self, locs, info):
        """Exact compositing on a synthetic cache: a saturated channel
        pixel keeps its pure color, an empty pixel shows the background."""
        # channel 0 saturated at (0, 0); everything else empty
        raw = np.zeros((2, 3, 3), dtype=np.float32)
        raw[0, 0, 0] = 1.0
        colors = [
            render.solid_to_lut((1.0, 0.0, 0.0)),
            render.solid_to_lut((0.0, 1.0, 0.0)),
        ]
        qimage, _ = render.render_scene(
            [locs, locs],
            [info, info],
            disp_px_size=PIXELSIZE,
            viewport=FULL_VIEWPORT,
            colors=colors,
            contrast=(0.0, 1.0),
            background_color=(0.2, 0.4, 0.6),
            raw_image_cache=raw,
        )
        bgra = _qimage_to_array(qimage)  # B, G, R, A
        # saturated channel-0 pixel stays pure red (background fully masked)
        assert tuple(int(v) for v in bgra[0, 0, :3]) == (0, 0, 255)
        # empty pixel shows the background color (0.2, 0.4, 0.6) -> bytes
        assert bgra[2, 2, 2] == pytest.approx(round(0.2 * 255), abs=1)  # R
        assert bgra[2, 2, 1] == pytest.approx(round(0.4 * 255), abs=1)  # G
        assert bgra[2, 2, 0] == pytest.approx(round(0.6 * 255), abs=1)  # B

    def test_background_color_default_is_none(self):
        """The public render_scene signature defaults background_color to
        None so existing callers are unaffected."""
        import inspect

        sig = inspect.signature(render.render_scene)
        assert sig.parameters["background_color"].default is None


# ---------------------------------------------------------------------------
# Rectangle pick polygon (pure geometry)
# ---------------------------------------------------------------------------


class TestRectanglePickPolygon:
    def test_polygon_has_four_points(self):
        poly = render.get_rectangle_pick_polygon(0.0, 0.0, 10.0, 0.0, 4.0)
        assert poly.size() == 4

    def test_polygon_opposite_sides_equal_length(self):
        poly = render.get_rectangle_pick_polygon(0.0, 0.0, 10.0, 0.0, 4.0)
        pts = [(poly.at(i).x(), poly.at(i).y()) for i in range(poly.size())]

        def dist(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        side01 = dist(pts[0], pts[1])
        side12 = dist(pts[1], pts[2])
        side23 = dist(pts[2], pts[3])
        side30 = dist(pts[3], pts[0])
        # opposite sides equal
        assert side01 == pytest.approx(side23, abs=1e-6)
        assert side12 == pytest.approx(side30, abs=1e-6)


# ---------------------------------------------------------------------------
# Drawing overlays — light smoke tests (need QGuiApplication)
# ---------------------------------------------------------------------------


def _fresh_canvas():
    img = QtGui.QImage(120, 120, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(0, 0, 0))
    return img


class TestDrawing:
    def test_draw_picks_circle(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas, ((0, 0), (32, 32)), "Circle", [(16, 16)], pick_size=4
        )
        assert isinstance(out, QtGui.QImage)
        assert out.width() == 120 and out.height() == 120
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_rectangle(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Rectangle",
            [((4, 4), (28, 28))],
            pick_size=2,
        )
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_polygon(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Polygon",
            [[(4, 4), (28, 4), (16, 28), (4, 4)]],
            pick_size=1,
        )
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_square(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas, ((0, 0), (32, 32)), "Square", [(16, 16)], pick_size=4
        )
        assert isinstance(out, QtGui.QImage)
        assert out.width() == 120 and out.height() == 120
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_box(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Box",
            [((4, 4), (20, 12))],
            pick_size=None,
        )
        assert isinstance(out, QtGui.QImage)
        assert out.width() == 120 and out.height() == 120
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_box_outside_viewport_is_culled(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Box",
            [((100, 100), (110, 110))],
            pick_size=None,
        )
        assert np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_box_overlapping_viewport_is_drawn(self):
        # culling is on intersection, not on the pick's center: this box
        # is centered at (40, 40), well outside the viewport, but its
        # top left corner reaches into it
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Box",
            [((20, 20), (60, 60))],
            pick_size=None,
        )
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_square_culls_on_its_center(self):
        # contrast with the box above: the click-placed shapes are
        # culled as soon as their center leaves the viewport
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas, ((0, 0), (32, 32)), "Square", [(40, 40)], pick_size=40
        )
        assert np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_brush(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Brush",
            [[(2.0, [(8.0, 8.0), (24.0, 24.0)])]],
            pick_size=None,
        )
        assert isinstance(out, QtGui.QImage)
        assert out.width() == 120 and out.height() == 120
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_brush_outside_viewport_is_culled(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Brush",
            [[(2.0, [(100.0, 100.0), (110.0, 110.0)])]],
            pick_size=None,
        )
        assert np.array_equal(before, _qimage_to_array(out))

    def test_draw_picks_brush_fills_overlaps_once(self):
        # the strokes of a merged pick always overlap, so filling them
        # one by one would leave a darker seam where they cross
        canvas = _fresh_canvas()
        out = render.draw_picks(
            canvas,
            ((0, 0), (32, 32)),
            "Brush",
            [
                [
                    (4.0, [(4.0, 16.0), (28.0, 16.0)]),
                    (4.0, [(16.0, 4.0), (16.0, 28.0)]),
                ]
            ],
            pick_size=None,
        )
        pixels = _qimage_to_array(out)
        center = pixels[60, 60, :3]  # where the two strokes cross
        arm = pixels[60, 30, :3]  # only the horizontal stroke
        np.testing.assert_array_equal(center, arm)

    def test_draw_picks_unknown_shape_raises(self):
        with pytest.raises(ValueError):
            render.draw_picks(
                _fresh_canvas(),
                ((0, 0), (32, 32)),
                "Hexagon",
                [(16, 16)],
                pick_size=4,
            )

    def test_draw_points(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_points(
            canvas, ((0, 0), (32, 32)), [(8, 8), (24, 24)], pixelsize=PIXELSIZE
        )
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_scalebar(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_scalebar(
            canvas,
            ((0, 0), (32, 32)),
            scalebar_length_nm=500,
            pixelsize=PIXELSIZE,
        )
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_legend(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_legend(
            canvas,
            channel_names=["ch1", "ch2"],
            channel_colors=[(255, 0, 0), (0, 255, 0)],
        )
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_minimap(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_minimap(
            canvas, ((10, 10), (20, 20)), max_viewport_size=(32, 32)
        )
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_rotation(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_rotation(canvas, ang=(0.4, 0.3, 0.2))
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))

    def test_draw_rotation_angles(self):
        canvas = _fresh_canvas()
        before = _qimage_to_array(canvas)
        out = render.draw_rotation_angles(canvas, ang=(0.4, 0.3, 0.2))
        assert isinstance(out, QtGui.QImage)
        assert not np.array_equal(before, _qimage_to_array(out))


# ---------------------------------------------------------------------------
# Appearance of the tool overlays (OverlayStyle)
# ---------------------------------------------------------------------------

# 32 camera pixels drawn onto the 120 display pixels of ``_fresh_canvas``
FOV_32 = ((0, 0), (32, 32))


def _lit(image) -> np.ndarray:
    """Mask of the pixels drawn onto the black canvas."""
    return (_qimage_to_array(image)[..., :3] > 0).any(axis=-1)


def _circle(style=None, **kwargs):
    # a circle of diameter 60 display pixels, centered at (60, 60)
    return render.draw_picks(
        _fresh_canvas(),
        FOV_32,
        "Circle",
        [(16, 16)],
        pick_size=16,
        style=style,
        **kwargs,
    )


class TestOverlayStyle:
    def test_default_style_draws_as_before(self):
        # no style and the default style draw the same yellow outline
        np.testing.assert_array_equal(
            _qimage_to_array(_circle()),
            _qimage_to_array(_circle(render.OverlayStyle())),
        )
        red = _qimage_to_array(_circle(color=QtGui.QColor("red")))
        assert red[..., 2].max() == 255  # BGRA: red channel
        assert red[..., 1].max() == 0

    def test_color_argument_overrides_style_color(self):
        style = render.OverlayStyle(color="blue", line_width=3)
        np.testing.assert_array_equal(
            _qimage_to_array(_circle(style, color="red")),
            _qimage_to_array(_circle(render.OverlayStyle("red", "Solid", 3))),
        )

    def test_wider_lines_cover_more_pixels(self):
        thin = _lit(_circle()).sum()
        wide = _lit(_circle(render.OverlayStyle(line_width=4))).sum()
        assert wide > 2.5 * thin

    @pytest.mark.parametrize("line_style", ["Dashed", "Dotted", "Dash-dot"])
    def test_patterned_lines_leave_gaps(self, line_style):
        solid = _lit(_circle()).sum()
        patterned = _lit(_circle(render.OverlayStyle(line_style=line_style)))
        assert 0 < patterned.sum() < 0.9 * solid

    def test_opacity_blends_lines_with_the_image(self):
        pixels = _qimage_to_array(_circle(render.OverlayStyle(opacity=0.5)))
        assert pixels[..., 2].max() == pytest.approx(128, abs=2)

    @pytest.mark.parametrize("shape", ["Circle", "Square"])
    def test_fill_opacity_fills_closed_shapes(self, shape):
        def center(style):
            out = render.draw_picks(
                _fresh_canvas(),
                FOV_32,
                shape,
                [(16, 16)],
                pick_size=16,
                style=style,
            )
            return _qimage_to_array(out)[60, 60, :3]

        assert not center(None).any()  # hollow by default
        filled = center(render.OverlayStyle(fill_opacity=0.5))
        assert filled[2] == pytest.approx(128, abs=2)  # half-bright red
        assert filled[0] == 0  # yellow has no blue

    def test_fill_of_rectangle_and_box(self):
        style = render.OverlayStyle(fill_opacity=0.5)
        for shape, pick, size in (
            ("Rectangle", ((4, 16), (28, 16)), 8),
            ("Box", ((8, 8), (24, 24)), None),
        ):
            out = render.draw_picks(
                _fresh_canvas(), FOV_32, shape, [pick], size, style=style
            )
            # a point inside, off the rectangle's center line
            assert _lit(out)[50, 45], shape

    def test_only_closed_polygons_are_filled(self):
        style = render.OverlayStyle(fill_opacity=0.5)
        triangle = [(4, 4), (28, 4), (16, 28)]

        def inside(pick):
            out = render.draw_picks(
                _fresh_canvas(), FOV_32, "Polygon", [pick], 1, style=style
            )
            return _lit(out)[45, 60]  # the centroid (16, 12)

        assert not inside(triangle)  # still being drawn
        assert inside(triangle + [triangle[0]])

    def test_brush_is_filled_by_default_and_can_be_hollow(self):
        pick = [[(8.0, [(4.0, 16.0), (28.0, 16.0)])]]

        def center(style):
            out = render.draw_picks(
                _fresh_canvas(), FOV_32, "Brush", pick, None, style=style
            )
            return _qimage_to_array(out)[60, 60, 2]

        assert center(None) == pytest.approx(render.BRUSH_FILL_ALPHA, abs=2)
        assert center(render.OverlayStyle(fill_opacity=0)) == 0
        assert center(render.OverlayStyle(fill_opacity=1)) == 255

    def test_font_size_scales_the_annotations(self):
        def lit(font_size):
            style = render.OverlayStyle(font_size=font_size)
            return _lit(_circle(style, annotate_picks=True)).sum()

        outline = _lit(_circle()).sum()
        assert lit(40) - outline > 4 * (lit(10) - outline)

    def test_draw_points_patterns_lines_but_not_crosses(self):
        points = [(4, 16), (28, 16)]  # crosses at x = 15 and 105

        def draw(style):
            out = render.draw_points(
                _fresh_canvas(),
                FOV_32,
                points,
                pixelsize=PIXELSIZE,
                style=style,
            )
            return _lit(out)

        solid = draw(None)
        dashed = draw(render.OverlayStyle(line_style="Dashed"))
        assert solid[60, 30:90].all()
        assert 0 < dashed[60, 30:90].sum() < 60
        # the vertical arm of a cross stays solid
        assert dashed[51:70, 15].all()

    def test_draw_points_font_size(self):
        def lit(font_size):
            out = render.draw_points(
                _fresh_canvas(),
                FOV_32,
                [(4, 4), (8, 4)],
                pixelsize=PIXELSIZE,
                style=render.OverlayStyle(font_size=font_size),
            )
            return _lit(out).sum()

        assert lit(30) > lit(10)
        default = render.draw_points(
            _fresh_canvas(), FOV_32, [(4, 4), (8, 4)], pixelsize=PIXELSIZE
        )
        assert _lit(default).sum() == lit(20)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"line_style": "Wavy"},
            {"line_width": 0},
            {"opacity": 1.5},
            {"fill_opacity": -0.1},
            {"font_size": 0},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValueError):
            render.OverlayStyle(**kwargs)


# ---------------------------------------------------------------------------
# Color bar of a rendered property
# ---------------------------------------------------------------------------


class TestColorbarImage:
    """``render.colorbar_image`` - the LUT saved next to an image that is
    color-coded by a property (e.g., z)."""

    @staticmethod
    def _colors(n=8, cmap="gist_rainbow"):
        return render.get_colors_from_colormap(n, cmap)

    def test_returns_image(self):
        out = render.colorbar_image(self._colors(), -300.0, 300.0)
        assert isinstance(out, QtGui.QImage)
        assert out.width() > 0 and out.height() > 0

    def test_vertical_is_taller_than_wide(self):
        out = render.colorbar_image(self._colors(), 0.0, 1.0)
        assert out.height() > out.width()

    def test_horizontal_is_wider_than_tall(self):
        out = render.colorbar_image(self._colors(), 0.0, 1.0, vertical=False)
        assert out.width() > out.height()

    def test_bands_are_the_rendered_colors(self):
        """Each band of the bar holds the exact color that the matching
        group of localizations is rendered with, the first at the bottom
        of a vertical bar."""
        colors = self._colors(n=4)
        bar_length, bar_width, margin = 400, 40, 12
        out = render.colorbar_image(
            colors,
            0.0,
            1.0,
            bar_length=bar_length,
            bar_width=bar_width,
            margin=margin,
            n_ticks=0,  # no ticks, so the bar starts right at the margin
        )
        array = _qimage_to_array(out)
        x = margin + bar_width // 2  # inside the bar, away from the frame
        for i, color in enumerate(colors):
            # center of band i, counted from the bottom of the bar
            y = margin + bar_length - int((i + 0.5) * bar_length / len(colors))
            expected = [int(round(255 * c)) for c in color]
            # the buffer is BGRA
            assert list(array[y, x, :3][::-1]) == expected

    def test_first_color_is_on_the_left_when_horizontal(self):
        colors = self._colors(n=4)
        bar_length, bar_width, margin = 400, 40, 12
        out = render.colorbar_image(
            colors,
            0.0,
            1.0,
            vertical=False,
            bar_length=bar_length,
            bar_width=bar_width,
            margin=margin,
            n_ticks=0,
        )
        array = _qimage_to_array(out)
        y = margin + bar_width // 2
        x = margin + int(0.5 * bar_length / len(colors))
        expected = [int(round(255 * c)) for c in colors[0]]
        assert list(array[y, x, :3][::-1]) == expected

    def test_background_is_kept(self):
        """The corner of the image shows the requested background, so
        that the bar matches an image rendered on white."""
        out = render.colorbar_image(
            self._colors(),
            0.0,
            1.0,
            color=QtGui.QColor("black"),
            background=QtGui.QColor("white"),
        )
        array = _qimage_to_array(out)
        assert list(array[0, 0, :3]) == [255, 255, 255]

    def test_labels_change_the_size(self):
        """The label and the tick text are given room, rather than being
        drawn over the bar."""
        plain = render.colorbar_image(self._colors(), 0.0, 1.0, n_ticks=0)
        labeled = render.colorbar_image(
            self._colors(), 0.0, 1.0, label="z (nm)", n_ticks=5
        )
        assert labeled.width() > plain.width()
        assert labeled.height() > plain.height()

    def test_single_color(self):
        out = render.colorbar_image(self._colors(n=1), 0.0, 1.0)
        assert isinstance(out, QtGui.QImage)

    def test_equal_limits_draw_no_ticks(self):
        """A degenerate property range must not divide by zero."""
        out = render.colorbar_image(self._colors(), 5.0, 5.0)
        assert isinstance(out, QtGui.QImage)

    def test_rejects_malformed_colors(self):
        with pytest.raises(AssertionError):
            render.colorbar_image([0.1, 0.2, 0.3], 0.0, 1.0)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, "0"),
            (-300.0, "-300"),
            (12.5, "12.5"),
            (0.125, "0.125"),
            (1e6, "1.0e+06"),
            (1e-4, "1.0e-04"),
        ],
    )
    def test_tick_format(self, value, expected):
        assert render._format_tick(value) == expected


class TestSaveColorbar:
    """``render.save_colorbar`` - the same bar written as a raster image
    or as a vector graphic."""

    @staticmethod
    def _colors(n=8, cmap="gist_rainbow"):
        return render.get_colors_from_colormap(n, cmap)

    def test_png(self, tmp_path):
        out = tmp_path / "bar.png"
        render.save_colorbar(
            str(out), colors=self._colors(), min_value=-300.0, max_value=300.0
        )
        assert out.exists() and out.stat().st_size > 0

    def test_svg_is_drawn_not_embedded(self, tmp_path):
        """The bands, ticks and text are vector objects, so the bar can
        be scaled and edited in figure software."""
        out = tmp_path / "bar.svg"
        render.save_colorbar(
            str(out),
            colors=self._colors(n=4),
            min_value=-300.0,
            max_value=300.0,
            label="z (nm)",
        )
        svg = out.read_text()
        assert svg.lstrip().startswith("<?xml")
        # one filled rectangle per color band, plus the frame
        assert svg.count("<rect") >= 5
        assert "z (nm)" in svg  # the label is text, not pixels
        assert "image/png" not in svg  # nothing rasterized

    def test_svg_and_png_have_the_same_size(self, tmp_path):
        colors = self._colors()
        image = render.colorbar_image(colors, 0.0, 1.0, label="z (nm)")
        out = tmp_path / "bar.svg"
        render.colorbar_svg(
            str(out),
            colors=colors,
            min_value=0.0,
            max_value=1.0,
            label="z (nm)",
        )
        svg = out.read_text()
        assert f'width="{image.width()}"' in svg
        assert f'height="{image.height()}"' in svg

    def test_extension_is_case_insensitive(self, tmp_path):
        out = tmp_path / "bar.SVG"
        render.save_colorbar(
            str(out), colors=self._colors(), min_value=0.0, max_value=1.0
        )
        assert "<rect" in out.read_text()


# ---------------------------------------------------------------------------
# QImage export
# ---------------------------------------------------------------------------


class TestExportQImage:
    def test_export_pdf(self, small_qimage, tmp_path):
        out = tmp_path / "out.pdf"
        render.export_qimage_to_pdf(small_qimage, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_svg(self, small_qimage, tmp_path):
        out = tmp_path / "out.svg"
        render.export_qimage_to_svg(small_qimage, str(out))
        assert out.exists()
        assert out.stat().st_size > 0
        head = out.read_bytes()[:200]
        assert b"<svg" in head or b"<?xml" in head


# ---------------------------------------------------------------------------
# rgb_to_qimage
# ---------------------------------------------------------------------------


class TestRgbToQImage:
    def test_round_trip_channels(self):
        """RGB input → QImage → BGRA bytes preserves channel values."""
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[..., 0] = 200  # R
        rgb[..., 1] = 100  # G
        rgb[..., 2] = 50  # B
        qimage = render.rgb_to_qimage(rgb)
        assert isinstance(qimage, QtGui.QImage)
        assert qimage.width() == 6 and qimage.height() == 4
        # _qimage_to_array reads raw memory of Format_RGB32 → bytes are BGRA
        bgra = _qimage_to_array(qimage)
        assert bgra.shape == (4, 6, 4)
        assert (bgra[..., 0] == 50).all()  # B
        assert (bgra[..., 1] == 100).all()  # G
        assert (bgra[..., 2] == 200).all()  # R
        assert (bgra[..., 3] == 255).all()  # A always opaque

    def test_return_bgra(self):
        rgb = np.full((2, 3, 3), 128, dtype=np.uint8)
        qimage, bgra = render.rgb_to_qimage(rgb, return_bgra=True)
        assert isinstance(qimage, QtGui.QImage)
        assert bgra.shape == (2, 3, 4)
        assert bgra.dtype == np.uint8


# ---------------------------------------------------------------------------
# build_animation
# ---------------------------------------------------------------------------


class TestAnimationSequence:
    def test_slerp_endpoints_and_midpoint(self):
        """Default interpolation is slerp: endpoints exact, midpoint at
        half the rotation angle."""
        R1 = Rotation.identity()
        R2 = Rotation.from_rotvec([0.0, 0.0, np.pi / 2])
        positions = [(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT)]
        rotations, viewports = render._animation_sequence(
            positions, [1.0], fps=3
        )
        assert len(rotations) == 3
        assert len(viewports) == 3
        assert (rotations[0] * R1.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )
        assert (rotations[-1] * R2.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )
        assert np.allclose(
            rotations[1].as_rotvec(), [0.0, 0.0, np.pi / 4], atol=1e-9
        )

    def test_full_turn_segment(self):
        """A 360-degree segment rotation spins the full turn even though
        both checkpoints have the same orientation."""
        R = Rotation.identity()
        positions = [(R, FULL_VIEWPORT), (R, FULL_VIEWPORT)]
        rotations, _ = render._animation_sequence(
            positions,
            [1.0],
            fps=5,
            segment_rotations=[np.array([0.0, 0.0, 2 * np.pi])],
        )
        assert len(rotations) == 5
        # quarter of the way through: 90 degrees around z
        assert np.allclose(
            rotations[1].as_rotvec(), [0.0, 0.0, np.pi / 2], atol=1e-9
        )
        # halfway through: 180 degrees around z
        assert rotations[2].magnitude() == pytest.approx(np.pi, abs=1e-9)
        # ends exactly at the checkpoint again
        assert rotations[-1].magnitude() == pytest.approx(0.0, abs=1e-9)

    def test_segment_rotation_snaps_to_checkpoints(self):
        """The segment rotation vector is snapped so that the segment
        ends exactly at the next checkpoint, keeping its turns."""
        R1 = Rotation.identity()
        R2 = Rotation.from_rotvec([0.0, 0.0, np.pi / 2])
        positions = [(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT)]
        # 450 degrees = 90 degrees + 1 full turn; pass a slightly off
        # target to verify snapping
        target = np.array([0.0, 0.0, np.radians(449)])
        rotations, _ = render._animation_sequence(
            positions, [1.0], fps=11, segment_rotations=[target]
        )
        assert (rotations[-1] * R2.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )
        # total rotation path is 450 degrees: 1/5 of the way is 90 deg
        assert np.allclose(
            rotations[2].as_rotvec(), [0.0, 0.0, np.pi / 2], atol=1e-6
        )

    def test_full_turn_survives_off_axis_residual(self):
        """A segment of more than one turn keeps its turns even when
        the two checkpoints are nearly the same orientation (the turn
        cannot be read off the checkpoints, so the given path defines
        it)."""
        R1 = Rotation.identity()
        # a wobbly full turn plus 18 degrees around y: the checkpoints
        # differ by a small, mostly off-axis rotation
        R2 = Rotation.from_rotvec(np.radians([-0.8, 18.0, -0.5]))
        segment = np.radians([-0.8, 378.0, -0.5])
        rotations, _ = render._animation_sequence(
            positions=[(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT)],
            durations=[1.0],
            fps=60,
            segment_rotations=[segment],
        )
        swept = sum(
            (rotations[i + 1] * rotations[i].inv()).magnitude()
            for i in range(len(rotations) - 1)
        )
        assert np.degrees(swept) == pytest.approx(378.0, abs=1.0)
        assert (rotations[-1] * R2.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )

    def test_segments_do_not_repeat_checkpoints(self):
        """Every checkpoint is rendered once, so no frame is held twice
        at the junction between two segments."""
        R1 = Rotation.identity()
        R2 = Rotation.from_rotvec([0.0, 0.0, np.pi / 2])
        R3 = Rotation.from_rotvec([0.0, 0.0, np.pi])
        positions = [
            (R1, FULL_VIEWPORT),
            (R2, FULL_VIEWPORT),
            (R3, FULL_VIEWPORT),
        ]
        rotations, viewports = render._animation_sequence(
            positions, [1.0, 1.0], fps=4
        )
        assert len(rotations) == 8
        assert len(viewports) == 8
        steps = [
            (rotations[i + 1] * rotations[i].inv()).magnitude()
            for i in range(len(rotations) - 1)
        ]
        assert min(steps) > 1e-9
        assert (rotations[-1] * R3.inv()).magnitude() == pytest.approx(
            0.0, abs=1e-9
        )

    def test_multi_segment_viewports(self):
        """Viewports interpolate linearly per segment."""
        R = Rotation.identity()
        vp1 = ((0.0, 0.0), (32.0, 32.0))
        vp2 = ((8.0, 8.0), (24.0, 24.0))
        positions = [(R, vp1), (R, vp2)]
        _, viewports = render._animation_sequence(positions, [1.0], fps=3)
        assert np.allclose(viewports[0], vp1)
        assert np.allclose(viewports[1], ((4.0, 4.0), (28.0, 28.0)))
        assert np.allclose(viewports[-1], vp2)

    @staticmethod
    def _velocities(rotations, fps):
        """World-frame angular velocity (rad/s) between frames."""
        return (
            np.array(
                [
                    (rotations[i + 1] * rotations[i].inv()).as_rotvec()
                    for i in range(len(rotations) - 1)
                ]
            )
            * fps
        )

    def _turning_sequence(self, fps, transition):
        """Three segments whose directions change at each checkpoint,
        the last one spinning more than a full turn."""
        segments = [
            np.radians([0.0, 0.0, 90.0]),
            np.radians([0.0, 50.0, 80.0]),
            np.radians([60.0, 20.0, 420.0]),
        ]
        checkpoints = [Rotation.identity()]
        for segment in segments:
            checkpoints.append(Rotation.from_rotvec(segment) * checkpoints[-1])
        rotations, _ = render._animation_sequence(
            [(R, FULL_VIEWPORT) for R in checkpoints],
            [1.0, 1.0, 2.0],
            fps,
            segment_rotations=segments,
            transition=transition,
        )
        return rotations, checkpoints

    @pytest.mark.parametrize("transition", ["smooth", "ease"])
    def test_eased_transitions_hit_checkpoints(self, transition):
        """Eased paths pass exactly through every checkpoint, start and
        end at rest and keep the requested full turn."""
        fps = 120
        rotations, checkpoints = self._turning_sequence(fps, transition)
        for frame, R in zip((0, fps, 2 * fps, -1), checkpoints):
            assert (rotations[frame] * R.inv()).magnitude() == pytest.approx(
                0.0, abs=1e-9
            )
        speeds = np.linalg.norm(self._velocities(rotations, fps), axis=1)
        assert speeds[0] < 0.05 * speeds.max()
        assert speeds[-1] < 0.05 * speeds.max()
        assert np.degrees(speeds.sum() / fps) > 560

    def test_smooth_velocity_is_continuous(self):
        """Unlike constant speed, the smooth path has no velocity jump
        at the intermediate checkpoints: the largest frame-to-frame
        change in velocity shrinks with the frame rate."""

        def max_jump(fps, transition):
            rotations, _ = self._turning_sequence(fps, transition)
            velocities = self._velocities(rotations, fps)
            return np.linalg.norm(np.diff(velocities, axis=0), axis=1).max()

        assert max_jump(960, "smooth") < 0.3 * max_jump(240, "smooth")
        # the constant-speed path jumps by the same amount at any rate
        assert max_jump(960, "linear") == pytest.approx(
            max_jump(240, "linear"), rel=0.05
        )
        assert max_jump(960, "smooth") < 0.05 * max_jump(960, "linear")

    def test_ease_rests_at_every_checkpoint(self):
        fps = 120
        rotations, _ = self._turning_sequence(fps, "ease")
        speeds = np.linalg.norm(self._velocities(rotations, fps), axis=1)
        for frame in (fps, 2 * fps):
            assert speeds[frame] < 0.05 * speeds.max()

    def test_smooth_rests_before_a_stay(self):
        """The motion comes to rest at a checkpoint followed by a stay
        segment instead of overshooting and coming back."""
        R1 = Rotation.identity()
        R2 = Rotation.from_rotvec([0.0, 0.0, np.pi / 2])
        fps = 100
        rotations, _ = render._animation_sequence(
            [(R1, FULL_VIEWPORT), (R2, FULL_VIEWPORT), (R2, FULL_VIEWPORT)],
            [1.0, 1.0],
            fps,
            transition="smooth",
        )
        for R in rotations[fps:]:
            assert (R * R2.inv()).magnitude() == pytest.approx(0.0, abs=1e-9)
        angles = [R.magnitude() for R in rotations[: fps + 1]]
        assert np.all(np.diff(angles) >= -1e-12)

    def test_smooth_viewport_zooms_geometrically(self):
        """Eased viewports interpolate the size logarithmically: halfway
        through a 4x zoom the view is 2x zoomed, centered in between."""
        R = Rotation.identity()
        vp1 = ((0.0, 0.0), (32.0, 32.0))
        vp2 = ((8.0, 8.0), (16.0, 16.0))
        _, viewports = render._animation_sequence(
            [(R, vp1), (R, vp2)], [1.0], fps=3, transition="smooth"
        )
        assert np.allclose(viewports[0], vp1)
        assert np.allclose(viewports[1], ((6.0, 6.0), (22.0, 22.0)))
        assert np.allclose(viewports[-1], vp2)

    def test_normalize_legacy_positions_raise(self):
        """Legacy Euler positions were removed in v0.12.0; the error
        names the replacement."""
        legacy = [(0.1, 0.2, 0.3, FULL_VIEWPORT)]
        with pytest.raises(ValueError, match="rotation_matrix"):
            render._normalize_animation_positions(legacy)

    def test_normalize_invalid_position_raises(self):
        with pytest.raises(ValueError):
            render._normalize_animation_positions([(0.1, FULL_VIEWPORT)])


class TestBuildAnimation:
    def test_smoke_quaternions_with_turns(self, locs_3d, info, tmp_path):
        """An animation from scipy Rotations with a multi-turn segment
        writes both an .mp4 and the sidecar .yaml."""
        out_path = tmp_path / "anim.mp4"
        positions = [
            (Rotation.identity(), FULL_VIEWPORT),
            (Rotation.from_rotvec([0.1, 0.0, 0.0]), FULL_VIEWPORT),
        ]
        render.build_animation(
            str(out_path),
            locs_3d,
            info,
            positions=positions,
            durations=[1.0],
            segment_rotations=[np.array([0.1 + 2 * np.pi, 0.0, 0.0])],
            disp_px_size=PIXELSIZE,
            image_size=(64, 64),
            fps=2,
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        yaml_path = out_path.with_suffix(".yaml")
        assert yaml_path.exists()
        assert yaml_path.stat().st_size > 0

    def test_reports_progress_and_completion(self, locs_3d, info, tmp_path):
        out_path = tmp_path / "anim.mp4"
        positions = [
            (Rotation.identity(), FULL_VIEWPORT),
            (Rotation.from_rotvec([0.1, 0.0, 0.0]), FULL_VIEWPORT),
        ]
        frames = []
        completed = render.build_animation(
            str(out_path),
            locs_3d,
            info,
            positions=positions,
            durations=[1.0],
            disp_px_size=PIXELSIZE,
            image_size=(64, 64),
            fps=3,
            progress_callback=frames.append,
        )
        assert completed is True
        assert frames == [0, 1, 2, 3]  # each frame, then the total

    def test_cancel_leaves_no_partial_output(self, locs_3d, info, tmp_path):
        """A build cancelled part-way removes the incomplete video and
        never writes the sidecar, and reports that it did not finish."""
        out_path = tmp_path / "anim.mp4"
        positions = [
            (Rotation.identity(), FULL_VIEWPORT),
            (Rotation.from_rotvec([0.1, 0.0, 0.0]), FULL_VIEWPORT),
        ]
        rendered = []
        completed = render.build_animation(
            str(out_path),
            locs_3d,
            info,
            positions=positions,
            durations=[2.0],
            disp_px_size=PIXELSIZE,
            image_size=(64, 64),
            fps=5,
            progress_callback=rendered.append,
            cancel=lambda: len(rendered) >= 3,
        )
        assert completed is False
        assert len(rendered) < 10  # stopped early
        assert not out_path.exists()
        assert not out_path.with_suffix(".yaml").exists()

    def test_transition_saved_and_validated(self, locs_3d, info, tmp_path):
        out_path = tmp_path / "anim.mp4"
        kwargs = dict(
            positions=[
                (Rotation.identity(), FULL_VIEWPORT),
                (Rotation.from_rotvec([0.1, 0.0, 0.0]), FULL_VIEWPORT),
            ],
            durations=[1.0],
            disp_px_size=PIXELSIZE,
            image_size=(64, 64),
            fps=2,
        )
        with pytest.raises(AssertionError, match="transition"):
            render.build_animation(
                str(out_path), locs_3d, info, transition="cubic", **kwargs
            )
        render.build_animation(
            str(out_path), locs_3d, info, transition="smooth", **kwargs
        )
        settings = io.load_info(str(out_path.with_suffix(".yaml")))[0]
        assert settings["Transition"] == "smooth"


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMasking:
    @pytest.mark.parametrize("method", MASKING_METHODS)
    def test_mask_image_methods(self, image, method):
        mask, _ = masking.mask_image(image, method=method)
        assert mask.shape == image.shape
        assert mask.dtype == bool
        assert (
            mask.sum() > 0
        ), f"Method {method!r} produced an empty mask on the test image"

    def test_mask_locs_partitions_input(self, locs, info, image):
        mask, _ = masking.mask_image(image, method="otsu")
        locs_in, locs_out = masking.mask_locs(locs, info, mask)
        assert len(locs_in) + len(locs_out) == len(locs)
        # in / out are disjoint
        assert set(locs_in.index).isdisjoint(set(locs_out.index))
        # both partitions only contain valid loc indices
        all_idx = set(locs_in.index) | set(locs_out.index)
        assert all_idx == set(locs.index)


# ---------------------------------------------------------------------------
# Rendering purity: inputs are never mutated, repeat calls are identical
# ---------------------------------------------------------------------------


PURITY_ANG = (0.35, -0.6, 0.8)


def _column_snapshot(df):
    return {name: df[name].to_numpy().copy() for name in df.columns}


def _assert_columns_unchanged(df, snapshot):
    for name, before in snapshot.items():
        np.testing.assert_array_equal(
            df[name].to_numpy(),
            before,
            err_msg=f"rendering mutated input column {name!r}",
        )


class TestRenderPurity:
    """Rendering must never mutate its inputs, so calling it twice with
    the same data gives bit-identical images.

    Before the in-place ``z /= pixelsize`` was removed from
    ``_render_setup3d(_anisotropic)``, this held for the 3D histogram
    renderers only if every caller defensively copied its z array.
    """

    @pytest.mark.parametrize("blur_method", [None] + BLUR_METHODS)
    def test_render_2d(self, locs, info, blur_method):
        snapshot = _column_snapshot(locs)
        kwargs = dict(
            disp_px_size=PIXELSIZE / 10,
            viewport=FULL_VIEWPORT,
            blur_method=blur_method,
        )
        n1, image1 = render.render(locs, info, **kwargs)
        n2, image2 = render.render(locs, info, **kwargs)
        _assert_columns_unchanged(locs, snapshot)
        assert n1 == n2
        np.testing.assert_array_equal(image1, image2)

    @pytest.mark.parametrize("blur_method", [None] + BLUR_METHODS)
    def test_render_rotated(self, locs_3d, info, blur_method):
        snapshot = _column_snapshot(locs_3d)
        kwargs = dict(
            disp_px_size=PIXELSIZE / 10,
            viewport=FULL_VIEWPORT,
            blur_method=blur_method,
            ang=PURITY_ANG,
        )
        n1, image1 = render.render(locs_3d, info, **kwargs)
        n2, image2 = render.render(locs_3d, info, **kwargs)
        _assert_columns_unchanged(locs_3d, snapshot)
        assert n1 == n2
        np.testing.assert_array_equal(image1, image2)

    def test_render_hist3d(self, locs_3d):
        x = locs_3d["x"].to_numpy()
        y = locs_3d["y"].to_numpy()
        z = locs_3d["z"].to_numpy()
        before = (x.copy(), y.copy(), z.copy())
        args = (10, 0, 0, 32, 32, -100.0, 100.0, PIXELSIZE)
        n1, image1 = render.render_hist3d(x, y, z, *args)
        n2, image2 = render.render_hist3d(x, y, z, *args)
        for arr, orig in zip((x, y, z), before):
            np.testing.assert_array_equal(arr, orig)
        assert n1 == n2
        np.testing.assert_array_equal(image1, image2)

    def test_render_hist3d_anisotropic(self, locs_3d):
        x = locs_3d["x"].to_numpy()
        y = locs_3d["y"].to_numpy()
        z = locs_3d["z"].to_numpy()
        before = (x.copy(), y.copy(), z.copy())
        args = (10, 10, 5, 0, 0, 32, 32, -100.0, 100.0, PIXELSIZE)
        n1, image1 = render.render_hist3d_anisotropic(x, y, z, *args)
        n2, image2 = render.render_hist3d_anisotropic(x, y, z, *args)
        for arr, orig in zip((x, y, z), before):
            np.testing.assert_array_equal(arr, orig)
        assert n1 == n2
        np.testing.assert_array_equal(image1, image2)


# ---------------------------------------------------------------------------
# Parallel channel rendering
# ---------------------------------------------------------------------------


class TestParallelChannels:
    """Chunked parallel rendering must match the sequential result,
    stay deterministic, and respect the user's render CPU budget."""

    CHANNEL_COLORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    def _force_budget(self, monkeypatch, n):
        monkeypatch.setattr(
            lib.io,
            "load_user_settings",
            lambda: {"Render": {"cpu_utilization": 0.9, "max_workers": n}},
        )

    def _scene_kwargs(self, rotated, blur="gaussian"):
        return dict(
            disp_px_size=PIXELSIZE / 10,
            colors=self.CHANNEL_COLORS,
            viewport=FULL_VIEWPORT,
            blur_method=blur,
            min_blur_width=0.0,
            ang=PURITY_ANG if rotated else None,
            contrast=(0.0, 5.0),
        )

    @pytest.mark.parametrize("blur", ["gaussian", "smooth"])
    @pytest.mark.parametrize("rotated", [False, True])
    def test_parallel_matches_sequential(
        self, locs, locs_3d, info, monkeypatch, rotated, blur
    ):
        # tiny chunk floor so the small fixture actually chunks
        monkeypatch.setattr(render.splat, "_MIN_CHUNK_LOCS", 50)
        source = locs_3d if rotated else locs
        channels = [source.iloc[i::3] for i in range(3)]
        kwargs = self._scene_kwargs(rotated, blur)
        self._force_budget(monkeypatch, 1)
        n_seq, rgb_seq, _, raw_seq = render._render_multi_channel(
            channels, [info] * 3, **kwargs
        )
        self._force_budget(monkeypatch, 3)
        n_par, rgb_par, _, raw_par = render._render_multi_channel(
            channels, [info] * 3, **kwargs
        )
        assert n_seq == n_par
        # chunk summing reorders float additions: equal to tolerance,
        # not bit-exact
        np.testing.assert_allclose(raw_par, raw_seq, rtol=1e-5, atol=1e-6)
        diff = np.abs(rgb_par.astype(np.int16) - rgb_seq.astype(np.int16))
        assert diff.max() <= 1

    def test_parallel_is_deterministic(self, locs, info, monkeypatch):
        monkeypatch.setattr(render.splat, "_MIN_CHUNK_LOCS", 50)
        self._force_budget(monkeypatch, 3)
        channels = [locs.iloc[i::3] for i in range(3)]

        def run():
            return render._render_multi_channel(
                channels, [info] * 3, **self._scene_kwargs(rotated=False)
            )

        _, rgb1, _, raw1 = run()
        _, rgb2, _, raw2 = run()
        np.testing.assert_array_equal(raw1, raw2)
        np.testing.assert_array_equal(rgb1, rgb2)

    def test_small_single_channel_parses_settings_at_most_once(
        self, locs, info, monkeypatch
    ):
        # backend selection reads the settings through an mtime cache;
        # the CPU pool itself is not consulted for a small channel
        calls = []

        def counting():
            calls.append(1)
            return {"Render": {"gpu": {"enabled": "off"}}}

        monkeypatch.setattr(lib.io, "load_user_settings", counting)
        for _ in range(3):
            ((n, image),) = render._render_channels(
                [locs],
                [info],
                disp_px_size=PIXELSIZE / 10,
                viewport=FULL_VIEWPORT,
                blur_method="gaussian",
                min_blur_width=0.0,
                ang=None,
            )
            assert n > 0
        assert len(calls) <= 1

    def test_convolve_never_chunks_but_gaussian_does(
        self, locs, info, monkeypatch
    ):
        monkeypatch.setattr(render.splat, "_MIN_CHUNK_LOCS", 10)
        self._force_budget(monkeypatch, 4)
        calls = []
        original = render._render_arrays

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        # patch where CpuBackend looks the name up (splat's global)
        monkeypatch.setattr(render.splat, "_render_arrays", counting)
        channels = [locs.iloc[i::2] for i in range(2)]
        common = dict(
            disp_px_size=PIXELSIZE / 10,
            viewport=FULL_VIEWPORT,
            min_blur_width=0.0,
            ang=None,
        )
        render._render_channels(
            channels, [info] * 2, blur_method="convolve", **common
        )
        assert len(calls) == 2  # one render per channel, no chunking
        calls.clear()
        render._render_channels(
            channels, [info] * 2, blur_method="gaussian", **common
        )
        assert len(calls) > 2  # per-loc blur chunks

    def test_worker_budget_caps(self, monkeypatch):
        self._force_budget(monkeypatch, 2)
        assert render._render_worker_budget() == 2

    def test_chunk_tasks_plan(self):
        sizes = [1_000_000, 100_000]
        tasks = render._chunk_tasks(sizes, budget=4)
        by_channel = {}
        for i, start, stop in tasks:
            by_channel.setdefault(i, []).append((start, stop))
        # the big channel splits, the small one stays whole
        assert len(by_channel[0]) > 1
        assert by_channel[1] == [(0, 100_000)]
        # the chunks of each channel tile it exactly, in row order
        for i, n in enumerate(sizes):
            spans = by_channel[i]
            assert spans[0][0] == 0 and spans[-1][1] == n
            for (_, stop), (start, _) in zip(spans, spans[1:]):
                assert stop == start
        # every *split* chunk respects the minimum size; only a whole
        # small channel may fall below it
        assert all(
            stop - start >= render._MIN_CHUNK_LOCS
            or (start, stop) == (0, sizes[i])
            for i, start, stop in tasks
        )


# ---------------------------------------------------------------------------
# Splat backend seam
# ---------------------------------------------------------------------------


class TestSplatBackend:
    """_render_channels extracts columns once and routes them through
    the backend selected by backend._get_backend(); a failing non-CPU
    backend falls back to the CPU reference without crashing."""

    KWARGS = dict(
        disp_px_size=PIXELSIZE / 10,
        viewport=FULL_VIEWPORT,
        blur_method="gaussian",
        min_blur_width=0.0,
        ang=None,
    )

    def _settings(self, monkeypatch, gpu):
        monkeypatch.setattr(
            lib.io, "load_user_settings", lambda: {"Render": {"gpu": gpu}}
        )

    def test_cpu_backend_when_gpu_is_off(self, monkeypatch):
        self._settings(monkeypatch, {"enabled": "off"})
        chosen = render.backend._get_backend()
        assert isinstance(chosen, render.CpuBackend)
        assert isinstance(chosen, render.SplatBackend)
        assert chosen is render.backend._cpu_backend()
        assert chosen is render.backend._get_backend()

    def test_render_channels_routes_through_selected_backend(
        self, locs, info, monkeypatch
    ):
        received = {}

        class Fake(render.SplatBackend):
            name = "fake"

            def render_channels(self, columns, info_arg, **kwargs):
                received["columns"] = columns
                received["kwargs"] = kwargs
                return render.backend._cpu_backend().render_channels(
                    columns, info_arg, **kwargs
                )

        fake = Fake()
        monkeypatch.setattr(render.scene, "_get_backend", lambda **kw: fake)
        renderings = render._render_channels([locs], [info], **self.KWARGS)
        assert len(renderings) == 1
        # the seam passes extracted column arrays, not DataFrames
        assert all(
            isinstance(c, render._RenderColumns) for c in received["columns"]
        )
        assert received["kwargs"]["blur_method"] == "gaussian"

    def test_failing_backend_falls_back_to_cpu(
        self, locs, info, monkeypatch, caplog
    ):
        class Failing(render.SplatBackend):
            name = "failing"

            def render_channels(self, columns, info_arg, **kwargs):
                raise render.SplatBackendError("no device")

        failing = Failing()
        monkeypatch.setattr(render.scene, "_get_backend", lambda **kw: failing)
        with caplog.at_level(logging.WARNING, logger="picasso.render.scene"):
            renderings = render._render_channels([locs], [info], **self.KWARGS)
        ((n, image),) = renderings
        n_ref, image_ref = render.render(locs, info, **self.KWARGS)
        assert n == n_ref
        np.testing.assert_array_equal(image, image_ref)
        assert any("failing" in record.message for record in caplog.records)

    def test_cpu_backend_has_no_resident_uploads(self, monkeypatch):
        assert render.CpuBackend.persistent_uploads is False
        assert render.backend._cpu_backend().persistent_uploads is False
        # releasing is always safe, GPU backend or not
        monkeypatch.setattr(render.backend, "_gpu_singleton", None)
        render.backend.release_uploads()

    def test_vram_budget_setting(self, monkeypatch):
        default = lib.RENDER_VRAM_BUDGET_MB_DEFAULT * 2**20

        def with_value(value):
            settings = {"Render": {"gpu": {"vram_budget_mb": value}}}
            monkeypatch.setattr(lib.io, "load_user_settings", lambda: settings)
            return render.backend.vram_budget_bytes()

        assert with_value(512) == 512 * 2**20
        assert with_value(1.5) == int(1.5 * 2**20)
        assert with_value(0) is None  # unlimited
        assert with_value(-1) == default
        assert with_value(True) == default
        assert with_value("lots") == default
        monkeypatch.setattr(lib.io, "load_user_settings", lambda: {})
        assert render.backend.vram_budget_bytes() == default

    def test_gpu_settings_parsing(self, monkeypatch):
        default_budget = lib.RENDER_VRAM_BUDGET_MB_DEFAULT * 2**20

        def parsed(gpu):
            self._settings(monkeypatch, gpu)
            return render.backend.gpu_settings()

        assert parsed({}) == {
            "enabled": "auto",
            "adapter": "high-performance",
            "vram_budget_bytes": default_budget,
        }
        assert parsed({"enabled": "ON"})["enabled"] == "on"
        # YAML's bare on/off parse as booleans
        assert parsed({"enabled": True})["enabled"] == "on"
        assert parsed({"enabled": False})["enabled"] == "off"
        assert parsed({"enabled": "maybe"})["enabled"] == "auto"
        assert parsed({"adapter": " NVIDIA "})["adapter"] == "NVIDIA"
        assert parsed({"adapter": ""})["adapter"] == "high-performance"
        assert parsed({"vram_budget_mb": 0})["vram_budget_bytes"] is None
        assert parsed("nonsense")["enabled"] == "auto"
        monkeypatch.setattr(lib.io, "load_user_settings", lambda: {})
        assert render.backend.gpu_settings()["enabled"] == "auto"

    def test_selection_honors_enabled_and_size(self, monkeypatch):
        class FakeGpu(render.SplatBackend):
            name = "fake-gpu"
            persistent_uploads = True

            def render_channels(self, *args, **kwargs):
                raise NotImplementedError

        fake = FakeGpu()
        requested = []

        def fake_gpu_backend(adapter, warn):
            requested.append((adapter, warn))
            return fake

        monkeypatch.setattr(render.backend, "_gpu_backend", fake_gpu_backend)
        cpu = render.backend._cpu_backend()
        # off: the GPU is never even probed
        self._settings(monkeypatch, {"enabled": "off"})
        assert render.backend._get_backend() is cpu
        assert render.backend._get_backend(n_locs=10**7) is cpu
        assert requested == []
        # auto / on: the GPU for large requests, the CPU for tiny ones
        self._settings(monkeypatch, {"enabled": "auto", "adapter": "Intel"})
        assert render.backend._get_backend() is fake
        assert render.backend._get_backend(n_locs=10**7) is fake
        assert requested[-1] == ("Intel", False)
        assert (
            render.backend._get_backend(n_locs=lib.RENDER_GPU_MIN_LOCS - 1)
            is cpu
        )
        self._settings(monkeypatch, {"enabled": "on"})
        assert render.backend._get_backend() is fake
        assert requested[-1] == ("high-performance", True)
        # an unavailable GPU means the CPU
        monkeypatch.setattr(
            render.backend, "_gpu_backend", lambda adapter, warn: None
        )
        assert render.backend._get_backend() is cpu
        assert render.backend.describe_active().startswith("CPU (")

    def test_convolve_blur_is_the_global_precision(self, locs, info):
        """'convolve' blurs with the caller's global precision, else with
        the median precision of the rows rendered (not of those in
        view), so the blur is the same at every zoom and rotation."""
        zoomed = ((8.0, 8.0), (20.0, 20.0))
        kwargs = dict(disp_px_size=PIXELSIZE / 4, blur_method="convolve")
        n, default = render.render(locs, info, viewport=zoomed, **kwargs)
        medians = (
            float(np.median(locs["lpx"])),
            float(np.median(locs["lpy"])),
        )
        _, explicit = render.render(
            locs, info, viewport=zoomed, global_precision=medians, **kwargs
        )
        np.testing.assert_array_equal(default, explicit)
        # an explicit blur is honored: wider blurs are flatter, and the
        # intensity is conserved
        _, narrow = render.render(
            locs, info, viewport=zoomed, global_precision=(0.5, 0.5), **kwargs
        )
        _, wide = render.render(
            locs, info, viewport=zoomed, global_precision=(2.0, 2.0), **kwargs
        )
        assert wide.max() < narrow.max() < default.max()
        # a wider blur spills a little more intensity over the border
        assert 0.9 * narrow.sum() < wide.sum() <= narrow.sum()
        # the scene entry point takes one pair per channel
        _, _, raw = render.render_scene(
            [locs, locs],
            [info, info],
            viewport=zoomed,
            global_precision=[(0.5, 0.5), (2.0, 2.0)],
            return_raw_image=True,
            **kwargs,
        )
        np.testing.assert_array_equal(raw[0], narrow)
        np.testing.assert_array_equal(raw[1], wide)

    def test_rotated_renders_reach_the_gpu_sooner(
        self, locs_3d, info, monkeypatch
    ):
        # a rotated 3D localization costs the CPU ~20x a 2D one, so the
        # small-render cutoff counts it that many times
        cpu = render.backend._cpu_backend()
        served = []

        class FakeGpu(render.SplatBackend):
            name = "fake-gpu"

            def render_channels(self, columns, info_arg, **kwargs):
                served.append(sum(len(c) for c in columns))
                return cpu.render_channels(columns, info_arg, **kwargs)

        fake = FakeGpu()
        monkeypatch.setattr(
            render.backend, "_gpu_backend", lambda adapter, warn: fake
        )
        self._settings(monkeypatch, {"enabled": "auto"})
        n = lib.RENDER_GPU_MIN_LOCS // lib.RENDER_ROTATED_COST_FACTOR + 1
        repeats = -(-n // len(locs_3d))  # the fixture is smaller than n
        small = pd.concat([locs_3d] * repeats).iloc[:n].reset_index(drop=True)
        assert len(small) == n < lib.RENDER_GPU_MIN_LOCS
        assert n * lib.RENDER_ROTATED_COST_FACTOR >= lib.RENDER_GPU_MIN_LOCS
        kwargs = dict(
            disp_px_size=PIXELSIZE / 4,
            viewport=((0.0, 0.0), (32.0, 32.0)),
            blur_method="gaussian",
            min_blur_width=0.0,
        )
        render.scene._render_channels([small], [info], ang=None, **kwargs)
        assert served == []  # 2D: below the cutoff, the CPU
        render.scene._render_channels(
            [small], [info], ang=(0.3, 0.2, 0.1), **kwargs
        )
        assert served == [n]  # rotated: weighted past the cutoff

    @pytest.mark.gpu_backend
    def test_unavailable_gpu_logs_once_per_preference(
        self, monkeypatch, caplog
    ):
        import builtins

        real_import = builtins.__import__

        def no_wgpu(name, *args, **kwargs):
            if name.startswith("picasso.render.gpu") or name == "wgpu":
                raise ImportError("no wgpu here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_wgpu)
        monkeypatch.setattr(render.backend, "_gpu_singleton", None)
        monkeypatch.setattr(render.backend, "_gpu_unavailable", False)
        monkeypatch.setattr(render.backend, "_gpu_adapter", None)
        with caplog.at_level(logging.INFO, logger="picasso.render.backend"):
            assert (
                render.backend._gpu_backend("high-performance", warn=True)
                is None
            )
            assert (
                render.backend._gpu_backend("high-performance", warn=True)
                is None
            )
        unavailable = [r for r in caplog.records if "unavailable" in r.message]
        assert len(unavailable) == 1
        assert unavailable[0].levelno == logging.WARNING

    def test_concurrent_renders_are_correct(self, locs, info):
        # contract: render_channels may be called from several threads
        # at once (async worker + a synchronous render)
        from concurrent import futures

        cpu = render.backend._cpu_backend()
        channels = [locs.iloc[i::2] for i in range(2)]
        columns = [
            render._extract_render_columns(channel, "gaussian", None)
            for channel in channels
        ]
        reference = cpu.render_channels(columns, [info] * 2, **self.KWARGS)
        with futures.ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(
                    lambda _: cpu.render_channels(
                        columns, [info] * 2, **self.KWARGS
                    ),
                    range(2),
                )
            )
        for renderings in results:
            for (n, image), (n_ref, image_ref) in zip(renderings, reference):
                assert n == n_ref
                np.testing.assert_array_equal(image, image_ref)


# ---------------------------------------------------------------------------
# Row selections (indices) through the render API
# ---------------------------------------------------------------------------


class TestIndexedRender:
    """``indices`` render a selection of rows, equivalently to slicing
    the DataFrame first, without copying the columns up front."""

    KWARGS = dict(
        disp_px_size=PIXELSIZE / 10,
        viewport=FULL_VIEWPORT,
        min_blur_width=0.0,
        ang=None,
    )

    @pytest.fixture
    def selection(self, locs):
        rng = np.random.default_rng(11)
        return np.sort(
            rng.choice(len(locs), size=len(locs) // 3, replace=False)
        ).astype(np.uint32)

    @pytest.mark.parametrize("blur", [None, "gaussian", "smooth"])
    def test_render_matches_sliced_locs(self, locs, info, selection, blur):
        n_ref, image_ref = render.render(
            locs.iloc[selection], info, blur_method=blur, **self.KWARGS
        )
        n, image = render.render(
            locs, info, blur_method=blur, indices=selection, **self.KWARGS
        )
        assert n == n_ref
        np.testing.assert_array_equal(image, image_ref)

    def test_columns_keep_the_full_arrays(self, locs, selection):
        columns = render._extract_render_columns(
            locs, "gaussian", None, indices=selection
        )
        assert len(columns) == len(selection)
        assert len(columns.x) == len(locs)  # no copy of the columns
        assert columns.indices.dtype == np.uint32
        part = columns.slice(5, 50)
        assert len(part) == 45 and part.x is columns.x
        dense = columns.materialize()
        assert dense.indices is None and len(dense.x) == len(selection)
        np.testing.assert_array_equal(dense.x, locs["x"].to_numpy()[selection])

    def test_max_blur_width_applies_to_the_selection(self, locs, selection):
        whaled = locs.copy()
        whaled.loc[whaled.index[selection[:10]], "lpx"] = 50.0
        columns = render._extract_render_columns(
            whaled, "gaussian", None, max_blur_width=5.0, indices=selection
        )
        assert len(columns) == len(selection) - 10
        assert columns.lpx.max() > 5.0  # the columns stay whole...
        assert columns.materialize().lpx.max() <= 5.0  # ...the rows do not

    def test_render_scene_takes_per_channel_indices(
        self, locs, info, selection
    ):
        _, n = render.render_scene(
            [locs, locs],
            [info, info],
            blur_method="gaussian",
            indices=[selection, None],
            **self.KWARGS,
        )
        _, n_ref = render.render_scene(
            [locs.iloc[selection], locs],
            [info, info],
            blur_method="gaussian",
            **self.KWARGS,
        )
        assert n == n_ref
        _, n_single = render.render_scene(
            locs,
            info,
            blur_method="gaussian",
            indices=selection,
            **self.KWARGS,
        )
        assert (
            n_single
            == render.render(
                locs,
                info,
                blur_method="gaussian",
                indices=selection,
                **self.KWARGS,
            )[0]
        )


# ---------------------------------------------------------------------------
# Maximum blur width (useless precisions are not rendered)
# ---------------------------------------------------------------------------


class TestMaxBlurWidth:
    """``max_blur_width`` drops localizations with unphysical
    precisions from the per-localization blur methods, consistently
    for every entry point, count included."""

    KWARGS = dict(
        disp_px_size=PIXELSIZE / 10,
        viewport=FULL_VIEWPORT,
        min_blur_width=0.0,
        ang=None,
    )
    LIMIT = 5.0  # camera pixels
    N_WHALES = 8

    @pytest.fixture
    def whaled(self, locs):
        whaled = locs.copy()
        whaled.loc[whaled.index[:5], "lpx"] = 50.0
        whaled.loc[whaled.index[5:8], "lpy"] = 50.0
        return whaled

    @pytest.mark.parametrize("blur", ["gaussian", "gaussian_iso"])
    def test_extraction_drops_wide_precisions(self, whaled, blur):
        columns = render._extract_render_columns(
            whaled, blur, None, max_blur_width=self.LIMIT
        )
        assert len(columns) == len(whaled) - self.N_WHALES
        assert columns.lpx.max() <= self.LIMIT
        assert columns.lpy.max() <= self.LIMIT

    @pytest.mark.parametrize("blur", [None, "smooth", "convolve"])
    def test_other_methods_render_everything(self, whaled, blur):
        columns = render._extract_render_columns(
            whaled, blur, None, max_blur_width=self.LIMIT
        )
        assert len(columns) == len(whaled)

    def test_no_limit_keeps_everything(self, whaled):
        columns = render._extract_render_columns(whaled, "gaussian", None)
        assert len(columns) == len(whaled)

    def test_render_matches_prefiltered_locs(self, whaled, info):
        keep = (whaled["lpx"] <= self.LIMIT) & (whaled["lpy"] <= self.LIMIT)
        n_ref, image_ref = render.render(
            whaled[keep], info, blur_method="gaussian", **self.KWARGS
        )
        n, image = render.render(
            whaled,
            info,
            blur_method="gaussian",
            max_blur_width=self.LIMIT,
            **self.KWARGS,
        )
        assert n == n_ref
        np.testing.assert_array_equal(image, image_ref)
        # without the limit the whales are drawn (and counted)
        n_all, image_all = render.render(
            whaled, info, blur_method="gaussian", **self.KWARGS
        )
        assert n_all > n
        assert not np.array_equal(image_all, image)

    def test_render_scene_plumbs_the_limit(self, whaled, info):
        _, n = render.render_scene(
            whaled,
            info,
            blur_method="gaussian",
            max_blur_width=self.LIMIT,
            **self.KWARGS,
        )
        _, n_all = render.render_scene(
            whaled, info, blur_method="gaussian", **self.KWARGS
        )
        assert n < n_all
        _, n_multi = render.render_scene(
            [whaled, whaled],
            [info, info],
            blur_method="gaussian",
            max_blur_width=self.LIMIT,
            **self.KWARGS,
        )
        assert n_multi == 2 * n


# ---------------------------------------------------------------------------
# Fused post-processing vs the legacy numpy chain
# ---------------------------------------------------------------------------


def _legacy_multi_lut(
    raw, luts, contrast, relative_intensities, background_color, invert
):
    """The pre-fusion numpy post-processing chain, kept as the reference
    the fused kernels are compared against (built from the public
    helpers plus the removed inline steps)."""
    vmin, vmax = contrast if contrast is not None else (None, None)
    images = render.scale_contrast(
        raw.copy(), vmin, vmax, autoscale=contrast is None
    )
    images = render.scale_intensities(
        images, relative_intensities=relative_intensities
    )
    colors_arr = np.asarray(luts, dtype=np.float32)
    images_f32 = np.ascontiguousarray(images, dtype=np.float32)
    idx = np.clip((images_f32 * 255.0).astype(np.int32), 0, 255)
    rgb = np.zeros(
        (images_f32.shape[1], images_f32.shape[2], 3), dtype=np.float32
    )
    for c in range(images_f32.shape[0]):
        rgb += colors_arr[c][idx[c]]
    np.minimum(rgb, 1.0, out=rgb)
    if background_color is not None and any(c > 0 for c in background_color):
        bg = np.asarray(background_color, dtype=np.float32)
        alpha = np.clip(images_f32.sum(axis=0), 0.0, 1.0)[..., None]
        rgb = rgb + bg * (1.0 - alpha)
        np.minimum(rgb, 1.0, out=rgb)
    rgb = render.to_8bit(rgb)
    if invert:
        rgb = 255 - rgb
    return rgb


def _legacy_single(raw, colormap, contrast, invert):
    vmin, vmax = contrast if contrast is not None else (None, None)
    image = render.scale_contrast(
        raw.copy(), vmin, vmax, autoscale=contrast is None
    )
    image = render.to_8bit(image)
    rgb = render.apply_colormap(image, colormap)
    if invert:
        rgb = 255 - rgb
    return rgb


class TestFusedCompose:
    """The fused post-processing kernels must reproduce the legacy
    numpy chain to within uint8 rounding (<= 1 count per pixel)."""

    @pytest.fixture(scope="class")
    def raw_stack(self):
        rng = np.random.default_rng(7)
        raw = rng.exponential(2.0, size=(3, 80, 90)).astype(np.float32)
        raw[:, ::7, ::5] = 0.0  # empty pixels
        raw[0, 3, 4] = np.nan  # non-finite must map to 0
        raw[1, 10, 11] = 500.0  # fiducial-grade hot pixel
        return raw

    @pytest.fixture(scope="class")
    def luts(self):
        return [render.solid_to_lut(rgb) for rgb in lib.get_colors(3)]

    @pytest.mark.parametrize(
        "contrast,rel,bg,invert",
        [
            ((0.0, 5.0), None, None, False),
            ((0.0, 5.0), [1.0, 0.7, 1.3], None, False),
            ((0.0, 5.0), None, (0.08, 0.08, 0.12), False),
            ((0.5, 3.0), [1.0, 0.7, 1.3], (0.1, 0.0, 0.2), True),
            (None, None, None, False),  # autoscale
        ],
    )
    def test_multi_lut_matches_legacy(
        self, raw_stack, luts, info, contrast, rel, bg, invert
    ):
        expected = _legacy_multi_lut(
            raw_stack, luts, contrast, rel, bg, invert
        )
        _, rgb, _, _ = render._render_multi_channel(
            [None] * 3,
            [info] * 3,
            disp_px_size=10.0,
            colors=luts,
            contrast=contrast,
            relative_intensities=rel,
            invert_colors=invert,
            background_color=bg,
            raw_image_cache=raw_stack,
        )
        diff = np.abs(rgb.astype(np.int16) - expected.astype(np.int16))
        assert diff.max() <= 1

    @pytest.mark.parametrize("contrast", [(0.0, 5.0), None])
    @pytest.mark.parametrize("invert", [False, True])
    def test_single_matches_legacy(self, raw_stack, info, contrast, invert):
        raw = raw_stack[0]
        for colormap in ["magma", np.linspace(0, 1, 256 * 4).reshape(256, 4)]:
            expected = _legacy_single(raw, colormap, contrast, invert)
            _, rgb, _, _ = render._render_single_channel(
                None,
                info,
                disp_px_size=10.0,
                contrast=contrast,
                invert_colors=invert,
                single_channel_colormap=colormap,
                raw_image_cache=raw,
            )
            diff = np.abs(rgb.astype(np.int16) - expected.astype(np.int16))
            assert diff.max() <= 1

    def test_contrast_limits_match_scale_contrast(self, raw_stack):
        for args in [
            (None, None, True),
            (0.5, 4.0, False),
            (None, 2.0, False),
        ]:
            _, expected = render.scale_contrast(
                raw_stack.copy(),
                args[0],
                args[1],
                autoscale=args[2],
                return_contrast_limits=True,
            )
            result = render._contrast_limits(raw_stack, *args)
            # NaN-aware comparison: with NaN pixels and vmin=None both
            # implementations agree on vmin=NaN, but NaN != NaN
            np.testing.assert_array_equal(
                np.asarray(result, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
            )


class TestChunkMemory:
    """The chunked CPU path bounds its memory: chunk images are summed
    as they arrive (at most one per worker plus one alive) and the
    worker count shrinks with the memory available."""

    def _locs(self, n=400_000, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "x": rng.uniform(0, 64, n),
                "y": rng.uniform(0, 64, n),
                "lpx": rng.uniform(0.05, 0.2, n),
                "lpy": rng.uniform(0.05, 0.2, n),
            }
        )

    def _info(self):
        return [{"Width": 64, "Height": 64, "Frames": 1, "Pixelsize": 130.0}]

    def test_streamed_sum_equals_the_sequential_render(self, monkeypatch):
        from picasso.render import splat

        locs = self._locs()
        info = self._info()
        kwargs = dict(
            disp_px_size=130 / 4,
            viewport=((0, 0), (64, 64)),
            blur_method="gaussian",
        )
        monkeypatch.setattr(splat, "_render_worker_budget", lambda: 1)
        n1, sequential = render.render_scene(
            locs, info, return_raw_image=True, **kwargs
        )[1:]
        monkeypatch.setattr(splat, "_render_worker_budget", lambda: 4)
        n4, chunked = render.render_scene(
            locs, info, return_raw_image=True, **kwargs
        )[1:]
        assert n1 == n4 == len(locs)
        np.testing.assert_allclose(chunked, sequential, rtol=1e-5, atol=1e-6)
        # and twice the same, whatever the completion order of the pool
        again = render.render_scene(
            locs, info, return_raw_image=True, **kwargs
        )[2]
        np.testing.assert_array_equal(chunked, again)

    def test_at_most_one_image_per_worker_plus_one_is_alive(self, monkeypatch):
        import threading
        import weakref

        from picasso.render import splat

        original = splat._render_arrays
        lock = threading.Lock()
        live = [0]
        peak = [0]
        created = [0]

        def freed():
            with lock:
                live[0] -= 1

        def tracked(*args, **kwargs):
            n, image = original(*args, **kwargs)
            with lock:
                live[0] += 1
                created[0] += 1
                peak[0] = max(peak[0], live[0])
            weakref.finalize(image, freed)  # runs when the chunk is dropped
            return n, image

        monkeypatch.setattr(splat, "_render_arrays", tracked)
        monkeypatch.setattr(splat, "_render_worker_budget", lambda: 3)
        monkeypatch.setattr(splat, "_MIN_CHUNK_LOCS", 10_000)
        render.render_scene(
            self._locs(300_000),  # 6 chunks for 3 workers
            self._info(),
            disp_px_size=130 / 4,
            viewport=((0, 0), (64, 64)),
            blur_method="gaussian",
        )
        # never all kept: two in flight per worker, one being summed,
        # the accumulator (six chunks here, so the bound is the window)
        assert created[0] >= 6
        assert peak[0] <= 2 * 3 + 2

    def test_workers_shrink_with_available_memory(self, monkeypatch):
        from picasso.render import splat

        class _Memory:
            def __init__(self, available):
                self.available = available

        image = 4 * 256 * 256
        monkeypatch.setattr(
            splat.psutil, "virtual_memory", lambda: _Memory(100 * image)
        )
        # room for (0.5 * 100 - 1) images, two per task: ~24 tasks,
        # two of them in flight per worker
        assert splat._memory_bounded_workers(32, 1, image, 0) == 11
        monkeypatch.setattr(
            splat.psutil, "virtual_memory", lambda: _Memory(3 * image)
        )
        assert splat._memory_bounded_workers(32, 1, image, 0) == 1  # never 0
        monkeypatch.setattr(
            splat.psutil, "virtual_memory", lambda: _Memory(10**12)
        )
        assert splat._memory_bounded_workers(8, 12, image, 100_000) == 8
        # the render still runs, on one worker, when memory is tight
        monkeypatch.setattr(
            splat.psutil, "virtual_memory", lambda: _Memory(3 * image)
        )
        monkeypatch.setattr(splat, "_render_worker_budget", lambda: 4)
        n, raw = render.render_scene(
            self._locs(),
            self._info(),
            disp_px_size=130 / 4,
            viewport=((0, 0), (64, 64)),
            blur_method="gaussian",
            return_raw_image=True,
        )[1:]
        assert n == 400_000 and raw.sum() > 0


class TestFallbackNote:
    """The reason of a CPU fallback reaches the info dialog's renderer
    line (the log warning is invisible in the windowed application)."""

    def test_reason_is_recorded_and_cleared(self, monkeypatch):
        from picasso.render import backend, scene, splat

        class _Flaky(backend.SplatBackend):
            name = "fake"
            persistent_uploads = True
            fail = True

            def describe(self):
                return "Fake GPU"

            def render_channels(self, columns, info, **kwargs):
                if self.fail:
                    raise backend.SplatBackendError(
                        "channel exceeds the limit"
                    )
                return splat.CpuBackend().render_channels(
                    columns, info, **kwargs
                )

        fake = _Flaky()
        monkeypatch.setattr(scene, "_get_backend", lambda *a, **k: fake)
        monkeypatch.setattr(backend, "_get_backend", lambda *a, **k: fake)
        backend.note_fallback(None)
        rng = np.random.default_rng(0)
        locs = pd.DataFrame(
            {"x": rng.uniform(0, 8, 500), "y": rng.uniform(0, 8, 500)}
        )
        info = [{"Width": 8, "Height": 8, "Frames": 1, "Pixelsize": 130.0}]
        render.render_scene(
            locs, info, disp_px_size=65, viewport=((0, 0), (8, 8))
        )
        assert backend.last_fallback() == "channel exceeds the limit"
        assert backend.describe_active() == (
            "GPU (Fake GPU) - last render on the CPU: channel exceeds the limit"
        )
        fake.fail = False
        render.render_scene(
            locs, info, disp_px_size=65, viewport=((0, 0), (8, 8))
        )
        assert backend.last_fallback() is None
        assert backend.describe_active() == "GPU (Fake GPU)"
