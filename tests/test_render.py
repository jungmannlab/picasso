"""Test picasso.render functions and the associated functions in
picasso.masking.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import sys

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtGui

from picasso import io, masking, render

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
def _qt_app():
    """Ensure a QGuiApplication exists for QImage / QPainter operations."""
    app = QtGui.QGuiApplication.instance()
    if app is None:
        app = QtGui.QGuiApplication(sys.argv)
    yield app


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
    return render.render(locs, info, oversampling=13)[1]


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
        n, im = render.render(locs, info, oversampling=13, viewport=VIEWPORT)
        assert im.sum() == n
        assert n > 0, "Test data should have locs in viewport"

    def test_viewport_exact_shape(self, locs, info):
        """Image shape is exactly oversampling * viewport size."""
        n, im = render.render(locs, info, oversampling=130, viewport=VIEWPORT)
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
        n, _ = render.render(locs, info, oversampling=13, viewport=VIEWPORT)
        assert n == expected

    def test_disp_px_size_equivalence(self, locs, info):
        """`oversampling=k` and `disp_px_size=pixelsize/k` must be
        bitwise-identical."""
        oversampling = 5
        disp_px = PIXELSIZE / oversampling
        n1, im1 = render.render(
            locs, info, oversampling=oversampling, viewport=FULL_VIEWPORT
        )
        n2, im2 = render.render(
            locs, info, disp_px_size=disp_px, viewport=FULL_VIEWPORT
        )
        assert n1 == n2
        assert np.array_equal(im1, im2)

    @pytest.mark.parametrize("blur_method", BLUR_METHODS)
    def test_blur_methods(self, locs, info, blur_method):
        """All four blur methods produce correctly-shaped, finite, non-zero
        images."""
        n, im = render.render(
            locs,
            info,
            oversampling=5,
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
            oversampling=5,
            viewport=FULL_VIEWPORT,
            blur_method=blur_method,
        )
        assert im.sum() == pytest.approx(n, rel=1e-3)

    def test_min_blur_width_broadens(self, locs, info):
        """Larger min_blur_width spreads mass: peak intensity must drop."""
        _, im_narrow = render.render(
            locs,
            info,
            oversampling=5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            min_blur_width=0.0,
        )
        _, im_wide = render.render(
            locs,
            info,
            oversampling=5,
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
                oversampling=5,
                viewport=FULL_VIEWPORT,
                blur_method="not_a_method",
            )

    def test_no_info_no_viewport_raises(self, locs):
        with pytest.raises(ValueError):
            render.render(locs, None, oversampling=5)

    def test_3d_rotation_changes_image(self, locs_3d, info):
        """A non-zero rotation must produce a different image."""
        _, im_no_rot = render.render(
            locs_3d,
            info,
            oversampling=5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            ang=(0.0, 0.0, 0.0),
        )
        _, im_rot = render.render(
            locs_3d,
            info,
            oversampling=5,
            viewport=FULL_VIEWPORT,
            blur_method="gaussian",
            ang=(0.5, 0.3, 0.2),
        )
        assert not np.array_equal(im_no_rot, im_rot)


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


class TestBuildAnimation:
    def test_smoke_two_frames(self, locs_3d, info, tmp_path):
        """A 2-frame animation writes both an .mp4 and the sidecar .yaml."""
        out_path = tmp_path / "anim.mp4"
        positions = [
            (0.0, 0.0, 0.0, FULL_VIEWPORT),
            (0.1, 0.0, 0.0, FULL_VIEWPORT),
        ]
        render.build_animation(
            str(out_path),
            locs_3d,
            info,
            positions=positions,
            durations=[1.0],
            disp_px_size=PIXELSIZE,
            image_size=(64, 64),
            fps=2,
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        yaml_path = out_path.with_suffix(".yaml")
        assert yaml_path.exists()
        assert yaml_path.stat().st_size > 0


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
        locs_in, locs_out = masking.mask_locs(locs, mask, info=info)
        assert len(locs_in) + len(locs_out) == len(locs)
        # in / out are disjoint
        assert set(locs_in.index).isdisjoint(set(locs_out.index))
        # both partitions only contain valid loc indices
        all_idx = set(locs_in.index) | set(locs_out.index)
        assert all_idx == set(locs.index)
