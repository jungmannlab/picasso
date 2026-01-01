"""Test picasso.render functions and the associates functions in
picasso.masking.

:author: Rafal Kowalewski, 2025
:copyright: Copyright (c) 2025 Jungmann Lab, MPI of Biochemistry
"""

import pytest
from picasso import io, lib, masking, render

# parameters for rendering and masking
VIEWPORT = ((15, 15), (16, 16))
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


@pytest.fixture(scope="module")
def locs_data():
    """Load localization data once per test module."""
    locs_data = io.load_locs("./tests/data/testdata_locs.hdf5")
    return locs_data


@pytest.fixture(scope="module")
def locs(locs_data):
    """Get locs for testing clusterers."""
    locs = locs_data[0]
    return locs


@pytest.fixture(scope="module")
def info(locs_data):
    """Get info for testing clusterers."""
    info = locs_data[1]
    return info


# rendering tests
def test_render_viewport(locs, info):
    """Test rendering with viewport."""
    im = render.render(locs, info, oversampling=130, viewport=VIEWPORT)[1]
    assert im.shape == (130, 130), f"Unexpected image shape: {im.shape}"
    assert im.max() > 0, f"Image max is not greater than zero: max={im.max()}"
    assert im.dtype == "float32", f"Unexpected image dtype: {im.dtype}"


def test_render_one_pixel_blur(locs, info):
    """Test rendering with one pixel blur."""
    im = render.render(locs, info, oversampling=13, blur_method="smooth")[1]
    assert im.max() > 0, f"Image max is not greater than zero: max={im.max()}"
    assert im.dtype == "float32", f"Unexpected image dtype: {im.dtype}"


def test_render_global_lp(locs, info):
    """Test rendering with global localization precision."""
    im = render.render(locs, info, oversampling=13, blur_method="convolve")[1]
    assert im.max() > 0, f"Image max is not greater than zero: max={im.max()}"
    assert im.dtype == "float32", f"Unexpected image dtype: {im.dtype}"


def test_render_gaussian_lp(locs, info):
    """Test rendering with Gaussian localization precision."""
    im = render.render(locs, info, oversampling=13, blur_method="gaussian")[1]
    assert im.max() > 0, f"Image max is not greater than zero: max={im.max()}"
    assert im.dtype == "float32", f"Unexpected image dtype: {im.dtype}"


def test_render_gaussian_iso(locs, info):
    """Test rendering with isotropic Gaussian localization precision."""
    im = render.render(
        locs, info, oversampling=13, blur_method="gaussian_iso"
    )[1]
    assert im.max() > 0, f"Image max is not greater than zero: max={im.max()}"
    assert im.dtype == "float32", f"Unexpected image dtype: {im.dtype}"


def test_render_gaussian_3d(locs, info):
    """Test rendering with 3D Gaussian localization precision."""
    locs_z = locs.copy()
    locs_z["z"] = locs_z["y"] * 0.0  # add z column with zeros
    im = render.render(
        locs_z, info, oversampling=13, blur_method="gaussian", ang=(0, 0, 0)
    )[1]
    assert im.max() > 0, f"Image max is not greater than zero: max={im.max()}"
    assert im.dtype == "float32", f"Unexpected image dtype: {im.dtype}"


# masking tests
@pytest.fixture(scope="module")
def image(locs, info):
    """Render image for masking tests."""
    image = render.render(locs, info, oversampling=13)[1]
    return image


def test_masking_methods(image):
    """Test all masking methods."""
    for method in MASKING_METHODS:
        mask, threshold = masking.mask_image(image, method=method)
        assert (
            mask.shape == image.shape
        ), f"Mask shape mismatch for method {method}"


def test_masking(locs, info, image):
    mask = masking.mask_image(image, method="otsu")[0]
    width = lib.get_from_metadata(info, "Width")
    height = lib.get_from_metadata(info, "Height")
    locs_in, locs_out = masking.mask_locs(locs, mask, width, height)
