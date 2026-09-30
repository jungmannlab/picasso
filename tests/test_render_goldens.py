"""Golden-image regression tests for picasso.render.

Every scene in ``tests/_render_golden_scenes.py`` is rendered with the
current implementation and compared against the frozen arrays in
``tests/data/render_goldens.npz``. Unlike the property tests in
``test_render.py`` (mass conservation, shapes, rotation semantics),
these tests pin the exact pixel values, so kernel refactors have to
prove they leave the output unchanged.

The table runs on two backends (``test_golden[cpu-*]`` and
``test_golden[gpu-*]``). The archive is always the CPU output; the GPU
run compares the wgpu backend against those same frozen pixels, so a
GPU regression shows up against pinned values rather than against
whatever the CPU currently renders. The GPU parametrization is opt-in
through the ``gpu_backend`` marker and skips without ``wgpu`` or a
usable adapter (the ``picasso_fast_render`` environment runs it).

Comparison policy, CPU: float arrays use a tight tolerance (rtol 1e-5)
rather than bit-exact equality -- ``exp`` differs at the last bit across
libm implementations and numba versions, and parallel splatting reorders
float summation; both effects sit far below 1e-5 while any real kernel
bug (centering, sigma scaling, normalization) shifts values at percent
level. uint8 RGB output may differ by at most 1 count per pixel (values
sitting exactly on a rounding boundary). The CPU chunk pool sums its
chunks in a fixed order, so for a given worker budget its output is
reproducible from run to run.

Comparison policy, GPU (float32 arithmetic against the CPU's float64,
and contributions to a pixel summed in a hardware-dependent order --
workgroup scheduling and atomics -- so even two GPU renders can differ
in the last bits; see ``TestGpuRepeatability``):

* gaussian family: rtol ``GPU_RTOL``, atol ``GPU_ATOL_FRACTION`` of the
  image maximum, total intensity within ``GPU_SUM_RTOL`` (measured on
  an Apple M4: at most 1.2e-4 of the maximum per pixel and 5e-5 of the
  total, both an order of magnitude inside the tolerance, which leaves
  room for other drivers' ``exp`` implementations);
* histograms: integer atomics, so counts are exact except for a
  localization sitting within float rounding of a pixel edge, which
  lands in the neighbor pixel -- at most 1 count per pixel, on fewer
  than ``GPU_FLIP_FRACTION`` of the pixels, totals equal (none of the
  564 fixture localizations flips; ``test_render_gpu.py`` exercises the
  clause on 150k);
* smooth / convolve: the GPU histogram filtered by the same CPU kernel,
  so a boundary flip is spread by the filter (``GPU_FILTER_ATOL``);
* 3D rotation happens in-shader in float32: a handful of localizations
  land a pixel over or flip in / out of view, each touching its whole
  footprint, so rotated scenes are judged by the fraction of pixels
  outside the gaussian tolerance (``GPU_ROTATED_MISMATCH``) and the
  count of rendered localizations by ``GPU_COUNT_SLACK``;
* uint8 RGB: at most 1 count per pixel, except that a boundary flip in
  a histogram-based scene becomes a full contrast step (``1 / contrast``
  of 255 counts), allowed on fewer than ``GPU_FLIP_FRACTION`` of the
  pixels.

To intentionally change rendering output, regenerate the archive with
``python tests/regen_render_goldens.py`` and commit the npz diff.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

import numpy as np
import pytest

from picasso import io, render

from tests._render_golden_scenes import (
    GOLDENS_PATH,
    KIND_KEYS,
    SCENES,
    prepare_inputs,
    run_scene,
    run_scene_counted,
)

REGEN_HINT = (
    "run 'python tests/regen_render_goldens.py' and commit the result if "
    "this rendering change is intended"
)

#: GPU-vs-golden tolerances (see the module docstring)
GPU_RTOL = 5e-3
GPU_ATOL_FRACTION = 2e-3
GPU_SUM_RTOL = 1e-3
GPU_FLIP_FRACTION = 1e-4
GPU_FILTER_ATOL = 0.2
GPU_ROTATED_MISMATCH = 5e-4


def GPU_COUNT_SLACK(n):
    """Localizations a float32 rotation may move in or out of view."""
    return max(3, 1e-4 * n)


BACKENDS = ["cpu", pytest.param("gpu", marks=pytest.mark.gpu_backend)]
SCENE_BY_NAME = {scene["name"]: scene for scene in SCENES}


@pytest.fixture(scope="module")
def golden_data():
    try:
        return np.load(GOLDENS_PATH)
    except FileNotFoundError:
        pytest.fail(f"{GOLDENS_PATH} is missing; {REGEN_HINT}")


@pytest.fixture(scope="module")
def locs_data():
    return io.load_locs("./tests/data/testdata_locs.hdf5")


@pytest.fixture(scope="module")
def inputs(locs_data):
    return prepare_inputs(locs_data[0])


@pytest.fixture(scope="module")
def info(locs_data):
    return locs_data[1]


@pytest.fixture(scope="module")
def gpu_backend():
    pytest.importorskip("wgpu")
    from picasso.render.backend import SplatBackendError
    from picasso.render.gpu import WgpuBackend

    try:
        backend = WgpuBackend()
    except SplatBackendError as error:
        pytest.skip(f"no usable GPU adapter: {error}")
    yield backend
    backend.close()


def _select_backend(monkeypatch, backend):
    """Every render through the splat seam uses ``backend``, whatever
    the localization count (the fixture is far below the GPU cutoff)."""
    monkeypatch.setattr(render.scene, "_get_backend", lambda **kw: backend)


@pytest.fixture
def render_backend(request, monkeypatch):
    """'cpu' renders the scene as the archive was made; 'gpu' renders it
    on the wgpu backend and returns the name of the backend in use."""
    if request.param == "gpu":
        _select_backend(monkeypatch, request.getfixturevalue("gpu_backend"))
    return request.param


def test_goldens_match_scene_table(golden_data):
    """The archive holds exactly the arrays the scene table defines --
    catches stale goldens after the table changes."""
    expected = {
        f"{scene['name']}::{key}"
        for scene in SCENES
        for key in KIND_KEYS[scene["kind"]]
    }
    assert set(golden_data.files) == expected, REGEN_HINT


def _assert_cpu_golden(label, arr, golden):
    if arr.dtype == np.uint8:
        diff = np.abs(arr.astype(np.int16) - golden.astype(np.int16))
        assert diff.max() <= 1, (
            f"{label}: uint8 output deviates by up to {diff.max()} counts "
            f"at {np.unravel_index(diff.argmax(), diff.shape)} "
            f"({(diff > 1).sum()} pixels beyond rounding noise); {REGEN_HINT}"
        )
    else:
        np.testing.assert_allclose(
            arr,
            golden,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"{label} deviates from golden; {REGEN_HINT}",
        )


def _assert_gpu_golden(label, scene, arr, golden):
    rotated = scene.get("ang") is not None
    blur = scene["blur"]
    if arr.dtype == np.uint8:
        diff = np.abs(arr.astype(np.int16) - golden.astype(np.int16))
        beyond = (diff > 1).mean()
        assert beyond < GPU_FLIP_FRACTION, (
            f"{label}: {beyond:.2e} of the pixels deviate by more than 1 "
            f"count from the CPU golden (max {diff.max()})"
        )
        return

    arr = arr.astype(np.float64)
    golden = golden.astype(np.float64)
    if blur is None:
        # integer atomics: exact up to pixel-boundary rounding
        diff = np.abs(arr - golden)
        assert diff.max() <= 1, f"{label}: a count moved by more than a pixel"
        flipped = (diff > 0).mean()
        assert flipped < GPU_FLIP_FRACTION, (
            f"{label}: {flipped:.2e} of the pixels differ from the golden "
            "histogram"
        )
        if rotated:
            assert abs(arr.sum() - golden.sum()) <= GPU_COUNT_SLACK(
                golden.sum()
            )
        else:
            assert arr.sum() == golden.sum(), f"{label}: counts not conserved"
        return

    if blur in ("smooth", "convolve"):
        np.testing.assert_allclose(
            arr, golden, rtol=1e-4, atol=GPU_FILTER_ATOL, err_msg=label
        )
        return

    close = np.isclose(
        arr, golden, rtol=GPU_RTOL, atol=GPU_ATOL_FRACTION * golden.max()
    )
    mismatch = (~close).mean()
    if rotated:
        assert (
            mismatch < GPU_ROTATED_MISMATCH
        ), f"{label}: {mismatch:.2e} of the pixels outside the GPU tolerance"
    else:
        assert close.all(), (
            f"{label}: {(~close).sum()} pixels outside the GPU tolerance "
            f"(max deviation {np.abs(arr - golden).max():.3g} of a maximum "
            f"{golden.max():.3g})"
        )
    assert (
        abs(arr.sum() - golden.sum()) < GPU_SUM_RTOL * golden.sum()
    ), f"{label}: total intensity {arr.sum():.6g} vs golden {golden.sum():.6g}"


def _cpu_count(scene, inputs, info):
    """Localizations the CPU path renders for ``scene``, whatever backend
    the seam currently selects."""
    with pytest.MonkeyPatch.context() as mp:
        _select_backend(mp, render.backend._cpu_backend())
        return run_scene_counted(scene, inputs, info)[0]


@pytest.mark.parametrize("render_backend", BACKENDS, indirect=True)
@pytest.mark.parametrize("scene", SCENES, ids=[s["name"] for s in SCENES])
def test_golden(scene, render_backend, inputs, info, golden_data):
    on_gpu = render_backend == "gpu"
    n, out = run_scene_counted(scene, inputs, info, via_backend=on_gpu)
    if on_gpu:
        n_cpu = _cpu_count(scene, inputs, info)
        if scene.get("ang") is None:
            assert n == n_cpu
        else:
            assert abs(n - n_cpu) <= GPU_COUNT_SLACK(n_cpu)
    for key in KIND_KEYS[scene["kind"]]:
        arr = out[key]
        label = f"{scene['name']}::{key}"
        golden = golden_data[label]
        assert (
            arr.shape == golden.shape
        ), f"{label} shape {arr.shape} != golden {golden.shape}; {REGEN_HINT}"
        if on_gpu:
            _assert_gpu_golden(label, scene, arr, golden)
        else:
            _assert_cpu_golden(label, arr, golden)


@pytest.mark.gpu_backend
class TestGpuRepeatability:
    """The GPU's summation order is not fixed, but its effect is bounded
    by the same tolerance the goldens allow; histograms (integer
    atomics) repeat bit for bit."""

    def test_repeat_gaussian_renders_agree(
        self, gpu_backend, inputs, info, monkeypatch
    ):
        _select_backend(monkeypatch, gpu_backend)
        scene = SCENE_BY_NAME["gaussian_full"]
        first = run_scene(scene, inputs, info, via_backend=True)["image"]
        second = run_scene(scene, inputs, info, via_backend=True)["image"]
        np.testing.assert_allclose(
            second, first, rtol=GPU_RTOL, atol=GPU_ATOL_FRACTION * first.max()
        )
        assert abs(second.sum() - first.sum()) < GPU_SUM_RTOL * first.sum()

    def test_repeat_histograms_are_identical(
        self, gpu_backend, inputs, info, monkeypatch
    ):
        _select_backend(monkeypatch, gpu_backend)
        scene = SCENE_BY_NAME["hist_full"]
        first = run_scene(scene, inputs, info, via_backend=True)["image"]
        second = run_scene(scene, inputs, info, via_backend=True)["image"]
        np.testing.assert_array_equal(second, first)
