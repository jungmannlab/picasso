"""Tests for the wgpu splat backend.

These self-skip when ``wgpu`` is not installed or no GPU adapter is
available.
"""

import numpy as np
import pandas as pd
import pytest

wgpu = pytest.importorskip("wgpu")

from picasso import render  # noqa: E402
from picasso.render import backend as backend_mod  # noqa: E402
from picasso.render.backend import SplatBackendError  # noqa: E402

PIXELSIZE = 130.0  # camera pixel size (nm); every length is in nm
INFO = [{"Height": 64, "Width": 64, "Pixelsize": PIXELSIZE}]


def _nm(viewport):
    """A viewport given in camera pixels, in nm."""
    return tuple(tuple(c * PIXELSIZE for c in corner) for corner in viewport)


KWARGS = dict(
    disp_px_size=13.0,  # 10 rendered pixels per camera pixel
    viewport=_nm(((3.0, 2.0), (61.0, 62.5))),
    min_blur_width=0.1 * PIXELSIZE,
    ang=None,
)
# f32 arithmetic vs the CPU's f64: measured ~1e-4 of the image maximum
RTOL = 5e-3
ATOL_FRACTION = 2e-3


def _assert_parity(img_gpu, img_cpu):
    np.testing.assert_allclose(
        img_gpu, img_cpu, rtol=RTOL, atol=ATOL_FRACTION * img_cpu.max()
    )
    assert abs(img_gpu.sum() - img_cpu.sum()) < 1e-3 * img_cpu.sum()


@pytest.fixture(scope="module")
def gpu():
    try:
        from picasso.render.gpu import WgpuBackend

        backend = WgpuBackend()
    except SplatBackendError as error:
        pytest.skip(f"no usable GPU adapter: {error}")
    yield backend
    backend.close()


@pytest.fixture(scope="module")
def locs():
    rng = np.random.default_rng(42)
    n = 150_000
    ps = PIXELSIZE
    return pd.DataFrame(
        {
            "x": rng.uniform(-2 * ps, 66 * ps, n).astype(np.float32),
            "y": rng.uniform(-2 * ps, 66 * ps, n).astype(np.float32),
            "lpx": rng.uniform(0.02 * ps, 0.5 * ps, n).astype(np.float32),
            "lpy": rng.uniform(0.02 * ps, 0.5 * ps, n).astype(np.float32),
        }
    )


@pytest.fixture(scope="module")
def tiered_locs(locs):
    """Precisions spanning all three GPU tiers at overview zoom:
    sub-pixel (the bulk), tile-sized, and whole-view whales."""
    tiered = locs.copy()
    rng = np.random.default_rng(7)
    ps = PIXELSIZE
    tiered.loc[tiered.index[:200], ["lpx", "lpy"]] = rng.uniform(
        0.5 * ps, 2.0 * ps, (200, 2)
    ).astype(np.float32)
    tiered.loc[tiered.index[200:220], ["lpx", "lpy"]] = rng.uniform(
        5.0 * ps, 40.0 * ps, (20, 2)
    ).astype(np.float32)
    return tiered


@pytest.fixture(scope="module")
def locs_3d(locs):
    """2D locs with z and lpz (nm, like x and y) and a
    per-localization ellipse angle (degrees)."""
    rng = np.random.default_rng(3)
    ps = PIXELSIZE
    locs_3d = locs.copy()
    locs_3d["z"] = rng.uniform(-2.0 * ps, 2.0 * ps, len(locs)).astype(
        np.float32
    )
    locs_3d["lpz"] = rng.uniform(0.05 * ps, 0.6 * ps, len(locs)).astype(
        np.float32
    )
    locs_3d["angle"] = rng.uniform(0.0, 180.0, len(locs)).astype(np.float32)
    return locs_3d


ANG = (0.4, 0.2, 0.1)


def _columns(locs, blur, ang=None):
    return [render._extract_render_columns(locs, blur, ang)]


def _cpu(locs, blur, **overrides):
    kwargs = dict(KWARGS, **overrides)
    return backend_mod._cpu_backend().render_channels(
        _columns(locs, blur, kwargs["ang"]), [INFO], blur_method=blur, **kwargs
    )


def _gpu(gpu, locs, blur, **overrides):
    kwargs = dict(KWARGS, **overrides)
    return gpu.render_channels(
        _columns(locs, blur, kwargs["ang"]), [INFO], blur_method=blur, **kwargs
    )


class TestWgpuParity:
    def test_gaussian_matches_cpu(self, gpu, locs):
        ((n_cpu, img_cpu),) = _cpu(locs, "gaussian")
        ((n_gpu, img_gpu),) = _gpu(gpu, locs, "gaussian")
        assert n_gpu == n_cpu
        assert img_gpu.shape == img_cpu.shape
        assert img_gpu.dtype == np.float32
        _assert_parity(img_gpu, img_cpu)

    def test_hist_matches_cpu(self, gpu, locs):
        ((n_cpu, img_cpu),) = _cpu(locs, None)
        ((n_gpu, img_gpu),) = _gpu(gpu, locs, None)
        assert n_gpu == n_cpu
        assert img_gpu.sum() == img_cpu.sum()  # counts conserved exactly
        # locs within float rounding of a pixel boundary may land in
        # the neighbor pixel (f32 vs f64 coordinate transform)
        diff = np.abs(img_gpu - img_cpu)
        assert diff.max() <= 1
        assert (diff > 0).mean() < 1e-4

    def test_all_tiers_at_overview(self, gpu, tiered_locs):
        ((n_cpu, img_cpu),) = _cpu(tiered_locs, "gaussian", min_blur_width=0.0)
        ((n_gpu, img_gpu),) = _gpu(
            gpu, tiered_locs, "gaussian", min_blur_width=0.0
        )
        assert n_gpu == n_cpu
        _assert_parity(img_gpu, img_cpu)

    @pytest.mark.parametrize(
        "viewport,disp_px_size",
        [
            (_nm(((20.0, 20.0), (44.0, 44.0))), 3.25),  # 40 px per cam. px
            (_nm(((30.0, 30.0), (34.0, 34.0))), 0.65),  # 200 px per cam. px
        ],
    )
    def test_zoomed_views(self, gpu, tiered_locs, viewport, disp_px_size):
        # deeper zooms push ordinary localizations into the tile and
        # whale tiers (footprints grow with the oversampling)
        overrides = dict(
            viewport=viewport, disp_px_size=disp_px_size, min_blur_width=0.0
        )
        ((n_cpu, img_cpu),) = _cpu(tiered_locs, "gaussian", **overrides)
        ((n_gpu, img_gpu),) = _gpu(gpu, tiered_locs, "gaussian", **overrides)
        assert n_gpu == n_cpu
        _assert_parity(img_gpu, img_cpu)

    def test_dense_hot_pixel(self, gpu):
        # many localizations on one pixel center: nothing to overflow
        # in a float gather, and the count stays exact
        n = 200_000
        ps = PIXELSIZE
        dense = pd.DataFrame(
            {
                "x": np.full(n, 32.05 * ps, np.float32),
                "y": np.full(n, 32.05 * ps, np.float32),
                "lpx": np.full(n, 0.01 * ps, np.float32),
                "lpy": np.full(n, 0.01 * ps, np.float32),
            }
        )
        ((n_cpu, img_cpu),) = _cpu(dense, "gaussian", min_blur_width=0.0)
        ((n_gpu, img_gpu),) = _gpu(gpu, dense, "gaussian", min_blur_width=0.0)
        assert n_gpu == n_cpu == n
        # 200k sequential float32 adds of ~16 accumulate ~0.1% rounding
        # error on BOTH backends (in different orders), so compare
        # looser than the general parity check
        np.testing.assert_allclose(
            img_gpu, img_cpu, rtol=1e-2, atol=1e-3 * img_cpu.max()
        )
        assert abs(img_gpu.sum() - img_cpu.sum()) < 1e-2 * img_cpu.sum()

    def test_multi_channel_order(self, gpu, locs):
        channels = [locs.iloc[i::3] for i in range(3)]
        columns = [
            render._extract_render_columns(c, "gaussian", None)
            for c in channels
        ]
        gpu_out = gpu.render_channels(
            columns, [INFO] * 3, blur_method="gaussian", **KWARGS
        )
        for channel, (n, image) in zip(channels, gpu_out):
            ((n_cpu, img_cpu),) = _cpu(channel, "gaussian")
            assert n == n_cpu
            _assert_parity(image, img_cpu)

    def test_empty_view(self, gpu, locs):
        viewport = _nm(((100.0, 100.0), (110.0, 110.0)))  # no localizations
        ((n_cpu, img_cpu),) = _cpu(locs, "gaussian", viewport=viewport)
        ((n_gpu, img_gpu),) = _gpu(gpu, locs, "gaussian", viewport=viewport)
        assert n_gpu == n_cpu == 0
        assert img_gpu.shape == img_cpu.shape
        assert not img_gpu.any()


#: the ways a shader compiler may evaluate a footprint edge
#: ``os * x + sign * 3 * (os * lp)``: rounding every operation on its
#: own, fusing ``3 * sx`` into the add, fusing ``os * x`` into the add
#: (what Metal on Apple silicon does in one of the two passes) and
#: applying the scale last
EDGE_ORDERS = ("rounded twice", "fma 3 * sx", "fma os * x", "scale last")


def _ambiguous_edge(rng, boundary, sign, sigma_range, os, order, reaches):
    """A float32 position and precision (nm) whose footprint edge
    ``xp + sign * 3 * sx`` (rendered pixels, ``xp = os * x``,
    ``sx = os * lp``, a viewport starting at 0) lies on the pixel
    ``boundary`` within float32 rounding, such that evaluated the
    ``order`` way (see ``EDGE_ORDERS``) it reaches ``boundary`` if
    ``reaches`` (falls short otherwise) while every other order does
    the opposite: a pass compiled that way and a pass compiled any
    other way disagree. None if no such edge is found."""
    f = np.float32
    os = f(os)
    lp = (rng.uniform(*sigma_range, 4096) / os).astype(f)[:, None]
    sx = os * lp
    off = f(3.0) * sx
    x0 = ((boundary - sign * off) / os).astype(f)
    # x0 and its float32 neighbors
    x = (x0 + np.arange(-4, 5) * np.spacing(x0).astype(np.float64)).astype(f)
    xp = os * x
    edges = {
        "rounded twice": xp + f(sign) * off,
        "fma 3 * sx": (
            xp.astype(np.float64) + sign * 3.0 * sx.astype(np.float64)
        ).astype(f),
        "fma os * x": (
            np.float64(os) * x.astype(np.float64)
            + sign * off.astype(np.float64)
        ).astype(f),
        "scale last": os * (x + f(sign * 3.0) * lp),
    }
    found = np.ones(x.shape, bool)
    for name, edge in edges.items():
        found &= ((np.trunc(edge) >= boundary) == reaches) == (name == order)
    rows, columns = np.nonzero(found)
    if len(rows) == 0:
        return None
    return x[rows[0], columns[0]], lp[rows[0], 0]


class TestTileBoundaries:
    """The tiled Gaussian pass computes every footprint twice: once to
    reserve slots in the tile bins (``cs_bin_count``), once to fill them
    (``cs_bin_scatter``). The shader compiler may round a footprint
    edge differently in the two (Metal fuses ``os * x`` into the edge's
    add in only one of them), so an edge within rounding of a tile
    boundary can reach one tile more or less in the fill than in the
    count. The surplus entry used to overflow into the next tile's bin
    and race with its entries, dropping or misplacing localizations
    there at random (found with a single such localization among 150k
    at 3.25 nm per rendered pixel).

    The edges are built so that each evaluation order in
    ``EDGE_ORDERS`` disagrees with all the others, in both directions,
    as another GPU may fuse differently; combinations that cannot occur
    at the tested widths skip. A 3.25 nm rendered pixel (``os`` != 1,
    without which some orders agree) and a viewport starting at 0.
    Renders repeat because the corruption was nondeterministic."""

    DISP_PX_SIZE = 3.25  # nm
    SIZE = 256  # rendered pixels = 16 x 16 tiles of 16 pixels
    VIEW = dict(
        viewport=((0.0, 0.0), (SIZE * DISP_PX_SIZE, SIZE * DISP_PX_SIZE)),
        disp_px_size=DISP_PX_SIZE,
        min_blur_width=0.0,
    )
    REPEATS = 5

    def _filler(self, rng, n=3000):
        """Ordinary pixel- and tile-tier localizations in every tile,
        which an overflowing bin entry would displace."""
        ps = self.DISP_PX_SIZE
        return pd.DataFrame(
            {
                "x": rng.uniform(8 * ps, (self.SIZE - 8) * ps, n),
                "y": rng.uniform(8 * ps, (self.SIZE - 8) * ps, n),
                "lpx": rng.uniform(0.2 * ps, 2.5 * ps, n),
                "lpy": rng.uniform(0.2 * ps, 2.5 * ps, n),
            }
        ).astype(np.float32)

    def _edge(self, rng, boundary, sign, sigma_range, order, reaches):
        return _ambiguous_edge(
            rng,
            boundary,
            sign,
            sigma_range,
            1.0 / self.DISP_PX_SIZE,
            order,
            reaches,
        )

    def _assert_repeated_parity(self, gpu, locs):
        ((n_cpu, img_cpu),) = _cpu(locs, "gaussian", **self.VIEW)
        for _ in range(self.REPEATS):
            ((n_gpu, img_gpu),) = _gpu(gpu, locs, "gaussian", **self.VIEW)
            assert n_gpu == n_cpu == len(locs)
            _assert_parity(img_gpu, img_cpu)

    # One render per evaluation order and direction: mixed together,
    # edges the count pass reaches and the fill pass does not leave
    # spare slots that absorb the opposite case's surplus in the same
    # tiles, which hid the overflow
    @pytest.mark.parametrize("order", EDGE_ORDERS)
    @pytest.mark.parametrize("reaches", [False, True])
    def test_footprint_edges_on_tile_boundaries(self, gpu, order, reaches):
        # every inner tile boundary, reached by the lower and the upper
        # edge of footprints, in x and in y
        rng = np.random.default_rng(11)
        ps = self.DISP_PX_SIZE
        rows = []
        for boundary in range(16, self.SIZE, 16):
            for sign in (-1, 1):
                for axis, other in (("x", "y"), ("y", "x")):
                    for _ in range(2):
                        edge = self._edge(
                            rng, boundary, sign, (0.5, 4.0), order, reaches
                        )
                        if edge is None:
                            continue
                        rows.append(
                            {
                                axis: edge[0],
                                "lp" + axis: edge[1],
                                other: rng.uniform(20 * ps, 236 * ps),
                                "lp" + other: rng.uniform(0.5 * ps, 4 * ps),
                            }
                        )
        if not rows:
            pytest.skip(f"no {order} edge that {reaches=} at these widths")
        edges = pd.DataFrame(rows)[["x", "y", "lpx", "lpy"]]
        locs = pd.concat(
            [self._filler(rng), edges.astype(np.float32)], ignore_index=True
        )
        self._assert_repeated_parity(gpu, locs)

    @pytest.mark.parametrize("order", EDGE_ORDERS)
    @pytest.mark.parametrize("reaches", [False, True])
    def test_footprints_on_the_whale_tile_limit(self, gpu, order, reaches):
        # 8 tiles high and 8 or 9 tiles wide depending on the rounding
        # of the right edge: 64 tiles is the tile tier's limit, 72 is a
        # whale, so the two passes may also disagree on the tier (the
        # harmful case: the count sees a whale and reserves nothing,
        # the fill scatters a tile-tier footprint). Widths keep the
        # left edge in tile 4 (the footprint at the limit) and the
        # height in tiles 4 to 11.
        rng = np.random.default_rng(12)
        ps = self.DISP_PX_SIZE
        rows = []
        # an overflowing wide footprint is faint itself; it shows as the
        # filler entries it displaces, which takes several of them
        for _ in range(12):
            edge = self._edge(rng, 192, 1, (18.8, 21.2), order, reaches)
            if edge is None:
                pytest.skip(f"no {order} edge that {reaches=} at 64 tiles")
            rows.append(
                {
                    "x": edge[0],
                    "y": rng.uniform(127 * ps, 129 * ps),
                    "lpx": edge[1],
                    "lpy": rng.uniform(20.2 * ps, 20.6 * ps),
                }
            )
        wide = pd.DataFrame(rows).astype(np.float32)
        locs = pd.concat([self._filler(rng), wide], ignore_index=True)
        self._assert_repeated_parity(gpu, locs)


class TestWgpuBlurMethods:
    def test_gaussian_iso_matches_cpu(self, gpu, locs):
        ((n_cpu, img_cpu),) = _cpu(locs, "gaussian_iso")
        ((n_gpu, img_gpu),) = _gpu(gpu, locs, "gaussian_iso")
        assert n_gpu == n_cpu
        _assert_parity(img_gpu, img_cpu)

    def test_angle_column_rotates_ellipses(self, gpu, locs_3d):
        # 'gaussian' honors a per-localization ellipse angle
        ((n_cpu, img_cpu),) = _cpu(locs_3d, "gaussian")
        ((n_gpu, img_gpu),) = _gpu(gpu, locs_3d, "gaussian")
        assert n_gpu == n_cpu
        _assert_parity(img_gpu, img_cpu)
        # and it is not the plain (axis-aligned) rendering
        ((_, img_plain),) = _cpu(locs_3d.drop(columns="angle"), "gaussian")
        assert not np.allclose(img_gpu, img_plain, rtol=RTOL)

    @pytest.mark.parametrize("blur", ["smooth", "convolve"])
    def test_filtered_hist_matches_cpu(self, gpu, locs, blur):
        # GPU histogram + the CPU backend's own image filter: identical
        # up to the rare pixel-boundary count (spread by the filter)
        ((n_cpu, img_cpu),) = _cpu(locs, blur)
        ((n_gpu, img_gpu),) = _gpu(gpu, locs, blur)
        assert n_gpu == n_cpu
        np.testing.assert_allclose(img_gpu, img_cpu, rtol=1e-4, atol=0.2)

    @pytest.mark.parametrize(
        "blur", [None, "smooth", "gaussian", "gaussian_iso"]
    )
    def test_rotated_matches_cpu(self, gpu, locs_3d, blur):
        # 3D rotation about the view center happens in-shader in f32
        # (the CPU rotates in f64): a handful of localizations land a
        # pixel over or flip in/out of view, each touching the ~10
        # pixels of its footprint, so compare by the fraction of
        # mismatching pixels rather than pixel by pixel
        ((n_cpu, img_cpu),) = _cpu(locs_3d, blur, ang=ANG)
        ((n_gpu, img_gpu),) = _gpu(gpu, locs_3d, blur, ang=ANG)
        assert abs(n_gpu - n_cpu) <= max(3, 1e-4 * n_cpu)
        close = np.isclose(
            img_gpu, img_cpu, rtol=RTOL, atol=ATOL_FRACTION * img_cpu.max()
        )
        assert (~close).mean() < 5e-4
        assert abs(img_gpu.sum() - img_cpu.sum()) < 1e-3 * img_cpu.sum()
        # the rotation really was applied (differs from the 2D render)
        ((_, img_2d),) = _cpu(locs_3d, blur)
        assert not np.allclose(img_gpu, img_2d, rtol=RTOL, atol=1e-3)

    def test_rotated_convolve_matches_cpu(self, gpu, locs_3d):
        # the blur is the channel's global precision on both backends,
        # so no rotated in-view mask is needed and the GPU renders it
        ((n_cpu, img_cpu),) = _cpu(locs_3d, "convolve", ang=ANG)
        ((n_gpu, img_gpu),) = _gpu(gpu, locs_3d, "convolve", ang=ANG)
        assert abs(n_gpu - n_cpu) <= max(3, 1e-4 * n_cpu)
        np.testing.assert_allclose(img_gpu, img_cpu, rtol=1e-4, atol=0.2)

    def test_global_precision_sets_the_convolve_blur(self, gpu, locs):
        ps = PIXELSIZE
        wide = render._extract_render_columns(
            locs, "convolve", None, global_precision=(0.6 * ps, 0.6 * ps)
        )
        narrow = render._extract_render_columns(
            locs, "convolve", None, global_precision=(0.1 * ps, 0.1 * ps)
        )
        ((_, img_wide),) = gpu.render_channels(
            [wide], [INFO], blur_method="convolve", **KWARGS
        )
        ((_, img_narrow),) = gpu.render_channels(
            [narrow], [INFO], blur_method="convolve", **KWARGS
        )
        assert img_wide.max() < img_narrow.max()  # wider blur, flatter
        ((_, img_cpu),) = backend_mod._cpu_backend().render_channels(
            [wide], [INFO], blur_method="convolve", **KWARGS
        )
        np.testing.assert_allclose(img_wide, img_cpu, rtol=1e-4, atol=0.2)


class TestResidentUploads:
    """Uploads persist across renders of the same memory, are bounded
    by the VRAM budget (LRU), and oversize channels render in chunks."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self, gpu):
        gpu.release_uploads()
        yield
        gpu.release_uploads()

    def _budget(self, monkeypatch, value):
        from picasso.render.gpu import backend_wgpu

        monkeypatch.setattr(backend_wgpu, "vram_budget_bytes", lambda: value)

    def test_fresh_views_reuse_the_upload(self, gpu, locs):
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        uploads = gpu.upload_count
        assert len(gpu._arrays) == 4  # x, y, lpx, lpy
        # new column objects over the same DataFrame memory: no upload,
        # and the histogram shares the resident x and y
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        gpu.render_channels(
            _columns(locs, None), [INFO], blur_method=None, **KWARGS
        )
        assert gpu.upload_count == uploads
        assert len(gpu._arrays) == 4
        assert gpu.resident_bytes == 4 * 4 * len(locs)

    def test_changed_data_uploads_again(self, gpu, locs):
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        uploads = gpu.upload_count
        modified = locs.copy()  # new memory, as filtering/undrifting make
        gpu.render_channels(
            _columns(modified, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        assert gpu.upload_count == uploads + 4
        assert len(gpu._arrays) == 8

    def test_budget_evicts_least_recently_rendered(
        self, gpu, locs, monkeypatch
    ):
        one = 4 * 4 * len(locs)
        self._budget(monkeypatch, int(1.5 * one))
        a, b = locs.copy(), locs.copy()
        gpu.render_channels(
            _columns(a, "gaussian"), [INFO], blur_method="gaussian", **KWARGS
        )
        gpu.render_channels(
            _columns(b, "gaussian"), [INFO], blur_method="gaussian", **KWARGS
        )
        # a's least recently rendered columns were evicted to fit b
        assert gpu.resident_bytes <= 1.5 * one
        uploads = gpu.upload_count
        gpu.render_channels(
            _columns(b, "gaussian"), [INFO], blur_method="gaussian", **KWARGS
        )
        assert gpu.upload_count == uploads  # b is fully resident
        # both channels in one request fit together within the budget
        # only if nothing else is resident: eviction never touches
        # channels of the current request
        self._budget(monkeypatch, int(2.2 * one))
        gpu.render_channels(
            [c for c in _columns(a, "gaussian") + _columns(b, "gaussian")],
            [INFO] * 2,
            blur_method="gaussian",
            **KWARGS,
        )
        assert len(gpu._arrays) == 8
        assert gpu.resident_bytes == 2 * one

    @pytest.mark.parametrize("blur", ["gaussian", None])
    def test_oversize_channel_renders_in_chunks(
        self, gpu, locs, monkeypatch, blur
    ):
        one = 4 * 4 * len(locs)
        self._budget(monkeypatch, one // 3)
        uploads = gpu.upload_count
        ((n_gpu, img_gpu),) = _gpu(gpu, locs, blur)
        ((n_cpu, img_cpu),) = _cpu(locs, blur)
        assert n_gpu == n_cpu
        if blur:
            _assert_parity(img_gpu, img_cpu)
        else:
            assert img_gpu.sum() == img_cpu.sum()
        assert gpu.upload_count > uploads  # chunk uploads happened...
        assert len(gpu._arrays) == 0  # ...but none stays resident
        assert gpu._temp_buffers == []  # and they are freed afterwards

    @pytest.mark.parametrize("step,start", [(7, 0), (5, 3), (1, 1000)])
    def test_strided_views_render_from_resident_buffers(
        self, gpu, locs, step, start
    ):
        # the GUI's interactive previews are iloc[::step] of the whole
        # channels: rendered through stride/offset, no upload at all
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        uploads = gpu.upload_count
        subset = locs.iloc[start::step]
        for blur in ("gaussian", None):
            ((n_gpu, img_gpu),) = _gpu(gpu, subset, blur)
            ((n_cpu, img_cpu),) = _cpu(subset, blur)
            assert n_gpu == n_cpu
            if blur:
                _assert_parity(img_gpu, img_cpu)
            else:
                assert img_gpu.sum() == img_cpu.sum()
        assert gpu.upload_count == uploads
        assert len(gpu._arrays) == 4

    def test_release_uploads(self, gpu, locs):
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        assert gpu.resident_bytes > 0
        gpu.release_uploads()
        assert gpu.resident_bytes == 0 and len(gpu._arrays) == 0
        # rendering afterwards works and uploads afresh
        uploads = gpu.upload_count
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        assert gpu.upload_count == uploads + 4


class TestIndexedRenders:
    """A row selection (the viewport pyramid's) renders from the
    resident columns through an index list: no column upload, and the
    shader's in-view mask prunes the pyramid's overspill."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self, gpu):
        gpu.release_uploads()
        yield
        gpu.release_uploads()

    @pytest.fixture
    def selection(self, locs):
        # a block-like superset: rows in and around a zoomed viewport
        x = locs["x"].to_numpy() / PIXELSIZE  # camera pixels
        y = locs["y"].to_numpy() / PIXELSIZE
        return np.flatnonzero(
            (x > 15) & (x < 50) & (y > 18) & (y < 46)
        ).astype(np.uint32)

    ZOOMED = dict(
        viewport=_nm(((20.0, 20.0), (44.0, 44.0))), disp_px_size=3.25
    )

    @pytest.mark.parametrize("blur", ["gaussian", None, "smooth"])
    def test_indexed_matches_cpu(self, gpu, locs, selection, blur):
        columns = [
            render._extract_render_columns(locs, blur, None, indices=selection)
        ]
        kwargs = dict(KWARGS, **self.ZOOMED)
        ((n_gpu, img_gpu),) = gpu.render_channels(
            columns, [INFO], blur_method=blur, **kwargs
        )
        ((n_cpu, img_cpu),) = _cpu(locs.iloc[selection], blur, **self.ZOOMED)
        assert n_gpu == n_cpu
        if blur == "gaussian":
            _assert_parity(img_gpu, img_cpu)
        elif blur is None:
            assert img_gpu.sum() == img_cpu.sum()
        else:
            np.testing.assert_allclose(img_gpu, img_cpu, rtol=1e-4, atol=0.2)

    def test_indexed_renders_reuse_resident_columns(
        self, gpu, locs, selection
    ):
        gpu.render_channels(
            _columns(locs, "gaussian"),
            [INFO],
            blur_method="gaussian",
            **KWARGS,
        )
        uploads = gpu.upload_count
        kwargs = dict(KWARGS, **self.ZOOMED)
        for step in (1, 5):
            columns = [
                render._extract_render_columns(
                    locs, "gaussian", None, indices=selection[::step]
                )
            ]
            ((n, image),) = gpu.render_channels(
                columns, [INFO], blur_method="gaussian", **kwargs
            )
            ((n_cpu, img_cpu),) = _cpu(
                locs.iloc[selection[::step]], "gaussian", **self.ZOOMED
            )
            assert n == n_cpu
            _assert_parity(image, img_cpu)
        assert gpu.upload_count == uploads  # only index lists moved
        assert gpu._temp_buffers == []  # and they were freed
        assert len(gpu._arrays) == 4

    def test_indexed_oversize_channel_is_gathered(
        self, gpu, locs, selection, monkeypatch
    ):
        from picasso.render.gpu import backend_wgpu

        monkeypatch.setattr(
            backend_wgpu, "vram_budget_bytes", lambda: 4 * 4 * len(locs) // 3
        )
        columns = [
            render._extract_render_columns(
                locs, "gaussian", None, indices=selection
            )
        ]
        kwargs = dict(KWARGS, **self.ZOOMED)
        ((n, image),) = gpu.render_channels(
            columns, [INFO], blur_method="gaussian", **kwargs
        )
        ((n_cpu, img_cpu),) = _cpu(
            locs.iloc[selection], "gaussian", **self.ZOOMED
        )
        assert n == n_cpu
        _assert_parity(image, img_cpu)

    def test_empty_selection(self, gpu, locs):
        columns = [
            render._extract_render_columns(
                locs, "gaussian", None, indices=np.zeros(0, np.uint32)
            )
        ]
        ((n, image),) = gpu.render_channels(
            columns, [INFO], blur_method="gaussian", **KWARGS
        )
        assert n == 0 and not image.any()


class TestRequestValidation:
    """Every request the GPU cannot honor raises ``SplatBackendError``, the
    one exception the seam turns into a CPU fallback - anything else would
    reach the user as a failed render."""

    @staticmethod
    def _render(gpu, columns, blur, **overrides):
        kwargs = dict(KWARGS, **overrides)
        return gpu.render_channels(
            [columns], [INFO], blur_method=blur, **kwargs
        )

    def test_unknown_blur_method(self, gpu, locs):
        columns = render._extract_render_columns(locs.head(100), None, None)
        with pytest.raises(SplatBackendError, match="unknown blur_method"):
            self._render(gpu, columns, "bilateral")

    def test_viewport_is_required(self, gpu, locs):
        columns = render._extract_render_columns(locs.head(100), None, None)
        with pytest.raises(SplatBackendError, match="explicit viewport"):
            self._render(gpu, columns, None, viewport=None)

    def test_rotation_needs_z(self, gpu, locs):
        columns = render._extract_render_columns(locs.head(100), None, None)
        with pytest.raises(SplatBackendError, match="needs z"):
            self._render(gpu, columns, None, ang=ANG)

    @pytest.mark.parametrize("blur", ["gaussian", "gaussian_iso"])
    def test_per_localization_blur_needs_the_precision(self, gpu, locs, blur):
        # extracted for a histogram, so without lpx and lpy
        columns = render._extract_render_columns(locs.head(100), None, None)
        with pytest.raises(SplatBackendError, match="localization precision"):
            self._render(gpu, columns, blur)

    def test_rotated_gaussian_needs_lpz(self, gpu, locs_3d):
        # "convolve" extracts lpx/lpy and, rotated, z - but never lpz
        columns = render._extract_render_columns(
            locs_3d.head(100), "convolve", ANG
        )
        assert columns.lpx is not None and columns.lpz is None
        with pytest.raises(SplatBackendError, match="needs lpz"):
            self._render(gpu, columns, "gaussian", ang=ANG)

    def test_channel_over_the_storage_limit(self, gpu, locs, monkeypatch):
        monkeypatch.setattr(gpu, "_max_channel_locs", 50)
        columns = render._extract_render_columns(locs.head(100), None, None)
        with pytest.raises(SplatBackendError, match="storage-binding limit"):
            self._render(gpu, columns, None)

    def test_empty_viewport(self, gpu, locs):
        columns = render._extract_render_columns(locs.head(100), None, None)
        flat = _nm(((3.0, 2.0), (3.0, 62.5)))
        with pytest.raises(SplatBackendError, match="empty viewport"):
            self._render(gpu, columns, None, viewport=flat)

    def test_an_unexpected_error_is_wrapped(self, gpu, locs, monkeypatch):
        def boom(*args, **kwargs):
            raise MemoryError("device lost")

        monkeypatch.setattr(gpu, "_render", boom)
        columns = render._extract_render_columns(locs.head(100), None, None)
        with pytest.raises(SplatBackendError, match="device lost") as info:
            self._render(gpu, columns, None)
        assert isinstance(info.value.__cause__, MemoryError)


class TestSeamSettings:
    """settings["Render"]["gpu"] selects the GPU; unsupported requests
    still fall back to the CPU through the seam."""

    def _settings(self, monkeypatch, gpu):
        from picasso import lib

        monkeypatch.setattr(
            lib.io, "load_user_settings", lambda: {"Render": {"gpu": gpu}}
        )

    def test_enabled_selects_gpu_and_falls_back(
        self, gpu, locs, locs_3d, monkeypatch
    ):
        from picasso.render.gpu import WgpuBackend

        self._settings(monkeypatch, {"enabled": "on"})
        monkeypatch.setattr(backend_mod, "_gpu_singleton", gpu)
        monkeypatch.setattr(backend_mod, "_gpu_unavailable", False)
        monkeypatch.setattr(backend_mod, "_gpu_adapter", "high-performance")
        assert isinstance(backend_mod._get_backend(), WgpuBackend)
        assert backend_mod.describe_active().startswith("GPU (")
        # supported: rendered on the GPU
        ((n, _),) = render._render_channels(
            [locs], [INFO], blur_method="gaussian", **KWARGS
        )
        ((n_ref, _),) = _cpu(locs, "gaussian")
        assert n == n_ref
        # a request the GPU refuses (a 3D rotation of data without z)
        # is silently rendered on the CPU, hence bit-identical to it
        kwargs = dict(KWARGS, ang=ANG)
        flat = locs_3d.assign(z=0.0)
        monkeypatch.setattr(
            gpu,
            "render_channels",
            lambda *a, **k: (_ for _ in ()).throw(SplatBackendError("forced")),
        )
        ((n_conv, img_conv),) = render._render_channels(
            [flat], [INFO], blur_method="convolve", **kwargs
        )
        ((n_cpu, img_cpu),) = _cpu(flat, "convolve", ang=ANG)
        assert n_conv == n_cpu
        np.testing.assert_array_equal(img_conv, img_cpu)

    def test_off_selects_cpu(self, monkeypatch):
        from picasso.render.splat import CpuBackend

        self._settings(monkeypatch, {"enabled": "off"})
        assert isinstance(backend_mod._get_backend(), CpuBackend)


class TestAdapterSelection:
    def test_name_substring_selects_the_adapter(self, gpu):
        from picasso.render.gpu import WgpuBackend

        name = gpu.describe().split(" via ")[0]
        chosen = WgpuBackend(adapter=name[:5].lower())
        try:
            assert chosen.describe().startswith(name)
        finally:
            chosen.close()

    def test_unknown_adapter_name_raises(self):
        from picasso.render.gpu import WgpuBackend

        with pytest.raises(SplatBackendError):
            WgpuBackend(adapter="no such graphics card")

    def test_low_power_preference_works(self):
        from picasso.render.gpu import WgpuBackend

        backend = WgpuBackend(adapter="low-power")
        try:
            assert " via " in backend.describe()
        finally:
            backend.close()


class TestIdleClose:
    """Releasing the GPU while the GUI is idle (``picasso.gui.idle``):
    never under a render in progress, and the next render starts a new
    device that renders the same."""

    def test_close_without_waiting_skips_a_render_in_progress(self, locs):
        from picasso.render.gpu import WgpuBackend

        backend = WgpuBackend()
        try:
            _gpu(backend, locs, "gaussian")
            assert backend.resident_bytes > 0
            with backend._lock:  # as held by a render on another thread
                assert backend.close(wait=False) is False
            assert backend.resident_bytes > 0  # nothing was released
            assert backend.close(wait=False) is True
            assert backend.resident_bytes == 0 and not backend._scratch
            # a caller that fetched the backend before it was closed
            # gets the error that sends its render to the CPU
            with pytest.raises(SplatBackendError):
                _gpu(backend, locs, "gaussian")
            assert backend.close() is True  # closing twice is harmless
        finally:
            backend.close()

    def test_the_next_render_starts_a_new_device(self, locs, monkeypatch):
        from picasso import lib
        from picasso.render.gpu import WgpuBackend

        monkeypatch.setattr(
            lib.io,
            "load_user_settings",
            lambda: {"Render": {"gpu": {"enabled": "on"}}},
        )
        monkeypatch.setattr(backend_mod, "_gpu_singleton", None)
        monkeypatch.setattr(backend_mod, "_gpu_adapter", None)
        monkeypatch.setattr(backend_mod, "_gpu_unavailable", False)
        first = backend_mod._get_backend()
        assert isinstance(first, WgpuBackend)
        ((n, image),) = render._render_channels(
            [locs], [INFO], blur_method="gaussian", **KWARGS
        )
        try:
            assert backend_mod.close(wait=False) is True
            assert backend_mod._gpu_singleton is None
            assert backend_mod.close(wait=False) is True  # nothing to close
            second = backend_mod._get_backend()
            assert isinstance(second, WgpuBackend) and second is not first
            ((n_again, image_again),) = render._render_channels(
                [locs], [INFO], blur_method="gaussian", **KWARGS
            )
            assert backend_mod.last_fallback() is None  # on the GPU
            assert n_again == n
            _assert_parity(image_again, image)  # float atomics
        finally:
            backend_mod.close()
