"""The asynchronous render worker of ``picasso.gui.render.View``.

Full renders run on a worker thread with a latest-wins queue
(``RenderWorker``): a burst of pan/zoom events collapses into rendering
the newest state, stale results are discarded by request id, and the
result must be pixel-identical to the synchronous path. The rest of the
suite runs with ``async_rendering`` disabled (see conftest); these tests
re-enable it explicitly.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore

from picasso import render
from picasso.gui import render as gui_render


WIDTH = HEIGHT = 64.0
PIXELSIZE = 130.0


def _locs(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.0, WIDTH, size=n),
            "y": rng.uniform(0.0, HEIGHT, size=n),
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "photons": rng.uniform(500.0, 5000.0, size=n),
            "frame": rng.integers(0, 1000, size=n).astype(np.int32),
        }
    )


def _info() -> list[dict]:
    return [
        {
            "Width": WIDTH,
            "Height": HEIGHT,
            "Frames": 1000,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _qimage_bytes(qimage) -> bytes:
    bits = qimage.bits()
    bits.setsize(qimage.sizeInBytes())
    return bytes(bits)


def _wait_until(qapp, condition, timeout: float = 15.0) -> None:
    start = time.monotonic()
    while not condition():
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("render worker result did not arrive")


@pytest.fixture
def window(qt_offscreen, tmp_path):
    """A Render window with one channel and async rendering enabled."""
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    window.view.async_rendering = True  # instance attr shadows the stub
    yield window
    window.view.stop_render_worker()


class _Pinch:
    """A macOS pinch gesture as the view's event() sees it."""

    def __init__(self, x, y, value):
        self._pos = QtCore.QPointF(x, y)
        self._value = value

    def type(self):
        return QtCore.QEvent.Type.NativeGesture

    def gestureType(self):
        return QtCore.Qt.NativeGestureType.ZoomNativeGesture

    def value(self):
        return self._value

    def position(self):
        return self._pos


def test_pinch_zooms_the_main_view_about_the_fingers(window):
    view = window.view
    view.async_rendering = False
    view.update_scene()
    x, y = 96, 32
    before = view.map_to_movie(QtCore.QPointF(x, y))
    vh, vw = render.viewport_size(view.viewport)
    assert view.event(_Pinch(x, y, 0.25))  # fingers apart: zoom in
    after = view.map_to_movie(QtCore.QPointF(x, y))
    assert after == pytest.approx(before, abs=1e-6)
    vh2, vw2 = render.viewport_size(view.viewport)
    assert vw2 / vw == pytest.approx(1 / 1.25, rel=1e-6)
    assert view.event(_Pinch(x, y, -0.2))  # fingers together: zoom out
    vh3, vw3 = render.viewport_size(view.viewport)
    assert vw3 / vw2 == pytest.approx(1 / 0.8, rel=1e-6)


class TestAsyncRender:
    def test_async_matches_sync(self, window, qapp, monkeypatch):
        # margin off: with it, async legitimately renders a larger
        # viewport than the sync path and raw images differ in coverage
        monkeypatch.setattr(gui_render, "VIEWPORT_MARGIN", 0.0)
        view = window.view
        view.update_scene()
        request_id = view._render_request_id
        assert request_id > 0
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        async_qimage = view.qimage
        async_raw = view.image.copy()

        # synchronous reference of the same state
        view.async_rendering = False
        view.image = None
        view.update_scene()
        assert view.image is not None
        np.testing.assert_array_equal(async_raw, view.image)
        assert _qimage_bytes(async_qimage) == _qimage_bytes(view.qimage)

    def test_burst_coalesces(self, window, qapp, monkeypatch):
        view = window.view
        calls = []
        original = render.render_scene
        # The worker is held inside its first render until the whole
        # burst has been submitted, so coalescing is asserted as a
        # property of the latest-wins queue rather than of thread
        # scheduling: on a fast workstation this tiny render finished
        # between two submissions and every request got rendered.
        burst_submitted = threading.Event()

        def counting(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                assert burst_submitted.wait(15.0), "burst never submitted"
            return original(*args, **kwargs)

        monkeypatch.setattr(render, "render_scene", counting)
        n_requests = 6
        for _ in range(n_requests):
            view.update_scene()
        final_id = view._render_request_id
        burst_submitted.set()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        # drain any second render picked up after the first finished
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert len(calls) <= 2 < n_requests
        assert view._render_request_id == final_id

    def test_stale_result_discarded(self, window, qapp, monkeypatch):
        monkeypatch.setattr(gui_render, "VIEWPORT_MARGIN", 0.0)
        view = window.view
        zoomed = ((0.0, 0.0), (HEIGHT / 2, WIDTH / 2))
        view.update_scene()
        stale_id = view._render_request_id
        # supersede immediately with a different viewport; a result is
        # only applied when its id matches the newest request, so
        # view.image stays unset until the *zoomed* render lands
        view.update_scene(viewport=zoomed)
        assert view._render_request_id > stale_id
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        async_raw = view.image.copy()

        # the applied frame must equal a synchronous render of the
        # final (zoomed) viewport, not of the superseded overview
        view.async_rendering = False
        view.update_scene(viewport=zoomed)
        np.testing.assert_array_equal(async_raw, view.image)

    def test_worker_stops_cleanly(self, window, qapp):
        view = window.view
        view.update_scene()
        view.stop_render_worker()
        assert view._render_thread is None
        # stopping twice is a no-op
        view.stop_render_worker()

    def test_interactive_preview_and_refine(self, window, qapp, monkeypatch):
        view = window.view
        monkeypatch.setattr(
            gui_render.View,
            "_interaction_subsample_target",
            lambda self, population=0: 500,
        )
        seen = []
        original = render.render_scene

        def recording(*args, **kwargs):
            locs = kwargs["locs"]
            channels = [locs] if isinstance(locs, pd.DataFrame) else locs
            seen.append((sum(len(c) for c in channels), kwargs["contrast"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(render, "render_scene", recording)
        view.update_scene(interactive=True)
        # the preview renders the strided subsample; the refine timer
        # follows up with the full-quality render, which is the one
        # allowed to update the caches and spinboxes. The dialog values
        # themselves shift with the dynamic display pixel size, so the
        # preview is checked against the refine render's contrast: it
        # must be the same limits scaled by the sampled fraction.
        _wait_until(qapp, lambda: len(seen) >= 2)
        n_preview, contrast_preview = seen[0]
        n_full, contrast_full = seen[1]
        assert n_preview == 500
        assert n_full == 3000
        fraction = n_preview / n_full
        assert contrast_preview == pytest.approx(
            (contrast_full[0] * fraction, contrast_full[1] * fraction)
        )
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)

    def test_interaction_subsample_setting(self, window, monkeypatch):
        view = window.view

        def with_value(value):
            settings = {"Render": {"interaction_subsample": value}}
            monkeypatch.setattr(
                gui_render.io, "load_user_settings", lambda: settings
            )
            return view._interaction_subsample_target()

        assert with_value("off") == 0
        assert with_value(0) == 0
        assert with_value(1234) == 1234
        assert with_value(True) == gui_render.INTERACTION_SUBSAMPLE_AUTO
        assert with_value("auto") == gui_render.INTERACTION_SUBSAMPLE_AUTO
        assert with_value(-5) == gui_render.INTERACTION_SUBSAMPLE_AUTO
        # auto: at least the fixed count and at least a fraction of the
        # visible population; an explicit count ignores the population
        auto = gui_render.INTERACTION_SUBSAMPLE_AUTO
        fraction = gui_render.INTERACTION_SUBSAMPLE_FRACTION
        monkeypatch.setattr(
            gui_render.io, "load_user_settings", lambda: {"Render": {}}
        )
        assert view._interaction_subsample_target(10**6) == auto
        assert view._interaction_subsample_target(10**8) == int(
            fraction * 10**8
        )
        monkeypatch.setattr(
            gui_render.io,
            "load_user_settings",
            lambda: {"Render": {"interaction_subsample": 1234}},
        )
        assert view._interaction_subsample_target(10**8) == 1234

    def test_max_blur_width_setting(self, window, monkeypatch):
        view = window.view
        default = gui_render.lib.RENDER_MAX_BLUR_WIDTH_DEFAULT

        def with_value(value):
            settings = {"Render": {"max_blur_width": value}}
            monkeypatch.setattr(
                gui_render.io, "load_user_settings", lambda: settings
            )
            return view._max_blur_width()

        assert with_value(250) == 250.0
        assert with_value(0.5) == 0.5
        assert with_value(0) is None
        assert with_value("off") is None
        assert with_value(-3) is None
        assert with_value(True) == default
        assert with_value("nonsense") == default
        # missing key (the window's closeEvent expects the section)
        monkeypatch.setattr(
            gui_render.io, "load_user_settings", lambda: {"Render": {}}
        )
        assert view._max_blur_width() == default
        # the render request carries it in camera pixels
        kwargs = view.get_render_kwargs()
        assert kwargs["max_blur_width"] == pytest.approx(
            default / view.pixelsize
        )

    def test_persistent_backend_gets_whole_channels(self, window, monkeypatch):
        view = window.view
        real_loader = gui_render.io.load_user_settings

        class Persistent(render.SplatBackend):
            name = "fake-gpu"
            persistent_uploads = True

            def render_channels(self, columns, info_arg, **kwargs):
                return render.backend._cpu_backend().render_channels(
                    columns, info_arg, **kwargs
                )

        backend_mod = gui_render.render.backend
        zoomed = ((0.0, 0.0), (HEIGHT / 4, WIDTH / 4))
        request = {
            "locs": view.locs[0],
            "viewport": zoomed,
            "contrast": (0.0, 1.0),
        }
        # CPU backend: previews subsample when over the target
        monkeypatch.setattr(
            view, "_interaction_subsample_target", lambda population=0: 10
        )
        assert view._subsample_request(dict(request))
        # a backend with resident uploads: whole channels; previews are
        # strided views of them (rendered from the resident buffers)
        monkeypatch.setattr(
            backend_mod, "_get_backend", lambda **kw: Persistent()
        )
        locs, _ = view._prepare_locs_for_rendering(viewport=zoomed)
        locs = locs if isinstance(locs, list) else [locs]  # single channel
        assert len(locs[0]) == len(view.locs[0])
        assert locs[0] is view.locs[0]  # the very same memory, no copy
        preview = dict(request)
        assert view._subsample_request(preview)
        assert np.shares_memory(
            preview["locs"]["x"].to_numpy(), view.locs[0]["x"].to_numpy()
        )
        # the request carries the pyramid's row selection per channel
        # (the zoomed view covers under 10% of the FOV, so the pyramid
        # is not bypassed) and previews subsample that selection
        monkeypatch.setattr(
            view, "_interaction_subsample_target", lambda population=0: 200
        )
        indices = view._render_indices(zoomed)
        assert len(indices) == 1 and indices[0] is not None
        in_view = len(indices[0])
        assert 0 < in_view < len(view.locs[0])
        full_view = view.viewport
        view.viewport = [tuple(zoomed[0]), tuple(zoomed[1])]
        request, _ = view._build_render_request()
        view.viewport = full_view
        assert request["indices"][0] is not None
        preview = {
            "locs": view.locs[0],
            "indices": [indices[0]],
            "viewport": zoomed,
            "contrast": (0.0, 1.0),
        }
        assert view._subsample_request(preview)
        expected_step = -(-in_view // 200)
        assert preview["locs"] is view.locs[0]  # the channel stays whole
        assert len(preview["indices"][0]) == len(indices[0][::expected_step])
        # the bypass regime (large viewport) hands over no selection
        assert view._render_indices(((0.0, 0.0), (HEIGHT, WIDTH))) == [None]
        # the CPU path keeps its own slicing: no indices in its requests
        monkeypatch.setattr(
            backend_mod, "_get_backend", render.backend._cpu_backend
        )
        request, _ = view._build_render_request()
        assert request["indices"] is None
        monkeypatch.setattr(
            backend_mod, "_get_backend", lambda **kw: Persistent()
        )
        # closing the datasets releases the resident uploads (the real
        # settings are needed again: the rebuilt UI reads other keys)
        monkeypatch.setattr(gui_render.io, "load_user_settings", real_loader)
        released = []
        monkeypatch.setattr(
            backend_mod, "release_uploads", lambda: released.append(1)
        )
        window.remove_locs()
        assert released

    def test_property_preview_targets_visible_population(
        self, window, monkeypatch
    ):
        # render by property renders whole channels without a row
        # selection; a zoomed-in preview must stride by the in-view
        # population, not the whole FOV's, or it thins out to nothing
        view = window.view
        window.display_settings_dlg.parameter.setCurrentText("frame")
        window.display_settings_dlg.render_check.setChecked(True)
        view.activate_render_property()
        assert view.x_render_state
        monkeypatch.setattr(
            view, "_interaction_subsample_target", lambda population=0: 50
        )
        zoomed = ((0.0, 0.0), (HEIGHT / 4, WIDTH / 4))
        locs, _ = view._prepare_locs_for_rendering(viewport=zoomed)
        assert sum(len(channel) for channel in locs) == len(view.locs[0])
        in_view = len(view._viewport_indices(0, zoomed))
        assert 50 < in_view < len(view.locs[0])
        preview = {"locs": locs, "viewport": zoomed, "contrast": (0.0, 1.0)}
        assert view._subsample_request(preview)
        sampled = pd.concat(preview["locs"])
        visible = sampled[
            (sampled["x"] < WIDTH / 4) & (sampled["y"] < HEIGHT / 4)
        ]
        # about the target in view (a whole-FOV stride would leave ~3)
        assert len(visible) >= 25
        # a full-FOV preview is unchanged: strided over all rows
        full = {
            "locs": locs,
            "viewport": ((0.0, 0.0), (HEIGHT, WIDTH)),
            "contrast": (0.0, 1.0),
        }
        assert view._whole_channel_view_fraction(full) == 1.0

    def test_ctrl_left_drag_pans_in_every_tool(self, window, qapp):
        from PyQt6 import QtCore, QtGui

        view = window.view
        # a rendered frame first, as in real use (picks draw onto it)
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        left = QtCore.Qt.MouseButton.LeftButton
        right = QtCore.Qt.MouseButton.RightButton
        ctrl = QtCore.Qt.KeyboardModifier.ControlModifier
        plain = QtCore.Qt.KeyboardModifier.NoModifier

        def send(kind, pos, button, modifiers):
            event = QtGui.QMouseEvent(
                kind,
                QtCore.QPointF(*pos),
                QtCore.QPointF(*pos),
                button,
                button,
                modifiers,
            )
            handler = {
                QtCore.QEvent.Type.MouseButtonPress: view.mousePressEvent,
                QtCore.QEvent.Type.MouseMove: view.mouseMoveEvent,
                QtCore.QEvent.Type.MouseButtonRelease: view.mouseReleaseEvent,
            }[kind]
            handler(event)

        def drag(button, modifiers):
            send(
                QtCore.QEvent.Type.MouseButtonPress,
                (60, 60),
                button,
                modifiers,
            )
            assert view._pan
            send(QtCore.QEvent.Type.MouseMove, (40, 60), button, modifiers)
            send(
                QtCore.QEvent.Type.MouseButtonRelease,
                (40, 60),
                button,
                modifiers,
            )
            assert not view._pan

        # Ctrl + left drag pans whatever the tool, adding no pick or point
        for mode in ("Zoom", "Pick", "Measure"):
            view._mode = mode
            before = [tuple(edge) for edge in view.viewport]
            drag(left, ctrl)
            assert [tuple(edge) for edge in view.viewport] != before
            assert view._picks == []
            assert view._points == []
        # the right button still pans in the zoom tool
        view._mode = "Zoom"
        before = [tuple(edge) for edge in view.viewport]
        drag(right, plain)
        assert [tuple(edge) for edge in view.viewport] != before
        # and a plain left click in the pick tool still picks
        view._mode = "Pick"
        send(QtCore.QEvent.Type.MouseButtonPress, (60, 60), left, plain)
        assert not view._pan
        send(QtCore.QEvent.Type.MouseButtonRelease, (60, 60), left, plain)
        assert len(view._picks) == 1

    def test_instant_geometric_preview(self, window, qapp):
        view = window.view
        # establish a fully rendered frame first
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        before = _qimage_bytes(view.qimage)
        shown_before = view._displayed_viewport
        # an interactive viewport change must update the displayed frame
        # synchronously (geometric blit of the stale frame), before any
        # worker result can possibly be applied
        shifted = ((10.0, 10.0), (10.0 + HEIGHT, 10.0 + WIDTH))
        view.update_scene(viewport=shifted, interactive=True)
        assert view._displayed_viewport != shown_before
        assert view._displayed_viewport == (
            tuple(view.viewport[0]),
            tuple(view.viewport[1]),
        )
        assert _qimage_bytes(view.qimage) != before

    def test_stale_result_adopted_for_display(self, window, qapp):
        view = window.view
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        image_cache = view.image
        n_locs_before = view.n_locs
        shown_before = _qimage_bytes(view.qimage)
        # fabricate a superseded completion: older id, shifted viewport
        stale_viewport = ((5.0, 5.0), (5.0 + HEIGHT, 5.0 + WIDTH))
        view._render_request_id += 1  # pretend a newer request exists
        fake = view.qimage_no_picks.copy()
        view._on_render_finished(
            view._render_request_id - 1,
            stale_viewport,
            fake,
            123,
            (0.0, 1.0),
            np.zeros((4, 4), dtype=np.float32),
        )
        # adopted for display: repositioned into the current viewport
        assert view._displayed_viewport == view._viewport_key()
        assert _qimage_bytes(view.qimage) != shown_before
        # ...but caches and spinboxes stay untouched by stale results
        assert view.image is image_cache
        assert view.n_locs == n_locs_before

    @pytest.mark.parametrize("size", [WIDTH, 20.0, 3.0, 0.7])
    def test_n_locs_counts_the_visible_viewport(self, window, qapp, size):
        # the render covers a margin and the pyramid query whole edge
        # blocks; neither may leak into the count shown in Info
        view = window.view
        viewport = ((10.2, 10.2), (10.2 + size, 10.2 + size))
        view.update_scene(viewport=viewport)
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        (y_min, x_min), (y_max, x_max) = view.viewport
        locs = view.locs[0]
        expected = int(
            (
                (locs["x"] > x_min)
                & (locs["x"] < x_max)
                & (locs["y"] > y_min)
                & (locs["y"] < y_max)
            ).sum()
        )
        assert view.n_locs == expected

    def test_remove_locs_stops_worker(self, window, qapp):
        # "Remove all localizations" rebuilds the view; the old view's
        # worker thread must be stopped first — a running QThread being
        # destroyed aborts the whole process (field-reported crash)
        old_view = window.view
        window.remove_locs()
        assert old_view._render_thread is None
        assert window.view is not old_view
        assert window.view._render_thread is not None
        window.view.stop_render_worker()

    def test_margin_inflates_render_viewport(self, window, qapp):
        view = window.view
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        (t0, t1), (t2, t3) = view._viewport_key()
        (r0, r1), (r2, r3) = view._blit_viewport
        expected = 1.0 + 2 * gui_render.VIEWPORT_MARGIN
        assert (r3 - r1) / (t3 - t1) == pytest.approx(expected)
        assert (r2 - r0) / (t2 - t0) == pytest.approx(expected)
        # the margin extends symmetrically beyond the visible view
        assert r0 < t0 and r1 < t1 and r2 > t2 and r3 > t3
        # while the displayed frame is the window-sized visible crop
        assert view.qimage_no_picks.width() >= view.width()

    def test_cache_redraw_after_margin_render(self, window, qapp):
        view = window.view
        view.update_scene()
        _wait_until(qapp, lambda: getattr(view, "image", None) is not None)
        assert view._image_viewport == view._blit_viewport
        shown = _qimage_bytes(view.qimage)
        # the contrast/cache path (synchronous) must crop the cached
        # margin image identically
        view.draw_scene(view.viewport, use_cache=True)
        assert _qimage_bytes(view.qimage) == shown
