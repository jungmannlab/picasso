"""The asynchronous render worker of the 3D rotation window
(``picasso.gui.rotation.ViewRotation``).

Rotation and pan drags used to re-render synchronously on the GUI
thread on every mouse move. Now full renders run on a worker thread
with a latest-wins queue (``render_worker.RenderWorker``): a burst of
drag events collapses into rendering the newest orientation, previews
of large picks are subsampled and refined on idle, and the result must
be pixel-identical to the synchronous path, which stays available. The
rest of the suite renders synchronously (see conftest); these tests
re-enable the worker explicitly.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtWidgets

from picasso import render
from picasso.gui import render as gui_render
from picasso.gui import rotation


WIDTH = HEIGHT = 64.0
PIXELSIZE = 130.0
N_LOCS = 4000


def _locs(n: int = N_LOCS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.uniform(24.0, 40.0, size=n),
            "y": rng.uniform(24.0, 40.0, size=n),
            "z": rng.uniform(-300.0, 300.0, size=n),  # nm
            "lpx": rng.uniform(0.05, 0.3, size=n),
            "lpy": rng.uniform(0.05, 0.3, size=n),
            "lpz": rng.uniform(10.0, 60.0, size=n),  # nm
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


def _drain(qapp, seconds: float = 0.5) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def rotation_view(qt_offscreen, tmp_path, monkeypatch):
    """The 3D view of a Render window showing one circular pick,
    rendered once synchronously, with async rendering then enabled."""
    # a display-pixel-size notice would block an offscreen test
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: None),
    )
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (HEIGHT, WIDTH)]
    window.view.resize(128, 128)
    window.view._pick_shape = "Circle"
    # diameter of 12 camera pixels, set in nm in the tools dialog
    window.tools_settings_dialog.pick_diameter.setValue(12.0 * PIXELSIZE)
    window.view._picks = [(32.0, 32.0)]
    window.rot_win()  # loads the pick, shows the window, renders (sync)
    window.window_rot.resize(128, 128)
    view = window.window_rot.view_rot
    # let the layout settle: a resize re-renders (resizeEvent), which
    # must not land in the middle of a test's own requests
    _drain(qt_offscreen, 0.2)
    assert view.image is not None
    view.async_rendering = True  # instance attr shadows the stub
    yield view
    view.stop_render_worker()
    window.view.stop_render_worker()


class TestAsyncRotationRender:
    def test_async_matches_sync(self, rotation_view, qapp):
        view = rotation_view
        view.async_rendering = False
        view.image = None
        view.update_scene()
        sync_qimage = _qimage_bytes(view.qimage)
        sync_raw = view.image.copy()

        view.async_rendering = True
        view.image = None
        view.update_scene()
        request_id = view._render_request_id
        assert request_id > 0
        _wait_until(qapp, lambda: view.image is not None)
        np.testing.assert_array_equal(view.image, sync_raw)
        assert _qimage_bytes(view.qimage) == sync_qimage

    def test_rotated_async_matches_sync(self, rotation_view, qapp):
        view = rotation_view
        view.apply_rotation(np.array([0.3, -0.5, 0.2]))
        view.async_rendering = False
        view.update_scene()
        sync_raw = view.image.copy()
        view.async_rendering = True
        view.image = None
        view.update_scene()
        _wait_until(qapp, lambda: view.image is not None)
        np.testing.assert_array_equal(view.image, sync_raw)

    def test_burst_coalesces(self, rotation_view, qapp, monkeypatch):
        view = rotation_view
        calls = []
        original = render.render_scene
        # hold the worker inside its first render until the burst has
        # been submitted, so coalescing is a property of the queue
        burst_submitted = threading.Event()

        def counting(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                assert burst_submitted.wait(15.0), "burst never submitted"
            return original(*args, **kwargs)

        monkeypatch.setattr(render, "render_scene", counting)
        view.image = None
        n_requests = 6
        for _ in range(n_requests):
            view.apply_rotation(np.array([0.0, 0.05, 0.0]))
            view.update_scene()
        final_id = view._render_request_id
        burst_submitted.set()
        _wait_until(qapp, lambda: view.image is not None)
        _drain(qapp)
        assert len(calls) <= 2 < n_requests
        assert view._render_request_id == final_id

    def test_drag_previews_never_poison_the_cache(
        self, rotation_view, qapp, monkeypatch
    ):
        view = rotation_view
        full_raw = view.image.copy()
        # every drag frame is a subsampled preview (target far below the
        # pick's population) with compensated contrast
        monkeypatch.setattr(
            view, "_interaction_subsample_target", lambda population=0: 100
        )
        adopted = []
        original = view._adopt_render_result

        def recording(contrast_limits, raw_image, cache=True):
            adopted.append(raw_image)
            return original(contrast_limits, raw_image, cache)

        monkeypatch.setattr(view, "_adopt_render_result", recording)
        view.apply_rotation(np.array([0.0, 0.4, 0.0]))
        shown_before = _qimage_bytes(view.qimage)
        view.update_scene(interactive=True)
        assert view._current_request_interactive
        assert view._refine_timer.isActive()
        # hold the refine so the preview's landing is observed on its
        # own (a cold first render can outlast the 150 ms idle timer)
        view._refine_timer.stop()
        # the preview lands and is shown, but the cache is untouched
        _wait_until(qapp, lambda: _qimage_bytes(view.qimage) != shown_before)
        assert adopted == []
        np.testing.assert_array_equal(view.image, full_raw)
        # the refine render follows on idle and is adopted in full
        view._refine_render()
        _wait_until(qapp, lambda: len(adopted) == 1)
        assert not view._current_request_interactive
        assert not np.array_equal(view.image, full_raw)  # rotated now
        # the full render counts every localization: brighter than a
        # preview of 100 of them would be
        assert view.image.sum() > adopted[0].sum() * 0.99

    def test_previews_are_sized_by_what_is_in_view(
        self, rotation_view, qapp, monkeypatch
    ):
        from picasso.gui.render_worker import subsample_request

        view = rotation_view
        main_view = view.window.window.view
        # the main view's rule, with a small target so the pick is thinned
        monkeypatch.setattr(
            main_view,
            "_interaction_subsample_target",
            lambda population=0: 300,
        )
        n_loaded = len(view.locs[0])
        assert n_loaded > 600
        # the whole pick in view: thinned to the target
        request = view._build_render_request()
        assert subsample_request(request, view._interaction_subsample_target)
        assert len(request["locs"]) == pytest.approx(300, rel=0.05)
        # zoomed in on a small part of it: far fewer in view than the
        # target, so the preview renders every localization
        for _ in range(6):
            view.zoom_in()
        assert view._visible_fraction() * n_loaded < 300
        request = view._build_render_request()
        assert not subsample_request(
            request, view._interaction_subsample_target
        )
        assert len(request["locs"]) == n_loaded
        # a view over more than the target thins so that about the
        # target stays in view (an integer stride: between half the
        # target and the target itself)
        for _ in range(3):
            view.zoom_out()
        visible = view._visible_fraction() * n_loaded
        if visible > 300:
            request = view._build_render_request()
            assert subsample_request(
                request, view._interaction_subsample_target
            )
            visible_sampled = len(request["locs"]) * visible / n_loaded
            assert 150 <= visible_sampled <= 315

    def test_global_precision_is_computed_once_per_channel(
        self, rotation_view, monkeypatch
    ):
        import picasso.gui.render_worker as worker_mod

        view = rotation_view
        calls = []
        original = worker_mod.np.median

        def counting(values, *args, **kwargs):
            calls.append(len(values))
            return original(values, *args, **kwargs)

        monkeypatch.setattr(worker_mod.np, "median", counting)
        view._precision_cache = {}  # the fixture's renders warmed it
        locs, _ = view._prepare_locs_for_rendering()
        assert view._global_precisions(locs, "gaussian") is None
        first = view._global_precisions(locs, "convolve")
        expected = (
            float(original(view.locs[0]["lpx"])),
            float(original(view.locs[0]["lpy"])),
        )
        assert first == pytest.approx(expected)
        assert len(calls) == 2  # lpx and lpy, once
        assert view._global_precisions(locs, "convolve") == first
        assert len(calls) == 2  # remembered
        # the main view remembers per channel the same way
        main = view.window.window.view
        main._precision_cache_ = {}
        main_locs, _ = main._prepare_locs_for_rendering(viewport=main.viewport)
        value = main._global_precisions(main_locs, "convolve")
        assert value == pytest.approx(
            (
                float(original(main.locs[0]["lpx"])),
                float(original(main.locs[0]["lpy"])),
            )
        )
        n_calls = len(calls)
        main._global_precisions(main_locs, "convolve")
        assert len(calls) == n_calls

    def test_cache_redraw_bypasses_the_worker(
        self, rotation_view, qapp, monkeypatch
    ):
        view = rotation_view
        calls = []
        original = render.render_scene

        def counting(*args, **kwargs):
            calls.append(kwargs.get("raw_image_cache") is not None)
            return original(*args, **kwargs)

        monkeypatch.setattr(render, "render_scene", counting)
        request_id = view._render_request_id
        view.update_scene(use_cache=True)
        assert calls == [True]  # synchronous, from the cache
        assert view._render_request_id == request_id  # nothing queued

    def test_synchronous_request_lands_before_returning(
        self, rotation_view, qapp
    ):
        view = rotation_view
        view.apply_rotation(np.array([0.0, 0.0, 0.7]))
        before = _qimage_bytes(view.qimage)
        view.update_scene(synchronous=True)
        assert _qimage_bytes(view.qimage) != before

    def test_worker_stops_and_restarts(self, rotation_view, qapp):
        view = rotation_view
        view.update_scene()
        assert view._render_thread is not None
        view.stop_render_worker()
        assert view._render_thread is None
        view.image = None
        view.update_scene()  # restarts the worker lazily
        _wait_until(qapp, lambda: view.image is not None)
        assert view._render_thread is not None

    def test_closing_the_window_stops_the_worker(self, rotation_view, qapp):
        view = rotation_view
        view.update_scene()
        _drain(qapp, 0.2)
        view.window.close()
        assert view._render_thread is None


def test_subsample_request_shared_with_the_main_view():
    """The main view's preview subsampling and the rotation view's are
    the same function: strided rows, contrast scaled by the fraction."""
    from picasso.gui.render_worker import subsample_request

    locs = _locs(1000)
    request = {"locs": locs, "contrast": (0.0, 2.0)}
    assert subsample_request(request, lambda population: 100) is True
    assert len(request["locs"]) == 100
    assert request["contrast"] == (0.0, 0.2)
    untouched = {"locs": locs, "contrast": (0.0, 2.0)}
    assert subsample_request(untouched, lambda population: 0) is False
    assert untouched["locs"] is locs
    assert (
        gui_render.View._subsample_request is not None
        and rotation.ViewRotation._submit_async_render is not None
    )
