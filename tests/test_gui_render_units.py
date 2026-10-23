"""Nanometer units in ``picasso.gui.render``: the files the Render
window reads and writes (pick regions, FOV .txt, screenshot .yaml,
localization files written before Picasso 1.0), the pixel size prompt
for legacy files, the display width replacing the zoom and the
``File > Convert files to nm...`` dialog.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from PyQt6 import QtWidgets

from picasso import io, lib_qt
from picasso.gui import convert_units
from picasso.gui import render as gui_render
from picasso.gui import render_link

WIDTH = HEIGHT = 32  # camera pixels
PIXELSIZE = 130.0  # nm
FOV = WIDTH * PIXELSIZE  # nm


def _locs(n: int = 500, seed: int = 0, nm: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scale = PIXELSIZE if nm else 1.0
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 100, size=n).astype(np.int32),
            "x": rng.uniform(0.0, WIDTH, size=n) * scale,
            "y": rng.uniform(0.0, HEIGHT, size=n) * scale,
            "lpx": np.full(n, 0.1 * scale),
            "lpy": np.full(n, 0.1 * scale),
            "photons": np.full(n, 1000.0),
        }
    )


def _info(pixelsize: float | None = PIXELSIZE) -> list[dict]:
    info = {"Width": WIDTH, "Height": HEIGHT, "Frames": 100}
    if pixelsize is not None:
        info["Pixelsize"] = pixelsize
    return [info]


def _write_legacy_locs(path, pixelsize: float | None = None) -> str:
    """A localization file as Picasso wrote it before 1.0: camera
    pixels, no format marker, the metadata in the .yaml next to it."""
    with h5py.File(path, "w") as f:
        f.create_dataset("locs", data=_locs(nm=False).to_records(index=False))
    with open(os.path.splitext(str(path))[0] + ".yaml", "w") as f:
        yaml.dump_all(_info(pixelsize), f)
    return str(path)


def _wait(qapp, condition, timeout: float = 30.0) -> None:
    start = time.monotonic()
    while not condition():
        qapp.processEvents()
        time.sleep(0.005)
        if time.monotonic() - start > timeout:
            raise TimeoutError("condition not met")
    qapp.processEvents()


@pytest.fixture
def window(qt_offscreen, tmp_path):
    window = gui_render.Window(plugins_loaded=True)
    path = str(tmp_path / "locs.hdf5")
    window.view.add(path, _locs(), _info(), render_=False)
    window.view.viewport = [(0.0, 0.0), (FOV, FOV)]
    window.view.resize(256, 256)
    yield window
    _wait(qt_offscreen, lambda: not lib_qt._running_tasks)
    window.close()


@pytest.fixture
def empty_window(qt_offscreen):
    window = gui_render.Window(plugins_loaded=True)
    yield window
    window.close()


@pytest.fixture
def prompt(monkeypatch):
    """Answer the camera pixel size prompt with ``PIXELSIZE`` and record
    the calls."""
    calls = []

    def get_double(parent, title, label, *args, **kwargs):
        calls.append(label)
        return PIXELSIZE, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", get_double)
    return calls


# ---------------------------------------------------------------------------
# View state
# ---------------------------------------------------------------------------


class TestViewUnits:
    def test_field_of_view_and_pick_size_are_nm(self, window):
        view = window.view
        assert view.movie_size() == (FOV, FOV)
        assert view.full_fov_viewport() == [(0, 0), (FOV, FOV)]
        window.tools_settings_dialog.pick_diameter.setValue(2 * PIXELSIZE)
        assert view._pick_size == 2 * PIXELSIZE
        window.tools_settings_dialog.pick_shape.setCurrentText("Brush")
        window.tools_settings_dialog.brush_width.setValue(50.0)
        assert view._brush_width == 50.0

    def test_display_width_replaces_the_zoom(self, window):
        view = window.view
        dialog = window.display_settings_dlg
        assert not hasattr(dialog, "zoom")
        assert not hasattr(dialog, "screen_px_size")
        # the whole 4160 nm field of view over 256 screen pixels
        assert view.display_pixels_per_viewport_pixels() == pytest.approx(
            view.width() / FOV
        )
        assert view.display_width() == pytest.approx(FOV)
        view.update_scene()
        assert dialog.display_width.decimals() == 4
        assert dialog.display_width.value() == pytest.approx(
            FOV / 1e3, abs=1e-4
        )
        # halving the display width zooms in by two
        dialog.display_width.setValue(FOV / 2e3)
        assert gui_render.render.viewport_width(
            view.viewport
        ) == pytest.approx(FOV / 2)
        assert view.display_width() == pytest.approx(FOV / 2)

    def test_render_kwargs_take_the_blur_in_nm(self, window):
        dialog = window.display_settings_dlg
        dialog.min_blur_width.setValue(12.5)
        kwargs = window.view.get_render_kwargs()
        assert kwargs["min_blur_width"] == 12.5
        assert kwargs["viewport"] == window.view.viewport

    def test_pick_areas_are_um2(self, window):
        view = window.view
        window.tools_settings_dialog.pick_diameter.setValue(1000.0)
        view._pick_shape = "Circle"
        view._picks = [(2000.0, 2000.0)]
        assert view.pick_areas()[0] == pytest.approx(np.pi * 0.5**2)


# ---------------------------------------------------------------------------
# Pick regions
# ---------------------------------------------------------------------------


class TestCameraPixelSize:
    """The camera pixel size comes from each channel's metadata; Render
    has no setting for it."""

    def test_no_camera_pixel_size_setting(self, window):
        assert not hasattr(window.display_settings_dlg, "pixelsize")
        assert not hasattr(window.view, "pixelsize")
        assert "pixelsize" not in render_link.CATEGORY_KEYS

    def test_one_pixel_size_per_channel(self, window, tmp_path):
        assert window.view.pixelsizes == [PIXELSIZE]
        path = str(tmp_path / "other_locs.hdf5")
        window.view.add(path, _locs(seed=1), _info(108.0), render_=False)
        assert window.view.pixelsizes == [PIXELSIZE, 108.0]
        assert window.view.default_pixelsize() == 108.0

    def test_legacy_drift_uses_its_channels_pixel_size(
        self, window, tmp_path, prompt
    ):
        window.view.add(
            str(tmp_path / "other_locs.hdf5"),
            _locs(seed=1),
            _info(108.0),
            render_=False,
        )
        path = str(tmp_path / "old_drift.txt")
        np.savetxt(path, np.tile([0.5, -0.25], (4, 1)))  # camera pixels
        drift = window.view._read_drift_file(path, 1)
        assert prompt == []
        np.testing.assert_allclose(drift["x"], 0.5 * 108.0)
        np.testing.assert_allclose(drift["y"], -0.25 * 108.0)
        # all channels: the first channel's
        drift = window.view._read_drift_file(path, 2)
        np.testing.assert_allclose(drift["x"], 0.5 * PIXELSIZE)

    def test_without_channels_no_pixel_size(self, empty_window):
        assert empty_window.view.pixelsizes == []
        assert empty_window.view.default_pixelsize() == 130.0


class TestPickFiles:
    def test_saved_picks_are_nm_with_the_format_marker(self, window, tmp_path):
        view = window.view
        window.tools_settings_dialog.pick_diameter.setValue(260.0)
        view._pick_shape = "Circle"
        view._picks = [(1300.0, 2600.0), (390.0, 520.0)]
        path = str(tmp_path / "picks.yaml")
        view.save_picks(path)
        with open(path) as f:
            regions = yaml.safe_load(f)
        assert regions[io.FORMAT_VERSION_ATTR] == io.FORMAT_VERSION
        assert regions["Diameter (nm)"] == 260.0
        assert regions["Centers"] == [[1300.0, 2600.0], [390.0, 520.0]]
        # and they load back unchanged, without a prompt
        view._picks = []
        view.load_picks(path)
        assert view._picks == [[1300.0, 2600.0], [390.0, 520.0]]
        assert window.tools_settings_dialog.pick_diameter.value() == 260.0

    def test_legacy_picks_use_the_channels_pixel_size(
        self, window, tmp_path, prompt
    ):
        path = str(tmp_path / "old_picks.yaml")
        with open(path, "w") as f:
            yaml.dump(
                {
                    "Centers": [[10.0, 20.0]],
                    "Diameter (nm)": 260.0,
                    "Shape": "Circle",
                },
                f,
            )
        window.view.load_picks(path)
        assert prompt == []  # the channel's metadata gives it
        assert window.view._picks == [[10.0 * PIXELSIZE, 20.0 * PIXELSIZE]]
        assert window.tools_settings_dialog.pick_diameter.value() == 260.0

    def test_without_channels_the_pixel_size_is_asked_for(
        self, empty_window, tmp_path, monkeypatch
    ):
        path = str(tmp_path / "old_picks.yaml")
        with open(path, "w") as f:
            yaml.dump({"Centers": [[1.0, 1.0]], "Diameter": 2.0}, f)
        asked = []

        def cancel(*args, **kwargs):
            asked.append(args)
            return 0.0, False

        monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", cancel)
        assert empty_window.view._read_picks_file(path) is None
        assert len(asked) == 1

    def test_subtract_legacy_picks(self, window, tmp_path, prompt):
        view = window.view
        window.tools_settings_dialog.pick_diameter.setValue(260.0)
        view._pick_shape = "Circle"
        view._picks = [(1300.0, 1300.0), (3000.0, 3000.0)]
        path = str(tmp_path / "old_picks.yaml")
        with open(path, "w") as f:
            yaml.dump(
                {
                    "Centers": [[10.0, 10.0]],  # 1300 nm
                    "Diameter (nm)": 260.0,
                    "Shape": "Circle",
                },
                f,
            )
        view.subtract_picks(path)
        assert prompt == []
        assert view._picks == [(3000.0, 3000.0)]


# ---------------------------------------------------------------------------
# FOV .txt and screenshot .yaml
# ---------------------------------------------------------------------------


class TestFovFiles:
    def test_fov_txt_round_trip_in_nm(self, window, tmp_path, monkeypatch):
        change_fov = window.info_dialog.change_fov
        change_fov.x_box.setValue(130.0)
        change_fov.y_box.setValue(260.0)
        change_fov.w_box.setValue(1300.0)
        change_fov.h_box.setValue(650.0)
        path = str(tmp_path / "fov.txt")
        monkeypatch.setattr(
            gui_render.lib,
            "get_save_filename_ext_dialog",
            lambda *a, **k: (path, "*.txt"),
        )
        change_fov.save_fov()
        with open(path) as f:
            first = f.readline()
        assert first.strip() == "# " + gui_render.FOV_FILE_HEADER
        assert np.loadtxt(path).tolist() == [130.0, 260.0, 1300.0, 650.0]
        assert gui_render.load_fov_file(path, 999.0).tolist() == [
            130.0,
            260.0,
            1300.0,
            650.0,
        ]

    def test_legacy_fov_txt_is_camera_pixels(self, tmp_path):
        path = str(tmp_path / "old_fov.txt")
        np.savetxt(path, np.array([1.0, 2.0, 10.0, 5.0]))
        fov = gui_render.load_fov_file(path, PIXELSIZE)
        assert fov.tolist() == [130.0, 260.0, 1300.0, 650.0]

    def test_dropped_legacy_fov_txt_uses_the_channels_pixel_size(
        self, window, tmp_path, prompt
    ):
        path = str(tmp_path / "old_fov.txt")
        np.savetxt(path, np.array([1.0, 2.0, 10.0, 10.0]))
        window.view.load_single_txt(path)
        assert prompt == []
        (y0, x0), (y1, x1) = window.view.viewport
        assert (x0, y0) == (130.0, 260.0)
        assert x1 - x0 == pytest.approx(1300.0)

    def test_legacy_fov_txt_needs_a_pixel_size(self, tmp_path):
        path = str(tmp_path / "old_fov.txt")
        np.savetxt(path, np.array([1.0, 2.0, 10.0, 5.0]))
        with pytest.raises(io.MissingPixelsizeError):
            gui_render.load_fov_file(path)

    def test_dropped_fov_txt_sets_the_viewport(self, window, tmp_path):
        path = str(tmp_path / "fov.txt")
        gui_render.save_fov_file(path, [130.0, 260.0, 1300.0, 1300.0])
        window.view.load_single_txt(path)
        (y0, x0), (y1, x1) = window.view.viewport
        assert (x0, y0) == (130.0, 260.0)
        assert (x1 - x0, y1 - y0) == pytest.approx((1300.0, 1300.0))
        assert window.info_dialog.wh_label.text().endswith("nm")


class TestScreenshotInfo:
    def test_exported_info_is_nm(self, window):
        window.view.update_scene()
        viewport = [(260.0, 130.0), (1560.0, 1430.0)]
        info = window.export_current_info(None, viewport=viewport)
        assert info[io.FORMAT_VERSION_ATTR] == io.FORMAT_VERSION
        assert info[gui_render.FOV_KEY] == [130.0, 260.0, 1300.0, 1300.0]
        assert gui_render.LEGACY_ZOOM_KEY not in info
        # the zoom, as nm per screen pixel
        assert info[gui_render.ZOOM_KEY] == pytest.approx(
            1e3
            * window.display_settings_dlg.display_width.value()
            / window.view.width(),
            rel=1e-3,
        )

    def test_legacy_screenshot_is_converted(self, window):
        window.view.load_screenshot(
            {
                gui_render.LEGACY_FOV_KEY: [1.0, 2.0, 10.0, 10.0],
                gui_render.LEGACY_ZOOM_KEY: 8.0,  # screen px per cam. px
                "Min. blur (cam. px)": 0.1,
            }
        )
        (y0, x0), (y1, x1) = window.view.viewport
        assert (x0, y0) == (130.0, 260.0)
        assert x1 - x0 == pytest.approx(1300.0)
        assert window.display_settings_dlg.min_blur_width.value() == (
            pytest.approx(13.0)
        )

    def test_new_screenshot_is_taken_as_is(self, window):
        window.view.load_screenshot(
            {
                io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION,
                gui_render.FOV_KEY: [130.0, 260.0, 1300.0, 1300.0],
                gui_render.ZOOM_KEY: 10.0,
            }
        )
        (y0, x0), (y1, x1) = window.view.viewport
        assert (x0, y0) == (130.0, 260.0)
        assert x1 - x0 == pytest.approx(1300.0)

    @pytest.mark.parametrize(
        "zoom",
        [
            {gui_render.ZOOM_KEY: 10.0},
            # before 1.0: screen pixels per camera pixel (130 nm)
            {gui_render.LEGACY_ZOOM_KEY: 13.0},
        ],
        ids=["zoom", "legacy zoom factor"],
    )
    def test_saved_zoom_sizes_the_view(self, window, monkeypatch, zoom):
        sizes = []
        monkeypatch.setattr(
            window, "resize_view", lambda w, h: sizes.append((w, h))
        )
        window.view.load_screenshot(
            {
                io.FORMAT_VERSION_ATTR: io.FORMAT_VERSION,
                gui_render.FOV_KEY: [130.0, 260.0, 1300.0, 650.0],
                **zoom,
            }
        )
        # 10 nm per screen pixel
        assert sizes == [(pytest.approx(130.0), pytest.approx(65.0))]

    def test_saved_zoom_round_trips(self, window, monkeypatch):
        window.view.update_scene()
        info = window.export_current_info(None)
        sizes = []
        monkeypatch.setattr(
            window, "resize_view", lambda w, h: sizes.append((w, h))
        )
        window.view.load_screenshot(info)
        x, y, w, h = info[gui_render.FOV_KEY]
        assert sizes == [
            (
                pytest.approx(w / info[gui_render.ZOOM_KEY]),
                pytest.approx(h / info[gui_render.ZOOM_KEY]),
            )
        ]
        assert sizes[0][0] == pytest.approx(window.view.width(), rel=1e-2)


# ---------------------------------------------------------------------------
# Legacy localization files
# ---------------------------------------------------------------------------


class TestLegacyLocs:
    def test_metadata_pixelsize_converts_without_a_prompt(
        self, window, tmp_path, prompt, qt_offscreen
    ):
        path = _write_legacy_locs(tmp_path / "old_locs.hdf5", PIXELSIZE)
        assert not gui_render._legacy_locs_without_pixelsize(path)
        window.view.add_multiple([path])
        _wait(qt_offscreen, lambda: window.view._load_thread is None)
        assert prompt == []
        assert len(window.view.locs) == 2
        locs = window.view.locs[1]
        assert locs["x"].max() > WIDTH  # nm, not camera pixels
        assert locs["x"].max() <= FOV

    def test_missing_pixelsize_is_asked_for(
        self, window, tmp_path, prompt, qt_offscreen
    ):
        path = _write_legacy_locs(tmp_path / "old_locs.hdf5", None)
        assert gui_render._legacy_locs_without_pixelsize(path)
        window.view.add_multiple([path])
        _wait(qt_offscreen, lambda: window.view._load_thread is None)
        assert len(prompt) == 1
        assert len(window.view.locs) == 2
        assert window.view.locs[1]["x"].max() > WIDTH
        assert window.view.infos[1][0]["Width"] == WIDTH
        assert (
            gui_render.lib.get_from_metadata(window.view.infos[1], "Pixelsize")
            == PIXELSIZE
        )


# ---------------------------------------------------------------------------
# File > Convert files to nm...
# ---------------------------------------------------------------------------


class TestConvertUnitsDialog:
    def test_menu_opens_the_dialog_once(self, window):
        dialog = window.convert_units()
        assert isinstance(dialog, convert_units.ConvertUnitsDialog)
        assert window.convert_units() is dialog
        assert dialog.pixelsize.value() == window.view.default_pixelsize()
        assert not dialog.replace.isChecked()
        dialog.close()

    def test_pixelsize_is_prefilled_from_the_first_locs_file(
        self, window, tmp_path
    ):
        path = _write_legacy_locs(tmp_path / "old_locs.hdf5", 108.0)
        dialog = window.convert_units()
        dialog.add_paths([str(tmp_path / "picks.yaml"), path])
        assert dialog.pixelsize.value() == 108.0
        assert dialog.paths() == [str(tmp_path / "picks.yaml"), path]
        dialog.files.selectAll()
        dialog.remove_files()
        assert dialog.paths() == []
        dialog.close()

    def test_converts_a_legacy_file_to_an_nm_copy(
        self, window, tmp_path, qt_offscreen
    ):
        path = _write_legacy_locs(tmp_path / "old_locs.hdf5", None)
        dialog = window.convert_units()
        dialog.add_paths([path])
        dialog.pixelsize.setValue(PIXELSIZE)
        dialog.convert()
        _wait(qt_offscreen, lambda: not lib_qt._running_tasks)
        output = str(tmp_path / "old_locs_nm.hdf5")
        assert os.path.isfile(output)
        assert os.path.isfile(str(tmp_path / "old_locs_nm.yaml"))
        assert dialog.results.rowCount() == 1
        assert "old_locs_nm.hdf5" in dialog.results.item(0, 1).text()
        locs, info = io.load_locs(output)
        assert locs["x"].max() > WIDTH
        assert gui_render.lib.get_from_metadata(info, "Pixelsize") == (
            PIXELSIZE
        )
        with h5py.File(output, "r") as f:
            assert f.attrs[io.FORMAT_VERSION_ATTR] == io.FORMAT_VERSION
        # the original is untouched
        with h5py.File(path, "r") as f:
            assert io.FORMAT_VERSION_ATTR not in f.attrs
        dialog.close()

    def test_files_in_nm_are_skipped(self, window, tmp_path, qt_offscreen):
        path = str(tmp_path / "new_locs.hdf5")
        io.save_locs(path, _locs(), _info())
        dialog = window.convert_units()
        dialog.add_paths([path])
        dialog.convert()
        _wait(qt_offscreen, lambda: not lib_qt._running_tasks)
        assert dialog.results.item(0, 1).text().startswith("skipped")
        assert not os.path.exists(str(tmp_path / "new_locs_nm.hdf5"))
        dialog.close()

    def test_json_metadata_fallback(self, tmp_path):
        # metadata_pixelsize reads the metadata like io.load_info
        path = str(tmp_path / "locs.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "locs", data=_locs(nm=False).to_records(index=False)
            )
            f.create_dataset("metadata", data=json.dumps(_info(42.0)))
        assert convert_units.metadata_pixelsize([path]) == 42.0
        assert convert_units.metadata_pixelsize(["nothing.yaml"]) is None


class TestNewerFormatVersion:
    """Files of a newer major format version are refused with a message
    instead of being read as the current format."""

    @pytest.fixture
    def reported(self, monkeypatch):
        messages = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "critical",
            lambda parent, title, text, *a, **k: messages.append(text),
        )
        return messages

    def test_dropped_fov_txt(self, window, tmp_path, reported):
        path = str(tmp_path / "fov.txt")
        gui_render.save_fov_file(path, [130.0, 260.0, 1300.0, 1300.0])
        text = (
            open(path)
            .read()
            .replace(
                f"{io.FORMAT_VERSION_ATTR} {io.FORMAT_VERSION}",
                f"{io.FORMAT_VERSION_ATTR} 2.0",
            )
        )
        open(path, "w").write(text)
        viewport = window.view.viewport
        window.view.load_single_txt(path)
        assert len(reported) == 1
        assert "field of view file format 2.0" in reported[0]
        assert window.view.viewport == viewport

    def test_view_settings(self, window, reported):
        viewport = window.view.viewport
        window.view.load_screenshot(
            {
                io.FORMAT_VERSION_ATTR: "2.0",
                gui_render.FOV_KEY: [130.0, 260.0, 1300.0, 1300.0],
                gui_render.ZOOM_KEY: 10.0,
            },
            "view.yaml",
        )
        assert len(reported) == 1
        assert "view settings file format 2.0" in reported[0]
        assert window.view.viewport == viewport

    def test_picks(self, window, tmp_path, reported):
        path = str(tmp_path / "picks.yaml")
        io.save_picks(path, [(100.0, 200.0)], "Circle", size=150.0)
        picks = io.load_yaml(path)
        picks[io.FORMAT_VERSION_ATTR] = "2.0"
        open(path, "w").write(yaml.safe_dump(picks))
        window.view.load_picks(path)
        assert len(reported) == 1
        assert "picks file format 2.0" in reported[0]
        assert window.view._picks == []
