"""Picasso: Localize's calibration and channel-registration dialogs.

The dialogs that collect a calibration's inputs (3D astigmatism, spline
PSF, sCMOS camera, bead and signal channel registration) and the window
actions that run them: cancel paths, the refusals for data that cannot be
calibrated, and that a finished calibration is saved and loaded for fitting.
Modal dialogs are answered by patching their ``exec`` (or the static method
that opens them) and message boxes are recorded instead of shown.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest
from PyQt6 import QtCore, QtWidgets

from picasso import io, lib, localize, scmos
from picasso.gui import localize as gui_localize

ACCEPTED = QtWidgets.QDialog.DialogCode.Accepted
REJECTED = QtWidgets.QDialog.DialogCode.Rejected
PIXELSIZE = 130.0
BOX = 7


@pytest.fixture
def window(qt_offscreen):
    window = gui_localize.Window()
    yield window
    for name in ("registration_worker", "spline_calibration_worker"):
        worker = getattr(window, name, None)
        if isinstance(worker, QtCore.QThread):
            worker.wait()
    window.close()


class _Messages:
    """Record message boxes instead of opening them."""

    def __init__(self, monkeypatch):
        self.shown = []
        for kind in ("information", "warning", "critical"):
            monkeypatch.setattr(
                QtWidgets.QMessageBox, kind, self._recorder(kind)
            )

    def _recorder(self, kind):
        def record(parent, title, text, *args, **kwargs):
            self.shown.append((kind, title, text))
            return QtWidgets.QMessageBox.StandardButton.Ok

        return record

    @property
    def kinds(self):
        return [kind for kind, _, _ in self.shown]

    def text(self, kind):
        return "\n".join(t for k, _, t in self.shown if k == kind)


def _answer(monkeypatch, cls, code, setup=None):
    """Make ``cls.exec`` fill the dialog in with ``setup`` and return
    ``code`` instead of blocking."""

    def fake_exec(dialog):
        if setup is not None:
            setup(dialog)
        return code

    monkeypatch.setattr(cls, "exec", fake_exec)


def _info(frames, height, width):
    return [
        {
            "Frames": frames,
            "Height": height,
            "Width": width,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _load(window, movies, tmp_path, names=None):
    """Load ``movies`` as the window's channels."""
    names = names or [f"Channel {i}" for i in range(len(movies))]
    paths = [str(tmp_path / f"{name}.tif") for name in names]
    window._set_channels(
        movies,
        [_info(*movie.shape) for movie in movies],
        paths,
        names,
    )
    window.movie_path = paths[0]


class _RunningWorker:
    """Stands in for a worker thread that has not finished yet."""

    def isRunning(self):
        return True


# ---------------------------------------------------------------------------
# The dialogs
# ---------------------------------------------------------------------------


class TestCalibrate3DDialog:
    def test_the_frame_order_needs_several_frames_per_step(self, qt_offscreen):
        dialog = gui_localize.Calibrate3DDialog(None)
        assert not dialog.frame_order.isEnabled()
        dialog.frames_per_step.setValue(3)
        assert dialog.frame_order.isEnabled()
        dialog.frames_per_step.setValue(1)
        assert not dialog.frame_order.isEnabled()

    def test_the_specs_are_read_back(self, qt_offscreen, monkeypatch):
        def fill(dialog):
            dialog.step.setValue(20.0)
            dialog.frames_per_step.setValue(3)
            dialog.frame_order.setCurrentIndex(1)
            dialog.z_binning.setValue(2)

        _answer(monkeypatch, gui_localize.Calibrate3DDialog, ACCEPTED, fill)
        specs = gui_localize.Calibrate3DDialog.getCalibrationSpecs()
        assert specs == (20.0, 3, "z", 2, True)

    def test_cancel_is_reported(self, qt_offscreen, monkeypatch):
        _answer(monkeypatch, gui_localize.Calibrate3DDialog, REJECTED)
        *_, accepted = gui_localize.Calibrate3DDialog.getCalibrationSpecs()
        assert accepted is False


class TestCalibrateSplineDialog:
    def test_multichannel_options_only_for_several_channels(
        self, qt_offscreen
    ):
        single = gui_localize.CalibrateSplineDialog(None)
        assert single.link_photons.isHidden()
        assert single.registration_model.isHidden()
        multi = gui_localize.CalibrateSplineDialog(None, multichannel=True)
        assert not multi.link_photons.isHidden()
        assert not multi.registration_model.isHidden()

    def test_the_specs_are_read_back(self, qt_offscreen, monkeypatch):
        def fill(dialog):
            dialog.step.setValue(10.0)
            dialog.frames_per_step.setValue(2)
            dialog.frame_order.setCurrentIndex(1)
            dialog.z_binning.setValue(3)
            dialog.model.setCurrentIndex(1)
            dialog.magnification_factor.setValue(0.9)
            dialog.correct_z_bias.setChecked(True)
            dialog.link_photons.setChecked(False)
            gui_localize._select_transform_model(
                dialog.registration_model, "translation"
            )

        _answer(
            monkeypatch, gui_localize.CalibrateSplineDialog, ACCEPTED, fill
        )
        specs = gui_localize.CalibrateSplineDialog.getCalibrationSpecs(
            multichannel=True
        )
        assert specs == (
            10.0,
            2,
            "z",
            3,
            "spline-2d",
            0.9,
            True,
            False,
            "translation",
            True,
        )


class TestRegisterChannelsDialog:
    def test_the_specs_are_read_back(self, qt_offscreen, monkeypatch):
        def fill(dialog):
            gui_localize._select_transform_model(
                dialog.registration_model, "projective"
            )
            dialog.multi_fov.setChecked(True)

        _answer(
            monkeypatch, gui_localize.RegisterChannelsDialog, ACCEPTED, fill
        )
        specs = gui_localize.RegisterChannelsDialog.getSpecs(None)
        assert specs == ("projective", True, True)

    def test_defaults_to_affine_on_one_field_of_view(
        self, qt_offscreen, monkeypatch
    ):
        _answer(monkeypatch, gui_localize.RegisterChannelsDialog, REJECTED)
        specs = gui_localize.RegisterChannelsDialog.getSpecs(None)
        assert specs == ("affine", False, False)


class TestRefineRegistrationDialog:
    def test_seeded_from_the_identification_frame_range(self, qt_offscreen):
        dialog = gui_localize.RefineRegistrationDialog(
            None, 100, [[10, 59]], None
        )
        # shown 1-indexed
        assert dialog.first_frame.value() == 11
        assert dialog.last_frame.value() == 60

    def test_the_window_stays_ordered(self, qt_offscreen):
        dialog = gui_localize.RefineRegistrationDialog(None, 100, None)
        assert (dialog.first_frame.value(), dialog.last_frame.value()) == (
            1,
            100,
        )
        dialog.last_frame.setValue(40)
        dialog.first_frame.setValue(70)
        assert dialog.last_frame.value() == 70
        dialog.last_frame.setValue(20)
        assert dialog.first_frame.value() == 20

    def test_the_loaded_model_is_preselected(self, qt_offscreen):
        dialog = gui_localize.RefineRegistrationDialog(
            None, 10, None, "polynomial2"
        )
        assert dialog.registration_model.currentData() == "polynomial2"

    def test_bounds_are_returned_zero_indexed(self, qt_offscreen, monkeypatch):
        def fill(dialog):
            dialog.first_frame.setValue(5)
            dialog.last_frame.setValue(25)
            dialog.max_frames.setValue(12)

        _answer(
            monkeypatch, gui_localize.RefineRegistrationDialog, ACCEPTED, fill
        )
        specs = gui_localize.RefineRegistrationDialog.getFrameSpecs(
            None, 30, model="affine"
        )
        assert specs == ([[4, 24]], 12, "affine", True)


class TestCameraCalibrationSpecs:
    def test_a_bad_level_column_shows_the_dialog_again(
        self, qt_offscreen, monkeypatch
    ):
        """The user is told what is wrong and gets the dialog back with
        their entries, instead of a calibration without levels."""
        messages = _Messages(monkeypatch)
        calls = []

        def fill(dialog):
            calls.append(True)
            if len(calls) == 1:
                dialog.dark_edit.setText("/dark.tif")
                dialog.out_edit.setText("/calib.hdf5")
                dialog.add_bright("/a.tif", "1")
                dialog.add_bright("/b.tif")
            else:
                # the entries survived; fill in the missing level
                assert dialog.bright_paths() == ["/a.tif", "/b.tif"]
                dialog.bright_list.item(1, 1).setText("2")

        _answer(
            monkeypatch, gui_localize.CameraCalibrationDialog, ACCEPTED, fill
        )
        dark, bright, levels, _, out, accepted = (
            gui_localize.CameraCalibrationDialog.getCalibrationSpecs()
        )
        assert len(calls) == 2
        assert messages.kinds == ["warning"]
        assert (dark, bright, levels, out, accepted) == (
            "/dark.tif",
            ["/a.tif", "/b.tif"],
            [1.0, 2.0],
            "/calib.hdf5",
            True,
        )

    def test_cancel_ignores_a_bad_level_column(
        self, qt_offscreen, monkeypatch
    ):
        messages = _Messages(monkeypatch)

        def fill(dialog):
            dialog.add_bright("/a.tif", "1")
            dialog.add_bright("/b.tif")

        _answer(
            monkeypatch, gui_localize.CameraCalibrationDialog, REJECTED, fill
        )
        _, _, levels, _, _, accepted = (
            gui_localize.CameraCalibrationDialog.getCalibrationSpecs()
        )
        assert (levels, accepted) == (None, False)
        assert messages.shown == []


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------


def _dark_movie(n_frames=120, shape=(8, 8), seed=0):
    rng = np.random.default_rng(seed)
    return (100 + rng.normal(0, 2, (n_frames, *shape))).astype(np.uint16)


def _bright_movie(level, n_frames=60, shape=(8, 8), seed=1):
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(level, (n_frames, *shape))
    return (100 + 2 * electrons).astype(np.uint16)


def _camera_specs(monkeypatch, specs):
    monkeypatch.setattr(
        gui_localize.CameraCalibrationDialog,
        "getCalibrationSpecs",
        staticmethod(lambda parent=None: specs),
    )


def _movies_by_path(monkeypatch, movies):
    def load_movie(path, *args, **kwargs):
        movie = movies[os.path.basename(path)]
        if isinstance(movie, Exception):
            raise movie
        return movie, _info(*movie.shape)

    monkeypatch.setattr(io, "load_movie", load_movie)


class TestCalibrateCamera:
    def test_cancel_loads_nothing(self, window, monkeypatch):
        _camera_specs(monkeypatch, ("/d.tif", [], None, "", "/c.hdf5", False))
        _movies_by_path(monkeypatch, {})  # any load would raise KeyError
        window.calibrate_camera()
        assert not window.parameters_dialog.camera_calibration

    def test_builds_saves_and_loads_the_calibration(
        self, window, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        out = str(tmp_path / "camera.hdf5")
        _camera_specs(
            monkeypatch,
            (
                "/dark.tif",
                ["/low.tif", "/high.tif"],
                [1.0, 2.0],
                "mW",
                out,
                True,
            ),
        )
        _movies_by_path(
            monkeypatch,
            {
                "dark.tif": _dark_movie(),
                "low.tif": _bright_movie(50, seed=2),
                "high.tif": _bright_movie(200, seed=3),
            },
        )
        window.calibrate_camera()

        assert messages.kinds == ["information"], messages.shown
        assert io.load_camera_calibration(out)["Gain levels"] == 2
        assert os.path.exists(scmos.plot_path(out))
        assert window.parameters_dialog.camera_calibration is not None
        summary = messages.text("information")
        assert f"Saved to {out}." in summary
        assert "Gain: median" in summary

    def test_an_unreadable_movie_is_reported(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        _camera_specs(monkeypatch, ("/dark.tif", [], None, "", "/c", True))
        _movies_by_path(monkeypatch, {"dark.tif": OSError("unreadable")})
        window.calibrate_camera()
        assert messages.shown == [
            ("critical", "Camera calibration", "unreadable")
        ]

    def test_a_too_short_dark_movie_is_reported(
        self, window, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        out = str(tmp_path / "camera.hdf5")
        _camera_specs(monkeypatch, ("/dark.tif", [], None, "", out, True))
        _movies_by_path(monkeypatch, {"dark.tif": _dark_movie(n_frames=10)})
        window.calibrate_camera()
        assert messages.kinds == ["critical"]
        assert "at least" in messages.text("critical")
        assert not os.path.exists(out)


class TestCheckCameraCalibration:
    @pytest.fixture
    def calibrated(self, window, tmp_path, monkeypatch):
        path = str(tmp_path / "camera.hdf5")
        calibration = scmos.calibrate_scmos(_dark_movie())
        calibration["Path"] = path
        io.save_camera_calibration(path, calibration)
        window.parameters_dialog.update_camera_calib(path)
        return window

    def _open(self, monkeypatch, path):
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *a, **k: (path, ""),
        )

    def test_asks_for_a_calibration_first(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.check_camera_calibration()
        assert messages.kinds == ["information"]
        assert "Load an sCMOS camera calibration" in messages.text(
            "information"
        )

    def test_reports_the_mean_p_value(self, calibrated, monkeypatch):
        messages = _Messages(monkeypatch)
        self._open(monkeypatch, "/fresh.tif")
        _movies_by_path(monkeypatch, {"fresh.tif": _dark_movie(seed=9)})
        calibrated.check_camera_calibration()
        assert messages.kinds == ["information"]
        assert "Mean p-value" in messages.text("information")

    def test_a_mismatched_movie_is_reported(self, calibrated, monkeypatch):
        messages = _Messages(monkeypatch)
        self._open(monkeypatch, "/fresh.tif")
        _movies_by_path(monkeypatch, {"fresh.tif": _dark_movie(shape=(6, 6))})
        calibrated.check_camera_calibration()
        assert messages.kinds == ["critical"]
        assert "same camera ROI" in messages.text("critical")

    def test_cancel_does_nothing(self, calibrated, monkeypatch):
        messages = _Messages(monkeypatch)
        self._open(monkeypatch, "")
        calibrated.check_camera_calibration()
        assert messages.shown == []


# ---------------------------------------------------------------------------
# Channel registration
# ---------------------------------------------------------------------------

SHIFT = (3.0, -2.0)


def _bead_movies(shift=SHIFT, shape=(64, 64)):
    """Two single-frame bead images, the second shifted by ``shift``."""
    grid = np.array(
        [[x, y] for x in range(12, 56, 10) for y in range(12, 56, 10)], float
    )
    yy, xx = np.indices(shape)
    movies = []
    for c, offset in enumerate(((0.0, 0.0), shift)):
        image = np.zeros(shape)
        for x, y in grid + offset:
            image += 4000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3.4)
        rng = np.random.RandomState(c)
        movies.append(rng.poisson(image + 100).astype(np.uint16)[None])
    return movies


def _bead_specs(monkeypatch, specs=("affine", False, True)):
    monkeypatch.setattr(
        gui_localize.RegisterChannelsDialog,
        "getSpecs",
        staticmethod(lambda parent: specs),
    )


def _save_to(monkeypatch, path):
    monkeypatch.setattr(
        gui_localize.Window, "_registration_save_path", lambda self: path
    )


def _beads_window(window, tmp_path, movies=None):
    _load(window, movies or _bead_movies(), tmp_path)
    window.parameters_dialog.box_spinbox.setValue(BOX)
    window.parameters_dialog.mng_slider.setValue(2000)
    return window


class TestRegisterChannelsFromBeads:
    def test_refuses_a_single_channel(self, window, tmp_path, monkeypatch):
        messages = _Messages(monkeypatch)
        _load(window, _bead_movies()[:1], tmp_path)
        window.register_channels_from_beads()
        assert messages.kinds == ["information"]
        assert "at least two channels" in messages.text("information")
        assert getattr(window, "registration_worker", None) is None

    def test_cancel_starts_nothing(self, window, tmp_path, monkeypatch):
        _beads_window(window, tmp_path)
        _bead_specs(monkeypatch, ("affine", False, False))
        _save_to(monkeypatch, str(tmp_path / "reg.yaml"))
        window.register_channels_from_beads()
        assert getattr(window, "registration_worker", None) is None

    def test_no_save_path_starts_nothing(self, window, tmp_path, monkeypatch):
        _beads_window(window, tmp_path)
        _bead_specs(monkeypatch)
        _save_to(monkeypatch, "")
        window.register_channels_from_beads()
        assert getattr(window, "registration_worker", None) is None

    def test_registers_and_loads_the_result(
        self, window, tmp_path, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        _beads_window(window, tmp_path)
        _bead_specs(monkeypatch)
        path = str(tmp_path / "reg.yaml")
        _save_to(monkeypatch, path)

        window.register_channels_from_beads()
        window.registration_worker.wait()
        QtWidgets.QApplication.processEvents()

        assert messages.kinds == ["information"], messages.shown
        dialog = window.parameters_dialog
        assert dialog.channel_registration_path == path
        calibration = dialog.channel_registration_calibration
        assert calibration["source"] == "beads"
        assert calibration["n_channels"] == 2
        # one line per non-reference channel, named as in the channel list
        assert "Channel 1: 25 correspondences" in messages.text("information")
        assert "camera px RMS" in messages.text("information")

    def test_a_failure_is_reported(self, window, tmp_path, monkeypatch):
        messages = _Messages(monkeypatch)
        blank = [np.full((1, 64, 64), 100, np.uint16) for _ in range(2)]
        _beads_window(window, tmp_path, blank)
        _bead_specs(monkeypatch)
        path = str(tmp_path / "reg.yaml")
        _save_to(monkeypatch, path)

        window.register_channels_from_beads()
        window.registration_worker.wait()
        QtWidgets.QApplication.processEvents()

        assert messages.kinds == ["warning"]
        assert window.parameters_dialog.channel_registration_path != path

    def test_a_running_registration_is_not_replaced(
        self, window, tmp_path, monkeypatch
    ):
        """Replacing the reference to a running QThread destroys it while
        it runs, which aborts the whole application."""
        messages = _Messages(monkeypatch)
        _beads_window(window, tmp_path)
        _bead_specs(monkeypatch)
        monkeypatch.setattr(
            gui_localize.RefineRegistrationDialog,
            "getFrameSpecs",
            staticmethod(lambda *a, **k: ([[0, 0]], 50, "affine", True)),
        )
        _save_to(monkeypatch, str(tmp_path / "reg.yaml"))
        running = _RunningWorker()
        window.registration_worker = running

        window.register_channels_from_beads()
        window.register_channels_from_signal()

        assert window.registration_worker is running
        assert messages.kinds == ["information", "information"]
        assert "already running" in messages.text("information")


class TestRegisterChannelsFromSignal:
    def test_cancel_starts_nothing(self, window, tmp_path, monkeypatch):
        _beads_window(window, tmp_path)
        monkeypatch.setattr(
            gui_localize.RefineRegistrationDialog,
            "getFrameSpecs",
            staticmethod(lambda *a, **k: ([[0, 0]], 50, "affine", False)),
        )
        _save_to(monkeypatch, str(tmp_path / "reg.yaml"))
        window.register_channels_from_signal()
        assert getattr(window, "registration_worker", None) is None

    def test_seeds_from_the_loaded_registration(
        self, window, tmp_path, monkeypatch
    ):
        """A loaded registration both preselects its model and seeds the
        pairing, which is what makes this a re-alignment."""
        _beads_window(window, tmp_path)
        loaded = {
            "registration_model": "translation",
            "channel_transforms": ["identity", "shifted"],
        }
        window.parameters_dialog.channel_registration_calibration = loaded
        asked = {}

        def specs(parent, n_frames, frame_range, model=None):
            asked.update(n_frames=n_frames, model=model)
            return [[0, 0]], 50, model, True

        monkeypatch.setattr(
            gui_localize.RefineRegistrationDialog,
            "getFrameSpecs",
            staticmethod(specs),
        )
        started = {}
        monkeypatch.setattr(
            gui_localize.Window,
            "_start_channel_registration",
            lambda self, source, **kwargs: started.update(
                kwargs, source=source
            ),
        )
        window.register_channels_from_signal()
        assert asked == {"n_frames": 1, "model": "translation"}
        assert started["source"] == "signal"
        assert started["model"] == "translation"
        assert started["seed_transforms"] == ["identity", "shifted"]

    def test_a_seed_for_other_channels_is_dropped(
        self, window, tmp_path, monkeypatch
    ):
        _beads_window(window, tmp_path)
        window.parameters_dialog.channel_registration_calibration = {
            "channel_transforms": ["a", "b", "c"]
        }
        monkeypatch.setattr(
            gui_localize.RefineRegistrationDialog,
            "getFrameSpecs",
            staticmethod(lambda *a, **k: ([[0, 0]], 50, "affine", True)),
        )
        started = {}
        monkeypatch.setattr(
            gui_localize.Window,
            "_start_channel_registration",
            lambda self, source, **kwargs: started.update(kwargs),
        )
        window.register_channels_from_signal()
        assert started["seed_transforms"] is None


# ---------------------------------------------------------------------------
# Spline PSF calibration
# ---------------------------------------------------------------------------


class _RecordingWorker:
    """Captures what the spline calibration worker would be started with."""

    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        type(self).created.append(self)
        for name in ("statusChanged", "finished", "failed"):
            setattr(self, name, types.SimpleNamespace(connect=lambda f: None))

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started


@pytest.fixture
def recording_worker(monkeypatch):
    _RecordingWorker.created = []
    monkeypatch.setattr(
        gui_localize, "SplineCalibrationWorker", _RecordingWorker
    )
    return _RecordingWorker.created


def _spline_specs(monkeypatch, accepted=True, asked=None):
    def specs(parent=None, multichannel=False):
        if asked is not None:
            asked.append(multichannel)
        return (
            10.0,
            1,
            "fov",
            1,
            "spline-3d",
            0.79,
            False,
            True,
            "affine",
            accepted,
        )

    monkeypatch.setattr(
        gui_localize.CalibrateSplineDialog,
        "getCalibrationSpecs",
        staticmethod(specs),
    )


def _spline_save_to(monkeypatch, path):
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *a, **k: (path, "*.hdf5"),
    )


def _stack(n_frames=20, shape=(32, 32)):
    return np.full((n_frames, *shape), 100, np.uint16)


class TestBuildSplineCalibration:
    def test_needs_a_movie(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.calibrate_spline()
        assert messages.shown == [
            ("information", "Spline PSF Calibration", "No file loaded.")
        ]

    def test_split_fov_needs_two_regions(
        self, window, tmp_path, monkeypatch, recording_worker
    ):
        messages = _Messages(monkeypatch)
        _load(window, [_stack()], tmp_path)
        window.view.split_fov_mode = True
        window.view.rois = [[[0, 0], [16, 16]]]
        window.build_spline_calibration()
        assert messages.kinds == ["information"]
        assert recording_worker == []

    def test_channels_must_share_the_z_scan(
        self, window, tmp_path, monkeypatch, recording_worker
    ):
        messages = _Messages(monkeypatch)
        _load(window, [_stack(20), _stack(30)], tmp_path)
        _spline_specs(monkeypatch)
        window.build_spline_calibration()
        assert "different frame counts" in messages.text("warning")
        assert recording_worker == []

    @pytest.mark.parametrize("accepted, path", [(False, "x"), (True, "")])
    def test_cancel_starts_nothing(
        self, window, tmp_path, monkeypatch, recording_worker, accepted, path
    ):
        _load(window, [_stack()], tmp_path)
        _spline_specs(monkeypatch, accepted=accepted)
        _spline_save_to(monkeypatch, path)
        window.build_spline_calibration()
        assert recording_worker == []

    def test_a_single_channel_build(
        self, window, tmp_path, monkeypatch, recording_worker
    ):
        _load(window, [_stack()], tmp_path)
        asked = []
        _spline_specs(monkeypatch, asked=asked)
        path = str(tmp_path / "spline.hdf5")
        _spline_save_to(monkeypatch, path)
        window.build_spline_calibration()
        (worker,) = recording_worker
        assert asked == [False]  # no multichannel options offered
        assert worker.started
        assert worker.kwargs["path"] == path
        assert worker.kwargs["movies"] is None
        assert worker.kwargs["regions"] is None
        assert worker.kwargs["step"] == 10.0

    def test_a_multichannel_build_passes_every_channel(
        self, window, tmp_path, monkeypatch, recording_worker
    ):
        _load(window, [_stack(), _stack()], tmp_path)
        asked = []
        _spline_specs(monkeypatch, asked=asked)
        _spline_save_to(monkeypatch, str(tmp_path / "spline.hdf5"))
        window.build_spline_calibration()
        (worker,) = recording_worker
        assert asked == [True]
        assert len(worker.kwargs["movies"]) == 2
        assert len(worker.kwargs["camera_infos"]) == 2
        assert "2 channels" in window.status_bar.currentMessage()

    def test_a_running_build_is_not_replaced(
        self, window, tmp_path, monkeypatch, recording_worker
    ):
        messages = _Messages(monkeypatch)
        _load(window, [_stack()], tmp_path)
        _spline_specs(monkeypatch)
        _spline_save_to(monkeypatch, str(tmp_path / "spline.hdf5"))
        running = _RunningWorker()
        window.spline_calibration_worker = running
        window.build_spline_calibration()
        assert window.spline_calibration_worker is running
        assert recording_worker == []
        assert "already running" in messages.text("information")


class _MessageBoxRecorder:
    """Answers ``QMessageBox.exec`` on an instance, optionally clicking the
    button with the given text."""

    def __init__(self, monkeypatch, click=None):
        self.texts = []
        self.buttons = []
        recorder = self

        def fake_exec(box):
            recorder.texts.append(box.text())
            recorder.buttons.append([b.text() for b in box.buttons()])
            for button in box.buttons():
                if button.text() == click:
                    monkeypatch.setattr(
                        box, "clickedButton", lambda b=button: b
                    )
            return 0

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)


class TestSplineCalibrationFinished:
    def _finish(self, window, tmp_path, diagnostics, n_beads, n_used):
        path = str(tmp_path / "beads_spline_calib.hdf5")
        for suffix in ("_beads.png", "_psf.png"):
            open(path[: -len(".hdf5")] + suffix, "wb").close()
        window.spline_calibration_worker = types.SimpleNamespace(
            bead_diagnostics=diagnostics
        )
        window.on_spline_calibration_finished(path, n_beads, n_used)
        return path

    def test_lists_the_diagnostics_and_the_rejected_beads(
        self, window, tmp_path, monkeypatch
    ):
        boxes = _MessageBoxRecorder(monkeypatch)
        self._finish(window, tmp_path, [{"bead": 1}], 5, 3)
        (text,) = boxes.texts
        assert "3 of 5 detected beads (2 rejected as outliers)" in text
        assert "beads_spline_calib_beads.png" in text
        assert "beads_spline_calib_psf.png" in text
        assert "Inspect beads..." in boxes.buttons[0]
        assert window.inspect_beads_action.isEnabled()

    def test_inspect_opens_the_bead_gallery(
        self, window, tmp_path, monkeypatch
    ):
        _MessageBoxRecorder(monkeypatch, click="Inspect beads...")
        opened = []

        class Gallery:
            def __init__(self, diagnostics, parent, title=""):
                opened.append((diagnostics, title))

            def show(self):
                pass

        monkeypatch.setattr(gui_localize, "BeadInspectionDialog", Gallery)
        self._finish(window, tmp_path, [{"bead": 1}], 4, 4)
        assert opened == [([{"bead": 1}], "beads_spline_calib.hdf5")]

    def test_no_gallery_without_diagnostics(
        self, window, tmp_path, monkeypatch
    ):
        boxes = _MessageBoxRecorder(monkeypatch)
        self._finish(window, tmp_path, [], 4, 4)
        assert "built from 4 beads" in boxes.texts[0]
        assert "Inspect beads..." not in boxes.buttons[0]
        assert not window.inspect_beads_action.isEnabled()

    def test_inspecting_before_a_build_explains_why_not(
        self, window, monkeypatch
    ):
        messages = _Messages(monkeypatch)
        window.bead_diagnostics = []
        window.inspect_calibration_beads()
        assert "No spline PSF calibration" in messages.text("information")

    def test_a_failure_is_reported(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.on_spline_calibration_failed("no beads")
        assert messages.shown == [
            ("critical", "Spline PSF Calibration", "no beads")
        ]


# ---------------------------------------------------------------------------
# Lateral (affine) calibration
# ---------------------------------------------------------------------------


class TestCalibrateAffine:
    def test_the_dialog_is_reused(self, window):
        window.calibrate_affine()
        first = window._affine_dialog
        window.calibrate_affine()
        assert window._affine_dialog is first
        assert first.isVisible()

    def test_all_paths_are_required(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.calibrate_affine()
        window._run_calibrate_affine()
        assert messages.kinds == ["warning"]
        assert "All three paths" in messages.text("warning")
        assert getattr(window, "affine_calibration_worker", None) is None

    def test_a_running_calibration_is_not_replaced(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.calibrate_affine()
        dialog = window._affine_dialog
        for name in ("reference_path", "target_path", "calibration_path"):
            monkeypatch.setattr(
                type(dialog), name, property(lambda self: "/x.tif")
            )
        running = _RunningWorker()
        window.affine_calibration_worker = running
        window._run_calibrate_affine()
        assert window.affine_calibration_worker is running
        assert "already running" in messages.text("information")

    def test_a_failure_is_reported(self, window, monkeypatch):
        messages = _Messages(monkeypatch)
        window.affine_calibration_worker = _RunningWorker()
        window.on_affine_calibration_failed("no beads paired")
        assert window.affine_calibration_worker is None
        assert messages.shown == [
            ("warning", "Calibrate lateral transform", "no beads paired")
        ]

    def test_cancel_clears_the_worker(self, window):
        window.affine_calibration_worker = _RunningWorker()
        window.on_affine_calibration_canceled()
        assert window.affine_calibration_worker is None
        assert "canceled" in window.status_bar.currentMessage()

    def test_a_failed_figure_keeps_the_fit(
        self, window, tmp_path, monkeypatch
    ):
        """The transform is already saved; the user is told where, and
        that only the figure is missing."""
        messages = _Messages(monkeypatch)
        path = str(tmp_path / "calib.yaml")
        window.on_affine_calibration_finished(
            path, 12, {"transform_type": "chromatic"}
        )
        assert window.affine_calibration_worker is None
        assert "12 bead pairs" in window.status_bar.currentMessage()
        assert messages.kinds == ["warning"]
        assert f"saved to {path}" in messages.text("warning")


# ---------------------------------------------------------------------------
# 3D (astigmatism) calibration, after the fit
# ---------------------------------------------------------------------------

N_STEPS = 50
STEP = 10.0  # nm


def _bead_stack_locs():
    """Bead localizations whose widths follow known polynomials of the
    stage z, as in ``tests/test_zfit.py``."""
    rng = np.random.default_rng(0)
    z_total = (N_STEPS - 1) * STEP
    rows = []
    for frame in range(N_STEPS):
        z = -(frame * STEP - z_total / 2)
        sx = 1.5 + 1e-3 * z + 1e-5 * z**2
        sy = 1.5 - 1e-3 * z + 1e-5 * z**2
        for _ in range(20):
            rows.append(
                {
                    "frame": frame,
                    "x": 16.0 * PIXELSIZE,
                    "y": 16.0 * PIXELSIZE,
                    "sx": (sx + rng.normal(0, 0.02)) * PIXELSIZE,
                    "sy": (sy + rng.normal(0, 0.02)) * PIXELSIZE,
                    "photons": 5000.0,
                    "bg": 10.0,
                    "lpx": 0.01 * PIXELSIZE,
                    "lpy": 0.01 * PIXELSIZE,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fitted(window, tmp_path, monkeypatch):
    """A window with a bead z-stack loaded, about to calibrate."""
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
    _load(window, [_stack(N_STEPS)], tmp_path, names=["beads"])
    return window


def _calibrate_3d(monkeypatch, specs, path):
    monkeypatch.setattr(
        gui_localize.Calibrate3DDialog,
        "getCalibrationSpecs",
        staticmethod(lambda parent=None: specs),
    )
    monkeypatch.setattr(
        gui_localize.lib,
        "get_save_filename_ext_dialog",
        lambda *a, **k: (path, "*.yaml"),
    )


def _fit_info(window):
    """The metadata the fit hands ``on_fit_finished``: the movie's, then
    the identification's and the fit's, with the camera information."""
    identify = lib.step_info(
        "Identify", **localize.identification_info(window.parameters)
    )
    fit = lib.step_info("Fit 2D", **{"Fit method": "gausslq"})
    return lib.append_metadata(
        window.info, [identify, fit | window.camera_info]
    )


class TestCalibrateZAfterTheFit:
    def test_writes_the_calibration(self, fitted, tmp_path, monkeypatch):
        path = str(tmp_path / "beads_3d_calib.yaml")
        _calibrate_3d(monkeypatch, (STEP, 1, "fov", 1, True), path)
        fitted.on_fit_finished(
            _bead_stack_locs(), _fit_info(fitted), 1.0, False, True
        )
        calibration = io.load_calibration(path)
        assert len(calibration["X Coefficients"]) == 7
        assert "3D calibrated" in fitted.status_bar.currentMessage()

    def test_cancel_writes_nothing(self, fitted, tmp_path, monkeypatch):
        path = str(tmp_path / "beads_3d_calib.yaml")
        _calibrate_3d(monkeypatch, (STEP, 1, "fov", 1, False), path)
        fitted.on_fit_finished(
            _bead_stack_locs(), _fit_info(fitted), 1.0, False, True
        )
        assert not os.path.exists(path)

    @pytest.mark.parametrize(
        "frames_per_step, z_binning, reason",
        [(N_STEPS + 1, 1, "frames per step"), (1, 10, "axial bins")],
    )
    def test_specs_that_do_not_fit_the_stack_are_reported(
        self,
        fitted,
        tmp_path,
        monkeypatch,
        frames_per_step,
        z_binning,
        reason,
    ):
        """Raised by the calibration for the user's input, so shown to the
        user rather than escaping the slot."""
        messages = _Messages(monkeypatch)
        path = str(tmp_path / "beads_3d_calib.yaml")
        _calibrate_3d(
            monkeypatch, (STEP, frames_per_step, "fov", z_binning, True), path
        )
        fitted.on_fit_finished(
            _bead_stack_locs(), _fit_info(fitted), 1.0, False, True
        )
        assert messages.kinds == ["warning"]
        assert reason in messages.text("warning").lower()
        assert not os.path.exists(path)

    def test_needs_the_elliptical_gaussian(self, fitted, monkeypatch):
        messages = _Messages(monkeypatch)
        combo = fitted.parameters_dialog.fit_model
        other = next(
            combo.itemText(i)
            for i in range(combo.count())
            if combo.itemText(i) != gui_localize.ASTIGMATISM_FIT_MODEL
        )
        combo.setCurrentText(other)
        fitted.calibrate_z()
        assert messages.kinds == ["warning"]
        assert gui_localize.ASTIGMATISM_FIT_MODEL in messages.text("warning")
