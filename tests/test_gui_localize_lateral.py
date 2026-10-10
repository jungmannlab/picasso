"""Localize's wiring of the lateral (x, y) corrections.

A correction may be appended to the 3D calibration - the recommended
route, since it then travels with the calibration - or kept in its own
file and loaded separately through the ``Parameters`` dialog. Both must
land on the same coordinates, they must compose (a 3D calibration
carrying the astigmatism correction plus a separately loaded chromatic
one), and the same correction coming in twice must be applied once.

``picasso.zfit`` does the applying; what is tested here is the window's
part of it: which corrections reach the fit, that a 3D run applies them
after the z fit rather than before, and that the file says which ones
were applied - the workers discard the info their fit returns, so
without that the GUI would leave no record where the CLI leaves one.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from picasso import lib, zfit
from picasso.gui import localize as gui_localize

from tests.conftest import CALIB_3D, affine

ASTIG = [[1.0, 0.0, 3.0], [0.0, 1.0, -1.5], [0.0, 0.0, 1.0]]
CHROMATIC = [[1.0, 0.0, 5.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
INFO = [{"Pixelsize": 130, "Frames": 4, "Height": 64, "Width": 64}]


def _entry(kind, matrix, **extra):
    return {"Type": kind, "Transform": affine(matrix).to_dict(), **extra}


def _calibration(*entries):
    calibration = dict(CALIB_3D)
    for entry in entries:
        lib.append_lateral_transform(calibration, entry)
    return calibration


def _locs(n=4):
    """Localizations with the columns the astigmatic z fit needs."""
    return pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int32),
            "x": np.full(n, 10.0, dtype=np.float32),
            "y": np.full(n, 20.0, dtype=np.float32),
            "photons": np.full(n, 1000.0, dtype=np.float32),
            "sx": np.full(n, 1.2, dtype=np.float32),
            "sy": np.full(n, 1.0, dtype=np.float32),
            "bg": np.full(n, 10.0, dtype=np.float32),
            "lpx": np.full(n, 0.01, dtype=np.float32),
            "lpy": np.full(n, 0.01, dtype=np.float32),
            "net_gradient": np.full(n, 100.0, dtype=np.float32),
        }
    )


class _Label:
    """The one QLabel the affine box owns."""

    def __init__(self):
        self.text = ""

    def setAlignment(self, *args):
        pass

    def setText(self, text):
        self.text = text

    def setToolTip(self, tip):
        pass


def _dialog(z_calibration=None, spline_calibration=None):
    """The parts of ``ParametersDialog`` the lateral corrections touch."""
    dialog = types.SimpleNamespace(
        z_calibration=z_calibration,
        spline_calibration=spline_calibration,
        lateral_transforms=[],
        affine_calibration_paths=[],
        affine_calib_label=_Label(),
    )
    for name in (
        "update_affine_calib",
        "_calibration_lateral_transforms",
        "_set_affine_state",
    ):
        setattr(
            dialog,
            name,
            getattr(gui_localize.ParametersDialog, name).__get__(dialog),
        )
    return dialog


def _load(dialog, files, monkeypatch, warnings_seen=None):
    """Run the real ``update_affine_calib`` on in-memory calibrations."""
    monkeypatch.setattr(
        gui_localize.io, "load_any_calibration", lambda path: files[path]
    )
    monkeypatch.setattr(
        gui_localize.QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, text: (
            warnings_seen.append(text) if warnings_seen is not None else None
        ),
    )
    dialog.update_affine_calib(list(files))
    return dialog


def _window(dialog, locs, status=None):
    """The parts of ``Window`` the z fit and the save touch."""
    window = types.SimpleNamespace(
        parameters_dialog=dialog,
        locs=locs,
        info=list(INFO),
        camera_info={"Pixelsize": 130},
        status_bar=types.SimpleNamespace(
            showMessage=(
                status.append if status is not None else lambda _: None
            )
        ),
        abort_action=types.SimpleNamespace(setEnabled=lambda _: None),
        _active_worker=None,
        _applied_lateral=[],
        on_fit_z_progress=lambda *a: None,
        on_fit_z_finished=lambda *a: None,
        on_worker_aborted=lambda *a: None,
    )
    for name in ("_extra_lateral_transforms", "_record_applied_lateral"):
        setattr(
            window,
            name,
            getattr(gui_localize.Window, name).__get__(window),
        )
    return window


def _run_fit_z(window, monkeypatch):
    """Drive ``Window.fit_z``, then the real worker body on what it built,
    returning the finished localizations and the worker."""
    real = gui_localize.FitZWorker
    built = {}

    class _Recorder:
        """Stands in for the QThread: records the arguments and runs
        nothing, so the worker body can be driven synchronously below."""

        FIELDS = (
            "locs",
            "info",
            "calibration",
            "magnification_factor",
            "pixelsize",
            "fitting_method",
            "gpu",
        )

        def __init__(self, *args, **kwargs):
            worker = types.SimpleNamespace(
                **dict(zip(self.FIELDS, args)),
                lateral_transforms=kwargs.get("lateral_transforms") or [],
                applied_lateral_transforms=[],
                skipped_lateral_transforms=[],
                isInterruptionRequested=lambda: False,
                on_progress=lambda n: None,
            )
            built["worker"] = worker
            for signal in ("progressMade", "finished", "aborted"):
                setattr(
                    self,
                    signal,
                    types.SimpleNamespace(connect=lambda *a: None),
                )

        def start(self):
            pass

    dialog = window.parameters_dialog
    dialog.fit_model = types.SimpleNamespace(
        currentText=lambda: "2D elliptical Gaussian"
    )
    dialog.fit_optimizer = types.SimpleNamespace(
        currentText=lambda: "Least squares"
    )
    dialog.magnification_factor = types.SimpleNamespace(value=lambda: 0.79)
    dialog.pixelsize = types.SimpleNamespace(value=lambda: 130)
    dialog.gpu_checkbox = types.SimpleNamespace(isChecked=lambda: False)
    monkeypatch.setattr(gui_localize, "FitZWorker", _Recorder)

    gui_localize.Window.fit_z(window)

    worker = built["worker"]
    out = {}
    worker.finished = types.SimpleNamespace(
        emit=lambda locs, dt: out.update(locs=locs)
    )
    real.run(worker)
    window._record_applied_lateral(worker)
    return out["locs"], worker


def _appended_reference(locs, *entries):
    """What the recommended route produces: everything in one calibration."""
    out, _ = zfit.zfit(
        locs,
        list(INFO),
        calibration=_calibration(*entries),
        magnification_factor=0.79,
        filter=0,
    )
    return out


class TestLoadingACorrectionSeparately:
    """What the ``Parameters`` dialog accepts alongside a 3D calibration."""

    def test_a_different_correction_is_accepted(self, monkeypatch):
        """The mixed case: the 3D calibration carries the astigmatism
        correction, a chromatic one is loaded on top of it."""
        dialog = _dialog(
            z_calibration=_calibration(_entry("astigmatism", ASTIG))
        )
        _load(
            dialog,
            {
                "chromatic.yaml": lib.append_lateral_transform(
                    {}, _entry("chromatic", CHROMATIC)
                )
            },
            monkeypatch,
        )
        assert lib.describe_lateral_transforms(dialog.lateral_transforms) == [
            "chromatic, affine"
        ]
        assert dialog.affine_calibration_paths == ["chromatic.yaml"]

    def test_a_copy_of_the_calibrations_own_is_refused(self, monkeypatch):
        """Loading the same correction the 3D calibration carries would
        correct the coordinates twice, so it is rejected with a warning
        rather than silently dropped."""
        dialog = _dialog(
            z_calibration=_calibration(_entry("astigmatism", ASTIG))
        )
        seen = []
        _load(
            dialog,
            {
                "astig_copy.yaml": lib.append_lateral_transform(
                    # the same transform, saved under another name
                    {},
                    _entry("astigmatism", ASTIG, **{"Bead pairs": 99}),
                )
            },
            monkeypatch,
            warnings_seen=seen,
        )
        assert dialog.lateral_transforms == []
        assert dialog.affine_calibration_paths == []
        assert seen and "correct the coordinates twice" in seen[0]

    def test_the_same_correction_picked_twice_loads_once(self, monkeypatch):
        """Two files carrying the same transform (the file and a copy of
        it) are applied once, with a warning, as on the command line."""
        dialog = _dialog()
        seen = []
        _load(
            dialog,
            {
                "chromatic.yaml": lib.append_lateral_transform(
                    {}, _entry("chromatic", CHROMATIC)
                ),
                "chromatic_copy.yaml": lib.append_lateral_transform(
                    {}, _entry("chromatic", CHROMATIC, **{"Bead pairs": 99})
                ),
            },
            monkeypatch,
            warnings_seen=seen,
        )
        assert lib.describe_lateral_transforms(dialog.lateral_transforms) == [
            "chromatic, affine"
        ]
        assert dialog.affine_calibration_paths == ["chromatic.yaml"]
        assert seen and "selected more than once" in seen[0]

    def test_without_a_3d_calibration_anything_loads(self, monkeypatch):
        dialog = _dialog()
        _load(
            dialog,
            {
                "astig.yaml": lib.append_lateral_transform(
                    {}, _entry("astigmatism", ASTIG)
                ),
                "chromatic.yaml": lib.append_lateral_transform(
                    {}, _entry("chromatic", CHROMATIC)
                ),
            },
            monkeypatch,
        )
        # applied in the order they were chosen
        assert lib.describe_lateral_transforms(dialog.lateral_transforms) == [
            "astigmatism, affine",
            "chromatic, affine",
        ]


class TestZFitAppliesThem:
    """``Window.fit_z`` -> ``FitZWorker`` -> ``zfit``: the corrections must
    land on the coordinates the z fit produced, exactly as they would had
    they been appended to the 3D calibration."""

    def test_separate_matches_appended(self, monkeypatch):
        locs = _locs()
        reference = _appended_reference(
            locs, _entry("astigmatism", ASTIG), _entry("chromatic", CHROMATIC)
        )
        dialog = _dialog(
            z_calibration=_calibration(_entry("astigmatism", ASTIG))
        )
        _load(
            dialog,
            {
                "chromatic.yaml": lib.append_lateral_transform(
                    {}, _entry("chromatic", CHROMATIC)
                )
            },
            monkeypatch,
        )
        window = _window(dialog, locs)

        out, worker = _run_fit_z(window, monkeypatch)

        np.testing.assert_allclose(
            out["x"].to_numpy(), reference["x"].to_numpy(), atol=1e-6
        )
        np.testing.assert_allclose(
            out["y"].to_numpy(), reference["y"].to_numpy(), atol=1e-6
        )
        # z is untouched by a lateral correction
        np.testing.assert_allclose(
            out["z"].to_numpy(), reference["z"].to_numpy(), atol=1e-6
        )
        assert worker.applied_lateral_transforms == [
            "astigmatism, affine",
            "chromatic, affine",
        ]

    def test_a_plain_3d_calibration_plus_a_loaded_correction(
        self, monkeypatch
    ):
        locs = _locs()
        dialog = _dialog(z_calibration=dict(CALIB_3D))
        _load(
            dialog,
            {
                "chromatic.yaml": lib.append_lateral_transform(
                    {}, _entry("chromatic", CHROMATIC)
                )
            },
            monkeypatch,
        )
        window = _window(dialog, locs)

        out, _ = _run_fit_z(window, monkeypatch)

        plain, _ = zfit.zfit(
            locs,
            list(INFO),
            calibration=dict(CALIB_3D),
            magnification_factor=0.79,
            filter=0,
        )
        np.testing.assert_allclose(
            out["x"].to_numpy(), plain["x"].to_numpy() + 5.0, atol=1e-6
        )
        np.testing.assert_allclose(
            out["y"].to_numpy(), plain["y"].to_numpy() + 2.0, atol=1e-6
        )

    def test_a_duplicate_slipping_past_the_dialog_is_skipped(
        self, monkeypatch
    ):
        """The dialog rejects a copy at load time, but the 3D calibration
        may be swapped afterwards - so the fit checks again, applies the
        correction once and says so in the status bar."""
        locs = _locs()
        calibration = _calibration(_entry("astigmatism", ASTIG))
        dialog = _dialog(z_calibration=calibration)
        # loaded while no 3D calibration was set, then one is chosen
        dialog.lateral_transforms = [_entry("astigmatism", ASTIG)]
        dialog.affine_calibration_paths = ["astig_copy.yaml"]
        status = []
        window = _window(dialog, locs, status=status)

        out, worker = _run_fit_z(window, monkeypatch)

        once, _ = zfit.zfit(
            locs,
            list(INFO),
            calibration=calibration,
            magnification_factor=0.79,
            filter=0,
        )
        np.testing.assert_allclose(
            out["x"].to_numpy(), once["x"].to_numpy(), atol=1e-6
        )
        assert worker.applied_lateral_transforms == ["astigmatism, affine"]
        assert any("already applied by the calibration" in m for m in status)


class TestTheFitLeavesARecord:
    """The workers throw away the info their fit returns, so the window has
    to carry what was applied into the saved metadata itself."""

    @staticmethod
    def _fit_worker(fit_z, monkeypatch, locs):
        """``FitWorker`` with the fit itself stubbed out, run to
        completion."""
        monkeypatch.setattr(
            gui_localize.localize,
            "fit",
            lambda **kwargs: (locs, list(INFO)),
        )
        out = {}
        worker = types.SimpleNamespace(
            movie=None,
            movie_info=list(INFO),
            camera_info={},
            identifications=locs,
            box=7,
            eps=None,
            max_it=None,
            method="gausslq",
            fit_z=fit_z,
            calibrate_z=False,
            spline_calibration=None,
            lateral_transforms=[_entry("chromatic", CHROMATIC)],
            applied_lateral_transforms=[],
            camera_calibration=None,
            N=len(locs),
            isInterruptionRequested=lambda: False,
            on_progress=lambda n: None,
            on_cut_progress=lambda n: None,
            progressMade=types.SimpleNamespace(emit=lambda *a: None),
            aborted=types.SimpleNamespace(emit=lambda *a: None),
            finished=types.SimpleNamespace(
                emit=lambda locs, *a: out.update(locs=locs)
            ),
        )
        gui_localize.FitWorker.run(worker)
        return out["locs"], worker

    def test_the_2d_fit_holds_them_back_for_the_z_fit(self, monkeypatch):
        """With a z fit to follow, ``FitWorker`` must not apply them: they
        belong on the coordinates the z fit produced, which is where
        ``FitZWorker`` puts them. Applying them here as well would correct
        twice."""
        locs = _locs()

        out, worker = self._fit_worker(True, monkeypatch, locs)

        np.testing.assert_array_equal(
            out["x"].to_numpy(), locs["x"].to_numpy()
        )
        assert worker.applied_lateral_transforms == []

    def test_a_2d_fit_applies_them_itself(self, monkeypatch):
        """Without a z fit there is nothing to wait for, so the same worker
        applies them and records what it applied."""
        locs = _locs()

        out, worker = self._fit_worker(False, monkeypatch, locs)

        np.testing.assert_allclose(
            out["x"].to_numpy(), locs["x"].to_numpy() + 5.0
        )
        np.testing.assert_allclose(
            out["y"].to_numpy(), locs["y"].to_numpy() + 2.0
        )
        assert worker.applied_lateral_transforms == ["chromatic, affine"]

    def test_save_locs_names_what_was_applied(self, monkeypatch):
        """``Lateral corrections applied`` in the file, as the CLI writes -
        a separately loaded correction has no other trace in the metadata."""
        saved = {}
        monkeypatch.setattr(
            gui_localize.io,
            "save_locs",
            lambda path, locs, info: saved.update(info=info),
        )
        monkeypatch.setattr(
            gui_localize.io, "strip_mm_metadata", lambda info: list(info)
        )
        window = types.SimpleNamespace(
            locs=_locs(),
            info=list(INFO),
            camera_info={"Pixelsize": 130},
            last_identification_info={},
            _applied_lateral=["astigmatism, affine", "chromatic, affine"],
            _region_suffix=None,
            channels=[],
            parameters_dialog=types.SimpleNamespace(
                fit_model=types.SimpleNamespace(
                    currentText=lambda: "2D elliptical Gaussian"
                ),
                fit_optimizer=types.SimpleNamespace(
                    currentText=lambda: "Least squares"
                ),
                fit_z_checkbox=types.SimpleNamespace(isChecked=lambda: True),
                z_calibration=dict(CALIB_3D),
                z_calibration_path="astig_3d_calib.yaml",
                camera_calibration={},
                affine_calibration_paths=["chromatic.yaml"],
            ),
            fit_mode=lambda: gui_localize.FIT_MODE_JOINT,
            select_locs_columns=lambda: None,
        )

        gui_localize.Window.save_locs(window, "out.hdf5")

        info = saved["info"][-1]
        assert info["Lateral corrections applied"] == [
            "astigmatism, affine",
            "chromatic, affine",
        ]
        assert info["Lateral correction paths"] == ["chromatic.yaml"]

    def test_nothing_applied_says_nothing(self, monkeypatch):
        saved = {}
        monkeypatch.setattr(
            gui_localize.io,
            "save_locs",
            lambda path, locs, info: saved.update(info=info),
        )
        monkeypatch.setattr(
            gui_localize.io, "strip_mm_metadata", lambda info: list(info)
        )
        window = types.SimpleNamespace(
            locs=_locs(),
            info=list(INFO),
            camera_info={},
            last_identification_info={},
            _applied_lateral=[],
            _region_suffix=None,
            channels=[],
            parameters_dialog=types.SimpleNamespace(
                fit_model=types.SimpleNamespace(
                    currentText=lambda: "2D elliptical Gaussian"
                ),
                fit_optimizer=types.SimpleNamespace(
                    currentText=lambda: "Least squares"
                ),
                fit_z_checkbox=types.SimpleNamespace(isChecked=lambda: False),
                camera_calibration={},
                affine_calibration_paths=[],
            ),
            fit_mode=lambda: gui_localize.FIT_MODE_JOINT,
            select_locs_columns=lambda: None,
        )

        gui_localize.Window.save_locs(window, "out.hdf5")

        assert "Lateral corrections applied" not in saved["info"][-1]
